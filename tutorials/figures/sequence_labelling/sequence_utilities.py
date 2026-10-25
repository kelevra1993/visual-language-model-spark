"""
Helpers shared by the figure scripts of the sequence labelling (M2M) and frame selection (SEL) tutorials
(tutorials/figures/sequence_labelling and tutorials/figures/frame_selection), which make_figures.py runs with the Python
of the repository at every build: the test sequences of a real run read by the SequenceDataset of the repository and
batched by its collate function, its checkpoint rebuilt on the CPU, the parameter count of every component of the model,
the evaluation block of an iteration read from Results.txt for the listings of the report scenes, and the frames of a
sequence drawn as captioned thumbnails on the card colour of the videos, so that every figure shows the files and the
code of the repository as they are.
"""

import json
import re

import numpy
import torch

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Any, Dict, List, Optional, Tuple

from tutorials.tutorial_builder.layout import (CARD_COLOUR, DIM_TEXT_COLOUR, MONOSPACE_FONT_PATH, SANS_FONT_PATH,
                                               SANS_SEMIBOLD_FONT_PATH)
from tutorials.tutorial_builder.settings import THREAD_COUNT
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.data.sequence.sequence_dataset import SequenceDataset, collate_padded_sequences
from vision_training.models.model_loading import load_trained_model
from vision_training.models.task_model import TaskModel
from vision_training.trainers.checkpointing import get_checkpoint_path

# Pattern of the first line of an evaluation block of Results.txt, "Iteration 1000 · 200 Test Sequences, ..."
RESULTS_BLOCK_HEADER_PATTERN = re.compile(pattern=r"^Iteration (\d+) · ")

# Metric of a headline line of an evaluation block: its name, then its value after at least two spaces
METRIC_PATTERN = re.compile(pattern=r"([A-Za-z][\w@\- ]*?[\w@])\s{2,}(\d+\.\d+)")


def read_run_sections(run_folder: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Reads the saved training configuration of a sequence run with the loader of the trainers, which also builds its
    label dictionary from the mapping file, so that the figures read the data and the model exactly as the run did.

    Args:
        run_folder (Path): The run folder, holding training_configuration.yaml.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: The experiment configuration and the model sections of the run.
    """
    return load_experiment_configuration(configuration_path=run_folder / "training_configuration.yaml")


def build_test_dataset(run_folder: Path, label_type: str) -> SequenceDataset:
    """
    Builds the test split of a sequence run with the SequenceDataset of the repository, in the order of its labels file
    (no shuffle), so that the figures read the frames and frame labels the evaluation of the run read.

    Args:
        run_folder (Path): The run folder.
        label_type (str): "frame" for the sequence labelling, "selection" for the frame selection.

    Returns:
        SequenceDataset: The test sequences of the run.
    """
    experiment_configuration, model_configuration = read_run_sections(run_folder=run_folder)
    test_folder = Path(experiment_configuration["test_data_folder"])
    image_settings = model_configuration["Data"]["image_settings"]
    return SequenceDataset(data_directory=str(test_folder / "data"), labels_file=str(test_folder / "labels.json"),
                           image_size=image_settings["size"], keep_ratio=image_settings["keep_ratio"],
                           label_dictionary=model_configuration["Data"]["label_dictionary"],
                           sequence_settings=model_configuration["Data"]["sequence_settings"], label_type=label_type,
                           split_name="Test", selection_settings=model_configuration.get("Selection"), shuffle=False)


def collate_named_sequences(dataset: SequenceDataset, sequence_names: List[str]) -> Dict[str, Any]:
    """
    Reads the named sequences of a split and batches them with the collate function of the loaders, in the given
    order, the batch a figure runs through the model or shows padded.

    Args:
        dataset (SequenceDataset): The split.
        sequence_names (List[str]): The names of the sequences, e.g. ["test_000004", "test_000134"].

    Returns:
        Dict[str, Any]: The padded batch: images [B, T, 3, S, S] uint8, sequence_mask [B, T], labels [B, T].

    Raises:
        KeyError: If a sequence is not in the split.
    """
    sequence_indices = {sequence["sequence_name"]: sequence_index
                        for sequence_index, sequence in enumerate(dataset.sequences)}
    missing_names = [sequence_name for sequence_name in sequence_names if sequence_name not in sequence_indices]
    if missing_names:
        raise KeyError(f"The sequences {missing_names} are not in the test split of the run.")
    return collate_padded_sequences(samples=[dataset[sequence_indices[sequence_name]]
                                             for sequence_name in sequence_names], members_per_sample=1)


def load_sequence_model(run_folder: Path, iteration: int) -> TaskModel:
    """
    Rebuilds the trained model of a sequence run at one checkpoint, on the CPU and in evaluation mode, with the loader
    of the evaluation and deployment scripts (the trained weights, not their moving average), the threads of PyTorch
    limited to those of the build so that the trainings keep the processors.

    Args:
        run_folder (Path): The run folder.
        iteration (int): The iteration of the checkpoint.

    Returns:
        TaskModel: The model.
    """
    torch.set_num_threads(THREAD_COUNT)
    checkpoint_path = get_checkpoint_path(experiment_folder=run_folder, iteration=iteration)
    return load_trained_model(configuration_path=run_folder / "training_configuration.yaml",
                              checkpoint_path=checkpoint_path, device=torch.device("cpu"), use_averaged_weights=False)


def predict_sequences(model: TaskModel, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Runs a padded batch of sequences through a sequence model as the evaluation does (float frames in [0, 255], the
    sequence mask given), without gradients.

    Args:
        model (TaskModel): The sequence model in evaluation mode.
        batch (Dict[str, Any]): The padded batch of collate_named_sequences.

    Returns:
        Dict[str, torch.Tensor]: The outputs of the model, with class_logits and class_probabilities [B, T, K].
    """
    with torch.no_grad():
        return model(input_tensor=batch["images"].float(), sequence_mask=batch["sequence_mask"])


def count_model_parameters(model: TaskModel) -> Dict[str, int]:
    """
    Counts the parameters of every component of a sequence model as its model size summary does (get_size_components:
    backbone, temporal module, classification head, whole model).

    Args:
        model (TaskModel): The sequence model.

    Returns:
        Dict[str, int]: The parameters of every component, keyed backbone, temporal_module, classification_head and
            total.
    """
    component_keys = {"Backbone": "backbone", "Temporal Module": "temporal_module",
                      "Classification Head": "classification_head", "Total Model": "total"}
    return {component_keys[component_name]: sum(parameter.numel() for parameter in component.parameters())
            for component_name, component in model.get_size_components().items()}


def read_results_block(results_path: Path, iteration: int) -> List[str]:
    """
    Reads the evaluation block of one iteration from the Results.txt of a run, from its header line to the line
    before the next block, its trailing blank lines removed, the text the report scenes list.

    Args:
        results_path (Path): The Results.txt file of the run.
        iteration (int): The evaluated iteration.

    Returns:
        List[str]: The lines of the block.

    Raises:
        ValueError: If the file holds no block of that iteration.
    """
    block_lines, inside_block = [], False
    for line in results_path.read_text(encoding="utf-8").splitlines():
        header_match = RESULTS_BLOCK_HEADER_PATTERN.match(string=line)
        if header_match is not None:
            inside_block = int(header_match.group(1)) == iteration
        if inside_block:
            block_lines.append(line.rstrip())
    while block_lines and not block_lines[-1].strip():
        block_lines.pop()
    if not block_lines:
        raise ValueError(f"No evaluation block of iteration {iteration} in {results_path}.")
    return block_lines


def parse_block_metrics(block_lines: List[str], fact_prefix: str) -> Dict[str, float]:
    """
    Reads the headline metrics of an evaluation block of Results.txt, every "<name>   <value>" pair of its lines
    before the per-class table, as facts named after the metric ("Segment F1@10" giving <prefix>.segment_f1_10), so
    that the narration quotes the report of the run.

    Args:
        block_lines (List[str]): The lines of the block (read_results_block).
        fact_prefix (str): The prefix of the fact names, e.g. "evaluation".

    Returns:
        Dict[str, float]: The value of every metric.
    """
    metrics = {}
    for line in block_lines[1:]:
        if line.strip().startswith("Per Class"):
            break
        for metric_name, metric_value in METRIC_PATTERN.findall(string=line):
            fact_name = re.sub(pattern=r"[^a-z0-9]+", repl="_", string=metric_name.lower()).strip("_")
            metrics[f"{fact_prefix}.{fact_name}"] = float(metric_value)
    return metrics


def write_listing(lines: List[str], listing_path: Path) -> None:
    """
    Writes the lines of a listing shown by a listing element of a diagram scene, as a text file of the figures folder.

    Args:
        lines (List[str]): The lines.
        listing_path (Path): The text file, its folder created if missing.
    """
    listing_path.parent.mkdir(parents=True, exist_ok=True)
    listing_path.write_text(data="\n".join(lines) + "\n", encoding="utf-8")


def format_sequence_entry(sequence: Dict[str, Any], maximum_listed_frames: int) -> List[str]:
    """
    Writes one sequence of a labels.json file as the lines of a listing, its frames shortened to the first and last
    ones when there are more than a few, so that the listing shows the real entry within the width of a scene.

    Args:
        sequence (Dict[str, Any]): The sequence of the labels file.
        maximum_listed_frames (int): The number of frames listed in full, the others replaced by "...".

    Returns:
        List[str]: The lines of the entry.
    """
    frames = [f'"{frame}"' for frame in sequence["frames"]]
    if len(frames) > maximum_listed_frames:
        frames = frames[:2] + ["..."] + frames[-1:]

    # The frames one per line under their key, then the category of every frame on one line
    frame_lines = [f'  "frames": [{frames[0]},'] + [f"             {frame}," for frame in frames[1:-1]] + \
        [f"             {frames[-1]}],"]
    return (["{", f'  "id": {sequence["id"]},', f'  "name": "{sequence["name"]}",'] + frame_lines +
            [f'  "frame_category_ids": {json.dumps(sequence["frame_category_ids"])}', "}"])


def load_thumbnail_fonts(caption_size: int) -> Dict[str, ImageFont.FreeTypeFont]:
    """
    Loads the Inter and Hack fonts of the thumbnail figures, at the size of their captions.

    Args:
        caption_size (int): The size of the captions in pixels.

    Returns:
        Dict[str, ImageFont.FreeTypeFont]: The 'caption', 'caption_bold' and 'monospace' fonts.
    """
    return {"caption": ImageFont.truetype(font=str(SANS_FONT_PATH), size=caption_size),
            "caption_bold": ImageFont.truetype(font=str(SANS_SEMIBOLD_FONT_PATH), size=caption_size),
            "monospace": ImageFont.truetype(font=str(MONOSPACE_FONT_PATH), size=caption_size - 2)}


def convert_frame_to_image(frame: torch.Tensor, size: int) -> Image.Image:
    """
    Turns one frame of a batch (the [3, S, S] uint8 BGR network input read by the dataset) into an RGB image of a
    thumbnail size.

    Args:
        frame (torch.Tensor): The frame.
        size (int): The side of the thumbnail in pixels.

    Returns:
        Image.Image: The RGB thumbnail.
    """
    rgb_array = numpy.ascontiguousarray(frame.permute(1, 2, 0).numpy()[:, :, ::-1])
    return Image.fromarray(obj=rgb_array).resize(size=(size, size), resample=Image.Resampling.LANCZOS)


def draw_captioned_frames(frames: List[Image.Image], captions: List[List[Tuple[str, Tuple[int, int, int]]]],
                          columns: int, frame_size: int, gap: int, caption_height: int, canvas_size: Tuple[int, int],
                          top: int, outlines: List[Optional[Tuple[int, int, int]]],
                          fonts: Dict[str, ImageFont.FreeTypeFont]) -> Tuple[Image.Image, List[List[int]]]:
    """
    Draws the frames of a sequence as a grid of thumbnails on the card colour, every thumbnail with its caption lines
    under it (each line with its colour) and an optional coloured outline, centred horizontally on a canvas of the
    size it is shown at, and gives the pixel box of every thumbnail and its caption, the views of the image scenes.

    Args:
        frames (List[Image.Image]): The RGB thumbnails, all frame_size pixels wide.
        captions (List[List[Tuple[str, Tuple[int, int, int]]]]): The caption lines of every frame and their colours.
        columns (int): The number of thumbnails per row.
        frame_size (int): The side of a thumbnail in pixels.
        gap (int): The pixels between two thumbnails.
        caption_height (int): The height of the caption band under every thumbnail.
        canvas_size (Tuple[int, int]): The width and height of the figure.
        top (int): The top of the first row in pixels.
        outlines (List[Optional[Tuple[int, int, int]]]): The outline colour of every thumbnail, None for none.
        fonts (Dict[str, ImageFont.FreeTypeFont]): The fonts of load_thumbnail_fonts.

    Returns:
        Tuple[Image.Image, List[List[int]]]: The figure, and the left, top, right and bottom of every thumbnail with
            its caption.
    """
    canvas = Image.new(mode="RGB", size=canvas_size, color=CARD_COLOUR)
    draw = ImageDraw.Draw(im=canvas)
    used_columns = min(columns, len(frames))
    grid_width = used_columns * frame_size + (used_columns - 1) * gap
    left_margin = (canvas_size[0] - grid_width) // 2
    frame_boxes = []

    # Every thumbnail at its place in the grid, outlined when asked, its caption lines centred under it
    for frame_index, (frame, caption_lines, outline_colour) in enumerate(zip(frames, captions, outlines)):
        row_index, column_index = divmod(frame_index, columns)
        frame_left = left_margin + column_index * (frame_size + gap)
        frame_top = top + row_index * (frame_size + caption_height + gap)
        canvas.paste(im=frame, box=(frame_left, frame_top))
        if outline_colour is not None:
            draw.rectangle(xy=(frame_left - 5, frame_top - 5, frame_left + frame_size + 4, frame_top + frame_size + 4),
                           outline=outline_colour, width=5)
        line_top = frame_top + frame_size + 12
        for line_index, (caption_text, caption_colour) in enumerate(caption_lines):
            font = fonts["caption_bold"] if line_index == 0 else fonts["caption"]
            draw.text(xy=(frame_left + frame_size / 2, line_top), text=caption_text, fill=caption_colour, font=font,
                      anchor="ma")
            line_top += font.size + 8
        frame_boxes.append([frame_left - 10, frame_top - 10, frame_left + frame_size + 10,
                            frame_top + frame_size + caption_height])
    return canvas, frame_boxes


def draw_figure_header(canvas: Image.Image, text: str, fonts: Dict[str, ImageFont.FreeTypeFont]) -> None:
    """
    Writes the header line of a thumbnail figure (the run, the split and the sequence it shows) in the secondary grey
    of the videos, at its top left corner.

    Args:
        canvas (Image.Image): The figure.
        text (str): The header.
        fonts (Dict[str, ImageFont.FreeTypeFont]): The fonts of load_thumbnail_fonts.
    """
    ImageDraw.Draw(im=canvas).text(xy=(12, 14), text=text, fill=DIM_TEXT_COLOUR, font=fonts["monospace"])

