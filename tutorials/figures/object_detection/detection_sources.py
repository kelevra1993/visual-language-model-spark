"""
Sources shared by the figure families of the object detection and instance segmentation series of the tutorial videos
(tutorials/figures/object_detection/figures.py and tutorials/figures/instance_segmentation/figures.py): the trained
models of the DET and INS runs of tutorials/tutorial_sources.yaml rebuilt on the CPU from their run folders, the test
image both series follow (test_014285.png, six shapes of the synthetic shapes test split, one of the eight sample
images of both runs) with its objects read from the test labels.json, the TensorBoard scalars, the blocks of
Results.txt (their table values read whether the report wrote the rates as percentages, "78.64 %", or with the
decimals of the reports written before, "0.7864"), the folder of the evaluations made again with the current code
(EVALUATION_FOLDER_NAME), the console logs of the runs stripped of their terminal codes, and
the drawing of boxes and listings in the palette of the videos. Both runs train on the same instance segmentation data
set, so that the two series show the same image and objects.
"""

import functools
import json
import re

import numpy
import torch

from pathlib import Path
from PIL import ImageDraw, ImageFont
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Any, Dict, List, Tuple

from tutorial_builder.layout import MONOSPACE_FONT_PATH, SANS_SEMIBOLD_FONT_PATH
from tutorial_builder.figure_context import read_thread_count
from vision_training.data.preprocessing import read_image
from vision_training.evaluation.checkpoint_evaluation import CHECKPOINT_EVALUATION_FOLDER_NAME
from vision_training.evaluation.report_formatting import PERCENT_SIGN
from vision_training.models.model_loading import load_trained_model
from vision_training.models.task_model import TaskModel
from vision_training.trainers.checkpointing import get_checkpoint_path

# Test image followed by both series: a triangle, a star, a parallelogram, a small and a large nonagon and a square,
# drawn in the sample images of every evaluation of both runs
SAMPLE_IMAGE_NAME = "test_014285.png"

# Checkpoint whose trained weights the instance segmentation figures run, the last one of the INS run (resumed from
# 2000 to 5000 iterations), and the evaluated checkpoints whose sample images are compared; the object detection figures
# read those of the DET run from its saved configuration (find_run_iterations of object_detection/figures.py)
CHECKPOINT_ITERATION = 5000
SAMPLE_ITERATIONS = [1000, 2000, 5000]

# Folder of the last checkpoint of both runs evaluated again with the current code by
# scripts/evaluation/evaluate_checkpoint.py, whose Results.txt tables, class metrics and thresholds, confusion matrix
# and proposal recall the evaluation chapters read, so that their listings write the rates as the reports of the
# repository do; the sample images and the curves of every evaluation stay those of the training
EVALUATION_FOLDER_NAME = CHECKPOINT_EVALUATION_FOLDER_NAME

# Value of a row of a Results.txt table: a number, followed by the percent sign of the reports when it is a rate written
# as a percentage ("78.64 %", the reports written since the percentages; "0.7864" before them), or the dash of a missing
# value; and a row of a table, its label then its values after at least two spaces
REPORT_VALUE_PATTERN = r"[\d.,]+(?:" + re.escape(PERCENT_SIGN) + r")?|—"
REPORT_ROW_PATTERN = re.compile(pattern=rf"^\s*(.+?)\s{{2,}}((?:{REPORT_VALUE_PATTERN})"
                                        rf"(?:\s+(?:{REPORT_VALUE_PATTERN}))*)\s*$")

# Terminal codes of the console logs: colours and styles, and the hyperlinks of the file paths, whose text is kept
TERMINAL_COLOUR_PATTERN = re.compile(pattern=r"\x1b\[[0-9;]*m")
TERMINAL_LINK_PATTERN = re.compile(pattern=r"\x1b\]8;;[^\x1b]*\x1b\\")

# Width of the outlines of the boxes drawn on the images, and the font size of their captions
BOX_OUTLINE_WIDTH = 3
BOX_CAPTION_FONT_SIZE = 18


def find_test_folder(run_configuration: Dict[str, Any]) -> Path:
    """
    Gives the test split of a run, read from its saved configuration, the split of the sample images and of the test
    image of the figures.

    Args:
        run_configuration (Dict[str, Any]): The saved configuration of the run (read_run_configuration).

    Returns:
        Path: The test folder, holding data/ and labels.json.

    Raises:
        FileNotFoundError: If the test folder or its labels.json is missing.
    """
    test_folder = Path(run_configuration["ExperimentConfiguration"]["test_data_folder"])
    if not (test_folder / "labels.json").exists():
        raise FileNotFoundError(f"No labels.json in the test split {test_folder} of the run: generate the instance "
                                f"segmentation data set with "
                                f"vision_training/data/synthetic/create_synthetic_dataset.py.")
    return test_folder


@functools.lru_cache(maxsize=2)
def read_test_manifest(test_folder: Path) -> Dict[str, Any]:
    """
    Reads the COCO labels.json of a test split once per process (80 MB for the synthetic shapes), shared by the
    figure families of both series that read the objects of the test image or the first annotations of the split.

    Args:
        test_folder (Path): The test split.

    Returns:
        Dict[str, Any]: The manifest, with its images, annotations and categories.
    """
    return json.loads((test_folder / "labels.json").read_text(encoding="utf-8"))


def read_image_annotations(test_folder: Path, image_name: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Gives the COCO entry of a test image and its annotations, in the order of the manifest, the objects the figures
    draw and the detector is compared with.

    Args:
        test_folder (Path): The test split.
        image_name (str): The file name of the image, e.g. "test_014285.png".

    Returns:
        Tuple[Dict[str, Any], List[Dict[str, Any]]]: The image entry and its annotations.

    Raises:
        KeyError: If the image is not in the manifest.
    """
    manifest = read_test_manifest(test_folder=test_folder)
    image_entries = [entry for entry in manifest["images"] if entry["file_name"] == image_name]
    if not image_entries:
        raise KeyError(f"No image {image_name} in {test_folder / 'labels.json'}.")
    image_entry = image_entries[0]
    annotations = [annotation for annotation in manifest["annotations"] if annotation["image_id"] == image_entry["id"]]
    return image_entry, annotations


def read_category_names(test_folder: Path) -> Dict[int, str]:
    """
    Gives the name of every category of the test split, the category identifiers being the class indices of the
    identity label mapping of the synthetic shapes (1 to 20, the background being 0).

    Args:
        test_folder (Path): The test split.

    Returns:
        Dict[int, str]: The name of every category identifier.
    """
    return {category["id"]: category["name"] for category in read_test_manifest(test_folder=test_folder)["categories"]}


def read_network_image(test_folder: Path, image_name: str) -> Tuple[numpy.ndarray, torch.Tensor]:
    """
    Reads a test image the way the COCO loader does (read_image, OpenCV BGR uint8), already at the 512 pixels of the
    network input of both runs, and gives it with its [1, 3, H, W] float input tensor in [0, 255].

    Args:
        test_folder (Path): The test split.
        image_name (str): The file name of the image.

    Returns:
        Tuple[numpy.ndarray, torch.Tensor]: The [H, W, 3] BGR image and the input tensor.
    """
    image = read_image(image_path=test_folder / "data" / image_name)
    return image, torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(dim=0)


def load_run_model(run_folder: Path, iteration: int) -> TaskModel:
    """
    Rebuilds the trained model of a run at a checkpoint, on the CPU and in evaluation mode, with the loader of the
    evaluation and deployment scripts (its trained weights, not their moving average), the threads of PyTorch limited
    to those of the build so that the trainings keep the processors.

    Args:
        run_folder (Path): The run folder, holding training_configuration.yaml and Weights/.
        iteration (int): The checkpoint iteration.

    Returns:
        TaskModel: The TwoStageDetector or InstanceSegmenter of the run.
    """
    torch.set_num_threads(read_thread_count())
    checkpoint_path = get_checkpoint_path(experiment_folder=run_folder, iteration=iteration)
    return load_trained_model(configuration_path=run_folder / "training_configuration.yaml",
                              checkpoint_path=checkpoint_path, device=torch.device("cpu"), use_averaged_weights=False)


def read_scalar_series(log_folder: Path, tag: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Reads every value of one scalar of a TensorBoard log folder of a run (Train, Test or Test_EMA), the curve
    TensorBoard draws.

    Args:
        log_folder (Path): The log folder.
        tag (str): The scalar, e.g. "Total Loss" or "mAP@0.5/area".

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The iterations and the values.

    Raises:
        FileNotFoundError: If the log folder is missing.
        KeyError: If the folder has no such scalar.
    """
    if not log_folder.is_dir():
        raise FileNotFoundError(f"No TensorBoard log folder {log_folder}.")
    accumulator = EventAccumulator(path=str(log_folder), size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags()["scalars"]:
        raise KeyError(f"No scalar {tag!r} in {log_folder}, whose scalars are {accumulator.Tags()['scalars']}.")
    events = accumulator.Scalars(tag)
    return numpy.array([event.step for event in events]), numpy.array([event.value for event in events])


def count_scalar_tags(log_folder: Path) -> int:
    """
    Counts the scalars of a TensorBoard log folder, the number of curves a Test folder holds per evaluation.

    Args:
        log_folder (Path): The log folder.

    Returns:
        int: The number of scalar tags.
    """
    accumulator = EventAccumulator(path=str(log_folder), size_guidance={"scalars": 1})
    accumulator.Reload()
    return len(accumulator.Tags()["scalars"])


def strip_terminal_codes(text: str) -> str:
    """
    Removes the colours, styles and hyperlinks of a console log line, the text the console showed.

    Args:
        text (str): A line of a console log.

    Returns:
        str: The visible text of the line.
    """
    return TERMINAL_COLOUR_PATTERN.sub(repl="", string=TERMINAL_LINK_PATTERN.sub(repl="", string=text))


def read_results_block(results_path: Path, iteration: int, averaged_weights: bool) -> List[str]:
    """
    Reads the report of one evaluated checkpoint from the Results.txt of a run: the lines from its header
    ("Iteration <n> · ... Test Images · ...", with "EMA Weights" for the moving average of the weights) to the next
    header, as the trainer printed them. The reports are appended to Results.txt, so the last block of the iteration
    and weights is read, that of the latest evaluation when the checkpoint was evaluated again.

    Args:
        results_path (Path): The Results.txt of the run.
        iteration (int): The evaluated iteration.
        averaged_weights (bool): Whether the block of the moving average of the weights is read.

    Returns:
        List[str]: The lines of the block, its header first.

    Raises:
        ValueError: If the file has no such block.
    """
    lines = results_path.read_text(encoding="utf-8").splitlines()
    header_indices = [line_index for line_index, line in enumerate(lines) if re.match(pattern=r"^Iteration \d+ · ",
                                                                                         string=line)]

    # The headers searched from the end of the file, the first match being the latest report of the checkpoint
    for position, header_index in reversed(list(enumerate(header_indices))):
        header = lines[header_index]
        if header.startswith(f"Iteration {iteration} · ") and (("EMA Weights" in header) == averaged_weights):
            end_index = header_indices[position + 1] if position + 1 < len(header_indices) else len(lines)
            return [line.rstrip() for line in lines[header_index:end_index]]
    raise ValueError(f"No block of iteration {iteration} (averaged weights {averaged_weights}) in {results_path}.")


def extract_table(block_lines: List[str], title: str) -> List[str]:
    """
    Gives one table of a Results.txt block: its title line and the lines after it up to the first blank line, the
    excerpt a listing of the videos shows.

    Args:
        block_lines (List[str]): The lines of a block (read_results_block).
        title (str): The title line of the table without its indentation, e.g. "Mean Average Precision".

    Returns:
        List[str]: The lines of the table, the indentation of the report removed.

    Raises:
        ValueError: If the block has no such table.
    """
    stripped_lines = [line[2:] if line.startswith("  ") else line for line in block_lines]
    if title not in stripped_lines:
        raise ValueError(f"No table {title!r} in the block {block_lines[0]!r}.")
    start_index = stripped_lines.index(title)
    table_lines = []
    for line in stripped_lines[start_index:]:
        if not line.strip():
            break
        table_lines.append(line)
    return table_lines


def parse_table_rows(table_lines: List[str]) -> Dict[str, List[str]]:
    """
    Splits the rows of a Results.txt table into the label of every row (the text before its first number) and its
    values, the table being read below its rule line. A rate written as a percentage stays one value with its sign
    ("78.64 %"), for convert_report_value of the evaluation figures.

    Args:
        table_lines (List[str]): The lines of a table (extract_table).

    Returns:
        Dict[str, List[str]]: The values of every row, keyed by its label, e.g. {"Small (< 32²)": ["184", "1.47 %"]},
            or {"Small (< 32²)": ["184", "0.0147"]} for a report written before the percentages.
    """
    rows = {}
    rule_indices = [line_index for line_index, line in enumerate(table_lines) if line.strip().startswith("─")]
    for line in table_lines[rule_indices[0] + 1:] if rule_indices else []:
        # Rows of the table end at its footnote, the first line that is not label then numbers, percentages or dashes
        match = REPORT_ROW_PATTERN.match(string=line)
        if match is None:
            break
        # Values found one by one, so that the space before the percent sign does not split a percentage in two
        rows[match.group(1).strip()] = re.findall(pattern=REPORT_VALUE_PATTERN, string=match.group(2))
    return rows


def write_listing(lines: List[str], output_path: Path) -> None:
    """
    Writes the lines of a listing shown by a diagram of the videos (a manifest excerpt, a table of a report, a console
    summary), read by the builder like the excerpts of the code scenes.

    Args:
        lines (List[str]): The lines.
        output_path (Path): The text file, its folder created if missing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(data="\n".join(lines) + "\n", encoding="utf-8")


def hexadecimal_to_rgb(colour: str) -> Tuple[int, int, int]:
    """
    Turns a "#rrggbb" colour of the chart palette into the RGB triple PIL draws with.

    Args:
        colour (str): The colour.

    Returns:
        Tuple[int, int, int]: Its red, green and blue values.
    """
    return int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)


def load_caption_fonts() -> Tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    """
    Loads the fonts of the captions drawn on the images: Inter semibold for the headings and Hack for the numbers and
    names.

    Returns:
        Tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]: The heading font and the caption font.
    """
    return (ImageFont.truetype(font=str(SANS_SEMIBOLD_FONT_PATH), size=30),
            ImageFont.truetype(font=str(MONOSPACE_FONT_PATH), size=BOX_CAPTION_FONT_SIZE))


def draw_box(draw: ImageDraw.ImageDraw, box: List[float], colour: Tuple[int, int, int], width: int,
             caption: str, font: ImageFont.FreeTypeFont, offset: Tuple[int, int], scale: float,
             caption_below: bool) -> None:
    """
    Draws one box on an image of a figure, with an optional caption on a dark band above its top left corner (inside
    the box when it touches the top of the image, below it on request), the box given in network input pixels.

    Args:
        draw (ImageDraw.ImageDraw): The drawing context of the figure.
        box (List[float]): The [x1, y1, x2, y2] box in network input pixels.
        colour (Tuple[int, int, int]): The colour of the outline and of the caption text.
        width (int): The width of the outline in pixels.
        caption (str): The caption, empty for none.
        font (ImageFont.FreeTypeFont): The font of the caption.
        offset (Tuple[int, int]): The position of the top left corner of the image in the figure.
        scale (float): The scale of the image in the figure.
        caption_below (bool): Whether the caption goes under the box, for a box sharing its corner with another.
    """
    left, top = offset[0] + box[0] * scale, offset[1] + box[1] * scale
    right, bottom = offset[0] + box[2] * scale, offset[1] + box[3] * scale
    draw.rectangle(xy=(left, top, right, bottom), outline=colour, width=width)

    # Caption on a dark band, above the box or inside it when the box touches the top of the image
    if caption:
        text_box = draw.textbbox(xy=(0, 0), text=caption, font=font)
        text_width, text_height = text_box[2] - text_box[0], text_box[3] - text_box[1]
        band_top = top - text_height - 8 if top - text_height - 8 >= offset[1] else top + 2
        if caption_below:
            band_top = bottom + 2
        draw.rectangle(xy=(left, band_top, left + text_width + 8, band_top + text_height + 8), fill=(13, 15, 18))
        draw.text(xy=(left + 4, band_top + 4 - text_box[1]), text=caption, fill=colour, font=font)
