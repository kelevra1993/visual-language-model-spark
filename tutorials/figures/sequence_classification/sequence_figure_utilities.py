"""
Helpers of the figure scripts of the sequence classification (M2O) and sequence comparison (CMP) tutorial series
(tutorials/figures/sequence_classification and tutorials/figures/sequence_comparison), run by make_figures.py with the
Python of the repository at every build: the runs compared with the main run of a series (the 'comparison_runs' of
tutorials/tutorial_sources.yaml), the trained sequence models rebuilt on the CPU from their checkpoints, the TensorBoard
curves, predictions and console logs of the runs, the frames of the datasets copied into the figures folder so that the
HTML scenes can show them, the frame vectors drawn as small heat squares, the composites of the evaluation images of
the runs, and the layout files of the animated batch scenes (animations/sequence_classification/frame_batches.js).
"""

import csv
import json
import re
import shutil

import numpy
import torch

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Any, Dict, List, Optional, Tuple

from tutorials.figures.figure_utilities import SOURCES_FILE, read_tutorial_sources, round_fact
from tutorials.tutorial_builder.layout import (CARD_COLOUR, DIM_TEXT_COLOUR, MONOSPACE_FONT_PATH, NAMED_COLOURS,
                                               SANS_BOLD_FONT_PATH, SANS_FONT_PATH, TERMINAL_COLOUR, TEXT_COLOUR)
from tutorials.tutorial_builder.settings import THREAD_COUNT
from vision_training.models.model_loading import load_trained_model
from vision_training.models.task_model import TaskModel

# Escape sequences of the coloured console output and of its terminal hyperlinks, removed from the logs of the runs
# before they are shown, the hyperlinks keeping the text they display
CONSOLE_COLOUR_PATTERN = re.compile(pattern=r"\x1b\[[0-9;]*m")
CONSOLE_LINK_PATTERN = re.compile(pattern=r"\x1b\]8;;[^\x1b]*\x1b\\")

# Colours of the vector squares, from the box colour of the diagrams (a zero) to the blue of the series (the largest
# value given), so that the zero vectors of the padding frames are plain squares
VECTOR_LOW_COLOUR = numpy.array(NAMED_COLOURS["box"], dtype=numpy.float64)
VECTOR_HIGH_COLOUR = numpy.array(NAMED_COLOURS["blue"], dtype=numpy.float64)

# Sizes of the fonts of the composite figures: Inter bold for the headings, regular for the notes under them
COMPOSITE_HEADING_FONT_SIZE = 34
COMPOSITE_NOTE_FONT_SIZE = 24
COMPOSITE_HEADER_HEIGHT = 110
COMPOSITE_GAP = 40


def find_comparison_run_folder(experiments_folder: Path, run_key: str, comparison_key: str) -> Path:
    """
    Finds the folder of a run compared with the main run of a series, listed under 'comparison_runs' of its run in
    tutorial_sources.yaml (the average pooling run of the sequence classification, for instance), so that a missing run
    stops the build with how to train it again.

    Args:
        experiments_folder (Path): The experiments folder of the build.
        run_key (str): The run of the series under 'runs', e.g. "sequence_classification".
        comparison_key (str): The compared run under its 'comparison_runs', e.g. "average_pooling_512".

    Returns:
        Path: The run folder.

    Raises:
        KeyError: If the sources file does not list the compared run.
        FileNotFoundError: If the run folder does not exist.
    """
    comparison_runs = read_tutorial_sources()["runs"].get(run_key, {}).get("comparison_runs", {})
    if comparison_key not in comparison_runs:
        raise KeyError(f"No comparison run {comparison_key} under the run {run_key} of {SOURCES_FILE}, whose "
                       f"comparison runs are {sorted(comparison_runs)}.")
    run_folder = experiments_folder / comparison_runs[comparison_key]["folder"]
    if not run_folder.is_dir():
        raise FileNotFoundError(f"No run folder {run_folder} for the comparison run {comparison_key} of {run_key} in "
                                f"{SOURCES_FILE}: make it again with "
                                f"{comparison_runs[comparison_key]['trained_with']}.")
    return run_folder


def find_console_log(experiments_folder: Path, run_key: str, comparison_key: Optional[str]) -> Path:
    """
    Finds the console log of the training of a run, its 'console_log' in tutorial_sources.yaml (the output of the
    training redirected to a file), for the console figures and the throughputs of the narration.

    Args:
        experiments_folder (Path): The experiments folder of the build.
        run_key (str): The run of the series under 'runs'.
        comparison_key (Optional[str]): The compared run under its 'comparison_runs', or None for the run itself.

    Returns:
        Path: The log file.

    Raises:
        KeyError: If the run has no 'console_log'.
        FileNotFoundError: If the log does not exist.
    """
    run_entry = read_tutorial_sources()["runs"][run_key]
    if comparison_key is not None:
        run_entry = run_entry["comparison_runs"][comparison_key]
    if "console_log" not in run_entry:
        raise KeyError(f"No console_log for the run {run_key} {comparison_key or ''} in {SOURCES_FILE}.")
    log_path = experiments_folder / run_entry["console_log"]
    if not log_path.exists():
        raise FileNotFoundError(f"No console log {log_path} for the run {run_key} {comparison_key or ''} of "
                                f"{SOURCES_FILE}: it is the output of the training, redirected to that file.")
    return log_path


def load_sequence_model(run_folder: Path, iteration: int) -> TaskModel:
    """
    Rebuilds the trained model of a sequence run at one of its checkpoints, on the CPU and in evaluation mode, with
    the loader of the evaluation and deployment scripts, so that the model figures show what the trained network
    computes; the threads of PyTorch are limited to those of the build.

    Args:
        run_folder (Path): The run folder, with its training_configuration.yaml and Weights folder.
        iteration (int): The iteration of the checkpoint.

    Returns:
        TaskModel: The trained model.

    Raises:
        FileNotFoundError: If the checkpoint is missing.
    """
    torch.set_num_threads(THREAD_COUNT)
    checkpoint_path = run_folder / "Weights" / f"Iteration_{iteration}" / f"model_{iteration:06d}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint {checkpoint_path} in the run {run_folder.name}.")
    return load_trained_model(configuration_path=run_folder / "training_configuration.yaml",
                              checkpoint_path=checkpoint_path, device=torch.device("cpu"), use_averaged_weights=False)


def read_scalar_curve(run_folder: Path, split: str, tag: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Reads every value of one scalar of the TensorBoard logs of a run, the curve TensorBoard draws for it.

    Args:
        run_folder (Path): The run folder.
        split (str): "Train", "Validation" or "Test".
        tag (str): The scalar, e.g. "Total Loss".

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The iterations and the values.

    Raises:
        FileNotFoundError: If the log folder is missing.
        KeyError: If the logs hold no such scalar.
    """
    log_folder = run_folder / "Tensorboard" / split
    if not log_folder.is_dir():
        raise FileNotFoundError(f"No TensorBoard log folder {log_folder}.")
    accumulator = EventAccumulator(path=str(log_folder), size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags()["scalars"]:
        raise KeyError(f"No scalar {tag!r} in {log_folder}, whose scalars are {accumulator.Tags()['scalars']}.")
    events = accumulator.Scalars(tag)
    return numpy.array([event.step for event in events]), numpy.array([event.value for event in events])


def read_predictions(run_folder: Path, iteration: int) -> List[Dict[str, str]]:
    """
    Reads the predictions file written by the evaluation of a checkpoint (save_predictions), one row per test
    sequence or pair with its true class, predicted class and the probability of every class.

    Args:
        run_folder (Path): The run folder.
        iteration (int): The evaluated iteration.

    Returns:
        List[Dict[str, str]]: The rows of the file, in its order.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    predictions_path = run_folder / "Evaluation" / f"Iteration_{iteration}" / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"No predictions file {predictions_path}.")
    with open(file=predictions_path, mode="r", newline="", encoding="utf-8") as predictions_file:
        return list(csv.DictReader(predictions_file))


def clean_console_text(text: str) -> str:
    """
    Removes the colour codes and terminal hyperlinks of a console log of a run, so that it reads as the terminal shows
    it, a hyperlink keeping the name it displays.

    Args:
        text (str): The raw log.

    Returns:
        str: The log as plain text.
    """
    return CONSOLE_LINK_PATTERN.sub(repl="", string=CONSOLE_COLOUR_PATTERN.sub(repl="", string=text))


def write_text_lines(output_path: Path, lines: List[str]) -> None:
    """
    Writes the lines of a listing read by the storyboards (a manifest entry, a console excerpt), its folder created
    if missing.

    Args:
        output_path (Path): The text file.
        lines (List[str]): Its lines.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(data="\n".join(lines) + "\n", encoding="utf-8")


def copy_frames(frame_paths: List[Path], series_folder: Path, subfolder: str, name_prefix: str) -> List[str]:
    """
    Copies frames of a dataset into the figures folder of a series, the only data folder the HTML scenes can read, and
    gives their paths with the figures: prefix of the storyboards.

    Args:
        frame_paths (List[Path]): The frame files, in order.
        series_folder (Path): The figures folder of the series in the cache (the output folder of make_figures).
        subfolder (str): The folder of the copies inside it, e.g. "frames".
        name_prefix (str): The start of the name of every copy, followed by its index.

    Returns:
        List[str]: The figures: paths of the copies, e.g. "figures:sequence_classification/frames/train_000000_00.png".

    Raises:
        FileNotFoundError: If a frame is missing.
    """
    (series_folder / subfolder).mkdir(parents=True, exist_ok=True)
    copied_paths = []
    for frame_index, frame_path in enumerate(frame_paths):
        if not frame_path.exists():
            raise FileNotFoundError(f"No frame {frame_path}.")
        copy_name = f"{name_prefix}_{frame_index:02d}.png"
        shutil.copyfile(src=frame_path, dst=series_folder / subfolder / copy_name)
        copied_paths.append(f"figures:{series_folder.name}/{subfolder}/{copy_name}")
    return copied_paths


def write_vector_square(vector: numpy.ndarray, map_shape: Tuple[int, int, int], maximum_value: float,
                        side_pixels: int, output_path: Path) -> None:
    """
    Draws a frame vector as a small heat square, the way the animated batch scenes show the vectors of the frame
    encoder: the vector read as the feature map it comes from (channels x height x width, a flattened 64 x 8 x 8 map,
    or 64 x 1 x 1 for an average pooling), every channel map placed on a square mosaic of the channels, so that the
    position of the shape shows in the maps of a flattened frame. The values are coloured from the box colour at zero
    to the blue of the series at maximum_value, the zero vectors of the padding frames being plain squares.

    Args:
        vector (numpy.ndarray): The values of the vector, zero or positive (after a ReLU).
        map_shape (Tuple[int, int, int]): The channels, height and width of the map the vector flattens.
        maximum_value (float): The value drawn in full blue, larger values saturating.
        side_pixels (int): The side of the square in pixels.
        output_path (Path): The PNG file.

    Raises:
        ValueError: If the vector does not hold the values of the map, or the channels are not a square number.
    """
    channels, height, width = map_shape
    mosaic_side = int(round(numpy.sqrt(channels)))
    if vector.size != channels * height * width or mosaic_side * mosaic_side != channels:
        raise ValueError(f"A vector square needs a {map_shape} map with a square number of channels, got "
                         f"{vector.size} values.")

    # Channel maps on a square mosaic, mapped to the colour scale, enlarged without smoothing so that every value
    # stays a cell
    mosaic = vector.reshape(mosaic_side, mosaic_side, height, width).transpose(0, 2, 1, 3).reshape(
        mosaic_side * height, mosaic_side * width)
    mix = numpy.clip(mosaic / max(maximum_value, 1e-12), 0.0, 1.0) ** 0.6
    colour_range = (VECTOR_HIGH_COLOUR - VECTOR_LOW_COLOUR)[None, None, :]
    pixels = VECTOR_LOW_COLOUR[None, None, :] + mix[:, :, None] * colour_range
    image = Image.fromarray(obj=numpy.round(pixels).astype(numpy.uint8), mode="RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.resize(size=(side_pixels, side_pixels), resample=Image.Resampling.NEAREST).save(fp=output_path)


def compute_cosine_similarities(vectors: torch.Tensor) -> List[List[float]]:
    """
    Gives the cosine similarity of every pair of frame vectors of a sequence, the grid the pooling scene shows: close
    to 1 everywhere when the vectors do not tell the frames apart.

    Args:
        vectors (torch.Tensor): The [T, F] frame vectors.

    Returns:
        List[List[float]]: The [T, T] similarities, rounded for facts.json.
    """
    normalised_vectors = torch.nn.functional.normalize(input=vectors.double(), dim=1)
    similarities = normalised_vectors @ normalised_vectors.T
    return [[round_fact(value=value) for value in row] for row in similarities.tolist()]


def write_animation_layout(output_path: Path, phases: List[str], tiles: List[Dict[str, Any]],
                           texts: List[Dict[str, Any]]) -> None:
    """
    Writes the layout file of an animated batch scene, read by animations/sequence_classification/frame_batches.js:
    its phases in order, every tile (a frame, a padding frame or a vector square) with its image, relative to the
    layout file, and its box and opacity at the phases where it changes, and every text with the phases it is shown
    in. The storyboard maps the phases to its sentences.

    Args:
        output_path (Path): The JSON file, in the figures folder of the series.
        phases (List[str]): The names of the phases, in their order.
        tiles (List[Dict[str, Any]]): The tiles: 'image' (path relative to the layout file, or None for a plain
            square), 'fill', 'outline', 'label' and 'states', a phase name mapped to its 'box' [left, top, width,
            height] and 'opacity'.
        texts (List[Dict[str, Any]]): The texts: 'text', 'position', 'anchor', 'font', 'colour' and 'phases', the
            names of the phases it is shown in.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(data=json.dumps(obj={"phases": phases, "tiles": tiles, "texts": texts}, indent=1),
                           encoding="utf-8")


def compose_image_grid(rows: List[List[Path]], row_headings: List[str], column_headings: List[str],
                       image_height: int, output_path: Path) -> Dict[str, Any]:
    """
    Lays out evaluation images of several runs on one figure, one row per run and one column per kind of image (the
    confusion matrix and the ROC curves of every run, for instance), every image scaled to the same height, each row
    under its heading, so that a scene compares the runs with camera views on the rows; gives the box of every row
    and of every image.

    Args:
        rows (List[List[Path]]): The image files of every row, in column order.
        row_headings (List[str]): The heading of every row, above it.
        column_headings (List[str]): The note under the heading of every column, written in the first row only.
        image_height (int): The height of every image on the figure.
        output_path (Path): The PNG file of the figure.

    Returns:
        Dict[str, Any]: The 'figure_box', and the 'row_<index>_box' and 'row_<index>_image_<index>_box' of the images.

    Raises:
        FileNotFoundError: If an image is missing.
    """
    missing_paths = [str(path) for row in rows for path in row if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing evaluation images: {missing_paths}")

    # Every image scaled to the common height, the widest row giving the width of the figure
    scaled_rows = []
    for row in rows:
        scaled_images = []
        for image_path in row:
            with Image.open(fp=image_path) as image:
                scale = image_height / image.height
                scaled_images.append(image.convert(mode="RGB").resize(
                    size=(round(image.width * scale), image_height), resample=Image.Resampling.LANCZOS))
        scaled_rows.append(scaled_images)
    figure_width = max(sum(image.width for image in row) + COMPOSITE_GAP * (len(row) - 1) for row in scaled_rows)
    row_height = COMPOSITE_HEADER_HEIGHT + image_height
    canvas = Image.new(mode="RGB", size=(figure_width, len(rows) * row_height + (len(rows) - 1) * COMPOSITE_GAP),
                       color=CARD_COLOUR)
    draw = ImageDraw.Draw(im=canvas)
    heading_font = ImageFont.truetype(font=str(SANS_BOLD_FONT_PATH), size=COMPOSITE_HEADING_FONT_SIZE)
    note_font = ImageFont.truetype(font=str(SANS_FONT_PATH), size=COMPOSITE_NOTE_FONT_SIZE)

    # Every row under its heading, the images left to right, then their boxes
    facts = {"figure_box": [0, 0, canvas.width, canvas.height]}
    for row_index, (scaled_images, heading) in enumerate(zip(scaled_rows, row_headings)):
        row_top = row_index * (row_height + COMPOSITE_GAP)
        draw.text(xy=(0, row_top + 34), text=heading, fill=TEXT_COLOUR, font=heading_font, anchor="lm")
        image_left = 0
        for image_index, image in enumerate(scaled_images):
            if row_index == 0 and image_index < len(column_headings):
                draw.text(xy=(image_left, row_top + 84), text=column_headings[image_index], fill=DIM_TEXT_COLOUR,
                          font=note_font, anchor="lm")
            image_top = row_top + COMPOSITE_HEADER_HEIGHT
            canvas.paste(im=image, box=(image_left, image_top))
            facts[f"row_{row_index}_image_{image_index}_box"] = [image_left, image_top, image_left + image.width,
                                                                 image_top + image.height]
            image_left += image.width + COMPOSITE_GAP
        facts[f"row_{row_index}_box"] = [0, row_top, image_left - COMPOSITE_GAP, row_top + row_height]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(fp=output_path)
    return facts


def read_interval_summaries(log_path: Path) -> List[Dict[str, float]]:
    """
    Reads the interval summaries a training prints every 'information_dump' iterations from the console log of a run:
    the iteration, the samples (sequences or pairs) and real frames per second, and the moving average of the
    training total loss over the interval.

    Args:
        log_path (Path): The console log of the run.

    Returns:
        List[Dict[str, float]]: The 'iteration', 'throughput', 'frame_throughput' and 'train_total_loss' of every
            interval, in order.

    Raises:
        FileNotFoundError: If the log is missing.
        ValueError: If the log holds no interval summary.
    """
    if not log_path.exists():
        raise FileNotFoundError(f"No console log {log_path}.")
    summaries = []
    header_pattern = re.compile(pattern=r"Iteration:\s+(\d+).*Throughput:\s+([\d.]+) \w+ Per Second.*?([\d.]+) Frames "
                                        r"Per Second")
    loss_pattern = re.compile(pattern=r"^Moving Average of Train Total Loss\s+:\s+([\d.]+)")
    for line in clean_console_text(text=log_path.read_text(encoding="utf-8", errors="replace")).splitlines():
        header_match = header_pattern.search(string=line)
        if header_match is not None:
            summaries.append({"iteration": int(header_match.group(1)), "throughput": float(header_match.group(2)),
                              "frame_throughput": float(header_match.group(3))})
        loss_match = loss_pattern.search(string=line)
        if loss_match is not None and summaries:
            summaries[-1]["train_total_loss"] = float(loss_match.group(1))
    if not summaries:
        raise ValueError(f"No interval summary in the console log {log_path}.")
    return summaries


def extract_interval_block(log_path: Path, iteration: int) -> List[str]:
    """
    Cuts the interval summary of one iteration out of the console log of a run, from the rule above its header to the
    rule closing it, as the terminal shows it.

    Args:
        log_path (Path): The console log of the run.
        iteration (int): The iteration of the summary.

    Returns:
        List[str]: The lines of the summary.

    Raises:
        ValueError: If the log holds no summary of that iteration.
    """
    lines = clean_console_text(text=log_path.read_text(encoding="utf-8", errors="replace")).splitlines()
    header_index = next((index for index, line in enumerate(lines)
                         if re.search(pattern=rf"Iteration:\s+{iteration}\s+\|", string=line)), None)
    if header_index is None:
        raise ValueError(f"No interval summary of iteration {iteration} in {log_path}.")
    end_index = next(index for index in range(header_index + 1, len(lines)) if lines[index].startswith("====="))
    return lines[header_index - 1:end_index + 1]


def draw_console_figure(lines: List[str], output_path: Path, font_size: int,
                        highlight_patterns: Dict[str, str]) -> Dict[str, List[int]]:
    """
    Draws lines of a console log as a terminal image in the Hack font of the videos, the way an image scene shows a
    console output too wide for a code excerpt, and gives the box of the lines matching every highlight pattern.

    Args:
        lines (List[str]): The lines of the log.
        output_path (Path): The PNG file.
        font_size (int): The size of the font in pixels.
        highlight_patterns (Dict[str, str]): A name for every regular expression whose matching lines get a box.

    Returns:
        Dict[str, List[int]]: The 'figure_box', and the '<name>_box' spanning the lines matched by every pattern.

    Raises:
        ValueError: If a pattern matches no line.
    """
    font = ImageFont.truetype(font=str(MONOSPACE_FONT_PATH), size=font_size)
    line_height = round(font_size * 1.45)
    margin = 24
    width = round(max(font.getlength(text=line) for line in lines)) + 2 * margin
    canvas = Image.new(mode="RGB", size=(width, len(lines) * line_height + 2 * margin), color=TERMINAL_COLOUR)
    draw = ImageDraw.Draw(im=canvas)
    for line_index, line in enumerate(lines):
        draw.text(xy=(margin, margin + line_index * line_height), text=line, fill=TEXT_COLOUR, font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(fp=output_path)

    # Box of the lines of every pattern, from the first matching line to the last one
    boxes = {"figure_box": [0, 0, canvas.width, canvas.height]}
    for name, pattern in highlight_patterns.items():
        matched_indices = [index for index, line in enumerate(lines) if re.search(pattern=pattern, string=line)]
        if not matched_indices:
            raise ValueError(f"The pattern {pattern!r} matches no line of the console figure {output_path.name}.")
        boxes[f"{name}_box"] = [margin - 8, margin + matched_indices[0] * line_height - 4, width - margin + 8,
                                margin + (matched_indices[-1] + 1) * line_height]
    return boxes


def extract_launch_block(log_path: Path, last_line_start: str) -> List[str]:
    """
    Cuts the experiment launch information out of the console log of a run, from its heading to the line starting
    with last_line_start, its file and folder links showing their names as in the terminal.

    Args:
        log_path (Path): The console log of the run.
        last_line_start (str): The start of the last line kept, e.g. "- Batch Size".

    Returns:
        List[str]: The lines of the block.

    Raises:
        ValueError: If the log holds no launch information or no such last line.
    """
    lines = clean_console_text(text=log_path.read_text(encoding="utf-8", errors="replace")).splitlines()
    heading_index = next((index for index, line in enumerate(lines) if "Experiment Launch Information" in line), None)
    if heading_index is None:
        raise ValueError(f"No experiment launch information in {log_path}.")
    end_index = next((index for index in range(heading_index, len(lines))
                      if lines[index].startswith(last_line_start)), None)
    if end_index is None:
        raise ValueError(f"No line starting with {last_line_start!r} after the launch information of {log_path}.")
    return lines[heading_index:end_index + 1]


# Words of the small numbers, which the voice reads and the speech recognition writes as words, so that the narration
# quotes them as words and the narration check compares words with words
SMALL_NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                      "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
                      "twenty"]


def add_number_words(facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adds to the facts of a series the words of every whole number from 0 to 20, as '<name>_words' (the lengths of the
    sequences of the animated batches, the frames of the dataset), which the narration quotes instead of the digits.

    Args:
        facts (Dict[str, Any]): The facts of the series.

    Returns:
        Dict[str, Any]: The facts with the words of their small whole numbers.

    Raises:
        ValueError: If a word fact would replace a fact of the same name.
    """
    word_facts = {f"{name}_words": SMALL_NUMBER_WORDS[value] for name, value in facts.items()
                  if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 20}
    repeated_names = sorted(set(word_facts) & set(facts))
    if repeated_names:
        raise ValueError(f"The word facts {repeated_names} would replace facts of the same name.")
    return {**facts, **word_facts}
