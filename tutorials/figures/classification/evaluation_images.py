"""
Evaluation figures of the classification tutorials (the chapter reading a classification evaluation), made at every
build by tutorials/figures/classification/figures.py from the evaluation folders of the classification run: the eight
fixed sample images of the test evaluations of two iterations side by side, each grid under its iteration, with the
samples predicted wrong at the first of them, and one error image of each of the largest confusions of the last
checkpoint, from the standalone evaluation of the run (classification_evaluation in tutorials/tutorial_sources.yaml),
each under its folder and file name, so that the scenes show the real images the repository writes. The images are
only placed on the card colour of the videos, never redrawn; the box of every image and grid is a fact, for the views
and highlights of the scenes, with the classes and probability of every shown error and the counts of error folders
and images written in the report block of the evaluation.
"""

import csv
import re

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Any, Dict, List, Tuple

from tutorials.figures.classification.evaluation_tables import (LAST_ITERATION, find_largest_confusions,
                                                                format_class_name, read_confusion_matrix)
from tutorials.figures.classification.snapshot_helpers import open_predictions_file
from tutorial_builder.layout import (CARD_COLOUR, DIM_TEXT_COLOUR, MONOSPACE_FONT_PATH, SANS_BOLD_FONT_PATH,
                                     TEXT_COLOUR)

# Iterations whose sample images are compared, the first and the last test evaluations of the run, and the number of
# sample images the scene narrates as wrong at the first of them
SAMPLE_ITERATIONS = [500, LAST_ITERATION]
FIRST_SAMPLE_ERRORS = 2

# Iteration of the standalone evaluation, and the number of largest confusions of that checkpoint (its confusion
# matrix in the training evaluation) whose first error image is shown, from its Errors/<true class>-TO-<predicted
# class>/ folder, the file name ending with the probability of the wrong class (the error images saved are the first
# ones the test loader meets, the first in name order is shown)
ERROR_ITERATION = LAST_ITERATION
SHOWN_ERROR_PAIRS = 4

# Folder name of the error images of a pair of classes, as ClassificationTrainer.dump_errors writes it
ERROR_FOLDER_SEPARATOR = "-TO-"

# Line of the report block of an evaluation counting its error images
ERROR_COUNT_PATTERN = re.compile(pattern=r"Error Images : (?P<images>\d+) in (?P<pairs>\d+) class pairs, at most "
                                         r"(?P<limit>\d+) per pair")

# Header of the report block of an evaluated checkpoint in Results.txt
REPORT_BLOCK_PATTERN = re.compile(pattern=r"^Iteration (?P<iteration>\d+) · ", flags=re.MULTILINE)

# Layout: four columns of images, the gap between two images and between two grids, the height of the header band
GRID_COLUMN_COUNT = 4
IMAGE_GAP = 16
GRID_GAP = 800
HEADER_HEIGHT = 90

# Sizes of the fonts of the videos: Inter for the iteration headings, Hack for the folder and file names, the label
# font shrunk by steps of LABEL_FONT_STEP, down to SMALLEST_LABEL_FONT_SIZE, until the longest label fits its image
HEADING_FONT_SIZE = 56
LABEL_FONT_SIZE = 36
LABEL_FONT_STEP = 2
SMALLEST_LABEL_FONT_SIZE = 24

# Height of a label line above an error image: the folder name split after its separator, then the file name
ERROR_LABEL_LINE_HEIGHT = 60


def load_images(image_paths: List[Path]) -> List[Image.Image]:
    """
    Reads images written by the evaluation of the repository, refusing a missing file so that a moved run never gives
    an incomplete figure.

    Args:
        image_paths (List[Path]): The image files.

    Returns:
        List[Image.Image]: The RGB images.

    Raises:
        FileNotFoundError: If an image is missing.
    """
    missing_paths = [str(path) for path in image_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing evaluation images: {missing_paths}")
    return [Image.open(fp=path).convert(mode="RGB") for path in image_paths]


def compute_grid_size(images: List[Image.Image]) -> Tuple[int, int]:
    """
    Gives the size of a grid of images of the same size on GRID_COLUMN_COUNT columns, its header band included.

    Args:
        images (List[Image.Image]): The images of the grid.

    Returns:
        Tuple[int, int]: The width and height of the grid in pixels.
    """
    row_count = (len(images) + GRID_COLUMN_COUNT - 1) // GRID_COLUMN_COUNT
    width = GRID_COLUMN_COUNT * images[0].width + (GRID_COLUMN_COUNT - 1) * IMAGE_GAP
    height = HEADER_HEIGHT + row_count * images[0].height + (row_count - 1) * IMAGE_GAP
    return width, height


def paste_grid(canvas: Image.Image, images: List[Image.Image], left: int, heading: str,
               font: ImageFont.FreeTypeFont) -> List[List[int]]:
    """
    Places a grid of images on the canvas under its heading, and gives the box of every image, the boxes the scenes
    use as camera views and highlights.

    Args:
        canvas (Image.Image): The figure.
        images (List[Image.Image]): The images of the grid, in reading order.
        left (int): The left edge of the grid on the canvas.
        heading (str): The heading written above the grid.
        font (ImageFont.FreeTypeFont): The font of the heading.

    Returns:
        List[List[int]]: The left, top, right and bottom of every image on the canvas.
    """
    draw = ImageDraw.Draw(im=canvas)
    draw.text(xy=(left, HEADER_HEIGHT // 2), text=heading, fill=TEXT_COLOUR, font=font, anchor="lm")
    image_boxes = []
    for image_index, image in enumerate(images):
        image_left = left + (image_index % GRID_COLUMN_COUNT) * (image.width + IMAGE_GAP)
        image_top = HEADER_HEIGHT + (image_index // GRID_COLUMN_COUNT) * (image.height + IMAGE_GAP)
        canvas.paste(im=image, box=(image_left, image_top))
        image_boxes.append([image_left, image_top, image_left + image.width, image_top + image.height])
    return image_boxes


def read_sample_predictions(predictions_path: Path, sample_names: List[str]) -> List[Tuple[str, str]]:
    """
    Reads the true and predicted classes of the sample images from the predictions file of an evaluation (the CSV, or
    the reduced gzipped copy of the snapshots holding the rows of the sample images).

    Args:
        predictions_path (Path): The predictions.csv of the evaluation.
        sample_names (List[str]): The sample images, in the order of the grid.

    Returns:
        List[Tuple[str, str]]: The true and predicted class of every sample image, in the same order.

    Raises:
        ValueError: If a sample image is not in the file.
    """
    with open_predictions_file(predictions_path=predictions_path, newline="") as predictions_file:
        rows = {row["sample"]: row for row in csv.DictReader(f=predictions_file) if row["sample"] in sample_names}
    missing_names = [name for name in sample_names if name not in rows]
    if missing_names:
        raise ValueError(f"The predictions file {predictions_path} lacks the sample images {missing_names}.")
    return [(rows[name]["true_class"], rows[name]["predicted_class"]) for name in sample_names]


def describe_sample_predictions(run_folder: Path, sample_names: List[str]) -> Dict[str, Any]:
    """
    Gives the sample images predicted wrong at every compared iteration: their count, and the true and predicted
    classes of the first two, with their number in the grid, so that the narration names the errors the captions show.

    Args:
        run_folder (Path): The folder of the classification run.
        sample_names (List[str]): The sample images, in the order of the grid.

    Returns:
        Dict[str, Any]: The facts, named evaluation.samples.iteration_<n>.wrong_*.

    Raises:
        ValueError: If the first iteration does not have FIRST_SAMPLE_ERRORS wrong images, the first of them being the
            only one wrong at the last iteration, as the sample scene narrates.
    """
    facts = {}
    for iteration in SAMPLE_ITERATIONS:
        predictions = read_sample_predictions(predictions_path=run_folder / "Evaluation" / f"Iteration_{iteration}" /
                                              "predictions.csv", sample_names=sample_names)
        wrong_numbers = [number for number, (true_class, predicted_class) in enumerate(predictions, start=1)
                         if true_class != predicted_class]
        prefix = f"evaluation.samples.iteration_{iteration}"
        facts[f"{prefix}.wrong_count"] = len(wrong_numbers)
        for wrong_rank, image_number in enumerate(wrong_numbers[:2], start=1):
            true_class, predicted_class = predictions[image_number - 1]
            facts.update({f"{prefix}.wrong_{wrong_rank}.image_number": image_number,
                          f"{prefix}.wrong_{wrong_rank}.true_class": format_class_name(class_name=true_class),
                          f"{prefix}.wrong_{wrong_rank}.predicted_class": format_class_name(
                              class_name=predicted_class)})

    # The sample scene names two errors at the first iteration, the first of them being the only one left at the last
    first_prefix, last_prefix = [f"evaluation.samples.iteration_{iteration}"
                                 for iteration in (SAMPLE_ITERATIONS[0], SAMPLE_ITERATIONS[-1])]
    if (facts[f"{first_prefix}.wrong_count"] != FIRST_SAMPLE_ERRORS or facts[f"{last_prefix}.wrong_count"] != 1 or
            facts[f"{last_prefix}.wrong_1.image_number"] != facts[f"{first_prefix}.wrong_1.image_number"]):
        wrong_images = {name: value for name, value in facts.items() if name.endswith("image_number")}
        raise ValueError(f"The sample scene of 02_judging_and_shipping.yaml narrates {FIRST_SAMPLE_ERRORS} wrong "
                         f"sample images at iteration {SAMPLE_ITERATIONS[0]}, the first one alone still wrong at "
                         f"{SAMPLE_ITERATIONS[-1]}, but the wrong images are {wrong_images}: rewrite its narration.")
    return facts


def draw_sample_figure(run_folder: Path, output_path: Path) -> Dict[str, Any]:
    """
    Draws the sample images of the test evaluations of SAMPLE_ITERATIONS side by side, the same eight test images
    captioned by the evaluation with their true class, predicted class and probability, and gives the box of the
    figure, of every grid and of every image.

    Args:
        run_folder (Path): The folder of the classification run.
        output_path (Path): The PNG file of the figure.

    Returns:
        Dict[str, Any]: The boxes and the iterations of the grids, named evaluation.samples.*.

    Raises:
        ValueError: If the two evaluations do not hold the same sample images.
    """
    sample_names, sample_grids = [], []
    for iteration in SAMPLE_ITERATIONS:
        sample_paths = sorted((run_folder / "Evaluation" / f"Iteration_{iteration}" / "Samples").glob("*.png"))
        sample_names.append([path.name for path in sample_paths])
        sample_grids.append(load_images(image_paths=sample_paths))
    if not sample_names[0] or any(names != sample_names[0] for names in sample_names):
        raise ValueError(f"The evaluations of iterations {SAMPLE_ITERATIONS} of {run_folder} do not hold the same "
                         f"sample images: {sample_names}")
    grid_width, grid_height = compute_grid_size(images=sample_grids[0])
    canvas = Image.new(mode="RGB", size=(len(sample_grids) * grid_width + (len(sample_grids) - 1) * GRID_GAP,
                                         grid_height), color=CARD_COLOUR)

    # Every grid under its iteration, then the boxes of the grid and of its images
    heading_font = ImageFont.truetype(font=str(SANS_BOLD_FONT_PATH), size=HEADING_FONT_SIZE)
    facts = {"evaluation.samples.figure_box": [0, 0, canvas.width, canvas.height],
             "evaluation.samples.image_count": len(sample_names[0]), "evaluation.samples.iterations": SAMPLE_ITERATIONS}
    facts.update(describe_sample_predictions(run_folder=run_folder, sample_names=sample_names[0]))
    for grid_index, (iteration, images) in enumerate(zip(SAMPLE_ITERATIONS, sample_grids)):
        grid_left = grid_index * (grid_width + GRID_GAP)
        image_boxes = paste_grid(canvas=canvas, images=images, left=grid_left, heading=f"Iteration {iteration}",
                                 font=heading_font)
        facts[f"evaluation.samples.iteration_{iteration}.grid_box"] = [grid_left, 0, grid_left + grid_width,
                                                                       grid_height]
        for image_index, image_box in enumerate(image_boxes):
            facts[f"evaluation.samples.iteration_{iteration}.image_{image_index + 1}_box"] = image_box

        # Boxes of the first sample images predicted wrong, highlighted by the narration
        for wrong_rank in [1, 2]:
            prefix = f"evaluation.samples.iteration_{iteration}.wrong_{wrong_rank}"
            if f"{prefix}.image_number" in facts:
                facts[f"{prefix}.box"] = image_boxes[facts[f"{prefix}.image_number"] - 1]
    canvas.save(fp=output_path)
    return facts


def find_shown_errors(run_folder: Path, errors_folder: Path) -> List[Tuple[str, str]]:
    """
    Gives the error images of the figure: for each of the SHOWN_ERROR_PAIRS largest confusions of the last checkpoint
    (its confusion matrix in the training evaluation), the first file in name order of its error folder in the
    standalone evaluation, the run snapshots holding only these files.

    Args:
        run_folder (Path): The folder of the classification run.
        errors_folder (Path): The Errors folder of the standalone evaluation of that checkpoint.

    Returns:
        List[Tuple[str, str]]: The folder and file name of every shown error image, the largest confusion first.

    Raises:
        FileNotFoundError: If the error folder of a pair holds no image.
    """
    class_names, counts = read_confusion_matrix(evaluation_folder=run_folder / "Evaluation" /
                                                f"Iteration_{ERROR_ITERATION}")
    shown_errors = []
    for true_index, predicted_index in find_largest_confusions(counts=counts, pair_count=SHOWN_ERROR_PAIRS):
        folder = f"{class_names[true_index]}{ERROR_FOLDER_SEPARATOR}{class_names[predicted_index]}"
        file_paths = sorted((errors_folder / folder).glob("*.png"))
        if not file_paths:
            raise FileNotFoundError(f"No error image in {errors_folder / folder}: evaluate the checkpoint with error "
                                    f"images on.")
        shown_errors.append((folder, file_paths[0].name))
    return shown_errors


def read_error_counts(evaluation_folder: Path) -> Dict[str, int]:
    """
    Reads the number of error images, of error folders (pairs of classes) and the limit per pair from the report block
    of ERROR_ITERATION in the Results.txt of the standalone evaluation, so that the counts do not need every error
    image in the run snapshots.

    Args:
        evaluation_folder (Path): The '<run>/Checkpoint_Evaluation' folder.

    Returns:
        Dict[str, int]: The 'images', 'pairs' and 'limit'.

    Raises:
        ValueError: If the report has no block of ERROR_ITERATION or no error image line in it.
    """
    report_text = (evaluation_folder / "Results.txt").read_text(encoding="utf-8")
    block_starts = {int(match.group("iteration")): match.start()
                    for match in REPORT_BLOCK_PATTERN.finditer(report_text)}
    if ERROR_ITERATION not in block_starts:
        raise ValueError(f"{evaluation_folder}/Results.txt has no block of iteration {ERROR_ITERATION}, only "
                         f"{sorted(block_starts)}.")
    block_ends = [start for start in block_starts.values() if start > block_starts[ERROR_ITERATION]]
    block_text = report_text[block_starts[ERROR_ITERATION]:min(block_ends) if block_ends else len(report_text)]
    match = ERROR_COUNT_PATTERN.search(block_text)
    if match is None:
        raise ValueError(f"The block of iteration {ERROR_ITERATION} of {evaluation_folder}/Results.txt counts no error "
                         f"images: evaluate the checkpoint with error images on.")
    return {name: int(match.group(name)) for name in ["images", "pairs", "limit"]}


def fit_label_font(labels: List[str], width: int) -> ImageFont.FreeTypeFont:
    """
    Gives the monospace font of the labels above the error images at the largest size, from LABEL_FONT_SIZE down, at
    which the longest label fits the width of an image, since the folder names of long class names (a hollow
    parallelogram taken for a hollow trapezoid) would run over the next image.

    Args:
        labels (List[str]): Every label line drawn above the images.
        width (int): The width of an image, in pixels.

    Returns:
        ImageFont.FreeTypeFont: The font.

    Raises:
        ValueError: If a label does not fit even at SMALLEST_LABEL_FONT_SIZE.
    """
    for font_size in range(LABEL_FONT_SIZE, SMALLEST_LABEL_FONT_SIZE - 1, -LABEL_FONT_STEP):
        font = ImageFont.truetype(font=str(MONOSPACE_FONT_PATH), size=font_size)
        if max(font.getlength(text=label) for label in labels) <= width:
            return font
    raise ValueError(f"The error image labels {labels} do not fit an image width of {width} pixels at "
                     f"{SMALLEST_LABEL_FONT_SIZE} pixels.")


def draw_error_figure(run_folder: Path, evaluation_folder: Path, output_path: Path) -> Dict[str, Any]:
    """
    Draws one error image of each of the largest confusions of the last checkpoint in a row, each under the name of its
    folder and file, and gives the box of the figure and of every image with its classes and probability.

    Args:
        run_folder (Path): The folder of the classification run.
        evaluation_folder (Path): The folder of the standalone evaluation of the classification run,
            '<run>/Checkpoint_Evaluation', holding its iteration folders.
        output_path (Path): The PNG file of the figure.

    Returns:
        Dict[str, Any]: The boxes, the classes and probabilities of the shown errors, the counts of the error folders
            and images and the iteration evaluated, named evaluation.errors.*.
    """
    errors_folder = evaluation_folder / f"Iteration_{ERROR_ITERATION}" / "Errors"
    shown_errors = find_shown_errors(run_folder=run_folder, errors_folder=errors_folder)
    images = load_images(image_paths=[errors_folder / folder / file_name for folder, file_name in shown_errors])
    image_width, image_height = images[0].size

    # Labels of every image: its folder name split after the separator on two lines, then its file name
    label_lines = []
    for folder, file_name in shown_errors:
        true_class, predicted_class = folder.split(ERROR_FOLDER_SEPARATOR)
        label_lines.append([f"{true_class}{ERROR_FOLDER_SEPARATOR}", f"{predicted_class}/", file_name])
    label_font = fit_label_font(labels=[line for lines in label_lines for line in lines], width=image_width)
    header_height = ERROR_LABEL_LINE_HEIGHT * len(label_lines[0])
    canvas = Image.new(mode="RGB", size=(len(images) * image_width + (len(images) - 1) * IMAGE_GAP * 2,
                                         header_height + image_height), color=CARD_COLOUR)
    draw = ImageDraw.Draw(im=canvas)

    # Every image under its folder name and its file name, ending with the probability of the wrong class
    facts = {"evaluation.errors.figure_box": [0, 0, canvas.width, canvas.height],
             "evaluation.errors.iteration": ERROR_ITERATION}
    for image_index, ((folder, file_name), image) in enumerate(zip(shown_errors, images)):
        image_left = image_index * (image_width + IMAGE_GAP * 2)
        for line_index, line in enumerate(label_lines[image_index]):
            line_colour = DIM_TEXT_COLOUR if line == file_name else TEXT_COLOUR
            draw.text(xy=(image_left, (line_index + 0.5) * ERROR_LABEL_LINE_HEIGHT), text=line, fill=line_colour,
                      font=label_font, anchor="lm")
        canvas.paste(im=image, box=(image_left, header_height))
        true_class, predicted_class = folder.split(ERROR_FOLDER_SEPARATOR)
        prefix = f"evaluation.errors.image_{image_index + 1}"
        facts.update({f"{prefix}_box": [image_left, 0, image_left + image_width, header_height + image_height],
                      f"{prefix}_probability": float(Path(file_name).stem.split("_")[-1]),
                      f"{prefix}_true_class": format_class_name(class_name=true_class),
                      f"{prefix}_predicted_class": format_class_name(class_name=predicted_class)})

    # Error images of the evaluation, their folders (one per pair of true and predicted classes) and the limit per pair,
    # as its report block counts them
    error_counts = read_error_counts(evaluation_folder=evaluation_folder)
    facts.update({"evaluation.errors.pair_folders": error_counts["pairs"],
                  "evaluation.errors.per_pair_limit": error_counts["limit"],
                  "evaluation.errors.images": error_counts["images"]})
    canvas.save(fp=output_path)
    return facts


def make_evaluation_figures(output_folder: Path, run_folder: Path, evaluation_folder: Path) -> Dict[str, Any]:
    """
    Draws the sample image and error image figures into the evaluation folder of the series, and gives their boxes.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the classification run of tutorial_sources.yaml.
        evaluation_folder (Path): The folder of its standalone evaluation, classification_evaluation of
            tutorial_sources.yaml.

    Returns:
        Dict[str, Any]: The facts of the figures, named evaluation.*.
    """
    figure_folder = output_folder / "evaluation"
    figure_folder.mkdir(parents=True, exist_ok=True)
    facts = draw_sample_figure(run_folder=run_folder, output_path=figure_folder / (
        f"sample_images_iterations_{'_'.join(str(iteration) for iteration in SAMPLE_ITERATIONS)}.png"))
    facts.update(draw_error_figure(run_folder=run_folder, evaluation_folder=evaluation_folder,
                                   output_path=figure_folder / f"error_images_iteration_{ERROR_ITERATION}.png"))
    return facts
