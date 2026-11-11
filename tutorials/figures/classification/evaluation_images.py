"""
Evaluation figures of the classification tutorials (the chapter reading a classification evaluation), made at every
build by tutorials/figures/classification/figures.py from the evaluation folders of the classification run: the eight
fixed sample images of the test evaluations of two iterations side by side, each grid under its iteration, and one
error image of four confused pairs of classes from the standalone evaluation of the run (classification_evaluation in
tutorials/tutorial_sources.yaml), each under its folder and file name, so that the scenes show the real images the
repository writes. The images are only placed on the card colour of the videos, never redrawn; the box of every image
and grid is a fact, for the views and highlights of the scenes, with the probability of every shown error and the
number of error folders and images.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Any, Dict, List, Tuple

from tutorial_builder.layout import (CARD_COLOUR, DIM_TEXT_COLOUR, MONOSPACE_FONT_PATH, SANS_BOLD_FONT_PATH,
                                     TEXT_COLOUR)

# Iterations whose sample images are compared, the first and the last test evaluations of the run
SAMPLE_ITERATIONS = [500, 1500]

# Iteration of the standalone evaluation and the error image shown for every confused pair of classes, in its
# Errors/<true class>-TO-<predicted class>/ folder, the file name ending with the probability of the wrong class
ERROR_ITERATION = 1500
ERROR_IMAGES = [("square-TO-rectangle", "test_003231_0.688.png"), ("pentagon-TO-hexagon", "test_001219_0.542.png"),
                ("circle-TO-octagon", "test_010208_0.688.png"), ("octagon-TO-hexagon", "test_004210_0.613.png")]

# Layout: four columns of images, the gap between two images and between two grids, the height of the header band
GRID_COLUMN_COUNT = 4
IMAGE_GAP = 16
GRID_GAP = 800
HEADER_HEIGHT = 90

# Sizes of the fonts of the videos: Inter for the iteration headings, Hack for the folder and file names
HEADING_FONT_SIZE = 56
LABEL_FONT_SIZE = 36


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


def draw_sample_figure(run_folder: Path, output_path: Path) -> Dict[str, Any]:
    """
    Draws the sample images of the test evaluations of SAMPLE_ITERATIONS side by side, the same eight test images
    captioned by the evaluation with their true class, predicted class and probability, and gives the box of the
    figure, of every grid and of every image.

    Args:
        run_folder (Path): The folder of the classification run.
        output_path (Path): The PNG file of the figure.

    Returns:
        Dict[str, Any]: The boxes, named evaluation.samples.*.

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
             "evaluation.samples.image_count": len(sample_names[0])}
    for grid_index, (iteration, images) in enumerate(zip(SAMPLE_ITERATIONS, sample_grids)):
        grid_left = grid_index * (grid_width + GRID_GAP)
        image_boxes = paste_grid(canvas=canvas, images=images, left=grid_left, heading=f"Iteration {iteration}",
                                 font=heading_font)
        facts[f"evaluation.samples.iteration_{iteration}.grid_box"] = [grid_left, 0, grid_left + grid_width,
                                                                       grid_height]
        for image_index, image_box in enumerate(image_boxes):
            facts[f"evaluation.samples.iteration_{iteration}.image_{image_index + 1}_box"] = image_box
    canvas.save(fp=output_path)
    return facts


def draw_error_figure(evaluation_folder: Path, output_path: Path) -> Dict[str, Any]:
    """
    Draws one error image of every pair of ERROR_IMAGES in a row, each under the name of its folder and file, and
    gives the box of the figure and of every image with its two names.

    Args:
        evaluation_folder (Path): The folder of the standalone evaluation of the classification run.
        output_path (Path): The PNG file of the figure.

    Returns:
        Dict[str, Any]: The boxes, named evaluation.errors.*.
    """
    errors_folder = evaluation_folder / "Evaluation" / f"Iteration_{ERROR_ITERATION}" / "Errors"
    images = load_images(image_paths=[errors_folder / folder / file_name for folder, file_name in ERROR_IMAGES])
    image_width, image_height = images[0].size
    canvas = Image.new(mode="RGB", size=(len(images) * image_width + (len(images) - 1) * IMAGE_GAP * 2,
                                         2 * HEADER_HEIGHT + image_height), color=CARD_COLOUR)
    draw = ImageDraw.Draw(im=canvas)

    # Every image under its folder name and its file name, ending with the probability of the wrong class
    label_font = ImageFont.truetype(font=str(MONOSPACE_FONT_PATH), size=LABEL_FONT_SIZE)
    facts = {"evaluation.errors.figure_box": [0, 0, canvas.width, canvas.height]}
    for image_index, ((folder, file_name), image) in enumerate(zip(ERROR_IMAGES, images)):
        image_left = image_index * (image_width + IMAGE_GAP * 2)
        draw.text(xy=(image_left, HEADER_HEIGHT // 2), text=f"{folder}/", fill=TEXT_COLOUR, font=label_font,
                  anchor="lm")
        draw.text(xy=(image_left, HEADER_HEIGHT + HEADER_HEIGHT // 2), text=file_name, fill=DIM_TEXT_COLOUR,
                  font=label_font, anchor="lm")
        canvas.paste(im=image, box=(image_left, 2 * HEADER_HEIGHT))
        facts[f"evaluation.errors.image_{image_index + 1}_box"] = [image_left, 0, image_left + image_width,
                                                                   2 * HEADER_HEIGHT + image_height]
        facts[f"evaluation.errors.image_{image_index + 1}_probability"] = float(Path(file_name).stem.split("_")[-1])

    # Error folders of the evaluation, one per pair of true and predicted classes, and their largest image count
    pair_image_counts = [len(list(pair_folder.glob("*.png"))) for pair_folder in errors_folder.iterdir()
                         if pair_folder.is_dir()]
    facts.update({"evaluation.errors.pair_folders": len(pair_image_counts),
                  "evaluation.errors.largest_pair_images": max(pair_image_counts),
                  "evaluation.errors.images": sum(pair_image_counts)})
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
    facts.update(draw_error_figure(evaluation_folder=evaluation_folder,
                                   output_path=figure_folder / f"error_images_iteration_{ERROR_ITERATION}.png"))
    return facts
