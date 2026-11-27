"""
When run, writes the figures showing every augmentation type of DataAugmenter (vision_training/data/augmentations.py)
at its two hard-coded strengths, to review them by eye before a training: for a few images of a split, read and resized
as the data loaders give them to the network, one figure per type holds the image, the image with the type applied alone
at the strengths of the supervised tasks (SUPERVISED_AUGMENTATION_STRENGTHS), then at those of the views of the
self-supervised task (SELF_SUPERVISED_AUGMENTATION_STRENGTHS), the values drawn for each written under it. Two more
figures explain the aspect ratio of the random resized crop of the views and the levels of the noise. Every figure is
written as '<type>.png' in the output folder, the explanations as '<type>_..._explained.png' next to their type. The
split and the output folder are required arguments, the configuration, the number of images and the seed optional
(DEFAULT_CONFIGURATION_PATH, DEFAULT_EXAMPLE_SETTINGS), the help being printed when no argument is given.
"""

import argparse
import math
import random
import sys

import cv2
import numpy as np
import torch

from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vision_training.configuration_processing.configuration_loader import load_configuration
from vision_training.data.augmentations import (AUGMENTATION_TYPES, GEOMETRIC_AUGMENTATION_TYPES,
                                                SECOND_VIEW_ONLY_TYPES, SELF_SUPERVISED_AUGMENTATION_STRENGTHS,
                                                SUPERVISED_AUGMENTATION_STRENGTHS, DataAugmenter)
from vision_training.data.preprocessing import preprocess_image, read_image
from vision_training.utilities.os_utilities import format_path_link, print_color, read_json

# Layout of the figures in pixels: the gap around the cells, the band of the titles and the lines of the captions
CELL_GAP = 6
TITLE_HEIGHT = 52
CAPTION_LINE_HEIGHT = 14

# Text of the figures, dark on a white background
FONT = cv2.FONT_HERSHEY_SIMPLEX
TITLE_FONT_SCALE = 0.55
CAPTION_FONT_SCALE = 0.37
TEXT_COLOUR = (30, 30, 30)
BACKGROUND_VALUE = 255

# Crop of the views explained: the share of the image area kept, and the aspect ratios (width / height) of the regions,
# the lower bound of crop_aspect_ratio_range, a square and its upper bound, each region framed on the image in its
# colour (BGR)
EXPLAINED_CROP_AREA = 0.5
EXPLAINED_ASPECT_RATIOS = [SELF_SUPERVISED_AUGMENTATION_STRENGTHS["crop_aspect_ratio_range"][0], 1.0,
                           SELF_SUPERVISED_AUGMENTATION_STRENGTHS["crop_aspect_ratio_range"][1]]
REGION_COLOURS = [(0, 0, 230), (0, 170, 0), (230, 0, 0)]

# Noise levels explained, standard deviations in levels of the [0, 255] images, the columns of the levels of the
# supervised tasks and of the views being named after them (name_noise_level)
EXPLAINED_NOISE_LEVELS = [10.0, 20.0, 30.0, 40.0]

# Configuration giving the input size, the padding and the black-out grid by default, the synthetic shapes classifier
# of the repository, and the number of example images (one per row of every figure) and the seed of their choice and of
# the draws by default
DEFAULT_CONFIGURATION_PATH = (Path(__file__).absolute().parents[2] / "vision_training" / "configurations" /
                              "classification" / "configuration_synthetic_shapes.yaml")
DEFAULT_EXAMPLE_SETTINGS = {"number_images": 4, "seed": 0}


def select_example_images(split_folder: Path, number_images: int, seed: int) -> List[Dict[str, Any]]:
    """
    Chooses the example images of the figures among the labelled images of a split ('labels.json' of the
    self-supervised and classification datasets, whose samples hold 'file_name', 'category_id' and, in some layouts, the
    'truck' of their 'groups', the source the image comes from), so that they show different sources and categories:
    the samples are shuffled with the seed, then taken in that order when their truck and their category are both new,
    the samples of a new truck only completing the choice when the split holds fewer categories than images. A sample
    without groups counts as its own truck.

    Args:
        split_folder (Path): The split folder holding 'data/' and 'labels.json'.
        number_images (int): The number of images chosen.
        seed (int): The seed of the shuffling, the same seed giving the same images.

    Returns:
        List[Dict[str, Any]]: The 'path', 'truck' and 'category' of every chosen image.

    Raises:
        ValueError: If the split holds fewer labelled images of different trucks than asked for.
    """
    manifest = read_json(path=str(split_folder / "labels.json"))
    category_names = {category["id"]: category["name"] for category in manifest["categories"]}
    labelled_samples = [sample for sample in manifest["samples"] if sample.get("category_id") in category_names]
    shuffled_samples = random.Random(seed).sample(population=labelled_samples, k=len(labelled_samples))

    # First the samples bringing a new truck and a new category, then those bringing a new truck only
    chosen_samples, chosen_trucks, chosen_categories = [], set(), set()
    for requires_new_category in [True, False]:
        for sample in shuffled_samples:
            truck = sample.get("groups", {}).get("truck", sample["file_name"])
            if len(chosen_samples) == number_images:
                break
            if truck in chosen_trucks or (requires_new_category and sample["category_id"] in chosen_categories):
                continue
            chosen_samples.append(sample)
            chosen_trucks.add(truck)
            chosen_categories.add(sample["category_id"])

    # Fail loudly instead of drawing fewer rows than asked for
    if len(chosen_samples) < number_images:
        raise ValueError(f"The split {format_path_link(path=split_folder)} holds labelled images of "
                         f"{len(chosen_samples)} different trucks, fewer than the {number_images} images asked for")
    return [{"path": split_folder / "data" / sample["file_name"],
             "truck": sample.get("groups", {}).get("truck", Path(sample["file_name"]).stem),
             "category": category_names[sample["category_id"]]} for sample in chosen_samples]


def load_example_images(image_paths: List[Path], image_size: int, keep_ratio: bool) -> torch.Tensor:
    """
    Reads the example images and resizes them as the data loaders do (read_image and preprocess_image of
    vision_training/data/preprocessing.py, as ClassificationDataset), so that the figures show the images the network
    receives, padded with black to a square when the aspect ratio is kept.

    Args:
        image_paths (List[Path]): The image files.
        image_size (int): The side of the square network input, 'Data.image_settings.size'.
        keep_ratio (bool): Whether the aspect ratio is kept by padding, 'Data.image_settings.keep_ratio'.

    Returns:
        torch.Tensor: The [N, 3, H, W] float32 BGR images in [0, 255].
    """
    resized_images = [preprocess_image(image=read_image(image_path=image_path), image_size=image_size,
                                       keep_ratio=keep_ratio) for image_path in image_paths]
    return torch.from_numpy(np.ascontiguousarray(a=np.stack(arrays=resized_images))).permute(
        dims=(0, 3, 1, 2)).to(dtype=torch.float32)


def apply_type_alone(augmenter: DataAugmenter, augmentation_type: str, image: torch.Tensor,
                     seed: int) -> Tuple[torch.Tensor, str]:
    """
    Applies one augmentation type alone to an image with the strengths of the augmenter, whatever its flag and
    probability in the configuration, by calling the method of the type as augment_sample does, after seeding the
    generators of Python (the parameters of the types) and of PyTorch (the noise and the plasma map), so that both
    strengths of a row of a figure draw the same random numbers.

    Args:
        augmenter (DataAugmenter): The augmenter of the supervised tasks or of the views.
        augmentation_type (str): The name of the type, one of AUGMENTATION_TYPES.
        image (torch.Tensor): The [3, H, W] float BGR image in [0, 255].
        seed (int): The seed of both generators.

    Returns:
        Tuple[torch.Tensor, str]: The augmented image in [0, 255], and the values drawn for it as written in the
            reports of the input viewer.
    """
    random.seed(seed)
    torch.manual_seed(seed=seed)

    # Geometric types take the targets, none here, and give the area scale of the objects with the values drawn
    augmentation_method = getattr(augmenter, augmentation_type)
    if augmentation_type in GEOMETRIC_AUGMENTATION_TYPES:
        augmented_image, _, _, _, drawn_values = augmentation_method(image=image, bounding_boxes=None, mask=None)
    else:
        augmented_image, drawn_values = augmentation_method(image=image)
    return augmented_image.clamp(min=0.0, max=255.0), drawn_values or "applied, no value drawn"


def convert_to_cell(image: torch.Tensor) -> np.ndarray:
    """
    Converts an image tensor into the cell of a figure, rounded to whole levels as an image file holds them.

    Args:
        image (torch.Tensor): The [3, H, W] float BGR image in [0, 255].

    Returns:
        np.ndarray: The contiguous [H, W, 3] uint8 BGR cell.
    """
    return np.ascontiguousarray(a=image.clamp(min=0.0, max=255.0).round().to(dtype=torch.uint8).permute(
        dims=(1, 2, 0)).numpy())


def wrap_caption(text: str, maximum_width: int) -> List[str]:
    """
    Splits a caption into the lines that fit the width of a cell of a figure, word by word, measured with the font of
    the captions, a word wider than the cell keeping a line of its own.

    Args:
        text (str): The caption.
        maximum_width (int): The width of the cell in pixels.

    Returns:
        List[str]: The lines of the caption.
    """
    lines, current_line = [], ""
    for word in text.split():
        candidate_line = f"{current_line} {word}".strip()
        candidate_width = cv2.getTextSize(text=candidate_line, fontFace=FONT, fontScale=CAPTION_FONT_SCALE,
                                          thickness=1)[0][0]
        if candidate_width <= maximum_width or not current_line:
            current_line = candidate_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines


def compose_figure(title: str, column_titles: List[str], cells: List[List[np.ndarray]], captions: List[List[str]],
                   column_title_colours: Optional[List[Tuple[int, int, int]]] = None) -> np.ndarray:
    """
    Lays out the cells of a figure on a white background: the title and the title of every column in the top band,
    then one row per example image, every cell above its caption, the captions taking as many lines as the longest
    of them.

    Args:
        title (str): The title of the figure, written top left.
        column_titles (List[str]): The title of every column.
        cells (List[List[np.ndarray]]): The [H, W, 3] uint8 BGR cells of every row, one per column, all of one size.
        captions (List[List[str]]): The caption of every cell.
        column_title_colours (Optional[List[Tuple[int, int, int]]]): The BGR colour of every column title, or None for
            the colour of the text. Defaults to None.

    Returns:
        np.ndarray: The [height, width, 3] uint8 BGR figure.
    """
    # Lines of every caption, the longest giving the height of the caption band of every row
    cell_height, cell_width = cells[0][0].shape[:2]
    caption_lines = [[wrap_caption(text=caption, maximum_width=cell_width) for caption in row_captions]
                     for row_captions in captions]
    number_caption_lines = max(len(lines) for row_lines in caption_lines for lines in row_lines)
    number_columns = len(column_titles)
    row_height = cell_height + number_caption_lines * CAPTION_LINE_HEIGHT + CELL_GAP
    figure_width = number_columns * cell_width + (number_columns + 1) * CELL_GAP
    figure_height = TITLE_HEIGHT + len(cells) * row_height
    figure = np.full(shape=(figure_height, figure_width, 3), fill_value=BACKGROUND_VALUE, dtype=np.uint8)

    # Title of the figure, then the title of every column above its cells
    cv2.putText(img=figure, text=title, org=(CELL_GAP, 20), fontFace=FONT, fontScale=TITLE_FONT_SCALE,
                color=TEXT_COLOUR, thickness=1, lineType=cv2.LINE_AA)
    for column_index, column_title in enumerate(column_titles):
        title_colour = column_title_colours[column_index] if column_title_colours else TEXT_COLOUR
        cv2.putText(img=figure, text=column_title, org=(CELL_GAP + column_index * (cell_width + CELL_GAP),
                                                        TITLE_HEIGHT - 10),
                    fontFace=FONT, fontScale=TITLE_FONT_SCALE, color=title_colour, thickness=1, lineType=cv2.LINE_AA)

    # Every cell with its caption under it
    for row_index, (row_cells, row_lines) in enumerate(zip(cells, caption_lines)):
        top = TITLE_HEIGHT + row_index * row_height
        for column_index, (cell, lines) in enumerate(zip(row_cells, row_lines)):
            left = CELL_GAP + column_index * (cell_width + CELL_GAP)
            figure[top:top + cell_height, left:left + cell_width] = cell
            for line_index, line in enumerate(lines):
                line_bottom = top + cell_height + (line_index + 1) * CAPTION_LINE_HEIGHT
                cv2.putText(img=figure, text=line, org=(left, line_bottom), fontFace=FONT, fontScale=CAPTION_FONT_SCALE,
                            color=TEXT_COLOUR, thickness=1, lineType=cv2.LINE_AA)
    return figure


def build_type_figure(images: torch.Tensor, image_descriptions: List[str],
                      augmenters: Tuple[DataAugmenter, DataAugmenter], augmentation_type: str, seed: int) -> np.ndarray:
    """
    Builds the figure of one augmentation type: every example image in a row, as the network receives it, then with
    the type applied alone at the supervised strengths, then at the self-supervised ones, the values drawn for each
    written under it. Both strengths of a row draw the same random numbers (apply_type_alone with the seed of the row),
    so that the two columns differ by the strengths only, e.g. the same rotation scaled from 5 to 10 degrees at most.

    Args:
        images (torch.Tensor): The [N, 3, H, W] float BGR example images in [0, 255].
        image_descriptions (List[str]): The caption of every example image.
        augmenters (Tuple[DataAugmenter, DataAugmenter]): The augmenter of the supervised tasks and the one of the
            views.
        augmentation_type (str): The name of the type, one of AUGMENTATION_TYPES.
        seed (int): The seed of the first row, the next rows taking the following seeds.

    Returns:
        np.ndarray: The [height, width, 3] uint8 BGR figure.
    """
    cells, captions = [], []
    for image_index, (image, image_description) in enumerate(zip(images, image_descriptions)):
        # The image, then the type at both strengths with the same random numbers
        row_cells, row_captions = [convert_to_cell(image=image)], [image_description]
        for augmenter in augmenters:
            augmented_image, drawn_values = apply_type_alone(augmenter=augmenter, augmentation_type=augmentation_type,
                                                             image=image, seed=seed + image_index)
            row_cells.append(convert_to_cell(image=augmented_image))
            row_captions.append(drawn_values)
        cells.append(row_cells)
        captions.append(row_captions)

    # The views draw some types for their second view only, said in the title of the column
    self_supervised_title = ("self-supervised (second view)" if augmentation_type in SECOND_VIEW_ONLY_TYPES else
                             "self-supervised")
    return compose_figure(title=f"{augmentation_type}, applied alone", cells=cells, captions=captions,
                          column_titles=["original", "supervised", self_supervised_title])


def build_crop_ratio_figure(images: torch.Tensor, image_descriptions: List[str],
                            augmenter: DataAugmenter) -> np.ndarray:
    """
    Builds the figure explaining the aspect ratio of the random resized crop of the views (random_crop with the
    self-supervised strengths): for every example image, the central regions covering EXPLAINED_CROP_AREA of its area
    at the width / height ratios of EXPLAINED_ASPECT_RATIOS are framed on the image, then each region is resized back to
    the input size by crop_and_resize, as random_crop does, its sides computed as draw_area_crop computes them. A region
    wider than tall (a ratio above 1) is stretched more vertically than horizontally, so the objects look taller and
    thinner, a region taller than wide (below 1) the opposite, and a square region (1) is only zoomed.

    Args:
        images (torch.Tensor): The [N, 3, H, W] float BGR example images in [0, 255].
        image_descriptions (List[str]): The caption of every example image.
        augmenter (DataAugmenter): The augmenter whose crop_and_resize resizes the regions.

    Returns:
        np.ndarray: The [height, width, 3] uint8 BGR figure.
    """
    cells, captions = [], []
    for image, image_description in zip(images, image_descriptions):
        image_height, image_width = image.shape[-2], image.shape[-1]
        framed_cell = convert_to_cell(image=image)
        row_cells, row_captions = [framed_cell], [f"{image_description}, regions of "
                                                  f"{EXPLAINED_CROP_AREA * 100:.0f} % of the area"]
        for aspect_ratio, region_colour in zip(EXPLAINED_ASPECT_RATIOS, REGION_COLOURS):
            # Sides of the region of the area and the ratio, as in draw_area_crop, centred on the image
            target_area = image_height * image_width * EXPLAINED_CROP_AREA
            crop_width = round(math.sqrt(target_area * aspect_ratio))
            crop_height = round(math.sqrt(target_area / aspect_ratio))
            top, left = (image_height - crop_height) // 2, (image_width - crop_width) // 2

            # Frame of the region on the image, and the region resized back to the input size
            cv2.rectangle(img=framed_cell, pt1=(left, top), pt2=(left + crop_width - 1, top + crop_height - 1),
                          color=region_colour, thickness=2)
            cropped_image, _, _ = augmenter.crop_and_resize(image=image, bounding_boxes=None, mask=None, top=top,
                                                            left=left, crop_height=crop_height, crop_width=crop_width)
            row_cells.append(convert_to_cell(image=cropped_image))
            row_captions.append(f"{crop_width}x{crop_height} pixels resized to {image_width}x{image_height}: "
                                f"stretched x{image_width / crop_width:.2f} across, x{image_height / crop_height:.2f} "
                                f"down")
        cells.append(row_cells)
        captions.append(row_captions)

    # Columns named by their ratio, in the colour of their frame
    column_titles = ["original"] + [f"ratio {Fraction(aspect_ratio).limit_denominator(max_denominator=10)}"
                                    for aspect_ratio in EXPLAINED_ASPECT_RATIOS]
    return compose_figure(title="random_crop of the views: aspect ratio (width / height) of the region",
                          column_titles=column_titles, cells=cells, captions=captions,
                          column_title_colours=[TEXT_COLOUR] + REGION_COLOURS)


def name_noise_level(noise_level: float) -> str:
    """
    Names a column of the figure of the noise levels by its standard deviation, followed by the strength sets using it,
    so that the levels of the supervised tasks and of the views are found at a glance in the ladder of levels.

    Args:
        noise_level (float): The standard deviation of the noise of the column, in levels.

    Returns:
        str: The title of the column, e.g. '20 levels (supervised)'.
    """
    strength_sets = [("supervised", SUPERVISED_AUGMENTATION_STRENGTHS),
                     ("views", SELF_SUPERVISED_AUGMENTATION_STRENGTHS)]
    users = [name for name, strengths in strength_sets if strengths["noise_standard_deviation"] == noise_level]
    return f"{noise_level:g} levels" + (f" ({', '.join(users)})" if users else "")


def build_noise_levels_figure(images: torch.Tensor, image_descriptions: List[str],
                              augmentation_configuration: Dict[str, Any], seed: int) -> np.ndarray:
    """
    Builds the figure explaining the levels of the noise: every example image, then with the Gaussian noise of
    random_noise at the standard deviations of EXPLAINED_NOISE_LEVELS. A level is one of the 256 intensities of a
    channel of a pixel, from 0 (black) to 255 (white), and a standard deviation of 40 levels moves two values in three
    by less than 40 levels up or down, 16 % of the range, and one in twenty by more than 80. Every column draws the
    same random numbers, so that its noise is the one of the previous column scaled.

    Args:
        images (torch.Tensor): The [N, 3, H, W] float BGR example images in [0, 255].
        image_descriptions (List[str]): The caption of every example image.
        augmentation_configuration (Dict[str, Any]): The 'Data.data_augmentation' section the augmenter of the figure
            is built from.
        seed (int): The seed of the first row, the next rows taking the following seeds.

    Returns:
        np.ndarray: The [height, width, 3] uint8 BGR figure.
    """
    # An augmenter of the figure only, whose standard deviation of the noise is set for every column
    noise_augmenter = DataAugmenter(augmentation_configuration=augmentation_configuration)
    cells, captions = [], []
    for image_index, (image, image_description) in enumerate(zip(images, image_descriptions)):
        row_cells, row_captions = [convert_to_cell(image=image)], [image_description]
        for noise_level in EXPLAINED_NOISE_LEVELS:
            noise_augmenter.strengths = {**SUPERVISED_AUGMENTATION_STRENGTHS, "noise_standard_deviation": noise_level}
            noisy_image, _ = apply_type_alone(augmenter=noise_augmenter, augmentation_type="random_noise", image=image,
                                              seed=seed + image_index)
            row_cells.append(convert_to_cell(image=noisy_image))
            row_captions.append(f"standard deviation {noise_level:g} levels, {noise_level / 255.0 * 100:.0f} % of "
                                f"the 255 levels")
        cells.append(row_cells)
        captions.append(row_captions)
    column_titles = ["original"] + [name_noise_level(noise_level=noise_level) for noise_level in EXPLAINED_NOISE_LEVELS]
    return compose_figure(title="random_noise: standard deviation of the noise, in levels of [0, 255]",
                          column_titles=column_titles, cells=cells, captions=captions)


def visualize_augmentation_strengths(configuration_path: Path, split_folder: Path, output_folder: Path,
                                     number_images: int, seed: int) -> List[Path]:
    """
    Writes the figure of every augmentation type and the two explanation figures for example images of a split, the
    images resized to the input of a configuration, whose 'Data.data_augmentation' section also gives the grid size of
    random_black_out, the strengths being the hard-coded ones. Every type is applied alone whatever its flag in the
    configuration, so that the figures show the whole catalogue.

    Args:
        configuration_path (Path): The configuration file giving 'Data.image_settings' and 'Data.data_augmentation'.
        split_folder (Path): The split the example images are chosen in.
        output_folder (Path): The folder of the figures, created if missing, the figures of a previous run being
            replaced.
        number_images (int): The number of example images, one per row of every figure.
        seed (int): The seed of the choice of the images and of the draws.

    Returns:
        List[Path]: The written figures, the types in the order of AUGMENTATION_TYPES, then the explanations.

    Raises:
        OSError: If a figure cannot be written.
    """
    # Input size and padding of the network, and the augmenters of both strengths
    model_configuration = load_configuration(configuration_path=configuration_path)
    image_settings = model_configuration["Data"]["image_settings"]
    augmentation_configuration = model_configuration["Data"]["data_augmentation"]
    augmenters = (DataAugmenter(augmentation_configuration=augmentation_configuration, self_supervised=False),
                  DataAugmenter(augmentation_configuration=augmentation_configuration, self_supervised=True))

    # Example images of different trucks and streams, as the network receives them
    examples = select_example_images(split_folder=split_folder, number_images=number_images, seed=seed)
    images = load_example_images(image_paths=[example["path"] for example in examples],
                                 image_size=image_settings["size"], keep_ratio=image_settings["keep_ratio"])
    image_descriptions = [f"{example['truck']}, {example['category']}" for example in examples]

    # One figure per type, then the explanations of the crop of the views and of the noise levels
    figures = {f"{augmentation_type}.png": build_type_figure(images=images, image_descriptions=image_descriptions,
                                                             augmenters=augmenters,
                                                             augmentation_type=augmentation_type, seed=seed)
               for augmentation_type in AUGMENTATION_TYPES}
    figures["random_crop_aspect_ratio_explained.png"] = build_crop_ratio_figure(
        images=images, image_descriptions=image_descriptions, augmenter=augmenters[1])
    figures["random_noise_levels_explained.png"] = build_noise_levels_figure(
        images=images, image_descriptions=image_descriptions, augmentation_configuration=augmentation_configuration,
        seed=seed)

    # Every figure written as a lossless image, so that the noise is shown as drawn
    output_folder.mkdir(parents=True, exist_ok=True)
    written_paths = []
    for file_name, figure in figures.items():
        figure_path = output_folder / file_name
        if not cv2.imwrite(filename=str(figure_path), img=figure):
            raise OSError(f"The figure {figure_path} could not be written")
        written_paths.append(figure_path)
    return written_paths


def parse_augmentation_strengths_arguments(arguments: List[str]) -> argparse.Namespace:
    """
    Reads the settings of the augmentation figures from the command line of the script, so that the figures of a split
    are written without editing the file: the split and the output folder are required, the configuration defaults to
    DEFAULT_CONFIGURATION_PATH and the number of images and the seed to DEFAULT_EXAMPLE_SETTINGS. Launched without any
    argument, the script prints the help of every argument and stops. A '~' left by the shell at the start of a path
    ('--split_folder=~/Data/...') is expanded to the home folder.

    Args:
        arguments (List[str]): The command line arguments, without the script name.

    Returns:
        argparse.Namespace: The arguments of visualize_augmentation_strengths.

    Raises:
        SystemExit: After printing the help when no argument is given, or the usage and the error when a folder is
            missing or a number is not a whole number.
    """
    # The description and the example keep their line breaks, argparse wrapping the help of every argument
    parser = argparse.ArgumentParser(
        prog="uv run python -m scripts.data.visualize_augmentation_strengths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Writes one figure per augmentation type, a few images of a split with the type applied alone at\n"
                    "the supervised and the self-supervised strengths, and the figures explaining the crop aspect\n"
                    "ratios and the noise levels.",
        epilog="Example, the validation split of the synthetic shapes:\n"
               "  uv run python -m scripts.data.visualize_augmentation_strengths \\\n"
               "      --split_folder ~/Data/classifier_data/synthetic_shapes/validation \\\n"
               "      --output_folder ~/Data/augmentation_examples/synthetic_shapes")
    # The split the example images are chosen in, the folder of the figures and the configuration of the input
    parser.add_argument("--split_folder", type=Path, required=True,
                        help="Split the example images are chosen in, holding 'data/' and a 'labels.json' of "
                             "samples with a category (required).")
    parser.add_argument("--output_folder", type=Path, required=True,
                        help="Folder of the figures, created if missing, the figures of a previous run being replaced "
                             "(required).")
    parser.add_argument("--configuration_path", type=Path, default=DEFAULT_CONFIGURATION_PATH,
                        help="Configuration giving Data.image_settings and Data.data_augmentation (default: the "
                             "synthetic shapes classifier of the repository, "
                             "classification/configuration_synthetic_shapes.yaml of vision_training/configurations).")
    # The number of example images and the seed of their choice and of the draws
    parser.add_argument("--number_images", type=int, default=DEFAULT_EXAMPLE_SETTINGS["number_images"],
                        help="Number of example images, one per row of every figure (default: %(default)s).")
    parser.add_argument("--seed", type=int, default=DEFAULT_EXAMPLE_SETTINGS["seed"],
                        help="Seed of the choice of the images and of the draws (default: %(default)s).")

    # Without any argument, the script describes its arguments instead of failing on the missing folders
    if not arguments:
        parser.print_help()
        parser.exit(status=0)
    parsed_arguments = parser.parse_args(args=arguments)

    # A '~' the shell leaves after '=' points to the home folder, as in the other scripts of the repository
    for path_name in ["split_folder", "output_folder", "configuration_path"]:
        setattr(parsed_arguments, path_name, getattr(parsed_arguments, path_name).expanduser())
    return parsed_arguments


if __name__ == "__main__":
    # Split, output folder, configuration, number of images and seed read from the command line, the help printed when
    # none is given
    figure_arguments = parse_augmentation_strengths_arguments(arguments=sys.argv[1:])

    written_paths = visualize_augmentation_strengths(configuration_path=figure_arguments.configuration_path,
                                                     split_folder=figure_arguments.split_folder,
                                                     output_folder=figure_arguments.output_folder,
                                                     number_images=figure_arguments.number_images,
                                                     seed=figure_arguments.seed)
    print_color(color="green", output=f"{len(written_paths)} figures written to "
                                      f"{format_path_link(path=figure_arguments.output_folder)}")
