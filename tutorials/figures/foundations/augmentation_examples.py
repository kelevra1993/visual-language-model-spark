"""
Augmentation examples of the foundations tutorials (the data pipeline chapter), made by
tutorials/figures/foundations/figures.py with the review figures of the repository
(scripts/data/visualize_augmentation_strengths.py): for two validation images of the classification run, resized to its
input as the data loaders give them to the network, one figure per augmentation type of DataAugmenter holds the image,
the type applied alone at the strengths of the supervised tasks and at those of the self-supervised views, and two more
figures explain the aspect ratio of the crop of the views and the levels of the noise. The scenes crop single cells
out of these figures, so the facts give the pixel box of every cell of every figure, and the name and class of the two
images. The figures take a while to draw, so they are reused from the previous build unless forced.
"""

from fractions import Fraction
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List

from scripts.data.visualize_augmentation_strengths import (CELL_GAP, EXPLAINED_ASPECT_RATIOS,
                                                           EXPLAINED_NOISE_LEVELS, TITLE_HEIGHT,
                                                           select_example_images, visualize_augmentation_strengths)
from tutorials.figures.figure_utilities import read_reused_facts, read_run_configuration, write_reused_facts
from vision_training.data.augmentations import AUGMENTATION_TYPES

# Number of example images, one per row of every figure, and the seed of their choice and of the draws
EXAMPLE_IMAGE_COUNT = 2
EXAMPLE_SEED = 0

# Names of the two explanation figures written by visualize_augmentation_strengths next to the figures of the types
CROP_RATIO_FIGURE_NAME = "random_crop_aspect_ratio_explained"
NOISE_LEVELS_FIGURE_NAME = "random_noise_levels_explained"

# Name of the JSON file keeping the facts of the figures for their reuse
FACTS_FILE_NAME = "augmentation_examples_facts.json"


def name_figure_columns(figure_name: str) -> List[str]:
    """
    Names the columns of a figure of visualize_augmentation_strengths in the facts: the image, then the supervised and
    self-supervised strengths for an augmentation type, the aspect ratios of the crop explained (ratio_1_2 for 1/2) or
    the noise levels explained (noise_20 for 20 levels).

    Args:
        figure_name (str): The name of the figure file without its extension.

    Returns:
        List[str]: The name of every column, from the left.

    Raises:
        ValueError: If the figure is neither an augmentation type nor one of the two explanation figures.
    """
    if figure_name in AUGMENTATION_TYPES:
        return ["original", "supervised", "self_supervised"]
    if figure_name == CROP_RATIO_FIGURE_NAME:
        return ["original"] + [f"ratio_{str(Fraction(aspect_ratio).limit_denominator(max_denominator=10))}".replace(
            "/", "_") for aspect_ratio in EXPLAINED_ASPECT_RATIOS]
    if figure_name == NOISE_LEVELS_FIGURE_NAME:
        return ["original"] + [f"noise_{noise_level:g}" for noise_level in EXPLAINED_NOISE_LEVELS]
    raise ValueError(f"Unknown figure {figure_name} written by visualize_augmentation_strengths: name its columns in "
                     f"{Path(__file__).name}.")


def measure_cell_boxes(figure_path: Path, image_size: int) -> Dict[str, List[int]]:
    """
    Gives the pixel box of every cell of a figure of visualize_augmentation_strengths, from the size of the written
    figure and the layout of compose_figure (cells of the input size separated by CELL_GAP under the TITLE_HEIGHT band,
    every row followed by the lines of its captions), so that a scene crops exactly one cell.

    Args:
        figure_path (Path): The figure file.
        image_size (int): The side of the square cells, the input size of the run.

    Returns:
        Dict[str, List[int]]: The boxes, named augmentation_examples.<figure>.image_<row>.<column>_box.

    Raises:
        ValueError: If the size of the figure does not follow the layout of compose_figure.
    """
    column_names = name_figure_columns(figure_name=figure_path.stem)
    with Image.open(fp=figure_path) as figure:
        figure_width, figure_height = figure.size

    # Width of a cell and height of a row with its captions, both whole numbers of pixels in the layout
    cell_width, width_remainder = divmod(figure_width - (len(column_names) + 1) * CELL_GAP, len(column_names))
    row_height, height_remainder = divmod(figure_height - TITLE_HEIGHT, EXAMPLE_IMAGE_COUNT)
    if width_remainder or height_remainder or cell_width != image_size or row_height < image_size:
        raise ValueError(f"The figure {figure_path} of {figure_width}x{figure_height} pixels does not hold "
                         f"{EXAMPLE_IMAGE_COUNT} rows of {len(column_names)} cells of {image_size} pixels: the layout "
                         f"of compose_figure changed.")

    # Every cell of every row, from the top left corner of the figure
    boxes = {}
    for row_index in range(EXAMPLE_IMAGE_COUNT):
        top = TITLE_HEIGHT + row_index * row_height
        for column_index, column_name in enumerate(column_names):
            left = CELL_GAP + column_index * (cell_width + CELL_GAP)
            boxes[f"augmentation_examples.{figure_path.stem}.image_{row_index + 1}.{column_name}_box"] = [
                left, top, left + cell_width, top + image_size]
    return boxes


def make_augmentation_examples(output_folder: Path, run_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Writes the augmentation figures of the repository for validation images of the classification run, resized to its
    input with the image settings and the augmentation section of its saved configuration, into the
    augmentation_examples folder of the series, and gives the boxes of their cells and the example images; the figures
    of the previous build are reused unless forced or incomplete.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the classification run of tutorial_sources.yaml.
        force (bool): Whether the figures are drawn again even when the previous build left them.

    Returns:
        Dict[str, Any]: The facts of the figures, named augmentation_examples.*.
    """
    # Figures of the previous build, complete for every augmentation type of the repository
    examples_folder = output_folder / "augmentation_examples"
    facts_path = examples_folder / FACTS_FILE_NAME
    reused_facts = read_reused_facts(facts_path=facts_path, force=force, required_paths=[
        examples_folder / f"{augmentation_type}.png" for augmentation_type in AUGMENTATION_TYPES])
    if reused_facts is not None:
        return reused_facts

    # Every figure drawn by the repository from the validation split and the configuration of the run
    run_configuration = read_run_configuration(run_folder=run_folder)
    split_folder = Path(run_configuration["ExperimentConfiguration"]["validation_data_folder"])
    written_paths = visualize_augmentation_strengths(configuration_path=run_folder / "training_configuration.yaml",
                                                     split_folder=split_folder, output_folder=examples_folder,
                                                     number_images=EXAMPLE_IMAGE_COUNT, seed=EXAMPLE_SEED)

    # Boxes of the cells of every figure, then the name and class of the example images, chosen again with the seed
    facts = {}
    for figure_path in written_paths:
        facts.update(measure_cell_boxes(figure_path=figure_path,
                                        image_size=run_configuration["Data"]["image_settings"]["size"]))
    examples = select_example_images(split_folder=split_folder, number_images=EXAMPLE_IMAGE_COUNT, seed=EXAMPLE_SEED)
    for example_index, example in enumerate(examples):
        facts[f"augmentation_examples.image_{example_index + 1}.name"] = example["truck"]
        facts[f"augmentation_examples.image_{example_index + 1}.category"] = example["category"]
    write_reused_facts(facts_path=facts_path, figure_paths=written_paths, facts=facts)
    return facts
