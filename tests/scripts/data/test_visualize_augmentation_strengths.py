"""
Tests of scripts/data/visualize_augmentation_strengths.py, the figures showing every augmentation type of DataAugmenter
alone at the supervised and the self-supervised strengths: the choice of example images of different trucks and
categories, the type applied alone with the same random numbers for both strengths, the layout of the figures, the
regions of the figure of the crop aspect ratio, the noise of the figure of the noise levels and the names of its
columns, and the files written for every type.
"""

import json
import re

import cv2
import numpy as np
import pytest
import torch
import yaml

from pathlib import Path
from typing import Any, Dict, List, Tuple

from scripts.data.visualize_augmentation_strengths import (CELL_GAP, EXPLAINED_CROP_AREA, EXPLAINED_NOISE_LEVELS,
                                                           REGION_COLOURS, TITLE_HEIGHT, apply_type_alone,
                                                           build_crop_ratio_figure, build_noise_levels_figure,
                                                           build_type_figure, convert_to_cell, name_noise_level,
                                                           select_example_images, visualize_augmentation_strengths)
from vision_training.configuration_processing.configuration_loader import load_configuration
from vision_training.data.augmentations import (AUGMENTATION_TYPES, SELF_SUPERVISED_AUGMENTATION_STRENGTHS,
                                                SUPERVISED_AUGMENTATION_STRENGTHS, DataAugmenter)

# Self-supervised configuration whose data augmentation section builds the augmenters of the tests
CONFIGURATION_PATH = (Path(__file__).parents[3] / "vision_training" / "configurations" / "self_supervised" /
                      "configuration_synthetic_shapes.yaml")

# Side of the test images, large enough for the captions of the crop on two lines
IMAGE_SIZE = 64


def write_fake_split(split_folder: Path, samples: List[Dict[str, Any]]) -> None:
    """
    Writes a split in the format of the ficha self-supervised layouts, a 'labels.json' of two categories whose samples
    hold their truck in their groups, with a random image file for every sample.

    Args:
        split_folder (Path): The split folder, receiving 'data/' and 'labels.json'.
        samples (List[Dict[str, Any]]): The samples of the manifest, each with its 'file_name'.
    """
    categories = [{"id": 1, "name": "sortable_waste"}, {"id": 2, "name": "glass_waste"}]
    (split_folder / "data").mkdir(parents=True)
    random_generator = np.random.default_rng(seed=0)
    for sample in samples:
        image_path = split_folder / "data" / sample["file_name"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(filename=str(image_path), img=random_generator.integers(low=0, high=256, size=(36, 48, 3),
                                                                            dtype=np.uint8))
    (split_folder / "labels.json").write_text(json.dumps({"categories": categories, "samples": samples}),
                                              encoding="utf-8")


def build_sample(sample_identifier: int, truck: str, category_identifier: int | None) -> Dict[str, Any]:
    """
    Builds a sample of a fake manifest, of the given truck and category, or without category when it is None.

    Args:
        sample_identifier (int): The identifier of the sample, giving its file name.
        truck (str): The truck of its groups.
        category_identifier (int | None): Its category identifier, or None for an image without category.

    Returns:
        Dict[str, Any]: The sample.
    """
    sample = {"id": sample_identifier, "file_name": f"{truck}/{sample_identifier}.png", "groups": {"truck": truck}}
    if category_identifier is not None:
        sample["category_id"] = category_identifier
    return sample


def build_augmenters() -> Tuple[DataAugmenter, DataAugmenter]:
    """
    Builds the supervised and the self-supervised augmenters from the data augmentation of CONFIGURATION_PATH.

    Returns:
        Tuple[DataAugmenter, DataAugmenter]: The augmenter of the supervised tasks and the one of the views.
    """
    augmentation_configuration = load_configuration(configuration_path=CONFIGURATION_PATH)["Data"]["data_augmentation"]
    return (DataAugmenter(augmentation_configuration=augmentation_configuration, self_supervised=False),
            DataAugmenter(augmentation_configuration=augmentation_configuration, self_supervised=True))


def create_test_images(number_images: int) -> torch.Tensor:
    """
    Creates random float BGR images in [0, 255], the same at every call.

    Args:
        number_images (int): The number of images.

    Returns:
        torch.Tensor: The [N, 3, IMAGE_SIZE, IMAGE_SIZE] images.
    """
    return torch.rand(size=(number_images, 3, IMAGE_SIZE, IMAGE_SIZE),
                      generator=torch.Generator().manual_seed(0)) * 255.0


def test_example_images_come_from_different_trucks_and_categories(tmp_path: Path) -> None:
    """
    Verifies select_example_images: the chosen images come from different trucks and, while the split has new
    categories, from different categories, an image without category is never chosen, and asking for more images than
    the split has trucks with labelled images stops with an error.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    samples = [build_sample(sample_identifier=1, truck="t1", category_identifier=1),
               build_sample(sample_identifier=2, truck="t1", category_identifier=1),
               build_sample(sample_identifier=3, truck="t2", category_identifier=1),
               build_sample(sample_identifier=4, truck="t3", category_identifier=2),
               build_sample(sample_identifier=5, truck="t4", category_identifier=None)]
    write_fake_split(split_folder=tmp_path / "validation", samples=samples)

    # Two images of two trucks and both categories, then three images of the three trucks with labels
    for seed in range(5):
        two_examples = select_example_images(split_folder=tmp_path / "validation", number_images=2, seed=seed)
        assert len({example["truck"] for example in two_examples}) == 2
        assert {example["category"] for example in two_examples} == {"sortable_waste", "glass_waste"}
        three_examples = select_example_images(split_folder=tmp_path / "validation", number_images=3, seed=seed)
        assert sorted(example["truck"] for example in three_examples) == ["t1", "t2", "t3"]
        assert all(example["path"].is_file() for example in three_examples)

    # The truck of the image without category does not count
    with pytest.raises(expected_exception=ValueError, match="3 different trucks"):
        select_example_images(split_folder=tmp_path / "validation", number_images=4, seed=0)


def test_both_strengths_of_a_type_draw_the_same_random_numbers() -> None:
    """
    Verifies apply_type_alone: with the same seed, the rotation of the views is the supervised rotation scaled by the
    ratio of the largest angles of both strength sets, and a type of the same strength in both sets, the flip, gives the
    same image.
    """
    supervised_augmenter, self_supervised_augmenter = build_augmenters()
    image = create_test_images(number_images=1)[0]

    # Ratio of the largest rotation of the views to that of the supervised tasks, scaling the same drawn number
    rotation_ratio = (SELF_SUPERVISED_AUGMENTATION_STRENGTHS["maximum_rotation_degrees"] /
                      SUPERVISED_AUGMENTATION_STRENGTHS["maximum_rotation_degrees"])
    for seed in range(4):
        _, supervised_values = apply_type_alone(augmenter=supervised_augmenter, augmentation_type="random_rotation",
                                                image=image, seed=seed)
        _, self_supervised_values = apply_type_alone(augmenter=self_supervised_augmenter,
                                                     augmentation_type="random_rotation", image=image, seed=seed)
        supervised_angle, self_supervised_angle = [float(re.match(pattern=r"([+-][0-9.]+) degrees",
                                                                  string=drawn_values).group(1))
                                                   for drawn_values in [supervised_values, self_supervised_values]]
        assert abs(self_supervised_angle - rotation_ratio * supervised_angle) <= 0.25

    # The same flip for both strengths, the caption of a type without drawn value saying so
    supervised_flip, flip_values = apply_type_alone(augmenter=supervised_augmenter,
                                                    augmentation_type="random_left_right_flip", image=image, seed=0)
    self_supervised_flip, _ = apply_type_alone(augmenter=self_supervised_augmenter,
                                               augmentation_type="random_left_right_flip", image=image, seed=0)
    assert torch.equal(input=supervised_flip, other=image.flip(dims=(-1,)))
    assert torch.equal(input=self_supervised_flip, other=supervised_flip)
    assert flip_values == "applied, no value drawn"


def test_type_figure_shows_the_image_then_both_strengths() -> None:
    """
    Verifies build_type_figure: three columns of cells separated by CELL_GAP under the title band, the first one holding
    the image itself and the next ones the type applied alone at each strength.
    """
    augmenters = build_augmenters()
    images = create_test_images(number_images=2)
    figure = build_type_figure(images=images, image_descriptions=["t1, sortable_waste", "t2, glass_waste"],
                               augmenters=augmenters, augmentation_type="random_up_down_flip", seed=0)
    assert figure.dtype == np.uint8 and figure.shape[1] == 3 * IMAGE_SIZE + 4 * CELL_GAP
    assert figure.shape[0] > TITLE_HEIGHT + 2 * IMAGE_SIZE

    # The image, then its flip at both strengths, in the cells of the first row
    first_row = figure[TITLE_HEIGHT:TITLE_HEIGHT + IMAGE_SIZE]
    cells = [first_row[:, CELL_GAP + column_index * (IMAGE_SIZE + CELL_GAP):
                          CELL_GAP + column_index * (IMAGE_SIZE + CELL_GAP) + IMAGE_SIZE] for column_index in range(3)]
    assert np.array_equal(cells[0], convert_to_cell(image=images[0]))
    assert np.array_equal(cells[1], convert_to_cell(image=images[0].flip(dims=(-2,))))
    assert np.array_equal(cells[2], cells[1])


def test_crop_ratio_figure_frames_and_resizes_the_region_of_each_ratio() -> None:
    """
    Verifies build_crop_ratio_figure: every region keeps half of the area at its width / height ratio, as draw_area_crop
    computes its sides, is framed on the image in its colour, and is resized back to the input size by crop_and_resize.
    """
    _, self_supervised_augmenter = build_augmenters()
    image = create_test_images(number_images=1)
    figure = build_crop_ratio_figure(images=image, image_descriptions=["t1, sortable_waste"],
                                     augmenter=self_supervised_augmenter)
    assert figure.shape[1] == 4 * IMAGE_SIZE + 5 * CELL_GAP

    # The region of the widest ratio of the views, half of the 64x64 pixels centred on the image (64x32 pixels at top
    # 16 and left 0 for a ratio of 2), framed last and resized in the last column
    widest_ratio = SELF_SUPERVISED_AUGMENTATION_STRENGTHS["crop_aspect_ratio_range"][1]
    crop_width = round((IMAGE_SIZE * IMAGE_SIZE * EXPLAINED_CROP_AREA * widest_ratio) ** 0.5)
    crop_height = round((IMAGE_SIZE * IMAGE_SIZE * EXPLAINED_CROP_AREA / widest_ratio) ** 0.5)
    top, left = (IMAGE_SIZE - crop_height) // 2, (IMAGE_SIZE - crop_width) // 2
    first_row = figure[TITLE_HEIGHT:TITLE_HEIGHT + IMAGE_SIZE]
    framed_cell = first_row[:, CELL_GAP:CELL_GAP + IMAGE_SIZE]
    assert tuple(int(value) for value in framed_cell[top, left]) == REGION_COLOURS[2]
    expected_crop, _, _ = self_supervised_augmenter.crop_and_resize(image=image[0], bounding_boxes=None, mask=None,
                                                                    top=top, left=left, crop_height=crop_height,
                                                                    crop_width=crop_width)
    last_left = CELL_GAP + 3 * (IMAGE_SIZE + CELL_GAP)
    assert np.array_equal(first_row[:, last_left:last_left + IMAGE_SIZE], convert_to_cell(image=expected_crop))


def test_noise_figure_adds_the_noise_of_every_level() -> None:
    """
    Verifies build_noise_levels_figure: on a flat grey image, the noise of every column has the standard deviation of
    its level of EXPLAINED_NOISE_LEVELS, within 5 %.
    """
    grey_images = torch.full(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE), fill_value=128.0)
    augmentation_configuration = load_configuration(configuration_path=CONFIGURATION_PATH)["Data"]["data_augmentation"]
    figure = build_noise_levels_figure(images=grey_images, image_descriptions=["t1, sortable_waste"],
                                       augmentation_configuration=augmentation_configuration, seed=0)
    first_row = figure[TITLE_HEIGHT:TITLE_HEIGHT + IMAGE_SIZE].astype(np.float64)
    for column_index, noise_level in enumerate(EXPLAINED_NOISE_LEVELS, start=1):
        left = CELL_GAP + column_index * (IMAGE_SIZE + CELL_GAP)
        noise = first_row[:, left:left + IMAGE_SIZE] - 128.0
        assert abs(float(noise.std()) - noise_level) <= 0.05 * noise_level


def test_noise_columns_name_the_levels_of_both_strength_sets() -> None:
    """
    Verifies name_noise_level: the column of the level of the supervised tasks and the one of the views name their
    set, and a level used by neither has no name.
    """
    supervised_level = SUPERVISED_AUGMENTATION_STRENGTHS["noise_standard_deviation"]
    self_supervised_level = SELF_SUPERVISED_AUGMENTATION_STRENGTHS["noise_standard_deviation"]
    assert name_noise_level(noise_level=supervised_level) == f"{supervised_level:g} levels (supervised)"
    assert name_noise_level(noise_level=self_supervised_level) == f"{self_supervised_level:g} levels (views)"
    assert name_noise_level(noise_level=13.0) == "13 levels"


def test_figures_are_written_for_every_type(tmp_path: Path) -> None:
    """
    Verifies visualize_augmentation_strengths on a fake split: one figure named after every augmentation type, then the
    explanations of the crop aspect ratio and of the noise levels, each a readable image of the expected width at the
    input size of the configuration.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    samples = [build_sample(sample_identifier=1, truck="t1", category_identifier=1),
               build_sample(sample_identifier=2, truck="t2", category_identifier=2)]
    write_fake_split(split_folder=tmp_path / "validation", samples=samples)

    # The configuration of the tests at the size of the test images
    configuration = yaml.safe_load(CONFIGURATION_PATH.read_text(encoding="utf-8"))
    configuration["Data"]["image_settings"]["size"] = IMAGE_SIZE
    configuration_path = tmp_path / "configuration.yaml"
    configuration_path.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")

    written_paths = visualize_augmentation_strengths(configuration_path=configuration_path,
                                                     split_folder=tmp_path / "validation",
                                                     output_folder=tmp_path / "figures", number_images=2, seed=0)
    # Three columns per type, the image and the regions of the three ratios, the image and every noise level
    explanation_columns = {"random_crop_aspect_ratio_explained.png": 4,
                           "random_noise_levels_explained.png": 1 + len(EXPLAINED_NOISE_LEVELS)}
    assert [path.name for path in written_paths] == ([f"{name}.png" for name in AUGMENTATION_TYPES] +
                                                     list(explanation_columns))
    for figure_path in written_paths:
        figure = cv2.imread(filename=str(figure_path))
        number_columns = explanation_columns.get(figure_path.name, 3)
        assert figure is not None and figure.shape[1] == number_columns * IMAGE_SIZE + (number_columns + 1) * CELL_GAP
