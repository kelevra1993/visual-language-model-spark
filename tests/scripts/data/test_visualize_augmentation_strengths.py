"""
Tests of scripts/data/visualize_augmentation_strengths.py, the figures showing every augmentation type of DataAugmenter
alone at the supervised and the self-supervised strengths: the files written for every type, followed by the figures of
the crop aspect ratio and of the noise levels, each a readable image of the expected width.
"""

import json

import cv2
import numpy as np
import yaml

from pathlib import Path
from typing import Any, Dict, List

from scripts.data.visualize_augmentation_strengths import (CELL_GAP, EXPLAINED_NOISE_LEVELS,
                                                           visualize_augmentation_strengths)
from vision_training.data.augmentations import AUGMENTATION_TYPES

# Self-supervised configuration whose data augmentation section builds the augmenters of the tests
CONFIGURATION_PATH = (Path(__file__).parents[3] / "vision_training" / "configurations" / "self_supervised" /
                      "configuration_synthetic_shapes.yaml")

# Side of the test images, large enough for the captions of the crop on two lines
IMAGE_SIZE = 64


def write_fake_split(split_folder: Path, samples: List[Dict[str, Any]]) -> None:
    """
    Writes a split in the format of the self-supervised layouts holding groups, a 'labels.json' of two categories whose
    samples hold their truck in their groups, with a random image file for every sample.

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
