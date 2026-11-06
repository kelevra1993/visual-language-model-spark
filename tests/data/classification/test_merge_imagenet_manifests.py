"""
Tests of vision_training/data/classification/merge_imagenet_manifests.py on small ImageNet-1k and ImageNet-22k folders
written by the tests: the original names of both naming schemes, the ImageNet-22k copies of the ImageNet-1k validation
images, and the merged splits with their sources, classes and left out copies.
"""

import json

import cv2
import numpy as np

from pathlib import Path
from typing import List

from vision_training.data.classification.merge_imagenet_manifests import (find_imagenet_22k_copies,
                                                                          get_original_name,
                                                                          merge_imagenet_manifests)


def write_images(class_folder: Path, file_names: List[str]) -> None:
    """
    Writes small images under the given names in a class folder, as the ImageNet downloads store them.

    Args:
        class_folder (Path): The class folder to fill.
        file_names (List[str]): The file names of the images.
    """
    class_folder.mkdir(parents=True, exist_ok=True)
    for file_name in file_names:
        cv2.imwrite(filename=str(class_folder / file_name), img=np.zeros(shape=(8, 8, 3), dtype=np.uint8))


def test_get_original_name() -> None:
    """
    Verifies that both naming schemes give the same original name, and that a test image name is kept as it is.
    """
    assert get_original_name(file_name="n01440764_10026_n01440764.JPEG") == "n01440764_10026"
    assert get_original_name(file_name="n01440764_10026.jpg") == "n01440764_10026"
    assert get_original_name(file_name="ILSVRC2012_val_00047760_n03777568.JPEG") == "ILSVRC2012_val_00047760_n03777568"


def test_merge_imagenet_manifests(tmp_path: Path) -> None:
    """
    Verifies the merged splits: every ImageNet-22k split and the ImageNet-1k train split in train, duplicates kept,
    the ImageNet-22k copies of the ImageNet-1k validation images left out of train wherever they are, the validation and
    test splits of ImageNet-1k only, and the classes of both datasets.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # ImageNet-1k: a train image also in 22k, a validation image whose copy is in the 22k test split, a class 22k lacks
    images_1k = tmp_path / "imagenet-1k" / "images"
    write_images(class_folder=images_1k / "train" / "n01440764", file_names=["n01440764_1_n01440764.JPEG"])
    write_images(class_folder=images_1k / "train" / "n04399382", file_names=["n04399382_5_n04399382.JPEG"])
    write_images(class_folder=images_1k / "validation" / "n01440764", file_names=["n01440764_2_n01440764.JPEG"])
    write_images(class_folder=images_1k / "test" / "n01440764", file_names=["ILSVRC2012_val_00000001_n01440764.JPEG"])

    # ImageNet-22k: the shared train image, the validation copy in its test split, a class of its own
    images_22k = tmp_path / "imagenet-22k" / "images"
    write_images(class_folder=images_22k / "train" / "n01440764", file_names=["n01440764_1.jpg", "n01440764_3.jpg"])
    write_images(class_folder=images_22k / "validation" / "n00004475", file_names=["n00004475_7.jpg"])
    write_images(class_folder=images_22k / "test" / "n01440764", file_names=["n01440764_2.jpg"])

    validation_copies = find_imagenet_22k_copies(imagenet_1k_split_folder=images_1k / "validation",
                                                 imagenet_22k_images_folder=images_22k)
    assert validation_copies == {"imagenet-22k-test/n01440764/n01440764_2.jpg"}

    output_folder = tmp_path / "imagenet-merged"
    merge_imagenet_manifests(imagenet_1k_folder=str(tmp_path / "imagenet-1k"),
                             imagenet_22k_folder=str(tmp_path / "imagenet-22k"), output_folder=str(output_folder))

    # Train: the shared image twice (22k and 1k), the other images of both, the validation copy left out
    manifests = {split_name: json.loads((output_folder / split_name / "labels.json").read_text())
                 for split_name in ["train", "validation", "test"]}
    assert sorted(sample["file_name"] for sample in manifests["train"]["samples"]) == \
        ["imagenet-1k-train/n01440764/n01440764_1_n01440764.JPEG",
         "imagenet-1k-train/n04399382/n04399382_5_n04399382.JPEG",
         "imagenet-22k-train/n01440764/n01440764_1.jpg", "imagenet-22k-train/n01440764/n01440764_3.jpg",
         "imagenet-22k-validation/n00004475/n00004475_7.jpg"]
    assert [sample["file_name"] for sample in manifests["validation"]["samples"]] == \
        ["imagenet-1k-validation/n01440764/n01440764_2_n01440764.JPEG"]
    assert [sample["file_name"] for sample in manifests["test"]["samples"]] == \
        ["imagenet-1k-test/n01440764/ILSVRC2012_val_00000001_n01440764.JPEG"]

    # Classes of both datasets, shared by the three splits, and every source linked under the train data folder
    assert [category["name"] for category in manifests["test"]["categories"]] == ["n00004475", "n01440764",
                                                                                  "n04399382"]
    assert sorted(path.name for path in (output_folder / "train" / "data").iterdir()) == \
        ["imagenet-1k-train", "imagenet-22k-test", "imagenet-22k-train", "imagenet-22k-validation"]
