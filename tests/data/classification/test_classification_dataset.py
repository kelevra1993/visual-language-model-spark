import json
import cv2
import numpy as np
import pytest
import torch

from pathlib import Path
from typing import Any, Dict, List

from vision_training.data.classification.class_folder_manifests import create_class_folder_manifests
from vision_training.data.classification.classification_dataset import ClassificationDataset
from vision_training.data.label_mapping import build_label_dictionary


def write_split(split_folder: Path, categories: List[Dict[str, Any]], samples: List[Dict[str, Any]]) -> None:
    """
    Writes a classification split in the layout read by the classification loader: one 32 pixel image per sample in
    'data/' and the 'labels.json' manifest listing the categories and the samples.

    Args:
        split_folder (Path): The folder of the split.
        categories (List[Dict[str, Any]]): The 'id' and 'name' of every category of the manifest.
        samples (List[Dict[str, Any]]): The samples of the manifest, with their 'id', 'file_name' and categories.
    """
    (split_folder / "data").mkdir(parents=True)
    for sample in samples:
        cv2.imwrite(filename=str(split_folder / "data" / sample["file_name"]),
                    img=np.full(shape=(32, 48, 3), fill_value=sample["id"], dtype=np.uint8))
    (split_folder / "labels.json").write_text(data=json.dumps(obj={"categories": categories, "samples": samples}))


def test_multi_class_manifest_loading(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the dataset reads a multi-class manifest: the images are resized to the network input, the raw
    categories become class indices, the samples of unmapped categories are left out with a warning, the class names
    come from the categories (joined when the label dictionary merges two categories) and the class counts match.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed warning.
    """
    categories = [{"id": 1, "name": "circle"}, {"id": 2, "name": "square"}, {"id": 3, "name": "box"},
                  {"id": 4, "name": "star"}]
    samples = [{"id": 1, "file_name": "a.png", "category_id": 1}, {"id": 2, "file_name": "b.png", "category_id": 2},
               {"id": 3, "file_name": "c.png", "category_id": 3}, {"id": 4, "file_name": "d.png", "category_id": 4}]
    write_split(split_folder=tmp_path, categories=categories, samples=samples)

    # Squares and boxes are merged into class 1, stars are not mapped
    dataset = ClassificationDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                                    image_size=64, keep_ratio=True, label_dictionary={1: 0, 2: 1, 3: 1},
                                    output_mode="multi_class", shuffle=False)
    assert len(dataset) == 3
    assert "1 samples of unmapped categories left out" in capsys.readouterr().out
    assert dataset.class_names == {0: "circle", 1: "square+box"}
    assert dataset.get_class_counts() == [1, 2]

    sample = dataset[1]
    assert sample["images"].shape == (3, 64, 64) and sample["images"].dtype == torch.uint8
    assert sample["labels"].dtype == torch.int64 and int(sample["labels"]) == 1
    assert sample["sample_identifiers"] == 2


def test_multi_label_manifest_loading(tmp_path: Path) -> None:
    """
    Verifies that the dataset reads a multi-label manifest: every sample gets the multi-hot vector of its mapped
    categories, a sample without any mapped category being kept as a negative of every class.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    categories = [{"id": 1, "name": "red"}, {"id": 2, "name": "round"}, {"id": 3, "name": "large"}]
    samples = [{"id": 1, "file_name": "a.png", "category_ids": [1, 2]}, {"id": 2, "file_name": "b.png",
                                                                         "category_ids": [3]}]
    write_split(split_folder=tmp_path, categories=categories, samples=samples)

    dataset = ClassificationDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                                    image_size=32, keep_ratio=False, label_dictionary={1: 0, 2: 1},
                                    output_mode="multi_label", shuffle=False)
    assert len(dataset) == 2
    assert dataset[0]["labels"].tolist() == [1.0, 1.0] and dataset[0]["labels"].dtype == torch.float32
    assert dataset[1]["labels"].tolist() == [0.0, 0.0]
    assert dataset.get_class_counts() == [1, 1]


def test_label_mapping_without_background_class(tmp_path: Path) -> None:
    """
    Verifies that the label dictionary of the classification task keeps the class indices of the mapping file without
    adding the background class of the object detector, and that the binary mode, which only makes sense next to a
    background class, gives no dictionary for it.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    mapping_path = tmp_path / "label_mapping.json"
    mapping_path.write_text(data=json.dumps(obj={"1": 0, "2": 1}))
    label_mapping_configuration = {"mode": "multi_class", "mapping_file_path": str(mapping_path)}

    assert build_label_dictionary(label_mapping_configuration=label_mapping_configuration,
                                  task="classification") == {1: 0, 2: 1}
    assert build_label_dictionary(label_mapping_configuration=label_mapping_configuration,
                                  task="object_detection") == {1: 0, 2: 1}
    assert build_label_dictionary(label_mapping_configuration={"mode": "binary", "mapping_file_path": None},
                                  task="classification") is None


def test_class_folder_manifests(tmp_path: Path) -> None:
    """
    Verifies the conversion of a dataset stored as one folder per class: every split gets a 'data' link to its class
    folders and a manifest of its images, the categories (the class folders of all the splits, in alphabetical order)
    are the same for every split even when a split lacks a class, the label mapping gives the classes 0 to K-1, and
    the classification dataset reads the converted split.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # Training split with two classes, validation split without circles
    for split_name, class_images in [("training_data", {"squares": 2, "circles": 1}),
                                     ("validation_data", {"squares": 1})]:
        for class_name, number_images in class_images.items():
            class_folder = tmp_path / split_name / class_name
            class_folder.mkdir(parents=True)
            for image_index in range(number_images):
                cv2.imwrite(filename=str(class_folder / f"image_{image_index}.png"),
                            img=np.zeros(shape=(16, 16, 3), dtype=np.uint8))
            (class_folder / "notes.txt").write_text(data="not an image")

    output_folder = tmp_path / "manifests"
    create_class_folder_manifests(split_folders={"train": str(tmp_path / "training_data"),
                                                 "validation": str(tmp_path / "validation_data")},
                                  output_folder=str(output_folder))

    train_labels = json.loads((output_folder / "train" / "labels.json").read_text())
    validation_labels = json.loads((output_folder / "validation" / "labels.json").read_text())
    assert train_labels["categories"] == validation_labels["categories"] == [{"id": 1, "name": "circles"},
                                                                             {"id": 2, "name": "squares"}]
    assert [sample["file_name"] for sample in train_labels["samples"]] == ["circles/image_0.png",
                                                                           "squares/image_0.png",
                                                                           "squares/image_1.png"]
    assert [sample["category_id"] for sample in validation_labels["samples"]] == [2]
    assert (output_folder / "train" / "data").is_symlink()
    assert json.loads((output_folder / "label_mapping.json").read_text()) == {"1": 0, "2": 1}

    # A second conversion replaces the links and manifests
    create_class_folder_manifests(split_folders={"train": str(tmp_path / "training_data")},
                                  output_folder=str(output_folder))

    dataset = ClassificationDataset(data_directory=str(output_folder / "train" / "data"),
                                    labels_file=str(output_folder / "train" / "labels.json"), image_size=16,
                                    keep_ratio=True, label_dictionary={1: 0, 2: 1}, output_mode="multi_class",
                                    shuffle=False)
    assert [int(dataset[sample_index]["labels"]) for sample_index in range(len(dataset))] == [0, 1, 1]
