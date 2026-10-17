"""
Tests of KeypointDataset (vision_training/data/keypoints/keypoint_dataset.py), the dataset of the keypoint detection
task read by get_keypoint_dataloaders: one sample per object of a COCO keypoint split, the objects mapped to the
background and those without any labelled keypoint being left out, every sample holding the letterboxed image, the box,
the class and the keypoints moved like the boxes by preprocess_boxes, padded to the most keypoints of a class with an
unlabelled padding, the unlabelled keypoints carrying no position. The listing of the objects without images, the
names of the evaluated samples and the refusal of malformed annotations are covered too.
"""

import json
import cv2
import numpy as np
import pytest
import torch

from pathlib import Path
from typing import Any, Dict, List

from vision_training.data.keypoints.keypoint_dataset import KeypointDataset
from vision_training.data.preprocessing import preprocess_boxes, preprocess_image

# Network input of the tests: the 40 x 20 first image is scaled twice with 20 rows of padding above it, the 20 x 40
# second image twice with 20 columns of padding on its left
IMAGE_SIZE = 80

# Class of every raw category: the triangle (3 keypoints) and the rectangle (4 keypoints) are learnt, the circle is
# mapped to the background
LABEL_DICTIONARY = {"background": 0, 1: 1, 2: 2, 3: 0}

# Annotations of the split: a triangle with a hidden and an unlabelled keypoint, a rectangle without any labelled
# keypoint, a circle of the background and a fully labelled rectangle
SPLIT_ANNOTATIONS = [
    {"id": 11, "image_id": 1, "category_id": 1, "bbox": [2, 2, 10, 8], "iscrowd": 0,
     "keypoints": [7.0, 2.5, 2, 3.0, 9.0, 1, 50.0, 50.0, 0]},
    {"id": 12, "image_id": 1, "category_id": 2, "bbox": [20, 5, 10, 10], "iscrowd": 0, "keypoints": [0] * 12},
    {"id": 21, "image_id": 2, "category_id": 3, "bbox": [2, 2, 10, 10], "iscrowd": 0, "keypoints": [7, 7, 2]},
    {"id": 22, "image_id": 2, "category_id": 2, "bbox": [4, 20, 12, 6], "iscrowd": 0,
     "keypoints": [4.0, 20.0, 2, 16.0, 20.0, 2, 16.0, 26.0, 1, 4.0, 26.0, 2]}]


def write_split(folder: Path, annotations: List[Dict[str, Any]]) -> Path:
    """
    Writes a two-image COCO keypoint split, a 40 x 20 image in a subfolder and a 20 x 40 image filled with random
    pixels, with the triangle, rectangle and circle categories and their keypoint names.

    Args:
        folder (Path): The folder of the split.
        annotations (List[Dict[str, Any]]): The annotations of the images 1 and 2.

    Returns:
        Path: The 'labels.json' file of the split.
    """
    (folder / "data" / "first_folder").mkdir(parents=True)
    random_generator = np.random.default_rng(seed=0)
    cv2.imwrite(filename=str(folder / "data" / "first_folder" / "first.png"),
                img=random_generator.integers(low=0, high=256, size=(20, 40, 3), dtype=np.uint8))
    cv2.imwrite(filename=str(folder / "data" / "second.png"),
                img=random_generator.integers(low=0, high=256, size=(40, 20, 3), dtype=np.uint8))
    labels_file = folder / "labels.json"
    labels_file.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "first_folder/first.png", "width": 40, "height": 20},
                   {"id": 2, "file_name": "second.png", "width": 20, "height": 40}],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "triangle", "keypoints": ["top", "left", "right"]},
                       {"id": 2, "name": "rectangle", "keypoints": ["top_left", "top_right", "bottom_right",
                                                                    "bottom_left"]},
                       {"id": 3, "name": "circle", "keypoints": ["centre"]}]}))
    return labels_file


def build_dataset(folder: Path, labels_file: Path, shuffle: bool) -> KeypointDataset:
    """
    Builds the keypoint dataset of a written split at the network input of 80 pixels with the letterbox, padded to the
    4 keypoints of the rectangle.

    Args:
        folder (Path): The folder of the split.
        labels_file (Path): The 'labels.json' file of the split.
        shuffle (bool): Whether to shuffle the objects.

    Returns:
        KeypointDataset: The dataset.
    """
    return KeypointDataset(data_directory=str(folder / "data"), labels_file=str(labels_file), image_size=IMAGE_SIZE,
                           keep_ratio=True, label_dictionary=LABEL_DICTIONARY, maximum_keypoints=4, shuffle=shuffle)


def test_objects_without_class_or_labelled_keypoint_are_left_out(tmp_path: Path) -> None:
    """
    Verifies that the circle mapped to the background and the rectangle without any labelled keypoint are left out,
    and that the listing of the objects gives the class and number of labelled keypoints of the two others.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    labels_file = write_split(folder=tmp_path, annotations=SPLIT_ANNOTATIONS)
    dataset = build_dataset(folder=tmp_path, labels_file=labels_file, shuffle=False)
    assert len(dataset) == 2
    assert dataset.list_objects() == [{"sample_identifier": 11, "label": 1, "labelled_keypoints": 2},
                                      {"sample_identifier": 22, "label": 2, "labelled_keypoints": 4}]


def test_samples_hold_the_letterboxed_image_box_and_keypoints(tmp_path: Path) -> None:
    """
    Verifies the keys, shapes and data types of the samples, that the image is the letterboxed image, that the box and
    the labelled keypoints are moved exactly like preprocess_boxes moves them (scaled twice and shifted by the
    padding), and that the unlabelled keypoint and the padding position of the triangle have a visibility of 0 and no
    position.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    labels_file = write_split(folder=tmp_path, annotations=SPLIT_ANNOTATIONS)
    dataset = build_dataset(folder=tmp_path, labels_file=labels_file, shuffle=False)
    triangle_sample, rectangle_sample = dataset[0], dataset[1]
    assert set(triangle_sample) == {"images", "bounding_boxes", "labels", "keypoints", "visibilities",
                                    "image_identifiers", "sample_identifiers"}
    assert triangle_sample["images"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert triangle_sample["images"].dtype == torch.uint8
    assert triangle_sample["keypoints"].shape == (4, 2) and triangle_sample["keypoints"].dtype == torch.float32
    assert triangle_sample["visibilities"].dtype == torch.int64 and triangle_sample["labels"].dtype == torch.int64
    assert triangle_sample["image_identifiers"] == 1 and triangle_sample["sample_identifiers"] == 11

    # The image is the letterboxed image of the first file
    image = cv2.imread(filename=str(tmp_path / "data" / "first_folder" / "first.png"))
    expected_image = preprocess_image(image=image, image_size=IMAGE_SIZE, keep_ratio=True)
    assert np.array_equal(triangle_sample["images"].permute(1, 2, 0).numpy(), expected_image)

    # The 40 x 20 image is scaled twice with 20 rows of padding above it
    assert triangle_sample["bounding_boxes"].tolist() == [4.0, 24.0, 24.0, 40.0]
    assert triangle_sample["keypoints"].tolist() == [[14.0, 25.0], [6.0, 38.0], [0.0, 0.0], [0.0, 0.0]]
    assert triangle_sample["visibilities"].tolist() == [2, 1, 0, 0] and int(triangle_sample["labels"]) == 1

    # The 20 x 40 image is scaled twice with 20 columns of padding on its left, as preprocess_boxes moves them
    moved_box, moved_keypoints = preprocess_boxes(
        bounding_boxes=[[4.0, 20.0, 16.0, 26.0], [4.0, 20.0, 16.0, 20.0, 16.0, 26.0, 4.0, 26.0]], image_width=20,
        image_height=40, image_size=IMAGE_SIZE, keep_ratio=True)
    assert rectangle_sample["bounding_boxes"].tolist() == moved_box == [28.0, 40.0, 52.0, 52.0]
    assert rectangle_sample["keypoints"].reshape(-1).tolist() == moved_keypoints
    assert rectangle_sample["visibilities"].tolist() == [2, 2, 1, 2] and int(rectangle_sample["labels"]) == 2


def test_sample_names_follow_the_shuffled_objects(tmp_path: Path) -> None:
    """
    Verifies that the name of an evaluated object joins the folders of its image file with underscores, without the
    extension, followed by its annotation identifier, and that the shuffled dataset still names every object by its
    own record.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    labels_file = write_split(folder=tmp_path, annotations=SPLIT_ANNOTATIONS)
    for shuffle in [False, True]:
        dataset = build_dataset(folder=tmp_path, labels_file=labels_file, shuffle=shuffle)
        assert dataset.get_sample_name(sample_identifier=11) == "first_folder_first_object_11"
        assert dataset.get_sample_name(sample_identifier=22) == "second_object_22"
        assert {dataset[index]["sample_identifiers"] for index in range(len(dataset))} == {11, 22}


def test_malformed_annotations_are_refused(tmp_path: Path) -> None:
    """
    Verifies that an annotation of a learnt class without 'keypoints', and an annotation whose number of keypoints
    differs from its category, are refused when the split is read.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    missing_folder, wrong_folder = tmp_path / "missing", tmp_path / "wrong"
    labels_file = write_split(folder=missing_folder, annotations=[
        {"id": 5, "image_id": 1, "category_id": 1, "bbox": [2, 2, 10, 8], "iscrowd": 0}])
    with pytest.raises(ValueError, match="annotation 5 of image 1 has no 'keypoints'"):
        build_dataset(folder=missing_folder, labels_file=labels_file, shuffle=False)

    labels_file = write_split(folder=wrong_folder, annotations=[
        {"id": 6, "image_id": 1, "category_id": 2, "bbox": [2, 2, 10, 8], "iscrowd": 0, "keypoints": [1, 1, 2]}])
    with pytest.raises(ValueError, match="annotation 6 has 1 keypoints, its category 2 defining 4"):
        build_dataset(folder=wrong_folder, labels_file=labels_file, shuffle=False)
