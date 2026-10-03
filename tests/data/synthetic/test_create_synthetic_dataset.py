"""
Tests of vision_training/data/synthetic/create_synthetic_dataset.py: the synthetic detection, classification, sequence
classification, sequence labelling and sequence comparison datasets are valid, load with the datasets of the framework
and follow their labels.
"""

import json
import cv2
import numpy as np
import pytest

from pathlib import Path

from vision_training.data.classification.classification_dataset import ClassificationDataset
from vision_training.data.detection.coco_dataset import CocoDetectionDataset
from vision_training.data.synthetic.create_synthetic_dataset import (MOVEMENT_CATEGORIES, MOVEMENT_DIRECTIONS,
                                                                     PAIR_CATEGORIES, SHAPE_CATEGORIES,
                                                                     SHAPE_DRAWING_FUNCTIONS, create_synthetic_dataset,
                                                                     draw_shapes_frame, get_mask_bounding_box)

IMAGE_SIZE = 256


@pytest.mark.parametrize(argnames="shape_name", argvalues=list(SHAPE_CATEGORIES))
def test_shape_boxes_are_tight(shape_name: str) -> None:
    """
    Verifies that the box computed for every shape class is tight: every drawn pixel lies inside it, and each of its
    four borders touches the shape.

    Args:
        shape_name (str): The shape class under test.
    """
    random_generator = np.random.default_rng(seed=0)

    for _ in range(20):
        mask = np.zeros(shape=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        SHAPE_DRAWING_FUNCTIONS[shape_name](mask=mask, center_x=128, center_y=128, size=64,
                                            random_generator=random_generator)
        x_1, y_1, x_2, y_2 = get_mask_bounding_box(mask=mask)

        # Every shape pixel is inside the box, and the first and last rows and columns of the box contain the shape
        assert mask.sum() == mask[y_1:y_2, x_1:x_2].sum()
        assert mask[y_1, x_1:x_2].any() and mask[y_2 - 1, x_1:x_2].any()
        assert mask[y_1:y_2, x_1].any() and mask[y_1:y_2, x_2 - 1].any()


def test_created_dataset_is_valid_coco_and_loads(tmp_path: Path) -> None:
    """
    Verifies that the created dataset has the folder layout and COCO annotations expected by the data loaders:
    every image listed with its size, boxes inside the images, the four shape categories, a label mapping for the
    multi-class mode, and that the Native dataset loads it with labels 1 to 4.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_synthetic_dataset(output_folder=str(tmp_path), image_size=IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 2, "test": 2}, maximum_objects=4, number_frames=1,
                             number_workers=2, task="object_detection")

    labels = json.loads((tmp_path / "train" / "labels.json").read_text())
    assert len(labels["images"]) == 8 and len(labels["annotations"]) >= 8
    assert [category["name"] for category in labels["categories"]] == list(SHAPE_CATEGORIES)
    assert all((tmp_path / "train" / "data" / image["file_name"]).is_file() for image in labels["images"])
    assert (tmp_path / "validation" / "labels.json").is_file() and (tmp_path / "test" / "labels.json").is_file()

    # Every COCO [x, y, width, height] box lies inside its image and has a positive size
    for annotation in labels["annotations"]:
        x, y, width, height = annotation["bbox"]
        assert width > 0 and height > 0 and x >= 0 and y >= 0
        assert x + width <= IMAGE_SIZE and y + height <= IMAGE_SIZE

    # The multi-class label mapping keeps every raw category identifier as its own model class
    label_mapping = json.loads((tmp_path / "label_mapping.json").read_text())
    label_dictionary = {int(raw_identifier): model_identifier for raw_identifier, model_identifier in
                        label_mapping.items()}
    label_dictionary["background"] = 0

    dataset = CocoDetectionDataset(data_directory=str(tmp_path / "train" / "data"),
                            labels_file=str(tmp_path / "train" / "labels.json"), image_size=IMAGE_SIZE,
                            keep_ratio=True, label_mode="multi_class", label_dictionary=label_dictionary)
    loaded_labels = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample["bounding_boxes"].shape[0] == sample["labels"].shape[0] > 0
        loaded_labels |= set(sample["labels"].tolist())
    assert loaded_labels <= {1, 2, 3, 4}


def test_created_classification_dataset_has_one_shape_per_image(tmp_path: Path) -> None:
    """
    Verifies the classification mode of the generator: every split gets a manifest with one sample per image, labelled
    with the category of its single shape, a label mapping giving the classes 0 to 3, and the classification dataset
    loads it with its class names.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_synthetic_dataset(output_folder=str(tmp_path), image_size=IMAGE_SIZE,
                             split_sizes={"train": 12, "validation": 2, "test": 2}, maximum_objects=4,
                             number_frames=1, number_workers=2, task="classification")

    labels = json.loads((tmp_path / "train" / "labels.json").read_text())
    assert len(labels["samples"]) == 12
    assert [category["name"] for category in labels["categories"]] == list(SHAPE_CATEGORIES)
    assert all(sample["category_id"] in SHAPE_CATEGORIES.values() for sample in labels["samples"])
    assert all((tmp_path / "train" / "data" / sample["file_name"]).is_file() for sample in labels["samples"])

    # The classifier predicts no background class: the four categories become the classes 0 to 3
    label_mapping = json.loads((tmp_path / "label_mapping.json").read_text())
    assert label_mapping == {"1": 0, "2": 1, "3": 2, "4": 3}

    dataset = ClassificationDataset(data_directory=str(tmp_path / "train" / "data"),
                                    labels_file=str(tmp_path / "train" / "labels.json"), image_size=IMAGE_SIZE,
                                    keep_ratio=True, label_dictionary={int(category): class_index for category,
                                                                       class_index in label_mapping.items()},
                                    output_mode="multi_class")
    assert dataset.class_names == {0: "square", 1: "circle", 2: "triangle", 3: "rectangle"}
    assert sum(dataset.get_class_counts()) == 12


def get_shape_centroid(frame: np.ndarray) -> np.ndarray:
    """
    Locates the shape of a synthetic sequence frame: the centroid of the pixels differing from the background, the
    background being the most frequent colour of the frame.

    Args:
        frame (np.ndarray): The [H, W, 3] uint8 BGR frame.

    Returns:
        np.ndarray: The [x, y] centroid of the shape in pixels.
    """
    colours, counts = np.unique(frame.reshape(-1, 3), axis=0, return_counts=True)
    shape_rows, shape_columns = np.nonzero(np.any(frame != colours[np.argmax(counts)], axis=-1))
    return np.array([shape_columns.mean(), shape_rows.mean()])


def test_created_sequence_dataset_moves_its_shapes_in_the_labelled_direction(tmp_path: Path) -> None:
    """
    Verifies the sequence classification mode of the generator: every split gets a manifest of sequences with their
    ordered frames, the label mapping gives the four directions the classes 0 to 3, and the shape of every sequence
    moves in the direction of its label between its first and its last frame.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_synthetic_dataset(output_folder=str(tmp_path), image_size=64,
                             split_sizes={"train": 12, "validation": 2, "test": 2}, maximum_objects=1,
                             number_frames=6, number_workers=2, task="sequence_classification")

    labels = json.loads((tmp_path / "train" / "labels.json").read_text())
    assert [category["name"] for category in labels["categories"]] == list(MOVEMENT_CATEGORIES)
    assert json.loads((tmp_path / "label_mapping.json").read_text()) == {"1": 0, "2": 1, "3": 2, "4": 3}
    assert len(labels["sequences"]) == 12

    # Displacement of the shape between the first and the last frame, along the direction of the label
    category_names = {category_identifier: name for name, category_identifier in MOVEMENT_CATEGORIES.items()}
    for sequence in labels["sequences"]:
        assert len(sequence["frames"]) == 6
        frames = [cv2.imread(filename=str(tmp_path / "train" / "data" / frame_file))
                  for frame_file in sequence["frames"]]
        displacement = get_shape_centroid(frame=frames[-1]) - get_shape_centroid(frame=frames[0])
        direction_x, direction_y = MOVEMENT_DIRECTIONS[category_names[sequence["category_id"]]]
        assert displacement @ np.array([direction_x, direction_y]) > 5.0
        assert abs(displacement @ np.array([direction_y, direction_x])) < 1.0


def test_created_sequence_labelling_dataset_labels_every_frame(tmp_path: Path) -> None:
    """
    Verifies the sequence labelling mode of the generator: every sequence lists the shape category of every frame,
    the shape changes class at least once, the frames of the same class segment are identical and two consecutive
    segments differ, and the label mapping gives the four shapes the classes 0 to 3.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_synthetic_dataset(output_folder=str(tmp_path), image_size=64,
                             split_sizes={"train": 10, "validation": 2, "test": 2}, maximum_objects=1,
                             number_frames=6, number_workers=2, task="sequence_labelling")

    labels = json.loads((tmp_path / "train" / "labels.json").read_text())
    assert [category["name"] for category in labels["categories"]] == list(SHAPE_CATEGORIES)
    assert json.loads((tmp_path / "label_mapping.json").read_text()) == {"1": 0, "2": 1, "3": 2, "4": 3}

    for sequence in labels["sequences"]:
        frame_category_ids = sequence["frame_category_ids"]
        assert len(frame_category_ids) == len(sequence["frames"]) == 6
        assert len(set(frame_category_ids)) >= 2 and set(frame_category_ids) <= set(SHAPE_CATEGORIES.values())

        # Identical frames within a class segment, different frames across a change of class
        frames = [cv2.imread(filename=str(tmp_path / "train" / "data" / frame_file))
                  for frame_file in sequence["frames"]]
        for frame_index in range(1, 6):
            same_class = frame_category_ids[frame_index] == frame_category_ids[frame_index - 1]
            assert np.array_equal(frames[frame_index], frames[frame_index - 1]) == same_class


def count_frame_objects(frame: np.ndarray) -> int:
    """
    Counts the objects of a synthetic frame: the 8-connected groups of the pixels differing from the background colour,
    the most frequent colour of the frame.

    Args:
        frame (np.ndarray): The [H, W, 3] uint8 BGR frame.

    Returns:
        int: The number of objects of the frame.
    """
    colours, counts = np.unique(frame.reshape(-1, 3), axis=0, return_counts=True)
    object_mask = np.any(frame != colours[np.argmax(counts)], axis=-1).astype(np.uint8)
    number_components, _ = cv2.connectedComponents(image=object_mask, connectivity=8)
    return number_components - 1


def test_created_sequence_comparison_dataset_loses_an_object_in_the_after_member(tmp_path: Path) -> None:
    """
    Verifies the sequence comparison mode of the generator: every split gets a manifest of pairs with the ordered frames
    of their before and after members (2 to 4 frames each), the label mapping gives the two classes 0 and 1, every
    frame of a before member shows the same 2 to 3 separate objects, and every frame of its after member shows one
    object less for the pairs labelled 'loss_of_object' and the same number of objects otherwise.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_synthetic_dataset(output_folder=str(tmp_path), image_size=64,
                             split_sizes={"train": 12, "validation": 2, "test": 2}, maximum_objects=3,
                             number_frames=4, number_workers=2, task="sequence_comparison")

    labels = json.loads((tmp_path / "train" / "labels.json").read_text())
    assert [category["name"] for category in labels["categories"]] == list(PAIR_CATEGORIES)
    assert json.loads((tmp_path / "label_mapping.json").read_text()) == {"1": 0, "2": 1}
    assert len(labels["pairs"]) == 12

    # Number of objects of every frame of both members, the same in every frame of a member
    for pair in labels["pairs"]:
        member_object_counts = {}
        for member_name in ["before", "after"]:
            assert 2 <= len(pair[member_name]) <= 4
            frame_object_counts = {count_frame_objects(frame=cv2.imread(filename=str(tmp_path / "train" / "data" /
                                                                                     frame_file)))
                                   for frame_file in pair[member_name]}
            assert len(frame_object_counts) == 1
            member_object_counts[member_name] = frame_object_counts.pop()
        assert 2 <= member_object_counts["before"] <= 3
        expected_lost_objects = 1 if pair["category_id"] == PAIR_CATEGORIES["loss_of_object"] else 0
        assert member_object_counts["before"] - member_object_counts["after"] == expected_lost_objects


def test_pair_frames_fail_loudly_when_objects_do_not_fit() -> None:
    """
    Verifies that a pair frame whose objects can never all fit without overlap stops the generation, instead of showing
    a wrong number of objects: two squares of 20 pixels cannot be 2 pixels apart in a 32 pixel frame.
    """
    square_mask = np.zeros(shape=(21, 21), dtype=np.uint8)
    square_mask[:20, :20] = 255
    shape_descriptors = [{"shape_name": "square", "mask": square_mask, "colour": np.full(shape=3, fill_value=255,
                                                                                         dtype=np.uint8)}] * 2

    with pytest.raises(expected_exception=RuntimeError, match="never fitted"):
        draw_shapes_frame(background=np.zeros(shape=(32, 32, 3), dtype=np.uint8), shape_descriptors=shape_descriptors,
                          random_generator=np.random.default_rng())
