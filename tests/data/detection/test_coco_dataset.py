"""
Tests of the instance masks of CocoDetectionDataset (vision_training/data/detection/coco_dataset.py), read by the
instance segmentation task: every object, the crowd annotations included, gets its mask at the network resolution in
the order of its box, the objects whose mask vanishes are removed with their box and label, the collate function
keeps the masks as a list, the object detection samples stay without masks, the instance segmentation boxes come from
the masks without any 'bbox', the selection of the objects of an image maps the categories, keeps the crowd
annotations and leaves out the background, the test samples of the
evaluation at the original resolution carry the masks and boxes of their objects at that resolution, every object with
a pixel in the original image kept at both resolutions, the keypoint samples carry as ignored regions the masks of
the objects of a category with keypoints left out for having no labelled keypoint, the instance index maps
('Data.instance_index_maps') give the objects, boxes and masks of the per-object masks, stacked by the collate function,
the packed records of the manifest give every image with its own annotations in the order of the manifest or shuffled,
without any Python object per image kept for the loader workers to copy, and images sharing an identifier are refused.
"""

import gc
import json
import random
import cv2
import numpy as np
import pytest
import torch

from pathlib import Path
from typing import Any, Dict, List, Optional

from vision_training.data.detection.coco_dataset import CocoDetectionDataset, collate_function, select_image_objects
from vision_training.data.detection.coco_masks import (build_instance_mask, compute_mask_bounding_boxes,
                                                       encode_run_length_encoding, expand_instance_index_maps)

IMAGE_WIDTH, IMAGE_HEIGHT = 40, 20


def write_split(folder: Path, annotations: List[Dict[str, Any]],
                categories: Optional[List[Dict[str, Any]]] = None) -> Path:
    """
    Writes a one-image COCO split (a black 40 x 20 image and the given annotations) in the layout read by the dataset.

    Args:
        folder (Path): The folder of the split.
        annotations (List[Dict[str, Any]]): The annotations of the image, whose 'image_id' is 1.
        categories (Optional[List[Dict[str, Any]]]): The categories of the split, or None for the categories 1 and 2
            without keypoints. Defaults to None.

    Returns:
        Path: The 'labels.json' file of the split.
    """
    (folder / "data").mkdir(parents=True)
    cv2.imwrite(filename=str(folder / "data" / "image.png"), img=np.zeros(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                                                                           dtype=np.uint8))
    labels_file = folder / "labels.json"
    if categories is None:
        categories = [{"id": 1, "name": "first"}, {"id": 2, "name": "second"}]
    labels_file.write_text(json.dumps({"images": [{"id": 1, "file_name": "image.png", "width": IMAGE_WIDTH,
                                                   "height": IMAGE_HEIGHT}],
                                       "annotations": annotations, "categories": categories}))
    return labels_file


def build_dataset(folder: Path, labels_file: Path, load_instance_masks: bool) -> CocoDetectionDataset:
    """
    Builds the dataset of a written split at a network resolution of 80 pixels, the 40 x 20 image being scaled twice
    and letterboxed with 20 rows of padding above it.

    Args:
        folder (Path): The folder of the split.
        labels_file (Path): The 'labels.json' file of the split.
        load_instance_masks (bool): Whether the samples carry the instance masks.

    Returns:
        CocoDetectionDataset: The dataset.
    """
    return CocoDetectionDataset(data_directory=str(folder / "data"), labels_file=str(labels_file), image_size=80,
                                keep_ratio=True, label_mode="multi_class", label_dictionary={1: 1, 2: 2},
                                load_instance_masks=load_instance_masks)


def test_instance_masks_follow_the_objects(tmp_path: Path) -> None:
    """
    Verifies that a polygon, an encoded crowd annotation and an encoded object get their masks in the order of their
    boxes, the crowd being an object like the others, that the object too small to cover a pixel centre is removed
    with its box and label, and that the collate function keeps the masks as a list. Without instance masks, the crowd
    and the tiny object stay.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    encoded_mask = np.zeros(shape=(IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    encoded_mask[10:15, 20:30] = 1
    annotations = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 2, 4, 4], "iscrowd": 0,
         "segmentation": [[2.0, 2.0, 6.0, 2.0, 6.0, 6.0, 2.0, 6.0]]},
        {"id": 2, "image_id": 1, "category_id": 2, "bbox": [0, 0, 40, 20], "iscrowd": 1,
         "segmentation": encode_run_length_encoding(mask=np.ones(shape=(IMAGE_HEIGHT, IMAGE_WIDTH)))},
        {"id": 3, "image_id": 1, "category_id": 1, "bbox": [10, 10, 0.1, 0.1], "iscrowd": 0,
         "segmentation": [[10.0, 10.0, 10.1, 10.0, 10.1, 10.1, 10.0, 10.1]]},
        {"id": 4, "image_id": 1, "category_id": 2, "bbox": [20, 10, 10, 5], "iscrowd": 0,
         "segmentation": encode_run_length_encoding(mask=encoded_mask)}]
    labels_file = write_split(folder=tmp_path, annotations=annotations)

    sample = build_dataset(folder=tmp_path, labels_file=labels_file, load_instance_masks=True)[0]
    assert sample["labels"].tolist() == [1, 2, 2]
    assert sample["instance_masks"].shape == (3, 80, 80) and sample["instance_masks"].dtype == torch.uint8

    # Every mask lies in its box, scaled twice and moved down by the 20 rows of padding, the crowd covering the image
    expected_regions = [(24, 32, 4, 12), (20, 60, 0, 80), (40, 50, 40, 60)]
    for mask, box, (first_row, end_row, first_column, end_column) in zip(sample["instance_masks"],
                                                                         sample["bounding_boxes"], expected_regions):
        assert int(mask.sum()) == (end_row - first_row) * (end_column - first_column)
        assert int(mask[first_row:end_row, first_column:end_column].sum()) == int(mask.sum())
        assert box.tolist() == [float(first_column), float(first_row), float(end_column), float(end_row)]

    # The collate function keeps one mask tensor per image
    batch = collate_function(batch=[sample, sample])
    assert isinstance(batch["instance_masks"], list) and len(batch["instance_masks"]) == 2

    # The object detection samples keep the crowd and the tiny object and carry no masks
    detection_sample = build_dataset(folder=tmp_path, labels_file=labels_file, load_instance_masks=False)[0]
    assert detection_sample["labels"].tolist() == [1, 2, 1, 2] and "instance_masks" not in detection_sample
    assert "instance_masks" not in collate_function(batch=[detection_sample])


def test_image_without_objects_gets_empty_masks(tmp_path: Path) -> None:
    """
    Verifies that an image whose only object is too small to cover a pixel centre keeps the [0, S, S] layout of its
    masks, with no box and no label.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    labels_file = write_split(folder=tmp_path, annotations=[
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 0.1, 0.1], "iscrowd": 0,
         "segmentation": [[10.0, 10.0, 10.1, 10.0, 10.1, 10.1, 10.0, 10.1]]}])

    sample = build_dataset(folder=tmp_path, labels_file=labels_file, load_instance_masks=True)[0]
    assert sample["instance_masks"].shape == (0, 80, 80)
    assert sample["bounding_boxes"].shape == (0, 4) and sample["labels"].shape == (0,)


def test_object_selection_keeps_the_crowd_and_leaves_out_the_background() -> None:
    """
    Verifies that select_image_objects, shared by the dataset and the anchor recall, leaves out the categories mapped
    to the background, maps every other category to its class (integer or string key), the crowd annotations included,
    and gives its [x1, y1, x2, y2] box, and that the binary mode gives class 1 to every category.
    """
    annotations = [{"category_id": 3, "bbox": [1.0, 2.0, 10.0, 5.0]},
                   {"category_id": 3, "bbox": [0.0, 0.0, 4.0, 4.0], "iscrowd": 1},
                   {"category_id": 7, "bbox": [5.0, 5.0, 2.0, 2.0]},
                   {"category_id": 9, "bbox": [6.0, 1.0, 3.0, 8.0]}]

    # Multi-class: the unmapped category 7 is left out and the crowd is kept
    boxes, labels, kept_annotations = select_image_objects(annotations=annotations, label_mode="multi_class",
                                                           label_dictionary={3: 1, "9": 2})
    assert boxes == [[1.0, 2.0, 11.0, 7.0], [0.0, 0.0, 4.0, 4.0], [6.0, 1.0, 9.0, 9.0]] and labels == [1, 1, 2]
    assert kept_annotations == [annotations[0], annotations[1], annotations[3]]

    # Binary: every category, the crowd included, becomes class 1
    _, binary_labels, _ = select_image_objects(annotations=annotations, label_mode="binary", label_dictionary={})
    assert binary_labels == [1, 1, 1, 1]


def test_instance_boxes_come_from_the_masks_without_bbox(tmp_path: Path) -> None:
    """
    Verifies that the instance segmentation reads annotations without 'bbox', its boxes being the tight boxes of the
    resized masks of a polygon, an encoded object and a diamond, whose mask box is narrower than its scaled outline,
    that a wrong 'bbox' is ignored, and that a segmentation without any usable part is left out by the loader and by
    the selection of the anchor recall alike.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    encoded_mask = np.zeros(shape=(IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    encoded_mask[3:9, 30:36] = 1
    labels_file = write_split(folder=tmp_path, annotations=[
        {"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0,
         "segmentation": [[4.0, 5.0, 14.0, 5.0, 14.0, 10.0, 4.0, 10.0]]},
        {"id": 2, "image_id": 1, "category_id": 2, "bbox": [0, 0, 1, 1], "iscrowd": 0,
         "segmentation": encode_run_length_encoding(mask=encoded_mask)},
        {"id": 3, "image_id": 1, "category_id": 1, "iscrowd": 0, "segmentation": [[20.0, 2.0, 25.0, 2.0]]},
        {"id": 4, "image_id": 1, "category_id": 2, "iscrowd": 0,
         "segmentation": [[20.0, 11.0, 23.5, 14.5, 20.0, 18.0, 16.5, 14.5]]}])

    # The 40 x 20 image is scaled twice with 20 rows of padding above it, the diamond outline becoming [33, 42, 47, 56]
    sample = build_dataset(folder=tmp_path, labels_file=labels_file, load_instance_masks=True)[0]
    assert sample["labels"].tolist() == [1, 2, 2]
    assert sample["bounding_boxes"][:2].tolist() == [[8.0, 30.0, 28.0, 40.0], [60.0, 26.0, 72.0, 38.0]]
    assert torch.equal(sample["bounding_boxes"], compute_mask_bounding_boxes(instance_masks=sample["instance_masks"]))
    assert sample["bounding_boxes"][2].tolist() != [33.0, 42.0, 47.0, 56.0]

    # The selection reads the boxes from the segmentation in original pixels, without the unusable segmentation
    boxes, labels, _ = select_image_objects(annotations=json.loads(labels_file.read_text())["annotations"],
                                            label_mode="multi_class", label_dictionary={1: 1, 2: 2},
                                            boxes_from_segmentation=True)
    assert boxes == [[4.0, 5.0, 14.0, 10.0], [30.0, 3.0, 36.0, 9.0], [16.5, 11.0, 23.5, 18.0]] and labels == [1, 2, 2]


def test_original_instances_keep_every_object_at_both_resolutions(tmp_path: Path) -> None:
    """
    Verifies that the test samples of the evaluation at the original resolution carry the masks and boxes of their
    objects at the original resolution, in the order of the network ones: an object whose mask vanishes once shrunk
    is kept with its empty mask and its scaled segmentation box, an object without any pixel in the original image is
    removed at both resolutions, the collate function keeps the original masks and boxes as lists, the training samples
    still remove the vanished object, and the original objects are refused without the instance masks.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    encoded_mask = np.zeros(shape=(IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    encoded_mask[10:18, 24:36] = 1
    labels_file = write_split(folder=tmp_path, annotations=[
        {"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0,
         "segmentation": [[4.0, 4.0, 12.0, 4.0, 12.0, 12.0, 4.0, 12.0]]},
        {"id": 2, "image_id": 1, "category_id": 2, "iscrowd": 0,
         "segmentation": [[21.0, 9.0, 22.0, 9.0, 22.0, 10.0, 21.0, 10.0]]},
        {"id": 3, "image_id": 1, "category_id": 1, "iscrowd": 0,
         "segmentation": [[10.6, 10.6, 10.7, 10.6, 10.7, 10.7, 10.6, 10.7]]},
        {"id": 4, "image_id": 1, "category_id": 2, "iscrowd": 0,
         "segmentation": encode_run_length_encoding(mask=encoded_mask)}])

    # The 40 x 20 image is shrunk four times into 10 pixels, with 2 rows of padding above it
    dataset = CocoDetectionDataset(data_directory=str(tmp_path / "data"), labels_file=str(labels_file), image_size=10,
                                   keep_ratio=True, label_mode="multi_class", label_dictionary={1: 1, 2: 2},
                                   load_instance_masks=True, load_original_instances=True)
    sample = dataset[0]
    assert sample["labels"].tolist() == [1, 2, 2]

    # The objects at the original resolution, the one without any pixel removed
    expected_original_masks = np.zeros(shape=(3, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    expected_original_masks[0, 4:12, 4:12] = 1
    expected_original_masks[1, 9, 21] = 1
    expected_original_masks[2] = encoded_mask
    assert sample["original_instance_masks"].dtype == torch.uint8
    assert np.array_equal(sample["original_instance_masks"].numpy(), expected_original_masks)
    assert sample["original_bounding_boxes"].tolist() == [[4.0, 4.0, 12.0, 12.0], [21.0, 9.0, 22.0, 10.0],
                                                          [24.0, 10.0, 36.0, 18.0]]

    # The same objects at the network resolution, the one-pixel object with its empty mask and its scaled box
    network_mask_areas = sample["instance_masks"].flatten(start_dim=1).sum(dim=1).tolist()
    assert sample["instance_masks"].shape == (3, 10, 10) and network_mask_areas[0] == 4
    assert network_mask_areas[1] == 0 and network_mask_areas[2] > 0
    assert sample["bounding_boxes"][1].tolist() == [5.25, 4.25, 5.5, 4.5]
    mask_boxes = compute_mask_bounding_boxes(instance_masks=sample["instance_masks"])
    assert sample["bounding_boxes"][0].tolist() == [1.0, 3.0, 3.0, 5.0]
    assert torch.equal(sample["bounding_boxes"][2], mask_boxes[2])

    # The collate function keeps the original masks and boxes of every image as lists
    batch = collate_function(batch=[sample, sample])
    assert isinstance(batch["original_instance_masks"], list) and len(batch["original_instance_masks"]) == 2
    assert isinstance(batch["original_bounding_boxes"], list) and len(batch["original_bounding_boxes"]) == 2

    # The training samples remove the object whose mask vanished and carry no original objects
    training_dataset = CocoDetectionDataset(data_directory=str(tmp_path / "data"), labels_file=str(labels_file),
                                            image_size=10, keep_ratio=True, label_mode="multi_class",
                                            label_dictionary={1: 1, 2: 2}, load_instance_masks=True)
    training_sample = training_dataset[0]
    assert training_sample["labels"].tolist() == [1, 2] and "original_instance_masks" not in training_sample

    # The original objects are those of the instance segmentation
    with pytest.raises(expected_exception=ValueError, match="needs load_instance_masks"):
        CocoDetectionDataset(data_directory=str(tmp_path / "data"), labels_file=str(labels_file), image_size=10,
                             keep_ratio=True, label_mode="multi_class", label_dictionary={1: 1, 2: 2},
                             load_instance_masks=False, load_original_instances=True)


def test_ignored_regions_are_the_objects_without_labelled_keypoint(tmp_path: Path) -> None:
    """
    Verifies the ignored regions of the keypoint detection: the union, at the network resolution, of the masks of the
    objects of a category with keypoints that have no labelled keypoint (a person drawn by a polygon and a crowd by a
    run length encoding), while the labelled person and the object of a category without keypoints are not ignored;
    that only the labelled person is kept with its keypoints; that the collate stacks the regions of the images; that
    the samples carry no regions without the setting; and that a removed person without segmentation, or the regions
    without the keypoints, are refused.

    Args:
        tmp_path (Path): The pytest temporary folder holding the written split.
    """
    categories = [{"id": 1, "name": "person", "keypoints": ["head", "foot"]}, {"id": 2, "name": "box"}]
    crowd_mask = np.zeros(shape=(IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    crowd_mask[4:16, 26:34] = 1
    annotations = [
        {"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0, "bbox": [2, 2, 8, 8], "num_keypoints": 2,
         "keypoints": [4, 4, 2, 8, 8, 2], "segmentation": [[2, 2, 10, 2, 10, 10, 2, 10]]},
        {"id": 2, "image_id": 1, "category_id": 1, "iscrowd": 0, "bbox": [14, 2, 8, 10], "num_keypoints": 0,
         "keypoints": [0, 0, 0, 0, 0, 0], "segmentation": [[14, 2, 22, 2, 22, 12, 14, 12]]},
        {"id": 3, "image_id": 1, "category_id": 1, "iscrowd": 1, "bbox": [26, 4, 8, 12], "num_keypoints": 0,
         "keypoints": [0, 0, 0, 0, 0, 0], "segmentation": encode_run_length_encoding(mask=crowd_mask)},
        {"id": 4, "image_id": 1, "category_id": 2, "iscrowd": 0, "bbox": [30, 16, 8, 3], "keypoints": [],
         "segmentation": [[30, 16, 38, 16, 38, 19, 30, 19]]}]
    labels_file = write_split(folder=tmp_path, annotations=annotations, categories=categories)
    dataset = CocoDetectionDataset(data_directory=str(tmp_path / "data"), labels_file=str(labels_file), image_size=80,
                                   keep_ratio=True, label_mode="multi_class", label_dictionary={1: 1, 2: 2},
                                   load_keypoints=True, maximum_keypoints=2, load_ignored_regions=True, shuffle=False)
    sample = dataset[0]

    # Only the labelled person is kept, and the regions are those of the unlabelled person and of the crowd
    object_masks = [torch.from_numpy(build_instance_mask(segmentation=annotation["segmentation"],
                                                         image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT,
                                                         image_size=80, keep_ratio=True))
                    for annotation in annotations]
    assert sample["labels"].tolist() == [1] and sample["keypoints"].shape == (1, 2, 3)
    assert sample["ignored_regions"].dtype == torch.uint8 and sample["ignored_regions"].shape == (80, 80)
    assert torch.equal(input=sample["ignored_regions"], other=torch.maximum(object_masks[1], object_masks[2]))
    assert all(bool(object_mask.any()) for object_mask in object_masks)
    assert not (sample["ignored_regions"] * (object_masks[0] + object_masks[3])).any()

    # The collate stacks the regions of the images, and the samples have none without the setting
    assert collate_function(batch=[sample, sample])["ignored_regions"].shape == (2, 80, 80)
    plain_dataset = CocoDetectionDataset(data_directory=str(tmp_path / "data"), labels_file=str(labels_file),
                                         image_size=80, keep_ratio=True, label_mode="multi_class",
                                         label_dictionary={1: 1, 2: 2}, load_keypoints=True, maximum_keypoints=2,
                                         shuffle=False)
    assert "ignored_regions" not in plain_dataset[0]

    # A removed person needs its segmentation, and the regions need the keypoints
    unsegmented_annotations = [{key: value for key, value in annotation.items() if key != "segmentation"}
                               if annotation["id"] == 2 else annotation for annotation in annotations]
    unsegmented_labels_file = write_split(folder=tmp_path / "unsegmented", annotations=unsegmented_annotations,
                                          categories=categories)
    unsegmented_dataset = CocoDetectionDataset(data_directory=str(tmp_path / "unsegmented" / "data"),
                                               labels_file=str(unsegmented_labels_file), image_size=80,
                                               keep_ratio=True, label_mode="multi_class",
                                               label_dictionary={1: 1, 2: 2}, load_keypoints=True,
                                               maximum_keypoints=2, load_ignored_regions=True, shuffle=False)
    with pytest.raises(expected_exception=ValueError, match="has no labelled keypoint and no 'segmentation'"):
        unsegmented_dataset.__getitem__(_index=0)
    with pytest.raises(expected_exception=ValueError, match="needs load_keypoints"):
        CocoDetectionDataset(data_directory=str(tmp_path / "data"), labels_file=str(labels_file), image_size=80,
                             keep_ratio=True, label_mode="multi_class", label_dictionary={1: 1, 2: 2},
                             load_ignored_regions=True)


def test_instance_index_maps_give_the_objects_of_the_instance_masks(tmp_path: Path) -> None:
    """
    Verifies that the samples with instance index maps keep the objects, labels and boxes of the samples with one mask
    per object, the 40 x 20 image being shrunk into 16 pixels: the one pixel object that vanishes is removed with its
    box and label and the objects after it renumbered, the map expanding into the instance masks; that the collate
    function stacks the maps without instance masks; and that the maps refuse a polygon, the original instances and
    the absence of instance masks.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # A rectangle, a one pixel object vanishing at 16 pixels, a crowd band and a compressed object, all disjoint
    object_masks = [np.zeros(shape=(IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8) for _ in range(4)]
    object_masks[0][2:9, 3:15] = 1
    object_masks[1][12, 22] = 1
    object_masks[2][15:20, :] = 1
    object_masks[3][4:12, 25:38] = 1
    annotations = [{"id": index + 1, "image_id": 1, "category_id": 1 + index % 2, "iscrowd": int(index == 2),
                    "segmentation": encode_run_length_encoding(mask=mask, compressed=index == 3)}
                   for index, mask in enumerate(object_masks)]
    labels_file = write_split(folder=tmp_path, annotations=annotations)
    arguments = {"data_directory": str(tmp_path / "data"), "labels_file": str(labels_file), "image_size": 16,
                 "keep_ratio": True, "label_mode": "multi_class", "label_dictionary": {1: 1, 2: 2},
                 "load_instance_masks": True}

    # Same objects, labels and boxes as the per-object masks, the map expanding into those masks
    mask_sample = CocoDetectionDataset(**arguments)[0]
    map_sample = CocoDetectionDataset(**arguments, load_instance_index_maps=True)[0]
    assert mask_sample["labels"].tolist() == [1, 1, 2] and "instance_masks" not in map_sample
    assert torch.equal(map_sample["labels"], mask_sample["labels"])
    assert torch.equal(map_sample["bounding_boxes"], mask_sample["bounding_boxes"])
    assert map_sample["instance_index_maps"].shape == (16, 16)
    assert map_sample["instance_index_maps"].dtype == torch.uint8
    assert sorted(torch.unique(map_sample["instance_index_maps"]).tolist()) == [0, 1, 2, 3]
    expanded_masks = expand_instance_index_maps(index_maps=map_sample["instance_index_maps"][None],
                                                object_counts=[map_sample["labels"].shape[0]])[0]
    assert torch.equal(expanded_masks, mask_sample["instance_masks"])

    # The collate function stacks the maps of the images, without instance masks
    batch = collate_function(batch=[map_sample, map_sample])
    assert batch["instance_index_maps"].shape == (2, 16, 16) and "instance_masks" not in batch

    # A polygon cannot be painted into the map, and the maps go with the instance masks, without the original objects
    polygon_file = write_split(folder=tmp_path / "polygons", annotations=[
        {"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0,
         "segmentation": [[2.0, 2.0, 6.0, 2.0, 6.0, 6.0, 2.0, 6.0]]}])
    with pytest.raises(expected_exception=ValueError, match="polygon"):
        CocoDetectionDataset(**{**arguments, "data_directory": str(tmp_path / "polygons" / "data"),
                                "labels_file": str(polygon_file)}, load_instance_index_maps=True)[0]
    with pytest.raises(expected_exception=ValueError, match="without instance index maps"):
        CocoDetectionDataset(**arguments, load_instance_index_maps=True, load_original_instances=True)
    with pytest.raises(expected_exception=ValueError, match="needs load_instance_masks"):
        CocoDetectionDataset(**{**arguments, "load_instance_masks": False}, load_instance_index_maps=True)


def test_image_records_keep_every_image_with_its_annotations(tmp_path: Path) -> None:
    """
    Verifies that the packed records of the dataset give every image entry with its own annotations, in the order of
    the manifest, and an empty list for an image without any, in the order of the manifest without shuffling and in a
    shuffled order otherwise, the categories of the manifest being kept.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    images = [{"id": 5, "file_name": "five.png", "width": 40, "height": 20},
              {"id": 3, "file_name": "three.png", "width": 40, "height": 20},
              {"id": 8, "file_name": "eight.png", "width": 40, "height": 20}]
    annotations = [{"id": 1, "image_id": 3, "category_id": 1, "iscrowd": 0, "bbox": [1, 2, 3, 4]},
                   {"id": 2, "image_id": 5, "category_id": 2, "iscrowd": 0, "bbox": [5, 6, 7, 8]},
                   {"id": 3, "image_id": 3, "category_id": 2, "iscrowd": 1, "bbox": [9, 10, 11, 12]}]
    categories = [{"id": 1, "name": "first"}, {"id": 2, "name": "second"}]
    labels_file = tmp_path / "labels.json"
    labels_file.write_text(data=json.dumps(obj={"images": images, "annotations": annotations,
                                                "categories": categories}))
    expected_annotations = {5: [annotations[1]], 3: [annotations[0], annotations[2]], 8: []}
    arguments = {"data_directory": str(tmp_path / "data"), "labels_file": str(labels_file), "image_size": 80,
                 "keep_ratio": True, "label_mode": "multi_class", "label_dictionary": {1: 1, 2: 2}}

    # The order of the manifest without shuffling
    dataset = CocoDetectionDataset(**arguments, shuffle=False)
    assert len(dataset) == 3 and dataset.categories == categories
    assert [dataset.get_image_record(image_index=image_index) for image_index in range(3)] == [
        (image, expected_annotations[image["id"]]) for image in images]

    # A shuffled order, every image keeping its own annotations
    random.seed(a=3)
    shuffled_dataset = CocoDetectionDataset(**arguments)
    shuffled_records = [shuffled_dataset.get_image_record(image_index=image_index) for image_index in range(3)]
    assert sorted(image["id"] for image, _ in shuffled_records) == [3, 5, 8]
    assert all(image_annotations == expected_annotations[image["id"]] for image, image_annotations in shuffled_records)


def test_dataset_keeps_no_python_object_per_image(tmp_path: Path) -> None:
    """
    Verifies that the dataset keeps its images and their annotations in its packed records, without any Python
    dictionary or list per image, the objects a forked loader worker copies page by page by reading them
    (copy-on-access): the objects followed by the garbage collector grow by a few objects only for 2000 images.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    number_images = 2000
    labels_file = tmp_path / "labels.json"
    labels_file.write_text(data=json.dumps(obj={
        "images": [{"id": image_index, "file_name": f"{image_index}.png", "width": 40, "height": 20}
                   for image_index in range(number_images)],
        "annotations": [{"id": image_index, "image_id": image_index, "category_id": 1, "iscrowd": 0,
                         "bbox": [1, 2, 3, 4]} for image_index in range(number_images)],
        "categories": [{"id": 1, "name": "first"}]}))

    # Objects followed by the garbage collector before and after the dataset is built, the dataset kept alive
    gc.collect()
    object_count_before = len(gc.get_objects())
    dataset = CocoDetectionDataset(data_directory=str(tmp_path / "data"), labels_file=str(labels_file), image_size=80,
                                   keep_ratio=True, label_mode="multi_class", label_dictionary={1: 1})
    gc.collect()
    assert len(dataset) == number_images
    assert len(gc.get_objects()) - object_count_before < 100


def test_images_sharing_an_identifier_are_refused(tmp_path: Path) -> None:
    """
    Verifies that a manifest whose two images share an identifier is refused, the identifier naming the annotations of
    one image only.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    labels_file = tmp_path / "labels.json"
    labels_file.write_text(data=json.dumps(obj={"images": [{"id": 1, "file_name": "a.png", "width": 40, "height": 20},
                                                           {"id": 1, "file_name": "b.png", "width": 40, "height": 20}],
                                                "annotations": [], "categories": [{"id": 1, "name": "first"}]}))

    with pytest.raises(expected_exception=ValueError, match="The 2 images of .* have only 1 distinct identifiers"):
        CocoDetectionDataset(data_directory=str(tmp_path / "data"), labels_file=str(labels_file), image_size=80,
                             keep_ratio=True, label_mode="multi_class", label_dictionary={1: 1})
