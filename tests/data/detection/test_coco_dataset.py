"""
Tests of the instance masks of CocoDetectionDataset (vision_training/data/detection/coco_dataset.py), read by the
instance segmentation task: the crowd annotations are left out (of the object detection samples too), every other
object gets its mask at the network resolution in the order of its box, the objects whose mask vanishes are removed
with their box and label, an annotation without segmentation is refused, the collate function keeps the masks as a
list, the object detection samples stay without masks, the instance segmentation boxes come from the masks without
any 'bbox', and the selection of the objects of an image maps the categories and leaves out the crowd and the
background.
"""

import json
import cv2
import numpy as np
import pytest
import torch

from pathlib import Path
from typing import Any, Dict, List

from vision_training.data.detection.coco_dataset import CocoDetectionDataset, collate_function, select_image_objects
from vision_training.data.detection.coco_masks import compute_mask_bounding_boxes, encode_run_length_encoding

IMAGE_WIDTH, IMAGE_HEIGHT = 40, 20


def write_split(folder: Path, annotations: List[Dict[str, Any]]) -> Path:
    """
    Writes a one-image COCO split (a black 40 x 20 image and the given annotations) in the layout read by the dataset.

    Args:
        folder (Path): The folder of the split.
        annotations (List[Dict[str, Any]]): The annotations of the image, whose 'image_id' is 1.

    Returns:
        Path: The 'labels.json' file of the split.
    """
    (folder / "data").mkdir(parents=True)
    cv2.imwrite(filename=str(folder / "data" / "image.png"), img=np.zeros(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                                                                           dtype=np.uint8))
    labels_file = folder / "labels.json"
    labels_file.write_text(json.dumps({"images": [{"id": 1, "file_name": "image.png", "width": IMAGE_WIDTH,
                                                   "height": IMAGE_HEIGHT}],
                                       "annotations": annotations,
                                       "categories": [{"id": 1, "name": "first"}, {"id": 2, "name": "second"}]}))
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
    Verifies that a polygon and an encoded object get their masks in the order of their boxes, that the crowd
    annotation and the object too small to cover a pixel centre are removed with their box and label, and that the
    collate function keeps the masks as a list. Without instance masks, the crowd is left out too and the tiny object
    stays.

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
    assert sample["labels"].tolist() == [1, 2]
    assert sample["instance_masks"].shape == (2, 80, 80) and sample["instance_masks"].dtype == torch.uint8

    # Every mask lies in its box, scaled twice and moved down by the 20 rows of padding
    expected_regions = [(24, 32, 4, 12), (40, 50, 40, 60)]
    for mask, box, (first_row, end_row, first_column, end_column) in zip(sample["instance_masks"],
                                                                         sample["bounding_boxes"], expected_regions):
        assert int(mask.sum()) == (end_row - first_row) * (end_column - first_column)
        assert int(mask[first_row:end_row, first_column:end_column].sum()) == int(mask.sum())
        assert box.tolist() == [float(first_column), float(first_row), float(end_column), float(end_row)]

    # The collate function keeps one mask tensor per image
    batch = collate_function(batch=[sample, sample])
    assert isinstance(batch["instance_masks"], list) and len(batch["instance_masks"]) == 2

    # The object detection samples leave the crowd out, keep the tiny object and carry no masks
    detection_sample = build_dataset(folder=tmp_path, labels_file=labels_file, load_instance_masks=False)[0]
    assert detection_sample["labels"].tolist() == [1, 1, 2] and "instance_masks" not in detection_sample
    assert "instance_masks" not in collate_function(batch=[detection_sample])


def test_image_without_objects_gets_empty_masks(tmp_path: Path) -> None:
    """
    Verifies that an image whose only annotation is a crowd keeps the [0, S, S] layout of its masks, with no box and no
    label.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    labels_file = write_split(folder=tmp_path, annotations=[
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 40, 20], "iscrowd": 1,
         "segmentation": encode_run_length_encoding(mask=np.ones(shape=(IMAGE_HEIGHT, IMAGE_WIDTH)))}])

    sample = build_dataset(folder=tmp_path, labels_file=labels_file, load_instance_masks=True)[0]
    assert sample["instance_masks"].shape == (0, 80, 80)
    assert sample["bounding_boxes"].shape == (0, 4) and sample["labels"].shape == (0,)


def test_annotation_without_segmentation_is_refused(tmp_path: Path) -> None:
    """
    Verifies that an object without segmentation stops the instance segmentation loading with the identifier of its
    annotation, instead of training on an object without mask.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    labels_file = write_split(folder=tmp_path, annotations=[
        {"id": 7, "image_id": 1, "category_id": 1, "bbox": [2, 2, 4, 4], "iscrowd": 0}])

    with pytest.raises(ValueError, match="annotation 7 of image 1"):
        _ = build_dataset(folder=tmp_path, labels_file=labels_file, load_instance_masks=True)[0]


def test_object_selection_leaves_out_the_crowd_and_the_background() -> None:
    """
    Verifies that select_image_objects, shared by the dataset and the anchor recall, leaves out the crowd annotations
    and the categories mapped to the background, maps every other category to its class (integer or string key) and
    gives its [x1, y1, x2, y2] box, and that the binary mode gives class 1 to every category.
    """
    annotations = [{"category_id": 3, "bbox": [1.0, 2.0, 10.0, 5.0]},
                   {"category_id": 3, "bbox": [0.0, 0.0, 4.0, 4.0], "iscrowd": 1},
                   {"category_id": 7, "bbox": [5.0, 5.0, 2.0, 2.0]},
                   {"category_id": 9, "bbox": [6.0, 1.0, 3.0, 8.0]}]

    # Multi-class: the crowd and the unmapped category 7 are left out
    boxes, labels, kept_annotations = select_image_objects(annotations=annotations, label_mode="multi_class",
                                                           label_dictionary={3: 1, "9": 2})
    assert boxes == [[1.0, 2.0, 11.0, 7.0], [6.0, 1.0, 9.0, 9.0]] and labels == [1, 2]
    assert kept_annotations == [annotations[0], annotations[3]]

    # Binary: every category except the crowd becomes class 1
    _, binary_labels, _ = select_image_objects(annotations=annotations, label_mode="binary", label_dictionary={})
    assert binary_labels == [1, 1, 1]


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
