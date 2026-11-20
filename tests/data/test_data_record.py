"""
Tests of vision_training/data/data_record.py, the record of the images of every split written by the base trainer in
the run folder ('data.json'): the images of the COCO and sample manifests, the classes counted with their categories,
the class records of every split, and the splits found changed between two starts of a run.
"""

import json
import pytest

from pathlib import Path
from typing import Any, Dict, List

from vision_training.data.data_record import (COMPARED_SPLIT_ENTRIES, RELABELLED_SPLIT_ENTRIES, build_class_records,
                                              count_manifest_classes, find_changed_splits, list_manifest_images,
                                              map_categories_to_classes)


@pytest.mark.parametrize(argnames=["labels", "expected_images"], argvalues=[
    ({"images": [{"id": 1, "file_name": "a.png"}, {"id": 2, "file_name": "b.png"}], "annotations": []},
     ["a.png", "b.png"]),
    ({"samples": [{"id": 1, "file_name": "cats/1.jpg", "category_id": 1}], "categories": []}, ["cats/1.jpg"])])
def test_images_of_every_manifest_format(labels: Dict[str, Any], expected_images: List[str]) -> None:
    """
    Verifies that the images of a split are read in manifest order from the COCO images and the samples.

    Args:
        labels (Dict[str, Any]): The labels file of the split.
        expected_images (List[str]): The images it lists.
    """
    assert list_manifest_images(labels=labels) == expected_images


@pytest.mark.parametrize(argnames=["labels", "expected_counts"], argvalues=[
    ({"images": [{"id": 1, "file_name": "a.png"}, {"id": 2, "file_name": "b.png"}],
      "annotations": [{"image_id": 1, "category_id": 1}, {"image_id": 1, "category_id": 1},
                      {"image_id": 2, "category_id": 2}, {"image_id": 2, "category_id": 9}]},
     {"images": {0: 1, 1: 1}, "annotations": {0: 2, 1: 1}})])
def test_classes_counted_in_every_manifest_format(labels: Dict[str, Any], expected_counts: Dict[str, Any]) -> None:
    """
    Verifies the counts of the classes of a COCO manifest, categories 1 and 2 being classes 0 and 1 and category 9
    unmapped: the images holding the class and its annotations.

    Args:
        labels (Dict[str, Any]): The labels file of the split.
        expected_counts (Dict[str, Any]): The number of units of every class index, by unit.
    """
    class_counts = count_manifest_classes(labels=labels, category_classes={1: 0, 2: 1, 3: 1})
    assert {unit: dict(counts) for unit, counts in class_counts.items()} == expected_counts


def test_binary_mode_maps_every_category_to_the_foreground() -> None:
    """
    Verifies that every manifest category is the foreground class in binary mode, that the categories the label
    dictionary does not map are left out in multi-class mode, and that so are the categories mapped to the background
    class of a detector, the background having no category, while a classifier counts its class 0.
    """
    categories = [{"id": 1, "name": "car"}, {"id": 2, "name": "person"}]
    assert map_categories_to_classes(categories=categories, label_dictionary={"background": 0, "foreground": 1},
                                     label_mode="binary", unmapped_class_index=None) == {1: 1, 2: 1}
    assert map_categories_to_classes(categories=categories, label_dictionary={"background": 0, 2: 1},
                                     label_mode="multi_class", unmapped_class_index=None) == {2: 1}
    assert map_categories_to_classes(categories=categories, label_dictionary={"background": 0, 1: 0, 2: 1},
                                     label_mode="multi_class", unmapped_class_index=None) == {2: 1}
    assert map_categories_to_classes(categories=categories, label_dictionary={1: 0, 2: 1},
                                     label_mode="multi_class", unmapped_class_index=None) == {1: 0, 2: 1}


def test_unmapped_frames_are_counted_in_the_unselected_class(tmp_path: Path) -> None:
    """
    Verifies the classes of the record of a frame selection run whose label mapping leaves the star out: the loader
    giving the star frames to the unselected class, the star category is listed and counted in the unselected class.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    labels = {"categories": [{"id": 1, "name": "unselected"}, {"id": 2, "name": "circle"},
                             {"id": 3, "name": "triangle"}, {"id": 4, "name": "star"}],
              "sequences": [{"frames": ["0.png", "1.png", "2.png", "3.png", "4.png", "5.png"],
                             "frame_category_ids": [1, 2, 1, 3, 4, 1]}]}
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "labels.json").write_text(data=json.dumps(obj=labels))

    class_records = build_class_records(class_names=["unselected", "circle", "triangle"],
                                        label_dictionary={1: 0, 2: 1, 3: 2}, label_mode="multi_class",
                                        unmapped_class_index=0, split_data_folders={"train": tmp_path / "train"})
    assert [(class_record["category_ids"], class_record["frames"]) for class_record in class_records] == [
        ([1, 4], {"train": 4}), ([2], {"train": 1}), ([3], {"train": 1})]


def test_class_records_count_every_split(tmp_path: Path) -> None:
    """
    Verifies the classes of the record of a detection run: the background first without category, the categories
    merged into a class, the category named by a split other than the first, and a split without any object counted
    as zero in every unit.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # A training split of two categories merged into one class, a validation split naming a third one, an empty test
    split_labels = {
        "train": {"categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "truck"}],
                  "images": [{"id": 1, "file_name": "1.png"}, {"id": 2, "file_name": "2.png"}],
                  "annotations": [{"image_id": 1, "category_id": 1}, {"image_id": 1, "category_id": 2},
                                  {"image_id": 2, "category_id": 1}]},
        "validation": {"categories": [{"id": 3, "name": "person"}], "images": [{"id": 1, "file_name": "3.png"}],
                       "annotations": [{"image_id": 1, "category_id": 3}]},
        "test": {"categories": [], "images": [], "annotations": []}}
    for split_name, labels in split_labels.items():
        (tmp_path / split_name).mkdir()
        (tmp_path / split_name / "labels.json").write_text(data=json.dumps(obj=labels))

    # One record per class in class index order, with its images and objects counted in every split
    class_records = build_class_records(
        class_names=["background", "car+truck", "person"], label_dictionary={"background": 0, 1: 1, 2: 1, 3: 2},
        label_mode="multi_class", unmapped_class_index=None,
        split_data_folders={split_name: tmp_path / split_name for split_name in split_labels})
    assert class_records == [
        {"class_index": 0, "class_name": "background", "category_ids": [],
         "images": {"train": 0, "validation": 0, "test": 0}, "annotations": {"train": 0, "validation": 0, "test": 0}},
        {"class_index": 1, "class_name": "car+truck", "category_ids": [1, 2],
         "images": {"train": 2, "validation": 0, "test": 0}, "annotations": {"train": 3, "validation": 0, "test": 0}},
        {"class_index": 2, "class_name": "person", "category_ids": [3],
         "images": {"train": 0, "validation": 1, "test": 0}, "annotations": {"train": 0, "validation": 1, "test": 0}}]


def test_changed_splits_are_found() -> None:
    """
    Verifies that a split is changed when its labels file or its data folder changed, or when the previous record did
    not have it, the date of the records and the lists of images being left out of the comparison, and that only the
    changed labels file and the new split refuse a resume, a moved data folder resuming.
    """
    previous_record = {"created": "2026-10-15 10:00:00", "splits": {
        "train": {"data_folder": "/data/train", "labels_sha256": "a", "number_images": 2, "images": ["1", "2"]},
        "validation": {"data_folder": "/data/validation", "labels_sha256": "b", "number_images": 1, "images": ["3"]}}}
    current_record = json.loads(s=json.dumps(obj=previous_record))
    current_record["created"] = "2026-10-16 10:00:00"
    assert find_changed_splits(previous_record=previous_record, current_record=current_record,
                               compared_entries=COMPARED_SPLIT_ENTRIES) == []

    # A new labels file, a moved data folder and a split the previous record did not have
    current_record["splits"]["train"]["labels_sha256"] = "c"
    current_record["splits"]["validation"]["data_folder"] = "/other/validation"
    current_record["splits"]["test"] = {"data_folder": "/data/test", "labels_sha256": "d", "number_images": 0,
                                        "images": []}
    assert find_changed_splits(previous_record=previous_record, current_record=current_record,
                               compared_entries=COMPARED_SPLIT_ENTRIES) == ["train", "validation", "test"]

    # Only the new labels file and the new split refuse a resume, the moved data folder resuming
    assert find_changed_splits(previous_record=previous_record, current_record=current_record,
                               compared_entries=RELABELLED_SPLIT_ENTRIES) == ["train", "test"]
