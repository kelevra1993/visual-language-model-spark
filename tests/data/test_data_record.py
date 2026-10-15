"""
Tests of vision_training/data/data_record.py, the record of the images of every split written by the base trainer in
the run folder ('data.json'): the images of the four manifest formats, the record of a split with the SHA-256 of its
labels file, and the splits found changed between two starts of a run.
"""

import hashlib
import json
import pytest

from pathlib import Path
from typing import Any, Dict, List

from vision_training.data.data_record import build_split_record, find_changed_splits, list_manifest_images


@pytest.mark.parametrize(argnames=["labels", "expected_images"], argvalues=[
    ({"images": [{"id": 1, "file_name": "a.png"}, {"id": 2, "file_name": "b.png"}], "annotations": []},
     ["a.png", "b.png"]),
    ({"samples": [{"id": 1, "file_name": "cats/1.jpg", "category_id": 1}], "categories": []}, ["cats/1.jpg"]),
    ({"sequences": [{"name": "first", "frames": ["first/0.png", "first/1.png"]}, {"name": "second",
                                                                                 "frames": ["second/0.png"]}]},
     ["first/0.png", "first/1.png", "second/0.png"]),
    ({"pairs": [{"before": ["pair/before_0.png"], "after": ["pair/after_0.png", "pair/after_1.png"]}]},
     ["pair/before_0.png", "pair/after_0.png", "pair/after_1.png"])])
def test_images_of_every_manifest_format(labels: Dict[str, Any], expected_images: List[str]) -> None:
    """
    Verifies that the images of a split are read in manifest order from the COCO images, the samples, the frames of
    the sequences and the frames of both members of the pairs, the before member first.

    Args:
        labels (Dict[str, Any]): The labels file of the split.
        expected_images (List[str]): The images it lists.
    """
    assert list_manifest_images(labels=labels) == expected_images


def test_unknown_manifest_format_fails_loudly() -> None:
    """
    Verifies that a labels file without any of the four lists of the framework stops the record with its keys.
    """
    with pytest.raises(expected_exception=ValueError, match=r"its keys are \['items'\]"):
        list_manifest_images(labels={"items": []})


def test_split_record_identifies_its_labels_file(tmp_path: Path) -> None:
    """
    Verifies the record of a split: its absolute data folder, the SHA-256 of the bytes of its labels file and its
    images with their number.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    split_folder = tmp_path / "train"
    split_folder.mkdir()
    (split_folder / "labels.json").write_text(data=json.dumps(obj={"samples": [{"id": 1, "file_name": "1.png"},
                                                                               {"id": 2, "file_name": "2.png"}]}))
    split_record = build_split_record(split_data_folder=split_folder)
    assert split_record == {"data_folder": str(split_folder.resolve()),
                            "labels_sha256": hashlib.sha256((split_folder / "labels.json").read_bytes()).hexdigest(),
                            "number_images": 2, "images": ["1.png", "2.png"]}


def test_changed_splits_are_found() -> None:
    """
    Verifies that a split is changed when its labels file or its data folder changed, or when the previous record did
    not have it, the date of the records and the lists of images being left out of the comparison.
    """
    previous_record = {"created": "2026-10-15 10:00:00", "splits": {
        "train": {"data_folder": "/data/train", "labels_sha256": "a", "number_images": 2, "images": ["1", "2"]},
        "validation": {"data_folder": "/data/validation", "labels_sha256": "b", "number_images": 1, "images": ["3"]}}}
    current_record = json.loads(s=json.dumps(obj=previous_record))
    current_record["created"] = "2026-10-16 10:00:00"
    assert find_changed_splits(previous_record=previous_record, current_record=current_record) == []

    # A new labels file, a moved data folder and a split the previous record did not have
    current_record["splits"]["train"]["labels_sha256"] = "c"
    current_record["splits"]["validation"]["data_folder"] = "/other/validation"
    current_record["splits"]["test"] = {"data_folder": "/data/test", "labels_sha256": "d", "number_images": 0,
                                        "images": []}
    assert find_changed_splits(previous_record=previous_record, current_record=current_record) == [
        "train", "validation", "test"]
