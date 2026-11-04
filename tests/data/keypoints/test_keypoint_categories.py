"""
Tests of vision_training/data/keypoints/keypoint_categories.py, the keypoint definitions the configuration loader
injects into 'Data.keypoint_categories' for the keypoint detector, the crop transform, the evaluation and the viewers:
the reading of the COCO 'categories' of a labels.json (the 1-based COCO skeleton turned 0-based, the other category
fields left out, the background and unmapped categories left out, and the merged categories sharing one class).
"""

import json
import pytest

from pathlib import Path
from typing import Any, Dict, List

from vision_training.data.keypoints.keypoint_categories import read_keypoint_categories


def write_categories(folder: Path, categories: List[Dict[str, Any]]) -> Path:
    """
    Writes a COCO 'labels.json' holding only the given categories, the part of the training split read by
    read_keypoint_categories.

    Args:
        folder (Path): The folder of the file.
        categories (List[Dict[str, Any]]): The COCO categories.

    Returns:
        Path: The written 'labels.json'.
    """
    labels_file = folder / "labels.json"
    labels_file.write_text(json.dumps({"images": [], "annotations": [], "categories": categories}))
    return labels_file


def test_categories_are_read_with_their_keypoint_definitions(tmp_path: Path) -> None:
    """
    Verifies that the categories mapped to a class are read with their keypoint names and their skeleton numbered from
    0, and nothing else of the category (a 'maximum_rotation_degrees' left by an older generator is ignored), sorted
    by class, that the categories mapped to the background or absent from the label dictionary are left out, that
    string keys of the label dictionary are accepted, and that two categories merged into one class with the same
    keypoints share it under both names.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    categories = [
        {"id": 4, "name": "rectangle", "keypoints": ["top_left", "top_right", "bottom_right", "bottom_left"],
         "skeleton": [[1, 2], [2, 3], [3, 4], [4, 1]], "maximum_rotation_degrees": 90},
        {"id": 3, "name": "triangle", "keypoints": ["top", "left", "right"], "skeleton": [[1, 2]]},
        {"id": 5, "name": "background_shape", "keypoints": ["centre"]},
        {"id": 6, "name": "unknown_shape"},
        {"id": 7, "name": "square", "keypoints": ["top_left", "top_right", "bottom_right", "bottom_left"]}]
    labels_file = write_categories(folder=tmp_path, categories=categories)
    label_dictionary = {"background": 0, 3: 1, "4": 2, 5: 0, 7: 2}
    keypoint_categories = read_keypoint_categories(labels_file=labels_file, label_dictionary=label_dictionary)
    assert list(keypoint_categories) == [1, 2]

    # The triangle keeps its keypoint names and its 0-based skeleton
    assert keypoint_categories[1] == {"name": "triangle", "keypoints": ["top", "left", "right"], "skeleton": [[0, 1]]}

    # The rectangle and the square share class 2, with the definition of the rectangle read first
    assert keypoint_categories[2]["name"] == "rectangle+square"
    assert keypoint_categories[2] == {"name": "rectangle+square",
                                      "keypoints": ["top_left", "top_right", "bottom_right", "bottom_left"],
                                      "skeleton": [[0, 1], [1, 2], [2, 3], [3, 0]]}
