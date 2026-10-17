"""
Tests of vision_training/data/keypoints/keypoint_categories.py, the keypoint definitions the configuration loader
injects into 'Data.keypoint_categories' for the keypoint detector, the crop transform, the evaluation and the viewers:
the flipped name of a keypoint, the flip pairs deduced from the 'left' and 'right' words of the names (a side word
without counterpart refused), the permutation renaming the keypoints of a flipped crop, and the reading of the COCO
'categories' of a labels.json (the 1-based COCO skeleton turned 0-based, the default rotation limit of 180 degrees,
the flip pairs of the configuration replacing the deduced ones, the background and unmapped categories left out, the
merged categories sharing one class, and the errors listed in its docstring).
"""

import json
import pytest

from pathlib import Path
from typing import Any, Dict, List

from vision_training.data.keypoints.keypoint_categories import (DEFAULT_MAXIMUM_ROTATION_DEGREES,
                                                                build_flip_permutation, deduce_flip_pairs,
                                                                find_flipped_name, read_keypoint_categories)


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


def test_flipped_names_exchange_their_side_words() -> None:
    """
    Verifies that every 'left' word of a keypoint name becomes 'right' and the reverse, only whole words between
    underscores counting, and that a name without side word has no flipped name.
    """
    assert find_flipped_name(keypoint_name="top_left") == "top_right"
    assert find_flipped_name(keypoint_name="left_elbow") == "right_elbow"
    assert find_flipped_name(keypoint_name="right") == "left"
    assert find_flipped_name(keypoint_name="left_to_right") == "right_to_left"
    assert find_flipped_name(keypoint_name="centre") is None
    assert find_flipped_name(keypoint_name="leftmost") is None


def test_flip_pairs_are_deduced_from_the_names() -> None:
    """
    Verifies that the flip pairs of a category are listed once each, from the keypoint that comes first, that a
    category without side words has none, and that a side word without counterpart is refused.
    """
    assert deduce_flip_pairs(keypoint_names=["top_left", "top_right", "bottom_right", "bottom_left", "centre"]) == [
        ["top_left", "top_right"], ["bottom_right", "bottom_left"]]
    assert deduce_flip_pairs(keypoint_names=["arm_top_outer", "inner_corner", "outer_corner"]) == []
    with pytest.raises(ValueError, match="'left_hand' has no counterpart 'right_hand'"):
        deduce_flip_pairs(keypoint_names=["head", "left_hand"])


def test_flip_permutation_swaps_the_pairs_and_keeps_the_padding() -> None:
    """
    Verifies that the permutation of a flipped crop gives every keypoint of a pair the position of the other one, the
    other keypoints and the padding positions beyond the keypoints of the category keeping their place.
    """
    permutation = build_flip_permutation(keypoint_names=["top", "left", "right", "bottom_left", "bottom_right"],
                                         flip_pairs=[["left", "right"], ["bottom_left", "bottom_right"]],
                                         maximum_keypoints=7)
    assert permutation == [0, 2, 1, 4, 3, 5, 6]
    assert build_flip_permutation(keypoint_names=["centre"], flip_pairs=[], maximum_keypoints=3) == [0, 1, 2]


def test_categories_are_read_with_their_keypoint_definitions(tmp_path: Path) -> None:
    """
    Verifies that the categories mapped to a class are read with their keypoint names, their skeleton numbered from 0,
    their rotation limit (180 degrees by default) and their flip pairs (deduced, or those of the configuration), sorted
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
    keypoint_categories = read_keypoint_categories(labels_file=labels_file, label_dictionary=label_dictionary,
                                                   flip_pairs_override={"triangle": [["top", "top"]]})
    assert list(keypoint_categories) == [1, 2]

    # The triangle keeps the flip pairs of the configuration, the default rotation limit and its 0-based skeleton
    assert keypoint_categories[1] == {"name": "triangle", "keypoints": ["top", "left", "right"], "skeleton": [[0, 1]],
                                      "maximum_rotation_degrees": DEFAULT_MAXIMUM_ROTATION_DEGREES,
                                      "flip_pairs": [["top", "top"]]}
    assert DEFAULT_MAXIMUM_ROTATION_DEGREES == 180.0

    # The rectangle and the square share class 2, with the definition of the rectangle read first
    assert keypoint_categories[2]["name"] == "rectangle+square"
    assert keypoint_categories[2]["skeleton"] == [[0, 1], [1, 2], [2, 3], [3, 0]]
    assert keypoint_categories[2]["maximum_rotation_degrees"] == 90.0
    assert isinstance(keypoint_categories[2]["maximum_rotation_degrees"], float)
    assert keypoint_categories[2]["flip_pairs"] == [["top_left", "top_right"], ["bottom_right", "bottom_left"]]


def test_inconsistent_categories_are_refused(tmp_path: Path) -> None:
    """
    Verifies the errors of the reading: a category of a class without keypoint names, two categories merged into one
    class with different keypoints, and flip pairs of the configuration naming an unknown keypoint.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    labels_file = write_categories(folder=tmp_path, categories=[{"id": 1, "name": "circle", "keypoints": []}])
    with pytest.raises(ValueError, match="'circle' .* has no 'keypoints' names"):
        read_keypoint_categories(labels_file=labels_file, label_dictionary={1: 1}, flip_pairs_override={})

    labels_file = write_categories(folder=tmp_path, categories=[{"id": 1, "name": "square", "keypoints": ["centre"]},
                                                                {"id": 2, "name": "circle", "keypoints": ["middle"]}])
    with pytest.raises(ValueError, match="'square' and 'circle' are merged into class 1"):
        read_keypoint_categories(labels_file=labels_file, label_dictionary={1: 1, 2: 1}, flip_pairs_override={})

    labels_file = write_categories(folder=tmp_path, categories=[{"id": 1, "name": "bar", "keypoints": ["a", "b"]}])
    with pytest.raises(ValueError, match=r"name unknown keypoints \['c'\]"):
        read_keypoint_categories(labels_file=labels_file, label_dictionary={1: 1},
                                 flip_pairs_override={"bar": [["a", "c"]]})
