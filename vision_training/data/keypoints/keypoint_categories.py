"""
Keypoint definitions of the classes of the keypoint detection task, read from the COCO 'categories' of the training
split: the names of the keypoints of every class, its skeleton, the largest rotation of its crops and its horizontal
flip pairs, deduced from the 'left' and 'right' words of the keypoint names or given by the configuration. The
configuration loader injects them into 'Data.keypoint_categories', from which the keypoint detector sizes its heatmap
channels, the crop transform swaps the flipped keypoints and bounds the rotations, and the evaluation and viewers name
and draw the keypoints.
"""

import json

from pathlib import Path
from typing import Any, Dict, List, Optional

# Largest rotation of the crops of a category whose 'categories' entry gives no 'maximum_rotation_degrees', in degrees:
# no limit beyond the one of the configuration
DEFAULT_MAXIMUM_ROTATION_DEGREES = 180.0

# Side words of the keypoint names exchanged by a horizontal flip, the name with one of them replaced by the other
# being the counterpart of the keypoint
FLIP_SIDE_WORDS = {"left": "right", "right": "left"}


def find_flipped_name(keypoint_name: str) -> Optional[str]:
    """
    Gives the name a keypoint takes once its image is flipped horizontally, used to deduce the flip pairs of a category
    from its keypoint names: every 'left' word of the name becomes 'right' and every 'right' word becomes 'left', the
    words being the parts of the name between underscores.

    Args:
        keypoint_name (str): The name of the keypoint, e.g. 'top_left' or 'left_elbow'.

    Returns:
        Optional[str]: The name with its side words exchanged, e.g. 'top_right', or None when the name has no side word.
    """
    words = keypoint_name.split("_")
    if not any(word in FLIP_SIDE_WORDS for word in words):
        return None
    return "_".join(FLIP_SIDE_WORDS[word] if word in FLIP_SIDE_WORDS else word for word in words)


def deduce_flip_pairs(keypoint_names: List[str]) -> List[List[str]]:
    """
    Deduces the keypoints a horizontal flip of the crop exchanges in a category from their names, for the crop
    augmentation of the keypoint trainer: a keypoint named with a 'left' word pairs with the keypoint of the same name
    with 'right' in its place. A category without side words has no pair, its keypoints keeping their names once
    flipped (the centre of a circle, the corners of an L shape).

    Args:
        keypoint_names (List[str]): The keypoint names of the category, in their order.

    Returns:
        List[List[str]]: The [left name, right name] pairs, each pair listed once.

    Raises:
        ValueError: If a keypoint named with a side word has no counterpart in the category.
    """
    flip_pairs = []
    for keypoint_name in keypoint_names:
        flipped_name = find_flipped_name(keypoint_name=keypoint_name)
        if flipped_name is None:
            continue

        # A side word without counterpart would leave the flipped keypoint under the wrong name
        if flipped_name not in keypoint_names:
            raise ValueError(f"The keypoint '{keypoint_name}' has no counterpart '{flipped_name}' for the horizontal "
                             f"flip; give the flip pairs of its category in Data.crop_augmentation.flip_pairs.")

        # Every pair is listed once, from the keypoint that comes first
        if keypoint_names.index(keypoint_name) < keypoint_names.index(flipped_name):
            flip_pairs.append([keypoint_name, flipped_name])
    return flip_pairs


def build_flip_permutation(keypoint_names: List[str], flip_pairs: List[List[str]], maximum_keypoints: int) -> List[int]:
    """
    Builds the index table renaming the keypoints of a horizontally flipped crop, used by swap_flipped_keypoints of the
    crop transform: the keypoint at position k of the flipped crop is the keypoint at position permutation[k] of the
    original one, the positions beyond the keypoints of the category (padding) keeping their place.

    Args:
        keypoint_names (List[str]): The keypoint names of the category, in their order.
        flip_pairs (List[List[str]]): The [name, name] pairs of keypoints exchanged by the flip.
        maximum_keypoints (int): The number of keypoint positions of the batches, the most keypoints of a category.

    Returns:
        List[int]: The source position of every position, of length maximum_keypoints.
    """
    permutation = list(range(maximum_keypoints))
    for first_name, second_name in flip_pairs:
        first_index, second_index = keypoint_names.index(first_name), keypoint_names.index(second_name)
        permutation[first_index], permutation[second_index] = second_index, first_index
    return permutation


def read_keypoint_categories(labels_file: str | Path, label_dictionary: Dict[Any, int],
                             flip_pairs_override: Dict[str, List[List[str]]]) -> Dict[int, Dict[str, Any]]:
    """
    Reads the keypoint definition of every class of the keypoint detection task from the 'categories' of a COCO
    'labels.json', called by the configuration loader on the training split so that the model, the crop transform, the
    evaluation and the viewers share one definition. The categories mapped to the background (0) or absent from the
    label dictionary are left out, and categories merged into one class must define the same keypoints.

    Args:
        labels_file (str | Path): The 'labels.json' of the training split.
        label_dictionary (Dict[Any, int]): The class index of every raw category identifier.
        flip_pairs_override (Dict[str, List[List[str]]]): The flip pairs of the categories named in
            'Data.crop_augmentation.flip_pairs', replacing the pairs deduced from their keypoint names.

    Returns:
        Dict[int, Dict[str, Any]]: For every class index, its 'name', 'keypoints' (names), 'skeleton' (0-based pairs of
            keypoint positions), 'maximum_rotation_degrees' and 'flip_pairs'.

    Raises:
        ValueError: If a category has no keypoint names, two categories of one class define different keypoints, or a
            flip pair names an unknown keypoint.
    """
    with open(file=labels_file, mode="r") as file_handler:
        categories = json.load(fp=file_handler)["categories"]

    keypoint_categories = {}
    for category in categories:
        # Categories without class are not learnt
        if category["id"] in label_dictionary:
            class_index = label_dictionary[category["id"]]
        elif str(category["id"]) in label_dictionary:
            class_index = label_dictionary[str(category["id"])]
        else:
            continue
        if class_index == 0:
            continue
        if "keypoints" not in category or not category["keypoints"]:
            raise ValueError(f"The category '{category['name']}' of {labels_file} has no 'keypoints' names, which the "
                             f"keypoint detection needs for every class.")
        keypoint_names = list(category["keypoints"])

        # Flip pairs of the configuration first, otherwise deduced from the side words of the names
        if category["name"] in flip_pairs_override:
            flip_pairs = [list(flip_pair) for flip_pair in flip_pairs_override[category["name"]]]
            unknown_names = [name for flip_pair in flip_pairs for name in flip_pair if name not in keypoint_names]
            if unknown_names:
                raise ValueError(f"The flip pairs of the category '{category['name']}' name unknown keypoints "
                                 f"{unknown_names}, its keypoints being {keypoint_names}.")
        else:
            flip_pairs = deduce_flip_pairs(keypoint_names=keypoint_names)

        # The COCO skeleton numbers the keypoints from 1
        definition = {"name": category["name"], "keypoints": keypoint_names,
                      "skeleton": [[first - 1, second - 1] for first, second in category.get("skeleton", [])],
                      "maximum_rotation_degrees": float(category.get("maximum_rotation_degrees",
                                                                     DEFAULT_MAXIMUM_ROTATION_DEGREES)),
                      "flip_pairs": flip_pairs}

        # Categories merged into one class share its heatmap channels, so they must define the same keypoints
        if class_index in keypoint_categories:
            if keypoint_categories[class_index]["keypoints"] != keypoint_names:
                raise ValueError(f"The categories '{keypoint_categories[class_index]['name']}' and "
                                 f"'{category['name']}' are merged into class {class_index} but define different "
                                 f"keypoints.")
            keypoint_categories[class_index]["name"] += f"+{category['name']}"
            continue
        keypoint_categories[class_index] = definition
    return dict(sorted(keypoint_categories.items()))
