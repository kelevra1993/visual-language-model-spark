"""
Keypoint definitions of the classes of the keypoint detection task, read from the COCO 'categories' of the training
split: the names of the keypoints of every class and its skeleton. The configuration loader injects them into
'Data.keypoint_categories', from which the keypoint detector sizes its heatmap channels and the evaluation and viewers
name and draw the keypoints. The keypoints keep their names whatever the augmentation of their image.
"""

import json

from pathlib import Path
from typing import Any, Dict


def read_keypoint_categories(labels_file: str | Path, label_dictionary: Dict[Any, int]) -> Dict[int, Dict[str, Any]]:
    """
    Reads the keypoint definition of every class of the keypoint detection task from the 'categories' of a COCO
    'labels.json', called by the configuration loader on the training split so that the model, the evaluation and the
    viewers share one definition. The categories mapped to the background (0) or absent from the label dictionary
    are left out, and categories merged into one class must define the same keypoints.

    Args:
        labels_file (str | Path): The 'labels.json' of the training split.
        label_dictionary (Dict[Any, int]): The class index of every raw category identifier.

    Returns:
        Dict[int, Dict[str, Any]]: For every class index, its 'name', 'keypoints' (names) and 'skeleton' (0-based pairs
            of keypoint positions).

    Raises:
        ValueError: If a category has no keypoint names, or two categories of one class define different keypoints.
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

        # The COCO skeleton numbers the keypoints from 1
        definition = {"name": category["name"], "keypoints": keypoint_names,
                      "skeleton": [[first - 1, second - 1] for first, second in category.get("skeleton", [])]}

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
