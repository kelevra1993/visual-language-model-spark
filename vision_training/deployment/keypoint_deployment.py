"""
Deployment of the keypoint detection task (KPT): the description of the heatmap channels of an exported keypoint
detector, written by deployment/export_model.py as 'keypoint_channels.json' next to the model of every format, from the
'Data.keypoint_categories' of the run (the keypoint names and skeleton of every class, read from the training
labels.json). A client of the exported model reads it back (read_keypoint_channels) to group the candidates of every
channel into objects with models/keypoints/keypoint_grouping.py:group_keypoint_candidates and to name and draw their
keypoints, without the training labels.
"""

import json
import numpy as np

from pathlib import Path
from typing import Any, Dict

from vision_training.models.keypoints.keypoint_detector import build_keypoint_channel_table

# Description of the heatmap channels of an exported keypoint detector, written next to the model of every format
KEYPOINT_CHANNELS_FILE_NAME = "keypoint_channels.json"


def build_keypoint_channels_description(keypoint_categories: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Describes the heatmap and tag channels of a keypoint detector for its export, in the channel order of
    build_keypoint_channel_table (the keypoints of every class in their order, the classes in increasing index order),
    so that a client groups and names the candidates of the exported model without the training labels.

    Args:
        keypoint_categories (Dict[Any, Dict[str, Any]]): The 'Data.keypoint_categories' of the run, the 'name',
            'keypoints' names and 'skeleton' (0-based pairs of keypoint positions) of every class index.

    Returns:
        Dict[str, Any]: The 'classes', one {'class_index', 'class_name', 'keypoints', 'skeleton', 'channels'} per class
            with keypoints in class index order, 'channels' being the channel of every keypoint of the class, and the
            'channels', one {'channel', 'class_index', 'class_name', 'keypoint_index', 'keypoint_name'} per channel in
            channel order.

    Raises:
        ValueError: If no class defines keypoints (build_keypoint_channel_table).
    """
    channel_table, _ = build_keypoint_channel_table(keypoint_categories=keypoint_categories)

    # Every class with its keypoints, skeleton and channels, and every channel with its class and keypoint
    classes, channels = [], []
    for class_index in sorted(keypoint_categories, key=int):
        definition = keypoint_categories[class_index]
        class_channels = [int(channel) for channel in channel_table[int(class_index), :len(definition["keypoints"])]]
        classes.append({"class_index": int(class_index), "class_name": definition["name"],
                        "keypoints": list(definition["keypoints"]),
                        "skeleton": [list(pair) for pair in definition["skeleton"]], "channels": class_channels})
        channels += [{"channel": channel, "class_index": int(class_index), "class_name": definition["name"],
                      "keypoint_index": keypoint_index, "keypoint_name": keypoint_name}
                     for keypoint_index, (channel, keypoint_name) in enumerate(zip(class_channels,
                                                                                   definition["keypoints"]))]
    return {"classes": classes, "channels": channels}


def write_keypoint_channels_file(model_folder: Path, keypoint_categories: Dict[Any, Dict[str, Any]]) -> Path:
    """
    Writes the description of the channels of an exported keypoint detector (build_keypoint_channels_description) as
    KEYPOINT_CHANNELS_FILE_NAME in the version folder of the model, next to its 'preprocessing.json'.

    Args:
        model_folder (Path): The version folder of the exported model, '<model name>/<version>/'.
        keypoint_categories (Dict[Any, Dict[str, Any]]): The 'Data.keypoint_categories' of the run.

    Returns:
        Path: The written file.
    """
    channels_path = model_folder / KEYPOINT_CHANNELS_FILE_NAME
    channels_path.write_text(data=json.dumps(obj=build_keypoint_channels_description(
        keypoint_categories=keypoint_categories), indent=2) + "\n")
    return channels_path


def read_keypoint_channels(channels_path: Path) -> Dict[str, Any]:
    """
    Reads the channels of an exported keypoint detector from its KEYPOINT_CHANNELS_FILE_NAME, for a client grouping the
    candidates of the model without the training labels: the keypoint definitions of every class, the channel table
    that group_keypoint_candidates reads, rebuilt with build_keypoint_channel_table, and the name of every channel,
    the channels of the file being checked against the rebuilt table.

    Args:
        channels_path (Path): The 'keypoint_channels.json' of the exported model.

    Returns:
        Dict[str, Any]: The 'keypoint_categories' ({class index: {'name', 'keypoints', 'skeleton'}}, the format of
            'Data.keypoint_categories'), the [number of classes + 1, most keypoints of a class] int64 'channel_table'
            (-1 for the padding positions and the classes without keypoints) and the 'channel_names'
            ('class.keypoint' of every channel).

    Raises:
        ValueError: If the channels listed in the file differ from those of its classes.
    """
    description = json.loads(s=channels_path.read_text())
    keypoint_categories = {class_record["class_index"]: {"name": class_record["class_name"],
                                                         "keypoints": class_record["keypoints"],
                                                         "skeleton": class_record["skeleton"]}
                           for class_record in description["classes"]}
    channel_table, channel_names = build_keypoint_channel_table(keypoint_categories=keypoint_categories)

    # The channels of the file must be those the classes give, in the same order
    listed_names = [f"{channel['class_name']}.{channel['keypoint_name']}" for channel in description["channels"]]
    listed_channels = [channel["channel"] for channel in description["channels"]]
    if listed_names != channel_names or listed_channels != list(range(len(channel_names))):
        raise ValueError(f"The channels of {channels_path} differ from those of its classes: the file was edited or "
                         f"written by another version of the repository.")
    return {"keypoint_categories": keypoint_categories, "channel_table": channel_table.numpy().astype(np.int64),
            "channel_names": channel_names}
