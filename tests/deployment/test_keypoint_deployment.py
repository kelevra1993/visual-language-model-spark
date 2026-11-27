"""
Tests of the channel description of the exported keypoint detectors (vision_training/deployment/keypoint_deployment.py):
'keypoint_channels.json' written from the keypoint categories of a run, read back by a client into the keypoint
definitions, the channel table of group_keypoint_candidates and the channel names, without the training labels, and an
edited file refused.
"""

import json
import numpy as np
import pytest

from pathlib import Path

from vision_training.deployment.keypoint_deployment import (KEYPOINT_CHANNELS_FILE_NAME, read_keypoint_channels,
                                                            write_keypoint_channels_file)
from vision_training.models.keypoints.keypoint_detector import build_keypoint_channel_table

# Keypoint categories of a run of two classes with keypoints, class 2 having none, as 'Data.keypoint_categories' holds
TEST_KEYPOINT_CATEGORIES = {1: {"name": "square", "keypoints": ["top_left", "bottom_right"], "skeleton": [[0, 1]]},
                            3: {"name": "triangle", "keypoints": ["apex", "left", "right"],
                                "skeleton": [[0, 1], [1, 2], [2, 0]]}}


def test_channels_are_written_and_read_back_without_the_training_labels(tmp_path: Path) -> None:
    """
    Verifies that the channels written next to an exported keypoint detector name the class, keypoint and skeleton of
    every channel in the order of build_keypoint_channel_table, and that a client reads back the keypoint definitions,
    the channel table and the channel names of the run from the file alone.

    Args:
        tmp_path (Path): The pytest temporary directory, standing for the version folder of the exported model.
    """
    channels_path = write_keypoint_channels_file(model_folder=tmp_path, keypoint_categories=TEST_KEYPOINT_CATEGORIES)
    assert channels_path == tmp_path / KEYPOINT_CHANNELS_FILE_NAME

    # Every channel with its class and keypoint, every class with its skeleton and channels
    description = json.loads(s=channels_path.read_text())
    assert [(channel["channel"], channel["class_name"], channel["keypoint_name"])
            for channel in description["channels"]] == [(0, "square", "top_left"), (1, "square", "bottom_right"),
                                                        (2, "triangle", "apex"), (3, "triangle", "left"),
                                                        (4, "triangle", "right")]
    assert [(class_record["class_index"], class_record["skeleton"], class_record["channels"])
            for class_record in description["classes"]] == [(1, [[0, 1]], [0, 1]),
                                                             (3, [[0, 1], [1, 2], [2, 0]], [2, 3, 4])]

    # The client gets the definitions, the table and the names of the run
    channels = read_keypoint_channels(channels_path=channels_path)
    channel_table, channel_names = build_keypoint_channel_table(keypoint_categories=TEST_KEYPOINT_CATEGORIES)
    assert channels["keypoint_categories"] == TEST_KEYPOINT_CATEGORIES
    assert np.array_equal(a1=channels["channel_table"], a2=channel_table.numpy())
    assert channels["channel_names"] == channel_names


def test_edited_channels_are_refused(tmp_path: Path) -> None:
    """
    Verifies that a channel file whose channels no longer follow its classes (two channels swapped) is refused, so
    that a client never groups the candidates of a channel under another keypoint.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    channels_path = write_keypoint_channels_file(model_folder=tmp_path, keypoint_categories=TEST_KEYPOINT_CATEGORIES)
    description = json.loads(s=channels_path.read_text())
    description["channels"][0], description["channels"][1] = description["channels"][1], description["channels"][0]
    channels_path.write_text(data=json.dumps(obj=description))
    with pytest.raises(expected_exception=ValueError, match="differ from those of its classes"):
        read_keypoint_channels(channels_path=channels_path)
