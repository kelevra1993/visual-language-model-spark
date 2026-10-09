"""
Counts the samples of a dataset split from its 'labels.json' (images and annotations, classification samples,
sequences and windows, pairs and kept pairs), printed by the task trainers at startup to describe the splits of a
run.
"""

import os

from pathlib import Path
from typing import Any, Dict, Tuple

from vision_training.data.sequence.sequence_dataset import compute_window_starts
from vision_training.data.sequence.sequence_pair_dataset import PAIR_MEMBERS, compute_member_frame_indices
from vision_training.utilities.os_utilities import read_json


def count_split_images_and_annotations(split_data_folder: str | Path) -> Tuple[int, int]:
    """
    Counts the images and annotations of a dataset split from its COCO 'labels.json', printed by the Trainer at
    startup. The file is read directly because the DALI loader rounds its length up to a whole number of batches,
    so neither loader length gives the exact number of images.

    Args:
        split_data_folder (str | Path): The root folder of the split, containing a 'labels.json'.

    Returns:
        Tuple[int, int]: The number of images, and the number of raw annotations (before label mapping, 0 for an
            unlabelled split without an 'annotations' list).
    """
    labels = read_json(path=os.path.join(str(split_data_folder), "labels.json"))
    number_annotations = len(labels["annotations"]) if "annotations" in labels else 0
    return len(labels["images"]), number_annotations


def count_split_samples(split_data_folder: str | Path) -> Tuple[int, int]:
    """
    Counts the samples and categories of a classification split from its 'labels.json' manifest, printed by the
    trainer at startup.

    Args:
        split_data_folder (str | Path): The root folder of the split, containing a 'labels.json'.

    Returns:
        Tuple[int, int]: The number of samples (before label mapping) and the number of categories of the manifest.
    """
    labels = read_json(path=os.path.join(str(split_data_folder), "labels.json"))
    return len(labels["samples"]), len(labels["categories"])


def count_split_sequences(split_data_folder: str | Path, sequence_settings: Dict[str, Any],
                          whole_sequences: bool) -> Tuple[int, int, int]:
    """
    Counts the sequences, frames and windows of a sequence split from its 'labels.json' manifest, printed by the
    sequence trainers at startup. The windows, cut as the sequence dataset does (one window per whole sequence for
    the frame selection), are the samples of an epoch.

    Args:
        split_data_folder (str | Path): The root folder of the split, containing a 'labels.json'.
        sequence_settings (Dict[str, Any]): The 'Data.sequence_settings' section.
        whole_sequences (bool): Whether every sequence is one window (frame selection, without 'window_stride' nor
            'short_sequences'), or is cut into windows.

    Returns:
        Tuple[int, int, int]: The number of sequences, of frames and of windows (before label mapping).
    """
    labels = read_json(path=os.path.join(str(split_data_folder), "labels.json"))
    number_frames, number_windows = 0, 0
    for sequence in labels["sequences"]:
        number_frames += len(sequence["frames"])

        # One window per whole sequence, or the windows cut every window stride frames
        if whole_sequences:
            number_windows += 1
        else:
            number_windows += len(compute_window_starts(
                number_frames=len(sequence["frames"]), sequence_length=sequence_settings["sequence_length"],
                window_stride=sequence_settings["window_stride"],
                short_sequences=sequence_settings["short_sequences"]))
    return len(labels["sequences"]), number_frames, number_windows


def count_split_pairs(split_data_folder: str | Path, sequence_settings: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Counts the pairs, frames and kept pairs of a sequence comparison split from its 'labels.json' manifest, printed by
    the sequence comparison trainer at startup. A pair is kept, as in the sequence pair dataset, when both of its
    members keep frames (compute_member_frame_indices); the kept pairs are the samples of an epoch.

    Args:
        split_data_folder (str | Path): The root folder of the split, containing a 'labels.json'.
        sequence_settings (Dict[str, Any]): The 'Data.sequence_settings' section.

    Returns:
        Tuple[int, int, int]: The number of pairs, of frames of both members, and of kept pairs (before label mapping).
    """
    labels = read_json(path=os.path.join(str(split_data_folder), "labels.json"))
    number_frames, number_kept_pairs = 0, 0
    for pair in labels["pairs"]:
        # Frames of both members, and whether every member keeps frames
        member_frame_indices = []
        for member_name in PAIR_MEMBERS:
            number_frames += len(pair[member_name])
            member_frame_indices.append(compute_member_frame_indices(
                number_frames=len(pair[member_name]),
                sequence_length=sequence_settings["sequence_length"],
                short_sequences=sequence_settings["short_sequences"]))
        number_kept_pairs += int(all(len(frame_indices) > 0 for frame_indices in member_frame_indices))
    return len(labels["pairs"]), number_frames, number_kept_pairs
