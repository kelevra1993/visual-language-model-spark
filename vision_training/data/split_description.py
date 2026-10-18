"""
Counts the samples of a dataset split from its 'labels.json' (images and annotations, classification samples,
sequences and kept sequences, pairs and kept pairs), printed by the task trainers at startup to describe the splits of a
run.
"""

import os

from pathlib import Path
from typing import Any, Dict, Tuple

from vision_training.data.sequence.sequence_pair_dataset import PAIR_MEMBERS
from vision_training.utilities.os_utilities import read_json


def count_split_images_and_annotations(split_data_folder: str | Path) -> Tuple[int, int]:
    """
    Counts the images and annotations of a dataset split from its COCO 'labels.json', printed by the Trainer at
    startup. The file is read directly, the length of a loader being a number of batches, not of images.

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


def count_split_sequences(split_data_folder: str | Path, sequence_settings: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Counts the sequences, frames and kept sequences of a sequence split from its 'labels.json' manifest, printed by
    the sequence trainers at startup. A sequence is kept, as in the sequence dataset, when it holds frames and at most
    'maximum_sequence_length' of them; the kept sequences are the samples of an epoch.

    Args:
        split_data_folder (str | Path): The root folder of the split, containing a 'labels.json'.
        sequence_settings (Dict[str, Any]): The 'Data.sequence_settings' section.

    Returns:
        Tuple[int, int, int]: The number of sequences, of frames and of kept sequences (before label mapping).
    """
    labels = read_json(path=os.path.join(str(split_data_folder), "labels.json"))
    number_frames, number_kept_sequences = 0, 0
    for sequence in labels["sequences"]:
        number_frames += len(sequence["frames"])
        number_kept_sequences += int(0 < len(sequence["frames"]) <= sequence_settings["maximum_sequence_length"])
    return len(labels["sequences"]), number_frames, number_kept_sequences


def count_split_pairs(split_data_folder: str | Path, sequence_settings: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Counts the pairs, frames and kept pairs of a sequence comparison split from its 'labels.json' manifest, printed by
    the sequence comparison trainer at startup. A pair is kept, as in the sequence pair dataset, when both of its
    members hold frames and at most 'maximum_sequence_length' of them; the kept pairs are the samples of an epoch.

    Args:
        split_data_folder (str | Path): The root folder of the split, containing a 'labels.json'.
        sequence_settings (Dict[str, Any]): The 'Data.sequence_settings' section.

    Returns:
        Tuple[int, int, int]: The number of pairs, of frames of both members, and of kept pairs (before label mapping).
    """
    labels = read_json(path=os.path.join(str(split_data_folder), "labels.json"))
    number_frames, number_kept_pairs = 0, 0
    for pair in labels["pairs"]:
        # Frames of both members, and whether every member holds frames, not too many
        member_lengths = [len(pair[member_name]) for member_name in PAIR_MEMBERS]
        number_frames += sum(member_lengths)
        number_kept_pairs += int(min(member_lengths) > 0 and
                                 max(member_lengths) <= sequence_settings["maximum_sequence_length"])
    return len(labels["pairs"]), number_frames, number_kept_pairs
