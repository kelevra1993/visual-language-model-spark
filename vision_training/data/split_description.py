import os

from pathlib import Path
from typing import Any, Dict, Tuple

from vision_training.data.sequence.sequence_dataset import compute_window_starts
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


def count_split_sequences(split_data_folder: str | Path, sequence_settings: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Counts the sequences, frames and windows of a sequence split from its 'labels.json' manifest, printed by the
    sequence trainers at startup. The windows, cut as the sequence dataset does after the frame stride, are the
    samples of an epoch.

    Args:
        split_data_folder (str | Path): The root folder of the split, containing a 'labels.json'.
        sequence_settings (Dict[str, Any]): The 'Data.sequence_settings' section.

    Returns:
        Tuple[int, int, int]: The number of sequences, of frames and of windows (before label mapping).
    """
    labels = read_json(path=os.path.join(str(split_data_folder), "labels.json"))
    number_frames, number_windows = 0, 0
    for sequence in labels["sequences"]:
        number_frames += len(sequence["frames"])
        number_windows += len(compute_window_starts(
            number_frames=len(sequence["frames"][::sequence_settings["frame_stride"]]),
            sequence_length=sequence_settings["sequence_length"], window_stride=sequence_settings["window_stride"],
            short_sequences=sequence_settings["short_sequences"]))
    return len(labels["sequences"]), number_frames, number_windows
