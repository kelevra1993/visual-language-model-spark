"""
SequenceDataset, the Native (OpenCV, CPU) dataset of the sequence classification and sequence labelling tasks, cutting
every sequence of a 'labels.json' split into padded windows of frames with sequence or frame labels, and the window
helpers shared with the sequence pair dataset and the split description. Instantiated by get_sequence_dataloaders of
data/loaders.py.
"""

import json
import os
import random
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
from pathlib import PurePosixPath
from typing import Any, Dict, List, Tuple

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.data.label_mapping import get_class_names
from vision_training.data.preprocessing import preprocess_image
from vision_training.utilities.os_utilities import print_color

# Labels read from the manifest: one class per sequence (many to one) or one class per frame (many to many)
VALID_LABEL_TYPES = ["sequence", "frame"]


def compute_window_starts(number_frames: int, sequence_length: int, window_stride: int,
                          short_sequences: str) -> List[int]:
    """
    Cuts a sequence into the windows of 'sequence_length' frames fed to the sequence models, one window starting
    every 'window_stride' frames, plus a last window ending on the last frame when the stride does not reach it, so
    that every frame belongs to a window. A sequence shorter than one window gives a single padded window ("pad"), or
    none ("skip"). Shared by the sequence dataset and the split description printed at startup.

    Args:
        number_frames (int): The number of frames of the sequence, after the frame stride.
        sequence_length (int): The number of frames of a window.
        window_stride (int): The number of frames between the starts of two windows.
        short_sequences (str): "pad" or "skip", for the sequences shorter than a window.

    Returns:
        List[int]: The index of the first frame of every window.
    """
    if number_frames < sequence_length:
        return [0] if short_sequences == "pad" and number_frames > 0 else []

    # Regular windows, then a last one aligned on the end of the sequence when needed
    window_starts = list(range(0, number_frames - sequence_length + 1, window_stride))
    if window_starts[-1] != number_frames - sequence_length:
        window_starts.append(number_frames - sequence_length)
    return window_starts


def read_window_frames(data_directory: str, frame_files: List[str], sequence_length: int, image_size: int,
                       keep_ratio: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reads the frames of one window, resized and padded like the images of the other tasks, into the fixed-size frame
    tensor fed to the sequence models: the real frames first, in order, then black padding frames up to the sequence
    length, marked as padding by the returned mask. Shared by the sequence dataset (one window per sample) and the
    sequence pair dataset (one window per member of a pair).

    Args:
        data_directory (str): The folder the frame paths are relative to.
        frame_files (List[str]): The paths of the real frames of the window, at most sequence_length of them.
        sequence_length (int): The number of frames of the returned window.
        image_size (int): The size of the square network input.
        keep_ratio (bool): Whether to keep the aspect ratio of the frames, padding them.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The uint8 [sequence_length, 3, image_size, image_size] BGR frames (black
            padding frames last) and the [sequence_length] boolean mask, True for the real frames.
    """
    frames = torch.zeros(size=(sequence_length, 3, image_size, image_size), dtype=torch.uint8)
    sequence_mask = torch.zeros(size=(sequence_length,), dtype=torch.bool)

    # Read the real frames in order, the remaining ones staying black padding frames
    for frame_index, frame_file in enumerate(frame_files):
        image = cv2.imread(filename=os.path.join(data_directory, frame_file))
        resized_image = preprocess_image(image=image, image_size=image_size, keep_ratio=keep_ratio)
        frames[frame_index] = torch.from_numpy(np.ascontiguousarray(resized_image)).permute(2, 0, 1)
        sequence_mask[frame_index] = True
    return frames, sequence_mask


class SequenceDataset(Dataset):
    """
    Dataset of the sequence tasks, read with OpenCV on the CPU (Native loader).

    A split is a folder holding the frames of every sequence in 'data/<sequence name>/' and a 'labels.json' manifest
    listing the 'categories' and the 'sequences', each sequence giving its 'id', 'name', ordered 'frames' (paths
    relative to 'data/'), and its 'category_id' (many to one) or 'frame_category_ids' (many to many). The frames are
    subsampled by the frame stride, then every sequence is cut into windows of 'sequence_length' frames, the samples
    of the dataset. The frames are resized and padded like the images of the other tasks, and a window shorter than
    'sequence_length' is padded with black frames, marked as padding by its sequence mask.
    """

    def __init__(self, data_directory: str, labels_file: str, image_size: int, keep_ratio: bool,
                 label_dictionary: Dict[Any, int], sequence_settings: Dict[str, Any], label_type: str,
                 shuffle: bool = True) -> None:
        """
        Reads the manifest of the split and cuts its sequences into windows.

        Args:
            data_directory (str): The folder holding the frame folders of the sequences.
            labels_file (str): The 'labels.json' manifest of the split.
            image_size (int): The size of the square network input.
            keep_ratio (bool): Whether to keep the aspect ratio of the frames, padding them.
            label_dictionary (Dict[Any, int]): The class index of every raw category identifier.
            sequence_settings (Dict[str, Any]): The 'Data.sequence_settings' section: 'sequence_length',
                'frame_stride', 'window_stride' and 'short_sequences'.
            label_type (str): "sequence" (one class per window, from 'category_id', windows of unmapped categories
                being left out) or "frame" (one class per frame, from 'frame_category_ids', frames of unmapped
                categories being ignored by the loss and the metrics).
            shuffle (bool): Whether to shuffle the windows once at creation. Defaults to True.
        """
        self.data_directory = data_directory
        self.image_size = image_size
        self.keep_ratio = keep_ratio
        self.label_type = label_type
        self.sequence_length = sequence_settings["sequence_length"]
        self.number_classes = len(set(label_dictionary.values()))

        with open(file=labels_file, mode="r") as labels_file_handler:
            manifest = json.load(fp=labels_file_handler)

        # Cut every sequence into windows of frames, with the class of the window or of each of its frames
        self.windows = []
        number_left_out_sequences = 0
        for sequence in manifest["sequences"]:
            frames = sequence["frames"][::sequence_settings["frame_stride"]]
            if label_type == "sequence":
                if sequence["category_id"] not in label_dictionary:
                    number_left_out_sequences += 1
                    continue
                frame_labels = [label_dictionary[sequence["category_id"]]] * len(frames)
            else:
                frame_labels = [label_dictionary[category] if category in label_dictionary else IGNORED_LABEL_INDEX
                                for category in sequence["frame_category_ids"][::sequence_settings["frame_stride"]]]

            window_starts = compute_window_starts(number_frames=len(frames), sequence_length=self.sequence_length,
                                                  window_stride=sequence_settings["window_stride"],
                                                  short_sequences=sequence_settings["short_sequences"])
            number_left_out_sequences += int(len(window_starts) == 0)
            for window_start in window_starts:
                window_end = window_start + self.sequence_length
                self.windows.append({"identifier": len(self.windows), "sequence_identifier": sequence["id"],
                                     "sequence_name": sequence["name"], "frames": frames[window_start:window_end],
                                     "frame_labels": frame_labels[window_start:window_end]})
        if number_left_out_sequences > 0:
            print_color(color="yellow", output=f"{number_left_out_sequences} sequences of unmapped categories or "
                                               f"shorter than {self.sequence_length} frames left out : "
                                               f"file://{os.path.abspath(labels_file)}")
        if shuffle:
            random.shuffle(self.windows)

        # Every window by identifier, to name the evaluated windows and frames in the evaluation outputs
        self.window_records = {window["identifier"]: window for window in self.windows}

        # Name of every class, taken from the categories of the manifest
        self.class_names = get_class_names(categories=manifest["categories"], label_dictionary=label_dictionary)

    def __len__(self) -> int:
        """
        Gives the number of windows.

        Returns:
            int: The number of windows.
        """
        return len(self.windows)

    def __getitem__(self, window_index: int) -> Dict[str, Any]:
        """
        Reads and preprocesses the frames of one window, with its target.

        Args:
            window_index (int): The index of the window.

        Returns:
            Dict[str, Any]: The uint8 [sequence_length, 3, image_size, image_size] BGR frames (black padding frames
                last), the [sequence_length] boolean sequence mask (True for the real frames), the label (an int64
                class index per window, or an int64 class index per frame with -100 for the padding frames) and the
                window identifier.
        """
        window = self.windows[window_index]

        # Real frames in order, followed by the black padding frames
        frames, sequence_mask = read_window_frames(data_directory=self.data_directory, frame_files=window["frames"],
                                                   sequence_length=self.sequence_length, image_size=self.image_size,
                                                   keep_ratio=self.keep_ratio)

        # One class per window, or one class per frame ignored on the padding frames
        if self.label_type == "sequence":
            label = torch.tensor(data=window["frame_labels"][0], dtype=torch.int64)
        else:
            label = torch.full(size=(self.sequence_length,), fill_value=IGNORED_LABEL_INDEX, dtype=torch.int64)
            label[:len(window["frame_labels"])] = torch.tensor(data=window["frame_labels"], dtype=torch.int64)

        return {"images": frames, "sequence_mask": sequence_mask, "labels": label,
                "sample_identifiers": window["identifier"]}

    def describe_sample(self, sample_identifier: int) -> Dict[str, str]:
        """
        Names an evaluated window in the predictions file written at every evaluation, from its identifier in the
        batches: its sequence, and its first and last real frames, a sequence being cut into several windows.

        Args:
            sample_identifier (int): The identifier of the window, as given in the 'sample_identifiers' of a batch.

        Returns:
            Dict[str, str]: The 'sequence', 'first_frame' and 'last_frame' columns, the frames given relative to the
                'data' folder.
        """
        window = self.window_records[sample_identifier]
        return {"sequence": window["sequence_name"], "first_frame": window["frames"][0],
                "last_frame": window["frames"][-1]}

    def get_sample_name(self, sample_identifier: int) -> str:
        """
        Gives the short name of an evaluated window used in the file names of the error images of the evaluation: its
        sequence followed by its first real frame, a sequence being cut into several windows.

        Args:
            sample_identifier (int): The identifier of the window, as given in the 'sample_identifiers' of a batch.

        Returns:
            str: The name of the window, e.g. 'test_000003_000004' for the window of 'test_000003' starting at
                'test_000003/000004.png'.
        """
        window = self.window_records[sample_identifier]
        return f"{window['sequence_name']}_{PurePosixPath(window['frames'][0]).stem}".replace("/", "_")

    def get_window_frame_names(self, sample_identifier: int) -> List[str]:
        """
        Lists the real frames of an evaluated window, in order, so that the predictions of the sequence labelling task
        name every frame they classify.

        Args:
            sample_identifier (int): The identifier of the window, as given in the 'sample_identifiers' of a batch.

        Returns:
            List[str]: The paths of the real frames of the window relative to the 'data' folder.
        """
        return self.window_records[sample_identifier]["frames"]

    def get_class_counts(self) -> List[int]:
        """
        Counts the windows of every class (one class per window), or the real frames of every class (one class per
        frame), used to weight the classes of the loss inversely to their frequency.

        Returns:
            List[int]: The number of windows or frames of every class index.
        """
        class_counts = [0] * self.number_classes
        for window in self.windows:
            window_labels = window["frame_labels"][:1] if self.label_type == "sequence" else window["frame_labels"]
            for class_index in window_labels:
                if class_index != IGNORED_LABEL_INDEX:
                    class_counts[class_index] += 1
        return class_counts
