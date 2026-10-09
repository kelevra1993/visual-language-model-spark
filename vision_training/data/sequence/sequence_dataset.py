"""
SequenceDataset, the Native (OpenCV, CPU) dataset of the sequence classification, sequence labelling and frame
selection tasks, cutting every sequence of a 'labels.json' split into padded windows of frames with sequence or frame
labels (one window per whole sequence for the frame selection, whose sequences are checked to hold every wanted class),
the window helpers shared with the sequence pair dataset and the split description, and the collate function padding
the batches of both datasets to their longest window. Instantiated by get_sequence_dataloaders of data/loaders.py.
"""

import json
import os
import random
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset, default_collate
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.data.label_mapping import get_class_names
from vision_training.data.preprocessing import preprocess_image
from vision_training.utilities.os_utilities import print_color

# Labels read from the manifest: one class per sequence (many to one), one class per frame (many to many), or one class
# per frame of a whole sequence for the frame selection, the frames of unmapped categories being unselected
VALID_LABEL_TYPES = ["sequence", "frame", "selection"]

# Largest number of invalid sequences listed in the error of the frame selection data check, the others being counted
MAXIMUM_LISTED_INVALID_SEQUENCES = 20

# Keys of a sample whose tensors have one entry per frame along their first axis, cut to the longest window of the batch
# by collate_longest_windows (the labels only when they are given per frame)
FRAME_AXIS_KEYS = ["images", "sequence_mask", "labels"]


def compute_window_starts(number_frames: int, sequence_length: int, window_stride: int,
                          short_sequences: str) -> List[int]:
    """
    Cuts a sequence into the windows of 'sequence_length' frames fed to the sequence models, one window starting
    every 'window_stride' frames, plus a last window ending on the last frame when the stride does not reach it, so
    that every frame belongs to a window. A sequence shorter than one window gives a single padded window ("pad"), or
    none ("skip"). Shared by the sequence dataset and the split description printed at startup.

    Args:
        number_frames (int): The number of frames of the sequence.
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


def collate_longest_windows(samples: List[Dict[str, Any]], members_per_sample: int) -> Dict[str, Any]:
    """
    Batches the samples of the sequence datasets padded to the longest window of the batch instead of the sequence
    length ('Data.sequence_settings.batch_padding: "longest"'), so that the loaders transfer, and the temporal modules
    run over, fewer padding frames when the windows are shorter than the sequence length. Every sample holds
    'members_per_sample' windows of sequence_length frames one after the other along its frame axis (one window, or
    the before and after members of a pair), with its real frames first, so every window is cut to the number of real
    frames of the longest window of the batch, the two members of the pairs keeping the same number of frames. Run by
    the worker processes of the loaders, which then only send the cut batches.

    Args:
        samples (List[Dict[str, Any]]): The samples of the batch, as returned by SequenceDataset or SequencePairDataset.
        members_per_sample (int): The number of windows of a sample: 1 for the sequence windows, 2 for the pairs.

    Returns:
        Dict[str, Any]: The batch of the default collate, with [B, members_per_sample x longest window, ...] frames,
            sequence mask and frame labels.
    """
    # Frames of every window of the samples, and real frames of the longest window of the batch
    window_length = samples[0]["sequence_mask"].shape[0] // members_per_sample
    longest_window = max(int(sample["sequence_mask"].unflatten(dim=0, sizes=(members_per_sample, window_length))
                             .sum(dim=1).max()) for sample in samples)

    # Keep the first frames of every window of every sample, the labels of one class per window having no frame axis
    cut_samples = []
    for sample in samples:
        cut_sample = dict(sample)
        for key in FRAME_AXIS_KEYS:
            if sample[key].dim() > 0:
                cut_sample[key] = sample[key].unflatten(dim=0, sizes=(members_per_sample, window_length))[
                    :, :longest_window].flatten(start_dim=0, end_dim=1)
        cut_samples.append(cut_sample)
    return default_collate(batch=cut_samples)


def find_unselected_class_index(categories: List[Dict[str, Any]], label_dictionary: Dict[Any, int],
                                unselected_category: str, labels_file: str) -> int:
    """
    Finds the class index of the unselected frames of the frame selection task, from the name of their category in
    the manifest ('Selection.unselected_category') and the label mapping, every other class being a wanted class.

    Args:
        categories (List[Dict[str, Any]]): The 'id' and 'name' of every category of the manifest.
        label_dictionary (Dict[Any, int]): The class index of every raw category identifier.
        unselected_category (str): The name of the category of the unselected frames.
        labels_file (str): The 'labels.json' manifest, named in the errors.

    Returns:
        int: The class index of the unselected frames.

    Raises:
        ValueError: If no category has that name, or if the label mapping does not map it.
    """
    category_identifiers = [category["id"] for category in categories if category["name"] == unselected_category]
    if not category_identifiers:
        raise ValueError(f"No category is named {unselected_category!r} ('Selection.unselected_category') in : "
                         f"file://{os.path.abspath(labels_file)}")
    mapped_identifiers = [identifier for identifier in category_identifiers if identifier in label_dictionary]
    if not mapped_identifiers:
        raise ValueError(f"The unselected category {unselected_category!r} (identifier {category_identifiers[0]}) is "
                         f"not mapped by the label mapping file, its frames would have no class : "
                         f"file://{os.path.abspath(labels_file)}")
    return label_dictionary[mapped_identifiers[0]]


def describe_selection_problem(frame_labels: List[int], wanted_class_indices: List[int], frames_per_class: int,
                               sequence_length: int, class_names: Dict[int, str]) -> Optional[str]:
    """
    Checks that one sequence of the frame selection task can be decoded and trained on: it fits in one window of
    'sequence_length' frames, and every wanted class holds exactly 'frames_per_class' of its frames, the selection
    choosing that many frames of every class.

    Args:
        frame_labels (List[int]): The class index of every frame of the sequence, the unmapped ones being unselected.
        wanted_class_indices (List[int]): The class indices of the wanted classes.
        frames_per_class (int): The number of frames of every wanted class ('Selection.frames_per_class').
        sequence_length (int): The number of frames of a window ('Data.sequence_settings.sequence_length').
        class_names (Dict[int, str]): The name of every class index.

    Returns:
        Optional[str]: The description of what is wrong with the sequence, or None when it is valid.
    """
    problems = []
    if len(frame_labels) > sequence_length:
        problems.append(f"{len(frame_labels)} frames, more than the sequence length {sequence_length}")

    # Wanted classes holding another number of frames than the selection chooses
    wrong_counts = [f"{class_names[class_index]} {frame_labels.count(class_index)} frames"
                    for class_index in wanted_class_indices if frame_labels.count(class_index) != frames_per_class]
    if wrong_counts:
        problems.append(", ".join(wrong_counts))
    return "; ".join(problems) if problems else None


class SequenceDataset(Dataset):
    """
    Dataset of the sequence tasks, read with OpenCV on the CPU (Native loader).

    A split is a folder holding the frames of every sequence in 'data/<sequence name>/' and a 'labels.json' manifest
    listing the 'categories' and the 'sequences', each sequence giving its 'id', 'name', ordered 'frames' (paths
    relative to 'data/'), and its 'category_id' (many to one) or 'frame_category_ids' (many to many). Every sequence
    is cut into windows of 'sequence_length' frames, the samples of the dataset. The frames are resized and padded
    like the images of the other tasks, and a window shorter than 'sequence_length' is padded with black frames,
    marked as padding by its sequence mask.

    For the frame selection, every sequence is one window holding the whole sequence, the frames of unmapped
    categories are unselected frames, and every sequence must hold every wanted class exactly 'frames_per_class'
    times and fit in 'sequence_length' frames.
    """

    def __init__(self, data_directory: str, labels_file: str, image_size: int, keep_ratio: bool,
                 label_dictionary: Dict[Any, int], sequence_settings: Dict[str, Any], label_type: str,
                 selection_settings: Optional[Dict[str, Any]] = None, shuffle: bool = True) -> None:
        """
        Reads the manifest of the split and cuts its sequences into windows.

        Args:
            data_directory (str): The folder holding the frame folders of the sequences.
            labels_file (str): The 'labels.json' manifest of the split.
            image_size (int): The size of the square network input.
            keep_ratio (bool): Whether to keep the aspect ratio of the frames, padding them.
            label_dictionary (Dict[Any, int]): The class index of every raw category identifier.
            sequence_settings (Dict[str, Any]): The 'Data.sequence_settings' section: 'sequence_length', and
                'window_stride' and 'short_sequences' unless the label type is "selection".
            label_type (str): "sequence" (one class per window, from 'category_id', windows of unmapped categories
                being left out), "frame" (one class per frame, from 'frame_category_ids', frames of unmapped
                categories being ignored by the loss and the metrics) or "selection" (one class per frame of a whole
                sequence, frames of unmapped categories being unselected).
            selection_settings (Optional[Dict[str, Any]]): The 'Selection' section ('unselected_category' and
                'frames_per_class') for the "selection" label type, None otherwise. Defaults to None.
            shuffle (bool): Whether to shuffle the windows once at creation. Defaults to True.

        Raises:
            ValueError: For the "selection" label type, if the unselected category is missing or unmapped, or if
                sequences do not hold every wanted class 'frames_per_class' times or are longer than
                'sequence_length', all of them being listed.
        """
        self.data_directory = data_directory
        self.image_size = image_size
        self.keep_ratio = keep_ratio
        self.label_type = label_type
        self.sequence_length = sequence_settings["sequence_length"]
        self.number_classes = len(set(label_dictionary.values()))

        with open(file=labels_file, mode="r") as labels_file_handler:
            manifest = json.load(fp=labels_file_handler)

        # Name of every class, taken from the categories of the manifest
        self.class_names = get_class_names(categories=manifest["categories"], label_dictionary=label_dictionary)

        # Frame selection: class of the unselected frames, given to the frames of unmapped categories, the other
        # classes being the wanted ones; the other label types ignore the frames of unmapped categories
        self.unselected_class_index = None
        unmapped_frame_label = IGNORED_LABEL_INDEX
        if label_type == "selection":
            self.unselected_class_index = find_unselected_class_index(
                categories=manifest["categories"], label_dictionary=label_dictionary,
                unselected_category=selection_settings["unselected_category"], labels_file=labels_file)
            unmapped_frame_label = self.unselected_class_index
        wanted_class_indices = [class_index for class_index in range(self.number_classes)
                                if class_index != self.unselected_class_index]

        # Cut every sequence into windows of frames, with the class of the window or of each of its frames
        self.windows = []
        number_left_out_sequences = 0
        invalid_sequences = []
        for sequence in manifest["sequences"]:
            frames = sequence["frames"]
            if label_type == "sequence":
                if sequence["category_id"] not in label_dictionary:
                    number_left_out_sequences += 1
                    continue
                frame_labels = [label_dictionary[sequence["category_id"]]] * len(frames)
            else:
                frame_labels = [label_dictionary[category] if category in label_dictionary else unmapped_frame_label
                                for category in sequence["frame_category_ids"]]

            # A whole sequence per window for the frame selection, checked to hold every wanted class, or windows cut
            # every window stride frames
            if label_type == "selection":
                selection_problem = describe_selection_problem(
                    frame_labels=frame_labels, wanted_class_indices=wanted_class_indices,
                    frames_per_class=selection_settings["frames_per_class"], sequence_length=self.sequence_length,
                    class_names=self.class_names)
                if selection_problem is not None:
                    invalid_sequences.append(f"{sequence['name']} : {selection_problem}")
                window_starts = [0]
            else:
                window_starts = compute_window_starts(number_frames=len(frames), sequence_length=self.sequence_length,
                                                      window_stride=sequence_settings["window_stride"],
                                                      short_sequences=sequence_settings["short_sequences"])
            number_left_out_sequences += int(len(window_starts) == 0)
            for window_start in window_starts:
                window_end = window_start + self.sequence_length
                self.windows.append({"identifier": len(self.windows), "sequence_identifier": sequence["id"],
                                     "sequence_name": sequence["name"], "frames": frames[window_start:window_end],
                                     "frame_labels": frame_labels[window_start:window_end]})
        # Every sequence of the frame selection must be valid, all the invalid ones being listed at once
        if invalid_sequences:
            listed_sequences = invalid_sequences[:MAXIMUM_LISTED_INVALID_SEQUENCES]
            if len(invalid_sequences) > MAXIMUM_LISTED_INVALID_SEQUENCES:
                listed_sequences.append(f"... and {len(invalid_sequences) - MAXIMUM_LISTED_INVALID_SEQUENCES} more")
            raise ValueError(f"{len(invalid_sequences)} sequences cannot be used for the frame selection, which needs "
                             f"every wanted class exactly {selection_settings['frames_per_class']} times "
                             f"('Selection.frames_per_class') and at most {self.sequence_length} frames per sequence "
                             f"('Data.sequence_settings.sequence_length') : file://{os.path.abspath(labels_file)}\n" +
                             "\n".join(f" - {listed_sequence}" for listed_sequence in listed_sequences))
        if number_left_out_sequences > 0:
            print_color(color="yellow", output=f"{number_left_out_sequences} sequences of unmapped categories or "
                                               f"shorter than {self.sequence_length} frames left out : "
                                               f"file://{os.path.abspath(labels_file)}")
        if shuffle:
            random.shuffle(self.windows)

        # Every window by identifier, to name the evaluated windows and frames in the evaluation outputs
        self.window_records = {window["identifier"]: window for window in self.windows}

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
        sequence followed by its first real frame, a sequence being cut into several windows, or its sequence alone for
        the frame selection, whose window is the whole sequence.

        Args:
            sample_identifier (int): The identifier of the window, as given in the 'sample_identifiers' of a batch.

        Returns:
            str: The name of the window, e.g. 'test_000003_000004' for the window of 'test_000003' starting at
                'test_000003/000004.png', or 'test_000003' for the frame selection.
        """
        window = self.window_records[sample_identifier]
        if self.label_type == "selection":
            return window["sequence_name"].replace("/", "_")
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

    def get_sample_class_indices(self) -> Dict[int, List[int]]:
        """
        Lists the classes of every window, from which the sequence trainers spread the fixed test windows drawn at
        every evaluation over the classes: the class of the window (one class per window), or the class of every
        labelled frame (one class per frame), so that a window counts for the class of most of its frames. The
        unselected frames of the frame selection are left out, every sequence holding the same wanted classes.

        Returns:
            Dict[int, List[int]]: The class indices of every window keyed by its identifier.
        """
        return {window["identifier"]: window["frame_labels"][:1] if self.label_type == "sequence" else
                [class_index for class_index in window["frame_labels"]
                 if class_index not in [IGNORED_LABEL_INDEX, self.unselected_class_index]]
                for window in self.windows}

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
