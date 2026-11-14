"""
SequenceDataset, the Native (OpenCV, CPU) dataset of the sequence classification, sequence labelling and frame
selection tasks, giving every sequence of a 'labels.json' split whole with its sequence or frame labels (the sequences
of the frame selection being checked to hold every wanted class), the sequences longer than
'Data.sequence_settings.maximum_sequence_length' being left out with a warning. Also holds the frame reading and the
warnings shared with the sequence pair dataset, and the collate function padding the batches of both datasets to their
longest sequence. The kept sequences are packed into SerializedRecords (data/serialized_records.py), which the forked
loader workers read without copying the Python objects of the main process. Instantiated by get_sequence_dataloaders
of data/loaders.py.
"""

import os
import random
import numpy as np
import torch

from torch.utils.data import Dataset, default_collate
from typing import Any, Dict, List, Optional, Tuple

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.data.label_mapping import get_class_names
from vision_training.data.manifest_cache import read_manifest
from vision_training.data.preprocessing import preprocess_image, read_image
from vision_training.data.serialized_records import SerializedRecords
from vision_training.utilities.os_utilities import format_path_link, print_color

# Largest number of invalid sequences listed in the error of the frame selection data check, the others being counted
MAXIMUM_LISTED_INVALID_SEQUENCES = 20

# Keys of a sample whose tensors have one entry per frame along their first axis, padded to the longest sequence of the
# batch by collate_padded_sequences with these values (the labels only when they are given per frame): black frames,
# padding frames out of the sequence mask, and labels ignored by the loss and the metrics
FRAME_AXIS_PADDING_VALUES = {"images": 0, "sequence_mask": False, "labels": IGNORED_LABEL_INDEX}


def read_sequence_frames(data_directory: str, frame_files: List[str], number_frames: int, image_size: int,
                         keep_ratio: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reads the frames of one sequence, resized and padded like the images of the other tasks, into the frame tensor fed
    to the sequence models: the real frames first, in order, then black padding frames up to 'number_frames', marked as
    padding by the returned mask. Shared by the sequence dataset (one sequence per sample, without padding frames) and
    the sequence pair dataset (both members of a pair padded to its longer member).

    Args:
        data_directory (str): The folder the frame paths are relative to.
        frame_files (List[str]): The paths of the real frames of the sequence, at most number_frames of them.
        number_frames (int): The number of frames of the returned tensor.
        image_size (int): The size of the square network input.
        keep_ratio (bool): Whether to keep the aspect ratio of the frames, padding them.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The uint8 [number_frames, 3, image_size, image_size] BGR frames (black
            padding frames last) and the [number_frames] boolean mask, True for the real frames.
    """
    frames = torch.zeros(size=(number_frames, 3, image_size, image_size), dtype=torch.uint8)
    sequence_mask = torch.zeros(size=(number_frames,), dtype=torch.bool)

    # Read the real frames in order, the remaining ones staying black padding frames
    for frame_index, frame_file in enumerate(frame_files):
        image = read_image(image_path=os.path.join(data_directory, frame_file))
        resized_image = preprocess_image(image=image, image_size=image_size, keep_ratio=keep_ratio)
        frames[frame_index] = torch.from_numpy(np.ascontiguousarray(resized_image)).permute(2, 0, 1)
        sequence_mask[frame_index] = True
    return frames, sequence_mask


def warn_long_sequence(sample_kind: str, sample_name: str, split_name: str, number_frames: int,
                       maximum_sequence_length: int) -> None:
    """
    Warns in red that a sequence, or a pair with a member, longer than 'Data.sequence_settings.maximum_sequence_length'
    is left out of a split, the training going on without it. Shared by the sequence dataset and the sequence pair
    dataset, every word capitalised so that the warning stands out of the startup output.

    Args:
        sample_kind (str): "Sequence" or "Pair".
        sample_name (str): The name of the sequence or of the pair in the manifest.
        split_name (str): "Training", "Validation" or "Test".
        number_frames (int): The number of frames of the sequence, or of the longest member of the pair.
        maximum_sequence_length (int): The largest number of frames of a sequence or of a member.
    """
    frame_description = (f"{number_frames} Frames" if sample_kind == "Sequence" else
                         f"A Member Of {number_frames} Frames")
    print_color(color="red", output=f"{sample_kind} '{sample_name}' Of The {split_name} Split Has {frame_description}, "
                                    f"More Than The Maximum Sequence Length ({maximum_sequence_length}): It Is Left "
                                    f"Out Of The {split_name}.")


def warn_long_sequence_total(sample_kind: str, number_long_samples: int, split_name: str,
                             maximum_sequence_length: int) -> None:
    """
    Warns in red, after the line of every left out sequence (warn_long_sequence), how many sequences or pairs of a
    split are left out for being longer than 'Data.sequence_settings.maximum_sequence_length', nothing being printed
    when none is.

    Args:
        sample_kind (str): "Sequence" or "Pair".
        number_long_samples (int): The number of left out sequences or pairs of the split.
        split_name (str): "Training", "Validation" or "Test".
        maximum_sequence_length (int): The largest number of frames of a sequence or of a member.
    """
    if number_long_samples == 0:
        return
    length_description = "Are Longer" if sample_kind == "Sequence" else "Have A Member Longer"
    print_color(color="red", output=f"{number_long_samples} {sample_kind}s Of The {split_name} Split "
                                    f"{length_description} Than The Maximum Sequence Length "
                                    f"({maximum_sequence_length}) And Are Left Out Of The {split_name}.")


def collate_padded_sequences(samples: List[Dict[str, Any]], members_per_sample: int) -> Dict[str, Any]:
    """
    Batches the samples of the sequence datasets, whose sequences have their own number of frames, padded to the
    longest sequence of the batch, so that the loaders transfer, and the temporal modules run over, no more padding
    frames than the batch needs. Every sample holds 'members_per_sample' sequences of the same number of frames one
    after the other along its frame axis (one sequence, or the before and after members of a pair padded to the longer
    member), so every member is padded to the longest member of the batch, the two members of the pairs keeping the
    same number of frames. Run by the worker processes of the loaders.

    Args:
        samples (List[Dict[str, Any]]): The samples of the batch, as returned by SequenceDataset or SequencePairDataset.
        members_per_sample (int): The number of sequences of a sample: 1 for the sequences, 2 for the pairs.

    Returns:
        Dict[str, Any]: The batch of the default collate, with [B, members_per_sample x longest member, ...] frames,
            sequence mask and frame labels.
    """
    # Frames of every member of the samples, and frames of the longest member of the batch
    member_lengths = [sample["sequence_mask"].shape[0] // members_per_sample for sample in samples]
    longest_member = max(member_lengths)

    # Pad every member of every sample at its end, the labels of one class per sequence having no frame axis
    padded_samples = []
    for sample, member_length in zip(samples, member_lengths):
        padded_sample = dict(sample)
        for key, padding_value in FRAME_AXIS_PADDING_VALUES.items():
            if sample[key].dim() > 0:
                members = sample[key].unflatten(dim=0, sizes=(members_per_sample, member_length))
                padding = members.new_full(size=(members_per_sample, longest_member - member_length,
                                                 *members.shape[2:]), fill_value=padding_value)
                padded_sample[key] = torch.cat(tensors=[members, padding], dim=1).flatten(start_dim=0, end_dim=1)
        padded_samples.append(padded_sample)
    return default_collate(batch=padded_samples)


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
                         f"{format_path_link(path=labels_file)}")
    mapped_identifiers = [identifier for identifier in category_identifiers if identifier in label_dictionary]
    if not mapped_identifiers:
        raise ValueError(f"The unselected category {unselected_category!r} (identifier {category_identifiers[0]}) is "
                         f"not mapped by the label mapping file, its frames would have no class : "
                         f"{format_path_link(path=labels_file)}")
    return label_dictionary[mapped_identifiers[0]]


def describe_selection_problem(frame_labels: List[int], wanted_class_indices: List[int], frames_per_class: int,
                               class_names: Dict[int, str]) -> Optional[str]:
    """
    Checks that one sequence of the frame selection task can be decoded and trained on: every wanted class holds
    exactly 'frames_per_class' of its frames, the selection choosing that many frames of every class.

    Args:
        frame_labels (List[int]): The class index of every frame of the sequence, the unmapped ones being unselected.
        wanted_class_indices (List[int]): The class indices of the wanted classes.
        frames_per_class (int): The number of frames of every wanted class ('Selection.frames_per_class').
        class_names (Dict[int, str]): The name of every class index.

    Returns:
        Optional[str]: The description of what is wrong with the sequence, or None when it is valid.
    """
    # Wanted classes holding another number of frames than the selection chooses
    wrong_counts = [f"{class_names[class_index]} {frame_labels.count(class_index)} frames"
                    for class_index in wanted_class_indices if frame_labels.count(class_index) != frames_per_class]
    return ", ".join(wrong_counts) if wrong_counts else None


class SequenceDataset(Dataset):
    """
    Dataset of the sequence tasks, read with OpenCV on the CPU (Native loader).

    A split is a folder holding the frames of every sequence in 'data/<sequence name>/' and a 'labels.json' manifest
    listing the 'categories' and the 'sequences', each sequence giving its 'id', 'name', ordered 'frames' (paths
    relative to 'data/'), and its 'category_id' (many to one) or 'frame_category_ids' (many to many). Every sequence
    is one sample, given whole, the sequences longer than 'maximum_sequence_length' frames being left out with a red
    warning. The frames are resized and padded like the images of the other tasks, and the collate of the loader pads
    the sequences of a batch to its longest one with black frames, marked as padding by the sequence mask.

    For the frame selection, the frames of unmapped categories are unselected frames, and every sequence must hold
    every wanted class exactly 'frames_per_class' times.
    """

    def __init__(self, data_directory: str, labels_file: str, image_size: int, keep_ratio: bool,
                 label_dictionary: Dict[Any, int], sequence_settings: Dict[str, Any], label_type: str,
                 split_name: str, selection_settings: Optional[Dict[str, Any]] = None, shuffle: bool = True) -> None:
        """
        Reads the manifest of the split and keeps its sequences of at most 'maximum_sequence_length' frames.

        Args:
            data_directory (str): The folder holding the frame folders of the sequences.
            labels_file (str): The 'labels.json' manifest of the split.
            image_size (int): The size of the square network input.
            keep_ratio (bool): Whether to keep the aspect ratio of the frames, padding them.
            label_dictionary (Dict[Any, int]): The class index of every raw category identifier.
            sequence_settings (Dict[str, Any]): The 'Data.sequence_settings' section: 'maximum_sequence_length'.
            label_type (str): "sequence" (one class per sequence, from 'category_id', sequences of unmapped categories
                being left out), "frame" (one class per frame, from 'frame_category_ids', frames of unmapped
                categories being ignored by the loss and the metrics) or "selection" (one class per frame of a whole
                sequence, frames of unmapped categories being unselected).
            split_name (str): "Training", "Validation" or "Test", named in the warnings of the left out sequences.
            selection_settings (Optional[Dict[str, Any]]): The 'Selection' section ('unselected_category' and
                'frames_per_class') for the "selection" label type, None otherwise. Defaults to None.
            shuffle (bool): Whether to shuffle the sequences once at creation. Defaults to True.

        Raises:
            ValueError: For the "selection" label type, if the unselected category is missing or unmapped, or if
                kept sequences do not hold every wanted class 'frames_per_class' times, all of them being listed.
        """
        self.data_directory = data_directory
        self.image_size = image_size
        self.keep_ratio = keep_ratio
        self.label_type = label_type
        self.maximum_sequence_length = sequence_settings["maximum_sequence_length"]
        self.number_classes = len(set(label_dictionary.values()))

        # The manifest of the split, parsed once per launch and shared with its other readers, never modified
        manifest = read_manifest(labels_path=labels_file)

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

        # Keep every sequence of a mapped category, with frames and not too long, with its class or the class of every
        # frame
        sequences = []
        number_left_out_sequences = 0
        number_long_sequences = 0
        invalid_sequences = []
        for sequence in manifest["sequences"]:
            frames = sequence["frames"]
            if len(frames) == 0 or (label_type == "sequence" and sequence["category_id"] not in label_dictionary):
                number_left_out_sequences += 1
                continue

            # Sequences longer than the maximum sequence length are left out with a warning, the training going on
            if len(frames) > self.maximum_sequence_length:
                warn_long_sequence(sample_kind="Sequence", sample_name=sequence["name"], split_name=split_name,
                                   number_frames=len(frames), maximum_sequence_length=self.maximum_sequence_length)
                number_long_sequences += 1
                continue
            if label_type == "sequence":
                frame_labels = [label_dictionary[sequence["category_id"]]] * len(frames)
            else:
                frame_labels = [label_dictionary[category] if category in label_dictionary else unmapped_frame_label
                                for category in sequence["frame_category_ids"]]

            # Frame selection: every sequence checked to hold every wanted class
            if label_type == "selection":
                selection_problem = describe_selection_problem(
                    frame_labels=frame_labels, wanted_class_indices=wanted_class_indices,
                    frames_per_class=selection_settings["frames_per_class"], class_names=self.class_names)
                if selection_problem is not None:
                    invalid_sequences.append(f"{sequence['name']} : {selection_problem}")
            sequences.append({"identifier": len(sequences), "sequence_identifier": sequence["id"],
                              "sequence_name": sequence["name"], "frames": frames, "frame_labels": frame_labels})
        warn_long_sequence_total(sample_kind="Sequence", number_long_samples=number_long_sequences,
                                 split_name=split_name, maximum_sequence_length=self.maximum_sequence_length)

        # Every sequence of the frame selection must be valid, all the invalid ones being listed at once
        if invalid_sequences:
            listed_sequences = invalid_sequences[:MAXIMUM_LISTED_INVALID_SEQUENCES]
            if len(invalid_sequences) > MAXIMUM_LISTED_INVALID_SEQUENCES:
                listed_sequences.append(f"... and {len(invalid_sequences) - MAXIMUM_LISTED_INVALID_SEQUENCES} more")
            raise ValueError(f"{len(invalid_sequences)} sequences cannot be used for the frame selection, which needs "
                             f"every wanted class exactly {selection_settings['frames_per_class']} times "
                             f"('Selection.frames_per_class') : {format_path_link(path=labels_file)}\n" +
                             "\n".join(f" - {listed_sequence}" for listed_sequence in listed_sequences))
        if number_left_out_sequences > 0:
            print_color(color="yellow", output=f"{number_left_out_sequences} sequences of unmapped categories or "
                                               f"without frames left out : {format_path_link(path=labels_file)}")
        if shuffle:
            random.shuffle(sequences)

        # The kept sequences packed in their order into one read-only buffer, which the loader workers read without
        # copying Python objects of the main process, every sequence found by its identifier to name the evaluated
        # sequences and frames in the evaluation outputs
        self.sequences = SerializedRecords(records=sequences, identifier_key="identifier")

    def __len__(self) -> int:
        """
        Gives the number of kept sequences.

        Returns:
            int: The number of sequences.
        """
        return len(self.sequences)

    def __getitem__(self, sequence_index: int) -> Dict[str, Any]:
        """
        Reads and preprocesses the frames of one sequence, with its target.

        Args:
            sequence_index (int): The index of the sequence.

        Returns:
            Dict[str, Any]: The uint8 [T, 3, image_size, image_size] BGR frames of the T frames of the sequence, the
                [T] boolean sequence mask (all True, the collate padding it), the label (an int64 class index per
                sequence, or an int64 class index per frame) and the sequence identifier.
        """
        sequence = self.sequences[sequence_index]
        frames, sequence_mask = read_sequence_frames(data_directory=self.data_directory, frame_files=sequence["frames"],
                                                     number_frames=len(sequence["frames"]), image_size=self.image_size,
                                                     keep_ratio=self.keep_ratio)

        # One class per sequence, or one class per frame
        if self.label_type == "sequence":
            label = torch.tensor(data=sequence["frame_labels"][0], dtype=torch.int64)
        else:
            label = torch.tensor(data=sequence["frame_labels"], dtype=torch.int64)

        return {"images": frames, "sequence_mask": sequence_mask, "labels": label,
                "sample_identifiers": sequence["identifier"]}

    def describe_sample(self, sample_identifier: int) -> Dict[str, str]:
        """
        Names an evaluated sequence in the predictions file written at every evaluation, from its identifier in the
        batches: its name, and its first and last frames.

        Args:
            sample_identifier (int): The identifier of the sequence, as given in the 'sample_identifiers' of a batch.

        Returns:
            Dict[str, str]: The 'sequence', 'first_frame' and 'last_frame' columns, the frames given relative to the
                'data' folder.
        """
        sequence = self.sequences.find_record(identifier=sample_identifier)
        return {"sequence": sequence["sequence_name"], "first_frame": sequence["frames"][0],
                "last_frame": sequence["frames"][-1]}

    def get_sample_name(self, sample_identifier: int) -> str:
        """
        Gives the short name of an evaluated sequence used in the file names of the error images of the evaluation.

        Args:
            sample_identifier (int): The identifier of the sequence, as given in the 'sample_identifiers' of a batch.

        Returns:
            str: The name of the sequence in the manifest, its folders joined with underscores, e.g. 'test_000003'.
        """
        return self.sequences.find_record(identifier=sample_identifier)["sequence_name"].replace("/", "_")

    def get_sequence_frame_names(self, sample_identifier: int) -> List[str]:
        """
        Lists the frames of an evaluated sequence, in order, so that the predictions of the sequence labelling and
        frame selection tasks name every frame they classify.

        Args:
            sample_identifier (int): The identifier of the sequence, as given in the 'sample_identifiers' of a batch.

        Returns:
            List[str]: The paths of the frames of the sequence relative to the 'data' folder.
        """
        return self.sequences.find_record(identifier=sample_identifier)["frames"]

    def get_sample_class_indices(self) -> Dict[int, List[int]]:
        """
        Lists the classes of every sequence, from which the sequence trainers spread the fixed test sequences drawn at
        every evaluation over the classes: the class of the sequence (one class per sequence), or the class of every
        labelled frame (one class per frame), so that a sequence counts for the class of most of its frames. The
        unselected frames of the frame selection are left out, every sequence holding the same wanted classes.

        Returns:
            Dict[int, List[int]]: The class indices of every sequence keyed by its identifier.
        """
        return {sequence["identifier"]: sequence["frame_labels"][:1] if self.label_type == "sequence" else
                [class_index for class_index in sequence["frame_labels"]
                 if class_index not in [IGNORED_LABEL_INDEX, self.unselected_class_index]]
                for sequence in self.sequences}

    def get_class_counts(self) -> List[int]:
        """
        Counts the sequences of every class (one class per sequence), or the labelled frames of every class (one class
        per frame), used to weight the classes of the loss inversely to their frequency.

        Returns:
            List[int]: The number of sequences or frames of every class index.
        """
        class_counts = [0] * self.number_classes
        for sequence in self.sequences:
            sequence_labels = (sequence["frame_labels"][:1] if self.label_type == "sequence" else
                               sequence["frame_labels"])
            for class_index in sequence_labels:
                if class_index != IGNORED_LABEL_INDEX:
                    class_counts[class_index] += 1
        return class_counts
