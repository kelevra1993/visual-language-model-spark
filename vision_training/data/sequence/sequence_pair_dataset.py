import json
import os
import random
import torch

from torch.utils.data import Dataset
from typing import Any, Dict, List, Tuple

from vision_training.data.label_mapping import get_class_names
from vision_training.data.sequence.sequence_dataset import read_window_frames
from vision_training.utilities.os_utilities import print_color

# Members of a pair in the manifests, in the order their windows are stacked along the frame axis of a sample
PAIR_MEMBERS = ["before", "after"]


def compute_member_frame_indices(number_frames: int, sequence_length: int, short_sequences: str) -> List[int]:
    """
    Chooses the frames of one member of a pair (its before or after sequence) fed to the comparison model as a single
    window of at most 'sequence_length' frames: every frame of a member that fits in the window, or the middle frame of
    each of 'sequence_length' equal parts of a longer member, so that the window covers the whole member in order.
    An empty member, or a member shorter than the window when short sequences are skipped, gives no frame, which
    leaves the pair out. Shared by the sequence pair dataset and the split description printed at startup.

    Args:
        number_frames (int): The number of frames of the member, after the frame stride.
        sequence_length (int): The maximum number of frames of a member window.
        short_sequences (str): "pad" (a shorter member is padded with black frames) or "skip" (its pair is left out).

    Returns:
        List[int]: The indices of the kept frames in increasing order, or [] when the pair is left out.
    """
    # An empty member, or a short member when the short members are skipped, leaves the pair out
    if number_frames == 0 or (number_frames < sequence_length and short_sequences == "skip"):
        return []

    # Every frame of a member that fits in the window, the remaining positions being padded
    if number_frames <= sequence_length:
        return list(range(number_frames))

    # Middle frame of each of the sequence_length equal parts of a longer member, in integer arithmetic
    return [((2 * part_index + 1) * number_frames) // (2 * sequence_length) for part_index in range(sequence_length)]


def split_pair_members(pair_tensor: torch.Tensor, sequence_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Splits the frames or the sequence mask of a batch of pairs, stored as the before window followed by the after
    window along the frame axis, into the before and after members given to the comparison model.

    Args:
        pair_tensor (torch.Tensor): The [B, 2 x sequence_length, ...] frames or [B, 2 x sequence_length] mask.
        sequence_length (int): The number of frames of a member window.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The [B, sequence_length, ...] before member and after member.

    Raises:
        ValueError: If the frame axis does not hold two member windows of sequence_length frames.
    """
    if pair_tensor.shape[1] != 2 * sequence_length:
        raise ValueError(f"A batch of pairs must hold 2 x {sequence_length} frames per pair, got "
                         f"{pair_tensor.shape[1]}.")
    return pair_tensor[:, :sequence_length], pair_tensor[:, sequence_length:]


class SequencePairDataset(Dataset):
    """
    Dataset of the sequence comparison task, read with OpenCV on the CPU (Native loader).

    A split is a folder holding the frames in 'data/' and a 'labels.json' manifest listing the 'categories' and the
    'pairs', each pair giving its 'id', 'name', the ordered frames of its 'before' and 'after' members (paths relative
    to 'data/') and its 'category_id'. Each member is subsampled by the frame stride, then becomes one window of at
    most 'sequence_length' frames (compute_member_frame_indices). A sample stores the before window followed by the
    after window along the frame axis, each padded to 'sequence_length' with black frames, so that the generic code
    of the trainers (augmentation, input viewer, evaluation) handles a pair as one sequence of 2 x sequence_length
    frames, both members sharing the same augmentation draws.
    """

    def __init__(self, data_directory: str, labels_file: str, image_size: int, keep_ratio: bool,
                 label_dictionary: Dict[Any, int], sequence_settings: Dict[str, Any], shuffle: bool = True) -> None:
        """
        Reads the manifest of the split and chooses the frames of every member of its pairs.

        Args:
            data_directory (str): The folder holding the frames of the pairs.
            labels_file (str): The 'labels.json' manifest of the split.
            image_size (int): The size of the square network input.
            keep_ratio (bool): Whether to keep the aspect ratio of the frames, padding them.
            label_dictionary (Dict[Any, int]): The class index of every raw category identifier, the pairs of unmapped
                categories being left out.
            sequence_settings (Dict[str, Any]): The 'Data.sequence_settings' section: 'sequence_length' (maximum number
                of frames of a member), 'frame_stride' and 'short_sequences'.
            shuffle (bool): Whether to shuffle the pairs once at creation. Defaults to True.
        """
        self.data_directory = data_directory
        self.image_size = image_size
        self.keep_ratio = keep_ratio
        self.sequence_length = sequence_settings["sequence_length"]
        self.number_classes = len(set(label_dictionary.values()))

        with open(file=labels_file, mode="r") as labels_file_handler:
            manifest = json.load(fp=labels_file_handler)

        # Keep the pairs of mapped categories whose two members both keep frames
        self.pairs = []
        number_left_out_pairs = 0
        for pair in manifest["pairs"]:
            if pair["category_id"] not in label_dictionary:
                number_left_out_pairs += 1
                continue

            # Frames of every member, after the frame stride and the choice of its window frames
            member_frames = {}
            for member_name in PAIR_MEMBERS:
                strided_frames = pair[member_name][::sequence_settings["frame_stride"]]
                frame_indices = compute_member_frame_indices(number_frames=len(strided_frames),
                                                             sequence_length=self.sequence_length,
                                                             short_sequences=sequence_settings["short_sequences"])
                member_frames[member_name] = [strided_frames[frame_index] for frame_index in frame_indices]
            if any(len(frames) == 0 for frames in member_frames.values()):
                number_left_out_pairs += 1
                continue
            self.pairs.append({"identifier": len(self.pairs), "pair_identifier": pair["id"], "name": pair["name"],
                               **member_frames, "label": label_dictionary[pair["category_id"]]})
        if number_left_out_pairs > 0:
            print_color(color="yellow", output=f"{number_left_out_pairs} pairs of unmapped categories, with an empty "
                                               f"member or with a member shorter than {self.sequence_length} frames "
                                               f"left out : file://{os.path.abspath(labels_file)}")
        if shuffle:
            random.shuffle(self.pairs)

        # Every pair by identifier, to name the evaluated pairs in the evaluation outputs
        self.pair_records = {pair["identifier"]: pair for pair in self.pairs}

        # Name of every class, taken from the categories of the manifest
        self.class_names = get_class_names(categories=manifest["categories"], label_dictionary=label_dictionary)

    def __len__(self) -> int:
        """
        Gives the number of kept pairs.

        Returns:
            int: The number of pairs.
        """
        return len(self.pairs)

    def __getitem__(self, pair_index: int) -> Dict[str, Any]:
        """
        Reads and preprocesses the frames of the two members of one pair, with its class.

        Args:
            pair_index (int): The index of the pair.

        Returns:
            Dict[str, Any]: The uint8 [2 x sequence_length, 3, image_size, image_size] BGR frames (the before window,
                then the after window, each with its black padding frames last), the [2 x sequence_length] boolean
                sequence mask (True for the real frames), the int64 class index of the pair and its identifier.
        """
        pair = self.pairs[pair_index]

        # One window per member, the before window followed by the after window along the frame axis
        member_frames, member_masks = [], []
        for member_name in PAIR_MEMBERS:
            frames, sequence_mask = read_window_frames(data_directory=self.data_directory,
                                                       frame_files=pair[member_name],
                                                       sequence_length=self.sequence_length,
                                                       image_size=self.image_size, keep_ratio=self.keep_ratio)
            member_frames.append(frames)
            member_masks.append(sequence_mask)

        return {"images": torch.cat(tensors=member_frames, dim=0),
                "sequence_mask": torch.cat(tensors=member_masks, dim=0),
                "labels": torch.tensor(data=pair["label"], dtype=torch.int64), "sample_identifiers": pair["identifier"]}

    def describe_sample(self, sample_identifier: int) -> Dict[str, str]:
        """
        Names an evaluated pair in the predictions file written at every evaluation, from its identifier in the
        batches.

        Args:
            sample_identifier (int): The identifier of the pair, as given in the 'sample_identifiers' of a batch.

        Returns:
            Dict[str, str]: The 'pair' column, the name of the pair in the manifest.
        """
        return {"pair": self.pair_records[sample_identifier]["name"]}

    def get_class_counts(self) -> List[int]:
        """
        Counts the pairs of every class, used to weight the classes of the loss inversely to their frequency.

        Returns:
            List[int]: The number of pairs of every class index.
        """
        class_counts = [0] * self.number_classes
        for pair in self.pairs:
            class_counts[pair["label"]] += 1
        return class_counts
