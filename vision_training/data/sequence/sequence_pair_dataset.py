"""
SequencePairDataset, the Native (OpenCV, CPU) dataset of the sequence comparison task, reading the before and after
members of every pair of a 'labels.json' split whole, both padded to the longer member and stored one after the other,
the pairs with a member longer than 'Data.sequence_settings.maximum_sequence_length' being left out with a warning, and
the helper splitting the pair tensors. Used by data/loaders.py and the sequence comparison trainer.
"""

import json
import random
import torch

from torch.utils.data import Dataset
from typing import Any, Dict, List, Tuple

from vision_training.data.label_mapping import get_class_names
from vision_training.data.sequence.sequence_dataset import (read_sequence_frames, warn_long_sequence,
                                                            warn_long_sequence_total)
from vision_training.utilities.os_utilities import format_path_link, print_color

# Members of a pair in the manifests, in the order they are stacked along the frame axis of a sample
PAIR_MEMBERS = ["before", "after"]


def split_pair_members(pair_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Splits the frames or the sequence mask of a batch of pairs, stored as the before member followed by the after
    member along the frame axis, into the before and after members given to the comparison model. Both members have
    the same number of frames, the longest member of the batch, so the frame axis is split in two halves.

    Args:
        pair_tensor (torch.Tensor): The [B, 2 x T, ...] frames or [B, 2 x T] mask.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The [B, T, ...] before member and after member.

    Raises:
        ValueError: If the frame axis does not hold two members of the same number of frames.
    """
    if pair_tensor.shape[1] % 2 != 0:
        raise ValueError(f"A batch of pairs must hold two members of the same number of frames, got "
                         f"{pair_tensor.shape[1]} frames per pair.")
    member_length = pair_tensor.shape[1] // 2
    return pair_tensor[:, :member_length], pair_tensor[:, member_length:]


class SequencePairDataset(Dataset):
    """
    Dataset of the sequence comparison task, read with OpenCV on the CPU (Native loader).

    A split is a folder holding the frames in 'data/' and a 'labels.json' manifest listing the 'categories' and the
    'pairs', each pair giving its 'id', 'name', the ordered frames of its 'before' and 'after' members (paths relative
    to 'data/') and its 'category_id'. Both members are given whole, the pairs with a member longer than
    'maximum_sequence_length' frames being left out with a red warning. A sample stores the before member followed by
    the after member along the frame axis, both padded to the longer member with black frames, so that the generic
    code of the trainers (augmentation, input viewer, evaluation) handles a pair as one sequence of 2 x T frames, both
    members sharing the same augmentation draws. The collate of the loader pads both members of every pair to the
    longest member of the batch.
    """

    def __init__(self, data_directory: str, labels_file: str, image_size: int, keep_ratio: bool,
                 label_dictionary: Dict[Any, int], sequence_settings: Dict[str, Any], split_name: str,
                 shuffle: bool = True) -> None:
        """
        Reads the manifest of the split and keeps its pairs whose members hold at most 'maximum_sequence_length'
        frames.

        Args:
            data_directory (str): The folder holding the frames of the pairs.
            labels_file (str): The 'labels.json' manifest of the split.
            image_size (int): The size of the square network input.
            keep_ratio (bool): Whether to keep the aspect ratio of the frames, padding them.
            label_dictionary (Dict[Any, int]): The class index of every raw category identifier, the pairs of unmapped
                categories being left out.
            sequence_settings (Dict[str, Any]): The 'Data.sequence_settings' section: 'maximum_sequence_length' (the
                largest number of frames of a member).
            split_name (str): "Training", "Validation" or "Test", named in the warnings of the left out pairs.
            shuffle (bool): Whether to shuffle the pairs once at creation. Defaults to True.
        """
        self.data_directory = data_directory
        self.image_size = image_size
        self.keep_ratio = keep_ratio
        self.maximum_sequence_length = sequence_settings["maximum_sequence_length"]
        self.number_classes = len(set(label_dictionary.values()))

        with open(file=labels_file, mode="r") as labels_file_handler:
            manifest = json.load(fp=labels_file_handler)

        # Keep the pairs of mapped categories whose two members both hold frames, and not too many
        self.pairs = []
        number_left_out_pairs = 0
        number_long_pairs = 0
        for pair in manifest["pairs"]:
            member_lengths = [len(pair[member_name]) for member_name in PAIR_MEMBERS]
            if pair["category_id"] not in label_dictionary or min(member_lengths) == 0:
                number_left_out_pairs += 1
                continue

            # Pairs with a member longer than the maximum sequence length are left out with a warning
            if max(member_lengths) > self.maximum_sequence_length:
                warn_long_sequence(sample_kind="Pair", sample_name=pair["name"], split_name=split_name,
                                   number_frames=max(member_lengths),
                                   maximum_sequence_length=self.maximum_sequence_length)
                number_long_pairs += 1
                continue
            self.pairs.append({"identifier": len(self.pairs), "pair_identifier": pair["id"], "name": pair["name"],
                               **{member_name: pair[member_name] for member_name in PAIR_MEMBERS},
                               "label": label_dictionary[pair["category_id"]]})
        warn_long_sequence_total(sample_kind="Pair", number_long_samples=number_long_pairs, split_name=split_name,
                                 maximum_sequence_length=self.maximum_sequence_length)
        if number_left_out_pairs > 0:
            print_color(color="yellow", output=f"{number_left_out_pairs} pairs of unmapped categories or with an empty "
                                               f"member left out : {format_path_link(path=labels_file)}")
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
            Dict[str, Any]: The uint8 [2 x T, 3, image_size, image_size] BGR frames, T being the frames of the longer
                member (the before member, then the after member, each with its black padding frames last), the
                [2 x T] boolean sequence mask (True for the real frames), the int64 class index of the pair and its
                identifier.
        """
        pair = self.pairs[pair_index]

        # Both members padded to the longer one, the before member followed by the after member along the frame axis
        member_length = max(len(pair[member_name]) for member_name in PAIR_MEMBERS)
        member_frames, member_masks = [], []
        for member_name in PAIR_MEMBERS:
            frames, sequence_mask = read_sequence_frames(data_directory=self.data_directory,
                                                         frame_files=pair[member_name], number_frames=member_length,
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

    def get_sample_name(self, sample_identifier: int) -> str:
        """
        Gives the short name of an evaluated pair used in the file names of the error images of the evaluation.

        Args:
            sample_identifier (int): The identifier of the pair, as given in the 'sample_identifiers' of a batch.

        Returns:
            str: The name of the pair in the manifest, its folders joined with underscores.
        """
        return self.pair_records[sample_identifier]["name"].replace("/", "_")

    def get_sample_class_indices(self) -> Dict[int, List[int]]:
        """
        Lists the class of every pair, from which the sequence comparison trainer spreads the fixed test pairs drawn
        at every evaluation over the classes.

        Returns:
            Dict[int, List[int]]: The class index of every pair, in a list, keyed by its identifier.
        """
        return {pair["identifier"]: [pair["label"]] for pair in self.pairs}

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
