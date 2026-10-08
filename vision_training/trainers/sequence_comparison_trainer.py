"""
Trainer of the sequence comparison task (CMP): pairs of sequences stored as the before window followed by the after
window of every pair, split again into the two members just before the sequence comparator, the rest being the
sequence classification trainer. Registered for the sequence comparison task in the task registry.
"""

import torch

from pathlib import Path
from typing import Any, Dict, Tuple

from vision_training.data.sequence.sequence_pair_dataset import PAIR_MEMBERS, split_pair_members
from vision_training.data.split_description import count_split_pairs
from vision_training.trainers.sequence_classification_trainer import (FRAME_GRID_COLUMNS,
                                                                      SequenceClassificationTrainer)
from vision_training.utilities.visualization import build_frame_grid


class SequenceComparisonTrainer(SequenceClassificationTrainer):
    """
    Trainer of the sequence comparison task: the sequence comparator trained on pairs of sequences, a before member
    and an after member, read from the 'pairs' manifests (Native loader), one class per pair. A batch stores the before
    window followed by the after window of every pair along the frame axis, so that the augmentation (one draw per
    pair), the input viewer and the evaluation of the sequence classification trainer apply unchanged, the members
    being split again just before the model. The losses, the accuracy tracked on every batch and the evaluation of the
    checkpoints are those of the classification task, a pair being classified as a whole.
    """

    TASK_NAME = "sequence_comparison"
    SAMPLE_NAME = "Pairs"

    # Labels read from the manifests: one class per pair of sequences
    LABEL_TYPE = "pair"

    # Member windows of a pair stored one after the other along its frame axis, the before one then the after one
    MEMBERS_PER_SAMPLE = len(PAIR_MEMBERS)

    def describe_split(self, split_data_folder: Path) -> Tuple[int, str]:
        """
        Counts the pairs, frames and kept pairs of a comparison split from its labels.json manifest.

        Args:
            split_data_folder (Path): The root folder of the split.

        Returns:
            Tuple[int, str]: The number of kept pairs (the samples of an epoch) and their description with the number of
                pairs and frames.
        """
        number_pairs, number_frames, number_kept_pairs = count_split_pairs(split_data_folder=split_data_folder,
                                                                           sequence_settings=self.sequence_settings)
        return number_kept_pairs, (f"{number_pairs} pairs, {number_frames} frames, {number_kept_pairs} kept pairs of "
                                   f"{self.get_window_size_qualifier()}2 x "
                                   f"{self.sequence_settings['sequence_length']} frames")

    def split_pair_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Splits the frames and the sequence mask of a prepared batch of pairs into the arguments of the comparison
        model.

        Args:
            batch (Dict[str, Any]): The prepared batch, with its [B, 2T, 3, H, W] frames and [B, 2T] sequence mask.

        Returns:
            Dict[str, torch.Tensor]: The 'before_images', 'after_images', 'before_mask' and 'after_mask' of the batch.
        """
        # Both member windows have the number of frames of the batch, the sequence length or its longest member
        before_images, after_images = split_pair_members(pair_tensor=batch["images"])
        before_mask, after_mask = split_pair_members(pair_tensor=batch["sequence_mask"])
        return {"before_images": before_images, "after_images": after_images, "before_mask": before_mask,
                "after_mask": after_mask}

    def forward_pass(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the comparison model on the members of the pairs with their labels, so that it returns its loss.

        Args:
            batch (Dict[str, Any]): The prepared batch.

        Returns:
            Dict[str, Any]: The model outputs, with its loss.
        """
        return self.model(**self.split_pair_batch(batch=batch), labels=batch["labels"])

    def predict_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the comparison model on the members of the pairs without labels, as in deployment.

        Args:
            batch (Dict[str, Any]): The prepared test batch.

        Returns:
            Dict[str, Any]: The class logits and probabilities.
        """
        return self.model(**self.split_pair_batch(batch=batch))

    def build_sample_frame_grid(self, frames: torch.Tensor, sequence_mask: torch.Tensor,
                                label: torch.Tensor) -> torch.Tensor:
        """
        Lays out the real frames of the before member of a pair above those of its after member in the input viewer,
        every frame captioned with its member.

        Args:
            frames (torch.Tensor): The [2T, 3, H, W] frames of the pair, the before window then the after window.
            sequence_mask (torch.Tensor): The [2T] booleans, True for the real frames.
            label (torch.Tensor): The label of the pair, written in the caption of the window.

        Returns:
            torch.Tensor: The [3, height, width] uint8 grid, the before member above the after member.
        """
        member_frames = split_pair_members(pair_tensor=frames.unsqueeze(dim=0))
        member_masks = split_pair_members(pair_tensor=sequence_mask.unsqueeze(dim=0))

        # One grid per member, of the same width since it only depends on the number of columns
        member_grids = []
        for member_name, member_frame_batch, member_mask_batch in zip(PAIR_MEMBERS, member_frames, member_masks):
            member_grids.append(build_frame_grid(frames=member_frame_batch[0], sequence_mask=member_mask_batch[0],
                                                 frame_captions=[member_name] * int(member_mask_batch[0].sum()),
                                                 number_columns=FRAME_GRID_COLUMNS))
        return torch.cat(tensors=member_grids, dim=1)
