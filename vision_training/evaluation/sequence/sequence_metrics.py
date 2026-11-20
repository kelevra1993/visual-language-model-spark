"""
Metrics of the sequence labelling task (M2M): frame accuracy and exact matches of whole sequences, ignoring the
padding frames (IGNORED_LABEL_INDEX). Used by the sequence labelling trainer to evaluate a checkpoint.
"""

import torch

from vision_training.core.contracts import IGNORED_LABEL_INDEX


def compute_frame_accuracy(predicted_labels: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the share of the labelled frames whose predicted class is right, the batch metric of the sequence
    labelling task, the padding frames and the frames of unmapped categories (label -100) being left out.

    Args:
        predicted_labels (torch.Tensor): The [B, T] predicted class index of every frame.
        labels (torch.Tensor): The [B, T] class index of every frame, -100 for the frames without class.

    Returns:
        torch.Tensor: The scalar float32 accuracy, 0 without labelled frame.
    """
    labelled_frames = labels != IGNORED_LABEL_INDEX
    correct_frames = (predicted_labels == labels) & labelled_frames
    return correct_frames.sum().to(dtype=torch.float32) / labelled_frames.sum().clamp(min=1).to(dtype=torch.float32)


def compute_sequence_exact_matches(predicted_labels: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Tells, for every sequence holding a labelled frame, whether every one of its labelled frames is predicted right,
    the strictest metric of the sequence labelling task. A sequence whose frames are all without class (unmapped
    categories) has nothing to match and is left out, so that it neither counts as an exact match nor in the number of
    test sequences, which the sequence labelling trainer takes from these booleans.

    Args:
        predicted_labels (torch.Tensor): The [B, T] predicted class index of every frame.
        labels (torch.Tensor): The [B, T] class index of every frame, -100 for the frames without class.

    Returns:
        torch.Tensor: The [S] booleans of the S sequences holding a labelled frame, in their order, True for those
            whose labelled frames are all right.
    """
    unlabelled_frames = labels == IGNORED_LABEL_INDEX
    exact_matches = ((predicted_labels == labels) | unlabelled_frames).all(dim=1)
    return exact_matches[~unlabelled_frames.all(dim=1)]
