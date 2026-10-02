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
    Tells, for every sequence, whether every one of its labelled frames is predicted right, the strictest metric of
    the sequence labelling task.

    Args:
        predicted_labels (torch.Tensor): The [B, T] predicted class index of every frame.
        labels (torch.Tensor): The [B, T] class index of every frame, -100 for the frames without class.

    Returns:
        torch.Tensor: The [B] booleans, True for the sequences whose labelled frames are all right.
    """
    return ((predicted_labels == labels) | (labels == IGNORED_LABEL_INDEX)).all(dim=1)
