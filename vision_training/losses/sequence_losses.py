import torch

from vision_training.core.contracts import IGNORED_LABEL_INDEX


def compute_sequence_labelling_loss(class_logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float,
                                    class_weights: torch.Tensor) -> torch.Tensor:
    """
    Computes the loss of the sequence labelling task: the cross entropy of the softmax of every frame, averaged over
    the labelled frames only, the padding frames and the frames of unmapped categories (label -100) being ignored.
    It supports label smoothing and per-class weights like the classification loss.

    Args:
        class_logits (torch.Tensor): The [B, T, K] class logits of every frame.
        labels (torch.Tensor): The [B, T] int64 class index of every frame, -100 for the frames without class.
        label_smoothing (float): The label smoothing, 0 for none.
        class_weights (torch.Tensor): The [K] weight of every class in the loss.

    Returns:
        torch.Tensor: The scalar loss.
    """
    return torch.nn.functional.cross_entropy(input=class_logits.reshape(-1, class_logits.shape[-1]),
                                             target=labels.reshape(-1), weight=class_weights,
                                             ignore_index=IGNORED_LABEL_INDEX, label_smoothing=label_smoothing)
