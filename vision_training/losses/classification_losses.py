"""
Classification losses shared by the classification-like tasks: the multi-class or multi-label loss of the logits,
the conversion of the logits to probabilities and the balanced class weights. Used by the image classifier, the
sequence models and the classification trainer.
"""

import torch

from typing import List


def compute_classification_loss(class_logits: torch.Tensor, labels: torch.Tensor, output_mode: str,
                                label_smoothing: float, class_weights: torch.Tensor) -> torch.Tensor:
    """
    Computes the classification loss of the classification-like tasks: the cross entropy of the softmax for a single
    class per sample, or the binary cross entropy of the sigmoid of every class for several classes per sample. Both
    support label smoothing and per-class weights (e.g. to balance rare classes).

    Args:
        class_logits (torch.Tensor): The [N, K] class logits.
        labels (torch.Tensor): The [N] int64 class indices ("multi_class"), or the [N, K] multi-hot targets
            ("multi_label").
        output_mode (str): "multi_class" or "multi_label".
        label_smoothing (float): The label smoothing, 0 for none.
        class_weights (torch.Tensor): The [K] weight of every class in the loss.

    Returns:
        torch.Tensor: The scalar loss.
    """
    if output_mode == "multi_class":
        return torch.nn.functional.cross_entropy(input=class_logits, target=labels, weight=class_weights,
                                                 label_smoothing=label_smoothing)

    # Smooth the multi-hot targets towards 0.5, the same amount for the positive and the negative classes
    smoothed_targets = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return torch.nn.functional.binary_cross_entropy_with_logits(input=class_logits, target=smoothed_targets,
                                                                weight=class_weights)


def convert_logits_to_probabilities(class_logits: torch.Tensor, output_mode: str) -> torch.Tensor:
    """
    Turns class logits into the class probabilities returned by the classification-like models: a softmax over the
    classes for a single class per sample, an independent sigmoid per class for several classes per sample.

    Args:
        class_logits (torch.Tensor): The [..., K] class logits.
        output_mode (str): "multi_class" or "multi_label".

    Returns:
        torch.Tensor: The [..., K] class probabilities.
    """
    if output_mode == "multi_class":
        return torch.softmax(input=class_logits, dim=-1)
    return torch.sigmoid(input=class_logits)


def compute_balanced_class_weights(class_counts: List[int]) -> torch.Tensor:
    """
    Computes class weights inversely proportional to the class frequencies of the training split, so that every class
    weighs the same in the loss whatever its number of samples ('ClassificationHead.class_weighting: "balanced"').
    A class without any sample gets the weight of a class with a single one.

    Args:
        class_counts (List[int]): The number of training samples of every class.

    Returns:
        torch.Tensor: The [K] class weights, averaging 1 over the training samples.
    """
    counts = torch.tensor(data=class_counts, dtype=torch.float64).clamp(min=1.0)
    return (counts.sum() / (len(class_counts) * counts)).to(dtype=torch.float32)
