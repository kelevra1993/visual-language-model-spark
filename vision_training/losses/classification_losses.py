"""
Classification losses shared by the classification-like tasks: the multi-class or multi-label loss of the logits,
the conversion of the logits to probabilities and the balanced class weights, and the decorrelation loss of the hidden
features of the image classifier ('ClassificationHead.decorrelation'). Used by the image classifier, the sequence
models and the classification trainer.
"""

import torch

from typing import Dict, List

from vision_training.losses.self_supervised_losses import compute_feature_cross_correlation


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


def compute_decorrelation_loss(hidden_features: torch.Tensor, weight: float) -> Dict[str, torch.Tensor]:
    """
    Computes the decorrelation loss the image classifier adds to its classification loss
    ('ClassificationHead.decorrelation'), so that the units of one hidden layer of its head carry different
    information: every feature is normalized over the batch, and the squared off-diagonal terms of their [D, D]
    correlation matrix are summed and multiplied by the weight. It is the redundancy reduction term of Barlow Twins
    without its invariance term, which needs two views of every image (the DeCov penalty of Cogswell et al., 2016).
    The correlation matrix of N images has a rank of at most N - 1, so the sum keeps a floor with more features than
    images, and the sampling noise alone gives about D(D - 1) / (N - 1). A feature whose variance over the batch is far
    below the epsilon of the normalization (1e-5) counts as constant, with correlations near 0. The mean absolute
    off-diagonal correlation is tracked by the classification trainer.

    Args:
        hidden_features (torch.Tensor): The [N, D] features of the decorrelated hidden layer, before its ReLU.
        weight (float): The weight of the sum of the squared off-diagonal correlations in the loss.

    Returns:
        Dict[str, torch.Tensor]: The scalar 'decorrelation_loss', and the 'hidden_feature_correlation' (mean absolute
            correlation of two different features) without gradient.
    """
    with torch.autocast(device_type=hidden_features.device.type, enabled=False):
        feature_dimension = hidden_features.shape[1]

        # Correlation of every pair of features over the batch, the same features being given as both views
        correlation = compute_feature_cross_correlation(first_embeddings=hidden_features,
                                                        second_embeddings=hidden_features)
        off_diagonal = correlation - torch.diag_embed(input=torch.diagonal(input=correlation))
        decorrelation_loss = weight * off_diagonal.pow(exponent=2).sum()

        # Mean absolute correlation of two different features
        with torch.no_grad():
            number_off_diagonal_terms = max(feature_dimension * (feature_dimension - 1), 1)
            hidden_feature_correlation = off_diagonal.abs().sum() / number_off_diagonal_terms
    return {"decorrelation_loss": decorrelation_loss, "hidden_feature_correlation": hidden_feature_correlation}


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
