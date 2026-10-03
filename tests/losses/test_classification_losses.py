"""
Tests of vision_training/losses/classification_losses.py: the weighted, smoothed multi-class and multi-label losses, the
probabilities following the output mode and the balanced class weights.
"""

import torch

from vision_training.losses.classification_losses import (compute_balanced_class_weights, compute_classification_loss,
                                                          convert_logits_to_probabilities)


def test_multi_class_loss_is_weighted_smoothed_cross_entropy() -> None:
    """
    Verifies that the multi-class loss is the cross entropy of the softmax, with the class weights and the label
    smoothing of the configuration.
    """
    class_logits = torch.tensor(data=[[2.0, 0.5, -1.0], [0.1, 0.2, 0.3]])
    labels = torch.tensor(data=[0, 2])
    class_weights = torch.tensor(data=[1.0, 1.0, 3.0])

    loss = compute_classification_loss(class_logits=class_logits, labels=labels, output_mode="multi_class",
                                       label_smoothing=0.1, class_weights=class_weights)
    expected_loss = torch.nn.functional.cross_entropy(input=class_logits, target=labels, weight=class_weights,
                                                      label_smoothing=0.1)
    assert torch.allclose(input=loss, other=expected_loss)


def test_multi_label_loss_smooths_targets_towards_one_half() -> None:
    """
    Verifies that the multi-label loss is the binary cross entropy of the sigmoid of every class, the targets being
    smoothed towards 0.5 (1 becomes 1 - s / 2 and 0 becomes s / 2), and every class weighted by its class weight.
    """
    class_logits = torch.tensor(data=[[1.0, -2.0], [0.0, 3.0]])
    labels = torch.tensor(data=[[1.0, 0.0], [0.0, 1.0]])
    class_weights = torch.tensor(data=[1.0, 2.0])

    loss = compute_classification_loss(class_logits=class_logits, labels=labels, output_mode="multi_label",
                                       label_smoothing=0.2, class_weights=class_weights)
    smoothed_targets = torch.tensor(data=[[0.9, 0.1], [0.1, 0.9]])
    expected_loss = torch.nn.functional.binary_cross_entropy_with_logits(input=class_logits, target=smoothed_targets,
                                                                         weight=class_weights)
    assert torch.allclose(input=loss, other=expected_loss)


def test_probabilities_follow_the_output_mode() -> None:
    """
    Verifies that the logits become softmax probabilities with one class per sample, and independent sigmoid
    probabilities with several.
    """
    class_logits = torch.tensor(data=[[1.0, 2.0, 3.0]])
    assert torch.allclose(input=convert_logits_to_probabilities(class_logits=class_logits, output_mode="multi_class"),
                          other=torch.softmax(input=class_logits, dim=-1))
    assert torch.allclose(input=convert_logits_to_probabilities(class_logits=class_logits, output_mode="multi_label"),
                          other=torch.sigmoid(input=class_logits))


def test_balanced_class_weights() -> None:
    """
    Verifies that the balanced class weights are inversely proportional to the class counts and average 1 over the
    training samples, a class without any sample being weighted like a class with a single one.
    """
    class_weights = compute_balanced_class_weights(class_counts=[10, 30, 60])
    assert torch.allclose(input=class_weights, other=torch.tensor(data=[100 / 30, 100 / 90, 100 / 180]))
    assert torch.isclose(input=(class_weights * torch.tensor(data=[10.0, 30.0, 60.0])).sum() / 100.0,
                         other=torch.tensor(data=1.0))
    assert torch.allclose(input=compute_balanced_class_weights(class_counts=[0, 1]),
                          other=torch.tensor(data=[1.0, 1.0]))
