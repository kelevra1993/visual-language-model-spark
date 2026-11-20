"""
Tests of vision_training/losses/classification_losses.py: the weighted, smoothed multi-class and multi-label losses, the
balanced class weights and the decorrelation loss of the hidden features.
"""

import math
import torch

from vision_training.losses.classification_losses import (compute_balanced_class_weights, compute_classification_loss,
                                                          compute_decorrelation_loss)


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


def test_decorrelation_loss_sums_the_squared_off_diagonal_correlations() -> None:
    """
    Verifies the decorrelation loss of the hidden features: the weight times the sum of the squared off-diagonal terms
    of the correlation matrix of the features over the batch, whatever their scale and offset, in float32 under the
    bfloat16 autocast, with the mean absolute off-diagonal correlation; 0 for uncorrelated features and a full penalty
    for copies of one feature.
    """
    generator = torch.Generator().manual_seed(0)
    hidden_features = torch.randn(size=(64, 6), generator=generator) @ torch.randn(size=(6, 6), generator=generator)
    correlation = torch.corrcoef(input=hidden_features.T)
    off_diagonal = correlation - torch.diag(input=torch.diagonal(input=correlation))

    outputs = compute_decorrelation_loss(hidden_features=hidden_features * 7.0 + 3.0, weight=0.2)
    assert torch.allclose(input=outputs["decorrelation_loss"], other=0.2 * off_diagonal.pow(exponent=2).sum(),
                          rtol=1e-4)
    assert torch.allclose(input=outputs["hidden_feature_correlation"], other=off_diagonal.abs().sum() / 30, rtol=1e-4)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast_outputs = compute_decorrelation_loss(hidden_features=hidden_features, weight=0.2)
    assert autocast_outputs["decorrelation_loss"].dtype == torch.float32

    # Orthogonal centred columns (the QR factor of centred columns) are uncorrelated, while copies of one feature are
    # fully correlated: D(D - 1) off-diagonal terms of 1
    centred_features = torch.randn(size=(32, 5), generator=generator)
    centred_features = centred_features - centred_features.mean(dim=0, keepdim=True)
    uncorrelated_features = torch.linalg.qr(A=centred_features)[0]
    uncorrelated_outputs = compute_decorrelation_loss(hidden_features=uncorrelated_features, weight=1.0)
    assert math.isclose(float(uncorrelated_outputs["decorrelation_loss"]), 0.0, abs_tol=1e-4)
    copied_features = torch.randn(size=(32, 1), generator=generator).repeat(1, 4)
    copied_outputs = compute_decorrelation_loss(hidden_features=copied_features, weight=1.0)
    assert math.isclose(float(copied_outputs["decorrelation_loss"]), 12.0, rel_tol=1e-3)
    assert math.isclose(float(copied_outputs["hidden_feature_correlation"]), 1.0, rel_tol=1e-3)
