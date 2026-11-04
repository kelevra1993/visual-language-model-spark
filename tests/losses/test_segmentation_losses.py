"""
Tests of vision_training/losses/segmentation_losses.py: the pixel cross entropy (ignored pixels, class weights, label
smoothing), the soft Dice loss (absent classes, ignored pixels), the Lovász-Softmax
loss (1 - IoU of hard predictions, absent classes, ignored pixels) and the boundary loss (errors weighted by their
closeness to the true edges).
"""

import pytest
import torch

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.losses.segmentation_losses import (compute_boundary_loss, compute_dice_loss,
                                                        compute_lovasz_softmax_loss,
                                                        compute_segmentation_cross_entropy)
from vision_training.utilities.mask_morphology import compute_edge_distance_weights


def build_logits_and_masks() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Builds random logits of 3 classes on two 4 x 4 images, with masks holding every class and two ignored pixels.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The [2, 3, 4, 4] logits and the [2, 4, 4] masks.
    """
    generator = torch.Generator().manual_seed(0)
    class_logits = torch.randn(size=(2, 3, 4, 4), generator=generator)
    masks = torch.randint(low=0, high=3, size=(2, 4, 4), generator=generator)
    masks[0, 0, 0] = IGNORED_LABEL_INDEX
    masks[1, 3, 3] = IGNORED_LABEL_INDEX
    return class_logits, masks


def test_cross_entropy_without_focal_factor_is_the_weighted_smoothed_cross_entropy() -> None:
    """
    Verifies that with a focal gamma of 0 the loss is exactly the weighted, smoothed cross entropy of the labelled
    pixels, as computed by PyTorch with its ignore index.
    """
    class_logits, masks = build_logits_and_masks()
    class_weights = torch.tensor(data=[0.5, 1.0, 3.0])

    loss = compute_segmentation_cross_entropy(class_logits=class_logits, masks=masks, label_smoothing=0.1,
                                              class_weights=class_weights, focal_gamma=0.0)
    expected_loss = torch.nn.functional.cross_entropy(input=class_logits, target=masks, weight=class_weights,
                                                      ignore_index=IGNORED_LABEL_INDEX, label_smoothing=0.1)
    assert torch.allclose(input=loss, other=expected_loss)


def test_dice_loss_leaves_out_the_absent_classes() -> None:
    """
    Verifies that a class absent from the labelled pixels does not enter the mean Dice, whatever its probabilities,
    and that a batch without labelled pixel gives a loss of 0.
    """
    masks = torch.tensor(data=[[[0, 0, 1, 1]]])
    logits = torch.zeros(size=(1, 3, 1, 4))
    logits[:, 2] = torch.tensor(data=[5.0, -5.0, 5.0, -5.0])

    # Mean Dice of the present classes 0 and 1 only, computed by hand from the probabilities
    probabilities = torch.softmax(input=logits, dim=1)[0, :, 0]
    targets = torch.tensor(data=[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    present_dice = [2.0 * (probabilities[class_index] * targets[class_index]).sum() /
                    (probabilities[class_index].sum() + targets[class_index].sum()) for class_index in [0, 1]]
    expected_loss = 1.0 - (present_dice[0] + present_dice[1]) / 2.0
    assert torch.allclose(input=compute_dice_loss(class_logits=logits, masks=masks), other=expected_loss)
    ignored_masks = torch.full(size=(1, 1, 4), fill_value=IGNORED_LABEL_INDEX)
    assert compute_dice_loss(class_logits=logits, masks=ignored_masks) == 0.0



def test_lovasz_loss_of_hard_predictions_is_one_minus_the_iou() -> None:
    """
    Verifies that for hard predictions (probabilities of 0 or 1) the Lovász-Softmax loss is exactly the mean of
    1 - IoU over the classes present in the labelled pixels: an absent class predicted on some pixels and the ignored
    pixels being left out.
    """
    # Class 0 on the first 4 pixels, class 1 on the next 4, the last pixel ignored; predictions: class 0 on the first
    # 3 pixels, class 1 on the next 3, the absent class 2 on the last 3
    masks = torch.tensor(data=[[[0, 0, 0, 0, 1, 1, 1, 1, IGNORED_LABEL_INDEX]]])
    predicted_classes = torch.tensor(data=[0, 0, 0, 1, 1, 1, 2, 2, 2])
    predicted_one_hot = torch.nn.functional.one_hot(input=predicted_classes, num_classes=3).float()
    class_logits = 100.0 * predicted_one_hot.T.reshape(1, 3, 1, 9)

    # IoU of class 0: 3 / 4, of class 1: 2 / (4 + 3 - 2) = 2 / 5
    expected_loss = ((1.0 - 3 / 4) + (1.0 - 2 / 5)) / 2.0
    assert compute_lovasz_softmax_loss(class_logits=class_logits, masks=masks).item() == pytest.approx(
        expected=expected_loss, abs=1e-6)

    # A perfect prediction gives 0, a batch without labelled pixel gives 0
    true_one_hot = torch.nn.functional.one_hot(input=masks.clamp(min=0)[0, 0], num_classes=3).float()
    perfect_logits = 100.0 * true_one_hot.T.reshape(1, 3, 1, 9)
    assert compute_lovasz_softmax_loss(class_logits=perfect_logits, masks=masks).item() == pytest.approx(
        expected=0.0, abs=1e-6)
    ignored_masks = torch.full(size=(1, 1, 9), fill_value=IGNORED_LABEL_INDEX)
    assert compute_lovasz_softmax_loss(class_logits=class_logits, masks=ignored_masks).item() == 0.0


def test_boundary_loss_only_counts_the_pixels_near_the_edges() -> None:
    """
    Verifies that the boundary loss is the cross entropy of the pixels weighted by their closeness to the true edge,
    so that any logits on the pixels farther than the boundary distance change nothing, and that a mask without edge
    gives a loss of 0.
    """
    masks = torch.zeros(size=(1, 4, 16), dtype=torch.int64)
    masks[0, :, 8:] = 1
    class_logits = torch.randn(size=(1, 2, 4, 16), generator=torch.Generator().manual_seed(0))
    edge_weights = compute_edge_distance_weights(masks=masks, maximum_distance=2)
    loss = compute_boundary_loss(class_logits=class_logits, masks=masks, edge_weights=edge_weights)

    # Weighted mean of the cross entropy of every pixel
    pixel_losses = torch.nn.functional.cross_entropy(input=class_logits, target=masks, reduction="none")
    assert loss.item() == pytest.approx(expected=float((pixel_losses * edge_weights).sum() / edge_weights.sum()))

    # Pixels 3 columns or more away from the edge between columns 7 and 8 take no part
    changed_logits = class_logits.clone()
    changed_logits[..., :5] = 50.0
    changed_logits[..., 11:] = -50.0
    assert compute_boundary_loss(class_logits=changed_logits, masks=masks, edge_weights=edge_weights).item() == \
        pytest.approx(expected=loss.item())

    # A mask of a single class has no edge
    uniform_masks = torch.zeros(size=(1, 4, 16), dtype=torch.int64)
    assert compute_boundary_loss(class_logits=class_logits, masks=uniform_masks,
                                 edge_weights=compute_edge_distance_weights(masks=uniform_masks,
                                                                            maximum_distance=2)).item() == 0.0
