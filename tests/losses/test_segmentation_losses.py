"""
Tests of vision_training/losses/segmentation_losses.py: the pixel cross entropy (ignored pixels, class weights, label
smoothing, focal factor) and the soft Dice loss (perfect prediction, absent classes, ignored pixels).
"""

import torch

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.losses.segmentation_losses import compute_dice_loss, compute_segmentation_cross_entropy


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


def test_ignored_pixels_give_no_loss_nor_gradient() -> None:
    """
    Verifies that the logits of the ignored pixels change neither the cross entropy nor the Dice loss, and receive no
    gradient.
    """
    class_logits, masks = build_logits_and_masks()
    class_logits.requires_grad_(requires_grad=True)
    loss = compute_segmentation_cross_entropy(class_logits=class_logits, masks=masks, label_smoothing=0.0,
                                              class_weights=torch.ones(size=(3,)), focal_gamma=2.0) + \
        compute_dice_loss(class_logits=class_logits, masks=masks)
    loss.backward()
    assert torch.all(class_logits.grad[0, :, 0, 0] == 0) and torch.all(class_logits.grad[1, :, 3, 3] == 0)

    # Other logits on the ignored pixels give the same losses
    changed_logits = class_logits.detach().clone()
    changed_logits[0, :, 0, 0] = torch.tensor(data=[10.0, -10.0, 5.0])
    assert torch.allclose(input=compute_dice_loss(class_logits=class_logits.detach(), masks=masks),
                          other=compute_dice_loss(class_logits=changed_logits, masks=masks))
    assert torch.allclose(input=compute_segmentation_cross_entropy(class_logits=class_logits.detach(), masks=masks,
                                                                   label_smoothing=0.0,
                                                                   class_weights=torch.ones(size=(3,)),
                                                                   focal_gamma=2.0),
                          other=compute_segmentation_cross_entropy(class_logits=changed_logits, masks=masks,
                                                                   label_smoothing=0.0,
                                                                   class_weights=torch.ones(size=(3,)),
                                                                   focal_gamma=2.0))


def test_focal_factor_lowers_the_loss_of_well_classified_pixels() -> None:
    """
    Verifies the focal factor: a pixel predicted with high confidence weighs much less than with the plain cross
    entropy, while a badly classified pixel keeps most of its loss.
    """
    masks = torch.tensor(data=[[[0, 1]]])
    class_logits = torch.tensor(data=[[[[4.0, 0.0]], [[0.0, 0.0]]]])
    class_weights = torch.ones(size=(2,))

    plain_loss = compute_segmentation_cross_entropy(class_logits=class_logits, masks=masks, label_smoothing=0.0,
                                                    class_weights=class_weights, focal_gamma=0.0)
    focal_loss = compute_segmentation_cross_entropy(class_logits=class_logits, masks=masks, label_smoothing=0.0,
                                                    class_weights=class_weights, focal_gamma=2.0)
    assert focal_loss < plain_loss
    assert focal_loss > plain_loss * 0.2


def test_dice_loss_of_a_perfect_prediction_is_zero() -> None:
    """
    Verifies that very confident correct logits give a Dice loss of almost 0, and inverted ones almost 1.
    """
    masks = torch.tensor(data=[[[0, 0, 1, 1], [0, 0, 1, 1]]])
    correct_logits = torch.nn.functional.one_hot(masks, num_classes=2).permute(0, 3, 1, 2).float() * 50.0

    assert compute_dice_loss(class_logits=correct_logits, masks=masks) < 1e-6
    assert compute_dice_loss(class_logits=correct_logits.flip(dims=(1,)), masks=masks) > 1.0 - 1e-6


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
