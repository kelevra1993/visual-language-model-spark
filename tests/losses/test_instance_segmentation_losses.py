"""
Tests of vision_training/losses/instance_segmentation_losses.py: the mask loss reads the channel of the ground truth
class of every proposal only, and the mask IoU loss of the scored masks is the squared error of the predicted IoU.
"""

import torch

from vision_training.losses.instance_segmentation_losses import compute_mask_iou_loss, compute_mask_loss


def test_mask_loss_reads_the_channel_of_the_ground_truth_class() -> None:
    """
    Verifies that the loss equals the binary cross entropy of the channel of the class of every proposal, whatever the
    other channels hold, and that a confident correct mask gives a lower loss than a wrong one.
    """
    targets = torch.zeros(size=(2, 4, 4))
    targets[:, :2] = 1.0
    confident_logits = (targets * 2.0 - 1.0) * 10.0
    mask_logits = torch.randn(size=(2, 3, 4, 4))
    mask_logits[0, 1], mask_logits[1, 2] = confident_logits[0], confident_logits[1]
    labels = torch.tensor(data=[1, 2])

    loss = compute_mask_loss(mask_logits=mask_logits, labels=labels, mask_targets=targets)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(input=confident_logits, target=targets)
    assert torch.allclose(input=loss, other=expected) and float(loss) < 1e-3

    # The same logits read on the wrong channels give a larger loss
    assert float(compute_mask_loss(mask_logits=mask_logits, labels=torch.tensor(data=[2, 1]),
                                   mask_targets=targets)) > float(loss)


def test_mask_iou_loss_is_the_squared_error_of_the_predicted_iou() -> None:
    """
    Verifies the loss of the mask IoU heads (Mask Scoring R-CNN): the mean squared error between the predicted mask IoU
    and its target over every positive proposal, computed in float32, and a zero that keeps the heads in the graph
    without positive proposal.
    """
    predicted_mask_iou = torch.tensor(data=[0.9, 0.2, 0.5], dtype=torch.bfloat16, requires_grad=True)
    mask_iou_targets = torch.tensor(data=[1.0, 0.5, 0.5])

    # ((0.9 - 1)^2 + (0.2 - 0.5)^2 + 0) / 3, the bfloat16 predictions read in float32
    loss = compute_mask_iou_loss(predicted_mask_iou=predicted_mask_iou, mask_iou_targets=mask_iou_targets)
    expected = ((predicted_mask_iou.detach().float() - mask_iou_targets) ** 2).mean()
    assert loss.dtype == torch.float32 and torch.allclose(input=loss, other=expected)
    assert abs(float(loss) - (0.01 + 0.09) / 3) < 2e-3
    loss.backward()
    assert predicted_mask_iou.grad is not None

    # Without positive proposal, a zero still holding the gradient path
    empty_predictions = torch.zeros(size=(0,), requires_grad=True)
    empty_loss = compute_mask_iou_loss(predicted_mask_iou=empty_predictions, mask_iou_targets=torch.zeros(size=(0,)))
    assert float(empty_loss) == 0.0 and empty_loss.requires_grad
