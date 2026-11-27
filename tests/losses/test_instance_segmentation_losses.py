"""
Tests of vision_training/losses/instance_segmentation_losses.py: the mask loss reads the channel of the ground truth
class of every proposal only.
"""

import torch

from vision_training.losses.instance_segmentation_losses import compute_mask_loss


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
