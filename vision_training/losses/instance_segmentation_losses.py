"""
Loss of the mask heads of the instance segmentation task: the binary cross entropy of the predicted mask of every
positive proposal against its mask target, on the channel of its ground truth class only, as in Mask R-CNN. Added by
the InstanceSegmenter to the four losses of the two-stage detector.
"""

import torch


def compute_mask_loss(mask_logits: torch.Tensor, labels: torch.Tensor, mask_targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the mask loss of the positive proposals: the binary cross entropy with logits of the mask channel of the
    ground truth class of every proposal, averaged over its pixels and over the proposals, so that the classes do not
    compete for a pixel (the classification is left to the detection head). It is computed in float32, stable under
    mixed precision, and is 0 without positive proposal while staying in the graph.

    Args:
        mask_logits (torch.Tensor): The [R, C, M, M] mask logits of the positive proposals.
        labels (torch.Tensor): The [R] ground truth class of every proposal.
        mask_targets (torch.Tensor): The [R, M, M] binary targets of every proposal.

    Returns:
        torch.Tensor: The scalar loss.
    """
    # Without positive proposal, a zero that keeps the mask heads in the graph
    if mask_logits.shape[0] == 0:
        return mask_logits.float().sum() * 0.0

    # Mask of the ground truth class of every proposal
    proposal_indices = torch.arange(mask_logits.shape[0], device=mask_logits.device)
    class_logits = mask_logits[proposal_indices, labels].float()
    return torch.nn.functional.binary_cross_entropy_with_logits(input=class_logits, target=mask_targets.float())
