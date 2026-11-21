"""
Losses of the mask heads of the instance segmentation task: the binary cross entropy of the predicted mask of every
positive proposal against its mask target, on the channel of its ground truth class only, as in Mask R-CNN, and with
'MaskHead.mask_scoring' the squared error of the predicted mask IoU, as in Mask Scoring R-CNN. Added by the
InstanceSegmenter to the four losses of the two-stage detector.
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


def compute_mask_iou_loss(predicted_mask_iou: torch.Tensor, mask_iou_targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the loss of the mask IoU head of Mask Scoring R-CNN ('MaskHead.mask_scoring'): the mean squared error
    between the predicted mask IoU of the ground truth class of every positive proposal and the IoU of its predicted
    mask with its object (build_mask_iou_targets), every positive counted as in the paper. It is computed in float32
    and is 0 without positive proposal while staying in the graph.

    Args:
        predicted_mask_iou (torch.Tensor): The [R] predicted mask IoU of the ground truth class of every proposal.
        mask_iou_targets (torch.Tensor): The [R] detached IoU targets.

    Returns:
        torch.Tensor: The scalar loss.
    """
    # Without positive proposal, a zero that keeps the mask IoU heads in the graph
    if predicted_mask_iou.shape[0] == 0:
        return predicted_mask_iou.float().sum() * 0.0
    return torch.nn.functional.mse_loss(input=predicted_mask_iou.float(), target=mask_iou_targets.float())
