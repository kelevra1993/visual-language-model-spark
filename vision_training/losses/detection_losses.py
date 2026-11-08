"""
Losses of the two-stage detector: the classification and localisation losses of the region proposal network and of
the detection head, the localisation losses being the smooth L1 on the regression offsets or the Complete-IoU loss on
the boxes they decode to ('box_loss' of 'RegionProposal' and 'Detector'). Called by the forward pass of
models/detection/two_stage_detector.py during training.
"""

import math
import torch

from typing import Any, Dict, List, Optional, Tuple

from vision_training.models.detection.box_operations import (apply_regression_predictions,
                                                             get_normalized_centre_distance)
from vision_training.models.detection.target_assignment import (batch_assign_targets_to_anchors,
                                                                sample_positive_and_negative_training_targets,
                                                                turn_boxes_to_transformation_targets)

# Transition point of the smooth L1 localisation losses of the region proposal network and of the detection heads,
# between their quadratic and linear parts, that of the Faster R-CNN implementations of Detectron and torchvision
LOCALISATION_LOSS_BETA = 1 / 9

# Localisation losses of 'RegionProposal.box_loss' and 'Detector.box_loss': the smooth L1 on the offsets, or 1 - CIoU
# between the decoded boxes and their targets (documentation/detection_box_loss_design.md)
BOX_LOSSES = ["smooth_l1", "complete_iou"]

# Small value added to the union and to the heights of the aspect ratios of the CIoU loss, so that empty or flat boxes
# give finite values, as in torchvision and mmdetection, the squared diagonal of the enclosing box getting the
# ENCLOSING_DIAGONAL_EPSILON of the same value of models/detection/box_operations.py
COMPLETE_IOU_EPSILON = 1e-7

# IoU from which the CIoU loss weighs the consistency of the aspect ratios, below which it is the DIoU loss
# (Zheng et al., "Enhancing Geometric Factors in Model Learning and Inference", Eq. 10)
ASPECT_RATIO_IOU_THRESHOLD = 0.5


def compute_complete_iou_loss(predicted_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes the Complete-IoU (CIoU) loss of every predicted box against its target box, the localisation loss of the
    region proposal network or of the detection heads when their 'box_loss' is "complete_iou" (Zheng et al., AAAI
    2020, Eq. 10, and their journal version, Eq. 9 and 10): 1 - IoU, plus the squared distance between the centres
    over the squared diagonal of the smallest box enclosing both, plus alpha times the consistency of the aspect ratios
    v = 4 / pi^2 (arctan(w_gt / h_gt) - arctan(w / h))^2, with alpha = v / ((1 - IoU) + v) from an IoU of 0.5 and 0
    below it, where the loss is the DIoU loss. Alpha is a constant of the gradient, as in the official code,
    torchvision and mmdetection. The three terms do not change with the scale of the boxes, given in image pixels.

    Args:
        predicted_boxes (torch.Tensor): The predicted boxes [x_min, y_min, x_max, y_max], of shape [..., 4].
        target_boxes (torch.Tensor): Their target boxes, of the same shape.

    Returns:
        torch.Tensor: The loss of every box, of shape [...], 0 for a box equal to its target.
    """
    # Widths and heights of the predicted and target boxes
    predicted_widths = predicted_boxes[..., 2] - predicted_boxes[..., 0]
    predicted_heights = predicted_boxes[..., 3] - predicted_boxes[..., 1]
    target_widths = target_boxes[..., 2] - target_boxes[..., 0]
    target_heights = target_boxes[..., 3] - target_boxes[..., 1]

    # Overlap term: the intersection over the union, the union increased by the epsilon for empty boxes
    intersection_sizes = (torch.minimum(input=predicted_boxes[..., 2:], other=target_boxes[..., 2:]) -
                          torch.maximum(input=predicted_boxes[..., :2], other=target_boxes[..., :2])).clamp(min=0)
    intersection = intersection_sizes[..., 0] * intersection_sizes[..., 1]
    union = predicted_widths * predicted_heights + target_widths * target_heights - intersection + COMPLETE_IOU_EPSILON
    intersection_over_union = intersection / union

    # Distance term: the squared distance between the centres over the squared diagonal of the smallest enclosing box,
    # shared with the DIoU-NMS of the final detections
    normalized_centre_distance = get_normalized_centre_distance(boxes_1=predicted_boxes, boxes_2=target_boxes)

    # Aspect ratio term: the consistency of the width over height ratios, the heights increased by the epsilon
    aspect_ratio_consistency = (4 / math.pi ** 2) * (
        torch.atan(input=target_widths / (target_heights + COMPLETE_IOU_EPSILON)) -
        torch.atan(input=predicted_widths / (predicted_heights + COMPLETE_IOU_EPSILON))).square()

    # Trade-off alpha, a constant of the gradient, weighing the aspect ratios from an IoU of 0.5 only
    with torch.no_grad():
        matched_boxes = (intersection_over_union >= ASPECT_RATIO_IOU_THRESHOLD).to(dtype=intersection_over_union.dtype)
        trade_off = matched_boxes * aspect_ratio_consistency / (
            (1 - intersection_over_union) + aspect_ratio_consistency + COMPLETE_IOU_EPSILON)

    return 1 - intersection_over_union + normalized_centre_distance + trade_off * aspect_ratio_consistency


def compute_localisation_loss(box_loss: str, regression_predictions: torch.Tensor, regression_targets: torch.Tensor,
                              reference_boxes: torch.Tensor) -> torch.Tensor:
    """
    Sums the localisation loss of the sampled positive anchors of the region proposal network or of the sampled
    positive proposals of the detection heads, which their callers divide by the number of positives: the smooth L1
    between the predicted and target offsets, or 1 - CIoU between the boxes both decode to from their anchors or
    proposals, the target offsets decoding back to the matched ground truth boxes.

    Args:
        box_loss (str): One of BOX_LOSSES, the 'box_loss' of 'RegionProposal' or 'Detector'.
        regression_predictions (torch.Tensor): The predicted offsets [dx, dy, dw, dh] of the positives, of shape [P, 4].
        regression_targets (torch.Tensor): Their target offsets, of shape [P, 4].
        reference_boxes (torch.Tensor): Their anchors or proposals [x_min, y_min, x_max, y_max], of shape [P, 4].

    Returns:
        torch.Tensor: The summed loss of the positives, a scalar (0 without positive).

    Raises:
        ValueError: If the box loss is not one of BOX_LOSSES.
    """
    if box_loss == "smooth_l1":
        return torch.nn.functional.smooth_l1_loss(input=regression_predictions, target=regression_targets,
                                                  beta=LOCALISATION_LOSS_BETA, reduction="sum")
    if box_loss == "complete_iou":
        # Decode the predicted and target offsets in the precision of the anchors or proposals, float32 whatever the
        # precision of the head outputs under the bfloat16 mixed precision, then compare the boxes
        predicted_boxes = apply_regression_predictions(
            regression_predictions=regression_predictions.to(dtype=reference_boxes.dtype).unsqueeze(dim=-2),
            boxes=reference_boxes).squeeze(dim=-2)
        target_boxes = apply_regression_predictions(
            regression_predictions=regression_targets.to(dtype=reference_boxes.dtype).unsqueeze(dim=-2),
            boxes=reference_boxes).squeeze(dim=-2)
        return compute_complete_iou_loss(predicted_boxes=predicted_boxes, target_boxes=target_boxes).sum()
    raise ValueError(f"Unknown box loss {box_loss!r}, expected one of {BOX_LOSSES}.")


def compute_region_proposal_losses(ground_truth_bounding_boxes: List[torch.Tensor],
                                   aggregated_proposals_dictionary: Dict[str, torch.Tensor],
                                   region_proposal_configuration: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the Region Proposal Network classification and localisation losses of the two-stage detector.

    This isolated logic extracts positive and negative target anchors, compares the predicted
    logits and regressed box offsets against the true targets, and generates the two loss
    components used to train the region proposer.

    Args:
        ground_truth_bounding_boxes (List[torch.Tensor]): The batched ground truth boxes.
        aggregated_proposals_dictionary (Dict[str, torch.Tensor]): The aggregated anchor and prediction tensor.
        region_proposal_configuration (Dict[str, Any]): The 'RegionProposal' configuration section, giving the
            anchor IoU thresholds, the sampling counts and the box loss.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The scalar classification loss and localisation loss.
    """

    # First assign targets based on ground truth bounding boxes
    # Here we consider absolutely all anchors targets based on all ground truth boxes
    region_proposal_anchor_targets, region_proposal_anchor_labels = batch_assign_targets_to_anchors(
        batched_ground_truth_boxes=ground_truth_bounding_boxes,
        batched_anchors=aggregated_proposals_dictionary["anchors"],
        background_iou_threshold=region_proposal_configuration['background_iou_threshold'],
        foreground_iou_threshold=region_proposal_configuration['foreground_iou_threshold'])

    # Compute region proposal regression targets
    region_proposal_regression_targets = turn_boxes_to_transformation_targets(
        ground_truth_boxes=region_proposal_anchor_targets, predicted_boxes=aggregated_proposals_dictionary["anchors"])

    # Get training samples for region proposal network
    sampled_positive_mask, sampled_negative_mask = sample_positive_and_negative_training_targets(
        labels=region_proposal_anchor_labels,
        desired_positives=region_proposal_configuration['number_training_positives'],
        desired_total=region_proposal_configuration['total_training_samples'])

    # Compute the classification loss: Simple binary cross entropy loss
    # We use the variant 'with_logits' since the classification scores are raw, un-sigmoid network outputs.
    # Computed over both the sampled positive (foreground) and sampled negative (background) anchors.
    sampled_mask = sampled_positive_mask | sampled_negative_mask

    region_proposal_classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        input=aggregated_proposals_dictionary["classification_scores"][sampled_mask],
        target=region_proposal_anchor_labels[sampled_mask], reduction="mean")

    # Compute the localization loss of 'RegionProposal.box_loss' (smooth L1 on the offsets, or 1 - CIoU between the
    # boxes decoded from the anchors and their ground truth boxes) on the positively sampled anchors only, background
    # anchors having no target. It is summed and divided by the number of positive anchors, so that it is averaged per
    # box rather than per coordinate, the count clamped to 1 so that a batch without any positive anchor (e.g. images
    # without ground truth boxes) yields a zero loss instead of a NaN from a 0/0 division.
    region_proposal_localisation_loss = compute_localisation_loss(
        box_loss=region_proposal_configuration["box_loss"],
        regression_predictions=aggregated_proposals_dictionary["bounding_box_regressions"][sampled_positive_mask],
        regression_targets=region_proposal_regression_targets[sampled_positive_mask],
        reference_boxes=aggregated_proposals_dictionary["anchors"][sampled_positive_mask]
    ) / torch.sum(input=sampled_positive_mask).clamp(min=1)

    return region_proposal_classification_loss, region_proposal_localisation_loss


def compute_detection_losses(aggregated_detections_dictionary: Dict[str, Optional[torch.Tensor]],
                             box_loss: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the final Detection Head classification and localisation losses of the two-stage detector.

    Args:
        aggregated_detections_dictionary (Dict[str, Optional[torch.Tensor]]): The aggregated predictions and targets
            from all detection heads.
            - "classification_scores" (torch.Tensor): Shape [total_proposals, number_classes]
            - "box_regressions" (torch.Tensor): Shape [total_proposals, number_classes, 4]
            - "sliced_proposals" (torch.Tensor): Shape [total_proposals, 5]
            - "sliced_labels" (torch.Tensor): Shape [total_proposals]
            - "sliced_regression_targets" (torch.Tensor): Shape [total_proposals, 4]
        box_loss (str): The 'Detector.box_loss', one of BOX_LOSSES.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The classification loss and the bounding box regression loss (scalar
            tensors).
    """
    # Compute the Classification Loss (Cross Entropy)
    # This evaluates all sampled proposals (both foreground and background).
    # The background proposals (label 0) will naturally train the network to suppress false positives.
    detection_classification_loss = torch.nn.functional.cross_entropy(
        input=aggregated_detections_dictionary["classification_scores"],
        target=aggregated_detections_dictionary["sliced_labels"])

    # Filter out background proposals: Bounding box regressions are ONLY trained on positive, foreground samples.
    sampled_positive_indices = torch.where(aggregated_detections_dictionary["sliced_labels"] > 0)[0]
    foreground_classes = aggregated_detections_dictionary["sliced_labels"][sampled_positive_indices].long()

    # The detector outputs regression predictions for EVERY class [K, number_classes, 4].
    # We use advanced indexing `[sampled_positive_indices, foreground_classes]` to pluck out ONLY the 4
    # coordinate regressions that correspond to the specific ground truth class of each proposal.
    positive_offsets = aggregated_detections_dictionary["box_regressions"][sampled_positive_indices, foreground_classes]

    # Compute the Localisation Loss of 'Detector.box_loss', the proposals losing their batch index column
    detection_localisation_loss = compute_localisation_loss(
        box_loss=box_loss, regression_predictions=positive_offsets,
        regression_targets=aggregated_detections_dictionary["sliced_regression_targets"][sampled_positive_indices],
        reference_boxes=aggregated_detections_dictionary["sliced_proposals"][sampled_positive_indices, 1:]
    ) / max(1, sampled_positive_indices.numel())

    return detection_classification_loss, detection_localisation_loss
