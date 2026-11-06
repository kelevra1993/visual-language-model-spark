"""
Losses of the two-stage detector: the classification and localisation losses of the region proposal network and of
the detection head. Called by the forward pass of models/detection/two_stage_detector.py during training.
"""

import torch

from typing import Any, Dict, List, Optional, Tuple

from vision_training.models.detection.target_assignment import (batch_assign_targets_to_anchors,
                                                                sample_positive_and_negative_training_targets,
                                                                turn_boxes_to_transformation_targets)

# Transition point of the smooth L1 localisation losses of the region proposal network and of the detection heads,
# between their quadratic and linear parts, that of the Faster R-CNN implementations of Detectron and torchvision
LOCALISATION_LOSS_BETA = 1 / 9


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
            anchor IoU thresholds, the sampling counts and the share of easy negatives.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The scalar classification loss and localisation loss.
    """

    # First assign targets based on ground truth bounding boxes
    # Here we consider absolutely all anchors targets based on all ground truth boxes
    region_proposal_anchor_targets, region_proposal_anchor_labels, easy_negative_mask = batch_assign_targets_to_anchors(
        batched_ground_truth_boxes=ground_truth_bounding_boxes,
        batched_anchors=aggregated_proposals_dictionary["anchors"],
        background_iou_threshold=region_proposal_configuration['background_iou_threshold'],
        foreground_iou_threshold=region_proposal_configuration['foreground_iou_threshold'])

    # Compute region proposal regression targets
    region_proposal_regression_targets = turn_boxes_to_transformation_targets(
        ground_truth_boxes=region_proposal_anchor_targets, predicted_boxes=aggregated_proposals_dictionary["anchors"])

    # Get training samples for region proposal network, the share 'easy_negative_fraction' of the negatives drawn
    # among the anchors below the background minimum
    sampled_positive_mask, sampled_negative_mask = sample_positive_and_negative_training_targets(
        labels=region_proposal_anchor_labels,
        desired_positives=region_proposal_configuration['number_training_positives'],
        desired_total=region_proposal_configuration['total_training_samples'], easy_negative_mask=easy_negative_mask,
        easy_negative_fraction=region_proposal_configuration['easy_negative_fraction'])

    # Compute the classification loss: Simple binary cross entropy loss
    # We use the variant 'with_logits' since the classification scores are raw, un-sigmoid network outputs.
    # Computed over both the sampled positive (foreground) and sampled negative (background) anchors, the target being
    # 1 for a positive and 0 for a negative, the easy negatives carrying the ignored label -1
    sampled_mask = sampled_positive_mask | sampled_negative_mask

    region_proposal_classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        input=aggregated_proposals_dictionary["classification_scores"][sampled_mask],
        target=sampled_positive_mask[sampled_mask].to(dtype=torch.float32), reduction="mean")

    # Compute the localization loss: Smooth L1 Loss (Huber Loss)
    # We only compute regression loss on the positively sampled anchors, as background anchors have no target.
    # We explicitly use reduction="sum" and divide by the total number of positive anchors instead of using
    # reduction="mean". This ensures the loss is averaged per bounding box rather than per individual
    # coordinate, aligning perfectly with standard Region Proposal Network normalization practices.
    # The positive count is clamped to a minimum of 1 so that a batch without any positive anchor
    # (e.g. images without ground truth boxes) yields a zero loss instead of a NaN from a 0/0 division.
    region_proposal_localisation_loss = torch.nn.functional.smooth_l1_loss(
        input=aggregated_proposals_dictionary["bounding_box_regressions"][sampled_positive_mask],
        target=region_proposal_regression_targets[sampled_positive_mask],
        beta=LOCALISATION_LOSS_BETA, reduction="sum") / torch.sum(input=sampled_positive_mask).clamp(min=1)

    return region_proposal_classification_loss, region_proposal_localisation_loss


def compute_detection_losses(aggregated_detections_dictionary: Dict[str, Optional[torch.Tensor]]
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    # Compute the Localisation Loss (Smooth L1 / Huber Loss)
    # The detector outputs regression predictions for EVERY class [K, number_classes, 4].
    # We use advanced indexing `[sampled_positive_indices, foreground_classes]` to pluck out ONLY the 4
    # coordinate regressions that correspond to the specific ground truth class of each proposal.
    detection_localisation_loss = torch.nn.functional.smooth_l1_loss(
        input=aggregated_detections_dictionary["box_regressions"][sampled_positive_indices, foreground_classes],
        target=aggregated_detections_dictionary["sliced_regression_targets"][sampled_positive_indices],
        beta=LOCALISATION_LOSS_BETA, reduction="sum") / max(1, sampled_positive_indices.numel())

    return detection_classification_loss, detection_localisation_loss
