"""
Training target assignment of the object detector: matching of the ground truth boxes to the anchors (region
proposal stage) and to the proposals (detection stage) by intersection over union, regression targets, and the
sampling of positive and negative examples. Used by the TwoStageDetector and vision_training/losses/detection_losses.py.
"""

import torch

from typing import Dict, List, Tuple

from vision_training.models.detection.box_operations import (get_box_dimensions_and_centers,
                                                             get_intersection_over_union)
from vision_training.utilities.os_utilities import print_color


def assign_targets_to_anchors(ground_truth_boxes: torch.Tensor,
                              anchors: torch.Tensor,
                              background_iou_threshold: Dict[str, float],
                              foreground_iou_threshold: Dict[str, float],
                              strict_fallback_assignment: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Assigns ground truth boxes to anchors to prepare the binary classification and regression targets
    for the Region Proposal Network.

    This function sits at the heart of the RPN training pipeline. It computes the IoU between every anchor 
    and every ground truth box, categorizing each anchor as foreground (positive object), background (negative), 
    or grey zone (ignored during training). It guarantees that every ground truth box has at least one positive 
    anchor assigned to it to prevent objects from being missed during training.

    Args:
        ground_truth_boxes (torch.Tensor): A tensor of ground truth boxes in [x_min, y_min, x_max, y_max] format of shape [Num_GT, 4].
        anchors (torch.Tensor): A tensor of the base anchors generated for the current feature map of shape [Num_Anchors, 4].
        background_iou_threshold (Dict[str, float]): Dictionary defining 'min' and 'max' IoU thresholds for background.
        foreground_iou_threshold (Dict[str, float]): Dictionary defining 'min' and 'max' IoU thresholds for foreground.
        strict_fallback_assignment (bool): If True, assigns resurrected anchors strictly to the ground truth box
        that triggered their resurrection. If False, follows default PyTorch RPN behavior of assigning them to
        the ground truth box they overlap with most. Defaults to False.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - matched_ground_truth_boxes (torch.Tensor): The assigned ground truth boxes for each anchor of shape [Num_Anchors, 4].
            - labels (torch.Tensor): The binary labels for each anchor of shape [Num_Anchors]
                                     (1.0 = foreground, 0.0 = background, -1.0 = ignored).
    """

    # Intercept images completely lacking valid ground truth boxes (due to filtering or native empty sets)
    # PyTorch's max() reduction will crash if it attempts to operate on a zero-length dimension.
    # We safely bypass the IoU matrix matching and strictly assign all generated anchors to the background.
    if ground_truth_boxes.size(0) == 0:
        matched_ground_truth_boxes = torch.zeros_like(anchors)
        # Background gets label 0.0, ignore gets -1.0. We can safely set all to background or ignore.
        # Setting to background (0.0) means RPN is penalized for ANY proposal.
        labels = torch.zeros(anchors.size(0), dtype=torch.float32, device=anchors.device)
        return matched_ground_truth_boxes, labels

    # Shape: [number_of_ground_truths, number_of_anchors]
    intersection_over_union_matrix = get_intersection_over_union(boxes_1=ground_truth_boxes, boxes_2=anchors)

    # For every anchor, find the ground truth box that has the highest IoU with it
    best_match_ground_truth_iou, best_match_ground_truth_index = intersection_over_union_matrix.max(dim=0)

    # Store a pristine copy of the best matches before we corrupt it with negative labels
    # This guarantees we can later recover the best match to fulfill the "at least one anchor per GT box" rule
    best_match_ground_truth_index_before_thresholding = best_match_ground_truth_index.clone()

    # Create readable condition variables based on the IoU thresholds
    below_background_min = best_match_ground_truth_iou < background_iou_threshold['min']
    above_or_equal_background_min = best_match_ground_truth_iou >= background_iou_threshold['min']
    below_background_max = best_match_ground_truth_iou < background_iou_threshold['max']

    above_or_equal_background_max = best_match_ground_truth_iou >= background_iou_threshold['max']

    below_foreground_min = best_match_ground_truth_iou < foreground_iou_threshold['min']

    # Anchors that fall strictly within the background IoU limits are true negatives
    background_indices = above_or_equal_background_min & below_background_max

    # Anchors that fall between the maximum background limit and the minimum foreground limit, 
    # or below the absolute minimum background limit, are ignored (grey zone)
    grey_zone_indices = (above_or_equal_background_max & below_foreground_min) | below_background_min

    # Assign labels to our different anchors
    # Each anchor for which it's highest scoring iou ground truth box
    # is either considered background or in the grey zone are labelled as such :
    # - background > -1
    # - grey zone > -2
    best_match_ground_truth_index[background_indices] = -1
    best_match_ground_truth_index[grey_zone_indices] = -2

    # Making sure that every ground truth will have at least one positive anchor bounding box
    # Get the best iou value for each ground truth box across all available anchors
    best_match_anchor_iou, _ = intersection_over_union_matrix.max(dim=1)

    # This gives us all the anchors that tie for the highest iou for each ground truth box.
    # We use torch.isclose to safely handle float32 truncation artifacts where mathematically 
    # identical bounding box areas might differ by ~1e-6.
    # We also enforce a strict > 0.0 threshold to prevent the "Zero-IoU Explosion", 
    # preventing a ground truth box with no overlap from resurrecting thousands of background anchors.
    best_anchor_indices_for_each_ground_truth_box = torch.where(
        torch.isclose(input=intersection_over_union_matrix,
                      other=best_match_anchor_iou.unsqueeze(dim=-1),
                      atol=1e-6) & (best_match_anchor_iou.unsqueeze(dim=-1) > 0.0))

    anchor_indices_to_retrieve = best_anchor_indices_for_each_ground_truth_box[1]

    if strict_fallback_assignment:
        ground_truths_for_retrieved_anchors = best_anchor_indices_for_each_ground_truth_box[0]
        best_match_ground_truth_index[anchor_indices_to_retrieve] = ground_truths_for_retrieved_anchors
    else:
        # We recover the positive labels for the anchors that had the highest IoU for each ground truth box.
        # IMPORTANT QUIRK: Notice that we are assigning the anchor to its absolute best match overall
        # (using best_match_ground_truth_index_before_thresholding) rather than strictly forcing it to
        # match the specific ground truth box that selected it. For example, if an anchor is forced
        # positive to cover Ground Truth A, but it actually overlaps more with Ground Truth B, it will
        # be assigned to predict Ground Truth B. This perfectly mirrors the official PyTorch RPN logic!
        best_match_ground_truth_index[anchor_indices_to_retrieve] = best_match_ground_truth_index_before_thresholding[
            anchor_indices_to_retrieve]

    # Get coordinates of best matching ground truth target boxes (so each anchor will have at least one target)
    # But we will not necessarily train on all the assigned targets because the -1 and -2 will be bogus targets.
    # Be careful with coordinates -1 and -2 to clamp to 0 to always have the same coordinates for background and grey zone
    target_ground_truth_boxes = ground_truth_boxes[best_match_ground_truth_index.clamp(0)]

    # Now set all labels for training so there is no ambiguity
    # For classification loss we consider labels above 0, where 0 is background and 1 is foreground
    # For bounding box regression we only consider labels equal to 1 which are foreground
    # Shape (anchor_boxes)
    labels = (best_match_ground_truth_index >= 0).to(dtype=torch.float32)
    labels[best_match_ground_truth_index == -1] = 0.0  # set -1 to 0 (background)
    labels[best_match_ground_truth_index == -2] = -1.0  # set -2 to -1 (ignored)

    return target_ground_truth_boxes, labels


def batch_assign_targets_to_anchors(batched_ground_truth_boxes: List[torch.Tensor],
                                    batched_anchors: torch.Tensor,
                                    background_iou_threshold: Dict[str, float],
                                    foreground_iou_threshold: Dict[str, float],
                                    strict_fallback_assignment: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies target assignment logic across an entire batch of images.
    
    This function iterates over the batch dimension to match anchors to ground truth boxes 
    independently for each image, avoiding the complexity and memory overhead of padding 
    varying numbers of ground truth boxes. The results are then stacked into unified batch tensors.
    
    Args:
        batched_ground_truth_boxes (List[torch.Tensor]): A list of length batch_size, where each tensor 
                                                         contains the ground truth boxes for that image.
        batched_anchors (torch.Tensor): A tensor of shape [batch_size, number_of_anchors, 4].
        background_iou_threshold (Dict[str, float]): Dictionary defining 'min' and 'max' IoU thresholds for background.
        foreground_iou_threshold (Dict[str, float]): Dictionary defining 'min' and 'max' IoU thresholds for foreground.
        strict_fallback_assignment (bool): Propagates the resurrection override flag to the assignment logic.
        Defaults to False.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - batched_target_boxes (torch.Tensor): Shape [batch_size, number_of_anchors, 4]
            - batched_labels (torch.Tensor): Shape [batch_size, number_of_anchors]
    """
    aggregated_target_boxes = []
    aggregated_labels = []

    # Iterate over each image in the batch to independently process target assignments
    for batch_index in range(len(batched_ground_truth_boxes)):
        # Calculate assignments for the current image using explicit argument naming
        target_boxes, labels = assign_targets_to_anchors(
            ground_truth_boxes=batched_ground_truth_boxes[batch_index],
            anchors=batched_anchors[batch_index],
            background_iou_threshold=background_iou_threshold,
            foreground_iou_threshold=foreground_iou_threshold,
            strict_fallback_assignment=strict_fallback_assignment)

        # Store the processed targets and labels
        aggregated_target_boxes.append(target_boxes)
        aggregated_labels.append(labels)

    # Stack the lists along a new batch dimension (dim=0)
    batched_target_boxes = torch.stack(tensors=aggregated_target_boxes, dim=0)
    batched_labels = torch.stack(tensors=aggregated_labels, dim=0)

    return batched_target_boxes, batched_labels


def turn_boxes_to_transformation_targets(ground_truth_boxes: torch.Tensor,
                                         predicted_boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes ideal regression targets (dx, dy, dw, dh) required to transform predicted boxes into ground truth boxes.
    
    In object detection pipelines, bounding box regression heads do not directly predict absolute coordinates. 
    Instead, they predict parameterized offsets relative to a base box (like an anchor or a previous proposal). 
    This function calculates those continuous targets by computing the scale-invariant center shifts (dx, dy) 
    and the log-space dimension adjustments (dw, dh) so the network can safely regress them using Smooth L1 Loss.

    Args:
        ground_truth_boxes (torch.Tensor): A tensor of shape (..., N, 4)
        containing the target bounding boxes in [x_min, y_min, x_max, y_max] format.
        predicted_boxes (torch.Tensor): A tensor of shape (..., N, 4)
        containing the base/reference bounding boxes in [x_min, y_min, x_max, y_max] format.

    Returns:
        torch.Tensor: A tensor of shape (..., N, 4) containing the regression targets [dx, dy, dw, dh].
    """

    # Get ground truth dimensions (Widths, Heights, Center X, Center Y)
    # Shapes for all returned variables: (..., N)
    (ground_truth_boxes_widths, ground_truth_boxes_heights,
     ground_truth_boxes_center_x, ground_truth_boxes_center_y) = get_box_dimensions_and_centers(
        boxes=ground_truth_boxes)

    # Get predicted (base) boxes dimensions (Widths, Heights, Center X, Center Y)
    # Shapes for all returned variables: (..., N)
    (predicted_boxes_widths, predicted_boxes_heights,
     predicted_boxes_center_x, predicted_boxes_center_y) = get_box_dimensions_and_centers(
        boxes=predicted_boxes)

    # Compute scale-invariant offsets for the center coordinates (with epsilon to prevent division by zero)
    # Shape: (..., N)
    target_dx = (ground_truth_boxes_center_x - predicted_boxes_center_x) / (predicted_boxes_widths + 1e-6)
    target_dy = (ground_truth_boxes_center_y - predicted_boxes_center_y) / (predicted_boxes_heights + 1e-6)

    # Compute log-space scale adjustments for width and height (clamped to prevent log(0) -> -inf)
    # Shape: (..., N)
    target_dw = torch.log(input=(ground_truth_boxes_widths / (predicted_boxes_widths + 1e-6)).clamp(min=1e-6))
    target_dh = torch.log(input=(ground_truth_boxes_heights / (predicted_boxes_heights + 1e-6)).clamp(min=1e-6))

    # Stack the individual parameterized targets into the final format [dx, dy, dw, dh]
    # Shape: (..., N, 4)
    regression_targets = torch.stack(tensors=[target_dx, target_dy, target_dw, target_dh], dim=-1)

    return regression_targets


def sample_positive_and_negative_training_targets(labels: torch.Tensor, desired_positives: int, desired_total: int,
                                                  verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly samples positive and negative targets (anchors or proposals) to maintain a fixed ratio during training.

    This utility is universally applicable to both the Region Proposal Network (sampling raw anchors) and the 
    Detector Network (sampling NMS-filtered proposals). To prevent the classification loss from being completely 
    overwhelmed by the massive number of background targets, this function constructs a balanced mini-batch 
    for each image. It guarantees up to `desired_positives` foreground targets, and fills the remainder 
    of the `desired_total` quota with background targets.

    Args:
        labels (torch.Tensor): A batched tensor of shape (B, N) containing target labels 
                               (Values >= 1.0 = positive foreground, 0.0 = negative background).
        desired_positives (int): The maximum number of positive targets to sample per image.
        desired_total (int): The total combined number of targets (positive + negative) to sample per image.
        verbose (bool): If True, prints a console summary detailing the total positive, negative,
                        and combined targets sampled across the batch. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing two boolean masks of shape (B, N):
            - sampled_positive_mask (torch.Tensor): True for selected positive targets.
            - sampled_negative_mask (torch.Tensor): True for selected negative targets.
    """
    # Create batched boolean masks highlighting only the randomly sampled positive and negative anchors
    sampled_positive_mask = torch.zeros_like(input=labels, dtype=torch.bool)
    sampled_negative_mask = torch.zeros_like(input=labels, dtype=torch.bool)

    # Iterate over each image in the batch to independently sample targets
    for batch_index in range(labels.size(0)):
        image_labels = labels[batch_index]

        # Identify the raw indices for all available positive and negative anchors in this specific image
        positive_labels_indices = torch.where(condition=image_labels >= 1.0)[0]
        negative_labels_indices = torch.where(condition=image_labels == 0.0)[0]

        # Calculate the actual number of positives to sample, capped by availability
        possible_positives = min(positive_labels_indices.numel(), desired_positives)

        # Fill the remaining slots with negatives, capped by availability
        desired_negatives = desired_total - possible_positives
        possible_negatives = min(negative_labels_indices.numel(), desired_negatives)

        # CRITICAL FIX: If we ran out of negatives, we must over-sample positives to maintain 
        # a uniform tensor shape for torch.stack() downstream across the batch!
        if possible_negatives < desired_negatives:
            shortfall = desired_negatives - possible_negatives
            possible_positives = min(positive_labels_indices.numel(), possible_positives + shortfall)

        # Generate random permutations to shuffle and select the required number of indices
        random_positive_indices = torch.randperm(n=positive_labels_indices.numel(),
                                                 device=positive_labels_indices.device)[:possible_positives]
        random_negative_indices = torch.randperm(n=negative_labels_indices.numel(),
                                                 device=negative_labels_indices.device)[:possible_negatives]

        # Map the shuffled local indices back to the absolute anchor indices
        final_positive_indices = positive_labels_indices[random_positive_indices]
        final_negative_indices = negative_labels_indices[random_negative_indices]

        # Apply the selected indices to the batched masks for this specific image
        sampled_positive_mask[batch_index, final_positive_indices] = True
        sampled_negative_mask[batch_index, final_negative_indices] = True

    # Calculate and display the final breakdown of sampled anchors per image in the batch for debugging
    if verbose:
        print_color(color="yellow", output="Sampling Information", add_separators=True)
        total_positives = torch.sum(input=sampled_positive_mask, dim=1)
        total_negatives = torch.sum(input=sampled_negative_mask, dim=1)
        print_color(color="yellow", output=f"Positive Training Samples   : {total_positives.tolist()}", indent=1)
        print_color(color="yellow", output=f"Negative Training Samples   : {total_negatives.tolist()}", indent=1)
        print_color(color="yellow",
                    output=f"All Batch Training Samples  : {(total_negatives + total_positives).tolist()}", indent=1)

    return sampled_positive_mask, sampled_negative_mask


def assign_targets_to_proposals(ground_truth_boxes: torch.Tensor, proposals: torch.Tensor,
                                ground_truth_labels: torch.Tensor,
                                foreground_iou_threshold: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assigns ground truth bounding boxes and class labels to region proposals for Detection Head training.
    
    This function matches the predicted region proposals (from the Region Proposal Network) against 
    the actual ground truth boxes using Intersection over Union (IoU). Proposals with an IoU 
    greater than or equal to the foreground threshold are assigned the corresponding ground truth class.
    Proposals falling below the threshold are assigned the background class (class index 0). The index of the matched
    ground truth is returned too, so that the instance segmenter picks the ground truth mask of every positive proposal.

    Args:
        ground_truth_boxes (torch.Tensor): A tensor of ground truth boxes of shape [Num_GT, 4] in [x_min, y_min, x_max, y_max] format.
        proposals (torch.Tensor): A tensor of proposed bounding boxes of shape [Num_Proposals, 4].
        ground_truth_labels (torch.Tensor): A tensor of the true class indices for each ground truth box of shape [Num_GT].
        foreground_iou_threshold (float): The minimum IoU required for a proposal to be considered a positive match.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - target_ground_truth_boxes (torch.Tensor): The assigned ground truth coordinates for each proposal of shape [Num_Proposals, 4].
            - target_labels (torch.Tensor): The assigned class labels for each proposal (0 for background) of shape [Num_Proposals].
            - matched_ground_truth_indices (torch.Tensor): The int64 index of the matched ground truth of each proposal
              within the image, -1 for the background, of shape [Num_Proposals].
    """
    # Note to self to check afterwards : ground truth boxes and labels should have the same first dimension shape which are the number of ground truth boxes in the image

    # Intercept images completely lacking valid ground truth boxes (due to filtering or native empty sets)
    # PyTorch's max() reduction will crash if it attempts to operate on a zero-length dimension.
    # We safely bypass the IoU matrix matching and map all generated proposals strictly to the background class.
    if ground_truth_boxes.size(0) == 0:
        target_ground_truth_boxes = torch.zeros_like(proposals)
        target_labels = torch.zeros(proposals.size(0), dtype=torch.int64, device=proposals.device)
        matched_ground_truth_indices = torch.full(size=(proposals.size(0),), fill_value=-1, dtype=torch.int64,
                                                  device=proposals.device)
        return target_ground_truth_boxes, target_labels, matched_ground_truth_indices

    # Shape: [number_of_ground_truths, number_of_proposals]
    intersection_over_union_matrix = get_intersection_over_union(boxes_1=ground_truth_boxes, boxes_2=proposals)

    # For every proposal, find the ground truth box that has the highest IoU with it
    best_match_ground_truth_iou, best_match_ground_truth_index = intersection_over_union_matrix.max(dim=0)

    # Create readable condition variables based on the IoU thresholds
    below_foreground_min = best_match_ground_truth_iou < foreground_iou_threshold

    # Proposals that fall strictly within the foreground IoU limits are considered negatives
    # Will never be considered for the detection loss
    background_indices = below_foreground_min

    # Assign background labels for these proposals
    # - background > -1
    best_match_ground_truth_index[background_indices] = -1

    # Get coordinates of best matching ground truth target boxes (so each proposal will have at least one target)
    # But we will not necessarily train on all the assigned targets because the -1 are bogus targets.
    # Be careful Coordinates -1 are clamped to 0 to always have the same coordinates for background
    target_ground_truth_boxes = ground_truth_boxes[best_match_ground_truth_index.clamp(min=0)]

    # Now set all labels for training so there is no ambiguity
    # For classification loss we get the labels of the best iou ground truth match for each proposal
    # We then set to -1 all negatives
    # Shape (proposal_boxes)
    target_labels = ground_truth_labels[best_match_ground_truth_index.clamp(min=0)].to(dtype=torch.int64)
    target_labels[background_indices] = 0
    # For bounding box regression we only consider labels larger than 0 which are foreground

    # The background proposals keep the index -1 of their match
    return target_ground_truth_boxes, target_labels, best_match_ground_truth_index


def batch_assign_targets_to_proposals(batched_ground_truth_boxes: List[torch.Tensor],
                                      batched_proposals: List[torch.Tensor],
                                      batched_ground_truth_labels: List[torch.Tensor], foreground_iou_threshold: float
                                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies proposal target assignment logic across an entire batch of images.
    
    This function iterates over the batch dimension to match Region Proposal Network (RPN) proposals 
    to ground truth boxes independently for each image, avoiding the complexity of padding 
    varying numbers of ground truth boxes. The results are stacked into unified batch tensors.
    
    Args:
        batched_ground_truth_boxes (List[torch.Tensor]): A list of length batch_size, where each tensor 
                                                         contains the ground truth boxes for that image.
        batched_proposals (List[torch.Tensor]): A list of length batch_size (or a unified tensor), 
                                                where each tensor contains the proposed bounding boxes for that image.
        batched_ground_truth_labels (List[torch.Tensor]): A list of length batch_size, containing the true class 
                                                          indices for the ground truth boxes of each image.
        foreground_iou_threshold (float): The minimum IoU required for a proposal to be considered foreground.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - batched_target_boxes (torch.Tensor): The assigned ground truth coordinates.
             Shape [batch_size, number_of_proposals, 4]
            - batched_labels (torch.Tensor): The assigned class labels.
             Shape [batch_size, number_of_proposals]
            - batched_matched_ground_truth_indices (torch.Tensor): The index of the matched ground truth within its
             image, -1 for the background. Shape [batch_size, number_of_proposals]
    """
    aggregated_target_boxes = []
    aggregated_labels = []
    aggregated_matched_indices = []

    # Iterate over each image in the batch to independently process target assignments
    for batch_index in range(len(batched_ground_truth_boxes)):
        # Calculate assignments for the current image using explicit argument naming
        target_boxes, target_labels, matched_ground_truth_indices = assign_targets_to_proposals(
            ground_truth_boxes=batched_ground_truth_boxes[batch_index],
            proposals=batched_proposals[batch_index],
            ground_truth_labels=batched_ground_truth_labels[batch_index],
            foreground_iou_threshold=foreground_iou_threshold)

        # Store the processed targets and labels
        aggregated_target_boxes.append(target_boxes)
        aggregated_labels.append(target_labels)
        aggregated_matched_indices.append(matched_ground_truth_indices)

    # Stack the lists along a new batch dimension (dim=0)
    batched_target_boxes = torch.stack(tensors=aggregated_target_boxes, dim=0)
    batched_target_labels = torch.stack(tensors=aggregated_labels, dim=0)
    batched_matched_ground_truth_indices = torch.stack(tensors=aggregated_matched_indices, dim=0)

    return batched_target_boxes, batched_target_labels, batched_matched_ground_truth_indices
