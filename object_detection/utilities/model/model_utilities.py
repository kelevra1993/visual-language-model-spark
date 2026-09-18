from collections import Counter

import cv2
import numpy as np
import torch

from typing import Tuple, Dict, List
from torchvision.ops import nms

from utilities.os_utilities import print_blue, print_green, print_red, print_yellow
from utilities.tensor_utilities import print_tensor_shape, print_tensor_list


def get_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes the spatial area of a set of bounding boxes.
    
    This utility calculates the area coverage for a tensor of bounding boxes provided 
    in [x_min, y_min, x_max, y_max] format. It uses vectorized operations across the 
    last dimension to efficiently support batched and unbatched tensors.
    
    Args:
        boxes (torch.Tensor): A tensor of shape (..., 4) representing bounding boxes.
        
    Returns:
        torch.Tensor: A tensor of shape (...,) containing the computed areas.
    """
    # Extract widths and heights using tensor slicing
    widths = boxes[..., 2] - boxes[..., 0]
    heights = boxes[..., 3] - boxes[..., 1]

    # Calculate and return the area
    areas = widths * heights
    return areas


def get_intersection_over_union(boxes_1: torch.Tensor, boxes_2: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Intersection over Union (IoU) matrix between two sets of bounding boxes.
    
    This function computes the overlap between anchor boxes and ground truth boxes (or other anchors),
    which is fundamentally required for matching ground truth targets to anchors during Region Proposal
    Network training and for non-maximum suppression during inference.
    
    Args:
        boxes_1 (torch.Tensor): A tensor of shape (N, 4) for first set of bounding boxes [x_min, y_min, x_max, y_max].
        boxes_2 (torch.Tensor): A tensor of shape (M, 4) for second set of bounding boxes [x_min, y_min, x_max, y_max].
        
    Returns:
        torch.Tensor: A tensor of shape (N, M) containing the computed IoU values for every pair of boxes.
    """
    # compute area of box 1 and box2
    area_1 = (boxes_1[:, 2] - boxes_1[:, 0]) * (boxes_1[:, 3] - boxes_1[:, 1])
    area_2 = (boxes_2[:, 2] - boxes_2[:, 0]) * (boxes_2[:, 3] - boxes_2[:, 1])

    # Get the top left coordinates of intersection area using broadcasting to create an (N, M, 2) tensor
    top_left_coordinates = torch.max(input=boxes_1.unsqueeze(dim=1)[:, :, :2],
                                     other=boxes_2.unsqueeze(dim=0)[:, :, :2])

    # Get the bottom right coordinates of intersection area using broadcasting to create an (N, M, 2) tensor
    bottom_right_coordinates = torch.min(input=boxes_1.unsqueeze(dim=1)[:, :, 2:],
                                         other=boxes_2.unsqueeze(dim=0)[:, :, 2:])

    # Compute intersection area (clamped to 0, to avoid negative values)
    intersection_dimensions = (bottom_right_coordinates - top_left_coordinates).clamp(min=0)
    intersection_area = intersection_dimensions[:, :, 0] * intersection_dimensions[:, :, 1]

    # Compute union area by summing individual areas and subtracting the overlapping intersection area
    union_area = area_1.unsqueeze(dim=1) + area_2.unsqueeze(dim=0) - intersection_area

    # Compute the final metric and return intersection_over_union variable, avoiding division by zero
    # Tensor of shape (N, M)
    intersection_over_union = intersection_area / (union_area + 1e-6)

    return intersection_over_union


def add_bounding_boxes(bounding_boxes: np.ndarray, image: np.ndarray, input_image_size: int,
                       color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draws one or more bounding boxes onto the provided image canvas.
    
    This function processes bounding box coordinates, clamps them to the image boundaries
    to prevent out-of-bounds drawing errors, and renders the rectangles using OpenCV. 
    It supports both single bounding boxes [4] and batches of bounding boxes [N, 4].
    
    Args:
        bounding_boxes (np.ndarray): The [x_min, y_min, x_max, y_max] coordinates. Shape (4,) or (N, 4).
        image (np.ndarray): The image canvas on which to draw the bounding boxes.
        input_image_size (int): The spatial size (height and width) of the image to clamp coordinates.
        color (Tuple[int, int, int]): The BGR color tuple for the bounding boxes. Defaults to green (0, 255, 0).
        
    Returns:
        np.ndarray: The updated image canvas containing the newly drawn bounding boxes.
    """
    # Standardize shape to [N, 4] for consistent iteration
    if bounding_boxes.ndim == 1:
        bounding_boxes = bounding_boxes.reshape((1, 4))

    for box in bounding_boxes:
        x_min, y_min, x_max, y_max = box

        # Ensure coordinates are within image boundaries for clean visualization
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(input_image_size, int(x_max))
        y_max = min(input_image_size, int(y_max))

        # Draw the bounding box on the canvas using the provided color
        cv2.rectangle(img=image, pt1=(x_min, y_min), pt2=(x_max, y_max), color=color, thickness=2)

    return image


def visualise_anchors(input_image_size: int, anchors: torch.Tensor, delayed: bool = False, delay: int = 0,
                      random_ratio: float = 1.0) -> None:
    """
    Visualizes the generated bounding box anchors on a black canvas using OpenCV.
    
    This utility function is critical for debugging the spatial configuration of anchors,
    ensuring they are placed and scaled correctly relative to the input image dimensions.
    It provides an interactive visual feedback loop during architecture development.
    
    Args:
        input_image_size (int): The height and width of the square input image.
        anchors (torch.Tensor): A tensor containing the [x_min, y_min, x_max, y_max] bounding boxes.
        delayed (bool): If True, renders and pauses at each anchor sequentially for step-by-step inspection.
        delay (int): The duration in milliseconds to pause during step-by-step inspection (0 means wait for key).
        random_ratio (float): The fraction of anchors to randomly sample and visualize (default is 1.0 for all).
    """
    # Create a completely black background image of the specified size with 3 channels (RGB)
    canvas = np.zeros(shape=(input_image_size, input_image_size, 3), dtype=np.uint8)

    # Convert the PyTorch tensor to a NumPy array for OpenCV compatibility
    anchors_array = anchors.cpu().numpy().astype(dtype=np.int32)

    # Randomly sample a fraction of the anchors for visualization if requested
    if random_ratio < 1.0:
        total_anchors = len(anchors_array)
        number_to_sample = int(total_anchors * random_ratio)

        # Select random indices without replacement, then sort them to preserve the original generation sequence
        sampled_indices = np.random.choice(a=total_anchors, size=number_to_sample, replace=False)
        sampled_indices.sort()

        anchors_array = anchors_array[sampled_indices]

    # Determine the window name for the OpenCV display
    window_name = "Anchors Visualization"

    if delayed:
        if not delay:
            print_blue(output="Showing Anchors Iteratively. Press any key to continue...", add_separators=True)

        for anchor_index, anchor in enumerate(anchors_array):
            canvas = add_bounding_boxes(bounding_boxes=anchor, image=canvas, input_image_size=input_image_size)

            # Display the updated canvas containing the newly drawn anchor
            cv2.imshow(winname=window_name, mat=canvas)

            # Pause execution for the specified delay or until the user presses a key
            cv2.waitKey(delay=delay)
    else:
        # Draw all anchors collectively on the canvas without a loop
        canvas = add_bounding_boxes(bounding_boxes=anchors_array, image=canvas, input_image_size=input_image_size)

        # Display the final canvas with all anchors drawn simultaneously
        cv2.imshow(winname=window_name, mat=canvas)
        print_blue(output="Showing all anchors collectively. Press any key to close the window...", add_separators=True)
        cv2.waitKey(delay=0)

    # Clean up the OpenCV windows after the visualization loop is complete
    cv2.destroyAllWindows()


def get_box_dimensions_and_centers(boxes: torch.Tensor) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts the widths, heights, and center coordinates from a tensor of bounding boxes.
    
    This function processes bounding boxes in [x_min, y_min, x_max, y_max] format to compute 
    their spatial dimensions and geometric centers. It utilizes ellipsis broadcasting to 
    support arbitrary batch dimensions seamlessly.
    
    Args:
        boxes (torch.Tensor): A tensor of shape (..., 4) containing the bounding boxes.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the 
        widths, heights, center_x, and center_y tensors, respectively.
    """
    # Get the width, height, x_center and y_center from the boxes using ellipsis to handle optional batch dimensions
    boxes_widths = boxes[..., 2] - boxes[..., 0]
    boxes_heights = boxes[..., 3] - boxes[..., 1]
    boxes_center_x = boxes[..., 0] + 0.5 * boxes_widths
    boxes_center_y = boxes[..., 1] + 0.5 * boxes_heights

    return boxes_widths, boxes_heights, boxes_center_x, boxes_center_y


def apply_regression_predictions(regression_predictions: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """
    Applies the predicted bounding box regression offsets to a set of reference boxes.
    
    This function takes the predicted bounding box delta coordinates (dx, dy, dw, dh) from
    the Region Proposal Network or the final Detector and applies them to the base anchor
    boxes (or proposed region boxes) to yield the final predicted spatial bounding boxes.
    The regression predictions must always be of shape (..., N, k, 4), where k=1 for region proposals 
    and k=number of classes for final detections. This supports both unbatched (N, k, 4) 
    and batched (B, N, k, 4) inputs seamlessly.
    
    Args:
        regression_predictions (torch.Tensor): A tensor of shape (..., N, k, 4) containing the predicted 
                                               offsets [dx, dy, dw, dh].
        boxes (torch.Tensor): A tensor of shape (..., N, 4) containing the reference bounding boxes
                              of shape [x_min, y_min, x_max, y_max].
        
    Returns:
        torch.Tensor: A tensor of shape (..., N, k, 4) containing the adjusted predicted 
                      bounding boxes in [x_min, y_min, x_max, y_max] format.
    """

    # Extract the foundational dimensions and centers from the base boxes
    boxes_widths, boxes_heights, boxes_center_x, boxes_center_y = get_box_dimensions_and_centers(boxes=boxes)

    # Unsqueeze across the last dimension (dim=-1) to append the `k` dimension.
    # This transforms shape (..., N) into (..., N, 1), which broadcasts correctly with (..., N, k).
    boxes_widths = boxes_widths.unsqueeze(dim=-1)
    boxes_heights = boxes_heights.unsqueeze(dim=-1)
    boxes_center_x = boxes_center_x.unsqueeze(dim=-1)
    boxes_center_y = boxes_center_y.unsqueeze(dim=-1)

    # Get the predictions from regression predictions using ellipsis to support arbitrary batch dims
    dx = regression_predictions[..., 0]
    dy = regression_predictions[..., 1]
    dw = regression_predictions[..., 2]
    dh = regression_predictions[..., 3]

    # Apply regressions to calculate the predicted centers
    predicted_center_x = boxes_widths * dx + boxes_center_x
    predicted_center_y = boxes_heights * dy + boxes_center_y

    # Apply regressions to calculate the predicted dimensions using the exponential function
    predicted_width = torch.exp(input=dw) * boxes_widths
    predicted_height = torch.exp(input=dh) * boxes_heights

    # Construct the final predicted boxes by converting back to [x_min, y_min, x_max, y_max] format
    predicted_boxes = torch.stack(tensors=[
        predicted_center_x - 0.5 * predicted_width,
        predicted_center_y - 0.5 * predicted_height,
        predicted_center_x + 0.5 * predicted_width,
        predicted_center_y + 0.5 * predicted_height], dim=-1)

    return predicted_boxes


def clamp_boxes_to_image_boundaries(boxes: torch.Tensor, input_image_size: int) -> torch.Tensor:
    """
    Clamps bounding box coordinates to ensure they remain strictly within the image dimensions.
    
    During region proposal or bounding box regression, the predicted offsets can sometimes 
    push the bounding box boundaries outside the actual image area. This utility function 
    safely truncates any out-of-bounds coordinates to the image edge (0 to input_image_size). 
    Since the network processes square images, a single dimension size is sufficient for both width and height.
    
    Args:
        boxes (torch.Tensor): A tensor of bounding boxes in [x_min, y_min, x_max, y_max] format.
        input_image_size (int): The spatial dimension (height and width) of the square input image.
        
    Returns:
        torch.Tensor: The bounding boxes tensor with all coordinates clamped within [0, input_image_size].
    """
    # Since we are always dealing with square images, we clamp both x and y coordinates uniformly
    boxes = boxes.clamp(min=0, max=input_image_size)

    return boxes


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
        ground_truth_boxes (torch.Tensor): A tensor of ground truth boxes in [x_min, y_min, x_max, y_max] format.
        anchors (torch.Tensor): A tensor of the base anchors generated for the current feature map.
        background_iou_threshold (Dict[str, float]): Dictionary defining 'min' and 'max' IoU thresholds for background.
        foreground_iou_threshold (Dict[str, float]): Dictionary defining 'min' and 'max' IoU thresholds for foreground.
        strict_fallback_assignment (bool): If True, assigns resurrected anchors strictly to the ground truth box
        that triggered their resurrection. If False, follows default PyTorch RPN behavior of assigning them to
        the ground truth box they overlap with most. Defaults to False.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - matched_ground_truth_boxes (torch.Tensor): The assigned ground truth boxes for each anchor.
            - labels (torch.Tensor): The binary labels for each anchor
                                     (1.0 = foreground, 0.0 = background, -1.0 = ignored).
    """

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
    Randomly samples positive and negative anchors to maintain a fixed ratio during Region Proposal Network training.

    To prevent the classification loss from being completely overwhelmed by the massive number of background 
    (negative) anchors, this function constructs a balanced mini-batch for each image in the batch. 
    It guarantees up to `desired_positives` foreground anchors, and fills the remainder of the `desired_total` quota 
    with background anchors.

    Args:
        labels (torch.Tensor): A tensor of shape (B, N) containing anchor labels (1.0 = positive, 0.0 = negative).
        desired_positives (int): The maximum number of positive anchors to sample per image.
        desired_total (int): The total combined number of anchors (positive + negative) to sample per image.
        verbose (bool): If True, prints a console summary detailing the total positive, negative,
         and combined anchors sampled across the batch. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing two boolean masks of shape (B, N):
            - sampled_positive_mask (torch.Tensor): True for selected positive anchors.
            - sampled_negative_mask (torch.Tensor): True for selected negative anchors.
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
        total_positives = torch.sum(input=sampled_positive_mask, dim=1)
        total_negatives = torch.sum(input=sampled_negative_mask, dim=1)
        print_yellow(output=f"Positive RPN Anchor Samples : {total_positives.tolist()}", indent=1)
        print_yellow(output=f"Negative RPN Anchor Samples : {total_negatives.tolist()}", indent=1)
        print_yellow(output=f"All Batched Anchor Samples  : {(total_negatives + total_positives).tolist()}", indent=1)

    return sampled_positive_mask, sampled_negative_mask


def assign_targets_to_proposals(ground_truth_boxes: torch.Tensor,
                                proposals: torch.Tensor,
                                ground_truth_labels: torch.Tensor,
                                foreground_iou_threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Assigns ground truth bounding boxes and class labels to region proposals for Detection Head training.
    
    This function matches the predicted region proposals (from the Region Proposal Network) against 
    the actual ground truth boxes using Intersection over Union (IoU). Proposals with an IoU 
    greater than or equal to the foreground threshold are assigned the corresponding ground truth class.
    Proposals falling below the threshold are assigned the background class (class index 0).

    Args:
        ground_truth_boxes (torch.Tensor): A tensor of ground truth boxes in [x_min, y_min, x_max, y_max] format.
        proposals (torch.Tensor): A tensor of proposed bounding boxes.
        ground_truth_labels (torch.Tensor): A tensor of the true class indices for each ground truth box.
        foreground_iou_threshold (float): The minimum IoU required for a proposal to be considered a positive match.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - target_ground_truth_boxes (torch.Tensor): The assigned ground truth coordinates for each proposal.
            - target_labels (torch.Tensor): The assigned class labels for each proposal (0 for background).
    """
    # Note to self to check afterwards : ground truth boxes and labels should have the same first dimension shape which are the number of ground truth boxes in the image

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

    return target_ground_truth_boxes, target_labels
