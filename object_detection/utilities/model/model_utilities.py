import cv2
import numpy as np
import torch

from typing import Tuple, Dict, List
from torchvision.ops import nms

from utilities.os_utilities import print_blue
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

    # Get the width, height, x_center and y_center from the boxes using ellipsis to handle optional batch dimensions
    boxes_widths = boxes[..., 2] - boxes[..., 0]
    boxes_heights = boxes[..., 3] - boxes[..., 1]
    boxes_center_x = boxes[..., 0] + 0.5 * boxes_widths
    boxes_center_y = boxes[..., 1] + 0.5 * boxes_heights

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
                              foreground_iou_threshold: Dict[str, float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Assigns ground truth boxes to anchors to prepare the binary classification and regression targets for the Region Proposal Network.

    This function sits at the heart of the RPN training pipeline. It computes the IoU between every anchor 
    and every ground truth box, categorizing each anchor as foreground (positive object), background (negative), 
    or grey zone (ignored during training). It guarantees that every ground truth box has at least one positive 
    anchor assigned to it to prevent objects from being missed during training.

    Args:
        ground_truth_boxes (torch.Tensor): A tensor of ground truth boxes in [x_min, y_min, x_max, y_max] format.
        anchors (torch.Tensor): A tensor of the base anchors generated for the current feature map.
        background_iou_threshold (Dict[str, float]): Dictionary defining the 'min' and 'max' IoU thresholds for background.
        foreground_iou_threshold (Dict[str, float]): Dictionary defining the 'min' and 'max' IoU thresholds for foreground.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - matched_ground_truth_boxes (torch.Tensor): The assigned ground truth boxes for each anchor.
            - labels (torch.Tensor): The binary labels for each anchor (1.0 = foreground, 0.0 = background, -1.0 = ignored).
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

    # # todo to be removed
    # # This gives us all the anchors that tie for the highest iou for each ground truth box
    # best_anchor_indices_for_each_ground_truth_box = torch.where(
    #     intersection_over_union_matrix == best_match_anchor_iou.unsqueeze(dim=-1))
    # print(best_anchor_indices_for_each_ground_truth_box)
    # # end of removal

    # This gives us all the anchors that tie for the highest iou for each ground truth box.
    # We use torch.isclose to safely handle float32 truncation artifacts where mathematically 
    # identical bounding box areas might differ by ~1e-7.
    best_anchor_indices_for_each_ground_truth_box = torch.where(
        torch.isclose(input=intersection_over_union_matrix,
                      other=best_match_anchor_iou.unsqueeze(dim=-1),
                      atol=1e-4))

    # We recover the positive labels for the anchors that had the highest IoU for each ground truth box.
    # IMPORTANT QUIRK: Notice that we are assigning the anchor to its absolute best match overall 
    # (using best_match_ground_truth_index_before_thresholding) rather than strictly forcing it to 
    # match the specific ground truth box that selected it. For example, if an anchor is forced 
    # positive to cover Ground Truth A, but it actually overlaps more with Ground Truth B, it will 
    # be assigned to predict Ground Truth B. This perfectly mirrors the official PyTorch RPN logic!
    anchor_indices_to_retrieve = best_anchor_indices_for_each_ground_truth_box[1]
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
                                    foreground_iou_threshold: Dict[str, float]) -> Tuple[torch.Tensor, torch.Tensor]:
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
            foreground_iou_threshold=foreground_iou_threshold)

        # Store the processed targets and labels
        aggregated_target_boxes.append(target_boxes)
        aggregated_labels.append(labels)

    # Stack the lists along a new batch dimension (dim=0)
    batched_target_boxes = torch.stack(tensors=aggregated_target_boxes, dim=0)
    batched_labels = torch.stack(tensors=aggregated_labels, dim=0)

    return batched_target_boxes, batched_labels
