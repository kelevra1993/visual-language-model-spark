import cv2
import numpy as np
import torch

from typing import Tuple
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


def add_bounding_box(bounding_box: np.ndarray, image: np.ndarray, input_image_size: int,
                     color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draws a single bounding box onto the provided image canvas.
    
    This function processes an individual bounding box's coordinates, clamps them to the image boundaries
    to prevent out-of-bounds drawing errors, and renders the rectangle using OpenCV. It centralizes 
    the bounding box drawing logic so it can be reused iteratively or collectively during visualization.
    
    Args:
        bounding_box (np.ndarray): The [x_min, y_min, x_max, y_max] coordinates of the bounding box.
        image (np.ndarray): The image canvas on which to draw the bounding box.
        input_image_size (int): The spatial size (height and width) of the image to clamp coordinates.
        color (Tuple[int, int, int]): The RGB color tuple for the bounding box. Defaults to green (0, 255, 0).
        
    Returns:
        np.ndarray: The updated image canvas containing the newly drawn bounding box.
    """
    x_min, y_min, x_max, y_max = bounding_box

    # Ensure coordinates are within image boundaries for clean visualization
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(input_image_size, x_max)
    y_max = min(input_image_size, y_max)

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

        # Iterate through each anchor and draw them one by one
        for anchor_index, anchor in enumerate(anchors_array):
            canvas = add_bounding_box(bounding_box=anchor, image=canvas, input_image_size=input_image_size)

            # Display the updated canvas containing the newly drawn anchor
            cv2.imshow(winname=window_name, mat=canvas)

            # Pause execution for the specified delay or until the user presses a key
            cv2.waitKey(delay=delay)
    else:
        # Iterate through all anchors and draw them collectively on the canvas
        for anchor in anchors_array:
            canvas = add_bounding_box(bounding_box=anchor, image=canvas, input_image_size=input_image_size)

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


def assign_targets_to_anchors(ground_truth_boxes, anchors, background_iou_threshold, foreground_iou_threshold):
    """todo add documentation"""

    # todo add small comment of shape
    intersection_over_union_matrix = get_intersection_over_union(boxes_1=ground_truth_boxes, boxes_2=anchors)
    print("Intersection Over Union Matrix")
    print_tensor_shape(intersection_over_union_matrix, "intersection_over_union_matrix")
    print_tensor_list(intersection_over_union_matrix)

    # todo add small comment explaination
    best_match_ground_truth_iou, best_match_ground_truth_index = intersection_over_union_matrix.max(dim=0)
    print_tensor_shape(best_match_ground_truth_index, "best_match_ground_truth_index")
    print("----Best Matching Ground Truth Index For Each Anchor Box----")
    print_tensor_list(best_match_ground_truth_index)
    print("----Best Matching Ground Truth IOU For Each Anchor Box----")
    print_tensor_list(best_match_ground_truth_iou)

    # todo add small comment explaination (to always get at least one positive anchors per ground truth box even if overlap is lower than threshold)
    best_match_ground_truth_index_before_thresholding = best_match_ground_truth_index.clone()

    # Create readable condition variables based on the IoU thresholds
    below_background_min = best_match_ground_truth_iou < background_iou_threshold['min']
    above_or_equal_background_min = best_match_ground_truth_iou >= background_iou_threshold['min']
    below_background_max = best_match_ground_truth_iou < background_iou_threshold['max']

    above_or_equal_background_max = best_match_ground_truth_iou >= background_iou_threshold['max']

    below_foreground_min = best_match_ground_truth_iou < foreground_iou_threshold['min']
    above_or_equal_foreground_min = best_match_ground_truth_iou >= foreground_iou_threshold['min']
    below_or_equal_foreground_max = best_match_ground_truth_iou <= foreground_iou_threshold['max']

    # todo add small comment explaination
    background_indices = above_or_equal_background_min & below_background_max
    print("----Background Indices----")
    print(background_indices)

    # todo add small comment explaination
    grey_zone_indices = (above_or_equal_background_max & below_foreground_min) | below_background_min
    print("----Grey Zone Indices----")
    print(grey_zone_indices)

    # todo add small comment explaination
    foreground_indices = above_or_equal_foreground_min & below_or_equal_foreground_max
    print("----Foreground Indices----")
    print(foreground_indices)

    # Assign labels to our different anchors
    # Each anchor for which it's highest scoring iou ground truth box
    # is either considered background or in the grey zone are labelled as such :
    # - background > -1
    # - grey zone > -2
    print("----Best Matching Ground Truth Box Index For Each Anchor Before And After Assignment----")
    print_tensor_list(best_match_ground_truth_index)
    # print_tensor_list(best_match_ground_truth_iou)
    best_match_ground_truth_index[background_indices] = -1
    best_match_ground_truth_index[grey_zone_indices] = -2
    print_tensor_list(best_match_ground_truth_index)

    # Making sure that every ground truth will have at least one positive anchor bounding box
    # Get the best iou value for each anchor
    best_match_anchor_iou, _ = intersection_over_union_matrix.max(dim=1)
    print("----Best Matching Anchor IOU For Each Ground Truth Box----")
    print_tensor_shape(best_match_anchor_iou, "best_match_anchor_iou")
    print_tensor_list(best_match_anchor_iou)

    # This gives us all the anchors with the highest iou for each ground truth box
    best_anchor_indices_for_each_ground_truth_box = torch.where(
        intersection_over_union_matrix == best_match_anchor_iou.unsqueeze(dim=-1))
    print(best_anchor_indices_for_each_ground_truth_box)

    # We assign each anchor that corresponds to the maximum iou for a given ground truth the index of that ground truth
    anchor_indices_to_retrieve = best_anchor_indices_for_each_ground_truth_box[1]
    best_match_ground_truth_index[anchor_indices_to_retrieve] = best_match_ground_truth_index_before_thresholding[
        anchor_indices_to_retrieve]

    # Get coordinates of best matching ground truth target boxes (so each anchor will have at least one target)
    # But we will not necessarily train on all the assigned targets because the -1 and -2 will be bogus targets.
    # Be careful with coordinates -1 and -2 to clamp to 0 to always have the same coordinates for background and grey zone
    matched_ground_truth_boxes = ground_truth_boxes[best_match_ground_truth_index.clamp(0)]
    print_tensor_shape(matched_ground_truth_boxes, "target_coordinates")
    print_tensor_shape(anchors, "anchors")

    # Now set all labels for training so there is no ambiguity
    # For classification loss we consider labels above 0, where 0 is background and 1 is foreground
    # For bounding box regression we only consider labels equal to 1 which are foreground
    # Shape (anchor_boxes)
    labels = (best_match_ground_truth_index >= 0).to(dtype=torch.float32)
    labels[best_match_ground_truth_index == -1] = 0.0  # set -1 to 0 (background)
    labels[best_match_ground_truth_index == -2] = -1.0  # set -2 to -1 (ignored)
    print_tensor_list(best_match_ground_truth_index)
    print_tensor_list(labels)

    return matched_ground_truth_boxes, labels
