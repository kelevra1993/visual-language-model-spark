import cv2
import numpy as np
import torch
from typing import Tuple

from utilities.os_utilities import print_blue
from torchvision.ops import nms


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


# TODO LATER : Might be moved elsewhere in the RPN Module but not sure
def filter_proposals(
    proposal_boxes: torch.Tensor,
    proposal_scores: torch.Tensor,
    input_image_size: int,
    pre_nms_filter_proposals: int,
    nms_iou_threshold: float,
    post_nms_filter_proposals: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filters region proposal bounding boxes using objectness scores and Non-Maximum Suppression (NMS).
    
    This function processes the raw proposals generated by the Region Proposal Network. It applies 
    a sigmoid activation to the scores, selects the top-k highest scoring proposals (Pre-NMS), 
    clamps their coordinates to the image boundaries, applies NMS to remove overlapping duplicates, 
    and finally selects the top-k proposals from the remaining set (Post-NMS).
    
    Args:
        proposal_boxes (torch.Tensor): A tensor containing the proposed bounding boxes.
        proposal_scores (torch.Tensor): A tensor containing the raw objectness logits for each box.
        input_image_size (int): The spatial dimension (height and width) of the square input image.
        pre_nms_filter_proposals (int): The maximum number of proposals to keep before applying NMS.
        nms_iou_threshold (float): The Intersection over Union (IoU) threshold for NMS.
        post_nms_filter_proposals (int): The maximum number of proposals to keep after applying NMS.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - proposal_boxes (torch.Tensor): The filtered bounding boxes.
            - proposal_scores (torch.Tensor): The objectness scores for the filtered boxes.
    """
    # Just putting all scores in one line
    proposal_scores = proposal_scores.reshape(-1)
    proposal_scores = torch.sigmoid(input=proposal_scores)
    _, top_proposal_scores = proposal_scores.topk(k=pre_nms_filter_proposals)

    proposal_scores = proposal_scores[top_proposal_scores]
    proposal_boxes = proposal_boxes[:, top_proposal_scores, :]

    # Clamp boxes to image boundary
    proposal_boxes = clamp_boxes_to_image_boundaries(boxes=proposal_boxes, input_image_size=input_image_size)

    # Apply NMS based on objectness : Remember here we kept the best proposal
    # Note that they do not all have values above 0.5
    proposal_post_nms_indices = nms(
        boxes=proposal_boxes,
        scores=proposal_scores,
        iou_threshold=nms_iou_threshold
    )

    # Post NMS top k filtering
    proposal_boxes = proposal_boxes[proposal_post_nms_indices][:post_nms_filter_proposals]
    proposal_scores = proposal_scores[proposal_post_nms_indices][:post_nms_filter_proposals]

    return proposal_boxes, proposal_scores