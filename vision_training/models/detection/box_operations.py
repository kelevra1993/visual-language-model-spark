"""
Bounding box operations of the object detection task: areas, intersection over union, widths, heights and centers,
decoding of the regression predictions and clamping to the image. Used by the region proposer, the detection heads,
the postprocessing, the inference detector, the target assignment and the mean average precision evaluation.
"""

import math
import torch

from typing import Tuple

from vision_training.utilities.os_utilities import print_color
from vision_training.utilities.tensor_utilities import print_tensor_shape


# Upper bound of the predicted log-scale width and height offsets (dw, dh) before exponentiation, as in torchvision.
# It caps a box at 1000/16 = 62.5 times its reference size and prevents exp() from overflowing to inf.
MAXIMUM_LOG_SCALE_OFFSET = math.log(1000.0 / 16)

# Weights of the regression offsets (dx, dy, dw, dh) of both stages: the targets are multiplied by them
# (turn_boxes_to_transformation_targets) and the predictions divided by them before decoding
# (apply_regression_predictions). Kept at (1, 1, 1, 1), the offsets being regressed unscaled: on the synthetic shapes,
# with the smooth L1 beta of 1/9, the (10, 10, 5, 5) of the second stage of torchvision and Detectron2 gave about
# 2.5 times lower box and mask mAP@0.5 after 2000 iterations. Fixed in the code, without configuration option.
BOX_REGRESSION_WEIGHTS = (1.0, 1.0, 1.0, 1.0)


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
    Calculates the Intersection over Union (IoU) matrix between two sets of bounding boxes, for every image of a batch
    at once when the boxes have leading batch dimensions.

    This function computes the overlap between anchor boxes and ground truth boxes (or proposals), which the target
    assignment of the region proposal network and of the detection heads needs for every image of a training batch,
    and between detections and ground truth boxes for the mean average precision and the confusion matrix. The union
    is clamped to the smallest positive value of its type instead of being increased by an epsilon, so that an exact
    ratio such as 3 / 4 stays exact and reaches its threshold (like the mask overlap of the instance segmentation),
    and two empty boxes still give 0.

    Args:
        boxes_1 (torch.Tensor): A tensor of shape (..., N, 4) for first set of bounding boxes [x_min, y_min, x_max,
            y_max].
        boxes_2 (torch.Tensor): A tensor of shape (..., M, 4) for second set of bounding boxes [x_min, y_min, x_max,
            y_max], its leading dimensions broadcasting with those of boxes_1.

    Returns:
        torch.Tensor: A tensor of shape (..., N, M) containing the computed IoU values for every pair of boxes.
    """
    # Areas of both sets of boxes, shapes (..., N) and (..., M)
    area_1 = get_area(boxes=boxes_1)
    area_2 = get_area(boxes=boxes_2)

    # Get the top left coordinates of intersection area using broadcasting to create an (..., N, M, 2) tensor
    top_left_coordinates = torch.max(input=boxes_1.unsqueeze(dim=-2)[..., :2], other=boxes_2.unsqueeze(dim=-3)[..., :2])

    # Get the bottom right coordinates of intersection area using broadcasting to create an (..., N, M, 2) tensor
    bottom_right_coordinates = torch.min(input=boxes_1.unsqueeze(dim=-2)[..., 2:],
                                         other=boxes_2.unsqueeze(dim=-3)[..., 2:])

    # Compute intersection area (clamped to 0, to avoid negative values)
    intersection_dimensions = (bottom_right_coordinates - top_left_coordinates).clamp(min=0)
    intersection_area = intersection_dimensions[..., 0] * intersection_dimensions[..., 1]

    # Compute union area by summing individual areas and subtracting the overlapping intersection area
    union_area = area_1.unsqueeze(dim=-1) + area_2.unsqueeze(dim=-2) - intersection_area

    # Divide by the union clamped to the smallest positive value of its type, exact for every non-empty union and 0
    # for two empty boxes, of shape (..., N, M)
    intersection_over_union = intersection_area / union_area.clamp(min=torch.finfo(union_area.dtype).tiny)

    return intersection_over_union


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
        widths, heights, center_x, and center_y tensors, respectively, each of shape (...).
    """
    # Check that the last dimension is of shape 4 if not raise an error. The check is skipped while TorchScript traces
    # the model for export, since the shapes are fixed by then and tracing would warn about turning it into a constant
    if not torch.jit.is_tracing() and boxes.shape[-1] != 4:
        print_tensor_shape(boxes)
        print_color(color="red",
                    output=f"Unexpected Tensor Shape For Boxes, The Last Dimension Is {boxes.shape[-1]} Instead Of 4")
        exit()

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
    and batched (B, N, k, 4) inputs seamlessly. The predictions are the weighted offsets the heads are trained on,
    divided by BOX_REGRESSION_WEIGHTS before being applied.
    
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

    # Get the offsets from the weighted regression predictions, using ellipsis to support arbitrary batch dims
    dx = regression_predictions[..., 0] / BOX_REGRESSION_WEIGHTS[0]
    dy = regression_predictions[..., 1] / BOX_REGRESSION_WEIGHTS[1]
    # Clamp the unweighted log-scale offsets so that exp() stays finite, even for extreme early-training predictions
    dw = (regression_predictions[..., 2] / BOX_REGRESSION_WEIGHTS[2]).clamp(max=MAXIMUM_LOG_SCALE_OFFSET)
    dh = (regression_predictions[..., 3] / BOX_REGRESSION_WEIGHTS[3]).clamp(max=MAXIMUM_LOG_SCALE_OFFSET)

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
        boxes (torch.Tensor): A tensor of bounding boxes in [x_min, y_min, x_max, y_max] format, shape (..., 4).
        input_image_size (int): The spatial dimension (height and width) of the square input image.
        
    Returns:
        torch.Tensor: The bounding boxes tensor with all coordinates clamped within [0, input_image_size], shape (..., 4).
    """
    # Since we are always dealing with square images, we clamp both x and y coordinates uniformly
    boxes = boxes.clamp(min=0, max=input_image_size)

    return boxes
