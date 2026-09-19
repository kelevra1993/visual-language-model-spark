import cv2
import numpy as np
import torch
from typing import Tuple

from utilities.model.model_utilities import batch_assign_targets_to_anchors, get_intersection_over_union, \
    add_bounding_boxes
from utilities.os_utilities import print_blue, print_green
from scripts.utilities.debugging_utilities import print_bounding_boxes, visualize_anchor_target_assignments, \
    visualize_foreground_and_background_anchors, get_assignment_debugger_input
from utilities.tensor_utilities import print_tensor_shape


def debug_assign_anchor_targets() -> None:
    """
    Creates ground truth boxes and anchors to test the target assignment logic.

    This debugging script is used to verify the behavior of the assign_targets_to_anchors
    function, which matches predicted region proposal anchors to the actual ground truth 
    bounding boxes to prepare for calculating the classification and regression losses.

    Args:
        None
    """
    device = torch.device(device="cpu")
    dtype = torch.float32
    batch_size = 2
    input_image_size = 600

    # Fetch mutualized debugger mock data
    anchors, ground_truth_boxes, _ = get_assignment_debugger_input(
        batch_size=batch_size, device=device, dtype=dtype)

    print_blue(output="Prepared Anchors:", add_separators=True)
    print_tensor_shape(anchors, "anchors")
    print_bounding_boxes(boxes=anchors[0], number_of_boxes=anchors[0].shape[0])

    print_blue(output="Prepared Ground Truth Boxes:", add_separators=True)
    print_tensor_shape(ground_truth_boxes, "ground_truth_boxes")
    print_bounding_boxes(boxes=ground_truth_boxes[0], number_of_boxes=ground_truth_boxes.shape[1])

    print_green(output="Mock data successfully instantiated and ready for target assignment testing!",
                add_separators=True)

    # Define the Intersection over Union (IoU) threshold dictionaries.
    background_iou_threshold = {"min": 0.1, "max": 0.3}
    foreground_iou_threshold = {"min": 0.5, "max": 1.0}

    # Test the target assignment module by matching the mock ground truth boxes to the predefined FPN anchors across the whole batch
    batched_target_ground_truth_boxes, batched_labels = batch_assign_targets_to_anchors(
        batched_ground_truth_boxes=ground_truth_boxes,
        batched_anchors=anchors,
        background_iou_threshold=background_iou_threshold,
        foreground_iou_threshold=foreground_iou_threshold)

    visualize_anchor_target_assignments(ground_truth_bounding_boxes=ground_truth_boxes,
                                        batched_target_ground_truth_boxes=batched_target_ground_truth_boxes,
                                        batched_labels=batched_labels,
                                        batched_anchors=anchors,
                                        input_image_size=input_image_size)

    # Only run for the first batch.
    visualize_foreground_and_background_anchors(ground_truth_boxes=ground_truth_boxes[0],
                                                anchors=anchors[0],
                                                background_iou_threshold=background_iou_threshold,
                                                foreground_iou_threshold=foreground_iou_threshold,
                                                input_image_size=input_image_size)


if __name__ == "__main__":
    debug_assign_anchor_targets()
