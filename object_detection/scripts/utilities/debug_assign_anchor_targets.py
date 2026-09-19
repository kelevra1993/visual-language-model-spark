import cv2
import numpy as np
import torch
from typing import Tuple

from utilities.model.model_utilities import batch_assign_targets_to_anchors, get_intersection_over_union, \
    add_bounding_boxes
from utilities.os_utilities import print_blue, print_green
from scripts.utilities.debugging_utilities import print_bounding_boxes, visualize_anchor_target_assignments, \
    visualize_foreground_and_background_anchors
from utilities.tensor_utilities import print_tensor_shape


def debug_assign_targets() -> None:
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

    # Define a small set of anchors in [x_min, y_min, x_max, y_max] format
    # Shape: [number_of_anchors, 4]
    anchors = torch.tensor(data=[[10.0, 10.0, 50.0, 50.0],
                                 [20.0, 20.0, 80.0, 80.0],
                                 [20.0, 20.0, 80.0, 80.0],
                                 [100.0, 100.0, 150.0, 150.0],
                                 [110.0, 100.0, 190.0, 250.0],
                                 [150.0, 90.0, 200.0, 120.0],
                                 [82.0, 56.0, 150.0, 70.0],
                                 [12.0, 80.0, 50.0, 240.0],
                                 [67.0, 35.0, 135.0, 176.0],
                                 [67.0, 12.0, 243.0, 245.0],
                                 [94.0, 12.0, 142.0, 352.0],
                                 [94.0, 65.0, 120.0, 245.0],
                                 [200.0, 200.0, 300.0, 300.0]], dtype=dtype, device=device)

    # Batched anchor shape: [batch_size, number_anchors, 4]
    anchors = anchors.unsqueeze(dim=0).expand(size=(batch_size, -1, 4))

    # Define a small set of ground truth bounding boxes in [x_min, y_min, x_max, y_max] format
    # Shape: [batch_size, number_of_ground_truths, 4].
    ground_truth_boxes = torch.tensor(data=[[12.0, 12.0, 48.0, 48.0],
                                            [210.0, 210.0, 290.0, 290.0],
                                            [0.0, 70.0, 200.0, 120.0],
                                            [150.0, 96.0, 230.0, 120.0],
                                            [30.0, 60.0, 50.0, 150.0]], dtype=dtype, device=device)

    # Batched ground truth boxes shape: [batch_size, number_anchors, 4]
    ground_truth_boxes = ground_truth_boxes.unsqueeze(dim=0).expand(size=(batch_size, -1, 4))

    print_blue(output="Prepared Anchors:", add_separators=True)
    print_tensor_shape(anchors, "anchors")
    print_bounding_boxes(boxes=anchors[0], number_of_boxes=anchors[0].shape[0])

    print_blue(output="Prepared Ground Truth Boxes:", add_separators=True)
    print_tensor_shape(ground_truth_boxes, "ground_truth_boxes")
    print_bounding_boxes(boxes=ground_truth_boxes[0], number_of_boxes=ground_truth_boxes.shape[1])

    print_green(output="Mock data successfully instantiated and ready for target assignment testing!",
                add_separators=True)

    # Define the Intersection over Union (IoU) threshold dictionaries.
    # These threshold bounds strictly follow the project's YAML configuration and dictate 
    # whether an anchor is considered a positive match (foreground) or negative (background).
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
    debug_assign_targets()
