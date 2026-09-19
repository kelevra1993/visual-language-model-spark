import cv2
import numpy as np
import torch
from typing import Tuple

from utilities.model.model_utilities import assign_targets_to_proposals, batch_assign_targets_to_proposals, \
    get_intersection_over_union, \
    add_bounding_boxes
from utilities.os_utilities import print_blue, print_green, print_red
from scripts.utilities.debugging_utilities import print_bounding_boxes, get_assignment_debugger_input, visualize_proposal_target_assignments
from utilities.tensor_utilities import print_tensor_shape


def debug_assign_proposal_targets() -> None:
    """
    Creates ground truth boxes, labels, and mock proposals to test the target assignment logic
    for the Detection Head.

    This debugging script is used to verify the behavior of the assign_targets_to_proposals
    function, which matches predicted region proposals to the actual ground truth 
    bounding boxes and labels to prepare for calculating the classification and regression losses.

    Args:
        None
    """
    device = torch.device(device="cpu")
    dtype = torch.float32
    input_image_size = 600
    batch_size = 2

    # Fetch mutualized debugger mock data
    batched_proposals, batched_ground_truth_boxes, batched_ground_truth_labels = get_assignment_debugger_input(
        batch_size=batch_size, device=device, dtype=dtype)

    print_blue(output="Prepared Batched Proposals:", add_separators=True)
    print_tensor_shape(batched_proposals, "batched_proposals")
    print_bounding_boxes(boxes=batched_proposals[0], number_of_boxes=batched_proposals.shape[1])

    print_blue(output="Prepared Batched Ground Truth Boxes & Labels:", add_separators=True)
    print_tensor_shape(batched_ground_truth_boxes, "batched_ground_truth_boxes")
    print_tensor_shape(batched_ground_truth_labels, "batched_ground_truth_labels")
    for box, label in zip(batched_ground_truth_boxes[0], batched_ground_truth_labels[0]):
        print(f"Box: {box.tolist()} | Label: {label.item()}")

    print_green(output="Mock data successfully instantiated and ready for batched target assignment testing!",
                add_separators=True)

    # Test the batched target assignment module by matching the mock ground truth boxes to the proposals
    foreground_iou_threshold = 0.5

    batched_target_ground_truth_boxes, batched_target_labels = batch_assign_targets_to_proposals(
        batched_ground_truth_boxes=batched_ground_truth_boxes,
        batched_proposals=batched_proposals,
        batched_ground_truth_labels=batched_ground_truth_labels,
        foreground_iou_threshold=foreground_iou_threshold)

    print_blue(output="Batched Assignment Results (Image 0):", add_separators=True)
    print_tensor_shape(batched_target_ground_truth_boxes, "batched_target_ground_truth_boxes")
    print_tensor_shape(batched_target_labels, "batched_target_labels")

    for idx, (box, label) in enumerate(zip(batched_target_ground_truth_boxes[0], batched_target_labels[0])):
        is_bg = " (Background)" if label.item() == 0 else " (Foreground)"
        print(f"Proposal {idx:02d} -> Assigned GT Box: {box.tolist()} | Assigned Label: {label.item()}{is_bg}")

    # Visualize the batched target assignments interactively
    visualize_proposal_target_assignments(
        ground_truth_bounding_boxes=batched_ground_truth_boxes,
        batched_target_ground_truth_boxes=batched_target_ground_truth_boxes,
        batched_labels=batched_target_labels,
        batched_proposals=batched_proposals,
        input_image_size=input_image_size)


if __name__ == "__main__":
    debug_assign_proposal_targets()
