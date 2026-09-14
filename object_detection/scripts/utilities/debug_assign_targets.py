import torch

from utilities.model.model_utilities import assign_targets_to_anchors
from utilities.os_utilities import print_blue, print_green
from scripts.utilities.debugging_utilities import print_bounding_boxes
from utilities.tensor_utilities import print_tensor_shape


# Uncomment the import below once the function is ready in model_utilities.py
# from utilities.model.model_utilities import assign_targets_to_anchors


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
    batch_size = 3

    # Define a small set of anchors in [x_min, y_min, x_max, y_max] format
    # Shape: [number_of_anchors, 4]
    anchors = torch.tensor(data=[[10.0, 10.0, 50.0, 50.0],
                                 [20.0, 20.0, 80.0, 80.0],
                                 [20.0, 20.0, 80.0, 80.0],
                                 [100.0, 100.0, 150.0, 150.0],
                                 [110.0, 100.0, 190.0, 250.0],
                                 [150.0, 90.0, 200.0, 120.0],
                                 [150.0, 56.0, 82.0, 70.0],
                                 [12.0, 80.0, 10.0, 550.0],
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

    # Iterate over the batched dimensions to process the target assignments independently per image
    for batch_index in range(batch_size):
        # Test the target assignment module by matching the mock ground truth boxes to the predefined FPN anchors
        assigned_targets = assign_targets_to_anchors(
            ground_truth_boxes=ground_truth_boxes[batch_index],
            anchors=anchors[batch_index],
            background_iou_threshold=background_iou_threshold,
            foreground_iou_threshold=foreground_iou_threshold)


if __name__ == "__main__":
    debug_assign_targets()
