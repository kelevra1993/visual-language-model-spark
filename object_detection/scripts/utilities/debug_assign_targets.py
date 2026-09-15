import cv2
import numpy as np
import torch
from typing import Tuple

from utilities.model.model_utilities import assign_targets_to_anchors, get_intersection_over_union, add_bounding_box
from utilities.os_utilities import print_blue, print_green
from scripts.utilities.debugging_utilities import print_bounding_boxes
from utilities.tensor_utilities import print_tensor_shape


# Uncomment the import below once the function is ready in model_utilities.py
# from utilities.model.model_utilities import assign_targets_to_anchors


def get_box_color(label: float) -> Tuple[int, int, int]:
    """
    Returns the BGR color tuple based on the target assignment label.
    
    Args:
        label (float): The assigned label (1.0 for foreground, 0.0 for background, -1.0 for ignored).
        
    Returns:
        Tuple[int, int, int]: The corresponding BGR color.
    """
    if label == 1.0:
        return (0, 255, 0)
    elif label == 0.0:
        return (0, 0, 255)
    else:
        return (128, 128, 128)


def add_legend(image: np.ndarray, input_image_size: int) -> np.ndarray:
    """
    Draws a color legend on the top right corner of the canvas.
    
    Args:
        image (np.ndarray): The image canvas to draw on.
        input_image_size (int): The dimension of the square input image.
        
    Returns:
        np.ndarray: The image with the legend added.
    """
    legend_x = input_image_size - 240
    cv2.putText(img=image, text="Blue: Anchor Box", org=(legend_x, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7, color=(255, 0, 0), thickness=1)
    cv2.putText(img=image, text="Green: Positive Match", org=(legend_x, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7, color=(0, 255, 0), thickness=1)
    cv2.putText(img=image, text="Red: Negative Match", org=(legend_x, 90), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7, color=(0, 0, 255), thickness=1)
    cv2.putText(img=image, text="Grey: Ignored Zone", org=(legend_x, 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7, color=(128, 128, 128), thickness=1)
    return image


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

    # Iterate over the batched dimensions to process the target assignments independently per image
    for batch_index in range(batch_size):
        # Test the target assignment module by matching the mock ground truth boxes to the predefined FPN anchors
        target_ground_truth_boxes, labels = assign_targets_to_anchors(
            ground_truth_boxes=ground_truth_boxes[batch_index], anchors=anchors[batch_index],
            background_iou_threshold=background_iou_threshold, foreground_iou_threshold=foreground_iou_threshold)

        # Iterate through each anchor and its corresponding target assignment to visualize the matching logic
        for anchor_index, (anchor, target_ground_truth, label) in enumerate(
                zip(anchors[batch_index], target_ground_truth_boxes, labels)):
            # Create a black canvas (600x600 resolution) for visualizing the bounding boxes
            canvas = np.zeros(shape=(input_image_size, input_image_size, 3), dtype=np.uint8)

            # Draw the anchor bounding box in blue
            anchor_array = anchor.cpu().numpy().astype(dtype=np.int32)
            canvas = add_bounding_box(bounding_box=anchor_array, image=canvas, input_image_size=input_image_size, color=(255, 0, 0))

            # Determine the bounding box color based on the assigned label class
            box_color = get_box_color(label=label.item())

            # Draw the target ground truth bounding box in the assigned color
            target_array = target_ground_truth.cpu().numpy().astype(dtype=np.int32)
            canvas = add_bounding_box(bounding_box=target_array, image=canvas, input_image_size=input_image_size, color=box_color)

            # Calculate the Intersection over Union (IoU) to overlay as text
            iou_matrix = get_intersection_over_union(boxes_1=anchor.unsqueeze(dim=0),
                                                     boxes_2=target_ground_truth.unsqueeze(dim=0))
            iou_value = iou_matrix[0, 0].item()

            # Overlay the IoU value as white text near the top-left of the anchor
            anchor_x_min, anchor_y_min, _, _ = map(int, anchor.tolist())
            cv2.putText(img=canvas, text=f"IoU: {iou_value:.3f}", org=(anchor_x_min, max(anchor_y_min - 10, 20)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2)

            # Draw a legend on the top right corner of the canvas for clarity
            canvas = add_legend(image=canvas, input_image_size=input_image_size)

            # Display the visualization and wait for any key press before proceeding to the next anchor
            cv2.imshow(winname="Anchor and Target Visualization", mat=canvas)
            cv2.waitKey(delay=0)

    # Destroy all OpenCV windows after the visualization loop completes
    cv2.destroyAllWindows()


if __name__ == "__main__":
    debug_assign_targets()
