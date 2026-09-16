import cv2
import torch
import numpy as np
from typing import List, Optional, Tuple

from utilities.model.model_utilities import get_area, get_intersection_over_union, add_bounding_boxes
from utilities.os_utilities import print_green, print_red
from utilities.tensor_utilities import print_tensor_list


def print_bounding_boxes(boxes: torch.Tensor, number_of_boxes: int = 5, indent: int = 0) -> None:
    """
    Prints a formatted summary of bounding box coordinates alongside their area and calculated scale.

    This debugging utility is used across the testing scripts to inspect generated anchors
    and predicted bounding box regressions, ensuring they conform to logical bounds
    and expected spatial scales before they enter the loss calculations.

    Args:
        boxes (torch.Tensor): A tensor containing bounding boxes in [x_min, y_min, x_max, y_max] format.
        number_of_boxes (int): The number of boxes to print from the top of the tensor. Defaults to 5.
        indent (int, optional): The number of indentation levels (2 spaces each). Defaults to 0.
    """
    indentation_string = (indent * 2 * " ") + "- " if indent > 0 else ""
    boxes_to_print = boxes[:number_of_boxes]
    box_areas = get_area(boxes=boxes_to_print)

    for box, area in zip(boxes_to_print, box_areas):
        box_coordinates = np.round(a=box.tolist(), decimals=4)
        area_value = area.item()
        scale_value = np.sqrt(area_value)
        coordinates_string = (f"[{box_coordinates[0]:>8.2f},"
                              f" {box_coordinates[1]:>8.2f},"
                              f" {box_coordinates[2]:>8.2f},"
                              f" {box_coordinates[3]:>8.2f}]")
        print(f"{indentation_string}"
              f"{coordinates_string}  ::"
              f"  Area: {area_value:>10.2f}  ::"
              f"  Scale: {scale_value:>8.2f}")


def generate_random_bounding_boxes(scales: List[float], aspect_ratios: List[float],
                                   input_image_size: int, number_of_boxes: int,
                                   device: Optional[torch.device] = None,
                                   dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Generates a tensor of random bounding boxes for debugging and mock testing.

    This function is used during testing to simulate ground truth targets by sampling
    random scales, aspect ratios, and positions. It ensures that the generated boxes
    are spatially realistic and fit mostly within the bounds of the given image size.

    Args:
        scales (List[float]): A list of base scales (areas) to sample from.
        aspect_ratios (List[float]): A list of aspect ratios (width/height) to sample from.
        input_image_size (int): The width and height of the input image.
        number_of_boxes (int): The total number of boxes to generate.
        device (Optional[torch.device], optional): The device on which to place the tensor.
        dtype (Optional[torch.dtype], optional): The data type for the tensor.

    Returns:
        torch.Tensor: A tensor of shape [number_of_boxes, 4] containing the mock
                      bounding boxes in [x_min, y_min, x_max, y_max] format.
    """
    boxes = []

    # Generate each box sequentially to apply unique random properties
    for _ in range(number_of_boxes):
        # Sample base properties from the provided valid configurations
        base_scale = np.random.choice(a=scales)
        aspect_ratio = np.random.choice(a=aspect_ratios)

        # Apply a random multiplier between 1.0 and 1.5 to introduce realistic variability
        multiplier = np.random.uniform(low=1.0, high=1.5)
        final_scale = base_scale * multiplier

        # Compute width and height based on the randomized scale and aspect ratio
        width = final_scale * np.sqrt(aspect_ratio)
        height = final_scale / np.sqrt(aspect_ratio)

        # Determine top-left corner coordinates randomly across the entire image
        x_min = np.random.uniform(low=0.0, high=float(input_image_size))
        y_min = np.random.uniform(low=0.0, high=float(input_image_size))

        # Calculate bottom-right corner based on width and height
        x_max = x_min + width
        y_max = y_min + height

        # Clamp all coordinates to ensure they strictly remain within the image boundaries
        x_min = max(0.0, min(x_min, float(input_image_size)))
        y_min = max(0.0, min(y_min, float(input_image_size)))
        x_max = max(0.0, min(x_max, float(input_image_size)))
        y_max = max(0.0, min(y_max, float(input_image_size)))

        boxes.append([x_min, y_min, x_max, y_max])

    return torch.tensor(data=boxes, dtype=dtype, device=device)


def get_box_color(label: float) -> Tuple[int, int, int]:
    """
    Returns the BGR color tuple based on the target assignment label.
    
    Args:
        label (float): The assigned label (1.0 for foreground, 0.0 for background, -1.0 for ignored).
        
    Returns:
        Tuple[int, int, int]: The corresponding BGR color.
    """
    if label == 1.0:
        return 0, 255, 0
    elif label == 0.0:
        return 0, 0, 255
    else:
        return 128, 128, 128


def add_legend(image: np.ndarray, input_image_size: int) -> np.ndarray:
    """
    Draws a color legend on the top right corner of the canvas.
    
    This utility adds visual context to the debugging windows, detailing what each color
    represents (Anchor, Positive Match, Negative Match, Ignored) and whether each match class
    contributes to the downstream regression loss computation.
    
    Args:
        image (np.ndarray): The image canvas to draw on.
        input_image_size (int): The dimension of the square input image to position the legend.
        
    Returns:
        np.ndarray: The image with the color legend added.
    """
    legend_x = input_image_size - 280
    cv2.putText(img=image, text="Blue: Anchor Box", org=(legend_x, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7, color=(255, 0, 0), thickness=1)
    cv2.putText(img=image, text="Green: Positive Match \nUsed For Regression Loss", org=(legend_x, 90),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7, color=(0, 255, 0), thickness=1)
    cv2.putText(img=image, text="Red: Negative Match \nIgnored For Regression Loss", org=(legend_x, 150),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7, color=(0, 0, 255), thickness=1)
    cv2.putText(img=image, text="Grey: Ignored Zone \nIgnored For Regression Loss", org=(legend_x, 210),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7, color=(128, 128, 128), thickness=1)
    return image


def visualize_anchor_target_assignments(ground_truth_bounding_boxes: List[torch.Tensor],
                                        batched_target_ground_truth_boxes: torch.Tensor,
                                        batched_labels: torch.Tensor,
                                        batched_anchors: torch.Tensor,
                                        input_image_size: int) -> None:
    """
    Visualizes the anchor to ground truth matching logic using interactive OpenCV windows.

    This debugging utility iterates over batched assignments and provides a dual-view:
    1. A holistic view of all ground truth boxes for the image.
    2. An interactive, anchor-by-anchor view detailing the matching logic. 
       Positive matches display both the anchor (blue) and the matched ground truth box (green). 
       Negative and ignored matches dynamically recalculate and draw the actual highest-IoU ground truth box 
       (in red or grey) that caused their rejection, along with text detailing whether they 
       are hard/easy negatives and how they factor into classification losses.
       Users can navigate interactively using Spacebar/Enter keybindings.

    Args:
        ground_truth_bounding_boxes (List[torch.Tensor]): The raw unbatched ground truth boxes for each image.
        batched_target_ground_truth_boxes (torch.Tensor): Tensor of shape [batch_size, num_anchors, 4]
         containing assigned targets.
        batched_labels (torch.Tensor): Tensor of shape [batch_size, num_anchors] with assigned match labels.
        batched_anchors (torch.Tensor): Tensor of shape [batch_size, num_anchors, 4] of all generated anchors.
        input_image_size (int): The height/width of the input image canvas.
    """
    batch_size = batched_anchors.shape[0]

    # Iterate over the batched dimensions to process the target assignments independently per image for visualization
    for batch_index in range(batch_size):
        # Create an initial canvas to show all raw ground truth bounding boxes for this image
        ground_truth_canvas = np.zeros(shape=(input_image_size, input_image_size, 3), dtype=np.uint8)

        # Draw all original ground truth boxes at once in yellow using the batched bounding box utility
        original_ground_truth_boxes = ground_truth_bounding_boxes[batch_index]
        boxes_array = original_ground_truth_boxes.cpu().numpy().astype(dtype=np.int32)
        ground_truth_canvas = add_bounding_boxes(bounding_boxes=boxes_array,
                                                 image=ground_truth_canvas,
                                                 input_image_size=input_image_size,
                                                 color=(0, 255, 255))

        cv2.putText(img=ground_truth_canvas, text="All Ground Truth Boxes", org=(20, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2)
        cv2.imshow(winname="All Ground Truth Boxes", mat=ground_truth_canvas)

        # Extract the specific assignments and targets for the current image in the batch
        target_ground_truth_boxes = batched_target_ground_truth_boxes[batch_index]
        labels = batched_labels[batch_index]
        anchors = batched_anchors[batch_index]
        # print_tensor_list(torch.cat([target_ground_truth_boxes, labels.unsqueeze(-1)], dim=-1))
        # exit()
        # Iterate through each anchor and its corresponding target assignment to visualize the matching logic
        for anchor_index, (anchor, target_ground_truth, label) in enumerate(
                zip(anchors, target_ground_truth_boxes, labels)):
            # Create a black canvas for visualizing the bounding boxes
            canvas = np.zeros(shape=(input_image_size, input_image_size, 3), dtype=np.uint8)

            # Draw the anchor bounding box in blue
            anchor_array = anchor.cpu().numpy().astype(dtype=np.int32)
            canvas = add_bounding_boxes(bounding_boxes=anchor_array,
                                        image=canvas,
                                        input_image_size=input_image_size,
                                        color=(255, 0, 0))

            # Determine the bounding box color based on the assigned label class
            box_color = get_box_color(label=label.item())

            if label.item() <= 0.0:
                # Recompute IoU against all true ground truth boxes
                # to find the one that caused this negative/ignored assignment
                iou_matrix = get_intersection_over_union(boxes_1=anchor.unsqueeze(dim=0),
                                                         boxes_2=original_ground_truth_boxes)
                max_iou_values, max_iou_indices = torch.max(input=iou_matrix, dim=1)
                best_match_index = max_iou_indices[0].item()
                target_ground_truth = original_ground_truth_boxes[best_match_index]
                iou_value = max_iou_values[0].item()
                print_red(f"Non Positive IOU Value :: {iou_value}")

            else:
                # For positive anchors, use the assigned target
                # since it might have bypassed max IoU due to fallback rules
                iou_matrix = get_intersection_over_union(boxes_1=anchor.unsqueeze(dim=0),
                                                         boxes_2=target_ground_truth.unsqueeze(dim=0))
                iou_value = iou_matrix[0, 0].item()
                print_green(f"Positive IOU Value :: {iou_value}")

            # Always draw the true target ground truth bounding box that the anchor was measured against
            target_array = target_ground_truth.cpu().numpy().astype(dtype=np.int32)
            canvas = add_bounding_boxes(bounding_boxes=target_array,
                                        image=canvas,
                                        input_image_size=input_image_size,
                                        color=box_color)

            if label.item() == 0.0:
                cv2.putText(img=canvas, text="This Anchor Is A Hard Negative [Considered For Classification]",
                            org=(20, input_image_size - 60),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=box_color, thickness=1)
            elif label.item() == -1.0:
                cv2.putText(img=canvas, text="This Anchor Is An Easy Negative [Not Considered For Classification]",
                            org=(20, input_image_size - 60),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=box_color, thickness=1)

            # Overlay the IoU value as white text near the top-left of the anchor
            anchor_x_min, anchor_y_min, _, _ = map(int, anchor.tolist())
            cv2.putText(img=canvas, text=f"IoU: {iou_value:.9f}", org=(anchor_x_min, max(anchor_y_min - 10, 20)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2)

            # Draw a legend on the top right corner of the canvas for clarity
            canvas = add_legend(image=canvas, input_image_size=input_image_size)

            # Add instructions for the interactive loop
            cv2.putText(img=canvas, text="Press Spacebar to continue, Enter to skip remaining anchors",
                        org=(20, input_image_size - 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255), thickness=1)

            # Display the visualization and wait for specific key presses
            cv2.imshow(winname="Anchor and Target Visualization", mat=canvas)

            key = cv2.waitKey(delay=0) & 0xFF
            while key not in [13, 32]:  # 13 is Enter, 32 is Spacebar
                key = cv2.waitKey(delay=0) & 0xFF

            if key == 13:
                break

    # Destroy all OpenCV windows after the visualization loop completes
    cv2.destroyAllWindows()
