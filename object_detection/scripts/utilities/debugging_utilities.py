import os

# Suppress annoying Qt font warnings from OpenCV's imshow backend
os.environ['QT_LOGGING_RULES'] = '*=false'
import cv2
import torch
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

from utilities.model.model_utilities import get_area, get_intersection_over_union, add_bounding_boxes
from utilities.os_utilities import print_green, print_red, print_blue
from utilities.tensor_utilities import print_tensor_list, print_tensor_shape


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

        # Determine safe center coordinates to prevent the box from completely leaving the image frame
        minimum_x = width / 2.0
        maximum_x = max(minimum_x, input_image_size - width / 2.0)
        center_x = np.random.uniform(low=minimum_x, high=maximum_x)

        minimum_y = height / 2.0
        maximum_y = max(minimum_y, input_image_size - height / 2.0)
        center_y = np.random.uniform(low=minimum_y, high=maximum_y)

        # Convert the center coordinates back to corner format [x_min, y_min, x_max, y_max]
        x_min = center_x - (width / 2.0)
        y_min = center_y - (height / 2.0)
        x_max = center_x + (width / 2.0)
        y_max = center_y + (height / 2.0)

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
                type_name = "Ign." if label.item() == -1 else "Neg."

                # Recompute IoU against all true ground truth boxes
                # to find the one that caused this negative/ignored assignment
                iou_matrix = get_intersection_over_union(boxes_1=anchor.unsqueeze(dim=0),
                                                         boxes_2=original_ground_truth_boxes)
                max_iou_values, max_iou_indices = torch.max(input=iou_matrix, dim=1)
                best_match_index = max_iou_indices[0].item()
                target_ground_truth = original_ground_truth_boxes[best_match_index]
                iou_value = max_iou_values[0].item()
                print_red(f"{type_name} Anchor IOU Value :: {iou_value:.4f}", indent=1)

            else:
                # For positive anchors, use the assigned target since it might have
                # bypassed max IoU due to fallback rules
                iou_matrix = get_intersection_over_union(boxes_1=anchor.unsqueeze(dim=0),
                                                         boxes_2=target_ground_truth.unsqueeze(dim=0))
                iou_value = iou_matrix[0, 0].item()
                print_green(f"Pos. Anchor IOU Value :: {iou_value:.4f}", indent=1)

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
            cv2.putText(img=canvas, text=f"IoU: {iou_value:.4f}", org=(anchor_x_min, max(anchor_y_min - 10, 20)),
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


def visualize_foreground_and_background_anchors(ground_truth_boxes: torch.Tensor,
                                                anchors: torch.Tensor,
                                                background_iou_threshold: Dict[str, float],
                                                foreground_iou_threshold: Dict[str, float],
                                                input_image_size: int) -> None:
    """
    Visualizes the foreground and background anchors assigned to each individual ground truth bounding box.
    
    This function acts as a debugging step for the Region Proposal Network's assignment logic. By isolating
    each ground truth box and displaying two side-by-side images (one for positive matches, one for negative
    matches), it helps developers visually verify whether the Intersection over Union (IoU) thresholds are
    appropriately capturing objects of varying scales and aspect ratios.
    
    Args:
        ground_truth_boxes (torch.Tensor): A tensor of ground truth boxes in [x_min, y_min, x_max, y_max] format.
        anchors (torch.Tensor): A tensor of base anchors generated for the current feature map.
        background_iou_threshold (Dict[str, float]): Dictionary defining 'min' and 'max' IoU thresholds for background.
        foreground_iou_threshold (Dict[str, float]): Dictionary defining 'min' and 'max' IoU thresholds for foreground.
        input_image_size (int): The spatial dimension (height and width) of the square input image canvas.
        
    Returns:
        None
    """
    # Compute the IoU matrix between the single image's ground truth boxes and all anchors.
    # Shape: [number_of_ground_truths, number_of_anchors]
    intersection_over_union_matrix = get_intersection_over_union(boxes_1=ground_truth_boxes, boxes_2=anchors)

    number_of_ground_truths = ground_truth_boxes.shape[0]

    for ground_truth_index in range(number_of_ground_truths):
        ground_truth_box = ground_truth_boxes[ground_truth_index]

        # Extract the IoU values specifically for this ground truth box across all anchors
        iou_values_for_current_box = intersection_over_union_matrix[ground_truth_index]

        # Determine which anchors meet the threshold criteria for this specific ground truth box
        foreground_indices = torch.where(condition=(iou_values_for_current_box >= foreground_iou_threshold['min']) &
                                                   (iou_values_for_current_box <= foreground_iou_threshold['max']))[0]
        background_indices = torch.where(condition=(iou_values_for_current_box >= background_iou_threshold['min']) &
                                                   (iou_values_for_current_box < background_iou_threshold['max']))[0]

        # Retrieve the actual anchor coordinates
        foreground_anchors = anchors[foreground_indices]
        background_anchors = anchors[background_indices]

        # Create separate empty black canvases for the positive and negative visualizations
        positive_canvas = np.zeros(shape=(input_image_size, input_image_size, 3), dtype=np.uint8)
        negative_canvas = np.zeros(shape=(input_image_size, input_image_size, 3), dtype=np.uint8)

        # Draw the target ground truth bounding box in blue on both canvases
        ground_truth_array = ground_truth_box.cpu().numpy().astype(dtype=np.int32)
        positive_canvas = add_bounding_boxes(bounding_boxes=ground_truth_array,
                                             image=positive_canvas,
                                             input_image_size=input_image_size,
                                             color=(255, 0, 0))
        negative_canvas = add_bounding_boxes(bounding_boxes=ground_truth_array,
                                             image=negative_canvas,
                                             input_image_size=input_image_size,
                                             color=(255, 0, 0))

        # Draw the positively matched anchors in green
        if foreground_anchors.shape[0] > 0:
            foreground_anchors_array = foreground_anchors.cpu().numpy().astype(dtype=np.int32)
            positive_canvas = add_bounding_boxes(bounding_boxes=foreground_anchors_array,
                                                 image=positive_canvas,
                                                 input_image_size=input_image_size,
                                                 color=(0, 255, 0))

        # Draw the negatively matched anchors in red
        if background_anchors.shape[0] > 0:
            background_anchors_array = background_anchors.cpu().numpy().astype(dtype=np.int32)
            negative_canvas = add_bounding_boxes(bounding_boxes=background_anchors_array,
                                                 image=negative_canvas,
                                                 input_image_size=input_image_size,
                                                 color=(0, 0, 255))

        # Add descriptive text to the top of both canvases including counts and thresholds
        number_of_positive_anchors = foreground_anchors.shape[0]
        number_of_negative_anchors = background_anchors.shape[0]

        cv2.putText(img=positive_canvas,
                    text=f"Ground Truth {ground_truth_index}: Positive Anchors ({number_of_positive_anchors})",
                    org=(20, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2)
        cv2.putText(img=positive_canvas,
                    text=f"IoU Threshold: {foreground_iou_threshold['min']} to {foreground_iou_threshold['max']}",
                    org=(20, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1)

        cv2.putText(img=negative_canvas,
                    text=f"Ground Truth {ground_truth_index}: Negative Anchors ({number_of_negative_anchors})",
                    org=(20, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2)
        cv2.putText(img=negative_canvas,
                    text=f"IoU Threshold: {background_iou_threshold['min']} to {background_iou_threshold['max']}",
                    org=(20, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1)

        # Add instructions for the interactive loop
        cv2.putText(img=positive_canvas, text="Press Spacebar for next box, Enter to quit",
                    org=(20, input_image_size - 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                    color=(255, 255, 255), thickness=1)

        # Display the visualizations
        cv2.imshow(winname="Positive Anchors", mat=positive_canvas)
        cv2.imshow(winname="Negative Anchors", mat=negative_canvas)

        # Wait for user interaction
        key = cv2.waitKey(delay=0) & 0xFF
        while key not in [13, 32]:  # 13 is Enter, 32 is Spacebar
            key = cv2.waitKey(delay=0) & 0xFF

        if key == 13:
            break

    cv2.destroyAllWindows()


def get_model_configuration() -> Tuple[Dict[str, Any], torch.device, torch.dtype]:
    """
    Supplies the default configuration dictionary, device, and dtype for testing the object detection Model pipeline.
    
    Returns:
        Tuple[Dict[str, Any], torch.device, torch.dtype]: The FPN F-RCNN configuration mapping, compute device, and tensor dtype.
    """
    configuration = {'Data': {'image_settings': {'size': 1024}},
                     'Backbone': {
                         'input_channels': 3,
                         'convolutions': {'1': [2, 16],
                                          '2': [2, 32],
                                          '3': [2, 32],
                                          '4': [2, 64],
                                          '5': [2, 64],
                                          '6': [2, 64],
                                          '7': [2, 64],
                                          '8': [2, 64],
                                          '9': [2, 64],
                                          },
                         'modules': {},
                         'last_max_pooling': True,
                         'normalization': {'feature_map_normalization': 'none'},
                         'detection_convolution_block_indices': [5, 6, 7]},
                     'Anchors': {
                         'scales_and_ratios': {
                             '5': {'scales': [64, 128], 'aspect_ratios': [0.5, 1.0, 2.0]},
                             '6': {'scales': [64, 128], 'aspect_ratios': [0.5, 1.0, 2.0]},
                             '7': {'scales': [64, 128], 'aspect_ratios': [0.5, 1.0, 2.0]}
                         }},
                     'RegionProposal': {
                         'nms_iou_threshold': 0.7,
                         'training': {'pre_nms_proposals': 500, 'post_nms_proposals': 250},
                         'inference': {'pre_nms_proposals': 100, 'post_nms_proposals': 50},
                         'foreground_iou_threshold': {"min": 0.35, "max": 1.0},
                         'background_iou_threshold': {"min": 0.2, "max": 0.3},
                         'strict_fallback_assignment': False,
                         'number_training_positives': 128,
                         'total_training_samples': 256,
                         'localisation_loss_beta': 1 / 9
                     },
                     'Detector': {
                         'head_configurations': {
                             '5': {'roi_align_pool_size': 7, 'convolutions': [2, 64], 'fully_connected': [256, 128]},
                             '6': {'roi_align_pool_size': 7, 'convolutions': [2, 64], 'fully_connected': [256, 128]},
                             '7': {'roi_align_pool_size': 7, 'convolutions': [2, 64], 'fully_connected': [256, 128]}
                         },
                         'normalization': {'feature_map_normalization': 'batch'},
                         'foreground_iou_threshold': 0.3,
                         'number_training_positives': 64,
                         'total_training_samples': 92
                     }}
    device = torch.device(device="cpu")
    dtype = torch.float32

    return configuration, device, dtype


def get_dummy_input_tensor(device: torch.device, dtype: torch.dtype) -> Tuple[int, int, int, torch.Tensor]:
    """
    Generates a dummy batched input image tensor to feed the object detection model during debugging.
    
    Args:
        device (torch.device): The device (CPU/GPU) to allocate the tensor on.
        dtype (torch.dtype): The specific precision type for the tensor.
        
    Returns:
        Tuple[int, int, int, torch.Tensor]: A tuple containing the batch size, channel dimension, 
                                            spatial resolution, and the generated mock tensor itself.
    """
    batch_size = 2
    input_channels = 3
    input_image_size = 1024

    input_tensor = torch.randn(size=(batch_size, input_channels, input_image_size, input_image_size), dtype=dtype,
                               device=device)

    return batch_size, input_channels, input_image_size, input_tensor


def get_dummy_ground_truth_boxes(batch_size: int, input_image_size: int, configuration: Dict[str, Any],
                                 device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
    """
    Generates a realistic list of randomly positioned ground truth bounding boxes for each image in the batch.

    It extracts the unique bounding box scales and aspect ratios directly from the provided model configuration
    to assure the randomly generated boxes closely resemble objects the Region Proposal Network expects.

    Args:
        batch_size (int): Number of images in the batch to generate boxes for.
        input_image_size (int): The maximum spatial resolution constraint for the random boxes.
        configuration (Dict[str, Any]): The master model configuration dict to dynamically extract scales and ratios.
        device (torch.device): The compute device allocation.
        dtype (torch.dtype): The specific precision type allocation.

    Returns:
        List[torch.Tensor]: A list containing the generated ground truth bounding boxes for each image in the batch.
    """
    # Dynamically retrieve a unique list of all scales and ratios from the FPN anchors configuration
    scales_and_ratios = configuration.get("Anchors", {}).get("scales_and_ratios", {})

    all_scales = set()
    all_aspect_ratios = set()
    for scale_level, config in scales_and_ratios.items():
        all_scales.update(config.get("scales", []))
        all_aspect_ratios.update(config.get("aspect_ratios", []))

    scales = list(all_scales)
    aspect_ratios = list(all_aspect_ratios)

    ground_truth_bounding_boxes = []

    for batch_index in range(batch_size):
        # Generate a realistic random quantity of boxes (e.g., between 10 and 20 objects per image)
        random_number_of_boxes = int(np.random.randint(low=10, high=20))
        mock_boxes = generate_random_bounding_boxes(
            scales=scales, aspect_ratios=aspect_ratios,
            input_image_size=input_image_size,
            number_of_boxes=random_number_of_boxes,
            device=device, dtype=dtype)
        ground_truth_bounding_boxes.append(mock_boxes)

    return ground_truth_bounding_boxes


def get_dummy_ground_truth_labels(ground_truth_boxes: List[torch.Tensor], number_classes: int,
                                  device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
    """
    Generates a realistic list of randomly assigned ground truth class labels for each image in the batch.
    
    This function pairs with the ground truth bounding box generator to assign a valid foreground class index 
    (from 1 to number_classes - 1) to each generated box, mimicking real annotations.

    Args:
        ground_truth_boxes (List[torch.Tensor]): A list of length batch_size, containing bounding box tensors.
        number_classes (int): The total number of classes including the background class at index 0.
        device (torch.device): The compute device allocation.
        dtype (torch.dtype): The specific precision type allocation (unused for labels as they are integer indices,
        but kept for signature consistency).

    Returns:
        List[torch.Tensor]: A list of length batch_size containing 1D tensors of class labels (int64).
    """
    ground_truth_labels = []

    for boxes in ground_truth_boxes:
        number_of_boxes = boxes.shape[0]
        # Generate random labels between 1 and number_classes - 1 (inclusive), as 0 is typically background.
        random_labels = torch.randint(low=1, high=number_classes, size=(number_of_boxes,), device=device,
                                      dtype=torch.int64)
        ground_truth_labels.append(random_labels)

    return ground_truth_labels


def get_assignment_debugger_input(batch_size: int, device: torch.device, dtype: torch.dtype) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates deterministic mock data (boxes, ground truth boxes, and labels) to test target assignment logic.
    
    This function mutualizes the mock data generation for both the Region Proposal Network (anchors) 
    and the Detector (proposals) debugging scripts to ensure consistency.
    
    Args:
        batch_size (int): The number of images to simulate in the batch.
        device (torch.device): The computational device.
        dtype (torch.dtype): The tensor data type.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - reference_boxes (torch.Tensor): The mock anchors/proposals of shape [batch_size, N, 4].
            - ground_truth_boxes (torch.Tensor): The mock ground truth boxes of shape [batch_size, M, 4].
            - ground_truth_labels (torch.Tensor): The mock ground truth class labels of shape [batch_size, M].
    """
    # Define a small set of reference boxes (anchors or proposals) in [x_min, y_min, x_max, y_max] format
    reference_boxes = torch.tensor(data=[[10.0, 10.0, 50.0, 50.0],
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

    # Define a small set of ground truth bounding boxes
    ground_truth_boxes = torch.tensor(data=[[12.0, 12.0, 48.0, 48.0],
                                            [210.0, 210.0, 290.0, 290.0],
                                            [0.0, 70.0, 200.0, 120.0],
                                            [150.0, 96.0, 230.0, 120.0],
                                            [30.0, 60.0, 50.0, 150.0]], dtype=dtype, device=device)

    # Define mock ground truth labels for these boxes
    ground_truth_labels = torch.tensor(data=[1, 2, 1, 3, 2], dtype=torch.int64, device=device)

    # Expand across the batch dimension
    reference_boxes = reference_boxes.unsqueeze(dim=0).expand(size=(batch_size, -1, 4))
    ground_truth_boxes = ground_truth_boxes.unsqueeze(dim=0).expand(size=(batch_size, -1, 4))
    ground_truth_labels = ground_truth_labels.unsqueeze(dim=0).expand(size=(batch_size, -1))

    return reference_boxes, ground_truth_boxes, ground_truth_labels


def visualize_proposal_target_assignments(ground_truth_bounding_boxes: List[torch.Tensor],
                                          batched_target_ground_truth_boxes: torch.Tensor,
                                          batched_labels: torch.Tensor,
                                          batched_proposals: torch.Tensor,
                                          input_image_size: int) -> None:
    """
    Visualizes the proposal to ground truth matching logic using interactive OpenCV windows.

    This debugging utility iterates over batched assignments and provides an interactive view.
    Positive matches display both the proposal (blue) and the matched ground truth box (green),
    along with the assigned object label.
    Negative matches display the proposal (blue) and the matched ground truth box (red) that
    caused the rejection (assigned class 0).
    Users can navigate interactively using Spacebar/Enter keybindings.

    Args:
        ground_truth_bounding_boxes (List[torch.Tensor]): The raw unbatched ground truth boxes for each image.
        batched_target_ground_truth_boxes (torch.Tensor): Tensor of shape [batch_size, num_proposals, 4].
        batched_labels (torch.Tensor): Tensor of shape [batch_size, num_proposals] with assigned class labels (0=bg).
        batched_proposals (torch.Tensor): Tensor of shape [batch_size, num_proposals, 4] of proposals.
        input_image_size (int): The height/width of the input image canvas.
    """
    import cv2
    import numpy as np
    import torch
    from utilities.model.model_utilities import get_intersection_over_union, add_bounding_boxes
    from utilities.os_utilities import print_green, print_red

    batch_size = batched_proposals.shape[0]

    for batch_index in range(batch_size):
        original_ground_truth_boxes = ground_truth_bounding_boxes[batch_index]

        target_ground_truth_boxes = batched_target_ground_truth_boxes[batch_index]
        labels = batched_labels[batch_index]
        proposals = batched_proposals[batch_index]

        for proposal_index, (proposal, target_ground_truth, label) in enumerate(
                zip(proposals, target_ground_truth_boxes, labels)):

            canvas = np.zeros(shape=(input_image_size, input_image_size, 3), dtype=np.uint8)

            # Draw the proposal bounding box in blue
            proposal_array = proposal.cpu().numpy().astype(dtype=np.int32)
            canvas = add_bounding_boxes(bounding_boxes=proposal_array,
                                        image=canvas,
                                        input_image_size=input_image_size,
                                        color=(255, 0, 0))

            # Foreground is green, Background (label 0) is red
            box_color = (0, 255, 0) if label.item() > 0 else (0, 0, 255)

            iou_matrix = get_intersection_over_union(boxes_1=proposal.unsqueeze(dim=0),
                                                     boxes_2=target_ground_truth.unsqueeze(dim=0))
            iou_value = iou_matrix[0, 0].item()

            if label.item() == 0:
                print_red(f"Background Proposal IOU Value :: {iou_value:.4f}", indent=1)
                text_info = f"Assigned Label: 0 (Background) | IoU: {iou_value:.4f}"
            else:
                print_green(f"Foreground Proposal IOU Value :: {iou_value:.4f}", indent=1)
                text_info = f"Assigned Label: {label.item()} (Foreground) | IoU: {iou_value:.4f}"

            # Draw the target ground truth bounding box that the proposal was measured against
            target_array = target_ground_truth.cpu().numpy().astype(dtype=np.int32)
            canvas = add_bounding_boxes(bounding_boxes=target_array,
                                        image=canvas,
                                        input_image_size=input_image_size,
                                        color=box_color)

            # Draw text over the proposal
            proposal_x_min, proposal_y_min, _, _ = map(int, proposal.tolist())
            cv2.putText(img=canvas, text=text_info, org=(max(proposal_x_min, 10), max(proposal_y_min - 10, 20)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1)

            # Add instructions for the interactive loop
            cv2.putText(img=canvas, text="Press Spacebar to continue, Enter to skip remaining proposals",
                        org=(20, input_image_size - 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255), thickness=1)

            cv2.imshow(winname="Proposal and Target Visualization", mat=canvas)

            key = cv2.waitKey(delay=0) & 0xFF
            while key not in [13, 32]:  # 13 is Enter, 32 is Spacebar
                key = cv2.waitKey(delay=0) & 0xFF

            if key == 13:
                break

    cv2.destroyAllWindows()
