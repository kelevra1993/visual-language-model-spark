import torch
import numpy as np
from typing import List, Optional

from utilities.model.model_utilities import get_area


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
