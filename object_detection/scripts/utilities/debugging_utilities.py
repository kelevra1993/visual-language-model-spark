import torch
import numpy as np

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
        coordinates_string = (f"[{box_coordinates[0]:>6.2f},"
                              f" {box_coordinates[1]:>6.2f},"
                              f" {box_coordinates[2]:>6.2f},"
                              f" {box_coordinates[3]:>6.2f}]")
        print(f"{indentation_string}"
              f"{coordinates_string}  ::"
              f"  Area: {area_value:>9.2f}  ::"
              f"  Scale: {scale_value:>6.2f}")
