"""
Object sizes of the object detection and instance segmentation evaluations, the COCO breakdown: small objects under
32 x 32 pixels, medium ones up to 96 x 96 pixels and large ones above, measured in network input pixels (the size the
detection levels see), by box area or, for the masks of the instance segmentation, by mask area. Used by the average
precision by size (mean_average_precision.py) and by the recalls of the anchors and of the proposals
(region_recall.py).
"""

import math
import torch

# Area range of every object size, in squared network input pixels, the lower bound included and the upper excluded
OBJECT_SIZE_RANGES = {"small": (0.0, 32.0 ** 2), "medium": (32.0 ** 2, 96.0 ** 2), "large": (96.0 ** 2, math.inf)}

# Displayed name of every object size, with its range
OBJECT_SIZE_NAMES = {"small": "Small (< 32²)", "medium": "Medium (32² to 96²)", "large": "Large (>= 96²)"}


def compute_box_areas(boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes the area of every box, the size of an object of the object detection and of the box average precision of
    the instance segmentation, an inverted box having no area.

    Args:
        boxes (torch.Tensor): The [..., 4] [x1, y1, x2, y2] boxes.

    Returns:
        torch.Tensor: The [...] areas, in squared pixels.
    """
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0.0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0.0)


def is_in_object_size(areas: torch.Tensor, object_size: str) -> torch.Tensor:
    """
    Tells which areas belong to an object size, for the average precision of one size (the objects of the other sizes
    being ignored) and for the recalls by size.

    Args:
        areas (torch.Tensor): The areas, in squared network input pixels.
        object_size (str): "small", "medium" or "large".

    Returns:
        torch.Tensor: The booleans, True for the areas within the range of the size.

    Raises:
        ValueError: If the object size is not one of OBJECT_SIZE_RANGES.
    """
    if object_size not in OBJECT_SIZE_RANGES:
        raise ValueError(f"Unknown object size {object_size!r}, expected one of {list(OBJECT_SIZE_RANGES)}.")
    lower_bound, upper_bound = OBJECT_SIZE_RANGES[object_size]
    return (areas >= lower_bound) & (areas < upper_bound)
