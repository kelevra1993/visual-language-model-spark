"""
Object sizes of the object detection, instance segmentation and keypoint detection evaluations and of the anchor and
proposal recalls: five sizes on the side of an object (the square root of its area) as a share of the side of its
image (the square root of the image area), each size doubling the side of the previous one like the anchor sizes from
one detection level to the next, so that an object keeps its size whatever the input size of the network. The image is
the network input (the size the detection levels see), or the original image for the evaluations made at the original
resolution, the objects being measured by their box area or, for the masks of the instance segmentation, by their mask
area. Used by the average precision and recall by size (mean_average_precision.py, average_precision_report.py) and by
the recalls of the anchors and of the proposals (region_recall.py).
"""

import math
import torch

from typing import Optional, Union

# Range of every object size on the side of the object as a share of the side of the image, the lower bound included
# and the upper excluded, the sides doubling from one size to the next
OBJECT_SIZE_SIDE_SHARES = {"under_1_16": (0.0, 1.0 / 16.0), "1_16_to_1_8": (1.0 / 16.0, 1.0 / 8.0),
                           "1_8_to_1_4": (1.0 / 8.0, 1.0 / 4.0), "1_4_to_1_2": (1.0 / 4.0, 1.0 / 2.0),
                           "1_2_and_above": (1.0 / 2.0, math.inf)}

# Displayed name of every object size, its range of side shares
OBJECT_SIZE_NAMES = {"under_1_16": "Under 1/16", "1_16_to_1_8": "1/16 to 1/8", "1_8_to_1_4": "1/8 to 1/4",
                     "1_4_to_1_2": "1/4 to 1/2", "1_2_and_above": "1/2 and above"}


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


def compute_side_shares(areas: torch.Tensor, image_areas: Union[torch.Tensor, float]) -> torch.Tensor:
    """
    Converts object areas into the side of every object as a share of the side of its image, the measure the object
    sizes are defined on, so that the same object falls in the same size at any network input size.

    Args:
        areas (torch.Tensor): The areas of the objects, in squared pixels of their image.
        image_areas (Union[torch.Tensor, float]): The area of the image of every object (one value for all the objects
            of a network input), in the same squared pixels.

    Returns:
        torch.Tensor: The side shares, 0 for an object without area.

    Raises:
        ValueError: If an image area is not positive.
    """
    image_areas = torch.as_tensor(data=image_areas, dtype=areas.dtype, device=areas.device)
    if bool((image_areas <= 0).any()):
        raise ValueError(f"Expected positive image areas to measure the object sizes, got a minimum of "
                         f"{float(image_areas.min())}.")
    return torch.sqrt(areas.clamp(min=0.0) / image_areas)


def is_in_object_size(side_shares: torch.Tensor, object_size: str) -> torch.Tensor:
    """
    Tells which objects belong to an object size, for the average precision of one size (the objects of the other sizes
    being ignored) and for the recalls by size.

    Args:
        side_shares (torch.Tensor): The side of every object as a share of the side of its image (compute_side_shares).
        object_size (str): One of OBJECT_SIZE_SIDE_SHARES, e.g. "1_16_to_1_8".

    Returns:
        torch.Tensor: The booleans, True for the objects within the range of the size.

    Raises:
        ValueError: If the object size is not one of OBJECT_SIZE_SIDE_SHARES.
    """
    if object_size not in OBJECT_SIZE_SIDE_SHARES:
        raise ValueError(f"Unknown object size {object_size!r}, expected one of {list(OBJECT_SIZE_SIDE_SHARES)}.")
    lower_bound, upper_bound = OBJECT_SIZE_SIDE_SHARES[object_size]
    return (side_shares >= lower_bound) & (side_shares < upper_bound)


def format_size_range(lower_bound: float, upper_bound: float, unit: str) -> str:
    """
    Writes a range of an object size for the tables of the reports, an open lower or upper end written with its sign.

    Args:
        lower_bound (float): The lower bound, 0 for a range open below.
        upper_bound (float): The upper bound, infinite for a range open above.
        unit (str): The unit written after the numbers, e.g. "" for pixels or " %" for area shares.

    Returns:
        str: The range, e.g. "< 32", "32 to 64" or ">= 256".
    """
    if lower_bound == 0.0:
        return f"< {upper_bound:g}{unit}"
    if math.isinf(upper_bound):
        return f">= {lower_bound:g}{unit}"
    return f"{lower_bound:g} to {upper_bound:g}{unit}"


def describe_object_size_pixels(object_size: str, image_side: float) -> str:
    """
    Writes the side of the objects of a size in pixels of an image of a given side, so that the anchor recall and the
    average precision tables tell at a glance which objects a size holds at the network input of the run.

    Args:
        object_size (str): One of OBJECT_SIZE_SIDE_SHARES.
        image_side (float): The side of the image, the square root of its area, e.g. 512 for a 512 x 512 input.

    Returns:
        str: The range of sides in pixels, e.g. "32 to 64".
    """
    lower_share, upper_share = OBJECT_SIZE_SIDE_SHARES[object_size]
    return format_size_range(lower_bound=round(lower_share * image_side, 1),
                             upper_bound=upper_share * image_side if math.isinf(upper_share) else round(
                                 upper_share * image_side, 1), unit="")


def describe_object_size_side(area_name: str, resolution_name: str, image_side: Optional[float]) -> str:
    """
    Writes how the object sizes are measured, for the legends of the anchor recall, proposal recall and average
    precision tables.

    Args:
        area_name (str): The area measuring the objects, "box" or "mask".
        resolution_name (str): The image the objects are measured in, "network input" or "original image".
        image_side (Optional[float]): The side of the image in pixels when all the images share it (a network input),
            None when every image has its own (the original images).

    Returns:
        str: The sentences of the legend, e.g. "Object side: the square root of the box area as a share of the side of
            the network input (512 pixels); Pixels: that side at this input."
    """
    if image_side is None:
        return (f"Object side: the square root of the {area_name} area as a share of the side of the {resolution_name} "
                f"(the square root of its area).")
    return (f"Object side: the square root of the {area_name} area as a share of the side of the {resolution_name} "
            f"({image_side:g} pixels); Pixels: that side at this input.")


def describe_object_size_area_shares(object_size: str) -> str:
    """
    Writes the share of the image area the objects of a size cover, the square of their side shares, for the tables
    of the anchor recall and of the average precision.

    Args:
        object_size (str): One of OBJECT_SIZE_SIDE_SHARES.

    Returns:
        str: The range of area shares in percent, e.g. "0.39 to 1.56 %".
    """
    lower_share, upper_share = OBJECT_SIZE_SIDE_SHARES[object_size]
    return format_size_range(lower_bound=round(100.0 * lower_share ** 2, 2),
                             upper_bound=100.0 * upper_share ** 2 if math.isinf(upper_share) else round(
                                 100.0 * upper_share ** 2, 2), unit=" %")
