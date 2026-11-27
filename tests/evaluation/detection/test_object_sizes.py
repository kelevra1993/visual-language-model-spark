"""
Tests of vision_training/evaluation/detection/object_sizes.py, the five object sizes shared by the average precision
by size of the detection tasks and by the anchor and proposal recalls: the sizes covering every side share without gap,
each doubling the side of the previous one, the side shares of the objects, the lower bound of a size included and the
upper one excluded, an object keeping its size when the image and the object are scaled together, and the pixel, area
and legend texts of the tables.
"""

import math
import pytest
import torch

from vision_training.evaluation.detection.object_sizes import (OBJECT_SIZE_NAMES, OBJECT_SIZE_SIDE_SHARES,
                                                              compute_side_shares, describe_object_size_area_shares,
                                                              describe_object_size_pixels, describe_object_size_side,
                                                              is_in_object_size)


def test_sizes_double_and_cover_every_side_share() -> None:
    """
    Verifies that the sizes start at 0, end at infinity, follow each other without gap or overlap and double their side
    from one size to the next, with one displayed name per size.
    """
    ranges = list(OBJECT_SIZE_SIDE_SHARES.values())
    assert ranges[0][0] == 0.0 and math.isinf(ranges[-1][1])
    assert all(upper_bound == next_lower_bound for (_, upper_bound), (next_lower_bound, _) in zip(ranges, ranges[1:]))
    assert all(upper_bound == 2.0 * lower_bound for lower_bound, upper_bound in ranges[1:-1])
    assert list(OBJECT_SIZE_NAMES) == list(OBJECT_SIZE_SIDE_SHARES)


def test_objects_fall_in_one_size_relative_to_their_image() -> None:
    """
    Verifies that the side share is the square root of the area share, that an object exactly on a bound falls in the
    upper size, that every object falls in exactly one size, that the same object and image scaled together keep their
    size, and that an unknown size or an image without area is refused.
    """
    # Squares of sides 10, 32, 64, 100 and 256 in a 512 x 512 image: 1/51.2, exactly 1/16, exactly 1/8, 1/5.12, exactly
    # 1/2
    sides = torch.tensor(data=[10.0, 32.0, 64.0, 100.0, 256.0], dtype=torch.float64)
    side_shares = compute_side_shares(areas=sides ** 2, image_areas=512.0 * 512.0)
    assert torch.allclose(input=side_shares, other=sides / 512.0)
    memberships = torch.stack(tensors=[is_in_object_size(side_shares=side_shares, object_size=object_size)
                                       for object_size in OBJECT_SIZE_SIDE_SHARES], dim=1)
    assert memberships.long().tolist() == [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0],
                                           [0, 0, 0, 0, 1]]

    # The same objects in an image twice as large on both sides, scaled with it, keep their side shares
    scaled_side_shares = compute_side_shares(areas=(2.0 * sides) ** 2, image_areas=1024.0 * 1024.0)
    assert torch.allclose(input=scaled_side_shares, other=side_shares)

    # One image area per object, an object without area falling under 1/16
    per_object_shares = compute_side_shares(areas=torch.tensor(data=[0.0, 400.0]),
                                            image_areas=torch.tensor(data=[100.0, 1600.0]))
    assert per_object_shares.tolist() == [0.0, 0.5]
    with pytest.raises(ValueError, match="Unknown object size"):
        is_in_object_size(side_shares=side_shares, object_size="small")
    with pytest.raises(ValueError, match="positive image areas"):
        compute_side_shares(areas=sides, image_areas=0.0)


def test_size_texts_of_the_tables() -> None:
    """
    Verifies the sides in pixels of every size at a 512 pixel input, rounded to one decimal at an odd input, the share
    of the image area of every size, and the legend sentence with and without a shared image side.
    """
    assert [describe_object_size_pixels(object_size=object_size, image_side=512.0)
            for object_size in OBJECT_SIZE_SIDE_SHARES] == ["< 32", "32 to 64", "64 to 128", "128 to 256", ">= 256"]
    assert describe_object_size_pixels(object_size="1_16_to_1_8", image_side=600.0) == "37.5 to 75"
    assert [describe_object_size_area_shares(object_size=object_size) for object_size in OBJECT_SIZE_SIDE_SHARES] == [
        "< 0.39 %", "0.39 to 1.56 %", "1.56 to 6.25 %", "6.25 to 25 %", ">= 25 %"]
    assert describe_object_size_side(area_name="mask", resolution_name="network input", image_side=512.0) == (
        "Object side: the square root of the mask area as a share of the side of the network input (512 pixels); "
        "Pixels: that side at this input.")
    assert describe_object_size_side(area_name="box", resolution_name="original image", image_side=None).endswith(
        "side of the original image (the square root of its area).")
