"""
Tests of vision_training/evaluation/detection/object_sizes.py, the COCO object sizes shared by the average precision
by object size and by the anchor and proposal recall of the detection tasks: the box areas, the bounds of every size
and the refused size name.
"""

import pytest
import torch

from vision_training.evaluation.detection.object_sizes import (OBJECT_SIZE_NAMES, OBJECT_SIZE_RANGES,
                                                               compute_box_areas, is_in_object_size)


def test_box_areas_of_valid_and_degenerate_boxes() -> None:
    """
    Verifies the area of a box, and the zero area of a box whose corners are swapped instead of a negative one.
    """
    areas = compute_box_areas(boxes=torch.tensor(data=[[0.0, 0.0, 10.0, 20.0], [10.0, 0.0, 0.0, 5.0]]))
    assert torch.equal(areas, torch.tensor(data=[200.0, 0.0]))


def test_every_area_has_one_size() -> None:
    """
    Verifies that every area belongs to exactly one size, the 32² and 96² bounds belonging to the larger size, and
    that the sizes have a displayed name each.
    """
    areas = torch.tensor(data=[0.0, 1023.0, 1024.0, 9215.0, 9216.0, 1.0e6])
    size_masks = {object_size: is_in_object_size(areas=areas, object_size=object_size)
                  for object_size in OBJECT_SIZE_RANGES}
    assert size_masks["small"].tolist() == [True, True, False, False, False, False]
    assert size_masks["medium"].tolist() == [False, False, True, True, False, False]
    assert size_masks["large"].tolist() == [False, False, False, False, True, True]
    assert torch.stack(tensors=list(size_masks.values())).sum(dim=0).tolist() == [1] * 6
    assert list(OBJECT_SIZE_NAMES) == list(OBJECT_SIZE_RANGES)


def test_unknown_size_is_refused() -> None:
    """
    Verifies that a size name outside the COCO sizes is refused.
    """
    with pytest.raises(expected_exception=ValueError, match="Unknown object size"):
        is_in_object_size(areas=torch.tensor(data=[10.0]), object_size="huge")
