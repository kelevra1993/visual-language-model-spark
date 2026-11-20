"""
Tests of the Anchors module of vision_training/models/detection/anchors.py: the best anchor overlap of every box found
from the nearest grid anchor of every anchor shape, read by the anchor recall of the detection trainers, must equal the
best overlap over every anchor of the level, for boxes of any size and position, the image borders included.
"""

import math
import pytest
import torch

from typing import List

import vision_training.models.detection.anchors as anchors_module
from vision_training.models.detection.anchors import Anchors
from vision_training.models.detection.box_operations import get_intersection_over_union

# Number of random boxes compared with every anchor of a level
NUMBER_RANDOM_BOXES = 4000


def draw_random_boxes(image_size: int, number_boxes: int, seed: int) -> torch.Tensor:
    """
    Draws boxes of the anchor recall tests: sides from 1 pixel to beyond the image (log-uniform), centres from outside
    the image to the opposite side, so that the nearest grid anchor is also clamped to the first and last grid rows.

    Args:
        image_size (int): The side of the network input, in pixels.
        number_boxes (int): The number of boxes.
        seed (int): The seed of the generator.

    Returns:
        torch.Tensor: The [number_boxes, 4] float32 [x_min, y_min, x_max, y_max] boxes.
    """
    generator = torch.Generator().manual_seed(seed)
    centres = torch.empty(size=(number_boxes, 2)).uniform_(-0.1 * image_size, 1.1 * image_size, generator=generator)

    # Width and height drawn separately, from 1 pixel to 1.5 image sides on a logarithmic scale
    maximum_log_side = math.log(1.5 * image_size)
    sides = torch.exp(input=torch.empty(size=(number_boxes, 2)).uniform_(0.0, maximum_log_side, generator=generator))
    return torch.cat(tensors=[centres - sides / 2, centres + sides / 2], dim=1)


@pytest.mark.parametrize(argnames=["feature_map_size", "scales", "aspect_ratios"], argvalues=[(8, [24.0], [0.25, 3.0])])
def test_best_overlaps_equal_the_best_of_every_anchor(feature_map_size: int, scales: List[float],
                                                       aspect_ratios: List[float],
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the best overlap of every box found from the nearest grid anchor of every shape equals the highest
    overlap with all the anchors of the level, for random boxes of every size and position, and for a level with
    several aspect ratios, the boxes being split into chunks of 999 whose last one is partial.

    Args:
        feature_map_size (int): The side of the feature map of the level.
        scales (List[float]): The anchor sizes of the level, in pixels.
        aspect_ratios (List[float]): The height / width ratios of the anchors of the level.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture lowering the chunk size.
    """
    monkeypatch.setattr(target=anchors_module, name="BEST_OVERLAP_CHUNK_SIZE", value=999)
    anchor_generator = Anchors(scales=scales, aspect_ratios=aspect_ratios, input_image_size=256,
                               feature_map_size=feature_map_size, dtype=torch.float32, device=torch.device("cpu"))
    boxes = draw_random_boxes(image_size=256, number_boxes=NUMBER_RANDOM_BOXES, seed=feature_map_size)

    # Highest overlap with every anchor of the level, compared in float64
    expected_overlaps = get_intersection_over_union(boxes_1=boxes.double(),
                                                    boxes_2=anchor_generator.anchors.double()).max(dim=1).values
    best_overlaps = anchor_generator.compute_best_overlaps(boxes=boxes)
    assert best_overlaps.dtype == torch.float32 and best_overlaps.device.type == "cpu"
    assert torch.allclose(input=best_overlaps.double(), other=expected_overlaps, atol=1e-6, rtol=0.0)


def test_best_overlaps_of_known_boxes() -> None:
    """
    Verifies the best overlap of boxes whose answer is known: a box equal to an anchor overlaps it fully, a box half the
    width of the anchor centred on it overlaps it by 0.5, a box far outside the image overlaps no anchor, and no box
    gives an empty overlap tensor.
    """
    anchor_generator = Anchors(scales=[32.0], aspect_ratios=[1.0], input_image_size=128, feature_map_size=4,
                               dtype=torch.float32, device=torch.device("cpu"))

    # Anchor centred on (32, 64) spans [16, 48] x [48, 80]
    boxes = torch.tensor(data=[[16.0, 48.0, 48.0, 80.0], [24.0, 48.0, 40.0, 80.0], [500.0, 500.0, 510.0, 510.0]])
    best_overlaps = anchor_generator.compute_best_overlaps(boxes=boxes)
    assert torch.allclose(input=best_overlaps, other=torch.tensor(data=[1.0, 0.5, 0.0]))
    assert anchor_generator.compute_best_overlaps(boxes=torch.zeros(size=(0, 4))).shape == (0,)
