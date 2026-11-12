"""
Tests of the Anchors module of vision_training/models/detection/anchors.py: the best anchor overlap of every box found
from the nearest grid anchor of every anchor shape, read by the anchor recall of the detection trainers, must equal the
best overlap over every anchor of the level, for boxes of any size and position, the image borders included, with the
anchors centred on the cell corners or on the cell centres ('RegionProposal.anchor_centres'), whose grid lies half a
stride further, symmetric in the image, in the anchors of every level of the detector.
"""

import math
import pytest
import torch

from typing import List

import vision_training.models.detection.anchors as anchors_module
from vision_training.models.detection.anchors import Anchors
from vision_training.models.detection.box_operations import get_intersection_over_union
from tests.testing_utilities import build_test_model

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


@pytest.mark.parametrize(argnames=["feature_map_size", "scales", "aspect_ratios"],
                         argvalues=[(8, [24.0], [0.25, 3.0])])
@pytest.mark.parametrize(argnames="anchor_centres", argvalues=["cell_corners", "cell_centres"])
def test_best_overlaps_equal_the_best_of_every_anchor(feature_map_size: int, scales: List[float],
                                                       aspect_ratios: List[float], anchor_centres: str,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the best overlap of every box found from the nearest grid anchor of every shape equals the highest
    overlap with all the anchors of the level, for random boxes of every size and position, and for a level with
    several aspect ratios, the boxes being split into chunks of 999 whose last one is partial.

    Args:
        feature_map_size (int): The side of the feature map of the level.
        scales (List[float]): The anchor sizes of the level, in pixels.
        aspect_ratios (List[float]): The height / width ratios of the anchors of the level.
        anchor_centres (str): The anchors centred on the cell corners or on the cell centres.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture lowering the chunk size.
    """
    monkeypatch.setattr(target=anchors_module, name="BEST_OVERLAP_CHUNK_SIZE", value=999)
    anchor_generator = Anchors(scales=scales, aspect_ratios=aspect_ratios, input_image_size=256,
                               feature_map_size=feature_map_size, anchor_centres=anchor_centres, dtype=torch.float32,
                               device=torch.device("cpu"))
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
                               anchor_centres="cell_corners", dtype=torch.float32, device=torch.device("cpu"))

    # Anchor centred on (32, 64) spans [16, 48] x [48, 80]
    boxes = torch.tensor(data=[[16.0, 48.0, 48.0, 80.0], [24.0, 48.0, 40.0, 80.0], [500.0, 500.0, 510.0, 510.0]])
    best_overlaps = anchor_generator.compute_best_overlaps(boxes=boxes)
    assert torch.allclose(input=best_overlaps, other=torch.tensor(data=[1.0, 0.5, 0.0]))
    assert anchor_generator.compute_best_overlaps(boxes=torch.zeros(size=(0, 4))).shape == (0,)


def test_centred_anchors_lie_half_a_stride_from_the_corner_anchors() -> None:
    """
    Verifies the grid of the anchors centred on the cell centres: every anchor is the anchor of the same cell and shape
    centred on the cell corner moved by half a stride on both axes, the first centre lies half a stride from the top
    left corner of the image and the last one half a stride from its bottom right corner, the grid being symmetric in
    the image, and an unknown placement is refused.
    """
    anchor_generators = {anchor_centres: Anchors(scales=[16.0, 32.0], aspect_ratios=[0.5, 1.0, 2.0],
                                                 input_image_size=128, feature_map_size=4,
                                                 anchor_centres=anchor_centres, dtype=torch.float32,
                                                 device=torch.device("cpu"))
                         for anchor_centres in ["cell_corners", "cell_centres"]}
    corner_anchors = anchor_generators["cell_corners"].anchors
    centred_anchors = anchor_generators["cell_centres"].anchors

    # Same shapes in the same order, moved by half the stride of 32 pixels
    assert torch.equal(input=centred_anchors, other=corner_anchors + 16.0)

    # Anchor centres from 16 to 112 on both axes, symmetric around the image centre
    centre_points = (centred_anchors[:, :2] + centred_anchors[:, 2:]) / 2
    assert centre_points.min().item() == 16.0 and centre_points.max().item() == 112.0
    assert torch.allclose(input=centre_points.mean(dim=0), other=torch.tensor(data=[64.0, 64.0]))

    # A placement the code does not build is refused
    with pytest.raises(expected_exception=ValueError, match="Unknown anchor centres 'cell_edges'"):
        Anchors(scales=[16.0], aspect_ratios=[1.0], input_image_size=128, feature_map_size=4,
                anchor_centres="cell_edges", dtype=torch.float32, device=torch.device("cpu"))


def test_best_overlaps_of_known_boxes_with_centred_anchors() -> None:
    """
    Verifies the best overlap of boxes whose answer is known with the anchors centred on the cell centres: the box of a
    centred anchor overlaps it fully, the box of the corner anchor of the same cell, half a stride away on both axes,
    overlaps its four nearest centred anchors by a seventh each (16 x 16 of two 32 x 32 boxes, 256 / (2 x 1024 - 256)),
    and a box at the bottom right corner of the image, beyond the last corner anchor, is matched fully by the last
    centred anchor.
    """
    anchor_generator = Anchors(scales=[32.0], aspect_ratios=[1.0], input_image_size=128, feature_map_size=4,
                               anchor_centres="cell_centres", dtype=torch.float32, device=torch.device("cpu"))

    # Anchor of the cell (1, 2) centred on (48, 80) spans [32, 64] x [64, 96], its corner anchor [16, 48] x [48, 80]
    boxes = torch.tensor(data=[[32.0, 64.0, 64.0, 96.0], [16.0, 48.0, 48.0, 80.0], [96.0, 96.0, 128.0, 128.0]])
    best_overlaps = anchor_generator.compute_best_overlaps(boxes=boxes)
    assert torch.allclose(input=best_overlaps, other=torch.tensor(data=[1.0, 256.0 / 1792.0, 1.0]))


def test_detector_anchors_follow_the_anchor_centres() -> None:
    """
    Verifies that the setting reaches the anchors of every level of the detector: with "cell_centres", the anchors of
    every level are those of "cell_corners" moved by half the stride of the level, and the anchors of the whole
    detector are those of its levels in their order.
    """
    detectors = {anchor_centres: build_test_model(dtype=torch.float32, anchor_centres=anchor_centres)
                 for anchor_centres in ["cell_corners", "cell_centres"]}

    # Every level moved by half its own stride
    for level_name, region_proposer in detectors["cell_centres"].region_proposer_dictionary.items():
        anchor_generator = region_proposer.region_proposal_anchor_object
        corner_proposer = detectors["cell_corners"].region_proposer_dictionary[level_name]
        corner_generator = corner_proposer.region_proposal_anchor_object
        stride = anchor_generator.input_image_size // anchor_generator.feature_map_size
        assert anchor_generator.anchor_centres == "cell_centres"
        assert torch.equal(input=anchor_generator.anchors, other=corner_generator.anchors + stride / 2)

    # Anchors of the whole detector, those of the levels in their order
    level_anchors = [region_proposer.region_proposal_anchor_object.anchors
                     for region_proposer in detectors["cell_centres"].region_proposer_dictionary.values()]
    assert torch.equal(input=detectors["cell_centres"].anchors, other=torch.cat(tensors=level_anchors, dim=0))
