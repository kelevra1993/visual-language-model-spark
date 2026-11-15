"""
Tests of scripts/models/instance_segmentation/draw_mask_head_designs.py, the figure of the mask head designs of stages
3 and 3b of the instance segmentation benchmark: the multiply-accumulates its hooks record for a convolution and a
transposed convolution called by position or by name, the multiply-accumulates of a mask branch equal to their count
by hand on the small test detector, and the figure of every design written from the configuration of that detector.
"""

import copy
import torch

from pathlib import Path
from PIL import Image
from torch import nn

from scripts.models.instance_segmentation.draw_mask_head_designs import (DESIGNS, count_operations_per_box,
                                                                         draw_mask_head_designs,
                                                                         record_multiply_accumulates)
from tests.fixtures.configuration_fixtures import get_detection_model_configuration
from tests.testing_utilities import build_test_model

# Mask heads of every level of the test detector for the count by hand: a pool of 7 doubled once by a transposed
# convolution to the 14 x 14 masks, two convolutions of 16 channels and one fusion convolution of 8 channels
COUNTED_MASK_HEAD_SETTINGS = {"MaskHead": {"mask_resolution": 14, "upsampling": "transposed",
                                           "fusion_convolutions": [1, 8]},
                              "level_mask_heads": {level: {"roi_align_pool_size": 7, "convolutions": [2, 16]}
                                                   for level in ["5", "6", "7"]}}


def test_hooks_count_convolutions_called_by_position_or_by_name() -> None:
    """
    A 3 x 3 convolution counts its kernel at every output position and a 2 x 2 transposed convolution at every input
    position, for one box whatever the number of boxes, its input being read by position or by name as the mask heads
    call their layers.
    """
    convolution = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
    transposed_convolution = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=2, stride=2)
    for layer in [convolution, transposed_convolution]:
        layer.register_forward_hook(hook=record_multiply_accumulates, with_kwargs=True)
    with torch.no_grad():
        features = convolution(torch.zeros(size=(2, 3, 10, 10)))
        transposed_convolution(input=features)
    assert convolution.recorded_multiply_accumulates == 10 * 10 * 8 * 3 * 9
    assert transposed_convolution.recorded_multiply_accumulates == 10 * 10 * 8 * 4 * 4


def test_mask_branch_operations_match_their_count_by_hand() -> None:
    """
    The multiply-accumulates of the mask branch of one box are those of its layers counted by hand: the two
    convolutions of the block at the pool size, the transposed convolution of the pooled grid, the fusion convolution
    and the predictor of every class at the mask resolution.
    """
    model = build_test_model(dtype=torch.float32, instance_segmentation=True,
                             mask_head_settings=COUNTED_MASK_HEAD_SETTINGS)
    level_channels = model.detection_input_information["5"]["number_channels"]
    number_classes = model.mask_heads_dictionary["5"].number_classes
    pooled_positions, mask_positions = 7 * 7, 14 * 14
    expected_operations = (pooled_positions * level_channels * 16 * 9 + pooled_positions * 16 * 16 * 9
                           + pooled_positions * 16 * 16 * 4 + mask_positions * 16 * 8 * 9
                           + mask_positions * 8 * number_classes)
    assert count_operations_per_box(model=model) == expected_operations


def test_figure_of_every_design_is_written(tmp_path: Path) -> None:
    """
    The figure is written from the configuration of the test detector, with the description of every design read from
    its built layers: the reference without finer inputs, the image branch of design A, the skips of design B and the
    fusion convolution of every arm of stage 3b.

    Args:
        tmp_path (Path): The temporary folder of the test, holding the figure.
    """
    configuration, _, _ = get_detection_model_configuration(mask_head=True)
    configuration = copy.deepcopy(configuration)
    configuration["Data"]["image_settings"]["size"] = 512
    output_path = tmp_path / "mask_head_designs.png"
    descriptions = draw_mask_head_designs(model_configuration=configuration, output_path=output_path)

    # One description per design, read from the layers the configuration of every design builds
    assert len(descriptions) == len(DESIGNS)
    assert [description["block_channels"] for description in descriptions] == [32, 64, 64, 64, 64]
    assert descriptions[0]["fine_channels"] == 0 and descriptions[0]["fusion"] == [0, 0]
    assert descriptions[2]["image_branch"] == [5, 32] and descriptions[2]["fine_channels"] == 32
    assert [skip["block"] for skip in descriptions[3]["skips_at_pool_size"]] == [4]
    assert [skip["block"] for skip in descriptions[3]["skips_at_mask_size"]] == [3]
    assert [description["fusion"] for description in descriptions[2:]] == [[1, 64]] * 3
    assert descriptions[1]["operations"] > descriptions[0]["operations"]
    with Image.open(fp=output_path) as figure:
        assert figure.width > 1000 and figure.height > 1000
