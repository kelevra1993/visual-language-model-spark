"""
Tests of the FeatureNeck of vision_training/models/detection/feature_neck.py, applied by the two-stage detector between
the backbone blocks of the detection levels and the heads: the levels and sizes are kept with the output channels, the
'none' neck is a parameter-free identity so that the runs written before the neck are unchanged, the feature pyramid
carries the coarse levels to the fine ones only, and path aggregation also carries the fine levels to the coarse ones.
"""

import pytest
import torch

from typing import Dict

from vision_training.models.components.convolution_block import BATCH_NORMALIZATION_MOMENTUM
from vision_training.models.detection.feature_neck import FeatureNeck

# Detection levels of the benchmark runs at 1024 pixels (blocks 4 to 7), with channels differing between the blocks so
# that the laterals are checked to bring every one of them to the output channels
LEVEL_INFORMATION = {"4": {"feature_map_size": 64, "number_channels": 16},
                     "5": {"feature_map_size": 32, "number_channels": 24},
                     "6": {"feature_map_size": 16, "number_channels": 32},
                     "7": {"feature_map_size": 16, "number_channels": 32}}

# Channels of every level after the neck
OUTPUT_CHANNELS = 8


def build_test_neck(neck_type: str, dtype: torch.dtype) -> FeatureNeck:
    """
    Builds the neck of the tests on LEVEL_INFORMATION with batch normalization, the normalization of the benchmark.

    Args:
        neck_type (str): 'none', 'feature_pyramid' or 'path_aggregation'.
        dtype (torch.dtype): The data type of the parameters.

    Returns:
        FeatureNeck: The neck, under a fixed seed.
    """
    torch.manual_seed(seed=0)
    return FeatureNeck(neck_type=neck_type, level_information=LEVEL_INFORMATION, output_channels=OUTPUT_CHANNELS,
                       feature_map_normalization="batch", batch_normalization_momentum=BATCH_NORMALIZATION_MOMENTUM,
                       device=torch.device("cpu"), dtype=dtype)


def draw_feature_maps(batch_size: int, dtype: torch.dtype, seed: int) -> Dict[str, torch.Tensor]:
    """
    Draws backbone feature maps of every level of LEVEL_INFORMATION, standing for the outputs of the backbone blocks.

    Args:
        batch_size (int): The number of images.
        dtype (torch.dtype): The data type of the feature maps.
        seed (int): The seed of the generator.

    Returns:
        Dict[str, torch.Tensor]: The [batch_size, C_level, S_level, S_level] feature map of every level.
    """
    generator = torch.Generator().manual_seed(seed)
    return {level_name: torch.randn(size=(batch_size, information["number_channels"], information["feature_map_size"],
                                          information["feature_map_size"]), generator=generator, dtype=dtype)
            for level_name, information in LEVEL_INFORMATION.items()}


@pytest.mark.parametrize(argnames="neck_type", argvalues=["feature_pyramid", "path_aggregation"])
def test_neck_keeps_the_levels_and_sizes_with_the_output_channels(neck_type: str) -> None:
    """
    Verifies that the feature pyramid and path aggregation give every level at the size of its block with the output
    channels, as announced by compute_output_information to the heads, with finite values.

    Args:
        neck_type (str): The neck under test.
    """
    neck = build_test_neck(neck_type=neck_type, dtype=torch.float32)
    output_maps = neck(feature_maps=draw_feature_maps(batch_size=2, dtype=torch.float32, seed=1))
    output_information = neck.compute_output_information()

    # Same levels, from the finest to the coarsest, each at the announced size and channels
    assert list(output_maps) == ["4", "5", "6", "7"]
    assert list(output_information) == ["4", "5", "6", "7"]
    for level_name, output_map in output_maps.items():
        size = LEVEL_INFORMATION[level_name]["feature_map_size"]
        assert output_information[level_name] == {"feature_map_size": size, "number_channels": OUTPUT_CHANNELS}
        assert tuple(output_map.shape) == (2, OUTPUT_CHANNELS, size, size)
        assert torch.isfinite(output_map).all()


def test_none_neck_is_a_parameter_free_identity() -> None:
    """
    Verifies that the 'none' neck has no parameter or buffer, consumes no random draw, returns the backbone feature maps
    themselves and announces the backbone sizes and channels, so that the runs written before the neck keep their
    state dictionaries, their initial weights and their results.
    """
    torch.manual_seed(seed=0)
    expected_next_draw = torch.rand(size=(1,))
    neck = build_test_neck(neck_type="none", dtype=torch.float32)
    assert torch.equal(torch.rand(size=(1,)), expected_next_draw)

    # No layer, the same tensors returned, the backbone information announced
    assert len(neck.state_dict()) == 0
    feature_maps = draw_feature_maps(batch_size=1, dtype=torch.float32, seed=2)
    output_maps = neck(feature_maps=feature_maps)
    assert all(output_maps[level_name] is feature_maps[level_name] for level_name in feature_maps)
    assert neck.compute_output_information() == LEVEL_INFORMATION


def test_neck_refuses_unknown_types_and_sizes_not_whole_multiples() -> None:
    """
    Verifies that the neck refuses an unknown type, and levels whose finer size is not a whole multiple of the next
    coarser one, which the top-down upsampling and the bottom-up stride cannot match.
    """
    with pytest.raises(expected_exception=ValueError, match="Unknown neck type"):
        build_test_neck(neck_type="bidirectional", dtype=torch.float32)
    level_information = {"3": {"feature_map_size": 24, "number_channels": 8},
                         "4": {"feature_map_size": 16, "number_channels": 8}}
    with pytest.raises(expected_exception=ValueError, match="whole multiple"):
        FeatureNeck(neck_type="feature_pyramid", level_information=level_information, output_channels=OUTPUT_CHANNELS,
                    feature_map_normalization="batch", batch_normalization_momentum=BATCH_NORMALIZATION_MOMENTUM,
                    device=torch.device("cpu"), dtype=torch.float32)


@pytest.mark.parametrize(argnames=["neck_type", "finest_reaches_coarsest"],
                         argvalues=[("feature_pyramid", False), ("path_aggregation", True)])
def test_paths_carry_information_in_their_direction(neck_type: str, finest_reaches_coarsest: bool) -> None:
    """
    Verifies the direction of the paths: changing the coarsest backbone map changes the finest output of both necks
    (top-down path), while changing the finest backbone map changes the coarsest output with path aggregation only
    (bottom-up path), the coarsest output of the feature pyramid reading its own block alone.

    Args:
        neck_type (str): The neck under test.
        finest_reaches_coarsest (bool): Whether the coarsest output depends on the finest backbone map.
    """
    neck = build_test_neck(neck_type=neck_type, dtype=torch.float64).eval()
    feature_maps = draw_feature_maps(batch_size=1, dtype=torch.float64, seed=3)
    with torch.no_grad():
        reference_maps = neck(feature_maps=feature_maps)

        # Top-down path: the coarsest block reaches the finest output
        changed_coarsest = dict(feature_maps, **{"7": feature_maps["7"] + 1.0})
        assert not torch.allclose(neck(feature_maps=changed_coarsest)["4"], reference_maps["4"])

        # Bottom-up path: the finest block reaches the coarsest output with path aggregation only
        changed_finest = dict(feature_maps, **{"4": feature_maps["4"] + 1.0})
        coarsest_changed = not torch.allclose(neck(feature_maps=changed_finest)["7"], reference_maps["7"])
    assert coarsest_changed == finest_reaches_coarsest


def test_path_aggregation_strides_and_gradients() -> None:
    """
    Verifies that the bottom-up convolutions of path aggregation stride by the ratio of the sizes of their levels (1
    between the two 16 x 16 levels), and that a loss on the outputs reaches the weights of every layer of the neck.
    """
    neck = build_test_neck(neck_type="path_aggregation", dtype=torch.float32)

    # Strides from the finer level of every bottom-up step: 64 to 32, 32 to 16, then 16 to 16
    strides = {level_name: [layer for layer in block.block if isinstance(layer, torch.nn.Conv2d)][0].stride
               for level_name, block in neck.downsampling_convolutions.items()}
    assert strides == {"5": (2, 2), "6": (2, 2), "7": (1, 1)}

    # Every weight of the laterals, the pyramid outputs and the bottom-up path receives a gradient
    output_maps = neck(feature_maps=draw_feature_maps(batch_size=2, dtype=torch.float32, seed=4))
    sum(output_map.square().mean() for output_map in output_maps.values()).backward()
    for parameter_name, parameter in neck.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, parameter_name
