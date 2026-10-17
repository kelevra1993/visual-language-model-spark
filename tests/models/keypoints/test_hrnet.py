"""
Tests of the high resolution network of vision_training/models/keypoints/hrnet.py, the body of the keypoint detector on
the 1/4 resolution stem feature map: for 2 and 3 stages, with and without the Squeeze-and-Excitation and residual
modules and with batch or layer normalization, the branches of every stage have twice the width and half the
resolution of the previous one, the network returns the highest resolution branch at the stem resolution with the base
width, the very last module only computes that branch, and every parameter receives a gradient from it. The stem
sides are divisible by the stride of the lowest resolution branch, as keypoint_checks.py requires of the crop.
"""

import pytest
import torch

from typing import Any, Dict, List

from vision_training.models.keypoints.hrnet import HighResolutionNetwork

# Channels of the stem feature map of the tests
STEM_CHANNELS = 16


def build_network(modules_per_stage: List[int], modules: Dict[str, Any],
                  feature_map_normalization: str) -> HighResolutionNetwork:
    """
    Builds a high resolution network of base width 8 on the CPU in float32, as the keypoint detector builds it from
    the 'HRNet' section.

    Args:
        modules_per_stage (List[int]): The number of exchange modules of every stage.
        modules (Dict[str, Any]): The 'squeeze_and_excite' and 'residual_nets' flags of the branch convolutions.
        feature_map_normalization (str): The normalization of every convolution, "batch" or "layer".

    Returns:
        HighResolutionNetwork: The network.
    """
    return HighResolutionNetwork(input_channels=STEM_CHANNELS, base_width=8, modules_per_stage=modules_per_stage,
                                 blocks_per_branch=1, modules=modules,
                                 normalization={"feature_map_normalization": feature_map_normalization,
                                                "batch_normalization_momentum": 0.1},
                                 device=torch.device("cpu"), dtype=torch.float32)


@pytest.mark.parametrize("modules_per_stage, stem_height, stem_width", [([1, 1], 16, 12), ([2, 1, 1], 32, 24)])
@pytest.mark.parametrize("squeeze_and_excite, residual_nets, feature_map_normalization",
                         [(True, True, "batch"), (False, False, "batch"), (True, False, "layer")])
def test_branches_halve_the_resolution_and_double_the_width(modules_per_stage: List[int], stem_height: int,
                                                             stem_width: int, squeeze_and_excite: bool,
                                                             residual_nets: bool,
                                                             feature_map_normalization: str) -> None:
    """
    Verifies that the stage s of the network has s + 2 branches, branch b with 8 x 2^b channels at 1/2^b of the stem
    resolution, that the output is the highest resolution branch with 8 channels at the stem resolution, that the last
    module only computes that branch, and that every parameter gets a gradient from it.

    Args:
        modules_per_stage (List[int]): The number of exchange modules of every stage.
        stem_height (int): The height of the stem feature map, a quarter of a 64 or 128 pixel high crop.
        stem_width (int): The width of the stem feature map, a quarter of a 48 or 96 pixel wide crop.
        squeeze_and_excite (bool): Whether the branch convolutions use Squeeze-and-Excitation.
        residual_nets (bool): Whether the branch convolutions are residual pairs.
        feature_map_normalization (str): The normalization of every convolution.
    """
    torch.manual_seed(seed=0)
    network = build_network(modules_per_stage=modules_per_stage,
                            modules={"squeeze_and_excite": squeeze_and_excite, "residual_nets": residual_nets},
                            feature_map_normalization=feature_map_normalization)
    stem_features = torch.randn(size=(2, STEM_CHANNELS, stem_height, stem_width))

    # Branch b of every stage at 8 x 2^b channels and 1/2^b of the stem resolution
    stage_shapes = network.trace_branch_shapes(input_tensor=stem_features)
    assert len(stage_shapes) == len(modules_per_stage)
    for stage_index, branch_shapes in enumerate(stage_shapes):
        assert branch_shapes == [[2, 8 * 2 ** branch_index, stem_height // 2 ** branch_index,
                                  stem_width // 2 ** branch_index] for branch_index in range(stage_index + 2)]

    # The output is the highest resolution branch, the only one computed by the very last module
    output = network(input_tensor=stem_features)
    assert output.shape == (2, 8, stem_height, stem_width) and bool(torch.isfinite(output).all())
    assert len(network.stages[-1][-1].exchange_layers) == 1
    assert all(len(exchange_module.exchange_layers) == len(exchange_module.branches)
               for stage_modules in network.stages for exchange_module in stage_modules
               if exchange_module is not network.stages[-1][-1])

    # Every branch feeds the output, so every parameter receives a gradient
    output.square().mean().backward()
    missing_gradients = [name for name, parameter in network.named_parameters() if parameter.grad is None]
    assert missing_gradients == []
