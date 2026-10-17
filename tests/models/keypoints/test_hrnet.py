"""
Tests of vision_training/models/keypoints/hrnet.py, the body of the keypoint detector on the 1/4 resolution stem
feature map: for 2 and 3 stages, with and without the Squeeze-and-Excitation and residual modules and with batch or
layer normalization, the branches of every stage of the high resolution network have twice the width and half the
resolution of the previous one, the network returns the highest resolution branch at the stem resolution with the base
width, the very last module only computes that branch, and every parameter receives a gradient from it. The stem
sides are divisible by the stride of the lowest resolution branch, as keypoint_checks.py requires of the image. The
higher resolution branch of HigherHRNet doubles the resolution of the features and of the predictions it is given,
and reads both, with or without the 3x3 hidden convolution before its heatmaps.
"""

import pytest
import torch

from typing import Any, Dict, List

from vision_training.models.keypoints.hrnet import HigherResolutionBranch, HighResolutionNetwork

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
        stem_height (int): The height of the stem feature map, a quarter of a 64 or 128 pixel high image.
        stem_width (int): The width of the stem feature map, a quarter of a 48 or 96 pixel wide image.
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


@pytest.mark.parametrize("feature_map_normalization, hidden_width", [("batch", 0), ("layer", 0), ("batch", 16)])
def test_higher_resolution_branch_doubles_the_resolution(feature_map_normalization: str, hidden_width: int) -> None:
    """
    Verifies that the higher resolution branch turns the features and predictions at a quarter of a 64 x 48 image into
    7 heatmaps at half of it, that the heatmaps change with the predictions as well as with the features, and that
    every parameter of the branch receives a gradient, its heatmap head being a 1x1 convolution or a 3x3 convolution
    of 16 channels followed by the 1x1 convolution.

    Args:
        feature_map_normalization (str): The normalization of every convolution, "batch" or "layer".
        hidden_width (int): The channels of the 3x3 convolution of the heatmap head, 0 for none.
    """
    torch.manual_seed(seed=0)
    branch = HigherResolutionBranch(feature_channels=8, prediction_channels=14, heatmap_channels=7,
                                    hidden_width=hidden_width, blocks_per_branch=1,
                                    modules={"squeeze_and_excite": True, "residual_nets": True},
                                    normalization={"feature_map_normalization": feature_map_normalization,
                                                   "batch_normalization_momentum": 0.1},
                                    device=torch.device("cpu"), dtype=torch.float32).eval()
    features, predictions = torch.randn(size=(2, 8, 16, 12)), torch.randn(size=(2, 14, 16, 12))
    heatmaps = branch(features=features, predictions=predictions)
    assert heatmaps.shape == (2, 7, 32, 24) and bool(torch.isfinite(heatmaps).all())

    # The heatmap head ends with the 1x1 convolution, preceded by the 3x3 convolution of the hidden width if any
    assert len(branch.heatmap_head) == (2 if hidden_width > 0 else 1)
    assert branch.heatmap_head[-1].kernel_size == (1, 1) and branch.heatmap_head[-1].in_channels == (hidden_width or 8)

    # The heatmaps read the predictions and the features
    with torch.no_grad():
        assert not torch.allclose(input=branch(features=features, predictions=predictions + 1.0), other=heatmaps)
        assert not torch.allclose(input=branch(features=features + 1.0, predictions=predictions), other=heatmaps)

    # Every parameter receives a gradient
    branch.train()
    branch(features=features, predictions=predictions).square().mean().backward()
    assert [name for name, parameter in branch.named_parameters() if parameter.grad is None] == []
