"""
High resolution network (HRNet) body of the keypoint detector: from the 1/4 resolution feature map of the Backbone
stem, it keeps a full resolution branch through the whole network and adds a branch at half the resolution and twice
the width at every stage (1/4, 1/8, 1/16, 1/32), the branches exchanging their features at the end of every module.
Its branches and exchange units are built from the ConvolutionBlock shared with the Backbone (residual pairs,
Squeeze-and-Excitation, batch or layer normalization). The highest resolution branch feeds the 1x1 heatmap head of
models/keypoints/keypoint_detector.py.
"""

import torch

from torch import nn
from typing import Any, Dict, List

from vision_training.models.components.convolution_block import ConvolutionBlock

# Ratio between the resolutions, and between the widths, of two consecutive branches
BRANCH_SCALE_FACTOR = 2


def build_exchange_convolution(input_channels: int, output_channels: int, kernel_size: int, stride: int,
                               activation: bool, normalization: Dict[str, Any], device: torch.device,
                               dtype: torch.dtype) -> ConvolutionBlock:
    """
    Builds one plain convolution layer of the transitions and exchange units of the high resolution network
    (convolution, normalization, optional activation), the 3x3 strided ones halving the resolution of a branch and the
    1x1 ones changing its width before an upsampling.

    Args:
        input_channels (int): The number of channels of the input feature map.
        output_channels (int): The number of channels of the output feature map.
        kernel_size (int): The size of the square kernel, 1 or 3.
        stride (int): The stride of the convolution, 2 to halve the resolution.
        activation (bool): Whether to apply a ReLU after the normalization.
        normalization (Dict[str, Any]): The 'feature_map_normalization' and 'batch_normalization_momentum' settings.
        device (torch.device): The device of the parameters.
        dtype (torch.dtype): The precision of the parameters.

    Returns:
        ConvolutionBlock: The single layer block, without pooling.
    """
    return ConvolutionBlock(input_channels=input_channels, output_channels=output_channels, bias=False, number_layers=1,
                            kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                            feature_map_normalization=normalization["feature_map_normalization"],
                            batch_normalization_momentum=normalization["batch_normalization_momentum"],
                            activation=activation, dropout_rate=0.0, add_pooling=False, device=device, dtype=dtype,
                            squeeze_and_excite=False, residual=False, pre_activation=False, inception=False)


class BranchTransition(nn.Module):
    """
    Transition between two stages of the high resolution network: the branches of the previous stage are kept (or
    brought to the width of the new stage by a 3x3 convolution), and the new lower resolution branch is computed from
    the lowest resolution branch of the previous stage by 3x3 convolutions of stride 2, one per halving.
    """

    def __init__(self, input_widths: List[int], output_widths: List[int], normalization: Dict[str, Any],
                 device: torch.device, dtype: torch.dtype) -> None:
        """
        Builds the layers of every output branch of the transition.

        Args:
            input_widths (List[int]): The channels of the branches of the previous stage (the stem alone for the first
                transition), from the highest resolution.
            output_widths (List[int]): The channels of the branches of the new stage, one more than the input.
            normalization (Dict[str, Any]): The normalization settings of the convolutions.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(BranchTransition, self).__init__()
        self.number_input_branches = len(input_widths)
        self.branch_layers = nn.ModuleList()

        # Every output branch is either an input branch kept as it is, an input branch brought to its width, or a new
        # branch halved from the lowest resolution input branch
        for branch_index, output_width in enumerate(output_widths):
            if branch_index < len(input_widths):
                if input_widths[branch_index] == output_width:
                    self.branch_layers.append(nn.Identity())
                else:
                    self.branch_layers.append(build_exchange_convolution(
                        input_channels=input_widths[branch_index], output_channels=output_width, kernel_size=3,
                        stride=1, activation=True, normalization=normalization, device=device, dtype=dtype))
                continue

            # One strided convolution per halving from the lowest resolution input branch, the last one giving the
            # width of the new branch
            number_halvings = branch_index - len(input_widths) + 1
            halving_layers = []
            for halving_index in range(number_halvings):
                halving_width = output_width if halving_index == number_halvings - 1 else input_widths[-1]
                halving_layers.append(build_exchange_convolution(
                    input_channels=input_widths[-1], output_channels=halving_width, kernel_size=3,
                    stride=BRANCH_SCALE_FACTOR, activation=True, normalization=normalization, device=device,
                    dtype=dtype))
            self.branch_layers.append(nn.Sequential(*halving_layers))

    def forward(self, branch_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Gives the input feature maps of the branches of the new stage.

        Args:
            branch_features (List[torch.Tensor]): The [B, C_b, H_b, W_b] feature map of every branch of the previous
                stage, from the highest resolution.

        Returns:
            List[torch.Tensor]: The feature map of every branch of the new stage.
        """
        output_features = []
        for branch_index, branch_layer in enumerate(self.branch_layers):
            source_index = min(branch_index, self.number_input_branches - 1)
            output_features.append(branch_layer(branch_features[source_index]))
        return output_features


class ExchangeModule(nn.Module):
    """
    Module of a stage of the high resolution network: every branch goes through its own convolutions (residual pairs
    of 3x3 convolutions with the residual setting, the 'basic blocks' of HRNet), then the exchange unit gives every
    output branch the sum of all the branches brought to its resolution and width: the higher resolution branches by
    3x3 convolutions of stride 2, the lower resolution ones by a 1x1 convolution followed by a nearest upsampling. The
    last module of the network only computes the highest resolution output, the one read by the heatmap head.
    """

    def __init__(self, branch_widths: List[int], blocks_per_branch: int, number_output_branches: int,
                 modules: Dict[str, Any], normalization: Dict[str, Any], device: torch.device,
                 dtype: torch.dtype) -> None:
        """
        Builds the convolutions of every branch and the layers of the exchange unit.

        Args:
            branch_widths (List[int]): The channels of every branch, from the highest resolution.
            blocks_per_branch (int): The number of pairs of 3x3 convolutions of every branch.
            number_output_branches (int): The number of branches computed by the exchange unit, all of them, or 1 for
                the last module of the network.
            modules (Dict[str, Any]): The 'squeeze_and_excite' and 'residual_nets' flags of the branch convolutions.
            normalization (Dict[str, Any]): The normalization settings of every convolution.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(ExchangeModule, self).__init__()

        # Convolutions of every branch at its own resolution, the residual pairs keeping its width
        self.branches = nn.ModuleList()
        for branch_width in branch_widths:
            self.branches.append(ConvolutionBlock(
                input_channels=branch_width, output_channels=branch_width, bias=False,
                number_layers=2 * blocks_per_branch, kernel_size=3, stride=1, padding=1,
                feature_map_normalization=normalization["feature_map_normalization"],
                batch_normalization_momentum=normalization["batch_normalization_momentum"], activation=True,
                dropout_rate=0.0, add_pooling=False, device=device, dtype=dtype,
                squeeze_and_excite=bool(modules["squeeze_and_excite"]), residual=bool(modules["residual_nets"]),
                pre_activation=False, inception=False))

        # Exchange layers from every input branch to every output branch, the identity for the branch itself
        self.exchange_layers = nn.ModuleList()
        for output_index in range(number_output_branches):
            output_layers = nn.ModuleList()
            for input_index, input_width in enumerate(branch_widths):
                output_width = branch_widths[output_index]
                if input_index == output_index:
                    output_layers.append(nn.Identity())

                # Lower resolution branch: width changed by a 1x1 convolution, then upsampled to the output resolution
                elif input_index > output_index:
                    output_layers.append(nn.Sequential(
                        build_exchange_convolution(input_channels=input_width, output_channels=output_width,
                                                   kernel_size=1, stride=1, activation=False,
                                                   normalization=normalization, device=device, dtype=dtype),
                        nn.Upsample(scale_factor=BRANCH_SCALE_FACTOR ** (input_index - output_index), mode="nearest")))

                # Higher resolution branch: one strided 3x3 convolution per halving, the last one without activation
                # since the sum is activated
                else:
                    number_halvings = output_index - input_index
                    halving_layers = []
                    for halving_index in range(number_halvings):
                        is_last_halving = halving_index == number_halvings - 1
                        halving_width = output_width if is_last_halving else input_width
                        halving_layers.append(build_exchange_convolution(
                            input_channels=input_width, output_channels=halving_width, kernel_size=3,
                            stride=BRANCH_SCALE_FACTOR, activation=not is_last_halving, normalization=normalization,
                            device=device, dtype=dtype))
                    output_layers.append(nn.Sequential(*halving_layers))
            self.exchange_layers.append(output_layers)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, branch_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Runs the convolutions of every branch, then exchanges the features between the branches.

        Args:
            branch_features (List[torch.Tensor]): The [B, C_b, H_b, W_b] feature map of every branch, from the
                highest resolution.

        Returns:
            List[torch.Tensor]: The activated sum of the exchanged features of every output branch.
        """
        # Convolutions of every branch at its own resolution
        branch_outputs = [branch(input_tensor=features) for branch, features in zip(self.branches, branch_features)]

        # Sum of all the branches brought to the resolution and width of every output branch, then activated
        exchanged_features = []
        for output_layers in self.exchange_layers:
            summed_features = output_layers[0](branch_outputs[0])
            for input_index in range(1, len(branch_outputs)):
                summed_features = summed_features + output_layers[input_index](branch_outputs[input_index])
            exchanged_features.append(self.activation(summed_features))
        return exchanged_features


class HighResolutionNetwork(nn.Module):
    """
    Body of the keypoint detector on the 1/4 resolution feature map of the Backbone stem: stages of exchange modules
    over 2, 3, 4, ... parallel branches of widths C, 2C, 4C, 8C, ... at 1/4, 1/8, 1/16, 1/32, ... of the crop, one
    branch added by the transition before every stage. It returns the feature map of the highest resolution branch,
    from which the heatmaps are read, at 1/4 of the crop with C channels.
    """

    def __init__(self, input_channels: int, base_width: int, modules_per_stage: List[int], blocks_per_branch: int,
                 modules: Dict[str, Any], normalization: Dict[str, Any], device: torch.device,
                 dtype: torch.dtype) -> None:
        """
        Builds the transitions and the exchange modules of every stage from the 'HRNet' configuration section.

        Args:
            input_channels (int): The channels of the stem feature map, the output of the last Backbone block.
            base_width (int): The width C of the highest resolution branch, doubled for every lower resolution one.
            modules_per_stage (List[int]): The number of exchange modules of every stage, the stage s (from 0) having
                s + 2 branches, e.g. [1, 4, 3] for the stages of 2, 3 and 4 branches of HRNet-W32.
            blocks_per_branch (int): The number of pairs of 3x3 convolutions of every branch of every module.
            modules (Dict[str, Any]): The 'squeeze_and_excite' and 'residual_nets' flags of the branch convolutions.
            normalization (Dict[str, Any]): The 'feature_map_normalization' and 'batch_normalization_momentum'.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(HighResolutionNetwork, self).__init__()
        self.base_width = base_width
        self.modules_per_stage = list(modules_per_stage)
        self.branch_widths = [base_width * BRANCH_SCALE_FACTOR ** branch_index
                              for branch_index in range(len(modules_per_stage) + 1)]

        # Every stage starts with the transition adding its new branch, then chains its exchange modules, the very
        # last module only computing the highest resolution output
        self.transitions = nn.ModuleList()
        self.stages = nn.ModuleList()
        previous_widths = [input_channels]
        for stage_index, number_modules in enumerate(modules_per_stage):
            stage_widths = self.branch_widths[:stage_index + 2]
            self.transitions.append(BranchTransition(input_widths=previous_widths, output_widths=stage_widths,
                                                     normalization=normalization, device=device, dtype=dtype))
            stage_modules = nn.ModuleList()
            for module_index in range(number_modules):
                is_last_module = stage_index == len(modules_per_stage) - 1 and module_index == number_modules - 1
                stage_modules.append(ExchangeModule(
                    branch_widths=stage_widths, blocks_per_branch=blocks_per_branch,
                    number_output_branches=1 if is_last_module else len(stage_widths), modules=modules,
                    normalization=normalization, device=device, dtype=dtype))
            self.stages.append(stage_modules)
            previous_widths = stage_widths

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Runs the stages of the network on the stem feature map.

        Args:
            input_tensor (torch.Tensor): The [B, C_stem, H / 4, W / 4] stem feature map.

        Returns:
            torch.Tensor: The [B, C, H / 4, W / 4] feature map of the highest resolution branch.
        """
        branch_features = [input_tensor]
        for transition, stage_modules in zip(self.transitions, self.stages):
            branch_features = transition(branch_features=branch_features)
            for exchange_module in stage_modules:
                branch_features = exchange_module(branch_features=branch_features)
        return branch_features[0]

    def trace_branch_shapes(self, input_tensor: torch.Tensor) -> List[List[List[int]]]:
        """
        Runs the network on a dummy stem feature map and lists the shape of every branch at the end of every stage,
        for the architecture summary printed by the keypoint detector at the start of a run.

        Args:
            input_tensor (torch.Tensor): A [1, C_stem, H / 4, W / 4] dummy stem feature map.

        Returns:
            List[List[List[int]]]: For every stage, the [B, C, H, W] shape of every branch entering its last module.
        """
        stage_shapes = []
        branch_features = [input_tensor]
        for transition, stage_modules in zip(self.transitions, self.stages):
            branch_features = transition(branch_features=branch_features)
            stage_shapes.append([list(features.shape) for features in branch_features])
            for exchange_module in stage_modules:
                branch_features = exchange_module(branch_features=branch_features)
        return stage_shapes
