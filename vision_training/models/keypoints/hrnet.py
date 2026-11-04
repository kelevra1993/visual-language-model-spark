"""
High resolution network (HRNet) body of the keypoint detector: from the 1/4 resolution feature map of the Backbone
stem, it keeps a full resolution branch through the whole network and adds a branch at half the resolution and twice
the width at every stage (1/4, 1/8, 1/16, 1/32), the branches exchanging their features at the end of every module.
Its branches and exchange units are built from the ConvolutionBlock shared with the Backbone (residual pairs,
Squeeze-and-Excitation, batch or layer normalization). The highest resolution branch feeds the heatmap and tag head of
models/keypoints/keypoint_detector.py (build_prediction_head), and the higher resolution branch of HigherHRNet brings
it with the predictions of that head to half the input resolution for a second set of finer heatmaps.
"""

import torch

from torch import nn
from typing import Any, Dict, List

from vision_training.models.components.convolution_block import (BATCH_NORMALIZATION_MOMENTUM, ChannelLayerNormalizer,
                                                               ConvolutionBlock)

# Ratio between the resolutions, and between the widths, of two consecutive branches
BRANCH_SCALE_FACTOR = 2

# Kernel size of the transposed convolution doubling the resolution of the higher resolution branch, with a padding of
# 1 and a stride of 2 giving exactly twice the input size
HIGHER_RESOLUTION_KERNEL_SIZE = 4


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
        normalization (Dict[str, Any]): The 'feature_map_normalization' setting.
        device (torch.device): The device of the parameters.
        dtype (torch.dtype): The precision of the parameters.

    Returns:
        ConvolutionBlock: The single layer block, without pooling.
    """
    return ConvolutionBlock(input_channels=input_channels, output_channels=output_channels, bias=False, number_layers=1,
                            kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                            feature_map_normalization=normalization["feature_map_normalization"],
                            batch_normalization_momentum=BATCH_NORMALIZATION_MOMENTUM,
                            activation=activation, dropout_rate=0.0, add_pooling=False, device=device, dtype=dtype,
                            squeeze_and_excite=False, residual=False, pre_activation=False, inception=False)


def build_feature_normalization(channels: int, normalization: Dict[str, Any], device: torch.device,
                                dtype: torch.dtype) -> nn.Module:
    """
    Builds a stand-alone normalization layer with the normalization strategy of the high resolution network, for the
    transposed convolution of the higher resolution branch, which is not a ConvolutionBlock.

    Args:
        channels (int): The number of channels of the normalized feature map.
        normalization (Dict[str, Any]): The 'feature_map_normalization' setting.
        device (torch.device): The device of the parameters.
        dtype (torch.dtype): The precision of the parameters.

    Returns:
        nn.Module: The batch or layer normalization layer, or the identity without normalization.

    Raises:
        ValueError: If the normalization strategy is not "batch", "layer" or "none".
    """
    feature_map_normalization = normalization["feature_map_normalization"]
    if feature_map_normalization == "batch":
        return nn.BatchNorm2d(num_features=channels, momentum=BATCH_NORMALIZATION_MOMENTUM,
                              device=device, dtype=dtype)
    if feature_map_normalization == "layer":
        return ChannelLayerNormalizer(channels=channels, device=device, dtype=dtype)
    if feature_map_normalization in ["none", None]:
        return nn.Identity()
    raise ValueError(f"The high resolution network supports the 'batch', 'layer' and 'none' normalizations, got "
                     f"{feature_map_normalization!r}.")


def build_prediction_head(input_channels: int, hidden_width: int, output_channels: int, normalization: Dict[str, Any],
                          device: torch.device, dtype: torch.dtype) -> nn.Sequential:
    """
    Builds the layers turning a feature map of the keypoint detector into its predictions, the heatmaps and tags at a
    quarter of the image (keypoint_detector.py) or the heatmaps of the higher resolution branch at half of it. A 1x1
    convolution gives at every cell a linear combination of the input channels, so that a feature map narrower than
    the number of heatmaps cannot raise each heatmap on its own keypoints and keep the others at zero. With a
    hidden_width, a 3x3 convolution (with its normalization and a ReLU) first widens the features, as in the heads of
    CenterNet ("Objects as Points" https://arxiv.org/abs/1904.07850); without it, the 1x1 convolution reads the
    features directly, as in HigherHRNet.

    Args:
        input_channels (int): The channels of the feature map.
        hidden_width (int): The channels of the 3x3 convolution before the predictions, 0 for none.
        output_channels (int): The number of predicted maps.
        normalization (Dict[str, Any]): The normalization settings of the 3x3 convolution.
        device (torch.device): The device of the parameters.
        dtype (torch.dtype): The precision of the parameters.

    Returns:
        nn.Sequential: The optional 3x3 convolution then the 1x1 prediction convolution, the last layer, which the
            keypoint detector initialises with small weights and the bias of the heatmap prior.
    """
    head_layers = []
    prediction_input_channels = input_channels

    # Wider features at every cell, read on the 3x3 neighbourhood
    if hidden_width > 0:
        head_layers.append(build_exchange_convolution(input_channels=input_channels, output_channels=hidden_width,
                                                      kernel_size=3, stride=1, activation=True,
                                                      normalization=normalization, device=device, dtype=dtype))
        prediction_input_channels = hidden_width

    # One map per output channel
    head_layers.append(nn.Conv2d(in_channels=prediction_input_channels, out_channels=output_channels, kernel_size=1,
                                 device=device, dtype=dtype))
    return nn.Sequential(*head_layers)


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
                batch_normalization_momentum=BATCH_NORMALIZATION_MOMENTUM, activation=True,
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
    over 2, 3, 4, ... parallel branches of widths C, 2C, 4C, 8C, ... at 1/4, 1/8, 1/16, 1/32, ... of the image, one
    branch added by the transition before every stage. It returns the feature map of the highest resolution branch,
    from which the heatmaps and tags are read, at 1/4 of the image with C channels.
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
            normalization (Dict[str, Any]): The 'feature_map_normalization' setting.
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


class HigherResolutionBranch(nn.Module):
    """
    Higher resolution branch of HigherHRNet (paper "HigherHRNet: Scale-Aware Representation Learning for Bottom-Up
    Human Pose Estimation" https://arxiv.org/abs/1908.10357) on the highest resolution branch of the high resolution
    network: its features, concatenated with the heatmap logits and tags predicted from them at a quarter of the input
    resolution, are brought to half the input resolution by a 4x4 transposed convolution of stride 2 (with its
    normalization and a ReLU), refined by residual pairs of 3x3 convolutions like the branches of the exchange modules,
    and turned into one heatmap logit per channel by the prediction head of build_prediction_head (a 1x1 convolution,
    after a hidden 3x3 convolution with a hidden width). The keypoint detector averages the sigmoid of these logits
    with the upsampled quarter resolution heatmaps, which locates the keypoints of small objects more precisely.
    """

    def __init__(self, feature_channels: int, prediction_channels: int, heatmap_channels: int, hidden_width: int,
                 blocks_per_branch: int, modules: Dict[str, Any], normalization: Dict[str, Any], device: torch.device,
                 dtype: torch.dtype) -> None:
        """
        Builds the transposed convolution, the refinement convolutions and the heatmap convolution of the branch.

        Args:
            feature_channels (int): The width of the highest resolution branch, kept by the higher resolution one.
            prediction_channels (int): The channels of the quarter resolution predictions, the heatmap logits and the
                tags.
            heatmap_channels (int): The number of heatmaps predicted at half the input resolution.
            hidden_width (int): The channels of the 3x3 convolution of the heatmap head, 0 for a 1x1 convolution only.
            blocks_per_branch (int): The number of residual pairs of 3x3 convolutions of the refinement.
            modules (Dict[str, Any]): The 'squeeze_and_excite' and 'residual_nets' flags of the refinement.
            normalization (Dict[str, Any]): The normalization settings of every convolution.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(HigherResolutionBranch, self).__init__()

        # Transposed convolution doubling the resolution, its normalization and activation
        self.transposed_convolution = nn.ConvTranspose2d(
            in_channels=feature_channels + prediction_channels, out_channels=feature_channels,
            kernel_size=HIGHER_RESOLUTION_KERNEL_SIZE, stride=BRANCH_SCALE_FACTOR, padding=1, bias=False,
            device=device, dtype=dtype)
        self.normalization = build_feature_normalization(channels=feature_channels, normalization=normalization,
                                                         device=device, dtype=dtype)
        self.activation = nn.ReLU(inplace=False)

        # Refinement at the doubled resolution, the residual pairs keeping the width
        self.refinement = ConvolutionBlock(
            input_channels=feature_channels, output_channels=feature_channels, bias=False,
            number_layers=2 * blocks_per_branch, kernel_size=3, stride=1, padding=1,
            feature_map_normalization=normalization["feature_map_normalization"],
            batch_normalization_momentum=BATCH_NORMALIZATION_MOMENTUM, activation=True,
            dropout_rate=0.0, add_pooling=False, device=device, dtype=dtype,
            squeeze_and_excite=bool(modules["squeeze_and_excite"]), residual=bool(modules["residual_nets"]),
            pre_activation=False, inception=False)

        # One heatmap per channel, the last convolution initialised by the keypoint detector like its quarter
        # resolution head
        self.heatmap_head = build_prediction_head(input_channels=feature_channels, hidden_width=hidden_width,
                                                  output_channels=heatmap_channels, normalization=normalization,
                                                  device=device, dtype=dtype)

    def forward(self, features: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """
        Predicts the heatmap logits at half the input resolution from the features and predictions at a quarter of it.

        Args:
            features (torch.Tensor): The [B, C_f, H / 4, W / 4] features of the highest resolution branch.
            predictions (torch.Tensor): The [B, C_p, H / 4, W / 4] heatmap logits and tags predicted from them.

        Returns:
            torch.Tensor: The [B, C_h, H / 2, W / 2] heatmap logits, before the sigmoid.
        """
        upsampled_features = self.transposed_convolution(torch.cat(tensors=(features, predictions), dim=1))
        refined_features = self.refinement(input_tensor=self.activation(self.normalization(upsampled_features)))
        return self.heatmap_head(refined_features)
