"""
Feature neck of the detection tasks, between the backbone blocks of the detection levels and the region proposal,
detection and mask heads: none (the heads read the blocks), a feature pyramid (1x1 laterals and a top-down path
bringing the context of the coarse levels to the fine ones) or path aggregation (the pyramid, then a bottom-up path
bringing the localisation of the fine levels back to the coarse ones). Built by two_stage_detector.py from the 'Neck'
section, applied to the backbone outputs in the training, the evaluation and the exports (inference_detector.py), its
outputs keeping the levels and sizes of the blocks with the channels of 'Neck.output_channels'.
"""

import torch
import torch.nn.functional as functional

from torch import nn
from typing import Dict, List

from vision_training.models.components.convolution_block import ConvolutionBlock

# Types of neck: the heads read the backbone blocks, the feature pyramid, or the feature pyramid then the bottom-up path
NECK_TYPES = ["none", "feature_pyramid", "path_aggregation"]


def build_neck_convolution(input_channels: int, output_channels: int, kernel_size: int, stride: int, activation: bool,
                           feature_map_normalization: str, batch_normalization_momentum: float, device: torch.device,
                           dtype: torch.dtype) -> ConvolutionBlock:
    """
    Builds one convolution of the neck as a single-layer ConvolutionBlock without module (no residual shortcut,
    inception or squeeze and excitation), padded to keep the size at stride 1, so that the neck shares the
    normalization layers of the backbone and the heads.

    Args:
        input_channels (int): The channels of the input feature map.
        output_channels (int): The channels of the output feature map, those of 'Neck.output_channels'.
        kernel_size (int): 1 for the laterals, 3 for the other convolutions.
        stride (int): 1, or the ratio of the sizes of two levels for the downsampling of the bottom-up path.
        activation (bool): Whether a ReLU follows the normalization (bottom-up path only).
        feature_map_normalization (str): 'Neck.normalization.feature_map_normalization'.
        batch_normalization_momentum (float): The momentum of the batch normalization running statistics.
        device (torch.device): The device of the parameters.
        dtype (torch.dtype): The data type of the parameters.

    Returns:
        ConvolutionBlock: The convolution, followed by its normalization and activation.
    """
    return ConvolutionBlock(input_channels=input_channels, output_channels=output_channels, bias=True, number_layers=1,
                            kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                            feature_map_normalization=feature_map_normalization,
                            batch_normalization_momentum=batch_normalization_momentum, activation=activation,
                            dropout_rate=0.0, add_pooling=False, device=device, dtype=dtype, squeeze_and_excite=False,
                            residual=False, pre_activation=False, inception=False)


class FeatureNeck(nn.Module):
    """
    Feature neck of the two-stage detector, mapping the backbone feature maps of the detection levels to the feature
    maps read by the heads, with the same levels and sizes. With 'none' it has no layer and returns the backbone
    feature maps, so that the runs without neck keep their weights and their results.
    """

    def __init__(self, neck_type: str, level_information: Dict[str, Dict[str, int]], output_channels: int,
                 feature_map_normalization: str, batch_normalization_momentum: float, device: torch.device,
                 dtype: torch.dtype) -> None:
        """
        Builds the layers of the neck for the detection levels of the backbone, the finer of two consecutive levels
        having to be a whole multiple of the coarser one, so that the top-down path upsamples by a whole factor and
        the bottom-up path downsamples with a whole stride (1 between levels of the same size).

        Args:
            neck_type (str): 'Neck.type', one of NECK_TYPES.
            level_information (Dict[str, Dict[str, int]]): The 'feature_map_size' and 'number_channels' of the backbone
                block of every level (Backbone.compute_detection_input_information), keyed by block number.
            output_channels (int): 'Neck.output_channels', the channels of every level after the neck.
            feature_map_normalization (str): 'Neck.normalization.feature_map_normalization'.
            batch_normalization_momentum (float): The momentum of the batch normalization running statistics.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The data type of the parameters.

        Raises:
            ValueError: If the neck type is unknown, or if a level is not a whole multiple of the next coarser one.
        """
        super(FeatureNeck, self).__init__()
        if neck_type not in NECK_TYPES:
            raise ValueError(f"Unknown neck type {neck_type!r}, expected one of {NECK_TYPES}.")

        # Levels from the finest to the coarsest (by block number), with their sizes as plain numbers so that the
        # TorchScript trace of the exports never reads them from a tensor shape
        self.neck_type = neck_type
        self.level_names: List[str] = sorted(level_information, key=int)
        self.level_information = level_information
        self.level_sizes = {level_name: level_information[level_name]["feature_map_size"]
                            for level_name in self.level_names}
        self.output_channels = output_channels

        # Without neck, no layer is built, so that the state dictionary and the random draws stay those of the runs
        # written before the neck
        if neck_type == "none":
            return

        # Every level must be a whole multiple of the next coarser one
        for finer_name, coarser_name in zip(self.level_names[:-1], self.level_names[1:]):
            if self.level_sizes[finer_name] % self.level_sizes[coarser_name] != 0:
                raise ValueError(f"The neck needs every detection level to be a whole multiple of the next coarser "
                                 f"one, but level {finer_name} is {self.level_sizes[finer_name]} x "
                                 f"{self.level_sizes[finer_name]} and level {coarser_name} is "
                                 f"{self.level_sizes[coarser_name]} x {self.level_sizes[coarser_name]}.")

        # Laterals bringing every block to the output channels, and the 3x3 convolution of every merged level, both
        # without activation as in the feature pyramid network
        self.lateral_convolutions = nn.ModuleDict()
        self.output_convolutions = nn.ModuleDict()
        for level_name in self.level_names:
            self.lateral_convolutions[level_name] = build_neck_convolution(
                input_channels=level_information[level_name]["number_channels"], output_channels=output_channels,
                kernel_size=1, stride=1, activation=False, feature_map_normalization=feature_map_normalization,
                batch_normalization_momentum=batch_normalization_momentum, device=device, dtype=dtype)
            self.output_convolutions[level_name] = build_neck_convolution(
                input_channels=output_channels, output_channels=output_channels, kernel_size=3, stride=1,
                activation=False, feature_map_normalization=feature_map_normalization,
                batch_normalization_momentum=batch_normalization_momentum, device=device, dtype=dtype)

        # Bottom-up path of path aggregation: for every level but the finest, a strided 3x3 convolution of the finer
        # output and a 3x3 convolution of the sum, both activated as in PANet
        self.downsampling_convolutions = nn.ModuleDict()
        self.smoothing_convolutions = nn.ModuleDict()
        if neck_type == "path_aggregation":
            for finer_name, level_name in zip(self.level_names[:-1], self.level_names[1:]):
                self.downsampling_convolutions[level_name] = build_neck_convolution(
                    input_channels=output_channels, output_channels=output_channels, kernel_size=3,
                    stride=self.level_sizes[finer_name] // self.level_sizes[level_name], activation=True,
                    feature_map_normalization=feature_map_normalization,
                    batch_normalization_momentum=batch_normalization_momentum, device=device, dtype=dtype)
                self.smoothing_convolutions[level_name] = build_neck_convolution(
                    input_channels=output_channels, output_channels=output_channels, kernel_size=3, stride=1,
                    activation=True, feature_map_normalization=feature_map_normalization,
                    batch_normalization_momentum=batch_normalization_momentum, device=device, dtype=dtype)

    def forward(self, feature_maps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Maps the backbone feature maps of the detection levels to those read by the heads.

        Args:
            feature_maps (Dict[str, torch.Tensor]): The [B, C_level, S_level, S_level] backbone feature map of every
                level, keyed by block number.

        Returns:
            Dict[str, torch.Tensor]: The feature map of every level from the finest to the coarsest, the backbone ones
                without neck, otherwise [B, output_channels, S_level, S_level].
        """
        if self.neck_type == "none":
            return feature_maps

        # Top-down path, from the coarsest level: every lateral receives the merged coarser level, upsampled to its
        # size by repeating its cells (added directly between levels of the same size)
        merged_maps: Dict[str, torch.Tensor] = {}
        coarsest_index = len(self.level_names) - 1
        for level_index in range(coarsest_index, -1, -1):
            level_name = self.level_names[level_index]
            merged_map = self.lateral_convolutions[level_name](input_tensor=feature_maps[level_name])
            if level_index < coarsest_index:
                coarser_name = self.level_names[level_index + 1]
                coarser_map = merged_maps[coarser_name]
                if self.level_sizes[coarser_name] != self.level_sizes[level_name]:
                    coarser_map = functional.interpolate(
                        input=coarser_map, size=(self.level_sizes[level_name], self.level_sizes[level_name]),
                        mode="nearest")
                merged_map = merged_map + coarser_map
            merged_maps[level_name] = merged_map

        # Feature pyramid: a 3x3 convolution of every merged level, in the order of the levels
        pyramid_maps = {level_name: self.output_convolutions[level_name](input_tensor=merged_maps[level_name])
                        for level_name in self.level_names}
        if self.neck_type == "feature_pyramid":
            return pyramid_maps

        # Bottom-up path of path aggregation, from the finest level: every pyramid level receives the downsampled
        # aggregated finer level, then a 3x3 convolution
        aggregated_maps = {self.level_names[0]: pyramid_maps[self.level_names[0]]}
        for finer_name, level_name in zip(self.level_names[:-1], self.level_names[1:]):
            downsampled_map = self.downsampling_convolutions[level_name](input_tensor=aggregated_maps[finer_name])
            aggregated_maps[level_name] = self.smoothing_convolutions[level_name](
                input_tensor=pyramid_maps[level_name] + downsampled_map)
        return aggregated_maps

    def compute_output_information(self) -> Dict[str, Dict[str, int]]:
        """
        Gives the size and channels of every level after the neck, from which the region proposal, detection and mask
        heads are built: the backbone ones without neck, otherwise the same sizes with the output channels.

        Returns:
            Dict[str, Dict[str, int]]: The 'feature_map_size' and 'number_channels' of every level, keyed by block
                number, from the finest to the coarsest.
        """
        if self.neck_type == "none":
            return {level_name: dict(self.level_information[level_name]) for level_name in self.level_names}
        return {level_name: {"feature_map_size": self.level_sizes[level_name], "number_channels": self.output_channels}
                for level_name in self.level_names}
