"""
Upsampling path of the U-Net of the semantic segmentation task: SegmentationDecoder brings the final feature map of the
shared Backbone back to the input resolution, merging at every level the output of a backbone block before its pooling
(skip connection), and compute_skip_block_numbers gives these levels. Used by the semantic segmenter, the
configuration checks and the run name.
"""

import torch

from torch import nn
from typing import Any, Dict, List, Tuple

from vision_training.models.components.convolution_block import ConvolutionBlock


def compute_skip_block_numbers(number_blocks: int, last_max_pooling: bool) -> List[int]:
    """
    Gives the backbone blocks whose output before pooling is a skip connection of the decoder, from the deepest to
    the finest, i.e. the decoder levels in their order: every block before the last one, whose output is the
    bottleneck, and the last block too when it ends with a max pooling, its pooled output being then the bottleneck.

    Args:
        number_blocks (int): The number of blocks of the backbone.
        last_max_pooling (bool): Whether the last block of the backbone ends with a max pooling.

    Returns:
        List[int]: The block numbers of the decoder levels, from the deepest to the finest (1, at the input resolution).
    """
    deepest_skip_block = number_blocks if last_max_pooling else number_blocks - 1
    return list(range(deepest_skip_block, 0, -1))


class SegmentationDecoder(nn.Module):
    """
    Upsampling path of the U-Net of the semantic segmentation task, read by the segmentation head.

    Every level doubles the resolution of its input (a learned 2x2 transposed convolution, or a bilinear upsampling
    followed by a 1x1 convolution) to the channels of the level, concatenates the skip connection of its backbone block
    before it (the block output before its pooling, at the same resolution), and refines the result with a
    ConvolutionBlock without pooling, with the normalization and the Squeeze-and-Excitation and residual options of the
    'SegmentationDecoder' section. The last level is at the input resolution of the backbone. The layers are keyed by
    the backbone block of their level, so that two decoders of the same levels share their weight names.
    """

    def __init__(self, decoder_configuration: Dict[str, Any], block_channels: List[int], last_max_pooling: bool,
                 device: torch.device, dtype: torch.dtype) -> None:
        """
        Builds the upsampling layer and the convolution block of every level.

        Args:
            decoder_configuration (Dict[str, Any]): The 'SegmentationDecoder' section: 'upsampling', 'convolutions'
                ([number of layers, output channels] of every level, keyed by backbone block), 'modules' and
                'feature_map_normalization'.
            block_channels (List[int]): The output channels of every backbone block, in order.
            last_max_pooling (bool): Whether the last backbone block ends with a max pooling.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(SegmentationDecoder, self).__init__()
        self.upsampling = decoder_configuration["upsampling"]
        self.feature_map_normalization = decoder_configuration["feature_map_normalization"]
        self.skip_block_numbers = compute_skip_block_numbers(number_blocks=len(block_channels),
                                                             last_max_pooling=last_max_pooling)
        self.skip_channels = {str(block_number): block_channels[block_number - 1]
                              for block_number in self.skip_block_numbers}

        # One upsampling layer and one convolution block per level, from the deepest to the finest
        self.upsampling_layers = nn.ModuleDict()
        self.convolution_blocks = nn.ModuleDict()
        current_channels = block_channels[-1]
        for block_number in self.skip_block_numbers:
            level_key = str(block_number)
            number_layers, level_channels = decoder_configuration["convolutions"][level_key]

            # Double the resolution to the channels of the level, learned or bilinear
            if self.upsampling == "transposed":
                self.upsampling_layers[level_key] = nn.ConvTranspose2d(in_channels=current_channels,
                                                                       out_channels=level_channels, kernel_size=2,
                                                                       stride=2, device=device, dtype=dtype)
            else:
                self.upsampling_layers[level_key] = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(in_channels=current_channels, out_channels=level_channels, kernel_size=1,
                              device=device, dtype=dtype))

            # Refine the concatenation of the skip connection and of the upsampled features
            self.convolution_blocks[level_key] = ConvolutionBlock(
                input_channels=self.skip_channels[level_key] + level_channels, output_channels=level_channels,
                bias=True, number_layers=number_layers, kernel_size=3, stride=1, padding=1,
                feature_map_normalization=self.feature_map_normalization, activation=True, dropout_rate=0.0,
                add_pooling=False, device=device, dtype=dtype,
                squeeze_and_excite=bool(decoder_configuration["modules"]["squeeze_and_excite"]),
                residual=bool(decoder_configuration["modules"]["residual_nets"]), inception=False)
            current_channels = level_channels

        # Channels of the decoded feature map, at the input resolution
        self.output_channels = current_channels

    def forward(self, bottleneck_tensor: torch.Tensor, skip_tensors: Dict[str, torch.Tensor],
                return_level_outputs: bool = False) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decodes the final feature map of the backbone into a feature map at the input resolution, with the output of
        every level when asked, read by the deep supervision heads of the segmenter.

        Args:
            bottleneck_tensor (torch.Tensor): The [B, C, h, w] final feature map of the backbone.
            skip_tensors (Dict[str, torch.Tensor]): The [B, C_b, H_b, W_b] output before pooling of every backbone
                block, keyed by block number (Backbone.forward with return_pre_pooling_outputs).
            return_level_outputs (bool): Whether to also return the output of every level. Defaults to False.

        Returns:
            torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]: The [B, output_channels, H, W] decoded feature
                map, at the resolution of the first block, and with return_level_outputs the [B, C_l, H_l, W_l]
                output of every level keyed by its backbone block.
        """
        current_tensor = bottleneck_tensor
        level_outputs = {}

        # Upsample, merge the skip connection of the level, then refine, from the deepest level to the finest
        for level_key, upsampling_layer in self.upsampling_layers.items():
            current_tensor = upsampling_layer(current_tensor)
            skip_tensor = skip_tensors[level_key]

            # A backbone input whose size is not a power of two may give an upsampled map one pixel off its skip
            if current_tensor.shape[-2:] != skip_tensor.shape[-2:]:
                current_tensor = torch.nn.functional.interpolate(input=current_tensor, size=skip_tensor.shape[-2:],
                                                                 mode="bilinear", align_corners=False)
            current_tensor = self.convolution_blocks[level_key](
                input_tensor=torch.cat(tensors=[skip_tensor, current_tensor], dim=1))
            level_outputs[level_key] = current_tensor

        if return_level_outputs:
            return current_tensor, level_outputs
        return current_tensor
