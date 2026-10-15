"""
Mask branch of the instance segmenter: the MaskHead of one detection level pools every box from the feature map of its
level with RoIAlign at the pool size of that level, runs the pooled features through a ConvolutionBlock, doubles their
resolution as many times as fits in the mask resolution (transposed convolutions, or bilinear upsamplings followed by a
convolution) and predicts one mask logit map per class, as in Mask R-CNN, the logits being resized bilinearly to the
mask resolution when the doublings fall short of it, so that every level gives masks of the same size. Built for every
detection level by the InstanceSegmenter, which trains it on the positive proposals and runs it on the final
detections, and reused by the InferenceInstanceSegmenter of the exports.
"""

import torch

from typing import Any, Dict, List
from torch import nn
from torch.nn import functional as torch_functional
from torchvision.ops import roi_align

from vision_training.models.components.convolution_block import ConvolutionBlock

# Parameters of the single transposed convolution of the mask heads written before the mask geometry could be
# configured, loaded into the first upsampling layer
LEGACY_UPSAMPLING_PARAMETER_NAMES = ["weight", "bias"]


def count_mask_upsamplings(roi_align_pool_size: int, mask_resolution: int) -> int:
    """
    Gives the number of doublings of the pooled features of a mask head: as many as fit in the mask resolution, the
    logits being resized to it when the last doubled size falls short, e.g. 1 for a pool of 14 and masks of 28, 2 for
    a pool of 7, 1 for a pool of 10 (20 x 20 logits resized to 28 x 28) and 0 for a pool equal to the masks.

    Args:
        roi_align_pool_size (int): The RoIAlign pool size P of the level, at most the mask resolution.
        mask_resolution (int): The side M of the masks of every level.

    Returns:
        int: The number of doublings, the largest n with P x 2^n <= M.
    """
    number_upsamplings = 0
    while roi_align_pool_size * 2 ** (number_upsamplings + 1) <= mask_resolution:
        number_upsamplings += 1
    return number_upsamplings


class MaskHead(nn.Module):
    """
    Predicts the binary mask of the object of every box of one detection level, one [M, M] logit map per class, the
    channel of the detected (or ground truth) class being read by the instance segmenter.
    """

    def __init__(self, head_configuration: Dict[str, Any], mask_resolution: int, upsampling: str, number_classes: int,
                 input_channels: int, feature_map_normalization: str, batch_normalization_momentum: float,
                 input_image_size: int, feature_map_size: int, modules: Dict[str, Any], dtype: torch.dtype,
                 device: torch.device) -> None:
        """
        Builds the mask head of one detection level from the 'MaskHead' section of the configuration.

        Args:
            head_configuration (Dict[str, Any]): The 'roi_align_pool_size' P of this level and the
                [number_layers, output_channels] 'convolutions' of its convolution block.
            mask_resolution (int): The side M of the masks, the same on every level, at least P.
            upsampling (str): The doubling of the pooled features: "transposed" (a 2x2 transposed convolution) or
                "bilinear" (a bilinear upsampling followed by a 3x3 convolution).
            number_classes (int): The number of classes, background included, one mask channel per class.
            input_channels (int): The number of channels of the feature map of this level.
            feature_map_normalization (str): The normalization of the convolution block ("batch", "layer" or "none").
            batch_normalization_momentum (float): The momentum of the running statistics of its batch normalization.
            input_image_size (int): The size of the square network input, giving the spatial scale of RoIAlign.
            feature_map_size (int): The size of the square feature map of this level.
            modules (Dict[str, Any]): The 'MaskHead.modules' flags ('squeeze_and_excite', 'residual_nets',
                'inception') of the convolution block.
            dtype (torch.dtype): The data type of the parameters.
            device (torch.device): The device of the parameters.
        """
        super(MaskHead, self).__init__()

        # Mask geometry of the level: its pool size, the doublings that fit in the mask resolution and their size
        self.number_classes = number_classes
        self.feature_map_size = feature_map_size
        self.roi_align_pool_size = head_configuration["roi_align_pool_size"]
        self.mask_size = mask_resolution
        self.upsampling = upsampling
        self.number_upsamplings = count_mask_upsamplings(roi_align_pool_size=self.roi_align_pool_size,
                                                         mask_resolution=mask_resolution)
        self.upsampled_size = self.roi_align_pool_size * 2 ** self.number_upsamplings
        self.convolution_number_layers, convolution_output_channels = head_configuration["convolutions"]

        # Ratio mapping the boxes from network input pixels to feature map cells, as in the detection head
        self.spatial_scale = feature_map_size / input_image_size

        # Convolutional feature extraction on the pooled features, at the pool resolution
        self.convolutional_block = ConvolutionBlock(
            input_channels=input_channels, output_channels=convolution_output_channels, bias=True,
            number_layers=self.convolution_number_layers, kernel_size=3, stride=1, padding=1,
            feature_map_normalization=feature_map_normalization,
            batch_normalization_momentum=batch_normalization_momentum, activation=True, dropout_rate=0.0,
            add_pooling=False, device=device, dtype=dtype, squeeze_and_excite=bool(modules["squeeze_and_excite"]),
            residual=bool(modules["residual_nets"]), pre_activation=False, inception=bool(modules["inception"]))

        # Doublings of the resolution, learned by a transposed convolution or interpolated then convolved with a 3x3
        # kernel (nothing but the 1x1 mask predictor follows them), each followed by a ReLU
        self.upsampling_layers = nn.ModuleList()
        for _ in range(self.number_upsamplings):
            if upsampling == "transposed":
                self.upsampling_layers.append(nn.ConvTranspose2d(
                    in_channels=convolution_output_channels, out_channels=convolution_output_channels, kernel_size=2,
                    stride=2, device=device, dtype=dtype))
            else:
                self.upsampling_layers.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(in_channels=convolution_output_channels, out_channels=convolution_output_channels,
                              kernel_size=3, padding=1, device=device, dtype=dtype)))
        self.activation = nn.ReLU()

        # One mask logit map per class
        self.mask_predictor = nn.Conv2d(in_channels=convolution_output_channels, out_channels=number_classes,
                                        kernel_size=1, device=device, dtype=dtype)

    def _load_from_state_dict(self, state_dictionary: Dict[str, Any], prefix: str, local_metadata: Dict[str, Any],
                              strict: bool, missing_keys: List[str], unexpected_keys: List[str],
                              error_messages: List[str]) -> None:
        """
        Loads the weights of this head from a checkpoint, the checkpoints written before the mask geometry could be
        configured holding their single transposed convolution as 'upsampling': its weights go to the first upsampling
        layer, the same layer of the default geometry (masks twice the pool size, transposed convolution), so that the
        instance segmentation runs migrated by configuration_migration.py still resume, evaluate and export.

        Args:
            state_dictionary (Dict[str, Any]): The state dictionary being loaded, modified in place.
            prefix (str): The prefix of the keys of this head in the state dictionary.
            local_metadata (Dict[str, Any]): The metadata of this head in the state dictionary.
            strict (bool): Whether the keys must match the parameters exactly.
            missing_keys (List[str]): The missing keys found so far, filled by PyTorch.
            unexpected_keys (List[str]): The unexpected keys found so far, filled by PyTorch.
            error_messages (List[str]): The loading errors found so far, filled by PyTorch.
        """
        for parameter_name in LEGACY_UPSAMPLING_PARAMETER_NAMES:
            legacy_key = f"{prefix}upsampling.{parameter_name}"
            if legacy_key in state_dictionary:
                state_dictionary[f"{prefix}upsampling_layers.0.{parameter_name}"] = state_dictionary.pop(legacy_key)
        super(MaskHead, self)._load_from_state_dict(state_dict=state_dictionary, prefix=prefix,
                                                    local_metadata=local_metadata, strict=strict,
                                                    missing_keys=missing_keys, unexpected_keys=unexpected_keys,
                                                    error_msgs=error_messages)

    def forward(self, proposal_boxes: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Predicts the mask logits of every box from the feature map of this level.

        Args:
            proposal_boxes (torch.Tensor): The [R, 5] boxes, the image index of every box followed by its
                [x1, y1, x2, y2] corners in network input pixels.
            input_tensor (torch.Tensor): The [B, C, H, W] feature map of this level.

        Returns:
            torch.Tensor: The [R, number_classes, M, M] mask logits.
        """
        # Pool every box with the RoIAlign settings of the detection head, exported as the ONNX RoiAlign operator
        pooled_features = roi_align(input=input_tensor, boxes=proposal_boxes,
                                    output_size=(self.roi_align_pool_size, self.roi_align_pool_size),
                                    spatial_scale=self.spatial_scale, sampling_ratio=0, aligned=True)

        # Refine the pooled features, double their resolution as many times as fits and predict one mask per class
        features = self.convolutional_block(input_tensor=pooled_features)
        for upsampling_layer in self.upsampling_layers:
            features = self.activation(input=upsampling_layer(input=features))
        mask_logits = self.mask_predictor(input=features)

        # Logits resized to the mask resolution when the doublings fall short of it, every level giving M x M masks
        if self.upsampled_size != self.mask_size:
            mask_logits = torch_functional.interpolate(input=mask_logits, size=(self.mask_size, self.mask_size),
                                                       mode="bilinear", align_corners=False)
        return mask_logits
