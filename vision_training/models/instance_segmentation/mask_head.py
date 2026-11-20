"""
Mask branch of the instance segmenter: the MaskHead of one detection level pools every box from the feature map of its
level with RoIAlign at the pool size of that level, runs the pooled features through a ConvolutionBlock, doubles their
resolution as many times as fits in the mask resolution (transposed convolutions, or bilinear upsamplings followed by a
convolution) and predicts one mask logit map per class, as in Mask R-CNN, the logits being resized bilinearly to the
mask resolution when the doublings fall short of it, so that every level gives masks of the same size. Built for every
detection level by the InstanceSegmenter, which trains it on the positive proposals and runs it on the final
detections, and reused by the InferenceInstanceSegmenter of the exports.
The mask heads can also read finer inputs than the map of their level (stage 3b of the instance segmentation
benchmark): backbone blocks before their pooling pooled at the pool size and concatenated before the convolutions
(U-Net skips), and the fine features of MaskFineInputs, shared by the levels (the image branch, RoIAlign of the
normalized input image followed by plain convolutions, and backbone blocks pooled at the mask resolution), concatenated
after the doublings and mixed by the fusion convolutions of the head before its predictor.
"""

import torch

from typing import Any, Dict, List, Optional
from torch import nn
from torch.nn import functional as torch_functional
from torchvision.ops import roi_align

from vision_training.configuration_processing.configuration_values import VALID_UPSAMPLINGS
from vision_training.models.backbones.backbone import SKIP_OUTPUT_PREFIX
from vision_training.models.components.convolution_block import ConvolutionBlock


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


def build_plain_convolutions(input_channels: int, convolutions: List[int], feature_map_normalization: str,
                             batch_normalization_momentum: float, dtype: torch.dtype,
                             device: torch.device) -> ConvolutionBlock:
    """
    Builds the plain 3x3 convolutions of the fine inputs of the mask heads, the image branch of MaskFineInputs and the
    fusion convolutions of every MaskHead: a convolution, the normalization of the mask heads and a ReLU per layer at
    the resolution of their input, without the residual shortcuts, recalibration or inception modules of
    'MaskHead.modules', as designed for stage 3b of the instance segmentation benchmark.

    Args:
        input_channels (int): The channels of their input.
        convolutions (List[int]): The [number_layers, output_channels] of the convolutions, at least one layer.
        feature_map_normalization (str): The normalization of the mask heads ("batch", "layer" or "none").
        batch_normalization_momentum (float): The momentum of the running statistics of its batch normalization.
        dtype (torch.dtype): The data type of the parameters.
        device (torch.device): The device of the parameters.

    Returns:
        ConvolutionBlock: The convolutions, keeping the height and width of their input.
    """
    number_layers, output_channels = convolutions
    return ConvolutionBlock(input_channels=input_channels, output_channels=output_channels, bias=True,
                            number_layers=number_layers, kernel_size=3, stride=1, padding=1,
                            feature_map_normalization=feature_map_normalization,
                            batch_normalization_momentum=batch_normalization_momentum, activation=True,
                            dropout_rate=0.0, add_pooling=False, device=device, dtype=dtype, squeeze_and_excite=False,
                            residual=False, pre_activation=False, inception=False)


class MaskHead(nn.Module):
    """
    Predicts the binary mask of the object of every box of one detection level, one [M, M] logit map per class, the
    channel of the detected (or ground truth) class being read by the instance segmenter.
    """

    def __init__(self, head_configuration: Dict[str, Any], mask_resolution: int, upsampling: str, number_classes: int,
                 input_channels: int, feature_map_normalization: str, batch_normalization_momentum: float,
                 input_image_size: int, feature_map_size: int, modules: Dict[str, Any],
                 skip_information: Dict[str, Dict[str, int]], fine_input_channels: int,
                 fusion_convolutions: List[int], dtype: torch.dtype, device: torch.device) -> None:
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
            skip_information (Dict[str, Dict[str, int]]): The 'feature_map_size' and 'number_channels' of every
                backbone skip output pooled at P and concatenated to the pooled features before the convolution
                block ('MaskHead.skip_blocks_at_pool_size'), keyed 'skip_<block>' in concatenation order, empty for
                none.
            fine_input_channels (int): The channels of the fine features of MaskFineInputs concatenated after the
                doublings, 0 without fine features.
            fusion_convolutions (List[int]): The [number_layers, output_channels] of the 3x3 convolutions at the
                mask resolution before the predictor ('MaskHead.fusion_convolutions'), [0, 0] for none.
            dtype (torch.dtype): The data type of the parameters.
            device (torch.device): The device of the parameters.

        Raises:
            ValueError: If the upsampling is not one of VALID_UPSAMPLINGS.
        """
        super(MaskHead, self).__init__()

        # A misspelled upsampling reaching the head without the configuration checks stops here instead of being built
        # as the bilinear one
        if upsampling not in VALID_UPSAMPLINGS:
            raise ValueError(f"'MaskHead.upsampling' must be one of {VALID_UPSAMPLINGS}, got {upsampling!r}.")

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

        # Backbone skip outputs pooled at the pool size with the pooled features, each with its own spatial scale
        self.skip_keys = list(skip_information)
        self.skip_spatial_scales = [skip_information[skip_key]["feature_map_size"] / input_image_size
                                    for skip_key in self.skip_keys]
        skip_channels = sum(skip_information[skip_key]["number_channels"] for skip_key in self.skip_keys)

        # Convolutional feature extraction on the pooled features, at the pool resolution
        self.convolutional_block = ConvolutionBlock(
            input_channels=input_channels + skip_channels, output_channels=convolution_output_channels, bias=True,
            number_layers=self.convolution_number_layers, kernel_size=3, stride=1, padding=1,
            feature_map_normalization=feature_map_normalization,
            batch_normalization_momentum=batch_normalization_momentum, activation=True, dropout_rate=0.0,
            add_pooling=False, device=device, dtype=dtype, squeeze_and_excite=bool(modules["squeeze_and_excite"]),
            residual=bool(modules["residual_nets"]), pre_activation=False, inception=bool(modules["inception"]))

        # Doublings of the resolution, learned by a transposed convolution or interpolated then convolved with a 3x3
        # kernel, each followed by a ReLU
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

        # Fine features concatenated after the doublings, the doubled features being first resized to the mask
        # resolution when the doublings fall short of it; without them the logits are resized instead
        self.fine_input_channels = fine_input_channels
        self.resizes_features = fine_input_channels > 0 and self.upsampled_size != self.mask_size
        self.resizes_logits = fine_input_channels == 0 and self.upsampled_size != self.mask_size
        predictor_input_channels = convolution_output_channels + fine_input_channels

        # Fusion convolutions at the mask resolution, mixing the doubled features with the fine features, none by
        # default so that the predictor follows the doublings as in Mask R-CNN
        self.fusion_number_layers = fusion_convolutions[0]
        self.fusion_block = None
        if self.fusion_number_layers > 0:
            self.fusion_block = build_plain_convolutions(
                input_channels=predictor_input_channels, convolutions=fusion_convolutions,
                feature_map_normalization=feature_map_normalization,
                batch_normalization_momentum=batch_normalization_momentum, dtype=dtype, device=device)
            predictor_input_channels = fusion_convolutions[1]

        # One mask logit map per class
        self.mask_predictor = nn.Conv2d(in_channels=predictor_input_channels, out_channels=number_classes,
                                        kernel_size=1, device=device, dtype=dtype)

    def forward(self, proposal_boxes: torch.Tensor, input_tensor: torch.Tensor,
                skip_feature_maps: Dict[str, torch.Tensor], fine_features: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Predicts the mask logits of every box from the feature map of this level, the backbone skip outputs pooled at
        its pool size and the fine features of its boxes.

        Args:
            proposal_boxes (torch.Tensor): The [R, 5] boxes, the image index of every box followed by its
                [x1, y1, x2, y2] corners in network input pixels.
            input_tensor (torch.Tensor): The [B, C, H, W] feature map of this level.
            skip_feature_maps (Dict[str, torch.Tensor]): The backbone skip outputs keyed 'skip_<block>', holding at
                least those this head pools at its pool size.
            fine_features (Optional[torch.Tensor]): The [R, fine_input_channels, M, M] fine features of the boxes
                (MaskFineInputs), None when the head has none.

        Returns:
            torch.Tensor: The [R, number_classes, M, M] mask logits.

        Raises:
            ValueError: If fine features are given to a head built without them, or missing for a head built with
                them.
        """
        if (fine_features is None) != (self.fine_input_channels == 0):
            raise ValueError(f"The mask head was built for {self.fine_input_channels} channels of fine features, got "
                             f"{'none' if fine_features is None else fine_features.shape[1]}.")

        # Pool every box with the RoIAlign settings of the detection head, exported as the ONNX RoiAlign operator
        pooled_features = roi_align(input=input_tensor, boxes=proposal_boxes,
                                    output_size=(self.roi_align_pool_size, self.roi_align_pool_size),
                                    spatial_scale=self.spatial_scale, sampling_ratio=0, aligned=True)

        # Backbone skip outputs pooled at the same grid and concatenated along the channels (U-Net skips)
        if self.skip_keys:
            pooled_skips = [roi_align(input=skip_feature_maps[skip_key], boxes=proposal_boxes,
                                      output_size=(self.roi_align_pool_size, self.roi_align_pool_size),
                                      spatial_scale=skip_spatial_scale, sampling_ratio=0, aligned=True)
                            for skip_key, skip_spatial_scale in zip(self.skip_keys, self.skip_spatial_scales)]
            pooled_features = torch.cat(tensors=[pooled_features] + pooled_skips, dim=1)

        # Refine the pooled features and double their resolution as many times as fits
        features = self.convolutional_block(input_tensor=pooled_features)
        for upsampling_layer in self.upsampling_layers:
            features = self.activation(input=upsampling_layer(input=features))

        # Fine features joined at the mask resolution, the doubled features resized to it first when needed
        if fine_features is not None:
            if self.resizes_features:
                features = torch_functional.interpolate(input=features, size=(self.mask_size, self.mask_size),
                                                        mode="bilinear", align_corners=False)
            features = torch.cat(tensors=[features, fine_features], dim=1)

        # Fusion convolutions, then one mask logit map per class
        if self.fusion_block is not None:
            features = self.fusion_block(input_tensor=features)
        mask_logits = self.mask_predictor(input=features)

        # Logits resized to the mask resolution when the doublings fall short of it, every level giving M x M masks
        if self.resizes_logits:
            mask_logits = torch_functional.interpolate(input=mask_logits, size=(self.mask_size, self.mask_size),
                                                       mode="bilinear", align_corners=False)
        return mask_logits


class MaskFineInputs(nn.Module):
    """
    Computes the fine features of the mask heads at the mask resolution, shared by every detection level since they
    read maps finer than the levels: the image branch (RoIAlign of the normalized input image at the mask resolution
    followed by plain 3x3 convolutions, 'MaskHead.image_branch_convolutions') and the backbone blocks before their
    pooling pooled at the mask resolution ('MaskHead.skip_blocks_at_mask_resolution'), concatenated in this order.
    Built by the InstanceSegmenter, run once on all the boxes of a batch before the mask heads of their levels.
    """

    def __init__(self, image_branch_convolutions: List[int],
                 mask_resolution_skip_information: Dict[str, Dict[str, int]],
                 image_information: Optional[Dict[str, int]], mask_resolution: int, input_image_size: int,
                 feature_map_normalization: str, batch_normalization_momentum: float, dtype: torch.dtype,
                 device: torch.device) -> None:
        """
        Builds the image branch and lists the backbone skip outputs pooled at the mask resolution.

        Args:
            image_branch_convolutions (List[int]): The [number_layers, output_channels] of the image branch, [0, 0]
                without image branch.
            mask_resolution_skip_information (Dict[str, Dict[str, int]]): The 'feature_map_size' and
                'number_channels' of every backbone skip output pooled at the mask resolution, keyed 'skip_<block>' in
                concatenation order, empty for none.
            image_information (Optional[Dict[str, int]]): The size and channels of the normalized input image
                ('skip_0'), None without image branch.
            mask_resolution (int): The side M of the masks.
            input_image_size (int): The size of the square network input, giving the spatial scales of RoIAlign.
            feature_map_normalization (str): The normalization of the mask heads ("batch", "layer" or "none").
            batch_normalization_momentum (float): The momentum of the running statistics of its batch normalization.
            dtype (torch.dtype): The data type of the parameters.
            device (torch.device): The device of the parameters.

        Raises:
            ValueError: If the image branch is set without the size and channels of the input image.
        """
        super(MaskFineInputs, self).__init__()
        self.mask_size = mask_resolution

        # Image branch on the crop of the normalized input image, one image pixel per input pixel
        self.image_branch_convolutions = list(image_branch_convolutions)
        self.image_branch = None
        self.number_channels = 0
        if image_branch_convolutions[0] > 0:
            if image_information is None:
                raise ValueError("The image branch of the mask heads needs the size and channels of the input image.")
            self.image_branch = build_plain_convolutions(
                input_channels=image_information["number_channels"], convolutions=image_branch_convolutions,
                feature_map_normalization=feature_map_normalization,
                batch_normalization_momentum=batch_normalization_momentum, dtype=dtype, device=device)
            self.number_channels += image_branch_convolutions[1]

        # Backbone skip outputs pooled at the mask resolution, each with its own spatial scale
        self.skip_keys = list(mask_resolution_skip_information)
        self.skip_spatial_scales = [mask_resolution_skip_information[skip_key]["feature_map_size"] / input_image_size
                                    for skip_key in self.skip_keys]
        self.number_channels += sum(mask_resolution_skip_information[skip_key]["number_channels"]
                                    for skip_key in self.skip_keys)

    def forward(self, proposal_boxes: torch.Tensor,
                skip_feature_maps: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Gives the fine features of every box, at the mask resolution.

        Args:
            proposal_boxes (torch.Tensor): The [R, 5] boxes, the image index of every box followed by its
                [x1, y1, x2, y2] corners in network input pixels.
            skip_feature_maps (Dict[str, torch.Tensor]): The backbone skip outputs keyed 'skip_<block>', 'skip_0'
                being the normalized input image.

        Returns:
            Optional[torch.Tensor]: The [R, number_channels, M, M] fine features, None without image branch and skip
                output pooled at the mask resolution.
        """
        fine_features = []

        # Crop of the normalized input image at the mask resolution, through the plain convolutions
        if self.image_branch is not None:
            image_crops = roi_align(input=skip_feature_maps[f"{SKIP_OUTPUT_PREFIX}0"], boxes=proposal_boxes,
                                    output_size=(self.mask_size, self.mask_size), spatial_scale=1.0, sampling_ratio=0,
                                    aligned=True)
            fine_features.append(self.image_branch(input_tensor=image_crops))

        # Backbone skip outputs pooled at the mask resolution, as they are
        for skip_key, skip_spatial_scale in zip(self.skip_keys, self.skip_spatial_scales):
            fine_features.append(roi_align(input=skip_feature_maps[skip_key], boxes=proposal_boxes,
                                           output_size=(self.mask_size, self.mask_size),
                                           spatial_scale=skip_spatial_scale, sampling_ratio=0, aligned=True))
        if not fine_features:
            return None
        return torch.cat(tensors=fine_features, dim=1)
