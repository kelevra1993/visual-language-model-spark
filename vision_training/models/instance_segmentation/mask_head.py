"""
Mask branch of the instance segmenter: the MaskHead of one detection level pools every box from the feature map of its
level with RoIAlign at the pool size of that level, runs the pooled features through a ConvolutionBlock, doubles their
resolution as many times as fits in the mask resolution (transposed convolutions, or bilinear upsamplings followed by a
convolution) and predicts one mask logit map per class, as in Mask R-CNN, the logits being resized bilinearly to the
mask resolution when the doublings fall short of it, so that every level gives masks of the same size. Built for every
detection level by the InstanceSegmenter, which trains it on the positive proposals and runs it on the final
detections, and reused by the InferenceInstanceSegmenter of the exports.
With 'MaskHead.shared' (stage 5 of the instance segmentation benchmark), a single MaskHead reads every level, as in
Mask R-CNN: it pools the boxes of every level from the map of that level at its own spatial scale
(pool_box_features), then runs its layers once on the pooled features of all the levels
(predict_from_pooled_features), so that it trains on every positive proposal of the batch.
The mask heads can also read finer inputs than the map of their level (stage 3b of the instance segmentation
benchmark): backbone blocks before their pooling pooled at the pool size and concatenated before the convolutions
(U-Net skips), and the fine features of MaskFineInputs, shared by the levels (the image branch, RoIAlign of the
normalized input image followed by plain convolutions, and backbone blocks pooled at the mask resolution), concatenated
after the doublings and mixed by the fusion convolutions of the head before its predictor.
With 'MaskHead.mask_scoring' (stage 8 of the instance segmentation benchmark, Mask Scoring R-CNN,
https://arxiv.org/abs/1903.00241), every MaskHead also holds a MaskIoUHead predicting the intersection over union of
the mask of every box with its object (predict_masks_and_mask_iou), from which the instance segmenter makes the mask
score of every detection, its class score times its predicted mask IoU.
"""

import torch

from typing import Any, Dict, List, Optional, Tuple
from torch import nn
from torch.nn import functional as torch_functional
from torchvision.ops import roi_align

from vision_training.configuration_processing.configuration_values import VALID_UPSAMPLINGS
from vision_training.models.backbones.backbone import SKIP_OUTPUT_PREFIX
from vision_training.models.components.convolution_block import ConvolutionBlock
from vision_training.models.components.fully_connected_block import build_fully_connected_block

# Standard deviation of the normal initialisation of the mask IoU predictor, its bias starting at 0, as in the official
# code of Mask Scoring R-CNN, so that the predicted IoUs start near 0
MASK_IOU_PREDICTOR_INITIALISATION_STANDARD_DEVIATION = 0.01


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
    Predicts the binary mask of the object of every box of the detection levels it reads, one [M, M] logit map per
    class, the channel of the detected (or ground truth) class being read by the instance segmenter. A head reads one
    level, or every level when the mask head is shared by the levels ('MaskHead.shared'), and holds a mask IoU head
    predicting the IoU of its masks when they are scored ('MaskHead.mask_scoring').
    """

    def __init__(self, head_configuration: Dict[str, Any], mask_resolution: int, upsampling: str, number_classes: int,
                 input_channels: int, feature_map_normalization: str, batch_normalization_momentum: float,
                 input_image_size: int, feature_map_sizes: Dict[str, int], modules: Dict[str, Any],
                 skip_information: Dict[str, Dict[str, int]], fine_input_channels: int,
                 fusion_convolutions: List[int], mask_scoring: Dict[str, Any], dtype: torch.dtype,
                 device: torch.device) -> None:
        """
        Builds the mask head of one detection level, or the mask head shared by every level, from the 'MaskHead'
        section of the configuration, with its mask IoU head when the masks are scored.

        Args:
            head_configuration (Dict[str, Any]): The 'roi_align_pool_size' P and the [number_layers, output_channels]
                'convolutions' of its convolution block, those of its level ('DetectionLevels.<level>.mask_head'),
                the same on every level for the shared head.
            mask_resolution (int): The side M of the masks, the same on every level, at least P.
            upsampling (str): The doubling of the pooled features: "transposed" (a 2x2 transposed convolution) or
                "bilinear" (a bilinear upsampling followed by a 3x3 convolution).
            number_classes (int): The number of classes, background included, one mask channel per class.
            input_channels (int): The number of channels of the feature maps it reads, the same on every level for
                the shared head.
            feature_map_normalization (str): The normalization of the convolution block ("batch", "layer" or "none").
            batch_normalization_momentum (float): The momentum of the running statistics of its batch normalization.
            input_image_size (int): The size of the square network input, giving the spatial scale of RoIAlign.
            feature_map_sizes (Dict[str, int]): The size of the square feature map of every level it reads, keyed by
                level: its own level alone, or every level for the shared head.
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
            mask_scoring (Dict[str, Any]): The 'MaskHead.mask_scoring' section: 'activated', and the 'convolutions'
                and 'fully_connected' layers of the mask IoU head built when it is activated.
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

        # Mask geometry of the head: its pool size, the doublings that fit in the mask resolution and their size
        self.number_classes = number_classes
        self.feature_map_sizes = dict(feature_map_sizes)
        self.level_names = list(feature_map_sizes)
        self.roi_align_pool_size = head_configuration["roi_align_pool_size"]
        self.mask_size = mask_resolution
        self.upsampling = upsampling
        self.number_upsamplings = count_mask_upsamplings(roi_align_pool_size=self.roi_align_pool_size,
                                                         mask_resolution=mask_resolution)
        self.upsampled_size = self.roi_align_pool_size * 2 ** self.number_upsamplings
        self.convolution_number_layers, convolution_output_channels = head_configuration["convolutions"]

        # Ratio mapping the boxes from network input pixels to the cells of the map of every level, as in the
        # detection head
        self.spatial_scales = {level_name: feature_map_size / input_image_size
                               for level_name, feature_map_size in feature_map_sizes.items()}

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

        # Mask IoU head reading the output of the convolution block with the mask of the class of every box, None
        # when the masks are not scored
        self.mask_iou_head = None
        if mask_scoring["activated"]:
            self.mask_iou_head = MaskIoUHead(
                input_channels=convolution_output_channels, roi_align_pool_size=self.roi_align_pool_size,
                mask_resolution=mask_resolution, convolutions=mask_scoring["convolutions"],
                fully_connected=mask_scoring["fully_connected"], number_classes=number_classes, dtype=dtype,
                device=device)

    def forward(self, proposal_boxes: torch.Tensor, input_tensor: torch.Tensor,
                skip_feature_maps: Dict[str, torch.Tensor], fine_features: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Predicts the mask logits of every box from the feature map of the one level this head reads, the backbone skip
        outputs pooled at its pool size and the fine features of its boxes. The shared head, which reads several
        levels, pools the boxes of every level with pool_box_features and predicts them at once with
        predict_from_pooled_features instead.

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
            ValueError: If the head reads more than one level, the map of the boxes being then ambiguous, or if the
                fine features do not match the head (predict_from_pooled_features).
        """
        if len(self.level_names) != 1:
            raise ValueError(f"The mask head reads the levels {self.level_names}: pool the boxes of every level with "
                             f"pool_box_features, then predict them with predict_from_pooled_features.")
        pooled_features = self.pool_box_features(proposal_boxes=proposal_boxes, input_tensor=input_tensor,
                                                 skip_feature_maps=skip_feature_maps, level_name=self.level_names[0])
        return self.predict_from_pooled_features(pooled_features=pooled_features, fine_features=fine_features)

    def pool_box_features(self, proposal_boxes: torch.Tensor, input_tensor: torch.Tensor,
                          skip_feature_maps: Dict[str, torch.Tensor], level_name: str) -> torch.Tensor:
        """
        Pools every box from the feature map of one level at the spatial scale of that level, with the backbone skip
        outputs pooled at the same grid, the input of the convolution block of the head.

        Args:
            proposal_boxes (torch.Tensor): The [R, 5] boxes, the image index of every box followed by its
                [x1, y1, x2, y2] corners in network input pixels.
            input_tensor (torch.Tensor): The [B, C, H, W] feature map of the level.
            skip_feature_maps (Dict[str, torch.Tensor]): The backbone skip outputs keyed 'skip_<block>', holding at
                least those this head pools at its pool size.
            level_name (str): The detection level of the feature map, one of the levels the head reads.

        Returns:
            torch.Tensor: The [R, C + skip channels, P, P] pooled features.

        Raises:
            ValueError: If the head does not read the level, whose spatial scale it does not know.
        """
        if level_name not in self.spatial_scales:
            raise ValueError(f"The mask head reads the levels {self.level_names}, not level {level_name!r}.")

        # Pool every box with the RoIAlign settings of the detection head, exported as the ONNX RoiAlign operator
        pooled_features = roi_align(input=input_tensor, boxes=proposal_boxes,
                                    output_size=(self.roi_align_pool_size, self.roi_align_pool_size),
                                    spatial_scale=self.spatial_scales[level_name], sampling_ratio=0, aligned=True)

        # Backbone skip outputs pooled at the same grid and concatenated along the channels (U-Net skips)
        if self.skip_keys:
            pooled_skips = [roi_align(input=skip_feature_maps[skip_key], boxes=proposal_boxes,
                                      output_size=(self.roi_align_pool_size, self.roi_align_pool_size),
                                      spatial_scale=skip_spatial_scale, sampling_ratio=0, aligned=True)
                            for skip_key, skip_spatial_scale in zip(self.skip_keys, self.skip_spatial_scales)]
            pooled_features = torch.cat(tensors=[pooled_features] + pooled_skips, dim=1)
        return pooled_features

    def predict_from_pooled_features(self, pooled_features: torch.Tensor,
                                     fine_features: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Predicts the mask logits of pooled boxes, whatever the levels they were pooled from: the convolution block,
        then the doublings, the fine features joined at the mask resolution, the fusion convolutions and the predictor
        (predict_mask_logits).

        Args:
            pooled_features (torch.Tensor): The [R, C + skip channels, P, P] pooled features (pool_box_features).
            fine_features (Optional[torch.Tensor]): The [R, fine_input_channels, M, M] fine features of the boxes
                (MaskFineInputs), in the order of the pooled features, None when the head has none.

        Returns:
            torch.Tensor: The [R, number_classes, M, M] mask logits.
        """
        box_features = self.convolutional_block(input_tensor=pooled_features)
        return self.predict_mask_logits(box_features=box_features, fine_features=fine_features)

    def predict_masks_and_mask_iou(self, pooled_features: torch.Tensor, fine_features: Optional[torch.Tensor],
                                   class_indices: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predicts the mask logits of pooled boxes like predict_from_pooled_features and, when the masks are scored, the
        intersection over union of the mask of the given class of every box with its object (the ground truth class
        for the mask IoU loss of the proposals, the detected class for the mask scores of the detections): the mask
        IoU head reads the output of the convolution block with the mask probabilities of that class, as in Mask
        Scoring R-CNN, neither of them detached so that the mask IoU loss also trains the mask head.

        Args:
            pooled_features (torch.Tensor): The [R, C + skip channels, P, P] pooled features (pool_box_features).
            fine_features (Optional[torch.Tensor]): The [R, fine_input_channels, M, M] fine features of the boxes
                (MaskFineInputs), in the order of the pooled features, None when the head has none.
            class_indices (torch.Tensor): The [R] int64 class of every box whose mask is scored.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: The [R, number_classes, M, M] mask logits, and the [R]
                predicted mask IoU of the class of every box, None without mask IoU head.
        """
        box_features = self.convolutional_block(input_tensor=pooled_features)
        mask_logits = self.predict_mask_logits(box_features=box_features, fine_features=fine_features)
        if self.mask_iou_head is None:
            return mask_logits, None

        # Mask probabilities of the class of every box, gathered so that the export keeps fixed shapes
        mask_size = mask_logits.shape[-1]
        class_masks = torch.sigmoid(input=torch.gather(
            input=mask_logits, dim=1, index=class_indices.reshape(-1, 1, 1, 1).expand(-1, 1, mask_size, mask_size)))

        # Predicted IoU of every class, the one of the class of the box kept
        class_mask_iou = self.mask_iou_head(box_features=box_features, class_masks=class_masks)
        return mask_logits, torch.gather(input=class_mask_iou, dim=1, index=class_indices.reshape(-1, 1)).squeeze(dim=1)

    def predict_mask_logits(self, box_features: torch.Tensor, fine_features: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Predicts the mask logits of boxes from the output of the convolution block: the doublings, the fine features
        joined at the mask resolution, the fusion convolutions and the predictor.

        Args:
            box_features (torch.Tensor): The [R, convolution channels, P, P] output of the convolution block.
            fine_features (Optional[torch.Tensor]): The [R, fine_input_channels, M, M] fine features of the boxes
                (MaskFineInputs), in the order of the box features, None when the head has none.

        Returns:
            torch.Tensor: The [R, number_classes, M, M] mask logits.

        Raises:
            ValueError: If fine features are given to a head built without them, or missing for a head built with
                them.
        """
        if (fine_features is None) != (self.fine_input_channels == 0):
            raise ValueError(f"The mask head was built for {self.fine_input_channels} channels of fine features, got "
                             f"{'none' if fine_features is None else fine_features.shape[1]}.")

        # Double the resolution of the refined features as many times as fits
        features = box_features
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


class MaskIoUHead(nn.Module):
    """
    MaskIoU head of Mask Scoring R-CNN (https://arxiv.org/abs/1903.00241, section 3.2): predicts, for every class, the
    intersection over union of the predicted mask of a box with its object, from the output of the convolution block
    of its mask head at the pool size P concatenated with the mask probabilities of its class max-pooled from M to P.
    Its 3x3 convolutions end with one of stride 2 and are followed by fully connected layers and one output per class,
    without normalization as in the paper ([4, 256] and [1024, 1024] there). Built by every MaskHead when
    'MaskHead.mask_scoring' is activated, so that it follows one head per level or the head shared by the levels.
    """

    def __init__(self, input_channels: int, roi_align_pool_size: int, mask_resolution: int, convolutions: List[int],
                 fully_connected: List[int], number_classes: int, dtype: torch.dtype, device: torch.device) -> None:
        """
        Builds the convolutions, fully connected layers and predictor of the mask IoU head, initialised as in the
        official code of Mask Scoring R-CNN (Kaiming normal convolutions, Kaiming uniform fully connected layers of
        gain 1, a predictor of standard deviation MASK_IOU_PREDICTOR_INITIALISATION_STANDARD_DEVIATION).

        Args:
            input_channels (int): The output channels of the convolution block of the mask head.
            roi_align_pool_size (int): The pool size P of the mask head, the side of its box features.
            mask_resolution (int): The side M of the masks, a multiple of P.
            convolutions (List[int]): The [number_layers, output_channels] of the 3x3 convolutions, the last one of
                stride 2.
            fully_connected (List[int]): The output size of every hidden fully connected layer, at least one.
            number_classes (int): The number of classes, background included, one predicted IoU per class.
            dtype (torch.dtype): The data type of the parameters.
            device (torch.device): The device of the parameters.

        Raises:
            ValueError: If the mask resolution is not a multiple of the pool size, the masks being max-pooled to it.
        """
        super(MaskIoUHead, self).__init__()
        if mask_resolution % roi_align_pool_size != 0:
            raise ValueError(f"The mask IoU head max-pools the masks of side {mask_resolution} to the pool size "
                             f"{roi_align_pool_size}, which must divide it.")
        self.mask_pooling_size = mask_resolution // roi_align_pool_size
        self.convolutions = list(convolutions)
        self.fully_connected = list(fully_connected)

        # 3x3 convolutions with a ReLU on the box features and the pooled mask, the last one halving the resolution
        number_layers, output_channels = convolutions
        convolution_layers, current_channels = [], input_channels + 1
        for layer_index in range(number_layers):
            convolution_layers += [nn.Conv2d(in_channels=current_channels, out_channels=output_channels, kernel_size=3,
                                             stride=2 if layer_index == number_layers - 1 else 1, padding=1,
                                             device=device, dtype=dtype), nn.ReLU()]
            current_channels = output_channels
        self.convolution_layers = nn.Sequential(*convolution_layers)

        # Fully connected layers on the flattened features, then one IoU per class
        output_size = (roi_align_pool_size - 1) // 2 + 1
        self.fully_connected_block = build_fully_connected_block(
            input_dimension=output_channels * output_size * output_size, dimensions=self.fully_connected,
            dropout_rate=0.0, add_dropout_layers=False, normalization="none", batch_normalization_momentum=0.1,
            device=device, dtype=dtype)
        self.iou_predictor = nn.Linear(in_features=self.fully_connected[-1], out_features=number_classes,
                                       device=device, dtype=dtype)

        # Initialisation of the official code: the convolutions for their ReLU, the predictor near 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(tensor=module.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(tensor=module.bias)
            elif isinstance(module, nn.Linear) and module is not self.iou_predictor:
                nn.init.kaiming_uniform_(tensor=module.weight, a=1)
                nn.init.zeros_(tensor=module.bias)
        nn.init.normal_(tensor=self.iou_predictor.weight, mean=0.0,
                        std=MASK_IOU_PREDICTOR_INITIALISATION_STANDARD_DEVIATION)
        nn.init.zeros_(tensor=self.iou_predictor.bias)

    def forward(self, box_features: torch.Tensor, class_masks: torch.Tensor) -> torch.Tensor:
        """
        Predicts the mask IoU of every class of every box.

        Args:
            box_features (torch.Tensor): The [R, input_channels, P, P] output of the convolution block of the mask
                head.
            class_masks (torch.Tensor): The [R, 1, M, M] mask probabilities of the class of every box.

        Returns:
            torch.Tensor: The [R, number_classes] predicted mask IoU, not bounded.
        """
        # Masks max-pooled to the pool size, then joined to the box features
        pooled_masks = torch_functional.max_pool2d(input=class_masks, kernel_size=self.mask_pooling_size,
                                                   stride=self.mask_pooling_size)
        features = torch.cat(tensors=[box_features, pooled_masks.to(dtype=box_features.dtype)], dim=1)

        # Convolutions, fully connected layers and the IoU of every class
        features = self.convolution_layers(input=features)
        return self.iou_predictor(input=self.fully_connected_block(input=features.flatten(start_dim=1)))


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
