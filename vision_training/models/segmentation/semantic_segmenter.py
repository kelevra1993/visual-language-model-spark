"""
Model of the semantic segmentation task: SemanticSegmenter is a U-Net whose downsampling path is the shared Backbone,
followed by the SegmentationDecoder and the segmentation head giving the logits of every class at every pixel, with
optional deep supervision heads on the intermediate decoder levels during the training, and returns the class logits,
probabilities and losses. Registered in model_registry.py for the "semantic_segmentation" task and wrapped by
inference_segmenter.py for the exports, which leave the deep supervision heads out.
"""

import torch

from rich.console import Console
from rich.panel import Panel
from torch import nn
from typing import Any, Dict, Optional

from vision_training.losses.segmentation_losses import (compute_boundary_loss, compute_dice_loss,
                                                        compute_lovasz_softmax_loss,
                                                        compute_segmentation_cross_entropy)
from vision_training.models.backbones.backbone import build_backbone
from vision_training.models.segmentation.segmentation_decoder import SegmentationDecoder
from vision_training.models.task_model import TaskModel
from vision_training.utilities.mask_morphology import compute_edge_distance_weights


class SemanticSegmenter(TaskModel):
    """
    Model of the semantic segmentation task, a U-Net: the shared Backbone extracts the feature maps of the image, the
    output of every block before its pooling being kept as a skip connection, the SegmentationDecoder brings the final
    feature map back to the input resolution through these skip connections, and the segmentation head (dropout and a
    1x1 convolution) gives the logits of every class, the background being class 0. The head is kept apart from the
    decoder, so that a decoder transferred from another run can serve other classes. With deep supervision, a 1x1
    convolution head on every level of 'SegmentationDecoder.deep_supervision_weights' gives class logits upsampled to
    the input size, trained with the same losses times the weight of its level, so that the gradient reaches the early
    layers more directly; these heads only serve the training and are left out of the predictions and the exports.

    Outputs: 'class_logits' [B, K, H, W], 'class_probabilities' [B, K, H, W] in evaluation mode only (the training
    does not need the softmax of every pixel), and with masks the 'segmentation_loss' (weighted cross entropy),
    'dice_loss', 'lovasz_loss' and 'boundary_loss' of the losses whose weight is not 0, and the
    'deep_supervision_loss' of the deep supervision heads.
    """

    def __init__(self, configuration: Dict[str, Any], verbose: bool = False, device: torch.device = None,
                 dtype: torch.dtype = None) -> None:
        """
        Builds the backbone, the decoder and the segmentation head from the configuration.

        Args:
            configuration (Dict[str, Any]): The model configuration (Data with its label dictionary, Backbone,
                SegmentationDecoder and SegmentationLoss sections).
            verbose (bool): Unused, kept so that every task model is built with the same arguments.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(SemanticSegmenter, self).__init__()
        self.device = device
        self.dtype = dtype
        self.decoder_configuration = configuration["SegmentationDecoder"]
        self.deep_supervision_weights = self.decoder_configuration["deep_supervision_weights"]
        self.loss_configuration = configuration["SegmentationLoss"]
        self.number_classes = len(set(configuration["Data"]["label_dictionary"].values()))

        # Shared feature extractor, the downsampling path of the U-Net
        self.backbone = build_backbone(backbone_configuration=configuration["Backbone"],
                                       input_image_size=configuration["Data"]["image_settings"]["size"],
                                       device=device, dtype=dtype)

        # Upsampling path, built from the channels of the backbone blocks feeding its skip connections
        self.segmentation_decoder = SegmentationDecoder(
            decoder_configuration=self.decoder_configuration,
            block_channels=[block.output_channels for block in self.backbone.convolution_blocks],
            last_max_pooling=self.backbone.convolution_blocks[-1].add_pooling, device=device, dtype=dtype)

        # Class logits of every pixel, the dropout layer being kept at a rate of 0 so that the weight names do not
        # depend on the rate
        self.segmentation_head = nn.Sequential(
            nn.Dropout2d(p=self.decoder_configuration["dropout_rate"]),
            nn.Conv2d(in_channels=self.segmentation_decoder.output_channels, out_channels=self.number_classes,
                      kernel_size=1, device=device, dtype=dtype))

        # Class logits of the intermediate decoder levels given a loss of their own, keyed by backbone block like the
        # decoder levels (none without deep supervision)
        self.deep_supervision_heads = nn.ModuleDict({
            level_key: nn.Conv2d(in_channels=self.segmentation_decoder.convolution_blocks[level_key].output_channels,
                                 out_channels=self.number_classes, kernel_size=1, device=device, dtype=dtype)
            for level_key in self.deep_supervision_weights})

        # Weight of every class in the cross entropy, uniform unless the trainer sets the balanced weights of its
        # training split
        self.register_buffer(name="class_weights",
                             tensor=torch.ones(size=(self.number_classes,), device=device, dtype=dtype))

    def forward(self, input_tensor: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Segments a batch of images, and computes the segmentation losses when the masks are given.

        Args:
            input_tensor (torch.Tensor): The [B, 3, H, W] float BGR images in [0, 255].
            masks (Optional[torch.Tensor]): The [B, H, W] int64 class index of every pixel, IGNORED_LABEL_INDEX on the
                ignored ones, or None.

        Returns:
            Dict[str, Any]: The 'class_logits' [B, K, H, W], the 'class_probabilities' in evaluation mode, and with
                masks the 'segmentation_loss', 'dice_loss', 'lovasz_loss' and 'boundary_loss' whose weight is not 0
                and the 'deep_supervision_loss' with deep supervision heads.
        """
        # The outputs of the intermediate decoder levels are only kept for the losses of the deep supervision heads
        final_feature_map, skip_tensors = self.backbone(input_tensor=input_tensor, return_pre_pooling_outputs=True)
        level_outputs = {}
        if masks is not None and len(self.deep_supervision_heads) > 0:
            decoded_feature_map, level_outputs = self.segmentation_decoder(
                bottleneck_tensor=final_feature_map, skip_tensors=skip_tensors, return_level_outputs=True)
        else:
            decoded_feature_map = self.segmentation_decoder(bottleneck_tensor=final_feature_map,
                                                            skip_tensors=skip_tensors)
        class_logits = self.segmentation_head(decoded_feature_map)

        outputs = {"class_logits": class_logits}
        if not self.training:
            outputs["class_probabilities"] = torch.softmax(input=class_logits, dim=1)
        if masks is None:
            return outputs

        # Losses of the final logits, the closeness of every pixel to the true edges being computed once for all heads
        edge_weights = None
        if self.loss_configuration["boundary_weight"] > 0:
            edge_weights = compute_edge_distance_weights(masks=masks,
                                                         maximum_distance=self.loss_configuration["boundary_distance"])
        outputs.update(self.compute_mask_losses(class_logits=class_logits, masks=masks, edge_weights=edge_weights))

        # Same losses for the logits of every deep supervision head, upsampled to the input size and weighted by level
        if len(self.deep_supervision_heads) > 0:
            deep_supervision_loss = class_logits.new_zeros(size=(), dtype=torch.float32)
            for level_key, supervision_head in self.deep_supervision_heads.items():
                level_logits = torch.nn.functional.interpolate(input=supervision_head(level_outputs[level_key]),
                                                               size=masks.shape[-2:], mode="bilinear",
                                                               align_corners=False)
                level_losses = self.compute_mask_losses(class_logits=level_logits, masks=masks,
                                                        edge_weights=edge_weights)
                deep_supervision_loss = deep_supervision_loss + self.deep_supervision_weights[level_key] * sum(
                    level_losses.values())
            outputs["deep_supervision_loss"] = deep_supervision_loss
        return outputs

    def compute_mask_losses(self, class_logits: torch.Tensor, masks: torch.Tensor,
                            edge_weights: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Computes the losses of one set of class logits against the masks, for the final logits of the segmenter and
        for those of every deep supervision head: the weighted cross entropy, the Dice loss, the Lovász-Softmax loss
        and the boundary loss, each multiplied by its weight of the 'SegmentationLoss' section and left out when its
        weight is 0.

        Args:
            class_logits (torch.Tensor): The [B, K, H, W] class logits at the resolution of the masks.
            masks (torch.Tensor): The [B, H, W] int64 class index of every pixel, IGNORED_LABEL_INDEX on the ignored
                ones.
            edge_weights (Optional[torch.Tensor]): The [B, H, W] closeness of every pixel to the true edges
                (compute_edge_distance_weights), None when the boundary loss is off.

        Returns:
            Dict[str, torch.Tensor]: The 'segmentation_loss', 'dice_loss', 'lovasz_loss' and 'boundary_loss' whose
                weight is not 0.
        """
        losses = {}
        cross_entropy_weight = self.loss_configuration["cross_entropy_weight"]
        dice_weight = self.loss_configuration["dice_weight"]
        lovasz_weight = self.loss_configuration["lovasz_weight"]
        boundary_weight = self.loss_configuration["boundary_weight"]

        # Losses of the regions: the weighted cross entropy, then the Dice and Lovász losses of their overlap
        if cross_entropy_weight > 0:
            losses["segmentation_loss"] = cross_entropy_weight * compute_segmentation_cross_entropy(
                class_logits=class_logits, masks=masks, label_smoothing=self.loss_configuration["label_smoothing"],
                class_weights=self.class_weights, focal_gamma=self.loss_configuration["focal_gamma"])
        if dice_weight > 0:
            losses["dice_loss"] = dice_weight * compute_dice_loss(class_logits=class_logits, masks=masks)
        if lovasz_weight > 0:
            losses["lovasz_loss"] = lovasz_weight * compute_lovasz_softmax_loss(class_logits=class_logits,
                                                                                masks=masks)

        # Loss of the contours, the errors weighted by their closeness to the true edges
        if boundary_weight > 0:
            losses["boundary_loss"] = boundary_weight * compute_boundary_loss(class_logits=class_logits, masks=masks,
                                                                              edge_weights=edge_weights)
        return losses

    def print_summary(self) -> None:
        """
        Prints the backbone summary, then the decoder and the head: for every level, the resolution and channels of
        the upsampled features, of the skip connection and of the level output, and the classes of the head, so that
        the user can check that the last level is at the input resolution.
        """
        self.backbone.print_summary()

        # Resolution of the bottleneck and of the output of every level, every backbone pooling halving the input
        input_image_size = self.backbone.input_image_size
        number_blocks = len(self.backbone.convolution_blocks)
        number_bottleneck_poolings = number_blocks if self.backbone.convolution_blocks[-1].add_pooling else \
            number_blocks - 1
        bottleneck_size = input_image_size // 2 ** number_bottleneck_poolings
        summary_text = (f"[bold]Upsampling[/bold]    : {self.segmentation_decoder.upsampling}\n"
                        f"[bold]Normalization[/bold] : {self.segmentation_decoder.feature_map_normalization}\n"
                        f"[bold]Bottleneck[/bold]    : {bottleneck_size}x{bottleneck_size}x"
                        f"{self.backbone.convolution_blocks[-1].output_channels}\n\n")
        for level_key, convolution_block in self.segmentation_decoder.convolution_blocks.items():
            level_size = input_image_size // 2 ** (int(level_key) - 1)
            summary_text += (f"[bold] - Level Of Block {level_key}[/bold]\n"
                             f"   - Skip Connection : {level_size}x{level_size}x"
                             f"{self.segmentation_decoder.skip_channels[level_key]}\n"
                             f"   - Convolutions    : {convolution_block.number_layers} x "
                             f"{convolution_block.output_channels} channels\n"
                             f"   - Output          : {level_size}x{level_size}x"
                             f"{convolution_block.output_channels}\n\n")
        summary_text += (f"[bold]Head Dropout Rate[/bold] : {self.decoder_configuration['dropout_rate']}\n"
                         f"[bold]Class Logits[/bold]      : {input_image_size}x{input_image_size}x"
                         f"{self.number_classes}")

        # Training-only heads of the intermediate levels, with the weight of their losses
        if len(self.deep_supervision_heads) > 0:
            summary_text += "\n[bold]Deep Supervision[/bold]  : " + ", ".join(
                f"level of block {level_key} x {level_weight:g}"
                for level_key, level_weight in self.deep_supervision_weights.items())
        Console(width=80).print(Panel(summary_text, title="[bold]Segmentation Decoder And Head[/bold]",
                                      border_style="green"))

    def get_size_components(self) -> Dict[str, nn.Module]:
        """
        Lists the backbone, the decoder, the segmentation head, the deep supervision heads when there are some and
        the whole segmenter for the model size summary.

        Returns:
            Dict[str, nn.Module]: The display name and module of every component.
        """
        size_components = {"Backbone": self.backbone, "Segmentation Decoder": self.segmentation_decoder,
                           "Segmentation Head": self.segmentation_head}
        if len(self.deep_supervision_heads) > 0:
            size_components["Deep Supervision Heads"] = self.deep_supervision_heads
        size_components["Total Model"] = self
        return size_components
