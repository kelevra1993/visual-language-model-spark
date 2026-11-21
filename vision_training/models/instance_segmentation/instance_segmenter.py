"""
Model of the instance segmentation task: the InstanceSegmenter is the TwoStageDetector of the object detection task
with one MaskHead per detection level, or with a single MaskHead read by every level as in Mask R-CNN
('MaskHead.shared', stage 5 of the instance segmentation benchmark). It reuses every stage of the detector through
forward_detection_stages, trains its mask heads on the positive sampled proposals and predicts the mask of every final
detection. Its mask heads can also read finer inputs than the levels (the image branch and the backbone skips of
MaskFineInputs and MaskHead), whose backbone outputs it requests through list_backbone_skip_blocks. With
'MaskHead.mask_scoring' (Mask Scoring R-CNN, stage 8 of the instance segmentation benchmark), its mask heads also
predict the IoU of every mask with its object, trained by the mask IoU loss, and every final detection gets a mask
score, its class score times its predicted mask IoU, which ranks the masks of the mask average precision. Registered
in model_registry.py for the "instance_segmentation" task and wrapped by inference_instance_segmenter.py for the
exports.
"""

import torch

from torch import nn
from rich.console import Console
from rich.panel import Panel
from typing import Any, Dict, List, Optional, Tuple

from vision_training.losses.instance_segmentation_losses import compute_mask_iou_loss, compute_mask_loss
from vision_training.models.backbones.backbone import SKIP_OUTPUT_PREFIX
from vision_training.models.components.convolution_block import BATCH_NORMALIZATION_MOMENTUM
from vision_training.models.detection.two_stage_detector import TwoStageDetector
from vision_training.models.instance_segmentation.mask_head import MaskFineInputs, MaskHead
from vision_training.models.instance_segmentation.mask_operations import (build_mask_iou_targets,
                                                                          build_mask_targets, compute_mask_scores)

# Key of the mask head shared by every level in the mask heads dictionary, which otherwise holds one head per level
# keyed by its detection block, so that the transfer, the optimizer groups and the size summary read both alike
SHARED_MASK_HEAD_KEY = "shared"


class InstanceSegmenter(TwoStageDetector):
    """
    Detects the objects of an image like the two-stage detector and predicts the binary mask of every detected object,
    every proposal and every detection using the feature map of its own detection level, read by the mask head of
    that level or by the mask head shared by every level.
    """

    def __init__(self, configuration: Dict[str, Any], device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        Builds the two-stage detector, then the mask head of every detection level, or the mask head shared by every
        level, from the 'MaskHead' section and the mask heads of the levels in 'DetectionLevels'.

        Args:
            configuration (Dict[str, Any]): The complete configuration dictionary of the experiment.
            device (torch.device): The device for parameter allocation.
            dtype (torch.dtype): The data type for parameter allocation.
        """
        super(InstanceSegmenter, self).__init__(configuration=configuration, device=device, dtype=dtype)

        # Side of the masks of every level, and the probability from which a pixel belongs to its object
        self.mask_head_configuration = configuration["MaskHead"]
        self.mask_size = self.mask_head_configuration["mask_resolution"]
        self.mask_threshold = self.mask_head_configuration["inference"]["mask_threshold"]

        # One mask head read by every level instead of one head per level, and the mask IoU heads scoring the masks
        self.shared_mask_head = bool(self.mask_head_configuration["shared"])
        self.mask_scoring = bool(self.mask_head_configuration["mask_scoring"]["activated"])

        # Sizes and channels of the backbone skip outputs the mask heads read, empty without fine inputs
        self.skip_output_information = self.backbone.compute_skip_output_information()

        # Fine features shared by the levels (image branch, skips at the mask resolution), then one mask head per
        # detection level, built on the feature map of that level, or one head shared by every level
        feature_map_normalization = self.mask_head_configuration["normalization"]["feature_map_normalization"]
        image_branch_convolutions = self.mask_head_configuration["image_branch_convolutions"]
        image_information = (self.skip_output_information[f"{SKIP_OUTPUT_PREFIX}0"]
                             if image_branch_convolutions[0] > 0 else None)
        self.mask_fine_inputs = MaskFineInputs(
            image_branch_convolutions=image_branch_convolutions,
            mask_resolution_skip_information=self.select_skip_information(
                block_indices=self.mask_head_configuration["skip_blocks_at_mask_resolution"]),
            image_information=image_information,
            mask_resolution=self.mask_size, input_image_size=self.input_image_size,
            feature_map_normalization=feature_map_normalization,
            batch_normalization_momentum=BATCH_NORMALIZATION_MOMENTUM, dtype=self.dtype, device=self.device)
        self.mask_heads_dictionary = self._build_mask_heads_dictionary(
            mask_head_configuration=self.mask_head_configuration,
            detection_levels_configuration=self.detection_levels_configuration)

    def list_backbone_skip_blocks(self, configuration: Dict[str, Any]) -> List[int]:
        """
        Lists the backbone blocks the mask heads read before their pooling, called by the TwoStageDetector before it
        builds the backbone: the blocks pooled at the pool size and at the mask resolution, and 0 (the normalized
        input image) for the image branch.

        Args:
            configuration (Dict[str, Any]): The complete configuration dictionary of the experiment.

        Returns:
            List[int]: The sorted skip blocks, empty without fine inputs.
        """
        mask_head_configuration = configuration["MaskHead"]
        skip_blocks = set(mask_head_configuration["skip_blocks_at_pool_size"]) | \
            set(mask_head_configuration["skip_blocks_at_mask_resolution"])
        if mask_head_configuration["image_branch_convolutions"][0] > 0:
            skip_blocks.add(0)
        return sorted(skip_blocks)

    def select_skip_information(self, block_indices: List[int]) -> Dict[str, Dict[str, int]]:
        """
        Gives the size and channels of the skip outputs of some backbone blocks, in the order of the configuration, for
        the RoIAlign and the layers of the mask heads reading them.

        Args:
            block_indices (List[int]): The backbone blocks, as listed in the 'MaskHead' section.

        Returns:
            Dict[str, Dict[str, int]]: The 'feature_map_size' and 'number_channels' of every block, keyed
                'skip_<block>'.
        """
        return {f"{SKIP_OUTPUT_PREFIX}{block_index}":
                self.skip_output_information[f"{SKIP_OUTPUT_PREFIX}{block_index}"] for block_index in block_indices}

    def _build_mask_heads_dictionary(self, mask_head_configuration: Dict[str, Any],
                                     detection_levels_configuration: Dict[str, Any]) -> nn.ModuleDict:
        """
        Builds the mask head of every detection level, with the pool size and convolutions of that level, the channel
        count and size of its backbone feature map, like the detection heads, and the mask resolution, upsampling,
        skips at the pool size, fine features and fusion convolutions shared by every level. With 'MaskHead.shared',
        a single head reads every level instead, which needs the same mask head settings and the same channel count on
        every level (the outputs of a neck, or backbone blocks of equal channels).

        Args:
            mask_head_configuration (Dict[str, Any]): The 'MaskHead' section of the configuration.
            detection_levels_configuration (Dict[str, Any]): The 'DetectionLevels' section, giving the RoIAlign
                                                             pool size and the convolutions of the mask head of
                                                             every level.

        Returns:
            nn.ModuleDict: The mask head of every level, keyed by the detection block index, or the shared head
                alone, keyed SHARED_MASK_HEAD_KEY.

        Raises:
            ValueError: If the shared head is asked for levels of different mask head settings or channel counts.
        """
        mask_heads_dictionary = nn.ModuleDict()
        pool_size_skip_information = self.select_skip_information(
            block_indices=mask_head_configuration["skip_blocks_at_pool_size"])
        level_mask_heads = {index_string: detection_levels_configuration[index_string]["mask_head"]
                            for index_string in self.detection_input_information}
        level_channels = {index_string: level_information["number_channels"]
                          for index_string, level_information in self.detection_input_information.items()}

        # Heads to build: the levels each head reads, keyed by the name of the head
        if self.shared_mask_head:
            first_level = next(iter(level_mask_heads))
            if any(level_mask_head != level_mask_heads[first_level] for level_mask_head in level_mask_heads.values()):
                raise ValueError(f"'MaskHead.shared' needs the same 'mask_head' on every detection level, got "
                                 f"{level_mask_heads}.")
            if len(set(level_channels.values())) != 1:
                raise ValueError(f"'MaskHead.shared' needs the same channel count on every detection level (a neck, "
                                 f"or backbone blocks of equal channels), got {level_channels}.")
            head_levels = {SHARED_MASK_HEAD_KEY: list(level_mask_heads)}
        else:
            head_levels = {index_string: [index_string] for index_string in level_mask_heads}

        # Iterate over the heads, the levels in the order of the backbone outputs
        for head_name, level_names in head_levels.items():
            mask_heads_dictionary[head_name] = MaskHead(
                head_configuration=level_mask_heads[level_names[0]],
                mask_resolution=mask_head_configuration["mask_resolution"],
                upsampling=mask_head_configuration["upsampling"],
                number_classes=self.number_classes, input_channels=level_channels[level_names[0]],
                feature_map_normalization=mask_head_configuration["normalization"]["feature_map_normalization"],
                batch_normalization_momentum=BATCH_NORMALIZATION_MOMENTUM, input_image_size=self.input_image_size,
                feature_map_sizes={level_name: self.detection_input_information[level_name]["feature_map_size"]
                                   for level_name in level_names},
                modules=mask_head_configuration["modules"], skip_information=pool_size_skip_information,
                fine_input_channels=self.mask_fine_inputs.number_channels,
                fusion_convolutions=mask_head_configuration["fusion_convolutions"],
                mask_scoring=mask_head_configuration["mask_scoring"], dtype=self.dtype, device=self.device)
        return mask_heads_dictionary

    def forward(self, input_tensor: torch.Tensor, ground_truth_bounding_boxes: Optional[List[torch.Tensor]] = None,
                ground_truth_labels: Optional[List[torch.Tensor]] = None,
                ground_truth_masks: Optional[List[torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Runs the two-stage detector, then the mask heads: on the positive sampled proposals with ground truth, giving
        the mask loss (and the mask IoU loss when the masks are scored), and in evaluation mode on the final
        detections, giving their masks (and their mask scores).

        Args:
            input_tensor (torch.Tensor): The [B, 3, H, W] input images.
            ground_truth_bounding_boxes (Optional[List[torch.Tensor]]): The [N_i, 4] ground truth boxes of every image.
            ground_truth_labels (Optional[List[torch.Tensor]]): The [N_i] ground truth labels of every image.
            ground_truth_masks (Optional[List[torch.Tensor]]): The [N_i, H, W] binary ground truth masks of every image,
                in the order of its boxes.

        Returns:
            Dict[str, Any]: The output dictionary of the two-stage detector, with the 'mask_loss' (and the
                'mask_iou_loss' when the masks are scored) when the ground truth is given, and in evaluation mode the
                'final_detection_masks' [B, K, M, M], the mask probabilities of the detected class of every final
                detection (0 for the padding detections), with their 'final_detection_mask_scores' [B, K] when the
                masks are scored (compute_mask_scores, 0 for the padding detections).

        Raises:
            ValueError: If ground truth boxes are given without their masks.
        """
        if ground_truth_bounding_boxes is not None and ground_truth_masks is None:
            raise ValueError("The instance segmenter needs the ground truth masks with the ground truth boxes.")

        # Every stage of the two-stage detector, keeping the feature maps, the skip outputs of the backbone and the
        # sampled proposals
        detection_stages = self.forward_detection_stages(input_tensor=input_tensor,
                                                         ground_truth_bounding_boxes=ground_truth_bounding_boxes,
                                                         ground_truth_labels=ground_truth_labels)
        feature_maps = detection_stages["backbone_output_tensor_dictionary"]
        skip_feature_maps = detection_stages["backbone_skip_dictionary"]
        model_output_dictionary = detection_stages["model_output_dictionary"]

        # Mask losses of the positive sampled proposals
        if ground_truth_bounding_boxes is not None:
            model_output_dictionary.update(self.compute_proposal_mask_loss(
                feature_maps=feature_maps, skip_feature_maps=skip_feature_maps,
                aggregated_detections=detection_stages["aggregated_detections_dictionary"],
                ground_truth_masks=ground_truth_masks))

        # Masks of the final detections, and their mask scores when the masks are scored
        if not self.training:
            detection_masks, detection_mask_iou = self.predict_detection_masks(
                feature_maps=feature_maps, skip_feature_maps=skip_feature_maps,
                model_output_dictionary=model_output_dictionary)
            model_output_dictionary["final_detection_masks"] = detection_masks
            if detection_mask_iou is not None:
                model_output_dictionary["final_detection_mask_scores"] = compute_mask_scores(
                    detection_scores=model_output_dictionary["final_detection_scores"],
                    predicted_mask_iou=detection_mask_iou)
        return model_output_dictionary

    def compute_proposal_mask_loss(self, feature_maps: Dict[str, torch.Tensor],
                                   skip_feature_maps: Dict[str, torch.Tensor], aggregated_detections: Dict[str, Any],
                                   ground_truth_masks: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Runs the mask heads on the positive sampled proposals of every level (predict_box_masks) and compares their
        masks of the ground truth class with the masks of their matched ground truths cropped to the proposals, and,
        when the masks are scored, their predicted mask IoU of the ground truth class with the IoU of their masks
        thresholded at the mask threshold (build_mask_iou_targets), on every positive proposal as in Mask Scoring
        R-CNN. The fine features shared by the levels are computed once on the positive proposals of all the levels,
        so that the batch statistics of the image branch cover every box of the batch, as those of the shared mask
        head do.

        Args:
            feature_maps (Dict[str, torch.Tensor]): The backbone feature maps, keyed by detection level.
            skip_feature_maps (Dict[str, torch.Tensor]): The skip outputs of the backbone keyed 'skip_<block>', empty
                without fine inputs.
            aggregated_detections (Dict[str, Any]): The sampled proposals of the detector (aggregate_detection_outputs),
                with their labels, levels and matched ground truths.
            ground_truth_masks (List[torch.Tensor]): The [N_i, H, W] binary ground truth masks of every image.

        Returns:
            Dict[str, torch.Tensor]: The scalar 'mask_loss', and the scalar 'mask_iou_loss' when the masks are scored.
        """
        positive_mask = aggregated_detections["sliced_labels"] > 0
        level_proposals, labels, matched_indices = {}, [], []

        # Positive proposals of every level, in level order
        for index_string in feature_maps:
            level_indices = torch.nonzero(input=positive_mask & (aggregated_detections["sliced_origins"] ==
                                                                 int(index_string))).flatten()
            level_proposals[index_string] = aggregated_detections["sliced_proposals"][level_indices]
            labels.append(aggregated_detections["sliced_labels"][level_indices])
            matched_indices.append(aggregated_detections["sliced_matched_ground_truth_indices"][level_indices])

        # Fine features of all the positive proposals at once, None without fine inputs
        proposals_tensor = torch.cat(tensors=list(level_proposals.values()), dim=0)
        fine_features = self.mask_fine_inputs(proposal_boxes=proposals_tensor, skip_feature_maps=skip_feature_maps)

        # Mask logits of the positive proposals of every level, in the same order, with their predicted mask IoU of
        # the ground truth class when the masks are scored
        proposal_labels = torch.cat(tensors=labels, dim=0)
        mask_logits, predicted_mask_iou = self.predict_box_masks(
            level_boxes=level_proposals, feature_maps=feature_maps, skip_feature_maps=skip_feature_maps,
            fine_features=fine_features, class_indices=proposal_labels)

        # Targets cropped from the matched ground truth masks, at the resolution of the predicted masks
        proposal_matched_indices = torch.cat(tensors=matched_indices, dim=0)
        mask_targets = build_mask_targets(ground_truth_masks=ground_truth_masks, proposal_boxes=proposals_tensor,
                                          matched_ground_truth_indices=proposal_matched_indices,
                                          mask_size=self.mask_size)
        mask_losses = {"mask_loss": compute_mask_loss(mask_logits=mask_logits, labels=proposal_labels,
                                                      mask_targets=mask_targets)}

        # IoU of the masks of the ground truth class thresholded as in the evaluation with their objects, the target of
        # the predicted mask IoU
        if predicted_mask_iou is not None:
            proposal_indices = torch.arange(mask_logits.shape[0], device=mask_logits.device)
            predicted_masks = torch.sigmoid(input=mask_logits[proposal_indices, proposal_labels].detach().float()) >= \
                self.mask_threshold
            mask_iou_targets = build_mask_iou_targets(
                predicted_masks=predicted_masks, mask_targets=mask_targets, ground_truth_masks=ground_truth_masks,
                proposal_boxes=proposals_tensor, matched_ground_truth_indices=proposal_matched_indices)
            mask_losses["mask_iou_loss"] = compute_mask_iou_loss(predicted_mask_iou=predicted_mask_iou,
                                                                 mask_iou_targets=mask_iou_targets)
        return mask_losses

    def predict_box_masks(self, level_boxes: Dict[str, torch.Tensor], feature_maps: Dict[str, torch.Tensor],
                          skip_feature_maps: Dict[str, torch.Tensor], fine_features: Optional[torch.Tensor],
                          class_indices: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Gives the mask logits of the boxes of every level, for the mask loss of the proposals and the masks of the
        final detections, with the predicted mask IoU of the given class of every box when the masks are scored: the
        mask head of every level runs on the boxes of its level, even when it has none so that it stays in the graph,
        or the shared mask head pools the boxes of every level from the map of that level and runs its layers once on
        all of them, so that it trains on every positive proposal of the batch.

        Args:
            level_boxes (Dict[str, torch.Tensor]): The [R_l, 5] boxes of every level in the RoIAlign format (the image
                index, then the corners in network input pixels), keyed by level in the order of the feature maps,
                possibly empty.
            feature_maps (Dict[str, torch.Tensor]): The backbone feature maps, keyed by detection level.
            skip_feature_maps (Dict[str, torch.Tensor]): The skip outputs of the backbone keyed 'skip_<block>', empty
                without fine inputs.
            fine_features (Optional[torch.Tensor]): The [sum R_l, fine channels, M, M] fine features of all the boxes,
                in the order of level_boxes, None without fine inputs.
            class_indices (torch.Tensor): The [sum R_l] int64 class whose mask IoU is predicted for every box (the
                ground truth class of the proposals, the detected class of the detections), in the order of
                level_boxes.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: The [sum R_l, number_classes, M, M] mask logits and the
                [sum R_l] predicted mask IoU (None when the masks are not scored), in the order of level_boxes.
        """
        # Shared head: the boxes of every level pooled at the spatial scale of their level, then predicted at once
        if self.shared_mask_head:
            mask_head = self.mask_heads_dictionary[SHARED_MASK_HEAD_KEY]
            pooled_features = torch.cat(tensors=[mask_head.pool_box_features(
                proposal_boxes=boxes, input_tensor=feature_maps[level_name], skip_feature_maps=skip_feature_maps,
                level_name=level_name) for level_name, boxes in level_boxes.items()], dim=0)
            return mask_head.predict_masks_and_mask_iou(pooled_features=pooled_features, fine_features=fine_features,
                                                        class_indices=class_indices)

        # One head per level, each given the rows of the fine features and classes of its own boxes
        mask_logits, predicted_mask_iou, first_row = [], [], 0
        for level_name, boxes in level_boxes.items():
            last_row = first_row + boxes.shape[0]
            mask_head = self.mask_heads_dictionary[level_name]
            pooled_features = mask_head.pool_box_features(proposal_boxes=boxes, input_tensor=feature_maps[level_name],
                                                          skip_feature_maps=skip_feature_maps, level_name=level_name)
            level_mask_logits, level_mask_iou = mask_head.predict_masks_and_mask_iou(
                pooled_features=pooled_features,
                fine_features=None if fine_features is None else fine_features[first_row:last_row],
                class_indices=class_indices[first_row:last_row])
            mask_logits.append(level_mask_logits)
            predicted_mask_iou.append(level_mask_iou)
            first_row = last_row
        return (torch.cat(tensors=mask_logits, dim=0),
                torch.cat(tensors=predicted_mask_iou, dim=0) if self.mask_scoring else None)

    def predict_detection_masks(self, feature_maps: Dict[str, torch.Tensor], skip_feature_maps: Dict[str, torch.Tensor],
                                model_output_dictionary: Dict[str, Any]) -> Tuple[torch.Tensor,
                                                                                  Optional[torch.Tensor]]:
        """
        Runs the mask heads on the final detection boxes, each read at the level its proposal came from
        (predict_box_masks), and keeps the mask probabilities of the detected class, as in Mask R-CNN where the masks
        follow the refined boxes, with the predicted mask IoU of the detected class when the masks are scored. The fine
        features shared by the levels are computed once on the valid detections of all the levels.

        Args:
            feature_maps (Dict[str, torch.Tensor]): The backbone feature maps, keyed by detection level.
            skip_feature_maps (Dict[str, torch.Tensor]): The skip outputs of the backbone keyed 'skip_<block>', empty
                without fine inputs.
            model_output_dictionary (Dict[str, Any]): The outputs of the detector in evaluation mode, with the final
                detections, the proposal of every detection and the level of every proposal.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: The [B, K, M, M] mask probabilities and the [B, K] predicted
                mask IoU (None when the masks are not scored), 0 for the padding detections.
        """
        final_boxes = model_output_dictionary["final_detection_boxes"]
        batch_size, number_detections, _ = final_boxes.shape
        flat_labels = model_output_dictionary["final_detection_labels"].reshape(-1)

        # Level of every detection, read at its proposal, and the valid detections of every image
        detection_levels = torch.gather(input=model_output_dictionary["sliced_origins"], dim=1,
                                        index=model_output_dictionary["final_detection_proposal_indices"]).reshape(-1)
        valid_detections = (torch.arange(number_detections, device=final_boxes.device)[None, :] <
                            model_output_dictionary["number_detections"][:, None]).reshape(-1)

        # Boxes in the RoIAlign format, the image index first
        image_indices = torch.arange(batch_size, device=final_boxes.device, dtype=final_boxes.dtype)
        region_boxes = torch.cat(tensors=[image_indices.repeat_interleave(repeats=number_detections).unsqueeze(dim=1),
                                          final_boxes.reshape(-1, 4)], dim=1)
        detection_masks = torch.zeros(size=(batch_size * number_detections, self.mask_size, self.mask_size),
                                      dtype=final_boxes.dtype, device=final_boxes.device)

        # Fine features of the valid detections at once, with the row of every detection among them (-1 for padding)
        valid_positions = torch.nonzero(input=valid_detections).flatten()
        fine_features = self.mask_fine_inputs(proposal_boxes=region_boxes[valid_positions],
                                              skip_feature_maps=skip_feature_maps)
        fine_rows = torch.full(size=(batch_size * number_detections,), fill_value=-1, dtype=torch.int64,
                               device=final_boxes.device)
        fine_rows[valid_positions] = torch.arange(valid_positions.numel(), device=final_boxes.device)

        # Valid detections of every level, then their masks in level order, the channel of the detected class being
        # kept
        level_indices = {index_string: torch.nonzero(input=valid_detections &
                                                     (detection_levels == int(index_string))).flatten()
                         for index_string in feature_maps}
        ordered_indices = torch.cat(tensors=list(level_indices.values()), dim=0)
        mask_logits, predicted_mask_iou = self.predict_box_masks(
            level_boxes={index_string: region_boxes[indices] for index_string, indices in level_indices.items()},
            feature_maps=feature_maps, skip_feature_maps=skip_feature_maps,
            fine_features=None if fine_features is None else fine_features[fine_rows[ordered_indices]],
            class_indices=flat_labels[ordered_indices])
        class_logits = mask_logits[torch.arange(ordered_indices.numel(), device=final_boxes.device),
                                   flat_labels[ordered_indices]]
        detection_masks[ordered_indices] = torch.sigmoid(input=class_logits).to(dtype=detection_masks.dtype)
        detection_masks = detection_masks.reshape(batch_size, number_detections, self.mask_size, self.mask_size)

        # Predicted mask IoU of the valid detections, 0 for the padding ones
        if predicted_mask_iou is None:
            return detection_masks, None
        detection_mask_iou = torch.zeros(size=(batch_size * number_detections,), dtype=final_boxes.dtype,
                                         device=final_boxes.device)
        detection_mask_iou[ordered_indices] = predicted_mask_iou.to(dtype=detection_mask_iou.dtype)
        return detection_masks, detection_mask_iou.reshape(batch_size, number_detections)

    def print_summary(self) -> None:
        """
        Prints the summary of the two-stage detector followed by the panel of the mask heads.
        """
        super(InstanceSegmenter, self).print_summary()
        self.print_mask_summary()

    def print_mask_summary(self) -> None:
        """
        Prints the mask head of every detection level, or the mask head shared by the levels: its modules, convolution
        layers, feature map size, RoIAlign pool, backbone skips at the pool size, doublings, fusion convolutions, mask
        size, mask IoU head of the scored masks and spatial scale (those of every level for the shared head), under the
        normalization and the upsampling of the mask heads and the fine inputs they share (image branch, backbone skips
        at the mask resolution).
        """
        console = Console(width=80)
        normalization = str(self.mask_head_configuration["normalization"]["feature_map_normalization"]).capitalize()
        summary_text = (f"[bold]Normalization Strategy:[/bold] {normalization}\n"
                        f"[bold]Upsampling:[/bold] {self.mask_head_configuration['upsampling'].capitalize()}\n")

        # Fine inputs shared by the levels, at the mask resolution
        image_layers, image_channels = self.mask_fine_inputs.image_branch_convolutions
        if image_layers > 0:
            summary_text += (f"[bold]Image Branch:[/bold] RoI Align [{self.mask_size}, {self.mask_size}] of the "
                             f"input, {image_layers} convolutions of {image_channels} channels\n")
        if self.mask_fine_inputs.skip_keys:
            summary_text += (f"[bold]Skips At Mask Size:[/bold] backbone blocks "
                             f"{', '.join(key[len(SKIP_OUTPUT_PREFIX):] for key in self.mask_fine_inputs.skip_keys)} "
                             f"before pooling\n")
        summary_text += "\n"

        # One block of lines per detection level, or one for the head shared by the levels
        for index_string, mask_head in self.mask_heads_dictionary.items():
            summary_text += (f"[bold]Mask Head Shared By Blocks {', '.join(mask_head.level_names)}[/bold]\n"
                             if index_string == SHARED_MASK_HEAD_KEY else f"[bold]Mask Block {index_string}[/bold]\n")
            active_module_names = mask_head.convolutional_block.get_active_module_names()
            if active_module_names:
                summary_text += f"  - Modules            : {', '.join(active_module_names)}\n"
            summary_text += f"  - Convolution Layers : {mask_head.convolution_number_layers}\n"
            summary_text += "  - Feature Map Size   : " + ", ".join(
                f"[{feature_map_size}, {feature_map_size}]" for feature_map_size in mask_head.feature_map_sizes.values())
            summary_text += "\n"
            pool_size, upsampled_size = mask_head.roi_align_pool_size, mask_head.upsampled_size
            summary_text += f"  - RoI Align Pool     : [{pool_size}, {pool_size}]\n"
            if mask_head.skip_keys:
                summary_text += (f"  - Skips At Pool Size : backbone blocks "
                                 f"{', '.join(key[len(SKIP_OUTPUT_PREFIX):] for key in mask_head.skip_keys)}\n")
            summary_text += (f"  - Doublings          : {mask_head.number_upsamplings}, to [{upsampled_size}, "
                             f"{upsampled_size}]\n")
            if mask_head.fusion_block is not None:
                summary_text += (f"  - Fusion Layers      : {mask_head.fusion_number_layers}, with "
                                 f"{mask_head.fine_input_channels} channels of fine features\n")

            # Masks resized bilinearly when the doublings fall short of the mask resolution
            resize_note = "" if upsampled_size == mask_head.mask_size else ", resized bilinearly"
            summary_text += f"  - Mask Size          : [{mask_head.mask_size}, {mask_head.mask_size}]{resize_note}\n"

            # Mask IoU head of the scored masks, its convolutions ending with a stride of 2
            if mask_head.mask_iou_head is not None:
                summary_text += (f"  - Mask IoU Head      : {mask_head.mask_iou_head.convolutions} convolutions, "
                                 f"{mask_head.mask_iou_head.fully_connected} fully connected\n")
            summary_text += "  - Spatial Scale      : " + ", ".join(
                f"{spatial_scale:.4f}" for spatial_scale in mask_head.spatial_scales.values()) + "\n\n"

        console.print(Panel(summary_text.rstrip("\n"), title="[bold]Mask Architecture[/bold]", border_style="magenta"))

    def get_size_components(self) -> Dict[str, nn.Module]:
        """
        Lists the components whose size is reported by the model size summary: those of the detector, with the mask
        heads, then the image branch of the mask heads when they have one, before the whole model.

        Returns:
            Dict[str, nn.Module]: The display name and module of every component.
        """
        size_components = super(InstanceSegmenter, self).get_size_components()
        total_model = size_components.pop("Total Model")
        image_branch_component = ({} if self.mask_fine_inputs.image_branch is None else
                                  {"Mask Image Branch": self.mask_fine_inputs.image_branch})
        return {**size_components, "Mask Heads": self.mask_heads_dictionary, **image_branch_component,
                "Total Model": total_model}
