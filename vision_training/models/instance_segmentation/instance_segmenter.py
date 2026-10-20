"""
Model of the instance segmentation task: the InstanceSegmenter is the TwoStageDetector of the object detection task
with one MaskHead per detection level, as in Mask R-CNN. It reuses every stage of the detector through
forward_detection_stages, trains its mask heads on the positive sampled proposals and predicts the mask of every final
detection. Registered in model_registry.py for the "instance_segmentation" task and wrapped by
inference_instance_segmenter.py for the exports.
"""

import torch

from torch import nn
from rich.console import Console
from rich.panel import Panel
from typing import Any, Dict, List, Optional

from vision_training.losses.instance_segmentation_losses import compute_mask_loss
from vision_training.models.detection.two_stage_detector import TwoStageDetector
from vision_training.models.instance_segmentation.mask_head import MaskHead
from vision_training.models.instance_segmentation.mask_operations import build_mask_targets


class InstanceSegmenter(TwoStageDetector):
    """
    Detects the objects of an image like the two-stage detector and predicts the binary mask of every detected object,
    every proposal and every detection using the mask head and the feature map of its own detection level.
    """

    def __init__(self, configuration: Dict[str, Any], verbose: bool = False, device: torch.device = None,
                 dtype: torch.dtype = None) -> None:
        """
        Builds the two-stage detector, then the mask head of every detection level from the 'MaskHead' section and
        the mask head of that level in 'DetectionLevels'.

        Args:
            configuration (Dict[str, Any]): The complete configuration dictionary of the experiment.
            verbose (bool): If True, prints the shapes of the intermediate tensors. Defaults to False.
            device (torch.device): The device for parameter allocation.
            dtype (torch.dtype): The data type for parameter allocation.
        """
        super(InstanceSegmenter, self).__init__(configuration=configuration, verbose=verbose, device=device,
                                                dtype=dtype)

        # Side of the masks of every level, and the probability from which a pixel belongs to its object
        self.mask_head_configuration = configuration["MaskHead"]
        self.mask_size = self.mask_head_configuration["mask_resolution"]
        self.mask_threshold = self.mask_head_configuration["inference"]["mask_threshold"]

        # One mask head per detection level, built on the feature map of that level
        self.mask_heads_dictionary = self._build_mask_heads_dictionary(
            mask_head_configuration=self.mask_head_configuration,
            detection_levels_configuration=self.detection_levels_configuration)

    def _build_mask_heads_dictionary(self, mask_head_configuration: Dict[str, Any],
                                     detection_levels_configuration: Dict[str, Any]) -> nn.ModuleDict:
        """
        Builds the mask head of every detection level, with the pool size and convolutions of that level, the channel
        count and size of its backbone feature map, like the detection heads, and the mask resolution and upsampling
        shared by every level.

        Args:
            mask_head_configuration (Dict[str, Any]): The 'MaskHead' section of the configuration.
            detection_levels_configuration (Dict[str, Any]): The 'DetectionLevels' section, giving the RoIAlign
                                                             pool size and the convolutions of the mask head of
                                                             every level.

        Returns:
            nn.ModuleDict: The mask head of every level, keyed by the detection block index.
        """
        mask_heads_dictionary = nn.ModuleDict()

        # Iterate over the detection levels in the order of the backbone outputs
        for index_string, level_information in self.detection_input_information.items():
            mask_heads_dictionary[index_string] = MaskHead(
                head_configuration=detection_levels_configuration[index_string]["mask_head"],
                mask_resolution=mask_head_configuration["mask_resolution"],
                upsampling=mask_head_configuration["upsampling"],
                number_classes=self.number_classes, input_channels=level_information["number_channels"],
                feature_map_normalization=mask_head_configuration["normalization"]["feature_map_normalization"],
                batch_normalization_momentum=mask_head_configuration["normalization"]["batch_normalization_momentum"],
                input_image_size=self.input_image_size, feature_map_size=level_information["feature_map_size"],
                modules=mask_head_configuration["modules"], dtype=self.dtype, device=self.device)
        return mask_heads_dictionary

    def forward(self, input_tensor: torch.Tensor, ground_truth_bounding_boxes: Optional[List[torch.Tensor]] = None,
                ground_truth_labels: Optional[List[torch.Tensor]] = None,
                ground_truth_masks: Optional[List[torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Runs the two-stage detector, then the mask heads: on the positive sampled proposals with ground truth, giving
        the mask loss, and in evaluation mode on the final detections, giving their masks.

        Args:
            input_tensor (torch.Tensor): The [B, 3, H, W] input images.
            ground_truth_bounding_boxes (Optional[List[torch.Tensor]]): The [N_i, 4] ground truth boxes of every image.
            ground_truth_labels (Optional[List[torch.Tensor]]): The [N_i] ground truth labels of every image.
            ground_truth_masks (Optional[List[torch.Tensor]]): The [N_i, H, W] binary ground truth masks of every image,
                in the order of its boxes.

        Returns:
            Dict[str, Any]: The output dictionary of the two-stage detector, with the 'mask_loss' when the ground truth
                is given, and in evaluation mode the 'final_detection_masks' [B, K, M, M], the mask probabilities of
                the detected class of every final detection (0 for the padding detections).

        Raises:
            ValueError: If ground truth boxes are given without their masks.
        """
        if ground_truth_bounding_boxes is not None and ground_truth_masks is None:
            raise ValueError("The instance segmenter needs the ground truth masks with the ground truth boxes.")

        # Every stage of the two-stage detector, keeping the feature maps and the sampled proposals
        detection_stages = self.forward_detection_stages(input_tensor=input_tensor,
                                                         ground_truth_bounding_boxes=ground_truth_bounding_boxes,
                                                         ground_truth_labels=ground_truth_labels)
        feature_maps = detection_stages["backbone_output_tensor_dictionary"]
        model_output_dictionary = detection_stages["model_output_dictionary"]

        # Mask loss of the positive sampled proposals
        if ground_truth_bounding_boxes is not None:
            model_output_dictionary["mask_loss"] = self.compute_proposal_mask_loss(
                feature_maps=feature_maps, aggregated_detections=detection_stages["aggregated_detections_dictionary"],
                ground_truth_masks=ground_truth_masks)

        # Masks of the final detections
        if not self.training:
            model_output_dictionary["final_detection_masks"] = self.predict_detection_masks(
                feature_maps=feature_maps, model_output_dictionary=model_output_dictionary)
        return model_output_dictionary

    def compute_proposal_mask_loss(self, feature_maps: Dict[str, torch.Tensor], aggregated_detections: Dict[str, Any],
                                   ground_truth_masks: List[torch.Tensor]) -> torch.Tensor:
        """
        Runs the mask head of every level on the positive sampled proposals of that level and compares their masks of
        the ground truth class with the masks of their matched ground truths cropped to the proposals.

        Args:
            feature_maps (Dict[str, torch.Tensor]): The backbone feature maps, keyed by detection level.
            aggregated_detections (Dict[str, Any]): The sampled proposals of the detector (aggregate_detection_outputs),
                with their labels, levels and matched ground truths.
            ground_truth_masks (List[torch.Tensor]): The [N_i, H, W] binary ground truth masks of every image.

        Returns:
            torch.Tensor: The scalar mask loss.
        """
        positive_mask = aggregated_detections["sliced_labels"] > 0
        mask_logits, labels, proposals, matched_indices = [], [], [], []

        # Every level predicts the masks of its own positive proposals, even when it has none, to stay in the graph
        for index_string, feature_map in feature_maps.items():
            level_indices = torch.nonzero(input=positive_mask & (aggregated_detections["sliced_origins"] ==
                                                                 int(index_string))).flatten()
            level_proposals = aggregated_detections["sliced_proposals"][level_indices]
            mask_logits.append(self.mask_heads_dictionary[index_string](proposal_boxes=level_proposals,
                                                                        input_tensor=feature_map))
            labels.append(aggregated_detections["sliced_labels"][level_indices])
            proposals.append(level_proposals)
            matched_indices.append(aggregated_detections["sliced_matched_ground_truth_indices"][level_indices])

        # Targets cropped from the matched ground truth masks, at the resolution of the predicted masks
        proposals_tensor = torch.cat(tensors=proposals, dim=0)
        mask_targets = build_mask_targets(ground_truth_masks=ground_truth_masks, proposal_boxes=proposals_tensor,
                                          matched_ground_truth_indices=torch.cat(tensors=matched_indices, dim=0),
                                          mask_size=self.mask_size)
        return compute_mask_loss(mask_logits=torch.cat(tensors=mask_logits, dim=0),
                                 labels=torch.cat(tensors=labels, dim=0), mask_targets=mask_targets)

    def predict_detection_masks(self, feature_maps: Dict[str, torch.Tensor],
                                model_output_dictionary: Dict[str, Any]) -> torch.Tensor:
        """
        Runs the mask head of every level on the final detection boxes whose proposal came from that level, and keeps
        the mask probabilities of the detected class, as in Mask R-CNN where the masks follow the refined boxes.

        Args:
            feature_maps (Dict[str, torch.Tensor]): The backbone feature maps, keyed by detection level.
            model_output_dictionary (Dict[str, Any]): The outputs of the detector in evaluation mode, with the final
                detections, the proposal of every detection and the level of every proposal.

        Returns:
            torch.Tensor: The [B, K, M, M] mask probabilities, 0 for the padding detections.
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

        # Every level predicts the masks of its own detections, the channel of the detected class being kept
        for index_string, feature_map in feature_maps.items():
            level_indices = torch.nonzero(input=valid_detections & (detection_levels == int(index_string))).flatten()
            level_logits = self.mask_heads_dictionary[index_string](proposal_boxes=region_boxes[level_indices],
                                                                    input_tensor=feature_map)
            class_logits = level_logits[torch.arange(level_indices.numel(), device=final_boxes.device),
                                        flat_labels[level_indices]]
            detection_masks[level_indices] = torch.sigmoid(input=class_logits).to(dtype=detection_masks.dtype)
        return detection_masks.reshape(batch_size, number_detections, self.mask_size, self.mask_size)

    def print_summary(self, auxiliary_heads: Optional[Dict[str, nn.Module]] = None) -> None:
        """
        Prints the summary of the two-stage detector followed by the panel of the mask heads.

        Args:
            auxiliary_heads (Optional[Dict[str, nn.Module]]): Passed on to the summary of the detector, the
                trainer of the task having no auxiliary head.
        """
        super(InstanceSegmenter, self).print_summary(auxiliary_heads=auxiliary_heads)
        self.print_mask_summary()

    def print_mask_summary(self) -> None:
        """
        Prints the mask head of every detection level: its modules, convolution layers, feature map size, RoIAlign
        pool, doublings, mask size and spatial scale, under the normalization and the upsampling of the mask heads.
        """
        console = Console(width=80)
        normalization = str(self.mask_head_configuration["normalization"]["feature_map_normalization"]).capitalize()
        summary_text = (f"[bold]Normalization Strategy:[/bold] {normalization}\n"
                        f"[bold]Upsampling:[/bold] {self.mask_head_configuration['upsampling'].capitalize()}\n\n")

        # One block of lines per detection level
        for index_string, mask_head in self.mask_heads_dictionary.items():
            summary_text += f"[bold]Mask Block {index_string}[/bold]\n"
            active_module_names = mask_head.convolutional_block.get_active_module_names()
            if active_module_names:
                summary_text += f"  - Modules            : {', '.join(active_module_names)}\n"
            summary_text += f"  - Convolution Layers : {mask_head.convolution_number_layers}\n"
            summary_text += f"  - Feature Map Size   : [{mask_head.feature_map_size}, {mask_head.feature_map_size}]\n"
            pool_size, upsampled_size = mask_head.roi_align_pool_size, mask_head.upsampled_size
            summary_text += f"  - RoI Align Pool     : [{pool_size}, {pool_size}]\n"
            summary_text += (f"  - Doublings          : {mask_head.number_upsamplings}, to [{upsampled_size}, "
                             f"{upsampled_size}]\n")

            # Masks resized bilinearly when the doublings fall short of the mask resolution
            resize_note = "" if upsampled_size == mask_head.mask_size else ", resized bilinearly"
            summary_text += f"  - Mask Size          : [{mask_head.mask_size}, {mask_head.mask_size}]{resize_note}\n"
            summary_text += f"  - Spatial Scale      : {mask_head.spatial_scale:.4f}\n\n"

        console.print(Panel(summary_text.rstrip("\n"), title="[bold]Mask Architecture[/bold]", border_style="magenta"))

    def get_size_components(self) -> Dict[str, nn.Module]:
        """
        Lists the components whose size is reported by the model size summary: those of the detector, with the mask
        heads before the whole model.

        Returns:
            Dict[str, nn.Module]: The display name and module of every component.
        """
        size_components = super(InstanceSegmenter, self).get_size_components()
        total_model = size_components.pop("Total Model")
        return {**size_components, "Mask Heads": self.mask_heads_dictionary, "Total Model": total_model}
