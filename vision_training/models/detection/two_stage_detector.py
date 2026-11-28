"""
Model of the object detection task: the TwoStageDetector combines the shared Backbone, the anchors of every level, the
region proposal heads and filter, the target assignment, the detection heads and the postprocessing, and returns the
losses and detections. Its region proposal heads and detection heads are one per level, or one head read by every
level ('RegionProposal.shared', 'Detector.shared', as the region proposal network and box head of Mask R-CNN).
Registered in model_registry.py for the "object_detection" task and wrapped by inference_detector.py for the exports.
"""

import torch
from torch import nn
from rich.console import Console
from rich.panel import Panel
from typing import Dict, Any, Tuple, Optional, List
from vision_training.models.backbones.backbone import build_backbone, split_skip_outputs
from vision_training.models.components.convolution_block import BATCH_NORMALIZATION_MOMENTUM
from vision_training.models.task_model import TaskModel
from vision_training.models.detection.anchors import Anchors
from vision_training.models.detection.feature_neck import FeatureNeck
from vision_training.models.detection.region_proposer import RegionProposal, RegionProposalFilter
from vision_training.models.detection.detection_head import DetectionHead
from vision_training.models.detection.target_assignment import (turn_boxes_to_transformation_targets,
                                                                sample_positive_and_negative_training_targets,
                                                                batch_assign_targets_to_proposals,
                                                                add_ground_truth_proposals)
from vision_training.models.detection.postprocessing import postprocess_detections
from vision_training.losses.detection_losses import (assign_region_proposal_targets, compute_detection_losses,
                                                     compute_region_proposal_losses)

# Key of a head shared by every level in the dictionaries of the region proposal, detection and mask heads, which
# otherwise hold one head per level keyed by its detection block, so that the transfer, the optimizer groups and the
# size summary read both alike
SHARED_HEAD_KEY = "shared"


class TwoStageDetector(TaskModel):
    """
    Constructs the overall object detection model by combining the backbone,
    region proposal, and detector modules based on the configuration.
    """

    def __init__(self, configuration: Dict[str, Any], device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        Builds the detector of the object detection task, and the base of the instance segmenter: the shared backbone
        with its outputs at the detection levels, the anchors of every level, the region proposal head and the
        detection head of every level (or one of each read by every level), and the region proposal filter, from the
        model configuration read by the detection trainer, the model registry and the export.

        Args:
            configuration (Dict[str, Any]): The complete configuration dictionary for the experiment.
            device (torch.device): The device for parameter allocation.
            dtype (torch.dtype): The data type for parameter allocation.
        """
        super(TwoStageDetector, self).__init__()

        # Assign device and dtype globally for the model
        self.device = device
        self.dtype = dtype

        # Get configuration for each component of our model
        self.data_configuration = configuration['Data']
        self.backbone_configuration = configuration['Backbone']
        self.detection_levels_configuration = configuration['DetectionLevels']
        self.neck_configuration = configuration['Neck']
        self.region_proposal_configuration = configuration['RegionProposal']
        self.detector_configuration = configuration['Detector']

        # Get input image size and total number of classes
        self.input_image_size = self.data_configuration["image_settings"]["size"]
        self.label_dictionary = self.data_configuration["label_dictionary"]
        self.number_classes = len(set(self.label_dictionary.values()))

        # Final detection post-processing settings, applied at inference only: the score threshold, the number of
        # detections per image, and the threshold, criterion and distance exponent of the final per-class NMS
        self.detection_score_threshold = self.detector_configuration["inference"]["score_threshold"]
        self.maximum_detections = self.detector_configuration["inference"]["maximum_detections"]
        self.detection_nms_iou_threshold = self.detector_configuration["nms_iou_threshold"]
        self.detection_nms_criterion = self.detector_configuration["nms_criterion"]
        self.detection_nms_distance_exponent = self.detector_configuration["nms_distance_exponent"]

        # One region proposal head and one detection head read by every level instead of one head per level
        self.shared_region_proposal = bool(self.region_proposal_configuration["shared"])
        self.shared_detector = bool(self.detector_configuration["shared"])

        # Initialize the backbone feature extractor, whose outputs are the blocks numbered by the detection levels, and
        # the skip outputs the mask heads of the instance segmenter read (none for the detector)
        detection_convolution_block_indices = sorted(int(level_key) for level_key in configuration["DetectionLevels"])
        self.backbone = build_backbone(backbone_configuration=self.backbone_configuration,
                                       input_image_size=self.input_image_size, device=self.device, dtype=self.dtype,
                                       detection_convolution_block_indices=detection_convolution_block_indices,
                                       skip_block_indices=self.list_backbone_skip_blocks(configuration=configuration))

        # Pre-compute spatial dimensions and channel depths of the backbone's feature maps just once
        backbone_level_information = self.backbone.compute_detection_input_information()

        # Build the neck between the backbone blocks and the heads ('none' builds no layer, the heads then reading the
        # blocks), whose output sizes and channels are those the heads of every level are built from
        neck_normalization = self.neck_configuration["normalization"]["feature_map_normalization"]
        self.neck = FeatureNeck(neck_type=self.neck_configuration["type"], level_information=backbone_level_information,
                                output_channels=self.neck_configuration["output_channels"],
                                feature_map_normalization=neck_normalization,
                                batch_normalization_momentum=BATCH_NORMALIZATION_MOMENTUM, device=self.device,
                                dtype=self.dtype)
        self.detection_input_information = self.neck.compute_output_information()

        # Anchors of every detection level, which the region proposal heads regress and the targets are assigned to
        self.level_anchor_generators = self._build_level_anchor_generators(
            detection_levels_configuration=self.detection_levels_configuration)

        # Initialize the region proposal head of every detection level, or the head shared by the levels
        self.region_proposer_dictionary = self._build_region_proposer_dictionary(
            region_proposal_configuration=self.region_proposal_configuration,
            detection_levels_configuration=self.detection_levels_configuration)

        # Initialize the detection head of every detection level, or the head shared by the levels
        self.detector_heads_dictionary = self._build_detector_heads_dictionary(
            detector_configuration=self.detector_configuration,
            detection_levels_configuration=self.detection_levels_configuration)

        # Aggregate all unbatched anchors across all scales into a single global tensor
        self.anchors = self._aggregate_all_anchors()
        self.number_anchors = self.anchors.shape[0]

        # Initialize the region proposal filter to refine the raw proposals generated by the network
        self.region_proposal_filter = self._build_region_proposal_filter(
            region_proposal_configuration=self.region_proposal_configuration)

    def _build_level_anchor_generators(self, detection_levels_configuration: Dict[str, Any]) -> nn.ModuleDict:
        """
        Builds the anchors of every detection level from its scales and aspect ratios and the size of its feature map
        after the neck, read by the region proposal heads (one per level or one shared by the levels), the target
        assignment, the anchor recall and the proposal filter.

        Args:
            detection_levels_configuration (Dict[str, Any]): The 'DetectionLevels' section, giving the anchor scales
                                                             and aspect ratios of every level.

        Returns:
            nn.ModuleDict: The Anchors of every level, keyed by the stringified detection block index, in the order of
                the backbone outputs.
        """
        level_anchor_generators = nn.ModuleDict()

        # Iterate over the sequentially ordered detection blocks given by the neck
        for index_string, level_information in self.detection_input_information.items():
            anchor_configuration = detection_levels_configuration[index_string]["anchors"]
            level_anchor_generators[index_string] = Anchors(
                scales=anchor_configuration["scales"], aspect_ratios=anchor_configuration["aspect_ratios"],
                input_image_size=self.input_image_size, feature_map_size=level_information["feature_map_size"],
                dtype=self.dtype, device=self.device)
        return level_anchor_generators

    def list_head_levels(self, section_name: str, level_component: str, shared: bool) -> Dict[str, List[str]]:
        """
        Lists the heads of one kind to build (region proposal, detection or mask heads) with the levels each reads:
        one head per detection level, keyed by its block, or with the sharing of the section a single head keyed
        SHARED_HEAD_KEY reading every level, which needs the same head settings ('DetectionLevels.<level>.<component>')
        and the same channel count on every level (the outputs of a neck, or backbone blocks of equal channels), the
        configuration checks refusing the other files before the model is built.

        Args:
            section_name (str): The section holding the sharing, named in the errors ('RegionProposal', 'Detector' or
                'MaskHead').
            level_component (str): The component of every level holding the head settings ('region_proposal',
                'detector' or 'mask_head').
            shared (bool): Whether a single head reads every level.

        Returns:
            Dict[str, List[str]]: The levels of every head, in the order of the backbone outputs, keyed by the name of
                the head in its dictionary.

        Raises:
            ValueError: If the shared head is asked for levels of different head settings or channel counts.
        """
        level_names = list(self.detection_input_information)
        if not shared:
            return {level_name: [level_name] for level_name in level_names}

        # One set of head settings and one channel count for every level
        level_heads = {level_name: self.detection_levels_configuration[level_name][level_component]
                       for level_name in level_names}
        level_channels = {level_name: self.detection_input_information[level_name]["number_channels"]
                          for level_name in level_names}
        if any(level_head != level_heads[level_names[0]] for level_head in level_heads.values()):
            raise ValueError(f"'{section_name}.shared' needs the same '{level_component}' on every detection level, "
                             f"got {level_heads}.")
        if len(set(level_channels.values())) != 1:
            raise ValueError(f"'{section_name}.shared' needs the same channel count on every detection level (a neck, "
                             f"or backbone blocks of equal channels), got {level_channels}.")
        return {SHARED_HEAD_KEY: level_names}

    def _build_region_proposer_dictionary(self, region_proposal_configuration: Dict[str, Any],
                                          detection_levels_configuration: Dict[str, Any]) -> nn.ModuleDict:
        """
        Dynamically constructs the Region Proposal components tailored for each detection feature map scale.

        This builder method reads the channel depths of the feature maps of every detection level after the neck
        (those of the backbone blocks without neck), the number of anchors per position of every level and the
        convolutions of every detection level to instantiate a distinct RegionProposal module for each level. With
        'RegionProposal.shared', a single head is applied to the map of every level instead, which needs the same
        convolutions, the same channel count and the same number of anchors per position on every level.

        Args:
            region_proposal_configuration (Dict[str, Any]): The 'RegionProposal' section, giving the normalization
                                                            shared by every level.
            detection_levels_configuration (Dict[str, Any]): The 'DetectionLevels' section, giving the region proposal
                                                             convolutions of every level.

        Returns:
            nn.ModuleDict: A PyTorch ModuleDict mapping the stringified detection block indices to
                           their initialized RegionProposal modules, or the shared head alone, keyed SHARED_HEAD_KEY.

        Raises:
            ValueError: If the shared head is asked for levels of different convolutions, channel counts or numbers of
                anchors per position.
        """
        region_proposer_dictionary = nn.ModuleDict()
        feature_map_normalization = region_proposal_configuration["normalization"]["feature_map_normalization"]
        head_levels = self.list_head_levels(section_name="RegionProposal", level_component="region_proposal",
                                            shared=self.shared_region_proposal)

        # The output layers of a shared head predict the same number of anchors at every position of every level
        anchors_per_location = {level_name: anchor_generator.number_anchors_per_location
                                for level_name, anchor_generator in self.level_anchor_generators.items()}
        if self.shared_region_proposal and len(set(anchors_per_location.values())) != 1:
            raise ValueError(f"'RegionProposal.shared' needs the same number of anchors per position (scales x aspect "
                             f"ratios) on every detection level, got {anchors_per_location}.")

        # Iterate over the heads, the levels in the order of the backbone outputs
        for head_name, level_names in head_levels.items():
            first_level = level_names[0]
            region_proposer_dictionary[head_name] = RegionProposal(
                convolutions=detection_levels_configuration[first_level]["region_proposal"]["convolutions"],
                input_channels=self.detection_input_information[first_level]["number_channels"],
                number_anchors_per_location=anchors_per_location[first_level],
                feature_map_normalization=feature_map_normalization,
                batch_normalization_momentum=BATCH_NORMALIZATION_MOMENTUM, dtype=self.dtype, device=self.device)

        return region_proposer_dictionary

    def _build_detector_heads_dictionary(self, detector_configuration: Dict[str, Any],
                                         detection_levels_configuration: Dict[str, Any]) -> nn.ModuleDict:
        """
        Dynamically constructs the Detection Head components tailored for each detection feature map scale.

        This builder method reads the exact spatial dimensions and channel depths of the feature maps of every
        detection level after the neck (those of the backbone blocks without neck). It then reads the detection head
        of every detection level to instantiate a distinct DetectionHead module for each level. With
        'Detector.shared', a single head reads the proposals of every level, each pooled at the scale of its level,
        which needs the same detection head and the same channel count on every level.

        Args:
            detector_configuration (Dict[str, Any]): The 'Detector' section, giving the normalization and the
                                                     modules shared by every level.
            detection_levels_configuration (Dict[str, Any]): The 'DetectionLevels' section, giving the RoIAlign pool
                                                             size, the convolutions and the fully connected layers
                                                             of the detection head of every level.

        Returns:
            nn.ModuleDict: A PyTorch ModuleDict mapping the stringified detection block indices to
                           their initialized DetectionHead modules, or the shared head alone, keyed SHARED_HEAD_KEY.

        Raises:
            ValueError: If the shared head is asked for levels of different detection heads or channel counts.
        """
        detector_heads_dictionary = nn.ModuleDict()
        feature_map_normalization = detector_configuration["normalization"]["feature_map_normalization"]
        head_levels = self.list_head_levels(section_name="Detector", level_component="detector",
                                            shared=self.shared_detector)

        # Iterate over the heads, the levels in the order of the backbone outputs
        for head_name, level_names in head_levels.items():
            first_level = level_names[0]
            detector_heads_dictionary[head_name] = DetectionHead(
                detector_configuration=detection_levels_configuration[first_level]["detector"],
                number_classes=self.number_classes,
                input_channels=self.detection_input_information[first_level]["number_channels"],
                feature_map_normalization=feature_map_normalization,
                batch_normalization_momentum=BATCH_NORMALIZATION_MOMENTUM, input_image_size=self.input_image_size,
                feature_map_sizes={level_name: self.detection_input_information[level_name]["feature_map_size"]
                                   for level_name in level_names},
                dtype=self.dtype, device=self.device, modules=detector_configuration["modules"])

        return detector_heads_dictionary

    def _aggregate_all_anchors(self) -> torch.Tensor:
        """
        Aggregates the base unbatched anchors across all detection levels.

        This method concatenates the absolute base anchors from every detection level
        into a single unified tensor. This provides easy global
        access to all anchors via `model.anchors`.

        Returns:
            torch.Tensor: A tensor of shape [total_anchors, 4] containing all base bounding boxes.
        """
        # Concatenate the anchors of every level vertically, in level order, to form a single list of bounding boxes
        return torch.cat(tensors=[anchor_generator.anchors
                                  for anchor_generator in self.level_anchor_generators.values()], dim=0)

    def get_level_anchors(self) -> Dict[int, torch.Tensor]:
        """
        Gives the anchors of every detection level, for the anchor recall the detection trainers measure on the
        training objects at launch, which tells how many objects no anchor of a level can become a foreground anchor
        for.

        Returns:
            Dict[int, torch.Tensor]: The [A, 4] [x_min, y_min, x_max, y_max] anchors of every level in network input
                pixels, keyed by block number.
        """
        return {int(index_string): anchor_generator.anchors
                for index_string, anchor_generator in self.level_anchor_generators.items()}

    def get_level_anchor_generators(self) -> Dict[int, Anchors]:
        """
        Gives the anchor generator of every detection level, whose grid, scales and aspect ratios let the anchor recall
        of the detection trainers (at launch and in the 'compute_anchor_coverage' diagnosis mode) find the best anchor
        of every object without comparing it with every anchor (Anchors.compute_best_overlaps).

        Returns:
            Dict[int, Anchors]: The Anchors module of every level, keyed by block number.
        """
        return {int(index_string): anchor_generator
                for index_string, anchor_generator in self.level_anchor_generators.items()}

    def get_region_proposal_head(self, level_name: str) -> RegionProposal:
        """
        Gives the region proposal head applied to the feature map of a level, its own head or the head shared by the
        levels, for the region proposals of the training and of the exported InferenceDetector.

        Args:
            level_name (str): The detection level, the stringified block index.

        Returns:
            RegionProposal: The head of the level.
        """
        return self.region_proposer_dictionary[SHARED_HEAD_KEY if self.shared_region_proposal else level_name]

    def _build_region_proposal_filter(self, region_proposal_configuration: Dict[str, Any]) -> RegionProposalFilter:
        """
        Instantiates the RegionProposalFilter module to prune and refine bounding box proposals.

        This builder method provisions the filtering component with the NMS thresholds and bounding box
        count limits explicitly tailored for the current execution mode (training vs. inference), and with the
        number of anchors of every level, which 'RegionProposal.proposal_selection' "per_level" ranks one by one.

        Args:
            region_proposal_configuration (Dict[str, Any]): Configuration governing the region proposal thresholds.

        Returns:
            RegionProposalFilter: The initialized module for filtering bounding box proposals.
        """
        # Anchors of every level in level order, the order in which aggregate_proposals_and_anchors concatenates their
        # proposals and _aggregate_all_anchors their anchors
        level_sizes = [int(anchor_generator.anchors.shape[0])
                       for anchor_generator in self.level_anchor_generators.values()]
        region_proposal_filter = RegionProposalFilter(configuration=region_proposal_configuration,
                                                      level_sizes=level_sizes)

        return region_proposal_filter

    def aggregate_proposals_and_anchors(
            self,
            region_proposal_output_tensor_dictionary: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Aggregates the multi-scale region proposals and batched anchors into a unified structure.

        This method iterates over the outputs from the various levels of the feature pyramid, concatenating the
        classification scores, bounding box regressions, and proposal boxes along the sequence dimension. It leverages
        the globally pre-aggregated `self.anchors` to construct the batched anchors efficiently in a single operation.

        Args:
            region_proposal_output_tensor_dictionary (Dict[str, Dict[str, torch.Tensor]]): A dictionary mapping
            detection indices to their respective region proposal predictions.

        Returns:
            Dict[str, torch.Tensor]: A single dictionary containing the concatenated predictions
                                     and batched anchors across all scales.
                - "classification_scores" (torch.Tensor): Shape [batch_size, total_anchors]
                - "bounding_box_regressions" (torch.Tensor): Shape [batch_size, total_anchors, 4]
                - "proposal_boxes" (torch.Tensor): Shape [batch_size, total_anchors, 4]
                - "anchors" (torch.Tensor): Shape [batch_size, total_anchors, 4]
                - "convolution_block_origin" (torch.Tensor): Shape [batch_size, total_anchors]
        """
        aggregated_scores = []
        aggregated_regressions = []
        aggregated_boxes = []
        aggregated_origins = []

        batch_size = None

        for detection_index_string, predictions in region_proposal_output_tensor_dictionary.items():
            # Shape: [batch_size, anchors_per_scale]
            classification_scores = predictions["classification_scores"]

            # Shape: [batch_size, anchors_per_scale, 4]
            bounding_box_regressions = predictions["bounding_box_regressions"]

            # Shape: [batch_size, anchors_per_scale, 4]
            proposal_boxes = predictions["proposal_boxes"]

            if batch_size is None:
                batch_size = classification_scores.shape[0]

            # Generate origin tracker to know which detection level these proposals came from
            origin_tensor = torch.full_like(input=classification_scores, fill_value=int(detection_index_string),
                                            dtype=torch.int64)

            aggregated_scores.append(classification_scores)
            aggregated_regressions.append(bounding_box_regressions)
            aggregated_boxes.append(proposal_boxes)
            aggregated_origins.append(origin_tensor)

        return {"classification_scores": torch.cat(tensors=aggregated_scores, dim=1),
                "bounding_box_regressions": torch.cat(tensors=aggregated_regressions, dim=1),
                "proposal_boxes": torch.cat(tensors=aggregated_boxes, dim=1),
                "convolution_block_origin": torch.cat(tensors=aggregated_origins, dim=1),
                # Batched anchor shape: [batch_size, anchors_per_scale, 4]
                "anchors": self.anchors.unsqueeze(dim=0).expand(size=(batch_size, -1, 4))}

    def aggregate_detection_outputs(self,
                                    detection_output_dictionary: Dict[str, Dict[str, Optional[torch.Tensor]]]
                                    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Aggregates the multi-scale detection outputs into a unified structure.

        This method iterates over the detection outputs from the various levels of the feature pyramid,
        concatenating the classification scores, bounding box regressions, and corresponding sliced
        targets (if in training mode) along the first dimension (total proposals across all scales).

        Args:
            detection_output_dictionary (Dict[str, Dict[str, Optional[torch.Tensor]]]): A dictionary mapping detection
            indices to their respective detection head predictions and sliced targets.

        Returns:
            Dict[str, Optional[torch.Tensor]]: A single dictionary containing the concatenated predictions
                                               and targets across all scales.
                - "classification_scores" (torch.Tensor): Shape [total_proposals, number_classes]
                - "box_regressions" (torch.Tensor): Shape [total_proposals, number_classes, 4]
                - "sliced_proposals" (torch.Tensor): Shape [total_proposals, 5]
                - "sliced_batch_indices" (torch.Tensor): Shape [total_proposals, 1]
                - "sliced_origins" (torch.Tensor): Shape [total_proposals], the detection level of every proposal
                - "sliced_valid_mask" (torch.Tensor): Shape [total_proposals], False for the padding copies of the
                  region proposal filter
                - "detection_boxes" (Optional[torch.Tensor]): Shape [total_proposals, number_classes, 4]
                - "sliced_labels" (Optional[torch.Tensor]): Shape [total_proposals] (only in training)
                - "sliced_regression_targets" (Optional[torch.Tensor]): Shape [total_proposals, 4] (only in training)
                - "sliced_matched_ground_truth_indices" (Optional[torch.Tensor]): Shape [total_proposals], the index
                  of the matched ground truth within its image, -1 for the background (only in training)
        """
        aggregated_scores = []
        aggregated_regressions = []
        aggregated_labels = []
        aggregated_regression_targets = []
        aggregated_proposals = []
        aggregated_detection_boxes = []
        aggregated_batch_indices = []
        aggregated_origins = []
        aggregated_valid_masks = []
        aggregated_matched_indices = []

        has_labels = False
        for detection_index_string, predictions in detection_output_dictionary.items():
            aggregated_scores.append(predictions["classification_scores"])
            aggregated_regressions.append(predictions["box_regressions"])
            aggregated_proposals.append(predictions["sliced_proposals"])
            aggregated_batch_indices.append(predictions["sliced_batch_indices"])
            aggregated_detection_boxes.append(predictions["detection_boxes"])
            aggregated_origins.append(predictions["sliced_origins"])
            aggregated_valid_masks.append(predictions["sliced_valid_mask"])

            # During both training and validation (when ground truth is available), we extract the targets
            # to compute the network's loss metrics. This condition dynamically passes targets through if they exist
            # instead of restricting them strictly to the training phase via a rigid 'self.training' check.
            if predictions.get("sliced_labels") is not None:
                has_labels = True
                aggregated_labels.append(predictions["sliced_labels"])
                aggregated_regression_targets.append(predictions["sliced_regression_targets"])
                aggregated_matched_indices.append(predictions["sliced_matched_ground_truth_indices"])

        return {
            "classification_scores": torch.cat(tensors=aggregated_scores, dim=0),
            "box_regressions": torch.cat(tensors=aggregated_regressions, dim=0),
            "detection_boxes": torch.cat(tensors=aggregated_detection_boxes, dim=0),
            "sliced_batch_indices": torch.cat(tensors=aggregated_batch_indices, dim=0),
            "sliced_proposals": torch.cat(tensors=aggregated_proposals, dim=0),
            "sliced_origins": torch.cat(tensors=aggregated_origins, dim=0),
            "sliced_valid_mask": torch.cat(tensors=aggregated_valid_masks, dim=0),
            "sliced_labels": torch.cat(tensors=aggregated_labels, dim=0) if has_labels else None,
            "sliced_regression_targets": torch.cat(tensors=aggregated_regression_targets,
                                                   dim=0) if has_labels else None,
            "sliced_matched_ground_truth_indices": torch.cat(tensors=aggregated_matched_indices,
                                                             dim=0) if has_labels else None}

    def _rearrange_predictions_by_batch(self, batch_size: int, detection_batch_indices: torch.Tensor,
                                        prediction_tensors_dictionary: Dict[str, Optional[torch.Tensor]]
                                        ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Re-arranges the flattened network predictions back into a batched list format.

        This method groups the predicted class scores, box regressions, final bounding boxes, the detection level
        of every proposal and the mask of the valid proposals by their original batch index. This is particularly
        useful during inference or metric calculation, where predictions need to be associated with their specific
        input images.

        Args:
            batch_size (int): The total number of images in the batch.
            detection_batch_indices (torch.Tensor): The batch indices mapping each proposal to an image. Shape: [K, 1]
                or [K].
            prediction_tensors_dictionary (Dict[str, Optional[torch.Tensor]]): A dictionary containing the flattened
                prediction tensors (e.g., classification_scores, detection_boxes). Each valid tensor must have a first
                dimension of size K.

        Returns:
            Dict[str, Optional[torch.Tensor]]: A dictionary where each key maps to a batched tensor of shape
                `[batch_size, ...]`, containing the predictions for each specific image.
        """

        tensor_names = ["classification_scores", "box_regressions", "detection_boxes", "sliced_origins",
                        "sliced_valid_mask"]
        rearranged_predictions = {key: [] for key in tensor_names}

        # Squeeze the indices since they are of shape [K, 1]
        squeezed_indices = detection_batch_indices.squeeze(dim=-1)

        for batch_index in range(batch_size):
            # Find the indices of the proposals that belong to the current image
            image_indices = torch.where(squeezed_indices == batch_index)[0]

            for tensor_name, tensor_data in prediction_tensors_dictionary.items():
                if tensor_data is not None and tensor_name in tensor_names:
                    rearranged_predictions[tensor_name].append(tensor_data[image_indices])

        # Stack the lists into batched tensors
        for tensor_name, tensor_list in rearranged_predictions.items():
            if tensor_list is not None and len(tensor_list) > 0:
                rearranged_predictions[tensor_name] = torch.stack(tensors=tensor_list, dim=0)

        return rearranged_predictions

    def _prepare_detector_inputs(self, filtered_boxes: torch.Tensor, filtered_origins: torch.Tensor,
                                 valid_proposal_mask: torch.Tensor,
                                 ground_truth_bounding_boxes: Optional[List[torch.Tensor]] = None,
                                 ground_truth_labels: Optional[List[torch.Tensor]] = None
                                 ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
                                            torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepares the formatted inputs required by the Detection Head's RoIAlign and loss functions.

        With ground truth (training and validation loss), it writes the ground truth boxes into the last slots of
        the filtered proposals of their image (add_ground_truth_proposals), assigns ground truth targets to the
        proposals, computes regression targets, and samples a balanced set of positive and negative proposals.
        During inference, it simply flattens the filtered proposals and prepends batch indices, with the mask of the
        valid proposals, so that the padding copies of the region proposal filter never become detections.

        Args:
            filtered_boxes (torch.Tensor): The filtered region proposals of shape [Batch, Num_Proposals, 4].
            filtered_origins (torch.Tensor): The detection scale origins of the proposals of shape
                [Batch, Num_Proposals].
            valid_proposal_mask (torch.Tensor): The boolean mask of the survivors of the suppression, False for the
                padding copies of random survivors, of shape [Batch, Num_Proposals].
            ground_truth_bounding_boxes (Optional[List[torch.Tensor]]): List of ground truth bounding boxes.
            ground_truth_labels (Optional[List[torch.Tensor]]): List of ground truth class labels.

        Returns:
            Tuple containing:
                - detection_proposals (torch.Tensor): Formatted proposals of shape [K, 5] (batch_index + coords).
                - detection_proposals_origins (torch.Tensor): Origins of shape [K].
                - detection_labels (Optional[torch.Tensor]): Assigned ground truth labels of shape [K].
                - detection_regression_targets (Optional[torch.Tensor]): Target regressions of shape [K, 4].
                - detection_batch_indices (torch.Tensor): The batch indices for each valid proposal of shape [K, 1].
                - detection_matched_ground_truth_indices (Optional[torch.Tensor]): The index of the matched ground truth
                  of every sampled proposal within its image, -1 for the background, of shape [K] (None without
                  ground truth), read by the instance segmenter to pick the ground truth masks.
                - detection_valid_mask (torch.Tensor): The boolean mask of the proposals that may give a final
                  detection, of shape [K]: the valid proposals at inference, every sampled proposal with ground truth.
                - detection_ground_truth_copies (Optional[torch.Tensor]): The boolean mask of the sampled proposals
                  that are ground truth boxes written by add_ground_truth_proposals, of shape [K] (None without
                  ground truth), drawn apart by the training target viewer ('view_training_targets').
        """
        batch_size, number_proposals, _ = filtered_boxes.shape

        # Generate batch indices of shape [Batch, Proposals] to track which image each proposal belongs to
        batch_indices = torch.arange(start=0, end=batch_size, device=self.device,
                                     dtype=self.dtype).view(-1, 1).expand(batch_size, number_proposals)

        if ground_truth_bounding_boxes is not None:
            # Add the ground truth boxes to the proposals of their image, at the level of their best anchor, so that
            # every object gives a positive proposal even when the region proposal network misses it
            filtered_boxes, filtered_origins, ground_truth_slots = add_ground_truth_proposals(
                batched_ground_truth_boxes=ground_truth_bounding_boxes, batched_proposals=filtered_boxes,
                batched_proposal_levels=filtered_origins, level_anchors=self.get_level_anchors())

            # Assign the ground truth boxes and classes to the proposals by their overlap
            (detector_proposal_targets, detector_proposal_labels,
             detector_matched_ground_truth_indices) = batch_assign_targets_to_proposals(
                batched_ground_truth_boxes=ground_truth_bounding_boxes, batched_proposals=filtered_boxes,
                batched_ground_truth_labels=ground_truth_labels,
                foreground_iou_threshold=self.detector_configuration["foreground_iou_threshold"])

            # Compute detection head regression targets
            detection_regression_targets = turn_boxes_to_transformation_targets(
                ground_truth_boxes=detector_proposal_targets, predicted_boxes=filtered_boxes)

            # Get training samples for detection network
            sampled_positive_mask, sampled_negative_mask = sample_positive_and_negative_training_targets(
                labels=detector_proposal_labels,
                desired_positives=self.detector_configuration['number_training_positives'],
                desired_total=self.detector_configuration['total_training_samples'])

            sampled_mask = sampled_positive_mask | sampled_negative_mask

            # Apply the mask, extracting only the valid elements across the entire batch (flattening them)
            detection_batch_indices = batch_indices[sampled_mask].unsqueeze(dim=-1)
            sampled_boxes = filtered_boxes[sampled_mask]

            # Concatenate the batch indices as the first column to create the [K, 5] tensor expected by RoIAlign
            detection_proposals = torch.cat(tensors=[detection_batch_indices, sampled_boxes], dim=-1)
            detection_proposals_origins = filtered_origins[sampled_mask]

            detection_labels = detector_proposal_labels[sampled_mask]
            detection_regression_targets = detection_regression_targets[sampled_mask]
            detection_matched_ground_truth_indices = detector_matched_ground_truth_indices[sampled_mask]
            detection_ground_truth_copies = ground_truth_slots[sampled_mask]

            # Every sampled proposal is a box of the image, a survivor, a copy trained like it or a ground truth box
            # written over the padding; the detections of this path are those of the validation loss only, the
            # evaluation running the detector without ground truth
            detection_valid_mask = torch.ones_like(input=detection_proposals_origins, dtype=torch.bool)

        else:
            # During inference, simply flatten the valid proposals and prepend the batch indices
            detection_batch_indices = batch_indices.reshape(-1).unsqueeze(dim=-1)
            flattened_boxes = filtered_boxes.reshape(-1, 4)

            detection_proposals = torch.cat(tensors=[detection_batch_indices, flattened_boxes], dim=-1)
            detection_proposals_origins = filtered_origins.reshape(-1)
            detection_valid_mask = valid_proposal_mask.reshape(-1)

            detection_labels = None
            detection_regression_targets = None
            detection_matched_ground_truth_indices = None
            detection_ground_truth_copies = None

        return (detection_proposals, detection_proposals_origins, detection_labels, detection_regression_targets,
                detection_batch_indices, detection_matched_ground_truth_indices, detection_valid_mask,
                detection_ground_truth_copies)

    def _forward_region_proposals(
            self, backbone_output_tensor_dictionary: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Executes the Region Proposal forward pass for all intermediate feature map scales, the head of every level (or
        the head shared by the levels) regressing the anchors of that level.

        Args:
            backbone_output_tensor_dictionary (Dict[str, torch.Tensor]): The multi-scale backbone feature maps of
                shape [Batch, Channels, Height, Width].

        Returns:
            Dict[str, Dict[str, torch.Tensor]]: The predicted proposal scores, bounding box regressions,
                                                and generated proposal boxes per detection block index.
                - "classification_scores" (torch.Tensor): Shape [Batch, Anchors_in_scale]
                - "bounding_box_regressions" (torch.Tensor): Shape [Batch, Anchors_in_scale, 4]
                - "proposal_boxes" (torch.Tensor): Shape [Batch, Anchors_in_scale, 4]
        """
        region_proposal_output_tensor_dictionary = {}

        for detection_index_string, feature_map_tensor in backbone_output_tensor_dictionary.items():
            # Pass the corresponding feature map through its region proposal head, with the anchors of its level
            region_proposal_head = self.get_region_proposal_head(level_name=detection_index_string)
            proposal_scores, proposal_boxes_transformations, proposal_boxes = region_proposal_head(
                input_tensor=feature_map_tensor, anchors=self.level_anchor_generators[detection_index_string].anchors)

            # Store the predictions structured securely by their detection indices
            region_proposal_output_tensor_dictionary[detection_index_string] = {
                "classification_scores": proposal_scores,
                "bounding_box_regressions": proposal_boxes_transformations,
                "proposal_boxes": proposal_boxes}

        return region_proposal_output_tensor_dictionary

    def _forward_detections(self, backbone_output_tensor_dictionary: Dict[str, torch.Tensor],
                            detection_proposals: torch.Tensor, detection_proposals_origins: torch.Tensor,
                            detection_batch_indices: torch.Tensor, detection_valid_mask: torch.Tensor,
                            detection_labels: Optional[torch.Tensor] = None,
                            detection_regression_targets: Optional[torch.Tensor] = None,
                            detection_matched_ground_truth_indices: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Executes the Detection Head forward pass for all proposals across their respective feature maps, with the head
        of their level or the head shared by the levels (predict_level_detections).

        Args:
            backbone_output_tensor_dictionary (Dict[str, torch.Tensor]): The multi-scale backbone feature maps of
                shape [Batch, Channels, Height, Width].
            detection_proposals (torch.Tensor): Formatted proposals of shape [K, 5] (batch_index + coords).
            detection_proposals_origins (torch.Tensor): Origins of shape [K].
            detection_batch_indices (torch.Tensor): Batch indices of shape [K, 1].
            detection_valid_mask (torch.Tensor): The mask of the proposals that may give a final detection, of shape
                [K].
            detection_labels (Optional[torch.Tensor]): Target labels for training of shape [K].
            detection_regression_targets (Optional[torch.Tensor]): Target regressions for training of shape [K, 4].
            detection_matched_ground_truth_indices (Optional[torch.Tensor]): Index of the matched ground truth of
                every proposal within its image for training, of shape [K].

        Returns:
            Dict[str, Any]: Dictionary mapping detection index to a dictionary containing the sliced and computed
                outputs:
                - "classification_scores" (torch.Tensor): Shape [K_scale, number_classes]
                - "box_regressions" (torch.Tensor): Shape [K_scale, number_classes, 4]
                - "sliced_labels" (Optional[torch.Tensor]): Shape [K_scale]
                - "sliced_regression_targets" (Optional[torch.Tensor]): Shape [K_scale, 4]
                - "sliced_proposals" (torch.Tensor): Shape [K_scale, 5]
                - "sliced_origins" (torch.Tensor): Shape [K_scale]
                - "sliced_valid_mask" (torch.Tensor): Shape [K_scale]
                - "sliced_matched_ground_truth_indices" (Optional[torch.Tensor]): Shape [K_scale]
        """
        detection_output_dictionary = {}

        # Indices of the proposals that originated from every feature map scale, and those proposals
        level_indices = {detection_index_string: torch.where(
            detection_proposals_origins == int(detection_index_string))[0]
            for detection_index_string in backbone_output_tensor_dictionary}
        level_proposals = {detection_index_string: detection_proposals[scale_indices]
                           for detection_index_string, scale_indices in level_indices.items()}

        # Classification logits, bounding box regressions and refined boxes of the proposals of every level, from the
        # head of their level or from the head shared by the levels
        level_head_outputs = self.predict_level_detections(feature_maps=backbone_output_tensor_dictionary,
                                                           level_proposals=level_proposals)

        for detection_index_string, scale_indices in level_indices.items():
            # Slice the batch indices of the proposals of this particular scale
            scale_batch_indices = detection_batch_indices[scale_indices]

            # Slice the labels and regression targets if they are provided (training mode)
            scale_labels = detection_labels[scale_indices] if detection_labels is not None else None
            scale_regression_targets = detection_regression_targets[
                scale_indices] if detection_regression_targets is not None else None
            scale_matched_indices = detection_matched_ground_truth_indices[
                scale_indices] if detection_matched_ground_truth_indices is not None else None
            classification_scores, box_regressions, detection_boxes = level_head_outputs[detection_index_string]

            # Store the computed tensors and the sliced ground truth targets for later loss computation
            detection_output_dictionary[detection_index_string] = {
                "classification_scores": classification_scores,
                "box_regressions": box_regressions,
                "detection_boxes": detection_boxes,
                "sliced_batch_indices": scale_batch_indices,
                "sliced_labels": scale_labels,
                "sliced_regression_targets": scale_regression_targets,
                "sliced_matched_ground_truth_indices": scale_matched_indices,
                "sliced_origins": detection_proposals_origins[scale_indices],
                "sliced_valid_mask": detection_valid_mask[scale_indices],
                "sliced_proposals": level_proposals[detection_index_string]}

        return detection_output_dictionary

    def predict_level_detections(self, feature_maps: Dict[str, torch.Tensor],
                                 level_proposals: Dict[str, torch.Tensor]
                                 ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Runs the detection heads on the proposals of every level, for the detection losses and the detections of the
        evaluation: the head of every level on the proposals of its level, even when it has none so that it stays in
        the graph, or the head shared by the levels ('Detector.shared'), which pools the proposals of every level from
        the map of that level at its spatial scale and runs its layers once on all of them, so that it trains on the
        sampled proposals of every level and its batch statistics cover them all.

        Args:
            feature_maps (Dict[str, torch.Tensor]): The feature maps after the neck, keyed by detection level.
            level_proposals (Dict[str, torch.Tensor]): The [K_l, 5] proposals of every level (image index, then the
                corners in network input pixels), keyed by level in the order of the feature maps, possibly empty.

        Returns:
            Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: The [K_l, number_classes] class logits, the
                [K_l, number_classes, 4] box regressions and the [K_l, number_classes, 4] refined boxes of the
                proposals of every level, keyed by level.
        """
        # One head per level, each run on the proposals of its own level
        if not self.shared_detector:
            return {level_name: self.detector_heads_dictionary[level_name](proposal_boxes=proposals,
                                                                            input_tensor=feature_maps[level_name])
                    for level_name, proposals in level_proposals.items()}

        # Shared head: the proposals of every level pooled at the spatial scale of their level, then predicted at once
        detection_head = self.detector_heads_dictionary[SHARED_HEAD_KEY]
        pooled_features = torch.cat(tensors=[detection_head.pool_box_features(
            proposal_boxes=proposals, input_tensor=feature_maps[level_name], level_name=level_name)
            for level_name, proposals in level_proposals.items()], dim=0)
        head_outputs = detection_head.predict_from_pooled_features(
            pooled_features=pooled_features, proposal_boxes=torch.cat(tensors=list(level_proposals.values()), dim=0))

        # Outputs split back into the proposals of every level, in level order
        level_counts = [int(proposals.shape[0]) for proposals in level_proposals.values()]
        split_outputs = [torch.split(output, split_size_or_sections=level_counts, dim=0) for output in head_outputs]
        return {level_name: (split_outputs[0][level_position], split_outputs[1][level_position],
                             split_outputs[2][level_position])
                for level_position, level_name in enumerate(level_proposals)}

    def forward(self, input_tensor: torch.Tensor, image_windows: Optional[torch.Tensor] = None,
                ground_truth_bounding_boxes: Optional[List[torch.Tensor]] = None,
                ground_truth_labels: Optional[List[torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Executes the forward pass of the TwoStageDetector, whose stages are run by forward_detection_stages.

        Args:
            input_tensor (torch.Tensor): The raw input image tensor of shape [Batch, Channels, Height, Width].
            image_windows (Optional[torch.Tensor]): The [Batch, 4] [x1, y1, x2, y2] window of every image in the
                network input (data/preprocessing.py:compute_image_window), to which the final detections are clipped,
                needed in evaluation mode only.
            ground_truth_bounding_boxes (Optional[List[torch.Tensor]]): The ground truth boxes of every image, each of
                shape [Num_GT, 4].
            ground_truth_labels (Optional[List[torch.Tensor]]): The ground truth labels of every image, each of shape
                [Num_GT].

        Returns:
            Dict[str, Any]: The model output dictionary described in forward_detection_stages.
        """
        return self.forward_detection_stages(input_tensor=input_tensor, image_windows=image_windows,
                                             ground_truth_bounding_boxes=ground_truth_bounding_boxes,
                                             ground_truth_labels=ground_truth_labels)["model_output_dictionary"]

    def list_backbone_skip_blocks(self, configuration: Dict[str, Any]) -> List[int]:
        """
        Lists the backbone blocks returned before their pooling for heads reading finer maps than the detection levels,
        called before the backbone is built. The detector reads the detection levels only; the instance segmenter
        overrides it with the blocks its mask heads read.

        Args:
            configuration (Dict[str, Any]): The complete configuration dictionary of the experiment.

        Returns:
            List[int]: The skip blocks, 0 standing for the normalized input image, empty for the detector.
        """
        return []

    def forward_detection_stages(self, input_tensor: torch.Tensor, image_windows: Optional[torch.Tensor] = None,
                                 ground_truth_bounding_boxes: Optional[List[torch.Tensor]] = None,
                                 ground_truth_labels: Optional[List[torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Runs every stage of the two-stage detector (backbone, neck, region proposals, proposal filter, target
        assignment, detection heads, postprocessing and losses) and returns its output dictionary with the
        intermediate results that the instance segmenter extends with its masks: the feature maps of the levels after
        the neck that the mask heads pool from, the skip outputs of the backbone, and the aggregated detections holding
        the level and matched ground truth of every sampled proposal.

        Args:
            input_tensor (torch.Tensor): The raw input image tensor of shape [Batch, Channels, Height, Width].
            image_windows (Optional[torch.Tensor]): The [Batch, 4] [x1, y1, x2, y2] window of every image in the
                network input (data/preprocessing.py:compute_image_window), to which postprocess_detections clips the
                final detections; needed in evaluation mode, unused in training mode.
            ground_truth_bounding_boxes (Optional[List[torch.Tensor]]): List containing ground truth boxes per image,
                each of shape [Num_GT, 4].
            ground_truth_labels (Optional[List[torch.Tensor]]): List containing ground truth labels per image, each
                of shape [Num_GT].

        Returns:
            Dict[str, Any]: The 'backbone_output_tensor_dictionary' (feature maps keyed by detection level, after the
                neck, the backbone blocks themselves without neck), the 'backbone_skip_dictionary' (the skip outputs
                of the backbone keyed 'skip_<block>', empty for the detector), the
                'aggregated_detections_dictionary' (see aggregate_detection_outputs), the 'training_targets'
                (described below) and the 'model_output_dictionary', a unified dictionary containing:
                - "classification_scores" (torch.Tensor): Shape [Batch, Post_NMS_Proposals, number_classes].
                - "box_regressions" (torch.Tensor): Shape [Batch, Post_NMS_Proposals, number_classes, 4].
                - "detection_boxes" (torch.Tensor): Shape [Batch, Post_NMS_Proposals, number_classes, 4].
                - "sliced_origins" (torch.Tensor): Shape [Batch, Post_NMS_Proposals], the level of every proposal.
                - "sliced_valid_mask" (torch.Tensor): Shape [Batch, Post_NMS_Proposals], False for the padding copies
                  of the region proposal filter, in the order of the head outputs.
                - the final detections of postprocess_detections, in evaluation mode only, the padding copies giving
                  none and the boxes clipped to the window of their image, as in the deployed InferenceDetector.
                - "proposal_boxes", "proposal_levels" and "valid_proposal_mask" (torch.Tensor): The [Batch,
                  Post_NMS_Proposals, 4] filtered proposals, their level and the mask of the survivors of the
                  suppression, in evaluation mode only, Post_NMS_Proposals being capped by the pre-NMS count and the
                  number of anchors.
                - "region_proposal_classification_loss" (Optional[torch.Tensor]): Scalar loss (if training).
                - "region_proposal_localisation_loss" (Optional[torch.Tensor]): Scalar loss (if training).
                - "detection_classification_loss" (Optional[torch.Tensor]): Scalar loss (if training).
                - "detection_regression_loss" (Optional[torch.Tensor]): Scalar loss (if training).
                The 'training_targets' (None without ground truth) hold the targets the losses were computed on, for
                the training target viewer ('view_training_targets'): the [A, 4] 'anchors' of every level with their
                [A] 'anchor_levels', the [B, A] 'anchor_labels' (1 foreground, 0 background, -1 ignored), the
                [B, A, 4] 'anchor_target_boxes' (the ground truth box every anchor regresses to) and the [B, A]
                'sampled_positive_anchors' and 'sampled_negative_anchors' of the region proposal losses, the
                [B, P] filtered 'proposal_boxes' (4 coordinates), 'proposal_scores' (objectness probabilities),
                'proposal_levels' and 'valid_proposal_mask' before the ground truth boxes are added, and the [K, 5]
                'sampled_proposals' (image index, then the corners) of the detection losses with their [K]
                'sampled_labels', 'sampled_matched_ground_truth_indices', 'sampled_ground_truth_copies' and
                'sampled_levels'.

        Raises:
            ValueError: If the image windows are missing in evaluation mode.
        """
        # The final detections of the evaluation mode are clipped to the windows, which must then be given
        if not self.training and image_windows is None:
            raise ValueError("The detector needs the image_windows of its images in evaluation mode, the [B, 4] window "
                             "of every image in the network input (compute_image_window), to clip its detections.")

        # Pass the raw image through the backbone to extract the multiscale feature maps, the skip outputs of the mask
        # heads kept apart from the detection levels
        final_backbone_tensor, backbone_output_tensor_dictionary = self.backbone(input_tensor=input_tensor)
        backbone_output_tensor_dictionary, backbone_skip_dictionary = split_skip_outputs(
            output_tensor_dictionary=backbone_output_tensor_dictionary)

        # Merge the levels through the neck, whose outputs are read by every head ('none' returns the blocks)
        backbone_output_tensor_dictionary = self.neck(feature_maps=backbone_output_tensor_dictionary)

        # Execute the Region Proposal networks across all extracted intermediate feature maps
        region_proposal_output_tensor_dictionary = self._forward_region_proposals(
            backbone_output_tensor_dictionary=backbone_output_tensor_dictionary)

        # Aggregate the proposals and anchors across all scales for downstream processing
        # The resulting dictionary structure contains the following concatenated tensors:
        # - "classification_scores": The foreground probability scores for all anchors across all feature maps
        # - "bounding_box_regressions": The predicted coordinate offsets (dx, dy, dw, dh) for all anchors
        # - "proposal_boxes": The absolute [x_min, y_min, x_max, y_max] coordinates of the predicted proposals
        # - "anchors": The absolute [x_min, y_min, x_max, y_max] coordinates of the base anchors
        aggregated_proposals_dictionary = self.aggregate_proposals_and_anchors(
            region_proposal_output_tensor_dictionary=region_proposal_output_tensor_dictionary)

        # Refine the raw bounding box predictions by filtering out low scores and suppressing overlapping boxes
        filtered_boxes, filtered_scores, filtered_origins, valid_proposal_mask = self.region_proposal_filter(
            proposal_boxes=aggregated_proposals_dictionary["proposal_boxes"],
            proposal_scores=aggregated_proposals_dictionary["classification_scores"],
            proposal_origins=aggregated_proposals_dictionary["convolution_block_origin"],
            input_image_size=self.input_image_size)

        # Prepare the filtered proposals to be safely digested by the Detection Head's RoIAlign
        (detection_proposals, detection_proposals_origins,
         detection_labels, detection_regression_targets, detection_batch_indices,
         detection_matched_ground_truth_indices, detection_valid_mask,
         detection_ground_truth_copies) = self._prepare_detector_inputs(
            filtered_boxes=filtered_boxes, filtered_origins=filtered_origins, valid_proposal_mask=valid_proposal_mask,
            ground_truth_bounding_boxes=ground_truth_bounding_boxes, ground_truth_labels=ground_truth_labels)

        # Route the formatted proposals to their appropriate detection heads to compute final box scores and regressions
        detection_output_tensor_dictionary = self._forward_detections(
            backbone_output_tensor_dictionary=backbone_output_tensor_dictionary,
            detection_proposals=detection_proposals, detection_proposals_origins=detection_proposals_origins,
            detection_batch_indices=detection_batch_indices, detection_valid_mask=detection_valid_mask,
            detection_labels=detection_labels, detection_regression_targets=detection_regression_targets,
            detection_matched_ground_truth_indices=detection_matched_ground_truth_indices)

        # Aggregate the final detection head outputs across all scales
        aggregated_detections_dictionary = self.aggregate_detection_outputs(
            detection_output_dictionary=detection_output_tensor_dictionary)

        # TODO Function to be renamed
        # Package the outputs into a unified dictionary for clean extraction
        # Rearrange the flattened detections back into a batched list format
        model_output_dictionary = self._rearrange_predictions_by_batch(
            batch_size=input_tensor.shape[0],
            detection_batch_indices=aggregated_detections_dictionary["sliced_batch_indices"],
            prediction_tensors_dictionary=aggregated_detections_dictionary)

        # At inference, turn the raw head outputs into the final thresholded, per-class NMS filtered detections clipped
        # to the window of their image, with the greedy NMS or DIoU-NMS of 'Detector.nms_criterion', the padding copies
        # of the proposal filter masked as in the deployed detector, since the NMS cannot remove identical boxes at a
        # threshold of 1.0 nor empty boxes
        if not self.training:
            model_output_dictionary.update(postprocess_detections(
                classification_scores=model_output_dictionary["classification_scores"],
                detection_boxes=model_output_dictionary["detection_boxes"],
                valid_proposal_mask=model_output_dictionary["sliced_valid_mask"], image_windows=image_windows,
                score_threshold=self.detection_score_threshold, nms_iou_threshold=self.detection_nms_iou_threshold,
                nms_criterion=self.detection_nms_criterion, nms_distance_exponent=self.detection_nms_distance_exponent,
                maximum_detections=self.maximum_detections, input_image_size=self.input_image_size,
                suppress_candidates_only=True))

            # Filtered proposals of every image with their level, the padding copies masked out, for the proposal
            # recall of the evaluation
            model_output_dictionary.update({"proposal_boxes": filtered_boxes, "proposal_levels": filtered_origins,
                                            "valid_proposal_mask": valid_proposal_mask})

        # Train, Validation and Test -> Loss Computation :
        # - For Region Proposal
        # - For Detection
        training_targets = None
        if ground_truth_bounding_boxes is not None:
            # Region Proposal Losses, on the anchors labelled and sampled after the detection heads, the order of the
            # random draws of a training step
            region_proposal_targets = assign_region_proposal_targets(
                ground_truth_bounding_boxes=ground_truth_bounding_boxes,
                anchors=aggregated_proposals_dictionary["anchors"],
                region_proposal_configuration=self.region_proposal_configuration)
            region_proposal_classification_loss, region_proposal_localisation_loss = compute_region_proposal_losses(
                aggregated_proposals_dictionary=aggregated_proposals_dictionary,
                region_proposal_targets=region_proposal_targets)

            # Detection Losses, with the box loss of 'Detector.box_loss'
            (detection_classification_loss, detection_regression_loss) = compute_detection_losses(
                aggregated_detections_dictionary=aggregated_detections_dictionary,
                box_loss=self.detector_configuration["box_loss"])

            # Append the assignments to the output dictionary
            model_output_dictionary["region_proposal_classification_loss"] = region_proposal_classification_loss
            model_output_dictionary["region_proposal_localisation_loss"] = region_proposal_localisation_loss

            model_output_dictionary["detection_classification_loss"] = detection_classification_loss
            model_output_dictionary["detection_regression_loss"] = detection_regression_loss

            # The targets the losses were computed on, read by the training target viewer: the labelled and sampled
            # anchors, the filtered proposals before the ground truth boxes are added, and the sampled proposals
            training_targets = {
                "anchors": self.anchors,
                "anchor_levels": aggregated_proposals_dictionary["convolution_block_origin"][0],
                "anchor_labels": region_proposal_targets["anchor_labels"],
                "anchor_target_boxes": region_proposal_targets["anchor_target_boxes"],
                "sampled_positive_anchors": region_proposal_targets["sampled_positive_mask"],
                "sampled_negative_anchors": region_proposal_targets["sampled_negative_mask"],
                "proposal_boxes": filtered_boxes, "proposal_scores": filtered_scores,
                "proposal_levels": filtered_origins, "valid_proposal_mask": valid_proposal_mask,
                "sampled_proposals": detection_proposals, "sampled_labels": detection_labels,
                "sampled_matched_ground_truth_indices": detection_matched_ground_truth_indices,
                "sampled_ground_truth_copies": detection_ground_truth_copies,
                "sampled_levels": detection_proposals_origins}

        return {"backbone_output_tensor_dictionary": backbone_output_tensor_dictionary,
                "backbone_skip_dictionary": backbone_skip_dictionary,
                "aggregated_detections_dictionary": aggregated_detections_dictionary,
                "model_output_dictionary": model_output_dictionary, "training_targets": training_targets}

    def print_summary(self) -> None:
        """
        Prints a structural summary of the overarching TwoStageDetector architecture.

        This method delegates to the underlying Backbone to print its layer-by-layer
        spatial dimension summary, then prints the Neck, the Region Proposal and finally
        the Detection summaries, ensuring the structural integrity of the visual pipeline.
        """
        self.backbone.print_summary()
        self.print_neck_summary()
        self.print_region_proposal_summary()
        self.print_detection_summary()

    def print_neck_summary(self) -> None:
        """
        Prints the neck between the backbone blocks and the heads at the launch of every detection and instance
        segmentation training: its type and, with a neck, its normalization and the size and channels of every level
        read by the heads.
        """
        console = Console(width=60)
        summary_text = f"[bold]Type:[/bold] {self.neck.neck_type}"

        # With a neck, list its normalization and the maps it gives to the heads of every level
        if self.neck.neck_type != "none":
            normalization_name = str(object=self.neck_configuration["normalization"]["feature_map_normalization"])
            summary_text += f"\n[bold]Normalization Strategy:[/bold] {normalization_name.capitalize()}\n"
            for level_name, level_information in self.detection_input_information.items():
                feature_map_size = level_information["feature_map_size"]
                summary_text += (f"\n  - Level {level_name} : [{feature_map_size}, {feature_map_size}], "
                                 f"{level_information['number_channels']} channels")

        console.print(Panel(summary_text, title="[bold]Neck Architecture[/bold]", border_style="blue"))

    def print_region_proposal_summary(self) -> None:
        """
        Prints a structural summary of the Region Proposal Network components.

        This summary leverages rich panels to cleanly display the feature map size, the number
        of scales, aspect ratios, and the absolute anchor count of every detection level, whether
        it has its own region proposal head or reads the head shared by the levels, culminating in
        the total number of region proposal anchors across the entire network.
        """
        console = Console(width=60)

        # The head shared by the levels is named first, the anchors staying those of every level
        summary_text = ""
        if self.shared_region_proposal:
            summary_text += (f"[bold]One Head Shared By Blocks "
                             f"{', '.join(self.level_anchor_generators)}[/bold]\n\n")

        for index_string, anchor_generator in self.level_anchor_generators.items():
            # Feature map spatial size
            feature_map_size = anchor_generator.feature_map_size

            # Anchor scales and ratios of the configuration
            anchor_configuration = self.detection_levels_configuration[index_string]["anchors"]
            scales = anchor_configuration["scales"]
            aspect_ratios = anchor_configuration["aspect_ratios"]
            number_of_anchors_per_location = anchor_generator.number_anchors_per_location

            # Calculate the total number of region proposal anchors generated by this specific block by multiplying
            # the spatial area of the feature map by the number of anchors spawned at each spatial location
            total_anchors_for_proposal = feature_map_size * feature_map_size * number_of_anchors_per_location

            summary_text += f"[bold]Convolution Block {index_string}[/bold]\n"
            summary_text += f"  - Feature Map Size : [{feature_map_size}, {feature_map_size}]\n"
            summary_text += f"  - Scales           : {scales}\n"
            summary_text += f"  - Aspect Ratios    : {aspect_ratios}\n"
            summary_text += f"  - Total Anchors    : {total_anchors_for_proposal}\n\n"

        summary_text += f"[bold blue]Total Network Anchors: {self.number_anchors}[/bold blue]"

        panel = Panel(summary_text, title="[bold]Region Proposal Architecture[/bold]", border_style="yellow")
        console.print(panel)

    def print_detection_summary(self) -> None:
        """
        Prints a structural summary of the Detection Network components.

        This summary leverages rich panels to cleanly display the feature map size, the RoI Align
        pool size, and the spatial scale used for RoI pooling across each configured Detection block,
        or those of every level for the detection head shared by the levels.
        """
        console = Console(width=80)

        # Initialize the textual representation of the architecture,
        # including the applied normalization strategy prominently
        normalization_name = str(object=self.detector_configuration["normalization"]["feature_map_normalization"])
        summary_text = f"[bold]Normalization Strategy:[/bold] {normalization_name.capitalize()}\n\n"

        for index_string, detection_head in self.detector_heads_dictionary.items():
            convolution_layers = detection_head.convolution_number_layers
            fully_connected_layers = detection_head.fully_connected_dimensions
            roi_align_pool_size = detection_head.roi_align_pool_size

            # One block of lines per detection level, or one for the head shared by the levels
            summary_text += (f"[bold]Detection Head Shared By Blocks {', '.join(detection_head.level_names)}[/bold]\n"
                             if index_string == SHARED_HEAD_KEY else f"[bold]Detection Block {index_string}[/bold]\n")

            # Show the convolution modules used by this head, only when it is not a plain convolution block
            active_module_names = detection_head.convolutional_block.get_active_module_names()
            if active_module_names:
                summary_text += f"  - Modules            : {', '.join(active_module_names)}\n"

            summary_text += f"  - Convolution Layers : {convolution_layers}\n"
            summary_text += f"  - Fully Conn. Layers : {fully_connected_layers}\n"
            summary_text += "  - Feature Map Size   : " + ", ".join(
                f"[{feature_map_size}, {feature_map_size}]"
                for feature_map_size in detection_head.feature_map_sizes.values()) + "\n"
            summary_text += f"  - RoI Align Pool     : [{roi_align_pool_size}, {roi_align_pool_size}]\n"
            summary_text += "  - Spatial Scale      : " + ", ".join(
                f"{spatial_scale:.4f}" for spatial_scale in detection_head.spatial_scales.values()) + "\n\n"

        if summary_text.endswith("\n\n"):
            summary_text = summary_text[:-2]

        panel = Panel(summary_text, title="[bold]Detection Architecture[/bold]", border_style="green")
        console.print(panel)

    def get_detection_block_annotations(self) -> Dict[str, str]:
        """
        Gives the anchor scales assigned to every detection block, printed below the block in the receptive field
        summary of TaskModel, since anchors much larger than the receptive field cannot be fully observed by their
        feature map cell.

        Returns:
            Dict[str, str]: The anchor scales line of every detection block, keyed by block number.
        """
        return {block_string: f"Detection Block {block_string} Anchor Scales : {level['anchors']['scales']}"
                for block_string, level in self.detection_levels_configuration.items()}

    def get_size_components(self) -> Dict[str, nn.Module]:
        """
        Lists the components of the detector whose size is reported by the model size summary of TaskModel: the
        backbone, the neck when it has layers, the region proposal heads, the detection heads and the whole detector.

        Returns:
            Dict[str, nn.Module]: The display name and module of every component.
        """
        neck_component = {"Neck": self.neck} if self.neck.neck_type != "none" else {}
        return {"Backbone": self.backbone, **neck_component, "Region Proposal Heads": self.region_proposer_dictionary,
                "Detection Heads": self.detector_heads_dictionary, "Total Model": self}
