"""
Inference-only version of the trained TwoStageDetector with batched, fixed-shape proposal filtering and detection,
returning a fixed number of detections per image, clipped to the window of their image (its second input) and kept
from the score threshold of their class (a buffer the export fills from the evaluation of the checkpoint). A region
proposal head shared by the levels runs on every level with the anchors of that level, and a shared detection head
pools every proposal from every level, keeps the features of its own level and runs once. Built by
vision_training/deployment/export_specifications.py for the TorchScript and ONNX exports served by Triton.
"""

import torch

from torch import nn
from typing import Any, Dict, List, Optional, Tuple

from vision_training.models.backbones.backbone import split_skip_outputs
from vision_training.models.detection.two_stage_detector import SHARED_HEAD_KEY, TwoStageDetector
from vision_training.models.detection.postprocessing import postprocess_detections
from vision_training.models.detection.region_proposer import select_best_proposals


class InferenceDetector(nn.Module):
    """
    Inference-only version of the trained detector, used for deployment (TorchScript and ONNX exports served by
    Triton) and for local inference.

    It reuses the submodules of the trained TwoStageDetector (no weight is duplicated) but replaces the
    data-dependent parts of its forward pass by batched, fixed-shape operations: proposals are filtered for the whole
    batch at once, padding proposals are masked instead of randomly duplicated, and no ground truth, target or loss
    code is involved. It returns a fixed number of detections per image, clipped to the window of their image (the
    'image_windows' input, the part of the square input holding the resized image), kept from the score threshold of
    their class ('class_score_thresholds' buffer, 'Detector.inference.score_threshold' for every class until
    set_class_score_thresholds changes it) and suppressed by the final NMS of the configuration of the detector, greedy
    or DIoU-NMS ('Detector.nms_criterion'), as in its evaluation.
    """

    def __init__(self, model: TwoStageDetector) -> None:
        """
        Wraps the submodules and inference settings of a trained TwoStageDetector.

        Args:
            model (TwoStageDetector): The trained detector, whose submodules and configuration are reused.
        """
        super(InferenceDetector, self).__init__()

        self.backbone = model.backbone
        self.neck = model.neck
        self.region_proposer_dictionary = model.region_proposer_dictionary
        self.detector_heads_dictionary = model.detector_heads_dictionary
        self.level_anchor_generators = model.level_anchor_generators

        # One region proposal head and one detection head read by every level instead of one head per level
        self.shared_region_proposal = model.shared_region_proposal
        self.shared_detector = model.shared_detector

        # Detection levels, ordered like the backbone outputs
        self.level_names = [str(index) for index in sorted(int(name) for name in model.level_anchor_generators)]

        # Region proposal filtering and final post-processing settings of the inference mode
        self.input_image_size = model.input_image_size
        # Number of anchors of every level in the order of level_names, the order of the concatenation of their
        # proposals, stored as plain numbers so that tracing never reads them from a tensor shape
        level_anchors = model.get_level_anchors()
        self.level_sizes = [int(level_anchors[int(level_name)].shape[0]) for level_name in self.level_names]
        self.pre_nms_proposals = model.region_proposal_filter.inference_pre_nms_proposals
        self.post_nms_proposals = model.region_proposal_filter.inference_post_nms_proposals
        self.proposal_nms_iou_threshold = model.region_proposal_filter.nms_iou_threshold
        self.proposal_selection = model.region_proposal_filter.proposal_selection
        self.score_threshold = model.detection_score_threshold
        self.number_classes = model.number_classes

        # Score threshold of every foreground class, the configured one until set_class_score_thresholds, on the device
        # of the detector so that the exports write it into their graphs as a constant
        self.register_buffer(name="class_score_thresholds", tensor=torch.full(
            size=(model.number_classes - 1,), fill_value=float(model.detection_score_threshold), dtype=torch.float32,
            device=model.device))
        self.detection_nms_iou_threshold = model.detection_nms_iou_threshold
        self.detection_nms_criterion = model.detection_nms_criterion
        self.detection_nms_distance_exponent = model.detection_nms_distance_exponent
        self.maximum_detections = model.maximum_detections

    def set_class_score_thresholds(self, class_score_thresholds: List[float]) -> None:
        """
        Sets the score threshold of every foreground class, read by deployment/detection_deployment.py from the
        'class_thresholds.json' of the evaluation of the exported checkpoint (the threshold reaching the target
        precision, or 'Detector.inference.score_threshold' for a class that never reaches it), so that the exported
        detector keeps the detections class by class.

        Args:
            class_score_thresholds (List[float]): The threshold of every foreground class, in class index order from 1.

        Raises:
            ValueError: If there is not one threshold per foreground class, or a threshold is outside [0, 1].
        """
        if len(class_score_thresholds) != self.class_score_thresholds.shape[0]:
            raise ValueError(f"Expected one score threshold per foreground class "
                             f"({self.class_score_thresholds.shape[0]}), got {len(class_score_thresholds)}.")
        if any(not 0.0 <= threshold <= 1.0 for threshold in class_score_thresholds):
            raise ValueError(f"Expected score thresholds between 0 and 1, got {class_score_thresholds}.")
        self.class_score_thresholds.copy_(torch.tensor(data=class_score_thresholds, dtype=torch.float32))

    def forward(self, images: torch.Tensor, image_windows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                                                    torch.Tensor, torch.Tensor]:
        """
        Detects the objects of a batch of preprocessed images.

        Args:
            images (torch.Tensor): Float BGR images in [0, 255] of shape [B, 3, H, W], resized and padded to the network
                input size.
            image_windows (torch.Tensor): The float [B, 4] [x1, y1, x2, y2] window of every image in network input
                pixels, the part holding the resized image (data/preprocessing.py:compute_image_window), [0, 0, S, S]
                for an image filling the input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The detection boxes [B, K, 4] in network
                input pixels inside the window of their image, their scores [B, K], their int64 labels [B, K] (0 for
                padding) and the int64 number of valid detections per image [B].
        """
        detections = self.detect_objects(images=images, image_windows=image_windows)["detections"]
        return (detections["final_detection_boxes"], detections["final_detection_scores"],
                detections["final_detection_labels"], detections["number_detections"])

    def detect_objects(self, images: torch.Tensor, image_windows: torch.Tensor) -> Dict[str, Any]:
        """
        Runs every detection step on a batch of preprocessed images (backbone, neck, region proposals, batched
        proposal filter, detection heads and postprocessing), keeping the intermediate results that the inference
        instance segmenter needs to predict the mask of every detection with the head of its level.

        Args:
            images (torch.Tensor): Float BGR images in [0, 255] of shape [B, 3, H, W], resized and padded to the network
                input size.
            image_windows (torch.Tensor): The [B, 4] window of every image, to which the detections are clipped.

        Returns:
            Dict[str, Any]: The 'feature_maps' after the neck keyed by level name (read by the mask heads too), the
                'skip_feature_maps' of the backbone keyed 'skip_<block>' (read by the mask heads, empty for the
                detector), the 'proposal_levels' [B, P] level index of every filtered proposal, and the 'detections'
                of postprocess_detections.
        """
        # Backbone outputs, the skip outputs of the mask heads kept apart from the detection levels
        _, backbone_feature_maps = self.backbone(input_tensor=images)
        backbone_feature_maps, skip_feature_maps = split_skip_outputs(output_tensor_dictionary=backbone_feature_maps)

        # Merge the levels through the neck, as in training ('none' returns the backbone blocks)
        feature_maps = self.neck(feature_maps=backbone_feature_maps)

        # Run the region proposal head of every level (or the head shared by the levels) on the map of the level with
        # its anchors, and concatenate the proposals of all levels, remembering their level
        proposal_scores, proposal_boxes, proposal_levels = [], [], []
        for level_index, level_name in enumerate(self.level_names):
            region_proposal_head = self.region_proposer_dictionary[
                SHARED_HEAD_KEY if self.shared_region_proposal else level_name]
            level_scores, _, level_boxes = region_proposal_head(
                input_tensor=feature_maps[level_name], anchors=self.level_anchor_generators[level_name].anchors)
            proposal_scores.append(level_scores)
            proposal_boxes.append(level_boxes)
            proposal_levels.append(torch.full_like(input=level_scores, fill_value=level_index, dtype=torch.int64))

        filtered_boxes, filtered_levels, valid_proposal_mask = self.filter_proposals(
            proposal_scores=torch.cat(tensors=proposal_scores, dim=1),
            proposal_boxes=torch.cat(tensors=proposal_boxes, dim=1),
            proposal_levels=torch.cat(tensors=proposal_levels, dim=1))

        classification_scores, detection_boxes = self.detect(feature_maps=feature_maps, proposal_boxes=filtered_boxes,
                                                             proposal_levels=filtered_levels)

        # Final detections clipped to the windows, kept from the threshold of their class, with the NMS of
        # 'Detector.nms_criterion' on every pair, greedy or DIoU-NMS, the form the TorchScript and ONNX exports keep
        detections = postprocess_detections(classification_scores=classification_scores,
                                            detection_boxes=detection_boxes, valid_proposal_mask=valid_proposal_mask,
                                            image_windows=image_windows, score_threshold=self.class_score_thresholds,
                                            nms_iou_threshold=self.detection_nms_iou_threshold,
                                            nms_criterion=self.detection_nms_criterion,
                                            nms_distance_exponent=self.detection_nms_distance_exponent,
                                            maximum_detections=self.maximum_detections,
                                            input_image_size=self.input_image_size, suppress_candidates_only=False)
        return {"feature_maps": feature_maps, "skip_feature_maps": skip_feature_maps,
                "proposal_levels": filtered_levels, "detections": detections}

    def filter_proposals(self, proposal_scores: torch.Tensor, proposal_boxes: torch.Tensor,
                         proposal_levels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Keeps, for every image, the best 'post_nms_proposals' proposals after non-maximum suppression (fewer when the
        pre-NMS count or the number of anchors is smaller), with the batched selection of the training-time region
        proposal filter (select_best_proposals), over all the levels or level by level as in the training
        ('RegionProposal.proposal_selection'), the padding slots being masked instead of filled with copies.

        Args:
            proposal_scores (torch.Tensor): The objectness logits of all anchors, shape [B, N].
            proposal_boxes (torch.Tensor): The decoded proposal boxes, shape [B, N, 4].
            proposal_levels (torch.Tensor): The detection level index of every proposal, shape [B, N].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The kept boxes [B, P, 4], their level indices [B, P] and a
                boolean mask [B, P] which is False for padding slots when fewer than P proposals survive.
        """
        # Batched selection shared with the region proposal filter of the training, with a single suppression call
        # over the batch for every group of anchors for the exports, the top-k limit clamped to the anchors of every
        # group, stored as plain numbers so that tracing never reads them from a tensor shape
        selected_boxes, _, selected_levels, valid_proposal_mask = select_best_proposals(
            proposal_scores=proposal_scores, proposal_boxes=proposal_boxes, proposal_levels=proposal_levels,
            level_sizes=self.level_sizes, pre_nms_proposals=self.pre_nms_proposals,
            post_nms_proposals=self.post_nms_proposals, nms_iou_threshold=self.proposal_nms_iou_threshold,
            input_image_size=self.input_image_size, suppress_per_image=False,
            proposal_selection=self.proposal_selection)
        return selected_boxes, selected_levels, valid_proposal_mask

    def detect(self, feature_maps: dict, proposal_boxes: torch.Tensor, proposal_levels: torch.Tensor) -> Tuple[
            torch.Tensor, torch.Tensor]:
        """
        Classifies and refines every proposal with the detection head of its own level.

        Every head processes all the proposals and each proposal keeps the output of its own level's head. Selecting
        the proposals of each level instead would give data-dependent tensor sizes, which the ONNX export cannot
        handle; the results are identical, at the cost of running the (small) detection heads on every proposal. The
        head shared by the levels ('Detector.shared') pools every proposal from every level instead, keeps the pooled
        features of its own level, and runs its layers once on all the proposals.

        Args:
            feature_maps (dict): The backbone feature maps, keyed by level name.
            proposal_boxes (torch.Tensor): The filtered proposals, shape [B, P, 4].
            proposal_levels (torch.Tensor): The level index of every proposal, shape [B, P].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The class logits [B, P, C] and per-class boxes [B, P, C, 4].
        """
        batch_size, number_proposals, _ = proposal_boxes.shape

        # Prepend the image index of every proposal, as expected by RoIAlign
        batch_indices = torch.arange(end=batch_size, device=proposal_boxes.device, dtype=proposal_boxes.dtype)
        batch_indices = batch_indices.unsqueeze(dim=1).expand(-1, number_proposals).reshape(-1, 1)
        region_boxes = torch.cat(tensors=[batch_indices, proposal_boxes.reshape(-1, 4)], dim=1)
        flat_levels = proposal_levels.reshape(-1)

        classification_scores: Optional[torch.Tensor] = None
        detection_boxes: Optional[torch.Tensor] = None

        # Shared head: every proposal pooled from every level, the pooled features of its own level kept, then the
        # layers of the head run once
        if self.shared_detector:
            detection_head = self.detector_heads_dictionary[SHARED_HEAD_KEY]
            pooled_features: Optional[torch.Tensor] = None
            for level_index, level_name in enumerate(self.level_names):
                level_pooled_features = detection_head.pool_box_features(
                    proposal_boxes=region_boxes, input_tensor=feature_maps[level_name], level_name=level_name)
                pooled_features = level_pooled_features if pooled_features is None else torch.where(
                    (flat_levels == level_index).reshape(-1, 1, 1, 1), level_pooled_features, pooled_features)
            classification_scores, _, detection_boxes = detection_head.predict_from_pooled_features(
                pooled_features=pooled_features, proposal_boxes=region_boxes)

        # Run every level's head on all the proposals and keep, for each proposal, the output of its own level
        else:
            for level_index, level_name in enumerate(self.level_names):
                level_scores, _, level_boxes = self.detector_heads_dictionary[level_name](
                    proposal_boxes=region_boxes, input_tensor=feature_maps[level_name])

                if classification_scores is None or detection_boxes is None:
                    classification_scores, detection_boxes = level_scores, level_boxes
                else:
                    level_mask = (flat_levels == level_index).unsqueeze(dim=1)
                    classification_scores = torch.where(level_mask, level_scores, classification_scores)
                    detection_boxes = torch.where(level_mask.unsqueeze(dim=2), level_boxes, detection_boxes)

        number_classes = classification_scores.shape[-1]
        return (classification_scores.reshape(batch_size, number_proposals, number_classes),
                detection_boxes.reshape(batch_size, number_proposals, number_classes, 4))
