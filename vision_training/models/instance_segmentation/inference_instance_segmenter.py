"""
Inference-only version of the trained InstanceSegmenter: the fixed-shape detection of the InferenceDetector followed by
the mask head of every level, or the mask head shared by the levels (with the fine inputs they share, when
configured), on the fixed number of detections of every image, returning a fifth output with the mask of every
detection and, when the masks are scored ('MaskHead.mask_scoring', Mask Scoring R-CNN), a sixth output with the mask
score of every detection, its class score times its predicted mask IoU. Built by
vision_training/deployment/export_specifications.py for the TorchScript and ONNX exports.
"""

import torch

from typing import Dict, Optional, Tuple

from vision_training.models.detection.inference_detector import InferenceDetector
from vision_training.models.instance_segmentation.instance_segmenter import InstanceSegmenter, SHARED_MASK_HEAD_KEY
from vision_training.models.instance_segmentation.mask_operations import compute_mask_scores


class InferenceInstanceSegmenter(InferenceDetector):
    """
    Inference-only version of the trained instance segmenter, used for deployment and local inference. It reuses the
    submodules of the trained model and adds to the four outputs of the InferenceDetector the [K, M, M] mask
    probabilities of the detected class of every detection, which the clients paste into their images inside the
    boxes, and the [K] mask scores of the detections when the masks are scored.
    """

    def __init__(self, model: InstanceSegmenter) -> None:
        """
        Wraps the submodules of a trained InstanceSegmenter, its mask heads included.

        Args:
            model (InstanceSegmenter): The trained instance segmenter.
        """
        super(InferenceInstanceSegmenter, self).__init__(model=model)
        self.mask_heads_dictionary = model.mask_heads_dictionary
        self.mask_fine_inputs = model.mask_fine_inputs
        self.shared_mask_head = model.shared_mask_head

        # Side of the exported masks, stored as a plain number for the export specification, and whether the mask
        # scores are exported after the masks
        self.mask_size = model.mask_size
        self.mask_scoring = model.mask_scoring

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Detects the objects of a batch of preprocessed images and predicts the mask of every detection, and its mask
        score when the masks are scored.

        Args:
            images (torch.Tensor): Float BGR images in [0, 255] of shape [B, 3, H, W], resized and padded to the network
                input size.

        Returns:
            Tuple[torch.Tensor, ...]: The detection boxes [B, K, 4] in network input pixels, their scores [B, K], their
                int64 labels [B, K] (0 for padding), the int64 number of valid detections per image [B], the mask
                probabilities [B, K, M, M] of the detected class inside every box (0 for padding), and when the masks
                are scored their mask scores [B, K] (compute_mask_scores, 0 for padding).
        """
        detected_objects = self.detect_objects(images=images)
        detections = detected_objects["detections"]

        # Level of every detection, read at its proposal
        detection_levels = torch.gather(input=detected_objects["proposal_levels"], dim=1,
                                        index=detections["final_detection_proposal_indices"])
        detection_masks, detection_mask_iou = self.predict_masks(
            feature_maps=detected_objects["feature_maps"], skip_feature_maps=detected_objects["skip_feature_maps"],
            detection_boxes=detections["final_detection_boxes"], detection_labels=detections["final_detection_labels"],
            detection_levels=detection_levels, number_detections=detections["number_detections"])
        outputs = (detections["final_detection_boxes"], detections["final_detection_scores"],
                   detections["final_detection_labels"], detections["number_detections"], detection_masks)

        # Mask score of every detection after its mask when the masks are scored
        if detection_mask_iou is not None:
            return outputs + (compute_mask_scores(detection_scores=detections["final_detection_scores"],
                                                  predicted_mask_iou=detection_mask_iou),)
        return outputs

    def predict_masks(self, feature_maps: Dict[str, torch.Tensor], skip_feature_maps: Dict[str, torch.Tensor],
                      detection_boxes: torch.Tensor, detection_labels: torch.Tensor, detection_levels: torch.Tensor,
                      number_detections: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predicts the mask of every detection with the mask head of its own level, and its predicted mask IoU when the
        masks are scored. Every head processes all the detections and each detection keeps the outputs of its own
        level, since selecting the detections of each level would give data-dependent tensor sizes that the ONNX
        export cannot handle, as in InferenceDetector.detect. The shared mask head pools every detection from every
        level instead, keeps the pooled features of its own level, and runs its layers once on all the detections. The
        fine features shared by the levels are computed once on all the detections.

        Args:
            feature_maps (Dict[str, torch.Tensor]): The backbone feature maps, keyed by level name.
            skip_feature_maps (Dict[str, torch.Tensor]): The skip outputs of the backbone keyed 'skip_<block>', empty
                without fine inputs.
            detection_boxes (torch.Tensor): The [B, K, 4] detection boxes.
            detection_labels (torch.Tensor): The [B, K] int64 detected classes.
            detection_levels (torch.Tensor): The [B, K] level index of every detection.
            number_detections (torch.Tensor): The [B] number of valid detections of every image.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: The [B, K, M, M] mask probabilities of the detected classes
                and the [B, K] predicted mask IoU of the detected classes (None when the masks are not scored), 0 for
                the padding detections.
        """
        batch_size, maximum_detections, _ = detection_boxes.shape

        # Boxes in the RoIAlign format, the image index first
        image_indices = torch.arange(end=batch_size, device=detection_boxes.device, dtype=detection_boxes.dtype)
        image_indices = image_indices.unsqueeze(dim=1).expand(-1, maximum_detections).reshape(-1, 1)
        region_boxes = torch.cat(tensors=[image_indices, detection_boxes.reshape(-1, 4)], dim=1)
        flat_levels = detection_levels.reshape(-1)
        flat_labels = detection_labels.reshape(-1)
        class_indices = flat_labels.reshape(-1, 1, 1, 1)

        detection_masks: Optional[torch.Tensor] = None
        detection_mask_iou: Optional[torch.Tensor] = None

        # Fine features of all the detections, read by the mask heads, None without fine inputs
        fine_features = self.mask_fine_inputs(proposal_boxes=region_boxes, skip_feature_maps=skip_feature_maps)

        # Shared head: every detection pooled from every level, the pooled features of its own level kept, then the
        # layers of the head run once
        if self.shared_mask_head:
            mask_head = self.mask_heads_dictionary[SHARED_MASK_HEAD_KEY]
            pooled_features: Optional[torch.Tensor] = None
            for level_index, level_name in enumerate(self.level_names):
                level_pooled_features = mask_head.pool_box_features(
                    proposal_boxes=region_boxes, input_tensor=feature_maps[level_name],
                    skip_feature_maps=skip_feature_maps, level_name=level_name)
                pooled_features = level_pooled_features if pooled_features is None else torch.where(
                    (flat_levels == level_index).reshape(-1, 1, 1, 1), level_pooled_features, pooled_features)
            mask_logits, detection_mask_iou = mask_head.predict_masks_and_mask_iou(
                pooled_features=pooled_features, fine_features=fine_features, class_indices=flat_labels)
            mask_size = mask_logits.shape[-1]
            detection_masks = torch.sigmoid(input=torch.gather(input=mask_logits, dim=1,
                                                               index=class_indices.expand(-1, 1, mask_size, mask_size)))

        # Run every level's head on all the detections and keep, for each detection, the mask and the predicted mask
        # IoU of its own level
        else:
            for level_index, level_name in enumerate(self.level_names):
                mask_head = self.mask_heads_dictionary[level_name]
                level_pooled_features = mask_head.pool_box_features(
                    proposal_boxes=region_boxes, input_tensor=feature_maps[level_name],
                    skip_feature_maps=skip_feature_maps, level_name=level_name)
                level_logits, level_mask_iou = mask_head.predict_masks_and_mask_iou(
                    pooled_features=level_pooled_features, fine_features=fine_features, class_indices=flat_labels)
                mask_size = level_logits.shape[-1]
                level_masks = torch.sigmoid(input=torch.gather(
                    input=level_logits, dim=1, index=class_indices.expand(-1, 1, mask_size, mask_size)))
                level_selection = flat_levels == level_index
                if detection_masks is None:
                    detection_masks, detection_mask_iou = level_masks, level_mask_iou
                else:
                    detection_masks = torch.where(level_selection.reshape(-1, 1, 1, 1), level_masks, detection_masks)
                    if level_mask_iou is not None and detection_mask_iou is not None:
                        detection_mask_iou = torch.where(level_selection, level_mask_iou, detection_mask_iou)

        # The padding detections get empty masks and a predicted mask IoU of 0
        mask_size = detection_masks.shape[-1]
        valid_detections = torch.arange(end=maximum_detections, device=detection_boxes.device).unsqueeze(dim=0) < \
            number_detections.unsqueeze(dim=1)
        detection_masks = detection_masks.reshape(batch_size, maximum_detections, mask_size, mask_size)
        detection_masks = torch.where(valid_detections.reshape(batch_size, maximum_detections, 1, 1), detection_masks,
                                      torch.zeros_like(input=detection_masks))
        if detection_mask_iou is None:
            return detection_masks, None
        detection_mask_iou = detection_mask_iou.reshape(batch_size, maximum_detections)
        return detection_masks, torch.where(valid_detections, detection_mask_iou,
                                            torch.zeros_like(input=detection_mask_iou))
