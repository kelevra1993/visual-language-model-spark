"""
Inference-only version of the trained InstanceSegmenter: the fixed-shape detection of the InferenceDetector followed by
the mask head of every level on the fixed number of detections of every image, returning a fifth output with the mask
of every detection. Built by vision_training/deployment/export_specifications.py for the TorchScript and ONNX exports.
"""

import torch

from typing import Dict, Optional, Tuple

from vision_training.models.detection.inference_detector import InferenceDetector
from vision_training.models.instance_segmentation.instance_segmenter import InstanceSegmenter


class InferenceInstanceSegmenter(InferenceDetector):
    """
    Inference-only version of the trained instance segmenter, used for deployment and local inference. It reuses the
    submodules of the trained model and adds to the four outputs of the InferenceDetector the [K, M, M] mask
    probabilities of the detected class of every detection, which the clients paste into their images inside the
    boxes.
    """

    def __init__(self, model: InstanceSegmenter) -> None:
        """
        Wraps the submodules of a trained InstanceSegmenter, its mask heads included.

        Args:
            model (InstanceSegmenter): The trained instance segmenter.
        """
        super(InferenceInstanceSegmenter, self).__init__(model=model)
        self.mask_heads_dictionary = model.mask_heads_dictionary

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                     torch.Tensor]:
        """
        Detects the objects of a batch of preprocessed images and predicts the mask of every detection.

        Args:
            images (torch.Tensor): Float BGR images in [0, 255] of shape [B, 3, H, W], resized and padded to the network
                input size.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The detection boxes [B, K, 4]
                in network input pixels, their scores [B, K], their int64 labels [B, K] (0 for padding), the int64
                number of valid detections per image [B], and the mask probabilities [B, K, M, M] of the detected
                class inside every box (0 for padding).
        """
        detected_objects = self.detect_objects(images=images)
        detections = detected_objects["detections"]

        # Level of every detection, read at its proposal
        detection_levels = torch.gather(input=detected_objects["proposal_levels"], dim=1,
                                        index=detections["final_detection_proposal_indices"])
        detection_masks = self.predict_masks(feature_maps=detected_objects["feature_maps"],
                                             detection_boxes=detections["final_detection_boxes"],
                                             detection_labels=detections["final_detection_labels"],
                                             detection_levels=detection_levels,
                                             number_detections=detections["number_detections"])
        return (detections["final_detection_boxes"], detections["final_detection_scores"],
                detections["final_detection_labels"], detections["number_detections"], detection_masks)

    def predict_masks(self, feature_maps: Dict[str, torch.Tensor], detection_boxes: torch.Tensor,
                      detection_labels: torch.Tensor, detection_levels: torch.Tensor,
                      number_detections: torch.Tensor) -> torch.Tensor:
        """
        Predicts the mask of every detection with the mask head of its own level. Every head processes all the
        detections and each detection keeps the output of its own level, since selecting the detections of each level
        would give data-dependent tensor sizes that the ONNX export cannot handle, as in InferenceDetector.detect.

        Args:
            feature_maps (Dict[str, torch.Tensor]): The backbone feature maps, keyed by level name.
            detection_boxes (torch.Tensor): The [B, K, 4] detection boxes.
            detection_labels (torch.Tensor): The [B, K] int64 detected classes.
            detection_levels (torch.Tensor): The [B, K] level index of every detection.
            number_detections (torch.Tensor): The [B] number of valid detections of every image.

        Returns:
            torch.Tensor: The [B, K, M, M] mask probabilities of the detected classes, 0 for the padding detections.
        """
        batch_size, maximum_detections, _ = detection_boxes.shape

        # Boxes in the RoIAlign format, the image index first
        image_indices = torch.arange(end=batch_size, device=detection_boxes.device, dtype=detection_boxes.dtype)
        image_indices = image_indices.unsqueeze(dim=1).expand(-1, maximum_detections).reshape(-1, 1)
        region_boxes = torch.cat(tensors=[image_indices, detection_boxes.reshape(-1, 4)], dim=1)
        flat_levels = detection_levels.reshape(-1)
        class_indices = detection_labels.reshape(-1, 1, 1, 1)

        detection_masks: Optional[torch.Tensor] = None

        # Run every level's head on all the detections and keep, for each detection, the mask of its own level
        for level_index, level_name in enumerate(self.level_names):
            level_logits = self.mask_heads_dictionary[level_name](proposal_boxes=region_boxes,
                                                                  input_tensor=feature_maps[level_name])
            mask_size = level_logits.shape[-1]
            level_masks = torch.sigmoid(input=torch.gather(input=level_logits, dim=1,
                                                           index=class_indices.expand(-1, 1, mask_size, mask_size)))
            if detection_masks is None:
                detection_masks = level_masks
            else:
                detection_masks = torch.where((flat_levels == level_index).reshape(-1, 1, 1, 1), level_masks,
                                              detection_masks)

        # The padding detections get empty masks
        mask_size = detection_masks.shape[-1]
        valid_detections = torch.arange(end=maximum_detections, device=detection_boxes.device).unsqueeze(dim=0) < \
            number_detections.unsqueeze(dim=1)
        detection_masks = detection_masks.reshape(batch_size, maximum_detections, mask_size, mask_size)
        return torch.where(valid_detections.reshape(batch_size, maximum_detections, 1, 1), detection_masks,
                           torch.zeros_like(input=detection_masks))
