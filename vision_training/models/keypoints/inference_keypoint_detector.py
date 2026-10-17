"""
Inference-only wrapper of the trained KeypointDetector returning the decoded keypoints, their scores and the heatmaps of
preprocessed object crops, without targets nor loss, for the keypoint detection of a deployment: the client cuts every
detected object out of its image (data/keypoints/crop_transform.py, without augmentation), keeps the channels of the
class of the object (consecutive channels per class, see build_keypoint_channel_table) and brings the keypoints back
to its image with the inverse affine of the crop. Built by vision_training/deployment/export_specifications.py for the
TorchScript and ONNX exports served by Triton.
"""

import torch

from torch import nn
from typing import Tuple

from vision_training.models.keypoints.keypoint_detector import KeypointDetector


class InferenceKeypointDetector(nn.Module):
    """
    Inference-only version of the trained keypoint detector, used for deployment (TorchScript and ONNX exports served
    by Triton) and for local inference: it decodes every heatmap channel of every crop, the selection of the channels
    of the class of the crop being left to the client, which knows the class of the box it cut the crop from.
    """

    def __init__(self, model: KeypointDetector) -> None:
        """
        Wraps a trained keypoint detector.

        Args:
            model (KeypointDetector): The trained keypoint detector.
        """
        super(InferenceKeypointDetector, self).__init__()
        self.model = model
        self.crop_height = model.crop_height
        self.crop_width = model.crop_width
        self.number_channels = len(model.channel_names)

    def forward(self, crops: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts the keypoints of a batch of preprocessed object crops.

        Args:
            crops (torch.Tensor): Float BGR crops in [0, 255] of shape [B, 3, crop height, crop width].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The [B, C, 2] keypoints (x, y) in crop pixels and the
                [B, C] scores of every heatmap channel, and the [B, C, crop height / 4, crop width / 4] heatmaps.
        """
        outputs = self.model(input_tensor=crops)
        return outputs["keypoints"], outputs["keypoint_scores"], outputs["heatmaps"]
