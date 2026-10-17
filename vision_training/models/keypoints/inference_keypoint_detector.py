"""
Inference-only wrapper of the trained bottom-up KeypointDetector returning the keypoint candidates of every heatmap
channel of preprocessed whole images (their positions, scores and tags), without targets nor loss, for the keypoint
detection of a deployment: the client groups the candidates of every image into objects with group_keypoint_candidates
(models/keypoints/keypoint_grouping.py), the channels of every class being given by build_keypoint_channel_table.
Built by vision_training/deployment/export_specifications.py for the TorchScript and ONNX exports served by Triton.
"""

import torch

from torch import nn
from typing import Tuple

from vision_training.models.keypoints.hrnet import BRANCH_SCALE_FACTOR
from vision_training.models.keypoints.keypoint_detector import HEATMAP_STRIDE, KeypointDetector


class InferenceKeypointDetector(nn.Module):
    """
    Inference-only version of the trained keypoint detector, used for deployment (TorchScript and ONNX exports served
    by Triton) and for local inference: it extracts the candidates of every heatmap channel of every image, their
    grouping into objects being left to the client, a Python function outside the exported network.
    """

    def __init__(self, model: KeypointDetector) -> None:
        """
        Wraps a trained keypoint detector.

        Args:
            model (KeypointDetector): The trained keypoint detector.
        """
        super(InferenceKeypointDetector, self).__init__()
        self.model = model
        self.input_image_size = model.image_size
        self.number_channels = model.number_channels

        # Number of candidates per channel, capped by the number of cells of the heatmaps the candidates are taken on
        heatmap_size = model.image_size // HEATMAP_STRIDE * (BRANCH_SCALE_FACTOR if model.higher_resolution else 1)
        self.number_candidates = min(model.maximum_candidates, heatmap_size * heatmap_size)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts the keypoint candidates of a batch of preprocessed images.

        Args:
            images (torch.Tensor): Float BGR images in [0, 255] of shape [B, 3, image_size, image_size].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The [B, C, N] candidates of every heatmap channel by
                decreasing score: their [B, C, N, 2] (x, y) positions in image pixels, their [B, C, N] scores and
                their [B, C, N] tags, N being number_candidates.
        """
        outputs = self.model(input_tensor=images)
        return outputs["candidate_positions"], outputs["candidate_scores"], outputs["candidate_tags"]
