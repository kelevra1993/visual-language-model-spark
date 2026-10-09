"""
Inference-only wrapper of the trained SemanticSegmenter returning the class map and the class probabilities of every
pixel of preprocessed images, without masks nor losses. Built by vision_training/deployment/export_specifications.py
for the TorchScript and ONNX exports served by Triton.
"""

import torch

from torch import nn
from typing import Tuple

from vision_training.models.segmentation.semantic_segmenter import SemanticSegmenter


class InferenceSegmenter(nn.Module):
    """
    Inference-only version of the trained semantic segmenter, used for deployment (TorchScript and ONNX exports served
    by Triton) and for local inference: it returns the class index of every pixel of a batch of preprocessed images,
    with the probabilities of every class, without masks nor losses, the deep supervision heads of the training
    being never run. Both are at the network input resolution, restore_prediction_to_original_image of
    vision_training/data/preprocessing.py bringing them back to their original image; a run trained on tiles
    ('Data.tiling') segments its scaled images tile by tile with segment_image_by_tiles of sliding_window.py.
    """

    def __init__(self, model: SemanticSegmenter) -> None:
        """
        Wraps a trained semantic segmenter.

        Args:
            model (SemanticSegmenter): The trained segmenter.
        """
        super(InferenceSegmenter, self).__init__()
        self.model = model
        self.input_image_size = model.backbone.input_image_size
        self.number_classes = model.number_classes

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Segments a batch of preprocessed images.

        Args:
            images (torch.Tensor): Float BGR images in [0, 255] of shape [B, 3, H, W], resized and padded to the network
                input size.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The [B, H, W] int64 class index of every pixel (the most probable class),
                and the [B, K, H, W] class probabilities of every pixel.
        """
        outputs = self.model(input_tensor=images)
        return outputs["class_logits"].argmax(dim=1), outputs["class_probabilities"]
