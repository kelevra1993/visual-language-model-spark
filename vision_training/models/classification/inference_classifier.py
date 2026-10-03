"""
Inference-only wrapper of the trained ImageClassifier returning the class probabilities of preprocessed images,
without labels nor loss. Built by vision_training/deployment/export_specifications.py for the TorchScript and ONNX
exports served by Triton.
"""

import torch

from torch import nn

from vision_training.models.classification.image_classifier import ImageClassifier


class InferenceClassifier(nn.Module):
    """
    Inference-only version of the trained image classifier, used for deployment (TorchScript and ONNX exports served
    by Triton) and for local inference: it returns the class probabilities of a batch of preprocessed images, without
    labels nor loss.
    """

    def __init__(self, model: ImageClassifier) -> None:
        """
        Wraps a trained image classifier.

        Args:
            model (ImageClassifier): The trained classifier.
        """
        super(InferenceClassifier, self).__init__()
        self.model = model
        self.input_image_size = model.backbone.input_image_size
        self.number_classes = model.number_classes

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Classifies a batch of preprocessed images.

        Args:
            images (torch.Tensor): Float BGR images in [0, 255] of shape [B, 3, H, W], resized and padded to the network
                input size.

        Returns:
            torch.Tensor: The [B, K] class probabilities (softmax, or sigmoid in multi-label mode).
        """
        return self.model(input_tensor=images)["class_probabilities"]
