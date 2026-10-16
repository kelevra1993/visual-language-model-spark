"""
Inference-only wrapper of the trained SelfSupervisedModel returning the L2 normalized representations of preprocessed
images, without views nor loss, for the similarity searches and nearest neighbour classifications of a deployment.
Built by vision_training/deployment/export_specifications.py for the TorchScript and ONNX exports served by Triton.
"""

import torch

from torch import nn
from torch.nn import functional as torch_functional

from vision_training.models.self_supervised.self_supervised_model import SelfSupervisedModel


class InferenceEncoder(nn.Module):
    """
    Inference-only version of the trained self-supervised model, used for deployment (TorchScript and ONNX exports
    served by Triton) and for local inference: it returns the representation of every image, the pooled final feature
    map of the backbone before the projection head, L2 normalized so that the dot product of two representations is
    their cosine similarity, the similarity of the k-nearest neighbour evaluation of the training.
    """

    def __init__(self, model: SelfSupervisedModel) -> None:
        """
        Wraps a trained self-supervised model.

        Args:
            model (SelfSupervisedModel): The trained self-supervised model.
        """
        super(InferenceEncoder, self).__init__()
        self.model = model
        self.input_image_size = model.backbone.input_image_size
        self.representation_size = model.representation_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of preprocessed images.

        Args:
            images (torch.Tensor): Float BGR images in [0, 255] of shape [B, 3, H, W], resized and padded to the network
                input size.

        Returns:
            torch.Tensor: The [B, F] L2 normalized representations.
        """
        return torch_functional.normalize(input=self.model.represent(images=images), dim=1)
