"""
Inference-only wrapper of the trained SequenceComparator taking the before and after windows as two inputs and
returning the class probabilities of the pairs. Built by vision_training/deployment/export_specifications.py for the
TorchScript and ONNX exports served by Triton.
"""

import torch

from torch import nn

from vision_training.models.sequence.sequence_comparator import SequenceComparator


class InferenceSequenceComparator(nn.Module):
    """
    Inference-only version of the trained sequence comparator, used for deployment (TorchScript and ONNX exports
    served by Triton, with the two inputs 'before_images' and 'after_images') and for local inference: it returns the
    class probabilities of pairs of windows of real frames. The windows sent to it hold no padding frame, so it runs
    the path of the comparator without sequence masks, which encodes and summarises every member on its own: the
    before and after windows may have different numbers of frames, each changing from one call to the next up to the
    sequence length of the training.
    """

    def __init__(self, model: SequenceComparator) -> None:
        """
        Wraps a trained sequence comparator.

        Args:
            model (SequenceComparator): The trained sequence comparator.
        """
        super(InferenceSequenceComparator, self).__init__()
        self.model = model
        self.input_image_size = model.backbone.input_image_size
        self.number_classes = model.number_classes
        self.sequence_length = model.sequence_length

    def forward(self, before_images: torch.Tensor, after_images: torch.Tensor) -> torch.Tensor:
        """
        Classifies a batch of pairs of windows of preprocessed frames.

        Args:
            before_images (torch.Tensor): Float BGR frames in [0, 255] of shape [B, T1, 3, H, W], the windows of the
                before members, resized and padded to the network input size, every frame being real.
            after_images (torch.Tensor): Float BGR frames in [0, 255] of shape [B, T2, 3, H, W], the windows of the
                after members.

        Returns:
            torch.Tensor: The [B, K] class probabilities of every pair.
        """
        return self.model(before_images=before_images, after_images=after_images)["class_probabilities"]
