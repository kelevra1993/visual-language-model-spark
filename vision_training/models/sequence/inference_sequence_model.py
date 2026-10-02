import torch

from torch import nn

from vision_training.models.task_model import TaskModel


class InferenceSequenceModel(nn.Module):
    """
    Inference-only version of the trained sequence models (sequence classifier and sequence labeller), used for
    deployment (TorchScript and ONNX exports served by Triton) and for local inference: it returns the class
    probabilities of windows of real frames, one vector per window (many to one) or per frame (many to many). The
    windows sent to it hold no padding frame, so it runs the path of the models without sequence mask, whose
    operations keep fixed shapes (no gathering of the real frames, no packed sequences), as the exports require. The
    number of frames may change from one call to the next, up to the sequence length of the training.
    """

    def __init__(self, model: TaskModel) -> None:
        """
        Wraps a trained sequence classifier or sequence labeller.

        Args:
            model (TaskModel): The trained sequence model.
        """
        super(InferenceSequenceModel, self).__init__()
        self.model = model
        self.input_image_size = model.backbone.input_image_size
        self.number_classes = model.number_classes
        self.sequence_length = model.sequence_length

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Classifies a batch of windows of preprocessed frames.

        Args:
            images (torch.Tensor): Float BGR frames in [0, 255] of shape [B, T, 3, H, W], resized and padded to the
                network input size, every frame being real.

        Returns:
            torch.Tensor: The [B, K] class probabilities of every window, or the [B, T, K] class probabilities of every
                frame for the sequence labeller.
        """
        return self.model(input_tensor=images)["class_probabilities"]
