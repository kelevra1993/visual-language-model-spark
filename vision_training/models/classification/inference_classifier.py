"""
Inference-only wrapper of the trained ImageClassifier returning the class probabilities of preprocessed images,
without labels nor loss, the class logits being divided by a logit temperature before the softmax (or sigmoid). Built
by vision_training/deployment/export_specifications.py for the TorchScript and ONNX exports served by Triton, the
export setting the temperature of the evaluation whose class thresholds are deployed.
"""

import torch

from torch import nn

from vision_training.losses.classification_losses import convert_logits_to_probabilities
from vision_training.models.classification.image_classifier import ImageClassifier


class InferenceClassifier(nn.Module):
    """
    Inference-only version of the trained image classifier, used for deployment (TorchScript and ONNX exports served
    by Triton) and for local inference: it returns the class probabilities of a batch of preprocessed images, without
    labels nor loss. The class logits are divided by the 'logit_temperature' buffer before the softmax, 1 by default,
    a temperature above 1 spreading probabilities squeezed against 0 and 1 without changing the predicted class. The
    buffer is written into the exported graphs as a constant.
    """

    def __init__(self, model: ImageClassifier) -> None:
        """
        Wraps a trained image classifier, with a logit temperature of 1 (the probabilities of the training).

        Args:
            model (ImageClassifier): The trained classifier.
        """
        super(InferenceClassifier, self).__init__()
        self.model = model
        self.input_image_size = model.backbone.input_image_size
        self.number_classes = model.number_classes
        self.output_mode = model.output_mode

        # Temperature dividing the class logits, on the device of the classifier so that the exports trace it there
        self.register_buffer(name="logit_temperature",
                             tensor=torch.tensor(data=1.0, device=model.class_weights.device, dtype=torch.float32))

    def set_logit_temperature(self, logit_temperature: float) -> None:
        """
        Sets the temperature dividing the class logits before the softmax, read by deployment/export_model.py from
        the 'class_thresholds.json' of the evaluation it deploys, so that the exported probabilities are those the
        thresholds were found on.

        Args:
            logit_temperature (float): The temperature, above 0.

        Raises:
            ValueError: If the temperature is not above 0.
        """
        if not logit_temperature > 0:
            raise ValueError(f"The logit temperature must be above 0, got {logit_temperature}.")
        self.logit_temperature.fill_(value=logit_temperature)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Classifies a batch of preprocessed images.

        Args:
            images (torch.Tensor): Float BGR images in [0, 255] of shape [B, 3, H, W], resized and padded to the network
                input size.

        Returns:
            torch.Tensor: The [B, K] class probabilities (softmax, or sigmoid in multi-label mode) of the class logits
                divided by the logit temperature.
        """
        class_logits = self.model(input_tensor=images)["class_logits"]
        return convert_logits_to_probabilities(class_logits=class_logits / self.logit_temperature,
                                               output_mode=self.output_mode)
