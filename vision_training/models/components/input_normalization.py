"""
Parameter-free normalization of the raw [0, 255] images at the entrance of the Backbone: the optional rescaling to
[-0.5, 0.5] and the optional per-image standardization, applied before feature extraction for every task.
"""

import math
import torch
from torch import nn


class InputNormalizer(nn.Module):
    """
    Normalizes the raw input images at the entrance of the Backbone.

    The data loaders intentionally deliver images in the raw [0, 255] pixel range.
    This parameter-free module optionally rescales them before feature extraction with two input
    normalization strategies:
        - input_normalization: maps pixels to [-0.5, 0.5] with x / 255 - 0.5.
        - per_sample_normalization: standardizes each image to zero mean and unit variance, mirroring
          tf.image.per_image_standardization.
    When both are enabled they are applied in that order.
    """

    def __init__(self, input_normalization: bool, per_sample_normalization: bool) -> None:
        """
        Stores the input normalization flags of the 'normalization' configuration section, read at the start of
        every backbone so that the raw [0, 255] images of the data loaders reach the first convolution rescaled.

        Args:
            input_normalization (bool): Whether to map pixel values from [0, 255] to [-0.5, 0.5].
            per_sample_normalization (bool): Whether to standardize each image to zero mean and unit variance.
        """
        super(InputNormalizer, self).__init__()

        self.input_normalization = bool(input_normalization)
        self.per_sample_normalization = bool(per_sample_normalization)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the enabled normalization strategies to a batch of images.

        Args:
            input_tensor (torch.Tensor): The raw image batch of shape [Batch, Channels, Height, Width].

        Returns:
            torch.Tensor: The normalized image batch of shape [Batch, Channels, Height, Width].
        """
        # Map pixels from [0, 255] to [-0.5, 0.5]
        if self.input_normalization:
            input_tensor = input_tensor / 255.0 - 0.5

        # Standardize each image independently over its channels and spatial dimensions
        if self.per_sample_normalization:
            number_elements = input_tensor[0].numel()
            mean = input_tensor.mean(dim=(1, 2, 3), keepdim=True)
            standard_deviation = input_tensor.std(dim=(1, 2, 3), keepdim=True, unbiased=False)

            # Lower-bound the deviation by 1/sqrt(N), as in TensorFlow, to avoid dividing by zero on uniform images
            adjusted_standard_deviation = standard_deviation.clamp(min=1.0 / math.sqrt(number_elements))
            input_tensor = (input_tensor - mean) / adjusted_standard_deviation

        return input_tensor
