"""
ViewAugmenter, the random views of the self-supervised task: every image of a batch is transformed twice, on its
device, by a random resized crop, flips, a rotation, a colour jitter, a grayscale conversion and a Gaussian blur, each
drawn independently for every image and every view. The self-supervised model then pulls the two views of an image
together. Built by the self-supervised trainer from the 'Data.view_augmentation' section, whose values are verified by
configuration_processing/self_supervised_checks.py.
"""

import warnings
import torch

from torch import nn
from typing import Any, Dict, Tuple

# kornia decorates some of its functions with torch.jit.script when it is imported, which recent PyTorch versions
# report as deprecated; the notice is harmless and cannot be fixed on our side, so it is silenced for this import only
with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", message=r"`torch\.jit\.script` is deprecated", category=FutureWarning)
    from kornia import augmentation as kornia_augmentation

# Largest value of the pixels of the BGR images delivered by the data loaders
MAXIMUM_PIXEL_VALUE = 255.0


def build_view_transformations(view_augmentation_configuration: Dict[str, Any], image_size: int) -> nn.Sequential:
    """
    Builds the chain of random transformations producing one view of every image for the self-supervised trainer, in
    the order of SimCLR: the random resized crop, always applied, then the flips, the rotation, the colour jitter, the
    grayscale conversion and the Gaussian blur, each with its own probability. The kornia transformations draw the
    parameters of every image of the batch independently, on the CPU with the random generator of PyTorch (the same
    seed giving the same views on every device), then transform the images on their device. The order of the four
    colour jitter changes is drawn once per batch.

    Args:
        view_augmentation_configuration (Dict[str, Any]): The 'Data.view_augmentation' section.
        image_size (int): The side of the square network input, the size of the crops resized back.

    Returns:
        nn.Sequential: The transformations, mapping [B, 3, H, W] RGB images in [0, 1] to views of the same shape.
    """
    crop_configuration = view_augmentation_configuration["random_resized_crop"]
    rotation_configuration = view_augmentation_configuration["rotation"]
    colour_jitter_configuration = view_augmentation_configuration["colour_jitter"]
    blur_configuration = view_augmentation_configuration["gaussian_blur"]
    blur_kernel_size = blur_configuration["kernel_size"]

    # Crop of a random area and aspect ratio resized back to the input size, then the geometric transformations
    transformations = [kornia_augmentation.RandomResizedCrop(
                           size=(image_size, image_size), scale=tuple(crop_configuration["scale"]),
                           ratio=tuple(crop_configuration["aspect_ratio"]), p=1.0),
                       kornia_augmentation.RandomHorizontalFlip(
                           p=view_augmentation_configuration["left_right_flip"]["probability"]),
                       kornia_augmentation.RandomVerticalFlip(
                           p=view_augmentation_configuration["up_down_flip"]["probability"]),
                       kornia_augmentation.RandomRotation(degrees=rotation_configuration["maximum_degrees"],
                                                          p=rotation_configuration["probability"])]

    # Photometric transformations, which keep every structure of the image
    transformations += [kornia_augmentation.ColorJitter(
                            brightness=colour_jitter_configuration["brightness"],
                            contrast=colour_jitter_configuration["contrast"],
                            saturation=colour_jitter_configuration["saturation"],
                            hue=colour_jitter_configuration["hue"], p=colour_jitter_configuration["probability"]),
                        kornia_augmentation.RandomGrayscale(
                            p=view_augmentation_configuration["grayscale"]["probability"]),
                        kornia_augmentation.RandomGaussianBlur(
                            kernel_size=(blur_kernel_size, blur_kernel_size), sigma=tuple(blur_configuration["sigma"]),
                            p=blur_configuration["probability"])]
    return nn.Sequential(*transformations)


class ViewAugmenter:
    """
    Draws the random views of the images of the self-supervised task on their device (usually the GPU), right before
    the forward pass of every training and validation batch: each image gets two views transformed independently, so
    that the only thing the two views of an image share is its content. The views are computed in float32 whatever
    the mixed precision of the forward pass, on float BGR images in [0, 255] like every batch of the framework.
    """

    def __init__(self, view_augmentation_configuration: Dict[str, Any], image_size: int) -> None:
        """
        Builds the random transformations of the views.

        Args:
            view_augmentation_configuration (Dict[str, Any]): The 'Data.view_augmentation' section.
            image_size (int): The side of the square network input.
        """
        self.transformations = build_view_transformations(
            view_augmentation_configuration=view_augmentation_configuration, image_size=image_size)

    def draw_view(self, images: torch.Tensor) -> torch.Tensor:
        """
        Draws one random view of every image of a batch. The kornia colour transformations expect RGB images in [0, 1],
        so the BGR images are reversed and scaled before the transformations and brought back afterwards.

        Args:
            images (torch.Tensor): The [B, 3, H, W] float BGR images in [0, 255].

        Returns:
            torch.Tensor: The [B, 3, H, W] float BGR views in [0, 255].
        """
        rgb_images = images.flip(dims=(1,)) / MAXIMUM_PIXEL_VALUE
        rgb_views = self.transformations(input=rgb_images).clamp(min=0.0, max=1.0)
        return rgb_views.flip(dims=(1,)) * MAXIMUM_PIXEL_VALUE

    def create_views(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draws the two views of every image of a batch, in float32 even within the bfloat16 autocast of a mixed
        precision forward pass, so that the interpolations of the crops and rotations keep their precision.

        Args:
            images (torch.Tensor): The [B, 3, H, W] float BGR images in [0, 255].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The first and second [B, 3, H, W] views of the images.
        """
        with torch.autocast(device_type=images.device.type, enabled=False):
            float_images = images.to(dtype=torch.float32)
            return self.draw_view(images=float_images), self.draw_view(images=float_images)
