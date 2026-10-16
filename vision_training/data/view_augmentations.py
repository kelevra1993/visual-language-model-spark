"""
ViewAugmenter, the random views of the self-supervised task: every image of a batch is transformed twice, on its
device, by a random resized crop, flips, a rotation, a colour jitter, a grayscale conversion and a Gaussian blur, each
drawn independently for every image and every view. The self-supervised model then pulls the two views of an image
together. Built by the self-supervised trainer from the 'Data.view_augmentation' section, whose values are verified by
configuration_processing/self_supervised_checks.py. The transformations drawn for every view, with their values, can
also be described for the information panels of the 'view_input_images' diagnosis mode.
"""

import warnings
import torch

from torch import nn
from typing import Any, Dict, List, Optional, Tuple

# kornia decorates some of its functions with torch.jit.script when it is imported, which recent PyTorch versions
# report as deprecated; the notice is harmless and cannot be fixed on our side, so it is silenced for this import only
with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", message=r"`torch\.jit\.script` is deprecated", category=FutureWarning)
    from kornia import augmentation as kornia_augmentation

# Largest value of the pixels of the BGR images delivered by the data loaders
MAXIMUM_PIXEL_VALUE = 255.0

# Names of the transformations built by build_view_transformations, in their order, as the 'Data.view_augmentation'
# keys name them, used to describe the drawn views
VIEW_TRANSFORMATION_NAMES = ["random_resized_crop", "left_right_flip", "up_down_flip", "rotation", "colour_jitter",
                             "grayscale", "gaussian_blur"]


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


def describe_view_parameters(transformation_name: str, parameters: Dict[str, List[Any]], applied_index: int) -> str:
    """
    Writes the values a kornia transformation drew for one image of the last view, for the information panel of the
    input viewer. The value lists of kornia only hold the images the transformation was applied to, so the image is
    found at its rank among them.

    Args:
        transformation_name (str): The name of the transformation, from VIEW_TRANSFORMATION_NAMES.
        parameters (Dict[str, List[Any]]): The parameters kornia kept for the last batch, converted to lists.
        applied_index (int): The rank of the image among the images the transformation was applied to.

    Returns:
        str: The drawn values, empty for the transformations without random values (the flips and the grayscale).
    """
    if transformation_name == "random_resized_crop":
        # The corners are inclusive pixel indices, [x1, y1], [x2, y1], [x2, y2], [x1, y2], of the [H, W] input
        top_left_corner, _, bottom_right_corner, _ = parameters["src"][applied_index]
        crop_width = round(bottom_right_corner[0] - top_left_corner[0] + 1)
        crop_height = round(bottom_right_corner[1] - top_left_corner[1] + 1)
        input_height, input_width = parameters["input_size"][applied_index]
        area_fraction = crop_width * crop_height / (input_height * input_width)
        return (f"{crop_width}x{crop_height} pixels at x {round(top_left_corner[0])}, y {round(top_left_corner[1])} "
                f"({area_fraction * 100:.0f} % of the area)")
    if transformation_name == "rotation":
        return f"{parameters['degrees'][applied_index]:+.1f} degrees"
    if transformation_name == "colour_jitter":
        # The hue factor is a fraction of the colour wheel, written in degrees
        return (f"brightness x{parameters['brightness_factor'][applied_index]:.2f}, "
                f"contrast x{parameters['contrast_factor'][applied_index]:.2f}, "
                f"saturation x{parameters['saturation_factor'][applied_index]:.2f}, "
                f"hue {parameters['hue_factor'][applied_index] * 360.0:+.1f} degrees")
    if transformation_name == "gaussian_blur":
        return f"sigma {parameters['sigma'][applied_index]:.2f}"
    return ""


def describe_view_transformations(transformations: nn.Sequential, number_images: int) -> List[List[str]]:
    """
    Describes the transformations drawn for every image by the last call of the view transformations, read from the
    parameters kornia keeps after a forward pass, for the information panels of the input viewer of the
    self-supervised task. It must be called right after the view is drawn, the next view overwriting the parameters.

    Args:
        transformations (nn.Sequential): The transformations of build_view_transformations, after a forward pass.
        number_images (int): The number of images of the last batch.

    Returns:
        List[List[str]]: For every image, one line per applied transformation, its name and drawn values, in order.
    """
    descriptions = [[] for image_index in range(number_images)]
    for transformation_name, transformation in zip(VIEW_TRANSFORMATION_NAMES, transformations):
        # Convert the parameters to lists once, the images the transformation applied to being flagged by batch_prob
        parameters = {key: value.tolist() for key, value in transformation._params.items() if torch.is_tensor(value)}
        applied_images = [probability > 0.5 for probability in parameters["batch_prob"]]

        # Describe every applied image from its rank among the applied images
        applied_index = 0
        for image_index in range(number_images):
            if not applied_images[image_index]:
                continue
            drawn_values = describe_view_parameters(transformation_name=transformation_name, parameters=parameters,
                                                    applied_index=applied_index)
            descriptions[image_index].append(f"{transformation_name} : {drawn_values}" if drawn_values
                                             else transformation_name)
            applied_index += 1
    return descriptions


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

    def create_views(self, images: torch.Tensor, describe_views: bool = False) -> Tuple[
                     torch.Tensor, torch.Tensor, Optional[List[List[List[str]]]]]:
        """
        Draws the two views of every image of a batch, in float32 even within the bfloat16 autocast of a mixed
        precision forward pass, so that the interpolations of the crops and rotations keep their precision. The input
        viewer also asks for the transformations drawn for every view, written in its information panels.

        Args:
            images (torch.Tensor): The [B, 3, H, W] float BGR images in [0, 255].
            describe_views (bool): Whether to describe the drawn transformations, which reads their parameters back
                from the device. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[List[List[List[str]]]]]: The first and second [B, 3, H, W] views
                of the images, and the description lines of every image for each of the two views (None unless
                describe_views).
        """
        with torch.autocast(device_type=images.device.type, enabled=False):
            float_images = images.to(dtype=torch.float32)

            # Every view is described right after it is drawn, the second view overwriting the kornia parameters
            view_descriptions = [] if describe_views else None
            views = []
            for view_index in range(2):
                views.append(self.draw_view(images=float_images))
                if describe_views:
                    view_descriptions.append(describe_view_transformations(transformations=self.transformations,
                                                                           number_images=images.shape[0]))
            return views[0], views[1], view_descriptions
