"""
ViewAugmenter, the random views of the self-supervised task: every image of a batch is transformed twice, on its
device, by a random resized crop, flips, a rotation, a perspective distortion, a jigsaw of its pieces, a colour jitter,
a grayscale conversion, a gamma, a haze, a linear illumination, a plasma shadow, a Gaussian blur, a motion blur, a
solarization, a Gaussian noise, a black-out of the cells of a grid and a random erasing, each drawn independently for
every image and every view (but the kernel size of the motion blur, drawn once per batch by kornia), with
probabilities that may differ between the first and the second view (the asymmetric views of BYOL and Barlow Twins).
The self-supervised model then pulls the two views of an image together. Built by the self-supervised trainer from the
'Data.view_augmentation' section, and by the classification trainer from
'ClassificationHead.view_losses.view_augmentation' to draw the second view of every image of the view losses of the
image classifier, both sections being verified by check_view_augmentation_configuration of
configuration_processing/common_checks.py. The transformations drawn for every view, with their values, can also be
described for the information panels of the 'view_input_images' diagnosis mode.
"""

import warnings
import torch

from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union

from vision_training.data.augmentations import (BLACK_OUT_CELL_PROBABILITY, apply_black_out, apply_haze,
                                                 describe_black_out)

# kornia decorates some of its functions with torch.jit.script when it is imported, which recent PyTorch versions
# report as deprecated; the notice is harmless and cannot be fixed on our side, so it is silenced for this import only
with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", message=r"`torch\.jit\.script` is deprecated", category=FutureWarning)
    from kornia import augmentation as kornia_augmentation
    from kornia.augmentation import random_generator as kornia_random_generator

# Largest value of the pixels of the BGR images delivered by the data loaders
MAXIMUM_PIXEL_VALUE = 255.0

# Names of the transformations built by build_view_transformations, in their order, as the 'Data.view_augmentation'
# keys name them, used to describe the drawn views
VIEW_TRANSFORMATION_NAMES = ["random_resized_crop", "left_right_flip", "up_down_flip", "rotation", "perspective",
                             "jigsaw", "colour_jitter", "grayscale", "gamma", "haze", "linear_illumination",
                             "plasma_shadow", "gaussian_blur", "motion_blur", "solarize", "gaussian_noise", "black_out",
                             "random_erasing"]

# Number of views drawn for every image of the self-supervised task, each with its own chain of transformations; the
# view losses of the image classifier draw the second one only
NUMBER_VIEWS = 2
SECOND_VIEW_INDEX = 1

# Value filling the rectangles of the random erasing, black in the RGB images in [0, 1]
ERASING_FILL_VALUE = 0.0


class RandomHaze(kornia_augmentation.IntensityAugmentationBase2D):
    """
    Haze of the views, the fog, condensation or low contrast of a camera, which kornia does not provide: every hazy
    image is blended with a uniform airlight, view * (1 - strength) + strength * airlight, by apply_haze, the blend
    of the random_haze type of the data augmentation of the other tasks (augmentations.py). The strength and the
    airlight of every image are drawn on the CPU by the random generator of PyTorch, like the parameters of the kornia
    transformations, so that a seed gives the same haze on every device and the drawn values can be described.
    """

    def __init__(self, strength: Tuple[float, float], airlight: Tuple[float, float], probability: float) -> None:
        """
        Builds the haze transformation of a chain of view transformations.

        Args:
            strength (Tuple[float, float]): The range of the share of the airlight in a hazy image, within [0, 1].
            airlight (Tuple[float, float]): The range of the intensity of the airlight, within [0, 1].
            probability (float): The probability of hazing an image.
        """
        super().__init__(p=probability, same_on_batch=False, keepdim=False)
        self._param_generator = kornia_random_generator.PlainUniformGenerator(
            (strength, "strength", None, None), (airlight, "airlight", None, None))

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any],
                        transform: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Blends the images drawn for the haze with their airlight, called by kornia on these images only.

        Args:
            input (torch.Tensor): The [B, 3, H, W] RGB images in [0, 1] drawn for the haze.
            params (Dict[str, torch.Tensor]): The 'strength' and 'airlight' drawn for every one of them.
            flags (Dict[str, Any]): The flags of the kornia transformation, unused.
            transform (Optional[torch.Tensor]): The transformation matrix of kornia, unused.

        Returns:
            torch.Tensor: The hazy images.
        """
        strength = params["strength"].to(input).view(-1, 1, 1, 1)
        airlight = params["airlight"].to(input).view(-1, 1, 1, 1)
        return apply_haze(images=input, strength=strength, airlight=airlight)


class RandomBlackOut(kornia_augmentation.IntensityAugmentationBase2D):
    """
    Black-out of the views, which kornia does not provide: every view drawn for it is cut into a grid of 'grid_size'
    rows and columns whose every cell is blacked out with a fixed probability of 0.2, by apply_black_out, the black-out
    of the random_black_out type of the data augmentation of the other tasks (augmentations.py), so that the backbone
    learns to match views missing parts of the scene. The hidden cells of every image are drawn on the CPU by the
    random generator of PyTorch, like the parameters of the kornia transformations, so that a seed gives the same
    black-out on every device and the hidden cells can be described.
    """

    def __init__(self, probability: float, grid_size: int) -> None:
        """
        Builds the black-out transformation of a chain of view transformations.

        Args:
            probability (float): The probability of blacking out cells of an image.
            grid_size (int): The number of rows and columns of the grid, 'black_out.grid_size'.
        """
        super().__init__(p=probability, same_on_batch=False, keepdim=False)
        self.grid_size = grid_size

    def generate_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        """
        Draws the hidden cells of every image the black-out applies to, called by kornia with the number of these
        images.

        Args:
            batch_shape (Tuple[int, ...]): The shape of the images drawn for the black-out, their number first.

        Returns:
            Dict[str, torch.Tensor]: The boolean [N, rows, columns] grids of the hidden cells, under 'hidden_cells'.
        """
        return {"hidden_cells": torch.rand(size=(batch_shape[0], self.grid_size, self.grid_size))
                < BLACK_OUT_CELL_PROBABILITY}

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any],
                        transform: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Blacks out the hidden cells of the images drawn for the black-out, called by kornia on these images only.

        Args:
            input (torch.Tensor): The [B, 3, H, W] RGB images in [0, 1] drawn for the black-out.
            params (Dict[str, torch.Tensor]): The 'hidden_cells' grid drawn for every one of them.
            flags (Dict[str, Any]): The flags of the kornia transformation, unused.
            transform (Optional[torch.Tensor]): The transformation matrix of kornia, unused.

        Returns:
            torch.Tensor: The images with their hidden cells at 0.
        """
        return apply_black_out(images=input, hidden_cells=params["hidden_cells"].unsqueeze(dim=1))


def get_view_probability(probability: Union[float, List[float]], view_index: int) -> float:
    """
    Reads the probability of a transformation for one of the two views when its chain is built: a number applies to
    both views, a list [first view, second view] gives each view its own probability, as the asymmetric views of BYOL
    and Barlow Twins (the Gaussian blur nearly always on the first view, the solarization on the second one only).

    Args:
        probability (Union[float, List[float]]): The probability of the configuration.
        view_index (int): The index of the view, 0 for the first view and 1 for the second one.

    Returns:
        float: The probability of the transformation in the chain of the view.
    """
    return float(probability[view_index]) if isinstance(probability, list) else float(probability)


def build_view_transformations(view_augmentation_configuration: Dict[str, Any], image_size: int,
                               view_index: int) -> nn.Sequential:
    """
    Builds the chain of random transformations producing one of the two views of every image for the self-supervised
    trainer, or the second view of the view losses of the image classifier. The random resized crop, always applied,
    comes first, then the geometric transformations (flips, rotation, perspective, jigsaw), the photometric ones
    (colour jitter, grayscale, gamma, haze, linear illumination, plasma shadow), the Gaussian and motion blurs, the
    solarization, the Gaussian noise after both blurs so that they do not smooth it away, and the black-out of grid
    cells and the random erasing last so that the occluded areas stay plainly black, each with its probability for this
    view. The kornia transformations draw the parameters of every image of the batch independently, on the CPU with the
    random generator of PyTorch (the same seed giving the same views on every device), then transform the images on
    their device; the plasma shadow and the Gaussian noise draw their fractal map and their noise on the device, the
    only transformations whose views differ between devices for the same seed. The order of the four colour jitter
    changes and the kernel size of the motion blur, an odd size of 'motion_blur.kernel_size' with both bounds
    included, are drawn once per batch, the angle and direction of the motion blur staying drawn for every image.

    Args:
        view_augmentation_configuration (Dict[str, Any]): The 'Data.view_augmentation' section.
        image_size (int): The side of the square network input, the size of the crops resized back.
        view_index (int): The index of the drawn view, choosing the probabilities given per view.

    Returns:
        nn.Sequential: The transformations, in the order of VIEW_TRANSFORMATION_NAMES, mapping [B, 3, H, W] RGB images
            in [0, 1] to views of the same shape.
    """
    # Probability of every transformation but the crop for this view, and the settings of every transformation
    probabilities = {transformation_name: get_view_probability(
                         probability=view_augmentation_configuration[transformation_name]["probability"],
                         view_index=view_index) for transformation_name in VIEW_TRANSFORMATION_NAMES[1:]}
    crop_configuration = view_augmentation_configuration["random_resized_crop"]
    rotation_configuration = view_augmentation_configuration["rotation"]
    colour_jitter_configuration = view_augmentation_configuration["colour_jitter"]
    haze_configuration = view_augmentation_configuration["haze"]
    plasma_shadow_configuration = view_augmentation_configuration["plasma_shadow"]
    blur_kernel_size = view_augmentation_configuration["gaussian_blur"]["kernel_size"]
    motion_blur_configuration = view_augmentation_configuration["motion_blur"]
    solarize_configuration = view_augmentation_configuration["solarize"]
    erasing_configuration = view_augmentation_configuration["random_erasing"]

    # Crop of a random area and aspect ratio resized back to the input size, then the geometric transformations
    transformations = [kornia_augmentation.RandomResizedCrop(
                           size=(image_size, image_size), scale=tuple(crop_configuration["scale"]),
                           ratio=tuple(crop_configuration["aspect_ratio"]), p=1.0),
                       kornia_augmentation.RandomHorizontalFlip(p=probabilities["left_right_flip"]),
                       kornia_augmentation.RandomVerticalFlip(p=probabilities["up_down_flip"]),
                       kornia_augmentation.RandomRotation(degrees=rotation_configuration["maximum_degrees"],
                                                          p=probabilities["rotation"]),
                       kornia_augmentation.RandomPerspective(
                           distortion_scale=view_augmentation_configuration["perspective"]["distortion_scale"],
                           p=probabilities["perspective"]),
                       kornia_augmentation.RandomJigsaw(grid=tuple(view_augmentation_configuration["jigsaw"]["grid"]),
                                                        p=probabilities["jigsaw"])]

    # Photometric transformations, which keep every structure of the image
    transformations += [kornia_augmentation.ColorJitter(
                            brightness=colour_jitter_configuration["brightness"],
                            contrast=colour_jitter_configuration["contrast"],
                            saturation=colour_jitter_configuration["saturation"],
                            hue=colour_jitter_configuration["hue"], p=probabilities["colour_jitter"]),
                        kornia_augmentation.RandomGrayscale(p=probabilities["grayscale"]),
                        kornia_augmentation.RandomGamma(gamma=tuple(view_augmentation_configuration["gamma"]["gamma"]),
                                                        p=probabilities["gamma"]),
                        RandomHaze(strength=tuple(haze_configuration["strength"]),
                                   airlight=tuple(haze_configuration["airlight"]),
                                   probability=probabilities["haze"]),
                        kornia_augmentation.RandomLinearIllumination(
                            gain=tuple(view_augmentation_configuration["linear_illumination"]["gain"]),
                            p=probabilities["linear_illumination"]),
                        kornia_augmentation.RandomPlasmaShadow(
                            roughness=tuple(plasma_shadow_configuration["roughness"]),
                            shade_intensity=tuple(plasma_shadow_configuration["shade_intensity"]),
                            shade_quantity=tuple(plasma_shadow_configuration["shade_quantity"]),
                            p=probabilities["plasma_shadow"])]

    # Blurs, then the solarization, the noise after both blurs, and the black-out of grid cells and the erasing of a
    # rectangle last, so that the occluded areas stay plainly black. kornia draws the motion blur kernel size as
    # 2 * int(Uniform(low // 2, high // 2)) + 1, which never reaches the upper bound, so the upper bound is passed
    # raised by 2 to draw every odd size of [low, high], both bounds included
    motion_kernel_sizes = motion_blur_configuration["kernel_size"]
    transformations += [kornia_augmentation.RandomGaussianBlur(
                            kernel_size=(blur_kernel_size, blur_kernel_size), p=probabilities["gaussian_blur"],
                            sigma=tuple(view_augmentation_configuration["gaussian_blur"]["sigma"])),
                        kornia_augmentation.RandomMotionBlur(
                            kernel_size=(motion_kernel_sizes[0], motion_kernel_sizes[1] + 2),
                            angle=float(motion_blur_configuration["maximum_angle"]),
                            direction=tuple(motion_blur_configuration["direction"]), p=probabilities["motion_blur"]),
                        kornia_augmentation.RandomSolarize(thresholds=tuple(solarize_configuration["threshold"]),
                                                           additions=tuple(solarize_configuration["addition"]),
                                                           p=probabilities["solarize"]),
                        kornia_augmentation.RandomGaussianNoise(
                            mean=0.0, std=view_augmentation_configuration["gaussian_noise"]["standard_deviation"],
                            p=probabilities["gaussian_noise"]),
                        RandomBlackOut(probability=probabilities["black_out"],
                                       grid_size=view_augmentation_configuration["black_out"]["grid_size"]),
                        kornia_augmentation.RandomErasing(scale=tuple(erasing_configuration["scale"]),
                                                          ratio=tuple(erasing_configuration["aspect_ratio"]),
                                                          value=ERASING_FILL_VALUE, p=probabilities["random_erasing"])]
    return nn.Sequential(*transformations)


def describe_view_parameters(transformation_name: str, parameters: Dict[str, torch.Tensor], applied_index: int) -> str:
    """
    Writes the values a kornia transformation drew for one image of the last view, for the information panel of the
    input viewer. The parameters of kornia only hold the images the transformation was applied to, so the image is
    found at its rank among them.

    Args:
        transformation_name (str): The name of the transformation, from VIEW_TRANSFORMATION_NAMES.
        parameters (Dict[str, torch.Tensor]): The parameters kornia kept for the last batch, on the CPU.
        applied_index (int): The rank of the image among the images the transformation was applied to.

    Returns:
        str: The drawn values, empty for the transformations without random values (the flips, the grayscale and the
            Gaussian noise, whose values are its noise itself).
    """
    # Values drawn for the image, read from every parameter of the transformation
    image_values = {key: value[applied_index] for key, value in parameters.items()
                    if value.dim() > 0 and value.shape[0] > applied_index}
    if transformation_name == "random_resized_crop":
        # The corners are inclusive pixel indices, [x1, y1], [x2, y1], [x2, y2], [x1, y2], of the [H, W] input
        top_left_corner, _, bottom_right_corner, _ = image_values["src"].tolist()
        crop_width = round(bottom_right_corner[0] - top_left_corner[0] + 1)
        crop_height = round(bottom_right_corner[1] - top_left_corner[1] + 1)
        input_height, input_width = image_values["input_size"].tolist()
        area_fraction = crop_width * crop_height / (input_height * input_width)
        return (f"{crop_width}x{crop_height} pixels at x {round(top_left_corner[0])}, y {round(top_left_corner[1])} "
                f"({area_fraction * 100:.0f} % of the area)")
    if transformation_name == "rotation":
        return f"{float(image_values['degrees']):+.1f} degrees"
    if transformation_name == "perspective":
        # Largest move of the four corners of the image, in pixels
        corner_moves = (image_values["end_points"] - image_values["start_points"]).norm(dim=-1)
        return f"corners moved by up to {float(corner_moves.max()):.1f} pixels"
    if transformation_name == "jigsaw":
        return "pieces in the order " + " ".join(str(piece) for piece in image_values["permutation"].tolist())
    if transformation_name == "colour_jitter":
        # The hue factor is a fraction of the colour wheel, written in degrees
        return (f"brightness x{float(image_values['brightness_factor']):.2f}, "
                f"contrast x{float(image_values['contrast_factor']):.2f}, "
                f"saturation x{float(image_values['saturation_factor']):.2f}, "
                f"hue {float(image_values['hue_factor']) * 360.0:+.1f} degrees")
    if transformation_name == "gamma":
        return f"gamma {float(image_values['gamma_factor']):.2f}"
    if transformation_name == "haze":
        return (f"strength {float(image_values['strength']):.2f}, "
                f"airlight {float(image_values['airlight']):.2f}")
    if transformation_name == "linear_illumination":
        # Value of the gradient added to the image where its change is largest, signed
        gradient = image_values["gradient"].flatten()
        return f"gradient up to {float(gradient[gradient.abs().argmax()]):+.2f}"
    if transformation_name == "plasma_shadow":
        return (f"intensity {float(image_values['shade_intensity']):+.2f}, "
                f"quantity {float(image_values['shade_quantity']):.2f}, "
                f"roughness {float(image_values['roughness']):.2f}")
    if transformation_name == "gaussian_blur":
        return f"sigma {float(image_values['sigma']):.2f}"
    if transformation_name == "motion_blur":
        # kornia blurs the whole batch with the kernel size drawn for the image of rank 'idx', the angle and the
        # direction staying those of every image
        applied_kernel_size = int(parameters["ksize_factor"][int(parameters["idx"][0])])
        return (f"kernel {applied_kernel_size}, angle {float(image_values['angle_factor']):+.1f} "
                f"degrees, direction {float(image_values['direction_factor']):+.2f}")
    if transformation_name == "solarize":
        return (f"threshold {float(image_values['thresholds']):.2f}, "
                f"addition {float(image_values['additions']):+.2f}")
    if transformation_name == "black_out":
        return describe_black_out(hidden_cells=image_values["hidden_cells"])
    if transformation_name == "random_erasing":
        return (f"{round(float(image_values['widths']))}x{round(float(image_values['heights']))} pixels at x "
                f"{round(float(image_values['xs']))}, y {round(float(image_values['ys']))}")
    return ""


def describe_view_transformations(transformations: nn.Sequential, number_images: int) -> List[List[str]]:
    """
    Describes the transformations drawn for every image by the last call of a chain of view transformations, read from
    the parameters kornia keeps after a forward pass, for the information panels of the input viewer of the
    self-supervised task and of the view losses of the image classifier. It must be called right after the view is
    drawn, the next call of the chain overwriting the parameters.

    Args:
        transformations (nn.Sequential): A chain of build_view_transformations, after a forward pass.
        number_images (int): The number of images of the last batch.

    Returns:
        List[List[str]]: For every image, one line per applied transformation, its name and drawn values, in order.
    """
    descriptions = [[] for image_index in range(number_images)]
    for transformation_name, transformation in zip(VIEW_TRANSFORMATION_NAMES, transformations):
        # Copy the parameters to the CPU once as tensors, the noise and the illumination gradient holding a value per
        # pixel, the images the transformation applied to being flagged by batch_prob
        parameters = {key: value.detach().cpu() for key, value in transformation._params.items()
                      if torch.is_tensor(value)}
        applied_images = (parameters["batch_prob"] > 0.5).tolist()

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
    the forward pass of every training and validation batch: each image gets two views transformed independently, each
    by the chain of its view, so that the only thing the two views of an image share is its content. The image
    classifier draws the second view of every image with it, the second view of its view losses. The views are computed
    in float32 whatever the mixed precision of the forward pass, on float BGR images in [0, 255] like every batch of the
    framework.
    """

    def __init__(self, view_augmentation_configuration: Dict[str, Any], image_size: int) -> None:
        """
        Builds the chain of random transformations of each of the two views.

        Args:
            view_augmentation_configuration (Dict[str, Any]): The 'Data.view_augmentation' section.
            image_size (int): The side of the square network input.
        """
        self.view_transformations = [build_view_transformations(
                                         view_augmentation_configuration=view_augmentation_configuration,
                                         image_size=image_size, view_index=view_index)
                                     for view_index in range(NUMBER_VIEWS)]

    def draw_view(self, images: torch.Tensor, view_index: int) -> torch.Tensor:
        """
        Draws one random view of every image of a batch with the chain of the view, in float32 even within the bfloat16
        autocast of a mixed precision forward pass, so that the interpolations of the crops and rotations keep their
        precision: each of the two views of the self-supervised task (create_views), and the second view of every image
        of the view losses of the image classifier. The kornia colour transformations expect RGB images in [0, 1], so
        the BGR images are reversed and scaled before the transformations and brought back afterwards.

        Args:
            images (torch.Tensor): The [B, 3, H, W] float BGR images in [0, 255].
            view_index (int): The index of the drawn view, 0 for the first view and 1 for the second one.

        Returns:
            torch.Tensor: The [B, 3, H, W] float32 BGR views in [0, 255].
        """
        with torch.autocast(device_type=images.device.type, enabled=False):
            rgb_images = images.to(dtype=torch.float32).flip(dims=(1,)) / MAXIMUM_PIXEL_VALUE
            rgb_views = self.view_transformations[view_index](input=rgb_images).clamp(min=0.0, max=1.0)
            return rgb_views.flip(dims=(1,)) * MAXIMUM_PIXEL_VALUE

    def create_views(self, images: torch.Tensor, describe_views: bool = False) -> Tuple[
                     torch.Tensor, torch.Tensor, Optional[List[List[List[str]]]]]:
        """
        Draws the two views of every image of a batch, each by draw_view with the chain of its view, in float32 even
        within the bfloat16 autocast of a mixed precision forward pass. The input viewer also asks for the
        transformations drawn for every view, written in its information panels.

        Args:
            images (torch.Tensor): The [B, 3, H, W] float BGR images in [0, 255].
            describe_views (bool): Whether to describe the drawn transformations, which reads their parameters back
                from the device. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[List[List[List[str]]]]]: The first and second [B, 3, H, W] views
                of the images, and the description lines of every image for each of the two views (None unless
                describe_views).
        """
        # Every view is drawn by its own chain and described right after it is drawn
        view_descriptions = [] if describe_views else None
        views = []
        for view_index in range(NUMBER_VIEWS):
            views.append(self.draw_view(images=images, view_index=view_index))
            if describe_views:
                view_descriptions.append(describe_view_transformations(
                    transformations=self.view_transformations[view_index], number_images=images.shape[0]))
        return views[0], views[1], view_descriptions
