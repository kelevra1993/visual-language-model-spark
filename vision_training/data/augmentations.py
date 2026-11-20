"""
DataAugmenter, the random geometric and photometric augmentations of every task, applied on the device right before
the forward pass: the training batches of the supervised tasks, moving the bounding boxes of the object detector, the
masks of the semantic segmentation, the instance masks of the instance segmentation and the keypoints and ignored
regions of the keypoint detection with the images and handling the frame shuffling and per-frame draws of the sequence
tasks, and the two random views of every training and validation image of the self-supervised task (create_views),
whose loss pulls the two views of an image together. The strength of every augmentation type is hard-coded in two
sets, SUPERVISED_AUGMENTATION_STRENGTHS and the stronger SELF_SUPERVISED_AUGMENTATION_STRENGTHS of the views, chosen
by the task, the configuration only turning the types on and setting their probabilities. Of the histogram
equalizations drawn together for an image (EXCLUSIVE_EQUALIZATION_TYPES), only one, chosen at random, is applied, the
geometric types of IMAGE_ONLY_AUGMENTATION_TYPES move the image only, and the types of SECOND_VIEW_ONLY_TYPES are drawn
for the second view only of the self-supervised task. The line kernel of the motion blur, the cells of the black-out,
the sampling grid of the perspective, the fractal map of the plasma shadow and the kernels of the sharpening are
computed on the device from numbers only (build_motion_blur_kernel, send_grid_to_device, compute_cell_indices,
compute_perspective_coefficients, build_plasma_map, get_sharpening_kernels), as are the rotation, the CLAHE and the
YUV conversions ported from torchvision and kornia (vision_training/data/device_image_operations.py), so that the
host never waits for the device at every image.
Built by the base trainer from the Data.data_augmentation configuration, whose augmentation types are verified by the
common checks. Every augmented sample comes with a report of the types drawn for it and their values, which the
'view_input_images' diagnosis mode writes in the information panel next to every training image or view.
"""

import math
import random
import warnings
import torch

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
from torch.nn import functional as torch_functional
from torchvision import tv_tensors
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as vision_functional

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.data.detection.coco_masks import compute_mask_bounding_boxes
from vision_training.data.device_image_operations import (convert_colour_space, equalize_clahe_on_device,
                                                          get_yuv_matrices, rotate_on_device)

# kornia decorates some of its functions with torch.jit.script when it is imported, which recent PyTorch versions
# report as deprecated; the notice is harmless and cannot be fixed on our side, so it is silenced for this import only
with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", message=r"`torch\.jit\.script` is deprecated", category=FutureWarning)
    from kornia.enhance import adjust_hue
    from kornia.filters import filter2d

# Geometric augmentations, applied first, in this order. They move the bounding boxes and the masks along with the
# image, except the types of IMAGE_ONLY_AUGMENTATION_TYPES. random_aspect_crop, the crop changing the aspect ratio of
# the content, follows random_crop, random_zoom_out follows both, so that a sample drawing a crop and the zoom-out is
# cropped first and then shrunk, its draws being independent, random_translation then shifts the result, and the
# perspective and the jigsaw come last, as in the views of the self-supervised methods
GEOMETRIC_AUGMENTATION_TYPES = ["random_crop", "random_aspect_crop", "random_zoom_out", "random_translation",
                                "random_left_right_flip", "random_up_down_flip", "random_rotation",
                                "random_90_rotation", "random_perspective", "random_jigsaw"]

# Photometric augmentations, applied after the geometric ones, in this order. They never move the bounding boxes and
# never change the masks. random_hue follows the brightness, contrast and saturation, the gamma, the linear
# illumination and the plasma shadow come before the blurs, random_motion_blur follows random_blur, the two blurs
# smoothing the noise drawn before them, random_haze comes after the others, so that the equalizations and the
# sharpening drawn with it do not restore the contrast it removes, the solarization follows it, and random_black_out
# and random_erasing come last, so that their cells and rectangles stay black
PHOTOMETRIC_AUGMENTATION_TYPES = ["brightness_contrast_saturation", "random_hue", "clahe_augment", "sharp_augment",
                                  "equalize_histogram_augment", "equalize_bgr_histogram_augment", "random_noise",
                                  "gray_scale_augment", "random_colour_augmentation", "random_gamma",
                                  "random_linear_illumination", "random_plasma_shadow", "random_blur",
                                  "random_motion_blur", "random_haze", "random_solarize", "random_black_out",
                                  "random_erasing"]

AUGMENTATION_TYPES = GEOMETRIC_AUGMENTATION_TYPES + PHOTOMETRIC_AUGMENTATION_TYPES

# Histogram equalizations of which an image receives at most one, chosen at random among those drawn: an
# equalization of the whole image after CLAHE stretches the one or two level steps CLAHE leaves between its tiles
# on a flat background into visible blocks
EXCLUSIVE_EQUALIZATION_TYPES = ["clahe_augment", "equalize_histogram_augment", "equalize_bgr_histogram_augment"]

# Geometric types moving the image only, which no box, mask or keypoint can follow: the checks of the object
# detection, the instance segmentation, the semantic segmentation and the keypoint detection refuse them
IMAGE_ONLY_AUGMENTATION_TYPES = ["random_perspective", "random_jigsaw"]

# Types drawn for the second view only of the self-supervised task, as the asymmetric views of BYOL and Barlow Twins,
# so that the two views of an image never share them; the supervised tasks draw them for every image
SECOND_VIEW_ONLY_TYPES = ["random_solarize", "random_black_out"]

# Number of views drawn for every image of the self-supervised task
NUMBER_VIEWS = 2

# Resampling of the masks resized by the crops and the zoom-out: the nearest pixel of every output pixel centre, aligned
# with the bilinear images, the plain nearest of torchvision shifting the masks by half a pixel to the bottom right
MASK_INTERPOLATION = InterpolationMode.NEAREST_EXACT


class AugmentationStrengths(TypedDict):
    """
    Strengths of the augmentation types, hard-coded rather than configured so that a configuration file only turns the
    types on and sets their probabilities: SUPERVISED_AUGMENTATION_STRENGTHS for the supervised tasks, and
    SELF_SUPERVISED_AUGMENTATION_STRENGTHS, the stronger views of the self-supervised task, DataAugmenter taking the set
    of its task. The intensities are in levels of the [0, 255] images, the fractions of a side or of the area within
    [0, 1], the angles in degrees, and every range [low, high] is drawn uniformly, a range whose bounds are equal giving
    its value without any draw. random_crop keeps the aspect ratio of the image and a share of each side within
    crop_side_range, or, given crop_area_range, a share of the area within it and an aspect ratio drawn within
    crop_aspect_ratio_range (the random resized crop of SimCLR); brightness_contrast_saturation shifts the brightness by
    up to brightness_maximum_delta levels, or, given brightness_factor_range, scales it by a factor drawn within it (the
    colour jitter of SimCLR). The unused setting of each pair is None.
    """
    crop_side_range: Optional[Tuple[float, float]]
    crop_area_range: Optional[Tuple[float, float]]
    crop_aspect_ratio_range: Optional[Tuple[float, float]]
    aspect_crop_side_range: Tuple[float, float]
    zoom_out_side_range: Tuple[float, float]
    maximum_translation_fraction: float
    maximum_rotation_degrees: float
    perspective_distortion_scale: float
    jigsaw_grid_size: int
    brightness_maximum_delta: Optional[float]
    brightness_factor_range: Optional[Tuple[float, float]]
    contrast_factor_range: Tuple[float, float]
    saturation_factor_range: Tuple[float, float]
    maximum_hue_shift: float
    clahe_clip_limit: float
    clahe_grid_size: Tuple[int, int]
    sharpening_sigma: float
    sharpening_strength: float
    noise_standard_deviation: float
    gamma_range: Tuple[float, float]
    linear_illumination_gain_range: Tuple[float, float]
    plasma_shadow_intensity_range: Tuple[float, float]
    plasma_shadow_quantity_range: Tuple[float, float]
    plasma_shadow_roughness_range: Tuple[float, float]
    blur_kernel_size: int
    blur_sigma_range: Tuple[float, float]
    motion_blur_kernel_size_range: Tuple[int, int]
    motion_blur_angle_range: Tuple[float, float]
    motion_blur_direction_range: Tuple[float, float]
    haze_strength_range: Tuple[float, float]
    haze_airlight_range: Tuple[float, float]
    solarize_threshold: float
    black_out_cell_probability: float
    erasing_area_range: Tuple[float, float]
    erasing_aspect_ratio_range: Tuple[float, float]


# Strengths of the supervised tasks, slight changes keeping every object recognisable: the crop keeps 75 % to 100 % of
# each side, the brightness moves by up to 40 levels, the contrast and saturation only decrease and the noise has a
# standard deviation of 20 levels. The types added with the self-supervised views (perspective, jigsaw, hue, gamma,
# linear illumination, plasma shadow, solarization, erasing) take mild values of their own
SUPERVISED_AUGMENTATION_STRENGTHS: AugmentationStrengths = {
    "crop_side_range": (0.75, 1.0), "crop_area_range": None, "crop_aspect_ratio_range": None,
    "aspect_crop_side_range": (0.75, 1.0), "zoom_out_side_range": (0.5, 1.0), "maximum_translation_fraction": 0.25,
    "maximum_rotation_degrees": 5.0, "perspective_distortion_scale": 0.05, "jigsaw_grid_size": 2,
    "brightness_maximum_delta": 40.0, "brightness_factor_range": None, "contrast_factor_range": (0.7, 1.0),
    "saturation_factor_range": (0.7, 1.0), "maximum_hue_shift": 0.05,
    "clahe_clip_limit": 2.0, "clahe_grid_size": (4, 4), "sharpening_sigma": 1.5, "sharpening_strength": 3.0,
    "noise_standard_deviation": 20.0, "gamma_range": (0.8, 1.25),
    "linear_illumination_gain_range": (0.05 * 255.0, 0.15 * 255.0),
    "plasma_shadow_intensity_range": (-0.3 * 255.0, 0.0), "plasma_shadow_quantity_range": (0.0, 0.5),
    "plasma_shadow_roughness_range": (0.1, 0.7),
    "blur_kernel_size": 9, "blur_sigma_range": (2.5, 2.5), "motion_blur_kernel_size_range": (5, 15),
    "motion_blur_angle_range": (0.0, 180.0), "motion_blur_direction_range": (-1.0, 1.0),
    "haze_strength_range": (0.2, 0.6), "haze_airlight_range": (180.0, 255.0), "solarize_threshold": 127.5,
    "black_out_cell_probability": 0.2, "erasing_area_range": (0.02, 0.15), "erasing_aspect_ratio_range": (0.3, 3.3)}

# Strengths of the self-supervised task, the views of SimCLR, BYOL and Barlow Twins, lowered from those of SimCLR on
# ImageNet, which were too strong for the images of the project: the crop covers 20 % to 100 % of the area with an
# aspect ratio of x1/2 to x2, the colour jitter scales the brightness, contrast and saturation by 0.6 to 1.2 and shifts
# the hue by up to a tenth of the hue circle, the translation reaches 25 %, the perspective 5 %, the rotation 10
# degrees, the noise 30 levels, the haze 70 % with an airlight of 160 to 255 levels and the black-out 35 % of the
# cells. The plasma shadows darken by up to 51 levels on up to 30 % of the image and the solarization inverts the
# pixels at or above 191.25 levels only, milder than in the supervised tasks. The other types keep the supervised values
SELF_SUPERVISED_AUGMENTATION_STRENGTHS: AugmentationStrengths = {
    "crop_side_range": None, "crop_area_range": (0.2, 1.0), "crop_aspect_ratio_range": (1.0 / 2.0, 2.0),
    "aspect_crop_side_range": (0.75, 1.0), "zoom_out_side_range": (0.5, 1.0), "maximum_translation_fraction": 0.25,
    "maximum_rotation_degrees": 10.0, "perspective_distortion_scale": 0.05, "jigsaw_grid_size": 2,
    "brightness_maximum_delta": None, "brightness_factor_range": (0.6, 1.2), "contrast_factor_range": (0.6, 1.2),
    "saturation_factor_range": (0.6, 1.2), "maximum_hue_shift": 0.1,
    "clahe_clip_limit": 2.0, "clahe_grid_size": (4, 4), "sharpening_sigma": 1.5, "sharpening_strength": 3.0,
    "noise_standard_deviation": 30.0, "gamma_range": (0.8, 1.25),
    "linear_illumination_gain_range": (0.05 * 255.0, 0.15 * 255.0),
    "plasma_shadow_intensity_range": (-0.2 * 255.0, 0.0), "plasma_shadow_quantity_range": (0.0, 0.3),
    "plasma_shadow_roughness_range": (0.1, 0.7),
    "blur_kernel_size": 9, "blur_sigma_range": (2.5, 2.5), "motion_blur_kernel_size_range": (5, 15),
    "motion_blur_angle_range": (0.0, 180.0), "motion_blur_direction_range": (-1.0, 1.0),
    "haze_strength_range": (0.2, 0.7), "haze_airlight_range": (160.0, 255.0), "solarize_threshold": 0.75 * 255.0,
    "black_out_cell_probability": 0.35, "erasing_area_range": (0.02, 0.15), "erasing_aspect_ratio_range": (0.3, 3.3)}

# Draws of the region of a crop of random area and aspect ratio before the central region is taken, as the random
# resized crop of torchvision, and of the rectangle of random_erasing before the image is left unchanged
MAXIMUM_REGION_ATTEMPTS = 10

# Sides of the image towards which the ramp of random_linear_illumination grows, one drawn with equal chances
LINEAR_ILLUMINATION_DIRECTIONS = ["bottom", "top", "right", "left"]

# Added to the range of the fractal map of random_plasma_shadow before its min-max normalization, as kornia does
PLASMA_NORMALIZATION_EPSILON = 1e-6

# Axes a translation shifts the image along, one of them drawn with equal chances: the width only, the height only, or
# both at once, each shift being drawn independently up to maximum_translation_fraction of its side
TRANSLATION_AXES = ["x", "y", "xy"]

# Boxes narrower or shorter than this many pixels after a geometric augmentation are removed with their label
MINIMUM_BOX_SIDE = 1.0

# Objects keeping less than this share of their area inside the image after a geometric augmentation (box area, or
# mask pixels for the instance segmentation) are removed with their label, so that the detector is never trained to
# find a fragment cut by the border of a crop or a rotation
MINIMUM_VISIBLE_AREA_FRACTION = 0.25

# Channels of the BGR images delivered by the data loaders, as slices so that taking them never makes the host wait
# for the device, which indexing with a list of channels does
BGR_CHANNEL_INDICES = {"B": slice(0, 1), "G": slice(1, 2), "R": slice(2, 3), "BGR": slice(0, 3)}


class AugmentationReport(TypedDict):
    """
    Record of the augmentations drawn for one training sample by DataAugmenter.augment_batch, kept in the batch by the
    trainers and written by describe_report in the information panel of the input viewer ('view_input_images'), so
    that the user sees which augmentations every training image received and how strong they were. 'augmented' is
    False when the global probability of the sample was not drawn (or the augmentation is off). Every applied type is a
    (type name, drawn values) pair, the values being empty for the types without random values (e.g. a flip). The
    photometric types hold one list for the sample, or one per frame when the frames of a sequence are drawn on their
    own ('independent_frames'). 'removed_labels' gives the labels of the objects removed after the geometric
    augmentations, None for the tasks without boxes.
    """
    augmented: bool
    geometric_augmentations: List[Tuple[str, str]]
    photometric_augmentations: List[List[Tuple[str, str]]]
    removed_labels: Optional[torch.Tensor]


def create_empty_report() -> AugmentationReport:
    """
    Creates the report of a sample that received no augmentation, filled by DataAugmenter.augment_sample when the
    sample is augmented.

    Returns:
        AugmentationReport: The report without any applied type.
    """
    return {"augmented": False, "geometric_augmentations": [], "photometric_augmentations": [],
            "removed_labels": None}


def apply_haze(images: torch.Tensor, strength: Union[float, torch.Tensor],
               airlight: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Hazes images, the fog, condensation or dirt on the lens of a camera: every pixel is blended with a uniform
    airlight, images * (1 - strength) + strength * airlight, which lowers the contrast and lifts the dark areas towards
    the airlight. Used by the random_haze type of DataAugmenter on the BGR images in [0, 255], a grey airlight being the
    same in every channel order.

    Args:
        images (torch.Tensor): The images or sequences to haze, of any shape.
        strength (Union[float, torch.Tensor]): The share of the airlight in every pixel, within [0, 1], a number or a
            tensor broadcast over the images.
        airlight (Union[float, torch.Tensor]): The intensity of the airlight, in the range of the images, a number or
            a tensor broadcast over the images.

    Returns:
        torch.Tensor: The hazy images, of the shape of the input.
    """
    return images * (1.0 - strength) + strength * airlight


def apply_black_out(images: torch.Tensor, hidden_cells: torch.Tensor) -> torch.Tensor:
    """
    Blacks out the hidden cells of a grid laid over images, an occlusion that makes a network learn from every region
    of an image instead of the most telling one, for the random_black_out type of DataAugmenter, one grid for an image,
    a view or a sequence. The grid drawn on the host reaches the device without a copy (send_grid_to_device) and the
    cell of every pixel is computed there (compute_cell_indices), so that the host never waits for the device.

    Args:
        images (torch.Tensor): The images [..., H, W] to black out.
        hidden_cells (torch.Tensor): The boolean [..., rows, columns] grid of the hidden cells, broadcast over the
            leading dimensions of the images: a single [rows, columns] grid, or a [B, 1, rows, columns] grid per image
            of a [B, C, H, W] batch.

    Returns:
        torch.Tensor: The images with their hidden cells at 0, of the shape of the input.
    """
    image_height, image_width = images.shape[-2], images.shape[-1]
    number_rows, number_columns = hidden_cells.shape[-2], hidden_cells.shape[-1]

    # Grid drawn on the host given to the device of the images as numbers, so that the host never waits for a copy
    if hidden_cells.device != images.device:
        hidden_cells = send_grid_to_device(hidden_cells=hidden_cells, device=images.device)

    # Cell of every row and column of pixels, then the kept cells spread over the pixels and the hidden ones zeroed
    row_cells = compute_cell_indices(number_pixels=image_height, number_cells=number_rows, device=images.device)
    column_cells = compute_cell_indices(number_pixels=image_width, number_cells=number_columns, device=images.device)
    kept_pixels = (~hidden_cells).index_select(dim=-2, index=row_cells).index_select(dim=-1, index=column_cells)
    return images * kept_pixels.to(dtype=images.dtype)


def send_grid_to_device(hidden_cells: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Gives the boolean grid of the hidden cells drawn on the host to the device of the images for apply_black_out
    without any copy: its cells are packed into 64-bit integers on the host, each written on the device by a fill whose
    value travels with the launch of the kernel, then unpacked there. A copy from the host would make the host wait
    for the device at every image (a pageable copy), or allocate new pinned memory while the earlier copies are still
    queued (a pinned copy), both many times slower than the black-out itself.

    Args:
        hidden_cells (torch.Tensor): The boolean [..., rows, columns] grid of the hidden cells, on the host.
        device (torch.device): The device of the images.

    Returns:
        torch.Tensor: The same grid on the device.
    """
    # Cells packed by 64 into integers on the host, the last integer padded with kept cells, the bit 63 giving the
    # sign of an int64
    number_cells = hidden_cells.numel()
    padded_cells = torch.zeros(size=(math.ceil(number_cells / 64) * 64,), dtype=torch.int64)
    padded_cells[:number_cells] = hidden_cells.flatten().to(dtype=torch.int64)
    packed_words = (padded_cells.view(-1, 64) << torch.arange(end=64, dtype=torch.int64)).sum(dim=1).tolist()

    # Every integer written on the device by a fill, then its bits read back into the cells of the grid
    device_words = torch.cat([torch.full(size=(1,), fill_value=packed_word, dtype=torch.int64, device=device)
                              for packed_word in packed_words])
    device_bits = (device_words.unsqueeze(dim=-1) >> get_bit_offsets(device=device)) & 1
    return device_bits.flatten()[:number_cells].view(hidden_cells.shape).to(dtype=torch.bool)


@lru_cache(maxsize=8)
def get_bit_offsets(device: torch.device) -> torch.Tensor:
    """
    Gives the offsets 0 to 63 of the bits of an int64 on a device, which send_grid_to_device shifts the packed
    integers by to read their bits, created once per device.

    Args:
        device (torch.device): The device of the images.

    Returns:
        torch.Tensor: The int64 [64] offsets.
    """
    return torch.arange(end=64, dtype=torch.int64, device=device)


@lru_cache(maxsize=64)
def compute_cell_indices(number_pixels: int, number_cells: int, device: torch.device) -> torch.Tensor:
    """
    Gives the cell of every row (or column) of pixels of an image cut into a grid, for apply_black_out, computed on the
    device of the images from numbers only, so that the host never waits for a copy to the device, and kept for the
    next images of the same size and grid, which never change it. The cell edges are index x number_pixels /
    number_cells rounded half to even, as Python's round, computed with integers so that they are exact: a side that
    is not a multiple of the grid size gives cells differing by one pixel.

    Args:
        number_pixels (int): The height (or width) of the images in pixels.
        number_cells (int): The number of rows (or columns) of the grid.
        device (torch.device): The device of the images.

    Returns:
        torch.Tensor: The int64 [number_pixels] cell index of every row (or column) of pixels.
    """
    # Inner edges of the cells, index x number_pixels / number_cells rounded half to even on the integers
    numerators = torch.arange(start=1, end=number_cells, device=device, dtype=torch.int64) * number_pixels
    quotients = torch.div(input=numerators, other=number_cells, rounding_mode="floor")
    doubled_remainders = 2 * (numerators - quotients * number_cells)
    rounded_up = (doubled_remainders > number_cells) | ((doubled_remainders == number_cells) & (quotients % 2 == 1))
    inner_edges = quotients + rounded_up.to(dtype=torch.int64)

    # Cell of every pixel: the number of inner edges at or before it
    pixel_positions = torch.arange(end=number_pixels, device=device, dtype=torch.int64)
    return torch.bucketize(input=pixel_positions, boundaries=inner_edges, right=True)


def describe_black_out(hidden_cells: torch.Tensor) -> str:
    """
    Writes the black-out drawn for one image, the number of hidden cells out of the cells of its grid, for the report
    of the random_black_out type of DataAugmenter, written in the information panels of the input viewer
    ('view_input_images').

    Args:
        hidden_cells (torch.Tensor): The boolean [rows, columns] grid of the hidden cells of the image.

    Returns:
        str: The description, e.g. '12 of the 64 cells of the 8x8 grid'.
    """
    number_rows, number_columns = hidden_cells.shape[-2], hidden_cells.shape[-1]
    return (f"{int(hidden_cells.sum())} of the {number_rows * number_columns} cells of the {number_rows}x"
            f"{number_columns} grid")


def build_motion_blur_kernel(kernel_size: int, angle: float, direction: float, device: torch.device) -> torch.Tensor:
    """
    Draws the line kernel of the random_motion_blur type of DataAugmenter on the device of the images, from numbers
    only, the kernel of kornia's get_motion_kernel2d (equal within float rounding): a ramp of weights along a line
    through the center of the kernel, rotated counter-clockwise on the displayed image with a nearest sampling, then
    normalized. kornia builds it on the CPU and copies it to the device, which makes the host wait for the device
    at every image, several milliseconds when other trainings share the GPU.

    Args:
        kernel_size (int): The odd side of the kernel, the length of the line in pixels.
        angle (float): The angle of the line in degrees, counter-clockwise from the horizontal on the displayed
            image.
        direction (float): The lean of the blur along the line, within [-1, 1]: the weights ramp linearly along
            the line, so that -1 trails the streak backward (to the left at an angle of 0), 1 forward, and 0
            spreads it evenly.
        device (torch.device): The device of the images to blur.

    Returns:
        torch.Tensor: The [1, kernel_size, kernel_size] kernel, its weights summing to 1.
    """
    # Coordinates of every kernel pixel from the center, the columns to the right and the rows downward
    radius = kernel_size // 2
    offsets = torch.arange(start=-radius, end=radius + 1, device=device, dtype=torch.float32)
    rows, columns = torch.meshgrid(offsets, offsets, indexing="ij")

    # Rotate every pixel back onto the horizontal line, rounded to the nearest pixel like the rotation of kornia
    cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    along_line = torch.round(input=columns * cosine - rows * sine)
    across_line = torch.round(input=columns * sine + rows * cosine)

    # Ramp of weights from the backward end of the line to its forward end, kept on the line and normalized
    backward_weight = (direction + 1.0) / 2.0
    weights = backward_weight + (1.0 - 2.0 * backward_weight) * (along_line + radius) / (kernel_size - 1)
    kernel = torch.where(condition=(across_line == 0.0) & (along_line.abs() <= radius), input=weights, other=0.0)
    return (kernel / kernel.sum()).unsqueeze(dim=0)


def draw_uniform(value_range: Tuple[float, float]) -> float:
    """
    Draws a value of a strength range [low, high] of AugmentationStrengths with Python's random module, the generator
    of every draw of DataAugmenter, or gives its value without any draw when both bounds are equal, so that a fixed
    strength (the sigma of the blur of the supervised tasks) leaves the random sequence of the other draws unchanged.

    Args:
        value_range (Tuple[float, float]): The [low, high] range of the strength.

    Returns:
        float: The drawn value.
    """
    low, high = value_range
    return low if low == high else random.uniform(low, high)


def compute_perspective_coefficients(output_points: List[Tuple[float, float]],
                                     input_points: List[Tuple[float, float]]) -> List[float]:
    """
    Solves on the host, from numbers only, the eight coefficients [a, b, c, d, e, f, g, h] of the perspective
    transformation of random_perspective taking every pixel (x, y) of the output to its position in the input,
    ((a x + b y + c) / (g x + h y + 1), (d x + e y + f) / (g x + h y + 1)), from four point pairs: the corners moved
    inwards in the output and the corners of the input. The sampling grid is then computed on the device from these
    numbers, so that the host never waits for a copy to the device, which torchvision's perspective makes at every
    image.

    Args:
        output_points (List[Tuple[float, float]]): The four (x, y) points of the output, in pixels.
        input_points (List[Tuple[float, float]]): The four (x, y) points of the input they show, in pixels.

    Returns:
        List[float]: The coefficients a, b, c, d, e, f, g, h.
    """
    # Two linear equations per point pair, the denominator of the transformation multiplied out
    equations, targets = [], []
    for (output_x, output_y), (input_x, input_y) in zip(output_points, input_points):
        equations.append([output_x, output_y, 1.0, 0.0, 0.0, 0.0, -output_x * input_x, -output_y * input_x])
        equations.append([0.0, 0.0, 0.0, output_x, output_y, 1.0, -output_x * input_y, -output_y * input_y])
        targets += [input_x, input_y]

    # The eight equations solved in double precision on the CPU
    coefficients = torch.linalg.solve(torch.tensor(data=equations, dtype=torch.float64),
                                      torch.tensor(data=targets, dtype=torch.float64))
    return coefficients.tolist()


@lru_cache(maxsize=16)
def get_pixel_coordinates(height: int, width: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gives the column and the row of every pixel of an image on a device, the positions random_perspective takes back
    into the input, created from numbers once per image size and device and only read afterwards.

    Args:
        height (int): The height of the images in pixels.
        width (int): The width of the images in pixels.
        device (torch.device): The device of the images.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The float32 [H, W] columns and rows of the pixels.
    """
    rows, columns = torch.meshgrid(torch.arange(end=height, device=device, dtype=torch.float32),
                                   torch.arange(end=width, device=device, dtype=torch.float32), indexing="ij")
    return columns, rows


@lru_cache(maxsize=8)
def get_plasma_kernels(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gives the [1, 3, 3] averaging kernels of the diamond and the square steps of build_plasma_map on a device, filled
    from numbers once per device: the four diagonal neighbours of a pixel, then its four direct neighbours, each
    weighted 0.25. kornia copies them from the host at every step of its diamond_square, which makes the host wait
    for the device.

    Args:
        device (torch.device): The device of the images.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The kernels of the diamond and of the square steps.
    """
    diamond_kernel = torch.zeros(size=(1, 3, 3), device=device)
    diamond_kernel[:, 0::2, 0::2] = 0.25
    square_kernel = torch.zeros(size=(1, 3, 3), device=device)
    square_kernel[:, 0::2, 1] = 0.25
    square_kernel[:, 1, 0::2] = 0.25
    return diamond_kernel, square_kernel


@lru_cache(maxsize=16)
def get_sharpening_kernels(dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gives the kernels of the sharp_augment type on a device, filled from numbers once per dtype and device, so that
    sharpening an image never makes the host wait for a copy to the device: the [3, 1, 1] luminance weights of the BGR
    channels and the [1, 1, 3, 3] Laplacian kernel.

    Args:
        dtype (torch.dtype): The floating dtype of the images.
        device (torch.device): The device of the images.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The luminance weights and the Laplacian kernel.
    """
    luminance_weights = torch.empty(size=(3, 1, 1), dtype=dtype, device=device)
    for channel_index, weight in enumerate([0.114, 0.587, 0.299]):
        luminance_weights[channel_index].fill_(weight)
    laplacian_kernel = torch.zeros(size=(1, 1, 3, 3), dtype=dtype, device=device)
    laplacian_kernel[..., 0::2, 1] = 1.0
    laplacian_kernel[..., 1, 0::2] = 1.0
    laplacian_kernel[..., 1, 1] = -4.0
    return luminance_weights, laplacian_kernel


def build_plasma_seed(rows: int, columns: int, device: torch.device) -> torch.Tensor:
    """
    Draws the coarse seed grid of build_plasma_map on the device, three rows (or three columns) of random values
    interpolated by one diamond and one square step, the seed of kornia's diamond_square for one image.

    Args:
        rows (int): The number of rows of the seed, odd, 3 or the columns being 3.
        columns (int): The number of columns of the seed, odd, 3 or the rows being 3.
        device (torch.device): The device of the images.

    Returns:
        torch.Tensor: The [1, 1, rows, columns] seed.
    """
    # The seed is drawn with three rows, a seed of three columns being drawn transposed
    transposed = columns == 3
    if transposed:
        rows, columns = columns, rows
    seed = torch.rand(size=(1, 1, rows, columns), device=device)
    seed[..., ::2, ::2] = torch.rand(size=(1, 1, 2, (columns + 1) // 2), device=device)

    # Diamond step: the middle row between the corners of every square of the first and last rows
    seed[..., 1, 1::2] = (seed[..., ::2, :-2:2] + seed[..., ::2, 2::2]).sum(dim=2) / 4.0

    # Square step: the borders from their neighbours, then the whole seed rescaled
    first_column_bottom = seed[..., 2, 0]
    seed[..., 1, 0] = seed[..., 0, 0] + seed[..., 1, 1] + first_column_bottom
    seed[..., 1, -1] = seed[..., -1, -1] + seed[..., 1, -2] + first_column_bottom
    diamond_centres = seed[..., 1, 1::2]
    seed[..., 0, 1::2] = seed[..., 0, 0:-2:2] + seed[..., 0, 2::2] + diamond_centres
    seed[..., 2, 1::2] = seed[..., 2, 0:-2:2] + seed[..., 2, 2::2] + diamond_centres
    seed = seed / 3.0
    return seed.transpose(2, 3) if transposed else seed


def refine_plasma_map(plasma_map: torch.Tensor, random_scale: float) -> torch.Tensor:
    """
    Doubles the resolution of the fractal map of build_plasma_map, one diamond and one square step of kornia's
    diamond_square: every new pixel takes the average of its neighbours plus a random value scaled by random_scale,
    which the roughness lowers at every step, the kernels being kept on the device (get_plasma_kernels).

    Args:
        plasma_map (torch.Tensor): The [1, 1, H, W] map, H and W odd.
        random_scale (float): The scale of the random values added at this step.

    Returns:
        torch.Tensor: The [1, 1, 2 H - 1, 2 W - 1] map.
    """
    diamond_kernel, square_kernel = get_plasma_kernels(device=plasma_map.device)
    map_height, map_width = plasma_map.shape[-2], plasma_map.shape[-1]

    # Known pixels spread on every second row and column of the finer map, the border pixels missing a quarter of
    # their neighbours being compensated
    refined_map = torch.zeros(size=(1, 1, 2 * (map_height - 1) + 1, 2 * (map_width - 1) + 1), device=plasma_map.device)
    refined_map[:, :, ::2, ::2] = plasma_map
    border_compensation = torch.ones_like(input=refined_map)
    border_compensation[:, :, :, 0] = border_compensation[:, :, :, -1] = 1.0 / 0.75
    border_compensation[:, :, 0, :] = border_compensation[:, :, -1, :] = 1.0 / 0.75
    random_values = torch.rand(size=refined_map.shape, device=plasma_map.device) * random_scale

    # Diamond step: the centres of the squares of known pixels from their four corners
    diamond_regions = filter2d(input=refined_map, kernel=diamond_kernel)
    diamond_centres = (diamond_regions > 0).to(dtype=refined_map.dtype)
    refined_map = (refined_map + (1.0 - random_scale) * diamond_regions * diamond_centres +
                   diamond_centres * random_values)

    # Square step: the remaining pixels from their four direct neighbours
    square_regions = filter2d(input=refined_map, kernel=square_kernel) * border_compensation
    square_centres = (square_regions > 0).to(dtype=refined_map.dtype)
    return refined_map + square_centres * random_values + (1.0 - random_scale) * square_centres * square_regions


def build_plasma_map(height: int, width: int, roughness: float, device: torch.device) -> torch.Tensor:
    """
    Draws the random fractal map of random_plasma_shadow on the device, a port of kornia's diamond_square for one
    image (the map of kornia's RandomPlasmaShadow) whose kernels and scales never leave the device: a coarse seed
    refined step by step up to the image size, every step adding random values scaled down by the roughness, so that a
    higher roughness gives a map with more fine detail, then normalized to [0, 1].

    Args:
        height (int): The height of the image in pixels, at least 3.
        width (int): The width of the image in pixels, at least 3.
        roughness (float): The scale applied to the random values at every refinement, within [0, 1].
        device (torch.device): The device of the image.

    Returns:
        torch.Tensor: The [H, W] map in [0, 1].

    Raises:
        ValueError: If a side of the image is below 3 pixels, too small for a diamond-square map.
    """
    if height < 3 or width < 3:
        raise ValueError(f"random_plasma_shadow needs images of at least 3x3 pixels, got {height}x{width}.")

    # Sides of the map, one more than the next power of two, the number of refinements and the sides of the seed
    padded_height = 2 ** math.ceil(math.log2(height - 1)) + 1
    padded_width = 2 ** math.ceil(math.log2(width - 1)) + 1
    number_refinements = int(min(math.log2(padded_height - 1) - 1, math.log2(padded_width - 1) - 1))
    plasma_map = build_plasma_seed(rows=(padded_height - 1) // 2 ** number_refinements + 1,
                                   columns=(padded_width - 1) // 2 ** number_refinements + 1, device=device)

    # Refinements with random values scaled down by the roughness at every step, then the image size kept
    random_scale = 1.0
    for refinement_index in range(number_refinements):
        random_scale *= roughness
        plasma_map = refine_plasma_map(plasma_map=plasma_map, random_scale=random_scale)
    plasma_map = plasma_map[0, 0, :height, :width]

    # Values spread over [0, 1] by their minimum and maximum, computed on the device
    minimum_value, maximum_value = plasma_map.min(), plasma_map.max()
    return (plasma_map - minimum_value) / (maximum_value - minimum_value + PLASMA_NORMALIZATION_EPSILON)


class DataAugmenter:
    """
    Randomly augments batches on their device (usually the GPU), right before the forward pass: the training batches of
    the supervised tasks (augment_batch), and the two views of every training and validation image of the
    self-supervised task (create_views).

    It is shared by every network of the project: the object detector passes bounding boxes and labels, which are
    moved, clipped and filtered together with the geometric augmentations, the semantic segmenter passes the class
    masks of its images, moved with the nearest interpolation on the pixel centres (MASK_INTERPOLATION) so that every
    pixel keeps a class and stays on its image pixel, the instance segmenter passes the binary mask of every object
    with its boxes, the boxes then becoming the tight boxes of the moved masks, the keypoint detector passes the
    keypoints of every object with its boxes, moved without being renamed, and as its masks the binary regions its
    heatmap loss leaves out, while a classifier simply omits them.
    Images are float BGR tensors in the [0, 255] range, either single images [B, C, H, W] or sequences
    [B, T, C, H, W] for the sequence networks, in which case every frame of a sequence receives exactly the same
    random augmentation so that the sequence stays coherent, unless the photometric types are drawn frame by frame
    ('independent_frames'). The real frames of the sequences can also be shuffled ('shuffle_frames', see
    shuffle_frames) for the sequences whose frame order carries no information. The strengths of the types are those
    of the task, SUPERVISED_AUGMENTATION_STRENGTHS or SELF_SUPERVISED_AUGMENTATION_STRENGTHS.
    """

    def __init__(self, augmentation_configuration: Dict[str, Any], self_supervised: bool = False,
                 sequence_augmentation: bool = False) -> None:
        """
        Reads the 'Data.data_augmentation' configuration section and takes the strengths of the task.

        Args:
            augmentation_configuration (Dict[str, Any]): The section holding the global 'activated' flag, the
                'probability' of augmenting an image, the 'independent_frames' flag and the 'shuffle_frames' settings
                of the sequences (sequence tasks only) and, for every augmentation type, its own 'activated' flag
                and 'probability' of being applied to an augmented image, with the 'grid_size' of random_black_out.
            self_supervised (bool): Whether the augmenter draws the views of the self-supervised task, with
                SELF_SUPERVISED_AUGMENTATION_STRENGTHS, instead of augmenting the training batches of a supervised task
                with SUPERVISED_AUGMENTATION_STRENGTHS. Defaults to False.
            sequence_augmentation (bool): Whether the section holds the sequence settings, 'independent_frames' and
                'shuffle_frames', which only the sequence tasks have, the other tasks keeping them off. Defaults to
                False.
        """
        self.activated = augmentation_configuration["activated"]
        self.probability = augmentation_configuration["probability"]

        # Strengths of the task, hard-coded, the self-supervised views being stronger
        self.self_supervised = self_supervised
        self.strengths = (SELF_SUPERVISED_AUGMENTATION_STRENGTHS if self_supervised else
                          SUPERVISED_AUGMENTATION_STRENGTHS)

        # Sequences only: the photometric types drawn frame by frame, and the shuffling of the frames of a sequence, off
        # for the tasks without frames
        self.independent_frames = False
        self.shuffle_frames_activated = False
        self.shuffle_frames_probability = 0.0
        if sequence_augmentation:
            self.independent_frames = augmentation_configuration["independent_frames"]
            self.shuffle_frames_activated = augmentation_configuration["shuffle_frames"]["activated"]
            self.shuffle_frames_probability = augmentation_configuration["shuffle_frames"]["probability"]

        # Keep the activation flag and probability of every supported augmentation type
        types_configuration = augmentation_configuration["types"]
        self.type_settings = {name: (types_configuration[name]["activated"], types_configuration[name]["probability"])
                              for name in AUGMENTATION_TYPES}

        # Rows and columns of the grid of random_black_out, finer grids hiding smaller parts of the image
        self.black_out_grid_size = types_configuration["random_black_out"]["grid_size"]

    def augment_batch(self, images: torch.Tensor, bounding_boxes: Optional[List[torch.Tensor]] = None,
                      labels: Optional[List[torch.Tensor]] = None, masks: Optional[torch.Tensor] = None,
                      instance_masks: Optional[List[torch.Tensor]] = None,
                      keypoints: Optional[List[torch.Tensor]] = None) -> Tuple[
                      torch.Tensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]], Optional[torch.Tensor],
                      Optional[List[torch.Tensor]], Optional[List[torch.Tensor]], List[AugmentationReport]]:
        """
        Augments every sample of a training batch independently, each one with probability 'probability', and reports
        the types drawn for every sample with their values, for the information panels of the input viewer.

        Args:
            images (torch.Tensor): Float BGR images in [0, 255], of shape [B, C, H, W] or [B, T, C, H, W].
            bounding_boxes (Optional[List[torch.Tensor]]): Per-image [N, 4] boxes in absolute [x1, y1, x2, y2]
                pixel coordinates, or None for networks without boxes. Defaults to None.
            labels (Optional[List[torch.Tensor]]): Per-image [N] labels matching the boxes. Defaults to None.
            masks (Optional[torch.Tensor]): The [B, H, W] int64 class index of every pixel of the images, filled with
                the ignored label where a geometric type uncovers the image, or the [B, H, W] uint8 binary regions the
                keypoint detector leaves out of its heatmap loss, filled with 0, or None for networks without masks.
                Defaults to None.
            instance_masks (Optional[List[torch.Tensor]]): Per-image [N, H, W] uint8 binary masks matching the boxes
                (instance segmentation), or None. Defaults to None.
            keypoints (Optional[List[torch.Tensor]]): Per-image [N, K, 3] float keypoints (x, y, visibility) of the
                objects of the boxes (keypoint detection), or None. Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]], Optional[torch.Tensor],
                Optional[List[torch.Tensor]], Optional[List[torch.Tensor]], List[AugmentationReport]]: The augmented
                images, boxes, labels, masks, instance masks and keypoints, with the same structure as the inputs, and
                the augmentation report of every sample.

        Raises:
            ValueError: If boxes, masks or keypoints are given with sequences, boxes and labels are not given together,
                or instance masks or keypoints are given without boxes, the instance masks also with class masks.
        """
        if not self.activated:
            return (images, bounding_boxes, labels, masks, instance_masks, keypoints,
                    [create_empty_report() for sample_index in range(images.shape[0])])

        # Boxes and masks are only supported for single images, and the boxes always travel together with their labels
        if images.dim() == 5 and (bounding_boxes is not None or masks is not None or instance_masks is not None or
                                  keypoints is not None):
            raise ValueError("Bounding boxes, masks and keypoints are not supported for image sequences "
                             "[B, T, C, H, W].")
        if (bounding_boxes is None) != (labels is None):
            raise ValueError("Bounding boxes and labels must be given together.")

        # The instance masks belong to the boxes, and share the mask argument of the geometric methods
        if instance_masks is not None and (bounding_boxes is None or masks is not None):
            raise ValueError("Instance masks must be given with their bounding boxes and without class masks.")

        # The keypoints belong to the objects of the boxes, removed with them
        if keypoints is not None and bounding_boxes is None:
            raise ValueError("Keypoints must be given with the bounding boxes of their objects.")

        augmented_images, augmented_boxes, augmented_labels, augmented_masks = [], [], [], []
        augmented_instance_masks, augmented_keypoints, augmentation_reports = [], [], []

        # Decide independently for every sample whether it is augmented, an unaugmented sample keeping an empty report
        for sample_index in range(images.shape[0]):
            sample_image = images[sample_index]
            sample_boxes = bounding_boxes[sample_index] if bounding_boxes is not None else None
            sample_labels = labels[sample_index] if labels is not None else None
            sample_mask = masks[sample_index] if masks is not None else None
            sample_instance_masks = instance_masks[sample_index] if instance_masks is not None else None
            sample_keypoints = keypoints[sample_index] if keypoints is not None else None
            sample_report = create_empty_report()

            if random.random() < self.probability:
                (sample_image, sample_boxes, sample_labels, sample_mask, sample_instance_masks, sample_keypoints,
                 sample_report) = self.augment_sample(image=sample_image, bounding_boxes=sample_boxes,
                                                      labels=sample_labels, mask=sample_mask,
                                                      instance_masks=sample_instance_masks, keypoints=sample_keypoints)

            # Collect the sample, augmented or not, with its report
            augmented_images.append(sample_image)
            augmented_boxes.append(sample_boxes)
            augmented_labels.append(sample_labels)
            augmented_masks.append(sample_mask)
            augmented_instance_masks.append(sample_instance_masks)
            augmented_keypoints.append(sample_keypoints)
            augmentation_reports.append(sample_report)

        output_boxes = augmented_boxes if bounding_boxes is not None else None
        output_labels = augmented_labels if labels is not None else None
        output_masks = torch.stack(tensors=augmented_masks) if masks is not None else None
        output_instance_masks = augmented_instance_masks if instance_masks is not None else None
        output_keypoints = augmented_keypoints if keypoints is not None else None

        return (torch.stack(tensors=augmented_images), output_boxes, output_labels, output_masks, output_instance_masks,
                output_keypoints, augmentation_reports)

    def create_views(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[AugmentationReport],
                                                          List[AugmentationReport]]:
        """
        Draws the two random views of every image of a training or validation batch of the self-supervised task, which
        the self-supervised loss pulls together: every view of every image goes through every activated type with its
        own draws and the self-supervised strengths, so that the only thing the two views of an image share is its
        content. The global 'probability' is ignored, every image getting its two views, and the types of
        SECOND_VIEW_ONLY_TYPES are drawn for the second view only. The views are computed in float32 even within the
        bfloat16 autocast of a mixed precision forward pass, so that the interpolations and blurs keep their precision.

        Args:
            images (torch.Tensor): The [B, 3, H, W] float BGR images in [0, 255].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[AugmentationReport], List[AugmentationReport]]: The first and the
                second [B, 3, H, W] float32 BGR views of the images in [0, 255], and the augmentation report of every
                image for each of the two views.

        Raises:
            ValueError: If the augmenter was not built for the self-supervised task, or the images are not a
                [B, C, H, W] batch.
        """
        if not self.self_supervised:
            raise ValueError("create_views draws the views of the self-supervised task: build the DataAugmenter with "
                             "self_supervised=True.")
        if images.dim() != 4:
            raise ValueError(f"The views are drawn for a [B, C, H, W] batch of images, got the shape "
                             f"{list(images.shape)}.")

        # Every view of the batch drawn image by image, all the first views first, without the global probability
        views, view_reports = [], []
        with torch.autocast(device_type=images.device.type, enabled=False):
            float_images = images.to(dtype=torch.float32)
            for view_index in range(NUMBER_VIEWS):
                view_images, reports = [], []
                for image in float_images:
                    view_image, report = image, create_empty_report()
                    if self.activated:
                        view_image, _, _, _, _, _, report = self.augment_sample(
                            image=image, bounding_boxes=None, labels=None, mask=None, view_index=view_index)
                    view_images.append(view_image)
                    reports.append(report)
                views.append(torch.stack(tensors=view_images))
                view_reports.append(reports)
        return views[0], views[1], view_reports[0], view_reports[1]

    def augment_sample(self, image: torch.Tensor, bounding_boxes: Optional[torch.Tensor],
                       labels: Optional[torch.Tensor], mask: Optional[torch.Tensor],
                       instance_masks: Optional[torch.Tensor] = None, keypoints: Optional[torch.Tensor] = None,
                       view_index: Optional[int] = None) -> Tuple[
                       torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
                       Optional[torch.Tensor], Optional[torch.Tensor], AugmentationReport]:
        """
        Applies every activated augmentation type to one sample, or one view of an image of the self-supervised task,
        each type with its own probability, and reports the applied types with their drawn values. A sequence
        [T, C, H, W] is transformed as a whole, so all its frames share the same random parameters, except the
        photometric types with 'independent_frames', drawn for every frame. The keypoints of the keypoint detection
        move with the image as boxes of zero size, so that every geometric type moves them exactly, and keep their
        names: a flip or a rotation never renames a keypoint, the user choosing the geometric types that suit the
        keypoints of the dataset.

        Args:
            image (torch.Tensor): A float BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].
            bounding_boxes (Optional[torch.Tensor]): The [N, 4] absolute [x1, y1, x2, y2] boxes, or None.
            labels (Optional[torch.Tensor]): The [N] labels of the boxes, or None.
            mask (Optional[torch.Tensor]): The [H, W] int64 class index of every pixel, the [H, W] uint8 binary
                regions the keypoint detector leaves out of its heatmap loss, or None.
            instance_masks (Optional[torch.Tensor]): The [N, H, W] uint8 binary masks of the boxes, or None. Defaults
                to None.
            keypoints (Optional[torch.Tensor]): The [N, K, 3] float keypoints (x, y, visibility) of the objects of the
                boxes, or None. Defaults to None.
            view_index (Optional[int]): The index of the drawn view of the self-supervised task (create_views), 0 for
                the first view, which never draws the types of SECOND_VIEW_ONLY_TYPES, or None for a supervised task.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
                Optional[torch.Tensor], Optional[torch.Tensor], AugmentationReport]: The augmented image, boxes,
                labels, mask, instance masks and keypoints, and the report of the applied types.
        """
        report = create_empty_report()
        report["augmented"] = True

        # The keypoints travel as [x, y, x, y] boxes of zero size after the boxes of the objects, split again after the
        # geometric types
        number_objects = bounding_boxes.shape[0] if bounding_boxes is not None else 0
        if keypoints is not None:
            point_boxes = keypoints[..., :2].reshape(-1, 2).to(dtype=bounding_boxes.dtype).repeat(1, 2)
            bounding_boxes = torch.cat(tensors=[bounding_boxes, point_boxes], dim=0)

        # The instance masks go through the mask argument of the geometric methods, except for an image without objects,
        # their pixels counted first to measure the share of every object left inside the image afterwards
        moves_instance_masks = instance_masks is not None and instance_masks.shape[0] > 0
        if moves_instance_masks:
            mask = instance_masks
            original_mask_pixels = instance_masks.flatten(start_dim=1).sum(dim=1, dtype=torch.float32)

        # Apply the geometric augmentations, which move the boxes and the mask together with the image, the area scale
        # of the objects (the zoom of a crop or a zoom-out) multiplying along the way, every applied type being
        # reported with its drawn values
        geometry_changed = False
        area_scale = 1.0
        for augmentation_type in GEOMETRIC_AUGMENTATION_TYPES:
            if self.should_apply(augmentation_type=augmentation_type, view_index=view_index):
                augmentation_method = getattr(self, augmentation_type)
                image, bounding_boxes, mask, type_area_scale, drawn_values = augmentation_method(
                    image=image, bounding_boxes=bounding_boxes, mask=mask)
                report["geometric_augmentations"].append((augmentation_type, drawn_values))
                area_scale *= type_area_scale
                geometry_changed = True

        # The moved keypoints taken back from their boxes of zero size, with their visibilities
        if keypoints is not None:
            moved_points = bounding_boxes[number_objects:, :2].reshape(*keypoints.shape[:2], 2)
            keypoints = torch.cat(tensors=[moved_points.to(dtype=keypoints.dtype), keypoints[..., 2:]], dim=-1)
            bounding_boxes = bounding_boxes[:number_objects]

        # The moved instance masks give the exact boxes of their objects, tighter than the moved boxes, and the share of
        # the pixels every object kept inside the image, out of those it would have at the new scale
        visible_fractions = None
        if moves_instance_masks:
            instance_masks, mask = mask, None
            if geometry_changed:
                bounding_boxes = compute_mask_bounding_boxes(instance_masks=instance_masks).to(
                    dtype=bounding_boxes.dtype)
                visible_mask_pixels = instance_masks.flatten(start_dim=1).sum(dim=1, dtype=torch.float32)
                expected_mask_pixels = original_mask_pixels * area_scale
                visible_fractions = torch.where(expected_mask_pixels > 0.0,
                                                visible_mask_pixels / expected_mask_pixels.clamp(min=1e-6),
                                                torch.ones_like(input=visible_mask_pixels))

        # Clip the boxes to the image and remove the ones that left it or kept too small a share of their area inside
        # it, together with their labels, masks and keypoints, the labels of the removed objects being reported
        if bounding_boxes is not None:
            bounding_boxes, labels, instance_masks, keypoints, report["removed_labels"] = self.clean_bounding_boxes(
                bounding_boxes=bounding_boxes, labels=labels, image_height=image.shape[-2],
                image_width=image.shape[-1], instance_masks=instance_masks, visible_fractions=visible_fractions,
                keypoints=keypoints)

        # Apply the photometric augmentations, which never move the boxes nor change the mask, once for the sample or
        # frame by frame, the report keeping one list of applied types per draw
        if image.dim() == 4 and self.independent_frames:
            augmented_frames = [self.apply_photometric_augmentations(image=frame, view_index=view_index)
                                for frame in image]
            image = torch.stack(tensors=[augmented_frame for augmented_frame, frame_types in augmented_frames])
            report["photometric_augmentations"] = [frame_types for augmented_frame, frame_types in augmented_frames]
        else:
            image, applied_types = self.apply_photometric_augmentations(image=image, view_index=view_index)
            report["photometric_augmentations"] = [applied_types]

        return image.clamp(min=0.0, max=255.0), bounding_boxes, labels, mask, instance_masks, keypoints, report

    def apply_photometric_augmentations(self, image: torch.Tensor, view_index: Optional[int] = None) -> Tuple[
                                        torch.Tensor, List[Tuple[str, str]]]:
        """
        Applies every activated photometric augmentation type to an image, a sequence or one frame of a sequence, each
        with its own probability and random parameters, a single one of the histogram equalizations drawn together
        (EXCLUSIVE_EQUALIZATION_TYPES) being kept at random.

        Args:
            image (torch.Tensor): A float BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].
            view_index (Optional[int]): The index of the drawn view of the self-supervised task, or None for a
                supervised task (see should_apply). Defaults to None.

        Returns:
            Tuple[torch.Tensor, List[Tuple[str, str]]]: The augmented image or sequence, and the applied types with
                their drawn values, in order.
        """
        # Draw every type, then keep one of the histogram equalizations drawn together, chosen at random
        drawn_types = [augmentation_type for augmentation_type in PHOTOMETRIC_AUGMENTATION_TYPES
                       if self.should_apply(augmentation_type=augmentation_type, view_index=view_index)]
        drawn_equalizations = [augmentation_type for augmentation_type in drawn_types
                               if augmentation_type in EXCLUSIVE_EQUALIZATION_TYPES]
        if len(drawn_equalizations) > 1:
            kept_equalization = random.choice(seq=drawn_equalizations)
            drawn_types = [augmentation_type for augmentation_type in drawn_types
                           if augmentation_type not in drawn_equalizations or augmentation_type == kept_equalization]

        # Apply the kept types in the order of PHOTOMETRIC_AUGMENTATION_TYPES, recording their drawn values
        applied_types = []
        for augmentation_type in drawn_types:
            image, drawn_values = getattr(self, augmentation_type)(image=image)
            applied_types.append((augmentation_type, drawn_values))
        return image, applied_types

    def shuffle_frames(self, images: torch.Tensor, sequence_mask: torch.Tensor,
                       frame_block_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Shuffles the real frames of a training batch of sequences, for the sequences whose frame order carries no
        information (e.g. sets of photos): every sequence is shuffled with the 'shuffle_frames' probability,
        independently of the global augmentation probability, its real frames being permuted and its padding frames
        staying last. The frames are shuffled within every block of frame_block_length frames, so that each member
        of a pair is shuffled on its own.

        Args:
            images (torch.Tensor): The [B, T, C, H, W] frames of the batch.
            sequence_mask (torch.Tensor): The [B, T] booleans, True for the real frames, which come first in every
                block.
            frame_block_length (int): The number of frames of a sequence or member, T being a multiple of it.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The shuffled [B, T, C, H, W] frames and the [B, T] frame order, the
                index of the original frame now at every position, to move the frame labels along.
        """
        batch_size, number_frames = sequence_mask.shape
        frame_orders = torch.arange(number_frames, device=images.device).repeat(batch_size, 1)
        if not self.activated or not self.shuffle_frames_activated:
            return images, frame_orders

        # Every sequence or member drawn on its own, its real frames permuted among themselves
        for sample_index in range(batch_size):
            for block_start in range(0, number_frames, frame_block_length):
                if random.random() >= self.shuffle_frames_probability:
                    continue
                block_end = block_start + frame_block_length
                number_real_frames = int(sequence_mask[sample_index, block_start:block_end].sum())
                frame_orders[sample_index, block_start:block_start + number_real_frames] = torch.randperm(
                    n=number_real_frames, device=images.device) + block_start

        # Frames gathered in their new order, sample by sample
        sample_indices = torch.arange(batch_size, device=images.device)[:, None]
        return images[sample_indices, frame_orders], frame_orders

    def should_apply(self, augmentation_type: str, view_index: Optional[int] = None) -> bool:
        """
        Draws whether an augmentation type is applied to the current sample or view, the first view of the
        self-supervised task never drawing the types of SECOND_VIEW_ONLY_TYPES.

        Args:
            augmentation_type (str): The name of the augmentation type.
            view_index (Optional[int]): The index of the drawn view of the self-supervised task, 0 for the first view,
                or None for a supervised task. Defaults to None.

        Returns:
            bool: True if the type is activated, allowed for the view, and its random draw falls below its probability.
        """
        is_activated, probability = self.type_settings[augmentation_type]
        if view_index == 0 and augmentation_type in SECOND_VIEW_ONLY_TYPES:
            return False
        return is_activated and random.random() < probability

    def describe_report(self, report: AugmentationReport, label_names: Optional[Dict[int, str]] = None,
                        real_frames: Optional[List[int]] = None) -> List[str]:
        """
        Writes the augmentation report of a training sample as the lines of its information panel in the input viewer
        ('view_input_images'): the applied geometric types, then the photometric ones (one line per frame when they are
        drawn frame by frame), each with its drawn values, and the objects removed by the geometric augmentations.

        Args:
            report (AugmentationReport): The report of the sample, given by augment_batch.
            label_names (Optional[Dict[int, str]]): The class name of every label index, to name the removed objects.
                Defaults to None, in which case the label indices are written.
            real_frames (Optional[List[int]]): The indices of the frames written in the frame by frame lines of a
                sequence, its real frames, so that its padding frames are left out. Defaults to None, in which case
                every frame is written.

        Returns:
            List[str]: The lines of the panel, the types indented under their group.
        """
        if not self.activated:
            return ["Augmentation off (data_augmentation.activated False)"]
        if not report["augmented"]:
            return [f"Not augmented (global probability {self.probability} not drawn)"]

        # Geometric types, shared by every frame of a sequence
        lines = ["Geometric :"]
        lines += [f"  {augmentation_type} : {drawn_values}" if drawn_values else f"  {augmentation_type}"
                  for augmentation_type, drawn_values in report["geometric_augmentations"]] or ["  none"]

        # Photometric types, drawn once for the sample or frame by frame
        photometric_augmentations = report["photometric_augmentations"]
        if len(photometric_augmentations) == 1:
            lines.append("Photometric :")
            lines += [f"  {augmentation_type} : {drawn_values}" if drawn_values else f"  {augmentation_type}"
                      for augmentation_type, drawn_values in photometric_augmentations[0]] or ["  none"]
        else:
            lines.append("Photometric, drawn frame by frame :")
            written_frames = real_frames if real_frames is not None else range(len(photometric_augmentations))
            for frame_index in written_frames:
                frame_descriptions = [f"{augmentation_type} ({drawn_values})" if drawn_values else augmentation_type
                                      for augmentation_type, drawn_values in photometric_augmentations[frame_index]]
                lines.append(f"  Frame {frame_index} : {', '.join(frame_descriptions) or 'none'}")

        # Objects removed after the geometric augmentations, for the tasks with boxes
        if report["removed_labels"] is not None:
            removed_labels = report["removed_labels"].tolist()
            removed_names = [label_names.get(label, str(label)) if label_names else str(label)
                             for label in removed_labels]
            lines.append(f"Removed objects : {len(removed_labels)} ({', '.join(removed_names)})" if removed_labels
                         else "Removed objects : none")
        return lines

    @staticmethod
    def clean_bounding_boxes(bounding_boxes: torch.Tensor, labels: torch.Tensor, image_height: int, image_width: int,
                             instance_masks: Optional[torch.Tensor] = None,
                             visible_fractions: Optional[torch.Tensor] = None,
                             keypoints: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor,
                                                                                Optional[torch.Tensor],
                                                                                Optional[torch.Tensor], torch.Tensor]:
        """
        Clips the boxes to the image borders after the geometric augmentations and removes the boxes that became
        too small (mostly cropped or rotated out of the image) or that kept less than MINIMUM_VISIBLE_AREA_FRACTION of
        their area inside the image (a sliver of an object at the border of a crop, which would teach the detector to
        fire on fragments), so the detector only receives valid targets. The instance masks of the removed boxes are
        removed with them, and their labels are returned for the augmentation report. The keypoints that left the
        image become unlabelled (visibility 0, position 0), and an object keeping a labelled keypoint inside the image
        is never removed, since the keypoint detector would otherwise learn its visible keypoints as background.

        Args:
            bounding_boxes (torch.Tensor): The [N, 4] absolute [x1, y1, x2, y2] boxes, moved without clipping so that
                their part outside the image is known.
            labels (torch.Tensor): The [N] labels of the boxes.
            image_height (int): The height of the augmented image.
            image_width (int): The width of the augmented image.
            instance_masks (Optional[torch.Tensor]): The [N, H, W] binary masks of the boxes, or None. Defaults to
                None.
            visible_fractions (Optional[torch.Tensor]): The [N] share of every object left inside the image measured
                on its mask pixels (instance segmentation), or None to measure it on the boxes, as the clipped area
                over the area of the moved box. Defaults to None.
            keypoints (Optional[torch.Tensor]): The [N, K, 3] moved keypoints (x, y, visibility) of the boxes, or None.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]: The kept
                boxes, their labels, their instance masks and their keypoints (None without them), and the labels of
                the removed boxes.
        """
        clipped_boxes = bounding_boxes.clone()
        clipped_boxes[:, 0::2] = clipped_boxes[:, 0::2].clamp(min=0.0, max=float(image_width))
        clipped_boxes[:, 1::2] = clipped_boxes[:, 1::2].clamp(min=0.0, max=float(image_height))

        # Keep only the boxes that still have a visible width and height
        box_widths = clipped_boxes[:, 2] - clipped_boxes[:, 0]
        box_heights = clipped_boxes[:, 3] - clipped_boxes[:, 1]
        kept_mask = (box_widths >= MINIMUM_BOX_SIDE) & (box_heights >= MINIMUM_BOX_SIDE)

        # Share of every box left inside the image, unless the masks measured it, and removal of the fragments
        if visible_fractions is None:
            moved_areas = ((bounding_boxes[:, 2] - bounding_boxes[:, 0]).clamp(min=0.0) *
                           (bounding_boxes[:, 3] - bounding_boxes[:, 1]).clamp(min=0.0))
            visible_fractions = box_widths.clamp(min=0.0) * box_heights.clamp(min=0.0) / moved_areas.clamp(min=1e-6)
        kept_mask &= visible_fractions >= MINIMUM_VISIBLE_AREA_FRACTION

        # Keypoints outside the image unlabelled, the objects keeping a labelled keypoint inside the image kept
        kept_keypoints = None
        if keypoints is not None:
            inside_image = ((keypoints[..., 0] >= 0.0) & (keypoints[..., 0] < image_width) &
                            (keypoints[..., 1] >= 0.0) & (keypoints[..., 1] < image_height))
            labelled = (keypoints[..., 2] > 0) & inside_image
            keypoints = torch.where(labelled[..., None], keypoints, torch.zeros_like(input=keypoints))
            kept_mask |= labelled.any(dim=1)
            kept_keypoints = keypoints[kept_mask]

        kept_instance_masks = instance_masks[kept_mask] if instance_masks is not None else None
        return clipped_boxes[kept_mask], labels[kept_mask], kept_instance_masks, kept_keypoints, labels[~kept_mask]

    @staticmethod
    def wrap_bounding_boxes(bounding_boxes: torch.Tensor, image: torch.Tensor) -> tv_tensors.BoundingBoxes:
        """
        Wraps plain boxes as torchvision BoundingBoxes so that torchvision's geometric functions transform them
        exactly like the image. The boxes are moved without being clipped to the image, so that clean_bounding_boxes
        can measure the share of every box left inside it.

        Args:
            bounding_boxes (torch.Tensor): The [N, 4] absolute [x1, y1, x2, y2] boxes.
            image (torch.Tensor): The image the boxes belong to, giving the canvas size.

        Returns:
            tv_tensors.BoundingBoxes: The wrapped boxes.
        """
        return tv_tensors.BoundingBoxes(data=bounding_boxes, format="XYXY",
                                        canvas_size=(image.shape[-2], image.shape[-1]), clamping_mode=None)

    def random_crop(self, image: torch.Tensor, bounding_boxes: Optional[torch.Tensor],
                    mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                                                           Optional[torch.Tensor], float, str]:
        """
        Crops a random region and resizes it back to the original size, which zooms into the image. In the supervised
        tasks (crop_side_range) the region keeps the aspect ratio of the image and 75 to 100% of each side, a slight
        zoom; in the self-supervised task (crop_area_range) it covers 20 to 100% of the area with an aspect ratio of
        x1/2 to x2 (draw_area_crop), the random resized crop of SimCLR, its content being stretched when it is resized
        back. Boxes are shifted and scaled accordingly; boxes leaving the crop, or keeping too small a share of their
        area inside it, are removed afterwards by 'clean_bounding_boxes'. The mask is cropped and resized with the
        nearest interpolation on the pixel centres, aligned with the image.

        Args:
            image (torch.Tensor): The image [C, H, W] or sequence [T, C, H, W].
            bounding_boxes (Optional[torch.Tensor]): The [N, 4] boxes, or None.
            mask (Optional[torch.Tensor]): The [H, W] class mask or [N, H, W] instance masks, or None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, str]: The cropped image, boxes
                and mask, the area scale of the objects (the squared zoom, above 1) and the drawn crop for the
                augmentation report.
        """
        image_height, image_width = image.shape[-2], image.shape[-1]

        # Draw a region of random area and aspect ratio (self-supervised task), or one crop fraction for both sides
        # to keep the aspect ratio, and a random position for the crop
        if self.strengths["crop_area_range"] is not None:
            top, left, crop_height, crop_width = self.draw_area_crop(image_height=image_height, image_width=image_width)
            area_fraction = crop_height * crop_width / (image_height * image_width)
            drawn_values = (f"{crop_width}x{crop_height} pixels ({area_fraction * 100:.0f} % of the area), at top "
                            f"{top} and left {left} pixels")
        else:
            crop_fraction = random.uniform(*self.strengths["crop_side_range"])
            crop_height, crop_width = round(image_height * crop_fraction), round(image_width * crop_fraction)
            top, left = random.randint(0, image_height - crop_height), random.randint(0, image_width - crop_width)
            drawn_values = f"{crop_fraction * 100:.0f} % of each side, at top {top} and left {left} pixels"

        cropped_image, cropped_boxes, cropped_mask = self.crop_and_resize(
            image=image, bounding_boxes=bounding_boxes, mask=mask, top=top, left=left, crop_height=crop_height,
            crop_width=crop_width)
        return (cropped_image, cropped_boxes, cropped_mask, (image_height * image_width) / (crop_height * crop_width),
                drawn_values)

    def draw_area_crop(self, image_height: int, image_width: int) -> Tuple[int, int, int, int]:
        """
        Draws the region of random_crop in the self-supervised task, as the random resized crop of torchvision used by
        SimCLR: an area within crop_area_range of the image area and a width / height ratio drawn log-uniformly within
        crop_aspect_ratio_range, drawn again up to MAXIMUM_REGION_ATTEMPTS times until the region fits in the image,
        the central region of the ratio of the range nearest to the image ratio being taken otherwise.

        Args:
            image_height (int): The height of the image in pixels.
            image_width (int): The width of the image in pixels.

        Returns:
            Tuple[int, int, int, int]: The top row, the left column, the height and the width of the region, in pixels.
        """
        minimum_ratio, maximum_ratio = self.strengths["crop_aspect_ratio_range"]
        logarithm_ratio_range = (math.log(minimum_ratio), math.log(maximum_ratio))

        # Area and ratio drawn until the region fits, at a random position
        for attempt_index in range(MAXIMUM_REGION_ATTEMPTS):
            target_area = image_height * image_width * random.uniform(*self.strengths["crop_area_range"])
            aspect_ratio = math.exp(random.uniform(*logarithm_ratio_range))
            crop_width = round(math.sqrt(target_area * aspect_ratio))
            crop_height = round(math.sqrt(target_area / aspect_ratio))
            if 0 < crop_width <= image_width and 0 < crop_height <= image_height:
                return (random.randint(0, image_height - crop_height), random.randint(0, image_width - crop_width),
                        crop_height, crop_width)

        # Central region of the nearest ratio of the range, the whole image when its ratio lies within the range
        image_ratio = image_width / image_height
        crop_height, crop_width = image_height, image_width
        if image_ratio < minimum_ratio:
            crop_height = round(image_width / minimum_ratio)
        elif image_ratio > maximum_ratio:
            crop_width = round(image_height * maximum_ratio)
        return (image_height - crop_height) // 2, (image_width - crop_width) // 2, crop_height, crop_width

    def random_aspect_crop(self, image: torch.Tensor, bounding_boxes: Optional[torch.Tensor],
                           mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                                                                  Optional[torch.Tensor], float, str]:
        """
        Crops a random region covering 75 to 100% of the width and, drawn on its own, 75 to 100% of the height, and
        resizes it back to the original size, which zooms slightly into the image and stretches its content by x0.75
        to x1.33 along one side: the network meets the objects with the proportions of another camera resolution,
        field of view or framing, which suits the images stretched to the input size without padding ('keep_ratio'
        False). Boxes, masks and keypoints follow exactly like those of 'random_crop'.

        Args:
            image (torch.Tensor): The image [C, H, W] or sequence [T, C, H, W].
            bounding_boxes (Optional[torch.Tensor]): The [N, 4] boxes, or None.
            mask (Optional[torch.Tensor]): The [H, W] class mask or [N, H, W] instance masks, or None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, str]: The cropped image, boxes
                and mask, the area scale of the objects (the product of the zooms of both sides, above 1) and the drawn
                crop for the augmentation report.
        """
        image_height, image_width = image.shape[-2], image.shape[-1]

        # Draw the kept share of the width and of the height independently, which changes the aspect ratio of the
        # content, and a random position for the crop
        width_fraction = random.uniform(*self.strengths["aspect_crop_side_range"])
        height_fraction = random.uniform(*self.strengths["aspect_crop_side_range"])
        crop_height, crop_width = round(image_height * height_fraction), round(image_width * width_fraction)
        top, left = random.randint(0, image_height - crop_height), random.randint(0, image_width - crop_width)

        cropped_image, cropped_boxes, cropped_mask = self.crop_and_resize(
            image=image, bounding_boxes=bounding_boxes, mask=mask, top=top, left=left, crop_height=crop_height,
            crop_width=crop_width)
        drawn_values = (f"{width_fraction * 100:.0f} % of the width and {height_fraction * 100:.0f} % of the height, "
                        f"at top {top} and left {left} pixels")
        return (cropped_image, cropped_boxes, cropped_mask, (image_height * image_width) / (crop_height * crop_width),
                drawn_values)

    def crop_and_resize(self, image: torch.Tensor, bounding_boxes: Optional[torch.Tensor],
                        mask: Optional[torch.Tensor], top: int, left: int, crop_height: int,
                        crop_width: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Cuts a region out of an image and resizes it back to the size of the image, moving the boxes and the mask with
        it, shared by 'random_crop', whose region keeps the aspect ratio of the image, and 'random_aspect_crop', whose
        region changes it. Boxes leaving the region, or keeping too small a share of their area inside it, are removed
        afterwards by 'clean_bounding_boxes'.

        Args:
            image (torch.Tensor): The image [C, H, W] or sequence [T, C, H, W].
            bounding_boxes (Optional[torch.Tensor]): The [N, 4] boxes, or None.
            mask (Optional[torch.Tensor]): The [H, W] class mask or [N, H, W] instance masks, or None.
            top (int): The row of the top edge of the region, in pixels.
            left (int): The column of the left edge of the region, in pixels.
            crop_height (int): The height of the region, in pixels.
            crop_width (int): The width of the region, in pixels.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: The resized region of the image, and
                the boxes and mask moved with it.
        """
        image_height, image_width = image.shape[-2], image.shape[-1]
        cropped_image = vision_functional.resized_crop(image, top=top, left=left, height=crop_height,
                                                       width=crop_width, size=[image_height, image_width],
                                                       antialias=True)

        # Crop the boxes exactly like the image
        cropped_boxes, cropped_mask = None, None
        if bounding_boxes is not None:
            cropped_boxes = vision_functional.resized_crop(
                self.wrap_bounding_boxes(bounding_boxes=bounding_boxes, image=image), top=top, left=left,
                height=crop_height, width=crop_width, size=[image_height, image_width]).as_subclass(torch.Tensor)

        # Cut the mask, then resize it on the pixel centres like the image, the resized crop of torchvision taking the
        # plain nearest for the masks whatever the interpolation asked
        if mask is not None:
            cropped_mask = vision_functional.resize(
                tv_tensors.Mask(mask[..., top:top + crop_height, left:left + crop_width]),
                size=[image_height, image_width], interpolation=MASK_INTERPOLATION).as_subclass(torch.Tensor)
        return cropped_image, cropped_boxes, cropped_mask

    def random_zoom_out(self, image: torch.Tensor, bounding_boxes: Optional[torch.Tensor],
                        mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                                                               Optional[torch.Tensor], float, str]:
        """
        Shrinks the image to 50 to 100% of each side and places it at a random position of a black canvas of the
        original size, the padding of the data loaders, which shows the objects smaller than in the dataset, the
        opposite of the crop: the finest detection levels then also train on the objects that are large in the
        dataset. Boxes are scaled and shifted with the image and never leave it. The mask is shrunk with the nearest
        interpolation on the pixel centres, aligned with the image, the uncovered border being ignored in a class mask,
        like the padding of the loaders, and background (0) in the binary instance masks.

        Args:
            image (torch.Tensor): The image [C, H, W] or sequence [T, C, H, W].
            bounding_boxes (Optional[torch.Tensor]): The [N, 4] boxes, or None.
            mask (Optional[torch.Tensor]): The [H, W] class mask or [N, H, W] instance masks, or None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, str]: The zoomed out image,
                boxes and mask, the area scale of the objects (the squared shrink factor, below 1) and the drawn
                shrink for the augmentation report.
        """
        image_height, image_width = image.shape[-2], image.shape[-1]

        # Draw one shrink fraction for both sides to keep the aspect ratio, and a random position on the canvas
        zoom_fraction = random.uniform(*self.strengths["zoom_out_side_range"])
        shrunk_height, shrunk_width = round(image_height * zoom_fraction), round(image_width * zoom_fraction)
        top, left = random.randint(0, image_height - shrunk_height), random.randint(0, image_width - shrunk_width)

        # Shrink the image and place it on the black canvas
        zoomed_image = torch.zeros_like(input=image)
        zoomed_image[..., top:top + shrunk_height, left:left + shrunk_width] = vision_functional.resize(
            image, size=[shrunk_height, shrunk_width], antialias=True)

        # Scale and shift the boxes exactly like the image
        zoomed_boxes, zoomed_mask = None, None
        if bounding_boxes is not None:
            scales = torch.tensor(data=[shrunk_width / image_width, shrunk_height / image_height] * 2,
                                  dtype=bounding_boxes.dtype, device=bounding_boxes.device)
            offsets = torch.tensor(data=[left, top] * 2, dtype=bounding_boxes.dtype, device=bounding_boxes.device)
            zoomed_boxes = bounding_boxes * scales + offsets

        # Shrink the mask with the nearest interpolation on the pixel centres onto its canvas, ignored around a class
        # mask and background around binary instance masks
        if mask is not None:
            mask_fill = IGNORED_LABEL_INDEX if mask.dtype == torch.int64 else 0
            zoomed_mask = torch.full_like(input=mask, fill_value=mask_fill)
            zoomed_mask[..., top:top + shrunk_height, left:left + shrunk_width] = vision_functional.resize(
                tv_tensors.Mask(mask), size=[shrunk_height, shrunk_width],
                interpolation=MASK_INTERPOLATION).as_subclass(torch.Tensor)
        drawn_values = f"{zoom_fraction * 100:.0f} % of each side, at top {top} and left {left} pixels"
        return (zoomed_image, zoomed_boxes, zoomed_mask, (shrunk_height * shrunk_width) / (image_height * image_width),
                drawn_values)

    @staticmethod
    def shift_tensor(tensor: torch.Tensor, shift_x: int, shift_y: int, fill_value: float) -> torch.Tensor:
        """
        Moves the content of an image, a sequence or masks by whole pixels along their last two dimensions, for
        random_translation: the part pushed past a border is dropped, unlike torch.roll which would bring it back on the
        other side, and the uncovered border is filled, so that the pixels are moved exactly without any resampling.

        Args:
            tensor (torch.Tensor): The [..., H, W] image [C, H, W], sequence [T, C, H, W], class mask [H, W] or
                instance masks [N, H, W].
            shift_x (int): The shift towards the right in pixels, negative towards the left.
            shift_y (int): The shift downwards in pixels, negative upwards.
            fill_value (float): The value of the uncovered border.

        Returns:
            torch.Tensor: The shifted tensor, of the same shape and type.
        """
        height, width = tensor.shape[-2], tensor.shape[-1]
        shifted_tensor = torch.full_like(input=tensor, fill_value=fill_value)

        # Rows and columns that stay inside the image, nothing being kept when a shift reaches the whole side
        kept_height, kept_width = height - abs(shift_y), width - abs(shift_x)
        if kept_height <= 0 or kept_width <= 0:
            return shifted_tensor

        # Source rows and columns of the kept part, and the rows and columns they land on
        source_top, source_left = max(0, -shift_y), max(0, -shift_x)
        destination_top, destination_left = max(0, shift_y), max(0, shift_x)
        source_rows, source_columns = (slice(source_top, source_top + kept_height),
                                       slice(source_left, source_left + kept_width))
        destination_rows, destination_columns = (slice(destination_top, destination_top + kept_height),
                                                 slice(destination_left, destination_left + kept_width))
        shifted_tensor[..., destination_rows, destination_columns] = tensor[..., source_rows, source_columns]
        return shifted_tensor

    def random_translation(self, image: torch.Tensor, bounding_boxes: Optional[torch.Tensor],
                           mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                                                                  Optional[torch.Tensor], float, str]:
        """
        Shifts the image along its width, its height or both (TRANSLATION_AXES, drawn with equal chances) by whole
        pixels, each shift being drawn up to maximum_translation_fraction of its side in either direction (25 % in the
        supervised and self-supervised tasks), so that every task, the classification included, sees
        its objects away from the positions of the dataset. The uncovered border is black like the padding of the data
        loaders. Boxes move by the same shifts without being clipped, so that 'clean_bounding_boxes' removes the
        objects pushed out of the image or keeping too small a share of their area inside it. The mask moves exactly,
        its uncovered border being ignored in a class mask and background (0) in the binary instance masks.

        Args:
            image (torch.Tensor): The image [C, H, W] or sequence [T, C, H, W].
            bounding_boxes (Optional[torch.Tensor]): The [N, 4] boxes, or None.
            mask (Optional[torch.Tensor]): The [H, W] class mask or [N, H, W] instance masks, or None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, str]: The shifted image, boxes
                and mask, the area scale of the objects (1, the sizes are kept) and the drawn shifts for the
                augmentation report.
        """
        image_height, image_width = image.shape[-2], image.shape[-1]

        # Draw the axes, then the shift along each of them, the other axis keeping a zero shift
        translation_axes = random.choice(TRANSLATION_AXES)
        maximum_shift_x = int(image_width * self.strengths["maximum_translation_fraction"])
        maximum_shift_y = int(image_height * self.strengths["maximum_translation_fraction"])
        shift_x = random.randint(-maximum_shift_x, maximum_shift_x) if "x" in translation_axes else 0
        shift_y = random.randint(-maximum_shift_y, maximum_shift_y) if "y" in translation_axes else 0

        # Shift the image onto a black canvas
        translated_image = self.shift_tensor(tensor=image, shift_x=shift_x, shift_y=shift_y, fill_value=0.0)

        # Shift the boxes by the same number of pixels, and the mask with its own border value
        translated_boxes, translated_mask = None, None
        if bounding_boxes is not None:
            translated_boxes = bounding_boxes + torch.tensor(data=[shift_x, shift_y] * 2, dtype=bounding_boxes.dtype,
                                                             device=bounding_boxes.device)
        if mask is not None:
            mask_fill = IGNORED_LABEL_INDEX if mask.dtype == torch.int64 else 0
            translated_mask = self.shift_tensor(tensor=mask, shift_x=shift_x, shift_y=shift_y, fill_value=mask_fill)

        # Report the shift along every drawn axis, positive towards the right and downwards
        axis_shifts = {"x": shift_x, "y": shift_y}
        drawn_values = ", ".join(f"{axis} {axis_shifts[axis]:+d} pixels" for axis in translation_axes)
        return translated_image, translated_boxes, translated_mask, 1.0, drawn_values

    def random_left_right_flip(self, image: torch.Tensor, bounding_boxes: Optional[torch.Tensor],
                               mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                                                                      Optional[torch.Tensor], float, str]:
        """
        Mirrors the image from left to right, together with its boxes and its mask.

        Args:
            image (torch.Tensor): The image [C, H, W] or sequence [T, C, H, W].
            bounding_boxes (Optional[torch.Tensor]): The [N, 4] boxes, or None.
            mask (Optional[torch.Tensor]): The [H, W] class mask or [N, H, W] instance masks, or None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, str]: The flipped image, boxes
                and mask, the area scale of the objects (1, the sizes are kept) and no drawn value (empty).
        """
        flipped_image = vision_functional.horizontal_flip(image)
        flipped_boxes = None
        if bounding_boxes is not None:
            flipped_boxes = vision_functional.horizontal_flip(
                self.wrap_bounding_boxes(bounding_boxes=bounding_boxes, image=image)).as_subclass(torch.Tensor)
        flipped_mask = mask.flip(dims=(-1,)) if mask is not None else None
        return flipped_image, flipped_boxes, flipped_mask, 1.0, ""

    def random_up_down_flip(self, image: torch.Tensor, bounding_boxes: Optional[torch.Tensor],
                            mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                                                                   Optional[torch.Tensor], float, str]:
        """
        Mirrors the image from top to bottom, together with its boxes and its mask.

        Args:
            image (torch.Tensor): The image [C, H, W] or sequence [T, C, H, W].
            bounding_boxes (Optional[torch.Tensor]): The [N, 4] boxes, or None.
            mask (Optional[torch.Tensor]): The [H, W] class mask or [N, H, W] instance masks, or None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, str]: The flipped image, boxes
                and mask, the area scale of the objects (1, the sizes are kept) and no drawn value (empty).
        """
        flipped_image = vision_functional.vertical_flip(image)
        flipped_boxes = None
        if bounding_boxes is not None:
            flipped_boxes = vision_functional.vertical_flip(
                self.wrap_bounding_boxes(bounding_boxes=bounding_boxes, image=image)).as_subclass(torch.Tensor)
        flipped_mask = mask.flip(dims=(-2,)) if mask is not None else None
        return flipped_image, flipped_boxes, flipped_mask, 1.0, ""

    def random_rotation(self, image: torch.Tensor, bounding_boxes: Optional[torch.Tensor],
                        mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                                                               Optional[torch.Tensor], float, str]:
        """
        Rotates the image by a random angle between -5 and 5 degrees around its centre (-10 and 10 degrees in the
        self-supervised task), filling the uncovered corners with black like the padding of the data loaders. A rotated
        box is no longer axis aligned, so it is replaced by the enclosing box of its rotated corners, which is slightly
        larger than the object. The mask is rotated with the nearest interpolation, its uncovered corners being ignored
        like its padding for a class mask, and background (0) for binary instance masks. The image and the mask are
        rotated on the device from numbers (rotate_on_device, the values of torchvision's rotation), and the boxes by
        torchvision.

        Args:
            image (torch.Tensor): The image [C, H, W] or sequence [T, C, H, W].
            bounding_boxes (Optional[torch.Tensor]): The [N, 4] boxes, or None.
            mask (Optional[torch.Tensor]): The [H, W] class mask or [N, H, W] instance masks, or None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, str]: The rotated image, boxes
                and mask, the area scale of the objects (1, the sizes are kept) and the drawn angle for the
                augmentation report.
        """
        maximum_degrees = self.strengths["maximum_rotation_degrees"]
        angle = random.uniform(-maximum_degrees, maximum_degrees)

        rotated_image = rotate_on_device(image=image, angle=angle, interpolation="bilinear", fill=0.0)

        # Rotate the boxes and the mask by the same angle, the uncovered corners of a class mask being ignored and
        # those of binary instance masks being background
        rotated_boxes, rotated_mask = None, None
        if bounding_boxes is not None:
            rotated_boxes = vision_functional.rotate(
                self.wrap_bounding_boxes(bounding_boxes=bounding_boxes, image=image),
                angle=angle).as_subclass(torch.Tensor)
        if mask is not None:
            mask_fill = IGNORED_LABEL_INDEX if mask.dtype == torch.int64 else 0
            rotated_mask = rotate_on_device(image=mask.unsqueeze(dim=0) if mask.dim() < 3 else mask, angle=angle,
                                            interpolation="nearest", fill=mask_fill)
            rotated_mask = rotated_mask.squeeze(dim=0) if mask.dim() < 3 else rotated_mask
        return rotated_image, rotated_boxes, rotated_mask, 1.0, f"{angle:+.1f} degrees"

    @staticmethod
    def random_90_rotation(image: torch.Tensor, bounding_boxes: Optional[torch.Tensor],
                           mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                                                                  Optional[torch.Tensor], float, str]:
        """
        Rotates a square image by 90 degrees clockwise or counter-clockwise, chosen at random. The rotation is an
        exact pixel permutation, so the boxes and the mask are transformed exactly.

        Args:
            image (torch.Tensor): The square image [C, H, W] or sequence [T, C, H, W].
            bounding_boxes (Optional[torch.Tensor]): The [N, 4] boxes, or None.
            mask (Optional[torch.Tensor]): The [H, W] class mask or [N, H, W] instance masks, or None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, str]: The rotated image, boxes
                and mask, the area scale of the objects (1, the sizes are kept) and the drawn direction for the
                augmentation report.

        Raises:
            ValueError: If the image is not square, since the batch could then no longer be stacked.
        """
        image_height, image_width = image.shape[-2], image.shape[-1]
        if image_height != image_width:
            raise ValueError(f"random_90_rotation requires square images, got {image_height}x{image_width}.")

        # Counter-clockwise rotation maps a point (x, y) to (y, W - x), clockwise maps it to (H - y, x)
        counter_clockwise = random.random() < 0.5
        rotated_image = torch.rot90(image, k=1 if counter_clockwise else -1, dims=(-2, -1))
        rotated_mask = torch.rot90(mask, k=1 if counter_clockwise else -1, dims=(-2, -1)) if mask is not None else None
        drawn_values = "counter-clockwise" if counter_clockwise else "clockwise"
        if bounding_boxes is None:
            return rotated_image, None, rotated_mask, 1.0, drawn_values

        x_1, y_1, x_2, y_2 = bounding_boxes.unbind(dim=-1)
        if counter_clockwise:
            rotated_boxes = torch.stack(tensors=[y_1, image_width - x_2, y_2, image_width - x_1], dim=-1)
        else:
            rotated_boxes = torch.stack(tensors=[image_height - y_2, x_1, image_height - y_1, x_2], dim=-1)
        return rotated_image, rotated_boxes, rotated_mask, 1.0, drawn_values

    @staticmethod
    def refuse_targets(augmentation_type: str, bounding_boxes: Optional[torch.Tensor],
                       mask: Optional[torch.Tensor]) -> None:
        """
        Refuses the boxes, keypoints and masks for the geometric types moving the image only
        (IMAGE_ONLY_AUGMENTATION_TYPES), which the checks of the configuration refuse for the tasks having them, so
        that a target that would stay at its place under a moved image never reaches the loss.

        Args:
            augmentation_type (str): The name of the geometric type.
            bounding_boxes (Optional[torch.Tensor]): The [N, 4] boxes, with the keypoints as boxes of zero size, or
                None.
            mask (Optional[torch.Tensor]): The [H, W] class mask or [N, H, W] instance masks, or None.

        Raises:
            ValueError: If boxes or a mask are given.
        """
        if bounding_boxes is not None or mask is not None:
            raise ValueError(f"{augmentation_type} moves the image only and cannot move bounding boxes, masks or "
                             f"keypoints: 'Data.data_augmentation.types.{augmentation_type}.activated' must be False "
                             f"for this task.")

    def random_perspective(self, image: torch.Tensor, bounding_boxes: Optional[torch.Tensor],
                           mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                                                                  Optional[torch.Tensor], float, str]:
        """
        Distorts the perspective of the image, a change of viewpoint: each of its four corners moves inwards by up to
        perspective_distortion_scale of half the image width and of half its height, drawn on its own along both
        axes (5 % in the supervised and self-supervised tasks, as kornia's RandomPerspective), and the
        image is warped into the quadrilateral of the moved corners with a bilinear sampling, its uncovered border being
        black like the padding of the data loaders. The coefficients of the transformation are solved on the host
        (compute_perspective_coefficients) and the sampling grid is computed on the device from them, so that the host
        never waits for the device. No box, mask or keypoint follows the distortion (refuse_targets). A sequence
        receives the same distortion on all its frames.

        Args:
            image (torch.Tensor): The image [C, H, W] or sequence [T, C, H, W].
            bounding_boxes (Optional[torch.Tensor]): None, the type moving the image only.
            mask (Optional[torch.Tensor]): None, the type moving the image only.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, str]: The distorted image, no
                boxes nor mask, the area scale of 1 and the largest move of a corner for the augmentation report.
        """
        self.refuse_targets(augmentation_type="random_perspective", bounding_boxes=bounding_boxes, mask=mask)
        image_height, image_width = image.shape[-2], image.shape[-1]
        horizontal_reach = self.strengths["perspective_distortion_scale"] * image_width / 2.0
        vertical_reach = self.strengths["perspective_distortion_scale"] * image_height / 2.0

        # Corners of the image, each one moved inwards by its own draw along both axes
        image_corners = [(0.0, 0.0), (image_width - 1.0, 0.0), (image_width - 1.0, image_height - 1.0),
                         (0.0, image_height - 1.0)]
        inward_directions = [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)]
        moved_corners = [(corner_x + direction_x * horizontal_reach * random.random(),
                          corner_y + direction_y * vertical_reach * random.random())
                         for (corner_x, corner_y), (direction_x, direction_y) in zip(image_corners, inward_directions)]

        # Position in the image of every pixel of the distorted image, computed on the device from the coefficients
        (column_from_column, column_from_row, column_shift, row_from_column, row_from_row, row_shift,
         denominator_from_column, denominator_from_row) = compute_perspective_coefficients(
            output_points=moved_corners, input_points=image_corners)
        columns, rows = get_pixel_coordinates(height=image_height, width=image_width, device=image.device)
        denominators = denominator_from_column * columns + denominator_from_row * rows + 1.0
        source_columns = (column_from_column * columns + column_from_row * rows + column_shift) / denominators
        source_rows = (row_from_column * columns + row_from_row * rows + row_shift) / denominators

        # Bilinear sampling of the image at these positions, the corners being pixel centres, black outside the image
        sampling_grid = torch.stack(tensors=[2.0 * source_columns / (image_width - 1) - 1.0,
                                             2.0 * source_rows / (image_height - 1) - 1.0], dim=-1)
        images = image if image.dim() == 4 else image.unsqueeze(dim=0)
        batch_grid = sampling_grid.expand(images.shape[0], -1, -1, -1)
        warped_images = torch_functional.grid_sample(input=images, grid=batch_grid, mode="bilinear",
                                                     padding_mode="zeros", align_corners=True)
        largest_move = max(math.hypot(moved_x - corner_x, moved_y - corner_y)
                           for (moved_x, moved_y), (corner_x, corner_y) in zip(moved_corners, image_corners))
        return warped_images.view_as(other=image), None, None, 1.0, f"corners moved by up to {largest_move:.1f} pixels"

    def random_jigsaw(self, image: torch.Tensor, bounding_boxes: Optional[torch.Tensor],
                      mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                                                             Optional[torch.Tensor], float, str]:
        """
        Cuts the image into a grid of jigsaw_grid_size x jigsaw_grid_size pieces (2 x 2) and puts them back in a random
        order, as kornia's RandomJigsaw, which breaks the global layout of the image for the datasets whose fixed
        background or framing would let the self-supervised views match without learning their content. The pieces are
        moved by slicing only, without any resampling. No box, mask or keypoint follows the pieces (refuse_targets). A
        sequence receives the same order of pieces on all its frames.

        Args:
            image (torch.Tensor): The image [C, H, W] or sequence [T, C, H, W].
            bounding_boxes (Optional[torch.Tensor]): None, the type moving the image only.
            mask (Optional[torch.Tensor]): None, the type moving the image only.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, str]: The shuffled image, no
                boxes nor mask, the area scale of 1 and the order of the pieces for the augmentation report.

        Raises:
            ValueError: If a side of the image is not a multiple of the grid size.
        """
        self.refuse_targets(augmentation_type="random_jigsaw", bounding_boxes=bounding_boxes, mask=mask)
        grid_size = self.strengths["jigsaw_grid_size"]
        image_height, image_width = image.shape[-2], image.shape[-1]
        if image_height % grid_size != 0 or image_width % grid_size != 0:
            raise ValueError(f"random_jigsaw cuts the image into {grid_size}x{grid_size} pieces, its sides must be "
                             f"multiples of {grid_size}, got {image_height}x{image_width}.")
        piece_height, piece_width = image_height // grid_size, image_width // grid_size

        # Pieces in reading order, shuffled, then put back row by row
        pieces = [image[..., row * piece_height:(row + 1) * piece_height,
                        column * piece_width:(column + 1) * piece_width]
                  for row in range(grid_size) for column in range(grid_size)]
        piece_order = list(range(len(pieces)))
        random.shuffle(piece_order)
        shuffled_rows = [torch.cat(tensors=[pieces[piece_index]
                                            for piece_index in piece_order[row * grid_size:(row + 1) * grid_size]],
                                   dim=-1) for row in range(grid_size)]
        drawn_values = "pieces in the order " + " ".join(str(piece_index) for piece_index in piece_order)
        return torch.cat(tensors=shuffled_rows, dim=-2), None, None, 1.0, drawn_values

    def brightness_contrast_saturation(self, image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Changes the brightness, contrast and saturation to make the network robust to lighting and camera
        differences: slightly in the supervised tasks, a brightness shift of up to 40 intensity levels and the contrast
        and saturation scaled by 0.7 to 1.0; in the self-supervised task, the colour jitter of SimCLR scaling the
        brightness, contrast and saturation by 0.6 to 1.2 each, milder than its 0.2 to 1.8 on ImageNet.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The adjusted BGR image in [0, 255] and the drawn brightness and factors for the
                augmentation report.
        """
        # Brightness scaled by a factor (self-supervised task) or shifted by a number of levels
        if self.strengths["brightness_factor_range"] is not None:
            brightness_factor = random.uniform(*self.strengths["brightness_factor_range"])
            brightened_image = (image * brightness_factor).clamp(min=0.0, max=255.0)
            brightness_description = f"brightness x{brightness_factor:.2f}"
        else:
            brightness_delta = random.uniform(-self.strengths["brightness_maximum_delta"],
                                              self.strengths["brightness_maximum_delta"])
            brightened_image = (image + brightness_delta).clamp(min=0.0, max=255.0)
            brightness_description = f"brightness {brightness_delta:+.1f} levels"

        # Contrast and saturation weight the channels as RGB, so the BGR image is reversed before and after
        contrast_factor = random.uniform(*self.strengths["contrast_factor_range"])
        saturation_factor = random.uniform(*self.strengths["saturation_factor_range"])
        rgb_image = brightened_image.flip(dims=(-3,)) / 255.0
        rgb_image = vision_functional.adjust_contrast(rgb_image, contrast_factor=contrast_factor)
        rgb_image = vision_functional.adjust_saturation(rgb_image, saturation_factor=saturation_factor)
        drawn_values = (f"{brightness_description}, contrast x{contrast_factor:.2f}, "
                        f"saturation x{saturation_factor:.2f}")
        return rgb_image.flip(dims=(-3,)) * 255.0, drawn_values

    def random_hue(self, image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Turns the hue of the image by up to maximum_hue_shift of the hue circle in either direction (18 degrees in the
        supervised tasks, 36 degrees in the self-supervised task, the hue of the colour jitter of SimCLR), which
        changes its colours while keeping their brightness and saturation, by kornia's adjust_hue in the HSV space of
        the image reversed to RGB. The turn is given to kornia as a number filled on the device, kornia copying a
        number of the host to the device, which makes the host wait for the device.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The BGR image in [0, 255] with its hue turned, and the drawn turn in degrees for
                the augmentation report.
        """
        hue_shift = random.uniform(-self.strengths["maximum_hue_shift"], self.strengths["maximum_hue_shift"])
        rgb_image = (image.flip(dims=(-3,)) / 255.0).clamp(min=0.0, max=1.0)
        hue_turn = torch.full(size=(), fill_value=hue_shift * 2.0 * math.pi, device=image.device, dtype=image.dtype)
        turned_image = adjust_hue(image=rgb_image, factor=hue_turn)
        return turned_image.flip(dims=(-3,)) * 255.0, f"hue {hue_shift * 360.0:+.1f} degrees"

    def clahe_augment(self, image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Applies contrast-limited adaptive histogram equalization (clip limit 2.0 on a 4x4 grid of tiles) to every
        channel, enhancing local contrast in dark or washed-out regions, with the values of kornia's equalize_clahe
        computed on the device (equalize_clahe_on_device).

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The equalized BGR image in [0, 255] and its fixed settings for the augmentation
                report.
        """
        clip_limit, grid_size = self.strengths["clahe_clip_limit"], self.strengths["clahe_grid_size"]
        normalized_image = (image / 255.0).clamp(min=0.0, max=1.0)
        drawn_values = f"clip limit {clip_limit}, {grid_size[0]}x{grid_size[1]} tiles"
        return (equalize_clahe_on_device(images=normalized_image, clip_limit=clip_limit, grid_size=grid_size) * 255.0,
                drawn_values)

    def sharp_augment(self, image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Sharpens the image with a Laplacian of Gaussian unsharp mask: the Laplacian of the blurred grayscale image
        highlights edges, and subtracting it (strength 3) makes them crisper.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The sharpened BGR image in [0, 255] and its fixed settings for the augmentation
                report.
        """
        image_height, image_width = image.shape[-2], image.shape[-1]
        sharpening_sigma = self.strengths["sharpening_sigma"]
        sharpening_strength = self.strengths["sharpening_strength"]

        # Grayscale luminance from the BGR channels, blurred to limit the noise amplified by the Laplacian
        luminance_weights, laplacian_kernel = get_sharpening_kernels(dtype=image.dtype, device=image.device)
        grayscale_image = (image * luminance_weights).sum(dim=-3, keepdim=True)
        blurred_image = vision_functional.gaussian_blur(grayscale_image, kernel_size=[5, 5],
                                                        sigma=[sharpening_sigma, sharpening_sigma])

        # Apply the Laplacian kernel on every frame with reflected borders
        padded_image = torch_functional.pad(blurred_image.reshape(-1, 1, image_height, image_width),
                                            pad=(1, 1, 1, 1), mode="reflect")
        laplacian = torch_functional.conv2d(input=padded_image, weight=laplacian_kernel)

        # Subtract the scaled Laplacian to sharpen the edges and report the fixed strength and blur
        sharpened_image = (image - sharpening_strength * laplacian.reshape(grayscale_image.shape)).clamp(min=0.0,
                                                                                                         max=255.0)
        return sharpened_image, f"strength {sharpening_strength}, sigma {sharpening_sigma}"

    @staticmethod
    def equalize_histogram_augment(image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Equalizes the histogram of the luminance (Y channel of YUV) only, which spreads the intensities over the
        full range while keeping the colours, the YUV conversions of kornia being computed with matrices kept on the
        device (convert_colour_space).

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The equalized BGR image in [0, 255] and the equalized channel for the
                augmentation report.
        """
        rgb_to_yuv_matrix, yuv_to_rgb_matrix = get_yuv_matrices(dtype=image.dtype, device=image.device)
        yuv_image = convert_colour_space(images=(image.flip(dims=(-3,)) / 255.0).clamp(min=0.0, max=1.0),
                                         matrix=rgb_to_yuv_matrix)

        # Equalize the luminance channel and keep the two chrominance channels unchanged
        equalized_luminance = vision_functional.equalize(yuv_image[..., 0:1, :, :].clamp(min=0.0, max=1.0))
        yuv_image = torch.cat(tensors=[equalized_luminance, yuv_image[..., 1:, :, :]], dim=-3)

        rgb_image = convert_colour_space(images=yuv_image, matrix=yuv_to_rgb_matrix)
        return (rgb_image.clamp(min=0.0, max=1.0) * 255.0).flip(dims=(-3,)), "luminance"

    @staticmethod
    def equalize_bgr_histogram_augment(image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Equalizes the histogram of one randomly chosen channel (B, G or R) or of all three channels, which changes
        the colour balance of the image.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The equalized BGR image in [0, 255] and the drawn channels for the augmentation
                report.
        """
        channel_name = random.choice(seq=list(BGR_CHANNEL_INDICES))
        channel_indices = BGR_CHANNEL_INDICES[channel_name]

        # Equalize the drawn channels only, the other channels keeping their values
        equalized_image = image.clone()
        normalized_channels = (image[..., channel_indices, :, :] / 255.0).clamp(min=0.0, max=1.0)
        equalized_image[..., channel_indices, :, :] = vision_functional.equalize(normalized_channels) * 255.0
        return equalized_image, f"channels {channel_name}"

    def random_noise(self, image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Adds Gaussian noise with a standard deviation of 20 intensity levels (30 levels in the self-supervised task),
        simulating sensor noise. For a sequence every frame receives its own noise sample, with the same strength, as a
        real camera would.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The noisy image and its strength for the augmentation report.
        """
        standard_deviation = self.strengths["noise_standard_deviation"]
        return (image + torch.randn_like(input=image) * standard_deviation,
                f"standard deviation {standard_deviation:.0f} levels")

    @staticmethod
    def gray_scale_augment(image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Converts the image to grayscale while keeping three channels, so the network does not rely on colour only.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The grayscale image replicated on three channels and no drawn value (empty).
        """
        return vision_functional.rgb_to_grayscale(image.flip(dims=(-3,)), num_output_channels=3), ""

    @staticmethod
    def random_colour_augmentation(image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Swaps the blue and red channels (BGR to RGB), a strong colour change that keeps every structure intact.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The image with its channel order reversed and the swap for the augmentation
                report.
        """
        return image.flip(dims=(-3,)), "blue and red swapped"

    def random_gamma(self, image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Corrects the gamma of the image, a non-linear change of the exposure darkening or lightening the mid-tones
        more than the dark and bright areas: every pixel of the image scaled to [0, 1] is raised to a power drawn
        within gamma_range (0.8 to 1.25 in the supervised and self-supervised tasks, kornia's RandomGamma of the
        views).

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The corrected image in [0, 255] and the drawn gamma for the augmentation report.
        """
        gamma = random.uniform(*self.strengths["gamma_range"])
        return 255.0 * (image / 255.0).clamp(min=0.0, max=1.0) ** gamma, f"gamma {gamma:.2f}"

    def random_linear_illumination(self, image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Lights the image unevenly, a light source on one side of the scene: a ramp growing linearly from 0 on one side
        of the image to a gain drawn within linear_illumination_gain_range on the opposite side (13 to 38 levels in the
        supervised and self-supervised tasks, kornia's RandomLinearIllumination of the views)
        brightens or darkens every pixel, the side towards which the ramp grows being drawn among the four
        (LINEAR_ILLUMINATION_DIRECTIONS). The ramp is computed on the device from numbers. A sequence receives the same
        illumination on all its frames, unless its photometric types are drawn frame by frame.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The lit image in [0, 255] and the drawn gain and side for the augmentation
                report.
        """
        # Gain of the ramp, brightening or darkening with equal chances, and the side it grows towards
        gain = random.uniform(*self.strengths["linear_illumination_gain_range"])
        sign = 1.0 if random.random() < 0.5 else -1.0
        direction = random.choice(LINEAR_ILLUMINATION_DIRECTIONS)

        # Ramp from 0 to 1 along the rows or the columns, reversed when it grows towards the top or the left
        vertical = direction in ["bottom", "top"]
        ramp = torch.linspace(start=0.0, end=1.0, steps=image.shape[-2] if vertical else image.shape[-1],
                              device=image.device, dtype=image.dtype)
        if direction in ["top", "left"]:
            ramp = ramp.flip(dims=(0,))
        ramp = ramp.view(-1, 1) if vertical else ramp.view(1, -1)
        lit_image = (image + sign * gain * ramp).clamp(min=0.0, max=255.0)
        return lit_image, f"{sign * gain:+.0f} levels growing towards the {direction}"

    def random_plasma_shadow(self, image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Casts irregular shadows on the image: the pixels where a random fractal map (build_plasma_map, the plasma of
        kornia's RandomPlasmaShadow of the views), whose detail grows with a roughness drawn within
        plasma_shadow_roughness_range, falls below a quantity drawn within plasma_shadow_quantity_range are darkened by
        an intensity drawn within plasma_shadow_intensity_range (up to 77 levels below a quantity of up to 0.5 in the
        supervised tasks, up to 51 levels below a quantity of up to 0.3 in the self-supervised task). A sequence
        receives the same shadows on all its frames, unless its photometric types are drawn frame by frame.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The shadowed image in [0, 255] and the drawn intensity, quantity and roughness for
                the augmentation report.
        """
        roughness = random.uniform(*self.strengths["plasma_shadow_roughness_range"])
        quantity = random.uniform(*self.strengths["plasma_shadow_quantity_range"])
        intensity = random.uniform(*self.strengths["plasma_shadow_intensity_range"])

        # Fractal map drawn on the device, the pixels below the quantity darkened by the intensity
        plasma_map = build_plasma_map(height=image.shape[-2], width=image.shape[-1], roughness=roughness,
                                      device=image.device)
        shadowed_image = (image + (plasma_map < quantity).to(dtype=image.dtype) * intensity).clamp(min=0.0, max=255.0)
        return shadowed_image, f"intensity {intensity:+.0f} levels, quantity {quantity:.2f}, roughness {roughness:.2f}"

    def random_blur(self, image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Blurs the image with a Gaussian kernel, 9x9 of sigma 2.5 in the supervised and self-supervised tasks,
        simulating defocus and motion softness.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The blurred image and the kernel and sigma for the augmentation report.
        """
        kernel_size = self.strengths["blur_kernel_size"]
        sigma = draw_uniform(value_range=self.strengths["blur_sigma_range"])
        blurred_image = vision_functional.gaussian_blur(image, kernel_size=[kernel_size, kernel_size],
                                                        sigma=[sigma, sigma])
        return blurred_image, f"{kernel_size}x{kernel_size} kernel, sigma {round(sigma, 2):g}"

    def random_motion_blur(self, image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Smears the image along a straight line, the motion of the camera or of the objects during the exposure (a
        conveyor belt, a vehicle, a shaking hand), so that the network learns to recognise objects whose edges are
        streaked in one direction instead of softened in all of them like by random_blur: the line is an odd kernel
        of 5 to 15 pixels (in the supervised and self-supervised tasks) at an angle of 0 to 180 degrees
        (counter-clockwise from the horizontal), the streak trailing backward or forward along it with a direction of
        -1 to 1, 0 spreading it evenly, drawn on the device (build_motion_blur_kernel). The borders are reflected so
        that the edges of the image do not darken. A sequence receives the same blur on all its frames, unless its
        photometric types are drawn frame by frame.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The blurred image and the drawn kernel size, angle and direction for the
                augmentation report.
        """
        # Draw an odd length of the line, its angle and the lean of its weights
        minimum_kernel_size, maximum_kernel_size = self.strengths["motion_blur_kernel_size_range"]
        kernel_size = random.randrange(start=minimum_kernel_size, stop=maximum_kernel_size + 1, step=2)
        angle = random.uniform(*self.strengths["motion_blur_angle_range"])
        direction = random.uniform(*self.strengths["motion_blur_direction_range"])

        # Draw the line on the device, then blur an image as a batch of one and the frames of a sequence as a batch
        # sharing the kernel
        kernel = build_motion_blur_kernel(kernel_size=kernel_size, angle=angle, direction=direction,
                                          device=image.device)
        images = image if image.dim() == 4 else image.unsqueeze(dim=0)
        blurred_images = filter2d(input=images, kernel=kernel, border_type="reflect")
        return (blurred_images.view_as(other=image),
                f"kernel of {kernel_size} pixels, angle {angle:.0f} degrees, direction {direction:+.2f}")

    def random_haze(self, image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Veils the image with a haze, the fog, condensation or dirt on the lens of a camera, so that the network learns
        to find objects whose contrast with their background is low: the image is blended with a uniform grey
        airlight of 180 to 255 intensity levels taking 20% to 60% of every pixel (160 to 255 levels taking 20% to 70%
        in the self-supervised task) (apply_haze). A sequence receives the same haze on all its
        frames, unless its photometric types are drawn frame by frame.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The hazy image in [0, 255] and the drawn strength and airlight for the
                augmentation report.
        """
        strength = random.uniform(*self.strengths["haze_strength_range"])
        airlight = random.uniform(*self.strengths["haze_airlight_range"])
        return (apply_haze(images=image, strength=strength, airlight=airlight),
                f"strength {strength:.2f}, airlight {airlight:.0f} levels")

    def random_solarize(self, image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Solarizes the image, a strong photometric distortion so that the two views of an image cannot be matched by
        their exact intensities (BYOL, Barlow Twins): every pixel at or above solarize_threshold (127.5 levels in the
        supervised tasks, 191.25 levels in the self-supervised task, the bright pixels only) is inverted, as kornia's
        RandomSolarize of the views. Drawn for the second view only in the
        self-supervised task (SECOND_VIEW_ONLY_TYPES), for every image in the supervised tasks. A sequence is
        solarized on all its frames, unless its photometric types are drawn frame by frame.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The solarized image in [0, 255] and its threshold for the augmentation report.
        """
        threshold = self.strengths["solarize_threshold"]
        clamped_image = image.clamp(min=0.0, max=255.0)
        return (torch.where(clamped_image < threshold, clamped_image, 255.0 - clamped_image),
                f"threshold {threshold:g} levels")

    def random_black_out(self, image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Cuts the image into a grid of 'types.random_black_out.grid_size' rows and columns (8 by default, chosen per
        task after the size of the objects to hide) and blacks out every cell with a fixed probability (0.2 in the
        supervised tasks, 0.35 in the self-supervised task), the number of hidden cells varying from image to image
        (Hide-and-Seek), so that the network learns from every region of the image instead of the most telling one
        and copes with occluded parts of the scene. The boxes, masks and keypoints are left unchanged, so an object
        can be hidden while keeping its label: the user chooses the tasks and datasets it suits. Applied after the
        other photometric types but the erasing, so that the cells stay black like the padding of the data loaders,
        and drawn for the second view only in the self-supervised task (SECOND_VIEW_ONLY_TYPES). A sequence has the
        same cells blacked out on all its frames, unless its photometric types are drawn frame by frame.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The image with its hidden cells at 0, and the number of hidden cells for the
                augmentation report.
        """
        # Draw every cell of the grid on its own, then black out the hidden ones on every channel and frame at once
        grid_size = self.black_out_grid_size
        cell_probability = self.strengths["black_out_cell_probability"]
        hidden_cells = torch.tensor(data=[random.random() < cell_probability
                                          for _ in range(grid_size * grid_size)]).view(grid_size, grid_size)
        return (apply_black_out(images=image, hidden_cells=hidden_cells),
                describe_black_out(hidden_cells=hidden_cells))

    def random_erasing(self, image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Erases a random rectangle of the image, an occlusion, filled with black like the padding of the data loaders: a
        rectangle covering a share of the image area within erasing_area_range (2% to 15%) with a height / width ratio
        drawn log-uniformly within erasing_aspect_ratio_range (0.3 to 3.3), as torchvision's and kornia's
        RandomErasing, drawn again up to MAXIMUM_REGION_ATTEMPTS times until it fits in the image, which is left
        unchanged otherwise. The boxes, masks and keypoints are left unchanged, so an object can be hidden while
        keeping its label. Applied last, so that the rectangle stays black. A sequence has the same rectangle erased on
        all its frames, unless its photometric types are drawn frame by frame.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The image with its rectangle at 0, and the rectangle for the augmentation report.
        """
        image_height, image_width = image.shape[-2], image.shape[-1]
        minimum_ratio, maximum_ratio = self.strengths["erasing_aspect_ratio_range"]
        logarithm_ratio_range = (math.log(minimum_ratio), math.log(maximum_ratio))

        # Area and ratio drawn until the rectangle fits, at a random position, the erased copy keeping the input intact
        for attempt_index in range(MAXIMUM_REGION_ATTEMPTS):
            erased_area = image_height * image_width * random.uniform(*self.strengths["erasing_area_range"])
            aspect_ratio = math.exp(random.uniform(*logarithm_ratio_range))
            erased_height = round(math.sqrt(erased_area * aspect_ratio))
            erased_width = round(math.sqrt(erased_area / aspect_ratio))
            if erased_height < image_height and erased_width < image_width:
                top = random.randint(0, image_height - erased_height)
                left = random.randint(0, image_width - erased_width)
                erased_image = image.clone()
                erased_image[..., top:top + erased_height, left:left + erased_width] = 0.0
                return erased_image, f"{erased_width}x{erased_height} pixels at top {top} and left {left} pixels"
        return image, "no rectangle fitting in the image"
