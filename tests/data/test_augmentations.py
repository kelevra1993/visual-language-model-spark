"""
Tests of DataAugmenter of vision_training/data/augmentations.py: geometric augmentations moving and removing boxes,
geometric augmentations moving the segmentation masks with nearest interpolation, geometric augmentations moving the
instance masks and recomputing their boxes, photometric augmentations keeping boxes, masks and pixel range, the motion
blur streaking a dot along the drawn angle with reflected borders, the haze blending every image with its own grey
airlight after the other photometric types, the black-out hiding whole cells of a set grid drawn one by one, last,
without making the host wait for the GPU, shared or per-frame draws of the sequence frames, the augmentation turned off
or of zero probability leaving the batch unchanged, frame shuffling, the zoom-out, the translation by whole pixels, the
removal of the fragments of objects left by the crop, the aspect crop keeping independent shares of the width and of the
height and stretching the image, its boxes and masks, the keypoints moved without being renamed and unlabelled once out
of the image, the ignored regions of the keypoint detection moved with them as binary masks, the augmentation reports
written in the information panels of the input viewer, the perspective warping the image like kornia, the jigsaw putting
its pieces back in the drawn order, the hue turned with the brightness and the saturation kept, the gamma, the ramp of
the linear illumination, the plasma map of kornia and its shadow, the solarization, the erasing of one black rectangle
after the noise, and these types never making the host wait for the GPU, nor the rotation, the CLAHE, the sharpening and
the histogram equalizations, which give the values of kornia and of their former kernels.
"""

import math
import random
import re
import pytest
import torch

from kornia.color import rgb_to_hsv, rgb_to_yuv, yuv_to_rgb
from kornia.contrib import diamond_square
from kornia.enhance import equalize_clahe
from kornia.filters import get_motion_kernel2d
from kornia.geometry.transform import get_perspective_transform, warp_perspective
from torchvision.transforms.v2 import functional as vision_functional
from typing import Any, Dict, List

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.data.augmentations import (DataAugmenter, AUGMENTATION_TYPES, GEOMETRIC_AUGMENTATION_TYPES,
                                                 IMAGE_ONLY_AUGMENTATION_TYPES, PHOTOMETRIC_AUGMENTATION_TYPES,
                                                 SUPERVISED_AUGMENTATION_STRENGTHS, build_motion_blur_kernel,
                                                 build_plasma_map, get_sharpening_kernels, send_grid_to_device)
from vision_training.data.detection.coco_masks import compute_mask_bounding_boxes

IMAGE_SIZE = 96

# Strengths of the supervised tasks whose drawn values the tests check
MOTION_BLUR_KERNEL_SIZE_RANGE = SUPERVISED_AUGMENTATION_STRENGTHS["motion_blur_kernel_size_range"]
MOTION_BLUR_ANGLE_RANGE = SUPERVISED_AUGMENTATION_STRENGTHS["motion_blur_angle_range"]
MOTION_BLUR_DIRECTION_RANGE = SUPERVISED_AUGMENTATION_STRENGTHS["motion_blur_direction_range"]
HAZE_STRENGTH_RANGE = SUPERVISED_AUGMENTATION_STRENGTHS["haze_strength_range"]
HAZE_AIRLIGHT_RANGE = SUPERVISED_AUGMENTATION_STRENGTHS["haze_airlight_range"]
NOISE_STANDARD_DEVIATION = SUPERVISED_AUGMENTATION_STRENGTHS["noise_standard_deviation"]
BLACK_OUT_CELL_PROBABILITY = SUPERVISED_AUGMENTATION_STRENGTHS["black_out_cell_probability"]
MAXIMUM_TRANSLATION_FRACTION = SUPERVISED_AUGMENTATION_STRENGTHS["maximum_translation_fraction"]
MINIMUM_CROP_FRACTION = SUPERVISED_AUGMENTATION_STRENGTHS["crop_side_range"][0]
MINIMUM_ASPECT_CROP_FRACTION = SUPERVISED_AUGMENTATION_STRENGTHS["aspect_crop_side_range"][0]

# Geometric types moving the boxes, masks and keypoints with the image, the others moving the image only
TARGET_MOVING_GEOMETRIC_TYPES = [name for name in GEOMETRIC_AUGMENTATION_TYPES
                                 if name not in IMAGE_ONLY_AUGMENTATION_TYPES]

# Grid size of random_black_out in the configurations of the repository
DEFAULT_BLACK_OUT_GRID_SIZE = 8

# Tolerance in pixels between the bright rectangle and its transformed box: exact pixel permutations must match
# exactly, while resampling (crop resize, small rotation) blurs the rectangle edges by a pixel or two
BOX_TOLERANCES = {"random_left_right_flip": 0.0, "random_up_down_flip": 0.0, "random_90_rotation": 0.0,
                  "random_crop": 2.0, "random_aspect_crop": 2.0, "random_zoom_out": 2.0, "random_translation": 0.0,
                  "random_rotation": 2.5}


def build_configuration(enabled_types: List[str], probability: float = 1.0, independent_frames: bool = False,
                        shuffle_probability: float = 0.0,
                        black_out_grid_size: int = DEFAULT_BLACK_OUT_GRID_SIZE) -> Dict[str, Any]:
    """
    Builds a 'Data.data_augmentation' configuration forcing the given augmentation types, so that each test
    controls exactly which transformations are applied.

    Args:
        enabled_types (List[str]): The augmentation types to activate, each with probability 1.
        probability (float): The global probability of augmenting an image. Defaults to 1.0.
        independent_frames (bool): Whether the photometric types are drawn frame by frame. Defaults to False.
        shuffle_probability (float): The probability of shuffling the frames of a sequence, 0 turning the shuffling
            off. Defaults to 0.0.
        black_out_grid_size (int): The rows and columns of the grid of random_black_out. Defaults to
            DEFAULT_BLACK_OUT_GRID_SIZE.

    Returns:
        Dict[str, Any]: The augmentation configuration, with the sequence settings read only by the augmenter of a
            sequence task.
    """
    types = {name: {"activated": name in enabled_types, "probability": 1.0} for name in AUGMENTATION_TYPES}
    types["random_black_out"]["grid_size"] = black_out_grid_size
    return {"activated": True, "probability": probability, "independent_frames": independent_frames,
            "shuffle_frames": {"activated": shuffle_probability > 0.0, "probability": shuffle_probability},
            "types": types}


def create_rectangle_image(box: List[float]) -> torch.Tensor:
    """
    Creates a black BGR image containing a single white rectangle covering exactly the given box, so that the
    position of the rectangle after a geometric augmentation can be compared with the transformed box.

    Args:
        box (List[float]): The [x1, y1, x2, y2] box of the rectangle, in pixels.

    Returns:
        torch.Tensor: The [3, IMAGE_SIZE, IMAGE_SIZE] image in [0, 255].
    """
    image = torch.zeros(size=(3, IMAGE_SIZE, IMAGE_SIZE))
    x_1, y_1, x_2, y_2 = [int(value) for value in box]
    image[:, y_1:y_2, x_1:x_2] = 255.0
    return image


def get_bright_region_box(image: torch.Tensor) -> List[float]:
    """
    Measures the [x1, y1, x2, y2] extent of the bright pixels of an image, i.e. where the rectangle ended up.

    Args:
        image (torch.Tensor): The [3, H, W] image in [0, 255].

    Returns:
        List[float]: The box enclosing every pixel brighter than half intensity.
    """
    row_indices, column_indices = torch.nonzero(input=image.mean(dim=0) > 127.5, as_tuple=True)
    return [float(column_indices.min()), float(row_indices.min()), float(column_indices.max()) + 1.0,
            float(row_indices.max()) + 1.0]


@pytest.mark.parametrize(argnames="augmentation_type", argvalues=["random_crop", "random_zoom_out",
                                                                  "random_left_right_flip", "random_rotation"])
def test_geometric_augmentation_moves_boxes_with_the_image(augmentation_type: str) -> None:
    """
    Verifies that every geometric augmentation moving the targets transforms the box exactly like the image content: the
    returned box must match the position of the rectangle drawn at the original box, for several random draws.

    Args:
        augmentation_type (str): The geometric augmentation type under test.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=[augmentation_type]))
    box = [20.0, 30.0, 50.0, 70.0]

    for seed in range(5):
        random.seed(seed)
        torch.manual_seed(seed)
        images, boxes, labels, _, _, _, _ = augmenter.augment_batch(
            images=create_rectangle_image(box=box).unsqueeze(dim=0), bounding_boxes=[torch.tensor(data=[box])],
            labels=[torch.tensor(data=[1])])

        # The rectangle stays fully inside the image, so its box and label must be kept
        assert boxes[0].shape == (1, 4) and labels[0].tolist() == [1]
        assert torch.allclose(input=boxes[0][0], other=torch.tensor(data=get_bright_region_box(image=images[0])),
                              atol=BOX_TOLERANCES[augmentation_type])


def test_boxes_leaving_the_image_are_removed_with_their_labels() -> None:
    """
    Verifies that boxes pushed out of the image by a geometric augmentation are removed together with their labels,
    while partially visible boxes are clipped to the image borders.
    """
    boxes = torch.tensor(data=[[-30.0, 10.0, -5.0, 40.0], [80.0, 20.0, 120.0, 60.0], [10.0, 10.0, 40.0, 40.0]])
    labels = torch.tensor(data=[3, 2, 1])

    kept_boxes, kept_labels, _, _, removed_labels = DataAugmenter.clean_bounding_boxes(
        bounding_boxes=boxes, labels=labels, image_height=IMAGE_SIZE, image_width=IMAGE_SIZE)

    assert kept_labels.tolist() == [2, 1] and removed_labels.tolist() == [3]
    assert kept_boxes.tolist() == [[80.0, 20.0, 96.0, 60.0], [10.0, 10.0, 40.0, 40.0]]


@pytest.mark.parametrize(argnames="augmentation_type", argvalues=["brightness_contrast_saturation"])
def test_photometric_augmentation_keeps_boxes_and_range(augmentation_type: str) -> None:
    """
    Verifies that photometric augmentations never move the boxes, keep the image shape and stay within [0, 255].

    Args:
        augmentation_type (str): The photometric augmentation type under test.
    """
    random.seed(0)
    torch.manual_seed(0)
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=[augmentation_type]))
    images = torch.rand(size=(2, 3, IMAGE_SIZE, IMAGE_SIZE)) * 255.0
    boxes = [torch.tensor(data=[[10.0, 20.0, 30.0, 40.0]]), torch.zeros(size=(0, 4))]
    labels = [torch.tensor(data=[1]), torch.zeros(size=(0,), dtype=torch.int64)]

    augmented_images, augmented_boxes, augmented_labels, _, _, _, _ = augmenter.augment_batch(
        images=images, bounding_boxes=boxes, labels=labels)

    assert augmented_images.shape == images.shape
    assert float(augmented_images.min()) >= 0.0 and float(augmented_images.max()) <= 255.0
    assert all(torch.equal(input=output, other=original) for output, original in zip(augmented_boxes, boxes))
    assert all(torch.equal(input=output, other=original) for output, original in zip(augmented_labels, labels))


@pytest.mark.parametrize(argnames="device", argvalues=["cpu"])
def test_clahe_and_luminance_equalization_give_the_values_of_kornia(device: str) -> None:
    """
    Verifies that clahe_augment and equalize_histogram_augment, computed on the device without copy from the host,
    give exactly the values of their former implementation with kornia: the CLAHE of kornia's equalize_clahe on every
    channel, and the equalization of the luminance in kornia's YUV space.

    Args:
        device (str): The device of the images.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["clahe_augment"]))
    images = (torch.rand(size=(2, 3, IMAGE_SIZE, IMAGE_SIZE), generator=torch.Generator().manual_seed(0)) *
              255.0).to(device=device)
    normalized_images = (images / 255.0).clamp(min=0.0, max=1.0)

    # CLAHE with the clip limit and the tiles of the supervised strengths
    expected_clahe_images = equalize_clahe(normalized_images, clip_limit=SUPERVISED_AUGMENTATION_STRENGTHS[
        "clahe_clip_limit"], grid_size=SUPERVISED_AUGMENTATION_STRENGTHS["clahe_grid_size"]) * 255.0
    assert torch.equal(input=augmenter.clahe_augment(image=images)[0], other=expected_clahe_images)

    # Equalization of the luminance, the chrominance kept, in the RGB order of the YUV conversions
    yuv_images = rgb_to_yuv((images.flip(dims=(-3,)) / 255.0).clamp(min=0.0, max=1.0))
    equalized_luminance = vision_functional.equalize(yuv_images[..., 0:1, :, :].clamp(min=0.0, max=1.0))
    expected_equalized_images = (yuv_to_rgb(torch.cat(tensors=[equalized_luminance, yuv_images[..., 1:, :, :]],
                                                      dim=-3)).clamp(min=0.0, max=1.0) * 255.0).flip(dims=(-3,))
    assert torch.equal(input=DataAugmenter.equalize_histogram_augment(image=images)[0], other=expected_equalized_images)


@pytest.mark.parametrize(argnames="dtype", argvalues=[torch.float32])
def test_sharpening_kernels_are_those_of_the_sharpening(dtype: torch.dtype) -> None:
    """
    Verifies that the kernels of sharp_augment, filled on the device once per dtype and device by
    get_sharpening_kernels, hold the luminance weights of the BGR channels and the Laplacian kernel that the type
    copied from the host before, in the dtype of the images.

    Args:
        dtype (torch.dtype): The dtype of the images.
    """
    luminance_weights, laplacian_kernel = get_sharpening_kernels(dtype=dtype, device=torch.device("cpu"))
    assert torch.equal(input=luminance_weights, other=torch.tensor(data=[0.114, 0.587, 0.299],
                                                                     dtype=dtype).view(3, 1, 1))
    assert torch.equal(input=laplacian_kernel, other=torch.tensor(
        data=[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=dtype).view(1, 1, 3, 3))


def test_motion_blur_streaks_the_image_along_the_drawn_angle(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the random_motion_blur type: a dot becomes a straight streak of the drawn odd length along the drawn
    angle, counter-clockwise from the horizontal as the image is displayed, trailing backward with a direction of
    -1, its intensity kept and a uniform image left uniform by the reflected borders; the line drawn on
    the device is the kernel of kornia's get_motion_kernel2d; the kernel sizes, angles and directions are drawn within
    their ranges and given in the report, the frames of a sequence share the same blur, and the motion blur comes after
    random_blur and before random_haze.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture forcing the kernel size, the angle and the direction.
    """
    random.seed(0)
    torch.manual_seed(seed=0)
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["random_motion_blur"]))
    center = IMAGE_SIZE // 2

    # The line drawn from numbers on the device is the kernel kornia builds on the CPU, for any draw
    for _ in range(200):
        kernel_size = random.randrange(start=MOTION_BLUR_KERNEL_SIZE_RANGE[0],
                                       stop=MOTION_BLUR_KERNEL_SIZE_RANGE[1] + 1, step=2)
        angle, direction = random.uniform(*MOTION_BLUR_ANGLE_RANGE), random.uniform(*MOTION_BLUR_DIRECTION_RANGE)
        assert torch.allclose(input=build_motion_blur_kernel(kernel_size=kernel_size, angle=angle, direction=direction,
                                                             device=torch.device("cpu")),
                              other=get_motion_kernel2d(kernel_size=kernel_size, angle=angle, direction=direction),
                              atol=1e-6)

    # Dots in the middle of black images, blurred with their own kernel size, angle and direction
    dot_images = torch.zeros(size=(64, 3, IMAGE_SIZE, IMAGE_SIZE))
    dot_images[..., center, center] = 255.0
    blurred_images, _, _, _, _, _, reports = augmenter.augment_batch(images=dot_images)
    drawn_values = [re.fullmatch(pattern=r"kernel of (\d+) pixels, angle (\d+) degrees, direction ([+-]\d\.\d\d)",
                                 string=report["photometric_augmentations"][0][0][1]) for report in reports]
    assert all(report["photometric_augmentations"][0][0][0] == "random_motion_blur" for report in reports)
    assert all(match is not None for match in drawn_values)
    kernel_sizes = [int(match.group(1)) for match in drawn_values]
    angles = [float(match.group(2)) for match in drawn_values]
    directions = [float(match.group(3)) for match in drawn_values]
    assert set(kernel_sizes) == set(range(MOTION_BLUR_KERNEL_SIZE_RANGE[0], MOTION_BLUR_KERNEL_SIZE_RANGE[1] + 1, 2))
    assert all(MOTION_BLUR_ANGLE_RANGE[0] <= angle <= MOTION_BLUR_ANGLE_RANGE[1] for angle in angles)
    assert all(MOTION_BLUR_DIRECTION_RANGE[0] <= direction <= MOTION_BLUR_DIRECTION_RANGE[1]
               for direction in directions)
    assert len(set(angles)) > 10 and len(set(directions)) > 10
    assert torch.allclose(input=blurred_images.sum(dim=(1, 2, 3)), other=dot_images.sum(dim=(1, 2, 3)), rtol=1e-4)

    # The longest kernel at an angle of 0 and a direction of -1 smears the dot along its row only, toward the left, the
    # weights decreasing from the dot backward to the last pixel, which gets none
    monkeypatch.setattr(target=random, name="randrange", value=lambda start, stop, step: stop - 1)
    monkeypatch.setattr(target=random, name="uniform", value=lambda low, high: low)
    blurred_images, _, _, _, _, _, reports = augmenter.augment_batch(images=dot_images[:1])
    assert reports[0]["photometric_augmentations"][0] == [("random_motion_blur", (
        f"kernel of {MOTION_BLUR_KERNEL_SIZE_RANGE[1]} pixels, angle 0 degrees, direction -1.00"))]
    lit_rows, lit_columns = torch.nonzero(input=blurred_images[0, 0] > 0.0, as_tuple=True)
    half_length = MOTION_BLUR_KERNEL_SIZE_RANGE[1] // 2
    assert set(lit_rows.tolist()) == {center}
    assert sorted(lit_columns.tolist()) == list(range(center - half_length, center + half_length))
    streak = blurred_images[0, 0, center, center - half_length:center + half_length]
    assert bool((streak[1:] < streak[:-1]).all())

    # At 45 degrees the streak rises to the right on the displayed image, rows going down
    drawn_uniform_values = iter([45.0, 0.0])
    monkeypatch.setattr(target=random, name="uniform", value=lambda low, high: next(drawn_uniform_values))
    blurred_images = augmenter.augment_batch(images=dot_images[:1])[0]
    assert float(blurred_images[0, 0, center - 3, center + 3]) > 0.0
    assert float(blurred_images[0, 0, center - 3, center - 3]) == 0.0

    # The reflected borders leave a uniform image uniform instead of darkening its edges
    monkeypatch.setattr(target=random, name="uniform", value=lambda low, high: high)
    uniform_images = torch.full(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE), fill_value=200.0)
    assert torch.allclose(input=augmenter.augment_batch(images=uniform_images)[0], other=uniform_images, atol=1e-3)

    # A sequence of frames of different intensities is blurred with the same kernel on all its frames
    sequences = torch.stack(tensors=[dot_images[0] * (frame_index + 1) / 4
                                     for frame_index in range(4)]).unsqueeze(dim=0)
    blurred_sequences = augmenter.augment_batch(images=sequences)[0]
    assert all(torch.allclose(input=blurred_sequences[0, frame_index] / (frame_index + 1),
                              other=blurred_sequences[0, 0], atol=1e-4) for frame_index in range(1, 4))

    # Drawn with the Gaussian blur and the haze, the motion blur comes between them
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(
        enabled_types=["random_blur", "random_motion_blur", "random_haze"]))
    reports = augmenter.augment_batch(images=dot_images[:1])[-1]
    assert [type_name for type_name, _ in reports[0]["photometric_augmentations"][0]] == [
        "random_blur", "random_motion_blur", "random_haze"]


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="The training batches are augmented on the GPU.")
def test_motion_blur_on_the_gpu_never_makes_the_host_wait() -> None:
    """
    Verifies the random_motion_blur type on the GPU, where the training batches are augmented: its line kernel being
    drawn on the device, blurring an image and a sequence never makes the host wait for the device, which the
    synchronization debug mode of PyTorch turns into an error, and the same draws give the blur of the CPU.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["random_motion_blur"]))
    images = torch.rand(size=(2, 3, IMAGE_SIZE, IMAGE_SIZE)) * 255.0
    sequences = torch.rand(size=(1, 3, 3, IMAGE_SIZE, IMAGE_SIZE)) * 255.0
    device_images, device_sequences = images.to(device="cuda"), sequences.to(device="cuda")
    augmenter.augment_batch(images=device_images)
    torch.cuda.synchronize()

    # Blur on the GPU with every synchronization of the host turned into an error
    random.seed(3)
    torch.cuda.set_sync_debug_mode(debug_mode="error")
    try:
        gpu_images, _, _, _, _, _, gpu_reports = augmenter.augment_batch(images=device_images)
        gpu_sequences = augmenter.augment_batch(images=device_sequences)[0]
    finally:
        torch.cuda.set_sync_debug_mode(debug_mode="default")

    # The same draws on the CPU give the same blur and the same reports
    random.seed(3)
    cpu_images, _, _, _, _, _, cpu_reports = augmenter.augment_batch(images=images)
    cpu_sequences = augmenter.augment_batch(images=sequences)[0]
    assert gpu_reports == cpu_reports
    assert torch.allclose(input=gpu_images.cpu(), other=cpu_images, atol=1e-3)
    assert torch.allclose(input=gpu_sequences.cpu(), other=cpu_sequences, atol=1e-3)


def test_haze_blends_every_image_with_a_uniform_grey_airlight(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the random_haze type: every image is blended with its own uniform grey airlight, whose strength and
    intensity are drawn within HAZE_STRENGTH_RANGE and HAZE_AIRLIGHT_RANGE and given in the report, and the haze is
    applied after the other photometric types, so that the noise drawn with it is veiled too.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture forcing the largest strength and airlight.
    """
    random.seed(0)
    torch.manual_seed(seed=0)

    # Images half black and half white, from which the drawn strength and airlight of every image are measured back
    images = torch.zeros(size=(32, 3, IMAGE_SIZE, IMAGE_SIZE))
    images[..., IMAGE_SIZE // 2:] = 255.0
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["random_haze"]))
    hazy_images, _, _, _, _, _, reports = augmenter.augment_batch(images=images)

    # The black pixels become strength x airlight and the white ones 255 x (1 - strength) + strength x airlight
    black_values, white_values = hazy_images[..., 0], hazy_images[..., -1]
    assert torch.equal(input=black_values, other=black_values[:, :1, :1].expand_as(black_values))
    strengths = 1.0 - (white_values[:, 0, 0] - black_values[:, 0, 0]) / 255.0
    airlights = black_values[:, 0, 0] / strengths
    assert float(strengths.min()) >= HAZE_STRENGTH_RANGE[0] - 1e-4
    assert float(strengths.max()) <= HAZE_STRENGTH_RANGE[1] + 1e-4
    assert float(airlights.min()) >= HAZE_AIRLIGHT_RANGE[0] - 1e-2
    assert float(airlights.max()) <= HAZE_AIRLIGHT_RANGE[1] + 1e-2
    assert float(strengths.std()) > 0.0 and float(airlights.std()) > 0.0

    # The report gives the values measured on the image
    type_name, drawn_values = reports[0]["photometric_augmentations"][0][0]
    assert type_name == "random_haze"
    assert drawn_values == f"strength {float(strengths[0]):.2f}, airlight {float(airlights[0]):.0f} levels"

    # With the largest strength and airlight, the noise drawn with the haze comes first and is veiled by it: its
    # standard deviation, NOISE_STANDARD_DEVIATION, shrinks by 1 - strength around the hazy grey of the image
    monkeypatch.setattr(target=random, name="uniform", value=lambda low, high: high)
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(
        enabled_types=["random_noise", "random_haze"]))
    hazy_images, _, _, _, _, _, reports = augmenter.augment_batch(images=torch.full(
        size=(1, 3, IMAGE_SIZE, IMAGE_SIZE), fill_value=100.0))
    assert [type_name for type_name, _ in reports[0]["photometric_augmentations"][0]] == ["random_noise",
                                                                                          "random_haze"]
    assert reports[0]["photometric_augmentations"][0][1][1] == (
        f"strength {HAZE_STRENGTH_RANGE[1]:.2f}, airlight {HAZE_AIRLIGHT_RANGE[1]:.0f} levels")
    largest_strength = HAZE_STRENGTH_RANGE[1]
    expected_mean = 100.0 * (1.0 - largest_strength) + largest_strength * HAZE_AIRLIGHT_RANGE[1]
    assert abs(float(hazy_images.mean()) - expected_mean) < 1.0
    assert abs(float(hazy_images.std()) - NOISE_STANDARD_DEVIATION * (1.0 - largest_strength)) < 1.0


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="The training batches are augmented on the GPU.")
def test_black_out_on_the_gpu_never_makes_the_host_wait() -> None:
    """
    Verifies the random_black_out type on the GPU, where the training batches are augmented: its grid, drawn on the
    host, reaches the device as numbers, so that blacking out images and a sequence on a grid whose size does not
    divide the image never makes the host wait for the device, which the synchronization debug mode of PyTorch turns
    into an error, and the same draws give the black-out and the reports of the CPU. The grids of one or several
    packed integers, the bit of their sign included, come back unchanged on the device.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["random_black_out"],
                                                                             black_out_grid_size=7))
    images = torch.rand(size=(4, 3, IMAGE_SIZE, IMAGE_SIZE)) * 255.0
    sequences = torch.rand(size=(1, 3, 3, IMAGE_SIZE, IMAGE_SIZE)) * 255.0
    device_images, device_sequences = images.to(device="cuda"), sequences.to(device="cuda")
    augmenter.augment_batch(images=device_images)
    torch.cuda.synchronize()

    # Black-out on the GPU with every synchronization of the host turned into an error
    random.seed(5)
    torch.cuda.set_sync_debug_mode(debug_mode="error")
    try:
        gpu_images, _, _, _, _, _, gpu_reports = augmenter.augment_batch(images=device_images)
        gpu_sequences = augmenter.augment_batch(images=device_sequences)[0]
    finally:
        torch.cuda.set_sync_debug_mode(debug_mode="default")

    # The same draws on the CPU give the same black-out and the same reports
    random.seed(5)
    cpu_images, _, _, _, _, _, cpu_reports = augmenter.augment_batch(images=images)
    cpu_sequences = augmenter.augment_batch(images=sequences)[0]
    assert gpu_reports == cpu_reports
    assert torch.equal(input=gpu_images.cpu(), other=cpu_images)
    assert torch.equal(input=gpu_sequences.cpu(), other=cpu_sequences)

    # Grids of one integer with every cell hidden (sign bit set), of two integers, and one grid per image of a batch
    for hidden_cells in [torch.ones(size=(8, 8), dtype=torch.bool), torch.rand(size=(10, 10)) < 0.5,
                         torch.rand(size=(3, 1, 8, 8)) < 0.5]:
        assert torch.equal(input=send_grid_to_device(hidden_cells=hidden_cells, device=torch.device("cuda")).cpu(),
                           other=hidden_cells)


def test_black_out_hides_whole_cells_of_the_configured_grid_drawn_one_by_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the random_black_out type: it hides whole cells of the grid of 'grid_size' rows and columns (4x4 here,
    8x8 by default), each drawn on its own with BLACK_OUT_CELL_PROBABILITY so that their number varies between images
    and is given in the report, covers the whole image when a side is not a multiple of the grid size, hides the same
    cells on every frame of a sequence, and keeps its cells black after the haze drawn with it.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture drawing every cell as hidden.
    """
    random.seed(0)
    torch.manual_seed(seed=0)
    grid_size = 4
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["random_black_out"],
                                                                             black_out_grid_size=grid_size))

    # Grey images of 96 pixels, cut into cells of 24 pixels: every cell is either hidden or left untouched
    images = torch.full(size=(64, 3, IMAGE_SIZE, IMAGE_SIZE), fill_value=200.0)
    hidden_images, _, _, _, _, _, reports = augmenter.augment_batch(images=images)
    cell_size = IMAGE_SIZE // grid_size
    cells = hidden_images.reshape(64, 3, grid_size, cell_size, grid_size, cell_size)
    hidden_cells = (cells == 0.0).all(dim=5).all(dim=3).all(dim=1)
    untouched_cells = (cells == 200.0).all(dim=5).all(dim=3).all(dim=1)
    assert bool((hidden_cells | untouched_cells).all())

    # The number of hidden cells varies between images, is given in the report, and their share is the probability
    hidden_counts = hidden_cells.flatten(start_dim=1).sum(dim=1).tolist()
    assert [report["photometric_augmentations"][0] for report in reports] == [
        [("random_black_out", f"{count} of the 16 cells of the 4x4 grid")] for count in hidden_counts]
    assert len(set(hidden_counts)) > 1
    assert abs(float(hidden_cells.float().mean()) - BLACK_OUT_CELL_PROBABILITY) < 0.04

    # A sequence of different frames has the same cells hidden on all its frames
    sequences = torch.stack(tensors=[torch.full(size=(3, IMAGE_SIZE, IMAGE_SIZE), fill_value=50.0 * (frame_index + 1))
                                     for frame_index in range(4)]).unsqueeze(dim=0)
    hidden_sequences = augmenter.augment_batch(images=sequences)[0]
    hidden_pixels = hidden_sequences[0] == 0.0
    assert bool(hidden_pixels.any()) and all(torch.equal(input=hidden_pixels[frame_index], other=hidden_pixels[0])
                                             for frame_index in range(1, 4))

    # Drawn after the haze, the hidden cells stay black while the other pixels are veiled
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(
        enabled_types=["random_haze", "random_black_out"]))
    hazy_images, _, _, _, _, _, reports = augmenter.augment_batch(images=images[:8])
    assert all([type_name for type_name, _ in report["photometric_augmentations"][0]] == ["random_haze",
                                                                                          "random_black_out"]
               for report in reports)
    hidden_pixels = hazy_images == 0.0
    assert bool(hidden_pixels.any()) and bool((hazy_images[~hidden_pixels] != 200.0).all())

    # With every cell of the default grid drawn as hidden, the cells of rounded edges cover a whole image of 100 x 75
    # pixels
    monkeypatch.setattr(target=random, name="random", value=lambda: 0.0)
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["random_black_out"]))
    hidden_images, _, _, _, _, _, reports = augmenter.augment_batch(images=torch.full(size=(1, 3, 100, 75),
                                                                                        fill_value=200.0))
    assert bool((hidden_images == 0.0).all())
    assert reports[0]["photometric_augmentations"] == [[("random_black_out", "64 of the 64 cells of the 8x8 grid")]]


def test_sequence_frames_share_the_same_augmentation() -> None:
    """
    Verifies that every frame of a sequence [B, T, C, H, W] receives exactly the same augmentation: a sequence of
    identical frames must stay identical after augmentation, while its content does change. Random noise is left
    out since each frame gets its own noise sample, with the same strength.
    """
    random.seed(0)
    torch.manual_seed(0)
    enabled_types = [name for name in AUGMENTATION_TYPES if name != "random_noise"]
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=enabled_types))
    frame = torch.rand(size=(3, IMAGE_SIZE, IMAGE_SIZE)) * 255.0
    sequences = frame.expand(2, 4, 3, IMAGE_SIZE, IMAGE_SIZE).clone()

    augmented_sequences, _, _, _, _, _, _ = augmenter.augment_batch(images=sequences)

    assert augmented_sequences.shape == sequences.shape
    assert not torch.allclose(input=augmented_sequences, other=sequences)
    for sequence in augmented_sequences:
        assert all(torch.allclose(input=sequence_frame, other=sequence[0], atol=1e-4) for sequence_frame in sequence)


def test_disabled_augmentation_is_identity() -> None:
    """
    Verifies that nothing changes when augmentation is deactivated globally or its probability is zero.
    """
    images = torch.rand(size=(2, 3, IMAGE_SIZE, IMAGE_SIZE)) * 255.0
    boxes = [torch.tensor(data=[[10.0, 20.0, 30.0, 40.0]]), torch.zeros(size=(0, 4))]
    labels = [torch.tensor(data=[1]), torch.zeros(size=(0,), dtype=torch.int64)]

    deactivated_configuration = build_configuration(enabled_types=AUGMENTATION_TYPES)
    deactivated_configuration["activated"] = False
    zero_probability_configuration = build_configuration(enabled_types=AUGMENTATION_TYPES, probability=0.0)

    for configuration in [deactivated_configuration, zero_probability_configuration]:
        augmenter = DataAugmenter(augmentation_configuration=configuration)
        augmented_images, augmented_boxes, _, _, _, _, _ = augmenter.augment_batch(images=images, bounding_boxes=boxes,
                                                                                   labels=labels)
        assert torch.equal(input=augmented_images, other=images)
        assert all(torch.equal(input=output, other=original) for output, original in zip(augmented_boxes, boxes))


def test_shuffling_permutes_the_real_frames_of_every_sequence() -> None:
    """
    Verifies the shuffling of the frames of the sequences: only the real frames move, the padding frames staying last,
    every member of a pair is shuffled within its own block, the frame order gives the original frame at every
    position, and nothing moves without shuffling.
    """
    torch.manual_seed(seed=0)
    random.seed(0)

    # Two pairs of members padded to 4 frames, the frame of position i filled with the value i
    frames = torch.arange(8, dtype=torch.float32)[None, :, None, None, None].expand(2, 8, 3, 4, 4).clone()
    sequence_mask = torch.tensor(data=[[True, True, True, False, True, True, True, True],
                                       [True, True, False, False, True, True, True, False]])
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=[], shuffle_probability=1.0),
                              sequence_augmentation=True)
    shuffled_frames, frame_orders = augmenter.shuffle_frames(images=frames, sequence_mask=sequence_mask,
                                                             frame_block_length=4)

    for sample_index in range(2):
        shuffled_values = shuffled_frames[sample_index, :, 0, 0, 0].tolist()
        assert shuffled_values == [float(position) for position in frame_orders[sample_index].tolist()]
        for block_start in [0, 4]:
            block_mask = sequence_mask[sample_index, block_start:block_start + 4]
            number_real_frames = int(block_mask.sum())
            real_values = shuffled_values[block_start:block_start + number_real_frames]
            assert sorted(real_values) == [float(block_start + index) for index in range(number_real_frames)]
            assert shuffled_values[block_start + number_real_frames:block_start + 4] == [
                float(block_start + index) for index in range(number_real_frames, 4)]
    assert not torch.equal(input=shuffled_frames, other=frames)

    # Without shuffling, every frame keeps its position
    unshuffled_frames, identity_orders = DataAugmenter(augmentation_configuration=build_configuration(
        enabled_types=[]), sequence_augmentation=True).shuffle_frames(images=frames, sequence_mask=sequence_mask,
                                                                       frame_block_length=4)
    assert torch.equal(input=unshuffled_frames, other=frames)
    assert identity_orders.tolist() == [list(range(8))] * 2


def test_independent_frames_draw_the_photometric_types_frame_by_frame() -> None:
    """
    Verifies the per-frame draws of a sequence: with 'independent_frames', the frames of a sequence of identical
    frames get different brightness changes while sharing the same flip; without it, they stay identical.
    """
    random.seed(1)
    torch.manual_seed(seed=1)
    frame = torch.zeros(size=(3, IMAGE_SIZE, IMAGE_SIZE))
    frame[:, :, :IMAGE_SIZE // 2] = 100.0
    sequences = frame.expand(1, 4, 3, IMAGE_SIZE, IMAGE_SIZE).clone()

    # Independent brightness draws on top of a shared flip, the bright half moving to the right in every frame
    independent_augmenter = DataAugmenter(augmentation_configuration=build_configuration(
        enabled_types=["random_left_right_flip", "brightness_contrast_saturation"], independent_frames=True),
        sequence_augmentation=True)
    augmented_sequence = independent_augmenter.augment_batch(images=sequences)[0][0]
    right_half_means = augmented_sequence[:, :, :, IMAGE_SIZE // 2:].mean(dim=(1, 2, 3))
    left_half_means = augmented_sequence[:, :, :, :IMAGE_SIZE // 2].mean(dim=(1, 2, 3))
    assert bool((right_half_means > left_half_means).all())
    assert len({round(float(mean_value), 3) for mean_value in right_half_means}) > 1

    # Shared draws keep the frames identical
    shared_augmenter = DataAugmenter(augmentation_configuration=build_configuration(
        enabled_types=["random_left_right_flip", "brightness_contrast_saturation"]), sequence_augmentation=True)
    shared_sequence = shared_augmenter.augment_batch(images=sequences)[0][0]
    assert all(torch.allclose(input=sequence_frame, other=shared_sequence[0]) for sequence_frame in shared_sequence)


@pytest.mark.parametrize(argnames="augmentation_type", argvalues=["random_crop", "random_rotation"])
def test_geometric_augmentation_moves_masks_with_the_image(augmentation_type: str) -> None:
    """
    Verifies that every geometric augmentation moving the targets moves the segmentation mask exactly like the image:
    the pixels of the class of a bright rectangle stay on the rectangle (exactly for the pixel permutations, within the
    resampling tolerance otherwise), and the mask only holds class indices or the ignored index, never interpolated
    values.

    Args:
        augmentation_type (str): The geometric augmentation type under test.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=[augmentation_type]))
    box = [20.0, 30.0, 50.0, 70.0]
    mask = torch.zeros(size=(IMAGE_SIZE, IMAGE_SIZE), dtype=torch.int64)
    mask[30:70, 20:50] = 3

    for seed in range(5):
        random.seed(seed)
        torch.manual_seed(seed)
        images, _, _, masks, _, _, _ = augmenter.augment_batch(images=create_rectangle_image(box=box).unsqueeze(dim=0),
                                                               masks=mask.unsqueeze(dim=0))

        # Only class indices or the ignored index, the class region following the rectangle
        assert masks.shape == (1, IMAGE_SIZE, IMAGE_SIZE) and masks.dtype == torch.int64
        assert set(masks.unique().tolist()) <= {0, 3, IGNORED_LABEL_INDEX}
        row_indices, column_indices = torch.nonzero(input=masks[0] == 3, as_tuple=True)
        mask_box = [float(column_indices.min()), float(row_indices.min()), float(column_indices.max()) + 1.0,
                    float(row_indices.max()) + 1.0]
        assert torch.allclose(input=torch.tensor(data=mask_box),
                              other=torch.tensor(data=get_bright_region_box(image=images[0])),
                              atol=BOX_TOLERANCES[augmentation_type])


def test_rotation_ignores_the_uncovered_corners_of_the_mask() -> None:
    """
    Verifies that the small rotation fills the corners it uncovers with the ignored index, like the padding of the
    resized masks, instead of assigning them a class.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["random_rotation"]))
    mask = torch.ones(size=(1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.int64)

    random.seed(0)
    _, _, _, rotated_masks, _, _, _ = augmenter.augment_batch(images=torch.zeros(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE)),
                                                              masks=mask)

    assert set(rotated_masks.unique().tolist()) == {1, IGNORED_LABEL_INDEX}
    assert int(rotated_masks[0, 0, 0]) == IGNORED_LABEL_INDEX and int(rotated_masks[0, IMAGE_SIZE // 2,
                                                                                     IMAGE_SIZE // 2]) == 1


def test_photometric_augmentations_leave_masks_unchanged() -> None:
    """
    Verifies that the photometric augmentations, which only change the pixel values of the images, never change the
    segmentation masks.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(
        enabled_types=PHOTOMETRIC_AUGMENTATION_TYPES))
    images = torch.rand(size=(2, 3, IMAGE_SIZE, IMAGE_SIZE)) * 255.0
    masks = torch.randint(low=0, high=4, size=(2, IMAGE_SIZE, IMAGE_SIZE))

    augmented_images, _, _, augmented_masks, _, _, _ = augmenter.augment_batch(images=images, masks=masks)

    assert not torch.equal(input=augmented_images, other=images)
    assert torch.equal(input=augmented_masks, other=masks)


@pytest.mark.parametrize(argnames="augmentation_type", argvalues=["random_crop", "random_rotation"])
def test_geometric_augmentation_moves_instance_masks_and_recomputes_boxes(augmentation_type: str) -> None:
    """
    Verifies that every geometric augmentation moving the targets moves the binary instance masks with the image (the
    mask following the rectangle drawn at its object), that they stay binary uint8, and that the box of every object
    becomes the tight box of its moved mask.

    Args:
        augmentation_type (str): The geometric augmentation type under test.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=[augmentation_type]))
    box = [20.0, 30.0, 50.0, 70.0]
    instance_masks = torch.zeros(size=(2, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
    instance_masks[0, 30:70, 20:50] = 1
    instance_masks[1, 40:56, 60:80] = 1

    for seed in range(5):
        random.seed(seed)
        torch.manual_seed(seed)
        images, boxes, labels, masks, moved_masks, _, _ = augmenter.augment_batch(
            images=create_rectangle_image(box=box).unsqueeze(dim=0),
            bounding_boxes=[torch.tensor(data=[box, [60.0, 40.0, 80.0, 56.0]])], labels=[torch.tensor(data=[1, 2])],
            instance_masks=[instance_masks])

        # Both objects stay inside the image, binary, the first one on the rectangle
        assert masks is None and labels[0].tolist() == [1, 2]
        assert moved_masks[0].shape == (2, IMAGE_SIZE, IMAGE_SIZE) and moved_masks[0].dtype == torch.uint8
        assert set(moved_masks[0].unique().tolist()) <= {0, 1}
        row_indices, column_indices = torch.nonzero(input=moved_masks[0][0], as_tuple=True)
        mask_box = [float(column_indices.min()), float(row_indices.min()), float(column_indices.max()) + 1.0,
                    float(row_indices.max()) + 1.0]
        assert torch.allclose(input=torch.tensor(data=mask_box),
                              other=torch.tensor(data=get_bright_region_box(image=images[0])),
                              atol=BOX_TOLERANCES[augmentation_type])

        # Every box is the tight box of its moved mask
        assert torch.equal(input=boxes[0], other=compute_mask_bounding_boxes(instance_masks=moved_masks[0]))


def test_rotation_leaves_the_uncovered_corners_of_instance_masks_as_background() -> None:
    """
    Verifies that the small rotation fills the corners it uncovers with 0 in the binary instance masks, instead of the
    ignored index of the class masks.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["random_rotation"]))
    instance_masks = torch.ones(size=(1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)

    random.seed(0)
    _, boxes, _, _, rotated_masks, _, _ = augmenter.augment_batch(
        images=torch.zeros(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE)),
        bounding_boxes=[torch.tensor(data=[[0.0, 0.0, float(IMAGE_SIZE), float(IMAGE_SIZE)]])],
        labels=[torch.tensor(data=[1])], instance_masks=[instance_masks])

    assert set(rotated_masks[0].unique().tolist()) == {0, 1}
    assert int(rotated_masks[0][0, 0, 0]) == 0 and int(rotated_masks[0][0, IMAGE_SIZE // 2, IMAGE_SIZE // 2]) == 1
    assert boxes[0].shape == (1, 4)


def test_instance_masks_of_removed_objects_and_images_without_objects() -> None:
    """
    Verifies that an object whose mask left the image is removed with its box, label and mask, that every geometric
    augmentation moving the targets keeps an image without objects at [0, H, W] masks, and that instance masks without
    boxes or with class masks are refused.
    """
    # An empty mask gives an empty box, removed together with its label and mask
    instance_masks = torch.zeros(size=(2, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
    instance_masks[1, 5:9, 10:20] = 1
    mask_boxes = compute_mask_bounding_boxes(instance_masks=instance_masks)
    assert mask_boxes.tolist() == [[0.0, 0.0, 0.0, 0.0], [10.0, 5.0, 20.0, 9.0]]
    kept_boxes, kept_labels, kept_masks, _, _ = DataAugmenter.clean_bounding_boxes(
        bounding_boxes=mask_boxes, labels=torch.tensor(data=[3, 4]), image_height=IMAGE_SIZE,
        image_width=IMAGE_SIZE, instance_masks=instance_masks)
    assert kept_labels.tolist() == [4] and torch.equal(input=kept_masks, other=instance_masks[1:])

    # An image without objects goes through every geometric augmentation
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(
        enabled_types=TARGET_MOVING_GEOMETRIC_TYPES))
    _, boxes, labels, _, moved_masks, _, _ = augmenter.augment_batch(
        images=torch.zeros(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE)), bounding_boxes=[torch.zeros(size=(0, 4))],
        labels=[torch.zeros(size=(0,), dtype=torch.int64)],
        instance_masks=[torch.zeros(size=(0, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)])
    assert moved_masks[0].shape == (0, IMAGE_SIZE, IMAGE_SIZE) and boxes[0].shape == (0, 4)

    # The instance masks need their boxes, and never come with class masks
    images = torch.zeros(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    with pytest.raises(ValueError, match="Instance masks must be given with their bounding boxes"):
        augmenter.augment_batch(images=images, instance_masks=[instance_masks])
    with pytest.raises(ValueError, match="Instance masks must be given with their bounding boxes"):
        augmenter.augment_batch(images=images, bounding_boxes=[mask_boxes], labels=[torch.tensor(data=[3, 4])],
                                masks=torch.zeros(size=(1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.int64),
                                instance_masks=[instance_masks])


def test_zoom_out_shrinks_the_image_boxes_and_masks_onto_a_black_canvas() -> None:
    """
    Verifies that the zoom-out shrinks the rectangle and its box together onto a black canvas, gives the squared shrink
    factor as area scale, ignores the uncovered border of a class mask and leaves it as background in the instance
    masks, for several random draws.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=[]))
    box = [20.0, 30.0, 50.0, 70.0]
    class_mask = torch.ones(size=(IMAGE_SIZE, IMAGE_SIZE), dtype=torch.int64)
    instance_masks = torch.ones(size=(1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)

    for seed in range(5):
        random.seed(seed)
        image, boxes, mask, area_scale, _ = augmenter.random_zoom_out(
            image=create_rectangle_image(box=box), bounding_boxes=torch.tensor(data=[box]), mask=class_mask)

        # The box follows the shrunk rectangle, and its area shrinks by the area scale
        assert torch.allclose(input=boxes[0], other=torch.tensor(data=get_bright_region_box(image=image)), atol=2.0)
        box_area = float((boxes[0, 2] - boxes[0, 0]) * (boxes[0, 3] - boxes[0, 1]))
        assert 0.25 <= area_scale <= 1.0 and box_area == pytest.approx(expected=30.0 * 40.0 * area_scale, rel=0.05)

        # The class mask keeps its class on the shrunk image and is ignored on the canvas around it
        assert int((mask == 1).sum()) == pytest.approx(expected=IMAGE_SIZE * IMAGE_SIZE * area_scale, rel=1e-6)
        assert set(mask.unique().tolist()) <= {1, IGNORED_LABEL_INDEX}

        # The instance masks stay binary, background around the shrunk image
        random.seed(seed)
        _, _, moved_masks, _, _ = augmenter.random_zoom_out(
            image=create_rectangle_image(box=box), bounding_boxes=torch.tensor(data=[box]), mask=instance_masks)
        assert moved_masks.dtype == torch.uint8 and set(moved_masks.unique().tolist()) <= {0, 1}
        assert int(moved_masks.sum()) == int((mask == 1).sum())


def test_translation_shifts_the_image_boxes_and_masks_by_whole_pixels() -> None:
    """
    Verifies that the translation moves the rectangle, its box, its class mask and its instance mask by the same whole
    number of pixels, at most MAXIMUM_TRANSLATION_FRACTION of each side, that the uncovered border is black in the
    image, ignored in the class mask and background in the instance masks, and that the draws cover the three axes
    (width only, height only and both).
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=[]))
    box = [30.0, 30.0, 60.0, 66.0]
    class_mask = torch.ones(size=(IMAGE_SIZE, IMAGE_SIZE), dtype=torch.int64)
    instance_masks = torch.zeros(size=(1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
    instance_masks[0, 30:66, 30:60] = 1
    maximum_shift = int(IMAGE_SIZE * MAXIMUM_TRANSLATION_FRACTION)
    observed_axes = set()

    for seed in range(30):
        random.seed(seed)
        image, boxes, mask, area_scale, _ = augmenter.random_translation(
            image=create_rectangle_image(box=box), bounding_boxes=torch.tensor(data=[box]), mask=class_mask)

        # The box moves by whole pixels within the limits, the sizes being kept
        shift_x, shift_y = float(boxes[0, 0] - box[0]), float(boxes[0, 1] - box[1])
        assert shift_x.is_integer() and shift_y.is_integer() and area_scale == 1.0
        assert abs(shift_x) <= maximum_shift and abs(shift_y) <= maximum_shift
        assert torch.equal(input=boxes[0], other=torch.tensor(data=box) + torch.tensor(data=[shift_x, shift_y] * 2))
        observed_axes.add((shift_x != 0.0, shift_y != 0.0))

        # The rectangle follows its box exactly, the rectangle staying inside the image at these limits
        assert get_bright_region_box(image=image) == boxes[0].tolist()

        # The class mask is ignored exactly on the uncovered border
        uncovered_pixels = IMAGE_SIZE * IMAGE_SIZE - (IMAGE_SIZE - abs(shift_x)) * (IMAGE_SIZE - abs(shift_y))
        assert int((mask == IGNORED_LABEL_INDEX).sum()) == uncovered_pixels
        assert set(mask.unique().tolist()) <= {1, IGNORED_LABEL_INDEX}

        # The instance mask moves with the same draw, stays binary and keeps all the pixels of its object
        random.seed(seed)
        _, _, moved_masks, _, _ = augmenter.random_translation(
            image=create_rectangle_image(box=box), bounding_boxes=torch.tensor(data=[box]), mask=instance_masks)
        assert moved_masks.dtype == torch.uint8 and set(moved_masks.unique().tolist()) <= {0, 1}
        assert torch.equal(input=compute_mask_bounding_boxes(instance_masks=moved_masks), other=boxes)
        assert int(moved_masks.sum()) == int(instance_masks.sum())

    # The draws cover the width only, the height only and both axes
    assert {(True, False), (False, True), (True, True)} <= observed_axes


def test_crop_removes_the_fragments_of_boxes_and_instance_masks(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that a crop keeping the top left 75% of each side removes the object left with 4 of its 28 columns inside
    the crop, from the boxes of the object detection and from the masks of the instance segmentation, and keeps the
    object fully inside the crop.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixes the crop fraction at 0.75 and its position at the top left corner.
    """
    monkeypatch.setattr(target=random, name="uniform", value=lambda low, high: MINIMUM_CROP_FRACTION)
    monkeypatch.setattr(target=random, name="randint", value=lambda low, high: 0)
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["random_crop"]))
    boxes = torch.tensor(data=[[68.0, 10.0, 96.0, 40.0], [10.0, 10.0, 40.0, 40.0]])
    instance_masks = torch.zeros(size=(2, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
    instance_masks[0, 10:40, 68:96] = 1
    instance_masks[1, 10:40, 10:40] = 1

    # Object detection: the box of the fragment keeps 1/7 of its area inside the crop
    _, kept_boxes, kept_labels, _, _, _, _ = augmenter.augment_batch(
        images=torch.zeros(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE)), bounding_boxes=[boxes],
        labels=[torch.tensor(data=[2, 1])])
    assert kept_labels[0].tolist() == [1] and kept_boxes[0].shape == (1, 4)

    # Instance segmentation: the mask of the fragment keeps 1/7 of the pixels it would have at the zoom of the crop
    _, kept_boxes, kept_labels, _, kept_masks, _, _ = augmenter.augment_batch(
        images=torch.zeros(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE)), bounding_boxes=[boxes],
        labels=[torch.tensor(data=[2, 1])], instance_masks=[instance_masks])
    assert kept_labels[0].tolist() == [1] and kept_masks[0].shape == (1, IMAGE_SIZE, IMAGE_SIZE)


def test_aspect_crop_keeps_independent_shares_of_the_width_and_the_height(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the random_aspect_crop type: its default draws keep shares of the width and of the height within
    [MINIMUM_ASPECT_CROP_FRACTION, 1] that differ for some images, given in the report, and a crop of the top left 75%
    of the width and the whole height stretches the image, its box and its class mask along the width only, the area
    scale of the objects being the product of the zooms of both sides, so that an instance mask left with a fifth of
    its pixels at that scale is removed while the object inside the crop is kept.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixes the shares of the width and of the height and the position of the crop.
    """
    random.seed(0)
    torch.manual_seed(seed=0)
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["random_aspect_crop"]))

    # Default draws: every share of the width and of the height lies within the range, and some images change their
    # aspect ratio
    reports = augmenter.augment_batch(images=torch.zeros(size=(32, 3, IMAGE_SIZE, IMAGE_SIZE)))[-1]
    drawn_shares = []
    for report in reports:
        type_name, drawn_values = report["geometric_augmentations"][0]
        report_match = re.fullmatch(pattern=r"(\d+) % of the width and (\d+) % of the height, at top \d+ and left \d+ "
                                            r"pixels", string=drawn_values)
        assert type_name == "random_aspect_crop" and report_match is not None
        drawn_shares.append((int(report_match.group(1)), int(report_match.group(2))))
    assert all(MINIMUM_ASPECT_CROP_FRACTION * 100 <= share <= 100 for shares in drawn_shares for share in shares)
    assert any(width_share != height_share for width_share, height_share in drawn_shares)

    # The left 72 of the 96 columns and every row, resized back: the columns stretch by 4/3 and the rows stay
    drawn_fractions = iter([MINIMUM_ASPECT_CROP_FRACTION, 1.0])
    monkeypatch.setattr(target=random, name="uniform", value=lambda low, high: next(drawn_fractions))
    monkeypatch.setattr(target=random, name="randint", value=lambda low, high: 0)
    box = [12.0, 24.0, 36.0, 72.0]
    mask = torch.zeros(size=(IMAGE_SIZE, IMAGE_SIZE), dtype=torch.int64)
    mask[24:72, 12:36] = 1
    images, boxes, labels, masks, _, _, reports = augmenter.augment_batch(
        images=create_rectangle_image(box=box).unsqueeze(dim=0), bounding_boxes=[torch.tensor(data=[box])],
        labels=[torch.tensor(data=[1])], masks=mask.unsqueeze(dim=0))
    stretched_box = [16.0, 24.0, 48.0, 72.0]
    assert boxes[0].tolist() == [stretched_box] and labels[0].tolist() == [1]
    assert torch.allclose(input=torch.tensor(data=get_bright_region_box(image=images[0])),
                          other=torch.tensor(data=stretched_box), atol=1.0)
    row_indices, column_indices = torch.nonzero(input=masks[0] == 1, as_tuple=True)
    assert [int(column_indices.min()), int(row_indices.min()), int(column_indices.max()) + 1,
            int(row_indices.max()) + 1] == [16, 24, 48, 72]
    assert reports[0]["geometric_augmentations"] == [("random_aspect_crop",
                                                      "75 % of the width and 100 % of the height, at top 0 and left 0 "
                                                      "pixels")]

    # Instance segmentation: the object with 6 of its 30 columns inside the crop keeps 8 stretched columns, a fifth of
    # the pixels it would have at the area scale of 4/3, and is removed, the other one being kept with the box of its
    # stretched mask, whose nearest resampling takes the output column j from the column floor(0.75 x j) of the crop
    drawn_fractions = iter([MINIMUM_ASPECT_CROP_FRACTION, 1.0])
    boxes = torch.tensor(data=[[66.0, 10.0, 96.0, 40.0], [10.0, 10.0, 40.0, 40.0]])
    instance_masks = torch.zeros(size=(2, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
    instance_masks[0, 10:40, 66:96] = 1
    instance_masks[1, 10:40, 10:40] = 1
    _, kept_boxes, kept_labels, _, kept_masks, _, _ = augmenter.augment_batch(
        images=torch.zeros(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE)), bounding_boxes=[boxes],
        labels=[torch.tensor(data=[2, 1])], instance_masks=[instance_masks])
    assert kept_labels[0].tolist() == [1] and kept_masks[0].shape == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert kept_boxes[0].tolist() == [[14.0, 10.0, 54.0, 40.0]]


def test_reports_name_the_drawn_types_their_values_and_the_removed_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the augmentation report of a sample and its panel lines in the input viewer: the largest translation
    along the width is reported with its shift, no photometric type is drawn, and the object pushed out of the image
    is named with its class.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture forcing the axes and the shift of the translation.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["random_translation"]))
    monkeypatch.setattr(target=random, name="choice", value=lambda sequence: "x")
    monkeypatch.setattr(target=random, name="randint", value=lambda low, high: high)
    maximum_shift = int(IMAGE_SIZE * MAXIMUM_TRANSLATION_FRACTION)

    _, _, _, _, _, _, reports = augmenter.augment_batch(
        images=create_rectangle_image(box=[10.0, 10.0, 30.0, 30.0]).unsqueeze(dim=0),
        bounding_boxes=[torch.tensor(data=[[10.0, 10.0, 30.0, 30.0], [80.0, 40.0, 95.0, 60.0]])],
        labels=[torch.tensor(data=[1, 2])])

    # The report holds the drawn shift, an empty photometric draw and the label of the removed object
    report = reports[0]
    assert report["augmented"] and report["geometric_augmentations"] == [("random_translation",
                                                                         f"x +{maximum_shift} pixels")]
    assert report["photometric_augmentations"] == [[]] and report["removed_labels"].tolist() == [2]

    # The panel lines group the types and name the removed object with its class
    assert augmenter.describe_report(report=report, label_names={1: "cat", 2: "dog"}) == [
        "Geometric :", f"  random_translation : x +{maximum_shift} pixels", "Photometric :", "  none",
        "Removed objects : 1 (dog)"]


@pytest.mark.parametrize(argnames="augmentation_type", argvalues=["random_crop", "random_left_right_flip",
                                                                  "random_rotation"])
def test_geometric_augmentation_moves_keypoints_with_the_image(augmentation_type: str) -> None:
    """
    Verifies that every geometric augmentation moving the targets moves the keypoints of the keypoint detection like the
    image content: a keypoint at the centre of the bright rectangle must land at the centre of the moved rectangle,
    keeping its visibility, and the box of its object must follow the rectangle, for several random draws.

    Args:
        augmentation_type (str): The geometric augmentation type under test.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=[augmentation_type]))

    # The rectangle lies within the central pixels 24 to 72 that every crop of at least 75% of a side keeps, so that a
    # crop never cuts it and the centre of its bright region stays the true centre the keypoint is compared with
    box = [24.0, 30.0, 54.0, 70.0]
    keypoints = torch.tensor(data=[[[39.0, 50.0, 2.0], [0.0, 0.0, 0.0]]])

    for seed in range(5):
        random.seed(seed)
        torch.manual_seed(seed)
        images, boxes, _, _, _, moved_keypoints, _ = augmenter.augment_batch(
            images=create_rectangle_image(box=box).unsqueeze(dim=0), bounding_boxes=[torch.tensor(data=[box])],
            labels=[torch.tensor(data=[1])], keypoints=[keypoints])

        # The centre keypoint follows the rectangle, the unlabelled one staying unlabelled at the origin
        bright_box = get_bright_region_box(image=images[0])
        bright_centre = torch.tensor(data=[(bright_box[0] + bright_box[2]) / 2, (bright_box[1] + bright_box[3]) / 2])
        assert moved_keypoints[0].shape == (1, 2, 3)
        assert torch.allclose(input=moved_keypoints[0][0, 0, :2], other=bright_centre,
                              atol=BOX_TOLERANCES[augmentation_type] + 1e-4)
        assert moved_keypoints[0][0, 0, 2] == 2.0 and moved_keypoints[0][0, 1].tolist() == [0.0, 0.0, 0.0]
        assert boxes[0].shape == (1, 4)


@pytest.mark.parametrize(argnames="augmentation_type", argvalues=["random_crop", "random_rotation"])
def test_geometric_augmentation_moves_the_ignored_regions_with_the_keypoints(augmentation_type: str) -> None:
    """
    Verifies that every geometric augmentation moving the targets moves the ignored regions of the keypoint detection,
    given as the binary masks of the augmenter together with the boxes and keypoints of the objects, with the image: the
    region drawn on the rectangle follows it and stays binary uint8, the border uncovered by a zoom-out, a translation
    or a rotation being left out of it.

    Args:
        augmentation_type (str): The geometric augmentation type under test.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=[augmentation_type]))
    box = [20.0, 30.0, 50.0, 70.0]
    ignored_regions = torch.zeros(size=(1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
    ignored_regions[0, 30:70, 20:50] = 1
    keypoints = torch.tensor(data=[[[35.0, 50.0, 2.0], [0.0, 0.0, 0.0]]])

    for seed in range(5):
        random.seed(seed)
        torch.manual_seed(seed=seed)
        images, _, _, moved_regions, _, moved_keypoints, _ = augmenter.augment_batch(
            images=create_rectangle_image(box=box).unsqueeze(dim=0), bounding_boxes=[torch.tensor(data=[box])],
            labels=[torch.tensor(data=[1])], masks=ignored_regions, keypoints=[keypoints])

        # The region stays binary on the moved rectangle, nothing else being ignored
        assert moved_regions.shape == (1, IMAGE_SIZE, IMAGE_SIZE) and moved_regions.dtype == torch.uint8
        assert set(moved_regions.unique().tolist()) <= {0, 1} and moved_keypoints[0].shape == (1, 2, 3)
        row_indices, column_indices = torch.nonzero(input=moved_regions[0], as_tuple=True)
        region_box = [float(column_indices.min()), float(row_indices.min()), float(column_indices.max()) + 1.0,
                      float(row_indices.max()) + 1.0]
        assert torch.allclose(input=torch.tensor(data=region_box),
                              other=torch.tensor(data=get_bright_region_box(image=images[0])),
                              atol=BOX_TOLERANCES[augmentation_type])


def test_flip_keeps_the_keypoint_names() -> None:
    """
    Verifies that a left-right flip mirrors every keypoint without renaming it: the keypoint at the top left corner of
    the object stays the first keypoint, now at the top right corner, since the user chooses the geometric types that
    suit the keypoints of the dataset.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["random_left_right_flip"]))
    keypoints = torch.tensor(data=[[[20.0, 30.0, 2.0], [50.0, 70.0, 1.0]]])

    _, _, _, _, _, flipped_keypoints, _ = augmenter.augment_batch(
        images=torch.zeros(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE)),
        bounding_boxes=[torch.tensor(data=[[20.0, 30.0, 50.0, 70.0]])], labels=[torch.tensor(data=[1])],
        keypoints=[keypoints])

    assert flipped_keypoints[0].tolist() == [[[IMAGE_SIZE - 20.0, 30.0, 2.0], [IMAGE_SIZE - 50.0, 70.0, 1.0]]]


def test_keypoints_leaving_the_image_are_unlabelled() -> None:
    """
    Verifies the cleaning of the keypoints after the geometric augmentations: a keypoint outside the image becomes
    unlabelled (visibility and position 0), an object keeping a labelled keypoint inside the image stays although too
    small a share of its box is left inside it, and an object whose keypoints all left the image is removed with its
    box.
    """
    boxes = torch.tensor(data=[[-50.0, 10.0, 10.0, 40.0], [-60.0, 10.0, -20.0, 40.0], [10.0, 10.0, 40.0, 40.0]])
    keypoints = torch.tensor(data=[[[-20.0, 20.0, 2.0], [5.0, 20.0, 2.0]], [[-30.0, 20.0, 2.0], [-25.0, 20.0, 1.0]],
                                   [[20.0, 20.0, 2.0], [30.0, float(IMAGE_SIZE), 2.0]]])

    kept_boxes, kept_labels, _, kept_keypoints, removed_labels = DataAugmenter.clean_bounding_boxes(
        bounding_boxes=boxes, labels=torch.tensor(data=[1, 2, 3]), image_height=IMAGE_SIZE, image_width=IMAGE_SIZE,
        keypoints=keypoints)

    assert kept_labels.tolist() == [1, 3] and removed_labels.tolist() == [2] and kept_boxes.shape == (2, 4)
    assert kept_keypoints.tolist() == [[[0.0, 0.0, 0.0], [5.0, 20.0, 2.0]], [[20.0, 20.0, 2.0], [0.0, 0.0, 0.0]]]


def test_perspective_warps_the_image_like_kornia(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies random_perspective, ported from kornia's RandomPerspective: every corner of the image moves inwards by its
    own drawn share of perspective_distortion_scale of half the image side along both axes, and the image is warped
    into the quadrilateral of the moved corners as kornia's warp_perspective does with the transform of
    get_perspective_transform, its uncovered border black and the largest move of a corner given in the report.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture recording the draws of the corners.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["random_perspective"]))
    images = torch.rand(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE), generator=torch.Generator().manual_seed(0)) * 255.0

    # Draws of a seeded generator, recorded so that the last eight, the shares of the corners, are known
    generator, drawn_shares = random.Random(5), []
    monkeypatch.setattr(target=random, name="random",
                        value=lambda: drawn_shares.append(generator.random()) or drawn_shares[-1])
    warped_images, _, _, _, _, _, reports = augmenter.augment_batch(images=images)

    # The moved corners, in the order of the corners and x before y, each moving inwards
    reach = SUPERVISED_AUGMENTATION_STRENGTHS["perspective_distortion_scale"] * IMAGE_SIZE / 2.0
    last_pixel = IMAGE_SIZE - 1.0
    corners = [(0.0, 0.0), (last_pixel, 0.0), (last_pixel, last_pixel), (0.0, last_pixel)]
    inward_directions = [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)]
    corner_shares = drawn_shares[-8:]
    moved_corners = [(corner_x + direction_x * reach * corner_shares[2 * index],
                      corner_y + direction_y * reach * corner_shares[2 * index + 1])
                     for index, ((corner_x, corner_y), (direction_x, direction_y))
                     in enumerate(zip(corners, inward_directions))]

    # The warp of kornia into the same corners, and the largest move of the report
    transform = get_perspective_transform(points_src=torch.tensor(data=[corners]),
                                          points_dst=torch.tensor(data=[moved_corners]))
    kornia_images = warp_perspective(src=images, M=transform, dsize=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear",
                                     padding_mode="zeros", align_corners=True)
    assert torch.allclose(input=warped_images, other=kornia_images, atol=0.05)
    assert float(warped_images[0, :, 0, 0].abs().max()) == 0.0 and float(warped_images[0, :, -1, -1].abs().max()) == 0.0
    largest_move = max(math.hypot(moved_x - corner_x, moved_y - corner_y)
                       for (moved_x, moved_y), (corner_x, corner_y) in zip(moved_corners, corners))
    assert reports[0]["geometric_augmentations"] == [("random_perspective",
                                                      f"corners moved by up to {largest_move:.1f} pixels")]


def test_jigsaw_puts_the_pieces_back_in_the_drawn_order() -> None:
    """
    Verifies random_jigsaw: the image cut into 2x2 pieces is put back with its pieces in the order given in the report,
    by slicing only so that every piece keeps its pixels exactly, a sequence receives the same order on all its
    frames, and an image whose sides are not multiples of the grid is refused.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["random_jigsaw"]))

    # Four pieces of distinct values, in reading order
    piece_size = IMAGE_SIZE // 2
    pieces = [torch.full(size=(3, piece_size, piece_size), fill_value=value) for value in [10.0, 80.0, 150.0, 220.0]]
    image = torch.cat(tensors=[torch.cat(tensors=pieces[:2], dim=2), torch.cat(tensors=pieces[2:], dim=2)], dim=1)

    # Every position of the shuffled image holds the piece the report names for it
    for seed in range(5):
        random.seed(seed)
        shuffled_images, _, _, _, _, _, reports = augmenter.augment_batch(images=image.unsqueeze(dim=0))
        drawn_values = dict(reports[0]["geometric_augmentations"])["random_jigsaw"]
        piece_order = [int(piece_index) for piece_index in drawn_values.removeprefix("pieces in the order ").split()]
        assert sorted(piece_order) == [0, 1, 2, 3]
        for position_index, piece_index in enumerate(piece_order):
            row_index, column_index = divmod(position_index, 2)
            assert torch.equal(input=shuffled_images[0, :, row_index * piece_size:(row_index + 1) * piece_size,
                                                     column_index * piece_size:(column_index + 1) * piece_size],
                               other=pieces[piece_index])

    # The frames of a sequence share their order, and sides that are not multiples of the grid are refused
    shuffled_sequences = augmenter.augment_batch(images=image.expand(1, 3, 3, IMAGE_SIZE, IMAGE_SIZE).clone())[0]
    assert all(torch.equal(input=frame, other=shuffled_sequences[0, 0]) for frame in shuffled_sequences[0])
    with pytest.raises(expected_exception=ValueError, match="multiples of 2"):
        augmenter.augment_batch(images=torch.zeros(size=(1, 3, IMAGE_SIZE + 1, IMAGE_SIZE + 1)))


def test_hue_turns_the_colours_keeping_their_brightness_and_saturation() -> None:
    """
    Verifies random_hue: the hue of every coloured pixel turns by the angle given in the report, within
    maximum_hue_shift of the hue circle in either direction, while the largest and the smallest channel of every pixel,
    its brightness and its saturation, are kept, and a grey image, which has no hue, is left unchanged.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["random_hue"]))
    images = torch.rand(size=(4, 3, IMAGE_SIZE, IMAGE_SIZE), generator=torch.Generator().manual_seed(0)) * 255.0
    random.seed(0)
    turned_images, _, _, _, _, _, reports = augmenter.augment_batch(images=images)

    # The brightest and the darkest channel of every pixel kept
    assert not torch.allclose(input=turned_images, other=images, atol=1.0)
    assert torch.allclose(input=turned_images.amax(dim=1), other=images.amax(dim=1), atol=0.1)
    assert torch.allclose(input=turned_images.amin(dim=1), other=images.amin(dim=1), atol=0.1)

    # The hue of the saturated pixels, in the HSV space of the images reversed to RGB, turned by the reported angle
    maximum_degrees = SUPERVISED_AUGMENTATION_STRENGTHS["maximum_hue_shift"] * 360.0
    original_hsv = rgb_to_hsv(image=images.flip(dims=(1,)) / 255.0)
    turned_hsv = rgb_to_hsv(image=turned_images.flip(dims=(1,)) / 255.0)
    for image_index, report in enumerate(reports):
        drawn_values = dict(report["photometric_augmentations"][0])["random_hue"]
        degrees = float(re.fullmatch(pattern=r"hue ([+-]\d+\.\d) degrees", string=drawn_values).group(1))
        assert -maximum_degrees <= degrees <= maximum_degrees
        saturated_pixels = original_hsv[image_index, 1] > 0.3
        hue_turns = torch.remainder(input=turned_hsv[image_index, 0] - original_hsv[image_index, 0],
                                    other=2.0 * math.pi)[saturated_pixels]
        expected_turn = math.radians(degrees) % (2.0 * math.pi)
        turn_errors = torch.remainder(input=hue_turns - expected_turn + math.pi, other=2.0 * math.pi) - math.pi
        assert float(turn_errors.abs().max()) < math.radians(0.5)

    # A grey image keeps its pixels
    grey_images = torch.full(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE), fill_value=90.0)
    assert torch.allclose(input=augmenter.augment_batch(images=grey_images)[0], other=grey_images, atol=1e-3)


def test_gamma_raises_every_pixel_to_the_drawn_power(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies random_gamma: every pixel of the image scaled to [0, 1] is raised to the drawn gamma, the bounds of
    gamma_range darkening the mid-tones for the highest gamma and lightening them for the lowest one, the gamma being
    given in the report.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture drawing the bounds of the range.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["random_gamma"]))
    images = torch.rand(size=(2, 3, IMAGE_SIZE, IMAGE_SIZE), generator=torch.Generator().manual_seed(0)) * 255.0
    for bound_index, expected_change in [(0, 1.0), (1, -1.0)]:
        gamma = SUPERVISED_AUGMENTATION_STRENGTHS["gamma_range"][bound_index]
        monkeypatch.setattr(target=random, name="uniform", value=lambda low, high, drawn_gamma=gamma: drawn_gamma)
        corrected_images, _, _, _, _, _, reports = augmenter.augment_batch(images=images)
        assert torch.allclose(input=corrected_images, other=255.0 * (images / 255.0) ** gamma, atol=1e-3)
        assert bool((torch.sign(input=corrected_images - images) != -expected_change).all())
        assert reports[0]["photometric_augmentations"][0] == [("random_gamma", f"gamma {gamma:.2f}")]


def test_linear_illumination_ramps_from_nothing_to_the_gain_towards_its_side(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies random_linear_illumination: a ramp growing linearly from 0 on one side of the image to the drawn gain on
    the opposite side brightens or darkens every pixel, along the rows or the columns towards the drawn side, the
    signed gain and the side being given in the report.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture drawing the gain, its sign and its side.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(
        enabled_types=["random_linear_illumination"]))
    images = torch.full(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE), fill_value=100.0)
    gain = SUPERVISED_AUGMENTATION_STRENGTHS["linear_illumination_gain_range"][1]
    ramp = torch.linspace(start=0.0, end=1.0, steps=IMAGE_SIZE)
    monkeypatch.setattr(target=random, name="uniform", value=lambda low, high: high)

    # A brightening ramp towards the right, then a darkening one towards the top, the sign drawn below or above 0.5
    for direction, sign_draw, sign, expected_ramp in [("right", 0.25, 1.0, ramp.view(1, -1)),
                                                      ("top", 0.75, -1.0, ramp.flip(dims=(0,)).view(-1, 1))]:
        monkeypatch.setattr(target=random, name="choice", value=lambda sequence, side=direction: side)
        monkeypatch.setattr(target=random, name="random", value=lambda drawn_sign=sign_draw: drawn_sign)
        lit_images, _, _, _, _, _, reports = augmenter.augment_batch(images=images)
        expected_images = (100.0 + sign * gain * expected_ramp).expand(3, IMAGE_SIZE, IMAGE_SIZE)
        assert torch.allclose(input=lit_images[0], other=expected_images, atol=1e-3)
        assert reports[0]["photometric_augmentations"][0] == [
            ("random_linear_illumination", f"{sign * gain:+.0f} levels growing towards the {direction}")]


@pytest.mark.parametrize(argnames="device_name", argvalues=["cpu"])
def test_plasma_map_is_the_diamond_square_map_of_kornia(device_name: str) -> None:
    """
    Verifies build_plasma_map, the fractal map of random_plasma_shadow ported from kornia's diamond_square so that it
    is drawn on the device without any copy from the host: for the same seed it gives the map of kornia normalized to
    [0, 1], for square and rectangular sizes and several roughnesses, on the CPU and on the GPU.

    Args:
        device_name (str): The device of the maps.
    """
    device = torch.device(device_name)
    for height, width in [(3, 3), (32, 32), (96, 128), (64, 256)]:
        for roughness in [0.1, 0.4, 0.7]:
            torch.manual_seed(seed=7)
            ported_map = build_plasma_map(height=height, width=width, roughness=roughness, device=device)
            torch.manual_seed(seed=7)
            kornia_map = diamond_square(output_size=(1, 1, height, width), roughness=roughness, device=device,
                                        dtype=torch.float32, normalize_range=(0.0, 1.0))[0, 0]
            assert ported_map.shape == (height, width) and ported_map.device.type == device.type
            assert torch.allclose(input=ported_map, other=kornia_map, atol=1e-4)


def test_plasma_shadow_darkens_the_pixels_of_the_map_below_the_quantity(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies random_plasma_shadow: the pixels whose fractal map lies below the drawn quantity are darkened by the drawn
    intensity and the others are kept, the map being the one of build_plasma_map for the drawn roughness and the same
    seed, and the intensity, the quantity and the roughness being given in the report.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture drawing the roughness, the quantity and the intensity.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["random_plasma_shadow"]))
    images = torch.full(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE), fill_value=200.0)

    # The roughness, the quantity and the intensity, in the order of their draws
    drawn_values = iter([0.4, 0.5, -60.0])
    monkeypatch.setattr(target=random, name="uniform", value=lambda low, high: next(drawn_values))
    torch.manual_seed(seed=0)
    shadowed_images, _, _, _, _, _, reports = augmenter.augment_batch(images=images)
    torch.manual_seed(seed=0)
    plasma_map = build_plasma_map(height=IMAGE_SIZE, width=IMAGE_SIZE, roughness=0.4, device=torch.device("cpu"))

    shadowed_pixels = plasma_map < 0.5
    assert 0 < int(shadowed_pixels.sum()) < IMAGE_SIZE * IMAGE_SIZE
    assert torch.equal(input=shadowed_images[0], other=(200.0 - 60.0 * shadowed_pixels.float()).expand(3, -1, -1))
    assert reports[0]["photometric_augmentations"][0] == [
        ("random_plasma_shadow", "intensity -60 levels, quantity 0.50, roughness 0.40")]


def test_solarize_inverts_the_pixels_at_or_above_the_threshold() -> None:
    """
    Verifies random_solarize: every pixel at or above the threshold of 127.5 levels is inverted (255 minus its value)
    and the darker ones are kept, the threshold being given in the report.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["random_solarize"]))
    images = torch.zeros(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    images[0, :, 0, :6] = torch.tensor(data=[0.0, 100.0, 127.0, 127.5, 200.0, 255.0])
    solarized_images, _, _, _, _, _, reports = augmenter.augment_batch(images=images)
    assert solarized_images[0, 0, 0, :6].tolist() == [0.0, 100.0, 127.0, 127.5, 55.0, 0.0]
    assert torch.equal(input=solarized_images[0, :, 1:], other=images[0, :, 1:])
    assert reports[0]["photometric_augmentations"][0] == [("random_solarize", "threshold 127.5 levels")]


def test_erasing_blacks_out_one_rectangle_after_the_noise() -> None:
    """
    Verifies random_erasing, drawn last: every image gets one black rectangle covering a share of its area within
    erasing_area_range, its pixels exactly at 0 although the noise drawn before it reached every pixel, the rectangle
    of the report being the one erased and the rest of the image carrying the noise.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(
        enabled_types=["random_noise", "random_erasing"]))
    grey_images = torch.full(size=(3, 3, IMAGE_SIZE, IMAGE_SIZE), fill_value=128.0)
    random.seed(0)
    torch.manual_seed(seed=0)
    erased_images, _, _, _, _, _, reports = augmenter.augment_batch(images=grey_images)

    # The black pixels of every image are exactly those of its reported rectangle
    minimum_area, maximum_area = SUPERVISED_AUGMENTATION_STRENGTHS["erasing_area_range"]
    for image, report in zip(erased_images, reports):
        photometric_augmentations = report["photometric_augmentations"][0]
        assert [name for name, _ in photometric_augmentations] == ["random_noise", "random_erasing"]
        width, height, top, left = (int(value) for value in re.fullmatch(
            pattern=r"(\d+)x(\d+) pixels at top (\d+) and left (\d+) pixels",
            string=dict(photometric_augmentations)["random_erasing"]).groups())
        erased_pixels = torch.zeros(size=(IMAGE_SIZE, IMAGE_SIZE), dtype=torch.bool)
        erased_pixels[top:top + height, left:left + width] = True
        assert torch.equal(input=(image == 0.0).all(dim=0), other=erased_pixels)
        assert minimum_area - 0.005 <= width * height / IMAGE_SIZE ** 2 <= maximum_area + 0.005
        assert float(image[:, ~erased_pixels].std()) > NOISE_STANDARD_DEVIATION / 2.0


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="The training batches are augmented on the GPU.")
@pytest.mark.parametrize(argnames="augmentation_type", argvalues=["random_perspective"])
def test_view_types_on_the_gpu_never_make_the_host_wait(augmentation_type: str) -> None:
    """
    Verifies the types of the views of the self-supervised methods on the GPU, where the batches are augmented: drawing
    one never makes the host wait for the device, which the synchronization debug mode of PyTorch turns into an error,
    and the same draws give the images and the reports of the CPU, but for the plasma shadow, whose map is drawn by the
    generator of the device.

    Args:
        augmentation_type (str): The type under test.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=[augmentation_type]))
    images = torch.rand(size=(2, 3, IMAGE_SIZE, IMAGE_SIZE), generator=torch.Generator().manual_seed(0)) * 255.0
    device_images = images.to(device="cuda")
    augmenter.augment_batch(images=device_images)
    torch.cuda.synchronize()

    # The type on the GPU with every synchronization of the host turned into an error
    random.seed(3)
    torch.cuda.set_sync_debug_mode(debug_mode="error")
    try:
        gpu_images, _, _, _, _, _, gpu_reports = augmenter.augment_batch(images=device_images)
    finally:
        torch.cuda.set_sync_debug_mode(debug_mode="default")
    assert gpu_images.device.type == "cuda" and gpu_images.shape == images.shape

    # The same draws on the CPU give the same images and the same reports
    if augmentation_type != "random_plasma_shadow":
        random.seed(3)
        cpu_images, _, _, _, _, _, cpu_reports = augmenter.augment_batch(images=images)
        assert gpu_reports == cpu_reports
        assert torch.allclose(input=gpu_images.cpu(), other=cpu_images, atol=1e-2)


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="The training batches are augmented on the GPU.")
@pytest.mark.parametrize(argnames="augmentation_type", argvalues=["clahe_augment"])
def test_device_computed_types_on_the_gpu_never_make_the_host_wait(augmentation_type: str) -> None:
    """
    Verifies the rotation, the CLAHE, the sharpening and the histogram equalizations on the GPU, where the batches are
    augmented: their operations ported from torchvision and kornia and their kernels being filled on the device,
    drawing one on images, and the rotation on their class masks too, never makes the host wait for the device, which
    the synchronization debug mode of PyTorch turns into an error, and the images and masks stay on the device.

    Args:
        augmentation_type (str): The type under test.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=[augmentation_type]))
    images = (torch.rand(size=(2, 3, IMAGE_SIZE, IMAGE_SIZE), generator=torch.Generator().manual_seed(0)) *
              255.0).to(device="cuda")
    masks = torch.ones(size=(2, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.int64, device="cuda") if \
        augmentation_type == "random_rotation" else None
    augmenter.augment_batch(images=images, masks=masks)
    torch.cuda.synchronize()

    # The type on the GPU with every synchronization of the host turned into an error
    random.seed(3)
    torch.cuda.set_sync_debug_mode(debug_mode="error")
    try:
        augmented_images, _, _, augmented_masks, _, _, reports = augmenter.augment_batch(images=images, masks=masks)
    finally:
        torch.cuda.set_sync_debug_mode(debug_mode="default")
    assert augmented_images.device.type == "cuda" and augmented_images.shape == images.shape
    drawn_types = [[name for name, _ in report["geometric_augmentations"] + report["photometric_augmentations"][0]]
                   for report in reports]
    assert drawn_types == [[augmentation_type], [augmentation_type]]
    if masks is not None:
        assert augmented_masks.device.type == "cuda" and augmented_masks.dtype == torch.int64
        assert set(augmented_masks.unique().tolist()) <= {1, IGNORED_LABEL_INDEX}
