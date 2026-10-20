"""
Tests of the ViewAugmenter of vision_training/data/view_augmentations.py, which draws the two random views of every
image of the self-supervised trainer before the forward pass: the views keep the shape and the [0, 255] range of the
BGR images, both views differ, every transformation turned off gives the image back, the BGR channels reach the RGB
transformations of kornia in the right order, a seed gives the same views, the views (and the single view drawn for the
view losses of the image classifier) stay in float32 under the bfloat16 autocast of mixed precision, they describe the
transformations drawn for them in the information panels of the input viewer, and on the GPU they stay on the device
and match the views drawn on the CPU from the same seed.
"""

import copy
import pytest
import torch

from typing import Any, Dict

from vision_training.data.view_augmentations import VIEW_TRANSFORMATION_NAMES, ViewAugmenter

# Side of the square images of the tests
IMAGE_SIZE = 32

# Largest difference, in [0, 255] pixel values, between the views drawn on the GPU and the CPU from the same seed, the
# interpolations of the crops and rotations and the blur rounding differently on the two devices (measured below 0.001)
GPU_VIEWS_TOLERANCE = 0.01

# View augmentation of the synthetic shapes configuration, every transformation being drawn
VIEW_AUGMENTATION_CONFIGURATION = {
    "random_resized_crop": {"scale": [0.6, 1.0], "aspect_ratio": [0.75, 1.3333]},
    "left_right_flip": {"probability": 0.5}, "up_down_flip": {"probability": 0.5},
    "rotation": {"probability": 0.5, "maximum_degrees": 180.0},
    "colour_jitter": {"probability": 0.8, "brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1},
    "grayscale": {"probability": 0.2}, "gaussian_blur": {"probability": 0.5, "kernel_size": 5, "sigma": [0.1, 2.0]}}


def build_identity_configuration() -> Dict[str, Any]:
    """
    Builds a view augmentation turning every transformation off: a crop of the whole image of the same aspect ratio,
    and a probability of 0 for every other transformation.

    Returns:
        Dict[str, Any]: The 'Data.view_augmentation' section.
    """
    configuration = copy.deepcopy(VIEW_AUGMENTATION_CONFIGURATION)
    configuration["random_resized_crop"] = {"scale": [1.0, 1.0], "aspect_ratio": [1.0, 1.0]}
    for transformation_name in ["left_right_flip", "up_down_flip", "rotation", "colour_jitter", "grayscale",
                                "gaussian_blur"]:
        configuration[transformation_name]["probability"] = 0.0
    return configuration


def create_test_images(number_images: int) -> torch.Tensor:
    """
    Creates float BGR images in [0, 255] of random content, as the loaders deliver them.

    Args:
        number_images (int): The number of images of the batch.

    Returns:
        torch.Tensor: The [N, 3, IMAGE_SIZE, IMAGE_SIZE] images.
    """
    generator = torch.Generator().manual_seed(0)
    return torch.randint(low=0, high=256, size=(number_images, 3, IMAGE_SIZE, IMAGE_SIZE),
                         generator=generator).to(dtype=torch.float32)


def test_views_keep_the_images_format_and_differ() -> None:
    """
    Verifies that the two views of a batch keep the shape, the float32 type and the [0, 255] range of the images, and
    that the views of an image differ from each other and from the image.
    """
    images = create_test_images(number_images=4)
    view_augmenter = ViewAugmenter(view_augmentation_configuration=VIEW_AUGMENTATION_CONFIGURATION,
                                   image_size=IMAGE_SIZE)
    torch.manual_seed(seed=0)
    first_views, second_views, _ = view_augmenter.create_views(images=images)
    for views in [first_views, second_views]:
        assert views.shape == images.shape and views.dtype == torch.float32
        assert float(views.min()) >= 0.0 and float(views.max()) <= 255.0
        assert not torch.allclose(input=views, other=images)
    assert not torch.allclose(input=first_views, other=second_views)


def test_views_without_transformation_give_the_images_back() -> None:
    """
    Verifies that turning every transformation off gives both views equal to the images, the conversion of the BGR
    images to the RGB images in [0, 1] of kornia and back leaving them unchanged.
    """
    images = create_test_images(number_images=3)
    view_augmenter = ViewAugmenter(view_augmentation_configuration=build_identity_configuration(),
                                   image_size=IMAGE_SIZE)
    first_views, second_views, _ = view_augmenter.create_views(images=images)
    assert torch.allclose(input=first_views, other=images, atol=1e-3)
    assert torch.allclose(input=second_views, other=images, atol=1e-3)


def test_views_convert_the_bgr_channels_in_order() -> None:
    """
    Verifies that the BGR images reach the RGB transformations of kornia with their channels in the right order: the
    grayscale conversion of a pure blue image gives 0.114 of its intensity (the weight of the blue channel), not the
    0.299 of the red channel.
    """
    configuration = build_identity_configuration()
    configuration["grayscale"]["probability"] = 1.0
    view_augmenter = ViewAugmenter(view_augmentation_configuration=configuration, image_size=IMAGE_SIZE)
    blue_image = torch.zeros(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    blue_image[:, 0] = 255.0
    first_views, _, _ = view_augmenter.create_views(images=blue_image)
    assert torch.allclose(input=first_views, other=torch.full_like(input=first_views, fill_value=0.114 * 255.0),
                          atol=0.5)


def test_views_follow_the_seed() -> None:
    """
    Verifies that the same seed gives the same views, the random parameters being drawn with the PyTorch generator,
    which lets the self-supervised trainer draw the same views of a sample image at every evaluation.
    """
    images = create_test_images(number_images=4)
    view_augmenter = ViewAugmenter(view_augmentation_configuration=VIEW_AUGMENTATION_CONFIGURATION,
                                   image_size=IMAGE_SIZE)
    drawn_views = []
    for _ in range(2):
        torch.manual_seed(seed=7)
        drawn_views.append(view_augmenter.create_views(images=images))
    assert all(torch.equal(input=first_draw, other=second_draw)
               for first_draw, second_draw in zip(drawn_views[0][:2], drawn_views[1][:2]))


def test_views_stay_in_float32_under_mixed_precision() -> None:
    """
    Verifies that bfloat16 images within the bfloat16 autocast of a mixed precision forward pass give float32 views,
    the interpolations of the crops and rotations keeping their precision, for the two views of the self-supervised
    task and for the single view drawn by draw_view, the second view of the view losses of the image classifier.
    """
    images = create_test_images(number_images=2).to(dtype=torch.bfloat16)
    view_augmenter = ViewAugmenter(view_augmentation_configuration=VIEW_AUGMENTATION_CONFIGURATION,
                                   image_size=IMAGE_SIZE)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        first_views, second_views, _ = view_augmenter.create_views(images=images)
        single_views = view_augmenter.draw_view(images=images)
    assert first_views.dtype == torch.float32 and second_views.dtype == torch.float32
    assert single_views.dtype == torch.float32 and single_views.shape == images.shape


def test_views_describe_their_drawn_transformations() -> None:
    """
    Verifies the descriptions of the views written in the information panels of the input viewer: none unless asked
    for, the whole image cropped with every other transformation turned off, and with every transformation drawn one
    list of lines per view and per image, each naming a transformation, the crop first with its size.
    """
    images = create_test_images(number_images=3)

    # Nothing is described unless asked for, which keeps the training free of reads from the device
    view_augmenter = ViewAugmenter(view_augmentation_configuration=VIEW_AUGMENTATION_CONFIGURATION,
                                   image_size=IMAGE_SIZE)
    assert view_augmenter.create_views(images=images)[2] is None

    # The identity configuration only crops the whole image
    identity_augmenter = ViewAugmenter(view_augmentation_configuration=build_identity_configuration(),
                                       image_size=IMAGE_SIZE)
    _, _, view_descriptions = identity_augmenter.create_views(images=images, describe_views=True)
    assert view_descriptions == [[[f"random_resized_crop : {IMAGE_SIZE}x{IMAGE_SIZE} pixels at x 0, y 0 "
                                   f"(100 % of the area)"]] * 3] * 2

    # Every view of every image starts with its crop, followed by the other transformations drawn for it
    torch.manual_seed(seed=3)
    _, _, view_descriptions = view_augmenter.create_views(images=images, describe_views=True)
    assert len(view_descriptions) == 2 and all(len(image_lines) == 3 for image_lines in view_descriptions)
    for image_lines in [lines for view_lines in view_descriptions for lines in view_lines]:
        assert image_lines[0].startswith("random_resized_crop : ") and "% of the area" in image_lines[0]
        assert all(line.split(" : ")[0] in VIEW_TRANSFORMATION_NAMES for line in image_lines)


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="The views of the training run on the GPU.")
def test_views_on_the_gpu_match_the_views_on_the_cpu() -> None:
    """
    Verifies the views drawn on the GPU, where the self-supervised trainer draws them: their parameters being drawn on
    the CPU with the PyTorch generator, the same seed gives the same transformations on both devices, so the GPU
    views stay on the device in float32, match the CPU views up to the rounding of the interpolations and are
    described with the same lines, and they stay in float32 within the bfloat16 autocast of the GPU.
    """
    images = create_test_images(number_images=8)
    view_augmenter = ViewAugmenter(view_augmentation_configuration=VIEW_AUGMENTATION_CONFIGURATION,
                                   image_size=IMAGE_SIZE)

    # The same seed on both devices, every transformation being drawn
    drawn_views = {}
    for device_name in ["cpu", "cuda"]:
        torch.manual_seed(seed=5)
        drawn_views[device_name] = view_augmenter.create_views(images=images.to(device=device_name),
                                                               describe_views=True)

    # Same views and same descriptions, the GPU views staying on the device
    for view_index in range(2):
        gpu_views = drawn_views["cuda"][view_index]
        assert gpu_views.device.type == "cuda" and gpu_views.dtype == torch.float32
        assert float(gpu_views.min()) >= 0.0 and float(gpu_views.max()) <= 255.0
        assert torch.allclose(input=gpu_views.cpu(), other=drawn_views["cpu"][view_index], atol=GPU_VIEWS_TOLERANCE)
    assert drawn_views["cuda"][2] == drawn_views["cpu"][2]

    # Float32 views within the bfloat16 autocast of a mixed precision forward pass on the GPU
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        first_views, second_views, _ = view_augmenter.create_views(
            images=images.to(device="cuda", dtype=torch.bfloat16))
    assert first_views.dtype == torch.float32 and second_views.dtype == torch.float32
