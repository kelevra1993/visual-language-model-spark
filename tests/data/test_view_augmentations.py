"""
Tests of the ViewAugmenter of vision_training/data/view_augmentations.py, which draws the two random views of every
image of the self-supervised trainer before the forward pass: the views keep the shape and the [0, 255] range of the
BGR images, both views differ, every transformation turned off gives the image back, the BGR channels reach the RGB
transformations of kornia in the right order, a seed gives the same views, the views (and the single view drawn for the
view losses of the image classifier) stay in float32 under the bfloat16 autocast of mixed precision, the chain of every
view follows the order of the transformations and takes the probabilities of its view, the haze blends the images with
their airlight, the random erasing leaves a plain rectangle after the noise, the views describe the transformations
drawn for them in the information panels of the input viewer, and on the GPU they stay on the device and match the
views drawn on the CPU from the same seed.
"""

import copy
import pytest
import re
import torch

from typing import Any, Dict

from vision_training.data.view_augmentations import SECOND_VIEW_INDEX, VIEW_TRANSFORMATION_NAMES, ViewAugmenter

# Side of the square images of the tests
IMAGE_SIZE = 32

# Largest difference, in [0, 255] pixel values, between the views drawn on the GPU and the CPU from the same seed, the
# interpolations of the crops, rotations and perspectives and the blurs rounding differently on the two devices
# (measured below 0.004)
GPU_VIEWS_TOLERANCE = 0.01

# View augmentation of the self-supervised configurations (with the crop of the synthetic shapes one and a smaller
# Gaussian blur kernel), the jigsaw turned on so that every transformation is drawn, the Gaussian blur and the
# solarization having a probability per view
VIEW_AUGMENTATION_CONFIGURATION = {
    "random_resized_crop": {"scale": [0.6, 1.0], "aspect_ratio": [0.75, 1.3333]},
    "left_right_flip": {"probability": 0.5}, "up_down_flip": {"probability": 0.5},
    "rotation": {"probability": 0.5, "maximum_degrees": 180.0},
    "perspective": {"probability": 0.5, "distortion_scale": 0.2}, "jigsaw": {"probability": 0.2, "grid": [2, 2]},
    "colour_jitter": {"probability": 0.8, "brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1},
    "grayscale": {"probability": 0.2}, "gamma": {"probability": 0.3, "gamma": [0.7, 1.5]},
    "haze": {"probability": 0.3, "strength": [0.0, 0.5], "airlight": [0.6, 1.0]},
    "linear_illumination": {"probability": 0.3, "gain": [0.05, 0.25]},
    "plasma_shadow": {"probability": 0.2, "shade_intensity": [-0.5, 0.0], "shade_quantity": [0.0, 0.5],
                      "roughness": [0.1, 0.7]},
    "gaussian_blur": {"probability": [1.0, 0.1], "kernel_size": 5, "sigma": [0.1, 2.0]},
    "motion_blur": {"probability": 0.1, "kernel_size": [3, 9], "maximum_angle": 180.0, "direction": [-1.0, 1.0]},
    "solarize": {"probability": [0.0, 0.2], "threshold": [0.5, 0.5], "addition": [0.0, 0.0]},
    "gaussian_noise": {"probability": 0.2, "standard_deviation": 0.03},
    "random_erasing": {"probability": 0.25, "scale": [0.02, 0.15], "aspect_ratio": [0.3, 3.3]}}

# Classes of the kornia transformations of the chain of every view, in the order of VIEW_TRANSFORMATION_NAMES: the crop
# and the geometric transformations first, the photometric ones, the Gaussian noise after both blurs and the random
# erasing last
VIEW_TRANSFORMATION_CLASSES = ["RandomResizedCrop", "RandomHorizontalFlip", "RandomVerticalFlip", "RandomRotation",
                               "RandomPerspective", "RandomJigsaw", "ColorJitter", "RandomGrayscale", "RandomGamma",
                               "RandomHaze", "RandomLinearIllumination", "RandomPlasmaShadow", "RandomGaussianBlur",
                               "RandomMotionBlur", "RandomSolarize", "RandomGaussianNoise", "RandomErasing"]


def build_identity_configuration() -> Dict[str, Any]:
    """
    Builds a view augmentation turning every transformation off: a crop of the whole image of the same aspect ratio,
    and a probability of 0 for every other transformation. Also used by the tests of the input viewer panels.

    Returns:
        Dict[str, Any]: The 'Data.view_augmentation' section.
    """
    configuration = copy.deepcopy(VIEW_AUGMENTATION_CONFIGURATION)
    configuration["random_resized_crop"] = {"scale": [1.0, 1.0], "aspect_ratio": [1.0, 1.0]}
    for transformation_name in VIEW_TRANSFORMATION_NAMES[1:]:
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
        single_views = view_augmenter.draw_view(images=images, view_index=SECOND_VIEW_INDEX)
    assert first_views.dtype == torch.float32 and second_views.dtype == torch.float32
    assert single_views.dtype == torch.float32 and single_views.shape == images.shape


def test_view_chains_follow_the_order_of_the_transformations() -> None:
    """
    Verifies the chain of every view: the crop and the geometric transformations first, the perspective and the
    jigsaw right after the rotation, the photometric transformations, the Gaussian noise after both blurs (which would
    smooth it away) and the random erasing last (so that the erased rectangles stay plainly filled), named in the same
    order by VIEW_TRANSFORMATION_NAMES for their descriptions.
    """
    view_augmenter = ViewAugmenter(view_augmentation_configuration=VIEW_AUGMENTATION_CONFIGURATION,
                                   image_size=IMAGE_SIZE)
    assert len(view_augmenter.view_transformations) == 2
    for transformations in view_augmenter.view_transformations:
        assert [type(transformation).__name__ for transformation in transformations] == VIEW_TRANSFORMATION_CLASSES
    assert len(VIEW_TRANSFORMATION_NAMES) == len(VIEW_TRANSFORMATION_CLASSES)


def test_probabilities_given_per_view_apply_to_their_view() -> None:
    """
    Verifies the probabilities given per view, [first view, second view]: the chain of each view takes the probability
    of its view, a number giving both views the same one, so that a grayscale conversion of probability [1, 0] turns
    the first view of every image gray and leaves its second view, also the single view of the view losses of the
    image classifier (draw_view of the second view), equal to the image.
    """
    # Probabilities of the chains of both views
    view_augmenter = ViewAugmenter(view_augmentation_configuration=VIEW_AUGMENTATION_CONFIGURATION,
                                   image_size=IMAGE_SIZE)
    probabilities = {transformation_name: [float(transformations[index].p)
                                           for transformations in view_augmenter.view_transformations]
                     for index, transformation_name in enumerate(VIEW_TRANSFORMATION_NAMES)}
    assert probabilities["gaussian_blur"] == [1.0, 0.1] and probabilities["solarize"] == [0.0, 0.2]
    assert probabilities["rotation"] == [0.5, 0.5] and probabilities["random_erasing"] == [0.25, 0.25]

    # A grayscale conversion drawn on the first view only
    configuration = build_identity_configuration()
    configuration["grayscale"]["probability"] = [1.0, 0.0]
    view_augmenter = ViewAugmenter(view_augmentation_configuration=configuration, image_size=IMAGE_SIZE)
    images = create_test_images(number_images=3)
    first_views, second_views, _ = view_augmenter.create_views(images=images)
    assert torch.allclose(input=first_views, other=first_views[:, :1].expand_as(other=first_views), atol=1e-3)
    assert not torch.allclose(input=first_views, other=images, atol=1.0)
    assert torch.allclose(input=second_views, other=images, atol=1e-3)
    assert torch.allclose(input=view_augmenter.draw_view(images=images, view_index=SECOND_VIEW_INDEX), other=images,
                          atol=1e-3)


def test_haze_blends_the_images_with_their_airlight() -> None:
    """
    Verifies the haze of the views, which kornia does not provide: every hazy image is blended with its airlight,
    view x (1 - strength) + strength x airlight in the RGB images in [0, 1], here half of a white airlight, the drawn
    strength and airlight being described in the information panels, and random strengths and airlights are drawn
    within their ranges for every image.
    """
    # Half of a white airlight on every image
    configuration = build_identity_configuration()
    configuration["haze"] = {"probability": 1.0, "strength": [0.5, 0.5], "airlight": [1.0, 1.0]}
    view_augmenter = ViewAugmenter(view_augmentation_configuration=configuration, image_size=IMAGE_SIZE)
    images = create_test_images(number_images=2)
    first_views, second_views, view_descriptions = view_augmenter.create_views(images=images, describe_views=True)
    expected_views = images * 0.5 + 0.5 * 255.0
    assert torch.allclose(input=first_views, other=expected_views, atol=1e-3)
    assert torch.allclose(input=second_views, other=expected_views, atol=1e-3)
    assert view_descriptions[0][0][-1] == "haze : strength 0.50, airlight 1.00"

    # Random strengths and airlights, drawn within their ranges and applied with the formula to every image
    configuration["haze"] = {"probability": 1.0, "strength": [0.1, 0.4], "airlight": [0.6, 0.9]}
    view_augmenter = ViewAugmenter(view_augmentation_configuration=configuration, image_size=IMAGE_SIZE)
    torch.manual_seed(seed=1)
    hazy_views = view_augmenter.draw_view(images=images, view_index=0)
    haze_parameters = view_augmenter.view_transformations[0][VIEW_TRANSFORMATION_NAMES.index("haze")]._params
    strengths = haze_parameters["strength"].view(-1, 1, 1, 1)
    airlights = haze_parameters["airlight"].view(-1, 1, 1, 1)
    assert bool(((strengths >= 0.1) & (strengths <= 0.4)).all() and ((airlights >= 0.6) & (airlights <= 0.9)).all())
    assert not torch.equal(input=strengths[0], other=strengths[1])
    assert torch.allclose(input=hazy_views, other=images * (1.0 - strengths) + strengths * airlights * 255.0,
                          atol=1e-3)


def test_random_erasing_leaves_a_plain_rectangle_after_the_noise() -> None:
    """
    Verifies that the random erasing, last in the chain, leaves its rectangle plainly black, the Gaussian noise drawn
    before it never reaching the erased pixels, and that the rectangle described in the information panels is the one
    erased: with mid-grey images, which the noise of standard deviation 0.03 (in [0, 1], about 8 of the 255 levels)
    never brings down to 0, the black pixels of every view are exactly those of its described rectangle, and the
    other pixels carry the noise.
    """
    configuration = build_identity_configuration()
    configuration["gaussian_noise"] = {"probability": 1.0, "standard_deviation": 0.03}
    configuration["random_erasing"] = {"probability": 1.0, "scale": [0.05, 0.15], "aspect_ratio": [0.3, 3.3]}
    view_augmenter = ViewAugmenter(view_augmentation_configuration=configuration, image_size=IMAGE_SIZE)
    grey_images = torch.full(size=(3, 3, IMAGE_SIZE, IMAGE_SIZE), fill_value=128.0)
    torch.manual_seed(seed=0)
    first_views, second_views, view_descriptions = view_augmenter.create_views(images=grey_images,
                                                                              describe_views=True)
    for views, image_descriptions in zip([first_views, second_views], view_descriptions):
        for view, lines in zip(views, image_descriptions):
            # The erased rectangle read from its description, of whole pixels from its top left corner
            assert len(lines) == 3 and lines[1] == "gaussian_noise"
            width, height, left, top = (int(value) for value in re.fullmatch(
                pattern=r"random_erasing : (\d+)x(\d+) pixels at x (\d+), y (\d+)", string=lines[2]).groups())
            erased_pixels = torch.zeros(size=(IMAGE_SIZE, IMAGE_SIZE), dtype=torch.bool)
            erased_pixels[top:top + height, left:left + width] = True
            assert torch.equal(input=(view == 0.0).all(dim=0), other=erased_pixels)
            assert 5.0 < float(view[:, ~erased_pixels].std()) < 10.0


def test_every_transformation_describes_its_drawn_values() -> None:
    """
    Verifies the description of every transformation drawn for a view, every one of them being drawn for every image:
    the lines follow the order of the chain and give the values drawn by the transformations with random values, the
    flips, the grayscale and the Gaussian noise being named alone.
    """
    configuration = copy.deepcopy(VIEW_AUGMENTATION_CONFIGURATION)
    for transformation_name in VIEW_TRANSFORMATION_NAMES[1:]:
        configuration[transformation_name]["probability"] = 1.0
    view_augmenter = ViewAugmenter(view_augmentation_configuration=configuration, image_size=IMAGE_SIZE)
    torch.manual_seed(seed=2)
    _, _, view_descriptions = view_augmenter.create_views(images=create_test_images(number_images=2),
                                                          describe_views=True)
    expected_patterns = [r"random_resized_crop : \d+x\d+ pixels at x \d+, y \d+ \(\d+ % of the area\)",
                         "left_right_flip", "up_down_flip", r"rotation : [+-]\d+\.\d degrees",
                         r"perspective : corners moved by up to \d+\.\d pixels",
                         r"jigsaw : pieces in the order( \d){4}",
                         r"colour_jitter : brightness x\d\.\d\d, contrast x\d\.\d\d, saturation x\d\.\d\d, "
                         r"hue [+-]\d+\.\d degrees", "grayscale", r"gamma : gamma \d\.\d\d",
                         r"haze : strength \d\.\d\d, airlight \d\.\d\d",
                         r"linear_illumination : gradient up to [+-]\d\.\d\d",
                         r"plasma_shadow : intensity [+-]\d\.\d\d, quantity \d\.\d\d, roughness \d\.\d\d",
                         r"gaussian_blur : sigma \d\.\d\d",
                         r"motion_blur : kernel [3579], angle [+-]\d+\.\d degrees, direction [+-]\d\.\d\d",
                         r"solarize : threshold 0\.50, addition \+0\.00", "gaussian_noise",
                         r"random_erasing : \d+x\d+ pixels at x \d+, y \d+"]
    for image_lines in [lines for view_lines in view_descriptions for lines in view_lines]:
        assert len(image_lines) == len(expected_patterns)
        assert all(re.fullmatch(pattern=pattern, string=line) for pattern, line in zip(expected_patterns, image_lines))


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
    Verifies the views drawn on the GPU, where the self-supervised trainer draws them: every transformation keeps the
    views on the device in float32 within [0, 255]; the parameters being drawn on the CPU with the PyTorch generator,
    the same seed gives the same transformations on both devices, so that, the plasma shadow and the Gaussian noise
    (which draw their fractal map and their noise on the device) turned off, the GPU views match the CPU views up to
    the rounding of the interpolations and are described with the same lines; and they stay in float32 within the
    bfloat16 autocast of the GPU.
    """
    images = create_test_images(number_images=8)

    # Every transformation drawn on the GPU keeps the views on the device, in float32 within [0, 255]
    view_augmenter = ViewAugmenter(view_augmentation_configuration=VIEW_AUGMENTATION_CONFIGURATION,
                                   image_size=IMAGE_SIZE)
    torch.manual_seed(seed=5)
    for gpu_views in view_augmenter.create_views(images=images.to(device="cuda"))[:2]:
        assert gpu_views.device.type == "cuda" and gpu_views.dtype == torch.float32
        assert float(gpu_views.min()) >= 0.0 and float(gpu_views.max()) <= 255.0

    # The same seed on both devices, without the transformations drawing their values on the device
    configuration = copy.deepcopy(VIEW_AUGMENTATION_CONFIGURATION)
    configuration["plasma_shadow"]["probability"] = 0.0
    configuration["gaussian_noise"]["probability"] = 0.0
    view_augmenter = ViewAugmenter(view_augmentation_configuration=configuration, image_size=IMAGE_SIZE)
    drawn_views = {}
    for device_name in ["cpu", "cuda"]:
        torch.manual_seed(seed=5)
        drawn_views[device_name] = view_augmenter.create_views(images=images.to(device=device_name),
                                                               describe_views=True)

    # Same views and same descriptions
    for view_index in range(2):
        assert torch.allclose(input=drawn_views["cuda"][view_index].cpu(), other=drawn_views["cpu"][view_index],
                              atol=GPU_VIEWS_TOLERANCE)
    assert drawn_views["cuda"][2] == drawn_views["cpu"][2]

    # Float32 views within the bfloat16 autocast of a mixed precision forward pass on the GPU
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        first_views, second_views, _ = view_augmenter.create_views(
            images=images.to(device="cuda", dtype=torch.bfloat16))
    assert first_views.dtype == torch.float32 and second_views.dtype == torch.float32
