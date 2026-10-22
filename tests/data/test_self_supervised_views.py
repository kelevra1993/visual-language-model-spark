"""
Tests of the two random views of the self-supervised task, drawn by DataAugmenter.create_views of
vision_training/data/augmentations.py from the types of 'Data.data_augmentation' with the self-supervised strengths,
right before the forward pass of the self-supervised trainer: the views keep the shape, the float32 type and the
[0, 255] range of the images and differ, every type turned off gives the images back, the global probability is
ignored, the types of SECOND_VIEW_ONLY_TYPES only reach the second view while the supervised tasks draw them for every
image, the views follow the seeds and stay in float32 under the bfloat16 autocast, the supervised augmenter and inputs
other than a batch are refused, the two strength sets hold every strength with one setting of each pair of the crop and
of the brightness, the random resized crop of the views draws an area and an aspect ratio within their ranges and
falls back on the whole image, and on the GPU the views of the self-supervised configuration never make the host wait
for the device, but for the types of HOST_WAITING_TYPES.
"""

import random
import re
import pytest
import torch
import yaml

from pathlib import Path
from typing import List, Optional

from vision_training.data.augmentations import (SECOND_VIEW_ONLY_TYPES, SELF_SUPERVISED_AUGMENTATION_STRENGTHS,
                                                 SUPERVISED_AUGMENTATION_STRENGTHS, AugmentationStrengths,
                                                 DataAugmenter)
from tests.data.test_augmentations import build_configuration

# Side of the square images of the tests
IMAGE_SIZE = 32

# Self-supervised configuration of the synthetic shapes, whose types draw the views of the tests
SYNTHETIC_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" / "self_supervised" /
                                "configuration_synthetic_shapes.yaml")

# Types making the host wait for the device at every image they change, for every task: the rotation, whose matrix
# torchvision copies from the host, the CLAHE, the sharpening and the histogram equalizations
HOST_WAITING_TYPES = ["random_rotation", "clahe_augment", "sharp_augment", "equalize_histogram_augment",
                      "equalize_bgr_histogram_augment"]


def create_test_images(number_images: int, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """
    Creates float BGR images in [0, 255] of random whole levels, as the loaders deliver them.

    Args:
        number_images (int): The number of images of the batch.
        image_size (int): The side of the square images. Defaults to IMAGE_SIZE.

    Returns:
        torch.Tensor: The [N, 3, image_size, image_size] images.
    """
    generator = torch.Generator().manual_seed(0)
    return torch.randint(low=0, high=256, size=(number_images, 3, image_size, image_size),
                         generator=generator).to(dtype=torch.float32)


def build_views_augmenter(turned_off_types: Optional[List[str]] = None) -> DataAugmenter:
    """
    Builds the augmenter of the views of the self-supervised configuration of the synthetic shapes, as the general
    trainer builds it for the self-supervised task.

    Args:
        turned_off_types (Optional[List[str]]): The types turned off in the configuration. Defaults to None, in
            which case every type keeps its setting.

    Returns:
        DataAugmenter: The augmenter with the self-supervised strengths.
    """
    augmentation_configuration = yaml.safe_load(stream=SYNTHETIC_CONFIGURATION_PATH.read_text(
        encoding="utf-8"))["Data"]["data_augmentation"]
    for augmentation_type in turned_off_types or []:
        augmentation_configuration["types"][augmentation_type]["activated"] = False
    return DataAugmenter(augmentation_configuration=augmentation_configuration, self_supervised=True)


def list_drawn_types(reports: List[dict]) -> List[List[str]]:
    """
    Lists the types drawn for every view of a batch, geometric then photometric, from the reports of create_views.

    Args:
        reports (List[dict]): The augmentation reports of the views.

    Returns:
        List[List[str]]: The names of the types drawn for every view.
    """
    return [[name for name, _ in report["geometric_augmentations"] + report["photometric_augmentations"][0]]
            for report in reports]


def test_views_keep_the_images_format_and_differ() -> None:
    """
    Verifies that the two views of a batch drawn with the types of the self-supervised configuration keep the shape,
    the float32 type and the [0, 255] range of the images, that every view is augmented and differs from its image, and
    that the two views of the batch differ.
    """
    images = create_test_images(number_images=4)
    random.seed(0)
    torch.manual_seed(seed=0)
    first_views, second_views, first_reports, second_reports = build_views_augmenter().create_views(images=images)
    for views, reports in [(first_views, first_reports), (second_views, second_reports)]:
        assert views.shape == images.shape and views.dtype == torch.float32
        assert float(views.min()) >= 0.0 and float(views.max()) <= 255.0 and not bool(torch.isnan(input=views).any())
        assert not torch.allclose(input=views, other=images)
        assert len(reports) == 4 and all(report["augmented"] for report in reports)
    assert not torch.allclose(input=first_views, other=second_views)


def test_views_without_any_type_are_the_images() -> None:
    """
    Verifies that turning every type off gives both views equal to the images, their reports listing no type.
    """
    images = create_test_images(number_images=3)
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=[]), self_supervised=True)
    first_views, second_views, first_reports, second_reports = augmenter.create_views(images=images)
    assert torch.equal(input=first_views, other=images) and torch.equal(input=second_views, other=images)
    assert augmenter.describe_report(report=first_reports[0]) == ["Geometric :", "  none", "Photometric :", "  none"]
    assert list_drawn_types(reports=first_reports + second_reports) == [[]] * 6


def test_views_ignore_the_global_probability() -> None:
    """
    Verifies that the views ignore the global probability, the share of the training images of the supervised tasks
    augmented at all: with a probability of 0, which leaves every image of a supervised batch unchanged, both views of
    every image still go through their types, here a grayscale conversion of probability 1.
    """
    images = create_test_images(number_images=3)
    configuration = build_configuration(enabled_types=["gray_scale_augment"], probability=0.0)
    first_views, second_views, _, _ = DataAugmenter(augmentation_configuration=configuration,
                                                    self_supervised=True).create_views(images=images)
    for views in [first_views, second_views]:
        assert torch.allclose(input=views, other=views[:, :1].expand_as(other=views))
        assert not torch.allclose(input=views, other=images)
    supervised_images = DataAugmenter(augmentation_configuration=configuration).augment_batch(images=images)[0]
    assert torch.equal(input=supervised_images, other=images)


def test_second_view_only_types_never_reach_the_first_view() -> None:
    """
    Verifies the types of SECOND_VIEW_ONLY_TYPES, the solarization and the black-out, drawn for the second view only
    as in the asymmetric views of BYOL and Barlow Twins: at a probability of 1, the first view of every image is the
    image itself and its second view is solarized and blacked out, both types being named in the reports of the second
    views only, while a supervised augmenter applies both to every image of its batch.
    """
    images = create_test_images(number_images=4)
    configuration = build_configuration(enabled_types=SECOND_VIEW_ONLY_TYPES)

    # Images solarized at the threshold of each strength set, the views inverting the bright pixels only
    solarized_images, supervised_solarized_images = [
        torch.where(condition=images < strengths["solarize_threshold"], input=images, other=255.0 - images)
        for strengths in [SELF_SUPERVISED_AUGMENTATION_STRENGTHS, SUPERVISED_AUGMENTATION_STRENGTHS]]

    # The first views untouched, the second views solarized with black cells
    random.seed(0)
    first_views, second_views, first_reports, second_reports = DataAugmenter(
        augmentation_configuration=configuration, self_supervised=True).create_views(images=images)
    assert torch.equal(input=first_views, other=images)
    assert list_drawn_types(reports=first_reports) == [[]] * 4
    assert list_drawn_types(reports=second_reports) == [SECOND_VIEW_ONLY_TYPES] * 4
    assert bool(((second_views == solarized_images) | (second_views == 0.0)).all())
    assert bool(((second_views == 0.0) & (solarized_images != 0.0)).any())

    # Both types drawn for every image of a supervised batch
    random.seed(0)
    supervised_images, _, _, _, _, _, supervised_reports = DataAugmenter(
        augmentation_configuration=configuration).augment_batch(images=images)
    assert list_drawn_types(reports=supervised_reports) == [SECOND_VIEW_ONLY_TYPES] * 4
    assert bool(((supervised_images == supervised_solarized_images) | (supervised_images == 0.0)).all())


def test_views_follow_the_seeds() -> None:
    """
    Verifies that the same seeds of Python and of PyTorch give the same views and reports, the parameters of the types
    being drawn by the generator of Python and the noise and the plasma maps by the generator of PyTorch, so that the
    self-supervised trainer draws the same views of a sample image at every evaluation.
    """
    images = create_test_images(number_images=4)
    augmenter = build_views_augmenter()
    drawn_views = []
    for _ in range(2):
        random.seed(7)
        torch.manual_seed(seed=7)
        drawn_views.append(augmenter.create_views(images=images))
    assert torch.equal(input=drawn_views[0][0], other=drawn_views[1][0])
    assert torch.equal(input=drawn_views[0][1], other=drawn_views[1][1])
    assert drawn_views[0][2] == drawn_views[1][2] and drawn_views[0][3] == drawn_views[1][3]


def test_views_stay_in_float32_under_mixed_precision() -> None:
    """
    Verifies that bfloat16 images within the bfloat16 autocast of a mixed precision forward pass give float32 views,
    the interpolations of the crops, rotations and perspectives and the blurs keeping their precision.
    """
    images = create_test_images(number_images=2).to(dtype=torch.bfloat16)
    augmenter = build_views_augmenter()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        first_views, second_views, _, _ = augmenter.create_views(images=images)
    assert first_views.dtype == torch.float32 and second_views.dtype == torch.float32
    assert first_views.shape == images.shape and second_views.shape == images.shape


def test_views_are_refused_to_a_supervised_augmenter_and_for_a_single_image() -> None:
    """
    Verifies that create_views refuses an augmenter built for a supervised task, whose strengths are not those of the
    views, and an image given without its batch dimension.
    """
    with pytest.raises(expected_exception=ValueError, match="self_supervised=True"):
        DataAugmenter(augmentation_configuration=build_configuration(enabled_types=[])).create_views(
            images=create_test_images(number_images=2))
    with pytest.raises(expected_exception=ValueError, match=re.escape("[B, C, H, W]")):
        DataAugmenter(augmentation_configuration=build_configuration(enabled_types=[]), self_supervised=True
                      ).create_views(images=create_test_images(number_images=1)[0])


def test_strength_sets_hold_every_strength_and_the_augmenter_takes_the_set_of_its_task() -> None:
    """
    Verifies the two hard-coded strength sets: each holds exactly the strengths of AugmentationStrengths, the crop and
    the brightness of each set using one setting of their pair and None for the other (a share of each side and a shift
    of levels for the supervised tasks, a share of the area with an aspect ratio and a factor for the views, as the
    random resized crop and the colour jitter of SimCLR), and an augmenter takes the set of its task.
    """
    for strengths in [SUPERVISED_AUGMENTATION_STRENGTHS, SELF_SUPERVISED_AUGMENTATION_STRENGTHS]:
        assert set(strengths) == set(AugmentationStrengths.__annotations__)

    # One setting of each pair, the other being None
    assert SUPERVISED_AUGMENTATION_STRENGTHS["crop_side_range"] is not None
    assert SUPERVISED_AUGMENTATION_STRENGTHS["crop_area_range"] is None
    assert SUPERVISED_AUGMENTATION_STRENGTHS["crop_aspect_ratio_range"] is None
    assert SUPERVISED_AUGMENTATION_STRENGTHS["brightness_maximum_delta"] is not None
    assert SUPERVISED_AUGMENTATION_STRENGTHS["brightness_factor_range"] is None
    assert SELF_SUPERVISED_AUGMENTATION_STRENGTHS["crop_side_range"] is None
    assert SELF_SUPERVISED_AUGMENTATION_STRENGTHS["crop_area_range"] is not None
    assert SELF_SUPERVISED_AUGMENTATION_STRENGTHS["crop_aspect_ratio_range"] is not None
    assert SELF_SUPERVISED_AUGMENTATION_STRENGTHS["brightness_maximum_delta"] is None
    assert SELF_SUPERVISED_AUGMENTATION_STRENGTHS["brightness_factor_range"] is not None

    # The set of the task of the augmenter
    assert DataAugmenter(augmentation_configuration=build_configuration(
        enabled_types=[])).strengths is SUPERVISED_AUGMENTATION_STRENGTHS
    assert DataAugmenter(augmentation_configuration=build_configuration(enabled_types=[]),
                         self_supervised=True).strengths is SELF_SUPERVISED_AUGMENTATION_STRENGTHS


def test_crop_of_the_views_draws_an_area_and_an_aspect_ratio_within_their_ranges() -> None:
    """
    Verifies the random resized crop of the views, random_crop with the self-supervised strengths: every crop keeps a
    share of the image area within crop_area_range and a width / height ratio within crop_aspect_ratio_range (up to
    the rounding of the sides to whole pixels), both varying from view to view, and is resized back to the input size,
    while the crop of the supervised tasks keeps the same share of both sides.
    """
    image_size = 64
    images = create_test_images(number_images=32, image_size=image_size)
    configuration = build_configuration(enabled_types=["random_crop"])
    random.seed(0)
    first_views, second_views, first_reports, second_reports = DataAugmenter(
        augmentation_configuration=configuration, self_supervised=True).create_views(images=images)
    assert first_views.shape == images.shape and second_views.shape == images.shape

    # The area and the aspect ratio of every crop, read from the reports
    minimum_area, maximum_area = SELF_SUPERVISED_AUGMENTATION_STRENGTHS["crop_area_range"]
    minimum_ratio, maximum_ratio = SELF_SUPERVISED_AUGMENTATION_STRENGTHS["crop_aspect_ratio_range"]
    areas, ratios = [], []
    for report in first_reports + second_reports:
        width, height = (int(side) for side in re.fullmatch(
            pattern=r"(\d+)x(\d+) pixels \(\d+ % of the area\), at top \d+ and left \d+ pixels",
            string=dict(report["geometric_augmentations"])["random_crop"]).groups())
        areas.append(width * height / image_size ** 2)
        ratios.append(width / height)
    assert all(minimum_area - 0.02 <= area <= maximum_area for area in areas) and len(set(areas)) > 32
    assert all(minimum_ratio - 0.05 <= ratio <= maximum_ratio + 0.05 for ratio in ratios) and len(set(ratios)) > 32

    # The supervised crop keeps a share of each side, the aspect ratio of the image
    random.seed(0)
    _, _, _, _, _, _, supervised_reports = DataAugmenter(augmentation_configuration=configuration).augment_batch(
        images=images)
    assert all(re.fullmatch(pattern=r"\d+ % of each side, at top \d+ and left \d+ pixels",
                            string=dict(report["geometric_augmentations"])["random_crop"])
               for report in supervised_reports)


def test_crop_of_the_views_falls_back_on_the_whole_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the fallback of the random resized crop of the views, as in torchvision: when none of its attempts draws a
    region fitting in the image (here the whole area at the widest aspect ratio), the crop takes the central region of
    the nearest ratio of the range, the whole square image, whose ratio lies within the range.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture drawing the upper bound of every range.
    """
    monkeypatch.setattr(target=random, name="uniform", value=lambda low, high: high)
    images = create_test_images(number_images=1)
    first_views, _, first_reports, _ = DataAugmenter(augmentation_configuration=build_configuration(
        enabled_types=["random_crop"]), self_supervised=True).create_views(images=images)
    assert first_reports[0]["geometric_augmentations"] == [
        ("random_crop", f"{IMAGE_SIZE}x{IMAGE_SIZE} pixels (100 % of the area), at top 0 and left 0 pixels")]
    assert torch.allclose(input=first_views, other=images, atol=1e-3)


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="The views of the training run on the GPU.")
def test_views_on_the_gpu_never_make_the_host_wait() -> None:
    """
    Verifies the views of the self-supervised configuration of the synthetic shapes on the GPU, where the trainer draws
    them: drawing both views of a batch with the types of the configuration, but those of HOST_WAITING_TYPES, never
    makes the host wait for the device, which the synchronization debug mode of PyTorch turns into an error, and the
    views stay on the device.
    """
    augmenter = build_views_augmenter(turned_off_types=HOST_WAITING_TYPES)
    device_images = create_test_images(number_images=8).to(device="cuda")
    augmenter.create_views(images=device_images)
    torch.cuda.synchronize()

    # Both views of the batch with every synchronization of the host turned into an error
    random.seed(0)
    torch.cuda.set_sync_debug_mode(debug_mode="error")
    try:
        first_views, second_views, _, _ = augmenter.create_views(images=device_images)
    finally:
        torch.cuda.set_sync_debug_mode(debug_mode="default")
    assert first_views.device.type == "cuda" and second_views.device.type == "cuda"
    assert first_views.shape == device_images.shape and second_views.shape == device_images.shape
