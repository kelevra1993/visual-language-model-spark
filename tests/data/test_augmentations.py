"""
Tests of DataAugmenter of vision_training/data/augmentations.py: geometric augmentations moving and removing boxes,
geometric augmentations moving the segmentation masks with nearest interpolation, geometric augmentations moving the
instance masks and recomputing their boxes, photometric augmentations keeping boxes, masks and pixel range, images
without boxes, a single histogram equalization per image, the haze blending every image with its own grey airlight
after the other photometric types, shared or per-frame draws of the sequence frames, frame shuffling, the rejected
inputs, the zoom-out, the translation by whole pixels with the removal of the objects it pushes out, the removal of the
fragments of objects left by the crop, the aspect crop keeping independent shares of the width and of the height and
stretching the image, its boxes and masks, the keypoints moved without being renamed and unlabelled once out of the
image, the ignored regions of the keypoint detection moved with them as binary masks, and the augmentation reports
written in the information panels of the input viewer.
"""

import random
import re
import pytest
import torch

from typing import Any, Dict, List

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.data.augmentations import (DataAugmenter, AUGMENTATION_TYPES, EXCLUSIVE_EQUALIZATION_TYPES,
                                                 GEOMETRIC_AUGMENTATION_TYPES, HAZE_AIRLIGHT_RANGE, HAZE_STRENGTH_RANGE,
                                                 MAXIMUM_TRANSLATION_FRACTION, MINIMUM_ASPECT_CROP_FRACTION,
                                                 MINIMUM_CROP_FRACTION, NOISE_STANDARD_DEVIATION,
                                                 PHOTOMETRIC_AUGMENTATION_TYPES)
from vision_training.data.detection.coco_masks import compute_mask_bounding_boxes

IMAGE_SIZE = 96

# Tolerance in pixels between the bright rectangle and its transformed box: exact pixel permutations must match
# exactly, while resampling (crop resize, small rotation) blurs the rectangle edges by a pixel or two
BOX_TOLERANCES = {"random_left_right_flip": 0.0, "random_up_down_flip": 0.0, "random_90_rotation": 0.0,
                  "random_crop": 2.0, "random_aspect_crop": 2.0, "random_zoom_out": 2.0, "random_translation": 0.0,
                  "random_rotation": 2.5}


def build_configuration(enabled_types: List[str], probability: float = 1.0, independent_frames: bool = False,
                        shuffle_probability: float = 0.0) -> Dict[str, Any]:
    """
    Builds a 'Data.data_augmentation' configuration forcing the given augmentation types, so that each test
    controls exactly which transformations are applied.

    Args:
        enabled_types (List[str]): The augmentation types to activate, each with probability 1.
        probability (float): The global probability of augmenting an image. Defaults to 1.0.
        independent_frames (bool): Whether the photometric types are drawn frame by frame. Defaults to False.
        shuffle_probability (float): The probability of shuffling the frames of a sequence, 0 turning the shuffling
            off. Defaults to 0.0.

    Returns:
        Dict[str, Any]: The augmentation configuration.
    """
    return {"activated": True, "probability": probability, "independent_frames": independent_frames,
            "shuffle_frames": {"activated": shuffle_probability > 0.0, "probability": shuffle_probability},
            "types": {name: {"activated": name in enabled_types, "probability": 1.0} for name in AUGMENTATION_TYPES}}


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


@pytest.mark.parametrize(argnames="augmentation_type", argvalues=GEOMETRIC_AUGMENTATION_TYPES)
def test_geometric_augmentation_moves_boxes_with_the_image(augmentation_type: str) -> None:
    """
    Verifies that every geometric augmentation transforms the box exactly like the image content: the returned box
    must match the position of the rectangle drawn at the original box, for several random draws.

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


@pytest.mark.parametrize(argnames="augmentation_type", argvalues=PHOTOMETRIC_AUGMENTATION_TYPES)
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


def test_a_single_histogram_equalization_is_applied_per_image() -> None:
    """
    Verifies that an image drawing several histogram equalizations (EXCLUSIVE_EQUALIZATION_TYPES) receives a single
    one of them, chosen at random so that each of them is kept for some images, the other photometric types being
    applied as before, and that an equalization drawn alone is applied.
    """
    random.seed(0)
    torch.manual_seed(seed=0)
    images = torch.rand(size=(32, 3, IMAGE_SIZE, IMAGE_SIZE)) * 255.0

    # The three equalizations and the noise all drawn for every image: one equalization kept next to the noise
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(
        enabled_types=EXCLUSIVE_EQUALIZATION_TYPES + ["random_noise"]))
    reports = augmenter.augment_batch(images=images)[-1]
    kept_equalizations = set()
    for report in reports:
        applied_names = [type_name for type_name, _ in report["photometric_augmentations"][0]]
        assert len(applied_names) == 2 and applied_names[-1] == "random_noise"
        assert applied_names[0] in EXCLUSIVE_EQUALIZATION_TYPES
        kept_equalizations.add(applied_names[0])
    assert kept_equalizations == set(EXCLUSIVE_EQUALIZATION_TYPES)

    # An equalization drawn alone is applied to every image
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["clahe_augment"]))
    reports = augmenter.augment_batch(images=images)[-1]
    assert all(report["photometric_augmentations"][0][0][0] == "clahe_augment" for report in reports)


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
    # standard deviation of 40 levels shrinks by 1 - strength around the hazy grey 100 x (1 - strength) + strength x 255
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


def test_images_without_boxes_for_classification() -> None:
    """
    Verifies the classification use case: without boxes and labels only the images are transformed.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(
        enabled_types=["random_colour_augmentation"]))
    images = torch.rand(size=(2, 3, IMAGE_SIZE, IMAGE_SIZE)) * 255.0

    augmented_images, augmented_boxes, augmented_labels, augmented_masks, _, _, _ = augmenter.augment_batch(
        images=images)

    assert augmented_boxes is None and augmented_labels is None and augmented_masks is None
    assert torch.equal(input=augmented_images, other=images.flip(dims=(1,)))


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


def test_invalid_inputs_are_rejected() -> None:
    """
    Verifies that unsupported inputs fail loudly: boxes with sequences, boxes without labels, and 90 degree
    rotations of non-square images.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["random_90_rotation"]))
    sequences = torch.zeros(size=(1, 2, 3, IMAGE_SIZE, IMAGE_SIZE))
    boxes = [torch.tensor(data=[[10.0, 10.0, 20.0, 20.0]])]

    with pytest.raises(expected_exception=ValueError):
        augmenter.augment_batch(images=sequences, bounding_boxes=boxes, labels=[torch.tensor(data=[1])])
    with pytest.raises(expected_exception=ValueError):
        augmenter.augment_batch(images=sequences, masks=torch.zeros(size=(1, IMAGE_SIZE, IMAGE_SIZE),
                                                                   dtype=torch.int64))
    with pytest.raises(expected_exception=ValueError):
        augmenter.augment_batch(images=sequences[:, 0], bounding_boxes=boxes)
    with pytest.raises(expected_exception=ValueError):
        augmenter.augment_batch(images=torch.zeros(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE + 8)))


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
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=[],
                                                                             shuffle_probability=1.0))
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
        enabled_types=[])).shuffle_frames(images=frames, sequence_mask=sequence_mask, frame_block_length=4)
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
        enabled_types=["random_left_right_flip", "brightness_contrast_saturation"], independent_frames=True))
    augmented_sequence = independent_augmenter.augment_batch(images=sequences)[0][0]
    right_half_means = augmented_sequence[:, :, :, IMAGE_SIZE // 2:].mean(dim=(1, 2, 3))
    left_half_means = augmented_sequence[:, :, :, :IMAGE_SIZE // 2].mean(dim=(1, 2, 3))
    assert bool((right_half_means > left_half_means).all())
    assert len({round(float(mean_value), 3) for mean_value in right_half_means}) > 1

    # Shared draws keep the frames identical
    shared_augmenter = DataAugmenter(augmentation_configuration=build_configuration(
        enabled_types=["random_left_right_flip", "brightness_contrast_saturation"]))
    shared_sequence = shared_augmenter.augment_batch(images=sequences)[0][0]
    assert all(torch.allclose(input=sequence_frame, other=shared_sequence[0]) for sequence_frame in shared_sequence)


@pytest.mark.parametrize(argnames="augmentation_type", argvalues=GEOMETRIC_AUGMENTATION_TYPES)
def test_geometric_augmentation_moves_masks_with_the_image(augmentation_type: str) -> None:
    """
    Verifies that every geometric augmentation moves the segmentation mask exactly like the image: the pixels of the
    class of a bright rectangle stay on the rectangle (exactly for the pixel permutations, within the resampling
    tolerance otherwise), and the mask only holds class indices or the ignored index, never interpolated values.

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


@pytest.mark.parametrize(argnames="augmentation_type", argvalues=GEOMETRIC_AUGMENTATION_TYPES)
def test_geometric_augmentation_moves_instance_masks_and_recomputes_boxes(augmentation_type: str) -> None:
    """
    Verifies that every geometric augmentation moves the binary instance masks with the image (the mask following the
    rectangle drawn at its object), that they stay binary uint8, and that the box of every object becomes the tight box
    of its moved mask.

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
        assert torch.equal(input=boxes[0], other=compute_mask_bounding_boxes(
            instance_masks=moved_masks[0]))


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
    augmentation keeps an image without objects at [0, H, W] masks, and that instance masks without boxes or with
    class masks are refused.
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
        enabled_types=GEOMETRIC_AUGMENTATION_TYPES))
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
        image, boxes, mask, area_scale, _ = augmenter.random_zoom_out(image=create_rectangle_image(box=box),
                                                                      bounding_boxes=torch.tensor(data=[box]),
                                                                      mask=class_mask)

        # The box follows the shrunk rectangle, and its area shrinks by the area scale
        assert torch.allclose(input=boxes[0], other=torch.tensor(data=get_bright_region_box(image=image)), atol=2.0)
        box_area = float((boxes[0, 2] - boxes[0, 0]) * (boxes[0, 3] - boxes[0, 1]))
        assert 0.25 <= area_scale <= 1.0 and box_area == pytest.approx(expected=30.0 * 40.0 * area_scale, rel=0.05)

        # The class mask keeps its class on the shrunk image and is ignored on the canvas around it
        assert int((mask == 1).sum()) == pytest.approx(expected=IMAGE_SIZE * IMAGE_SIZE * area_scale, rel=1e-6)
        assert set(mask.unique().tolist()) <= {1, IGNORED_LABEL_INDEX}

        # The instance masks stay binary, background around the shrunk image
        random.seed(seed)
        _, _, moved_masks, _, _ = augmenter.random_zoom_out(image=create_rectangle_image(box=box),
                                                            bounding_boxes=torch.tensor(data=[box]),
                                                            mask=instance_masks)
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
        image, boxes, mask, area_scale, _ = augmenter.random_translation(image=create_rectangle_image(box=box),
                                                                         bounding_boxes=torch.tensor(data=[box]),
                                                                         mask=class_mask)

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
        _, _, moved_masks, _, _ = augmenter.random_translation(image=create_rectangle_image(box=box),
                                                               bounding_boxes=torch.tensor(data=[box]),
                                                               mask=instance_masks)
        assert moved_masks.dtype == torch.uint8 and set(moved_masks.unique().tolist()) <= {0, 1}
        assert torch.equal(input=compute_mask_bounding_boxes(instance_masks=moved_masks), other=boxes)
        assert int(moved_masks.sum()) == int(instance_masks.sum())

    # The draws cover the width only, the height only and both axes
    assert {(True, False), (False, True), (True, True)} <= observed_axes


def test_objects_translated_out_of_the_image_are_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the largest translation along the width (a quarter of the side) keeps an object that stays inside
    the image and removes, with its label, an object at the right border pushed out of it, while the left border
    uncovered by the shift is black.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture forcing the axes and the shift of the translation.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["random_translation"]))
    monkeypatch.setattr(target=random, name="choice", value=lambda sequence: "x")
    monkeypatch.setattr(target=random, name="randint", value=lambda low, high: high)
    maximum_shift = int(IMAGE_SIZE * MAXIMUM_TRANSLATION_FRACTION)

    images, boxes, labels, _, _, _, _ = augmenter.augment_batch(
        images=create_rectangle_image(box=[10.0, 10.0, 30.0, 30.0]).unsqueeze(dim=0),
        bounding_boxes=[torch.tensor(data=[[10.0, 10.0, 30.0, 30.0], [80.0, 40.0, 95.0, 60.0]])],
        labels=[torch.tensor(data=[1, 2])])

    # The first object moves right by the largest shift, the second one leaves the image with its label
    assert labels[0].tolist() == [1]
    assert boxes[0].tolist() == [[10.0 + maximum_shift, 10.0, 30.0 + maximum_shift, 30.0]]
    assert float(images[0][:, :, :maximum_shift].abs().sum()) == 0.0


def test_fragments_of_objects_are_removed_with_their_labels() -> None:
    """
    Verifies that a box keeping less than a quarter of its area inside the image is removed, while one keeping at least
    a quarter is clipped and kept, and that the shares measured on the instance masks replace those of the boxes.
    """
    boxes = torch.tensor(data=[[-30.0, 10.0, 10.0, 40.0], [-35.0, 10.0, 5.0, 40.0], [10.0, 10.0, 40.0, 40.0]])
    labels = torch.tensor(data=[3, 2, 1])

    # 10 of 40 pixels of width inside the image is a quarter, kept, 5 of 40 is removed
    kept_boxes, kept_labels, _, _, _ = DataAugmenter.clean_bounding_boxes(
        bounding_boxes=boxes, labels=labels, image_height=IMAGE_SIZE, image_width=IMAGE_SIZE)
    assert kept_labels.tolist() == [3, 1]
    assert kept_boxes.tolist() == [[0.0, 10.0, 10.0, 40.0], [10.0, 10.0, 40.0, 40.0]]

    # The shares of the mask pixels decide when they are given, whatever the boxes
    instance_masks = torch.zeros(size=(3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
    _, kept_labels, kept_masks, _, _ = DataAugmenter.clean_bounding_boxes(
        bounding_boxes=boxes, labels=labels, image_height=IMAGE_SIZE, image_width=IMAGE_SIZE,
        instance_masks=instance_masks, visible_fractions=torch.tensor(data=[0.1, 0.9, 0.5]))
    assert kept_labels.tolist() == [2, 1] and kept_masks.shape[0] == 2


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


def test_report_lines_of_sequences_and_of_unaugmented_samples() -> None:
    """
    Verifies the panel lines of the reports in the input viewer: one photometric line per real frame of a sequence
    drawn frame by frame, with its drawn values and without the padding frames, and a single line for a sample left
    unaugmented by the global probability or with the augmentation off.
    """
    random.seed(0)
    torch.manual_seed(seed=0)
    sequences = torch.full(size=(1, 4, 3, IMAGE_SIZE, IMAGE_SIZE), fill_value=100.0)

    # Every frame gets its own brightness draw, the padding frames 2 and 3 being left out of the lines
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(
        enabled_types=["brightness_contrast_saturation"], independent_frames=True))
    _, _, _, _, _, _, reports = augmenter.augment_batch(images=sequences)
    assert len(reports[0]["photometric_augmentations"]) == 4 and reports[0]["removed_labels"] is None
    lines = augmenter.describe_report(report=reports[0], real_frames=[0, 1])
    assert lines[:3] == ["Geometric :", "  none", "Photometric, drawn frame by frame :"] and len(lines) == 5
    assert all(line.startswith(f"  Frame {frame_index} : brightness_contrast_saturation (brightness ")
               for frame_index, line in zip([0, 1], lines[3:]))

    # A sample not drawn by the global probability, and an augmenter turned off
    unaugmented_augmenter = DataAugmenter(augmentation_configuration=build_configuration(
        enabled_types=["brightness_contrast_saturation"], probability=0.0))
    _, _, _, _, _, _, reports = unaugmented_augmenter.augment_batch(images=sequences[:, 0])
    assert unaugmented_augmenter.describe_report(report=reports[0]) == [
        "Not augmented (global probability 0.0 not drawn)"]
    deactivated_augmenter = DataAugmenter(augmentation_configuration={
        **build_configuration(enabled_types=["brightness_contrast_saturation"]), "activated": False})
    _, _, _, _, _, _, reports = deactivated_augmenter.augment_batch(images=sequences[:, 0])
    assert deactivated_augmenter.describe_report(report=reports[0]) == [
        "Augmentation off (data_augmentation.activated False)"]


@pytest.mark.parametrize(argnames="augmentation_type", argvalues=GEOMETRIC_AUGMENTATION_TYPES)
def test_geometric_augmentation_moves_keypoints_with_the_image(augmentation_type: str) -> None:
    """
    Verifies that every geometric augmentation moves the keypoints of the keypoint detection like the image content:
    a keypoint at the centre of the bright rectangle must land at the centre of the moved rectangle, keeping its
    visibility, and the box of its object must follow the rectangle, for several random draws.

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


@pytest.mark.parametrize(argnames="augmentation_type", argvalues=GEOMETRIC_AUGMENTATION_TYPES)
def test_geometric_augmentation_moves_the_ignored_regions_with_the_keypoints(augmentation_type: str) -> None:
    """
    Verifies that every geometric augmentation moves the ignored regions of the keypoint detection, given as the
    binary masks of the augmenter together with the boxes and keypoints of the objects, with the image: the region
    drawn on the rectangle follows it and stays binary uint8, the border uncovered by a zoom-out, a translation or a
    rotation being left out of it.

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


def test_keypoints_need_their_boxes() -> None:
    """
    Verifies that keypoints given without the boxes of their objects, or with sequences, are rejected.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(enabled_types=["random_crop"]))
    keypoints = [torch.zeros(size=(1, 2, 3))]

    with pytest.raises(expected_exception=ValueError, match="Keypoints must be given"):
        augmenter.augment_batch(images=torch.zeros(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE)), keypoints=keypoints)
    with pytest.raises(expected_exception=ValueError, match="not supported for image sequences"):
        augmenter.augment_batch(images=torch.zeros(size=(1, 2, 3, IMAGE_SIZE, IMAGE_SIZE)), keypoints=keypoints)
