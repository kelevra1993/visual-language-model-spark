"""
Tests of DataAugmenter of vision_training/data/augmentations.py: geometric augmentations moving and removing boxes,
geometric augmentations moving the segmentation masks with nearest interpolation, photometric augmentations keeping
boxes, masks and pixel range, images without boxes, shared or per-frame draws of the sequence frames, frame shuffling
and the rejected inputs.
"""

import random
import pytest
import torch

from typing import Any, Dict, List

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.data.augmentations import (DataAugmenter, AUGMENTATION_TYPES, GEOMETRIC_AUGMENTATION_TYPES,
                                                 PHOTOMETRIC_AUGMENTATION_TYPES)

IMAGE_SIZE = 96

# Tolerance in pixels between the bright rectangle and its transformed box: exact pixel permutations must match
# exactly, while resampling (crop resize, small rotation) blurs the rectangle edges by a pixel or two
BOX_TOLERANCES = {"random_left_right_flip": 0.0, "random_up_down_flip": 0.0, "random_90_rotation": 0.0,
                  "random_crop": 2.0, "random_rotation": 2.5}


def build_configuration(enabled_types: List[str], probability: float = 1.0, independent_frames: bool = False,
                        shuffle_probability: float = 0.0) -> Dict[str, Any]:
    """
    Builds a 'Data.data_augmentation' configuration forcing the given augmentation types, so that each test
    controls exactly which transformations are applied.

    Args:
        enabled_types (List[str]): The augmentation types to activate, each with probability 1.
        probability (float): The global probability of augmenting an image. Defaults to 1.0.
        independent_frames (bool): Whether the photometric types are drawn frame by frame. Defaults to False.
        shuffle_probability (float): The probability of shuffling the frames of a window, 0 turning the shuffling
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
        images, boxes, labels, _ = augmenter.augment_batch(images=create_rectangle_image(box=box).unsqueeze(dim=0),
                                                           bounding_boxes=[torch.tensor(data=[box])],
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

    kept_boxes, kept_labels = DataAugmenter.clean_bounding_boxes(bounding_boxes=boxes, labels=labels,
                                                                 image_height=IMAGE_SIZE, image_width=IMAGE_SIZE)

    assert kept_labels.tolist() == [2, 1]
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

    augmented_images, augmented_boxes, augmented_labels, _ = augmenter.augment_batch(images=images,
                                                                                     bounding_boxes=boxes,
                                                                                     labels=labels)

    assert augmented_images.shape == images.shape
    assert float(augmented_images.min()) >= 0.0 and float(augmented_images.max()) <= 255.0
    assert all(torch.equal(input=output, other=original) for output, original in zip(augmented_boxes, boxes))
    assert all(torch.equal(input=output, other=original) for output, original in zip(augmented_labels, labels))


def test_images_without_boxes_for_classification() -> None:
    """
    Verifies the classification use case: without boxes and labels only the images are transformed.
    """
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(
        enabled_types=["random_colour_augmentation"]))
    images = torch.rand(size=(2, 3, IMAGE_SIZE, IMAGE_SIZE)) * 255.0

    augmented_images, augmented_boxes, augmented_labels, augmented_masks = augmenter.augment_batch(images=images)

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

    augmented_sequences, _, _, _ = augmenter.augment_batch(images=sequences)

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
        augmented_images, augmented_boxes, _, _ = augmenter.augment_batch(images=images, bounding_boxes=boxes,
                                                                          labels=labels)
        assert torch.equal(input=augmented_images, other=images)
        assert all(torch.equal(input=output, other=original) for output, original in zip(augmented_boxes, boxes))


def test_shuffling_permutes_the_real_frames_of_every_window() -> None:
    """
    Verifies the shuffling of the frames of the windows: only the real frames move, the padding frames staying last,
    every member window of a pair is shuffled within its own block, the frame order gives the original frame at every
    position, and nothing moves without shuffling.
    """
    torch.manual_seed(seed=0)
    random.seed(0)

    # Two pairs of member windows of 4 frames, the frame of position i filled with the value i
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
        images, _, _, masks = augmenter.augment_batch(images=create_rectangle_image(box=box).unsqueeze(dim=0),
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
    _, _, _, rotated_masks = augmenter.augment_batch(images=torch.zeros(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE)),
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

    augmented_images, _, _, augmented_masks = augmenter.augment_batch(images=images, masks=masks)

    assert not torch.equal(input=augmented_images, other=images)
    assert torch.equal(input=augmented_masks, other=masks)
