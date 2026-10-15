"""
Tests of DataAugmenter of vision_training/data/augmentations.py: geometric augmentations moving and removing boxes,
geometric augmentations moving the segmentation masks with nearest interpolation, geometric augmentations moving the
instance masks and recomputing their boxes, photometric augmentations keeping boxes, masks and pixel range, images
without boxes, shared or per-frame draws of the sequence frames, frame shuffling, the rejected inputs, the zoom-out
and the removal of the fragments of objects left by the crop.
"""

import random
import pytest
import torch

from typing import Any, Dict, List

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.data.augmentations import (DataAugmenter, AUGMENTATION_TYPES, GEOMETRIC_AUGMENTATION_TYPES,
                                                 MINIMUM_CROP_FRACTION, PHOTOMETRIC_AUGMENTATION_TYPES)
from vision_training.data.detection.coco_masks import compute_mask_bounding_boxes

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
        images, boxes, labels, _, _ = augmenter.augment_batch(images=create_rectangle_image(box=box).unsqueeze(dim=0),
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

    kept_boxes, kept_labels, _ = DataAugmenter.clean_bounding_boxes(bounding_boxes=boxes, labels=labels,
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

    augmented_images, augmented_boxes, augmented_labels, _, _ = augmenter.augment_batch(images=images,
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

    augmented_images, augmented_boxes, augmented_labels, augmented_masks, _ = augmenter.augment_batch(images=images)

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

    augmented_sequences, _, _, _, _ = augmenter.augment_batch(images=sequences)

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
        augmented_images, augmented_boxes, _, _, _ = augmenter.augment_batch(images=images, bounding_boxes=boxes,
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
        images, _, _, masks, _ = augmenter.augment_batch(images=create_rectangle_image(box=box).unsqueeze(dim=0),
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
    _, _, _, rotated_masks, _ = augmenter.augment_batch(images=torch.zeros(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE)),
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

    augmented_images, _, _, augmented_masks, _ = augmenter.augment_batch(images=images, masks=masks)

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
    instance_masks[1, 10:20, 60:80] = 1

    for seed in range(5):
        random.seed(seed)
        torch.manual_seed(seed)
        images, boxes, labels, masks, moved_masks = augmenter.augment_batch(
            images=create_rectangle_image(box=box).unsqueeze(dim=0),
            bounding_boxes=[torch.tensor(data=[box, [60.0, 10.0, 80.0, 20.0]])], labels=[torch.tensor(data=[1, 2])],
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
    _, boxes, _, _, rotated_masks = augmenter.augment_batch(
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
    kept_boxes, kept_labels, kept_masks = DataAugmenter.clean_bounding_boxes(
        bounding_boxes=mask_boxes, labels=torch.tensor(data=[3, 4]), image_height=IMAGE_SIZE,
        image_width=IMAGE_SIZE, instance_masks=instance_masks)
    assert kept_labels.tolist() == [4] and torch.equal(input=kept_masks, other=instance_masks[1:])

    # An image without objects goes through every geometric augmentation
    augmenter = DataAugmenter(augmentation_configuration=build_configuration(
        enabled_types=GEOMETRIC_AUGMENTATION_TYPES))
    _, boxes, labels, _, moved_masks = augmenter.augment_batch(
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
        image, boxes, mask, area_scale = augmenter.random_zoom_out(image=create_rectangle_image(box=box),
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
        _, _, moved_masks, _ = augmenter.random_zoom_out(image=create_rectangle_image(box=box),
                                                         bounding_boxes=torch.tensor(data=[box]), mask=instance_masks)
        assert moved_masks.dtype == torch.uint8 and set(moved_masks.unique().tolist()) <= {0, 1}
        assert int(moved_masks.sum()) == int((mask == 1).sum())


def test_fragments_of_objects_are_removed_with_their_labels() -> None:
    """
    Verifies that a box keeping less than a quarter of its area inside the image is removed, while one keeping at least
    a quarter is clipped and kept, and that the shares measured on the instance masks replace those of the boxes.
    """
    boxes = torch.tensor(data=[[-30.0, 10.0, 10.0, 40.0], [-35.0, 10.0, 5.0, 40.0], [10.0, 10.0, 40.0, 40.0]])
    labels = torch.tensor(data=[3, 2, 1])

    # 10 of 40 pixels of width inside the image is a quarter, kept, 5 of 40 is removed
    kept_boxes, kept_labels, _ = DataAugmenter.clean_bounding_boxes(bounding_boxes=boxes, labels=labels,
                                                                    image_height=IMAGE_SIZE, image_width=IMAGE_SIZE)
    assert kept_labels.tolist() == [3, 1]
    assert kept_boxes.tolist() == [[0.0, 10.0, 10.0, 40.0], [10.0, 10.0, 40.0, 40.0]]

    # The shares of the mask pixels decide when they are given, whatever the boxes
    instance_masks = torch.zeros(size=(3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
    _, kept_labels, kept_masks = DataAugmenter.clean_bounding_boxes(
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
    _, kept_boxes, kept_labels, _, _ = augmenter.augment_batch(images=torch.zeros(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE)),
                                                               bounding_boxes=[boxes],
                                                               labels=[torch.tensor(data=[2, 1])])
    assert kept_labels[0].tolist() == [1] and kept_boxes[0].shape == (1, 4)

    # Instance segmentation: the mask of the fragment keeps 1/7 of the pixels it would have at the zoom of the crop
    _, kept_boxes, kept_labels, _, kept_masks = augmenter.augment_batch(
        images=torch.zeros(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE)), bounding_boxes=[boxes],
        labels=[torch.tensor(data=[2, 1])], instance_masks=[instance_masks])
    assert kept_labels[0].tolist() == [1] and kept_masks[0].shape == (1, IMAGE_SIZE, IMAGE_SIZE)
