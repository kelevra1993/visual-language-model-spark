"""
Tests of vision_training/data/preprocessing.py: restore_boxes_to_original_image inverts preprocess_boxes, bringing the
detected boxes back to the original image coordinates, clipped to the image or kept whole for the pasting of the
instance masks, preprocess_boxes moves polygons like boxes,
restore_prediction_to_original_image brings segmentation class maps and probabilities back to their original images,
and the square crop of the classification and self-supervised inputs cuts the resized image without padding, centred or
at every random position along its longer side.
"""

import cv2
import numpy as np
import pytest
import random
import torch

from vision_training.data.preprocessing import (compute_letterbox_region, compute_square_crop_region, preprocess_boxes,
                                                preprocess_image, restore_boxes_to_original_image,
                                                restore_prediction_to_original_image)


@pytest.mark.parametrize(argnames=("image_width", "image_height"), argvalues=[(640, 480), (480, 640), (500, 500)])
@pytest.mark.parametrize(argnames="keep_ratio", argvalues=[True, False])
def test_restored_boxes_invert_the_preprocessing(image_width: int, image_height: int, keep_ratio: bool) -> None:
    """
    Verifies that mapping boxes from the network input back to the original image exactly inverts the box
    preprocessing, for landscape, portrait and square images, with and without aspect ratio preservation.

    Args:
        image_width (int): The width of the original image.
        image_height (int): The height of the original image.
        keep_ratio (bool): Whether the preprocessing keeps the aspect ratio.
    """
    boxes = [[10.0, 20.0, 300.0, 150.0], [200.0, 5.0, 470.0, 470.0]]

    preprocessed_boxes = preprocess_boxes(bounding_boxes=boxes, image_width=image_width, image_height=image_height,
                                          image_size=1024, keep_ratio=keep_ratio)
    restored_boxes = restore_boxes_to_original_image(bounding_boxes=np.array(preprocessed_boxes),
                                                     image_width=image_width, image_height=image_height,
                                                     image_size=1024, keep_ratio=keep_ratio)

    assert np.allclose(restored_boxes, np.array(boxes))


def test_restored_boxes_are_clipped_unless_kept_whole() -> None:
    """
    Verifies that a box predicted partly on the padding of a letterboxed image is clipped to the original image by
    default, and kept whole without clipping, as the instance segmentation evaluated at the original resolution needs
    to paste the mask of a detection into the whole box it was predicted in.
    """
    # A 200 x 100 image letterboxed into 400 pixels is scaled twice and moved down by 100 rows of padding
    network_box = np.array(object=[[-20.0, 60.0, 120.0, 140.0]])
    clipped_box = restore_boxes_to_original_image(bounding_boxes=network_box, image_width=200, image_height=100,
                                                  image_size=400, keep_ratio=True)
    whole_box = restore_boxes_to_original_image(bounding_boxes=network_box, image_width=200, image_height=100,
                                                image_size=400, keep_ratio=True, clip=False)

    assert np.allclose(clipped_box, [[0.0, 0.0, 60.0, 20.0]])
    assert np.allclose(whole_box, [[-10.0, -20.0, 60.0, 20.0]])


@pytest.mark.parametrize(argnames="keep_ratio", argvalues=[True, False])
def test_polygons_are_transformed_like_the_boxes(keep_ratio: bool) -> None:
    """
    Verifies that a flat polygon [x, y, x, y, ...] goes through exactly the scaling and padding of the boxes, every
    vertex landing where the corner of a box at the same place lands, so that the masks rasterised from polygons stay
    aligned with the boxes.

    Args:
        keep_ratio (bool): Whether the preprocessing keeps the aspect ratio.
    """
    polygon = [10.0, 20.0, 300.0, 20.0, 300.0, 150.0]
    transformed_polygon = preprocess_boxes(bounding_boxes=[polygon], image_width=640, image_height=480,
                                           image_size=512, keep_ratio=keep_ratio)[0]
    transformed_box = preprocess_boxes(bounding_boxes=[[10.0, 20.0, 300.0, 150.0]], image_width=640, image_height=480,
                                       image_size=512, keep_ratio=keep_ratio)[0]

    assert len(transformed_polygon) == 6
    assert transformed_polygon[:2] == transformed_box[:2] and transformed_polygon[4:] == transformed_box[2:]
    assert transformed_polygon[2:4] == [transformed_box[2], transformed_box[1]]


@pytest.mark.parametrize(argnames="keep_ratio", argvalues=[True, False])
def test_class_map_is_restored_to_the_original_image(keep_ratio: bool) -> None:
    """
    Verifies that a class map at the network resolution is brought back exactly to its original image when the
    resizing factors are whole numbers: the centred padding of the kept aspect ratio is cut off, and the nearest
    resizing undoes the nearest resizing of the preprocessing.

    Args:
        keep_ratio (bool): Whether the image was padded to keep its aspect ratio, or stretched.
    """
    original_class_map = np.random.default_rng(seed=0).integers(low=0, high=5, size=(16, 8)).astype(np.int64)
    network_class_map = preprocess_image(image=original_class_map.astype(np.uint8), image_size=32,
                                         keep_ratio=keep_ratio, interpolation=cv2.INTER_NEAREST,
                                         padding_value=255).astype(np.int64)

    restored_class_map = restore_prediction_to_original_image(prediction=torch.from_numpy(network_class_map),
                                                              original_height=16, original_width=8,
                                                              keep_ratio=keep_ratio)
    assert restored_class_map.dtype == torch.int64
    assert np.array_equal(restored_class_map.numpy(), original_class_map)


def test_probabilities_are_restored_without_the_padding() -> None:
    """
    Verifies that class probabilities at the network resolution are restored to a landscape original image from the
    part of the input holding the image only: a class predicted everywhere on the image and another one on the padding
    give the first class to every original pixel, the restored probabilities keeping the class axis.
    """
    start_y, start_x, resized_height, resized_width = compute_letterbox_region(image_height=30, image_width=60,
                                                                               image_size=32)
    assert (start_y, start_x, resized_height, resized_width) == (8, 0, 16, 32)

    # Class 1 on the image part of the input, class 0 on its padding
    probabilities = torch.zeros(size=(2, 32, 32))
    probabilities[0] = 1.0
    probabilities[:, start_y:start_y + resized_height, start_x:start_x + resized_width] = torch.tensor(
        data=[0.2, 0.8]).view(2, 1, 1)

    restored_probabilities = restore_prediction_to_original_image(prediction=probabilities, original_height=30,
                                                                  original_width=60, keep_ratio=True)
    assert restored_probabilities.shape == (2, 30, 60)
    assert torch.allclose(input=restored_probabilities[1], other=torch.full(size=(30, 60), fill_value=0.8))
    assert bool((restored_probabilities.argmax(dim=0) == 1).all())


@pytest.mark.parametrize(argnames=("image_height", "image_width", "expected_region"), argvalues=[
    (300, 400, (0, 42, 256, 341)), (500, 333, (64, 0, 384, 256)), (256, 256, (0, 0, 256, 256))])
def test_centred_square_crop_cuts_the_resized_image(image_height: int, image_width: int,
                                                    expected_region: tuple[int, int, int, int]) -> None:
    """
    Verifies the centred square crop of the validation, test and inference images: the shorter side is resized to the
    input size, the square is cut at the centre of the longer side, for landscape, portrait and square images, and the
    input holds the pixels of the resized image only, without the padding of the letterbox.

    Args:
        image_height (int): The height of the original image.
        image_width (int): The width of the original image.
        expected_region (tuple[int, int, int, int]): The expected top row and left column of the square and the height
            and width of the resized image.
    """
    region = compute_square_crop_region(image_height=image_height, image_width=image_width, image_size=256,
                                        random_crop_position=False)
    assert region == expected_region

    # The input is the square of the resized image, with no pixel left black by a padding
    image = np.random.default_rng(seed=0).integers(low=1, high=256, size=(image_height, image_width, 3), dtype=np.uint8)
    crop_top, crop_left, resized_height, resized_width = region
    expected_image = cv2.resize(src=image, dsize=(resized_width, resized_height),
                                interpolation=cv2.INTER_LINEAR)[crop_top:crop_top + 256, crop_left:crop_left + 256]
    cropped_image = preprocess_image(image=image, image_size=256, keep_ratio=True, square_crop=True)
    assert cropped_image.shape == (256, 256, 3)
    assert np.array_equal(cropped_image, expected_image)
    assert not (cropped_image == 0).all(axis=2).any()


def test_random_square_crop_covers_every_position() -> None:
    """
    Verifies the random square crop of the training images: the square moves along the longer side only, and every
    whole pixel position from one end to the other is drawn, so that the network sees every part of the image across
    the epochs.
    """
    # A 300 x 400 image resized to 256 x 341 leaves 85 extra columns, hence the 86 positions 0 to 85 of the square
    random.seed(a=0)
    positions = {compute_square_crop_region(image_height=300, image_width=400, image_size=256,
                                            random_crop_position=True)[:2] for _ in range(2000)}
    assert positions == {(0, crop_left) for crop_left in range(86)}


def test_square_crop_needs_keep_ratio() -> None:
    """
    Verifies that the square crop asked for a stretched image stops with an error instead of silently stretching it,
    the configuration checks refusing the same combination at launch.
    """
    with pytest.raises(expected_exception=ValueError, match="it needs keep_ratio True"):
        preprocess_image(image=np.zeros(shape=(30, 40, 3), dtype=np.uint8), image_size=32, keep_ratio=False,
                         square_crop=True)
