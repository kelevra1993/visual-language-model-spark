"""
Tests of vision_training/data/preprocessing.py: restore_boxes_to_original_image inverts preprocess_boxes, bringing the
detected boxes back to the original image coordinates, and restore_prediction_to_original_image brings segmentation
class maps and probabilities back to their original images.
"""

import cv2
import numpy as np
import pytest
import torch

from vision_training.data.preprocessing import (compute_letterbox_region, preprocess_boxes, preprocess_image,
                                                restore_boxes_to_original_image,
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
