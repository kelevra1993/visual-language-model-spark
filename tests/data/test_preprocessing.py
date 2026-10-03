"""
Tests of vision_training/data/preprocessing.py: restore_boxes_to_original_image inverts preprocess_boxes, bringing the
detected boxes back to the original image coordinates.
"""

import numpy as np
import pytest

from vision_training.data.preprocessing import preprocess_boxes, restore_boxes_to_original_image


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
