"""
Tests of the evaluation drawings of vision_training/utilities/visualization.py: the class colours are distinct with
an unpainted background, the mask overlay paints the classes and grays the ignored pixels only, the error map gives
every kind of pixel its colour, the segmentation sample grid lays out its four panels and the detection sample grid its
two panels between their caption and legend bars, and the caption bar is written in the given colour.
"""

import numpy as np

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.utilities.visualization import (CAPTION_BAR_HEIGHT, CORRECT_PREDICTION_COLOUR, GRID_GAP,
                                                     IGNORED_PIXEL_COLOUR, LEGEND_BAR_HEIGHT, WRONG_PREDICTION_COLOUR,
                                                     add_caption_bar, build_class_colours,
                                                     build_detection_sample_grid, build_segmentation_error_map,
                                                     build_segmentation_sample_grid, overlay_segmentation_mask)


def build_image_and_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds a gray 4 x 4 image with a true mask holding the background, two classes and an ignored pixel, and a
    prediction holding one pixel of every kind: right class, wrong class, class missed as background, background
    predicted as a class.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The [4, 4, 3] uint8 image, the [4, 4] true mask and the [4, 4]
            predicted class map.
    """
    image = np.full(shape=(4, 4, 3), fill_value=100, dtype=np.uint8)
    mask = np.zeros(shape=(4, 4), dtype=np.int64)
    mask[0, :2], mask[1, :2], mask[3, 3] = 1, 2, IGNORED_LABEL_INDEX
    class_map = mask.copy()
    class_map[0, 1], class_map[1, 1], class_map[2, 0], class_map[3, 3] = 2, 0, 1, 0
    return image, mask, class_map


def test_class_colours_are_distinct_with_a_black_background() -> None:
    """
    Verifies that the background colour is black and that every class gets its own colour, also beyond the classes
    whose hues stay distinct.
    """
    for number_classes in [2, 10, 30]:
        class_colours = build_class_colours(number_classes=number_classes)
        assert class_colours.shape == (number_classes, 3) and class_colours.dtype == np.uint8
        assert class_colours[0].tolist() == [0, 0, 0]
        assert len({tuple(colour) for colour in class_colours.tolist()}) == number_classes


def test_overlay_paints_the_classes_and_grays_the_ignored_pixels() -> None:
    """
    Verifies that the overlay leaves the background pixels untouched, blends the class pixels with the colour of their
    class, and blends the ignored pixels with gray.
    """
    image, mask, _ = build_image_and_masks()
    class_colours = build_class_colours(number_classes=3)
    overlay = overlay_segmentation_mask(image=image, class_map=mask, class_colours=class_colours)

    assert np.array_equal(overlay[2, 2], image[2, 2])
    expected_class_colour = (image[0, 0].astype(np.float64) + class_colours[1]) / 2.0
    assert np.allclose(overlay[0, 0], expected_class_colour, atol=1.0)
    assert np.allclose(overlay[3, 3], (image[3, 3].astype(np.float64) + IGNORED_PIXEL_COLOUR) / 2.0, atol=1.0)


def test_error_map_colours_every_kind_of_pixel() -> None:
    """
    Verifies that the error map colours the right class pixels in green, the pixels given a wrong class (another class,
    or a class on the background) in red, the missed class pixels in yellow and the ignored pixels in gray, the right
    background staying a darkened grayscale.
    """
    image, mask, class_map = build_image_and_masks()
    error_map = build_segmentation_error_map(image=image, class_map=class_map, mask=mask).astype(np.int64)

    # Dominant channel of every coloured pixel (BGR): green for right, red for a wrong class, red and green for missed
    assert np.argmax(error_map[0, 0]) == 1 and error_map[0, 0, 1] > 100
    assert np.argmax(error_map[0, 1]) == 2 and np.argmax(error_map[2, 0]) == 2
    assert error_map[1, 1, 1] > 150 and error_map[1, 1, 2] > 150 and error_map[1, 1, 0] < 50
    assert np.all(np.abs(error_map[3, 3] - np.array(IGNORED_PIXEL_COLOUR)) < 40)
    assert np.all(error_map[2, 2] == error_map[2, 2, 0]) and error_map[2, 2, 0] < 100


def test_sample_grid_lays_out_four_panels() -> None:
    """
    Verifies the size of a sample grid: four panels side by side separated by gaps, between the caption and legend
    bars, the input panel holding the image under its title.
    """
    image = np.full(shape=(96, 128, 3), fill_value=100, dtype=np.uint8)
    mask = np.zeros(shape=(96, 128), dtype=np.int64)
    mask[40:60, 40:60] = 1
    grid = build_segmentation_sample_grid(image=image, mask=mask, class_map=np.zeros_like(mask),
                                          class_names=["background", "circle"],
                                          class_colours=build_class_colours(number_classes=2), caption="test")

    assert grid.shape == (CAPTION_BAR_HEIGHT + 96 + LEGEND_BAR_HEIGHT, 4 * 128 + 3 * GRID_GAP, 3)
    assert np.all(grid[CAPTION_BAR_HEIGHT + 50, 100] == 100)


def test_detection_grid_draws_the_boxes_in_their_class_colour() -> None:
    """
    Verifies the detection sample grid: the ground truth and prediction panels side by side between the caption and
    legend bars, the ground truth box drawn in the colour of its class on the left panel only, the detection drawn in
    the colour of its class on the right panel only, and an image without any box left as it is.
    """
    image = np.full(shape=(96, 128, 3), fill_value=100, dtype=np.uint8)
    class_colours = build_class_colours(number_classes=3)
    class_names = {0: "background", 1: "circle", 2: "square"}
    grid = build_detection_sample_grid(
        image=image, ground_truth_boxes=np.array(object=[[20.0, 30.0, 60.0, 70.0]]),
        ground_truth_labels=np.array(object=[1]), predicted_boxes=np.array(object=[[70.0, 30.0, 110.0, 70.0]]),
        predicted_labels=np.array(object=[2]), predicted_scores=np.array(object=[0.9]), class_names=class_names,
        class_colours=class_colours, caption="test")

    assert grid.shape == (CAPTION_BAR_HEIGHT + 96 + LEGEND_BAR_HEIGHT, 2 * 128 + GRID_GAP, 3)
    panels = grid[CAPTION_BAR_HEIGHT:CAPTION_BAR_HEIGHT + 96]
    ground_truth_panel, prediction_panel = panels[:, :128], panels[:, 128 + GRID_GAP:]

    # Left edge of every box, below its caption, in the colour of its class on its own panel only
    assert np.all(ground_truth_panel[50, 20] == class_colours[1]) and np.all(prediction_panel[50, 20] == 100)
    assert np.all(prediction_panel[50, 70] == class_colours[2]) and np.all(ground_truth_panel[50, 70] == 100)

    # Without any box, both panels keep the image
    empty_boxes, empty_labels = np.zeros(shape=(0, 4)), np.zeros(shape=(0,), dtype=np.int64)
    empty_grid = build_detection_sample_grid(
        image=image, ground_truth_boxes=empty_boxes, ground_truth_labels=empty_labels, predicted_boxes=empty_boxes,
        predicted_labels=empty_labels, predicted_scores=np.zeros(shape=(0,)), class_names=class_names,
        class_colours=class_colours, caption="test")
    assert np.all(empty_grid[CAPTION_BAR_HEIGHT + 50, 60] == 100)


def test_caption_bar_is_written_in_its_colour() -> None:
    """
    Verifies that the caption bar is written in the given colour and in none of the others, among the right and wrong
    prediction colours of the sample images and the default white, the text edges being blended with the black bar.
    """
    image = np.full(shape=(32, 200, 3), fill_value=100, dtype=np.uint8)
    caption_colours = [CORRECT_PREDICTION_COLOUR, WRONG_PREDICTION_COLOUR, (255, 255, 255)]
    for caption_colour in caption_colours:
        captioned_image = add_caption_bar(image=image, caption="circle -> square 0.912", caption_colour=caption_colour)
        caption_bar = captioned_image[:CAPTION_BAR_HEIGHT]
        for other_colour in caption_colours:
            colour_present = bool(np.any(np.all(caption_bar == np.array(object=other_colour), axis=-1)))
            assert colour_present == (other_colour == caption_colour)
    assert np.array_equal(add_caption_bar(image=image, caption="test"),
                          add_caption_bar(image=image, caption="test", caption_colour=(255, 255, 255)))
