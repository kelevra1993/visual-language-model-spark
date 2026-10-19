"""
Tests of the evaluation drawings of vision_training/utilities/visualization.py: the class colours are distinct with
an unpainted background, the mask overlay paints the classes and grays the ignored pixels only, the error map gives
every kind of pixel its colour, the segmentation sample grid lays out its four panels and the detection sample grid its
two panels between their caption and legend bars, outlining the boxes in error of the error images, the caption bar
is written in the given colour, the frame selection image draws the borders of the selection over its grid of
thumbnails above the probability heat map, the keypoint drawings colour every keypoint by the side its name gives, join
the drawn keypoints by the skeleton in the colour of their class, leave the unlabelled keypoints, the predicted ones not
found and the padding positions out, draw the occluded ones as rings and caption the objects, the keypoint sample grid
lays out its ground truth and prediction panels between their caption and legend bars, the frame grid numbers its
frames from the given number, and the input viewer wraps the lines of its information panels, places each image next
to its panel, centres its windows on the screen and is skipped without a display.
"""

import cv2
import numpy as np
import pytest
import torch

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.utilities.visualization import (CAPTION_BAR_HEIGHT, CORRECT_PREDICTION_COLOUR,
                                                     ERROR_OUTLINE_MARGIN, GRID_GAP,
                                                     HEAT_MAP_ROW_HEIGHT, IGNORED_PIXEL_COLOUR,
                                                     INFORMATION_PANEL_WIDTH, KEYPOINT_SIDE_COLOURS,
                                                     LEGEND_BAR_HEIGHT, PANEL_GROUP_COLOUR,
                                                     PANEL_LINE_COLOUR, PANEL_LINE_FONT_SCALE, PANEL_LINE_HEIGHT,
                                                     PANEL_MARGIN, PANEL_TITLE_COLOUR, PANEL_TITLE_HEIGHT,
                                                     SELECTION_COLOURS, SELECTION_THUMBNAIL_SIZE,
                                                     UNSELECTED_FRAME_BRIGHTNESS, VIEWER_TITLE_BAR_HEIGHT,
                                                     VIEWER_WINDOW_SPACING, WRONG_PREDICTION_COLOUR,
                                                     add_caption_bar, build_class_colours,
                                                     build_detection_sample_grid, build_frame_grid,
                                                     build_frame_selection_image,
                                                     build_information_panel, build_keypoint_sample_grid,
                                                     build_segmentation_error_map, build_segmentation_sample_grid,
                                                     build_viewer_window, compute_window_positions,
                                                     draw_keypoint_objects, find_keypoint_side,
                                                     overlay_segmentation_mask, view_input_data, wrap_panel_line)

# Classes of the keypoint drawing tests on a 96 x 96 image: a figure of 4 keypoints and a marker of a single centre
# keypoint, padded to the 4 keypoints of the figure
KEYPOINT_CATEGORIES = {1: {"keypoints": ["left_hand", "right_hand", "top", "bottom"],
                           "skeleton": [[0, 1], [1, 2], [0, 3]]}, 2: {"keypoints": ["apex"], "skeleton": []}}
KEYPOINT_CLASS_NAMES = {1: "figure", 2: "marker"}

# Keypoints of a figure and of a marker, at pixel centres so that their drawn pixels round(coordinate - 0.5) are whole:
# the left hand at (16, 32), the right hand at (48, 32), the top at (32, 48) and the bottom at (32, 80), the top being
# unlabelled and the bottom occluded, and the apex of the marker at (70, 70) followed by its padding positions at 0
KEYPOINT_COORDINATES = np.array(object=[[[16.5, 32.5], [48.5, 32.5], [32.5, 48.5], [32.5, 80.5]],
                                        [[70.5, 70.5], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
KEYPOINT_VISIBILITIES = np.array(object=[[2, 2, 0, 1], [2, 0, 0, 0]])
KEYPOINT_LABELS = np.array(object=[1, 2])


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


def test_detection_grid_outlines_the_boxes_in_error() -> None:
    """
    Verifies the outline of the error images: a true box in error gets red dashes a few pixels outside its edges on
    the ground truth panel, the detection not in error gets none, and the legend names the outline first, its colour
    square in red.
    """
    image = np.full(shape=(96, 128, 3), fill_value=100, dtype=np.uint8)
    grid = build_detection_sample_grid(
        image=image, ground_truth_boxes=np.array(object=[[20.0, 30.0, 60.0, 70.0]]),
        ground_truth_labels=np.array(object=[1]), predicted_boxes=np.array(object=[[70.0, 30.0, 110.0, 70.0]]),
        predicted_labels=np.array(object=[2]), predicted_scores=np.array(object=[0.9]),
        class_names={0: "background", 1: "circle", 2: "square"}, class_colours=build_class_colours(number_classes=3),
        caption="MISSED circle", highlighted_ground_truths=np.array(object=[True]),
        highlighted_detections=np.array(object=[False]))
    panels = grid[CAPTION_BAR_HEIGHT:CAPTION_BAR_HEIGHT + 96]
    ground_truth_panel, prediction_panel = panels[:, :128], panels[:, 128 + GRID_GAP:]
    is_red = np.all(a=ground_truth_panel == np.array(object=WRONG_PREDICTION_COLOUR), axis=-1)

    # Dashes on the outline left of the true box only, none around the detection
    outline_column = 20 - ERROR_OUTLINE_MARGIN
    assert is_red[:, outline_column - 1:outline_column + 2].any()
    assert not np.all(a=prediction_panel == np.array(object=WRONG_PREDICTION_COLOUR), axis=-1)[
        :, 70 - ERROR_OUTLINE_MARGIN - 1:70 - ERROR_OUTLINE_MARGIN + 2].any()

    # The legend starts with the colour square of the outline
    legend_bar = grid[-LEGEND_BAR_HEIGHT:]
    assert np.all(legend_bar[12, 10] == np.array(object=WRONG_PREDICTION_COLOUR))


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


def test_frame_selection_image_draws_the_selection() -> None:
    """
    Verifies the image of a sequence of 10 frames whose class A is chosen right (frame 0) and class B wrong (frame 3
    instead of frame 1): two rows of 8 thumbnails, a solid green border on frame 0, a solid red border on frame 3, a
    dashed yellow border on frame 1, the rightly unselected frames dimmed, the heat map of the two wanted classes and
    the legend below, and the caption in red for 1 of 2 selections right.
    """
    frames = np.full(shape=(10, 32, 32, 3), fill_value=100, dtype=np.uint8)
    probabilities = np.full(shape=(10, 3), fill_value=0.1)
    probabilities[:, 2] = 0.8
    frame_labels = np.array(object=[0, 1] + [2] * 8)
    image = build_frame_selection_image(frames=frames, probabilities=probabilities, frame_labels=frame_labels,
                                        selected_frames=np.array(object=[[0], [3]]), wanted_class_indices=[0, 1],
                                        unselected_class_index=2, class_names=["A", "B", "unselected"])

    # Caption, two rows of thumbnails, the heat map of the header row and two classes, and the legend
    cell_size = SELECTION_THUMBNAIL_SIZE + GRID_GAP
    assert image.shape == (CAPTION_BAR_HEIGHT + 2 * cell_size + GRID_GAP + 3 * HEAT_MAP_ROW_HEIGHT + GRID_GAP +
                           LEGEND_BAR_HEIGHT, 8 * cell_size + GRID_GAP, 3)
    assert np.any(np.all(image[:CAPTION_BAR_HEIGHT] == np.array(object=WRONG_PREDICTION_COLOUR), axis=-1))

    # Border colour on the top edge of the thumbnails of the chosen and missed frames
    thumbnail_top = CAPTION_BAR_HEIGHT + GRID_GAP
    for frame_index, colour_name in [(0, "Chosen right"), (3, "Chosen wrong"), (1, "True not chosen")]:
        assert image[thumbnail_top + 1, GRID_GAP + frame_index * cell_size + 7].tolist() == list(
            SELECTION_COLOURS[colour_name])

    # A chosen frame keeps its brightness, a rightly unselected frame is dimmed
    thumbnail_centre = thumbnail_top + SELECTION_THUMBNAIL_SIZE // 2
    assert image[thumbnail_centre, GRID_GAP + SELECTION_THUMBNAIL_SIZE // 2].tolist() == [100, 100, 100]
    assert image[thumbnail_centre, GRID_GAP + 2 * cell_size + SELECTION_THUMBNAIL_SIZE // 2].tolist() == [
        int(100 * UNSELECTED_FRAME_BRIGHTNESS)] * 3


@pytest.mark.parametrize(argnames=("keypoint_name", "expected_side"), argvalues=[
    ("left_elbow", "Left"), ("bottom_left", "Left"), ("right_1", "Right"), ("top_right", "Right"), ("nose", "Centre"),
    ("leftover", "Centre"), ("centre", "Centre")])
def test_keypoint_side_is_read_from_the_whole_words_of_its_name(keypoint_name: str, expected_side: str) -> None:
    """
    Verifies that the side of a keypoint comes from 'left' or 'right' as a whole word of its name, wherever it stands,
    any other keypoint being in the centre.

    Args:
        keypoint_name (str): The name of the keypoint.
        expected_side (str): The expected side.
    """
    assert find_keypoint_side(keypoint_name=keypoint_name) == expected_side


def test_keypoint_objects_draw_the_labelled_keypoints_by_side() -> None:
    """
    Verifies the drawing of the true keypoints of the objects of an image: the visible left and right keypoints as
    filled discs of their side colours, the occluded keypoint as a ring of the centre colour whose hole shows the end
    of its skeleton line, the skeleton lines between two drawn keypoints in the colour of the class, neither the
    unlabelled keypoint nor its skeleton line, the single keypoint of the marker without its padding positions, and
    the caption of an object above its highest keypoint. The predicted keypoints, without visibilities, are drawn as
    filled discs when found, the keypoints at NaN being left out with their skeleton lines.
    """
    image = np.full(shape=(96, 96, 3), fill_value=100, dtype=np.uint8)
    class_colours = build_class_colours(number_classes=3)
    figure_colour = class_colours[1]
    drawing = draw_keypoint_objects(image=image, keypoints=KEYPOINT_COORDINATES, visibilities=KEYPOINT_VISIBILITIES,
                                    labels=KEYPOINT_LABELS, keypoint_categories=KEYPOINT_CATEGORIES,
                                    class_colours=class_colours)
    assert drawing.shape == (96, 96, 3) and np.all(image == 100)

    # Filled discs of the left and right sides, and the ring of the occluded centre keypoint, its hole showing the
    # skeleton line drawn under it
    assert np.all(drawing[32, 16] == KEYPOINT_SIDE_COLOURS["Left"])
    assert np.all(drawing[32, 48] == KEYPOINT_SIDE_COLOURS["Right"])
    assert np.all(drawing[80, 32] == figure_colour) and np.all(drawing[80, 36] == KEYPOINT_SIDE_COLOURS["Centre"])

    # Skeleton between the two hands and from the left hand to the occluded bottom, none to the unlabelled top
    assert np.all(drawing[32, 32] == figure_colour) and np.all(drawing[56, 24] == figure_colour)
    assert np.all(drawing[44:53, 28:37] == 100) and np.all(drawing[40, 40] == 100)

    # The apex of the marker drawn, its padding positions left out
    assert np.all(drawing[70, 70] == KEYPOINT_SIDE_COLOURS["Centre"]) and np.all(drawing[:4, :4] == 100)

    # The caption of the figure on a black background above its hands, the highest keypoints
    captioned_drawing = draw_keypoint_objects(image=image, keypoints=KEYPOINT_COORDINATES,
                                              visibilities=KEYPOINT_VISIBILITIES, labels=KEYPOINT_LABELS,
                                              keypoint_categories=KEYPOINT_CATEGORIES, class_colours=class_colours,
                                              captions=["figure", "marker"])
    assert not np.all(drawing[:26, 16:40] == 0, axis=2).any()
    assert np.all(captioned_drawing[:26, 16:40] == 0, axis=2).any()

    # Every predicted keypoint found drawn as a filled disc, the top included and the bottom at NaN left out
    predicted_keypoints = KEYPOINT_COORDINATES[:1].copy()
    predicted_keypoints[0, 3] = np.nan
    predicted_drawing = draw_keypoint_objects(image=image, keypoints=predicted_keypoints, visibilities=None,
                                              labels=KEYPOINT_LABELS[:1], keypoint_categories=KEYPOINT_CATEGORIES,
                                              class_colours=class_colours)
    assert np.all(predicted_drawing[48, 32] == KEYPOINT_SIDE_COLOURS["Centre"])
    assert np.all(predicted_drawing[40, 40] == figure_colour)
    assert np.all(predicted_drawing[56, 24] == 100) and np.all(predicted_drawing[80, 32] == 100)


def test_keypoint_grid_lays_out_the_true_and_predicted_objects() -> None:
    """
    Verifies the keypoint sample grid: the ground truth and prediction panels side by side between the caption and
    legend bars, the unlabelled true keypoint drawn on the prediction panel only, the marker missed by the prediction
    drawn on the ground truth panel only, and the legend naming the drawn sides in the order left, right, centre, an
    image holding only classes without left nor right keypoints starting with the centre.
    """
    image = np.full(shape=(96, 96, 3), fill_value=100, dtype=np.uint8)
    class_colours = build_class_colours(number_classes=3)
    grid = build_keypoint_sample_grid(image=image, ground_truth_keypoints=KEYPOINT_COORDINATES,
                                      visibilities=KEYPOINT_VISIBILITIES, ground_truth_labels=KEYPOINT_LABELS,
                                      predicted_keypoints=KEYPOINT_COORDINATES[:1],
                                      predicted_labels=KEYPOINT_LABELS[:1], predicted_scores=np.array(object=[0.9]),
                                      keypoint_categories=KEYPOINT_CATEGORIES, class_names=KEYPOINT_CLASS_NAMES,
                                      class_colours=class_colours, caption="test")
    assert grid.shape == (CAPTION_BAR_HEIGHT + 96 + LEGEND_BAR_HEIGHT, 2 * 96 + GRID_GAP, 3)
    panels = grid[CAPTION_BAR_HEIGHT:CAPTION_BAR_HEIGHT + 96]
    ground_truth_panel, prediction_panel = panels[:, :96], panels[:, 96 + GRID_GAP:]

    # The unlabelled top keypoint on the prediction panel only, the missed marker on the ground truth panel only
    assert np.all(ground_truth_panel[48, 32] == 100)
    assert np.all(prediction_panel[48, 32] == KEYPOINT_SIDE_COLOURS["Centre"])
    assert np.all(ground_truth_panel[70, 70] == KEYPOINT_SIDE_COLOURS["Centre"])
    assert np.all(prediction_panel[70, 70] == 100)

    # The legend starts with the colour square of the left side, or of the centre for an image of markers only
    assert np.all(grid[-LEGEND_BAR_HEIGHT:][12, 10] == KEYPOINT_SIDE_COLOURS["Left"])
    centre_grid = build_keypoint_sample_grid(image=image, ground_truth_keypoints=KEYPOINT_COORDINATES[1:],
                                             visibilities=KEYPOINT_VISIBILITIES[1:],
                                             ground_truth_labels=KEYPOINT_LABELS[1:],
                                             predicted_keypoints=np.zeros(shape=(0, 4, 2)),
                                             predicted_labels=np.zeros(shape=(0,), dtype=np.int64),
                                             predicted_scores=np.zeros(shape=(0,)),
                                             keypoint_categories=KEYPOINT_CATEGORIES,
                                             class_names=KEYPOINT_CLASS_NAMES, class_colours=class_colours,
                                             caption="test")
    assert np.all(centre_grid[-LEGEND_BAR_HEIGHT:][12, 10] == KEYPOINT_SIDE_COLOURS["Centre"])


def test_panel_lines_wrap_within_the_panel() -> None:
    """
    Verifies the wrapping of the lines of the information panels of the input viewer: a short line is kept as it is,
    and a long indented line is split word by word into lines fitting the width, the continuation lines four spaces
    deeper, without losing any word.
    """
    assert wrap_panel_line(line="  random_left_right_flip", maximum_width=300) == ["  random_left_right_flip"]

    long_line = "  Frame 0 : " + ", ".join(["brightness_contrast_saturation (brightness +12.0 levels)"] * 4)
    wrapped_lines = wrap_panel_line(line=long_line, maximum_width=400)
    assert len(wrapped_lines) > 1 and wrapped_lines[0].startswith("  Frame 0")
    assert all(wrapped_line.startswith("      ") for wrapped_line in wrapped_lines[1:])
    assert " ".join(word for wrapped_line in wrapped_lines for word in wrapped_line.split()) == " ".join(
        long_line.split())
    for wrapped_line in wrapped_lines:
        (text_width, _), _ = cv2.getTextSize(text=wrapped_line, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                             fontScale=PANEL_LINE_FONT_SCALE, thickness=1)
        assert text_width <= 400


def test_information_panel_matches_the_image_and_grows_with_its_lines() -> None:
    """
    Verifies the black information panel of the input viewer: it keeps the height of the image for a few lines and
    grows for many, writes its title in white, its group lines in green and its indented lines in light gray.
    """
    panel = build_information_panel(title="Training", lines=["Class : cat", "  random_left_right_flip"],
                                    minimum_height=300)
    assert panel.shape == (300, INFORMATION_PANEL_WIDTH, 3)
    for colour in [PANEL_TITLE_COLOUR, PANEL_GROUP_COLOUR, PANEL_LINE_COLOUR]:
        assert bool((panel == np.array(colour, dtype=np.uint8)).all(axis=2).any())

    # Fifty lines need more room than the image, the panel growing downwards
    tall_panel = build_information_panel(title="Training", lines=[f"Line {line_index}" for line_index in range(50)],
                                         minimum_height=100)
    assert tall_panel.shape[0] == 2 * PANEL_MARGIN + PANEL_TITLE_HEIGHT + 52 * PANEL_LINE_HEIGHT


def test_viewer_window_places_the_resized_image_next_to_its_panel() -> None:
    """
    Verifies the content of a window of the input viewer: the image enlarged with sharp pixels until its longest side
    is the display size, at the top left, then the gray gap and the information panel, the window taking the height
    of the panel when it is taller than the image.
    """
    image = torch.zeros(size=(3, 10, 20))
    image[:, :, :10] = 200.0
    window_image = build_viewer_window(image=image, bounding_boxes=None, labels=None, label_names=None,
                                       title="Validation", lines=["Class : dog"], display_input_size=80)
    assert window_image.shape[1] == 80 + GRID_GAP + INFORMATION_PANEL_WIDTH
    assert window_image.shape[0] == build_information_panel(title="Validation", lines=["Class : dog"],
                                                            minimum_height=40).shape[0]
    assert bool((window_image[:40, :40] == 200).all()) and bool((window_image[:40, 40:80] == 0).all())
    assert bool((window_image[:, 80:80 + GRID_GAP] == 128).all())


def test_windows_are_centred_on_the_screen() -> None:
    """
    Verifies the placement of the windows of the input viewer: side by side and centred on a large screen at their
    own size, stacked and scaled down to stay inside a small screen, and side by side from the top left of the screen
    when its size is unknown.
    """
    window_sizes = [(1276, 410), (1276, 410)]

    # Side by side on a 4K screen, the block centred horizontally and vertically
    window_positions, display_scale = compute_window_positions(window_sizes=window_sizes, screen_size=(3840, 2160))
    assert display_scale == 1.0 and window_positions[0][1] == window_positions[1][1]
    block_width = 2 * 1276 + VIEWER_WINDOW_SPACING
    assert abs(window_positions[0][0] - (3840 - block_width) / 2) <= 1
    assert window_positions[1][0] == window_positions[0][0] + 1276 + VIEWER_WINDOW_SPACING
    assert abs(window_positions[0][1] - (2160 - 410 - VIEWER_TITLE_BAR_HEIGHT) / 2) <= 1

    # Stacked on a small screen, scaled down so that both windows stay inside it
    window_positions, display_scale = compute_window_positions(window_sizes=[(1300, 900), (1300, 900)],
                                                               screen_size=(1920, 1080))
    assert display_scale < 1.0 and window_positions[0][0] == window_positions[1][0]
    assert window_positions[1][1] + 900 * display_scale + VIEWER_TITLE_BAR_HEIGHT <= 1080
    assert window_positions[0][0] + 1300 * display_scale <= 1920

    # Unknown screen size: side by side from the top left, at their own size
    assert compute_window_positions(window_sizes=window_sizes, screen_size=None) == (
        [(0, VIEWER_TITLE_BAR_HEIGHT), (1276 + VIEWER_WINDOW_SPACING, VIEWER_TITLE_BAR_HEIGHT)], 1.0)


def test_frame_grid_numbers_the_frames_from_the_given_number() -> None:
    """
    Verifies that the frames of a grid are numbered from 0 by default, as in the input viewer of the sequence trainers,
    and from the given first number otherwise, as the ranks of the images of a cluster in the mosaics of the image
    embedding clustering: the first cell of a grid numbered from 1 is the second cell of the same frames numbered from
    0, the padding frames being left out.
    """
    frames = torch.full(size=(3, 3, 24, 32), fill_value=100, dtype=torch.uint8)
    default_grid = build_frame_grid(frames=frames, sequence_mask=torch.tensor(data=[True, True, False]),
                                    frame_captions=None, number_columns=2)
    shifted_grid = build_frame_grid(frames=frames[:1], sequence_mask=torch.tensor(data=[True]), frame_captions=None,
                                    number_columns=2, first_frame_number=1)

    # Two columns of 32 pixel wide frames between gaps of 4 pixels, the padding frame not shown
    assert default_grid.shape == shifted_grid.shape == (3, 24 + 2 * 4, 2 * 32 + 3 * 4)
    assert torch.equal(input=default_grid, other=build_frame_grid(
        frames=frames, sequence_mask=torch.tensor(data=[True, True, False]), frame_captions=None, number_columns=2,
        first_frame_number=0))
    assert torch.equal(input=shifted_grid[:, 4:28, 4:36], other=default_grid[:, 4:28, 40:72])
    assert not torch.equal(input=shifted_grid[:, 4:28, 4:36], other=default_grid[:, 4:28, 4:36])


def test_viewer_is_skipped_without_a_display(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the input viewer stops at once, without opening any window, when no graphical display is reachable.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture removing the display variables.
    """
    monkeypatch.delenv(name="DISPLAY", raising=False)
    monkeypatch.delenv(name="WAYLAND_DISPLAY", raising=False)
    assert view_input_data(images=[torch.zeros(size=(3, 8, 8))], bounding_boxes=None, labels=None,
                           window_titles=["Training"], display_input_size=8, panel_lines=[["Class : cat"]])
