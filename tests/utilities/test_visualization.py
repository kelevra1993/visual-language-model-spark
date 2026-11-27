"""
Tests of the evaluation drawings of vision_training/utilities/visualization.py: the mask overlay paints the classes and
grays the ignored pixels only, the detection sample grid lays out its two panels between their caption and legend bars
with the boxes in the colour of their class, the frame selection image draws the borders of the selection over its grid
of thumbnails above the probability heat map, the keypoint drawings colour every keypoint by its side, join the drawn
keypoints by the skeleton in the colour of their class, leave the unlabelled keypoints, the predicted ones not found and
the padding positions out, draw the occluded ones as rings and caption the objects, the input viewer is skipped
without a display, and its information panel draws a table of the objects per class in row pairs fitting its width,
and tables of more rows. The captions of the detections give their overlaps with their object, and the training target
viewer draws the anchors, proposals and sampled regions of interest of an image in three panels, dashing the ground
truth copies, and the mask targets of a few regions next to their predicted masks, the image cropped to every region as
RoIAlign maps it.
"""

import cv2
import numpy as np
import pytest
import torch

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.utilities.visualization import (CAPTION_BAR_HEIGHT, GRID_GAP, HEAT_MAP_ROW_HEIGHT,
                                                     IGNORED_PIXEL_COLOUR, INFORMATION_PANEL_WIDTH,
                                                     KEYPOINT_SIDE_COLOURS, LEGEND_BAR_HEIGHT, MASK_TARGET_CELL_SIZE,
                                                     MASK_TARGET_SPACING, PANEL_LINE_HEIGHT, PANEL_MARGIN,
                                                     PANEL_TABLE_GAP, PANEL_TABLE_ROW_HEIGHT, PANEL_TITLE_HEIGHT,
                                                     SELECTION_COLOURS, SELECTION_THUMBNAIL_SIZE,
                                                     TRAINING_TARGET_COLOURS, TRAINING_TARGET_VIEWER_NAME,
                                                     UNSELECTED_FRAME_BRIGHTNESS, WRONG_PREDICTION_COLOUR, PanelTable,
                                                     ViewerWindow, add_bounding_boxes, build_class_colours,
                                                     build_detection_sample_grid, build_frame_selection_image,
                                                     build_information_panel, build_mask_target_strip,
                                                     build_training_target_grid, crop_box_region,
                                                     draw_keypoint_objects, format_detection_caption,
                                                     overlay_segmentation_mask, split_panel_table, view_input_data)

# Classes of the keypoint drawing tests on a 96 x 96 image: a figure of 4 keypoints and a marker of a single centre
# keypoint, padded to the 4 keypoints of the figure
KEYPOINT_CATEGORIES = {1: {"keypoints": ["left_hand", "right_hand", "top", "bottom"],
                           "skeleton": [[0, 1], [1, 2], [0, 3]]}, 2: {"keypoints": ["apex"], "skeleton": []}}

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


def test_viewer_is_skipped_without_a_display(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the input viewer stops at once, without opening any window, when no graphical display is reachable,
    its message naming the viewer: the input image viewer by default, the training target viewer when asked.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture removing the display variables.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed message.
    """
    monkeypatch.delenv(name="DISPLAY", raising=False)
    monkeypatch.delenv(name="WAYLAND_DISPLAY", raising=False)
    windows = [ViewerWindow(image=torch.zeros(size=(3, 8, 8)), bounding_boxes=None, labels=None, label_names=None,
                            title="Training", target_lines=["Class : cat"], augmentation_lines=[],
                            display_input_size=8)]
    assert view_input_data(windows=windows)
    assert "Skipping the input image viewer." in capsys.readouterr().out
    assert view_input_data(windows=windows, viewer_name=TRAINING_TARGET_VIEWER_NAME)
    assert "Skipping the training target viewer." in capsys.readouterr().out


def test_information_panel_draws_its_tables_in_row_pairs() -> None:
    """
    Verifies that a table of the input viewer panel (the objects of a detection sample per class) keeps its columns in
    one row pair when they fit, starts a new pair below when the next column does not fit the panel, and makes the
    panel taller by the height of its pairs, the text lines keeping their place.
    """
    short_table = PanelTable(row_titles=("Class", "Count"), columns=[("square", "1"), ("circle", "2")])
    long_table = PanelTable(row_titles=("Class", "Count"), columns=[(f"hollow_parallelogram_{index}", "1")
                                                                    for index in range(6)])
    maximum_width = INFORMATION_PANEL_WIDTH - 2 * PANEL_MARGIN

    # The short table holds one pair, the long one wraps into pairs that each fit the width
    _, short_pairs = split_panel_table(table=short_table, maximum_width=maximum_width)
    title_width, long_pairs = split_panel_table(table=long_table, maximum_width=maximum_width)
    assert [[column[:2] for column in pair] for pair in short_pairs] == [[("square", "1"), ("circle", "2")]]
    assert len(long_pairs) > 1 and sum(len(pair) for pair in long_pairs) == 6
    assert all(title_width + sum(column[2] for column in pair) <= maximum_width for pair in long_pairs)

    # Every pair adds its two rows and its gap to the height of the panel, the title and the line above the table
    # drawn where they are without it
    text_panel = build_information_panel(title="Training", lines=["Objects : 6"], minimum_height=0)
    table_panel = build_information_panel(title="Training", lines=["Objects : 6", long_table], minimum_height=0)
    assert table_panel.shape[0] - text_panel.shape[0] == len(long_pairs) * (2 * PANEL_TABLE_ROW_HEIGHT +
                                                                              PANEL_TABLE_GAP)
    line_area_bottom = PANEL_MARGIN + PANEL_TITLE_HEIGHT + PANEL_LINE_HEIGHT + 5
    assert np.array_equal(table_panel[:line_area_bottom], text_panel[:line_area_bottom])
    assert table_panel[line_area_bottom:].any()


def test_information_panel_draws_tables_of_more_rows() -> None:
    """
    Verifies that a table of four rows (an object over its targets in the training target viewer) keeps the text of
    every row in its columns, sized on their longest text, and makes the panel taller by its four rows and its gap.
    """
    table = PanelTable(row_titles=("Object", "Positive anchors", "Best proposal IoU", "Positive RoIs"),
                       columns=[("#1 circle", "12", "0.83", "9"), ("#2 square", "3", "0.41", "1")])
    _, row_groups = split_panel_table(table=table, maximum_width=INFORMATION_PANEL_WIDTH - 2 * PANEL_MARGIN)
    assert [[column[:4] for column in row_group] for row_group in row_groups] == [[("#1 circle", "12", "0.83", "9"),
                                                                                   ("#2 square", "3", "0.41", "1")]]
    text_panel = build_information_panel(title="Training Targets", lines=["Objects : 2"], minimum_height=0)
    table_panel = build_information_panel(title="Training Targets", lines=["Objects : 2", table], minimum_height=0)
    assert table_panel.shape[0] - text_panel.shape[0] == 4 * PANEL_TABLE_ROW_HEIGHT + PANEL_TABLE_GAP


def test_detection_caption_gives_the_overlaps_with_the_object() -> None:
    """
    Verifies the captions of the detections of the sample and error images: the class and score alone without
    overlaps, the box overlap of the object detection, the box and mask overlaps of the instance segmentation, the
    overlap with the closest object followed by "no match" for a detection matched to no object, and "no match" alone
    in an image without any object.
    """
    assert format_detection_caption(class_name="circle", score=0.974, overlaps=None, matched=True) == "circle 0.97"
    assert format_detection_caption(class_name="circle", score=0.974, overlaps={"boxes": 0.912},
                                    matched=True) == "circle 0.97 IoU 0.91"
    assert format_detection_caption(class_name="circle", score=0.974, overlaps={"boxes": 0.912, "masks": 0.876},
                                    matched=True) == "circle 0.97 IoU box 0.91 mask 0.88"
    assert format_detection_caption(class_name="circle", score=0.41, overlaps={"boxes": 0.32},
                                    matched=False) == "circle 0.41 IoU 0.32 no match"
    assert format_detection_caption(class_name="circle", score=0.41, overlaps={"boxes": float("nan")},
                                    matched=False) == "circle 0.41 no match"


def test_detection_grid_with_overlaps_keeps_its_layout() -> None:
    """
    Verifies that the detection sample grid given the overlaps of its detections keeps the layout and the box colours
    of the grid without them, only its captions changing.
    """
    image = np.full(shape=(96, 128, 3), fill_value=100, dtype=np.uint8)
    class_colours = build_class_colours(number_classes=3)
    grid_arguments = {"image": image, "ground_truth_boxes": np.array(object=[[20.0, 30.0, 60.0, 70.0]]),
                      "ground_truth_labels": np.array(object=[1]),
                      "predicted_boxes": np.array(object=[[22.0, 30.0, 60.0, 70.0]]),
                      "predicted_labels": np.array(object=[1]), "predicted_scores": np.array(object=[0.9]),
                      "class_names": {0: "background", 1: "circle", 2: "square"}, "class_colours": class_colours,
                      "caption": "test"}
    plain_grid = build_detection_sample_grid(**grid_arguments)
    overlap_grid = build_detection_sample_grid(**grid_arguments, detection_overlaps={"boxes": np.array(object=[0.95])},
                                               matched_detections=np.array(object=[True]))
    assert overlap_grid.shape == plain_grid.shape
    prediction_panel = overlap_grid[CAPTION_BAR_HEIGHT:CAPTION_BAR_HEIGHT + 96, 128 + GRID_GAP:]
    assert np.all(prediction_panel[50, 22] == class_colours[1])
    assert not np.array_equal(overlap_grid, plain_grid)


def test_dashed_boxes_leave_gaps_between_their_dashes() -> None:
    """
    Verifies that add_bounding_boxes draws a dashed outline with gaps between its dashes, of the given thickness, where
    the solid outline is continuous.
    """
    solid_image = np.zeros(shape=(64, 64, 3), dtype=np.uint8)
    dashed_image = np.zeros(shape=(64, 64, 3), dtype=np.uint8)
    box = np.array(object=[[8.0, 8.0, 56.0, 56.0]])
    add_bounding_boxes(bounding_boxes=box, image=solid_image, input_image_size=64, color=(0, 255, 0), thickness=1)
    add_bounding_boxes(bounding_boxes=box, image=dashed_image, input_image_size=64, color=(0, 255, 0), thickness=1,
                       dash_length=4)
    top_edge_solid, top_edge_dashed = solid_image[8, 8:57, 1] > 0, dashed_image[8, 8:57, 1] > 0
    assert top_edge_solid.all() and 0 < top_edge_dashed.sum() < top_edge_solid.sum()
    assert not solid_image[9, 20].any() and not dashed_image[10, 20].any()


def test_box_caption_is_moved_inside_the_right_border() -> None:
    """
    Verifies that the caption of a box near the right border of the image, longer than the space left, is moved left
    so that it ends inside the image, its black background starting left of the box, instead of being cut by the
    border.
    """
    image = np.full(shape=(100, 100, 3), fill_value=100, dtype=np.uint8)
    caption = "hollow_ellipse 0.65 IoU box 0.83 mask 0.71"
    add_bounding_boxes(bounding_boxes=np.array(object=[[80.0, 40.0, 98.0, 60.0]]), image=image, input_image_size=100,
                       color=(0, 255, 0), captions=[caption])
    (caption_width, _), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                                            thickness=1)
    caption_left = max(0, 100 - caption_width - 4)
    assert caption_left < 80 and image[36, caption_left].tolist() != [100, 100, 100]
    assert image[36, max(0, caption_left - 1)].tolist() == [100, 100, 100] or caption_left == 0


def test_training_target_grid_draws_every_target_on_its_panel() -> None:
    """
    Verifies the three panels of the training target viewer between nothing above and the legend bar below: the
    positive and negative anchors on the first panel, the proposals on the second, the positive and negative regions
    of interest and the dashed ground truth copies on the third, the ground truth boxes in white over the first two
    panels only.
    """
    image = np.full(shape=(64, 64, 3), fill_value=50, dtype=np.uint8)
    no_boxes = np.zeros(shape=(0, 4))
    grid = build_training_target_grid(
        image=image, ground_truth_boxes=np.array(object=[[40.0, 40.0, 60.0, 60.0]]), ground_truth_captions=["#1"],
        positive_anchors=np.array(object=[[4.0, 24.0, 20.0, 40.0]]),
        negative_anchors=np.array(object=[[24.0, 24.0, 36.0, 40.0]]),
        proposals=np.array(object=[[4.0, 24.0, 20.0, 40.0]]), positive_regions=np.array(object=[[4.0, 24.0, 20.0, 40.0]]),
        ground_truth_copies=np.array(object=[[40.0, 40.0, 60.0, 60.0]]), negative_regions=no_boxes)
    assert grid.shape == (64 + LEGEND_BAR_HEIGHT, 3 * 64 + 2 * GRID_GAP, 3)
    anchor_panel, proposal_panel = grid[:64, :64], grid[:64, 64 + GRID_GAP:128 + GRID_GAP]
    region_panel = grid[:64, 128 + 2 * GRID_GAP:]

    # Left edges of the boxes, below the panel titles
    assert anchor_panel[50, 40].tolist() == list(TRAINING_TARGET_COLOURS["Ground truth"])
    assert proposal_panel[50, 40].tolist() == list(TRAINING_TARGET_COLOURS["Ground truth"])
    assert anchor_panel[32, 4].tolist() == list(TRAINING_TARGET_COLOURS["Positive"])
    assert anchor_panel[32, 24].tolist() == list(TRAINING_TARGET_COLOURS["Negative"])
    assert proposal_panel[32, 4].tolist() == list(TRAINING_TARGET_COLOURS["Proposal"])
    assert region_panel[32, 4].tolist() == list(TRAINING_TARGET_COLOURS["Positive"])

    # The ground truth copy dashed in green on the regions panel, with gaps along its left edge, and no white box under
    # the title of the panel
    copy_edge = np.all(region_panel[40:61, 40] == TRAINING_TARGET_COLOURS["Positive"], axis=-1)
    assert 0 < copy_edge.sum() < copy_edge.shape[0]
    assert not np.any(np.all(region_panel[24:] == TRAINING_TARGET_COLOURS["Ground truth"], axis=-1))


def test_box_region_is_cropped_as_roi_align_maps_it() -> None:
    """
    Verifies that crop_box_region maps the box corners onto the cell borders: a box equal to a white square fills the
    cell, a box twice as large around it puts the square in the middle half of the cell, and the parts of a box
    outside the image are black.
    """
    image = np.zeros(shape=(40, 40), dtype=np.uint8)
    image[10:20, 10:20] = 255
    full_crop = crop_box_region(image=image, box=np.array(object=[10.0, 10.0, 20.0, 20.0]), output_size=20,
                                interpolation=cv2.INTER_NEAREST)
    assert np.all(full_crop == 255)
    wide_crop = crop_box_region(image=image, box=np.array(object=[5.0, 5.0, 25.0, 25.0]), output_size=20,
                                interpolation=cv2.INTER_NEAREST)
    assert np.all(wide_crop[5:15, 5:15] == 255) and not wide_crop[:5].any() and not wide_crop[15:].any()
    outside_crop = crop_box_region(image=np.full(shape=(40, 40), fill_value=255, dtype=np.uint8),
                                   box=np.array(object=[-20.0, 0.0, 20.0, 40.0]), output_size=20,
                                   interpolation=cv2.INTER_NEAREST)
    assert not outside_crop[:, :10].any() and np.all(outside_crop[:, 10:] == 255)


def test_mask_target_strip_places_every_region_with_its_cells() -> None:
    """
    Verifies the mask target strip of the training target viewer: every region in a block of three cells (the image
    crop with the target contour, the target, the predicted mask) under its caption bar, the blocks of a row placed
    from the left, a block not fitting the width starting a new row, the target cell holding the target enlarged and
    its contour drawn in green on the crop.
    """
    image = np.full(shape=(64, 64, 3), fill_value=80, dtype=np.uint8)
    mask_targets = np.zeros(shape=(3, 4, 4), dtype=np.float32)
    mask_targets[:, 1:3, 1:3] = 1.0
    block_width = 3 * MASK_TARGET_CELL_SIZE + 2 * GRID_GAP
    strip = build_mask_target_strip(image=image, region_boxes=np.array(object=[[8.0, 8.0, 40.0, 40.0]] * 3),
                                    mask_targets=mask_targets, mask_probabilities=np.full(shape=(3, 4, 4),
                                                                                          fill_value=0.5),
                                    region_captions=["#1", "#2", "#3"], width=2 * block_width + MASK_TARGET_SPACING,
                                    title="Mask targets")
    block_height = CAPTION_BAR_HEIGHT + MASK_TARGET_CELL_SIZE
    assert strip.shape == (CAPTION_BAR_HEIGHT + 2 * (block_height + MASK_TARGET_SPACING) + MASK_TARGET_SPACING,
                           2 * block_width + MASK_TARGET_SPACING, 3)

    # Target cell of the first region: white in its middle half, black at its border
    cell_top = CAPTION_BAR_HEIGHT + MASK_TARGET_SPACING + CAPTION_BAR_HEIGHT
    target_cell = strip[cell_top:cell_top + MASK_TARGET_CELL_SIZE,
                        MASK_TARGET_CELL_SIZE + GRID_GAP:2 * MASK_TARGET_CELL_SIZE + GRID_GAP]
    quarter = MASK_TARGET_CELL_SIZE // 4
    assert np.all(target_cell[quarter + 2:3 * quarter - 2, quarter + 2:3 * quarter - 2] == 255)
    assert not target_cell[:quarter - 2].any()

    # Contour of the target in green on the crop, the predicted mask a gray of half intensity
    crop_cell = strip[cell_top:cell_top + MASK_TARGET_CELL_SIZE, :MASK_TARGET_CELL_SIZE]
    assert np.any(np.all(crop_cell == TRAINING_TARGET_COLOURS["Positive"], axis=-1))
    probability_cell = strip[cell_top:cell_top + MASK_TARGET_CELL_SIZE, 2 * (MASK_TARGET_CELL_SIZE + GRID_GAP):
                             3 * MASK_TARGET_CELL_SIZE + 2 * GRID_GAP]
    assert probability_cell[5, 5].tolist() == [127, 127, 127]
