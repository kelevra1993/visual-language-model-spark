"""
Tests of vision_training/data/synthetic/create_synthetic_dataset.py: the synthetic detection, instance segmentation,
classification, semantic segmentation, sequence classification, sequence labelling, sequence comparison, frame
selection, self-supervised and multi-label classification datasets are valid, load with the datasets of the framework
and follow their labels. Every task draws the shared ingredients (flat or textured backgrounds, shapes of a colour close
to a flat background, shapes cut by the border and lying over each other, log-uniform sizes placed from the largest),
a low contrast shape left without a low contrast colour apart from the other shapes gets a contrasting colour instead,
every shape a label counts, names, compares or selects stays visible enough, and the objects mostly hidden leave the
detection, instance segmentation and keypoint labels. The largest IoU between two objects reaches the placement of every
object, 0.3 by default. The objects placed next to one of their class for the occlusion test split of stage 5b overlap
the object placed before them by an IoU from 0.5 to that cap, with its class and a close size, in pairs, and that
placement is refused where it would have no effect. The shapes stay inside their square, every hollow shape is its
filled shape with one hole cut inside it, and the regular polygons keep the size at which their sides stay apart. The
console describes every split in a panel with its class counts and generation time, and closes with a panel of the
whole dataset.
"""

import copy
import itertools
import json
import cv2
import numpy as np
import os
import pytest
import re
import subprocess
import sys

from pathlib import Path
from rich.console import Console
from typing import Any, Callable, Dict, List, Optional, Tuple

from vision_training.data.classification.classification_dataset import ClassificationDataset
from vision_training.data.detection.coco_dataset import CocoDetectionDataset
from vision_training.data.detection.coco_masks import decode_run_length_encoding
from vision_training.data.segmentation.segmentation_dataset import SegmentationDataset
from vision_training.data.sequence.sequence_dataset import SequenceDataset
from vision_training.data.synthetic import create_synthetic_dataset as synthetic_dataset_module
from vision_training.data.synthetic.create_synthetic_dataset import (
    CHANGE_CATEGORIES, HOLLOW_CLASS_NAMES, HOLLOW_DISTRACTOR_SHAPE_NAMES, HOLLOW_INNER_KEYPOINT_SHAPE_NAMES,
    HOLLOW_MINIMUM_SIZE, KEYPOINT_DRAWING_FUNCTIONS, KEYPOINT_MINIMUM_OBJECTS, KEYPOINT_NAMES,
    KEYPOINT_OBJECT_MINIMUM_SIZE, LABELLING_MINIMUM_OBJECTS, LOW_CONTRAST_DIFFERENCE_RANGE,
    MAXIMUM_OBJECT_INTERSECTION_OVER_UNION, MINIMUM_COLOUR_DIFFERENCE, MINIMUM_OBJECTS_PER_IMAGE,
    MINIMUM_VISIBLE_OBJECT_PIXELS, MOVEMENT_CATEGORIES, MOVEMENT_DIRECTIONS, OBJECT_MAXIMUM_SIZE_FRACTION,
    OBJECT_MINIMUM_SIZE, OBJECT_MINIMUM_VISIBLE_FRACTION, PAIR_CATEGORIES, PAIR_MINIMUM_OBJECTS,
    SAME_CLASS_NEIGHBOUR_MINIMUM_INTERSECTION_OVER_UNION, SAME_CLASS_NEIGHBOUR_SIZE_RATIO_RANGE, SELECTION_CATEGORIES,
    SELECTION_LARGEST_MINIMUM_SIZE, SELECTION_SMALLER_AREA_RATIO, SELECTION_SMALLER_SIZE_RATIO, SELECTION_WANTED_SHAPES,
    SHAPE_CATEGORIES, SHAPE_COLOUR_MINIMUM_DIFFERENCE, SHAPE_COUNT_CATEGORIES, SHAPE_DRAWING_FUNCTIONS,
    SHAPE_MINIMUM_SIZES, SHAPE_MINIMUM_VISIBLE_FRACTION, SUMMARY_PANEL_WIDTH, TASK_SAMPLE_NAMES, TEXTURE_AMPLITUDE,
    build_class_count_table, create_synthetic_dataset, draw_background, draw_shape_descriptor, draw_shape_descriptors,
    find_neighbour_centre_distance, format_generation_rate, generate_counted_shapes_image, generate_image,
    generate_moving_shape_sequence, generate_object_loss_pair, generate_selection_sequence,
    generate_shape_change_sequence, get_canvas_windows, get_intersection_over_union, get_mask_bounding_box,
    keep_visible_objects)

IMAGE_SIZE = 256

# Root of the repository, from which the generator runs as a module like in the README
REPOSITORY_FOLDER = Path(__file__).absolute().parents[3]

# Side of the images and frames of the tests following the drawing of the samples, the number of frames of their
# sequences, and the largest number of shapes of a sample of every task drawn by draw_task_sample
SAMPLE_IMAGE_SIZE = 128
SAMPLE_FRAMES = 4
TASK_SAMPLE_MAXIMUM_OBJECTS = {"object_detection": 8, "keypoint_detection": 5, "classification": 1,
                               "semantic_segmentation": 6, "self_supervised": 7, "sequence_classification": 1,
                               "sequence_labelling": 6, "sequence_comparison": 4, "frame_selection": 1}


class CallRecorder:
    """
    Wraps a function called with keyword arguments only and records the arguments of every call before running it,
    then its result, so that a test can read the sizes and vertices the generator draws its shapes with, the
    placements that put an object in the image, and the backgrounds, frames and instance maps it draws. The list
    arguments can be recorded as shallow copies, so that a list of shapes the generator changes after the call keeps its
    content at the call, and the results as deep copies, so that an image drawn in place afterwards keeps its pixels at
    the return.
    """

    def __init__(self, function: Callable[..., Any], copy_lists: bool = False, copy_results: bool = False) -> None:
        """
        Wraps the function.

        Args:
            function (Callable[..., Any]): The function to record and run.
            copy_lists (bool): Whether every list argument is recorded as a shallow copy. Defaults to False.
            copy_results (bool): Whether every result is recorded as a deep copy. Defaults to False.
        """
        self.function = function
        self.copy_lists = copy_lists
        self.copy_results = copy_results
        self.calls: List[Dict[str, Any]] = []
        self.results: List[Any] = []

    def __call__(self, **keyword_arguments: Any) -> Any:
        """
        Records the arguments of the call, then runs the wrapped function with them and records its result.

        Args:
            **keyword_arguments (Any): The arguments of the call.

        Returns:
            Any: The result of the wrapped function.
        """
        # Arguments recorded before the call, the lists as copies on demand, the function receiving the originals
        self.calls.append({name: list(value) if self.copy_lists and isinstance(value, list) else value
                           for name, value in keyword_arguments.items()})
        result = self.function(**keyword_arguments)
        self.results.append(copy.deepcopy(result) if self.copy_results else result)
        return result


def draw_task_sample(task: str, image_size: int, maximum_objects: int, random_generator: np.random.Generator) -> Any:
    """
    Draws one sample of a task with the drawing function and the settings its dataset chunks use (generate_image_chunk
    and generate_sequence_chunk), so that a test follows the backgrounds, colours, sizes and placements of every task.

    Args:
        task (str): The task of the sample, "self_supervised" standing for the multi-label classification images too.
        image_size (int): The height and width of the square images or frames in pixels.
        maximum_objects (int): The largest number of shapes of an image or frame.
        random_generator (np.random.Generator): The random generator of the sample.

    Returns:
        Any: The sample as its drawing function returns it.

    Raises:
        ValueError: If the task has no drawing function here.
    """
    # Single images of the image chunks: the objects of the detection tasks, the visible shapes of the class maps and
    # the counted shapes
    if task in ["object_detection", "keypoint_detection"]:
        return generate_image(image_size=image_size, maximum_objects=maximum_objects, random_generator=random_generator,
                              keypoints=task == "keypoint_detection", object_pixel_counts=[])
    if task in ["classification", "semantic_segmentation"]:
        return generate_image(image_size=image_size, maximum_objects=maximum_objects, random_generator=random_generator,
                              minimum_visible_fraction=SHAPE_MINIMUM_VISIBLE_FRACTION)
    if task == "self_supervised":
        return generate_counted_shapes_image(image_size=image_size, maximum_objects=maximum_objects,
                                             random_generator=random_generator)

    # Sequences and pairs of the sequence chunks
    if task == "sequence_classification":
        return generate_moving_shape_sequence(image_size=image_size, number_frames=SAMPLE_FRAMES,
                                              random_generator=random_generator)
    if task == "sequence_labelling":
        return generate_shape_change_sequence(image_size=image_size, number_frames=SAMPLE_FRAMES,
                                              maximum_objects=maximum_objects, random_generator=random_generator)
    if task == "sequence_comparison":
        return generate_object_loss_pair(image_size=image_size, minimum_frames=2, maximum_frames=SAMPLE_FRAMES,
                                         maximum_objects=maximum_objects, random_generator=random_generator)
    if task == "frame_selection":
        return generate_selection_sequence(image_size=image_size, number_frames=SAMPLE_FRAMES + 2,
                                           random_generator=random_generator)
    raise ValueError(f"Expected a task drawn by draw_task_sample, got '{task}'.")


def find_canvas_position(shape_pixels: np.ndarray, shape_mask: np.ndarray) -> Tuple[int, int]:
    """
    Finds where the canvas of a shape was pasted in a synthetic frame, from the pixels of the shape inside the frame:
    every canvas position whose shape box holds the visible box is pasted again, and exactly one must give the pixels
    of the shape, so that a test follows a shape partly outside the frame (a border cutting one axis can also remove
    the extreme pixels of the other axis) and makes sure the frame shows that shape only.

    Args:
        shape_pixels (np.ndarray): The [H, W] bool pixels of the shape in the frame.
        shape_mask (np.ndarray): The [S, S] bool pixels of the shape on its canvas.

    Returns:
        Tuple[int, int]: The column and row of the frame at the top left corner of the canvas.
    """
    frame_height, frame_width = shape_pixels.shape
    visible_box = get_mask_bounding_box(mask=shape_pixels)
    shape_box = get_mask_bounding_box(mask=shape_mask)

    # Positions of the canvas whose shape box holds the visible box, one more on an axis for every row or column cut
    left_positions = range(visible_box[2] - shape_box[2], visible_box[0] - shape_box[0] + 1)
    top_positions = range(visible_box[3] - shape_box[3], visible_box[1] - shape_box[1] + 1)

    # The canvas pasted at exactly one of them gives the pixels of the shape
    matching_positions = []
    for left, top in itertools.product(left_positions, top_positions):
        image_window, canvas_window = get_canvas_windows(canvas_shape=shape_mask.shape, left=left, top=top,
                                                         image_shape=(frame_height, frame_width))
        pasted_pixels = np.zeros_like(shape_pixels)
        pasted_pixels[image_window] = shape_mask[canvas_window]
        if np.array_equal(pasted_pixels, shape_pixels):
            matching_positions.append((left, top))
    assert len(matching_positions) == 1, f"Expected the shape at one canvas position, found {matching_positions}."
    return matching_positions[0]


def check_frame_shows_its_shapes(frame: np.ndarray, instance_map: np.ndarray, background: np.ndarray,
                                 shape_descriptors: List[Dict[str, Any]]) -> None:
    """
    Checks that a frame drawn by draw_shapes_frame shows exactly its shapes over the background of its sequence: the
    pixels differing from the background are those of the instance map, and every shape keeps
    SHAPE_MINIMUM_VISIBLE_FRACTION of its pixels visible in its own colour, so that the labels counting, naming or
    comparing the shapes of the frame are true.

    Args:
        frame (np.ndarray): The [H, W, 3] uint8 BGR frame.
        instance_map (np.ndarray): The [H, W] index of the shape of every pixel (shape i of the list at i + 1, 0 for the
            background).
        background (np.ndarray): The [H, W, 3] uint8 BGR background of the sequence, pair or image.
        shape_descriptors (List[Dict[str, Any]]): The shapes of the frame, in the order of the instance map.
    """
    # Every pixel of a shape differs from the background, every other pixel is the background
    assert np.array_equal(np.any(frame != background, axis=-1), instance_map > 0)
    assert int(instance_map.max()) <= len(shape_descriptors)

    # Every shape shows its colour on its visible share of its pixels
    visible_pixel_counts = np.bincount(instance_map.reshape(-1), minlength=len(shape_descriptors) + 1)[1:]
    for shape_index, (shape_descriptor, visible_pixel_count) in enumerate(zip(shape_descriptors, visible_pixel_counts,
                                                                              strict=True)):
        assert visible_pixel_count >= SHAPE_MINIMUM_VISIBLE_FRACTION * np.count_nonzero(shape_descriptor["mask"])
        assert np.all(frame[instance_map == shape_index + 1] == shape_descriptor["colour"])


def are_same_shapes(first_shapes: List[Dict[str, Any]], second_shapes: List[Dict[str, Any]]) -> bool:
    """
    Tells whether two lists of shape descriptors hold the same shapes in the same order, the very same descriptors
    and not equal copies, so that a test knows that a shape kept its class, size, colour and pixels from frame to frame.

    Args:
        first_shapes (List[Dict[str, Any]]): The first shapes.
        second_shapes (List[Dict[str, Any]]): The second shapes.

    Returns:
        bool: True when both lists hold the same descriptors in the same order.
    """
    return len(first_shapes) == len(second_shapes) and all(
        first_shape is second_shape for first_shape, second_shape in zip(first_shapes, second_shapes))


@pytest.mark.parametrize(argnames="shape_name", argvalues=["triangle"])
def test_shape_boxes_are_tight(shape_name: str) -> None:
    """
    Verifies that the box computed for every shape class is tight: every drawn pixel lies inside it, and each of its
    four borders touches the shape.

    Args:
        shape_name (str): The shape class under test.
    """
    random_generator = np.random.default_rng(seed=0)

    for _ in range(20):
        mask = np.zeros(shape=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        SHAPE_DRAWING_FUNCTIONS[shape_name](mask=mask, center_x=128, center_y=128, size=64,
                                            random_generator=random_generator)
        x_1, y_1, x_2, y_2 = get_mask_bounding_box(mask=mask)

        # Every shape pixel is inside the box, and the first and last rows and columns of the box contain the shape
        assert mask.sum() == mask[y_1:y_2, x_1:x_2].sum()
        assert mask[y_1, x_1:x_2].any() and mask[y_2 - 1, x_1:x_2].any()
        assert mask[y_1:y_2, x_1].any() and mask[y_1:y_2, x_2 - 1].any()


@pytest.mark.parametrize(argnames="shape_name", argvalues=["parallelogram", "capsule"])
def test_every_shape_stays_inside_its_square(shape_name: str) -> None:
    """
    Verifies that every shape class, whatever its rotation, draws its pixels inside the 'size' x 'size' square around
    its centre, the canvas of draw_shape_descriptor, so that no shape is cut by its canvas, down to OBJECT_MINIMUM_SIZE.

    Args:
        shape_name (str): The shape class under test.
    """
    random_generator = np.random.default_rng(seed=0)

    for size in [OBJECT_MINIMUM_SIZE, 33, 101]:
        for _ in range(20):
            mask = np.zeros(shape=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            SHAPE_DRAWING_FUNCTIONS[shape_name](mask=mask, center_x=128, center_y=128, size=size,
                                                random_generator=random_generator)
            row_indices, column_indices = np.nonzero(mask)
            assert row_indices.min() >= 128 - size // 2 and row_indices.max() <= 128 + size // 2
            assert column_indices.min() >= 128 - size // 2 and column_indices.max() <= 128 + size // 2


@pytest.mark.parametrize(argnames="shape_name", argvalues=["square", "triangle", "capsule"])
@pytest.mark.parametrize(argnames="size", argvalues=[64])
@pytest.mark.parametrize(argnames="upright", argvalues=[False])
def test_hollow_shapes_are_closed_rings_inside_their_filled_shape(shape_name: str, size: int, upright: bool) -> None:
    """
    Verifies that every hollow shape is its filled shape with one hole cut inside it, like the annulus is the hollow
    circle: drawn with the same random generator state, its pixels are a strict part of the filled shape, they form one
    group, and the hole is a background region of its own, apart from the background around the walls. The hollow shape
    keeps the keypoints of its filled shape, followed for HOLLOW_INNER_KEYPOINT_SHAPE_NAMES by the corners of the hole,
    each less than one pixel away from a hole pixel, while the distractors of HOLLOW_DISTRACTOR_SHAPE_NAMES get none.

    Args:
        shape_name (str): The name of the filled shape of HOLLOW_SHAPE_NAMES.
        size (int): The side of the square the shape is drawn in.
        upright (bool): Whether the shape is drawn upright or rotated.
    """
    random_generator = np.random.default_rng(seed=0)

    for _ in range(20):
        # Draw the filled shape and the hollow shape from the same random generator state
        filled_mask = np.zeros(shape=(128, 128), dtype=np.uint8)
        hollow_mask = np.zeros(shape=(128, 128), dtype=np.uint8)
        generator_state = random_generator.bit_generator.state
        filled_keypoints = SHAPE_DRAWING_FUNCTIONS[shape_name](mask=filled_mask, center_x=64, center_y=64, size=size,
                                                               random_generator=random_generator, upright=upright)
        random_generator.bit_generator.state = generator_state
        hollow_keypoints = SHAPE_DRAWING_FUNCTIONS[f"hollow_{shape_name}"](
            mask=hollow_mask, center_x=64, center_y=64, size=size, random_generator=random_generator, upright=upright)

        # The walls lie inside the filled shape and cover less of it
        assert np.all(filled_mask[hollow_mask > 0] > 0)
        assert np.count_nonzero(hollow_mask) < np.count_nonzero(filled_mask)

        # The walls form one group closed around a single hole
        number_ring_groups, _ = cv2.connectedComponents(image=(hollow_mask > 0).astype(np.uint8), connectivity=8)
        number_background_groups, background_groups = cv2.connectedComponents(
            image=(hollow_mask == 0).astype(np.uint8), connectivity=4)
        assert number_ring_groups - 1 == 1 and number_background_groups - 1 == 2

        # The distractors have no keypoints, the other shapes keep the outer keypoints of their filled shape
        if shape_name in HOLLOW_DISTRACTOR_SHAPE_NAMES:
            assert hollow_keypoints.shape == (0, 2)
            continue
        number_outer_keypoints = len(filled_keypoints)
        np.testing.assert_array_equal(hollow_keypoints[:number_outer_keypoints], filled_keypoints)
        if shape_name not in HOLLOW_INNER_KEYPOINT_SHAPE_NAMES:
            assert len(hollow_keypoints) == number_outer_keypoints
            continue

        # Every corner of the hole lies less than one pixel away from the centre of a hole pixel
        assert len(hollow_keypoints) == 2 * number_outer_keypoints
        hole_rows, hole_columns = np.nonzero(background_groups != background_groups[0, 0])
        hole_pixel_centres = np.stack([hole_columns + 0.5, hole_rows + 0.5], axis=1)
        inner_keypoints = hollow_keypoints[number_outer_keypoints:]
        hole_distances = np.linalg.norm(inner_keypoints[:, np.newaxis] - hole_pixel_centres[np.newaxis], axis=2)
        assert np.all(hole_distances.min(axis=1) < 1.0)


@pytest.mark.parametrize(argnames=("keypoints", "size"), argvalues=[(False, OBJECT_MINIMUM_SIZE),
                                                                    (True, KEYPOINT_OBJECT_MINIMUM_SIZE)])
def test_hollow_shapes_keep_one_hole_at_the_smallest_size(keypoints: bool, size: int) -> None:
    """
    Verifies that every hollow class drawn by draw_shape_descriptor at the smallest size of the images
    (OBJECT_MINIMUM_SIZE, KEYPOINT_OBJECT_MINIMUM_SIZE for the upright keypoint objects), the regular polygons included,
    is one ring closed around one hole, so that a small hollow shape still shows the hole that tells it from its filled
    class.

    Args:
        keypoints (bool): Whether the shapes are keypoint objects, drawn upright.
        size (int): The size of the shapes.
    """
    random_generator = np.random.default_rng(seed=0)
    background_colour = np.zeros(shape=(3,), dtype=np.uint8)
    for shape_name in HOLLOW_CLASS_NAMES:
        for _ in range(10):
            shape_descriptor = draw_shape_descriptor(minimum_size=size, maximum_size=size,
                                                     background_colour=background_colour,
                                                     random_generator=random_generator, keypoints=keypoints,
                                                     shape_name=shape_name)

            # Canvas padded by one pixel, so that the background around the walls is one region
            ring_mask = np.pad(array=(shape_descriptor["mask"] > 0).astype(np.uint8), pad_width=1)
            number_ring_groups, _ = cv2.connectedComponents(image=ring_mask, connectivity=8)
            number_background_groups, _ = cv2.connectedComponents(image=1 - ring_mask, connectivity=4)
            assert number_ring_groups - 1 == 1 and number_background_groups - 1 == 2


@pytest.mark.parametrize(argnames="keypoints", argvalues=[False])
def test_hollow_shapes_are_drawn_at_their_minimum_size(keypoints: bool) -> None:
    """
    Verifies that draw_shape_descriptor raises the size of every hollow class to HOLLOW_MINIMUM_SIZE when its range is
    smaller, as in an image too small for OBJECT_MINIMUM_SIZE, where a hollow parallelogram of 8 pixels leaves no hole
    in 16% of its drawings, so that the hole always fits inside the walls, while a filled class keeps the size of its
    range.

    Args:
        keypoints (bool): Whether the objects are keypoint objects, drawn upright with their keypoints.
    """
    random_generator = np.random.default_rng(seed=0)
    background_colour = np.zeros(shape=(3,), dtype=np.uint8)
    for shape_name in HOLLOW_CLASS_NAMES:
        for _ in range(20):
            shape_descriptor = draw_shape_descriptor(minimum_size=8, maximum_size=8,
                                                     background_colour=background_colour,
                                                     random_generator=random_generator, keypoints=keypoints,
                                                     shape_name=shape_name)
            assert shape_descriptor["mask"].shape == (HOLLOW_MINIMUM_SIZE + 1, HOLLOW_MINIMUM_SIZE + 1)
    filled_descriptor = draw_shape_descriptor(minimum_size=8, maximum_size=8, background_colour=background_colour,
                                              random_generator=random_generator, keypoints=keypoints,
                                              shape_name="parallelogram")
    assert filled_descriptor["mask"].shape == (9, 9)


def test_regular_polygons_keep_their_minimum_size() -> None:
    """
    Verifies that draw_shape_descriptor draws every regular polygon of SHAPE_MINIMUM_SIZES, filled or hollow, from the
    size at which the middle of its sides lies 2 pixels inside the circle through its vertices (21 pixels for the
    pentagon to 67 for the nonagon), below which it looks like the circle, and within a range ending below that size,
    while the other classes keep the whole range.
    """
    assert {shape_name: SHAPE_MINIMUM_SIZES[shape_name] for shape_name in
            ["pentagon", "hexagon", "heptagon", "octagon", "nonagon"]} == {"pentagon": 21, "hexagon": 30,
                                                                          "heptagon": 41, "octagon": 53, "nonagon": 67}
    assert all(SHAPE_MINIMUM_SIZES[f"hollow_{shape_name}"] == SHAPE_MINIMUM_SIZES[shape_name]
               for shape_name in ["pentagon", "hexagon", "heptagon", "octagon", "nonagon"])
    random_generator = np.random.default_rng(seed=0)
    background_colour = np.zeros(shape=(3,), dtype=np.uint8)

    # Every polygon drawn from its minimum size, a range ending below it giving its largest size
    for shape_name, minimum_size in SHAPE_MINIMUM_SIZES.items():
        sizes = [draw_shape_descriptor(minimum_size=OBJECT_MINIMUM_SIZE, maximum_size=100,
                                       background_colour=background_colour, random_generator=random_generator,
                                       shape_name=shape_name)["mask"].shape[0] - 1 for _ in range(20)]
        assert minimum_size <= min(sizes) and max(sizes) <= 100
        capped_descriptor = draw_shape_descriptor(minimum_size=OBJECT_MINIMUM_SIZE, maximum_size=20,
                                                  background_colour=background_colour,
                                                  random_generator=random_generator, shape_name=shape_name)
        assert capped_descriptor["mask"].shape[0] - 1 == 20

    # Another class keeps the smallest sizes of the range
    circle_sizes = [draw_shape_descriptor(minimum_size=OBJECT_MINIMUM_SIZE, maximum_size=100,
                                          background_colour=background_colour, random_generator=random_generator,
                                          shape_name="circle")["mask"].shape[0] - 1 for _ in range(100)]
    assert min(circle_sizes) < 21


@pytest.mark.parametrize(argnames="category_name", argvalues=["square", "circle", "annulus", "capsule"])
@pytest.mark.parametrize(argnames="upright", argvalues=[False])
def test_keypoint_drawing_functions_give_one_point_per_keypoint_name(category_name: str, upright: bool) -> None:
    """
    Verifies that every keypoint category draws one keypoint per name of KEYPOINT_NAMES, at any size and rotation, so
    that the annotations of build_keypoint_labels hold 3 values per keypoint of their category: none for the circle,
    the ellipse, the annulus and the hollow ellipse, the distractors of the keypoint images, and four tips without
    centre for the plus sign.

    Args:
        category_name (str): The name of the keypoint category of KEYPOINT_CATEGORIES.
        upright (bool): Whether the shape is drawn upright or rotated.
    """
    random_generator = np.random.default_rng(seed=0)
    for size in [KEYPOINT_OBJECT_MINIMUM_SIZE, 64, 101]:
        mask = np.zeros(shape=(128, 128), dtype=np.uint8)
        keypoints = KEYPOINT_DRAWING_FUNCTIONS[category_name](mask=mask, center_x=64, center_y=64, size=size,
                                                              random_generator=random_generator, upright=upright)
        assert keypoints.shape == (len(KEYPOINT_NAMES[category_name]), 2) and np.any(mask)
    assert all(not KEYPOINT_NAMES[name] for name in ["circle", "ellipse", "annulus", "hollow_ellipse"])
    assert "centre" not in KEYPOINT_NAMES["plus_sign"] and "centre" not in KEYPOINT_NAMES["capsule"]


def test_upright_capsule_keypoints_lie_on_its_outline() -> None:
    """
    Verifies that the 6 keypoints of an upright capsule lie on its outline, without centre: the left and right ends of
    the long axis on the leftmost and rightmost columns of its pixels, at the middle of its height, and the four
    corners where the straight sides meet the round ends on its top and bottom rows, symmetric about its centre.
    """
    random_generator = np.random.default_rng(seed=0)
    for _ in range(20):
        mask = np.zeros(shape=(128, 128), dtype=np.uint8)
        keypoints = KEYPOINT_DRAWING_FUNCTIONS["capsule"](mask=mask, center_x=64, center_y=64, size=101,
                                                          random_generator=random_generator, upright=True)
        named_keypoints = dict(zip(KEYPOINT_NAMES["capsule"], keypoints, strict=True))
        rows, columns = np.nonzero(mask)

        # The ends of the long axis on the left and right edges of the pixels, at the middle of the height
        assert abs(named_keypoints["left_end"][0] - columns.min()) <= 1.0
        assert abs(named_keypoints["right_end"][0] - (columns.max() + 1)) <= 1.0
        assert abs(named_keypoints["left_end"][1] - (rows.min() + rows.max() + 1) / 2) <= 1.0

        # The corners on the top and bottom edges, the top ones above the bottom ones and the left ones on the left
        for corner_name in ["top_left", "top_right"]:
            assert abs(named_keypoints[corner_name][1] - rows.min()) <= 1.0
        for corner_name in ["bottom_left", "bottom_right"]:
            assert abs(named_keypoints[corner_name][1] - (rows.max() + 1)) <= 1.0
        np.testing.assert_allclose(named_keypoints["top_left"][0], named_keypoints["bottom_left"][0])
        assert (named_keypoints["left_end"][0] < named_keypoints["top_left"][0] < named_keypoints["top_right"][0]
                < named_keypoints["right_end"][0])


def test_backgrounds_are_flat_or_textured_around_their_colour() -> None:
    """
    Verifies draw_background, shared by the images and sequences of every task: about half of the backgrounds are
    textured, changing their colour by up to TEXTURE_AMPLITUDE levels on every channel, the others are flat; and a
    shape colour MINIMUM_COLOUR_DIFFERENCE levels from the background colour still differs from every pixel of a
    texture by the smallest difference of a low contrast shape, so that a shape stays at least as visible on a texture
    as a low contrast shape on a flat background.
    """
    random_generator = np.random.default_rng(seed=0)
    textured_flags = []
    for _ in range(100):
        image, background_colour, textured = draw_background(image_size=64, random_generator=random_generator)
        changes = np.abs(image.astype(np.int64) - background_colour.astype(np.int64))
        assert changes.max() <= TEXTURE_AMPLITUDE and (changes.max() > 0) == textured
        textured_flags.append(textured)
    assert 0.35 <= np.mean(textured_flags) <= 0.65
    assert MINIMUM_COLOUR_DIFFERENCE - TEXTURE_AMPLITUDE >= LOW_CONTRAST_DIFFERENCE_RANGE[0]


@pytest.mark.parametrize(argnames="textured", argvalues=[False, True])
def test_shape_colours_stand_apart_from_each_other_and_from_the_background(textured: bool) -> None:
    """
    Verifies the colours of draw_shape_descriptors: every shape differs from the other shapes of its image by
    SHAPE_COLOUR_MINIMUM_DIFFERENCE levels on one channel at least, so that two overlapping shapes keep a visible edge,
    and from the background colour by MINIMUM_COLOUR_DIFFERENCE, except about a fifth of the shapes of a flat
    background, which differ from it by LOW_CONTRAST_DIFFERENCE_RANGE only, a textured background getting none.

    Args:
        textured (bool): Whether the shapes are drawn for a textured background.
    """
    random_generator = np.random.default_rng(seed=0)
    background_differences = []
    for _ in range(50):
        background_colour = random_generator.integers(low=0, high=256, size=3, dtype=np.uint8)
        shape_descriptors = draw_shape_descriptors(shape_names=[None] * 10, minimum_size=OBJECT_MINIMUM_SIZE,
                                                   maximum_size=32, background_colour=background_colour,
                                                   textured=textured, random_generator=random_generator)
        colours = np.array([shape_descriptor["colour"] for shape_descriptor in shape_descriptors], dtype=np.int64)

        # Colours of two shapes of the image apart on their most different channel
        pair_differences = np.abs(colours[:, np.newaxis] - colours[np.newaxis]).max(axis=-1)
        assert np.all(pair_differences[~np.eye(len(colours), dtype=bool)] >= SHAPE_COLOUR_MINIMUM_DIFFERENCE)
        background_differences.extend(np.abs(colours - background_colour.astype(np.int64)).max(axis=-1).tolist())

    # Contrasting shapes, and the low contrast ones within their range, on the flat backgrounds only
    background_differences = np.array(background_differences)
    low_contrast = background_differences < MINIMUM_COLOUR_DIFFERENCE
    assert np.all((background_differences[low_contrast] >= LOW_CONTRAST_DIFFERENCE_RANGE[0]) &
                  (background_differences[low_contrast] <= LOW_CONTRAST_DIFFERENCE_RANGE[1]))
    if textured:
        assert not low_contrast.any()
    else:
        assert 0.1 <= low_contrast.mean() <= 0.3


def test_low_contrast_shape_without_a_free_low_contrast_colour_gets_a_contrasting_colour() -> None:
    """
    Verifies the case that stopped the creation of a 512 pixel multi-label classification dataset of up to 10 shapes:
    on a black background, seven low contrast shapes of the image take every low contrast colour (each one lies within
    SHAPE_COLOUR_MINIMUM_DIFFERENCE levels of one of them), so that the tenth shape, drawn with a low contrast, has no
    colour apart from the nine others; draw_shape_descriptor then gives it a contrasting colour,
    MINIMUM_COLOUR_DIFFERENCE levels from the background and apart from every other shape, where
    draw_contrasting_colour raised a RuntimeError.
    """
    background_colour = np.zeros(shape=3, dtype=np.uint8)
    low_contrast_colours = [np.array(levels, dtype=np.uint8) for levels in itertools.product([10, 30], repeat=3)
                            if max(levels) == 30]
    shape_colours = (*low_contrast_colours, np.array([200, 90, 140], dtype=np.uint8),
                     np.array([0, 0, 255], dtype=np.uint8))
    other_colours = np.array(shape_colours, dtype=np.int64)

    # Every low contrast colour of a black background (every channel within the largest difference, one channel at
    # least the smallest) lies too close to one of the low contrast shapes
    candidate_colours = np.array(list(itertools.product(range(LOW_CONTRAST_DIFFERENCE_RANGE[1] + 1), repeat=3)))
    candidate_colours = candidate_colours[candidate_colours.max(axis=-1) >= LOW_CONTRAST_DIFFERENCE_RANGE[0]]
    candidate_differences = np.abs(candidate_colours[:, np.newaxis] - other_colours[np.newaxis]).max(axis=-1)
    assert np.all(candidate_differences.min(axis=-1) < SHAPE_COLOUR_MINIMUM_DIFFERENCE)

    # The shape drawn with a low contrast gets a contrasting colour, still apart from the nine other shapes
    random_generator = np.random.default_rng(seed=0)
    for _ in range(3):
        shape_descriptor = draw_shape_descriptor(minimum_size=OBJECT_MINIMUM_SIZE, maximum_size=32,
                                                 background_colour=background_colour, random_generator=random_generator,
                                                 low_contrast_range=LOW_CONTRAST_DIFFERENCE_RANGE,
                                                 shape_colours=shape_colours)
        colour = shape_descriptor["colour"].astype(np.int64)
        assert colour.max() >= MINIMUM_COLOUR_DIFFERENCE
        assert np.abs(other_colours - colour).max(axis=-1).min() >= SHAPE_COLOUR_MINIMUM_DIFFERENCE


@pytest.mark.parametrize(argnames="task", argvalues=list(TASK_SAMPLE_MAXIMUM_OBJECTS))
def test_every_task_draws_the_shared_ingredients(monkeypatch: pytest.MonkeyPatch, task: str) -> None:
    """
    Verifies that the images and frames of every task share the ingredients of the synthetic datasets: one background
    per image, sequence or pair, textured for about half of them and flat for the others; a colour within
    LOW_CONTRAST_DIFFERENCE_RANGE of a flat background for part of the shapes, never on a textured background; and
    shapes placed partly outside the image and over the shapes placed before them where the samples hold several shapes
    (the moving shape of the sequence classification, placed along its own path, being followed by its own test).

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture recording the backgrounds, colours and placements.
        task (str): The task whose samples are drawn.
    """
    background_recorder = CallRecorder(function=synthetic_dataset_module.draw_background)
    colour_recorder = CallRecorder(function=synthetic_dataset_module.draw_contrasting_colour)
    placement_recorder = CallRecorder(function=synthetic_dataset_module.place_shape, copy_lists=True)
    for function_name, recorder in [("draw_background", background_recorder),
                                    ("draw_contrasting_colour", colour_recorder), ("place_shape", placement_recorder)]:
        monkeypatch.setattr(target=synthetic_dataset_module, name=function_name, value=recorder)
    random_generator = np.random.default_rng(seed=0)
    number_samples = 40
    textured_samples, flat_shape_colours, low_contrast_colours, cut_shapes, overlapping_shapes = 0, 0, 0, 0, 0

    for sample_index in range(number_samples):
        first_colour_call, first_placement_call = len(colour_recorder.calls), len(placement_recorder.calls)
        draw_task_sample(task=task, image_size=SAMPLE_IMAGE_SIZE, maximum_objects=TASK_SAMPLE_MAXIMUM_OBJECTS[task],
                         random_generator=random_generator)

        # One background per sample, shared by its frames
        assert len(background_recorder.calls) == sample_index + 1
        textured = background_recorder.results[-1][2]
        textured_samples += textured

        # Colours of the shapes (the parts of a stick figure taking no low contrast), low contrast on flat backgrounds
        shape_colour_calls = [call for call in colour_recorder.calls[first_colour_call:]
                              if "low_contrast_range" in call]
        low_contrast_calls = [call for call in shape_colour_calls if call["low_contrast_range"] is not None]
        assert not (textured and low_contrast_calls)
        if not textured:
            flat_shape_colours += len(shape_colour_calls)
            low_contrast_colours += len(low_contrast_calls)

        # Shapes placed partly outside the image, and over the shapes placed before them
        for call, box in zip(placement_recorder.calls[first_placement_call:],
                             placement_recorder.results[first_placement_call:], strict=True):
            if box is None:
                continue
            cut_shapes += box[0] < 0 or box[1] < 0 or box[2] > SAMPLE_IMAGE_SIZE or box[3] > SAMPLE_IMAGE_SIZE
            clipped_box = [max(0, box[0]), max(0, box[1]), min(SAMPLE_IMAGE_SIZE, box[2]),
                           min(SAMPLE_IMAGE_SIZE, box[3])]
            overlapping_shapes += any(get_intersection_over_union(first_box=clipped_box, second_box=placed_box) > 0
                                      for placed_box in call["placed_boxes"])

    assert 0.25 <= textured_samples / number_samples <= 0.75
    assert 0 < low_contrast_colours < 0.5 * flat_shape_colours
    if task != "sequence_classification":
        assert cut_shapes > 0
    if TASK_SAMPLE_MAXIMUM_OBJECTS[task] > 1:
        assert overlapping_shapes > 0


@pytest.mark.parametrize(argnames=("keypoints", "maximum_objects", "minimum_size", "minimum_objects"), argvalues=[
    (False, 12, OBJECT_MINIMUM_SIZE, MINIMUM_OBJECTS_PER_IMAGE),
    (True, 5, KEYPOINT_OBJECT_MINIMUM_SIZE, KEYPOINT_MINIMUM_OBJECTS)])
def test_image_shapes_draw_log_uniform_sizes_from_the_largest(monkeypatch: pytest.MonkeyPatch, keypoints: bool,
                                                               maximum_objects: int, minimum_size: int,
                                                               minimum_objects: int) -> None:
    """
    Verifies that an image of generate_image draws 'minimum_objects' (MINIMUM_OBJECTS_PER_IMAGE,
    KEYPOINT_MINIMUM_OBJECTS for the keypoint images) to maximum_objects shapes, from 'minimum_size' pixels
    (OBJECT_MINIMUM_SIZE, KEYPOINT_OBJECT_MINIMUM_SIZE for the keypoint objects, the SHAPE_MINIMUM_SIZES of a regular
    polygon) to OBJECT_MAXIMUM_SIZE_FRACTION of the image side, placed from the largest to the smallest so that the
    small shapes lie over the large ones, and log-uniform: about half of the shapes fall below the geometric mean of the
    size range, where a uniform draw in pixels would put about a fifth.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture recording the shapes drawn and placed.
        keypoints (bool): Whether the images are keypoint images.
        maximum_objects (int): The largest number of shapes of an image.
        minimum_size (int): The expected smallest size of the shapes.
        minimum_objects (int): The expected smallest number of shapes of an image.
    """
    random_generator = np.random.default_rng(seed=0)
    maximum_size = int(OBJECT_MAXIMUM_SIZE_FRACTION * IMAGE_SIZE)
    range_sizes = []

    for _ in range(100):
        descriptor_recorder = CallRecorder(function=synthetic_dataset_module.draw_shape_descriptor)
        placement_recorder = CallRecorder(function=synthetic_dataset_module.place_shape)
        monkeypatch.setattr(target=synthetic_dataset_module, name="draw_shape_descriptor", value=descriptor_recorder)
        monkeypatch.setattr(target=synthetic_dataset_module, name="place_shape", value=placement_recorder)
        generate_image(image_size=IMAGE_SIZE, maximum_objects=maximum_objects, random_generator=random_generator,
                       keypoints=keypoints)

        # Shapes drawn within the size range, the regular polygons from their own minimum size
        assert minimum_objects <= len(descriptor_recorder.calls) <= maximum_objects
        assert all((call["minimum_size"], call["maximum_size"]) == (minimum_size, maximum_size)
                   for call in descriptor_recorder.calls)
        for shape_descriptor in descriptor_recorder.results:
            size = shape_descriptor["mask"].shape[0] - 1
            shape_minimum_size = (max(minimum_size, SHAPE_MINIMUM_SIZES[shape_descriptor["shape_name"]])
                                  if shape_descriptor["shape_name"] in SHAPE_MINIMUM_SIZES else minimum_size)
            assert shape_minimum_size <= size <= maximum_size
            if shape_descriptor["shape_name"] not in SHAPE_MINIMUM_SIZES:
                range_sizes.append(size)

        # Every shape placed once, from the largest to the smallest
        placed_sizes = [call["shape_descriptor"]["mask"].shape[0] - 1 for call in placement_recorder.calls]
        assert len(placed_sizes) == len(descriptor_recorder.calls)
        assert placed_sizes == sorted(placed_sizes, reverse=True)

    # Log-uniform sizes: half of them below the geometric mean of the range
    geometric_mean = np.sqrt(minimum_size * maximum_size)
    assert 0.4 < np.mean(np.array(range_sizes) < geometric_mean) < 0.6


@pytest.mark.parametrize(argnames=("task", "maximum_objects", "expected_size_range"), argvalues=[
    ("sequence_labelling", 8, (OBJECT_MINIMUM_SIZE, 181)), ("self_supervised", 10, (OBJECT_MINIMUM_SIZE, 161)),
    ("sequence_comparison", 6, (OBJECT_MINIMUM_SIZE, 209))])
def test_frame_shapes_get_the_frame_size_range(monkeypatch: pytest.MonkeyPatch, task: str, maximum_objects: int,
                                               expected_size_range: Tuple[int, int]) -> None:
    """
    Verifies that the shapes of the sequence labelling frames, of the counted shapes images (drawn like those frames, so
    that the backbone they train sees the shapes of the frames it then initialises on) and of the comparison pairs
    measure OBJECT_MINIMUM_SIZE pixels up to the size at which the most shapes of a 512 pixel frame would cover it once
    (512 x sqrt(1 / 8) = 181 pixels for 8 shapes), so that every shape fits visible enough.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture recording the shape descriptors drawn.
        task (str): The task whose sample is drawn.
        maximum_objects (int): The largest number of shapes of a frame.
        expected_size_range (Tuple[int, int]): The expected smallest and largest shape sizes.
    """
    shape_descriptor_recorder = CallRecorder(function=synthetic_dataset_module.draw_shape_descriptor)
    monkeypatch.setattr(target=synthetic_dataset_module, name="draw_shape_descriptor", value=shape_descriptor_recorder)
    draw_task_sample(task=task, image_size=512, maximum_objects=maximum_objects,
                     random_generator=np.random.default_rng(seed=0))

    assert len(shape_descriptor_recorder.calls) >= PAIR_MINIMUM_OBJECTS
    assert all((call["minimum_size"], call["maximum_size"]) == expected_size_range
               for call in shape_descriptor_recorder.calls)


def test_created_dataset_is_valid_coco_and_loads(tmp_path: Path) -> None:
    """
    Verifies that the created dataset has the folder layout and COCO annotations expected by the data loaders:
    every image listed with its size, boxes inside the images, every shape category, a label mapping for the
    multi-class mode, and that the Native dataset loads it with the category identifiers as labels.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_synthetic_dataset(output_folder=str(tmp_path), image_size=IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 2, "test": 2}, maximum_objects=4, minimum_frames=1,
                             maximum_frames=1, number_workers=2, task="object_detection")

    labels = json.loads((tmp_path / "train" / "labels.json").read_text())
    assert len(labels["images"]) == 8 and len(labels["annotations"]) >= 8
    assert [category["name"] for category in labels["categories"]] == list(SHAPE_CATEGORIES)
    assert all((tmp_path / "train" / "data" / image["file_name"]).is_file() for image in labels["images"])
    assert (tmp_path / "validation" / "labels.json").is_file() and (tmp_path / "test" / "labels.json").is_file()

    # Every COCO [x, y, width, height] box lies inside its image and has a positive size
    for annotation in labels["annotations"]:
        left, top, width, height = annotation["bbox"]
        assert width > 0 and height > 0 and left >= 0 and top >= 0
        assert left + width <= IMAGE_SIZE and top + height <= IMAGE_SIZE

    # The multi-class label mapping keeps every raw category identifier as its own model class
    label_mapping = json.loads((tmp_path / "label_mapping.json").read_text())
    label_dictionary = {int(raw_identifier): model_identifier for raw_identifier, model_identifier in
                        label_mapping.items()}
    label_dictionary["background"] = 0

    # The detection dataset loads every image with its boxes, the category identifiers as labels
    dataset = CocoDetectionDataset(data_directory=str(tmp_path / "train" / "data"),
                                   labels_file=str(tmp_path / "train" / "labels.json"), image_size=IMAGE_SIZE,
                                   keep_ratio=True, label_mode="multi_class", label_dictionary=label_dictionary)
    loaded_labels = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample["bounding_boxes"].shape[0] == sample["labels"].shape[0] > 0
        loaded_labels |= set(sample["labels"].tolist())
    assert loaded_labels <= set(SHAPE_CATEGORIES.values())


@pytest.mark.parametrize("task, maximum_objects, maximum_frames, sample_name, count_name",
                         [("classification", 1, 1, "Images", None), ("object_detection", 4, 1, "Images", "Objects"),
                          ("sequence_labelling", 5, 4, "Sequences", "Frames")])
def test_created_dataset_prints_a_panel_per_split_and_the_generation_times(tmp_path: Path,
                                                                          capsys: pytest.CaptureFixture, task: str,
                                                                          maximum_objects: int, maximum_frames: int,
                                                                          sample_name: str,
                                                                          count_name: Optional[str]) -> None:
    """
    Verifies the console of the generator: every split gets its panel with its samples, its labels when they are not
    the samples (objects, frames), its class range, its generation time and rate, and the count of every class, and
    the closing panel gives the label mapping file and the samples and generation time of every split and of the whole
    dataset, instead of the dictionary of class counts printed before.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The captured console output.
        task (str): The task of the dataset.
        maximum_objects (int): The largest number of shapes per image or frame.
        maximum_frames (int): The largest number of frames of a sequence (1 for the image tasks).
        sample_name (str): The samples named in the panels.
        count_name (Optional[str]): What the class counts count when it is not the samples, None otherwise.
    """
    split_sizes = {"train": 6, "validation": 3, "test": 2}
    create_synthetic_dataset(output_folder=str(tmp_path), image_size=SAMPLE_IMAGE_SIZE, split_sizes=split_sizes,
                             maximum_objects=maximum_objects, minimum_frames=min(2, maximum_frames),
                             maximum_frames=maximum_frames, number_workers=2, task=task)
    output = capsys.readouterr().out

    # One panel per split, with its samples, generation time and rate, and the classes of the task in its table
    for split_name, number_samples in split_sizes.items():
        assert f"{split_name.title()} Split" in output
        assert re.search(rf"│ {sample_name} +: {number_samples} ", output)
    assert output.count("Generation Time") == len(split_sizes) + 1
    assert output.count(f"{sample_name} Per Second") == 2 * len(split_sizes) + 1
    assert (re.search(rf"Labelled {count_name} +: \d", output) is not None) == (count_name is not None)
    class_name = list(synthetic_dataset_module.TASK_CATEGORIES[task])[-1]
    assert output.count(f" {class_name} ") == len(split_sizes)
    assert "Labels per class" not in output

    # Closing panel: the label mapping file, a row per split and the total, in the HH:MM:SS layout with the seconds
    assert "Synthetic Dataset" in output and "Label Mapping File : label_mapping.json" in output
    total_samples = sum(split_sizes.values())
    assert re.search(rf"│ +Total +{total_samples} .*\d\d:\d\d:\d\d \(\d+\.\d seconds\)", output)
    assert all(len(line) <= SUMMARY_PANEL_WIDTH for line in output.splitlines() if line.startswith(("│", "╭", "╰")))


@pytest.mark.parametrize("arguments, expected_message",
                         [(["--output_folder", "dataset"], "the following arguments are required: --task"),
                          (["--task", "classifier", "--output_folder", "dataset"],
                           "argument --task: invalid choice: 'classifier'"),
                          (["--task", "classification"], "the following arguments are required: --output_folder")])
def test_generator_takes_its_task_and_folder_from_the_command_line(tmp_path: Path, arguments: List[str],
                                                                   expected_message: str) -> None:
    """
    Verifies the command line of the generator, which takes the task of the dataset as a required argument instead of
    a variable of its main block: a missing task, a task that is not one of the generator and a missing folder stop
    it before anything is written, and its help lists every task of the generator.

    Args:
        tmp_path (Path): The pytest temporary directory, the folder the refused commands run from.
        arguments (List[str]): The arguments given to the generator.
        expected_message (str): The start of the error printed by its argument parser.
    """
    command = [sys.executable, "-m", "vision_training.data.synthetic.create_synthetic_dataset"]
    # Run from the temporary folder, the repository on the module path, so that a dataset written by mistake lands there
    completed_process = subprocess.run(args=command + arguments, cwd=tmp_path, capture_output=True, text=True,
                                       env={**os.environ, "PYTHONPATH": str(REPOSITORY_FOLDER)})
    assert completed_process.returncode == 2
    assert expected_message in completed_process.stderr
    assert list(tmp_path.iterdir()) == []

    # The help lists the eleven tasks of the generator as the choices of --task
    help_process = subprocess.run(args=command + ["--help"], cwd=REPOSITORY_FOLDER, capture_output=True, text=True)
    assert help_process.returncode == 0
    assert all(task in help_process.stdout for task in TASK_SAMPLE_NAMES)


def test_class_count_table_fits_the_panel_and_marks_the_empty_classes() -> None:
    """
    Verifies the class table of a split panel: the classes read column by column in as many class and count columns
    as fit in the panel (three for the 35 shape classes, two for the 71 longer change classes of the sequence
    labelling), the counts written with their digits grouped by three, a class without labels in yellow, and that a
    generation time that is not positive gives no rate.
    """
    shape_counts = {class_name: 1000 * index for index, class_name in enumerate(SHAPE_CATEGORIES)}
    shape_table = build_class_count_table(class_counts=shape_counts, count_name="objects")
    assert len(shape_table.columns) == 6 and shape_table.row_count == 12
    assert [column.header for column in shape_table.columns[:2]] == ["Class", "Objects"]

    # Down the first column, then the next one, the empty first class in yellow
    first_column_cells = list(shape_table.columns[0].cells)
    assert first_column_cells[0].plain == "square" and first_column_cells[0].style == "yellow"
    assert first_column_cells[1].plain == "circle" and first_column_cells[1].style == ""
    assert list(shape_table.columns[2].cells)[0].plain == list(SHAPE_CATEGORIES)[12]
    assert list(shape_table.columns[1].cells)[11].plain == "11 000"

    # The long change classes take two columns, and the table stays inside the panel
    change_table = build_class_count_table(class_counts={class_name: 7 for class_name in CHANGE_CATEGORIES},
                                           count_name="frames")
    assert len(change_table.columns) == 4 and change_table.row_count == 36
    console = Console(width=SUMMARY_PANEL_WIDTH)
    with console.capture() as capture:
        console.print(change_table)
    assert max(len(line.rstrip()) for line in capture.get().splitlines()) <= SUMMARY_PANEL_WIDTH - 7

    # A rate needs a positive time
    assert format_generation_rate(number_samples=3000, duration_seconds=2.0, sample_name="images") == \
        "1 500.0 Images Per Second"
    with pytest.raises(ValueError, match="must be positive"):
        format_generation_rate(number_samples=3, duration_seconds=0.0, sample_name="images")


def test_created_classification_dataset_has_one_shape_per_image(tmp_path: Path) -> None:
    """
    Verifies the classification mode of the generator: every split gets a manifest with one sample per image, labelled
    with the category of its single shape, a label mapping giving the classes 0 to 34, and the classification dataset
    loads it with its class names.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_synthetic_dataset(output_folder=str(tmp_path), image_size=IMAGE_SIZE,
                             split_sizes={"train": 12, "validation": 2, "test": 2}, maximum_objects=4,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="classification")

    labels = json.loads((tmp_path / "train" / "labels.json").read_text())
    assert len(labels["samples"]) == 12
    assert [category["name"] for category in labels["categories"]] == list(SHAPE_CATEGORIES)
    assert all(sample["category_id"] in SHAPE_CATEGORIES.values() for sample in labels["samples"])
    assert all((tmp_path / "train" / "data" / sample["file_name"]).is_file() for sample in labels["samples"])

    # The classifier predicts no background class: the 35 categories become the classes 0 to 34
    label_mapping = json.loads((tmp_path / "label_mapping.json").read_text())
    assert label_mapping == {str(category_identifier): category_identifier - 1
                             for category_identifier in SHAPE_CATEGORIES.values()}

    dataset = ClassificationDataset(data_directory=str(tmp_path / "train" / "data"),
                                    labels_file=str(tmp_path / "train" / "labels.json"), image_size=IMAGE_SIZE,
                                    keep_ratio=True, label_dictionary={int(category): class_index for category,
                                                                       class_index in label_mapping.items()},
                                    output_mode="multi_class")
    assert dataset.class_names == {class_index: shape_name for class_index, shape_name in enumerate(SHAPE_CATEGORIES)}
    assert sum(dataset.get_class_counts()) == 12


def test_created_segmentation_dataset_masks_match_the_shapes(tmp_path: Path) -> None:
    """
    Verifies the segmentation mode of the generator: every image gets a single-channel mask of its size in 'masks/'
    whose labelled pixels hold shape categories, the 'category_ids' of every sample being the categories visible in its
    mask, a label mapping keeping the category identifiers as classes after the background class 0, and the segmentation
    dataset loads it with its class names (the masks giving exactly the shape pixels being checked on the drawn images,
    whose background may be textured).

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_synthetic_dataset(output_folder=str(tmp_path), image_size=IMAGE_SIZE,
                             split_sizes={"train": 6, "validation": 2, "test": 2}, maximum_objects=4,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="semantic_segmentation")

    labels = json.loads((tmp_path / "train" / "labels.json").read_text())
    assert len(labels["samples"]) == 6
    assert [category["name"] for category in labels["categories"]] == list(SHAPE_CATEGORIES)

    # Every mask is a single-channel map of the shape categories at the size of its image
    for sample in labels["samples"]:
        image = cv2.imread(filename=str(tmp_path / "train" / "data" / sample["file_name"]))
        mask = cv2.imread(filename=str(tmp_path / "train" / "masks" / sample["mask_file_name"]),
                          flags=cv2.IMREAD_UNCHANGED)
        assert mask.shape == image.shape[:2] == (IMAGE_SIZE, IMAGE_SIZE) and mask.dtype == np.uint8
        assert np.any(mask > 0) and set(np.unique(mask[mask > 0]).tolist()) <= set(SHAPE_CATEGORIES.values())

        # The categories of the sample are those visible in its mask, from which data.json counts the masks
        assert sample["category_ids"] == np.unique(mask[mask > 0]).tolist()

    # The segmenter predicts the background as class 0: the 35 categories keep their identifiers as classes
    label_mapping = json.loads((tmp_path / "label_mapping.json").read_text())
    assert label_mapping == {str(category_identifier): category_identifier
                             for category_identifier in SHAPE_CATEGORIES.values()}

    label_dictionary = {**{int(category): class_index for category, class_index in label_mapping.items()},
                        "background": 0}
    dataset = SegmentationDataset(data_directory=str(tmp_path / "train" / "data"),
                                  masks_directory=str(tmp_path / "train" / "masks"),
                                  labels_file=str(tmp_path / "train" / "labels.json"), image_size=IMAGE_SIZE,
                                  keep_ratio=True, label_mode="multi_class", label_dictionary=label_dictionary,
                                  tiling_configuration={"activated": False, "scale": 1.0, "overlap": 0},
                                  tile_sampling="random")
    assert dataset.class_names == {0: "background", **{class_index: shape_name for shape_name, class_index
                                                       in SHAPE_CATEGORIES.items()}}
    assert set(dataset[0]["masks"].unique().tolist()) <= set(range(len(SHAPE_CATEGORIES) + 1))


@pytest.mark.parametrize(argnames="maximum_objects", argvalues=[1, 6])
def test_visible_shapes_fill_the_class_map(monkeypatch: pytest.MonkeyPatch, maximum_objects: int) -> None:
    """
    Verifies the classification (one shape) and semantic segmentation (several shapes) images, drawn with
    SHAPE_MINIMUM_VISIBLE_FRACTION: every shape keeps that share of its pixels visible, a classification image holds its
    single shape, and the class map saved as the segmentation mask gives exactly the pixels that differ from the
    background, flat or textured, with the category of the shape drawn last there.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture recording the backgrounds.
        maximum_objects (int): The largest number of shapes of an image.
    """
    background_recorder = CallRecorder(function=synthetic_dataset_module.draw_background, copy_results=True)
    monkeypatch.setattr(target=synthetic_dataset_module, name="draw_background", value=background_recorder)
    random_generator = np.random.default_rng(seed=0)
    number_objects = 0

    for image_index in range(20):
        object_pixel_counts: List[int] = []
        image, objects, class_map, instance_map = generate_image(
            image_size=IMAGE_SIZE, maximum_objects=maximum_objects, random_generator=random_generator,
            object_pixel_counts=object_pixel_counts, minimum_visible_fraction=SHAPE_MINIMUM_VISIBLE_FRACTION)
        background = background_recorder.results[image_index][0]

        # The pixels differing from the background are the labelled pixels, of the category of their object
        assert np.array_equal(np.any(image != background, axis=-1), class_map > 0)
        object_categories = np.array(object=[0] + [category_identifier for category_identifier, _ in objects])
        assert np.array_equal(object_categories[instance_map], class_map)

        # Every shape keeps its visible share, a classification image its single shape
        visible_pixel_counts = np.bincount(instance_map.reshape(-1), minlength=len(objects) + 1)[1:]
        assert np.all(visible_pixel_counts >= SHAPE_MINIMUM_VISIBLE_FRACTION * np.array(object_pixel_counts))
        assert 1 <= len(objects) <= maximum_objects
        number_objects += len(objects)

    # Most segmentation images keep their 5 or 6 shapes
    assert number_objects == 20 if maximum_objects == 1 else number_objects >= 4 * 20


def test_instance_map_gives_the_visible_part_of_every_object() -> None:
    """
    Verifies that the instance map painted with the class map gives every pixel of a shape the index of the object
    drawn last there, so that its category is the one of the class map at every pixel.
    """
    random_generator = np.random.default_rng(seed=0)

    for _ in range(10):
        _, objects, class_map, instance_map = generate_image(image_size=IMAGE_SIZE, maximum_objects=6,
                                                             random_generator=random_generator)
        assert np.array_equal(instance_map > 0, class_map > 0) and int(instance_map.max()) <= len(objects)
        object_categories = np.array(object=[0] + [category_identifier for category_identifier, _ in objects])
        assert np.array_equal(object_categories[instance_map], class_map)


def test_covered_objects_are_left_out_of_the_labels() -> None:
    """
    Verifies that an object hidden by a later one down to fewer than MINIMUM_VISIBLE_OBJECT_PIXELS visible pixels is
    left out of the detection and instance segmentation labels, while a partly covered object stays: with the box of
    the whole shape for the detection, and with the box and the encoding of its visible part for the instance
    segmentation.
    """
    # First object fully covered by the third one, second object half covered by it, the third drawn last
    instance_map = np.zeros(shape=(64, 64), dtype=np.uint16)
    instance_map[0:10, 0:10] = 1
    instance_map[20:40, 0:20] = 2
    instance_map[0:10, 0:10] = 3
    instance_map[20:40, 10:30] = 3
    objects = [(1, [0, 0, 10, 10]), (2, [0, 20, 20, 40]), (3, [0, 0, 30, 40])]
    object_pixel_counts = [100, 400, 500]

    # The detection keeps the whole boxes of the visible objects
    detection_objects, detection_segmentations = keep_visible_objects(objects=objects, instance_map=instance_map,
                                                                      keep_visible_parts=False,
                                                                      object_pixel_counts=object_pixel_counts)
    assert detection_objects == [(2, [0, 20, 20, 40]), (3, [0, 0, 30, 40])] and detection_segmentations == []

    # The instance segmentation keeps the box and the compressed encoding of the visible part of every visible object,
    # nothing else
    instance_objects, instance_parts = keep_visible_objects(objects=objects, instance_map=instance_map,
                                                            keep_visible_parts=True,
                                                            object_pixel_counts=object_pixel_counts)
    assert instance_objects == [(2, [0, 20, 10, 40]), (3, [0, 0, 30, 40])]
    assert isinstance(instance_parts[0]["segmentation"]["counts"], str)
    assert np.array_equal(decode_run_length_encoding(segmentation=instance_parts[0]["segmentation"]), instance_map == 2)
    assert [sorted(instance_part) for instance_part in instance_parts] == [["segmentation"], ["segmentation"]]


def test_small_shapes_stay_in_the_labels_half_visible() -> None:
    """
    Verifies that the objects of a detection, instance segmentation or keypoint image stay in the labels with
    OBJECT_MINIMUM_VISIBLE_FRACTION (half) of the pixels of their whole shape visible when that is below
    MINIMUM_VISIBLE_OBJECT_PIXELS (a small shape), and with MINIMUM_VISIBLE_OBJECT_PIXELS visible pixels otherwise (a
    large shape partly covered by others), and are left out below that.
    """
    # Small shapes of 40 pixels with 20 and 19 visible pixels, large shapes of 1000 pixels with 64 and 63 visible
    instance_map = np.zeros(shape=(64, 64), dtype=np.uint16)
    instance_map.reshape(-1)[0:20] = 1
    instance_map.reshape(-1)[100:119] = 2
    instance_map.reshape(-1)[200:264] = 3
    instance_map.reshape(-1)[300:363] = 4
    objects = [(1, [0, 0, 20, 1]), (2, [36, 1, 55, 2]), (3, [8, 3, 9, 4]), (4, [44, 4, 45, 5])]

    kept_objects, _ = keep_visible_objects(objects=objects, instance_map=instance_map, keep_visible_parts=False,
                                           object_pixel_counts=[40, 40, 1000, 1000])
    assert [category_identifier for category_identifier, _ in kept_objects] == [1, 3]
    assert OBJECT_MINIMUM_VISIBLE_FRACTION == 0.5 and MINIMUM_VISIBLE_OBJECT_PIXELS == 64


def test_keypoint_labels_follow_the_visible_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the keypoints of the keypoint detection images (generate_image with keypoints): a keypoint outside the
    image is not labelled (visibility 0, coordinates 0), a keypoint inside it keeps its position and is labelled visible
    (2) only when a pixel of its own object touches it, or hidden (1) when an object drawn after its own covers it, and
    keep_visible_objects keeps exactly the objects showing MINIMUM_VISIBLE_OBJECT_PIXELS visible pixels, or half of a
    small shape, with their keypoints.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture recording the keypoints before their visibility.
    """
    visibility_recorder = CallRecorder(function=synthetic_dataset_module.compute_keypoint_visibilities)
    monkeypatch.setattr(target=synthetic_dataset_module, name="compute_keypoint_visibilities",
                        value=visibility_recorder)
    random_generator = np.random.default_rng(seed=0)
    visibility_counts = {0: 0, 1: 0, 2: 0}

    for _ in range(30):
        object_pixel_counts: List[int] = []
        _, objects, _, instance_map = generate_image(image_size=IMAGE_SIZE, maximum_objects=5,
                                                     random_generator=random_generator, keypoints=True,
                                                     object_pixel_counts=object_pixel_counts)

        # Keypoints of every object, before and after their visibility
        image_calls = visibility_recorder.calls[-len(objects):]
        for object_index, ((_, _, labelled_keypoints), call) in enumerate(zip(objects, image_calls, strict=True)):
            assert call["object_index"] == object_index
            for (keypoint_x, keypoint_y), (labelled_x, labelled_y, visibility) in zip(call["keypoints"],
                                                                                       labelled_keypoints, strict=True):
                inside = 0 <= keypoint_x <= IMAGE_SIZE and 0 <= keypoint_y <= IMAGE_SIZE
                assert (visibility > 0) == inside
                visibility_counts[int(visibility)] += 1
                if not inside:
                    assert labelled_x == labelled_y == 0
                    continue
                assert (labelled_x, labelled_y) == (keypoint_x, keypoint_y)

                # Pixels touching the keypoint: a pixel of its own object for a visible keypoint, of a later object for
                # a hidden one
                column, row = int(np.floor(keypoint_x)), int(np.floor(keypoint_y))
                touching_pixels = instance_map[max(0, row - 1):row + 2, max(0, column - 1):column + 2]
                if visibility == 2:
                    assert np.any(touching_pixels == object_index + 1)
                else:
                    assert np.any(touching_pixels > object_index + 1)

        # The objects kept in the labels are those visible enough, with their keypoints
        visible_pixel_counts = np.bincount(instance_map.reshape(-1), minlength=len(objects) + 1)[1:]
        kept_objects, _ = keep_visible_objects(objects=objects, instance_map=instance_map, keep_visible_parts=False,
                                               object_pixel_counts=object_pixel_counts)
        expected_objects = [image_object for image_object, visible_pixel_count, pixel_count
                            in zip(objects, visible_pixel_counts, object_pixel_counts, strict=True)
                            if visible_pixel_count >= min(MINIMUM_VISIBLE_OBJECT_PIXELS,
                                                          OBJECT_MINIMUM_VISIBLE_FRACTION * pixel_count)]
        assert len(kept_objects) == len(expected_objects)
        assert all(kept_object[2] is expected_object[2]
                   for kept_object, expected_object in zip(kept_objects, expected_objects, strict=True))

    # Visible, hidden and unlabelled keypoints all occur
    assert all(count > 0 for count in visibility_counts.values())


def test_created_instance_segmentation_dataset_encodes_small_and_large_visible_parts(tmp_path: Path) -> None:
    """
    Verifies the instance segmentation mode of the generator: the COCO labels hold 5 to 12 objects per image on
    average, every one with the compressed encoding of its visible part, not empty, whose tight box is its box and
    no 'area' or 'iscrowd' key, the parts of an image never overlapping, down to small shapes and up to large ones,
    some cut by the image border, and the instance segmentation dataset of the framework loads the same masks.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_synthetic_dataset(output_folder=str(tmp_path), image_size=IMAGE_SIZE, split_sizes={"train": 12},
                             maximum_objects=12, minimum_frames=1, maximum_frames=1, number_workers=2,
                             task="instance_segmentation")

    # Objects per image, and the keys, compressed encoding and box of every visible part
    labels = json.loads((tmp_path / "train" / "labels.json").read_text())
    objects_per_image = np.bincount([annotation["image_id"] for annotation in labels["annotations"]], minlength=13)[1:]
    assert objects_per_image.max() <= 12 and objects_per_image.mean() >= MINIMUM_OBJECTS_PER_IMAGE
    image_masks = {image["id"]: [] for image in labels["images"]}
    for annotation in labels["annotations"]:
        assert isinstance(annotation["segmentation"]["counts"], str)
        mask = decode_run_length_encoding(segmentation=annotation["segmentation"])
        left, top, width, height = annotation["bbox"]
        assert sorted(annotation) == ["bbox", "category_id", "id", "image_id", "segmentation"]
        assert int(mask.sum()) > 0
        assert get_mask_bounding_box(mask=mask) == [left, top, left + width, top + height]
        image_masks[annotation["image_id"]].append(mask)

    # Small and large visible parts, some reaching the image border, never overlapping within an image
    long_sides = [max(annotation["bbox"][2:]) for annotation in labels["annotations"]]
    assert min(long_sides) < 32 and max(long_sides) > IMAGE_SIZE // 3
    assert any(left == 0 or top == 0 or left + width == IMAGE_SIZE or top + height == IMAGE_SIZE
               for left, top, width, height in (annotation["bbox"] for annotation in labels["annotations"]))
    for image in labels["images"]:
        assert int(np.stack(arrays=image_masks[image["id"]]).sum(axis=0).max()) == 1

    # The instance segmentation dataset loads the same masks at the image size, in the order of the annotations
    label_mapping = json.loads((tmp_path / "label_mapping.json").read_text())
    dataset = CocoDetectionDataset(data_directory=str(tmp_path / "train" / "data"),
                                   labels_file=str(tmp_path / "train" / "labels.json"), image_size=IMAGE_SIZE,
                                   keep_ratio=True, label_mode="multi_class",
                                   label_dictionary={int(category): class_index for category, class_index in
                                                     label_mapping.items()}, load_instance_masks=True, shuffle=False)
    for sample in (dataset[index] for index in range(len(dataset))):
        assert np.array_equal(sample["instance_masks"].numpy(),
                              np.stack(arrays=image_masks[sample["sample_identifiers"]]))


def test_overlap_cap_reaches_the_placement_of_every_object(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the largest IoU between two objects given to generate_image reaches the placement of every object,
    MAXIMUM_OBJECT_INTERSECTION_OVER_UNION (0.3) by default, with which every dataset was drawn.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture recording the placements.
    """
    random_generator = np.random.default_rng(seed=0)
    assert MAXIMUM_OBJECT_INTERSECTION_OVER_UNION == 0.3
    for keyword_arguments, expected_cap in [({}, 0.3), ({"maximum_object_intersection_over_union": 0.6}, 0.6)]:
        placement_recorder = CallRecorder(function=synthetic_dataset_module.place_shape)
        monkeypatch.setattr(target=synthetic_dataset_module, name="place_shape", value=placement_recorder)
        generate_image(image_size=IMAGE_SIZE, maximum_objects=12, random_generator=random_generator,
                       **keyword_arguments)
        assert len(placement_recorder.calls) >= MINIMUM_OBJECTS_PER_IMAGE
        assert all(call["maximum_intersection_over_union"] == expected_cap for call in placement_recorder.calls)


def count_overlapping_label_pairs(labels: Dict[str, Any], minimum_intersection_over_union: float = 0.0,
                                  same_class: bool = False) -> int:
    """
    Counts the pairs of objects of the same image whose boxes in the COCO labels of a detection split overlap by more
    than an IoU, of any classes or of one class, so that a test tells whether the overlap cap of the generator reached
    the placement of the objects, or how many pairs of objects of one class the greedy NMS of a detector would merge.

    Args:
        labels (Dict[str, Any]): The COCO labels, with the 'category_id' and the [x, y, width, height] 'bbox' of every
            annotation.
        minimum_intersection_over_union (float): The IoU a pair must exceed, 0 counting the boxes sharing a positive
            area. Defaults to 0.
        same_class (bool): True to count the pairs of objects of one class only. Defaults to False.

    Returns:
        int: The number of pairs of boxes of an image whose IoU exceeds the minimum.
    """
    # Category and corner box of every object, gathered by image
    image_objects: Dict[int, List[Tuple[int, List[float]]]] = {}
    for annotation in labels["annotations"]:
        left, top, width, height = annotation["bbox"]
        image_objects.setdefault(annotation["image_id"], []).append((annotation["category_id"],
                                                                     [left, top, left + width, top + height]))

    # Pairs of every image overlapping by more than the minimum, of one class when asked
    overlapping_pairs = 0
    for objects in image_objects.values():
        for (first_category, first_box), (second_category, second_box) in itertools.combinations(objects, 2):
            overlapping_pairs += ((not same_class or first_category == second_category) and get_intersection_over_union(
                first_box=first_box, second_box=second_box) > minimum_intersection_over_union)
    return overlapping_pairs


def test_created_detection_split_follows_its_overlap_cap(tmp_path: Path) -> None:
    """
    Verifies that create_synthetic_dataset passes its largest IoU between two objects down to the worker processes
    drawing the images: an object detection split drawn with a cap of 0 has no two overlapping boxes in its labels,
    while the same split drawn with the default cap of 0.3 has some.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    overlapping_pairs = {}
    for overlap_cap in [0.0, MAXIMUM_OBJECT_INTERSECTION_OVER_UNION]:
        output_folder = tmp_path / f"cap_{overlap_cap}"
        create_synthetic_dataset(output_folder=str(output_folder), image_size=IMAGE_SIZE, split_sizes={"test": 8},
                                 maximum_objects=12, minimum_frames=1, maximum_frames=1, number_workers=1,
                                 task="object_detection", maximum_object_intersection_over_union=overlap_cap)
        labels = json.loads((output_folder / "test" / "labels.json").read_text())
        assert len(labels["annotations"]) >= 8
        overlapping_pairs[overlap_cap] = count_overlapping_label_pairs(labels=labels)
    assert overlapping_pairs[0.0] == 0 and overlapping_pairs[MAXIMUM_OBJECT_INTERSECTION_OVER_UNION] > 0


@pytest.mark.parametrize(argnames=("task", "overlap_cap", "expected_message"), argvalues=[
    ("object_detection", 1.5, "must be within \\[0, 1\\], got 1.5"),
    ("instance_segmentation", float("nan"), "must be within \\[0, 1\\], got nan"),
    ("sequence_labelling", 0.6, "not those of the 'sequence_labelling' task, got 0.6"),
    ("self_supervised", 0.0, "not those of the 'self_supervised' task, got 0.0")])
def test_invalid_overlap_cap_is_refused(tmp_path: Path, task: str, overlap_cap: float, expected_message: str) -> None:
    """
    Verifies that the largest IoU between two objects must lie within [0, 1], and that it is refused for the tasks
    whose shapes are placed apart by other functions than generate_image, on which it would have no effect.

    Args:
        tmp_path (Path): The pytest temporary directory.
        task (str): The task of the refused dataset.
        overlap_cap (float): The refused largest IoU between two objects.
        expected_message (str): A pattern of the error message.
    """
    with pytest.raises(expected_exception=ValueError, match=expected_message):
        create_synthetic_dataset(output_folder=str(tmp_path), image_size=IMAGE_SIZE, split_sizes={"train": 2},
                                 maximum_objects=6, minimum_frames=2, maximum_frames=3, number_workers=1, task=task,
                                 maximum_object_intersection_over_union=overlap_cap)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(argnames="target_intersection_over_union", argvalues=[0.1, 0.55])
def test_neighbour_centre_distance_gives_the_target_overlap(target_intersection_over_union: float) -> None:
    """
    Verifies that find_neighbour_centre_distance gives, along every direction, the distance between the centre of a
    box and the centre of its neighbour at which their IoU is the target, down to a small target far from the centre,
    and None when even a box sharing the centre of its neighbour overlaps it by less than the target.

    Args:
        target_intersection_over_union (float): The IoU wanted between the two boxes.
    """
    neighbour_box, shape_width, shape_height = [100, 100, 200, 160], 90, 54
    for angle in np.linspace(start=0.0, stop=2 * np.pi, num=16, endpoint=False):
        # Box moved from the centre of its neighbour by the distance found, built here from its centre and its sides
        direction = np.array([np.cos(angle), np.sin(angle)])
        centre_distance = find_neighbour_centre_distance(
            neighbour_box=neighbour_box, shape_width=shape_width, shape_height=shape_height, direction=direction,
            target_intersection_over_union=target_intersection_over_union)
        centre_x, centre_y = np.array([150.0, 130.0]) + centre_distance * direction
        moved_box = [centre_x - shape_width / 2, centre_y - shape_height / 2, centre_x + shape_width / 2,
                     centre_y + shape_height / 2]
        assert get_intersection_over_union(first_box=moved_box, second_box=neighbour_box) == pytest.approx(
            expected=target_intersection_over_union, abs=1e-6)

    # A box of 30 x 18 pixels inside its 100 x 60 neighbour overlaps it by an IoU of 0.09 at most
    assert find_neighbour_centre_distance(neighbour_box=neighbour_box, shape_width=30, shape_height=18,
                                          direction=np.array([1.0, 0.0]),
                                          target_intersection_over_union=target_intersection_over_union) is None


def test_same_class_neighbours_overlap_the_object_placed_before_them(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the same-class placement of the occlusion test split of stage 5b on images drawn with the cap 0.6 and a
    probability of 1: about half of the objects are placed next to the object placed just before them, every one with
    the class of that object, a size from 0.8 to 1 times its size and an IoU with its box from 0.5 to 0.6, in pairs (an
    object placed that way never getting a neighbour of its own), no two objects of an image overlap by more than the
    cap, and the pixel counts that keep_visible_objects reads are those of the shapes placed.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture recording the placements.
    """
    overlap_cap, number_objects, number_neighbours = 0.6, 0, 0
    for seed in range(8):
        placement_recorder = CallRecorder(function=synthetic_dataset_module.place_shape)
        monkeypatch.setattr(target=synthetic_dataset_module, name="place_shape", value=placement_recorder)
        object_pixel_counts: List[int] = []
        _, objects, _, _ = generate_image(image_size=IMAGE_SIZE, maximum_objects=12,
                                          random_generator=np.random.default_rng(seed=seed),
                                          object_pixel_counts=object_pixel_counts,
                                          maximum_object_intersection_over_union=overlap_cap,
                                          same_class_neighbour_probability=1.0)

        # Placements that put an object in the image, in the order of the objects, given the boxes placed before and
        # counted with the pixels of their shape
        placed_calls = [call for call, box in zip(placement_recorder.calls, placement_recorder.results, strict=True)
                        if box is not None]
        assert len(placed_calls) == len(objects)
        assert all(call["placed_boxes"] == [box for _, box in objects[:object_index]]
                   for object_index, call in enumerate(placed_calls))
        assert object_pixel_counts == [int(np.count_nonzero(call["shape_descriptor"]["mask"])) for call in placed_calls]

        # Every attempt next to a neighbour aims at the object placed last, itself placed as any other object
        for call in placement_recorder.calls:
            if "neighbour_box" in call:
                assert len(call["placed_boxes"]) >= 1 and call["neighbour_box"] == call["placed_boxes"][-1]
                assert "neighbour_box" not in placed_calls[len(call["placed_boxes"]) - 1]

        # Every object placed next to a neighbour has its class, a close size and an IoU with it from 0.5 to the cap
        for object_index, call in enumerate(placed_calls):
            if "neighbour_box" not in call:
                continue
            number_neighbours += 1
            assert object_index >= 1 and objects[object_index][0] == objects[object_index - 1][0]
            neighbour_size = placed_calls[object_index - 1]["shape_descriptor"]["mask"].shape[0] - 1
            size = call["shape_descriptor"]["mask"].shape[0] - 1
            assert round(SAME_CLASS_NEIGHBOUR_SIZE_RATIO_RANGE[0] * neighbour_size) <= size <= neighbour_size
            assert SAME_CLASS_NEIGHBOUR_MINIMUM_INTERSECTION_OVER_UNION <= get_intersection_over_union(
                first_box=objects[object_index][1], second_box=objects[object_index - 1][1]) <= overlap_cap

        # No two objects of the image overlap by more than the cap
        assert all(get_intersection_over_union(first_box=first_box, second_box=second_box) <= overlap_cap
                   for (_, first_box), (_, second_box) in itertools.combinations(objects, 2))
        number_objects += len(objects)

    # Close to one object in two placed next to a neighbour, the most that pairs allow
    assert number_neighbours >= 0.4 * number_objects


def test_created_detection_split_holds_same_class_pairs_above_the_nms_threshold(tmp_path: Path) -> None:
    """
    Verifies that create_synthetic_dataset passes the same-class neighbour probability down to the worker processes
    drawing the images: an object detection split drawn with the cap 0.6 and a probability of 1 holds about four pairs
    of objects of one class per image whose IoU exceeds the NMS threshold 0.5, the pairs the greedy NMS of a detector
    merges (18 at least over 8 images in 100,000 draws), while the same split drawn without it holds almost none (0.01
    per image), and no two objects of either split overlap by more than the cap.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    same_class_pairs = {}
    for probability in [0.0, 1.0]:
        output_folder = tmp_path / f"probability_{probability}"
        create_synthetic_dataset(output_folder=str(output_folder), image_size=IMAGE_SIZE, split_sizes={"test": 8},
                                 maximum_objects=12, minimum_frames=1, maximum_frames=1, number_workers=1,
                                 task="object_detection", maximum_object_intersection_over_union=0.6,
                                 same_class_neighbour_probability=probability)
        labels = json.loads((output_folder / "test" / "labels.json").read_text())
        assert count_overlapping_label_pairs(labels=labels, minimum_intersection_over_union=0.6) == 0
        same_class_pairs[probability] = count_overlapping_label_pairs(
            labels=labels, minimum_intersection_over_union=SAME_CLASS_NEIGHBOUR_MINIMUM_INTERSECTION_OVER_UNION,
            same_class=True)
    assert same_class_pairs[1.0] >= 8 > same_class_pairs[0.0]


@pytest.mark.parametrize(argnames=("task", "overlap_cap", "probability", "expected_message"), argvalues=[
    ("object_detection", 0.6, 1.5, "same_class_neighbour_probability must be within \\[0, 1\\], got 1.5"),
    ("instance_segmentation", 0.6, float("nan"), "same_class_neighbour_probability must be within \\[0, 1\\], got nan"),
    ("keypoint_detection", 0.6, 0.3, "instance segmentation images, got task 'keypoint_detection'"),
    ("object_detection", MAXIMUM_OBJECT_INTERSECTION_OVER_UNION, 0.3, "must then be 0.5 at least, got 0.3")])
def test_invalid_same_class_neighbour_probability_is_refused(tmp_path: Path, task: str, overlap_cap: float,
                                                             probability: float, expected_message: str) -> None:
    """
    Verifies that the same-class neighbour probability must lie within [0, 1], that it is refused for the tasks whose
    labels are not the boxes or the visible parts of the objects, and that it needs a largest IoU between two objects of
    0.5 at least, the IoU at which it places the objects starting at the NMS threshold 0.5.

    Args:
        tmp_path (Path): The pytest temporary directory.
        task (str): The task of the refused dataset.
        overlap_cap (float): The largest IoU between two objects.
        probability (float): The refused same-class neighbour probability.
        expected_message (str): A pattern of the error message.
    """
    with pytest.raises(expected_exception=ValueError, match=expected_message):
        create_synthetic_dataset(output_folder=str(tmp_path), image_size=IMAGE_SIZE, split_sizes={"train": 2},
                                 maximum_objects=6, minimum_frames=1, maximum_frames=1, number_workers=1, task=task,
                                 maximum_object_intersection_over_union=overlap_cap,
                                 same_class_neighbour_probability=probability)
    assert not any(tmp_path.iterdir())


def test_created_sequence_classification_dataset_lists_its_sequences(tmp_path: Path) -> None:
    """
    Verifies the sequence classification mode of the generator: every split gets a manifest of sequences with their
    ordered frames, 3 to 6 frames whose number varies from sequence to sequence, each labelled with a direction, and the
    label mapping gives the four directions the classes 0 to 3 (the movement of the shape in the labelled direction
    being checked on the drawn sequences, whose background may be textured).

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_synthetic_dataset(output_folder=str(tmp_path), image_size=64,
                             split_sizes={"train": 12, "validation": 2, "test": 2}, maximum_objects=1,
                             minimum_frames=3, maximum_frames=6, number_workers=2, task="sequence_classification")

    labels = json.loads((tmp_path / "train" / "labels.json").read_text())
    assert [category["name"] for category in labels["categories"]] == list(MOVEMENT_CATEGORIES)
    assert json.loads((tmp_path / "label_mapping.json").read_text()) == {"1": 0, "2": 1, "3": 2, "4": 3}
    assert len(labels["sequences"]) == 12

    # Every sequence draws its own number of frames in the requested range, every frame saved, with one direction
    sequence_lengths = [len(sequence["frames"]) for sequence in labels["sequences"]]
    assert all(3 <= sequence_length <= 6 for sequence_length in sequence_lengths) and len(set(sequence_lengths)) > 1
    for sequence in labels["sequences"]:
        assert sequence["category_id"] in MOVEMENT_CATEGORIES.values()
        assert all((tmp_path / "train" / "data" / frame_file).is_file() for frame_file in sequence["frames"])


def test_moving_shape_crosses_the_frames_in_the_labelled_direction(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the sequences of generate_moving_shape_sequence: every frame shows the background of its sequence (flat
    or textured) and the one shape of the sequence in its colour, its canvas moved along a straight line in the
    direction of the label (constant across the movement, never going back along it, moving from the first frame to
    the last), the shape keeping SHAPE_MINIMUM_VISIBLE_FRACTION of its pixels inside every frame, some frames cutting it
    with the border.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture recording the backgrounds and shapes.
    """
    background_recorder = CallRecorder(function=synthetic_dataset_module.draw_background)
    descriptor_recorder = CallRecorder(function=synthetic_dataset_module.draw_shape_descriptor)
    monkeypatch.setattr(target=synthetic_dataset_module, name="draw_background", value=background_recorder)
    monkeypatch.setattr(target=synthetic_dataset_module, name="draw_shape_descriptor", value=descriptor_recorder)
    random_generator = np.random.default_rng(seed=0)
    category_names = {category_identifier: name for name, category_identifier in MOVEMENT_CATEGORIES.items()}
    cut_frames = 0

    for sequence_index in range(30):
        frames, category_identifier = generate_moving_shape_sequence(image_size=SAMPLE_IMAGE_SIZE, number_frames=6,
                                                                     random_generator=random_generator)
        assert len(background_recorder.calls) == len(descriptor_recorder.calls) == sequence_index + 1
        background = background_recorder.results[-1][0]
        shape_mask = descriptor_recorder.results[-1]["mask"] > 0

        # The shape of the sequence in its colour over the background, visible enough, at the position of its canvas
        canvas_positions = []
        for frame in frames:
            shape_pixels = np.any(frame != background, axis=-1)
            assert np.all(frame[shape_pixels] == descriptor_recorder.results[-1]["colour"])
            assert np.count_nonzero(shape_pixels) >= SHAPE_MINIMUM_VISIBLE_FRACTION * np.count_nonzero(shape_mask)
            cut_frames += np.count_nonzero(shape_pixels) < np.count_nonzero(shape_mask)
            canvas_positions.append(find_canvas_position(shape_pixels=shape_pixels, shape_mask=shape_mask))

        # Steps of the canvas along the labelled direction only, never going back
        direction_x, direction_y = MOVEMENT_DIRECTIONS[category_names[category_identifier]]
        steps = np.diff(np.array(canvas_positions), axis=0)
        steps_along = steps @ np.array([direction_x, direction_y])
        steps_across = steps @ np.array([direction_y, direction_x])
        assert np.all(steps_across == 0) and np.all(steps_along >= 0) and steps_along.sum() > 0

    assert cut_frames > 0


def test_created_sequence_labelling_dataset_labels_every_frame(tmp_path: Path) -> None:
    """
    Verifies the sequence labelling mode of the generator: every sequence lists the change category of every frame,
    2 to 6 frames whose number varies from sequence to sequence, the first frame is labelled 'no_change', and the label
    mapping gives the 71 changes the classes 0 to 70 (the shapes of every frame and their changes being checked on the
    drawn sequences, whose shapes may overlap).

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_synthetic_dataset(output_folder=str(tmp_path), image_size=128,
                             split_sizes={"train": 12, "validation": 2, "test": 2}, maximum_objects=6,
                             minimum_frames=2, maximum_frames=6, number_workers=2, task="sequence_labelling")

    labels = json.loads((tmp_path / "train" / "labels.json").read_text())
    assert [category["name"] for category in labels["categories"]] == list(CHANGE_CATEGORIES)
    assert json.loads((tmp_path / "label_mapping.json").read_text()) == {
        str(category_identifier): category_identifier - 1 for category_identifier in CHANGE_CATEGORIES.values()}
    assert len(CHANGE_CATEGORIES) == 2 * len(SHAPE_CATEGORIES) + 1
    sequence_lengths = [len(sequence["frames"]) for sequence in labels["sequences"]]
    assert all(2 <= sequence_length <= 6 for sequence_length in sequence_lengths) and len(set(sequence_lengths)) > 1

    for sequence in labels["sequences"]:
        frame_category_ids = sequence["frame_category_ids"]
        assert len(frame_category_ids) == len(sequence["frames"])
        assert frame_category_ids[0] == CHANGE_CATEGORIES["no_change"]
        assert all((tmp_path / "train" / "data" / frame_file).is_file() for frame_file in sequence["frames"])


def test_shape_change_frames_show_their_labelled_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the sequences of generate_shape_change_sequence: every frame shows its 5 to maximum_objects shapes over the
    background of its sequence, each keeping SHAPE_MINIMUM_VISIBLE_FRACTION of its pixels visible in its own colour
    (check_frame_shows_its_shapes); a shape keeps its class, size, colour and pixels from frame to frame, and every
    frame after the first adds the shape its label names ('appearing_<class>'), removes one shape of the class its label
    names ('disappearing_<class>') or keeps the same shapes ('no_change', always the label of the first frame).

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture recording the backgrounds and the frames.
    """
    background_recorder = CallRecorder(function=synthetic_dataset_module.draw_background)
    frame_recorder = CallRecorder(function=synthetic_dataset_module.draw_shapes_frame, copy_lists=True)
    monkeypatch.setattr(target=synthetic_dataset_module, name="draw_background", value=background_recorder)
    monkeypatch.setattr(target=synthetic_dataset_module, name="draw_shapes_frame", value=frame_recorder)
    random_generator = np.random.default_rng(seed=0)
    category_names = {category_identifier: name for name, category_identifier in CHANGE_CATEGORIES.items()}
    seen_changes = set()

    for sequence_index in range(20):
        frames, frame_category_ids = generate_shape_change_sequence(image_size=SAMPLE_IMAGE_SIZE, number_frames=6,
                                                                    maximum_objects=6,
                                                                    random_generator=random_generator)
        assert len(background_recorder.calls) == sequence_index + 1

        # Frames of the drawing kept, the last ones drawn, each showing its shapes
        previous_shapes = None
        for frame, category_identifier, call, (drawn_frame, instance_map) in zip(
                frames, frame_category_ids, frame_recorder.calls[-len(frames):],
                frame_recorder.results[-len(frames):], strict=True):
            shapes = call["shape_descriptors"]
            assert drawn_frame is frame and LABELLING_MINIMUM_OBJECTS <= len(shapes) <= 6
            check_frame_shows_its_shapes(frame=frame, instance_map=instance_map,
                                         background=background_recorder.results[-1][0], shape_descriptors=shapes)

            # The change of the label between the shapes of the previous frame and those of this frame
            change_name = category_names[category_identifier]
            if previous_shapes is None or change_name == "no_change":
                assert change_name == "no_change" and (previous_shapes is None or
                                                       are_same_shapes(first_shapes=shapes,
                                                                       second_shapes=previous_shapes))
            elif change_name.startswith("appearing_"):
                assert are_same_shapes(first_shapes=shapes[:-1], second_shapes=previous_shapes)
                assert shapes[-1]["shape_name"] == change_name[len("appearing_"):]
            else:
                removed_shapes = [shape for shape in previous_shapes if not any(shape is kept for kept in shapes)]
                assert len(removed_shapes) == 1 and are_same_shapes(
                    first_shapes=[shape for shape in previous_shapes if shape is not removed_shapes[0]],
                    second_shapes=shapes)
                assert removed_shapes[0]["shape_name"] == change_name[len("disappearing_"):]
            seen_changes.add(change_name.split("_")[0])
            previous_shapes = shapes

    assert seen_changes == {"no", "appearing", "disappearing"}


def test_created_self_supervised_dataset_counts_its_shapes(tmp_path: Path) -> None:
    """
    Verifies the self-supervised mode of the generator: every image lists the categories of its 5 to maximum_objects
    shapes and is labelled with their number, the label mapping gives the 6 shape counts the classes 0 to 5, and the
    classification dataset read by the self-supervised trainer loads it (the listed shapes being the visible ones is
    checked on the drawn images, whose shapes may overlap).

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_synthetic_dataset(output_folder=str(tmp_path), image_size=IMAGE_SIZE,
                             split_sizes={"train": 12, "validation": 2, "test": 2}, maximum_objects=7,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="self_supervised")

    labels = json.loads((tmp_path / "train" / "labels.json").read_text())
    assert [category["name"] for category in labels["categories"]] == list(SHAPE_COUNT_CATEGORIES)
    assert [category["name"] for category in labels["shape_categories"]] == list(SHAPE_CATEGORIES)
    label_mapping = json.loads((tmp_path / "label_mapping.json").read_text())
    assert label_mapping == {str(class_index + 1): class_index for class_index in range(6)}

    # The label of every image is the number of its listed shapes
    for sample in labels["samples"]:
        object_count = len(sample["shape_category_ids"])
        assert LABELLING_MINIMUM_OBJECTS <= object_count <= 7
        assert sample["category_id"] == SHAPE_COUNT_CATEGORIES[f"{object_count}_shapes"]
        assert all(category_identifier in SHAPE_CATEGORIES.values()
                   for category_identifier in sample["shape_category_ids"])

    dataset = ClassificationDataset(data_directory=str(tmp_path / "train" / "data"),
                                    labels_file=str(tmp_path / "train" / "labels.json"), image_size=IMAGE_SIZE,
                                    keep_ratio=True, label_dictionary={int(category): class_index for category,
                                                                       class_index in label_mapping.items()},
                                    output_mode="multi_class")
    assert dataset.class_names == dict(enumerate(SHAPE_COUNT_CATEGORIES))
    assert sum(dataset.get_class_counts()) == 12


def test_counted_shapes_images_show_every_listed_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the images of generate_counted_shapes_image, labelled with their number of shapes (self-supervised) or the
    distinct classes of their shapes (multi-label classification): the image shows exactly the listed shapes, 5 to
    maximum_objects of them in the order of the list, each keeping SHAPE_MINIMUM_VISIBLE_FRACTION of its pixels visible
    in its own colour over the background (check_frame_shows_its_shapes).

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture recording the backgrounds and the drawn images.
    """
    background_recorder = CallRecorder(function=synthetic_dataset_module.draw_background)
    frame_recorder = CallRecorder(function=synthetic_dataset_module.draw_shapes_frame)
    monkeypatch.setattr(target=synthetic_dataset_module, name="draw_background", value=background_recorder)
    monkeypatch.setattr(target=synthetic_dataset_module, name="draw_shapes_frame", value=frame_recorder)
    random_generator = np.random.default_rng(seed=0)

    for _ in range(20):
        image, shape_names = generate_counted_shapes_image(image_size=SAMPLE_IMAGE_SIZE, maximum_objects=7,
                                                           random_generator=random_generator)
        drawn_image, instance_map = frame_recorder.results[-1]
        shape_descriptors = frame_recorder.calls[-1]["shape_descriptors"]
        assert drawn_image is image and LABELLING_MINIMUM_OBJECTS <= len(shape_names) <= 7
        assert [shape_descriptor["shape_name"] for shape_descriptor in shape_descriptors] == shape_names
        check_frame_shows_its_shapes(frame=image, instance_map=instance_map,
                                     background=background_recorder.results[-1][0], shape_descriptors=shape_descriptors)


def test_created_multi_label_classification_dataset_lists_the_classes_of_its_shapes(tmp_path: Path) -> None:
    """
    Verifies the multi-label classification mode of the generator, which pretrains the backbone of the sequence
    labelling model: every image, drawn like a sequence labelling frame, lists the sorted distinct categories of its 5
    to maximum_objects shapes in 'category_ids' (several shapes may share a class), the label mapping gives the 35
    shapes the classes 0 to 34, and the classification dataset loads it in the multi-label output mode with one positive
    per listed category.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_synthetic_dataset(output_folder=str(tmp_path), image_size=IMAGE_SIZE,
                             split_sizes={"train": 12, "validation": 2, "test": 2}, maximum_objects=7,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="multi_label_classification")

    labels = json.loads((tmp_path / "train" / "labels.json").read_text())
    assert [category["name"] for category in labels["categories"]] == list(SHAPE_CATEGORIES)
    label_mapping = json.loads((tmp_path / "label_mapping.json").read_text())
    assert label_mapping == {str(category_identifier): category_identifier - 1
                             for category_identifier in SHAPE_CATEGORIES.values()}

    # Distinct categories of every image, sorted and known, at most as many as the shapes of an image
    for sample in labels["samples"]:
        assert "category_id" not in sample
        assert sample["category_ids"] == sorted(set(sample["category_ids"]))
        assert 1 <= len(sample["category_ids"]) <= 7
        assert all(category_identifier in SHAPE_CATEGORIES.values() for category_identifier in sample["category_ids"])

    # Multi-hot label of every image with one positive per listed category, counted once per image in every class
    dataset = ClassificationDataset(data_directory=str(tmp_path / "train" / "data"),
                                    labels_file=str(tmp_path / "train" / "labels.json"), image_size=IMAGE_SIZE,
                                    keep_ratio=True, label_dictionary={int(category): class_index for category,
                                                                       class_index in label_mapping.items()},
                                    output_mode="multi_label", shuffle=False)
    assert dataset.class_names == dict(enumerate(SHAPE_CATEGORIES))
    for sample_index, sample in enumerate(labels["samples"]):
        label = dataset[sample_index]["labels"]
        assert label.nonzero().flatten().tolist() == [label_mapping[str(category_identifier)]
                                                       for category_identifier in sample["category_ids"]]
    assert sum(dataset.get_class_counts()) == sum(len(sample["category_ids"]) for sample in labels["samples"])


def test_created_sequence_comparison_dataset_lists_its_pairs(tmp_path: Path) -> None:
    """
    Verifies the sequence comparison mode of the generator: every split gets a manifest of pairs with the ordered frames
    of their before and after members (2 to 4 frames each), each labelled with one of the two classes, and the label
    mapping gives them the classes 0 and 1 (the objects of every frame and the lost object being checked on the drawn
    pairs, whose objects may overlap).

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_synthetic_dataset(output_folder=str(tmp_path), image_size=64,
                             split_sizes={"train": 12, "validation": 2, "test": 2}, maximum_objects=3,
                             minimum_frames=2, maximum_frames=4, number_workers=2, task="sequence_comparison")

    labels = json.loads((tmp_path / "train" / "labels.json").read_text())
    assert [category["name"] for category in labels["categories"]] == list(PAIR_CATEGORIES)
    assert json.loads((tmp_path / "label_mapping.json").read_text()) == {"1": 0, "2": 1}
    assert len(labels["pairs"]) == 12

    # Both members of every pair with their frames saved
    for pair in labels["pairs"]:
        assert pair["category_id"] in PAIR_CATEGORIES.values()
        for member_name in ["before", "after"]:
            assert 2 <= len(pair[member_name]) <= 4
            assert all((tmp_path / "train" / "data" / frame_file).is_file() for frame_file in pair[member_name])


def test_object_loss_pairs_show_the_same_objects_but_the_lost_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the pairs of generate_object_loss_pair: every frame of both members shows its objects over the background
    of the pair, each keeping SHAPE_MINIMUM_VISIBLE_FRACTION of its pixels visible in its own colour
    (check_frame_shows_its_shapes); every before frame shows the same 2 to maximum_objects objects, and every after
    frame shows the same objects but one for the pairs labelled 'loss_of_object', all of them otherwise.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture recording the backgrounds and the frames.
    """
    background_recorder = CallRecorder(function=synthetic_dataset_module.draw_background)
    frame_recorder = CallRecorder(function=synthetic_dataset_module.draw_shapes_frame, copy_lists=True)
    monkeypatch.setattr(target=synthetic_dataset_module, name="draw_background", value=background_recorder)
    monkeypatch.setattr(target=synthetic_dataset_module, name="draw_shapes_frame", value=frame_recorder)
    random_generator = np.random.default_rng(seed=0)
    category_names = {category_identifier: name for name, category_identifier in PAIR_CATEGORIES.items()}
    seen_categories = set()

    for pair_index in range(20):
        member_frames, category_identifier = generate_object_loss_pair(
            image_size=SAMPLE_IMAGE_SIZE, minimum_frames=2, maximum_frames=4, maximum_objects=4,
            random_generator=random_generator)
        assert len(background_recorder.calls) == pair_index + 1

        # Frames of the drawing kept, the before frames then the after frames, each showing its objects
        frames = member_frames["before"] + member_frames["after"]
        frame_calls = frame_recorder.calls[-len(frames):]
        for frame, call, (drawn_frame, instance_map) in zip(frames, frame_calls, frame_recorder.results[-len(frames):],
                                                            strict=True):
            assert drawn_frame is frame
            check_frame_shows_its_shapes(frame=frame, instance_map=instance_map,
                                         background=background_recorder.results[-1][0],
                                         shape_descriptors=call["shape_descriptors"])

        # The same objects in every frame of a member, the after member losing one of them for a loss of object
        before_shapes = frame_calls[0]["shape_descriptors"]
        after_shapes = frame_calls[len(member_frames["before"])]["shape_descriptors"]
        assert all(are_same_shapes(first_shapes=call["shape_descriptors"], second_shapes=before_shapes)
                   for call in frame_calls[:len(member_frames["before"])])
        assert all(are_same_shapes(first_shapes=call["shape_descriptors"], second_shapes=after_shapes)
                   for call in frame_calls[len(member_frames["before"]):])
        assert PAIR_MINIMUM_OBJECTS <= len(before_shapes) <= 4
        lost_shapes = [shape for shape in before_shapes if not any(shape is kept for kept in after_shapes)]
        expected_lost_shapes = 1 if category_names[category_identifier] == "loss_of_object" else 0
        assert len(lost_shapes) == expected_lost_shapes
        assert are_same_shapes(first_shapes=[shape for shape in before_shapes if not any(
            shape is lost for lost in lost_shapes)], second_shapes=after_shapes)
        seen_categories.add(category_names[category_identifier])

    assert seen_categories == set(PAIR_CATEGORIES)


@pytest.mark.parametrize(argnames=("task", "minimum_frames", "maximum_frames"), argvalues=[
    ("sequence_classification", 1, 4)])
def test_sequence_frame_range_is_checked(tmp_path: Path, task: str, minimum_frames: int, maximum_frames: int) -> None:
    """
    Verifies that a sequence dataset is refused when its sequences could have a single frame, which shows no movement,
    no change of class and no comparison.

    Args:
        tmp_path (Path): The pytest temporary directory.
        task (str): The sequence task of the dataset.
        minimum_frames (int): The minimum number of frames of a sequence.
        maximum_frames (int): The maximum number of frames of a sequence.
    """
    with pytest.raises(expected_exception=ValueError, match="minimum_frames"):
        create_synthetic_dataset(output_folder=str(tmp_path), image_size=64, split_sizes={"train": 2},
                                 maximum_objects=1, minimum_frames=minimum_frames, maximum_frames=maximum_frames,
                                 number_workers=1, task=task)


def test_created_frame_selection_dataset_holds_every_wanted_shape_once(tmp_path: Path) -> None:
    """
    Verifies the frame selection mode of the generator: the wanted shapes and the unselected category with their
    label mapping (the unselected frames giving the class 0), sequences of 4 to 8 frames whose number varies, every
    wanted shape selected in exactly one frame of every sequence, and the splits loading with the frame selection
    dataset, which checks every sequence (the selected frames showing the largest copies being checked on the drawn
    sequences).

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_synthetic_dataset(output_folder=str(tmp_path), image_size=64,
                             split_sizes={"train": 10, "validation": 2, "test": 2}, maximum_objects=1,
                             minimum_frames=4, maximum_frames=8, number_workers=2, task="frame_selection")

    labels = json.loads((tmp_path / "train" / "labels.json").read_text())
    assert [category["name"] for category in labels["categories"]] == list(SELECTION_CATEGORIES)
    label_mapping = json.loads((tmp_path / "label_mapping.json").read_text())
    assert label_mapping == {"1": 0, "2": 1, "3": 2, "4": 3}
    sequence_lengths = [len(sequence["frames"]) for sequence in labels["sequences"]]
    assert all(4 <= sequence_length <= 8 for sequence_length in sequence_lengths) and len(set(sequence_lengths)) > 1

    # One frame of every wanted shape, every other frame unselected
    for sequence in labels["sequences"]:
        assert sorted(category_identifier for category_identifier in sequence["frame_category_ids"]
                      if category_identifier != SELECTION_CATEGORIES["unselected"]) == [2, 3, 4]

    # Every sequence of the split passes the check of the frame selection dataset
    dataset = SequenceDataset(data_directory=str(tmp_path / "train" / "data"),
                              labels_file=str(tmp_path / "train" / "labels.json"), image_size=64, keep_ratio=True,
                              label_dictionary={int(key): value for key, value in label_mapping.items()},
                              sequence_settings={"maximum_sequence_length": 8}, label_type="selection",
                              split_name="train",
                              selection_settings={"unselected_category": "unselected", "frames_per_class": 1})
    assert len(dataset) == 10 and dataset.unselected_class_index == 0


@pytest.mark.parametrize(argnames="image_size", argvalues=[64, 256])
def test_selected_frames_show_the_largest_copy_of_their_shape(monkeypatch: pytest.MonkeyPatch, image_size: int) -> None:
    """
    Verifies the sequences of generate_selection_sequence: every frame shows one shape in the colour of the sequence
    over its background, keeping SHAPE_MINIMUM_VISIBLE_FRACTION of its pixels inside the frame; the frame selected for a
    wanted shape shows its largest copy, from SELECTION_LARGEST_MINIMUM_SIZE pixels, every other copy measuring at most
    SELECTION_SMALLER_SIZE_RATIO of its size (OBJECT_MINIMUM_SIZE at least) and SELECTION_SMALLER_AREA_RATIO of its
    area, so that the selected frame also shows the most pixels of that shape.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture recording the backgrounds and the frames.
        image_size (int): The side of the frames.
    """
    background_recorder = CallRecorder(function=synthetic_dataset_module.draw_background)
    frame_recorder = CallRecorder(function=synthetic_dataset_module.draw_selection_frame)
    monkeypatch.setattr(target=synthetic_dataset_module, name="draw_background", value=background_recorder)
    monkeypatch.setattr(target=synthetic_dataset_module, name="draw_selection_frame", value=frame_recorder)
    random_generator = np.random.default_rng(seed=0)
    unselected_category = SELECTION_CATEGORIES["unselected"]

    for _ in range(20):
        frames, frame_category_ids = generate_selection_sequence(image_size=image_size, number_frames=8,
                                                                 random_generator=random_generator)
        background = background_recorder.results[-1][0]

        # Shape of every frame, the frames coming in a random order, shown in the colour of the sequence
        drawn_shapes = list(zip(frame_recorder.results[-len(frames):],
                                [call["shape_descriptor"] for call in frame_recorder.calls[-len(frames):]]))
        frame_shapes = [next(shape for drawn_frame, shape in drawn_shapes if drawn_frame is frame) for frame in frames]
        assert len({tuple(shape["colour"].tolist()) for shape in frame_shapes}) == 1
        visible_pixel_counts = []
        for frame, shape in zip(frames, frame_shapes, strict=True):
            shape_pixels = np.any(frame != background, axis=-1)
            assert np.all(frame[shape_pixels] == shape["colour"])
            find_canvas_position(shape_pixels=shape_pixels, shape_mask=shape["mask"] > 0)
            visible_pixel_counts.append(int(np.count_nonzero(shape_pixels)))
            assert visible_pixel_counts[-1] >= SHAPE_MINIMUM_VISIBLE_FRACTION * np.count_nonzero(shape["mask"])

        # The selected frame of every wanted shape shows its largest copy, clearly larger than the others
        for shape_name in SELECTION_WANTED_SHAPES:
            selected_indices = [frame_index for frame_index, category_identifier in enumerate(frame_category_ids)
                                if category_identifier == SELECTION_CATEGORIES[shape_name]]
            assert len(selected_indices) == 1
            selected_shape = frame_shapes[selected_indices[0]]
            largest_size = selected_shape["mask"].shape[0] - 1
            largest_area = np.count_nonzero(selected_shape["mask"])
            assert selected_shape["shape_name"] == shape_name
            assert largest_size >= min(SELECTION_LARGEST_MINIMUM_SIZE, int(OBJECT_MAXIMUM_SIZE_FRACTION * image_size))
            for frame_index, shape in enumerate(frame_shapes):
                if frame_index == selected_indices[0] or shape["shape_name"] != shape_name:
                    continue
                assert frame_category_ids[frame_index] == unselected_category
                assert shape["mask"].shape[0] - 1 <= max(OBJECT_MINIMUM_SIZE,
                                                         int(SELECTION_SMALLER_SIZE_RATIO * largest_size))
                assert np.count_nonzero(shape["mask"]) <= SELECTION_SMALLER_AREA_RATIO * largest_area
                assert visible_pixel_counts[frame_index] < visible_pixel_counts[selected_indices[0]]

        # The other shapes are never selected
        assert all(frame_category_ids[frame_index] == unselected_category for frame_index, shape
                   in enumerate(frame_shapes) if shape["shape_name"] not in SELECTION_WANTED_SHAPES)
