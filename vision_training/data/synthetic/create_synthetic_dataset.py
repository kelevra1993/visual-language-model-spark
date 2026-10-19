"""
Creates the synthetic shapes datasets of every task (object detection, instance segmentation, keypoint detection,
classification, semantic segmentation, sequence classification, sequence labelling, sequence comparison, frame
selection, self-supervised) with their labels and 'label_mapping.json': trivial datasets used by the synthetic
configurations and the trainer tests to check that a model learns, and the counted shapes images of the self-supervised
configurations, drawn like the sequence labelling frames so that the self-supervised backbone can initialise the
sequence labelling model. Run as a script with the task set in its main block.
"""

import json
import os
import cv2
import numpy as np

from functools import partial
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Optional, Tuple

from vision_training.data.detection.coco_masks import encode_run_length_encoding
from vision_training.data.sequence.sequence_pair_dataset import PAIR_MEMBERS
from vision_training.utilities.os_utilities import format_path_link, print_color

# Filled shapes that have a hollow class of their own, the same outline with a hole of the same outline cut out of it
# (the annulus being the hollow circle). The plus sign, the star and the L-shape have arms too thin for a hole
HOLLOW_SHAPE_NAMES = ["square", "triangle", "rectangle", "pentagon", "hexagon", "octagon", "rhombus", "parallelogram",
                      "trapezoid", "kite", "heptagon", "nonagon", "ellipse", "semicircle", "capsule"]

# Shape classes of the synthetic dataset and their COCO category identifiers, the first nine keeping the identifiers
# of the datasets created before the other eleven were added, and the first twenty those of the datasets created
# before the fifteen hollow shapes were added
SHAPE_CATEGORIES = {"square": 1, "circle": 2, "triangle": 3, "rectangle": 4, "pentagon": 5, "hexagon": 6,
                    "plus_sign": 7, "star": 8, "octagon": 9, "rhombus": 10, "parallelogram": 11, "trapezoid": 12,
                    "kite": 13, "heptagon": 14, "nonagon": 15, "ellipse": 16, "semicircle": 17, "annulus": 18,
                    "capsule": 19, "l_shape": 20,
                    **{f"hollow_{shape_name}": 21 + index for index, shape_name in enumerate(HOLLOW_SHAPE_NAMES)}}

# Smallest size of the shapes whose class is the label (objects of the detection, segmentation and classification
# images, shapes of the sequence labelling frames): the sides of a nonagon, the roundest polygon, then bend 2 pixels
# away from its circle (0.03 x the size), so that the regular polygons stay distinct from each other and from the circle
MINIMUM_SHAPE_SIZE = 64

# Rectangles are elongated to exercise the 0.5 and 2.0 anchor aspect ratios (ratio of the long to the short side)
RECTANGLE_ASPECT_RATIO_RANGE = (1.5, 3.0)

# Minimum area of a triangle, relative to the area of the square region its vertices are drawn in
MINIMUM_TRIANGLE_AREA_FRACTION = 0.15

# Ratio of the inner to the outer radius of the five-pointed stars, and width of the arms of the plus signs relative
# to their length
STAR_INNER_RADIUS_RATIO = 0.45
PLUS_SIGN_ARM_WIDTH_RATIO = 0.35

# Ratio of the short to the long diagonal of the rhombuses, giving acute angles of 44 to 70 degrees, far from the
# right angles of the squares
RHOMBUS_DIAGONAL_RATIO_RANGE = (0.4, 0.7)

# Ratio of the long to the short side of the parallelograms, far from the equal sides of the rhombuses, and their
# acute angle in degrees, far from the right angles of the rectangles
PARALLELOGRAM_SIDE_RATIO_RANGE = (1.5, 2.5)
PARALLELOGRAM_ANGLE_RANGE_DEGREES = (45.0, 65.0)

# Length of the short parallel side of the trapezoids relative to the long one, far from both the parallelograms
# (equal sides) and the triangles (no short side), their height relative to the long side, and the position of the
# short side over the rest of the long side (0.5 centring it, an isosceles trapezoid), which keeps both angles of the
# long side acute so that the parallel sides stay visible
TRAPEZOID_TOP_RATIO_RANGE = (0.35, 0.65)
TRAPEZOID_HEIGHT_RATIO_RANGE = (0.35, 0.65)
TRAPEZOID_TOP_POSITION_RANGE = (0.25, 0.75)

# Kites: part of the long diagonal above its crossing with the short diagonal relative to the part below it (1 being
# a rhombus), and ratio of the short to the long diagonal, which keep the upper corner at most 107 degrees so that it
# does not flatten into a triangle
KITE_UPPER_RATIO_RANGE = (0.35, 0.6)
KITE_WIDTH_RATIO_RANGE = (0.5, 0.7)

# Ratio of the long to the short axis of the ellipses and of the capsules, far from the circles, and number of
# vertices drawing each round end of a capsule
ELLIPSE_ASPECT_RATIO_RANGE = (1.5, 3.0)
CAPSULE_ASPECT_RATIO_RANGE = (1.5, 3.0)
CAPSULE_END_VERTICES = 16

# Radius of the hole of the annuluses relative to their outer radius
ANNULUS_INNER_RADIUS_RATIO_RANGE = (0.4, 0.6)

# Size of the hole of the hollow shapes relative to their outline, the hole being the outline scaled down around its
# centroid, the thinnest wall in pixels kept at the acute corners of the flattest shapes, and the number of shapes
# drawn before failing when none leaves a hole inside that wall (flat triangles at small sizes)
HOLLOW_INNER_SIZE_RATIO_RANGE = (0.4, 0.6)
HOLLOW_MINIMUM_WALL_PIXELS = 2
HOLLOW_MAXIMUM_ATTEMPTS = 20

# Smallest size of a hollow shape drawn in an image, the drawn size of a hollow class being raised to it. A drawing of
# the hollow parallelogram, the worst of the fifteen hollow classes, leaves no hole inside the walls 91% of the time at
# 8 pixels and 1.8% of the time at 16 pixels, so that its HOLLOW_MAXIMUM_ATTEMPTS drawings almost never all fail; none
# of 400 drawings failed at 18 pixels
HOLLOW_MINIMUM_SIZE = 16
HOLLOW_CLASS_NAMES = [f"hollow_{shape_name}" for shape_name in HOLLOW_SHAPE_NAMES]

# Keypoints of the hollow shapes: the hollow polygons and the hollow capsule also have a keypoint at every corner of
# their hole, and the hollow ellipse has none, a distractor drawn among the objects of the keypoint images that is left
# out of the keypoint label mapping, like the circle, the annulus and the ellipse, so that the keypoint detector must tell them
# apart from the shapes it learns. The other hollow shapes keep the keypoints of their filled shape
HOLLOW_INNER_KEYPOINT_SHAPE_NAMES = ["square", "triangle", "rectangle", "pentagon", "hexagon", "octagon", "rhombus",
                                     "parallelogram", "trapezoid", "kite", "heptagon", "nonagon", "capsule"]
HOLLOW_DISTRACTOR_SHAPE_NAMES = ["ellipse"]

# L-shapes: length of the foot relative to the long arm, and width of both arms relative to the long arm, the foot
# sticking out of the long arm by at least a quarter of its length
L_SHAPE_FOOT_RATIO_RANGE = (0.6, 1.0)
L_SHAPE_ARM_WIDTH_RATIO_RANGE = (0.25, 0.35)

# Keypoint detection: categories of the shapes with keypoints, the shape classes followed by a stick figure and a
# house. Their keypoints are named after their place on the upright shape, the keypoint images drawing every shape
# without rotation nor mirror
KEYPOINT_CATEGORIES = {**SHAPE_CATEGORIES, "stick_figure": len(SHAPE_CATEGORIES) + 1,
                       "house": len(SHAPE_CATEGORIES) + 2}

# Keypoints of every category, in the order the drawing functions return them. The vertices of the regular polygons
# and the tips of the star go clockwise from the top one, the vertices of each side numbered from the top. The left and
# right of the shapes are those of the image, those of the stick figure its own: it faces the viewer, its left limbs on
# the right of the image as in COCO. A horizontal flip swaps the keypoints whose names differ by left and right. The
# circle, the annulus and the ellipse have no keypoints, distractors left out of the keypoint label mapping
KEYPOINT_NAMES = {"square": ["top_left", "top_right", "bottom_right", "bottom_left"],
                  "circle": [],
                  "triangle": ["top", "bottom_left", "bottom_right"],
                  "rectangle": ["top_left", "top_right", "bottom_right", "bottom_left"],
                  "pentagon": ["top", "upper_right", "lower_right", "lower_left", "upper_left"],
                  "hexagon": ["top", "upper_right", "lower_right", "bottom", "lower_left", "upper_left"],
                  "plus_sign": ["top", "right", "bottom", "left"],
                  "star": ["top", "upper_right", "lower_right", "lower_left", "upper_left"],
                  "octagon": ["top", "right_1", "right_2", "right_3", "bottom", "left_3", "left_2", "left_1"],
                  "rhombus": ["top", "right", "bottom", "left"],
                  "parallelogram": ["top_left", "top_right", "bottom_right", "bottom_left"],
                  "trapezoid": ["top_left", "top_right", "bottom_right", "bottom_left"],
                  "kite": ["top", "right", "bottom", "left"],
                  "heptagon": ["top", "right_1", "right_2", "right_3", "left_3", "left_2", "left_1"],
                  "nonagon": ["top", "right_1", "right_2", "right_3", "right_4", "left_4", "left_3", "left_2",
                              "left_1"],
                  "ellipse": [],
                  "semicircle": ["arc_top", "base_centre", "base_left", "base_right"],
                  "annulus": [],
                  "capsule": ["left_end", "top_left", "top_right", "right_end", "bottom_right", "bottom_left"],
                  "l_shape": ["arm_top_outer", "arm_top_inner", "inner_corner", "foot_end_top", "foot_end_bottom",
                              "outer_corner"],
                  "stick_figure": ["head", "neck", "left_elbow", "right_elbow", "left_hand", "right_hand", "pelvis",
                                   "left_knee", "right_knee", "left_foot", "right_foot"],
                  "house": ["roof_apex", "roof_left", "roof_right", "base_left", "base_right"]}

# Skeleton of every category, drawn by the viewers: the keypoints of an outline linked in a closed loop, in its order,
# and the other links between two keypoints
KEYPOINT_OUTLINES = {"square": KEYPOINT_NAMES["square"],
                     "triangle": ["top", "bottom_right", "bottom_left"],
                     "rectangle": ["top_left", "top_right", "bottom_right", "bottom_left"],
                     "pentagon": KEYPOINT_NAMES["pentagon"], "hexagon": KEYPOINT_NAMES["hexagon"],
                     "star": KEYPOINT_NAMES["star"], "octagon": KEYPOINT_NAMES["octagon"],
                     "rhombus": KEYPOINT_NAMES["rhombus"], "parallelogram": KEYPOINT_NAMES["parallelogram"],
                     "trapezoid": KEYPOINT_NAMES["trapezoid"], "kite": KEYPOINT_NAMES["kite"],
                     "heptagon": KEYPOINT_NAMES["heptagon"], "nonagon": KEYPOINT_NAMES["nonagon"],
                     "semicircle": ["base_left", "arc_top", "base_right", "base_centre"],
                     "capsule": KEYPOINT_NAMES["capsule"],
                     "l_shape": KEYPOINT_NAMES["l_shape"],
                     "house": ["roof_apex", "roof_right", "base_right", "base_left", "roof_left"]}
KEYPOINT_LINKS = {"plus_sign": [["top", "bottom"], ["left", "right"]],
                  "stick_figure": [["head", "neck"], ["neck", "pelvis"], ["neck", "left_elbow"],
                                   ["left_elbow", "left_hand"], ["neck", "right_elbow"], ["right_elbow", "right_hand"],
                                   ["pelvis", "left_knee"], ["left_knee", "left_foot"], ["pelvis", "right_knee"],
                                   ["right_knee", "right_foot"]]}

# Hollow shapes keep, unless they are distractors, the keypoints and the skeleton of their filled shape, their outer
# vertices being those of the filled shape. The hollow polygons add the inner_ keypoints of the corners of their hole
# after the outer ones, linked in a closed loop in the order of the outer outline
for hollow_source_name in HOLLOW_SHAPE_NAMES:
    hollow_name = f"hollow_{hollow_source_name}"

    # Distractors have no keypoints, nor any skeleton
    if hollow_source_name in HOLLOW_DISTRACTOR_SHAPE_NAMES:
        KEYPOINT_NAMES[hollow_name] = []
        continue
    KEYPOINT_NAMES[hollow_name] = list(KEYPOINT_NAMES[hollow_source_name])
    if hollow_source_name in KEYPOINT_OUTLINES:
        KEYPOINT_OUTLINES[hollow_name] = KEYPOINT_OUTLINES[hollow_source_name]
    hollow_links = list(KEYPOINT_LINKS[hollow_source_name]) if hollow_source_name in KEYPOINT_LINKS else []

    # Inner keypoints after the outer ones, and the loop of the hole after the links of the filled shape
    if hollow_source_name in HOLLOW_INNER_KEYPOINT_SHAPE_NAMES:
        KEYPOINT_NAMES[hollow_name] += [f"inner_{name}" for name in KEYPOINT_NAMES[hollow_source_name]]
        inner_outline = [f"inner_{name}" for name in KEYPOINT_OUTLINES[hollow_source_name]]
        hollow_links += [[inner_outline[index], inner_outline[(index + 1) % len(inner_outline)]]
                         for index in range(len(inner_outline))]
    if hollow_links:
        KEYPOINT_LINKS[hollow_name] = hollow_links

# Stick figures, in torso lengths: radius of the head disc, which touches the neck, and for the arms and the legs the
# joint they hang from, the names of their middle and end joints, the lengths of their two segments, the angle of the
# first segment in degrees from straight down towards the side of the limb, the bend of the second segment, and the
# limits of the angle of the second segment, which keep every limb on its own side of the torso
STICK_FIGURE_HEAD_RADIUS = 0.3
STICK_FIGURE_LIMBS = {"arm": {"root": "neck", "joints": ["elbow", "hand"], "lengths": [0.6, 0.55],
                              "first_angle_range": [20.0, 120.0], "bend_range": [-60.0, 60.0],
                              "second_angle_limits": [5.0, 170.0]},
                      "leg": {"root": "pelvis", "joints": ["knee", "foot"], "lengths": [0.65, 0.6],
                              "first_angle_range": [10.0, 45.0], "bend_range": [-25.0, 25.0],
                              "second_angle_limits": [0.0, 60.0]}}

# Width of the lines of a stick figure relative to its size, and mask values of the limbs of each side, which
# draw_shape_descriptor gives colours of their own, the head and the torso having the value 255 of every shape
STICK_FIGURE_THICKNESS_RATIO = 0.06
STICK_FIGURE_SIDE_VALUES = {"left": 1, "right": 2}

# Houses: half width of the walls relative to half the height of the house, and height of the roof relative to the
# height of the house
HOUSE_HALF_WIDTH_RATIO_RANGE = (0.6, 0.9)
HOUSE_ROOF_HEIGHT_RATIO_RANGE = (0.3, 0.5)

# Keypoint images: smallest number of objects drawn, and largest part of the canvas of a shape that may stick out of
# the image, so that some keypoints fall outside it
KEYPOINT_MINIMUM_OBJECTS = 2
KEYPOINT_MAXIMUM_OUTSIDE_FRACTION = 0.3

# Maximum overlap allowed between two objects of the same image, and number of placement attempts per object
MAXIMUM_OBJECT_INTERSECTION_OVER_UNION = 0.3
MAXIMUM_PLACEMENT_ATTEMPTS = 50

# Minimum difference (on the most different channel) between an object colour and the background colour
MINIMUM_COLOUR_DIFFERENCE = 60

# Number of images generated by one worker task
IMAGES_PER_TASK = 100

# Number of sequences generated by one worker task
SEQUENCES_PER_TASK = 20

# Movement directions of the synthetic sequence classification dataset, their category identifiers, and their unit
# displacement in image coordinates (x to the right, y downwards)
MOVEMENT_CATEGORIES = {"moving_left": 1, "moving_right": 2, "moving_up": 3, "moving_down": 4}
MOVEMENT_DIRECTIONS = {"moving_left": (-1, 0), "moving_right": (1, 0), "moving_up": (0, -1), "moving_down": (0, 1)}

# Classes of the synthetic sequence labelling dataset: the frame where a shape of a class appears, the frame where a
# shape of a class disappears, and the frame showing the shapes of the previous frame (always the first frame), 71
# classes for the 35 shapes
CHANGE_CATEGORIES = {"no_change": 1,
                     **{f"appearing_{shape_name}": 2 + index for index, shape_name in enumerate(SHAPE_CATEGORIES)},
                     **{f"disappearing_{shape_name}": 2 + len(SHAPE_CATEGORIES) + index
                        for index, shape_name in enumerate(SHAPE_CATEGORIES)}}

# Minimum number of shapes of a sequence labelling frame, the maximum being the maximum_objects of the dataset, and the
# changes from one frame to the next, drawn with the same probability among those the number of shapes allows
LABELLING_MINIMUM_OBJECTS = 5
LABELLING_CHANGES = ["appearing", "disappearing", "no_change"]

# Classes of the synthetic self-supervised dataset, images drawn like the sequence labelling frames: the number of
# shapes of the image, from LABELLING_MINIMUM_OBJECTS to COUNTED_SHAPES_MAXIMUM_OBJECTS, only used by the k-nearest
# neighbour evaluation of the test split
COUNTED_SHAPES_MAXIMUM_OBJECTS = 10
SHAPE_COUNT_CATEGORIES = {f"{number_objects}_shapes": number_objects - LABELLING_MINIMUM_OBJECTS + 1
                          for number_objects in range(LABELLING_MINIMUM_OBJECTS, COUNTED_SHAPES_MAXIMUM_OBJECTS + 1)}

# Classes of the synthetic sequence comparison dataset: the after member of a pair lost one of the objects of its
# before member, or shows all of them
PAIR_CATEGORIES = {"loss_of_object": 1, "same_number_of_objects": 2}

# Classes of the synthetic frame selection dataset: the largest copy of every wanted shape of a sequence is the frame to
# select, every other frame (a smaller copy of a wanted shape, or another shape) being unselected
SELECTION_CATEGORIES = {"unselected": 1, "circle": 2, "triangle": 3, "star": 4}
SELECTION_WANTED_SHAPES = ["circle", "triangle", "star"]
SELECTION_OTHER_SHAPES = ["square", "rectangle", "plus_sign"]

# Frame selection shapes: size of the largest copies as a fraction of the image side, probability that an unselected
# frame shows a smaller copy of a wanted shape rather than another shape, largest size of a smaller copy relative to
# the largest copy, and largest area of a smaller copy relative to the largest copy (a triangle filling its square more
# or less), so that the largest copy is always clearly the largest
SELECTION_LARGEST_SIZE_RANGE = (0.4, 0.6)
SELECTION_SMALLER_COPY_PROBABILITY = 0.5
SELECTION_SMALLER_SIZE_RATIO = 0.7
SELECTION_SMALLER_AREA_RATIO = 0.6

# Minimum number of objects of the before member of a pair
PAIR_MINIMUM_OBJECTS = 2

# Minimum number of frames of a sequence or of a member of a pair: a movement, a change of shapes or a comparison of
# the numbers of objects needs two frames
MINIMUM_SEQUENCE_FRAMES = 2

# Minimum gap in pixels between two objects of a pair frame, so that every object is countable, and number of
# attempts at drawing a frame where every object fits
MINIMUM_OBJECT_GAP = 2
MAXIMUM_FRAME_ATTEMPTS = 20

# Detection, instance segmentation and keypoint detection: an object with fewer visible pixels than this, mostly
# covered by later objects or outside the image, is left out of the labels
MINIMUM_VISIBLE_OBJECT_PIXELS = 64
TASKS_REMOVING_COVERED_OBJECTS = ["object_detection", "instance_segmentation", "keypoint_detection"]

# Categories of the synthetic dataset of every task, and the tasks whose samples are sequences of frames (one sequence,
# or a pair of sequences)
TASK_CATEGORIES = {"object_detection": SHAPE_CATEGORIES, "instance_segmentation": SHAPE_CATEGORIES,
                   "keypoint_detection": KEYPOINT_CATEGORIES, "classification": SHAPE_CATEGORIES,
                   "semantic_segmentation": SHAPE_CATEGORIES, "sequence_classification": MOVEMENT_CATEGORIES,
                   "sequence_labelling": CHANGE_CATEGORIES, "sequence_comparison": PAIR_CATEGORIES,
                   "frame_selection": SELECTION_CATEGORIES, "self_supervised": SHAPE_COUNT_CATEGORIES}

# Tasks whose model predicts a background class at index 0, so that the category identifiers numbered from 1 are their
# class indices, the keypoint detection keeping the class indices of the detector that will give its boxes; the other
# tasks predict their categories from class index 0
TASKS_WITH_BACKGROUND_CLASS = ["object_detection", "instance_segmentation", "semantic_segmentation",
                               "keypoint_detection"]
SEQUENCE_TASKS = ["sequence_classification", "sequence_labelling", "sequence_comparison", "frame_selection"]

# Key of the samples in the manifest of every sequence task, and name of the samples of every task in the console
TASK_MANIFEST_KEYS = {"sequence_classification": "sequences", "sequence_labelling": "sequences",
                      "sequence_comparison": "pairs", "frame_selection": "sequences"}
TASK_SAMPLE_NAMES = {"object_detection": "images", "instance_segmentation": "images", "keypoint_detection": "images",
                     "classification": "images", "semantic_segmentation": "images",
                     "sequence_classification": "sequences", "sequence_labelling": "sequences",
                     "sequence_comparison": "pairs", "frame_selection": "sequences", "self_supervised": "images"}


def get_rectangle_corners(left: int, top: int, width: int, height: int) -> np.ndarray:
    """
    Computes the keypoints of the four corners of a filled axis-aligned rectangle of pixels, the outer corners of its
    corner pixels, shared by the squares and rectangles of the keypoint images.

    Args:
        left (int): The column of the leftmost pixels of the rectangle.
        top (int): The row of the topmost pixels of the rectangle.
        width (int): The number of columns of the rectangle.
        height (int): The number of rows of the rectangle.

    Returns:
        np.ndarray: The [4, 2] float top left, top right, bottom right and bottom left corners, in the continuous
            coordinates of the mask (pixel (x, y) covering [x, x + 1] x [y, y + 1]).
    """
    return np.array([[left, top], [left + width, top], [left + width, top + height], [left, top + height]],
                    dtype=np.float64)


def draw_square_mask(mask: np.ndarray, center_x: int, center_y: int, size: int, random_generator: np.random.Generator,
                     upright: bool = False) -> np.ndarray:
    """
    Draws a filled axis-aligned square of side 'size' on a binary mask, one of the synthetic shape classes, and returns
    its keypoints for the keypoint images.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The side length of the square in pixels.
        random_generator (np.random.Generator): Unused, kept so that every shape shares the same signature.
        upright (bool): Unused, every square being axis-aligned, kept so that every shape shares the same signature.
            Defaults to False.

    Returns:
        np.ndarray: The [4, 2] float outer corners in the order of KEYPOINT_NAMES, in the continuous coordinates of
            the mask (pixel (x, y) covering [x, x + 1] x [y, y + 1]).
    """
    half_size = size // 2
    left, top = center_x - half_size, center_y - half_size
    cv2.rectangle(img=mask, pt1=(left, top), pt2=(left + size - 1, top + size - 1), color=255, thickness=-1)

    # Four outer corners, clockwise from the top left one
    return get_rectangle_corners(left=left, top=top, width=size, height=size)


def draw_circle_mask(mask: np.ndarray, center_x: int, center_y: int, size: int, random_generator: np.random.Generator,
                     upright: bool = False) -> np.ndarray:
    """
    Draws a filled circle of diameter 'size' on a binary mask, one of the synthetic shape classes, a distractor
    without keypoints in the keypoint images.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The diameter of the circle in pixels.
        random_generator (np.random.Generator): Unused, kept so that every shape shares the same signature.
        upright (bool): Unused, kept so that every shape shares the same signature. Defaults to False.

    Returns:
        np.ndarray: The empty [0, 2] float keypoints of the circle.
    """
    cv2.circle(img=mask, center=(center_x, center_y), radius=size // 2, color=255, thickness=-1)
    return np.zeros(shape=(0, 2), dtype=np.float64)


def draw_triangle_mask(mask: np.ndarray, center_x: int, center_y: int, size: int,
                       random_generator: np.random.Generator, upright: bool = False) -> np.ndarray:
    """
    Draws a filled triangle with three random vertices inside a 'size' x 'size' square on a binary mask, one of the
    synthetic shape classes. Vertices are redrawn until the triangle is not too flat to be recognisable. The upright
    triangles of the keypoint images have their apex on the top side of the square and their base on its bottom side,
    so that their vertices are named by their height.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape region in pixels.
        center_y (int): The vertical centre of the shape region in pixels.
        size (int): The side length of the square region containing the vertices.
        random_generator (np.random.Generator): The random generator drawing the vertices.
        upright (bool): True for an apex above a horizontal base (keypoint images). Defaults to False.

    Returns:
        np.ndarray: The [3, 2] float vertices, the top, bottom left and bottom right ones for an upright triangle, the
            centres of their pixels in the continuous coordinates of the mask.
    """
    half_size = size // 2
    while True:
        # Upright triangles: the apex anywhere on the top side, the two ends of the base on the bottom side
        if upright:
            base_abscissas = np.sort(random_generator.integers(low=-half_size, high=half_size + 1, size=2))
            apex_abscissa = int(random_generator.integers(low=-half_size, high=half_size + 1))
            vertices = np.array([[apex_abscissa, -half_size], [base_abscissas[0], half_size],
                                 [base_abscissas[1], half_size]])
        else:
            vertices = random_generator.integers(low=-half_size, high=half_size + 1, size=(3, 2))

        # Twice the triangle area, from the cross product of two of its edges
        first_edge, second_edge = vertices[1] - vertices[0], vertices[2] - vertices[0]
        doubled_area = abs(int(first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]))
        if doubled_area >= 2 * MINIMUM_TRIANGLE_AREA_FRACTION * size * size:
            break

    points = (vertices + np.array([center_x, center_y])).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(img=mask, pts=[points], color=255)
    return points.reshape(-1, 2) + 0.5


def draw_rectangle_mask(mask: np.ndarray, center_x: int, center_y: int, size: int,
                        random_generator: np.random.Generator, upright: bool = False) -> np.ndarray:
    """
    Draws a filled axis-aligned rectangle whose long side is 'size' and whose aspect ratio is between 1.5 and 3,
    either horizontal or vertical, on a binary mask. One of the synthetic shape classes. The upright rectangles of the
    keypoint images are horizontal, so that their corners keep their names under a rotation of up to 90 degrees.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The length of the long side in pixels.
        random_generator (np.random.Generator): The random generator drawing the aspect ratio and orientation.
        upright (bool): True for a horizontal rectangle (keypoint images). Defaults to False.

    Returns:
        np.ndarray: The [4, 2] float outer corners from the top left one clockwise, in the continuous coordinates of
            the mask.
    """
    short_side = max(2, int(round(size / random_generator.uniform(*RECTANGLE_ASPECT_RATIO_RANGE))))
    width, height = (size, short_side) if upright or random_generator.random() < 0.5 else (short_side, size)

    top_left_x, top_left_y = center_x - width // 2, center_y - height // 2
    cv2.rectangle(img=mask, pt1=(top_left_x, top_left_y), pt2=(top_left_x + width - 1, top_left_y + height - 1),
                  color=255, thickness=-1)
    return get_rectangle_corners(left=top_left_x, top=top_left_y, width=width, height=height)


def centre_vertices(vertices: np.ndarray) -> np.ndarray:
    """
    Moves the vertices of a polygon built by one of the vertex builders below so that the centre of its bounding box
    is at the origin, the centre fill_rotated_polygon turns the shape around and places at the centre of its square.

    Args:
        vertices (np.ndarray): The [N, 2] vertices of the polygon.

    Returns:
        np.ndarray: The [N, 2] vertices, centred.
    """
    return vertices - (vertices.min(axis=0) + vertices.max(axis=0)) / 2


def fill_rotated_polygon(mask: np.ndarray, center_x: int, center_y: int, size: int, vertices: np.ndarray,
                         random_generator: np.random.Generator, upright: bool = False,
                         keypoint_vertices: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Fills on a binary mask a polygon given by its vertices around its centre, mirrored for half of the shapes and
    turned by a random angle, then scaled so that its farthest vertex lies on the circle of diameter 'size' and kept
    inside the 'size' x 'size' square of the shape. Shared by the regular polygons and the stars (fill_radial_polygon)
    and by the rhombuses, parallelograms, trapezoids, kites, capsules, L-shapes and houses, whose mirror images (a
    parallelogram leaning the other way, an L turned into a J) belong to the same class. The upright shapes of the
    keypoint images keep the orientation of their vertices, so that their keypoints are named by their position.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The diameter of the circle the farthest vertex lies on, in pixels.
        vertices (np.ndarray): The [N, 2] vertices of the polygon in order around its outline, centred on the origin,
            at any scale.
        random_generator (np.random.Generator): The random generator drawing the mirror and the rotation.
        upright (bool): True to draw the vertices without mirror nor rotation (keypoint images). Defaults to False.
        keypoint_vertices (Optional[np.ndarray]): The [K, 2] keypoints of a shape whose keypoints are not its vertices
            (the ends and corners of a capsule), in the frame of 'vertices', or None for the vertices. Defaults to None.

    Returns:
        np.ndarray: The [N, 2] float vertices drawn, the centres of their pixels, or the [K, 2] float keypoint
            vertices moved with them, in the continuous coordinates of the mask.
    """
    # Mirror half of the shapes, then turn them by a random angle around their centre, with their keypoint vertices
    if not upright:
        mirror = np.array([-1.0, 1.0]) if random_generator.random() < 0.5 else np.array([1.0, 1.0])
        rotation_angle = random_generator.uniform(low=0.0, high=2 * np.pi)
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                    [np.sin(rotation_angle), np.cos(rotation_angle)]])
        vertices = (vertices * mirror) @ rotation_matrix.T
        if keypoint_vertices is not None:
            keypoint_vertices = (keypoint_vertices * mirror) @ rotation_matrix.T

    # Scale the farthest vertex to the outer radius, then keep the vertices inside the 'size' x 'size' square
    farthest_norm = np.linalg.norm(vertices, axis=1).max()
    vertices = vertices * (size / 2) / farthest_norm + np.array([center_x, center_y])
    points = np.clip(np.round(vertices), a_min=[center_x - size // 2, center_y - size // 2],
                     a_max=[center_x + size // 2, center_y + size // 2]).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(img=mask, pts=[points], color=255)

    # Keypoints at the centres of the vertex pixels, or the keypoint vertices scaled and moved like the vertices
    if keypoint_vertices is None:
        return points.reshape(-1, 2) + 0.5
    return keypoint_vertices * (size / 2) / farthest_norm + np.array([center_x, center_y]) + 0.5


def fill_radial_polygon(mask: np.ndarray, center_x: int, center_y: int, size: int, vertex_radii: List[float],
                        random_generator: np.random.Generator, upright: bool = False) -> np.ndarray:
    """
    Fills on a binary mask a polygon whose vertices are evenly spaced around its centre, at the given fractions of the
    outer radius 'size' / 2, at a random rotation: a regular polygon when every radius is 1, a star when the radii
    alternate between 1 and a smaller one. Shared by the pentagons, hexagons, heptagons, octagons, nonagons and stars.
    The upright shapes of the keypoint images have their first vertex at the top and the next ones clockwise.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The diameter of the circle the outer vertices lie on, in pixels.
        vertex_radii (List[float]): The radius of every vertex in turn, as a fraction of the outer radius, the
            largest being 1.
        random_generator (np.random.Generator): The random generator drawing the rotation of the shape.
        upright (bool): True for the first vertex at the top, without rotation (keypoint images). Defaults to False.

    Returns:
        np.ndarray: The [N, 2] float vertices in order, the centres of their pixels in the continuous coordinates of
            the mask.
    """
    # Vertices evenly spaced around the centre at their radius, from the top clockwise when upright (the y axis
    # pointing down), turned by fill_rotated_polygon otherwise
    start_angle = -np.pi / 2 if upright else 0.0
    angles = start_angle + 2 * np.pi * np.arange(len(vertex_radii)) / len(vertex_radii)
    radii = np.array(vertex_radii)
    vertices = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=-1)
    return fill_rotated_polygon(mask=mask, center_x=center_x, center_y=center_y, size=size, vertices=vertices,
                                random_generator=random_generator, upright=upright)


def draw_pentagon_mask(mask: np.ndarray, center_x: int, center_y: int, size: int,
                       random_generator: np.random.Generator, upright: bool = False) -> np.ndarray:
    """
    Draws a filled regular pentagon inscribed in a circle of diameter 'size', at a random rotation, on a binary mask.
    One of the synthetic shape classes.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The diameter of the circle the vertices lie on, in pixels.
        random_generator (np.random.Generator): The random generator drawing the rotation.
        upright (bool): True for a vertex at the top, without rotation (keypoint images). Defaults to False.

    Returns:
        np.ndarray: The [5, 2] float vertices, from the top one clockwise when upright.
    """
    return fill_radial_polygon(mask=mask, center_x=center_x, center_y=center_y, size=size, vertex_radii=[1.0] * 5,
                               random_generator=random_generator, upright=upright)


def draw_hexagon_mask(mask: np.ndarray, center_x: int, center_y: int, size: int,
                      random_generator: np.random.Generator, upright: bool = False) -> np.ndarray:
    """
    Draws a filled regular hexagon inscribed in a circle of diameter 'size', at a random rotation, on a binary mask.
    One of the synthetic shape classes.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The diameter of the circle the vertices lie on, in pixels.
        random_generator (np.random.Generator): The random generator drawing the rotation.
        upright (bool): True for a vertex at the top, without rotation (keypoint images). Defaults to False.

    Returns:
        np.ndarray: The [6, 2] float vertices, from the top one clockwise when upright.
    """
    return fill_radial_polygon(mask=mask, center_x=center_x, center_y=center_y, size=size, vertex_radii=[1.0] * 6,
                               random_generator=random_generator, upright=upright)


def draw_octagon_mask(mask: np.ndarray, center_x: int, center_y: int, size: int,
                      random_generator: np.random.Generator, upright: bool = False) -> np.ndarray:
    """
    Draws a filled regular octagon inscribed in a circle of diameter 'size', at a random rotation, on a binary mask.
    One of the synthetic shape classes, close to the circle at small sizes.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The diameter of the circle the vertices lie on, in pixels.
        random_generator (np.random.Generator): The random generator drawing the rotation.
        upright (bool): True for a vertex at the top, without rotation (keypoint images). Defaults to False.

    Returns:
        np.ndarray: The [8, 2] float vertices, from the top one clockwise when upright.
    """
    return fill_radial_polygon(mask=mask, center_x=center_x, center_y=center_y, size=size, vertex_radii=[1.0] * 8,
                               random_generator=random_generator, upright=upright)


def draw_star_mask(mask: np.ndarray, center_x: int, center_y: int, size: int,
                   random_generator: np.random.Generator, upright: bool = False) -> np.ndarray:
    """
    Draws a filled five-pointed star whose points lie on a circle of diameter 'size' and whose inner vertices lie at
    0.45 of that radius, at a random rotation, on a binary mask. One of the synthetic shape classes.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The diameter of the circle the points lie on, in pixels.
        random_generator (np.random.Generator): The random generator drawing the rotation.
        upright (bool): True for a point at the top, without rotation (keypoint images). Defaults to False.

    Returns:
        np.ndarray: The [5, 2] float tips of the points, from the top one clockwise when upright.
    """
    vertices = fill_radial_polygon(mask=mask, center_x=center_x, center_y=center_y, size=size,
                                   vertex_radii=[1.0, STAR_INNER_RADIUS_RATIO] * 5, random_generator=random_generator,
                                   upright=upright)
    return vertices[0::2]


def draw_plus_sign_mask(mask: np.ndarray, center_x: int, center_y: int, size: int,
                        random_generator: np.random.Generator, upright: bool = False) -> np.ndarray:
    """
    Draws a filled axis-aligned plus sign whose two arms are 'size' long and 0.35 of that wide, on a binary mask. One
    of the synthetic shape classes.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The length of the arms in pixels.
        random_generator (np.random.Generator): Unused, kept so that every shape shares the same signature.
        upright (bool): Unused, every plus sign being axis-aligned, kept so that every shape shares the same
            signature. Defaults to False.

    Returns:
        np.ndarray: The [4, 2] float middles of the top, right, bottom and left ends of the arms, in the continuous
            coordinates of the mask.
    """
    half_size = size // 2
    arm_width = max(2, int(round(size * PLUS_SIGN_ARM_WIDTH_RATIO)))
    half_width = arm_width // 2

    # Horizontal arm, then vertical arm, crossing at the centre
    cv2.rectangle(img=mask, pt1=(center_x - half_size, center_y - half_width),
                  pt2=(center_x - half_size + size - 1, center_y - half_width + arm_width - 1), color=255, thickness=-1)
    cv2.rectangle(img=mask, pt1=(center_x - half_width, center_y - half_size),
                  pt2=(center_x - half_width + arm_width - 1, center_y - half_size + size - 1), color=255, thickness=-1)

    # Middle lines of the vertical and horizontal arms, and the outer edges of their ends
    vertical_arm_x, horizontal_arm_y = center_x - half_width + arm_width / 2, center_y - half_width + arm_width / 2
    left_edge, right_edge = center_x - half_size, center_x - half_size + size
    top_edge, bottom_edge = center_y - half_size, center_y - half_size + size
    return np.array([[vertical_arm_x, top_edge], [right_edge, horizontal_arm_y], [vertical_arm_x, bottom_edge],
                     [left_edge, horizontal_arm_y]])


def build_rhombus_vertices(random_generator: np.random.Generator) -> np.ndarray:
    """
    Builds the vertices of a rhombus for draw_rhombus_mask: four equal sides around perpendicular diagonals, the short
    diagonal 0.4 to 0.7 of the long one, so that its acute angles (44 to 70 degrees) keep it apart from the squares.

    Args:
        random_generator (np.random.Generator): The random generator drawing the ratio of the diagonals.

    Returns:
        np.ndarray: The [4, 2] vertices, centred, the long diagonal along the x axis.
    """
    diagonal_ratio = random_generator.uniform(low=RHOMBUS_DIAGONAL_RATIO_RANGE[0], high=RHOMBUS_DIAGONAL_RATIO_RANGE[1])
    return np.array([[1.0, 0.0], [0.0, diagonal_ratio], [-1.0, 0.0], [0.0, -diagonal_ratio]])


def draw_rhombus_mask(mask: np.ndarray, center_x: int, center_y: int, size: int,
                      random_generator: np.random.Generator, upright: bool = False) -> np.ndarray:
    """
    Draws a filled rhombus whose long diagonal is 'size' long, at a random rotation, on a binary mask. One of the
    synthetic shape classes.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The length of the long diagonal in pixels.
        random_generator (np.random.Generator): The random generator drawing the diagonals and the rotation.
        upright (bool): True for a horizontal long diagonal, without mirror nor rotation (keypoint images). Defaults
            to False.

    Returns:
        np.ndarray: The [4, 2] float vertices, the top, right, bottom and left ones when upright.
    """
    vertices = fill_rotated_polygon(mask=mask, center_x=center_x, center_y=center_y, size=size,
                                    vertices=build_rhombus_vertices(random_generator=random_generator),
                                    random_generator=random_generator, upright=upright)
    return vertices[[3, 0, 1, 2]]


def build_parallelogram_vertices(random_generator: np.random.Generator) -> np.ndarray:
    """
    Builds the vertices of a parallelogram for draw_parallelogram_mask: a long side 1.5 to 2.5 times its short side,
    unlike the equal sides of the rhombuses, meeting it at an acute angle of 45 to 65 degrees, unlike the right angles
    of the rectangles.

    Args:
        random_generator (np.random.Generator): The random generator drawing the side ratio and the angle.

    Returns:
        np.ndarray: The [4, 2] vertices, centred, the long sides along the x axis.
    """
    side_ratio = random_generator.uniform(low=PARALLELOGRAM_SIDE_RATIO_RANGE[0], high=PARALLELOGRAM_SIDE_RATIO_RANGE[1])
    angle = np.radians(random_generator.uniform(low=PARALLELOGRAM_ANGLE_RANGE_DEGREES[0],
                                                high=PARALLELOGRAM_ANGLE_RANGE_DEGREES[1]))

    # Short side leaning from the end of the long side, the opposite sides being parallel and equal
    short_side_x, short_side_y = np.cos(angle) / side_ratio, np.sin(angle) / side_ratio
    return centre_vertices(vertices=np.array([[0.0, 0.0], [1.0, 0.0], [1.0 + short_side_x, short_side_y],
                                              [short_side_x, short_side_y]]))


def draw_parallelogram_mask(mask: np.ndarray, center_x: int, center_y: int, size: int,
                            random_generator: np.random.Generator, upright: bool = False) -> np.ndarray:
    """
    Draws a filled parallelogram inscribed in a circle of diameter 'size', at a random rotation and mirrored for half
    of them, on a binary mask. One of the synthetic shape classes.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The diameter of the circle its farthest vertices lie on, in pixels.
        random_generator (np.random.Generator): The random generator drawing the sides, the angle and the rotation.
        upright (bool): True for horizontal long sides, the bottom one shifted to the right, without mirror nor
            rotation (keypoint images). Defaults to False.

    Returns:
        np.ndarray: The [4, 2] float vertices, from the top left one clockwise when upright.
    """
    return fill_rotated_polygon(mask=mask, center_x=center_x, center_y=center_y, size=size,
                                vertices=build_parallelogram_vertices(random_generator=random_generator),
                                random_generator=random_generator, upright=upright)


def build_trapezoid_vertices(random_generator: np.random.Generator) -> np.ndarray:
    """
    Builds the vertices of a trapezoid for draw_trapezoid_mask: two parallel sides, the short one 0.35 to 0.65 of the
    long one (neither the equal sides of a parallelogram nor the single corner of a triangle) above the middle half of
    the rest of the long side, isosceles trapezoids included, at a height of 0.35 to 0.65 of the long side.

    Args:
        random_generator (np.random.Generator): The random generator drawing the short side, its position and the
            height.

    Returns:
        np.ndarray: The [4, 2] vertices, centred, the parallel sides along the x axis.
    """
    top_ratio = random_generator.uniform(low=TRAPEZOID_TOP_RATIO_RANGE[0], high=TRAPEZOID_TOP_RATIO_RANGE[1])
    height = random_generator.uniform(low=TRAPEZOID_HEIGHT_RATIO_RANGE[0], high=TRAPEZOID_HEIGHT_RATIO_RANGE[1])

    # Short side above the middle of the long side, both angles of the long side staying acute
    top_offset = (1.0 - top_ratio) * random_generator.uniform(low=TRAPEZOID_TOP_POSITION_RANGE[0],
                                                              high=TRAPEZOID_TOP_POSITION_RANGE[1])
    return centre_vertices(vertices=np.array([[0.0, 0.0], [1.0, 0.0], [top_offset + top_ratio, height],
                                              [top_offset, height]]))


def draw_trapezoid_mask(mask: np.ndarray, center_x: int, center_y: int, size: int,
                        random_generator: np.random.Generator, upright: bool = False) -> np.ndarray:
    """
    Draws a filled trapezoid inscribed in a circle of diameter 'size', at a random rotation and mirrored for half of
    them, on a binary mask. One of the synthetic shape classes. The upright trapezoids of the keypoint images stand on
    their long side.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The diameter of the circle its farthest vertices lie on, in pixels.
        random_generator (np.random.Generator): The random generator drawing the sides, the height and the rotation.
        upright (bool): True for a trapezoid standing on its long side, without mirror nor rotation (keypoint images).
            Defaults to False.

    Returns:
        np.ndarray: The [4, 2] float vertices, from the top left one clockwise when upright.
    """
    vertices = build_trapezoid_vertices(random_generator=random_generator)

    # Upright trapezoids: the long side, built at the top (the y axis pointing down), turned to the bottom, which
    # reverses the order of the vertices
    if upright:
        keypoints = fill_rotated_polygon(mask=mask, center_x=center_x, center_y=center_y, size=size,
                                         vertices=centre_vertices(vertices=vertices * np.array([1.0, -1.0])),
                                         random_generator=random_generator, upright=True)
        return keypoints[[3, 2, 1, 0]]
    return fill_rotated_polygon(mask=mask, center_x=center_x, center_y=center_y, size=size, vertices=vertices,
                                random_generator=random_generator)


def build_kite_vertices(random_generator: np.random.Generator) -> np.ndarray:
    """
    Builds the vertices of a kite for draw_kite_mask: two pairs of equal adjacent sides around perpendicular
    diagonals, the short diagonal crossing the long one at 26 to 38 % of its length (at its middle for a rhombus) and
    being 0.5 to 0.7 of its length.

    Args:
        random_generator (np.random.Generator): The random generator drawing the crossing and the width.

    Returns:
        np.ndarray: The [4, 2] vertices, centred, the long diagonal along the y axis.
    """
    upper_ratio = random_generator.uniform(low=KITE_UPPER_RATIO_RANGE[0], high=KITE_UPPER_RATIO_RANGE[1])
    width_ratio = random_generator.uniform(low=KITE_WIDTH_RATIO_RANGE[0], high=KITE_WIDTH_RATIO_RANGE[1])

    # Short corner above the crossing of the diagonals, long corner below it, side corners on the short diagonal
    half_width = width_ratio * (1.0 + upper_ratio) / 2
    return centre_vertices(vertices=np.array([[0.0, -upper_ratio], [half_width, 0.0], [0.0, 1.0],
                                              [-half_width, 0.0]]))


def draw_kite_mask(mask: np.ndarray, center_x: int, center_y: int, size: int,
                   random_generator: np.random.Generator, upright: bool = False) -> np.ndarray:
    """
    Draws a filled kite inscribed in a circle of diameter 'size', at a random rotation, on a binary mask. One of the
    synthetic shape classes.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The diameter of the circle its farthest vertices lie on, in pixels.
        random_generator (np.random.Generator): The random generator drawing the diagonals and the rotation.
        upright (bool): True for a vertical long diagonal, the short corner at the top, without rotation (keypoint
            images). Defaults to False.

    Returns:
        np.ndarray: The [4, 2] float vertices, the top, right, bottom and left ones when upright.
    """
    return fill_rotated_polygon(mask=mask, center_x=center_x, center_y=center_y, size=size,
                                vertices=build_kite_vertices(random_generator=random_generator),
                                random_generator=random_generator, upright=upright)


def draw_heptagon_mask(mask: np.ndarray, center_x: int, center_y: int, size: int,
                       random_generator: np.random.Generator, upright: bool = False) -> np.ndarray:
    """
    Draws a filled regular heptagon inscribed in a circle of diameter 'size', at a random rotation, on a binary mask.
    One of the synthetic shape classes.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The diameter of the circle the vertices lie on, in pixels.
        random_generator (np.random.Generator): The random generator drawing the rotation.
        upright (bool): True for a vertex at the top, without rotation (keypoint images). Defaults to False.

    Returns:
        np.ndarray: The [7, 2] float vertices, from the top one clockwise when upright.
    """
    return fill_radial_polygon(mask=mask, center_x=center_x, center_y=center_y, size=size, vertex_radii=[1.0] * 7,
                               random_generator=random_generator, upright=upright)


def draw_nonagon_mask(mask: np.ndarray, center_x: int, center_y: int, size: int,
                      random_generator: np.random.Generator, upright: bool = False) -> np.ndarray:
    """
    Draws a filled regular nonagon inscribed in a circle of diameter 'size', at a random rotation, on a binary mask.
    One of the synthetic shape classes, the closest to the circle, which sets MINIMUM_SHAPE_SIZE.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The diameter of the circle the vertices lie on, in pixels.
        random_generator (np.random.Generator): The random generator drawing the rotation.
        upright (bool): True for a vertex at the top, without rotation (keypoint images). Defaults to False.

    Returns:
        np.ndarray: The [9, 2] float vertices, from the top one clockwise when upright.
    """
    return fill_radial_polygon(mask=mask, center_x=center_x, center_y=center_y, size=size, vertex_radii=[1.0] * 9,
                               random_generator=random_generator, upright=upright)


def draw_ellipse_mask(mask: np.ndarray, center_x: int, center_y: int, size: int,
                      random_generator: np.random.Generator, upright: bool = False) -> np.ndarray:
    """
    Draws a filled ellipse whose long axis is 'size' long and 1.5 to 3 times its short axis, far from the circles, at
    a random rotation, on a binary mask. One of the synthetic shape classes, a distractor without keypoints in the
    keypoint images.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The length of the long axis in pixels.
        random_generator (np.random.Generator): The random generator drawing the elongation and the rotation.
        upright (bool): True for a horizontal long axis, without rotation (keypoint images). Defaults to False.

    Returns:
        np.ndarray: The empty [0, 2] float keypoints of the ellipse.
    """
    aspect_ratio = random_generator.uniform(low=ELLIPSE_ASPECT_RATIO_RANGE[0], high=ELLIPSE_ASPECT_RATIO_RANGE[1])
    half_short_axis = max(1, int(round(size / 2 / aspect_ratio)))
    angle = 0.0 if upright else random_generator.uniform(low=0.0, high=180.0)
    cv2.ellipse(img=mask, center=(center_x, center_y), axes=(size // 2, half_short_axis), angle=angle, startAngle=0.0,
                endAngle=360.0, color=255, thickness=-1)
    return np.zeros(shape=(0, 2), dtype=np.float64)


def draw_semicircle_mask(mask: np.ndarray, center_x: int, center_y: int, size: int,
                         random_generator: np.random.Generator, upright: bool = False) -> np.ndarray:
    """
    Draws a filled half disc of diameter 'size', its straight side through the centre of the shape, at a random
    rotation, on a binary mask. One of the synthetic shape classes, filling half of its square.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The diameter of the half disc in pixels.
        random_generator (np.random.Generator): The random generator drawing the rotation.
        upright (bool): True for a half disc lying on its straight side, the arc at the top (keypoint images).
            Defaults to False.

    Returns:
        np.ndarray: The [4, 2] float keypoints, the middle of the arc, the centre of the straight side and its two
            ends (left then right when upright), in the continuous coordinates of the mask.
    """
    angle = 180.0 if upright else random_generator.uniform(low=0.0, high=360.0)
    cv2.ellipse(img=mask, center=(center_x, center_y), axes=(size // 2, size // 2), angle=angle, startAngle=0.0,
                endAngle=180.0, color=255, thickness=-1)

    # Directions of the middle and of the start of the arc, which cv2.ellipse draws clockwise on the image from its
    # turned x axis. The straight side runs along the centre row of pixels, on its edge away from the arc
    radians = np.radians(angle)
    arc_direction, side_direction = np.array([-np.sin(radians), np.cos(radians)]), np.array([np.cos(radians),
                                                                                             np.sin(radians)])
    centre, radius = np.array([center_x + 0.5, center_y + 0.5]), size // 2 + 0.5
    side_centre = centre - 0.5 * arc_direction
    return np.stack([centre + radius * arc_direction, side_centre, side_centre + radius * side_direction,
                     side_centre - radius * side_direction])


def draw_annulus_mask(mask: np.ndarray, center_x: int, center_y: int, size: int,
                      random_generator: np.random.Generator, upright: bool = False) -> np.ndarray:
    """
    Draws a filled ring of outer diameter 'size' whose hole has 0.4 to 0.6 of its radius, on a binary mask. One of
    the synthetic shape classes, the hollow circle, whose hole, like those of the hollow shapes (draw_hollow_mask),
    shows the background (or the objects below it) and is left out of its instance mask. The hole is cleared on the
    mask, which must hold this shape only. Like the circle, it is a distractor without keypoints in the keypoint images.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The outer diameter of the ring in pixels.
        random_generator (np.random.Generator): The random generator drawing the size of the hole.
        upright (bool): Unused, kept so that every shape shares the same signature. Defaults to False.

    Returns:
        np.ndarray: The empty [0, 2] float keypoints of the ring.
    """
    outer_radius = size // 2
    inner_radius_ratio = random_generator.uniform(low=ANNULUS_INNER_RADIUS_RATIO_RANGE[0],
                                                  high=ANNULUS_INNER_RADIUS_RATIO_RANGE[1])

    # Disc of the ring, then its hole cleared
    cv2.circle(img=mask, center=(center_x, center_y), radius=outer_radius, color=255, thickness=-1)
    cv2.circle(img=mask, center=(center_x, center_y), radius=max(1, int(round(outer_radius * inner_radius_ratio))),
               color=0, thickness=-1)
    return np.zeros(shape=(0, 2), dtype=np.float64)


def build_capsule_vertices(random_generator: np.random.Generator) -> np.ndarray:
    """
    Builds the outline of a capsule for draw_capsule_mask: a rectangle closed by two half discs, 1.5 to 3 times as
    long as it is wide, its straight sides keeping it apart from the ellipses and its round ends from the rectangles.

    Args:
        random_generator (np.random.Generator): The random generator drawing the elongation.

    Returns:
        np.ndarray: The [2 x CAPSULE_END_VERTICES, 2] vertices of the outline, centred, the long axis along the x axis
            from -1 to 1.
    """
    aspect_ratio = random_generator.uniform(low=CAPSULE_ASPECT_RATIO_RANGE[0], high=CAPSULE_ASPECT_RATIO_RANGE[1])
    half_width = 1.0 / aspect_ratio
    end_centre = 1.0 - half_width

    # Round end on the right from the bottom to the top, then round end on the left from the top to the bottom
    right_angles = np.linspace(start=-np.pi / 2, stop=np.pi / 2, num=CAPSULE_END_VERTICES)
    left_angles = right_angles + np.pi
    right_end = np.stack([end_centre + half_width * np.cos(right_angles), half_width * np.sin(right_angles)], axis=-1)
    left_end = np.stack([-end_centre + half_width * np.cos(left_angles), half_width * np.sin(left_angles)], axis=-1)
    return np.concatenate([right_end, left_end], axis=0)


def draw_capsule_mask(mask: np.ndarray, center_x: int, center_y: int, size: int,
                      random_generator: np.random.Generator, upright: bool = False) -> np.ndarray:
    """
    Draws a filled capsule (pill shape) 'size' long, at a random rotation, on a binary mask. One of the synthetic shape
    classes.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The length of the capsule in pixels.
        random_generator (np.random.Generator): The random generator drawing the elongation and the rotation.
        upright (bool): True for a horizontal capsule, without mirror nor rotation (keypoint images). Defaults to
            False.

    Returns:
        np.ndarray: The [6, 2] float keypoints in the order of KEYPOINT_NAMES, clockwise from the left end of the long
            axis when upright: the two ends of the long axis and the four corners where the straight sides meet the
            round ends, in the continuous coordinates of the mask.
    """
    vertices = build_capsule_vertices(random_generator=random_generator)

    # Corners at the ends of the straight sides, the first vertex being the bottom end of the right half disc
    end_centre, half_width = vertices[0, 0], vertices[:, 1].max()
    keypoint_vertices = np.array([[-1.0, 0.0], [-end_centre, -half_width], [end_centre, -half_width], [1.0, 0.0],
                                  [end_centre, half_width], [-end_centre, half_width]])
    return fill_rotated_polygon(mask=mask, center_x=center_x, center_y=center_y, size=size, vertices=vertices,
                                random_generator=random_generator, upright=upright,
                                keypoint_vertices=keypoint_vertices)


def build_l_shape_vertices(random_generator: np.random.Generator) -> np.ndarray:
    """
    Builds the vertices of an L-shape for draw_l_shape_mask: a long arm and a foot 0.6 to 1 times its length, both
    0.25 to 0.35 of the long arm wide, so that the foot always sticks out of the long arm.

    Args:
        random_generator (np.random.Generator): The random generator drawing the foot and the width of the arms.

    Returns:
        np.ndarray: The [6, 2] vertices, centred, the long arm along the y axis and the foot at its bottom.
    """
    foot_ratio = random_generator.uniform(low=L_SHAPE_FOOT_RATIO_RANGE[0], high=L_SHAPE_FOOT_RATIO_RANGE[1])
    arm_width = random_generator.uniform(low=L_SHAPE_ARM_WIDTH_RATIO_RANGE[0], high=L_SHAPE_ARM_WIDTH_RATIO_RANGE[1])

    # Outline going down the long arm and back along the foot
    return centre_vertices(vertices=np.array([[0.0, 0.0], [arm_width, 0.0], [arm_width, 1.0 - arm_width],
                                              [foot_ratio, 1.0 - arm_width], [foot_ratio, 1.0], [0.0, 1.0]]))


def draw_l_shape_mask(mask: np.ndarray, center_x: int, center_y: int, size: int,
                      random_generator: np.random.Generator, upright: bool = False) -> np.ndarray:
    """
    Draws a filled L-shape inscribed in a circle of diameter 'size', at a random rotation and mirrored for half of
    them (a J), on a binary mask. One of the synthetic shape classes.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The diameter of the circle its farthest vertices lie on, in pixels.
        random_generator (np.random.Generator): The random generator drawing the arms and the rotation.
        upright (bool): True for a vertical long arm whose foot points to the right, without mirror nor rotation
            (keypoint images). Defaults to False.

    Returns:
        np.ndarray: The [6, 2] float vertices, from the outer top of the long arm along the outline, in the order of
            KEYPOINT_NAMES.
    """
    return fill_rotated_polygon(mask=mask, center_x=center_x, center_y=center_y, size=size,
                                vertices=build_l_shape_vertices(random_generator=random_generator),
                                random_generator=random_generator, upright=upright)


def draw_hollow_mask(mask: np.ndarray, center_x: int, center_y: int, size: int, random_generator: np.random.Generator,
                     filled_drawing_function: Callable[..., np.ndarray], upright: bool = False,
                     outer_keypoints: bool = True, inner_keypoints: bool = False) -> np.ndarray:
    """
    Draws the hollow version of a filled shape class on a binary mask: the filled shape, out of which the same outline
    scaled down to 0.4 to 0.6 of its size around its centroid is cut, a hole showing the background (or the objects
    below it) that is left out of its instance mask. Shared by the fifteen hollow classes of HOLLOW_SHAPE_NAMES, which
    differ from their filled class by the hole only, so that a model must look inside the outline to tell them apart.
    Every filled shape of HOLLOW_SHAPE_NAMES being convex, the scaled outline stays inside it; the hole is further
    kept HOLLOW_MINIMUM_WALL_PIXELS away from the outside, and only its largest piece is cut, so that the shape is one
    ring around one hole. A shape leaving no hole is drawn again, up to HOLLOW_MAXIMUM_ATTEMPTS times. The keypoints
    of the keypoint images are the outer vertices of the filled shape, followed for the hollow polygons by the corners
    of the hole, the outer vertices scaled like the hole, and none for the distractors.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The size of the filled shape, as its drawing function reads it.
        random_generator (np.random.Generator): The random generator of the filled shape and of the size of the hole.
        filled_drawing_function (Callable[..., np.ndarray]): The drawing function of the filled shape, e.g.
            draw_square_mask.
        upright (bool): True for the upright filled shape of the keypoint images. Defaults to False.
        outer_keypoints (bool): Whether to return the keypoints of the filled shape, False for the distractors of the
            keypoint images (HOLLOW_DISTRACTOR_SHAPE_NAMES). Defaults to True.
        inner_keypoints (bool): Whether to add the corners of the hole after the outer keypoints, for the hollow
            polygons (HOLLOW_INNER_KEYPOINT_SHAPE_NAMES). Defaults to False.

    Returns:
        np.ndarray: The [K, 2] float keypoints in the order of KEYPOINT_NAMES and in the continuous coordinates of the
            mask: the K outer vertices of the filled shape, the [2K, 2] outer vertices then the corners of the hole
            with inner_keypoints, or a [0, 2] array without outer_keypoints.

    Raises:
        ValueError: If inner keypoints are asked without the outer ones, or if no drawn shape leaves a hole inside its
            walls, the size being too small for the shape.
    """
    if inner_keypoints and not outer_keypoints:
        raise ValueError("Expected the inner keypoints of a hollow shape to follow its outer keypoints, but "
                         "inner_keypoints is True while outer_keypoints is False.")
    wall_kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(2 * HOLLOW_MINIMUM_WALL_PIXELS + 1,
                                                                             2 * HOLLOW_MINIMUM_WALL_PIXELS + 1))
    for _ in range(HOLLOW_MAXIMUM_ATTEMPTS):
        # Filled shape on a canvas of its own, so that the hole only cuts this shape
        filled_mask = np.zeros_like(mask)
        keypoints = filled_drawing_function(mask=filled_mask, center_x=center_x, center_y=center_y, size=size,
                                            random_generator=random_generator, upright=upright)

        # Same outline scaled down around the centroid of its pixels, kept inside the minimum wall of the filled shape
        inner_size_ratio = random_generator.uniform(low=HOLLOW_INNER_SIZE_RATIO_RANGE[0],
                                                    high=HOLLOW_INNER_SIZE_RATIO_RANGE[1])
        row_indices, column_indices = np.nonzero(filled_mask)
        centroid_x, centroid_y = column_indices.mean(), row_indices.mean()
        scaling_matrix = np.array([[inner_size_ratio, 0.0, centroid_x * (1.0 - inner_size_ratio)],
                                   [0.0, inner_size_ratio, centroid_y * (1.0 - inner_size_ratio)]])
        hole_mask = cv2.warpAffine(src=filled_mask, M=scaling_matrix, dsize=(mask.shape[1], mask.shape[0]),
                                   flags=cv2.INTER_NEAREST)
        hole_mask = (hole_mask > 0) & (cv2.erode(src=filled_mask, kernel=wall_kernel) > 0)

        # Largest 4-connected piece of the hole only, the pieces cut off at the acute corners being left filled
        number_hole_groups, hole_groups, hole_statistics, _ = cv2.connectedComponentsWithStats(
            image=hole_mask.astype(np.uint8), connectivity=4)
        if number_hole_groups < 2:
            continue
        largest_hole_group = 1 + int(np.argmax(hole_statistics[1:, cv2.CC_STAT_AREA]))
        mask[(filled_mask > 0) & (hole_groups != largest_hole_group)] = 255

        # Distractors have no keypoints; the corners of the hole are the outer vertices scaled like the hole, around
        # the centroid in the continuous coordinates (pixel centres at + 0.5)
        if not outer_keypoints:
            return np.zeros(shape=(0, 2), dtype=np.float64)
        if inner_keypoints:
            continuous_centroid = np.array([centroid_x + 0.5, centroid_y + 0.5])
            hole_corners = continuous_centroid + inner_size_ratio * (keypoints - continuous_centroid)
            return np.concatenate([keypoints, hole_corners], axis=0)
        return keypoints
    raise ValueError(f"Expected a hollow shape of size {size} to leave a hole inside walls of "
                     f"{HOLLOW_MINIMUM_WALL_PIXELS} pixels, but {HOLLOW_MAXIMUM_ATTEMPTS} drawn shapes left none.")


def build_house_vertices(random_generator: np.random.Generator) -> np.ndarray:
    """
    Builds the vertices of a house for draw_house_mask, a keypoint category drawn with the shapes of the keypoint
    images: walls under a triangular roof as wide as them, drawn as one pentagon whose walls, unlike those of the
    regular pentagons, are vertical.

    Args:
        random_generator (np.random.Generator): The random generator drawing the width of the walls and the height of
            the roof.

    Returns:
        np.ndarray: The [5, 2] vertices, centred, from the apex of the roof clockwise, the house going from -1 at the
            apex to 1 at its base along the y axis.
    """
    half_width = random_generator.uniform(low=HOUSE_HALF_WIDTH_RATIO_RANGE[0], high=HOUSE_HALF_WIDTH_RATIO_RANGE[1])
    roof_height = random_generator.uniform(low=HOUSE_ROOF_HEIGHT_RATIO_RANGE[0], high=HOUSE_ROOF_HEIGHT_RATIO_RANGE[1])

    # Height of the eaves, where the roof meets the walls
    eave_height = -1.0 + 2.0 * roof_height
    return np.array([[0.0, -1.0], [half_width, eave_height], [half_width, 1.0], [-half_width, 1.0],
                     [-half_width, eave_height]])


def draw_house_mask(mask: np.ndarray, center_x: int, center_y: int, size: int,
                    random_generator: np.random.Generator, upright: bool = False) -> np.ndarray:
    """
    Draws a filled house inscribed in a circle of diameter 'size' on a binary mask, a keypoint category drawn with the
    shapes of the keypoint images, at a random rotation and mirrored for half of them when not upright.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The diameter of the circle its farthest vertices lie on, in pixels.
        random_generator (np.random.Generator): The random generator drawing the walls, the roof and the rotation.
        upright (bool): True for a standing house, without mirror nor rotation (keypoint images). Defaults to False.

    Returns:
        np.ndarray: The [5, 2] float vertices in the order of KEYPOINT_NAMES: the apex, the left and right ends of
            the roof, then the left and right ends of the base when upright.
    """
    vertices = fill_rotated_polygon(mask=mask, center_x=center_x, center_y=center_y, size=size,
                                    vertices=build_house_vertices(random_generator=random_generator),
                                    random_generator=random_generator, upright=upright)
    return vertices[[0, 4, 1, 3, 2]]


def get_limb_direction(angle_degrees: float, side_sign: float) -> np.ndarray:
    """
    Computes the unit direction of a segment of a stick figure limb for build_stick_figure_joints, from its angle
    measured from straight down towards the side of the limb.

    Args:
        angle_degrees (float): The angle of the segment in degrees, 0 pointing down and 90 pointing to the side.
        side_sign (float): 1 for the limbs on the right of the image (the left limbs of the figure facing the viewer),
            -1 for those on its left.

    Returns:
        np.ndarray: The [2] unit direction, the y axis pointing down.
    """
    return np.array([side_sign * np.sin(np.radians(angle_degrees)), np.cos(np.radians(angle_degrees))])


def build_stick_figure_joints(random_generator: np.random.Generator) -> np.ndarray:
    """
    Builds the joints of a stick figure for draw_stick_figure_mask, the keypoint category with the most keypoints and
    the only one whose keypoints are inside its outline: a torso one unit long under the head, and two arms and two
    legs of two segments each, posed at random angles that keep every limb on its own side of the torso.

    Args:
        random_generator (np.random.Generator): The random generator drawing the angles of the limbs.

    Returns:
        np.ndarray: The [11, 2] joints in the order of KEYPOINT_NAMES["stick_figure"], in torso lengths, the neck at
            the origin and the y axis pointing down.
    """
    joints = {"head": np.array([0.0, -STICK_FIGURE_HEAD_RADIUS]), "neck": np.array([0.0, 0.0]),
              "pelvis": np.array([0.0, 1.0])}

    # Two segments per limb and side, the first turned from straight down towards the side of the limb, the second
    # bent from it within limits that keep it on the same side. The left limbs of the figure, which faces the viewer,
    # are on the right of the image
    for limb in STICK_FIGURE_LIMBS.values():
        for side_name, side_sign in (("left", 1.0), ("right", -1.0)):
            first_angle = random_generator.uniform(low=limb["first_angle_range"][0], high=limb["first_angle_range"][1])
            bend = random_generator.uniform(low=limb["bend_range"][0], high=limb["bend_range"][1])
            second_angle = float(np.clip(first_angle + bend, limb["second_angle_limits"][0],
                                         limb["second_angle_limits"][1]))
            middle_joint = joints[limb["root"]] + limb["lengths"][0] * get_limb_direction(angle_degrees=first_angle,
                                                                                          side_sign=side_sign)
            joints[f"{side_name}_{limb['joints'][0]}"] = middle_joint
            joints[f"{side_name}_{limb['joints'][1]}"] = middle_joint + limb["lengths"][1] * get_limb_direction(
                angle_degrees=second_angle, side_sign=side_sign)
    return np.stack([joints[name] for name in KEYPOINT_NAMES["stick_figure"]])


def draw_stick_figure_mask(mask: np.ndarray, center_x: int, center_y: int, size: int,
                           random_generator: np.random.Generator, upright: bool = False) -> np.ndarray:
    """
    Draws a standing stick figure fitting a 'size' x 'size' square on a mask, a keypoint category drawn with the
    shapes of the keypoint images: a head disc, a torso, and two arms and two legs in a random pose. The limbs of each
    side get the mask value of STICK_FIGURE_SIDE_VALUES, which draw_shape_descriptor paints in a colour of its own so
    that the left and right of the figure can be told apart, the head and the torso the value 255 of every shape.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask drawn in place.
        center_x (int): The horizontal centre of the shape in pixels.
        center_y (int): The vertical centre of the shape in pixels.
        size (int): The side of the square the figure fits in, in pixels.
        random_generator (np.random.Generator): The random generator drawing the pose.
        upright (bool): Unused, every stick figure standing, kept so that every shape shares the same signature.
            Defaults to False.

    Returns:
        np.ndarray: The [11, 2] float joints in the order of KEYPOINT_NAMES["stick_figure"], the centres of their
            pixels in the continuous coordinates of the mask.
    """
    names = KEYPOINT_NAMES["stick_figure"]
    joints = build_stick_figure_joints(random_generator=random_generator)
    thickness = max(2, int(round(STICK_FIGURE_THICKNESS_RATIO * size)))

    # Scale and centre the figure so that its joints and its head disc, widened by the lines, fit the square
    head = joints[names.index("head")]
    extent_points = np.concatenate([joints, np.stack([head - STICK_FIGURE_HEAD_RADIUS,
                                                      head + STICK_FIGURE_HEAD_RADIUS])])
    minimum_corner, maximum_corner = extent_points.min(axis=0), extent_points.max(axis=0)
    scale = (size - thickness) / (maximum_corner - minimum_corner).max()
    points = np.round((joints - (minimum_corner + maximum_corner) / 2) * scale + np.array([center_x, center_y]))
    points = points.astype(np.int32)

    # Limbs of each side in the mask value of the side, then the torso and the head over them
    for limb in STICK_FIGURE_LIMBS.values():
        for side_name, side_value in STICK_FIGURE_SIDE_VALUES.items():
            limb_points = points[[names.index(limb["root"]), names.index(f"{side_name}_{limb['joints'][0]}"),
                                  names.index(f"{side_name}_{limb['joints'][1]}")]]
            cv2.polylines(img=mask, pts=[limb_points.reshape(-1, 1, 2)], isClosed=False, color=side_value,
                          thickness=thickness)
    neck, pelvis, head = points[names.index("neck")], points[names.index("pelvis")], points[names.index("head")]
    cv2.line(img=mask, pt1=(int(neck[0]), int(neck[1])), pt2=(int(pelvis[0]), int(pelvis[1])), color=255,
             thickness=thickness)
    cv2.circle(img=mask, center=(int(head[0]), int(head[1])),
               radius=max(1, int(round(STICK_FIGURE_HEAD_RADIUS * scale))), color=255, thickness=-1)
    return points + 0.5


# Drawing function of every shape class
SHAPE_DRAWING_FUNCTIONS = {"square": draw_square_mask, "circle": draw_circle_mask, "triangle": draw_triangle_mask,
                           "rectangle": draw_rectangle_mask, "pentagon": draw_pentagon_mask,
                           "hexagon": draw_hexagon_mask, "plus_sign": draw_plus_sign_mask, "star": draw_star_mask,
                           "octagon": draw_octagon_mask, "rhombus": draw_rhombus_mask,
                           "parallelogram": draw_parallelogram_mask, "trapezoid": draw_trapezoid_mask,
                           "kite": draw_kite_mask, "heptagon": draw_heptagon_mask, "nonagon": draw_nonagon_mask,
                           "ellipse": draw_ellipse_mask, "semicircle": draw_semicircle_mask,
                           "annulus": draw_annulus_mask, "capsule": draw_capsule_mask, "l_shape": draw_l_shape_mask}

# Hollow shapes, drawn by draw_hollow_mask around the drawing function of their filled shape, with the corners of
# their hole for the hollow polygons and without keypoints for the distractors
SHAPE_DRAWING_FUNCTIONS.update({
    f"hollow_{shape_name}": partial(draw_hollow_mask, filled_drawing_function=SHAPE_DRAWING_FUNCTIONS[shape_name],
                                    outer_keypoints=shape_name not in HOLLOW_DISTRACTOR_SHAPE_NAMES,
                                    inner_keypoints=shape_name in HOLLOW_INNER_KEYPOINT_SHAPE_NAMES)
    for shape_name in HOLLOW_SHAPE_NAMES})

# Drawing function of every keypoint category, the shape classes followed by the stick figure and the house
KEYPOINT_DRAWING_FUNCTIONS = {**SHAPE_DRAWING_FUNCTIONS, "stick_figure": draw_stick_figure_mask,
                              "house": draw_house_mask}


def get_mask_bounding_box(mask: np.ndarray) -> List[int]:
    """
    Computes the tight [x1, y1, x2, y2] box of the drawn pixels of a mask, with exclusive x2 / y2 so that the box
    width and height equal the number of covered columns and rows. Boxes computed from the masks are pixel exact,
    whatever the rasterisation rules of OpenCV.

    Args:
        mask (np.ndarray): The [H, W] mask of a single shape.

    Returns:
        List[int]: The tight bounding box of the shape.
    """
    row_indices, column_indices = np.nonzero(mask)
    return [int(column_indices.min()), int(row_indices.min()), int(column_indices.max()) + 1,
            int(row_indices.max()) + 1]


def get_intersection_over_union(first_box: List[int], second_box: List[int]) -> float:
    """
    Computes the intersection over union of two [x1, y1, x2, y2] boxes, used to limit how much two synthetic objects
    of the same image may overlap.

    Args:
        first_box (List[int]): The first box.
        second_box (List[int]): The second box.

    Returns:
        float: The intersection over union, between 0 and 1.
    """
    intersection_width = max(0, min(first_box[2], second_box[2]) - max(first_box[0], second_box[0]))
    intersection_height = max(0, min(first_box[3], second_box[3]) - max(first_box[1], second_box[1]))
    intersection = intersection_width * intersection_height
    first_area = (first_box[2] - first_box[0]) * (first_box[3] - first_box[1])
    second_area = (second_box[2] - second_box[0]) * (second_box[3] - second_box[1])
    return intersection / (first_area + second_area - intersection)


def draw_contrasting_colour(background_colour: np.ndarray, random_generator: np.random.Generator,
                            avoided_colours: Tuple[np.ndarray, ...] = ()) -> np.ndarray:
    """
    Draws a random BGR colour that differs from the background by at least 60 levels on one channel, so that every
    synthetic object stays clearly visible, and as much from every avoided colour, so that the parts of a stick figure
    drawn in colours of their own are told apart from its body and from each other.

    Args:
        background_colour (np.ndarray): The [3] uint8 BGR background colour.
        random_generator (np.random.Generator): The random generator drawing the colour.
        avoided_colours (Tuple[np.ndarray, ...]): The [3] uint8 BGR colours the drawn colour must differ from as well.
            Defaults to none.

    Returns:
        np.ndarray: The [3] uint8 BGR object colour.
    """
    while True:
        colour = random_generator.integers(low=0, high=256, size=3, dtype=np.int64)
        if all(np.abs(colour - reference_colour.astype(np.int64)).max() >= MINIMUM_COLOUR_DIFFERENCE
               for reference_colour in (background_colour, *avoided_colours)):
            return colour.astype(np.uint8)


def draw_background(image_size: int, random_generator: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draws the background of a synthetic image or sequence: black for half of them, a random colour for the other half.

    Args:
        image_size (int): The height and width of the square image in pixels.
        random_generator (np.random.Generator): The random generator drawing the colour.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The [H, W, 3] uint8 BGR background image and its [3] uint8 colour.
    """
    if random_generator.random() < 0.5:
        background_colour = np.zeros(shape=3, dtype=np.uint8)
    else:
        background_colour = random_generator.integers(low=0, high=256, size=3, dtype=np.uint8)
    image = np.empty(shape=(image_size, image_size, 3), dtype=np.uint8)
    image[:, :] = background_colour
    return image, background_colour


def draw_shape_descriptor(minimum_size: int, maximum_size: int, background_colour: np.ndarray,
                          random_generator: np.random.Generator, keypoints: bool = False,
                          shape_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Draws one synthetic object of random class (or of the given class), size and colour, its pixels drawn once on a
    canvas of their own so that the same object can be pasted at any position of an image or of the frames of a
    sequence. A hollow shape is drawn at HOLLOW_MINIMUM_SIZE at least, above maximum_size in the small images. Shared
    by the images of the detection, keypoint detection and classification datasets, the moving shape of the
    sequence classification dataset, the objects of the comparison pairs and the shapes of the sequence
    labelling frames, drawn again in every frame with their class kept. The objects of the keypoint images are drawn
    upright from the keypoint categories, with their keypoints, the limbs of each side of a stick figure getting a
    colour of their own.

    Args:
        minimum_size (int): The minimum size of the shape in pixels.
        maximum_size (int): The maximum size of the shape in pixels.
        background_colour (np.ndarray): The [3] uint8 BGR background colour the object must contrast with.
        random_generator (np.random.Generator): The random generator of the object.
        keypoints (bool): True for the objects of the keypoint images, drawn from KEYPOINT_CATEGORIES. Defaults to
            False.
        shape_name (Optional[str]): The class of the object, or None for a random class. Defaults to None.

    Returns:
        Dict[str, Any]: The 'shape_name' of the object, the [size + 1, size + 1] uint8 'mask' of its pixels (255, or
            a value of STICK_FIGURE_SIDE_VALUES for the limbs of a stick figure), its [3] uint8 BGR 'colour', the
            'part_colours' of the mask values other than 255 when it has some, and for the keypoint images its [K, 2]
            float 'keypoints' in the continuous coordinates of the canvas (pixel (x, y) covering [x, x + 1] x
            [y, y + 1]).
    """
    categories, drawing_functions = ((KEYPOINT_CATEGORIES, KEYPOINT_DRAWING_FUNCTIONS) if keypoints else
                                     (SHAPE_CATEGORIES, SHAPE_DRAWING_FUNCTIONS))
    shape_names = list(categories)
    if shape_name is None:
        shape_name = shape_names[int(random_generator.integers(low=0, high=len(shape_names)))]
    size = int(random_generator.integers(low=minimum_size, high=maximum_size + 1))

    # A hollow shape needs room for its hole inside its walls, its size being raised above the range when it is smaller
    if shape_name in HOLLOW_CLASS_NAMES:
        size = max(size, HOLLOW_MINIMUM_SIZE)

    # Pixels of the shape centred on its own canvas, then a colour visible on the background
    mask = np.zeros(shape=(size + 1, size + 1), dtype=np.uint8)
    shape_keypoints = drawing_functions[shape_name](mask=mask, center_x=size // 2, center_y=size // 2, size=size,
                                                    random_generator=random_generator, upright=keypoints)
    colour = draw_contrasting_colour(background_colour=background_colour, random_generator=random_generator)
    shape_descriptor = {"shape_name": shape_name, "mask": mask, "colour": colour}

    # Parts of the mask drawn with values of their own (the limbs of each side of a stick figure) get colours that
    # differ from the background, from the shape and from each other
    part_values = [int(value) for value in np.unique(mask) if 0 < value < 255]
    if len(part_values) > 0:
        part_colours = {}
        for part_value in part_values:
            part_colours[part_value] = draw_contrasting_colour(background_colour=background_colour,
                                                               random_generator=random_generator,
                                                               avoided_colours=(colour, *part_colours.values()))
        shape_descriptor["part_colours"] = part_colours
    if keypoints:
        shape_descriptor["keypoints"] = shape_keypoints
    return shape_descriptor


def place_shape(image: np.ndarray, shape_descriptor: Dict[str, Any], placed_boxes: List[List[int]],
                maximum_intersection_over_union: float, minimum_gap: int, random_generator: np.random.Generator,
                class_map: Optional[np.ndarray], instance_map: Optional[np.ndarray] = None,
                maximum_outside_fraction: float = 0.0) -> Optional[List[int]]:
    """
    Pastes a synthetic object at a random position of an image where it overlaps every object already placed by at
    most 'maximum_intersection_over_union', its box being grown by 'minimum_gap' pixels on every side for that test,
    trying up to 50 positions. Shared by the detection, keypoint detection and segmentation images, which tolerate
    some overlap between objects, and the frames of the comparison pairs, whose objects must not touch so that every
    object stays countable. For the segmentation images, the category of the object is painted at the same pixels of
    the class map, so that a later object covering an earlier one covers it in the class map too, and for the instance
    segmentation and keypoint images its index (one more than the number of placed boxes) at the same pixels of the
    instance map, which then gives the visible part of every object. The canvas of a keypoint image object may stick
    out of the image, so that some of its keypoints fall outside it.

    Args:
        image (np.ndarray): The [H, W, 3] uint8 BGR image drawn in place.
        shape_descriptor (Dict[str, Any]): The 'shape_name', 'mask', 'colour' and optional 'part_colours' of the
            object (draw_shape_descriptor).
        placed_boxes (List[List[int]]): The tight [x1, y1, x2, y2] boxes, inside the image, of the objects already in
            the image.
        maximum_intersection_over_union (float): The largest overlap allowed with every placed box.
        minimum_gap (int): The number of pixels the box of the object is grown by on every side for the overlap test.
        random_generator (np.random.Generator): The random generator of the positions.
        class_map (Optional[np.ndarray]): The [H, W] uint8 category identifier of every pixel, painted in place, or
            None for the tasks without masks.
        instance_map (Optional[np.ndarray]): The [H, W] uint16 object index of every pixel (0 for the background),
            painted in place, or None for the tasks without instance masks. Defaults to None.
        maximum_outside_fraction (float): The largest part of the canvas width and height that may stick out of the
            image on each side, 0 keeping the whole canvas inside it. Defaults to 0.

    Returns:
        Optional[List[int]]: The tight [x1, y1, x2, y2] box of the pasted object, which sticks out of the image with
            its canvas, or None when no position fits.
    """
    mask = shape_descriptor["mask"]
    canvas_height, canvas_width = mask.shape
    image_height, image_width = image.shape[:2]
    shape_box = get_mask_bounding_box(mask=mask)

    # Columns and rows of the canvas that may stick out of the image on every side
    margin_x = int(round(maximum_outside_fraction * canvas_width))
    margin_y = int(round(maximum_outside_fraction * canvas_height))

    # Try random positions of the shape canvas until the grown box of the part of the shape inside the image is far
    # enough from every placed box
    for _ in range(MAXIMUM_PLACEMENT_ATTEMPTS):
        left = int(random_generator.integers(low=-margin_x, high=image_width - canvas_width + margin_x + 1))
        top = int(random_generator.integers(low=-margin_y, high=image_height - canvas_height + margin_y + 1))
        box = [left + shape_box[0], top + shape_box[1], left + shape_box[2], top + shape_box[3]]
        clipped_box = [max(0, box[0]), max(0, box[1]), min(image_width, box[2]), min(image_height, box[3])]
        if clipped_box[2] <= clipped_box[0] or clipped_box[3] <= clipped_box[1]:
            continue
        grown_box = [clipped_box[0] - minimum_gap, clipped_box[1] - minimum_gap, clipped_box[2] + minimum_gap,
                     clipped_box[3] + minimum_gap]
        if all(get_intersection_over_union(first_box=grown_box, second_box=placed_box) <=
               maximum_intersection_over_union for placed_box in placed_boxes):
            # Window of the canvas inside the image, and the same window of the image
            canvas_left, canvas_top = max(0, -left), max(0, -top)
            canvas_right, canvas_bottom = min(canvas_width, image_width - left), min(canvas_height, image_height - top)
            image_window = (slice(top + canvas_top, top + canvas_bottom), slice(left + canvas_left, left + canvas_right))
            mask_window = mask[canvas_top:canvas_bottom, canvas_left:canvas_right]

            # Colour of the shape, then the colours of its parts
            image[image_window][mask_window > 0] = shape_descriptor["colour"]
            if "part_colours" in shape_descriptor:
                for part_value, part_colour in shape_descriptor["part_colours"].items():
                    image[image_window][mask_window == part_value] = part_colour
            if class_map is not None:
                class_map[image_window][mask_window > 0] = KEYPOINT_CATEGORIES[shape_descriptor["shape_name"]]
            if instance_map is not None:
                instance_map[image_window][mask_window > 0] = len(placed_boxes) + 1
            return box
    return None


def generate_image(image_size: int, maximum_objects: int, random_generator: np.random.Generator,
                   keypoints: bool = False) -> Tuple[np.ndarray, List[Tuple[Any, ...]], np.ndarray, np.ndarray]:
    """
    Generates one synthetic detection, segmentation, keypoint detection or classification image: a random-colour or
    black background with 1 to 'maximum_objects' shapes of the SHAPE_CATEGORIES classes, of random colours and sizes
    from MINIMUM_SHAPE_SIZE (64) pixels to a third of the image side, overlapping each other by at most an intersection
    over union of 0.3, with the class map and the instance map of its pixels. On images too small for the minimum size
    (the 64 pixel images of the tests), every shape gets the largest size. The keypoint images draw at least
    KEYPOINT_MINIMUM_OBJECTS upright shapes of the KEYPOINT_CATEGORIES, which may stick out of the image, with their
    keypoints.

    Args:
        image_size (int): The height and width of the square image in pixels.
        maximum_objects (int): The maximum number of objects in the image.
        random_generator (np.random.Generator): The random generator of the image.
        keypoints (bool): True for a keypoint detection image. Defaults to False.

    Returns:
        Tuple[np.ndarray, List[Tuple[Any, ...]], np.ndarray, np.ndarray]: The [H, W, 3] uint8 BGR image, for every
            object its category identifier, its tight [x1, y1, x2, y2] box clipped to the image and, for the keypoint
            images, its [K, 3] float keypoints as the COCO [x, y, visibility] (compute_keypoint_visibilities), the
            [H, W] uint8 category identifier of every pixel (0 for the background), and the [H, W] uint16 index of the
            object of every pixel (object i of the list at i + 1, 0 for the background).
    """
    # Half of the images have a black background, the other half a random colour, every pixel being background
    image, background_colour = draw_background(image_size=image_size, random_generator=random_generator)
    class_map = np.zeros(shape=(image_size, image_size), dtype=np.uint8)
    instance_map = np.zeros(shape=(image_size, image_size), dtype=np.uint16)
    objects, box_centres, inner_keypoint_flags = [], [], []

    # Shapes up to a third of the image side, at least MINIMUM_SHAPE_SIZE pixels when the image is large enough
    maximum_size = max(33, image_size // 3)
    minimum_size = min(MINIMUM_SHAPE_SIZE, maximum_size)

    # Place every object at a random position, an object overlapping the placed ones too much at every attempt being
    # left out
    minimum_objects = min(KEYPOINT_MINIMUM_OBJECTS, maximum_objects) if keypoints else 1
    for _ in range(int(random_generator.integers(low=minimum_objects, high=maximum_objects + 1))):
        shape_descriptor = draw_shape_descriptor(minimum_size=minimum_size, maximum_size=maximum_size,
                                                 background_colour=background_colour,
                                                 random_generator=random_generator, keypoints=keypoints)
        box = place_shape(image=image, shape_descriptor=shape_descriptor,
                          placed_boxes=[placed_object[1] for placed_object in objects],
                          maximum_intersection_over_union=MAXIMUM_OBJECT_INTERSECTION_OVER_UNION, minimum_gap=0,
                          random_generator=random_generator, class_map=class_map, instance_map=instance_map,
                          maximum_outside_fraction=KEYPOINT_MAXIMUM_OUTSIDE_FRACTION if keypoints else 0.0)
        if box is None:
            continue

        # Box of the part of the shape inside the image, and keypoints moved from the canvas to the image
        category_identifier = KEYPOINT_CATEGORIES[shape_descriptor["shape_name"]]
        clipped_box = [max(0, box[0]), max(0, box[1]), min(image_size, box[2]), min(image_size, box[3])]
        if keypoints:
            shape_box = get_mask_bounding_box(mask=shape_descriptor["mask"])
            image_keypoints = shape_descriptor["keypoints"] + np.array([box[0] - shape_box[0], box[1] - shape_box[1]])
            objects.append((category_identifier, clipped_box, image_keypoints))
            box_centres.append(np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]))
            inner_keypoint_flags.append(np.array([name.startswith("inner_")
                                                  for name in KEYPOINT_NAMES[shape_descriptor["shape_name"]]],
                                                 dtype=bool))
        else:
            objects.append((category_identifier, clipped_box))

    # Visibility of the keypoints once every object is drawn, a keypoint being hidden by the objects drawn after it
    if keypoints:
        objects = [(category_identifier, box, compute_keypoint_visibilities(
            keypoints=object_keypoints, box_centre=box_centre, instance_map=instance_map, object_index=object_index,
            inner_keypoints=object_inner_keypoints))
                   for object_index, ((category_identifier, box, object_keypoints), box_centre, object_inner_keypoints)
                   in enumerate(zip(objects, box_centres, inner_keypoint_flags, strict=True))]
    return image, objects, class_map, instance_map


def compute_keypoint_visibilities(keypoints: np.ndarray, box_centre: np.ndarray, instance_map: np.ndarray,
                                  object_index: int, inner_keypoints: np.ndarray) -> np.ndarray:
    """
    Gives the keypoints of an object of a synthetic keypoint image their COCO visibility flag once every object of the
    image is drawn: 0 (not labelled, the coordinates set to 0) outside the image, 1 (labelled but hidden) when an
    object drawn after it covers it, and 2 (visible) otherwise. A keypoint on the outline of a shape lies on the edge
    of its pixels, so the pixel tested is the one a quarter of a pixel from the keypoint towards the centre of the box
    of the shape, the pixel of the shape for both the outer pixel corners and the pixel centres the drawing functions
    return. The corners of the hole of a hollow polygon test the pixel a quarter of a pixel away from that centre, the
    pixel of its wall rather than of its hole.

    Args:
        keypoints (np.ndarray): The [K, 2] float keypoints in the continuous coordinates of the image.
        box_centre (np.ndarray): The [2] centre of the box of the whole shape, which may stick out of the image.
        instance_map (np.ndarray): The [H, W] uint16 index of the object of every pixel (object i at i + 1, 0 for the
            background).
        object_index (int): The index of the object in drawing order.
        inner_keypoints (np.ndarray): The [K] bool flag of the keypoints at the corners of a hole (the inner_
            keypoints of the hollow polygons).

    Returns:
        np.ndarray: The [K, 3] float keypoints with their visibility, as the COCO [x, y, visibility].
    """
    image_height, image_width = instance_map.shape

    # Keypoints outside the image are not labelled, as in COCO
    inside_mask = ((keypoints[:, 0] >= 0) & (keypoints[:, 0] <= image_width) & (keypoints[:, 1] >= 0) &
                   (keypoints[:, 1] <= image_height))

    # Pixel of every keypoint a quarter of a pixel towards the centre of the shape, away from it for the corners of a
    # hole, inside the image
    shape_directions = np.where(inner_keypoints[:, np.newaxis], -1.0, 1.0) * np.sign(box_centre - keypoints)
    pixels = np.floor(keypoints + 0.25 * shape_directions).astype(np.int64)
    pixel_columns = np.clip(pixels[:, 0], a_min=0, a_max=image_width - 1)
    pixel_rows = np.clip(pixels[:, 1], a_min=0, a_max=image_height - 1)
    covered_mask = instance_map[pixel_rows, pixel_columns] > object_index + 1

    # Hidden keypoints keep their position, the keypoints outside the image are zeroed
    visibilities = np.where(inside_mask, np.where(covered_mask, 1.0, 2.0), 0.0)
    return np.concatenate([np.where(inside_mask[:, np.newaxis], keypoints, 0.0), visibilities[:, np.newaxis]], axis=1)


def keep_visible_objects(objects: List[Tuple[Any, ...]], instance_map: np.ndarray, keep_visible_parts: bool
                         ) -> Tuple[List[Tuple[Any, ...]], List[Dict[str, Any]]]:
    """
    Leaves out of the labels of a synthetic detection, instance segmentation or keypoint detection image the objects
    mostly covered by the objects drawn after them or outside the image, those with fewer than
    MINIMUM_VISIBLE_OBJECT_PIXELS visible pixels in the instance map: place_shape only limits the box overlap of the
    shapes, so a shape can be hidden while its box stays in the image. For the instance segmentation, every kept object
    gets the box and the run length encoding of its visible part; the detection and the keypoint detection keep the
    box of the whole shape, and the keypoints follow their object.

    Args:
        objects (List[Tuple[Any, ...]]): The category identifier, tight [x1, y1, x2, y2] box and, for the keypoint
            images, keypoints of every object, in drawing order (generate_image).
        instance_map (np.ndarray): The [H, W] uint16 index of the object of every pixel (object i of the list at
            i + 1, 0 for the background).
        keep_visible_parts (bool): True for the instance segmentation (boxes and encodings of the visible parts), False
            for the detection and the keypoint detection (boxes of the whole shapes).

    Returns:
        Tuple[List[Tuple[Any, ...]], List[Dict[str, Any]]]: The objects kept with their category identifier, box and
            keypoints, and the uncompressed run length encoding of the visible part of every kept object (empty for
            the detection and the keypoint detection).
    """
    kept_objects, segmentations = [], []
    for object_index, (category_identifier, box, *extra_fields) in enumerate(objects):
        # Pixels of the object left visible by the objects drawn after it
        visible_mask = instance_map == object_index + 1
        if int(visible_mask.sum()) < MINIMUM_VISIBLE_OBJECT_PIXELS:
            continue

        # Box and encoding of the visible part for the instance segmentation, box of the whole shape for the detection
        if keep_visible_parts:
            kept_objects.append((category_identifier, get_mask_bounding_box(mask=visible_mask), *extra_fields))
            segmentations.append(encode_run_length_encoding(mask=visible_mask))
        else:
            kept_objects.append((category_identifier, box, *extra_fields))
    return kept_objects, segmentations


def generate_image_chunk(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generates and saves one chunk of images of a split, run by a worker process, with the class mask of every image
    for the semantic segmentation task, the objects mostly covered by later objects left out for the detection,
    instance segmentation and keypoint detection tasks, the visible part of every object for the instance segmentation
    task and the keypoints of every object for the keypoint detection task. Each chunk draws its images with its own
    random generator, seeded from the operating system, so every generation creates a different dataset.

    Args:
        chunk (Dict[str, Any]): The chunk description: 'data_folder', 'masks_folder' (None without masks),
            'split_name', 'first_index', 'number_images', 'image_size', 'maximum_objects', 'remove_covered_objects'
            (whether the objects with fewer than MINIMUM_VISIBLE_OBJECT_PIXELS visible pixels are left out),
            'instance_segmentation' (whether the visible parts of the objects are kept), 'keypoints' (whether the
            images are keypoint detection images) and 'counted_shapes' (whether the images are self-supervised images,
            drawn like the sequence labelling frames).

    Returns:
        List[Dict[str, Any]]: For every image, its 'index', 'file_name' and list of (category, box) 'objects', the
            keypoint detection objects being (category, box, keypoints), for the semantic segmentation the
            'mask_category_ids' of the categories visible in its mask, and for the instance segmentation its
            'segmentations', the uncompressed run length encoding of the visible part of every object, the box of every
            object being the box of that part (see keep_visible_objects). The self-supervised images give the
            'shape_names' of their shapes instead of 'objects'.
    """
    random_generator = np.random.default_rng()
    image_records = []

    for image_index in range(chunk["first_index"], chunk["first_index"] + chunk["number_images"]):
        # Self-supervised images: drawn like a sequence labelling frame, recorded with the classes of their shapes
        if chunk["counted_shapes"]:
            image, shape_names = generate_counted_shapes_image(image_size=chunk["image_size"],
                                                               maximum_objects=chunk["maximum_objects"],
                                                               random_generator=random_generator)
            file_name = f"{chunk['split_name']}_{image_index:06d}.png"
            cv2.imwrite(filename=os.path.join(chunk["data_folder"], file_name), img=image)
            image_records.append({"index": image_index, "file_name": file_name, "shape_names": shape_names})
            continue

        image, objects, class_map, instance_map = generate_image(image_size=chunk["image_size"],
                                                                 maximum_objects=chunk["maximum_objects"],
                                                                 random_generator=random_generator,
                                                                 keypoints=chunk["keypoints"])
        file_name = f"{chunk['split_name']}_{image_index:06d}.png"
        cv2.imwrite(filename=os.path.join(chunk["data_folder"], file_name), img=image)

        image_record = {"index": image_index, "file_name": file_name, "objects": objects}

        # The mask of a segmentation image is saved under the same name in the masks folder, with the categories
        # visible in it, a shape hidden by the later shapes being absent from the mask
        if chunk["masks_folder"] is not None:
            cv2.imwrite(filename=os.path.join(chunk["masks_folder"], file_name), img=class_map)
            image_record["mask_category_ids"] = [int(value) for value in np.unique(class_map) if value > 0]

        # Detection, instance segmentation and keypoint detection: the objects hidden by the later objects are left
        # out, the instance segmentation keeping the visible part of every other object
        if chunk["remove_covered_objects"]:
            image_record["objects"], segmentations = keep_visible_objects(
                objects=objects, instance_map=instance_map, keep_visible_parts=chunk["instance_segmentation"])
            if chunk["instance_segmentation"]:
                image_record["segmentations"] = segmentations
        image_records.append(image_record)

    return image_records


def build_detection_labels(image_records: List[Dict[str, Any]], image_size: int, split_name: str,
                           categories: Dict[str, int]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Assembles the COCO annotation file of a synthetic object detection split, read by both detection loaders, and the
    boxes of the instance segmentation and keypoint detection labels.

    Args:
        image_records (List[Dict[str, Any]]): The 'index', 'file_name' and (category, box, ...) 'objects' of every
            image.
        image_size (int): The height and width of the images in pixels.
        split_name (str): The name of the split, written in the description.
        categories (Dict[str, int]): The category identifier of every shape name of the task (SHAPE_CATEGORIES, or
            KEYPOINT_CATEGORIES for the keypoint detection).

    Returns:
        Tuple[Dict[str, Any], Dict[str, int]]: The COCO labels and the number of objects of every shape class.
    """
    images, annotations = [], []
    class_counts = {shape_name: 0 for shape_name in categories}
    category_names = {category_identifier: name for name, category_identifier in categories.items()}

    # One image entry per image, one annotation per object with its [x, y, width, height] box
    for record in image_records:
        image_identifier = record["index"] + 1
        images.append({"id": image_identifier, "file_name": record["file_name"], "width": image_size,
                       "height": image_size})
        for category_identifier, (x_1, y_1, x_2, y_2), *_ in record["objects"]:
            annotations.append({"id": len(annotations) + 1, "image_id": image_identifier,
                                "category_id": category_identifier, "bbox": [x_1, y_1, x_2 - x_1, y_2 - y_1],
                                "area": (x_2 - x_1) * (y_2 - y_1), "iscrowd": 0})
            class_counts[category_names[category_identifier]] += 1

    labels = {"info": {"description": f"Synthetic shapes detection dataset, {split_name} split"},
              "images": images, "annotations": annotations,
              "categories": [{"id": category_identifier, "name": name, "supercategory": "shape"}
                             for name, category_identifier in categories.items()]}
    return labels, class_counts


def build_instance_segmentation_labels(image_records: List[Dict[str, Any]], image_size: int, split_name: str) -> Tuple[
        Dict[str, Any], Dict[str, int]]:
    """
    Assembles the COCO annotation file of a synthetic instance segmentation split, read by the COCO dataset with its
    instance masks: the detection labels with the 'segmentation' of every object, the uncompressed run length encoding
    of its visible part (holes included), and its number of visible pixels as 'area'.

    Args:
        image_records (List[Dict[str, Any]]): The 'index', 'file_name', (category, box) 'objects' and 'segmentations'
            of every image.
        image_size (int): The height and width of the images in pixels.
        split_name (str): The name of the split, written in the description.

    Returns:
        Tuple[Dict[str, Any], Dict[str, int]]: The COCO labels and the number of objects of every shape class.
    """
    labels, class_counts = build_detection_labels(image_records=image_records, image_size=image_size,
                                                  split_name=split_name, categories=SHAPE_CATEGORIES)
    labels["info"]["description"] = f"Synthetic shapes instance segmentation dataset, {split_name} split"

    # The annotations follow the objects of the records in order, each one getting the encoding of its object
    segmentations = [segmentation for record in image_records for segmentation in record["segmentations"]]
    for annotation, segmentation in zip(labels["annotations"], segmentations, strict=True):
        annotation["segmentation"] = segmentation
        annotation["area"] = int(sum(segmentation["counts"][1::2]))
    return labels, class_counts


def build_keypoint_skeleton(shape_name: str) -> List[List[int]]:
    """
    Builds the COCO 'skeleton' of a keypoint category, the pairs of keypoints the viewers link with a line: every
    keypoint of its outline to the next one, the last one closing the loop, then its other links.

    Args:
        shape_name (str): The name of the category, a key of KEYPOINT_NAMES.

    Returns:
        List[List[int]]: The pairs of keypoints, numbered from 1 in the order of KEYPOINT_NAMES as in COCO.
    """
    names = KEYPOINT_NAMES[shape_name]
    skeleton = []

    # Outline closed in a loop, then the links between the other keypoints
    if shape_name in KEYPOINT_OUTLINES:
        outline = KEYPOINT_OUTLINES[shape_name]
        skeleton += [[names.index(outline[index]) + 1, names.index(outline[(index + 1) % len(outline)]) + 1]
                     for index in range(len(outline))]
    if shape_name in KEYPOINT_LINKS:
        skeleton += [[names.index(first_name) + 1, names.index(second_name) + 1]
                     for first_name, second_name in KEYPOINT_LINKS[shape_name]]
    return skeleton


def build_keypoint_label_mapping() -> Dict[str, int]:
    """
    Builds the 'label_mapping.json' of the synthetic keypoint dataset, read by the 'multi_class' label mapping of
    keypoint_detection/configuration_synthetic_shapes.yaml: the categories with keypoints are numbered from 1 in the
    order of KEYPOINT_CATEGORIES, and the distractors without keypoints (the circle, the ellipse, the annulus and
    the hollow ellipse) are left out, so that the keypoint dataset maps them to the background, while they still
    appear in the images next to the other objects.

    Returns:
        Dict[str, int]: The class index of every category identifier with keypoints, as a string key.
    """
    keypoint_category_names = [name for name in KEYPOINT_CATEGORIES if KEYPOINT_NAMES[name]]
    return {str(KEYPOINT_CATEGORIES[name]): class_index + 1 for class_index, name in enumerate(keypoint_category_names)}


def build_keypoint_labels(image_records: List[Dict[str, Any]], image_size: int, split_name: str) -> Tuple[
        Dict[str, Any], Dict[str, int]]:
    """
    Assembles the COCO annotation file of a synthetic keypoint detection split, read by the keypoint dataset: the
    detection labels of the KEYPOINT_CATEGORIES with the 'keypoints' of every object, flattened as the COCO
    [x1, y1, v1, x2, y2, v2, ...] in the continuous coordinates of the image, and their 'num_keypoints' labelled, and
    for every category the names of its keypoints and its skeleton.

    Args:
        image_records (List[Dict[str, Any]]): The 'index', 'file_name' and (category, box, keypoints) 'objects' of
            every image.
        image_size (int): The height and width of the images in pixels.
        split_name (str): The name of the split, written in the description.

    Returns:
        Tuple[Dict[str, Any], Dict[str, int]]: The COCO labels and the number of objects of every category.
    """
    labels, class_counts = build_detection_labels(image_records=image_records, image_size=image_size,
                                                  split_name=split_name, categories=KEYPOINT_CATEGORIES)
    labels["info"]["description"] = f"Synthetic shapes keypoint detection dataset, {split_name} split"

    # The annotations follow the objects of the records in order, each one getting the keypoints of its object
    object_keypoints = [keypoints for record in image_records for _, _, keypoints in record["objects"]]
    for annotation, keypoints in zip(labels["annotations"], object_keypoints, strict=True):
        annotation["keypoints"] = [value for x, y, visibility in keypoints
                                   for value in (round(float(x), 2), round(float(y), 2), int(visibility))]
        annotation["num_keypoints"] = int((keypoints[:, 2] > 0).sum())

    # Every category lists the names of its keypoints and its skeleton
    for category in labels["categories"]:
        category["keypoints"] = KEYPOINT_NAMES[category["name"]]
        category["skeleton"] = build_keypoint_skeleton(shape_name=category["name"])
    return labels, class_counts


def build_classification_labels(image_records: List[Dict[str, Any]], split_name: str) -> Tuple[
        Dict[str, Any], Dict[str, int]]:
    """
    Assembles the 'labels.json' manifest of a synthetic classification split, read by the classification loader:
    every image holds a single shape, whose category is the label of the image.

    Args:
        image_records (List[Dict[str, Any]]): The 'index', 'file_name' and single (category, box) 'objects' of every
            image.
        split_name (str): The name of the split, written in the description.

    Returns:
        Tuple[Dict[str, Any], Dict[str, int]]: The manifest and the number of images of every shape class.
    """
    class_counts = {shape_name: 0 for shape_name in SHAPE_CATEGORIES}
    category_names = {category_identifier: name for name, category_identifier in SHAPE_CATEGORIES.items()}

    # One sample per image, labelled with the category of its single shape
    samples = []
    for record in image_records:
        category_identifier = record["objects"][0][0]
        samples.append({"id": record["index"] + 1, "file_name": record["file_name"],
                        "category_id": category_identifier})
        class_counts[category_names[category_identifier]] += 1

    labels = {"info": {"description": f"Synthetic shapes classification dataset, {split_name} split"},
              "categories": [{"id": category_identifier, "name": name}
                             for name, category_identifier in SHAPE_CATEGORIES.items()],
              "samples": samples}
    return labels, class_counts


def build_shape_count_labels(image_records: List[Dict[str, Any]], split_name: str) -> Tuple[
        Dict[str, Any], Dict[str, int]]:
    """
    Assembles the 'labels.json' manifest of a synthetic self-supervised split, read by the classification loader of
    the self-supervised trainer: every image is labelled with its number of shapes, the class compared by the
    k-nearest neighbour evaluation of the test split, and keeps the categories of its shapes (SHAPE_CATEGORIES) in
    'shape_category_ids', for the evaluations that check which shapes the representation tells apart.

    Args:
        image_records (List[Dict[str, Any]]): The 'index', 'file_name' and 'shape_names' of every image.
        split_name (str): The name of the split, written in the description.

    Returns:
        Tuple[Dict[str, Any], Dict[str, int]]: The manifest and the number of images of every shape count.
    """
    class_counts = {category_name: 0 for category_name in SHAPE_COUNT_CATEGORIES}

    # One sample per image, labelled with its number of shapes
    samples = []
    for record in image_records:
        category_name = f"{len(record['shape_names'])}_shapes"
        samples.append({"id": record["index"] + 1, "file_name": record["file_name"],
                        "category_id": SHAPE_COUNT_CATEGORIES[category_name],
                        "shape_category_ids": [SHAPE_CATEGORIES[shape_name] for shape_name in record["shape_names"]]})
        class_counts[category_name] += 1

    labels = {"info": {"description": f"Synthetic counted shapes self-supervised dataset, {split_name} split"},
              "categories": [{"id": category_identifier, "name": name}
                             for name, category_identifier in SHAPE_COUNT_CATEGORIES.items()],
              "shape_categories": [{"id": category_identifier, "name": name}
                                   for name, category_identifier in SHAPE_CATEGORIES.items()],
              "samples": samples}
    return labels, class_counts


def build_segmentation_labels(image_records: List[Dict[str, Any]], split_name: str) -> Tuple[
        Dict[str, Any], Dict[str, int]]:
    """
    Assembles the 'labels.json' manifest of a synthetic semantic segmentation split, read by the segmentation loader:
    every image has its class mask under the same file name in the 'masks' folder, and the 'category_ids' of the
    categories visible in that mask, from which the record of the data of a run counts the masks of every class
    (data_record.py).

    Args:
        image_records (List[Dict[str, Any]]): The 'index', 'file_name', (category, box) 'objects' and
            'mask_category_ids' of every image.
        split_name (str): The name of the split, written in the description.

    Returns:
        Tuple[Dict[str, Any], Dict[str, int]]: The manifest and the number of objects of every shape class.
    """
    class_counts = {shape_name: 0 for shape_name in SHAPE_CATEGORIES}
    category_names = {category_identifier: name for name, category_identifier in SHAPE_CATEGORIES.items()}

    # One sample per image with its mask, the objects of every class being counted
    samples = []
    for record in image_records:
        samples.append({"id": record["index"] + 1, "file_name": record["file_name"],
                        "mask_file_name": record["file_name"], "category_ids": record["mask_category_ids"]})
        for category_identifier, _ in record["objects"]:
            class_counts[category_names[category_identifier]] += 1

    labels = {"info": {"description": f"Synthetic shapes segmentation dataset, {split_name} split"},
              "categories": [{"id": category_identifier, "name": name}
                             for name, category_identifier in SHAPE_CATEGORIES.items()],
              "samples": samples}
    return labels, class_counts


def generate_moving_shape_sequence(image_size: int, number_frames: int, random_generator: np.random.Generator) -> Tuple[
        List[np.ndarray], int]:
    """
    Generates one synthetic sequence classification sample: a single shape of random class, colour and size crossing
    the image in a straight line at constant speed, over a background shared by every frame. Its label is the
    direction of the movement, which no single frame shows (a frame only shows a shape at some position), so that it
    can only be learned from the order of the frames.

    Args:
        image_size (int): The height and width of the square frames in pixels.
        number_frames (int): The number of frames of the sequence.
        random_generator (np.random.Generator): The random generator of the sequence.

    Returns:
        Tuple[List[np.ndarray], int]: The [H, W, 3] uint8 BGR frames and the category identifier of the movement.
    """
    background, background_colour = draw_background(image_size=image_size, random_generator=random_generator)
    direction_names = list(MOVEMENT_CATEGORIES)
    direction_name = direction_names[int(random_generator.integers(low=0, high=len(direction_names)))]

    # Draw the shape once on its own canvas, its pixels then being moved from frame to frame
    shape_descriptor = draw_shape_descriptor(minimum_size=max(8, image_size // 8), maximum_size=max(9, image_size // 4),
                                             background_colour=background_colour, random_generator=random_generator)
    shape_mask, colour = shape_descriptor["mask"], shape_descriptor["colour"]
    size = shape_mask.shape[0] - 1

    # Straight path covering 30 to 60 % of the free span of the image along the movement axis, starting from the far
    # end for the leftward and upward movements
    step_x, step_y = MOVEMENT_DIRECTIONS[direction_name]
    free_span = image_size - size - 1
    travel = int(round(random_generator.uniform(low=0.3, high=0.6) * free_span))
    start_along = int(random_generator.integers(low=0, high=free_span - travel + 1)) + (travel if step_x + step_y < 0
                                                                                        else 0)
    start_across = int(random_generator.integers(low=0, high=free_span + 1))

    # Paste the shape at its position of every frame
    frames = []
    for frame_index in range(number_frames):
        offset_along = start_along + int(round((step_x + step_y) * travel * frame_index / max(1, number_frames - 1)))
        left, top = (offset_along, start_across) if step_x != 0 else (start_across, offset_along)
        frame = background.copy()
        frame[top:top + size + 1, left:left + size + 1][shape_mask > 0] = colour
        frames.append(frame)

    return frames, MOVEMENT_CATEGORIES[direction_name]


def get_labelling_shape_size_range(image_size: int) -> Tuple[int, int]:
    """
    Gives the sizes of the shapes of the sequence labelling frames and of the self-supervised images drawn like them,
    so that a backbone trained on the self-supervised images sees shapes of the sizes of the sequence labelling
    frames it then initialises: MINIMUM_SHAPE_SIZE pixels (an eighth of the image side when larger) to a fifth of the
    image side, every shape getting the largest size on images too small for the minimum.

    Args:
        image_size (int): The height and width of the square images in pixels.

    Returns:
        Tuple[int, int]: The smallest and the largest size of the shapes in pixels.
    """
    maximum_size = max(9, image_size // 5)
    minimum_size = min(max(image_size // 8, MINIMUM_SHAPE_SIZE), maximum_size)
    return minimum_size, maximum_size


def generate_counted_shapes_image(image_size: int, maximum_objects: int, random_generator: np.random.Generator
                                  ) -> Tuple[np.ndarray, List[str]]:
    """
    Generates one image of the synthetic self-supervised dataset, drawn like a frame of a sequence labelling sample so
    that the backbone it trains can initialise the sequence labelling model: 5 to 'maximum_objects' shapes of random
    classes of SHAPE_CATEGORIES (several shapes may share a class), of random sizes, colours and rotations, placed
    without overlap on a black or random-colour background. The number of shapes labels the image for the k-nearest
    neighbour evaluation, and the classes of the shapes are kept with it.

    Args:
        image_size (int): The height and width of the square image in pixels.
        maximum_objects (int): The maximum number of shapes of the image, from 5 to COUNTED_SHAPES_MAXIMUM_OBJECTS.
        random_generator (np.random.Generator): The random generator of the image.

    Returns:
        Tuple[np.ndarray, List[str]]: The [H, W, 3] uint8 BGR image and the class of every shape.

    Raises:
        ValueError: If maximum_objects is outside 5 to COUNTED_SHAPES_MAXIMUM_OBJECTS.
    """
    if not LABELLING_MINIMUM_OBJECTS <= maximum_objects <= COUNTED_SHAPES_MAXIMUM_OBJECTS:
        raise ValueError(f"The self-supervised images hold {LABELLING_MINIMUM_OBJECTS} to "
                         f"{COUNTED_SHAPES_MAXIMUM_OBJECTS} shapes, got maximum_objects {maximum_objects}.")
    background, background_colour = draw_background(image_size=image_size, random_generator=random_generator)
    shape_names = list(SHAPE_CATEGORIES)
    minimum_size, maximum_size = get_labelling_shape_size_range(image_size=image_size)

    # Number and classes of the shapes, then every shape drawn and placed without overlap
    number_objects = int(random_generator.integers(low=LABELLING_MINIMUM_OBJECTS, high=maximum_objects + 1))
    image_shape_names = [shape_names[int(random_generator.integers(low=0, high=len(shape_names)))]
                         for _ in range(number_objects)]
    shape_descriptors = [draw_shape_descriptor(minimum_size=minimum_size, maximum_size=maximum_size,
                                               background_colour=background_colour, random_generator=random_generator,
                                               shape_name=shape_name)
                         for shape_name in image_shape_names]
    image = draw_shapes_frame(background=background, shape_descriptors=shape_descriptors,
                              random_generator=random_generator)
    return image, image_shape_names


def generate_shape_change_sequence(image_size: int, number_frames: int, maximum_objects: int,
                                   random_generator: np.random.Generator) -> Tuple[List[np.ndarray], List[int]]:
    """
    Generates one synthetic sequence labelling sample: frames of 5 to 'maximum_objects' shapes over a background shared
    by every frame, every frame from the second one adding a shape of random class (label 'appearing_<class>'),
    removing one of the shapes ('disappearing_<class>') or keeping them ('no_change', always the label of the first
    frame), with the same probability among the changes the number of shapes allows. Every frame draws all its shapes
    again with a new size, colour, rotation and position, without overlap, so that a change is told by the classes of
    the shapes only, compared with the previous frame. Several shapes of a frame may share a class. The shapes measure
    MINIMUM_SHAPE_SIZE pixels (an eighth of the frame side when larger) to a fifth of the frame side.

    Args:
        image_size (int): The height and width of the square frames in pixels.
        number_frames (int): The number of frames of the sequence, at least 2.
        maximum_objects (int): The maximum number of shapes of a frame, at least 5.
        random_generator (np.random.Generator): The random generator of the sequence.

    Returns:
        Tuple[List[np.ndarray], List[int]]: The [H, W, 3] uint8 BGR frames and the category identifier of every frame.

    Raises:
        ValueError: If maximum_objects is below 5.
    """
    if maximum_objects < LABELLING_MINIMUM_OBJECTS:
        raise ValueError(f"The sequence labelling frames need at least {LABELLING_MINIMUM_OBJECTS} shapes, got "
                         f"maximum_objects {maximum_objects}.")
    background, background_colour = draw_background(image_size=image_size, random_generator=random_generator)
    shape_names = list(SHAPE_CATEGORIES)

    # Size of the shapes, their smallest value raised to MINIMUM_SHAPE_SIZE within the largest one
    minimum_size, maximum_size = get_labelling_shape_size_range(image_size=image_size)

    # Classes of the shapes of the first frame
    number_objects = int(random_generator.integers(low=LABELLING_MINIMUM_OBJECTS, high=maximum_objects + 1))
    frame_shape_names = [shape_names[int(random_generator.integers(low=0, high=len(shape_names)))]
                         for _ in range(number_objects)]

    frames, frame_category_ids = [], []
    for frame_index in range(number_frames):
        # From the second frame, a change among those keeping the number of shapes within its bounds
        change_name = "no_change"
        if frame_index > 0:
            number_objects = len(frame_shape_names)
            allowed_changes = [change for change in LABELLING_CHANGES
                               if not (change == "appearing" and number_objects == maximum_objects)
                               and not (change == "disappearing" and number_objects == LABELLING_MINIMUM_OBJECTS)]
            change_name = allowed_changes[int(random_generator.integers(low=0, high=len(allowed_changes)))]

        # A shape of random class added, or one of the shapes removed, the label naming its class
        category_name = "no_change"
        if change_name == "appearing":
            changed_shape_name = shape_names[int(random_generator.integers(low=0, high=len(shape_names)))]
            frame_shape_names.append(changed_shape_name)
            category_name = f"appearing_{changed_shape_name}"
        elif change_name == "disappearing":
            changed_shape_name = frame_shape_names.pop(int(random_generator.integers(low=0,
                                                                                     high=len(frame_shape_names))))
            category_name = f"disappearing_{changed_shape_name}"

        # Every shape drawn again with a new size, colour and rotation, then placed without overlap
        shape_descriptors = [draw_shape_descriptor(minimum_size=minimum_size, maximum_size=maximum_size,
                                                   background_colour=background_colour,
                                                   random_generator=random_generator, shape_name=shape_name)
                             for shape_name in frame_shape_names]
        frames.append(draw_shapes_frame(background=background, shape_descriptors=shape_descriptors,
                                        random_generator=random_generator))
        frame_category_ids.append(CHANGE_CATEGORIES[category_name])

    return frames, frame_category_ids


def draw_selection_frame(background: np.ndarray, shape_name: str, size: int, colour: np.ndarray,
                         random_generator: np.random.Generator) -> Tuple[np.ndarray, int]:
    """
    Draws one frame of a synthetic frame selection sequence: a single shape of the given class and size at a random
    position of the background shared by the sequence, with the area of its pixels, which tells the largest copy of a
    shape apart from its smaller copies.

    Args:
        background (np.ndarray): The [H, W, 3] uint8 BGR background shared by the frames of the sequence.
        shape_name (str): The class of the shape, a key of SHAPE_DRAWING_FUNCTIONS.
        size (int): The size of the shape in pixels, smaller than the image side minus 2.
        colour (np.ndarray): The [3] uint8 BGR colour of the shape.
        random_generator (np.random.Generator): The random generator of the position and of the shape.

    Returns:
        Tuple[np.ndarray, int]: The [H, W, 3] uint8 BGR frame and the number of pixels of the shape.
    """
    image_size = background.shape[0]
    mask = np.zeros(shape=(image_size, image_size), dtype=np.uint8)
    center_x, center_y = random_generator.integers(low=size // 2 + 1, high=image_size - size // 2 - 1, size=2)
    SHAPE_DRAWING_FUNCTIONS[shape_name](mask=mask, center_x=int(center_x), center_y=int(center_y), size=size,
                                        random_generator=random_generator)
    frame = background.copy()
    frame[mask > 0] = colour
    return frame, int(np.count_nonzero(mask))


def generate_selection_sequence(image_size: int, number_frames: int,
                                random_generator: np.random.Generator) -> Tuple[List[np.ndarray], List[int]]:
    """
    Generates one synthetic frame selection sample: a sequence of frames showing one shape each, over a background and
    in a colour shared by its frames. Every wanted shape (circle, triangle, star) appears once at a large size, the
    frame to select for its class, and the other frames show a smaller copy of a wanted shape (at most 70 % of its
    size and 60 % of its area) or another shape (square, rectangle, plus sign), all unselected. Whether a circle is
    the one to select depends on the other circles of the sequence, so that the model must compare the frames. The
    frames come in random order.

    Args:
        image_size (int): The height and width of the square frames in pixels.
        number_frames (int): The number of frames of the sequence, at least the number of wanted shapes.
        random_generator (np.random.Generator): The random generator of the sequence.

    Returns:
        Tuple[List[np.ndarray], List[int]]: The [H, W, 3] uint8 BGR frames and the category identifier of every frame.
    """
    background, background_colour = draw_background(image_size=image_size, random_generator=random_generator)
    colour = draw_contrasting_colour(background_colour=background_colour, random_generator=random_generator)
    minimum_size = max(6, image_size // 8)
    largest_size_low, largest_size_high = [int(image_size * fraction) for fraction in SELECTION_LARGEST_SIZE_RANGE]
    frames, frame_category_ids, largest_copies = [], [], {}

    # Largest copy of every wanted shape, the frame to select for its class
    for shape_name in SELECTION_WANTED_SHAPES:
        size = int(random_generator.integers(low=largest_size_low, high=largest_size_high + 1))
        frame, area = draw_selection_frame(background=background, shape_name=shape_name, size=size, colour=colour,
                                           random_generator=random_generator)
        frames.append(frame)
        frame_category_ids.append(SELECTION_CATEGORIES[shape_name])
        largest_copies[shape_name] = (size, area)

    # Unselected frames: a clearly smaller copy of a wanted shape, redrawn until its area is small enough, or else
    # another shape
    for _ in range(number_frames - len(SELECTION_WANTED_SHAPES)):
        frame = None
        if random_generator.random() < SELECTION_SMALLER_COPY_PROBABILITY:
            shape_name = SELECTION_WANTED_SHAPES[int(random_generator.integers(low=0,
                                                                                high=len(SELECTION_WANTED_SHAPES)))]
            largest_size, largest_area = largest_copies[shape_name]
            maximum_size = max(minimum_size, int(largest_size * SELECTION_SMALLER_SIZE_RATIO))
            for _ in range(MAXIMUM_FRAME_ATTEMPTS):
                size = int(random_generator.integers(low=minimum_size, high=maximum_size + 1))
                candidate_frame, area = draw_selection_frame(background=background, shape_name=shape_name, size=size,
                                                             colour=colour, random_generator=random_generator)
                if area <= SELECTION_SMALLER_AREA_RATIO * largest_area:
                    frame = candidate_frame
                    break
        if frame is None:
            shape_name = SELECTION_OTHER_SHAPES[int(random_generator.integers(low=0,
                                                                               high=len(SELECTION_OTHER_SHAPES)))]
            size = int(random_generator.integers(low=minimum_size, high=largest_size_high + 1))
            frame, _ = draw_selection_frame(background=background, shape_name=shape_name, size=size, colour=colour,
                                            random_generator=random_generator)
        frames.append(frame)
        frame_category_ids.append(SELECTION_CATEGORIES["unselected"])

    # Frames in random order, the frames to select anywhere in the sequence
    frame_order = random_generator.permutation(number_frames).tolist()
    return [frames[frame_index] for frame_index in frame_order], [frame_category_ids[frame_index]
                                                                  for frame_index in frame_order]


def draw_shapes_frame(background: np.ndarray, shape_descriptors: List[Dict[str, Any]],
                      random_generator: np.random.Generator) -> np.ndarray:
    """
    Draws one frame of a member of a comparison pair or of a sequence labelling sample: every given object pasted at a
    random position of the background, without overlap and with a gap of 2 pixels between two objects, so that the
    frame shows exactly its number of objects. The whole frame is drawn again when an object does not fit, up to 20
    times.

    Args:
        background (np.ndarray): The [H, W, 3] uint8 BGR background shared by the frames of the pair or sequence.
        shape_descriptors (List[Dict[str, Any]]): The objects of the frame (draw_shape_descriptor).
        random_generator (np.random.Generator): The random generator of the positions.

    Returns:
        np.ndarray: The [H, W, 3] uint8 BGR frame.

    Raises:
        RuntimeError: If the objects never all fit, e.g. too many objects for the image size.
    """
    for _ in range(MAXIMUM_FRAME_ATTEMPTS):
        frame = background.copy()
        placed_boxes = []

        # Place the objects one after the other, starting the frame again as soon as one does not fit
        for shape_descriptor in shape_descriptors:
            box = place_shape(image=frame, shape_descriptor=shape_descriptor, placed_boxes=placed_boxes,
                              maximum_intersection_over_union=0.0, minimum_gap=MINIMUM_OBJECT_GAP,
                              random_generator=random_generator, class_map=None)
            if box is None:
                break
            placed_boxes.append(box)
        if len(placed_boxes) == len(shape_descriptors):
            return frame
    raise RuntimeError(f"{len(shape_descriptors)} objects never fitted without overlap in a {background.shape[0]} "
                       f"pixel frame after {MAXIMUM_FRAME_ATTEMPTS} attempts.")


def generate_object_loss_pair(image_size: int, minimum_frames: int, maximum_frames: int, maximum_objects: int,
                              random_generator: np.random.Generator) -> Tuple[Dict[str, List[np.ndarray]], int]:
    """
    Generates one synthetic sequence comparison sample: a before member and an after member, two sequences of
    independent views of the same objects (2 to 'maximum_objects' shapes of fixed class, size and colour, placed at
    random positions in every frame, over a background shared by the pair). The after member lost one of the objects
    (class 'loss_of_object') or shows all of them (class 'same_number_of_objects'), with the same probability. A single
    member does not give the class, only the comparison of the numbers of objects of the two members, and every frame
    shows exactly its number of objects. Each member draws its own number of frames, e.g. 5 before frames and 12
    after frames.

    Args:
        image_size (int): The height and width of the square frames in pixels.
        minimum_frames (int): The minimum number of frames of a member, at least 2.
        maximum_frames (int): The maximum number of frames of a member, at least minimum_frames.
        maximum_objects (int): The maximum number of objects of the before member, at least 2.
        random_generator (np.random.Generator): The random generator of the pair.

    Returns:
        Tuple[Dict[str, List[np.ndarray]], int]: The [H, W, 3] uint8 BGR frames of the 'before' and 'after' members,
            and the category identifier of the pair.

    Raises:
        ValueError: If maximum_objects is below 2.
    """
    if maximum_objects < PAIR_MINIMUM_OBJECTS:
        raise ValueError(f"The comparison pairs need at least {PAIR_MINIMUM_OBJECTS} objects in their before member, "
                         f"got maximum_objects {maximum_objects}.")
    background, background_colour = draw_background(image_size=image_size, random_generator=random_generator)

    # Objects of the before member, drawn once so that both members show the same objects
    number_objects = int(random_generator.integers(low=PAIR_MINIMUM_OBJECTS, high=maximum_objects + 1))
    before_descriptors = [draw_shape_descriptor(minimum_size=max(8, image_size // 8),
                                                maximum_size=max(9, image_size // 5),
                                                background_colour=background_colour,
                                                random_generator=random_generator) for _ in range(number_objects)]

    # The after member loses one of the objects for half of the pairs
    category_name = "loss_of_object" if random_generator.random() < 0.5 else "same_number_of_objects"
    after_descriptors = list(before_descriptors)
    if category_name == "loss_of_object":
        after_descriptors.pop(int(random_generator.integers(low=0, high=number_objects)))

    # Independent views of the objects of every member, each member with its own number of frames
    member_frames = {}
    for member_name, member_descriptors in zip(PAIR_MEMBERS, [before_descriptors, after_descriptors]):
        member_length = int(random_generator.integers(low=minimum_frames, high=maximum_frames + 1))
        member_frames[member_name] = [draw_shapes_frame(background=background, shape_descriptors=member_descriptors,
                                                        random_generator=random_generator)
                                      for _ in range(member_length)]
    return member_frames, PAIR_CATEGORIES[category_name]


def save_frames(data_folder: str, frame_folder: str, frames: List[np.ndarray]) -> List[str]:
    """
    Saves the frames of a sequence, or of a member of a pair, in order in their own folder
    '<data folder>/<frame folder>/', for the manifests of every sequence task.

    Args:
        data_folder (str): The data folder of the split.
        frame_folder (str): The folder of the frames, relative to the data folder.
        frames (List[np.ndarray]): The [H, W, 3] uint8 BGR frames, in order.

    Returns:
        List[str]: The paths of the saved frames, relative to the data folder.
    """
    os.makedirs(name=os.path.join(data_folder, frame_folder), exist_ok=True)
    frame_files = [f"{frame_folder}/{frame_index:06d}.png" for frame_index in range(len(frames))]
    for frame_file, frame in zip(frame_files, frames):
        cv2.imwrite(filename=os.path.join(data_folder, frame_file), img=frame)
    return frame_files


def generate_sequence_chunk(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generates and saves one chunk of sequences (or pairs of sequences) of a split, run by a worker process, the frames
    of every sequence in its own folder '<data folder>/<split>_<index>/', and those of a pair in its member folders
    '<data folder>/<split>_<index>/before/' and '.../after/'. Every sequence, and every member of a pair, has its own
    number of frames between 'minimum_frames' and 'maximum_frames'. Each chunk draws its samples with its own random
    generator, seeded from the operating system, so every generation creates a different dataset.

    Args:
        chunk (Dict[str, Any]): The chunk description: 'data_folder', 'split_name', 'first_index',
            'number_sequences', 'image_size', 'minimum_frames', 'maximum_frames', 'maximum_objects' and 'task'.

    Returns:
        List[Dict[str, Any]]: For every sample, its 'index', 'name', its 'frames' (or the frames of its 'before' and
            'after' members, paths relative to the data folder) and 'category_id' (sequence classification and
            comparison) or 'frame_category_ids' (sequence labelling and frame selection).
    """
    random_generator = np.random.default_rng()
    sequence_records = []

    for sequence_index in range(chunk["first_index"], chunk["first_index"] + chunk["number_sequences"]):
        sequence_name = f"{chunk['split_name']}_{sequence_index:06d}"

        # Number of frames of the sequence, the members of a pair drawing their own
        number_frames = int(random_generator.integers(low=chunk["minimum_frames"], high=chunk["maximum_frames"] + 1))

        # A shape crossing the frames labelled with its direction, shapes appearing and disappearing labelled per frame,
        # two members labelled with the comparison of their numbers of objects, or shapes of several sizes whose largest
        # copies are selected, the frames saved in their folders
        if chunk["task"] == "sequence_comparison":
            member_frames, category_identifier = generate_object_loss_pair(
                image_size=chunk["image_size"], minimum_frames=chunk["minimum_frames"],
                maximum_frames=chunk["maximum_frames"], maximum_objects=chunk["maximum_objects"],
                random_generator=random_generator)
            sequence_frames = {member_name: save_frames(data_folder=chunk["data_folder"],
                                                        frame_folder=f"{sequence_name}/{member_name}", frames=frames)
                               for member_name, frames in member_frames.items()}
            sequence_labels = {"category_id": category_identifier}
        elif chunk["task"] == "sequence_classification":
            frames, category_identifier = generate_moving_shape_sequence(image_size=chunk["image_size"],
                                                                         number_frames=number_frames,
                                                                         random_generator=random_generator)
            sequence_frames = {"frames": save_frames(data_folder=chunk["data_folder"], frame_folder=sequence_name,
                                                     frames=frames)}
            sequence_labels = {"category_id": category_identifier}
        elif chunk["task"] == "frame_selection":
            frames, frame_category_ids = generate_selection_sequence(image_size=chunk["image_size"],
                                                                     number_frames=number_frames,
                                                                     random_generator=random_generator)
            sequence_frames = {"frames": save_frames(data_folder=chunk["data_folder"], frame_folder=sequence_name,
                                                     frames=frames)}
            sequence_labels = {"frame_category_ids": frame_category_ids}
        else:
            frames, frame_category_ids = generate_shape_change_sequence(image_size=chunk["image_size"],
                                                                        number_frames=number_frames,
                                                                        maximum_objects=chunk["maximum_objects"],
                                                                        random_generator=random_generator)
            sequence_frames = {"frames": save_frames(data_folder=chunk["data_folder"], frame_folder=sequence_name,
                                                     frames=frames)}
            sequence_labels = {"frame_category_ids": frame_category_ids}
        sequence_records.append({"index": sequence_index, "name": sequence_name, **sequence_frames,
                                 **sequence_labels})

    return sequence_records


def build_sequence_labels(sequence_records: List[Dict[str, Any]], split_name: str, task: str) -> Tuple[
        Dict[str, Any], Dict[str, int]]:
    """
    Assembles the 'labels.json' manifest of a synthetic sequence split, read by the sequence loader: every sequence
    with its ordered frames and its category (sequence classification) or the category of every frame (sequence
    labelling, frame selection), or every pair with the ordered frames of its before and after members and its
    category (sequence comparison, under the 'pairs' key).

    Args:
        sequence_records (List[Dict[str, Any]]): The 'index', 'name', frames ('frames', or 'before' and 'after') and
            'category_id' or 'frame_category_ids' of every sample.
        split_name (str): The name of the split, written in the description.
        task (str): The sequence task of the dataset, giving its categories and its manifest key.

    Returns:
        Tuple[Dict[str, Any], Dict[str, int]]: The manifest and the number of samples (or frames) of every class.
    """
    categories = TASK_CATEGORIES[task]
    category_names = {category_identifier: name for name, category_identifier in categories.items()}
    class_counts = {category_name: 0 for category_name in categories}

    # One entry per sample, with its frames in order and its category, or the category of every frame
    sequences = []
    for record in sequence_records:
        sequence = {"id": record["index"] + 1, **{key: value for key, value in record.items() if key != "index"}}
        label_key = "category_id" if "category_id" in record else "frame_category_ids"
        for category_identifier in ([record["category_id"]] if label_key == "category_id" else
                                    record["frame_category_ids"]):
            class_counts[category_names[category_identifier]] += 1
        sequences.append(sequence)

    labels = {"info": {"description": f"Synthetic {task.replace('_', ' ')} dataset, {split_name} split"},
              "categories": [{"id": category_identifier, "name": name}
                             for name, category_identifier in categories.items()],
              TASK_MANIFEST_KEYS[task]: sequences}
    return labels, class_counts


def generate_split(split_folder: str, split_name: str, number_samples: int, image_size: int, maximum_objects: int,
                   minimum_frames: int, maximum_frames: int, number_workers: int, task: str) -> Dict[str, int]:
    """
    Generates one split of the synthetic dataset in the folder layout the loaders of the task read: the images (or
    the frame folders of the sequences) in '<split_folder>/data', the class masks of the segmentation images in
    '<split_folder>/masks', and their labels in '<split_folder>/labels.json' (COCO annotations for object detection,
    COCO annotations with the run length encoding of every object for instance segmentation, COCO annotations with the
    keypoints of every object and the keypoint names and skeleton of every category for keypoint detection, a samples
    manifest with one shape per image for classification, a samples manifest with the mask of every image for semantic
    segmentation, a sequences manifest for the sequence tasks and the frame selection, a pairs manifest for the
    sequence comparison, a samples manifest with the shape count and the shape classes of every image for the
    self-supervised images).

    Args:
        split_folder (str): The folder of the split.
        split_name (str): The name of the split, used in the file names ('train', 'validation' or 'test').
        number_samples (int): The number of images, of sequences or of pairs of the split.
        image_size (int): The height and width of the images in pixels.
        maximum_objects (int): The maximum number of objects per image (object detection, instance segmentation,
            keypoint detection, semantic segmentation, self-supervised, the classification images holding a single
            shape), per frame (sequence labelling) or per before member of a pair (sequence comparison).
        minimum_frames (int): The minimum number of frames of a sequence or of a member of a pair (sequence tasks
            only).
        maximum_frames (int): The maximum number of frames of a sequence or of a member of a pair (sequence tasks
            only).
        number_workers (int): The number of worker processes generating the images.
        task (str): "object_detection", "instance_segmentation", "keypoint_detection", "classification",
            "semantic_segmentation", "sequence_classification", "sequence_labelling", "sequence_comparison",
            "frame_selection" or "self_supervised".

    Returns:
        Dict[str, int]: The number of labels (objects, images, sequences, frames or pairs) of every class in the split.
    """
    data_folder = os.path.join(split_folder, "data")
    os.makedirs(name=data_folder, exist_ok=True)

    # Sequences and pairs: split them into fixed-size chunks shared among the worker processes, then assemble their
    # manifest
    if task in SEQUENCE_TASKS:
        chunks = [{"data_folder": data_folder, "split_name": split_name, "first_index": first_index,
                   "number_sequences": min(SEQUENCES_PER_TASK, number_samples - first_index),
                   "image_size": image_size, "minimum_frames": minimum_frames, "maximum_frames": maximum_frames,
                   "maximum_objects": maximum_objects, "task": task}
                  for first_index in range(0, number_samples, SEQUENCES_PER_TASK)]
        with Pool(processes=number_workers) as pool:
            sequence_records = [record for chunk_records in pool.map(generate_sequence_chunk, chunks)
                                for record in chunk_records]
        labels, class_counts = build_sequence_labels(sequence_records=sequence_records, split_name=split_name,
                                                     task=task)

    # Images: split them into fixed-size chunks too, a classification image holding a single shape, a segmentation
    # image saving its class mask and a self-supervised image holding 5 to maximum_objects shapes
    else:
        masks_folder = os.path.join(split_folder, "masks") if task == "semantic_segmentation" else None
        if masks_folder is not None:
            os.makedirs(name=masks_folder, exist_ok=True)
        chunks = [{"data_folder": data_folder, "masks_folder": masks_folder, "split_name": split_name,
                   "first_index": first_index, "number_images": min(IMAGES_PER_TASK, number_samples - first_index),
                   "image_size": image_size,
                   "maximum_objects": (maximum_objects if task in TASKS_WITH_BACKGROUND_CLASS + ["self_supervised"]
                                       else 1),
                   "remove_covered_objects": task in TASKS_REMOVING_COVERED_OBJECTS,
                   "instance_segmentation": task == "instance_segmentation",
                   "keypoints": task == "keypoint_detection", "counted_shapes": task == "self_supervised"}
                  for first_index in range(0, number_samples, IMAGES_PER_TASK)]
        with Pool(processes=number_workers) as pool:
            image_records = [record for chunk_records in pool.map(generate_image_chunk, chunks)
                             for record in chunk_records]

        # Assemble the labels of the split in the format of the task
        if task == "object_detection":
            labels, class_counts = build_detection_labels(image_records=image_records, image_size=image_size,
                                                          split_name=split_name, categories=SHAPE_CATEGORIES)
        elif task == "keypoint_detection":
            labels, class_counts = build_keypoint_labels(image_records=image_records, image_size=image_size,
                                                         split_name=split_name)
        elif task == "instance_segmentation":
            labels, class_counts = build_instance_segmentation_labels(image_records=image_records,
                                                                      image_size=image_size, split_name=split_name)
        elif task == "semantic_segmentation":
            labels, class_counts = build_segmentation_labels(image_records=image_records, split_name=split_name)
        elif task == "self_supervised":
            labels, class_counts = build_shape_count_labels(image_records=image_records, split_name=split_name)
        else:
            labels, class_counts = build_classification_labels(image_records=image_records, split_name=split_name)
    with open(os.path.join(split_folder, "labels.json"), "w") as labels_file:
        json.dump(labels, labels_file)

    return class_counts


def create_synthetic_dataset(output_folder: str, image_size: int, split_sizes: Dict[str, int], maximum_objects: int,
                             minimum_frames: int, maximum_frames: int, number_workers: int, task: str) -> None:
    """
    Creates the whole synthetic shapes dataset of a task: the train, validation and test splits, all labelled, and the
    'label_mapping.json' file used by the 'multi_class' label mapping mode of the configuration. The dataset is a
    trivial task used to check that a model learns (it should overfit it within minutes): several shapes per image
    to detect, to segment object by object, to locate the keypoints of or to segment, a single shape per image to
    classify, a shape crossing the frames of a sequence whose movement direction is classified, shapes appearing and
    disappearing during a sequence whose every frame is classified with its change, a pair of sequences of the same
    objects whose after member may have lost one of them, or shapes of several sizes whose largest circle, triangle
    and star are the frames to select. Every sequence, and every member of a pair, has its own number of frames between
    minimum_frames and maximum_frames. The self-supervised dataset is not a trivial task: 5 to maximum_objects shapes
    per image drawn like a sequence labelling frame, the shape count labelling the k-nearest neighbour evaluation of
    the self-supervised trainer.

    Args:
        output_folder (str): The root folder of the dataset.
        image_size (int): The height and width of the images in pixels.
        split_sizes (Dict[str, int]): The number of images, of sequences or of pairs of every split, keyed by split
            name.
        maximum_objects (int): The maximum number of objects per image (object detection, instance segmentation,
            keypoint detection, semantic segmentation, self-supervised, 5 to COUNTED_SHAPES_MAXIMUM_OBJECTS), per frame
            (sequence labelling, at least 5) or per before member of a pair (sequence comparison).
        minimum_frames (int): The minimum number of frames of a sequence or of a member of a pair, at least 2, and at
            least the 3 wanted shapes for the frame selection (sequence tasks only).
        maximum_frames (int): The maximum number of frames of a sequence or of a member of a pair, at least
            minimum_frames (sequence tasks only).
        number_workers (int): The number of worker processes generating the images.
        task (str): "object_detection", "instance_segmentation", "keypoint_detection", "classification",
            "semantic_segmentation", "sequence_classification", "sequence_labelling", "sequence_comparison",
            "frame_selection" or "self_supervised".

    Raises:
        ValueError: If the frame range of a sequence task starts below 2 (below the number of wanted shapes for the
            frame selection) or is empty.
    """
    # A movement, a change of shapes or a comparison needs two frames, a selection one frame per wanted shape, in a
    # range that holds at least one length
    minimum_sequence_frames = len(SELECTION_WANTED_SHAPES) if task == "frame_selection" else MINIMUM_SEQUENCE_FRAMES
    if task in SEQUENCE_TASKS and not minimum_sequence_frames <= minimum_frames <= maximum_frames:
        raise ValueError(f"The sequences need {minimum_sequence_frames} <= minimum_frames <= maximum_frames, got "
                         f"minimum_frames {minimum_frames} and maximum_frames {maximum_frames}.")

    # Generate every split in its own folder
    sample_name = TASK_SAMPLE_NAMES[task]
    for split_name, number_samples in split_sizes.items():
        split_folder = os.path.abspath(os.path.join(output_folder, split_name))
        print_color(color="blue", output=f"Generating {number_samples} {split_name} {sample_name}...")
        class_counts = generate_split(split_folder=split_folder, split_name=split_name, number_samples=number_samples,
                                      image_size=image_size, maximum_objects=maximum_objects,
                                      minimum_frames=minimum_frames, maximum_frames=maximum_frames,
                                      number_workers=number_workers, task=task)
        print_color(color="green", output=f"{split_name} split : {format_path_link(path=split_folder)}", indent=1)
        print_color(color="green", output=f"Labels per class : {class_counts}", indent=1)

    # Map every raw category identifier to its class index: itself for the detector and the segmenters (index 0 being
    # their background), from 0 for the models of the other tasks, and the categories with keypoints numbered from 1
    # for the keypoint detection
    if task == "keypoint_detection":
        label_mapping = build_keypoint_label_mapping()
    else:
        class_index_offset = 0 if task in TASKS_WITH_BACKGROUND_CLASS else 1
        label_mapping = {str(category_identifier): category_identifier - class_index_offset
                         for category_identifier in TASK_CATEGORIES[task].values()}
    label_mapping_path = os.path.abspath(os.path.join(output_folder, "label_mapping.json"))
    with open(label_mapping_path, "w") as label_mapping_file:
        json.dump(label_mapping, label_mapping_file, indent=4)
    print_color(color="green", output=f"Label Mapping File : {format_path_link(path=label_mapping_path)}",
                add_separators=True)


if __name__ == "__main__":
    # Task of the dataset: "object_detection" (several shapes per image), "instance_segmentation" (several shapes per
    # image, with the visible part of every shape), "keypoint_detection" (several upright shapes per image, with the
    # keypoints of every shape), "classification" (one shape per image),
    # "semantic_segmentation" (several shapes per image, with the class mask of every image),
    # "sequence_classification" (one shape crossing the frames, labelled with its movement direction),
    # "sequence_labelling" (5 to 10 shapes per frame, a shape appearing or disappearing from one frame to the next,
    # labelled per frame), "sequence_comparison" (a before and an after sequence of the same objects, labelled with
    # whether the after member lost one of them), "frame_selection" (one shape per frame, the largest circle,
    # triangle and star being the frames to select) or "self_supervised" (5 to 10 shapes per image drawn like a
    # sequence labelling frame, labelled with their number of shapes and keeping the classes of their shapes)
    task = "classification"
    # Root folder of the created dataset of every task, the ones expected by the synthetic configurations of
    # vision_training/configurations/, the object detection configuration reading the instance segmentation dataset
    # (the boxes of the visible part of every shape). Existing images and labels of the same splits are overwritten
    output_folders = {"object_detection": "/home/robert_kelevra/Data/object_detection_data/synthetic_shapes",
                      "instance_segmentation": "/home/robert_kelevra/Data/instance_segmentation_data/synthetic_shapes",
                      "keypoint_detection": "/home/robert_kelevra/Data/keypoint_data/synthetic_shapes",
                      "classification": "/home/robert_kelevra/Data/classifier_data/synthetic_shapes",
                      "semantic_segmentation": "/home/robert_kelevra/Data/segmentation_data/synthetic_shapes",
                      "sequence_classification": "/home/robert_kelevra/Data/sequence_data/synthetic_moving_shapes",
                      "sequence_labelling": "/home/robert_kelevra/Data/sequence_data/synthetic_changing_shapes",
                      "sequence_comparison": "/home/robert_kelevra/Data/sequence_data/synthetic_object_loss_pairs",
                      "frame_selection": "/home/robert_kelevra/Data/sequence_data/synthetic_selected_shapes",
                      "self_supervised": "/home/robert_kelevra/Data/self_supervised_data/synthetic_counted_shapes"}
    # Height and width of the square images in pixels, for every task
    image_sizes = {"object_detection": 1024, "instance_segmentation": 512, "keypoint_detection": 512,
                   "classification": 512, "semantic_segmentation": 512,
                   "sequence_classification": 512, "sequence_labelling": 512, "sequence_comparison": 512,
                   "frame_selection": 256, "self_supervised": 512}
    # Number of images (or sequences, or pairs) of every split, all of them labelled, for every task. The 128000
    # training images of the instance segmentation are the 2000 iterations of 64 images of the synthetic object
    # detection and instance segmentation configurations, every image being seen once. The 250000 training images of
    # the keypoint detection, of 2 to 5 objects, are seen about 1.3 times by the 20000 iterations of 16 images of its
    # synthetic configuration. The 250000 training sequences of the sequence labelling are seen about 0.6 times by the
    # 5000 iterations of 32 sequences of the launches of its hyperparameter experiment (run-<n>.yaml). The 500000
    # training images of the self-supervised dataset are not labelled for the training, their shape counts only
    # labelling the k-nearest neighbour evaluation of the test split
    split_sizes = {"object_detection": {"train": 100, "validation": 100, "test": 100},
                   "instance_segmentation": {"train": 128000, "validation": 20000, "test": 20000},
                   "keypoint_detection": {"train": 250000, "validation": 50000, "test": 50000},
                   "classification": {"train": 500000, "validation": 50000, "test": 50000},
                   "semantic_segmentation": {"train": 2000, "validation": 200, "test": 200},
                   "sequence_classification": {"train": 2000, "validation": 200, "test": 200},
                   "sequence_labelling": {"train": 250000, "validation": 50000, "test": 50000},
                   "sequence_comparison": {"train": 2000, "validation": 200, "test": 200},
                   "frame_selection": {"train": 2000, "validation": 200, "test": 200},
                   "self_supervised": {"train": 500000, "validation": 10000, "test": 10000}}
    # Maximum number of shapes drawn in one image (at least one is always drawn, object detection, instance
    # segmentation and semantic segmentation, at least two for the keypoint detection), in a sequence labelling frame
    # or a self-supervised image (at least five) or in the before member of a pair (at least two, sequence
    # comparison), for every task (unused by the other tasks)
    maximum_objects = {"object_detection": 6, "instance_segmentation": 6, "keypoint_detection": 5, "classification": 1,
                       "semantic_segmentation": 6, "sequence_classification": 1, "sequence_labelling": 10,
                       "sequence_comparison": 6, "frame_selection": 1, "self_supervised": 10}
    # Minimum and maximum number of frames of a sequence or of a member of a pair, every sequence and every member
    # drawing its own number of frames in that range (at least two, at least three for the frame selection, unused by
    # the image tasks)
    minimum_frames = {"object_detection": 1, "instance_segmentation": 1, "keypoint_detection": 1, "classification": 1,
                      "semantic_segmentation": 1, "sequence_classification": 2, "sequence_labelling": 2,
                      "sequence_comparison": 2, "frame_selection": 4, "self_supervised": 1}
    maximum_frames = {"object_detection": 1, "instance_segmentation": 1, "keypoint_detection": 1, "classification": 1,
                      "semantic_segmentation": 1, "sequence_classification": 12, "sequence_labelling": 12,
                      "sequence_comparison": 12, "frame_selection": 12, "self_supervised": 1}
    # Number of worker processes generating the images
    number_workers = 8

    create_synthetic_dataset(output_folder=output_folders[task], image_size=image_sizes[task],
                             split_sizes=split_sizes[task], maximum_objects=maximum_objects[task],
                             minimum_frames=minimum_frames[task], maximum_frames=maximum_frames[task],
                             number_workers=number_workers, task=task)
