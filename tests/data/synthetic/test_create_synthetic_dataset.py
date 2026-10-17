"""
Tests of vision_training/data/synthetic/create_synthetic_dataset.py: the synthetic detection, instance segmentation,
classification, semantic segmentation, sequence classification, sequence labelling, sequence comparison and frame
selection datasets are valid, load with the datasets of the framework and follow their labels, the objects hidden by
later ones being left out of the detection and instance segmentation labels. The shapes stay inside their square, keep
the geometry that tells their class apart from the neighbouring classes (rhombus and square, parallelogram and
rectangle, trapezoid and triangle, kite and rhombus, annulus and circle), every hollow shape is its filled shape
with one hole cut inside it, and the shapes whose class is the label get the minimum size.
"""

import json
import cv2
import numpy as np
import pytest

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from vision_training.data.classification.classification_dataset import ClassificationDataset
from vision_training.data.detection.coco_dataset import CocoDetectionDataset
from vision_training.data.detection.coco_masks import decode_run_length_encoding
from vision_training.data.segmentation.segmentation_dataset import SegmentationDataset
from vision_training.data.sequence.sequence_dataset import SequenceDataset
from vision_training.data.synthetic import create_synthetic_dataset as synthetic_dataset_module
from vision_training.data.synthetic.create_synthetic_dataset import (HOLLOW_SHAPE_NAMES, KITE_UPPER_RATIO_RANGE,
                                                                     KITE_WIDTH_RATIO_RANGE,
                                                                     MINIMUM_SHAPE_SIZE,
                                                                     MINIMUM_VISIBLE_OBJECT_PIXELS,
                                                                     MOVEMENT_CATEGORIES, MOVEMENT_DIRECTIONS,
                                                                     PAIR_CATEGORIES,
                                                                     PARALLELOGRAM_ANGLE_RANGE_DEGREES,
                                                                     PARALLELOGRAM_SIDE_RATIO_RANGE,
                                                                     RHOMBUS_DIAGONAL_RATIO_RANGE,
                                                                     SELECTION_CATEGORIES, SHAPE_CATEGORIES,
                                                                     SHAPE_DRAWING_FUNCTIONS,
                                                                     TRAPEZOID_TOP_RATIO_RANGE,
                                                                     build_kite_vertices,
                                                                     build_parallelogram_vertices,
                                                                     build_rhombus_vertices,
                                                                     build_trapezoid_vertices,
                                                                     create_synthetic_dataset, draw_annulus_mask,
                                                                     draw_hollow_mask,
                                                                     draw_shapes_frame,
                                                                     generate_changing_shape_sequence,
                                                                     generate_image, get_mask_bounding_box,
                                                                     keep_visible_objects)

IMAGE_SIZE = 256

# Tolerance of the geometric comparisons of the vertex builders, computed in float64
GEOMETRY_TOLERANCE = 1e-9


class CallRecorder:
    """
    Wraps a function called with keyword arguments only and records the arguments of every call before running it,
    so that a test can read the sizes and vertices the generator draws its shapes with.
    """

    def __init__(self, function: Callable[..., Any]) -> None:
        """
        Wraps the function.

        Args:
            function (Callable[..., Any]): The function to record and run.
        """
        self.function = function
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **keyword_arguments: Any) -> Any:
        """
        Records the arguments of the call, then runs the wrapped function with them.

        Args:
            **keyword_arguments (Any): The arguments of the call.

        Returns:
            Any: The result of the wrapped function.
        """
        self.calls.append(keyword_arguments)
        return self.function(**keyword_arguments)


def compute_side_lengths(vertices: np.ndarray) -> np.ndarray:
    """
    Computes the side lengths of a polygon, the side from every vertex to the next one in turn.

    Args:
        vertices (np.ndarray): The [N, 2] vertices of the polygon in order around its outline.

    Returns:
        np.ndarray: The [N] side lengths.
    """
    return np.linalg.norm(np.roll(vertices, shift=-1, axis=0) - vertices, axis=1)


def compute_polygon_angles(vertices: np.ndarray) -> np.ndarray:
    """
    Computes the interior angles of a convex polygon, the angle at every vertex between its two sides.

    Args:
        vertices (np.ndarray): The [N, 2] vertices of the polygon in order around its outline.

    Returns:
        np.ndarray: The [N] angles in degrees.
    """
    previous_edges = np.roll(vertices, shift=1, axis=0) - vertices
    next_edges = np.roll(vertices, shift=-1, axis=0) - vertices
    cosines = np.sum(previous_edges * next_edges, axis=1) / (np.linalg.norm(previous_edges, axis=1) *
                                                             np.linalg.norm(next_edges, axis=1))
    return np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))


def compute_cross_product(first_vector: np.ndarray, second_vector: np.ndarray) -> float:
    """
    Computes the z component of the cross product of two 2D vectors, zero when they are parallel.

    Args:
        first_vector (np.ndarray): The [2] first vector.
        second_vector (np.ndarray): The [2] second vector.

    Returns:
        float: The cross product.
    """
    return float(first_vector[0] * second_vector[1] - first_vector[1] * second_vector[0])


@pytest.mark.parametrize(argnames="shape_name", argvalues=list(SHAPE_CATEGORIES))
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


@pytest.mark.parametrize(argnames="shape_name", argvalues=list(SHAPE_CATEGORIES))
def test_every_shape_stays_inside_its_square(shape_name: str) -> None:
    """
    Verifies that every shape class, whatever its rotation, draws its pixels inside the 'size' x 'size' square around
    its centre, the canvas of draw_shape_descriptor, so that no shape is cut by its canvas.

    Args:
        shape_name (str): The shape class under test.
    """
    random_generator = np.random.default_rng(seed=0)

    for size in [33, MINIMUM_SHAPE_SIZE, 101]:
        for _ in range(20):
            mask = np.zeros(shape=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            SHAPE_DRAWING_FUNCTIONS[shape_name](mask=mask, center_x=128, center_y=128, size=size,
                                                random_generator=random_generator)
            row_indices, column_indices = np.nonzero(mask)
            assert row_indices.min() >= 128 - size // 2 and row_indices.max() <= 128 + size // 2
            assert column_indices.min() >= 128 - size // 2 and column_indices.max() <= 128 + size // 2


def test_quadrilaterals_keep_their_defining_geometry() -> None:
    """
    Verifies the vertices of the new polygon classes against their neighbouring classes: the rhombus has four equal
    sides and acute angles far from the right angles of the square, the parallelogram has equal opposite sides of
    clearly different lengths and acute angles far from the right angles of the rectangle, the trapezoid has one pair of
    parallel sides of clearly different lengths and acute angles at its long side, and the kite has two pairs of equal
    adjacent sides of different lengths and an upper corner that does not flatten.
    """
    random_generator = np.random.default_rng(seed=0)

    for _ in range(200):
        # Rhombus: equal sides, acute angles from the ratio range of its diagonals
        rhombus = build_rhombus_vertices(random_generator=random_generator)
        rhombus_sides, rhombus_angles = compute_side_lengths(vertices=rhombus), compute_polygon_angles(vertices=rhombus)
        assert np.allclose(rhombus_sides, rhombus_sides[0], atol=GEOMETRY_TOLERANCE)
        assert (np.degrees(2 * np.arctan(RHOMBUS_DIAGONAL_RATIO_RANGE[0])) - GEOMETRY_TOLERANCE <= rhombus_angles.min()
                <= np.degrees(2 * np.arctan(RHOMBUS_DIAGONAL_RATIO_RANGE[1])) + GEOMETRY_TOLERANCE)

        # Parallelogram: equal and parallel opposite sides, long side 1.5 to 2.5 times the short one, acute angles
        parallelogram = build_parallelogram_vertices(random_generator=random_generator)
        parallelogram_sides = compute_side_lengths(vertices=parallelogram)
        assert np.isclose(parallelogram_sides[0], parallelogram_sides[2], atol=GEOMETRY_TOLERANCE)
        assert np.isclose(parallelogram_sides[1], parallelogram_sides[3], atol=GEOMETRY_TOLERANCE)
        assert (PARALLELOGRAM_SIDE_RATIO_RANGE[0] - GEOMETRY_TOLERANCE <= parallelogram_sides.max() /
                parallelogram_sides.min() <= PARALLELOGRAM_SIDE_RATIO_RANGE[1] + GEOMETRY_TOLERANCE)
        assert (PARALLELOGRAM_ANGLE_RANGE_DEGREES[0] - GEOMETRY_TOLERANCE <=
                compute_polygon_angles(vertices=parallelogram).min() <=
                PARALLELOGRAM_ANGLE_RANGE_DEGREES[1] + GEOMETRY_TOLERANCE)

        # Trapezoid: parallel long and short sides in their ratio range, legs not parallel, acute angles on the base
        trapezoid = build_trapezoid_vertices(random_generator=random_generator)
        trapezoid_sides = compute_side_lengths(vertices=trapezoid)
        assert abs(compute_cross_product(first_vector=trapezoid[1] - trapezoid[0],
                                         second_vector=trapezoid[2] - trapezoid[3])) < GEOMETRY_TOLERANCE
        assert abs(compute_cross_product(first_vector=trapezoid[2] - trapezoid[1],
                                         second_vector=trapezoid[3] - trapezoid[0])) > 0.01
        assert (TRAPEZOID_TOP_RATIO_RANGE[0] - GEOMETRY_TOLERANCE <= trapezoid_sides[2] / trapezoid_sides[0] <=
                TRAPEZOID_TOP_RATIO_RANGE[1] + GEOMETRY_TOLERANCE)
        assert compute_polygon_angles(vertices=trapezoid)[:2].max() < 90.0

        # Kite: short upper sides and long lower sides, two by two equal, upper corner at most 107 degrees
        kite = build_kite_vertices(random_generator=random_generator)
        kite_sides = compute_side_lengths(vertices=kite)
        assert np.isclose(kite_sides[0], kite_sides[3], atol=GEOMETRY_TOLERANCE)
        assert np.isclose(kite_sides[1], kite_sides[2], atol=GEOMETRY_TOLERANCE)
        assert kite_sides[0] < 0.9 * kite_sides[1]
        largest_upper_angle = np.degrees(2 * np.arctan(KITE_WIDTH_RATIO_RANGE[1] * (1 + KITE_UPPER_RATIO_RANGE[0]) /
                                                       (2 * KITE_UPPER_RATIO_RANGE[0])))
        assert compute_polygon_angles(vertices=kite)[0] <= largest_upper_angle + GEOMETRY_TOLERANCE


def test_annulus_is_a_closed_ring_around_a_hole() -> None:
    """
    Verifies that the annulus is a single ring closed around its hole: its centre is left empty, its pixels form one
    group, and its hole is a background region of its own, apart from the background around the ring.
    """
    random_generator = np.random.default_rng(seed=0)

    for _ in range(20):
        mask = np.zeros(shape=(128, 128), dtype=np.uint8)
        draw_annulus_mask(mask=mask, center_x=64, center_y=64, size=MINIMUM_SHAPE_SIZE,
                          random_generator=random_generator)
        assert mask[64, 64] == 0
        number_ring_groups, _ = cv2.connectedComponents(image=(mask > 0).astype(np.uint8), connectivity=8)
        number_background_groups, _ = cv2.connectedComponents(image=(mask == 0).astype(np.uint8), connectivity=4)
        assert number_ring_groups - 1 == 1 and number_background_groups - 1 == 2


@pytest.mark.parametrize(argnames="shape_name", argvalues=HOLLOW_SHAPE_NAMES)
@pytest.mark.parametrize(argnames="size", argvalues=[33, MINIMUM_SHAPE_SIZE, 101])
@pytest.mark.parametrize(argnames="upright", argvalues=[False, True])
def test_hollow_shapes_are_closed_rings_inside_their_filled_shape(shape_name: str, size: int, upright: bool) -> None:
    """
    Verifies that every hollow shape is its filled shape with one hole cut inside it, like the annulus is the hollow
    circle: drawn with the same random generator state, its pixels are a strict part of the filled shape and keep its
    keypoints, they form one group, and the hole is a background region of its own, apart from the background around
    the walls.

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

        # The walls lie inside the filled shape, cover less of it and keep its keypoints
        assert np.all(filled_mask[hollow_mask > 0] > 0) and np.count_nonzero(hollow_mask) < np.count_nonzero(filled_mask)
        np.testing.assert_array_equal(hollow_keypoints, filled_keypoints)

        # The walls form one group closed around a single hole
        number_ring_groups, _ = cv2.connectedComponents(image=(hollow_mask > 0).astype(np.uint8), connectivity=8)
        number_background_groups, _ = cv2.connectedComponents(image=(hollow_mask == 0).astype(np.uint8),
                                                              connectivity=4)
        assert number_ring_groups - 1 == 1 and number_background_groups - 1 == 2


def test_hollow_shape_too_small_for_a_hole_raises() -> None:
    """
    Verifies that draw_hollow_mask raises instead of returning a filled shape under a hollow class when the shape is too
    small to leave a hole inside its minimum walls.
    """
    mask = np.zeros(shape=(32, 32), dtype=np.uint8)

    with pytest.raises(expected_exception=ValueError, match="left none"):
        draw_hollow_mask(mask=mask, center_x=16, center_y=16, size=3, random_generator=np.random.default_rng(seed=0),
                         filled_drawing_function=SHAPE_DRAWING_FUNCTIONS["square"])
    assert not np.any(mask)


@pytest.mark.parametrize(argnames=("image_size", "expected_size_range"), argvalues=[
    (512, (MINIMUM_SHAPE_SIZE, 170)), (IMAGE_SIZE, (MINIMUM_SHAPE_SIZE, 85)), (64, (33, 33))])
def test_image_shapes_get_the_minimum_size(monkeypatch: pytest.MonkeyPatch, image_size: int,
                                           expected_size_range: Tuple[int, int]) -> None:
    """
    Verifies that the shapes of the detection, segmentation and classification images are drawn from
    MINIMUM_SHAPE_SIZE pixels to a third of the image side, every shape getting the largest size on images too small
    for the minimum.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture recording the shape descriptors drawn.
        image_size (int): The side of the image.
        expected_size_range (Tuple[int, int]): The expected smallest and largest shape sizes.
    """
    shape_descriptor_recorder = CallRecorder(function=synthetic_dataset_module.draw_shape_descriptor)
    monkeypatch.setattr(target=synthetic_dataset_module, name="draw_shape_descriptor", value=shape_descriptor_recorder)
    generate_image(image_size=image_size, maximum_objects=6, random_generator=np.random.default_rng(seed=0))

    assert len(shape_descriptor_recorder.calls) >= 1
    assert all((call["minimum_size"], call["maximum_size"]) == expected_size_range
               for call in shape_descriptor_recorder.calls)


@pytest.mark.parametrize(argnames=("image_size", "expected_size_range"), argvalues=[
    (512, (85, 170)), (IMAGE_SIZE, (MINIMUM_SHAPE_SIZE, 85)), (64, (21, 21))])
def test_changing_shapes_get_the_minimum_size(monkeypatch: pytest.MonkeyPatch, image_size: int,
                                              expected_size_range: Tuple[int, int]) -> None:
    """
    Verifies that the changing shape of the sequence labelling measures a sixth to a third of the frame side, raised
    to MINIMUM_SHAPE_SIZE pixels when the frames are large enough, and the largest size on frames too small for it.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture recording the sizes of the drawn shapes.
        image_size (int): The side of the frames.
        expected_size_range (Tuple[int, int]): The expected smallest and largest shape sizes.
    """
    drawing_recorders = {shape_name: CallRecorder(function=drawing_function)
                         for shape_name, drawing_function in SHAPE_DRAWING_FUNCTIONS.items()}
    for shape_name, drawing_recorder in drawing_recorders.items():
        monkeypatch.setitem(dic=SHAPE_DRAWING_FUNCTIONS, name=shape_name, value=drawing_recorder)

    random_generator = np.random.default_rng(seed=0)
    for _ in range(20):
        generate_changing_shape_sequence(image_size=image_size, number_frames=4, random_generator=random_generator)

    drawn_sizes = [call["size"] for drawing_recorder in drawing_recorders.values() for call in drawing_recorder.calls]
    assert len(drawn_sizes) >= 40
    assert expected_size_range[0] <= min(drawn_sizes) and max(drawn_sizes) <= expected_size_range[1]


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
        x, y, width, height = annotation["bbox"]
        assert width > 0 and height > 0 and x >= 0 and y >= 0
        assert x + width <= IMAGE_SIZE and y + height <= IMAGE_SIZE

    # The multi-class label mapping keeps every raw category identifier as its own model class
    label_mapping = json.loads((tmp_path / "label_mapping.json").read_text())
    label_dictionary = {int(raw_identifier): model_identifier for raw_identifier, model_identifier in
                        label_mapping.items()}
    label_dictionary["background"] = 0

    dataset = CocoDetectionDataset(data_directory=str(tmp_path / "train" / "data"),
                            labels_file=str(tmp_path / "train" / "labels.json"), image_size=IMAGE_SIZE,
                            keep_ratio=True, label_mode="multi_class", label_dictionary=label_dictionary)
    loaded_labels = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample["bounding_boxes"].shape[0] == sample["labels"].shape[0] > 0
        loaded_labels |= set(sample["labels"].tolist())
    assert loaded_labels <= set(SHAPE_CATEGORIES.values())


def test_created_classification_dataset_has_one_shape_per_image(tmp_path: Path) -> None:
    """
    Verifies the classification mode of the generator: every split gets a manifest with one sample per image, labelled
    with the category of its single shape, a label mapping giving the classes 0 to 19, and the classification dataset
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
    Verifies the segmentation mode of the generator: every image gets a single-channel mask of its size in 'masks/',
    whose labelled pixels are exactly the pixels of the drawn shapes (every pixel differing from the background
    colour) with shape categories, a label mapping keeping the category identifiers as classes after the background
    class 0, and the segmentation dataset loads it with its class names.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_synthetic_dataset(output_folder=str(tmp_path), image_size=IMAGE_SIZE,
                             split_sizes={"train": 6, "validation": 2, "test": 2}, maximum_objects=4,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="semantic_segmentation")

    labels = json.loads((tmp_path / "train" / "labels.json").read_text())
    assert len(labels["samples"]) == 6
    assert [category["name"] for category in labels["categories"]] == list(SHAPE_CATEGORIES)

    # The labelled pixels of every mask are the pixels of the shapes, the background being the most frequent colour
    for sample in labels["samples"]:
        image = cv2.imread(filename=str(tmp_path / "train" / "data" / sample["file_name"]))
        mask = cv2.imread(filename=str(tmp_path / "train" / "masks" / sample["mask_file_name"]),
                          flags=cv2.IMREAD_UNCHANGED)
        assert mask.shape == (IMAGE_SIZE, IMAGE_SIZE) and mask.dtype == np.uint8
        colours, counts = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
        shape_pixels = np.any(image != colours[np.argmax(counts)], axis=-1)
        assert np.array_equal(mask > 0, shape_pixels)
        assert set(np.unique(mask[mask > 0]).tolist()) <= set(SHAPE_CATEGORIES.values())

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

    # The detection keeps the whole boxes of the visible objects
    detection_objects, detection_segmentations = keep_visible_objects(objects=objects, instance_map=instance_map,
                                                                      keep_visible_parts=False)
    assert detection_objects == [(2, [0, 20, 20, 40]), (3, [0, 0, 30, 40])] and detection_segmentations == []

    # The instance segmentation keeps the box and the encoding of the visible part of every visible object
    instance_objects, instance_segmentations = keep_visible_objects(objects=objects, instance_map=instance_map,
                                                                    keep_visible_parts=True)
    assert instance_objects == [(2, [0, 20, 10, 40]), (3, [0, 0, 30, 40])]
    assert np.array_equal(decode_run_length_encoding(segmentation=instance_segmentations[0]), instance_map == 2)


def test_created_instance_segmentation_dataset_encodes_the_visible_parts(tmp_path: Path) -> None:
    """
    Verifies the instance segmentation mode of the generator: every object of the COCO labels has the encoding of its
    visible part, at least MINIMUM_VISIBLE_OBJECT_PIXELS pixels and disjoint from the other objects of its image, its
    box being the tight box of that part and its area its number of pixels, the parts covering the shape pixels of the
    image apart from the left out objects, a label mapping keeping the category identifiers after the background class
    0, and the COCO dataset loads the same masks.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_synthetic_dataset(output_folder=str(tmp_path), image_size=IMAGE_SIZE,
                             split_sizes={"train": 6, "validation": 2, "test": 2}, maximum_objects=6,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="instance_segmentation")

    labels = json.loads((tmp_path / "train" / "labels.json").read_text())
    assert len(labels["images"]) == 6 and len(labels["annotations"]) >= 6
    image_masks = {image["id"]: [] for image in labels["images"]}
    for annotation in labels["annotations"]:
        mask = decode_run_length_encoding(segmentation=annotation["segmentation"])
        x, y, width, height = annotation["bbox"]
        assert int(mask.sum()) == annotation["area"] >= MINIMUM_VISIBLE_OBJECT_PIXELS
        assert get_mask_bounding_box(mask=mask) == [x, y, x + width, y + height]
        image_masks[annotation["image_id"]].append(mask)

    # The visible parts of an image never overlap, and cover its shape pixels apart from the left out objects
    for image in labels["images"]:
        stacked_masks = np.stack(arrays=image_masks[image["id"]])
        assert int(stacked_masks.sum(axis=0).max()) == 1
        picture = cv2.imread(filename=str(tmp_path / "train" / "data" / image["file_name"]))
        colours, counts = np.unique(picture.reshape(-1, 3), axis=0, return_counts=True)
        shape_pixels = np.any(picture != colours[np.argmax(counts)], axis=-1)
        covered_pixels = stacked_masks.any(axis=0)
        assert not np.any(covered_pixels & ~shape_pixels)
        assert int(np.sum(shape_pixels & ~covered_pixels)) < 6 * MINIMUM_VISIBLE_OBJECT_PIXELS

    # The instance segmenter predicts the background as class 0
    label_mapping = json.loads((tmp_path / "label_mapping.json").read_text())
    assert label_mapping == {str(category_identifier): category_identifier
                             for category_identifier in SHAPE_CATEGORIES.values()}

    # The dataset loads the same masks at the image size, in the order of the annotations
    dataset = CocoDetectionDataset(data_directory=str(tmp_path / "train" / "data"),
                                   labels_file=str(tmp_path / "train" / "labels.json"), image_size=IMAGE_SIZE,
                                   keep_ratio=True, label_mode="multi_class",
                                   label_dictionary={int(category): class_index for category, class_index in
                                                     label_mapping.items()}, load_instance_masks=True, shuffle=False)
    sample = dataset[0]
    assert np.array_equal(sample["instance_masks"].numpy(), np.stack(arrays=image_masks[sample["sample_identifiers"]]))


def get_shape_centroid(frame: np.ndarray) -> np.ndarray:
    """
    Locates the shape of a synthetic sequence frame: the centroid of the pixels differing from the background, the
    background being the most frequent colour of the frame.

    Args:
        frame (np.ndarray): The [H, W, 3] uint8 BGR frame.

    Returns:
        np.ndarray: The [x, y] centroid of the shape in pixels.
    """
    colours, counts = np.unique(frame.reshape(-1, 3), axis=0, return_counts=True)
    shape_rows, shape_columns = np.nonzero(np.any(frame != colours[np.argmax(counts)], axis=-1))
    return np.array([shape_columns.mean(), shape_rows.mean()])


def test_created_sequence_dataset_moves_its_shapes_in_the_labelled_direction(tmp_path: Path) -> None:
    """
    Verifies the sequence classification mode of the generator: every split gets a manifest of sequences with their
    ordered frames, 3 to 6 frames whose number varies from sequence to sequence, the label mapping gives the four
    directions the classes 0 to 3, and the shape of every sequence moves in the direction of its label between its
    first and its last frame.

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

    # Every sequence draws its own number of frames in the requested range
    sequence_lengths = [len(sequence["frames"]) for sequence in labels["sequences"]]
    assert all(3 <= sequence_length <= 6 for sequence_length in sequence_lengths) and len(set(sequence_lengths)) > 1

    # Displacement of the shape between the first and the last frame, along the direction of the label
    category_names = {category_identifier: name for name, category_identifier in MOVEMENT_CATEGORIES.items()}
    for sequence in labels["sequences"]:
        frames = [cv2.imread(filename=str(tmp_path / "train" / "data" / frame_file))
                  for frame_file in sequence["frames"]]
        displacement = get_shape_centroid(frame=frames[-1]) - get_shape_centroid(frame=frames[0])
        direction_x, direction_y = MOVEMENT_DIRECTIONS[category_names[sequence["category_id"]]]
        assert displacement @ np.array([direction_x, direction_y]) > 5.0
        assert abs(displacement @ np.array([direction_y, direction_x])) < 1.0


def test_created_sequence_labelling_dataset_labels_every_frame(tmp_path: Path) -> None:
    """
    Verifies the sequence labelling mode of the generator: every sequence lists the shape category of every frame,
    2 to 6 frames whose number varies from sequence to sequence, the shape changes class at least once, the frames of
    the same class segment are identical and two consecutive segments differ, and the label mapping gives the 35
    shapes the classes 0 to 34.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_synthetic_dataset(output_folder=str(tmp_path), image_size=64,
                             split_sizes={"train": 10, "validation": 2, "test": 2}, maximum_objects=1,
                             minimum_frames=2, maximum_frames=6, number_workers=2, task="sequence_labelling")

    labels = json.loads((tmp_path / "train" / "labels.json").read_text())
    assert [category["name"] for category in labels["categories"]] == list(SHAPE_CATEGORIES)
    assert json.loads((tmp_path / "label_mapping.json").read_text()) == {
        str(category_identifier): category_identifier - 1 for category_identifier in SHAPE_CATEGORIES.values()}
    sequence_lengths = [len(sequence["frames"]) for sequence in labels["sequences"]]
    assert all(2 <= sequence_length <= 6 for sequence_length in sequence_lengths) and len(set(sequence_lengths)) > 1

    for sequence in labels["sequences"]:
        frame_category_ids = sequence["frame_category_ids"]
        assert len(frame_category_ids) == len(sequence["frames"])
        assert len(set(frame_category_ids)) >= 2 and set(frame_category_ids) <= set(SHAPE_CATEGORIES.values())

        # Identical frames within a class segment, different frames across a change of class
        frames = [cv2.imread(filename=str(tmp_path / "train" / "data" / frame_file))
                  for frame_file in sequence["frames"]]
        for frame_index in range(1, len(frames)):
            same_class = frame_category_ids[frame_index] == frame_category_ids[frame_index - 1]
            assert np.array_equal(frames[frame_index], frames[frame_index - 1]) == same_class


def count_frame_objects(frame: np.ndarray) -> int:
    """
    Counts the objects of a synthetic frame: the 8-connected groups of the pixels differing from the background colour,
    the most frequent colour of the frame.

    Args:
        frame (np.ndarray): The [H, W, 3] uint8 BGR frame.

    Returns:
        int: The number of objects of the frame.
    """
    colours, counts = np.unique(frame.reshape(-1, 3), axis=0, return_counts=True)
    object_mask = np.any(frame != colours[np.argmax(counts)], axis=-1).astype(np.uint8)
    number_components, _ = cv2.connectedComponents(image=object_mask, connectivity=8)
    return number_components - 1


def test_created_sequence_comparison_dataset_loses_an_object_in_the_after_member(tmp_path: Path) -> None:
    """
    Verifies the sequence comparison mode of the generator: every split gets a manifest of pairs with the ordered frames
    of their before and after members (2 to 4 frames each), the label mapping gives the two classes 0 and 1, every
    frame of a before member shows the same 2 to 3 separate objects, and every frame of its after member shows one
    object less for the pairs labelled 'loss_of_object' and the same number of objects otherwise.

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

    # Number of objects of every frame of both members, the same in every frame of a member
    for pair in labels["pairs"]:
        member_object_counts = {}
        for member_name in ["before", "after"]:
            assert 2 <= len(pair[member_name]) <= 4
            frame_object_counts = {count_frame_objects(frame=cv2.imread(filename=str(tmp_path / "train" / "data" /
                                                                                     frame_file)))
                                   for frame_file in pair[member_name]}
            assert len(frame_object_counts) == 1
            member_object_counts[member_name] = frame_object_counts.pop()
        assert 2 <= member_object_counts["before"] <= 3
        expected_lost_objects = 1 if pair["category_id"] == PAIR_CATEGORIES["loss_of_object"] else 0
        assert member_object_counts["before"] - member_object_counts["after"] == expected_lost_objects


def test_pair_frames_fail_loudly_when_objects_do_not_fit() -> None:
    """
    Verifies that a pair frame whose objects can never all fit without overlap stops the generation, instead of showing
    a wrong number of objects: two squares of 20 pixels cannot be 2 pixels apart in a 32 pixel frame.
    """
    square_mask = np.zeros(shape=(21, 21), dtype=np.uint8)
    square_mask[:20, :20] = 255
    shape_descriptors = [{"shape_name": "square", "mask": square_mask, "colour": np.full(shape=3, fill_value=255,
                                                                                         dtype=np.uint8)}] * 2

    with pytest.raises(expected_exception=RuntimeError, match="never fitted"):
        draw_shapes_frame(background=np.zeros(shape=(32, 32, 3), dtype=np.uint8), shape_descriptors=shape_descriptors,
                          random_generator=np.random.default_rng())


@pytest.mark.parametrize(argnames=("task", "minimum_frames", "maximum_frames"), argvalues=[
    ("sequence_classification", 1, 4), ("sequence_classification", 5, 4), ("frame_selection", 2, 6)])
def test_sequence_frame_range_is_checked(tmp_path: Path, task: str, minimum_frames: int, maximum_frames: int) -> None:
    """
    Verifies that a sequence dataset is refused when its sequences could have a single frame, which shows no movement,
    no change of class and no comparison, when a selection sequence could have fewer frames than wanted shapes, or
    when its frame range is empty.

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
    wanted shape selected in exactly one frame of every sequence, every frame showing a single shape, and the splits
    loading with the frame selection dataset, which checks every sequence.

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

    for sequence in labels["sequences"]:
        # One frame of every wanted shape, every other frame unselected, one shape per frame
        assert sorted(category_identifier for category_identifier in sequence["frame_category_ids"]
                      if category_identifier != SELECTION_CATEGORIES["unselected"]) == [2, 3, 4]
        for frame_file in sequence["frames"]:
            assert count_frame_objects(frame=cv2.imread(filename=str(tmp_path / "train" / "data" / frame_file))) == 1

    # Every sequence of the split passes the check of the frame selection dataset
    dataset = SequenceDataset(data_directory=str(tmp_path / "train" / "data"),
                              labels_file=str(tmp_path / "train" / "labels.json"), image_size=64, keep_ratio=True,
                              label_dictionary={int(key): value for key, value in label_mapping.items()},
                              sequence_settings={"sequence_length": 8, "batch_padding": "longest"},
                              label_type="selection",
                              selection_settings={"unselected_category": "unselected", "frames_per_class": 1})
    assert len(dataset) == 10 and dataset.unselected_class_index == 0
