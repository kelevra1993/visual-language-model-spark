"""
Tests of vision_training/data/synthetic/create_synthetic_dataset.py: the synthetic detection, instance segmentation,
classification, semantic segmentation, sequence classification, sequence labelling, sequence comparison, frame
selection and self-supervised datasets are valid, load with the datasets of the framework and follow their labels,
the objects hidden by later ones being left out of the detection and instance segmentation labels. The hard images
draw log-uniform sizes from the largest shape and keep their small shapes half visible, and their instance segmentation
dataset loads. The largest IoU between two objects reaches the placement of every object, 0.3 by default. The shapes
stay inside their square, every hollow shape is its filled shape with one hole cut inside it, and the shapes whose
class is the label get the minimum size.
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
from vision_training.data.synthetic.create_synthetic_dataset import (CHANGE_CATEGORIES,
                                                                     HOLLOW_CLASS_NAMES, HOLLOW_DISTRACTOR_SHAPE_NAMES,
                                                                     HOLLOW_INNER_KEYPOINT_SHAPE_NAMES,
                                                                     HOLLOW_MINIMUM_SIZE,
                                                                     HARD_OBJECTS_MAXIMUM_SIZE_FRACTION,
                                                                     HARD_OBJECTS_MINIMUM_OBJECTS,
                                                                     HARD_OBJECTS_MINIMUM_SIZE,
                                                                     KEYPOINT_DRAWING_FUNCTIONS, KEYPOINT_NAMES,
                                                                     LABELLING_MINIMUM_OBJECTS,
                                                                     MAXIMUM_OBJECT_INTERSECTION_OVER_UNION,
                                                                     MINIMUM_SHAPE_SIZE,
                                                                     MINIMUM_VISIBLE_OBJECT_PIXELS,
                                                                     MOVEMENT_CATEGORIES, MOVEMENT_DIRECTIONS,
                                                                     PAIR_CATEGORIES,
                                                                     SELECTION_CATEGORIES, SHAPE_CATEGORIES,
                                                                     SHAPE_COUNT_CATEGORIES,
                                                                     SHAPE_DRAWING_FUNCTIONS,
                                                                     create_synthetic_dataset, draw_shape_descriptor,
                                                                     generate_counted_shapes_image,
                                                                     generate_shape_change_sequence,
                                                                     generate_image, get_mask_bounding_box,
                                                                     keep_visible_objects)

IMAGE_SIZE = 256


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


@pytest.mark.parametrize(argnames="shape_name", argvalues=["square", "triangle", "capsule"])
@pytest.mark.parametrize(argnames="size", argvalues=[MINIMUM_SHAPE_SIZE])
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


@pytest.mark.parametrize(argnames="keypoints", argvalues=[False])
def test_hollow_shapes_are_drawn_at_their_minimum_size(keypoints: bool) -> None:
    """
    Verifies that draw_shape_descriptor raises the size of every hollow class to HOLLOW_MINIMUM_SIZE when its range is
    smaller, 8 pixels being the smallest shapes of the 64 pixel frames of the sequence classification and comparison,
    where a hollow parallelogram leaves no hole in 16% of its drawings, so that the hole always fits inside the walls,
    for the shapes, while a filled class keeps the size of its range.

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
    for size in [33, MINIMUM_SHAPE_SIZE, 101]:
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


@pytest.mark.parametrize(argnames=("image_size", "expected_size_range"), argvalues=[(512, (MINIMUM_SHAPE_SIZE, 170))])
def test_image_shapes_get_the_minimum_size(monkeypatch: pytest.MonkeyPatch, image_size: int,
                                           expected_size_range: Tuple[int, int]) -> None:
    """
    Verifies that the shapes of the detection, segmentation and classification images are drawn from
    MINIMUM_SHAPE_SIZE pixels to a third of the image side.

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


@pytest.mark.parametrize(argnames=("image_size", "expected_size_range"), argvalues=[(512, (MINIMUM_SHAPE_SIZE, 102))])
def test_shape_change_shapes_get_the_minimum_size(monkeypatch: pytest.MonkeyPatch, image_size: int,
                                                  expected_size_range: Tuple[int, int]) -> None:
    """
    Verifies that the shapes of the sequence labelling frames measure MINIMUM_SHAPE_SIZE pixels to a fifth of the frame
    side.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture recording the shape descriptors drawn.
        image_size (int): The side of the frames.
        expected_size_range (Tuple[int, int]): The expected smallest and largest shape sizes.
    """
    shape_descriptor_recorder = CallRecorder(function=synthetic_dataset_module.draw_shape_descriptor)
    monkeypatch.setattr(target=synthetic_dataset_module, name="draw_shape_descriptor", value=shape_descriptor_recorder)
    generate_shape_change_sequence(image_size=image_size, number_frames=4, maximum_objects=8,
                                   random_generator=np.random.default_rng(seed=0))

    assert len(shape_descriptor_recorder.calls) >= 4 * LABELLING_MINIMUM_OBJECTS
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
    colour) with shape categories, the 'category_ids' of every sample being the categories visible in its mask, a label
    mapping keeping the category identifiers as classes after the background class 0, and the segmentation dataset
    loads it with its class names.

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

    # The instance segmentation keeps the box, the compressed encoding and the pixel count of the visible part of every
    # visible object
    instance_objects, instance_parts = keep_visible_objects(objects=objects, instance_map=instance_map,
                                                            keep_visible_parts=True)
    assert instance_objects == [(2, [0, 20, 10, 40]), (3, [0, 0, 30, 40])]
    assert isinstance(instance_parts[0]["segmentation"]["counts"], str)
    assert np.array_equal(decode_run_length_encoding(segmentation=instance_parts[0]["segmentation"]), instance_map == 2)
    assert [instance_part["area"] for instance_part in instance_parts] == [200, int(np.sum(instance_map == 3))]


def test_hard_images_draw_log_uniform_sizes_from_the_largest(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that a hard object detection image draws HARD_OBJECTS_MINIMUM_OBJECTS to maximum_objects shapes of one
    size each, from HARD_OBJECTS_MINIMUM_SIZE pixels to HARD_OBJECTS_MAXIMUM_SIZE_FRACTION of the image side, from the
    largest to the smallest so that the small shapes lie over the large ones, and log-uniform: about half of the shapes
    fall below the geometric mean of the size range, where a uniform draw in pixels would put about a fifth.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture recording the shape descriptors drawn.
    """
    random_generator = np.random.default_rng(seed=0)
    maximum_size = int(HARD_OBJECTS_MAXIMUM_SIZE_FRACTION * IMAGE_SIZE)
    all_sizes = []

    # Every image draws its shapes with a fixed size each, in decreasing order
    for _ in range(100):
        shape_descriptor_recorder = CallRecorder(function=synthetic_dataset_module.draw_shape_descriptor)
        monkeypatch.setattr(target=synthetic_dataset_module, name="draw_shape_descriptor",
                            value=shape_descriptor_recorder)
        generate_image(image_size=IMAGE_SIZE, maximum_objects=12, random_generator=random_generator,
                       hard_objects=True)
        sizes = [call["minimum_size"] for call in shape_descriptor_recorder.calls]
        assert all(call["minimum_size"] == call["maximum_size"] for call in shape_descriptor_recorder.calls)
        assert HARD_OBJECTS_MINIMUM_OBJECTS <= len(sizes) <= 12 and sizes == sorted(sizes, reverse=True)
        assert all(HARD_OBJECTS_MINIMUM_SIZE <= size <= maximum_size for size in sizes)
        all_sizes.extend(sizes)

    # Log-uniform sizes: half of them below the geometric mean of the range
    geometric_mean = np.sqrt(HARD_OBJECTS_MINIMUM_SIZE * maximum_size)
    assert 0.4 < np.mean(np.array(all_sizes) < geometric_mean) < 0.6


def test_hard_images_keep_small_shapes_half_visible() -> None:
    """
    Verifies that the objects of a hard image stay in the labels with half of the pixels of their whole shape visible
    when that is below MINIMUM_VISIBLE_OBJECT_PIXELS (a small shape), and with MINIMUM_VISIBLE_OBJECT_PIXELS visible
    pixels otherwise (a large shape partly covered by others), and are left out below that.
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


def test_hard_objects_are_refused_for_another_task(tmp_path: Path) -> None:
    """
    Verifies that the hard images are only drawn for the object detection and the instance segmentation, whose labels
    keep the box or the visible part of every object, and not for the keypoint detection, whose keypoints they do not
    draw.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    with pytest.raises(expected_exception=ValueError,
                       match="hard_objects only draws object detection and instance segmentation images"):
        create_synthetic_dataset(output_folder=str(tmp_path), image_size=IMAGE_SIZE, split_sizes={"train": 2},
                                 maximum_objects=6, minimum_frames=1, maximum_frames=1, number_workers=1,
                                 task="keypoint_detection", hard_objects=True)


def test_created_hard_instance_segmentation_dataset_encodes_small_and_large_visible_parts(tmp_path: Path) -> None:
    """
    Verifies the hard instance segmentation mode of the generator: the COCO labels hold 5 to 12 objects per image on
    average, every one with the compressed encoding of its visible part, whose pixel count is its area and whose tight
    box is its box, the parts of an image never overlapping, down to small shapes and up to large ones, some cut by the
    image border, and the instance segmentation dataset of the framework loads the same masks.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_synthetic_dataset(output_folder=str(tmp_path), image_size=IMAGE_SIZE, split_sizes={"train": 12},
                             maximum_objects=12, minimum_frames=1, maximum_frames=1, number_workers=2,
                             task="instance_segmentation", hard_objects=True)

    # Objects per image, and the compressed encoding, area and box of every visible part
    labels = json.loads((tmp_path / "train" / "labels.json").read_text())
    objects_per_image = np.bincount([annotation["image_id"] for annotation in labels["annotations"]],
                                    minlength=13)[1:]
    assert objects_per_image.max() <= 12 and objects_per_image.mean() >= HARD_OBJECTS_MINIMUM_OBJECTS
    image_masks = {image["id"]: [] for image in labels["images"]}
    for annotation in labels["annotations"]:
        assert isinstance(annotation["segmentation"]["counts"], str)
        mask = decode_run_length_encoding(segmentation=annotation["segmentation"])
        x, y, width, height = annotation["bbox"]
        assert int(mask.sum()) == annotation["area"] > 0
        assert get_mask_bounding_box(mask=mask) == [x, y, x + width, y + height]
        image_masks[annotation["image_id"]].append(mask)

    # Small and large visible parts, some reaching the image border, never overlapping within an image
    long_sides = [max(annotation["bbox"][2:]) for annotation in labels["annotations"]]
    assert min(long_sides) < 32 and max(long_sides) > IMAGE_SIZE // 3
    assert any(x == 0 or y == 0 or x + width == IMAGE_SIZE or y + height == IMAGE_SIZE
               for x, y, width, height in (annotation["bbox"] for annotation in labels["annotations"]))
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
        generate_image(image_size=IMAGE_SIZE, maximum_objects=12, random_generator=random_generator, hard_objects=True,
                       **keyword_arguments)
        assert len(placement_recorder.calls) >= HARD_OBJECTS_MINIMUM_OBJECTS
        assert all(call["maximum_intersection_over_union"] == expected_cap for call in placement_recorder.calls)


def count_overlapping_label_pairs(labels: Dict[str, Any]) -> int:
    """
    Counts the pairs of objects of the same image whose boxes in the COCO labels of a detection split overlap, so that
    a test tells whether the overlap cap of the generator reached the placement of the objects.

    Args:
        labels (Dict[str, Any]): The COCO labels, with the [x, y, width, height] 'bbox' of every annotation.

    Returns:
        int: The number of pairs of boxes of an image sharing a positive area.
    """
    image_boxes: Dict[int, List[List[float]]] = {}
    for annotation in labels["annotations"]:
        image_boxes.setdefault(annotation["image_id"], []).append(annotation["bbox"])
    overlapping_pairs = 0
    for boxes in image_boxes.values():
        for first_index, (first_x, first_y, first_width, first_height) in enumerate(boxes):
            for second_x, second_y, second_width, second_height in boxes[first_index + 1:]:
                overlap_width = min(first_x + first_width, second_x + second_width) - max(first_x, second_x)
                overlap_height = min(first_y + first_height, second_y + second_height) - max(first_y, second_y)
                overlapping_pairs += overlap_width > 0 and overlap_height > 0
    return overlapping_pairs


def test_created_detection_split_follows_its_overlap_cap(tmp_path: Path) -> None:
    """
    Verifies that create_synthetic_dataset passes its largest IoU between two objects down to the worker processes
    drawing the images: a hard object detection split drawn with a cap of 0 has no two overlapping boxes in its labels,
    while the same split drawn with the default cap of 0.3 has some.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    overlapping_pairs = {}
    for overlap_cap in [0.0, MAXIMUM_OBJECT_INTERSECTION_OVER_UNION]:
        output_folder = tmp_path / f"cap_{overlap_cap}"
        create_synthetic_dataset(output_folder=str(output_folder), image_size=IMAGE_SIZE, split_sizes={"test": 8},
                                 maximum_objects=12, minimum_frames=1, maximum_frames=1, number_workers=1,
                                 task="object_detection", hard_objects=True,
                                 maximum_object_intersection_over_union=overlap_cap)
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
    Verifies the sequence labelling mode of the generator: every sequence lists the change category of every frame,
    2 to 6 frames whose number varies from sequence to sequence, the first frame is labelled 'no_change', every frame
    shows 5 to 6 separate shapes, one more than the previous frame when labelled 'appearing_<class>', one less when
    labelled 'disappearing_<class>' and as many otherwise, and the label mapping gives the 71 changes the classes 0 to
    70.

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

    category_names = {category_identifier: name for name, category_identifier in CHANGE_CATEGORIES.items()}
    expected_count_changes = {"appearing": 1, "disappearing": -1, "no": 0}
    for sequence in labels["sequences"]:
        frame_category_ids = sequence["frame_category_ids"]
        assert len(frame_category_ids) == len(sequence["frames"])
        assert frame_category_ids[0] == CHANGE_CATEGORIES["no_change"]

        # Number of shapes of every frame within its bounds, changed by the label of the frame
        object_counts = [count_frame_objects(frame=cv2.imread(filename=str(tmp_path / "train" / "data" / frame_file)))
                         for frame_file in sequence["frames"]]
        assert all(LABELLING_MINIMUM_OBJECTS <= object_count <= 6 for object_count in object_counts)
        for frame_index in range(1, len(object_counts)):
            change_name = category_names[frame_category_ids[frame_index]].split("_")[0]
            assert object_counts[frame_index] - object_counts[frame_index - 1] == expected_count_changes[change_name]


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


@pytest.mark.parametrize(argnames=("image_size", "expected_size_range"), argvalues=[(512, (MINIMUM_SHAPE_SIZE, 102))])
def test_counted_shapes_get_the_sequence_labelling_size(monkeypatch: pytest.MonkeyPatch, image_size: int,
                                                        expected_size_range: Tuple[int, int]) -> None:
    """
    Verifies that the shapes of the self-supervised images get the sizes of the sequence labelling frames, so that the
    backbone they train sees the shapes of the frames it then initialises on.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture recording the shape descriptors drawn.
        image_size (int): The side of the images.
        expected_size_range (Tuple[int, int]): The expected smallest and largest shape sizes.
    """
    shape_descriptor_recorder = CallRecorder(function=synthetic_dataset_module.draw_shape_descriptor)
    monkeypatch.setattr(target=synthetic_dataset_module, name="draw_shape_descriptor", value=shape_descriptor_recorder)
    _, shape_names = generate_counted_shapes_image(image_size=image_size, maximum_objects=LABELLING_MINIMUM_OBJECTS,
                                                   random_generator=np.random.default_rng(seed=0))

    assert len(shape_descriptor_recorder.calls) == len(shape_names) == LABELLING_MINIMUM_OBJECTS
    assert all((call["minimum_size"], call["maximum_size"]) == expected_size_range
               for call in shape_descriptor_recorder.calls)


def test_created_self_supervised_dataset_counts_its_shapes(tmp_path: Path) -> None:
    """
    Verifies the self-supervised mode of the generator: every image shows 5 to maximum_objects separate shapes, its
    sample is labelled with its number of shapes and lists the category of every shape, the label mapping gives the 6
    shape counts the classes 0 to 5, and the classification dataset read by the self-supervised trainer loads it.

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

    # Number of drawn shapes of every image equal to its label and to its list of shape categories
    for sample in labels["samples"]:
        object_count = count_frame_objects(frame=cv2.imread(filename=str(tmp_path / "train" / "data" /
                                                                         sample["file_name"])))
        assert LABELLING_MINIMUM_OBJECTS <= object_count <= 7
        assert sample["category_id"] == SHAPE_COUNT_CATEGORIES[f"{object_count}_shapes"]
        assert len(sample["shape_category_ids"]) == object_count
        assert all(category_identifier in SHAPE_CATEGORIES.values()
                   for category_identifier in sample["shape_category_ids"])

    dataset = ClassificationDataset(data_directory=str(tmp_path / "train" / "data"),
                                    labels_file=str(tmp_path / "train" / "labels.json"), image_size=IMAGE_SIZE,
                                    keep_ratio=True, label_dictionary={int(category): class_index for category,
                                                                       class_index in label_mapping.items()},
                                    output_mode="multi_class")
    assert dataset.class_names == dict(enumerate(SHAPE_COUNT_CATEGORIES))
    assert sum(dataset.get_class_counts()) == 12


def test_created_multi_label_classification_dataset_lists_the_classes_of_its_shapes(tmp_path: Path) -> None:
    """
    Verifies the multi-label classification mode of the generator, which pretrains the backbone of the sequence
    labelling model: every image shows 5 to maximum_objects separate shapes drawn like a sequence labelling frame, its
    sample lists the sorted distinct categories of its shapes in 'category_ids' (several shapes may share a class), the
    label mapping gives the 35 shapes the classes 0 to 34, and the classification dataset loads it in the multi-label
    output mode with one positive per listed category.

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

    # Drawn shapes of every image, at least as many as its distinct categories, which are sorted and known
    for sample in labels["samples"]:
        object_count = count_frame_objects(frame=cv2.imread(filename=str(tmp_path / "train" / "data" /
                                                                         sample["file_name"])))
        assert LABELLING_MINIMUM_OBJECTS <= object_count <= 7
        assert "category_id" not in sample
        assert sample["category_ids"] == sorted(set(sample["category_ids"]))
        assert 1 <= len(sample["category_ids"]) <= object_count
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
                              sequence_settings={"maximum_sequence_length": 8}, label_type="selection",
                              split_name="train",
                              selection_settings={"unselected_category": "unselected", "frames_per_class": 1})
    assert len(dataset) == 10 and dataset.unselected_class_index == 0
