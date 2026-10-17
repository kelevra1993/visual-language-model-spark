"""
Tests of the keypoint detection mode of vision_training/data/synthetic/create_synthetic_dataset.py, which creates the
synthetic shapes keypoint dataset of configuration_synthetic_shapes.yaml: the COCO labels.json holds every keypoint
category with its keypoint names, its 1-based skeleton and the rotation limit of its crops, every annotation holds
3 x K keypoint values with its number of labelled keypoints, the labelled keypoints lying in the image and in the box
of their object and the unlabelled ones at 0, the label mapping numbers the categories with keypoints from 1 and leaves
out the hollow distractors, and the keypoint categories and the KeypointDataset of the training pipeline read the
split.
"""

import json

from pathlib import Path

from vision_training.data.keypoints.keypoint_categories import read_keypoint_categories
from vision_training.data.keypoints.keypoint_dataset import KeypointDataset
from vision_training.data.synthetic.create_synthetic_dataset import (KEYPOINT_CATEGORIES,
                                                                     HOLLOW_DISTRACTOR_SHAPE_NAMES,
                                                                     KEYPOINT_MAXIMUM_ROTATION_DEGREES, KEYPOINT_NAMES,
                                                                     build_keypoint_label_mapping,
                                                                     create_synthetic_dataset)

IMAGE_SIZE = 256

# Distance a labelled keypoint may lie outside the box of its object: the keypoints of an outline lie on the edges of
# its pixels, and the keypoints of the scaled polygons a fraction of a pixel beyond them
BOX_TOLERANCE = 0.5


def test_created_keypoint_dataset_is_valid_coco_and_loads(tmp_path: Path) -> None:
    """
    Verifies that the created keypoint split lists every keypoint category with its keypoint names, its skeleton
    numbered from 1 within its keypoints and its rotation limit, that every annotation holds 3 values per keypoint of
    its category and counts its labelled keypoints, the labelled keypoints lying inside the image and the box of their
    object and the unlabelled ones at 0, that the label mapping gives the categories with keypoints the classes 1 to N
    and leaves out the hollow distractors, and that the keypoint categories and the keypoint dataset read the split.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_synthetic_dataset(output_folder=str(tmp_path), image_size=IMAGE_SIZE,
                             split_sizes={"train": 6, "validation": 2, "test": 2}, maximum_objects=6, minimum_frames=1,
                             maximum_frames=1, number_workers=2, task="keypoint_detection")
    labels = json.loads((tmp_path / "train" / "labels.json").read_text())
    assert len(labels["images"]) == 6 and len(labels["annotations"]) >= 6

    # Every category gives its keypoint names, its 1-based skeleton and the rotation limit of its crops
    assert [category["name"] for category in labels["categories"]] == list(KEYPOINT_CATEGORIES)
    keypoint_counts = {}
    for category in labels["categories"]:
        assert category["keypoints"] == KEYPOINT_NAMES[category["name"]]
        assert category["maximum_rotation_degrees"] == KEYPOINT_MAXIMUM_ROTATION_DEGREES[category["name"]]
        assert all(1 <= keypoint_number <= len(category["keypoints"]) for link in category["skeleton"]
                   for keypoint_number in link)
        keypoint_counts[category["id"]] = len(category["keypoints"])

    # Every annotation holds 3 values per keypoint, the labelled keypoints inside the image and the box of the object
    for annotation in labels["annotations"]:
        keypoint_values = annotation["keypoints"]
        assert len(keypoint_values) == 3 * keypoint_counts[annotation["category_id"]]
        x, y, width, height = annotation["bbox"]
        visibilities = keypoint_values[2::3]
        assert set(visibilities) <= {0, 1, 2} and annotation["num_keypoints"] == sum(1 for visibility in visibilities
                                                                                      if visibility > 0)
        for keypoint_x, keypoint_y, visibility in zip(keypoint_values[0::3], keypoint_values[1::3], visibilities):
            if visibility == 0:
                assert keypoint_x == keypoint_y == 0
                continue
            assert 0 <= keypoint_x <= IMAGE_SIZE and 0 <= keypoint_y <= IMAGE_SIZE
            assert x - BOX_TOLERANCE <= keypoint_x <= x + width + BOX_TOLERANCE
            assert y - BOX_TOLERANCE <= keypoint_y <= y + height + BOX_TOLERANCE

    # The categories with keypoints get the classes 1 to N in their order, the hollow distractors none
    label_mapping = json.loads((tmp_path / "label_mapping.json").read_text())
    distractor_identifiers = {str(KEYPOINT_CATEGORIES[f"hollow_{name}"]) for name in HOLLOW_DISTRACTOR_SHAPE_NAMES}
    assert label_mapping == build_keypoint_label_mapping() and not distractor_identifiers & set(label_mapping)
    assert list(label_mapping.values()) == list(range(1, len(KEYPOINT_CATEGORIES) - len(distractor_identifiers) + 1))
    label_dictionary = {int(category): class_index for category, class_index in label_mapping.items()}
    label_dictionary["background"] = 0

    # The training pipeline reads the categories, then one sample per object with a labelled keypoint
    keypoint_categories = read_keypoint_categories(labels_file=tmp_path / "train" / "labels.json",
                                                   label_dictionary=label_dictionary, flip_pairs_override={})
    assert len(keypoint_categories) == len(label_mapping)
    square_class = label_dictionary[KEYPOINT_CATEGORIES["square"]]
    assert keypoint_categories[square_class]["flip_pairs"] == [["top_left", "top_right"],
                                                               ["bottom_right", "bottom_left"]]
    maximum_keypoints = max(len(definition["keypoints"]) for definition in keypoint_categories.values())
    dataset = KeypointDataset(data_directory=str(tmp_path / "train" / "data"),
                              labels_file=str(tmp_path / "train" / "labels.json"), image_size=IMAGE_SIZE,
                              keep_ratio=True, label_dictionary=label_dictionary, maximum_keypoints=maximum_keypoints,
                              shuffle=False)
    assert len(dataset) == sum(1 for annotation in labels["annotations"] if annotation["num_keypoints"] > 0)
    sample = dataset[0]
    assert sample["keypoints"].shape == (maximum_keypoints, 2) and int(sample["visibilities"].gt(0).sum()) > 0
