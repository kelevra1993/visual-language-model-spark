"""
Tests of the keypoint detection mode of vision_training/data/synthetic/create_synthetic_dataset.py, which creates the
synthetic shapes keypoint dataset of configuration_synthetic_shapes.yaml: the COCO labels.json holds every keypoint
category with its keypoint names and its 1-based skeleton and nothing else, every annotation holds
3 x K keypoint values with its number of labelled keypoints, the labelled keypoints lying in the image and in the box
of their object and the unlabelled ones at 0, the label mapping numbers the categories with keypoints from 1 and leaves
out the distractors without keypoints, and the keypoint categories and the whole-image CocoDetectionDataset of the
training pipeline read the split.
"""

import json
import torch

from pathlib import Path

from vision_training.data.detection.coco_dataset import CocoDetectionDataset
from vision_training.data.keypoints.keypoint_categories import read_keypoint_categories
from vision_training.data.synthetic.create_synthetic_dataset import (KEYPOINT_CATEGORIES, KEYPOINT_NAMES,
                                                                     build_keypoint_label_mapping,
                                                                     create_synthetic_dataset)

IMAGE_SIZE = 256

# Distance a labelled keypoint may lie outside the box of its object: the keypoints of an outline lie on the edges of
# its pixels, and the keypoints of the scaled polygons a fraction of a pixel beyond them
BOX_TOLERANCE = 0.5


def test_created_keypoint_dataset_is_valid_coco_and_loads(tmp_path: Path) -> None:
    """
    Verifies that the created keypoint split lists every keypoint category with its keypoint names and its skeleton
    numbered from 1 within its keypoints, that every annotation holds 3 values per keypoint of its category and counts
    its labelled keypoints, the labelled keypoints lying inside the image and the box of their object and the
    unlabelled ones at 0, that the label mapping gives the categories with keypoints the classes 1 to N and leaves out
    the distractors without keypoints, and that the keypoint categories and the whole-image dataset read the split.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_synthetic_dataset(output_folder=str(tmp_path), image_size=IMAGE_SIZE,
                             split_sizes={"train": 6, "validation": 2, "test": 2}, maximum_objects=6, minimum_frames=1,
                             maximum_frames=1, number_workers=2, task="keypoint_detection")
    labels = json.loads((tmp_path / "train" / "labels.json").read_text())
    assert len(labels["images"]) == 6 and len(labels["annotations"]) >= 6

    # Every category gives its keypoint names and its 1-based skeleton, without any augmentation setting
    assert [category["name"] for category in labels["categories"]] == list(KEYPOINT_CATEGORIES)
    keypoint_counts = {}
    for category in labels["categories"]:
        assert category["keypoints"] == KEYPOINT_NAMES[category["name"]]
        assert set(category) == {"id", "name", "supercategory", "keypoints", "skeleton"}
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

    # The categories with keypoints get the classes 1 to N in their order, the distractors without keypoints none
    label_mapping = json.loads((tmp_path / "label_mapping.json").read_text())
    distractor_identifiers = {str(identifier) for name, identifier in KEYPOINT_CATEGORIES.items()
                              if not KEYPOINT_NAMES[name]}
    assert distractor_identifiers == {str(KEYPOINT_CATEGORIES[name]) for name in ["circle", "ellipse", "annulus",
                                                                                   "hollow_ellipse"]}
    assert label_mapping == build_keypoint_label_mapping() and not distractor_identifiers & set(label_mapping)
    assert list(label_mapping.values()) == list(range(1, len(KEYPOINT_CATEGORIES) - len(distractor_identifiers) + 1))
    label_dictionary = {int(category): class_index for category, class_index in label_mapping.items()}
    label_dictionary["background"] = 0

    # The training pipeline reads the categories, then one sample per image with the keypoints of its objects
    keypoint_categories = read_keypoint_categories(labels_file=tmp_path / "train" / "labels.json",
                                                   label_dictionary=label_dictionary)
    assert len(keypoint_categories) == len(label_mapping)
    square_class = label_dictionary[KEYPOINT_CATEGORIES["square"]]
    assert keypoint_categories[square_class]["keypoints"] == ["top_left", "top_right", "bottom_right", "bottom_left"]
    maximum_keypoints = max(len(definition["keypoints"]) for definition in keypoint_categories.values())
    dataset = CocoDetectionDataset(data_directory=str(tmp_path / "train" / "data"),
                                   labels_file=str(tmp_path / "train" / "labels.json"), image_size=IMAGE_SIZE,
                                   keep_ratio=True, label_mode="multi_class", label_dictionary=label_dictionary,
                                   load_keypoints=True, maximum_keypoints=maximum_keypoints, shuffle=False)
    assert len(dataset) == len(labels["images"])

    # Every image keeps its objects with a labelled keypoint, their keypoints unchanged at the size of the generator
    for sample in dataset:
        image_annotations = [annotation for annotation in labels["annotations"]
                             if annotation["image_id"] == sample["sample_identifiers"]
                             and str(annotation["category_id"]) in label_mapping and annotation["num_keypoints"] > 0]
        assert sample["keypoints"].shape == (len(image_annotations), maximum_keypoints, 3)
        assert sample["labels"].tolist() == [label_dictionary[annotation["category_id"]]
                                             for annotation in image_annotations]
        for object_keypoints, annotation in zip(sample["keypoints"], image_annotations):
            expected_keypoints = torch.tensor(data=annotation["keypoints"], dtype=torch.float32).reshape(-1, 3)
            assert torch.allclose(object_keypoints[:len(expected_keypoints)], expected_keypoints, atol=1e-4)
            assert not object_keypoints[len(expected_keypoints):].any()
