"""
Tests of the information panels written by the trainers for the input viewer ('view_input_images'), the viewer itself
being replaced by a recorder of its arguments: the object detection panels count the objects per class and name the
objects removed by the augmentations, the panels of a pair of sequences give the real frames and the shuffled order of
every member window, the self-supervised panels describe the transformations drawn for each view, and the keypoint
panels count the keypoints of the crop by visibility and those trained inside it, with the crop augmentation of the
training crop. The trainers are created without their datasets, only the attributes read by show_input_samples being
set.
"""

import random
import pytest
import torch

from typing import Any, Dict, List

import vision_training.trainers.detection_trainer as detection_trainer_module
import vision_training.trainers.keypoint_trainer as keypoint_trainer_module
import vision_training.trainers.self_supervised_trainer as self_supervised_trainer_module
import vision_training.trainers.sequence_classification_trainer as sequence_classification_trainer_module
from vision_training.data.augmentations import AUGMENTATION_TYPES, MAXIMUM_TRANSLATION_FRACTION, DataAugmenter
from vision_training.data.view_augmentations import ViewAugmenter
from vision_training.utilities.visualization import KEYPOINT_SIDE_COLOURS
from vision_training.trainers.detection_trainer import DetectionTrainer
from vision_training.trainers.keypoint_trainer import KeypointTrainer, build_keypoint_counts
from vision_training.trainers.self_supervised_trainer import SelfSupervisedTrainer
from vision_training.trainers.sequence_comparison_trainer import SequenceComparisonTrainer

# Side of the square images of the tests
IMAGE_SIZE = 96


class ViewerRecorder:
    """
    Stands in for view_input_data in the trainer modules, keeping the arguments of its last call so that the tests
    read the panel lines without opening any window.
    """

    def __init__(self) -> None:
        """
        Starts without any recorded call.
        """
        self.arguments: Dict[str, Any] = {}

    def __call__(self, **arguments: Any) -> bool:
        """
        Records the arguments of the viewer call.

        Args:
            **arguments (Any): The keyword arguments given to view_input_data.

        Returns:
            bool: True, which stops the viewer.
        """
        self.arguments = arguments
        return True


def build_augmenter(enabled_types: List[str], shuffle_probability: float = 0.0) -> DataAugmenter:
    """
    Builds an augmenter applying exactly the given types to every sample, and shuffling the frames of every window
    with the given probability.

    Args:
        enabled_types (List[str]): The augmentation types to activate, each with probability 1.
        shuffle_probability (float): The probability of shuffling the frames of a window, 0 turning the shuffling
            off. Defaults to 0.0.

    Returns:
        DataAugmenter: The augmenter.
    """
    return DataAugmenter(augmentation_configuration={
        "activated": True, "probability": 1.0, "independent_frames": False,
        "shuffle_frames": {"activated": shuffle_probability > 0.0, "probability": shuffle_probability},
        "types": {name: {"activated": name in enabled_types, "probability": 1.0} for name in AUGMENTATION_TYPES}})


def test_detection_panels_count_the_objects_and_name_the_removed_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the panels of the object detection input viewer: the training image translated by the largest shift
    along its width keeps one object and names the one pushed out of the image, while the validation image counts its
    objects per class without augmentation.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture forcing the translation and replacing the viewer.
    """
    trainer = DetectionTrainer.__new__(DetectionTrainer)
    trainer.data_augmenter = build_augmenter(enabled_types=["random_translation"])
    trainer.label_dictionary, trainer.input_image_size = {"cat": 1, "dog": 2}, IMAGE_SIZE
    viewer_recorder = ViewerRecorder()
    monkeypatch.setattr(target=detection_trainer_module, name="view_input_data", value=viewer_recorder)
    monkeypatch.setattr(target=random, name="choice", value=lambda sequence: "x")
    monkeypatch.setattr(target=random, name="randint", value=lambda low, high: high)
    maximum_shift = int(IMAGE_SIZE * MAXIMUM_TRANSLATION_FRACTION)

    # Two objects in both images, the dog at the right border leaving the training image
    batch = {"images": torch.zeros(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE)),
             "bounding_boxes": [torch.tensor(data=[[10.0, 10.0, 30.0, 30.0], [80.0, 40.0, 95.0, 60.0]])],
             "labels": [torch.tensor(data=[1, 2])]}
    training_batch = trainer.augment_batch(batch=batch)
    assert trainer.show_input_samples(training_batch=training_batch, validation_batch=batch, sample_index=0)

    training_lines, validation_lines = viewer_recorder.arguments["panel_lines"]
    assert training_lines == ["Objects : 1 (cat 1)", "", "Augmentations", "Geometric :",
                              f"  random_translation : x +{maximum_shift} pixels", "Photometric :", "  none",
                              "Removed objects : 1 (dog)"]
    assert validation_lines == ["Objects : 2 (cat 1, dog 1)", "", "Augmentations : none (validation sample)"]
    assert viewer_recorder.arguments["label_names"] == {1: "cat", 2: "dog"}


def test_pair_panels_give_the_real_frames_and_the_shuffled_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the panels of a pair of sequences in the input viewer: the real frames of both member windows, the
    augmentations of the training pair and the original frames shown at every position of each shuffled member window,
    the validation pair being neither augmented nor shuffled.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the viewer.
    """
    torch.manual_seed(seed=0)
    random.seed(0)
    trainer = SequenceComparisonTrainer.__new__(SequenceComparisonTrainer)
    trainer.data_augmenter = build_augmenter(enabled_types=[], shuffle_probability=1.0)
    trainer.sequence_settings = {"sequence_length": 4}
    trainer.class_names, trainer.output_mode = ["same", "changed"], "multi_class"
    viewer_recorder = ViewerRecorder()
    monkeypatch.setattr(target=sequence_classification_trainer_module, name="view_input_data", value=viewer_recorder)

    # A before window of 3 real frames and an after window of 2, the frame of position i filled with i
    frames = torch.arange(8, dtype=torch.float32)[None, :, None, None, None].expand(1, 8, 3, 8, 8).clone()
    batch = {"images": frames, "labels": torch.tensor(data=[1]),
             "sequence_mask": torch.tensor(data=[[True, True, True, False, True, True, False, False]])}
    training_batch = trainer.augment_batch(batch=batch)
    assert trainer.show_input_samples(training_batch=training_batch, validation_batch=batch, sample_index=0)

    training_lines, validation_lines = viewer_recorder.arguments["panel_lines"]
    assert training_lines[:7] == ["Class : changed", "Real frames : 3 + 2", "", "Augmentations", "Geometric :",
                                  "  none", "Photometric :"]
    assert validation_lines == ["Class : changed", "Real frames : 3 + 2", "",
                                "Augmentations : none (validation sample)"]

    # The frame order names the original frames of each member window, as the shuffled frames show them
    frame_order_line = training_lines[-1]
    assert frame_order_line.startswith("Frame order : shuffled, showing frames ")
    member_orders = [[int(frame_index) for frame_index in member_order.split(", ")]
                     for member_order in frame_order_line.split("showing frames ")[1].split(" | ")]
    assert sorted(member_orders[0]) == [0, 1, 2] and sorted(member_orders[1]) == [4, 5]
    assert member_orders == [training_batch["images"][0, [0, 1, 2], 0, 0, 0].to(dtype=torch.int64).tolist(),
                             training_batch["images"][0, [4, 5], 0, 0, 0].to(dtype=torch.int64).tolist()]


def test_self_supervised_panels_describe_the_transformations_of_each_view(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the panels of the self-supervised input viewer: both views of the training image are written with the
    class of the image and the transformations drawn for them, the whole image being cropped when every other
    transformation is turned off.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the viewer.
    """
    trainer = SelfSupervisedTrainer.__new__(SelfSupervisedTrainer)
    trainer.view_augmenter = ViewAugmenter(view_augmentation_configuration={
        "random_resized_crop": {"scale": [1.0, 1.0], "aspect_ratio": [1.0, 1.0]},
        "left_right_flip": {"probability": 0.0}, "up_down_flip": {"probability": 0.0},
        "rotation": {"probability": 0.0, "maximum_degrees": 180.0},
        "colour_jitter": {"probability": 0.0, "brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1},
        "grayscale": {"probability": 0.0},
        "gaussian_blur": {"probability": 0.0, "kernel_size": 5, "sigma": [0.1, 2.0]}}, image_size=32)
    trainer.class_names, trainer.input_image_size = ["circle", "square"], 32
    viewer_recorder = ViewerRecorder()
    monkeypatch.setattr(target=self_supervised_trainer_module, name="view_input_data", value=viewer_recorder)

    batch = {"images": torch.rand(size=(2, 3, 32, 32)) * 255.0, "labels": torch.tensor(data=[0, 1])}
    assert trainer.show_input_samples(training_batch=batch, validation_batch=batch, sample_index=1)

    assert viewer_recorder.arguments["window_titles"] == ["View 1", "View 2"]
    assert viewer_recorder.arguments["panel_lines"] == [
        ["Class : square", "", "View transformations",
         "  random_resized_crop : 32x32 pixels at x 0, y 0 (100 % of the area)"]] * 2


def test_keypoint_panels_count_the_keypoints_and_describe_the_crop(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the panels of the keypoint detection input viewer for a figure of 4 keypoints (a visible head, an
    occluded left hand, an unlabelled right hand and a visible foot below its box): both crops count the keypoints by
    visibility, the training crop, cut from a region 1.5 times larger and flipped, trains the foot that falls outside
    the validation crop and keeps the names of the keypoints (the left hand mirrored to the right half of the crop
    stays in its channel, with its visibility), and its panel writes the crop augmentation between the augmentation title and the photometric
    augmentations. Both crops are drawn at the display size of the viewer with the head at its place in the crop.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the viewer.
    """
    torch.manual_seed(seed=0)
    trainer = KeypointTrainer.__new__(KeypointTrainer)
    trainer.crop_height, trainer.crop_width, trainer.box_enlargement, trainer.heatmap_sigma = 64, 32, 1.25, 2.0
    trainer.keypoint_categories = {1: {"name": "figure", "keypoints": ["head", "left_hand", "right_hand", "foot"],
                                       "skeleton": [[0, 1], [0, 2], [0, 3]]}}
    trainer.keypoint_counts = build_keypoint_counts(keypoint_categories=trainer.keypoint_categories)
    trainer.crop_augmentation_configuration = {"scale_range": [1.5, 1.5], "maximum_rotation_degrees": 180.0,
                                               "rotation_probability": 0.0, "flip_probability": 1.0,
                                               "maximum_shift_fraction": 0.0}
    trainer.data_augmenter, trainer.input_image_size = build_augmenter(enabled_types=[]), IMAGE_SIZE
    viewer_recorder = ViewerRecorder()
    monkeypatch.setattr(target=keypoint_trainer_module, name="view_input_data", value=viewer_recorder)

    # One figure whose 24 x 56 box gives a 35 x 70 validation region centred on (48, 48), from y = 13 to y = 83,
    # the foot at y = 90 lying outside it but inside the 52.5 x 105 training region
    batch = {"images": torch.zeros(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE)),
             "bounding_boxes": torch.tensor(data=[[36.0, 20.0, 60.0, 76.0]]), "labels": torch.tensor(data=[1]),
             "keypoints": torch.tensor(data=[[[48.0, 30.0], [40.0, 50.0], [56.0, 50.0], [48.0, 90.0]]]),
             "visibilities": torch.tensor(data=[[2, 1, 0, 2]])}
    training_batch = trainer.augment_batch(batch=batch)

    # The flipped left hand, 8 pixels left of the region centre, lands 8 x 32 / 52.5 pixels right of the crop centre
    # without changing channel nor visibility
    assert torch.equal(input=training_batch["crop_visibilities"], other=batch["visibilities"])
    assert abs(float(training_batch["crop_keypoints"][0, 1, 0]) - (16.0 + 8.0 * 32.0 / 52.5)) < 1e-4
    assert trainer.show_input_samples(training_batch=training_batch, validation_batch=batch, sample_index=0)

    training_lines, validation_lines = viewer_recorder.arguments["panel_lines"]
    target_lines = ["Class : figure", "Keypoints : 2 visible, 1 occluded, 1 unlabelled, of 4"]
    colour_line = "Colours : left blue, right orange, others yellow, rings occluded"
    # Without shift the drawn zeros may be negative, written without a minus sign
    assert training_lines == target_lines + ["Trained : 3 labelled keypoints inside the crop", colour_line, "",
                                             "Augmentations", "Crop :", "  scale : 1.50", "  rotation : 0.0 degrees",
                                             "  flip : yes", "  shift : +0.00 width, +0.00 height",
                                             "Geometric :", "  none", "Photometric :", "  none"]
    assert validation_lines == target_lines + ["Trained : 2 labelled keypoints inside the crop", colour_line, "",
                                               "Augmentations : none (validation sample)"]

    # Both crops drawn at the display size, 1.5 times the 64 x 32 crop, the head on the vertical axis of the crop at
    # y = 32 - 18 x 32 / 52.5 in the training crop and y = 32 - 18 x 32 / 35 in the validation crop
    training_image, validation_image = viewer_recorder.arguments["images"]
    assert training_image.shape == validation_image.shape == (3, IMAGE_SIZE, IMAGE_SIZE // 2)
    assert training_image[:, 31, 24].tolist() == list(KEYPOINT_SIDE_COLOURS["Centre"])
    assert validation_image[:, 23, 24].tolist() == list(KEYPOINT_SIDE_COLOURS["Centre"])
    assert viewer_recorder.arguments["window_titles"] == ["Training", "Validation"]
    assert viewer_recorder.arguments["bounding_boxes"] is None
