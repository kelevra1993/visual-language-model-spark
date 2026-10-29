"""
Tests of the information panels written by the trainers for the input viewer ('view_input_images'), the viewer itself
being replaced by a recorder of its arguments: the object detection panels count the objects per class and name the
objects removed by the augmentations, the panels of a pair of sequences give the real frames and the shuffled order of
every member, the self-supervised panels describe the augmentations drawn for each view, the solarization reaching the
second view only, the self-supervised batch mosaic describes every batch of the grouped batch sampler (grouped or
random) and is not shown for random batches, and the keypoint panels count the objects per class and their keypoints
by visibility, the flipped training image moving the keypoints without renaming them. The trainers are created without
their datasets, only the attributes read by show_input_samples being set.
"""

import random
import pytest
import torch

from pathlib import Path
from typing import Any, Dict, List

import vision_training.trainers.detection_trainer as detection_trainer_module
import vision_training.trainers.keypoint_trainer as keypoint_trainer_module
import vision_training.trainers.self_supervised_trainer as self_supervised_trainer_module
import vision_training.trainers.sequence_classification_trainer as sequence_classification_trainer_module
from tests.data.classification.test_classification_dataset import write_split
from tests.data.test_grouped_batch_sampler import SITE_CAMERA_IMAGES
from vision_training.data.augmentations import AUGMENTATION_TYPES, SUPERVISED_AUGMENTATION_STRENGTHS, DataAugmenter
from vision_training.data.loaders import get_classification_dataloaders
from vision_training.utilities.visualization import KEYPOINT_SIDE_COLOURS, build_class_colours
from vision_training.trainers.detection_trainer import DetectionTrainer
from vision_training.trainers.keypoint_trainer import KeypointTrainer
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


def build_augmenter(enabled_types: List[str], shuffle_probability: float = 0.0, self_supervised: bool = False,
                    sequence_augmentation: bool = False) -> DataAugmenter:
    """
    Builds an augmenter applying exactly the given types to every sample, and shuffling the frames of every sequence
    with the given probability.

    Args:
        enabled_types (List[str]): The augmentation types to activate, each with probability 1.
        shuffle_probability (float): The probability of shuffling the frames of a sequence, 0 turning the shuffling
            off. Defaults to 0.0.
        self_supervised (bool): Whether the augmenter draws the views of the self-supervised task with the
            self-supervised strengths. Defaults to False.
        sequence_augmentation (bool): Whether the augmenter belongs to a sequence task, the only tasks reading the
            frame shuffling. Defaults to False.

    Returns:
        DataAugmenter: The augmenter.
    """
    types = {name: {"activated": name in enabled_types, "probability": 1.0} for name in AUGMENTATION_TYPES}
    types["random_black_out"]["grid_size"] = 8
    return DataAugmenter(augmentation_configuration={
        "activated": True, "probability": 1.0, "independent_frames": False,
        "shuffle_frames": {"activated": shuffle_probability > 0.0, "probability": shuffle_probability},
        "types": types}, self_supervised=self_supervised, sequence_augmentation=sequence_augmentation)


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
    maximum_shift = int(IMAGE_SIZE * SUPERVISED_AUGMENTATION_STRENGTHS["maximum_translation_fraction"])

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
    Verifies the panels of a pair of sequences in the input viewer: the real frames of both members, the
    augmentations of the training pair and the original frames shown at every position of each shuffled member, the
    validation pair being neither augmented nor shuffled.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the viewer.
    """
    torch.manual_seed(seed=0)
    random.seed(0)
    trainer = SequenceComparisonTrainer.__new__(SequenceComparisonTrainer)
    trainer.data_augmenter = build_augmenter(enabled_types=[], shuffle_probability=1.0, sequence_augmentation=True)
    trainer.sequence_settings = {"maximum_sequence_length": 4}
    trainer.class_names, trainer.output_mode = ["same", "changed"], "multi_class"
    viewer_recorder = ViewerRecorder()
    monkeypatch.setattr(target=sequence_classification_trainer_module, name="view_input_data", value=viewer_recorder)

    # A before member of 3 real frames and an after member of 2, the frame of position i filled with i
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

    # The frame order names the original frames of each member, as the shuffled frames show them
    frame_order_line = training_lines[-1]
    assert frame_order_line.startswith("Frame order : shuffled, showing frames ")
    member_orders = [[int(frame_index) for frame_index in member_order.split(", ")]
                     for member_order in frame_order_line.split("showing frames ")[1].split(" | ")]
    assert sorted(member_orders[0]) == [0, 1, 2] and sorted(member_orders[1]) == [4, 5]
    assert member_orders == [training_batch["images"][0, [0, 1, 2], 0, 0, 0].to(dtype=torch.int64).tolist(),
                             training_batch["images"][0, [4, 5], 0, 0, 0].to(dtype=torch.int64).tolist()]


def test_self_supervised_panels_describe_the_augmentations_of_each_view(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the panels of the self-supervised input viewer: both views of the training image, drawn by the data
    augmentation with the self-supervised strengths, are written with the class of the image and the augmentations
    drawn for them, the flip reaching both views and the solarization, drawn for the second view only, the second one.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the viewer.
    """
    trainer = SelfSupervisedTrainer.__new__(SelfSupervisedTrainer)
    trainer.data_augmenter = build_augmenter(enabled_types=["random_left_right_flip", "random_solarize"],
                                             self_supervised=True)
    trainer.class_names, trainer.input_image_size = ["circle", "square"], 32
    viewer_recorder = ViewerRecorder()
    monkeypatch.setattr(target=self_supervised_trainer_module, name="view_input_data", value=viewer_recorder)

    batch = {"images": torch.rand(size=(2, 3, 32, 32)) * 255.0, "labels": torch.tensor(data=[0, 1])}
    assert trainer.show_input_samples(training_batch=batch, validation_batch=batch, sample_index=1)

    # Both views flipped and the second one solarized too, each panel listing the augmentations of its view
    assert viewer_recorder.arguments["window_titles"] == ["View 1", "View 2"]
    panel_start = ["Class : square", "", "View augmentations", "Geometric :", "  random_left_right_flip",
                   "Photometric :"]
    assert viewer_recorder.arguments["panel_lines"] == [panel_start + ["  none"],
                                                        panel_start + ["  random_solarize : threshold 191.25 levels"]]
    flipped_image = batch["images"][1].flip(dims=(2,))
    solarized_image = torch.where(condition=flipped_image < 191.25, input=flipped_image, other=255.0 - flipped_image)
    first_view, second_view = viewer_recorder.arguments["images"]
    assert torch.equal(input=first_view, other=flipped_image)
    assert torch.equal(input=second_view, other=solarized_image)


def build_grouped_training_loader(tmp_path: Path, activated: bool) -> torch.utils.data.DataLoader:
    """
    Builds the self-supervised training loader of the images of SITE_CAMERA_IMAGES, every image holding its site and
    camera in its groups, with batches of 8 images made of blocks of 4 images of one camera within one site: 3 grouped
    batches and 1 random batch per pass.

    Args:
        tmp_path (Path): The pytest temporary directory, receiving the splits.
        activated (bool): Whether the grouped batches are on, the batches being random otherwise.

    Returns:
        torch.utils.data.DataLoader: The training loader.
    """
    # A training split whose images hold their site and camera, and validation and test splits without groups
    categories = [{"id": 1, "name": "circle"}]
    group_values = [group_value for group_value, number_images in SITE_CAMERA_IMAGES.items()
                    for _ in range(number_images)]
    write_split(split_folder=tmp_path / "train", categories=categories,
                samples=[{"id": sample_index + 1, "file_name": f"{sample_index + 1}.png", "category_id": 1,
                          "groups": {"site": site, "camera": camera}}
                         for sample_index, (site, camera) in enumerate(group_values)])
    for split_name in ["validation", "test"]:
        write_split(split_folder=tmp_path / split_name, categories=categories,
                    samples=[{"id": sample_identifier, "file_name": f"{sample_identifier}.png", "category_id": 1}
                             for sample_identifier in range(1, 9)])

    # Self-supervised loaders with blocks of 4 images of one camera within batches of 8 images of one site
    experiment_configuration = {f"{split_name}_data_folder": str(tmp_path / split_name)
                                for split_name in ["train", "validation", "test"]}
    model_configuration = {"Data": {"image_settings": {"size": 32, "keep_ratio": True}, "label_dictionary": {1: 0},
                                    "grouped_batches": {"activated": activated, "block_key": "camera",
                                                        "images_per_block": 4, "batch_key": "site"}}}
    training_loader, _, _ = get_classification_dataloaders(experiment_configuration=experiment_configuration,
                                                           model_configuration=model_configuration, batch_size=8,
                                                           self_supervised=True)
    return training_loader


def test_self_supervised_batch_mosaic_describes_the_grouped_batches(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the batch mosaic of the self-supervised input viewer over a pass of the grouped batch sampler: every
    batch is shown in a 'Training Batch' window with its images in the order of the batch, the stop asked in the window
    being returned, a grouped batch titled as such with its site and every block captioned with its camera, and the
    random batch of the leftover pool counting the images of every site and camera and captioning every image with its
    site and camera; with the grouped batches off, nothing is shown.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the mosaic and its window.
    """
    # A trainer with only the grouped training loader and its keys, the mosaic and its window replaced by recorders
    trainer = SelfSupervisedTrainer.__new__(SelfSupervisedTrainer)
    trainer.train_dataloader = build_grouped_training_loader(tmp_path=tmp_path / "grouped", activated=True)
    trainer.data_configuration = {"grouped_batches": {"block_key": "camera", "batch_key": "site"}}
    mosaic_recorder, window_recorder = ViewerRecorder(), ViewerRecorder()
    monkeypatch.setattr(target=self_supervised_trainer_module, name="build_batch_mosaic", value=mosaic_recorder)
    monkeypatch.setattr(target=self_supervised_trainer_module, name="show_viewer_windows", value=window_recorder)

    # Every batch of a pass shown in its window, with its images and the groups read back from the dataset
    torch.manual_seed(seed=0)
    batch_kinds = []
    for batch_index, batch in enumerate(trainer.train_dataloader):
        assert trainer.show_input_batch(training_batch=batch, batch_index=batch_index)
        assert window_recorder.arguments["window_titles"] == ["Training Batch"]
        arguments = mosaic_recorder.arguments
        assert torch.equal(input=arguments["images"], other=batch["images"]) and arguments["block_size"] == 4
        image_groups = [trainer.train_dataloader.dataset.sample_records[sample_identifier]["groups"]
                        for sample_identifier in batch["sample_identifiers"].tolist()]
        sites_cameras = [(groups["site"], groups["camera"]) for groups in image_groups]
        assert arguments["lines"][0] == "Pass : 3 grouped and 1 random batches of 8 images"

        # A grouped batch: one site, and one camera per block
        if arguments["title"] == f"Training Batch {batch_index + 1} / 4 : Grouped":
            batch_kinds.append("grouped")
            assert arguments["lines"][2:] == [f"Batch key 'site' : {sites_cameras[0][0]}",
                                              "Block key 'camera' : 2 blocks of 4 images"]
            assert arguments["block_captions"] == [sites_cameras[0][1], sites_cameras[4][1]]
            assert arguments["image_captions"] is None
            continue

        # The random batch: the images of every value, and the values of every image
        batch_kinds.append("random")
        assert arguments["title"] == f"Training Batch {batch_index + 1} / 4 : Random (Leftover Pool)"
        number_sites = len({site for site, _ in sites_cameras})
        number_cameras = len({camera for _, camera in sites_cameras})
        assert arguments["lines"][2] == f"Batch key 'site' : {number_sites} values"
        assert arguments["lines"][4] == f"Block key 'camera' : {number_cameras} values"
        assert arguments["block_captions"] == ["Images 1 to 4", "Images 5 to 8"]
        assert arguments["image_captions"] == [f"{site} / {camera}" for site, camera in sites_cameras]
    assert sorted(batch_kinds) == ["grouped", "grouped", "grouped", "random"]

    # Random batches without the grouped batch sampler: nothing shown
    mosaic_recorder.arguments = {}
    trainer.train_dataloader = build_grouped_training_loader(tmp_path=tmp_path / "random", activated=False)
    assert not trainer.show_input_batch(training_batch=next(iter(trainer.train_dataloader)), batch_index=0)
    assert mosaic_recorder.arguments == {}


def test_keypoint_panels_count_the_keypoints_of_the_flipped_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the panels of the keypoint detection input viewer for an image holding a figure of 4 keypoints (a visible
    head, an occluded left hand, an unlabelled right hand and a visible foot) in its left half: both panels count the
    objects per class and the keypoints by visibility, the training image flipped from left to right moves the box and
    the keypoints to the right half without renaming them (the left hand stays in its channel with its visibility, the
    unlabelled keypoint stays at 0), and both images are drawn at their own resolution, the head disc at its place and
    the left hand coloured as a left keypoint in the half of the image holding the figure.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the viewer.
    """
    trainer = KeypointTrainer.__new__(KeypointTrainer)
    trainer.keypoint_categories = {1: {"name": "figure", "keypoints": ["head", "left_hand", "right_hand", "foot"],
                                       "skeleton": [[0, 1], [0, 2], [0, 3]]}}
    # Label dictionary keyed by the raw category identifier of the COCO labels, the class names naming the objects
    trainer.label_dictionary, trainer.class_names = {7: 1}, {1: "figure"}
    trainer.class_colours = build_class_colours(number_classes=2)
    trainer.data_augmenter = build_augmenter(enabled_types=["random_left_right_flip"])
    trainer.input_image_size = IMAGE_SIZE
    viewer_recorder = ViewerRecorder()
    monkeypatch.setattr(target=keypoint_trainer_module, name="view_input_data", value=viewer_recorder)

    # One figure in the left half of the image, the flip mirroring every x to 96 - x
    batch = {"images": torch.zeros(size=(1, 3, IMAGE_SIZE, IMAGE_SIZE)),
             "bounding_boxes": [torch.tensor(data=[[18.0, 20.0, 42.0, 94.0]])], "labels": [torch.tensor(data=[1])],
             "keypoints": [torch.tensor(data=[[[30.0, 30.0, 2.0], [22.0, 50.0, 1.0], [0.0, 0.0, 0.0],
                                               [30.0, 90.0, 2.0]]])]}
    training_batch = trainer.augment_batch(batch=batch)
    assert torch.equal(input=training_batch["bounding_boxes"][0], other=torch.tensor(data=[[54.0, 20.0, 78.0, 94.0]]))
    assert torch.equal(input=training_batch["keypoints"][0],
                       other=torch.tensor(data=[[[66.0, 30.0, 2.0], [74.0, 50.0, 1.0], [0.0, 0.0, 0.0],
                                                 [66.0, 90.0, 2.0]]]))
    assert trainer.show_input_samples(training_batch=training_batch, validation_batch=batch, sample_index=0)

    # Same objects and keypoints in both panels, the training panel naming the flip
    training_lines, validation_lines = viewer_recorder.arguments["panel_lines"]
    target_lines = ["Objects : 1 (figure 1)", "Keypoints : 2 visible, 1 occluded, 1 unlabelled",
                    "Colours : left blue, right orange, others yellow, rings occluded, skeleton in the class colour"]
    assert training_lines == target_lines + ["", "Augmentations", "Geometric :", "  random_left_right_flip",
                                             "Photometric :", "  none", "Removed objects : none"]
    assert validation_lines == target_lines + ["", "Augmentations : none (validation sample)"]

    # Both images drawn at their own resolution, the head disc at its pixel, the left hand ring in the half of the
    # image holding the figure
    training_image, validation_image = viewer_recorder.arguments["images"]
    assert training_image.shape == validation_image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert training_image[:, 30, 66].tolist() == list(KEYPOINT_SIDE_COLOURS["Centre"])
    assert validation_image[:, 30, 30].tolist() == list(KEYPOINT_SIDE_COLOURS["Centre"])
    left_colour = torch.tensor(data=KEYPOINT_SIDE_COLOURS["Left"], dtype=torch.float32)[:, None, None]
    training_left_columns = torch.nonzero(input=(training_image == left_colour).all(dim=0))[:, 1]
    validation_left_columns = torch.nonzero(input=(validation_image == left_colour).all(dim=0))[:, 1]
    assert len(training_left_columns) > 0 and bool((training_left_columns > IMAGE_SIZE // 2).all())
    assert len(validation_left_columns) > 0 and bool((validation_left_columns < IMAGE_SIZE // 2).all())

    # The viewer draws the box of every image, captioned with the class name
    assert viewer_recorder.arguments["window_titles"] == ["Training", "Validation"]
    assert torch.equal(input=viewer_recorder.arguments["bounding_boxes"][0], other=training_batch["bounding_boxes"][0])
    assert viewer_recorder.arguments["label_names"] == {1: "figure"}
