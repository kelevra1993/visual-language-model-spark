"""
Smoke test of InstanceSegmentationTrainer of vision_training/trainers/instance_segmentation_trainer.py on a synthetic
instance segmentation dataset: training with the geometric augmentations moving the masks saves checkpoints,
evaluates the box and mask average precisions with its sample images, their rates written as percentages in the
report, and resumes. The weights of the losses in the
total loss (Optimization.loss_weights) are checked on a training batch, the detections and objects of a test image
at the original resolution of the image (evaluation.original_resolution), and the instance index maps of the loaders
(Data.instance_index_maps) expanded into the masks by prepare_batch, by augment_batch and by the input viewer. The
training target viewer draws the mask targets of the loosest positive region of every object under the targets of the
detector, and the detections of the sample images are captioned with their box and mask overlaps with their object.
"""

import csv
import numpy as np
import pytest
import re
import torch
import yaml

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.data.detection.coco_masks import expand_instance_index_maps
from vision_training.data.synthetic.create_synthetic_dataset import SHAPE_CATEGORIES, create_synthetic_dataset
from vision_training.evaluation.detection.object_sizes import OBJECT_SIZE_NAMES
from vision_training.trainers.instance_segmentation_trainer import (MAXIMUM_MASK_TARGET_REGIONS,
                                                                    InstanceSegmentationTrainer)
from vision_training.utilities.visualization import (GRID_GAP, LEGEND_BAR_HEIGHT, build_class_colours,
                                                     build_detection_sample_grid)
from tests.fixtures.evaluation_checks import (check_detection_error_analysis, check_sample_images,
                                              read_results_blocks)

SYNTHETIC_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" /
                                "instance_segmentation" / "configuration_synthetic_shapes.yaml")

# Image size of the smoke test, small enough for a few quick iterations
SMOKE_IMAGE_SIZE = 256

# Anchor scales of the smoke test: the configuration checker requires every scale to fit in the image
SMOKE_ANCHOR_SCALES = {"5": [16, 32, 64], "6": [32, 64, 128], "7": [64, 128, 256]}

# Geometric augmentations turned on, so that the instance masks are moved and their boxes recomputed
SMOKE_GEOMETRIC_AUGMENTATIONS = ["random_crop", "random_zoom_out", "random_left_right_flip", "random_rotation"]

# Largest number of error images saved per folder by the smoke test
SMOKE_MAXIMUM_ERROR_IMAGES = 2


def write_smoke_configuration(temporary_directory: Path, dataset_folder: Path, number_iterations: int,
                              original_resolution: bool) -> Path:
    """
    Writes a copy of the synthetic shapes instance segmentation configuration adapted to a quick training smoke test:
    the generated dataset, 256 pixel images, batches of 2 images, every training image augmented with the geometric
    augmentations, small mask heads, a saving schedule that triggers an intermediate save, a save with test
    evaluation and a validation pass within a handful of iterations, the error images of the evaluation, at most 2
    per folder, no moving average of the weights, and the resolution of the evaluation.

    Args:
        temporary_directory (Path): The pytest temporary directory receiving the configuration and the experiments.
        dataset_folder (Path): The root folder of the generated synthetic dataset.
        number_iterations (int): The number of training iterations of the run.
        original_resolution (bool): Whether the evaluation scores the boxes and masks at the original resolution of
            the test images.

    Returns:
        Path: The path of the written configuration file.
    """
    configuration = yaml.safe_load(stream=SYNTHETIC_CONFIGURATION_PATH.read_text())

    # Run in the temporary directory, on the generated splits, with a saving schedule fitting a few iterations
    experiment_configuration = configuration["ExperimentConfiguration"]
    experiment_configuration.update({"experiment_folder": str(temporary_directory / "experiments"),
                                     "train_data_folder": str(dataset_folder / "train"),
                                     "validation_data_folder": str(dataset_folder / "validation"),
                                     "test_data_folder": str(dataset_folder / "test"), "batch_size": 2,
                                     "information_dump": 2, "weight_saving_iterations": 4,
                                     "intermediate_weight_saving_iterations": 2, "number_iterations": number_iterations,
                                     "validation_iteration_ratio": 2})
    experiment_configuration["diagnosis_modes"].update({"compute_receptive_field": False, "view_input_images": False})
    # Trained weights only, the evaluation of the moving average of the weights being tested by the base trainer tests
    configuration["Optimization"]["exponential_moving_average"]["activated"] = False

    # Error images of the evaluation, a few per folder, with the deployment threshold lowered to the evaluation one so
    # that the barely trained detector gives false positives and confusions next to its missed objects
    experiment_configuration["evaluation"]["error_dumps"] = {"activated": True,
                                                             "maximum_per_class_pair": SMOKE_MAXIMUM_ERROR_IMAGES}
    experiment_configuration["evaluation"]["original_resolution"] = original_resolution
    evaluation_score_threshold = experiment_configuration["evaluation"]["score_threshold"]
    configuration["Detector"]["inference"]["score_threshold"] = evaluation_score_threshold

    # Small images and mask heads, and the generated categories mapped to the model classes
    configuration["Data"]["image_settings"]["size"] = SMOKE_IMAGE_SIZE
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(dataset_folder / "label_mapping.json")
    for level, scales in SMOKE_ANCHOR_SCALES.items():
        configuration["DetectionLevels"][level]["anchors"]["scales"] = scales
        configuration["DetectionLevels"][level]["mask_head"] = {"roi_align_pool_size": 7, "convolutions": [1, 16]}
    configuration["MaskHead"]["mask_resolution"] = 14

    # Every training image augmented by the geometric augmentations
    augmentation_configuration = configuration["Data"]["data_augmentation"]
    augmentation_configuration["probability"] = 1.0
    for augmentation_type in SMOKE_GEOMETRIC_AUGMENTATIONS:
        augmentation_configuration["types"][augmentation_type]["activated"] = True

    resolution_name = "original" if original_resolution else "network"
    configuration_path = temporary_directory / f"configuration_{number_iterations}_iterations_{resolution_name}.yaml"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    return configuration_path


def build_trainer(configuration_path: Path) -> InstanceSegmentationTrainer:
    """
    Loads a configuration file and builds the InstanceSegmentationTrainer from it, exactly like main.py does.

    Args:
        configuration_path (Path): The configuration file of the run.

    Returns:
        InstanceSegmentationTrainer: The trainer, resumed from the last checkpoint of its run folder when there is one.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    return InstanceSegmentationTrainer(experiment_configuration=experiment_configuration,
                                       model_configuration=model_configuration, configuration_path=configuration_path)


def test_training_saves_evaluates_and_resumes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the whole instance segmentation pipeline on a generated synthetic dataset: 5 iterations with the masks
    moved by the geometric augmentations save the intermediate, evaluated and final checkpoints and track the mask loss,
    the evaluation reports the mask then box tables of the methods, of the object sizes (by mask and by box area), of
    the mean average recalls and of the classes with the proposal recall, the mask mAP@0.5 of the area method as
    headline metric, the box scalars under the object detection names and the mask scalars under 'Mask' names,
    class_metrics.csv with the box and mask columns of all the objects and of every size and the class thresholds of the
    masks, the confusion matrix, class thresholds and error images of the detections matched by their masks, and the
    sample images, the launch writing the anchor recall. A second run with the same settings resumes from the last
    checkpoint with exactly its weights and trains up to the total of 7 iterations, without any failed step.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 4}, maximum_objects=4,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="instance_segmentation")

    # First run: saves at iterations 2 (intermediate), 4 (with test evaluation) and 5 (end of the run)
    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=5, original_resolution=False))
    assert trainer.start_iteration == 1 and trainer.experiment_folder.name.startswith("INS_")
    assert "mask_loss" in trainer.get_tracked_metrics()
    trainer.run_training_loop()

    weights_directory = trainer.weights_directory
    assert sorted(path.name for path in weights_directory.glob("Iteration_*")) == ["Iteration_2", "Iteration_4",
                                                                                 "Iteration_5"]
    results, averaged_weights_results = trainer.run_test_evaluation(iteration=5)
    assert averaged_weights_results is None
    assert results["headline_metric"] == "Mask mAP@0.5/area" and results["headline_metric"] in results["scalars"]
    assert {"mAP@0.5/area", "Mask mAP@[0.5:0.95]/101_point"} <= set(results["scalars"])

    # Report of the two evaluations, with the mask then box tables, the mean average recalls and the proposal recall
    # before the per-class tables
    results_blocks = read_results_blocks(results_path=trainer.experiment_folder / "Results.txt")
    assert [block["iteration"] for block in results_blocks] == [4, 5]
    assert [table["title"] for table in results["tables"]] == [
        "Mask Mean Average Precision", "Mask Mean Average Precision By Object Size (Area)",
        "Box Mean Average Precision", "Box Mean Average Precision By Object Size (Area)", "Mask Mean Average Recall",
        "Box Mean Average Recall", "Proposal Recall", "Mask Per Class Average Precision (Area)",
        "Mask Per Class Recall", "Box Per Class Average Precision (Area)", "Box Per Class Recall"]
    assert "Mask Per Class Average Precision (Area)" in results_blocks[1]["text"]
    assert all(scalar_name.startswith(("AP@0.5/", "Mask AP@0.5/")) for scalar_name in results["class_scalars"])
    assert any(scalar_name.startswith("Mask mAP@0.5 By Size/") for scalar_name in results["scalars"])
    assert {"mAR@0.5/all", "Mask mAR@[0.5:0.95]/all"} <= set(results["scalars"])
    assert "Proposal Recall@0.5/all" in results["scalars"] and "Proposal Recall" in results["files"]

    # The rates of every table written as percentages in Results.txt, the headline mask mAP@0.5 in the best checkpoint
    # line
    assert results["percent_metrics"] == ["Mask mAP@0.5/area"]
    assert all(table["percent_columns"] for table in results["tables"])
    area_rows = [line for line in results_blocks[1]["text"].splitlines() if line.strip().startswith("Area")]
    assert len(area_rows) == 2 and all(re.fullmatch(pattern=r"\s+Area(\s+\d+\.\d{2} %){4}", string=area_row)
                                       for area_row in area_rows)

    # Box and mask average precisions and recalls of every class, the ground truth objects summing to those of the test
    # images, counted by their labels since the test samples of the synthetic configuration carry instance index maps
    test_dataset = trainer.test_dataloader.dataset
    assert test_dataset.load_instance_index_maps and "instance_index_maps" in test_dataset[0]
    number_objects = sum(test_dataset[sample_index]["labels"].shape[0] for sample_index in range(len(test_dataset)))
    with open(file=trainer.experiment_folder / "Evaluation" / "Iteration_5" / "class_metrics.csv", mode="r",
              newline="") as class_metrics_file:
        class_rows = list(csv.DictReader(f=class_metrics_file))
    precision_recall_columns = ["average_precision_50", "average_precision_75", "average_precision_95",
                                "average_precision_50_95"] + [
        f"average_precision_{threshold}_{object_size}" for object_size in OBJECT_SIZE_NAMES
        for threshold in ["50", "50_95"]] + [f"recall_{threshold}{bound}" for threshold in ["50", "75"]
                                             for bound in ["", "_lower", "_upper"]]
    assert list(class_rows[0]) == ["class_index", "class", "ground_truths"] + precision_recall_columns + [
        f"mask_{column_name}" for column_name in precision_recall_columns] + ["threshold_at_target_precision",
                                                                              "recall_at_target_precision"]
    assert sum(int(class_row["ground_truths"]) for class_row in class_rows) == number_objects
    assert [class_row["class"] for class_row in class_rows] == list(SHAPE_CATEGORIES)

    # Every object has one size, by mask area for the mask table and by box area for the box table, each size with
    # its sides in pixels of the network input
    for size_table in [results["tables"][1], results["tables"][3]]:
        assert size_table["columns"][:3] == ["Object Side", "Pixels", "Objects"]
        assert sum(row[2] for row in size_table["rows"]) == number_objects

    # Anchor recall of the training and test objects written at the first launch, like the object detection
    assert (trainer.experiment_folder / "anchor_recall.csv").is_file()
    assert (trainer.experiment_folder / "anchor_recall.png").is_file()

    # Confusion matrix, class thresholds and error images of the detections matched by their masks
    assert trainer.MATCHING_OVERLAP_TYPE == "masks"
    check_detection_error_analysis(
        evaluation_folder=trainer.experiment_folder / "Evaluation" / "Iteration_5", results=results,
        class_names=[trainer.class_names[class_index] for class_index in range(len(trainer.class_names))],
        number_objects=number_objects, maximum_per_class_pair=SMOKE_MAXIMUM_ERROR_IMAGES)

    # Every test image is a sample image, its ground truth panel next to its detection panel
    check_sample_images(evaluation_folder=trainer.experiment_folder / "Evaluation" / "Iteration_5", results=results,
                        number_sample_images=4, sample_kind="images", correct_predictions=None)

    # Second run with the same settings: same run folder, resumed from iteration 5 with exactly its weights
    resumed_trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=7, original_resolution=False))
    assert resumed_trainer.experiment_folder == trainer.experiment_folder
    assert resumed_trainer.start_iteration == 6
    saved_model_state = torch.load(f=weights_directory / "Iteration_5" / "model_000005.pt", map_location="cpu",
                                   weights_only=True)["model_state"]
    for parameter_name, parameter_value in resumed_trainer.model.state_dict().items():
        assert torch.equal(input=parameter_value.cpu(), other=saved_model_state[parameter_name])
    resumed_trainer.run_training_loop()
    assert (weights_directory / "full-checkpoint").read_text().splitlines()[0] == 'model_checkpoint_path: "Iteration_7"'

    # The last checkpoint scored at the original resolution of the test images, here the size of the network input:
    # the restored boxes and the original masks are those of the network, so every average precision is unchanged
    network_results, _ = resumed_trainer.run_test_evaluation(iteration=7)
    original_resolution_trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=7, original_resolution=True))
    assert original_resolution_trainer.start_iteration == 8
    original_resolution_results, _ = original_resolution_trainer.run_test_evaluation(iteration=7)
    assert original_resolution_results["scalars"] == network_results["scalars"]
    test_batch = next(iter(original_resolution_trainer.test_dataloader))
    assert [masks.shape[-2:] for masks in test_batch["original_instance_masks"]] == [
        (SMOKE_IMAGE_SIZE, SMOKE_IMAGE_SIZE)] * len(test_batch["original_instance_masks"])

    # The training loop catches errors and keeps going, so make sure that no iteration failed along the way
    console_output = capsys.readouterr().out
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output


def test_loss_weights_weight_the_total_loss(tmp_path: Path) -> None:
    """
    Verifies 'Optimization.loss_weights' on a training batch of the instance segmenter: the total loss minimised by the
    training is the sum of the five losses of the model, each multiplied by its weight, while the tracked losses
    written to TensorBoard stay unweighted next to the weighted total loss.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 2, "validation": 2, "test": 2}, maximum_objects=4,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="instance_segmentation")
    loss_weights = {"region_proposal_classification_loss": 2.0, "region_proposal_localisation_loss": 0.5,
                    "detection_classification_loss": 1.0, "detection_regression_loss": 3.0, "mask_loss": 0.25}

    # Configuration of the smoke test with weights of every loss differing from 1
    configuration_path = write_smoke_configuration(temporary_directory=tmp_path, dataset_folder=dataset_folder,
                                                   number_iterations=1, original_resolution=False)
    configuration = yaml.safe_load(stream=configuration_path.read_text())
    configuration["Optimization"]["loss_weights"] = loss_weights
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    trainer = build_trainer(configuration_path=configuration_path)
    assert trainer.loss_weights == loss_weights

    # One training batch, its metrics captured where they are written to TensorBoard
    batch = next(iter(trainer.train_dataloader))
    with patch.object(target=trainer.metric_tracker, attribute="log_to_tensorboard") as tensorboard_mock:
        total_loss, model_outputs = trainer.run_model_iteration(batch=batch, writer=None, iteration=1,
                                                                tracker_dictionary=None, apply_augmentation=False)
    logged_metrics = tensorboard_mock.call_args.kwargs["metric_dictionary"]

    expected_total_loss = sum(weight * model_outputs[loss_name] for loss_name, weight in loss_weights.items())
    assert sorted(loss_name for loss_name in model_outputs if loss_name.endswith("_loss")) == sorted(loss_weights)
    assert torch.allclose(input=total_loss, other=expected_total_loss)
    assert not torch.allclose(input=total_loss, other=sum(model_outputs[loss_name] for loss_name in loss_weights))
    assert torch.equal(input=logged_metrics["total_loss"], other=total_loss)
    for loss_name in loss_weights:
        assert torch.equal(input=logged_metrics[loss_name], other=model_outputs[loss_name])


def build_original_resolution_test_image() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Builds one test image of 40 x 20 pixels seen by a 10 pixel network keeping the ratio (scale 0.25, two padding rows
    above the image), with two detections and three objects: a square found by the first detection, a band along the
    top border found by the second detection, whose box starts on the padding, and a one pixel object that vanishes at
    the network resolution.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: The prepared test batch with its objects at both resolutions, and the
            predictions of the instance segmenter for it.
    """
    # Objects of the original annotations: the square, the band along the top border and the one pixel object
    original_masks = torch.zeros(size=(3, 20, 40), dtype=torch.uint8)
    original_masks[0, 4:12, 4:12] = 1
    original_masks[1, 0:4, 0:8] = 1
    original_masks[2, 9, 21] = 1

    # Same objects at the network resolution, the one pixel object keeping its scaled box and an empty mask
    network_masks = torch.zeros(size=(3, 10, 10), dtype=torch.uint8)
    network_masks[0, 3:5, 1:3] = 1
    network_masks[1, 2, 0:2] = 1
    batch = {"images": torch.zeros(size=(1, 3, 10, 10)),
             "bounding_boxes": [torch.tensor(data=[[1.0, 3.0, 3.0, 5.0], [0.0, 2.0, 2.0, 3.0],
                                                   [5.25, 4.25, 5.5, 4.5]])],
             "labels": [torch.tensor(data=[1, 2, 2])], "instance_masks": [network_masks],
             "original_instance_masks": [original_masks],
             "original_bounding_boxes": [torch.tensor(data=[[4.0, 4.0, 12.0, 12.0], [0.0, 0.0, 8.0, 4.0],
                                                            [21.0, 9.0, 22.0, 10.0]])]}

    # Detections of the square (a full mask) and of the band, whose box covers one padding row and whose mask fills
    # its lower half, the visible part of the box, followed by a padding detection
    detection_masks = torch.zeros(size=(1, 3, 4, 4))
    detection_masks[0, 0] = 1.0
    detection_masks[0, 1, 2:] = 1.0
    predictions = {"number_detections": torch.tensor(data=[2]), "final_detection_masks": detection_masks,
                   "final_detection_boxes": torch.tensor(data=[[[1.0, 3.0, 3.0, 5.0], [0.0, 1.0, 2.0, 3.0],
                                                                [0.0, 0.0, 0.0, 0.0]]]),
                   "final_detection_scores": torch.tensor(data=[[0.9, 0.8, 0.0]]),
                   "final_detection_labels": torch.tensor(data=[[1, 2, 0]])}
    return batch, predictions


def test_detections_and_objects_are_gathered_at_the_original_resolution() -> None:
    """
    Verifies the evaluation at the original resolution of a test image: the detection boxes are restored to the image
    and clipped to it, the masks are pasted into the whole restored boxes (the band, whose box starts on the padding,
    then covers exactly its object, where a clipped box would only cover half of it), the overlaps and areas are
    those of the original masks, and the objects keep the boxes and mask areas of the original annotations, the one
    pixel object vanishing at the network resolution included, and the image area sizing the objects is that of the
    original image. At the network resolution, the same image keeps its network boxes and areas, and the area of the
    network input.
    """
    batch, predictions = build_original_resolution_test_image()

    # Only the attributes read by the gathering of the detections and objects, without building the model
    trainer = InstanceSegmentationTrainer.__new__(InstanceSegmentationTrainer)
    trainer.original_resolution, trainer.mask_threshold, trainer.device = True, 0.5, torch.device("cpu")
    trainer.model_configuration = {"Data": {"image_settings": {"size": 10, "keep_ratio": True}}}

    # Detections restored to the original image, the band found by the mask pasted into its whole box
    detection = trainer.gather_image_detections(batch=batch, predictions=predictions, image_index=0)
    restored_boxes = torch.tensor(data=[[4.0, 4.0, 12.0, 12.0], [0.0, 0.0, 8.0, 4.0]])
    assert torch.equal(input=detection["boxes"], other=restored_boxes)
    assert torch.equal(input=detection["mask_overlaps"], other=torch.tensor(data=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                                                                            dtype=torch.float64))
    assert detection["mask_areas"].tolist() == [64, 32]

    # Objects of the original annotations, in the order of the objects at the network resolution
    ground_truth = trainer.gather_image_ground_truth(batch=batch, image_index=0)
    assert torch.equal(input=ground_truth["boxes"], other=batch["original_bounding_boxes"][0])
    assert ground_truth["mask_areas"].tolist() == [64, 32, 1] and ground_truth["labels"].tolist() == [1, 2, 2]
    assert ground_truth["image_area"] == 40.0 * 20.0

    # The same image at the network resolution keeps the network boxes and areas
    trainer.original_resolution = False
    detection = trainer.gather_image_detections(batch=batch, predictions=predictions, image_index=0)
    assert torch.equal(input=detection["boxes"], other=predictions["final_detection_boxes"][0, :2])
    assert detection["mask_areas"].tolist() == [4, 2]
    ground_truth = trainer.gather_image_ground_truth(batch=batch, image_index=0)
    assert torch.equal(input=ground_truth["boxes"], other=batch["bounding_boxes"][0])
    assert ground_truth["mask_areas"].tolist() == [4, 2, 0] and ground_truth["image_area"] == 10.0 * 10.0


def pass_batch_through(images: torch.Tensor, bounding_boxes: List[torch.Tensor], labels: List[torch.Tensor],
                       instance_masks: Optional[List[torch.Tensor]]) -> Tuple[Any, ...]:
    """
    Stands for DataAugmenter.augment_batch without augmenting anything, returning its outputs in its order, so that the
    test reads the instance masks the trainer gives to the augmenter.

    Args:
        images (torch.Tensor): The images of the batch.
        bounding_boxes (List[torch.Tensor]): The boxes of every image.
        labels (List[torch.Tensor]): The labels of every image.
        instance_masks (Optional[List[torch.Tensor]]): The masks of every image.

    Returns:
        Tuple[Any, ...]: The images, boxes, labels, no segmentation masks, the instance masks, no keypoints and one
            empty report per image.
    """
    return images, bounding_boxes, labels, None, instance_masks, None, [{} for _ in range(images.shape[0])]


def test_instance_index_maps_are_expanded_into_the_masks() -> None:
    """
    Verifies that a loader batch carrying instance index maps reaches the instance segmenter with the masks of its
    objects: prepare_batch expands the maps on the device into one [N_i, S, S] uint8 mask per object in the order of the
    labels (an image without objects giving [0, S, S]) without keeping the maps; augment_batch expands the maps of the
    input viewer batch, which comes without prepare_batch, and leaves them out of the augmented batch; the input viewer
    paints the validation image from its map; and the masks of a loader without maps are moved as they are.
    """
    # Two images of 6 pixels: two objects numbered 1 and 2, and an image without objects
    index_maps = torch.zeros(size=(2, 6, 6), dtype=torch.uint8)
    index_maps[0, 1:3, 1:3] = 1
    index_maps[0, 4:6, 0:6] = 2
    expected_masks = torch.stack(tensors=[index_maps[0] == 1, index_maps[0] == 2]).to(dtype=torch.uint8)
    batch = {"images": torch.zeros(size=(2, 3, 6, 6), dtype=torch.uint8),
             "bounding_boxes": [torch.tensor(data=[[1.0, 1.0, 3.0, 3.0], [0.0, 4.0, 6.0, 6.0]]),
                                torch.zeros(size=(0, 4))],
             "labels": [torch.tensor(data=[3, 1]), torch.zeros(size=(0,), dtype=torch.int64)],
             "sample_identifiers": torch.tensor(data=[7, 8]), "instance_index_maps": index_maps}

    # Only the attributes read by the preparation, the augmentation and the input viewer, without building the model
    trainer = InstanceSegmentationTrainer.__new__(InstanceSegmentationTrainer)
    trainer.device, trainer.dtype = torch.device("cpu"), torch.float32
    trainer.data_augmenter = MagicMock()
    trainer.data_augmenter.augment_batch.side_effect = pass_batch_through
    trainer.class_colours = np.array(object=[[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0]], dtype=np.uint8)

    # The prepared batch holds the masks of every image, without the maps
    prepared_batch = trainer.prepare_batch(batch=batch)
    assert "instance_index_maps" not in prepared_batch
    assert torch.equal(prepared_batch["instance_masks"][0], expected_masks)
    assert prepared_batch["instance_masks"][1].shape == (0, 6, 6)

    # The input viewer batch, augmented without prepare_batch, gets the same masks and loses its maps
    augmented_batch = trainer.augment_batch(batch={**batch, "images": batch["images"].float()})
    assert "instance_index_maps" not in augmented_batch
    assert torch.equal(augmented_batch["instance_masks"][0], expected_masks)

    # The validation image of the input viewer painted from its map, as from its masks
    painted_from_map = trainer.get_input_sample_image(batch=batch, sample_index=0)
    painted_from_masks = trainer.get_input_sample_image(
        batch={**batch, "instance_masks": expand_instance_index_maps(index_maps=index_maps, object_counts=[2, 0])},
        sample_index=0)
    assert torch.equal(painted_from_map, painted_from_masks) and painted_from_map.abs().sum() > 0

    # A loader without maps gives its masks, moved as they are
    mask_batch = {key: value for key, value in batch.items() if key != "instance_index_maps"}
    mask_batch["instance_masks"] = [expected_masks, torch.zeros(size=(0, 6, 6), dtype=torch.uint8)]
    assert torch.equal(trainer.prepare_batch(batch=mask_batch)["instance_masks"][0], expected_masks)


def test_training_target_viewer_draws_the_mask_targets(tmp_path: Path) -> None:
    """
    Verifies the 'view_training_targets' diagnosis mode of the instance segmentation on a generated synthetic dataset:
    every training image holding objects is shown with the mask targets of one positive region per object (the GT box
    copies making every object positive) below the three target panels, its panel counting them, and an image left
    without object by the augmentations shows none; the program then exits without creating the run folder.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 4, "validation": 2, "test": 2}, maximum_objects=3,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="instance_segmentation")
    configuration_path = write_smoke_configuration(temporary_directory=tmp_path, dataset_folder=dataset_folder,
                                                   number_iterations=5, original_resolution=False)
    configuration = yaml.safe_load(stream=configuration_path.read_text())
    configuration["ExperimentConfiguration"]["diagnosis_modes"]["view_training_targets"] = True
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))

    with patch(target="vision_training.trainers.detection_trainer.view_input_data", return_value=False) as view_mock:
        with pytest.raises(expected_exception=SystemExit):
            build_trainer(configuration_path=configuration_path)
    assert not (tmp_path / "experiments").exists() or not any((tmp_path / "experiments").iterdir())
    assert view_mock.call_count == 4
    grid_height = SMOKE_IMAGE_SIZE + LEGEND_BAR_HEIGHT
    for call in view_mock.call_args_list:
        (window,) = call.kwargs["windows"]
        lines = [line for line in window.target_lines if isinstance(line, str)]
        object_line = next(line for line in lines if line.startswith("Objects :"))
        number_objects = 0 if object_line == "Objects : none" else int(object_line.split(": ")[1])

        # The mask targets of one region per object under the grid, none for an image without object
        assert window.image.shape[2] >= 3 * SMOKE_IMAGE_SIZE + 2 * GRID_GAP
        if number_objects == 0:
            assert window.image.shape[1] == grid_height and "Mask targets : none (no positive RoI)" in lines
        else:
            assert window.image.shape[1] > grid_height
            assert any(line.startswith("Mask targets : ") and
                       f"{min(number_objects, MAXIMUM_MASK_TARGET_REGIONS)} drawn" in line for line in lines)


def test_sample_image_captions_give_the_box_and_mask_overlaps() -> None:
    """
    Verifies the overlaps given to the captions of the detections of an instance segmentation sample image, measured
    at the network resolution: the square found with a box and a mask equal to its own, and the band found by its mask
    (IoU 1) with a box overlapping its box by half, both matched by their masks as in the evaluation.
    """
    batch, predictions = build_original_resolution_test_image()

    # Only the attributes read by the drawing of the sample image, without building the model
    trainer = InstanceSegmentationTrainer.__new__(InstanceSegmentationTrainer)
    trainer.mask_threshold, trainer.deployment_score_threshold = 0.5, 0.5
    trainer.class_names = {0: "background", 1: "square", 2: "band"}
    trainer.class_colours = build_class_colours(number_classes=3)
    with patch(target="vision_training.trainers.detection_trainer.build_detection_sample_grid", autospec=True,
               side_effect=build_detection_sample_grid) as grid_mock:
        trainer.build_sample_image(batch=batch, predictions=predictions, sample_index=0)
    grid_arguments = grid_mock.call_args.kwargs
    assert np.allclose(grid_arguments["detection_overlaps"]["boxes"], [1.0, 0.5])
    assert np.allclose(grid_arguments["detection_overlaps"]["masks"], [1.0, 1.0])
    assert grid_arguments["matched_detections"].tolist() == [True, True]
