"""
Smoke test of InstanceSegmentationTrainer of vision_training/trainers/instance_segmentation_trainer.py on a synthetic
instance segmentation dataset: training with the geometric augmentations moving the masks saves checkpoints,
evaluates the box and mask average precisions with its sample images, and resumes. The detections and objects of a
test image are also checked at the original resolution of the image (evaluation.original_resolution).
"""

import csv
import pytest
import torch
import yaml

from pathlib import Path
from typing import Any, Dict, Tuple

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.data.synthetic.create_synthetic_dataset import SHAPE_CATEGORIES, create_synthetic_dataset
from vision_training.trainers.instance_segmentation_trainer import InstanceSegmentationTrainer
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
                                     "validation_iteration_ratio": 2, "resume_training": True})
    experiment_configuration["diagnosis_modes"].update({"compute_receptive_field": False,
                                                        "compute_model_size": False, "view_input_images": False})
    # Trained weights only, the evaluation of the moving average of the weights being tested by the base trainer tests
    configuration["Optimization"]["exponential_moving_average"]["activated"] = False

    # Error images of the evaluation, a few per folder, with the deployment threshold lowered to the evaluation one so
    # that the barely trained detector gives false positives and confusions next to its missed objects
    experiment_configuration["evaluation"]["error_dumps"] = {"activated": True,
                                                             "maximum_per_class_pair": SMOKE_MAXIMUM_ERROR_IMAGES}
    experiment_configuration["evaluation"]["original_resolution"] = original_resolution
    evaluation_score_threshold = experiment_configuration["evaluation"]["score_threshold"]
    configuration["Detector"]["inference"]["score_threshold"] = evaluation_score_threshold

    # Small images and mask heads, two loader workers, and the generated categories mapped to the model classes
    configuration["Data"]["image_settings"]["size"] = SMOKE_IMAGE_SIZE
    configuration["Data"]["loading"]["number_workers"] = 2
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
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
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
    checkpoint with exactly its weights and trains 2 more iterations, without any failed step.

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
        "Box Per Class Average Precision (Area)"]
    assert "Mask Per Class Average Precision (Area)" in results_blocks[1]["text"]
    assert all(scalar_name.startswith(("AP@0.5/", "Mask AP@0.5/")) for scalar_name in results["class_scalars"])
    assert any(scalar_name.startswith("Mask mAP@0.5 By Size/") for scalar_name in results["scalars"])
    assert {"mAR@0.5/all", "Mask mAR@[0.5:0.95]/all"} <= set(results["scalars"])
    assert "Proposal Recall@0.5/all" in results["scalars"] and "Proposal Recall" in results["files"]

    # Box and mask average precisions of every class, the ground truth objects summing to those of the test images
    test_dataset = trainer.test_dataloader.dataset
    number_objects = sum(test_dataset[sample_index]["instance_masks"].shape[0]
                         for sample_index in range(len(test_dataset)))
    with open(file=trainer.experiment_folder / "Evaluation" / "Iteration_5" / "class_metrics.csv", mode="r",
              newline="") as class_metrics_file:
        class_rows = list(csv.DictReader(f=class_metrics_file))
    average_precision_columns = ["average_precision_50", "average_precision_75", "average_precision_50_95"] + [
        f"average_precision_{threshold}_{object_size}" for object_size in ["small", "medium", "large"]
        for threshold in ["50", "50_95"]]
    assert list(class_rows[0]) == ["class_index", "class", "ground_truths"] + average_precision_columns + [
        f"mask_{column_name}" for column_name in average_precision_columns] + ["threshold_at_target_precision",
                                                                               "recall_at_target_precision"]
    assert sum(int(class_row["ground_truths"]) for class_row in class_rows) == number_objects
    assert [class_row["class"] for class_row in class_rows] == list(SHAPE_CATEGORIES)

    # Every object has one size, by mask area for the mask table and by box area for the box table
    for size_table in [results["tables"][1], results["tables"][3]]:
        assert sum(row[1] for row in size_table["rows"]) == number_objects

    # Anchor recall of the training objects written at launch, like the object detection
    assert (trainer.experiment_folder / "anchor_recall.csv").is_file()

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
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=2, original_resolution=False))
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
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=2, original_resolution=True))
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
    batch = {"bounding_boxes": [torch.tensor(data=[[1.0, 3.0, 3.0, 5.0], [0.0, 2.0, 2.0, 3.0],
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
    pixel object vanishing at the network resolution included. At the network resolution, the same image keeps its
    network boxes and areas.
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

    # The same image at the network resolution keeps the network boxes and areas
    trainer.original_resolution = False
    detection = trainer.gather_image_detections(batch=batch, predictions=predictions, image_index=0)
    assert torch.equal(input=detection["boxes"], other=predictions["final_detection_boxes"][0, :2])
    assert detection["mask_areas"].tolist() == [4, 2]
    ground_truth = trainer.gather_image_ground_truth(batch=batch, image_index=0)
    assert torch.equal(input=ground_truth["boxes"], other=batch["bounding_boxes"][0])
    assert ground_truth["mask_areas"].tolist() == [4, 2, 0]
