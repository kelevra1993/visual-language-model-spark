"""
Smoke test of DetectionTrainer of vision_training/trainers/detection_trainer.py on a synthetic dataset: training saves
checkpoints, evaluates with its sample images and resumes.
"""

import csv
import pytest
import torch
import yaml

from pathlib import Path

from vision_training.data.synthetic.create_synthetic_dataset import SHAPE_CATEGORIES, create_synthetic_dataset
from vision_training.trainers.detection_trainer import DetectionTrainer
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from tests.fixtures.evaluation_checks import (check_detection_error_analysis, check_sample_images,
                                              read_results_blocks)

SYNTHETIC_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" / "object_detection" /
                                "configuration_synthetic_shapes.yaml")

# Image size of the smoke test, small enough for a few quick iterations
SMOKE_IMAGE_SIZE = 256

# Anchor scales of the smoke test: the configuration checker requires every scale to fit in the image
SMOKE_ANCHOR_SCALES = {"5": [16, 32, 64], "6": [32, 64, 128], "7": [64, 128, 256]}

# Largest number of error images saved per folder by the smoke test
SMOKE_MAXIMUM_ERROR_IMAGES = 2


def write_smoke_configuration(temporary_directory: Path, dataset_folder: Path, number_iterations: int) -> Path:
    """
    Writes a copy of the synthetic shapes configuration adapted to a quick training smoke test: the generated
    dataset, 256 pixel images, the Native loader, batches of 2 images, a saving schedule that triggers an
    intermediate save, a save with test evaluation and a validation pass within a handful of iterations, the error
    images of the evaluation, at most 2 per folder, and no moving average of the weights.

    Args:
        temporary_directory (Path): The pytest temporary directory receiving the configuration and the experiments.
        dataset_folder (Path): The root folder of the generated synthetic dataset.
        number_iterations (int): The number of training iterations of the run.

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
    evaluation_score_threshold = experiment_configuration["evaluation"]["score_threshold"]
    configuration["Detector"]["inference"]["score_threshold"] = evaluation_score_threshold

    # Read the small images, and map the generated categories to the model classes
    configuration["Data"]["image_settings"]["size"] = SMOKE_IMAGE_SIZE
    # Two loader workers per split, enough for the small generated dataset
    configuration["Data"]["loading"]["number_workers"] = 2
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(dataset_folder / "label_mapping.json")
    for level, scales in SMOKE_ANCHOR_SCALES.items():
        configuration["DetectionLevels"][level]["anchors"]["scales"] = scales

    configuration_path = temporary_directory / f"configuration_{number_iterations}_iterations.yaml"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    return configuration_path


def build_trainer(configuration_path: Path) -> DetectionTrainer:
    """
    Loads a configuration file and builds the DetectionTrainer from it, exactly like main.py does.

    Args:
        configuration_path (Path): The configuration file of the run.

    Returns:
        DetectionTrainer: The trainer, resumed from the last checkpoint of its run folder when there is one.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    return DetectionTrainer(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                   configuration_path=configuration_path)


def test_training_saves_evaluates_and_resumes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the whole training pipeline on a generated synthetic dataset: the launch prints the anchor recall of the
    training objects and writes anchor_recall.csv to the run folder, 5 iterations save the intermediate and evaluated
    checkpoints and the final one, evaluate the test split into Results.txt (the tables of the average precision
    methods, of the object sizes, of the mean average recall and of the proposal recall, and the per-class table with
    the class thresholds),
    class_metrics.csv (the ground truth boxes of every class summing to those of the test images, the average
    precisions of every object size), proposal_recall.csv, the confusion matrix, class thresholds and error images of
    the detections (check_detection_error_analysis) and the sample images of the test images, and write the three
    TensorBoard folders. A second run with the same settings resolves to the same run folder, resumes from the last
    checkpoint with exactly its weights, and trains up to the total of 7 iterations. No step may fail silently.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 4}, maximum_objects=4,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="object_detection")

    # First run: saves at iterations 2 (intermediate), 4 (with test evaluation) and 5 (end of the run)
    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=5))
    assert trainer.start_iteration == 1
    trainer.run_training_loop()

    weights_directory = trainer.weights_directory
    assert sorted(path.name for path in weights_directory.glob("Iteration_*")) == ["Iteration_2", "Iteration_4",
                                                                                 "Iteration_5"]
    assert (weights_directory / "full-checkpoint").read_text().splitlines()[0] == 'model_checkpoint_path: "Iteration_5"'
    assert sorted(path.name for path in trainer.tensorboard_directory.iterdir()) == ["Test", "Train", "Validation"]
    results, averaged_weights_results = trainer.run_test_evaluation(iteration=5)
    assert averaged_weights_results is None
    assert results["headline_metric"] == "mAP@0.5/area" and results["headline_metric"] in results["scalars"]

    # Report of the two evaluations, with the table of the methods and the per-class table
    results_blocks = read_results_blocks(results_path=trainer.experiment_folder / "Results.txt")
    assert [(block["iteration"], block["evaluated_samples"]) for block in results_blocks] == [
        (4, "4 Test Images"), (5, "4 Test Images")]
    assert [table["title"] for table in results["tables"]] == ["Mean Average Precision",
                                                              "Mean Average Precision By Object Size (Area)",
                                                              "Mean Average Recall", "Proposal Recall",
                                                              "Per Class Average Precision (Area)"]
    assert "Per Class Average Precision (Area)" in results_blocks[1]["text"]
    assert "Proposal Recall" in results_blocks[1]["text"]
    assert all(scalar_name.startswith("AP@0.5/") for scalar_name in results["class_scalars"])
    assert {"Proposal Recall@0.5/all", "Proposal Recall@0.7/block_5", "Proposals Per Image/all"} <= set(
        results["scalars"])
    assert any(scalar_name.startswith("mAP@0.5 By Size/") for scalar_name in results["scalars"])
    assert {"mAR@0.5/all", "mAR@[0.5:0.95]/all"} <= set(results["scalars"])

    # The ground truth boxes of every class sum to the boxes of the test images, with the average precisions of every
    # object size before the thresholds
    test_dataset = trainer.test_dataloader.dataset
    number_boxes = sum(test_dataset[sample_index]["bounding_boxes"].shape[0]
                       for sample_index in range(len(test_dataset)))
    with open(file=trainer.experiment_folder / "Evaluation" / "Iteration_5" / "class_metrics.csv", mode="r",
              newline="") as class_metrics_file:
        class_rows = list(csv.DictReader(f=class_metrics_file))
    assert list(class_rows[0]) == ["class_index", "class", "ground_truths", "average_precision_50",
                                   "average_precision_75", "average_precision_95", "average_precision_50_95"] + [
        f"average_precision_{threshold}_{object_size}" for object_size in ["small", "medium", "large"]
        for threshold in ["50", "50_95"]] + ["threshold_at_target_precision", "recall_at_target_precision"]
    assert sum(int(class_row["ground_truths"]) for class_row in class_rows) == number_boxes
    assert [class_row["class"] for class_row in class_rows] == list(SHAPE_CATEGORIES)

    # Proposal recall of the test objects: every object counted once, by size, and the proposals of every level
    # summing to those of all the levels
    with open(file=trainer.experiment_folder / "Evaluation" / "Iteration_5" / "proposal_recall.csv", mode="r",
              newline="") as proposal_recall_file:
        proposal_rows = {row["group"]: row for row in csv.DictReader(f=proposal_recall_file)}
    assert list(proposal_rows) == ["all", "small", "medium", "large", "block_5", "block_6", "block_7"]
    assert int(proposal_rows["all"]["objects"]) == number_boxes == sum(
        int(proposal_rows[object_size]["objects"]) for object_size in ["small", "medium", "large"])
    assert float(proposal_rows["all"]["proposals_per_image"]) == pytest.approx(expected=sum(
        float(proposal_rows[f"block_{level}"]["proposals_per_image"]) for level in [5, 6, 7]), abs=1e-5)
    assert results["files"]["Proposal Recall"].name == "proposal_recall.csv"

    # Anchor recall of the training objects, written to the run folder at launch
    train_dataset = trainer.train_dataloader.dataset
    number_training_boxes = sum(train_dataset[sample_index]["bounding_boxes"].shape[0]
                                for sample_index in range(len(train_dataset)))
    assert trainer.training_data_analysis_files == {"Anchor Recall": trainer.experiment_folder / "anchor_recall.csv"}
    with open(file=trainer.experiment_folder / "anchor_recall.csv", mode="r", newline="") as anchor_recall_file:
        anchor_rows = list(csv.DictReader(f=anchor_recall_file))
    assert list(anchor_rows[0]) == ["objects", "count", "recall_block_5", "recall_block_6", "recall_block_7",
                                    "recall_all_levels"]
    assert [row["objects"] for row in anchor_rows] == ["all", "small", "medium", "large"]
    assert int(anchor_rows[0]["count"]) == number_training_boxes
    assert float(anchor_rows[0]["recall_all_levels"]) >= max(float(anchor_rows[0][f"recall_block_{level}"])
                                                             for level in [5, 6, 7])
    first_run_output = capsys.readouterr().out
    assert "Anchor Recall Of The Training Objects" in first_run_output

    # Confusion matrix with its background row and column, class thresholds and error images of the detections
    check_detection_error_analysis(
        evaluation_folder=trainer.experiment_folder / "Evaluation" / "Iteration_5", results=results,
        class_names=[trainer.class_names[class_index] for class_index in range(len(trainer.class_names))],
        number_objects=number_boxes, maximum_per_class_pair=SMOKE_MAXIMUM_ERROR_IMAGES)

    # Every test image is a sample image (8 asked for 4 images), named after its file: its ground truth panel next to
    # its detection panel, between the caption and legend bars
    check_sample_images(evaluation_folder=trainer.experiment_folder / "Evaluation" / "Iteration_5", results=results,
                        number_sample_images=4, sample_kind="images", correct_predictions=None)
    assert sorted(results["sample_images"]) == sorted(f"Samples/{Path(file_name).stem}"
                                                      for file_name in trainer.test_image_file_names.values())
    assert all(sample_image.shape[0] == 24 + SMOKE_IMAGE_SIZE + 24 and sample_image.shape[1] >= 2 * SMOKE_IMAGE_SIZE + 4
               for sample_image in results["sample_images"].values())

    # Second run with the same settings: same run folder, resumed from iteration 5 with exactly its weights
    resumed_trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=7))
    assert resumed_trainer.experiment_folder == trainer.experiment_folder
    assert resumed_trainer.start_iteration == 6
    saved_model_state = torch.load(f=weights_directory / "Iteration_5" / "model_000005.pt", map_location="cpu",
                                   weights_only=True)["model_state"]
    for parameter_name, parameter_value in resumed_trainer.model.state_dict().items():
        assert torch.equal(input=parameter_value.cpu(), other=saved_model_state[parameter_name])

    # The resumed run trains iterations 6 (intermediate save) and 7 (end of the run)
    resumed_trainer.run_training_loop()
    assert sorted(path.name for path in weights_directory.glob("Iteration_*")) == ["Iteration_2", "Iteration_4",
                                                                                 "Iteration_5", "Iteration_6",
                                                                                 "Iteration_7"]
    assert (weights_directory / "full-checkpoint").read_text().splitlines()[0] == 'model_checkpoint_path: "Iteration_7"'

    # The training loop catches errors and keeps going, so make sure that no iteration of both runs failed
    console_output = first_run_output + capsys.readouterr().out
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output
