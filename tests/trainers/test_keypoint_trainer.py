"""
Smoke test of KeypointTrainer of vision_training/trainers/keypoint_trainer.py on a generated synthetic keypoint
dataset, on the CPU: training on the whole images saves checkpoints, evaluates the objects found in the test images
with the object keypoint similarity (the average precision tables, the matched objects, PCK and mean OKS, the
per-class and per-keypoint tables, their CSV files and the sample images), its rates written as percentages in the
report, and resumes from its last checkpoint, with the keypoint tables of its own classes when the mapping file
changed since.
"""

import csv
import json
import numpy as np
import pytest
import torch
import yaml

from pathlib import Path
from typing import Dict, List

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.data.synthetic.create_synthetic_dataset import create_synthetic_dataset
from vision_training.evaluation.keypoints.object_keypoint_similarity import CORRECT_KEYPOINTS_NAME
from vision_training.models.keypoints.keypoint_detector import build_keypoint_channel_table
from vision_training.trainers.keypoint_trainer import KeypointTrainer
from tests.fixtures.evaluation_checks import check_sample_images, read_results_blocks

SYNTHETIC_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" /
                                "keypoint_detection" / "configuration_synthetic_shapes.yaml")

# Image size of the smoke test, small enough for a few quick iterations on the CPU, and a multiple of the stride 32
# of the lowest branch of the 3 stages of the synthetic high resolution network
SMOKE_IMAGE_SIZE = 128

# Number of sample images asked for, more than the generated test images so that every one of them is drawn
SMOKE_NUMBER_SAMPLE_IMAGES = 30


def write_smoke_configuration(temporary_directory: Path, dataset_folder: Path, number_iterations: int) -> Path:
    """
    Writes a copy of the synthetic keypoint configuration adapted to a quick training smoke test: the generated
    dataset, 128 pixel images, batches of 2 images, a saving schedule that triggers an intermediate save, a save with
    test evaluation and a validation pass within a handful of iterations, a sample image of every test image, no
    moving average of the weights and none of the slow diagnoses.

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
                                     "validation_iteration_ratio": 2})
    experiment_configuration["diagnosis_modes"].update({"compute_receptive_field": False, "view_input_images": False})
    experiment_configuration["evaluation"]["number_sample_images"] = SMOKE_NUMBER_SAMPLE_IMAGES
    # Trained weights only, the evaluation of the moving average of the weights being tested by the base trainer tests
    configuration["Optimization"]["exponential_moving_average"]["activated"] = False

    # Read the small images, and map the generated categories to the model classes
    configuration["Data"]["image_settings"]["size"] = SMOKE_IMAGE_SIZE
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(dataset_folder / "label_mapping.json")

    configuration_path = temporary_directory / f"configuration_{number_iterations}_iterations.yaml"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    return configuration_path


def build_trainer(configuration_path: Path) -> KeypointTrainer:
    """
    Loads a configuration file and builds the KeypointTrainer from it, exactly like main.py does, the loader
    injecting the keypoint definitions of the generated training labels.

    Args:
        configuration_path (Path): The configuration file of the run.

    Returns:
        KeypointTrainer: The trainer, resumed from the last checkpoint of its run folder when there is one.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    return KeypointTrainer(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                           configuration_path=configuration_path)


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """
    Reads the rows of a metrics file written by the keypoint evaluation, keyed by the column names of its header.

    Args:
        csv_path (Path): The path of the CSV file.

    Returns:
        List[Dict[str, str]]: The rows of the file.
    """
    with open(file=csv_path, mode="r", newline="") as csv_file:
        return list(csv.DictReader(f=csv_file))


def test_training_saves_evaluates_and_resumes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the whole keypoint training pipeline on a generated synthetic dataset: 5 iterations save the intermediate
    and evaluated checkpoints and the final one, evaluate the test images into Results.txt (the OKS average precision
    tables of the methods, object sizes and recalls, the per-class OKS average precisions, and the keypoint accuracy
    per class and per keypoint, with the matched objects, PCK and mean OKS of all the keypoints), class_metrics.csv
    (one row per class with its objects summing to the objects of the test images, its average precisions and its
    keypoint accuracy), keypoint_metrics.csv (one row per keypoint of every class) and a sample image of every test
    image, and write the three TensorBoard folders. A second run with the same settings resolves to the same run
    folder, resumes from the last checkpoint with exactly its weights, and trains up to the total of 7 iterations.
    No step may fail silently.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    dataset_folder = tmp_path / "synthetic_keypoints"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 4, "validation": 2, "test": 2}, maximum_objects=2,
                             minimum_frames=1, maximum_frames=1, number_workers=1, task="keypoint_detection")

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
    assert results["headline_metric"] == "OKS mAP@0.5/area" and results["headline_metric"] in results["scalars"]

    # Every test image is evaluated, in both evaluations of Results.txt, with the objects holding a labelled keypoint
    test_dataset = trainer.test_dataloader.dataset
    number_test_images = len(test_dataset)
    number_test_objects = sum(len(test_dataset[image_index]["labels"]) for image_index in range(number_test_images))
    assert number_test_images == 2 and number_test_objects > 0
    results_blocks = read_results_blocks(results_path=trainer.experiment_folder / "Results.txt")
    assert [(block["iteration"], block["evaluated_samples"]) for block in results_blocks] == [
        (4, f"{number_test_images} Test Images"), (5, f"{number_test_images} Test Images")]

    # Tables of the OKS average precisions, then of the keypoint accuracy per class and per keypoint, all written in
    # Results.txt with the matched objects, the PCK and the mean OKS of all the keypoints
    table_titles = ["OKS Mean Average Precision", "OKS Mean Average Precision By Object Size (Area)",
                    "OKS Mean Average Recall", "OKS Per Class Average Precision (Area)", "OKS Per Class Recall",
                    "Keypoint Accuracy Per Class", "Keypoint Accuracy Per Keypoint"]
    assert [table["title"] for table in results["tables"]] == table_titles
    assert all(table_title in results_blocks[1]["text"] for table_title in table_titles)
    assert list(results["metrics"]) == ["Matched Objects", CORRECT_KEYPOINTS_NAME, "Mean OKS",
                                        "Mean Keypoint Error (pixels)"]
    assert all(0.0 <= results["metrics"][metric_name] <= 1.0
               for metric_name in ["Matched Objects", CORRECT_KEYPOINTS_NAME, "Mean OKS"])
    assert CORRECT_KEYPOINTS_NAME in results_blocks[1]["text"] and "Mean OKS" in results_blocks[1]["text"]

    # The matched objects, PCK and mean OKS of the grid written as percentages, like the rates of every table, and the
    # headline OKS mAP@0.5 in the best checkpoint line, the values staying between 0 and 1
    assert results["percent_metrics"] == ["Matched Objects", CORRECT_KEYPOINTS_NAME, "Mean OKS", "OKS mAP@0.5/area"]
    assert all(table["percent_columns"] for table in results["tables"])
    grid_lines = results_blocks[1]["text"].splitlines()[:2]
    assert f"{100.0 * results['metrics']['Matched Objects']:.2f} %" in grid_lines[0]
    assert f"{100.0 * results['metrics'][CORRECT_KEYPOINTS_NAME]:.2f} %" in grid_lines[0]
    assert f"{100.0 * results['metrics']['Mean OKS']:.2f} %" in grid_lines[1]
    assert {f"Keypoint Accuracy/{CORRECT_KEYPOINTS_NAME}", "Keypoint Accuracy/Mean OKS"} <= set(results["scalars"])
    assert all(scalar_name.startswith("OKS AP@0.5/") for scalar_name in results["class_scalars"])

    # One row per class in class_metrics.csv, its objects summing to the objects of the test images, with the OKS
    # average precisions and the keypoint accuracy of the class
    evaluation_folder = trainer.experiment_folder / "Evaluation" / "Iteration_5"
    class_rows = read_csv_rows(csv_path=evaluation_folder / "class_metrics.csv")
    assert {"class_index", "class", "objects", "matched_objects", "labelled_keypoints", "percentage_correct_keypoints",
            "mean_object_keypoint_similarity", "mean_error_pixels"} <= set(class_rows[0])
    assert any(column_name.startswith("oks_") for column_name in class_rows[0])
    assert [class_row["class"] for class_row in class_rows] == [trainer.class_names[class_index]
                                                                for class_index in sorted(trainer.class_names)
                                                                if class_index != 0]
    assert sum(int(class_row["objects"]) for class_row in class_rows) == number_test_objects

    # One row per keypoint of every class in keypoint_metrics.csv, in the order of the keypoint definitions
    keypoint_rows = read_csv_rows(csv_path=evaluation_folder / "keypoint_metrics.csv")
    assert list(keypoint_rows[0]) == ["class_index", "class", "keypoint", "labelled", "found",
                                      "percentage_correct_keypoints", "mean_error_pixels"]
    assert [(int(keypoint_row["class_index"]), keypoint_row["keypoint"]) for keypoint_row in keypoint_rows] == [
        (class_index, keypoint_name) for class_index, definition in sorted(trainer.keypoint_categories.items())
        for keypoint_name in definition["keypoints"]]
    assert results["files"]["Class Metrics"] == evaluation_folder / "class_metrics.csv"
    assert results["files"]["Keypoint Metrics"] == evaluation_folder / "keypoint_metrics.csv"

    # Every test image is a sample image named after its file: its true objects next to the objects found, between
    # the caption and legend bars, the caption bar at least as wide as the two panels
    check_sample_images(evaluation_folder=evaluation_folder, results=results, number_sample_images=number_test_images,
                        sample_kind="images", correct_predictions=None)
    assert sorted(results["sample_images"]) == sorted(f"Samples/{Path(file_name).stem}"
                                                      for file_name in trainer.test_image_file_names.values())
    assert all(sample_image.shape[0] == 24 + SMOKE_IMAGE_SIZE + 24 and sample_image.shape[1] >= 2 * SMOKE_IMAGE_SIZE + 4
               for sample_image in results["sample_images"].values())
    first_run_output = capsys.readouterr().out

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


def test_resumed_run_builds_the_keypoint_tables_of_its_own_classes(tmp_path: Path) -> None:
    """
    Verifies that a run resumed after its mapping file swapped two classes with different numbers of keypoints groups
    and scores its keypoints with the definitions of the classes it keeps (restore_run_label_dictionary of the base
    trainer): the keypoint categories, the heatmap channel table of the grouping and the maximum number of keypoints of
    the trainer are those of the restored classes, not those of the edited mapping file.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    dataset_folder = tmp_path / "synthetic_keypoints"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 4, "validation": 2, "test": 2}, maximum_objects=2,
                             minimum_frames=1, maximum_frames=1, number_workers=1, task="keypoint_detection")
    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=2))
    trainer.run_training_loop()
    trained_keypoint_categories = trainer.keypoint_categories

    # Swap in the mapping file the classes of two categories holding different numbers of keypoints
    mapping_path = dataset_folder / "label_mapping.json"
    label_mapping = json.loads(s=mapping_path.read_text())
    category_by_class = {class_index: category for category, class_index in label_mapping.items()}
    first_class, second_class = next(
        (first_index, second_index) for first_index in sorted(trained_keypoint_categories)
        for second_index in sorted(trained_keypoint_categories)
        if len(trained_keypoint_categories[first_index]["keypoints"])
        != len(trained_keypoint_categories[second_index]["keypoints"]))
    label_mapping[category_by_class[first_class]] = second_class
    label_mapping[category_by_class[second_class]] = first_class
    mapping_path.write_text(data=json.dumps(obj=label_mapping))

    # The resumed run keeps its classes, and the tables of its grouping and scoring follow them
    resumed_trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=4))
    assert resumed_trainer.start_iteration == 3
    assert resumed_trainer.label_dictionary == trainer.label_dictionary
    assert resumed_trainer.keypoint_categories == trained_keypoint_categories
    assert resumed_trainer.keypoint_categories == resumed_trainer.data_configuration["keypoint_categories"]
    restored_channel_table, _ = build_keypoint_channel_table(keypoint_categories=trained_keypoint_categories)
    assert np.array_equal(resumed_trainer.channel_table, restored_channel_table.numpy())
    assert np.array_equal(resumed_trainer.channel_table, trainer.channel_table)
    assert resumed_trainer.maximum_keypoints == trainer.maximum_keypoints
