"""
Smoke tests of ClassificationTrainer of vision_training/trainers/classification_trainer.py on a synthetic dataset:
multi-class and multi-label training save, evaluate and resume, with the predictions file and error images checked,
and a head reading the flattened feature map trains.
"""

import json
import pytest
import torch
import yaml

from pathlib import Path
from typing import Any, Dict

from vision_training.configuration.configuration_loader import load_experiment_configuration
from vision_training.data.synthetic.create_synthetic_dataset import SHAPE_CATEGORIES, create_synthetic_dataset
from vision_training.trainers.classification_trainer import ClassificationTrainer
from tests.fixtures.evaluation_checks import check_error_images, check_predictions_file, read_predictions_file

SYNTHETIC_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" / "classification" /
                                "configuration_synthetic_shapes.yaml")

# Image size of the smoke test, small enough for a few quick iterations
SMOKE_IMAGE_SIZE = 64


def write_smoke_configuration(temporary_directory: Path, dataset_folder: Path, number_iterations: int,
                              head_settings: Dict[str, Any], evaluation_settings: Dict[str, Any]) -> Path:
    """
    Writes a copy of the synthetic shapes classification configuration adapted to a quick training smoke test: the
    generated dataset, 64 pixel images, batches of 4 images and a saving schedule that triggers an intermediate save,
    a save with test evaluation and a validation pass within a handful of iterations.

    Args:
        temporary_directory (Path): The pytest temporary directory receiving the configuration and the experiments.
        dataset_folder (Path): The root folder of the generated synthetic dataset.
        number_iterations (int): The number of training iterations of the run.
        head_settings (Dict[str, Any]): The 'ClassificationHead' settings replaced for the test.
        evaluation_settings (Dict[str, Any]): The 'ExperimentConfiguration.evaluation' settings replaced for the test.

    Returns:
        Path: The path of the written configuration file.
    """
    configuration = yaml.safe_load(stream=SYNTHETIC_CONFIGURATION_PATH.read_text())

    # Run in the temporary directory, on the generated splits, with a saving schedule fitting a few iterations
    experiment_configuration = configuration["ExperimentConfiguration"]
    experiment_configuration.update({"experiment_folder": str(temporary_directory / "experiments"),
                                     "train_data_folder": str(dataset_folder / "train"),
                                     "validation_data_folder": str(dataset_folder / "validation"),
                                     "test_data_folder": str(dataset_folder / "test"), "batch_size": 4,
                                     "information_dump": 2, "weight_saving_iterations": 4,
                                     "intermediate_weight_saving_iterations": 2, "number_iterations": number_iterations,
                                     "validation_iteration_ratio": 2, "resume_training": True})
    experiment_configuration["diagnosis_modes"].update({"compute_receptive_field": True, "compute_model_size": True,
                                                        "view_input_images": False})
    experiment_configuration["evaluation"].update(evaluation_settings)

    # Small images, the generated label mapping, and the head settings of the test
    configuration["Data"]["image_settings"]["size"] = SMOKE_IMAGE_SIZE
    # Two loader workers per split, enough for the small generated dataset
    configuration["Data"]["loading"]["number_workers"] = 2
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(dataset_folder / "label_mapping.json")
    configuration["ClassificationHead"].update(head_settings)

    configuration_path = temporary_directory / f"configuration_{number_iterations}_iterations.yaml"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    return configuration_path


def build_trainer(configuration_path: Path) -> ClassificationTrainer:
    """
    Loads a configuration file and builds the ClassificationTrainer from it, exactly like main.py does.

    Args:
        configuration_path (Path): The configuration file of the run.

    Returns:
        ClassificationTrainer: The trainer, resumed from the last checkpoint of its run folder when there is one.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    return ClassificationTrainer(experiment_configuration=experiment_configuration,
                                 model_configuration=model_configuration, configuration_path=configuration_path)


def test_classification_training_saves_evaluates_and_resumes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the classification training pipeline on a generated synthetic dataset: the balanced class weights of the
    training split are set in the model, 5 iterations save the intermediate, evaluated and final checkpoints, the
    evaluation of the test split is written to Results.txt with its confusion matrix and class thresholds files, and
    a second run with the same settings resumes from the last checkpoint with exactly its weights. No training step
    may fail silently.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 16, "validation": 8, "test": 6}, maximum_objects=1,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="classification")
    head_settings = {"class_weighting": "balanced", "label_smoothing": 0.1}

    # First run: saves at iterations 2 (intermediate), 4 (with test evaluation) and 5 (end of the run)
    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=5, head_settings=head_settings,
        evaluation_settings={"error_dumps": {"activated": True, "maximum_per_class_pair": 1}}))
    assert trainer.run_name.startswith("CLS_") and trainer.run_name.endswith("-LS0.1-CWB_MC9")
    assert trainer.class_names == list(SHAPE_CATEGORIES)
    # Balanced weights, a class without training image counting as one, as some of the 9 classes may be empty
    class_counts = torch.tensor(data=trainer.train_dataloader.dataset.get_class_counts(), dtype=torch.float32)
    clamped_class_counts = class_counts.clamp(min=1.0)
    expected_class_weights = clamped_class_counts.sum() / (len(SHAPE_CATEGORIES) * clamped_class_counts)
    assert torch.allclose(input=trainer.model.class_weights.cpu(), other=expected_class_weights)
    trainer.run_training_loop()

    weights_directory = trainer.weights_directory
    assert sorted(path.name for path in weights_directory.glob("Iteration_*")) == ["Iteration_2", "Iteration_4",
                                                                                 "Iteration_5"]
    results_lines = (trainer.experiment_folder / "Results.txt").read_text().splitlines()
    assert results_lines[0] == "Iteration 4 : 6 Test Images"
    assert results_lines[1].startswith("  Accuracy ") and "| Top-2 Accuracy " in results_lines[1]
    evaluation_folder = trainer.experiment_folder / "Evaluation" / "Iteration_4"
    assert sorted(path.name for path in evaluation_folder.iterdir() if path.name != "Errors") == [
        "class_thresholds.json", "confusion_matrix.csv", "confusion_matrix.png", "predictions.csv"]
    rows = check_predictions_file(evaluation_folder=evaluation_folder, results_lines=results_lines, expected_rows=6,
                                  class_names=trainer.class_names, identification_columns=["sample"])
    assert all(row["sample"].endswith(".png") for row in rows)
    check_error_images(evaluation_folder=evaluation_folder, rows=rows, results_lines=results_lines,
                       maximum_per_class_pair=1)
    class_thresholds = json.loads((evaluation_folder / "class_thresholds.json").read_text())
    assert list(class_thresholds["classes"]) == trainer.class_names
    assert sorted(path.name for path in trainer.tensorboard_directory.iterdir()) == ["Test", "Train", "Validation"]

    # Second run with the same settings: same run folder, resumed from iteration 5 with exactly its weights
    resumed_trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=2, head_settings=head_settings,
        evaluation_settings={}))
    assert resumed_trainer.experiment_folder == trainer.experiment_folder
    assert resumed_trainer.start_iteration == 6
    saved_model_state = torch.load(f=weights_directory / "Iteration_5" / "model_000005.pt", map_location="cpu",
                                   weights_only=True)["model_state"]
    for parameter_name, parameter_value in resumed_trainer.model.state_dict().items():
        assert torch.equal(input=parameter_value.cpu(), other=saved_model_state[parameter_name])
    resumed_trainer.run_training_loop()

    # The training loop catches errors and keeps going, so make sure that no iteration failed along the way
    console_output = capsys.readouterr().out
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output


def test_multi_label_classification_training(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the multi-label mode of the classification trainer on the synthetic dataset rewritten as a multi-label
    one (every sample listing its single category in 'category_ids'): the training runs, and the evaluation reports
    the multi-label metrics and saves the class thresholds without confusion matrix.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 4}, maximum_objects=1,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="classification")

    # Rewrite every manifest in the multi-label format
    for split_name in ["train", "validation", "test"]:
        labels_path = dataset_folder / split_name / "labels.json"
        labels = json.loads(labels_path.read_text())
        labels["samples"] = [{"id": sample["id"], "file_name": sample["file_name"],
                              "category_ids": [sample["category_id"]]} for sample in labels["samples"]]
        labels_path.write_text(data=json.dumps(obj=labels))

    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=5,
        head_settings={"output_mode": "multi_label"}, evaluation_settings={}))
    assert trainer.run_name.endswith("_ML9")
    trainer.run_training_loop()

    results_lines = (trainer.experiment_folder / "Results.txt").read_text().splitlines()
    assert results_lines[0] == "Iteration 4 : 4 Test Images"
    assert results_lines[1].startswith("  Subset Accuracy ")
    evaluation_folder = trainer.experiment_folder / "Evaluation" / "Iteration_4"
    assert sorted(path.name for path in evaluation_folder.iterdir()) == ["class_thresholds.json", "predictions.csv"]
    multi_label_rows = read_predictions_file(evaluation_folder=evaluation_folder)
    assert len(multi_label_rows) == 4
    assert list(multi_label_rows[0])[:4] == ["sample", "true_classes", "predicted_classes", "correct"]
    assert all(row["true_classes"] in trainer.class_names for row in multi_label_rows)

    console_output = capsys.readouterr().out
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output


def test_predictions_file_can_be_turned_off(tmp_path: Path) -> None:
    """
    Verifies that the evaluation writes no predictions file when 'evaluation.save_predictions' is False, for the large
    test sets whose file would be too big.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 4}, maximum_objects=1,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="classification")

    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=4, head_settings={},
        evaluation_settings={"save_predictions": False}))
    trainer.run_training_loop()

    evaluation_folder = trainer.experiment_folder / "Evaluation" / "Iteration_4"
    assert sorted(path.name for path in evaluation_folder.iterdir()) == ["class_thresholds.json",
                                                                         "confusion_matrix.csv",
                                                                         "confusion_matrix.png"]


def test_flattened_feature_map_training(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies a short classification training whose head reads the flattened final feature map: the run name carries
    HFLAT, the head reads every value of the 64 pixel feature map, and the test evaluation is written to Results.txt
    without any failed iteration.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 4}, maximum_objects=1,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="classification")

    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=4,
        head_settings={"global_pooling": "flatten"}, evaluation_settings={}))
    assert "_HFLAT." in trainer.run_name

    # The head reads the channels, height and width of the final feature map of a 64 pixel image
    with torch.no_grad():
        final_feature_map, _ = trainer.model.backbone(input_tensor=torch.zeros(
            size=(1, 3, SMOKE_IMAGE_SIZE, SMOKE_IMAGE_SIZE), device=trainer.device, dtype=trainer.dtype))
    assert trainer.model.classification_head.input_features == final_feature_map[0].numel()
    trainer.run_training_loop()

    assert (trainer.experiment_folder / "Results.txt").read_text().startswith("Iteration 4 : 4 Test Images")
    console_output = capsys.readouterr().out
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output
