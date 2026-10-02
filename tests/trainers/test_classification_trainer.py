import json
import pytest
import torch
import yaml

from pathlib import Path
from typing import Any, Dict

from vision_training.configuration.configuration_loader import load_experiment_configuration
from vision_training.data.synthetic.create_synthetic_dataset import create_synthetic_dataset
from vision_training.trainers.classification_trainer import ClassificationTrainer

SYNTHETIC_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" / "classification" /
                                "configuration_synthetic_shapes.yaml")

# Image size of the smoke test, small enough for a few quick iterations
SMOKE_IMAGE_SIZE = 64


def write_smoke_configuration(temporary_directory: Path, dataset_folder: Path, number_iterations: int,
                              head_settings: Dict[str, Any]) -> Path:
    """
    Writes a copy of the synthetic shapes classification configuration adapted to a quick training smoke test: the
    generated dataset, 64 pixel images, batches of 4 images and a saving schedule that triggers an intermediate save,
    a save with test evaluation and a validation pass within a handful of iterations.

    Args:
        temporary_directory (Path): The pytest temporary directory receiving the configuration and the experiments.
        dataset_folder (Path): The root folder of the generated synthetic dataset.
        number_iterations (int): The number of training iterations of the run.
        head_settings (Dict[str, Any]): The 'ClassificationHead' settings replaced for the test.

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

    # Small images, the generated label mapping, and the head settings of the test
    configuration["Data"]["image_settings"]["size"] = SMOKE_IMAGE_SIZE
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
                             number_frames=1, number_workers=2, task="classification")
    head_settings = {"class_weighting": "balanced", "label_smoothing": 0.1}

    # First run: saves at iterations 2 (intermediate), 4 (with test evaluation) and 5 (end of the run)
    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=5, head_settings=head_settings))
    assert trainer.run_name.startswith("CLS_") and "-LS0.1-CWB_MC4_" in trainer.run_name
    assert trainer.class_names == ["square", "circle", "triangle", "rectangle"]
    class_counts = torch.tensor(data=trainer.train_dataloader.dataset.get_class_counts(), dtype=torch.float32)
    expected_class_weights = class_counts.sum() / (4 * class_counts.clamp(min=1.0))
    assert torch.allclose(input=trainer.model.class_weights.cpu(), other=expected_class_weights)
    trainer.run_training_loop()

    weights_directory = trainer.weights_directory
    assert sorted(path.name for path in weights_directory.glob("Iteration_*")) == ["Iteration_2", "Iteration_4",
                                                                                 "Iteration_5"]
    results_lines = (trainer.experiment_folder / "Results.txt").read_text().splitlines()
    assert results_lines[0] == "Iteration 4 : 6 Test Images"
    assert results_lines[1].startswith("  Accuracy ") and "| Top-2 Accuracy " in results_lines[1]
    evaluation_folder = trainer.experiment_folder / "Evaluation" / "Iteration_4"
    assert sorted(path.name for path in evaluation_folder.iterdir()) == ["class_thresholds.json",
                                                                         "confusion_matrix.csv",
                                                                         "confusion_matrix.png"]
    class_thresholds = json.loads((evaluation_folder / "class_thresholds.json").read_text())
    assert list(class_thresholds["classes"]) == trainer.class_names
    assert sorted(path.name for path in trainer.tensorboard_directory.iterdir()) == ["Test", "Train", "Validation"]

    # Second run with the same settings: same run folder, resumed from iteration 5 with exactly its weights
    resumed_trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=2, head_settings=head_settings))
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
                             number_frames=1, number_workers=2, task="classification")

    # Rewrite every manifest in the multi-label format
    for split_name in ["train", "validation", "test"]:
        labels_path = dataset_folder / split_name / "labels.json"
        labels = json.loads(labels_path.read_text())
        labels["samples"] = [{"id": sample["id"], "file_name": sample["file_name"],
                              "category_ids": [sample["category_id"]]} for sample in labels["samples"]]
        labels_path.write_text(data=json.dumps(obj=labels))

    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=5,
        head_settings={"output_mode": "multi_label"}))
    assert "_ML4_" in trainer.run_name
    trainer.run_training_loop()

    results_lines = (trainer.experiment_folder / "Results.txt").read_text().splitlines()
    assert results_lines[0] == "Iteration 4 : 4 Test Images"
    assert results_lines[1].startswith("  Subset Accuracy ")
    evaluation_folder = trainer.experiment_folder / "Evaluation" / "Iteration_4"
    assert sorted(path.name for path in evaluation_folder.iterdir()) == ["class_thresholds.json"]

    console_output = capsys.readouterr().out
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output
