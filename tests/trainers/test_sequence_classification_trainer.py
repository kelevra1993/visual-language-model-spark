import pytest
import torch
import yaml

from pathlib import Path

from vision_training.configuration.configuration_loader import load_experiment_configuration
from vision_training.data.synthetic.create_synthetic_dataset import create_synthetic_dataset
from vision_training.trainers.sequence_classification_trainer import SequenceClassificationTrainer
from tests.fixtures.evaluation_checks import check_error_images, check_predictions_file

SYNTHETIC_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" /
                                "sequence_classification" / "configuration_synthetic_sequences.yaml")

# Frame size and number of frames of the smoke test, small enough for a few quick iterations
SMOKE_IMAGE_SIZE = 64
SMOKE_NUMBER_FRAMES = 4


def write_smoke_configuration(temporary_directory: Path, dataset_folder: Path, number_iterations: int) -> Path:
    """
    Writes a copy of the synthetic sequences configuration adapted to a quick training smoke test: the generated
    dataset, 64 pixel frames, windows of 4 frames, batches of 2 sequences and a saving schedule that triggers an
    intermediate save, a save with test evaluation and a validation pass within a handful of iterations.

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
    experiment_configuration["diagnosis_modes"].update({"compute_receptive_field": True, "compute_model_size": True,
                                                        "view_input_images": False})

    # Small frames, short windows and the generated label mapping
    configuration["Data"]["image_settings"]["size"] = SMOKE_IMAGE_SIZE
    configuration["Data"]["sequence_settings"].update({"sequence_length": SMOKE_NUMBER_FRAMES,
                                                       "window_stride": SMOKE_NUMBER_FRAMES})
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(dataset_folder / "label_mapping.json")
    configuration["ClassificationHead"]["class_weighting"] = "balanced"
    configuration["ExperimentConfiguration"]["evaluation"]["error_dumps"] = {"activated": True,
                                                                           "maximum_per_class_pair": 2}

    configuration_path = temporary_directory / f"configuration_{number_iterations}_iterations.yaml"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    return configuration_path


def build_trainer(configuration_path: Path) -> SequenceClassificationTrainer:
    """
    Loads a configuration file and builds the SequenceClassificationTrainer from it, exactly like main.py does.

    Args:
        configuration_path (Path): The configuration file of the run.

    Returns:
        SequenceClassificationTrainer: The trainer, resumed from the last checkpoint of its run folder when there is
            one.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    return SequenceClassificationTrainer(experiment_configuration=experiment_configuration,
                                         model_configuration=model_configuration,
                                         configuration_path=configuration_path)


def test_sequence_classification_training_saves_evaluates_and_resumes(tmp_path: Path,
                                                                      capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the sequence classification training pipeline on a generated synthetic moving shapes dataset: 5
    iterations save the intermediate, evaluated and final checkpoints, the evaluation of the test sequences is
    written to Results.txt with its confusion matrix and class thresholds files, the frame throughput is printed,
    and a second run with the same settings resumes from the last checkpoint with exactly its weights. No training
    step may fail silently.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    dataset_folder = tmp_path / "synthetic_moving_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 6}, maximum_objects=1,
                             number_frames=SMOKE_NUMBER_FRAMES, number_workers=2, task="sequence_classification")

    # First run: saves at iterations 2 (intermediate), 4 (with test evaluation) and 5 (end of the run)
    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=5))
    assert trainer.run_name.startswith("M2O_") and "_T4_GRU1x128-BI_" in trainer.run_name
    assert trainer.class_names == ["moving_left", "moving_right", "moving_up", "moving_down"]
    trainer.run_training_loop()

    weights_directory = trainer.weights_directory
    assert sorted(path.name for path in weights_directory.glob("Iteration_*")) == ["Iteration_2", "Iteration_4",
                                                                                 "Iteration_5"]
    results_lines = (trainer.experiment_folder / "Results.txt").read_text().splitlines()
    assert results_lines[0] == "Iteration 4 : 6 Test Sequences"
    assert results_lines[1].startswith("  Accuracy ")
    evaluation_folder = trainer.experiment_folder / "Evaluation" / "Iteration_4"
    assert sorted(path.name for path in evaluation_folder.iterdir() if path.name != "Errors") == [
        "class_thresholds.json", "confusion_matrix.csv", "confusion_matrix.png", "predictions.csv"]
    rows = check_predictions_file(evaluation_folder=evaluation_folder, results_lines=results_lines, expected_rows=6,
                                  class_names=trainer.class_names,
                                  identification_columns=["sequence", "first_frame", "last_frame"])
    check_error_images(evaluation_folder=evaluation_folder, rows=rows, results_lines=results_lines,
                       maximum_per_class_pair=2)

    # Second run with the same settings: same run folder, resumed from iteration 5 with exactly its weights
    resumed_trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=2))
    assert resumed_trainer.experiment_folder == trainer.experiment_folder
    assert resumed_trainer.start_iteration == 6
    saved_model_state = torch.load(f=weights_directory / "Iteration_5" / "model_000005.pt", map_location="cpu",
                                   weights_only=True)["model_state"]
    for parameter_name, parameter_value in resumed_trainer.model.state_dict().items():
        assert torch.equal(input=parameter_value.cpu(), other=saved_model_state[parameter_name])
    resumed_trainer.run_training_loop()

    # The sequence throughput comes with the frame throughput, and no iteration failed along the way
    console_output = capsys.readouterr().out
    assert "seqs/sec" in console_output and "frames/sec" in console_output
    assert "8 sequences, 32 frames, 8 windows of 4 frames" in console_output
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output
