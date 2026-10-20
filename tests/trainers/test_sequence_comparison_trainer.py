"""
Smoke test of SequenceComparisonTrainer of vision_training/trainers/sequence_comparison_trainer.py on synthetic pairs:
training saves, evaluates and resumes, with the predictions file, error images and sample images checked.
"""

import pytest
import torch
import yaml

from pathlib import Path
from unittest.mock import patch

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.data.split_description import count_split_pairs
from vision_training.data.synthetic.create_synthetic_dataset import create_synthetic_dataset
from vision_training.trainers.sequence_comparison_trainer import SequenceComparisonTrainer
from tests.fixtures.evaluation_checks import (check_class_metrics_file, check_error_images, check_predictions_file,
                                              check_sample_images, read_results_blocks)

SYNTHETIC_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" /
                                "sequence_comparison" / "configuration_synthetic_pairs.yaml")

# Frame size and maximum number of frames of a member of the smoke test, small enough for a few quick iterations
SMOKE_IMAGE_SIZE = 64
SMOKE_NUMBER_FRAMES = 4


def write_smoke_configuration(temporary_directory: Path, dataset_folder: Path, number_iterations: int) -> Path:
    """
    Writes a copy of the synthetic pairs configuration adapted to a quick training smoke test: the generated dataset,
    64 pixel frames, members of up to 4 frames padded to the longest member of their batch and summarised by a
    bidirectional GRU, the cross attention fusion, batches of 2 pairs and a saving schedule that triggers an
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

    # Small frames, short members, a recurrent temporal module, the cross attention fusion and the generated label
    # mapping
    configuration["Data"]["image_settings"]["size"] = SMOKE_IMAGE_SIZE
    # Two loader workers per split, enough for the small generated dataset
    configuration["Data"]["loading"]["number_workers"] = 2
    configuration["Data"]["sequence_settings"]["maximum_sequence_length"] = SMOKE_NUMBER_FRAMES
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(dataset_folder / "label_mapping.json")
    configuration["TemporalModel"].update({"type": "gru", "hidden_size": 128, "bidirectional": True})
    configuration["Fusion"]["type"] = "cross_attention"
    configuration["Data"]["data_augmentation"]["shuffle_frames"] = {"activated": True, "probability": 0.5}
    configuration["ClassificationHead"]["class_weighting"] = "balanced"
    configuration["ExperimentConfiguration"]["evaluation"]["error_dumps"] = {"activated": True,
                                                                           "maximum_per_class_pair": 2}

    configuration_path = temporary_directory / f"configuration_{number_iterations}_iterations.yaml"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    return configuration_path


def build_trainer(configuration_path: Path) -> SequenceComparisonTrainer:
    """
    Loads a configuration file and builds the SequenceComparisonTrainer from it, exactly like main.py does.

    Args:
        configuration_path (Path): The configuration file of the run.

    Returns:
        SequenceComparisonTrainer: The trainer, resumed from the last checkpoint of its run folder when there is one.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    return SequenceComparisonTrainer(experiment_configuration=experiment_configuration,
                                     model_configuration=model_configuration, configuration_path=configuration_path)


def test_sequence_comparison_training_saves_evaluates_and_resumes(tmp_path: Path,
                                                                  capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the sequence comparison training pipeline on a generated synthetic object loss pairs dataset: 5
    iterations save the intermediate, evaluated and final checkpoints, the evaluation of the test pairs is reported
    in Results.txt with its per-class metrics, confusion matrix and class thresholds files, the pair and frame
    throughputs are printed with the description of the splits, and a second run with the same settings resumes from
    the last checkpoint with exactly its weights. No training step may fail silently.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    dataset_folder = tmp_path / "synthetic_object_loss_pairs"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 6}, maximum_objects=3,
                             minimum_frames=2, maximum_frames=SMOKE_NUMBER_FRAMES,
                             number_workers=2, task="sequence_comparison")

    # First run: saves at iterations 2 (intermediate), 4 (with test evaluation) and 5 (end of the run)
    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=5))
    assert trainer.run_name.startswith("CMP_") and "_T4_GRU1x128-BI_FXA4_" in trainer.run_name
    assert "AUG" not in trainer.run_name and "SHUF" not in trainer.run_name
    assert trainer.class_names == ["loss_of_object", "same_number_of_objects"]
    with patch.object(target=trainer, attribute="report_evaluation", wraps=trainer.report_evaluation) as report_mock:
        trainer.run_training_loop()
    results = report_mock.call_args.kwargs["results"]

    weights_directory = trainer.weights_directory
    assert sorted(path.name for path in weights_directory.glob("Iteration_*")) == ["Iteration_2", "Iteration_4",
                                                                                 "Iteration_5"]
    results_blocks = read_results_blocks(results_path=trainer.experiment_folder / "Results.txt")
    assert [(block["iteration"], block["evaluated_samples"]) for block in results_blocks] == [(4, "6 Test Pairs")]
    assert "Accuracy" in results["metrics"] and "Per Class Metrics" in results_blocks[0]["text"]
    evaluation_folder = trainer.experiment_folder / "Evaluation" / "Iteration_4"
    assert sorted(path.name for path in evaluation_folder.iterdir() if path.name != "Errors") == [
        "Samples", "class_metrics.csv", "class_thresholds.json", "confusion_matrix.csv", "confusion_matrix.png",
        "precision_recall_curves.png", "predictions.csv", "roc_curves.png"]
    check_class_metrics_file(evaluation_folder=evaluation_folder, class_names=trainer.class_names, number_samples=6,
                             per_class_report=True)
    rows = check_predictions_file(evaluation_folder=evaluation_folder, reported_accuracy=results["metrics"]["Accuracy"],
                                  expected_rows=6, class_names=trainer.class_names, identification_columns=["pair"])
    assert sorted(row["pair"] for row in rows) == [f"test_{pair_index:06d}" for pair_index in range(6)]
    check_error_images(evaluation_folder=evaluation_folder, rows=rows, notes=results["notes"],
                       maximum_per_class_pair=2)

    # Every test pair is a sample image (8 asked for 6 pairs), captioned in the colour of its prediction
    check_sample_images(evaluation_folder=evaluation_folder, results=results, number_sample_images=6,
                        sample_kind="pairs", correct_predictions={row["pair"]: row["correct"] == "1" for row in rows})

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

    # The pair throughput comes with the frame throughput, the splits are described, and no iteration failed
    console_output = capsys.readouterr().out
    number_pairs, number_frames, number_kept_pairs = count_split_pairs(
        split_data_folder=dataset_folder / "train",
        sequence_settings={"maximum_sequence_length": SMOKE_NUMBER_FRAMES})
    assert number_pairs == number_kept_pairs == 8
    assert "Pairs Per Second" in console_output and "Frames Per Second" in console_output
    assert f"8 pairs, {number_frames} frames, 8 pairs of members of at most 4 frames" in console_output
    assert "while the 'gru' temporal module models their order" in console_output
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output
