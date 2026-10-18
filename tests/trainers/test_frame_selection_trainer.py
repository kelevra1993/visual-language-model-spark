"""
Smoke test of FrameSelectionTrainer of vision_training/trainers/frame_selection_trainer.py on synthetic selection
sequences: training saves, evaluates both decodings and resumes, with the thresholds, class metrics, selections file,
error images and sample images checked.
"""

import csv
import json
import pytest
import torch
import yaml

from pathlib import Path
from unittest.mock import patch

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.data.synthetic.create_synthetic_dataset import (SELECTION_CATEGORIES, SELECTION_WANTED_SHAPES,
                                                                     create_synthetic_dataset)
from vision_training.trainers.frame_selection_trainer import FrameSelectionTrainer
from tests.fixtures.evaluation_checks import check_sample_images, read_results_blocks

SYNTHETIC_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" /
                                "frame_selection" / "configuration_synthetic_sequences.yaml")

# Frame size and frames per sequence of the smoke test, small enough for a few quick iterations
SMOKE_IMAGE_SIZE = 64
SMOKE_MINIMUM_FRAMES = 4
SMOKE_MAXIMUM_FRAMES = 6


def write_smoke_configuration(temporary_directory: Path, dataset_folder: Path, number_iterations: int) -> Path:
    """
    Writes a copy of the synthetic frame selection configuration adapted to a quick training smoke test: the
    generated dataset, 64 pixel frames, sequences of 4 to 6 frames fed whole, batches of 2 sequences, the error
    images on, and a saving schedule that triggers an intermediate save, a save with test evaluation and a validation
    pass within a handful of iterations.

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
    experiment_configuration["evaluation"]["error_dumps"] = {"activated": True, "maximum_per_class_pair": 2}

    # Small frames, whole sequences and the generated label mapping, two loader workers per split
    configuration["Data"]["image_settings"]["size"] = SMOKE_IMAGE_SIZE
    configuration["Data"]["loading"]["number_workers"] = 2
    configuration["Data"]["sequence_settings"]["maximum_sequence_length"] = SMOKE_MAXIMUM_FRAMES
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(dataset_folder / "label_mapping.json")

    configuration_path = temporary_directory / f"configuration_{number_iterations}_iterations.yaml"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    return configuration_path


def build_trainer(configuration_path: Path) -> FrameSelectionTrainer:
    """
    Loads a configuration file and builds the FrameSelectionTrainer from it, exactly like main.py does.

    Args:
        configuration_path (Path): The configuration file of the run.

    Returns:
        FrameSelectionTrainer: The trainer, resumed from the last checkpoint of its run folder when there is one.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    return FrameSelectionTrainer(experiment_configuration=experiment_configuration,
                                 model_configuration=model_configuration, configuration_path=configuration_path)


def test_frame_selection_training_saves_evaluates_and_resumes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the frame selection training pipeline on a generated synthetic selection dataset: 5 iterations save the
    intermediate, evaluated and final checkpoints with the unselected class in the weights, the evaluation of the
    test sequences reports both decodings, the ranks and the per-frame metrics in Results.txt, saves the thresholds of
    the wanted classes, their metrics, the decoded confusion matrix, the selections of every sequence and wanted
    class, the error images per class and the sample images, and a second run with the same settings resumes from the
    last checkpoint with exactly its weights. No training step may fail silently.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    dataset_folder = tmp_path / "synthetic_selected_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 3}, maximum_objects=1,
                             minimum_frames=SMOKE_MINIMUM_FRAMES, maximum_frames=SMOKE_MAXIMUM_FRAMES,
                             number_workers=2, task="frame_selection")
    test_manifest = json.loads(s=(dataset_folder / "test" / "labels.json").read_text())
    number_test_frames = sum(len(sequence["frames"]) for sequence in test_manifest["sequences"])

    # First run: saves at iterations 2 (intermediate), 4 (with test evaluation) and 5 (end of the run)
    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=5))
    assert trainer.run_name.startswith("SEL_") and trainer.run_name.endswith("_T6_GRU1x128-BI_HAVG.64_MC4_FPC1-SLW1")
    assert trainer.class_names == list(SELECTION_CATEGORIES) and trainer.unselected_class_index == 0
    with patch.object(target=trainer, attribute="report_evaluation", wraps=trainer.report_evaluation) as report_mock:
        trainer.run_training_loop()
    results = report_mock.call_args.kwargs["results"]

    weights_directory = trainer.weights_directory
    assert sorted(path.name for path in weights_directory.glob("Iteration_*")) == ["Iteration_2", "Iteration_4",
                                                                                 "Iteration_5"]
    saved_model_state = torch.load(f=weights_directory / "Iteration_5" / "model_000005.pt", map_location="cpu",
                                   weights_only=True)["model_state"]
    assert saved_model_state["wanted_class_mask"].tolist() == [0.0, 1.0, 1.0, 1.0]

    # Report of the evaluation: both decodings, the ranks and the per-frame metrics, one row per wanted class
    results_blocks = read_results_blocks(results_path=trainer.experiment_folder / "Results.txt")
    assert [(block["iteration"], block["evaluated_samples"]) for block in results_blocks] == [
        (4, f"3 Test Sequences, {number_test_frames} Frames")]
    for metric_name in ["Hungarian Selection Accuracy", "Greedy Selection Accuracy", "Hungarian Fully Right Sequences",
                        "Greedy Unselected Recall", "Top-2 Selection Accuracy", "Mean True Frame Rank",
                        "Frame Accuracy"]:
        assert metric_name in results["metrics"]
    assert results["headline_metric"] == "Hungarian Selection Accuracy"
    assert [row[0] for row in results["tables"][0]["rows"]] == SELECTION_WANTED_SHAPES

    # Files of the evaluation folder
    evaluation_folder = trainer.experiment_folder / "Evaluation" / "Iteration_4"
    assert sorted(path.name for path in evaluation_folder.iterdir() if path.name != "Errors") == [
        "Samples", "class_metrics.csv", "class_thresholds.json", "confusion_matrix.csv", "confusion_matrix.png",
        "selections.csv"]
    thresholds = json.loads(s=(evaluation_folder / "class_thresholds.json").read_text())
    assert thresholds["decoding"] == "hungarian" and thresholds["unselected_class"] == "unselected"
    assert list(thresholds["classes"]) == SELECTION_WANTED_SHAPES

    # One selection row per test sequence and wanted class, each decoding choosing one frame of the sequence
    with open(file=evaluation_folder / "selections.csv", mode="r", newline="") as selections_file:
        selection_rows = list(csv.DictReader(f=selections_file))
    assert len(selection_rows) == 3 * len(SELECTION_WANTED_SHAPES)
    assert all(len(row["true_frames"].split(";")) == 1 and len(row["hungarian_frames"].split(";")) == 1 and
               row["true_frames"].startswith(row["sequence"]) for row in selection_rows)
    number_hungarian_right = sum(int(row["hungarian_right_selections"]) for row in selection_rows)
    assert number_hungarian_right / len(selection_rows) == pytest.approx(
        results["metrics"]["Hungarian Selection Accuracy"])

    # Error images in the folder of every wanted class picked wrong, named after their sequence
    wrong_rows = [row for row in selection_rows if row["hungarian_right_selections"] == "0"]
    for row in wrong_rows:
        error_image = evaluation_folder / "Errors" / row["class"] / f"{row['sequence']}.png"
        number_class_errors = sum(other_row["class"] == row["class"] for other_row in wrong_rows)
        assert error_image.exists() or number_class_errors > 2

    # Every test sequence is a sample image (8 asked for 3 sequences)
    check_sample_images(evaluation_folder=evaluation_folder, results=results, number_sample_images=3,
                        sample_kind="sequences", correct_predictions=None)

    # Second run with the same settings: same run folder, resumed from iteration 5 with exactly its weights
    resumed_trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=2))
    assert resumed_trainer.experiment_folder == trainer.experiment_folder
    assert resumed_trainer.start_iteration == 6
    for parameter_name, parameter_value in resumed_trainer.model.state_dict().items():
        assert torch.equal(input=parameter_value.cpu(), other=saved_model_state[parameter_name])
    resumed_trainer.run_training_loop()

    # The splits are described, and no iteration failed along the way
    console_output = capsys.readouterr().out
    assert "8 sequences" in console_output and "8 sequences of at most 6 frames" in console_output
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output
