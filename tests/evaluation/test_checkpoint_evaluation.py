"""
Tests of evaluate_checkpoints of vision_training/evaluation/checkpoint_evaluation.py on the dummy task: every checkpoint
of the registry is evaluated, deleted checkpoints are skipped and the evaluation summary holds one row per iteration,
with the columns and the best iteration of the averaged weights for a run keeping a moving average, and checkpoints
sharing the best value are all reported as a tie.
"""

import csv
import pytest

from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

from vision_training.evaluation.checkpoint_evaluation import evaluate_checkpoints
from vision_training.trainers.checkpointing import get_checkpoint_file, get_checkpoint_iterations
from tests.trainers.test_base_trainer import build_dummy_trainer


@pytest.mark.parametrize(argnames="per_class_report, moving_average_activated",
                         argvalues=[(True, False), (False, False), (True, True), (False, True)])
def test_every_checkpoint_is_evaluated_and_summarised(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                      per_class_report: bool, moving_average_activated: bool) -> None:
    """
    Verifies the evaluation of every checkpoint of a run on the dummy task: the checkpoints of the registry (the
    intermediate one included) are evaluated in turn, a deleted checkpoint is skipped, the summary holds one row per
    evaluated iteration with every global evaluation scalar, and those of every class only with the per-class report,
    and the best iteration of the headline metric is printed; a run keeping a moving average of its weights gets the
    same columns preceded by 'ema_' and the best iteration of its averaged weights.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
        per_class_report (bool): Whether the evaluations report every class.
        moving_average_activated (bool): Whether the run keeps a moving average of its weights.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=5, view_input_images=False,
                                  number_sample_images=0, moving_average_activated=moving_average_activated)
    trainer.per_class_report = per_class_report
    trainer.run_training_loop()
    iterations = get_checkpoint_iterations(weights_directory=trainer.weights_directory)
    assert iterations == [2, 4, 5]

    # The intermediate checkpoint of iteration 2 deleted from the disk, then every listed checkpoint evaluated
    get_checkpoint_file(weights_directory=trainer.weights_directory, iteration=2).unlink()
    capsys.readouterr()
    summary_path = evaluate_checkpoints(trainer=trainer, experiment_folder=trainer.experiment_folder,
                                        iterations=iterations)

    with open(file=summary_path, mode="r", newline="") as summary_file:
        summary_rows = list(csv.DictReader(f=summary_file))
    assert summary_path == trainer.experiment_folder / "Evaluation" / "evaluation_summary.csv"
    assert [row["iteration"] for row in summary_rows] == ["4", "5"]
    column_names = list(summary_rows[0])
    assert column_names[:2] == ["iteration", "Accuracy"]

    # Columns of the trained weights, followed by the same columns of the averaged weights
    trained_columns = [column_name for column_name in column_names[1:] if not column_name.startswith("ema_")]
    if per_class_report:
        assert trained_columns[1:] and all(column_name.startswith("Recall/") for column_name in trained_columns[1:])
    else:
        assert trained_columns == ["Accuracy"]
    expected_averaged_columns = [f"ema_{column_name}" for column_name in trained_columns] if (
        moving_average_activated) else []
    assert column_names == ["iteration"] + trained_columns + expected_averaged_columns

    # The skipped checkpoint and the best iterations of the headline metric, every tied one in the evaluation order,
    # are reported
    console_output = capsys.readouterr().out
    assert "Checkpoint Of Iteration 2 Not Found, Skipped" in console_output
    assert f"Best Checkpoint : {describe_expected_best(summary_rows=summary_rows, column_name='Accuracy')}" in (
        console_output)

    # Best iterations of the averaged weights, for a run keeping a moving average only
    assert ("Best Checkpoint (EMA Weights)" in console_output) == moving_average_activated
    if moving_average_activated:
        assert (f"Best Checkpoint (EMA Weights) : "
                f"{describe_expected_best(summary_rows=summary_rows, column_name='ema_Accuracy')}") in console_output


def describe_expected_best(summary_rows: List[Dict[str, str]], column_name: str) -> str:
    """
    Gives the expected best checkpoint description printed by summarise_checkpoint_evaluations, computed from the
    written evaluation summary: the best iteration, or every iteration sharing the best value in the evaluation order
    followed by 'tie', with the best value.

    Args:
        summary_rows (List[Dict[str, str]]): The rows of the evaluation summary, in the evaluation order.
        column_name (str): The column of the headline metric, 'ema_' prefixed for the averaged weights.

    Returns:
        str: The expected description, e.g. 'iteration 4 (Accuracy 0.5000)' or 'iterations 5, 4 tie (Accuracy 0.5000)'.
    """
    # Best value of the column and the iterations reaching it, in the evaluation order
    best_value = max(float(summary_row[column_name]) for summary_row in summary_rows)
    best_iterations = [summary_row["iteration"] for summary_row in summary_rows
                       if float(summary_row[column_name]) == best_value]
    iterations_description = f"iteration {best_iterations[0]}" if len(best_iterations) == 1 else (
        f"iterations {', '.join(best_iterations)} tie")
    return f"{iterations_description} (Accuracy {best_value:.4f})"


def test_evaluation_folder_keeps_the_name_of_the_evaluated_run(tmp_path: Path) -> None:
    """
    Verifies that a standalone evaluation writes into a folder with the name of the evaluated run folder, as
    scripts/evaluation/evaluate_checkpoint.py gives it, and not with the name built from its settings, which lose the
    tokens of the transfer of pretrained components once it is turned off: here the dummy trainer names its runs
    'dummy_run' and the evaluated run folder is 'dummy_run-PT'.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=4, view_input_images=False,
                                  number_sample_images=0, moving_average_activated=False)
    trainer.run_training_loop()

    # The trained run folder renamed as the run of a fine-tuned backbone, then evaluated from an evaluations folder
    evaluated_folder = trainer.experiment_folder.rename(target=trainer.experiment_folder.parent / "dummy_run-PT")
    (tmp_path / "evaluations").mkdir()
    evaluation_trainer = build_dummy_trainer(temporary_directory=tmp_path / "evaluations", number_iterations=4,
                                             view_input_images=False, number_sample_images=0,
                                             moving_average_activated=False, run_name=evaluated_folder.name)
    summary_path = evaluate_checkpoints(trainer=evaluation_trainer, experiment_folder=evaluated_folder,
                                        iterations=[4])

    output_folder = tmp_path / "evaluations" / "experiments" / "dummy_run-PT"
    assert evaluation_trainer.run_name == "dummy_run-PT"
    assert evaluation_trainer.experiment_folder == output_folder
    assert summary_path == output_folder / "Evaluation" / "evaluation_summary.csv" and summary_path.exists()
    assert not (tmp_path / "evaluations" / "experiments" / "dummy_run").exists()


def test_tied_checkpoints_are_all_reported(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that checkpoints sharing the best headline metric are all named in the best checkpoint line, in the
    evaluation order and followed by 'tie', instead of the first one evaluated being reported as the best.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=5, view_input_images=False,
                                  number_sample_images=0, moving_average_activated=False)
    trainer.run_training_loop()
    iterations = get_checkpoint_iterations(weights_directory=trainer.weights_directory)[::-1]
    assert iterations == [5, 4, 2]

    # Every checkpoint given the results of the last one, so that the three checkpoints share the best accuracy
    tied_results = trainer.run_test_evaluation(iteration=5)
    accuracy = tied_results[0]["scalars"]["Accuracy"]
    capsys.readouterr()
    with patch.object(target=trainer, attribute="run_test_evaluation", return_value=tied_results):
        evaluate_checkpoints(trainer=trainer, experiment_folder=trainer.experiment_folder, iterations=iterations)
    console_output = capsys.readouterr().out
    assert f"Best Checkpoint : iterations 5, 4, 2 tie (Accuracy {accuracy:.4f})" in console_output
    assert "Best Checkpoint : iteration " not in console_output
