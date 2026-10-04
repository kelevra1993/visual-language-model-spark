"""
Tests of evaluate_checkpoints of vision_training/evaluation/checkpoint_evaluation.py on the dummy task: every checkpoint
of the registry is evaluated, deleted checkpoints are skipped and the evaluation summary holds one row per iteration.
"""

import csv
import pytest

from pathlib import Path

from vision_training.evaluation.checkpoint_evaluation import evaluate_checkpoints
from vision_training.trainers.checkpointing import get_checkpoint_file, get_checkpoint_iterations
from tests.trainers.test_base_trainer import build_dummy_trainer


@pytest.mark.parametrize(argnames="per_class_report", argvalues=[True, False])
def test_every_checkpoint_is_evaluated_and_summarised(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                      per_class_report: bool) -> None:
    """
    Verifies the evaluation of every checkpoint of a run on the dummy task: the checkpoints of the registry (the
    intermediate one included) are evaluated in turn, a deleted checkpoint is skipped, the summary holds one row per
    evaluated iteration with every global evaluation scalar, and those of every class only with the per-class report,
    and the best iteration of the headline metric is printed.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
        per_class_report (bool): Whether the evaluations report every class.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=5, view_input_images=False)
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
    if per_class_report:
        assert column_names[2:] and all(column_name.startswith("Recall/") for column_name in column_names[2:])
    else:
        assert column_names == ["iteration", "Accuracy"]

    # The skipped checkpoint and the best iteration of the headline metric, the first one on ties, are reported
    console_output = capsys.readouterr().out
    best_row = summary_rows[0]
    for summary_row in summary_rows[1:]:
        if float(summary_row["Accuracy"]) > float(best_row["Accuracy"]):
            best_row = summary_row
    assert "Checkpoint Of Iteration 2 Not Found, Skipped" in console_output
    assert f"Best Checkpoint : iteration {best_row['iteration']} (Accuracy {float(best_row['Accuracy']):.4f})" in (
        console_output)
