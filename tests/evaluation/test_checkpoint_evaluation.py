"""
Tests of evaluate_checkpoints of vision_training/evaluation/checkpoint_evaluation.py on the dummy task: every checkpoint
of the registry is evaluated, deleted checkpoints are skipped and the evaluation summary holds one row per iteration,
with the columns and the best iteration of the averaged weights for a run keeping a moving average, a standalone
evaluation (scripts/evaluation/evaluate_checkpoint.py) writes only the Checkpoint_Evaluation folder of the run, its
evaluation curves included, the summary of its folder keeping the rows of the earlier standalone evaluations, and the
best checkpoint line writes as a percentage a headline metric kept outside the metrics grid (the keypoint detection).
The evaluation of a run moved since its training is tracked in the backend the launch panel names, an evaluation of
nothing fails with its MLflow run, and the error images of an earlier evaluation of the same weights are replaced.
"""

import csv
import functools
import pytest
import shutil

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from vision_training.core.contracts import EvaluationResults
from vision_training.evaluation import checkpoint_evaluation
from vision_training.evaluation.checkpoint_evaluation import CHECKPOINT_EVALUATION_FOLDER_NAME, evaluate_checkpoints
from vision_training.tracking.experiment_tracker import FAILED_STATUS, DisabledTracker, ExperimentTracker
from vision_training.trainers.checkpointing import get_checkpoint_file, get_checkpoint_iterations
from tests.trainers.test_base_trainer import build_dummy_trainer


@pytest.mark.parametrize(argnames="per_class_report, moving_average_activated", argvalues=[(True, True)])
def test_every_checkpoint_is_evaluated_and_summarised(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                      per_class_report: bool, moving_average_activated: bool) -> None:
    """
    Verifies the evaluation of every checkpoint of a run on the dummy task: the checkpoints of the registry (the
    intermediate one included) are evaluated in turn, a deleted checkpoint is skipped, the summary holds one row per
    evaluated iteration with every global evaluation scalar, and those of every class only with the per-class report,
    and the best iteration of the headline metric is printed; a run keeping a moving average of its weights gets the
    same columns preceded by 'ema_', the number of updates of the averaged weights restored with every checkpoint, and
    the best iteration of its averaged weights.

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
    expected_averaged_columns = [f"ema_{column_name}" for column_name in trained_columns] + ["ema_number_updates"] if (
        moving_average_activated) else []
    assert column_names == ["iteration"] + trained_columns + expected_averaged_columns

    # Number of updates of the averaged weights restored with every checkpoint, one per training iteration here
    if moving_average_activated:
        assert [row["ema_number_updates"] for row in summary_rows] == ["4", "5"]

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


def test_standalone_evaluation_writes_only_its_checkpoint_evaluation_folder(tmp_path: Path) -> None:
    """
    Verifies that a standalone evaluation of a trained run, pointed to the run as
    scripts/evaluation/evaluate_checkpoint.py does, writes its iteration folders, its summary and its evaluation curves
    directly in '<run>/Checkpoint_Evaluation/', next to the Evaluation folder of the training, without a Weights folder
    of its own, and leaves every other file of the run as it was.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=5, view_input_images=False,
                                  number_sample_images=0)
    trainer.run_training_loop()
    run_folder = trainer.experiment_folder
    run_files_before = list_run_files(run_folder=run_folder)

    # Standalone evaluation of the last two checkpoints of the run
    evaluation_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=5, view_input_images=False,
                                             number_sample_images=0, evaluated_run_folder=run_folder)
    summary_path = evaluate_checkpoints(trainer=evaluation_trainer, experiment_folder=run_folder, iterations=[4, 5])

    # Iteration folders and summary directly in the Checkpoint_Evaluation folder, with the results and test curves,
    # those of the training staying in its Evaluation folder
    evaluation_folder = run_folder / CHECKPOINT_EVALUATION_FOLDER_NAME
    assert evaluation_trainer.experiment_folder == evaluation_trainer.evaluation_root == evaluation_folder
    assert evaluation_trainer.get_evaluation_folder(iteration=4) == evaluation_folder / "Iteration_4"
    assert trainer.get_evaluation_folder(iteration=4) == run_folder / "Evaluation" / "Iteration_4"
    assert summary_path == evaluation_folder / "evaluation_summary.csv"
    assert (evaluation_folder / "Results.txt").is_file() and (evaluation_folder / "Tensorboard" / "Test").is_dir()
    assert (evaluation_folder / "evaluation_curves.png").is_file()
    assert not (evaluation_folder / "Evaluation").exists() and not (evaluation_folder / "Weights").exists()

    # The files of the run itself untouched
    assert list_run_files(run_folder=run_folder) == run_files_before


def test_standalone_evaluations_add_their_rows_to_the_summary_of_their_folder(tmp_path: Path,
                                                                             capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the summary of the Checkpoint_Evaluation folder keeps the rows of every standalone evaluation written
    into it, in increasing iteration order: a launch evaluating iteration 5, then one evaluating iterations 4 and 2,
    leave the rows of iterations 2, 4 and 5, while the best checkpoint printed by each launch compares its own
    checkpoints only.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=5, view_input_images=False,
                                  number_sample_images=0)
    trainer.run_training_loop()
    run_folder = trainer.experiment_folder

    # Two standalone launches, the second one evaluating earlier checkpoints than the first one
    launched_iterations = [[5], [4, 2]]
    for iterations in launched_iterations:
        evaluation_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=5,
                                                 view_input_images=False, number_sample_images=0,
                                                 evaluated_run_folder=run_folder)
        capsys.readouterr()
        summary_path = evaluate_checkpoints(trainer=evaluation_trainer, experiment_folder=run_folder,
                                            iterations=iterations)
        console_output = capsys.readouterr().out
        assert f"Evaluation Summary Of {len(iterations)} Checkpoints : " in console_output

    # The best checkpoint of the last launch among its own checkpoints, the summary holding the rows of both launches
    best_checkpoint_line = next(line for line in console_output.splitlines() if "Best Checkpoint : " in line)
    assert "iteration 5" not in best_checkpoint_line
    with open(file=summary_path, mode="r", newline="") as summary_file:
        summary_rows = list(csv.DictReader(f=summary_file))
    assert [row["iteration"] for row in summary_rows] == ["2", "4", "5"]
    assert all(row["Accuracy"] != "" for row in summary_rows)


def move_headline_out_of_grid(original_compute: Callable[..., EvaluationResults], accumulator: Dict[str, Any],
                              iteration: int) -> EvaluationResults:
    """
    Computes the results of the dummy task with its headline accuracy kept as a scalar outside the metrics grid, which
    holds a pixel error instead, like the keypoint detection whose grid holds its keypoint accuracy while its headline
    OKS mAP@0.5/area is a percent metric of the best checkpoint line only.

    Args:
        original_compute (Callable[..., EvaluationResults]): The compute_evaluation_results method of the dummy trainer.
        accumulator (Dict[str, Any]): The accumulated predictions.
        iteration (int): The training iteration of the evaluated checkpoint.

    Returns:
        EvaluationResults: The results, the accuracy in the scalars and the percent metrics but not in the grid.
    """
    results = original_compute(accumulator=accumulator, iteration=iteration)
    return {**results, "metrics": {"Mean Error (pixels)": 1.5}}


def test_headline_metric_outside_the_grid_is_written_as_a_percentage(tmp_path: Path,
                                                                      capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a task whose headline metric is a scalar outside the metrics grid, listed among its percent metrics
    like the OKS mAP@0.5/area of the keypoint detection, gets its report written, the grid value that is not a rate with
    4 decimals, and its best checkpoint line written as a percentage, the summary keeping the value between 0 and 1.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=5, view_input_images=False,
                                  number_sample_images=0)
    trainer.run_training_loop()

    # Evaluations of the last two checkpoints with the headline accuracy outside the grid
    trainer.compute_evaluation_results = functools.partial(move_headline_out_of_grid,
                                                           original_compute=trainer.compute_evaluation_results)
    capsys.readouterr()
    summary_path = evaluate_checkpoints(trainer=trainer, experiment_folder=trainer.experiment_folder, iterations=[4, 5])
    with open(file=summary_path, mode="r", newline="") as summary_file:
        summary_rows = list(csv.DictReader(f=summary_file))
    console_output = capsys.readouterr().out
    assert f"Best Checkpoint : {describe_expected_best(summary_rows=summary_rows, column_name='Accuracy')}" in (
        console_output)
    assert "Mean Error (pixels)   1.5000" in (trainer.experiment_folder / "Results.txt").read_text(encoding="utf-8")


def list_run_files(run_folder: Path) -> Dict[str, Tuple[int, int]]:
    """
    Lists the files of a run folder outside its Checkpoint_Evaluation folder with their size and modification time, to
    check that a standalone evaluation leaves the files of the evaluated run as they were.

    Args:
        run_folder (Path): The run folder.

    Returns:
        Dict[str, Tuple[int, int]]: The size and modification time in nanoseconds of every file, by relative path.
    """
    return {path.relative_to(run_folder).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in run_folder.rglob(pattern="*")
            if path.is_file() and path.relative_to(run_folder).parts[0] != CHECKPOINT_EVALUATION_FOLDER_NAME}


def describe_expected_best(summary_rows: List[Dict[str, str]], column_name: str) -> str:
    """
    Gives the expected best checkpoint description printed by summarise_checkpoint_evaluations, computed from the
    written evaluation summary: the best iteration, or every iteration sharing the best value in the evaluation order
    followed by 'tie', with the best value written as a percentage, the accuracy of the dummy task being one of its
    percent metrics while the summary keeps it between 0 and 1.

    Args:
        summary_rows (List[Dict[str, str]]): The rows of the evaluation summary, in the evaluation order.
        column_name (str): The column of the headline metric, 'ema_' prefixed for the averaged weights.

    Returns:
        str: The expected description, e.g. 'iteration 4 (Accuracy 50.00 %)' or 'iterations 5, 4 tie (Accuracy
            50.00 %)'.
    """
    # Best value of the column and the iterations reaching it, in the evaluation order
    best_value = max(float(summary_row[column_name]) for summary_row in summary_rows)
    best_iterations = [summary_row["iteration"] for summary_row in summary_rows
                       if float(summary_row[column_name]) == best_value]
    iterations_description = f"iteration {best_iterations[0]}" if len(best_iterations) == 1 else (
        f"iterations {', '.join(best_iterations)} tie")
    # The summary keeps the accuracy between 0 and 1, the console line writes it as a percentage
    assert 0.0 <= best_value <= 1.0
    return f"{iterations_description} (Accuracy {100.0 * best_value:.2f} %)"


class RecordingTracker(DisabledTracker):
    """
    Tracker of the tests of the standalone evaluation, doing nothing like DisabledTracker in place of the MLflow
    tracker built by create_experiment_tracker with the tracking on, but keeping the experiments root folder of the
    backend it was created for and the status every close gave its run.
    """

    def __init__(self, tracking_root: Path) -> None:
        """
        Keeps the backend folder of the tracker, its run being closed later by the evaluation.

        Args:
            tracking_root (Path): The experiments root folder whose local database would receive the run.
        """
        self.tracking_root = tracking_root
        self.closed_statuses: List[str] = []

    def close(self, status: str) -> None:
        """
        Keeps the status the evaluation closes its MLflow run with.

        Args:
            status (str): The status of the run, e.g. FINISHED_STATUS or FAILED_STATUS.
        """
        self.closed_statuses.append(status)


class RecordingTrackerFactory:
    """
    Replacement of create_experiment_tracker in the checkpoint evaluation module, giving a RecordingTracker and keeping
    every tracker it gave, so that the tests read the backend and the closing status of the evaluation run without
    MLflow.
    """

    def __init__(self) -> None:
        """
        Starts without any tracker given.
        """
        self.trackers: List[RecordingTracker] = []

    def __call__(self, tracking_configuration: Dict[str, Any], tracking_root: Path, run_name: str, task: str,
                 run_record_path: Optional[Path]) -> ExperimentTracker:
        """
        Gives the tracker of a standalone evaluation, with the arguments of create_experiment_tracker.

        Args:
            tracking_configuration (Dict[str, Any]): The 'ExperimentConfiguration.tracking' section, unused.
            tracking_root (Path): The experiments root folder of the backend, kept by the tracker.
            run_name (str): The name of the MLflow run, unused.
            task (str): The task of the run, unused.
            run_record_path (Optional[Path]): The record of the run, None for an evaluation, unused.

        Returns:
            ExperimentTracker: The recording tracker.
        """
        tracker = RecordingTracker(tracking_root=tracking_root)
        self.trackers.append(tracker)
        return tracker


def test_evaluation_of_a_moved_run_is_tracked_in_the_backend_of_the_launch_panel(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the standalone evaluation of a run moved to an archive folder after its training is tracked in the
    backend of the experiments root folder of its configuration ('tracking_root', the one the launch panel names and
    the training run was tracked in), not in a database of the archive folder.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the tracker factory of the evaluation.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=5, view_input_images=False,
                                  number_sample_images=0)
    trainer.run_training_loop()
    archived_run_folder = tmp_path / "archive" / trainer.experiment_folder.name
    archived_run_folder.parent.mkdir()
    shutil.move(src=trainer.experiment_folder, dst=archived_run_folder)

    # Standalone evaluation of the archived run, pointed to it as scripts/evaluation/evaluate_checkpoint.py does
    evaluation_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=5,
                                             view_input_images=False, number_sample_images=0,
                                             evaluated_run_folder=archived_run_folder)
    tracker_factory = RecordingTrackerFactory()
    monkeypatch.setattr(target=checkpoint_evaluation, name="create_experiment_tracker", value=tracker_factory)
    evaluate_checkpoints(trainer=evaluation_trainer, experiment_folder=archived_run_folder, iterations=[5])
    assert evaluation_trainer.experiment_configuration["tracking_root"] == tmp_path / "experiments"
    assert [tracker.tracking_root for tracker in tracker_factory.trackers] == [tmp_path / "experiments"]


def test_evaluation_without_any_checkpoint_on_the_disk_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that evaluating checkpoints that are all missing from the disk, e.g. those of a run stopped before its
    first save, raises an error naming the Weights folder instead of ending normally, its MLflow run closed as failed.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the tracker factory of the evaluation.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=5, view_input_images=False,
                                  number_sample_images=0)
    trainer.run_training_loop()
    iterations = get_checkpoint_iterations(weights_directory=trainer.weights_directory)
    for iteration in iterations:
        get_checkpoint_file(weights_directory=trainer.weights_directory, iteration=iteration).unlink()

    # Every listed checkpoint skipped, then the evaluation of nothing refused
    tracker_factory = RecordingTrackerFactory()
    monkeypatch.setattr(target=checkpoint_evaluation, name="create_experiment_tracker", value=tracker_factory)
    with pytest.raises(expected_exception=FileNotFoundError, match="No Checkpoint Evaluated"):
        evaluate_checkpoints(trainer=trainer, experiment_folder=trainer.experiment_folder, iterations=iterations[::-1])
    assert tracker_factory.trackers[0].closed_statuses == [FAILED_STATUS]


def add_errors_folder(original_create: Callable[..., Dict[str, Any]], trainer: Any,
                      iteration: int) -> Dict[str, Any]:
    """
    Creates the evaluation accumulator of the dummy task with the folder and counts of the error images of the tasks
    saving them (classification, detection, frame selection), so that the base trainer handles the 'Errors' folder
    of its evaluations as it does for them.

    Args:
        original_create (Callable[..., Dict[str, Any]]): The create_evaluation_accumulator method of the dummy trainer.
        trainer (Any): The dummy trainer, giving the evaluation folder of the iteration.
        iteration (int): The training iteration of the evaluated checkpoint.

    Returns:
        Dict[str, Any]: The accumulator of the dummy task, with its 'errors_folder' and empty 'error_counts'.
    """
    return {**original_create(iteration=iteration),
            "errors_folder": trainer.get_evaluation_folder(iteration=iteration) / "Errors", "error_counts": {}}


@pytest.mark.parametrize(argnames="error_images_activated", argvalues=[True, False])
def test_evaluation_saving_error_images_replaces_those_of_an_earlier_evaluation(tmp_path: Path,
                                                                               error_images_activated: bool) -> None:
    """
    Verifies that a standalone evaluation saving error images first removes the error images an earlier evaluation of
    the same checkpoint left in its 'Errors' folder (e.g. with a larger limit per pair of classes), so that the folder
    never mixes two evaluations, and that an evaluation saving none leaves them.

    Args:
        tmp_path (Path): The pytest temporary directory.
        error_images_activated (bool): Whether the evaluation saves error images.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=5, view_input_images=False,
                                  number_sample_images=0)
    trainer.run_training_loop()
    evaluation_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=5,
                                             view_input_images=False, number_sample_images=0,
                                             evaluated_run_folder=trainer.experiment_folder)
    evaluation_trainer.error_dumps_configuration = {"activated": error_images_activated, "maximum_per_class_pair": 1}
    evaluation_trainer.create_evaluation_accumulator = functools.partial(
        add_errors_folder, original_create=evaluation_trainer.create_evaluation_accumulator,
        trainer=evaluation_trainer)

    # Error image of an earlier evaluation of iteration 5, then the evaluation of the same checkpoint
    earlier_image_path = (evaluation_trainer.get_evaluation_folder(iteration=5) / "Errors" / "positive-TO-negative" /
                          "sample_3_0.91.png")
    earlier_image_path.parent.mkdir(parents=True)
    earlier_image_path.write_bytes(data=b"")
    evaluate_checkpoints(trainer=evaluation_trainer, experiment_folder=trainer.experiment_folder, iterations=[5])
    assert earlier_image_path.exists() != error_images_activated
