"""
Evaluation of several checkpoints of a trained run on the test split, one after the other, writing
'evaluation_summary.csv' next to the iteration folders of the evaluations with one row per iteration (the averaged
weights of the runs keeping a moving average in 'ema_' columns) and printing the best iteration according to the
headline metric of the task, for the trained and the averaged weights, tracked in its own MLflow run when the tracking
is on. Called by scripts/evaluation/evaluate_checkpoint.py, which writes into '<run>/Checkpoint_Evaluation/'.
"""

import csv
import torch

from pathlib import Path
from typing import Any, Dict, List, Tuple

from vision_training.core.contracts import EvaluationResults
from vision_training.evaluation.report_formatting import format_report_value
from vision_training.tracking.experiment_tracker import (FAILED_STATUS, FINISHED_STATUS, KILLED_STATUS,
                                                         RUN_RECORD_FILE_NAME, DisabledTracker, ExperimentTracker,
                                                         create_experiment_tracker, read_run_record)
from vision_training.trainers.base_trainer import AVERAGED_WEIGHTS_LABEL, BaseTrainer
from vision_training.trainers.checkpointing import get_checkpoint_file
from vision_training.utilities.os_utilities import format_path_link, print_color

# Prefix of the columns of the averaged weights in the evaluation summary, next to those of the trained weights
AVERAGED_WEIGHTS_COLUMN_PREFIX = "ema_"

# Folder of the evaluations of scripts/evaluation/evaluate_checkpoint.py inside the evaluated run folder, next to the
# 'Evaluation' folder of its training: '<run>/Checkpoint_Evaluation/', holding the iteration folders directly, where the
# export also looks for the thresholds saved from the temperature explorer
CHECKPOINT_EVALUATION_FOLDER_NAME = "Checkpoint_Evaluation"


def start_evaluation_tracking(trainer: BaseTrainer, experiment_folder: Path,
                              iterations: List[int]) -> ExperimentTracker:
    """
    Opens the MLflow run of a standalone evaluation when 'tracking.activated' is True: a new run named
    '<evaluated run name>-evaluation' in the backend and experiment of the evaluated training, with the parameters of
    the trainer, tagged with the MLflow run of the evaluated training ('evaluation_of', absent for a run trained
    without tracking), the evaluated experiment folder, the evaluated test split and the evaluated iterations. The
    training run is only read, so that an evaluation on another split never mixes into its curves.

    Args:
        trainer (BaseTrainer): The trainer of the evaluation, whose tracker receives the MLflow run.
        experiment_folder (Path): The evaluated training run, holding its 'mlflow_run.json' when it was tracked.
        iterations (List[int]): The iterations of the evaluated checkpoints.

    Returns:
        ExperimentTracker: The tracker of the evaluation run, a DisabledTracker when the tracking is off.
    """
    evaluation_tracker = create_experiment_tracker(
        tracking_configuration=trainer.tracking_configuration, tracking_root=experiment_folder.parent,
        run_name=f"{experiment_folder.name}-evaluation", task=trainer.TASK_NAME, run_record_path=None)

    # Tags linking the evaluation to the evaluated training and to the evaluated split
    tags = {**trainer.build_tracking_tags(), "evaluated_experiment_folder": str(experiment_folder.absolute()),
            "test_data_folder": str(trainer.experiment_configuration["test_data_folder"]),
            "evaluated_iterations": ", ".join(str(iteration) for iteration in iterations)}
    training_run_record = read_run_record(run_record_path=experiment_folder / RUN_RECORD_FILE_NAME)
    if training_run_record is not None:
        tags["evaluation_of"] = training_run_record["run_id"]

    evaluation_tracker.start(parameters=trainer.build_tracking_parameters(), tags=tags, resumed_iteration=None)
    return evaluation_tracker


def evaluate_checkpoints(trainer: BaseTrainer, experiment_folder: Path, iterations: List[int]) -> Path:
    """
    Evaluates checkpoints of a trained run one after the other on the test split of the trainer, to choose the
    checkpoint to deploy: every checkpoint is loaded into the model of the trainer, with its moving average of the
    weights when the configuration keeps one, and evaluated like during the training (Results.txt, TensorBoard, the
    evaluation files of the task in the run folder of the trainer, the averaged weights next to the trained ones), then
    'evaluation_summary.csv', next to the iteration folders (evaluation_root of the trainer), gathers one row per
    evaluated iteration with every evaluation scalar (those of every class only with the per-class report, the averaged
    weights in 'ema_' columns), and the best iteration
    according to the headline metric of the task is printed, for the trained and the averaged weights. A checkpoint
    listed in the registry but deleted from the disk is skipped. When the tracking is on, the evaluations, the best
    headline metric and the summary go to an MLflow run of their own (start_evaluation_tracking), closed whatever ends
    the evaluation.

    Args:
        trainer (BaseTrainer): The trainer of the run, built from its configuration, usually writing into the
            '<run>/Checkpoint_Evaluation/' folder of a standalone evaluation
            (scripts/evaluation/evaluate_checkpoint.py).
        experiment_folder (Path): The trained run holding the 'Weights' folder of the checkpoints.
        iterations (List[int]): The iterations of the checkpoints to evaluate, in order.

    Returns:
        Path: The written evaluation summary.
    """
    # The evaluations reported by the trainer are tracked in the MLflow run of the evaluation
    trainer.experiment_tracker = start_evaluation_tracking(trainer=trainer, experiment_folder=experiment_folder,
                                                           iterations=iterations)
    run_status = FAILED_STATUS
    try:
        summary_path = summarise_checkpoint_evaluations(trainer=trainer, experiment_folder=experiment_folder,
                                                        iterations=iterations)
        trainer.experiment_tracker.log_artifact(file_path=summary_path, artifact_folder=None)
        run_status = FINISHED_STATUS
    except KeyboardInterrupt:
        run_status = KILLED_STATUS
        raise
    finally:
        trainer.experiment_tracker.close(status=run_status)
        trainer.experiment_tracker = DisabledTracker()
    return summary_path


def summarise_checkpoint_evaluations(trainer: BaseTrainer, experiment_folder: Path, iterations: List[int]) -> Path:
    """
    Evaluates the checkpoints one after the other through the trainer, writes the evaluation summary of their scalars
    (between 0 and 1) and prints the best iteration, or every tied iteration when several checkpoints share the best
    value, for evaluate_checkpoints, the value written as a percentage when the headline metric is one of the percent
    metrics of the task ("Accuracy 94.03 %").

    Args:
        trainer (BaseTrainer): The trainer of the run.
        experiment_folder (Path): The trained run holding the 'Weights' folder of the checkpoints.
        iterations (List[int]): The iterations of the checkpoints to evaluate, in order.

    Returns:
        Path: The written evaluation summary.
    """
    summary_rows: List[Dict[str, Any]] = []
    headline_metric, headline_in_percent = None, False
    best_checkpoints: Dict[str, Tuple[List[int], float]] = {}
    for iteration in iterations:
        checkpoint_path = get_checkpoint_file(weights_directory=experiment_folder / "Weights", iteration=iteration)
        if not checkpoint_path.exists():
            print_color(color="yellow", output=f"Checkpoint Of Iteration {iteration} Not Found, Skipped : "
                                               f"{format_path_link(path=checkpoint_path)}")
            continue

        # Weights of the checkpoint in the model of the trainer, with their moving average, then the evaluation of the
        # test split
        checkpoint = torch.load(f=checkpoint_path, map_location=trainer.device, weights_only=True)
        trainer.model.load_state_dict(state_dict=checkpoint["model_state"])
        trainer.load_exponential_moving_average(checkpoint=checkpoint, iteration=iteration)
        print_color(color="blue", output=f"Evaluating : {format_path_link(path=checkpoint_path)}", add_separators=True)
        results, averaged_weights_results = trainer.run_test_evaluation(iteration=iteration)
        summary_row = {"iteration": iteration, **build_summary_columns(trainer=trainer, results=results, prefix="")}

        # Scalars of the averaged weights in their own 'ema_' columns, for the runs keeping a moving average
        evaluated_weights = {"": results}
        if averaged_weights_results is not None:
            summary_row.update(build_summary_columns(trainer=trainer, results=averaged_weights_results,
                                                     prefix=AVERAGED_WEIGHTS_COLUMN_PREFIX))
            evaluated_weights[AVERAGED_WEIGHTS_LABEL] = averaged_weights_results
        summary_rows.append(summary_row)

        # Best iterations of the headline metric for every kind of weights, the higher the better, the iterations
        # sharing the best value being all kept so that a tie is reported instead of one of them being picked
        headline_metric = results["headline_metric"]
        headline_in_percent = headline_metric in results["percent_metrics"]
        for weights_label, weights_results in evaluated_weights.items():
            value = weights_results["scalars"][headline_metric]
            if weights_label not in best_checkpoints or value > best_checkpoints[weights_label][1]:
                best_checkpoints[weights_label] = ([iteration], value)
            elif value == best_checkpoints[weights_label][1]:
                best_checkpoints[weights_label][0].append(iteration)

    # One row per evaluated iteration, one column per scalar in the order they first appear
    column_names = ["iteration"]
    for summary_row in summary_rows:
        column_names += [column_name for column_name in summary_row if column_name not in column_names]
    summary_path = trainer.evaluation_root / "evaluation_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file=summary_path, mode="w", newline="") as summary_file:
        writer = csv.DictWriter(f=summary_file, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(summary_rows)

    print_color(color="green", output=f"Evaluation Summary Of {len(summary_rows)} Checkpoints : "
                                      f"{format_path_link(path=summary_path)}", add_separators=True)
    for weights_label, (best_iterations, best_value) in best_checkpoints.items():
        # One best iteration, or the tied iterations in their evaluation order
        weights_description = f" ({weights_label})" if weights_label else ""
        iterations_description = f"iteration {best_iterations[0]}"
        if len(best_iterations) > 1:
            tied_iterations = ", ".join(str(best_iteration) for best_iteration in best_iterations)
            iterations_description = f"iterations {tied_iterations} tie"
        # Best value written like in the report, as a percentage when the headline metric is a rate of the task
        best_text = format_report_value(value=best_value, in_percent=headline_in_percent)
        print_color(color="green", output=f"Best Checkpoint{weights_description} : {iterations_description} "
                                          f"({headline_metric} {best_text})", indent=1)
    return summary_path


def build_summary_columns(trainer: BaseTrainer, results: EvaluationResults, prefix: str) -> Dict[str, float]:
    """
    Gives the columns of one kind of weights in a row of the evaluation summary: the global scalars of the evaluation
    and, with the per-class report, those of every class, each name preceded by the prefix of the weights ('' for the
    trained weights, 'ema_' for the averaged weights).

    Args:
        trainer (BaseTrainer): The trainer of the run, giving the per-class report setting.
        results (EvaluationResults): The results of the evaluation of the weights.
        prefix (str): The prefix of the column names.

    Returns:
        Dict[str, float]: The value of every column.
    """
    scalars = {**results["scalars"], **(results["class_scalars"] if trainer.per_class_report else {})}
    return {f"{prefix}{scalar_name}": value for scalar_name, value in scalars.items()}
