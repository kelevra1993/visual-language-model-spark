"""
Evaluation of several checkpoints of a trained run on the test split, one after the other, writing
'Evaluation/evaluation_summary.csv' with one row per iteration and printing the best iteration according to the
headline metric of the task, tracked in its own MLflow run when the tracking is on. Called by
scripts/evaluation/evaluate_checkpoint.py with iteration "all".
"""

import csv
import torch

from pathlib import Path
from typing import Any, Dict, List

from vision_training.tracking.experiment_tracker import (FAILED_STATUS, FINISHED_STATUS, KILLED_STATUS,
                                                         RUN_RECORD_FILE_NAME, DisabledTracker, ExperimentTracker,
                                                         create_experiment_tracker, read_run_record)
from vision_training.trainers.base_trainer import BaseTrainer
from vision_training.trainers.checkpointing import get_checkpoint_file
from vision_training.utilities.os_utilities import format_path_link, print_color


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
    'Evaluation/evaluation_summary.csv' gathers one row per evaluated iteration with every evaluation scalar (those of
    every class only with the per-class report), and the best iteration according to the headline metric of the task
    is printed. A checkpoint listed in the registry but deleted from the disk is skipped. When the tracking is on, the
    evaluations, the best headline metric and the summary go to an MLflow run of their own (start_evaluation_tracking),
    closed whatever ends the evaluation.

    Args:
        trainer (BaseTrainer): The trainer of the run, built from its configuration, usually writing into a separate
            evaluation folder (see scripts/evaluation/evaluate_checkpoint.py).
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
    and prints the best iteration, for evaluate_checkpoints.

    Args:
        trainer (BaseTrainer): The trainer of the run.
        experiment_folder (Path): The trained run holding the 'Weights' folder of the checkpoints.
        iterations (List[int]): The iterations of the checkpoints to evaluate, in order.

    Returns:
        Path: The written evaluation summary.
    """
    summary_rows: List[Dict[str, Any]] = []
    headline_metric, best_iteration, best_value = None, None, None
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
        results = trainer.run_test_evaluation(iteration=iteration)
        summary_rows.append({"iteration": iteration, **results["scalars"],
                             **(results["class_scalars"] if trainer.per_class_report else {})})

        # Best iteration of the headline metric, the higher the better
        headline_metric = results["headline_metric"]
        if best_value is None or results["scalars"][headline_metric] > best_value:
            best_iteration, best_value = iteration, results["scalars"][headline_metric]

    # One row per evaluated iteration, one column per scalar in the order they first appear
    column_names = ["iteration"]
    for summary_row in summary_rows:
        column_names += [column_name for column_name in summary_row if column_name not in column_names]
    summary_path = trainer.experiment_folder / "Evaluation" / "evaluation_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file=summary_path, mode="w", newline="") as summary_file:
        writer = csv.DictWriter(f=summary_file, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(summary_rows)

    print_color(color="green", output=f"Evaluation Summary Of {len(summary_rows)} Checkpoints : "
                                      f"{format_path_link(path=summary_path)}", add_separators=True)
    if best_iteration is not None:
        print_color(color="green", output=f"Best Checkpoint : iteration {best_iteration} ({headline_metric} "
                                          f"{best_value:.4f})", indent=1)
    return summary_path
