"""
Evaluation of several checkpoints of a trained run on the test split, one after the other, writing
'Evaluation/evaluation_summary.csv' with one row per iteration and printing the best iteration according to the
headline metric of the task. Called by scripts/evaluation/evaluate_checkpoint.py with iteration "all".
"""

import csv
import torch

from pathlib import Path
from typing import Any, Dict, List

from vision_training.trainers.base_trainer import BaseTrainer
from vision_training.trainers.checkpointing import get_checkpoint_file
from vision_training.utilities.os_utilities import print_color


def evaluate_checkpoints(trainer: BaseTrainer, experiment_folder: Path, iterations: List[int]) -> Path:
    """
    Evaluates checkpoints of a trained run one after the other on the test split of the trainer, to choose the
    checkpoint to deploy: every checkpoint is loaded into the model of the trainer and evaluated like during the
    training (Results.txt, TensorBoard, the evaluation files of the task in the run folder of the trainer), then
    'Evaluation/evaluation_summary.csv' gathers one row per evaluated iteration with every evaluation scalar, and the
    best iteration according to the headline metric of the task is printed. A checkpoint listed in the registry but
    deleted from the disk is skipped.

    Args:
        trainer (BaseTrainer): The trainer of the run, built from its configuration, usually writing into a separate
            evaluation folder (see scripts/evaluation/evaluate_checkpoint.py).
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
                                               f"file://{checkpoint_path.absolute()}")
            continue

        # Weights of the checkpoint in the model of the trainer, then the evaluation of the test split
        checkpoint = torch.load(f=checkpoint_path, map_location=trainer.device, weights_only=True)
        trainer.model.load_state_dict(state_dict=checkpoint["model_state"])
        print_color(color="blue", output=f"Evaluating : file://{checkpoint_path.absolute()}", add_separators=True)
        results = trainer.run_test_evaluation(iteration=iteration)
        summary_rows.append({"iteration": iteration, **results["scalars"]})

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
                                      f"file://{summary_path.absolute()}", add_separators=True)
    if best_iteration is not None:
        print_color(color="green", output=f"Best Checkpoint : iteration {best_iteration} ({headline_metric} "
                                          f"{best_value:.4f})", indent=1)
    return summary_path
