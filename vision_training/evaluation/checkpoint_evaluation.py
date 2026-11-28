"""
Evaluation of several checkpoints of a trained run on the test split, one after the other, printing the best iteration
according to the headline metric of the task, for the trained and the averaged weights, tracked in its own MLflow run
when the tracking is on. Every evaluation adds its row to 'evaluation_summary.csv', next to the iteration folders of the
evaluations (the averaged weights of the runs keeping a moving average in 'ema_' columns), and draws
'evaluation_curves.png' again next to it, as during a training (evaluation/evaluation_curves.py). Called by
scripts/evaluation/evaluate_checkpoint.py, which writes into '<run>/Checkpoint_Evaluation/'. Also gives the export and
the local inference of deployment/ the file of the evaluation of an exported checkpoint (find_evaluation_file), the
class thresholds of a frame selector, detector or instance segmenter, or the thresholds saved from the temperature
explorer of a classifier, the standalone evaluation taking precedence over the evaluation of the training.
"""

import torch

from pathlib import Path
from typing import Dict, List, Optional

from vision_training.evaluation.evaluation_curves import (EVALUATION_CURVES_FIGURE_NAME, EVALUATION_SUMMARY_FILE_NAME,
                                                          find_best_iterations)
from vision_training.evaluation.report_formatting import format_report_value
from vision_training.tracking.experiment_tracker import (FAILED_STATUS, FINISHED_STATUS, KILLED_STATUS,
                                                         RUN_RECORD_FILE_NAME, DisabledTracker, ExperimentTracker,
                                                         create_experiment_tracker, read_run_record)
from vision_training.trainers.base_trainer import (AVERAGED_WEIGHTS_FOLDER_NAME, AVERAGED_WEIGHTS_LABEL,
                                                   EVALUATION_FOLDER_NAME, BaseTrainer)
from vision_training.trainers.checkpointing import get_checkpoint_file
from vision_training.utilities.os_utilities import format_path_link, print_color

# Folder of the evaluations of scripts/evaluation/evaluate_checkpoint.py inside the evaluated run folder, next to the
# 'Evaluation' folder of its training: '<run>/Checkpoint_Evaluation/', holding the iteration folders directly, where the
# export also looks for the thresholds of the evaluation of the exported checkpoint
CHECKPOINT_EVALUATION_FOLDER_NAME = "Checkpoint_Evaluation"


def find_evaluation_file(experiment_folder: Path, checkpoint_path: Path, use_averaged_weights: bool,
                         file_name: str) -> Optional[Path]:
    """
    Finds a file of the evaluation of an exported checkpoint, read by the export (deployment/export_model.py) and the
    local inference (deployment/deployment_utilities.py): the confidence thresholds of a frame selector deployed with
    its decoder, the score thresholds of the classes built into a detector or instance segmenter (both
    'class_thresholds.json'), or the thresholds saved from the temperature explorer of a classifier
    ('saved_temperature_thresholds.json'). The file is looked for in the folder of the iteration of the checkpoint,
    in its 'EMA' subfolder for the averaged weights, in '<run>/Checkpoint_Evaluation/' (scripts/evaluation/
    evaluate_checkpoint.py) and in '<run>/Evaluation/' (evaluation of the training). When both hold it, the one of
    Checkpoint_Evaluation is taken, the deliberate standalone evaluation of that checkpoint made after its training
    (with the current code, on the chosen split and target precision), and a yellow line names both files and the one
    taken, so that the export never stops on the evaluation of the training the run keeps anyway.

    Args:
        experiment_folder (Path): The exported run folder.
        checkpoint_path (Path): The exported checkpoint, '<run>/Weights/Iteration_<n>/model_<n>.pt'.
        use_averaged_weights (bool): True to read the evaluation of the moving average of the weights (the 'EMA'
            subfolder of the iteration folder), False for the trained weights.
        file_name (str): The name of the evaluation file.

    Returns:
        Optional[Path]: The file of the evaluation of the exported checkpoint and weights, None when neither
            evaluation folder holds it.
    """
    # Folder of the iteration and weights in the folder of the standalone evaluation, then of the training
    weights_folders = [AVERAGED_WEIGHTS_FOLDER_NAME] if use_averaged_weights else []
    candidate_paths = [Path(experiment_folder, evaluation_folder_name, checkpoint_path.parent.name, *weights_folders,
                            file_name)
                       for evaluation_folder_name in [CHECKPOINT_EVALUATION_FOLDER_NAME, EVALUATION_FOLDER_NAME]]
    found_paths = [candidate_path for candidate_path in candidate_paths if candidate_path.is_file()]
    if not found_paths:
        return None

    # Both evaluations hold the file: the standalone one is taken, both named in the console
    if len(found_paths) > 1:
        used_link, unused_link = [format_path_link(path=found_path,
                                                   label=str(found_path.relative_to(experiment_folder)))
                                  for found_path in found_paths]
        print_color(color="yellow", output=f"Two {file_name} for the exported checkpoint, the one of "
                                           f"{CHECKPOINT_EVALUATION_FOLDER_NAME} is used : {used_link} (not "
                                           f"{unused_link})", indent=1)
    return found_paths[0]


def start_evaluation_tracking(trainer: BaseTrainer, experiment_folder: Path,
                              iterations: List[int]) -> ExperimentTracker:
    """
    Opens the MLflow run of a standalone evaluation when 'tracking.activated' is True: a new run named
    '<evaluated run name>-evaluation' in the backend and experiment of the evaluated training, with the parameters of
    the trainer, tagged with the MLflow run of the evaluated training ('evaluation_of', absent for a run trained
    without tracking), the evaluated experiment folder, the evaluated test split and the evaluated iterations. The
    backend is the one of the 'tracking_root' of the trainer, the experiments root folder of the configuration of the
    run, which the launch panel names, so that the evaluation of a run moved or archived since its training stays
    next to the training run it points to. The training run is only read, so that an evaluation on another split
    never mixes into its curves.

    Args:
        trainer (BaseTrainer): The trainer of the evaluation, whose tracker receives the MLflow run.
        experiment_folder (Path): The evaluated training run, holding its 'mlflow_run.json' when it was tracked.
        iterations (List[int]): The iterations of the evaluated checkpoints.

    Returns:
        ExperimentTracker: The tracker of the evaluation run, a DisabledTracker when the tracking is off.
    """
    evaluation_tracker = create_experiment_tracker(
        tracking_configuration=trainer.tracking_configuration,
        tracking_root=trainer.experiment_configuration["tracking_root"],
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
    evaluation files of the task in the run folder of the trainer, the averaged weights next to the trained ones, the
    row of the iteration in 'evaluation_summary.csv' next to the iteration folders, evaluation_root of the trainer, with
    every evaluation scalar, those of every class only with the per-class report and the averaged weights in 'ema_'
    columns, and 'evaluation_curves.png' drawn again next to it), then the best iteration according to the headline
    metric of the task is printed, for the trained and the averaged weights. A checkpoint listed in the registry but
    deleted from the disk is skipped, and an error is raised when none of them is on the disk. When the tracking is on,
    the evaluations, the best headline metric, the summary and the curves go to an MLflow run of their own
    (start_evaluation_tracking), closed whatever ends the evaluation, as failed when nothing was evaluated.

    Args:
        trainer (BaseTrainer): The trainer of the run, built from its configuration, usually writing into the
            '<run>/Checkpoint_Evaluation/' folder of a standalone evaluation
            (scripts/evaluation/evaluate_checkpoint.py).
        experiment_folder (Path): The trained run holding the 'Weights' folder of the checkpoints.
        iterations (List[int]): The iterations of the checkpoints to evaluate, in order.

    Returns:
        Path: The evaluation summary of the folder of the evaluations.

    Raises:
        FileNotFoundError: If none of the checkpoints of the iterations is on the disk.
    """
    # The evaluations reported by the trainer, its summary and its curves are tracked in the MLflow run of the
    # evaluation
    trainer.experiment_tracker = start_evaluation_tracking(trainer=trainer, experiment_folder=experiment_folder,
                                                           iterations=iterations)
    run_status = FAILED_STATUS
    try:
        summary_path = summarise_checkpoint_evaluations(trainer=trainer, experiment_folder=experiment_folder,
                                                        iterations=iterations)
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
    Evaluates the checkpoints one after the other through the trainer, which writes the row of every evaluation into
    the evaluation summary of its folder and draws the evaluation curves again (BaseTrainer.update_evaluation_curves),
    then prints the summary, the curves and the best iteration of the evaluated checkpoints, or every tied iteration
    when several share the best value, for evaluate_checkpoints (find_best_iterations), the value written as a
    percentage when the headline metric is one of the percent metrics of the task ("Accuracy 94.03 %").

    Args:
        trainer (BaseTrainer): The trainer of the run.
        experiment_folder (Path): The trained run holding the 'Weights' folder of the checkpoints.
        iterations (List[int]): The iterations of the checkpoints to evaluate, in order.

    Returns:
        Path: The evaluation summary, which also keeps the rows of the earlier evaluations written into the same
            folder.

    Raises:
        FileNotFoundError: If none of the checkpoints of the iterations is on the disk, so that an evaluation of
            nothing never ends as a finished one.
    """
    headline_metric, headline_in_percent = None, False
    headline_values: Dict[str, Dict[int, float]] = {}
    for iteration in iterations:
        checkpoint_path = get_checkpoint_file(weights_directory=experiment_folder / "Weights", iteration=iteration)
        if not checkpoint_path.exists():
            print_color(color="yellow", output=f"Checkpoint Of Iteration {iteration} Not Found, Skipped : "
                                               f"{format_path_link(path=checkpoint_path)}")
            continue

        # Weights of the checkpoint in the model of the trainer, with their moving average, then the evaluation of the
        # test split, which adds its row to the summary
        checkpoint = torch.load(f=checkpoint_path, map_location=trainer.device, weights_only=True)
        trainer.model.load_state_dict(state_dict=checkpoint["model_state"])
        trainer.load_exponential_moving_average(checkpoint=checkpoint, iteration=iteration)
        print_color(color="blue", output=f"Evaluating : {format_path_link(path=checkpoint_path)}", add_separators=True)
        results, averaged_weights_results = trainer.run_test_evaluation(iteration=iteration)

        # Headline metric of every kind of weights at every evaluated iteration, in the evaluation order, the averaged
        # weights for the runs keeping a moving average only
        headline_metric = results["headline_metric"]
        headline_in_percent = headline_metric in results["percent_metrics"]
        evaluated_weights = {"": results}
        if averaged_weights_results is not None:
            evaluated_weights[AVERAGED_WEIGHTS_LABEL] = averaged_weights_results
        for weights_label, weights_results in evaluated_weights.items():
            headline_values.setdefault(weights_label, {})[iteration] = weights_results["scalars"][headline_metric]

    # Nothing evaluated when every listed checkpoint is missing from the disk, e.g. a run stopped before its first save
    if not headline_values:
        raise FileNotFoundError(f"No Checkpoint Evaluated: none of the iterations {iterations} has its checkpoint in : "
                                f"{format_path_link(path=experiment_folder / 'Weights')}")

    # The summary and the curves of the folder, then the best iterations of the evaluated checkpoints
    summary_path = trainer.evaluation_root / EVALUATION_SUMMARY_FILE_NAME
    figure_path = Path(trainer.experiment_folder) / EVALUATION_CURVES_FIGURE_NAME
    print_color(color="green", output=f"Evaluation Summary Of {len(headline_values[''])} Checkpoints : "
                                      f"{format_path_link(path=summary_path)}", add_separators=True)
    print_color(color="green", output=f"Evaluation Curves : {format_path_link(path=figure_path)}", indent=1)
    for weights_label, values in headline_values.items():
        # One best iteration, or the tied iterations in their evaluation order, none when every value is not a number
        weights_description = f" ({weights_label})" if weights_label else ""
        best_evaluation = find_best_iterations(values=values)
        if best_evaluation is None:
            print_color(color="yellow", output=f"Best Checkpoint{weights_description} : none, {headline_metric} is "
                                               f"not a number at every evaluated iteration", indent=1)
            continue
        best_iterations, best_value = best_evaluation
        iterations_description = f"iteration {best_iterations[0]}"
        if len(best_iterations) > 1:
            tied_iterations = ", ".join(str(best_iteration) for best_iteration in best_iterations)
            iterations_description = f"iterations {tied_iterations} tie"
        # Best value written like in the report, as a percentage when the headline metric is a rate of the task
        best_text = format_report_value(value=best_value, in_percent=headline_in_percent)
        print_color(color="green", output=f"Best Checkpoint{weights_description} : {iterations_description} "
                                          f"({headline_metric} {best_text})", indent=1)
    return summary_path
