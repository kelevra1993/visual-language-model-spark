"""
When run, gathers the test evaluations of the launches of a hyperparameter experiment (classification, sequence
labelling or object detection) into one CSV file, to compare the settings of every stage: each run_<n> launch folder of
the experiment
folder (run_<n><letters> for a repeat launch of the same settings, such as run_23b) holds the run folder of one training
(vision_training/main.py, one configuration file per launch), whose Tensorboard/Test curves give the test metric at
every evaluated checkpoint and whose training_configuration.yaml gives the swept settings; a launch keeping a moving
average of its weights also gets the rows of its averaged weights from its Tensorboard/Test_EMA curves. The launches of
the same settings are followed by their mean row, whose mean column decides the setting kept for the next stage, next to
the mean of the last evaluations, which compares where the settings end whatever their start. Run as a script with the
task given as an argument and the experiment folder, the number of last evaluations and the results file as optional
ones (parse_collection_arguments), its help printed when none is given.
"""

import argparse
import csv
import re
import statistics
import sys
import yaml

from pathlib import Path
from typing import Any, Dict, List, Tuple

from vision_training.evaluation.evaluation_curves import read_test_curves
from vision_training.evaluation.report_formatting import format_file_value
from vision_training.trainers.base_trainer import AVERAGED_WEIGHTS_TENSORBOARD_FOLDER
from vision_training.utilities.os_utilities import format_path_link, print_color

# Name of the launch folders of an experiment, run_<n> or run_<n><letters> for a repeat launch (run_23b), sorted by
# their number then their letters
LAUNCH_FOLDER_PATTERN = re.compile(pattern=r"run_(\d+)([a-z]*)")

# Settings changed by the stages of the experiment of every task, written as columns: their column name and their keys
# in training_configuration.yaml, from the section down to the setting, the exponential_moving_average setting telling
# which launches have averaged weights. In the classification experiment, the training data folder and the source runs
# of the transferred backbone and classification head (empty without transfer) separate the launches adding classes to
# a trained run (stage 14) from the launches whose settings they copy. In the sequence labelling experiment, the
# bidirectional and causal flags give the direction of the temporal module, and the source run of the transferred
# backbone with its freeze and learning rate multiplier separate the launches of the pretraining stage (fine-tuned,
# frozen or fine-tuned slower) from the launches trained from scratch whose other settings they copy
CLASSIFICATION_SWEPT_SETTINGS = {
    "learning_rate": ("ExperimentConfiguration", "learning_rate"),
    "train_data_folder": ("ExperimentConfiguration", "train_data_folder"),
    "pretrained_backbone": ("Backbone", "pretrained", "experiment_folder"),
    "pretrained_classification_head": ("ClassificationHead", "pretrained", "experiment_folder"),
    "data_augmentation": ("Data", "data_augmentation", "activated"),
    "global_pooling": ("ClassificationHead", "global_pooling"),
    "residual_nets": ("Backbone", "modules", "residual_nets"),
    "residual_nets_v2": ("Backbone", "modules", "residual_nets_v2"),
    "squeeze_and_excite": ("Backbone", "modules", "squeeze_and_excite"),
    "fully_connected_normalization": ("ClassificationHead", "fully_connected_normalization"),
    "feature_map_normalization": ("Backbone", "normalization", "feature_map_normalization"),
    "exponential_moving_average": ("Optimization", "exponential_moving_average", "activated"),
    "class_weighting": ("ClassificationHead", "class_weighting"),
    "dropout_rate": ("ClassificationHead", "dropout_rate")}
SEQUENCE_LABELLING_SWEPT_SETTINGS = {
    "learning_rate": ("ExperimentConfiguration", "learning_rate"), "temporal_type": ("TemporalModel", "type"),
    "bidirectional": ("TemporalModel", "bidirectional"), "causal": ("TemporalModel", "causal"),
    "number_layers": ("TemporalModel", "number_layers"), "global_pooling": ("ClassificationHead", "global_pooling"),
    "feature_map_normalization": ("Backbone", "normalization", "feature_map_normalization"),
    "class_weighting": ("ClassificationHead", "class_weighting"),
    "pretrained_backbone": ("Backbone", "pretrained", "experiment_folder"),
    "freeze_backbone": ("Backbone", "pretrained", "freeze"),
    "backbone_learning_rate_multiplier": ("Backbone", "pretrained", "learning_rate_multiplier"),
    "exponential_moving_average": ("Optimization", "exponential_moving_average", "activated")}
OBJECT_DETECTION_SWEPT_SETTINGS = {
    "learning_rate": ("ExperimentConfiguration", "learning_rate"),
    "exponential_moving_average": ("Optimization", "exponential_moving_average", "activated")}
TASK_SWEPT_SETTINGS = {"classification": CLASSIFICATION_SWEPT_SETTINGS,
                       "sequence_labelling": SEQUENCE_LABELLING_SWEPT_SETTINGS,
                       "object_detection": OBJECT_DETECTION_SWEPT_SETTINGS}

# Folder of the hyperparameter experiment of every task, holding a run_<n> or run_<n><letters> folder per launch (the
# experiment_folder of every launch configuration file): the git-ignored hyperparameter_experiments folder of the
# repository for the classification, its sequence_labelling and object_detection subfolders for the others
EXPERIMENTS_ROOT_FOLDER = Path(__file__).absolute().parents[2] / "hyperparameter_experiments"
TASK_EXPERIMENT_FOLDERS = {"classification": EXPERIMENTS_ROOT_FOLDER,
                           "sequence_labelling": EXPERIMENTS_ROOT_FOLDER / "sequence_labelling",
                           "object_detection": EXPERIMENTS_ROOT_FOLDER / "object_detection"}

# Test curve compared between the launches of every task, one of the scalars of Tensorboard/Test: the accuracy for the
# classification task, the Macro F1 for the sequence labelling task, which weighs the frames of every change as much as
# the far more frequent frames without change, and the mean average precision over the intersection over union
# thresholds 0.5 to 0.95 (the area under the precision recall curves) for the object detection task. The metric columns
# of the results file are named after the metric of every task
TASK_METRICS = {"classification": "Accuracy", "sequence_labelling": "Macro F1",
                "object_detection": "mAP@[0.5:0.95]/area"}
TASK_METRIC_COLUMNS = {"classification": "accuracy", "sequence_labelling": "macro_f1", "object_detection": "map_50_95"}

# Number of last iterations evaluated in the experiment averaged by the <metric>_last_<n>_mean column unless the
# command line gives another one, 4 giving the mean of iterations 1250 to 2000 when the launches are evaluated every
# 250 iterations up to 2000 (the classification and the sequence labelling experiments)
DEFAULT_NUMBER_LAST_EVALUATIONS = 4

# Results file written in the experiment folder unless the command line gives another file
RESULTS_FILE_NAME = "results.csv"

# Launch name of the row averaging the launches of the same settings
MEAN_ROW_NAME = "mean"

# Column telling which weights a row evaluates, and the test curve folder of each kind of weights: the trained weights
# of every launch, the averaged weights of the launches keeping a moving average
EVALUATED_WEIGHTS_COLUMN = "evaluated_weights"
TEST_FOLDERS = {"trained": "Test", "ema": AVERAGED_WEIGHTS_TENSORBOARD_FOLDER}


def get_launch_order(launch_folder: Path) -> Tuple[int, str]:
    """
    Gives the sorting key of a run_<n> or run_<n><letters> launch folder, so that the launches are listed in launch
    order in the results file (run_2 before run_10), every repeat launch right after its original one (run_23,
    run_23b, run_23c, then run_24).

    Args:
        launch_folder (Path): The launch folder, named run_<n> or run_<n><letters>.

    Returns:
        Tuple[int, str]: The launch number n and the letters of a repeat launch, empty for the original launch.
    """
    launch_match = LAUNCH_FOLDER_PATTERN.fullmatch(string=launch_folder.name)
    return int(launch_match.group(1)), launch_match.group(2)


def read_setting(configuration: Dict[str, Any], keys: Tuple[str, ...], configuration_path: Path) -> Any:
    """
    Reads a swept setting of a launch from its training configuration, walking down its nested keys, so that a key
    missing from the configuration stops the collection instead of writing an empty setting.

    Args:
        configuration (Dict[str, Any]): The training configuration of the launch (training_configuration.yaml).
        keys (Tuple[str, ...]): The keys of the setting, from its section down to its name.
        configuration_path (Path): The training_configuration.yaml file of the launch, named in the error.

    Returns:
        Any: The value of the setting.

    Raises:
        KeyError: If the setting is missing, e.g. from a launch trained before the setting existed.
    """
    value = configuration
    for key in keys:
        # A launch trained before the setting existed needs it written in its configuration with the value it used
        if key not in value:
            raise KeyError(f"No '{'.'.join(keys)}' setting in {format_path_link(path=configuration_path)}, written "
                           f"before the setting existed: add it with the value the launch was trained with")
        value = value[key]
    return value


def collect_launches(experiment_folder: Path, metric: str,
                     swept_settings: Dict[str, Tuple[str, ...]]) -> List[Dict[str, Any]]:
    """
    Gathers every launch of a hyperparameter experiment in launch order: the swept settings of its training
    configuration, its test metric at every evaluated checkpoint and whether it was evaluated at its last iteration,
    so that the launches still training are written with the evaluations they already have.

    Args:
        experiment_folder (Path): The folder of the experiment, holding a run_<n> or run_<n><letters> folder per
            launch.
        metric (str): The name of the test curve, e.g. "Accuracy".
        swept_settings (Dict[str, Tuple[str, ...]]): The column name and configuration keys of every setting changed
            by the stages of the experiment (TASK_SWEPT_SETTINGS), with exponential_moving_average.

    Returns:
        List[Dict[str, Any]]: For every run folder and evaluated weights, its 'launch' (run_<n> or run_<n><letters>),
            'settings' (with the evaluated weights, "trained" or "ema"), 'evaluations' (the metric at every evaluated
            iteration), 'finished' and 'run_folder' (the name of the folder named after the settings).

    Raises:
        FileNotFoundError: If the experiment folder does not exist, e.g. before the first launch.
        ValueError: If a launch has test curves but none of the metric.
    """
    # The experiment folder is created by the first launch writing into it
    if not experiment_folder.is_dir():
        raise FileNotFoundError(f"No experiment folder {format_path_link(path=experiment_folder)}, created by the "
                                f"first launch whose experiment_folder points inside it")
    launch_folders = sorted([folder for folder in experiment_folder.iterdir()
                             if folder.is_dir() and LAUNCH_FOLDER_PATTERN.fullmatch(string=folder.name) is not None],
                            key=get_launch_order)

    launches = []
    for launch_folder in launch_folders:
        # Run folder of the launch, named after its settings and holding the copy of its configuration file
        for configuration_path in sorted(launch_folder.glob(pattern="*/training_configuration.yaml")):
            configuration = yaml.safe_load(stream=configuration_path.read_text(encoding="utf-8"))
            run_folder = configuration_path.parent

            settings = {name: read_setting(configuration=configuration, keys=keys,
                                           configuration_path=configuration_path)
                        for name, keys in swept_settings.items()}

            # Test metric of every evaluated checkpoint for every kind of weights, the averaged weights only for a
            # launch keeping a moving average, the launch being finished once its last iteration is evaluated
            for evaluated_weights, test_folder_name in TEST_FOLDERS.items():
                if evaluated_weights == "ema" and not settings["exponential_moving_average"]:
                    continue
                test_folder = run_folder / "Tensorboard" / test_folder_name
                test_curves = read_test_curves(test_folder=test_folder)
                if test_curves and metric not in test_curves:
                    raise ValueError(f"No {metric} test curve in {format_path_link(path=test_folder)}, available "
                                     f"curves : {', '.join(test_curves)}")
                evaluations = test_curves[metric] if test_curves else {}
                finished = configuration["ExperimentConfiguration"]["number_iterations"] in evaluations
                launches.append({"launch": launch_folder.name, "evaluations": evaluations, "finished": finished,
                                 "settings": {**settings, EVALUATED_WEIGHTS_COLUMN: evaluated_weights},
                                 "run_folder": run_folder.name})
    return launches


def make_hashable(value: Any) -> Any:
    """
    Turns a swept setting read from a training configuration into a value usable in the key grouping the launches of
    the same settings (build_result_rows): lists become tuples and mappings sorted tuples of their items, at every
    depth, so that the settings given as lists, such as the [layers, channels] of the mask heads or the backbone blocks
    of their skips in the instance segmentation experiment, are compared by their values like the scalar settings,
    which are kept as they are.

    Args:
        value (Any): The setting, a scalar, a list or a mapping.

    Returns:
        Any: The setting in a hashable form, equal for two equal settings.
    """
    if isinstance(value, list):
        return tuple(make_hashable(value=item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, make_hashable(value=item)) for key, item in value.items()))
    return value


def build_result_rows(launches: List[Dict[str, Any]], metric_column: str, number_last_evaluations: int,
                      swept_settings: Dict[str, Tuple[str, ...]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Lays out the launches as the rows of the results file, as in the tables of the experiment plan: the launches of
    the same settings together, in the order of their first launch, followed by their mean row, the mean column of the
    mean rows deciding the setting kept for the next stage. Every launch has a column per iteration evaluated in the
    experiment, the mean of its own evaluations and the mean of its values at the last iterations evaluated in the
    experiment, which leaves out how fast the launch starts, placed with the finished column right after the launch
    column so that the results are read before the settings.

    Args:
        launches (List[Dict[str, Any]]): The launches (collect_launches).
        metric_column (str): The prefix of the metric columns, e.g. "accuracy".
        number_last_evaluations (int): The number of last iterations evaluated in the experiment averaged by the
            <metric_column>_last_<number_last_evaluations>_mean column, e.g. 4 for iterations 1250 to 2000.
        swept_settings (Dict[str, Tuple[str, ...]]): The settings changed by the stages of the experiment
            (TASK_SWEPT_SETTINGS), one column each.

    Returns:
        Tuple[List[str], List[Dict[str, Any]]]: The columns of the file and its rows, None for the missing values.

    Raises:
        ValueError: If number_last_evaluations is below 1.
    """
    if number_last_evaluations < 1:
        raise ValueError(f"number_last_evaluations must be at least 1, got {number_last_evaluations}")

    # One column per iteration evaluated by any launch, the launches still training keeping empty cells. The results
    # come right after the launch, before the settings
    evaluated_iterations = sorted({iteration for launch in launches for iteration in launch["evaluations"]})
    iteration_columns = [f"{metric_column}_{iteration}" for iteration in evaluated_iterations]
    mean_column = f"{metric_column}_mean"
    last_iterations = evaluated_iterations[-number_last_evaluations:]
    last_mean_column = f"{metric_column}_last_{number_last_evaluations}_mean"
    columns = ["launch", EVALUATED_WEIGHTS_COLUMN, *iteration_columns, mean_column, last_mean_column, "finished",
               *swept_settings, "run_folder"]

    # Launches of the same settings, in the order of their first launch, the list settings compared by their values
    setting_groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for launch in launches:
        settings_key = tuple(make_hashable(value=value) for value in launch["settings"].values())
        setting_groups.setdefault(settings_key, []).append(launch)

    rows = []
    for group_launches in setting_groups.values():
        # One row per launch, with the mean of its own evaluations and the mean of the ones it has at the last
        # iterations, a launch still training having only some of them or none
        group_rows = []
        for launch in group_launches:
            row = {"launch": launch["launch"], **launch["settings"], "finished": launch["finished"],
                   "run_folder": launch["run_folder"]}
            for iteration, column in zip(evaluated_iterations, iteration_columns):
                row[column] = launch["evaluations"].get(iteration)
            row[mean_column] = statistics.fmean(data=launch["evaluations"].values()) if launch["evaluations"] else None
            last_values = [launch["evaluations"][iteration] for iteration in last_iterations
                           if iteration in launch["evaluations"]]
            row[last_mean_column] = statistics.fmean(data=last_values) if last_values else None
            group_rows.append(row)

        # Mean row of the settings, averaging every metric column over the launches that have a value
        mean_row = {"launch": MEAN_ROW_NAME, **group_launches[0]["settings"],
                    "finished": all(launch["finished"] for launch in group_launches), "run_folder": ""}
        for column in [*iteration_columns, mean_column, last_mean_column]:
            values = [row[column] for row in group_rows if row[column] is not None]
            mean_row[column] = statistics.fmean(data=values) if values else None
        rows += group_rows + [mean_row]
    return columns, rows


def write_results(columns: List[str], rows: List[Dict[str, Any]], output_path: Path,
                  swept_settings: Dict[str, Tuple[str, ...]]) -> None:
    """
    Writes the results file of the experiment, readable by any spreadsheet or CSV reader: the settings as they are in
    the configuration, the metric values with 6 decimals and the missing ones as empty cells, as in the per-class
    metrics files of the evaluations (format_file_value).

    Args:
        columns (List[str]): The columns of the file (build_result_rows).
        rows (List[Dict[str, Any]]): The rows of the file (build_result_rows).
        output_path (Path): The CSV file, overwritten.
        swept_settings (Dict[str, Tuple[str, ...]]): The settings changed by the stages of the experiment
            (TASK_SWEPT_SETTINGS), written as they are in the configuration.
    """
    with open(file=output_path, mode="w", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(f=results_file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            setting_columns = [*swept_settings, EVALUATED_WEIGHTS_COLUMN]
            writer.writerow({column: row[column] if column in setting_columns else format_file_value(value=row[column])
                             for column in columns})


def parse_collection_arguments(arguments: List[str]) -> argparse.Namespace:
    """
    Reads the settings of a collection from the command line of the script, so that the results file of a stage is
    written again without editing the file: the task is required and chooses the swept settings (TASK_SWEPT_SETTINGS)
    and the compared metric (TASK_METRICS), the experiment folder defaults to the folder of the task
    (TASK_EXPERIMENT_FOLDERS), the number of last evaluations to DEFAULT_NUMBER_LAST_EVALUATIONS and the results file to
    RESULTS_FILE_NAME in the experiment folder. Launched without any argument, the script prints the help of every
    argument and stops. A '~' left by the shell at the start of a path ('--experiment_folder=~/...') is expanded to the
    home folder.

    Args:
        arguments (List[str]): The command line arguments, without the script name.

    Returns:
        argparse.Namespace: The task, experiment_folder, number_last_evaluations and output_path, the default paths
            filled in.

    Raises:
        SystemExit: After printing the help when no argument is given, or the usage and the error when the task is
            missing or a value is not one the collection accepts.
    """
    # The description and the example keep their line breaks, argparse wrapping the help of every argument
    parser = argparse.ArgumentParser(
        prog="uv run python -m scripts.evaluation.collect_experiment_results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Gathers the test metric of every launch of a hyperparameter experiment, one run_<n> folder per\n"
                    "launch, into one CSV file with a mean row per setting, to choose the setting of every stage.",
        epilog="Example, the sequence labelling experiment written to the results.csv of its folder:\n"
               "  uv run python -m scripts.evaluation.collect_experiment_results --task sequence_labelling")
    # The experiment, its folder of launches and the metric compared
    task_metrics = ", ".join(f"{task}: {metric}" for task, metric in TASK_METRICS.items())
    parser.add_argument("--task", choices=list(TASK_SWEPT_SETTINGS), required=True,
                        help=f"Experiment, choosing the swept settings and the compared test curve ({task_metrics}) "
                             f"(required).")
    parser.add_argument("--experiment_folder", type=Path, default=None,
                        help="Folder of the run_<n> launch folders (default: hyperparameter_experiments of the "
                             "repository for classification, its <task> subfolder for the other tasks).")
    # The second mean column and the written file
    parser.add_argument("--number_last_evaluations", type=int, default=DEFAULT_NUMBER_LAST_EVALUATIONS,
                        help="Last evaluations averaged by the <metric>_last_<n>_mean column (default: %(default)s).")
    parser.add_argument("--output_path", type=Path, default=None,
                        help=f"CSV file, written again from every launch (default: {RESULTS_FILE_NAME} in the "
                             f"experiment folder).")

    # Without any argument, the script describes its arguments instead of failing on the missing task
    if not arguments:
        parser.print_help()
        parser.exit(status=0)
    parsed_arguments = parser.parse_args(args=arguments)

    # Folder of the task and results file inside it unless given, a '~' the shell leaves after '=' pointing to the home
    # folder
    if parsed_arguments.experiment_folder is None:
        parsed_arguments.experiment_folder = TASK_EXPERIMENT_FOLDERS[parsed_arguments.task]
    parsed_arguments.experiment_folder = parsed_arguments.experiment_folder.expanduser()
    if parsed_arguments.output_path is None:
        parsed_arguments.output_path = parsed_arguments.experiment_folder / RESULTS_FILE_NAME
    parsed_arguments.output_path = parsed_arguments.output_path.expanduser()
    return parsed_arguments


if __name__ == "__main__":
    # Task, experiment folder, number of last evaluations and results file read from the command line, the help
    # printed when none is given
    collection_arguments = parse_collection_arguments(arguments=sys.argv[1:])
    swept_settings = TASK_SWEPT_SETTINGS[collection_arguments.task]
    metric = TASK_METRICS[collection_arguments.task]

    # Read every launch, group the launches of the same settings and write the file, written again from all the
    # launches at every run so that it can be refreshed while launches train
    launches = collect_launches(experiment_folder=collection_arguments.experiment_folder, metric=metric,
                                swept_settings=swept_settings)
    columns, rows = build_result_rows(launches=launches, metric_column=TASK_METRIC_COLUMNS[collection_arguments.task],
                                      number_last_evaluations=collection_arguments.number_last_evaluations,
                                      swept_settings=swept_settings)
    write_results(columns=columns, rows=rows, output_path=collection_arguments.output_path,
                  swept_settings=swept_settings)

    # Launches counted once, whatever the number of weights they evaluate
    trained_launches = [launch for launch in launches if launch["settings"][EVALUATED_WEIGHTS_COLUMN] == "trained"]
    number_finished = sum(launch["finished"] for launch in trained_launches)
    print_color(color="green", output=f"Test {metric} of {len(trained_launches)} launches ({number_finished} finished) "
                                      f"written to {format_path_link(path=collection_arguments.output_path)}")
