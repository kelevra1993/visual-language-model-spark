"""
When run, gathers the test evaluations of the launches of a hyperparameter experiment into one CSV file, to compare
the settings of every stage: each run_<n> launch folder of the experiment folder (run_<n><letters> for a repeat launch
of the same settings, such as run_23b) holds the run folder of one training (vision_training/main.py, one
configuration file per launch), whose Tensorboard/Test curves give the test metric at every evaluated checkpoint and
whose training_configuration.yaml gives the swept settings. The launches of the same settings are followed by their
mean row, whose mean column decides the setting kept for the next stage, next to the mean of the last evaluations,
which compares where the settings end whatever their start.
"""

import csv
import re
import statistics
import yaml

from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Any, Dict, List, Tuple

from vision_training.evaluation.report_formatting import format_file_value
from vision_training.utilities.os_utilities import format_path_link, print_color

# Name of the launch folders of an experiment, run_<n> or run_<n><letters> for a repeat launch (run_23b), sorted by
# their number then their letters
LAUNCH_FOLDER_PATTERN = re.compile(pattern=r"run_(\d+)([a-z]*)")

# Settings changed by the stages of the classification experiment, written as columns: their column name and their
# keys in training_configuration.yaml, from the section down to the setting
SWEPT_SETTINGS = {"learning_rate": ("ExperimentConfiguration", "learning_rate"),
                  "data_augmentation": ("Data", "data_augmentation", "activated"),
                  "global_pooling": ("ClassificationHead", "global_pooling"),
                  "residual_nets": ("Backbone", "modules", "residual_nets"),
                  "residual_nets_v2": ("Backbone", "modules", "residual_nets_v2"),
                  "squeeze_and_excite": ("Backbone", "modules", "squeeze_and_excite"),
                  "fully_connected_normalization": ("ClassificationHead", "fully_connected_normalization"),
                  "backbone_batch_normalization_momentum": ("Backbone", "normalization",
                                                            "batch_normalization_momentum"),
                  "head_batch_normalization_momentum": ("ClassificationHead", "batch_normalization_momentum"),
                  "feature_map_normalization": ("Backbone", "normalization", "feature_map_normalization"),
                  "mixed_precision": ("Optimization", "mixed_precision"),
                  "weight_initialization": ("Optimization", "weight_initialization"),
                  "exponential_moving_average": ("Optimization", "exponential_moving_average", "activated"),
                  "class_weighting": ("ClassificationHead", "class_weighting"),
                  "dropout_rate": ("ClassificationHead", "dropout_rate")}

# Launch name of the row averaging the launches of the same settings
MEAN_ROW_NAME = "mean"


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
        # A launch trained before the setting existed gets it from the migration of its run folder
        if key not in value:
            raise KeyError(f"No '{'.'.join(keys)}' setting in {format_path_link(path=configuration_path)}, written "
                           f"before the setting existed: migrate its run folder with "
                           f"scripts/configuration_processing/migrate_configuration.py")
        value = value[key]
    return value


def read_test_metric(test_folder: Path, metric: str) -> Dict[int, float]:
    """
    Reads the test metric of a launch at every evaluated checkpoint from its Tensorboard/Test curves, written at full
    precision by the trainer at every weight_saving_iterations checkpoint. A launch evaluated again at the same
    iteration after a restart keeps its last evaluation.

    Args:
        test_folder (Path): The Tensorboard/Test folder of the run folder.
        metric (str): The name of the test curve, e.g. "Accuracy".

    Returns:
        Dict[int, float]: The metric at every evaluated iteration, empty before the first evaluation.

    Raises:
        ValueError: If the launch has test curves but none of that name.
    """
    # No test curve before the first evaluation of the launch
    if not test_folder.is_dir():
        return {}
    events = EventAccumulator(path=str(test_folder), size_guidance={"scalars": 0})
    events.Reload()
    curve_names = events.Tags()["scalars"]
    if not curve_names:
        return {}
    if metric not in curve_names:
        raise ValueError(f"No {metric} test curve in {format_path_link(path=test_folder)}, available curves : "
                         f"{', '.join(curve_names)}")

    # Later events of the same iteration overwrite the earlier ones
    return {event.step: event.value for event in events.Scalars(tag=metric)}


def collect_launches(experiment_folder: Path, metric: str) -> List[Dict[str, Any]]:
    """
    Gathers every launch of a hyperparameter experiment in launch order: the swept settings of its training
    configuration, its test metric at every evaluated checkpoint and whether it was evaluated at its last iteration,
    so that the launches still training are written with the evaluations they already have.

    Args:
        experiment_folder (Path): The folder of the experiment, holding a run_<n> or run_<n><letters> folder per
            launch.
        metric (str): The name of the test curve, e.g. "Accuracy".

    Returns:
        List[Dict[str, Any]]: For every run folder, its 'launch' (run_<n> or run_<n><letters>), 'settings',
            'evaluations' (the metric at every evaluated iteration), 'finished' and 'run_folder' (the name of the
            folder named after the settings).

    Raises:
        FileNotFoundError: If the experiment folder does not exist, e.g. before the first launch.
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

            # Test metric of every evaluated checkpoint, the launch being finished once its last iteration is evaluated
            evaluations = read_test_metric(test_folder=run_folder / "Tensorboard" / "Test", metric=metric)
            settings = {name: read_setting(configuration=configuration, keys=keys,
                                           configuration_path=configuration_path)
                        for name, keys in SWEPT_SETTINGS.items()}
            launches.append({"launch": launch_folder.name, "settings": settings, "evaluations": evaluations,
                             "finished": configuration["ExperimentConfiguration"]["number_iterations"] in evaluations,
                             "run_folder": run_folder.name})
    return launches


def build_result_rows(launches: List[Dict[str, Any]], metric_column: str,
                      number_last_evaluations: int) -> Tuple[List[str], List[Dict[str, Any]]]:
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
    columns = ["launch", *iteration_columns, mean_column, last_mean_column, "finished", *SWEPT_SETTINGS, "run_folder"]

    # Launches of the same settings, in the order of their first launch
    setting_groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for launch in launches:
        setting_groups.setdefault(tuple(launch["settings"].values()), []).append(launch)

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


def write_results(columns: List[str], rows: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Writes the results file of the experiment, readable by any spreadsheet or CSV reader: the settings as they are in
    the configuration, the metric values with 6 decimals and the missing ones as empty cells, as in the per-class
    metrics files of the evaluations (format_file_value).

    Args:
        columns (List[str]): The columns of the file (build_result_rows).
        rows (List[Dict[str, Any]]): The rows of the file (build_result_rows).
        output_path (Path): The CSV file, overwritten.
    """
    with open(file=output_path, mode="w", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(f=results_file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] if column in SWEPT_SETTINGS else format_file_value(value=row[column])
                             for column in columns})


if __name__ == "__main__":
    # Folder of the hyperparameter experiment, holding a run_<n> or run_<n><letters> folder per launch (the
    # experiment_folder of every launch configuration file)
    experiment_folder = Path(__file__).absolute().parents[2] / "hyperparameter_experiments"
    # Test curve compared between the launches, one of the scalars of Tensorboard/Test: "Accuracy", "Balanced
    # Accuracy", "Macro F1" or "Top-5 Accuracy" for the classification task
    metric = "Accuracy"
    # Number of last iterations evaluated in the experiment averaged by the <metric>_last_<n>_mean column, 4 giving the
    # mean of iterations 1250 to 2000 when the launches are evaluated every 250 iterations up to 2000
    number_last_evaluations = 4
    # Results file, written again from all the launches at every run, so that it can be refreshed while launches train
    output_path = experiment_folder / "results.csv"

    # Read every launch, group the launches of the same settings and write the file
    launches = collect_launches(experiment_folder=experiment_folder, metric=metric)
    columns, rows = build_result_rows(launches=launches, metric_column=metric.lower().replace(" ", "_"),
                                      number_last_evaluations=number_last_evaluations)
    write_results(columns=columns, rows=rows, output_path=output_path)
    number_finished = sum(launch["finished"] for launch in launches)
    print_color(color="green", output=f"Test {metric} of {len(launches)} launches ({number_finished} finished) written "
                                      f"to {format_path_link(path=output_path)}")
