"""
When run, compares the best test evaluation of several trained runs (vision_training/main.py) in Markdown tables, e.g.
the same dataset trained at two image sizes or two datasets trained with the same settings: for every run folder of a
list of experiment folders, the evaluation with the best value of a chosen test curve over the trained weights
(Tensorboard/Test) and the averaged weights (Tensorboard/Test_EMA), with the columns set in their order in the main
block: the curves at that evaluation (global metrics, and per-class metrics expanded to one column per class), the
settings of its training_configuration.yaml (the image size, the learning rate), and the totals and per-class counts of
its splits read from its data.json (recomputed from the labels.json of the splits for the runs written before the
classes were counted, after checking that the labels did not change since the run). The curves of the classification
evaluations are rates written as the percentages of the reports ("94.03 %"); the curves of the tasks not yet moved to
them keep their decimals.
"""

import hashlib
import re
import yaml

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.evaluation.collect_experiment_results import TEST_FOLDERS, read_setting, read_test_curves
from vision_training.data.data_record import (CLASS_NAMES_RECORD_KEY, CLASSES_RECORD_KEY, DATA_RECORD_FILE_NAME,
                                              build_class_records)
from vision_training.data.label_mapping import build_label_dictionary
from vision_training.data.sequence.sequence_dataset import find_unselected_class_index
from vision_training.evaluation.report_formatting import format_percentage
from vision_training.utilities.os_utilities import format_path_link, print_color, read_json

# Placeholder of a column expanded to one column per class of the run, in class index order
CLASS_PLACEHOLDER = "{class}"

# Prefix of a column giving a setting of training_configuration.yaml, followed by its keys joined by dots
CONFIGURATION_COLUMN_PREFIX = "configuration:"

# Column of a count of data.json: '<split>_<unit>' for the total of the split, '<split>_<unit>/{class}' per class, the
# unit being one of the counts of the classes of data.json (data_record.count_manifest_classes)
COUNT_COLUMN_PATTERN = re.compile(pattern=r"(train|validation|test)_(images|annotations|sequences|frames|pairs)")

# Suffix of the best iteration evaluated on the averaged weights
AVERAGED_WEIGHTS_SUFFIX = " (EMA)"

# Ways of grouping the runs into tables: one table per run, or one table for the runs sharing the same column titles
TABLE_GROUPINGS = ["run", "columns"]


def find_run_folders(experiment_folders: List[Path]) -> List[Path]:
    """
    Lists the run folders to compare: every folder of the list holding a training_configuration.yaml is a run folder,
    and every other folder is an experiment folder whose run folders are its subfolders holding one, sorted by name.

    Args:
        experiment_folders (List[Path]): The experiment folders (the 'experiment_folder' of the configurations) or run
            folders.

    Returns:
        List[Path]: The run folders, in the order of the list.

    Raises:
        FileNotFoundError: If a folder does not exist or holds no run folder.
    """
    run_folders = []
    for experiment_folder in experiment_folders:
        # A run folder given directly, or the run folders of an experiment folder
        if (experiment_folder / "training_configuration.yaml").is_file():
            run_folders.append(experiment_folder)
            continue
        experiment_run_folders = sorted(configuration_path.parent for configuration_path in
                                        experiment_folder.glob(pattern="*/training_configuration.yaml"))
        if not experiment_run_folders:
            raise FileNotFoundError(f"No run folder (holding a training_configuration.yaml) in "
                                    f"{format_path_link(path=experiment_folder)}")
        run_folders.extend(experiment_run_folders)
    return run_folders


def find_best_evaluation(run_folder: Path, best_metric: str, evaluated_weights: List[str],
                         higher_is_better: bool) -> Optional[Dict[str, Any]]:
    """
    Finds the test evaluation of a run with the best value of the chosen curve over its trained and averaged weights,
    the run being then described by the curves of that evaluation. Ties keep the first weights of the list, then the
    earliest iteration.

    Args:
        run_folder (Path): The run folder.
        best_metric (str): The test curve choosing the best evaluation, e.g. "Balanced Accuracy".
        evaluated_weights (List[str]): The weights whose evaluations compete, "trained" and/or "ema" (TEST_FOLDERS).
        higher_is_better (bool): True when a higher value of the curve is better, False for a loss.

    Returns:
        Optional[Dict[str, Any]]: The 'weights', 'iteration' and 'curves' (every curve of those weights at that
            iteration) of the best evaluation, None for a run not evaluated yet.

    Raises:
        ValueError: If evaluated weights have test curves but none of the chosen curve.
    """
    best_evaluation = None
    for weights in evaluated_weights:
        test_folder = run_folder / "Tensorboard" / TEST_FOLDERS[weights]
        test_curves = read_test_curves(test_folder=test_folder)
        # Weights not evaluated yet, or never evaluated (no moving average)
        if not test_curves:
            continue
        if best_metric not in test_curves:
            raise ValueError(f"No {best_metric} test curve in {format_path_link(path=test_folder)}, available curves "
                             f": {', '.join(test_curves)}")

        # A later evaluation or other weights only win with a strictly better value
        for iteration, value in sorted(test_curves[best_metric].items()):
            best_value = None if best_evaluation is None else best_evaluation["value"]
            if best_value is None or (value > best_value if higher_is_better else value < best_value):
                best_evaluation = {"weights": weights, "iteration": iteration, "value": value,
                                   "curves": {curve_name: curve_values[iteration]
                                              for curve_name, curve_values in test_curves.items()
                                              if iteration in curve_values}}
    return best_evaluation


def check_labels_unchanged(data_record: Dict[str, Any], data_record_path: Path) -> None:
    """
    Checks that the labels.json of every split of a run is still the one it was trained on, by its SHA-256 in the
    data.json of the run, before the counts of its classes are recomputed from it for a record written before the
    classes were counted.

    Args:
        data_record (Dict[str, Any]): The data.json of the run.
        data_record_path (Path): The data.json file, named in the error.

    Raises:
        ValueError: If the labels.json of a split changed since the run was trained.
    """
    for split_name, split_record in data_record["splits"].items():
        labels_path = Path(split_record["data_folder"]) / "labels.json"
        with open(file=labels_path, mode="rb") as labels_file:
            labels_sha256 = hashlib.file_digest(labels_file, "sha256").hexdigest()
        if labels_sha256 != split_record["labels_sha256"]:
            raise ValueError(f"The {format_path_link(path=labels_path)} of the {split_name} split changed since "
                             f"{format_path_link(path=data_record_path)} was written: the counts of the classes the "
                             f"run was trained on cannot be recomputed from it")


def read_class_records(data_record: Dict[str, Any], data_record_path: Path,
                       configuration: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Reads the classes of a run with their counts in every split from its data.json, or recomputes them from the
    labels.json of its splits, with the label mapping of its configuration, for a record written before the classes
    were counted (data.json holding 'class_names').

    Args:
        data_record (Dict[str, Any]): The data.json of the run.
        data_record_path (Path): The data.json file, named in the errors.
        configuration (Dict[str, Any]): The training_configuration.yaml of the run.

    Returns:
        List[Dict[str, Any]]: The 'class_index', 'class_name', 'category_ids' and counts of every class
            (data_record.build_class_records).

    Raises:
        ValueError: If the run has no class names in its data.json.
    """
    if CLASSES_RECORD_KEY in data_record and data_record[CLASSES_RECORD_KEY] is not None:
        return data_record[CLASSES_RECORD_KEY]
    if CLASS_NAMES_RECORD_KEY not in data_record or data_record[CLASS_NAMES_RECORD_KEY] is None:
        raise ValueError(f"No classes in {format_path_link(path=data_record_path)}: the run has no class to count")

    # Record written before the classes were counted: recomputed from the labels files the run was trained on
    check_labels_unchanged(data_record=data_record, data_record_path=data_record_path)
    task = configuration["ExperimentConfiguration"]["task"]
    label_mapping_configuration = configuration["Data"]["label_mapping"]
    label_dictionary = build_label_dictionary(label_mapping_configuration=label_mapping_configuration, task=task)
    split_data_folders = {split_name: split_record["data_folder"]
                          for split_name, split_record in data_record["splits"].items()}

    # The frame selection gives the frames of the unmapped categories to its unselected class, as its loader does
    unmapped_class_index = None
    if task == "frame_selection":
        labels_path = Path(split_data_folders["train"]) / "labels.json"
        unmapped_class_index = find_unselected_class_index(
            categories=read_json(path=str(labels_path))["categories"], label_dictionary=label_dictionary,
            unselected_category=configuration["Selection"]["unselected_category"], labels_file=str(labels_path))
    return build_class_records(class_names=data_record[CLASS_NAMES_RECORD_KEY], label_dictionary=label_dictionary,
                               label_mode=label_mapping_configuration["mode"],
                               unmapped_class_index=unmapped_class_index, split_data_folders=split_data_folders)


def expand_columns(columns: List[str], class_names: List[str], curve_names: List[str]) -> List[Tuple[str, str]]:
    """
    Expands every column holding the class placeholder to one column per class, in class index order: every class for
    a count column, and the classes having the curve for a curve column (the background of the detection tasks having
    no per-class curve), so that the same column list serves runs of other classes.

    Args:
        columns (List[str]): The columns, in their order of appearance.
        class_names (List[str]): The class names of the run, in class index order.
        curve_names (List[str]): The test curves of the best evaluation of the run.

    Returns:
        List[Tuple[str, str]]: The column and the class it is expanded for ("" for a column without placeholder).

    Raises:
        ValueError: If no class of the run has the curve of a per-class curve column.
    """
    expanded_columns = []
    for column in columns:
        if CLASS_PLACEHOLDER not in column:
            expanded_columns.append((column, ""))
            continue

        # Count columns are given for every class, curve columns for the classes having the curve
        if COUNT_COLUMN_PATTERN.fullmatch(string=column.split("/")[0]) is not None:
            column_classes = class_names
        else:
            column_classes = [class_name for class_name in class_names
                              if column.replace(CLASS_PLACEHOLDER, class_name) in curve_names]
        if not column_classes:
            raise ValueError(f"No class of {class_names} has the curve {column!r}, available curves : "
                             f"{', '.join(curve_names)}")
        expanded_columns.extend((column, class_name) for class_name in column_classes)
    return expanded_columns


def format_count_value(column: str, class_name: str, class_records: List[Dict[str, Any]],
                       data_record: Dict[str, Any]) -> str:
    """
    Gives the value of a count column of a run: the total of a split ('train_images', the images of the split from its
    data.json, every image counting once whatever its classes; the sum over the classes for the other units) or the
    count of one class ('train_images/{class}').

    Args:
        column (str): The count column, e.g. "validation_images/{class}" or "train_annotations".
        class_name (str): The class of an expanded column, "" for a total.
        class_records (List[Dict[str, Any]]): The classes of the run with their counts (read_class_records).
        data_record (Dict[str, Any]): The data.json of the run.

    Returns:
        str: The count with thousands separators.

    Raises:
        ValueError: If the classes of the run have no count of that unit, e.g. frames for a classification run.
    """
    split_name, unit = COUNT_COLUMN_PATTERN.fullmatch(string=column.split("/")[0]).groups()
    if unit == "images" and not class_name:
        return f"{data_record['splits'][split_name]['number_images']:,}"
    available_units = [key for key in class_records[0] if key not in ["class_index", "class_name", "category_ids"]]
    if unit not in available_units:
        raise ValueError(f"The classes of the run have no {unit} count for the column {column!r}, available counts : "
                         f"{', '.join(available_units) or 'none'}")

    # One class, or the sum over the classes
    counts = [class_record[unit][split_name] for class_record in class_records
              if not class_name or class_record["class_name"] == class_name]
    return f"{sum(counts):,}"


def build_run_row(run_folder: Path, best_evaluation: Dict[str, Any], columns: List[str],
                  column_titles: Dict[str, str], number_decimals: int, curves_in_percent: bool) -> Dict[str, str]:
    """
    Builds the row of one run for the comparison tables: every column in its order of appearance, titled by
    column_titles (with the class name in place of the placeholder) or else by the column itself.

    Args:
        run_folder (Path): The run folder.
        best_evaluation (Dict[str, Any]): The best evaluation of the run (find_best_evaluation).
        columns (List[str]): The columns, in their order of appearance.
        column_titles (Dict[str, str]): The title of the columns not titled by themselves, e.g.
            {"Recall/{class}": "Recall {class}"}.
        number_decimals (int): The number of decimals of the curve values not written as percentages.
        curves_in_percent (bool): Whether the curve values are rates written as percentages ("94.03 %").

    Returns:
        Dict[str, str]: The formatted value of every column title.

    Raises:
        ValueError: If a curve column is missing from the best evaluation.
    """
    configuration_path = run_folder / "training_configuration.yaml"
    configuration = yaml.safe_load(stream=configuration_path.read_text(encoding="utf-8"))

    # Record of the data and classes of the run, read only when a column needs them
    data_record, class_records = {}, []
    if any(CLASS_PLACEHOLDER in column or COUNT_COLUMN_PATTERN.fullmatch(string=column) is not None
           for column in columns):
        data_record_path = run_folder / DATA_RECORD_FILE_NAME
        data_record = read_json(path=str(data_record_path))
        class_records = read_class_records(data_record=data_record, data_record_path=data_record_path,
                                           configuration=configuration)
    class_names = [class_record["class_name"] for class_record in class_records]

    row = {}
    for column, class_name in expand_columns(columns=columns, class_names=class_names,
                                             curve_names=list(best_evaluation["curves"])):
        title = column_titles[column] if column in column_titles else column
        title = title.replace(CLASS_PLACEHOLDER, class_name)

        # Run and best evaluation
        if column == "experiment":
            row[title] = run_folder.parent.name
        elif column == "run":
            row[title] = run_folder.name
        elif column == "best_iteration":
            row[title] = f"{best_evaluation['iteration']}" + (
                AVERAGED_WEIGHTS_SUFFIX if best_evaluation["weights"] == "ema" else "")

        # Setting of the training configuration, its keys joined by dots
        elif column.startswith(CONFIGURATION_COLUMN_PREFIX):
            row[title] = str(read_setting(configuration=configuration,
                                          keys=tuple(column[len(CONFIGURATION_COLUMN_PREFIX):].split(".")),
                                          configuration_path=configuration_path))

        # Count of a split, total or per class
        elif COUNT_COLUMN_PATTERN.fullmatch(string=column.split("/")[0]) is not None:
            row[title] = format_count_value(column=column, class_name=class_name, class_records=class_records,
                                            data_record=data_record)

        # Test curve at the best evaluation
        else:
            curve_name = column.replace(CLASS_PLACEHOLDER, class_name)
            if curve_name not in best_evaluation["curves"]:
                raise ValueError(f"No {curve_name} test curve at iteration {best_evaluation['iteration']} of "
                                 f"{format_path_link(path=run_folder)}, available curves : "
                                 f"{', '.join(best_evaluation['curves'])}")
            curve_value = best_evaluation["curves"][curve_name]
            row[title] = (format_percentage(value=curve_value) if curves_in_percent else
                          f"{curve_value:.{number_decimals}f}")
    return row


def format_markdown_table(title: str, rows: List[Dict[str, str]]) -> str:
    """
    Writes the rows of one comparison table as a Markdown table under its title, the rows sharing the same columns.

    Args:
        title (str): The title of the table, written as a level 3 heading.
        rows (List[Dict[str, str]]): The rows of the table (build_run_row).

    Returns:
        str: The Markdown text of the table.
    """
    column_titles = list(rows[0])
    lines = [f"### {title}", "", "| " + " | ".join(column_titles) + " |", "|---" * len(column_titles) + "|"]
    lines.extend("| " + " | ".join(row[column_title] for column_title in column_titles) + " |" for row in rows)
    return "\n".join(lines)


def compare_best_evaluations(experiment_folders: List[Path], best_metric: str, higher_is_better: bool,
                             evaluated_weights: List[str], columns: List[str], column_titles: Dict[str, str],
                             table_grouping: str, number_decimals: int, curves_in_percent: bool) -> str:
    """
    Builds the Markdown tables comparing the best evaluation of every run of the experiment folders: one table per run,
    or one table for the runs sharing the same column titles (the runs of the same classes), in the order of the runs.

    Args:
        experiment_folders (List[Path]): The experiment folders or run folders (find_run_folders).
        best_metric (str): The test curve choosing the best evaluation of every run.
        higher_is_better (bool): True when a higher value of the curve is better.
        evaluated_weights (List[str]): The weights whose evaluations compete, "trained" and/or "ema".
        columns (List[str]): The columns, in their order of appearance.
        column_titles (Dict[str, str]): The title of the columns not titled by themselves.
        table_grouping (str): "run" for one table per run, "columns" for one table per set of column titles.
        number_decimals (int): The number of decimals of the curve values not written as percentages.
        curves_in_percent (bool): Whether the curve values are rates written as percentages ("94.03 %").

    Returns:
        str: The Markdown tables, separated by an empty line.

    Raises:
        ValueError: If the table grouping or an evaluated weights name is unknown.
    """
    if table_grouping not in TABLE_GROUPINGS:
        raise ValueError(f"table_grouping must be one of {TABLE_GROUPINGS}, got {table_grouping!r}")
    unknown_weights = [weights for weights in evaluated_weights if weights not in TEST_FOLDERS]
    if unknown_weights:
        raise ValueError(f"evaluated_weights must be among {list(TEST_FOLDERS)}, got {unknown_weights}")

    # Row of every evaluated run, the runs not evaluated yet being listed and left out
    tables = {}
    for run_folder in find_run_folders(experiment_folders=experiment_folders):
        best_evaluation = find_best_evaluation(run_folder=run_folder, best_metric=best_metric,
                                               evaluated_weights=evaluated_weights, higher_is_better=higher_is_better)
        if best_evaluation is None:
            print_color(color="yellow",
                        output=f"No Test Evaluation Yet, Left Out : {format_path_link(path=run_folder)}")
            continue
        row = build_run_row(run_folder=run_folder, best_evaluation=best_evaluation, columns=columns,
                            column_titles=column_titles, number_decimals=number_decimals,
                            curves_in_percent=curves_in_percent)

        # One table per run, or the runs of the same column titles together under the names of their experiments
        table_key = f"{run_folder.parent.name} · {run_folder.name}" if table_grouping == "run" else tuple(row)
        tables.setdefault(table_key, []).append((run_folder, row))

    # Title of every table: its run, or the experiments of its runs
    markdown_tables = []
    for table_key, table_rows in tables.items():
        title = table_key if table_grouping == "run" else " · ".join(
            dict.fromkeys(run_folder.parent.name for run_folder, _ in table_rows))
        markdown_tables.append(format_markdown_table(title=title, rows=[row for _, row in table_rows]))
    return "\n\n".join(markdown_tables)


if __name__ == "__main__":
    repository_folder = Path(__file__).absolute().parents[2]
    # Experiment folders (the 'experiment_folder' of the configurations) whose run folders are compared, or run folders,
    # here the synthetic shapes classifier trained at four learning rates, told apart by their run name suffix
    run_name = "CLS_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HFLAT.128-HLN-CWB_MC9"
    experiment_folders = [repository_folder / "experiments" / "readme_screenshots" / f"{run_name}{suffix}"
                          for suffix in ["", "_lr-0.001", "_lr-0.0005", "_lr-0.0001"]]
    # Test curve choosing the best evaluation of every run, one of the scalars of Tensorboard/Test, e.g. "Accuracy",
    # "Balanced Accuracy" or "Macro F1" for the classification, and True when a higher value is better
    best_metric = "Balanced Accuracy"
    higher_is_better = True
    # Weights whose evaluations compete for the best: "trained" (Tensorboard/Test) and "ema" (Tensorboard/Test_EMA, for
    # the runs keeping a moving average of their weights), the best iteration ending with ' (EMA)' when they win
    evaluated_weights = ["trained", "ema"]
    # Columns of the tables, in their order of appearance:
    # - "experiment", "run": the experiment folder and the run folder; "best_iteration": the best evaluated iteration
    # - a test curve at the best evaluation, e.g. "Accuracy", "Balanced Accuracy", "Macro F1", "Recall/{class}",
    #   "Precision/{class}", "F1/{class}", {class} giving one column per class of the run having the curve
    # - "configuration:<keys joined by dots>": a setting of training_configuration.yaml, e.g.
    #   "configuration:Data.image_settings.size", "configuration:ExperimentConfiguration.learning_rate"
    # - "<split>_<unit>" and "<split>_<unit>/{class}": the total and the per-class counts of a split from data.json,
    #   the split being train, validation or test and the unit images (classification, self-supervised, semantic
    #   segmentation, detection tasks), annotations (detection tasks), sequences (sequence classification), frames
    #   (sequence labelling, frame selection) or pairs (sequence comparison)
    columns = ["configuration:Data.image_settings.size", "best_iteration", best_metric, "Recall/{class}",
               "validation_images", "validation_images/{class}", "train_images", "train_images/{class}"]
    # Titles of the columns not titled by themselves, {class} being replaced by the class name
    column_titles = {"configuration:Data.image_settings.size": "Image Size", "best_iteration": "Best Iteration",
                     "Recall/{class}": "Recall {class}", "validation_images": "Validation Images",
                     "validation_images/{class}": "Validation {class}", "train_images": "Train Images",
                     "train_images/{class}": "Train {class}"}
    # "run" for one table per run, "columns" for one table gathering the runs of the same column titles (same classes)
    table_grouping = "run"
    # Whether the curve values are written as the percentages of the reports ("94.03 %"), True for the classification
    # curves, which are all rates, and the number of decimals of the curve values otherwise
    curves_in_percent = True
    number_decimals = 3
    # Markdown file receiving the tables, None to only print them
    output_path = None

    markdown_text = compare_best_evaluations(experiment_folders=experiment_folders, best_metric=best_metric,
                                             higher_is_better=higher_is_better, evaluated_weights=evaluated_weights,
                                             columns=columns, column_titles=column_titles,
                                             table_grouping=table_grouping, number_decimals=number_decimals,
                                             curves_in_percent=curves_in_percent)
    print(markdown_text)
    if output_path is not None:
        Path(output_path).write_text(data=markdown_text + "\n", encoding="utf-8")
        print_color(color="green", output=f"Tables written to {format_path_link(path=Path(output_path))}")
