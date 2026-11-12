"""
Per-class part of the evaluation reports of the classification-like tasks: the metrics of every class (support,
precision, recall with the bounds of its confidence interval, F1, with the per-class report its ROC AUC and average
precision, and the threshold reaching every target precision with the recall it keeps) as the rows of
class_metrics.csv, as the per-class table of the report (with the table of the thresholds of every target precision
when several are given, the recall intervals being shown in the recall table of recall_intervals.py) and as the
per-class TensorBoard scalars. Used by the classification trainer, from which the sequence trainers inherit, and,
without the curves and thresholds, by the k-nearest neighbour report of the self-supervised trainer.
"""

from typing import Any, Dict, List, Optional, Tuple

from vision_training.core.contracts import EvaluationTable
from vision_training.evaluation.classification.score_curves import CURVE_METRIC_NAMES, ScoreCurves
from vision_training.evaluation.report_formatting import format_percentage_label

# Name of the per-class table of the report and of its sort column, the worst classes being shown first, and name of
# the table of the thresholds of every target precision
CLASS_TABLE_TITLE = "Per Class Metrics"
CLASS_TABLE_SORT_COLUMN = "F1"
CLASS_THRESHOLDS_TABLE_TITLE = "Per Class Thresholds"


def write_target_precision(target_precision: float, in_percent: bool) -> str:
    """
    Writes a target precision of the configuration in the headers and legends of the per-class tables: "99 %" with the
    percentages of the classification report, "0.99" for the reports kept between 0 and 1.

    Args:
        target_precision (float): The target precision, e.g. 0.99.
        in_percent (bool): Whether the report writes its rates as percentages.

    Returns:
        str: The written target precision.
    """
    return format_percentage_label(value=target_precision) if in_percent else f"{target_precision:g}"


def get_threshold_column_names(target_index: int, target_precision: float) -> Tuple[str, str]:
    """
    Names the columns of class_metrics.csv holding the threshold of a target precision and the recall it keeps: the
    main target precision (the first one, of the deployment) keeps 'threshold_at_target_precision' and
    'recall_at_target_precision', read by the tools of the repository, and every other one is named by its value,
    e.g. 'threshold_at_precision_0.95' and 'recall_at_precision_0.95'.

    Args:
        target_index (int): The position of the target precision in 'evaluation.target_precision', 0 for the main one.
        target_precision (float): The target precision, e.g. 0.95.

    Returns:
        Tuple[str, str]: The names of the threshold and recall columns.
    """
    if target_index == 0:
        return "threshold_at_target_precision", "recall_at_target_precision"
    return f"threshold_at_precision_{target_precision:g}", f"recall_at_precision_{target_precision:g}"


def build_class_metrics_rows(class_names: List[str], metrics: Dict[str, Any],
                             class_thresholds: Optional[List[List[Optional[Tuple[float, float]]]]] = None,
                             target_precisions: Optional[List[float]] = None,
                             class_curves: Optional[List[ScoreCurves]] = None) -> List[Dict[str, Any]]:
    """
    Gathers the metrics of every class of an evaluated checkpoint, in class index order, for the per-class metrics
    file and the per-class tables: its number of evaluated samples (support), its precision, its recall with the
    bounds of its confidence interval (recall_intervals.py) and its F1, with the per-class report its ROC AUC and
    average precision (None without positive, or negative for the ROC AUC), and, for the evaluations giving
    thresholds, the threshold reaching every target precision with the recall it keeps (None when no threshold
    reaches it).

    Args:
        class_names (List[str]): The name of every class, in class index order.
        metrics (Dict[str, Any]): The per-class 'support', 'precision', 'recall', 'recall_lower', 'recall_upper' and
            'f1' lists of the evaluation, the bounds None for a class without evaluated samples.
        class_thresholds (Optional[List[List[Optional[Tuple[float, float]]]]]): For every target precision, the
            threshold and recall of every class (None for a class without one), or None for an evaluation without
            thresholds (the k-nearest neighbour classification of the self-supervised task). Defaults to None.
        target_precisions (Optional[List[float]]): The target precisions of the thresholds, in the order of
            'class_thresholds', the first one being the main one, or None without thresholds. Defaults to None.
        class_curves (Optional[List[ScoreCurves]]): The ROC and precision-recall curves of every class, computed with
            the per-class report, or None without them. Defaults to None.

    Returns:
        List[Dict[str, Any]]: One row per class, keyed by the columns of class_metrics.csv: 'class_index', 'class',
            'samples', 'precision', 'recall', 'recall_lower', 'recall_upper', 'f1', then 'roc_auc' and
            'average_precision' when the curves are given,
            then the threshold and recall columns of every target precision when the thresholds are given
            (get_threshold_column_names).

    Raises:
        ValueError: If the thresholds are given without one target precision each.
    """
    if class_thresholds is not None and (target_precisions is None or len(target_precisions) != len(class_thresholds)):
        raise ValueError(f"Expected one target precision per list of thresholds, got {target_precisions!r} for "
                         f"{len(class_thresholds)} lists of thresholds.")
    class_rows = []
    for class_index, class_name in enumerate(class_names):
        class_row = {"class_index": class_index, "class": class_name, "samples": int(metrics["support"][class_index]),
                     "precision": metrics["precision"][class_index], "recall": metrics["recall"][class_index],
                     "recall_lower": metrics["recall_lower"][class_index],
                     "recall_upper": metrics["recall_upper"][class_index], "f1": metrics["f1"][class_index]}

        # ROC AUC and average precision of the class, with the per-class report
        if class_curves is not None:
            class_row.update({metric_key: class_curves[class_index][metric_key] for metric_key in CURVE_METRIC_NAMES})

        # Threshold reaching every target precision and the recall it keeps, for the evaluations giving thresholds
        for target_index, target_thresholds in enumerate(class_thresholds or []):
            threshold_column, recall_column = get_threshold_column_names(
                target_index=target_index, target_precision=target_precisions[target_index])
            class_threshold = target_thresholds[class_index]
            class_row.update({threshold_column: None if class_threshold is None else class_threshold[0],
                              recall_column: None if class_threshold is None else class_threshold[1]})
        class_rows.append(class_row)
    return class_rows


def build_class_metrics_table(class_rows: List[Dict[str, Any]], target_precision: Optional[float],
                              support_column: str, title: str = CLASS_TABLE_TITLE, legend_introduction: str = "",
                              support_description: Optional[str] = None, in_percent: bool = False) -> EvaluationTable:
    """
    Builds the per-class table of the report of an evaluated checkpoint, shown with the per-class report and sorted by
    ascending F1 so that the worst classes come first: the evaluated samples of every class (named by the task, e.g.
    'Samples', 'Positives' in multi-label mode, 'Frames' for sequence labelling), its precision, recall and F1, its ROC
    AUC and average precision when the rows hold them (with the per-class report), and, with a target precision, the
    threshold reaching it with the recall it keeps. A class without evaluated samples has no precision, recall nor F1
    in the table, like in the macro averages, and is listed last. The classification report writes the rates as
    percentages ("94.03 %") and the target precision as "99 %", the thresholds staying between 0 and 1; the k-nearest
    neighbour report of the self-supervised task keeps 4 decimals.

    Args:
        class_rows (List[Dict[str, Any]]): The metrics of every class (build_class_metrics_rows), with or without the
            'roc_auc' and 'average_precision' of the curves.
        target_precision (Optional[float]): The precision targeted by the thresholds, e.g. 0.99, or None for rows
            without thresholds (the k-nearest neighbour classification of the self-supervised task).
        support_column (str): The name of the column of the evaluated samples of every class.
        title (str): The title of the table. Defaults to CLASS_TABLE_TITLE.
        legend_introduction (str): The sentences put before the legend, describing the evaluation. Defaults to "".
        support_description (Optional[str]): The evaluated samples named in the legend for a class without any, e.g.
            "labelled test image", or None for the lowered support column. Defaults to None.
        in_percent (bool): Whether the rates are written as percentages and the target precision as "99 %". Defaults to
            False.

    Returns:
        EvaluationTable: The per-class table.
    """
    has_curve_metrics = bool(class_rows) and all(metric_key in class_rows[0] for metric_key in CURVE_METRIC_NAMES)
    rows = []
    for class_row in class_rows:
        has_samples = class_row["samples"] > 0
        row = [class_row["class"], class_row["samples"], class_row["precision"] if has_samples else None,
               class_row["recall"] if has_samples else None, class_row["f1"] if has_samples else None]

        # ROC AUC and average precision, when the rows hold the curve metrics
        if has_curve_metrics:
            row += [class_row[metric_key] for metric_key in CURVE_METRIC_NAMES]

        # Threshold and recall at the target precision, when the evaluation gives thresholds
        if target_precision is not None:
            row += [class_row["threshold_at_target_precision"], class_row["recall_at_target_precision"]]
        rows.append(row)

    # Columns and legend, the curve and threshold columns and their part of the legend only when they are given, the
    # rates listed as percent columns for the classification report (every column but the class, its samples and the
    # threshold)
    columns = ["Class", support_column, "Precision", "Recall", CLASS_TABLE_SORT_COLUMN]
    legend = (f"{legend_introduction}Worst F1 first, ties in class order. — : no "
              f"{support_column.lower() if support_description is None else support_description} of the class")
    if has_curve_metrics:
        columns += ["ROC AUC", "Average\nPrecision"]
        legend += ", no negative for the ROC AUC"
    threshold_columns = []
    if target_precision is not None:
        written_precision = write_target_precision(target_precision=target_precision, in_percent=in_percent)
        threshold_columns = [f"Threshold\n@ P{written_precision}"]
        columns += threshold_columns + [f"Recall\n@ P{written_precision}"]
        legend += f", or the precision {written_precision} is never reached"
    percent_columns = [column for column in columns[2:] if column not in threshold_columns] if in_percent else []
    return {"title": title, "columns": columns, "rows": rows, "sort_column": CLASS_TABLE_SORT_COLUMN,
            "per_class": True, "legend": f"{legend}.", "percent_columns": percent_columns}


def build_class_thresholds_table(class_rows: List[Dict[str, Any]], target_precisions: List[float],
                                 class_curves: Optional[List[ScoreCurves]] = None) -> EvaluationTable:
    """
    Builds the per-class thresholds table of the report of an evaluated checkpoint given several target precisions,
    shown with the per-class report next to the per-class table (which keeps the main target precision, so that it
    stays within the width of the report): the threshold of every class reaching every target precision with the
    recall it keeps, in class index order, to choose a threshold from the report. With the curves, every threshold is
    written as on precision_recall_thresholds.png, with the fewest decimals selecting the same samples
    (format_threshold_label), otherwise with the 4 decimals of the report. The recalls are written as percentages
    ("94.03 %") and the target precisions as "99 %", like in the per-class table of the classification report.

    Args:
        class_rows (List[Dict[str, Any]]): The metrics of every class (build_class_metrics_rows), with the threshold
            columns of every target precision.
        target_precisions (List[float]): The precisions targeted by the thresholds, the main one first.
        class_curves (Optional[List[ScoreCurves]]): The ROC and precision-recall curves of every class, with the
            written value of every threshold, or None without them. Defaults to None.

    Returns:
        EvaluationTable: The per-class thresholds table.
    """
    rows = []
    for class_index, class_row in enumerate(class_rows):
        row = [class_row["class"]]
        for target_index, target_precision in enumerate(target_precisions):
            threshold_column, recall_column = get_threshold_column_names(target_index=target_index,
                                                                         target_precision=target_precision)
            # Threshold written as on the image when the curves give it, the recall it keeps after it
            threshold = class_row[threshold_column]
            if threshold is not None and class_curves is not None:
                threshold = class_curves[class_index]["threshold_points"][target_index][2]
            row += [threshold, class_row[recall_column]]
        rows.append(row)

    # Two columns per target precision, the main one first, the recalls written as percentages
    written_precisions = [write_target_precision(target_precision=target_precision, in_percent=True)
                          for target_precision in target_precisions]
    columns, recall_columns = ["Class"], []
    for written_precision in written_precisions:
        recall_columns.append(f"Recall\n@ P{written_precision}")
        columns += [f"Threshold\n@ P{written_precision}", recall_columns[-1]]
    legend = (f"Lowest score threshold of every class keeping the precision, which gives the highest recall, in class "
              f"order, exact in class_thresholds.json. — : the precision {' or '.join(written_precisions)} is never "
              f"reached.")
    return {"title": CLASS_THRESHOLDS_TABLE_TITLE, "columns": columns, "rows": rows, "sort_column": None,
            "per_class": True, "legend": legend, "percent_columns": recall_columns}


def build_class_metric_scalars(class_names: List[str], metrics: Dict[str, Any], tag_prefix: str = "",
                               class_curves: Optional[List[ScoreCurves]] = None) -> Dict[str, float]:
    """
    Builds the per-class TensorBoard scalars of an evaluated checkpoint, written with the per-class report: the
    precision, recall and F1 of every class, '<prefix>Precision/<class>', '<prefix>Recall/<class>' and
    '<prefix>F1/<class>', and, when the curves are given, the 'ROC AUC/<class>' and 'Average Precision/<class>' of
    every class defining them.

    Args:
        class_names (List[str]): The name of every class, in class index order.
        metrics (Dict[str, Any]): The per-class 'precision', 'recall' and 'f1' lists of the evaluation.
        tag_prefix (str): The text put before every tag, e.g. "k-NN " for the k-nearest neighbour classification of
            the self-supervised task. Defaults to "".
        class_curves (Optional[List[ScoreCurves]]): The ROC and precision-recall curves of every class, or None without
            them. Defaults to None.

    Returns:
        Dict[str, float]: The per-class scalars, keyed by TensorBoard tag.
    """
    class_scalars = {}
    for class_index, class_name in enumerate(class_names):
        for metric_name in ["precision", "recall", "f1"]:
            class_scalars[f"{tag_prefix}{metric_name.capitalize()}/{class_name}"] = metrics[metric_name][class_index]

        # ROC AUC and average precision of the classes defining them, when the curves are given
        if class_curves is not None:
            for metric_key, metric_name in CURVE_METRIC_NAMES.items():
                if class_curves[class_index][metric_key] is not None:
                    class_scalars[f"{tag_prefix}{metric_name}/{class_name}"] = class_curves[class_index][metric_key]
    return class_scalars
