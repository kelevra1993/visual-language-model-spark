"""
Per-class part of the evaluation reports of the classification-like tasks: the metrics of every class (support,
precision, recall, F1, with the per-class report its ROC AUC and average precision, and the threshold reaching the
target precision with the recall it keeps) as the rows of class_metrics.csv, as the per-class table of the report and
as the per-class TensorBoard scalars. Used by the classification trainer, from which the sequence trainers inherit,
and, without the curves and thresholds, by the k-nearest neighbour report of the self-supervised trainer.
"""

from typing import Any, Dict, List, Optional, Tuple

from vision_training.core.contracts import EvaluationTable
from vision_training.evaluation.classification.score_curves import CURVE_METRIC_NAMES, ScoreCurves

# Name of the per-class table of the report and of its sort column, the worst classes being shown first
CLASS_TABLE_TITLE = "Per Class Metrics"
CLASS_TABLE_SORT_COLUMN = "F1"


def build_class_metrics_rows(class_names: List[str], metrics: Dict[str, Any],
                             class_thresholds: Optional[List[Optional[Tuple[float, float]]]] = None,
                             class_curves: Optional[List[ScoreCurves]] = None) -> List[Dict[str, Any]]:
    """
    Gathers the metrics of every class of an evaluated checkpoint, in class index order, for the per-class metrics
    file and the per-class table: its number of evaluated samples (support), its precision, recall and F1, with the
    per-class report its ROC AUC and average precision (None without positive, or negative for the ROC AUC), and, for
    the evaluations giving thresholds, the threshold reaching the target precision with the recall it keeps (None when
    no threshold reaches it).

    Args:
        class_names (List[str]): The name of every class, in class index order.
        metrics (Dict[str, Any]): The per-class 'support', 'precision', 'recall' and 'f1' lists of the evaluation.
        class_thresholds (Optional[List[Optional[Tuple[float, float]]]]): The threshold and recall of every class (None
            for a class without one), or None for an evaluation without thresholds (the k-nearest neighbour
            classification of the self-supervised task). Defaults to None.
        class_curves (Optional[List[ScoreCurves]]): The ROC and precision-recall curves of every class, computed with
            the per-class report, or None without them. Defaults to None.

    Returns:
        List[Dict[str, Any]]: One row per class, keyed by the columns of class_metrics.csv: 'class_index', 'class',
            'samples', 'precision', 'recall', 'f1', then 'roc_auc' and 'average_precision' when the curves are given,
            then 'threshold_at_target_precision' and 'recall_at_target_precision' when the thresholds are given.
    """
    class_rows = []
    for class_index, class_name in enumerate(class_names):
        class_row = {"class_index": class_index, "class": class_name, "samples": int(metrics["support"][class_index]),
                     "precision": metrics["precision"][class_index], "recall": metrics["recall"][class_index],
                     "f1": metrics["f1"][class_index]}

        # ROC AUC and average precision of the class, with the per-class report
        if class_curves is not None:
            class_row.update({metric_key: class_curves[class_index][metric_key] for metric_key in CURVE_METRIC_NAMES})

        # Threshold reaching the target precision and the recall it keeps, for the evaluations giving thresholds
        if class_thresholds is not None:
            class_threshold = class_thresholds[class_index]
            class_row.update({"threshold_at_target_precision": None if class_threshold is None else class_threshold[0],
                              "recall_at_target_precision": None if class_threshold is None else class_threshold[1]})
        class_rows.append(class_row)
    return class_rows


def build_class_metrics_table(class_rows: List[Dict[str, Any]], target_precision: Optional[float],
                              support_column: str, title: str = CLASS_TABLE_TITLE, legend_introduction: str = "",
                              support_description: Optional[str] = None) -> EvaluationTable:
    """
    Builds the per-class table of the report of an evaluated checkpoint, shown with the per-class report and sorted by
    ascending F1 so that the worst classes come first: the evaluated samples of every class (named by the task, e.g.
    'Samples', 'Positives' in multi-label mode, 'Frames' for sequence labelling), its precision, recall and F1, its ROC
    AUC and average precision when the rows hold them (with the per-class report), and, with a target precision, the
    threshold reaching it with the recall it keeps. A class without evaluated samples has no precision, recall nor F1
    in the table, like in the macro averages, and is listed last.

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

    # Columns and legend, the curve and threshold columns and their part of the legend only when they are given
    columns = ["Class", support_column, "Precision", "Recall", CLASS_TABLE_SORT_COLUMN]
    legend = (f"{legend_introduction}Worst F1 first, ties in class order. — : no "
              f"{support_column.lower() if support_description is None else support_description} of the class")
    if has_curve_metrics:
        columns += ["ROC AUC", "Average\nPrecision"]
        legend += ", no negative for the ROC AUC"
    if target_precision is not None:
        columns += [f"Threshold\n@ P{target_precision:g}", f"Recall\n@ P{target_precision:g}"]
        legend += f", or the precision {target_precision:g} is never reached"
    return {"title": title, "columns": columns, "rows": rows, "sort_column": CLASS_TABLE_SORT_COLUMN,
            "per_class": True, "legend": f"{legend}."}


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
