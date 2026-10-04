"""
Per-class part of the evaluation reports of the classification-like tasks: the metrics of every class (support,
precision, recall, F1, and the threshold reaching the target precision with the recall it keeps) as the rows of
class_metrics.csv, as the per-class table of the report and as the per-class TensorBoard scalars. Used by the
classification trainer, from which the sequence trainers inherit.
"""

from typing import Any, Dict, List, Optional, Tuple

from vision_training.core.contracts import EvaluationTable

# Name of the per-class table of the report and of its sort column, the worst classes being shown first
CLASS_TABLE_TITLE = "Per Class Metrics"
CLASS_TABLE_SORT_COLUMN = "F1"


def build_class_metrics_rows(class_names: List[str], metrics: Dict[str, Any],
                             class_thresholds: List[Optional[Tuple[float, float]]]) -> List[Dict[str, Any]]:
    """
    Gathers the metrics of every class of an evaluated checkpoint, in class index order, for the per-class metrics
    file and the per-class table: its number of evaluated samples (support), its precision, recall and F1, and the
    threshold reaching the target precision with the recall it keeps (None when no threshold reaches it).

    Args:
        class_names (List[str]): The name of every class, in class index order.
        metrics (Dict[str, Any]): The per-class 'support', 'precision', 'recall' and 'f1' lists of the evaluation.
        class_thresholds (List[Optional[Tuple[float, float]]]): The threshold and recall of every class, or None.

    Returns:
        List[Dict[str, Any]]: One row per class, keyed by the columns of class_metrics.csv: 'class_index', 'class',
            'samples', 'precision', 'recall', 'f1', 'threshold_at_target_precision' and 'recall_at_target_precision'.
    """
    class_rows = []
    for class_index, class_name in enumerate(class_names):
        class_threshold = class_thresholds[class_index]
        class_rows.append({"class_index": class_index, "class": class_name,
                           "samples": int(metrics["support"][class_index]),
                           "precision": metrics["precision"][class_index], "recall": metrics["recall"][class_index],
                           "f1": metrics["f1"][class_index],
                           "threshold_at_target_precision": None if class_threshold is None else class_threshold[0],
                           "recall_at_target_precision": None if class_threshold is None else class_threshold[1]})
    return class_rows


def build_class_metrics_table(class_rows: List[Dict[str, Any]], target_precision: float,
                              support_column: str) -> EvaluationTable:
    """
    Builds the per-class table of the report of an evaluated checkpoint, shown with the per-class report and sorted by
    ascending F1 so that the worst classes come first: the evaluated samples of every class (named by the task, e.g.
    'Samples', 'Positives' in multi-label mode, 'Frames' for sequence labelling), its precision, recall and F1, and the
    threshold reaching the target precision with the recall it keeps. A class without evaluated samples has no
    precision, recall nor F1 in the table, like in the macro averages, and is listed last.

    Args:
        class_rows (List[Dict[str, Any]]): The metrics of every class (build_class_metrics_rows).
        target_precision (float): The precision targeted by the thresholds, e.g. 0.99.
        support_column (str): The name of the column of the evaluated samples of every class.

    Returns:
        EvaluationTable: The per-class table.
    """
    rows = []
    for class_row in class_rows:
        has_samples = class_row["samples"] > 0
        rows.append([class_row["class"], class_row["samples"], class_row["precision"] if has_samples else None,
                     class_row["recall"] if has_samples else None, class_row["f1"] if has_samples else None,
                     class_row["threshold_at_target_precision"], class_row["recall_at_target_precision"]])
    return {"title": CLASS_TABLE_TITLE,
            "columns": ["Class", support_column, "Precision", "Recall", CLASS_TABLE_SORT_COLUMN,
                        f"Threshold\n@ P{target_precision:g}", f"Recall\n@ P{target_precision:g}"],
            "rows": rows, "sort_column": CLASS_TABLE_SORT_COLUMN, "per_class": True,
            "legend": f"Worst F1 first, ties in class order. — : no {support_column.lower()} of the class, or the "
                      f"precision {target_precision:g} is never reached."}


def build_class_metric_scalars(class_names: List[str], metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Builds the per-class TensorBoard scalars of an evaluated checkpoint, written with the per-class report: the
    precision, recall and F1 of every class, 'Precision/<class>', 'Recall/<class>' and 'F1/<class>'.

    Args:
        class_names (List[str]): The name of every class, in class index order.
        metrics (Dict[str, Any]): The per-class 'precision', 'recall' and 'f1' lists of the evaluation.

    Returns:
        Dict[str, float]: The per-class scalars, keyed by TensorBoard tag.
    """
    class_scalars = {}
    for class_index, class_name in enumerate(class_names):
        for metric_name in ["precision", "recall", "f1"]:
            class_scalars[f"{metric_name.capitalize()}/{class_name}"] = metrics[metric_name][class_index]
    return class_scalars
