"""
Per-class part of the evaluation reports of the semantic segmentation task: the metrics of every class (true pixels,
IoU, Dice, boundary IoU, boundary F1, precision, recall with the bounds of its Korn-Graubard interval over the test
images, and with the per-class report the ROC AUC and average precision of its pixels) as the rows of
class_metrics.csv, as the per-class table of the report (the recall intervals in the recall table of
recall_intervals.py), its rates written as percentages ("94.03 %"), and as the per-class TensorBoard scalars, which
keep them between 0 and 1 like class_metrics.csv. Used by the segmentation trainer.
"""

from typing import Any, Dict, List, Optional

from vision_training.core.contracts import EvaluationTable
from vision_training.evaluation.classification.score_curves import CURVE_METRIC_NAMES, ScoreCurves

# Name of the per-class table of the report and of its sort column, the worst classes being shown first
CLASS_TABLE_TITLE = "Per Class Metrics"
CLASS_TABLE_SORT_COLUMN = "IoU"

# Metrics of every class, in the order of their columns, with their names in TensorBoard
CLASS_METRIC_NAMES = {"iou": "IoU", "dice": "Dice", "boundary_iou": "Boundary IoU", "boundary_f1": "Boundary F1",
                      "precision": "Precision", "recall": "Recall"}

# Headers of the metric columns of the per-class table, the long names on two lines like in the classification table,
# so that the ten columns of percentages ("94.03 %") stay within the width of the report without cutting any value
CLASS_COLUMN_HEADERS = {"iou": "IoU", "dice": "Dice", "boundary_iou": "Boundary\nIoU", "boundary_f1": "Boundary\nF1",
                        "precision": "Precision", "recall": "Recall", "roc_auc": "ROC AUC",
                        "average_precision": "Average\nPrecision"}


def build_segmentation_class_rows(class_names: List[str], metrics: Dict[str, Any],
                                  class_curves: Optional[List[ScoreCurves]] = None) -> List[Dict[str, Any]]:
    """
    Gathers the metrics of every class of an evaluated checkpoint, in class index order, for the per-class metrics
    file, tables and scalars: its number of true pixels in the test split, its IoU, Dice, boundary IoU, boundary F1,
    precision and recall with the bounds of its Korn-Graubard interval over the test images, and with the per-class
    report the ROC AUC and average precision of its pixels. A class without true pixels has no metric (None, an empty
    cell of the file and "—" in the table), being left out of the means, a class without true edge has no boundary
    metric, and a class covering every labelled pixel has no ROC AUC.

    Args:
        class_names (List[str]): The name of every class, in class index order, the background first.
        metrics (Dict[str, Any]): The per-class 'support', 'iou', 'dice', 'precision' and 'recall' lists of the
            evaluation (compute_segmentation_metrics), its 'boundary_iou' and 'boundary_f1' lists
            (compute_boundary_metrics), and its 'recall_lower' and 'recall_upper' lists
            (compute_bootstrap_recall_intervals).
        class_curves (Optional[List[ScoreCurves]]): The ROC and precision-recall curves of the pixels of every class,
            computed with the per-class report, or None without them. Defaults to None.

    Returns:
        List[Dict[str, Any]]: One row per class, keyed by the columns of class_metrics.csv: 'class_index', 'class',
            'pixels', 'iou', 'dice', 'boundary_iou', 'boundary_f1', 'precision', 'recall', 'recall_lower' and
            'recall_upper', then 'roc_auc' and 'average_precision' when the curves are given.
    """
    class_rows = []
    for class_index, class_name in enumerate(class_names):
        number_pixels = int(metrics["support"][class_index])
        class_row = {"class_index": class_index, "class": class_name, "pixels": number_pixels}
        for metric_key in CLASS_METRIC_NAMES:
            class_row[metric_key] = metrics[metric_key][class_index] if number_pixels > 0 else None

        # Bounds of the interval of the recall, after the recall, None without true pixels
        class_row["recall_lower"] = metrics["recall_lower"][class_index] if number_pixels > 0 else None
        class_row["recall_upper"] = metrics["recall_upper"][class_index] if number_pixels > 0 else None

        # ROC AUC and average precision of the pixels of the class, with the per-class report
        if class_curves is not None:
            class_row.update({metric_key: class_curves[class_index][metric_key] for metric_key in CURVE_METRIC_NAMES})
        class_rows.append(class_row)
    return class_rows


def build_segmentation_class_table(class_rows: List[Dict[str, Any]]) -> EvaluationTable:
    """
    Builds the per-class table of the report of an evaluated checkpoint, shown with the per-class report and sorted by
    ascending IoU so that the worst segmented classes come first, the classes without true pixels being listed last,
    with the ROC AUC and average precision of every class when the rows hold them. Every metric column is a rate
    written as a percentage ("94.03 %"), under the headers of CLASS_COLUMN_HEADERS.

    Args:
        class_rows (List[Dict[str, Any]]): The metrics of every class (build_segmentation_class_rows).

    Returns:
        EvaluationTable: The per-class table.
    """
    metric_keys = list(CLASS_METRIC_NAMES)
    legend = ("Worst IoU first, ties in class order. Boundary IoU and F1 : contours within the boundary tolerance. — : "
              "no pixel (or no edge for the boundary metrics) of the class in the test masks")

    # ROC AUC and average precision columns, when the rows hold the curve metrics
    if class_rows and all(metric_key in class_rows[0] for metric_key in CURVE_METRIC_NAMES):
        metric_keys += list(CURVE_METRIC_NAMES)
        legend += ", or no pixel of another class for the ROC AUC"
    rows = [[class_row["class"], class_row["pixels"]] + [class_row[metric_key] for metric_key in metric_keys]
            for class_row in class_rows]

    # Every column after the class and its pixels is a rate, written as a percentage
    metric_columns = [CLASS_COLUMN_HEADERS[metric_key] for metric_key in metric_keys]
    return {"title": CLASS_TABLE_TITLE, "columns": ["Class", "Pixels"] + metric_columns, "rows": rows,
            "sort_column": CLASS_TABLE_SORT_COLUMN, "per_class": True, "percent_columns": metric_columns,
            "legend": f"{legend}."}


def build_segmentation_class_scalars(class_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Builds the per-class TensorBoard scalars of an evaluated checkpoint, written with the per-class report: the IoU,
    Dice, boundary IoU, boundary F1, precision and recall of every class with true pixels, e.g. 'IoU/<class>' and
    'Boundary F1/<class>', the boundary metrics of a class without true edge being left out, and the 'ROC AUC/<class>'
    and 'Average Precision/<class>' of the classes defining them when the rows hold them.

    Args:
        class_rows (List[Dict[str, Any]]): The metrics of every class (build_segmentation_class_rows).

    Returns:
        Dict[str, float]: The per-class scalars, keyed by TensorBoard tag.
    """
    class_scalars = {}
    for class_row in class_rows:
        if class_row["pixels"] == 0:
            continue
        for metric_key, metric_name in {**CLASS_METRIC_NAMES, **CURVE_METRIC_NAMES}.items():
            if metric_key in class_row and class_row[metric_key] is not None:
                class_scalars[f"{metric_name}/{class_row['class']}"] = class_row[metric_key]
    return class_scalars
