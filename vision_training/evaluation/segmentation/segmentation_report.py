"""
Per-class part of the evaluation reports of the semantic segmentation task: the metrics of every class (true pixels,
IoU, Dice, precision, recall) as the rows of class_metrics.csv, as the per-class table of the report and as the
per-class TensorBoard scalars. Used by the segmentation trainer.
"""

from typing import Any, Dict, List

from vision_training.core.contracts import EvaluationTable

# Name of the per-class table of the report and of its sort column, the worst classes being shown first
CLASS_TABLE_TITLE = "Per Class Metrics"
CLASS_TABLE_SORT_COLUMN = "IoU"

# Metrics of every class, in the order of their columns, with their names in the table and in TensorBoard
CLASS_METRIC_NAMES = {"iou": "IoU", "dice": "Dice", "precision": "Precision", "recall": "Recall"}


def build_segmentation_class_rows(class_names: List[str], metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Gathers the metrics of every class of an evaluated checkpoint, in class index order, for the per-class metrics
    file, table and scalars: its number of true pixels in the test split, its IoU, Dice, precision and recall. A class
    without true pixels has no metric (None, an empty cell of the file and "—" in the table), being left out of the
    means.

    Args:
        class_names (List[str]): The name of every class, in class index order, the background first.
        metrics (Dict[str, Any]): The per-class 'support', 'iou', 'dice', 'precision' and 'recall' lists of the
            evaluation (compute_segmentation_metrics).

    Returns:
        List[Dict[str, Any]]: One row per class, keyed by the columns of class_metrics.csv: 'class_index', 'class',
            'pixels', 'iou', 'dice', 'precision' and 'recall'.
    """
    class_rows = []
    for class_index, class_name in enumerate(class_names):
        number_pixels = int(metrics["support"][class_index])
        class_row = {"class_index": class_index, "class": class_name, "pixels": number_pixels}
        for metric_key in CLASS_METRIC_NAMES:
            class_row[metric_key] = metrics[metric_key][class_index] if number_pixels > 0 else None
        class_rows.append(class_row)
    return class_rows


def build_segmentation_class_table(class_rows: List[Dict[str, Any]]) -> EvaluationTable:
    """
    Builds the per-class table of the report of an evaluated checkpoint, shown with the per-class report and sorted by
    ascending IoU so that the worst segmented classes come first, the classes without true pixels being listed last.

    Args:
        class_rows (List[Dict[str, Any]]): The metrics of every class (build_segmentation_class_rows).

    Returns:
        EvaluationTable: The per-class table.
    """
    rows = [[class_row["class"], class_row["pixels"]] + [class_row[metric_key] for metric_key in CLASS_METRIC_NAMES]
            for class_row in class_rows]
    return {"title": CLASS_TABLE_TITLE, "columns": ["Class", "Pixels"] + list(CLASS_METRIC_NAMES.values()),
            "rows": rows, "sort_column": CLASS_TABLE_SORT_COLUMN, "per_class": True,
            "legend": "Worst IoU first, ties in class order. — : no pixel of the class in the test masks."}


def build_segmentation_class_scalars(class_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Builds the per-class TensorBoard scalars of an evaluated checkpoint, written with the per-class report: the IoU,
    Dice, precision and recall of every class with true pixels, 'IoU/<class>', 'Dice/<class>', 'Precision/<class>' and
    'Recall/<class>'.

    Args:
        class_rows (List[Dict[str, Any]]): The metrics of every class (build_segmentation_class_rows).

    Returns:
        Dict[str, float]: The per-class scalars, keyed by TensorBoard tag.
    """
    class_scalars = {}
    for class_row in class_rows:
        if class_row["pixels"] == 0:
            continue
        for metric_key, metric_name in CLASS_METRIC_NAMES.items():
            class_scalars[f"{metric_name}/{class_row['class']}"] = class_row[metric_key]
    return class_scalars
