"""
Tables and per-class rows of the evaluation reports of the object detection task: the mean average precisions of the
three average precision methods, always shown, and the average precisions of every class (area method, at an
intersection over union of 0.5, 0.75 and 0.5 to 0.95, with its ground truth boxes), as the rows of class_metrics.csv
and as the per-class table of the report. Used by the detection trainer.
"""

from typing import Any, Dict, List

from vision_training.core.contracts import EvaluationTable
from vision_training.evaluation.detection.mean_average_precision import AVERAGE_PRECISION_METHODS

# Displayed name of every average precision method, the area method being the headline one
AVERAGE_PRECISION_METHOD_NAMES = {"area": "Area", "11_point": "11 Point", "101_point": "101 Point"}

# Name of the per-class table of the report and of its sort column, the worst classes being shown first
CLASS_TABLE_TITLE = "Per Class Average Precision (Area)"
CLASS_TABLE_SORT_COLUMN = "AP@0.5"


def build_mean_average_precision_table(results: Dict[str, Dict[str, Any]]) -> EvaluationTable:
    """
    Builds the table of the mean average precisions of an evaluated checkpoint, shown whatever the per-class report:
    one row per average precision method, with its mAP at an intersection over union of 0.5 (PASCAL VOC), 0.75 and
    averaged over 0.5 to 0.95 (COCO).

    Args:
        results (Dict[str, Dict[str, Any]]): The results of evaluate_detections, keyed by method.

    Returns:
        EvaluationTable: The table of the methods.
    """
    rows = [[AVERAGE_PRECISION_METHOD_NAMES[method], results[method]["mAP@0.5"], results[method]["mAP@0.75"],
             results[method]["mAP@[0.5:0.95]"]] for method in AVERAGE_PRECISION_METHODS]
    return {"title": "Mean Average Precision", "columns": ["Method", "mAP@0.5", "mAP@0.75", "mAP@[0.5:0.95]"],
            "rows": rows, "sort_column": None, "per_class": False,
            "legend": "The mAP@0.5 of the area method compares the checkpoints of the run."}


def build_class_average_precision_rows(results: Dict[str, Dict[str, Any]],
                                       class_names: Dict[int, str]) -> List[Dict[str, Any]]:
    """
    Gathers the average precisions of every foreground class of an evaluated checkpoint with the area method, in class
    index order, for the per-class metrics file and the per-class table: its ground truth boxes and its average
    precision at 0.5, 0.75 and averaged over 0.5 to 0.95 (None for a class without ground truth, which has no recall).

    Args:
        results (Dict[str, Dict[str, Any]]): The results of evaluate_detections, keyed by method.
        class_names (Dict[int, str]): The name of every class index.

    Returns:
        List[Dict[str, Any]]: One row per foreground class, keyed by the columns of class_metrics.csv: 'class_index',
            'class', 'ground_truths', 'average_precision_50', 'average_precision_75' and 'average_precision_50_95'.
    """
    area_results = results["area"]
    class_rows = []
    for class_index in sorted(area_results["class_ground_truths"]):
        class_rows.append({"class_index": class_index, "class": class_names[class_index],
                           "ground_truths": area_results["class_ground_truths"][class_index]})

        # Average precisions of the class, undefined without ground truth
        for column_name, result_key in [("average_precision_50", "class_AP@0.5"),
                                        ("average_precision_75", "class_AP@0.75"),
                                        ("average_precision_50_95", "class_AP@[0.5:0.95]")]:
            class_average_precisions = area_results[result_key]
            class_rows[-1][column_name] = (class_average_precisions[class_index]
                                           if class_index in class_average_precisions else None)
    return class_rows


def build_class_average_precision_table(class_rows: List[Dict[str, Any]]) -> EvaluationTable:
    """
    Builds the per-class table of the report of an evaluated checkpoint, shown with the per-class report and sorted by
    ascending AP@0.5 so that the worst classes come first: the ground truth boxes of every class and its average
    precisions with the area method, a class without ground truth being listed last.

    Args:
        class_rows (List[Dict[str, Any]]): The average precisions of every class (build_class_average_precision_rows).

    Returns:
        EvaluationTable: The per-class table.
    """
    rows = [[class_row["class"], class_row["ground_truths"], class_row["average_precision_50"],
             class_row["average_precision_75"], class_row["average_precision_50_95"]] for class_row in class_rows]
    return {"title": CLASS_TABLE_TITLE,
            "columns": ["Class", "Ground\nTruths", CLASS_TABLE_SORT_COLUMN, "AP@0.75", "AP@[0.5:0.95]"],
            "rows": rows, "sort_column": CLASS_TABLE_SORT_COLUMN, "per_class": True,
            "legend": "Worst AP@0.5 first, ties in class order. — : no ground truth box of the class."}
