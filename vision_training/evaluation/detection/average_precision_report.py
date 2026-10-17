"""
Tables and per-class rows of the evaluation reports of the object detection and instance segmentation tasks: the mean
average precisions of the three average precision methods, those of the small, medium and large objects (area method)
and the mean average recalls of all the objects and of every size, always shown, and the average precisions of every
class (area method, at an intersection over union of 0.5, 0.75 and 0.5 to 0.95, with its ground truth boxes, those of
every object size, and the score threshold reaching the target precision with the recall it keeps), as the rows of
class_metrics.csv and as the per-class table of the report. The instance segmentation reports the box and the mask
average precisions and recalls with the same builders, told apart by a title prefix and a column prefix. Used by the
detection and instance segmentation trainers.
"""

from typing import Any, Dict, List, Optional, Tuple

from vision_training.core.contracts import EvaluationTable
from vision_training.evaluation.detection.mean_average_precision import AVERAGE_PRECISION_METHODS
from vision_training.evaluation.detection.object_sizes import OBJECT_SIZE_NAMES

# Displayed name of every average precision method, the area method being the headline one
AVERAGE_PRECISION_METHOD_NAMES = {"area": "Area", "11_point": "11 Point", "101_point": "101 Point"}

# Name of the per-class table of the report and of its sort column, the worst classes being shown first
CLASS_TABLE_TITLE = "Per Class Average Precision (Area)"
CLASS_TABLE_SORT_COLUMN = "AP@0.5"

# Legend of the table of the methods whose mAP compares the checkpoints of the run
HEADLINE_LEGEND = "The mAP@0.5 of the area method compares the checkpoints of the run."


def add_title_prefix(title: str, title_prefix: str) -> str:
    """
    Puts the prefix of the evaluated outputs ("Box", "Mask") before a table title, so that the box and mask tables of
    the instance segmentation report are told apart while the object detection titles stay unchanged.

    Args:
        title (str): The title of the table.
        title_prefix (str): The prefix, empty for none.

    Returns:
        str: The prefixed title.
    """
    return f"{title_prefix} {title}" if title_prefix else title


def build_mean_average_precision_table(results: Dict[str, Dict[Any, Any]], title_prefix: str = "",
                                       legend: Optional[str] = None) -> EvaluationTable:
    """
    Builds the table of the mean average precisions of an evaluated checkpoint, shown whatever the per-class report:
    one row per average precision method, with its mAP at an intersection over union of 0.5 (PASCAL VOC), 0.75 and
    averaged over 0.5 to 0.95 (COCO).

    Args:
        results (Dict[str, Dict[Any, Any]]): The results of evaluate_detections, keyed by method.
        title_prefix (str): The prefix of the title ("Box", "Mask"), empty for the object detection. Defaults to "".
        legend (Optional[str]): The legend naming the mAP that compares the checkpoints, None for HEADLINE_LEGEND.
            Defaults to None.

    Returns:
        EvaluationTable: The table of the methods.
    """
    rows = [[AVERAGE_PRECISION_METHOD_NAMES[method], results[method]["mAP@0.5"], results[method]["mAP@0.75"],
             results[method]["mAP@[0.5:0.95]"]] for method in AVERAGE_PRECISION_METHODS]
    return {"title": add_title_prefix(title="Mean Average Precision", title_prefix=title_prefix),
            "columns": ["Method", "mAP@0.5", "mAP@0.75", "mAP@[0.5:0.95]"], "rows": rows, "sort_column": None,
            "per_class": False, "legend": HEADLINE_LEGEND if legend is None else legend}


def build_object_size_table(results: Dict[str, Dict[Any, Any]], title_prefix: str = "", area_name: str = "box",
                            resolution_name: str = "network input") -> EvaluationTable:
    """
    Builds the table of the mean average precisions of the small, medium and large objects of an evaluated checkpoint
    with the area method, shown whatever the per-class report, telling whether the detector misses the objects of one
    size: one row per size, with its objects and its mAP at an intersection over union of 0.5, 0.75 and averaged over
    0.5 to 0.95.

    Args:
        results (Dict[str, Dict[Any, Any]]): The results of evaluate_detections, keyed by method.
        title_prefix (str): The prefix of the title ("Box", "Mask"), empty for the object detection. Defaults to "".
        area_name (str): The area measuring the objects, "box", or "mask" for the mask average precisions. Defaults
            to "box".
        resolution_name (str): The resolution of the pixels measuring the objects, "network input", or "original
            image" for the instance segmentation evaluated at the original resolution. Defaults to "network input".

    Returns:
        EvaluationTable: The table of the object sizes.
    """
    size_results = results["area"]["sizes"]
    rows = [[size_name, sum(size_results[object_size]["class_ground_truths"].values()),
             size_results[object_size]["mAP@0.5"], size_results[object_size]["mAP@0.75"],
             size_results[object_size]["mAP@[0.5:0.95]"]] for object_size, size_name in OBJECT_SIZE_NAMES.items()]
    return {"title": add_title_prefix(title="Mean Average Precision By Object Size (Area)", title_prefix=title_prefix),
            "columns": ["Object Size", "Objects", "mAP@0.5", "mAP@0.75", "mAP@[0.5:0.95]"], "rows": rows,
            "sort_column": None, "per_class": False,
            "legend": f"COCO sizes by {area_name} area in {resolution_name} pixels, the objects of the other sizes and "
                      f"the detections finding them being ignored. — : no object of the size."}


def build_average_recall_table(results: Dict[str, Dict[Any, Any]], title_prefix: str = "", area_name: str = "box",
                               resolution_name: str = "network input") -> EvaluationTable:
    """
    Builds the table of the mean average recalls of an evaluated checkpoint, shown whatever the per-class report,
    telling how many objects the detector finds at all, whatever the score of the detections: one row for all the
    objects then one per size, with its objects and the recall of all the detections at an intersection over union of
    0.5, 0.75 and averaged over 0.5 to 0.95, averaged over the classes (the recalls do not depend on the method).

    Args:
        results (Dict[str, Dict[Any, Any]]): The results of evaluate_detections, keyed by method.
        title_prefix (str): The prefix of the title ("Box", "Mask"), empty for the object detection. Defaults to "".
        area_name (str): The area measuring the objects, "box", or "mask" for the mask recalls. Defaults to "box".
        resolution_name (str): The resolution of the pixels measuring the objects, "network input", or "original
            image" for the instance segmentation evaluated at the original resolution. Defaults to "network input".

    Returns:
        EvaluationTable: The table of the mean average recalls.
    """
    # All the objects, then every object size, a size without object having no recall
    area_results = results["area"]
    group_results = [("All", area_results)] + [(size_name, area_results["sizes"][object_size])
                                               for object_size, size_name in OBJECT_SIZE_NAMES.items()]
    rows = [[group_name, sum(group_result["class_ground_truths"].values()), group_result["mAR@0.5"],
             group_result["mAR@0.75"], group_result["mAR@[0.5:0.95]"]] for group_name, group_result in group_results]
    return {"title": add_title_prefix(title="Mean Average Recall", title_prefix=title_prefix),
            "columns": ["Objects Of", "Objects", "mAR@0.5", "mAR@0.75", "mAR@[0.5:0.95]"], "rows": rows,
            "sort_column": None, "per_class": False,
            "legend": f"Objects found by any detection kept for the evaluation, averaged over the classes. COCO sizes "
                      f"by {area_name} area in {resolution_name} pixels. — : no object of the size."}


def build_class_average_precision_rows(results: Dict[str, Dict[Any, Any]], class_names: Dict[int, str],
                                       column_prefix: str = "") -> List[Dict[str, Any]]:
    """
    Gathers the average precisions of every foreground class of an evaluated checkpoint with the area method, in class
    index order, for the per-class metrics file and the per-class table: its ground truth boxes and its average
    precision at 0.5, 0.75 and averaged over 0.5 to 0.95 (None for a class without ground truth, which has no recall),
    its average precision at 0.5 and averaged over 0.5 to 0.95 for the objects of every size (None without object of
    the size), then, for results evaluated with a target precision, the score threshold reaching it and the recall it
    keeps (None when no threshold reaches it). The threshold columns are not prefixed: the instance segmentation
    computes them on the masks only.

    Args:
        results (Dict[str, Dict[Any, Any]]): The results of evaluate_detections, keyed by method, with the
            'class_score_thresholds' when they were evaluated with a target precision.
        class_names (Dict[int, str]): The name of every class index.
        column_prefix (str): The prefix of the average precision columns ("mask_" for the mask average precisions),
            empty for the object detection. Defaults to "".

    Returns:
        List[Dict[str, Any]]: One row per foreground class, keyed by the columns of class_metrics.csv: 'class_index',
            'class', 'ground_truths', the prefixed 'average_precision_50', 'average_precision_75' and
            'average_precision_50_95', the prefixed 'average_precision_50_<size>' and 'average_precision_50_95_<size>'
            of every object size, and with the thresholds 'threshold_at_target_precision' and
            'recall_at_target_precision'.
    """
    area_results = results["area"]
    class_rows = []
    for class_index in sorted(area_results["class_ground_truths"]):
        class_rows.append({"class_index": class_index, "class": class_names[class_index],
                           "ground_truths": area_results["class_ground_truths"][class_index]})

        # Average precisions of the class, undefined without ground truth, then those of every object size, undefined
        # without object of the size
        column_sources = [("average_precision_50", "class_AP@0.5", area_results),
                          ("average_precision_75", "class_AP@0.75", area_results),
                          ("average_precision_50_95", "class_AP@[0.5:0.95]", area_results)]
        for object_size in OBJECT_SIZE_NAMES:
            size_results = area_results["sizes"][object_size]
            column_sources += [(f"average_precision_50_{object_size}", "class_AP@0.5", size_results),
                               (f"average_precision_50_95_{object_size}", "class_AP@[0.5:0.95]", size_results)]
        for column_name, result_key, source_results in column_sources:
            class_average_precisions = source_results[result_key]
            class_rows[-1][f"{column_prefix}{column_name}"] = (class_average_precisions[class_index]
                                                               if class_index in class_average_precisions else None)

        # Score threshold of the class reaching the target precision, with the recall it keeps
        if "class_score_thresholds" in results:
            class_threshold: Optional[Tuple[float, float]] = results["class_score_thresholds"][class_index]
            class_rows[-1]["threshold_at_target_precision"] = None if class_threshold is None else class_threshold[0]
            class_rows[-1]["recall_at_target_precision"] = None if class_threshold is None else class_threshold[1]
    return class_rows


def build_class_average_precision_table(class_rows: List[Dict[str, Any]], title_prefix: str = "",
                                        column_prefix: str = "",
                                        target_precision: Optional[float] = None) -> EvaluationTable:
    """
    Builds the per-class table of the report of an evaluated checkpoint, shown with the per-class report and sorted by
    ascending AP@0.5 so that the worst classes come first: the ground truth boxes of every class and its average
    precisions with the area method, a class without ground truth being listed last, and with a target precision the
    score threshold reaching it with the recall it keeps.

    Args:
        class_rows (List[Dict[str, Any]]): The average precisions of every class (build_class_average_precision_rows).
        title_prefix (str): The prefix of the title ("Box", "Mask"), empty for the object detection. Defaults to "".
        column_prefix (str): The prefix of the average precision columns of the rows to show. Defaults to "".
        target_precision (Optional[float]): The precision targeted by the thresholds of the rows, None for a table
            without thresholds (the box table of the instance segmentation). Defaults to None.

    Returns:
        EvaluationTable: The per-class table.
    """
    columns = ["Class", "Ground\nTruths", CLASS_TABLE_SORT_COLUMN, "AP@0.75", "AP@[0.5:0.95]"]
    rows = [[class_row["class"], class_row["ground_truths"], class_row[f"{column_prefix}average_precision_50"],
             class_row[f"{column_prefix}average_precision_75"], class_row[f"{column_prefix}average_precision_50_95"]]
            for class_row in class_rows]
    legend = "Worst AP@0.5 first, ties in class order. — : no ground truth box of the class."

    # Threshold of every class reaching the target precision, with the recall it keeps
    if target_precision is not None:
        columns += [f"Threshold\n@ P{target_precision:g}", f"Recall\n@ P{target_precision:g}"]
        for row, class_row in zip(rows, class_rows, strict=True):
            row += [class_row["threshold_at_target_precision"], class_row["recall_at_target_precision"]]
        legend = (f"Worst AP@0.5 first, ties in class order. — : no ground truth box of the class, or the precision "
                  f"{target_precision:g} is never reached.")
    return {"title": add_title_prefix(title=CLASS_TABLE_TITLE, title_prefix=title_prefix), "columns": columns,
            "rows": rows, "sort_column": CLASS_TABLE_SORT_COLUMN, "per_class": True, "legend": legend}
