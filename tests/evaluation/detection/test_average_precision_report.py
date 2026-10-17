"""
Tests of vision_training/evaluation/detection/average_precision_report.py on the results of evaluate_detections: the
table of the average precision methods, the table of the object sizes, the table of the mean average recalls with the
resolution of the object sizes, the per-class rows of class_metrics.csv (a class
without ground truth or without object of a size having no average precision), the per-class table sorted by AP@0.5,
the class thresholds of both, and the title and column prefixes of the box and mask tables of the instance
segmentation.
"""

import pytest
import torch

from typing import Any, Dict, List

from vision_training.evaluation.detection.average_precision_report import (build_average_recall_table,
                                                                           build_class_average_precision_rows,
                                                                           build_class_average_precision_table,
                                                                           build_mean_average_precision_table,
                                                                           build_object_size_table)
from vision_training.evaluation.detection.mean_average_precision import evaluate_detections


def get_two_class_results() -> Dict[str, Dict[str, Any]]:
    """
    Evaluates one image whose class 1 box is found exactly and whose class 2 box is missed, class 3 having no ground
    truth: AP 1 for class 1, 0 for class 2, none for class 3.

    Returns:
        Dict[str, Dict[str, Any]]: The results of evaluate_detections.
    """
    ground_truths = [{"boxes": torch.tensor(data=[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
                      "labels": torch.tensor(data=[1, 2])}]
    detections = [{"boxes": torch.tensor(data=[[0.0, 0.0, 10.0, 10.0]]), "scores": torch.tensor(data=[0.9]),
                   "labels": torch.tensor(data=[1])}]
    return evaluate_detections(detections=detections, ground_truths=ground_truths, number_classes=4)


def test_mean_average_precision_table() -> None:
    """
    Verifies the table of the methods, always shown: one row per method with its four mean average precisions.
    """
    table = build_mean_average_precision_table(results=get_two_class_results())
    assert [row[0] for row in table["rows"]] == ["Area", "11 Point", "101 Point"]
    assert table["columns"] == ["Method", "mAP@0.5", "mAP@0.75", "mAP@0.95", "mAP@[0.5:0.95]"]
    assert table["rows"][0][1:] == pytest.approx(expected=[0.5, 0.5, 0.5, 0.5])
    assert table["per_class"] is False and table["sort_column"] is None


def test_object_size_table() -> None:
    """
    Verifies the table of the object sizes, always shown: the two 10 x 10 objects are small (one found, one missed,
    mAP 0.5), the medium and large rows have no object and no mAP, and the prefix and the area name follow the mask
    average precisions of the instance segmentation.
    """
    table = build_object_size_table(results=get_two_class_results())
    assert table["columns"] == ["Object Size", "Objects", "mAP@0.5", "mAP@0.75", "mAP@0.95", "mAP@[0.5:0.95]"]
    assert table["rows"][0] == ["Small (< 32²)", 2, 0.5, 0.5, 0.5, 0.5]
    assert table["rows"][1][1:] == [0, None, None, None, None] and table["rows"][2][1:] == [0, None, None, None, None]
    assert table["per_class"] is False and "by box area" in table["legend"]

    mask_table = build_object_size_table(results=get_two_class_results(), title_prefix="Mask", area_name="mask")
    assert mask_table["title"] == "Mask Mean Average Precision By Object Size (Area)"
    assert "by mask area" in mask_table["legend"]


def test_average_recall_table_and_resolution_of_the_sizes() -> None:
    """
    Verifies the table of the mean average recalls, always shown: all the objects then every size, the found class 1
    object and the missed class 2 object giving a recall of 0.5 at every threshold, the medium and large rows having no
    object and no recall, and the prefix, area and resolution of the mask recalls evaluated at the original resolution,
    also named by the legend of the object size table.
    """
    table = build_average_recall_table(results=get_two_class_results())
    assert table["title"] == "Mean Average Recall"
    assert table["columns"] == ["Objects Of", "Objects", "mAR@0.5", "mAR@0.75", "mAR@0.95", "mAR@[0.5:0.95]"]
    assert table["rows"][0] == ["All", 2, 0.5, 0.5, 0.5, 0.5]
    assert table["rows"][1] == ["Small (< 32²)", 2, 0.5, 0.5, 0.5, 0.5]
    assert table["rows"][2][1:] == [0, None, None, None, None] and table["rows"][3][1:] == [0, None, None, None, None]
    assert table["per_class"] is False and "by box area in network input pixels" in table["legend"]

    # Mask recalls and object sizes measured at the original resolution of the test images
    mask_table = build_average_recall_table(results=get_two_class_results(), title_prefix="Mask", area_name="mask",
                                            resolution_name="original image")
    assert mask_table["title"] == "Mask Mean Average Recall"
    assert "by mask area in original image pixels" in mask_table["legend"]
    size_table = build_object_size_table(results=get_two_class_results(), title_prefix="Mask", area_name="mask",
                                         resolution_name="original image")
    assert "by mask area in original image pixels" in size_table["legend"]


def test_class_average_precision_rows_and_table() -> None:
    """
    Verifies the per-class rows in class index order (the ground truth boxes and the four average precisions of the
    area method, None for the class without ground truth) and the per-class table sorted by AP@0.5.
    """
    class_names = {0: "background", 1: "square", 2: "circle", 3: "star"}
    class_rows = build_class_average_precision_rows(results=get_two_class_results(), class_names=class_names)

    # The 10 x 10 boxes are small objects, the medium and large ones have no average precision
    empty_size_columns = {"average_precision_50_medium": None, "average_precision_50_95_medium": None,
                          "average_precision_50_large": None, "average_precision_50_95_large": None}
    expected_rows: List[Dict[str, Any]] = [
        {"class_index": 1, "class": "square", "ground_truths": 1, "average_precision_50": 1.0,
         "average_precision_75": 1.0, "average_precision_95": 1.0, "average_precision_50_95": 1.0,
         "average_precision_50_small": 1.0,
         "average_precision_50_95_small": 1.0, **empty_size_columns},
        {"class_index": 2, "class": "circle", "ground_truths": 1, "average_precision_50": 0.0,
         "average_precision_75": 0.0, "average_precision_95": 0.0, "average_precision_50_95": 0.0,
         "average_precision_50_small": 0.0,
         "average_precision_50_95_small": 0.0, **empty_size_columns},
        {"class_index": 3, "class": "star", "ground_truths": 0, "average_precision_50": None,
         "average_precision_75": None, "average_precision_95": None, "average_precision_50_95": None,
         "average_precision_50_small": None,
         "average_precision_50_95_small": None, **empty_size_columns}]
    assert class_rows == expected_rows

    table = build_class_average_precision_table(class_rows=class_rows)
    assert table["columns"] == ["Class", "Ground\nTruths", "AP@0.5", "AP@0.75", "AP@0.95", "AP@[0.5:0.95]"]
    assert table["sort_column"] == "AP@0.5" and table["per_class"] is True
    assert table["rows"][2] == ["star", 0, None, None, None, None]


def test_class_thresholds_join_the_rows_and_the_table() -> None:
    """
    Verifies that the results evaluated with a target precision add the threshold of every class and the recall it
    keeps, unprefixed, at the end of the class rows (None without threshold), and that the per-class table shows them
    in two columns named after the target precision.
    """
    ground_truths = [{"boxes": torch.tensor(data=[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
                      "labels": torch.tensor(data=[1, 2])}]
    detections = [{"boxes": torch.tensor(data=[[0.0, 0.0, 10.0, 10.0]]), "scores": torch.tensor(data=[0.9]),
                   "labels": torch.tensor(data=[1])}]
    results = evaluate_detections(detections=detections, ground_truths=ground_truths, number_classes=4,
                                  target_precision=0.99)
    class_rows = build_class_average_precision_rows(results=results, class_names={0: "background", 1: "square",
                                                                                  2: "circle", 3: "star"},
                                                    column_prefix="mask_")
    assert list(class_rows[0])[-2:] == ["threshold_at_target_precision", "recall_at_target_precision"]
    assert class_rows[0]["threshold_at_target_precision"] == pytest.approx(expected=0.9)
    assert class_rows[0]["recall_at_target_precision"] == pytest.approx(expected=1.0)
    assert class_rows[1]["threshold_at_target_precision"] is None
    assert class_rows[1]["recall_at_target_precision"] is None

    table = build_class_average_precision_table(class_rows=class_rows, column_prefix="mask_", target_precision=0.99)
    assert table["columns"][-2:] == ["Threshold\n@ P0.99", "Recall\n@ P0.99"]
    assert table["rows"][0][-2:] == pytest.approx(expected=[0.9, 1.0])
    assert "precision 0.99 is never reached" in table["legend"]


def test_box_and_mask_tables_are_told_apart_by_their_prefixes() -> None:
    """
    Verifies that the title prefixes name the box and mask tables, that the legend can name another mAP comparing the
    checkpoints, and that the mask rows of the column prefix merge with the box rows of the same classes and feed the
    mask per-class table, the object detection titles and columns staying unprefixed.
    """
    results = get_two_class_results()
    class_names = {0: "background", 1: "square", 2: "circle", 3: "star"}

    mask_table = build_mean_average_precision_table(results=results, title_prefix="Mask")
    box_table = build_mean_average_precision_table(results=results, title_prefix="Box", legend="Mask mAP@0.5 instead.")
    assert mask_table["title"] == "Mask Mean Average Precision" and box_table["title"] == "Box Mean Average Precision"
    assert box_table["legend"] == "Mask mAP@0.5 instead." and mask_table["legend"].startswith("The mAP@0.5")

    # Box and mask rows of the same classes merged into one row per class
    box_rows = build_class_average_precision_rows(results=results, class_names=class_names)
    mask_rows = build_class_average_precision_rows(results=results, class_names=class_names, column_prefix="mask_")
    merged_rows = [{**box_row, **mask_row} for box_row, mask_row in zip(box_rows, mask_rows)]
    average_precision_columns = ["average_precision_50", "average_precision_75", "average_precision_95",
                                 "average_precision_50_95"] + [
        f"average_precision_{threshold}_{object_size}" for object_size in ["small", "medium", "large"]
        for threshold in ["50", "50_95"]]
    assert list(merged_rows[0]) == ["class_index", "class", "ground_truths"] + average_precision_columns + [
        f"mask_{column_name}" for column_name in average_precision_columns]

    mask_class_table = build_class_average_precision_table(class_rows=merged_rows, title_prefix="Mask",
                                                           column_prefix="mask_")
    assert mask_class_table["title"] == "Mask Per Class Average Precision (Area)"
    assert mask_class_table["rows"][0] == ["square", 1, 1.0, 1.0, 1.0, 1.0]
