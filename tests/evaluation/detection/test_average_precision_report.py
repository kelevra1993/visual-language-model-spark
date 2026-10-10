"""
Tests of vision_training/evaluation/detection/average_precision_report.py on the results of evaluate_detections: the
table of the average precision methods, the per-class rows of class_metrics.csv (a class without ground truth having
no average precision), the per-class table sorted by AP@0.5, and the title and column prefixes of the box and mask
tables of the instance segmentation.
"""

import pytest
import torch

from typing import Any, Dict, List

from vision_training.evaluation.detection.average_precision_report import (build_class_average_precision_rows,
                                                                           build_class_average_precision_table,
                                                                           build_mean_average_precision_table)
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
    Verifies the table of the methods, always shown: one row per method with its three mean average precisions.
    """
    table = build_mean_average_precision_table(results=get_two_class_results())
    assert [row[0] for row in table["rows"]] == ["Area", "11 Point", "101 Point"]
    assert table["columns"] == ["Method", "mAP@0.5", "mAP@0.75", "mAP@[0.5:0.95]"]
    assert table["rows"][0][1:] == pytest.approx(expected=[0.5, 0.5, 0.5])
    assert table["per_class"] is False and table["sort_column"] is None


def test_class_average_precision_rows_and_table() -> None:
    """
    Verifies the per-class rows in class index order (the ground truth boxes and the three average precisions of the
    area method, None for the class without ground truth) and the per-class table sorted by AP@0.5.
    """
    class_names = {0: "background", 1: "square", 2: "circle", 3: "star"}
    class_rows = build_class_average_precision_rows(results=get_two_class_results(), class_names=class_names)
    expected_rows: List[Dict[str, Any]] = [
        {"class_index": 1, "class": "square", "ground_truths": 1, "average_precision_50": 1.0,
         "average_precision_75": 1.0, "average_precision_50_95": 1.0},
        {"class_index": 2, "class": "circle", "ground_truths": 1, "average_precision_50": 0.0,
         "average_precision_75": 0.0, "average_precision_50_95": 0.0},
        {"class_index": 3, "class": "star", "ground_truths": 0, "average_precision_50": None,
         "average_precision_75": None, "average_precision_50_95": None}]
    assert class_rows == expected_rows

    table = build_class_average_precision_table(class_rows=class_rows)
    assert table["columns"] == ["Class", "Ground\nTruths", "AP@0.5", "AP@0.75", "AP@[0.5:0.95]"]
    assert table["sort_column"] == "AP@0.5" and table["per_class"] is True
    assert table["rows"][2] == ["star", 0, None, None, None]


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
    assert list(merged_rows[0]) == ["class_index", "class", "ground_truths", "average_precision_50",
                                    "average_precision_75", "average_precision_50_95", "mask_average_precision_50",
                                    "mask_average_precision_75", "mask_average_precision_50_95"]

    mask_class_table = build_class_average_precision_table(class_rows=merged_rows, title_prefix="Mask",
                                                           column_prefix="mask_")
    assert mask_class_table["title"] == "Mask Per Class Average Precision (Area)"
    assert mask_class_table["rows"][0] == ["square", 1, 1.0, 1.0, 1.0]
