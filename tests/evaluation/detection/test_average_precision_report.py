"""
Tests of vision_training/evaluation/detection/average_precision_report.py on the results of evaluate_detections: the
table of the average precision methods, the per-class rows of class_metrics.csv (a class without ground truth having
no average precision) and the per-class table sorted by AP@0.5.
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
