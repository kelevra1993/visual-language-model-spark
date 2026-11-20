"""
Tests of vision_training/evaluation/detection/average_precision_report.py on the results of evaluate_detections: the
per-class rows of class_metrics.csv (a class without ground truth or without object of a size having no average
precision), the per-class table sorted by AP@0.5, the title and column prefixes of the box and mask tables of the
instance segmentation, and the rates of every table written as percentages in the report while the rows and scalars
stay between 0 and 1.
"""

import pytest
import torch

from typing import Any, Dict, List, Tuple

from vision_training.core.contracts import EvaluationResults
from vision_training.evaluation.detection.average_precision_report import (build_class_average_precision_rows,
                                                                           build_class_average_precision_table,
                                                                           build_mean_average_precision_table,
                                                                           summarise_average_precisions)
from vision_training.evaluation.detection.mean_average_precision import evaluate_detections
from vision_training.evaluation.report_formatting import REPORT_WIDTH, build_report_renderables, render_report_text

# Class names of the evaluated image, the background first
CLASS_NAMES = {0: "background", 1: "square", 2: "circle", 3: "star"}


def build_two_class_image() -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
    """
    Builds one evaluated image whose class 1 box is found exactly, with a score of 0.9, and whose class 2 box is
    missed, class 3 having no ground truth.

    Returns:
        Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]: The detections and the ground truths.
    """
    ground_truths = [{"boxes": torch.tensor(data=[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
                      "labels": torch.tensor(data=[1, 2])}]
    detections = [{"boxes": torch.tensor(data=[[0.0, 0.0, 10.0, 10.0]]), "scores": torch.tensor(data=[0.9]),
                   "labels": torch.tensor(data=[1])}]
    return detections, ground_truths


def get_two_class_results() -> Dict[str, Dict[str, Any]]:
    """
    Evaluates the image of build_two_class_image: AP 1 for class 1, 0 for class 2, none for class 3.

    Returns:
        Dict[str, Dict[str, Any]]: The results of evaluate_detections.
    """
    detections, ground_truths = build_two_class_image()
    return evaluate_detections(detections=detections, ground_truths=ground_truths, number_classes=4)


def test_class_average_precision_rows_and_table() -> None:
    """
    Verifies the per-class rows in class index order (the ground truth boxes, the four average precisions of the
    area method and the recalls at 0.5 and 0.75 with their exact interval, None for the class without ground truth)
    and the per-class table sorted by AP@0.5.
    """
    class_names = {0: "background", 1: "square", 2: "circle", 3: "star"}
    class_rows = build_class_average_precision_rows(results=get_two_class_results(), class_names=class_names)

    # The 10 x 10 boxes are small objects, the medium and large ones have no average precision. The single ground
    # truth found gives the exact interval [0.025, 1] of a recall of 1 of 1, the one missed [0, 0.975]
    empty_size_columns = {"average_precision_50_medium": None, "average_precision_50_95_medium": None,
                          "average_precision_50_large": None, "average_precision_50_95_large": None}
    found_recalls = {f"recall_{threshold}{bound}": value for threshold in ["50", "75"]
                     for bound, value in [("", 1.0), ("_lower", pytest.approx(expected=0.025)), ("_upper", 1.0)]}
    missed_recalls = {f"recall_{threshold}{bound}": value for threshold in ["50", "75"]
                      for bound, value in [("", 0.0), ("_lower", 0.0), ("_upper", pytest.approx(expected=0.975))]}
    absent_recalls = {f"recall_{threshold}{bound}": None for threshold in ["50", "75"]
                      for bound in ["", "_lower", "_upper"]}

    # The square found, the circle missed and the star without ground truth, in class index order
    expected_rows: List[Dict[str, Any]] = [
        {"class_index": 1, "class": "square", "ground_truths": 1, "average_precision_50": 1.0,
         "average_precision_75": 1.0, "average_precision_95": 1.0, "average_precision_50_95": 1.0,
         "average_precision_50_small": 1.0,
         "average_precision_50_95_small": 1.0, **empty_size_columns, **found_recalls},
        {"class_index": 2, "class": "circle", "ground_truths": 1, "average_precision_50": 0.0,
         "average_precision_75": 0.0, "average_precision_95": 0.0, "average_precision_50_95": 0.0,
         "average_precision_50_small": 0.0,
         "average_precision_50_95_small": 0.0, **empty_size_columns, **missed_recalls},
        {"class_index": 3, "class": "star", "ground_truths": 0, "average_precision_50": None,
         "average_precision_75": None, "average_precision_95": None, "average_precision_50_95": None,
         "average_precision_50_small": None,
         "average_precision_50_95_small": None, **empty_size_columns, **absent_recalls}]
    assert class_rows == expected_rows

    # The table sorted by AP@0.5 puts the star, without average precision, last
    table = build_class_average_precision_table(class_rows=class_rows)
    assert table["columns"] == ["Class", "Ground\nTruths", "AP@0.5", "AP@0.75", "AP@0.95", "AP@[0.5:0.95]"]
    assert table["sort_column"] == "AP@0.5" and table["per_class"] is True
    assert table["rows"][2] == ["star", 0, None, None, None, None]


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
        for threshold in ["50", "50_95"]] + [
        f"recall_{threshold}{bound}" for threshold in ["50", "75"] for bound in ["", "_lower", "_upper"]]
    assert list(merged_rows[0]) == ["class_index", "class", "ground_truths"] + average_precision_columns + [
        f"mask_{column_name}" for column_name in average_precision_columns]

    mask_class_table = build_class_average_precision_table(class_rows=merged_rows, title_prefix="Mask",
                                                           column_prefix="mask_")
    assert mask_class_table["title"] == "Mask Per Class Average Precision (Area)"
    assert mask_class_table["rows"][0] == ["square", 1, 1.0, 1.0, 1.0, 1.0]


def test_tables_write_the_rates_as_percentages() -> None:
    """
    Verifies that every table of the average precision report lists its rates as percent columns (the mean average
    precisions and recalls, the average precisions of every class, the recall at the target precision, the recalls and
    their interval bounds), never the object counts, the ground truths nor the score threshold, that the target
    precision is written "99 %" and the confidence level "95 %", and that the rendered report writes "100.00 %" while
    the class rows and the scalars stay between 0 and 1.
    """
    detections, ground_truths = build_two_class_image()
    summary = summarise_average_precisions(detections=detections, ground_truths=ground_truths, number_classes=4,
                                           class_names=CLASS_NAMES, target_precision=0.99)
    tables = summary["tables"]

    # Every rate column in percent, the method, the object counts, the ground truths and the threshold kept as they are
    assert tables["methods"]["percent_columns"] == ["mAP@0.5", "mAP@0.75", "mAP@0.95", "mAP@[0.5:0.95]"]
    assert tables["sizes"]["percent_columns"] == ["mAP@0.5", "mAP@0.75", "mAP@0.95", "mAP@[0.5:0.95]"]
    assert tables["recalls"]["percent_columns"] == ["mAR@0.5", "mAR@0.75", "mAR@0.95", "mAR@[0.5:0.95]"]
    assert tables["classes"]["columns"][-2:] == ["Threshold\n@ P99 %", "Recall\n@ P99 %"]
    assert tables["classes"]["percent_columns"] == ["AP@0.5", "AP@0.75", "AP@0.95", "AP@[0.5:0.95]",
                                                    "Recall\n@ P99 %"]
    assert "the precision 99 % is never reached" in tables["classes"]["legend"]
    assert tables["class_recalls"]["percent_columns"] == tables["class_recalls"]["columns"][2:]
    assert "95 %\nLower" in tables["class_recalls"]["columns"]

    # The rendered report in percent, the threshold of the found class with 4 decimals and its count as an integer
    results: EvaluationResults = {"evaluated_samples": "1 Test Images", "metrics": {}, "notes": [],
                                  "tables": list(tables.values()), "scalars": summary["scalars"], "class_scalars": {},
                                  "images": {}, "sample_images": {}, "files": {}, "headline_metric": "mAP@0.5/area",
                                  "percent_metrics": ["mAP@0.5/area"]}
    lines = render_report_text(renderables=build_report_renderables(
        results=results, iteration=4, timestamp="2026-11-15 12:00", per_class_report=True)).splitlines()
    assert next(line for line in lines if line.startswith("   Area")).split()[1:] == ["50.00", "%"] * 4
    square_rows = [line.split()[1:] for line in lines if line.startswith("   square")]
    assert square_rows == [["1"] + ["100.00", "%"] * 4 + ["0.9000", "100.00", "%"],
                           ["1", "100.00", "%", "2.50", "%", "100.00", "%", "100.00", "%", "2.50", "%", "100.00", "%"]]
    assert all(len(line) <= REPORT_WIDTH for line in lines)

    # The class rows of class_metrics.csv and the scalars stay between 0 and 1
    assert summary["class_rows"][0]["average_precision_50"] == 1.0
    assert summary["class_rows"][0]["recall_at_target_precision"] == 1.0
    assert summary["scalars"]["mAP@0.5/area"] == 0.5
