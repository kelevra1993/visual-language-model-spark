"""
Tests of vision_training/evaluation/classification/class_metrics_report.py: the per-class rows of class_metrics.csv in
class index order, with the recall interval and the threshold reaching the target precision of every class, and the
rates of the per-class tables of the classification report listed as percent columns.
"""

import pytest

from vision_training.evaluation.classification.class_metrics_report import (build_class_metrics_rows,
                                                                             build_class_metrics_table,
                                                                             build_class_thresholds_table)
from vision_training.evaluation.recall_intervals import CLOPPER_PEARSON_DESCRIPTION, build_recall_interval_table

# Metrics of 3 classes, the last one without any evaluated sample and so without recall interval
CLASS_NAMES = ["circle", "square", "hexagon"]
METRICS = {"support": [3, 2, 0], "precision": [1.0, 0.5, 0.0], "recall": [2 / 3, 1.0, 0.0],
           "recall_lower": [0.09, 0.16, None], "recall_upper": [0.99, 1.0, None], "f1": [0.8, 2 / 3, 0.0]}
CLASS_THRESHOLDS = [[(0.91, 0.5), None, None]]


def test_class_metrics_rows() -> None:
    """
    Verifies the per-class rows of class_metrics.csv: one per class in class index order, with its samples, precision,
    recall with the bounds of its interval, F1 and the threshold reaching the target precision with its recall (None
    when unreachable).
    """
    class_rows = build_class_metrics_rows(class_names=CLASS_NAMES, metrics=METRICS, class_thresholds=CLASS_THRESHOLDS,
                                          target_precisions=[0.99])
    assert [class_row["class"] for class_row in class_rows] == CLASS_NAMES
    assert class_rows[0] == {"class_index": 0, "class": "circle", "samples": 3, "precision": 1.0,
                             "recall": pytest.approx(expected=2 / 3), "recall_lower": 0.09, "recall_upper": 0.99,
                             "f1": 0.8, "threshold_at_target_precision": 0.91, "recall_at_target_precision": 0.5}
    assert class_rows[2]["recall_lower"] is None and class_rows[2]["recall_upper"] is None
    assert class_rows[1]["threshold_at_target_precision"] is None
    assert class_rows[1]["recall_at_target_precision"] is None


def test_classification_tables_list_their_rates_as_percent_columns() -> None:
    """
    Verifies the per-class tables of the classification report: every rate column (precision, recall, F1, recall at
    the target precision, recall with its bounds) is a percent column and the thresholds are not, the target precisions
    being written "99 %" in the headers and legends, while a table built without in_percent, which no trainer does any
    more, keeps its columns between 0 and 1.
    """
    class_rows = build_class_metrics_rows(class_names=CLASS_NAMES, metrics=METRICS,
                                          class_thresholds=CLASS_THRESHOLDS * 2, target_precisions=[0.99, 0.955])
    class_table = build_class_metrics_table(class_rows=class_rows, target_precision=0.99, support_column="Samples",
                                            in_percent=True)
    assert class_table["percent_columns"] == ["Precision", "Recall", "F1", "Recall\n@ P99 %"]
    assert "Threshold\n@ P99 %" in class_table["columns"] and "the precision 99 % is never" in class_table["legend"]

    # Thresholds of every target precision: the recalls in percent, the thresholds between 0 and 1
    thresholds_table = build_class_thresholds_table(class_rows=class_rows, target_precisions=[0.99, 0.955])
    assert thresholds_table["columns"] == ["Class", "Threshold\n@ P99 %", "Recall\n@ P99 %", "Threshold\n@ P95.5 %",
                                           "Recall\n@ P95.5 %"]
    assert thresholds_table["percent_columns"] == ["Recall\n@ P99 %", "Recall\n@ P95.5 %"]

    # Recall with the bounds of its interval in percent, the confidence level written "95 %"
    recall_table = build_recall_interval_table(class_rows=class_rows, support_key="samples", support_column="Samples",
                                               recall_columns=[("recall", "Recall")],
                                               interval_description=CLOPPER_PEARSON_DESCRIPTION, in_percent=True)
    assert recall_table["columns"] == ["Class", "Samples", "Recall", "95 %\nLower", "95 %\nUpper"]
    assert recall_table["percent_columns"] == ["Recall", "95 %\nLower", "95 %\nUpper"]

    # Tables built without in_percent, which every trainer now passes, keep their rates between 0 and 1 and "95%"
    neighbour_table = build_class_metrics_table(class_rows=class_rows, target_precision=None, support_column="Samples")
    kept_recall_table = build_recall_interval_table(class_rows=class_rows, support_key="samples",
                                                    support_column="Samples", recall_columns=[("recall", "Recall")],
                                                    interval_description=CLOPPER_PEARSON_DESCRIPTION)
    assert neighbour_table["percent_columns"] == [] and kept_recall_table["percent_columns"] == []
    assert kept_recall_table["columns"][3] == "95%\nLower"
