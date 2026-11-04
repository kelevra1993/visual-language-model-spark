"""
Tests of vision_training/evaluation/classification/class_metrics_report.py: the per-class rows of class_metrics.csv in
class index order, with the recall interval and the threshold reaching the target precision of every class.
"""

import pytest

from vision_training.evaluation.classification.class_metrics_report import build_class_metrics_rows

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
