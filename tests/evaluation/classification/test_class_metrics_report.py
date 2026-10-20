"""
Tests of vision_training/evaluation/classification/class_metrics_report.py: the per-class rows of class_metrics.csv in
class index order, the per-class table of the report (the target precision in its columns, no precision, recall nor F1
for a class without samples, sorted by F1), the per-class TensorBoard scalars, and the ROC AUC and average precision
of the per-class report in the rows, table and scalars.
"""

import pytest

from vision_training.evaluation.classification.class_metrics_report import (build_class_metric_scalars,
                                                                            build_class_metrics_rows,
                                                                            build_class_metrics_table)

# Metrics of 3 classes, the last one without any evaluated sample
CLASS_NAMES = ["circle", "square", "hexagon"]
METRICS = {"support": [3, 2, 0], "precision": [1.0, 0.5, 0.0], "recall": [2 / 3, 1.0, 0.0], "f1": [0.8, 2 / 3, 0.0]}
CLASS_THRESHOLDS = [(0.91, 0.5), None, None]


def test_class_metrics_rows() -> None:
    """
    Verifies the per-class rows of class_metrics.csv: one per class in class index order, with its samples, precision,
    recall, F1 and the threshold reaching the target precision with its recall (None when unreachable).
    """
    class_rows = build_class_metrics_rows(class_names=CLASS_NAMES, metrics=METRICS, class_thresholds=CLASS_THRESHOLDS)
    assert [class_row["class"] for class_row in class_rows] == CLASS_NAMES
    assert class_rows[0] == {"class_index": 0, "class": "circle", "samples": 3, "precision": 1.0,
                             "recall": pytest.approx(expected=2 / 3), "f1": 0.8, "threshold_at_target_precision": 0.91,
                             "recall_at_target_precision": 0.5}
    assert class_rows[1]["threshold_at_target_precision"] is None
    assert class_rows[1]["recall_at_target_precision"] is None


def test_class_metrics_table() -> None:
    """
    Verifies the per-class table of the report: the support column named by the task, the target precision in the
    threshold columns, the rows sorted by F1, and no precision, recall nor F1 for the class without samples.
    """
    class_rows = build_class_metrics_rows(class_names=CLASS_NAMES, metrics=METRICS, class_thresholds=CLASS_THRESHOLDS)
    table = build_class_metrics_table(class_rows=class_rows, target_precision=0.99, support_column="Frames")
    assert table["columns"] == ["Class", "Frames", "Precision", "Recall", "F1", "Threshold\n@ P0.99", "Recall\n@ P0.99"]
    assert table["sort_column"] == "F1" and table["per_class"] is True
    assert table["rows"][2] == ["hexagon", 0, None, None, None, None, None]
    assert "no frames of the class" in table["legend"]


def test_class_metric_scalars() -> None:
    """
    Verifies the per-class TensorBoard scalars: the precision, recall and F1 of every class, under their tags.
    """
    class_scalars = build_class_metric_scalars(class_names=CLASS_NAMES[:2], metrics=METRICS)
    assert list(class_scalars) == ["Precision/circle", "Recall/circle", "F1/circle", "Precision/square",
                                   "Recall/square", "F1/square"]
    assert class_scalars["F1/square"] == pytest.approx(expected=2 / 3)


def test_curve_metrics_of_the_per_class_report() -> None:
    """
    Verifies the ROC AUC and average precision of the per-class report: their columns of class_metrics.csv after the
    F1 and before the thresholds, their columns and legend in the table, and their scalars for the classes defining
    them, a class without positive having none.
    """
    class_curves = [{"roc_auc": 0.9, "average_precision": 0.8}, {"roc_auc": None, "average_precision": 1.0},
                    {"roc_auc": None, "average_precision": None}]
    class_rows = build_class_metrics_rows(class_names=CLASS_NAMES, metrics=METRICS, class_thresholds=CLASS_THRESHOLDS,
                                          class_curves=class_curves)
    assert list(class_rows[0]) == ["class_index", "class", "samples", "precision", "recall", "f1", "roc_auc",
                                   "average_precision", "threshold_at_target_precision", "recall_at_target_precision"]

    # Table: both columns between the F1 and the thresholds, "—" for the undefined ones
    table = build_class_metrics_table(class_rows=class_rows, target_precision=0.99, support_column="Samples")
    assert table["columns"] == ["Class", "Samples", "Precision", "Recall", "F1", "ROC AUC", "Average\nPrecision",
                                "Threshold\n@ P0.99", "Recall\n@ P0.99"]
    assert table["rows"][1][5:7] == [None, 1.0] and "no negative for the ROC AUC" in table["legend"]

    # Scalars of the classes defining every metric
    class_scalars = build_class_metric_scalars(class_names=CLASS_NAMES, metrics=METRICS, class_curves=class_curves)
    assert [tag for tag in class_scalars if tag.startswith(("ROC AUC/", "Average Precision/"))] == [
        "ROC AUC/circle", "Average Precision/circle", "Average Precision/square"]
