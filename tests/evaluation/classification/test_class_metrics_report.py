"""
Tests of vision_training/evaluation/classification/class_metrics_report.py: the per-class rows of class_metrics.csv in
class index order, the per-class table of the report (the target precision in its columns, no precision, recall nor F1
for a class without samples, sorted by F1), the per-class TensorBoard scalars, the ROC AUC and average precision of
the per-class report in the rows, table and scalars, and the threshold columns and table of several target
precisions.
"""

import pytest

from vision_training.evaluation.classification.class_metrics_report import (build_class_metric_scalars,
                                                                            build_class_metrics_rows,
                                                                            build_class_metrics_table,
                                                                            build_class_thresholds_table)

# Metrics of 3 classes, the last one without any evaluated sample
CLASS_NAMES = ["circle", "square", "hexagon"]
METRICS = {"support": [3, 2, 0], "precision": [1.0, 0.5, 0.0], "recall": [2 / 3, 1.0, 0.0], "f1": [0.8, 2 / 3, 0.0]}
CLASS_THRESHOLDS = [[(0.91, 0.5), None, None]]


def test_class_metrics_rows() -> None:
    """
    Verifies the per-class rows of class_metrics.csv: one per class in class index order, with its samples, precision,
    recall, F1 and the threshold reaching the target precision with its recall (None when unreachable).
    """
    class_rows = build_class_metrics_rows(class_names=CLASS_NAMES, metrics=METRICS, class_thresholds=CLASS_THRESHOLDS,
                                          target_precisions=[0.99])
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
    class_rows = build_class_metrics_rows(class_names=CLASS_NAMES, metrics=METRICS, class_thresholds=CLASS_THRESHOLDS,
                                          target_precisions=[0.99])
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
                                          target_precisions=[0.99], class_curves=class_curves)
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


def test_threshold_columns_of_several_target_precisions() -> None:
    """
    Verifies the thresholds of several target precisions: the main target precision (the first one) keeps the columns
    read by the tools ('threshold_at_target_precision'), every other one gets columns named by its value after them in
    class_metrics.csv, the per-class table keeps the main target precision only, and the thresholds table gives every
    target precision in class order, the thresholds written as on the image when the curves give them.
    """
    class_thresholds = [[(0.91, 0.5), None, None], [(0.42, 0.75), (0.3, 1.0), None]]
    class_rows = build_class_metrics_rows(class_names=CLASS_NAMES, metrics=METRICS, class_thresholds=class_thresholds,
                                          target_precisions=[0.99, 0.95])
    assert list(class_rows[0])[6:] == ["threshold_at_target_precision", "recall_at_target_precision",
                                       "threshold_at_precision_0.95", "recall_at_precision_0.95"]
    assert [class_rows[1][column_name] for column_name in list(class_rows[1])[6:]] == [None, None, 0.3, 1.0]

    # Per-class table: the main target precision only, so that it stays within the width of the report
    table = build_class_metrics_table(class_rows=class_rows, target_precision=0.99, support_column="Samples")
    assert table["columns"][5:] == ["Threshold\n@ P0.99", "Recall\n@ P0.99"]

    # Thresholds table: the columns of every target precision in its order, a dash for a precision never reached
    thresholds_table = build_class_thresholds_table(class_rows=class_rows, target_precisions=[0.99, 0.95])
    assert thresholds_table["columns"] == ["Class", "Threshold\n@ P0.99", "Recall\n@ P0.99", "Threshold\n@ P0.95",
                                           "Recall\n@ P0.95"]
    assert thresholds_table["rows"][0] == ["circle", 0.91, 0.5, 0.42, 0.75]
    assert thresholds_table["sort_column"] is None and thresholds_table["per_class"] is True
    assert "the precision 0.99 or 0.95 is never reached" in thresholds_table["legend"]

    # With the curves, the thresholds written as on the image
    class_curves = [{"threshold_points": [(0.5, 1.0, "0.9100"), (0.75, 0.96, "0.42001")]},
                    {"threshold_points": [None, (1.0, 0.95, "0.3000")]}, {"threshold_points": [None, None]}]
    thresholds_table = build_class_thresholds_table(class_rows=class_rows, target_precisions=[0.99, 0.95],
                                                    class_curves=class_curves)
    assert thresholds_table["rows"][:2] == [["circle", "0.9100", 0.5, "0.42001", 0.75],
                                            ["square", None, None, "0.3000", 1.0]]


def test_thresholds_need_one_target_precision_each() -> None:
    """
    Verifies that thresholds given without their target precisions, or with a different number of them, are refused
    instead of naming their columns after the wrong precision.
    """
    for target_precisions in [None, [0.99, 0.95]]:
        with pytest.raises(expected_exception=ValueError, match="one target precision per list of thresholds"):
            build_class_metrics_rows(class_names=CLASS_NAMES, metrics=METRICS, class_thresholds=CLASS_THRESHOLDS,
                                     target_precisions=target_precisions)
