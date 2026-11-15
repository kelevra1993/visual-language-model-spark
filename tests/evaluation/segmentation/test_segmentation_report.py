"""
Tests of vision_training/evaluation/segmentation/segmentation_report.py: the per-class table of the semantic
segmentation report writes every metric column as a percentage ("47.29 %") under its two-line headers, within the width
of the report and without cutting any value, while the per-class TensorBoard scalars keep their names and their values
between 0 and 1.
"""

import pytest

from typing import Any, Dict, List

from vision_training.evaluation.report_formatting import REPORT_WIDTH, build_table_renderables, render_report_text
from vision_training.evaluation.segmentation.segmentation_report import (build_segmentation_class_rows,
                                                                         build_segmentation_class_scalars,
                                                                         build_segmentation_class_table)

# Classes of the tests: the background covering most pixels with every rate at 1 (the widest cells, "100.00 %"), a
# shape segmented halfway, and a class without true pixels
CLASS_NAMES = ["background", "hexagon", "star"]
CLASS_METRICS = {"support": [47351637, 608283, 0], "iou": [1.0, 0.309698, 0.0], "dice": [1.0, 0.472897, 0.0],
                 "boundary_iou": [1.0, 0.118903, 0.0], "boundary_f1": [1.0, 0.308134, 0.0],
                 "precision": [1.0, 0.554362, 0.0], "recall": [1.0, 0.412308, 0.0],
                 "recall_lower": [1.0, 0.323712, None], "recall_upper": [1.0, 0.505301, None]}

# ROC AUC and average precision of the pixels of every class, given by the curves of the per-class report
CLASS_CURVES = [{"roc_auc": 1.0, "average_precision": 1.0}, {"roc_auc": 0.983799, "average_precision": 0.524975},
                {"roc_auc": None, "average_precision": None}]

# Headers of the metric columns of the table, without and with the curves of the per-class report
METRIC_COLUMNS = ["IoU", "Dice", "Boundary\nIoU", "Boundary\nF1", "Precision", "Recall"]
CURVE_COLUMNS = ["ROC AUC", "Average\nPrecision"]


def build_class_rows(with_curves: bool) -> List[Dict[str, Any]]:
    """
    Builds the per-class rows of the classes of the tests, as the segmentation trainer gives them to the table.

    Args:
        with_curves (bool): Whether the rows hold the ROC AUC and average precision of the per-class report.

    Returns:
        List[Dict[str, Any]]: The rows of the three classes (build_segmentation_class_rows).
    """
    return build_segmentation_class_rows(class_names=CLASS_NAMES, metrics=CLASS_METRICS,
                                         class_curves=CLASS_CURVES if with_curves else None)


@pytest.mark.parametrize(argnames="with_curves", argvalues=[True, False])
def test_class_table_writes_its_rates_as_percentages(with_curves: bool) -> None:
    """
    Verifies the per-class table of the report: every metric column is a percent column under its two-line header,
    the rates are written "47.29 %" (the widest "100.00 %"), the class without true pixels "—", worst IoU first, and
    the table stays within the width of the report without any value cut with an ellipsis.

    Args:
        with_curves (bool): Whether the rows hold the ROC AUC and average precision of the per-class report.
    """
    table = build_segmentation_class_table(class_rows=build_class_rows(with_curves=with_curves))
    expected_metric_columns = METRIC_COLUMNS + (CURVE_COLUMNS if with_curves else [])
    assert table["columns"] == ["Class", "Pixels"] + expected_metric_columns
    assert table["percent_columns"] == expected_metric_columns and table["sort_column"] == "IoU"

    # Rendered rows: the shape first (worst IoU), the background with its widest cells, the class without pixels last
    lines = render_report_text(renderables=build_table_renderables(table=table)).splitlines()
    row_lines = [line for line in lines if line.strip().startswith(tuple(CLASS_NAMES))]
    assert [line.split()[0] for line in row_lines] == ["hexagon", "background", "star"]
    expected_hexagon = ["30.97 %", "47.29 %", "11.89 %", "30.81 %", "55.44 %", "41.23 %"] + (
        ["98.38 %", "52.50 %"] if with_curves else [])
    assert row_lines[0].split()[1:] == ["608283"] + [part for value in expected_hexagon for part in value.split()]
    assert row_lines[1].count("100.00 %") == len(expected_metric_columns)
    assert row_lines[2].split()[1:] == ["0"] + ["—"] * len(expected_metric_columns)
    assert all(len(line) <= REPORT_WIDTH for line in lines) and "…" not in "\n".join(lines)


def test_class_scalars_keep_their_names_and_rates() -> None:
    """
    Verifies that the per-class TensorBoard scalars keep the one-line metric names of the earlier runs ('Boundary
    IoU/<class>', 'Average Precision/<class>') and their values between 0 and 1, whatever the headers of the report,
    the class without true pixels being left out.
    """
    class_scalars = build_segmentation_class_scalars(class_rows=build_class_rows(with_curves=True))
    assert class_scalars["Boundary IoU/hexagon"] == 0.118903 and class_scalars["Boundary F1/hexagon"] == 0.308134
    assert class_scalars["ROC AUC/hexagon"] == 0.983799 and class_scalars["Average Precision/hexagon"] == 0.524975
    assert len(class_scalars) == 2 * len(METRIC_COLUMNS + CURVE_COLUMNS)
    assert not any("\n" in tag or tag.endswith("/star") for tag in class_scalars)
    assert all(0.0 <= value <= 1.0 for value in class_scalars.values())
