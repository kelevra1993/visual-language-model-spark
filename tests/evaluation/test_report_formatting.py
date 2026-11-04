"""
Tests of vision_training/evaluation/report_formatting.py: the formatting of the per-class metrics files, and the
block of the report rendered as plain text for Results.txt (header, metrics, notes, the per-class table worst first
with its legend, width, class names kept as text).
"""

import math

from vision_training.core.contracts import EvaluationResults, EvaluationTable
from vision_training.evaluation.report_formatting import (REPORT_WIDTH, build_report_renderables, format_file_value,
                                                          render_report_text)


def build_class_table(rows: list) -> EvaluationTable:
    """
    Builds a per-class table of the tests: the class name, its samples and its F1, sorted by ascending F1.

    Args:
        rows (list): The rows of the table, in class index order.

    Returns:
        EvaluationTable: The per-class table.
    """
    return {"title": "Per Class Metrics", "columns": ["Class", "Samples", "F1"], "rows": rows, "sort_column": "F1",
            "per_class": True, "legend": "Worst F1 first."}


def build_results(tables: list) -> EvaluationResults:
    """
    Builds the evaluation results of the tests: 6 evaluated images, three global metrics, a note and the given tables.

    Args:
        tables (list): The tables of the results.

    Returns:
        EvaluationResults: The results.
    """
    return {"evaluated_samples": "6 Test Images",
            "metrics": {"Accuracy": 0.8333333, "Balanced Accuracy": 0.75, "Macro F1": 0.7222},
            "notes": ["Error Images : 1 in 1 class pairs, at most 1 per pair"], "tables": tables, "scalars": {},
            "class_scalars": {}, "images": {}, "sample_images": {}, "files": {}, "headline_metric": "Accuracy"}


def test_file_values_are_formatted() -> None:
    """
    Verifies the values of the per-class metrics files: floats with 6 decimals, missing values as empty cells, the
    other values as they are.
    """
    assert [format_file_value(value=value) for value in [0.25, None, math.nan, 7, "star"]] == [
        "0.250000", "", "", 7, "star"]


def test_report_block_is_rendered_as_plain_text() -> None:
    """
    Verifies the block of the report written to Results.txt: the header with the iteration, the evaluated samples and
    the time, the global metrics with 4 decimals, the note, the per-class table worst first with its legend, no
    console colour codes, no line wider than the report, and the class names kept as text (rich markup brackets) or
    cut with an ellipsis when too long.
    """
    table = build_class_table(rows=[["circle", 3, 1.0], ["[unk]", 1, 0.0], ["a" * 40, 2, 0.5]])
    text = render_report_text(renderables=build_report_renderables(results=build_results(tables=[table]), iteration=4,
                                                                   timestamp="2026-10-04 14:32",
                                                                   per_class_report=True))
    lines = text.splitlines()

    assert lines[0].startswith("Iteration 4 · 6 Test Images · 2026-10-04 14:32 ─")
    assert "Accuracy" in lines[1] and "0.8333" in lines[1] and "Balanced Accuracy" in lines[1]
    assert "Error Images : 1 in 1 class pairs, at most 1 per pair" in text
    assert "Per Class Metrics" in text and "Worst F1 first." in text
    table_rows = [line.split()[0] for line in lines if line.strip().startswith(("circle", "[unk]", "aaa"))]
    assert table_rows == ["[unk]", "a" * 31 + "…", "circle"]
    assert "\x1b" not in text and all(len(line) <= REPORT_WIDTH for line in lines)
