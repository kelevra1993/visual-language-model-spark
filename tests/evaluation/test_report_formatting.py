"""
Tests of vision_training/evaluation/report_formatting.py: the formatting of the report cells and of the per-class
metrics files, the order of the table rows (worst first, missing values last), and the block of the report rendered as
plain text for Results.txt (header, metrics, notes, tables, width, class names kept as text, per-class tables hidden
without the per-class report), and the layout of a table printed alone with its title and legend.
"""

import math
import pytest

from typing import Any, Dict

from vision_training.core.contracts import EvaluationResults, EvaluationTable
from vision_training.evaluation.report_formatting import (REPORT_WIDTH, build_report_header, build_report_renderables,
                                                          build_table_renderables, format_file_value,
                                                          format_report_value, order_table_rows, render_report_text)


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


@pytest.mark.parametrize(argnames=("value", "expected_text"),
                         argvalues=[(None, "—"), (math.nan, "—"), (3, "3"), (0.5, "0.5000"), (1.0, "1.0000"),
                                    ("circle", "circle")])
def test_report_values_are_formatted(value: Any, expected_text: str) -> None:
    """
    Verifies the cells of the report: integers and text as they are, other numbers with 4 decimals, missing values as
    "—".

    Args:
        value (Any): The raw cell.
        expected_text (str): Its expected text.
    """
    assert format_report_value(value=value) == expected_text


def test_unknown_report_values_are_refused() -> None:
    """
    Verifies that a cell of another type than text, a number or None is refused rather than written as its repr.
    """
    with pytest.raises(expected_exception=TypeError, match="text, a number or None"):
        format_report_value(value=[1, 2])


def test_file_values_are_formatted() -> None:
    """
    Verifies the values of the per-class metrics files: floats with 6 decimals, missing values as empty cells, the
    other values as they are.
    """
    assert [format_file_value(value=value) for value in [0.25, None, math.nan, 7, "star"]] == [
        "0.250000", "", "", 7, "star"]


def test_table_rows_are_shown_worst_first() -> None:
    """
    Verifies the display order of a per-class table: ascending sort column, the rows without value last, ties kept in
    class index order, and the stored order for a table without sort column.
    """
    table = build_class_table(rows=[["circle", 3, 1.0], ["hexagon", 0, None], ["square", 2, 0.5],
                                    ["triangle", 2, 0.5], ["star", 1, 0.0]])
    assert [row[0] for row in order_table_rows(table=table)] == ["star", "square", "triangle", "circle", "hexagon"]

    unsorted_table: Dict[str, Any] = {**table, "sort_column": None}
    assert [row[0] for row in order_table_rows(table=unsorted_table)] == ["circle", "hexagon", "square", "triangle",
                                                                         "star"]


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


def test_per_class_tables_are_hidden_without_the_per_class_report() -> None:
    """
    Verifies that the per-class tables are left out of the block without the per-class report, while the global
    metrics, the notes and the tables that are not per-class are kept.
    """
    global_table: EvaluationTable = {"title": "Mean Average Precision", "columns": ["Method", "mAP@0.5"],
                                     "rows": [["Area", 0.5]], "sort_column": None, "per_class": False, "legend": ""}
    results = build_results(tables=[global_table, build_class_table(rows=[["circle", 3, 1.0]])])
    text = render_report_text(renderables=build_report_renderables(results=results, iteration=4,
                                                                   timestamp="2026-10-04 14:32",
                                                                   per_class_report=False))
    assert "Mean Average Precision" in text and "Error Images" in text and "Macro F1" in text
    assert "Per Class Metrics" not in text and "circle" not in text


def test_table_is_laid_out_with_its_title_and_legend() -> None:
    """
    Verifies the layout of one table printed alone, such as the anchor recall at the launch of a detection run: its
    title above it, its rows, and its legend below it only when it has one.
    """
    table: EvaluationTable = {"title": "Anchor Recall", "columns": ["Objects", "Recall"], "rows": [["All", 0.5]],
                              "sort_column": None, "per_class": False, "legend": "Share of the objects."}
    renderables = build_table_renderables(table=table)
    text = render_report_text(renderables=renderables)
    assert len(renderables) == 3
    assert text.splitlines()[0] == "Anchor Recall" and "0.5000" in text
    assert text.rstrip().endswith("Share of the objects.")
    assert len(build_table_renderables(table={**table, "legend": ""})) == 2


def test_report_header() -> None:
    """
    Verifies the header starting every block of Results.txt.
    """
    assert build_report_header(iteration=4000, evaluated_samples="50000 Test Images",
                               timestamp="2026-10-04 14:32") == "Iteration 4000 · 50000 Test Images · 2026-10-04 14:32"
