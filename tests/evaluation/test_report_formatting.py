"""
Tests of vision_training/evaluation/report_formatting.py: the formatting of the per-class metrics files, the
percentages of the texts read by a person ("94.03 %", "99 %", "+0.52 points"), and the block of the report rendered as
plain text for Results.txt (header, metrics, notes, the per-class table worst first with its legend, width, class
names kept as text, the rates listed by the task written as percentages and the other values with 4 decimals, the
headline metric outside the grid allowed among the percent metrics), and the rates of the graded columns coloured
from red to green in the console only.
"""

import math
import pytest

from typing import List

from vision_training.core.contracts import EvaluationResults, EvaluationTable
from vision_training.evaluation.report_formatting import (REPORT_WIDTH, build_report_renderables,
                                                          build_table_renderable, build_table_renderables,
                                                          format_file_value, format_percentage,
                                                          format_percentage_label, format_percentage_tick,
                                                          format_points, format_points_tick, format_report_value,
                                                          grade_rate_style, render_report_text)


def build_class_table(rows: list, percent_columns: List[str]) -> EvaluationTable:
    """
    Builds a per-class table of the tests: the class name, its samples and its F1, sorted by ascending F1.

    Args:
        rows (list): The rows of the table, in class index order.
        percent_columns (List[str]): The columns written as percentages, e.g. ["F1"].

    Returns:
        EvaluationTable: The per-class table.
    """
    return {"title": "Per Class Metrics", "columns": ["Class", "Samples", "F1"], "rows": rows, "sort_column": "F1",
            "per_class": True, "legend": "Worst F1 first.", "percent_columns": percent_columns}


def build_results(tables: list, percent_metrics: List[str]) -> EvaluationResults:
    """
    Builds the evaluation results of the tests: 6 evaluated images, three global metrics and a value that is not a rate,
    a note and the given tables.

    Args:
        tables (list): The tables of the results.
        percent_metrics (List[str]): The global metrics written as percentages, e.g. ["Accuracy"].

    Returns:
        EvaluationResults: The results.
    """
    return {"evaluated_samples": "6 Test Images",
            "metrics": {"Accuracy": 0.8333333, "Balanced Accuracy": 0.75, "Macro F1": 0.7222,
                        "Effective Rank": 12.5},
            "notes": ["Error Images : 1 in 1 class pairs, at most 1 per pair"], "tables": tables, "scalars": {},
            "class_scalars": {}, "images": {}, "sample_images": {}, "files": {}, "headline_metric": "Accuracy",
            "percent_metrics": percent_metrics}


def render_block(results: EvaluationResults) -> List[str]:
    """
    Renders the block of the report of the tests as the plain text lines of Results.txt.

    Args:
        results (EvaluationResults): The results of the tests (build_results).

    Returns:
        List[str]: The lines of the block.
    """
    return render_report_text(renderables=build_report_renderables(results=results, iteration=4,
                                                                   timestamp="2026-10-04 14:32",
                                                                   per_class_report=True)).splitlines()


def test_file_values_are_formatted() -> None:
    """
    Verifies the values of the per-class metrics files: floats with 6 decimals, missing values as empty cells, the
    other values as they are.
    """
    assert [format_file_value(value=value) for value in [0.25, None, math.nan, 7, "star"]] == [
        "0.250000", "", "", 7, "star"]


def test_percentages_follow_the_research_reports() -> None:
    """
    Verifies the percentages read by a person, as in the research reports: a measured rate with two decimals and a
    space before the sign, an axis tick or a target precision with the fewest decimals, a difference in points with its
    sign and never "-0.00", a report cell in percent only when asked, the missing values still written "—".
    """
    assert [format_percentage(value=value) for value in [0.9403, 1.0, 0.0, 0.123456]] == [
        "94.03 %", "100.00 %", "0.00 %", "12.35 %"]
    assert [format_percentage_label(value=value) for value in [0.99, 0.985, 0.2, 0.0, 1.0]] == [
        "99 %", "98.5 %", "20 %", "0 %", "100 %"]
    assert format_percentage_tick(value=0.4, position=3) == "40 %"
    assert [format_points(difference=difference) for difference in [0.0052, -0.0052, -0.000001, 0.0]] == [
        "+0.52 points", "-0.52 points", "+0.00 points", "+0.00 points"]
    assert [format_points_tick(difference=difference) for difference in [0.005, -0.01, 0.0, -0.0000001]] == [
        "+0.5", "-1", "0", "0"]
    assert [format_report_value(value=value, in_percent=True) for value in [0.5, 1, None, math.nan, "0.9998"]] == [
        "50.00 %", "100.00 %", "—", "—", "0.9998"]
    assert format_report_value(value=0.5) == "0.5000"


def test_report_block_is_rendered_as_plain_text() -> None:
    """
    Verifies the block of the report written to Results.txt: the header with the iteration, the evaluated samples and
    the time, the global metrics with 4 decimals when the task lists no percent metric, the note, the per-class table
    worst first with its legend, no console colour codes, no line wider than the report, and the class names kept as
    text (rich markup brackets) or cut with an ellipsis when too long.
    """
    table = build_class_table(rows=[["circle", 3, 1.0], ["[unk]", 1, 0.0], ["a" * 40, 2, 0.5]], percent_columns=[])
    lines = render_block(results=build_results(tables=[table], percent_metrics=[]))
    text = "\n".join(lines)

    assert lines[0].startswith("Iteration 4 · 6 Test Images · 2026-10-04 14:32 ─")
    assert "Accuracy" in lines[1] and "0.8333" in lines[1] and "Balanced Accuracy" in lines[1]
    assert "Error Images : 1 in 1 class pairs, at most 1 per pair" in text
    assert "Per Class Metrics" in text and "Worst F1 first." in text
    table_rows = [line.split()[0] for line in lines if line.strip().startswith(("circle", "[unk]", "aaa"))]
    assert table_rows == ["[unk]", "a" * 31 + "…", "circle"]
    assert "\x1b" not in text and all(len(line) <= REPORT_WIDTH for line in lines)


def test_percent_metrics_and_columns_are_written_as_percentages() -> None:
    """
    Verifies that the rates listed by the task are written as percentages in the metrics grid and the tables of
    Results.txt, the other values keeping 4 decimals or their integer, and that the rows are still sorted on the raw
    values.
    """
    table = build_class_table(rows=[["circle", 3, 1.0], ["square", 1, 0.0], ["star", 2, 0.12345]],
                              percent_columns=["F1"])
    lines = render_block(results=build_results(tables=[table], percent_metrics=["Accuracy", "Balanced Accuracy",
                                                                                "Macro F1"]))
    text = "\n".join(lines)

    # Rates of the grid in percent, the value that is not a rate with 4 decimals
    assert "83.33 %" in lines[1] and "75.00 %" in lines[1] and "0.8333" not in text
    assert "72.22 %" in lines[2] and "12.5000" in lines[2]

    # Per-class rows worst F1 first, the samples as integers and the F1 in percent
    table_rows = [line.split() for line in lines if line.strip().startswith(("circle", "square", "star"))]
    assert table_rows == [["square", "1", "0.00", "%"], ["star", "2", "12.35", "%"], ["circle", "3", "100.00", "%"]]


def test_wide_tables_cut_the_names_rather_than_the_values() -> None:
    """
    Verifies that a per-class table wider than the report, like the percent table of a run of 71 classes with long
    names and the curve and threshold columns, narrows its first column: the long names are cut with an ellipsis while
    every header and value of the other columns stays whole, and no line is wider than the report.
    """
    columns = ["Class", "Frames", "Precision", "Recall", "F1", "ROC AUC", "Average\nPrecision", "Threshold\n@ P99 %",
               "Recall\n@ P99 %"]
    rows = [["disappearing_hollow_parallelogram", 1234, 0.307612, 1.0, 0.123456, 0.65715, 1.0, 0.999912, 0.004321],
            ["no_change", 98765, 0.5, 0.25, 0.75, 0.5, 0.5, 0.5, 0.5]]
    percent_columns = [column for column in columns[2:] if "Threshold" not in column]
    table = {"title": "Per Class Metrics", "columns": columns, "rows": rows, "sort_column": "F1", "per_class": True,
             "legend": "Worst F1 first.", "percent_columns": percent_columns}
    lines = render_block(results=build_results(tables=[table], percent_metrics=[]))

    # Long name cut, the headers and the values of the other columns whole
    name_row = next(line for line in lines if line.strip().startswith("disappearing"))
    assert name_row.split()[0].endswith("…") and name_row.count("…") == 1
    assert name_row.split()[1:] == ["1234", "30.76", "%", "100.00", "%", "12.35", "%", "65.72", "%", "100.00", "%",
                                    "0.9999", "0.43", "%"]
    header_text = "\n".join(lines[:12])
    assert "…" not in header_text.replace(name_row, "") and "Precision" in header_text and "@ P99 %" in header_text
    assert all(len(line) <= REPORT_WIDTH for line in lines)


def test_headline_metric_outside_the_grid_may_be_a_percent_metric() -> None:
    """
    Verifies that the percent metrics may name the headline metric when it is a scalar outside the grid, like the
    OKS mAP@0.5/area of the keypoint detection written as a percentage in the best checkpoint line, the grid writing
    its own percent metrics as percentages and the others with 4 decimals, while another scalar outside the grid is
    still refused.
    """
    results = {**build_results(tables=[], percent_metrics=["Accuracy", "OKS mAP@0.5/area"]),
               "headline_metric": "OKS mAP@0.5/area"}
    lines = render_block(results=results)
    assert "83.33 %" in lines[1] and "0.7500" in lines[1] and "OKS mAP@0.5/area" not in "\n".join(lines)

    # A scalar outside the grid that is not the headline metric stops the report
    with pytest.raises(ValueError, match="headline metric 'OKS mAP@0.5/area'"):
        render_block(results={**results, "percent_metrics": ["Accuracy", "OKS mAP@0.75/area"]})


@pytest.mark.parametrize(argnames="percent_metrics, percent_columns",
                         argvalues=[(["Top-5 Accuracy"], []), ([], ["Recall"])])
def test_unknown_percent_names_are_refused(percent_metrics: List[str], percent_columns: List[str]) -> None:
    """
    Verifies that a percent metric missing from the metrics, or a percent column missing from the columns of its
    table, stops the report instead of leaving a rate with 4 decimals.

    Args:
        percent_metrics (List[str]): The percent metrics given by the task.
        percent_columns (List[str]): The percent columns of the per-class table.
    """
    table = build_class_table(rows=[["circle", 3, 1.0]], percent_columns=percent_columns)
    with pytest.raises(ValueError, match="percent"):
        render_block(results=build_results(tables=[table], percent_metrics=percent_metrics))


def test_graded_columns_colour_their_rates_from_red_to_green() -> None:
    """
    Verifies that the rates of the graded columns (the anchor recall) are coloured red at 0, amber at 0.5 and green at
    1, blended in between and clamped outside, a missing value and the other columns keeping the default colour, that
    the plain text of Results.txt carries no colour code, and that a graded column outside the percent columns is
    refused.
    """
    assert grade_rate_style(rate=0.0) == "rgb(215,58,50)" and grade_rate_style(rate=1.0) == "rgb(45,160,70)"
    assert grade_rate_style(rate=0.5) == "rgb(205,150,20)" and grade_rate_style(rate=0.25) == "rgb(210,104,35)"
    assert grade_rate_style(rate=1.4) == grade_rate_style(rate=1.0) and grade_rate_style(rate=-0.1) == "rgb(215,58,50)"
    assert grade_rate_style(rate=None) == "" and grade_rate_style(rate=float("nan")) == ""

    # Cells of the graded column coloured by their rate, those of the other columns left plain
    table: EvaluationTable = {"title": "Anchor Recall", "columns": ["Objects", "Count", "All Levels"],
                              "rows": [["All Objects", 4, 0.0], ["Small", 2, 1.0], ["Large", 0, None]],
                              "sort_column": None, "per_class": False, "legend": "", "percent_columns": ["All Levels"],
                              "graded_columns": ["All Levels"]}
    columns = build_table_renderable(table=table).columns
    assert [str(cell.style) for cell in columns[2].cells] == ["rgb(215,58,50)", "rgb(45,160,70)", ""]
    assert all(str(cell.style) == "" for column in columns[:2] for cell in column.cells)

    # The plain text of Results.txt without colour codes, the rates still in percent
    text = render_report_text(renderables=build_table_renderables(table=table))
    assert "\x1b[" not in text and "100.00 %" in text

    # A graded column must be a percent column, whose cells hold rates
    with pytest.raises(ValueError, match="graded columns"):
        build_table_renderable(table={**table, "graded_columns": ["Count"]})
