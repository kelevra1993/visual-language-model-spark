"""
Formatting of the report of an evaluated checkpoint, identical in the console and in the Results.txt file of the run:
the header naming the iteration, the evaluated samples and the time, the grid of the global metrics, the notes and the
tables of the task (the per-class ones only with the per-class report), drawn with rich, plus the formatting of the
values written to the per-class metrics files. Pure functions, used by evaluation_report.py.
"""

import math
import numbers

from io import StringIO
from rich import box
from rich.console import Console, RenderableType
from rich.padding import Padding
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from typing import Any, Dict, List

from vision_training.core.contracts import EvaluationResults, EvaluationTable

# Width of the report in characters, the same in the console and in Results.txt so that both layouts match, and the
# indentation of the content below the header
REPORT_WIDTH = 120
REPORT_INDENT = 2

# Number of name and value pairs of the global metrics written on every line of the metrics grid
METRICS_PER_ROW = 2

# Format of the time written in the header of every block, and separator of the parts of the header
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M"
HEADER_SEPARATOR = " · "

# Text of a missing value (a class without samples, a precision never reached), and decimals of the values written in
# the report and in the per-class metrics files
MISSING_VALUE_TEXT = "—"
REPORT_DECIMALS = 4
FILE_DECIMALS = 6

# Largest width of the first column of a table (the class names), longer names being cut with an ellipsis
FIRST_COLUMN_MAXIMUM_WIDTH = 32

# Console styles of the parts of the report, left out of Results.txt
HEADER_STYLE = "bold cyan"
METRIC_NAME_STYLE = "bold"
METRIC_VALUE_STYLE = "bold green"
NOTE_STYLE = "yellow"
TABLE_TITLE_STYLE = "bold"
TABLE_HEADER_STYLE = "bold"
LEGEND_STYLE = "dim"


def is_missing_value(value: Any) -> bool:
    """
    Tells whether a cell of a report table has no value, None or a NaN float (e.g. the recall of a class without test
    samples), so that it is written "—" and sorted after the classes with a value.

    Args:
        value (Any): The raw cell.

    Returns:
        bool: True if the cell has no value.
    """
    return value is None or (isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral) and
                             math.isnan(value))


def format_report_value(value: Any) -> str:
    """
    Writes a cell of the report of an evaluated checkpoint: text as it is, integers as they are, other numbers with 4
    decimals, and missing values as "—".

    Args:
        value (Any): The raw cell: text, an integer, a float or None.

    Returns:
        str: The text of the cell.

    Raises:
        TypeError: If the cell is of another type.
    """
    if is_missing_value(value=value):
        return MISSING_VALUE_TEXT
    if isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return str(value)
    if isinstance(value, numbers.Real):
        return f"{value:.{REPORT_DECIMALS}f}"
    raise TypeError(f"A report cell must be text, a number or None, got {type(value).__name__}.")


def format_file_value(value: Any) -> Any:
    """
    Writes a value of a per-class metrics file (class_metrics.csv): floats with 6 decimals, missing values as empty
    cells, the other values as they are, so that the file stays easy to read back with any CSV reader.

    Args:
        value (Any): The raw value.

    Returns:
        Any: The value written to the file.
    """
    if is_missing_value(value=value):
        return ""
    if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
        return f"{value:.{FILE_DECIMALS}f}"
    return value


def order_table_rows(table: EvaluationTable) -> List[List[Any]]:
    """
    Orders the rows of a report table for display: by ascending value of its sort column, the worst classes first, the
    rows without value last and ties kept in their stored order (class index order), or in stored order when the table
    has no sort column.

    Args:
        table (EvaluationTable): The table.

    Returns:
        List[List[Any]]: The rows in display order.
    """
    if table["sort_column"] is None:
        return list(table["rows"])

    # Sort keys built as tuples, the row index breaking the ties so that the rows themselves are never compared
    sort_index = table["columns"].index(table["sort_column"])
    decorated_rows = []
    for row_index, row in enumerate(table["rows"]):
        value = row[sort_index]
        is_missing = is_missing_value(value=value)
        decorated_rows.append((is_missing, 0.0 if is_missing else float(value), row_index, row))
    return [decorated_row[3] for decorated_row in sorted(decorated_rows)]


def build_report_header(iteration: int, evaluated_samples: str, timestamp: str) -> str:
    """
    Builds the header of the block of an evaluated checkpoint, which starts every block of Results.txt so that the
    blocks can be found again, e.g. 'Iteration 4000 · 50000 Test Images · 2026-10-04 14:32'.

    Args:
        iteration (int): The training iteration of the evaluated checkpoint.
        evaluated_samples (str): The description of the evaluated samples, e.g. '50000 Test Images'.
        timestamp (str): The time of the evaluation, in TIMESTAMP_FORMAT.

    Returns:
        str: The header.
    """
    return HEADER_SEPARATOR.join([f"Iteration {iteration}", evaluated_samples, timestamp])


def build_metrics_grid(metrics: Dict[str, float]) -> Table:
    """
    Lays out the global metrics of an evaluated checkpoint as a grid of name and value pairs, two pairs per line, the
    values with 4 decimals.

    Args:
        metrics (Dict[str, float]): The global metrics, in display order.

    Returns:
        Table: The grid of the metrics.
    """
    grid = Table.grid(padding=(0, 3))
    for _ in range(2 * METRICS_PER_ROW):
        grid.add_column()

    # One line per group of pairs, the last line completed with empty cells
    metric_items = list(metrics.items())
    for first_index in range(0, len(metric_items), METRICS_PER_ROW):
        cells = []
        for metric_name, metric_value in metric_items[first_index:first_index + METRICS_PER_ROW]:
            cells += [Text(text=metric_name, style=METRIC_NAME_STYLE),
                      Text(text=format_report_value(value=metric_value), style=METRIC_VALUE_STYLE)]
        cells += [Text(text="")] * (2 * METRICS_PER_ROW - len(cells))
        grid.add_row(*cells)
    return grid


def build_table_renderable(table: EvaluationTable) -> Table:
    """
    Draws a report table: its header (bold in the console) and its rows in display order (order_table_rows), the first
    column (the names) left aligned and cut with an ellipsis beyond 32 characters, the values right aligned. Every text
    goes through rich Text, so that a class name such as '[unk]' is never read as console markup. The title is written
    on its own line by build_report_renderables, since a rich table title wraps at the width of a narrow table.

    Args:
        table (EvaluationTable): The table.

    Returns:
        Table: The rich table.
    """
    renderable = Table(box=box.SIMPLE_HEAD, show_edge=False, header_style=TABLE_HEADER_STYLE)
    for column_index, column_name in enumerate(table["columns"]):
        if column_index == 0:
            renderable.add_column(header=Text(text=column_name), justify="left", no_wrap=True, overflow="ellipsis",
                                  max_width=FIRST_COLUMN_MAXIMUM_WIDTH)
        else:
            renderable.add_column(header=Text(text=column_name), justify="right", no_wrap=True)
    for row in order_table_rows(table=table):
        renderable.add_row(*[Text(text=format_report_value(value=cell)) for cell in row])
    return renderable


def build_table_renderables(table: EvaluationTable) -> List[RenderableType]:
    """
    Lays out one table of a report with its title above it and its legend below it, for the block of an evaluated
    checkpoint (build_report_renderables) and for the tables printed at the launch of a run, such as the anchor recall
    of the detection tasks, which take the layout of the report.

    Args:
        table (EvaluationTable): The table.

    Returns:
        List[RenderableType]: The title, the table and the legend when it has one.
    """
    renderables: List[RenderableType] = [Text(text=table["title"], style=TABLE_TITLE_STYLE),
                                         build_table_renderable(table=table)]
    if table["legend"]:
        renderables.append(Text(text=table["legend"], style=LEGEND_STYLE))
    return renderables


def build_report_renderables(results: EvaluationResults, iteration: int, timestamp: str,
                             per_class_report: bool) -> List[RenderableType]:
    """
    Builds the block of the report of an evaluated checkpoint, printed in the console and written to Results.txt: the
    header rule, then indented below it the grid of the global metrics, the notes (e.g. the error images) and the tables
    of the task with their legends, the per-class tables only with the per-class report.

    Args:
        results (EvaluationResults): The results of the evaluation, computed by the task.
        iteration (int): The training iteration of the evaluated checkpoint.
        timestamp (str): The time of the evaluation, in TIMESTAMP_FORMAT.
        per_class_report (bool): Whether the per-class tables are shown.

    Returns:
        List[RenderableType]: The parts of the block, in order.
    """
    header = build_report_header(iteration=iteration, evaluated_samples=results["evaluated_samples"],
                                 timestamp=timestamp)
    renderables: List[RenderableType] = [Rule(title=Text(text=header, style=HEADER_STYLE), align="left")]
    content: List[RenderableType] = []

    # Global metrics and notes
    if results["metrics"]:
        content.append(build_metrics_grid(metrics=results["metrics"]))
    content += [Text(text=note, style=NOTE_STYLE) for note in results["notes"]]

    # Tables, separated from what comes before them by an empty line, each with its title above it and its legend
    # below it
    for table in results["tables"]:
        if table["per_class"] and not per_class_report:
            continue
        if content:
            content.append(Text(text=" "))
        content += build_table_renderables(table=table)

    renderables += [Padding(renderable=part, pad=(0, 0, 0, REPORT_INDENT)) for part in content]
    return renderables


def render_report_text(renderables: List[RenderableType]) -> str:
    """
    Renders the block of the report of an evaluated checkpoint as plain text, without colours, for the Results.txt
    file of the run: the same layout and width as in the console, the trailing spaces of every line removed.

    Args:
        renderables (List[RenderableType]): The parts of the block (build_report_renderables).

    Returns:
        str: The text of the block, ending with a line break.
    """
    buffer = StringIO()
    console = Console(file=buffer, width=REPORT_WIDTH, color_system=None, highlight=False, emoji=False)
    for renderable in renderables:
        console.print(renderable)
    return "\n".join(line.rstrip() for line in buffer.getvalue().rstrip("\n").split("\n")) + "\n"


def print_report(renderables: List[RenderableType]) -> None:
    """
    Prints the block of the report of an evaluated checkpoint in the console, with the colours of the report, at the
    same width as in Results.txt.

    Args:
        renderables (List[RenderableType]): The parts of the block (build_report_renderables).
    """
    console = Console(width=REPORT_WIDTH, highlight=False, emoji=False)
    for renderable in renderables:
        console.print(renderable)
