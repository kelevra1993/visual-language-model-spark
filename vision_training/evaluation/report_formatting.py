"""
Formatting of the report of an evaluated checkpoint, identical in the console and in the Results.txt file of the run:
the header naming the iteration, the evaluated samples and the time, the grid of the global metrics, the notes and the
tables of the task (the per-class ones only with the per-class report), drawn with rich, plus the formatting of the
values written to the per-class metrics files. Pure functions, used by evaluation_report.py.

It also holds the percentages of every text read by a person, so that the rates (accuracy, precision, recall, F1, ROC
AUC, average precision, shares) are written alike in Results.txt, the console summaries, the figures and the comparison
scripts: "94.03 %" for a measured rate, "99 %" for an axis tick or a target precision of the configuration, "+0.52
points" for a difference between two rates. The thresholds, probabilities, temperatures and losses stay between 0 and
1, and so do the files read back by the scripts and the export (class_metrics.csv, class_thresholds.json, the
evaluation summary, the TensorBoard and MLflow scalars). Its twin in the temperature explorer page is formatPercent and
formatPoints of temperature_explorer_template.html.
"""

import math
import numbers

from io import StringIO
from rich import box
from rich.console import Console, RenderableType
from rich.measure import Measurement
from rich.padding import Padding
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from typing import Any, Dict, List, Optional

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

# Percentages of the rates read by a person, as in the research reports: two decimals and a space before the sign
# ("94.03 %"), the fewest decimals up to two for the values that are not measured (axis ticks, target precisions), and
# the unit of a difference between two rates ("+0.52 points")
PERCENT_DECIMALS = 2
PERCENT_SIGN = " %"
POINTS_UNIT = " points"

# Largest width of the first column of a table (the class names), longer names being cut with an ellipsis, and the
# width it may be narrowed to when the other columns need the room of the report (percent columns of a wide table)
FIRST_COLUMN_MAXIMUM_WIDTH = 32
FIRST_COLUMN_MINIMUM_WIDTH = 12

# Width of the console measuring the natural width of a table, wide enough for any report table
MEASUREMENT_CONSOLE_WIDTH = 1000

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


def convert_to_percentage(value: float) -> float:
    """
    Turns a rate between 0 and 1 into its percentage, the single place where the formatting of the percentages read by
    a person multiplies by 100, so that the reports, figures and scripts cannot drift apart.

    Args:
        value (float): The rate, e.g. 0.9403, or a difference between two rates.

    Returns:
        float: The percentage, e.g. 94.03, or the difference in percentage points.
    """
    return 100.0 * value


def round_percentage(value: float) -> float:
    """
    Rounds the percentage of a rate or of a difference to the PERCENT_DECIMALS written, without a negative zero, so that
    a tiny negative difference is written "+0.00 points" and tells the console summary it is no change.

    Args:
        value (float): The rate or the difference, between 0 and 1.

    Returns:
        float: The rounded percentage, e.g. 94.03, 0.0 for a difference rounding to zero.
    """
    # Adding 0.0 turns the negative zero of a rounded tiny negative value into a positive zero
    return round(convert_to_percentage(value=value), PERCENT_DECIMALS) + 0.0


def format_percentage(value: float) -> str:
    """
    Writes a measured rate as a percentage with two decimals and a space before the sign, e.g. 0.9403 as "94.03 %",
    the form of the accuracies, precisions, recalls, F1 scores, ROC AUC, average precisions and shares of the reports,
    console summaries, figures and comparison scripts.

    Args:
        value (float): The rate, between 0 and 1.

    Returns:
        str: The percentage, e.g. "94.03 %".
    """
    return f"{round_percentage(value=value):.{PERCENT_DECIMALS}f}{PERCENT_SIGN}"


def format_percentage_label(value: float) -> str:
    """
    Writes a percentage that is not a measured value, an axis tick or a target precision of the configuration, with
    the fewest decimals up to two, e.g. 0.99 as "99 %", 0.985 as "98.5 %" and 0.2 as "20 %", so that the axes and the
    table headers stay short next to the measured rates written with two decimals.

    Args:
        value (float): The tick or the target precision, between 0 and 1.

    Returns:
        str: The percentage, e.g. "99 %".
    """
    percentage_text = f"{round_percentage(value=value):.{PERCENT_DECIMALS}f}".rstrip("0").rstrip(".")
    return f"{percentage_text}{PERCENT_SIGN}"


def format_percentage_tick(value: float, position: Optional[int] = None) -> str:
    """
    Writes a tick of a matplotlib axis of rates as a percentage label (format_percentage_label), given to a
    FuncFormatter by the confusion matrix, the ROC and precision-recall images and the comparison figures.

    Args:
        value (float): The value of the tick, between 0 and 1.
        position (Optional[int]): The position of the tick, given by matplotlib and not used. Defaults to None.

    Returns:
        str: The tick label, e.g. "20 %".
    """
    return format_percentage_label(value=value)


def format_points(difference: float) -> str:
    """
    Writes a difference between two rates in percentage points with two decimals and its sign, e.g. 0.0052 as "+0.52
    points", for the changes of the console summary, the recall changes of the temperature explorer and the gaps of the
    comparison figures. A difference rounding to zero is written "+0.00 points".

    Args:
        difference (float): The difference between two rates.

    Returns:
        str: The difference, e.g. "+0.52 points".
    """
    return f"{round_percentage(value=difference):+.{PERCENT_DECIMALS}f}{POINTS_UNIT}"


def format_points_tick(difference: float, position: Optional[int] = None) -> str:
    """
    Writes a tick of a matplotlib axis of differences between two rates in percentage points, with its sign and the
    fewest decimals up to two and without the unit given by the axis name, e.g. 0.005 as "+0.5" and 0 as "0", for the
    gap panels of the comparison figures.

    Args:
        difference (float): The value of the tick, a difference between two rates.
        position (Optional[int]): The position of the tick, given by matplotlib and not used. Defaults to None.

    Returns:
        str: The tick label, e.g. "+0.5".
    """
    rounded_difference = round_percentage(value=difference)
    difference_text = f"{abs(rounded_difference):.{PERCENT_DECIMALS}f}".rstrip("0").rstrip(".")
    if rounded_difference == 0:
        return difference_text
    return f"{'+' if rounded_difference > 0 else '-'}{difference_text}"


def format_report_value(value: Any, in_percent: bool = False) -> str:
    """
    Writes a cell of the report of an evaluated checkpoint: text as it is, integers as they are, other numbers with 4
    decimals, and missing values as "—". The cells of a rate shown as a percentage (a metric listed in the
    'percent_metrics' of the results, a column listed in the 'percent_columns' of its table) are written "94.03 %".

    Args:
        value (Any): The raw cell: text, an integer, a float or None.
        in_percent (bool): Whether a number of the cell is a rate written as a percentage. Defaults to False.

    Returns:
        str: The text of the cell.

    Raises:
        TypeError: If the cell is of another type.
    """
    if is_missing_value(value=value):
        return MISSING_VALUE_TEXT
    if isinstance(value, str):
        return value
    if in_percent and isinstance(value, numbers.Real) and not isinstance(value, bool):
        return format_percentage(value=float(value))
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


def build_metrics_grid(metrics: Dict[str, float], percent_metrics: List[str]) -> Table:
    """
    Lays out the global metrics of an evaluated checkpoint as a grid of name and value pairs, two pairs per line, the
    rates listed by the task written as percentages ("94.03 %") and the other values with 4 decimals.

    Args:
        metrics (Dict[str, float]): The global metrics, in display order.
        percent_metrics (List[str]): The names of the metrics written as percentages, e.g. ['Accuracy', 'Macro F1'].

    Returns:
        Table: The grid of the metrics.

    Raises:
        ValueError: If a percent metric is not one of the metrics.
    """
    unknown_metrics = [metric_name for metric_name in percent_metrics if metric_name not in metrics]
    if unknown_metrics:
        raise ValueError(f"Expected the percent metrics to be among the metrics {list(metrics)}, got "
                         f"{unknown_metrics}.")
    grid = Table.grid(padding=(0, 3))
    for _ in range(2 * METRICS_PER_ROW):
        grid.add_column()

    # One line per group of pairs, the last line completed with empty cells
    metric_items = list(metrics.items())
    for first_index in range(0, len(metric_items), METRICS_PER_ROW):
        cells = []
        for metric_name, metric_value in metric_items[first_index:first_index + METRICS_PER_ROW]:
            metric_text = format_report_value(value=metric_value, in_percent=metric_name in percent_metrics)
            cells += [Text(text=metric_name, style=METRIC_NAME_STYLE), Text(text=metric_text, style=METRIC_VALUE_STYLE)]
        cells += [Text(text="")] * (2 * METRICS_PER_ROW - len(cells))
        grid.add_row(*cells)
    return grid


def build_table_renderable(table: EvaluationTable) -> Table:
    """
    Draws a report table: its header (bold in the console) and its rows in display order (order_table_rows), the first
    column (the names) left aligned and cut with an ellipsis beyond 32 characters, the values right aligned, the
    columns listed in its 'percent_columns' written as percentages ("94.03 %"). A table with percent columns wider than
    the report narrows its first column down to 12 characters, so that the long names are cut rather than the headers
    and values of the wider percent columns. Every text goes through rich Text, so that a class name such as '[unk]' is
    never read as console markup. The title is written on its own line by build_report_renderables, since a rich table
    title wraps at the width of a narrow table.

    Args:
        table (EvaluationTable): The table.

    Returns:
        Table: The rich table.

    Raises:
        ValueError: If a percent column is not one of the columns of the table.
    """
    unknown_columns = [column for column in table["percent_columns"] if column not in table["columns"]]
    if unknown_columns:
        raise ValueError(f"Expected the percent columns of the table '{table['title']}' to be among its columns "
                         f"{table['columns']}, got {unknown_columns}.")
    percent_column_flags = [column in table["percent_columns"] for column in table["columns"]]
    renderable = Table(box=box.SIMPLE_HEAD, show_edge=False, header_style=TABLE_HEADER_STYLE)
    for column_index, column_name in enumerate(table["columns"]):
        if column_index == 0:
            renderable.add_column(header=Text(text=column_name), justify="left", no_wrap=True, overflow="ellipsis",
                                  max_width=FIRST_COLUMN_MAXIMUM_WIDTH)
        else:
            renderable.add_column(header=Text(text=column_name), justify="right", no_wrap=True)
    # Rows in display order, the cells of the percent columns written as percentages
    ordered_rows = order_table_rows(table=table)
    for row in ordered_rows:
        renderable.add_row(*[Text(text=format_report_value(value=cell, in_percent=in_percent))
                             for cell, in_percent in zip(row, percent_column_flags)])

    # A table with percent columns wider than the room of the report gives the excess back from its first column, whose
    # long names are then cut with an ellipsis, instead of rich cutting every column evenly ("Precisi…", "30.76…"); the
    # tables without percentages keep the layout they always had
    excess_width = measure_table_width(renderable=renderable) - (REPORT_WIDTH - REPORT_INDENT)
    if table["percent_columns"] and excess_width > 0:
        first_column_texts = table["columns"][0].split("\n") + [format_report_value(value=row[0], in_percent=False)
                                                               for row in ordered_rows]
        first_column_width = min(max(Text(text=text).cell_len for text in first_column_texts),
                                 FIRST_COLUMN_MAXIMUM_WIDTH)
        renderable.columns[0].max_width = max(first_column_width - excess_width, FIRST_COLUMN_MINIMUM_WIDTH)
    return renderable


def measure_table_width(renderable: Table) -> int:
    """
    Measures the width a report table takes when nothing is cut, its paddings and column separators included, so that
    build_table_renderable can narrow the first column of a table that would not fit in the report.

    Args:
        renderable (Table): The rich table, with its columns and rows.

    Returns:
        int: The width of the table in characters.
    """
    console = Console(file=StringIO(), width=MEASUREMENT_CONSOLE_WIDTH)
    return Measurement.get(console=console, options=console.options, renderable=renderable).maximum


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
        content.append(build_metrics_grid(metrics=results["metrics"], percent_metrics=results["percent_metrics"]))
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
