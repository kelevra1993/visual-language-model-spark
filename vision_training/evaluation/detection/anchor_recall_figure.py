"""
Figure of the anchor recall of a detection run, anchor_recall.png in the run folder: the anchor recall tables of the
training and test splits (summarise_anchor_recall of region_recall.py) drawn one under the other as an image, their
recalls on cells shaded from red (no object covered) to green (every object covered), so that the fit of the anchors to
the objects of every size can be kept and looked at again without the console of the first launch. Written by the
detection trainer at the first launch of a run (DetectionTrainer.analyse_training_data), next to anchor_recall.csv.
"""

import matplotlib
import textwrap

from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Any, List

from vision_training.core.contracts import EvaluationTable
from vision_training.evaluation.report_formatting import format_report_value
from vision_training.trainers.metric_tracking import MetricTracker

# Inter font of the research report figures when it is installed, the default sans-serif of matplotlib otherwise
FIGURE_FONT_FOLDER = Path("/usr/share/fonts/opentype/inter")
FIGURE_FONT_FAMILIES = ["Inter", "DejaVu Sans"]

# Width of every column in figure inches: the size name, its pixels and area share, the count, then every recall
NAME_COLUMN_WIDTH = 1.6
TEXT_COLUMN_WIDTHS = {"Pixels": 1.2, "Area": 1.45, "Count": 1.15}
RECALL_COLUMN_WIDTH = 1.05
ROW_HEIGHT = 0.62

# Characters of a legend line per inch of table width, the legend being wrapped to the width of the table
LEGEND_CHARACTERS_PER_INCH = 17

# Colours of the figure: the text, the header band, the band of the row of all the objects and every other row, the
# separating line and the legend
TEXT_COLOUR = "#1f2933"
HEADER_COLOUR = "#1f2933"
TOTAL_ROW_COLOUR = "#e4e7eb"
STRIPE_COLOUR = "#f5f7fa"
SEPARATOR_COLOUR = "#9aa5b1"
LEGEND_COLOUR = "#52606d"

# Shading of the recall cells: the matplotlib colour map from red to green, the range of it used (its darkest ends
# left out) and the opacity of the cells
RECALL_COLOUR_MAP = "RdYlGn"
RECALL_COLOUR_RANGE = (0.1, 0.9)
RECALL_CELL_OPACITY = 0.55


def use_figure_font() -> None:
    """
    Registers the Inter font of the research report figures with matplotlib when it is installed, so that the anchor
    recall figure reads like the other figures of the reports, the default sans-serif being kept otherwise.
    """
    for font_path in sorted(FIGURE_FONT_FOLDER.glob("*.otf")):
        font_manager.fontManager.addfont(path=str(font_path))


def format_figure_cell(value: Any, column: str, percent_columns: List[str]) -> str:
    """
    Writes a cell of the anchor recall figure: the recalls as the percentages of the reports, the counts with their
    digits grouped by three as in the console summaries, the other cells as they are.

    Args:
        value (Any): The raw cell of the table.
        column (str): The column of the cell.
        percent_columns (List[str]): The columns written as percentages.

    Returns:
        str: The text of the cell.
    """
    if column in percent_columns:
        return format_report_value(value=value, in_percent=True)
    if column == "Count" and isinstance(value, int):
        return MetricTracker.format_grouped_number(value=value)
    return format_report_value(value=value)


def draw_anchor_recall_figure(tables: List[EvaluationTable], output_path: Path) -> Path:
    """
    Draws the anchor recall tables of the splits of a run one under the other, each with its title, a dark header, the
    row of all the objects then one striped row per object size, its recalls on cells shaded from red to green, and
    its legend, then saves the image.

    Args:
        tables (List[EvaluationTable]): The anchor recall table of every split (summarise_anchor_recall), sharing their
            columns.
        output_path (Path): The PNG file written, anchor_recall.png of the run folder.

    Returns:
        Path: The written file.

    Raises:
        ValueError: If no table is given.
    """
    if not tables:
        raise ValueError("Expected at least one anchor recall table to draw.")
    use_figure_font()
    colour_map = matplotlib.colormaps[RECALL_COLOUR_MAP]

    # Column widths and left edges, shared by the tables of every split
    columns = tables[0]["columns"]
    column_widths = [NAME_COLUMN_WIDTH] + [TEXT_COLUMN_WIDTHS[column] if column in TEXT_COLUMN_WIDTHS
                                           else RECALL_COLUMN_WIDTH for column in columns[1:]]
    left_edges = [sum(column_widths[:index]) for index in range(len(columns))]
    total_width = sum(column_widths)

    # Height of every table: its title, its header and rows, its legend and the gap before the next table
    table_heights = [1.0 + ROW_HEIGHT * (len(table["rows"]) + 1) + 1.1 for table in tables]
    figure = Figure(figsize=(total_width + 0.6, sum(table_heights)), dpi=200, facecolor="white")
    FigureCanvasAgg(figure=figure)
    axis = figure.add_axes(rect=(0.0, 0.0, 1.0, 1.0))
    axis.set_xlim(left=-0.3, right=total_width + 0.3)
    axis.set_ylim(bottom=-sum(table_heights), top=0.0)
    axis.axis("off")

    top = 0.0
    for table in tables:
        # Title, then the header row on a dark band
        axis.text(x=-0.2, y=top - 0.45, s=table["title"], fontsize=16, fontweight="bold", color=TEXT_COLOUR,
                  family=FIGURE_FONT_FAMILIES, va="center")
        header_top = top - 0.9
        axis.add_patch(Rectangle(xy=(-0.2, header_top - ROW_HEIGHT), width=total_width + 0.4,
                                                    height=ROW_HEIGHT, color=HEADER_COLOUR))
        for index, column in enumerate(columns):
            axis.text(x=left_edges[index] + 0.05 if index == 0 else left_edges[index] + column_widths[index] - 0.08,
                      y=header_top - ROW_HEIGHT / 2, s=column, ha="left" if index == 0 else "right", va="center",
                      fontsize=11, fontweight="bold", color="white", family=FIGURE_FONT_FAMILIES)

        # Rows: the row of all the objects on a grey band, the sizes striped, every recall on its shaded cell
        for row_index, row in enumerate(table["rows"]):
            row_top = header_top - (row_index + 1) * ROW_HEIGHT
            band_colour = TOTAL_ROW_COLOUR if row_index == 0 else (STRIPE_COLOUR if row_index % 2 == 0 else "white")
            axis.add_patch(Rectangle(xy=(-0.2, row_top - ROW_HEIGHT), width=total_width + 0.4,
                                                        height=ROW_HEIGHT, color=band_colour))
            for index, (column, value) in enumerate(zip(columns, row, strict=True)):
                if column in table["percent_columns"] and isinstance(value, float):
                    shade_range = RECALL_COLOUR_RANGE[1] - RECALL_COLOUR_RANGE[0]
                    axis.add_patch(Rectangle(
                        xy=(left_edges[index] + 0.04, row_top - ROW_HEIGHT + 0.07),
                        width=column_widths[index] - 0.06, height=ROW_HEIGHT - 0.14, linewidth=0,
                        color=colour_map(RECALL_COLOUR_RANGE[0] + shade_range * value), alpha=RECALL_CELL_OPACITY))
                axis.text(x=left_edges[index] + 0.05 if index == 0 else left_edges[index] + column_widths[index] - 0.08,
                          y=row_top - ROW_HEIGHT / 2,
                          s=format_figure_cell(value=value, column=column, percent_columns=table["percent_columns"]),
                          ha="left" if index == 0 else "right", va="center", fontsize=11, color=TEXT_COLOUR,
                          fontweight="bold" if row_index == 0 or index == len(columns) - 1 else "normal",
                          family=FIGURE_FONT_FAMILIES)

        # Line under the row of all the objects, then the legend wrapped under the table
        separator_height = header_top - 2 * ROW_HEIGHT
        axis.plot([-0.2, total_width + 0.2], [separator_height, separator_height], color=SEPARATOR_COLOUR,
                  linewidth=1.2)
        legend = textwrap.fill(text=table["legend"], width=int(LEGEND_CHARACTERS_PER_INCH * total_width))
        axis.text(x=-0.2, y=header_top - (len(table["rows"]) + 1) * ROW_HEIGHT - 0.2, s=legend, va="top",
                  fontsize=9, color=LEGEND_COLOUR, family=FIGURE_FONT_FAMILIES, linespacing=1.5)
        top -= 1.0 + ROW_HEIGHT * (len(table["rows"]) + 1) + 1.1

    # Saved without the margins left around the tables
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(fname=output_path, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    return output_path
