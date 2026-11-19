"""
Evaluation figures of the instance segmentation tutorials (the chapter on scoring and shipping masks), made at every
build by tutorials/figures/instance_segmentation/figures.py from the last evaluation of the INS run: the AP@0.75 of
every class on the masks against the boxes of the same detections (class_metrics.csv, its mask_ columns against the box
columns), where the 28 x 28 masks lose the thin parts of the shapes, and the listings and values of the mask and box
tables of the report of that checkpoint (Results.txt): the methods, the object sizes counted on the mask areas and on
the box areas, and the mean average recall of both. The checkpoint is the one evaluated again with the current code
(EVALUATION_FOLDER_NAME of detection_sources.py), whose report writes the rates as percentages ("78.64 %"), read back
between 0 and 1 in the facts; the figure writes its average precisions on an axis of percentages.
"""

import math

import numpy

from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (DOTS_PER_INCH, POINTS_PER_PIXEL, SERIES_COLOURS,
                                                TEXT_SECONDARY_COLOUR, compute_pixel_box, register_fonts, round_fact,
                                                save_figure, style_chart_panel)
from tutorials.figures.object_detection.detection_sources import (describe_class_name, extract_table,
                                                                  parse_table_rows, read_results_block, write_listing)
from tutorials.figures.object_detection.evaluation_figures import convert_report_value, read_class_metrics
from vision_training.evaluation.report_formatting import format_percentage_tick

# Tables of the report of an instance segmentation checkpoint, with the names of their listings and facts
REPORT_TABLES = [("Mask Mean Average Precision", "mask_methods"), ("Box Mean Average Precision", "box_methods"),
                 ("Mask Mean Average Precision By Object Size (Area)", "mask_sizes"),
                 ("Box Mean Average Precision By Object Size (Area)", "box_sizes"),
                 ("Mask Mean Average Recall", "mask_recall"), ("Box Mean Average Recall", "box_recall")]

# Step the axes of the object size bars end on, above the largest count of the mask and box tables
SIZE_AXIS_STEP = 1000

# Size of the charts: the card of an image scene
CHART_WIDTH = 1720
CHART_HEIGHT = 700

# Prefix of the names of the hollow shapes of the synthetic data, and the pointed shape the chapter names, whose class
# claims check_class_claims verifies
HOLLOW_PREFIX = "hollow_"
POINTED_CLASS_NAME = "star"


def check_class_claims(rows: List[Dict[str, Any]]) -> None:
    """
    Checks the claims the chapter on scoring masks makes about the classes, so that a run trained again stops the build
    instead of being described wrongly: at an IoU of 0.75 the class losing most on its masks is a hollow shape, every
    hollow shape loses and the star loses; at 0.95 the class gaining most on its masks is a filled shape, and more
    filled shapes gain than lose.

    Args:
        rows (List[Dict[str, Any]]): The rows of class_metrics.csv, sorted by the mask minus box AP@0.75.

    Raises:
        ValueError: If one of the claims does not hold.
    """
    hollow_rows = [row for row in rows if row["class"].startswith(HOLLOW_PREFIX)]
    filled_rows = [row for row in rows if not row["class"].startswith(HOLLOW_PREFIX)]
    star_rows = [row for row in rows if row["class"] == POINTED_CLASS_NAME]
    largest_gain_row = max(rows, key=lambda row: row["mask_average_precision_95"] - row["average_precision_95"])
    gaining_filled = sum(row["mask_average_precision_95"] > row["average_precision_95"] for row in filled_rows)
    problems = []
    if not rows[0]["class"].startswith(HOLLOW_PREFIX):
        problems.append(f"the largest mask loss at 0.75 is the {rows[0]['class']}, not a hollow shape")
    if any(row["mask_average_precision_75"] >= row["average_precision_75"] for row in hollow_rows):
        problems.append("a hollow shape does not lose on its masks at 0.75")
    if len(star_rows) != 1 or star_rows[0]["mask_average_precision_75"] >= star_rows[0]["average_precision_75"]:
        problems.append(f"the {POINTED_CLASS_NAME} does not lose on its masks at 0.75")
    if largest_gain_row["class"].startswith(HOLLOW_PREFIX) or (largest_gain_row["mask_average_precision_95"]
                                                                <= largest_gain_row["average_precision_95"]):
        problems.append(f"the largest mask gain at 0.95 is not a filled shape gaining ({largest_gain_row['class']})")
    if 2 * gaining_filled <= len(filled_rows):
        problems.append(f"only {gaining_filled} of the {len(filled_rows)} filled shapes gain on their masks at 0.95")
    if problems:
        raise ValueError("The class claims of the chapter on scoring masks do not hold: " + "; ".join(problems) + ".")


def draw_mask_box_figure(evaluation_folder: Path, output_path: Path) -> Dict[str, Any]:
    """
    Draws the AP@0.75 of every class on the masks next to the AP@0.75 of the boxes of the same detections, the classes
    sorted by the difference, the masks losing most first, and gives the values of the classes the narration quotes,
    the class losing most at 0.75 and the class gaining most at 0.95 named by their role, after checking the claims the
    chapter makes about them (check_class_claims).

    Args:
        evaluation_folder (Path): The evaluation folder of the last checkpoint of the INS run.
        output_path (Path): The PNG file of the figure.

    Returns:
        Dict[str, Any]: The box and mask AP of every class at 0.5, 0.75 and 0.95, those of the classes named by their
            role, and the box of the chart, named classes.*.
    """
    rows = read_class_metrics(evaluation_folder=evaluation_folder)
    differences = [row["mask_average_precision_75"] - row["average_precision_75"] for row in rows]
    order = numpy.argsort(a=differences, kind="stable")
    rows = [rows[index] for index in order]
    check_class_claims(rows=rows)

    # Pairs of bars, the masks then the boxes, for every class
    register_fonts()
    figure, axes = pyplot.subplots(figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH))
    # Left margin leaving room for the percentage ticks and the name of the axis on their left, bottom margin for the
    # slanted names of the classes
    figure.subplots_adjust(left=0.065, right=0.995, top=0.9, bottom=0.27)
    positions = numpy.arange(len(rows))
    axes.bar(x=positions - 0.2, height=[row["mask_average_precision_75"] for row in rows], width=0.4,
             color=SERIES_COLOURS[3], label="mask AP@0.75")
    axes.bar(x=positions + 0.2, height=[row["average_precision_75"] for row in rows], width=0.4,
             color=SERIES_COLOURS[0], label="box AP@0.75")
    axes.set_xticks(ticks=positions, labels=[describe_class_name(class_name=row["class"]) for row in rows], rotation=50,
                    ha="right", rotation_mode="anchor", fontsize=15 * POINTS_PER_PIXEL)
    style_chart_panel(axes=axes, title="AP@0.75 of every class, masks against the boxes of the same detections",
                      x_label="", y_label="AP@0.75")
    axes.set_ylim(bottom=0, top=1.0)
    axes.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))
    axes.set_xlim(left=-0.6, right=len(rows) - 0.4)
    axes.grid(visible=False, axis="x")
    axes.legend(loc="upper right", frameon=False, labelcolor=TEXT_SECONDARY_COLOUR, fontsize=18 * POINTS_PER_PIXEL)
    figure.canvas.draw()
    facts = {"classes.figure_box": compute_pixel_box(figure=figure, axes=axes, with_labels=True, margin=8),
             "classes.largest_loss_name": rows[0]["class"], "classes.count": len(rows)}

    # Classes named by their role: the largest mask loss at 0.75 and the largest mask gain at 0.95, as said aloud
    largest_gain_row = max(rows, key=lambda row: row["mask_average_precision_95"] - row["average_precision_95"])
    for role, row in [("largest_loss", rows[0]), ("largest_gain_95", largest_gain_row)]:
        facts.update({f"classes.{role}.name_text": describe_class_name(class_name=row["class"]),
                      f"classes.{role}.mask_ap_75": round_fact(value=row["mask_average_precision_75"]),
                      f"classes.{role}.box_ap_75": round_fact(value=row["average_precision_75"]),
                      f"classes.{role}.mask_ap_95": round_fact(value=row["mask_average_precision_95"]),
                      f"classes.{role}.box_ap_95": round_fact(value=row["average_precision_95"])})
    for row in rows:
        for column, name in [("average_precision_50", "box_ap_50"), ("average_precision_75", "box_ap_75"),
                             ("average_precision_95", "box_ap_95"), ("mask_average_precision_50", "mask_ap_50"),
                             ("mask_average_precision_75", "mask_ap_75"),
                             ("mask_average_precision_95", "mask_ap_95")]:
            facts[f"classes.{row['class']}.{name}"] = round_fact(value=row[column])
    save_figure(figure=figure, output_path=output_path)
    pyplot.close(fig=figure)
    return facts


def write_mask_report_listings(results_path: Path, iteration: int, output_folder: Path) -> Dict[str, Any]:
    """
    Writes the mask and box tables of the report of one checkpoint of the INS run as listings, and gives their values
    as facts for the trained weights and their moving average.

    Args:
        results_path (Path): The Results.txt holding the report, that of the evaluation made again with the current
            code (EVALUATION_FOLDER_NAME of detection_sources.py), whose tables write the rates as percentages.
        iteration (int): The evaluated checkpoint.
        output_folder (Path): The listings folder.

    Returns:
        Dict[str, Any]: The values of the tables, named report.<weights>.<table>.<row>.<column index>, the object
            counts of every size by mask and by box area with their labels, and the common end of the axes of their
            bars (report.sizes_axis_maximum).
    """
    facts = {}
    for averaged_weights, weights_name in [(False, "trained"), (True, "averaged")]:
        block_lines = read_results_block(results_path=results_path, iteration=iteration,
                                         averaged_weights=averaged_weights)
        for title, table_name in REPORT_TABLES:
            table_lines = extract_table(block_lines=block_lines, title=title)
            if not averaged_weights:
                write_listing(lines=table_lines, output_path=output_folder / f"results_{table_name}.txt")
            for row_label, values in parse_table_rows(table_lines=table_lines).items():
                row_key = row_label.split(" ")[0].lower()
                for value_index, value in enumerate(values):
                    facts[f"report.{weights_name}.{table_name}.{row_key}.{value_index}"] = convert_report_value(
                        value=value)

    # Objects of every size by mask area and by box area, the counts of the bars of the size scene
    for table_name in ["mask_sizes", "box_sizes"]:
        facts[f"report.{table_name}_counts"] = [facts[f"report.trained.{table_name}.{size}.0"] for size in
                                                ["small", "medium", "large"]]
        facts[f"report.{table_name}_count_labels"] = [f"{count:,}" for count in facts[f"report.{table_name}_counts"]]

    # Common end of the axes of both bar charts, the largest count rounded up to the next SIZE_AXIS_STEP
    largest_count = max(facts["report.mask_sizes_counts"] + facts["report.box_sizes_counts"])
    facts["report.sizes_axis_maximum"] = SIZE_AXIS_STEP * math.ceil(largest_count / SIZE_AXIS_STEP)
    return facts
