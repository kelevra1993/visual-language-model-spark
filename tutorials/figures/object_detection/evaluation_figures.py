"""
Evaluation figures of the object detection tutorials (the chapter on the evaluation and the export), made at every
build by tutorials/figures/object_detection/figures.py from the evaluation folder of the last checkpoint of the DET
run and from its Results.txt: the average precision of every class at IoU 0.5 and 0.75 with the score threshold and
recall of the class at the target precision (class_metrics.csv, class_thresholds.json), the confusion matrix of the
detections reaching the deployment threshold (confusion_matrix.csv) with its background row of false positives and
column of missed objects, the listings of the tables of the report of that checkpoint (methods, object sizes, mean
average recall), and the proposal recall of every block (proposal_recall.csv). The values the narration quotes are
read from the same files and given as facts, the rates of the report read back between 0 and 1 whether it wrote them as
percentages ("78.64 %") or with decimals ("0.7864"); the figures and bar labels write the rates as the percentages of
the reports of the repository (the target precision "99 %"), the score thresholds keeping their decimals.
"""

import csv
import json
import operator
import re

import numpy

from matplotlib import pyplot
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (DOTS_PER_INCH, POINTS_PER_PIXEL, SERIES_COLOURS, SURFACE_COLOUR,
                                                TEXT_PRIMARY_COLOUR, TEXT_SECONDARY_COLOUR, compute_pixel_box,
                                                register_fonts, round_fact, save_figure, style_chart_panel)
from tutorials.figures.object_detection.detection_sources import (describe_class_name, extract_table, join_words,
                                                                  parse_table_rows, read_results_block, write_listing)
from tutorial_builder.layout import NAMED_COLOURS
from vision_training.evaluation.classification.class_metrics_report import write_target_precision
from vision_training.evaluation.classification.confusion_matrix import MAXIMUM_CLASSES_WITH_SHARES
from vision_training.evaluation.detection.detection_confusion import MATCHING_INTERSECTION_OVER_UNION_THRESHOLD
from vision_training.evaluation.report_formatting import (PERCENT_DECIMALS, PERCENT_SIGN, format_percentage,
                                                          format_percentage_tick)

# Tables of the report of a checkpoint shown as listings, with the names of their files
REPORT_TABLES = [("Mean Average Precision", "results_methods.txt"),
                 ("Mean Average Precision By Object Size (Area)", "results_sizes.txt"),
                 ("Mean Average Recall", "results_recall.txt")]

# Decimals of a rate read back from its percentage, the two decimals of the percentage and the two of the division
RATE_DECIMALS = PERCENT_DECIMALS + 2

# Size of the charts: the card of an image scene
CHART_WIDTH = 1720
CHART_HEIGHT = 700

# Classes of every panel of the class chart, the 35 classes of the synthetic shapes going over two panels side by side
CLASSES_PER_PANEL = 18

# Positions of the threshold and recall columns beyond the average precision bars of a panel, in units of its AP axis,
# far enough right that the percentage of the longest bar ("98.05 %") ends before the threshold column and that the
# threshold heading ends before the recall column
THRESHOLD_COLUMN_POSITIONS = [1.16, 1.42]

# Most classes, the background left out, whose confusion matrix cells are labelled with their count, the cells of more
# classes being too small to hold one: the limit beyond which the confusion matrices of the evaluation drop their shares
MATRIX_COUNTS_MAXIMUM_CLASSES = MAXIMUM_CLASSES_WITH_SHARES

# Colours of the confusion matrix: the empty cells, the cells holding the whole row, and the outline of the
# background row and column
MATRIX_LOW_COLOUR = "#%02x%02x%02x" % NAMED_COLOURS["box"]
MATRIX_HIGH_COLOUR = SERIES_COLOURS[0]
BACKGROUND_OUTLINE_COLOUR = "#%02x%02x%02x" % NAMED_COLOURS["accent"]


def read_class_metrics(evaluation_folder: Path) -> List[Dict[str, Any]]:
    """
    Reads the average precisions of every class from the class_metrics.csv of an evaluation, the box columns of the
    detection (and the mask_ columns of the instance segmentation) as numbers, an empty value as None.

    Args:
        evaluation_folder (Path): The evaluation folder of a checkpoint.

    Returns:
        List[Dict[str, Any]]: One row per class, in the order of the file.
    """
    with open(file=evaluation_folder / "class_metrics.csv", mode="r", encoding="utf-8") as file_handler:
        rows = list(csv.DictReader(f=file_handler))
    for row in rows:
        for key, value in row.items():
            if key != "class":
                row[key] = float(value) if value != "" else None
    return rows


def draw_class_average_precision_figure(evaluation_folder: Path, output_path: Path) -> Dict[str, Any]:
    """
    Draws the AP@0.5 of every class, worst first, as horizontal bars, the classes split over panels side by side
    (CLASSES_PER_PANEL in each, so that the 35 classes of the synthetic shapes keep readable rows), with the AP@0.75 of
    the class as a mark on its bar, and next to the bars the score threshold of the class at the target precision with
    the recall it keeps, read from class_thresholds.json, a dash for a class whose detections never reach the
    precision; the average precisions, the recall and the target precision are written as percentages, the score
    thresholds with their decimals, the class names as said. The classes the narration names are given by their role:
    the worst and the best by AP@0.5, the class keeping the most recall at the target precision, and the classes that
    never reach it.

    Args:
        evaluation_folder (Path): The evaluation folder of the last checkpoint.
        output_path (Path): The PNG file of the figure.

    Returns:
        Dict[str, Any]: The worst and best classes with their values, the threshold and recall of every class reaching
            the target precision, the class keeping the most recall at it, the classes never reaching it, and the box
            of the chart, named classes.*.
    """
    rows = sorted(read_class_metrics(evaluation_folder=evaluation_folder),
                  key=operator.itemgetter("average_precision_50"))
    thresholds = json.loads((evaluation_folder / "class_thresholds.json").read_text(encoding="utf-8"))
    target_precision = thresholds["target_precision"]
    target_precision_text = write_target_precision(target_precision=target_precision, in_percent=True)

    # Panels of at most CLASSES_PER_PANEL classes, the worst class at the top of the first one
    register_fonts()
    panel_rows = [rows[start:start + CLASSES_PER_PANEL] for start in range(0, len(rows), CLASSES_PER_PANEL)]
    figure, axes_list = pyplot.subplots(nrows=1, ncols=len(panel_rows), squeeze=False,
                                        figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH))
    axes_list = axes_list[0]
    # Margins leaving the class names on the left of every panel, and the threshold and recall columns on its right
    figure.subplots_adjust(left=0.105, right=0.837, top=0.87, bottom=0.09, wspace=1.1)
    figure.suptitle(t=f"Average precision of every class, iteration {int(evaluation_folder.name.split('_')[-1]):,}",
                    x=0.01, ha="left", color=TEXT_PRIMARY_COLOUR, fontsize=24 * POINTS_PER_PIXEL)
    for panel_index, (axes, rows_of_panel) in enumerate(zip(axes_list, panel_rows)):
        # Bars of the AP@0.5 and the AP@0.75 as a mark on every bar, the rows of a shorter last panel kept at the
        # height of the others
        positions = numpy.arange(len(rows_of_panel))[::-1] + (CLASSES_PER_PANEL - len(rows_of_panel))
        axes.barh(y=positions, width=[row["average_precision_50"] for row in rows_of_panel], height=0.68,
                  color=SERIES_COLOURS[0], label="AP@0.5")
        axes.scatter(x=[row["average_precision_75"] for row in rows_of_panel], y=positions, marker="|", s=300,
                     linewidths=3 * POINTS_PER_PIXEL, color=SERIES_COLOURS[2], zorder=3, label="AP@0.75")
        style_chart_panel(axes=axes, title="", x_label="AP", y_label=None)
        axes.set_yticks(ticks=positions, labels=[describe_class_name(class_name=row["class"]) for row in rows_of_panel])
        axes.tick_params(axis="y", labelsize=15 * POINTS_PER_PIXEL, pad=6)
        axes.tick_params(axis="x", labelsize=15 * POINTS_PER_PIXEL)
        axes.set_xlim(left=0, right=1.0)
        axes.xaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))
        axes.set_ylim(bottom=-0.6, top=CLASSES_PER_PANEL - 0.4)
        axes.grid(visible=False, axis="y")
        # Legend of the marks once, on the row of the title, clear of the bars
        if panel_index == len(panel_rows) - 1:
            figure.legend(handles=axes.get_legend_handles_labels()[0], labels=axes.get_legend_handles_labels()[1],
                          loc="upper right", bbox_to_anchor=(0.99, 0.995), ncol=2, frameon=False,
                          labelcolor=TEXT_SECONDARY_COLOUR, fontsize=16 * POINTS_PER_PIXEL)

        # Value of every bar and, beyond the bars, the threshold of the class at the target precision and its recall,
        # the target precision written like the headers of the report ("P99 %") under each column heading
        for column_position, column_title in zip(THRESHOLD_COLUMN_POSITIONS,
                                                 [f"threshold\n@ P{target_precision_text}",
                                                  f"recall\n@ P{target_precision_text}"]):
            axes.text(x=column_position, y=CLASSES_PER_PANEL - 0.25, s=column_title, transform=axes.transData,
                      color=TEXT_PRIMARY_COLOUR, fontsize=15 * POINTS_PER_PIXEL, va="bottom", ha="left",
                      clip_on=False)
        for position, row in zip(positions, rows_of_panel):
            axes.text(x=row["average_precision_50"] + 0.012, y=position,
                      s=format_percentage(value=row["average_precision_50"]), color=TEXT_SECONDARY_COLOUR,
                      fontsize=13 * POINTS_PER_PIXEL, va="center", ha="left")
            class_threshold = thresholds["classes"][row["class"]]
            column_texts = (["—", "—"] if class_threshold is None else
                            [f"{class_threshold['threshold']:.3f}", format_percentage(value=class_threshold["recall"])])
            for column_position, column_text in zip(THRESHOLD_COLUMN_POSITIONS, column_texts):
                axes.text(x=column_position, y=position, s=column_text, color=TEXT_SECONDARY_COLOUR,
                          fontsize=13 * POINTS_PER_PIXEL, va="center", ha="left", clip_on=False, family="Hack")
    figure.canvas.draw()

    # Classes named by their role: the worst and best by AP@0.5, the class keeping the most recall at the target
    # precision, and the classes never reaching it, in the order of the chart
    thresholded_rows = [row for row in rows if thresholds["classes"][row["class"]] is not None]
    most_recall_row = max(thresholded_rows, key=lambda row: thresholds["classes"][row["class"]]["recall"])
    without_threshold = [row["class"] for row in rows if thresholds["classes"][row["class"]] is None]
    facts = {"classes.figure_box": [0, 0, CHART_WIDTH, CHART_HEIGHT],
             "classes.worst_name": rows[0]["class"], "classes.worst_ap_50": round_fact(
                 value=rows[0]["average_precision_50"]),
             "classes.worst_name_text": describe_class_name(class_name=rows[0]["class"]),
             "classes.best_name": rows[-1]["class"], "classes.best_ap_50": round_fact(
                 value=rows[-1]["average_precision_50"]),
             "classes.best_name_text": describe_class_name(class_name=rows[-1]["class"]),
             "classes.target_precision": target_precision,
             "classes.most_recall.name_text": describe_class_name(class_name=most_recall_row["class"]),
             "classes.most_recall.threshold": round_fact(value=thresholds["classes"][most_recall_row["class"]][
                 "threshold"]),
             "classes.most_recall.recall": round_fact(value=thresholds["classes"][most_recall_row["class"]]["recall"]),
             "classes.without_threshold": without_threshold,
             "classes.without_threshold_count": len(without_threshold),
             "classes.without_threshold_text": join_words(words=[describe_class_name(class_name=class_name)
                                                                 for class_name in without_threshold])}

    # Values of every class by name, its threshold and recall only when it reaches the target precision
    for row in rows:
        facts[f"classes.{row['class']}.ap_50"] = round_fact(value=row["average_precision_50"])
        facts[f"classes.{row['class']}.ap_75"] = round_fact(value=row["average_precision_75"])
        class_threshold = thresholds["classes"][row["class"]]
        if class_threshold is not None:
            facts[f"classes.{row['class']}.threshold"] = round_fact(value=class_threshold["threshold"])
            facts[f"classes.{row['class']}.recall"] = round_fact(value=class_threshold["recall"])
    save_figure(figure=figure, output_path=output_path)
    pyplot.close(fig=figure)
    return facts


def draw_confusion_matrix_figure(evaluation_folder: Path, score_threshold: float,
                                 output_path: Path) -> Dict[str, Any]:
    """
    Draws the confusion matrix of the detections reaching the deployment threshold from confusion_matrix.csv, rows the
    true classes and columns the predicted ones, the background first, every cell coloured by its share of its row and,
    for at most MATRIX_COUNTS_MAXIMUM_CLASSES classes, labelled with its count (too small to read beyond, as the
    matrices of the evaluation drop their shares), the background row (false positives) and column (missed objects)
    and the cell of the largest confusion between two shapes outlined; and gives the totals of the four outcomes and
    the recall of every class at the threshold. The caption writes the deployment threshold of the run and the overlap
    of the matching (MATCHING_INTERSECTION_OVER_UNION_THRESHOLD of detection_confusion.py), the values the evaluation
    counted the matrix with.

    Args:
        evaluation_folder (Path): The evaluation folder of the last checkpoint.
        score_threshold (float): 'Detector.inference.score_threshold' of the run, the deployment threshold.
        output_path (Path): The PNG file of the figure.

    Returns:
        Dict[str, Any]: The outcomes, the classes never right at the threshold, the largest confusion between two
            shapes and the boxes of the background row and column and of its cell, named confusion.*.
    """
    with open(file=evaluation_folder / "confusion_matrix.csv", mode="r", encoding="utf-8") as file_handler:
        table = list(csv.reader(file_handler))
    class_names = table[0][1:]
    counts = numpy.array(object=[[int(value) for value in row[1:]] for row in table[1:]])

    # Share of every cell in its row, the background row by its total of false positives
    row_totals = counts.sum(axis=1, keepdims=True)
    shares = numpy.divide(counts, numpy.maximum(row_totals, 1))
    register_fonts()
    figure, axes = pyplot.subplots(figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH))
    figure.subplots_adjust(left=0.08, right=0.56, top=0.8, bottom=0.02)
    colour_map = LinearSegmentedColormap.from_list(name="matrix", colors=[MATRIX_LOW_COLOUR, MATRIX_HIGH_COLOUR])
    axes.imshow(X=shares, cmap=colour_map, vmin=0.0, vmax=1.0, aspect="equal", interpolation="nearest")
    if len(class_names) <= MATRIX_COUNTS_MAXIMUM_CLASSES + 1:
        for row_index in range(counts.shape[0]):
            for column_index in range(counts.shape[1]):
                if counts[row_index, column_index] > 0:
                    axes.text(x=column_index, y=row_index, s=str(counts[row_index, column_index]), ha="center",
                              va="center", color=TEXT_PRIMARY_COLOUR, fontsize=8.5 * POINTS_PER_PIXEL * 1.39)

    # Class names on both axes as said, the predicted classes above the matrix
    label_names = [describe_class_name(class_name=class_name) for class_name in class_names]
    axes.set_xticks(ticks=range(len(class_names)), labels=label_names, rotation=90)
    axes.set_yticks(ticks=range(len(class_names)), labels=label_names)
    axes.xaxis.tick_top()
    axes.tick_params(axis="both", colors=TEXT_SECONDARY_COLOUR, labelsize=11 * POINTS_PER_PIXEL, length=0, pad=3)
    axes.set_facecolor(SURFACE_COLOUR)
    for side in axes.spines.values():
        side.set_visible(False)

    # Background row and column outlined, the false positives and the missed objects
    for rectangle_origin, width, height in [((-0.5, -0.5), counts.shape[1], 1.0), ((-0.5, -0.5), 1.0,
                                                                                   counts.shape[0])]:
        axes.add_patch(Rectangle(xy=rectangle_origin, width=width, height=height, fill=False,
                                 edgecolor=BACKGROUND_OUTLINE_COLOUR, linewidth=3 * POINTS_PER_PIXEL))

    # Largest confusion between two shapes, read from the matrix rather than named, so that a run trained again keeps
    # a true sentence, its cell outlined: the class of the objects, the class of their detections and the count
    wrong_class_counts = counts[1:, 1:].copy()
    numpy.fill_diagonal(a=wrong_class_counts, val=0)
    true_index, predicted_index = numpy.unravel_index(indices=int(wrong_class_counts.argmax()),
                                                      shape=wrong_class_counts.shape)
    largest_cell = (int(predicted_index) + 1, int(true_index) + 1)
    axes.add_patch(Rectangle(xy=(largest_cell[0] - 0.5, largest_cell[1] - 0.5), width=1.0, height=1.0, fill=False,
                             edgecolor=SERIES_COLOURS[2], linewidth=3 * POINTS_PER_PIXEL))

    # Totals of the four outcomes on the right of the matrix, the background row and column first
    diagonal = numpy.diag(counts)[1:]
    outcomes = [("right class", int(diagonal.sum()), SERIES_COLOURS[0]),
                ("wrong class", int(counts[1:, 1:].sum() - diagonal.sum()), SERIES_COLOURS[4]),
                ("false positives: background row", int(counts[0, 1:].sum()), BACKGROUND_OUTLINE_COLOUR),
                ("missed: background column", int(counts[1:, 0].sum()), BACKGROUND_OUTLINE_COLOUR)]
    for outcome_index, (outcome_name, outcome_count, outcome_colour) in enumerate(outcomes):
        figure.text(x=0.62, y=0.78 - outcome_index * 0.17, s=f"{outcome_count:,}", color=outcome_colour,
                    fontsize=44 * POINTS_PER_PIXEL, weight="bold")
        figure.text(x=0.62, y=0.73 - outcome_index * 0.17, s=outcome_name, color=TEXT_SECONDARY_COLOUR,
                    fontsize=20 * POINTS_PER_PIXEL)

    # Caption: the deployment threshold of the run and the overlap matching a detection to an object
    figure.text(x=0.62, y=0.9, s=f"Detections with score >= {score_threshold:g}, IoU >= "
                                 f"{MATCHING_INTERSECTION_OVER_UNION_THRESHOLD:g}", color=TEXT_PRIMARY_COLOUR,
                fontsize=22 * POINTS_PER_PIXEL)
    figure.canvas.draw()

    # Pixel boxes of the background row and column, from the data coordinates of the matrix
    figure_height = figure.bbox.height
    facts = {}
    for name, (data_left, data_top, data_right, data_bottom) in [
            ("background_row", (-0.5, -0.5, counts.shape[1] - 0.5, 0.5)),
            ("background_column", (-0.5, -0.5, 0.5, counts.shape[0] - 0.5)),
            ("largest_cell", (largest_cell[0] - 0.5, largest_cell[1] - 0.5, largest_cell[0] + 0.5,
                              largest_cell[1] + 0.5))]:
        (pixel_left, pixel_top), (pixel_right, pixel_bottom) = axes.transData.transform(
            values=[(data_left, data_top), (data_right, data_bottom)])
        facts[f"confusion.{name}_box"] = [int(pixel_left) - 4, int(figure_height - pixel_top) - 4,
                                          int(pixel_right) + 4, int(figure_height - pixel_bottom) + 4]
    facts["confusion.matrix_box"] = compute_pixel_box(figure=figure, axes=axes, with_labels=True, margin=6)

    # Outcomes of the detections at the threshold, and the classes never found with their class
    never_right = [class_names[index + 1] for index, value in enumerate(diagonal) if value == 0]
    never_right_names = [describe_class_name(class_name=class_name) for class_name in never_right]
    facts.update({"confusion.right": int(diagonal.sum()),
                  "confusion.wrong_class": int(counts[1:, 1:].sum() - diagonal.sum()),
                  "confusion.false_positives": int(counts[0, 1:].sum()), "confusion.missed": int(counts[1:, 0].sum()),
                  "confusion.never_right": never_right, "confusion.never_right_count": len(never_right),
                  "confusion.never_right_text": join_words(words=never_right_names) if never_right_names else ""})
    for class_index, class_name in enumerate(class_names[1:]):
        facts[f"confusion.{class_name}.recall"] = round_fact(value=diagonal[class_index] /
                                                             max(int(row_totals[class_index + 1, 0]), 1))

    # Largest confusion between two shapes with the right detections of its class, as said
    facts.update({"confusion.largest.true_name": class_names[true_index + 1],
                  "confusion.largest.predicted_name": class_names[predicted_index + 1],
                  "confusion.largest.true_name_text": describe_class_name(class_name=class_names[true_index + 1]),
                  "confusion.largest.predicted_name_text": describe_class_name(
                      class_name=class_names[predicted_index + 1]),
                  "confusion.largest.count": int(wrong_class_counts[true_index, predicted_index]),
                  "confusion.largest.right_count": int(diagonal[true_index])})
    save_figure(figure=figure, output_path=output_path)
    pyplot.close(fig=figure)
    return facts


def convert_report_value(value: str) -> Any:
    """
    Turns a value of a table of the report into a number: an integer for a count of objects, a float between 0 and 1
    for an average precision or recall, whether the report wrote it as a percentage ("78.64 %", read back as 0.7864) or
    with decimals ("0.7864", the reports written before the percentages), so that the facts keep one scale whichever
    code evaluated the run, and the dash of a size without object kept as it is.

    Args:
        value (str): The value as printed in Results.txt (parse_table_rows).

    Returns:
        Any: The integer, float or dash.
    """
    if value.isdigit():
        return int(value)
    if value == "—":
        return value
    # A percentage divided back to its rate, rounded to the decimals it was written with
    if value.endswith(PERCENT_SIGN):
        return round(float(value[:-len(PERCENT_SIGN)]) / 100.0, RATE_DECIMALS)
    return float(value)


def write_report_listings(results_path: Path, iteration: int, output_folder: Path) -> Dict[str, Any]:
    """
    Writes the tables of the report of one checkpoint (Results.txt) shown by the listings of the evaluation scene,
    and gives their values as facts: the mAP of every method at every IoU with the number of methods, the objects and
    mAP of every size, and the mean average recall of every group, for the trained weights and for their moving
    average.

    Args:
        results_path (Path): The Results.txt holding the report, that of the evaluation made again with the current
            code (EVALUATION_FOLDER_NAME of detection_sources.py), whose tables write the rates as percentages.
        iteration (int): The evaluated checkpoint.
        output_folder (Path): The listings folder.

    Returns:
        Dict[str, Any]: The values of the tables, named report.*.
    """
    facts = {}
    for averaged_weights, weights_name in [(False, "trained"), (True, "averaged")]:
        block_lines = read_results_block(results_path=results_path, iteration=iteration,
                                         averaged_weights=averaged_weights)
        facts[f"report.{weights_name}.header"] = block_lines[0].split(" ─")[0]
        for title, file_name in REPORT_TABLES:
            table_lines = extract_table(block_lines=block_lines, title=title)
            if not averaged_weights:
                write_listing(lines=table_lines, output_path=output_folder / file_name)
            table_name = file_name.split(".")[0].split("_")[1]
            table_rows = parse_table_rows(table_lines=table_lines)

            # Number of rows of the methods table, one per method computing the mAP
            if table_name == "methods":
                facts[f"report.{weights_name}.method_count"] = len(table_rows)
            for row_label, values in table_rows.items():
                row_key = row_label.split(" ")[0].lower()
                for value_index, value in enumerate(values):
                    facts[f"report.{weights_name}.{table_name}.{row_key}.{value_index}"] = convert_report_value(
                        value=value)
        facts[f"report.{weights_name}.confusion_note"] = [line.strip() for line in block_lines if
                                                          "Confusion Matrix :" in line][0]
    return facts


def read_proposal_recall(evaluation_folder: Path) -> Dict[str, Any]:
    """
    Reads the proposal recall of one checkpoint (proposal_recall.csv): the share of the test objects overlapped by a
    proposal by 0.5 and 0.7, for all the objects, every size and the proposals of every block alone, with the mean
    number of valid proposals per image of every block, the values of the bars of the proposal recall scene (one per
    block row of the file, in the order of the blocks), the recall labels written as percentages.

    Args:
        evaluation_folder (Path): The evaluation folder of the checkpoint.

    Returns:
        Dict[str, Any]: The values, named proposal_recall.<group>.<column>, the lists of the bars and their block
            labels.
    """
    with open(file=evaluation_folder / "proposal_recall.csv", mode="r", encoding="utf-8") as file_handler:
        rows = {row["group"]: row for row in csv.DictReader(f=file_handler)}
    facts = {}
    for group, row in rows.items():
        for column in ["objects", "recall_50", "recall_70", "proposals_per_image"]:
            if row[column] != "":
                facts[f"proposal_recall.{group}.{column}"] = round_fact(value=float(row[column]))

    # Bars of the proposal recall scene: the proposals and the recall of every block in block order, with their
    # labels, then the recall of every object size
    blocks = sorted([group for group in rows if group.startswith("block_")], key=lambda group: int(group.split("_")[1]))
    facts["proposal_recall.block_labels"] = [block.replace("_", " ") for block in blocks]
    facts["proposal_recall.proposals_per_block"] = [round_fact(value=float(rows[block]["proposals_per_image"]))
                                                    for block in blocks]
    facts["proposal_recall.proposals_per_block_labels"] = [f"{float(rows[block]['proposals_per_image']):.1f}"
                                                           for block in blocks]
    facts["proposal_recall.recall_per_block"] = [round_fact(value=float(rows[block]["recall_50"])) for block in blocks]
    facts["proposal_recall.recall_per_block_labels"] = [format_percentage(value=float(rows[block]["recall_50"]))
                                                        for block in blocks]
    facts["proposal_recall.recall_per_size"] = [round_fact(value=float(rows[group]["recall_50"])) for group in
                                                ["small", "medium", "large", "all"]]
    facts["proposal_recall.recall_per_size_labels"] = [format_percentage(value=float(rows[group]["recall_50"]))
                                                       for group in ["small", "medium", "large", "all"]]
    return facts


def read_confusion_notes(run_folder: Path, iterations: List[int]) -> Dict[str, Any]:
    """
    Reads the confusion matrix note of the report of every evaluated iteration of a run (Results.txt, trained
    weights): the detections reaching the deployment threshold that are right, of the wrong class, false positives,
    and the objects missed, over the whole test split.

    Args:
        run_folder (Path): The run folder.
        iterations (List[int]): The evaluated iterations.

    Returns:
        Dict[str, Any]: The four counts of every iteration, named outcomes.<iteration>.<outcome>.

    Raises:
        ValueError: If a note does not have the four counts.
    """
    facts = {}
    for iteration in iterations:
        block_lines = read_results_block(results_path=run_folder / "Results.txt", iteration=iteration,
                                         averaged_weights=False)
        note = " ".join(line.strip() for line in block_lines[1:3])
        match = re.search(pattern=r"(\d+) right, (\d+) wrong class, (\d+) false positives, (\d+)\s+missed",
                          string=note)
        if match is None:
            raise ValueError(f"No confusion matrix counts in the note {note!r} of iteration {iteration}.")
        for outcome, value in zip(["right", "wrong_class", "false_positives", "missed"], match.groups()):
            facts[f"outcomes.{iteration}.{outcome}"] = int(value)
    return facts
