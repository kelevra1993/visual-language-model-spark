"""
Evaluation figures of the object detection tutorials (the chapter on the evaluation and the export), made at every
build by tutorials/figures/object_detection/figures.py from the evaluation folder of the last checkpoint of the DET
run and from its Results.txt: the average precision of every class at IoU 0.5 and 0.75 with the score threshold and
recall of the class at the target precision (class_metrics.csv, class_thresholds.json), the confusion matrix of the
detections reaching the deployment threshold (confusion_matrix.csv) with its background row of false positives and
column of missed objects, the listings of the tables of the report of that checkpoint (methods, object sizes, mean
average recall), and the proposal recall of every block (proposal_recall.csv). The values the narration quotes are
read from the same files and given as facts, the rates of the report read back between 0 and 1 whether it wrote them as
percentages ("78.64 %") or with decimals ("0.7864").
"""

import csv
import json
import operator
import re

import numpy

from matplotlib import pyplot
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (DOTS_PER_INCH, POINTS_PER_PIXEL, SERIES_COLOURS, SURFACE_COLOUR,
                                                TEXT_PRIMARY_COLOUR, TEXT_SECONDARY_COLOUR, compute_pixel_box,
                                                register_fonts, round_fact, save_figure, style_chart_panel)
from tutorials.figures.object_detection.detection_sources import (extract_table, parse_table_rows,
                                                                  read_results_block, write_listing)
from tutorial_builder.layout import NAMED_COLOURS
from vision_training.evaluation.detection.detection_confusion import MATCHING_INTERSECTION_OVER_UNION_THRESHOLD
from vision_training.evaluation.report_formatting import PERCENT_DECIMALS, PERCENT_SIGN

# Tables of the report of a checkpoint shown as listings, with the names of their files
REPORT_TABLES = [("Mean Average Precision", "results_methods.txt"),
                 ("Mean Average Precision By Object Size (Area)", "results_sizes.txt"),
                 ("Mean Average Recall", "results_recall.txt")]

# Decimals of a rate read back from its percentage, the two decimals of the percentage and the two of the division
RATE_DECIMALS = PERCENT_DECIMALS + 2

# Size of the charts: the card of an image scene
CHART_WIDTH = 1720
CHART_HEIGHT = 700

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
    Draws the AP@0.5 of every class, worst first, as horizontal bars, with the AP@0.75 of the class as a mark on its
    bar, and on the right the score threshold of the class at the target precision with the recall it keeps, read
    from class_thresholds.json, a dash for a class whose detections never reach the precision.

    Args:
        evaluation_folder (Path): The evaluation folder of the last checkpoint.
        output_path (Path): The PNG file of the figure.

    Returns:
        Dict[str, Any]: The worst and best classes with their values, the threshold and recall of some classes, and
            the box of the chart, named classes.*.
    """
    rows = sorted(read_class_metrics(evaluation_folder=evaluation_folder),
                  key=operator.itemgetter("average_precision_50"))
    thresholds = json.loads((evaluation_folder / "class_thresholds.json").read_text(encoding="utf-8"))

    # Bars of the AP@0.5, the worst class at the top, the AP@0.75 as a mark on every bar
    register_fonts()
    figure, axes = pyplot.subplots(figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH))
    figure.subplots_adjust(left=0.11, right=0.74, top=0.9, bottom=0.1)
    positions = numpy.arange(len(rows))[::-1]
    axes.barh(y=positions, width=[row["average_precision_50"] for row in rows], height=0.68, color=SERIES_COLOURS[0],
              label="AP@0.5")
    axes.scatter(x=[row["average_precision_75"] for row in rows], y=positions, marker="|", s=420,
                 linewidths=3 * POINTS_PER_PIXEL, color=SERIES_COLOURS[2], zorder=3, label="AP@0.75")
    axes.set_yticks(ticks=positions, labels=[row["class"] for row in rows])
    style_chart_panel(axes=axes, title="Average precision of every class, iteration "
                                       f"{int(evaluation_folder.name.split('_')[-1]):,}", x_label="AP", y_label=None)
    axes.set_xlim(left=0, right=1.0)
    axes.set_ylim(bottom=-0.6, top=len(rows) - 0.4)
    axes.grid(visible=False, axis="y")
    axes.legend(loc="upper right", frameon=False, labelcolor=TEXT_SECONDARY_COLOUR, fontsize=17 * POINTS_PER_PIXEL)

    # Value of every bar and, beyond the chart, the threshold of the class at the target precision and its recall
    target_precision = thresholds["target_precision"]
    for column_position, column_title in [(1.04, f"threshold @ P{target_precision}"), (1.2, "recall")]:
        axes.text(x=column_position, y=len(rows) - 0.1, s=column_title, transform=axes.transData,
                  color=TEXT_PRIMARY_COLOUR, fontsize=17 * POINTS_PER_PIXEL, va="bottom", ha="left", clip_on=False)
    for position, row in zip(positions, rows):
        axes.text(x=row["average_precision_50"] + 0.008, y=position, s=f"{row['average_precision_50']:.3f}",
                  color=TEXT_SECONDARY_COLOUR, fontsize=15 * POINTS_PER_PIXEL, va="center", ha="left")
        class_threshold = thresholds["classes"][row["class"]]
        column_texts = (["—", "—"] if class_threshold is None else
                        [f"{class_threshold['threshold']:.3f}", f"{class_threshold['recall']:.3f}"])
        for column_position, column_text in zip([1.04, 1.2], column_texts):
            axes.text(x=column_position, y=position, s=column_text, color=TEXT_SECONDARY_COLOUR,
                      fontsize=15 * POINTS_PER_PIXEL, va="center", ha="left", clip_on=False, family="Hack")
    figure.canvas.draw()
    facts = {"classes.figure_box": compute_pixel_box(figure=figure, axes=axes, with_labels=True, margin=8),
             "classes.worst_name": rows[0]["class"], "classes.worst_ap_50": round_fact(
                 value=rows[0]["average_precision_50"]),
             "classes.best_name": rows[-1]["class"], "classes.best_ap_50": round_fact(
                 value=rows[-1]["average_precision_50"]),
             "classes.target_precision": target_precision,
             "classes.without_threshold": [row["class"] for row in rows if thresholds["classes"][row["class"]] is None]}
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
    true classes and columns the predicted ones, the background first, every cell coloured by its share of its row and
    labelled with its count, the background row (false positives) and column (missed objects) outlined; and gives the
    totals of the four outcomes and the recall of every class at the threshold. The caption writes the deployment
    threshold of the run and the overlap of the matching (MATCHING_INTERSECTION_OVER_UNION_THRESHOLD of
    detection_confusion.py), the values the evaluation counted the matrix with.

    Args:
        evaluation_folder (Path): The evaluation folder of the last checkpoint.
        score_threshold (float): 'Detector.inference.score_threshold' of the run, the deployment threshold.
        output_path (Path): The PNG file of the figure.

    Returns:
        Dict[str, Any]: The outcomes, the classes never right at the threshold, the largest confusion between two
            shapes and the boxes of the background row and column, named confusion.*.
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
    figure.subplots_adjust(left=0.1, right=0.56, top=0.86, bottom=0.04)
    colour_map = LinearSegmentedColormap.from_list(name="matrix", colors=[MATRIX_LOW_COLOUR, MATRIX_HIGH_COLOUR])
    axes.imshow(X=shares, cmap=colour_map, vmin=0.0, vmax=1.0, aspect="equal", interpolation="nearest")
    for row_index in range(counts.shape[0]):
        for column_index in range(counts.shape[1]):
            if counts[row_index, column_index] > 0:
                axes.text(x=column_index, y=row_index, s=str(counts[row_index, column_index]), ha="center",
                          va="center", color=TEXT_PRIMARY_COLOUR, fontsize=8.5 * POINTS_PER_PIXEL * 1.39)

    # Class names on both axes, the predicted classes above the matrix
    axes.set_xticks(ticks=range(len(class_names)), labels=class_names, rotation=90)
    axes.set_yticks(ticks=range(len(class_names)), labels=class_names)
    axes.xaxis.tick_top()
    axes.tick_params(axis="both", colors=TEXT_SECONDARY_COLOUR, labelsize=13 * POINTS_PER_PIXEL, length=0, pad=4)
    axes.set_facecolor(SURFACE_COLOUR)
    for side in axes.spines.values():
        side.set_visible(False)

    # Background row and column outlined, the false positives and the missed objects
    for rectangle_origin, width, height in [((-0.5, -0.5), counts.shape[1], 1.0), ((-0.5, -0.5), 1.0,
                                                                                   counts.shape[0])]:
        axes.add_patch(Rectangle(xy=rectangle_origin, width=width, height=height, fill=False,
                                 edgecolor=BACKGROUND_OUTLINE_COLOUR, linewidth=3 * POINTS_PER_PIXEL))
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
            ("row", (-0.5, -0.5, counts.shape[1] - 0.5, 0.5)), ("column", (-0.5, -0.5, 0.5, counts.shape[0] - 0.5))]:
        (pixel_left, pixel_top), (pixel_right, pixel_bottom) = axes.transData.transform(
            values=[(data_left, data_top), (data_right, data_bottom)])
        facts[f"confusion.background_{name}_box"] = [int(pixel_left) - 4, int(figure_height - pixel_top) - 4,
                                                     int(pixel_right) + 4, int(figure_height - pixel_bottom) + 4]
    facts["confusion.matrix_box"] = compute_pixel_box(figure=figure, axes=axes, with_labels=True, margin=6)

    # Outcomes of the detections at the threshold, and the classes never found with their class
    facts.update({"confusion.right": int(diagonal.sum()),
                  "confusion.wrong_class": int(counts[1:, 1:].sum() - diagonal.sum()),
                  "confusion.false_positives": int(counts[0, 1:].sum()), "confusion.missed": int(counts[1:, 0].sum()),
                  "confusion.never_right": [class_names[index + 1] for index, value in enumerate(diagonal) if
                                            value == 0]})
    for class_index, class_name in enumerate(class_names[1:]):
        facts[f"confusion.{class_name}.recall"] = round_fact(value=diagonal[class_index] /
                                                             max(int(row_totals[class_index + 1, 0]), 1))

    # Largest confusion between two shapes, read from the matrix rather than named, so that a run trained again keeps
    # a true sentence: the class of the objects, the class of their detections and the count
    wrong_class_counts = counts[1:, 1:].copy()
    numpy.fill_diagonal(a=wrong_class_counts, val=0)
    true_index, predicted_index = numpy.unravel_index(indices=int(wrong_class_counts.argmax()),
                                                      shape=wrong_class_counts.shape)
    facts.update({"confusion.largest.true_name": class_names[true_index + 1],
                  "confusion.largest.predicted_name": class_names[predicted_index + 1],
                  "confusion.largest.count": int(wrong_class_counts[true_index, predicted_index])})
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


def write_report_listings(run_folder: Path, iteration: int, output_folder: Path) -> Dict[str, Any]:
    """
    Writes the tables of the report of one checkpoint (Results.txt) shown by the listings of the evaluation scene,
    and gives their values as facts: the mAP of every method at every IoU with the number of methods, the objects and
    mAP of every size, and the mean average recall of every group, for the trained weights and for their moving
    average.

    Args:
        run_folder (Path): The run folder.
        iteration (int): The evaluated checkpoint.
        output_folder (Path): The listings folder.

    Returns:
        Dict[str, Any]: The values of the tables, named report.*.
    """
    facts = {}
    for averaged_weights, weights_name in [(False, "trained"), (True, "averaged")]:
        block_lines = read_results_block(results_path=run_folder / "Results.txt", iteration=iteration,
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
    number of valid proposals per image of every block, the values of the bars of the proposal recall scene.

    Args:
        evaluation_folder (Path): The evaluation folder of the checkpoint.

    Returns:
        Dict[str, Any]: The values, named proposal_recall.<group>.<column>, and the lists of the bars.
    """
    with open(file=evaluation_folder / "proposal_recall.csv", mode="r", encoding="utf-8") as file_handler:
        rows = {row["group"]: row for row in csv.DictReader(f=file_handler)}
    facts = {}
    for group, row in rows.items():
        for column in ["objects", "recall_50", "recall_70", "proposals_per_image"]:
            if row[column] != "":
                facts[f"proposal_recall.{group}.{column}"] = round_fact(value=float(row[column]))
    blocks = ["block_5", "block_6", "block_7"]
    facts["proposal_recall.proposals_per_block"] = [round_fact(value=float(rows[block]["proposals_per_image"]))
                                                    for block in blocks]
    facts["proposal_recall.proposals_per_block_labels"] = [f"{float(rows[block]['proposals_per_image']):.1f}"
                                                           for block in blocks]
    facts["proposal_recall.recall_per_block"] = [round_fact(value=float(rows[block]["recall_50"])) for block in blocks]
    facts["proposal_recall.recall_per_block_labels"] = [f"{float(rows[block]['recall_50']):.3f}" for block in blocks]
    facts["proposal_recall.recall_per_size"] = [round_fact(value=float(rows[group]["recall_50"])) for group in
                                                ["small", "medium", "large", "all"]]
    facts["proposal_recall.recall_per_size_labels"] = [f"{float(rows[group]['recall_50']):.3f}" for group in
                                                       ["small", "medium", "large", "all"]]
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
