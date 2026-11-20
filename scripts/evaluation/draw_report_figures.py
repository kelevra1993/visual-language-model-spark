"""
When run, draws the figures of the research report of a kept classifier from its run folder, so that a report is
refreshed with one command after a new evaluation of its checkpoint or a new logit temperature saved from its
temperature explorer. From the evaluation of the checkpoint by scripts/evaluation/evaluate_checkpoint.py
('<run>/Checkpoint_Evaluation/Iteration_<n>[/EMA]/') it draws the confusion matrix with the trainer's drawing and the
accuracy and balanced accuracy in percent (confusion_matrix.csv), the precision-recall curve of every class with the
threshold of every target precision, at the temperature saved with scripts/evaluation/serve_temperature_explorer.py
(class_logits.npz, saved_temperature_thresholds.json) and, given the group of every sample in a JSON Lines manifest
(e.g. its camera), the balanced accuracy on the groups seen in training and on those never seen (predictions.csv,
written by the evaluation with save_predictions); from the training of the run, the validation balanced accuracy of
the trained and averaged weights at every evaluated checkpoint, the kept one ringed (Tensorboard/Test,
Tensorboard/Test_EMA), drawn like the evaluation curves of every run (vision_training/evaluation/evaluation_curves.py).
The figures, and report_values.json with the numbers they show (the thresholds rounded down to
the fewest decimals, four at least, selecting the same samples, their recalls, the balanced accuracy of every group),
are written to '<evaluation folder>/Report_Figures/'.
"""

import csv
import json
import math
import matplotlib
import torch
import yaml

from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.evaluation.collect_experiment_results import TEST_FOLDERS
from vision_training.evaluation.checkpoint_evaluation import CHECKPOINT_EVALUATION_FOLDER_NAME
from vision_training.evaluation.classification.classification_metrics import (compute_confusion_matrix,
                                                                              compute_metrics_from_confusion_matrix)
from vision_training.evaluation.classification.confusion_matrix import save_confusion_matrix_image
from vision_training.evaluation.classification.score_curves import ScoreCurves, compute_class_score_curves
from vision_training.evaluation.classification.temperature_explorer import (CLASS_LOGITS_FILE_NAME,
                                                                            SAVED_THRESHOLDS_FILE_NAME,
                                                                            build_positive_masks,
                                                                            check_saved_thresholds,
                                                                            compute_temperature_probabilities,
                                                                            compute_temperature_thresholds,
                                                                            list_target_thresholds, load_class_logits)
from vision_training.evaluation.evaluation_curves import (GRID_COLOR, TEXT_PRIMARY_COLOR, TEXT_SECONDARY_COLOR,
                                                          read_test_curves, save_evaluation_curves_figure)
from vision_training.evaluation.report_formatting import (format_percentage, format_percentage_label,
                                                          format_percentage_tick)
from vision_training.trainers.base_trainer import AVERAGED_WEIGHTS_FOLDER_NAME
from vision_training.utilities.os_utilities import format_path_link, print_color

# Folder of the figures inside the evaluation folder of the kept checkpoint, the files they are drawn from and the
# files written (save_confusion_matrix_image naming the confusion matrix image 'confusion_matrix.png')
REPORT_FIGURES_FOLDER_NAME = "Report_Figures"
CONFUSION_MATRIX_TABLE_NAME = "confusion_matrix.csv"
PREDICTIONS_FILE_NAME = "predictions.csv"
PRECISION_RECALL_FIGURE_NAME = "precision_recall_curves.png"
TRAINING_CURVES_FIGURE_NAME = "training_curves.png"
GROUPS_FIGURE_NAME = "seen_and_never_seen_groups.png"
REPORT_VALUES_FILE_NAME = "report_values.json"

# Folder of the TensorBoard curves of a run, its configuration copy and the labels file of a data folder
TENSORBOARD_FOLDER_NAME = "Tensorboard"
TRAINING_CONFIGURATION_FILE_NAME = "training_configuration.yaml"
LABELS_FILE_NAME = "labels.json"

# Curve of the validation balanced accuracy, as the trainer logs it at every evaluated checkpoint
BALANCED_ACCURACY_CURVE_NAME = "Balanced Accuracy"

# Columns of the predictions file read for the groups figure (one class per sample)
PREDICTION_COLUMNS = ["sample", "true_class", "predicted_class"]

# Reference lines of the report figures, their inks and grid being those of the evaluation curves; the two classes of a
# binary classifier (the first one red) and the curve of every panel of more classes (blue); the groups seen and never
# seen in training (the colours of the two recall columns of the table under the chart of the reports)
REFERENCE_LINE_COLOR = "#c9ced6"
BINARY_CLASS_COLORS = ["#b3261e", "#2b5fa8"]
PANEL_CURVE_COLOR = "#2b5fa8"
SEEN_GROUP_COLOR, NEVER_SEEN_GROUP_COLOR = "#2a78d6", "#d9730d"

# Offsets in points of the thresholds written next to their point: above and right for the first class of a binary
# classifier, below and left for the second one and for the panels of more classes
BINARY_TEXT_OFFSETS = [(8, 7), (-8, -16)]
PANEL_TEXT_OFFSET = (-8, -16)

# Share of the recall axis from its edge within which a threshold is written on the other side of its point, so that
# it stays inside the plot
THRESHOLD_TEXT_EDGE_SHARE = 0.15

# Lowest precision shown by the plot of two classes and by the panels of more classes, lowered to stay
# PRECISION_AXIS_MARGIN below the lowest target precision, and the panels per row
BINARY_MINIMUM_PRECISION = 0.8
PANEL_MINIMUM_PRECISION = 0.6
PRECISION_AXIS_MARGIN = 0.05
PANEL_COLUMNS = 3

# Sizes in inches and resolutions of the figures, so that a Notion page shows each one at its full width: the
# precision-recall figure (the height of a row of panels) and the groups chart, the training curves keeping those of
# the evaluation curves
BINARY_FIGURE_SIZE = (11.0, 7.0)
PANEL_FIGURE_WIDTH, PANEL_ROW_HEIGHT = 15.0, 4.5
GROUPS_FIGURE_SIZE = (6.4, 3.8)
PRECISION_RECALL_DOTS_PER_INCH = 150
GROUPS_DOTS_PER_INCH = 230

# Font of the figures of the reports, used when installed in this folder, matplotlib's default font otherwise
REPORT_FONT_FAMILY = "Inter"
REPORT_FONT_FOLDER = Path("/usr/share/fonts/opentype/inter")


def find_evaluation_folder(experiment_folder: Path, evaluation_folder_name: str, iteration: int, weights: str) -> Path:
    """
    Gives the evaluation folder of the kept checkpoint of a run, '<run>/<evaluation folder>/Iteration_<n>/' for the
    trained weights and its 'EMA/' subfolder for the averaged ones, where scripts/evaluation/evaluate_checkpoint.py
    writes the files the figures of the report are drawn from.

    Args:
        experiment_folder (Path): The run folder of the kept classifier.
        evaluation_folder_name (str): The folder of the evaluations in the run folder, 'Checkpoint_Evaluation' for the
            evaluation script or 'Evaluation' for the evaluations of the training.
        iteration (int): The iteration of the kept checkpoint.
        weights (str): The kept weights, 'trained' or 'ema'.

    Returns:
        Path: The evaluation folder of the checkpoint.

    Raises:
        ValueError: If the weights are neither 'trained' nor 'ema'.
        FileNotFoundError: If the checkpoint has no evaluation folder.
    """
    if weights not in TEST_FOLDERS:
        raise ValueError(f"Expected the weights to be one of {sorted(TEST_FOLDERS)}, got {weights!r}.")
    evaluation_folder = experiment_folder / evaluation_folder_name / f"Iteration_{iteration}"
    if weights == "ema":
        evaluation_folder = evaluation_folder / AVERAGED_WEIGHTS_FOLDER_NAME
    if not evaluation_folder.is_dir():
        raise FileNotFoundError(f"No evaluation of iteration {iteration} ({weights} weights) at "
                                f"{format_path_link(path=evaluation_folder)}: evaluate the checkpoint with "
                                f"scripts/evaluation/evaluate_checkpoint.py first.")
    return evaluation_folder


def register_report_font() -> Optional[str]:
    """
    Registers the Inter font of the research report figures with matplotlib when it is installed, so that the figures
    drawn again match the ones already in the reports, matplotlib's default font being kept otherwise.

    Returns:
        Optional[str]: The font family to draw the figures with, None to keep matplotlib's default font.
    """
    font_paths = (font_manager.findSystemFonts(fontpaths=[str(REPORT_FONT_FOLDER)]) if REPORT_FONT_FOLDER.is_dir()
                  else [])
    if not font_paths:
        print_color(color="yellow", output=f"The {REPORT_FONT_FAMILY} font of the reports is not installed in "
                                           f"{REPORT_FONT_FOLDER}, the figures are drawn with matplotlib's default "
                                           f"font.")
        return None
    for font_path in font_paths:
        font_manager.fontManager.addfont(path=font_path)
    return REPORT_FONT_FAMILY


def read_confusion_matrix(table_path: Path) -> Tuple[List[str], torch.Tensor]:
    """
    Reads the confusion matrix saved by the evaluation of a checkpoint (save_confusion_matrix_table), drawn again for
    the report with its accuracies in percent, and the reference the predictions file of the groups figure must match.

    Args:
        table_path (Path): The confusion_matrix.csv of the evaluation folder.

    Returns:
        Tuple[List[str], torch.Tensor]: The class names, in class index order, and the [K, K] int64 matrix (rows true
            classes, columns predicted classes).

    Raises:
        FileNotFoundError: If the table is missing.
        ValueError: If the rows do not name the classes of the columns, in the same order, with one count per class.
    """
    if not table_path.is_file():
        raise FileNotFoundError(f"No confusion matrix table at {format_path_link(path=table_path)}, written by the "
                                f"evaluation of every classifier.")
    with table_path.open(newline="") as table_file:
        rows = list(csv.reader(table_file))

    # One row per class named in the header, in the same order, each with one count per class
    class_names = rows[0][1:] if rows else []
    row_names = [row[0] for row in rows[1:]]
    if not class_names or row_names != class_names or any(len(row) != len(class_names) + 1 for row in rows[1:]):
        raise ValueError(f"Expected {table_path} to hold one row of counts per class of its header {class_names}, in "
                         f"the same order, got the rows {row_names}.")
    return class_names, torch.tensor(data=[[int(count) for count in row[1:]] for row in rows[1:]], dtype=torch.int64)


def draw_confusion_matrix_figure(confusion_matrix: torch.Tensor, class_names: List[str], iteration: int,
                                 output_folder: Path) -> Tuple[Path, Dict[str, Any]]:
    """
    Draws the confusion matrix of the kept checkpoint with the trainer's own drawing, its title giving the accuracy and
    the balanced accuracy in percent as the reports write them, so that an evaluation written before the percentages of
    the reports gets the figure of a recent one.

    Args:
        confusion_matrix (torch.Tensor): The [K, K] confusion matrix of the evaluation.
        class_names (List[str]): The class names, in class index order.
        iteration (int): The iteration of the checkpoint, written in the title.
        output_folder (Path): The folder of the figures.

    Returns:
        Tuple[Path, Dict[str, Any]]: The written image and the metrics of the matrix
            (compute_metrics_from_confusion_matrix), the recall of every class among them.
    """
    metrics = compute_metrics_from_confusion_matrix(confusion_matrix=confusion_matrix)
    title = (f"Confusion Matrix · Iteration {iteration} · Accuracy {format_percentage(value=metrics['accuracy'])} · "
             f"Balanced Accuracy {format_percentage(value=metrics['balanced_accuracy'])}")
    image_path, _ = save_confusion_matrix_image(confusion_matrix=confusion_matrix, class_names=class_names,
                                                output_folder=output_folder, title=title, in_percent=True)
    return image_path, metrics


def read_report_thresholds(evaluation_folder: Path, class_logits_record: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Gives the thresholds of every class at every target precision shown by the report: those saved from the
    temperature explorer of the checkpoint (saved_temperature_thresholds.json, at the temperature of the export),
    checked to be exactly the ones found again from the class logits at that temperature, so that the figure marks the
    deployed thresholds; without a saved temperature, the ones found from the logits at a temperature of 1.

    Args:
        evaluation_folder (Path): The evaluation folder of the checkpoint.
        class_logits_record (Dict[str, Any]): The class logits of the evaluation (load_class_logits).

    Returns:
        Tuple[Dict[str, Any], bool]: The thresholds, in the format of class_thresholds.json with their
            'logit_temperature', and whether they were saved from the temperature explorer.

    Raises:
        ValueError: If the saved thresholds are not those of the evaluation (check_saved_thresholds), or differ from the
            ones found again from the logits at their temperature.
    """
    saved_thresholds_path = evaluation_folder / SAVED_THRESHOLDS_FILE_NAME
    if not saved_thresholds_path.is_file():
        print_color(color="yellow", output=f"No {SAVED_THRESHOLDS_FILE_NAME} in "
                                           f"{format_path_link(path=evaluation_folder)}: the precision-recall curves "
                                           f"are drawn at a logit temperature of 1. Save the temperature of the export "
                                           f"from scripts/evaluation/serve_temperature_explorer.py to draw them at it.")
        return compute_temperature_thresholds(class_logits_record=class_logits_record, logit_temperature=1.0), False

    # The saved thresholds must be those of this evaluation, and found again identically from its logits
    saved_thresholds = json.loads(s=saved_thresholds_path.read_text())
    check_saved_thresholds(saved_thresholds=saved_thresholds, evaluation_folder=evaluation_folder)
    found_thresholds = compute_temperature_thresholds(class_logits_record=class_logits_record,
                                                      logit_temperature=saved_thresholds["logit_temperature"])
    if found_thresholds != saved_thresholds:
        raise ValueError(f"The thresholds found again from {CLASS_LOGITS_FILE_NAME} at the logit temperature "
                         f"{saved_thresholds['logit_temperature']} differ from "
                         f"{format_path_link(path=saved_thresholds_path)}: save them again from the page of the "
                         f"checkpoint served by scripts/evaluation/serve_temperature_explorer.py.")
    return saved_thresholds, True


def list_class_target_thresholds(thresholds: Dict[str, Any], class_names: List[str]
                                 ) -> Tuple[List[float], List[List[Optional[Tuple[float, float]]]]]:
    """
    Turns thresholds in the format of class_thresholds.json into their target precisions and, for every target, the
    threshold and recall of every class in class index order, the form in which compute_class_score_curves marks them
    on the curves.

    Args:
        thresholds (Dict[str, Any]): The thresholds, in the format of class_thresholds.json.
        class_names (List[str]): The class names, in class index order.

    Returns:
        Tuple[List[float], List[List[Optional[Tuple[float, float]]]]]: The target precisions, the main one first, and
            for every target the threshold and recall of every class, None for a class never reaching it.
    """
    target_precisions, target_thresholds = [], []
    for target_precision, classes in list_target_thresholds(thresholds=thresholds):
        target_precisions.append(float(target_precision))
        target_thresholds.append([None if classes[class_name] is None else
                                  (classes[class_name]["threshold"], classes[class_name]["recall"])
                                  for class_name in class_names])
    return target_precisions, target_thresholds


def style_precision_recall_axis(axis: Axes, target_precisions: List[float], minimum_precision: float,
                                minimum_recall: float, label_targets: bool) -> None:
    """
    Draws the dashed line of every target precision and keeps the axes of a precision-recall plot recessive, the
    precision and recall written as the percentages of the reports. The recall limits are set here, before the curves,
    since the thresholds written near an edge of the recall axis are moved to the other side of their point.

    Args:
        axis (Axes): The plot.
        target_precisions (List[float]): The target precisions, each drawn as a dashed line.
        minimum_precision (float): The bottom of the precision axis.
        minimum_recall (float): The left end of the recall axis.
        label_targets (bool): Whether the target precision lines are named on the right of the plot.
    """
    # Dashed line of every target precision, named outside the plot on its right
    for target_precision in target_precisions:
        axis.axhline(y=target_precision, color=REFERENCE_LINE_COLOR, linewidth=1.2, linestyle="--", zorder=1)
        if label_targets:
            axis.text(x=1.015, y=target_precision, s=f"{format_percentage_label(value=target_precision)} precision",
                      fontsize=10, color=TEXT_SECONDARY_COLOR, va="center", ha="left",
                      transform=axis.get_yaxis_transform())

    # Limits a little beyond 100 % so that the points at a recall or precision of 100 % stay whole, recessive axes
    axis.set_xlim(left=minimum_recall, right=1.0 + (1.0 - minimum_recall) / 100.0)
    axis.set_ylim(bottom=minimum_precision, top=1.005)
    axis.xaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))
    axis.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))
    axis.grid(color=GRID_COLOR, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(REFERENCE_LINE_COLOR)
    axis.tick_params(colors=TEXT_SECONDARY_COLOR)


def draw_class_precision_recall(axis: Axes, curves: ScoreCurves, color: str, label: Optional[str],
                                text_offset: Tuple[int, int]) -> None:
    """
    Draws the precision-recall curve of one class as steps, the point of its threshold at every target precision it
    reaches, and that threshold written next to its point in the colour of the curve, as the research reports show
    them; a threshold whose point lies near an edge of the recall axis is written on the other side of the point.

    Args:
        axis (Axes): The plot, whose recall limits are already set (style_precision_recall_axis).
        curves (ScoreCurves): The curves of the class (compute_class_score_curves), its threshold points marked.
        color (str): The colour of the curve, of its points and of its thresholds.
        label (Optional[str]): The legend entry of the curve, None for none.
        text_offset (Tuple[int, int]): The offset in points of the thresholds from their point, to the right for a
            positive first value.
    """
    axis.plot(curves["recalls"], curves["precisions"], color=color, linewidth=2, drawstyle="steps-pre", label=label,
              zorder=3)
    left_end, right_end = axis.get_xlim()
    for threshold_point in curves["threshold_points"]:
        # A target precision the class never reaches has no point
        if threshold_point is None:
            continue
        point_recall, point_precision, threshold_label = threshold_point
        axis.scatter(x=[point_recall], y=[point_precision], s=70, color=color, edgecolors="white", linewidths=2,
                     zorder=4)

        # Threshold written on the other side of its point when it would leave the plot by the nearest edge
        horizontal_offset, vertical_offset = text_offset
        axis_share = (point_recall - left_end) / (right_end - left_end)
        if ((horizontal_offset > 0 and axis_share > 1.0 - THRESHOLD_TEXT_EDGE_SHARE) or
                (horizontal_offset < 0 and axis_share < THRESHOLD_TEXT_EDGE_SHARE)):
            horizontal_offset = -horizontal_offset
        axis.annotate(text=threshold_label, xy=(point_recall, point_precision),
                      xytext=(horizontal_offset, vertical_offset), textcoords="offset points",
                      ha="left" if horizontal_offset > 0 else "right", fontsize=10.5, color=color,
                      fontweight="semibold", zorder=5,
                      bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.9})


def draw_precision_recall_figure(class_logits_record: Dict[str, Any], thresholds: Dict[str, Any],
                                 class_display_names: Dict[str, str], minimum_recall: float, title: str,
                                 output_path: Path) -> Dict[str, List[Optional[Dict[str, Any]]]]:
    """
    Draws the precision-recall curve of every class of the kept checkpoint at the temperature of its thresholds, the
    figure of the Threshold section of the report: the scores are those of the exported model (the logits divided by
    the temperature, then the softmax), and the threshold of every target precision is marked on the curve, rounded
    down to the fewest decimals, four at least, selecting the same samples (format_threshold_label). The two classes
    of a binary classifier share one plot, the first one in red with its thresholds above and right of their point,
    the second in blue below and left; more classes get one panel each, PANEL_COLUMNS per row.

    Args:
        class_logits_record (Dict[str, Any]): The class logits of the evaluation (load_class_logits).
        thresholds (Dict[str, Any]): The thresholds shown, in the format of class_thresholds.json with their
            'logit_temperature' (read_report_thresholds).
        class_display_names (Dict[str, str]): The name shown for every class, in the order of the curves or panels.
        minimum_recall (float): The left end of the recall axis, raised for a classifier whose points all lie near a
            recall of 100 %.
        title (str): The title of the figure.
        output_path (Path): The PNG written.

    Returns:
        Dict[str, List[Optional[Dict[str, Any]]]]: For every class, at every target precision in the order of the
            thresholds, its 'target_precision', the 'threshold_text' written, the exact 'threshold' and the 'recall'
            and 'precision' it keeps, None for a target the class never reaches.
    """
    # Probabilities of the exported model at the temperature, and the curves of every class with their thresholds
    class_names = class_logits_record["class_names"]
    probabilities = compute_temperature_probabilities(
        class_logits=class_logits_record["class_logits"], output_mode=class_logits_record["output_mode"],
        logit_temperature=thresholds["logit_temperature"])
    positive_masks = build_positive_masks(
        labels=class_logits_record["labels"], output_mode=class_logits_record["output_mode"],
        number_classes=len(class_names))
    target_precisions, target_thresholds = list_class_target_thresholds(thresholds=thresholds, class_names=class_names)
    class_curves = compute_class_score_curves(probabilities=probabilities, positive_masks=positive_masks,
                                              class_thresholds=target_thresholds)
    drawn_class_names = list(class_display_names)

    # Two classes: both curves on one plot, with a legend naming the classes
    if len(class_names) == 2:
        figure = Figure(figsize=BINARY_FIGURE_SIZE)
        FigureCanvasAgg(figure=figure)
        axis = figure.subplots()
        style_precision_recall_axis(axis=axis, target_precisions=target_precisions,
                                    minimum_precision=min(BINARY_MINIMUM_PRECISION,
                                                          min(target_precisions) - PRECISION_AXIS_MARGIN),
                                    minimum_recall=minimum_recall, label_targets=True)
        for class_position, class_name in enumerate(drawn_class_names):
            draw_class_precision_recall(axis=axis, curves=class_curves[class_names.index(class_name)],
                                        color=BINARY_CLASS_COLORS[class_position],
                                        label=class_display_names[class_name],
                                        text_offset=BINARY_TEXT_OFFSETS[class_position])
        axis.set_xlabel(xlabel="Recall", fontsize=12, color=TEXT_PRIMARY_COLOR)
        axis.set_ylabel(ylabel="Precision", fontsize=12, color=TEXT_PRIMARY_COLOR)
        axis.legend(loc="lower left", frameon=False, fontsize=11, title="Class (threshold on its score at each mark)",
                    title_fontsize=10)
        axis.set_title(label=title, fontsize=13, color=TEXT_PRIMARY_COLOR, loc="left", pad=14)

    # More classes: one panel per class, named by its title, the panels left over in the last row hidden
    else:
        number_rows = math.ceil(len(drawn_class_names) / PANEL_COLUMNS)
        figure = Figure(figsize=(PANEL_FIGURE_WIDTH, PANEL_ROW_HEIGHT * number_rows))
        FigureCanvasAgg(figure=figure)
        axes = figure.subplots(nrows=number_rows, ncols=PANEL_COLUMNS, sharex=True, sharey=True, squeeze=False)
        for panel_index, axis in enumerate(axes.flat):
            if panel_index >= len(drawn_class_names):
                axis.set_visible(False)
                continue
            class_name = drawn_class_names[panel_index]
            is_row_end = panel_index % PANEL_COLUMNS == PANEL_COLUMNS - 1 or panel_index == len(drawn_class_names) - 1
            style_precision_recall_axis(axis=axis, target_precisions=target_precisions,
                                        minimum_precision=min(PANEL_MINIMUM_PRECISION,
                                                              min(target_precisions) - PRECISION_AXIS_MARGIN),
                                        minimum_recall=minimum_recall, label_targets=is_row_end)
            draw_class_precision_recall(axis=axis, curves=class_curves[class_names.index(class_name)],
                                        color=PANEL_CURVE_COLOR, label=None, text_offset=PANEL_TEXT_OFFSET)
            axis.set_title(label=class_display_names[class_name], fontsize=12, color=TEXT_PRIMARY_COLOR, loc="left")

            # Recall written under the lowest panel of every column, precision left of the first panel of every row
            if panel_index + PANEL_COLUMNS >= len(drawn_class_names):
                axis.tick_params(axis="x", labelbottom=True)
                axis.set_xlabel(xlabel="Recall", fontsize=11, color=TEXT_PRIMARY_COLOR)
            if panel_index % PANEL_COLUMNS == 0:
                axis.set_ylabel(ylabel="Precision", fontsize=11, color=TEXT_PRIMARY_COLOR)
        figure.suptitle(t=title, fontsize=13, color=TEXT_PRIMARY_COLOR, x=0.06, ha="left")
        figure.tight_layout()
    figure.savefig(fname=output_path, dpi=PRECISION_RECALL_DOTS_PER_INCH, bbox_inches="tight", facecolor="white")

    # Written threshold, exact threshold, recall and precision of every class at every target precision
    class_threshold_values = {}
    for class_name in drawn_class_names:
        class_index = class_names.index(class_name)
        class_threshold_values[class_name] = [
            None if threshold_point is None else
            {"target_precision": target_precision, "threshold_text": threshold_point[2],
             "threshold": target_thresholds[target_index][class_index][0], "recall": threshold_point[0],
             "precision": threshold_point[1]}
            for target_index, (target_precision, threshold_point) in enumerate(
                zip(target_precisions, class_curves[class_index]["threshold_points"]))]
    return class_threshold_values


def read_balanced_accuracy_curves(experiment_folder: Path, iteration: int, weights: str) -> Dict[str, Dict[int, float]]:
    """
    Reads the validation balanced accuracy of the trained and averaged weights at every checkpoint evaluated during
    the training, from the Tensorboard/Test and Tensorboard/Test_EMA curves of the run (read_test_curves), the curves
    of the training figure of the report, which rings the kept checkpoint on them.

    Args:
        experiment_folder (Path): The run folder.
        iteration (int): The iteration of the kept checkpoint.
        weights (str): The kept weights, 'trained' or 'ema'.

    Returns:
        Dict[str, Dict[int, float]]: The balanced accuracy at every evaluated iteration of the 'trained' weights and,
            when the run kept a moving average of its weights, of the 'ema' ones.

    Raises:
        ValueError: If the run has no balanced accuracy curve of its trained weights, or the training evaluated no kept
            weights at the kept iteration.
    """
    curves = {}
    for curve_weights, test_folder_name in TEST_FOLDERS.items():
        test_curves = read_test_curves(test_folder=experiment_folder / TENSORBOARD_FOLDER_NAME / test_folder_name)
        if BALANCED_ACCURACY_CURVE_NAME in test_curves:
            curves[curve_weights] = test_curves[BALANCED_ACCURACY_CURVE_NAME]
    if "trained" not in curves:
        trained_test_folder = experiment_folder / TENSORBOARD_FOLDER_NAME / TEST_FOLDERS["trained"]
        raise ValueError(f"No '{BALANCED_ACCURACY_CURVE_NAME}' curve in {format_path_link(path=trained_test_folder)}, "
                         f"written at every checkpoint evaluated during the training of a classifier.")

    # The kept checkpoint must be one of the evaluations of its weights during the training
    if iteration not in curves.get(weights, {}):
        raise ValueError(f"The training evaluated no {weights} weights at iteration {iteration}, its evaluated "
                         f"iterations being {sorted(curves.get(weights, {}))}.")
    return curves


def draw_training_curves_figure(curves: Dict[str, Dict[int, float]], iteration: int, weights: str, split_name: str,
                                output_path: Path) -> float:
    """
    Draws the validation balanced accuracy of the trained and averaged weights at every checkpoint evaluated during
    the training, the kept checkpoint ringed with its value, the training figure of the report showing when the kept
    weights were reached and how the curves move around them, with the drawing of the evaluation curves of every run
    (save_evaluation_curves_figure, without the window of the moving average).

    Args:
        curves (Dict[str, Dict[int, float]]): The balanced accuracy of the weights at every evaluated iteration, the
            kept checkpoint among them (read_balanced_accuracy_curves).
        iteration (int): The iteration of the kept checkpoint.
        weights (str): The kept weights, 'trained' or 'ema'.
        split_name (str): The name of the evaluated split, written on the vertical axis, e.g. 'validation'.
        output_path (Path): The PNG written.

    Returns:
        float: The balanced accuracy of the kept checkpoint.
    """
    # The kept checkpoint ringed on the curve of its weights, its value written above it
    kept_value = curves[weights][iteration]
    kept_point = {"weights": weights, "iterations": [iteration],
                  "label": f"kept: iteration {iteration:,}, {format_percentage(value=kept_value)}"}
    save_evaluation_curves_figure(curves=curves, marked_points=[kept_point],
                                  metric_label=f"{split_name.capitalize()} balanced accuracy", in_percent=True,
                                  moving_average=None, title=None, output_path=output_path)
    return kept_value


def read_sample_groups(manifest_path: Path, file_name_key: str, group_key: str) -> Dict[str, str]:
    """
    Reads the group of every sample from a JSON Lines manifest, one object per sample, e.g. a manifest of a dataset
    built from the images of several cameras, whose 'camera' names the camera of every image, so that the evaluated
    samples are split into the groups seen in training and the groups never seen.

    Args:
        manifest_path (Path): The JSON Lines manifest.
        file_name_key (str): The key of the file name of a sample, as the labels files and predictions.csv name it.
        group_key (str): The key of the group of a sample.

    Returns:
        Dict[str, str]: The group of every sample, by file name.

    Raises:
        ValueError: If a line has no file name or no group, or a file name is given two groups.
    """
    sample_groups = {}
    with manifest_path.open(encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if not line.strip():
                continue

            # Every sample needs its file name and a group, a sample listed twice keeping the same group
            record = json.loads(s=line)
            if record.get(file_name_key) in (None, "") or record.get(group_key) in (None, ""):
                raise ValueError(f"Line {line_number} of {manifest_path} gives no {file_name_key!r} or no "
                                 f"{group_key!r}.")
            file_name, group = record[file_name_key], str(record[group_key])
            if sample_groups.setdefault(file_name, group) != group:
                raise ValueError(f"{manifest_path} gives {file_name} the groups {sample_groups[file_name]!r} and "
                                 f"{group!r}.")
    return sample_groups


def read_predictions(predictions_path: Path) -> List[Dict[str, str]]:
    """
    Reads the predictions of every evaluated sample written by the evaluation of a classifier with save_predictions
    (write_predictions_file), the samples named so that each is given its group.

    Args:
        predictions_path (Path): The predictions.csv of the evaluation folder.

    Returns:
        List[Dict[str, str]]: One row per evaluated sample, with its 'sample', 'true_class' and 'predicted_class'.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If it lacks a column of one class per sample.
    """
    if not predictions_path.is_file():
        raise FileNotFoundError(f"No predictions file at {format_path_link(path=predictions_path)}: evaluate the "
                                f"checkpoint with save_predictions = True in "
                                f"scripts/evaluation/evaluate_checkpoint.py, or give the predictions file of the same "
                                f"evaluation.")
    with predictions_path.open(newline="") as predictions_file:
        reader = csv.DictReader(predictions_file)
        missing_columns = [column for column in PREDICTION_COLUMNS if column not in (reader.fieldnames or [])]
        if missing_columns:
            raise ValueError(f"{predictions_path} has no {missing_columns} columns, written for a classifier giving "
                             f"one class per sample.")
        return list(reader)


def find_training_labels_path(experiment_folder: Path) -> Path:
    """
    Gives the labels file of the training split of a run, from the 'train_data_folder' of its configuration copy, the
    samples whose groups are the ones seen in training.

    Args:
        experiment_folder (Path): The run folder.

    Returns:
        Path: The labels.json of the training split.

    Raises:
        FileNotFoundError: If the labels file is missing, e.g. a data folder moved since the training.
    """
    configuration = yaml.safe_load(stream=(experiment_folder / TRAINING_CONFIGURATION_FILE_NAME).read_text())
    labels_path = Path(configuration["ExperimentConfiguration"]["train_data_folder"]) / LABELS_FILE_NAME
    if not labels_path.is_file():
        raise FileNotFoundError(f"No training labels file at {format_path_link(path=labels_path)}, the "
                                f"train_data_folder of the run: give training_labels_path, the labels.json of its "
                                f"training split.")
    return labels_path


def describe_sample_set(predictions: List[Dict[str, str]], sample_groups: Dict[str, str],
                        class_names: List[str]) -> Dict[str, Any]:
    """
    Describes the predictions of a set of evaluated samples, the samples of the groups seen in training or never seen:
    its numbers of groups and samples, the groups, samples and recall of every class, and its balanced accuracy, from
    its confusion matrix as the evaluation computes them (compute_metrics_from_confusion_matrix), so that the balanced
    accuracy is the mean recall of the classes present in the set.

    Args:
        predictions (List[Dict[str, str]]): The predictions of the samples of the set, their classes known.
        sample_groups (Dict[str, str]): The group of every sample, by file name.
        class_names (List[str]): The class names, in class index order.

    Returns:
        Dict[str, Any]: The 'groups', 'samples' and 'balanced_accuracy' of the set, and its 'classes', each with its
            'groups', 'samples' and 'recall' (None for a class without samples in the set).
    """
    true_indices = torch.tensor(data=[class_names.index(row["true_class"]) for row in predictions], dtype=torch.int64)
    predicted_indices = torch.tensor(data=[class_names.index(row["predicted_class"]) for row in predictions],
                                     dtype=torch.int64)
    metrics = compute_metrics_from_confusion_matrix(confusion_matrix=compute_confusion_matrix(
        predicted_labels=predicted_indices, labels=true_indices, number_classes=len(class_names)))

    # Groups, samples and recall of every class, a class without samples having no recall
    classes = {}
    for class_index, class_name in enumerate(class_names):
        class_rows = [row for row in predictions if row["true_class"] == class_name]
        classes[class_name] = {"groups": len({sample_groups[row["sample"]] for row in class_rows}),
                               "samples": len(class_rows),
                               "recall": metrics["recall"][class_index] if class_rows else None}
    return {"groups": len({sample_groups[row["sample"]] for row in predictions}), "samples": len(predictions),
            "balanced_accuracy": metrics["balanced_accuracy"], "classes": classes}


def compare_seen_and_never_seen_groups(predictions: List[Dict[str, str]], sample_groups: Dict[str, str],
                                       training_file_names: List[str], class_names: List[str],
                                       confusion_matrix: torch.Tensor) -> Dict[str, Any]:
    """
    Splits the evaluated samples of the kept checkpoint into the samples of the groups seen in training (a group with
    at least one training sample) and those of the groups never seen, and describes both sets and all the samples,
    the chart and the table of the never seen groups section of the report. The predictions must be those of the
    evaluation: they give its confusion matrix exactly.

    Args:
        predictions (List[Dict[str, str]]): The predictions of every evaluated sample (read_predictions).
        sample_groups (Dict[str, str]): The group of every sample, by file name (read_sample_groups).
        training_file_names (List[str]): The file names of the training samples.
        class_names (List[str]): The class names, in class index order.
        confusion_matrix (torch.Tensor): The [K, K] confusion matrix of the evaluation.

    Returns:
        Dict[str, Any]: The descriptions of the 'seen', 'never_seen' and 'all' samples (describe_sample_set), and the
            sorted 'never_seen_groups'.

    Raises:
        ValueError: If a prediction names a class of another classifier, the predictions do not give the confusion
            matrix of the evaluation, a sample has no group, or no evaluated sample is seen, or never seen.
    """
    # The predictions of the evaluation, with the classes of the classifier
    unknown_classes = sorted({row[column] for row in predictions for column in ("true_class", "predicted_class")} -
                             set(class_names))
    if unknown_classes:
        raise ValueError(f"The predictions name the classes {unknown_classes}, not classes of the evaluation "
                         f"{class_names}.")
    predictions_matrix = compute_confusion_matrix(
        predicted_labels=torch.tensor(data=[class_names.index(row["predicted_class"]) for row in predictions]),
        labels=torch.tensor(data=[class_names.index(row["true_class"]) for row in predictions]),
        number_classes=len(class_names))
    if not torch.equal(input=predictions_matrix, other=confusion_matrix):
        raise ValueError(f"The predictions do not give the confusion matrix of the evaluation ({len(predictions)} "
                         f"samples for {int(confusion_matrix.sum())} evaluated): give the predictions file of the "
                         f"same evaluation.")

    # Every evaluated and training sample needs its group
    missing_samples = sorted({row["sample"] for row in predictions if row["sample"] not in sample_groups} |
                             {file_name for file_name in training_file_names if file_name not in sample_groups})
    if missing_samples:
        raise ValueError(f"{len(missing_samples)} samples have no group in the manifest, e.g. {missing_samples[:3]}.")

    # Samples of the groups with at least one training sample, and of the other groups
    seen_groups = {sample_groups[file_name] for file_name in training_file_names}
    seen_predictions = [row for row in predictions if sample_groups[row["sample"]] in seen_groups]
    never_seen_predictions = [row for row in predictions if sample_groups[row["sample"]] not in seen_groups]
    if not seen_predictions or not never_seen_predictions:
        raise ValueError(f"{len(seen_predictions)} evaluated samples come from groups seen in training and "
                         f"{len(never_seen_predictions)} from groups never seen: the chart compares both.")
    return {"seen": describe_sample_set(predictions=seen_predictions, sample_groups=sample_groups,
                                        class_names=class_names),
            "never_seen": describe_sample_set(predictions=never_seen_predictions, sample_groups=sample_groups,
                                              class_names=class_names),
            "all": describe_sample_set(predictions=predictions, sample_groups=sample_groups, class_names=class_names),
            "never_seen_groups": sorted({sample_groups[row["sample"]] for row in never_seen_predictions})}


def draw_groups_figure(group_comparison: Dict[str, Any], group_plural: str, sample_kind: str, split_name: str,
                       output_path: Path) -> None:
    """
    Draws the balanced accuracy of the kept checkpoint on the samples of the groups seen in training and on those of
    the groups never seen, two bars from zero with their value above them and, under each, its numbers of groups and
    samples and the classes it has no sample of, the chart of the never seen groups section of the report.

    Args:
        group_comparison (Dict[str, Any]): The seen and never seen sets (compare_seen_and_never_seen_groups).
        group_plural (str): What the groups are, in the plural, e.g. 'cameras'.
        sample_kind (str): What the samples are, in the plural, e.g. 'images' or 'frames'.
        split_name (str): The name of the evaluated split, written on the vertical axis, e.g. 'validation'.
        output_path (Path): The PNG written.
    """
    figure = Figure(figsize=GROUPS_FIGURE_SIZE)
    FigureCanvasAgg(figure=figure)
    axis = figure.subplots()

    # Label of every bar: its groups and samples, then the classes without sample in it
    bar_labels, bar_values = [], []
    for set_key, set_title in [("seen", "seen in training"), ("never_seen", "never seen in training")]:
        sample_set = group_comparison[set_key]
        bar_label = (f"{group_plural.capitalize()} {set_title}\n{sample_set['groups']:,} {group_plural}, "
                     f"{sample_set['samples']:,} {sample_kind}")
        missing_classes = [class_name.replace("_", " ") for class_name, class_description
                           in sample_set["classes"].items() if class_description["samples"] == 0]
        if missing_classes:
            bar_label += "\nno " + ", ".join(missing_classes)
        bar_labels.append(bar_label)
        bar_values.append(sample_set["balanced_accuracy"])

    # Two bars from zero, their value written above them
    axis.bar(x=[0, 1], height=bar_values, width=0.5, color=[SEEN_GROUP_COLOR, NEVER_SEEN_GROUP_COLOR])
    for bar_position, bar_value in enumerate(bar_values):
        axis.text(x=bar_position, y=bar_value + 0.015, s=format_percentage(value=bar_value), ha="center", fontsize=11,
                  color=TEXT_PRIMARY_COLOR, fontweight="semibold")

    # Recessive axes and grid, the rates in percent
    axis.set_xticks(ticks=[0, 1], labels=bar_labels, fontsize=9, color=TEXT_SECONDARY_COLOR)
    axis.set_ylim(bottom=0.0, top=1.08)
    axis.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))
    axis.set_ylabel(ylabel=f"{split_name.capitalize()} balanced accuracy", fontsize=9, color=TEXT_SECONDARY_COLOR)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID_COLOR)
    axis.tick_params(colors=TEXT_SECONDARY_COLOR, labelsize=9)
    figure.tight_layout()
    figure.savefig(fname=output_path, dpi=GROUPS_DOTS_PER_INCH, facecolor="white")


def draw_report_figures(experiment_folder: Path, iteration: int, weights: str, evaluation_folder_name: str,
                        class_display_names: Optional[Dict[str, str]], sample_kind: str, split_name: str,
                        minimum_recall: float, group_manifest_path: Optional[Path], file_name_key: str, group_key: str,
                        group_plural: str, training_labels_path: Optional[Path], predictions_path: Optional[Path],
                        font_family: Optional[str]) -> Path:
    """
    Draws every figure of the research report of a kept classifier into '<evaluation folder>/Report_Figures/' and
    writes the numbers they show to report_values.json, so that refreshing a report after a new evaluation or a new
    saved temperature takes one run. Every input is read and checked before the first figure is drawn, so that a
    wrong one stops the run without writing anything.

    Args:
        experiment_folder (Path): The run folder of the kept classifier.
        iteration (int): The iteration of the kept checkpoint.
        weights (str): The kept weights, 'trained' or 'ema'.
        evaluation_folder_name (str): The folder of the evaluations in the run folder, e.g. 'Checkpoint_Evaluation'.
        class_display_names (Optional[Dict[str, str]]): The name shown for every class, in the order of the curves or
            panels, None for the class names in class index order.
        sample_kind (str): What the samples are, in the plural, e.g. 'images' or 'frames'.
        split_name (str): The name of the evaluated split, e.g. 'validation'.
        minimum_recall (float): The left end of the recall axis of the precision-recall figure.
        group_manifest_path (Optional[Path]): The JSON Lines manifest giving the group of every sample, None to skip
            the groups chart.
        file_name_key (str): The key of the file name of a sample in the manifest.
        group_key (str): The key of the group of a sample in the manifest.
        group_plural (str): What the groups are, in the plural, e.g. 'cameras'.
        training_labels_path (Optional[Path]): The labels.json of the training split, None for the one of the
            train_data_folder of the run.
        predictions_path (Optional[Path]): The predictions file of the evaluation, None for the predictions.csv of the
            evaluation folder.
        font_family (Optional[str]): The font of the figures (register_report_font), None for matplotlib's default;
            the confusion matrix keeps the font of the trainer's drawing.

    Returns:
        Path: The folder of the figures.

    Raises:
        ValueError: If the classifier gives several classes per sample, or the names shown, the confusion matrix and
            the class logits do not have the same classes (and the errors of the readers of every figure).
        FileNotFoundError: If the group manifest is given but missing.
    """
    evaluation_folder = find_evaluation_folder(experiment_folder=experiment_folder,
                                               evaluation_folder_name=evaluation_folder_name, iteration=iteration,
                                               weights=weights)

    # Class logits of the evaluation, one class per sample, and the names shown for every class
    class_logits_record = load_class_logits(evaluation_folder=evaluation_folder)
    if class_logits_record["output_mode"] != "multi_class":
        raise ValueError(f"The figures of a report are drawn for a classifier giving one class per sample, the "
                         f"evaluation of {evaluation_folder} is {class_logits_record['output_mode']}.")
    class_names = class_logits_record["class_names"]
    class_display_names = class_display_names or {class_name: class_name for class_name in class_names}
    if sorted(class_display_names) != sorted(class_names):
        raise ValueError(f"Expected a name shown for every class of the evaluation {class_names}, got "
                         f"{list(class_display_names)}.")

    # Confusion matrix of the evaluation, of the same classes in the same order
    matrix_class_names, confusion_matrix = read_confusion_matrix(
        table_path=evaluation_folder / CONFUSION_MATRIX_TABLE_NAME)
    if matrix_class_names != class_names:
        raise ValueError(f"The confusion matrix of {evaluation_folder} has the classes {matrix_class_names}, its class "
                         f"logits {class_names}.")

    # Thresholds at the saved temperature and training curves with the kept checkpoint, checked before any drawing
    thresholds, is_saved_temperature = read_report_thresholds(evaluation_folder=evaluation_folder,
                                                              class_logits_record=class_logits_record)
    training_curves = read_balanced_accuracy_curves(experiment_folder=experiment_folder, iteration=iteration,
                                                    weights=weights)

    # Groups seen and never seen in training, from the predictions of the evaluation and the group of every sample
    group_comparison = None
    if group_manifest_path is not None:
        predictions = read_predictions(predictions_path=predictions_path or evaluation_folder / PREDICTIONS_FILE_NAME)
        training_labels_path = training_labels_path or find_training_labels_path(experiment_folder=experiment_folder)
        if not group_manifest_path.is_file():
            raise FileNotFoundError(f"No group manifest at {format_path_link(path=group_manifest_path)}.")
        training_file_names = [sample["file_name"] for sample
                               in json.loads(s=training_labels_path.read_text())["samples"]]
        group_comparison = compare_seen_and_never_seen_groups(
            predictions=predictions, sample_groups=read_sample_groups(
                manifest_path=group_manifest_path, file_name_key=file_name_key, group_key=group_key),
            training_file_names=training_file_names, class_names=class_names, confusion_matrix=confusion_matrix)

    # Confusion matrix drawn by the trainer's function, in its own font
    output_folder = evaluation_folder / REPORT_FIGURES_FOLDER_NAME
    output_folder.mkdir(exist_ok=True)
    confusion_matrix_path, metrics = draw_confusion_matrix_figure(confusion_matrix=confusion_matrix,
                                                                  class_names=class_names, iteration=iteration,
                                                                  output_folder=output_folder)
    written_paths = [confusion_matrix_path, output_folder / PRECISION_RECALL_FIGURE_NAME,
                     output_folder / TRAINING_CURVES_FIGURE_NAME]

    # Figures of the report in its font: the precision-recall curves at the temperature of the thresholds, the
    # training curves, and the seen and never seen groups when a manifest is given
    with matplotlib.rc_context(rc={"font.family": font_family} if font_family else None):
        title = (f"Precision-recall curves of the kept model on the {len(class_logits_record['labels']):,} "
                 f"{split_name} {sample_kind}, scores at logit temperature {thresholds['logit_temperature']:g}")
        class_threshold_values = draw_precision_recall_figure(
            class_logits_record=class_logits_record, thresholds=thresholds, class_display_names=class_display_names,
            minimum_recall=minimum_recall, title=title, output_path=output_folder / PRECISION_RECALL_FIGURE_NAME)
        kept_balanced_accuracy = draw_training_curves_figure(
            curves=training_curves, iteration=iteration, weights=weights, split_name=split_name,
            output_path=output_folder / TRAINING_CURVES_FIGURE_NAME)
        if group_comparison is not None:
            draw_groups_figure(group_comparison=group_comparison, group_plural=group_plural, sample_kind=sample_kind,
                               split_name=split_name, output_path=output_folder / GROUPS_FIGURE_NAME)
            written_paths.append(output_folder / GROUPS_FIGURE_NAME)

    # Numbers shown by the figures, to fill the tables of the report
    report_values = {
        "run": experiment_folder.name, "iteration": iteration, "weights": weights,
        "evaluation_folder": str(evaluation_folder), "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "class_recalls": {class_name: metrics["recall"][class_names.index(class_name)]
                          for class_name in class_display_names},
        "logit_temperature": thresholds["logit_temperature"], "temperature_saved": is_saved_temperature,
        "class_thresholds": class_threshold_values, "training_kept_balanced_accuracy": kept_balanced_accuracy,
        "groups": group_comparison}
    report_values_path = output_folder / REPORT_VALUES_FILE_NAME
    report_values_path.write_text(data=json.dumps(obj=report_values, indent=4))
    written_paths.append(report_values_path)

    # Every written file, then the accuracies and the temperature of the figures
    for written_path in written_paths:
        print(f"Written: {format_path_link(path=written_path)}")
    print(f"Accuracy {format_percentage(value=metrics['accuracy'])}, balanced accuracy "
          f"{format_percentage(value=metrics['balanced_accuracy'])}, thresholds at the logit temperature "
          f"{thresholds['logit_temperature']:g}" + ("" if is_saved_temperature else " (no temperature saved)"))
    return output_folder


if __name__ == "__main__":
    # Run folder of the kept classifier, in the experiments root folder of its configuration ('experiment_folder'),
    # and its kept checkpoint: the iteration and the weights, "trained" or "ema" (the moving average of the weights),
    # here the 35-class synthetic shapes classifier of the README screenshots, trained without moving average
    run_name = "CLS_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128-PT_HFLAT.128-HLN-CWB_MC35"
    experiment_folder = Path.home() / "Data" / "tutorial_runs" / "readme_screenshots" / run_name
    iteration = 2000
    weights = "trained"
    # Folder of the evaluation of the checkpoint in the run folder: 'Checkpoint_Evaluation' of
    # scripts/evaluation/evaluate_checkpoint.py, whose temperature explorer page saves the temperature of the export,
    # or 'Evaluation' of the training
    evaluation_folder_name = CHECKPOINT_EVALUATION_FOLDER_NAME
    # Name shown for every class, in the order of the curves (two classes: the first in red) or of the panels, None for
    # the class names in class index order, e.g. {"square": "Square", "circle": "Circle", ...} naming every class
    class_display_names = None
    # What the samples are and the evaluated split, written in the titles and labels ("on the 50,000 test images")
    sample_kind = "images"
    split_name = "test"
    # Left end of the recall axis of the precision-recall figure, 0, or close to 1 when every threshold keeps nearly
    # every sample (e.g. 0.98)
    minimum_recall = 0.0
    # JSON Lines manifest giving the group of every training and evaluated sample, e.g. a manifest of a dataset built
    # from the images of several cameras ('file_name', 'camera'), None to skip the seen and never seen chart, the
    # synthetic shapes having no group. The chart also reads the predictions.csv of the evaluation (save_predictions =
    # True in evaluate_checkpoint.py, or predictions_path) and the labels.json of the training split
    # (training_labels_path, None for the one of the train_data_folder of the run)
    group_manifest_path = None
    file_name_key, group_key, group_plural = "file_name", "camera", "cameras"
    training_labels_path = None
    predictions_path = None

    # Figures drawn in the font of the reports when it is installed
    draw_report_figures(experiment_folder=experiment_folder, iteration=iteration, weights=weights,
                        evaluation_folder_name=evaluation_folder_name, class_display_names=class_display_names,
                        sample_kind=sample_kind, split_name=split_name, minimum_recall=minimum_recall,
                        group_manifest_path=group_manifest_path, file_name_key=file_name_key, group_key=group_key,
                        group_plural=group_plural, training_labels_path=training_labels_path,
                        predictions_path=predictions_path, font_family=register_report_font())
