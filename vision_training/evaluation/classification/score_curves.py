"""
ROC and precision-recall curves of an evaluated checkpoint, every class being treated one against the rest: the curves
of every class, its area under the ROC curve (ROC AUC) and its average precision (the area under its step-wise
precision-recall curve, as scikit-learn computes it), their macro means over the classes defining them, and the
roc_curves.png and precision_recall_curves.png images, also logged to TensorBoard. Computed only with the per-class
report, by the classification trainer (and the sequence trainers inheriting its evaluation) from the class
probabilities of the test samples, exactly, and by the segmentation trainer from the histograms of the probabilities
of the labelled test pixels (segmentation_metrics.py), exactly at the edges of their bins. With the thresholds of the
classifiers at their target precisions, every threshold is marked on the precision-recall curves, and
precision_recall_thresholds.png zooms on them with every threshold written in the colour of its curve, so that a
threshold can be chosen from the image. The classification images write their rates as the percentages of the reports
(axes "20 %", ROC AUC and average precision "98.58 %", target precisions "99 %"), the thresholds staying between 0 and
1; the segmentation images keep 4 decimals and axes from 0 to 1 until the segmentation is moved to them.
"""

import cv2
import math
import numpy as np
import torch

from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from vision_training.evaluation.classification.confusion_matrix import IMAGE_DOTS_PER_INCH, MAXIMUM_NAME_LENGTH
from vision_training.evaluation.report_formatting import (format_percentage_label, format_percentage_tick,
                                                          format_report_value)

# Names of the macro means in the headline metrics, and names of the curve metrics of every class in the per-class
# tables and TensorBoard scalars, keyed by their column in class_metrics.csv
MACRO_ROC_AUC_NAME = "Macro ROC AUC"
MACRO_AVERAGE_PRECISION_NAME = "Macro Average Precision"
CURVE_METRIC_NAMES = {"roc_auc": "ROC AUC", "average_precision": "Average Precision"}

# Images of the curves: their name in TensorBoard (the files of the report adding ' Image'), file, title, axes, the
# curve arrays they draw and the metric given with every class
CURVE_IMAGES = {
    "ROC Curves": {"file_name": "roc_curves.png", "title": "ROC Curves", "x_label": "False Positive Rate",
                   "y_label": "True Positive Rate", "x_key": "false_positive_rates", "y_key": "true_positive_rates",
                   "metric_key": "roc_auc", "macro_name": MACRO_ROC_AUC_NAME},
    "Precision Recall Curves": {"file_name": "precision_recall_curves.png", "title": "Precision-Recall Curves",
                                "x_label": "Recall", "y_label": "Precision", "x_key": "recalls", "y_key": "precisions",
                                "metric_key": "average_precision", "macro_name": MACRO_AVERAGE_PRECISION_NAME},
    "Precision Recall Thresholds": {"file_name": "precision_recall_thresholds.png",
                                    "title": "Precision-Recall Thresholds", "x_label": "Recall",
                                    "y_label": "Precision", "x_key": "recalls", "y_key": "precisions",
                                    "metric_key": "average_precision", "macro_name": MACRO_AVERAGE_PRECISION_NAME}}

# Image zooming on the thresholds of the target precisions, drawn only for the evaluations giving thresholds
THRESHOLDS_IMAGE_NAME = "Precision Recall Thresholds"

# Largest number of points kept to draw a curve, the metrics being computed on every point
MAXIMUM_CURVE_POINTS = 2000

# Largest number of classes drawn on a single plot, above which every class has a panel of its own, the largest
# number of panels per row, and the largest number of panels, the classes with the lowest metric being drawn above it
MAXIMUM_OVERLAID_CLASSES = 10
MAXIMUM_GRID_COLUMNS = 10
MAXIMUM_GRID_PANELS = 100

# Sizes of the images in inches: the single plot, and every panel of the grid with the room of the titles
OVERLAY_FIGURE_SIZE = (7.5, 6.5)
PANEL_SIDE = 2.0
GRID_TITLE_HEIGHT = 0.8

# Largest number of intervals between the percent ticks of the horizontal axis of a panel of the grid, whose labels
# ("50 %") would run into each other with the default ticks, and the steps between them (0.5, 0.25, 0.025...)
PANEL_PERCENT_TICK_INTERVALS = 3
PANEL_PERCENT_TICK_STEPS = [1, 2.5, 5, 10]

# Colour map of the classes on the single plot, colour of the curve of the panels, and colour of the chance diagonal
# of the ROC curves and of the target precision line of the precision-recall curves
CLASS_COLOUR_MAP_NAME = "tab10"
PANEL_CURVE_COLOUR = "#1f77b4"
REFERENCE_LINE_COLOUR = "#9e9e9e"

# Written thresholds: at least MINIMUM_THRESHOLD_DECIMALS decimals, more until the threshold rounded down selects the
# same samples as the exact one, up to MAXIMUM_THRESHOLD_SIGNIFICANT_DIGITS significant digits (beyond the precision of
# a float32, so that the tiny thresholds of a confident class, e.g. 0.000000321, stay apart from the next score)
MINIMUM_THRESHOLD_DECIMALS = 4
MAXIMUM_THRESHOLD_SIGNIFICANT_DIGITS = 9

# Zoom of the thresholds image: the precision axis starts this much below the lowest target precision, and the recall
# axis this much (at least) left of the lowest marked recall, rounded down to a step of the axis
THRESHOLD_VIEW_PRECISION_MARGIN = 0.1
THRESHOLD_VIEW_MINIMUM_RECALL_MARGIN = 0.02
THRESHOLD_VIEW_AXIS_STEP = 0.02

# Offsets in points of the thresholds written on the single plot, above right for every other class and below left for
# the others (and on the panels), and the share of the recall axis beyond which a threshold is written on the left
THRESHOLD_TEXT_OFFSETS = [(6, 9), (-6, -12)]
THRESHOLD_TEXT_RIGHT_EDGE = 0.85


class ScoreCurves(TypedDict):
    """
    Curves of one class of an evaluated checkpoint, the class treated one against the rest: its numbers of positive
    and negative samples (or pixels), its ROC AUC (None without positive or negative) and average precision (None
    without positive), the points of its ROC curve (from (0, 0)) and of its step-wise precision-recall curve (from
    recall 0, empty arrays when the curve is undefined), reduced to MAXIMUM_CURVE_POINTS for the drawing, and the
    recall, precision and written value of its threshold reaching every target precision, marked on its
    precision-recall curve (None for a target precision the class never reaches, an empty list without thresholds).
    """
    number_positives: int
    number_negatives: int
    roc_auc: Optional[float]
    average_precision: Optional[float]
    false_positive_rates: np.ndarray
    true_positive_rates: np.ndarray
    recalls: np.ndarray
    precisions: np.ndarray
    threshold_points: List[Optional[Tuple[float, float, str]]]


def select_curve_point_indices(number_points: int) -> np.ndarray:
    """
    Chooses the points of a curve kept for its drawing, evenly spread along the curve with its first and last points,
    so that the curves of large test splits stay light to draw while their metrics use every point.

    Args:
        number_points (int): The number of points of the curve.

    Returns:
        np.ndarray: The indices of the kept points, in increasing order.
    """
    if number_points <= MAXIMUM_CURVE_POINTS:
        return np.arange(number_points)
    return np.unique(ar=np.linspace(start=0, stop=number_points - 1, num=MAXIMUM_CURVE_POINTS).round().astype(np.int64))


def format_threshold_label(threshold: float, next_lower_score: Optional[float]) -> str:
    """
    Writes a threshold marked on a precision-recall curve as briefly as possible without changing the samples it
    selects: rounded down to MINIMUM_THRESHOLD_DECIMALS decimals, or more while the rounded value would also select
    the next lower score, so that a threshold read from the image or a report keeps the recall and precision of the
    exact one (0.999885 and not 0.9999, which leaves samples out). Rounding down keeps every sample of the threshold.

    Args:
        threshold (float): The exact threshold, one of the scores of the class, within [0, 1].
        next_lower_score (Optional[float]): The highest score of the class below the threshold, or None when no score
            is lower.

    Returns:
        str: The threshold with at least MINIMUM_THRESHOLD_DECIMALS decimals and at most
            MAXIMUM_THRESHOLD_SIGNIFICANT_DIGITS significant digits.
    """
    # Zeros between the decimal point and the first significant digit, e.g. 3 for 0.000521
    leading_zeros = max(0, -math.floor(math.log10(threshold)) - 1) if threshold > 0 else 0
    maximum_decimals = max(MINIMUM_THRESHOLD_DECIMALS, leading_zeros + MAXIMUM_THRESHOLD_SIGNIFICANT_DIGITS)

    # Round down with more and more decimals until the rounded threshold stays above the next lower score
    for decimals in range(MINIMUM_THRESHOLD_DECIMALS, maximum_decimals + 1):
        rounded_threshold = math.floor(threshold * 10 ** decimals) / 10 ** decimals
        if next_lower_score is None or rounded_threshold > next_lower_score:
            return f"{rounded_threshold:.{decimals}f}"
    return f"{threshold:.{maximum_decimals}f}"


def compute_curves_from_score_counts(thresholds: torch.Tensor, positive_counts: torch.Tensor,
                                     negative_counts: torch.Tensor,
                                     marked_thresholds: Optional[List[Optional[float]]] = None) -> ScoreCurves:
    """
    Computes the ROC and precision-recall curves of one class from its positive and negative samples counted at every
    threshold, the samples whose score reaches a threshold being predicted as the class: every threshold cuts the
    curves once, so that tied scores move the curves together. The ROC AUC is the trapezoidal area under the ROC curve
    and the average precision the sum of the precision of every cut weighted by the recall it adds, both as
    scikit-learn computes them. Shared by the exact curves of the classifiers (every distinct score a threshold) and
    the curves of the pixels of the segmentation (every bin of probabilities a threshold).

    Args:
        thresholds (torch.Tensor): The [D] thresholds, in decreasing order.
        positive_counts (torch.Tensor): The [D] numbers of positives whose score falls at every threshold (from it to
            the next higher threshold).
        negative_counts (torch.Tensor): The [D] numbers of negatives whose score falls at every threshold.
        marked_thresholds (Optional[List[Optional[float]]]): The threshold reaching every target precision, one of
            the thresholds or None for a target never reached, whose recall and precision are marked on the
            precision-recall curve, or None to mark none. Defaults to None.

    Returns:
        ScoreCurves: The curves and metrics of the class.

    Raises:
        ValueError: If a marked threshold is not one of the thresholds.
    """
    true_positives = torch.cumsum(input=positive_counts.to(dtype=torch.float64), dim=0)
    false_positives = torch.cumsum(input=negative_counts.to(dtype=torch.float64), dim=0)
    number_positives = int(true_positives[-1]) if len(thresholds) > 0 else 0
    number_negatives = int(false_positives[-1]) if len(thresholds) > 0 else 0
    curves = {"number_positives": number_positives, "number_negatives": number_negatives, "roc_auc": None,
              "average_precision": None, "false_positive_rates": np.empty(shape=0),
              "true_positive_rates": np.empty(shape=0), "recalls": np.empty(shape=0), "precisions": np.empty(shape=0),
              "threshold_points": []}

    # ROC curve from (0, 0), defined with positives and negatives, and the trapezoidal area under it
    if number_positives > 0 and number_negatives > 0:
        origin = torch.zeros(size=(1,), dtype=torch.float64)
        false_positive_rates = torch.cat(tensors=[origin, false_positives / number_negatives])
        true_positive_rates = torch.cat(tensors=[origin, true_positives / number_positives])
        curves["roc_auc"] = float(torch.trapezoid(y=true_positive_rates, x=false_positive_rates))
        kept_points = select_curve_point_indices(number_points=len(false_positive_rates))
        curves["false_positive_rates"] = false_positive_rates.numpy()[kept_points]
        curves["true_positive_rates"] = true_positive_rates.numpy()[kept_points]
    if number_positives == 0:
        return curves

    # Precision-recall curve, defined with positives: the precision of every cut over the recall it adds
    recalls = true_positives / number_positives
    precisions = true_positives / (true_positives + false_positives)
    added_recalls = recalls - torch.cat(tensors=[torch.zeros(size=(1,), dtype=torch.float64), recalls[:-1]])
    curves["average_precision"] = float((added_recalls * precisions).sum())

    # Points of the step-wise curve from recall 0, every step taking the precision of the cut ending it
    step_recalls = torch.cat(tensors=[torch.zeros(size=(1,), dtype=torch.float64), recalls])
    step_precisions = torch.cat(tensors=[precisions[:1], precisions])
    kept_points = select_curve_point_indices(number_points=len(step_recalls))
    curves["recalls"], curves["precisions"] = step_recalls.numpy()[kept_points], step_precisions.numpy()[kept_points]

    # Recall, precision and written value of the threshold reaching every target precision, marked on the curve
    for marked_threshold in marked_thresholds or []:
        if marked_threshold is None:
            curves["threshold_points"].append(None)
            continue
        marked_indices = torch.nonzero(input=thresholds == marked_threshold).flatten()
        if len(marked_indices) != 1:
            raise ValueError(f"The marked threshold {marked_threshold} is not one of the thresholds of the curves.")
        marked_index = int(marked_indices[0])
        next_lower_score = float(thresholds[marked_index + 1]) if marked_index + 1 < len(thresholds) else None
        curves["threshold_points"].append((float(recalls[marked_index]), float(precisions[marked_index]),
                                           format_threshold_label(threshold=marked_threshold,
                                                                  next_lower_score=next_lower_score)))
    return curves


def compute_class_score_curves(probabilities: torch.Tensor, positive_masks: torch.Tensor,
                               class_thresholds: Optional[List[List[Optional[Tuple[float, float]]]]] = None
                               ) -> List[ScoreCurves]:
    """
    Computes the exact ROC and precision-recall curves of every class of a classifier from the class probabilities of
    the test samples, every class treated one against the rest like its thresholds at the target precisions
    (find_class_thresholds), whose recall and precision are marked on its precision-recall curve.

    Args:
        probabilities (torch.Tensor): The [N, K] class probabilities of the test samples (or labelled frames).
        positive_masks (torch.Tensor): The [N, K] booleans telling whether every sample belongs to every class.
        class_thresholds (Optional[List[List[Optional[Tuple[float, float]]]]]): For every target precision, the
            threshold reaching it and its recall for every class (None for a class without one), or None to mark
            none. Defaults to None.

    Returns:
        List[ScoreCurves]: The curves and metrics of every class, in class index order.
    """
    class_curves = []
    for class_index in range(probabilities.shape[1]):
        # Distinct scores of the class, with the positive and negative samples at each, by decreasing score
        distinct_scores, score_indices, score_counts = torch.unique(input=probabilities[:, class_index], sorted=True,
                                                                    return_inverse=True, return_counts=True)
        positive_counts = torch.bincount(input=score_indices,
                                         weights=positive_masks[:, class_index].to(dtype=torch.float64),
                                         minlength=len(distinct_scores))
        negative_counts = score_counts.to(dtype=torch.float64) - positive_counts

        # Curves of the class, its threshold at every target precision being marked when it has one
        marked_thresholds = None if class_thresholds is None else [
            None if target_thresholds[class_index] is None else target_thresholds[class_index][0]
            for target_thresholds in class_thresholds]
        class_curves.append(compute_curves_from_score_counts(
            thresholds=distinct_scores.flip(dims=[0]), positive_counts=positive_counts.flip(dims=[0]),
            negative_counts=negative_counts.flip(dims=[0]), marked_thresholds=marked_thresholds))
    return class_curves


def compute_histogram_score_curves(score_histograms: torch.Tensor) -> List[ScoreCurves]:
    """
    Computes the ROC and precision-recall curves of every class of the segmentation from the histograms of the
    probabilities of the labelled test pixels (compute_pixel_score_histograms), every class treated one against the
    rest: the lower edge of every bin holding pixels is a threshold, so that the curves are exact at the edges of the
    bins, the pixels of a bin moving the curves together like tied scores.

    Args:
        score_histograms (torch.Tensor): The [K, 2, B] numbers of pixels of every class whose probability of the
            class falls in every one of B bins of equal width over [0, 1], the pixels of other classes (negatives)
            first and the pixels of the class (positives) second.

    Returns:
        List[ScoreCurves]: The curves and metrics of every class, in class index order.
    """
    number_bins = score_histograms.shape[-1]
    class_curves = []
    for class_histograms in score_histograms.to(dtype=torch.float64):
        # Bins holding pixels, by decreasing probability, the lower edge of every bin being its threshold
        filled_bins = torch.nonzero(input=class_histograms.sum(dim=0) > 0).flatten().flip(dims=[0])
        class_curves.append(compute_curves_from_score_counts(
            thresholds=filled_bins.to(dtype=torch.float64) / number_bins,
            positive_counts=class_histograms[1, filled_bins], negative_counts=class_histograms[0, filled_bins]))
    return class_curves


def compute_macro_curve_metrics(class_curves: List[ScoreCurves]) -> Dict[str, float]:
    """
    Computes the headline means of the curve metrics of an evaluated checkpoint: the ROC AUC averaged over the classes
    with positives and negatives, and the average precision averaged over the classes with positives, like the macro
    averages of the other metrics leaving out the classes absent from the test split. A mean without any class is
    NaN, written "—" in the report.

    Args:
        class_curves (List[ScoreCurves]): The curves and metrics of every class.

    Returns:
        Dict[str, float]: The 'Macro ROC AUC' and the 'Macro Average Precision'.
    """
    macro_metrics = {}
    for metric_key, macro_name in [("roc_auc", MACRO_ROC_AUC_NAME),
                                   ("average_precision", MACRO_AVERAGE_PRECISION_NAME)]:
        defined_values = [curves[metric_key] for curves in class_curves if curves[metric_key] is not None]
        macro_metrics[macro_name] = float(np.mean(a=defined_values)) if defined_values else float("nan")
    return macro_metrics


def select_drawn_classes(class_curves: List[ScoreCurves], metric_key: str) -> List[int]:
    """
    Chooses the classes drawn on the image of a kind of curve: every class up to MAXIMUM_GRID_PANELS, otherwise the
    classes with the lowest metric, the classes without metric being left out, so that the image stays readable and
    shows the classes needing attention.

    Args:
        class_curves (List[ScoreCurves]): The curves and metrics of every class.
        metric_key (str): The metric of the curve, 'roc_auc' or 'average_precision'.

    Returns:
        List[int]: The indices of the drawn classes, in class index order.
    """
    if len(class_curves) <= MAXIMUM_GRID_PANELS:
        return list(range(len(class_curves)))
    # Classes with a metric, sorted by increasing metric (ties in class order), the lowest ones being kept
    defined_classes = sorted((curves[metric_key], class_index) for class_index, curves in enumerate(class_curves)
                             if curves[metric_key] is not None)
    return sorted(class_index for _, class_index in defined_classes[:MAXIMUM_GRID_PANELS])


def compute_threshold_view_limits(class_curves: List[ScoreCurves], target_precisions: List[float]
                                  ) -> Tuple[float, float]:
    """
    Chooses the lower ends of the axes of the thresholds image, so that it zooms on the marked thresholds: the precision
    axis starts THRESHOLD_VIEW_PRECISION_MARGIN below the lowest target precision, and the recall axis left of the
    lowest marked recall by half the recall missing from it (at least THRESHOLD_VIEW_MINIMUM_RECALL_MARGIN), rounded
    down to THRESHOLD_VIEW_AXIS_STEP, e.g. from 0.96 when every marked recall is above 0.98.

    Args:
        class_curves (List[ScoreCurves]): The curves of every class, with their marked thresholds.
        target_precisions (List[float]): The target precisions of the thresholds.

    Returns:
        Tuple[float, float]: The lower ends of the recall and precision axes, the recall axis starting at 0 when no
            class reaches a target precision.
    """
    minimum_precision = max(0.0, min(target_precisions) - THRESHOLD_VIEW_PRECISION_MARGIN)
    marked_recalls = [threshold_point[0] for curves in class_curves for threshold_point in curves["threshold_points"]
                      if threshold_point is not None]
    if not marked_recalls:
        return 0.0, minimum_precision

    # Left of the lowest marked recall by half the recall it misses, on a round step of the axis
    lowest_recall = min(marked_recalls)
    recall_margin = max(THRESHOLD_VIEW_MINIMUM_RECALL_MARGIN, (1.0 - lowest_recall) / 2)
    minimum_recall = math.floor((lowest_recall - recall_margin) / THRESHOLD_VIEW_AXIS_STEP) * THRESHOLD_VIEW_AXIS_STEP
    return max(0.0, minimum_recall), minimum_precision


def draw_reference_lines(axes: Axes, image_name: str, target_precisions: Optional[List[float]],
                         add_labels: bool, in_percent: bool) -> None:
    """
    Draws the reference lines of a plot of curves: the chance diagonal of the ROC curves, and one line per target
    precision of the thresholds on the precision-recall curves when the evaluation gives thresholds, with their legend
    entries (and the entry of the points of the thresholds) on the single plot or on the first panel of the grid.

    Args:
        axes (Axes): The axes of the plot.
        image_name (str): The name of the image, a key of CURVE_IMAGES.
        target_precisions (Optional[List[float]]): The precisions targeted by the thresholds, or None without
            thresholds.
        add_labels (bool): Whether the lines get legend entries.
        in_percent (bool): Whether the target precisions of the legend are written as percentages ("99 %").
    """
    if image_name == "ROC Curves":
        axes.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color=REFERENCE_LINE_COLOUR,
                  label="Chance" if add_labels else None)
    elif target_precisions:
        # One dotted line per target precision, a single legend entry naming them all
        written_precisions = " · ".join(format_percentage_label(value=target_precision) if in_percent else
                                        f"{target_precision:g}" for target_precision in target_precisions)
        for line_index, target_precision in enumerate(target_precisions):
            axes.axhline(y=target_precision, linestyle=":", linewidth=1.0, color=REFERENCE_LINE_COLOUR,
                         label=f"Target Precision {written_precisions}" if add_labels and line_index == 0 else None)

        # Legend entry of the points of the thresholds, drawn without any point
        if add_labels:
            axes.scatter(x=[], y=[], color="white", edgecolors="black", s=30,
                         label=f"Threshold Reaching {written_precisions}")


def draw_class_curve(axes: Axes, curves: ScoreCurves, image_name: str, colour: Any, label: Optional[str],
                     threshold_text_offset: Optional[Tuple[int, int]] = None) -> None:
    """
    Draws the curve of one class on a plot, the precision-recall curve as steps like its average precision, with the
    point of its threshold at every target precision on the precision-recall curves and, on the thresholds image, the
    value of every threshold written next to its point in the colour of the curve.

    Args:
        axes (Axes): The axes of the plot, whose limits are already set when the thresholds are written.
        curves (ScoreCurves): The curves and metrics of the class.
        image_name (str): The name of the image, a key of CURVE_IMAGES.
        colour (Any): The matplotlib colour of the curve.
        label (Optional[str]): The legend entry of the curve, or None.
        threshold_text_offset (Optional[Tuple[int, int]]): The offset in points of the written thresholds from their
            point, to the right for a positive first value, or None to write none. Defaults to None.
    """
    image_settings = CURVE_IMAGES[image_name]
    axes.plot(curves[image_settings["x_key"]], curves[image_settings["y_key"]], color=colour, linewidth=1.5,
              drawstyle="default" if image_name == "ROC Curves" else "steps-pre", label=label)
    if image_name == "ROC Curves":
        return

    # Point of the threshold of every target precision the class reaches
    left_end, right_end = axes.get_xlim()
    for threshold_point in curves["threshold_points"]:
        if threshold_point is None:
            continue
        point_recall, point_precision, threshold_label = threshold_point
        axes.scatter(x=[point_recall], y=[point_precision], color=colour, edgecolors="black", s=30, zorder=3)
        if threshold_text_offset is None:
            continue

        # Threshold written in the colour of the curve on a white box, on the left of the points near the right edge
        horizontal_offset, vertical_offset = threshold_text_offset
        if horizontal_offset > 0 and (point_recall - left_end) / (right_end - left_end) > THRESHOLD_TEXT_RIGHT_EDGE:
            horizontal_offset = -horizontal_offset
        axes.annotate(text=threshold_label, xy=(point_recall, point_precision),
                      xytext=(horizontal_offset, vertical_offset), textcoords="offset points",
                      ha="left" if horizontal_offset > 0 else "right", va="center", fontsize=7, fontweight="bold",
                      color=colour, bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none",
                                          "alpha": 0.85}, zorder=4)


def format_axes(axes: Axes, font_size: int, in_percent: bool,
                view_limits: Optional[Tuple[float, float]] = None) -> None:
    """
    Sets the range, the aspect, the grid and the tick size of a plot of curves: the [0, 1] square, or the zoom of the
    thresholds image starting at the given lower ends, the ticks written as percentages ("20 %") for the classification
    images.

    Args:
        axes (Axes): The axes of the plot.
        font_size (int): The size of the tick labels.
        in_percent (bool): Whether the ticks of both axes are written as percentages.
        view_limits (Optional[Tuple[float, float]]): The lower ends of the recall and precision axes of the thresholds
            image (compute_threshold_view_limits), or None for the whole square. Defaults to None.
    """
    if view_limits is None:
        axes.set_xlim(left=0.0, right=1.0)
        axes.set_ylim(bottom=0.0, top=1.02)
        axes.set_aspect(aspect="equal")
    else:
        # Zoom on the thresholds, with a small margin above and right of 1 so that the points there stay whole
        minimum_recall, minimum_precision = view_limits
        axes.set_xlim(left=minimum_recall, right=1.0 + 0.01 * (1.0 - minimum_recall))
        axes.set_ylim(bottom=minimum_precision, top=1.0 + 0.02 * (1.0 - minimum_precision))
    axes.grid(visible=True, alpha=0.3)
    axes.tick_params(labelsize=font_size)
    if in_percent:
        axes.xaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))
        axes.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))


def draw_score_curves(class_curves: List[ScoreCurves], class_names: List[str], image_name: str, title: str,
                      target_precisions: Optional[List[float]] = None, in_percent: bool = False) -> np.ndarray:
    """
    Draws the ROC or precision-recall curves of every class of an evaluated checkpoint as an image, saved in its
    evaluation folder and logged to TensorBoard by the per-class report: up to MAXIMUM_OVERLAID_CLASSES classes on a
    single plot, each in its colour with its metric in the legend, otherwise one panel per class titled with its
    metric (the MAXIMUM_GRID_PANELS classes with the lowest metric above that number). The ROC curves come with the
    chance diagonal, the precision-recall curves with the target precisions and the point of the threshold reaching
    each on every class, and the thresholds image zooms on these points with every threshold written in the colour of
    its curve. It is drawn on a matplotlib figure of its own, without pyplot.

    Args:
        class_curves (List[ScoreCurves]): The curves and metrics of every class.
        class_names (List[str]): The name of every class.
        image_name (str): The name of the image, a key of CURVE_IMAGES.
        title (str): The title of the image.
        target_precisions (Optional[List[float]]): The precisions targeted by the thresholds, or None for an
            evaluation without thresholds (the segmentation). Defaults to None.
        in_percent (bool): Whether the axes, the metric of every class and the target precisions are written as
            percentages, True for the classification images. Defaults to False.

    Returns:
        np.ndarray: The [H, W, 3] uint8 RGB image.

    Raises:
        ValueError: If the thresholds image is asked without target precisions.
    """
    image_settings = CURVE_IMAGES[image_name]
    metric_key, metric_name = image_settings["metric_key"], CURVE_METRIC_NAMES[image_settings["metric_key"]]

    # The thresholds image zooms on the marked thresholds and writes them, the other images show the whole square
    view_limits = None
    if image_name == THRESHOLDS_IMAGE_NAME:
        if not target_precisions:
            raise ValueError(f"The '{THRESHOLDS_IMAGE_NAME}' image needs the target precisions of the thresholds.")
        view_limits = compute_threshold_view_limits(class_curves=class_curves, target_precisions=target_precisions)

    # Few classes: a single plot, every class in its colour with its metric in the legend
    if len(class_curves) <= MAXIMUM_OVERLAID_CLASSES:
        figure = Figure(figsize=OVERLAY_FIGURE_SIZE, dpi=IMAGE_DOTS_PER_INCH, layout="constrained")
        canvas = FigureCanvasAgg(figure=figure)
        axes = figure.add_subplot()
        class_colour_map = colormaps[CLASS_COLOUR_MAP_NAME]
        format_axes(axes=axes, font_size=8, in_percent=in_percent, view_limits=view_limits)
        draw_reference_lines(axes=axes, image_name=image_name, target_precisions=target_precisions, add_labels=True,
                             in_percent=in_percent)
        for class_index, (curves, class_name) in enumerate(zip(class_curves, class_names)):
            # Thresholds written above right of the points of every other class and below left of the others
            metric_text = format_report_value(value=curves[metric_key], in_percent=in_percent)
            class_label = f"{class_name[:MAXIMUM_NAME_LENGTH]} · {metric_text}"
            draw_class_curve(axes=axes, curves=curves, image_name=image_name, colour=class_colour_map(class_index),
                             label=class_label, threshold_text_offset=None if view_limits is None else
                             THRESHOLD_TEXT_OFFSETS[class_index % len(THRESHOLD_TEXT_OFFSETS)])
        axes.set_xlabel(xlabel=image_settings["x_label"], fontsize=10, fontweight="bold")
        axes.set_ylabel(ylabel=image_settings["y_label"], fontsize=10, fontweight="bold")
        axes.legend(title=f"Class · {metric_name}", fontsize=8, title_fontsize=8,
                    loc="lower right" if image_name == "ROC Curves" else "lower left")
        figure.suptitle(t=title, fontsize=12, fontweight="bold")
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[..., :3].copy()

    # Many classes: one panel per drawn class, in rows of at most MAXIMUM_GRID_COLUMNS panels
    drawn_classes = select_drawn_classes(class_curves=class_curves, metric_key=metric_key)
    number_columns = min(MAXIMUM_GRID_COLUMNS, max(1, math.ceil(math.sqrt(len(drawn_classes)))))
    number_rows = max(1, math.ceil(len(drawn_classes) / number_columns))
    figure = Figure(figsize=(number_columns * PANEL_SIDE, number_rows * PANEL_SIDE + GRID_TITLE_HEIGHT),
                    dpi=IMAGE_DOTS_PER_INCH, layout="constrained")
    canvas = FigureCanvasAgg(figure=figure)
    # Panels with their own axes and the same limits: shared axes would make every limit set on a panel update all the
    # others, 1.9 instead of 1.0 seconds for an image of 71 panels
    panel_axes = figure.subplots(nrows=number_rows, ncols=number_columns, squeeze=False)

    # Every panel: the reference line, the curve of its class and its metric in its title, the unused panels hidden
    for panel_index, axes in enumerate(panel_axes.flatten()):
        if panel_index >= len(drawn_classes):
            axes.set_axis_off()
            continue
        class_index = drawn_classes[panel_index]
        format_axes(axes=axes, font_size=6, in_percent=in_percent, view_limits=view_limits)
        if in_percent:
            # Fewer percent ticks along the narrow panel, so that their labels stay apart
            axes.xaxis.set_major_locator(locator=MaxNLocator(nbins=PANEL_PERCENT_TICK_INTERVALS,
                                                             steps=PANEL_PERCENT_TICK_STEPS))
        draw_reference_lines(axes=axes, image_name=image_name, target_precisions=target_precisions,
                             add_labels=panel_index == 0, in_percent=in_percent)
        draw_class_curve(axes=axes, curves=class_curves[class_index], image_name=image_name,
                         colour=PANEL_CURVE_COLOUR, label=None,
                         threshold_text_offset=None if view_limits is None else THRESHOLD_TEXT_OFFSETS[-1])
        metric_text = format_report_value(value=class_curves[class_index][metric_key], in_percent=in_percent)
        axes.set_title(label=f"{class_names[class_index][:MAXIMUM_NAME_LENGTH]}\n{metric_name} {metric_text}",
                       fontsize=7)

        # Axis names and tick values on the lowest panel of every column and on the first column only
        is_lowest_panel = panel_index + number_columns >= len(drawn_classes)
        is_first_column = panel_index % number_columns == 0
        axes.tick_params(labelbottom=is_lowest_panel, labelleft=is_first_column)
        if is_lowest_panel:
            axes.set_xlabel(xlabel=image_settings["x_label"], fontsize=8, fontweight="bold")
        if is_first_column:
            axes.set_ylabel(ylabel=image_settings["y_label"], fontsize=8, fontweight="bold")

    # Reference legend of the whole grid below the panels, and the title telling when only the lowest classes are drawn
    if panel_axes[0, 0].get_legend_handles_labels()[0]:
        figure.legend(loc="outside lower center", ncols=2, fontsize=8)
    if len(drawn_classes) < len(class_curves):
        title += f" · The {len(drawn_classes)} Lowest {metric_name} Of {len(class_curves)} Classes"
    figure.suptitle(t=title, fontsize=12, fontweight="bold")
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def save_score_curve_images(class_curves: List[ScoreCurves], class_names: List[str], output_folder: Path,
                            title_suffix: str, target_precisions: Optional[List[float]] = None,
                            in_percent: bool = False) -> Tuple[Dict[str, Path], Dict[str, np.ndarray]]:
    """
    Draws the ROC and precision-recall curves of an evaluated checkpoint (draw_score_curves) and saves them as
    'roc_curves.png', 'precision_recall_curves.png' and, for the evaluations giving thresholds, the zoom on the
    thresholds 'precision_recall_thresholds.png' in its evaluation folder, for the per-class report, which also logs
    the returned images to TensorBoard: both come from the same arrays, so the saved images and the logged ones are
    identical. Every title gives its macro mean on a second line, so that the longest titles (the pixels of a
    segmentation evaluation) stay within the width of the single plot.

    Args:
        class_curves (List[ScoreCurves]): The curves and metrics of every class.
        class_names (List[str]): The name of every class.
        output_folder (Path): The evaluation folder of the checkpoint, created if needed.
        title_suffix (str): The text put after the name of the curves in the titles, e.g. ' · Iteration 4000'.
        target_precisions (Optional[List[float]]): The precisions targeted by the thresholds, or None for an
            evaluation without thresholds. Defaults to None.
        in_percent (bool): Whether the images write their rates as percentages, True for the classification images.
            Defaults to False.

    Returns:
        Tuple[Dict[str, Path], Dict[str, np.ndarray]]: The paths of the images keyed by '<name> Image', for the files
            of the report, and the [H, W, 3] uint8 RGB images keyed by name ('ROC Curves', 'Precision Recall Curves'
            and, with thresholds, 'Precision Recall Thresholds').
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    macro_metrics = compute_macro_curve_metrics(class_curves=class_curves)
    image_paths, images = {}, {}
    for image_name, image_settings in CURVE_IMAGES.items():
        # The thresholds image only for the evaluations giving thresholds
        if image_name == THRESHOLDS_IMAGE_NAME and not target_precisions:
            continue

        # Draw the curves with their macro mean on the second line of the title, then save them as the TensorBoard image
        macro_name = image_settings["macro_name"]
        images[image_name] = draw_score_curves(
            class_curves=class_curves, class_names=class_names, image_name=image_name,
            title=f"{image_settings['title']}{title_suffix}\n{macro_name} "
                  f"{format_report_value(value=macro_metrics[macro_name], in_percent=in_percent)}",
            target_precisions=target_precisions, in_percent=in_percent)
        image_paths[f"{image_name} Image"] = output_folder / image_settings["file_name"]
        cv2.imwrite(filename=str(image_paths[f"{image_name} Image"]),
                    img=cv2.cvtColor(src=images[image_name], code=cv2.COLOR_RGB2BGR))
    return image_paths, images
