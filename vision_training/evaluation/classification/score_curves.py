"""
ROC and precision-recall curves of an evaluated checkpoint, every class being treated one against the rest: the curves
of every class, its area under the ROC curve (ROC AUC) and its average precision (the area under its step-wise
precision-recall curve, as scikit-learn computes it), their macro means over the classes defining them, and the
roc_curves.png and precision_recall_curves.png images, also logged to TensorBoard. Computed only with the per-class
report, by the classification trainer (and the sequence trainers inheriting its evaluation) from the class
probabilities of the test samples, exactly, and by the segmentation trainer from the histograms of the probabilities
of the labelled test pixels (segmentation_metrics.py), exactly at the edges of their bins.
"""

import cv2
import math
import numpy as np
import torch

from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from vision_training.evaluation.classification.confusion_matrix import IMAGE_DOTS_PER_INCH, MAXIMUM_NAME_LENGTH
from vision_training.evaluation.report_formatting import format_report_value

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
                                "metric_key": "average_precision", "macro_name": MACRO_AVERAGE_PRECISION_NAME}}

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

# Colour map of the classes on the single plot, colour of the curve of the panels, and colour of the chance diagonal
# of the ROC curves and of the target precision line of the precision-recall curves
CLASS_COLOUR_MAP_NAME = "tab10"
PANEL_CURVE_COLOUR = "#1f77b4"
REFERENCE_LINE_COLOUR = "#9e9e9e"


class ScoreCurves(TypedDict):
    """
    Curves of one class of an evaluated checkpoint, the class treated one against the rest: its numbers of positive
    and negative samples (or pixels), its ROC AUC (None without positive or negative) and average precision (None
    without positive), the points of its ROC curve (from (0, 0)) and of its step-wise precision-recall curve (from
    recall 0, empty arrays when the curve is undefined), reduced to MAXIMUM_CURVE_POINTS for the drawing, and the
    recall and precision of its threshold reaching the target precision, marked on its precision-recall curve (None
    without one).
    """
    number_positives: int
    number_negatives: int
    roc_auc: Optional[float]
    average_precision: Optional[float]
    false_positive_rates: np.ndarray
    true_positive_rates: np.ndarray
    recalls: np.ndarray
    precisions: np.ndarray
    threshold_point: Optional[Tuple[float, float]]


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


def compute_curves_from_score_counts(thresholds: torch.Tensor, positive_counts: torch.Tensor,
                                     negative_counts: torch.Tensor,
                                     marked_threshold: Optional[float] = None) -> ScoreCurves:
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
        marked_threshold (Optional[float]): The threshold reaching the target precision, one of the thresholds, whose
            recall and precision are marked on the precision-recall curve, or None. Defaults to None.

    Returns:
        ScoreCurves: The curves and metrics of the class.

    Raises:
        ValueError: If the marked threshold is not one of the thresholds.
    """
    true_positives = torch.cumsum(input=positive_counts.to(dtype=torch.float64), dim=0)
    false_positives = torch.cumsum(input=negative_counts.to(dtype=torch.float64), dim=0)
    number_positives = int(true_positives[-1]) if len(thresholds) > 0 else 0
    number_negatives = int(false_positives[-1]) if len(thresholds) > 0 else 0
    curves = {"number_positives": number_positives, "number_negatives": number_negatives, "roc_auc": None,
              "average_precision": None, "false_positive_rates": np.empty(shape=0),
              "true_positive_rates": np.empty(shape=0), "recalls": np.empty(shape=0), "precisions": np.empty(shape=0),
              "threshold_point": None}

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

    # Recall and precision of the threshold reaching the target precision, marked on the curve
    if marked_threshold is not None:
        marked_indices = torch.nonzero(input=thresholds == marked_threshold).flatten()
        if len(marked_indices) != 1:
            raise ValueError(f"The marked threshold {marked_threshold} is not one of the thresholds of the curves.")
        marked_index = int(marked_indices[0])
        curves["threshold_point"] = (float(recalls[marked_index]), float(precisions[marked_index]))
    return curves


def compute_class_score_curves(probabilities: torch.Tensor, positive_masks: torch.Tensor,
                               class_thresholds: Optional[List[Optional[Tuple[float, float]]]] = None
                               ) -> List[ScoreCurves]:
    """
    Computes the exact ROC and precision-recall curves of every class of a classifier from the class probabilities of
    the test samples, every class treated one against the rest like its threshold at the target precision
    (find_class_thresholds), whose recall and precision are marked on its precision-recall curve.

    Args:
        probabilities (torch.Tensor): The [N, K] class probabilities of the test samples (or labelled frames).
        positive_masks (torch.Tensor): The [N, K] booleans telling whether every sample belongs to every class.
        class_thresholds (Optional[List[Optional[Tuple[float, float]]]]): The threshold reaching the target precision
            and its recall for every class (None for a class without one), or None to mark none. Defaults to None.

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

        # Curves of the class, its threshold at the target precision being marked when it has one
        class_threshold = None if class_thresholds is None else class_thresholds[class_index]
        class_curves.append(compute_curves_from_score_counts(
            thresholds=distinct_scores.flip(dims=[0]), positive_counts=positive_counts.flip(dims=[0]),
            negative_counts=negative_counts.flip(dims=[0]),
            marked_threshold=None if class_threshold is None else class_threshold[0]))
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


def draw_reference_lines(axes: Axes, image_name: str, target_precision: Optional[float], add_labels: bool) -> None:
    """
    Draws the reference lines of a plot of curves: the chance diagonal of the ROC curves, and the target precision of
    the thresholds on the precision-recall curves when the evaluation gives thresholds, with their legend entries (and
    the entry of the points of the thresholds) on the single plot or on the first panel of the grid.

    Args:
        axes (Axes): The axes of the plot.
        image_name (str): The name of the image, a key of CURVE_IMAGES.
        target_precision (Optional[float]): The precision targeted by the thresholds, or None without thresholds.
        add_labels (bool): Whether the lines get legend entries.
    """
    if image_name == "ROC Curves":
        axes.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color=REFERENCE_LINE_COLOUR,
                  label="Chance" if add_labels else None)
    elif target_precision is not None:
        axes.axhline(y=target_precision, linestyle=":", linewidth=1.0, color=REFERENCE_LINE_COLOUR,
                     label=f"Target Precision {target_precision:g}" if add_labels else None)

        # Legend entry of the points of the thresholds, drawn without any point
        if add_labels:
            axes.scatter(x=[], y=[], color="white", edgecolors="black", s=30,
                         label=f"Threshold Reaching {target_precision:g}")


def draw_class_curve(axes: Axes, curves: ScoreCurves, image_name: str, colour: Any, label: Optional[str]) -> None:
    """
    Draws the curve of one class on a plot, the precision-recall curve as steps like its average precision, with the
    point of its threshold at the target precision on the precision-recall curve.

    Args:
        axes (Axes): The axes of the plot.
        curves (ScoreCurves): The curves and metrics of the class.
        image_name (str): The name of the image, a key of CURVE_IMAGES.
        colour (Any): The matplotlib colour of the curve.
        label (Optional[str]): The legend entry of the curve, or None.
    """
    image_settings = CURVE_IMAGES[image_name]
    axes.plot(curves[image_settings["x_key"]], curves[image_settings["y_key"]], color=colour, linewidth=1.5,
              drawstyle="default" if image_name == "ROC Curves" else "steps-pre", label=label)
    if image_name == "Precision Recall Curves" and curves["threshold_point"] is not None:
        axes.scatter(x=[curves["threshold_point"][0]], y=[curves["threshold_point"][1]], color=colour,
                     edgecolors="black", s=30, zorder=3)


def format_axes(axes: Axes, font_size: int) -> None:
    """
    Sets the [0, 1] range, the square aspect, the grid and the tick size of a plot of curves.

    Args:
        axes (Axes): The axes of the plot.
        font_size (int): The size of the tick labels.
    """
    axes.set_xlim(left=0.0, right=1.0)
    axes.set_ylim(bottom=0.0, top=1.02)
    axes.set_aspect(aspect="equal")
    axes.grid(visible=True, alpha=0.3)
    axes.tick_params(labelsize=font_size)


def draw_score_curves(class_curves: List[ScoreCurves], class_names: List[str], image_name: str, title: str,
                      target_precision: Optional[float] = None) -> np.ndarray:
    """
    Draws the ROC or precision-recall curves of every class of an evaluated checkpoint as an image, saved in its
    evaluation folder and logged to TensorBoard by the per-class report: up to MAXIMUM_OVERLAID_CLASSES classes on a
    single plot, each in its colour with its metric in the legend, otherwise one panel per class titled with its
    metric (the MAXIMUM_GRID_PANELS classes with the lowest metric above that number). The ROC curves come with the
    chance diagonal, the precision-recall curves with the target precision and the point of the threshold reaching it
    on every class having one. It is drawn on a matplotlib figure of its own, without pyplot.

    Args:
        class_curves (List[ScoreCurves]): The curves and metrics of every class.
        class_names (List[str]): The name of every class.
        image_name (str): The name of the image, 'ROC Curves' or 'Precision Recall Curves'.
        title (str): The title of the image.
        target_precision (Optional[float]): The precision targeted by the thresholds, or None for an evaluation
            without thresholds (the segmentation). Defaults to None.

    Returns:
        np.ndarray: The [H, W, 3] uint8 RGB image.
    """
    image_settings = CURVE_IMAGES[image_name]
    metric_key, metric_name = image_settings["metric_key"], CURVE_METRIC_NAMES[image_settings["metric_key"]]

    # Few classes: a single plot, every class in its colour with its metric in the legend
    if len(class_curves) <= MAXIMUM_OVERLAID_CLASSES:
        figure = Figure(figsize=OVERLAY_FIGURE_SIZE, dpi=IMAGE_DOTS_PER_INCH, layout="constrained")
        canvas = FigureCanvasAgg(figure=figure)
        axes = figure.add_subplot()
        class_colour_map = colormaps[CLASS_COLOUR_MAP_NAME]
        draw_reference_lines(axes=axes, image_name=image_name, target_precision=target_precision, add_labels=True)
        for class_index, (curves, class_name) in enumerate(zip(class_curves, class_names)):
            class_label = f"{class_name[:MAXIMUM_NAME_LENGTH]} · {format_report_value(value=curves[metric_key])}"
            draw_class_curve(axes=axes, curves=curves, image_name=image_name, colour=class_colour_map(class_index),
                             label=class_label)
        axes.set_xlabel(xlabel=image_settings["x_label"], fontsize=10, fontweight="bold")
        axes.set_ylabel(ylabel=image_settings["y_label"], fontsize=10, fontweight="bold")
        format_axes(axes=axes, font_size=8)
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
    panel_axes = figure.subplots(nrows=number_rows, ncols=number_columns, sharex=True, sharey=True, squeeze=False)

    # Every panel: the reference line, the curve of its class and its metric in its title, the unused panels hidden
    for panel_index, axes in enumerate(panel_axes.flatten()):
        if panel_index >= len(drawn_classes):
            axes.set_axis_off()
            continue
        class_index = drawn_classes[panel_index]
        draw_reference_lines(axes=axes, image_name=image_name, target_precision=target_precision,
                             add_labels=panel_index == 0)
        draw_class_curve(axes=axes, curves=class_curves[class_index], image_name=image_name,
                         colour=PANEL_CURVE_COLOUR, label=None)
        axes.set_title(label=f"{class_names[class_index][:MAXIMUM_NAME_LENGTH]}\n{metric_name} "
                             f"{format_report_value(value=class_curves[class_index][metric_key])}", fontsize=7)
        format_axes(axes=axes, font_size=6)

        # Axis names on the lowest panel of every column and on the first column, with the ticks of the lowest panels
        if panel_index + number_columns >= len(drawn_classes):
            axes.set_xlabel(xlabel=image_settings["x_label"], fontsize=8, fontweight="bold")
            axes.tick_params(labelbottom=True)
        if panel_index % number_columns == 0:
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
                            title_suffix: str, target_precision: Optional[float] = None
                            ) -> Tuple[Dict[str, Path], Dict[str, np.ndarray]]:
    """
    Draws the ROC and precision-recall curves of an evaluated checkpoint (draw_score_curves) and saves them as
    'roc_curves.png' and 'precision_recall_curves.png' in its evaluation folder, for the per-class report, which also
    logs the returned images to TensorBoard: both come from the same arrays, so the saved images and the logged ones
    are identical. Every title gives its macro mean on a second line, so that the longest titles (the pixels of a
    segmentation evaluation) stay within the width of the single plot.

    Args:
        class_curves (List[ScoreCurves]): The curves and metrics of every class.
        class_names (List[str]): The name of every class.
        output_folder (Path): The evaluation folder of the checkpoint, created if needed.
        title_suffix (str): The text put after the name of the curves in the titles, e.g. ' · Iteration 4000'.
        target_precision (Optional[float]): The precision targeted by the thresholds, or None for an evaluation
            without thresholds. Defaults to None.

    Returns:
        Tuple[Dict[str, Path], Dict[str, np.ndarray]]: The paths of the images keyed by '<name> Image', for the files
            of the report, and the [H, W, 3] uint8 RGB images keyed by name ('ROC Curves', 'Precision Recall Curves').
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    macro_metrics = compute_macro_curve_metrics(class_curves=class_curves)
    image_paths, images = {}, {}
    for image_name, image_settings in CURVE_IMAGES.items():
        # Draw the curves with their macro mean on the second line of the title, then save them as the TensorBoard image
        macro_name = image_settings["macro_name"]
        images[image_name] = draw_score_curves(
            class_curves=class_curves, class_names=class_names, image_name=image_name,
            title=f"{image_settings['title']}{title_suffix}\n{macro_name} "
                  f"{format_report_value(value=macro_metrics[macro_name])}", target_precision=target_precision)
        image_paths[f"{image_name} Image"] = output_folder / image_settings["file_name"]
        cv2.imwrite(filename=str(image_paths[f"{image_name} Image"]),
                    img=cv2.cvtColor(src=images[image_name], code=cv2.COLOR_RGB2BGR))
    return image_paths, images
