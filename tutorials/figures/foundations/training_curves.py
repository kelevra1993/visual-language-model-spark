"""
Training curves of the classification run for the foundations tutorials (the training loop and TensorBoard chapters),
made at every build by tutorials/figures/foundations/figures.py: the training and validation total loss and accuracy of
every iteration, read from the TensorBoard logs of the run, with the test accuracy of every evaluated checkpoint, on
two panels, the raw values drawn faintly under their curve smoothed like TensorBoard, the loss axis reaching above the
highest loss of the run (its first losses grow with the number of classes). Two PNG files are written: the
empty axes with the legend, the base image of the scene, and the whole chart, its overlay, whose curves the scene
uncovers while they are explained; the boxes of the panels (highlights and reveal regions) and the values of the
curves are the facts of the storyboards. The accuracies are written as the percentages of the reports of the
repository ("94.03 %", ticks "20 %"), the facts keeping them between 0 and 1.
"""

import math

import matplotlib
import numpy

from matplotlib import pyplot
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Any, Dict, List, Tuple

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, MONOSPACE_FONT_FAMILY,
                                                POINTS_PER_PIXEL, REVEAL_MARGIN, SERIES_COLOURS, SURFACE_COLOUR,
                                                TEXT_PRIMARY_COLOUR, TEXT_SECONDARY_COLOUR, compute_pixel_box,
                                                format_thousands, register_fonts, round_fact, save_figure,
                                                style_chart_panel)
from vision_training.evaluation.report_formatting import format_percentage, format_percentage_tick

# Names of the figures, kept from the first tutorials
FIGURE_NAME = "training_curves_readme_classification_run"

# Colour of every series, in the order of the categorical palette of the videos
CURVE_COLOURS = {"Train": SERIES_COLOURS[0], "Validation": SERIES_COLOURS[1], "Test": SERIES_COLOURS[2]}

# Smoothing weight of the curves, the exponential moving average of TensorBoard
SMOOTHING_WEIGHT = 0.95

# Iterations between two labelled ticks of the horizontal axis, which runs from 0 to the last iteration of the run
ITERATION_TICK_STEP = 250

# Step the top of the loss axis is rounded up to, above the highest loss of the training and validation batches, so
# that the first losses of a run of any number of classes (about the logarithm of the class count) stay on the panel
LOSS_AXIS_STEP = 0.5


def read_scalars(log_folder: Path, tag: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Reads every value of one scalar of a TensorBoard log folder of the run, the curve TensorBoard draws.

    Args:
        log_folder (Path): The Train, Validation or Test folder of the run.
        tag (str): The scalar, e.g. "Total Loss".

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The iterations and the values.

    Raises:
        FileNotFoundError: If the log folder is missing.
    """
    if not log_folder.is_dir():
        raise FileNotFoundError(f"No TensorBoard log folder {log_folder} in the classification run.")
    accumulator = EventAccumulator(path=str(log_folder), size_guidance={"scalars": 0})
    accumulator.Reload()
    events = accumulator.Scalars(tag)
    return numpy.array([event.step for event in events]), numpy.array([event.value for event in events])


def smooth_values(values: numpy.ndarray, weight: float) -> numpy.ndarray:
    """
    Smooths a curve the way TensorBoard does: an exponential moving average of the values, divided by the sum of its
    weights so that the first values are not pulled towards zero.

    Args:
        values (numpy.ndarray): The raw values.
        weight (float): The smoothing weight within [0, 1).

    Returns:
        numpy.ndarray: The smoothed values.
    """
    smoothed_values = numpy.empty_like(values, dtype=float)
    running_average = 0.0
    for value_index, value in enumerate(values):
        running_average = running_average * weight + (1.0 - weight) * value
        smoothed_values[value_index] = running_average / (1.0 - weight ** (value_index + 1))
    return smoothed_values


def compute_loss_axis_top(loss_curves: List[numpy.ndarray]) -> float:
    """
    Gives the top of the value axis of the loss panel of the training curves from the losses of the run, so that every
    raw loss of the training and validation batches, the first ones quoted by the narration included, stays on the
    panel whatever the number of classes of the run: the highest loss rounded up to the next multiple of LOSS_AXIS_STEP.

    Args:
        loss_curves (List[numpy.ndarray]): The raw total losses of every split drawn on the panel.

    Returns:
        float: The top of the loss axis.

    Raises:
        ValueError: If a curve is empty or holds a loss that is not finite.
    """
    if any(len(loss_values) == 0 for loss_values in loss_curves):
        raise ValueError("A total loss curve of the classification run has no value.")
    highest_loss = max(float(loss_values.max()) for loss_values in loss_curves)
    if not math.isfinite(highest_loss):
        raise ValueError(f"The total loss of the classification run reaches {highest_loss}, expected finite values.")
    return (math.floor(highest_loss / LOSS_AXIS_STEP) + 1) * LOSS_AXIS_STEP


def style_panel(axes: matplotlib.axes.Axes, title: str, last_iteration: int, y_limit: Tuple[float, float],
                y_label: str) -> None:
    """
    Gives a panel of the training curves the anatomy of the charts of the videos, with the iterations of the run on its
    horizontal axis, a tick every ITERATION_TICK_STEP iterations.

    Args:
        axes (matplotlib.axes.Axes): The panel.
        title (str): The title of the panel.
        last_iteration (int): The last iteration of the run, the right end of the axis.
        y_limit (Tuple[float, float]): The bottom and top of the value axis.
        y_label (str): The name of the value axis.
    """
    style_chart_panel(axes=axes, title=title, x_label="Iteration", y_label=y_label)
    axes.set_xlim(left=0, right=last_iteration)
    axes.set_ylim(bottom=y_limit[0], top=y_limit[1])
    axes.set_xticks(ticks=list(range(0, last_iteration + 1, ITERATION_TICK_STEP)))
    axes.xaxis.set_major_formatter(formatter=FuncFormatter(func=format_thousands))


def draw_curve(axes: matplotlib.axes.Axes, iterations: numpy.ndarray, values: numpy.ndarray, colour: str) -> None:
    """
    Draws one curve of a panel: the raw value of every iteration as a faint line, and its smoothed curve on top.

    Args:
        axes (matplotlib.axes.Axes): The panel.
        iterations (numpy.ndarray): The iterations.
        values (numpy.ndarray): The raw values.
        colour (str): The colour of the series.
    """
    axes.plot(iterations, values, color=colour, alpha=0.22, linewidth=1.2 * POINTS_PER_PIXEL)
    axes.plot(iterations, smooth_values(values=values, weight=SMOOTHING_WEIGHT), color=colour,
              linewidth=3 * POINTS_PER_PIXEL, solid_capstyle="round", solid_joinstyle="round")


def make_training_curves_figure(output_folder: Path, run_folder: Path, run_label: str) -> Dict[str, Any]:
    """
    Draws the training curves of the classification run, as the base image and the overlay of its scene, and gives
    the boxes of its panels and the values of its curves: for every curve its number of iterations, its first, last
    and smoothed last values, and the test accuracy of every evaluation.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the classification run of tutorial_sources.yaml.
        run_label (str): The run folder written in the header, from the experiments folder of the repository.

    Returns:
        Dict[str, Any]: The facts of the figure, named training_curves.*.
    """
    # Curves of the run: total loss and accuracy of the training and validation batches, test accuracy
    curves = {}
    for split in ["Train", "Validation"]:
        for tag in ["Total Loss", "Accuracy"]:
            curves[(split, tag)] = read_scalars(log_folder=run_folder / "Tensorboard" / split, tag=tag)
    test_iterations, test_accuracies = read_scalars(log_folder=run_folder / "Tensorboard" / "Test", tag="Accuracy")
    last_iteration = int(curves[("Train", "Total Loss")][0][-1])

    # Figure of the size of the viewport of a scene, a loss panel and an accuracy panel, apart enough for the
    # percentage ticks of the accuracy
    register_fonts()
    figure, (loss_axes, accuracy_axes) = pyplot.subplots(nrows=1, ncols=2, figsize=(CHART_WIDTH / DOTS_PER_INCH,
                                                                                 CHART_HEIGHT / DOTS_PER_INCH),
                                                         dpi=DOTS_PER_INCH)
    figure.patch.set_facecolor(SURFACE_COLOUR)
    figure.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.12, wspace=0.2)

    # Header: the run and the smoothing, then the legend of the three series, saved with the empty panels as the base
    figure.text(x=0.012, y=0.955, s=run_label, color=TEXT_SECONDARY_COLOUR, fontsize=16 * POINTS_PER_PIXEL,
                va="center", family=MONOSPACE_FONT_FAMILY)
    figure.text(x=0.012, y=0.905, s=f"Faint: every iteration · solid: smoothed like TensorBoard ({SMOOTHING_WEIGHT})",
                color=TEXT_SECONDARY_COLOUR, fontsize=18 * POINTS_PER_PIXEL, va="center")
    legend_handles = [Line2D(xdata=[0], ydata=[0], color=CURVE_COLOURS["Train"], linewidth=3 * POINTS_PER_PIXEL),
                      Line2D(xdata=[0], ydata=[0], color=CURVE_COLOURS["Validation"], linewidth=3 * POINTS_PER_PIXEL),
                      Line2D(xdata=[0], ydata=[0], color=CURVE_COLOURS["Test"], marker="o", linestyle="none",
                             markersize=10 * POINTS_PER_PIXEL)]
    figure.legend(handles=legend_handles, labels=["Training batches", "Validation batches", "Test split"],
                  loc="center right", bbox_to_anchor=(0.985, 0.905), ncols=3, frameon=False,
                  labelcolor=TEXT_PRIMARY_COLOUR, fontsize=18 * POINTS_PER_PIXEL, handlelength=1.6)
    # Loss axis from zero to above the highest loss of the batches, taken from the run
    loss_curves = [curves[(split, "Total Loss")][1] for split in ["Train", "Validation"]]
    loss_axis_top = compute_loss_axis_top(loss_curves=loss_curves)
    style_panel(axes=loss_axes, title="Total loss", last_iteration=last_iteration, y_limit=(0.0, loss_axis_top),
                y_label="Loss")
    style_panel(axes=accuracy_axes, title="Accuracy", last_iteration=last_iteration, y_limit=(0.0, 1.0),
                y_label="Accuracy")
    # Accuracy ticks written as the percentages of the reports of the repository
    accuracy_axes.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))
    save_figure(figure=figure, output_path=output_folder / f"{FIGURE_NAME}_axes.png")

    # Curves of the training and validation batches on both panels, the test accuracy of every evaluation as points
    for split in ["Train", "Validation"]:
        draw_curve(axes=loss_axes, iterations=curves[(split, "Total Loss")][0], values=curves[(split, "Total Loss")][1],
                   colour=CURVE_COLOURS[split])
        draw_curve(axes=accuracy_axes, iterations=curves[(split, "Accuracy")][0],
                   values=curves[(split, "Accuracy")][1], colour=CURVE_COLOURS[split])
    accuracy_axes.plot(test_iterations, test_accuracies, color=CURVE_COLOURS["Test"], marker="o", linestyle="none",
                       markersize=12 * POINTS_PER_PIXEL, markeredgecolor=SURFACE_COLOUR,
                       markeredgewidth=2 * POINTS_PER_PIXEL, zorder=4, clip_on=False)
    # Every test accuracy labelled with its rounded fact, the value the narration quotes, so that both round alike
    for test_iteration, test_accuracy in zip(test_iterations, test_accuracies):
        accuracy_axes.annotate(text=format_percentage(value=round_fact(value=test_accuracy)),
                               xy=(test_iteration, test_accuracy),
                               xytext=(-12 * POINTS_PER_PIXEL, -22 * POINTS_PER_PIXEL), textcoords="offset points",
                               ha="right", va="center", color=TEXT_PRIMARY_COLOUR, fontsize=17 * POINTS_PER_PIXEL)

    # Whole chart written as the overlay of the scene, then the boxes of its highlights and reveals
    save_figure(figure=figure, output_path=output_folder / f"{FIGURE_NAME}.png")
    facts = {}
    for panel_name, panel_axes in [("loss", loss_axes), ("accuracy", accuracy_axes)]:
        facts[f"training_curves.{panel_name}_panel_box"] = compute_pixel_box(figure=figure, axes=panel_axes,
                                                                             with_labels=True, margin=0)
        facts[f"training_curves.{panel_name}_reveal_region"] = compute_pixel_box(figure=figure, axes=panel_axes,
                                                                                 with_labels=False,
                                                                                 margin=REVEAL_MARGIN)
    pyplot.close(fig=figure)

    # Values of every curve and the test accuracy of every evaluation
    facts["training_curves.smoothing_weight"] = SMOOTHING_WEIGHT
    facts["training_curves.loss_axis_top"] = loss_axis_top
    for (split, tag), (iterations, values) in curves.items():
        curve_name = f"training_curves.{split.lower()}_{tag.lower().replace(' ', '_')}"
        facts[f"{curve_name}.iteration_count"] = len(values)
        facts[f"{curve_name}.first"] = round_fact(value=values[0])
        facts[f"{curve_name}.last"] = round_fact(value=values[-1])
        facts[f"{curve_name}.smoothed_last"] = round_fact(value=smooth_values(values=values,
                                                                               weight=SMOOTHING_WEIGHT)[-1])
    for test_iteration, test_accuracy in zip(test_iterations, test_accuracies):
        facts[f"training_curves.test_accuracy.iteration_{int(test_iteration)}"] = round_fact(value=test_accuracy)
    return facts
