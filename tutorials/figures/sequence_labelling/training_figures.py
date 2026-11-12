"""
Training curves of the sequence labelling tutorials (the chapter on the model and the loss), made at every build by
tutorials/figures/sequence_labelling/figures.py from the TensorBoard logs of the real sequence labelling run: the total
loss and the frame accuracy of the training and validation batches, drawn with the references a run of this task is
read against, the cross entropy of a uniform guess over the classes, the cross entropy of the position prior of the
training split and the share of no_change frames. Two PNG files are written, the empty panels with the references (the
base image of the scene) and the whole chart (its overlay, whose curves the scene uncovers), with the boxes of the
panels and the values of the curves as facts. The accuracy ticks are the percentages of the reports of the repository
("20 %"), the facts keeping every value between 0 and 1.
"""

import matplotlib
import numpy

from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Any, Dict, Tuple

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, MONOSPACE_FONT_FAMILY,
                                                POINTS_PER_PIXEL, REVEAL_MARGIN, SERIES_COLOURS, SURFACE_COLOUR,
                                                TEXT_SECONDARY_COLOUR, compute_pixel_box,
                                                format_thousands, register_fonts, round_fact, save_figure,
                                                style_chart_panel)
from tutorials.figures.foundations.training_curves import read_scalars, smooth_values
from vision_training.evaluation.report_formatting import format_percentage_tick

# Name of the figures of the training curves
FIGURE_NAME = "training_curves"

# Smoothing weight of the training curves, the exponential moving average of TensorBoard
SMOOTHING_WEIGHT = 0.95

# Iterations between two labelled ticks of the horizontal axis
ITERATION_TICK_STEP = 500

# Iteration by which the narration says the training loss has dropped from the uniform guess, and the iteration after
# which the narration averages the training loss on its plateau, the position prior
EARLY_LOSS_ITERATION = 100
LATE_MEAN_START_ITERATION = 500

# Colours of the curves and of the reference lines
CURVE_COLOURS = {"Train": SERIES_COLOURS[0], "Validation": SERIES_COLOURS[1]}
REFERENCE_COLOUR = SERIES_COLOURS[3]


def draw_reference_line(axes: matplotlib.axes.Axes, value: float, label: str,
                        label_position: Tuple[float, float]) -> None:
    """
    Draws a dashed horizontal reference of a panel, its label written at a free place of the panel with an arrow to
    the line, so that the label never covers the curves drawn over the line.

    Args:
        axes (matplotlib.axes.Axes): The panel.
        value (float): The value of the reference.
        label (str): Its label.
        label_position (Tuple[float, float]): The iteration and value of the label, the arrow pointing straight down or
            up from it to the line.
    """
    axes.axhline(y=value, color=REFERENCE_COLOUR, linewidth=2 * POINTS_PER_PIXEL, linestyle=(0, (6, 5)), zorder=2)
    axes.annotate(text=label, xy=(label_position[0], value), xytext=label_position, color=REFERENCE_COLOUR,
                  fontsize=18 * POINTS_PER_PIXEL, ha="center", va="center", zorder=5,
                  arrowprops={"arrowstyle": "-|>", "color": REFERENCE_COLOUR, "linewidth": 2 * POINTS_PER_PIXEL,
                              "shrinkA": 6, "shrinkB": 2})


def style_curve_panel(axes: matplotlib.axes.Axes, title: str, y_label: str, y_limit: Tuple[float, float],
                      last_iteration: int) -> None:
    """
    Gives a panel of the training curves the anatomy of the charts of the videos, the iterations of the run on its
    horizontal axis.

    Args:
        axes (matplotlib.axes.Axes): The panel.
        title (str): The title of the panel.
        y_label (str): The name of the value axis.
        y_limit (Tuple[float, float]): The bottom and top of the value axis.
        last_iteration (int): The last iteration of the run.
    """
    style_chart_panel(axes=axes, title=title, x_label="Iteration", y_label=y_label)
    axes.set_xlim(left=0, right=last_iteration)
    axes.set_ylim(bottom=y_limit[0], top=y_limit[1])
    axes.set_xticks(ticks=list(range(0, last_iteration + 1, ITERATION_TICK_STEP)))
    axes.xaxis.set_major_formatter(formatter=FuncFormatter(func=format_thousands))


def draw_split_curves(axes: matplotlib.axes.Axes, curves: Dict[Tuple[str, str], Tuple[numpy.ndarray, numpy.ndarray]],
                      tag: str) -> None:
    """
    Draws the curves of one scalar of the training and validation batches on a panel: the raw training values faint
    under their smoothed curve, and the validation values as a smoothed curve.

    Args:
        axes (matplotlib.axes.Axes): The panel.
        curves (Dict[Tuple[str, str], Tuple[numpy.ndarray, numpy.ndarray]]): The iterations and values of every split
            and scalar.
        tag (str): The scalar, e.g. "Total Loss".
    """
    training_iterations, training_values = curves[("Train", tag)]
    axes.plot(training_iterations, training_values, color=CURVE_COLOURS["Train"], alpha=0.2,
              linewidth=1.2 * POINTS_PER_PIXEL)
    axes.plot(training_iterations, smooth_values(values=training_values, weight=SMOOTHING_WEIGHT),
              color=CURVE_COLOURS["Train"], linewidth=3 * POINTS_PER_PIXEL, zorder=3)
    validation_iterations, validation_values = curves[("Validation", tag)]
    axes.plot(validation_iterations, smooth_values(values=validation_values, weight=0.6),
              color=CURVE_COLOURS["Validation"], linewidth=3 * POINTS_PER_PIXEL, zorder=3)


def make_training_figures(output_folder: Path, run_folder: Path, run_label: str,
                          reference_facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Draws the training curves of the sequence labelling run with their references, as the base image and the overlay
    of their scene, and gives the boxes of the panels and the values of the curves.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the sequence labelling run.
        run_label (str): The run folder written in the header, from the experiments folder.
        reference_facts (Dict[str, Any]): The facts of the data figures: the uniform and prior cross entropies and
            the share of no_change frames.

    Returns:
        Dict[str, Any]: The facts of the figure, named training.*.
    """
    curves = {(split, tag): read_scalars(log_folder=run_folder / "Tensorboard" / split, tag=tag)
              for split in ["Train", "Validation"] for tag in ["Total Loss", "Accuracy"]}
    last_iteration = int(curves[("Train", "Total Loss")][0][-1])

    # Figure of the size of the viewport, a loss panel and an accuracy panel, with the references in the base image,
    # apart enough for the percentage ticks of the accuracy
    register_fonts()
    figure, (loss_axes, accuracy_axes) = pyplot.subplots(nrows=1, ncols=2, figsize=(CHART_WIDTH / DOTS_PER_INCH,
                                                                                 CHART_HEIGHT / DOTS_PER_INCH),
                                                         dpi=DOTS_PER_INCH)
    figure.patch.set_facecolor(SURFACE_COLOUR)
    figure.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.12, wspace=0.2)
    figure.text(x=0.012, y=0.955, s=run_label, color=TEXT_SECONDARY_COLOUR, fontsize=16 * POINTS_PER_PIXEL,
                va="center", family=MONOSPACE_FONT_FAMILY)
    figure.text(x=0.012, y=0.905, s=f"Blue: training batches (faint raw, solid smoothed {SMOOTHING_WEIGHT}) · "
                                    f"green: validation batches · dashed: references", color=TEXT_SECONDARY_COLOUR,
                fontsize=18 * POINTS_PER_PIXEL, va="center")
    style_curve_panel(axes=loss_axes, title="Total loss (cross entropy of the labelled frames)", y_label="Loss",
                      y_limit=(2.4, 4.6), last_iteration=last_iteration)
    style_curve_panel(axes=accuracy_axes, title="Accuracy of the labelled frames", y_label="Accuracy",
                      y_limit=(0.0, 0.7), last_iteration=last_iteration)
    # Accuracy ticks written as the percentages of the reports of the repository, the share of no_change frames of
    # the reference line keeping its value between 0 and 1 like every share of the data
    accuracy_axes.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))
    draw_reference_line(axes=loss_axes, value=reference_facts["dataset.uniform_cross_entropy"],
                        label=f"uniform over {reference_facts['dataset.class_count']} classes: "
                              f"ln {reference_facts['dataset.class_count']} = "
                              f"{reference_facts['dataset.uniform_cross_entropy']:.3f}",
                        label_position=(last_iteration * 0.62, 4.45))
    draw_reference_line(axes=loss_axes, value=reference_facts["dataset.prior_cross_entropy"],
                        label=f"position prior of the training split: "
                              f"{reference_facts['dataset.prior_cross_entropy']:.3f}",
                        label_position=(last_iteration * 0.62, 3.6))
    draw_reference_line(axes=accuracy_axes, value=reference_facts["dataset.no_change_share"],
                        label=f"share of no_change training frames: {reference_facts['dataset.no_change_share']:.3f}",
                        label_position=(last_iteration * 0.6, 0.2))
    save_figure(figure=figure, output_path=output_folder / "training" / f"{FIGURE_NAME}_axes.png")

    # Curves of both splits, written as the overlay
    draw_split_curves(axes=loss_axes, curves=curves, tag="Total Loss")
    draw_split_curves(axes=accuracy_axes, curves=curves, tag="Accuracy")
    save_figure(figure=figure, output_path=output_folder / "training" / f"{FIGURE_NAME}.png")
    facts = {}
    for panel_name, panel_axes in [("loss", loss_axes), ("accuracy", accuracy_axes)]:
        facts[f"training.{panel_name}_panel_box"] = compute_pixel_box(figure=figure, axes=panel_axes,
                                                                      with_labels=True, margin=0)
        facts[f"training.{panel_name}_reveal_region"] = compute_pixel_box(figure=figure, axes=panel_axes,
                                                                          with_labels=False, margin=REVEAL_MARGIN)
    pyplot.close(fig=figure)

    # Values of the curves: first, last and smoothed last, and the smoothed training loss at a few iterations
    facts["training.last_iteration"] = last_iteration
    for (split, tag), (iterations, values) in curves.items():
        curve_name = f"training.{split.lower()}_{tag.lower().replace(' ', '_')}"
        smoothed = smooth_values(values=values, weight=SMOOTHING_WEIGHT if split == "Train" else 0.6)
        facts[f"{curve_name}.count"] = len(values)
        facts[f"{curve_name}.first"] = round_fact(value=values[0])
        facts[f"{curve_name}.last"] = round_fact(value=values[-1])
        facts[f"{curve_name}.smoothed_last"] = round_fact(value=smoothed[-1])
        facts[f"{curve_name}.minimum"] = round_fact(value=values.min())
    training_iterations, training_losses = curves[("Train", "Total Loss")]
    smoothed_losses = smooth_values(values=training_losses, weight=SMOOTHING_WEIGHT)
    for iteration in [100, 200, 500, 1000, 1500, 2000]:
        facts[f"training.train_total_loss.smoothed_at_{iteration}"] = round_fact(
            value=smoothed_losses[numpy.searchsorted(training_iterations, iteration)])

    # Iterations the narration reads the loss at: the early drop, and the mean of the loss after its plateau starts
    facts["training.early_iteration"] = EARLY_LOSS_ITERATION
    facts["training.train_total_loss.smoothed_at_early"] = round_fact(
        value=smoothed_losses[numpy.searchsorted(training_iterations, EARLY_LOSS_ITERATION)])
    facts["training.late_mean_start_iteration"] = LATE_MEAN_START_ITERATION
    facts["training.train_total_loss.late_mean"] = round_fact(
        value=training_losses[training_iterations > LATE_MEAN_START_ITERATION].mean())
    return facts
