"""
Training curves of the frame selection tutorials (the chapter on the losses and the decoding), made at every build by
tutorials/figures/frame_selection/figures.py from the TensorBoard logs of the real frame selection run: the per-frame
classification loss and the selection loss of the training batches on a logarithmic axis, with the references of equal
logits (ln K for the per-frame loss over the K categories, the mean ln T of the training sequences for the selection
loss), and the frame accuracy of the training and validation batches with the share of unselected frames. Two PNG files
are written, the empty panels with the references (the base image of the scene) and the whole chart (its overlay), with
the boxes of the panels, the values of the curves and the median throughput of the training (from the wall times of
the logs) as facts. The accuracy and the share of unselected frames are written as the percentages of the reports of
the repository ("94.03 %", ticks "60 %"), the facts keeping them between 0 and 1.
"""

import math

import matplotlib
import numpy

from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Any, Dict, Tuple

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, MONOSPACE_FONT_FAMILY,
                                                POINTS_PER_PIXEL, REVEAL_MARGIN, SERIES_COLOURS, SURFACE_COLOUR,
                                                TEXT_PRIMARY_COLOUR, TEXT_SECONDARY_COLOUR, compute_pixel_box,
                                                format_thousands, register_fonts, round_fact, save_figure,
                                                style_chart_panel)
from tutorials.figures.foundations.training_curves import read_scalars, smooth_values
from tutorials.figures.sequence_labelling.sequence_utilities import read_run_sections
from vision_training.evaluation.report_formatting import format_percentage, format_percentage_tick

# Name of the figures of the training curves
FIGURE_NAME = "training_curves"

# Smoothing weight of the training curves, the exponential moving average of TensorBoard
SMOOTHING_WEIGHT = 0.9

# Iterations between two labelled ticks of the horizontal axis
ITERATION_TICK_STEP = 500

# Colours of the curves and of the references
CURVE_COLOURS = {"Classification Loss": SERIES_COLOURS[0], "Selection Loss": SERIES_COLOURS[2],
                 "Train": SERIES_COLOURS[0], "Validation": SERIES_COLOURS[1]}
REFERENCE_COLOUR = SERIES_COLOURS[3]

# Iterations at which the facts give the smoothed training curves
FACT_ITERATIONS = [100, 500, 1000, 2000]

# Lowest bottom of the accuracy axis, lowered to the tenth under the lowest accuracy of the batches when a curve starts
# under it (the first training batches of an untrained model)
ACCURACY_AXIS_BOTTOM = 0.5

# Scalar of the training batches, written at every iteration, whose wall times give the throughput of the run
THROUGHPUT_TAG = "Total Loss"


def compute_training_throughput(log_folder: Path, batch_size: int) -> float:
    """
    Measures the throughput of the training of the run from the wall times TensorBoard recorded with its scalar of
    every iteration: the sequences of a batch over the time between two consecutive iterations, the median over the
    run, so that the evaluations, the validation batches and the start of the run do not weigh on it. The narration
    of the commands quotes it next to the iterations of the run.

    Args:
        log_folder (Path): The Train folder of the TensorBoard logs of the run.
        batch_size (int): The sequences of a training batch of the run.

    Returns:
        float: The median number of training sequences per second.

    Raises:
        ValueError: If the scalar is not written at consecutive iterations, the time between two events then not
            being the time of one iteration.
    """
    accumulator = EventAccumulator(path=str(log_folder), size_guidance={"scalars": 0})
    accumulator.Reload()
    events = accumulator.Scalars(THROUGHPUT_TAG)
    iterations = numpy.array([event.step for event in events])
    wall_times = numpy.array([event.wall_time for event in events])

    # Time of every iteration, from the scalars of two consecutive iterations
    if not numpy.all(numpy.diff(iterations) == 1):
        raise ValueError(f"The '{THROUGHPUT_TAG}' scalar of {log_folder} is not written at every iteration, so the "
                         f"time between two of its events is not the time of one iteration.")
    return float(numpy.median(batch_size / numpy.diff(wall_times)))


def style_curve_panel(axes: matplotlib.axes.Axes, title: str, y_label: str, last_iteration: int) -> None:
    """
    Gives a panel of the training curves the anatomy of the charts of the videos, the iterations of the run on its
    horizontal axis.

    Args:
        axes (matplotlib.axes.Axes): The panel.
        title (str): The title of the panel.
        y_label (str): The name of the value axis.
        last_iteration (int): The last iteration of the run.
    """
    style_chart_panel(axes=axes, title=title, x_label="Iteration", y_label=y_label)
    axes.set_xlim(left=0, right=last_iteration)
    axes.set_xticks(ticks=list(range(0, last_iteration + 1, ITERATION_TICK_STEP)))
    axes.xaxis.set_major_formatter(formatter=FuncFormatter(func=format_thousands))


def draw_reference_line(axes: matplotlib.axes.Axes, value: float, label: str, last_iteration: int,
                        vertical_alignment: str) -> None:
    """
    Draws a dashed horizontal reference of a panel with its label at its right end.

    Args:
        axes (matplotlib.axes.Axes): The panel.
        value (float): The value of the reference.
        label (str): Its label.
        last_iteration (int): The right end of the horizontal axis.
        vertical_alignment (str): "bottom" for a label above the line, "top" for a label under it.
    """
    axes.axhline(y=value, color=REFERENCE_COLOUR, linewidth=2 * POINTS_PER_PIXEL, linestyle=(0, (6, 5)), zorder=2)
    axes.text(x=last_iteration, y=value, s=label, color=REFERENCE_COLOUR, fontsize=18 * POINTS_PER_PIXEL, ha="right",
              va=vertical_alignment, zorder=5)


def draw_curve(axes: matplotlib.axes.Axes, curve: Tuple[numpy.ndarray, numpy.ndarray], colour: str, label: str) -> None:
    """
    Draws one training curve: its raw values faint under its smoothed curve, with the name of the curve.

    Args:
        axes (matplotlib.axes.Axes): The panel.
        curve (Tuple[numpy.ndarray, numpy.ndarray]): The iterations and values.
        colour (str): The colour of the curve.
        label (str): The name of the curve in the legend.
    """
    iterations, values = curve
    axes.plot(iterations, values, color=colour, alpha=0.2, linewidth=1.2 * POINTS_PER_PIXEL)
    axes.plot(iterations, smooth_values(values=values, weight=SMOOTHING_WEIGHT), color=colour,
              linewidth=3 * POINTS_PER_PIXEL, zorder=3, label=label)


def make_training_figures(output_folder: Path, run_folder: Path, run_label: str,
                          reference_facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Draws the training curves of the frame selection run with their references, as the base image and the overlay of
    their scene, and gives the boxes of the panels and the values of the curves.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the frame selection run.
        run_label (str): The run folder written in the header, from the experiments folder.
        reference_facts (Dict[str, Any]): The facts of the data figures: the mean ln T, the number of categories and
            its natural log, and the share of unselected frames.

    Returns:
        Dict[str, Any]: The facts of the figure, named training.*.
    """
    curves = {(split, tag): read_scalars(log_folder=run_folder / "Tensorboard" / split, tag=tag)
              for split in ["Train", "Validation"]
              for tag in ["Total Loss", "Classification Loss", "Selection Loss", "Accuracy"]}
    last_iteration = int(curves[("Train", "Total Loss")][0][-1])

    # Figure of the size of the viewport, the losses and the accuracy, with the references in the base image
    register_fonts()
    figure, (loss_axes, accuracy_axes) = pyplot.subplots(nrows=1, ncols=2, figsize=(CHART_WIDTH / DOTS_PER_INCH,
                                                                                 CHART_HEIGHT / DOTS_PER_INCH),
                                                         dpi=DOTS_PER_INCH, gridspec_kw={"width_ratios": [1.3, 1.0]})
    figure.patch.set_facecolor(SURFACE_COLOUR)
    figure.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.12, wspace=0.18)
    figure.text(x=0.012, y=0.955, s=run_label, color=TEXT_SECONDARY_COLOUR, fontsize=16 * POINTS_PER_PIXEL,
                va="center", family=MONOSPACE_FONT_FAMILY)
    figure.text(x=0.012, y=0.905, s=f"Training batches, faint raw and solid smoothed ({SMOOTHING_WEIGHT}) · "
                                    f"dashed: the losses of equal logits and the share of unselected frames",
                color=TEXT_SECONDARY_COLOUR, fontsize=18 * POINTS_PER_PIXEL, va="center")
    style_curve_panel(axes=loss_axes, title="Losses of the training batches (logarithmic)", y_label="Loss",
                      last_iteration=last_iteration)
    loss_axes.set_yscale(value="log")
    loss_axes.set_ylim(bottom=1e-4, top=6.0)
    style_curve_panel(axes=accuracy_axes, title="Accuracy of the frames", y_label="Accuracy",
                      last_iteration=last_iteration)
    # Accuracy axis from ACCURACY_AXIS_BOTTOM, or lower so that the first batches of a curve starting under it show
    lowest_accuracy = min(float(numpy.min(curves[(split, "Accuracy")][1])) for split in ["Train", "Validation"])
    accuracy_axes.set_ylim(bottom=min(ACCURACY_AXIS_BOTTOM, math.floor(lowest_accuracy * 10) / 10), top=1.03)

    # Accuracy ticks written as the percentages of the reports of the repository, the losses keeping their decimals
    accuracy_axes.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))
    draw_reference_line(axes=loss_axes, value=reference_facts["dataset.mean_log_length"],
                        label=f"selection, equal logits: mean ln T = {reference_facts['dataset.mean_log_length']:.3f}",
                        last_iteration=last_iteration, vertical_alignment="bottom")
    draw_reference_line(axes=loss_axes, value=reference_facts["dataset.log_class_count"],
                        label=f"per frame, equal logits: ln {reference_facts['dataset.category_count']} = "
                              f"{reference_facts['dataset.log_class_count']:.3f}",
                        last_iteration=last_iteration, vertical_alignment="top")
    draw_reference_line(axes=accuracy_axes, value=reference_facts["dataset.unselected_share"],
                        label=f"share of unselected frames: "
                              f"{format_percentage(value=reference_facts['dataset.unselected_share'])}",
                        last_iteration=last_iteration, vertical_alignment="top")
    save_figure(figure=figure, output_path=output_folder / "training" / f"{FIGURE_NAME}_axes.png")

    # Curves of the two losses and of the accuracy of both splits, written as the overlay
    draw_curve(axes=loss_axes, curve=curves[("Train", "Classification Loss")],
               colour=CURVE_COLOURS["Classification Loss"], label="per-frame classification loss")
    draw_curve(axes=loss_axes, curve=curves[("Train", "Selection Loss")], colour=CURVE_COLOURS["Selection Loss"],
               label="selection loss")
    loss_axes.legend(loc="lower left", frameon=False, labelcolor=TEXT_PRIMARY_COLOUR, fontsize=18 * POINTS_PER_PIXEL)
    draw_curve(axes=accuracy_axes, curve=curves[("Train", "Accuracy")], colour=CURVE_COLOURS["Train"],
               label="training batches")
    validation_iterations, validation_values = curves[("Validation", "Accuracy")]
    accuracy_axes.plot(validation_iterations, validation_values, color=CURVE_COLOURS["Validation"],
                       linewidth=3 * POINTS_PER_PIXEL, zorder=3, label="validation batches")
    accuracy_axes.legend(loc="lower right", frameon=False, labelcolor=TEXT_PRIMARY_COLOUR,
                         fontsize=18 * POINTS_PER_PIXEL)
    save_figure(figure=figure, output_path=output_folder / "training" / f"{FIGURE_NAME}.png")

    # Box of every panel with its labels for the highlights, and without them for the reveal region of the overlay
    facts = {}
    for panel_name, panel_axes in [("loss", loss_axes), ("accuracy", accuracy_axes)]:
        facts[f"training.{panel_name}_panel_box"] = compute_pixel_box(figure=figure, axes=panel_axes,
                                                                      with_labels=True, margin=0)
        facts[f"training.{panel_name}_reveal_region"] = compute_pixel_box(figure=figure, axes=panel_axes,
                                                                          with_labels=False, margin=REVEAL_MARGIN)
    pyplot.close(fig=figure)

    # First and last values of every curve, the smoothed training curves at a few iterations, and the throughput
    experiment_configuration, _ = read_run_sections(run_folder=run_folder)
    facts["training.last_iteration"] = last_iteration
    facts["training.fact_iterations"] = FACT_ITERATIONS
    facts["training.median_sequences_per_second"] = round_fact(value=compute_training_throughput(
        log_folder=run_folder / "Tensorboard" / "Train", batch_size=int(experiment_configuration["batch_size"])))
    for (split, tag), (iterations, values) in curves.items():
        curve_name = f"training.{split.lower()}_{tag.lower().replace(' ', '_')}"
        facts[f"{curve_name}.first"] = round_fact(value=values[0])
        facts[f"{curve_name}.last"] = round_fact(value=values[-1])
        if split == "Train":
            smoothed = smooth_values(values=values, weight=SMOOTHING_WEIGHT)
            for iteration in FACT_ITERATIONS:
                facts[f"{curve_name}.smoothed_at_{iteration}"] = round_fact(
                    value=smoothed[numpy.searchsorted(iterations, iteration)])
    return facts
