"""
Figures and facts of the training scenes of the semantic segmentation series of the tutorial videos, made by
tutorials/figures/semantic_segmentation/figures.py from the synthetic segmentation run: its training and validation
curves read from its TensorBoard logs (the losses on one panel, the batch metrics on the other, an empty-axes base image
and its overlay whose curves the scene uncovers), the values the narration quotes (the first and last values, the
uniform cross entropy ln K), the share of the test pixels of every class read from its class_metrics.csv, and the
throughput of its training log. The batch metrics and the pixel shares drawn are written as the percentages of the
reports of the repository ("94.03 %", ticks "20 %"), the facts keeping them between 0 and 1.
"""

import math
import re

import numpy

from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Any, Dict

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, POINTS_PER_PIXEL,
                                                REVEAL_MARGIN, SERIES_COLOURS, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR,
                                                compute_pixel_box, format_thousands, register_fonts, round_fact,
                                                save_figure, style_chart_panel)
from tutorials.figures.keypoint_detection.scalar_logs import (SMOOTHING_WEIGHT, draw_smoothed_curve, read_scalars,
                                                             smooth_values)
from tutorials.figures.semantic_segmentation.segmentation_sources import CHECKPOINT_ITERATION, EVALUATION_FOLDER_NAME
from vision_training.evaluation.report_formatting import format_percentage, format_percentage_tick

# Curves of both panels, logged by the segmentation trainer in Tensorboard/Train and Validation, with their colours
LOSS_CURVES = {"Total Loss": SERIES_COLOURS[0], "Cross Entropy Loss": SERIES_COLOURS[1], "Dice Loss": SERIES_COLOURS[2]}
METRIC_CURVES = {"Pixel Accuracy": SERIES_COLOURS[3], "Mean IoU": SERIES_COLOURS[4]}

# Iterations between two labelled ticks of the curves
ITERATION_TICK_STEP = 500

# Pixel accuracy of the smoothed training batches whose first iteration the narration quotes, and the share of its
# fall from its first to its last smoothed value that the smoothed cross entropy has made at the iteration quoted
PIXEL_ACCURACY_THRESHOLD = 0.9
LOSS_FALL_SHARE = 0.9


def style_curve_panel(axes: Any, title: str, last_iteration: int, y_label: str) -> None:
    """
    Gives a panel of the training curves the anatomy of the charts of the videos, with the iterations of the run on
    its horizontal axis.

    Args:
        axes (Any): The matplotlib panel.
        title (str): The title of the panel.
        last_iteration (int): The last iteration of the run.
        y_label (str): The name of the value axis.
    """
    style_chart_panel(axes=axes, title=title, x_label="Iteration", y_label=y_label)
    axes.set_xlim(left=0, right=last_iteration)
    axes.set_xticks(ticks=list(range(0, last_iteration + 1, ITERATION_TICK_STEP)))
    axes.xaxis.set_major_formatter(formatter=FuncFormatter(func=format_thousands))


def fact_name(tag: str) -> str:
    """
    Gives the name of the facts of a scalar, its tag in lower case with underscores.

    Args:
        tag (str): The scalar, e.g. "Cross Entropy Loss".

    Returns:
        str: Its fact name, e.g. "cross_entropy_loss".
    """
    return re.sub(pattern=r"[^a-z0-9]+", repl="_", string=tag.lower()).strip("_")


def draw_training_curves(output_folder: Path, run_folder: Path, with_curves: bool) -> Dict[str, Any]:
    """
    Draws the training curves of the segmentation run: the total, cross entropy and Dice losses, and the pixel accuracy
    and mean IoU of the batches, the training values smoothed like TensorBoard and the validation values as dots,
    read from Tensorboard/Train and Validation; the empty axes for the base image of the scene, or the whole chart.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the segmentation run.
        with_curves (bool): Whether the curves are drawn (the overlay) or only the axes and legend (the base image).

    Returns:
        Dict[str, Any]: The facts of the curves and the boxes of the panels.
    """
    register_fonts()
    figure = pyplot.figure(figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH,
                           facecolor=SURFACE_COLOUR)
    loss_axes = figure.add_axes(rect=(0.07, 0.14, 0.40, 0.74))
    metric_axes = figure.add_axes(rect=(0.57, 0.14, 0.40, 0.74))
    last_iteration = int(read_scalars(log_folder=run_folder / "Tensorboard" / "Train", tag="Total Loss")[0][-1])
    style_curve_panel(axes=loss_axes, title="Losses", last_iteration=last_iteration, y_label="Loss")
    style_curve_panel(axes=metric_axes, title="Batch metrics", last_iteration=last_iteration, y_label="Value")
    loss_axes.set_ylim(bottom=0.0, top=3.6)
    metric_axes.set_ylim(bottom=0.0, top=1.0)

    # Batch metric ticks written as the percentages of the reports of the repository, the losses keeping their decimals
    metric_axes.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))

    # Every curve: the training values smoothed, the validation values as dots of the same colour
    facts = {"curves_last_iteration": last_iteration}
    for axes, panel_curves in [(loss_axes, LOSS_CURVES), (metric_axes, METRIC_CURVES)]:
        for tag, colour in panel_curves.items():
            name = fact_name(tag=tag)
            train_iterations, train_values = read_scalars(log_folder=run_folder / "Tensorboard" / "Train", tag=tag)
            validation_iterations, validation_values = read_scalars(log_folder=run_folder / "Tensorboard" /
                                                                    "Validation", tag=tag)
            if with_curves:
                draw_smoothed_curve(axes=axes, iterations=train_iterations, values=train_values, colour=colour,
                                    label=f"{tag.replace('Loss', '').strip()}, train", smoothing=True)
                axes.scatter(validation_iterations, validation_values, s=(5 * POINTS_PER_PIXEL) ** 2, color=colour,
                             edgecolors=TEXT_PRIMARY_COLOUR, linewidths=0.6 * POINTS_PER_PIXEL, zorder=3)
            else:
                # The base image only holds the legend entry of the curve, its line uncovered by the reveal
                axes.plot([], [], color=colour, linewidth=3 * POINTS_PER_PIXEL,
                          label=f"{tag.replace('Loss', '').strip()}, train")
            smoothed = smooth_values(values=train_values, weight=SMOOTHING_WEIGHT)
            facts.update({f"curves_{name}_first": round_fact(value=train_values[0]),
                          f"curves_{name}_last": round_fact(value=train_values[-1]),
                          f"curves_{name}_last_smoothed": round_fact(value=smoothed[-1]),
                          f"curves_{name}_validation_last": round_fact(value=validation_values[-1]),
                          f"curves_{name}_validation_count": int(len(validation_values))})
        axes.scatter([], [], s=(5 * POINTS_PER_PIXEL) ** 2, color=TEXT_PRIMARY_COLOUR, label="validation batches")
        axes.legend(loc="center right", frameon=False, labelcolor=TEXT_PRIMARY_COLOUR, fontsize=18 * POINTS_PER_PIXEL)
    figure.canvas.draw()
    facts["curves_validation_step"] = int(validation_iterations[0])

    # First iteration whose smoothed training pixel accuracy reaches the threshold, read again outside the drawing loop
    accuracy_iterations, accuracy_values = read_scalars(log_folder=run_folder / "Tensorboard" / "Train",
                                                        tag="Pixel Accuracy")
    reached = numpy.nonzero(smooth_values(values=accuracy_values, weight=SMOOTHING_WEIGHT) >=
                            PIXEL_ACCURACY_THRESHOLD)[0]
    facts["curves_pixel_accuracy_threshold"] = PIXEL_ACCURACY_THRESHOLD
    facts["curves_pixel_accuracy_reaches_0_9"] = int(accuracy_iterations[reached[0]]) if len(reached) else -1

    # First iteration at which the smoothed cross entropy has made LOSS_FALL_SHARE of its fall, the Dice loss falling
    # until the end of the run
    loss_iterations, loss_values = read_scalars(log_folder=run_folder / "Tensorboard" / "Train",
                                                tag="Cross Entropy Loss")
    smoothed_losses = smooth_values(values=loss_values, weight=SMOOTHING_WEIGHT)
    fall_target = smoothed_losses[0] - LOSS_FALL_SHARE * (smoothed_losses[0] - smoothed_losses[-1])
    facts["curves_loss_fall_share"] = LOSS_FALL_SHARE
    facts["curves_cross_entropy_loss_fall_iteration"] = int(loss_iterations[numpy.nonzero(smoothed_losses <=
                                                                                          fall_target)[0][0]])
    facts["curves_loss_panel_box"] = compute_pixel_box(figure=figure, axes=loss_axes, with_labels=True, margin=6)
    facts["curves_metric_panel_box"] = compute_pixel_box(figure=figure, axes=metric_axes, with_labels=True, margin=6)
    facts["curves_loss_reveal_region"] = compute_pixel_box(figure=figure, axes=loss_axes, with_labels=False,
                                                           margin=REVEAL_MARGIN)
    facts["curves_metric_reveal_region"] = compute_pixel_box(figure=figure, axes=metric_axes, with_labels=False,
                                                             margin=REVEAL_MARGIN)
    save_figure(figure=figure, output_path=output_folder / "training" /
                ("training_curves.png" if with_curves else "training_curves_axes.png"))
    pyplot.close(fig=figure)
    return facts


def describe_pixel_shares(run_folder: Path, number_classes: int) -> Dict[str, Any]:
    """
    Gives the share of the labelled test pixels of every class, read from the class_metrics.csv of the last evaluation
    of the run, the bars of the loss scene with their percentages ("90.32 %"), and the cross entropy of a uniform
    prediction, ln K.

    Args:
        run_folder (Path): The folder of the segmentation run.
        number_classes (int): The number of classes, the background included.

    Returns:
        Dict[str, Any]: The facts of the pixel shares.
    """
    csv_path = run_folder / EVALUATION_FOLDER_NAME / f"Iteration_{CHECKPOINT_ITERATION}" / "class_metrics.csv"
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    rows = [dict(zip(header, line.split(","))) for line in lines[1:]]
    pixels = numpy.array([int(row["pixels"]) for row in rows], dtype=numpy.float64)
    shares = pixels / pixels.sum()
    order = numpy.argsort(-shares)
    return {"shares_values": [round_fact(value=shares[index]) for index in order],
            "shares_labels": [rows[index]["class"].replace("_", " ") for index in order],
            "shares_value_labels": [format_percentage(value=shares[index]) for index in order],
            "shares_background": round_fact(value=shares[0]), "shares_labelled_pixels": int(pixels.sum()),
            "shares_smallest_class": rows[int(order[-1])]["class"].replace("_", " "),
            "shares_smallest": round_fact(value=shares[order[-1]]),
            "shares_uniform_cross_entropy": round_fact(value=math.log(number_classes))}


def read_training_throughput(run_folder: Path) -> Dict[str, Any]:
    """
    Reads the throughput of the training of the segmentation run from the summaries of its log, the
    semantic_segmentation.log written next to its run folder.

    Args:
        run_folder (Path): The folder of the segmentation run.

    Returns:
        Dict[str, Any]: The mean throughput in images per second and the number of summaries.

    Raises:
        FileNotFoundError: If the log is missing.
    """
    log_path = run_folder.parent / "semantic_segmentation.log"
    if not log_path.exists():
        raise FileNotFoundError(f"No training log {log_path} of the segmentation run.")
    log_text = log_path.read_text(encoding="utf-8")
    throughputs = [float(value) for value in re.findall(pattern=r"Throughput: ([\d.]+) Images Per Second",
                                                        string=log_text)]
    durations = re.findall(pattern=r"Interval Of (\d+) Iterations Took (\d+):(\d+):(\d+)", string=log_text)
    seconds = [int(hours) * 3600 + int(minutes) * 60 + int(remaining)
               for interval_iterations, hours, minutes, remaining in durations]
    return {"training_throughput": round_fact(value=sum(throughputs) / len(throughputs)),
            "training_interval_iterations": int(durations[0][0]) if durations else 0,
            "training_interval_seconds": round_fact(value=float(numpy.median(seconds))) if seconds else 0.0}
