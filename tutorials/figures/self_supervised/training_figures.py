"""
Training figures of the self-supervised series of the tutorial videos (chapter 2, from views to a loss, and chapter 3,
the batch normalization pitfall), made by tutorials/figures/self_supervised/figures.py from the TensorBoard logs and the
training log of the SimCLR run: the InfoNCE loss against ln(2N - 1), the positive pair accuracy against its chance
1 / (2N - 1) and the embedding standard deviation against 1, for the training batches and the validation batches (an
empty-axes base image and its overlay, whose curves the scene uncovers); the first iterations of the validation, whose
running statistics of the batch normalization had not settled yet; and the last console summary of the run, a listing.
"""

import math
import re

import numpy

from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Any, Dict, List, Tuple

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, POINTS_PER_PIXEL,
                                                REVEAL_MARGIN, SERIES_COLOURS, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR,
                                                TEXT_SECONDARY_COLOUR, compute_pixel_box, format_thousands,
                                                read_run_configuration, register_fonts, round_fact, save_figure,
                                                style_chart_panel)
from tutorials.figures.self_supervised.ssl_sources import LOG_FILE_NAME

# Smoothing weight of the training curves, the exponential moving average of TensorBoard
SMOOTHING_WEIGHT = 0.95

# Panels of the training curves: the scalar, its title, its value axis, its limits and the fact name of its scalar
CURVE_PANELS = [{"tag": "InfoNCE Loss", "title": "InfoNCE loss", "y_label": "Loss", "limits": (3.0, 6.0),
                 "name": "info_nce"},
                {"tag": "Positive Pair Accuracy", "title": "Positive pair accuracy", "y_label": "Share of the views",
                 "limits": (0.0, 0.4), "name": "positive_pair_accuracy"},
                {"tag": "Embedding Standard Deviation", "title": "Embedding standard deviation", "y_label": "Value",
                 "limits": (0.0, 1.1), "name": "embedding_standard_deviation"}]

# Iterations shown by the figure of the first validations, and between two of its labelled ticks
EARLY_ITERATIONS = 200
EARLY_TICK_STEP = 50

# Iterations between two labelled ticks of the training curves
ITERATION_TICK_STEP = 500

# Pattern of the escape sequences of the colours and links of the console, removed from the log
ESCAPE_SEQUENCE_PATTERN = re.compile(pattern=r"\x1b\[[0-9;]*m|\x1b\]8;;[^\x1b]*\x1b\\")


def read_scalars(log_folder: Path, tag: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Reads every value of one scalar of a TensorBoard log folder of the run with tensorboard's EventAccumulator.

    Args:
        log_folder (Path): The Train, Validation or Test folder of the run.
        tag (str): The scalar, e.g. "InfoNCE Loss".

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The iterations and the values.

    Raises:
        FileNotFoundError: If the log folder is missing.
        KeyError: If the folder has no such scalar.
    """
    if not log_folder.is_dir():
        raise FileNotFoundError(f"No TensorBoard log folder {log_folder}.")
    accumulator = EventAccumulator(path=str(log_folder), size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags()["scalars"]:
        raise KeyError(f"No scalar {tag!r} in {log_folder}, whose scalars are {accumulator.Tags()['scalars']}.")
    events = accumulator.Scalars(tag)
    return numpy.array([event.step for event in events]), numpy.array([event.value for event in events])


def smooth_values(values: numpy.ndarray, weight: float) -> numpy.ndarray:
    """
    Smooths a curve the way TensorBoard does: an exponential moving average divided by the sum of its weights, so that
    the first values are not pulled towards zero.

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


def fit_axis_limits(limits: Tuple[float, float], values: List[numpy.ndarray]) -> Tuple[float, float]:
    """
    Widens the value axis of a panel so that every value it draws stays inside: a side of the designed limits is kept
    when the values of the run stay within it, and moved past the farthest value by a tenth of the designed span
    otherwise, a margin for the labels above the points, so that a run trained again with other views or for longer
    never draws outside its panel.

    Args:
        limits (Tuple[float, float]): The designed bottom and top of the value axis.
        values (List[numpy.ndarray]): The values drawn in the panel, the reference lines included.

    Returns:
        Tuple[float, float]: The bottom and top of the value axis.
    """
    margin = 0.1 * (limits[1] - limits[0])
    lowest = min(float(numpy.nanmin(array)) for array in values)
    highest = max(float(numpy.nanmax(array)) for array in values)
    bottom = limits[0] if lowest >= limits[0] else lowest - margin
    top = limits[1] if highest <= limits[1] else highest + margin
    return bottom, top


def draw_reference_line(axes: Any, value: float, label: str, right: float) -> None:
    """
    Draws a dashed horizontal reference of a panel (the loss of random embeddings, a chance level, the spread of
    embeddings covering the sphere) with its label at the right end.

    Args:
        axes (Any): The matplotlib panel.
        value (float): The height of the line.
        label (str): The text written above the line.
        right (float): The right end of the horizontal axis, where the label is written.
    """
    axes.axhline(y=value, color=TEXT_SECONDARY_COLOUR, linewidth=2 * POINTS_PER_PIXEL, linestyle=(0, (6, 5)),
                 zorder=1)
    axes.text(x=right, y=value, s=label, color=TEXT_SECONDARY_COLOUR, fontsize=17 * POINTS_PER_PIXEL, ha="right",
              va="bottom")


def compute_reference_values(batch_size: int) -> Dict[str, float]:
    """
    Gives the values of random embeddings for a batch of N images: an InfoNCE loss of ln(2N - 1), every view being
    as similar to its partner as to the 2N - 2 other views, and a positive pair accuracy of 1 / (2N - 1).

    Args:
        batch_size (int): The number N of images of a batch.

    Returns:
        Dict[str, float]: The 'info_nce' and 'positive_pair_accuracy' of random embeddings.
    """
    return {"info_nce": math.log(2 * batch_size - 1), "positive_pair_accuracy": 1.0 / (2 * batch_size - 1)}


def draw_training_curves(output_folder: Path, run_folder: Path, with_curves: bool) -> Dict[str, Any]:
    """
    Draws the three panels of the training curves of the run, the training values smoothed like TensorBoard over
    their raw values and the validation values as dots, each panel with its reference line; the empty axes with the
    legend and the references for the base image of the scene, or the whole chart for its overlay.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the self-supervised run.
        with_curves (bool): Whether the curves are drawn (the overlay) or only the axes, legend and references.

    Returns:
        Dict[str, Any]: The facts of the curves and the boxes of the panels.
    """
    register_fonts()
    batch_size = int(read_run_configuration(run_folder=run_folder)["ExperimentConfiguration"]["batch_size"])
    references = compute_reference_values(batch_size=batch_size)
    figure = pyplot.figure(figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH,
                           facecolor=SURFACE_COLOUR)
    facts = {}
    for panel_index, panel in enumerate(CURVE_PANELS):
        axes = figure.add_axes(rect=(0.06 + panel_index * 0.33, 0.14, 0.26, 0.72))
        train_iterations, train_values = read_scalars(log_folder=run_folder / "Tensorboard" / "Train",
                                                      tag=panel["tag"])
        validation_iterations, validation_values = read_scalars(log_folder=run_folder / "Tensorboard" /
                                                                "Validation", tag=panel["tag"])
        last_iteration = int(train_iterations[-1])
        style_chart_panel(axes=axes, title=panel["title"], x_label="Iteration", y_label=panel["y_label"])
        axes.set_xlim(left=0, right=last_iteration)
        reference_value = {"info_nce": references["info_nce"],
                           "positive_pair_accuracy": references["positive_pair_accuracy"]}.get(panel["name"], 1.0)
        bottom, top = fit_axis_limits(limits=panel["limits"], values=[train_values, validation_values,
                                                                      numpy.array([reference_value])])
        axes.set_ylim(bottom=bottom, top=top)
        axes.set_xticks(ticks=list(range(0, last_iteration + 1, ITERATION_TICK_STEP)))
        axes.xaxis.set_major_formatter(formatter=FuncFormatter(func=format_thousands))

        # Reference line of the panel: random embeddings for the loss and the accuracy, the whole sphere for the spread
        if panel["name"] == "info_nce":
            draw_reference_line(axes=axes, value=references["info_nce"], right=last_iteration,
                                label=f"random: ln {2 * batch_size - 1} = {references['info_nce']:.2f}")
        elif panel["name"] == "positive_pair_accuracy":
            draw_reference_line(axes=axes, value=references["positive_pair_accuracy"], right=last_iteration,
                                label=f"chance: 1 / {2 * batch_size - 1}")
        else:
            draw_reference_line(axes=axes, value=1.0, right=last_iteration, label="spread over the sphere: 1")

        # Training values smoothed over their faint raw values, the validation values as small dots
        smoothed_values = smooth_values(values=train_values, weight=SMOOTHING_WEIGHT)
        colour = SERIES_COLOURS[panel_index]
        if with_curves:
            axes.plot(train_iterations, train_values, color=colour, alpha=0.22, linewidth=1.2 * POINTS_PER_PIXEL)
            axes.plot(train_iterations, smoothed_values, color=colour, linewidth=3 * POINTS_PER_PIXEL,
                      label="training, smoothed", solid_capstyle="round")
            axes.scatter(validation_iterations, validation_values, s=(3 * POINTS_PER_PIXEL) ** 2,
                         color=TEXT_PRIMARY_COLOUR, alpha=0.55, linewidths=0, label="validation batches", zorder=3)
        else:
            axes.plot([], [], color=colour, linewidth=3 * POINTS_PER_PIXEL, label="training, smoothed")
            axes.scatter([], [], s=(3 * POINTS_PER_PIXEL) ** 2, color=TEXT_PRIMARY_COLOUR, label="validation batches")
        legend_places = {"info_nce": "lower left", "positive_pair_accuracy": "upper left"}
        axes.legend(loc=legend_places.get(panel["name"], "lower right"), frameon=False,
                    labelcolor=TEXT_PRIMARY_COLOUR, fontsize=16 * POINTS_PER_PIXEL)

        # Facts of the panel: first and last values, the last smoothed value and the last validation value
        facts.update({f"curves.{panel['name']}_first": round_fact(value=train_values[0]),
                      f"curves.{panel['name']}_last": round_fact(value=train_values[-1]),
                      f"curves.{panel['name']}_last_smoothed": round_fact(value=smoothed_values[-1]),
                      f"curves.{panel['name']}_validation_last": round_fact(value=validation_values[-1]),
                      f"curves.{panel['name']}_validation_first": round_fact(value=validation_values[0])})
        figure.canvas.draw()
        facts[f"curves.{panel['name']}_panel_box"] = compute_pixel_box(figure=figure, axes=axes, with_labels=True,
                                                                     margin=6)
        facts[f"curves.{panel['name']}_reveal_region"] = compute_pixel_box(figure=figure, axes=axes,
                                                                         with_labels=False, margin=REVEAL_MARGIN)
        facts["curves.last_iteration"] = last_iteration
        facts["curves.validation_step"] = int(validation_iterations[0])
        facts["curves.validation_count"] = int(len(validation_values))
    facts.update({"curves.batch_size": batch_size, "curves.view_count": 2 * batch_size,
                  "curves.other_view_count": 2 * batch_size - 1,
                  "curves.random_info_nce": round_fact(value=references["info_nce"]),
                  "curves.chance_positive_pair_accuracy": round_fact(value=references["positive_pair_accuracy"])})
    facts["curves.positive_pair_accuracy_first_over_chance"] = round_fact(
        value=facts["curves.positive_pair_accuracy_first"] / references["positive_pair_accuracy"])
    save_figure(figure=figure, output_path=output_folder / "training" /
                ("training_curves.png" if with_curves else "training_curves_axes.png"))
    pyplot.close(fig=figure)
    return facts


def draw_early_validation(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Draws the first iterations of the run, where the validation batches, encoded in evaluation mode with the running
    statistics of the batch normalization, gave embeddings all alike and the loss of random embeddings, while the
    training batches, normalized by their own statistics, did not: the embedding standard deviation and the InfoNCE
    loss of both, raw.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the self-supervised run.

    Returns:
        Dict[str, Any]: The facts of the first validations and the boxes of the panels.
    """
    register_fonts()
    batch_size = int(read_run_configuration(run_folder=run_folder)["ExperimentConfiguration"]["batch_size"])
    figure = pyplot.figure(figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH,
                           facecolor=SURFACE_COLOUR)
    facts = {}
    panels = [("Embedding Standard Deviation", "Embedding standard deviation", (0.0, 1.0), "spread"),
              ("InfoNCE Loss", "InfoNCE loss", (4.4, 5.8), "loss")]
    for panel_index, (tag, title, limits, name) in enumerate(panels):
        axes = figure.add_axes(rect=(0.07 + panel_index * 0.5, 0.14, 0.40, 0.72))
        style_chart_panel(axes=axes, title=title, x_label="Iteration", y_label="Value" if panel_index == 0 else "Loss")
        axes.set_xlim(left=0, right=EARLY_ITERATIONS)
        axes.set_xticks(ticks=list(range(0, EARLY_ITERATIONS + 1, EARLY_TICK_STEP)))

        # Training values of every iteration and the validation values, raw, the value axis holding all of them
        train_iterations, train_values = read_scalars(log_folder=run_folder / "Tensorboard" / "Train", tag=tag)
        validation_iterations, validation_values = read_scalars(log_folder=run_folder / "Tensorboard" /
                                                                "Validation", tag=tag)
        train_kept = train_iterations <= EARLY_ITERATIONS
        validation_kept = validation_iterations <= EARLY_ITERATIONS
        drawn_values = [train_values[train_kept], validation_values[validation_kept]]
        if name == "loss":
            drawn_values.append(numpy.array([math.log(2 * batch_size - 1)]))
        bottom, top = fit_axis_limits(limits=limits, values=drawn_values)
        axes.set_ylim(bottom=bottom, top=top)
        axes.plot(train_iterations[train_kept], train_values[train_kept], color=SERIES_COLOURS[0],
                  linewidth=2 * POINTS_PER_PIXEL, label="training batches (batch statistics)")
        axes.plot(validation_iterations[validation_kept], validation_values[validation_kept], color=SERIES_COLOURS[3],
                  linewidth=2 * POINTS_PER_PIXEL, marker="o", markersize=7 * POINTS_PER_PIXEL,
                  label="validation batches (running statistics)")
        if name == "loss":
            draw_reference_line(axes=axes, value=math.log(2 * batch_size - 1), right=EARLY_ITERATIONS,
                                label=f"random: ln {2 * batch_size - 1}")
        axes.legend(loc="lower right" if name == "spread" else "upper right", frameon=False,
                    labelcolor=TEXT_PRIMARY_COLOUR, fontsize=17 * POINTS_PER_PIXEL)
        figure.canvas.draw()
        facts[f"early.{name}_panel_box"] = compute_pixel_box(figure=figure, axes=axes, with_labels=True, margin=6)

        # The first validations and their training counterparts
        for validation_index in range(3):
            iteration = int(validation_iterations[validation_index])
            facts[f"early.{name}_validation_{validation_index}"] = round_fact(value=validation_values[validation_index])
            facts[f"early.validation_iteration_{validation_index}"] = iteration
            facts[f"early.{name}_train_at_validation_{validation_index}"] = round_fact(
                value=train_values[numpy.nonzero(train_iterations == iteration)[0][0]])
    # How long the validation stayed at the loss of random embeddings, and when its spread reached half a sphere
    validation_iterations, validation_losses = read_scalars(log_folder=run_folder / "Tensorboard" / "Validation",
                                                            tag="InfoNCE Loss")
    _, validation_spreads = read_scalars(log_folder=run_folder / "Tensorboard" / "Validation",
                                         tag="Embedding Standard Deviation")
    at_random = numpy.abs(validation_losses - math.log(2 * batch_size - 1)) < 0.005
    if not at_random[0]:
        raise ValueError(f"The first validation loss of {run_folder} is {validation_losses[0]:.4f}, not the "
                         f"ln({2 * batch_size - 1}) = {math.log(2 * batch_size - 1):.4f} of random embeddings: reword "
                         f"the first validations scene of the third self-supervised chapter, which says it is.")
    facts["early.validations_at_random"] = int(numpy.argmin(at_random)) if not at_random.all() else len(at_random)
    facts["early.last_iteration_at_random"] = int(validation_iterations[facts["early.validations_at_random"] - 1])
    half_spread_indices = numpy.nonzero(validation_spreads >= 0.5)[0]
    if len(half_spread_indices) == 0:
        raise ValueError(f"The validation embedding standard deviation of {run_folder} never reaches 0.5: reword the "
                         f"first validations scene of the third self-supervised chapter, which says when it does.")
    facts["early.spread_half_iteration"] = int(validation_iterations[half_spread_indices[0]])
    save_figure(figure=figure, output_path=output_folder / "training" / "early_validation.png")
    pyplot.close(fig=figure)
    return facts


def read_last_console_summary(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Copies the last console summary of the training log of the run, its colours and links removed, as a listing of
    the scenes, and gives the throughputs of the summaries of the log.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the self-supervised run, the log being written next to it.

    Returns:
        Dict[str, Any]: The facts of the log.

    Raises:
        FileNotFoundError: If the training log is missing.
        ValueError: If the log holds no console summary.
    """
    log_path = run_folder.parent / LOG_FILE_NAME
    if not log_path.exists():
        raise FileNotFoundError(f"No training log {log_path} of the self-supervised run.")
    log_lines = ESCAPE_SEQUENCE_PATTERN.sub(repl="", string=log_path.read_text(encoding="utf-8")).splitlines()

    # Every summary starts with its iteration line between two rules and ends with its rule after the validation line
    summary_starts = [line_index for line_index, line in enumerate(log_lines) if line.startswith("|") and
                      "Iteration:" in line]
    if not summary_starts:
        raise ValueError(f"No console summary in {log_path}.")
    last_start = summary_starts[-1]
    last_end = next(line_index for line_index in range(last_start, len(log_lines)) if
                    log_lines[line_index].startswith("Validation Iterations In This Interval"))
    summary_lines: List[str] = [line.replace("Moving Average of ", "") for line in
                                log_lines[last_start:last_end + 1] if not line.startswith("---")]
    listing_path = output_folder / "listings" / "last_console_summary.txt"
    listing_path.parent.mkdir(parents=True, exist_ok=True)
    listing_path.write_text(data="\n".join(summary_lines) + "\n", encoding="utf-8")

    # Throughputs of the summaries, the first interval holding the start of the loaders and of the views
    throughputs = [float(value) for value in re.findall(pattern=r"Throughput: ([\d.]+) Images Per Second",
                                                        string="\n".join(log_lines))]
    return {"log.summary_count": len(throughputs), "log.first_throughput": round_fact(value=throughputs[0]),
            "log.last_throughput": round_fact(value=throughputs[-1]),
            "log.median_throughput": round_fact(value=float(numpy.median(throughputs)))}


def make_training_figures(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Makes the training figures of the series and gives their facts: the base and overlay of the training curves, the
    figure of the first validations and the listing of the last console summary.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the self-supervised run.

    Returns:
        Dict[str, Any]: The facts, named curves.*, early.* and log.*.
    """
    facts = draw_training_curves(output_folder=output_folder, run_folder=run_folder, with_curves=True)
    draw_training_curves(output_folder=output_folder, run_folder=run_folder, with_curves=False)
    facts.update(draw_early_validation(output_folder=output_folder, run_folder=run_folder))
    facts.update(read_last_console_summary(output_folder=output_folder, run_folder=run_folder))
    return facts
