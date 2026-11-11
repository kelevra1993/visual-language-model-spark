"""
Learning rate schedule of a real configuration of the repository for the foundations tutorials (the optimization
chapter), made at every build by tutorials/figures/foundations/figures.py: the learning rate at every iteration of the
configuration named learning_rate_schedule_configuration in tutorials/tutorial_sources.yaml (the synthetic keypoint
configuration, linear warm-up then cosine decay), computed by compute_learning_rate_factor of the repository, the
function its trainer follows, on two panels of the same scale, the warm-up enlarged on the left and the whole run on the
right. Two PNG files are written: the empty axes, the base image of the scene, and the whole chart, its overlay, whose
curves the scene uncovers while they are explained; the boxes of the panels and the learning rates are the facts of the
storyboards.
"""

import matplotlib
import numpy

from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, MONOSPACE_FONT_FAMILY,
                                                POINTS_PER_PIXEL, REVEAL_MARGIN, SERIES_COLOURS, SURFACE_COLOUR,
                                                TEXT_PRIMARY_COLOUR, TEXT_SECONDARY_COLOUR, compute_pixel_box,
                                                register_fonts, round_fact, save_figure, style_chart_panel)
from tutorial_builder.figure_context import find_repository_root
from vision_training.configuration_processing.configuration_loader import load_configuration
from vision_training.trainers.learning_rate_schedule import compute_learning_rate_factor

# Names of the figures, kept from the first tutorials
FIGURE_NAME = "learning_rate_schedule_keypoint_synthetic_shapes"

# Colour of the curve, the first of the categorical palette of the videos
CURVE_COLOUR = SERIES_COLOURS[0]

# Iterations shown by the panel enlarging the warm-up, and the number of labelled ticks of both panels
WARMUP_PANEL_ITERATIONS = 1000
TICK_COUNT = 5

# Names of the usual ratios of the minimum learning rate, written under the label of the last iteration
RATIO_NAMES = {0.1: "a tenth", 0.01: "a hundredth", 0.001: "a thousandth"}


def read_schedule_settings(configuration_path: Path) -> Tuple[float, int, Dict[str, Any]]:
    """
    Reads the learning rate, the number of iterations and the Optimization section of a training configuration, the
    settings the schedule of the figure is computed from.

    Args:
        configuration_path (Path): The YAML configuration of the repository.

    Returns:
        Tuple[float, int, Dict[str, Any]]: The learning rate, the number of iterations and the Optimization section.

    Raises:
        ValueError: If the schedule is not the cosine one drawn by the figure.
    """
    configuration = load_configuration(configuration_path=configuration_path)
    learning_rate = float(configuration["ExperimentConfiguration"]["learning_rate"])
    number_iterations = int(float(configuration["ExperimentConfiguration"]["number_iterations"]))
    optimization_configuration = configuration["Optimization"]
    if optimization_configuration["learning_rate_schedule"] != "cosine":
        raise ValueError(f"Expected the cosine schedule in {configuration_path}, got "
                         f"{optimization_configuration['learning_rate_schedule']!r}.")
    return learning_rate, number_iterations, optimization_configuration


def compute_learning_rates(learning_rate: float, number_iterations: int,
                           optimization_configuration: Dict[str, Any]) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Computes the learning rate of every iteration of the run, from 0 (before the first step) to its last iteration,
    with the schedule function of the repository.

    Args:
        learning_rate (float): The learning rate of the configuration.
        number_iterations (int): The last iteration of the run.
        optimization_configuration (Dict[str, Any]): The Optimization section of the configuration.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The iterations and their learning rates.
    """
    iterations = numpy.arange(number_iterations + 1)
    factors = [compute_learning_rate_factor(iteration=int(iteration),
                                            optimization_configuration=optimization_configuration)
               for iteration in iterations]
    return iterations, learning_rate * numpy.array(factors)


def format_learning_rate(value: float, position: int) -> str:
    """
    Writes a learning rate in the notation of the configuration files (1.0e-3), zero as 0, for the ticks and labels.

    Args:
        value (float): The learning rate.
        position (int): The tick position, required by FuncFormatter.

    Returns:
        str: The written learning rate.
    """
    if abs(value) < 1e-12:
        return "0"
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def format_compact_iteration(value: float, position: int) -> str:
    """
    Writes an iteration tick compactly from 1,000 up (1K, 5K, 20K), as the axis ticks of the reference charts.

    Args:
        value (float): The tick value.
        position (int): The tick position, required by FuncFormatter.

    Returns:
        str: The tick label.
    """
    if value >= 1000:
        return f"{value / 1000:g}K"
    return f"{value:g}"


def name_learning_rate_ratio(ratio: float) -> str:
    """
    Names the minimum learning rate as a share of the learning rate, for the label of the last iteration: "a hundredth
    of the learning rate" for a ratio of 0.01.

    Args:
        ratio (float): The minimum_learning_rate_ratio of the configuration.

    Returns:
        str: The share in words.
    """
    if ratio in RATIO_NAMES:
        return f"{RATIO_NAMES[ratio]} of the learning rate"
    return f"{ratio:g} times the learning rate"


def style_panel(axes: matplotlib.axes.Axes, title: str, x_limit: float, x_ticks: List[float], y_limit: float,
                y_label: Optional[str]) -> None:
    """
    Gives a panel of the schedule the anatomy of the charts of the videos, the iterations in compact ticks and the
    learning rates in the notation of the configuration files, on the same learning rate scale on both panels.

    Args:
        axes (matplotlib.axes.Axes): The panel.
        title (str): The title of the panel.
        x_limit (float): The last iteration shown.
        x_ticks (List[float]): The iterations labelled on the axis.
        y_limit (float): The top of the learning rate axis.
        y_label (Optional[str]): The name of the learning rate axis, None for the right panel.
    """
    style_chart_panel(axes=axes, title=title, x_label="Iteration", y_label=y_label)
    axes.set_xlim(left=0, right=x_limit)
    axes.set_ylim(bottom=0, top=y_limit)
    axes.set_xticks(ticks=x_ticks)
    axes.xaxis.set_major_formatter(formatter=FuncFormatter(func=format_compact_iteration))
    axes.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_learning_rate))


def draw_learning_rate(axes: matplotlib.axes.Axes, iterations: numpy.ndarray, learning_rates: numpy.ndarray) -> None:
    """
    Draws the learning rate curve of a panel: the line in the series colour over its light wash.

    Args:
        axes (matplotlib.axes.Axes): The panel.
        iterations (numpy.ndarray): The iterations.
        learning_rates (numpy.ndarray): Their learning rates.
    """
    axes.fill_between(x=iterations, y1=0, y2=learning_rates, color=CURVE_COLOUR, alpha=0.10, linewidth=0)
    axes.plot(iterations, learning_rates, color=CURVE_COLOUR, linewidth=3 * POINTS_PER_PIXEL,
              solid_capstyle="round", solid_joinstyle="round")


def mark_point(axes: matplotlib.axes.Axes, iteration: float, learning_rate: float, label: str,
               label_offset: Tuple[float, float], horizontal_alignment: str, leader: bool) -> None:
    """
    Marks one point of the curve with an end-dot ringed by the surface and its direct label in the primary grey,
    joined to the dot by a thin leader line when the label sits away from it, clear of the curve.

    Args:
        axes (matplotlib.axes.Axes): The panel.
        iteration (float): The iteration of the point.
        learning_rate (float): Its learning rate.
        label (str): The label written next to it, on one or several lines.
        label_offset (Tuple[float, float]): The offset of the label from the point, in pixels.
        horizontal_alignment (str): The alignment of the label on its offset point.
        leader (bool): Whether a leader line joins the bottom corner of the label to the dot.
    """
    axes.plot([iteration], [learning_rate], marker="o", markersize=12 * POINTS_PER_PIXEL, color=CURVE_COLOUR,
              markeredgecolor=SURFACE_COLOUR, markeredgewidth=2 * POINTS_PER_PIXEL, clip_on=False, zorder=4)

    # Leader line from the bottom corner of the label on the side of its alignment, stopping short of the dot
    leader_properties = None
    if leader:
        leader_properties = {"arrowstyle": "-", "color": TEXT_SECONDARY_COLOUR, "linewidth": 1.5 * POINTS_PER_PIXEL,
                             "shrinkA": 6, "shrinkB": 9,
                             "relpos": (1.0 if horizontal_alignment == "right" else 0.0, 0.0)}
    axes.annotate(text=label, xy=(iteration, learning_rate), xytext=(label_offset[0] * POINTS_PER_PIXEL,
                                                                    label_offset[1] * POINTS_PER_PIXEL),
                  textcoords="offset points", color=TEXT_PRIMARY_COLOUR, fontsize=19 * POINTS_PER_PIXEL,
                  ha=horizontal_alignment, va="center", multialignment=horizontal_alignment,
                  arrowprops=leader_properties)


def make_learning_rate_schedule_figure(output_folder: Path, configuration_path: Path) -> Dict[str, Any]:
    """
    Draws the learning rate schedule of a configuration of the repository, as the base image and the overlay of its
    scene, and gives the boxes of its panels and its settings and learning rates.

    Args:
        output_folder (Path): The figures folder of the series.
        configuration_path (Path): The configuration of the schedule, in the repository.

    Returns:
        Dict[str, Any]: The facts of the figure, named learning_rate_schedule.*.
    """
    # Learning rate of every iteration of the configuration, from the schedule function of the repository
    learning_rate, number_iterations, optimization_configuration = read_schedule_settings(
        configuration_path=configuration_path)
    iterations, learning_rates = compute_learning_rates(learning_rate=learning_rate,
                                                        number_iterations=number_iterations,
                                                        optimization_configuration=optimization_configuration)
    warmup_iterations = optimization_configuration["warmup_iterations"]
    decay_iterations = optimization_configuration["decay_iterations"]
    minimum_ratio = optimization_configuration["minimum_learning_rate_ratio"]

    # Figure of the size of the viewport of a scene, the warm-up panel narrower than the whole run panel
    register_fonts()
    figure, (warmup_axes, run_axes) = pyplot.subplots(nrows=1, ncols=2, figsize=(CHART_WIDTH / DOTS_PER_INCH,
                                                                              CHART_HEIGHT / DOTS_PER_INCH),
                                                      dpi=DOTS_PER_INCH, gridspec_kw={"width_ratios": [2, 3]})
    figure.patch.set_facecolor(SURFACE_COLOUR)
    figure.subplots_adjust(left=0.075, right=0.975, top=0.735, bottom=0.12, wspace=0.12)

    # Header: the configuration file, then the schedule settings drawn on two lines, in the notation of the file
    figure.text(x=0.012, y=0.957, s=str(configuration_path.relative_to(find_repository_root())),
                color=TEXT_SECONDARY_COLOUR, fontsize=19 * POINTS_PER_PIXEL, va="center", family=MONOSPACE_FONT_FAMILY)
    figure.text(x=0.012, y=0.905, s=f"learning_rate: {format_learning_rate(value=learning_rate, position=0)}   "
                                    f"learning_rate_schedule: \"cosine\"   warmup_iterations: {warmup_iterations}",
                color=TEXT_PRIMARY_COLOUR, fontsize=20 * POINTS_PER_PIXEL, va="center", family=MONOSPACE_FONT_FAMILY)
    figure.text(x=0.012, y=0.858, s=f"decay_iterations: {decay_iterations}   minimum_learning_rate_ratio: "
                                    f"{minimum_ratio:g}",
                color=TEXT_PRIMARY_COLOUR, fontsize=20 * POINTS_PER_PIXEL, va="center", family=MONOSPACE_FONT_FAMILY)

    # Both panels on the same learning rate scale, saved empty as the base image of the scene
    y_limit = learning_rate * 1.12
    style_panel(axes=warmup_axes, title=f"Warm-up · first {WARMUP_PANEL_ITERATIONS:,} iterations",
                x_limit=WARMUP_PANEL_ITERATIONS,
                x_ticks=numpy.linspace(start=0, stop=WARMUP_PANEL_ITERATIONS, num=TICK_COUNT).tolist(),
                y_limit=y_limit, y_label="Learning rate")
    style_panel(axes=run_axes, title=f"Whole run · {number_iterations:,} iterations", x_limit=number_iterations,
                x_ticks=numpy.linspace(start=0, stop=number_iterations, num=TICK_COUNT).tolist(), y_limit=y_limit,
                y_label=None)
    run_axes.tick_params(axis="y", labelleft=False)
    save_figure(figure=figure, output_path=output_folder / f"{FIGURE_NAME}_axes.png")

    # Curves of the repository schedule on both panels
    warmup_shown = iterations <= WARMUP_PANEL_ITERATIONS
    draw_learning_rate(axes=warmup_axes, iterations=iterations[warmup_shown],
                       learning_rates=learning_rates[warmup_shown])
    draw_learning_rate(axes=run_axes, iterations=iterations, learning_rates=learning_rates)

    # Direct labels on the points the narration names: the end of the warm-up, under the plateau right of its dot, and
    # the last iteration, in the empty upper right of its panel with a leader line down to its dot
    mark_point(axes=warmup_axes, iteration=warmup_iterations, learning_rate=learning_rates[warmup_iterations],
               label=f"{format_learning_rate(value=learning_rates[warmup_iterations], position=0)} at iteration "
                     f"{warmup_iterations}", label_offset=(16, -34), horizontal_alignment="left", leader=False)
    mark_point(axes=run_axes, iteration=number_iterations, learning_rate=learning_rates[-1],
               label=f"{format_learning_rate(value=learning_rates[-1], position=0)} at iteration {number_iterations:,}"
                     f"\n{name_learning_rate_ratio(ratio=minimum_ratio)}", label_offset=(-20, 230),
               horizontal_alignment="right", leader=True)

    # Whole chart written as the overlay of the scene, then the boxes of its highlights and reveals
    save_figure(figure=figure, output_path=output_folder / f"{FIGURE_NAME}.png")
    facts = {}
    for panel_name, panel_axes in [("warmup", warmup_axes), ("whole_run", run_axes)]:
        facts[f"learning_rate_schedule.{panel_name}_panel_box"] = compute_pixel_box(figure=figure, axes=panel_axes,
                                                                                    with_labels=True, margin=0)
        facts[f"learning_rate_schedule.{panel_name}_reveal_region"] = compute_pixel_box(
            figure=figure, axes=panel_axes, with_labels=False, margin=REVEAL_MARGIN)
    pyplot.close(fig=figure)

    # Settings of the schedule and the learning rates the labels and the narration quote
    facts.update({"learning_rate_schedule.learning_rate": round_fact(value=learning_rate),
                  "learning_rate_schedule.number_iterations": number_iterations,
                  "learning_rate_schedule.warmup_iterations": warmup_iterations,
                  "learning_rate_schedule.decay_iterations": decay_iterations,
                  "learning_rate_schedule.minimum_learning_rate_ratio": round_fact(value=minimum_ratio),
                  "learning_rate_schedule.warmup_end_learning_rate": round_fact(
                      value=learning_rates[warmup_iterations]),
                  "learning_rate_schedule.last_learning_rate": round_fact(value=learning_rates[-1]),
                  "learning_rate_schedule.maximum_learning_rate": round_fact(value=learning_rates.max())})
    return facts
