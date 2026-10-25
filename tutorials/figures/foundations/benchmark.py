"""
Benchmark charts of the foundations tutorials, made at every build by tutorials/figures/foundations/figures.py from the
launches recorded in documentation/architecture_benchmarking/results.csv (the benchmark_results file of
tutorials/tutorial_sources.yaml: synthetic shapes, 9 classes, one setting changed at a time, test accuracy at every 250
iterations up to 2000): the residual shortcuts and the feature map normalizations of the backbone chapter, the learning
rates and the moving average of the weights of the optimization chapter, and the spread between launches of the same
settings of the evaluation chapter. Every launch is drawn faintly under the mean of the launches of its setting, and
every chart is written twice, its empty axes with the legend (the base image of the scene) and the whole chart (its
overlay, whose curves the scene uncovers while they are explained); the boxes of the panels and the means of the
launches are the facts of the storyboards.
"""

import csv

import matplotlib
import numpy

from matplotlib import pyplot
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, GRID_COLOUR,
                                                POINTS_PER_PIXEL, REVEAL_MARGIN, SERIES_COLOURS, SURFACE_COLOUR,
                                                TEXT_PRIMARY_COLOUR, TEXT_SECONDARY_COLOUR, compute_pixel_box,
                                                format_thousands, register_fonts, round_fact, save_figure,
                                                style_chart_panel)

# Evaluated iterations of every launch, the columns of results.csv
EVALUATED_ITERATIONS = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]

# Accuracy of a network predicting a single class out of the 9 balanced classes
SINGLE_CLASS_ACCURACY = 1 / 9

# Launches of every chart, named as in results.csv: the launches kept by the earlier stages (run_13 and run_14) and
# the stage 6 launches without residual shortcuts and the stage 4 launches with layer normalization (stopped at
# iteration 500); the two launches of every learning rate of stage 1; the two launches of stage 13 evaluated with
# their trained and averaged weights; the six launches with and without squeeze and excitation of the repeat stage
KEPT_LAUNCHES = ["run_13", "run_14"]
RESIDUAL_OFF_LAUNCHES = ["run_21", "run_22"]
LAYER_NORMALIZATION_LAUNCHES = ["run_17", "run_18"]
LEARNING_RATE_LAUNCHES = {"1e-4": ["run_1", "run_2"], "2e-4": ["run_3", "run_4"], "5e-4": ["run_9", "run_10"],
                          "1e-3": ["run_5", "run_6"], "2e-3": ["run_7", "run_8"]}
MOVING_AVERAGE_LAUNCHES = ["run_37", "run_38"]
LAUNCH_NOISE_LAUNCHES = {"squeeze-and-excitation off": ["run_23", "run_23b", "run_23c", "run_24", "run_24b", "run_24c"],
                         "squeeze-and-excitation on": ["run_13", "run_13b", "run_13c", "run_14", "run_14b", "run_14c"]}


def read_launches(results_path: Path) -> Dict[Tuple[str, str], List[float]]:
    """
    Reads the test accuracies of every launch of the benchmark, the mean rows left out, the launches of the moving
    average stages being read once per evaluated weights.

    Args:
        results_path (Path): The results.csv of the benchmark.

    Returns:
        Dict[Tuple[str, str], List[float]]: The accuracies at the evaluated iterations (NaN when not reached), keyed by
            launch and evaluated weights ("" for the launches evaluating their trained weights only).
    """
    launches = {}
    with results_path.open(encoding="utf-8", newline="") as results_file:
        for row in csv.DictReader(results_file):
            if row["launch"] == "mean":
                continue
            launches[(row["launch"], row["evaluated_weights"])] = [
                float(row[f"accuracy_{iteration}"]) if row[f"accuracy_{iteration}"] else numpy.nan
                for iteration in EVALUATED_ITERATIONS]
    return launches


def select_launches(launches: Dict[Tuple[str, str], List[float]], launch_names: List[str],
                    evaluated_weights: str) -> List[List[float]]:
    """
    Gives the test accuracies of some launches of the benchmark, refusing a launch missing from results.csv so that a
    renamed launch never gives a chart with fewer curves.

    Args:
        launches (Dict[Tuple[str, str], List[float]]): The test accuracies of every launch.
        launch_names (List[str]): The launches, as named in results.csv.
        evaluated_weights (str): The evaluated weights of the launches, "" for the trained weights only.

    Returns:
        List[List[float]]: The accuracies of every launch, in the order of the names.

    Raises:
        ValueError: If a launch is not in results.csv.
    """
    missing_launches = [name for name in launch_names if (name, evaluated_weights) not in launches]
    if missing_launches:
        raise ValueError(f"Launches missing from results.csv with evaluated weights {evaluated_weights!r}: "
                         f"{missing_launches}")
    return [launches[(name, evaluated_weights)] for name in launch_names]


def style_panel(axes: matplotlib.axes.Axes, title: str, y_limit: Tuple[float, float],
                x_limit: Tuple[float, float]) -> None:
    """
    Gives a panel of test accuracies the anatomy of the charts of the videos, the evaluated iterations as ticks.

    Args:
        axes (matplotlib.axes.Axes): The panel.
        title (str): The title of the panel.
        y_limit (Tuple[float, float]): The bottom and top of the test accuracy axis.
        x_limit (Tuple[float, float]): The first and last iteration shown.
    """
    style_chart_panel(axes=axes, title=title, x_label="Iteration", y_label="Test accuracy")
    axes.set_xlim(left=x_limit[0], right=x_limit[1])
    axes.set_ylim(bottom=y_limit[0], top=y_limit[1])
    axes.set_xticks(ticks=[iteration for iteration in [0] + EVALUATED_ITERATIONS
                           if x_limit[0] <= iteration <= x_limit[1]])
    axes.xaxis.set_major_formatter(formatter=FuncFormatter(func=format_thousands))


def draw_single_class_line(axes: matplotlib.axes.Axes, x_limit: Tuple[float, float]) -> None:
    """
    Draws the accuracy of a network predicting a single class as a dashed reference line, labelled at its start.

    Args:
        axes (matplotlib.axes.Axes): The panel.
        x_limit (Tuple[float, float]): The first and last iteration shown.
    """
    axes.axhline(y=SINGLE_CLASS_ACCURACY, color=TEXT_SECONDARY_COLOUR, linewidth=1.5 * POINTS_PER_PIXEL,
                 linestyle=(0, (6, 5)), zorder=1)
    axes.text(x=x_limit[0] + 0.015 * (x_limit[1] - x_limit[0]), y=SINGLE_CLASS_ACCURACY + 0.025,
              s=f"one class in nine: {SINGLE_CLASS_ACCURACY:.3f}", color=TEXT_SECONDARY_COLOUR,
              fontsize=17 * POINTS_PER_PIXEL, va="bottom")


def draw_setting(axes: matplotlib.axes.Axes, launch_curves: List[List[float]], colour: str,
                 clip_to_panel: bool) -> numpy.ndarray:
    """
    Draws one setting of a panel: every launch as a faint line and the mean of the launches on top, with a marker at
    every evaluation, drawn whole on the edges of a panel showing every evaluation, cut at the edges of a zoomed panel.

    Args:
        axes (matplotlib.axes.Axes): The panel.
        launch_curves (List[List[float]]): The test accuracies of every launch of the setting.
        colour (str): The colour of the setting.
        clip_to_panel (bool): Whether the markers outside the limits of the panel are cut, for a zoomed panel.

    Returns:
        numpy.ndarray: The mean curve of the setting.
    """
    curves = numpy.array(launch_curves, dtype=float)
    for curve in curves:
        axes.plot(EVALUATED_ITERATIONS, curve, color=colour, alpha=0.3, linewidth=1.5 * POINTS_PER_PIXEL)
    mean_curve = curves.mean(axis=0)
    axes.plot(EVALUATED_ITERATIONS, mean_curve, color=colour, linewidth=3 * POINTS_PER_PIXEL, marker="o",
              markersize=9 * POINTS_PER_PIXEL, markeredgecolor=SURFACE_COLOUR, markeredgewidth=2 * POINTS_PER_PIXEL,
              solid_capstyle="round", solid_joinstyle="round", zorder=3, clip_on=clip_to_panel)
    return mean_curve


def label_point(axes: matplotlib.axes.Axes, iteration: float, value: float, text: str,
                offset: Tuple[float, float], colour: str) -> None:
    """
    Writes the direct label of a setting next to one of its points, in the primary grey, a short line in the colour of
    the setting carrying its identity.

    Args:
        axes (matplotlib.axes.Axes): The panel.
        iteration (float): The iteration of the labelled point.
        value (float): Its test accuracy.
        text (str): The label.
        offset (Tuple[float, float]): The offset of the label from the point, in pixels.
        colour (str): The colour of the setting.
    """
    axes.annotate(text=text, xy=(iteration, value), xytext=(offset[0] * POINTS_PER_PIXEL, offset[1] * POINTS_PER_PIXEL),
                  textcoords="offset points", color=TEXT_PRIMARY_COLOUR, fontsize=18 * POINTS_PER_PIXEL, ha="right",
                  va="center", arrowprops={"arrowstyle": "-", "color": colour, "linewidth": 2 * POINTS_PER_PIXEL,
                                           "shrinkA": 4, "shrinkB": 6})


def create_figure(column_count: int) -> Tuple[matplotlib.figure.Figure, List[matplotlib.axes.Axes]]:
    """
    Creates a chart of the size of the viewport of a scene with its panels side by side, on the card colour.

    Args:
        column_count (int): The number of panels.

    Returns:
        Tuple[matplotlib.figure.Figure, List[matplotlib.axes.Axes]]: The figure and its panels.
    """
    register_fonts()
    figure, panels = pyplot.subplots(nrows=1, ncols=column_count, figsize=(CHART_WIDTH / DOTS_PER_INCH,
                                                                          CHART_HEIGHT / DOTS_PER_INCH),
                                     dpi=DOTS_PER_INCH, squeeze=False)
    figure.patch.set_facecolor(SURFACE_COLOUR)
    figure.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.12, wspace=0.16)
    return figure, list(panels[0])


def add_header(figure: matplotlib.figure.Figure, description: str, legend_labels: List[str],
               legend_colours: List[str]) -> None:
    """
    Writes the header of a chart: the description of the launches on the left and the legend of the settings on the
    right, both kept in the base image of the scene.

    Args:
        figure (matplotlib.figure.Figure): The figure.
        description (str): The line describing the launches drawn.
        legend_labels (List[str]): The name of every setting.
        legend_colours (List[str]): The colour of every setting.
    """
    figure.text(x=0.012, y=0.905, s=description, color=TEXT_SECONDARY_COLOUR, fontsize=18 * POINTS_PER_PIXEL,
                va="center")
    legend_handles = [Line2D(xdata=[0], ydata=[0], color=colour, linewidth=3 * POINTS_PER_PIXEL, marker="o",
                             markersize=9 * POINTS_PER_PIXEL) for colour in legend_colours]
    figure.legend(handles=legend_handles, labels=legend_labels, loc="center right", bbox_to_anchor=(0.985, 0.905),
                  ncols=len(legend_labels), frameon=False, labelcolor=TEXT_PRIMARY_COLOUR,
                  fontsize=18 * POINTS_PER_PIXEL, handlelength=1.6)


def measure_panels(figure: matplotlib.figure.Figure, chart_name: str,
                   panels: Dict[str, matplotlib.axes.Axes]) -> Dict[str, List[int]]:
    """
    Gives the highlight box and the reveal region of every panel of a drawn chart, the facts the image scene of the
    chart uses to outline a panel and uncover its curves.

    Args:
        figure (matplotlib.figure.Figure): The drawn figure.
        chart_name (str): The name of the chart in the facts, e.g. "model_settings".
        panels (Dict[str, matplotlib.axes.Axes]): The panels, keyed by their name in the facts.

    Returns:
        Dict[str, List[int]]: The boxes, named benchmark.<chart>.<panel>_panel_box and _reveal_region.
    """
    boxes = {}
    for panel_name, panel_axes in panels.items():
        boxes[f"benchmark.{chart_name}.{panel_name}_panel_box"] = compute_pixel_box(figure=figure, axes=panel_axes,
                                                                                   with_labels=True, margin=0)
        boxes[f"benchmark.{chart_name}.{panel_name}_reveal_region"] = compute_pixel_box(
            figure=figure, axes=panel_axes, with_labels=False, margin=REVEAL_MARGIN)
    return boxes


def name_curve_values(fact_prefix: str, curve: numpy.ndarray) -> Dict[str, float]:
    """
    Turns the mean curve of a setting into one fact per evaluated iteration, the iterations a setting did not reach
    (NaN) left out.

    Args:
        fact_prefix (str): The name of the setting in the facts, e.g. "benchmark.model_settings.residual_on".
        curve (numpy.ndarray): The mean test accuracy at every evaluated iteration.

    Returns:
        Dict[str, float]: The facts, named <prefix>.accuracy_<iteration>.
    """
    return {f"{fact_prefix}.accuracy_{iteration}": round_fact(value=value)
            for iteration, value in zip(EVALUATED_ITERATIONS, curve) if not numpy.isnan(value)}


def draw_model_settings_chart(output_folder: Path, launches: Dict[Tuple[str, str], List[float]]) -> Dict[str, Any]:
    """
    Draws the model chart: the residual shortcuts on and off (stage 6) and the batch and layer normalizations of the
    feature maps (stage 4, the layer launches stopped at iteration 500), against the launches kept by the earlier
    stages.

    Args:
        output_folder (Path): The benchmark folder of the series.
        launches (Dict[Tuple[str, str], List[float]]): The test accuracies of every launch.

    Returns:
        Dict[str, Any]: The boxes of the panels and the mean curves of the settings.
    """
    figure, (residual_axes, normalization_axes) = create_figure(column_count=2)
    add_header(figure=figure, description="Synthetic shapes, 2 launches per setting (faint) and their mean",
               legend_labels=["kept setting", "changed setting"], legend_colours=SERIES_COLOURS[:2])
    style_panel(axes=residual_axes, title="Residual shortcuts", y_limit=(0.0, 1.0), x_limit=(0, 2000))
    style_panel(axes=normalization_axes, title="Feature map normalization", y_limit=(0.0, 1.0), x_limit=(0, 2000))
    for axes in [residual_axes, normalization_axes]:
        draw_single_class_line(axes=axes, x_limit=(0, 2000))
    save_figure(figure=figure, output_path=output_folder / "benchmark_model_settings_axes.png")

    # Residual shortcuts on (kept) and off, both launches of each
    kept_launches = select_launches(launches=launches, launch_names=KEPT_LAUNCHES, evaluated_weights="")
    kept_mean = draw_setting(axes=residual_axes, launch_curves=kept_launches, colour=SERIES_COLOURS[0],
                             clip_to_panel=False)
    off_mean = draw_setting(axes=residual_axes, launch_curves=select_launches(
        launches=launches, launch_names=RESIDUAL_OFF_LAUNCHES, evaluated_weights=""), colour=SERIES_COLOURS[1],
                            clip_to_panel=False)
    label_point(axes=residual_axes, iteration=1750, value=kept_mean[6], text="residual_nets: True",
                offset=(-20, -50), colour=SERIES_COLOURS[0])
    label_point(axes=residual_axes, iteration=1750, value=off_mean[6], text="residual_nets: False",
                offset=(-20, 60), colour=SERIES_COLOURS[1])

    # Feature map normalizations: batch (kept) and layer, stopped at iteration 500
    kept_mean = draw_setting(axes=normalization_axes, launch_curves=kept_launches, colour=SERIES_COLOURS[0],
                             clip_to_panel=False)
    layer_mean = draw_setting(axes=normalization_axes, launch_curves=select_launches(
        launches=launches, launch_names=LAYER_NORMALIZATION_LAUNCHES, evaluated_weights=""),
                              colour=SERIES_COLOURS[1], clip_to_panel=False)
    label_point(axes=normalization_axes, iteration=1750, value=kept_mean[6], text="batch", offset=(-20, -50),
                colour=SERIES_COLOURS[0])
    label_point(axes=normalization_axes, iteration=500, value=layer_mean[1], text="layer, stopped at 500",
                offset=(260, 60), colour=SERIES_COLOURS[1])
    save_figure(figure=figure, output_path=output_folder / "benchmark_model_settings.png")

    # Boxes of the panels and the mean curve of every setting, the kept one being the batch normalization too
    facts = measure_panels(figure=figure, chart_name="model_settings",
                           panels={"residual": residual_axes, "normalization": normalization_axes})
    pyplot.close(fig=figure)
    facts.update(name_curve_values(fact_prefix="benchmark.model_settings.residual_on", curve=kept_mean))
    facts.update(name_curve_values(fact_prefix="benchmark.model_settings.residual_off", curve=off_mean))
    facts.update(name_curve_values(fact_prefix="benchmark.model_settings.layer_normalization", curve=layer_mean))
    return facts


def draw_learning_rate_chart(output_folder: Path, launches: Dict[Tuple[str, str], List[float]]) -> Dict[str, Any]:
    """
    Draws the learning rate chart of stage 1: the test accuracy of the five learning rates, two launches each.

    Args:
        output_folder (Path): The benchmark folder of the series.
        launches (Dict[Tuple[str, str], List[float]]): The test accuracies of every launch.

    Returns:
        Dict[str, Any]: The boxes of the panel and the mean curve of every learning rate.
    """
    figure, (axes,) = create_figure(column_count=1)
    add_header(figure=figure, description="Synthetic shapes, 2 launches per learning rate (faint) and their mean",
               legend_labels=list(LEARNING_RATE_LAUNCHES), legend_colours=SERIES_COLOURS)
    style_panel(axes=axes, title="Learning rate", y_limit=(0.0, 1.0), x_limit=(0, 2000))
    draw_single_class_line(axes=axes, x_limit=(0, 2000))
    save_figure(figure=figure, output_path=output_folder / "benchmark_learning_rate_axes.png")

    # Every learning rate, its launches and their mean, named in the facts as 1e_4 for 1e-4
    mean_curves = {}
    for colour, (learning_rate, launch_names) in zip(SERIES_COLOURS, LEARNING_RATE_LAUNCHES.items()):
        mean_curves[learning_rate] = draw_setting(axes=axes, launch_curves=select_launches(
            launches=launches, launch_names=launch_names, evaluated_weights=""), colour=colour, clip_to_panel=False)
    save_figure(figure=figure, output_path=output_folder / "benchmark_learning_rate.png")
    facts = measure_panels(figure=figure, chart_name="learning_rate", panels={"learning_rate": axes})
    pyplot.close(fig=figure)
    for learning_rate, mean_curve in mean_curves.items():
        facts.update(name_curve_values(fact_prefix=f"benchmark.learning_rate.learning_rate_"
                                                   f"{learning_rate.replace('-', '_')}", curve=mean_curve))
    return facts


def draw_moving_average_chart(output_folder: Path, launches: Dict[Tuple[str, str], List[float]]) -> Dict[str, Any]:
    """
    Draws the moving average chart of stage 13: the test accuracy of the trained weights and of the averaged weights
    of the same two launches, over the whole training and over its last 1,000 iterations.

    Args:
        output_folder (Path): The benchmark folder of the series.
        launches (Dict[Tuple[str, str], List[float]]): The test accuracies of every launch.

    Returns:
        Dict[str, Any]: The boxes of the panels and the mean curves of both weights.
    """
    figure, (whole_axes, zoom_axes) = create_figure(column_count=2)
    add_header(figure=figure, description="Synthetic shapes, the same 2 launches (faint) and their mean",
               legend_labels=["trained weights", "averaged weights"], legend_colours=SERIES_COLOURS[:2])
    style_panel(axes=whole_axes, title="Whole training", y_limit=(0.4, 1.0), x_limit=(0, 2000))
    style_panel(axes=zoom_axes, title="Last 1,000 iterations", y_limit=(0.93, 1.0), x_limit=(1000, 2000))
    save_figure(figure=figure, output_path=output_folder / "benchmark_moving_average_axes.png")

    # Trained and averaged weights of both launches on both panels, the zoom labelled at its last evaluation
    trained_launches = select_launches(launches=launches, launch_names=MOVING_AVERAGE_LAUNCHES,
                                       evaluated_weights="trained")
    averaged_launches = select_launches(launches=launches, launch_names=MOVING_AVERAGE_LAUNCHES,
                                        evaluated_weights="ema")
    for axes in [whole_axes, zoom_axes]:
        trained_mean = draw_setting(axes=axes, launch_curves=trained_launches, colour=SERIES_COLOURS[0],
                                    clip_to_panel=axes is zoom_axes)
        averaged_mean = draw_setting(axes=axes, launch_curves=averaged_launches, colour=SERIES_COLOURS[1],
                                     clip_to_panel=axes is zoom_axes)
    label_point(axes=zoom_axes, iteration=2000, value=trained_mean[-1], text=f"{trained_mean[-1]:.3f}",
                offset=(-30, -40), colour=SERIES_COLOURS[0])
    label_point(axes=zoom_axes, iteration=2000, value=averaged_mean[-1], text=f"{averaged_mean[-1]:.3f}",
                offset=(-30, 40), colour=SERIES_COLOURS[1])
    save_figure(figure=figure, output_path=output_folder / "benchmark_moving_average.png")
    facts = measure_panels(figure=figure, chart_name="moving_average", panels={"whole": whole_axes, "zoom": zoom_axes})
    pyplot.close(fig=figure)
    facts.update(name_curve_values(fact_prefix="benchmark.moving_average.trained_weights", curve=trained_mean))
    facts.update(name_curve_values(fact_prefix="benchmark.moving_average.averaged_weights", curve=averaged_mean))
    return facts


def read_launch_means(results_path: Path, launch_names: List[str]) -> Tuple[List[float], List[float]]:
    """
    Reads the two mean columns of launches of the benchmark: the mean test accuracy of the eight evaluations and the
    mean of the last four, the second leaving out how fast a launch starts.

    Args:
        results_path (Path): The results.csv of the benchmark.
        launch_names (List[str]): The launches read, evaluating their trained weights only.

    Returns:
        Tuple[List[float], List[float]]: The means of the eight evaluations and of the last four, in launch order.

    Raises:
        ValueError: If a launch is not in results.csv.
    """
    rows = {}
    with results_path.open(encoding="utf-8", newline="") as results_file:
        for row in csv.DictReader(results_file):
            if row["launch"] in launch_names and row["evaluated_weights"] == "":
                rows[row["launch"]] = row
    missing_launches = [name for name in launch_names if name not in rows]
    if missing_launches:
        raise ValueError(f"Launches missing from {results_path}: {missing_launches}")
    return ([float(rows[name]["accuracy_mean"]) for name in launch_names],
            [float(rows[name]["accuracy_last_4_mean"]) for name in launch_names])


def draw_launch_rows(axes: matplotlib.axes.Axes, settings: Dict[str, List[float]]) -> None:
    """
    Draws the launches of every setting as one row of dots, with the mean of the setting as a vertical tick, the dot
    plot comparing launches of the same settings.

    Args:
        axes (matplotlib.axes.Axes): The panel.
        settings (Dict[str, List[float]]): The value of every launch, by setting, the first setting on the top row.
    """
    for row_index, (colour, values) in enumerate(zip(SERIES_COLOURS, settings.values())):
        row = len(settings) - 1 - row_index
        axes.scatter(values, [row] * len(values), s=(16 * POINTS_PER_PIXEL) ** 2, color=colour, alpha=0.85,
                     edgecolors=SURFACE_COLOUR, linewidths=2 * POINTS_PER_PIXEL, zorder=3)
        axes.plot([numpy.mean(values)] * 2, [row - 0.28, row + 0.28], color=TEXT_PRIMARY_COLOUR,
                  linewidth=3 * POINTS_PER_PIXEL, zorder=4)


def style_launch_panel(axes: matplotlib.axes.Axes, title: str, x_limit: Tuple[float, float],
                       setting_names: List[str]) -> None:
    """
    Gives a dot plot panel the anatomy of the other charts: vertical hairline grid, the settings as row labels, the
    test accuracy along the horizontal axis.

    Args:
        axes (matplotlib.axes.Axes): The panel.
        title (str): The title of the panel.
        x_limit (Tuple[float, float]): The lowest and highest test accuracy shown.
        setting_names (List[str]): The name of every row, the first one on the top row.
    """
    # Surface, vertical hairline grid and the bottom axis only
    axes.set_facecolor(SURFACE_COLOUR)
    axes.grid(visible=True, axis="x", color=GRID_COLOUR, linewidth=POINTS_PER_PIXEL, linestyle="-")
    axes.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID_COLOUR)
    axes.spines["bottom"].set_linewidth(POINTS_PER_PIXEL)

    # Limits, the settings as row labels, the axis label and the title
    axes.set_xlim(left=x_limit[0], right=x_limit[1])
    axes.set_ylim(bottom=-0.7, top=len(setting_names) - 0.3)
    axes.set_yticks(ticks=list(range(len(setting_names))), labels=list(reversed(setting_names)))
    axes.tick_params(axis="both", colors=TEXT_SECONDARY_COLOUR, labelsize=18 * POINTS_PER_PIXEL, length=0, pad=8)
    axes.set_xlabel(xlabel="Mean test accuracy", color=TEXT_SECONDARY_COLOUR, fontsize=19 * POINTS_PER_PIXEL,
                    labelpad=8)
    axes.set_title(label=title, color=TEXT_PRIMARY_COLOUR, fontsize=24 * POINTS_PER_PIXEL, loc="left", pad=14)


def summarize_launch_means(setting_name: str, panel_name: str, launch_names: List[str],
                           launch_means: List[float]) -> Dict[str, float]:
    """
    Gives the facts of one row of the launch noise chart: the value of every launch, and the mean (the tick of the
    row), the lowest, the highest and the range of the values of the launches of the setting.

    Args:
        setting_name (str): The setting in the facts, "squeeze_excitation_off" or "squeeze_excitation_on".
        panel_name (str): The panel in the facts, "whole" for the mean of the eight evaluations, "last" for the mean
            of the last four.
        launch_names (List[str]): The launches of the setting.
        launch_means (List[float]): Their values on the panel, in launch order.

    Returns:
        Dict[str, float]: The facts, named benchmark.launch_noise.<setting>.<launch>.<panel>_mean and
            benchmark.launch_noise.<setting>_<panel>_mean_average, _minimum, _maximum and _range.
    """
    fact_prefix = f"benchmark.launch_noise.{setting_name}"
    facts = {f"{fact_prefix}.{launch_name}.{panel_name}_mean": round_fact(value=launch_mean)
             for launch_name, launch_mean in zip(launch_names, launch_means)}
    facts.update({f"{fact_prefix}_{panel_name}_mean_average": round_fact(value=numpy.mean(launch_means)),
                  f"{fact_prefix}_{panel_name}_mean_minimum": round_fact(value=min(launch_means)),
                  f"{fact_prefix}_{panel_name}_mean_maximum": round_fact(value=max(launch_means)),
                  f"{fact_prefix}_{panel_name}_mean_range": round_fact(value=max(launch_means) - min(launch_means))})
    return facts


def draw_launch_noise_chart(output_folder: Path, results_path: Path) -> Dict[str, Any]:
    """
    Draws the spread between launches of the same settings, from the repeat stage of the benchmark: the six launches
    with squeeze and excitation off and the six with it on, by the mean of their eight evaluations and by the mean of
    their last four, every dot a launch and every tick the mean of a setting.

    Args:
        output_folder (Path): The benchmark folder of the series.
        results_path (Path): The results.csv of the benchmark.

    Returns:
        Dict[str, Any]: The boxes of the panels, the value of every launch and the means and ranges of the settings.
    """
    whole_means, last_means = {}, {}
    for setting, names in LAUNCH_NOISE_LAUNCHES.items():
        whole_means[setting], last_means[setting] = read_launch_means(results_path=results_path, launch_names=names)
    figure, (whole_axes, last_axes) = create_figure(column_count=2)
    figure.subplots_adjust(left=0.15, right=0.985, top=0.80, bottom=0.14, wspace=0.42)
    add_header(figure=figure, description="Synthetic shapes, the same settings launched 6 times · tick: mean",
               legend_labels=list(LAUNCH_NOISE_LAUNCHES), legend_colours=SERIES_COLOURS[:2])
    style_launch_panel(axes=whole_axes, title="Mean of the 8 evaluations", x_limit=(0.84, 0.94),
                       setting_names=["off", "on"])
    style_launch_panel(axes=last_axes, title="Mean of the last 4 evaluations", x_limit=(0.95, 1.0),
                       setting_names=["off", "on"])
    save_figure(figure=figure, output_path=output_folder / "benchmark_launch_noise_axes.png")

    # Every launch of both settings on both panels
    draw_launch_rows(axes=whole_axes, settings=whole_means)
    draw_launch_rows(axes=last_axes, settings=last_means)
    save_figure(figure=figure, output_path=output_folder / "benchmark_launch_noise.png")
    facts = measure_panels(figure=figure, chart_name="launch_noise", panels={"whole": whole_axes, "last": last_axes})
    pyplot.close(fig=figure)

    # Value of every launch and the means and ranges of every setting, squeeze-and-excitation off named
    # squeeze_excitation_off in the facts
    for setting, launch_names in LAUNCH_NOISE_LAUNCHES.items():
        setting_name = setting.replace("squeeze-and-excitation", "squeeze_excitation").replace(" ", "_")
        facts.update(summarize_launch_means(setting_name=setting_name, panel_name="whole", launch_names=launch_names,
                                            launch_means=whole_means[setting]))
        facts.update(summarize_launch_means(setting_name=setting_name, panel_name="last", launch_names=launch_names,
                                            launch_means=last_means[setting]))
    return facts


def make_benchmark_figures(output_folder: Path, results_path: Path) -> Dict[str, Any]:
    """
    Draws the four benchmark charts into the benchmark folder of the series, from the launches of results.csv, and
    gives their facts.

    Args:
        output_folder (Path): The figures folder of the series.
        results_path (Path): The results.csv of the benchmark, the benchmark_results file of tutorial_sources.yaml.

    Returns:
        Dict[str, Any]: The facts of the charts, named benchmark.*.
    """
    benchmark_folder = output_folder / "benchmark"
    launches = read_launches(results_path=results_path)
    facts = draw_model_settings_chart(output_folder=benchmark_folder, launches=launches)
    facts.update(draw_learning_rate_chart(output_folder=benchmark_folder, launches=launches))
    facts.update(draw_moving_average_chart(output_folder=benchmark_folder, launches=launches))
    facts.update(draw_launch_noise_chart(output_folder=benchmark_folder, results_path=results_path))
    return facts
