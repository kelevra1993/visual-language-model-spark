"""
Figures of the training and evaluation logs of a detection run, shared by the object detection series (DET run) and
the instance segmentation series (INS run, imported by tutorials/figures/instance_segmentation/figures.py): the sample
images of a test image at several evaluated iterations, the ground truth panel followed by the prediction panel of
every iteration, cropped from the Samples/ images of the evaluation folders; the losses of the Train folder of the
TensorBoard logs, every curve smoothed like TensorBoard over its raw values; the headline metrics of every evaluated
checkpoint with the trained weights (Test) against their moving average (Test_EMA); and the console summary of an
iteration read from the console log of the run. Every figure gives the boxes of its panels and the values the
narration quotes as facts, the headline metrics written on the figures as the percentages of the reports of the
repository ("72.69 %", ticks "20 %") and kept between 0 and 1 in the facts.
"""

import re

import numpy

from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter, MultipleLocator
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Any, Dict, List, Tuple

from tutorials.figures.figure_utilities import (DOTS_PER_INCH, POINTS_PER_PIXEL, SERIES_COLOURS,
                                                TEXT_SECONDARY_COLOUR, compute_pixel_box,
                                                format_thousands, register_fonts, round_fact, save_figure,
                                                style_chart_panel)
from tutorials.figures.object_detection.detection_sources import (read_scalar_series, strip_terminal_codes,
                                                                  write_listing)
from tutorial_builder.layout import (CARD_COLOUR, DIM_TEXT_COLOUR, MONOSPACE_FONT_PATH, SANS_BOLD_FONT_PATH,
                                     TEXT_COLOUR)
from vision_training.evaluation.report_formatting import format_percentage, format_percentage_tick

# Regions of a sample image of the detection evaluations (1028 x 560 pixels): the caption band above the panels, the
# ground truth panel, the 4 pixel separator and the prediction panel, each 512 pixels, and the legend band below them
SAMPLE_CAPTION_HEIGHT = 24
SAMPLE_PANEL_SIDE = 512
GROUND_TRUTH_PANEL_LEFT = 0
PREDICTION_PANEL_LEFT = 516

# Layout of the figure of the sample images: the gap between two panels and the bands of the headings and notes
SAMPLE_PANEL_GAP = 28
SAMPLE_HEADING_HEIGHT = 72
SAMPLE_NOTE_HEIGHT = 64

# Size of the charts: the card of an image scene
CHART_WIDTH = 1720
CHART_HEIGHT = 700

# Smoothing weight of the loss curves, the exponential moving average of TensorBoard
SMOOTHING_WEIGHT = 0.98

# Bottom of the metric axes, below zero so that the percentage written under a point near zero stays above the ticks
# of the iterations
METRIC_AXIS_BOTTOM = -0.08


def draw_sample_iterations_figure(run_folder: Path, image_name: str, iterations: List[int], metric_tag: str,
                                  metric_label: str, output_path: Path, fact_prefix: str) -> Dict[str, Any]:
    """
    Draws the ground truth panel of a sample image followed by its prediction panel at every evaluated iteration, each
    under its heading and above the headline metric of its evaluation, the panels cropped from the Samples/ images the
    evaluation wrote, never redrawn, so that the scene shows what the run shows in TensorBoard, the metric written as
    the percentage of the reports.

    Args:
        run_folder (Path): The run folder.
        image_name (str): The file name of the sample image, e.g. "test_014285.png".
        iterations (List[int]): The evaluated iterations, in increasing order.
        metric_tag (str): The headline scalar of the Test folder quoted under every panel, e.g. "mAP@0.5/area".
        metric_label (str): Its name written under the panels.
        output_path (Path): The PNG file of the figure.
        fact_prefix (str): The prefix of the names of the facts.

    Returns:
        Dict[str, Any]: The box of the figure and of every panel, and the metric of every iteration.

    Raises:
        FileNotFoundError: If a sample image is missing.
    """
    sample_paths = [run_folder / "Evaluation" / f"Iteration_{iteration}" / "Samples" / image_name
                    for iteration in iterations]
    missing_paths = [str(path) for path in sample_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing sample images of the run: {missing_paths}")
    metric_iterations, metric_values = read_scalar_series(log_folder=run_folder / "Tensorboard" / "Test",
                                                          tag=metric_tag)
    metric_by_iteration = {int(iteration): float(value) for iteration, value in zip(metric_iterations, metric_values)}

    # Panels: the ground truth of the last evaluation, then the prediction of every iteration
    panel_count = len(iterations) + 1
    canvas = Image.new(mode="RGB", size=(panel_count * SAMPLE_PANEL_SIDE + (panel_count - 1) * SAMPLE_PANEL_GAP,
                                         SAMPLE_HEADING_HEIGHT + SAMPLE_PANEL_SIDE + SAMPLE_NOTE_HEIGHT),
                       color=CARD_COLOUR)
    draw = ImageDraw.Draw(im=canvas)
    heading_font = ImageFont.truetype(font=str(SANS_BOLD_FONT_PATH), size=38)
    note_font = ImageFont.truetype(font=str(MONOSPACE_FONT_PATH), size=28)
    panels = [("Ground truth", "", sample_paths[-1], GROUND_TRUTH_PANEL_LEFT)]
    panels += [(f"Iteration {iteration:,}", f"{metric_label} {format_percentage(value=metric_by_iteration[iteration])}",
                sample_path, PREDICTION_PANEL_LEFT) for iteration, sample_path in zip(iterations, sample_paths)]
    facts = {}
    for panel_index, (heading, note, sample_path, crop_left) in enumerate(panels):
        left = panel_index * (SAMPLE_PANEL_SIDE + SAMPLE_PANEL_GAP)
        with Image.open(fp=sample_path) as sample_image:
            panel = sample_image.convert(mode="RGB").crop(box=(crop_left, SAMPLE_CAPTION_HEIGHT,
                                                               crop_left + SAMPLE_PANEL_SIDE,
                                                               SAMPLE_CAPTION_HEIGHT + SAMPLE_PANEL_SIDE))
        canvas.paste(im=panel, box=(left, SAMPLE_HEADING_HEIGHT))
        draw.text(xy=(left, SAMPLE_HEADING_HEIGHT // 2), text=heading, fill=TEXT_COLOUR, font=heading_font,
                  anchor="lm")
        draw.text(xy=(left, SAMPLE_HEADING_HEIGHT + SAMPLE_PANEL_SIDE + SAMPLE_NOTE_HEIGHT // 2), text=note,
                  fill=DIM_TEXT_COLOUR, font=note_font, anchor="lm")
        facts[f"{fact_prefix}.panel_{panel_index + 1}_box"] = [left, 0, left + SAMPLE_PANEL_SIDE, canvas.height]
    for iteration in iterations:
        facts[f"{fact_prefix}.metric_{iteration}"] = round_fact(value=metric_by_iteration[iteration])
    facts[f"{fact_prefix}.figure_box"] = [0, 0, canvas.width, canvas.height]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(fp=output_path)
    return facts


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


def draw_loss_curves_figure(run_folder: Path, panels: List[Tuple[str, List[Tuple[str, str]]]], output_path: Path,
                            fact_prefix: str) -> Dict[str, Any]:
    """
    Draws the training losses of a run from its Train TensorBoard folder, one panel per group of losses, every loss
    as its raw values drawn faintly under its smoothed curve, its name and last smoothed value at the end of the curve,
    and gives the first and last values of every loss and the boxes of the panels.

    Args:
        run_folder (Path): The run folder.
        panels (List[Tuple[str, List[Tuple[str, str]]]]): The title of every panel and its losses, each its scalar tag
            and the short name of its fact.
        output_path (Path): The PNG file of the figure.
        fact_prefix (str): The prefix of the names of the facts.

    Returns:
        Dict[str, Any]: The facts of the figure: the first, last and last smoothed value of every loss, the number of
            iterations and the box of every panel.
    """
    register_fonts()
    figure, axes_list = pyplot.subplots(nrows=1, ncols=len(panels), figsize=(CHART_WIDTH / DOTS_PER_INCH,
                                                                             CHART_HEIGHT / DOTS_PER_INCH))
    figure.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.12, wspace=0.16)
    axes_list = numpy.atleast_1d(axes_list)
    facts = {}
    for panel_index, ((title, losses), axes) in enumerate(zip(panels, axes_list)):
        maximum_value = 0.0
        for loss_index, (tag, short_name) in enumerate(losses):
            iterations, values = read_scalar_series(log_folder=run_folder / "Tensorboard" / "Train", tag=tag)
            smoothed_values = smooth_values(values=values, weight=SMOOTHING_WEIGHT)
            colour = SERIES_COLOURS[loss_index % len(SERIES_COLOURS)]
            axes.plot(iterations, values, color=colour, alpha=0.2, linewidth=1.0 * POINTS_PER_PIXEL)
            axes.plot(iterations, smoothed_values, color=colour, linewidth=3 * POINTS_PER_PIXEL,
                      solid_capstyle="round", label=f"{tag.replace(' Loss', '')} · {smoothed_values[-1]:.3f}")
            maximum_value = max(maximum_value, float(numpy.percentile(a=values, q=99.5)))
            facts.update({f"{fact_prefix}.{short_name}.first": round_fact(value=values[0]),
                          f"{fact_prefix}.{short_name}.last": round_fact(value=values[-1]),
                          f"{fact_prefix}.{short_name}.last_smoothed": round_fact(value=smoothed_values[-1])})
            facts[f"{fact_prefix}.iterations"] = int(iterations[-1])

        # Panel with the iterations of the run on its horizontal axis
        style_chart_panel(axes=axes, title=title, x_label="Iteration", y_label="Loss" if panel_index == 0 else None)
        axes.set_xlim(left=0, right=facts[f"{fact_prefix}.iterations"])
        axes.set_ylim(bottom=0, top=maximum_value * 1.05)
        axes.xaxis.set_major_formatter(formatter=FuncFormatter(func=format_thousands))
        axes.legend(loc="upper right", frameon=False, labelcolor=TEXT_SECONDARY_COLOUR,
                    fontsize=17 * POINTS_PER_PIXEL)
    figure.canvas.draw()
    for panel_index, axes in enumerate(axes_list):
        facts[f"{fact_prefix}.panel_{panel_index + 1}_box"] = compute_pixel_box(figure=figure, axes=axes,
                                                                                 with_labels=True, margin=6)
    save_figure(figure=figure, output_path=output_path)
    pyplot.close(fig=figure)
    return facts


def draw_metric_iterations_figure(run_folder: Path, panels: List[Tuple[str, str, str]], output_path: Path,
                                  fact_prefix: str) -> Dict[str, Any]:
    """
    Draws the headline metrics of every evaluated checkpoint of a run, the trained weights (Test folder) against
    their moving average (Test_EMA folder), one panel per metric, every point labelled with its value as the
    percentage of the reports on an axis of percentages, and gives the values, between 0 and 1, and the boxes of the
    panels.

    Args:
        run_folder (Path): The run folder.
        panels (List[Tuple[str, str, str]]): The scalar tag of every panel, its title and the short name of its facts.
        output_path (Path): The PNG file of the figure.
        fact_prefix (str): The prefix of the names of the facts.

    Returns:
        Dict[str, Any]: The value of every metric at every evaluated iteration for both weights, and the panel boxes.
    """
    register_fonts()
    figure, axes_list = pyplot.subplots(nrows=1, ncols=len(panels), figsize=(CHART_WIDTH / DOTS_PER_INCH,
                                                                             CHART_HEIGHT / DOTS_PER_INCH))
    figure.subplots_adjust(left=0.06, right=0.97, top=0.86, bottom=0.12, wspace=0.18)
    facts = {}
    for panel_index, ((tag, title, short_name), axes) in enumerate(zip(panels, numpy.atleast_1d(axes_list))):
        series = [read_scalar_series(log_folder=run_folder / "Tensorboard" / folder, tag=tag)
                  for folder in ["Test", "Test_EMA"]]
        for series_index, (folder, label) in enumerate([("Test", "trained weights"),
                                                         ("Test_EMA", "moving average")]):
            iterations, values = series[series_index]
            other_values = series[1 - series_index][1]
            colour = SERIES_COLOURS[series_index]
            axes.plot(iterations, values, color=colour, linewidth=3 * POINTS_PER_PIXEL, marker="o",
                      markersize=9 * POINTS_PER_PIXEL, label=label)
            for point_index, (iteration, value) in enumerate(zip(iterations, values)):
                facts[f"{fact_prefix}.{short_name}.{folder.lower()}_{int(iteration)}"] = round_fact(value=value)

                # Value of every point as a percentage, above the point of the higher series and below the point of
                # the lower one
                vertical_offset = 12 if value >= other_values[point_index] else -24
                axes.annotate(text=format_percentage(value=value), xy=(iteration, value), xytext=(0, vertical_offset),
                              textcoords="offset points", ha="center", color=colour, fontsize=16 * POINTS_PER_PIXEL)
        style_chart_panel(axes=axes, title=title, x_label="Iteration", y_label="mAP" if panel_index == 0 else None)
        # Limits, the bottom a little below zero so that the percentage under the first point clears the iteration
        # ticks, the iterations with thousands separators and the metric as percentages every twenty points
        axes.set_ylim(bottom=METRIC_AXIS_BOTTOM, top=1.0)
        axes.set_xlim(left=0, right=float(iterations[-1]) * 1.06)
        axes.xaxis.set_major_formatter(formatter=FuncFormatter(func=format_thousands))
        axes.yaxis.set_major_locator(locator=MultipleLocator(base=0.2))
        axes.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))
        axes.legend(loc="upper left", frameon=False, labelcolor=TEXT_SECONDARY_COLOUR, fontsize=17 * POINTS_PER_PIXEL)
    figure.canvas.draw()
    for panel_index, axes in enumerate(numpy.atleast_1d(axes_list)):
        facts[f"{fact_prefix}.panel_{panel_index + 1}_box"] = compute_pixel_box(figure=figure, axes=axes,
                                                                                 with_labels=True, margin=6)
    save_figure(figure=figure, output_path=output_path)
    pyplot.close(fig=figure)
    return facts


def write_console_summary_listing(log_path: Path, iteration: int, output_path: Path) -> List[str]:
    """
    Writes the console summary of one iteration of a run, read from its console log, as a listing: from the line of
    the iteration to the duration of the interval, the colours and links of the terminal removed.

    Args:
        log_path (Path): The console log of the run.
        iteration (int): The iteration of the summary.
        output_path (Path): The text file of the listing.

    Returns:
        List[str]: The lines of the summary.

    Raises:
        ValueError: If the log has no summary of the iteration.
    """
    lines = [strip_terminal_codes(text=line).rstrip() for line in log_path.read_text(encoding="utf-8").splitlines()]
    starts = [line_index for line_index, line in enumerate(lines) if f"Iteration: {iteration} " in line]
    if not starts:
        raise ValueError(f"No console summary of iteration {iteration} in {log_path}.")
    summary_lines = []
    for line in lines[starts[0]:]:
        summary_lines.append(line)
        if line.startswith("Interval Of "):
            break
    write_listing(lines=summary_lines, output_path=output_path)
    return summary_lines


def read_console_summary_facts(summary_lines: List[str], fact_prefix: str) -> Dict[str, Any]:
    """
    Reads the values of a console summary the narration quotes: the throughput of the interval, its duration, and
    the moving average of every training loss with its change when the summary shows one.

    Args:
        summary_lines (List[str]): The lines of the summary (write_console_summary_listing).
        fact_prefix (str): The prefix of the names of the facts.

    Returns:
        Dict[str, Any]: The values, named <prefix>.throughput, <prefix>.duration and <prefix>.<loss> (snake case).
    """
    facts = {}
    for line in summary_lines:
        throughput = re.search(pattern=r"Throughput: ([\d.]+) Images Per Second", string=line)
        if throughput is not None:
            facts[f"{fact_prefix}.throughput"] = float(throughput.group(1))
        duration = re.search(pattern=r"Took (\d\d):(\d\d):(\d\d)", string=line)
        if duration is not None:
            facts[f"{fact_prefix}.minutes"] = int(duration.group(1)) * 60 + int(duration.group(2))
        loss = re.match(pattern=r"Moving Average of Train (.+?)\s+:\s+([\d.]+)", string=line)
        if loss is not None:
            facts[f"{fact_prefix}.{loss.group(1).lower().replace(' ', '_')}"] = float(loss.group(2))
    return facts
