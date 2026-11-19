"""
Reading and drawing of the TensorBoard scalars of a real run for the training curve figures of the task series of the
tutorial videos (keypoint detection and semantic segmentation): the value of a scalar at every iteration read with
tensorboard's EventAccumulator (the last written one when a relaunched run logged an iteration twice), the TensorBoard
smoothing of a curve, and the drawing of one curve as its faint raw values under its smoothed line, in the anatomy of
the charts of figure_utilities.py.
"""

import matplotlib
import numpy

from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Dict, List, Tuple

from tutorials.figures.figure_utilities import POINTS_PER_PIXEL, round_fact

# Smoothing weight of the curves, the exponential moving average of TensorBoard
SMOOTHING_WEIGHT = 0.95


def read_scalars(log_folder: Path, tag: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Reads every value of one scalar of a TensorBoard log folder of a run, one value per iteration in the order of the
    iterations: when several event files log the same iteration (a launch stopped early and started again from the
    same iteration, which writes a new file), the value written last is kept, so that the curve is the one of the
    launch that went on.

    Args:
        log_folder (Path): The Train, Validation, Test or Test_EMA folder of the run.
        tag (str): The scalar, e.g. "Total Loss".

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
        raise KeyError(f"No scalar {tag!r} in {log_folder}, whose scalars are {accumulator.Tags()['scalars'][:20]}.")

    # The last written value of every iteration, by wall time, then the iterations in order
    latest_events = {}
    for event in accumulator.Scalars(tag):
        if event.step not in latest_events or event.wall_time >= latest_events[event.step].wall_time:
            latest_events[event.step] = event
    events = [latest_events[step] for step in sorted(latest_events)]
    return numpy.array([event.step for event in events]), numpy.array([event.value for event in events])


def read_scalar_values(log_folder: Path, tags: List[str]) -> Dict[str, Dict[int, float]]:
    """
    Reads several scalars of a TensorBoard log folder as dictionaries of their value at every iteration, for the
    metrics of the evaluated checkpoints.

    Args:
        log_folder (Path): The log folder.
        tags (List[str]): The scalars.

    Returns:
        Dict[str, Dict[int, float]]: The value of every scalar at every logged iteration, rounded for the facts.
    """
    values = {}
    for tag in tags:
        iterations, tag_values = read_scalars(log_folder=log_folder, tag=tag)
        values[tag] = {int(iteration): round_fact(value=value) for iteration, value in zip(iterations, tag_values)}
    return values


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


def draw_smoothed_curve(axes: matplotlib.axes.Axes, iterations: numpy.ndarray, values: numpy.ndarray, colour: str,
                        label: str, smoothing: bool, raw_opacity: float = 0.22) -> numpy.ndarray:
    """
    Draws one curve of a chart panel: the raw value of every iteration as a faint line and its smoothed curve on top,
    or the values alone with markers for a sparse curve (a validation every 25 iterations, a test every evaluation).

    Args:
        axes (matplotlib.axes.Axes): The panel.
        iterations (numpy.ndarray): The iterations.
        values (numpy.ndarray): The raw values.
        colour (str): The colour of the series.
        label (str): The name of the series in the legend.
        smoothing (bool): Whether the curve is smoothed, the raw values drawn faintly under it.
        raw_opacity (float): The opacity of the raw values under a smoothed curve, 0 to leave them out (a loss
            jumping between 0 and its maximum from one batch to the next). Defaults to 0.22.

    Returns:
        numpy.ndarray: The values drawn on top, smoothed or not.
    """
    if not smoothing:
        axes.plot(iterations, values, color=colour, linewidth=2 * POINTS_PER_PIXEL, marker="o",
                  markersize=8 * POINTS_PER_PIXEL, label=label, solid_capstyle="round", solid_joinstyle="round")
        return values
    smoothed = smooth_values(values=values, weight=SMOOTHING_WEIGHT)
    if raw_opacity > 0:
        axes.plot(iterations, values, color=colour, alpha=raw_opacity, linewidth=1.2 * POINTS_PER_PIXEL)
    axes.plot(iterations, smoothed, color=colour, linewidth=3 * POINTS_PER_PIXEL, label=label,
              solid_capstyle="round", solid_joinstyle="round")
    return smoothed
