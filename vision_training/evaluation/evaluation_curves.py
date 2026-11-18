"""
Evaluation summary and evaluation curves of a run, kept up to date after every test evaluation so that the test results
can be read at a glance without TensorBoard. The base trainer writes the scalars of the evaluated weights into the row
of their iteration of 'evaluation_summary.csv' (one row per evaluated iteration, the averaged weights in 'ema_'
columns with their number of updates), in '<run>/Evaluation/' for a training and in '<run>/Checkpoint_Evaluation/' for
scripts/evaluation/evaluate_checkpoint.py, then draws 'evaluation_curves.png' next to the Results.txt of the run: the
headline metric of the task for the trained and the averaged weights against the iteration, the best iteration of each
ringed with its value and, with the moving average on, the last 1 / (1 − d) iterations of the averaged weights of the
highest evaluated iteration shaded, d being the decay of their last update, with the configured window behind while
the warm-up of the decay shortens it. A summary missing at
the first evaluation of a run evaluated before the summary existed is started from the TensorBoard test curves of the
run. The same drawing gives the training curves of the research reports (scripts/evaluation/draw_report_figures.py),
and the best iteration logic gives the best checkpoint printed by checkpoint_evaluation.py.
"""

import csv
import math

from functools import partial
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from vision_training.core.contracts import EvaluationResults
from vision_training.evaluation.report_formatting import (format_percentage_label, format_percentage_tick,
                                                          format_report_value)
from vision_training.trainers.weight_averaging import compute_recent_updates_weight, compute_update_decay
from vision_training.utilities.os_utilities import format_path_link, print_color

# Summary of the evaluations of a folder, next to its iteration folders ('<run>/Evaluation/' for a training,
# '<run>/Checkpoint_Evaluation/' for evaluate_checkpoint.py), and the figure of its headline metric, next to the
# Results.txt of the run or of the standalone evaluation
EVALUATION_SUMMARY_FILE_NAME = "evaluation_summary.csv"
EVALUATION_CURVES_FIGURE_NAME = "evaluation_curves.png"

# First column of the evaluation summary, and prefix of the columns of the averaged weights next to those of the
# trained weights
ITERATION_COLUMN = "iteration"
AVERAGED_WEIGHTS_COLUMN_PREFIX = "ema_"

# Column of the number of updates of the evaluated averaged weights, restored with their checkpoint, giving the decay of
# their last update during the warm-up of the decay, hence the window shaded at their iteration
AVERAGED_WEIGHTS_UPDATES_COLUMN = f"{AVERAGED_WEIGHTS_COLUMN_PREFIX}number_updates"

# Column prefix of the trained weights ('trained') and of the averaged weights ('ema'), the keys of the curves
WEIGHTS_COLUMN_PREFIXES = {"trained": "", "ema": AVERAGED_WEIGHTS_COLUMN_PREFIX}

# Suffix of the file written next to the summary or the figure before it replaces them, so that a reader copying them
# meanwhile (the MLflow worker, an image viewer) never meets a half-written file
PARTIAL_FILE_SUFFIX = ".part"

# Inks of the evaluation figures and of the figures of the research reports, the trained and averaged weights in the
# first two hues of the categorical palette, named in the legend and, shorter, in the labels of their best iteration
TEXT_PRIMARY_COLOR, TEXT_SECONDARY_COLOR, GRID_COLOR = "#1f2328", "#5f6670", "#e3e6ea"
WEIGHTS_COLORS = {"trained": "#2a78d6", "ema": "#eb6834"}
WEIGHTS_LABELS = {"trained": "Trained weights", "ema": "Moving average of the weights (EMA)"}
WEIGHTS_SHORT_NAMES = {"trained": "Trained", "ema": "EMA"}

# Opacity of the spans the moving average of the weights looks back over, washes of their colours: the window of its
# last update in the colour of the averaged weights, and the configured window, longer during the warm-up of the
# decay, in grey behind it
AVERAGING_WINDOW_OPACITY = 0.12
CONFIGURED_WINDOW_COLOR = "#5f6670"

# Size in inches and resolution of the figure, so that a Notion page or an image viewer shows it at its full width
CURVES_FIGURE_SIZE = (9.0, 4.2)
CURVES_DOTS_PER_INCH = 200

# Evaluations of a curve above which its points are no longer marked, too close to be told apart
MAXIMUM_MARKED_EVALUATIONS = 60

# Room left above and below the curves for the labels of the ringed points, as shares of the height of the plot; the
# smallest range of values of a rate axis (5 points), so that a few close evaluations are not magnified, and the range
# given to a flat curve of another metric (a single evaluation, or a constant value), a share of its value with a floor
TOP_MARGIN_SHARE, BOTTOM_MARGIN_SHARE = 0.18, 0.12
MINIMUM_RATE_RANGE = 0.05
FLAT_CURVE_RANGE_SHARE, MINIMUM_VALUE_RANGE = 0.1, 0.01

# Offset in points of the label of a ringed point from its centre, beyond the ring
LABEL_OFFSET_POINTS = 10

# Share of the iteration axis from its ends within which the label of a ringed point is written towards the inside of
# the plot, so that it stays within it
LABEL_EDGE_SHARE = 0.15

# Iteration from which the ticks of the iteration axis are written in thousands ("2.5k"), the most intervals between
# the ticks of a rate axis and their steps (those of matplotlib's default ticks, "80 %", "85 %"), and the tolerance
# beyond 0 and 100 % within which a tick of a rate axis is still drawn
ITERATIONS_IN_THOUSANDS_FROM = 2000
MAXIMUM_RATE_TICK_INTERVALS = 7
RATE_TICK_STEPS = [1, 2, 2.5, 5, 10]
RATE_TICK_TOLERANCE = 1e-9


class MarkedPoint(TypedDict):
    """
    Point ringed on a curve of an evaluation figure, with its label written next to it: the best iteration of the
    trained or the averaged weights of a run (every tied iteration ringed, the label next to the first), or the kept
    checkpoint of a research report.
    """
    weights: str
    iterations: List[int]
    label: str


class MovingAverageState(TypedDict):
    """
    Averaged weights of the last evaluated iteration, whose windows are shaded on the evaluation curves: the configured
    decay of the moving average, its number of updates at that iteration (one per training iteration since it
    started, restored with the checkpoint) and the iteration, where the windows end.
    """
    decay: float
    number_updates: int
    iteration: int


class AveragingWindow(TypedDict):
    """
    Span shaded on the evaluation curves, ending at the evaluation of the averaged weights: its length in iterations,
    the colour of its wash and its legend label.
    """
    length: float
    color: str
    label: str


def build_summary_columns(results: EvaluationResults, prefix: str, per_class_report: bool) -> Dict[str, float]:
    """
    Gives the columns of one kind of weights in the row of an evaluated iteration of the evaluation summary, written by
    the base trainer after every evaluation: the global scalars of the evaluation and, with the per-class report, those
    of every class, each name preceded by the prefix of the weights ('' for the trained weights, 'ema_' for the
    averaged weights).

    Args:
        results (EvaluationResults): The results of the evaluation of the weights.
        prefix (str): The prefix of the column names.
        per_class_report (bool): Whether the scalars of every class are reported ('evaluation.per_class_report').

    Returns:
        Dict[str, float]: The value of every column.
    """
    scalars = {**results["scalars"], **(results["class_scalars"] if per_class_report else {})}
    return {f"{prefix}{scalar_name}": value for scalar_name, value in scalars.items()}


def read_evaluation_summary(summary_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Reads an evaluation summary as write_evaluation_summary writes it, to add the row of a new evaluation to it or to
    draw its curves, a missing summary (no evaluation yet) giving no column and no row.

    Args:
        summary_path (Path): The evaluation_summary.csv file.

    Returns:
        Tuple[List[str], List[Dict[str, str]]]: The column names, 'iteration' first, and the rows as written, an empty
            cell standing for a value the evaluation of its row did not give.

    Raises:
        ValueError: If the file does not start with the 'iteration' column.
    """
    if not summary_path.is_file():
        return [], []
    with open(file=summary_path, mode="r", newline="") as summary_file:
        reader = csv.DictReader(f=summary_file)
        summary_rows = list(reader)
        column_names = list(reader.fieldnames) if reader.fieldnames is not None else []

    # A summary written by the trainer always starts with the iteration of its rows
    if not column_names or column_names[0] != ITERATION_COLUMN:
        raise ValueError(f"Expected {format_path_link(path=summary_path)} to start with the '{ITERATION_COLUMN}' "
                         f"column, got {column_names[:3]}: move it away so that the next evaluation starts it again.")
    return column_names, summary_rows


def write_evaluation_summary(summary_path: Path, column_names: List[str],
                             rows_by_iteration: Dict[int, Dict[str, Any]]) -> None:
    """
    Writes the evaluation summary, one row per evaluated iteration in increasing order, into a partial file that then
    replaces the summary, so that a reader copying it meanwhile never meets a half-written file and a stop during the
    write keeps the previous summary.

    Args:
        summary_path (Path): The evaluation_summary.csv file.
        column_names (List[str]): The columns, 'iteration' first.
        rows_by_iteration (Dict[int, Dict[str, Any]]): The row of every evaluated iteration, its missing columns left
            empty.
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = summary_path.with_name(f"{summary_path.name}{PARTIAL_FILE_SUFFIX}")
    with open(file=partial_path, mode="w", newline="") as summary_file:
        writer = csv.DictWriter(f=summary_file, fieldnames=column_names)
        writer.writeheader()
        writer.writerows([{**rows_by_iteration[iteration], ITERATION_COLUMN: iteration}
                          for iteration in sorted(rows_by_iteration)])
    partial_path.replace(target=summary_path)


def write_summary_columns(summary_path: Path, iteration: int, columns: Dict[str, float], starts_row: bool) -> None:
    """
    Writes the columns of an evaluated kind of weights into the row of their iteration of the evaluation summary,
    called by the base trainer after every evaluation: the trained weights start the row of their iteration, replacing
    the row of an earlier evaluation of the same iteration (e.g. a resumed run evaluating a checkpoint again), and the
    averaged weights, evaluated right after them, add their 'ema_' columns to it. A column met for the first time is
    added after the others, left empty in the rows of the earlier evaluations.

    Args:
        summary_path (Path): The evaluation_summary.csv file, created by the first evaluation.
        iteration (int): The evaluated iteration.
        columns (Dict[str, float]): The columns of the evaluated weights (build_summary_columns).
        starts_row (bool): Whether the columns start the row of the iteration (the trained weights) instead of being
            added to it (the averaged weights).
    """
    column_names, summary_rows = read_evaluation_summary(summary_path=summary_path)
    rows_by_iteration = {int(summary_row[ITERATION_COLUMN]): summary_row for summary_row in summary_rows}

    # The trained weights start the row afresh, the averaged weights complete the row of their iteration
    iteration_row = {} if starts_row or iteration not in rows_by_iteration else rows_by_iteration[iteration]
    iteration_row.update(columns)
    rows_by_iteration[iteration] = iteration_row

    # Columns in the order they first appear, those of this evaluation after the earlier ones
    column_names = column_names or [ITERATION_COLUMN]
    column_names += [column_name for column_name in columns if column_name not in column_names]
    write_evaluation_summary(summary_path=summary_path, column_names=column_names, rows_by_iteration=rows_by_iteration)


def read_test_curves(test_folder: Path) -> Dict[str, Dict[int, float]]:
    """
    Reads every test curve of a run at every evaluated checkpoint from its Tensorboard/Test (or Test_EMA) folder,
    written by the trainer at every weight_saving_iterations checkpoint: the global metrics and the per-class ones
    ('Recall/<class name>'), compared between the launches of an experiment by
    scripts/evaluation/collect_experiment_results.py, between the best evaluations of several runs by
    compare_best_evaluations.py, drawn by draw_report_figures.py, and starting the evaluation summary of a run
    evaluated before the summary existed (seed_evaluation_summary). A run evaluated again at the same iteration after a
    restart keeps its last evaluation.

    Args:
        test_folder (Path): The Tensorboard/Test or Tensorboard/Test_EMA folder of the run folder.

    Returns:
        Dict[str, Dict[int, float]]: The value at every evaluated iteration of every curve, empty before the first
            evaluation.
    """
    # No test curve before the first evaluation of the run
    if not test_folder.is_dir():
        return {}
    events = EventAccumulator(path=str(test_folder), size_guidance={"scalars": 0})
    events.Reload()

    # Later events of the same iteration overwrite the earlier ones
    return {curve_name: {event.step: event.value for event in events.Scalars(tag=curve_name)}
            for curve_name in events.Tags()["scalars"]}


def seed_evaluation_summary(summary_path: Path, test_folders: Dict[str, Path], current_iteration: int) -> int:
    """
    Starts the evaluation summary of a run from its TensorBoard test curves, at its first evaluation without summary,
    so that a run evaluated before the summary existed keeps its earlier evaluations in its curves: one row per
    evaluated iteration with every scalar of the Test folder, and those of the Test_EMA folder in 'ema_' columns, as the
    trainer writes them, the iteration being evaluated left out since its evaluation writes its exact row right after.
    TensorBoard keeps float32 copies of the scalars (about 7 significant digits), which a later evaluation of the same
    iteration replaces with its exact values. Reading the folders takes a few seconds for the tens of megabytes of
    sample images of a long run, once.

    Args:
        summary_path (Path): The evaluation_summary.csv file to start, written only when the folders hold an earlier
            evaluation.
        test_folders (Dict[str, Path]): The TensorBoard folder of every kind of weights, keyed by the prefix of its
            columns ('' for Tensorboard/Test, 'ema_' for Tensorboard/Test_EMA).
        current_iteration (int): The iteration being evaluated, left out.

    Returns:
        int: The number of evaluated iterations written, 0 when the folders hold no earlier evaluation.
    """
    # Every scalar of every earlier evaluated iteration, the columns in the order they first appear
    rows_by_iteration: Dict[int, Dict[str, Any]] = {}
    column_names = [ITERATION_COLUMN]
    for prefix, test_folder in test_folders.items():
        for curve_name, curve in read_test_curves(test_folder=test_folder).items():
            column_names.append(f"{prefix}{curve_name}")
            for iteration, value in curve.items():
                if iteration != current_iteration:
                    rows_by_iteration.setdefault(iteration, {})[f"{prefix}{curve_name}"] = value

    # Written and announced only for a run evaluated before, so that its float32 rows are not taken for exact ones
    if rows_by_iteration:
        write_evaluation_summary(summary_path=summary_path, column_names=column_names,
                                 rows_by_iteration=rows_by_iteration)
        print_color(color="blue", output=f"Evaluation Summary Started From The TensorBoard Test Curves Of "
                                         f"{len(rows_by_iteration)} Evaluated Iterations : "
                                         f"{format_path_link(path=summary_path)}", indent=1)
    return len(rows_by_iteration)


def find_best_iterations(values: Dict[int, float]) -> Optional[Tuple[List[int], float]]:
    """
    Finds the best evaluations of a headline metric (the higher the better), for the best checkpoint printed by the
    checkpoint evaluation and the rings of the evaluation curves: the iterations sharing the highest value, in the
    order of the values, so that a tie is reported instead of one of the tied iterations being picked. A value that is
    not a number (NaN) is never the best.

    Args:
        values (Dict[int, float]): The value of the metric at every evaluated iteration.

    Returns:
        Optional[Tuple[List[int], float]]: The best iterations and their value, None without any value.
    """
    compared_values = {iteration: value for iteration, value in values.items() if not math.isnan(value)}
    if not compared_values:
        return None
    best_value = max(compared_values.values())
    return [iteration for iteration, value in compared_values.items() if value == best_value], best_value


def read_summary_curves(column_names: List[str], summary_rows: List[Dict[str, str]],
                        metric_name: str) -> Dict[str, Dict[int, float]]:
    """
    Gives the curves of a metric over the evaluated iterations of an evaluation summary, drawn by draw_summary_curves:
    the curve of the trained weights ('trained', from the column of the metric) and that of the averaged weights
    ('ema', from its 'ema_' column), a row without value (e.g. the averaged weights of a run before its moving average
    was turned on) being left out, and a kind of weights without any value too.

    Args:
        column_names (List[str]): The columns of the summary (read_evaluation_summary).
        summary_rows (List[Dict[str, str]]): The rows of the summary.
        metric_name (str): The metric drawn, e.g. the headline metric of the task.

    Returns:
        Dict[str, Dict[int, float]]: The value at every evaluated iteration of every kind of weights holding one.
    """
    curves = {}
    for weights, prefix in WEIGHTS_COLUMN_PREFIXES.items():
        column_name = f"{prefix}{metric_name}"
        if column_name not in column_names:
            continue
        curve = {int(summary_row[ITERATION_COLUMN]): float(summary_row[column_name]) for summary_row in summary_rows
                 if summary_row[column_name] not in (None, "")}
        if curve:
            curves[weights] = curve
    return curves


def read_moving_average_state(averaged_curve: Dict[int, float], summary_rows: List[Dict[str, str]],
                              moving_average_decay: Optional[float]) -> Optional[MovingAverageState]:
    """
    Gives the averaged weights whose windows are shaded on the evaluation curves, read by draw_summary_curves: those
    of the highest iteration evaluated with averaged weights, whatever the order of the evaluations
    (evaluate_checkpoint.py evaluates "all" from the last checkpoint down), with the number of updates written next
    to their scalars, so that the warm-up of the decay is read from their checkpoint rather than assumed from the
    iteration.

    Args:
        averaged_curve (Dict[int, float]): The values of the averaged weights at every evaluated iteration
            (read_summary_curves), empty when they were never evaluated.
        summary_rows (List[Dict[str, str]]): The rows of the summary.
        moving_average_decay (Optional[float]): The configured decay of the moving average of the run, None when it
            is off.

    Returns:
        Optional[MovingAverageState]: The averaged weights of the highest iteration, None without moving average,
            without averaged evaluation, or when the row of that iteration holds no number of updates (a row started
            from TensorBoard).
    """
    if moving_average_decay is None or not averaged_curve:
        return None

    # Number of updates in the row of the highest averaged evaluation, absent from the rows started from TensorBoard
    last_iteration = max(averaged_curve)
    last_row = next(summary_row for summary_row in summary_rows if int(summary_row[ITERATION_COLUMN]) == last_iteration)
    if AVERAGED_WEIGHTS_UPDATES_COLUMN not in last_row or last_row[AVERAGED_WEIGHTS_UPDATES_COLUMN] in (None, ""):
        return None
    return {"decay": moving_average_decay, "number_updates": int(last_row[AVERAGED_WEIGHTS_UPDATES_COLUMN]),
            "iteration": last_iteration}


def format_iteration_count(value: float) -> str:
    """
    Writes a number of iterations of a window of the moving average in the legend of the evaluation curves, rounded to
    a whole iteration with thousands separators, e.g. "5,000 iterations".

    Args:
        value (float): The number of iterations.

    Returns:
        str: The rounded number followed by its unit.
    """
    rounded_value = round(value)
    return f"{rounded_value:,} iteration{'' if rounded_value == 1 else 's'}"


def describe_averaging_windows(decay: float, number_updates: int) -> List[AveragingWindow]:
    """
    Describes how far back the moving average of the weights evaluated last looks, for the spans shaded on the
    evaluation curves: a weight of k updates ago counts with a factor (1 − d) × d^k, so the average follows about its
    last 1 / (1 − d) updates, one per training iteration, d being the decay of its last update. Once the warm-up of the
    decay is over, d is the configured decay, whose window (5,000 iterations at 0.9998) carries about 1 − 1/e = 63 %
    of the weight of the average. During the warm-up (compute_update_decay), d is lower and the window of the last
    update shorter, about 1,670 iterations after 15,000 updates at 0.9998, the configured window being shaded behind
    it in grey.

    Args:
        decay (float): The configured decay of the average, strictly between 0 and 1.
        number_updates (int): The number of updates of the evaluated average, at least 1.

    Returns:
        List[AveragingWindow]: The windows to shade, the longest first: "EMA window ≈ 1/(1 − decay) = 5,000
            iterations" with the decay and the weight the window carries on a second line once the warm-up is over,
            and otherwise the configured window and the window of the last update with its decay.
    """
    configured_length = 1.0 / (1.0 - decay)
    last_update_decay = compute_update_decay(decay=decay, update_index=number_updates - 1)

    # Warm-up over: the configured window alone, with the share of the weight of the average its updates carry
    if last_update_decay >= decay:
        window_weight = compute_recent_updates_weight(decay=decay, number_updates=number_updates,
                                                      recent_updates=min(max(round(configured_length), 1),
                                                                         number_updates))
        return [{"length": configured_length, "color": WEIGHTS_COLORS["ema"],
                 "label": f"EMA window ≈ 1/(1 − decay) = {format_iteration_count(value=configured_length)}\n"
                          f"(decay {decay:g}, {format_percentage_label(value=round(window_weight, 2))} of the weight "
                          f"of the average)"}]

    # During the warm-up, the configured window in grey behind the shorter window of the last update
    last_update_length = 1.0 / (1.0 - last_update_decay)
    return [{"length": configured_length, "color": CONFIGURED_WINDOW_COLOR,
             "label": f"Configured EMA window ≈ 1/(1 − decay) = {format_iteration_count(value=configured_length)}\n"
                      f"(decay {decay:g}, reached once the warm-up is over)"},
            {"length": last_update_length, "color": WEIGHTS_COLORS["ema"],
             "label": f"EMA window of the last update ≈ 1/(1 − d) = "
                      f"{format_iteration_count(value=last_update_length)}\n"
                      f"(d = {last_update_decay:.6g} at update {number_updates:,}, lowered by the warm-up)"}]


def describe_best_iterations(weights: str, best_iterations: List[int], best_value: float, in_percent: bool) -> str:
    """
    Writes the label of the best iteration of a kind of weights on the evaluation curves, its value written like in
    the report, e.g. "Trained best: iteration 112,500, 94.03 %", the number of other iterations reaching the same
    value added when several tie.

    Args:
        weights (str): The kind of weights, 'trained' or 'ema'.
        best_iterations (List[int]): The iterations sharing the best value (find_best_iterations).
        best_value (float): The best value.
        in_percent (bool): Whether the metric is a rate written as a percentage.

    Returns:
        str: The label.
    """
    tie_text = f" (+{len(best_iterations) - 1} tied)" if len(best_iterations) > 1 else ""
    return (f"{WEIGHTS_SHORT_NAMES[weights]} best: iteration {best_iterations[0]:,}{tie_text}, "
            f"{format_report_value(value=best_value, in_percent=in_percent)}")


def format_iteration_tick(value: float, position: Optional[int] = None, in_thousands: bool = False) -> str:
    """
    Writes a tick of the iteration axis of the evaluation figures, in thousands on an axis reaching
    ITERATIONS_IN_THOUSANDS_FROM iterations, so that the ticks of a long run stay short ("2.5k", "120k"), and in full
    on the axis of a short run ("750"), 0 being written "0" in both.

    Args:
        value (float): The iteration of the tick.
        position (Optional[int]): The position of the tick, given by matplotlib and not used. Defaults to None.
        in_thousands (bool): Whether the ticks of the axis are written in thousands. Defaults to False.

    Returns:
        str: The tick label.
    """
    if in_thousands and value != 0:
        return f"{value / 1000.0:g}k"
    return f"{value:,.0f}"


def compute_value_limits(values: List[float], in_percent: bool) -> Tuple[float, float]:
    """
    Gives the limits of the value axis of an evaluation figure: the values in the middle of the plot, with room above
    them and below them for the labels of the ringed points, the values of a rate spread over 5 points at least so
    that a few close evaluations are not magnified, and a flat curve of another metric given a range of its own.

    Args:
        values (List[float]): The finite values drawn, at least one.
        in_percent (bool): Whether the metric is a rate written as a percentage.

    Returns:
        Tuple[float, float]: The bottom and the top of the value axis.
    """
    lowest_value, highest_value = min(values), max(values)
    minimum_range = MINIMUM_RATE_RANGE if in_percent else (
        max(abs(highest_value) * FLAT_CURVE_RANGE_SHARE, MINIMUM_VALUE_RANGE) if highest_value == lowest_value else 0.0)

    # A range too small is widened around its middle, within 0 and 100 % for a rate
    if highest_value - lowest_value < minimum_range:
        middle_value = (lowest_value + highest_value) / 2.0
        if in_percent:
            middle_value = min(max(middle_value, minimum_range / 2.0), 1.0 - minimum_range / 2.0)
        lowest_value, highest_value = middle_value - minimum_range / 2.0, middle_value + minimum_range / 2.0

    # The values take the height of the plot left by the room above and below them
    axis_range = (highest_value - lowest_value) / (1.0 - TOP_MARGIN_SHARE - BOTTOM_MARGIN_SHARE)
    return lowest_value - BOTTOM_MARGIN_SHARE * axis_range, highest_value + TOP_MARGIN_SHARE * axis_range


def draw_evaluation_curves(axis: Axes, curves: Dict[str, Dict[int, float]], marked_points: List[MarkedPoint],
                           metric_label: str, in_percent: bool, moving_average: Optional[MovingAverageState]) -> None:
    """
    Draws the evaluations of a metric during a training on an axis, shared by the evaluation curves of a run and the
    training curves of a research report: one line per kind of weights with a point at every evaluation, every marked
    point ringed with its label (above it, below it for the lower of two marked points), the windows of the moving
    average evaluated last shaded up to its evaluation when it is on (describe_averaging_windows), the rates in
    percent, the iterations in thousands on a long run, and a legend above the plot naming the curves and the shaded
    windows (none for a single curve, which the axis names).

    Args:
        axis (Axes): The plot.
        curves (Dict[str, Dict[int, float]]): The value at every evaluated iteration of the 'trained' and/or 'ema'
            weights.
        marked_points (List[MarkedPoint]): The points to ring, on the curves of their weights.
        metric_label (str): The name of the metric, written on the vertical axis.
        in_percent (bool): Whether the metric is a rate written as a percentage.
        moving_average (Optional[MovingAverageState]): The moving average of the weights evaluated last, giving the
            shaded windows, None for no window.
    """
    # One line per kind of weights, a point ringed in white at every evaluation unless they are too many to be told
    # apart
    for weights, curve in curves.items():
        curve_iterations = sorted(curve)
        marker = "o" if len(curve_iterations) <= MAXIMUM_MARKED_EVALUATIONS else None
        axis.plot(curve_iterations, [curve[curve_iteration] for curve_iteration in curve_iterations],
                  color=WEIGHTS_COLORS[weights], linewidth=2, marker=marker, markersize=5, markeredgecolor="white",
                  markeredgewidth=0.7, label=WEIGHTS_LABELS[weights], zorder=3)

    # Iterations the moving average evaluated last looks back over, up to its evaluation, the longest window behind
    if moving_average is not None:
        for averaging_window in describe_averaging_windows(decay=moving_average["decay"],
                                                           number_updates=moving_average["number_updates"]):
            axis.axvspan(xmin=max(0.0, moving_average["iteration"] - averaging_window["length"]),
                         xmax=moving_average["iteration"], color=averaging_window["color"],
                         alpha=AVERAGING_WINDOW_OPACITY, linewidth=0, label=averaging_window["label"], zorder=1)

    # Room above and below the curves for the labels of the ringed points, the ticks of a rate kept within 0 and 100 %
    values = [value for curve in curves.values() for value in curve.values() if math.isfinite(value)]
    if values:
        bottom_value, top_value = compute_value_limits(values=values, in_percent=in_percent)
        axis.set_ylim(bottom=bottom_value, top=top_value)
        if in_percent:
            tick_values = MaxNLocator(nbins=MAXIMUM_RATE_TICK_INTERVALS, steps=RATE_TICK_STEPS).tick_values(
                vmin=bottom_value, vmax=top_value)
            axis.set_yticks(ticks=[tick_value for tick_value in tick_values
                                   if max(bottom_value, -RATE_TICK_TOLERANCE) <= tick_value
                                   <= min(top_value, 1.0 + RATE_TICK_TOLERANCE)])

    # Whole iterations from 0 at most, the limits being fixed before the labels are aligned on them
    axis.xaxis.set_major_locator(locator=MaxNLocator(integer=True))
    left_end, right_end = axis.get_xlim()
    left_end = max(0.0, left_end)
    axis.set_xlim(left=left_end, right=right_end)

    # Ring around every marked point, every tied iteration included
    marked_values = []
    for marked_point in marked_points:
        curve = curves[marked_point["weights"]]
        axis.plot(marked_point["iterations"], [curve[iteration] for iteration in marked_point["iterations"]],
                  linestyle="none", marker="o", markersize=13, markerfacecolor="none",
                  markeredgecolor=TEXT_PRIMARY_COLOR, markeredgewidth=1.6, zorder=4)
        marked_values.append(curve[marked_point["iterations"][0]])

    # Label of every marked point next to its first ring, above it for the highest marked point and below it for the
    # others, aligned towards the inside of the plot near its ends
    for point_index, marked_point in enumerate(marked_points):
        label_iteration = marked_point["iterations"][0]
        is_above = point_index == marked_values.index(max(marked_values))
        axis_share = (label_iteration - left_end) / (right_end - left_end) if right_end > left_end else 0.5
        horizontal_alignment = "center"
        if axis_share > 1.0 - LABEL_EDGE_SHARE:
            horizontal_alignment = "right"
        elif axis_share < LABEL_EDGE_SHARE:
            horizontal_alignment = "left"
        axis.annotate(text=marked_point["label"], xy=(label_iteration, marked_values[point_index]),
                      xytext=(0, LABEL_OFFSET_POINTS if is_above else -LABEL_OFFSET_POINTS),
                      textcoords="offset points", ha=horizontal_alignment, va="bottom" if is_above else "top",
                      fontsize=9, color=TEXT_PRIMARY_COLOR, zorder=5,
                      bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.9})

    # Rates in percent, iterations in thousands on a long run, recessive axes and grid
    if in_percent:
        axis.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))
    axis.xaxis.set_major_formatter(formatter=FuncFormatter(func=partial(
        format_iteration_tick, in_thousands=right_end >= ITERATIONS_IN_THOUSANDS_FROM)))
    axis.set_xlabel(xlabel="Training iteration", color=TEXT_SECONDARY_COLOR, fontsize=10)
    axis.set_ylabel(ylabel=metric_label, color=TEXT_SECONDARY_COLOR, fontsize=10)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID_COLOR)
    axis.tick_params(colors=TEXT_SECONDARY_COLOR, labelsize=9)

    # Legend above the plot, two columns: the curves, then the shaded windows
    _, legend_labels = axis.get_legend_handles_labels()
    if len(legend_labels) > 1:
        axis.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=False, fontsize=9,
                    labelcolor=TEXT_PRIMARY_COLOR, borderaxespad=0.0, columnspacing=2.5)


def save_evaluation_curves_figure(curves: Dict[str, Dict[int, float]], marked_points: List[MarkedPoint],
                                  metric_label: str, in_percent: bool, moving_average: Optional[MovingAverageState],
                                  title: Optional[str], output_path: Path) -> None:
    """
    Draws the evaluations of a metric during a training (draw_evaluation_curves) on a figure of its own, without
    pyplot, and writes it into a partial file that then replaces the figure, so that an image viewer or the MLflow
    worker reading it meanwhile never meets a half-written image.

    Args:
        curves (Dict[str, Dict[int, float]]): The value at every evaluated iteration of the 'trained' and/or 'ema'
            weights.
        marked_points (List[MarkedPoint]): The points to ring.
        metric_label (str): The name of the metric, written on the vertical axis.
        in_percent (bool): Whether the metric is a rate written as a percentage.
        moving_average (Optional[MovingAverageState]): The moving average of the weights evaluated last, None for no
            shaded window.
        title (Optional[str]): The title written above the figure, None for none.
        output_path (Path): The PNG written.
    """
    figure = Figure(figsize=CURVES_FIGURE_SIZE)
    FigureCanvasAgg(figure=figure)
    axis = figure.subplots()
    draw_evaluation_curves(axis=axis, curves=curves, marked_points=marked_points, metric_label=metric_label,
                           in_percent=in_percent, moving_average=moving_average)

    # Title in the band kept free at the top of the figure, above the legend
    layout_top = 1.0
    if title is not None:
        figure.text(x=0.012, y=0.985, s=title, ha="left", va="top", fontsize=12, color=TEXT_PRIMARY_COLOR)
        layout_top = 0.945
    figure.tight_layout(rect=(0.0, 0.0, 1.0, layout_top))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f"{output_path.name}{PARTIAL_FILE_SUFFIX}")
    figure.savefig(fname=partial_path, format="png", dpi=CURVES_DOTS_PER_INCH, facecolor="white")
    partial_path.replace(target=output_path)


def draw_summary_curves(summary_path: Path, figure_path: Path, metric_name: str, in_percent: bool,
                        moving_average_decay: Optional[float]) -> None:
    """
    Draws the evaluation curves of a run or of a standalone evaluation from its evaluation summary, redrawn by the
    base trainer after every evaluation: the metric (the headline metric of the task) of the trained and the averaged
    weights at every evaluated iteration, the best iteration of each ringed with its value ("Trained best: iteration
    112,500, 94.03 %", find_best_iterations), and the windows of the averaged weights of the highest evaluated
    iteration when the moving average is on (read_moving_average_state).

    Args:
        summary_path (Path): The evaluation_summary.csv file.
        figure_path (Path): The evaluation_curves.png written.
        metric_name (str): The column of the drawn metric, the headline metric of the task.
        in_percent (bool): Whether the metric is a rate written as a percentage.
        moving_average_decay (Optional[float]): The configured decay of the moving average of the weights of the run,
            None when it is off.
    """
    column_names, summary_rows = read_evaluation_summary(summary_path=summary_path)
    curves = read_summary_curves(column_names=column_names, summary_rows=summary_rows, metric_name=metric_name)
    moving_average = read_moving_average_state(averaged_curve=curves["ema"] if "ema" in curves else {},
                                               summary_rows=summary_rows, moving_average_decay=moving_average_decay)

    # Best iterations of every kind of weights, every tied one ringed
    marked_points = []
    for weights, curve in curves.items():
        best_evaluation = find_best_iterations(values=curve)
        if best_evaluation is None:
            continue
        best_iterations, best_value = best_evaluation
        marked_points.append({"weights": weights, "iterations": best_iterations,
                              "label": describe_best_iterations(weights=weights, best_iterations=best_iterations,
                                                                best_value=best_value, in_percent=in_percent)})

    save_evaluation_curves_figure(curves=curves, marked_points=marked_points, metric_label=metric_name,
                                  in_percent=in_percent, moving_average=moving_average,
                                  title=f"Test {metric_name} at every evaluation", output_path=figure_path)
