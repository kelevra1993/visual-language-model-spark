"""
When run, draws one figure per comparison of the hyperparameter experiments (classification, sequence labelling,
object detection and instance segmentation), so that the verdict of every stage and the noise of a single launch can be
seen at a glance. The stages are listed in documentation/architecture_benchmarking/experiment_comparisons.yaml (the
settings compared, their launches, the reference setting, the verdict of the plan), and the test curves of their
launches are read from the results files kept next to the experiment plans (collect_experiment_results.py and the
detection driver write them in the git-ignored hyperparameter_experiments folder, the plans keep a copy, the instance
segmentation results giving the test mask mAP). Each figure shows, for the trained and the averaged weights, every
launch as a thin line and the mean of the launches of each setting as a thick one, with the gap of every setting to the
reference setting of the stage below; the figures are written to
documentation/architecture_benchmarking/figures/<experiment>/ and linked from the experiment plans. Every compared
metric is a rate written as the percentages of the reports (labels "85.81 %", gaps "+0.95 points", axes "86 %"): the
accuracy of the classification, the macro F1 of the sequence labelling, the mAP of the detection and the mask mAP of
the instance segmentation. Run as a script with the registry, the figures folder, the experiment and the stage given as
optional arguments (parse_comparison_arguments), every stage of every experiment being drawn when none is given.
"""

import argparse
import csv
import re
import statistics
import sys
import textwrap
import yaml

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vision_training.evaluation.report_formatting import (format_percentage, format_percentage_tick, format_points,
                                                          format_points_tick)
from vision_training.utilities.os_utilities import format_path_link, print_color

# Folder of the experiment plans, their results files, the comparison registry and the figures
BENCHMARKING_FOLDER = Path(__file__).resolve().parents[2] / "documentation" / "architecture_benchmarking"

# Registry of the compared stages, next to the results files it reads, and folder of the figures, one subfolder per
# experiment, the experiment plans linking every figure: the defaults of the command line of the script
DEFAULT_REGISTRY_PATH = BENCHMARKING_FOLDER / "experiment_comparisons.yaml"
DEFAULT_OUTPUT_FOLDER = BENCHMARKING_FOLDER / "figures"

# Titles of the experiments, of the test metric of their results files (the prefix of the metric columns) and of the
# evaluated weights, in the order of the figure columns
EXPERIMENT_TITLES = {"classification": "Classification", "sequence_labelling": "Sequence labelling",
                     "object_detection": "Object detection", "instance_segmentation": "Instance segmentation"}
METRIC_TITLES = {"accuracy": "accuracy", "macro_f1": "Macro F1", "map_50_95": "mAP@[0.5:0.95]",
                 "mask_map_50_95": "Mask mAP@[0.5:0.95]"}

# Metric prefixes written as the percentages of the reports, their gaps in points: every metric of METRIC_TITLES, the
# classification and sequence labelling rates and the detection and mask mAP
PERCENT_METRICS = ["accuracy", "macro_f1", "map_50_95", "mask_map_50_95"]
WEIGHTS_TITLES = {"trained": "Trained weights", "ema": "Averaged weights (EMA)"}

# Metric column of a results file, the metric prefix followed by the evaluated iteration (map_50_95_15000)
METRIC_COLUMN_PATTERN = re.compile(pattern=r"^(?P<prefix>[a-z0-9_]+?)_(?P<iteration>\d+)$")

# Categorical colours of the settings in their fixed order, the reference setting always taking the first one (the
# validated default palette of the dataviz guidelines, every adjacent pair separable under colour vision deficiency),
# the light surface, the text inks and the recessive grid
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE_COLOR, TEXT_PRIMARY_COLOR, TEXT_SECONDARY_COLOR, GRID_COLOR = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"

# Line style of the successive launches of a setting (a repeated setting has up to three launches), and the smallest
# vertical spacing between two direct labels
LAUNCH_LINE_STYLES = ["-", "--", ":", "-."]
LAUNCH_STYLE_NAMES = ["solid", "dashed", "dotted", "dash-dotted"]
MINIMUM_LABEL_SPACING_POINTS = 11.0

# Space before the unit of a percentage or a gap in points of a verdict ("85.81 %", "+0.95 points"), replaced by a
# non-breaking space before wrapping so that a number never ends a line without its unit
VERDICT_UNIT_PATTERN = re.compile(pattern=r"(?<=\d) (?=%|points\b)")


def read_results_curves(results_path: Path) -> Tuple[str, Dict[Tuple[str, str], Dict[int, float]]]:
    """
    Reads the test curve of every launch of a results file, as written by collect_experiment_results.py, so that the
    comparison figures can be drawn without the git-ignored run folders. The mean rows are left out, the figures
    computing their own means over the launches of each setting.

    Args:
        results_path (Path): The results file next to the experiment plan.

    Returns:
        Tuple[str, Dict[Tuple[str, str], Dict[int, float]]]: The metric prefix of the file (accuracy, macro_f1,
            map_50_95, mask_map_50_95) and, for every launch and evaluated weights ("trained" or "ema", the launches
            written before the averaged weights having no label and counting as trained), the value at every evaluated
            iteration.

    Raises:
        ValueError: If the file has metric columns of several prefixes or none, or the same launch twice.
    """
    with results_path.open(encoding="utf-8", newline="") as results_file:
        reader = csv.DictReader(results_file)

        # Find the metric columns and the iteration of each, every file holding a single test metric
        metric_columns = {}
        for column in reader.fieldnames:
            column_match = METRIC_COLUMN_PATTERN.match(string=column)
            if column_match is not None:
                metric_columns[column] = (column_match["prefix"], int(column_match["iteration"]))
        metric_prefixes = {metric_prefix for metric_prefix, iteration in metric_columns.values()}
        if len(metric_prefixes) != 1:
            raise ValueError(f"{results_path} should hold the columns of a single metric, found {metric_prefixes}.")

        # Keep the evaluated iterations of every launch row, the mean rows left out
        launch_curves = {}
        for row in reader:
            if row["launch"] == "mean":
                continue
            curve_key = (row["launch"], row["evaluated_weights"] or "trained")
            if curve_key in launch_curves:
                raise ValueError(f"{results_path} holds the {curve_key[1]} weights of {curve_key[0]} twice.")
            launch_curves[curve_key] = {iteration: float(row[column])
                                        for column, (metric_prefix, iteration) in metric_columns.items() if row[column]}
    return metric_prefixes.pop(), launch_curves


def compute_mean_curve(launch_curves: List[Dict[int, float]]) -> Dict[int, float]:
    """
    Averages the curves of the launches of one setting, the score each stage of the experiments compares, at the
    iterations evaluated by every launch (a launch stopped earlier ends the mean there).

    Args:
        launch_curves (List[Dict[int, float]]): The value at every evaluated iteration of each launch of the setting.

    Returns:
        Dict[int, float]: The mean of the launches at every iteration they all evaluated, in increasing order.
    """
    common_iterations = sorted(set.intersection(*[set(launch_curve) for launch_curve in launch_curves]))
    return {iteration: statistics.fmean(launch_curve[iteration] for launch_curve in launch_curves)
            for iteration in common_iterations}


def compute_gap_curve(setting_curve: Dict[int, float], reference_curve: Dict[int, float]) -> Dict[int, float]:
    """
    Gives the gap of a setting to the reference setting of its stage at every iteration both evaluated, the lower panel
    of the comparison figures, positive where the setting leads.

    Args:
        setting_curve (Dict[int, float]): The mean curve of the compared setting.
        reference_curve (Dict[int, float]): The mean curve of the reference setting.

    Returns:
        Dict[int, float]: The difference at every common iteration, in increasing order.
    """
    return {iteration: setting_curve[iteration] - reference_curve[iteration]
            for iteration in sorted(set(setting_curve) & set(reference_curve))}


def find_label_iteration(curve: Dict[int, float], last_iteration: int) -> int:
    """
    Finds where a curve is labelled: at the iteration the stage compares, or at its last evaluation when it stopped
    before it (a pair stopped once its verdict was known).

    Args:
        curve (Dict[int, float]): The curve to label.
        last_iteration (int): The iteration the stage compares at.

    Returns:
        int: The last evaluated iteration up to the compared one.

    Raises:
        ValueError: If the curve has no evaluation up to the compared iteration.
    """
    eligible_iterations = [iteration for iteration in curve if iteration <= last_iteration]
    if not eligible_iterations:
        raise ValueError(f"The curve has no evaluation up to iteration {last_iteration}.")
    return max(eligible_iterations)


def find_source_weights(arm: Dict[str, Any], weights: str) -> str:
    """
    Gives the weights whose curves stand for an arm in a column: its own, except for an arm trained without a moving
    average whose trained weights stand in the averaged-weights column (trained_as_averaged), as when a stage compares
    the averaged weights of a setting with a reference that had none.

    Args:
        arm (Dict[str, Any]): The arm of the registry.
        weights (str): The column, "trained" or "ema".

    Returns:
        str: The evaluated weights read for the arm.
    """
    return "trained" if weights == "ema" and arm["trained_as_averaged"] else weights


def collect_setting_curves(comparison: Dict[str, Any], launch_curves: Dict[Tuple[str, str], Dict[int, float]],
                           weights: str) -> List[Dict[str, Any]]:
    """
    Gathers, for one kind of weights, the launch curves and the mean curve of every setting of a comparison, with its
    colour, the reference setting taking the first colour of the palette and the others the next ones in plan order.

    Args:
        comparison (Dict[str, Any]): The comparison of the registry (its arms, each with its label, launches,
            reference flag and trained_as_averaged flag).
        launch_curves (Dict[Tuple[str, str], Dict[int, float]]): The curves of the results file, keyed by launch and
            evaluated weights.
        weights (str): "trained" or "ema".

    Returns:
        List[Dict[str, Any]]: For every setting, its label (marked when its trained weights stand for averaged ones),
            colour, reference flag, launches, launch curves and mean curve, the reference setting first.

    Raises:
        ValueError: If the comparison does not have exactly one reference setting, or has more settings than colours.
    """
    reference_arms = [arm for arm in comparison["arms"] if arm["reference"]]
    if len(reference_arms) != 1:
        raise ValueError(f"Stage {comparison['stage']} of {comparison['experiment']} should have one reference "
                         f"setting, found {len(reference_arms)}.")
    if len(comparison["arms"]) > len(SERIES_COLORS):
        raise ValueError(f"Stage {comparison['stage']} of {comparison['experiment']} compares "
                         f"{len(comparison['arms'])} settings, more than the {len(SERIES_COLORS)} colours.")

    # Reference setting first, then the compared settings in the order of the plan, each with its colour slot
    ordered_arms = reference_arms + [arm for arm in comparison["arms"] if not arm["reference"]]
    setting_curves = []
    for color, arm in zip(SERIES_COLORS, ordered_arms):
        source_weights = find_source_weights(arm=arm, weights=weights)
        curves_of_launches = [launch_curves[(launch, source_weights)] for launch in arm["launches"]]
        label = arm["label"] + (" (trained weights, no average)" if source_weights != weights else "")
        setting_curves.append({"label": label, "color": color, "reference": arm["reference"],
                               "launches": arm["launches"], "launch_curves": curves_of_launches,
                               "mean_curve": compute_mean_curve(launch_curves=curves_of_launches)})
    return setting_curves


def spread_label_offsets(axis: Axes, label_points: List[Tuple[float, float]]) -> List[float]:
    """
    Computes the vertical offsets, in points, that keep the direct labels of close curves from overlapping, each label
    moving down from the one above it by at least the minimum spacing.

    Args:
        axis (Axes): The panel, its limits already set.
        label_points (List[Tuple[float, float]]): The iteration and value of every label.

    Returns:
        List[float]: The vertical offset of every label, in points, in the order of the label points.
    """
    points_per_pixel = 72.0 / axis.figure.dpi
    display_heights = [axis.transData.transform(values=label_point)[1] * points_per_pixel
                       for label_point in label_points]

    # Walk the labels from the highest down, pushing each one below the previous by the minimum spacing if needed
    offsets = [0.0] * len(label_points)
    previous_height = None
    for label_index in sorted(range(len(label_points)), key=display_heights.__getitem__, reverse=True):
        placed_height = display_heights[label_index]
        if previous_height is not None and previous_height - placed_height < MINIMUM_LABEL_SPACING_POINTS:
            placed_height = previous_height - MINIMUM_LABEL_SPACING_POINTS
        offsets[label_index] = placed_height - display_heights[label_index]
        previous_height = placed_height
    return offsets


def style_axis(axis: Axes) -> None:
    """
    Applies the light surface, the recessive grid and the secondary ink of the axes of the comparison figures.

    Args:
        axis (Axes): The panel.
    """
    axis.set_facecolor(SURFACE_COLOR)
    axis.grid(color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ["top", "right"]:
        axis.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        axis.spines[side].set_color(GRID_COLOR)
    axis.tick_params(colors=TEXT_SECONDARY_COLOR, labelsize=9)


def draw_weights_column(curve_axis: Axes, gap_axis: Axes, setting_curves: List[Dict[str, Any]], last_iteration: int,
                        column_title: str, metric_title: str,
                        in_percent: bool) -> Tuple[float, List[Tuple[int, float]], List[str]]:
    """
    Draws the curves of one kind of weights: every launch thin, the mean of each setting thick with its direct label
    at the compared iteration, and below the gap of every compared setting to the reference setting. A rate is written
    as a percentage on its labels and axis ("85.81 %"), its gaps in points ("+0.95 points").

    Args:
        curve_axis (Axes): The upper panel.
        gap_axis (Axes): The lower panel.
        setting_curves (List[Dict[str, Any]]): The settings from collect_setting_curves, the reference first.
        last_iteration (int): The iteration the stage compares at.
        column_title (str): The title of the column (the weights, and whether the stage compares them).
        metric_title (str): The name of the test metric.
        in_percent (bool): Whether the metric is a rate written as a percentage, its gaps in points.

    Returns:
        Tuple[float, List[Tuple[int, float]], List[str]]: The largest absolute gap drawn, so that both columns share
            the scale of their gap panels, and the point and text of the label of every gap, placed once that scale
            is set.
    """
    reference_setting = setting_curves[0]
    plotted_values, plotted_iterations, label_points, label_texts = [], [], [], []
    for setting in setting_curves:
        # Every launch of the setting as a thin line, the successive launches told apart by their line style (the
        # styles repeating for the six launches of a repeated setting)
        for launch_index, launch_curve in enumerate(setting["launch_curves"]):
            iterations = sorted(launch_curve)
            curve_axis.plot(iterations, [launch_curve[iteration] for iteration in iterations], color=setting["color"],
                            linewidth=1, alpha=0.45,
                            linestyle=LAUNCH_LINE_STYLES[launch_index % len(LAUNCH_LINE_STYLES)])
            plotted_values += list(launch_curve.values())
            plotted_iterations += iterations

        # Mean of the launches as a thick line, labelled where the stage compares it
        mean_iterations = list(setting["mean_curve"])
        legend_label = (f"{setting['label']} ({', '.join(setting['launches'])})"
                        + (", reference" if setting["reference"] else ""))
        curve_axis.plot(mean_iterations, list(setting["mean_curve"].values()), color=setting["color"], linewidth=2.2,
                        marker="o", markersize=3.5, label=legend_label)
        label_iteration = find_label_iteration(curve=setting["mean_curve"], last_iteration=last_iteration)
        label_points.append((label_iteration, setting["mean_curve"][label_iteration]))
        label_value = setting["mean_curve"][label_iteration]
        label_texts.append(f"{setting['label']} "
                           f"{format_percentage(value=label_value) if in_percent else f'{label_value:.4f}'}")

    # Fix the limits before placing the direct labels, whose spacing is computed on the screen
    value_margin = 0.04 * (max(plotted_values) - min(plotted_values) or 1.0)
    curve_axis.set_ylim(min(plotted_values) - value_margin, max(plotted_values) + value_margin)
    curve_axis.set_xlim(0, max(plotted_iterations) * 1.03)

    # Direct labels on a backing of the surface colour, readable over the launches trained beyond the compared iteration
    for label_point, label_text, offset in zip(label_points, label_texts,
                                               spread_label_offsets(axis=curve_axis, label_points=label_points)):
        curve_axis.annotate(text=label_text, xy=label_point, xytext=(6, offset), textcoords="offset points",
                            va="center", color=TEXT_PRIMARY_COLOR, fontsize=9,
                            bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE_COLOR, "edgecolor": "none",
                                  "alpha": 0.85})

    # Mark the compared iteration when some launches trained beyond it
    if max(plotted_iterations) > last_iteration:
        for axis in [curve_axis, gap_axis]:
            axis.axvline(x=last_iteration, color=TEXT_SECONDARY_COLOR, linewidth=1, linestyle=":")
        curve_axis.annotate(text=f"compared at {last_iteration}", xy=(last_iteration, 1.0),
                            xycoords=("data", "axes fraction"), xytext=(4, -10), textcoords="offset points",
                            color=TEXT_SECONDARY_COLOR, fontsize=8)

    # Gap of every compared setting to the reference, with its value where the stage compares it
    gap_axis.axhline(y=0, color=TEXT_SECONDARY_COLOR, linewidth=1)
    largest_gap, gap_label_points, gap_label_texts = 0.0, [], []
    for setting in setting_curves[1:]:
        gap_curve = compute_gap_curve(setting_curve=setting["mean_curve"],
                                      reference_curve=reference_setting["mean_curve"])
        if not gap_curve:
            continue
        gap_axis.plot(list(gap_curve), list(gap_curve.values()), color=setting["color"], linewidth=2, marker="o",
                      markersize=3.5)
        gap_iteration = find_label_iteration(curve=gap_curve, last_iteration=last_iteration)
        gap_label_points.append((gap_iteration, gap_curve[gap_iteration]))
        gap_value = gap_curve[gap_iteration]
        gap_label_texts.append(format_points(difference=gap_value) if in_percent else f"{gap_value:+.4f}")
        largest_gap = max(largest_gap, max(abs(gap) for gap in gap_curve.values()))

    # Titles, axis labels and the recessive style
    curve_axis.set_title(column_title, color=TEXT_PRIMARY_COLOR, fontsize=12, loc="left")
    curve_axis.set_ylabel(f"Test {metric_title}", color=TEXT_SECONDARY_COLOR)
    # The unit of the gaps on a line of its own, so that a long reference label still fits the height of the panel
    gap_axis.set_ylabel(f"Gap to {reference_setting['label']}" + ("\n(points)" if in_percent else ""),
                        color=TEXT_SECONDARY_COLOR)

    # Rates on percent axes, their gaps in points
    if in_percent:
        curve_axis.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))
        gap_axis.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_points_tick))
    gap_axis.set_xlabel("Iteration", color=TEXT_SECONDARY_COLOR)
    for axis in [curve_axis, gap_axis]:
        style_axis(axis=axis)
    return largest_gap, gap_label_points, gap_label_texts


def make_figure_name(comparison: Dict[str, Any]) -> str:
    """
    Names the figure of a comparison after its stage and title, so that the figures of an experiment sort by stage.

    Args:
        comparison (Dict[str, Any]): The comparison of the registry.

    Returns:
        str: The file name, such as stage_1_learning_rate.png.
    """
    title_words = re.sub(pattern=r"[^a-z0-9]+", repl="_", string=comparison["title"].lower()).strip("_")
    return f"stage_{comparison['stage']}_{title_words}.png"


def wrap_verdict(verdict: str, width: int) -> str:
    """
    Wraps the verdict of a stage written under the title of its comparison figure, keeping every percentage and gap in
    points with its unit ("85.81 %", "+0.95 points") on one line, since textwrap breaks at any space.

    Args:
        verdict (str): The verdict of the stage, from the registry.
        width (int): The largest number of characters of a line.

    Returns:
        str: The wrapped verdict, its lines separated by newlines.
    """
    # A non-breaking space between a number and its unit, which textwrap does not take as a break
    return textwrap.fill(text=VERDICT_UNIT_PATTERN.sub(repl=" ", string=verdict), width=width)


def plot_comparison(comparison: Dict[str, Any], benchmarking_folder: Path, output_folder: Path) -> Path:
    """
    Draws the figure of one comparison of the registry: one column per kind of weights evaluated for every launch of
    the stage (trained, then averaged), the curves above and the gaps to the reference setting below.

    Args:
        comparison (Dict[str, Any]): The comparison of the registry.
        benchmarking_folder (Path): The folder of the results files.
        output_folder (Path): The folder of the figures, one subfolder per experiment.

    Returns:
        Path: The written figure.

    Raises:
        ValueError: If no kind of weights was evaluated for every launch of the stage.
    """
    metric_prefix, launch_curves = read_results_curves(results_path=benchmarking_folder / comparison["results_file"])
    weights_kinds = [weights for weights in WEIGHTS_TITLES
                     if all((launch, find_source_weights(arm=arm, weights=weights)) in launch_curves
                            for arm in comparison["arms"] for launch in arm["launches"])]
    if not weights_kinds:
        raise ValueError(f"No weights were evaluated for every launch of stage {comparison['stage']} of "
                         f"{comparison['experiment']} in {comparison['results_file']}.")

    # One column per kind of weights, the curves above the gaps
    figure = Figure(figsize=(7.2 * len(weights_kinds), 8.8), facecolor=SURFACE_COLOR)
    FigureCanvasAgg(figure=figure)
    axes = figure.subplots(nrows=2, ncols=len(weights_kinds), sharex="col", squeeze=False,
                           gridspec_kw={"height_ratios": [2.2, 1]})
    largest_gaps, gap_labels = [], []
    for column_index, weights in enumerate(weights_kinds):
        column_title = WEIGHTS_TITLES[weights] + (", compared" if weights == comparison["compared_weights"] else "")
        setting_curves = collect_setting_curves(comparison=comparison, launch_curves=launch_curves, weights=weights)
        largest_gap, gap_label_points, gap_label_texts = draw_weights_column(
            curve_axis=axes[0][column_index], gap_axis=axes[1][column_index], setting_curves=setting_curves,
            last_iteration=comparison["last_iteration"], column_title=column_title,
            metric_title=METRIC_TITLES[metric_prefix], in_percent=metric_prefix in PERCENT_METRICS)
        largest_gaps.append(largest_gap)
        gap_labels.append((gap_label_points, gap_label_texts))

    # Same gap scale in every column, so that the noise of the two kinds of weights can be compared, then the gap
    # labels spread apart on that scale
    shared_gap_limit = 1.15 * max(max(largest_gaps), 1e-3)
    for column_index, (gap_label_points, gap_label_texts) in enumerate(gap_labels):
        gap_axis = axes[1][column_index]
        gap_axis.set_ylim(-shared_gap_limit, shared_gap_limit)
        for label_point, label_text, offset in zip(gap_label_points, gap_label_texts,
                                                   spread_label_offsets(axis=gap_axis, label_points=gap_label_points)):
            gap_axis.annotate(text=label_text, xy=label_point, xytext=(6, offset), textcoords="offset points",
                              va="center", color=TEXT_PRIMARY_COLOR, fontsize=9,
                              bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE_COLOR, "edgecolor": "none",
                                    "alpha": 0.85})

    # Legend of the settings, the line styles of the launches in order (repeating past the fourth launch), the title
    # of the stage and the verdict of the plan
    longest_arm = max(len(arm["launches"]) for arm in comparison["arms"])
    style_order = ", ".join(LAUNCH_STYLE_NAMES[:longest_arm]) + (" in order" if longest_arm <= len(LAUNCH_LINE_STYLES)
                                                                 else " in order, repeating")
    axes[0][0].legend(loc="best", frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY_COLOR, title_fontsize=8,
                      title=f"thin: each launch ({style_order})")
    figure.suptitle(f"{EXPERIMENT_TITLES[comparison['experiment']]}, stage {comparison['stage']}: "
                    f"{comparison['title']}", color=TEXT_PRIMARY_COLOR, fontsize=13, x=0.01, y=0.985, ha="left")
    verdict_text = wrap_verdict(verdict=comparison["verdict"], width=int(24 * 7.2 * len(weights_kinds) / 1.6))
    figure.text(x=0.01, y=0.955, s=verdict_text, color=TEXT_SECONDARY_COLOR, fontsize=9, ha="left", va="top")
    figure.tight_layout(rect=(0, 0, 1, 0.95 - 0.018 * verdict_text.count("\n")))

    # Write the figure in the folder of its experiment
    figure_path = output_folder / comparison["experiment"] / make_figure_name(comparison=comparison)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=130, facecolor=SURFACE_COLOR)
    return figure_path


def plot_registry(registry_path: Path, output_folder: Path, experiment: Optional[str] = None,
                  stage: Optional[str] = None) -> List[Path]:
    """
    Draws the figures of every comparison of the registry, or of one experiment or stage, the results files being read
    from the folder of the registry.

    Args:
        registry_path (Path): The comparison registry (experiment_comparisons.yaml).
        output_folder (Path): The folder of the figures, one subfolder per experiment.
        experiment (Optional[str]): Draws only the comparisons of this experiment when given.
        stage (Optional[str]): Draws only the comparisons of this stage when given.

    Returns:
        List[Path]: The written figures.

    Raises:
        ValueError: If no comparison of the registry matches the experiment and stage asked for.
    """
    with registry_path.open(encoding="utf-8") as registry_file:
        comparisons = yaml.safe_load(stream=registry_file)["comparisons"]
    selected_comparisons = [comparison for comparison in comparisons
                            if (experiment is None or comparison["experiment"] == experiment)
                            and (stage is None or str(comparison["stage"]) == stage)]
    if not selected_comparisons:
        raise ValueError(f"No comparison of {registry_path} matches the experiment {experiment} and stage {stage}.")
    return [plot_comparison(comparison=comparison, benchmarking_folder=registry_path.parent,
                            output_folder=output_folder) for comparison in selected_comparisons]


def parse_comparison_arguments(arguments: List[str]) -> argparse.Namespace:
    """
    Reads the settings of plot_registry from the command line of the script, so that the figure of a new stage is drawn
    without editing the file: the registry and the figures folder default to DEFAULT_REGISTRY_PATH and
    DEFAULT_OUTPUT_FOLDER, those of the repository, and without an experiment and a stage every stage of every
    experiment is drawn again. A '~' left by the shell at the start of a path ('--registry_path=~/...') is expanded to
    the home folder.

    Args:
        arguments (List[str]): The command line arguments, without the script name.

    Returns:
        argparse.Namespace: The registry_path, output_folder, experiment and stage of plot_registry.

    Raises:
        SystemExit: After printing the help with --help, or the usage and the error when the experiment is not one of
            EXPERIMENT_TITLES.
    """
    # The description and the examples keep their line breaks, argparse wrapping the help of every argument
    parser = argparse.ArgumentParser(
        prog="uv run python -m scripts.evaluation.plot_experiment_comparisons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Draws the figure of every compared stage of the hyperparameter experiments, from the results\n"
                    "files next to the experiment plans and the stages of the comparison registry.",
        epilog="Example, stage 3 of the instance segmentation experiment alone:\n"
               "  uv run python -m scripts.evaluation.plot_experiment_comparisons \\\n"
               "      --experiment instance_segmentation --stage 3")
    # The registry of the stages and the folder of the figures
    parser.add_argument("--registry_path", type=Path, default=DEFAULT_REGISTRY_PATH,
                        help="Registry of the compared stages, the results files being read from its folder (default: "
                             "documentation/architecture_benchmarking/experiment_comparisons.yaml of the repository).")
    parser.add_argument("--output_folder", type=Path, default=DEFAULT_OUTPUT_FOLDER,
                        help="Folder of the figures, one subfolder per experiment (default: "
                             "documentation/architecture_benchmarking/figures of the repository).")
    # The stages drawn
    parser.add_argument("--experiment", choices=list(EXPERIMENT_TITLES), default=None,
                        help="Experiment whose stages are drawn alone (default: every experiment).")
    parser.add_argument("--stage", default=None,
                        help="Stage drawn alone, as the registry names it, such as 3 or 2b (default: every stage).")
    parsed_arguments = parser.parse_args(args=arguments)

    # A '~' the shell leaves after '=' points to the home folder, as in the other scripts of the repository
    for path_name in ["registry_path", "output_folder"]:
        setattr(parsed_arguments, path_name, getattr(parsed_arguments, path_name).expanduser())
    return parsed_arguments


if __name__ == "__main__":
    # Registry, figures folder, experiment and stage read from the command line, every stage of every experiment drawn
    # when none is given
    comparison_arguments = parse_comparison_arguments(arguments=sys.argv[1:])

    # Draw the figures and print a link to each
    figure_paths = plot_registry(registry_path=comparison_arguments.registry_path,
                                 output_folder=comparison_arguments.output_folder,
                                 experiment=comparison_arguments.experiment, stage=comparison_arguments.stage)
    print_color(color="green", output=f"{len(figure_paths)} comparison figures written:")
    for written_path in figure_paths:
        print_color(color="blue", output=format_path_link(path=written_path), indent=1)
