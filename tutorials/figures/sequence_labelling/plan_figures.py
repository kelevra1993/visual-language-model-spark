"""
Benchmark figure of the sequence labelling tutorials (the chapter on reading an evaluation), made at every build by
tutorials/figures/sequence_labelling/figures.py from the results of the sequence labelling benchmark
(documentation/architecture_benchmarking/sequence_labelling_results.csv, the copy of the results.csv of
documentation/architecture_benchmarking/sequence_labelling_experiment_plan.md named under repository_files of
tutorial_sources.yaml): the test macro F1 of the weights the run of the series was evaluated with (the averaged
weights) at every evaluation of three pairs of launches on the same synthetic changing shapes, trained from scratch
without class weights (one of them held at the score of a model predicting no_change on every frame), from scratch
with balanced class weights, and with the settings of the run of the series (balanced weights, pre-trained backbone,
bidirectional LSTM, flatten pooling), drawn with the macro F1 of a model predicting no_change on every frame of the
test split and the number of iterations of the run, so that the chapter compares the run with the benchmark through
measured values only. The settings every group is named after are checked against the columns of the results file,
and the collapsed launch against the macro F1 of predicting no_change everywhere, as the narration says.
"""

import numpy
import pandas

from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Any, Dict, List

from tutorial_builder.figure_context import find_repository_root
from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, MONOSPACE_FONT_FAMILY,
                                                POINTS_PER_PIXEL, SERIES_COLOURS, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR,
                                                TEXT_SECONDARY_COLOUR, compute_pixel_box, find_repository_file,
                                                format_thousands, register_fonts, round_fact, save_figure,
                                                style_chart_panel)
from tutorials.figures.sequence_labelling.sequence_utilities import read_run_sections
from vision_training.evaluation.report_formatting import format_percentage_tick

# Key of the results of the benchmark under repository_files of tutorial_sources.yaml
RESULTS_FILE_KEY = "sequence_labelling_benchmark_results"

# Value of the evaluated_weights column of the results for the averaged and for the trained weights, keyed by whether
# the run of the series was evaluated with its averaged weights, and the name of those weights in the figure header
RESULTS_WEIGHTS = {True: "ema", False: "trained"}
WEIGHTS_HEADER_NAMES = {True: "averaged weights", False: "trained weights"}

# Launches drawn, by the pair of the experiment plan they come from: stage 2 trained from scratch without class weights
# and with balanced weights, and stage 13 (flatten) with the settings kept by the plan, those of the run of the series;
# the columns of the results every label states, an empty pretrained_backbone being a launch from random weights, and
# for the last group the run folder and learning rate of the run of the series (check_launch_settings)
LAUNCH_GROUPS = [{"name": "scratch_none", "label": "from scratch, no class weights",
                  "launches": ["run_11", "run_12"], "colour": SERIES_COLOURS[3],
                  "settings": {"class_weighting": "none", "pretrained_backbone": ""}, "run_settings": False},
                 {"name": "scratch_balanced", "label": "from scratch, balanced weights",
                  "launches": ["run_9", "run_10"], "colour": SERIES_COLOURS[2],
                  "settings": {"class_weighting": "balanced", "pretrained_backbone": ""}, "run_settings": False},
                 {"name": "pretrained", "label": "pre-trained, the settings of this run",
                  "launches": ["run_49", "run_50"], "colour": SERIES_COLOURS[1], "settings": {}, "run_settings": True}]

# Largest difference between the highest macro F1 of the collapsed launch and the macro F1 of predicting no_change
# everywhere, half a unit of the sixth decimal the results file is written with
COLLAPSE_TOLERANCE = 0.5e-6

# Prefix of the columns of the macro F1 at every evaluation, "macro_f1_2000"
MACRO_F1_COLUMN_PREFIX = "macro_f1_"

# Colour of the reference lines (the trivial macro F1 and the iterations of the run)
REFERENCE_COLOUR = SERIES_COLOURS[4]


def read_launch_curves(results: pandas.DataFrame, launches: List[str], evaluated_weights: str) -> Dict[str, Any]:
    """
    Reads the test macro F1 of the given weights of the given launches at every evaluation they reached from the
    results of the benchmark, the empty cells of a launch stopped earlier being left out.

    Args:
        results (pandas.DataFrame): The rows of the results file.
        launches (List[str]): The launch names, e.g. ["run_9", "run_10"].
        evaluated_weights (str): The value of the evaluated_weights column read, "ema" or "trained".

    Returns:
        Dict[str, Any]: The 'iterations' and the macro F1 'values' of every launch, keyed by its name.

    Raises:
        KeyError: If a launch has no row of those weights in the results.
    """
    # Columns of the macro F1 at every evaluation, named after their iteration
    iteration_columns = [column for column in results.columns if column.startswith(MACRO_F1_COLUMN_PREFIX) and
                         column[len(MACRO_F1_COLUMN_PREFIX):].isdigit()]
    curves = {}

    # Row of every launch for the weights read, its empty cells after the last evaluation it reached left out
    for launch in launches:
        rows = results[(results["launch"] == launch) & (results["evaluated_weights"] == evaluated_weights)]
        if rows.empty:
            raise KeyError(f"No row of {launch} with {evaluated_weights} weights in the sequence labelling results.")
        values = rows.iloc[0][iteration_columns].astype(float)
        kept = values.notna().to_numpy()
        curves[launch] = {"iterations": numpy.array([int(column[len(MACRO_F1_COLUMN_PREFIX):])
                                                     for column in iteration_columns])[kept],
                          "values": values.to_numpy()[kept]}
    return curves


def check_launch_settings(results: pandas.DataFrame, run_folder: Path) -> None:
    """
    Checks that the launches of every group were trained with the settings its label and the narration give them
    (no class weights or balanced weights from random weights, or the settings of the run of the series), read from
    the columns of the results file, so that a launch list edited by hand cannot draw a curve under a wrong label.

    Args:
        results (pandas.DataFrame): The rows of the results file.
        run_folder (Path): The folder of the run of the series, whose name gives its settings (build_run_name).

    Raises:
        ValueError: If a launch of a group was trained with other settings.
    """
    experiment_configuration, _ = read_run_sections(run_folder=run_folder)
    for group in LAUNCH_GROUPS:
        # Settings of the label, and for the group of the run its run folder and learning rate
        expected_settings: Dict[str, Any] = dict(group["settings"])
        if group["run_settings"]:
            expected_settings.update({"run_folder": run_folder.name,
                                      "learning_rate": float(experiment_configuration["learning_rate"])})

        # Every row of the launches of the group, an empty cell read as an empty setting
        for _, row in results[results["launch"].isin(group["launches"])].iterrows():
            for column_name, expected_value in expected_settings.items():
                value = "" if pandas.isna(row[column_name]) else row[column_name]
                if value != expected_value:
                    raise ValueError(f"The launch {row['launch']} of the group '{group['label']}' has {column_name} "
                                     f"{value!r} in the sequence labelling results, the label expects "
                                     f"{expected_value!r}.")


def read_launch_value(curve: Dict[str, Any], launch: str, iteration: int) -> float:
    """
    Reads the macro F1 of a launch at one evaluation from its curve (read_launch_curves), for the facts comparing the
    groups at the iterations of the run and at their last common evaluation.

    Args:
        curve (Dict[str, Any]): The 'iterations' and 'values' of the launch.
        launch (str): The name of the launch, for the error message.
        iteration (int): The evaluated iteration.

    Returns:
        float: The macro F1.

    Raises:
        ValueError: If the launch has no evaluation at that iteration.
    """
    values = curve["values"][curve["iterations"] == iteration]
    if len(values) == 0:
        raise ValueError(f"The launch {launch} of the sequence labelling results has no macro F1 at iteration "
                         f"{iteration}, its evaluations being {curve['iterations'].tolist()}.")
    return float(values[0])


def draw_benchmark_curves(group_curves: Dict[str, Dict[str, Any]], trivial_macro_f1: float, run_iterations: int,
                          run_macro_f1: float, header: str, output_path: Path) -> Dict[str, Any]:
    """
    Draws the macro F1 curves of the launch groups, every launch of a group in the colour of the group, with the macro
    F1 of a model predicting no_change on every frame and the number of iterations of the run as dashed references,
    the macro F1 of the run of the series marked at its last iteration, and gives the box of the panel.

    Args:
        group_curves (Dict[str, Dict[str, Any]]): The curves of every launch of every group, keyed by group name.
        trivial_macro_f1 (float): The macro F1 of a model predicting no_change on every test frame.
        run_iterations (int): The number of iterations the run of the series was trained for.
        run_macro_f1 (float): The test macro F1 of the run of the series at its last evaluation.
        header (str): The line above the chart, naming the results file, the test split and the weights.
        output_path (Path): The PNG file.

    Returns:
        Dict[str, Any]: The box of the panel, named panel_box.
    """
    register_fonts()
    figure, axes = pyplot.subplots(nrows=1, ncols=1, figsize=(CHART_WIDTH / DOTS_PER_INCH,
                                                              CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH)
    figure.patch.set_facecolor(SURFACE_COLOUR)
    figure.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.12)
    figure.text(x=0.012, y=0.955, s=header, color=TEXT_SECONDARY_COLOUR, fontsize=16 * POINTS_PER_PIXEL, va="center",
                family=MONOSPACE_FONT_FAMILY)

    # Both launches of every group, the first one labelled for the legend
    for group in LAUNCH_GROUPS:
        for launch_index, (launch, curve) in enumerate(group_curves[group["name"]].items()):
            axes.plot(curve["iterations"], curve["values"], color=group["colour"], linewidth=3 * POINTS_PER_PIXEL,
                      marker="o", markersize=5 * POINTS_PER_PIXEL, label=group["label"] if launch_index == 0 else None)

    # References: the macro F1 of predicting no_change everywhere, and the iterations of the run of the series
    axes.axhline(y=trivial_macro_f1, color=REFERENCE_COLOUR, linewidth=2 * POINTS_PER_PIXEL, linestyle=(0, (6, 5)),
                 label="no_change on every frame")
    axes.axvline(x=run_iterations, color=TEXT_SECONDARY_COLOUR, linewidth=2 * POINTS_PER_PIXEL, linestyle=(0, (2, 4)))
    axes.plot([run_iterations], [run_macro_f1], linestyle="none", marker="*", markersize=26 * POINTS_PER_PIXEL,
              color=SERIES_COLOURS[0], zorder=5, label="the run of this series")
    style_chart_panel(axes=axes, title="Test macro F1 of the benchmark launches", x_label="Iteration",
                      y_label="Macro F1")
    axes.set_ylim(bottom=0.0, top=0.85)
    axes.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))
    axes.xaxis.set_major_formatter(formatter=FuncFormatter(func=format_thousands))
    axes.legend(loc="center right", frameon=False, labelcolor=TEXT_PRIMARY_COLOUR, fontsize=18 * POINTS_PER_PIXEL)
    save_figure(figure=figure, output_path=output_path)
    facts = {"panel_box": compute_pixel_box(figure=figure, axes=axes, with_labels=True, margin=0)}
    pyplot.close(fig=figure)
    return facts


def make_plan_figures(output_folder: Path, run_folder: Path, trivial_macro_f1: float, run_iterations: int,
                      run_macro_f1: float, averaged_weights: bool, test_split: str) -> Dict[str, Any]:
    """
    Makes the benchmark figure of the sequence labelling series and gives its facts: the mean macro F1 of every group
    at the iterations of the run and at the last evaluation of its launches, and the lowest and highest macro F1 of
    the launches without class weights over all their evaluations, the lowest being the score a launch held at when it
    only predicts no_change.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the run of the series, whose settings the last group must have.
        trivial_macro_f1 (float): The macro F1 of a model predicting no_change on every test frame.
        run_iterations (int): The number of iterations the run of the series was trained for.
        run_macro_f1 (float): The test macro F1 of the run of the series at its last evaluation.
        averaged_weights (bool): Whether the run was evaluated with its averaged weights, the weights of the launches
            read.
        test_split (str): The name of the test split of the evaluation of the run, e.g. "test_10000".

    Returns:
        Dict[str, Any]: The facts, named plan.*.

    Raises:
        ValueError: If a launch was trained with other settings than its group, misses an evaluation, or if the
            collapsed launch did not stay at the macro F1 of predicting no_change everywhere.
    """
    results_path = find_repository_file(file_key=RESULTS_FILE_KEY)
    results = pandas.read_csv(filepath_or_buffer=results_path)
    check_launch_settings(results=results, run_folder=run_folder)
    group_curves = {group["name"]: read_launch_curves(results=results, launches=group["launches"],
                                                      evaluated_weights=RESULTS_WEIGHTS[averaged_weights])
                    for group in LAUNCH_GROUPS}
    header = (f"{results_path.relative_to(find_repository_root())} · {test_split} · "
              f"{WEIGHTS_HEADER_NAMES[averaged_weights]}")
    facts = {f"plan.{fact_name}": value for fact_name, value in draw_benchmark_curves(
        group_curves=group_curves, trivial_macro_f1=trivial_macro_f1, run_iterations=run_iterations,
        run_macro_f1=run_macro_f1, header=header,
        output_path=output_folder / "evaluation" / "benchmark_curves.png").items()}

    # Mean of the two launches of every group at the iterations of the run and at their last common evaluation
    for group in LAUNCH_GROUPS:
        curves = group_curves[group["name"]]
        last_iteration = int(min(curve["iterations"].max() for curve in curves.values()))
        facts[f"plan.{group['name']}_at_run_iterations"] = round_fact(value=numpy.mean(
            [read_launch_value(curve=curve, launch=launch, iteration=run_iterations)
             for launch, curve in curves.items()]))
        facts[f"plan.{group['name']}_last"] = round_fact(value=numpy.mean(
            [read_launch_value(curve=curve, launch=launch, iteration=last_iteration)
             for launch, curve in curves.items()]))
        facts[f"plan.{group['name']}_last_iteration"] = last_iteration
        facts[f"plan.{group['name']}_launch_count"] = len(curves)

    # The launch without class weights that never left the score of predicting no_change: its highest macro F1, which
    # the narration calls the macro F1 of predicting no_change everywhere
    collapsed_maximum = min(float(curve["values"].max()) for curve in group_curves["scratch_none"].values())
    if abs(collapsed_maximum - trivial_macro_f1) > COLLAPSE_TOLERANCE:
        raise ValueError(f"No launch from scratch without class weights stayed at the macro F1 of predicting "
                         f"no_change everywhere ({trivial_macro_f1}): the lowest highest macro F1 is "
                         f"{collapsed_maximum}.")
    facts.update({"plan.collapsed_launch_maximum": round_fact(value=collapsed_maximum),
                  "plan.run_iterations": run_iterations})
    return facts
