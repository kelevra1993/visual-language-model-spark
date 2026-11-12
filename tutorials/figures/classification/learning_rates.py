"""
Learning rate figure of the classification tutorials (the chapter from labels to logits), made at every build by
tutorials/figures/classification/figures.py from the TensorBoard logs of the classification run and of its three
learning rate variants (classification and classification_learning_rates of tutorials/tutorial_sources.yaml, the
launches the README compares in MLflow): the validation accuracy of every iteration of the first 600 iterations of the
four runs, smoothed like TensorBoard, and their mean validation accuracy over the last 50 of these iterations next to
the test accuracy of their first evaluation (600 for the variants, 500 for the run). The facts give these numbers with
the learning rate of every run, read from its saved configuration. The accuracies are written as the percentages of
the reports of the repository ("94.03 %", ticks "20 %"), the facts keeping them between 0 and 1.
"""

import numpy

from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, POINTS_PER_PIXEL,
                                                SERIES_COLOURS, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR,
                                                compute_pixel_box, read_run_configuration, read_tutorial_sources,
                                                register_fonts, round_fact, save_figure, style_chart_panel)
from tutorials.figures.foundations.training_curves import read_scalars, smooth_values
from vision_training.evaluation.report_formatting import format_percentage, format_percentage_tick

# Iterations compared: the length of the variants, and the window of the last iterations averaged
COMPARED_ITERATIONS = 600
AVERAGED_LAST_ITERATIONS = 50

# Smoothing weight of the curves, the exponential moving average of TensorBoard
SMOOTHING_WEIGHT = 0.95

# Name of the figure file
FIGURE_NAME = "learning_rates_validation_accuracy.png"


def list_learning_rate_runs(experiments_folder: Path, run_folder: Path) -> List[Path]:
    """
    Lists the classification run and its learning rate variants of tutorial_sources.yaml, refusing a missing variant
    with what it takes to train it again.

    Args:
        experiments_folder (Path): The folder of the real runs.
        run_folder (Path): The folder of the classification run.

    Returns:
        List[Path]: The run folders, the classification run first.

    Raises:
        FileNotFoundError: If a variant folder is missing.
    """
    variants = read_tutorial_sources()["runs"]["classification_learning_rates"]
    variant_folders = [experiments_folder / folder for folder in variants["folders"]]
    missing_folders = [str(folder) for folder in variant_folders if not folder.is_dir()]
    if missing_folders:
        raise FileNotFoundError(f"Missing learning rate variants {missing_folders}: train them with "
                                f"{variants['trained_with']}.")
    return [run_folder] + variant_folders


def describe_run(run_folder: Path) -> Dict[str, Any]:
    """
    Reads what the figure needs of one run: its learning rate, its validation accuracy of every iteration up to
    COMPARED_ITERATIONS and the test accuracy of its first evaluation.

    Args:
        run_folder (Path): The run folder.

    Returns:
        Dict[str, Any]: The 'learning_rate', the 'iterations' and 'accuracies' of the validation batches, the
            'last_mean' validation accuracy of the last AVERAGED_LAST_ITERATIONS iterations, and the 'test_iteration'
            and 'test_accuracy' of the first evaluation.

    Raises:
        ValueError: If the run stops before COMPARED_ITERATIONS.
    """
    iterations, accuracies = read_scalars(log_folder=run_folder / "Tensorboard" / "Validation", tag="Accuracy")
    kept = iterations <= COMPARED_ITERATIONS
    if iterations[kept].max() < COMPARED_ITERATIONS:
        raise ValueError(f"The validation logs of {run_folder} stop at iteration {iterations.max()}, before "
                         f"{COMPARED_ITERATIONS}.")
    test_iterations, test_accuracies = read_scalars(log_folder=run_folder / "Tensorboard" / "Test", tag="Accuracy")
    last_window = kept & (iterations > COMPARED_ITERATIONS - AVERAGED_LAST_ITERATIONS)
    return {"learning_rate": float(read_run_configuration(run_folder=run_folder)["ExperimentConfiguration"]
                                   ["learning_rate"]),
            "iterations": iterations[kept], "accuracies": accuracies[kept],
            "last_mean": float(accuracies[last_window].mean()), "test_iteration": int(test_iterations[0]),
            "test_accuracy": float(test_accuracies[0])}


def make_learning_rate_figure(output_folder: Path, experiments_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Draws the learning rate figure and gives its facts: the number of runs compared and, for every learning rate, its
    value, the mean validation accuracy of the last iterations, the test accuracy of the first evaluation and its
    iteration, and the boxes of both panels.

    Args:
        output_folder (Path): The figures folder of the series.
        experiments_folder (Path): The folder of the real runs.
        run_folder (Path): The folder of the classification run of tutorial_sources.yaml.

    Returns:
        Dict[str, Any]: The facts, named learning_rates.*.
    """
    runs = sorted([describe_run(run_folder=folder) for folder in
                   list_learning_rate_runs(experiments_folder=experiments_folder, run_folder=run_folder)],
                  key=read_learning_rate)

    # Curve panel on the left, the bars of the two accuracies on the right, wide enough for their percentage labels
    register_fonts()
    figure = pyplot.figure(figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH)
    figure.patch.set_facecolor(SURFACE_COLOUR)
    curve_axes = figure.add_axes(rect=[0.075, 0.13, 0.48, 0.76])
    bar_axes = figure.add_axes(rect=[0.635, 0.13, 0.355, 0.76])
    style_chart_panel(axes=curve_axes, title="Validation accuracy, smoothed", x_label="Iteration", y_label="Accuracy")
    style_chart_panel(axes=bar_axes, title="Validation (last 50) and test", x_label="Learning rate", y_label=None)

    # Validation curve of every run, the raw values faint under the smoothed ones
    for run_index, run in enumerate(runs):
        colour = SERIES_COLOURS[run_index]
        curve_axes.plot(run["iterations"], run["accuracies"], color=colour, alpha=0.12, linewidth=POINTS_PER_PIXEL)
        curve_axes.plot(run["iterations"], smooth_values(values=run["accuracies"], weight=SMOOTHING_WEIGHT),
                        color=colour, linewidth=3 * POINTS_PER_PIXEL, label=f"{run['learning_rate']:g}")
    curve_axes.set_xlim(left=0, right=COMPARED_ITERATIONS)
    curve_axes.set_ylim(bottom=0, top=1)
    curve_axes.axhline(y=1 / 9, color=TEXT_PRIMARY_COLOUR, linestyle=":", linewidth=1.5 * POINTS_PER_PIXEL, alpha=0.6)
    curve_axes.text(x=COMPARED_ITERATIONS - 8, y=1 / 9 - 0.045, s="chance, 1 / 9", color=TEXT_PRIMARY_COLOUR,
                    alpha=0.7, ha="right", fontsize=16 * POINTS_PER_PIXEL)
    curve_axes.legend(loc="upper left", frameon=False, labelcolor=TEXT_PRIMARY_COLOUR, fontsize=18 * POINTS_PER_PIXEL,
                      title="learning rate", title_fontsize=16 * POINTS_PER_PIXEL)
    curve_axes.get_legend().get_title().set_color(TEXT_PRIMARY_COLOUR)

    # Two bars per run: its last validation mean, plain, and its first test accuracy, hatched, with their values; the
    # bars are wide enough for every percentage label to stay above its own bar, clear of the bar next to it
    positions = numpy.arange(len(runs))
    bar_width = 0.45
    for run_index, run in enumerate(runs):
        colour = SERIES_COLOURS[run_index]
        bar_axes.bar(x=positions[run_index] - bar_width / 2, height=run["last_mean"], width=bar_width, color=colour)
        bar_axes.bar(x=positions[run_index] + bar_width / 2, height=run["test_accuracy"], width=bar_width,
                     color=SURFACE_COLOUR, edgecolor=colour, linewidth=2.5 * POINTS_PER_PIXEL, hatch="//")
        for offset, value in [(-bar_width / 2, run["last_mean"]), (bar_width / 2, run["test_accuracy"])]:
            bar_axes.text(x=positions[run_index] + offset, y=value + 0.015, s=format_percentage(value=value),
                          ha="center", color=TEXT_PRIMARY_COLOUR, fontsize=14 * POINTS_PER_PIXEL)
    bar_axes.set_xticks(ticks=positions, labels=[f"{run['learning_rate']:g}" for run in runs])
    bar_axes.set_ylim(bottom=0, top=1)

    # Accuracy ticks of both panels written as the percentages of the reports of the repository
    for axes in [curve_axes, bar_axes]:
        axes.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))
    # Note of the bars: the test iteration of most runs, and the runs evaluated first at another iteration
    test_iterations = [run["test_iteration"] for run in runs]
    usual_iteration = max(set(test_iterations), key=test_iterations.count)
    exceptions = "".join(f", {run['test_iteration']} for {run['learning_rate']:g}" for run in runs
                         if run["test_iteration"] != usual_iteration)
    bar_axes.text(x=-0.5, y=0.97, s=f"plain: validation, last {AVERAGED_LAST_ITERATIONS} iterations\n"
                                    f"hatched: test, iteration {usual_iteration}{exceptions}",
                  color=TEXT_PRIMARY_COLOUR, alpha=0.8, va="top", fontsize=15 * POINTS_PER_PIXEL)
    output_path = output_folder / "training" / FIGURE_NAME
    save_figure(figure=figure, output_path=output_path)
    facts = {"learning_rates.curve_panel_box": compute_pixel_box(figure=figure, axes=curve_axes, with_labels=True,
                                                                 margin=8),
             "learning_rates.bar_panel_box": compute_pixel_box(figure=figure, axes=bar_axes, with_labels=True,
                                                               margin=8),
             "learning_rates.compared_iterations": COMPARED_ITERATIONS,
             "learning_rates.averaged_last_iterations": AVERAGED_LAST_ITERATIONS,
             "learning_rates.run_count": len(runs)}
    pyplot.close(fig=figure)

    # Numbers of every run, named after its learning rate (0.0002 -> lr_0_0002)
    for run in runs:
        run_name = "learning_rates.lr_" + f"{run['learning_rate']:g}".replace(".", "_").replace("-", "_")
        facts.update({f"{run_name}.learning_rate": run["learning_rate"],
                      f"{run_name}.validation_last_mean": round_fact(value=run["last_mean"]),
                      f"{run_name}.test_accuracy": round_fact(value=run["test_accuracy"]),
                      f"{run_name}.test_iteration": run["test_iteration"]})
    return facts


def read_learning_rate(run: Dict[str, Any]) -> float:
    """
    Gives the sort key of the runs of the figure, their learning rate, so that the colours and bars go up with it.

    Args:
        run (Dict[str, Any]): A run read by describe_run.

    Returns:
        float: The learning rate of the run.
    """
    return run["learning_rate"]
