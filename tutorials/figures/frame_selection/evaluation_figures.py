"""
Evaluation figures of the frame selection tutorials (the chapter on the evaluation and the deployment), made at every
build by tutorials/figures/frame_selection/figures.py from the evaluation folders of the real frame selection run: the
listing of the evaluation block of iteration 1000 in Results.txt, and the probability of every frame chosen by the
decoding of the run (selections.csv) for every wanted class at iterations 1000 and 2000, the right and wrong choices
apart, against the confidence threshold of the class in class_thresholds.json, found with that decoding, so that the
scene shows which wrong choices the deployed decoder flags as low confidence and which go through.
"""

import json

import matplotlib
import numpy

from matplotlib import pyplot
from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, MONOSPACE_FONT_FAMILY,
                                                POINTS_PER_PIXEL, SERIES_COLOURS, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR,
                                                TEXT_SECONDARY_COLOUR, compute_pixel_box, register_fonts, round_fact,
                                                save_figure, style_chart_panel)
from tutorials.figures.frame_selection.selection_figures import read_selection_rows
from tutorials.figures.sequence_labelling.sequence_utilities import (parse_block_metrics, read_results_block,
                                                                     read_run_sections, write_listing)
from tutorial_builder.layout import NAMED_COLOURS

# Iterations of the evaluations compared, the one with wrong choices and the last one
THRESHOLD_ITERATIONS = [1000, 2000]

# Iteration of the evaluation block listed in the report scene
LISTED_ITERATION = 1000

# Characters kept of every listed line, the header line ending with a rule as wide as the console
LISTED_LINE_LENGTH = 118

# Colours of the right and wrong choices, and of the thresholds
RIGHT_COLOUR = SERIES_COLOURS[1]
WRONG_COLOUR = "#%02x%02x%02x" % NAMED_COLOURS["red"]
THRESHOLD_COLOUR = SERIES_COLOURS[2]


def draw_threshold_panel(axes: matplotlib.axes.Axes, selection_rows: Dict[Any, Dict[str, str]],
                         thresholds: Dict[str, Any], class_names: List[str], iteration: int) -> Dict[str, Any]:
    """
    Draws one evaluation on a panel: for every wanted class (one row), the probability of every frame the decoding of
    the thresholds (the decoding of the run) chose for it, right choices and wrong choices in their colours, and the
    threshold of the class as a tick with its value, and gives the facts of the panel.

    Args:
        axes (matplotlib.axes.Axes): The panel.
        selection_rows (Dict[Any, Dict[str, str]]): The rows of selections.csv of the evaluation.
        thresholds (Dict[str, Any]): The class_thresholds.json of the evaluation, with the decoding they were found
            with.
        class_names (List[str]): The wanted classes, one row each from the top.
        iteration (int): The evaluated iteration.

    Returns:
        Dict[str, Any]: The thresholds, recalls and choices of every class, named thresholds.iteration_<n>_*.
    """
    facts = {}
    random_generator = numpy.random.default_rng(seed=0)
    decoding = thresholds["decoding"]
    for row_index, class_name in enumerate(class_names):
        # Choices of the decoding the thresholds were found with, in the columns of selections.csv named after it
        rows = [row for row in selection_rows.values() if row["class"] == class_name]
        chosen_probabilities = numpy.array([float(row[f"{decoding}_probabilities"]) for row in rows])
        right_choices = numpy.array([row[f"{decoding}_right_selections"] == "1" for row in rows])
        flagged_choices = numpy.array([row[f"{decoding}_low_confidence"] == "1" for row in rows])

        # Every right choice as a dot jittered around the row of its class, the wrong ones larger, on the row and on top
        row_position = len(class_names) - 1 - row_index
        jitter = random_generator.uniform(low=-0.18, high=0.18, size=len(rows))
        axes.scatter(x=chosen_probabilities[right_choices], y=row_position + jitter[right_choices], s=60, alpha=0.6,
                     color=RIGHT_COLOUR, edgecolors="none", zorder=3)
        axes.scatter(x=chosen_probabilities[~right_choices], y=numpy.full(shape=int((~right_choices).sum()),
                                                                         fill_value=float(row_position)),
                     s=260, color=WRONG_COLOUR, edgecolors=SURFACE_COLOUR, linewidths=2, zorder=4)

        # Threshold of the class, the lowest probability from which its choices reach the target precision
        class_threshold = thresholds["classes"][class_name]
        if class_threshold is not None:
            axes.plot([class_threshold["threshold"]] * 2, [row_position - 0.34, row_position + 0.34],
                      color=THRESHOLD_COLOUR, linewidth=4 * POINTS_PER_PIXEL, zorder=5)
            axes.text(x=class_threshold["threshold"], y=row_position + 0.38, s=f"{class_threshold['threshold']:.4f}",
                      color=THRESHOLD_COLOUR, fontsize=18 * POINTS_PER_PIXEL, ha="center", va="bottom", zorder=5)
        facts.update({
            f"thresholds.iteration_{iteration}_{class_name}_threshold":
                None if class_threshold is None else round_fact(value=class_threshold["threshold"]),
            f"thresholds.iteration_{iteration}_{class_name}_recall":
                None if class_threshold is None else round_fact(value=class_threshold["recall"]),
            f"thresholds.iteration_{iteration}_{class_name}_wrong": int((~right_choices).sum()),
            f"thresholds.iteration_{iteration}_{class_name}_flagged": int(flagged_choices.sum()),
            f"thresholds.iteration_{iteration}_{class_name}_minimum_right": round_fact(
                value=chosen_probabilities[right_choices].min())})

    # Rows named after their class, the probability axis from 0 to 1
    style_chart_panel(axes=axes, title=f"Iteration {iteration}: probability of the chosen frames",
                      x_label="Probability of the chosen frame for its class", y_label=None)
    axes.set_xlim(left=-0.02, right=1.02)
    axes.set_ylim(bottom=-0.6, top=len(class_names) - 0.2)
    axes.set_yticks(ticks=list(range(len(class_names))), labels=list(reversed(class_names)))
    axes.tick_params(axis="y", labelsize=20 * POINTS_PER_PIXEL, labelcolor=TEXT_PRIMARY_COLOUR)
    return facts


def draw_thresholds_figure(run_folder: Path, output_path: Path) -> Dict[str, Any]:
    """
    Draws the chosen frames and thresholds of the evaluations of THRESHOLD_ITERATIONS side by side and gives their
    facts and the boxes of the panels.

    Args:
        run_folder (Path): The folder of the frame selection run.
        output_path (Path): The PNG file.

    Returns:
        Dict[str, Any]: The facts, named thresholds.*, with the iterations of the panels.
    """
    register_fonts()
    figure, panel_axes = pyplot.subplots(nrows=1, ncols=len(THRESHOLD_ITERATIONS), figsize=(
        CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH, sharey=True)
    figure.patch.set_facecolor(SURFACE_COLOUR)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.79, bottom=0.13, wspace=0.08)

    # Thresholds of every panel, the header writing the decoding and the target precision they were found with
    panel_thresholds = {iteration: json.loads((run_folder / "Evaluation" / f"Iteration_{iteration}" /
                                               "class_thresholds.json").read_text(encoding="utf-8"))
                        for iteration in THRESHOLD_ITERATIONS}
    first_thresholds = panel_thresholds[THRESHOLD_ITERATIONS[0]]
    figure.text(x=0.012, y=0.955, s=f"Evaluation/Iteration_<n>/selections.csv · class_thresholds.json · "
                                    f"{first_thresholds['decoding']} decoding, target precision "
                                    f"{first_thresholds['target_precision']}", color=TEXT_SECONDARY_COLOUR,
                fontsize=16 * POINTS_PER_PIXEL, va="center", family=MONOSPACE_FONT_FAMILY)
    figure.text(x=0.012, y=0.905, s="Green: right choice · red: wrong choice · yellow tick: threshold of the class, "
                                    "a choice under it is flagged low confidence", color=TEXT_SECONDARY_COLOUR,
                fontsize=18 * POINTS_PER_PIXEL, va="center")
    facts = {}
    for axes, iteration in zip(panel_axes, THRESHOLD_ITERATIONS):
        evaluation_folder = run_folder / "Evaluation" / f"Iteration_{iteration}"
        thresholds = panel_thresholds[iteration]
        facts.update(draw_threshold_panel(axes=axes, selection_rows=read_selection_rows(
            selections_path=evaluation_folder / "selections.csv"), thresholds=thresholds,
            class_names=list(thresholds["classes"]), iteration=iteration))
        facts[f"thresholds.iteration_{iteration}_panel_box"] = compute_pixel_box(figure=figure, axes=axes,
                                                                                 with_labels=True, margin=0)
        facts["thresholds.target_precision"] = thresholds["target_precision"]

    # Iterations of the panels, left to right, quoted by the narration instead of typed numbers
    facts["thresholds.iterations"] = THRESHOLD_ITERATIONS
    save_figure(figure=figure, output_path=output_path)
    pyplot.close(fig=figure)
    return facts


def make_evaluation_figures(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Makes the evaluation listing and the thresholds figure of the frame selection series and gives their facts, with
    the top-k of the evaluation of the run.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the frame selection run.

    Returns:
        Dict[str, Any]: The facts, named evaluation.* and thresholds.*.
    """
    block_lines = read_results_block(results_path=run_folder / "Results.txt", iteration=LISTED_ITERATION)
    write_listing(lines=[line[:LISTED_LINE_LENGTH] for line in block_lines],
                  listing_path=output_folder / "listings" / "results_block.txt")
    facts = draw_thresholds_figure(run_folder=run_folder, output_path=output_folder / "evaluation" /
                                   "thresholds.png")

    # Top-k of the evaluation of the run, naming its top-k selection accuracy and the candidates of its decoder
    experiment_configuration, _ = read_run_sections(run_folder=run_folder)
    facts.update({"evaluation.listed_iteration": LISTED_ITERATION,
                  "evaluation.last_iteration": THRESHOLD_ITERATIONS[-1],
                  "evaluation.results_listing_lines": len(block_lines),
                  "evaluation.top_k": int(experiment_configuration["evaluation"]["top_k"])})
    facts.update(parse_block_metrics(block_lines=block_lines, fact_prefix="evaluation"))
    facts.update(parse_block_metrics(block_lines=read_results_block(results_path=run_folder / "Results.txt",
                                                                    iteration=THRESHOLD_ITERATIONS[-1]),
                                     fact_prefix=f"evaluation.iteration_{THRESHOLD_ITERATIONS[-1]}"))
    return facts
