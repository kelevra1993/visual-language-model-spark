"""
Evaluation figures of the self-supervised series of the tutorial videos (chapter 3, judging without a classifier), made
by tutorials/figures/self_supervised/figures.py from the files the SimCLR run wrote at its evaluated checkpoints: the
test metrics of Tensorboard/Test across the checkpoints (k-NN accuracy against its chance level, effective rank,
representation standard deviation), the per-class k-NN F1 of the class_metrics.csv of the last checkpoint for a bars
element, and the block of the last checkpoint in Results.txt, a listing of the scenes. The k-NN rates drawn are written
as the percentages of the reports of the repository ("27.66 %", ticks "20 %"), the facts keeping them between 0 and 1,
while the effective rank and the standard deviation keep their values.
"""

import csv
import math
import textwrap

from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, POINTS_PER_PIXEL,
                                                SERIES_COLOURS, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR,
                                                compute_pixel_box, format_thousands, register_fonts, round_fact,
                                                save_figure, style_chart_panel)
from tutorials.figures.self_supervised.ssl_sources import list_evaluated_iterations
from tutorials.figures.self_supervised.training_figures import (choose_iteration_tick_step, draw_reference_line,
                                                                fit_axis_limits, read_scalars)
from vision_training.evaluation.report_formatting import format_percentage, format_percentage_tick

# Panels of the test metrics: the scalar of Tensorboard/Test, its title, its value axis limits, its fact name and
# whether it is a rate drawn on an axis of percentages
TEST_PANELS = [{"tag": "k-NN Accuracy", "title": "k-NN accuracy", "limits": (0.0, 0.3), "name": "knn_accuracy",
                "in_percent": True},
               {"tag": "Effective Rank", "title": "Effective rank", "limits": (0.0, 32.0), "name": "effective_rank",
                "in_percent": False},
               {"tag": "Representation Standard Deviation", "title": "Representation standard deviation",
                "limits": (0.0, 1.0), "name": "representation_standard_deviation", "in_percent": False}]

# Other global metrics of the evaluations quoted by the narration
OTHER_TEST_TAGS = {"k-NN Balanced Accuracy": "knn_balanced_accuracy", "k-NN Macro F1": "knn_macro_f1"}

# Room left after the longest bar of the per-class F1 bars of the Results.txt scene for its value label, as a factor of
# the best F1, the axis ending at the next tenth above it
BAR_HEADROOM = 1.15

# Width of the lines of the Results.txt listing, in characters of its monospace font, so that its lines end before the
# per-class F1 bars beside it; the longer notes are wrapped, while a longer line of the metrics grid or of the table
# stops the build, a wrapped grid line breaking its columns (the grid lines take exactly 80 characters with two k-NN
# rates of two digits, "27.66 %", and an effective rank of two digits, "40.4249")
RESULTS_LISTING_WIDTH = 80

# Separator of the name and the text of a note of the report ("k-NN : 20 neighbours among ..."), the lines wrapped
NOTE_SEPARATOR = " : "


def draw_test_metrics(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Draws the three global test metrics of the evaluated checkpoints of the run, one panel each with the evaluated
    iterations as points: the k-NN accuracy with the chance level of its classes, the effective rank, and the
    standard deviation of the representations. The iteration ticks are spaced for the length of the run
    (choose_iteration_tick_step), and the values are written above the first checkpoint and the checkpoints on a tick,
    so that the labels of a run evaluated often stay apart.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the self-supervised run.

    Returns:
        Dict[str, Any]: The value of every metric at every evaluated checkpoint, and the boxes of the panels.
    """
    register_fonts()
    number_classes = len(read_class_rows(run_folder=run_folder))
    figure = pyplot.figure(figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH,
                           facecolor=SURFACE_COLOUR)
    facts = {"test.class_count": number_classes, "test.chance_accuracy": round_fact(value=1.0 / number_classes)}
    for panel_index, panel in enumerate(TEST_PANELS):
        axes = figure.add_axes(rect=(0.06 + panel_index * 0.33, 0.14, 0.26, 0.72))
        iterations, values = read_scalars(log_folder=run_folder / "Tensorboard" / "Test", tag=panel["tag"])
        style_chart_panel(axes=axes, title=panel["title"], x_label="Iteration", y_label="Value")
        axes.set_xlim(left=0, right=int(iterations[-1]) + 250)
        bottom, top = fit_axis_limits(limits=panel["limits"], values=[values])
        axes.set_ylim(bottom=bottom, top=top)
        tick_step = choose_iteration_tick_step(last_iteration=int(iterations[-1]))
        axes.set_xticks(ticks=list(range(tick_step, int(iterations[-1]) + 1, tick_step)))
        axes.xaxis.set_major_formatter(formatter=FuncFormatter(func=format_thousands))

        # Ticks of a rate written as the percentages of the reports of the repository, the others keeping their values
        if panel["in_percent"]:
            axes.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))
        if panel["name"] == "knn_accuracy":
            draw_reference_line(axes=axes, value=1.0 / number_classes, label=f"chance: 1 / {number_classes}",
                                right=int(iterations[-1]) + 250)
        axes.plot(iterations, values, color=SERIES_COLOURS[panel_index], linewidth=3 * POINTS_PER_PIXEL, marker="o",
                  markersize=11 * POINTS_PER_PIXEL, solid_capstyle="round")

        # Value of the first checkpoint and of the checkpoints on a tick written above their point, a rate as a
        # percentage, and the value of every checkpoint as a fact
        for iteration, value in zip(iterations, values):
            facts[f"test.{panel['name']}_{int(iteration)}"] = round_fact(value=value)
            if iteration != iterations[0] and int(iteration) % tick_step != 0:
                continue
            if panel["in_percent"]:
                value_label = format_percentage(value=value)
            else:
                value_label = f"{value:.1f}" if panel["name"] == "effective_rank" else f"{value:.3f}"
            axes.text(x=iteration, y=value + (top - bottom) * 0.035, s=value_label, color=TEXT_PRIMARY_COLOUR,
                      fontsize=17 * POINTS_PER_PIXEL, ha="center", va="bottom")
        figure.canvas.draw()
        facts[f"test.{panel['name']}_panel_box"] = compute_pixel_box(figure=figure, axes=axes, with_labels=True,
                                                                   margin=6)

    # The other global metrics of every evaluated checkpoint
    for tag, name in OTHER_TEST_TAGS.items():
        iterations, values = read_scalars(log_folder=run_folder / "Tensorboard" / "Test", tag=tag)
        for iteration, value in zip(iterations, values):
            facts[f"test.{name}_{int(iteration)}"] = round_fact(value=value)
    # The metrics of the first and last evaluated checkpoints under a second name, whatever the length of the run
    for name in [panel["name"] for panel in TEST_PANELS] + list(OTHER_TEST_TAGS.values()):
        facts[f"test.{name}_first"] = facts[f"test.{name}_{int(iterations[0])}"]
        facts[f"test.{name}_last"] = facts[f"test.{name}_{int(iterations[-1])}"]
    facts["test.knn_accuracy_last_over_chance"] = round_fact(
        value=facts[f"test.knn_accuracy_{int(iterations[-1])}"] * number_classes)
    save_figure(figure=figure, output_path=output_folder / "evaluation" / "test_metrics.png")
    pyplot.close(fig=figure)
    return facts


def read_class_rows(run_folder: Path) -> List[Dict[str, str]]:
    """
    Reads the class_metrics.csv of the last evaluated checkpoint of the run: the k-NN precision, recall and F1 of
    every class, in class index order.

    Args:
        run_folder (Path): The folder of the self-supervised run.

    Returns:
        List[Dict[str, str]]: The rows of the file.
    """
    last_iteration = list_evaluated_iterations(run_folder=run_folder)[-1]
    csv_path = run_folder / "Evaluation" / f"Iteration_{last_iteration}" / "class_metrics.csv"
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(f=csv_file))


def read_negative_f1(row: Dict[str, str]) -> float:
    """
    Gives the opposite of the F1 of a row of class_metrics.csv, the sort key ordering the classes best first.

    Args:
        row (Dict[str, str]): A row of the file.

    Returns:
        float: Minus its F1.
    """
    return -float(row["f1"])


def join_class_names(names: List[str]) -> str:
    """
    Writes class names as a spoken list, "the octagon, the pentagon, the circle and the hexagon", for the narration.

    Args:
        names (List[str]): The class names, as in class_metrics.csv.

    Returns:
        str: The spoken list.
    """
    spoken_names = [f"the {name.replace('_', ' ')}" for name in names]
    return ", ".join(spoken_names[:-1]) + " and " + spoken_names[-1]


def describe_class_metrics(run_folder: Path) -> Dict[str, Any]:
    """
    Gives the per-class k-NN F1 of the last evaluated checkpoint, best first, as the values, labels and value labels
    (percentages, "44.55 %") of a bars element, with the largest value of its axis (the next tenth above the best F1
    times BAR_HEADROOM) and the best and worst classes the narration names.

    Args:
        run_folder (Path): The folder of the self-supervised run.

    Returns:
        Dict[str, Any]: The facts, named classes.*.
    """
    rows = sorted(read_class_rows(run_folder=run_folder), key=read_negative_f1)
    return {"classes.f1_bar_maximum": max(0.1, math.ceil(float(rows[0]["f1"]) * BAR_HEADROOM * 10.0) / 10.0),
            "classes.f1_values": [round_fact(value=float(row["f1"])) for row in rows],
            "classes.f1_labels": [row["class"].replace("_", " ") for row in rows],
            "classes.f1_value_labels": [format_percentage(value=float(row["f1"])) for row in rows],
            "classes.best_class": rows[0]["class"].replace("_", " "),
            "classes.best_f1": round_fact(value=float(rows[0]["f1"])),
            "classes.second_class": rows[1]["class"].replace("_", " "),
            "classes.second_f1": round_fact(value=float(rows[1]["f1"])),
            "classes.bottom_four": join_class_names(names=[row["class"] for row in rows[-4:][::-1]]),
            "classes.worst_class": rows[-1]["class"].replace("_", " "),
            "classes.worst_f1": round_fact(value=float(rows[-1]["f1"])),
            "classes.samples_minimum": min(int(row["samples"]) for row in rows),
            "classes.samples_maximum": max(int(row["samples"]) for row in rows)}


def copy_last_results_block(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Copies the block of the last evaluated checkpoint of Results.txt as a listing of the scenes, up to the end of its
    per-class table, its notes longer than RESULTS_LISTING_WIDTH wrapped, and gives its number of test images. The
    rates of the block are written as the percentages of the reports, copied as they are.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the self-supervised run.

    Returns:
        Dict[str, Any]: The facts of the block.

    Raises:
        ValueError: If Results.txt holds no block of the last evaluated checkpoint, or if a line of its metrics grid or
            of its table is longer than RESULTS_LISTING_WIDTH.
    """
    last_iteration = list_evaluated_iterations(run_folder=run_folder)[-1]
    result_lines = (run_folder / "Results.txt").read_text(encoding="utf-8").splitlines()
    block_starts = [line_index for line_index, line in enumerate(result_lines)
                    if line.startswith(f"Iteration {last_iteration} ")]
    if not block_starts:
        raise ValueError(f"No block of iteration {last_iteration} in {run_folder / 'Results.txt'}.")

    # The block from its title, its rule shortened to the width of the scene, to the last row of its table, the notes
    # longer than the listing (the k-NN note of the reference images) wrapped under their first words
    block_lines = []
    for line in result_lines[block_starts[0]:]:
        if line.startswith("  k-nearest neighbour classification"):
            break
        if line.startswith("Iteration"):
            block_lines.append(line.rstrip(" ─"))
        elif len(line) > RESULTS_LISTING_WIDTH and NOTE_SEPARATOR in line:
            indent = " " * (len(line) - len(line.lstrip()))
            block_lines += textwrap.wrap(text=line.strip(), width=RESULTS_LISTING_WIDTH, initial_indent=indent,
                                         subsequent_indent=indent + "  ")
        elif len(line) > RESULTS_LISTING_WIDTH:
            # A line of the metrics grid or of the table, whose columns a wrap would break
            raise ValueError(f"The line {line.strip()!r} of the block of iteration {last_iteration} of "
                             f"{run_folder / 'Results.txt'} takes {len(line)} characters, more than the "
                             f"{RESULTS_LISTING_WIDTH} of the listing of the Results.txt scene of the third "
                             f"self-supervised chapter: widen RESULTS_LISTING_WIDTH and the listing box of the scene, "
                             f"narrowing the per-class F1 bars beside it.")
        else:
            block_lines.append(line)
    listing_path = output_folder / "listings" / "last_results_block.txt"
    listing_path.parent.mkdir(parents=True, exist_ok=True)
    listing_path.write_text(data="\n".join(block_lines).rstrip() + "\n", encoding="utf-8")
    test_images = int(block_lines[0].split("·")[1].split()[0])
    return {"results.last_iteration": last_iteration, "results.test_images": test_images}


def make_evaluation_figures(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Makes the evaluation figures of the series and gives their facts: the test metrics across the checkpoints, the
    per-class F1 and the listing of the last block of Results.txt.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the self-supervised run.

    Returns:
        Dict[str, Any]: The facts, named test.*, classes.* and results.*.
    """
    facts = draw_test_metrics(output_folder=output_folder, run_folder=run_folder)
    facts.update(describe_class_metrics(run_folder=run_folder))
    facts.update(copy_last_results_block(output_folder=output_folder, run_folder=run_folder))
    facts["test.evaluated_iterations"] = list_evaluated_iterations(run_folder=run_folder)
    return facts
