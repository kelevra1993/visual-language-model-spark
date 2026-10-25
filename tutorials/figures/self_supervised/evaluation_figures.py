"""
Evaluation figures of the self-supervised series of the tutorial videos (chapter 3, judging without a classifier), made
by tutorials/figures/self_supervised/figures.py from the files the SimCLR run wrote at its evaluated checkpoints: the
test metrics of Tensorboard/Test across the checkpoints (k-NN accuracy against its chance level, effective rank,
representation standard deviation), the per-class k-NN F1 of the class_metrics.csv of the last checkpoint for a bars
element, and the block of the last checkpoint in Results.txt, a listing of the scenes.
"""

import csv

from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, POINTS_PER_PIXEL,
                                                SERIES_COLOURS, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR,
                                                compute_pixel_box, format_thousands, register_fonts, round_fact,
                                                save_figure, style_chart_panel)
from tutorials.figures.self_supervised.ssl_sources import list_evaluated_iterations
from tutorials.figures.self_supervised.training_figures import draw_reference_line, read_scalars

# Panels of the test metrics: the scalar of Tensorboard/Test, its title, its value axis limits and its fact name
TEST_PANELS = [{"tag": "k-NN Accuracy", "title": "k-NN accuracy", "limits": (0.0, 0.3), "name": "knn_accuracy"},
               {"tag": "Effective Rank", "title": "Effective rank", "limits": (0.0, 32.0), "name": "effective_rank"},
               {"tag": "Representation Standard Deviation", "title": "Representation standard deviation",
                "limits": (0.0, 1.0), "name": "representation_standard_deviation"}]

# Other global metrics of the evaluations quoted by the narration
OTHER_TEST_TAGS = {"k-NN Balanced Accuracy": "knn_balanced_accuracy", "k-NN Macro F1": "knn_macro_f1"}


def draw_test_metrics(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Draws the three global test metrics of the evaluated checkpoints of the run, one panel each with the evaluated
    iterations as points: the k-NN accuracy with the chance level of its classes, the effective rank, and the
    standard deviation of the representations.

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
        axes.set_ylim(bottom=panel["limits"][0], top=panel["limits"][1])
        axes.set_xticks(ticks=[int(iteration) for iteration in iterations])
        axes.xaxis.set_major_formatter(formatter=FuncFormatter(func=format_thousands))
        if panel["name"] == "knn_accuracy":
            draw_reference_line(axes=axes, value=1.0 / number_classes, label=f"chance: 1 / {number_classes}",
                                right=int(iterations[-1]) + 250)
        axes.plot(iterations, values, color=SERIES_COLOURS[panel_index], linewidth=3 * POINTS_PER_PIXEL, marker="o",
                  markersize=11 * POINTS_PER_PIXEL, solid_capstyle="round")

        # Value of every evaluated checkpoint written above its point
        for iteration, value in zip(iterations, values):
            axes.text(x=iteration, y=value + (panel["limits"][1] - panel["limits"][0]) * 0.035,
                      s=f"{value:.1f}" if panel["name"] == "effective_rank" else f"{value:.3f}",
                      color=TEXT_PRIMARY_COLOUR, fontsize=17 * POINTS_PER_PIXEL, ha="center", va="bottom")
            facts[f"test.{panel['name']}_{int(iteration)}"] = round_fact(value=value)
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
    of a bars element, with the best and worst classes the narration names.

    Args:
        run_folder (Path): The folder of the self-supervised run.

    Returns:
        Dict[str, Any]: The facts, named classes.*.
    """
    rows = sorted(read_class_rows(run_folder=run_folder), key=read_negative_f1)
    return {"classes.f1_values": [round_fact(value=float(row["f1"])) for row in rows],
            "classes.f1_labels": [row["class"].replace("_", " ") for row in rows],
            "classes.f1_value_labels": [f"{float(row['f1']):.2f}" for row in rows],
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
    per-class table, and gives its number of test images.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the self-supervised run.

    Returns:
        Dict[str, Any]: The facts of the block.

    Raises:
        ValueError: If Results.txt holds no block of the last evaluated checkpoint.
    """
    last_iteration = list_evaluated_iterations(run_folder=run_folder)[-1]
    result_lines = (run_folder / "Results.txt").read_text(encoding="utf-8").splitlines()
    block_starts = [line_index for line_index, line in enumerate(result_lines)
                    if line.startswith(f"Iteration {last_iteration} ")]
    if not block_starts:
        raise ValueError(f"No block of iteration {last_iteration} in {run_folder / 'Results.txt'}.")

    # The block from its title, its rule shortened to the width of the scene, to the last row of its table
    block_lines = []
    for line in result_lines[block_starts[0]:]:
        if line.startswith("  k-nearest neighbour classification"):
            break
        block_lines.append(line.rstrip(" ─") if line.startswith("Iteration") else line)
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
