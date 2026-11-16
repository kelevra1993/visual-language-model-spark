"""
Training and evaluation figures of the sequence comparison tutorials (chapter 3), made by
tutorials/figures/sequence_comparison/figures.py from the README run of the task: its launch information and the first
console summary, whose metric lines are written again with their logged values by the console formatter of the current
code (reformat_console_summary of figure_utilities.py) so that the scene shows the columns and units a run trained
today prints, its training and validation curves (an empty-axes base and its overlay, uncovered one panel at a time),
and,
from the evaluations of both checkpoints made again with the current code in its Checkpoint_Evaluation folder, its
Results.txt, the confusion matrix, ROC and precision-recall curves and the precision-recall curves zoomed on the
thresholds of its last evaluation side by side, the probability of same_number_of_objects of every test pair with the
thresholds of the evaluation, and the sample image of a wrong and of a right pair. Its facts are the metrics, losses,
throughputs, thresholds and wrong pairs the narration quotes. The accuracies and the target precision drawn are
written as the percentages of the reports of the repository ("94.03 %", "99 %"), the facts and the probability
thresholds keeping their values between 0 and 1.
"""

import csv
import json

import numpy

from matplotlib import pyplot
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, POINTS_PER_PIXEL,
                                                REVEAL_MARGIN, SERIES_COLOURS, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR,
                                                TEXT_SECONDARY_COLOUR, build_console_metric_tracker,
                                                compute_pixel_box, format_thousands, reformat_console_summary,
                                                register_fonts, round_fact, save_figure, style_chart_panel)
from tutorials.figures.foundations.training_curves import smooth_values
from tutorials.figures.sequence_classification.sequence_figure_utilities import (compose_image_grid,
                                                                                 draw_console_figure,
                                                                                 extract_interval_block,
                                                                                 extract_launch_block,
                                                                                 read_interval_summaries,
                                                                                 read_predictions, read_scalar_curve,
                                                                                 write_text_lines)
from vision_training.evaluation.checkpoint_evaluation import CHECKPOINT_EVALUATION_FOLDER_NAME
from vision_training.evaluation.report_formatting import (format_percentage, format_percentage_label,
                                                          format_percentage_tick)
from vision_training.trainers.sequence_comparison_trainer import SequenceComparisonTrainer

# Smoothing of the training curves, the exponential moving average of TensorBoard
SMOOTHING_WEIGHT = 0.95

# Folder of the evaluations the figures read, inside the run folder: both checkpoints evaluated again with the current
# code by scripts/evaluation/evaluate_checkpoint.py (rates as percentages, top-5 accuracy, Per Class Recall table,
# precision_recall_thresholds.png), the Evaluation folder of the training having been written by an older evaluation
EVALUATION_FOLDER_NAME = CHECKPOINT_EVALUATION_FOLDER_NAME

# Headline scalars of the test evaluations kept as facts, with the names of their facts
TEST_SCALARS = {"Accuracy": "accuracy", "Balanced Accuracy": "balanced_accuracy", "Macro F1": "macro_f1",
                "Macro ROC AUC": "macro_roc_auc", "Macro Average Precision": "macro_average_precision"}

# Sample images compared: a pair the run gets wrong and a pair it gets right, both fixed test samples
WRONG_SAMPLE = "test_000003"
RIGHT_SAMPLE = "test_000000"

# Bins of the histogram of the probability of same_number_of_objects
PROBABILITY_BINS = numpy.linspace(start=0.0, stop=1.0, num=41)


def draw_curve_figure(output_folder: Path, run_folder: Path, class_count: int) -> Dict[str, Any]:
    """
    Draws the total loss and the accuracy of the training and validation batches of the run on two panels, every
    curve smoothed like TensorBoard over its faint raw values, with the test accuracy of every evaluation and the loss
    of a uniform guess over the classes of the run; the empty axes are written as the base image and the whole chart
    as its overlay. Gives the boxes of the panels and their reveal regions, and the first and last values of the
    curves.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The comparison run.
        class_count (int): The number of classes of the run, whose natural log is the loss of a uniform guess.

    Returns:
        Dict[str, Any]: The facts, named training.*.
    """
    curves = {(split, tag): read_scalar_curve(run_folder=run_folder, split=split, tag=tag)
              for split in ["Train", "Validation"] for tag in ["Total Loss", "Accuracy"]}
    # Test accuracies of the evaluations of the current code, whose TensorBoard logs are in Checkpoint_Evaluation
    test_iterations, test_accuracies = read_scalar_curve(run_folder=run_folder / EVALUATION_FOLDER_NAME, split="Test",
                                                         tag="Accuracy")
    last_iteration = int(curves[("Train", "Total Loss")][0][-1])
    register_fonts()
    figure, (loss_axes, accuracy_axes) = pyplot.subplots(nrows=1, ncols=2, figsize=(CHART_WIDTH / DOTS_PER_INCH,
                                                                                   CHART_HEIGHT / DOTS_PER_INCH),
                                                         dpi=DOTS_PER_INCH)
    figure.patch.set_facecolor(SURFACE_COLOUR)
    # Panels apart enough for the percentage ticks of the accuracy
    figure.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.12, wspace=0.2)
    for axes, title, y_label, y_limit in [(loss_axes, "Total loss", "Loss", (0.0, 0.8)),
                                          (accuracy_axes, "Accuracy", "Accuracy", (0.0, 1.05))]:
        style_chart_panel(axes=axes, title=title, x_label="Iteration", y_label=y_label)
        axes.set_xlim(left=0, right=last_iteration)
        axes.set_ylim(bottom=y_limit[0], top=y_limit[1])
        axes.xaxis.set_major_formatter(formatter=FuncFormatter(func=format_thousands))
    # Accuracy ticks written as the percentages of the reports of the repository, the loss keeping its decimals
    accuracy_axes.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))
    # Loss of a uniform guess over the classes of the run, the natural log of their number
    uniform_guess_loss = numpy.log(class_count)
    loss_axes.axhline(y=uniform_guess_loss, color=TEXT_SECONDARY_COLOUR, linewidth=1.5 * POINTS_PER_PIXEL,
                      linestyle="--")
    loss_axes.annotate(text=f"ln {class_count} = {uniform_guess_loss:.3f}, a uniform guess",
                       xy=(last_iteration, uniform_guess_loss), xytext=(-8, 8), textcoords="offset points", ha="right",
                       va="bottom", color=TEXT_SECONDARY_COLOUR, fontsize=17 * POINTS_PER_PIXEL)
    colours = {"Train": SERIES_COLOURS[0], "Validation": SERIES_COLOURS[1], "Test": SERIES_COLOURS[2]}
    legend_handles = [Line2D(xdata=[0], ydata=[0], color=colours["Train"], linewidth=3 * POINTS_PER_PIXEL),
                      Line2D(xdata=[0], ydata=[0], color=colours["Validation"], linewidth=3 * POINTS_PER_PIXEL),
                      Line2D(xdata=[0], ydata=[0], color=colours["Test"], marker="o", linestyle="none",
                             markersize=10 * POINTS_PER_PIXEL)]
    figure.legend(handles=legend_handles, labels=["Training batches", "Validation batches", "Test split"],
                  loc="center left", bbox_to_anchor=(0.05, 0.93), ncols=3, frameon=False,
                  labelcolor=TEXT_PRIMARY_COLOUR, fontsize=18 * POINTS_PER_PIXEL, handlelength=1.6)
    figure.text(x=0.985, y=0.93, s=f"Faint: every iteration · solid: smoothed ({SMOOTHING_WEIGHT})", ha="right",
                va="center", color=TEXT_SECONDARY_COLOUR, fontsize=17 * POINTS_PER_PIXEL)
    save_figure(figure=figure, output_path=output_folder / "evaluation" / "training_curves_axes.png")

    # Curves of both splits on both panels, the test accuracy of every evaluation as points
    for split in ["Train", "Validation"]:
        for axes, tag in [(loss_axes, "Total Loss"), (accuracy_axes, "Accuracy")]:
            iterations, values = curves[(split, tag)]
            axes.plot(iterations, values, color=colours[split], alpha=0.22, linewidth=1.2 * POINTS_PER_PIXEL)
            axes.plot(iterations, smooth_values(values=values, weight=SMOOTHING_WEIGHT), color=colours[split],
                      linewidth=3 * POINTS_PER_PIXEL, solid_capstyle="round")
    accuracy_axes.plot(test_iterations, test_accuracies, color=colours["Test"], marker="o", linestyle="none",
                       markersize=12 * POINTS_PER_PIXEL, markeredgecolor=SURFACE_COLOUR,
                       markeredgewidth=2 * POINTS_PER_PIXEL, zorder=4, clip_on=False)
    for test_iteration, test_accuracy in zip(test_iterations, test_accuracies):
        # The label rounded like the fact the narration quotes, so that the chart and the caption agree
        accuracy_axes.annotate(text=format_percentage(value=round_fact(value=test_accuracy)),
                               xy=(test_iteration, test_accuracy), xytext=(-12, -22), textcoords="offset points",
                               ha="right", va="center", color=TEXT_PRIMARY_COLOUR, fontsize=17 * POINTS_PER_PIXEL)
    save_figure(figure=figure, output_path=output_folder / "evaluation" / "training_curves.png")
    facts = {}
    for panel_name, axes in [("loss", loss_axes), ("accuracy", accuracy_axes)]:
        facts[f"training.{panel_name}_panel_box"] = compute_pixel_box(figure=figure, axes=axes, with_labels=True,
                                                                      margin=0)
        facts[f"training.{panel_name}_reveal_region"] = compute_pixel_box(figure=figure, axes=axes,
                                                                          with_labels=False, margin=REVEAL_MARGIN)
    pyplot.close(fig=figure)
    for (split, tag), (iterations, values) in curves.items():
        curve_name = f"training.{split.lower()}_{tag.lower().replace(' ', '_')}"
        facts[f"{curve_name}.first"] = round_fact(value=values[0])
        facts[f"{curve_name}.last"] = round_fact(value=values[-1])
    facts["training.iterations"] = last_iteration
    return facts


def draw_probability_figure(output_path: Path, predictions: List[Dict[str, str]], thresholds: Dict[str, Any],
                            class_names: List[str]) -> Dict[str, Any]:
    """
    Draws the histogram of the probability of same_number_of_objects of the test pairs, one colour per true class,
    with the thresholds of class_thresholds.json that keep a precision of 0.99 for each class (the threshold of
    loss_of_object read on its own probability, one minus this one); gives the box of the panel.

    Args:
        output_path (Path): The PNG file.
        predictions (List[Dict[str, str]]): The rows of predictions.csv.
        thresholds (Dict[str, Any]): The class_thresholds.json of the evaluation.
        class_names (List[str]): The two classes, loss_of_object then same_number_of_objects.

    Returns:
        Dict[str, Any]: The box, named probabilities.panel_box.
    """
    register_fonts()
    figure, axes = pyplot.subplots(figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH),
                                   dpi=DOTS_PER_INCH)
    figure.patch.set_facecolor(SURFACE_COLOUR)
    figure.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.13)
    same_name = class_names[1]
    bin_width = PROBABILITY_BINS[1] - PROBABILITY_BINS[0]
    for class_index, class_name in enumerate(class_names):
        probabilities = [float(row[f"probability_{same_name}"]) for row in predictions if row["true_class"] ==
                         class_name]
        counts, _ = numpy.histogram(a=probabilities, bins=PROBABILITY_BINS)
        axes.bar(x=PROBABILITY_BINS[:-1] + class_index * bin_width / 2, height=counts, width=bin_width / 2 - 0.002,
                 align="edge", color=SERIES_COLOURS[3] if class_index == 0 else SERIES_COLOURS[1],
                 label=f"true {class_name} ({len(probabilities)})")
    style_chart_panel(axes=axes, title=f"Probability of {same_name}, every test pair",
                      x_label=f"probability_{same_name} of predictions.csv", y_label="Test pairs")
    axes.set_yscale(value="function", functions=(numpy.sqrt, numpy.square))
    axes.set_yticks(ticks=[0, 5, 20, 50, 100])
    axes.set_xlim(left=0.0, right=1.0)
    axes.grid(visible=False, axis="x")

    # Thresholds keeping a precision of 0.99, on the probability of same_number_of_objects
    for class_name, line_style in [(class_names[0], ":"), (same_name, "--")]:
        threshold = thresholds["classes"][class_name]["threshold"]
        position = threshold if class_name == same_name else 1.0 - threshold
        axes.axvline(x=position, color=TEXT_SECONDARY_COLOUR, linewidth=2 * POINTS_PER_PIXEL, linestyle=line_style)
        axes.annotate(text=f"{class_name} at precision "
                           f"{format_percentage_label(value=thresholds['target_precision'])}: "
                           f"p({class_name}) ≥ {threshold:.3f}", xy=(position, 1.0),
                      xycoords=("data", "axes fraction"), xytext=(-8, -12 if class_name == same_name else -40),
                      textcoords="offset points", ha="right", va="top", color=TEXT_PRIMARY_COLOUR,
                      fontsize=17 * POINTS_PER_PIXEL)
    axes.legend(loc="upper center", frameon=False, labelcolor=TEXT_PRIMARY_COLOUR, fontsize=18 * POINTS_PER_PIXEL)
    save_figure(figure=figure, output_path=output_path)
    box = compute_pixel_box(figure=figure, axes=axes, with_labels=True, margin=6)
    pyplot.close(fig=figure)
    return {"probabilities.panel_box": box}


def make_pair_evaluation_figures(output_folder: Path, run_folder: Path, log_path: Path, iteration: int,
                                 test_folder: Path, class_count: int, information_dump: int) -> Dict[str, Any]:
    """
    Makes the training and evaluation figures of the comparison run and gives their facts.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The comparison run.
        log_path (Path): The console log of its training.
        iteration (int): Its last evaluated iteration.
        test_folder (Path): Its test split, which gives the member lengths of the wrong pairs.
        class_count (int): The number of classes of the run, for the loss of a uniform guess on the curves.
        information_dump (int): The interval of the console summaries of the run, the iteration of the first summary
            the console figure shows.

    Returns:
        Dict[str, Any]: The facts, named launch.*, console.*, training.*, evaluation.*, probabilities.*, errors.* and
            samples.*.
    """
    facts = {}

    # Launch information and the console summary of the first interval, its metric lines written by the current
    # formatter, with the throughputs of every interval
    launch_lines = extract_launch_block(log_path=log_path, last_line_start="- Batch Size")
    launch_boxes = draw_console_figure(lines=launch_lines, output_path=output_folder / "evaluation" /
                                       "launch_console.png", font_size=17,
                                       highlight_patterns={"splits": r"Data Folder", "epoch": r"Iterations Per Epoch"})
    facts.update({f"launch.{name}": box for name, box in launch_boxes.items()})
    metric_tracker = build_console_metric_tracker(run_folder=run_folder, trainer_class=SequenceComparisonTrainer,
                                                  trainer_attributes={"decorrelation_activated": False})
    console_lines = reformat_console_summary(lines=extract_interval_block(log_path=log_path,
                                                                          iteration=information_dump),
                                             metric_tracker=metric_tracker)
    console_boxes = draw_console_figure(lines=console_lines, output_path=output_folder / "evaluation" /
                                        "training_console.png", font_size=17,
                                        highlight_patterns={"header": rf"Iteration:\s+{information_dump}\b",
                                                            "interval": r"Interval Of"})
    facts.update({f"console.{name}": box for name, box in console_boxes.items()})
    summaries = read_interval_summaries(log_path=log_path)
    throughputs = [summary["throughput"] for summary in summaries[1:]]
    frame_throughputs = [summary["frame_throughput"] for summary in summaries[1:]]
    facts.update({"console.first_throughput": summaries[0]["throughput"],
                  "console.first_frame_throughput": summaries[0]["frame_throughput"],
                  "console.later_minimum_throughput": min(throughputs),
                  "console.later_maximum_throughput": max(throughputs),
                  "console.later_mean_frame_throughput": round_fact(value=float(numpy.mean(frame_throughputs))),
                  "console.first_interval_loss": summaries[0]["train_total_loss"],
                  "console.last_interval_loss": summaries[-1]["train_total_loss"]})
    interval_line = next(line for line in console_lines if line.startswith("Interval Of"))
    facts["console.first_interval_time"] = interval_line.split("Took")[-1].strip().rstrip(".")

    # Curves, Results.txt and the headline metrics of every evaluation
    facts.update(draw_curve_figure(output_folder=output_folder, run_folder=run_folder, class_count=class_count))
    write_text_lines(output_path=output_folder / "listings" / "results.txt",
                     lines=(run_folder / EVALUATION_FOLDER_NAME / "Results.txt").read_text(
                         encoding="utf-8").splitlines())
    for scalar_name, fact_name in TEST_SCALARS.items():
        iterations, values = read_scalar_curve(run_folder=run_folder / EVALUATION_FOLDER_NAME, split="Test",
                                               tag=scalar_name)
        for evaluated_iteration, value in zip(iterations, values):
            facts[f"evaluation.{fact_name}.iteration_{int(evaluated_iteration)}"] = round_fact(value=value)
        facts["evaluation.iterations"] = [int(evaluated_iteration) for evaluated_iteration in iterations]

    # Confusion matrix, ROC and precision-recall curves and the precision-recall curves zoomed on the thresholds of the
    # last evaluation side by side
    evaluation_folder = run_folder / EVALUATION_FOLDER_NAME / f"Iteration_{iteration}"
    image_names = ["confusion_matrix.png", "roc_curves.png", "precision_recall_curves.png",
                   "precision_recall_thresholds.png"]
    composite_facts = compose_image_grid(rows=[[evaluation_folder / image_name for image_name in image_names]],
                                         row_headings=[f"{EVALUATION_FOLDER_NAME}/Iteration_{iteration}"],
                                         column_headings=image_names, image_height=560,
                                         output_path=output_folder / "evaluation" / "evaluation_images.png")
    facts.update({f"evaluation.composite.{name}": box for name, box in composite_facts.items()})

    # Probabilities of the predictions file, with the thresholds of the evaluation and the wrong pairs
    predictions = read_predictions(run_folder=run_folder, iteration=iteration,
                                   evaluation_folder_name=EVALUATION_FOLDER_NAME)
    thresholds = json.loads((evaluation_folder / "class_thresholds.json").read_text(encoding="utf-8"))
    class_names = list(thresholds["classes"])
    facts.update(draw_probability_figure(output_path=output_folder / "evaluation" / "pair_probabilities.png",
                                         predictions=predictions, thresholds=thresholds, class_names=class_names))
    facts["probabilities.target_precision"] = thresholds["target_precision"]
    for class_name, class_threshold in thresholds["classes"].items():
        facts[f"probabilities.{class_name}.threshold"] = round_fact(value=class_threshold["threshold"])
        facts[f"probabilities.{class_name}.recall"] = round_fact(value=class_threshold["recall"])
    test_pairs = {pair["name"]: pair for pair in json.loads((test_folder / "labels.json").read_text(
        encoding="utf-8"))["pairs"]}
    wrong_rows = [row for row in predictions if row["correct"] == "0"]
    facts["errors.count"] = len(wrong_rows)
    facts["errors.other_count"] = len(wrong_rows) - 1
    facts["errors.true_classes"] = sorted({row["true_class"] for row in wrong_rows})
    sorted_rows = sorted(wrong_rows, key=read_pair_name)
    for error_index, row in enumerate(sorted_rows):
        facts[f"errors.pair_{error_index}.name"] = row["pair"]
        facts[f"errors.pair_{error_index}.before"] = len(test_pairs[row["pair"]]["before"])
        facts[f"errors.pair_{error_index}.after"] = len(test_pairs[row["pair"]]["after"])
        facts[f"errors.pair_{error_index}.probability"] = round_fact(
            value=float(row[f"probability_{row['predicted_class']}"]))
    facts["evaluation.test_pairs"] = len(predictions)
    for error_index, row in enumerate(sorted_rows):
        facts[f"errors.pair_{error_index}.number"] = int(row["pair"].split("_")[-1])

    # Precision and recall of every class at the last evaluation, with the bounds of the interval of the recall, from
    # class_metrics.csv
    with open(file=evaluation_folder / "class_metrics.csv", mode="r", newline="", encoding="utf-8") as metrics_file:
        for metrics_row in csv.DictReader(metrics_file):
            short_name = "loss" if metrics_row["class"] == class_names[0] else "same"
            for column in ["precision", "recall", "recall_lower", "recall_upper"]:
                facts[f"evaluation_class.{short_name}_{column}"] = round_fact(value=float(metrics_row[column]))
            facts[f"evaluation_class.{short_name}_samples"] = int(metrics_row["samples"])

    # Sample images of a wrong and of a right pair
    sample_facts = compose_image_grid(rows=[[evaluation_folder / "Samples" / f"{WRONG_SAMPLE}.png",
                                             evaluation_folder / "Samples" / f"{RIGHT_SAMPLE}.png"]],
                                      row_headings=[f"{EVALUATION_FOLDER_NAME}/Iteration_{iteration}/Samples"],
                                      column_headings=[f"{WRONG_SAMPLE}.png", f"{RIGHT_SAMPLE}.png"],
                                      image_height=660, output_path=output_folder / "evaluation" / "samples.png")
    facts.update({f"samples.{name}": box for name, box in sample_facts.items()})
    facts["samples.wrong_number"] = int(WRONG_SAMPLE.split("_")[-1])
    facts["samples.right_number"] = int(RIGHT_SAMPLE.split("_")[-1])
    for sample_name in [WRONG_SAMPLE, RIGHT_SAMPLE]:
        sample_row = next(row for row in predictions if row["pair"] == sample_name)
        facts[f"samples.{sample_name}.true_class"] = sample_row["true_class"]
        facts[f"samples.{sample_name}.predicted_class"] = sample_row["predicted_class"]
        facts[f"samples.{sample_name}.probability"] = round_fact(
            value=float(sample_row[f"probability_{sample_row['predicted_class']}"]))
    return facts


def read_pair_name(row: Dict[str, str]) -> str:
    """
    Gives the name of the pair of a row of predictions.csv, the key sorting the wrong pairs by name.

    Args:
        row (Dict[str, str]): The row.

    Returns:
        str: The name of its pair.
    """
    return row["pair"]
