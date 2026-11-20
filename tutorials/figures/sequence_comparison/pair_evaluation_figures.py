"""
Training and evaluation figures of the sequence comparison tutorials (chapter 3), made by
tutorials/figures/sequence_comparison/figures.py from the README run of the task: its launch information and the first
console summary, whose metric lines are written again with their logged values by the console formatter of the current
code (reformat_console_summary of figure_utilities.py) so that the scene shows the columns and units a run trained
today prints, its training and validation curves up to its last checkpoint (an empty-axes base and its overlay,
uncovered one panel at a time), and, from the evaluations of its last checkpoint and of the one closest below its half
in its Checkpoint_Evaluation folder, its Results.txt, the confusion matrix, ROC and precision-recall curves and the
precision-recall curves zoomed on the thresholds of its last evaluation side by side, the probability of
same_number_of_objects of every test pair with the thresholds of the evaluation, and the sample image of a wrong pair,
drawn by the code of the sample images since none of the fixed test samples is wrong, beside the saved sample image of
a right pair. Its facts are the metrics, losses, throughputs, thresholds and errors the narration quotes, named by their
role (first and last evaluation, worst class, wrong and right pair) rather than by an iteration or a pair name. The
accuracies and the target precision drawn are written as the percentages of the reports of the repository ("94.03 %",
"99 %"), the facts and the probability thresholds keeping their values between 0 and 1.
"""

import csv
import json

import cv2
import numpy
import torch

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
from tutorials.figures.sequence_comparison.pair_dataset_figures import SHOWN_TEST_PAIR
from vision_training.data.sequence.sequence_pair_dataset import PAIR_MEMBERS, SequencePairDataset, split_pair_members
from vision_training.evaluation.checkpoint_evaluation import CHECKPOINT_EVALUATION_FOLDER_NAME
from vision_training.evaluation.report_formatting import (format_percentage, format_percentage_label,
                                                          format_percentage_tick)
from vision_training.evaluation.sample_selection import choose_sample_identifiers
from vision_training.trainers.sequence_classification_trainer import FRAME_GRID_COLUMNS
from vision_training.trainers.sequence_comparison_trainer import SequenceComparisonTrainer
from vision_training.utilities.visualization import (CORRECT_PREDICTION_COLOUR, WRONG_PREDICTION_COLOUR,
                                                     add_caption_bar, build_frame_grid, convert_image_to_array)

# Smoothing of the training curves, the exponential moving average of TensorBoard
SMOOTHING_WEIGHT = 0.95

# Folder of the evaluations the figures read, inside the run folder: the last checkpoint and the one closest below its
# half, evaluated after the training with the helpers of scripts/evaluation/evaluate_checkpoint.py (predictions saved,
# per-class report), whatever iterations the training evaluated itself in its Evaluation folder
EVALUATION_FOLDER_NAME = CHECKPOINT_EVALUATION_FOLDER_NAME

# Headline scalars of the test evaluations kept as facts, with the names of their facts
TEST_SCALARS = {"Accuracy": "accuracy", "Balanced Accuracy": "balanced_accuracy", "Macro F1": "macro_f1",
                "Macro ROC AUC": "macro_roc_auc", "Macro Average Precision": "macro_average_precision"}

# Pairs whose sample images the scene compares: the loss pair of the first chapter, which the run gets wrong and which
# is not one of the fixed test samples of the evaluation (none of them is wrong), drawn by the code of the sample
# images, and a fixed test sample keeping its objects that the run gets right, drawn the same way and checked equal to
# its saved sample image
WRONG_SAMPLE = SHOWN_TEST_PAIR
RIGHT_SAMPLE = "test_000000"

# Wrong pairs of the last evaluation, looked at frame by frame: every one a loss given same, whose lost object is small,
# dark or close to the colour of its background, as the narration says; a build whose wrong pairs differ stops, the
# narration needing the same look at the new ones
LOOKED_AT_WRONG_PAIRS = {"test_000021": "a tiny purple shape, on a textured olive background",
                         "test_000028": "a small dark navy parallelogram, on black",
                         "test_000051": "a tiny cyan hollow shape, on a light blue gradient",
                         "test_000063": "a tiny magenta hollow square, on black crossed by dim lines",
                         "test_000096": "a small green shape, on a textured background of coloured blotches",
                         "test_000097": "a tiny dark teal hollow square, on black",
                         "test_000150": "a small dark red annulus, on black",
                         "test_000154": "the dark purple hollow semicircle, on black crossed by dim lines",
                         "test_000168": "a small dark green plus sign, on green",
                         "test_000179": "a tiny purple annulus, on a purple background"}

# Bins of the histogram of the probability of same_number_of_objects
PROBABILITY_BINS = numpy.linspace(start=0.0, stop=1.0, num=41)


def draw_curve_figure(output_folder: Path, run_folder: Path, class_count: int, iteration: int) -> Dict[str, Any]:
    """
    Draws the total loss and the accuracy of the training and validation batches of the run on two panels, every
    curve smoothed like TensorBoard over its faint raw values, with the test accuracy of every evaluation and the loss
    of a uniform guess over the classes of the run; the empty axes are written as the base image and the whole chart
    as its overlay. The curves end at the last checkpoint of the run, a run stopped once a checkpoint passed its target
    having logged a few iterations more before it ended. Gives the boxes of the panels and their reveal regions, and
    the first and last values of the curves.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The comparison run.
        class_count (int): The number of classes of the run, whose natural log is the loss of a uniform guess.
        iteration (int): The last checkpoint of the run, where the curves end.

    Returns:
        Dict[str, Any]: The facts, named training.*.
    """
    # Curves of the training and validation batches up to the last checkpoint
    curves = {}
    for split in ["Train", "Validation"]:
        for tag in ["Total Loss", "Accuracy"]:
            iterations, values = read_scalar_curve(run_folder=run_folder, split=split, tag=tag)
            curves[(split, tag)] = (iterations[iterations <= iteration], values[iterations <= iteration])
    # Test accuracies of the checkpoint evaluations, whose TensorBoard logs are in Checkpoint_Evaluation
    test_iterations, test_accuracies = read_scalar_curve(run_folder=run_folder / EVALUATION_FOLDER_NAME, split="Test",
                                                         tag="Accuracy")
    last_iteration = int(curves[("Train", "Total Loss")][0][-1])
    # Top of the loss panel: the loss of a uniform guess or the highest smoothed loss, whichever is higher, so that no
    # smoothed curve leaves the panel
    loss_top = max(0.8, 1.05 * max(float(smooth_values(values=curves[(split, "Total Loss")][1],
                                                       weight=SMOOTHING_WEIGHT).max())
                                   for split in ["Train", "Validation"]))
    register_fonts()
    figure, (loss_axes, accuracy_axes) = pyplot.subplots(nrows=1, ncols=2, figsize=(CHART_WIDTH / DOTS_PER_INCH,
                                                                                   CHART_HEIGHT / DOTS_PER_INCH),
                                                         dpi=DOTS_PER_INCH)
    figure.patch.set_facecolor(SURFACE_COLOUR)
    # Panels apart enough for the percentage ticks of the accuracy
    figure.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.12, wspace=0.2)
    for axes, title, y_label, y_limit in [(loss_axes, "Total loss", "Loss", (0.0, loss_top)),
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

    # Histogram of every true class on the probability of same_number_of_objects, the two bars of a bin side by side
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
    # Legend over the low bars between the tallest bar of loss_of_object and the threshold lines, clear of their labels
    axes.legend(loc="center left", bbox_to_anchor=(0.2, 0.6), frameon=False, labelcolor=TEXT_PRIMARY_COLOUR,
                fontsize=18 * POINTS_PER_PIXEL)
    save_figure(figure=figure, output_path=output_path)
    box = compute_pixel_box(figure=figure, axes=axes, with_labels=True, margin=6)
    pyplot.close(fig=figure)
    return {"probabilities.panel_box": box}


def make_pair_evaluation_figures(output_folder: Path, run_folder: Path, log_path: Path, iteration: int,
                                 test_folder: Path, class_count: int, information_dump: int,
                                 image_settings: Dict[str, Any], sequence_settings: Dict[str, Any],
                                 label_dictionary: Dict[int, int], number_sample_images: int) -> Dict[str, Any]:
    """
    Makes the training and evaluation figures of the third chapter from the comparison run (the console figures, the
    curves, the evaluation images, the probability histogram and the sample images of a wrong and a right pair) and
    gives their facts, after checking what the narration states of them: the last evaluation is the last checkpoint,
    the wrong pairs are those looked at, no fixed test sample is wrong, and the sample image drawn for the right pair
    equals the saved one.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The comparison run.
        log_path (Path): The console log of its training.
        iteration (int): Its last checkpoint, which its last checkpoint evaluation must be of.
        test_folder (Path): Its test split, which gives the member lengths of the wrong pairs, its fixed test samples
            and the frames of the pairs whose sample images are drawn.
        class_count (int): The number of classes of the run, for the loss of a uniform guess on the curves.
        information_dump (int): The interval of the console summaries of the run, the iteration of the first summary
            the console figure shows.
        image_settings (Dict[str, Any]): The Data.image_settings of the run, its frame size and keep_ratio.
        sequence_settings (Dict[str, Any]): The Data.sequence_settings of the run, its maximum sequence length.
        label_dictionary (Dict[int, int]): The class index of every category of the run.
        number_sample_images (int): The number of fixed test samples of its evaluations
            (ExperimentConfiguration.evaluation.number_sample_images).

    Returns:
        Dict[str, Any]: The facts, named launch.*, console.*, training.*, evaluation.*, probabilities.*, errors.* and
            samples.*.

    Raises:
        ValueError: If a property the narration states of the run does not hold.
        OSError: If a sample image cannot be written.
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

    # Throughputs of the first interval and of the later ones, the first and last interval losses, and the duration of
    # the first interval as its last line prints it
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

    # Curves up to the last checkpoint, Results.txt and the headline metrics of every evaluation, the first and the
    # last evaluation named by their role, the last one being the last checkpoint of the run
    facts.update(draw_curve_figure(output_folder=output_folder, run_folder=run_folder, class_count=class_count,
                                   iteration=iteration))
    write_text_lines(output_path=output_folder / "listings" / "results.txt",
                     lines=(run_folder / EVALUATION_FOLDER_NAME / "Results.txt").read_text(
                         encoding="utf-8").splitlines())
    for scalar_name, fact_name in TEST_SCALARS.items():
        iterations, values = read_scalar_curve(run_folder=run_folder / EVALUATION_FOLDER_NAME, split="Test",
                                               tag=scalar_name)
        evaluated_iterations = [int(evaluated_iteration) for evaluated_iteration in iterations]
        if evaluated_iterations[-1] != iteration:
            raise ValueError(f"The last evaluation of {run_folder / EVALUATION_FOLDER_NAME} is of iteration "
                             f"{evaluated_iterations[-1]}, expected the last checkpoint {iteration} of the run.")
        facts[f"evaluation.{fact_name}.first"] = round_fact(value=values[0])
        facts[f"evaluation.{fact_name}.last"] = round_fact(value=values[-1])
        facts["evaluation.iterations"] = evaluated_iterations

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

    # Probabilities of the predictions file, with the thresholds of the evaluation
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

    # Wrong pairs of the last evaluation: their number for every true and predicted class, the most confident one,
    # and every one by name with the lengths of its members
    test_pairs = {pair["name"]: pair for pair in json.loads((test_folder / "labels.json").read_text(
        encoding="utf-8"))["pairs"]}
    wrong_rows = sorted([row for row in predictions if row["correct"] == "0"], key=read_pair_name)
    facts["errors.count"] = len(wrong_rows)
    facts["evaluation.test_pairs"] = len(predictions)
    for true_class in class_names:
        for predicted_class in [class_name for class_name in class_names if class_name != true_class]:
            facts[f"errors.{true_class}.{predicted_class}"] = sum(
                int(row["true_class"] == true_class and row["predicted_class"] == predicted_class)
                for row in wrong_rows)

    # Every wrong pair by its index in name order, and the most confident one, with its classes, member lengths and
    # probability
    named_errors = {f"pair_{error_index}": row for error_index, row in enumerate(wrong_rows)}
    if wrong_rows:
        named_errors["most_confident"] = max(wrong_rows, key=read_predicted_probability)
    for error_name, row in named_errors.items():
        facts[f"errors.{error_name}.name"] = row["pair"]
        facts[f"errors.{error_name}.number"] = int(row["pair"].split("_")[-1])
        facts[f"errors.{error_name}.true_class"] = row["true_class"]
        facts[f"errors.{error_name}.before"] = len(test_pairs[row["pair"]]["before"])
        facts[f"errors.{error_name}.after"] = len(test_pairs[row["pair"]]["after"])
        facts[f"errors.{error_name}.probability"] = round_fact(value=read_predicted_probability(row=row))

    # Precision, recall and F1 of every class at the last evaluation, with the bounds of the interval of the recall,
    # from class_metrics.csv, and the class of the worst F1 the per-class table of Results.txt lists first (ties in
    # class order)
    with open(file=evaluation_folder / "class_metrics.csv", mode="r", newline="", encoding="utf-8") as metrics_file:
        metrics_rows = list(csv.DictReader(metrics_file))
    for metrics_row in metrics_rows:
        short_name = "loss" if metrics_row["class"] == class_names[0] else "same"
        for column in ["precision", "recall", "recall_lower", "recall_upper", "f1"]:
            facts[f"evaluation_class.{short_name}_{column}"] = round_fact(value=float(metrics_row[column]))
        facts[f"evaluation_class.{short_name}_samples"] = int(metrics_row["samples"])
    worst_row = min(metrics_rows, key=read_class_f1)
    facts["evaluation_class.worst_f1.class"] = worst_row["class"]
    facts["evaluation_class.worst_f1.class_words"] = worst_row["class"].replace("_", " ")

    # Errors looked at one by one, and the fixed test samples of the evaluation, chosen by the code of the trainer from
    # the classes of the test pairs, none of them wrong, as the narration says
    if [row["pair"] for row in wrong_rows] != sorted(LOOKED_AT_WRONG_PAIRS):
        raise ValueError(f"The wrong pairs of iteration {iteration} are {[row['pair'] for row in wrong_rows]}, the "
                         f"narration describes {sorted(LOOKED_AT_WRONG_PAIRS)}: look at the new ones and update "
                         f"LOOKED_AT_WRONG_PAIRS and the narration.")
    test_dataset = SequencePairDataset(data_directory=str(test_folder / "data"),
                                       labels_file=str(test_folder / "labels.json"), image_size=image_settings["size"],
                                       keep_ratio=image_settings["keep_ratio"], label_dictionary=label_dictionary,
                                       sequence_settings=sequence_settings, split_name="Test", shuffle=False)
    sample_names = sorted(test_dataset.pairs.find_record(identifier=identifier)["name"] for identifier in
                          choose_sample_identifiers(sample_classes=test_dataset.get_sample_class_indices(),
                                                    number_samples=number_sample_images))
    wrong_sample_names = [row["pair"] for row in wrong_rows if row["pair"] in sample_names]
    if wrong_sample_names or RIGHT_SAMPLE not in sample_names or WRONG_SAMPLE in sample_names:
        raise ValueError(f"The narration says that no fixed test sample is wrong, that {RIGHT_SAMPLE} is one and that "
                         f"{WRONG_SAMPLE} is not, but the fixed samples are {sample_names} and the wrong ones among "
                         f"them {wrong_sample_names}.")
    facts["samples.fixed_count"] = len(sample_names)

    # Sample images of the wrong pair the first chapter shows and of a right pair, drawn by the code of the sample
    # images, the drawing of the right pair checked equal to the sample image the evaluation saved
    sample_paths = {}
    for role, pair_name, expected_correct in [("wrong", WRONG_SAMPLE, "0"), ("right", RIGHT_SAMPLE, "1")]:
        sample_row = next(row for row in predictions if row["pair"] == pair_name)
        if sample_row["correct"] != expected_correct:
            raise ValueError(f"The {role} pair {pair_name} is {'right' if sample_row['correct'] == '1' else 'wrong'} "
                             f"at iteration {iteration}, the narration showing it {role}: choose another pair.")
        sample_paths[role] = output_folder / "evaluation" / f"{role}_pair_sample.png"
        sample_image = draw_pair_sample_image(test_dataset=test_dataset, pair_name=pair_name, prediction_row=sample_row)
        if not cv2.imwrite(filename=str(sample_paths[role]), img=sample_image):
            raise OSError(f"The sample image {sample_paths[role]} could not be written.")
        # Name, classes, probability and member lengths of the pair, quoted by the sample scene
        facts[f"samples.{role}.name"] = pair_name
        facts[f"samples.{role}.number"] = int(pair_name.split("_")[-1])
        facts[f"samples.{role}.true_class"] = sample_row["true_class"]
        facts[f"samples.{role}.predicted_class"] = sample_row["predicted_class"]
        facts[f"samples.{role}.probability"] = round_fact(value=read_predicted_probability(row=sample_row))
        facts[f"samples.{role}.before"] = len(test_pairs[pair_name]["before"])
        facts[f"samples.{role}.after"] = len(test_pairs[pair_name]["after"])
    saved_right_image = cv2.imread(filename=str(evaluation_folder / "Samples" / f"{RIGHT_SAMPLE}.png"))
    drawn_right_image = cv2.imread(filename=str(sample_paths["right"]))
    if saved_right_image is None or not numpy.array_equal(saved_right_image, drawn_right_image):
        raise ValueError(f"The sample image drawn for {RIGHT_SAMPLE} differs from the one the evaluation saved in "
                         f"{evaluation_folder / 'Samples'}: draw_pair_sample_image no longer draws like the trainer.")

    # Both sample images side by side, the right one being the saved sample image of the evaluation
    sample_facts = compose_image_grid(rows=[[sample_paths["wrong"],
                                             evaluation_folder / "Samples" / f"{RIGHT_SAMPLE}.png"]],
                                      row_headings=[f"{EVALUATION_FOLDER_NAME}/Iteration_{iteration}"],
                                      column_headings=[f"{WRONG_SAMPLE}, drawn like a sample image",
                                                       f"Samples/{RIGHT_SAMPLE}.png"],
                                      image_height=660, output_path=output_folder / "evaluation" / "samples.png")
    facts.update({f"samples.{name}": box for name, box in sample_facts.items()})
    return facts


def draw_pair_sample_image(test_dataset: SequencePairDataset, pair_name: str,
                           prediction_row: Dict[str, str]) -> numpy.ndarray:
    """
    Draws for any test pair the image the evaluation saves for its fixed test samples (build_sample_image of the
    classification trainer around build_sample_frame_grid of the comparison trainer), so that the evaluation chapter
    can show a wrong pair when none of the fixed test samples is wrong: the real before frames above the real after
    frames, read by the pair dataset at the frame size of the run, FRAME_GRID_COLUMNS per row, every frame numbered and
    captioned with its member, under the caption 'true -> predicted probability' of its row of predictions.csv, in
    green when right and in red when wrong.

    Args:
        test_dataset (SequencePairDataset): The test pairs, read at the frame size of the run.
        pair_name (str): The name of the pair.
        prediction_row (Dict[str, str]): Its row of predictions.csv.

    Returns:
        numpy.ndarray: The [H, W, 3] uint8 BGR image.
    """
    pair_index = next(index for index in range(len(test_dataset)) if test_dataset.pairs[index]["name"] == pair_name)
    sample = test_dataset[pair_index]

    # One grid per member, the before member above the after member, as build_sample_frame_grid lays them out
    member_frames = split_pair_members(pair_tensor=sample["images"].unsqueeze(dim=0))
    member_masks = split_pair_members(pair_tensor=sample["sequence_mask"].unsqueeze(dim=0))
    member_grids = [build_frame_grid(frames=frames[0], sequence_mask=mask[0],
                                     frame_captions=[member_name] * int(mask[0].sum()),
                                     number_columns=FRAME_GRID_COLUMNS)
                    for member_name, frames, mask in zip(PAIR_MEMBERS, member_frames, member_masks)]

    # Caption of the prediction above the grids, as build_sample_image writes it
    caption = (f"{prediction_row['true_class']} -> {prediction_row['predicted_class']} "
               f"{read_predicted_probability(row=prediction_row):.3f}")
    caption_colour = CORRECT_PREDICTION_COLOUR if prediction_row["correct"] == "1" else WRONG_PREDICTION_COLOUR
    return add_caption_bar(image=convert_image_to_array(image=torch.cat(tensors=member_grids, dim=1)),
                           caption=caption, caption_colour=caption_colour)


def read_predicted_probability(row: Dict[str, str]) -> float:
    """
    Gives the probability of the predicted class of a row of predictions.csv, the probability the sample images and the
    narration quote for a pair, and the key ranking the wrong pairs by confidence.

    Args:
        row (Dict[str, str]): The row.

    Returns:
        float: The probability of its predicted class.
    """
    return float(row[f"probability_{row['predicted_class']}"])


def read_class_f1(row: Dict[str, str]) -> float:
    """
    Gives the F1 of a row of class_metrics.csv, the key finding the class of the worst F1.

    Args:
        row (Dict[str, str]): The row.

    Returns:
        float: Its F1.
    """
    return float(row["f1"])


def read_pair_name(row: Dict[str, str]) -> str:
    """
    Gives the name of the pair of a row of predictions.csv, the key sorting the wrong pairs by name.

    Args:
        row (Dict[str, str]): The row.

    Returns:
        str: The name of its pair.
    """
    return row["pair"]
