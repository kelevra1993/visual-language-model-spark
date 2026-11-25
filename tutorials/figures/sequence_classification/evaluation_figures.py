"""
Training and evaluation figures of the sequence classification tutorials (chapters 1 and 3), made by
tutorials/figures/sequence_classification/figures.py from the three tutorial runs of the task: the flatten run at 256
pixels of the synthetic configuration, the average pooling run at 512 pixels and the flatten run at 512 pixels. It
draws the launch information and the first console summary of the flatten run at 256 pixels, its metric lines written
by the console formatter of the current code, the training loss of the three runs (an empty-axes base and its overlay,
uncovered one panel at a time), the confusion matrices and ROC curves of the flatten and average runs side by side, the
probability of the true class of every test sequence from their predictions files, the accuracy of the flatten run on
the test sequences of every number of frames, and one sample sequence of both runs; its facts are the metrics of every
evaluation, the throughputs and losses of the logs and the predictions the narration quotes, and the properties the
narration states of them (the single class the average run predicts, a sample right with flatten and wrong with
average pooling) are checked. The evaluations of the flatten run at 256 pixels and of the average run are read from
their Checkpoint_Evaluation folder, where their checkpoints were evaluated again after the training, those of the
flatten run at 512 pixels from the evaluations of its training.
"""

import csv
import json
import math
import re

import numpy

from collections import defaultdict
from matplotlib import pyplot
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, CONSOLE_INTERVAL_LINE_PATTERN,
                                                CONSOLE_VALIDATION_LINE_PATTERN, DOTS_PER_INCH, POINTS_PER_PIXEL,
                                                REVEAL_MARGIN, SERIES_COLOURS, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR,
                                                TEXT_SECONDARY_COLOUR, build_console_metric_tracker,
                                                compute_pixel_box, format_thousands, parse_console_number,
                                                read_run_configuration, reformat_console_summary, register_fonts,
                                                round_fact, save_figure, style_chart_panel)
from tutorials.figures.foundations.training_curves import smooth_values
from tutorials.figures.sequence_classification.model_figures import AVERAGE_RUN_NAME, MAIN_RUN_NAME
from tutorials.figures.sequence_classification.sequence_figure_utilities import (add_probability_axis,
                                                                                 compose_image_grid,
                                                                                 draw_console_figure,
                                                                                 extract_interval_block,
                                                                                 extract_launch_block,
                                                                                 read_interval_summaries,
                                                                                 read_predictions, read_scalar_curve,
                                                                                 write_text_lines)
from vision_training.evaluation.checkpoint_evaluation import CHECKPOINT_EVALUATION_FOLDER_NAME
from vision_training.evaluation.classification.classification_metrics import TOP_K_ACCURACY_CLASSES
from vision_training.evaluation.report_formatting import format_percentage, format_percentage_tick
from vision_training.trainers.sequence_classification_trainer import SequenceClassificationTrainer

# Smoothing of the training curves, the exponential moving average of TensorBoard
SMOOTHING_WEIGHT = 0.95

# Headline scalars of the test evaluations kept as facts, with the names of their facts, the k of the top-k accuracy
# being the fixed one of the current evaluation
TEST_SCALARS = {"Accuracy": "accuracy", "Balanced Accuracy": "balanced_accuracy", "Macro F1": "macro_f1",
                f"Top-{TOP_K_ACCURACY_CLASSES} Accuracy": f"top_{TOP_K_ACCURACY_CLASSES}_accuracy",
                "Macro ROC AUC": "macro_roc_auc", "Macro Average Precision": "macro_average_precision"}

# Runs whose evaluations the scenes show, evaluated again into their Checkpoint_Evaluation folder, and the flatten run
# at 512 pixels, whose evaluations of its training give its accuracy
EVALUATED_RUNS = [MAIN_RUN_NAME, AVERAGE_RUN_NAME]
FLATTEN_512_RUN_NAME = "flatten_512"

# Sample sequence compared between the runs, one of the 8 fixed test sequences of both, which the sample scene says
# the flatten run gets right and the average run wrong, and the fewest frames it shows so that its movement is seen
SAMPLE_SEQUENCE = "test_000004"
SAMPLE_MINIMUM_FRAMES = 6

# Bins of the histograms of the probability of the true class
PROBABILITY_BINS = numpy.linspace(start=0.0, stop=1.0, num=41)

# Probability of the true class above which a test sequence counts as confidently right
CONFIDENT_PROBABILITY = 0.9

# Folder of the evaluation of a checkpoint, named after its iteration
EVALUATION_FOLDER_PATTERN = re.compile(pattern=r"^Iteration_(\d+)$")


def find_evaluated_iterations(run_folder: Path) -> List[int]:
    """
    Finds the evaluated checkpoints of a run, its Checkpoint_Evaluation/Iteration_<n> folders, whose reports and
    predictions the report scenes quote, so that the facts follow the evaluations of the run rather than typed
    iterations.

    Args:
        run_folder (Path): The run folder.

    Returns:
        List[int]: The evaluated iterations, in order.

    Raises:
        FileNotFoundError: If the run holds no evaluation folder.
    """
    evaluation_root = run_folder / CHECKPOINT_EVALUATION_FOLDER_NAME
    iterations = [int(match.group(1)) for folder in evaluation_root.iterdir()
                  if (match := EVALUATION_FOLDER_PATTERN.match(folder.name)) is not None and folder.is_dir()]
    if not iterations:
        raise FileNotFoundError(f"No Iteration_<n> folder in {evaluation_root}.")
    return sorted(iterations)


def read_test_metrics(evaluation_root: Path, run_name: str, scalar_names: List[str]) -> Dict[str, Any]:
    """
    Reads headline metrics of every evaluated checkpoint of a run from the Test TensorBoard logs of its
    Checkpoint_Evaluation folder (or of the run folder, for the evaluations of its training), as facts named
    evaluation.<run name>.<metric>.iteration_<n>.

    Args:
        evaluation_root (Path): The folder holding the Tensorboard/Test logs.
        run_name (str): The short name of the run in the facts, e.g. "flatten_256".
        scalar_names (List[str]): The scalars read, keys of TEST_SCALARS.

    Returns:
        Dict[str, Any]: The facts.
    """
    facts = {}
    for scalar_name in scalar_names:
        fact_name = TEST_SCALARS[scalar_name]
        iterations, values = read_scalar_curve(run_folder=evaluation_root, split="Test", tag=scalar_name)
        for iteration, value in zip(iterations, values):
            facts[f"evaluation.{run_name}.{fact_name}.iteration_{int(iteration)}"] = round_fact(value=value)
    return facts


def read_class_recalls(run_folder: Path, iteration: int) -> List[Dict[str, Any]]:
    """
    Reads the recall of every class of an evaluated checkpoint from its class_metrics.csv, in class order, for the
    class the Per Class Recall table writes first.

    Args:
        run_folder (Path): The run folder.
        iteration (int): The evaluated iteration.

    Returns:
        List[Dict[str, Any]]: The 'class', 'samples' and 'recall' of every class.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    metrics_path = run_folder / CHECKPOINT_EVALUATION_FOLDER_NAME / f"Iteration_{iteration}" / "class_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No class metrics file {metrics_path}.")
    with open(file=metrics_path, mode="r", newline="", encoding="utf-8") as metrics_file:
        return [{"class": row["class"], "samples": int(row["samples"]), "recall": float(row["recall"])}
                for row in csv.DictReader(metrics_file)]


def draw_loss_figure(output_folder: Path, curves: Dict[str, Dict[str, Any]], number_classes: int) -> Dict[str, Any]:
    """
    Draws the training total loss of the three runs on two panels, the flatten run at 256 pixels on the left and the
    two runs at 512 pixels on the right, every curve smoothed like TensorBoard over its faint raw values, with the line
    of the natural log of the number of classes, the loss of a uniform guess. The empty axes are written as the base
    image and the whole chart as its overlay; gives the boxes of the panels and their reveal regions.

    Args:
        output_folder (Path): The figures folder of the series.
        curves (Dict[str, Dict[str, Any]]): For every run name, its 'label', 'pooling', 'image_size', 'colour',
            'iterations' and 'values'.
        number_classes (int): The classes of the runs, whose uniform guess the dashed line marks.

    Returns:
        Dict[str, Any]: The boxes, named training.*.
    """
    register_fonts()
    figure, panels = pyplot.subplots(nrows=1, ncols=2, figsize=(CHART_WIDTH / DOTS_PER_INCH,
                                                                CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH)
    figure.patch.set_facecolor(SURFACE_COLOUR)
    figure.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.12, wspace=0.14)
    panel_runs = [[MAIN_RUN_NAME], [AVERAGE_RUN_NAME, FLATTEN_512_RUN_NAME]]
    main_curve, average_curve, flatten_curve = (curves[run_name] for run_name in
                                                [MAIN_RUN_NAME, AVERAGE_RUN_NAME, FLATTEN_512_RUN_NAME])
    panel_titles = [f"{main_curve['pooling'].capitalize()} pooling, {main_curve['image_size']} pixels",
                    f"{average_curve['image_size']} pixels: {average_curve['pooling']} and "
                    f"{flatten_curve['pooling']} pooling"]
    if average_curve["image_size"] != flatten_curve["image_size"]:
        raise ValueError("The right panel of the loss figure holds two runs of one frame size, but the average and "
                         "flatten runs it draws have different ones.")
    chance_loss = math.log(number_classes)
    loss_top = max(1.6, math.ceil(max(float(numpy.max(curve["values"])) for curve in curves.values()) * 10) / 10)

    # Panels with the chance line, the legend and the axes, saved empty as the base image of the scene
    for axes, run_names, title in zip(panels, panel_runs, panel_titles):
        style_chart_panel(axes=axes, title=title, x_label="Iteration", y_label="Training total loss"
                          if axes is panels[0] else None)
        last_iteration = max(int(curves[run_name]["iterations"][-1]) for run_name in run_names)
        axes.set_xlim(left=0, right=last_iteration)
        axes.set_ylim(bottom=0.0, top=loss_top)
        axes.xaxis.set_major_formatter(formatter=FuncFormatter(func=format_thousands))
        axes.axhline(y=chance_loss, color=TEXT_SECONDARY_COLOUR, linewidth=1.5 * POINTS_PER_PIXEL, linestyle="--")
        axes.annotate(text=f"ln {number_classes} = {chance_loss:.3f}, a uniform guess",
                      xy=(last_iteration, chance_loss), xytext=(-8, 10), textcoords="offset points", ha="right",
                      va="bottom", color=TEXT_SECONDARY_COLOUR, fontsize=17 * POINTS_PER_PIXEL)

    # Legend of the three runs above the panels, and the note telling the faint raw curves from the smoothed ones
    run_order = [MAIN_RUN_NAME, AVERAGE_RUN_NAME, FLATTEN_512_RUN_NAME]
    legend_handles = [Line2D(xdata=[0], ydata=[0], color=curves[run_name]["colour"], linewidth=3 * POINTS_PER_PIXEL)
                      for run_name in run_order]
    figure.legend(handles=legend_handles, labels=[curves[run_name]["label"] for run_name in run_order],
                  loc="center left", bbox_to_anchor=(0.05, 0.93), ncols=3, frameon=False,
                  labelcolor=TEXT_PRIMARY_COLOUR, fontsize=18 * POINTS_PER_PIXEL, handlelength=1.6)
    figure.text(x=0.985, y=0.93, s=f"Faint: every iteration · solid: smoothed ({SMOOTHING_WEIGHT})", ha="right",
                va="center", color=TEXT_SECONDARY_COLOUR, fontsize=17 * POINTS_PER_PIXEL)
    save_figure(figure=figure, output_path=output_folder / "evaluation" / "training_loss_axes.png")

    # Curves of every run, raw and smoothed, written as the overlay
    for axes, run_names in zip(panels, panel_runs):
        for run_name in run_names:
            curve = curves[run_name]
            axes.plot(curve["iterations"], curve["values"], color=curve["colour"], alpha=0.22,
                      linewidth=1.2 * POINTS_PER_PIXEL)
            axes.plot(curve["iterations"], smooth_values(values=curve["values"], weight=SMOOTHING_WEIGHT),
                      color=curve["colour"], linewidth=3 * POINTS_PER_PIXEL, solid_capstyle="round")
    save_figure(figure=figure, output_path=output_folder / "evaluation" / "training_loss.png")
    facts = {}
    for panel_index, axes in enumerate(panels):
        facts[f"training.panel_{panel_index}_box"] = compute_pixel_box(figure=figure, axes=axes, with_labels=True,
                                                                       margin=0)
        facts[f"training.panel_{panel_index}_reveal_region"] = compute_pixel_box(figure=figure, axes=axes,
                                                                                 with_labels=False,
                                                                                 margin=REVEAL_MARGIN)
    pyplot.close(fig=figure)
    return facts


def compute_true_probabilities(predictions: List[Dict[str, str]]) -> numpy.ndarray:
    """
    Gives the probability the model gave to the true class of every test sequence of a predictions file, for the
    probability histograms and the length accuracy facts of the evaluation scenes.

    Args:
        predictions (List[Dict[str, str]]): The rows of predictions.csv.

    Returns:
        numpy.ndarray: The probability of the true class of every row.
    """
    return numpy.array([float(row[f"probability_{row['true_class']}"]) for row in predictions])


def draw_probability_figure(output_path: Path, probability_sets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Draws the histograms of the probability of the true class of the test sequences, one panel per run, the counts on
    a square-root scale so that the few low probabilities of a good run stay visible; gives the boxes of the panels.

    Args:
        output_path (Path): The PNG file.
        probability_sets (Dict[str, Dict[str, Any]]): For every run name, its 'title', 'colour' and 'probabilities'.

    Returns:
        Dict[str, Any]: The boxes, named probabilities.*.
    """
    register_fonts()
    figure, panels = pyplot.subplots(nrows=1, ncols=len(probability_sets), figsize=(CHART_WIDTH / DOTS_PER_INCH,
                                                                                    CHART_HEIGHT / DOTS_PER_INCH),
                                     dpi=DOTS_PER_INCH, sharey=True)
    figure.patch.set_facecolor(SURFACE_COLOUR)
    figure.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.13, wspace=0.08)

    # One histogram panel per run on a square-root count scale, its box kept as a fact for the highlights
    facts = {}
    for panel_index, (axes, (run_name, probability_set)) in enumerate(zip(panels, probability_sets.items())):
        counts, _ = numpy.histogram(a=probability_set["probabilities"], bins=PROBABILITY_BINS)
        axes.bar(x=PROBABILITY_BINS[:-1], height=counts, width=PROBABILITY_BINS[1] - PROBABILITY_BINS[0] - 0.004,
                 align="edge", color=probability_set["colour"])
        style_chart_panel(axes=axes, title=probability_set["title"], x_label="Probability of the true class",
                          y_label="Test sequences" if panel_index == 0 else None)
        axes.set_yscale(value="function", functions=(numpy.sqrt, numpy.square))
        axes.set_xlim(left=0.0, right=1.0)
        axes.set_yticks(ticks=[0, 10, 50, 100, 200])
        axes.grid(visible=False, axis="x")
        facts[f"probabilities.panel_{panel_index}_box"] = compute_pixel_box(figure=figure, axes=axes,
                                                                            with_labels=True, margin=6)
    save_figure(figure=figure, output_path=output_path)
    pyplot.close(fig=figure)
    return facts


def compute_length_facts(predictions: List[Dict[str, str]], test_folder: Path) -> Dict[str, Any]:
    """
    Gives the accuracy, the mean probability of the true class and the number of test sequences of every number of
    frames, from the predictions file of an evaluation and the lengths of the test manifest, and the weakest length
    (the lowest accuracy, the shortest length among ties).

    Args:
        predictions (List[Dict[str, str]]): The rows of predictions.csv of the evaluation.
        test_folder (Path): The test split, holding labels.json.

    Returns:
        Dict[str, Any]: The facts, named length_accuracy.*.

    Raises:
        ValueError: If a predicted sequence is not in the test manifest.
    """
    manifest = json.loads((test_folder / "labels.json").read_text(encoding="utf-8"))
    lengths = {sequence["name"]: len(sequence["frames"]) for sequence in manifest["sequences"]}
    missing_sequences = sorted({row["sequence"] for row in predictions} - set(lengths))
    if missing_sequences:
        raise ValueError(f"The predicted sequences {missing_sequences[:5]} are not in {test_folder / 'labels.json'}.")

    # Rows of every number of frames: correctness and probability of the true class
    rows_by_length = defaultdict(list)
    for row in predictions:
        rows_by_length[lengths[row["sequence"]]].append(row)
    facts = {"length_accuracy.lengths": sorted(rows_by_length)}
    accuracies = {}
    for length, rows in sorted(rows_by_length.items()):
        accuracies[length] = float(numpy.mean([row["correct"] == "1" for row in rows]))
        facts[f"length_accuracy.frames_{length}.accuracy"] = round_fact(value=accuracies[length])
        facts[f"length_accuracy.frames_{length}.mean_true_probability"] = round_fact(
            value=float(compute_true_probabilities(predictions=rows).mean()))
        facts[f"length_accuracy.frames_{length}.sequences"] = len(rows)
        facts[f"length_accuracy.frames_{length}.wrong"] = sum(row["correct"] != "1" for row in rows)
    weakest_length = min(accuracies, key=lambda length: (accuracies[length], length))
    facts.update({"length_accuracy.weakest_length": weakest_length,
                  "length_accuracy.weakest_accuracy": facts[f"length_accuracy.frames_{weakest_length}.accuracy"],
                  "length_accuracy.weakest_sequences": len(rows_by_length[weakest_length]),
                  "length_accuracy.perfect_lengths": sum(accuracy == 1.0 for accuracy in accuracies.values())})
    return facts


def draw_length_figure(output_path: Path, facts: Dict[str, Any], title: str) -> Dict[str, Any]:
    """
    Draws the accuracy and the mean probability of the true class of the flatten run on the test sequences of every
    number of frames, both on the same 0 to 1 scale, written in percent on the left axis for the accuracy and with
    decimals on the right axis for the probability, the number of test sequences of every length under its tick;
    gives the box of the panel.

    Args:
        output_path (Path): The PNG file.
        facts (Dict[str, Any]): The facts of compute_length_facts.
        title (str): The title of the panel, naming the run and its evaluated iteration.

    Returns:
        Dict[str, Any]: The box, named length_accuracy.panel_box.
    """
    register_fonts()
    figure, axes = pyplot.subplots(figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH),
                                   dpi=DOTS_PER_INCH)
    figure.patch.set_facecolor(SURFACE_COLOUR)
    # Margins leaving room for the probability axis on the right and the two lines of the length ticks
    figure.subplots_adjust(left=0.07, right=0.93, top=0.84, bottom=0.17)
    lengths = facts["length_accuracy.lengths"]
    series = [("accuracy", "Accuracy", SERIES_COLOURS[0]),
              ("mean_true_probability", "Mean probability of the true class", SERIES_COLOURS[1])]
    for fact_name, label, colour in series:
        values = [facts[f"length_accuracy.frames_{length}.{fact_name}"] for length in lengths]
        axes.plot(lengths, values, color=colour, linewidth=3 * POINTS_PER_PIXEL, marker="o",
                  markersize=10 * POINTS_PER_PIXEL, markeredgecolor=SURFACE_COLOUR,
                  markeredgewidth=2 * POINTS_PER_PIXEL)
        # The accuracy of every length written as the percentages of the reports, like its axis
        if fact_name == "accuracy":
            for length, value in zip(lengths, values):
                axes.annotate(text=format_percentage(value=value), xy=(length, value), xytext=(0, 12),
                              textcoords="offset points", ha="center", va="bottom", color=TEXT_PRIMARY_COLOUR,
                              fontsize=16 * POINTS_PER_PIXEL)
    style_chart_panel(axes=axes, title=title, x_label="Frames of the test sequence (test sequences of that length)",
                      y_label="Accuracy")
    axes.set_xticks(ticks=lengths, labels=[f"{length}\n({facts[f'length_accuracy.frames_{length}.sequences']})"
                                           for length in lengths])
    axes.set_ylim(bottom=0.0, top=1.1)
    # Accuracy ticks in percent on the left, the same scale with the decimals of the probability on the right
    axes.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))
    legend_handles = [Line2D(xdata=[0], ydata=[0], color=colour, linewidth=3 * POINTS_PER_PIXEL, marker="o",
                             markersize=8 * POINTS_PER_PIXEL) for _, _, colour in series]
    figure.legend(handles=legend_handles, labels=[label for _, label, _ in series], loc="center right",
                  bbox_to_anchor=(0.93, 0.93), ncols=2, frameon=False, labelcolor=TEXT_PRIMARY_COLOUR,
                  fontsize=18 * POINTS_PER_PIXEL)
    _, box = add_probability_axis(figure=figure, axes=axes, y_label="Mean probability of the true class", margin=6)
    save_figure(figure=figure, output_path=output_path)
    pyplot.close(fig=figure)
    return {"length_accuracy.panel_box": box}


def describe_wrong_predictions(predictions: List[Dict[str, str]], prefix: str) -> Dict[str, Any]:
    """
    Gives the wrong predictions of an evaluation: their number, and the most confident of them (the highest
    probability given to a wrong class) with its true and predicted classes, for the report scene.

    Args:
        predictions (List[Dict[str, str]]): The rows of predictions.csv of the evaluation.
        prefix (str): The start of the names of the facts, e.g. "evaluation.flatten_256.first".

    Returns:
        Dict[str, Any]: The facts '<prefix>.wrong_count', and '<prefix>.wrong.*' when a sequence is wrong.
    """
    wrong_rows = [row for row in predictions if row["correct"] == "0"]
    facts = {f"{prefix}.wrong_count": len(wrong_rows)}
    if wrong_rows:
        confident_row = max(wrong_rows, key=lambda row: float(row[f"probability_{row['predicted_class']}"]))
        facts.update({f"{prefix}.wrong.sequence": confident_row["sequence"],
                      f"{prefix}.wrong.number": int(confident_row["sequence"].split("_")[-1]),
                      f"{prefix}.wrong.true_class": confident_row["true_class"],
                      f"{prefix}.wrong.predicted_class": confident_row["predicted_class"],
                      f"{prefix}.wrong.probability": round_fact(
                          value=float(confident_row[f"probability_{confident_row['predicted_class']}"])),
                      f"{prefix}.wrong.true_probability": round_fact(
                          value=float(confident_row[f"probability_{confident_row['true_class']}"]))})
    return facts


def make_evaluation_figures(output_folder: Path, run_folders: Dict[str, Path], log_paths: Dict[str, Path],
                            evaluation_iterations: Dict[str, int], test_folder: Path) -> Dict[str, Any]:
    """
    Makes the training and evaluation figures of the three runs and gives their facts: the metrics of every evaluated
    checkpoint of the flatten run at 256 pixels and of the average run, the accuracy of the training evaluations of the
    flatten run at 512 pixels, the throughput and interval losses of the console logs, the first and last training
    losses, the wrong predictions of the flatten run at its first and last evaluations, the probabilities of the
    predictions files and the accuracy by number of frames.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folders (Dict[str, Path]): The folders of the runs MAIN_RUN_NAME, AVERAGE_RUN_NAME and 'flatten_512'.
        log_paths (Dict[str, Path]): The console logs of the same runs.
        evaluation_iterations (Dict[str, int]): The last checkpoint of every run, evaluated for the flatten run at 256
            pixels and the average run.
        test_folder (Path): The test split of the runs, holding labels.json.

    Returns:
        Dict[str, Any]: The facts, named evaluation.*, launch.*, console.*, training.*, probabilities.*,
            length_accuracy.* and samples.*.

    Raises:
        ValueError: If the average run predicts several classes, if its accuracy is not the share of its class, if the
            last evaluation of the flatten run is not its last checkpoint, or if the sample sequence is not right with
            flatten, wrong with average pooling and long enough.
    """
    # Metrics of the evaluated checkpoints of the two runs whose evaluations the scenes show, and the accuracies of
    # the evaluations of the training of the flatten run at 512 pixels
    facts = {}
    for run_name in EVALUATED_RUNS:
        facts.update(read_test_metrics(evaluation_root=run_folders[run_name] / CHECKPOINT_EVALUATION_FOLDER_NAME,
                                       run_name=run_name, scalar_names=list(TEST_SCALARS)))
    facts.update(read_test_metrics(evaluation_root=run_folders[FLATTEN_512_RUN_NAME], run_name=FLATTEN_512_RUN_NAME,
                                   scalar_names=["Accuracy"]))
    flatten_512_iterations, flatten_512_accuracies = read_scalar_curve(run_folder=run_folders[FLATTEN_512_RUN_NAME],
                                                                       split="Test", tag="Accuracy")
    facts.update({f"evaluation.{FLATTEN_512_RUN_NAME}.last_iteration": int(flatten_512_iterations[-1]),
                  f"evaluation.{FLATTEN_512_RUN_NAME}.last.accuracy": round_fact(value=flatten_512_accuracies[-1]),
                  f"evaluation.{FLATTEN_512_RUN_NAME}.best.accuracy": round_fact(
                      value=float(numpy.max(flatten_512_accuracies)))})

    # Launch information of the flatten run: its split descriptions and the length of an epoch
    launch_lines = extract_launch_block(log_path=log_paths[MAIN_RUN_NAME], last_line_start="- Batch Size")
    launch_boxes = draw_console_figure(lines=launch_lines, output_path=output_folder / "evaluation" /
                                       "launch_console.png", font_size=17,
                                       highlight_patterns={"splits": r"Data Folder", "epoch": r"Iterations Per Epoch"})
    facts.update({f"launch.{name}": box for name, box in launch_boxes.items()})
    epoch_line = next(line for line in launch_lines if line.startswith("- Iterations Per Epoch"))
    facts["launch.iterations_per_epoch"] = int(epoch_line.split(":")[-1])

    # Console summary of the first interval of the flatten run, its header, metric and interval lines written by the
    # current formatter with their logged values, and the throughputs and losses of every log
    logged_lines = extract_interval_block(log_path=log_paths[MAIN_RUN_NAME], iteration=100)
    metric_tracker = build_console_metric_tracker(
        run_folder=run_folders[MAIN_RUN_NAME], trainer_class=SequenceClassificationTrainer,
        trainer_attributes={"decorrelation_activated": False})
    console_lines = reformat_console_summary(lines=logged_lines, metric_tracker=metric_tracker)
    console_boxes = draw_console_figure(lines=console_lines, output_path=output_folder / "evaluation" /
                                        "training_console.png", font_size=17,
                                        highlight_patterns={"header": r"Iteration:\s+100", "train_loss":
                                                            r"Train Total Loss", "accuracy": r" Accuracy +:",
                                                            "validation": r"Validation Iter",
                                                            "interval": r"Interval Of"})
    facts.update({f"console.{name}": box for name, box in console_boxes.items()})

    # Interval length and validation batches of that summary, as its last lines print them
    validation_match = next(CONSOLE_VALIDATION_LINE_PATTERN.match(string=line) for line in console_lines
                            if line.startswith("Validation Iterations"))
    interval_match = next(CONSOLE_INTERVAL_LINE_PATTERN.match(string=line) for line in console_lines
                          if line.startswith("Interval Of"))
    facts.update({f"console.{MAIN_RUN_NAME}.validation_batches": int(parse_console_number(
                      text=validation_match.group("validation_count"))),
                  f"console.{MAIN_RUN_NAME}.validation_ratio": int(parse_console_number(
                      text=validation_match.group("validation_ratio"))),
                  f"console.{MAIN_RUN_NAME}.interval_iterations": int(parse_console_number(
                      text=interval_match.group("interval_iterations")))})

    # Throughputs and interval losses of the console log of every run
    for run_name, log_path in log_paths.items():
        summaries = read_interval_summaries(log_path=log_path)
        throughputs = [summary["throughput"] for summary in summaries]
        losses = [summary["train_total_loss"] for summary in summaries]
        facts.update({f"console.{run_name}.first_throughput": summaries[0]["throughput"],
                      f"console.{run_name}.first_frame_throughput": summaries[0]["frame_throughput"],
                      f"console.{run_name}.minimum_throughput": min(throughputs),
                      f"console.{run_name}.maximum_throughput": max(throughputs),
                      f"console.{run_name}.median_throughput": round_fact(value=float(numpy.median(throughputs))),
                      f"console.{run_name}.first_interval_loss": losses[0],
                      f"console.{run_name}.last_interval_loss": losses[-1],
                      f"console.{run_name}.minimum_interval_loss": min(losses),
                      f"console.{run_name}.maximum_interval_loss": max(losses),
                      f"console.{run_name}.last_interval_iteration": summaries[-1]["iteration"]})

    # Training loss of the three runs, raw from TensorBoard, every run named by its pooling and frame size
    run_colours = {MAIN_RUN_NAME: SERIES_COLOURS[0], AVERAGE_RUN_NAME: SERIES_COLOURS[2],
                   FLATTEN_512_RUN_NAME: SERIES_COLOURS[3]}
    run_labels, curves = {}, {}
    for run_name, colour in run_colours.items():
        run_configuration = read_run_configuration(run_folder=run_folders[run_name])
        pooling = run_configuration["ClassificationHead"]["global_pooling"]
        image_size = run_configuration["Data"]["image_settings"]["size"]
        run_labels[run_name] = (f"{pooling.capitalize()}, {image_size} pixels", colour)
        # Losses up to the last checkpoint of the run, a run stopped right after an evaluation having logged a few
        # iterations past it
        iterations, values = read_scalar_curve(run_folder=run_folders[run_name], split="Train", tag="Total Loss")
        kept_steps = iterations <= evaluation_iterations[run_name]
        iterations, values = iterations[kept_steps], values[kept_steps]
        curves[run_name] = {"label": run_labels[run_name][0], "pooling": pooling, "image_size": image_size,
                            "colour": colour, "iterations": iterations, "values": values}
        facts[f"training.{run_name}.first_loss"] = round_fact(value=values[0])
        facts[f"training.{run_name}.last_loss"] = round_fact(value=values[-1])
        facts[f"training.{run_name}.smoothed_last_loss"] = round_fact(
            value=smooth_values(values=values, weight=SMOOTHING_WEIGHT)[-1])
        facts[f"training.{run_name}.iterations"] = int(iterations[-1])
    number_classes = len(set(json.loads((run_folders[MAIN_RUN_NAME] / "label_dictionary.json").read_text(
        encoding="utf-8")).values()))
    facts.update(draw_loss_figure(output_folder=output_folder, curves=curves, number_classes=number_classes))

    # Confusion matrices and ROC curves of the flatten and average runs at their last evaluation
    composite_rows, row_headings = [], []
    for run_name in EVALUATED_RUNS:
        evaluation_folder = (run_folders[run_name] / CHECKPOINT_EVALUATION_FOLDER_NAME /
                             f"Iteration_{evaluation_iterations[run_name]}")
        composite_rows.append([evaluation_folder / "confusion_matrix.png", evaluation_folder / "roc_curves.png"])
        row_headings.append(f"{run_labels[run_name][0]}, iteration {evaluation_iterations[run_name]:,}")
    composite_facts = compose_image_grid(rows=composite_rows, row_headings=row_headings,
                                         column_headings=["confusion_matrix.png", "roc_curves.png"],
                                         image_height=640, output_path=output_folder / "evaluation" /
                                         "confusion_and_roc.png")
    facts.update({f"evaluation.composite.{name}": box for name, box in composite_facts.items()})

    # Probability of the true class of every test sequence at the last evaluation of both runs
    probability_sets, last_predictions = {}, {}
    for run_name in EVALUATED_RUNS:
        predictions = read_predictions(run_folder=run_folders[run_name], iteration=evaluation_iterations[run_name],
                                       evaluation_folder_name=CHECKPOINT_EVALUATION_FOLDER_NAME)
        last_predictions[run_name] = predictions
        true_probabilities = compute_true_probabilities(predictions=predictions)
        all_probabilities = numpy.array([float(value) for row in predictions for key, value in row.items()
                                         if key.startswith("probability_")])
        probability_sets[run_name] = {"title": row_headings[len(probability_sets)], "colour":
                                      run_labels[run_name][1], "probabilities": true_probabilities}
        # Lowest and median probability of the true class, the confident sequences and the range of every probability
        facts.update({f"probabilities.{run_name}.minimum_true": round_fact(value=true_probabilities.min()),
                      f"probabilities.{run_name}.median_true": round_fact(value=numpy.median(true_probabilities)),
                      f"probabilities.{run_name}.above_0_9": int((true_probabilities > CONFIDENT_PROBABILITY).sum()),
                      f"probabilities.{run_name}.minimum_any": round_fact(value=all_probabilities.min()),
                      f"probabilities.{run_name}.maximum_any": round_fact(value=all_probabilities.max()),
                      f"probabilities.{run_name}.predicted_classes": sorted({row["predicted_class"]
                                                                             for row in predictions}),
                      f"probabilities.{run_name}.sequences": len(predictions)})
    facts.update(draw_probability_figure(output_path=output_folder / "evaluation" / "true_class_probabilities.png",
                                         probability_sets=probability_sets))
    facts["probabilities.threshold"] = CONFIDENT_PROBABILITY

    # The single class the average run gives every test sequence, whose share of the test split is its accuracy
    average_classes = facts[f"probabilities.{AVERAGE_RUN_NAME}.predicted_classes"]
    if len(average_classes) != 1:
        raise ValueError(f"The average pooling run predicts {average_classes} at iteration "
                         f"{evaluation_iterations[AVERAGE_RUN_NAME]}, while the evaluation scenes say it gives one "
                         f"class to every test sequence.")
    average_rows = last_predictions[AVERAGE_RUN_NAME]
    class_sequences = sum(row["true_class"] == average_classes[0] for row in average_rows)
    average_iteration = evaluation_iterations[AVERAGE_RUN_NAME]
    average_accuracy = facts[f"evaluation.{AVERAGE_RUN_NAME}.accuracy.iteration_{average_iteration}"]
    if abs(class_sequences / len(average_rows) - average_accuracy) > 1e-4:
        raise ValueError(f"The accuracy of the average run, {average_accuracy}, is not the share of "
                         f"{average_classes[0]} in the test split, {class_sequences / len(average_rows)}.")
    facts.update({f"probabilities.{AVERAGE_RUN_NAME}.predicted_class": average_classes[0],
                  f"probabilities.{AVERAGE_RUN_NAME}.predicted_class_sequences": class_sequences})

    # Wrong predictions of the first and last evaluated checkpoints of the flatten run, and the class its first Per
    # Class Recall table writes first (the lowest recall, ties in class order)
    flatten_iterations = find_evaluated_iterations(run_folder=run_folders[MAIN_RUN_NAME])
    # The last evaluation is the one the confusion, probability, sample and length figures draw, so that the narration
    # quoting it by its role describes these figures
    if flatten_iterations[-1] != evaluation_iterations[MAIN_RUN_NAME]:
        raise ValueError(f"The last evaluated checkpoint of the flatten run, {flatten_iterations[-1]}, is not its last "
                         f"checkpoint, {evaluation_iterations[MAIN_RUN_NAME]}, which the figures draw.")
    facts.update({f"evaluation.{MAIN_RUN_NAME}.first_iteration": flatten_iterations[0],
                  f"evaluation.{MAIN_RUN_NAME}.last_iteration": flatten_iterations[-1]})
    for position_name, iteration in [("first", flatten_iterations[0]), ("last", flatten_iterations[-1])]:
        # Headline metrics of the evaluation named by its place, so that the narration follows the evaluated
        # checkpoints of the run rather than typed iterations
        for fact_name in TEST_SCALARS.values():
            facts[f"evaluation.{MAIN_RUN_NAME}.{position_name}.{fact_name}"] = facts[
                f"evaluation.{MAIN_RUN_NAME}.{fact_name}.iteration_{iteration}"]
        predictions = read_predictions(run_folder=run_folders[MAIN_RUN_NAME], iteration=iteration,
                                       evaluation_folder_name=CHECKPOINT_EVALUATION_FOLDER_NAME)
        facts.update(describe_wrong_predictions(predictions=predictions,
                                                prefix=f"evaluation.{MAIN_RUN_NAME}.{position_name}"))
        class_recalls = read_class_recalls(run_folder=run_folders[MAIN_RUN_NAME], iteration=iteration)
        worst_class = min(class_recalls, key=lambda class_recall: class_recall["recall"])
        facts.update({f"evaluation.{MAIN_RUN_NAME}.{position_name}.worst_recall_class": worst_class["class"],
                      f"evaluation.{MAIN_RUN_NAME}.{position_name}.worst_recall": round_fact(
                          value=worst_class["recall"]),
                      f"evaluation.{MAIN_RUN_NAME}.{position_name}.fewest_class_sequences": min(
                          class_recall["samples"] for class_recall in class_recalls),
                      f"evaluation.{MAIN_RUN_NAME}.{position_name}.most_class_sequences": max(
                          class_recall["samples"] for class_recall in class_recalls)})

    # Results.txt of the evaluation of the flatten run, read by the code scenes of the report
    results_path = run_folders[MAIN_RUN_NAME] / CHECKPOINT_EVALUATION_FOLDER_NAME / "Results.txt"
    write_text_lines(output_path=output_folder / "listings" / f"results_{MAIN_RUN_NAME}.txt",
                     lines=results_path.read_text(encoding="utf-8").splitlines())

    # Accuracy of the flatten run on the test sequences of every number of frames, at its last evaluation
    facts.update(compute_length_facts(predictions=last_predictions[MAIN_RUN_NAME], test_folder=test_folder))
    facts.update(draw_length_figure(output_path=output_folder / "evaluation" / "accuracy_by_length.png", facts=facts,
                                    title=f"{run_labels[MAIN_RUN_NAME][0]}, iteration "
                                          f"{evaluation_iterations[MAIN_RUN_NAME]:,}, on the "
                                          f"{len(last_predictions[MAIN_RUN_NAME])} test sequences"))

    # The same sample sequence drawn by the evaluation of both runs, right with flatten and wrong with average pooling
    sample_facts = compose_image_grid(
        rows=[[run_folders[run_name] / CHECKPOINT_EVALUATION_FOLDER_NAME /
               f"Iteration_{evaluation_iterations[run_name]}" / "Samples" / f"{SAMPLE_SEQUENCE}.png"
               for run_name in EVALUATED_RUNS]],
        row_headings=[f"Samples/{SAMPLE_SEQUENCE}.png"], column_headings=[row_headings[0], row_headings[1]],
        image_height=400, output_path=output_folder / "evaluation" / "samples.png")
    facts.update({f"samples.{name}": box for name, box in sample_facts.items()})

    # Frames of the sample sequence, then the class, correctness and probability every run gives it, checked against
    # what the sample scene says
    test_manifest = json.loads((test_folder / "labels.json").read_text(encoding="utf-8"))
    sample_entry = next(sequence for sequence in test_manifest["sequences"] if sequence["name"] == SAMPLE_SEQUENCE)
    facts.update({"samples.sequence": SAMPLE_SEQUENCE, "samples.frames": len(sample_entry["frames"])})
    for run_name in EVALUATED_RUNS:
        sample_row = next(row for row in last_predictions[run_name] if row["sequence"] == SAMPLE_SEQUENCE)
        facts.update({f"samples.{run_name}.true_class": sample_row["true_class"],
                      f"samples.{run_name}.predicted_class": sample_row["predicted_class"],
                      f"samples.{run_name}.correct": sample_row["correct"] == "1",
                      f"samples.{run_name}.probability": round_fact(
                          value=float(sample_row[f"probability_{sample_row['predicted_class']}"]))})
    if not (facts[f"samples.{MAIN_RUN_NAME}.correct"] and not facts[f"samples.{AVERAGE_RUN_NAME}.correct"]
            and facts["samples.frames"] >= SAMPLE_MINIMUM_FRAMES):
        raise ValueError(f"The sample sequence {SAMPLE_SEQUENCE} ({facts['samples.frames']} frames) should be right "
                         f"with the flatten run, wrong with the average run and hold at least {SAMPLE_MINIMUM_FRAMES} "
                         f"frames, as the sample scene says: choose another of the fixed sample sequences.")
    return facts
