"""
Training and evaluation figures of the sequence classification tutorials (chapters 1 and 3), made by
tutorials/figures/sequence_classification/figures.py from the three README runs of the task: the current flatten run
at 128 pixels, the average pooling run at 512 pixels and the flatten run at 512 pixels. It draws the launch information
and the console summary of the flatten run, the training loss of the three runs (an empty-axes base and its overlay,
uncovered one panel at a time), the confusion matrices and ROC curves of the flatten and average runs side by side, the
probability of the true class of every test sequence from their predictions files, and one sample sequence of both
runs; its facts are the metrics of every evaluation, the throughputs and losses of the logs and the predictions the
narration quotes. The evaluations of the flatten run at 128 pixels and the average run are read from their
Checkpoint_Evaluation folder, where scripts/evaluation/evaluate_checkpoint.py evaluated their checkpoints again with
the current code, so that the report, the images and the metrics show what the current evaluation writes.
"""

import math
import re

import numpy

from matplotlib import pyplot
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, POINTS_PER_PIXEL,
                                                REVEAL_MARGIN, SERIES_COLOURS, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR,
                                                TEXT_SECONDARY_COLOUR, compute_pixel_box,
                                                format_thousands, register_fonts, round_fact, save_figure,
                                                style_chart_panel)
from tutorials.figures.foundations.training_curves import smooth_values
from vision_training.evaluation.checkpoint_evaluation import CHECKPOINT_EVALUATION_FOLDER_NAME
from vision_training.evaluation.classification.classification_metrics import TOP_K_ACCURACY_CLASSES
from tutorials.figures.sequence_classification.sequence_figure_utilities import (compose_image_grid,
                                                                                 draw_console_figure,
                                                                                 extract_interval_block,
                                                                                 extract_launch_block,
                                                                                 read_interval_summaries,
                                                                                 read_predictions, read_scalar_curve,
                                                                                 write_text_lines)

# Smoothing of the training curves, the exponential moving average of TensorBoard
SMOOTHING_WEIGHT = 0.95

# Headline scalars of the test evaluations kept as facts, with the names of their facts, the k of the top-k accuracy
# being the fixed one of the current evaluation
TEST_SCALARS = {"Accuracy": "accuracy", "Balanced Accuracy": "balanced_accuracy", "Macro F1": "macro_f1",
                f"Top-{TOP_K_ACCURACY_CLASSES} Accuracy": f"top_{TOP_K_ACCURACY_CLASSES}_accuracy",
                "Macro ROC AUC": "macro_roc_auc", "Macro Average Precision": "macro_average_precision"}

# Runs whose evaluations the scenes show, evaluated again into their Checkpoint_Evaluation folder
EVALUATED_RUNS = ["flatten_128", "average_512"]

# Sample sequence compared between the runs, one of the 8 fixed test sequences of both
SAMPLE_SEQUENCE = "test_000000"

# Bins of the histograms of the probability of the true class
PROBABILITY_BINS = numpy.linspace(start=0.0, stop=1.0, num=41)

# Probability of the true class above which a test sequence counts as confidently right
CONFIDENT_PROBABILITY = 0.9

# Folder of the evaluation of a checkpoint, named after its iteration
EVALUATION_FOLDER_PATTERN = re.compile(pattern=r"^Iteration_(\d+)$")


def find_first_evaluated_iteration(run_folder: Path) -> int:
    """
    Finds the first evaluated checkpoint of a run, the earliest Checkpoint_Evaluation/Iteration_<n> folder, whose
    wrong predictions the report scene quotes, so that the facts follow the evaluations of the run rather than a typed
    iteration.

    Args:
        run_folder (Path): The run folder.

    Returns:
        int: The iteration of the first evaluation.

    Raises:
        FileNotFoundError: If the run holds no evaluation folder.
    """
    evaluation_root = run_folder / CHECKPOINT_EVALUATION_FOLDER_NAME
    iterations = [int(match.group(1)) for folder in evaluation_root.iterdir()
                  if (match := EVALUATION_FOLDER_PATTERN.match(folder.name)) is not None and folder.is_dir()]
    if not iterations:
        raise FileNotFoundError(f"No Iteration_<n> folder in {evaluation_root}.")
    return min(iterations)


def read_test_metrics(run_folder: Path, run_name: str) -> Dict[str, Any]:
    """
    Reads the headline metrics of every evaluated checkpoint of a run from the TensorBoard logs of its
    Checkpoint_Evaluation folder, as facts named evaluation.<run name>.<metric>.iteration_<n>.

    Args:
        run_folder (Path): The run folder.
        run_name (str): The short name of the run in the facts, e.g. "flatten_128".

    Returns:
        Dict[str, Any]: The facts.
    """
    facts = {}
    for scalar_name, fact_name in TEST_SCALARS.items():
        iterations, values = read_scalar_curve(run_folder=run_folder / CHECKPOINT_EVALUATION_FOLDER_NAME, split="Test",
                                               tag=scalar_name)
        for iteration, value in zip(iterations, values):
            facts[f"evaluation.{run_name}.{fact_name}.iteration_{int(iteration)}"] = round_fact(value=value)
    return facts


def draw_loss_figure(output_folder: Path, curves: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Draws the training total loss of the three runs on two panels, the flatten run at 128 pixels on the left and the
    two runs at 512 pixels on the right, every curve smoothed like TensorBoard over its faint raw values, with the line
    of ln 4, the loss of a uniform guess over the four classes. The empty axes are written as the base image and the
    whole chart as its overlay; gives the boxes of the panels and their reveal regions.

    Args:
        output_folder (Path): The figures folder of the series.
        curves (Dict[str, Dict[str, Any]]): For every run name, its 'label', 'colour', 'iterations' and 'values'.

    Returns:
        Dict[str, Any]: The boxes, named training.*.
    """
    register_fonts()
    figure, panels = pyplot.subplots(nrows=1, ncols=2, figsize=(CHART_WIDTH / DOTS_PER_INCH,
                                                                CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH)
    figure.patch.set_facecolor(SURFACE_COLOUR)
    figure.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.12, wspace=0.14)
    panel_runs = [["flatten_128"], ["average_512", "flatten_512"]]
    panel_titles = ["Flatten pooling, 128 pixels", "512 pixels: average and flatten pooling"]
    chance_loss = math.log(4)

    # Panels with the chance line, the legend and the axes, saved empty as the base image of the scene
    for axes, run_names, title in zip(panels, panel_runs, panel_titles):
        style_chart_panel(axes=axes, title=title, x_label="Iteration", y_label="Training total loss"
                          if axes is panels[0] else None)
        last_iteration = max(int(curves[run_name]["iterations"][-1]) for run_name in run_names)
        axes.set_xlim(left=0, right=last_iteration)
        axes.set_ylim(bottom=0.0, top=1.6)
        axes.xaxis.set_major_formatter(formatter=FuncFormatter(func=format_thousands))
        axes.axhline(y=chance_loss, color=TEXT_SECONDARY_COLOUR, linewidth=1.5 * POINTS_PER_PIXEL, linestyle="--")
        axes.annotate(text="ln 4 = 1.386, a uniform guess", xy=(last_iteration, chance_loss), xytext=(-8, 10),
                      textcoords="offset points", ha="right", va="bottom", color=TEXT_SECONDARY_COLOUR,
                      fontsize=17 * POINTS_PER_PIXEL)
    legend_handles = [Line2D(xdata=[0], ydata=[0], color=curves[run_name]["colour"], linewidth=3 * POINTS_PER_PIXEL)
                      for run_name in ["flatten_128", "average_512", "flatten_512"]]
    figure.legend(handles=legend_handles, labels=[curves[run_name]["label"] for run_name in
                                                  ["flatten_128", "average_512", "flatten_512"]],
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
    Gives the probability the model gave to the true class of every test sequence of a predictions file.

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


def make_evaluation_figures(output_folder: Path, run_folders: Dict[str, Path], log_paths: Dict[str, Path],
                            evaluation_iterations: Dict[str, int]) -> Dict[str, Any]:
    """
    Makes the training and evaluation figures of the three runs and gives their facts: the metrics of every evaluated
    checkpoint of the flatten run at 128 pixels and the average run, the throughput and interval losses of the console
    logs, the first and last training losses, the wrong predictions of the flatten run at its first evaluation, and the
    probabilities of the predictions files.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folders (Dict[str, Path]): The folders of the runs 'flatten_128', 'average_512' and 'flatten_512'.
        log_paths (Dict[str, Path]): The console logs of the same runs.
        evaluation_iterations (Dict[str, int]): The last evaluated iteration of every run.

    Returns:
        Dict[str, Any]: The facts, named evaluation.*, launch.*, console.*, training.*, probabilities.* and
            samples.*.
    """
    # Metrics of the evaluated checkpoints of the two runs whose evaluations the scenes show
    facts = {}
    for run_name in EVALUATED_RUNS:
        facts.update(read_test_metrics(run_folder=run_folders[run_name], run_name=run_name))

    # Launch information of the flatten run: its split descriptions and the length of an epoch
    launch_lines = extract_launch_block(log_path=log_paths["flatten_128"], last_line_start="- Batch Size")
    launch_boxes = draw_console_figure(lines=launch_lines, output_path=output_folder / "evaluation" /
                                       "launch_console.png", font_size=17,
                                       highlight_patterns={"splits": r"Data Folder", "epoch": r"Iterations Per Epoch"})
    facts.update({f"launch.{name}": box for name, box in launch_boxes.items()})
    epoch_line = next(line for line in launch_lines if line.startswith("- Iterations Per Epoch"))
    facts["launch.iterations_per_epoch"] = int(epoch_line.split(":")[-1])

    # Console summary of the first interval of the flatten run, and the throughputs and losses of every log
    console_lines = extract_interval_block(log_path=log_paths["flatten_128"], iteration=100)
    console_boxes = draw_console_figure(lines=console_lines, output_path=output_folder / "evaluation" /
                                        "training_console.png", font_size=17,
                                        highlight_patterns={"header": r"Iteration:\s+100", "train_loss":
                                                            r"Train Total Loss", "validation": r"Validation Iter",
                                                            "interval": r"Interval Of"})
    facts.update({f"console.{name}": box for name, box in console_boxes.items()})

    # Interval length and validation batches of that summary, as its last lines print them
    validation_match = next(re.search(pattern=r"Validation Iterations In This Interval: (\d+) \(Ratio: Every (\d+)",
                                      string=line) for line in console_lines if line.startswith("Validation Iterations"))
    interval_match = next(re.search(pattern=r"Interval Of (\d+) Iterations", string=line) for line in console_lines
                          if line.startswith("Interval Of"))
    facts.update({"console.flatten_128.validation_batches": int(validation_match.group(1)),
                  "console.flatten_128.validation_ratio": int(validation_match.group(2)),
                  "console.flatten_128.interval_iterations": int(interval_match.group(1))})
    for run_name, log_path in log_paths.items():
        summaries = read_interval_summaries(log_path=log_path)
        throughputs = [summary["throughput"] for summary in summaries]
        losses = [summary["train_total_loss"] for summary in summaries]
        facts.update({f"console.{run_name}.first_throughput": summaries[0]["throughput"],
                      f"console.{run_name}.first_frame_throughput": summaries[0]["frame_throughput"],
                      f"console.{run_name}.minimum_throughput": min(throughputs),
                      f"console.{run_name}.maximum_throughput": max(throughputs),
                      f"console.{run_name}.first_interval_loss": losses[0],
                      f"console.{run_name}.last_interval_loss": losses[-1],
                      f"console.{run_name}.minimum_interval_loss": min(losses),
                      f"console.{run_name}.maximum_interval_loss": max(losses)})

    # Training loss of the three runs, raw from TensorBoard
    run_labels = {"flatten_128": ("Flatten, 128 pixels", SERIES_COLOURS[0]),
                  "average_512": ("Average, 512 pixels", SERIES_COLOURS[2]),
                  "flatten_512": ("Flatten, 512 pixels", SERIES_COLOURS[3])}
    curves = {}
    for run_name, (label, colour) in run_labels.items():
        iterations, values = read_scalar_curve(run_folder=run_folders[run_name], split="Train", tag="Total Loss")
        curves[run_name] = {"label": label, "colour": colour, "iterations": iterations, "values": values}
        facts[f"training.{run_name}.first_loss"] = round_fact(value=values[0])
        facts[f"training.{run_name}.last_loss"] = round_fact(value=values[-1])
        facts[f"training.{run_name}.smoothed_last_loss"] = round_fact(
            value=smooth_values(values=values, weight=SMOOTHING_WEIGHT)[-1])
        facts[f"training.{run_name}.iterations"] = int(iterations[-1])
    facts.update(draw_loss_figure(output_folder=output_folder, curves=curves))

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

    # Probability of the true class of every test sequence, and the wrong predictions of every evaluation
    probability_sets = {}
    for run_name in EVALUATED_RUNS:
        predictions = read_predictions(run_folder=run_folders[run_name], iteration=evaluation_iterations[run_name],
                                       evaluation_folder_name=CHECKPOINT_EVALUATION_FOLDER_NAME)
        true_probabilities = compute_true_probabilities(predictions=predictions)
        all_probabilities = numpy.array([float(value) for row in predictions for key, value in row.items()
                                         if key.startswith("probability_")])
        probability_sets[run_name] = {"title": row_headings[len(probability_sets)], "colour":
                                      run_labels[run_name][1], "probabilities": true_probabilities}
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
    # Wrong predictions of the first evaluated checkpoint of the flatten run
    first_iteration = find_first_evaluated_iteration(run_folder=run_folders["flatten_128"])
    first_predictions = read_predictions(run_folder=run_folders["flatten_128"], iteration=first_iteration,
                                         evaluation_folder_name=CHECKPOINT_EVALUATION_FOLDER_NAME)
    wrong_rows = [row for row in first_predictions if row["correct"] == "0"]
    facts.update({"probabilities.threshold": CONFIDENT_PROBABILITY,
                  "evaluation.flatten_128.first_iteration": first_iteration})
    facts["evaluation.flatten_128.wrong_at_first_iteration"] = [row["sequence"] for row in wrong_rows]
    if wrong_rows:
        facts.update({"evaluation.flatten_128.first_wrong.sequence": wrong_rows[0]["sequence"],
                      "evaluation.flatten_128.first_wrong.true_class": wrong_rows[0]["true_class"],
                      "evaluation.flatten_128.first_wrong.predicted_class": wrong_rows[0]["predicted_class"],
                      "evaluation.flatten_128.first_wrong.probability": round_fact(
                          value=float(wrong_rows[0][f"probability_{wrong_rows[0]['predicted_class']}"])),
                      "evaluation.flatten_128.first_wrong.true_probability": round_fact(
                          value=float(wrong_rows[0][f"probability_{wrong_rows[0]['true_class']}"])),
                      "evaluation.flatten_128.first_wrong.number": int(wrong_rows[0]["sequence"].split("_")[-1])})

    # Results.txt of the evaluation of the flatten run, read by the code scene of the report
    results_path = run_folders["flatten_128"] / CHECKPOINT_EVALUATION_FOLDER_NAME / "Results.txt"
    write_text_lines(output_path=output_folder / "listings" / "results_flatten_128.txt",
                     lines=results_path.read_text(encoding="utf-8").splitlines())

    # The same sample sequence drawn by the evaluation of both runs
    sample_facts = compose_image_grid(
        rows=[[run_folders[run_name] / CHECKPOINT_EVALUATION_FOLDER_NAME /
               f"Iteration_{evaluation_iterations[run_name]}" / "Samples" / f"{SAMPLE_SEQUENCE}.png"
               for run_name in EVALUATED_RUNS]],
        row_headings=[f"Samples/{SAMPLE_SEQUENCE}.png"], column_headings=[row_headings[0], row_headings[1]],
        image_height=400, output_path=output_folder / "evaluation" / "samples.png")
    facts.update({f"samples.{name}": box for name, box in sample_facts.items()})
    facts["samples.sequence"] = SAMPLE_SEQUENCE
    for run_name in EVALUATED_RUNS:
        sample_row = next(row for row in read_predictions(run_folder=run_folders[run_name],
                                                          iteration=evaluation_iterations[run_name],
                                                          evaluation_folder_name=CHECKPOINT_EVALUATION_FOLDER_NAME)
                          if row["sequence"] == SAMPLE_SEQUENCE)
        facts.update({f"samples.{run_name}.true_class": sample_row["true_class"],
                      f"samples.{run_name}.predicted_class": sample_row["predicted_class"],
                      f"samples.{run_name}.probability": round_fact(
                          value=float(sample_row[f"probability_{sample_row['predicted_class']}"]))})
    return facts
