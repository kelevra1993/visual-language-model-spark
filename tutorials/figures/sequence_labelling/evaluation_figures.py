"""
Evaluation figures and facts of the sequence labelling tutorials (the chapters on the model and on reading an
evaluation), made at every build by tutorials/figures/sequence_labelling/figures.py from the real sequence labelling
run of tutorial_sources.yaml (pre-trained backbone, balanced class weights, moving average of its weights). The report
scenes read its last evaluation, from Checkpoint_Evaluation/ when scripts/evaluation/evaluate_checkpoint.py evaluated
it again with the per-class report and the predictions file (the training writing neither), from Evaluation/
otherwise, and the evaluation of the averaged weights when the run keeps them: the listings of the evaluation block in
Results.txt (the headline metrics with the head of the Per Class Metrics table, and the Per Class Recall table with
its intervals), the per-class precision, recall and intervals of class_metrics.csv, the confidence thresholds of
class_thresholds.json, one of its sample images with the boxes of its captions, the probability of no_change in
predictions.csv on the first and on the later frames and by frame position, the macro ROC AUC computed again on the
later frames only (whether the scores rank the frames by their content or by their position), and the values every
metric takes for a model predicting no_change on every test frame (the signature of a degenerate solution). Also the
segment metrics of the example of the tasks reference (documentation/reference/tasks_reference.md, formerly in the
README) computed by the code of the repository, and the shapes, residual blocks, head layers and parameter counts of
the trained model run on the CPU on a real padded batch. The shares of the frame position panel are written as the
percentages of the reports of the repository (ticks "20 %"), the probabilities keeping their decimals on its right axis.
A build from the committed snapshots (tutorials/run_snapshots, without the checkpoint and with the columns of
predictions.csv the sample image reads, compressed) reuses the committed prediction figure and model facts.
"""

import cv2
import json
import operator
import re
import shutil

import numpy
import pandas
import torch

from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter, NullFormatter
from pathlib import Path
from PIL import Image
from scipy.stats import rankdata
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, MONOSPACE_FONT_FAMILY,
                                                POINTS_PER_PIXEL, SERIES_COLOURS, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR,
                                                TEXT_SECONDARY_COLOUR, compute_pixel_box, is_snapshot_folder,
                                                restore_committed_figures, round_fact, register_fonts, save_figure,
                                                style_chart_panel, write_reused_facts)
from tutorials.figures.sequence_classification.sequence_figure_utilities import add_probability_axis
from tutorials.figures.sequence_labelling.plan_figures import make_plan_figures
from tutorials.figures.sequence_labelling.sequence_utilities import (build_test_dataset, collate_named_sequences,
                                                                     count_model_parameters, describe_run_source,
                                                                     load_sequence_model, parse_block_metrics,
                                                                     predict_sequences, read_results_block,
                                                                     read_run_reused_facts, read_run_sections,
                                                                     write_listing)
from vision_training.evaluation.checkpoint_evaluation import CHECKPOINT_EVALUATION_FOLDER_NAME
from vision_training.evaluation.report_formatting import format_percentage_tick
from vision_training.evaluation.sequence.segment_metrics import compute_segment_metrics
from vision_training.models.sequence.frame_encoder import encode_frames
from vision_training.trainers.checkpointing import get_checkpoint_path
from vision_training.trainers.sequence_classification_trainer import FRAME_GRID_COLUMNS
from vision_training.utilities.visualization import CAPTION_BAR_HEIGHT

# Name of the class of the frames without change, always the class of the first frame
NO_CHANGE_CLASS_NAME = "no_change"

# Number of rows of the class with the worst F1 kept at the head of the per-class table of the listing, before the
# no_change row, and of the classes with the worst and the best recall kept in the listing of the recall table
LISTED_WORST_ROWS = 2
LISTED_WORST_RECALL_ROWS = 3
LISTED_BEST_RECALL_ROWS = 2

# Characters kept of every listed line, the widest table line of the reports being 120 characters
LISTED_LINE_LENGTH = 120

# Example of the segment metrics of documentation/reference/tasks_reference.md (Sequence Labelling), formerly in the
# README, hence the names: a true sequence A A A A A B B B B B predicted A A B A A B B A B B, the classes A and B
# written as 0 and 1
README_TRUE_CLASSES = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
README_PREDICTED_CLASSES = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]

# Bins of the histogram of the probability of no_change, over the whole probability range
PROBABILITY_BIN_WIDTH = 0.01

# Note of the segments of an evaluation block, "Segments : 58133 true and 10000 predicted in ..."
SEGMENT_NOTE_PATTERN = re.compile(pattern=r"Segments : (\d+) true and (\d+) predicted")

# Folder name of an evaluation of the training, "Iteration_2000"
ITERATION_FOLDER_PATTERN = re.compile(pattern=r"^Iteration_(\d+)$")

# Name of a top-k accuracy fact read from a report, "evaluation.top_5_accuracy" (top-2 in the reports written before
# the top-5 accuracy)
TOP_K_FACT_PATTERN = re.compile(pattern=r"\.top_(\d+)_accuracy$")

# Subfolder of an iteration folder holding the evaluation of the moving average of the weights
AVERAGED_WEIGHTS_FOLDER_NAME = "EMA"

# Layout of the frame grids of the sample images (vision_training/utilities/visualization.py build_frame_grid): the
# gap around every frame, the height of the black caption bar of every frame, and the width of that bar, 12 pixels
# plus 9 per character of its text
FRAME_GRID_GAP = 4
FRAME_CAPTION_HEIGHT = 20
FRAME_CAPTION_BASE_WIDTH = 12
FRAME_CAPTION_CHARACTER_WIDTH = 9

# Font scale of the caption bar of a sample image (add_caption_bar), whose text width gives the box of the caption,
# and the margin of the box around the text
SAMPLE_CAPTION_FONT_SCALE = 0.45
SAMPLE_CAPTION_MARGIN = 8

# Recall under which a change class counts as poorly found in the narration of the recall table
LOW_RECALL_THRESHOLD = 0.5

# Shapes, with their hollow versions, that documentation/architecture_benchmarking/sequence_labelling_experiment_plan.md
# reports as not learned by the multi-label shape classifier the backbone of the run was pre-trained on
UNLEARNED_PRETRAINING_SHAPES = ["triangle", "parallelogram", "rhombus", "kite", "trapezoid"]

# Facts of the model a reused model_facts.json must hold, those added after its first version
REQUIRED_MODEL_FACTS = ["model.backbone_channels", "model.backbone_map_size", "model.flatten_features",
                        "model.temporal_features", "model.residual_blocks", "model.residual_block_count",
                        "model.head_layers"]


def find_evaluation_source(run_folder: Path) -> Dict[str, Any]:
    """
    Finds the evaluation the report scenes read: the last iteration evaluated again by
    scripts/evaluation/evaluate_checkpoint.py in Checkpoint_Evaluation/ (with the per-class report and the predictions
    file the training of the run does not write), or else the last one of the training in Evaluation/, and the
    evaluation of the moving average of the weights when the run keeps one, the weights the benchmark compares.

    Args:
        run_folder (Path): The folder of the run.

    Returns:
        Dict[str, Any]: The 'iteration', whether its weights are 'averaged', the 'results_path' of its Results.txt, the
            'evaluation_folder', the 'files_folder' holding its class_metrics.csv, class_thresholds.json,
            predictions.csv and Samples/, and the 'configuration_folder' holding the training_configuration.yaml and
            data.json of the evaluation (Checkpoint_Evaluation/ keeping those the evaluation script ran with, its test
            split included, the run folder for an evaluation of the training).

    Raises:
        FileNotFoundError: If the run holds no finished evaluation.
    """
    _, model_configuration = read_run_sections(run_folder=run_folder)
    averaged = bool(model_configuration["Optimization"]["exponential_moving_average"]["activated"])
    for evaluation_folder, results_path in [(run_folder / CHECKPOINT_EVALUATION_FOLDER_NAME,
                                             run_folder / CHECKPOINT_EVALUATION_FOLDER_NAME / "Results.txt"),
                                            (run_folder / "Evaluation", run_folder / "Results.txt")]:
        # Iterations of the folder whose evaluation of the chosen weights is finished
        iterations = []
        for folder in evaluation_folder.glob("Iteration_*"):
            match = ITERATION_FOLDER_PATTERN.match(string=folder.name)
            files_folder = folder / AVERAGED_WEIGHTS_FOLDER_NAME if averaged else folder
            if match is not None and (files_folder / "class_metrics.csv").is_file():
                iterations.append(int(match.group(1)))
        if iterations:
            iteration_folder = evaluation_folder / f"Iteration_{max(iterations)}"
            reevaluated = evaluation_folder.name == CHECKPOINT_EVALUATION_FOLDER_NAME
            return {"iteration": max(iterations), "averaged": averaged, "results_path": results_path,
                    "evaluation_folder": evaluation_folder,
                    "files_folder": iteration_folder / AVERAGED_WEIGHTS_FOLDER_NAME if averaged else iteration_folder,
                    "configuration_folder": evaluation_folder if reevaluated else run_folder}
    raise FileNotFoundError(f"No finished evaluation (Iteration_<n>/class_metrics.csv) in {run_folder}/"
                            f"{CHECKPOINT_EVALUATION_FOLDER_NAME} or {run_folder}/Evaluation.")


def read_test_split_name(configuration_folder: Path) -> str:
    """
    Reads the name of the test split an evaluation ran on (the last folder of its test_data_folder, "test" for the
    whole split, "test_10000" for the fixed subset), which the figure headers and the narration name.

    Args:
        configuration_folder (Path): The folder holding the training_configuration.yaml of the evaluation
            (find_evaluation_source).

    Returns:
        str: The name of the test folder.
    """
    experiment_configuration, _ = read_run_sections(run_folder=configuration_folder)
    return Path(experiment_configuration["test_data_folder"]).name


def read_test_labels(configuration_folder: Path) -> Dict[str, Any]:
    """
    Reads the labels.json of the test split an evaluation ran on, whose sequences give the frames of the sample images
    and the sequences without change of the trivial values.

    Args:
        configuration_folder (Path): The folder holding the training_configuration.yaml of the evaluation
            (find_evaluation_source).

    Returns:
        Dict[str, Any]: The labels file, with its 'categories' and 'sequences'.
    """
    experiment_configuration, _ = read_run_sections(run_folder=configuration_folder)
    return json.loads((Path(experiment_configuration["test_data_folder"]) / "labels.json").read_text(encoding="utf-8"))


def read_block_facts(block_lines: List[str], fact_prefix: str) -> Dict[str, Any]:
    """
    Reads the headline metrics and the counts of segments of an evaluation block of Results.txt as facts, the top-k
    accuracy also under a name independent of its k (<prefix>.top_k_accuracy and <prefix>.top_k), the reports having
    written a top-2 accuracy before the top-5 one.

    Args:
        block_lines (List[str]): The lines of the block (read_results_block).
        fact_prefix (str): The prefix of the fact names, e.g. "evaluation".

    Returns:
        Dict[str, Any]: The facts of the block.

    Raises:
        ValueError: If the block holds no note of its segments or no top-k accuracy.
    """
    # Headline metrics of the block, then the top-k accuracy and the note of the segments the report adds to them
    facts: Dict[str, Any] = dict(parse_block_metrics(block_lines=block_lines, fact_prefix=fact_prefix))
    top_k_names = [fact_name for fact_name in facts if TOP_K_FACT_PATTERN.search(string=fact_name) is not None]
    segment_match = SEGMENT_NOTE_PATTERN.search(string="\n".join(block_lines))
    if not top_k_names or segment_match is None:
        raise ValueError(f"The evaluation block starting with '{block_lines[0]}' holds no top-k accuracy or no note of "
                         f"its segments.")
    facts[f"{fact_prefix}.top_k_accuracy"] = facts[top_k_names[0]]
    facts[f"{fact_prefix}.top_k"] = int(TOP_K_FACT_PATTERN.search(string=top_k_names[0]).group(1))
    facts[f"{fact_prefix}.true_segments"] = int(segment_match.group(1))
    facts[f"{fact_prefix}.predicted_segments"] = int(segment_match.group(2))
    return facts


def find_table_rows(block_lines: List[str], table_title: str) -> Dict[str, Any]:
    """
    Finds a table of an evaluation block of Results.txt (Per Class Metrics or Per Class Recall): its title line, its
    three header lines (two header rows and the rule) and its class rows up to the first note line, so that a listing
    can keep its head and some of its rows.

    Args:
        block_lines (List[str]): The lines of the block.
        table_title (str): The title of the table, e.g. "Per Class Recall".

    Returns:
        Dict[str, Any]: The 'head' lines (title and header), the class 'rows' in the order of the report, the 'notes'
            lines after them, and the 'classes' named by every row.

    Raises:
        ValueError: If the block holds no such table.
    """
    title_indices = [line_index for line_index, line in enumerate(block_lines) if line.strip() == table_title]
    if not title_indices:
        raise ValueError(f"No '{table_title}' table in the evaluation block starting with '{block_lines[0]}'.")
    head_end = title_indices[0] + 4
    rows = []
    for line in block_lines[head_end:]:
        # Class rows are indented by three spaces; the notes by two, and a blank line ends the table
        if not line.startswith("   ") or not line.strip():
            break
        rows.append(line)
    notes_start = head_end + len(rows)
    notes = []
    for line in block_lines[notes_start:]:
        if not line.strip():
            break
        notes.append(line)
    return {"head": block_lines[title_indices[0]:head_end], "rows": rows, "notes": notes,
            "classes": [row.split()[0] for row in rows]}


def write_results_listing(block_lines: List[str], listing_path: Path) -> Dict[str, Any]:
    """
    Writes the listing of the report scene: the headline metrics and notes of the evaluation block, then the head of
    its Per Class Metrics table (worst F1 first), its first rows and the no_change row, and gives the names of the
    listed change classes as a pattern for the highlights of the scene.

    Args:
        block_lines (List[str]): The lines of the evaluation block of the last iteration.
        listing_path (Path): The text file of the listing.

    Returns:
        Dict[str, Any]: The number of lines of the listing and the listed classes, named evaluation.results_listing_*.
    """
    table = find_table_rows(block_lines=block_lines, table_title="Per Class Metrics")
    headline_end = block_lines.index(table["head"][0])
    headline_lines = block_lines[:headline_end]
    while headline_lines and not headline_lines[-1].strip():
        headline_lines.pop()

    # The worst rows of the table that are change classes, then the no_change row wherever it stands
    worst_rows = [row for row, class_name in zip(table["rows"], table["classes"])
                  if class_name != NO_CHANGE_CLASS_NAME][:LISTED_WORST_ROWS]
    no_change_row = table["rows"][table["classes"].index(NO_CHANGE_CLASS_NAME)]
    listed_lines = headline_lines + [""] + table["head"] + worst_rows + ["   ..."] + [no_change_row]
    write_listing(lines=[line[:LISTED_LINE_LENGTH] for line in listed_lines], listing_path=listing_path)
    listed_classes = [row.split()[0] for row in worst_rows]
    return {"evaluation.results_listing_lines": len(listed_lines),
            "evaluation.results_listing_classes": listed_classes,
            "evaluation.results_listing_pattern": "|".join(f"^   {class_name} " for class_name in listed_classes)}


def write_recall_listing(block_lines: List[str], listing_path: Path) -> Dict[str, Any]:
    """
    Writes the listing of the recall scene: the head of the Per Class Recall table of the evaluation block (the frame
    recall of every class with its interval over the test sequences, worst recall first), its worst rows, its best
    rows, the no_change row when it is not among them, and its notes, and gives the listed classes as patterns for the
    highlights of the scene.

    Args:
        block_lines (List[str]): The lines of the evaluation block of the last iteration.
        listing_path (Path): The text file of the listing.

    Returns:
        Dict[str, Any]: The number of lines of the listing and the listed classes, named evaluation.recall_listing_*.
    """
    table = find_table_rows(block_lines=block_lines, table_title="Per Class Recall")
    worst_rows = table["rows"][:LISTED_WORST_RECALL_ROWS]
    best_rows = table["rows"][-LISTED_BEST_RECALL_ROWS:]
    no_change_row = table["rows"][table["classes"].index(NO_CHANGE_CLASS_NAME)]

    # The no_change row joins the listing where it stands when neither the worst nor the best rows hold it
    middle_rows = ["   ..."] if no_change_row in worst_rows + best_rows else ["   ...", no_change_row, "   ..."]
    listed_lines = table["head"] + worst_rows + middle_rows + best_rows + [""] + table["notes"]
    write_listing(lines=[line[:LISTED_LINE_LENGTH] for line in listed_lines], listing_path=listing_path)
    worst_classes = [row.split()[0] for row in worst_rows]
    best_classes = [row.split()[0] for row in best_rows]
    return {"evaluation.recall_listing_lines": len(listed_lines),
            "evaluation.recall_listing_worst_pattern": "|".join(f"^   {name} " for name in worst_classes),
            "evaluation.recall_listing_best_pattern": "|".join(f"^   {name} " for name in best_classes),
            "evaluation.no_change_recall_rank": table["classes"].index(NO_CHANGE_CLASS_NAME) + 1,
            "evaluation.recall_table_classes": len(table["rows"])}


def count_sequences_without_change(test_labels: Dict[str, Any], fact_prefix: str) -> Dict[str, Any]:
    """
    Counts the test sequences of an evaluation whose frames are all no_change, the sequences a model giving no_change
    to every frame gets fully right, so that the sequence exact match of a degenerate solution can be read against it.

    Args:
        test_labels (Dict[str, Any]): The labels file of the test split of the evaluation (read_test_labels).
        fact_prefix (str): The prefix of the fact names, e.g. "evaluation".

    Returns:
        Dict[str, Any]: The facts, named <prefix>.sequences_without_change and <prefix>.test_sequences.
    """
    no_change_category = next(category["id"] for category in test_labels["categories"]
                              if category["name"] == NO_CHANGE_CLASS_NAME)
    return {f"{fact_prefix}.sequences_without_change": sum(all(category == no_change_category
                                                               for category in sequence["frame_category_ids"])
                                                           for sequence in test_labels["sequences"]),
            f"{fact_prefix}.test_sequences": len(test_labels["sequences"])}


def compute_rank_roc_auc(scores: numpy.ndarray, positives: numpy.ndarray) -> float:
    """
    Computes the area under the ROC curve of one class from the ranks of its scores (the Mann-Whitney statistic,
    ties sharing their mean rank), to check the macro ROC AUC of the report on a subset of the frames.

    Args:
        scores (numpy.ndarray): The probability of the class for every frame.
        positives (numpy.ndarray): Whether every frame is of the class.

    Returns:
        float: The ROC AUC.
    """
    ranks = rankdata(scores)
    positive_count = int(positives.sum())
    negative_count = len(positives) - positive_count
    return float((ranks[positives].sum() - positive_count * (positive_count + 1) / 2) /
                 (positive_count * negative_count))


def compute_macro_roc_auc(predictions: pandas.DataFrame, class_names: List[str],
                          frame_selection: numpy.ndarray) -> float:
    """
    Averages the ROC AUC of every class over the selected frames of predictions.csv, a class needing frames of its
    own and of the others among them, as the macro ROC AUC of the report does on all the frames.

    Args:
        predictions (pandas.DataFrame): The rows of predictions.csv.
        class_names (List[str]): The names of the classes, the probability columns being probability_<name>.
        frame_selection (numpy.ndarray): Whether every frame is kept.

    Returns:
        float: The macro ROC AUC of the kept frames.
    """
    true_classes = predictions["true_class"].to_numpy()[frame_selection]
    class_areas = []
    for class_name in class_names:
        positives = true_classes == class_name
        if 0 < positives.sum() < len(positives):
            class_areas.append(compute_rank_roc_auc(
                scores=predictions[f"probability_{class_name}"].to_numpy()[frame_selection], positives=positives))
    return float(numpy.mean(class_areas))


def draw_no_change_probabilities(predictions: pandas.DataFrame, frame_positions: numpy.ndarray, header: str,
                                 output_path: Path) -> Dict[str, Any]:
    """
    Draws the probability of no_change given by a run to every labelled test frame: its histogram on the first frames
    and on the later frames (left), and its mean at every frame position next to the true share of no_change at that
    position (right, the share in percent on the left axis, the probability with decimals on the right axis), and
    gives the boxes of the two panels.

    Args:
        predictions (pandas.DataFrame): The rows of predictions.csv.
        frame_positions (numpy.ndarray): The position of every labelled frame in its sequence.
        header (str): The header line of the figure, naming the file and the frames it shows.
        output_path (Path): The PNG file.

    Returns:
        Dict[str, Any]: The boxes of the panels and the curves of the right panel.
    """
    no_change_probabilities = predictions[f"probability_{NO_CHANGE_CLASS_NAME}"].to_numpy()
    is_no_change = (predictions["true_class"] == NO_CHANGE_CLASS_NAME).to_numpy()
    first_frames = frame_positions == 0
    register_fonts()
    figure, (histogram_axes, position_axes) = pyplot.subplots(nrows=1, ncols=2, figsize=(
        CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH,
        gridspec_kw={"width_ratios": [1.3, 1.0]})
    figure.patch.set_facecolor(SURFACE_COLOUR)
    # Margins leaving room for the probability axis on the right of the position panel
    figure.subplots_adjust(left=0.075, right=0.94, top=0.80, bottom=0.13, wspace=0.2)
    figure.text(x=0.012, y=0.955, s=header, color=TEXT_SECONDARY_COLOUR, fontsize=16 * POINTS_PER_PIXEL,
                va="center", family=MONOSPACE_FONT_FAMILY)

    # Histogram of the probability of no_change, first frames and later frames, on a logarithmic count axis
    bins = numpy.arange(0.0, 1.0 + PROBABILITY_BIN_WIDTH, PROBABILITY_BIN_WIDTH)
    histogram_axes.hist(x=no_change_probabilities[~first_frames], bins=bins, color=SERIES_COLOURS[0],
                        label=f"later frames ({int((~first_frames).sum()):,})")
    histogram_axes.hist(x=no_change_probabilities[first_frames], bins=bins, color=SERIES_COLOURS[2],
                        histtype="step", linewidth=3 * POINTS_PER_PIXEL,
                        label=f"first frames ({int(first_frames.sum()):,})")
    style_chart_panel(axes=histogram_axes, title="Probability of no_change given to every frame",
                      x_label="probability_no_change", y_label="Frames")
    histogram_axes.set_yscale(value="log")
    # Only the powers of ten labelled, the minor ticks of a narrow count range staying unlabelled
    histogram_axes.yaxis.set_minor_formatter(formatter=NullFormatter())
    histogram_axes.set_xlim(left=0.0, right=1.02)
    histogram_axes.legend(loc="best", frameon=False, labelcolor=TEXT_PRIMARY_COLOUR,
                          fontsize=18 * POINTS_PER_PIXEL)

    # Mean probability of no_change at every position against the true share of no_change there
    positions = numpy.arange(int(frame_positions.max()) + 1)
    mean_probabilities = [no_change_probabilities[frame_positions == position].mean() for position in positions]
    true_shares = [is_no_change[frame_positions == position].mean() for position in positions]
    position_axes.plot(positions, true_shares, color=SERIES_COLOURS[1], linewidth=3 * POINTS_PER_PIXEL,
                       linestyle=(0, (6, 5)), marker="o", markersize=7 * POINTS_PER_PIXEL,
                       label="true share of no_change")
    position_axes.plot(positions, mean_probabilities, color=SERIES_COLOURS[0], linewidth=3 * POINTS_PER_PIXEL,
                       marker="o", markersize=7 * POINTS_PER_PIXEL, label="mean probability_no_change")
    style_chart_panel(axes=position_axes, title="By frame position", x_label="Frame position",
                      y_label="Share of no_change")
    position_axes.set_ylim(bottom=0.0, top=1.08)
    position_axes.set_xticks(ticks=positions)
    # Share ticks in percent on the left, the same scale with the decimals of the probability on the right
    position_axes.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))
    position_axes.legend(loc="upper right", frameon=False, labelcolor=TEXT_PRIMARY_COLOUR,
                         fontsize=18 * POINTS_PER_PIXEL)
    _, position_panel_box = add_probability_axis(figure=figure, axes=position_axes, y_label="probability_no_change",
                                                 margin=0)
    save_figure(figure=figure, output_path=output_path)
    facts = {"histogram_panel_box": compute_pixel_box(figure=figure, axes=histogram_axes, with_labels=True, margin=0),
             "position_panel_box": position_panel_box,
             "mean_by_position": [round_fact(value=value) for value in mean_probabilities],
             "true_share_by_position": [round_fact(value=value) for value in true_shares]}
    pyplot.close(fig=figure)
    return facts


def compute_prediction_facts(predictions: pandas.DataFrame, frame_positions: numpy.ndarray) -> Dict[str, Any]:
    """
    Gives the facts of the predictions file of a run: the probabilities of no_change on the first and later frames,
    the frames predicted no_change, the predicted classes, the frame accuracy on the later frames, and the macro ROC
    AUC on all the frames and on the later frames only, the latter telling whether the scores of the change classes
    rank the frames by their content or only by their position.

    Args:
        predictions (pandas.DataFrame): The rows of predictions.csv.
        frame_positions (numpy.ndarray): The position of every labelled frame in its sequence.

    Returns:
        Dict[str, Any]: The facts, without prefix.
    """
    first_frames = frame_positions == 0
    no_change_probabilities = predictions[f"probability_{NO_CHANGE_CLASS_NAME}"].to_numpy()
    class_names = [column[len("probability_"):] for column in predictions.columns if column.startswith("probability_")]
    true_classes = predictions["true_class"].to_numpy()
    predicted_classes = predictions["predicted_class"].to_numpy()
    later_no_change = true_classes[~first_frames] == NO_CHANGE_CLASS_NAME

    # Probability of no_change on the first and on the later frames, the later ones split by their true class
    return {"frame_count": len(predictions), "first_frame_count": int(first_frames.sum()),
            "later_frame_count": int((~first_frames).sum()),
            "first_minimum": round_fact(value=no_change_probabilities[first_frames].min()),
            "first_mean": round_fact(value=no_change_probabilities[first_frames].mean()),
            "later_mean": round_fact(value=no_change_probabilities[~first_frames].mean()),
            "later_minimum": round_fact(value=no_change_probabilities[~first_frames].min()),
            "later_maximum": round_fact(value=no_change_probabilities[~first_frames].max()),
            "later_true_no_change_share": round_fact(value=later_no_change.mean()),
            "later_no_change_frames_mean": round_fact(
                value=no_change_probabilities[~first_frames][later_no_change].mean()),
            "later_change_frames_mean": round_fact(
                value=no_change_probabilities[~first_frames][~later_no_change].mean()),
            "later_true_no_change_count": int(later_no_change.sum()),
            # Frames predicted no_change, the classes predicted at all and the accuracies of the first and later frames
            "predicted_no_change_frames": int((predicted_classes == NO_CHANGE_CLASS_NAME).sum()),
            "predicted_no_change_share": round_fact(value=(predicted_classes == NO_CHANGE_CLASS_NAME).mean()),
            "predicted_class_count": int(len(numpy.unique(predicted_classes))),
            "later_frame_accuracy": round_fact(value=(true_classes == predicted_classes)[~first_frames].mean()),
            "first_frame_accuracy": round_fact(value=(true_classes == predicted_classes)[first_frames].mean()),
            "later_no_change_recall": round_fact(value=(predicted_classes[~first_frames][later_no_change] ==
                                                        NO_CHANGE_CLASS_NAME).mean()),
            # Macro ROC AUC on every frame, then on the later frames only, where the position no longer gives the class
            "macro_roc_auc_all_frames": round_fact(value=compute_macro_roc_auc(
                predictions=predictions, class_names=class_names,
                frame_selection=numpy.ones(shape=len(predictions), dtype=bool))),
            "macro_roc_auc_later_frames": round_fact(value=compute_macro_roc_auc(
                predictions=predictions, class_names=class_names, frame_selection=~first_frames))}


def make_prediction_facts(run_folder: Path, source: Dict[str, Any], output_folder: Path,
                          force: bool) -> Dict[str, Any]:
    """
    Reads predictions.csv of the evaluation of the run (a row of probabilities per labelled test frame), draws the
    probability of no_change and gives the facts of the predictions. Reused from the previous build unless forced or
    unless that build read another run, another evaluation or a predictions file replaced since.

    Args:
        run_folder (Path): The folder of the run.
        source (Dict[str, Any]): The evaluation read (find_evaluation_source).
        output_folder (Path): The figures folder of the series.
        force (bool): Whether the reused figure is made again.

    Returns:
        Dict[str, Any]: The facts, named predictions.*.
    """
    figure_path = output_folder / "evaluation" / "no_change_probabilities.png"
    reused_facts_path = output_folder / "evaluation" / "no_change_probabilities_facts.json"
    source_name = str(source["files_folder"].relative_to(run_folder))
    predictions_path = source["files_folder"] / "predictions.csv"
    reused_facts = read_run_reused_facts(facts_path=reused_facts_path, run_folder=run_folder, source_name=source_name,
                                         source_paths=[predictions_path], fact_prefix="predictions", force=force,
                                         committed_mode=is_snapshot_folder(folder=run_folder))
    if reused_facts is not None:
        return reused_facts
    predictions = pandas.read_csv(filepath_or_buffer=predictions_path)
    frame_positions = predictions["frame"].str.extract(pat=r"/(\d+)\.png$")[0].astype(int).to_numpy()
    header = (f"{source_name}/predictions.csv · {len(predictions):,} labelled frames of "
              f"{read_test_split_name(configuration_folder=source['configuration_folder'])}")
    facts = draw_no_change_probabilities(predictions=predictions, frame_positions=frame_positions, header=header,
                                         output_path=figure_path)
    facts.update(compute_prediction_facts(predictions=predictions, frame_positions=frame_positions))

    # Facts under their prefix, with the run, the files they were read from and their fingerprint for the next reuse
    prefixed_facts = {f"predictions.{fact_name}": value for fact_name, value in facts.items()}
    prefixed_facts.update(describe_run_source(run_folder=run_folder, source_name=source_name,
                                              source_paths=[predictions_path], fact_prefix="predictions"))
    write_reused_facts(facts_path=reused_facts_path, figure_paths=[figure_path], facts=prefixed_facts)
    return prefixed_facts


def build_sample_captions(true_classes: List[str], predicted_classes: List[str]) -> List[str]:
    """
    Writes the caption of every frame of a sample image as the sequence labelling trainer does
    (build_frame_prediction_captions, numbered by build_frame_grid): the frame number, an X before a wrong frame, then
    the true and predicted classes.

    Args:
        true_classes (List[str]): The true class of every frame, in order.
        predicted_classes (List[str]): The predicted class of every frame, in order.

    Returns:
        List[str]: The caption of every frame.
    """
    return [f"{frame_index} " + ("X " if true_class != predicted_class else "") + f"{true_class}/{predicted_class}"
            for frame_index, (true_class, predicted_class) in enumerate(zip(true_classes, predicted_classes))]


def compute_sample_boxes(image_size: List[int], captions: List[str], header: str) -> Dict[str, Any]:
    """
    Computes the pixel boxes of a sample image of the evaluation from the layout of its frame grid
    (vision_training/utilities/visualization.py build_frame_grid and add_caption_bar): the caption bar of the image,
    the caption of every frame, every frame, and the first row, for the views and highlights of its scene.

    Args:
        image_size (List[int]): The width and height of the sample image.
        captions (List[str]): The caption of every frame (build_sample_captions).
        header (str): The caption of the image, written by the trainer above the grid.

    Returns:
        Dict[str, Any]: The boxes, named header_box (around the text of the caption of the image), caption_boxes,
            frame_boxes, first_row_box and image_box.
    """
    row_count = (len(captions) + FRAME_GRID_COLUMNS - 1) // FRAME_GRID_COLUMNS
    frame_size = (image_size[1] - CAPTION_BAR_HEIGHT - FRAME_GRID_GAP) // row_count - FRAME_GRID_GAP
    caption_boxes, frame_boxes = [], []

    # Every frame in its cell, row by row, its caption bar at its top left corner
    for frame_index, caption in enumerate(captions):
        left = FRAME_GRID_GAP + (frame_index % FRAME_GRID_COLUMNS) * (frame_size + FRAME_GRID_GAP)
        top = CAPTION_BAR_HEIGHT + FRAME_GRID_GAP + (frame_index // FRAME_GRID_COLUMNS) * (frame_size + FRAME_GRID_GAP)
        caption_width = min(frame_size - 1, FRAME_CAPTION_BASE_WIDTH + FRAME_CAPTION_CHARACTER_WIDTH * len(caption))
        caption_boxes.append([left, top, left + caption_width, top + FRAME_CAPTION_HEIGHT])
        frame_boxes.append([left, top, left + frame_size, top + frame_size])
    first_row_width = min(len(captions), FRAME_GRID_COLUMNS) * (frame_size + FRAME_GRID_GAP) + FRAME_GRID_GAP
    (header_width, _), _ = cv2.getTextSize(text=header, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=SAMPLE_CAPTION_FONT_SCALE, thickness=1)
    return {"header_box": [0, 0, header_width + SAMPLE_CAPTION_MARGIN, CAPTION_BAR_HEIGHT],
            "caption_boxes": caption_boxes,
            "frame_boxes": frame_boxes, "first_row_box": [0, 0, first_row_width,
                                                          CAPTION_BAR_HEIGHT + frame_size + 2 * FRAME_GRID_GAP],
            "image_box": [0, 0, image_size[0], image_size[1]]}


def make_sample_facts(source: Dict[str, Any], test_labels: Dict[str, Any], output_folder: Path) -> Dict[str, Any]:
    """
    Chooses the sample image of the evaluation scene among the fixed test sequences of the last evaluation
    (Samples/ of the evaluation read): the one with the most change frames predicted right, then the most change
    frames, then the first name, so that the scene shows what the run finds and misses. Copies it into the figures
    folder and gives the classes of its frames, read from predictions.csv, and the boxes of its captions and frames.

    Args:
        source (Dict[str, Any]): The evaluation read (find_evaluation_source).
        test_labels (Dict[str, Any]): The labels file of the test split of the evaluation (read_test_labels).
        output_folder (Path): The figures folder of the series.

    Returns:
        Dict[str, Any]: The facts, named sample.*.

    Raises:
        FileNotFoundError: If the evaluation holds no sample image.
        ValueError: If the rows of a sample sequence in predictions.csv are not the frames of that sequence in the
            test labels file, in their order, the captions being numbered over those rows and the frame grid of the
            image over every frame.
    """
    sample_paths = sorted((source["files_folder"] / "Samples").glob("*.png"))
    if not sample_paths:
        raise FileNotFoundError(f"No sample image in {source['files_folder']}/Samples.")
    # The snapshots keep the columns of predictions.csv read here, compressed (77 megabytes for the whole file)
    predictions_name = "predictions.csv.gz" if is_snapshot_folder(folder=source["files_folder"]) else "predictions.csv"
    predictions = pandas.read_csv(filepath_or_buffer=source["files_folder"] / predictions_name)
    sample_rows = predictions[predictions["sequence"].isin([path.stem for path in sample_paths])]
    sequence_frames = {sequence["name"]: sequence["frames"] for sequence in test_labels["sequences"]}

    # Change frames of every sample and those predicted right, the first frames being always no_change
    candidates = []
    for sample_path in sample_paths:
        rows = sample_rows[sample_rows["sequence"] == sample_path.stem].sort_values(by="frame")
        if rows["frame"].tolist() != sequence_frames.get(sample_path.stem):
            raise ValueError(f"The {len(rows)} rows of the sample {sample_path.name} in predictions.csv are not the "
                             f"frames of its sequence in the test labels file, in their order.")
        is_change = (rows["true_class"] != NO_CHANGE_CLASS_NAME).to_numpy()
        is_right = (rows["true_class"] == rows["predicted_class"]).to_numpy()
        right_changes, changes = int((is_change & is_right).sum()), int(is_change.sum())
        candidates.append({"path": sample_path, "rows": rows, "right_changes": right_changes, "changes": changes,
                           "order": (-right_changes, -changes, sample_path.name)})
    chosen = min(candidates, key=operator.itemgetter("order"))

    # The chosen image copied for the scene, its captions written as the trainer writes them
    figure_path = output_folder / "evaluation" / "sample_sequence.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src=chosen["path"], dst=figure_path)
    true_classes = chosen["rows"]["true_class"].tolist()
    predicted_classes = chosen["rows"]["predicted_class"].tolist()
    captions = build_sample_captions(true_classes=true_classes, predicted_classes=predicted_classes)

    # Frames predicted wrong and the caption bar the trainer writes above the grid, which place the boxes of the scene
    with Image.open(fp=figure_path) as sample_image:
        image_size = list(sample_image.size)
    wrong_frames = [index for index, (true_class, predicted_class) in enumerate(zip(true_classes, predicted_classes))
                    if true_class != predicted_class]
    header = f"{len(true_classes) - len(wrong_frames)}/{len(true_classes)} labelled frames right (true/predicted)"
    boxes = compute_sample_boxes(image_size=image_size, captions=captions, header=header)
    facts = {"sample.name": chosen["path"].stem, "sample.sample_count": len(sample_paths),
             "sample.frame_count": len(true_classes), "sample.right_frames": len(true_classes) - len(wrong_frames),
             "sample.wrong_frames": wrong_frames, "sample.change_frames": chosen["changes"],
             "sample.right_change_frames": chosen["right_changes"],
             "sample.no_change_frames": len(true_classes) - chosen["changes"],
             "sample.true_classes": true_classes, "sample.predicted_classes": predicted_classes,
             "sample.captions": captions, "sample.image_size": image_size}
    facts.update({f"sample.{box_name}": box for box_name, box in boxes.items()})
    return facts


def compute_segment_facts(configuration_folder: Path) -> Dict[str, Any]:
    """
    Computes with compute_segment_metrics of the repository the segment metrics of the example of the tasks reference
    (its frame accuracy, edit score and segment F1 at every overlap), and reads the test frames of no_change of the
    evaluation, the accuracy of a model giving no_change to every frame.

    Args:
        configuration_folder (Path): The folder holding the data.json of the evaluation (find_evaluation_source).

    Returns:
        Dict[str, Any]: The facts, named segments.*.
    """
    true_classes, predicted_classes = numpy.array(README_TRUE_CLASSES), numpy.array(README_PREDICTED_CLASSES)
    metrics = compute_segment_metrics(sequences=[{"frame_indices": numpy.arange(len(true_classes)),
                                                  "true_classes": true_classes,
                                                  "predicted_classes": predicted_classes}])
    facts = {"segments.readme_frame_accuracy": round_fact(value=numpy.mean(true_classes == predicted_classes)),
             "segments.readme_edit_score": round_fact(value=metrics["edit_score"]),
             "segments.readme_true_segments": metrics["number_true_segments"],
             "segments.readme_predicted_segments": metrics["number_predicted_segments"]}
    for overlap_threshold, segment_f1 in metrics["segment_f1"].items():
        facts[f"segments.readme_f1_at_{round(overlap_threshold * 100)}"] = round_fact(value=segment_f1)
    facts.update(read_test_frame_facts(configuration_folder=configuration_folder, fact_prefix="segments"))
    return facts


def read_test_frame_facts(configuration_folder: Path, fact_prefix: str) -> Dict[str, Any]:
    """
    Reads the test frames of no_change and of every class from the record of the data of an evaluation (data.json),
    whose share is the frame accuracy of a model giving no_change to every frame of its test split.

    Args:
        configuration_folder (Path): The folder holding the data.json of the evaluation (find_evaluation_source).
        fact_prefix (str): The prefix of the fact names, e.g. "segments".

    Returns:
        Dict[str, Any]: The facts, named <prefix>.no_change_test_frames, <prefix>.test_frames and
            <prefix>.no_change_test_share.
    """
    data_record = json.loads((configuration_folder / "data.json").read_text(encoding="utf-8"))
    no_change_record = next(record for record in data_record["classes"] if record["class_name"] == NO_CHANGE_CLASS_NAME)
    test_frames = sum(record["frames"]["test"] for record in data_record["classes"])
    return {f"{fact_prefix}.no_change_test_frames": no_change_record["frames"]["test"],
            f"{fact_prefix}.test_frames": test_frames,
            f"{fact_prefix}.no_change_test_share": round_fact(value=no_change_record["frames"]["test"] / test_frames)}


def compute_model_facts(run_folder: Path, iteration: int) -> Dict[str, Any]:
    """
    Rebuilds the trained model of the last checkpoint on the CPU, counts the parameters of its components, and runs
    the first test sequences of the run through it, padded into one batch, to give the shape of every stage: the
    backbone feature maps of the real frames, the pooled frame vectors, the frame vectors in their sequences, the
    output of the temporal module and the class logits, with the channels and side of the last feature map, the
    features a flatten pooling would give, and the backbone blocks holding a residual shortcut.

    Args:
        run_folder (Path): The folder of the sequence labelling run.
        iteration (int): The iteration of the checkpoint.

    Returns:
        Dict[str, Any]: The facts, named model.*.
    """
    model = load_sequence_model(run_folder=run_folder, iteration=iteration)
    dataset = build_test_dataset(run_folder=run_folder, label_type="frame")
    batch = collate_named_sequences(dataset=dataset, sequence_names=[sequence["sequence_name"]
                                                                     for sequence in dataset.sequences[:4]])
    images = batch["images"].float()
    with torch.no_grad():
        backbone_maps, _ = model.backbone(input_tensor=images[batch["sequence_mask"]])
        pooled_vectors = model.global_pooling(input=backbone_maps)
        frame_features = encode_frames(backbone=model.backbone, global_pooling=model.global_pooling,
                                       input_tensor=images, sequence_mask=batch["sequence_mask"])
        temporal_features = model.temporal_module(frame_features=frame_features, sequence_mask=batch["sequence_mask"])
    outputs = predict_sequences(model=model, batch=batch)
    parameters = count_model_parameters(model=model)
    padding_zero_vectors = (frame_features.abs().sum(dim=2) == 0).sum(dim=1).tolist()

    # Blocks of the backbone closing at least one residual pair, with the condition of ConvolutionBlock.forward: a
    # block whose first layer changes the channel count shifts its pairs by one layer and closes none with two layers
    residual_blocks = [block_index + 1 for block_index, block in enumerate(model.backbone.convolution_blocks)
                       if block.residual and any(layer_index % 2 == 1 - block.residual_parity and layer_index != 0
                                                 for layer_index in range(len(block.layer_module_indices)))]
    return {"model.backbone_channels": int(backbone_maps.shape[1]),
            "model.backbone_map_size": int(backbone_maps.shape[2]),
            "model.flatten_features": int(backbone_maps[0].numel()),
            "model.temporal_features": int(temporal_features.shape[2]),
            "model.residual_blocks": residual_blocks, "model.residual_block_count": len(residual_blocks),
            # Parameters of every component, then the shape of every stage of the padded batch
            "model.backbone_parameters": parameters["backbone"],
            "model.temporal_parameters": parameters["temporal_module"],
            "model.head_parameters": parameters["classification_head"], "model.total_parameters": parameters["total"],
            "model.total_megabytes": round_fact(value=parameters["total"] * 4 / 1e6),
            "model.input_shape": str(list(batch["images"].shape)),
            "model.backbone_shape": str(list(backbone_maps.shape)),
            "model.pooled_shape": str(list(pooled_vectors.shape)),
            "model.frame_features_shape": str(list(frame_features.shape)),
            "model.temporal_shape": str(list(temporal_features.shape)),
            "model.logits_shape": str(list(outputs["class_logits"].shape)),
            "model.padding_zero_vectors": padding_zero_vectors, "model.head": str(model.classification_head),
            "model.head_layers": describe_head_layers(head=model.classification_head)}


def describe_head_layers(head: torch.nn.Module) -> str:
    """
    Writes the layers of the classification head of the trained model in their order of execution, one line per
    hidden layer starting at its dropout ("Dropout · Linear(192, 96) · LayerNorm · ReLU"), the last line ending with the
    classifier, so that the diagram of the model shows the head the run built rather than a typed one.

    Args:
        head (torch.nn.Module): The classification head of the model.

    Returns:
        str: The lines of the layers, joined by line breaks.
    """
    lines: List[List[str]] = []
    for module in head.modules():
        # Only the layers themselves, not the containers holding them
        if list(module.children()):
            continue
        name = type(module).__name__
        if isinstance(module, torch.nn.Linear):
            name = f"Linear({module.in_features}, {module.out_features})"
        if isinstance(module, torch.nn.Dropout) or not lines:
            lines.append([])
        lines[-1].append(name)
    return "\n".join(" · ".join(line) for line in lines)


def check_low_recall_classes(class_names: List[str]) -> None:
    """
    Checks the claim of the chapter on reading an evaluation that every change class found on fewer than half of its
    frames shows a shape the experiment plan reports as not learned by the classifier the backbone was pre-trained on,
    so that a run contradicting it stops the build instead of leaving a false sentence in the video.

    Args:
        class_names (List[str]): The change classes under the low recall threshold, e.g. ["appearing_hollow_kite"].

    Raises:
        ValueError: If one of them shows a shape the experiment plan does not report as unlearned.
    """
    for class_name in class_names:
        # appearing_hollow_kite and disappearing_kite both show the kite
        shape_name = class_name.split(sep="_", maxsplit=1)[1].removeprefix("hollow_")
        if shape_name not in UNLEARNED_PRETRAINING_SHAPES:
            raise ValueError(f"The change class {class_name} is found on fewer than {LOW_RECALL_THRESHOLD:.0%} of its "
                             f"frames, but its shape is not among {UNLEARNED_PRETRAINING_SHAPES}, the shapes the "
                             f"experiment plan reports as not learned by the pre-training classifier: reword the "
                             f"recall scene of the chapter on reading an evaluation.")


def read_class_metric_facts(files_folder: Path) -> Dict[str, Any]:
    """
    Reads the per-class metrics of an evaluation (class_metrics.csv): the precision, recall and recall interval of
    no_change, the recall of the change classes (lowest, median, highest, mean, and how many are never found), and the
    mean recall over every class with test frames, which is the balanced accuracy of the report.

    Args:
        files_folder (Path): The folder of the files of the evaluation read (find_evaluation_source).

    Returns:
        Dict[str, Any]: The facts, named evaluation.*.
    """
    class_metrics = pandas.read_csv(filepath_or_buffer=files_folder / "class_metrics.csv")
    class_metrics = class_metrics[class_metrics["samples"] > 0]
    is_no_change = (class_metrics["class"] == NO_CHANGE_CLASS_NAME).to_numpy()
    no_change_metrics = class_metrics[is_no_change].iloc[0]
    change_recalls = class_metrics["recall"][~is_no_change].to_numpy()
    facts = {"evaluation.no_change_precision": round_fact(value=float(no_change_metrics["precision"])),
             "evaluation.no_change_recall": round_fact(value=float(no_change_metrics["recall"])),
             "evaluation.change_class_minimum_recall": round_fact(value=float(change_recalls.min())),
             "evaluation.change_class_median_recall": round_fact(value=float(numpy.median(change_recalls))),
             "evaluation.change_class_maximum_recall": round_fact(value=float(change_recalls.max())),
             "evaluation.change_class_mean_recall": round_fact(value=float(change_recalls.mean())),
             "evaluation.change_classes_never_found": int((change_recalls == 0).sum()),
             "evaluation.change_classes_with_frames": int(len(change_recalls)),
             "evaluation.low_recall_threshold": LOW_RECALL_THRESHOLD,
             "evaluation.change_classes_below_threshold": int((change_recalls < LOW_RECALL_THRESHOLD).sum()),
             "evaluation.change_classes_below_threshold_names": class_metrics["class"][~is_no_change][
                 change_recalls < LOW_RECALL_THRESHOLD].tolist(),
             "evaluation.mean_class_recall": round_fact(value=float(class_metrics["recall"].mean()))}

    # The recall interval over the test sequences, written by the evaluations since the per-class recall intervals
    if "recall_lower" in class_metrics.columns:
        facts.update({"evaluation.no_change_recall_lower": round_fact(value=float(no_change_metrics["recall_lower"])),
                      "evaluation.no_change_recall_upper": round_fact(value=float(no_change_metrics["recall_upper"]))})
    return facts


def read_threshold_facts(files_folder: Path) -> Dict[str, Any]:
    """
    Reads the confidence thresholds of an evaluation (class_thresholds.json): the target precision, the classes that
    reach it and those that never do (a null threshold), and the threshold and recall of no_change when it reaches it.

    Args:
        files_folder (Path): The folder of the files of the evaluation read (find_evaluation_source).

    Returns:
        Dict[str, Any]: The facts, named thresholds.*.
    """
    thresholds = json.loads((files_folder / "class_thresholds.json").read_text(encoding="utf-8"))
    class_thresholds = thresholds["classes"]
    facts = {"thresholds.target_precision": thresholds["target_precision"],
             "thresholds.null_classes": sum(class_threshold is None for class_threshold in class_thresholds.values()),
             "thresholds.reached_classes": sum(class_threshold is not None
                                               for class_threshold in class_thresholds.values()),
             "thresholds.no_change_reached": class_thresholds[NO_CHANGE_CLASS_NAME] is not None}
    no_change_threshold = class_thresholds[NO_CHANGE_CLASS_NAME]
    if no_change_threshold is not None:
        facts.update({"thresholds.no_change_threshold": round_fact(value=no_change_threshold["threshold"]),
                      "thresholds.no_change_recall": round_fact(value=no_change_threshold["recall"])})
    return facts


def compute_trivial_facts(facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes the value every headline metric of the report takes for a model predicting no_change on every test frame
    of the run, the degenerate solution a sequence labelling run can fall into: the accuracy is the share of no_change
    frames, the balanced accuracy one over the classes with test frames (no_change found on all its frames, the others
    on none), the macro F1 the F1 of no_change (its precision being that share and its recall one) over the same
    classes, one predicted segment per sequence, and an exact match on the sequences without any change.

    Args:
        facts (Dict[str, Any]): The facts of the run, holding segments.no_change_test_share,
            evaluation.change_classes_with_frames, evaluation.sequences_without_change and evaluation.test_sequences.

    Returns:
        Dict[str, Any]: The facts, named trivial.*.
    """
    no_change_share = facts["segments.no_change_test_share"]
    class_count = facts["evaluation.change_classes_with_frames"] + 1
    return {"trivial.accuracy": round_fact(value=no_change_share),
            "trivial.balanced_accuracy": round_fact(value=1.0 / class_count),
            "trivial.macro_f1": round_fact(value=2.0 * no_change_share / (1.0 + no_change_share) / class_count),
            "trivial.no_change_f1": round_fact(value=2.0 * no_change_share / (1.0 + no_change_share)),
            "trivial.class_count": class_count,
            "trivial.predicted_segments": facts["evaluation.test_sequences"],
            "trivial.sequence_exact_match": round_fact(value=facts["evaluation.sequences_without_change"] /
                                                       facts["evaluation.test_sequences"])}


def make_evaluation_facts(output_folder: Path, run_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Makes the evaluation listings and figures of the sequence labelling series and gives their facts, with those of
    the thresholds file, the per-class metrics, the sample image, the segment metrics, the trivial values of every
    metric, the benchmark figure and the model, the model and prediction facts being reused from the previous build
    unless forced, unless that build read another run or evaluation, or unless it wrote fewer facts than the storyboards
    quote. A run read from the committed snapshots reuses the committed model and prediction facts.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the sequence labelling run of the series.
        force (bool): Whether the reused figures are made again.

    Returns:
        Dict[str, Any]: The facts, named evaluation.*, predictions.*, thresholds.*, sample.*, segments.*, trivial.*,
            plan.* and model.*.

    Raises:
        ValueError: If a change class found on fewer than half of its frames shows a shape the experiment plan does
            not report as unlearned by the pre-training classifier (check_low_recall_classes).
    """
    # A run read from the snapshots, without its checkpoint and its whole predictions file, takes the committed model
    # and prediction facts
    snapshot_mode = is_snapshot_folder(folder=run_folder)
    if snapshot_mode:
        restore_committed_figures(output_folder=output_folder, family_folder="evaluation")
    source = find_evaluation_source(run_folder=run_folder)
    block_lines = read_results_block(results_path=source["results_path"], iteration=source["iteration"],
                                     averaged_weights=source["averaged"])
    facts = read_block_facts(block_lines=block_lines, fact_prefix="evaluation")
    facts.update(write_results_listing(block_lines=block_lines,
                                       listing_path=output_folder / "listings" / "results_block.txt"))
    facts.update(write_recall_listing(block_lines=block_lines,
                                      listing_path=output_folder / "listings" / "recall_table.txt"))
    facts.update(make_prediction_facts(run_folder=run_folder, source=source, output_folder=output_folder, force=force))

    # Thresholds, per-class metrics, sample image, segment metrics and the sequences without change of the test split
    # of the evaluation
    test_labels = read_test_labels(configuration_folder=source["configuration_folder"])
    facts.update(read_threshold_facts(files_folder=source["files_folder"]))
    facts.update(read_class_metric_facts(files_folder=source["files_folder"]))
    check_low_recall_classes(class_names=facts["evaluation.change_classes_below_threshold_names"])
    facts.update(make_sample_facts(source=source, test_labels=test_labels, output_folder=output_folder))
    facts.update(compute_segment_facts(configuration_folder=source["configuration_folder"]))
    facts.update(count_sequences_without_change(test_labels=test_labels, fact_prefix="evaluation"))
    facts.update({"evaluation.iteration": source["iteration"], "evaluation.averaged_weights": source["averaged"],
                  "evaluation.source": str(source["files_folder"].relative_to(run_folder)),
                  "evaluation.reevaluated": source["evaluation_folder"].name == CHECKPOINT_EVALUATION_FOLDER_NAME,
                  "evaluation.test_split": read_test_split_name(configuration_folder=source["configuration_folder"])})

    # Values of a model predicting no_change everywhere, and the benchmark launches read against them
    facts.update(compute_trivial_facts(facts=facts))
    facts.update(make_plan_figures(output_folder=output_folder, run_folder=run_folder,
                                   trivial_macro_f1=facts["trivial.macro_f1"], run_iterations=source["iteration"],
                                   run_macro_f1=facts["evaluation.macro_f1"], averaged_weights=source["averaged"],
                                   test_split=facts["evaluation.test_split"]))

    # Shapes and parameters of the trained model, run on the CPU, reused from the previous build unless forced or
    # unless its checkpoint file changed since, the committed facts for a run read from the snapshots
    model_facts_path = output_folder / "evaluation" / "model_facts.json"
    checkpoint_name = f"Weights/Iteration_{source['iteration']}"
    checkpoint_paths = [] if snapshot_mode else [get_checkpoint_path(experiment_folder=run_folder,
                                                                     iteration=source["iteration"])]
    model_facts = read_run_reused_facts(facts_path=model_facts_path, run_folder=run_folder, source_name=checkpoint_name,
                                        source_paths=checkpoint_paths, fact_prefix="model", force=force,
                                        committed_mode=snapshot_mode, required_fact_names=REQUIRED_MODEL_FACTS)
    if model_facts is None:
        model_facts = compute_model_facts(run_folder=run_folder, iteration=source["iteration"])
        model_facts.update(describe_run_source(run_folder=run_folder, source_name=checkpoint_name,
                                               source_paths=checkpoint_paths, fact_prefix="model"))
        write_reused_facts(facts_path=model_facts_path, figure_paths=[], facts=model_facts)
    facts.update(model_facts)
    return facts
