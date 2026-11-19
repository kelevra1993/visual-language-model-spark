"""
Prediction confidence figure of the classification tutorials (the chapter judging and shipping a classifier), made at
every build by tutorials/figures/classification/figures.py from the predictions.csv written by the evaluation of the
last checkpoint of the classification run (save_predictions): on the left, the distribution of the probability of the
predicted class over the right and the wrong predictions of the test images, which tells whether a threshold on that
probability can separate them; on the right, for every test image of the true class of the largest confusion of that
checkpoint, its probability of that class against its probability of the class it is most often taken for, which shows
a confusion between two classes taking the probability mass from each other. The boxes of the two panels and the
numbers the narration quotes are the facts of the scene, the pair named by its role. The shares of the images of a
group are written as the percentages of the reports of the repository (ticks "20 %"), the probabilities keeping their
decimals. A build from the run snapshots reads the reduced copy of the predictions file
(snapshot_reductions.reduce_predictions_file), which gives the same facts.
"""

import csv
import json

import numpy

from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.classification.evaluation_tables import (LAST_ITERATION, find_largest_confusions,
                                                                format_class_name, read_confusion_matrix)
from tutorials.figures.classification.snapshot_helpers import open_predictions_file
from tutorials.figures.classification.snapshot_reductions import (PREDICTED_PROBABILITY_COLUMN,
                                                                  PROBABILITY_COLUMN_PREFIX)
from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, POINTS_PER_PIXEL,
                                                SERIES_COLOURS, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR,
                                                compute_pixel_box, register_fonts, round_fact, save_figure,
                                                style_chart_panel)
from tutorial_builder.layout import NAMED_COLOURS
from vision_training.evaluation.report_formatting import format_percentage_tick

# Evaluated checkpoint whose predictions are drawn, the last one of the run
PREDICTIONS_ITERATION = LAST_ITERATION

# Number of bins of the probability of the predicted class, which goes from one over the number of classes to 1
PROBABILITY_BIN_COUNT = 45

# Probability above which a prediction counts as confident in the facts
CONFIDENT_PROBABILITY = 0.9

# Sum of the two probabilities above which an image counts as split between the two classes only
PAIR_MASS_THRESHOLD = 0.95

# Colours of the right and wrong predictions: aqua and red of the palette of the videos
CORRECT_COLOUR = SERIES_COLOURS[1]
WRONG_COLOUR = "#%02x%02x%02x" % NAMED_COLOURS["red"]

# Name of the figure file
FIGURE_NAME = f"prediction_confidence_iteration_{PREDICTIONS_ITERATION}.png"


def read_predictions(predictions_path: Path, pair_classes: List[str]) -> Dict[str, Any]:
    """
    Reads the predictions file of an evaluation: the true and predicted class of every test image, whether it is right,
    the probability of its predicted class and its probabilities of the two classes of the pair; from the original CSV
    (a probability column per class) or, in a run read from the snapshots, from its reduced gzipped copy, which keeps
    the probability of the predicted class in its own column and the probability columns of the pair, the same texts.

    Args:
        predictions_path (Path): The predictions.csv file.
        pair_classes (List[str]): The true and the confused class of the pair.

    Returns:
        Dict[str, Any]: The 'true_classes' and 'predicted_classes' lists, the 'correct' [N] booleans, the [N]
            'predicted_probabilities' and the [N, 2] 'pair_probabilities'.

    Raises:
        FileNotFoundError: If the file is missing (save_predictions was off).
        KeyError: If the file lacks the probability column of a class of the pair.
    """
    with open_predictions_file(predictions_path=predictions_path, newline="") as predictions_file:
        reader = csv.DictReader(f=predictions_file)
        pair_columns = [PROBABILITY_COLUMN_PREFIX + class_name for class_name in pair_classes]
        missing_columns = [column for column in pair_columns if column not in reader.fieldnames]
        if missing_columns:
            raise KeyError(f"The predictions file {predictions_path} has no columns {missing_columns}: a reduced copy "
                           f"must keep them (kept_classes of tutorials/figures/classification/snapshot_files.yaml).")
        rows = list(reader)

    # Probability of the predicted class: its own column in the reduced copy, the column of that class otherwise
    reduced = PREDICTED_PROBABILITY_COLUMN in rows[0]
    return {"true_classes": [row["true_class"] for row in rows],
            "predicted_classes": [row["predicted_class"] for row in rows],
            "correct": numpy.array([row["correct"] == "1" for row in rows]),
            "predicted_probabilities": numpy.array([float(row[PREDICTED_PROBABILITY_COLUMN] if reduced else
                                                          row[PROBABILITY_COLUMN_PREFIX + row["predicted_class"]])
                                                    for row in rows]),
            "pair_probabilities": numpy.array([[float(row[column]) for column in pair_columns] for row in rows])}


def draw_confidence_panel(axes: Any, predicted_probabilities: numpy.ndarray, correct: numpy.ndarray,
                          class_count: int) -> None:
    """
    Draws the left panel: the distribution of the probability of the predicted class over the right predictions and
    over the wrong ones, each as the share of its own images per bin, so that both are visible whatever their counts,
    from one over the number of classes, the lowest probability a predicted class can have, to 1.

    Args:
        axes (Any): The matplotlib panel.
        predicted_probabilities (numpy.ndarray): The [N] probabilities of the predicted classes.
        correct (numpy.ndarray): The [N] booleans of the right predictions.
        class_count (int): The number of classes of the run.
    """
    style_chart_panel(axes=axes, title="Probability of the predicted class", x_label="Probability of the predicted class",
                      y_label="Share of the images of the group")
    probability_bins = numpy.linspace(start=1 / class_count, stop=1.0, num=PROBABILITY_BIN_COUNT + 1)
    for selection, colour, label in [(correct, CORRECT_COLOUR, f"right: {int(correct.sum()):,} images"),
                                     (~correct, WRONG_COLOUR, f"wrong: {int((~correct).sum()):,} images")]:
        values = predicted_probabilities[selection]
        axes.hist(x=values, bins=probability_bins, weights=numpy.full(shape=len(values), fill_value=1 / len(values)),
                  color=colour, alpha=0.55, label=label, edgecolor=colour, linewidth=1.5 * POINTS_PER_PIXEL)
    axes.set_xlim(left=0, right=1.0)
    # Shares of the images written as the percentages of the reports, the probability axis keeping its decimals
    axes.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))
    axes.legend(loc="upper left", frameon=False, labelcolor=TEXT_PRIMARY_COLOUR, fontsize=18 * POINTS_PER_PIXEL)


def draw_pair_panel(axes: Any, pair_probabilities: numpy.ndarray, correct: numpy.ndarray, true_name: str,
                    confused_name: str) -> None:
    """
    Draws the right panel: for every test image of the true class of the pair, its probability of that class against
    its probability of the class it is confused with, right predictions in aqua and wrong ones in red, with the line
    where the two probabilities sum to 1.

    Args:
        axes (Any): The matplotlib panel.
        pair_probabilities (numpy.ndarray): The [M, 2] probabilities of the true and of the confused class.
        correct (numpy.ndarray): The [M] booleans of the right predictions.
        true_name (str): The true class of the pair, as the narration reads it.
        confused_name (str): The class it is confused with, as the narration reads it.
    """
    style_chart_panel(axes=axes, title=f"Test images of {true_name}",
                      x_label=f"Probability of {true_name}", y_label=f"Probability of {confused_name}")
    axes.plot([0, 1], [1, 0], color=TEXT_PRIMARY_COLOUR, linewidth=1.5 * POINTS_PER_PIXEL, linestyle="--", alpha=0.6)
    for selection, colour, label in [(correct, CORRECT_COLOUR, f"right: {int(correct.sum()):,}"),
                                     (~correct, WRONG_COLOUR, f"wrong: {int((~correct).sum()):,}")]:
        axes.scatter(x=pair_probabilities[selection, 0], y=pair_probabilities[selection, 1], s=10, color=colour,
                     alpha=0.45, linewidths=0, label=label)
    axes.set_xlim(left=0, right=1)
    axes.set_ylim(bottom=0, top=1)
    axes.set_aspect(aspect="equal")
    axes.legend(loc="upper right", frameon=False, labelcolor=TEXT_PRIMARY_COLOUR, fontsize=18 * POINTS_PER_PIXEL,
                markerscale=3)


def make_prediction_confidence_figure(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Draws the prediction confidence figure from the predictions of the last checkpoint and gives its facts: the counts
    and median probabilities of the right and wrong predictions, the share of confident ones in each group, the pair of
    the largest confusion of the checkpoint (read from its confusion matrix) and the split of the test images of its
    true class between the two classes, and the boxes of both panels.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the classification run of tutorial_sources.yaml.

    Returns:
        Dict[str, Any]: The facts, named confidence.*.
    """
    # Pair of the largest confusion of the checkpoint, from its confusion matrix
    evaluation_folder = run_folder / "Evaluation" / f"Iteration_{PREDICTIONS_ITERATION}"
    class_names, counts = read_confusion_matrix(evaluation_folder=evaluation_folder)
    (true_index, confused_index), = find_largest_confusions(counts=counts, pair_count=1)
    true_class, confused_class = class_names[true_index], class_names[confused_index]
    class_count = len(set(json.loads((run_folder / "label_dictionary.json").read_text(encoding="utf-8")).values()))

    # Predictions of every test image, and the test images of the true class of the pair
    predictions = read_predictions(predictions_path=evaluation_folder / "predictions.csv",
                                   pair_classes=[true_class, confused_class])
    predicted_probabilities = predictions["predicted_probabilities"]
    correct = predictions["correct"]
    pair_selection = numpy.array([name == true_class for name in predictions["true_classes"]])
    pair_probabilities = predictions["pair_probabilities"][pair_selection]
    pair_correct = correct[pair_selection]
    pair_to_confused = sum(1 for true_name, predicted_name in zip(predictions["true_classes"],
                                                                   predictions["predicted_classes"])
                           if true_name == true_class and predicted_name == confused_class)

    # Two panels of the size of the viewport of a scene, the right one square
    register_fonts()
    figure = pyplot.figure(figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH)
    figure.patch.set_facecolor(SURFACE_COLOUR)
    confidence_axes = figure.add_axes(rect=[0.065, 0.13, 0.50, 0.78])
    pair_axes = figure.add_axes(rect=[0.66, 0.13, 0.33, 0.78])
    draw_confidence_panel(axes=confidence_axes, predicted_probabilities=predicted_probabilities, correct=correct,
                          class_count=class_count)
    draw_pair_panel(axes=pair_axes, pair_probabilities=pair_probabilities, correct=pair_correct,
                    true_name=format_class_name(class_name=true_class),
                    confused_name=format_class_name(class_name=confused_class))
    output_path = output_folder / "evaluation" / FIGURE_NAME
    save_figure(figure=figure, output_path=output_path)
    facts = {"confidence.figure_box": [0, 0, round(CHART_WIDTH), round(CHART_HEIGHT)],
             "confidence.confidence_panel_box": compute_pixel_box(figure=figure, axes=confidence_axes,
                                                                  with_labels=True, margin=8),
             "confidence.pair_panel_box": compute_pixel_box(figure=figure, axes=pair_axes, with_labels=True,
                                                            margin=8)}
    pyplot.close(fig=figure)

    # Numbers of the narration
    facts.update({
        "confidence.iteration": PREDICTIONS_ITERATION, "confidence.test_images": int(len(correct)),
        "confidence.class_count": class_count, "confidence.lowest_probability": round_fact(value=1 / class_count),
        "confidence.correct_count": int(correct.sum()), "confidence.wrong_count": int((~correct).sum()),
        "confidence.correct_median": round_fact(value=numpy.median(predicted_probabilities[correct])),
        "confidence.wrong_median": round_fact(value=numpy.median(predicted_probabilities[~correct])),
        "confidence.confident_probability": CONFIDENT_PROBABILITY,
        "confidence.correct_confident_share": round_fact(
            value=(predicted_probabilities[correct] > CONFIDENT_PROBABILITY).mean()),
        "confidence.wrong_confident_share": round_fact(
            value=(predicted_probabilities[~correct] > CONFIDENT_PROBABILITY).mean()),
        "confidence.wrong_confident_count": int((predicted_probabilities[~correct] > CONFIDENT_PROBABILITY).sum()),
        "confidence.pair_true_class": format_class_name(class_name=true_class),
        "confidence.pair_predicted_class": format_class_name(class_name=confused_class),
        "confidence.pair_images": int(pair_selection.sum()), "confidence.pair_wrong": int((~pair_correct).sum()),
        "confidence.pair_wrong_to_predicted": pair_to_confused,
        "confidence.pair_mass_threshold": PAIR_MASS_THRESHOLD,
        "confidence.pair_mass_share": round_fact(value=(pair_probabilities.sum(axis=1) > PAIR_MASS_THRESHOLD).mean())})
    return facts
