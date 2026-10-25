"""
Prediction confidence figure of the classification tutorials (the chapter judging and shipping a classifier), made at
every build by tutorials/figures/classification/figures.py from the predictions.csv written by the evaluation of the
last checkpoint of the classification run (save_predictions): on the left, the distribution of the probability of the
predicted class over the right and the wrong predictions of the 50,000 test images, which tells whether a threshold on
that probability can separate them; on the right, the probabilities of pentagon and hexagon of every test pentagon,
the most confused pair of the run, which shows a confusion between two classes taking the probability mass from each
other. The boxes of the two panels and the numbers the narration quotes are the facts of the scene.
"""

import csv

import numpy

from matplotlib import pyplot
from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, POINTS_PER_PIXEL,
                                                SERIES_COLOURS, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR,
                                                compute_pixel_box, register_fonts, round_fact, save_figure,
                                                style_chart_panel)
from tutorials.tutorial_builder.layout import NAMED_COLOURS

# Evaluated checkpoint whose predictions are drawn
PREDICTIONS_ITERATION = 1500

# Bins of the probability of the predicted class, which is at least 1/9 with 9 classes
PROBABILITY_BINS = numpy.linspace(start=0.1, stop=1.0, num=46)

# Probability above which a prediction counts as confident in the facts
CONFIDENT_PROBABILITY = 0.9

# Pair of classes of the right panel: the true class of the drawn images and the class it is confused with
CONFUSED_TRUE_CLASS = "pentagon"
CONFUSED_PREDICTED_CLASS = "hexagon"

# Sum of the two probabilities above which an image counts as split between the two classes only
PAIR_MASS_THRESHOLD = 0.95

# Colours of the right and wrong predictions: aqua and red of the palette of the videos
CORRECT_COLOUR = SERIES_COLOURS[1]
WRONG_COLOUR = "#%02x%02x%02x" % NAMED_COLOURS["red"]

# Name of the figure file
FIGURE_NAME = "prediction_confidence_iteration_1500.png"


def read_predictions(predictions_path: Path) -> Dict[str, Any]:
    """
    Reads the predictions file of an evaluation: the true and predicted class of every test image, whether it is right
    and the probability of every class, the columns named probability_<class>.

    Args:
        predictions_path (Path): The predictions.csv file.

    Returns:
        Dict[str, Any]: The 'class_names', the 'true_classes' and 'predicted_classes' lists, the 'correct' [N] booleans
            and the [N, K] 'probabilities'.

    Raises:
        FileNotFoundError: If the file is missing (save_predictions was off).
    """
    if not predictions_path.exists():
        raise FileNotFoundError(f"No predictions file {predictions_path}: the evaluation needs save_predictions.")
    with open(file=predictions_path, encoding="utf-8", newline="") as predictions_file:
        reader = csv.DictReader(predictions_file)
        probability_columns = [column for column in reader.fieldnames if column.startswith("probability_")]
        rows = list(reader)
    return {"class_names": [column.removeprefix("probability_") for column in probability_columns],
            "true_classes": [row["true_class"] for row in rows],
            "predicted_classes": [row["predicted_class"] for row in rows],
            "correct": numpy.array([row["correct"] == "1" for row in rows]),
            "probabilities": numpy.array([[float(row[column]) for column in probability_columns] for row in rows])}


def draw_confidence_panel(axes: Any, maximum_probabilities: numpy.ndarray, correct: numpy.ndarray) -> None:
    """
    Draws the left panel: the distribution of the probability of the predicted class over the right predictions and
    over the wrong ones, each as the share of its own images per bin, so that both are visible whatever their counts.

    Args:
        axes (Any): The matplotlib panel.
        maximum_probabilities (numpy.ndarray): The [N] probabilities of the predicted classes.
        correct (numpy.ndarray): The [N] booleans of the right predictions.
    """
    style_chart_panel(axes=axes, title="Probability of the predicted class", x_label="Probability of the predicted class",
                      y_label="Share of the images of the group")
    for selection, colour, label in [(correct, CORRECT_COLOUR, f"right: {int(correct.sum()):,} images"),
                                     (~correct, WRONG_COLOUR, f"wrong: {int((~correct).sum()):,} images")]:
        values = maximum_probabilities[selection]
        axes.hist(x=values, bins=PROBABILITY_BINS, weights=numpy.full(shape=len(values), fill_value=1 / len(values)),
                  color=colour, alpha=0.55, label=label, edgecolor=colour, linewidth=1.5 * POINTS_PER_PIXEL)
    axes.set_xlim(left=0.1, right=1.0)
    axes.legend(loc="upper left", frameon=False, labelcolor=TEXT_PRIMARY_COLOUR, fontsize=18 * POINTS_PER_PIXEL)


def draw_pair_panel(axes: Any, pair_probabilities: numpy.ndarray, correct: numpy.ndarray) -> None:
    """
    Draws the right panel: for every test image of CONFUSED_TRUE_CLASS, its probability of that class against its
    probability of CONFUSED_PREDICTED_CLASS, right predictions in aqua and wrong ones in red, with the line where the
    two probabilities sum to 1.

    Args:
        axes (Any): The matplotlib panel.
        pair_probabilities (numpy.ndarray): The [M, 2] probabilities of the true and of the confused class.
        correct (numpy.ndarray): The [M] booleans of the right predictions.
    """
    style_chart_panel(axes=axes, title=f"Test {CONFUSED_TRUE_CLASS}s: two probabilities",
                      x_label=f"Probability of {CONFUSED_TRUE_CLASS}", y_label=f"Probability of {CONFUSED_PREDICTED_CLASS}")
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
    and median probabilities of the right and wrong predictions, the share of confident ones in each group, the split of
    the test pentagons between pentagon and hexagon, and the boxes of both panels.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the classification run of tutorial_sources.yaml.

    Returns:
        Dict[str, Any]: The facts, named confidence.*.
    """
    predictions = read_predictions(predictions_path=run_folder / "Evaluation" / f"Iteration_{PREDICTIONS_ITERATION}" /
                                   "predictions.csv")
    maximum_probabilities = predictions["probabilities"].max(axis=1)
    correct = predictions["correct"]

    # Test images of the confused pair: their probabilities of both classes
    class_names: List[str] = predictions["class_names"]
    pair_indices = [class_names.index(CONFUSED_TRUE_CLASS), class_names.index(CONFUSED_PREDICTED_CLASS)]
    pair_selection = numpy.array([true_class == CONFUSED_TRUE_CLASS for true_class in predictions["true_classes"]])
    pair_probabilities = predictions["probabilities"][pair_selection][:, pair_indices]
    pair_correct = correct[pair_selection]

    # Two panels of the size of the viewport of a scene, the right one square
    register_fonts()
    figure = pyplot.figure(figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH)
    figure.patch.set_facecolor(SURFACE_COLOUR)
    confidence_axes = figure.add_axes(rect=[0.065, 0.13, 0.50, 0.78])
    pair_axes = figure.add_axes(rect=[0.66, 0.13, 0.33, 0.78])
    draw_confidence_panel(axes=confidence_axes, maximum_probabilities=maximum_probabilities, correct=correct)
    draw_pair_panel(axes=pair_axes, pair_probabilities=pair_probabilities, correct=pair_correct)
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
        "confidence.correct_count": int(correct.sum()), "confidence.wrong_count": int((~correct).sum()),
        "confidence.correct_median": round_fact(value=numpy.median(maximum_probabilities[correct])),
        "confidence.wrong_median": round_fact(value=numpy.median(maximum_probabilities[~correct])),
        "confidence.confident_probability": CONFIDENT_PROBABILITY,
        "confidence.correct_confident_share": round_fact(
            value=(maximum_probabilities[correct] > CONFIDENT_PROBABILITY).mean()),
        "confidence.wrong_confident_share": round_fact(
            value=(maximum_probabilities[~correct] > CONFIDENT_PROBABILITY).mean()),
        "confidence.wrong_confident_count": int((maximum_probabilities[~correct] > CONFIDENT_PROBABILITY).sum()),
        "confidence.pair_true_class": CONFUSED_TRUE_CLASS, "confidence.pair_predicted_class": CONFUSED_PREDICTED_CLASS,
        "confidence.pair_images": int(pair_selection.sum()), "confidence.pair_wrong": int((~pair_correct).sum()),
        "confidence.pair_mass_threshold": PAIR_MASS_THRESHOLD,
        "confidence.pair_mass_share": round_fact(value=(pair_probabilities.sum(axis=1) > PAIR_MASS_THRESHOLD).mean())})
    return facts
