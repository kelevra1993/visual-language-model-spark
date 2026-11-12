"""
Tests of the percentages written on the figures of the classification evaluation: the confusion matrix image of
vision_training/evaluation/classification/confusion_matrix.py (cells, recall column, precision row, accuracy corner and
colour bar) and the ROC and precision-recall images of score_curves.py (title, legend, target precisions, axis ticks),
written as the percentages of the reports for the classification and kept as before for the other tasks.
"""

import pytest
import torch

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Dict, List
from unittest.mock import patch

from vision_training.evaluation.classification.confusion_matrix import draw_confusion_matrix, format_share
from vision_training.evaluation.classification.score_curves import compute_class_score_curves, draw_score_curves
from vision_training.evaluation.classification.threshold_finder import find_class_thresholds

# Confusion matrix of 3 classes: 7 of the 8 circles right (87.5 %), 1 of 3 squares right, no hexagon
CONFUSION_MATRIX = torch.tensor(data=[[7, 1, 0], [2, 1, 0], [0, 0, 0]])
CLASS_NAMES = ["circle", "square", "hexagon"]


def record_figure_texts(in_percent: bool) -> Dict[str, List[str]]:
    """
    Draws the confusion matrix and the precision-recall thresholds image of the tests and records the texts written on
    them: the texts of the cells and strips, the tick labels of the colour bar and of the curve axes, and the legend
    entries.

    Args:
        in_percent (bool): Whether the figures write the percentages of the reports.

    Returns:
        Dict[str, List[str]]: The 'matrix' texts, 'colour_bar' tick labels, legend 'labels', 'curve_ticks' of the
            precision axis and the 'curve_metrics' expected in the legend.
    """
    with patch.object(target=Axes, attribute="text", autospec=True, side_effect=Axes.text) as text_mock:
        draw_confusion_matrix(confusion_matrix=CONFUSION_MATRIX, class_names=CLASS_NAMES, title="Confusion Matrix",
                              in_percent=in_percent)
    matrix_texts = [call.kwargs["s"] for call in text_mock.call_args_list]

    # Colour bar labels, read from the formatter of the last colour bar drawn
    with patch.object(target=Figure, attribute="colorbar", autospec=True, side_effect=Figure.colorbar) as bar_mock:
        draw_confusion_matrix(confusion_matrix=CONFUSION_MATRIX, class_names=CLASS_NAMES, title="Confusion Matrix",
                              in_percent=in_percent)
    colour_bar_axis = bar_mock.call_args.kwargs["cax"].yaxis
    colour_bar_labels = [colour_bar_axis.get_major_formatter()(value, index)
                         for index, value in enumerate([0.0, 0.2, 1.0])]

    # Precision-recall thresholds image of random scores, two classes on a single plot
    generator = torch.Generator().manual_seed(1)
    labels = torch.randint(low=0, high=2, size=(200,), generator=generator)
    logits = torch.randn(size=(200, 2), generator=generator)
    logits[torch.arange(200), labels] += 2.0
    probabilities = logits.softmax(dim=1)
    positive_masks = torch.nn.functional.one_hot(labels, num_classes=2).to(dtype=torch.bool)
    class_thresholds = [find_class_thresholds(probabilities=probabilities, positive_masks=positive_masks,
                                              target_precision=target_precision) for target_precision in [0.99, 0.9]]
    class_curves = compute_class_score_curves(probabilities=probabilities, positive_masks=positive_masks,
                                              class_thresholds=class_thresholds)
    with patch.object(target=Axes, attribute="legend", autospec=True, side_effect=Axes.legend) as legend_mock:
        draw_score_curves(class_curves=class_curves, class_names=["cat", "dog"],
                          image_name="Precision Recall Thresholds", title="Precision-Recall Thresholds",
                          target_precisions=[0.99, 0.9], in_percent=in_percent)

    # Legend entries and tick labels of the precision axis of the single plot
    legend_axes = legend_mock.call_args.args[0]
    precision_axis = legend_axes.yaxis
    return {"matrix": matrix_texts, "colour_bar": colour_bar_labels,
            "labels": legend_axes.get_legend_handles_labels()[1],
            "curve_ticks": [precision_axis.get_major_formatter()(value, 0) for value in [0.9, 0.95]],
            "curve_metrics": [f"{100.0 * curves['average_precision']:.2f} %" for curves in class_curves]}


def test_classification_figures_write_the_percentages_of_the_reports() -> None:
    """
    Verifies the classification figures: every share of the confusion matrix "87.50 %", its colour bar "20 %", the
    legend of the curves "cat · 97.12 %" with the target precisions "99 % · 90 %", and the precision axis "95 %".
    """
    texts = record_figure_texts(in_percent=True)
    assert texts["matrix"] == ["7\n87.50 %", "1\n12.50 %", "2\n66.67 %", "1\n33.33 %", "0", "87.50 %", "33.33 %",
                               "77.78 %", "50.00 %", "72.73 %"]
    assert texts["colour_bar"] == ["0 %", "20 %", "100 %"]
    assert texts["labels"][:2] == ["Target Precision 99 % · 90 %", "Threshold Reaching 99 % · 90 %"]
    assert texts["labels"][2:] == [f"cat · {texts['curve_metrics'][0]}", f"dog · {texts['curve_metrics'][1]}"]
    assert texts["curve_ticks"] == ["90 %", "95 %"]


def test_other_tasks_keep_their_figures() -> None:
    """
    Verifies that the figures drawn without the percentages, for the tasks not yet moved to them (detection,
    segmentation, frame selection), keep "87.5%", the colour bar "20%", the average precisions with 4 decimals and the
    target precisions "0.99 · 0.9".
    """
    texts = record_figure_texts(in_percent=False)
    assert texts["matrix"] == ["7\n87.5%", "1\n12.5%", "2\n66.7%", "1\n33.3%", "0", "87.5%", "33.3%", "77.8%", "50.0%",
                               "72.7%"]
    assert texts["colour_bar"] == ["0%", "20%", "100%"]
    assert texts["labels"][:2] == ["Target Precision 0.99 · 0.9", "Threshold Reaching 0.99 · 0.9"]
    assert all("%" not in label for label in texts["labels"])


@pytest.mark.parametrize(argnames="share, in_percent, expected_text",
                         argvalues=[(0.875, True, "87.50 %"), (0.875, False, "87.5%"), (1.0, True, "100.00 %")])
def test_matrix_shares_are_written_by_their_task(share: float, in_percent: bool, expected_text: str) -> None:
    """
    Verifies the writing of one share of the confusion matrix image, as the percentage of the reports for the
    classification and with one decimal for the other tasks.

    Args:
        share (float): The share, between 0 and 1.
        in_percent (bool): Whether the image writes the percentages of the reports.
        expected_text (str): The expected text.
    """
    assert format_share(share=share, in_percent=in_percent) == expected_text
