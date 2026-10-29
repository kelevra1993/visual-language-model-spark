"""
Tests of vision_training/evaluation/classification/temperature_explorer.py: the calibrated temperature finds the
temperature that made calibrated logits overconfident, the calibration error is 0 for probabilities matching their
frequencies, and the written page holds, for every temperature of its slider (the grid, the evaluated and the
calibrated temperatures), the histograms of every class over its images and the other images and the thresholds found
by find_class_thresholds on the probabilities of the divided logits, the recall of two classes staying the same from
the temperature 1, in multi-class and multi-label mode, the classes the lowest F1 first and at most
MAXIMUM_EXPLORER_CLASSES, a class name holding '</' never closing the script of the data, and no page without sample.
"""

import json
import math
import pytest
import re
import torch

from pathlib import Path
from typing import Any, Dict, Tuple

from vision_training.evaluation.classification import temperature_explorer
from vision_training.evaluation.classification.temperature_explorer import (EXPLORER_FILE_NAME, NUMBER_SCORE_BINS,
                                                                           TEMPERATURE_GRID, compute_calibration,
                                                                           find_calibrated_temperature,
                                                                           save_temperature_explorer)
from vision_training.evaluation.classification.threshold_finder import find_class_thresholds
from vision_training.losses.classification_losses import convert_logits_to_probabilities


def build_overconfident_logits(number_samples: int, number_classes: int, overconfidence: float,
                               seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Builds the logits of an overconfident classifier: calibrated logits, whose labels are drawn from their softmax,
    multiplied by an overconfidence factor, the temperature giving back the calibrated probabilities.

    Args:
        number_samples (int): The number of samples N.
        number_classes (int): The number of classes K.
        overconfidence (float): The factor multiplying the calibrated logits.
        seed (int): The seed of the random generator.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The [N, K] float32 overconfident logits and the [N] int64 labels.
    """
    generator = torch.Generator().manual_seed(seed)
    calibrated_logits = 2.0 * torch.randn(size=(number_samples, number_classes), generator=generator)
    labels = torch.multinomial(input=torch.softmax(input=calibrated_logits, dim=1), num_samples=1,
                               generator=generator).flatten()
    return calibrated_logits * overconfidence, labels


def read_explorer_data(explorer_path: Path) -> Dict[str, Any]:
    """
    Reads the data written into a temperature explorer page, as the script of the page parses it.

    Args:
        explorer_path (Path): The written page.

    Returns:
        Dict[str, Any]: The data of the page.
    """
    match = re.search(pattern=r'<script id="explorer-data" type="application/json">(.*?)</script>',
                      string=explorer_path.read_text(), flags=re.DOTALL)
    return json.loads(s=match.group(1))


def test_calibrated_temperature_finds_the_overconfidence() -> None:
    """
    Verifies that the calibrated temperature of logits made 3 times overconfident is close to 3, and close to 1 for
    the calibrated logits themselves.
    """
    overconfident_logits, labels = build_overconfident_logits(number_samples=20000, number_classes=4,
                                                              overconfidence=3.0, seed=0)
    assert abs(find_calibrated_temperature(class_logits=overconfident_logits, labels=labels,
                                           output_mode="multi_class") - 3.0) < 0.15
    assert abs(find_calibrated_temperature(class_logits=overconfident_logits / 3.0, labels=labels,
                                           output_mode="multi_class") - 1.0) < 0.05


def test_calibration_error_compares_confidences_and_frequencies() -> None:
    """
    Verifies the reliability bins on hand-made decisions: four decisions at 0.75 with three right and two at 0.5 with
    one right are calibrated (error 0), the same probabilities with every decision wrong give an error equal to the
    mean confidence, the empty bins having no confidence nor frequency, and the multi-label decisions are the presence
    or absence of every class, of confidence max(p, 1 - p).
    """
    probabilities = torch.tensor(data=[[0.75, 0.25]] * 4 + [[0.5, 0.5]] * 2, dtype=torch.float64)
    calibration = compute_calibration(probabilities=probabilities, labels=torch.tensor(data=[0, 0, 0, 1, 0, 1]),
                                      output_mode="multi_class")
    assert calibration["calibration_error"] == pytest.approx(expected=0.0, abs=1e-12)
    assert sum(calibration["calibration_counts"]) == 6
    assert calibration["mean_confidence"] == pytest.approx(expected=4.0 / 6.0)
    occupied_bins = [bin_index for bin_index, count in enumerate(calibration["calibration_counts"]) if count]
    assert [calibration["calibration_frequencies"][bin_index] for bin_index in occupied_bins] == [0.5, 0.75]
    assert all(calibration["calibration_confidences"][bin_index] is None
               for bin_index, count in enumerate(calibration["calibration_counts"]) if not count)

    # The decisions at 0.5 take the first class, so the label 1 makes every decision wrong
    wrong_calibration = compute_calibration(probabilities=probabilities, labels=torch.tensor(data=[1] * 6),
                                            output_mode="multi_class")
    assert wrong_calibration["calibration_error"] == pytest.approx(expected=4.0 / 6.0)

    # Multi-label: a probability of 0.2 decides the absence with the confidence 0.8, right for an absent class
    multi_label_calibration = compute_calibration(probabilities=torch.tensor(data=[[0.2, 0.9]], dtype=torch.float64),
                                                  labels=torch.tensor(data=[[0.0, 1.0]]), output_mode="multi_label")
    assert multi_label_calibration["mean_confidence"] == pytest.approx(expected=0.85)
    assert multi_label_calibration["calibration_error"] == pytest.approx(expected=0.15)


@pytest.mark.parametrize(argnames=("number_classes", "output_mode"),
                         argvalues=[(2, "multi_class"), (5, "multi_class"), (3, "multi_label")])
def test_explorer_page_describes_every_temperature(tmp_path: Path, number_classes: int, output_mode: str) -> None:
    """
    Verifies the page written for an evaluated temperature of 2: its slider holds the grid, the evaluated and the
    calibrated temperatures, every temperature counts every image of every class once in its histograms, its thresholds
    equal those found on the probabilities of the logits divided by it (those class_thresholds.json would hold), the
    recall of two classes (or of independent sigmoids) stays the same from the temperature 1 since only the threshold
    moves, and the predicted class and accuracy of the decisions do not depend on the temperature.
    """
    overconfident_logits, labels = build_overconfident_logits(number_samples=400, number_classes=number_classes,
                                                              overconfidence=1.0, seed=1)
    if output_mode == "multi_label":
        labels = torch.nn.functional.one_hot(input=labels, num_classes=number_classes).to(dtype=torch.float32)
    positive_masks = labels >= 0.5 if output_mode == "multi_label" else torch.nn.functional.one_hot(
        input=labels, num_classes=number_classes).to(dtype=torch.bool)
    class_names = [f"class_{class_index}" for class_index in range(number_classes)]
    explorer_path = save_temperature_explorer(
        class_logits=overconfident_logits, labels=labels, class_names=class_names, output_mode=output_mode,
        target_precisions=[0.99, 0.9], evaluated_temperature=2.0, class_f1_scores=[0.5] * number_classes,
        output_folder=tmp_path, title="Iteration 7")
    assert explorer_path == tmp_path / EXPLORER_FILE_NAME
    data = read_explorer_data(explorer_path=explorer_path)

    # Slider of the grid with the evaluated and calibrated temperatures, the page opening on the evaluated one
    temperatures = [state["temperature"] for state in data["states"]]
    assert temperatures == sorted(set(TEMPERATURE_GRID + [2.0, data["calibrated_temperature"]]))
    assert data["evaluated_temperature"] == 2.0 and data["title"] == "Iteration 7"
    assert data["class_names"] == class_names and data["positives"] == positive_masks.sum(dim=0).tolist()
    assert "__EXPLORER_DATA__" not in explorer_path.read_text()

    for state in data["states"]:
        # Every image counted once per class, in its histogram or in that of the other images
        probabilities = convert_logits_to_probabilities(class_logits=overconfident_logits / state["temperature"],
                                                        output_mode=output_mode).to(dtype=torch.float64)
        for class_index, class_state in enumerate(state["classes"]):
            assert len(class_state["positive_counts"]) == NUMBER_SCORE_BINS
            assert sum(class_state["positive_counts"]) == data["positives"][class_index]
            assert sum(class_state["negative_counts"]) == 400 - data["positives"][class_index]

        # Thresholds of every target precision, those found on the probabilities of the divided logits
        for target_index, target_precision in enumerate(data["target_precisions"]):
            expected_thresholds = find_class_thresholds(probabilities=probabilities, positive_masks=positive_masks,
                                                        target_precision=target_precision)
            assert [None if class_state["thresholds"][target_index] is None else
                    tuple(class_state["thresholds"][target_index]) for class_state in state["classes"]] == \
                expected_thresholds

        # The decisions and their accuracy do not depend on the temperature, only their confidence
        decisions = sum(state["calibration_counts"])
        assert decisions == (400 if output_mode == "multi_class" else 400 * number_classes)
        right_decisions = sum(frequency * count for frequency, count in
                              zip(state["calibration_frequencies"], state["calibration_counts"]) if count)
        reference_state = data["states"][temperatures.index(1.0)]
        assert right_decisions == pytest.approx(expected=sum(
            frequency * count for frequency, count in zip(reference_state["calibration_frequencies"],
                                                          reference_state["calibration_counts"]) if count))

    # With two classes, or independent sigmoids, the ranking of the images of every class does not change, so the
    # recall at every target precision is the same from the temperature 1 (below it, the float32 probabilities of the
    # largest logit gaps round to exactly 1 and tie, which changes the recall)
    if number_classes == 2 or output_mode == "multi_label":
        for target_index in range(len(data["target_precisions"])):
            recalls = {tuple(None if class_state["thresholds"][target_index] is None else
                             class_state["thresholds"][target_index][1] for class_state in state["classes"])
                       for state in data["states"] if state["temperature"] >= 1.0}
            assert len(recalls) == 1

    # Softer probabilities at higher temperatures: the mean confidence decreases along the slider
    mean_confidences = [state["mean_confidence"] for state in data["states"]]
    assert all(later <= earlier + 1e-12 for earlier, later in zip(mean_confidences, mean_confidences[1:]))
    assert all(math.isfinite(state["negative_log_likelihood"]) for state in data["states"])


def test_explorer_draws_the_lowest_f1_first_and_at_most_the_maximum(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the classes are drawn by increasing F1, at most MAXIMUM_EXPLORER_CLASSES of them (lowered to 2 for
    the test), and that a class name holding '</' is written so that it does not close the script of the data.
    """
    monkeypatch.setattr(target=temperature_explorer, name="MAXIMUM_EXPLORER_CLASSES", value=2)
    overconfident_logits, labels = build_overconfident_logits(number_samples=200, number_classes=4,
                                                              overconfidence=2.0, seed=2)
    class_names = ["zero", "one</script>", "two", "three"]
    explorer_path = save_temperature_explorer(
        class_logits=overconfident_logits, labels=labels, class_names=class_names, output_mode="multi_class",
        target_precisions=[0.99], evaluated_temperature=1.0, class_f1_scores=[0.9, 0.2, 0.7, 0.4],
        output_folder=tmp_path, title=None)
    data = read_explorer_data(explorer_path=explorer_path)
    assert data["class_names"] == ["one</script>", "three"] and data["number_classes"] == 4
    assert all(len(state["classes"]) == 2 for state in data["states"])
    assert explorer_path.read_text().count("</script>") == 2


def test_explorer_without_sample_is_refused(tmp_path: Path) -> None:
    """
    Verifies that an evaluation without any test sample gives no page, with an error naming the explorer.
    """
    with pytest.raises(expected_exception=ValueError, match="at least one evaluated test sample"):
        save_temperature_explorer(class_logits=torch.zeros(size=(0, 3)), labels=torch.zeros(size=(0,),
                                                                                            dtype=torch.int64),
                                  class_names=["a", "b", "c"], output_mode="multi_class", target_precisions=[0.99],
                                  evaluated_temperature=1.0, class_f1_scores=[0.0, 0.0, 0.0], output_folder=tmp_path,
                                  title=None)
    assert not (tmp_path / EXPLORER_FILE_NAME).exists()
