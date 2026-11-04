"""
Tests of vision_training/evaluation/classification/temperature_explorer.py: the calibrated temperature finds the
temperature that made calibrated logits overconfident, the calibration error is 0 for probabilities matching their
frequencies, and the thresholds saved from the page are written next to it, those of other classes than its
class_thresholds.json being refused; read_explorer_data reads the data of a written page for the classification
trainer tests.
"""

import json
import pytest
import re
import torch

from pathlib import Path
from typing import Any, Dict, Tuple

from vision_training.evaluation.classification.temperature_explorer import (SAVED_THRESHOLDS_FILE_NAME,
                                                                           compute_calibration,
                                                                           find_calibrated_temperature,
                                                                           write_saved_thresholds)


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


def build_saved_thresholds(logit_temperature: Any, first_threshold: Any) -> Dict[str, Any]:
    """
    Builds thresholds as the save button of the page posts them for an evaluation of two target precisions: the main
    target in 'target_precision' and 'classes', the second in 'additional_targets', a class not reaching 0.99 being
    null.

    Args:
        logit_temperature (Any): The temperature of the thresholds.
        first_threshold (Any): The threshold of the first class at the main target precision.

    Returns:
        Dict[str, Any]: The saved thresholds.
    """
    return {"logit_temperature": logit_temperature, "target_precision": 0.99,
            "classes": {"bad": {"threshold": first_threshold, "recall": 0.4}, "good": None},
            "additional_targets": [{"target_precision": 0.9,
                                    "classes": {"bad": {"threshold": 0.6, "recall": 0.8},
                                                "good": {"threshold": 0.7, "recall": 0.9}}}]}


def write_class_thresholds_file(evaluation_folder: Path) -> None:
    """
    Writes the class_thresholds.json of an evaluation of the classes bad and good at the target precisions 0.99 and
    0.9, the reference the saved thresholds are checked against.

    Args:
        evaluation_folder (Path): The evaluation folder of the checkpoint.
    """
    (evaluation_folder / "class_thresholds.json").write_text(
        data=json.dumps(obj=build_saved_thresholds(logit_temperature=1.0, first_threshold=0.95)))


def test_saved_thresholds_are_written_next_to_the_page(tmp_path: Path) -> None:
    """
    Verifies that thresholds of the classes and target precisions of the evaluation, at a temperature above 0, are
    written next to the page as posted, a second save replacing the first.
    """
    write_class_thresholds_file(evaluation_folder=tmp_path)
    for logit_temperature in [2.0, 3]:
        saved_thresholds = build_saved_thresholds(logit_temperature=logit_temperature, first_threshold=0.8)
        saved_path = write_saved_thresholds(evaluation_folder=tmp_path, saved_thresholds=saved_thresholds)
        assert saved_path == tmp_path / SAVED_THRESHOLDS_FILE_NAME
        assert json.loads(s=saved_path.read_text()) == saved_thresholds


@pytest.mark.parametrize(argnames=("saved_thresholds", "expected_message"), argvalues=[
    ({**build_saved_thresholds(logit_temperature=2.0, first_threshold=0.8), "classes": {"bad": None}},
     "Expected the class names")])
def test_saved_thresholds_of_another_evaluation_are_refused(tmp_path: Path, saved_thresholds: Any,
                                                             expected_message: str) -> None:
    """
    Verifies that thresholds of other classes than the class_thresholds.json of the folder are refused with the
    reason, and that nothing is written.
    """
    write_class_thresholds_file(evaluation_folder=tmp_path)
    with pytest.raises(expected_exception=ValueError, match=expected_message):
        write_saved_thresholds(evaluation_folder=tmp_path, saved_thresholds=saved_thresholds)
    assert not (tmp_path / SAVED_THRESHOLDS_FILE_NAME).exists()
