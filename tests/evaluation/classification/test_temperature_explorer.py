"""
Tests of vision_training/evaluation/classification/temperature_explorer.py: the calibrated temperature finds the
temperature that made calibrated logits overconfident, the calibration error is 0 for probabilities matching their
frequencies, and the written page holds, for every temperature of its slider (the grid and the calibrated
temperature), the histograms of every class over its images and the other images and the thresholds found
by find_class_thresholds on the probabilities of the divided logits, the recall of two classes staying the same from
the temperature 1, in multi-class and multi-label mode, the classes the lowest F1 first and at most
MAXIMUM_EXPLORER_CLASSES (the save button then being turned off), a class name holding '</' never closing the
script of the data, no page without sample,
class histogram bins 0.02 wide, two scores of 0.977 and 0.985 falling in neighbouring bins, and reliability bins
0.02 wide, two confidences of 0.97 and 0.99 falling in neighbouring bins; the thresholds saved from the page are
written next to it only when they are those of its class_thresholds.json at a temperature above 0.
"""

import json
import math
import pytest
import re
import torch

from pathlib import Path
from typing import Any, Dict, Tuple

from vision_training.evaluation.classification import temperature_explorer
from vision_training.evaluation.classification.temperature_explorer import (EXPLORER_FILE_NAME,
                                                                           NUMBER_CALIBRATION_BINS, NUMBER_SCORE_BINS,
                                                                           SAVE_THRESHOLDS_ROUTE,
                                                                           SAVED_THRESHOLDS_FILE_NAME,
                                                                           TEMPERATURE_GRID, compute_calibration,
                                                                           describe_temperature,
                                                                           find_calibrated_temperature,
                                                                           save_temperature_explorer,
                                                                           write_saved_thresholds)
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
    Verifies the written page: its slider holds the grid and the calibrated temperature, every class is shown with
    its index so that the save button can write the thresholds of every class in the order of the classes, every
    temperature counts every image of every class once in its histograms, its thresholds
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
        target_precisions=[0.99, 0.9], class_f1_scores=[0.5] * number_classes, output_folder=tmp_path,
        title="Iteration 7")
    assert explorer_path == tmp_path / EXPLORER_FILE_NAME
    data = read_explorer_data(explorer_path=explorer_path)

    # Slider of the grid with the calibrated temperature, every class shown with its index and the save settings
    temperatures = [state["temperature"] for state in data["states"]]
    assert temperatures == sorted(set(TEMPERATURE_GRID + [data["calibrated_temperature"]]))
    assert data["title"] == "Iteration 7" and "evaluated_temperature" not in data
    assert data["class_names"] == class_names and data["positives"] == positive_masks.sum(dim=0).tolist()
    assert data["class_indices"] == list(range(number_classes)) and data["all_classes_shown"]
    assert data["saved_file_name"] == SAVED_THRESHOLDS_FILE_NAME and data["save_route"] == SAVE_THRESHOLDS_ROUTE
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
    the test, the page then telling that not every class is shown, which turns its save button off), and that a class
    name holding '</' is written so that it does not close the script of the data.
    """
    monkeypatch.setattr(target=temperature_explorer, name="MAXIMUM_EXPLORER_CLASSES", value=2)
    overconfident_logits, labels = build_overconfident_logits(number_samples=200, number_classes=4,
                                                              overconfidence=2.0, seed=2)
    class_names = ["zero", "one</script>", "two", "three"]
    explorer_path = save_temperature_explorer(
        class_logits=overconfident_logits, labels=labels, class_names=class_names, output_mode="multi_class",
        target_precisions=[0.99], class_f1_scores=[0.9, 0.2, 0.7, 0.4], output_folder=tmp_path, title=None)
    data = read_explorer_data(explorer_path=explorer_path)
    assert data["class_names"] == ["one</script>", "three"] and data["number_classes"] == 4
    assert data["class_indices"] == [1, 3] and not data["all_classes_shown"]
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
                                  class_f1_scores=[0.0, 0.0, 0.0], output_folder=tmp_path, title=None)
    assert not (tmp_path / EXPLORER_FILE_NAME).exists()


def test_class_histograms_separate_scores_near_one() -> None:
    """
    Verifies that the class histograms have bins 0.02 wide, so that two images of a class scored 0.977 and 0.985, where
    the thresholds of the kept classifiers gather, fall in the neighbouring bins 48 and 49 instead of one bar.
    """
    # Logits of two classes whose softmax gives the first class 0.977 and 0.985 at the temperature 1
    first_class_logits = [math.log(0.977 / 0.023), math.log(0.985 / 0.015)]
    class_logits = torch.tensor(data=[[logit, 0.0] for logit in first_class_logits], dtype=torch.float32)
    labels = torch.zeros(size=(2,), dtype=torch.int64)
    positive_masks = torch.nn.functional.one_hot(input=labels, num_classes=2).to(dtype=torch.bool)
    state = describe_temperature(class_logits=class_logits, labels=labels, positive_masks=positive_masks,
                                 output_mode="multi_class", logit_temperature=1.0, target_precisions=[0.99],
                                 shown_class_indices=[0])

    # One image in each of the last two bins of the first class, none elsewhere
    positive_counts = state["classes"][0]["positive_counts"]
    assert NUMBER_SCORE_BINS == 50 and len(positive_counts) == 50
    assert positive_counts[48] == 1 and positive_counts[49] == 1 and sum(positive_counts) == 2


def test_reliability_bins_separate_confidences_two_hundredths_apart() -> None:
    """
    Verifies that the confidence histogram and the reliability diagram have bins 0.02 wide, so that two decisions of a
    confidence of 0.97 and 0.99 fall in the neighbouring bins 48 and 49 instead of one bar.
    """
    probabilities = torch.tensor(data=[[0.97, 0.03], [0.99, 0.01]], dtype=torch.float64)
    calibration = compute_calibration(probabilities=probabilities, labels=torch.tensor(data=[0, 0]),
                                      output_mode="multi_class")

    # One decision in each of the last two bins, none elsewhere
    calibration_counts = calibration["calibration_counts"]
    assert NUMBER_CALIBRATION_BINS == 50 and len(calibration_counts) == 50
    assert calibration_counts[48] == 1 and calibration_counts[49] == 1 and sum(calibration_counts) == 2


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
    (build_saved_thresholds(logit_temperature=0, first_threshold=0.8), "above 0"),
    (build_saved_thresholds(logit_temperature=True, first_threshold=0.8), "above 0"),
    (build_saved_thresholds(logit_temperature=float("nan"), first_threshold=0.8), "above 0"),
    (build_saved_thresholds(logit_temperature="2", first_threshold=0.8), "above 0"),
    (build_saved_thresholds(logit_temperature=2.0, first_threshold=1.5), "within \\[0, 1\\]"),
    (build_saved_thresholds(logit_temperature=2.0, first_threshold="0.8"), "within \\[0, 1\\]"),
    ({key: value for key, value in build_saved_thresholds(logit_temperature=2.0, first_threshold=0.8).items()
      if key != "additional_targets"}, "to hold the fields"),
    ({**build_saved_thresholds(logit_temperature=2.0, first_threshold=0.8), "extra": 1}, "to hold the fields"),
    ({**build_saved_thresholds(logit_temperature=2.0, first_threshold=0.8), "target_precision": 0.9},
     "Expected the target precisions"),
    ({**build_saved_thresholds(logit_temperature=2.0, first_threshold=0.8), "classes": {"bad": None}},
     "Expected the class names"),
    ({**build_saved_thresholds(logit_temperature=2.0, first_threshold=0.8), "additional_targets": [1]},
     "additional_targets"),
    ([1, 2], "to hold the fields")])
def test_saved_thresholds_of_another_evaluation_are_refused(tmp_path: Path, saved_thresholds: Any,
                                                             expected_message: str) -> None:
    """
    Verifies that thresholds at a temperature that is not a number above 0, with a threshold outside [0, 1], with
    missing or extra fields, other target precisions or other classes than the class_thresholds.json of the folder are
    refused with the reason, and that nothing is written.
    """
    write_class_thresholds_file(evaluation_folder=tmp_path)
    with pytest.raises(expected_exception=ValueError, match=expected_message):
        write_saved_thresholds(evaluation_folder=tmp_path, saved_thresholds=saved_thresholds)
    assert not (tmp_path / SAVED_THRESHOLDS_FILE_NAME).exists()


def test_saved_thresholds_without_evaluation_are_refused(tmp_path: Path) -> None:
    """
    Verifies that thresholds saved to a folder without class_thresholds.json are refused, nothing being checked
    against.
    """
    with pytest.raises(expected_exception=FileNotFoundError, match="No class_thresholds.json"):
        write_saved_thresholds(evaluation_folder=tmp_path,
                               saved_thresholds=build_saved_thresholds(logit_temperature=2.0, first_threshold=0.8))
    assert not (tmp_path / SAVED_THRESHOLDS_FILE_NAME).exists()
