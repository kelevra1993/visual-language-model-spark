"""
Tests of vision_training/evaluation/classification/temperature_explorer.py: the page draws the classes in class index
order, each with the histogram of its score over its own images at every temperature of the grid, the thresholds saved
from the page are written next to it, those of other classes than its class_thresholds.json being refused, and the
class logits written next to the page give it back with any other temperature and the thresholds of every class at
it; the score ranges of the served page count the images of every
class by their own class score, a score equal to a cut point counting in the range above, and wrong cut points are
refused; write_explorer_evaluation writes the evaluation folder of the serve script tests. The percentages written by the script of the page are those of the report helpers
(run with Node.js when it is installed).
"""

import json
import math
import pytest
import re
import shutil
import subprocess
import torch

from pathlib import Path
from typing import Any, Dict, List, Tuple

from vision_training.evaluation.classification import temperature_explorer
from vision_training.evaluation.classification.temperature_explorer import (MAXIMUM_SCORE_CUT_POINTS,
                                                                           NUMBER_SCORE_BINS,
                                                                           SAVED_THRESHOLDS_FILE_NAME, TEMPLATE_PATH,
                                                                           TEMPERATURE_GRID,
                                                                           build_explorer_data_from_logits,
                                                                           check_logit_temperature,
                                                                           check_saved_thresholds,
                                                                           check_score_cut_points,
                                                                           compute_score_ranges,
                                                                           compute_temperature_probabilities,
                                                                           compute_temperature_thresholds,
                                                                           describe_explorer_temperature,
                                                                           load_class_logits,
                                                                           read_explorer_page_data,
                                                                           save_class_logits,
                                                                           save_temperature_explorer,
                                                                           write_saved_thresholds)
from vision_training.evaluation.report_formatting import format_percentage, format_percentage_label, format_points

# Classes of the evaluations written by write_explorer_evaluation, their F1 ordering the classes on the page
EXPLORER_CLASS_NAMES = ["square", "circle", "triangle"]
EXPLORER_CLASS_F1_SCORES = [0.6, 0.2, 0.9]


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


def write_explorer_evaluation(evaluation_folder: Path, seed: int, with_logits: bool) -> Dict[str, Any]:
    """
    Writes the evaluation folder of a classifier of three classes as the classification trainer writes it with the
    per-class report: the temperature explorer page, the class logits (when asked) and class_thresholds.json, the
    thresholds at the temperature 1 found from the logits.

    Args:
        evaluation_folder (Path): The evaluation folder, created.
        seed (int): The seed of the logits.
        with_logits (bool): True to write class_logits.npz, False for a page written without its logits.

    Returns:
        Dict[str, Any]: The class logits of the evaluation, as load_class_logits reads them.
    """
    evaluation_folder.mkdir(parents=True, exist_ok=True)
    class_logits, labels = build_overconfident_logits(number_samples=600, number_classes=3, overconfidence=3.0,
                                                      seed=seed)
    evaluation_settings = {"class_logits": class_logits, "labels": labels, "class_names": EXPLORER_CLASS_NAMES,
                           "output_mode": "multi_class", "target_precisions": [0.99, 0.9],
                           "class_f1_scores": EXPLORER_CLASS_F1_SCORES, "output_folder": evaluation_folder,
                           "title": "Iteration 6"}
    save_temperature_explorer(**evaluation_settings)
    logits_path = save_class_logits(**evaluation_settings)
    class_logits_record = load_class_logits(evaluation_folder=evaluation_folder)
    (evaluation_folder / "class_thresholds.json").write_text(data=json.dumps(obj=compute_temperature_thresholds(
        class_logits_record=class_logits_record, logit_temperature=1.0)))
    if not with_logits:
        logits_path.unlink()
    return class_logits_record


def list_page_thresholds(page_state: Dict[str, Any], class_indices: List[int]) -> Dict[str, List[Any]]:
    """
    Gives the [threshold, recall] of every class at every target precision of one temperature of the page, keyed by
    class name, to compare with a thresholds file.

    Args:
        page_state (Dict[str, Any]): One temperature of the page.
        class_indices (List[int]): The class index of every drawn class, in their order on the page.

    Returns:
        Dict[str, List[Any]]: The thresholds of every class.
    """
    return {EXPLORER_CLASS_NAMES[class_index]: page_state["classes"][shown_index]["thresholds"]
            for shown_index, class_index in enumerate(class_indices)}


def list_file_thresholds(thresholds: Dict[str, Any]) -> Dict[str, List[Any]]:
    """
    Gives the [threshold, recall] of every class at every target precision of a thresholds file, keyed by class name,
    to compare with the page.

    Args:
        thresholds (Dict[str, Any]): The thresholds file, in the format of class_thresholds.json.

    Returns:
        Dict[str, List[Any]]: The thresholds of every class.
    """
    targets = [thresholds] + thresholds["additional_targets"]
    return {class_name: [None if target["classes"][class_name] is None else
                         [target["classes"][class_name]["threshold"], target["classes"][class_name]["recall"]]
                         for target in targets]
            for class_name in EXPLORER_CLASS_NAMES}


def test_class_logits_give_back_the_page_and_any_temperature(tmp_path: Path) -> None:
    """
    Verifies that the class logits written next to the page give back its data exactly, the description of a
    temperature of its slider (2) and of a typed one (2.3, on no slider), and the thresholds of every class at both,
    equal to those the page shows and accepted against class_thresholds.json.
    """
    class_logits_record = write_explorer_evaluation(evaluation_folder=tmp_path, seed=3, with_logits=True)
    page_data = read_explorer_page_data(explorer_path=tmp_path / "temperature_explorer.html")
    assert build_explorer_data_from_logits(class_logits_record=class_logits_record) == page_data
    assert page_data["class_indices"] == [0, 1, 2] and page_data["all_classes_shown"]

    # A temperature of the slider is described again exactly, and its thresholds are those of the page
    grid_state = next(state for state in page_data["states"] if state["temperature"] == 2.0)
    assert describe_explorer_temperature(class_logits_record=class_logits_record,
                                         shown_class_indices=page_data["class_indices"],
                                         logit_temperature=2.0) == grid_state
    grid_thresholds = compute_temperature_thresholds(class_logits_record=class_logits_record, logit_temperature=2.0)
    assert list_file_thresholds(thresholds=grid_thresholds) == list_page_thresholds(
        page_state=grid_state, class_indices=page_data["class_indices"])

    # A typed temperature: the same thresholds on the page and in the saved file, checked against the evaluation
    typed_state = describe_explorer_temperature(class_logits_record=class_logits_record,
                                                shown_class_indices=page_data["class_indices"], logit_temperature=2.3)
    typed_thresholds = compute_temperature_thresholds(class_logits_record=class_logits_record, logit_temperature=2.3)
    assert typed_state["temperature"] == typed_thresholds["logit_temperature"] == 2.3
    assert list_file_thresholds(thresholds=typed_thresholds) == list_page_thresholds(
        page_state=typed_state, class_indices=page_data["class_indices"])
    assert list(typed_thresholds["classes"]) == EXPLORER_CLASS_NAMES
    check_saved_thresholds(saved_thresholds=typed_thresholds, evaluation_folder=tmp_path)


def test_class_histograms_count_the_own_images_in_class_index_order(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the page describes the temperatures of the grid only, draws the classes in class index order, each
    with the histogram of its score over its own images (adding up to its images, the other images left out), and that
    above the largest number of classes the classes of lowest F1 are kept, still in class index order.
    """
    class_logits_record = write_explorer_evaluation(evaluation_folder=tmp_path, seed=3, with_logits=True)
    page_data = build_explorer_data_from_logits(class_logits_record=class_logits_record)
    assert [state["temperature"] for state in page_data["states"]] == TEMPERATURE_GRID
    assert (page_data["class_indices"], page_data["class_names"]) == ([0, 1, 2], EXPLORER_CLASS_NAMES)

    # Own class scores of the images of every class at T = 2, counted bin by bin, a score of 1 in the last bin
    state = next(state for state in page_data["states"] if state["temperature"] == 2.0)
    assert set(state) == {"temperature", "classes"}
    probabilities = compute_temperature_probabilities(class_logits=class_logits_record["class_logits"],
                                                      output_mode="multi_class", logit_temperature=2.0)
    for class_index, class_state in enumerate(state["classes"]):
        own_scores = probabilities[class_logits_record["labels"] == class_index, class_index].tolist()
        expected_counts = [0] * NUMBER_SCORE_BINS
        for own_score in own_scores:
            expected_counts[min(math.floor(own_score * NUMBER_SCORE_BINS), NUMBER_SCORE_BINS - 1)] += 1
        assert set(class_state) == {"positive_counts", "thresholds"}
        assert class_state["positive_counts"] == expected_counts
        assert sum(expected_counts) == page_data["positives"][class_index] > 0

    # Two classes at most: the F1 0.6, 0.2 and 0.9 keep square and circle, drawn in class index order
    monkeypatch.setattr(target=temperature_explorer, name="MAXIMUM_EXPLORER_CLASSES", value=2)
    limited_page_data = build_explorer_data_from_logits(class_logits_record=class_logits_record)
    assert limited_page_data["class_indices"] == [0, 1] and not limited_page_data["all_classes_shown"]


@pytest.mark.parametrize(argnames="logit_temperature", argvalues=[0.0, -1.0, 0.01, 101.0, math.nan, True, "2"])
def test_temperatures_outside_the_range_are_refused(logit_temperature: Any) -> None:
    """
    Verifies that a temperature typed on the page outside LOGIT_TEMPERATURE_RANGE, or not a number, is refused, and
    that one within it is kept.
    """
    with pytest.raises(expected_exception=ValueError, match="Expected a logit temperature from 0.05 to 100"):
        check_logit_temperature(logit_temperature=logit_temperature)
    assert check_logit_temperature(logit_temperature=2.3) == 2.3


def test_score_ranges_count_every_class_by_its_own_score(tmp_path: Path) -> None:
    """
    Verifies that the score ranges of the served page count the images of every drawn class, in the order of the page,
    by the range their score for their own class falls in at the temperature asked, the counts of a class adding up to
    its images on the page.
    """
    class_logits_record = write_explorer_evaluation(evaluation_folder=tmp_path, seed=3, with_logits=True)
    page_data = read_explorer_page_data(explorer_path=tmp_path / "temperature_explorer.html")
    score_ranges = compute_score_ranges(class_logits_record=class_logits_record,
                                        shown_class_indices=page_data["class_indices"], logit_temperature=2.3,
                                        cut_points=[0.4, 0.8])
    assert (score_ranges["temperature"], score_ranges["cut_points"]) == (2.3, [0.4, 0.8])

    # Own class scores of the images of every drawn class, counted range by range
    probabilities = compute_temperature_probabilities(class_logits=class_logits_record["class_logits"],
                                                      output_mode="multi_class", logit_temperature=2.3)
    expected_counts = []
    for class_index in page_data["class_indices"]:
        own_scores = probabilities[class_logits_record["labels"] == class_index, class_index]
        expected_counts.append([int((own_scores < 0.4).sum()), int(((own_scores >= 0.4) & (own_scores < 0.8)).sum()),
                                int((own_scores >= 0.8).sum())])
    assert score_ranges["counts"] == expected_counts
    assert [sum(counts) for counts in score_ranges["counts"]] == page_data["positives"]

    # Every range holds images of every class, so that a count given to the wrong range is caught
    assert all(count > 0 for counts in expected_counts for count in counts)


def test_score_equal_to_a_cut_point_counts_in_the_range_above() -> None:
    """
    Verifies that an image scoring exactly a cut point is counted in the range above it, as an image scoring the
    threshold of its class is given the class: two equal logits score 0.5 at any temperature.
    """
    class_logits_record = {"class_logits": torch.tensor(data=[[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [-2.0, 0.0],
                                                              [0.0, 3.0]]),
                           "labels": torch.tensor(data=[0, 1, 0, 0, 1]), "output_mode": "multi_class"}
    score_ranges = compute_score_ranges(class_logits_record=class_logits_record, shown_class_indices=[1, 0],
                                        logit_temperature=1.0, cut_points=[0.5, 0.9])

    # The second class first, as the page draws it: own scores 0.5 and 0.95, then 0.5, 0.88 and 0.12
    assert score_ranges["counts"] == [[0, 1, 1], [1, 2, 0]]


@pytest.mark.parametrize(argnames="cut_points", argvalues=[
    [], [0.0], [1.0], [-0.2, 0.5], [0.5, 0.5], [math.nan], [math.inf], [True], [0.5, "0.8"], "0.5",
    [index / (MAXIMUM_SCORE_CUT_POINTS + 2) for index in range(1, MAXIMUM_SCORE_CUT_POINTS + 2)]])
def test_wrong_cut_points_are_refused(cut_points: Any) -> None:
    """
    Verifies that cut points of the score ranges outside ]0, 1[, repeated, not numbers or too many are refused, and that
    accepted ones are sorted, up to the largest number of cut points.
    """
    with pytest.raises(expected_exception=ValueError,
                       match=f"Expected 1 to {MAXIMUM_SCORE_CUT_POINTS} distinct scores strictly between 0 and 1"):
        check_score_cut_points(cut_points=cut_points)
    assert check_score_cut_points(cut_points=[0.8, 0.5, 0.95]) == [0.5, 0.8, 0.95]
    largest_list = [index / (MAXIMUM_SCORE_CUT_POINTS + 1) for index in range(MAXIMUM_SCORE_CUT_POINTS, 0, -1)]
    assert check_score_cut_points(cut_points=largest_list) == sorted(largest_list)


def test_page_written_before_the_save_button_is_refused(tmp_path: Path) -> None:
    """
    Verifies that the data of a page written before the save button (no class indices) is refused with the way out,
    the serve script being unable to save its thresholds.
    """
    write_explorer_evaluation(evaluation_folder=tmp_path, seed=3, with_logits=False)
    page_path = tmp_path / "temperature_explorer.html"
    page_path.write_text(data=page_path.read_text().replace('"class_indices"', '"former_class_indices"'))
    with pytest.raises(expected_exception=ValueError, match="written before the save button"):
        read_explorer_page_data(explorer_path=page_path)


@pytest.mark.skipif(condition=shutil.which("node") is None, reason="Node.js runs the script of the page")
def test_page_percentages_are_the_twins_of_the_report_helpers() -> None:
    """
    Verifies that formatPercent, formatPercentLabel and formatPoints of the page, run by Node.js from the template,
    write the rates exactly as format_percentage, format_percentage_label and format_points of the reports, so that a
    recall reads the same on the page, in Results.txt and on the figures.
    """
    template_text = TEMPLATE_PATH.read_text()
    helper_match = re.search(pattern=r"  const PERCENT_DECIMALS = .*?\n(?=\n  // Threshold written)",
                             string=template_text, flags=re.DOTALL)
    assert helper_match is not None

    # Rates, ticks and differences of the page, written by the script of the template
    rates = [0.9403, 1.0, 0.0, 0.123456, 0.05, 0.876543]
    labels = [0.99, 0.985, 0.2, 0.0, 1.0, 0.5]
    differences = [0.0052, -0.0052, -0.000001, 0.0, 0.1234, -0.00004]
    script = (f"{helper_match.group(0)}\nconsole.log(JSON.stringify(["
              f"{rates}.map(formatPercent), {labels}.map(formatPercentLabel), {differences}.map(formatPoints)]));")
    page_texts = json.loads(s=subprocess.run(args=["node", "-e", script], capture_output=True, text=True,
                                             check=True).stdout)
    assert page_texts == [[format_percentage(value=rate) for rate in rates],
                          [format_percentage_label(value=label) for label in labels],
                          [format_points(difference=difference) for difference in differences]]
