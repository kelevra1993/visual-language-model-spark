"""
Temperature explorer of the classification evaluation: the class logits of the test split are divided by every
temperature of a grid, giving for each the distribution of the probability of every class over its images and over the
other images, the thresholds and recalls of the target precisions, and the calibration of the probabilities (negative
log likelihood, calibration error, reliability), written as the self-contained page 'temperature_explorer.html' in the
evaluation folder of the checkpoint, next to the logits themselves ('class_logits.npz'). Written by the classification
trainer with the per-class report, to choose the temperature of the export:
scripts/evaluation/serve_temperature_explorer.py serves the page again from the logits with the current template,
shows any temperature, counts the images of every class in score ranges of the user's choice at the temperature shown
(compute_score_ranges), saves the thresholds of the chosen one to 'saved_temperature_thresholds.json' (checked here
against class_thresholds.json), and exports the classifier with that temperature
(vision_training/deployment/explorer_export.py).
"""

import json
import math
import numpy as np
import re
import torch

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vision_training.evaluation.classification.threshold_finder import (build_class_thresholds_file,
                                                                        find_class_thresholds)
from vision_training.losses.classification_losses import compute_classification_loss, convert_logits_to_probabilities

# Temperatures of the slider of the page, round values to choose a temperature from, the calibrated one being added to
# them (and, on a served page, every temperature typed in)
TEMPERATURE_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0,
                    7.0, 8.0, 9.0, 10.0]

# Range searched for the calibrated temperature, and the number of steps of the golden section search
CALIBRATION_TEMPERATURE_RANGE = (0.05, 100.0)
NUMBER_CALIBRATION_SEARCH_STEPS = 40

# Bins of the histograms of the class probabilities over [0, 1], and of the confidence histogram, the reliability
# diagram and the calibration error, both 0.02 wide to see where the scores gather near 0 and 1
NUMBER_SCORE_BINS = 50
NUMBER_CALIBRATION_BINS = 50

# Largest number of classes drawn, the classes of the lowest F1 being kept above it
MAXIMUM_EXPLORER_CLASSES = 100

# Template of the page, the data of the temperatures replacing its placeholder
TEMPLATE_PATH = Path(__file__).absolute().parent / "temperature_explorer_template.html"
DATA_PLACEHOLDER = "__EXPLORER_DATA__"
EXPLORER_FILE_NAME = "temperature_explorer.html"
EXPLORER_DATA_PATTERN = re.compile(pattern=r'<script id="explorer-data" type="application/json">(.*?)</script>',
                                   flags=re.DOTALL)

# Logits of the evaluated test samples with what the page needs to be built again, written next to the page up to
# MAXIMUM_EXPLORER_CLASSES classes
CLASS_LOGITS_FILE_NAME = "class_logits.npz"

# Thresholds of the temperature chosen on the page, in the format of class_thresholds.json, written next to the page by
# scripts/evaluation/serve_temperature_explorer.py and read by the export
SAVED_THRESHOLDS_FILE_NAME = "saved_temperature_thresholds.json"

# Largest number of cut points of the score ranges table of the served page, which splits the images of every class by
# the range of [0, 1] their own class score falls in
MAXIMUM_SCORE_CUT_POINTS = 10


def compute_negative_log_likelihood(class_logits: torch.Tensor, labels: torch.Tensor, output_mode: str,
                                    logit_temperature: float) -> float:
    """
    Computes the mean negative log likelihood of the labels under the probabilities of the class logits divided by a
    temperature, the measure of calibration minimised by the calibrated temperature: the unweighted cross entropy of the
    softmax, or the binary cross entropy of the sigmoid of every class in multi-label mode, without label smoothing.

    Args:
        class_logits (torch.Tensor): The [N, K] class logits.
        labels (torch.Tensor): The [N] class indices, or the [N, K] multi-hot labels in multi-label mode.
        output_mode (str): "multi_class" or "multi_label".
        logit_temperature (float): The temperature dividing the logits.

    Returns:
        float: The mean negative log likelihood.
    """
    # Computed in float64 with uniform class weights, the labels in the precision of the logits in multi-label mode
    scaled_logits = class_logits.to(dtype=torch.float64) / logit_temperature
    loss_labels = labels if output_mode == "multi_class" else labels.to(dtype=torch.float64)
    uniform_weights = torch.ones(size=(class_logits.shape[1],), dtype=torch.float64)
    return float(compute_classification_loss(class_logits=scaled_logits, labels=loss_labels, output_mode=output_mode,
                                             label_smoothing=0.0, class_weights=uniform_weights))


def find_calibrated_temperature(class_logits: torch.Tensor, labels: torch.Tensor, output_mode: str) -> float:
    """
    Finds the temperature giving the lowest negative log likelihood of the test labels (temperature scaling, Guo et
    al., 2017), the temperature at which the probabilities are calibrated, suggested by the page as the starting point
    of the choice. The negative log likelihood is convex in the inverse of the temperature, so a golden section search
    over the logarithm of the temperature finds its minimum within CALIBRATION_TEMPERATURE_RANGE.

    Args:
        class_logits (torch.Tensor): The [N, K] class logits.
        labels (torch.Tensor): The [N] class indices, or the [N, K] multi-hot labels in multi-label mode.
        output_mode (str): "multi_class" or "multi_label".

    Returns:
        float: The calibrated temperature, at a bound of the range when the minimum lies beyond it.
    """
    lower_bound, upper_bound = (math.log(bound) for bound in CALIBRATION_TEMPERATURE_RANGE)
    golden_ratio = (math.sqrt(5.0) - 1.0) / 2.0

    # Narrow the interval around the minimum, keeping the side of the lower of the two inner points
    for _ in range(NUMBER_CALIBRATION_SEARCH_STEPS):
        left_point = upper_bound - golden_ratio * (upper_bound - lower_bound)
        right_point = lower_bound + golden_ratio * (upper_bound - lower_bound)
        left_value, right_value = (compute_negative_log_likelihood(class_logits=class_logits, labels=labels,
                                                                   output_mode=output_mode,
                                                                   logit_temperature=math.exp(point))
                                   for point in [left_point, right_point])
        if left_value <= right_value:
            upper_bound = right_point
        else:
            lower_bound = left_point
    return math.exp((lower_bound + upper_bound) / 2.0)


def compute_calibration(probabilities: torch.Tensor, labels: torch.Tensor, output_mode: str) -> Dict[str, Any]:
    """
    Measures how well the probabilities match the observed frequencies, for the reliability diagram and the calibration
    error of the page: the decisions are binned by confidence, and every bin compares its mean confidence with the
    share of its decisions that are right. A decision is the most probable class of every sample, whose confidence is
    its probability, or in multi-label mode the presence or absence of every class of every sample (probability from
    0.5), whose confidence is the probability of the decided outcome, max(p, 1 - p).

    Args:
        probabilities (torch.Tensor): The [N, K] float64 class probabilities.
        labels (torch.Tensor): The [N] class indices, or the [N, K] multi-hot labels in multi-label mode.
        output_mode (str): "multi_class" or "multi_label".

    Returns:
        Dict[str, Any]: The 'calibration_counts' (decisions of every bin), 'calibration_confidences' and
            'calibration_frequencies' (None for an empty bin), the 'calibration_error' (mean absolute gap between the
            confidence and the frequency, weighted by the decisions of the bins) and the 'mean_confidence'.
    """
    # Confidence of every decision and whether it is right
    if output_mode == "multi_class":
        confidences, predicted_classes = probabilities.max(dim=1)
        outcomes = (predicted_classes == labels).to(dtype=torch.float64)
    else:
        class_probabilities = probabilities.flatten()
        confidences = torch.maximum(input=class_probabilities, other=1.0 - class_probabilities)
        outcomes = ((class_probabilities >= 0.5) == (labels >= 0.5).flatten()).to(dtype=torch.float64)

    # Sums of the confidences and outcomes of every bin, a probability of 1 falling in the last bin
    bin_indices = (confidences * NUMBER_CALIBRATION_BINS).long().clamp(max=NUMBER_CALIBRATION_BINS - 1)
    counts = torch.bincount(input=bin_indices, minlength=NUMBER_CALIBRATION_BINS)
    confidence_sums = torch.bincount(input=bin_indices, weights=confidences, minlength=NUMBER_CALIBRATION_BINS)
    outcome_sums = torch.bincount(input=bin_indices, weights=outcomes, minlength=NUMBER_CALIBRATION_BINS)
    return {"calibration_counts": counts.tolist(),
            "calibration_confidences": [float(confidence_sum) / int(count) if count else None
                                        for confidence_sum, count in zip(confidence_sums, counts)],
            "calibration_frequencies": [float(outcome_sum) / int(count) if count else None
                                        for outcome_sum, count in zip(outcome_sums, counts)],
            "calibration_error": float((confidence_sums - outcome_sums).abs().sum()) / len(confidences),
            "mean_confidence": float(confidences.mean())}


def compute_temperature_probabilities(class_logits: torch.Tensor, output_mode: str,
                                      logit_temperature: float) -> torch.Tensor:
    """
    Computes the probabilities of the class logits divided by a temperature as the exported classifier does (float32
    softmax, or sigmoid in multi-label mode, of the divided logits), then in float64, so that the histograms of the
    page and the thresholds saved from it are found on the same probabilities.

    Args:
        class_logits (torch.Tensor): The [N, K] float32 class logits.
        output_mode (str): "multi_class" or "multi_label".
        logit_temperature (float): The temperature dividing the logits.

    Returns:
        torch.Tensor: The [N, K] float64 probabilities.
    """
    return convert_logits_to_probabilities(class_logits=class_logits / logit_temperature,
                                           output_mode=output_mode).to(dtype=torch.float64)


def build_positive_masks(labels: torch.Tensor, output_mode: str, number_classes: int) -> torch.Tensor:
    """
    Tells for every sample whether it belongs to every class, the positives of the histograms and thresholds of the
    page: the one-hot labels of a single class per sample, the labels from 0.5 in multi-label mode.

    Args:
        labels (torch.Tensor): The [N] class indices, or the [N, K] multi-hot labels in multi-label mode.
        output_mode (str): "multi_class" or "multi_label".
        number_classes (int): The number of classes K.

    Returns:
        torch.Tensor: The [N, K] booleans.
    """
    if output_mode == "multi_class":
        return torch.nn.functional.one_hot(input=labels, num_classes=number_classes).to(dtype=torch.bool)
    return labels >= 0.5


def describe_temperature(class_logits: torch.Tensor, labels: torch.Tensor, positive_masks: torch.Tensor,
                         output_mode: str, logit_temperature: float, target_precisions: List[float],
                         shown_class_indices: List[int]) -> Dict[str, Any]:
    """
    Describes the probabilities of the class logits divided by one temperature of the slider: their calibration, and for
    every drawn class the histograms of its probability over its images and over the other images, with its threshold
    and recall at every target precision, found by the function writing class_thresholds.json from probabilities
    computed the same way as the export (compute_temperature_probabilities).

    Args:
        class_logits (torch.Tensor): The [N, K] float32 class logits.
        labels (torch.Tensor): The [N] class indices, or the [N, K] multi-hot labels in multi-label mode.
        positive_masks (torch.Tensor): The [N, K] booleans telling whether every sample belongs to every class.
        output_mode (str): "multi_class" or "multi_label".
        logit_temperature (float): The temperature dividing the logits.
        target_precisions (List[float]): The precisions targeted by the class thresholds.
        shown_class_indices (List[int]): The indices of the drawn classes, in their order on the page.

    Returns:
        Dict[str, Any]: The 'temperature', its 'negative_log_likelihood', the calibration of compute_calibration, and
            the 'classes' list holding for every drawn class its 'positive_counts' and 'negative_counts' per score bin
            and its [threshold, recall] at every target precision ('thresholds', None when not reached).
    """
    probabilities = compute_temperature_probabilities(class_logits=class_logits, output_mode=output_mode,
                                                      logit_temperature=logit_temperature)
    shown_probabilities = probabilities[:, shown_class_indices]
    shown_masks = positive_masks[:, shown_class_indices]

    # Threshold and recall of every drawn class at every target precision
    target_thresholds = [find_class_thresholds(probabilities=shown_probabilities, positive_masks=shown_masks,
                                               target_precision=target_precision)
                         for target_precision in target_precisions]

    # Histograms of the probability of every drawn class over its images and over the other images
    bin_indices = (shown_probabilities * NUMBER_SCORE_BINS).long().clamp(max=NUMBER_SCORE_BINS - 1)
    classes = []
    for column_index in range(len(shown_class_indices)):
        class_mask = shown_masks[:, column_index]
        class_bins = bin_indices[:, column_index]
        classes.append({
            "positive_counts": torch.bincount(input=class_bins[class_mask], minlength=NUMBER_SCORE_BINS).tolist(),
            "negative_counts": torch.bincount(input=class_bins[~class_mask], minlength=NUMBER_SCORE_BINS).tolist(),
            "thresholds": [None if thresholds[column_index] is None else list(thresholds[column_index])
                           for thresholds in target_thresholds]})

    return {"temperature": logit_temperature,
            "negative_log_likelihood": compute_negative_log_likelihood(class_logits=class_logits, labels=labels,
                                                                       output_mode=output_mode,
                                                                       logit_temperature=logit_temperature),
            **compute_calibration(probabilities=probabilities, labels=labels, output_mode=output_mode),
            "classes": classes}


def build_explorer_data(class_logits: torch.Tensor, labels: torch.Tensor, class_names: List[str], output_mode: str,
                        target_precisions: List[float], class_f1_scores: List[float],
                        title: Optional[str]) -> Dict[str, Any]:
    """
    Builds the data of the temperature explorer page of an evaluated checkpoint: the description of every temperature
    of the slider (the grid from 0.5 to 10 and the calibrated temperature), with the drawn classes, the lowest F1 first
    like the per-class table of the report, at most MAXIMUM_EXPLORER_CLASSES of them, the save button needing all of
    them. Written into the page by the evaluation, and built again from 'class_logits.npz' by the serve script.

    Args:
        class_logits (torch.Tensor): The [N, K] float32 class logits of the evaluated test samples.
        labels (torch.Tensor): The [N] class indices, or the [N, K] multi-hot labels in multi-label mode.
        class_names (List[str]): The class names, in class index order.
        output_mode (str): "multi_class" or "multi_label".
        target_precisions (List[float]): The precisions targeted by the class thresholds, the main one first.
        class_f1_scores (List[float]): The F1 of every class, in class index order, ordering the drawn classes.
        title (Optional[str]): The title of the evaluation shown by the page, e.g. 'Iteration 800', or None.

    Returns:
        Dict[str, Any]: The data of the page.

    Raises:
        ValueError: If no test sample was evaluated.
    """
    if len(class_logits) == 0:
        raise ValueError("The temperature explorer needs at least one evaluated test sample.")

    # Classes of every sample, and the drawn classes, the lowest F1 first like the per-class table of the report
    number_classes = class_logits.shape[1]
    positive_masks = build_positive_masks(labels=labels, output_mode=output_mode, number_classes=number_classes)
    shown_class_indices = sorted(range(number_classes),
                                 key=lambda class_index: class_f1_scores[class_index])[:MAXIMUM_EXPLORER_CLASSES]

    # Temperatures of the slider: the grid and the calibrated temperature with 3 significant digits
    calibrated_temperature = find_calibrated_temperature(class_logits=class_logits, labels=labels,
                                                         output_mode=output_mode)
    calibrated_temperature = float(f"{calibrated_temperature:.3g}")
    temperatures = sorted(set(TEMPERATURE_GRID + [calibrated_temperature]))

    # Data of the page, the description of every temperature of the slider, with the class index of every drawn class
    # and the file of its save button, offered when every class is drawn
    return {"title": title, "output_mode": output_mode, "number_samples": len(class_logits),
            "number_classes": number_classes, "class_indices": shown_class_indices,
            "class_names": [class_names[class_index] for class_index in shown_class_indices],
            "positives": [int(positive_masks[:, class_index].sum()) for class_index in shown_class_indices],
            "all_classes_shown": len(shown_class_indices) == number_classes,
            "saved_file_name": SAVED_THRESHOLDS_FILE_NAME, "target_precisions": target_precisions,
            "calibrated_temperature": calibrated_temperature, "number_score_bins": NUMBER_SCORE_BINS,
            "number_calibration_bins": NUMBER_CALIBRATION_BINS,
            "states": [describe_temperature(class_logits=class_logits, labels=labels, positive_masks=positive_masks,
                                            output_mode=output_mode, logit_temperature=temperature,
                                            target_precisions=target_precisions,
                                            shown_class_indices=shown_class_indices)
                       for temperature in temperatures]}


def render_explorer_page(explorer_data: Dict[str, Any]) -> str:
    """
    Fills the current template of the page with the data of an evaluation, for the page written by the evaluation and
    for the page served again by the serve script.

    Args:
        explorer_data (Dict[str, Any]): The data of the page (build_explorer_data), with the fields of the serve script
            on a served page.

    Returns:
        str: The HTML of the page.
    """
    # The data inserted into the page as JSON, a class name holding '</' being unable to close its script element
    explorer_json = json.dumps(obj=explorer_data).replace("</", "<\\/")
    return TEMPLATE_PATH.read_text().replace(DATA_PLACEHOLDER, explorer_json)


def save_temperature_explorer(class_logits: torch.Tensor, labels: torch.Tensor, class_names: List[str],
                              output_mode: str, target_precisions: List[float], class_f1_scores: List[float],
                              output_folder: Path, title: Optional[str]) -> Path:
    """
    Writes 'temperature_explorer.html' in the evaluation folder of a checkpoint: a self-contained page whose slider
    divides the class logits of the test split by temperatures from 0.5 to 10 (with the calibrated temperature),
    showing for every temperature the calibration of the probabilities and, for every class, the distribution of its
    probability over its images and the other images with its thresholds and recalls at the target precisions
    (build_explorer_data). Opened as a plain file, its save button downloads the thresholds of the temperature shown;
    the serve script builds it again with any temperature from 'class_logits.npz'.

    Args:
        class_logits (torch.Tensor): The [N, K] float32 class logits of the evaluated test samples.
        labels (torch.Tensor): The [N] class indices, or the [N, K] multi-hot labels in multi-label mode.
        class_names (List[str]): The class names, in class index order.
        output_mode (str): "multi_class" or "multi_label".
        target_precisions (List[float]): The precisions targeted by the class thresholds, the main one first.
        class_f1_scores (List[float]): The F1 of every class, in class index order, ordering the drawn classes.
        output_folder (Path): The evaluation folder of the checkpoint.
        title (Optional[str]): The title of the evaluation shown by the page, e.g. 'Iteration 800', or None.

    Returns:
        Path: The written page.

    Raises:
        ValueError: If no test sample was evaluated.
    """
    explorer_data = build_explorer_data(class_logits=class_logits, labels=labels, class_names=class_names,
                                        output_mode=output_mode, target_precisions=target_precisions,
                                        class_f1_scores=class_f1_scores, title=title)
    explorer_path = output_folder / EXPLORER_FILE_NAME
    explorer_path.write_text(data=render_explorer_page(explorer_data=explorer_data))
    return explorer_path


def save_class_logits(class_logits: torch.Tensor, labels: torch.Tensor, class_names: List[str], output_mode: str,
                      target_precisions: List[float], class_f1_scores: List[float], output_folder: Path,
                      title: Optional[str]) -> Path:
    """
    Writes 'class_logits.npz' next to the temperature explorer of an evaluated checkpoint: the logits of the test
    samples with everything the page is built from (load_class_logits), so that the serve script builds the page again
    with the template of the day, describes any temperature and finds the thresholds of every class at the chosen one,
    without evaluating the checkpoint again. About 0.3 MB for 40,000 images of 2 classes; the classification trainer
    writes it up to MAXIMUM_EXPLORER_CLASSES classes.

    Args:
        class_logits (torch.Tensor): The [N, K] float32 class logits of the evaluated test samples.
        labels (torch.Tensor): The [N] class indices, or the [N, K] multi-hot labels in multi-label mode.
        class_names (List[str]): The class names, in class index order.
        output_mode (str): "multi_class" or "multi_label".
        target_precisions (List[float]): The precisions targeted by the class thresholds, the main one first.
        class_f1_scores (List[float]): The F1 of every class, in class index order, ordering the drawn classes.
        output_folder (Path): The evaluation folder of the checkpoint.
        title (Optional[str]): The title of the evaluation shown by the page, e.g. 'Iteration 800', or None.

    Returns:
        Path: The written file.

    Raises:
        ValueError: If no test sample was evaluated, or the logits do not hold one column per class.
    """
    if len(class_logits) == 0 or class_logits.shape[1] != len(class_names):
        raise ValueError(f"Expected the logits of at least one sample and {len(class_names)} classes, got "
                         f"{tuple(class_logits.shape)}.")

    # Plain arrays only (strings as unicode arrays), so that the file is read back without pickle, a title of None
    # being written as an empty string
    logits_path = output_folder / CLASS_LOGITS_FILE_NAME
    np.savez(file=logits_path, class_logits=class_logits.to(dtype=torch.float32).numpy(),
             labels=labels.numpy(), class_names=np.array(class_names, dtype=str), output_mode=np.array(output_mode),
             target_precisions=np.array(target_precisions, dtype=np.float64),
             class_f1_scores=np.array(class_f1_scores, dtype=np.float64), title=np.array(title or ""))
    return logits_path


def load_class_logits(evaluation_folder: Path) -> Dict[str, Any]:
    """
    Reads the 'class_logits.npz' of an evaluated checkpoint, written by save_class_logits, for the page and the
    thresholds the serve script builds from it.

    Args:
        evaluation_folder (Path): The evaluation folder of the checkpoint.

    Returns:
        Dict[str, Any]: The 'class_logits' ([N, K] float32 tensor), the 'labels' tensor, the 'class_names',
            'output_mode', 'target_precisions', 'class_f1_scores' and 'title' (None when written empty).

    Raises:
        FileNotFoundError: If the folder holds no class logits.
        ValueError: If the file misses one of its arrays.
    """
    logits_path = evaluation_folder / CLASS_LOGITS_FILE_NAME
    if not logits_path.is_file():
        raise FileNotFoundError(f"No {CLASS_LOGITS_FILE_NAME} in {evaluation_folder}: it is written by the "
                                f"classification evaluations with evaluation.per_class_report, up to "
                                f"{MAXIMUM_EXPLORER_CLASSES} classes.")
    with np.load(file=logits_path, allow_pickle=False) as arrays:
        expected_arrays = {"class_logits", "labels", "class_names", "output_mode", "target_precisions",
                           "class_f1_scores", "title"}
        if set(arrays.files) != expected_arrays:
            raise ValueError(f"Expected the arrays {sorted(expected_arrays)} in {logits_path}, got "
                             f"{sorted(arrays.files)}.")
        title = str(arrays["title"])
        return {"class_logits": torch.from_numpy(arrays["class_logits"]),
                "labels": torch.from_numpy(arrays["labels"]),
                "class_names": [str(class_name) for class_name in arrays["class_names"]],
                "output_mode": str(arrays["output_mode"]),
                "target_precisions": [float(target_precision) for target_precision in arrays["target_precisions"]],
                "class_f1_scores": [float(class_f1_score) for class_f1_score in arrays["class_f1_scores"]],
                "title": title if title else None}


def build_explorer_data_from_logits(class_logits_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the data of the page of an evaluated checkpoint from its 'class_logits.npz', as the evaluation built it,
    for the page served by the serve script.

    Args:
        class_logits_record (Dict[str, Any]): The class logits of the evaluation (load_class_logits).

    Returns:
        Dict[str, Any]: The data of the page.
    """
    return build_explorer_data(class_logits=class_logits_record["class_logits"],
                               labels=class_logits_record["labels"], class_names=class_logits_record["class_names"],
                               output_mode=class_logits_record["output_mode"],
                               target_precisions=class_logits_record["target_precisions"],
                               class_f1_scores=class_logits_record["class_f1_scores"],
                               title=class_logits_record["title"])


def describe_explorer_temperature(class_logits_record: Dict[str, Any], shown_class_indices: List[int],
                                  logit_temperature: float) -> Dict[str, Any]:
    """
    Describes one more temperature of the page of an evaluated checkpoint from its class logits, a temperature typed
    on the served page, in the format of the temperatures of its slider.

    Args:
        class_logits_record (Dict[str, Any]): The class logits of the evaluation (load_class_logits).
        shown_class_indices (List[int]): The indices of the classes drawn by the page, in their order on the page.
        logit_temperature (float): The temperature dividing the logits.

    Returns:
        Dict[str, Any]: The description of the temperature (describe_temperature).
    """
    class_logits = class_logits_record["class_logits"]
    positive_masks = build_positive_masks(labels=class_logits_record["labels"],
                                          output_mode=class_logits_record["output_mode"],
                                          number_classes=class_logits.shape[1])
    return describe_temperature(class_logits=class_logits, labels=class_logits_record["labels"],
                                positive_masks=positive_masks, output_mode=class_logits_record["output_mode"],
                                logit_temperature=logit_temperature,
                                target_precisions=class_logits_record["target_precisions"],
                                shown_class_indices=shown_class_indices)


def compute_temperature_thresholds(class_logits_record: Dict[str, Any], logit_temperature: float) -> Dict[str, Any]:
    """
    Finds the threshold of every class at every target precision of an evaluated checkpoint at a temperature, from its
    class logits, in the format of class_thresholds.json: the thresholds the serve script saves for the export, found
    on the probabilities of the page (compute_temperature_probabilities), so that the saved file holds exactly the
    thresholds the page shows at that temperature, for every class.

    Args:
        class_logits_record (Dict[str, Any]): The class logits of the evaluation (load_class_logits).
        logit_temperature (float): The temperature dividing the logits.

    Returns:
        Dict[str, Any]: The thresholds of every class at the temperature.
    """
    class_logits = class_logits_record["class_logits"]
    probabilities = compute_temperature_probabilities(class_logits=class_logits,
                                                      output_mode=class_logits_record["output_mode"],
                                                      logit_temperature=logit_temperature)
    positive_masks = build_positive_masks(labels=class_logits_record["labels"],
                                          output_mode=class_logits_record["output_mode"],
                                          number_classes=class_logits.shape[1])
    target_thresholds = [find_class_thresholds(probabilities=probabilities, positive_masks=positive_masks,
                                               target_precision=target_precision)
                         for target_precision in class_logits_record["target_precisions"]]
    return build_class_thresholds_file(class_names=class_logits_record["class_names"],
                                       target_precisions=class_logits_record["target_precisions"],
                                       target_thresholds=target_thresholds, logit_temperature=logit_temperature)


def check_score_cut_points(cut_points: Any) -> List[float]:
    """
    Checks the cut points typed on the served page for its score ranges table, the scores splitting [0, 1] into the
    ranges the images of every class are counted in (0.5 and 0.8 giving below 0.5, from 0.5 to 0.8, and from 0.8):
    one to MAXIMUM_SCORE_CUT_POINTS distinct finite numbers strictly between 0 and 1, in any order.

    Args:
        cut_points (Any): The cut points, read from the address of the request.

    Returns:
        List[float]: The cut points in increasing order.

    Raises:
        ValueError: If the cut points are not 1 to MAXIMUM_SCORE_CUT_POINTS distinct numbers strictly between 0 and 1.
    """
    if (not isinstance(cut_points, list) or not 1 <= len(cut_points) <= MAXIMUM_SCORE_CUT_POINTS or
            not all(is_json_number(value=cut_point) and 0 < cut_point < 1 for cut_point in cut_points) or
            len(set(cut_points)) != len(cut_points)):
        raise ValueError(f"Expected 1 to {MAXIMUM_SCORE_CUT_POINTS} distinct scores strictly between 0 and 1 as cut "
                         f"points, got {cut_points!r}.")
    return sorted(float(cut_point) for cut_point in cut_points)


def compute_score_ranges(class_logits_record: Dict[str, Any], shown_class_indices: List[int], logit_temperature: float,
                         cut_points: List[float]) -> Dict[str, Any]:
    """
    Counts, at a temperature, the images of every drawn class by the range their score for their own class falls in,
    for the score ranges table of the served page: with the cut points 0.5 and 0.8, the images of a class scoring below
    0.5, from 0.5 to below 0.8, and from 0.8, a score equal to a cut point counting in the range above it, as an image
    is given a class from its threshold. The scores are those of the thresholds and of the export
    (compute_temperature_probabilities); in multi-label mode, the images of a class are those labelled with it.

    Args:
        class_logits_record (Dict[str, Any]): The class logits of the evaluation (load_class_logits).
        shown_class_indices (List[int]): The indices of the classes drawn by the page, in their order on the page.
        logit_temperature (float): The temperature dividing the logits.
        cut_points (List[float]): The cut points in increasing order (check_score_cut_points).

    Returns:
        Dict[str, Any]: The 'temperature', the 'cut_points' and the 'counts' of every drawn class, in the order of the
            page, one count per range (len(cut_points) + 1), adding up to the images of the class.
    """
    class_logits = class_logits_record["class_logits"]
    probabilities = compute_temperature_probabilities(class_logits=class_logits,
                                                      output_mode=class_logits_record["output_mode"],
                                                      logit_temperature=logit_temperature)
    positive_masks = build_positive_masks(labels=class_logits_record["labels"],
                                          output_mode=class_logits_record["output_mode"],
                                          number_classes=class_logits.shape[1])

    # Range of every score: the number of cut points at or below it, so that a score equal to a cut point is counted
    # in the range above it
    boundaries = torch.tensor(data=cut_points, dtype=torch.float64)
    range_indices = torch.bucketize(input=probabilities, boundaries=boundaries, right=True)

    # Images of every drawn class, counted by the range of their own class score
    counts = [torch.bincount(input=range_indices[positive_masks[:, class_index], class_index],
                             minlength=len(cut_points) + 1).tolist()
              for class_index in shown_class_indices]
    return {"temperature": logit_temperature, "cut_points": cut_points, "counts": counts}


def read_explorer_page_data(explorer_path: Path) -> Dict[str, Any]:
    """
    Reads the data written into a temperature explorer page, for the evaluations without class logits (written before
    them, or of more than MAXIMUM_EXPLORER_CLASSES classes), whose page the serve script fills again with the current
    template, its temperatures being those of the written page.

    Args:
        explorer_path (Path): The written page.

    Returns:
        Dict[str, Any]: The data of the page.

    Raises:
        ValueError: If the page holds no data, or was written before the save button (no class indices).
    """
    match = EXPLORER_DATA_PATTERN.search(string=explorer_path.read_text())
    if match is None:
        raise ValueError(f"No explorer data in {explorer_path}.")
    explorer_data = json.loads(s=match.group(1).replace("<\\/", "</"))
    if "class_indices" not in explorer_data or "all_classes_shown" not in explorer_data:
        raise ValueError(f"{explorer_path} was written before the save button of the page: evaluate its checkpoint "
                         f"again with scripts/evaluation/evaluate_checkpoint.py.")
    return explorer_data


def check_logit_temperature(logit_temperature: Any) -> float:
    """
    Checks a temperature typed on the served page or posted to the serve script: a finite number within
    CALIBRATION_TEMPERATURE_RANGE, the range of the calibrated temperature.

    Args:
        logit_temperature (Any): The temperature, read from JSON or from the address of the request.

    Returns:
        float: The temperature.

    Raises:
        ValueError: If the temperature is not a number within the range.
    """
    lower_bound, upper_bound = CALIBRATION_TEMPERATURE_RANGE
    if not is_json_number(value=logit_temperature) or not lower_bound <= logit_temperature <= upper_bound:
        raise ValueError(f"Expected a logit temperature from {lower_bound:g} to {upper_bound:g}, got "
                         f"{logit_temperature!r}.")
    return float(logit_temperature)


def is_json_number(value: Any) -> bool:
    """
    Tells whether a value read from a thresholds file is a finite number, a boolean (a subclass of int in Python) and
    NaN being refused, for the checks of the saved thresholds.

    Args:
        value (Any): The value read from the file.

    Returns:
        bool: True for a finite number.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def list_target_thresholds(thresholds: Dict[str, Any]) -> List[Tuple[Any, Any]]:
    """
    Lists the target precisions of a thresholds file with the thresholds of their classes, the main target first, then
    the 'additional_targets' of the evaluations of several target precisions, so that the saved thresholds are compared
    with class_thresholds.json target by target.

    Args:
        thresholds (Dict[str, Any]): The thresholds file, in the format of class_thresholds.json.

    Returns:
        List[Tuple[Any, Any]]: The target precision and the 'classes' of every target, in the order of the file.
    """
    additional_targets = thresholds["additional_targets"] if "additional_targets" in thresholds else []
    return [(thresholds["target_precision"], thresholds["classes"])] + [
        (target["target_precision"], target["classes"]) for target in additional_targets]


def check_saved_thresholds(saved_thresholds: Any, evaluation_folder: Path) -> None:
    """
    Checks the thresholds saved from the temperature explorer against the class_thresholds.json of the evaluation
    folder they are saved to or exported from, so that the export never deploys the thresholds of another evaluation or
    an invalid temperature: the fields of class_thresholds.json, a temperature above 0, the same target precisions in
    the same order, the same classes at every target, and every threshold and recall within [0, 1] (null for a class
    not reaching the target precision).

    Args:
        saved_thresholds (Any): The saved thresholds, read from JSON.
        evaluation_folder (Path): The evaluation folder of the checkpoint, holding class_thresholds.json.

    Raises:
        FileNotFoundError: If the folder holds no class_thresholds.json.
        ValueError: If the saved thresholds are not those of the evaluation of the folder at a temperature above 0.
    """
    class_thresholds_path = evaluation_folder / "class_thresholds.json"
    if not class_thresholds_path.is_file():
        raise FileNotFoundError(f"No class_thresholds.json in {evaluation_folder} to check the saved thresholds "
                                f"against.")
    class_thresholds = json.loads(s=class_thresholds_path.read_text())

    # Fields of the thresholds of the evaluation, with the temperature, and the additional targets well formed
    expected_fields = {"logit_temperature", "target_precision", "classes"} | (
        {"additional_targets"} if "additional_targets" in class_thresholds else set())
    if not isinstance(saved_thresholds, dict) or set(saved_thresholds) != expected_fields:
        found_fields = sorted(saved_thresholds) if isinstance(saved_thresholds, dict) else type(saved_thresholds)
        raise ValueError(f"Expected the saved thresholds to hold the fields {sorted(expected_fields)}, got "
                         f"{found_fields}.")
    temperature = saved_thresholds["logit_temperature"]
    if not is_json_number(value=temperature) or temperature <= 0:
        raise ValueError(f"Expected a logit temperature above 0, got {temperature!r}.")
    if "additional_targets" in saved_thresholds and not (
            isinstance(saved_thresholds["additional_targets"], list) and
            all(isinstance(target, dict) and set(target) == {"target_precision", "classes"}
                for target in saved_thresholds["additional_targets"])):
        raise ValueError("Expected 'additional_targets' to list objects of a 'target_precision' and its 'classes'.")

    # Target precisions of the evaluation, in its order
    saved_targets = list_target_thresholds(thresholds=saved_thresholds)
    expected_targets = list_target_thresholds(thresholds=class_thresholds)
    saved_precisions = [target_precision for target_precision, _ in saved_targets]
    expected_precisions = [target_precision for target_precision, _ in expected_targets]
    if saved_precisions != expected_precisions:
        raise ValueError(f"Expected the target precisions {expected_precisions} of {class_thresholds_path}, got "
                         f"{saved_precisions}.")

    # Classes of the evaluation at every target, each with a threshold and a recall within [0, 1], or null
    for (target_precision, saved_classes), (_, expected_classes) in zip(saved_targets, expected_targets):
        if not isinstance(saved_classes, dict) or set(saved_classes) != set(expected_classes):
            found_classes = sorted(saved_classes) if isinstance(saved_classes, dict) else type(saved_classes)
            raise ValueError(f"Expected the class names {sorted(expected_classes)} at the target precision "
                             f"{target_precision}, got {found_classes}.")
        for class_name, class_threshold in saved_classes.items():
            if class_threshold is not None and not (
                    isinstance(class_threshold, dict) and set(class_threshold) == {"threshold", "recall"} and
                    all(is_json_number(value=value) and 0.0 <= value <= 1.0 for value in class_threshold.values())):
                raise ValueError(f"Expected the threshold and recall of {class_name} at the target precision "
                                 f"{target_precision} within [0, 1], or null, got {class_threshold!r}.")


def write_saved_thresholds(evaluation_folder: Path, saved_thresholds: Any) -> Path:
    """
    Writes the thresholds posted by the save button of the temperature explorer next to the page, as
    'saved_temperature_thresholds.json', after checking them against the class_thresholds.json of the folder, so that
    the export of the checkpoint finds the temperature chosen on the page with its thresholds. An earlier save of the
    folder is replaced.

    Args:
        evaluation_folder (Path): The evaluation folder of the checkpoint, holding the page and class_thresholds.json.
        saved_thresholds (Any): The posted thresholds, read from JSON.

    Returns:
        Path: The written file.

    Raises:
        FileNotFoundError: If the folder holds no class_thresholds.json.
        ValueError: If the thresholds are not those of the evaluation of the folder at a temperature above 0.
    """
    check_saved_thresholds(saved_thresholds=saved_thresholds, evaluation_folder=evaluation_folder)
    saved_path = evaluation_folder / SAVED_THRESHOLDS_FILE_NAME
    saved_path.write_text(data=json.dumps(obj=saved_thresholds, indent=4))
    return saved_path
