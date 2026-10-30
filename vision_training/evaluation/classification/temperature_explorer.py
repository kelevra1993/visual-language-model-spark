"""
Temperature explorer of the classification evaluation: the class logits of the test split are divided by every
temperature of a grid, giving for each the distribution of the probability of every class over its images and over the
other images, the thresholds and recalls of the target precisions, and the calibration of the probabilities (negative
log likelihood, calibration error, reliability), written as the self-contained page 'temperature_explorer.html' in the
evaluation folder of the checkpoint. Written by the classification trainer with the per-class report, to choose the
'logit_temperature' of scripts/evaluation/evaluate_checkpoint.py, which class_thresholds.json records and
vision_training/deployment/export_model.py builds into the exported classifier.
"""

import json
import math
import torch

from pathlib import Path
from typing import Any, Dict, List, Optional

from vision_training.evaluation.classification.threshold_finder import find_class_thresholds
from vision_training.losses.classification_losses import compute_classification_loss, convert_logits_to_probabilities

# Temperatures of the slider of the page, round values to choose a temperature from, the evaluated temperature and the
# calibrated one being added to them
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


def describe_temperature(class_logits: torch.Tensor, labels: torch.Tensor, positive_masks: torch.Tensor,
                         output_mode: str, logit_temperature: float, target_precisions: List[float],
                         shown_class_indices: List[int]) -> Dict[str, Any]:
    """
    Describes the probabilities of the class logits divided by one temperature of the slider: their calibration, and for
    every drawn class the histograms of its probability over its images and over the other images, with its threshold
    and recall at every target precision, found by the function writing class_thresholds.json from probabilities
    computed the same way as the evaluation (float32 softmax of the divided logits, then float64).

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
    probabilities = convert_logits_to_probabilities(class_logits=class_logits / logit_temperature,
                                                    output_mode=output_mode).to(dtype=torch.float64)
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


def save_temperature_explorer(class_logits: torch.Tensor, labels: torch.Tensor, class_names: List[str],
                              output_mode: str, target_precisions: List[float], evaluated_temperature: float,
                              class_f1_scores: List[float], output_folder: Path, title: Optional[str]) -> Path:
    """
    Writes 'temperature_explorer.html' in the evaluation folder of a checkpoint: a self-contained page whose slider
    divides the class logits of the test split by temperatures from 0.5 to 10 (with the evaluated and the calibrated
    temperatures), showing for every temperature the calibration of the probabilities and, for every class, the
    distribution of its probability over its images and the other images with its thresholds and recalls at the target
    precisions. The logits are computed once by the evaluation, every temperature being computed here, so that the
    temperature of the export can be chosen without evaluating again. The classes are drawn the lowest F1 first, at
    most MAXIMUM_EXPLORER_CLASSES of them.

    Args:
        class_logits (torch.Tensor): The [N, K] float32 class logits of the evaluated test samples.
        labels (torch.Tensor): The [N] class indices, or the [N, K] multi-hot labels in multi-label mode.
        class_names (List[str]): The class names, in class index order.
        output_mode (str): "multi_class" or "multi_label".
        target_precisions (List[float]): The precisions targeted by the class thresholds, the main one first.
        evaluated_temperature (float): The temperature of the probabilities of the evaluation and of
            class_thresholds.json.
        class_f1_scores (List[float]): The F1 of every class, in class index order, ordering the drawn classes.
        output_folder (Path): The evaluation folder of the checkpoint.
        title (Optional[str]): The title of the evaluation shown by the page, e.g. 'Iteration 800', or None.

    Returns:
        Path: The written page.

    Raises:
        ValueError: If no test sample was evaluated.
    """
    if len(class_logits) == 0:
        raise ValueError("The temperature explorer needs at least one evaluated test sample.")

    # Classes of every sample, and the drawn classes, the lowest F1 first like the per-class table of the report
    number_classes = class_logits.shape[1]
    if output_mode == "multi_class":
        positive_masks = torch.nn.functional.one_hot(input=labels, num_classes=number_classes).to(dtype=torch.bool)
    else:
        positive_masks = labels >= 0.5
    shown_class_indices = sorted(range(number_classes),
                                 key=lambda class_index: class_f1_scores[class_index])[:MAXIMUM_EXPLORER_CLASSES]

    # Temperatures of the slider: the grid, the evaluated temperature and the calibrated one with 3 significant digits
    calibrated_temperature = find_calibrated_temperature(class_logits=class_logits, labels=labels,
                                                         output_mode=output_mode)
    calibrated_temperature = float(f"{calibrated_temperature:.3g}")
    temperatures = sorted(set(TEMPERATURE_GRID + [float(evaluated_temperature), calibrated_temperature]))

    # Data of the page, the description of every temperature of the slider
    explorer_data = {"title": title, "output_mode": output_mode, "number_samples": len(class_logits),
                     "number_classes": number_classes,
                     "class_names": [class_names[class_index] for class_index in shown_class_indices],
                     "positives": [int(positive_masks[:, class_index].sum()) for class_index in shown_class_indices],
                     "target_precisions": target_precisions, "evaluated_temperature": float(evaluated_temperature),
                     "calibrated_temperature": calibrated_temperature, "number_score_bins": NUMBER_SCORE_BINS,
                     "number_calibration_bins": NUMBER_CALIBRATION_BINS,
                     "states": [describe_temperature(class_logits=class_logits, labels=labels,
                                                     positive_masks=positive_masks, output_mode=output_mode,
                                                     logit_temperature=temperature,
                                                     target_precisions=target_precisions,
                                                     shown_class_indices=shown_class_indices)
                                for temperature in temperatures]}

    # The data inserted into the page as JSON, a class name holding '</' being unable to close its script element
    explorer_json = json.dumps(obj=explorer_data).replace("</", "<\\/")
    explorer_path = output_folder / EXPLORER_FILE_NAME
    explorer_path.write_text(data=TEMPLATE_PATH.read_text().replace(DATA_PLACEHOLDER, explorer_json))
    return explorer_path
