"""
Search of the per-class score thresholds reaching a target precision ('ExperimentConfiguration.evaluation'), for
deployments that only accept a prediction above a per-class threshold. Used by the classification trainer at every
evaluation, by the frame selection metrics, and by the mean average precision of the detection tasks for the
detections of every class. build_class_thresholds_file gives the 'class_thresholds.json' of the classification, written
by the trainer at a temperature of 1 and saved from the temperature explorer at the chosen temperature.
"""

import torch

from typing import Any, Dict, List, Optional, Tuple


def find_score_threshold(scores: torch.Tensor, positives: torch.Tensor, number_positives: int,
                         target_precision: float) -> Optional[Tuple[float, float]]:
    """
    Finds the score threshold of one class reaching a target precision: the predictions whose score reaches the
    threshold are accepted, and the lowest such threshold is kept, i.e. the one keeping the highest recall at the
    target precision. The recall is counted over the true members of the class, some of which may never be predicted
    (the objects found by no detection).

    Args:
        scores (torch.Tensor): The [N] scores of the predictions of the class.
        positives (torch.Tensor): The [N] booleans telling whether every prediction is right.
        number_positives (int): The number of true members of the class, the denominator of the recall.
        target_precision (float): The precision the threshold must reach, e.g. 0.99.

    Returns:
        Optional[Tuple[float, float]]: The threshold and the recall it keeps, or None if no threshold reaches the
            target precision (or the class has no member or no prediction).
    """
    if number_positives == 0 or len(scores) == 0:
        return None

    # Precision of every cut of the scores sorted by decreasing value
    order = torch.argsort(input=scores, descending=True, stable=True)
    sorted_scores = scores[order]
    true_positives = torch.cumsum(input=positives[order].to(dtype=torch.float64), dim=0)
    precision = true_positives / torch.arange(start=1, end=len(scores) + 1, dtype=torch.float64)

    # A threshold can only cut between two different scores, so only the last position of every score counts
    is_cut = torch.cat(tensors=[sorted_scores[1:] != sorted_scores[:-1], torch.tensor(data=[True])])
    valid_cuts = torch.nonzero(input=is_cut & (precision >= target_precision) & (true_positives > 0)).flatten()
    if len(valid_cuts) == 0:
        return None

    # The deepest valid cut keeps the most positives at the target precision
    last_valid_cut = int(valid_cuts[-1])
    return float(sorted_scores[last_valid_cut]), float(true_positives[last_valid_cut]) / number_positives


def find_class_thresholds(probabilities: torch.Tensor, positive_masks: torch.Tensor,
                          target_precision: float) -> List[Optional[Tuple[float, float]]]:
    """
    Finds, for every class, the score threshold reaching a target precision, for deployments that only accept a
    prediction above a per-class threshold. Every class is treated one-versus-rest: the samples whose class
    probability reaches the threshold are predicted as the class (find_score_threshold).

    Args:
        probabilities (torch.Tensor): The [N, K] class probabilities.
        positive_masks (torch.Tensor): The [N, K] booleans telling whether every sample belongs to every class.
        target_precision (float): The precision the threshold must reach, e.g. 0.99.

    Returns:
        List[Optional[Tuple[float, float]]]: For every class, its threshold and the recall it keeps, or None if no
            threshold reaches the target precision (or the class has no sample).
    """
    return [find_score_threshold(scores=probabilities[:, class_index], positives=positive_masks[:, class_index],
                                 number_positives=int(positive_masks[:, class_index].sum()),
                                 target_precision=target_precision)
            for class_index in range(probabilities.shape[1])]


def build_class_thresholds_file(class_names: List[str], target_precisions: List[float],
                                target_thresholds: List[List[Optional[Tuple[float, float]]]],
                                logit_temperature: float) -> Dict[str, Any]:
    """
    Builds the content of 'class_thresholds.json', read by the deployment of a classifier: the logit temperature of
    the probabilities the thresholds were found on, the main target precision in 'target_precision' and 'classes', the
    others in 'additional_targets', every class keyed by name with its 'threshold' and the 'recall' it keeps, or None
    when it never reaches the target precision. The classification trainer writes it at a temperature of 1, and the
    temperature explorer saves it at the chosen temperature as 'saved_temperature_thresholds.json'.

    Args:
        class_names (List[str]): The class names, in class index order.
        target_precisions (List[float]): The target precisions, the main one first.
        target_thresholds (List[List[Optional[Tuple[float, float]]]]): For every target precision, the threshold and
            recall of every class (find_class_thresholds), None for a class never reaching it.
        logit_temperature (float): The temperature dividing the logits of the probabilities.

    Returns:
        Dict[str, Any]: The content of the thresholds file.
    """
    # Threshold and recall of every class keyed by name, for every target precision
    target_descriptions = [{"target_precision": target_precision,
                            "classes": {class_name: None if class_threshold is None else
                                        {"threshold": class_threshold[0], "recall": class_threshold[1]}
                                        for class_name, class_threshold in zip(class_names, class_thresholds)}}
                           for target_precision, class_thresholds in zip(target_precisions, target_thresholds)]

    # Main target first, the others after it
    thresholds_file = {"logit_temperature": logit_temperature, **target_descriptions[0]}
    if len(target_descriptions) > 1:
        thresholds_file["additional_targets"] = target_descriptions[1:]
    return thresholds_file
