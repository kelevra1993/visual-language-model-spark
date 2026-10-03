"""
Search of the per-class score thresholds reaching a target precision ('ExperimentConfiguration.evaluation'), for
deployments that only accept a prediction above a per-class threshold. Used by the classification trainer at every
evaluation.
"""

import torch

from typing import List, Optional, Tuple


def find_class_thresholds(probabilities: torch.Tensor, positive_masks: torch.Tensor,
                          target_precision: float) -> List[Optional[Tuple[float, float]]]:
    """
    Finds, for every class, the score threshold reaching a target precision, for deployments that only accept a
    prediction above a per-class threshold. Every class is treated one-versus-rest: the samples whose class
    probability reaches the threshold are predicted as the class. The lowest such threshold is kept, i.e. the one
    keeping the highest recall at the target precision.

    Args:
        probabilities (torch.Tensor): The [N, K] class probabilities.
        positive_masks (torch.Tensor): The [N, K] booleans telling whether every sample belongs to every class.
        target_precision (float): The precision the threshold must reach, e.g. 0.99.

    Returns:
        List[Optional[Tuple[float, float]]]: For every class, its threshold and the recall it keeps, or None if no
            threshold reaches the target precision (or the class has no sample).
    """
    class_thresholds = []
    for class_index in range(probabilities.shape[1]):
        scores, positives = probabilities[:, class_index], positive_masks[:, class_index]
        number_positives = int(positives.sum())
        if number_positives == 0:
            class_thresholds.append(None)
            continue

        # Precision of every cut of the scores sorted by decreasing value
        order = torch.argsort(input=scores, descending=True, stable=True)
        sorted_scores = scores[order]
        true_positives = torch.cumsum(input=positives[order].to(dtype=torch.float64), dim=0)
        precision = true_positives / torch.arange(start=1, end=len(scores) + 1, dtype=torch.float64)

        # A threshold can only cut between two different scores, so only the last position of every score counts
        is_cut = torch.cat(tensors=[sorted_scores[1:] != sorted_scores[:-1], torch.tensor(data=[True])])
        valid_cuts = torch.nonzero(input=is_cut & (precision >= target_precision) & (true_positives > 0)).flatten()
        if len(valid_cuts) == 0:
            class_thresholds.append(None)
            continue

        # The deepest valid cut keeps the most positives at the target precision
        last_valid_cut = int(valid_cuts[-1])
        class_thresholds.append((float(sorted_scores[last_valid_cut]),
                                 float(true_positives[last_valid_cut]) / number_positives))
    return class_thresholds
