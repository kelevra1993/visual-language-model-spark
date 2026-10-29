"""
Metrics of the classification-like tasks: confusion matrix, per-class precision, recall, F1 and support, accuracy,
top-k accuracy and the multi-label metrics. Used by the classification trainer, from which the sequence classification
and comparison trainers inherit, to evaluate a checkpoint on the test split.
"""

import torch

from typing import Any, Dict

# Number of most probable classes among which the true class must be for the top-k accuracy of the classification-like
# tasks, the frame selection taking its own k from 'evaluation.top_k'
TOP_K_ACCURACY_CLASSES = 5


def compute_confusion_matrix(predicted_labels: torch.Tensor, labels: torch.Tensor, number_classes: int) -> torch.Tensor:
    """
    Counts, for every pair of classes, the samples of the first class predicted as the second one, the basis of the
    classification metrics of the evaluated checkpoints.

    Args:
        predicted_labels (torch.Tensor): The [N] predicted class indices.
        labels (torch.Tensor): The [N] true class indices.
        number_classes (int): The number of classes.

    Returns:
        torch.Tensor: The [K, K] int64 confusion matrix, rows being the true classes and columns the predicted ones.
    """
    flat_indices = labels.to(dtype=torch.int64) * number_classes + predicted_labels.to(dtype=torch.int64)
    return torch.bincount(input=flat_indices, minlength=number_classes * number_classes).reshape(number_classes,
                                                                                                 number_classes)


def compute_precision_recall_f1(true_positives: torch.Tensor, predicted_counts: torch.Tensor,
                                actual_counts: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Computes the per-class precision, recall and F1 score from the per-class counts, a class without predictions (or
    without samples) getting a precision (or recall) of 0.

    Args:
        true_positives (torch.Tensor): The [K] correctly predicted samples of every class.
        predicted_counts (torch.Tensor): The [K] samples predicted as every class.
        actual_counts (torch.Tensor): The [K] samples of every class.

    Returns:
        Dict[str, torch.Tensor]: The [K] float64 'precision', 'recall' and 'f1' of every class.
    """
    true_positives, predicted_counts, actual_counts = (true_positives.to(dtype=torch.float64),
                                                       predicted_counts.to(dtype=torch.float64),
                                                       actual_counts.to(dtype=torch.float64))
    precision = torch.where(condition=predicted_counts > 0, input=true_positives / predicted_counts.clamp(min=1.0),
                            other=0.0)
    recall = torch.where(condition=actual_counts > 0, input=true_positives / actual_counts.clamp(min=1.0), other=0.0)
    f1 = torch.where(condition=precision + recall > 0,
                     input=2.0 * precision * recall / (precision + recall).clamp(min=1e-12), other=0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_metrics_from_confusion_matrix(confusion_matrix: torch.Tensor) -> Dict[str, Any]:
    """
    Computes the multi-class metrics of an evaluation from its confusion matrix: the accuracy, the balanced accuracy
    (mean recall over the classes present in the evaluated samples, insensitive to class imbalance, as in
    scikit-learn), the macro F1 over the classes present or predicted (a class predicted without any sample counting
    with its F1 of 0, as in scikit-learn), and the precision, recall, F1 and number of evaluated samples (support) of
    every class.

    Args:
        confusion_matrix (torch.Tensor): The [K, K] confusion matrix (rows true classes, columns predicted classes).

    Returns:
        Dict[str, Any]: The 'accuracy', 'balanced_accuracy' and 'macro_f1' floats, the per-class 'precision',
            'recall' and 'f1' lists, and the per-class 'support' list of evaluated samples.
    """
    actual_counts, predicted_counts = confusion_matrix.sum(dim=1), confusion_matrix.sum(dim=0)
    class_metrics = compute_precision_recall_f1(true_positives=confusion_matrix.diagonal(),
                                                predicted_counts=predicted_counts, actual_counts=actual_counts)

    # Balanced accuracy over the classes present in the evaluated samples, macro F1 over the classes present or
    # predicted, so that a class predicted without any sample lowers it
    present_classes = actual_counts > 0
    averaged_classes = present_classes | (predicted_counts > 0)
    number_samples = int(confusion_matrix.sum())
    return {"accuracy": float(confusion_matrix.diagonal().sum()) / number_samples if number_samples else 0.0,
            "balanced_accuracy": float(class_metrics["recall"][present_classes].mean()) if present_classes.any()
            else 0.0,
            "macro_f1": float(class_metrics["f1"][averaged_classes].mean()) if averaged_classes.any() else 0.0,
            "precision": class_metrics["precision"].tolist(), "recall": class_metrics["recall"].tolist(),
            "f1": class_metrics["f1"].tolist(), "support": actual_counts.tolist()}


def compute_top_k_accuracy(probabilities: torch.Tensor, labels: torch.Tensor, top_k: int) -> float:
    """
    Computes the share of samples whose true class is among the top_k most probable predicted classes.

    Args:
        probabilities (torch.Tensor): The [N, K] class probabilities.
        labels (torch.Tensor): The [N] true class indices.
        top_k (int): The number of most probable classes considered (clipped to the number of classes).

    Returns:
        float: The top-k accuracy, 0 without samples.
    """
    if len(labels) == 0:
        return 0.0
    top_classes = probabilities.topk(k=min(top_k, probabilities.shape[1]), dim=1).indices
    return float((top_classes == labels.unsqueeze(dim=1)).any(dim=1).to(dtype=torch.float64).mean())


def compute_multi_label_metrics(probabilities: torch.Tensor, labels: torch.Tensor, threshold: float) -> Dict[str, Any]:
    """
    Computes the metrics of a multi-label evaluation, every class being predicted when its probability reaches the
    threshold: the subset accuracy (samples whose every class is right), the micro F1 (counts pooled over the
    classes), the macro F1 (mean over the classes positive or predicted in the evaluated samples), and the precision,
    recall, F1 and number of positive samples (support) of every class.

    Args:
        probabilities (torch.Tensor): The [N, K] class probabilities.
        labels (torch.Tensor): The [N, K] multi-hot targets.
        threshold (float): The probability from which a class is predicted.

    Returns:
        Dict[str, Any]: The 'subset_accuracy', 'micro_f1' and 'macro_f1' floats, the per-class 'precision',
            'recall' and 'f1' lists, and the per-class 'support' list of positive samples.
    """
    predictions, targets = probabilities >= threshold, labels >= 0.5
    true_positives, predicted_counts, actual_counts = ((predictions & targets).sum(dim=0), predictions.sum(dim=0),
                                                       targets.sum(dim=0))
    class_metrics = compute_precision_recall_f1(true_positives=true_positives, predicted_counts=predicted_counts,
                                                actual_counts=actual_counts)
    pooled_metrics = compute_precision_recall_f1(true_positives=true_positives.sum().unsqueeze(dim=0),
                                                 predicted_counts=predicted_counts.sum().unsqueeze(dim=0),
                                                 actual_counts=actual_counts.sum().unsqueeze(dim=0))

    # Macro F1 over the classes positive or predicted in some sample, a class predicted but never positive lowering it
    averaged_classes = (actual_counts > 0) | (predicted_counts > 0)
    return {"subset_accuracy": float((predictions == targets).all(dim=1).to(dtype=torch.float64).mean())
            if len(labels) else 0.0,
            "micro_f1": float(pooled_metrics["f1"][0]),
            "macro_f1": float(class_metrics["f1"][averaged_classes].mean()) if averaged_classes.any() else 0.0,
            "precision": class_metrics["precision"].tolist(), "recall": class_metrics["recall"].tolist(),
            "f1": class_metrics["f1"].tolist(), "support": actual_counts.tolist()}
