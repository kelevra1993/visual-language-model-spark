"""
Tests of the classification evaluation of vision_training/evaluation/classification: the confusion matrix and metrics
against hand-computed values, a class predicted but absent lowering the macro F1 as in scikit-learn, the top-k
accuracy, the multi-label metrics, the class thresholds of threshold_finder.py and the sparse confusion matrix of the
evaluations of many classes against the dense one.
"""

import pytest
import torch

from sklearn.metrics import balanced_accuracy_score, f1_score

from vision_training.evaluation.classification.classification_metrics import (compute_confusion_matrix,
                                                                              compute_metrics_from_confusion_matrix,
                                                                              compute_multi_label_metrics,
                                                                              compute_top_k_accuracy)
from vision_training.evaluation.classification.threshold_finder import find_class_thresholds


def test_confusion_matrix_and_metrics_match_hand_computed_values() -> None:
    """
    Verifies the confusion matrix and the multi-class metrics on 6 samples of 3 classes, against values computed by
    hand: class 0 has 3 samples (2 right, 1 predicted as class 1), class 1 has 2 samples (both right) and class 2 has
    1 sample (predicted as class 1).
    """
    labels = torch.tensor(data=[0, 0, 0, 1, 1, 2])
    predicted_labels = torch.tensor(data=[0, 0, 1, 1, 1, 1])

    confusion_matrix = compute_confusion_matrix(predicted_labels=predicted_labels, labels=labels, number_classes=3)
    assert confusion_matrix.tolist() == [[2, 1, 0], [0, 2, 0], [0, 1, 0]]

    metrics = compute_metrics_from_confusion_matrix(confusion_matrix=confusion_matrix)
    assert metrics["accuracy"] == pytest.approx(expected=4 / 6)
    assert metrics["precision"] == pytest.approx(expected=[1.0, 0.5, 0.0])
    assert metrics["recall"] == pytest.approx(expected=[2 / 3, 1.0, 0.0])
    assert metrics["f1"] == pytest.approx(expected=[0.8, 2 / 3, 0.0])
    assert metrics["support"] == [3, 2, 1]
    assert metrics["true_positives"] == [2, 2, 0]
    assert metrics["balanced_accuracy"] == pytest.approx(expected=(2 / 3 + 1.0 + 0.0) / 3)
    assert metrics["macro_f1"] == pytest.approx(expected=(0.8 + 2 / 3 + 0.0) / 3)


def test_class_predicted_but_absent_lowers_the_macro_f1() -> None:
    """
    Verifies that a class predicted but absent from the evaluated samples counts in the macro F1 with its F1 of 0, as
    in scikit-learn (the labels of the true and predicted classes), while the balanced accuracy stays the mean recall
    of the classes present, and that both match scikit-learn on random predictions spread over more classes than the
    true labels.
    """
    # Two samples of class 0 and two of class 1, one of each predicted as class 2, which has no sample
    confusion_matrix = compute_confusion_matrix(predicted_labels=torch.tensor(data=[0, 2, 1, 2]),
                                                labels=torch.tensor(data=[0, 0, 1, 1]), number_classes=3)
    metrics = compute_metrics_from_confusion_matrix(confusion_matrix=confusion_matrix)
    assert metrics["f1"] == pytest.approx(expected=[2 / 3, 2 / 3, 0.0])
    assert metrics["macro_f1"] == pytest.approx(expected=(2 / 3 + 2 / 3 + 0.0) / 3)
    assert metrics["balanced_accuracy"] == pytest.approx(expected=0.5)

    # Random true labels among 4 of 6 classes and predictions among all 6, compared with scikit-learn
    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        labels = torch.randint(low=0, high=4, size=(50,), generator=generator)
        predicted_labels = torch.randint(low=0, high=6, size=(50,), generator=generator)
        metrics = compute_metrics_from_confusion_matrix(confusion_matrix=compute_confusion_matrix(
            predicted_labels=predicted_labels, labels=labels, number_classes=6))
        assert metrics["macro_f1"] == pytest.approx(expected=f1_score(y_true=labels.numpy(),
                                                                      y_pred=predicted_labels.numpy(),
                                                                      average="macro", zero_division=0))
        assert metrics["balanced_accuracy"] == pytest.approx(expected=balanced_accuracy_score(
            y_true=labels.numpy(), y_pred=predicted_labels.numpy()))


def test_top_k_accuracy() -> None:
    """
    Verifies the top-k accuracy: a sample counts as right when its true class is among its top_k most probable
    classes, top_k being clipped to the number of classes.
    """
    probabilities = torch.tensor(data=[[0.5, 0.3, 0.2], [0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
    labels = torch.tensor(data=[1, 2, 2])
    assert compute_top_k_accuracy(probabilities=probabilities, labels=labels, top_k=1) == pytest.approx(expected=1 / 3)
    assert compute_top_k_accuracy(probabilities=probabilities, labels=labels, top_k=2) == pytest.approx(expected=2 / 3)
    assert compute_top_k_accuracy(probabilities=probabilities, labels=labels, top_k=10) == 1.0


def test_multi_label_metrics() -> None:
    """
    Verifies the multi-label metrics against hand-computed values: 3 samples, 2 classes, classes predicted from a
    probability of 0.5.
    """
    probabilities = torch.tensor(data=[[0.9, 0.8], [0.7, 0.2], [0.1, 0.6]])
    labels = torch.tensor(data=[[1.0, 1.0], [0.0, 0.0], [0.0, 1.0]])

    metrics = compute_multi_label_metrics(probabilities=probabilities, labels=labels, threshold=0.5)
    # Class 0: 1 true positive, 1 false positive. Class 1: 2 true positives
    assert metrics["precision"] == pytest.approx(expected=[0.5, 1.0])
    assert metrics["recall"] == pytest.approx(expected=[1.0, 1.0])
    assert metrics["support"] == [1, 2]
    assert metrics["true_positives"] == [1, 2]
    assert metrics["subset_accuracy"] == pytest.approx(expected=2 / 3)
    assert metrics["micro_f1"] == pytest.approx(expected=2 * 0.75 * 1.0 / 1.75)
    assert metrics["macro_f1"] == pytest.approx(expected=(2 / 3 + 1.0) / 2)


def test_class_thresholds_reach_the_target_precision() -> None:
    """
    Verifies the per-class thresholds: the threshold keeping the highest recall at the target precision is kept with
    its recall, a class whose target precision is never reached gets None, and so does a class without sample.
    """
    probabilities = torch.tensor(data=[[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.7, 0.9, 0.0], [0.6, 0.3, 0.0],
                                       [0.4, 0.95, 0.0]])
    positive_masks = torch.tensor(data=[[True, False, False], [True, False, False], [False, True, False],
                                        [True, False, False], [False, False, False]])

    class_thresholds = find_class_thresholds(probabilities=probabilities, positive_masks=positive_masks,
                                             target_precision=0.75)
    # Class 0: scores 0.9 (+), 0.8 (+), 0.7 (-), 0.6 (+), 0.4 (-): cutting at 0.6 keeps a precision of 3 / 4
    assert class_thresholds[0] == pytest.approx(expected=(0.6, 1.0))
    # Class 1: the highest score 0.95 is a negative, then 0.9 is the positive: the precision never exceeds 1 / 2
    assert class_thresholds[1] is None
    # Class 2 has no positive sample
    assert class_thresholds[2] is None


def test_class_threshold_stops_at_the_last_positive_it_needs() -> None:
    """
    Verifies that among the thresholds reaching the target precision with the highest recall the highest one is kept:
    a lower threshold adding only wrong predictions keeps the same recall at a precision closer to the target, which
    new data would more likely fall below.
    """
    # Scores 0.9 (+), 0.9 (-), 0.8 (+), 0.7 (+), 0.7 (+), 0.1 (-): cutting at 0.7 keeps all 4 positives at a precision
    # of 4 / 5, cutting at 0.1 keeps the same 4 at 4 / 6, both above the target 0.6
    probabilities = torch.tensor(data=[[0.9], [0.9], [0.8], [0.7], [0.7], [0.1]])
    positive_masks = torch.tensor(data=[[True], [False], [True], [True], [True], [False]])

    class_thresholds = find_class_thresholds(probabilities=probabilities, positive_masks=positive_masks,
                                             target_precision=0.6)
    assert class_thresholds[0] == pytest.approx(expected=(0.7, 1.0))


def test_sparse_confusion_matrix_gives_the_metrics_of_the_dense_one() -> None:
    """
    Verifies the sparse confusion matrix of the evaluations above MAXIMUM_DENSE_CONFUSION_CLASSES classes: it holds
    the non-zero cells of the dense matrix only, sorted by true class then predicted class, and gives exactly the
    metrics of the dense matrix, on random labels with absent classes and classes predicted without samples, on a
    single sample and on an empty split.
    """
    generator = torch.Generator().manual_seed(3)
    for number_samples in [0, 1, 300]:
        # True labels among 9 of 12 classes, predictions among all 12
        labels = torch.randint(low=0, high=9, size=(number_samples,), generator=generator)
        predicted_labels = torch.randint(low=0, high=12, size=(number_samples,), generator=generator)
        dense_matrix = compute_confusion_matrix(predicted_labels=predicted_labels, labels=labels, number_classes=12)
        sparse_matrix = compute_confusion_matrix(predicted_labels=predicted_labels, labels=labels, number_classes=12,
                                                 sparse=True)

        # The non-zero cells only, in the row order of the dense matrix
        assert sparse_matrix.is_sparse and torch.equal(input=sparse_matrix.to_dense(), other=dense_matrix)
        assert sparse_matrix.indices().t().tolist() == torch.nonzero(input=dense_matrix).tolist()
        assert sparse_matrix.values().tolist() == dense_matrix[dense_matrix > 0].tolist()

        # The same metrics, to the last bit
        assert (compute_metrics_from_confusion_matrix(confusion_matrix=sparse_matrix) ==
                compute_metrics_from_confusion_matrix(confusion_matrix=dense_matrix))
