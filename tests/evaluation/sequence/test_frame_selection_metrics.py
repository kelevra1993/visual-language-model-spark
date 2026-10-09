"""
Tests of vision_training/evaluation/sequence/frame_selection_metrics.py: the selection accuracy, fully right sequences
and unselected precision and recall of a decoding, the top-k accuracy and mean rank of the true frames, and the
confidence thresholds of the chosen frames.
"""

import numpy as np
import pytest

from vision_training.evaluation.sequence.frame_selection_metrics import (compute_ranking_metrics,
                                                                         compute_selection_metrics,
                                                                         find_selection_thresholds)
from vision_training.models.sequence.frame_selection_decoding import convert_selection_to_frame_labels

# Two sequences of 4 frames, wanted classes 0 and 1, unselected class 2: the first sequence is decoded right, the
# second one chooses frame 1 for class 0 instead of frame 3
FRAME_LABELS = [np.array(object=[0, 2, 1, 2]), np.array(object=[2, 2, 1, 0])]
SELECTED_FRAMES = [np.array(object=[[0], [2]]), np.array(object=[[1], [2]])]
PROBABILITIES = [np.array(object=[[0.9, 0.05, 0.05], [0.2, 0.1, 0.7], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
                 np.array(object=[[0.1, 0.1, 0.8], [0.6, 0.1, 0.3], [0.1, 0.8, 0.1], [0.5, 0.1, 0.4]])]


def test_selection_metrics_of_a_decoding() -> None:
    """
    Verifies the metrics of a decoding: class 0 has one of its two true frames chosen, class 1 both, one sequence of
    two is fully right, and the unselected class has 3 of its 4 decoded frames truly unselected and 3 of its 4 true
    frames decoded unselected.
    """
    decoded_frame_labels = [convert_selection_to_frame_labels(selected_frames=selection, wanted_class_indices=[0, 1],
                                                              number_frames=4, unselected_class_index=2)
                            for selection in SELECTED_FRAMES]
    metrics = compute_selection_metrics(selected_frames=SELECTED_FRAMES, frame_labels=FRAME_LABELS,
                                        decoded_frame_labels=decoded_frame_labels, wanted_class_indices=[0, 1],
                                        unselected_class_index=2)
    assert metrics["class_accuracies"] == [0.5, 1.0]
    assert metrics["class_true_frames"] == [2, 2]
    assert metrics["selection_accuracy"] == pytest.approx(0.75)
    assert metrics["fully_right_sequences"] == pytest.approx(0.5)
    assert metrics["unselected_precision"] == pytest.approx(0.75)
    assert metrics["unselected_recall"] == pytest.approx(0.75)


def test_ranking_metrics_before_decoding() -> None:
    """
    Verifies the ranks of the true frames: the true frame of class 0 of the second sequence comes second behind frame
    1, every other true frame first, so that the top-1 accuracy misses it and the top-2 accuracy does not.
    """
    top_one_metrics = compute_ranking_metrics(probabilities=PROBABILITIES, frame_labels=FRAME_LABELS,
                                              wanted_class_indices=[0, 1], top_k=1)
    assert top_one_metrics["class_mean_ranks"] == [1.5, 1.0]
    assert top_one_metrics["mean_rank"] == pytest.approx(1.25)
    assert top_one_metrics["class_top_k_accuracies"] == [0.5, 1.0]
    assert top_one_metrics["top_k_accuracy"] == pytest.approx(0.75)
    assert compute_ranking_metrics(probabilities=PROBABILITIES, frame_labels=FRAME_LABELS, wanted_class_indices=[0, 1],
                                   top_k=2)["top_k_accuracy"] == pytest.approx(1.0)


def test_selection_thresholds_keep_the_target_precision() -> None:
    """
    Verifies the confidence thresholds: the frames chosen for class 0 are right at 0.9 and wrong at 0.6, so the
    threshold reaching the precision 1 is 0.9, keeping one of the two true frames of the class; the frames chosen for
    class 1 are all right, the threshold being the lowest of them and keeping both true frames.
    """
    class_thresholds = find_selection_thresholds(probabilities=PROBABILITIES, frame_labels=FRAME_LABELS,
                                                 selected_frames=SELECTED_FRAMES, wanted_class_indices=[0, 1],
                                                 target_precision=1.0)
    assert class_thresholds[0] == pytest.approx((0.9, 0.5))
    assert class_thresholds[1] == pytest.approx((0.8, 1.0))
