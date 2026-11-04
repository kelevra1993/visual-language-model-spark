"""
Tests of vision_training/evaluation/sequence/frame_selection_metrics.py: the selection accuracy, fully right sequences
and unselected precision and recall of a decoding.
"""

import numpy as np
import pytest

from vision_training.evaluation.sequence.frame_selection_metrics import compute_selection_metrics
from vision_training.models.sequence.frame_selection_decoding import convert_selection_to_frame_labels

# Two sequences of 4 frames, wanted classes 0 and 1, unselected class 2: the first sequence is decoded right, the
# second one chooses frame 1 for class 0 instead of frame 3
FRAME_LABELS = [np.array(object=[0, 2, 1, 2]), np.array(object=[2, 2, 1, 0])]
SELECTED_FRAMES = [np.array(object=[[0], [2]]), np.array(object=[[1], [2]])]


def test_selection_metrics_of_a_decoding() -> None:
    """
    Verifies the metrics of a decoding: class 0 has one of its two true frames chosen (in the first sequence, not in
    the second one, as the per-sequence counts of its Korn-Graubard interval give), class 1 both, one sequence of two is
    fully right, and the unselected class has 3 of its 4 decoded frames truly unselected and 3 of its 4 true frames
    decoded unselected.
    """
    decoded_frame_labels = [convert_selection_to_frame_labels(selected_frames=selection, wanted_class_indices=[0, 1],
                                                              number_frames=4, unselected_class_index=2)
                            for selection in SELECTED_FRAMES]
    metrics = compute_selection_metrics(selected_frames=SELECTED_FRAMES, frame_labels=FRAME_LABELS,
                                        decoded_frame_labels=decoded_frame_labels, wanted_class_indices=[0, 1],
                                        unselected_class_index=2)
    assert metrics["class_accuracies"] == [0.5, 1.0]
    assert metrics["class_true_frames"] == [2, 2]
    assert metrics["sequence_class_right_frames"].tolist() == [[1, 1], [0, 1]]
    assert metrics["sequence_class_true_frames"].tolist() == [[1, 1], [1, 1]]
    assert metrics["selection_accuracy"] == pytest.approx(0.75)
    assert metrics["fully_right_sequences"] == pytest.approx(0.5)
    assert metrics["unselected_precision"] == pytest.approx(0.75)
    assert metrics["unselected_recall"] == pytest.approx(0.75)
