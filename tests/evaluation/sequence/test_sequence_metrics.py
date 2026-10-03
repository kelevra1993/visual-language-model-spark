"""
Tests of vision_training/evaluation/sequence/sequence_metrics.py: the frame accuracy and the sequence exact matches of
the sequence labelling task ignore the frames without class.
"""

import pytest
import torch

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.evaluation.sequence.sequence_metrics import compute_frame_accuracy, compute_sequence_exact_matches


def test_frame_accuracy_and_exact_matches_ignore_the_frames_without_class() -> None:
    """
    Verifies the frame accuracy and the sequence exact matches against hand-computed values, the frames labelled -100
    (padding frames, frames of unmapped categories) counting neither as right nor as wrong.
    """
    labels = torch.tensor(data=[[0, 1, IGNORED_LABEL_INDEX], [2, 2, 1], [IGNORED_LABEL_INDEX, 0, 0]])
    predicted_labels = torch.tensor(data=[[0, 1, 2], [2, 0, 1], [1, 0, 0]])

    # 6 of the 7 labelled frames are right, the middle sequence has a wrong frame
    assert float(compute_frame_accuracy(predicted_labels=predicted_labels, labels=labels)) == pytest.approx(
        expected=6 / 7)
    assert compute_sequence_exact_matches(predicted_labels=predicted_labels, labels=labels).tolist() == [True, False,
                                                                                                         True]
    assert float(compute_frame_accuracy(predicted_labels=predicted_labels,
                                        labels=torch.full(size=(1, 3), fill_value=IGNORED_LABEL_INDEX))) == 0.0
