"""
Tests of vision_training/evaluation/sequence/sequence_metrics.py: the frame accuracy and the sequence exact matches of
the sequence labelling task ignore the frames without class, and the sequences without any labelled frame are left out
of the exact matches.
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


def test_sequences_without_labelled_frame_are_left_out_of_the_exact_matches() -> None:
    """
    Verifies that the sequences whose frames are all without class have no exact match to give: one sequence with
    every frame wrong and two sequences of unmapped categories give one exact match, False, so a Sequence Exact Match
    of 0 over one test sequence like the frame accuracy, instead of 2 / 3 over three sequences.
    """
    labels = torch.tensor(data=[[0, 1, 2], [IGNORED_LABEL_INDEX] * 3, [IGNORED_LABEL_INDEX] * 3])
    predicted_labels = torch.tensor(data=[[1, 2, 0], [0, 0, 0], [1, 1, 1]])

    exact_matches = compute_sequence_exact_matches(predicted_labels=predicted_labels, labels=labels)
    assert exact_matches.tolist() == [False]
    assert float(compute_frame_accuracy(predicted_labels=predicted_labels, labels=labels)) == 0.0

    # A batch of unlabelled sequences only gives no exact match at all
    assert compute_sequence_exact_matches(predicted_labels=predicted_labels[1:], labels=labels[1:]).shape == (0,)
