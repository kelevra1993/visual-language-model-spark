"""
Tests of vision_training/evaluation/sequence/segment_metrics.py: the edit score and segment F1 of a fragmented
prediction, of a perfect prediction and without sequence, the segments ended by unlabelled frames, the edit distance,
and the windows of the test sequences merged back into whole sequences.
"""

import numpy as np
import pytest
import torch

from vision_training.evaluation.sequence.segment_metrics import (compute_edit_distance, compute_segment_metrics,
                                                                 extract_segments, merge_window_predictions)


def test_fragmented_prediction_scores_low_on_segments() -> None:
    """
    Verifies the segment metrics of a sequence of 10 frames, A on frames 0 to 4 and B on frames 5 to 9, predicted
    A A B A A B B A B B: 80 % of the frames are right, but the 6 predicted segments need 4 edits (edit score 1/3), and
    only one predicted segment per true segment is right, overlapping it by 2/5 (segment F1 of 0.5 at 10 and 25 %
    overlap, 0 at 50 %).
    """
    sequence = {"frame_indices": np.arange(10), "true_classes": np.array(object=[0] * 5 + [1] * 5),
                "predicted_classes": np.array(object=[0, 0, 1, 0, 0, 1, 1, 0, 1, 1])}
    segment_results = compute_segment_metrics(sequences=[sequence])

    assert segment_results["edit_score"] == pytest.approx(expected=1.0 / 3.0)
    assert segment_results["segment_f1"] == {0.10: pytest.approx(expected=0.5), 0.25: pytest.approx(expected=0.5),
                                             0.50: 0.0}
    assert (segment_results["number_sequences"], segment_results["number_true_segments"],
            segment_results["number_predicted_segments"]) == (1, 2, 6)


def test_perfect_prediction_and_no_sequence() -> None:
    """
    Verifies that predicting every frame right gives an edit score and segment F1 of 1, and that an evaluation without
    sequence gives 0 everywhere.
    """
    true_classes = np.array(object=[2, 2, 0, 0, 0, 1])
    perfect_results = compute_segment_metrics(sequences=[{"frame_indices": np.arange(6), "true_classes": true_classes,
                                                          "predicted_classes": true_classes.copy()}])
    assert perfect_results["edit_score"] == 1.0
    assert all(segment_f1 == 1.0 for segment_f1 in perfect_results["segment_f1"].values())

    empty_results = compute_segment_metrics(sequences=[])
    assert empty_results["edit_score"] == 0.0 and all(segment_f1 == 0.0
                                                      for segment_f1 in empty_results["segment_f1"].values())


def test_unlabelled_frames_end_a_segment() -> None:
    """
    Verifies that a gap of unlabelled frames, absent from the labelled frames, ends a segment even when the class
    goes on after it, and that the edit distance counts insertions, deletions and replacements.
    """
    assert extract_segments(frame_classes=np.array(object=[0, 0, 0, 0, 1]),
                            frame_indices=np.array(object=[0, 1, 2, 5, 6])) == [(0, 0, 2), (0, 5, 5), (1, 6, 6)]
    assert extract_segments(frame_classes=np.array(object=[], dtype=np.int64),
                            frame_indices=np.array(object=[], dtype=np.int64)) == []
    assert compute_edit_distance(predicted_classes=[0, 1, 0, 1], true_classes=[0, 1]) == 2
    assert compute_edit_distance(predicted_classes=[2, 1], true_classes=[0, 1, 3]) == 2
    assert compute_edit_distance(predicted_classes=[], true_classes=[0]) == 1


def test_windows_are_merged_into_whole_sequences() -> None:
    """
    Verifies that the labelled frames of overlapping windows are merged back into their sequences: every frame is kept
    once, in order, sequence by sequence, and a frame held by two windows that disagree gets the class of their mean
    probabilities.
    """
    # Sequence 1: windows of frames 0 to 2 and 1 to 3, disagreeing on frames 1 and 2; sequence 0: frames 0 and 1
    sequence_indices = torch.tensor(data=[1, 1, 1, 1, 1, 1, 0, 0])
    frame_indices = torch.tensor(data=[0, 1, 2, 1, 2, 3, 1, 0])
    probabilities = torch.tensor(data=[[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9],
                                       [0.3, 0.7], [0.8, 0.2]], dtype=torch.float64)
    labels = torch.tensor(data=[0, 0, 1, 0, 1, 1, 1, 0])
    sequences = merge_window_predictions(sequence_indices=sequence_indices, frame_indices=frame_indices,
                                         probabilities=probabilities, labels=labels)

    assert len(sequences) == 2
    assert sequences[0]["frame_indices"].tolist() == [0, 1] and sequences[0]["true_classes"].tolist() == [0, 1]
    assert sequences[0]["predicted_classes"].tolist() == [0, 1]
    # Frame 1: mean (0.4, 0.6) gives class 1; frame 2: mean (0.45, 0.55) gives class 1
    assert sequences[1]["frame_indices"].tolist() == [0, 1, 2, 3]
    assert sequences[1]["true_classes"].tolist() == [0, 0, 1, 1]
    assert sequences[1]["predicted_classes"].tolist() == [0, 1, 1, 1]
    assert merge_window_predictions(sequence_indices=sequence_indices[:0], frame_indices=frame_indices[:0],
                                    probabilities=probabilities[:0], labels=labels[:0]) == []
