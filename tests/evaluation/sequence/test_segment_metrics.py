"""
Tests of vision_training/evaluation/sequence/segment_metrics.py: the edit score and segment F1 of a fragmented
prediction, and the labelled frames of the test sequences grouped sequence by sequence.
"""

import numpy as np
import pytest
import torch

from vision_training.evaluation.sequence.segment_metrics import compute_segment_metrics, group_sequence_predictions


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


def test_frames_are_grouped_by_sequence() -> None:
    """
    Verifies that the labelled frames of the test sequences are grouped sequence by sequence, in frame order, every
    frame predicted as its most probable class, that a frame given twice raises an error and that no frame gives no
    sequence.
    """
    # Sequence 1: frames 0, 2 and 3 given out of order (frame 1 unlabelled); sequence 0: frames 0 and 1
    sequence_indices = torch.tensor(data=[1, 1, 0, 1, 0])
    frame_indices = torch.tensor(data=[3, 0, 1, 2, 0])
    probabilities = torch.tensor(data=[[0.1, 0.9], [0.9, 0.1], [0.3, 0.7], [0.6, 0.4], [0.8, 0.2]], dtype=torch.float64)
    labels = torch.tensor(data=[1, 0, 1, 1, 0])
    sequences = group_sequence_predictions(sequence_indices=sequence_indices, frame_indices=frame_indices,
                                           probabilities=probabilities, labels=labels)

    assert len(sequences) == 2
    assert sequences[0]["frame_indices"].tolist() == [0, 1] and sequences[0]["true_classes"].tolist() == [0, 1]
    assert sequences[0]["predicted_classes"].tolist() == [0, 1]
    assert sequences[1]["frame_indices"].tolist() == [0, 2, 3]
    assert sequences[1]["true_classes"].tolist() == [0, 1, 1]
    assert sequences[1]["predicted_classes"].tolist() == [0, 0, 1]

    # A frame given twice raises an error, no frame gives no sequence
    with pytest.raises(expected_exception=ValueError, match="more than once"):
        group_sequence_predictions(sequence_indices=torch.tensor(data=[0, 0]), frame_indices=torch.tensor(data=[1, 1]),
                                   probabilities=probabilities[:2], labels=labels[:2])
    assert group_sequence_predictions(sequence_indices=sequence_indices[:0], frame_indices=frame_indices[:0],
                                      probabilities=probabilities[:0], labels=labels[:0]) == []
