"""
Tests of vision_training/evaluation/sequence/frame_selection_metrics.py: the selection accuracy, fully right sequences
and the background and swapped picks of a decoding.
"""

import numpy as np
import pytest

from vision_training.evaluation.sequence.frame_selection_metrics import compute_selection_metrics
from vision_training.models.sequence.frame_selection_decoding import NO_FRAME

# Two sequences of 4 frames, wanted classes 0 and 1, unselected class 2: the first sequence is decoded right, the
# second one chooses frame 1 for class 0 instead of frame 3
FRAME_LABELS = [np.array(object=[0, 2, 1, 2]), np.array(object=[2, 2, 1, 0])]
SELECTED_FRAMES = [np.array(object=[[0], [2]]), np.array(object=[[1], [2]])]


def test_selection_metrics_of_a_decoding() -> None:
    """
    Verifies the metrics of a decoding: class 0 has one of its two true frames chosen (in the first sequence, not in
    the second one, as the per-sequence counts of its Korn-Graubard interval give), class 1 both, one sequence of two is
    fully right, and the only wrong pick of the 4 took an unselected frame, a background pick, none taking the true
    frame of the other wanted class.
    """
    metrics = compute_selection_metrics(selected_frames=SELECTED_FRAMES, frame_labels=FRAME_LABELS,
                                        wanted_class_indices=[0, 1], unselected_class_index=2)
    assert metrics["class_accuracies"] == [0.5, 1.0]
    assert metrics["class_true_frames"] == [2, 2]
    assert metrics["sequence_class_right_frames"].tolist() == [[1, 1], [0, 1]]
    assert metrics["sequence_class_true_frames"].tolist() == [[1, 1], [1, 1]]
    assert metrics["selection_accuracy"] == pytest.approx(0.75)
    assert metrics["fully_right_sequences"] == pytest.approx(0.5)
    assert metrics["background_picks"] == pytest.approx(0.25)
    assert metrics["swapped_picks"] == pytest.approx(0.0)


def test_wrong_picks_are_split_into_background_and_swapped_picks() -> None:
    """
    Verifies that a pick taking the true frame of another wanted class is a swapped pick: in a sequence of 5 frames
    with wanted classes 0 and 1 of two true frames each, class 0 takes one of its frames and the true frame 2 of class
    1, class 1 takes the true frame 1 of class 0 and the unselected frame 4, so that of the 4 picks one is right, two
    are swapped and one is a background pick, the three shares adding up to every pick; a frame left without pick
    (NO_FRAME) counts in none of them.
    """
    frame_labels = [np.array(object=[0, 0, 1, 1, 2])]
    selected_frames = [np.array(object=[[0, 2], [1, 4]])]
    metrics = compute_selection_metrics(selected_frames=selected_frames, frame_labels=frame_labels,
                                        wanted_class_indices=[0, 1], unselected_class_index=2)
    assert metrics["selection_accuracy"] == pytest.approx(0.25)
    assert metrics["swapped_picks"] == pytest.approx(0.5)
    assert metrics["background_picks"] == pytest.approx(0.25)
    assert metrics["selection_accuracy"] + metrics["swapped_picks"] + metrics["background_picks"] == pytest.approx(1.0)

    # A missing pick of class 1 leaves 3 picks, one right, two swapped
    metrics = compute_selection_metrics(selected_frames=[np.array(object=[[0, 2], [1, NO_FRAME]])],
                                        frame_labels=frame_labels, wanted_class_indices=[0, 1],
                                        unselected_class_index=2)
    assert metrics["swapped_picks"] == pytest.approx(2 / 3)
    assert metrics["background_picks"] == pytest.approx(0.0)


def test_a_pick_of_an_unknown_class_is_refused() -> None:
    """
    Verifies that a chosen frame whose class is neither a wanted class nor the unselected class stops the evaluation
    instead of being counted in neither share.
    """
    with pytest.raises(ValueError, match="got class 3"):
        compute_selection_metrics(selected_frames=[np.array(object=[[3], [2]])],
                                  frame_labels=[np.array(object=[0, 2, 1, 3])], wanted_class_indices=[0, 1],
                                  unselected_class_index=2)
