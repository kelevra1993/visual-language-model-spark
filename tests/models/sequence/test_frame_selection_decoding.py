"""
Tests of vision_training/models/sequence/frame_selection_decoding.py: the optimality of the Hungarian assignment of the
frame selection, the greedy completion of the classes short of frames, the ranks, candidates and decoded frame labels,
and the outputs of the deployed decoding with its low-confidence flags.
"""

import itertools
import numpy as np
import pytest

from vision_training.models.sequence.frame_selection_decoding import (NO_FRAME, compute_frame_ranks,
                                                                      convert_selection_to_frame_labels,
                                                                      decode_frame_selection,
                                                                      find_candidate_frames, run_frame_selection)

# Example of the documentation: three frames, two wanted classes A and B (indices 0 and 1) and the unselected class 2
EXAMPLE_PROBABILITIES = np.array(object=[[0.50, 0.45, 0.05], [0.45, 0.05, 0.50], [0.02, 0.08, 0.90]])


def compute_labelling_log_probability(probabilities: np.ndarray, selected_frames: np.ndarray,
                                      unselected_class_index: int) -> float:
    """
    Computes the log-probability of the labelling of a whole sequence given by a selection, every chosen frame taking
    the class of its row and every other frame the unselected class, the quantity the Hungarian decoding maximises.

    Args:
        probabilities (np.ndarray): The [T, K] class probabilities of every frame.
        selected_frames (np.ndarray): The [W, N] chosen frames of every wanted class.
        unselected_class_index (int): The class index of the unselected frames.

    Returns:
        float: The log-probability of the labelling.
    """
    wanted_class_indices = [class_index for class_index in range(probabilities.shape[1])
                            if class_index != unselected_class_index]
    frame_labels = convert_selection_to_frame_labels(
        selected_frames=selected_frames, wanted_class_indices=wanted_class_indices,
        number_frames=probabilities.shape[0], unselected_class_index=unselected_class_index)
    return float(np.log(probabilities[np.arange(probabilities.shape[0]), frame_labels]).sum())


@pytest.mark.parametrize(argnames="seed", argvalues=[0])
def test_hungarian_decoding_gives_the_most_probable_labelling(seed: int) -> None:
    """
    Verifies on random sequences of 6 frames, with two wanted classes of two frames each, that the Hungarian decoding
    gives the most probable labelling of the sequence among every labelling holding two frames of every wanted class,
    enumerated exhaustively, and that it chooses every frame at most once.

    Args:
        seed (int): The seed of the random probabilities.
    """
    random_generator = np.random.default_rng(seed=seed)
    probabilities = random_generator.dirichlet(alpha=np.ones(3), size=6)
    selected_frames = decode_frame_selection(probabilities=probabilities, unselected_class_index=0,
                                             frames_per_class=2, decoding="hungarian")
    assert len(set(selected_frames.flatten().tolist())) == 4

    # Every way of giving two frames to each wanted class
    best_log_probability = max(
        compute_labelling_log_probability(probabilities=probabilities,
                                          selected_frames=np.array(object=[frames[:2], frames[2:]]),
                                          unselected_class_index=0)
        for frames in itertools.permutations(range(6), 4))
    assert compute_labelling_log_probability(probabilities=probabilities, selected_frames=selected_frames,
                                             unselected_class_index=0) == pytest.approx(best_log_probability)


def test_greedy_decoding_keeps_the_best_frames_and_completes_the_short_classes() -> None:
    """
    Verifies the greedy rule with one frame per class: a class predicted by two frames keeps the more probable one,
    the other becoming unselected, and a class predicted by no frame takes the unselected frame most probable for it,
    including a frame given up by another class.
    """
    probabilities = np.array(object=[[0.1, 0.8, 0.1], [0.1, 0.6, 0.3], [0.7, 0.1, 0.2], [0.3, 0.1, 0.6]])
    # Wanted classes 1 and 2, unselected class 0: frames 0 and 1 predict class 1, no frame predicts class 2
    selected_frames = decode_frame_selection(probabilities=probabilities, unselected_class_index=0,
                                             frames_per_class=1, decoding="greedy")
    assert selected_frames.tolist() == [[0], [3]]


def test_ranks_candidates_and_frame_labels() -> None:
    """
    Verifies the rank of every frame for every wanted class (1 for the most probable, ties sharing the best rank), the
    candidate frames by decreasing probability padded with NO_FRAME, and the decoded class of every frame.
    """
    probabilities = np.array(object=[[0.5, 0.2, 0.3], [0.5, 0.4, 0.1], [0.1, 0.1, 0.8]])
    assert compute_frame_ranks(probabilities=probabilities, wanted_class_indices=[0, 1]).tolist() == [
        [1, 2], [1, 1], [3, 3]]
    assert find_candidate_frames(probabilities=probabilities, wanted_class_indices=[0, 1],
                                 number_candidates=4).tolist() == [[0, 1, 2, NO_FRAME], [1, 0, 2, NO_FRAME]]
    assert convert_selection_to_frame_labels(selected_frames=np.array(object=[[2], [NO_FRAME]]),
                                             wanted_class_indices=[0, 1], number_frames=3,
                                             unselected_class_index=2).tolist() == [2, 2, 0]


def test_deployed_decoding_outputs() -> None:
    """
    Verifies the outputs of the decoding run by the Triton decoder and the local inference on the documented example:
    the chosen frames with their probabilities, the low-confidence flags against the thresholds of the classes (a
    class without threshold flagging all its frames), and the candidate frames with their probabilities.
    """
    outputs = run_frame_selection(probabilities=EXAMPLE_PROBABILITIES, decoding_settings={
        "decoding": "hungarian", "frames_per_class": 1, "unselected_class_index": 2, "number_candidates": 2,
        "class_thresholds": [0.4, None]})
    assert outputs["selected_frames"].tolist() == [[1], [0]]
    assert outputs["selected_probabilities"].flatten().tolist() == pytest.approx([0.45, 0.45])
    assert outputs["low_confidence"].tolist() == [[False], [True]]
    assert outputs["candidate_frames"].tolist() == [[0, 1], [0, 2]]
    assert outputs["candidate_probabilities"].flatten().tolist() == pytest.approx([0.50, 0.45, 0.45, 0.08])
    assert outputs["selected_frames"].dtype == np.int64 and outputs["low_confidence"].dtype == bool
