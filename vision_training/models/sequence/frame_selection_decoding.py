"""
Decoding of the frame selection task (SEL): from the per-frame class probabilities of the frame selector, the frames
chosen for every wanted class (optimal Hungarian assignment or greedy rule), the k most probable candidate frames of
every class, the rank of every frame for every class and the low-confidence flags of the chosen frames. Only numpy and
scipy are imported, so that the same file runs in the evaluation of trainers/frame_selection_trainer.py, in
deployment/run_inference.py and, copied next to the Python backend model by deployment/frame_selection_deployment.py,
inside the Triton server.
"""

import numpy as np

from scipy.optimize import linear_sum_assignment
from typing import Any, Dict, List, Optional

# Decodings of the selection: the optimal assignment of the frames to the wanted classes, or the greedy rule
VALID_DECODINGS = ["hungarian", "greedy"]

# Frame index written where a class gets no frame (fewer frames than selections), or where a class has fewer
# candidate frames than asked
NO_FRAME = -1

# Smallest probability entering a logarithm, so that a probability rounded to 0 gives a large finite cost
MINIMUM_PROBABILITY = 1e-12


def get_wanted_class_indices(number_classes: int, unselected_class_index: int) -> List[int]:
    """
    Lists the wanted classes of the frame selection, every class predicted by the model except the unselected one, in
    class index order: the rows of every selection array of this module.

    Args:
        number_classes (int): The number of classes K predicted by the model, the unselected class included.
        unselected_class_index (int): The class index of the unselected frames.

    Returns:
        List[int]: The class indices of the wanted classes.
    """
    return [class_index for class_index in range(number_classes) if class_index != unselected_class_index]


def order_selected_frames(probabilities: np.ndarray, class_frames: List[List[int]],
                          wanted_class_indices: List[int], frames_per_class: int) -> np.ndarray:
    """
    Writes the frames chosen for every wanted class into the selection array shared by both decodings, every row
    holding the frames of one class by decreasing probability for that class, followed by NO_FRAME when the sequence
    has too few frames to fill it.

    Args:
        probabilities (np.ndarray): The [T, K] class probabilities of every frame.
        class_frames (List[List[int]]): The frames chosen for every wanted class, in wanted class order.
        wanted_class_indices (List[int]): The class indices of the wanted classes.
        frames_per_class (int): The number of frames N chosen for every wanted class.

    Returns:
        np.ndarray: The [W, N] int64 chosen frames.
    """
    selected_frames = np.full(shape=(len(wanted_class_indices), frames_per_class), fill_value=NO_FRAME, dtype=np.int64)
    for wanted_index, (class_index, frames) in enumerate(zip(wanted_class_indices, class_frames)):
        # Most probable frames for the class first, ties kept in their given order
        frame_array = np.array(frames, dtype=np.int64)
        ordered_frames = frame_array[np.argsort(-probabilities[frame_array, class_index], kind="stable")]
        selected_frames[wanted_index, :len(ordered_frames)] = ordered_frames
    return selected_frames


def decode_hungarian_selection(probabilities: np.ndarray, unselected_class_index: int,
                               frames_per_class: int) -> np.ndarray:
    """
    Chooses the frames of every wanted class with the optimal assignment: the gain of choosing frame f for class c is
    log p(f, c) - log p(f, unselected), every class column being repeated frames_per_class times, and the assignment
    with the largest total gain (scipy.optimize.linear_sum_assignment) is the most probable labelling of the whole
    sequence in which every wanted class holds frames_per_class frames. A frame wanted by two classes goes to the one
    that loses least without it, unlike the greedy rule.

    Args:
        probabilities (np.ndarray): The [T, K] class probabilities of every frame of the sequence.
        unselected_class_index (int): The class index of the unselected frames.
        frames_per_class (int): The number of frames N chosen for every wanted class.

    Returns:
        np.ndarray: The [W, N] int64 chosen frames of every wanted class, by decreasing probability, NO_FRAME where
            the sequence has fewer frames than selections.
    """
    wanted_class_indices = get_wanted_class_indices(number_classes=probabilities.shape[1],
                                                    unselected_class_index=unselected_class_index)

    # Gain of every frame for every wanted class against staying unselected, one column per selection
    log_probabilities = np.log(np.clip(a=probabilities.astype(np.float64), a_min=MINIMUM_PROBABILITY, a_max=None))
    gains = log_probabilities[:, wanted_class_indices] - log_probabilities[:, [unselected_class_index]]
    selection_gains = np.repeat(a=gains, repeats=frames_per_class, axis=1)

    # Assignment of the frames to the selections with the largest total gain, every frame chosen at most once
    frame_indices, selection_indices = linear_sum_assignment(cost_matrix=selection_gains, maximize=True)
    class_frames = [[] for _ in wanted_class_indices]
    for frame_index, selection_index in zip(frame_indices.tolist(), selection_indices.tolist()):
        class_frames[selection_index // frames_per_class].append(frame_index)
    return order_selected_frames(probabilities=probabilities, class_frames=class_frames,
                                 wanted_class_indices=wanted_class_indices, frames_per_class=frames_per_class)


def decode_greedy_selection(probabilities: np.ndarray, unselected_class_index: int,
                            frames_per_class: int) -> np.ndarray:
    """
    Chooses the frames of every wanted class with the greedy rule: every frame takes its most probable class, a wanted
    class taken by too many frames keeps the most probable ones (the others becoming unselected), then a wanted class
    taken by too few frames is completed, class by class in class order, with the unselected frames most probable for
    it. Cheaper than the Hungarian assignment, it can give a frame to a class that only slightly prefers it over
    another class that has no better frame.

    Args:
        probabilities (np.ndarray): The [T, K] class probabilities of every frame of the sequence.
        unselected_class_index (int): The class index of the unselected frames.
        frames_per_class (int): The number of frames N chosen for every wanted class.

    Returns:
        np.ndarray: The [W, N] int64 chosen frames of every wanted class, by decreasing probability, NO_FRAME where
            the sequence has fewer frames than selections.
    """
    wanted_class_indices = get_wanted_class_indices(number_classes=probabilities.shape[1],
                                                    unselected_class_index=unselected_class_index)
    frame_labels = probabilities.argmax(axis=1)
    class_frames = []

    # Every wanted class keeps its most probable frames among those predicting it, the others become unselected
    for class_index in wanted_class_indices:
        predicted_frames = np.flatnonzero(frame_labels == class_index)
        ordered_frames = predicted_frames[np.argsort(-probabilities[predicted_frames, class_index], kind="stable")]
        class_frames.append(ordered_frames[:frames_per_class].tolist())
        frame_labels[ordered_frames[frames_per_class:]] = unselected_class_index

    # Every class still short of frames takes the unselected frames most probable for it, class by class
    for wanted_index, class_index in enumerate(wanted_class_indices):
        number_missing_frames = frames_per_class - len(class_frames[wanted_index])
        if number_missing_frames <= 0:
            continue
        free_frames = np.flatnonzero(frame_labels == unselected_class_index)
        completing_frames = free_frames[np.argsort(-probabilities[free_frames, class_index],
                                                   kind="stable")][:number_missing_frames]
        class_frames[wanted_index] += completing_frames.tolist()
        frame_labels[completing_frames] = class_index
    return order_selected_frames(probabilities=probabilities, class_frames=class_frames,
                                 wanted_class_indices=wanted_class_indices, frames_per_class=frames_per_class)


def decode_frame_selection(probabilities: np.ndarray, unselected_class_index: int, frames_per_class: int,
                           decoding: str) -> np.ndarray:
    """
    Chooses the frames of every wanted class of a sequence with the decoding of the run ('Selection.decoding'), the
    single entry point of the evaluation, the local inference and the Triton decoder.

    Args:
        probabilities (np.ndarray): The [T, K] class probabilities of every frame of the sequence.
        unselected_class_index (int): The class index of the unselected frames.
        frames_per_class (int): The number of frames N chosen for every wanted class.
        decoding (str): "hungarian" or "greedy".

    Returns:
        np.ndarray: The [W, N] int64 chosen frames of every wanted class, by decreasing probability, NO_FRAME where
            the sequence has fewer frames than selections.

    Raises:
        ValueError: If the decoding is unknown.
    """
    if decoding == "hungarian":
        return decode_hungarian_selection(probabilities=probabilities, unselected_class_index=unselected_class_index,
                                          frames_per_class=frames_per_class)
    if decoding == "greedy":
        return decode_greedy_selection(probabilities=probabilities, unselected_class_index=unselected_class_index,
                                       frames_per_class=frames_per_class)
    raise ValueError(f"The decoding must be one of {VALID_DECODINGS}, got {decoding!r}.")


def gather_frame_probabilities(probabilities: np.ndarray, frames: np.ndarray,
                               wanted_class_indices: List[int]) -> np.ndarray:
    """
    Reads the probability of every chosen (or candidate) frame for the class of its row, 0 where there is no frame.

    Args:
        probabilities (np.ndarray): The [T, K] class probabilities of every frame.
        frames (np.ndarray): The [W, M] frame indices of every wanted class, NO_FRAME for no frame.
        wanted_class_indices (List[int]): The class indices of the wanted classes, one per row.

    Returns:
        np.ndarray: The [W, M] float32 probabilities.
    """
    frame_probabilities = np.zeros(shape=frames.shape, dtype=np.float32)
    for wanted_index, class_index in enumerate(wanted_class_indices):
        valid_columns = frames[wanted_index] != NO_FRAME
        frame_probabilities[wanted_index, valid_columns] = probabilities[frames[wanted_index, valid_columns],
                                                                         class_index]
    return frame_probabilities


def find_candidate_frames(probabilities: np.ndarray, wanted_class_indices: List[int],
                          number_candidates: int) -> np.ndarray:
    """
    Lists the k frames most probable for every wanted class, before any decoding, so that a human checking a
    low-confidence selection, or the top-k metric of the evaluation, can see the frames competing for the class.

    Args:
        probabilities (np.ndarray): The [T, K] class probabilities of every frame.
        wanted_class_indices (List[int]): The class indices of the wanted classes.
        number_candidates (int): The number of candidates k of every class.

    Returns:
        np.ndarray: The [W, k] int64 candidate frames by decreasing probability, NO_FRAME after the last frame of a
            sequence shorter than k.
    """
    candidate_frames = np.full(shape=(len(wanted_class_indices), number_candidates), fill_value=NO_FRAME,
                               dtype=np.int64)
    for wanted_index, class_index in enumerate(wanted_class_indices):
        ordered_frames = np.argsort(-probabilities[:, class_index], kind="stable")[:number_candidates]
        candidate_frames[wanted_index, :len(ordered_frames)] = ordered_frames
    return candidate_frames


def compute_frame_ranks(probabilities: np.ndarray, wanted_class_indices: List[int]) -> np.ndarray:
    """
    Ranks every frame for every wanted class among the frames of its sequence ordered by their probability for that
    class: 1 for the most probable frame, a frame tied with others taking the best of their ranks. The rank of the
    true frames measures how far the model is from choosing them, before any decoding.

    Args:
        probabilities (np.ndarray): The [T, K] class probabilities of every frame.
        wanted_class_indices (List[int]): The class indices of the wanted classes.

    Returns:
        np.ndarray: The [T, W] int64 rank of every frame for every wanted class.
    """
    class_probabilities = probabilities[:, wanted_class_indices]
    return (class_probabilities[None, :, :] > class_probabilities[:, None, :]).sum(axis=1).astype(np.int64) + 1


def convert_selection_to_frame_labels(selected_frames: np.ndarray, wanted_class_indices: List[int],
                                      number_frames: int, unselected_class_index: int) -> np.ndarray:
    """
    Turns the chosen frames into the decoded class of every frame of the sequence, the chosen frames taking the class
    of their row and every other frame the unselected class, for the per-frame metrics of the decoding (unselected
    precision and recall, confusion matrix).

    Args:
        selected_frames (np.ndarray): The [W, N] chosen frames, NO_FRAME for no frame.
        wanted_class_indices (List[int]): The class indices of the wanted classes, one per row.
        number_frames (int): The number of frames T of the sequence.
        unselected_class_index (int): The class index of the unselected frames.

    Returns:
        np.ndarray: The [T] int64 decoded class of every frame.
    """
    frame_labels = np.full(shape=number_frames, fill_value=unselected_class_index, dtype=np.int64)
    for wanted_index, class_index in enumerate(wanted_class_indices):
        chosen_frames = selected_frames[wanted_index]
        frame_labels[chosen_frames[chosen_frames != NO_FRAME]] = class_index
    return frame_labels


def flag_low_confidence(selected_frames: np.ndarray, selected_probabilities: np.ndarray,
                        class_thresholds: List[Optional[float]]) -> np.ndarray:
    """
    Flags the chosen frames a human should check: those below the confidence threshold of their class (the
    probability from which the chosen frames of the class reached the target precision on the test split), all the
    frames of a class whose threshold is None (the target precision was never reached), and the missing frames.

    Args:
        selected_frames (np.ndarray): The [W, N] chosen frames, NO_FRAME for no frame.
        selected_probabilities (np.ndarray): The [W, N] probability of every chosen frame for its class.
        class_thresholds (List[Optional[float]]): The threshold of every wanted class, None when the target precision
            was never reached.

    Returns:
        np.ndarray: The [W, N] booleans, True for the low-confidence frames.
    """
    low_confidence = selected_frames == NO_FRAME
    for wanted_index, class_threshold in enumerate(class_thresholds):
        if class_threshold is None:
            low_confidence[wanted_index] = True
        else:
            low_confidence[wanted_index] |= selected_probabilities[wanted_index] < class_threshold
    return low_confidence


def run_frame_selection(probabilities: np.ndarray, decoding_settings: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Decodes the selection of one sequence for the deployed frame selector, the Triton decoder and the local inference
    returning the same outputs: the chosen frames of every wanted class with their probabilities and low-confidence
    flags, and its k most probable candidate frames with their probabilities.

    Args:
        probabilities (np.ndarray): The [T, K] class probabilities of every frame of the sequence.
        decoding_settings (Dict[str, Any]): The settings of the run (deployment/frame_selection_deployment.py):
            'decoding', 'frames_per_class', 'unselected_class_index', 'class_thresholds' (one per wanted class) and
            'number_candidates'.

    Returns:
        Dict[str, np.ndarray]: The [W, N] 'selected_frames' (int64), 'selected_probabilities' (float32) and
            'low_confidence' (bool), and the [W, k] 'candidate_frames' (int64) and 'candidate_probabilities'
            (float32).
    """
    wanted_class_indices = get_wanted_class_indices(number_classes=probabilities.shape[1],
                                                    unselected_class_index=decoding_settings["unselected_class_index"])

    # Chosen frames of the decoding of the run, with their probabilities and confidence
    selected_frames = decode_frame_selection(probabilities=probabilities,
                                             unselected_class_index=decoding_settings["unselected_class_index"],
                                             frames_per_class=decoding_settings["frames_per_class"],
                                             decoding=decoding_settings["decoding"])
    selected_probabilities = gather_frame_probabilities(probabilities=probabilities, frames=selected_frames,
                                                        wanted_class_indices=wanted_class_indices)
    low_confidence = flag_low_confidence(selected_frames=selected_frames, selected_probabilities=selected_probabilities,
                                         class_thresholds=decoding_settings["class_thresholds"])

    # Most probable frames of every class, whatever the decoding
    candidate_frames = find_candidate_frames(probabilities=probabilities, wanted_class_indices=wanted_class_indices,
                                             number_candidates=decoding_settings["number_candidates"])
    candidate_probabilities = gather_frame_probabilities(probabilities=probabilities, frames=candidate_frames,
                                                         wanted_class_indices=wanted_class_indices)
    return {"selected_frames": selected_frames, "selected_probabilities": selected_probabilities,
            "low_confidence": low_confidence, "candidate_frames": candidate_frames,
            "candidate_probabilities": candidate_probabilities}
