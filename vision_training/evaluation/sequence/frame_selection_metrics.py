"""
Metrics of the frame selection task (SEL), computed at every evaluation of the frame selection trainer on the test
sequences: the selection accuracy of every wanted class (with its right and true frames in every sequence, for its
Korn-Graubard interval in recall_intervals.py), the share of sequences with every selection right and the precision and
recall of the unselected class for a decoding, the top-k accuracy and mean rank of the true frames
before any decoding, and the confidence thresholds of the chosen frames saved in 'class_thresholds.json'. The chosen
frames come from models/sequence/frame_selection_decoding.py.
"""

import numpy as np
import torch

from typing import Any, Dict, List, Optional, Tuple

from vision_training.evaluation.classification.threshold_finder import find_class_thresholds
from vision_training.models.sequence.frame_selection_decoding import NO_FRAME, compute_frame_ranks


def compute_share(numerator: float, denominator: float) -> float:
    """
    Divides two counts of the selection metrics, 0 when nothing was counted, so that every metric stays a number for
    the report and TensorBoard.

    Args:
        numerator (float): The counted successes.
        denominator (float): The counted cases.

    Returns:
        float: The share, 0 when the denominator is 0.
    """
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def compute_selection_metrics(selected_frames: List[np.ndarray], frame_labels: List[np.ndarray],
                              decoded_frame_labels: List[np.ndarray], wanted_class_indices: List[int],
                              unselected_class_index: int) -> Dict[str, Any]:
    """
    Scores the frames chosen by one decoding on the test sequences: the selection accuracy of every wanted class (the
    share of its true frames chosen for it, its recall), the overall selection accuracy, the share of sequences whose
    every selection is right, and the precision and recall of the unselected class (the frames left unchosen that
    should be, among the unchosen frames and among the truly unselected frames). The right and true frames of every
    wanted class in every sequence are returned too, for the Korn-Graubard interval over the sequences of its selection
    accuracy.

    Args:
        selected_frames (List[np.ndarray]): The [W, N] chosen frames of every sequence, NO_FRAME for no frame.
        frame_labels (List[np.ndarray]): The [T] true class of every frame of every sequence.
        decoded_frame_labels (List[np.ndarray]): The [T] decoded class of every frame of every sequence.
        wanted_class_indices (List[int]): The class indices of the wanted classes, one per row of the selections.
        unselected_class_index (int): The class index of the unselected frames.

    Returns:
        Dict[str, Any]: The 'class_accuracies' (one per wanted class, None without true frame), 'class_true_frames'
            (the true frames of every wanted class), the [S, W] int64 'sequence_class_right_frames' and
            'sequence_class_true_frames' of every sequence and wanted class, 'selection_accuracy',
            'fully_right_sequences', 'unselected_precision' and 'unselected_recall'.
    """
    sequence_class_right_frames = np.zeros(shape=(len(selected_frames), len(wanted_class_indices)), dtype=np.int64)
    sequence_class_true_frames = np.zeros(shape=(len(selected_frames), len(wanted_class_indices)), dtype=np.int64)
    number_fully_right_sequences = 0
    unselected_true_positives, unselected_predicted, unselected_true = 0, 0, 0

    for sequence_index, (sequence_selection, sequence_labels, sequence_decoded_labels) in enumerate(
            zip(selected_frames, frame_labels, decoded_frame_labels)):
        # True frames of every wanted class chosen for it
        sequence_fully_right = True
        for wanted_index, class_index in enumerate(wanted_class_indices):
            true_frames = set(np.flatnonzero(sequence_labels == class_index).tolist())
            chosen_frames = set(sequence_selection[wanted_index][sequence_selection[wanted_index] != NO_FRAME].tolist())
            sequence_class_right_frames[sequence_index, wanted_index] = len(true_frames & chosen_frames)
            sequence_class_true_frames[sequence_index, wanted_index] = len(true_frames)
            sequence_fully_right &= true_frames == chosen_frames
        number_fully_right_sequences += int(sequence_fully_right)

        # Frames left unchosen, against the truly unselected frames
        true_unselected = sequence_labels == unselected_class_index
        decoded_unselected = sequence_decoded_labels == unselected_class_index
        unselected_true_positives += int((true_unselected & decoded_unselected).sum())
        unselected_predicted += int(decoded_unselected.sum())
        unselected_true += int(true_unselected.sum())

    # Right and true frames of every wanted class over all the sequences
    class_right_frames = sequence_class_right_frames.sum(axis=0)
    class_true_frames = sequence_class_true_frames.sum(axis=0)
    return {"class_accuracies": [compute_share(numerator=right, denominator=true) if true > 0 else None
                                 for right, true in zip(class_right_frames, class_true_frames)],
            "class_true_frames": class_true_frames.tolist(),
            "sequence_class_right_frames": sequence_class_right_frames,
            "sequence_class_true_frames": sequence_class_true_frames,
            "selection_accuracy": compute_share(numerator=class_right_frames.sum(),
                                                denominator=class_true_frames.sum()),
            "fully_right_sequences": compute_share(numerator=number_fully_right_sequences,
                                                   denominator=len(selected_frames)),
            "unselected_precision": compute_share(numerator=unselected_true_positives,
                                                  denominator=unselected_predicted),
            "unselected_recall": compute_share(numerator=unselected_true_positives, denominator=unselected_true)}


def compute_ranking_metrics(probabilities: List[np.ndarray], frame_labels: List[np.ndarray],
                            wanted_class_indices: List[int], top_k: int) -> Dict[str, Any]:
    """
    Scores the probabilities of the model before any decoding: the rank of every true frame among the frames of its
    sequence ordered by their probability for its class (1 when it is the most probable), averaged per class and over
    every true frame, and the share of the true frames among the k frames most probable for their class (top-k
    selection accuracy), which tells how far a wrong selection was from the right frame.

    Args:
        probabilities (List[np.ndarray]): The [T, K] class probabilities of every frame of every sequence.
        frame_labels (List[np.ndarray]): The [T] true class of every frame of every sequence.
        wanted_class_indices (List[int]): The class indices of the wanted classes.
        top_k (int): The number of most probable frames k of every class ('evaluation.top_k').

    Returns:
        Dict[str, Any]: The 'class_top_k_accuracies' and 'class_mean_ranks' (one per wanted class, None without true
            frame), the 'top_k_accuracy' and the 'mean_rank' over every true frame, and the [T, W] 'frame_ranks' of
            every sequence.
    """
    class_ranks = [[] for _ in wanted_class_indices]
    frame_ranks = []
    for sequence_probabilities, sequence_labels in zip(probabilities, frame_labels):
        # Rank of every frame for every wanted class, read at the true frames of each class
        sequence_ranks = compute_frame_ranks(probabilities=sequence_probabilities,
                                             wanted_class_indices=wanted_class_indices)
        frame_ranks.append(sequence_ranks)
        for wanted_index, class_index in enumerate(wanted_class_indices):
            class_ranks[wanted_index] += sequence_ranks[sequence_labels == class_index, wanted_index].tolist()

    all_ranks = np.array([rank for ranks in class_ranks for rank in ranks], dtype=np.float64)
    return {"class_top_k_accuracies": [compute_share(numerator=np.sum(np.array(ranks) <= top_k),
                                                     denominator=len(ranks)) if ranks else None
                                       for ranks in class_ranks],
            "class_mean_ranks": [float(np.mean(ranks)) if ranks else None for ranks in class_ranks],
            "top_k_accuracy": compute_share(numerator=np.sum(all_ranks <= top_k), denominator=len(all_ranks)),
            "mean_rank": float(all_ranks.mean()) if len(all_ranks) else 0.0, "frame_ranks": frame_ranks}


def find_selection_thresholds(probabilities: List[np.ndarray], frame_labels: List[np.ndarray],
                              selected_frames: List[np.ndarray], wanted_class_indices: List[int],
                              target_precision: float) -> List[Optional[Tuple[float, float]]]:
    """
    Finds the confidence threshold of every wanted class: the probability from which the frames chosen for the class
    are true frames of it with the target precision keeping the most of them, the highest such probability when a
    lower one adds only wrong choices (find_class_thresholds), so that a chosen frame below it is flagged as low
    confidence (the sequence may hold no good frame of the class) for a human to check. The recall is the share of
    all the true frames of the class kept, i.e. chosen and above the threshold.

    Args:
        probabilities (List[np.ndarray]): The [T, K] class probabilities of every frame of every sequence.
        frame_labels (List[np.ndarray]): The [T] true class of every frame of every sequence.
        selected_frames (List[np.ndarray]): The [W, N] chosen frames of every sequence, NO_FRAME for no frame.
        wanted_class_indices (List[int]): The class indices of the wanted classes, one per row of the selections.
        target_precision (float): The precision the chosen frames above the threshold must reach.

    Returns:
        List[Optional[Tuple[float, float]]]: For every wanted class, its threshold and the share of its true frames
            kept, or None if no threshold reaches the target precision.
    """
    class_thresholds = []
    for wanted_index, class_index in enumerate(wanted_class_indices):
        # Probability of every frame chosen for the class, and whether it is a true frame of the class
        chosen_probabilities, chosen_true, number_true_frames = [], [], 0
        for sequence_probabilities, sequence_labels, sequence_selection in zip(probabilities, frame_labels,
                                                                               selected_frames):
            chosen_frames = sequence_selection[wanted_index][sequence_selection[wanted_index] != NO_FRAME]
            chosen_probabilities += sequence_probabilities[chosen_frames, class_index].tolist()
            chosen_true += (sequence_labels[chosen_frames] == class_index).tolist()
            number_true_frames += int((sequence_labels == class_index).sum())

        # Threshold of the chosen frames, its recall among the true chosen frames brought to all the true frames
        class_threshold = find_class_thresholds(
            probabilities=torch.tensor(data=chosen_probabilities, dtype=torch.float64)[:, None],
            positive_masks=torch.tensor(data=chosen_true, dtype=torch.bool)[:, None],
            target_precision=target_precision)[0]
        if class_threshold is None:
            class_thresholds.append(None)
            continue
        class_thresholds.append((class_threshold[0], compute_share(numerator=class_threshold[1] * sum(chosen_true),
                                                                   denominator=number_true_frames)))
    return class_thresholds
