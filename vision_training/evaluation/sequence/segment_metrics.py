"""
Segment metrics of the sequence labelling task (M2M), which score the predicted events instead of the frames: a segment
is a run of consecutive labelled frames of one class. The predictions of the labelled frames of the test sequences are
first grouped sequence by sequence, then every sequence gets its edit score (how many segments must be inserted,
removed or replaced to turn the predicted order of segments into the true one) and the segment F1 at several overlaps
(a predicted segment is right when it overlaps a true segment of its class enough). Used by the sequence labelling
trainer to evaluate a checkpoint, next to the frame metrics.
"""

import numpy as np
import torch

from typing import Any, Dict, List, Tuple

# Overlaps (intersection over union of the frames) from which a predicted segment matches a true segment of its class,
# each giving a segment F1 named after its percentage, e.g. 'Segment F1@50' for 0.50
SEGMENT_OVERLAP_THRESHOLDS = [0.10, 0.25, 0.50]


def group_sequence_predictions(sequence_indices: torch.Tensor, frame_indices: torch.Tensor,
                               probabilities: torch.Tensor, labels: torch.Tensor) -> List[Dict[str, np.ndarray]]:
    """
    Groups the labelled frames of the evaluated test sequences sequence by sequence, in frame order, for the segment
    metrics of the sequence labelling task, the predicted class of every frame being its most probable class.

    Args:
        sequence_indices (torch.Tensor): The [N] int64 index of the sequence of every evaluated labelled frame.
        frame_indices (torch.Tensor): The [N] int64 index of every frame in its sequence.
        probabilities (torch.Tensor): The [N, K] class probabilities of every frame.
        labels (torch.Tensor): The [N] int64 true class index of every frame.

    Returns:
        List[Dict[str, np.ndarray]]: One entry per sequence, in sequence index order, with the increasing
            'frame_indices' of its labelled frames, their 'true_classes' and their 'predicted_classes'.

    Raises:
        ValueError: If a frame of a sequence is given more than once.
    """
    if len(labels) == 0:
        return []

    # Frames sorted by sequence then by frame, every frame of a sequence being evaluated once
    sequence_column, frame_column = sequence_indices.numpy(), frame_indices.numpy()
    frame_order = np.lexsort(keys=(frame_column, sequence_column))
    sequence_column, frame_column = sequence_column[frame_order], frame_column[frame_order]
    if np.any((np.diff(sequence_column) == 0) & (np.diff(frame_column) == 0)):
        raise ValueError("A frame of a test sequence was evaluated more than once, every sequence being expected once.")
    predicted_classes = probabilities.argmax(dim=1).numpy()[frame_order]
    true_classes = labels.numpy()[frame_order]

    # Split the sorted frames at every change of sequence
    boundaries = [0] + (np.flatnonzero(np.diff(sequence_column)) + 1).tolist() + [len(sequence_column)]
    return [{"frame_indices": frame_column[start:end], "true_classes": true_classes[start:end],
             "predicted_classes": predicted_classes[start:end]}
            for start, end in zip(boundaries[:-1], boundaries[1:])]


def extract_segments(frame_classes: np.ndarray, frame_indices: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Cuts the labelled frames of a sequence into its segments, the events scored by the segment metrics: a segment
    ends at a change of class, and at a gap of unlabelled frames, whose class is unknown.

    Args:
        frame_classes (np.ndarray): The [N] class index of every labelled frame, true or predicted.
        frame_indices (np.ndarray): The [N] increasing index of every labelled frame in the sequence.

    Returns:
        List[Tuple[int, int, int]]: The class, first frame index and last frame index of every segment, in order.
    """
    if len(frame_classes) == 0:
        return []

    # A segment starts at the first frame, at a change of class and after a gap of unlabelled frames
    segment_starts = np.flatnonzero((np.diff(frame_classes) != 0) | (np.diff(frame_indices) != 1)) + 1
    boundaries = [0] + segment_starts.tolist() + [len(frame_classes)]
    return [(int(frame_classes[start]), int(frame_indices[start]), int(frame_indices[end - 1]))
            for start, end in zip(boundaries[:-1], boundaries[1:])]


def compute_edit_distance(predicted_classes: List[int], true_classes: List[int]) -> int:
    """
    Computes the Levenshtein distance between the predicted and true orders of segment classes of a sequence, the
    number of segments to insert, remove or replace, from which the edit score penalises a fragmented prediction.

    Args:
        predicted_classes (List[int]): The class of every predicted segment, in order.
        true_classes (List[int]): The class of every true segment, in order.

    Returns:
        int: The edit distance.
    """
    # Distances between the first predicted segments and the first true segments, one row of the table at a time
    previous_row = list(range(len(true_classes) + 1))
    for predicted_index, predicted_class in enumerate(predicted_classes, start=1):
        current_row = [predicted_index]
        for true_index, true_class in enumerate(true_classes, start=1):
            current_row.append(min(previous_row[true_index] + 1, current_row[true_index - 1] + 1,
                                   previous_row[true_index - 1] + int(predicted_class != true_class)))
        previous_row = current_row
    return previous_row[-1]


def count_segment_matches(predicted_segments: List[Tuple[int, int, int]], true_segments: List[Tuple[int, int, int]],
                          overlap_threshold: float) -> Tuple[int, int, int]:
    """
    Matches the predicted segments of a sequence with its true segments for the segment F1: every predicted segment,
    in order, is compared with the true segment of its class it overlaps most, and is right when that overlap reaches
    the threshold and the true segment was not matched by an earlier predicted segment, so that a fragmented
    prediction gets a single right segment per event.

    Args:
        predicted_segments (List[Tuple[int, int, int]]): The class, first and last frame of every predicted segment.
        true_segments (List[Tuple[int, int, int]]): The class, first and last frame of every true segment.
        overlap_threshold (float): The intersection over union from which a predicted segment can be right.

    Returns:
        Tuple[int, int, int]: The true positives, false positives and false negatives of the sequence.
    """
    matched_true_segments = [False] * len(true_segments)
    true_positives = 0
    for predicted_class, predicted_start, predicted_end in predicted_segments:
        # True segment of the same class overlapping the predicted segment most, the first one on a tie
        best_overlap, best_index = 0.0, None
        for true_index, (true_class, true_start, true_end) in enumerate(true_segments):
            intersection = min(predicted_end, true_end) - max(predicted_start, true_start) + 1
            if true_class != predicted_class or intersection <= 0:
                continue
            union = (predicted_end - predicted_start + 1) + (true_end - true_start + 1) - intersection
            if intersection / union > best_overlap:
                best_overlap, best_index = intersection / union, true_index

        # Right when it overlaps enough a true segment that no earlier predicted segment matched
        if best_index is not None and best_overlap >= overlap_threshold and not matched_true_segments[best_index]:
            matched_true_segments[best_index] = True
            true_positives += 1
    return true_positives, len(predicted_segments) - true_positives, len(true_segments) - true_positives


def compute_segment_metrics(sequences: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
    """
    Computes the segment metrics of the test sequences of an evaluated sequence labelling checkpoint: the edit score
    of every sequence (1 minus its edit distance over its larger number of segments), averaged over the sequences, and
    the segment F1 at every overlap of SEGMENT_OVERLAP_THRESHOLDS, from the matches counted over all the sequences.

    Args:
        sequences (List[Dict[str, np.ndarray]]): The 'frame_indices', 'true_classes' and 'predicted_classes' of the
            labelled frames of every sequence (group_sequence_predictions).

    Returns:
        Dict[str, Any]: The 'edit_score' float, the 'segment_f1' float of every overlap threshold, and the
            'number_sequences', 'number_true_segments' and 'number_predicted_segments' counts, the metrics being 0
            without sequence.
    """
    edit_scores = []
    match_counts = {overlap_threshold: np.zeros(shape=(3,), dtype=np.int64)
                    for overlap_threshold in SEGMENT_OVERLAP_THRESHOLDS}
    number_true_segments, number_predicted_segments = 0, 0
    for sequence in sequences:
        true_segments = extract_segments(frame_classes=sequence["true_classes"],
                                         frame_indices=sequence["frame_indices"])
        predicted_segments = extract_segments(frame_classes=sequence["predicted_classes"],
                                              frame_indices=sequence["frame_indices"])
        number_true_segments += len(true_segments)
        number_predicted_segments += len(predicted_segments)

        # Edit score of the order of the segment classes, every sequence holding at least one labelled frame
        edit_distance = compute_edit_distance(
            predicted_classes=[segment[0] for segment in predicted_segments],
            true_classes=[segment[0] for segment in true_segments])
        edit_scores.append(1.0 - edit_distance / max(len(predicted_segments), len(true_segments)))

        # True positives, false positives and false negatives of the sequence at every overlap
        for overlap_threshold in SEGMENT_OVERLAP_THRESHOLDS:
            match_counts[overlap_threshold] += count_segment_matches(
                predicted_segments=predicted_segments, true_segments=true_segments, overlap_threshold=overlap_threshold)

    # Segment F1 of the counts summed over the sequences, 0 when nothing is right
    segment_f1 = {}
    for overlap_threshold, (true_positives, false_positives, false_negatives) in match_counts.items():
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        segment_f1[overlap_threshold] = float(2.0 * precision * recall / (precision + recall)) \
            if precision + recall > 0 else 0.0
    return {"edit_score": float(np.mean(edit_scores)) if edit_scores else 0.0, "segment_f1": segment_f1,
            "number_sequences": len(sequences), "number_true_segments": number_true_segments,
            "number_predicted_segments": number_predicted_segments}
