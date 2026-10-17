"""
Grouping of the keypoint candidates of the bottom-up keypoint detector (HigherHRNet) into objects, the last step of the
keypoint detection after the model, which gives the candidates of every heatmap channel with their score and tag
(extract_keypoint_candidates of keypoint_detector.py, also the output of the exported model). The keypoint trainer
groups them for the evaluation and the sample images, and the deployment after the exported model. The tags of the
keypoints of one object are close to each other, and far from those of the other objects of the same class (the pull
and push losses of losses/keypoint_losses.py), so the candidates of every keypoint of a class are matched to the
objects formed by the previous keypoints of the class by the distance of their tag to the mean tag of every object.
"""

import numpy as np

from scipy.optimize import linear_sum_assignment
from typing import Dict, List

# Cost of one unit of tag distance in the matching of the candidates to the objects: the distances are rounded to the
# unit, so that the matching follows the tags first and, between objects of equally close tags, the higher scores
TAG_DISTANCE_COST = 100.0

# Cost of the extra columns of the matching when a keypoint has more candidates than there are objects, the candidates
# matched to them starting new objects
UNMATCHED_COST = 1e10


def match_candidates_to_objects(candidate_tags: np.ndarray, candidate_scores: np.ndarray,
                                object_tags: List[List[float]], tag_threshold: float) -> List[int]:
    """
    Matches the candidates of one keypoint of a class to the objects already formed by the previous keypoints of the
    class, for group_keypoint_candidates: one candidate per object at most (Hungarian matching on the rounded distance
    of the candidate tag to the mean tag of the object, minus the candidate score), a match being kept when that
    distance is below tag_threshold.

    Args:
        candidate_tags (np.ndarray): The [n] tags of the candidates.
        candidate_scores (np.ndarray): The [n] heatmap scores of the candidates.
        object_tags (List[List[float]]): The tags of the keypoints of every object formed so far.
        tag_threshold (float): The largest tag distance between a candidate and the mean tag of its object.

    Returns:
        List[int]: The index of the object of every candidate, -1 for the candidates starting a new object.
    """
    number_candidates, number_objects = len(candidate_tags), len(object_tags)
    object_assignments = [-1] * number_candidates
    if number_objects == 0:
        return object_assignments

    # Distance of every candidate tag to the mean tag of every object, and the matching cost
    mean_tags = np.array([np.mean(tags) for tags in object_tags], dtype=np.float64)
    tag_distances = np.abs(candidate_tags[:, None].astype(np.float64) - mean_tags[None, :])
    costs = np.round(tag_distances) * TAG_DISTANCE_COST - candidate_scores[:, None].astype(np.float64)

    # Extra columns so that every candidate is matched, those matched to them starting new objects
    if number_candidates > number_objects:
        extra_columns = np.full(shape=(number_candidates, number_candidates - number_objects),
                                fill_value=UNMATCHED_COST)
        costs = np.concatenate((costs, extra_columns), axis=1)

    # One object per candidate at most, the matches with too distant tags starting new objects
    candidate_indices, object_indices = linear_sum_assignment(cost_matrix=costs)
    for candidate_index, object_index in zip(candidate_indices, object_indices, strict=True):
        if object_index < number_objects and tag_distances[candidate_index, object_index] < tag_threshold:
            object_assignments[candidate_index] = int(object_index)
    return object_assignments


def group_keypoint_candidates(candidate_positions: np.ndarray, candidate_scores: np.ndarray,
                              candidate_tags: np.ndarray, channel_table: np.ndarray, keypoint_threshold: float,
                              tag_threshold: float, object_score_threshold: float) -> Dict[str, np.ndarray]:
    """
    Groups the keypoint candidates of one image into objects, class by class. The candidates of every keypoint of a
    class above keypoint_threshold are taken in the order of the keypoints of the class and matched to the objects
    formed by the previous keypoints (match_candidates_to_objects), the unmatched ones starting new objects. The score
    of an object is the sum of the scores of its keypoints divided by the number of keypoints of its class, so that an
    object missing keypoints scores lower; the objects below object_score_threshold are dropped (0 keeps them all, for
    the evaluation).

    Args:
        candidate_positions (np.ndarray): The [C, N, 2] (x, y) candidates of every channel in image pixels.
        candidate_scores (np.ndarray): The [C, N] heatmap scores of the candidates.
        candidate_tags (np.ndarray): The [C, N] tags of the candidates.
        channel_table (np.ndarray): The [number of classes + 1, K] channel of every keypoint position of every class,
            -1 for the padding positions and the classes without keypoints (build_keypoint_channel_table).
        keypoint_threshold (float): The lowest score of a candidate kept for the grouping.
        tag_threshold (float): The largest tag distance between a keypoint and the mean tag of its object.
        object_score_threshold (float): The lowest score of an object kept.

    Returns:
        Dict[str, np.ndarray]: The 'labels' [D] int64 class, 'keypoints' [D, K, 2] float32 (x, y) in image pixels (NaN
            for the keypoints not found), 'keypoint_scores' [D, K] float32 (0 for the keypoints not found) and 'scores'
            [D] float32 of the D objects found, by decreasing score.
    """
    maximum_keypoints = channel_table.shape[1]
    kept_candidates = candidate_scores > keypoint_threshold
    labels, keypoints, keypoint_scores, scores = [], [], [], []

    # Every class with keypoints and at least one candidate above the threshold
    for class_index in range(channel_table.shape[0]):
        class_channels = channel_table[class_index][channel_table[class_index] >= 0]
        if len(class_channels) == 0 or not kept_candidates[class_channels].any():
            continue
        object_tags, object_keypoints, object_keypoint_scores = [], [], []

        # Candidates of every keypoint of the class matched to the objects formed by the previous keypoints
        for keypoint_index, channel in enumerate(class_channels):
            kept = kept_candidates[channel]
            positions, channel_scores, tags = (candidate_positions[channel][kept], candidate_scores[channel][kept],
                                               candidate_tags[channel][kept])
            object_assignments = match_candidates_to_objects(candidate_tags=tags, candidate_scores=channel_scores,
                                                             object_tags=object_tags, tag_threshold=tag_threshold)

            # Matched candidates added to their object, the others starting new objects
            for candidate_index, object_index in enumerate(object_assignments):
                if object_index < 0:
                    object_index = len(object_tags)
                    object_tags.append([])
                    object_keypoints.append(np.full(shape=(maximum_keypoints, 2), fill_value=np.nan,
                                                    dtype=np.float32))
                    object_keypoint_scores.append(np.zeros(shape=(maximum_keypoints,), dtype=np.float32))
                object_tags[object_index].append(float(tags[candidate_index]))
                object_keypoints[object_index][keypoint_index] = positions[candidate_index]
                object_keypoint_scores[object_index][keypoint_index] = channel_scores[candidate_index]

        # Objects of the class with their score over all the keypoints of the class
        for single_object_keypoints, single_object_scores in zip(object_keypoints, object_keypoint_scores,
                                                                  strict=True):
            object_score = float(single_object_scores.sum()) / len(class_channels)
            if object_score >= object_score_threshold:
                labels.append(class_index)
                keypoints.append(single_object_keypoints)
                keypoint_scores.append(single_object_scores)
                scores.append(object_score)

    # Objects of every class by decreasing score
    order = np.argsort(-np.array(scores, dtype=np.float32), kind="stable")
    return {"labels": np.array(labels, dtype=np.int64)[order],
            "keypoints": np.array(keypoints, dtype=np.float32).reshape(-1, maximum_keypoints, 2)[order],
            "keypoint_scores": np.array(keypoint_scores, dtype=np.float32).reshape(-1, maximum_keypoints)[order],
            "scores": np.array(scores, dtype=np.float32)[order]}
