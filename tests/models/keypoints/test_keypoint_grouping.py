"""
Tests of vision_training/models/keypoints/keypoint_grouping.py, the grouping of the keypoint candidates of the bottom-up
keypoint detector into objects used by the keypoint trainer and by the clients of the exported model: the matching of
the candidates of a keypoint to the objects formed so far (by tag distance first, then by score, one candidate per
object, the too distant tags starting new objects), and the grouping of the candidates of one image (two triangles
told apart by their tags, a rectangle grouped on its own whatever its tags, a candidate of an outlying tag starting an
object of its own, the missing keypoints at NaN, the object scores over all the keypoints of the class, the order by
decreasing score, the object score threshold, and an image without any candidate above the keypoint threshold).
"""

import numpy as np

from typing import Tuple

from vision_training.models.keypoints.keypoint_grouping import (group_keypoint_candidates,
                                                                match_candidates_to_objects)

# Channels of the classes of the keypoint fixture: the 3 keypoints of the triangle (class 1), then the 4 of the
# rectangle (class 2)
CHANNEL_TABLE = np.array([[-1, -1, -1, -1], [0, 1, 2, -1], [3, 4, 5, 6]])

# Keypoints of two triangles and a rectangle, and the score and tag of the candidates of every object
TRIANGLE_A = [[10.0, 10.0], [5.0, 20.0], [15.0, 20.0]]
TRIANGLE_B = [[40.0, 10.0], [35.0, 20.0], [45.0, 20.0]]
RECTANGLE = [[20.0, 40.0], [50.0, 40.0], [50.0, 60.0], [20.0, 60.0]]


def build_candidates() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds the candidates of 3 per channel the detector would give for an image of two triangles and a rectangle, by
    decreasing score in every channel: triangle A (score 0.9, tag about 0), triangle B (score 0.6, tag about 5, except
    its right keypoint, whose outlying tag of 20 detaches it), the rectangle (score 0.8, tag about 0 like triangle A),
    and below them candidates under the keypoint threshold.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The [7, 3, 2] positions, [7, 3] scores and [7, 3] tags.
    """
    positions = np.zeros(shape=(7, 3, 2), dtype=np.float32)
    scores = np.full(shape=(7, 3), fill_value=0.05, dtype=np.float32)
    tags = np.zeros(shape=(7, 3), dtype=np.float32)

    # Candidates of the two triangles in the channels 0 to 2, the right keypoint of triangle B of an outlying tag
    for keypoint_index in range(3):
        positions[keypoint_index, 0], scores[keypoint_index, 0] = TRIANGLE_A[keypoint_index], 0.9
        positions[keypoint_index, 1], scores[keypoint_index, 1] = TRIANGLE_B[keypoint_index], 0.6
        tags[keypoint_index, :2] = [0.1 * keypoint_index, 5.0 + 0.1 * keypoint_index]
    tags[2, 1] = 20.0

    # Candidates of the rectangle in the channels 3 to 6, with the tags of triangle A
    for keypoint_index in range(4):
        positions[3 + keypoint_index, 0], scores[3 + keypoint_index, 0] = RECTANGLE[keypoint_index], 0.8
    return positions, scores, tags


def test_candidates_are_matched_by_tag_then_score() -> None:
    """
    Verifies that without objects every candidate starts a new one, that candidates go to the object of the closest
    mean tag, that between two candidates of equally close rounded tags the higher score gets the object and the other
    starts a new one, and that a tag farther than the threshold starts a new object.
    """
    no_objects = match_candidates_to_objects(candidate_tags=np.array([0.0, 1.0]), candidate_scores=np.array([0.5, 0.5]),
                                             object_tags=[], tag_threshold=1.0)
    assert no_objects == [-1, -1]

    # The candidates go to the object of the closest mean tag, the objects of tags 0 and 5 crossed
    crossed = match_candidates_to_objects(candidate_tags=np.array([5.2, -0.1]), candidate_scores=np.array([0.5, 0.5]),
                                          object_tags=[[0.0, 0.2], [5.0]], tag_threshold=1.0)
    assert crossed == [1, 0]

    # Equally close rounded tags: the higher score gets the object
    competing = match_candidates_to_objects(candidate_tags=np.array([0.2, 0.4]), candidate_scores=np.array([0.5, 0.9]),
                                            object_tags=[[0.0]], tag_threshold=1.0)
    assert competing == [-1, 0]

    # A tag beyond the threshold starts a new object even when it is the only candidate
    distant = match_candidates_to_objects(candidate_tags=np.array([1.5]), candidate_scores=np.array([0.9]),
                                          object_tags=[[0.0]], tag_threshold=1.0)
    assert distant == [-1]


def test_candidates_of_an_image_are_grouped_into_objects() -> None:
    """
    Verifies the objects of the image of build_candidates: triangle A (score 0.9), the rectangle grouped on its own
    despite the tags of triangle A (score 0.8), triangle B without its right keypoint (score 2 x 0.6 / 3 = 0.4) and its
    right keypoint alone (score 0.6 / 3 = 0.2), by decreasing score, the missing keypoints and the padding position of
    the triangles at NaN with a score of 0, and the object score threshold dropping the lone keypoint.
    """
    positions, scores, tags = build_candidates()
    objects = group_keypoint_candidates(candidate_positions=positions, candidate_scores=scores, candidate_tags=tags,
                                        channel_table=CHANNEL_TABLE, keypoint_threshold=0.1, tag_threshold=1.0,
                                        object_score_threshold=0.0)
    assert objects["labels"].tolist() == [1, 2, 1, 1]
    assert np.allclose(objects["scores"], [0.9, 0.8, 0.4, 0.2])
    assert objects["keypoints"].shape == (4, 4, 2) and objects["keypoint_scores"].shape == (4, 4)

    # Keypoints of every object, NaN where not found, with the scores of their candidates
    expected_keypoints = [TRIANGLE_A + [[np.nan, np.nan]], RECTANGLE, TRIANGLE_B[:2] + [[np.nan, np.nan]] * 2,
                          [[np.nan, np.nan]] * 2 + [TRIANGLE_B[2], [np.nan, np.nan]]]
    assert np.allclose(objects["keypoints"], np.array(expected_keypoints, dtype=np.float32), equal_nan=True)
    assert np.allclose(objects["keypoint_scores"], [[0.9, 0.9, 0.9, 0.0], [0.8, 0.8, 0.8, 0.8], [0.6, 0.6, 0.0, 0.0],
                                                    [0.0, 0.0, 0.6, 0.0]])

    # The object score threshold drops the lone keypoint
    thresholded = group_keypoint_candidates(candidate_positions=positions, candidate_scores=scores,
                                            candidate_tags=tags, channel_table=CHANNEL_TABLE, keypoint_threshold=0.1,
                                            tag_threshold=1.0, object_score_threshold=0.3)
    assert thresholded["labels"].tolist() == [1, 2, 1]


def test_image_without_candidates_gives_no_object() -> None:
    """
    Verifies that an image whose candidates all lie below the keypoint threshold gives no object, with the empty
    arrays keeping their shapes and types for the evaluation.
    """
    positions, scores, tags = build_candidates()
    objects = group_keypoint_candidates(candidate_positions=positions, candidate_scores=scores, candidate_tags=tags,
                                        channel_table=CHANNEL_TABLE, keypoint_threshold=0.95, tag_threshold=1.0,
                                        object_score_threshold=0.0)
    assert objects["labels"].shape == (0,) and objects["labels"].dtype == np.int64
    assert objects["keypoints"].shape == (0, 4, 2) and objects["keypoint_scores"].shape == (0, 4)
    assert objects["scores"].shape == (0,) and objects["scores"].dtype == np.float32
