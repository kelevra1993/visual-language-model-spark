"""
Tests of vision_training/models/detection/box_operations.py, used by the target assignment of the detector, its heads
and its evaluation: the intersection over union (an exact ratio stays exact and reaches its threshold, two empty boxes
give 0, leading batch dimensions give the overlaps of every image at once), the normalized centre distance and the
Distance-IoU of the CIoU loss and of DIoU-NMS (equal to those of torchvision, with an exponent, for every image at
once), and the box regression weights (the targets of turn_boxes_to_transformation_targets are the offsets scaled by
BOX_REGRESSION_WEIGHTS, and apply_regression_predictions divides them out, so that decoding the targets gives back the
ground truth boxes), and the shape check of the boxes (a last dimension other than 4 raises a ValueError instead of
ending the process).
"""

import math
import pytest
import torch

from torchvision.ops import box_iou, distance_box_iou

from vision_training.models.detection.box_operations import (BOX_REGRESSION_WEIGHTS, apply_regression_predictions,
                                                             get_box_dimensions_and_centers,
                                                             get_distance_intersection_over_union,
                                                             get_intersection_over_union,
                                                             get_normalized_centre_distance)
from vision_training.models.detection.target_assignment import turn_boxes_to_transformation_targets


def test_exact_overlap_ratio_reaches_its_threshold() -> None:
    """
    Verifies that a detection covering three quarters of a ground truth box gives an overlap of exactly 0.75 in
    float32 and float64, so that it counts at the 0.75 threshold of the mAP@0.75, and that a box overlapping nothing
    and two empty boxes give 0.
    """
    for dtype in [torch.float32, torch.float64]:
        ground_truth = torch.tensor(data=[[0.0, 0.0, 100.0, 100.0]], dtype=dtype)
        detections = torch.tensor(data=[[0.0, 0.0, 100.0, 75.0], [200.0, 200.0, 220.0, 220.0]], dtype=dtype)
        overlaps = get_intersection_over_union(boxes_1=detections, boxes_2=ground_truth)
        assert overlaps.tolist() == [[0.75], [0.0]] and bool(overlaps[0, 0] >= 0.75)

    empty_boxes = torch.zeros(size=(1, 4))
    assert get_intersection_over_union(boxes_1=empty_boxes, boxes_2=empty_boxes).tolist() == [[0.0]]


def test_batched_boxes_give_the_overlaps_of_every_image() -> None:
    """
    Verifies that boxes with a leading batch dimension give, for every image, the overlaps of its own boxes, equal to
    those computed image by image, and that the second set of boxes broadcasts over the batch.
    """
    generator = torch.Generator().manual_seed(0)
    corners = torch.rand(size=(3, 5, 2), generator=generator) * 100
    boxes_1 = torch.cat(tensors=[corners, corners + 10 + torch.rand(size=(3, 5, 2), generator=generator) * 50], dim=2)
    corners = torch.rand(size=(3, 7, 2), generator=generator) * 100
    boxes_2 = torch.cat(tensors=[corners, corners + 10 + torch.rand(size=(3, 7, 2), generator=generator) * 50], dim=2)

    batched_overlaps = get_intersection_over_union(boxes_1=boxes_1, boxes_2=boxes_2)
    assert batched_overlaps.shape == (3, 5, 7)
    for image_index in range(3):
        image_overlaps = get_intersection_over_union(boxes_1=boxes_1[image_index], boxes_2=boxes_2[image_index])
        assert torch.equal(input=batched_overlaps[image_index], other=image_overlaps)

    # Anchors shared by every image broadcast over the batch
    shared_overlaps = get_intersection_over_union(boxes_1=boxes_1, boxes_2=boxes_2[0])
    assert torch.equal(input=shared_overlaps[2], other=get_intersection_over_union(boxes_1=boxes_1[2],
                                                                                   boxes_2=boxes_2[0]))


def test_normalized_centre_distance_of_hand_made_pairs() -> None:
    """
    Verifies the distance term of the Distance-IoU on pairs computed by hand: a box and a wider box sharing its right
    edge, whose centres are 4.5 pixels apart in an enclosing box of 19 x 10 pixels (20.25 / 461), boxes sharing their
    centre and two boxes reduced to the same point (0, the epsilon keeping it finite), the same pair scaled by 10
    (unchanged), and the broadcasting of pairs given as [N, 1, 4] and [1, M, 4].
    """
    first_box = torch.tensor(data=[0.0, 0.0, 10.0, 10.0], dtype=torch.float64)
    wider_box = torch.tensor(data=[-9.0, 0.0, 10.0, 10.0], dtype=torch.float64)
    assert float(get_normalized_centre_distance(boxes_1=first_box, boxes_2=wider_box)) == 20.25 / (461.0 + 1e-7)
    assert math.isclose(float(get_normalized_centre_distance(boxes_1=first_box * 10, boxes_2=wider_box * 10)),
                        20.25 / 461.0, rel_tol=1e-9)

    # Boxes sharing their centre, and two boxes reduced to the same point
    centred_box = torch.tensor(data=[-5.0, 2.0, 15.0, 8.0], dtype=torch.float64)
    point_box = torch.tensor(data=[3.0, 3.0, 3.0, 3.0], dtype=torch.float64)
    assert float(get_normalized_centre_distance(boxes_1=first_box, boxes_2=centred_box)) == 0.0
    assert float(get_normalized_centre_distance(boxes_1=point_box, boxes_2=point_box)) == 0.0

    # Every pair of two sets, from boxes unsqueezed to [N, 1, 4] and [1, M, 4]
    boxes_1 = torch.stack(tensors=[first_box, wider_box])
    boxes_2 = torch.stack(tensors=[centred_box, first_box, point_box])
    pair_distances = get_normalized_centre_distance(boxes_1=boxes_1.unsqueeze(dim=1), boxes_2=boxes_2.unsqueeze(dim=0))
    assert pair_distances.shape == (2, 3)
    for first_index in range(2):
        for second_index in range(3):
            assert pair_distances[first_index, second_index] == get_normalized_centre_distance(
                boxes_1=boxes_1[first_index], boxes_2=boxes_2[second_index])


def test_distance_intersection_over_union_matches_torchvision() -> None:
    """
    Verifies that the Distance-IoU matrix with the exponent 1.0 equals the distance_box_iou of torchvision on random
    boxes, that another exponent raises the distance term of torchvision to that power, and that leading batch
    dimensions give the matrices of every image at once.
    """
    generator = torch.Generator().manual_seed(0)
    corners = torch.rand(size=(3, 6, 2), generator=generator, dtype=torch.float64) * 100
    boxes_1 = torch.cat(tensors=[corners, corners + 5 + torch.rand(size=(3, 6, 2), generator=generator,
                                                                   dtype=torch.float64) * 60], dim=2)
    corners = torch.rand(size=(3, 4, 2), generator=generator, dtype=torch.float64) * 100
    boxes_2 = torch.cat(tensors=[corners, corners + 5 + torch.rand(size=(3, 4, 2), generator=generator,
                                                                   dtype=torch.float64) * 60], dim=2)

    for exponent in [1.0, 0.6]:
        batched_criteria = get_distance_intersection_over_union(boxes_1=boxes_1, boxes_2=boxes_2,
                                                                distance_exponent=exponent)
        assert batched_criteria.shape == (3, 6, 4)
        for image_index in range(3):
            # Distance term of torchvision, the IoU minus its Distance-IoU, raised to the exponent
            reference_iou = box_iou(boxes1=boxes_1[image_index], boxes2=boxes_2[image_index])
            reference_distance = reference_iou - distance_box_iou(boxes1=boxes_1[image_index],
                                                                  boxes2=boxes_2[image_index])
            assert torch.allclose(input=batched_criteria[image_index],
                                  other=reference_iou - reference_distance.pow(exponent=exponent), atol=1e-12)
            assert torch.equal(input=batched_criteria[image_index], other=get_distance_intersection_over_union(
                boxes_1=boxes_1[image_index], boxes_2=boxes_2[image_index], distance_exponent=exponent))


def test_regression_targets_are_weighted_and_decoded_back() -> None:
    """
    Verifies that a ground truth box shifted by a tenth of the reference width and twice as wide gives the targets
    (0.1, 0, log 2, 0), its offsets left unscaled by the weights (1, 1, 1, 1), and that decoding the targets of random
    boxes, in float32 and float64, gives back their ground truth boxes.
    """
    assert BOX_REGRESSION_WEIGHTS == (1.0, 1.0, 1.0, 1.0)

    # A reference box of 100 x 50 pixels and its ground truth, centred 10 pixels to the right and twice as wide
    reference_boxes = torch.tensor(data=[[100.0, 100.0, 200.0, 150.0]], dtype=torch.float64)
    ground_truth_boxes = torch.tensor(data=[[60.0, 100.0, 260.0, 150.0]], dtype=torch.float64)
    targets = turn_boxes_to_transformation_targets(ground_truth_boxes=ground_truth_boxes,
                                                   predicted_boxes=reference_boxes)
    expected_targets = torch.tensor(data=[[0.1, 0.0, math.log(2.0), 0.0]], dtype=torch.float64)
    assert torch.allclose(input=targets, other=expected_targets, atol=1e-6)

    # Decoding the weighted targets of random boxes gives back the ground truth boxes
    generator = torch.Generator().manual_seed(0)
    for dtype in [torch.float32, torch.float64]:
        corners = torch.rand(size=(2, 6, 2), generator=generator, dtype=dtype) * 400
        reference_boxes = torch.cat(tensors=[corners, corners + 8 + torch.rand(size=(2, 6, 2), generator=generator,
                                                                                dtype=dtype) * 100], dim=2)
        corners = torch.rand(size=(2, 6, 2), generator=generator, dtype=dtype) * 400
        ground_truth_boxes = torch.cat(tensors=[corners, corners + 8 + torch.rand(size=(2, 6, 2), generator=generator,
                                                                                   dtype=dtype) * 100], dim=2)
        targets = turn_boxes_to_transformation_targets(ground_truth_boxes=ground_truth_boxes,
                                                       predicted_boxes=reference_boxes)
        decoded_boxes = apply_regression_predictions(regression_predictions=targets.unsqueeze(dim=2),
                                                     boxes=reference_boxes)
        assert decoded_boxes.shape == (2, 6, 1, 4) and decoded_boxes.dtype == dtype
        assert torch.allclose(input=decoded_boxes[:, :, 0], other=ground_truth_boxes, atol=1e-2)


def test_boxes_of_a_wrong_shape_raise_a_value_error() -> None:
    """
    Verifies that boxes whose last dimension is not 4 raise a ValueError naming the received shape, both directly and
    through apply_regression_predictions, instead of ending the process with exit() and status 0, which no except
    Exception catches and which schedulers take for a success.
    """
    with pytest.raises(expected_exception=ValueError, match=r"Last Dimension Of 5 Instead Of 4"):
        get_box_dimensions_and_centers(boxes=torch.zeros(size=(2, 5)))
    with pytest.raises(expected_exception=ValueError, match=r"Shape \[2, 5\]"):
        apply_regression_predictions(regression_predictions=torch.zeros(size=(2, 1, 4)), boxes=torch.zeros(size=(2, 5)))
