"""
Tests of vision_training/models/detection/box_operations.py, used by the target assignment of the detector, its heads
and its evaluation: the intersection over union (an exact ratio stays exact and reaches its threshold, two empty boxes
give 0, leading batch dimensions give the overlaps of every image at once), and the box regression weights (the
targets of turn_boxes_to_transformation_targets are the offsets scaled by BOX_REGRESSION_WEIGHTS, and
apply_regression_predictions divides them out, so that decoding the targets gives back the ground truth boxes).
"""

import math
import torch

from vision_training.models.detection.box_operations import (BOX_REGRESSION_WEIGHTS, apply_regression_predictions,
                                                             get_intersection_over_union)
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
