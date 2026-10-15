"""
Tests of the intersection over union of vision_training/models/detection/box_operations.py, used by the target
assignment of the detector and by its evaluation: an exact ratio stays exact and reaches its threshold, two empty
boxes give 0, and leading batch dimensions give the overlaps of every image at once.
"""

import torch

from vision_training.models.detection.box_operations import get_intersection_over_union


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
