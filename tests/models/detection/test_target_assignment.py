"""
Tests of the batched training target assignment of vision_training/models/detection/target_assignment.py: the
foreground, background and ignored anchors of every image (an image without object included), the best anchor of
every object made foreground with both fallback rules, the proposal targets of images with different object counts,
the number, kind and uniformity of the sampled targets, the ground truth boxes added to the proposals of the
detection heads giving a positive proposal per object, and the anchor matching of a crowded batch in chunks of objects
giving the targets of a single pass, compiled on the GPU as eager on the CPU.
"""

import functools
import pytest
import torch

from typing import List, Tuple

import vision_training.models.detection.target_assignment as target_assignment

from vision_training.models.detection.box_operations import get_intersection_over_union
from vision_training.models.detection.target_assignment import (add_ground_truth_proposals,
                                                                batch_assign_targets_to_anchors,
                                                                batch_assign_targets_to_proposals,
                                                                sample_positive_and_negative_training_targets,
                                                                turn_boxes_to_transformation_targets)

# Anchor thresholds of the tests: background from 0.1 to 0.3, foreground from 0.7
BACKGROUND_THRESHOLD = {"min": 0.1, "max": 0.3}
FOREGROUND_THRESHOLD = {"min": 0.7, "max": 1.0}


def test_anchor_labels_of_every_image() -> None:
    """
    Verifies the labels and target boxes of the anchors of a batch: foreground from an overlap of 0.7, ignored
    between 0.3 and 0.7 and below 0.1, background between 0.1 and 0.3, every non-foreground anchor targeting the
    first object of its image, and an image without object whose anchors are all background with empty targets.
    """
    ground_truth = torch.tensor(data=[[0.0, 0.0, 100.0, 100.0]])
    anchors = torch.tensor(data=[[0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 100.0, 40.0], [0.0, 0.0, 100.0, 20.0],
                                 [300.0, 300.0, 400.0, 400.0]])
    target_boxes, labels = batch_assign_targets_to_anchors(
        batched_ground_truth_boxes=[ground_truth, torch.zeros(size=(0, 4))],
        batched_anchors=anchors.unsqueeze(dim=0).expand(2, -1, -1), background_iou_threshold=BACKGROUND_THRESHOLD,
        foreground_iou_threshold=FOREGROUND_THRESHOLD)
    assert labels.dtype == torch.float32 and labels.tolist() == [[1.0, -1.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]]
    assert target_boxes[0].tolist() == [[0.0, 0.0, 100.0, 100.0]] * 4
    assert target_boxes[1].tolist() == [[0.0, 0.0, 0.0, 0.0]] * 4


def test_best_anchor_of_every_object_is_foreground() -> None:
    """
    Verifies that the anchor overlapping an object the most is foreground even below the foreground threshold, and
    that it targets the object it overlaps the most when that is another object than the one that made it foreground,
    as the region proposal network of torchvision.
    """
    # The first object's best anchor (overlap 0.4) overlaps the second object more (0.75)
    ground_truth = torch.tensor(data=[[0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 100.0, 30.0]])
    anchors = torch.tensor(data=[[0.0, 0.0, 100.0, 40.0], [0.0, 0.0, 100.0, 30.0], [300.0, 300.0, 400.0, 400.0]])
    target_boxes, labels = batch_assign_targets_to_anchors(
        batched_ground_truth_boxes=[ground_truth], batched_anchors=anchors.unsqueeze(dim=0),
        background_iou_threshold=BACKGROUND_THRESHOLD, foreground_iou_threshold=FOREGROUND_THRESHOLD)
    assert labels.tolist() == [[1.0, 1.0, -1.0]]
    assert target_boxes[0, 0].tolist() == [0.0, 0.0, 100.0, 30.0]


def test_proposal_targets_of_images_with_different_object_counts() -> None:
    """
    Verifies the targets of the proposals of a batch whose images have different numbers of objects: every proposal
    targets the object of its own image it overlaps enough, with its label and index, the padding of the image with
    fewer objects never being matched.
    """
    boxes = [torch.tensor(data=[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]]),
             torch.tensor(data=[[20.0, 20.0, 40.0, 40.0]])]
    labels = [torch.tensor(data=[3, 5]), torch.tensor(data=[8])]
    proposals = torch.tensor(data=[[21.0, 21.0, 40.0, 40.0], [0.0, 0.0, 9.0, 10.0], [0.0, 0.0, 0.0, 0.0]])
    target_boxes, target_labels, matched_indices = batch_assign_targets_to_proposals(
        batched_ground_truth_boxes=boxes, batched_proposals=proposals.unsqueeze(dim=0).expand(2, -1, -1),
        batched_ground_truth_labels=labels, foreground_iou_threshold=0.5)
    assert matched_indices.tolist() == [[1, 0, -1], [0, -1, -1]]
    assert target_labels.tolist() == [[5, 3, 0], [8, 0, 0]]
    assert target_boxes[1].tolist() == [[20.0, 20.0, 40.0, 40.0]] * 3


def test_sampling_counts_kinds_and_uniformity() -> None:
    """
    Verifies the sampled targets of every image: at most the desired positives, filled with negatives, more positives
    when an image lacks negatives, only negatives without positives, never an ignored target nor a target sampled as
    both, and a uniform draw (every positive of the first image sampled three times out of ten).
    """
    labels = torch.full(size=(3, 50), fill_value=-1.0)
    labels[0, :10], labels[0, 10:40] = 1.0, 0.0
    labels[1, :10], labels[1, 10:14] = 2.0, 0.0
    labels[2, :] = 0.0

    torch.manual_seed(seed=0)
    positive_frequencies = torch.zeros(size=(10,))
    for _ in range(2000):
        sampled_positives, sampled_negatives = sample_positive_and_negative_training_targets(
            labels=labels, desired_positives=3, desired_total=12)
        assert sampled_positives.sum(dim=1).tolist() == [3, 8, 0]
        assert sampled_negatives.sum(dim=1).tolist() == [9, 4, 12]
        assert bool((labels[sampled_positives] >= 1.0).all()) and bool((labels[sampled_negatives] == 0.0).all())
        assert not bool((sampled_positives & sampled_negatives).any())
        positive_frequencies += sampled_positives[0, :10].float()
    assert torch.allclose(input=positive_frequencies / 2000, other=torch.full(size=(10,), fill_value=0.3), atol=0.05)


def test_every_object_gives_a_positive_proposal_with_its_ground_truth_box() -> None:
    """
    Verifies that proposals far from every object give no positive target, and that once the ground truth boxes are
    added every object gives a positive proposal of its class with a zero regression target.
    """
    level_anchors = {5: torch.tensor(data=[[0.0, 0.0, 32.0, 32.0]])}
    boxes = [torch.tensor(data=[[100.0, 100.0, 140.0, 150.0], [10.0, 200.0, 60.0, 240.0]])]
    labels = [torch.tensor(data=[4, 7])]
    proposals = torch.tensor(data=[[[300.0, 300.0, 340.0, 340.0]] * 6])
    levels = torch.full(size=(1, 6), fill_value=5, dtype=torch.int64)
    _, target_labels, _ = batch_assign_targets_to_proposals(
        batched_ground_truth_boxes=boxes, batched_proposals=proposals, batched_ground_truth_labels=labels,
        foreground_iou_threshold=0.5)
    assert int((target_labels > 0).sum()) == 0

    # The added ground truth boxes are the positives of their objects, already in place
    new_proposals, _, written_slots = add_ground_truth_proposals(
        batched_ground_truth_boxes=boxes, batched_proposals=proposals, batched_proposal_levels=levels,
        level_anchors=level_anchors)
    assert written_slots.tolist() == [[False, False, False, False, True, True]]
    target_boxes, target_labels, matched_indices = batch_assign_targets_to_proposals(
        batched_ground_truth_boxes=boxes, batched_proposals=new_proposals, batched_ground_truth_labels=labels,
        foreground_iou_threshold=0.5)
    assert target_labels.tolist() == [[0, 0, 0, 0, 4, 7]] and matched_indices[0, 4:].tolist() == [0, 1]
    regression_targets = turn_boxes_to_transformation_targets(ground_truth_boxes=target_boxes,
                                                              predicted_boxes=new_proposals)
    assert torch.allclose(input=regression_targets[0, 4:], other=torch.zeros(size=(2, 4)), atol=1e-6)


def test_written_slots_flag_the_ground_truth_copies_of_every_image() -> None:
    """
    Verifies the mask of the slots holding a ground truth box, read by the training target viewer to tell these
    copies from the proposals: the last slots of an image with objects, none for an image without object, and every
    slot for an image with more objects than slots, whose last objects are kept.
    """
    level_anchors = {5: torch.tensor(data=[[0.0, 0.0, 32.0, 32.0]])}
    boxes = [torch.tensor(data=[[100.0, 100.0, 140.0, 150.0]]), torch.zeros(size=(0, 4)),
             torch.tensor(data=[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0], [40.0, 40.0, 50.0, 50.0],
                                [60.0, 60.0, 70.0, 70.0]])]
    proposals = torch.tensor(data=[[[300.0, 300.0, 340.0, 340.0]] * 3] * 3)
    levels = torch.full(size=(3, 3), fill_value=5, dtype=torch.int64)
    new_proposals, _, written_slots = add_ground_truth_proposals(
        batched_ground_truth_boxes=boxes, batched_proposals=proposals, batched_proposal_levels=levels,
        level_anchors=level_anchors)
    assert written_slots.tolist() == [[False, False, True], [False, False, False], [True, True, True]]

    # The flagged slots hold the ground truth boxes, the others their proposals
    assert torch.equal(input=new_proposals[~written_slots], other=proposals[~written_slots])
    assert torch.equal(input=new_proposals[2], other=boxes[2][1:])


def record_overlap_objects(recorded_object_counts: List[int], boxes_1: torch.Tensor,
                           boxes_2: torch.Tensor) -> torch.Tensor:
    """
    Computes the overlaps of get_intersection_over_union and records the number of objects of the call, put in place of
    the overlaps of target_assignment.py so that the tests see the chunks of objects of the anchor target assignments.

    Args:
        recorded_object_counts (List[int]): The object counts of the calls, appended to.
        boxes_1 (torch.Tensor): The [..., N, 4] objects.
        boxes_2 (torch.Tensor): The [..., M, 4] anchors.

    Returns:
        torch.Tensor: The [..., N, M] overlaps.
    """
    recorded_object_counts.append(int(boxes_1.shape[-2]))
    return get_intersection_over_union(boxes_1=boxes_1, boxes_2=boxes_2)


def build_crowded_batch() -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Builds a batch of three images with 9, 7 and 0 objects and the anchors of three sizes (16, 32 and 64 pixels) centred
    every 8 pixels. The objects 1 and 6 of the first image overlap the first anchor [0, 0, 16, 16] by 0.5 each and
    are its only objects, so that a matching in chunks of 2 objects must keep the first one on this tie.

    Returns:
        Tuple[List[torch.Tensor], torch.Tensor]: The [G_i, 4] objects of every image and the [1875, 4] anchors, the 625
            anchors of 16 pixels first, then those of 32 and 64 pixels.
    """
    generator = torch.Generator().manual_seed(0)
    corners = torch.rand(size=(2, 9, 2, 2), generator=generator) * 200.0 + 40.0
    boxes = torch.cat(tensors=[corners.amin(dim=2), corners.amax(dim=2) + 8.0], dim=-1)
    boxes[0, 1] = torch.tensor(data=[0.0, 0.0, 16.0, 32.0])
    boxes[0, 6] = torch.tensor(data=[0.0, 0.0, 32.0, 16.0])

    # Anchors of every size at every centre of the grid
    centres = torch.arange(start=8.0, end=208.0, step=8.0)
    centre_y, centre_x = torch.meshgrid(centres, centres, indexing="ij")
    centre_points = torch.stack(tensors=[centre_x.flatten(), centre_y.flatten()], dim=-1)
    anchors = torch.cat(tensors=[torch.cat(tensors=[centre_points - size / 2, centre_points + size / 2], dim=-1)
                                 for size in (16.0, 32.0, 64.0)], dim=0)
    return [boxes[0], boxes[1, :7], torch.zeros(size=(0, 4))], anchors


def test_anchor_targets_in_chunks_match_a_single_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that matching the objects of a crowded batch to the anchors in chunks of 2 objects gives the anchor
    targets of a single pass, the first of two objects overlapping an anchor equally being kept across chunks, and that
    no overlap tensor of the chunked pass holds more than 2 objects, so that the memory of the assignment stays bounded
    whatever the number of objects of an image.
    """
    batched_boxes, anchors = build_crowded_batch()
    arguments = {"batched_ground_truth_boxes": batched_boxes,
                 "batched_anchors": anchors.unsqueeze(dim=0).expand(len(batched_boxes), -1, -1),
                 "background_iou_threshold": BACKGROUND_THRESHOLD, "foreground_iou_threshold": FOREGROUND_THRESHOLD}
    single_pass_boxes, single_pass_labels = batch_assign_targets_to_anchors(**arguments)

    # The tie of the first anchor: forced foreground for both objects, it regresses to the first one
    assert single_pass_labels[0, 0] == 1.0 and single_pass_boxes[0, 0].tolist() == [0.0, 0.0, 16.0, 32.0]

    # Chunks of 2 objects: 5 overlap tensors for the 9 objects of the most crowded image
    recorded_object_counts = []
    monkeypatch.setattr(target_assignment, "MAXIMUM_ANCHOR_OVERLAPS", len(batched_boxes) * anchors.shape[0] * 2)
    monkeypatch.setattr(target_assignment, "get_intersection_over_union",
                        functools.partial(record_overlap_objects, recorded_object_counts))
    chunked_boxes, chunked_labels = batch_assign_targets_to_anchors(**arguments)
    assert recorded_object_counts == [2, 2, 2, 2, 1]
    assert torch.equal(input=chunked_boxes, other=single_pass_boxes)
    assert torch.equal(input=chunked_labels, other=single_pass_labels)


def test_ground_truth_proposal_levels_in_chunks_match_a_single_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the ground truth boxes added to the proposals of a crowded batch take the same levels whether their
    best anchor overlaps are computed in one pass or in chunks of 2 objects, no overlap tensor of the chunked pass
    holding more than 2 objects.
    """
    batched_boxes, anchors = build_crowded_batch()
    level_anchors = {3: anchors[:625], 4: anchors[625:1250], 5: anchors[1250:]}
    generator = torch.Generator().manual_seed(1)
    corners = torch.rand(size=(3, 12, 2, 2), generator=generator) * 200.0
    arguments = {"batched_ground_truth_boxes": batched_boxes,
                 "batched_proposals": torch.cat(tensors=[corners.amin(dim=2), corners.amax(dim=2) + 8.0], dim=-1),
                 "batched_proposal_levels": torch.full(size=(3, 12), fill_value=4, dtype=torch.int64),
                 "level_anchors": level_anchors}
    single_pass_outputs = add_ground_truth_proposals(**arguments)

    # Chunks of 2 objects in every level: 5 overlap tensors per level for the 9 objects of the most crowded image
    recorded_object_counts = []
    monkeypatch.setattr(target_assignment, "MAXIMUM_ANCHOR_OVERLAPS", len(batched_boxes) * 625 * 2)
    monkeypatch.setattr(target_assignment, "get_intersection_over_union",
                        functools.partial(record_overlap_objects, recorded_object_counts))
    chunked_outputs = add_ground_truth_proposals(**arguments)
    assert recorded_object_counts == [2, 2, 2, 2, 1] * 3
    assert all(torch.equal(input=chunked_output, other=single_pass_output)
               for chunked_output, single_pass_output in zip(chunked_outputs, single_pass_outputs))

    # The levels of the written slots are not all the same, so that the best level of every object was compared
    assert len(set(single_pass_outputs[1][single_pass_outputs[2]].tolist())) > 1


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="The fused matching runs on the GPU.")
def test_fused_gpu_matching_gives_the_eager_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the anchor targets and the levels of the ground truth proposals of a crowded batch on the GPU, whose
    matching runs compiled (compile_for_gpu) in chunks of 2 objects, are those of the eager matching on the CPU, the
    first of two objects overlapping an anchor equally included.
    """
    batched_boxes, anchors = build_crowded_batch()
    monkeypatch.setattr(target_assignment, "MAXIMUM_ANCHOR_OVERLAPS", len(batched_boxes) * anchors.shape[0] * 2)
    level_anchors = {3: anchors[:625], 4: anchors[625:1250], 5: anchors[1250:]}
    proposals = torch.rand(size=(3, 12, 2), generator=torch.Generator().manual_seed(1)) * 200.0
    outputs = {}
    for device in ["cpu", "cuda"]:
        device_boxes = [boxes.to(device=device) for boxes in batched_boxes]
        device_anchors = anchors.to(device=device)
        target_boxes, labels = batch_assign_targets_to_anchors(
            batched_ground_truth_boxes=device_boxes,
            batched_anchors=device_anchors.unsqueeze(dim=0).expand(len(batched_boxes), -1, -1),
            background_iou_threshold=BACKGROUND_THRESHOLD, foreground_iou_threshold=FOREGROUND_THRESHOLD)
        _, proposal_levels, _ = add_ground_truth_proposals(
            batched_ground_truth_boxes=device_boxes,
            batched_proposals=torch.cat(tensors=[proposals, proposals + 8.0], dim=-1).to(device=device),
            batched_proposal_levels=torch.full(size=(3, 12), fill_value=4, dtype=torch.int64, device=device),
            level_anchors={level: level_anchor.to(device=device) for level, level_anchor in level_anchors.items()})
        outputs[device] = [target_boxes.cpu(), labels.cpu(), proposal_levels.cpu()]

    # Both matching functions were compiled for the GPU, and give the targets of the eager matching
    assert target_assignment.compile_for_gpu.cache_info().currsize == 2
    assert all(torch.equal(input=gpu_output, other=cpu_output)
               for gpu_output, cpu_output in zip(outputs["cuda"], outputs["cpu"]))
    assert outputs["cuda"][0][0, 0].tolist() == [0.0, 0.0, 16.0, 32.0]
