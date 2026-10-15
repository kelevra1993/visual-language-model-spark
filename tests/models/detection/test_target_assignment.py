"""
Tests of the batched training target assignment of vision_training/models/detection/target_assignment.py: the
padding of the ground truths of a batch, the foreground, background and ignored anchors of every image (an image
without object included), the best anchor of every object made foreground with both fallback rules, the proposal
targets of images with different object counts, and the number, kind and uniformity of the sampled targets.
"""

import torch

from vision_training.models.detection.target_assignment import (batch_assign_targets_to_anchors,
                                                                batch_assign_targets_to_proposals, pad_ground_truths,
                                                                sample_positive_and_negative_training_targets)

# Anchor thresholds of the tests: background from 0.1 to 0.3, foreground from 0.7
BACKGROUND_THRESHOLD = {"min": 0.1, "max": 0.3}
FOREGROUND_THRESHOLD = {"min": 0.7, "max": 1.0}


def test_ground_truths_are_padded_to_the_largest_image() -> None:
    """
    Verifies that the ground truths of a batch are padded with empty boxes and the background label up to the largest
    number of objects, with the mask of the real objects, and that a batch without any object keeps one padding row.
    """
    boxes = [torch.tensor(data=[[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 20.0, 20.0]]), torch.zeros(size=(0, 4)),
             torch.tensor(data=[[1.0, 2.0, 3.0, 4.0]])]
    labels = [torch.tensor(data=[3, 4]), torch.zeros(size=(0,), dtype=torch.int64), torch.tensor(data=[7])]
    padded_boxes, padded_labels, valid_ground_truths = pad_ground_truths(batched_ground_truth_boxes=boxes,
                                                                         batched_ground_truth_labels=labels)
    assert padded_boxes.shape == (3, 2, 4) and padded_boxes[2].tolist() == [[1.0, 2.0, 3.0, 4.0], [0.0] * 4]
    assert padded_labels.tolist() == [[3, 4], [0, 0], [7, 0]]
    assert valid_ground_truths.tolist() == [[True, True], [False, False], [True, False]]

    # A batch without any object keeps one empty row per image
    padded_boxes, padded_labels, valid_ground_truths = pad_ground_truths(batched_ground_truth_boxes=[boxes[1]] * 2)
    assert padded_boxes.shape == (2, 1, 4) and padded_labels is None and not bool(valid_ground_truths.any())


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
    the two fallback rules when that anchor overlaps another object more: by default it targets the object it
    overlaps the most, and with the strict rule the object that made it foreground.
    """
    # The first object's best anchor (overlap 0.4) overlaps the second object more (0.75)
    ground_truth = torch.tensor(data=[[0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 100.0, 30.0]])
    anchors = torch.tensor(data=[[0.0, 0.0, 100.0, 40.0], [0.0, 0.0, 100.0, 30.0], [300.0, 300.0, 400.0, 400.0]])
    target_boxes, labels = batch_assign_targets_to_anchors(
        batched_ground_truth_boxes=[ground_truth], batched_anchors=anchors.unsqueeze(dim=0),
        background_iou_threshold=BACKGROUND_THRESHOLD, foreground_iou_threshold=FOREGROUND_THRESHOLD)
    assert labels.tolist() == [[1.0, 1.0, -1.0]]
    assert target_boxes[0, 0].tolist() == [0.0, 0.0, 100.0, 30.0]

    strict_target_boxes, strict_labels = batch_assign_targets_to_anchors(
        batched_ground_truth_boxes=[ground_truth], batched_anchors=anchors.unsqueeze(dim=0),
        background_iou_threshold=BACKGROUND_THRESHOLD, foreground_iou_threshold=FOREGROUND_THRESHOLD,
        strict_fallback_assignment=True)
    assert torch.equal(input=strict_labels, other=labels)
    assert strict_target_boxes[0, 0].tolist() == [0.0, 0.0, 100.0, 100.0]


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
