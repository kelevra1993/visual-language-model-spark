"""
Tests of the batched training target assignment of vision_training/models/detection/target_assignment.py: the
foreground, background and ignored anchors of every image (an image without object included), the best anchor of
every object made foreground with both fallback rules, the easy negatives below the background minimum, the proposal
targets of images with different object counts, the number, kind and uniformity of the sampled targets, the share of
easy negatives among the sampled negatives with its shortages and its unchanged draws at 0.0, and the ground truth
boxes added to the proposals of the detection heads giving a positive proposal per object.
"""

import torch

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
    first object of its image, and an image without object whose anchors are all background with empty targets. Only
    the anchor ignored below 0.1 is an easy negative.
    """
    ground_truth = torch.tensor(data=[[0.0, 0.0, 100.0, 100.0]])
    anchors = torch.tensor(data=[[0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 100.0, 40.0], [0.0, 0.0, 100.0, 20.0],
                                 [300.0, 300.0, 400.0, 400.0]])
    target_boxes, labels, easy_negative_mask = batch_assign_targets_to_anchors(
        batched_ground_truth_boxes=[ground_truth, torch.zeros(size=(0, 4))],
        batched_anchors=anchors.unsqueeze(dim=0).expand(2, -1, -1), background_iou_threshold=BACKGROUND_THRESHOLD,
        foreground_iou_threshold=FOREGROUND_THRESHOLD)
    assert labels.dtype == torch.float32 and labels.tolist() == [[1.0, -1.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]]
    assert target_boxes[0].tolist() == [[0.0, 0.0, 100.0, 100.0]] * 4
    assert target_boxes[1].tolist() == [[0.0, 0.0, 0.0, 0.0]] * 4
    assert easy_negative_mask.tolist() == [[False, False, False, True], [False, False, False, False]]


def test_easy_negatives_leave_out_the_best_anchors_and_a_zero_minimum() -> None:
    """
    Verifies that the best anchor of a small object, below the background minimum but foreground, is no easy
    negative, and that a background minimum of 0.0 gives no easy negative, every anchor overlapping nothing being
    background.
    """
    # The small object overlaps its best anchor by 0.0025 only, the third anchor overlapping nothing
    ground_truth = torch.tensor(data=[[0.0, 0.0, 100.0, 100.0], [500.0, 500.0, 505.0, 505.0]])
    anchors = torch.tensor(data=[[0.0, 0.0, 100.0, 100.0], [500.0, 500.0, 600.0, 600.0],
                                 [300.0, 300.0, 400.0, 400.0]]).unsqueeze(dim=0)
    _, labels, easy_negative_mask = batch_assign_targets_to_anchors(
        batched_ground_truth_boxes=[ground_truth], batched_anchors=anchors,
        background_iou_threshold=BACKGROUND_THRESHOLD, foreground_iou_threshold=FOREGROUND_THRESHOLD)
    assert labels.tolist() == [[1.0, 1.0, -1.0]] and easy_negative_mask.tolist() == [[False, False, True]]

    # From a minimum of 0.0, the anchor overlapping nothing is background
    _, labels, easy_negative_mask = batch_assign_targets_to_anchors(
        batched_ground_truth_boxes=[ground_truth], batched_anchors=anchors,
        background_iou_threshold={"min": 0.0, "max": 0.3}, foreground_iou_threshold=FOREGROUND_THRESHOLD)
    assert labels.tolist() == [[1.0, 1.0, 0.0]] and not bool(easy_negative_mask.any())


def test_best_anchor_of_every_object_is_foreground() -> None:
    """
    Verifies that the anchor overlapping an object the most is foreground even below the foreground threshold, and
    that it targets the object it overlaps the most when that is another object than the one that made it foreground,
    as the region proposal network of torchvision.
    """
    # The first object's best anchor (overlap 0.4) overlaps the second object more (0.75)
    ground_truth = torch.tensor(data=[[0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 100.0, 30.0]])
    anchors = torch.tensor(data=[[0.0, 0.0, 100.0, 40.0], [0.0, 0.0, 100.0, 30.0], [300.0, 300.0, 400.0, 400.0]])
    target_boxes, labels, _ = batch_assign_targets_to_anchors(
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


def test_easy_negative_share_and_shortages() -> None:
    """
    Verifies the negatives drawn with a quarter of easy negatives: 2 easy and 7 background of 9 negatives, the
    background ones filling a shortage of easy ones, the easy ones a shortage of background ones, positives filling a
    shortage of both, and an image without easy negative (an image without object) sampled as without the share; never
    an ignored target nor a target sampled twice.
    """
    # Per image: 10 positives, then background (label 0), easy and ignored targets
    labels = torch.full(size=(5, 100), fill_value=-1.0)
    easy_negative_mask = torch.zeros(size=(5, 100), dtype=torch.bool)
    labels[:4, :10] = 1.0
    for image_index, (number_background, number_easy) in enumerate([(40, 40), (40, 1), (2, 40), (2, 1)]):
        labels[image_index, 10:10 + number_background] = 0.0
        easy_negative_mask[image_index, 50:50 + number_easy] = True
    labels[4] = 0.0

    torch.manual_seed(seed=0)
    for _ in range(200):
        sampled_positives, sampled_negatives = sample_positive_and_negative_training_targets(
            labels=labels, desired_positives=3, desired_total=12, easy_negative_mask=easy_negative_mask,
            easy_negative_fraction=0.25)
        sampled_easy_negatives = sampled_negatives & easy_negative_mask
        assert sampled_positives.sum(dim=1).tolist() == [3, 3, 3, 9, 0]
        assert (sampled_negatives & ~easy_negative_mask).sum(dim=1).tolist() == [7, 8, 2, 2, 12]
        assert sampled_easy_negatives.sum(dim=1).tolist() == [2, 1, 7, 1, 0]
        assert bool((labels[sampled_negatives & ~easy_negative_mask] == 0.0).all())
        assert not bool((sampled_positives & sampled_negatives).any())


def test_zero_easy_share_draws_as_without_easy_negatives() -> None:
    """
    Verifies that under the same seed the easy negatives with a share of 0.0, or a share without any easy negative,
    give the very samples of a call without easy negatives, so that the runs of the background IoU minimum stage and
    the detection heads keep their draws.
    """
    labels = torch.full(size=(3, 200), fill_value=-1.0)
    labels[:, :20], labels[:, 20:120] = 1.0, 0.0
    easy_negative_mask = torch.zeros(size=(3, 200), dtype=torch.bool)
    easy_negative_mask[:, 120:180] = True

    # Reference draw, without easy negatives
    torch.manual_seed(seed=3)
    reference_masks = sample_positive_and_negative_training_targets(labels=labels, desired_positives=8,
                                                                    desired_total=32)

    # Same draws with a share of 0.0, and with a share but no easy negative
    for mask, fraction in [(easy_negative_mask, 0.0), (torch.zeros_like(input=easy_negative_mask), 0.25)]:
        torch.manual_seed(seed=3)
        masks = sample_positive_and_negative_training_targets(labels=labels, desired_positives=8, desired_total=32,
                                                              easy_negative_mask=mask, easy_negative_fraction=fraction)
        assert all(torch.equal(input=mask_value, other=reference_value)
                   for mask_value, reference_value in zip(masks, reference_masks))


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
    _, target_labels, _ = batch_assign_targets_to_proposals(batched_ground_truth_boxes=boxes,
                                                            batched_proposals=proposals,
                                                            batched_ground_truth_labels=labels,
                                                            foreground_iou_threshold=0.5)
    assert int((target_labels > 0).sum()) == 0

    # The added ground truth boxes are the positives of their objects, already in place
    new_proposals, _ = add_ground_truth_proposals(batched_ground_truth_boxes=boxes, batched_proposals=proposals,
                                                  batched_proposal_levels=levels, level_anchors=level_anchors)
    target_boxes, target_labels, matched_indices = batch_assign_targets_to_proposals(
        batched_ground_truth_boxes=boxes, batched_proposals=new_proposals, batched_ground_truth_labels=labels,
        foreground_iou_threshold=0.5)
    assert target_labels.tolist() == [[0, 0, 0, 0, 4, 7]] and matched_indices[0, 4:].tolist() == [0, 1]
    regression_targets = turn_boxes_to_transformation_targets(ground_truth_boxes=target_boxes,
                                                              predicted_boxes=new_proposals)
    assert torch.allclose(input=regression_targets[0, 4:], other=torch.zeros(size=(2, 4)), atol=1e-6)
