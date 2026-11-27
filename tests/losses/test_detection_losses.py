"""
Tests of vision_training/losses/detection_losses.py: the Complete-IoU loss against the CIoU and DIoU losses of
torchvision on both sides of the IoU of 0.5 from which it weighs the aspect ratios, its values for equal, scaled and
degenerate boxes, and the localisation losses: the smooth L1 on the offsets of the region proposal network, and the
smooth L1 or the CIoU loss on the boxes they decode to of the detection heads ('Detector.box_loss').
"""

import math
import pytest
import torch

from torchvision.ops import box_iou, complete_box_iou_loss, distance_box_iou_loss
from typing import Tuple

from vision_training.losses.detection_losses import (ASPECT_RATIO_IOU_THRESHOLD, LOCALISATION_LOSS_BETA,
                                                     assign_region_proposal_targets, compute_complete_iou_loss,
                                                     compute_detection_losses, compute_localisation_loss,
                                                     compute_region_proposal_losses)
from vision_training.models.detection.target_assignment import turn_boxes_to_transformation_targets

# Anchor assignment and sampling settings of the region proposal losses, those of the detection fixture, sampling every
# positive anchor of the region proposal test
REGION_PROPOSAL_CONFIGURATION = {"foreground_iou_threshold": {"min": 0.35, "max": 1.0},
                                 "background_iou_threshold": {"min": 0.2, "max": 0.3},
                                 "number_training_positives": 128, "total_training_samples": 256}


def draw_box_pairs(number_pairs: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Draws target boxes in a 512 pixel image and predicted boxes around them, shifted by up to half the target sizes
    and from half to twice as large, so that their IoUs spread on both sides of 0.5, in float64 so that the comparisons
    with torchvision do not depend on rounding.

    Args:
        number_pairs (int): The number of box pairs.
        seed (int): The seed of the generator.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The predicted and target boxes [x_min, y_min, x_max, y_max], of shape
            [number_pairs, 4].
    """
    generator = torch.Generator().manual_seed(seed)
    uniform_draws = torch.rand(size=(4, number_pairs, 2), generator=generator, dtype=torch.float64)

    # Targets from 8 to 128 pixels, then predictions shifted and scaled around their centres
    target_sizes = 8 + 120 * uniform_draws[0]
    target_minimums = 300 * uniform_draws[1]
    target_boxes = torch.cat(tensors=[target_minimums, target_minimums + target_sizes], dim=1)
    predicted_sizes = target_sizes * torch.exp(input=(2 * uniform_draws[2] - 1) * math.log(2))
    predicted_centres = target_minimums + target_sizes / 2 + (uniform_draws[3] - 0.5) * target_sizes
    predicted_boxes = torch.cat(tensors=[predicted_centres - predicted_sizes / 2,
                                         predicted_centres + predicted_sizes / 2], dim=1)
    return predicted_boxes, target_boxes


def test_complete_iou_loss_weighs_the_aspect_ratios_from_an_iou_of_one_half() -> None:
    """
    Verifies that the CIoU loss and its gradient equal those of the CIoU loss of torchvision for the boxes overlapping
    their targets by an IoU of 0.5 or more, and those of the DIoU loss below it, where the journal version of the
    paper sets the trade-off alpha to 0. The trade-off is a constant of the gradient in all of them.
    """
    predicted_boxes, target_boxes = draw_box_pairs(number_pairs=2000, seed=0)
    intersection_over_union = box_iou(boxes1=predicted_boxes, boxes2=target_boxes).diagonal()

    # Pairs on both sides of the threshold, those within 0.001 of it left out, the epsilons moving their IoU slightly
    kept_pairs = (intersection_over_union - ASPECT_RATIO_IOU_THRESHOLD).abs() > 1e-3
    matched_pairs = intersection_over_union >= ASPECT_RATIO_IOU_THRESHOLD
    assert int((kept_pairs & matched_pairs).sum()) > 200 and int((kept_pairs & ~matched_pairs).sum()) > 200

    # Losses and gradients of the predicted boxes, compared with the CIoU loss above the threshold, the DIoU one below
    predicted_boxes.requires_grad_(True)
    losses = compute_complete_iou_loss(predicted_boxes=predicted_boxes, target_boxes=target_boxes)
    gradients, = torch.autograd.grad(outputs=losses.sum(), inputs=predicted_boxes)
    for reference_function, compared_pairs in [(complete_box_iou_loss, kept_pairs & matched_pairs),
                                               (distance_box_iou_loss, kept_pairs & ~matched_pairs)]:
        reference_losses = reference_function(boxes1=predicted_boxes, boxes2=target_boxes, reduction="none")
        reference_gradients, = torch.autograd.grad(outputs=reference_losses.sum(), inputs=predicted_boxes)
        assert torch.allclose(input=losses[compared_pairs], other=reference_losses[compared_pairs], atol=1e-6)
        assert torch.allclose(input=gradients[compared_pairs], other=reference_gradients[compared_pairs], atol=1e-6)

    # Above the threshold the aspect ratio term exceeds the tolerance a thousand times, so that the comparisons tell
    # the CIoU loss from the DIoU loss
    distance_losses = distance_box_iou_loss(boxes1=predicted_boxes, boxes2=target_boxes, reduction="none")
    assert float((losses - distance_losses).detach()[kept_pairs & matched_pairs].max()) > 1e-3


def test_complete_iou_loss_of_equal_scaled_and_degenerate_boxes() -> None:
    """
    Verifies that the CIoU loss is zero for boxes equal to their targets, does not change when the boxes and their
    targets are scaled and translated together, its terms being ratios of areas and lengths, stays between 0 and 2,
    and gives finite values and gradients for flat, empty and distant boxes, which keep a gradient.
    """
    predicted_boxes, target_boxes = draw_box_pairs(number_pairs=200, seed=1)
    losses = compute_complete_iou_loss(predicted_boxes=predicted_boxes, target_boxes=target_boxes)
    assert float(compute_complete_iou_loss(predicted_boxes=target_boxes, target_boxes=target_boxes).abs().max()) < 1e-6
    scaled_losses = compute_complete_iou_loss(predicted_boxes=predicted_boxes * 3 + 40,
                                              target_boxes=target_boxes * 3 + 40)
    assert torch.allclose(input=scaled_losses, other=losses, atol=1e-6)
    assert bool((losses >= 0).all()) and bool((losses < 2).all())

    # Flat, empty and distant predictions, the distant one between 1 and 2 like the DIoU loss of disjoint boxes
    degenerate_predictions = torch.tensor(data=[[10, 10, 10, 30], [10, 10, 30, 10], [10, 10, 10, 10],
                                                [400, 400, 420, 430]], dtype=torch.float64, requires_grad=True)
    degenerate_targets = torch.tensor(data=[[10, 10, 20, 30], [10, 10, 30, 30], [10, 10, 10, 10], [10, 10, 30, 30]],
                                      dtype=torch.float64)
    degenerate_losses = compute_complete_iou_loss(predicted_boxes=degenerate_predictions,
                                                  target_boxes=degenerate_targets)
    degenerate_gradients, = torch.autograd.grad(outputs=degenerate_losses.sum(), inputs=degenerate_predictions)
    degenerate_losses = degenerate_losses.detach()
    assert bool(torch.isfinite(input=degenerate_losses).all())
    assert bool(torch.isfinite(input=degenerate_gradients).all())
    assert 1 < float(degenerate_losses[3]) < 2 and float(degenerate_gradients[3].abs().sum()) > 0


def test_localisation_loss_reads_the_box_loss_of_its_section() -> None:
    """
    Verifies that the localisation loss shared by the region proposal network and the detection heads is the summed
    smooth L1 of the offsets with "smooth_l1", unchanged by the CIoU loss, and the summed CIoU loss of the boxes the
    predicted and target offsets decode to from their anchors with "complete_iou": the CIoU loss of the anchors
    themselves for zero offsets, zero for the target offsets, decreasing after a gradient step on the offsets, and
    computed in the float32 of the anchors from bfloat16 offsets, as under the mixed precision of the trainers. Both
    losses are zero without positive, and an unknown loss is refused.
    """
    predicted_boxes, target_boxes = draw_box_pairs(number_pairs=64, seed=2)
    anchors, target_boxes = predicted_boxes.to(dtype=torch.float32), target_boxes.to(dtype=torch.float32)
    regression_targets = turn_boxes_to_transformation_targets(ground_truth_boxes=target_boxes, predicted_boxes=anchors)
    regression_predictions = regression_targets + 0.2 * torch.randn(size=regression_targets.shape,
                                                                    generator=torch.Generator().manual_seed(3))

    # Smooth L1 on the offsets, as before the CIoU loss
    smooth_l1_loss = compute_localisation_loss(box_loss="smooth_l1", regression_predictions=regression_predictions,
                                               regression_targets=regression_targets, reference_boxes=anchors)
    assert torch.equal(input=smooth_l1_loss, other=torch.nn.functional.smooth_l1_loss(
        input=regression_predictions, target=regression_targets, beta=LOCALISATION_LOSS_BETA, reduction="sum"))

    # CIoU loss of the decoded boxes: the anchors for zero offsets, the targets for the target offsets
    zero_offsets_loss = compute_localisation_loss(box_loss="complete_iou", reference_boxes=anchors,
                                                  regression_predictions=torch.zeros_like(input=regression_targets),
                                                  regression_targets=regression_targets)
    assert torch.allclose(input=zero_offsets_loss, rtol=1e-4,
                          other=compute_complete_iou_loss(predicted_boxes=anchors, target_boxes=target_boxes).sum())
    assert float(compute_localisation_loss(box_loss="complete_iou", regression_predictions=regression_targets,
                                           regression_targets=regression_targets, reference_boxes=anchors)) < 1e-4

    # A gradient step on the offsets lowers the loss, whose gradient is finite
    trained_predictions = regression_predictions.clone().requires_grad_(True)
    loss = compute_localisation_loss(box_loss="complete_iou", regression_predictions=trained_predictions,
                                     regression_targets=regression_targets, reference_boxes=anchors)
    gradients, = torch.autograd.grad(outputs=loss, inputs=trained_predictions)
    assert bool(torch.isfinite(input=gradients).all())
    stepped_loss = compute_localisation_loss(box_loss="complete_iou", regression_targets=regression_targets,
                                             regression_predictions=regression_predictions - 0.1 * gradients,
                                             reference_boxes=anchors)
    assert float(stepped_loss) < float(loss.detach())

    # Offsets of the heads in bfloat16, the boxes decoded and compared in the float32 of the anchors
    bfloat16_loss = compute_localisation_loss(box_loss="complete_iou", regression_targets=regression_targets,
                                              regression_predictions=regression_predictions.to(dtype=torch.bfloat16),
                                              reference_boxes=anchors)
    assert bfloat16_loss.dtype == torch.float32
    assert torch.allclose(input=bfloat16_loss, other=loss.detach(), rtol=1e-2)

    # A batch without positive, e.g. images without objects, and an unknown box loss
    for box_loss in ["smooth_l1", "complete_iou"]:
        assert float(compute_localisation_loss(box_loss=box_loss, regression_predictions=torch.zeros(size=(0, 4)),
                                               regression_targets=torch.zeros(size=(0, 4)),
                                               reference_boxes=torch.zeros(size=(0, 4)))) == 0.0
    with pytest.raises(expected_exception=ValueError, match="Unknown box loss 'generalized_iou'"):
        compute_localisation_loss(box_loss="generalized_iou", regression_predictions=regression_predictions,
                                  regression_targets=regression_targets, reference_boxes=anchors)


def test_region_proposal_localisation_loss_is_the_smooth_l1_of_the_offsets() -> None:
    """
    Verifies that the localisation loss of the region proposal network is always the smooth L1 of the offsets of the
    sampled positive anchors, averaged over the positives of the batch, its 'RegionProposal' section holding no box
    loss: zero offsets give the mean summed smooth L1 of the target offsets of the positive anchors, which differs from
    their CIoU loss, and the target offsets give zero. Each image has one ground truth box overlapping two of the four
    anchors, the other two not overlapping it, which assign_region_proposal_targets labels and samples as the positives
    of the image with their target offsets.
    """
    anchors = torch.tensor(data=[[104, 96, 196, 186], [90, 110, 220, 170], [310, 250, 350, 410], [280, 280, 380, 380]],
                           dtype=torch.float32).expand(size=(2, -1, -1))
    ground_truth_boxes = [torch.tensor(data=[[100, 100, 200, 180]], dtype=torch.float32),
                          torch.tensor(data=[[300, 260, 360, 400]], dtype=torch.float32)]
    positive_anchors = torch.cat(tensors=[anchors[0, :2], anchors[1, 2:]])
    matched_boxes = torch.cat(tensors=[ground_truth_boxes[0].expand(size=(2, -1)),
                                       ground_truth_boxes[1].expand(size=(2, -1))])
    target_offsets = turn_boxes_to_transformation_targets(ground_truth_boxes=matched_boxes,
                                                          predicted_boxes=positive_anchors)

    # Anchor targets: the two overlapping anchors of every image are foreground and sampled, with their offsets
    region_proposal_targets = assign_region_proposal_targets(
        ground_truth_bounding_boxes=ground_truth_boxes, anchors=anchors,
        region_proposal_configuration=REGION_PROPOSAL_CONFIGURATION)
    positive_layout = [[True, True, False, False], [False, False, True, True]]
    assert region_proposal_targets["sampled_positive_mask"].tolist() == positive_layout
    assert (region_proposal_targets["anchor_labels"] == 1).tolist() == positive_layout
    assert torch.allclose(input=region_proposal_targets["regression_targets"][torch.tensor(data=positive_layout)],
                          other=target_offsets)

    # Zero offsets: the smooth L1 of the target offsets, summed over the coordinates and averaged over the positives
    proposals_dictionary = {"anchors": anchors, "classification_scores": torch.zeros(size=(2, 4)),
                            "bounding_box_regressions": torch.zeros(size=(2, 4, 4))}
    _, localisation_loss = compute_region_proposal_losses(aggregated_proposals_dictionary=proposals_dictionary,
                                                          region_proposal_targets=region_proposal_targets)
    expected_loss = torch.nn.functional.smooth_l1_loss(input=torch.zeros_like(input=target_offsets),
                                                       target=target_offsets, beta=LOCALISATION_LOSS_BETA,
                                                       reduction="sum") / len(positive_anchors)
    complete_iou_loss = compute_complete_iou_loss(predicted_boxes=positive_anchors, target_boxes=matched_boxes).mean()
    assert float(expected_loss) > 0.01 and torch.allclose(input=localisation_loss, other=expected_loss, rtol=1e-4)
    assert not torch.allclose(input=localisation_loss, other=complete_iou_loss, rtol=1e-2)

    # Target offsets of the positive anchors, decoding to their ground truth boxes
    regressions = torch.zeros(size=(2, 4, 4))
    regressions[0, :2], regressions[1, 2:] = target_offsets[:2], target_offsets[2:]
    _, localisation_loss = compute_region_proposal_losses(
        aggregated_proposals_dictionary={**proposals_dictionary, "bounding_box_regressions": regressions},
        region_proposal_targets=region_proposal_targets)
    assert float(localisation_loss) < 1e-6


def test_detection_complete_iou_loss_decodes_the_offsets_of_the_ground_truth_class() -> None:
    """
    Verifies that the CIoU loss of the detection heads reads the offsets of the class of every positive proposal,
    decodes them from the proposal box without its batch index column and averages over the positive proposals: zero
    offsets give the mean CIoU loss of the positive proposals against their ground truth boxes, and the target offsets
    give zero whatever the offsets of the other classes and of the background proposals.
    """
    proposal_boxes, ground_truth_boxes = draw_box_pairs(number_pairs=6, seed=4)
    proposal_boxes = proposal_boxes.to(dtype=torch.float32)
    ground_truth_boxes = ground_truth_boxes.to(dtype=torch.float32)
    labels = torch.tensor(data=[1, 2, 0, 1, 0, 2])
    positive_mask = labels > 0
    regression_targets = turn_boxes_to_transformation_targets(ground_truth_boxes=ground_truth_boxes,
                                                              predicted_boxes=proposal_boxes)
    box_regressions = torch.randn(size=(6, 3, 4), generator=torch.Generator().manual_seed(5))
    batch_indices = torch.tensor(data=[[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]])
    detections_dictionary = {"classification_scores": torch.zeros(size=(6, 3)), "sliced_labels": labels,
                             "sliced_proposals": torch.cat(tensors=[batch_indices, proposal_boxes], dim=1),
                             "sliced_regression_targets": regression_targets}

    # Zero offsets in the channel of the class: the positive proposals against their ground truth boxes
    zero_class_regressions = box_regressions.clone()
    zero_class_regressions[torch.arange(end=6), labels] = 0.0
    _, localisation_loss = compute_detection_losses(box_loss="complete_iou", aggregated_detections_dictionary={
        **detections_dictionary, "box_regressions": zero_class_regressions})
    expected_loss = compute_complete_iou_loss(predicted_boxes=proposal_boxes[positive_mask],
                                              target_boxes=ground_truth_boxes[positive_mask]).mean()
    assert float(expected_loss) > 0.1 and torch.allclose(input=localisation_loss, other=expected_loss, rtol=1e-4)

    # Target offsets in the channel of the class, the other channels keeping their random offsets
    target_class_regressions = box_regressions.clone()
    target_class_regressions[torch.arange(end=6), labels] = regression_targets
    _, localisation_loss = compute_detection_losses(box_loss="complete_iou", aggregated_detections_dictionary={
        **detections_dictionary, "box_regressions": target_class_regressions})
    assert float(localisation_loss) < 1e-4
