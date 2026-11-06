"""
Tests of vision_training/losses/detection_losses.py: the region proposal classification loss takes the easy negatives
drawn by 'RegionProposal.easy_negative_fraction' as background, with a target of 0 although their anchor label is the
ignored -1.
"""

import math
import torch

from vision_training.losses.detection_losses import compute_region_proposal_losses


def test_region_proposal_loss_takes_easy_negatives_as_background() -> None:
    """
    Verifies that with every negative drawn among the easy ones (anchors overlapping no object, below the background
    minimum 0.1), the classification loss of the objectness logit 2 is the binary cross entropy of one positive and
    three negatives of target 0, and that a share of 0.0 leaves the easy anchors out, the image having no background
    anchor to draw.
    """
    # One object, its anchor, then six anchors overlapping nothing, all with the objectness logit 2
    ground_truth = [torch.tensor(data=[[0.0, 0.0, 100.0, 100.0]])]
    anchors = torch.tensor(data=[[[0.0, 0.0, 100.0, 100.0]] + [[300.0 + 10.0 * index, 300.0, 400.0, 400.0]
                                                               for index in range(6)]])
    proposals = {"anchors": anchors, "classification_scores": torch.full(size=(1, 7), fill_value=2.0),
                 "bounding_box_regressions": torch.zeros(size=(1, 7, 4))}
    configuration = {"background_iou_threshold": {"min": 0.1, "max": 0.3},
                     "foreground_iou_threshold": {"min": 0.7, "max": 1.0}, "number_training_positives": 1,
                     "total_training_samples": 4, "easy_negative_fraction": 1.0}

    # One positive of loss -log(sigmoid(2)) and three easy negatives of loss -log(1 - sigmoid(2))
    classification_loss, localisation_loss = compute_region_proposal_losses(
        ground_truth_bounding_boxes=ground_truth, aggregated_proposals_dictionary=proposals,
        region_proposal_configuration=configuration)
    sigmoid = 1.0 / (1.0 + math.exp(-2.0))
    expected_loss = (-math.log(sigmoid) - 3.0 * math.log(1.0 - sigmoid)) / 4.0
    assert math.isclose(float(classification_loss), expected_loss, rel_tol=1e-5) and float(localisation_loss) == 0.0

    # Without the share, the positive alone is sampled, the image lacking background anchors
    classification_loss, _ = compute_region_proposal_losses(
        ground_truth_bounding_boxes=ground_truth, aggregated_proposals_dictionary=proposals,
        region_proposal_configuration={**configuration, "easy_negative_fraction": 0.0})
    assert math.isclose(float(classification_loss), -math.log(sigmoid), rel_tol=1e-5)
