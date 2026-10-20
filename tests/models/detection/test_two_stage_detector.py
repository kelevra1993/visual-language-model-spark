"""
Tests of the intermediate results of vision_training/models/detection/two_stage_detector.py read by the instance
segmenter: the matched ground truth of every proposal returned by the target assignment, and the level and matched
ground truth of the sampled proposals and the proposal of every final detection carried through
forward_detection_stages, and the filtered proposals and the anchors of every level read by the anchor and proposal
recall of the detection trainers.
"""

import torch

from tests.models.detection.test_detector_baseline import get_fixed_batch
from tests.testing_utilities import build_test_model
from vision_training.models.detection.target_assignment import batch_assign_targets_to_proposals


def test_proposals_give_the_index_of_their_matched_ground_truth() -> None:
    """
    Verifies that every proposal overlapping a ground truth box enough gives the index of that box and its label,
    that the other proposals give -1 and the background label, and that an image without ground truth gives -1
    everywhere, alone or in a batch.
    """
    ground_truth_boxes = torch.tensor(data=[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]])
    proposals = torch.tensor(data=[[21.0, 21.0, 40.0, 40.0], [0.0, 0.0, 9.0, 10.0], [60.0, 60.0, 70.0, 70.0]])
    _, labels, matched_indices = batch_assign_targets_to_proposals(
        batched_ground_truth_boxes=[ground_truth_boxes], batched_proposals=proposals.unsqueeze(dim=0),
        batched_ground_truth_labels=[torch.tensor(data=[3, 5])], foreground_iou_threshold=0.5)
    assert labels.tolist() == [[5, 3, 0]] and matched_indices.tolist() == [[1, 0, -1]]

    # A batch with an image without ground truth
    _, batched_labels, batched_matched_indices = batch_assign_targets_to_proposals(
        batched_ground_truth_boxes=[ground_truth_boxes, torch.zeros(size=(0, 4))],
        batched_proposals=torch.stack(tensors=[proposals, proposals]), foreground_iou_threshold=0.5,
        batched_ground_truth_labels=[torch.tensor(data=[3, 5]), torch.zeros(size=(0,), dtype=torch.int64)])
    assert batched_matched_indices.tolist() == [[1, 0, -1], [-1, -1, -1]]
    assert batched_labels[1].tolist() == [0, 0, 0]


def test_detection_stages_carry_levels_and_matched_ground_truths() -> None:
    """
    Verifies that with ground truth the sampled proposals carry their level and their matched ground truth (an index
    exactly for the positive proposals, whose label is the one of that ground truth), and that in evaluation mode the
    level of every proposal and the proposal of every final detection are returned, so that the level of every
    detection can be read, with the filtered proposals, their level and their validity for the proposal recall. The
    anchors of every level are given for the anchor recall.
    """
    model = build_test_model(dtype=torch.float64)
    images, boxes, labels = get_fixed_batch()
    level_numbers = {int(level) for level in model.detector_heads_dictionary}

    # Training path: matched ground truth of every positive sampled proposal
    model.train()
    with torch.no_grad():
        stages = model.forward_detection_stages(input_tensor=images, ground_truth_bounding_boxes=boxes,
                                                ground_truth_labels=labels)
    detections = stages["aggregated_detections_dictionary"]
    positive_mask = detections["sliced_labels"] > 0
    assert bool(positive_mask.any())
    assert torch.equal(input=detections["sliced_matched_ground_truth_indices"] >= 0, other=positive_mask)
    assert set(detections["sliced_origins"].tolist()) <= level_numbers
    for proposal_index in torch.nonzero(input=positive_mask).flatten().tolist():
        image_index = int(detections["sliced_batch_indices"][proposal_index, 0])
        ground_truth_index = int(detections["sliced_matched_ground_truth_indices"][proposal_index])
        assert int(labels[image_index][ground_truth_index]) == int(detections["sliced_labels"][proposal_index])
    assert set(stages["backbone_output_tensor_dictionary"]) == {str(level) for level in level_numbers}

    # Inference path: level of every proposal, and proposal of every final detection
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor=images)
    proposal_levels = outputs["sliced_origins"]
    assert proposal_levels.shape == outputs["classification_scores"].shape[:2]
    detection_levels = torch.gather(input=proposal_levels, dim=1, index=outputs["final_detection_proposal_indices"])
    assert set(detection_levels.flatten().tolist()) <= level_numbers

    # Filtered proposals of every image with their level and the mask of the valid ones, read by the proposal recall
    assert outputs["proposal_boxes"].shape == (*outputs["valid_proposal_mask"].shape, 4)
    assert outputs["proposal_levels"].shape == outputs["valid_proposal_mask"].shape
    assert outputs["valid_proposal_mask"].dtype == torch.bool and bool(outputs["valid_proposal_mask"].any())
    assert set(outputs["proposal_levels"][outputs["valid_proposal_mask"]].tolist()) <= level_numbers

    # Anchors of every level in network input pixels, read by the anchor recall at launch
    level_anchors = model.get_level_anchors()
    assert set(level_anchors) == level_numbers
    assert all(anchors.dim() == 2 and anchors.shape[1] == 4 and anchors.shape[0] > 0
               for anchors in level_anchors.values())

    # Every valid detection is the decoded box of its proposal for its class, clipped to the image
    for image_index in range(images.shape[0]):
        for detection_index in range(int(outputs["number_detections"][image_index])):
            proposal_index = int(outputs["final_detection_proposal_indices"][image_index, detection_index])
            label = int(outputs["final_detection_labels"][image_index, detection_index])
            proposal_box = outputs["detection_boxes"][image_index, proposal_index, label].clamp(
                min=0.0, max=float(model.input_image_size))
            assert torch.allclose(input=proposal_box, other=outputs["final_detection_boxes"][image_index,
                                                                                             detection_index])
