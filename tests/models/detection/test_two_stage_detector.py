"""
Tests of the intermediate results of vision_training/models/detection/two_stage_detector.py read by the instance
segmenter: the level and matched ground truth of the sampled proposals and the proposal of every final detection
carried through forward_detection_stages, and the filtered proposals and the anchors of every level read by the anchor
and proposal recall of the detection trainers. The box loss of the configuration trains the detection heads, the
criterion of its final NMS suppresses the detections of the evaluation, and the padding copies of the proposal filter
never become detections of the evaluation, which match those of the deployed detector.
"""

import pytest
import torch

from tests.models.detection.test_detector_baseline import get_fixed_batch
from tests.testing_utilities import build_test_model
from vision_training.deployment.export_specifications import compare_detections
from vision_training.models.detection.inference_detector import InferenceDetector
from vision_training.models.detection.postprocessing import postprocess_detections


@pytest.mark.parametrize(argnames="neck_type", argvalues=["none", "path_aggregation"])
def test_detection_stages_carry_levels_and_matched_ground_truths(neck_type: str) -> None:
    """
    Verifies that with ground truth the sampled proposals carry their level and their matched ground truth (an index
    exactly for the positive proposals, whose label is the one of that ground truth), and that in evaluation mode the
    level of every proposal and the proposal of every final detection are returned, so that the level of every
    detection can be read, with the filtered proposals, their level and their validity for the proposal recall. The
    anchors of every level are given for the anchor recall. With a neck, the feature maps returned to the mask heads
    are those of the neck, at the size of their block with the neck channels.

    Args:
        neck_type (str): The 'Neck.type' of the detector.
    """
    model = build_test_model(dtype=torch.float64, neck_type=neck_type)
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
    for level_name, feature_map in stages["backbone_output_tensor_dictionary"].items():
        assert feature_map.shape[1:] == (model.detection_input_information[level_name]["number_channels"],
                                         *[model.detection_input_information[level_name]["feature_map_size"]] * 2)
    assert {information["number_channels"] for information in model.detection_input_information.values()} == (
        {32} if neck_type != "none" else {64})

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

    # Anchor generator of every level, holding those anchors, read by the anchor recall to find the best anchors
    level_anchor_generators = model.get_level_anchor_generators()
    assert set(level_anchor_generators) == level_numbers
    assert all(level_anchor_generators[level].anchors is level_anchors[level] for level in level_numbers)

    # Every valid detection is the decoded box of its proposal for its class, clipped to the image
    for image_index in range(images.shape[0]):
        for detection_index in range(int(outputs["number_detections"][image_index])):
            proposal_index = int(outputs["final_detection_proposal_indices"][image_index, detection_index])
            label = int(outputs["final_detection_labels"][image_index, detection_index])
            proposal_box = outputs["detection_boxes"][image_index, proposal_index, label].clamp(
                min=0.0, max=float(model.input_image_size))
            assert torch.allclose(input=proposal_box, other=outputs["final_detection_boxes"][image_index,
                                                                                             detection_index])


def test_complete_iou_box_loss_trains_the_detection_heads_only() -> None:
    """
    Verifies that the 'Detector.box_loss' set to "complete_iou" changes the localisation loss of the detection heads
    and nothing else, the region proposal network keeping its smooth L1 loss: the same seed sampling the same positives
    with both box losses, the classification losses and the region proposal localisation loss are equal, while the
    detection CIoU loss is finite, different from the smooth L1 one and backpropagates to the box regression layers of
    the detection heads.
    """
    images, boxes, labels = get_fixed_batch()
    losses = {}
    for box_loss in ["smooth_l1", "complete_iou"]:
        model = build_test_model(dtype=torch.float64, box_loss=box_loss).train()
        torch.manual_seed(seed=2)
        outputs = model(input_tensor=images, ground_truth_bounding_boxes=boxes, ground_truth_labels=labels)
        losses[box_loss] = {key: value for key, value in outputs.items() if key.endswith("_loss")}

    # Classification losses and the region proposal localisation loss unchanged, the detection one changed and finite
    for loss_name in ["region_proposal_classification_loss", "detection_classification_loss",
                      "region_proposal_localisation_loss"]:
        assert torch.equal(input=losses["complete_iou"][loss_name], other=losses["smooth_l1"][loss_name])
    complete_iou_loss = losses["complete_iou"]["detection_regression_loss"].detach()
    assert bool(torch.isfinite(input=complete_iou_loss)) and 0 < float(complete_iou_loss) < 2
    assert not torch.isclose(input=complete_iou_loss, other=losses["smooth_l1"]["detection_regression_loss"].detach())

    # Gradients of the CIoU loss reaching the box regression layers of the detection heads, the last model built
    model.zero_grad()
    losses["complete_iou"]["detection_regression_loss"].backward()
    regressor_gradients = [parameter.grad for name, parameter in model.named_parameters()
                           if name.startswith("detector_heads_dictionary") and ".bounding_box_regressor." in name]
    assert regressor_gradients and all(gradient is not None and bool(torch.isfinite(input=gradient).all())
                                       for gradient in regressor_gradients)
    assert any(float(gradient.abs().sum()) > 0 for gradient in regressor_gradients)


def test_final_nms_criterion_of_the_configuration_suppresses_the_evaluated_detections() -> None:
    """
    Verifies that the 'nms_criterion' and 'nms_distance_exponent' of the 'Detector' section reach the final
    postprocessing of the eager detector in evaluation mode: the same weights and images give the same head outputs
    with the greedy NMS and with DIoU-NMS, the final detections of each model being those postprocess_detections gives
    with its own criterion, and different from one another on the untrained detector, whose boxes overlap.
    """
    images, _, _ = get_fixed_batch()
    nms_settings = {"iou": 1.0, "distance_iou": 0.6}
    outputs, models = {}, {}
    for nms_criterion, nms_distance_exponent in nms_settings.items():
        models[nms_criterion] = build_test_model(dtype=torch.float64, nms_criterion=nms_criterion,
                                                 nms_distance_exponent=nms_distance_exponent)
        assert models[nms_criterion].detection_nms_criterion == nms_criterion
        assert models[nms_criterion].detection_nms_distance_exponent == nms_distance_exponent
        torch.manual_seed(seed=3)
        with torch.no_grad():
            outputs[nms_criterion] = models[nms_criterion](input_tensor=images)

    # Same head outputs, and the final detections of the criterion and exponent of every model
    for key in ["classification_scores", "detection_boxes"]:
        assert torch.equal(input=outputs["iou"][key], other=outputs["distance_iou"][key])
    for nms_criterion, nms_distance_exponent in nms_settings.items():
        model, model_outputs = models[nms_criterion], outputs[nms_criterion]
        expected_detections = postprocess_detections(
            classification_scores=model_outputs["classification_scores"],
            detection_boxes=model_outputs["detection_boxes"],
            valid_proposal_mask=torch.ones_like(input=model_outputs["classification_scores"][..., 0], dtype=torch.bool),
            score_threshold=model.detection_score_threshold, nms_iou_threshold=model.detection_nms_iou_threshold,
            nms_criterion=nms_criterion, nms_distance_exponent=nms_distance_exponent,
            maximum_detections=model.maximum_detections, input_image_size=model.input_image_size,
            suppress_candidates_only=True)
        for key, expected_value in expected_detections.items():
            assert torch.equal(input=model_outputs[key], other=expected_value), f"{key} differs for {nms_criterion}"
    assert not torch.equal(input=outputs["iou"]["final_detection_boxes"],
                           other=outputs["distance_iou"]["final_detection_boxes"])


def test_padding_proposals_never_become_evaluated_detections() -> None:
    """
    Verifies that the eager detector in evaluation mode masks the padding copies of the proposal filter like the
    deployed InferenceDetector: with a proposal suppression leaving half of the 50 proposal slots to padding copies of
    random survivors, and a final NMS threshold of 1.0, which keeps identical boxes, the evaluation gives the detections
    of the deployed detector, none of them read on a padding proposal, instead of every detection of a copied survivor
    twice. The mask of the valid proposals is carried in the order of the head outputs, with as many valid proposals
    per image as the proposal filter kept.
    """
    model = build_test_model(dtype=torch.float32)
    model.region_proposal_filter.nms_iou_threshold = 0.05
    model.detection_nms_iou_threshold = 1.0
    model.maximum_detections = 5000
    detector = InferenceDetector(model=model).eval()
    images = torch.rand(size=(2, 3, 512, 512), generator=torch.Generator().manual_seed(0)) * 255.0

    with torch.no_grad():
        eager_outputs = model(input_tensor=images)
        inference_outputs = detector(images=images)

    # Half of the proposal slots are padding copies
    survivor_counts = eager_outputs["valid_proposal_mask"].sum(dim=1)
    assert survivor_counts.tolist() == [25, 25] and eager_outputs["valid_proposal_mask"].shape[1] == 50

    # Same detections as the deployed detector
    assert torch.equal(input=eager_outputs["number_detections"], other=inference_outputs[3])
    assert compare_detections(reference=[eager_outputs[key].numpy() for key in [
        "final_detection_boxes", "final_detection_scores", "final_detection_labels", "number_detections"]],
        candidate=[output.numpy() for output in inference_outputs], box_tolerance=1e-4, score_tolerance=1e-6) == []

    # The mask in the order of the head outputs counts the same survivors per image, every detection being read on
    # a valid proposal
    assert torch.equal(input=eager_outputs["sliced_valid_mask"].sum(dim=1), other=survivor_counts)
    for image_index in range(images.shape[0]):
        number_detections = int(eager_outputs["number_detections"][image_index])
        detection_proposals = eager_outputs["final_detection_proposal_indices"][image_index, :number_detections]
        assert bool(eager_outputs["sliced_valid_mask"][image_index, detection_proposals].all())
