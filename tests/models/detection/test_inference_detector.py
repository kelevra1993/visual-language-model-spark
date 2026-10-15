"""
Tests of the detection inference path of vision_training/models/detection: postprocess_detections (threshold, NMS and
padding, fewer candidates than detections), InferenceDetector matching the eager detector, its TorchScript round
trip, and a detector asking for more proposals than it has anchors.
"""

import torch

from pathlib import Path

from vision_training.models.detection.inference_detector import InferenceDetector
from tests.testing_utilities import build_test_model
from vision_training.models.detection.postprocessing import postprocess_detections


def test_postprocess_detections_threshold_nms_and_padding() -> None:
    """
    Verifies the final post-processing on hand-made predictions: the background class is ignored, low scores are
    removed, overlapping boxes of the same class are suppressed while other classes are kept, boxes are clipped to
    the image, padding proposals never produce detections, the output is padded to a fixed size, and every detection
    gives the index of its proposal.
    """
    # One image, four proposals, classes (background, A, B); the last proposal is padding
    probabilities = torch.tensor(data=[[[0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.05, 0.1, 0.85], [0.0, 0.9, 0.1]]])
    first_box, shifted_box = [10.0, 10.0, 50.0, 50.0], [12.0, 12.0, 52.0, 52.0]
    far_box, padding_box = [100.0, 100.0, 150.0, 150.0], [0.0, 0.0, 40.0, 40.0]
    boxes = torch.tensor(data=[[[box] * 3 for box in [first_box, shifted_box, far_box, padding_box]]])

    detections = postprocess_detections(classification_scores=torch.log(input=probabilities), detection_boxes=boxes,
                                        valid_proposal_mask=torch.tensor(data=[[True, True, True, False]]),
                                        score_threshold=0.5, nms_iou_threshold=0.5, maximum_detections=4,
                                        input_image_size=128)

    assert detections["number_detections"].tolist() == [2]
    assert detections["final_detection_labels"].tolist() == [[2, 1, 0, 0]]
    assert torch.allclose(input=detections["final_detection_scores"], other=torch.tensor(data=[[0.85, 0.8, 0.0, 0.0]]))
    assert detections["final_detection_boxes"][0, :2].tolist() == [[100.0, 100.0, 128.0, 128.0], first_box]
    assert detections["final_detection_boxes"][0, 2:].abs().sum() == 0.0
    assert detections["final_detection_proposal_indices"].tolist() == [[2, 0, 0, 0]]


def test_inference_detector_matches_the_eager_model() -> None:
    """
    Verifies that the export-friendly inference detector reproduces exactly the detections of the eager
    TwoStageDetector in evaluation mode, on the same weights and images.
    """
    model = build_test_model(dtype=torch.float32)
    detector = InferenceDetector(model=model).eval()
    images = torch.rand(size=(2, 3, 512, 512)) * 255.0

    with torch.no_grad():
        eager_outputs = model(input_tensor=images)
        boxes, scores, labels, numbers = detector(images=images)

    assert torch.equal(input=numbers, other=eager_outputs["number_detections"])
    assert torch.equal(input=labels, other=eager_outputs["final_detection_labels"])
    assert torch.allclose(input=scores, other=eager_outputs["final_detection_scores"])
    assert torch.allclose(input=boxes, other=eager_outputs["final_detection_boxes"])


def test_torchscript_round_trip(tmp_path: Path) -> None:
    """
    Verifies that the traced TorchScript detector, saved and reloaded like the Triton PyTorch backend does,
    reproduces the eager inference detector, including for a batch size different from the traced one.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    detector = InferenceDetector(model=build_test_model(dtype=torch.float32)).eval()
    model_path = tmp_path / "model.pt"

    with torch.no_grad():
        torch.jit.trace(detector, example_inputs=(torch.rand(size=(2, 3, 512, 512)) * 255.0,),
                        check_trace=False).save(str(model_path))
        images = torch.rand(size=(3, 3, 512, 512)) * 255.0
        reference_outputs = detector(images=images)
        traced_outputs = torch.jit.load(str(model_path))(images)

    for reference, traced in zip(reference_outputs, traced_outputs):
        assert torch.allclose(input=traced.to(dtype=reference.dtype), other=reference)


def test_postprocess_detections_with_fewer_candidates_than_detections() -> None:
    """
    Verifies that an image with fewer (proposal, foreground class) pairs than maximum_detections, as in the validation
    classifying the sampled proposals only, still gives the fixed number of detections, the missing ones padded.
    """
    # One image, two proposals, classes (background, A, B): four candidates for six detections
    probabilities = torch.tensor(data=[[[0.1, 0.8, 0.1], [0.2, 0.1, 0.7]]])
    boxes = torch.tensor(data=[[[[10.0, 10.0, 50.0, 50.0]] * 3, [[60.0, 60.0, 90.0, 90.0]] * 3]])
    detections = postprocess_detections(classification_scores=torch.log(input=probabilities), detection_boxes=boxes,
                                        valid_proposal_mask=torch.ones(size=(1, 2), dtype=torch.bool),
                                        score_threshold=0.5, nms_iou_threshold=0.5, maximum_detections=6,
                                        input_image_size=128)
    assert detections["final_detection_scores"].shape == (1, 6) and detections["number_detections"].tolist() == [2]
    assert detections["final_detection_labels"].tolist() == [[1, 2, 0, 0, 0, 0]]
    assert detections["final_detection_proposal_indices"].tolist() == [[0, 1, 0, 0, 0, 0]]

    # The padded detections carry a zero score and a zero box, like the detections removed by the threshold
    assert torch.allclose(input=detections["final_detection_scores"],
                          other=torch.tensor(data=[[0.8, 0.7, 0.0, 0.0, 0.0, 0.0]]))
    final_boxes = detections["final_detection_boxes"][0]
    assert final_boxes[2:].abs().sum() == 0 and final_boxes[:2].abs().sum() > 0


def test_detector_with_fewer_anchors_than_proposals(tmp_path: Path) -> None:
    """
    Verifies that a detector asking for more proposals than it has anchors (a small image with coarse levels) trains,
    validates and is traced for the export, every image keeping one proposal per anchor at most, instead of crashing
    in the top-k selections, and that the proposals returned with ground truth are those of the network, without the
    ground truth boxes added for the detection heads.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    model = build_test_model(dtype=torch.float32)
    proposal_filter = model.region_proposal_filter
    too_many_proposals = model.number_anchors + 50
    proposal_filter.training_pre_nms_proposals = proposal_filter.training_post_nms_proposals = too_many_proposals
    proposal_filter.inference_pre_nms_proposals = proposal_filter.inference_post_nms_proposals = too_many_proposals
    images = torch.rand(size=(2, 3, 512, 512)) * 255.0
    boxes = [torch.tensor(data=[[40.0, 60.0, 200.0, 220.0]]), torch.tensor(data=[[250.0, 100.0, 480.0, 330.0]])]
    labels = [torch.tensor(data=[1]), torch.tensor(data=[2])]

    with torch.no_grad():
        # Training and validation losses, with the ground truth boxes added to the proposals
        training_outputs = model.train()(input_tensor=images, ground_truth_bounding_boxes=boxes,
                                         ground_truth_labels=labels)
        validation_outputs = model.eval()(input_tensor=images, ground_truth_bounding_boxes=boxes,
                                          ground_truth_labels=labels)
        assert all(bool(torch.isfinite(input=training_outputs[key])) for key in training_outputs
                   if key.endswith("_loss"))
        assert validation_outputs["final_detection_scores"].shape == (2, model.maximum_detections)

        # Evaluation without ground truth, one proposal per anchor at most
        evaluation_outputs = model(input_tensor=images)
        assert evaluation_outputs["proposal_boxes"].shape == (2, model.number_anchors, 4)

        # The proposal recall reads the proposals of the network, the ground truth boxes being added to a copy only:
        # the real proposals are the same with and without ground truth, the padding slots being random copies
        valid_proposal_mask = evaluation_outputs["valid_proposal_mask"]
        assert torch.equal(validation_outputs["valid_proposal_mask"], valid_proposal_mask)
        for key in ["proposal_boxes", "proposal_levels"]:
            assert torch.equal(validation_outputs[key][valid_proposal_mask],
                               evaluation_outputs[key][valid_proposal_mask])
        for image_index, image_boxes in enumerate(boxes):
            assert not (validation_outputs["proposal_boxes"][image_index] == image_boxes[0]).all(dim=1).any()

        # Deployed detector, eager and traced
        detector = InferenceDetector(model=model).eval()
        model_path = tmp_path / "model.pt"
        torch.jit.trace(detector, example_inputs=(images,), check_trace=False).save(str(model_path))
        reference_outputs = detector(images=images)
        traced_outputs = torch.jit.load(str(model_path))(images)
    assert reference_outputs[0].shape == (2, model.maximum_detections, 4)
    for reference, traced in zip(reference_outputs, traced_outputs):
        assert torch.allclose(input=traced.to(dtype=reference.dtype), other=reference)
