"""
Tests of the detection inference path of vision_training/models/detection: postprocess_detections (threshold, NMS and
padding, the NMS run on the candidates reaching the score threshold only and giving the detections of the NMS on every
pair run by the exports, DIoU-NMS in the eager path only), InferenceDetector matching the eager detector, its
TorchScript round trip, and its refusal of DIoU-NMS.
"""

import pytest
import torch

from functools import partial
from pathlib import Path
from typing import List, Tuple
from torchvision.ops import batched_nms

from vision_training.models.detection import postprocessing
from vision_training.models.detection.inference_detector import InferenceDetector
from vision_training.models.instance_segmentation.inference_instance_segmenter import InferenceInstanceSegmenter
from tests.testing_utilities import build_test_model
from vision_training.models.detection.postprocessing import postprocess_detections


@pytest.mark.parametrize(argnames="suppress_candidates_only", argvalues=[True])
def test_postprocess_detections_threshold_nms_and_padding(suppress_candidates_only: bool) -> None:
    """
    Verifies the final post-processing on hand-made predictions: the background class is ignored, low scores are
    removed, overlapping boxes of the same class are suppressed while other classes are kept, boxes are clipped to
    the image, padding proposals never produce detections, the output is padded to a fixed size, and every detection
    gives the index of its proposal.

    Args:
        suppress_candidates_only (bool): The NMS on the candidates reaching the threshold (eager) or on every pair.
    """
    # One image, four proposals, classes (background, A, B); the last proposal is padding
    probabilities = torch.tensor(data=[[[0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.05, 0.1, 0.85], [0.0, 0.9, 0.1]]])
    first_box, shifted_box = [10.0, 10.0, 50.0, 50.0], [12.0, 12.0, 52.0, 52.0]
    far_box, padding_box = [100.0, 100.0, 150.0, 150.0], [0.0, 0.0, 40.0, 40.0]
    boxes = torch.tensor(data=[[[box] * 3 for box in [first_box, shifted_box, far_box, padding_box]]])

    detections = postprocess_detections(classification_scores=torch.log(input=probabilities), detection_boxes=boxes,
                                        valid_proposal_mask=torch.tensor(data=[[True, True, True, False]]),
                                        score_threshold=0.5, nms_iou_threshold=0.5, nms_criterion="iou",
                                        nms_distance_exponent=1.0, maximum_detections=4, input_image_size=128,
                                        suppress_candidates_only=suppress_candidates_only)

    assert detections["number_detections"].tolist() == [2]
    assert detections["final_detection_labels"].tolist() == [[2, 1, 0, 0]]
    assert torch.allclose(input=detections["final_detection_scores"], other=torch.tensor(data=[[0.85, 0.8, 0.0, 0.0]]))
    assert detections["final_detection_boxes"][0, :2].tolist() == [[100.0, 100.0, 128.0, 128.0], first_box]
    assert detections["final_detection_boxes"][0, 2:].abs().sum() == 0.0
    assert detections["final_detection_proposal_indices"].tolist() == [[2, 0, 0, 0]]


def test_postprocess_detections_with_distance_nms_and_its_refusals() -> None:
    """
    Verifies the criterion of the final NMS of postprocess_detections on hand-made predictions: a lower scoring box of
    the class of the best box, overlapping it by 0.526 with its centre 4.5 pixels away, is suppressed by the greedy NMS
    and kept by DIoU-NMS, the box of another class staying in both, with the proposal of every detection; the export
    path refuses DIoU-NMS, and both paths refuse an unknown criterion.
    """
    # One image, three proposals, classes (background, A, B): the best box of A, a wider box of A sharing its right
    # edge, and a box of B apart
    probabilities = torch.tensor(data=[[[0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.05, 0.1, 0.85]]])
    first_box, wider_box, far_box = [10.0, 10.0, 20.0, 20.0], [1.0, 10.0, 20.0, 20.0], [60.0, 60.0, 90.0, 90.0]
    boxes = torch.tensor(data=[[[box] * 3 for box in [first_box, wider_box, far_box]]])
    settings = {"classification_scores": torch.log(input=probabilities), "detection_boxes": boxes,
                "valid_proposal_mask": torch.ones(size=(1, 3), dtype=torch.bool), "score_threshold": 0.5,
                "nms_iou_threshold": 0.5, "nms_distance_exponent": 1.0, "maximum_detections": 4,
                "input_image_size": 128}

    greedy_detections = postprocess_detections(**settings, nms_criterion="iou", suppress_candidates_only=True)
    assert greedy_detections["number_detections"].tolist() == [2]
    assert greedy_detections["final_detection_labels"].tolist() == [[2, 1, 0, 0]]
    distance_detections = postprocess_detections(**settings, nms_criterion="distance_iou",
                                                 suppress_candidates_only=True)
    assert distance_detections["number_detections"].tolist() == [3]
    assert distance_detections["final_detection_labels"].tolist() == [[2, 1, 1, 0]]
    assert distance_detections["final_detection_boxes"][0, :3].tolist() == [far_box, first_box, wider_box]
    assert distance_detections["final_detection_proposal_indices"].tolist() == [[2, 0, 1, 0]]

    # DIoU-NMS has no exportable form, and an unknown criterion is refused by both paths
    with pytest.raises(expected_exception=ValueError, match=r"greedy NMS \('iou'\) only, got 'distance_iou'"):
        postprocess_detections(**settings, nms_criterion="distance_iou", suppress_candidates_only=False)
    for suppress_candidates_only in [True, False]:
        with pytest.raises(expected_exception=ValueError, match="'soft_nms'"):
            postprocess_detections(**settings, nms_criterion="soft_nms",
                                   suppress_candidates_only=suppress_candidates_only)


@pytest.mark.parametrize(argnames="instance_segmentation", argvalues=[False, True])
def test_inference_models_refuse_the_distance_nms(instance_segmentation: bool) -> None:
    """
    Verifies that the deployed detector and instance segmenter, which run the greedy NMS on every pair, refuse a model
    whose 'Detector.nms_criterion' is "distance_iou", so that a deployed model never suppresses otherwise than in its
    evaluation, while the eager model evaluates it.

    Args:
        instance_segmentation (bool): Whether the model is the instance segmenter.
    """
    model = build_test_model(dtype=torch.float32, instance_segmentation=instance_segmentation,
                             nms_criterion="distance_iou")
    assert model.detection_nms_criterion == "distance_iou"
    inference_class = InferenceInstanceSegmenter if instance_segmentation else InferenceDetector
    with pytest.raises(expected_exception=ValueError, match="'Detector.nms_criterion' is 'distance_iou'"):
        inference_class(model=model)


@pytest.mark.parametrize(argnames="neck_type", argvalues=["none", "path_aggregation"])
def test_inference_detector_matches_the_eager_model(neck_type: str) -> None:
    """
    Verifies that the export-friendly inference detector reproduces exactly the detections of the eager
    TwoStageDetector in evaluation mode, on the same weights and images, without neck and with the neck applying both
    the top-down and the bottom-up paths.

    Args:
        neck_type (str): The 'Neck.type' of the detector.
    """
    model = build_test_model(dtype=torch.float32, neck_type=neck_type)
    detector = InferenceDetector(model=model).eval()
    images = torch.rand(size=(2, 3, 512, 512)) * 255.0

    with torch.no_grad():
        eager_outputs = model(input_tensor=images)
        boxes, scores, labels, numbers = detector(images=images)

    assert torch.equal(input=numbers, other=eager_outputs["number_detections"])
    assert torch.equal(input=labels, other=eager_outputs["final_detection_labels"])
    assert torch.allclose(input=scores, other=eager_outputs["final_detection_scores"])
    assert torch.allclose(input=boxes, other=eager_outputs["final_detection_boxes"])


@pytest.mark.parametrize(argnames="neck_type", argvalues=["none", "path_aggregation"])
def test_torchscript_round_trip(tmp_path: Path, neck_type: str) -> None:
    """
    Verifies that the traced TorchScript detector, saved and reloaded like the Triton PyTorch backend does,
    reproduces the eager inference detector, including for a batch size different from the traced one, without neck
    and with the neck whose level sizes are fixed numbers in the trace.

    Args:
        tmp_path (Path): The pytest temporary directory.
        neck_type (str): The 'Neck.type' of the detector.
    """
    detector = InferenceDetector(model=build_test_model(dtype=torch.float32, neck_type=neck_type)).eval()
    model_path = tmp_path / "model.pt"

    with torch.no_grad():
        torch.jit.trace(detector, example_inputs=(torch.rand(size=(2, 3, 512, 512)) * 255.0,),
                        check_trace=False).save(str(model_path))
        images = torch.rand(size=(3, 3, 512, 512)) * 255.0
        reference_outputs = detector(images=images)
        traced_outputs = torch.jit.load(str(model_path))(images)

    for reference, traced in zip(reference_outputs, traced_outputs):
        assert torch.allclose(input=traced.to(dtype=reference.dtype), other=reference)


def record_batched_nms(boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float,
                       recorded_box_counts: List[int]) -> torch.Tensor:
    """
    Wraps the torchvision batched NMS called by postprocess_detections to record the number of boxes it receives.

    Args:
        boxes (torch.Tensor): The [N, 4] boxes given to the NMS.
        scores (torch.Tensor): Their [N] scores.
        idxs (torch.Tensor): Their [N] NMS groups (image and class).
        iou_threshold (float): The IoU above which a lower scoring box of the same group is suppressed.
        recorded_box_counts (List[int]): The list receiving the number of boxes of every call.

    Returns:
        torch.Tensor: The indices kept by the torchvision batched NMS.
    """
    recorded_box_counts.append(len(boxes))
    return batched_nms(boxes=boxes, scores=scores, idxs=idxs, iou_threshold=iou_threshold)


def build_random_head_outputs(seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Builds random detection head outputs of a batch for the postprocessing tests: class logits spread enough for some
    pairs to pass the score thresholds, boxes gathered around a few centres so that the NMS suppresses many of them
    (some crossing the image border), and a few padding proposals.

    Args:
        seed (int): The seed of the random generator.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The [B, P, C] class logits, the [B, P, C, 4] boxes and the
            [B, P] valid proposal mask.
    """
    generator = torch.Generator().manual_seed(seed)
    batch_size, number_proposals, number_classes = 3, 120, 5

    # Logits of the background class raised, so that most pairs stay below the thresholds as on real images
    classification_scores = torch.randn(size=(batch_size, number_proposals, number_classes), generator=generator) * 2.0
    classification_scores[..., 0] += 2.0

    # Boxes around four centres per image, of 10 to 50 pixels, in a 128 pixel image
    centre_indices = torch.randint(low=0, high=4, size=(batch_size, number_proposals, number_classes),
                                   generator=generator)
    centres = torch.rand(size=(batch_size, 4, 2), generator=generator) * 128.0
    box_centres = torch.gather(input=centres, dim=1, index=centre_indices.reshape(batch_size, -1, 1).expand(-1, -1, 2))
    box_centres = box_centres.reshape(batch_size, number_proposals, number_classes, 2)
    box_centres = box_centres + torch.randn(size=box_centres.shape, generator=generator) * 3.0
    half_sizes = 5.0 + torch.rand(size=box_centres.shape, generator=generator) * 20.0
    detection_boxes = torch.cat(tensors=[box_centres - half_sizes, box_centres + half_sizes], dim=-1)

    # The last proposals of every image are padding
    valid_proposal_mask = torch.ones(size=(batch_size, number_proposals), dtype=torch.bool)
    valid_proposal_mask[:, -10:] = False
    return classification_scores, detection_boxes, valid_proposal_mask


def test_postprocess_detections_runs_the_nms_on_the_candidates_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the eager path gives the batched NMS the pairs of valid proposals reaching the score threshold only,
    instead of every (proposal, foreground class) pair of the batch, the evaluation of a whole batch otherwise falling
    into the slow per-group NMS loop of torchvision.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the batched NMS of postprocess_detections by a recording one.
    """
    classification_scores, detection_boxes, valid_proposal_mask = build_random_head_outputs(seed=0)
    recorded_box_counts = []
    monkeypatch.setattr(target=postprocessing, name="batched_nms",
                        value=partial(record_batched_nms, recorded_box_counts=recorded_box_counts))
    postprocess_detections(classification_scores=classification_scores, detection_boxes=detection_boxes,
                           valid_proposal_mask=valid_proposal_mask, score_threshold=0.3, nms_iou_threshold=0.5,
                           nms_criterion="iou", nms_distance_exponent=1.0, maximum_detections=20, input_image_size=128,
                           suppress_candidates_only=True)

    # Number of valid pairs reaching the threshold, a small part of all the pairs
    probabilities = torch.softmax(input=classification_scores, dim=-1)[..., 1:]
    number_candidates = int(((probabilities >= 0.3) & valid_proposal_mask.unsqueeze(dim=-1)).sum())
    assert 0 < number_candidates < probabilities.numel() // 2
    assert recorded_box_counts == [number_candidates]


def test_postprocess_detections_match_the_nms_on_every_pair() -> None:
    """
    Verifies that the NMS on the candidates reaching the score threshold (eager) returns exactly the detections of the
    NMS on every pair (exports), on random batches with the evaluation threshold, the inference threshold and a
    threshold no pair reaches (an empty NMS input).
    """
    for seed in range(4):
        for score_threshold in [0.05, 0.5, 1.5]:
            classification_scores, detection_boxes, valid_proposal_mask = build_random_head_outputs(seed=seed)
            settings = {"classification_scores": classification_scores, "detection_boxes": detection_boxes,
                        "valid_proposal_mask": valid_proposal_mask, "score_threshold": score_threshold,
                        "nms_iou_threshold": 0.5, "nms_criterion": "iou", "nms_distance_exponent": 1.0,
                        "maximum_detections": 100, "input_image_size": 128}

            # Eager path on the candidates, then the reference on every pair
            detections = postprocess_detections(**settings, suppress_candidates_only=True)
            reference_detections = postprocess_detections(**settings, suppress_candidates_only=False)

            # Same detections, and detections found on the batch except with the threshold no pair reaches
            for key, reference in reference_detections.items():
                assert torch.equal(detections[key], reference), \
                    f"{key} differs, seed {seed}, threshold {score_threshold}"
            assert (int(detections["number_detections"].sum()) == 0) == (score_threshold > 1.0)
