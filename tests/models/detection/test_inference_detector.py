"""
Tests of the detection inference path of vision_training/models/detection: postprocess_detections (threshold, NMS and
padding, the NMS run on the candidates reaching the score threshold only and giving the detections of the NMS on every
pair run by the exports, with the greedy NMS and with DIoU-NMS, the boxes clipped to the window of their image),
InferenceDetector matching the eager detector with either criterion, and its TorchScript round trip with the image
windows as a second input.
"""

import pytest
import torch

from functools import partial
from pathlib import Path
from typing import List, Tuple
from torchvision.ops import batched_nms

from vision_training.deployment.export_specifications import compare_detections
from vision_training.models.detection import postprocessing
from vision_training.models.detection.inference_detector import InferenceDetector
from tests.testing_utilities import build_test_model, build_whole_input_windows
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
                                        image_windows=torch.tensor(data=[[0.0, 0.0, 128.0, 128.0]]),
                                        score_threshold=0.5, nms_iou_threshold=0.5, nms_criterion="iou",
                                        nms_distance_exponent=1.0, maximum_detections=4, input_image_size=128,
                                        suppress_candidates_only=suppress_candidates_only)

    assert detections["number_detections"].tolist() == [2]
    assert detections["final_detection_labels"].tolist() == [[2, 1, 0, 0]]
    assert torch.allclose(input=detections["final_detection_scores"], other=torch.tensor(data=[[0.85, 0.8, 0.0, 0.0]]))
    assert detections["final_detection_boxes"][0, :2].tolist() == [[100.0, 100.0, 128.0, 128.0], first_box]
    assert detections["final_detection_boxes"][0, 2:].abs().sum() == 0.0
    assert detections["final_detection_proposal_indices"].tolist() == [[2, 0, 0, 0]]


def test_postprocess_detections_with_distance_nms_in_both_paths() -> None:
    """
    Verifies the criterion of the final NMS of postprocess_detections on hand-made predictions: a lower scoring box of
    the class of the best box, overlapping it by 0.526 with its centre 4.5 pixels away, is suppressed by the greedy NMS
    and kept by DIoU-NMS, the box of another class staying in both, with the proposal of every detection, in the eager
    path and in the export path; both paths refuse an unknown criterion.
    """
    # One image, three proposals, classes (background, A, B): the best box of A, a wider box of A sharing its right
    # edge, and a box of B apart
    probabilities = torch.tensor(data=[[[0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.05, 0.1, 0.85]]])
    first_box, wider_box, far_box = [10.0, 10.0, 20.0, 20.0], [1.0, 10.0, 20.0, 20.0], [60.0, 60.0, 90.0, 90.0]
    boxes = torch.tensor(data=[[[box] * 3 for box in [first_box, wider_box, far_box]]])
    settings = {"classification_scores": torch.log(input=probabilities), "detection_boxes": boxes,
                "valid_proposal_mask": torch.ones(size=(1, 3), dtype=torch.bool),
                "image_windows": torch.tensor(data=[[0.0, 0.0, 128.0, 128.0]]), "score_threshold": 0.5,
                "nms_iou_threshold": 0.5, "nms_distance_exponent": 1.0, "maximum_detections": 4,
                "input_image_size": 128}

    # The eager path on the candidates and the export path on every pair keep the same detections with either criterion
    for suppress_candidates_only in [True, False]:
        greedy_detections = postprocess_detections(**settings, nms_criterion="iou",
                                                   suppress_candidates_only=suppress_candidates_only)
        assert greedy_detections["number_detections"].tolist() == [2]
        assert greedy_detections["final_detection_labels"].tolist() == [[2, 1, 0, 0]]
        distance_detections = postprocess_detections(**settings, nms_criterion="distance_iou",
                                                     suppress_candidates_only=suppress_candidates_only)
        assert distance_detections["number_detections"].tolist() == [3]
        assert distance_detections["final_detection_labels"].tolist() == [[2, 1, 1, 0]]
        assert distance_detections["final_detection_boxes"][0, :3].tolist() == [far_box, first_box, wider_box]
        assert distance_detections["final_detection_proposal_indices"].tolist() == [[2, 0, 1, 0]]

    # An unknown criterion is refused by both paths
    for suppress_candidates_only in [True, False]:
        with pytest.raises(expected_exception=ValueError, match="'soft_nms'"):
            postprocess_detections(**settings, nms_criterion="soft_nms",
                                   suppress_candidates_only=suppress_candidates_only)


def test_distance_nms_paths_agree_at_the_threshold_and_on_equal_scores() -> None:
    """
    Verifies the two edge cases on which the export path of DIoU-NMS could differ from its eager path on hand-made
    predictions: two boxes of a class sharing their centre at an IoU of exactly 0.5 are both kept, the suppression
    being strictly above the threshold, and of two overlapping boxes of a class with exactly equal scores, the one of
    the earlier proposal is kept, as the stable sort of batched_distance_nms orders them.
    """
    # One image, four proposals, classes (background, A): a box of A and a narrower box of A sharing its centre (IoU
    # of 0.5), then two boxes of A overlapping by 0.905 with the same logits, hence exactly equal scores
    probabilities = torch.tensor(data=[[[0.4, 0.6], [0.5, 0.5], [0.3, 0.7], [0.3, 0.7]]])
    boxes = [[10.0, 10.0, 14.0, 14.0], [11.0, 10.0, 13.0, 14.0], [40.0, 40.0, 60.0, 60.0], [41.0, 40.0, 61.0, 60.0]]
    settings = {"classification_scores": torch.log(input=probabilities),
                "detection_boxes": torch.tensor(data=[[[box] * 2 for box in boxes]]),
                "valid_proposal_mask": torch.ones(size=(1, 4), dtype=torch.bool),
                "image_windows": torch.tensor(data=[[0.0, 0.0, 128.0, 128.0]]), "score_threshold": 0.3,
                "nms_iou_threshold": 0.5, "nms_criterion": "distance_iou", "nms_distance_exponent": 0.6,
                "maximum_detections": 4, "input_image_size": 128}

    # Both paths keep the best box of the tied pair, the earlier proposal, and both boxes at the threshold
    for suppress_candidates_only in [True, False]:
        detections = postprocess_detections(**settings, suppress_candidates_only=suppress_candidates_only)
        assert detections["number_detections"].tolist() == [3]
        assert detections["final_detection_proposal_indices"].tolist() == [[2, 0, 1, 0]]
        assert detections["final_detection_boxes"][0, :3].tolist() == [boxes[2], boxes[0], boxes[1]]


@pytest.mark.parametrize(argnames=("neck_type", "nms_criterion", "nms_distance_exponent", "proposal_selection"),
                         argvalues=[("none", "iou", 1.0, "joint"), ("path_aggregation", "iou", 1.0, "joint"),
                                    ("none", "distance_iou", 0.6, "joint"),
                                    ("path_aggregation", "iou", 1.0, "per_level")])
def test_inference_detector_matches_the_eager_model(neck_type: str, nms_criterion: str, nms_distance_exponent: float,
                                                    proposal_selection: str) -> None:
    """
    Verifies that the export-friendly inference detector reproduces exactly the detections of the eager
    TwoStageDetector in evaluation mode, on the same weights and images, without neck and with the neck applying both
    the top-down and the bottom-up paths, with the greedy NMS and with DIoU-NMS, whose suppression on every pair of
    the exports keeps the detections of the eager evaluation, and with the proposals selected over all the levels and
    level by level. The detections are paired by label and score like the checks of the exports: the untrained
    detector gives exactly equal scores to many proposals, which DIoU-NMS keeps side by side, and the two models,
    holding their proposals in another order, list such detections in another order.

    Args:
        neck_type (str): The 'Neck.type' of the detector.
        nms_criterion (str): The 'Detector.nms_criterion' of the final NMS.
        nms_distance_exponent (float): The 'Detector.nms_distance_exponent' of "distance_iou".
        proposal_selection (str): The 'RegionProposal.proposal_selection' of the proposal filter.
    """
    model = build_test_model(dtype=torch.float32, neck_type=neck_type, nms_criterion=nms_criterion,
                             nms_distance_exponent=nms_distance_exponent, proposal_selection=proposal_selection)
    detector = InferenceDetector(model=model).eval()
    images = torch.rand(size=(2, 3, 512, 512)) * 255.0

    with torch.no_grad():
        eager_outputs = model(input_tensor=images, image_windows=build_whole_input_windows(images=images))
        inference_outputs = detector(images=images, image_windows=build_whole_input_windows(images=images))

    # Same numbers of detections, scores sorted alike, and every detection paired with one of the same label, score
    # and box
    eager_detections = [eager_outputs[key].numpy() for key in ["final_detection_boxes", "final_detection_scores",
                                                                "final_detection_labels", "number_detections"]]
    assert torch.equal(input=inference_outputs[3], other=eager_outputs["number_detections"])
    assert torch.allclose(input=inference_outputs[1], other=eager_outputs["final_detection_scores"])
    assert compare_detections(reference=eager_detections, candidate=[output.numpy() for output in inference_outputs],
                              box_tolerance=1e-4, score_tolerance=1e-6) == []

    # The greedy NMS suppresses the tied overlapping detections, so its detections also come in the same order
    if nms_criterion == "iou":
        assert torch.equal(input=inference_outputs[2], other=eager_outputs["final_detection_labels"])
        assert torch.allclose(input=inference_outputs[0], other=eager_outputs["final_detection_boxes"])


@pytest.mark.parametrize(argnames=("neck_type", "nms_criterion", "proposal_selection"),
                         argvalues=[("none", "iou", "joint"), ("path_aggregation", "iou", "joint"),
                                    ("none", "distance_iou", "joint"), ("none", "iou", "per_level")])
def test_torchscript_round_trip(tmp_path: Path, neck_type: str, nms_criterion: str, proposal_selection: str) -> None:
    """
    Verifies that the traced TorchScript detector, saved and reloaded like the Triton PyTorch backend does,
    reproduces the eager inference detector, including for a batch size different from the traced one and windows
    other than the traced ones (the windows being an input of the graph, every box staying inside the window of its
    image), without neck and with the neck whose level sizes are fixed numbers in the trace, with the greedy NMS and
    with DIoU-NMS, whose scan of the proposal ranks the trace unrolls, and with the proposals selected level by level,
    whose loop over the levels the trace unrolls too.

    Args:
        tmp_path (Path): The pytest temporary directory.
        neck_type (str): The 'Neck.type' of the detector.
        nms_criterion (str): The 'Detector.nms_criterion' of the final NMS, "distance_iou" with the exponent 0.6.
        proposal_selection (str): The 'RegionProposal.proposal_selection' of the proposal filter.
    """
    detector = InferenceDetector(model=build_test_model(dtype=torch.float32, neck_type=neck_type,
                                                        nms_criterion=nms_criterion, nms_distance_exponent=0.6,
                                                        proposal_selection=proposal_selection)).eval()
    model_path = tmp_path / "model.pt"

    # Traced with the whole input as window, then checked on a letterboxed landscape, a portrait and a whole image
    with torch.no_grad():
        traced_images = torch.rand(size=(2, 3, 512, 512)) * 255.0
        torch.jit.trace(detector, example_inputs=(traced_images, build_whole_input_windows(images=traced_images)),
                        check_trace=False).save(str(model_path))
        images = torch.rand(size=(3, 3, 512, 512)) * 255.0
        image_windows = torch.tensor(data=[[0.0, 128.0, 512.0, 384.0], [192.0, 0.0, 320.0, 512.0],
                                           [0.0, 0.0, 512.0, 512.0]])
        reference_outputs = detector(images=images, image_windows=image_windows)
        traced_outputs = torch.jit.load(str(model_path))(images, image_windows)

    for reference, traced in zip(reference_outputs, traced_outputs):
        assert torch.allclose(input=traced.to(dtype=reference.dtype), other=reference)

    # Every valid detection inside the window of its image
    valid_detections = torch.arange(end=reference_outputs[0].shape[1]).unsqueeze(dim=0) < \
        reference_outputs[3].unsqueeze(dim=1)
    boxes = reference_outputs[0]
    assert bool(valid_detections[:2].any())
    assert bool(((boxes[..., 0] >= image_windows[:, None, 0]) & (boxes[..., 1] >= image_windows[:, None, 1]) &
                 (boxes[..., 2] <= image_windows[:, None, 2]) & (boxes[..., 3] <= image_windows[:, None, 3]))[
        valid_detections].all())


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
                           valid_proposal_mask=valid_proposal_mask,
                           image_windows=torch.tensor(data=[[0.0, 0.0, 128.0, 128.0]]).expand(3, -1),
                           score_threshold=0.3, nms_iou_threshold=0.5,
                           nms_criterion="iou", nms_distance_exponent=1.0, maximum_detections=20, input_image_size=128,
                           suppress_candidates_only=True)

    # Number of valid pairs reaching the threshold, a small part of all the pairs
    probabilities = torch.softmax(input=classification_scores, dim=-1)[..., 1:]
    number_candidates = int(((probabilities >= 0.3) & valid_proposal_mask.unsqueeze(dim=-1)).sum())
    assert 0 < number_candidates < probabilities.numel() // 2
    assert recorded_box_counts == [number_candidates]


@pytest.mark.parametrize(argnames=("nms_criterion", "nms_distance_exponent"),
                         argvalues=[("iou", 1.0), ("distance_iou", 1.0), ("distance_iou", 0.6)])
def test_postprocess_detections_match_the_nms_on_every_pair(nms_criterion: str, nms_distance_exponent: float) -> None:
    """
    Verifies that the NMS on the candidates reaching the score threshold (eager) returns exactly the detections of the
    NMS on every pair (exports), on random batches with the evaluation threshold, the inference threshold and a
    threshold no pair reaches (an empty NMS input), with the greedy NMS and with DIoU-NMS at the exponents 1.0 and
    0.6, DIoU-NMS keeping more detections than the greedy NMS on these clustered boxes.

    Args:
        nms_criterion (str): The criterion of the final NMS.
        nms_distance_exponent (float): The exponent of the distance term of "distance_iou".
    """
    number_more_detections = 0
    for seed in range(4):
        for score_threshold in [0.05, 0.5, 1.5]:
            classification_scores, detection_boxes, valid_proposal_mask = build_random_head_outputs(seed=seed)
            settings = {"classification_scores": classification_scores, "detection_boxes": detection_boxes,
                        "valid_proposal_mask": valid_proposal_mask,
                        "image_windows": torch.tensor(data=[[0.0, 0.0, 128.0, 128.0]]).expand(3, -1),
                        "score_threshold": score_threshold,
                        "nms_iou_threshold": 0.5, "nms_distance_exponent": nms_distance_exponent,
                        "maximum_detections": 100, "input_image_size": 128}

            # Eager path on the candidates, then the reference on every pair
            detections = postprocess_detections(**settings, nms_criterion=nms_criterion, suppress_candidates_only=True)
            reference_detections = postprocess_detections(**settings, nms_criterion=nms_criterion,
                                                          suppress_candidates_only=False)

            # Same detections, and detections found on the batch except with the threshold no pair reaches
            for key, reference in reference_detections.items():
                assert torch.equal(detections[key], reference), \
                    f"{key} differs, seed {seed}, threshold {score_threshold}"
            assert (int(detections["number_detections"].sum()) == 0) == (score_threshold > 1.0)

            # Detections kept beyond those of the greedy NMS, by the overlapping boxes whose centres are apart
            greedy_detections = postprocess_detections(**settings, nms_criterion="iou", suppress_candidates_only=True)
            number_more_detections += int(detections["number_detections"].sum() -
                                          greedy_detections["number_detections"].sum())
    assert (number_more_detections > 0) == (nms_criterion == "distance_iou")


def test_postprocess_detections_clips_the_boxes_to_the_window_of_their_image() -> None:
    """
    Verifies that postprocess_detections clips every box to the window of its image, the part of the input holding
    the resized image, in both NMS paths: a box reaching into the padding of a letterboxed image stops at the window,
    a box inside it is unchanged, and the image filling the input keeps the clamp to the input.
    """
    # Two images of one proposal each, classes (background, A): a box reaching above and right of its window
    probabilities = torch.tensor(data=[[[0.1, 0.9]], [[0.1, 0.9]]])
    boxes = torch.tensor(data=[[[[20.0, 10.0, 140.0, 60.0]] * 2], [[[20.0, 10.0, 140.0, 60.0]] * 2]])
    image_windows = torch.tensor(data=[[0.0, 32.0, 128.0, 96.0], [0.0, 0.0, 128.0, 128.0]])
    for suppress_candidates_only in [True, False]:
        detections = postprocess_detections(classification_scores=torch.log(input=probabilities), detection_boxes=boxes,
                                            valid_proposal_mask=torch.ones(size=(2, 1), dtype=torch.bool),
                                            image_windows=image_windows, score_threshold=0.5, nms_iou_threshold=0.5,
                                            nms_criterion="iou", nms_distance_exponent=1.0, maximum_detections=2,
                                            input_image_size=128, suppress_candidates_only=suppress_candidates_only)
        assert detections["final_detection_boxes"][:, 0].tolist() == [[20.0, 32.0, 128.0, 60.0],
                                                                      [20.0, 10.0, 128.0, 60.0]]

