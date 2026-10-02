import pytest
import torch

from typing import Any, Dict, List, Tuple

from tests.testing_utilities import build_test_model

# Ground truth of the fixed batch: [x1, y1, x2, y2] boxes in 512 pixel network input coordinates, and their classes
FIXED_GROUND_TRUTH_BOXES = [[[40.0, 60.0, 200.0, 220.0], [300.0, 280.0, 460.0, 400.0], [100.0, 350.0, 180.0, 500.0]],
                            [[20.0, 20.0, 150.0, 120.0], [250.0, 100.0, 480.0, 330.0]]]
FIXED_GROUND_TRUTH_LABELS = [[1, 3, 5], [2, 4]]

# Values computed by the detector before the major refactoring (major_refactoring.md, Phase 0). They are not ground
# truth: they record the current behaviour, so that moving, splitting or generalising the detector code is proven to
# leave every number unchanged. They may only be updated together with a deliberate change of the detector behaviour
EXPECTED_TRAINING_LOSSES = {"region_proposal_classification_loss": 0.6914948225021362,
                            "region_proposal_localisation_loss": 0.9389231171898904,
                            "detection_classification_loss": 1.8252800713607074,
                            "detection_regression_loss": 0.8443907026391789}
EXPECTED_INFERENCE_SUMMARY = {"number_detections": [100, 100], "score_sum": 34.051725473826025,
                              "box_sum": 178328.6940448164, "label_sum": 478}


def get_fixed_batch() -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    Builds the fixed batch fed to the baseline detector: two seeded random images and their hand-written ground
    truth, in float64 like the baseline detector.

    Returns:
        Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]: The [2, 3, 512, 512] images in [0, 255], and the
            per-image ground truth boxes and labels.
    """
    torch.manual_seed(seed=1)
    images = torch.rand(size=(2, 3, 512, 512), dtype=torch.float64) * 255.0
    boxes = [torch.tensor(data=image_boxes, dtype=torch.float64) for image_boxes in FIXED_GROUND_TRUTH_BOXES]
    labels = [torch.tensor(data=image_labels, dtype=torch.int64) for image_labels in FIXED_GROUND_TRUTH_LABELS]
    return images, boxes, labels


def compute_training_losses() -> Dict[str, float]:
    """
    Runs the training path of the baseline detector on the fixed batch: training mode, with ground truth, so that the
    anchor and proposal target assignment, the sampling and the four losses are all exercised. The random sampling
    and proposal padding draw from a fixed seed.

    Returns:
        Dict[str, float]: The value of every loss returned by the detector.
    """
    model = build_test_model(dtype=torch.float64).train()
    images, boxes, labels = get_fixed_batch()

    torch.manual_seed(seed=2)
    with torch.no_grad():
        outputs = model(input_tensor=images, ground_truth_bounding_boxes=boxes, ground_truth_labels=labels)
    return {key: float(value) for key, value in outputs.items() if key.endswith("_loss")}


def compute_inference_summary() -> Dict[str, Any]:
    """
    Runs the inference path of the baseline detector on the fixed images: evaluation mode, without ground truth, so
    that the inference proposal filtering and the final post-processing are exercised, and summarises its detections.

    Returns:
        Dict[str, Any]: The number of detections of every image, and the sums of the detection scores, boxes and
            labels.
    """
    model = build_test_model(dtype=torch.float64)
    images, _, _ = get_fixed_batch()

    torch.manual_seed(seed=3)
    with torch.no_grad():
        outputs = model(input_tensor=images)
    return {"number_detections": outputs["number_detections"].tolist(),
            "score_sum": float(outputs["final_detection_scores"].sum()),
            "box_sum": float(outputs["final_detection_boxes"].sum()),
            "label_sum": int(outputs["final_detection_labels"].sum())}


def test_training_losses_match_the_baseline() -> None:
    """
    Verifies that the four training losses of the detector on the fixed batch are the ones recorded before the
    refactoring.
    """
    assert compute_training_losses() == pytest.approx(expected=EXPECTED_TRAINING_LOSSES, rel=1e-9)


def test_inference_detections_match_the_baseline() -> None:
    """
    Verifies that the detections of the detector on the fixed images are the ones recorded before the refactoring:
    the same number of detections per image, the same labels, and the same scores and boxes.
    """
    summary = compute_inference_summary()
    assert summary["number_detections"] == EXPECTED_INFERENCE_SUMMARY["number_detections"]
    assert summary["label_sum"] == EXPECTED_INFERENCE_SUMMARY["label_sum"]
    assert summary["score_sum"] == pytest.approx(expected=EXPECTED_INFERENCE_SUMMARY["score_sum"], rel=1e-9)
    assert summary["box_sum"] == pytest.approx(expected=EXPECTED_INFERENCE_SUMMARY["box_sum"], rel=1e-9)
