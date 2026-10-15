"""
Tests of the InstanceSegmenter (vision_training/models/instance_segmentation/instance_segmenter.py) and its inference
version: the mask loss with ground truth, the masks of the final detections in evaluation mode, the refused ground
truth without masks, the summaries, the inference instance segmenter reproducing the eager model, its TorchScript and
ONNX exports, and the comparison of the exported outputs, with the default mask geometry and with a pool size per
level and bilinear upsamplings.
"""

import numpy as np
import onnxruntime
import pytest
import torch

from pathlib import Path
from typing import Any, Dict, List, Optional

from tests.models.detection.test_detector_baseline import get_fixed_batch
from tests.testing_utilities import build_test_model
from vision_training.deployment.export_model import export_onnx, export_torchscript, hide_known_export_warnings
from vision_training.deployment.export_specifications import (build_inference_model, compare_instance_segmentations,
                                                              convert_outputs_to_numpy, create_random_inputs,
                                                              get_export_specification)
from vision_training.models.instance_segmentation.inference_instance_segmenter import InferenceInstanceSegmenter

# Largest batch size of the test exports, above the batch sizes of the traced and checked batches
EXPORT_MAXIMUM_BATCH_SIZE = 4

# Mask heads with a pool size per level and bilinear upsamplings: 7 doubled once to the 14 x 14 masks, 14 never
# doubled, and 5 doubled once then resized from 10 x 10, on level 7 which detects objects in the test model
MIXED_MASK_HEAD_SETTINGS = {"MaskHead": {"mask_resolution": 14, "upsampling": "bilinear"}, "level_mask_heads": {
    "5": {"roi_align_pool_size": 7, "convolutions": [2, 32]}, "6": {"roi_align_pool_size": 14, "convolutions": [2, 32]},
    "7": {"roi_align_pool_size": 5, "convolutions": [2, 32]}}}
MASK_HEAD_GEOMETRIES = [None, MIXED_MASK_HEAD_SETTINGS]


def build_box_masks(boxes: List[torch.Tensor], image_size: int) -> List[torch.Tensor]:
    """
    Builds the ground truth masks of the fixed batch, every object filling its box, so that the instance segmenter is
    trained on the boxes of the detector baseline.

    Args:
        boxes (List[torch.Tensor]): The [N_i, 4] ground truth boxes of every image.
        image_size (int): The side of the network input.

    Returns:
        List[torch.Tensor]: The [N_i, image_size, image_size] uint8 masks of every image.
    """
    masks = []
    for image_boxes in boxes:
        image_masks = torch.zeros(size=(image_boxes.shape[0], image_size, image_size), dtype=torch.uint8)
        for object_index, (x_1, y_1, x_2, y_2) in enumerate(image_boxes.int().tolist()):
            image_masks[object_index, y_1:y_2, x_1:x_2] = 1
        masks.append(image_masks)
    return masks


@pytest.mark.parametrize(argnames="mask_head_settings", argvalues=MASK_HEAD_GEOMETRIES)
def test_instance_segmenter_trains_and_predicts_masks(mask_head_settings: Optional[Dict[str, Any]]) -> None:
    """
    Verifies that with ground truth the instance segmenter adds a finite mask loss that reaches the mask heads, and
    that in evaluation mode it gives the [B, K, M, M] mask probabilities of the final detections, within [0, 1] and
    zero for the padding detections.

    Args:
        mask_head_settings (Optional[Dict[str, Any]]): The mask geometry replacing the one of the fixture, None to
            keep it.
    """
    model = build_test_model(dtype=torch.float32, instance_segmentation=True, mask_head_settings=mask_head_settings)
    images, boxes, labels = get_fixed_batch()
    images, boxes = images.float(), [image_boxes.float() for image_boxes in boxes]
    masks = build_box_masks(boxes=boxes, image_size=model.input_image_size)

    # Training: the mask loss joins the four losses of the detector and trains the mask heads
    model.train()
    torch.manual_seed(seed=2)
    outputs = model(input_tensor=images, ground_truth_bounding_boxes=boxes, ground_truth_labels=labels,
                    ground_truth_masks=masks)
    assert torch.isfinite(input=outputs["mask_loss"]) and float(outputs["mask_loss"].detach()) > 0.0
    outputs["mask_loss"].backward()
    assert all(parameter.grad is not None for parameter in model.mask_heads_dictionary.parameters())

    # Evaluation: the mask of every final detection, empty for the padding detections
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor=images)
    detection_masks = outputs["final_detection_masks"]
    assert detection_masks.shape == (2, model.maximum_detections, model.mask_size, model.mask_size)
    assert float(detection_masks.min()) >= 0.0 and float(detection_masks.max()) <= 1.0
    for image_index in range(2):
        number_detections = int(outputs["number_detections"][image_index])
        assert bool(detection_masks[image_index, number_detections:].eq(0.0).all())
        assert float(detection_masks[image_index, :number_detections].sum()) > 0.0

    # Validation: evaluation mode with ground truth gives both the mask loss and the detection masks
    with torch.no_grad():
        outputs = model(input_tensor=images, ground_truth_bounding_boxes=boxes, ground_truth_labels=labels,
                        ground_truth_masks=masks)
    assert torch.isfinite(input=outputs["mask_loss"]) and "final_detection_masks" in outputs

    # Ground truth boxes without their masks are refused
    with pytest.raises(ValueError, match="needs the ground truth masks"):
        model(input_tensor=images, ground_truth_bounding_boxes=boxes, ground_truth_labels=labels)


def test_instance_segmenter_summaries_list_the_mask_heads() -> None:
    """
    Verifies that the summary prints the mask heads panel and that the size summary lists the mask heads.
    """
    model = build_test_model(dtype=torch.float32, instance_segmentation=True)
    model.print_summary()
    assert list(model.get_size_components()) == ["Backbone", "Region Proposal Heads", "Detection Heads", "Mask Heads",
                                                 "Total Model"]


@pytest.mark.parametrize(argnames="mask_head_settings", argvalues=MASK_HEAD_GEOMETRIES)
def test_inference_instance_segmenter_matches_the_eager_model(mask_head_settings: Optional[Dict[str, Any]]) -> None:
    """
    Verifies that the export-friendly inference instance segmenter reproduces the detections and masks of the eager
    model in evaluation mode, every detection using the mask head of its own level.

    Args:
        mask_head_settings (Optional[Dict[str, Any]]): The mask geometry replacing the one of the fixture, None to
            keep it.
    """
    model = build_test_model(dtype=torch.float32, instance_segmentation=True, mask_head_settings=mask_head_settings)
    segmenter = InferenceInstanceSegmenter(model=model).eval()
    images = torch.rand(size=(2, 3, 512, 512)) * 255.0

    with torch.no_grad():
        eager_outputs = model(input_tensor=images)
        boxes, scores, labels, numbers, masks = segmenter(images=images)

    assert torch.equal(input=numbers, other=eager_outputs["number_detections"])
    assert torch.equal(input=labels, other=eager_outputs["final_detection_labels"])
    assert torch.allclose(input=boxes, other=eager_outputs["final_detection_boxes"])
    assert torch.allclose(input=scores, other=eager_outputs["final_detection_scores"])
    assert torch.allclose(input=masks, other=eager_outputs["final_detection_masks"], atol=1e-5)


@pytest.mark.parametrize(argnames="mask_head_settings", argvalues=MASK_HEAD_GEOMETRIES)
def test_exports_match_the_eager_inference_instance_segmenter(tmp_path: Path,
                                                              mask_head_settings: Optional[Dict[str, Any]]) -> None:
    """
    Verifies the TorchScript and ONNX exports of the inference instance segmenter, through the export specification
    of the task: five outputs, the masks [K, M, M] last, reproduced by both exports on a batch of another size than
    the traced one, the detections compared independently of tied scores and the masks paired in the same order.

    Args:
        tmp_path (Path): The pytest temporary directory.
        mask_head_settings (Optional[Dict[str, Any]]): The mask geometry replacing the one of the fixture, None to
            keep it.
    """
    hide_known_export_warnings()
    inference_model = build_inference_model(task="instance_segmentation", model=build_test_model(
        dtype=torch.float32, instance_segmentation=True, mask_head_settings=mask_head_settings))
    specification = get_export_specification(task="instance_segmentation", inference_model=inference_model)
    assert [output["name"] for output in specification["outputs"]] == ["detection_boxes", "detection_scores",
                                                                       "detection_labels", "number_detections",
                                                                       "detection_masks"]
    assert specification["outputs"][4]["dimensions"] == [inference_model.maximum_detections, 14, 14]

    example_inputs = create_random_inputs(specification=specification, shape_name="example_shape",
                                          maximum_batch_size=EXPORT_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                          generator=torch.Generator().manual_seed(1))
    check_inputs = create_random_inputs(specification=specification, shape_name="check_shape",
                                        maximum_batch_size=EXPORT_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                        generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        reference_outputs = convert_outputs_to_numpy(outputs=inference_model(**check_inputs))

    # TorchScript export, run on the check batch
    export_torchscript(inference_model=inference_model, example_inputs=example_inputs,
                       output_path=tmp_path / "model.pt")
    with torch.no_grad():
        torchscript_outputs = convert_outputs_to_numpy(outputs=torch.jit.load(str(tmp_path / "model.pt"))(
            **check_inputs))
    assert specification["compare_outputs"](reference=reference_outputs, candidate=torchscript_outputs,
                                            **specification["comparison_tolerances"]) == []

    # ONNX export with a dynamic batch, run by ONNX Runtime on the check batch
    export_onnx(inference_model=inference_model, example_inputs=example_inputs, output_path=tmp_path / "model.onnx",
                specification=specification, maximum_batch_size=EXPORT_MAXIMUM_BATCH_SIZE)
    session = onnxruntime.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {input_name: input_batch.numpy()
                                      for input_name, input_batch in check_inputs.items()})
    assert specification["compare_outputs"](reference=reference_outputs, candidate=onnx_outputs,
                                            **specification["comparison_tolerances"]) == []


def test_instance_segmentation_comparison_reports_mask_differences() -> None:
    """
    Verifies that the comparison of the exported outputs pairs the masks by the canonical order of the detections, so
    that detections of tied scores swapped by the export still match, and reports masks beyond the tolerance.
    """
    boxes = np.array(object=[[[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 20.0, 20.0]]], dtype=np.float32)
    scores = np.array(object=[[0.5, 0.5]], dtype=np.float32)
    labels = np.array(object=[[1, 2]], dtype=np.int64)
    numbers = np.array(object=[2], dtype=np.int64)
    masks = np.stack(arrays=[np.zeros(shape=(4, 4)), np.ones(shape=(4, 4))])[None].astype(np.float32)
    reference = [boxes, scores, labels, numbers, masks]

    # The same detections in the other order give no mismatch
    swapped = [boxes[:, ::-1], scores, labels[:, ::-1], numbers, masks[:, ::-1]]
    assert compare_instance_segmentations(reference=reference, candidate=swapped, box_tolerance=1e-2,
                                          score_tolerance=1e-4, mask_tolerance=1e-3) == []

    # Different masks are reported
    assert compare_instance_segmentations(reference=reference, candidate=[boxes, scores, labels, numbers, masks + 0.1],
                                          box_tolerance=1e-2, score_tolerance=1e-4,
                                          mask_tolerance=1e-3) == ["image 0: mask difference 1.00e-01"]
