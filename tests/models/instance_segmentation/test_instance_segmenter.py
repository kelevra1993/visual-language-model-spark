"""
Tests of the InstanceSegmenter (vision_training/models/instance_segmentation/instance_segmenter.py) and its inference
version: the mask loss with ground truth, the masks of the final detections in evaluation mode, the refused ground
truth without masks and the inference instance segmenter reproducing the eager model, with a pool size per level and
bilinear upsamplings.
"""

import pytest
import torch

from typing import Any, Dict, List, Optional

from tests.models.detection.test_detector_baseline import get_fixed_batch
from tests.testing_utilities import build_test_model
from vision_training.models.instance_segmentation.inference_instance_segmenter import InferenceInstanceSegmenter

# Mask heads with a pool size per level and bilinear upsamplings: 7 doubled once to the 14 x 14 masks, 14 never
# doubled, and 5 doubled once then resized from 10 x 10, on level 7 which detects objects in the test model
MIXED_MASK_HEAD_SETTINGS = {"MaskHead": {"mask_resolution": 14, "upsampling": "bilinear"}, "level_mask_heads": {
    "5": {"roi_align_pool_size": 7, "convolutions": [2, 32]}, "6": {"roi_align_pool_size": 14, "convolutions": [2, 32]},
    "7": {"roi_align_pool_size": 5, "convolutions": [2, 32]}}}

# Mask geometries and neck types of the tests, the mask heads reading the levels after the neck
MASK_HEAD_GEOMETRIES_AND_NECKS = [(MIXED_MASK_HEAD_SETTINGS, "none"), (MIXED_MASK_HEAD_SETTINGS, "path_aggregation")]


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


@pytest.mark.parametrize(argnames=["mask_head_settings", "neck_type"], argvalues=MASK_HEAD_GEOMETRIES_AND_NECKS)
def test_instance_segmenter_trains_and_predicts_masks(mask_head_settings: Optional[Dict[str, Any]],
                                                      neck_type: str) -> None:
    """
    Verifies that with ground truth the instance segmenter adds a finite mask loss that reaches the mask heads, and
    that in evaluation mode it gives the [B, K, M, M] mask probabilities of the final detections, within [0, 1] and
    zero for the padding detections.

    Args:
        mask_head_settings (Optional[Dict[str, Any]]): The mask geometry replacing the one of the fixture, None to
            keep it.
        neck_type (str): The 'Neck.type' of the instance segmenter.
    """
    model = build_test_model(dtype=torch.float32, instance_segmentation=True, mask_head_settings=mask_head_settings,
                             neck_type=neck_type)
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


@pytest.mark.parametrize(argnames=["mask_head_settings", "neck_type", "nms_criterion"],
                         argvalues=[(mask_head_settings, neck_type, "iou")
                                    for mask_head_settings, neck_type in MASK_HEAD_GEOMETRIES_AND_NECKS] +
                                   [(MIXED_MASK_HEAD_SETTINGS, "none", "distance_iou")])
def test_inference_instance_segmenter_matches_the_eager_model(mask_head_settings: Optional[Dict[str, Any]],
                                                              neck_type: str, nms_criterion: str) -> None:
    """
    Verifies that the export-friendly inference instance segmenter reproduces the detections and masks of the eager
    model in evaluation mode, every detection using the mask head of its own level, with the greedy NMS and with
    DIoU-NMS, which the inference segmenter inherits from the inference detector.

    Args:
        mask_head_settings (Optional[Dict[str, Any]]): The mask geometry replacing the one of the fixture, None to
            keep it.
        neck_type (str): The 'Neck.type' of the instance segmenter.
        nms_criterion (str): The 'Detector.nms_criterion' of the final NMS, "distance_iou" with the exponent 0.6.
    """
    model = build_test_model(dtype=torch.float32, instance_segmentation=True, mask_head_settings=mask_head_settings,
                             neck_type=neck_type, nms_criterion=nms_criterion, nms_distance_exponent=0.6)
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
