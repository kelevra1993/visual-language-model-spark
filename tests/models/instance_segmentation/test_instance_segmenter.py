"""
Tests of the InstanceSegmenter (vision_training/models/instance_segmentation/instance_segmenter.py) and its inference
version: the mask loss with ground truth, the masks of the final detections in evaluation mode, the refused ground
truth without masks and the inference instance segmenter reproducing the eager model, with a pool size per level and
bilinear upsamplings, without and with the finer inputs of the mask heads (image branch, backbone skips, fusion
convolutions), whose layers and channels are checked against the configuration.
"""

import pytest
import torch

from typing import Any, Dict, List, Optional

from tests.models.detection.test_detector_baseline import get_fixed_batch
from tests.testing_utilities import build_test_model
from vision_training.losses.instance_segmentation_losses import compute_mask_loss
from vision_training.models.instance_segmentation.inference_instance_segmenter import InferenceInstanceSegmenter
from vision_training.models.instance_segmentation.mask_operations import build_mask_targets

# Mask heads with a pool size per level and bilinear upsamplings: 7 doubled once to the 14 x 14 masks, 14 never
# doubled, and 5 doubled once then resized from 10 x 10, on level 7 which detects objects in the test model
MIXED_MASK_HEAD_SETTINGS = {"MaskHead": {"mask_resolution": 14, "upsampling": "bilinear"}, "level_mask_heads": {
    "5": {"roi_align_pool_size": 7, "convolutions": [2, 32]}, "6": {"roi_align_pool_size": 14, "convolutions": [2, 32]},
    "7": {"roi_align_pool_size": 5, "convolutions": [2, 32]}}}

# Finer inputs of the mask heads on the mixed geometry, whose pool of 5 doubled to 10 x 10 is resized to the 14 x 14
# fine features: the image branch with fusion convolutions (design A of stage 3b of the benchmark), the backbone
# skips of blocks 4 and 3 at the pool size and block 2 at the mask resolution with fusion convolutions (design B), the
# fusion convolutions alone, and every input at once without fusion convolutions, the predictor reading them directly
IMAGE_BRANCH_MASK_HEAD_SETTINGS = {"MaskHead": {**MIXED_MASK_HEAD_SETTINGS["MaskHead"],
                                                "image_branch_convolutions": [2, 8], "fusion_convolutions": [1, 16]},
                                   "level_mask_heads": MIXED_MASK_HEAD_SETTINGS["level_mask_heads"]}
BACKBONE_SKIP_MASK_HEAD_SETTINGS = {"MaskHead": {**MIXED_MASK_HEAD_SETTINGS["MaskHead"],
                                                 "skip_blocks_at_pool_size": [4, 3],
                                                 "skip_blocks_at_mask_resolution": [2], "fusion_convolutions": [1, 16]},
                                    "level_mask_heads": MIXED_MASK_HEAD_SETTINGS["level_mask_heads"]}
FUSION_ONLY_MASK_HEAD_SETTINGS = {"MaskHead": {**MIXED_MASK_HEAD_SETTINGS["MaskHead"], "fusion_convolutions": [2, 16]},
                                  "level_mask_heads": MIXED_MASK_HEAD_SETTINGS["level_mask_heads"]}
EVERY_FINE_INPUT_MASK_HEAD_SETTINGS = {"MaskHead": {**MIXED_MASK_HEAD_SETTINGS["MaskHead"],
                                                    "skip_blocks_at_pool_size": [4],
                                                    "image_branch_convolutions": [1, 8],
                                                    "skip_blocks_at_mask_resolution": [0, 2]},
                                       "level_mask_heads": MIXED_MASK_HEAD_SETTINGS["level_mask_heads"]}

# Mask geometries and neck types of the tests, the mask heads reading the levels after the neck, and those with finer
# inputs, the backbone skips being read before the neck
MASK_HEAD_GEOMETRIES_AND_NECKS = [(MIXED_MASK_HEAD_SETTINGS, "none"), (MIXED_MASK_HEAD_SETTINGS, "path_aggregation")]
FINE_INPUT_GEOMETRIES_AND_NECKS = [(IMAGE_BRANCH_MASK_HEAD_SETTINGS, "none"),
                                   (BACKBONE_SKIP_MASK_HEAD_SETTINGS, "none"),
                                   (FUSION_ONLY_MASK_HEAD_SETTINGS, "none"),
                                   (EVERY_FINE_INPUT_MASK_HEAD_SETTINGS, "path_aggregation")]


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


@pytest.mark.parametrize(argnames=["mask_head_settings", "neck_type"],
                         argvalues=MASK_HEAD_GEOMETRIES_AND_NECKS + FINE_INPUT_GEOMETRIES_AND_NECKS)
def test_instance_segmenter_trains_and_predicts_masks(mask_head_settings: Optional[Dict[str, Any]],
                                                      neck_type: str) -> None:
    """
    Verifies that with ground truth the instance segmenter adds a finite mask loss that reaches the mask heads and the
    image branch of their fine inputs, and that in evaluation mode it gives the [B, K, M, M] mask probabilities of the
    final detections, within [0, 1] and zero for the padding detections.

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
    assert all(parameter.grad is not None for parameter in model.mask_fine_inputs.parameters())

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


def order_tied_detections(boxes: torch.Tensor, scores: torch.Tensor, masks: torch.Tensor,
                          number_detections: torch.Tensor) -> List[torch.Tensor]:
    """
    Orders the valid detections of every image by decreasing score, then by their box corners, so that detections of
    equal scores, which the eager model and the inference segmenter may list in either order, are compared with each
    other. The untrained test model gives such ties for some images.

    Args:
        boxes (torch.Tensor): The [B, K, 4] detection boxes.
        scores (torch.Tensor): The [B, K] detection scores.
        masks (torch.Tensor): The [B, K, M, M] detection masks.
        number_detections (torch.Tensor): The [B] number of valid detections of every image.

    Returns:
        List[torch.Tensor]: The ordered boxes, scores and masks of the valid detections, concatenated over the images.
    """
    ordered_boxes, ordered_scores, ordered_masks = [], [], []
    for image_index in range(boxes.shape[0]):
        number_valid = int(number_detections[image_index])
        image_boxes, image_scores = boxes[image_index, :number_valid], scores[image_index, :number_valid]
        order = sorted(range(number_valid), key=lambda detection_index: (-float(image_scores[detection_index]),
                                                                         *image_boxes[detection_index].tolist()))
        ordered_boxes.append(image_boxes[order])
        ordered_scores.append(image_scores[order])
        ordered_masks.append(masks[image_index, :number_valid][order])
    return [torch.cat(tensors=ordered_boxes), torch.cat(tensors=ordered_scores), torch.cat(tensors=ordered_masks)]


@pytest.mark.parametrize(argnames=["mask_head_settings", "neck_type"], argvalues=FINE_INPUT_GEOMETRIES_AND_NECKS)
def test_inference_instance_segmenter_with_fine_inputs_matches_the_eager_model(mask_head_settings: Dict[str, Any],
                                                                               neck_type: str) -> None:
    """
    Verifies that the export-friendly inference instance segmenter, which computes the fine features of every
    detection before the masks of every level, reproduces the detections and masks of the eager model, which computes
    them on the valid detections only, the detections of equal scores being compared in the same order, and that the
    padding detections get empty masks.

    Args:
        mask_head_settings (Dict[str, Any]): The mask geometry and finer inputs replacing those of the fixture.
        neck_type (str): The 'Neck.type' of the instance segmenter.
    """
    model = build_test_model(dtype=torch.float32, instance_segmentation=True, mask_head_settings=mask_head_settings,
                             neck_type=neck_type)
    images = torch.rand(size=(2, 3, 512, 512), generator=torch.Generator().manual_seed(1)) * 255.0

    # A score threshold at the median score of the first image, so that padding detections sit among the detections
    # whose fine features the eager model leaves out
    with torch.no_grad():
        model.detection_score_threshold = float(model(input_tensor=images)["final_detection_scores"][0].median())
    segmenter = InferenceInstanceSegmenter(model=model).eval()

    with torch.no_grad():
        eager_outputs = model(input_tensor=images)
        boxes, scores, labels, numbers, masks = segmenter(images=images)

    assert 0 < int(numbers.min()) < model.maximum_detections
    assert torch.equal(input=numbers, other=eager_outputs["number_detections"])
    assert torch.equal(input=labels, other=eager_outputs["final_detection_labels"])
    inference_detections = order_tied_detections(boxes=boxes, scores=scores, masks=masks, number_detections=numbers)
    eager_detections = order_tied_detections(boxes=eager_outputs["final_detection_boxes"],
                                             scores=eager_outputs["final_detection_scores"],
                                             masks=eager_outputs["final_detection_masks"], number_detections=numbers)
    for inference_tensor, eager_tensor in zip(inference_detections, eager_detections):
        assert torch.allclose(input=inference_tensor, other=eager_tensor, atol=1e-5)
    for image_index in range(2):
        assert bool(masks[image_index, int(numbers[image_index]):].eq(0.0).all())


def test_mask_loss_gives_every_level_the_fine_features_of_its_proposals() -> None:
    """
    Verifies that the mask loss, which computes the fine features of the positive proposals of all the levels at once,
    gives the mask head of every level the rows of its own proposals: in evaluation mode, where the batch statistics no
    longer depend on the boxes of the batch, it equals the loss computed level by level with the fine features of the
    proposals of that level alone, the positive proposals spreading over several levels.
    """
    model = build_test_model(dtype=torch.float32, instance_segmentation=True,
                             mask_head_settings=IMAGE_BRANCH_MASK_HEAD_SETTINGS, neck_type="none")
    images, boxes, labels = get_fixed_batch()
    images, boxes = images.float(), [image_boxes.float() for image_boxes in boxes]
    masks = build_box_masks(boxes=boxes, image_size=model.input_image_size)

    with torch.no_grad():
        detection_stages = model.forward_detection_stages(input_tensor=images, ground_truth_bounding_boxes=boxes,
                                                          ground_truth_labels=labels)
        feature_maps = detection_stages["backbone_output_tensor_dictionary"]
        skip_feature_maps = detection_stages["backbone_skip_dictionary"]
        aggregated_detections = detection_stages["aggregated_detections_dictionary"]
        mask_loss = model.compute_proposal_mask_loss(feature_maps=feature_maps, skip_feature_maps=skip_feature_maps,
                                                     aggregated_detections=aggregated_detections,
                                                     ground_truth_masks=masks)

        # Level by level, the fine features computed on the positive proposals of the level alone
        positive_mask = aggregated_detections["sliced_labels"] > 0
        level_logits, level_labels, level_proposals, level_matched_indices = [], [], [], []
        for index_string, feature_map in feature_maps.items():
            level_indices = torch.nonzero(input=positive_mask & (aggregated_detections["sliced_origins"] ==
                                                                 int(index_string))).flatten()
            proposals = aggregated_detections["sliced_proposals"][level_indices]
            fine_features = model.mask_fine_inputs(proposal_boxes=proposals, skip_feature_maps=skip_feature_maps)
            level_logits.append(model.mask_heads_dictionary[index_string](
                proposal_boxes=proposals, input_tensor=feature_map, skip_feature_maps=skip_feature_maps,
                fine_features=fine_features))
            level_labels.append(aggregated_detections["sliced_labels"][level_indices])
            level_proposals.append(proposals)
            level_matched_indices.append(aggregated_detections["sliced_matched_ground_truth_indices"][level_indices])
        mask_targets = build_mask_targets(ground_truth_masks=masks, proposal_boxes=torch.cat(tensors=level_proposals),
                                          matched_ground_truth_indices=torch.cat(tensors=level_matched_indices),
                                          mask_size=model.mask_size)
        expected_loss = compute_mask_loss(mask_logits=torch.cat(tensors=level_logits),
                                          labels=torch.cat(tensors=level_labels), mask_targets=mask_targets)

    # Positive proposals on several levels, in different numbers, so that a shifted slice would change the loss
    level_counts = [proposals.shape[0] for proposals in level_proposals]
    assert sum(count > 0 for count in level_counts) >= 2 and len(set(level_counts)) > 1
    assert torch.allclose(input=mask_loss, other=expected_loss, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(argnames=["mask_head_settings", "neck_type", "expected_channels"], argvalues=[
    (MIXED_MASK_HEAD_SETTINGS, "none", {"fine": 0, "block": 64, "fusion": None, "predictor": 32}),
    (IMAGE_BRANCH_MASK_HEAD_SETTINGS, "none", {"fine": 8, "block": 64, "fusion": 40, "predictor": 16}),
    (BACKBONE_SKIP_MASK_HEAD_SETTINGS, "none", {"fine": 32, "block": 64 + 64 + 32, "fusion": 64, "predictor": 16}),
    (FUSION_ONLY_MASK_HEAD_SETTINGS, "none", {"fine": 0, "block": 64, "fusion": 32, "predictor": 16}),
    (EVERY_FINE_INPUT_MASK_HEAD_SETTINGS, "path_aggregation",
     {"fine": 8 + 3 + 32, "block": 32 + 64, "fusion": None, "predictor": 32 + 8 + 3 + 32})])
def test_mask_head_layers_follow_the_fine_inputs(mask_head_settings: Dict[str, Any], neck_type: str,
                                                 expected_channels: Dict[str, Optional[int]]) -> None:
    """
    Verifies the layers the finer inputs give the mask heads of the test model (levels of 64 channels, 32 after the
    neck, blocks 2, 3 and 4 of 32, 32 and 64 channels before their pooling, raw 3-channel images): the channels of the
    fine features shared by the levels (image branch, then the skips at the mask resolution in their order), the input
    of the convolution block of every level (its map, then the skips at the pool size in their order), the input of the
    fusion convolutions and of the predictor, the backbone outputs the model requests, the resizing of the doubled
    features to the fine features (pool 5 doubled to 10 on level 7) or of the logits without fine features, and the
    refusal of fine features a head was not built for.

    Args:
        mask_head_settings (Dict[str, Any]): The mask geometry and finer inputs replacing those of the fixture.
        neck_type (str): The 'Neck.type' of the instance segmenter.
        expected_channels (Dict[str, Optional[int]]): The expected channels of the fine features, of the input of the
            convolution blocks, of the input of the fusion convolutions (None without them) and of the predictors.
    """
    model = build_test_model(dtype=torch.float32, instance_segmentation=True, mask_head_settings=mask_head_settings,
                             neck_type=neck_type)
    mask_head_configuration = model.mask_head_configuration
    image_branch_layers = mask_head_configuration["image_branch_convolutions"][0]
    skip_blocks_at_pool_size = mask_head_configuration["skip_blocks_at_pool_size"]
    skip_blocks_at_mask_resolution = mask_head_configuration["skip_blocks_at_mask_resolution"]

    # Backbone outputs requested by the mask heads, the input image only for the image branch
    expected_skip_blocks = set(skip_blocks_at_pool_size) | set(skip_blocks_at_mask_resolution) | \
        ({0} if image_branch_layers > 0 else set())
    assert model.backbone.skip_block_indices == sorted(expected_skip_blocks)

    # Fine features shared by the levels, the image branch being plain convolutions on the image channels
    fine_inputs = model.mask_fine_inputs
    assert fine_inputs.number_channels == expected_channels["fine"]
    assert fine_inputs.skip_keys == [f"skip_{block}" for block in skip_blocks_at_mask_resolution]
    if image_branch_layers > 0:
        assert fine_inputs.image_branch.input_channels == 3 and not fine_inputs.image_branch.residual
        assert fine_inputs.image_branch.number_layers == image_branch_layers
    else:
        assert fine_inputs.image_branch is None

    # Layers of the mask head of every level
    for level_key, mask_head in model.mask_heads_dictionary.items():
        assert mask_head.skip_keys == [f"skip_{block}" for block in skip_blocks_at_pool_size]
        assert mask_head.convolutional_block.input_channels == expected_channels["block"]
        if expected_channels["fusion"] is None:
            assert mask_head.fusion_block is None
        else:
            assert mask_head.fusion_block.input_channels == expected_channels["fusion"]
        assert mask_head.mask_predictor.in_channels == expected_channels["predictor"]

        # Pool 5 of level 7 doubled to 10 x 10, resized to the 14 x 14 masks before the fine features, or after the
        # predictor without them
        falls_short = level_key == "7"
        assert mask_head.resizes_features == (falls_short and expected_channels["fine"] > 0)
        assert mask_head.resizes_logits == (falls_short and expected_channels["fine"] == 0)

    # A mask head refuses fine features it was not built for, and needs those it was built for
    level_head = model.mask_heads_dictionary["5"]
    skip_feature_maps = {"skip_0": torch.rand(size=(1, 3, 512, 512)), "skip_2": torch.rand(size=(1, 32, 256, 256)),
                         "skip_3": torch.rand(size=(1, 32, 128, 128)), "skip_4": torch.rand(size=(1, 64, 64, 64))}
    level_channels = level_head.convolutional_block.input_channels - sum(
        skip_feature_maps[skip_key].shape[1] for skip_key in level_head.skip_keys)
    wrong_fine_features = None if expected_channels["fine"] > 0 else torch.rand(size=(1, 4, 14, 14))
    with pytest.raises(expected_exception=ValueError, match="channels of fine features"):
        level_head(proposal_boxes=torch.tensor(data=[[0.0, 10.0, 10.0, 90.0, 120.0]]),
                   input_tensor=torch.rand(size=(1, level_channels, 16, 16)), skip_feature_maps=skip_feature_maps,
                   fine_features=wrong_fine_features)
