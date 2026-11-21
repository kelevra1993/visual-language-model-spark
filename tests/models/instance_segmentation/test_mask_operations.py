"""
Tests of vision_training/models/instance_segmentation/mask_operations.py and the mask head: the mask geometry of the
mask head (its doublings and the shapes of its outputs), the pooling of the mask head shared by the levels, the mask
targets cropped from the ground truth masks, the mask IoU head of the scored masks with its targets and the mask
scores, the pasting of the predicted masks into their boxes and the intersection over union of binary masks.
"""

import pytest
import torch

from torch import nn
from torchvision.ops import roi_align

from tests.fixtures.configuration_fixtures import get_detection_model_configuration
from vision_training.models.instance_segmentation.mask_head import MaskHead, MaskIoUHead, count_mask_upsamplings
from vision_training.models.instance_segmentation.mask_operations import (build_mask_iou_targets, build_mask_targets,
                                                                          compute_mask_intersection_over_union,
                                                                          compute_mask_scores, paste_instance_masks)


def build_mask_head(roi_align_pool_size: int, mask_resolution: int, upsampling: str) -> MaskHead:
    """
    Builds the mask head of a 16 x 16 feature map of 8 channels for a 128 pixel input and 6 classes, with the modules
    of the mask head fixture and the given mask geometry, without skips, fine features or fusion convolutions, in
    evaluation mode.

    Args:
        roi_align_pool_size (int): The RoIAlign pool size of the level.
        mask_resolution (int): The side of the masks.
        upsampling (str): The doubling of the pooled features, "transposed" or "bilinear".

    Returns:
        MaskHead: The mask head.
    """
    configuration, device, dtype = get_detection_model_configuration(mask_head=True)
    return MaskHead(head_configuration={"roi_align_pool_size": roi_align_pool_size, "convolutions": [2, 16]},
                    mask_resolution=mask_resolution, upsampling=upsampling, number_classes=6, input_channels=8,
                    feature_map_normalization="batch", batch_normalization_momentum=0.1, input_image_size=128,
                    feature_map_sizes={"5": 16}, modules=configuration["MaskHead"]["modules"], skip_information={},
                    fine_input_channels=0, fusion_convolutions=[0, 0],
                    mask_scoring=configuration["MaskHead"]["mask_scoring"], dtype=dtype, device=device).eval()


@pytest.mark.parametrize(argnames=["roi_align_pool_size", "mask_resolution", "upsampling", "number_upsamplings"],
                         argvalues=[(7, 28, "bilinear", 2)])
def test_mask_head_doubles_its_pooled_features_up_to_the_mask_resolution(roi_align_pool_size: int,
                                                                          mask_resolution: int, upsampling: str,
                                                                          number_upsamplings: int) -> None:
    """
    Verifies that the mask head of a level doubles its pooled features as many times as fits in the mask resolution,
    with layers of the chosen upsampling, and gives [R, C, M, M] logits for R boxes, including no box at all.

    Args:
        roi_align_pool_size (int): The RoIAlign pool size of the level.
        mask_resolution (int): The side of the masks.
        upsampling (str): The doubling of the pooled features.
        number_upsamplings (int): The expected number of doublings.
    """
    assert count_mask_upsamplings(roi_align_pool_size=roi_align_pool_size,
                                  mask_resolution=mask_resolution) == number_upsamplings
    mask_head = build_mask_head(roi_align_pool_size=roi_align_pool_size, mask_resolution=mask_resolution,
                                upsampling=upsampling)
    expected_layer_type = nn.ConvTranspose2d if upsampling == "transposed" else nn.Sequential
    assert len(mask_head.upsampling_layers) == number_upsamplings
    assert all(isinstance(layer, expected_layer_type) for layer in mask_head.upsampling_layers)

    feature_map = torch.rand(size=(2, 8, 16, 16))
    boxes = torch.tensor(data=[[0.0, 10.0, 10.0, 60.0, 80.0], [1.0, 0.0, 0.0, 128.0, 128.0], [1.0, 5.0, 5.0, 9.0, 9.0]])
    assert mask_head(proposal_boxes=boxes, input_tensor=feature_map, skip_feature_maps={},
                     fine_features=None).shape == (3, 6, mask_resolution, mask_resolution)
    assert mask_head(proposal_boxes=boxes[:0], input_tensor=feature_map, skip_feature_maps={},
                     fine_features=None).shape == (0, 6, mask_resolution, mask_resolution)


def test_unknown_mask_head_upsampling_is_refused() -> None:
    """
    Verifies that a misspelled upsampling of the mask head reaching the model without the configuration checks stops
    with the list of the valid upsamplings, instead of being built as the bilinear upsampling.
    """
    with pytest.raises(expected_exception=ValueError, match="'MaskHead.upsampling' must be one of .* got 'nearest'"):
        build_mask_head(roi_align_pool_size=7, mask_resolution=28, upsampling="nearest")


def test_shared_mask_head_pools_every_level_at_its_own_scale() -> None:
    """
    Verifies that a mask head read by two levels (the head shared by the levels of 'MaskHead.shared') pools the boxes
    of each level from its map at the spatial scale of that level, predicts the pooled boxes of both levels at once,
    and refuses the single-level forward, whose map would be ambiguous, and a level it does not read.
    """
    configuration, device, dtype = get_detection_model_configuration(mask_head=True)
    mask_head = MaskHead(head_configuration={"roi_align_pool_size": 7, "convolutions": [2, 16]}, mask_resolution=14,
                         upsampling="transposed", number_classes=6, input_channels=8, feature_map_normalization="batch",
                         batch_normalization_momentum=0.1, input_image_size=128, feature_map_sizes={"5": 32, "6": 16},
                         modules=configuration["MaskHead"]["modules"], skip_information={}, fine_input_channels=0,
                         fusion_convolutions=[0, 0], mask_scoring=configuration["MaskHead"]["mask_scoring"],
                         dtype=dtype, device=device).eval()
    feature_maps = {"5": torch.rand(size=(2, 8, 32, 32)), "6": torch.rand(size=(2, 8, 16, 16))}
    boxes = torch.tensor(data=[[0.0, 10.0, 10.0, 60.0, 80.0], [1.0, 0.0, 0.0, 128.0, 128.0]])

    # Every level pooled at its own scale: 32 / 128 for level 5, 16 / 128 for level 6
    pooled_features = {}
    for level_name, feature_map in feature_maps.items():
        pooled_features[level_name] = mask_head.pool_box_features(proposal_boxes=boxes, input_tensor=feature_map,
                                                                   skip_feature_maps={}, level_name=level_name)
        expected_features = roi_align(input=feature_map, boxes=boxes, output_size=(7, 7),
                                      spatial_scale=feature_map.shape[-1] / 128, sampling_ratio=0, aligned=True)
        assert torch.equal(input=pooled_features[level_name], other=expected_features)

    # The boxes of both levels predicted at once, row by row as when predicted level by level
    with torch.no_grad():
        joint_logits = mask_head.predict_from_pooled_features(
            pooled_features=torch.cat(tensors=[pooled_features["5"], pooled_features["6"]]), fine_features=None)
        level_logits = [mask_head.predict_from_pooled_features(pooled_features=pooled_features[level_name],
                                                               fine_features=None) for level_name in ["5", "6"]]
    assert joint_logits.shape == (4, 6, 14, 14)
    assert torch.allclose(input=joint_logits, other=torch.cat(tensors=level_logits), atol=1e-6)

    # The single-level forward and an unknown level are refused
    with pytest.raises(expected_exception=ValueError, match="reads the levels \\['5', '6'\\]: pool the boxes"):
        mask_head(proposal_boxes=boxes, input_tensor=feature_maps["5"], skip_feature_maps={}, fine_features=None)
    with pytest.raises(expected_exception=ValueError, match="not level '7'"):
        mask_head.pool_box_features(proposal_boxes=boxes, input_tensor=feature_maps["6"], skip_feature_maps={},
                                    level_name="7")


def test_mask_targets_crop_the_matched_ground_truth_masks() -> None:
    """
    Verifies that a proposal equal to the box of a square ground truth mask gets a full target, that a proposal
    covering the left half of a mask gets the left half of its target when the mask covers only that half, and that
    the matched index picks the mask of the right object in the right image.
    """
    first_image_masks = torch.zeros(size=(2, 64, 64), dtype=torch.uint8)
    first_image_masks[0, 8:24, 8:24] = 1
    first_image_masks[1, 30:50, 20:40] = 1
    second_image_masks = torch.zeros(size=(1, 64, 64), dtype=torch.uint8)
    second_image_masks[0, 0:32, 0:16] = 1

    proposals = torch.tensor(data=[[0.0, 8.0, 8.0, 24.0, 24.0], [1.0, 0.0, 0.0, 32.0, 32.0],
                                   [0.0, 20.0, 30.0, 40.0, 50.0]])
    targets = build_mask_targets(ground_truth_masks=[first_image_masks, second_image_masks], proposal_boxes=proposals,
                                 matched_ground_truth_indices=torch.tensor(data=[0, 0, 1]), mask_size=8)

    assert targets.shape == (3, 8, 8) and set(targets.unique().tolist()) <= {0.0, 1.0}
    assert bool(targets[0].eq(1.0).all()) and bool(targets[2].eq(1.0).all())
    assert bool(targets[1][:, :4].eq(1.0).all()) and bool(targets[1][:, 4:].eq(0.0).all())


def test_mask_iou_targets_count_the_whole_object() -> None:
    """
    Verifies the targets of the mask IoU head on known masks: a predicted mask filling the box of its whole object has
    an IoU of 1, the same mask on a proposal covering half of the object 0.5 (the half outside the proposal counting in
    the union, as on the pasted masks), half of the mask on the whole box 0.5, an empty mask 0, and the matched index
    reading the area of the right object of the right image.
    """
    # First image: an object of 8 x 4 pixels (32 pixels); second image: a full object and a square of 4 x 4 pixels
    first_image_masks = torch.zeros(size=(1, 16, 16), dtype=torch.uint8)
    first_image_masks[0, 0:4, 0:8] = 1
    second_image_masks = torch.zeros(size=(2, 16, 16), dtype=torch.uint8)
    second_image_masks[0] = 1
    second_image_masks[1, 8:12, 8:12] = 1
    ground_truth_masks = [first_image_masks, second_image_masks]

    # Whole box, left half of the object, whole box twice, then the square of the second image (matched index 1)
    proposals = torch.tensor(data=[[0.0, 0.0, 0.0, 8.0, 4.0], [0.0, 0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 0.0, 8.0, 4.0],
                                   [0.0, 0.0, 0.0, 8.0, 4.0], [1.0, 8.0, 8.0, 12.0, 12.0]])
    matched_indices = torch.tensor(data=[0, 0, 0, 0, 1])
    mask_targets = build_mask_targets(ground_truth_masks=ground_truth_masks, proposal_boxes=proposals,
                                      matched_ground_truth_indices=matched_indices, mask_size=4)
    assert bool(mask_targets.eq(1.0).all())

    # Full predicted masks, but the left half of the third and an empty fourth
    predicted_masks = torch.ones(size=(5, 4, 4), dtype=torch.bool)
    predicted_masks[2, :, 2:] = False
    predicted_masks[3] = False
    targets = build_mask_iou_targets(predicted_masks=predicted_masks, mask_targets=mask_targets,
                                     ground_truth_masks=ground_truth_masks, proposal_boxes=proposals,
                                     matched_ground_truth_indices=matched_indices)
    assert targets.dtype == torch.float32 and not targets.requires_grad
    assert torch.allclose(input=targets, other=torch.tensor(data=[1.0, 0.5, 0.5, 0.0, 1.0]))


def test_mask_scores_multiply_the_class_scores_by_the_clipped_iou() -> None:
    """
    Verifies the mask score of every detection: its class score times its predicted mask IoU, the IoU clipped to
    [0, 1] so that the mask scores stay within [0, 1], in the dtype of the class scores.
    """
    detection_scores = torch.tensor(data=[[0.9, 0.8, 0.6, 0.0]])
    predicted_mask_iou = torch.tensor(data=[[1.3, 0.5, -0.2, 0.7]], dtype=torch.bfloat16)
    mask_scores = compute_mask_scores(detection_scores=detection_scores, predicted_mask_iou=predicted_mask_iou)
    assert mask_scores.dtype == torch.float32
    assert torch.allclose(input=mask_scores, other=torch.tensor(data=[[0.9, 0.4, 0.0, 0.0]]))


def test_mask_iou_head_reads_the_box_features_and_the_pooled_mask() -> None:
    """
    Verifies the mask IoU head of the scored masks: one predicted IoU per class from the box features of the mask head
    and the mask of the class of every box max-pooled to the pool size, its last convolution halving the resolution
    (7 to 4), its predictor starting near 0; a mask resolution that the pool size does not divide is refused, and the
    mask head gives the IoU of the asked class of every box, None without mask scoring.
    """
    torch.manual_seed(seed=0)
    mask_iou_head = MaskIoUHead(input_channels=16, roi_align_pool_size=7, mask_resolution=14, convolutions=[2, 8],
                                fully_connected=[32, 16], number_classes=6, dtype=torch.float32,
                                device=torch.device("cpu"))
    assert mask_iou_head.mask_pooling_size == 2
    assert mask_iou_head.convolution_layers[0].in_channels == 17
    assert mask_iou_head.fully_connected_block[0].in_features == 8 * 4 * 4
    box_features, class_masks = torch.rand(size=(3, 16, 7, 7)), torch.rand(size=(3, 1, 14, 14))
    predicted_mask_iou = mask_iou_head(box_features=box_features, class_masks=class_masks)
    assert predicted_mask_iou.shape == (3, 6) and float(predicted_mask_iou.detach().abs().max()) < 0.5

    # The pooled mask is the largest probability of every 2 x 2 cell: a mask moved within its cells changes nothing
    moved_masks = class_masks.reshape(3, 1, 7, 2, 7, 2).flip(dims=(3,)).reshape(3, 1, 14, 14)
    assert torch.allclose(input=mask_iou_head(box_features=box_features, class_masks=moved_masks),
                          other=predicted_mask_iou)
    with pytest.raises(expected_exception=ValueError, match="pool size 10, which must divide it"):
        MaskIoUHead(input_channels=16, roi_align_pool_size=10, mask_resolution=28, convolutions=[2, 8],
                    fully_connected=[32], number_classes=6, dtype=torch.float32, device=torch.device("cpu"))

    # A scored mask head gives the IoU of the asked class of every box, an unscored one none
    configuration, device, dtype = get_detection_model_configuration(mask_head=True)
    scored_head = MaskHead(head_configuration={"roi_align_pool_size": 7, "convolutions": [2, 16]}, mask_resolution=14,
                           upsampling="transposed", number_classes=6, input_channels=8,
                           feature_map_normalization="batch", batch_normalization_momentum=0.1, input_image_size=128,
                           feature_map_sizes={"5": 16}, modules=configuration["MaskHead"]["modules"],
                           skip_information={}, fine_input_channels=0, fusion_convolutions=[0, 0],
                           mask_scoring={"activated": True, "convolutions": [2, 8], "fully_connected": [32]},
                           dtype=dtype, device=device).eval()
    pooled_features, class_indices = torch.rand(size=(3, 8, 7, 7)), torch.tensor(data=[1, 5, 2])
    with torch.no_grad():
        mask_logits, class_mask_iou = scored_head.predict_masks_and_mask_iou(
            pooled_features=pooled_features, fine_features=None, class_indices=class_indices)
        box_features = scored_head.convolutional_block(input_tensor=pooled_features)
        every_class_mask_iou = scored_head.mask_iou_head(
            box_features=box_features, class_masks=torch.sigmoid(input=mask_logits[torch.arange(3), class_indices])[
                :, None])
    assert torch.equal(input=mask_logits, other=scored_head.predict_from_pooled_features(
        pooled_features=pooled_features, fine_features=None))
    assert torch.allclose(input=class_mask_iou, other=every_class_mask_iou[torch.arange(3), class_indices])
    unscored_head = build_mask_head(roi_align_pool_size=7, mask_resolution=14, upsampling="transposed")
    assert unscored_head.mask_iou_head is None
    assert unscored_head.predict_masks_and_mask_iou(pooled_features=pooled_features, fine_features=None,
                                                    class_indices=class_indices)[1] is None


def test_pasted_masks_fill_their_boxes() -> None:
    """
    Verifies that a full mask pasted into a box covers exactly the pixels of the box, that the mask of a right half
    covers the right half of its box, and that nothing is pasted outside the boxes.
    """
    full_mask = torch.ones(size=(14, 14))
    right_half_mask = torch.zeros(size=(14, 14))
    right_half_mask[:, 7:] = 1.0
    boxes = torch.tensor(data=[[10.0, 20.0, 30.0, 34.0], [0.0, 0.0, 16.0, 16.0]])

    pasted = paste_instance_masks(masks=torch.stack(tensors=[full_mask, right_half_mask]), boxes=boxes,
                                  image_height=48, image_width=40)

    assert pasted.shape == (2, 48, 40)
    expected_box = torch.zeros(size=(48, 40), dtype=torch.bool)
    expected_box[20:34, 10:30] = True
    assert torch.equal(input=pasted[0] >= 0.5, other=expected_box)
    expected_half = torch.zeros(size=(48, 40), dtype=torch.bool)
    expected_half[0:16, 8:16] = True
    assert torch.equal(input=pasted[1] >= 0.5, other=expected_half)


def test_mask_intersection_over_union_of_known_masks() -> None:
    """
    Verifies the intersection over union of hand-made masks, computed in chunks of one predicted mask, with 0 for two
    empty masks and a [0, G] result without predicted masks.
    """
    predicted_masks = torch.zeros(size=(3, 8, 8), dtype=torch.bool)
    predicted_masks[0, 0:4, 0:4] = True
    predicted_masks[1, 0:4, 0:2] = True
    ground_truth_masks = torch.zeros(size=(2, 8, 8), dtype=torch.bool)
    ground_truth_masks[0, 0:4, 0:4] = True

    overlaps = compute_mask_intersection_over_union(predicted_masks=predicted_masks,
                                                    ground_truth_masks=ground_truth_masks, chunk_size=1)

    assert torch.equal(input=overlaps, other=torch.tensor(data=[[1.0, 0.0], [0.5, 0.0], [0.0, 0.0]],
                                                          dtype=torch.float64))
    assert compute_mask_intersection_over_union(predicted_masks=predicted_masks[:0],
                                                ground_truth_masks=ground_truth_masks, chunk_size=4).shape == (0, 2)


@pytest.mark.parametrize(argnames="device", argvalues=[
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(condition=not torch.cuda.is_available(),
                                                         reason="The float16 segment products run on the GPU."))])
@pytest.mark.parametrize(argnames=["image_height", "image_width"], argvalues=[(37, 53), (64, 64), (8, 2048)])
def test_mask_intersection_over_union_matches_the_pixel_counts(device: str, image_height: int,
                                                               image_width: int) -> None:
    """
    Verifies that the segment products (float16 on the GPU, float32 on the CPU) give the intersection over union of
    the pixel counts of the masks exactly, on images whose pixels fill whole segments or not, with a full predicted mask
    and a full ground truth mask counting more pixels than float16 holds exactly, and the [D, 0] and [0, G] results
    without ground truth or predicted masks.

    Args:
        device (str): The device of the masks.
        image_height (int): The height of the masks.
        image_width (int): The width of the masks.
    """
    generator = torch.Generator().manual_seed(0)
    predicted_masks = torch.rand(size=(6, image_height, image_width), generator=generator) > 0.5
    ground_truth_masks = torch.rand(size=(4, image_height, image_width), generator=generator) > 0.7
    predicted_masks[0], ground_truth_masks[0], predicted_masks[1] = True, True, False

    # Intersection over union of the pixel counts in int64, the ratio in float64
    intersections = (predicted_masks[:, None] & ground_truth_masks[None]).flatten(start_dim=2).sum(dim=2)
    unions = (predicted_masks.flatten(start_dim=1).sum(dim=1)[:, None] +
              ground_truth_masks.flatten(start_dim=1).sum(dim=1)[None, :] - intersections)
    expected_overlaps = intersections.to(dtype=torch.float64) / unions.to(dtype=torch.float64)

    overlaps = compute_mask_intersection_over_union(predicted_masks=predicted_masks.to(device=device),
                                                    ground_truth_masks=ground_truth_masks.to(device=device),
                                                    chunk_size=4)

    assert torch.equal(input=overlaps.cpu(), other=expected_overlaps)
    assert compute_mask_intersection_over_union(predicted_masks=predicted_masks.to(device=device),
                                                ground_truth_masks=ground_truth_masks[:0].to(device=device),
                                                chunk_size=4).shape == (6, 0)
    assert compute_mask_intersection_over_union(predicted_masks=predicted_masks[:0].to(device=device),
                                                ground_truth_masks=ground_truth_masks.to(device=device),
                                                chunk_size=4).shape == (0, 4)
