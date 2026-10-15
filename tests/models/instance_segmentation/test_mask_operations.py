"""
Tests of vision_training/models/instance_segmentation/mask_operations.py and the mask head: the mask geometry of the
mask head (its doublings, the resize of its masks and the shapes of its outputs), the loading of the checkpoints
written before the mask geometry could be configured, the mask targets cropped from the ground truth masks, the pasting
of the predicted masks into their boxes and the intersection over union of binary masks.
"""

import pytest
import torch

from torch import nn

from tests.fixtures.configuration_fixtures import get_detection_model_configuration
from vision_training.models.instance_segmentation.mask_head import MaskHead, count_mask_upsamplings
from vision_training.models.instance_segmentation.mask_operations import (build_mask_targets,
                                                                          compute_mask_intersection_over_union,
                                                                          paste_instance_masks)


def build_mask_head(roi_align_pool_size: int, mask_resolution: int, upsampling: str) -> MaskHead:
    """
    Builds the mask head of a 16 x 16 feature map of 8 channels for a 128 pixel input and 6 classes, with the modules
    of the mask head fixture and the given mask geometry, in evaluation mode.

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
                    feature_map_size=16, modules=configuration["MaskHead"]["modules"], dtype=dtype,
                    device=device).eval()


@pytest.mark.parametrize(argnames=["roi_align_pool_size", "mask_resolution", "upsampling", "number_upsamplings"],
                         argvalues=[(7, 14, "transposed", 1), (7, 28, "bilinear", 2), (10, 28, "transposed", 1),
                                    (14, 14, "bilinear", 0)])
def test_mask_head_doubles_its_pooled_features_up_to_the_mask_resolution(roi_align_pool_size: int,
                                                                          mask_resolution: int, upsampling: str,
                                                                          number_upsamplings: int) -> None:
    """
    Verifies that the mask head of a level doubles its pooled features as many times as fits in the mask resolution,
    with layers of the chosen upsampling, and gives [R, C, M, M] logits for R boxes, including no box at all, the
    logits of a pool size whose doublings fall short of M being resized to it.

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
    assert mask_head(proposal_boxes=boxes, input_tensor=feature_map).shape == (3, 6, mask_resolution, mask_resolution)
    assert mask_head(proposal_boxes=boxes[:0], input_tensor=feature_map).shape == (0, 6, mask_resolution,
                                                                                    mask_resolution)


def test_mask_head_loads_the_checkpoints_of_the_single_transposed_convolution() -> None:
    """
    Verifies that the weights of a mask head saved before the mask geometry could be configured, its transposed
    convolution named 'upsampling', load into the first upsampling layer of the same default geometry, strictly and
    giving the same masks, inside a larger model as in the checkpoints.
    """
    torch.manual_seed(seed=0)
    saved_head = nn.ModuleDict(modules={"5": build_mask_head(roi_align_pool_size=7, mask_resolution=14,
                                                             upsampling="transposed")})
    legacy_state = {key.replace("upsampling_layers.0.", "upsampling."): value
                    for key, value in saved_head.state_dict().items()}
    assert "5.upsampling.weight" in legacy_state and "5.upsampling.bias" in legacy_state

    loaded_head = nn.ModuleDict(modules={"5": build_mask_head(roi_align_pool_size=7, mask_resolution=14,
                                                              upsampling="transposed")})
    loaded_head.load_state_dict(state_dict=legacy_state, strict=True)
    feature_map = torch.rand(size=(1, 8, 16, 16))
    boxes = torch.tensor(data=[[0.0, 10.0, 10.0, 60.0, 80.0]])
    assert torch.equal(input=loaded_head["5"](proposal_boxes=boxes, input_tensor=feature_map),
                       other=saved_head["5"](proposal_boxes=boxes, input_tensor=feature_map))


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
