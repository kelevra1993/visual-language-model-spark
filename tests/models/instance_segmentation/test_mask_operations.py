"""
Tests of vision_training/models/instance_segmentation/mask_operations.py and the mask head: the shapes of the mask head
outputs, the mask targets cropped from the ground truth masks, the pasting of the predicted masks into their boxes and
the intersection over union of binary masks.
"""

import torch

from tests.fixtures.configuration_fixtures import get_detection_model_configuration
from vision_training.models.instance_segmentation.mask_head import MaskHead
from vision_training.models.instance_segmentation.mask_operations import (build_mask_targets,
                                                                          compute_mask_intersection_over_union,
                                                                          paste_instance_masks)


def test_mask_head_predicts_one_doubled_mask_per_class() -> None:
    """
    Verifies that the mask head of a level gives [R, C, 2P, 2P] logits for R boxes, including no box at all.
    """
    configuration, device, dtype = get_detection_model_configuration(mask_head=True)
    mask_head_configuration = configuration["MaskHead"]
    mask_head = MaskHead(head_configuration=[2, 16], roi_align_pool_size=7, number_classes=6, input_channels=8,
                         feature_map_normalization="batch", input_image_size=128, feature_map_size=16,
                         modules=mask_head_configuration["modules"], dtype=dtype, device=device).eval()
    feature_map = torch.rand(size=(2, 8, 16, 16))
    boxes = torch.tensor(data=[[0.0, 10.0, 10.0, 60.0, 80.0], [1.0, 0.0, 0.0, 128.0, 128.0], [1.0, 5.0, 5.0, 9.0, 9.0]])

    assert mask_head(proposal_boxes=boxes, input_tensor=feature_map).shape == (3, 6, 14, 14)
    assert mask_head(proposal_boxes=boxes[:0], input_tensor=feature_map).shape == (0, 6, 14, 14)


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

    assert torch.allclose(input=overlaps, other=torch.tensor(data=[[1.0, 0.0], [0.5, 0.0], [0.0, 0.0]]))
    assert compute_mask_intersection_over_union(predicted_masks=predicted_masks[:0],
                                                ground_truth_masks=ground_truth_masks, chunk_size=4).shape == (0, 2)
