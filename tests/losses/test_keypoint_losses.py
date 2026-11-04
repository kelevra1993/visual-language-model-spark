"""
Tests of vision_training/losses/keypoint_losses.py, the losses of the bottom-up keypoint detector: compute_heatmap_loss,
the focal loss of the heatmap logits, on a hand-computed case, computed in float32; and compute_tag_losses, the
associative embedding pull and push losses, on a hand-computed image of three objects (two of the same class, which
push each other apart, and one of another class, which pushes nothing) next to an image without object, the untrained
keypoints giving neither loss nor gradient.
"""

import math
import torch

from vision_training.losses.keypoint_losses import compute_heatmap_loss, compute_tag_losses


def test_heatmap_loss_matches_a_hand_computed_case() -> None:
    """
    Verifies the focal loss of a 2 x 2 heatmap whose logits are all 0 (probability 0.5): the keypoint cell at 1 gives
    (1 - 0.5) ** 2 x log(2), the cell at 0.5 next to it (1 - 0.5) ** 4 x 0.5 ** 2 x log(2) and each of the two empty
    cells 0.5 ** 2 x log(2), divided by the single keypoint cell. A bfloat16 prediction gives a float32 loss.
    """
    target_heatmaps = torch.tensor(data=[[[[1.0, 0.5], [0.0, 0.0]]]])
    loss = compute_heatmap_loss(predicted_logits=torch.zeros(size=(1, 1, 2, 2)), target_heatmaps=target_heatmaps)
    expected_loss = (0.25 + 0.0625 * 0.25 + 2.0 * 0.25) * math.log(2.0)
    assert math.isclose(loss.item(), expected_loss, rel_tol=1e-6)

    # A bfloat16 prediction is compared in float32
    bfloat16_loss = compute_heatmap_loss(predicted_logits=torch.zeros(size=(1, 1, 2, 2), dtype=torch.bfloat16),
                                         target_heatmaps=target_heatmaps)
    assert bfloat16_loss.dtype == torch.float32 and math.isclose(bfloat16_loss.item(), expected_loss, rel_tol=1e-6)


def test_tag_losses_match_a_hand_computed_case() -> None:
    """
    Verifies the pull and push losses on a batch of two images with 2 x 2 tag maps of 2 channels. In the first image,
    object 0 (class 1) has the tags 1 and 3 (reference 2, pull 1), object 1 (class 1) the tags 4 and 4 (reference 4,
    pull 0), and object 2 (class 2) the tag 2 on its only trained keypoint (reference 2, pull 0): the pull is
    (1 + 0 + 0) / 3, and the push exp(-(2 - 4) ** 2) for the two ordered pairs of class 1, object 2 having the same
    reference as object 0 but another class. The second image has no object and gives 0 to both, halving the means.
    The tags of the cells read by no trained keypoint receive no gradient.
    """
    tag_maps = torch.zeros(size=(2, 2, 2, 2))
    tag_maps[0].view(-1)[[0, 4, 1, 5, 2]] = torch.tensor(data=[1.0, 3.0, 4.0, 4.0, 2.0])
    tag_maps[0].view(-1)[7] = 50.0
    tag_maps.requires_grad_(True)

    # Flat [C x h x w] indices of the keypoints, the second keypoint of object 2 reading the untrained cell 7
    tag_indices = torch.zeros(size=(2, 3, 2), dtype=torch.int64)
    tag_indices[0] = torch.tensor(data=[[0, 4], [1, 5], [2, 7]])
    tag_valid = torch.zeros(size=(2, 3, 2), dtype=torch.bool)
    tag_valid[0] = torch.tensor(data=[[True, True], [True, True], [True, False]])
    object_labels = torch.tensor(data=[[1, 1, 2], [0, 0, 0]])
    pull_loss, push_loss = compute_tag_losses(tag_maps=tag_maps, tag_indices=tag_indices, tag_valid=tag_valid,
                                              object_labels=object_labels)
    assert math.isclose(pull_loss.item(), (1.0 / 3.0 + 0.0) / 2.0, rel_tol=1e-6)
    assert math.isclose(push_loss.item(), (math.exp(-4.0) + 0.0) / 2.0, rel_tol=1e-6)

    # Only the cells of the trained keypoints receive a gradient, the untrained cell 7 none whatever its tag
    (pull_loss + push_loss).backward()
    gradient_cells = tag_maps.grad.reshape(2, -1) != 0.0
    assert gradient_cells[0].nonzero().flatten().tolist() == [0, 1, 4, 5] and not gradient_cells[1].any()
