"""
Tests of vision_training/losses/keypoint_losses.py, the losses of the bottom-up keypoint detector: compute_heatmap_loss,
the focal loss of the heatmap logits, on a hand-computed case, divided by the number of keypoint cells whatever the
number of images, channels and empty cells, close to 0 for confident right predictions, trained towards 0 on the empty
channels, computed in float32 and leaving out the ignored cells but their keypoint cells; and compute_tag_losses, the
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


def test_heatmap_loss_is_divided_by_the_number_of_keypoints() -> None:
    """
    Verifies that the focal loss is the loss of one keypoint whatever the number of keypoints, channels and empty cells:
    the same image twice gives the same loss as once, and an empty channel and empty cells predicted confidently at 0
    add nearly nothing; that confident right predictions give nearly 0; that a batch without keypoint is divided by 1;
    and that an empty channel at the probability 0.5 receives a gradient towards lower logits.
    """
    target_heatmaps = torch.tensor(data=[[[[1.0, 0.5], [0.0, 0.0]]]])
    logits = torch.tensor(data=[[[[0.5, 0.0], [-1.0, -2.0]]]])
    loss = compute_heatmap_loss(predicted_logits=logits, target_heatmaps=target_heatmaps)
    twice_loss = compute_heatmap_loss(predicted_logits=logits.repeat(2, 1, 1, 1),
                                      target_heatmaps=target_heatmaps.repeat(2, 1, 1, 1))
    assert math.isclose(twice_loss.item(), loss.item(), rel_tol=1e-6)

    # An empty channel and empty cells predicted at a logit of -20 add nearly nothing
    padded_logits = torch.full(size=(1, 2, 8, 8), fill_value=-20.0)
    padded_logits[:, :1, :2, :2] = logits
    padded_targets = torch.zeros(size=(1, 2, 8, 8))
    padded_targets[:, :1, :2, :2] = target_heatmaps
    padded_loss = compute_heatmap_loss(predicted_logits=padded_logits, target_heatmaps=padded_targets)
    assert math.isclose(padded_loss.item(), loss.item(), rel_tol=1e-6)

    # Confident right predictions: a logit of 20 at the keypoint cell and of -20 at the other cells
    confident_logits = torch.tensor(data=[[[[20.0, -20.0], [-20.0, -20.0]]]])
    confident_loss = compute_heatmap_loss(predicted_logits=confident_logits, target_heatmaps=target_heatmaps)
    assert confident_loss.item() < 1e-3

    # A batch without keypoint is divided by 1, and its empty channel is trained towards lower logits
    empty_logits = torch.zeros(size=(1, 1, 2, 2), requires_grad=True)
    empty_loss = compute_heatmap_loss(predicted_logits=empty_logits, target_heatmaps=torch.zeros(size=(1, 1, 2, 2)))
    assert math.isclose(empty_loss.item(), 4.0 * 0.25 * math.log(2.0), rel_tol=1e-6)
    empty_loss.backward()
    assert bool((empty_logits.grad > 0.0).all())


def test_heatmap_loss_leaves_out_the_ignored_cells() -> None:
    """
    Verifies that the ignored cells give no focal term and no gradient in every channel, while a keypoint cell lying
    in them stays trained: of the 2 x 2 heatmap of the hand-computed case at the probability 0.5, with every cell but
    the last one ignored, only the keypoint cell and the last empty cell count, in both channels sharing the ignored
    cells; and that no ignored cell gives the loss of every cell.
    """
    target_heatmaps = torch.tensor(data=[[[[1.0, 0.5], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]])
    logits = torch.zeros(size=(1, 2, 2, 2), requires_grad=True)
    ignored_cells = torch.tensor(data=[[[[True, True], [True, False]]]])
    loss = compute_heatmap_loss(predicted_logits=logits, target_heatmaps=target_heatmaps, ignored_cells=ignored_cells)
    assert math.isclose(loss.item(), (0.25 + 0.25 + 0.25) * math.log(2.0), rel_tol=1e-6)

    # Only the keypoint cell and the last cell of every channel receive a gradient
    loss.backward()
    assert (logits.grad != 0.0).tolist() == [[[[True, False], [False, True]], [[False, False], [False, True]]]]

    # Without any ignored cell, the loss is that of every cell
    unignored_loss = compute_heatmap_loss(predicted_logits=logits.detach(), target_heatmaps=target_heatmaps,
                                          ignored_cells=torch.zeros(size=(1, 1, 2, 2), dtype=torch.bool))
    assert math.isclose(unignored_loss.item(), compute_heatmap_loss(predicted_logits=logits.detach(),
                                                                    target_heatmaps=target_heatmaps).item(),
                        rel_tol=1e-6)


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


def test_objects_of_one_keypoint_and_empty_batches_give_finite_losses() -> None:
    """
    Verifies that objects of a single trained keypoint have no pull loss, that two such objects of the same class with
    equal tags give the largest push loss of 1, and that a batch without any trained keypoint gives 0 to both losses
    instead of a NaN.
    """
    tag_maps = torch.full(size=(1, 1, 2, 2), fill_value=0.5)
    pull_loss, push_loss = compute_tag_losses(tag_maps=tag_maps, tag_indices=torch.tensor(data=[[[0], [3]]]),
                                              tag_valid=torch.ones(size=(1, 2, 1), dtype=torch.bool),
                                              object_labels=torch.tensor(data=[[1, 1]]))
    assert pull_loss.item() == 0.0 and math.isclose(push_loss.item(), 1.0, rel_tol=1e-6)

    # No trained keypoint at all
    empty_pull, empty_push = compute_tag_losses(tag_maps=tag_maps, tag_indices=torch.zeros(size=(1, 1, 1),
                                                                                          dtype=torch.int64),
                                                tag_valid=torch.zeros(size=(1, 1, 1), dtype=torch.bool),
                                                object_labels=torch.zeros(size=(1, 1), dtype=torch.int64))
    assert empty_pull.item() == 0.0 and empty_push.item() == 0.0
