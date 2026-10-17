"""
Tests of vision_training/losses/keypoint_losses.py: compute_keypoint_heatmap_loss, the weighted mean squared error of
the heatmaps that the keypoint detector adds to the total loss, on a hand-computed case, with the unlabelled keypoints,
the keypoints outside the crop and the padding positions (weight 0) giving neither loss nor gradient.
"""

import torch

from vision_training.losses.keypoint_losses import compute_keypoint_heatmap_loss


def test_equal_heatmaps_give_no_loss() -> None:
    """
    Verifies that heatmaps predicted exactly as their targets give a loss of 0, and that a batch without any trained
    keypoint gives 0 instead of a NaN whatever its heatmaps.
    """
    target_heatmaps = torch.rand(size=(2, 3, 4, 5), generator=torch.Generator().manual_seed(0))
    loss = compute_keypoint_heatmap_loss(predicted_heatmaps=target_heatmaps.clone(), target_heatmaps=target_heatmaps,
                                         heatmap_weights=torch.ones(size=(2, 3)))
    assert loss.item() == 0.0

    # No trained keypoint at all
    untrained_loss = compute_keypoint_heatmap_loss(predicted_heatmaps=target_heatmaps + 1.0,
                                                   target_heatmaps=target_heatmaps,
                                                   heatmap_weights=torch.zeros(size=(2, 3)))
    assert untrained_loss.item() == 0.0


def test_heatmap_loss_matches_a_hand_computed_case() -> None:
    """
    Verifies the loss on two crops of two keypoints with 2 x 2 heatmaps: the mean squared errors over the cells are 1,
    4, 9 and 0.25, and only the three trained keypoints are averaged, (1 + 9 + 0.25) / 3.
    """
    predicted_heatmaps = torch.zeros(size=(2, 2, 2, 2))
    target_heatmaps = torch.stack(tensors=[torch.stack(tensors=[torch.full(size=(2, 2), fill_value=1.0),
                                                                torch.full(size=(2, 2), fill_value=2.0)]),
                                           torch.stack(tensors=[torch.full(size=(2, 2), fill_value=3.0),
                                                                torch.full(size=(2, 2), fill_value=0.5)])])
    heatmap_weights = torch.tensor(data=[[1.0, 0.0], [1.0, 1.0]])
    loss = compute_keypoint_heatmap_loss(predicted_heatmaps=predicted_heatmaps, target_heatmaps=target_heatmaps,
                                         heatmap_weights=heatmap_weights)
    assert torch.isclose(input=loss, other=torch.tensor(data=(1.0 + 9.0 + 0.25) / 3.0))

    # Only some cells differ: an error of 2 on one cell of four gives a mean squared error of 1
    partial_targets = torch.zeros(size=(1, 1, 2, 2))
    partial_targets[0, 0, 1, 0] = 2.0
    partial_loss = compute_keypoint_heatmap_loss(predicted_heatmaps=torch.zeros(size=(1, 1, 2, 2)),
                                                 target_heatmaps=partial_targets,
                                                 heatmap_weights=torch.ones(size=(1, 1)))
    assert torch.isclose(input=partial_loss, other=torch.tensor(data=1.0))


def test_untrained_keypoints_give_neither_loss_nor_gradient() -> None:
    """
    Verifies that the heatmaps of the keypoints of weight 0 change nothing to the loss, whatever their values, and
    receive no gradient, while every trained heatmap does, and that a bfloat16 prediction is compared in float32.
    """
    generator = torch.Generator().manual_seed(1)
    target_heatmaps = torch.rand(size=(2, 4, 3, 3), generator=generator)
    heatmap_weights = torch.tensor(data=[[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
    untrained_keypoints = heatmap_weights == 0.0
    predicted_heatmaps = torch.rand(size=(2, 4, 3, 3), generator=generator).requires_grad_(True)
    loss = compute_keypoint_heatmap_loss(predicted_heatmaps=predicted_heatmaps, target_heatmaps=target_heatmaps,
                                         heatmap_weights=heatmap_weights)

    # Any values on the untrained heatmaps give the same loss
    changed_heatmaps = predicted_heatmaps.detach().clone()
    changed_heatmaps[untrained_keypoints] = 100.0
    changed_loss = compute_keypoint_heatmap_loss(predicted_heatmaps=changed_heatmaps, target_heatmaps=target_heatmaps,
                                                 heatmap_weights=heatmap_weights)
    assert torch.isclose(input=changed_loss, other=loss.detach())

    # The gradient reaches the trained heatmaps only
    loss.backward()
    heatmap_gradients = predicted_heatmaps.grad
    assert torch.all(heatmap_gradients[untrained_keypoints] == 0.0)
    assert torch.all(heatmap_gradients[~untrained_keypoints].abs().sum(dim=(1, 2)) > 0.0)

    # A bfloat16 prediction gives a float32 loss
    bfloat16_loss = compute_keypoint_heatmap_loss(predicted_heatmaps=predicted_heatmaps.detach().to(
        dtype=torch.bfloat16), target_heatmaps=target_heatmaps, heatmap_weights=heatmap_weights)
    assert bfloat16_loss.dtype == torch.float32
