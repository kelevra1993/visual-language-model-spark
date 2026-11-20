"""
Tests of the self-supervised losses of vision_training/losses/self_supervised_losses.py, which the self-supervised
model computes on the embeddings of the two views of every image: InfoNCE against the cross-entropy written by hand,
Barlow Twins on decorrelated and on collapsed embeddings, and both losses in float32 under the bfloat16 autocast of
mixed precision.
"""

import pytest
import torch

from torch.nn import functional as torch_functional

from vision_training.losses.self_supervised_losses import compute_barlow_twins_loss, compute_info_nce_loss


def test_info_nce_loss_matches_the_cross_entropy_written_by_hand() -> None:
    """
    Verifies InfoNCE on a batch of 3 images against the cross-entropy written by hand: every one of the 6 views is a
    classification among the 5 other views by cosine similarity divided by the temperature, whose right answer is the
    other view of its image (index i + 3 for a first view, i - 3 for a second view).
    """
    generator = torch.Generator().manual_seed(0)
    first_embeddings = torch.randn(size=(3, 5), generator=generator)
    second_embeddings = torch.randn(size=(3, 5), generator=generator)
    temperature = 0.5

    # Cross-entropy of every view over the other views, the partner being the right answer
    embeddings = torch_functional.normalize(input=torch.cat(tensors=[first_embeddings, second_embeddings]), dim=1)
    expected_losses = []
    for view_index in range(6):
        partner_index = (view_index + 3) % 6
        other_indices = [other_index for other_index in range(6) if other_index != view_index]
        logits = torch.stack(tensors=[embeddings[view_index] @ embeddings[other_index] / temperature
                                      for other_index in other_indices])
        expected_losses.append(-torch.log_softmax(input=logits, dim=0)[other_indices.index(partner_index)])

    outputs = compute_info_nce_loss(first_embeddings=first_embeddings, second_embeddings=second_embeddings,
                                    temperature=temperature)
    assert torch.isclose(input=outputs["info_nce_loss"], other=torch.stack(tensors=expected_losses).mean(), atol=1e-6)


def test_barlow_twins_loss_of_decorrelated_and_collapsed_embeddings() -> None:
    """
    Verifies the two reference values of Barlow Twins: identical views of independent features give a
    cross-correlation close to the identity (diagonal 1, off-diagonal near 0, loss near 0), and constant embeddings,
    the collapse, give a zero cross-correlation, the invariance term then being the embedding dimension.
    """
    generator = torch.Generator().manual_seed(2)
    decorrelated_embeddings = torch.randn(size=(4096, 8), generator=generator)
    outputs = compute_barlow_twins_loss(first_embeddings=decorrelated_embeddings,
                                        second_embeddings=decorrelated_embeddings, off_diagonal_weight=0.005)
    assert float(outputs["cross_correlation_diagonal"]) == pytest.approx(expected=1.0, abs=1e-4)
    assert float(outputs["cross_correlation_off_diagonal"]) < 0.05
    assert float(outputs["barlow_twins_loss"]) < 1e-3

    # Every image giving the same embedding
    collapsed_embeddings = torch.ones(size=(16, 8))
    outputs = compute_barlow_twins_loss(first_embeddings=collapsed_embeddings, second_embeddings=collapsed_embeddings,
                                        off_diagonal_weight=0.005)
    assert float(outputs["cross_correlation_diagonal"]) == 0.0
    assert float(outputs["barlow_twins_loss"]) == pytest.approx(expected=8.0)


def test_losses_stay_in_float32_under_mixed_precision() -> None:
    """
    Verifies that both losses and their metrics are computed in float32 from bfloat16 embeddings within the bfloat16
    autocast of a mixed precision forward pass, the similarities and correlations losing too much precision otherwise,
    and that the gradient reaches the embeddings.
    """
    generator = torch.Generator().manual_seed(3)
    first_embeddings = torch.randn(size=(6, 16), generator=generator).to(dtype=torch.bfloat16).requires_grad_()
    second_embeddings = torch.randn(size=(6, 16), generator=generator).to(dtype=torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        info_nce_outputs = compute_info_nce_loss(first_embeddings=first_embeddings,
                                                 second_embeddings=second_embeddings, temperature=0.1)
        barlow_twins_outputs = compute_barlow_twins_loss(first_embeddings=first_embeddings,
                                                         second_embeddings=second_embeddings, off_diagonal_weight=0.005)
    assert all(value.dtype == torch.float32 for value in [*info_nce_outputs.values(), *barlow_twins_outputs.values()])

    # The loss of both methods trains the embeddings
    (info_nce_outputs["info_nce_loss"] + barlow_twins_outputs["barlow_twins_loss"]).backward()
    assert first_embeddings.grad is not None and bool(first_embeddings.grad.abs().sum() > 0)
