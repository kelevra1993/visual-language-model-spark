"""
Tests of the self-supervised losses of vision_training/losses/self_supervised_losses.py, which the self-supervised
model computes on the embeddings of the two views of every image: InfoNCE against the cross-entropy written by hand,
its value on identical and on random views, Barlow Twins on decorrelated and on collapsed embeddings, both losses in
float32 under the bfloat16 autocast of mixed precision, and the standard deviation of the embeddings watching the
collapse. InfoNCE given the classes of the images (SupCon, the contrastive loss of the image classifier) is checked
against the cross-entropy written by hand, gives the loss of SimCLR when no two images share a class, and counts a view
of the same class as a found positive.
"""

import math
import pytest
import torch

from torch.nn import functional as torch_functional

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.losses.self_supervised_losses import (compute_barlow_twins_loss,
                                                           compute_embedding_standard_deviation,
                                                           compute_info_nce_loss)


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


def test_info_nce_loss_of_identical_and_random_views() -> None:
    """
    Verifies the two reference values of InfoNCE: views identical to each other and orthogonal to the views of the
    other images all find their partner (positive pair accuracy 1, loss near 0 at a small temperature), and random
    embeddings, nearly orthogonal in a large dimension, give a loss near log(2N - 1) at temperature 1.
    """
    # Identical views of 4 images with orthogonal embeddings
    orthogonal_embeddings = torch.eye(n=4, m=6)
    outputs = compute_info_nce_loss(first_embeddings=orthogonal_embeddings, second_embeddings=orthogonal_embeddings,
                                    temperature=0.05)
    assert float(outputs["positive_pair_accuracy"]) == 1.0
    assert float(outputs["info_nce_loss"]) < 1e-6

    # Random embeddings of 8 images in 4096 dimensions
    generator = torch.Generator().manual_seed(1)
    outputs = compute_info_nce_loss(first_embeddings=torch.randn(size=(8, 4096), generator=generator),
                                    second_embeddings=torch.randn(size=(8, 4096), generator=generator),
                                    temperature=1.0)
    assert float(outputs["info_nce_loss"]) == pytest.approx(expected=math.log(2 * 8 - 1), abs=0.02)


def test_supervised_info_nce_loss_matches_the_cross_entropy_written_by_hand() -> None:
    """
    Verifies InfoNCE given the classes of 4 images (classes 0, 1, 0 and 2) against the cross-entropy written by hand:
    every one of the 8 views is a classification among the 7 other views, whose right answers are the other view of its
    image and, for the views of class 0, both views of the other image of class 0, the loss of a view averaging the
    cross-entropy of its positives (3 for the views of class 0, 1 for the others).
    """
    generator = torch.Generator().manual_seed(5)
    first_embeddings = torch.randn(size=(4, 6), generator=generator)
    second_embeddings = torch.randn(size=(4, 6), generator=generator)
    labels = torch.tensor(data=[0, 1, 0, 2])
    temperature = 0.2

    # Average cross-entropy of the positives of every view over the other views
    embeddings = torch_functional.normalize(input=torch.cat(tensors=[first_embeddings, second_embeddings]), dim=1)
    view_labels = torch.cat(tensors=[labels, labels]).tolist()
    expected_losses, positive_counts = [], []
    for view_index in range(8):
        other_indices = [other_index for other_index in range(8) if other_index != view_index]
        logits = torch.stack(tensors=[embeddings[view_index] @ embeddings[other_index] / temperature
                                      for other_index in other_indices])
        log_probabilities = torch.log_softmax(input=logits, dim=0)
        positive_positions = [position for position, other_index in enumerate(other_indices)
                              if view_labels[other_index] == view_labels[view_index]]
        expected_losses.append(-log_probabilities[positive_positions].mean())
        positive_counts.append(len(positive_positions))
    assert positive_counts == [3, 1, 3, 1, 3, 1, 3, 1]

    outputs = compute_info_nce_loss(first_embeddings=first_embeddings, second_embeddings=second_embeddings,
                                    temperature=temperature, labels=labels)
    assert torch.isclose(input=outputs["info_nce_loss"], other=torch.stack(tensors=expected_losses).mean(), atol=1e-6)


def test_supervised_info_nce_loss_without_shared_classes_is_the_simclr_loss() -> None:
    """
    Verifies that the classes only add positives: images of different classes, or without class, give exactly the loss
    and the positive pair accuracy of SimCLR, their other view being their only positive.
    """
    generator = torch.Generator().manual_seed(6)
    first_embeddings = torch.randn(size=(5, 8), generator=generator)
    second_embeddings = torch.randn(size=(5, 8), generator=generator)
    simclr_outputs = compute_info_nce_loss(first_embeddings=first_embeddings, second_embeddings=second_embeddings,
                                           temperature=0.1)
    for labels in [torch.tensor(data=[4, 0, 2, 1, 3]), torch.full(size=(5,), fill_value=IGNORED_LABEL_INDEX)]:
        outputs = compute_info_nce_loss(first_embeddings=first_embeddings, second_embeddings=second_embeddings,
                                        temperature=0.1, labels=labels)
        assert torch.isclose(input=outputs["info_nce_loss"], other=simclr_outputs["info_nce_loss"], atol=1e-6)
        assert torch.equal(input=outputs["positive_pair_accuracy"], other=simclr_outputs["positive_pair_accuracy"])


def test_supervised_positive_pair_accuracy_counts_the_views_of_the_same_class() -> None:
    """
    Verifies the positive pair accuracy given the classes: images 0 and 1 of the same class share an embedding, so the
    most similar view of each of their views is a view of the other image, a positive with the classes and a negative
    without them, while the views of images 2 and 3, orthogonal to every other image, find their partner either way.
    """
    embeddings = torch.tensor(data=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    second_embeddings = embeddings + torch.tensor(data=[[0.0, 0.1, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.0],
                                                         [0.0, 0.0, 0.0]])
    supervised_outputs = compute_info_nce_loss(first_embeddings=embeddings, second_embeddings=second_embeddings,
                                               temperature=0.1, labels=torch.tensor(data=[7, 7, 1, 2]))
    simclr_outputs = compute_info_nce_loss(first_embeddings=embeddings, second_embeddings=second_embeddings,
                                           temperature=0.1)
    assert float(supervised_outputs["positive_pair_accuracy"]) == 1.0
    assert float(simclr_outputs["positive_pair_accuracy"]) < 1.0


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
                                                         second_embeddings=second_embeddings,
                                                         off_diagonal_weight=0.005)
    assert all(value.dtype == torch.float32 for value in [*info_nce_outputs.values(), *barlow_twins_outputs.values()])

    # The loss of both methods trains the embeddings
    (info_nce_outputs["info_nce_loss"] + barlow_twins_outputs["barlow_twins_loss"]).backward()
    assert first_embeddings.grad is not None and bool(first_embeddings.grad.abs().sum() > 0)


def test_embedding_standard_deviation_tells_the_collapse() -> None:
    """
    Verifies the standard deviation of the normalized embeddings, times the square root of their dimension: about 1
    for embeddings spread over the whole sphere, 0 for collapsed ones, and well below 1 for non-negative
    representations, which only cover their part of the sphere.
    """
    generator = torch.Generator().manual_seed(4)
    spread_embeddings = torch.randn(size=(4096, 64), generator=generator)
    assert float(compute_embedding_standard_deviation(embeddings=spread_embeddings)) == pytest.approx(expected=1.0,
                                                                                                     abs=0.02)
    assert float(compute_embedding_standard_deviation(embeddings=torch.ones(size=(32, 64)))) == pytest.approx(
        expected=0.0, abs=1e-6)
    non_negative_representations = spread_embeddings.relu()
    assert 0.5 < float(compute_embedding_standard_deviation(embeddings=non_negative_representations)) < 0.9
