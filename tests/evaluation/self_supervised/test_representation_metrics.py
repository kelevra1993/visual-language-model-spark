"""
Tests of the label-free evaluation of the self-supervised representations of
vision_training/evaluation/self_supervised/representation_metrics.py, which the self-supervised trainer computes on
the test split at every evaluated checkpoint: the k-nearest neighbour classification (the vote weighted by
similarity, ties) and the effective rank (orthogonal, rank-one and diverged representations).
"""

import math
import pytest
import torch

from vision_training.evaluation.self_supervised.representation_metrics import (compute_effective_rank,
                                                                               compute_nearest_neighbour_predictions)


def test_nearest_neighbour_vote_is_weighted_and_ties_go_to_the_lowest_class() -> None:
    """
    Verifies that the neighbours vote with the weight exp(similarity / 0.07), so that one very close neighbour
    outweighs two distant ones, and that two classes of equal votes give the class of the lowest index.
    """
    # The first image has one very close neighbour of class 1 and two distant neighbours of class 0
    labels = torch.tensor(data=[2, 1, 0, 0])
    representations = torch.tensor(data=[[1.0, 0.0], [1.0, 0.05], [0.0, 1.0], [0.0, 1.0]])
    predicted_labels = compute_nearest_neighbour_predictions(representations=representations, labels=labels,
                                                             number_neighbours=3, number_classes=3)
    assert int(predicted_labels[0]) == 1

    # The first image has one neighbour of class 2 and one of class 1 at the same similarity
    labels = torch.tensor(data=[0, 2, 1])
    representations = torch.tensor(data=[[1.0, 0.0], [0.6, 0.8], [0.6, -0.8]])
    predicted_labels = compute_nearest_neighbour_predictions(representations=representations, labels=labels,
                                                             number_neighbours=2, number_classes=3)
    assert int(predicted_labels[0]) == 1


def test_effective_rank_counts_the_directions_of_the_representations() -> None:
    """
    Verifies the effective rank: representations spread evenly over r orthogonal directions give r, representations
    along a single direction give 1, and diverged representations give NaN instead of stopping the evaluation.
    """
    # Up to the epsilon added to the singular value distribution before its logarithm
    orthogonal_representations = torch.cat(tensors=[torch.eye(n=5, m=12), -torch.eye(n=5, m=12)], dim=0)
    assert compute_effective_rank(representations=orthogonal_representations) == pytest.approx(expected=5.0, abs=1e-6)

    rank_one_representations = torch.arange(start=1, end=9, dtype=torch.float32)[:, None] * torch.ones(size=(1, 12))
    assert compute_effective_rank(representations=rank_one_representations) == pytest.approx(expected=1.0, abs=1e-6)

    diverged_representations = orthogonal_representations.clone()
    diverged_representations[3, 2] = float("nan")
    assert math.isnan(compute_effective_rank(representations=diverged_representations))
    diverged_representations[3, 2] = float("inf")
    assert math.isnan(compute_effective_rank(representations=diverged_representations))
