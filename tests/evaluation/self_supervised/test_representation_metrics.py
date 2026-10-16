"""
Tests of the label-free evaluation of the self-supervised representations of
vision_training/evaluation/self_supervised/representation_metrics.py, which the self-supervised trainer computes on
the test split at every evaluated checkpoint: the k-nearest neighbour classification (separated clusters, the image
left out of its own neighbours, the vote weighted by similarity, ties, the chunks of the similarity matrix) and the
effective rank (orthogonal, rank-one and diverged representations).
"""

import math
import pytest
import torch

from unittest.mock import patch

from vision_training.evaluation.self_supervised import representation_metrics
from vision_training.evaluation.self_supervised.representation_metrics import (compute_effective_rank,
                                                                               compute_nearest_neighbour_predictions)


def test_nearest_neighbours_classify_separated_clusters() -> None:
    """
    Verifies that representations gathered in one cluster per class are all classified right by their neighbours,
    with more neighbours asked than the other images of the split (reduced to N - 1).
    """
    generator = torch.Generator().manual_seed(0)
    labels = torch.tensor(data=[0, 0, 0, 1, 1, 1, 2, 2, 2])
    cluster_centres = torch.eye(n=3, m=16) * 10.0
    representations = cluster_centres[labels] + 0.1 * torch.randn(size=(9, 16), generator=generator)
    for number_neighbours in [2, 20]:
        predicted_labels = compute_nearest_neighbour_predictions(representations=representations, labels=labels,
                                                                 number_neighbours=number_neighbours, number_classes=3)
        assert torch.equal(input=predicted_labels, other=labels)


def test_nearest_neighbours_leave_the_image_out() -> None:
    """
    Verifies the leave-one-out evaluation: an image is never its own neighbour, so the only image of a class is
    classified by the images of the other classes, even when its representation is far from all of them.
    """
    labels = torch.tensor(data=[0, 0, 0, 1])
    representations = torch.tensor(data=[[1.0, 0.0], [1.0, 0.1], [1.0, -0.1], [0.0, 1.0]])
    predicted_labels = compute_nearest_neighbour_predictions(representations=representations, labels=labels,
                                                             number_neighbours=1, number_classes=2)
    assert predicted_labels.tolist() == [0, 0, 0, 0]


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


def test_nearest_neighbours_give_the_same_predictions_by_chunks() -> None:
    """
    Verifies that splitting the similarity matrix in chunks of images, which keeps its memory bounded on large test
    splits, gives the predictions of a single chunk, the image of every chunk still being left out of its neighbours.
    """
    generator = torch.Generator().manual_seed(1)
    representations = torch.randn(size=(23, 8), generator=generator)
    labels = torch.randint(low=0, high=4, size=(23,), generator=generator)
    single_chunk_predictions = compute_nearest_neighbour_predictions(representations=representations, labels=labels,
                                                                     number_neighbours=5, number_classes=4)
    with patch.object(target=representation_metrics, attribute="SIMILARITY_CHUNK_SIZE", new=4):
        chunked_predictions = compute_nearest_neighbour_predictions(representations=representations, labels=labels,
                                                                    number_neighbours=5, number_classes=4)
    assert torch.equal(input=chunked_predictions, other=single_chunk_predictions)


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
