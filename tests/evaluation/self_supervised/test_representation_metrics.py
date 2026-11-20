"""
Tests of the label-free evaluation of the self-supervised representations of
vision_training/evaluation/self_supervised/representation_metrics.py, which the self-supervised trainer computes on
the test split at every evaluated checkpoint: the k-nearest neighbour classification of the test images by the
reference training images (the vote weighted by similarity, ties, a test image whose only same-class neighbour is a
reference image), the choice of the reference images (the same whatever the order of the samples, bounded per class),
and the effective rank (orthogonal, rank-one and diverged representations).
"""

import math
import pytest
import random
import torch

from vision_training.evaluation.self_supervised.representation_metrics import (compute_effective_rank,
                                                                               compute_nearest_neighbour_predictions,
                                                                               select_reference_samples)


def test_nearest_neighbour_vote_is_weighted_and_ties_go_to_the_lowest_class() -> None:
    """
    Verifies that the reference images vote with the weight exp(similarity / 0.07), so that one very close reference
    image outweighs two distant ones, and that two classes of equal votes give the class of the lowest index.
    """
    # The test image has one very close reference image of class 1 and two distant reference images of class 0
    query_representations = torch.tensor(data=[[1.0, 0.0]])
    reference_labels = torch.tensor(data=[1, 0, 0])
    reference_representations = torch.tensor(data=[[1.0, 0.05], [0.0, 1.0], [0.0, 1.0]])
    predicted_labels = compute_nearest_neighbour_predictions(
        query_representations=query_representations, reference_representations=reference_representations,
        reference_labels=reference_labels, number_neighbours=3, number_classes=3)
    assert int(predicted_labels[0]) == 1

    # The test image has one reference image of class 2 and one of class 1 at the same similarity, and the number of
    # neighbours is reduced to the 2 reference images
    reference_labels = torch.tensor(data=[2, 1])
    reference_representations = torch.tensor(data=[[0.6, 0.8], [0.6, -0.8]])
    predicted_labels = compute_nearest_neighbour_predictions(
        query_representations=query_representations, reference_representations=reference_representations,
        reference_labels=reference_labels, number_neighbours=20, number_classes=3)
    assert int(predicted_labels[0]) == 1

    # No reference image leaves the test images without neighbours
    with pytest.raises(expected_exception=ValueError, match="at least one reference image"):
        compute_nearest_neighbour_predictions(query_representations=query_representations,
                                              reference_representations=torch.zeros(size=(0, 2)),
                                              reference_labels=torch.zeros(size=(0,), dtype=torch.int64),
                                              number_neighbours=3, number_classes=3)


def test_test_image_is_classified_by_its_only_same_class_reference_image() -> None:
    """
    Verifies that a test image whose only image of the same class is a reference training image is classified
    correctly, as with the held-out images of ImageNet-22k, one per class: the other test images, of other classes and
    closer to it, are never its neighbours.
    """
    # Two test images of classes 0 and 1, closer to each other than to the reference images of their classes
    query_representations = torch.tensor(data=[[1.0, 0.0], [0.95, 0.31]])
    reference_representations = torch.tensor(data=[[1.0, -0.5], [0.7, 0.7]])
    reference_labels = torch.tensor(data=[0, 1])
    predicted_labels = compute_nearest_neighbour_predictions(
        query_representations=query_representations, reference_representations=reference_representations,
        reference_labels=reference_labels, number_neighbours=1, number_classes=2)
    assert predicted_labels.tolist() == [0, 1]


def test_reference_samples_are_the_same_in_any_order_and_bounded_per_class() -> None:
    """
    Verifies the choice of the reference images among the training samples: the same images whatever the order of the
    samples (the dataset shuffles them at every creation), at most the asked number per class, every image of a smaller
    class, and no image without class.
    """
    # Five images of class 0, two of class 1 and three without class
    samples = ([{"identifier": index, "file_name": f"zero/{index}.png", "class_indices": [0]} for index in range(5)] +
               [{"identifier": 10 + index, "file_name": f"one/{index}.png", "class_indices": [1]}
                for index in range(2)] +
               [{"identifier": 20 + index, "file_name": f"none/{index}.png", "class_indices": []}
                for index in range(3)])
    reference_samples = select_reference_samples(samples=samples, number_images_per_class=3)
    reference_identifiers = [sample["identifier"] for sample in reference_samples]

    # Three images of class 0 then the two of class 1, the same after shuffling the samples
    assert [sample["class_indices"][0] for sample in reference_samples] == [0, 0, 0, 1, 1]
    shuffled_samples = list(samples)
    random.Random(x=7).shuffle(x=shuffled_samples)
    assert [sample["identifier"] for sample in select_reference_samples(
        samples=shuffled_samples, number_images_per_class=3)] == reference_identifiers

    # A larger bound keeps every labelled image
    assert len(select_reference_samples(samples=samples, number_images_per_class=100)) == 7


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
