"""
Metrics of the representations learned by the self-supervised task, computed on the test split at every evaluated
checkpoint: the k-nearest neighbour classification of the labelled test images by the other labelled test images
(scored by the classification metrics of evaluation/classification), and the effective rank of the representations,
which falls when they collapse to few directions. Used by the self-supervised trainer.
"""

import torch

from torch.nn import functional as torch_functional

# Temperature of the similarity weights of the neighbours in the vote of the k-nearest neighbour classification, as in
# the evaluations of DINO: a neighbour of cosine similarity s weighs exp(s / temperature)
NEAREST_NEIGHBOUR_TEMPERATURE = 0.07

# Number of test images whose similarities with every other image are computed at once, so that the memory of the
# similarity matrix stays bounded for large test splits
SIMILARITY_CHUNK_SIZE = 1024

# Small value added to the singular value distribution of the effective rank before its logarithm
EFFECTIVE_RANK_EPSILON = 1e-12


def compute_nearest_neighbour_predictions(representations: torch.Tensor, labels: torch.Tensor,
                                          number_neighbours: int, number_classes: int) -> torch.Tensor:
    """
    Classifies every image by its k nearest neighbours among the other images, the evaluation of the representations
    of a self-supervised run without training any classifier: a representation placing the images of a class close
    to each other classifies them well. The representations are L2 normalized and compared by cosine similarity, the
    image itself being left out of its neighbours (leave-one-out), and every neighbour votes for its class with the
    weight exp(similarity / NEAREST_NEIGHBOUR_TEMPERATURE), the class of the largest total weight being predicted.

    Args:
        representations (torch.Tensor): The [N, F] representations of the labelled images, N being at least 2.
        labels (torch.Tensor): The [N] class indices of the images.
        number_neighbours (int): The number k of neighbours voting, reduced to N - 1 for smaller splits.
        number_classes (int): The number of classes K.

    Returns:
        torch.Tensor: The [N] predicted class indices.
    """
    normalized_representations = torch_functional.normalize(input=representations.float(), dim=1)
    number_images = normalized_representations.shape[0]
    number_voting_neighbours = min(number_neighbours, number_images - 1)
    predicted_labels = []

    # Similarities of a chunk of images with every image, the image itself being masked out of its neighbours
    for chunk_start in range(0, number_images, SIMILARITY_CHUNK_SIZE):
        chunk_end = min(chunk_start + SIMILARITY_CHUNK_SIZE, number_images)
        similarities = normalized_representations[chunk_start:chunk_end] @ normalized_representations.T
        chunk_indices = torch.arange(chunk_start, chunk_end, device=similarities.device)
        similarities[chunk_indices - chunk_start, chunk_indices] = float("-inf")

        # Vote of the nearest neighbours, weighted by their similarity
        neighbour_similarities, neighbour_indices = similarities.topk(k=number_voting_neighbours, dim=1)
        neighbour_weights = torch.exp(neighbour_similarities / NEAREST_NEIGHBOUR_TEMPERATURE)
        class_votes = torch.zeros(size=(chunk_end - chunk_start, number_classes), device=similarities.device)
        class_votes.scatter_add_(dim=1, index=labels[neighbour_indices], src=neighbour_weights)
        predicted_labels.append(class_votes.argmax(dim=1))
    return torch.cat(tensors=predicted_labels, dim=0)


def compute_effective_rank(representations: torch.Tensor) -> float:
    """
    Measures in how many directions the representations of the test images spread, the label-free warning of a
    collapse (RankMe, Garrido et al. 2023): the exponential of the entropy of the singular values of the [N, F]
    representation matrix, normalized to sum to 1. Representations spread evenly over r orthogonal directions give r,
    collapsed representations give about 1, the largest value being min(N, F).

    Args:
        representations (torch.Tensor): The [N, F] representations of the test images.

    Returns:
        float: The effective rank.
    """
    singular_values = torch.linalg.svdvals(A=representations.double())
    singular_value_distribution = singular_values / singular_values.sum().clamp(min=EFFECTIVE_RANK_EPSILON)
    entropy = -(singular_value_distribution * torch.log(singular_value_distribution + EFFECTIVE_RANK_EPSILON)).sum()
    return float(torch.exp(entropy))
