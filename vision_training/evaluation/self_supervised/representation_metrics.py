"""
Metrics of the representations learned by the self-supervised task, computed on the test split at every evaluated
checkpoint: the k-nearest neighbour classification of the labelled test images by a fixed sample of labelled training
images, the reference images (scored by the classification metrics of evaluation/classification), the selection of
those reference images, and the effective rank of the representations, which falls when they collapse to few
directions. Used by the self-supervised trainer.
"""

import hashlib
import torch

from collections import defaultdict
from torch.nn import functional as torch_functional
from typing import Any, Dict, List

# Temperature of the similarity weights of the neighbours in the vote of the k-nearest neighbour classification, as in
# the evaluations of DINO: a neighbour of cosine similarity s weighs exp(s / temperature)
NEAREST_NEIGHBOUR_TEMPERATURE = 0.07

# Number of test images whose similarities with every reference image are computed at once, so that the memory of the
# similarity matrix stays bounded for large splits
SIMILARITY_CHUNK_SIZE = 1024

# Small value added to the singular value distribution of the effective rank before its logarithm
EFFECTIVE_RANK_EPSILON = 1e-12


def select_reference_samples(samples: List[Dict[str, Any]], number_images_per_class: int) -> List[Dict[str, Any]]:
    """
    Chooses the reference images of the k-nearest neighbour evaluation of the self-supervised task among the labelled
    training images: in every class, the number_images_per_class images whose file names have the smallest MD5 hashes,
    so that the same images are chosen at every evaluation, resume and checkpoint evaluation whatever the order of the
    samples (the dataset shuffles them), and the time and memory of their encoding stay bounded on a large training
    split (14M images of ImageNet-22k). A class with fewer labelled training images gives all of them, and the images
    without class are left out.

    Args:
        samples (List[Dict[str, Any]]): The sample records of the training split, each with its 'file_name' and its
            'class_indices' (one class index, or none for an image without class).
        number_images_per_class (int): The largest number of reference images of a class.

    Returns:
        List[Dict[str, Any]]: The sample records of the reference images, by class index then hash.
    """
    # Labelled training images of every class, the images without class left out
    class_samples = defaultdict(list)
    for sample in samples:
        if len(sample["class_indices"]) == 1:
            class_samples[sample["class_indices"][0]].append(sample)

    # Images of the smallest hashes of their file names, the file name breaking a tie of hashes
    reference_samples = []
    for class_index in sorted(class_samples):
        ordered_samples = sorted(class_samples[class_index], key=lambda sample: (
            hashlib.md5(string=sample["file_name"].encode(encoding="utf-8")).hexdigest(), sample["file_name"]))
        reference_samples.extend(ordered_samples[:number_images_per_class])
    return reference_samples


def compute_nearest_neighbour_predictions(query_representations: torch.Tensor,
                                          reference_representations: torch.Tensor, reference_labels: torch.Tensor,
                                          number_neighbours: int, number_classes: int) -> torch.Tensor:
    """
    Classifies every query image (a labelled test image) by its k nearest neighbours among the reference images (a
    fixed sample of labelled training images), the evaluation of the representations of a self-supervised run without
    training any classifier: a representation placing the images of a class close to each other classifies them well.
    A test image is never compared with the other test images, so that a test split holding one image per class (the
    held-out images of ImageNet-22k) is scored like any other. The representations are L2 normalized and compared by
    cosine similarity, and every neighbour votes for its class with the weight
    exp(similarity / NEAREST_NEIGHBOUR_TEMPERATURE), the class of the largest total weight being predicted, the lowest
    class index on a tie.

    Args:
        query_representations (torch.Tensor): The [N, F] representations of the classified images.
        reference_representations (torch.Tensor): The [M, F] representations of the reference images, M at least 1.
        reference_labels (torch.Tensor): The [M] class indices of the reference images.
        number_neighbours (int): The number k of neighbours voting, reduced to M for a smaller reference sample.
        number_classes (int): The number of classes K.

    Returns:
        torch.Tensor: The [N] predicted class indices.

    Raises:
        ValueError: If there is no reference image, which would leave every query image without a neighbour.
    """
    if reference_representations.shape[0] == 0:
        raise ValueError("The k-nearest neighbour classification needs at least one reference image, got none.")
    normalized_queries = torch_functional.normalize(input=query_representations.float(), dim=1)
    normalized_references = torch_functional.normalize(input=reference_representations.float(), dim=1)
    number_queries = normalized_queries.shape[0]
    number_voting_neighbours = min(number_neighbours, normalized_references.shape[0])
    predicted_labels = [torch.zeros(size=(0,), dtype=torch.int64, device=normalized_queries.device)]

    # Similarities of a chunk of query images with every reference image
    for chunk_start in range(0, number_queries, SIMILARITY_CHUNK_SIZE):
        chunk_end = min(chunk_start + SIMILARITY_CHUNK_SIZE, number_queries)
        similarities = normalized_queries[chunk_start:chunk_end] @ normalized_references.T

        # Vote of the nearest neighbours, weighted by their similarity
        neighbour_similarities, neighbour_indices = similarities.topk(k=number_voting_neighbours, dim=1)
        neighbour_weights = torch.exp(input=neighbour_similarities / NEAREST_NEIGHBOUR_TEMPERATURE)
        class_votes = torch.zeros(size=(chunk_end - chunk_start, number_classes), device=similarities.device)
        class_votes.scatter_add_(dim=1, index=reference_labels[neighbour_indices], src=neighbour_weights)
        predicted_labels.append(class_votes.argmax(dim=1))
    return torch.cat(tensors=predicted_labels, dim=0)


def compute_effective_rank(representations: torch.Tensor) -> float:
    """
    Measures in how many directions the representations of the test images spread, the label-free warning of a
    collapse (RankMe, Garrido et al. 2023): the exponential of the entropy of the singular values of the [N, F]
    representation matrix, normalized to sum to 1. Representations spread evenly over r orthogonal directions give r,
    collapsed representations give about 1, the largest value being min(N, F). The representations of a backbone ending
    with a ReLU are not negative, so their mean direction dominates the first singular value before the training
    spreads them, giving an effective rank close to 1 at the start of a run.

    Args:
        representations (torch.Tensor): The [N, F] representations of the test images.

    Returns:
        float: The effective rank, NaN when a representation is not finite (diverged weights), so that the evaluation
            reports it instead of stopping the run.
    """
    # The singular value decomposition fails on non-finite values
    if not bool(torch.isfinite(input=representations).all()):
        return float("nan")
    singular_values = torch.linalg.svdvals(A=representations.double())
    singular_value_distribution = singular_values / singular_values.sum().clamp(min=EFFECTIVE_RANK_EPSILON)
    entropy = -(singular_value_distribution * torch.log(input=singular_value_distribution +
                                                         EFFECTIVE_RANK_EPSILON)).sum()
    return float(torch.exp(input=entropy))
