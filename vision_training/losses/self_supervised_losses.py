"""
Losses of the self-supervised task, comparing the embeddings of the two random views of every image of a batch:
InfoNCE (SimCLR), where every view must find the view of its own image among all the views of the batch, and Barlow
Twins, where the cross-correlation of the embedding features of both views must be the identity. Each loss comes with
the batch metrics tracked by the trainer, and the standard deviation of the normalized embeddings watches the collapse
of every method. Computed by the self-supervised model in float32, outside the bfloat16 autocast of mixed precision.
"""

import torch

from torch.nn import functional as torch_functional
from typing import Dict

# Self-supervised methods of 'ProjectionHead.method'
SELF_SUPERVISED_METHODS = ["simclr", "barlow_twins"]

# Small value added to the variance of every embedding feature before the normalization of Barlow Twins, so that a
# constant feature does not divide by zero
BARLOW_TWINS_NORMALIZATION_EPSILON = 1e-5


def compute_info_nce_loss(first_embeddings: torch.Tensor, second_embeddings: torch.Tensor,
                          temperature: float) -> Dict[str, torch.Tensor]:
    """
    Computes the InfoNCE loss of SimCLR on the embeddings of the two views of a batch of N images, the loss pulling
    the views of the same image together and pushing the views of different images apart. The 2N embeddings are L2
    normalized and compared by cosine similarity, divided by the temperature: every view i is a classification among
    the 2N - 1 other views (its own similarity being masked), whose right answer is the other view of its image, at
    index i + N for a first view and i - N for a second view. The cross-entropy of these 2N classifications is the
    loss, log(2N - 1) for random embeddings. The share of views whose most similar view is their partner is tracked
    as the positive pair accuracy.

    Args:
        first_embeddings (torch.Tensor): The [N, D] embeddings of the first views.
        second_embeddings (torch.Tensor): The [N, D] embeddings of the second views, in the order of the first ones.
        temperature (float): The temperature dividing the cosine similarities, e.g. 0.1.

    Returns:
        Dict[str, torch.Tensor]: The scalar 'info_nce_loss', and the 'positive_pair_accuracy' without gradient.
    """
    with torch.autocast(device_type=first_embeddings.device.type, enabled=False):
        number_images = first_embeddings.shape[0]

        # Cosine similarity of every pair of the 2N views, divided by the temperature
        embeddings = torch_functional.normalize(input=torch.cat(tensors=[first_embeddings, second_embeddings],
                                                                dim=0).float(), dim=1)
        similarity_logits = embeddings @ embeddings.T / temperature

        # A view never chooses itself: its own similarity is masked out of its classification
        self_similarity_mask = torch.eye(n=2 * number_images, dtype=torch.bool, device=embeddings.device)
        similarity_logits = similarity_logits.masked_fill(mask=self_similarity_mask, value=float("-inf"))

        # The partner of the first view i is the second view at i + N, and the partner of the second view i + N is
        # the first view at i
        view_indices = torch.arange(end=number_images, device=embeddings.device)
        partner_indices = torch.cat(tensors=[view_indices + number_images, view_indices], dim=0)
        info_nce_loss = torch_functional.cross_entropy(input=similarity_logits, target=partner_indices)

        # Share of the views whose most similar view is their partner
        with torch.no_grad():
            positive_pair_accuracy = (similarity_logits.argmax(dim=1) == partner_indices).float().mean()
    return {"info_nce_loss": info_nce_loss, "positive_pair_accuracy": positive_pair_accuracy}


def compute_barlow_twins_loss(first_embeddings: torch.Tensor, second_embeddings: torch.Tensor,
                              off_diagonal_weight: float) -> Dict[str, torch.Tensor]:
    """
    Computes the Barlow Twins loss on the embeddings of the two views of a batch of N images. Every embedding feature
    is normalized over the batch (zero mean, unit variance), then the [D, D] cross-correlation of the features of the
    first views with those of the second views is computed. Its diagonal is pushed to 1, so that every feature takes
    the same value on both views of an image (invariance), and its off-diagonal terms to 0, weighted by the
    off-diagonal weight, so that the features carry different information (redundancy reduction), which also prevents
    the collapse to a constant embedding. The mean diagonal and the mean absolute off-diagonal correlations are tracked.

    Args:
        first_embeddings (torch.Tensor): The [N, D] embeddings of the first views.
        second_embeddings (torch.Tensor): The [N, D] embeddings of the second views, in the order of the first ones.
        off_diagonal_weight (float): The weight of the off-diagonal terms against the diagonal ones, e.g. 0.005.

    Returns:
        Dict[str, torch.Tensor]: The scalar 'barlow_twins_loss', and the 'cross_correlation_diagonal' and
            'cross_correlation_off_diagonal' without gradient.
    """
    with torch.autocast(device_type=first_embeddings.device.type, enabled=False):
        number_images, embedding_dimension = first_embeddings.shape

        # Every feature normalized over the views of the batch
        normalized_views = []
        for view_embeddings in [first_embeddings.float(), second_embeddings.float()]:
            feature_means = view_embeddings.mean(dim=0, keepdim=True)
            feature_variances = view_embeddings.var(dim=0, unbiased=False, keepdim=True)
            normalized_views.append((view_embeddings - feature_means) /
                                    torch.sqrt(input=feature_variances + BARLOW_TWINS_NORMALIZATION_EPSILON))

        # Cross-correlation of the features of the first views with those of the second views
        cross_correlation = normalized_views[0].T @ normalized_views[1] / number_images
        diagonal = torch.diagonal(input=cross_correlation)
        off_diagonal = cross_correlation - torch.diag_embed(input=diagonal)

        # Invariance term on the diagonal, redundancy reduction term off the diagonal
        invariance_term = (diagonal - 1.0).pow(2).sum()
        redundancy_term = off_diagonal.pow(2).sum()
        barlow_twins_loss = invariance_term + off_diagonal_weight * redundancy_term

        # Mean correlation of every feature with itself, and mean absolute correlation of two different features
        with torch.no_grad():
            number_off_diagonal_terms = max(embedding_dimension * (embedding_dimension - 1), 1)
            cross_correlation_diagonal = diagonal.mean()
            cross_correlation_off_diagonal = off_diagonal.abs().sum() / number_off_diagonal_terms
    return {"barlow_twins_loss": barlow_twins_loss, "cross_correlation_diagonal": cross_correlation_diagonal,
            "cross_correlation_off_diagonal": cross_correlation_off_diagonal}


def compute_embedding_standard_deviation(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Measures how spread the embeddings of a batch are, the warning sign of a collapse of the self-supervised training:
    the standard deviation over the batch of every feature of the L2 normalized embeddings, averaged over the features
    and multiplied by the square root of the dimension D. Embeddings spread over the whole sphere give about 1, and
    collapsed embeddings, all equal whatever the image, give 0.

    Args:
        embeddings (torch.Tensor): The [N, D] embeddings of a batch.

    Returns:
        torch.Tensor: The scalar standard deviation, without gradient.
    """
    with torch.no_grad(), torch.autocast(device_type=embeddings.device.type, enabled=False):
        normalized_embeddings = torch_functional.normalize(input=embeddings.float(), dim=1)
        feature_standard_deviations = normalized_embeddings.std(dim=0, unbiased=False)
        return feature_standard_deviations.mean() * normalized_embeddings.shape[1] ** 0.5
