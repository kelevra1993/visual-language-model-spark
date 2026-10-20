"""
View losses of the image classifier ('ClassificationHead.view_losses'), added to its classification loss: Barlow Twins
and the contrastive loss (SimCLR, or SupCon with the classes of the images) compare the representations of two views of
every image, the image as the classifier sees it and a second random view of it. The representation of a view is the
final feature map of the backbone pooled by the pooling of the view losses ('global_pooling', the average over the
positions by default, whatever the pooling of the classification head), and every loss turned on has its own projector,
the fully connected layers mapping that representation to the embeddings the loss compares. The classification trainer
builds the projectors as auxiliary heads of the general trainer: trained with the classifier but kept outside it, never
exported, transferred nor averaged, and saved in the checkpoints with their own optimizers, so that a loss can be turned
on or off when a run resumes. This module builds the projectors, computes the losses on the projected views and prints
their summary panel at the start of a run.
"""

import torch

from rich.console import Console
from rich.panel import Panel
from torch import nn
from typing import Any, Dict, Optional

from vision_training.losses.self_supervised_losses import compute_barlow_twins_loss, compute_info_nce_loss
from vision_training.models.self_supervised.projection_head import ProjectionHead

# View losses of 'ClassificationHead.view_losses', in the order they are computed and tracked
VIEW_LOSS_NAMES = ["barlow_twins", "contrastive"]

# Positives of the contrastive loss: the other view of the image only (SimCLR), or also every view of an image of the
# same class (SupCon)
CONTRASTIVE_POSITIVES = ["views", "classes"]

# Normalization of the hidden layers of the projectors, the batch normalization of SimCLR and Barlow Twins, with the
# momentum of PyTorch
PROJECTOR_NORMALIZATION = "batch"
PROJECTOR_BATCH_NORMALIZATION_MOMENTUM = 0.1

# Suffix of the name of the projector of a view loss among the auxiliary heads of the trainer, e.g.
# 'barlow_twins_projector', the name under which it is saved in the checkpoints
PROJECTOR_NAME_SUFFIX = "_projector"


def build_view_loss_projectors(view_losses_configuration: Dict[str, Any], input_features: int, device: torch.device,
                               dtype: torch.dtype) -> Dict[str, nn.Module]:
    """
    Builds the projector of every view loss turned on, given to the general trainer as its auxiliary heads by the
    classification trainer: the hidden layers of 'projector', each linear, batch normalized and followed by a ReLU, then
    the linear layer giving the embedding, the last number of 'projector', as the projection heads of SimCLR and Barlow
    Twins. A projector of a single layer is that linear layer alone.

    Args:
        view_losses_configuration (Dict[str, Any]): The 'ClassificationHead.view_losses' section.
        input_features (int): The size of the representation of a view, the final feature map of the backbone pooled
            by the pooling of the view losses.
        device (torch.device): The device of the parameters.
        dtype (torch.dtype): The precision of the parameters.

    Returns:
        Dict[str, nn.Module]: The projector of every view loss turned on, keyed by its name (the loss name followed by
            PROJECTOR_NAME_SUFFIX), empty when every view loss is off.
    """
    projectors = {}
    for loss_name in VIEW_LOSS_NAMES:
        loss_configuration = view_losses_configuration[loss_name]
        if not loss_configuration["activated"]:
            continue

        # Batch normalized hidden layers, none for a projector made of the embedding layer alone
        normalization = PROJECTOR_NORMALIZATION if len(loss_configuration["projector"]) > 1 else "none"
        projectors[f"{loss_name}{PROJECTOR_NAME_SUFFIX}"] = ProjectionHead(
            input_features=input_features, fully_connected=loss_configuration["projector"],
            fully_connected_normalization=normalization,
            batch_normalization_momentum=PROJECTOR_BATCH_NORMALIZATION_MOMENTUM, device=device, dtype=dtype)
    return projectors


def compute_view_losses(first_feature_maps: torch.Tensor, second_feature_maps: torch.Tensor,
                        labels: Optional[torch.Tensor], global_pooling: nn.Module, projectors: Dict[str, nn.Module],
                        view_losses_configuration: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Computes the view losses turned on, which the classification trainer adds to the outputs of the classifier so that
    they are summed with its classification loss into the total loss. The final feature maps of both views are pooled
    into their representations by the pooling of the view losses, then every projector maps the representations of
    both views in one pass, so that its batch normalization computes its statistics over all the views, then Barlow
    Twins pulls the cross-correlation of the embedding features of both views towards the identity
    (compute_barlow_twins_loss) and the contrastive loss makes every view find its positives among all the views of the
    batch (InfoNCE of compute_info_nce_loss): the other view of its image (positives "views", SimCLR), or every view of
    an image of the same class (positives "classes", SupCon). Every loss is multiplied by its weight, the value tracked,
    and comes with the batch metrics of its method. The losses are computed in float32 outside the bfloat16 autocast.

    Args:
        first_feature_maps (torch.Tensor): The [N, C, H, W] final feature maps of the images as the classifier sees
            them.
        second_feature_maps (torch.Tensor): The [N, C, H, W] final feature maps of their second views, in the same
            order.
        labels (Optional[torch.Tensor]): The [N] class indices of the images, read by the contrastive loss with the
            positives "classes" only.
        global_pooling (nn.Module): The pooling of the view losses ('global_pooling', from build_global_pooling),
            turning a final feature map into the representation of a view.
        projectors (Dict[str, nn.Module]): The projector of every view loss turned on, as built by
            build_view_loss_projectors.
        view_losses_configuration (Dict[str, Any]): The 'ClassificationHead.view_losses' section.

    Returns:
        Dict[str, torch.Tensor]: For Barlow Twins, the weighted 'barlow_twins_loss' with the mean
            'cross_correlation_diagonal' and 'cross_correlation_off_diagonal'; for the contrastive loss, the weighted
            'contrastive_loss' with the 'positive_pair_accuracy'. Only the losses turned on are given.
    """
    number_images = first_feature_maps.shape[0]

    # Representations of both views, pooled from their final feature maps in one pass
    representations = global_pooling(input=torch.cat(tensors=[first_feature_maps, second_feature_maps], dim=0))
    outputs = {}
    for loss_name in VIEW_LOSS_NAMES:
        projector_name = f"{loss_name}{PROJECTOR_NAME_SUFFIX}"
        if projector_name not in projectors:
            continue
        loss_configuration = view_losses_configuration[loss_name]

        # Both views projected together, then split back into the first and second views
        embeddings = projectors[projector_name](representations=representations)
        first_embeddings, second_embeddings = embeddings[:number_images], embeddings[number_images:]

        # Barlow Twins: invariance on the diagonal of the cross-correlation, redundancy reduction off the diagonal
        if loss_name == "barlow_twins":
            barlow_twins_outputs = compute_barlow_twins_loss(
                first_embeddings=first_embeddings, second_embeddings=second_embeddings,
                off_diagonal_weight=loss_configuration["off_diagonal_weight"])
            barlow_twins_loss = loss_configuration["weight"] * barlow_twins_outputs["barlow_twins_loss"]
            outputs.update({"barlow_twins_loss": barlow_twins_loss,
                            "cross_correlation_diagonal": barlow_twins_outputs["cross_correlation_diagonal"],
                            "cross_correlation_off_diagonal": barlow_twins_outputs["cross_correlation_off_diagonal"]})
            continue

        # Contrastive loss, the classes of the images giving more positives with the positives "classes"
        contrastive_outputs = compute_info_nce_loss(
            first_embeddings=first_embeddings, second_embeddings=second_embeddings,
            temperature=loss_configuration["temperature"],
            labels=labels if loss_configuration["positives"] == "classes" else None)
        outputs.update({"contrastive_loss": loss_configuration["weight"] * contrastive_outputs["info_nce_loss"],
                        "positive_pair_accuracy": contrastive_outputs["positive_pair_accuracy"]})
    return outputs


def print_view_losses_summary(view_losses_configuration: Dict[str, Any], projectors: Dict[str, nn.Module]) -> None:
    """
    Prints the view losses of a run in the architecture summary at its start, under the classification head, when at
    least one of them is on: the pooling giving the representation of a view with its size, the area range of the crop
    of the second views, and for every loss its weight and setting with the layers of its projector and their number of
    parameters, or 'off'.

    Args:
        view_losses_configuration (Dict[str, Any]): The 'ClassificationHead.view_losses' section.
        projectors (Dict[str, nn.Module]): The projector of every view loss turned on, as built by
            build_view_loss_projectors.
    """
    if not projectors:
        return

    # Pooling and size of the representation of a view, read by every projector, then the area range of the crop of
    # the second views, the transformation most likely to remove the object of the class
    representation_features = next(iter(projectors.values())).input_features
    crop_scale = view_losses_configuration["view_augmentation"]["random_resized_crop"]["scale"]
    summary_lines = [f"[bold]{'Representation':<22}[/bold] : {view_losses_configuration['global_pooling']} pooling, "
                     f"{representation_features:,} features",
                     f"[bold]{'Second View Crop':<22}[/bold] : {crop_scale[0]:g} to {crop_scale[1]:g} of the image "
                     f"area"]

    # Every view loss, with its setting and its projector when it is on
    for loss_name in VIEW_LOSS_NAMES:
        loss_configuration = view_losses_configuration[loss_name]
        loss_title = loss_name.replace("_", " ").title()
        projector_name = f"{loss_name}{PROJECTOR_NAME_SUFFIX}"
        if projector_name not in projectors:
            summary_lines.append(f"[bold]{loss_title:<22}[/bold] : off")
            continue
        if loss_name == "barlow_twins":
            setting_text = f"off-diagonal weight {loss_configuration['off_diagonal_weight']:g}"
        else:
            setting_text = (f"temperature {loss_configuration['temperature']:g}, positives "
                            f"{loss_configuration['positives']}")
        number_parameters = sum(parameter.numel() for parameter in projectors[projector_name].parameters())
        summary_lines.append(f"[bold]{loss_title:<22}[/bold] : weight {loss_configuration['weight']:g}, {setting_text}")
        summary_lines.append(f"[bold]{loss_title + ' Projector':<22}[/bold] : {loss_configuration['projector']}, "
                             f"{number_parameters:,} parameters")
    Console(width=80).print(Panel(renderable="\n".join(summary_lines), title="[bold]View Losses[/bold]",
                                  border_style="green"))
