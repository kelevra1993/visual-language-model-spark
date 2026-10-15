"""
Model of the self-supervised task: SelfSupervisedModel chains the shared Backbone, a global pooling (or the flattening
of the feature map) giving the representation of every image, and the projection head giving its embedding, then
computes the loss of the self-supervised method (SimCLR or Barlow Twins) on the embeddings of the two views of every
image. Registered in model_registry.py for the "self_supervised" task, wrapped by inference_encoder.py for the exports,
and its backbone initialises the other tasks through 'Backbone.pretrained'.
"""

import torch

from torch import nn
from typing import Any, Dict, Optional

from vision_training.losses.self_supervised_losses import (compute_barlow_twins_loss,
                                                           compute_embedding_standard_deviation,
                                                           compute_info_nce_loss)
from vision_training.models.backbones.backbone import build_backbone
from vision_training.models.classification.classification_head import build_global_pooling, compute_pooled_features
from vision_training.models.self_supervised.projection_head import ProjectionHead, print_projection_head_summary
from vision_training.models.task_model import TaskModel


class SelfSupervisedModel(TaskModel):
    """
    Model of the self-supervised task: the shared Backbone extracts the final feature map of an image, a global
    pooling reduces it to the representation of the image (or the map is flattened, keeping where the features are),
    and the projection head maps the representation to the embedding compared by the loss of
    'ProjectionHead.method'. Both views of a batch go through the network in one forward pass, so that the batch
    normalization layers compute their statistics over all the views.

    Outputs: 'representations' [B, F] and 'embeddings' [B, D] of the images; with second views, the 'embeddings' of
    the first views, the loss of the method ('info_nce_loss' or 'barlow_twins_loss'), its batch metrics and the
    'embedding_standard_deviation' watching the collapse.
    """

    def __init__(self, configuration: Dict[str, Any], verbose: bool = False, device: torch.device = None,
                 dtype: torch.dtype = None) -> None:
        """
        Builds the backbone, the global pooling and the projection head from the configuration.

        Args:
            configuration (Dict[str, Any]): The model configuration (Data, Backbone and ProjectionHead sections).
            verbose (bool): Unused, kept so that every task model is built with the same arguments.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(SelfSupervisedModel, self).__init__()
        self.device = device
        self.dtype = dtype
        self.head_configuration = configuration["ProjectionHead"]
        self.method = self.head_configuration["method"]

        # Shared feature extractor, whose final feature map gives the size of the representation: its channels for the
        # poolings, its channels times its height and width when flattened
        self.backbone = build_backbone(backbone_configuration=configuration["Backbone"],
                                       input_image_size=configuration["Data"]["image_settings"]["size"],
                                       device=device, dtype=dtype)
        self.representation_size = compute_pooled_features(global_pooling=self.head_configuration["global_pooling"],
                                                           backbone=self.backbone)

        # Global pooling (or flattening) of the final feature map, then the embedding
        self.global_pooling = build_global_pooling(global_pooling=self.head_configuration["global_pooling"])
        self.projection_head = ProjectionHead(input_features=self.representation_size,
                                              fully_connected=self.head_configuration["fully_connected"],
                                              fully_connected_normalization=self.head_configuration[
                                                  "fully_connected_normalization"],
                                              batch_normalization_momentum=self.head_configuration[
                                                  "batch_normalization_momentum"], device=device, dtype=dtype)

    def encode(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes the representation and the embedding of every image: the pooled final feature map of the backbone,
        used by the evaluation and the other tasks, and its projection, used by the self-supervised losses.

        Args:
            images (torch.Tensor): The [N, 3, H, W] float BGR images in [0, 255].

        Returns:
            Dict[str, torch.Tensor]: The [N, F] 'representations' and [N, D] 'embeddings'.
        """
        final_feature_map, _ = self.backbone(input_tensor=images)
        representations = self.global_pooling(input=final_feature_map)
        return {"representations": representations, "embeddings": self.projection_head(representations=representations)}

    def compute_method_loss(self, first_embeddings: torch.Tensor,
                            second_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes the loss of the self-supervised method of the run on the embeddings of both views, with its batch
        metrics: InfoNCE and the positive pair accuracy for SimCLR, the Barlow Twins loss and the mean diagonal and
        off-diagonal cross-correlations for Barlow Twins.

        Args:
            first_embeddings (torch.Tensor): The [N, D] embeddings of the first views.
            second_embeddings (torch.Tensor): The [N, D] embeddings of the second views.

        Returns:
            Dict[str, torch.Tensor]: The loss of the method, its key ending in '_loss', and its batch metrics.
        """
        if self.method == "simclr":
            return compute_info_nce_loss(first_embeddings=first_embeddings, second_embeddings=second_embeddings,
                                         temperature=self.head_configuration["simclr"]["temperature"])
        return compute_barlow_twins_loss(first_embeddings=first_embeddings, second_embeddings=second_embeddings,
                                         off_diagonal_weight=self.head_configuration["barlow_twins"][
                                             "off_diagonal_weight"])

    def forward(self, input_tensor: torch.Tensor, second_view_tensor: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Encodes a batch of images, or the two views of a batch of images with the loss of the self-supervised method.
        The two views go through the backbone together, as one batch of 2N images.

        Args:
            input_tensor (torch.Tensor): The [N, 3, H, W] float BGR images in [0, 255], the first views with second
                views.
            second_view_tensor (Optional[torch.Tensor]): The [N, 3, H, W] second views of the same images, in the order
                of the first views, or None to only encode the images.

        Returns:
            Dict[str, Any]: The 'representations' and 'embeddings' of the images without second views; with them, the
                'embeddings' of the first views, the loss of the method, its batch metrics and the
                'embedding_standard_deviation'.
        """
        if second_view_tensor is None:
            return self.encode(images=input_tensor)

        # Both views in one forward pass, then split back into the first and second views
        number_images = input_tensor.shape[0]
        encoded_views = self.encode(images=torch.cat(tensors=[input_tensor, second_view_tensor], dim=0))
        first_embeddings = encoded_views["embeddings"][:number_images]
        second_embeddings = encoded_views["embeddings"][number_images:]

        # Loss and metrics of the method, and the spread of the embeddings watching the collapse
        outputs = {"embeddings": first_embeddings}
        outputs.update(self.compute_method_loss(first_embeddings=first_embeddings,
                                                second_embeddings=second_embeddings))
        outputs["embedding_standard_deviation"] = compute_embedding_standard_deviation(embeddings=first_embeddings)
        return outputs

    def print_summary(self) -> None:
        """
        Prints the backbone summary, then the projection head: pooling, size of the representation, layers,
        normalization, embedding size and self-supervised method.
        """
        self.backbone.print_summary()
        print_projection_head_summary(head_configuration=self.head_configuration,
                                      input_features=self.representation_size)

    def get_size_components(self) -> Dict[str, nn.Module]:
        """
        Lists the backbone, the projection head and the whole model for the model size summary.

        Returns:
            Dict[str, nn.Module]: The display name and module of every component.
        """
        return {"Backbone": self.backbone, "Projection Head": self.projection_head, "Total Model": self}
