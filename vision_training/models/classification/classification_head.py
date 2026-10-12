"""
Classification head of the classification-like tasks: the global pooling (or the flattening, for the image classifier)
turning the last backbone feature map into one vector, the fully connected head giving the class logits and its
summary panel. Used by the image classifier and by the sequence classifier, labeller and comparator.
"""

import torch

from rich.console import Console
from rich.panel import Panel
from torch import nn
from typing import Any, Dict, List, Optional

from vision_training.models.backbones.backbone import Backbone
from vision_training.models.components.fully_connected_block import build_fully_connected_block

# Global pooling layer of the 'ClassificationHead.global_pooling' settings reducing a feature map to one value per
# channel, the "flatten" setting keeping every value of the feature map instead
GLOBAL_POOLING_LAYERS = {"average": nn.AdaptiveAvgPool2d, "max": nn.AdaptiveMaxPool2d}


def build_global_pooling(global_pooling: str) -> nn.Sequential:
    """
    Builds the global pooling turning the final feature map of the backbone into one feature vector per image, the
    input of the classification head of the image classifier and of the frame encoder of the sequence models: its
    average or maximum over the positions, or for the image classifier all its values flattened, which keeps where the
    features are in the image.

    Args:
        global_pooling (str): "average", "max" or "flatten".

    Returns:
        nn.Sequential: The pooling, mapping [B, C, H, W] feature maps to [B, C] vectors, or to [B, C x H x W] vectors
            when flattening.
    """
    if global_pooling == "flatten":
        return nn.Sequential(nn.Flatten(start_dim=1))
    return nn.Sequential(GLOBAL_POOLING_LAYERS[global_pooling](output_size=1), nn.Flatten(start_dim=1))


def compute_pooled_features(global_pooling: str, backbone: Backbone) -> int:
    """
    Gives the size of the vector the global pooling makes from the final feature map of the backbone, the input size
    of the classification head of the image classifier: the channels of the map for the average and max poolings,
    whatever the image size, and its channels times its height and width when flattening, read from the output size
    of the last spatial layer of the backbone (Data.image_settings.size halved by every max pooling).

    Args:
        global_pooling (str): "average", "max" or "flatten".
        backbone (Backbone): The backbone, built for the input image size of the run.

    Returns:
        int: The number of features of the pooled vector.
    """
    final_layer = backbone.compute_receptive_field_information()[-1]
    if global_pooling == "flatten":
        return final_layer["output_channels"] * final_layer["output_size"] * final_layer["output_size"]
    return final_layer["output_channels"]


def print_classification_head_summary(head_configuration: Dict[str, Any], input_features: int,
                                      number_classes: int) -> None:
    """
    Prints the classification head settings in the architecture summary of the classification and sequence models:
    pooling, size of the vectors the head classifies (the pooled or flattened feature map of the image classifier,
    the output of the temporal module or of the fusion of the sequence models), hidden layers, their batch
    normalization and its momentum, dropout, output mode and number of classes.

    Args:
        head_configuration (Dict[str, Any]): The 'ClassificationHead' section.
        input_features (int): The number of features of the vectors given to the head.
        number_classes (int): The number of predicted classes.
    """
    # The momentum only applies to the normalized hidden layers
    normalization_text = (f"batch (momentum {head_configuration['batch_normalization_momentum']})"
                          if head_configuration["fully_connected_normalization"] else "none")
    summary_text = (f"[bold]Global Pooling[/bold]     : {head_configuration['global_pooling']}\n"
                    f"[bold]Input Features[/bold]     : {input_features}\n"
                    f"[bold]Hidden Layers[/bold]      : {head_configuration['fully_connected']}\n"
                    f"[bold]Normalization[/bold]      : {normalization_text}\n"
                    f"[bold]Dropout Rate[/bold]       : {head_configuration['dropout_rate']}\n"
                    f"[bold]Output Mode[/bold]        : {head_configuration['output_mode']}\n"
                    f"[bold]Number Of Classes[/bold]  : {number_classes}")
    Console(width=80).print(Panel(summary_text, title="[bold]Classification Head[/bold]", border_style="green"))


class ClassificationHead(nn.Module):
    """
    Classification head shared by the classification-like tasks: hidden fully connected layers, optionally batch
    normalized, followed by the linear layer giving one logit per class. It applies to the last dimension of its input,
    so the same head classifies one vector per image ([B, F]), one vector per sequence, or one vector per frame of a
    sequence ([B, T, F]).
    """

    def __init__(self, input_features: int, fully_connected: List[int], fully_connected_normalization: bool,
                 batch_normalization_momentum: float, dropout_rate: float, number_classes: int, device: torch.device,
                 dtype: torch.dtype) -> None:
        """
        Builds the hidden layers and the class logits layer.

        Args:
            input_features (int): The dimension of the input feature vectors.
            fully_connected (List[int]): The units of every hidden layer, possibly none.
            fully_connected_normalization (bool): Whether every hidden layer normalizes its outputs with batch
                normalization before its ReLU.
            batch_normalization_momentum (float): The momentum of the running statistics of these normalization layers
                (PyTorch convention), unused without normalization.
            dropout_rate (float): The dropout applied before every fully connected layer, 0 for none.
            number_classes (int): The number of predicted classes.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(ClassificationHead, self).__init__()
        self.input_features = input_features
        self.fully_connected = fully_connected
        self.fully_connected_normalization = fully_connected_normalization
        self.dropout_rate = dropout_rate

        # Hidden layers, then a dropout and the linear layer giving one logit per class. The dropout layers are built
        # even at a rate of 0, so that the names of the weights do not depend on the dropout rate
        self.fully_connected_block = build_fully_connected_block(
            input_dimension=input_features, dimensions=fully_connected, dropout_rate=dropout_rate,
            add_dropout_layers=True, batch_normalization=fully_connected_normalization,
            batch_normalization_momentum=batch_normalization_momentum, device=device, dtype=dtype)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(in_features=fully_connected[-1] if fully_connected else input_features,
                                    out_features=number_classes, device=device, dtype=dtype)

    def compute_logits(self, features: torch.Tensor) -> torch.Tensor:
        """
        Runs the hidden layers, the dropout and the class logits layer on feature vectors, the computation of the head
        once the vectors it classifies are chosen.

        Args:
            features (torch.Tensor): The [..., input_features] feature vectors.

        Returns:
            torch.Tensor: The [..., number_classes] class logits.
        """
        return self.classifier(input=self.dropout(input=self.fully_connected_block(input=features)))

    def forward(self, features: torch.Tensor, feature_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the class logits of feature vectors. With normalized hidden layers and a mask (the padding frames of
        the windows of the sequence labeller), only the real vectors go through the head, so that the padding frames
        take no part in the batch statistics, and the padding frames get zero logits, ignored by the loss and the
        evaluation like their labels.

        Args:
            features (torch.Tensor): The [..., input_features] feature vectors.
            feature_mask (Optional[torch.Tensor]): The [...] booleans of the leading dimensions of the features, True
                for the real vectors, or None when every vector is real.

        Returns:
            torch.Tensor: The [..., number_classes] class logits.
        """
        # Every vector goes through the head when no batch statistics are computed or when every vector is real
        if not self.fully_connected_normalization or feature_mask is None:
            return self.compute_logits(features=features)

        # Classify the real vectors only, then place their logits back among the zero logits of the padding vectors
        real_logits = self.compute_logits(features=features[feature_mask])
        class_logits = real_logits.new_zeros(size=(*feature_mask.shape, real_logits.shape[-1]))
        class_logits[feature_mask] = real_logits
        return class_logits
