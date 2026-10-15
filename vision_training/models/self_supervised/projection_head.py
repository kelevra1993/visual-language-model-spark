"""
Projection head of the self-supervised task: the fully connected layers mapping the pooled representation of the
backbone to the embedding compared by the self-supervised losses, and its summary panel. The head only serves the
self-supervised training: the backbone, not the head, initialises the other tasks. Used by the self-supervised model.
"""

import torch

from rich.console import Console
from rich.panel import Panel
from torch import nn
from typing import Any, Dict, List

from vision_training.models.components.fully_connected_block import build_fully_connected_block


def print_projection_head_summary(head_configuration: Dict[str, Any], input_features: int) -> None:
    """
    Prints the projection head settings in the architecture summary of the self-supervised model at the start of a
    run: pooling, size of the representation, layers with the embedding size last, normalization of the hidden layers
    (with its momentum for the batch normalization) and the self-supervised method with its setting.

    Args:
        head_configuration (Dict[str, Any]): The 'ProjectionHead' section.
        input_features (int): The size of the pooled representation given to the head.
    """
    # The momentum only applies to the batch normalization of the hidden layers
    normalization_text = head_configuration["fully_connected_normalization"]
    if normalization_text == "batch":
        normalization_text += f" (momentum {head_configuration['batch_normalization_momentum']})"

    # The setting of the method of the run
    method = head_configuration["method"]
    if method == "simclr":
        method_text = f"simclr (temperature {head_configuration['simclr']['temperature']})"
    else:
        method_text = f"barlow_twins (off-diagonal weight {head_configuration['barlow_twins']['off_diagonal_weight']})"
    summary_text = (f"[bold]Global Pooling[/bold]       : {head_configuration['global_pooling']}\n"
                    f"[bold]Representation Size[/bold]  : {input_features}\n"
                    f"[bold]Layers[/bold]               : {head_configuration['fully_connected']}\n"
                    f"[bold]Normalization[/bold]        : {normalization_text}\n"
                    f"[bold]Embedding Size[/bold]       : {head_configuration['fully_connected'][-1]}\n"
                    f"[bold]Method[/bold]               : {method_text}")
    Console(width=80).print(Panel(summary_text, title="[bold]Projection Head[/bold]", border_style="green"))


class ProjectionHead(nn.Module):
    """
    Projection head of the self-supervised model: hidden fully connected layers, each one linear, optionally batch or
    layer normalized, then ReLU, followed by the linear layer giving the embedding, without normalization nor
    activation, as in SimCLR (one hidden layer) and Barlow Twins (two hidden layers). The losses are computed on the
    embeddings, so that the representation before the head keeps the information the augmentations remove.
    """

    def __init__(self, input_features: int, fully_connected: List[int], fully_connected_normalization: str,
                 batch_normalization_momentum: float, device: torch.device, dtype: torch.dtype) -> None:
        """
        Builds the hidden layers and the embedding layer.

        Args:
            input_features (int): The size of the pooled representation.
            fully_connected (List[int]): The units of every layer, the last one being the embedding size.
            fully_connected_normalization (str): The normalization of the outputs of every hidden layer before its
                ReLU: "batch", "layer" or "none".
            batch_normalization_momentum (float): The momentum of the running statistics of the batch normalization
                layers (PyTorch convention), unused by the other normalizations.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(ProjectionHead, self).__init__()
        self.input_features = input_features
        self.embedding_size = fully_connected[-1]
        hidden_layers = fully_connected[:-1]

        # Hidden layers without dropout, then the linear layer giving the embedding
        self.fully_connected_block = build_fully_connected_block(
            input_dimension=input_features, dimensions=hidden_layers, dropout_rate=0.0, add_dropout_layers=False,
            normalization=fully_connected_normalization, batch_normalization_momentum=batch_normalization_momentum,
            device=device, dtype=dtype)
        self.embedding_layer = nn.Linear(in_features=hidden_layers[-1] if hidden_layers else input_features,
                                         out_features=self.embedding_size, device=device, dtype=dtype)

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        """
        Projects the pooled representations to the embeddings compared by the self-supervised losses.

        Args:
            representations (torch.Tensor): The [N, input_features] pooled representations.

        Returns:
            torch.Tensor: The [N, embedding_size] embeddings.
        """
        return self.embedding_layer(input=self.fully_connected_block(input=representations))
