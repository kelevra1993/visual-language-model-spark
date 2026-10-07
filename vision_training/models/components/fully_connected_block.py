"""
Builder of the hidden fully connected layers (linear, ReLU, optional dropout layers) shared by the detection heads
after RoIAlign, built without dropout layers to keep their checkpoint names, and by the classification heads after the
global pooling, always built with them so that their checkpoint names do not depend on the dropout rate.
"""

import torch

from torch import nn
from typing import List


def build_fully_connected_block(input_dimension: int, dimensions: List[int], dropout_rate: float,
                                add_dropout_layers: bool, device: torch.device, dtype: torch.dtype) -> nn.Sequential:
    """
    Builds the hidden fully connected layers shared by the task heads: the detection heads (after RoIAlign) and the
    classification heads (after the global pooling). Every layer is a linear layer followed by a ReLU, preceded by a
    dropout layer when they are added. The classification heads always add them, a rate of 0 giving an identity, so
    that the dropout rate can change between two resumptions of a run without renaming the weights. The detection
    heads do not, their block being exactly the sequence of linear and ReLU layers of their checkpoints.

    Args:
        input_dimension (int): The dimension of the flattened input features.
        dimensions (List[int]): The output dimension of every fully connected layer, in order (may be empty).
        dropout_rate (float): The probability of zeroing an input feature before every layer, 0 for no dropout.
        add_dropout_layers (bool): True to add a dropout layer before every linear layer, even at a rate of 0.
        device (torch.device): The device of the parameters.
        dtype (torch.dtype): The precision of the parameters.

    Returns:
        nn.Sequential: The fully connected layers.
    """
    fully_connected_layers = []
    current_dimension = input_dimension

    # Stack every hidden layer, each one taking the output dimension of the previous one
    for next_dimension in dimensions:
        if add_dropout_layers:
            fully_connected_layers.append(nn.Dropout(p=dropout_rate))
        fully_connected_layers.append(nn.Linear(in_features=current_dimension, out_features=next_dimension,
                                                device=device, dtype=dtype))
        fully_connected_layers.append(nn.ReLU(inplace=True))
        current_dimension = next_dimension

    return nn.Sequential(*fully_connected_layers)
