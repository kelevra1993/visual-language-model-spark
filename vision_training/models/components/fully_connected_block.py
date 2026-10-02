import torch

from torch import nn
from typing import List


def build_fully_connected_block(input_dimension: int, dimensions: List[int], dropout_rate: float,
                                device: torch.device, dtype: torch.dtype) -> nn.Sequential:
    """
    Builds the hidden fully connected layers shared by the task heads: the detection heads (after RoIAlign) and the
    classification heads (after the global pooling). Every layer is a linear layer followed by a ReLU, preceded by a
    dropout when a rate is given; without dropout, the block is exactly the sequence of linear and ReLU layers, so the
    weights of the detection heads keep their checkpoint names.

    Args:
        input_dimension (int): The dimension of the flattened input features.
        dimensions (List[int]): The output dimension of every fully connected layer, in order (may be empty).
        dropout_rate (float): The probability of zeroing an input feature before every layer, 0 for no dropout.
        device (torch.device): The device of the parameters.
        dtype (torch.dtype): The precision of the parameters.

    Returns:
        nn.Sequential: The fully connected layers.
    """
    fully_connected_layers = []
    current_dimension = input_dimension

    # Stack every hidden layer, each one taking the output dimension of the previous one
    for next_dimension in dimensions:
        if dropout_rate > 0.0:
            fully_connected_layers.append(nn.Dropout(p=dropout_rate))
        fully_connected_layers.append(nn.Linear(in_features=current_dimension, out_features=next_dimension,
                                                device=device, dtype=dtype))
        fully_connected_layers.append(nn.ReLU(inplace=True))
        current_dimension = next_dimension

    return nn.Sequential(*fully_connected_layers)
