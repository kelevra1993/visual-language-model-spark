"""
Builder of the hidden fully connected layers (linear, optional batch normalization, ReLU, optional dropout layers)
shared by the detection heads after RoIAlign, built without dropout layers nor normalization to keep their checkpoint
names, and by the classification heads after the global pooling, always built with dropout layers so that their
checkpoint names do not depend on the dropout rate, and normalized when
'ClassificationHead.fully_connected_normalization' is on. Also holds the batch normalization of feature vectors those
normalized layers use.
"""

import torch

from torch import nn
from typing import List


class FeatureBatchNormalization(nn.BatchNorm1d):
    """
    Batch normalization of feature vectors of any leading shape, used by the normalized hidden layers of the
    classification heads: every feature is normalized with its mean and variance over all the vectors of the batch,
    whether the head classifies one vector per image ([B, F]) or one vector per frame of a sequence ([B, T, F]), which
    nn.BatchNorm1d would read as T channels. It holds the parameters and buffers of nn.BatchNorm1d, so that the frozen
    components keep its statistics like those of the other batch normalization layers.
    """

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the vectors of the batch, all their leading dimensions being one batch dimension.

        Args:
            input_tensor (torch.Tensor): The [..., F] feature vectors.

        Returns:
            torch.Tensor: The [..., F] normalized feature vectors.
        """
        return super().forward(input=input_tensor.reshape(-1, input_tensor.shape[-1])).reshape(input_tensor.shape)


def build_fully_connected_block(input_dimension: int, dimensions: List[int], dropout_rate: float,
                                add_dropout_layers: bool, batch_normalization: bool,
                                batch_normalization_momentum: float, device: torch.device,
                                dtype: torch.dtype) -> nn.Sequential:
    """
    Builds the hidden fully connected layers shared by the task heads: the detection heads (after RoIAlign) and the
    classification heads (after the global pooling). Every layer is a linear layer followed by a ReLU, preceded by a
    dropout layer when they are added, and with a batch normalization of its outputs between the two when asked, so
    that the activations of the head keep their scale whatever the initialization and the learning rate. The
    classification heads always add the dropout layers, a rate of 0 giving an identity, so that the dropout rate can
    change between two resumptions of a run without renaming the weights. The detection heads do not, their block
    being exactly the sequence of linear and ReLU layers of their checkpoints.

    Args:
        input_dimension (int): The dimension of the flattened input features.
        dimensions (List[int]): The output dimension of every fully connected layer, in order (may be empty).
        dropout_rate (float): The probability of zeroing an input feature before every layer, 0 for no dropout.
        add_dropout_layers (bool): True to add a dropout layer before every linear layer, even at a rate of 0.
        batch_normalization (bool): True to normalize the outputs of every linear layer before its ReLU.
        batch_normalization_momentum (float): The momentum of the running statistics of these normalization layers
            (PyTorch convention, 0.1 by default in PyTorch), unused without normalization.
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

        # Normalize the outputs of the linear layer before the activation, as in the convolution blocks
        if batch_normalization:
            fully_connected_layers.append(FeatureBatchNormalization(num_features=next_dimension,
                                                                    momentum=batch_normalization_momentum,
                                                                    device=device, dtype=dtype))
        fully_connected_layers.append(nn.ReLU(inplace=True))
        current_dimension = next_dimension

    return nn.Sequential(*fully_connected_layers)
