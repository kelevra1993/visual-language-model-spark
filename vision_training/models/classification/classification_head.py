import torch

from torch import nn
from typing import List

from vision_training.models.components.fully_connected_block import build_fully_connected_block

# Global pooling layer of every 'ClassificationHead.global_pooling' setting, reducing a feature map to one vector
GLOBAL_POOLING_LAYERS = {"average": nn.AdaptiveAvgPool2d, "max": nn.AdaptiveMaxPool2d}


def build_global_pooling(global_pooling: str) -> nn.Sequential:
    """
    Builds the global pooling turning the final feature map of the backbone into one feature vector per image, the
    input of the classification head of the image classifier and of the frame encoder of the sequence models.

    Args:
        global_pooling (str): "average" or "max".

    Returns:
        nn.Sequential: The pooling, mapping [B, C, H, W] feature maps to [B, C] vectors.
    """
    return nn.Sequential(GLOBAL_POOLING_LAYERS[global_pooling](output_size=1), nn.Flatten(start_dim=1))


class ClassificationHead(nn.Module):
    """
    Classification head shared by the classification-like tasks: hidden fully connected layers followed by the linear
    layer giving one logit per class. It applies to the last dimension of its input, so the same head classifies one
    vector per image ([B, F]), one vector per sequence, or one vector per frame of a sequence ([B, T, F]).
    """

    def __init__(self, input_features: int, fully_connected: List[int], dropout_rate: float, number_classes: int,
                 device: torch.device, dtype: torch.dtype) -> None:
        """
        Builds the hidden layers and the class logits layer.

        Args:
            input_features (int): The dimension of the input feature vectors.
            fully_connected (List[int]): The units of every hidden layer, possibly none.
            dropout_rate (float): The dropout applied before every fully connected layer, 0 for none.
            number_classes (int): The number of predicted classes.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(ClassificationHead, self).__init__()
        self.fully_connected = fully_connected
        self.dropout_rate = dropout_rate

        # Hidden layers, then a dropout and the linear layer giving one logit per class
        self.fully_connected_block = build_fully_connected_block(input_dimension=input_features,
                                                                 dimensions=fully_connected, dropout_rate=dropout_rate,
                                                                 device=device, dtype=dtype)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(in_features=fully_connected[-1] if fully_connected else input_features,
                                    out_features=number_classes, device=device, dtype=dtype)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Computes the class logits of feature vectors.

        Args:
            features (torch.Tensor): The [..., input_features] feature vectors.

        Returns:
            torch.Tensor: The [..., number_classes] class logits.
        """
        return self.classifier(self.dropout(self.fully_connected_block(features)))
