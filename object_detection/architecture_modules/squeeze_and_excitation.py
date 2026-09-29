import torch
from torch import nn


class SqueezeAndExcitation(nn.Module):
    """
    Applies the Squeeze-and-Excitation channel recalibration from "Squeeze and Excitation Networks"
    (https://arxiv.org/pdf/1709.01507.pdf).

    Within the object detection pipeline, this module is appended after each convolution layer of a
    ConvolutionBlock (backbone or detection head). It summarises every channel with a global average,
    learns a per-channel importance weight from that summary, and rescales the feature map so that the
    channels most useful to the downstream Region Proposal and Detection heads are emphasised.
    """

    def __init__(self, channels: int, device: torch.device, dtype: torch.dtype, reduction_ratio: int = 2) -> None:
        """
        Initializes the two fully connected recalibration layers of the excitation phase.

        Args:
            channels (int): The number of channels of the feature map to recalibrate.
            device (torch.device): The device on which to allocate the parameters.
            dtype (torch.dtype): The desired data type of the parameters.
            reduction_ratio (int): The bottleneck reduction applied between the two recalibration layers.
                Defaults to 2, halving the channels in the bottleneck.
        """
        super(SqueezeAndExcitation, self).__init__()

        # Bottleneck dimension of the excitation phase, kept at a minimum of one neuron for very thin feature maps
        reduced_channels = max(1, channels // reduction_ratio)

        # The recalibrators have no bias, so the channel weights depend only on the pooled channel statistics
        self.first_adaptive_recalibrator = nn.Linear(in_features=channels, out_features=reduced_channels, bias=False,
                                                     device=device, dtype=dtype)
        self.second_adaptive_recalibrator = nn.Linear(in_features=reduced_channels, out_features=channels, bias=False,
                                                      device=device, dtype=dtype)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Rescales each channel of the feature map by its learned importance weight.

        Args:
            input_tensor (torch.Tensor): The feature map of shape [Batch, Channels, Height, Width].

        Returns:
            torch.Tensor: The recalibrated feature map of shape [Batch, Channels, Height, Width].
        """
        # Squeeze phase: summarise every channel with its global spatial average, shape [Batch, Channels]
        squeezed_tensor = input_tensor.mean(dim=(2, 3))

        # Excitation phase: bottleneck with ReLU, then sigmoid to obtain per-channel weights in (0, 1)
        excitation = torch.relu(input=self.first_adaptive_recalibrator(input=squeezed_tensor))
        excitation = torch.sigmoid(input=self.second_adaptive_recalibrator(input=excitation))

        # Broadcast the channel weights over the spatial dimensions to rescale the feature map
        return input_tensor * excitation.unsqueeze(dim=-1).unsqueeze(dim=-1)
