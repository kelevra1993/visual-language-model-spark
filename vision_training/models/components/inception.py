import torch
from torch import nn


class InceptionConvolution(nn.Module):
    """
    Implements the Inception module from "Going Deeper with Convolutions" (https://arxiv.org/abs/1409.4842).

    Within the object detection pipeline, this module replaces the plain 3x3 convolution of a ConvolutionBlock
    layer. Parallel branches with different kernel sizes let a single layer capture features at several spatial
    extents, which helps the downstream Region Proposal and Detection heads handle objects of varying sizes.
    The output channels are split across the four branches as follows:
    1x1 (1/4), 1x1 -> 3x3 (1/2), 1x1 -> 5x5 (1/8) and 3x3 max pooling -> 3x3 (1/8).
    """

    def __init__(self, input_channels: int, output_channels: int, bias: bool,
                 device: torch.device, dtype: torch.dtype) -> None:
        """
        Initializes the four parallel convolution branches of the Inception module.

        Args:
            input_channels (int): The number of channels of the input feature map.
            output_channels (int): The total number of channels after concatenating all branches.
                Must be a multiple of 8 so that every branch receives a whole number of channels.
            bias (bool): Whether to include a learnable bias term in the branch convolutions.
            device (torch.device): The device on which to allocate the parameters.
            dtype (torch.dtype): The desired data type of the parameters.

        Raises:
            ValueError: If output_channels is not a multiple of 8.
        """
        super(InceptionConvolution, self).__init__()

        # Guarantee an integer channel split across the branches, since the smallest branches receive 1/8 of them
        if output_channels % 8 != 0:
            raise ValueError(f"Inception modules require output channels that are multiples of 8,"
                             f" but {output_channels} were requested.")

        # Store the channel sizes for downstream introspection (e.g. summaries)
        self.in_channels = input_channels
        self.out_channels = output_channels

        # Direct 1x1 branch capturing point-wise channel interactions
        self.branch_1x1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels // 4, kernel_size=1,
                                    bias=bias, device=device, dtype=dtype)

        # 1x1 channel reduction followed by a 3x3 convolution for medium spatial extents
        self.branch_3x3_reduction = nn.Conv2d(in_channels=input_channels, out_channels=output_channels // 2,
                                              kernel_size=1, bias=bias, device=device, dtype=dtype)
        self.branch_3x3 = nn.Conv2d(in_channels=output_channels // 2, out_channels=output_channels // 2,
                                    kernel_size=3, padding=1, bias=bias, device=device, dtype=dtype)

        # 1x1 channel reduction followed by a 5x5 convolution for larger spatial extents
        self.branch_5x5_reduction = nn.Conv2d(in_channels=input_channels, out_channels=output_channels // 8,
                                              kernel_size=1, bias=bias, device=device, dtype=dtype)
        self.branch_5x5 = nn.Conv2d(in_channels=output_channels // 8, out_channels=output_channels // 8,
                                    kernel_size=5, padding=2, bias=bias, device=device, dtype=dtype)

        # Stride-1 max pooling followed by a 3x3 convolution, keeping the spatial size of the other branches
        self.branch_pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pooling_projection = nn.Conv2d(in_channels=input_channels, out_channels=output_channels // 8,
                                                   kernel_size=3, padding=1, bias=bias, device=device, dtype=dtype)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Processes the feature map through the four parallel branches and concatenates their outputs.

        Args:
            input_tensor (torch.Tensor): The feature map of shape [Batch, In_Channels, Height, Width].

        Returns:
            torch.Tensor: The concatenated feature map of shape [Batch, Out_Channels, Height, Width].
        """
        # Point-wise branch without activation
        output_1x1 = self.branch_1x1(input=input_tensor)

        # Reduced 3x3 and 5x5 branches, with a ReLU after their channel reduction
        output_3x3 = self.branch_3x3(input=torch.relu(input=self.branch_3x3_reduction(input=input_tensor)))
        output_5x5 = self.branch_5x5(input=torch.relu(input=self.branch_5x5_reduction(input=input_tensor)))

        # Pooling branch projected back to its channel share
        output_pooling = self.branch_pooling_projection(input=self.branch_pooling(input=input_tensor))

        # Concatenate every branch along the channel dimension
        return torch.cat(tensors=[output_1x1, output_3x3, output_5x5, output_pooling], dim=1)
