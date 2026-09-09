import torch
import numpy as np

from utilities.tensor_utilities import print_tensor_list, print_tensor_shape
from utilities.os_utilities import print_blue, print_green, print_red
from architecture_modules.convolution_block import ChannelLayerNormalizer


def debug_layer_normalization() -> None:
    """
    Validates that the ChannelLayerNormalizer correctly normalizes over 
    the channel dimension for each spatial location.
    """
    number_channels = 4
    batch_size = 2
    height = 5
    width = 5

    device = torch.device("cpu")
    dtype = torch.float32

    # Instantiate the custom normalizer
    layer = ChannelLayerNormalizer(channels=number_channels, device=device, dtype=dtype)

    # Create a random input tensor [B, C, H, W]
    # We multiply by 50 and add 100 to ensure the initial mean/variance is far from 0 and 1
    input_tensor = torch.randn(batch_size, number_channels, height, width, device=device, dtype=dtype) * 10.0 + 20.0
    print_tensor_shape(tensor=input_tensor, name="input_tensor")

    # Apply the channel layer normalization
    output_tensor = layer(input_tensor=input_tensor)
    print_tensor_shape(tensor=output_tensor, name="output_tensor")

    # Calculate mean and standard deviation along the channel dimension (dim=1)
    # If the normalizer works, the mean should be 0 and the std should be 1 across this dimension
    # unbiased=False is used because LayerNorm calculates the population variance
    mean_across_channels = output_tensor.mean(dim=1)
    std_across_channels = output_tensor.std(dim=1, unbiased=False)

    print_blue(output="Mean across the channel dimension for the first batch item (should be ~0.0):",
               add_separators=True)
    print_red("Before Normalisation : ")
    print_tensor_list(tensor=input_tensor.mean(dim=1)[0])
    print_green("After Normalisation : ")
    print_tensor_list(tensor=mean_across_channels[0])

    print_green(output="Standard Deviation across the channel dimension for the first batch item (should be ~1.0):",
                add_separators=True)
    print_red("Before Normalisation : ")
    print_tensor_list(tensor=input_tensor.std(dim=1, unbiased=False)[0])
    print_green("After Normalisation : ")
    print_tensor_list(tensor=std_across_channels[0])


if __name__ == "__main__":
    debug_layer_normalization()
