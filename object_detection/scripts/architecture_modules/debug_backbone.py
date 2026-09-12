import torch

from architecture_modules.backbone import Backbone
from utilities.os_utilities import print_green, print_blue
from utilities.tensor_utilities import print_tensor_status, print_tensor_shape


def debug_backbone() -> None:
    """
    Instantiates the Backbone module to serve as a foundational debugging sandbox.
    
    This function sets up a standard configuration for the backbone, instantiates it,
    prints its structural summary, and performs a single forward pass with a dummy tensor
    to verify the computational graph and spatial downsampling.
    """
    print_blue(output="------------------------------------------------------------", add_separators=False)
    print_blue(output="Prepared Backbone configuration:", add_separators=False)
    print_blue(output="------------------------------------------------------------", add_separators=False)

    input_channels = 3
    # Standard VGG-like configuration mapping block index to [number_layers, output_channels]
    convolutions = {"1": [2, 32], "2": [2, 64], "3": [3, 64], "4": [3, 128], "5": [3, 128]}
    modules = {"residual": False}
    last_max_pooling = False
    normalization = {"feature_map_normalization": "none"}
    enhancer_convolution_indices = []

    device = torch.device(device="cpu")
    dtype = torch.float32

    # Instantiate the backbone module
    backbone_module = Backbone(
        input_channels=input_channels,
        convolutions=convolutions,
        modules=modules,
        last_max_pooling=last_max_pooling,
        normalization=normalization,
        enhancer_convolution_indices=enhancer_convolution_indices,
        device=device,
        dtype=dtype)

    print_green(output="-----------------------------------------------------------------------", add_separators=False)
    print_green(output="Backbone module successfully instantiated and ready for testing!", add_separators=False)
    print_green(output="-----------------------------------------------------------------------", add_separators=False)

    # Print the architectural summary
    expected_image_size = (1024, 1024)
    backbone_module.print_summary(expected_image_size=expected_image_size)

    # Generate a dummy input tensor
    batch_size = 2
    image_height = 1024
    image_width = 1024
    input_tensor = torch.randn(size=(batch_size, input_channels, image_height, image_width), dtype=dtype, device=device)

    print_tensor_status(tensor=input_tensor, name="input_tensor")
    print_tensor_shape(tensor=input_tensor, name="input_tensor")

    print_blue(output="--------------------------------------------", add_separators=False)
    print_blue(output="Executing forward pass with mock tensor...", add_separators=False)
    print_blue(output="--------------------------------------------", add_separators=False)

    # Execute the forward pass
    output_tensor = backbone_module(input_tensor=input_tensor)

    print_tensor_status(tensor=output_tensor, name="output_tensor")
    print_tensor_shape(tensor=output_tensor, name="output_tensor")


if __name__ == "__main__":
    debug_backbone()
