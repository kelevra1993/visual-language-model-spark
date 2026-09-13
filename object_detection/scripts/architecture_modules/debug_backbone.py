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
    print_blue(output="Prepared Backbone configuration:", add_separators=True)

    input_channels = 3
    input_image_size = 1024

    # Standard VGG-like configuration mapping block index to [number_layers, output_channels]
    convolutions = {"1": [2, 16], "2": [2, 32], "3": [3, 32], "4": [3, 64], "5": [3, 128]}
    modules = {"residual": False}
    last_max_pooling = False
    normalization = {"feature_map_normalization": "none"}
    enhancer_convolution_indices = [3, 4, 5]

    device = torch.device(device="cpu")
    dtype = torch.float32

    # Instantiate the backbone module
    backbone_module = Backbone(input_channels=input_channels,
                               convolutions=convolutions,
                               modules=modules,
                               last_max_pooling=last_max_pooling,
                               normalization=normalization,
                               enhancer_convolution_indices=enhancer_convolution_indices,
                               input_image_size=input_image_size,
                               device=device,
                               dtype=dtype)

    print_green(output="Backbone module successfully instantiated and ready for testing!", add_separators=True)

    # Print the architectural summary
    backbone_module.print_summary()

    # Generate a dummy input tensor
    batch_size = 2
    input_tensor = torch.randn(size=(batch_size, input_channels, input_image_size, input_image_size), dtype=dtype,
                               device=device)

    print_tensor_shape(tensor=input_tensor, name="input_tensor")

    print_blue(output="Executing forward pass with mock tensor...", add_separators=True)

    # Execute the forward pass
    output_tensor, output_tensor_dictionary = backbone_module(input_tensor=input_tensor)

    print_tensor_shape(tensor=output_tensor, name="final_backbone_output_tensor")

    print_blue(output="Enhancer Output Dictionary Iteration:", add_separators=True)

    for block_index, tensor in output_tensor_dictionary.items():
        print_tensor_shape(tensor=tensor, name=f"enhancer_output_block_{block_index}")

    print_blue(output="Testing dynamic structural information computation...", add_separators=True)

    # Compute structural information dynamically without a full forward pass
    computed_information = backbone_module.compute_enhancer_input_information()

    # Verify that the computed structural information matches the actual tensor dimensions
    for block_index, expected_structural_dictionary in computed_information.items():
        actual_size = output_tensor_dictionary[block_index].shape[2]
        actual_channels = output_tensor_dictionary[block_index].shape[1]

        expected_size = expected_structural_dictionary["feature_map_size"]
        expected_channels = expected_structural_dictionary["number_channels"]

        match_status = "SUCCESS" if (
                actual_size == expected_size and actual_channels == expected_channels) else "FAILED"

        output_color_function = print_green if match_status == "SUCCESS" else print_blue
        output_color_function(output=f"Block {block_index} | Computed Size: {expected_size} (Actual: {actual_size})"
                                     f" | Computed Channels: {expected_channels} (Actual: {actual_channels})"
                                     f" | {match_status}", add_separators=False)


if __name__ == "__main__":
    debug_backbone()
