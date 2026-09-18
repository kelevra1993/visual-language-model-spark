from architecture_modules.backbone import Backbone
from scripts.utilities.debugging_utilities import get_model_configuration, get_dummy_input_tensor
from utilities.os_utilities import print_green, print_blue
from utilities.tensor_utilities import print_tensor_shape


def debug_backbone() -> None:
    """
    Instantiates the Backbone module to serve as a foundational debugging sandbox.
    
    This function sets up a standard configuration for the backbone, instantiates it,
    prints its structural summary, and performs a single forward pass with a dummy tensor
    to verify the computational graph and spatial downsampling.
    """
    print_blue(output="Prepared Backbone configuration:", add_separators=True)

    # Fetch standard configuration and tensor generating tools via utility
    configuration, device, dtype = get_model_configuration()
    backbone_configuration = configuration.get("Backbone")
    input_image_size = configuration.get("Data").get("image_settings").get("size")

    # Instantiate the backbone module
    backbone_module = Backbone(input_channels=backbone_configuration.get("input_channels"),
                               convolutions=backbone_configuration.get("convolutions"),
                               modules=backbone_configuration.get("modules"),
                               last_max_pooling=backbone_configuration.get("last_max_pooling"),
                               normalization=backbone_configuration.get("normalization"),
                               detection_convolution_block_indices=backbone_configuration.get("detection_convolution_block_indices"),
                               input_image_size=input_image_size,
                               device=device,
                               dtype=dtype)

    print_green(output="Backbone module successfully instantiated and ready for testing!", add_separators=True)

    # Print the architectural summary
    backbone_module.print_summary()

    # Generate a dummy input tensor natively using the utility
    batch_size, input_channels, image_size, input_tensor = get_dummy_input_tensor(device=device, dtype=dtype)

    print_tensor_shape(tensor=input_tensor, indent=1)

    # Execute the forward pass
    output_tensor, output_tensor_dictionary = backbone_module(input_tensor=input_tensor)

    print_tensor_shape(tensor=output_tensor, indent=1)

    print_blue(output="Testing dynamic structural information computation...", add_separators=True)

    # Compute structural information dynamically without a full forward pass
    computed_information = backbone_module.compute_detection_input_information()

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
