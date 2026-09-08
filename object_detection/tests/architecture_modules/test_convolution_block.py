import torch
from pathlib import Path
from object_detection.architecture_modules.convolution_block import ConvolutionBlock
from object_detection.tests.utilities.testing_utilities import check_nn_module_method, create_deterministic_tensor


def test_convolution_block():
    # Define batch size for the test
    batch_size = 4
    image_shape = (3, 32, 32)

    # Create input tensors and populate the simple and batched dictionaries
    input_tensor = create_deterministic_tensor(shape=(1,) + image_shape)
    simple_input_dictionary = {"input_tensor": input_tensor}
    
    batched_input_tensor = input_tensor.broadcast_to(size=(batch_size,) + image_shape)
    batched_input_dictionary = {"input_tensor": batched_input_tensor}

    # Initialize device and dtype for configurations
    device = torch.device(device="cpu")
    dtype = torch.float32

    # Create a dictionary of configurations to test various paths of the ConvolutionBlock
    convolution_configurations = {
        "basic_convolution": {
            "input_channels": 3, "output_channels": 16, "bias": True, "number_layers": 1,
            "kernel_size": 3, "stride": 1, "padding": 1,
            "feature_map_normalization": "none", "activation": False, "dropout_rate": 0.0, "add_pooling": False,
            "device": device, "dtype": dtype
        },
        "batch_norm_and_activation": {
            "input_channels": 3, "output_channels": 32, "bias": False, "number_layers": 2,
            "kernel_size": 3, "stride": 1, "padding": 1,
            "feature_map_normalization": "batch", "activation": True, "dropout_rate": 0.0, "add_pooling": False,
            "device": device, "dtype": dtype
        },
        "layer_norm_and_activation": {
            "input_channels": 3, "output_channels": 32, "bias": False, "number_layers": 2,
            "kernel_size": 3, "stride": 1, "padding": 1,
            "feature_map_normalization": "layer", "activation": True, "dropout_rate": 0.0, "add_pooling": False,
            "device": device, "dtype": dtype
        },
        "deep_with_pooling_and_dropout": {
            "input_channels": 3, "output_channels": 64, "bias": False, "number_layers": 3,
            "kernel_size": 5, "stride": 2, "padding": 2,
            "feature_map_normalization": "batch", "activation": True, "dropout_rate": 0.5, "add_pooling": True,
            "device": device, "dtype": dtype
        }
    }

    # Iterate through the defined configurations to test each structural permutation
    for configuration_name, configuration_dictionary in convolution_configurations.items():
        print(f"Testing ConvolutionBlock Configuration: {configuration_name}")

        # Instantiate the convolution block with the current configuration parameters
        convolution_module = ConvolutionBlock(**configuration_dictionary)

        # Call the testing utility to verify the forward pass deterministically
        check_nn_module_method(
            module=convolution_module,
            simple_input_dictionary=simple_input_dictionary,
            output_tensor_names=[f"convolution_block_{configuration_name}"],
            reference_folder=Path(__file__).parent / "reference_values",
            batch_size=batch_size,
            batched_input_dictionary=batched_input_dictionary
        )

        print(f" - {configuration_name} Test Completed Successfully.")
