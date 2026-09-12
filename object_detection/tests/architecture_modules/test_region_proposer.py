import torch
from pathlib import Path

from object_detection.architecture_modules.region_proposer import RegionProposal
from object_detection.tests.utilities.testing_utilities import check_nn_module_method, create_deterministic_tensor


def test_region_proposer() -> None:
    # Define batch size for the test
    batch_size = 4
    input_channels = 256
    feature_map_size = 6
    feature_map_shape = (input_channels, feature_map_size, feature_map_size)

    # Number of ground truth boxes for the mock target tensor
    number_of_ground_truth_boxes = 5
    target_shape = (number_of_ground_truth_boxes, 4)

    # Create input tensors and populate the simple and batched dictionaries
    input_tensor = create_deterministic_tensor(shape=(1,) + feature_map_shape)
    target_tensor = create_deterministic_tensor(shape=(1,) + target_shape)

    simple_input_dictionary = {
        "input_tensor": input_tensor,
        "target_tensor": target_tensor
    }

    batched_input_tensor = input_tensor.broadcast_to(size=(batch_size,) + feature_map_shape)
    batched_target_tensor = target_tensor.broadcast_to(size=(batch_size,) + target_shape)

    batched_input_dictionary = {
        "input_tensor": batched_input_tensor,
        "target_tensor": batched_target_tensor
    }

    # Initialize device and dtype for configurations
    device = torch.device(device="cpu")
    dtype = torch.float32

    # Create a dictionary of configurations to test various paths of the RegionProposal
    region_proposal_configurations = {
        "basic_region_proposal": {
            "input_channels": input_channels,
            "scales": [32.0, 64.0],
            "aspect_ratios": [0.5, 1.0, 2.0],
            "input_image_size": 1024,
            "feature_map_size": feature_map_size,
            "device": device,
            "dtype": dtype
        },
        "complex_region_proposal": {
            "input_channels": input_channels,
            "scales": [16.0, 32.0, 64.0, 128.0],
            "aspect_ratios": [0.5, 1.0, 2.0],
            "input_image_size": 1024,
            "feature_map_size": feature_map_size,
            "device": device,
            "dtype": dtype
        }
    }

    # Iterate through the defined configurations to test each structural permutation
    for configuration_name, configuration_dictionary in region_proposal_configurations.items():
        print(f"Testing RegionProposal Configuration: {configuration_name}")

        # Instantiate the region proposal module with the current configuration parameters
        region_proposal_module = RegionProposal(**configuration_dictionary)

        # Call the testing utility to verify the forward pass deterministically
        # Note: RegionProposal returns (proposal_scores, proposal_boxes) so we specify two tensor names
        check_nn_module_method(
            module=region_proposal_module,
            simple_input_dictionary=simple_input_dictionary,
            output_tensor_names=[f"region_proposal_{configuration_name}_scores",
                                 f"region_proposal_{configuration_name}_boxes"],
            reference_folder=Path(__file__).parent / "reference_values",
            batch_size=batch_size,
            batched_input_dictionary=batched_input_dictionary,
            save_output=False
        )

        print(f" - {configuration_name} Test Completed Successfully.")


if __name__ == "__main__":
    test_region_proposer()
