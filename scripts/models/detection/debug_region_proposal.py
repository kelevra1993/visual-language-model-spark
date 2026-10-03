"""
When run, builds the RegionProposal module of vision_training.models.detection, runs a forward pass on a mock
backbone feature map and applies the predicted regressions to the anchors with apply_regression_predictions, printing
the anchors and proposals to debug the region proposal network.
"""

import torch

from vision_training.models.detection.region_proposer import RegionProposal
from vision_training.utilities.os_utilities import print_color
from vision_training.models.detection.box_operations import apply_regression_predictions
from scripts.models.detection.detection_debugging_utilities import print_bounding_boxes


def debug_region_proposal() -> None:
    """
    Instantiates the RegionProposal module to serve as a foundational debugging sandbox.
    
    This debugging script sets up the Region Proposal Network with its requisite 
    configuration parameters (channels, scales, feature map sizes, IoU thresholds).
    It ensures the module can be instantiated independently, setting the stage for 
    iteratively testing the forward pass and anchor bounding box regression logic.
    
    Args:
        None
    """
    # Setup computational device and data type for the tensors
    device = torch.device(device="cpu")
    dtype = torch.float32

    # Define the scales and aspect ratios for the anchors
    scales_and_ratios = {"scales": [32.0, 64.0, 128.0], "aspect_ratios": [0.5, 1.0, 2.0]}
    scales = scales_and_ratios["scales"]
    aspect_ratios = scales_and_ratios["aspect_ratios"]

    # Define example parameters based on typical feature map resolutions
    input_image_size = 1024
    feature_map_size = 6
    input_channels = 256  # Typical backbone output channel count

    # Region Proposal head definition [number of layers, output channels] and normalization of its convolutions,
    # matching the RegionProposal section of the example configuration
    head_configurations = [2, 64]
    feature_map_normalization = "batch"

    print_color(color="blue",
                output=f"Prepared Region Proposal configuration for Feature Map {feature_map_size}x{feature_map_size}:",
                add_separators=True)
    print(f"Scales: {scales}")
    print(f"Aspect Ratios: {aspect_ratios}")
    print(f"Input Channels: {input_channels}")
    print(f"Head Configurations: {head_configurations}")
    print(f"Feature Map Normalization: {feature_map_normalization}")

    # Instantiate the RegionProposal module with explicitly named arguments
    region_proposal_module = RegionProposal(head_configurations=head_configurations, input_channels=input_channels,
                                            scales=scales, aspect_ratios=aspect_ratios,
                                            input_image_size=input_image_size, feature_map_size=feature_map_size,
                                            feature_map_normalization=feature_map_normalization,
                                            dtype=dtype, device=device)

    # Inform the developer that the module has been instantiated
    print_color(color="green",
                output="RegionProposal module successfully instantiated and ready for testing!", add_separators=True)

    # -----------------------------------------------------------------------------------------
    # Forward Pass Testing
    # -----------------------------------------------------------------------------------------
    batch_size = 1

    # Create a mock feature map tensor that simulates the output from the Backbone
    # Shape: (batch_size, input_channels, feature_map_height, feature_map_width)
    input_tensor = torch.randn(size=(batch_size, input_channels, feature_map_size, feature_map_size),
                               dtype=dtype,
                               device=device)

    print_color(color="blue", output="Executing forward pass with mock tensors...", add_separators=True)

    # Execute the forward pass
    proposal_scores, proposal_boxes_transformations, proposal_boxes = region_proposal_module(input_tensor=input_tensor)

    # Extract original anchors
    original_anchors = region_proposal_module.region_proposal_anchor_object.anchors

    print_color(color="blue", output="Original Anchors (First 5):", add_separators=True)
    print_bounding_boxes(boxes=original_anchors, number_of_boxes=5, indent=1)

    # Apply the predicted bounding box transformations to our anchors manually for debugging
    applied_proposal_boxes = apply_regression_predictions(
        regression_predictions=proposal_boxes_transformations.detach().unsqueeze(dim=-2),
        boxes=original_anchors).squeeze(dim=-2)

    print_color(color="blue", output="Region Proposal Transformed Boxes (First 5):", add_separators=True)
    # The proposal boxes returned from the region_proposal_module (shape: B, N, 4)
    print_bounding_boxes(boxes=proposal_boxes[0], number_of_boxes=5, indent=1)

    print_color(color="blue", output="Applied Regressions (First 5):", add_separators=True)
    # The proposal boxes returned are the applied regressions (shape: B, N, 4)
    print_bounding_boxes(boxes=applied_proposal_boxes[0], number_of_boxes=5, indent=1)


if __name__ == "__main__":
    debug_region_proposal()
