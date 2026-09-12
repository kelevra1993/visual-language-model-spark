import torch

from architecture_modules.region_proposer import RegionProposal
from utilities.os_utilities import print_green, print_blue
from utilities.tensor_utilities import print_tensor_status, print_tensor_shape


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
    scales_and_ratios = {"scales": [32.0, 64.0], "aspect_ratios": [0.5, 1.0, 2.0]}
    scales = scales_and_ratios["scales"]
    aspect_ratios = scales_and_ratios["aspect_ratios"]

    # Define example parameters based on typical feature map resolutions
    input_image_size = 1024
    feature_map_size = 6
    input_channels = 256  # Typical backbone output channel count

    # Define IoU thresholds for Region Proposal target generation and filtering
    foreground_iou_threshold = 0.7
    background_iou_threholds = 0.3
    nms_iou_threshold = 0.7

    print_blue(output=f"Prepared Region Proposal configuration for Feature Map {feature_map_size}x{feature_map_size}:",
               add_separators=True)
    print(f"Scales: {scales}")
    print(f"Aspect Ratios: {aspect_ratios}")
    print(f"Input Channels: {input_channels}")

    # Instantiate the RegionProposal module with explicitly named arguments
    region_proposal_module = RegionProposal(input_channels=input_channels,
                                            scales=scales,
                                            aspect_ratios=aspect_ratios,
                                            input_image_size=input_image_size,
                                            feature_map_size=feature_map_size,
                                            foreground_iou_threshold=foreground_iou_threshold,
                                            background_iou_threholds=background_iou_threholds,
                                            nms_iou_threshold=nms_iou_threshold,
                                            dtype=dtype,
                                            device=device)

    # Inform the developer that the module has been instantiated
    print_green(output="RegionProposal module successfully instantiated and ready for testing!", add_separators=True)

    # -----------------------------------------------------------------------------------------
    # Forward Pass Testing
    # -----------------------------------------------------------------------------------------
    batch_size = 1

    # Create a mock feature map tensor that simulates the output from the Backbone
    # Shape: (batch_size, input_channels, feature_map_height, feature_map_width)
    input_tensor = torch.randn(size=(batch_size, input_channels, feature_map_size, feature_map_size),
                               dtype=dtype,
                               device=device)

    # Create a mock target tensor (e.g., ground truth bounding boxes)
    # Shape: (batch_size, num_ground_truth_boxes, 4)
    number_of_ground_truth_boxes = 3
    target_tensor = torch.randn(size=(batch_size, number_of_ground_truth_boxes, 4),
                                dtype=dtype,
                                device=device)

    print_blue(output="Executing forward pass with mock tensors...", add_separators=True)

    # Execute the forward pass
    # todo : Be careful, are we returning proposal boxes or anchor_box/proposal_box_transformations ?
    proposal_scores, proposal_boxes = region_proposal_module(input_tensor=input_tensor, target_tensor=target_tensor)

    print_tensor_shape(proposal_boxes, "proposal_boxes")
    print_tensor_shape(proposal_scores, "proposal_scores")


if __name__ == "__main__":
    debug_region_proposal()
