"""
When run, generates the anchors of one feature map with the Anchors module of vision_training.models.detection,
prints their shape and first boxes, and draws a random subset of them on a black canvas to debug the anchor grid of
the region proposal.
"""

import torch

from vision_training.models.detection.anchors import Anchors
from vision_training.utilities.os_utilities import print_color
from vision_training.utilities.tensor_utilities import print_tensor_shape
from vision_training.utilities.visualization import visualise_anchors
from scripts.models.detection.detection_debugging_utilities import print_bounding_boxes


def debug_anchors() -> None:
    """
    Instantiates the Anchors class to serve as a foundational debugging sandbox.
    
    This debugging script is essential for the region proposal pipeline. It currently
    initializes the Anchors module and sets up the configuration dictionaries (like 
    scales and ratios) required for generating bounding box anchors across different 
    feature map levels, allowing for independent testing and verification without 
    running the entire neural network.
    
    Args:
        None
    """
    # Setup computational device and data type for the tensors
    device = torch.device(device="cpu")
    dtype = torch.float32

    # Define the scales and aspect ratios for the anchors
    # Scales determine the base size of the anchor boxes, and aspect ratios determine their aspect ratios (width/height)
    scales_and_ratios = {"scales": [32.0, 64.0], "aspect_ratios": [0.5, 1.0, 2.0]}

    # Define example parameters based on typical feature map resolutions
    input_image_size = 1024
    feature_map_size = 6

    # We will pick the scales and aspect ratios for a specific feature map level (e.g., '5')
    scales = scales_and_ratios["scales"]
    aspect_ratios = scales_and_ratios["aspect_ratios"]

    # Display the configuration explicitly to assist with tracking spatial logic during debugging
    print_color(color="blue",
                output=f"Prepared Anchors configuration for Feature Map {feature_map_size}x{feature_map_size}:",
                add_separators=True)
    print(f"Scales: {scales}")
    print(f"Aspect Ratios: {aspect_ratios}")

    # Instantiate the Anchors module with the explicitly named arguments
    anchors_module = Anchors(scales=scales,
                             aspect_ratios=aspect_ratios,
                             input_image_size=input_image_size,
                             feature_map_size=feature_map_size,
                             dtype=dtype,
                             device=device)

    # Inform the developer that the module has been instantiated and is ready for further anchor generation logic
    print_color(color="green",
                output="Anchors module successfully instantiated and ready for spatial testing!", add_separators=True)

    # Print out detailed information about the generated anchors for debuggability
    generated_anchors = anchors_module.anchors

    print_color(color="blue", output="Anchors Tensor Shape:", add_separators=True)
    print_tensor_shape(tensor=generated_anchors, name="generated_anchors")

    print_color(color="blue", output="First 10 Generated Anchors [x_min, y_min, x_max, y_max]:", add_separators=True)
    
    print_bounding_boxes(boxes=generated_anchors, number_of_boxes=10)

    # Visually debug the generated anchors on a black canvas
    # Using random_ratio to only visualize a random subset to avoid clutter
    visualise_anchors(input_image_size=input_image_size,
                      anchors=generated_anchors,
                      delayed=True,
                      delay=200,
                      random_ratio=0.5)


if __name__ == "__main__":
    debug_anchors()
