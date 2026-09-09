import torch

from architecture_modules.anchors import Anchors
from utilities.os_utilities import print_green, print_blue
from utilities.tensor_utilities import print_tensor_shape, print_tensor_list
from utilities.model.model_utilities import visualise_anchors


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

    # Define the scales and ratios for the anchors
    # Scales determine the base size of the anchor boxes, and ratios determine their aspect ratios (width/height)
    scales_and_ratios = {"scales": [32.0, 64.0], "ratios": [0.5, 1.0, 2.0]}

    # Define example parameters based on typical feature map resolutions
    input_image_size = 1024
    feature_map_size = 6

    # We will pick the scales and ratios for a specific feature map level (e.g., '5')
    scales = scales_and_ratios["scales"]
    aspect_ratios = scales_and_ratios["ratios"]

    # Display the configuration explicitly to assist with tracking spatial logic during debugging
    print_blue(output=f"Prepared Anchors configuration for Feature Map {feature_map_size}x{feature_map_size}:",
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
    print_green(output="Anchors module successfully instantiated and ready for spatial testing!", add_separators=True)

    # Print out detailed information about the generated anchors for debuggability
    generated_anchors = anchors_module.anchors

    print_blue(output="Anchors Tensor Shape:", add_separators=True)
    print_tensor_shape(tensor=generated_anchors, name="generated_anchors")

    print_blue(output="First 5 Generated Anchors [x_min, y_min, x_max, y_max]:", add_separators=True)
    print_tensor_list(tensor=generated_anchors[:5])

    print_blue(output="Last 5 Generated Anchors [x_min, y_min, x_max, y_max]:", add_separators=True)
    print_tensor_list(tensor=generated_anchors[-5:])

    # Visually debug the generated anchors on a black canvas
    # Using random_ratio to only visualize a random subset to avoid clutter
    visualise_anchors(input_image_size=input_image_size,
                      anchors=generated_anchors,
                      delayed=True,
                      delay=200,
                      random_ratio=0.5)


if __name__ == "__main__":
    debug_anchors()
