import cv2
import numpy as np
import torch

from utilities.os_utilities import print_blue


def add_anchor(anchor: np.ndarray, image: np.ndarray, input_image_size: int) -> np.ndarray:
    """
    Draws a single bounding box anchor onto the provided image canvas.
    
    This function processes an individual anchor's coordinates, clamps them to the image boundaries
    to prevent out-of-bounds drawing errors, and renders the rectangle using OpenCV. It centralizes 
    the bounding box drawing logic so it can be reused iteratively or collectively during visualization.
    
    Args:
        anchor (np.ndarray): The [x_min, y_min, x_max, y_max] coordinates of the bounding box.
        image (np.ndarray): The image canvas on which to draw the bounding box.
        input_image_size (int): The spatial size (height and width) of the image to clamp coordinates.
        
    Returns:
        np.ndarray: The updated image canvas containing the newly drawn anchor.
    """
    x_min, y_min, x_max, y_max = anchor

    # Ensure coordinates are within image boundaries for clean visualization
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(input_image_size, x_max)
    y_max = min(input_image_size, y_max)

    # Draw the bounding box on the canvas using a green color
    cv2.rectangle(img=image, pt1=(x_min, y_min), pt2=(x_max, y_max), color=(0, 255, 0), thickness=2)

    return image


def visualise_anchors(input_image_size: int, anchors: torch.Tensor, delayed: bool = False, delay: int = 0,
                      random_ratio: float = 1.0) -> None:
    """
    Visualizes the generated bounding box anchors on a black canvas using OpenCV.
    
    This utility function is critical for debugging the spatial configuration of anchors,
    ensuring they are placed and scaled correctly relative to the input image dimensions.
    It provides an interactive visual feedback loop during architecture development.
    
    Args:
        input_image_size (int): The height and width of the square input image.
        anchors (torch.Tensor): A tensor containing the [x_min, y_min, x_max, y_max] bounding boxes.
        delayed (bool): If True, renders and pauses at each anchor sequentially for step-by-step inspection.
        delay (int): The duration in milliseconds to pause during step-by-step inspection (0 means wait for key).
        random_ratio (float): The fraction of anchors to randomly sample and visualize (default is 1.0 for all).
    """
    # Create a completely black background image of the specified size with 3 channels (RGB)
    canvas = np.zeros(shape=(input_image_size, input_image_size, 3), dtype=np.uint8)

    # Convert the PyTorch tensor to a NumPy array for OpenCV compatibility
    anchors_array = anchors.cpu().numpy().astype(dtype=np.int32)

    # Randomly sample a fraction of the anchors for visualization if requested
    if random_ratio < 1.0:
        total_anchors = len(anchors_array)
        number_to_sample = int(total_anchors * random_ratio)

        # Select random indices without replacement, then sort them to preserve the original generation sequence
        sampled_indices = np.random.choice(a=total_anchors, size=number_to_sample, replace=False)
        sampled_indices.sort()

        anchors_array = anchors_array[sampled_indices]

    # Determine the window name for the OpenCV display
    window_name = "Anchors Visualization"

    if delayed:
        if not delay:
            print_blue(output="Showing Anchors Iteratively. Press any key to continue...", add_separators=True)

        # Iterate through each anchor and draw them one by one
        for anchor_index, anchor in enumerate(anchors_array):
            canvas = add_anchor(anchor=anchor, image=canvas, input_image_size=input_image_size)

            # Display the updated canvas containing the newly drawn anchor
            cv2.imshow(winname=window_name, mat=canvas)

            # Pause execution for the specified delay or until the user presses a key
            cv2.waitKey(delay=delay)
    else:
        # Iterate through all anchors and draw them collectively on the canvas
        for anchor in anchors_array:
            canvas = add_anchor(anchor=anchor, image=canvas, input_image_size=input_image_size)

        # Display the final canvas with all anchors drawn simultaneously
        cv2.imshow(winname=window_name, mat=canvas)
        print_blue(output="Showing all anchors collectively. Press any key to close the window...", add_separators=True)
        cv2.waitKey(delay=0)

    # Clean up the OpenCV windows after the visualization loop is complete
    cv2.destroyAllWindows()
