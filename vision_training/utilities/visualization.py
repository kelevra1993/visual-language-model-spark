import os
import sys
import cv2
import numpy as np
import torch

from typing import Dict, List, Optional, Tuple

from vision_training.utilities.os_utilities import print_color

# Height in pixels of the caption bar written above the error images of the evaluation
CAPTION_BAR_HEIGHT = 24


def add_bounding_boxes(bounding_boxes: np.ndarray, image: np.ndarray, input_image_size: int,
                       color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draws one or more bounding boxes onto the provided image canvas.
    
    This function processes bounding box coordinates, clamps them to the image boundaries
    to prevent out-of-bounds drawing errors, and renders the rectangles using OpenCV. 
    It supports both single bounding boxes [4] and batches of bounding boxes [N, 4].
    
    Args:
        bounding_boxes (np.ndarray): The [x_min, y_min, x_max, y_max] coordinates. Shape (4,) or (N, 4).
        image (np.ndarray): The image canvas on which to draw the bounding boxes.
        input_image_size (int): The spatial size (height and width) of the image to clamp coordinates.
        color (Tuple[int, int, int]): The BGR color tuple for the bounding boxes. Defaults to green (0, 255, 0).
        
    Returns:
        np.ndarray: The updated image canvas containing the newly drawn bounding boxes.
    """
    # Standardize shape to [N, 4] for consistent iteration
    if bounding_boxes.ndim == 1:
        bounding_boxes = bounding_boxes.reshape((1, 4))

    for box in bounding_boxes:
        x_min, y_min, x_max, y_max = box

        # Ensure coordinates are within image boundaries for clean visualization
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(input_image_size, int(x_max))
        y_max = min(input_image_size, int(y_max))

        # Draw the bounding box on the canvas using the provided color
        cv2.rectangle(img=image, pt1=(x_min, y_min), pt2=(x_max, y_max), color=color, thickness=2)

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
            print_color(color="blue",
                        output="Showing Anchors Iteratively. Press any key to continue...", add_separators=True)

        for anchor_index, anchor in enumerate(anchors_array):
            canvas = add_bounding_boxes(bounding_boxes=anchor, image=canvas, input_image_size=input_image_size)

            # Display the updated canvas containing the newly drawn anchor
            cv2.imshow(winname=window_name, mat=canvas)

            # Pause execution for the specified delay or until the user presses a key
            cv2.waitKey(delay=delay)
    else:
        # Draw all anchors collectively on the canvas without a loop
        canvas = add_bounding_boxes(bounding_boxes=anchors_array, image=canvas, input_image_size=input_image_size)

        # Display the final canvas with all anchors drawn simultaneously
        cv2.imshow(winname=window_name, mat=canvas)
        print_color(color="blue",
                    output="Showing all anchors collectively. Press any key to close the window...",
                    add_separators=True)
        cv2.waitKey(delay=0)

    # Clean up the OpenCV windows after the visualization loop is complete
    cv2.destroyAllWindows()


def view_input_data(images: List[torch.Tensor], bounding_boxes: Optional[List[torch.Tensor]],
                    labels: Optional[List[torch.Tensor]], window_titles: List[str], display_input_size: int,
                    label_names: Optional[Dict[int, str]] = None, captions: Optional[List[str]] = None) -> bool:
    """
    Displays preprocessed input images with their ground truth boxes and labels, or with a caption describing their
    target, one window per image, so the user can check exactly what is fed to the neural network.

    It is shared by the dataloader benchmarker (a single window per image) and by the trainers' 'view_input_images'
    diagnosis mode (a Training and a Validation window side by side): the object detection trainer draws the boxes,
    the classification trainer writes the class of every image as its caption. Execution halts until the user
    presses SPACE (next image) or ENTER (stop viewing).

    Args:
        images (List[torch.Tensor]): The preprocessed image tensors of shape [Channels, Height, Width], on CPU or GPU.
        bounding_boxes (Optional[List[torch.Tensor]]): The [N, 4] bounding boxes associated with each image, or None
            for the tasks without boxes.
        labels (Optional[List[torch.Tensor]]): The [N] class labels associated with each image's bounding boxes, or
            None for the tasks without boxes.
        window_titles (List[str]): The title of the window displaying each image, also written on the image.
        display_input_size (int): The size in pixels of the longest side of each displayed window, typically the
            network input image size so that images are shown exactly as the network receives them.
        label_names (Optional[Dict[int, str]]): Mapping from class index to class name used for the box captions.
            Defaults to None, in which case the numeric class index is written.
        captions (Optional[List[str]]): A text written below the title of each image (e.g. its class). Defaults to
            None, in which case no caption is written.

    Returns:
        bool: True if the user pressed ENTER to stop the visualization (or no display is available), False otherwise.
    """
    # Skip the viewer when no graphical display is reachable (e.g. SSH without X forwarding) instead of crashing
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print_color(color="yellow",
                    output="No display available (DISPLAY is not set). Skipping the input image viewer.")
        return True

    # Iterate through each image to annotate it and show it in its own window
    for window_index, (image, window_title) in enumerate(zip(images, window_titles)):
        # Move the tensor to the CPU, transpose it from (C, H, W) to (H, W, C) and cast it to a contiguous uint8 array,
        # since OpenCV cannot draw on the non-contiguous memory layout left by the transposition
        image_numpy = np.ascontiguousarray(image.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8))

        # Draw each object's bounding box rectangle and its class caption, for the tasks with boxes
        if bounding_boxes is not None:
            boxes_numpy = bounding_boxes[window_index].detach().cpu().numpy()
            labels_numpy = labels[window_index].detach().cpu().numpy()
            for box, label in zip(boxes_numpy, labels_numpy):
                box_x1, box_y1, box_x2, box_y2 = map(int, box)
                caption = label_names.get(int(label), str(int(label))) if label_names else str(int(label))
                cv2.rectangle(img=image_numpy, pt1=(box_x1, box_y1), pt2=(box_x2, box_y2), color=(0, 255, 0),
                              thickness=2)
                cv2.putText(img=image_numpy, text=caption, org=(box_x1, max(box_y1 - 10, 0)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0), thickness=2)

        # Write the window title at the top left
        # Draw solid black background blocks first to guarantee the text is visible against any image background
        image_height, image_width, _ = image_numpy.shape
        cv2.rectangle(img=image_numpy, pt1=(5, 5), pt2=(300, 45), color=(0, 0, 0), thickness=-1)
        cv2.putText(img=image_numpy, text=window_title, org=(10, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.9, color=(255, 255, 255), thickness=2)

        # Write the caption of the image below its title, sized to the caption length
        if captions is not None:
            caption_width = cv2.getTextSize(text=captions[window_index], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                            fontScale=0.7, thickness=2)[0][0]
            cv2.rectangle(img=image_numpy, pt1=(5, 50), pt2=(caption_width + 15, 85), color=(0, 0, 0), thickness=-1)
            cv2.putText(img=image_numpy, text=captions[window_index], org=(10, 77), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7, color=(0, 255, 0), thickness=2)

        # Write the navigation instructions at the bottom left
        cv2.rectangle(img=image_numpy, pt1=(5, image_height - 45), pt2=(550, image_height - 5), color=(0, 0, 0),
                      thickness=-1)
        cv2.putText(img=image_numpy, text="Press SPACE For Next Image, Or ENTER To Quit",
                    org=(10, image_height - 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7, color=(150, 150, 150), thickness=2)

        # Scale the window so that its longest side matches the display size and place the windows side by side
        display_scale = display_input_size / max(image_height, image_width)
        display_width, display_height = int(image_width * display_scale), int(image_height * display_scale)
        cv2.namedWindow(winname=window_title, flags=cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winname=window_title, width=display_width, height=display_height)
        cv2.moveWindow(winname=window_title, x=window_index * (display_width + 20), y=50)
        cv2.imshow(winname=window_title, mat=image_numpy)

    # Enter a polling loop to restrict progression exclusively to the Spacebar or Enter keys
    while True:
        pressed_key = cv2.waitKey(delay=0) & 0xFF
        if pressed_key == 13:  # 13 represents the Enter/Return key
            cv2.destroyAllWindows()
            return True
        if pressed_key == ord(' '):
            break

    return False


def build_frame_grid(frames: torch.Tensor, sequence_mask: torch.Tensor, frame_captions: Optional[List[str]],
                     number_columns: int) -> torch.Tensor:
    """
    Lays out the real frames of a sequence in reading order on a grid, each numbered and optionally captioned (e.g.
    with its class), so that the input viewer of the sequence trainers shows a whole sequence in one window.

    Args:
        frames (torch.Tensor): The [T, 3, H, W] frames of the sequence, on CPU or GPU.
        sequence_mask (torch.Tensor): The [T] booleans, True for the real frames (padding frames are not shown).
        frame_captions (Optional[List[str]]): A text written on every frame, or None to only number them.
        number_columns (int): The number of frames per row of the grid.

    Returns:
        torch.Tensor: The [3, rows x H, columns x W] uint8 grid, gray between the frames.
    """
    real_frames = frames[sequence_mask.to(device=frames.device)].detach().cpu()
    number_frames, _, frame_height, frame_width = real_frames.shape
    number_rows = (number_frames + number_columns - 1) // number_columns

    # Gray canvas with a 4 pixel gap around every frame
    gap = 4
    grid = np.full(shape=(number_rows * (frame_height + gap) + gap, number_columns * (frame_width + gap) + gap, 3),
                   fill_value=128, dtype=np.uint8)
    for frame_index in range(number_frames):
        frame_numpy = np.ascontiguousarray(real_frames[frame_index].permute(1, 2, 0).numpy().astype(np.uint8))

        # Number of the frame, followed by its caption when given
        frame_text = str(frame_index) if frame_captions is None else f"{frame_index} {frame_captions[frame_index]}"
        cv2.rectangle(img=frame_numpy, pt1=(0, 0), pt2=(min(frame_width - 1, 12 + 9 * len(frame_text)), 20),
                      color=(0, 0, 0), thickness=-1)
        cv2.putText(img=frame_numpy, text=frame_text, org=(4, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45,
                    color=(0, 255, 0), thickness=1)

        # Place the frame in its cell, row by row
        top = gap + (frame_index // number_columns) * (frame_height + gap)
        left = gap + (frame_index % number_columns) * (frame_width + gap)
        grid[top:top + frame_height, left:left + frame_width] = frame_numpy

    return torch.from_numpy(grid).permute(2, 0, 1)


def add_caption_bar(image: np.ndarray, caption: str) -> np.ndarray:
    """
    Writes a caption in a black bar above an image, e.g. the true and predicted classes of a misclassified test sample
    in the error images of the evaluation. The bar is widened to the caption when the image is narrower, the image
    being padded with gray on its right.

    Args:
        image (np.ndarray): The [H, W, 3] uint8 BGR image.
        caption (str): The text written in the bar.

    Returns:
        np.ndarray: The [H + 24, max(W, caption width), 3] uint8 BGR image with its caption bar.
    """
    (caption_width, _), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45,
                                            thickness=1)
    output_width = max(image.shape[1], caption_width + 8)

    # Gray canvas holding the bar above the image, then the caption written in the bar
    captioned_image = np.full(shape=(image.shape[0] + CAPTION_BAR_HEIGHT, output_width, 3), fill_value=128,
                              dtype=np.uint8)
    captioned_image[:CAPTION_BAR_HEIGHT] = 0
    captioned_image[CAPTION_BAR_HEIGHT:, :image.shape[1]] = image
    cv2.putText(img=captioned_image, text=caption, org=(4, 17), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45,
                color=(255, 255, 255), thickness=1)
    return captioned_image
