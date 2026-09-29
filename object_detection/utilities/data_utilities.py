import os
import sys
import torch
import cv2
import numpy as np

from typing import Dict, List, Optional

from utilities.os_utilities import print_color


def preprocess_image(image: np.ndarray, image_size: int, keep_ratio: bool = True) -> np.ndarray:
    """
    Preprocesses an image by resizing it to a specified square size, optionally maintaining its aspect ratio.

    If `keep_ratio` is True, the image is padded with zeros (black) to make it square before resizing,
    preventing distortion. If False, the image is simply stretched or squashed to the target size.

    Args:
        image (np.ndarray): The input image array, typically of shape (Height, Width, Channels).
        image_size (int): The target width and height for the output square image.
        keep_ratio (bool): If True, pads the image to maintain its original aspect ratio before resizing.
        Defaults to True.

    Returns:
        np.ndarray: The preprocessed and resized image of shape (image_size, image_size, Channels).
    """
    if keep_ratio:
        image_height, image_width, _ = image.shape
        scale = image_size / max(image_height, image_width)
        
        # Calculate new dimensions after scaling, matching DALI's not_larger logic
        new_width = int(round(image_width * scale))
        new_height = int(round(image_height * scale))
        
        # Resize first to pure image pixels to avoid bilinear black padding bleed
        resized_image = cv2.resize(src=image, dsize=(new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create the padded square canvas
        processed_image = np.zeros((image_size, image_size, 3), dtype=image.dtype)
        
        # Calculate centering offsets (matching DALI crop_pos=0.5 symmetric padding)
        start_x = (image_size - new_width) // 2
        start_y = (image_size - new_height) // 2
        
        # Drop the resized image into the center
        processed_image[start_y: start_y + new_height, start_x: start_x + new_width, :] = resized_image
    else:
        processed_image = cv2.resize(src=image, dsize=(image_size, image_size), interpolation=cv2.INTER_LINEAR)

    return processed_image


def preprocess_boxes(bounding_boxes: list[list[float]], image_width: int, image_height: int, image_size: int,
                     keep_ratio: bool = True) -> list[list[float]]:
    """
    Adjusts bounding box coordinates to match the preprocessing transformations applied to the image.

    This function scales and optionally shifts bounding boxes in the [x_1, y_1, x_2, y_2] format
    to ensure spatial alignment is preserved when the original image is resized and padded.

    Args:
        bounding_boxes (list[list[float]]): A list of bounding boxes in [x_1, y_1, x_2, y_2] format.
        image_width (int): The original width of the image before preprocessing.
        image_height (int): The original height of the image before preprocessing.
        image_size (int): The target width and height for the output image.
        keep_ratio (bool): If True, computes the scaling based on zero-padding the image to maintain its original aspect ratio.
        Defaults to True.

    Returns:
        list[list[float]]: The scaled and shifted bounding boxes in [x_1, y_1, x_2, y_2] format.
    """
    processed_bounding_boxes = []

    if keep_ratio:
        scale = image_size / max(image_height, image_width)
        new_width = int(round(image_width * scale))
        new_height = int(round(image_height * scale))
        
        start_x = (image_size - new_width) // 2
        start_y = (image_size - new_height) // 2

        for bounding_box in bounding_boxes:
            box_x_1, box_y_1, box_x_2, box_y_2 = bounding_box
            
            # Scale coordinates first
            scaled_x_1 = box_x_1 * scale
            scaled_y_1 = box_y_1 * scale
            scaled_x_2 = box_x_2 * scale
            scaled_y_2 = box_y_2 * scale
            
            # Then shift them by the padding offset
            new_x_1 = scaled_x_1 + start_x
            new_y_1 = scaled_y_1 + start_y
            new_x_2 = scaled_x_2 + start_x
            new_y_2 = scaled_y_2 + start_y
            
            processed_bounding_boxes.append([new_x_1, new_y_1, new_x_2, new_y_2])
    else:
        scale_x = image_size / image_width
        scale_y = image_size / image_height

        for bounding_box in bounding_boxes:
            box_x_1, box_y_1, box_x_2, box_y_2 = bounding_box
            new_x_1 = box_x_1 * scale_x
            new_y_1 = box_y_1 * scale_y
            new_x_2 = box_x_2 * scale_x
            new_y_2 = box_y_2 * scale_y
            processed_bounding_boxes.append([new_x_1, new_y_1, new_x_2, new_y_2])

    return processed_bounding_boxes


def preprocess_image_and_boxes(image: np.ndarray, bounding_boxes: list[list[float]], image_size,
                               keep_ratio: bool = True) -> tuple[np.ndarray, list[list[float]]]:
    """
    Preprocesses both an image and its corresponding bounding boxes to the target spatial size.

    This function acts as a wrapper to process the image using OpenCV and shift the bounding boxes accordingly
    to fit the expected input dimensions of the object detection pipeline while optionally preserving the aspect ratio.

    Args:
        image (np.ndarray): The input image array, typically of shape (Height, Width, Channels).
        bounding_boxes (list[list[float]]): A list of bounding boxes in [x_1, y_1, x_2, y_2] format.
        image_size (int, optional): The desired output size (width and height).
        keep_ratio (bool, optional): If True, pads the image to maintain its original aspect ratio before resizing.
    Returns:
        tuple[np.ndarray, list[list[float]]]: The preprocessed image and the adjusted bounding boxes.
    """
    image_height, image_width, _ = image.shape

    processed_image = preprocess_image(image=image, image_size=image_size, keep_ratio=keep_ratio)
    processed_bounding_boxes = preprocess_boxes(bounding_boxes=bounding_boxes, image_width=image_width,
                                                image_height=image_height, image_size=image_size, keep_ratio=keep_ratio)

    return processed_image, processed_bounding_boxes


def view_input_data(images: List[torch.Tensor], bounding_boxes: List[torch.Tensor], labels: List[torch.Tensor],
                    window_titles: List[str], display_input_size: int,
                    label_names: Optional[Dict[int, str]] = None) -> bool:
    """
    Displays preprocessed input images with their ground truth boxes and labels, one window per image, so the user
    can check exactly what is fed to the neural network.

    It is shared by the dataloader benchmarker (a single window per image) and by the Trainer's 'view_input_images'
    diagnosis mode (a Training and a Validation window side by side). Execution halts until the user presses
    SPACE (next image) or ENTER (stop viewing).

    Args:
        images (List[torch.Tensor]): The preprocessed image tensors of shape [Channels, Height, Width], on CPU or GPU.
        bounding_boxes (List[torch.Tensor]): The [N, 4] bounding boxes associated with each image.
        labels (List[torch.Tensor]): The [N] class labels associated with each image's bounding boxes.
        window_titles (List[str]): The title of the window displaying each image, also written on the image.
        display_input_size (int): The size in pixels of the longest side of each displayed window, typically the
            network input image size so that images are shown exactly as the network receives them.
        label_names (Optional[Dict[int, str]]): Mapping from class index to class name used for the box captions.
            Defaults to None, in which case the numeric class index is written.

    Returns:
        bool: True if the user pressed ENTER to stop the visualization (or no display is available), False otherwise.
    """
    # Skip the viewer when no graphical display is reachable (e.g. SSH without X forwarding) instead of crashing
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print_color(color="yellow",
                    output="No display available (DISPLAY is not set). Skipping the input image viewer.")
        return True

    # Iterate through each image to annotate it and show it in its own window
    for window_index, (image, image_bounding_boxes, image_labels, window_title) in enumerate(
            zip(images, bounding_boxes, labels, window_titles)):
        # Move the tensor to the CPU, transpose it from (C, H, W) to (H, W, C) and cast it to a contiguous uint8 array,
        # since OpenCV cannot draw on the non-contiguous memory layout left by the transposition
        image_numpy = np.ascontiguousarray(image.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8))
        boxes_numpy = image_bounding_boxes.detach().cpu().numpy()
        labels_numpy = image_labels.detach().cpu().numpy()

        # Draw each object's bounding box rectangle and its class caption
        for box, label in zip(boxes_numpy, labels_numpy):
            box_x1, box_y1, box_x2, box_y2 = map(int, box)
            caption = label_names.get(int(label), str(int(label))) if label_names else str(int(label))
            cv2.rectangle(img=image_numpy, pt1=(box_x1, box_y1), pt2=(box_x2, box_y2), color=(0, 255, 0), thickness=2)
            cv2.putText(img=image_numpy, text=caption, org=(box_x1, max(box_y1 - 10, 0)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0), thickness=2)

        # Write the window title at the top left and the instructions at the bottom left
        # Draw solid black background blocks first to guarantee the text is visible against any image background
        image_height, image_width, _ = image_numpy.shape
        cv2.rectangle(img=image_numpy, pt1=(5, 5), pt2=(300, 45), color=(0, 0, 0), thickness=-1)
        cv2.putText(img=image_numpy, text=window_title, org=(10, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.9, color=(255, 255, 255), thickness=2)
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
