"""
Image and bounding box preprocessing shared by the data loaders and the inference clients: resizing to the square
network input with optional centred padding, the matching box transformation, and its inverse restoring predicted
boxes to the original image.
"""

import cv2
import numpy as np


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


def restore_boxes_to_original_image(bounding_boxes: np.ndarray, image_width: int, image_height: int,
                                    image_size: int, keep_ratio: bool) -> np.ndarray:
    """
    Maps boxes predicted on the preprocessed network input back to the original image, the exact inverse of
    'preprocess_boxes'. Used by the inference clients, which preprocess the images before sending them to the model
    and must report the detections in the coordinates of the images they were given.

    Args:
        bounding_boxes (np.ndarray): The [N, 4] [x1, y1, x2, y2] boxes in network input pixels.
        image_width (int): The width of the original image.
        image_height (int): The height of the original image.
        image_size (int): The size of the square network input.
        keep_ratio (bool): Whether the image was resized keeping its aspect ratio and padded to a square.

    Returns:
        np.ndarray: The [N, 4] boxes in original image pixels, clipped to the image.
    """
    restored_boxes = np.array(bounding_boxes, dtype=np.float64).reshape(-1, 4)

    if keep_ratio:
        # Remove the centred padding offset, then undo the single scaling factor
        scale = image_size / max(image_height, image_width)
        start_x = (image_size - int(round(image_width * scale))) // 2
        start_y = (image_size - int(round(image_height * scale))) // 2
        restored_boxes[:, 0::2] = (restored_boxes[:, 0::2] - start_x) / scale
        restored_boxes[:, 1::2] = (restored_boxes[:, 1::2] - start_y) / scale
    else:
        # Undo the independent horizontal and vertical scaling factors
        restored_boxes[:, 0::2] = restored_boxes[:, 0::2] / (image_size / image_width)
        restored_boxes[:, 1::2] = restored_boxes[:, 1::2] / (image_size / image_height)

    # Boxes predicted partly on the padding are clipped to the original image
    restored_boxes[:, 0::2] = restored_boxes[:, 0::2].clip(min=0.0, max=image_width)
    restored_boxes[:, 1::2] = restored_boxes[:, 1::2].clip(min=0.0, max=image_height)
    return restored_boxes


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
