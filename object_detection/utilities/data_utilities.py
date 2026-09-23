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

        maximum_size = max([image_height, image_width])
        processed_image = np.zeros((maximum_size, maximum_size, 3), dtype=image.dtype)

        start_x = int((maximum_size - image_width) / 2)
        start_y = int((maximum_size - image_height) / 2)

        processed_image[start_y: start_y + image_height, start_x: start_x + image_width, :] = image[
            :image_height, :image_width, :]
        processed_image = cv2.resize(processed_image, (image_size, image_size))
    else:
        processed_image = cv2.resize(image, (image_size, image_size))

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
        maximum_size = max([image_height, image_width])
        start_x = int((maximum_size - image_width) / 2)
        start_y = int((maximum_size - image_height) / 2)
        scale = image_size / maximum_size

        for bounding_box in bounding_boxes:
            box_x_1, box_y_1, box_x_2, box_y_2 = bounding_box
            new_x_1 = (box_x_1 + start_x) * scale
            new_y_1 = (box_y_1 + start_y) * scale
            new_x_2 = (box_x_2 + start_x) * scale
            new_y_2 = (box_y_2 + start_y) * scale
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
