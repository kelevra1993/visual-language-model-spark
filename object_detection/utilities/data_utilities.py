import cv2
import numpy as np


def preprocess(image: np.ndarray, image_size: int, keep_ratio: bool = True) -> np.ndarray:
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



def resize_image_and_boxes(image_pil: Image.Image, bounding_boxes: List[List[float]], target_size: Tuple[int, int] = (1024, 1024)) -> Tuple[Image.Image, List[List[float]]]:
    """
    Resizes a PIL image and its corresponding bounding boxes to a target spatial size.

    This function processes the image to fit the expected input dimensions of the object detection
    pipeline. It computes the independent x and y scaling factors and adjusts all bounding box
    coordinates accordingly to ensure spatial alignment is preserved for the Region Proposal Network.

    Args:
        image_pil (Image.Image): The original PIL image.
        bounding_boxes (List[List[float]]): A list of bounding boxes in [x_min, y_min, width, height] format.
        target_size (Tuple[int, int], optional): The desired output size (width, height). Defaults to (1024, 1024).

    Returns:
        Tuple[Image.Image, List[List[float]]]: The resized image and the scaled bounding boxes.
    """
    width, height = image_pil.size
    image_resized = image_pil.resize(target_size, Image.BILINEAR)
    scale_x = target_size[0] / width
    scale_y = target_size[1] / height

    resized_bounding_boxes = []
    for bounding_box in bounding_boxes:
        box_x, box_y, box_width, box_height = bounding_box
        new_x = box_x * scale_x
        new_y = box_y * scale_y
        new_width = box_width * scale_x
        new_height = box_height * scale_y
        resized_bounding_boxes.append([new_x, new_y, new_width, new_height])

    return image_resized, resized_bounding_boxes