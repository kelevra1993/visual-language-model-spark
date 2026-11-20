"""
Image, mask and bounding box preprocessing shared by the data loaders and the deployment (the requests of
deployment/deployment_utilities.py, the input contract of deployment/input_contract.py): the reading of the dataset
and inference images, naming the files that cannot be read, the resizing to the square network input with
optional centred padding (bilinear for the images, nearest on the pixel centres for the segmentation masks), the
matching box transformation, and their inverses restoring predicted boxes and segmentation predictions (class maps or
probabilities) to the original image, called by the evaluations at the original resolution (the instance segmentation
trainer for the boxes, the segmentation trainer for the predictions) and left to the clients of the exported models,
no inference code of the repository calling them.
"""

import cv2
import numpy as np
import torch

from pathlib import Path
from typing import Tuple

from vision_training.utilities.os_utilities import format_path_link, print_color

# Pixel types of the image files of more than 8 bits per channel already reported by read_image, each one being reported
# once per process (the main process and every loader worker)
REPORTED_IMAGE_DEPTHS = set()


def read_image(image_path: str | Path) -> np.ndarray:
    """
    Reads an image as BGR, for the classification, COCO, segmentation and sequence datasets and the inference folder
    of the deployment (deployment_utilities.py): OpenCV gives None for a missing, unreadable or corrupted file, which
    would otherwise fail later with an unrelated error, so the file is named in the error. The training loop then
    drops the batch holding it and prints the error once (BaseTrainer.get_next_batch), stopping once a split drops
    too many batches, and a test evaluation stopped by it stops the training with this error. A file of more than 8
    bits per channel (e.g. a 16 bit PNG holding 12 bit data) is read with 8 bits like before, its values divided by
    256, with a yellow warning the first time a file of its pixel type is read, since a source using fewer bits than
    its type loses most of its signal.

    Args:
        image_path (str | Path): The image file.

    Returns:
        np.ndarray: The uint8 [H, W, 3] BGR image.

    Raises:
        ValueError: If the file cannot be read as an image.
    """
    # Decoded at the depth of the file, which gives the pixels of IMREAD_COLOR for a file of 8 bits per channel
    image = cv2.imread(filename=str(image_path), flags=cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    if image is None:
        raise ValueError(f"The image cannot be read : {format_path_link(path=Path(image_path))}")

    # A file of more bits per channel is reported, then decoded again exactly as before, with 8 bits per channel
    if image.dtype != np.uint8:
        if image.dtype.name not in REPORTED_IMAGE_DEPTHS:
            REPORTED_IMAGE_DEPTHS.add(image.dtype.name)
            print_color(color="yellow", output=f"Image Of {image.dtype.name} Pixels Read With 8 Bits Per Channel, Its "
                                               f"Values Scaled Down (A 12 Bit Source In 16 Bits Keeps 4 Bits) : "
                                               f"{format_path_link(path=Path(image_path))}")
        image = cv2.imread(filename=str(image_path))
        if image is None:
            raise ValueError(f"The image cannot be read : {format_path_link(path=Path(image_path))}")
    return image


def compute_letterbox_region(image_height: int, image_width: int, image_size: int) -> Tuple[int, int, int, int]:
    """
    Gives the part of the square network input holding an image resized with its aspect ratio kept, centred between
    the padding, so that the preprocessing places the image and the restorations of the predictions cut it out at
    exactly the same pixels.

    Args:
        image_height (int): The height of the original image in pixels.
        image_width (int): The width of the original image in pixels.
        image_size (int): The size of the square network input.

    Returns:
        Tuple[int, int, int, int]: The top row and left column of the resized image in the network input, and its
            height and width.
    """
    # Single scaling factor of the longest side, the resized sides rounded to the nearest pixel but never below one, so
    # that a very elongated image (e.g. 1 x 2000 pixels at 512) keeps its short side
    scale = image_size / max(image_height, image_width)
    resized_width, resized_height = max(1, int(round(image_width * scale))), max(1, int(round(image_height * scale)))

    # Symmetric padding around the resized image
    return (image_size - resized_height) // 2, (image_size - resized_width) // 2, resized_height, resized_width


def preprocess_image(image: np.ndarray, image_size: int, keep_ratio: bool = True,
                     interpolation: int = cv2.INTER_LINEAR, padding_value: int = 0) -> np.ndarray:
    """
    Preprocesses an image by resizing it to a specified square size, optionally maintaining its aspect ratio. The
    segmentation masks go through the same resizing and padding as their images, with the nearest interpolation on the
    pixel centres (cv2.INTER_NEAREST_EXACT) so that every pixel keeps a category of the mask and stays aligned with
    the image, and the padding at the ignored value of the masks.

    If `keep_ratio` is True, the image is padded with 'padding_value' (black for the images) to make it square before
    resizing, preventing distortion. If False, the image is simply stretched or squashed to the target size.

    Args:
        image (np.ndarray): The input image array, of shape (Height, Width, Channels), or (Height, Width) for a mask.
        image_size (int): The target width and height for the output square image.
        keep_ratio (bool): If True, pads the image to maintain its original aspect ratio before resizing.
        Defaults to True.
        interpolation (int): The OpenCV interpolation of the resizing, bilinear for the images and
            cv2.INTER_NEAREST_EXACT for the masks, the plain cv2.INTER_NEAREST shifting them by half a pixel to the
            bottom right. Defaults to cv2.INTER_LINEAR.
        padding_value (int): The value of the padding pixels, 0 (black) for the images. Defaults to 0.

    Returns:
        np.ndarray: The preprocessed and resized image of shape (image_size, image_size, Channels), or
            (image_size, image_size) for a mask.
    """
    if keep_ratio:
        # Place of the resized image in the square input, centred between the padding
        start_y, start_x, new_height, new_width = compute_letterbox_region(
            image_height=image.shape[0], image_width=image.shape[1], image_size=image_size)

        # Resize first to pure image pixels to avoid bilinear black padding bleed
        resized_image = cv2.resize(src=image, dsize=(new_width, new_height), interpolation=interpolation)

        # Create the padded square canvas, with the channels of the image when it has some
        processed_image = np.full(shape=(image_size, image_size) + image.shape[2:], fill_value=padding_value,
                                  dtype=image.dtype)

        # Drop the resized image into the center
        processed_image[start_y: start_y + new_height, start_x: start_x + new_width] = resized_image
    else:
        processed_image = cv2.resize(src=image, dsize=(image_size, image_size), interpolation=interpolation)

    return processed_image


def preprocess_boxes(bounding_boxes: list[list[float]], image_width: int, image_height: int, image_size: int,
                     keep_ratio: bool = True) -> list[list[float]]:
    """
    Adjusts bounding box coordinates to match the preprocessing transformations applied to the image.

    This function scales and optionally shifts bounding boxes in the [x_1, y_1, x_2, y_2] format
    to ensure spatial alignment is preserved when the original image is resized and padded. Any flat list of
    alternating x and y coordinates goes through the same transformation, so that the polygon vertices of the instance
    masks [x_1, y_1, x_2, y_2, x_3, y_3, ...] are placed exactly like the boxes before their rasterisation.

    Args:
        bounding_boxes (list[list[float]]): A list of bounding boxes in [x_1, y_1, x_2, y_2] format, or of flat
            [x, y, x, y, ...] coordinate lists of even length.
        image_width (int): The original width of the image before preprocessing.
        image_height (int): The original height of the image before preprocessing.
        image_size (int): The target width and height for the output image.
        keep_ratio (bool): If True, computes the scaling based on zero-padding the image to maintain its original aspect
            ratio. Defaults to True.

    Returns:
        list[list[float]]: The scaled and shifted coordinate lists, in the format of the input.
    """
    if keep_ratio:
        # A single scaling factor for both axes, then the offset of the centred padding
        start_y, start_x, _, _ = compute_letterbox_region(image_height=image_height, image_width=image_width,
                                                          image_size=image_size)
        scale = image_size / max(image_height, image_width)
        scale_x, scale_y, offset_x, offset_y = scale, scale, start_x, start_y
    else:
        # Independent horizontal and vertical scaling factors, without padding
        scale_x, scale_y, offset_x, offset_y = image_size / image_width, image_size / image_height, 0, 0

    # The x coordinates sit at the even positions of every list, the y coordinates at the odd ones
    return [[coordinate * scale_x + offset_x if coordinate_index % 2 == 0 else coordinate * scale_y + offset_y
             for coordinate_index, coordinate in enumerate(coordinates)] for coordinates in bounding_boxes]


def restore_boxes_to_original_image(bounding_boxes: np.ndarray, image_width: int, image_height: int,
                                    image_size: int, keep_ratio: bool, clip: bool = True) -> np.ndarray:
    """
    Maps boxes predicted on the preprocessed network input back to the original image, the exact inverse of
    'preprocess_boxes'. Called only by the evaluation of the instance segmentation at the original resolution
    (InstanceSegmentationTrainer.restore_detection_boxes), which pastes the masks of the detections into their
    unclipped boxes, a mask covering the whole box it was predicted in. No inference code of the repository calls it
    (deployment/run_inference.py measures the throughput only); a client of an exported detector would, to report the
    detections in the coordinates of the images it was given.

    Args:
        bounding_boxes (np.ndarray): The [N, 4] [x1, y1, x2, y2] boxes in network input pixels.
        image_width (int): The width of the original image.
        image_height (int): The height of the original image.
        image_size (int): The size of the square network input.
        keep_ratio (bool): Whether the image was resized keeping its aspect ratio and padded to a square.
        clip (bool): Whether to clip the boxes to the original image, a box predicted partly on the padding then
            stopping at the image border. Defaults to True.

    Returns:
        np.ndarray: The [N, 4] boxes in original image pixels, clipped to the image with clip.
    """
    restored_boxes = np.array(bounding_boxes, dtype=np.float64).reshape(-1, 4)

    if keep_ratio:
        # Remove the centred padding offset placed by compute_letterbox_region, then undo the single scaling factor
        scale = image_size / max(image_height, image_width)
        start_y, start_x, _, _ = compute_letterbox_region(image_height=image_height, image_width=image_width,
                                                          image_size=image_size)
        restored_boxes[:, 0::2] = (restored_boxes[:, 0::2] - start_x) / scale
        restored_boxes[:, 1::2] = (restored_boxes[:, 1::2] - start_y) / scale
    else:
        # Undo the independent horizontal and vertical scaling factors
        restored_boxes[:, 0::2] = restored_boxes[:, 0::2] / (image_size / image_width)
        restored_boxes[:, 1::2] = restored_boxes[:, 1::2] / (image_size / image_height)

    # Boxes predicted partly on the padding are clipped to the original image, unless they are kept whole
    if clip:
        restored_boxes[:, 0::2] = restored_boxes[:, 0::2].clip(min=0.0, max=image_width)
        restored_boxes[:, 1::2] = restored_boxes[:, 1::2].clip(min=0.0, max=image_height)
    return restored_boxes


def restore_prediction_to_original_image(prediction: torch.Tensor, original_height: int, original_width: int,
                                         keep_ratio: bool) -> torch.Tensor:
    """
    Brings a segmentation prediction made at the network resolution back to its original image, undoing the
    preprocessing of the images (preprocess_image): the centred padding added to keep the aspect ratio is cut off,
    then the prediction is resized to the size of the original image. A class map (integer tensor) is resized with
    the nearest interpolation on the pixel centres ("nearest-exact"), so that every pixel keeps one of the predicted
    classes and the map stays aligned with the bilinear probabilities and the image, and class probabilities (float
    tensor) bilinearly, so that the class of every original pixel is the argmax of its interpolated probabilities.
    Called only by the evaluation of the segmenter at the original resolution
    (SegmentationTrainer.accumulate_evaluation_batch), on the device of the prediction. deployment_utilities.py and
    InferenceSegmenter name it as the step a client of the exported segmenter runs on its outputs, no inference code of
    the repository calling it (deployment/run_inference.py measures the throughput only).

    Args:
        prediction (torch.Tensor): The [..., S, S] prediction of the network input, e.g. a [S, S] class map or [K, S, S]
            class probabilities. Without keep_ratio, any [..., h, w] prediction of the whole image.
        original_height (int): The height of the original image in pixels.
        original_width (int): The width of the original image in pixels.
        keep_ratio (bool): Whether the image was padded to keep its aspect ratio ('Data.image_settings.keep_ratio'),
            or stretched to the square network input.

    Returns:
        torch.Tensor: The [..., original_height, original_width] prediction of the original image, of the type of the
            input prediction.
    """
    # Part of the network input holding the image, centred between the padding exactly as preprocess_image places it
    image_part = prediction
    if keep_ratio:
        start_y, start_x, resized_height, resized_width = compute_letterbox_region(
            image_height=original_height, image_width=original_width, image_size=prediction.shape[-1])
        image_part = prediction[..., start_y:start_y + resized_height, start_x:start_x + resized_width]

    # Resize every leading channel at once, bilinearly for the probabilities and nearest on the pixel centres for the
    # class indices, the plain nearest shifting them by half an original pixel to the bottom right
    leading_shape = image_part.shape[:-2]
    stacked_channels = image_part.reshape(1, -1, image_part.shape[-2], image_part.shape[-1])
    if prediction.is_floating_point():
        restored_channels = torch.nn.functional.interpolate(input=stacked_channels,
                                                            size=(original_height, original_width), mode="bilinear",
                                                            align_corners=False)
    else:
        restored_channels = torch.nn.functional.interpolate(input=stacked_channels.float(),
                                                            size=(original_height, original_width),
                                                            mode="nearest-exact").to(dtype=prediction.dtype)
    return restored_channels.reshape(*leading_shape, original_height, original_width)


def preprocess_image_and_boxes(image: np.ndarray, bounding_boxes: list[list[float]], image_size: int,
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
