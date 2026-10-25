"""
Figures of the image size scene of the foundations tutorials (the data pipeline chapter), made at every build by
tutorials/figures/foundations/figures.py: a wide test image of 1920 x 1080 pixels holding a disc and a square, and the
two inputs of 512 x 512 pixels that preprocess_image of the repository gives for it, with the aspect ratio kept (resized
and centred between black padding) and without (stretched, the disc becoming an ellipse), so that the scene shows what
the data loaders really do. The sizes and the rows of padding are the facts its narration quotes.
"""

import cv2
import numpy

from pathlib import Path
from typing import Any, Dict

from vision_training.data.preprocessing import preprocess_image

# Size of the wide test image and of the network input, that of the synthetic configurations
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
INPUT_SIZE = 512

# Colours of the test image in BGR, the order of OpenCV: a slate background, an orange disc and a light square
BACKGROUND_COLOUR = (110, 90, 40)
DISC_COLOUR = (50, 140, 255)
SQUARE_COLOUR = (232, 232, 232)


def draw_test_image() -> numpy.ndarray:
    """
    Draws the wide test image of the scene: a disc and a square of the same size side by side on a plain background,
    two shapes whose deformation by a stretch is easy to see.

    Returns:
        numpy.ndarray: The BGR image of IMAGE_HEIGHT x IMAGE_WIDTH pixels.
    """
    image = numpy.full(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), fill_value=BACKGROUND_COLOUR, dtype=numpy.uint8)
    cv2.circle(img=image, center=(640, 540), radius=280, color=DISC_COLOUR, thickness=-1, lineType=cv2.LINE_AA)
    cv2.rectangle(img=image, pt1=(1000, 260), pt2=(1560, 820), color=SQUARE_COLOUR, thickness=-1)
    return image


def make_aspect_ratio_figures(output_folder: Path) -> Dict[str, Any]:
    """
    Writes the wide test image and its two network inputs, from the preprocessing of the repository, into the
    aspect_ratio folder of the series, and gives their sizes and the rows of black padding of the kept ratio.

    Args:
        output_folder (Path): The figures folder of the series.

    Returns:
        Dict[str, Any]: The facts of the figures, named aspect_ratio.*.

    Raises:
        OSError: If a figure cannot be written.
    """
    # Wide test image and its two inputs of the network
    test_image = draw_test_image()
    kept_ratio_input = preprocess_image(image=test_image, image_size=INPUT_SIZE, keep_ratio=True)
    stretched_input = preprocess_image(image=test_image, image_size=INPUT_SIZE, keep_ratio=False)

    # Figures written in the folder of the scene
    figure_folder = output_folder / "aspect_ratio"
    figure_folder.mkdir(parents=True, exist_ok=True)
    for figure_name, image in [(f"original_{IMAGE_WIDTH}x{IMAGE_HEIGHT}", test_image),
                               (f"keep_ratio_true_{INPUT_SIZE}", kept_ratio_input),
                               (f"keep_ratio_false_{INPUT_SIZE}", stretched_input)]:
        figure_path = figure_folder / f"{figure_name}.png"
        if not cv2.imwrite(filename=str(figure_path), img=image):
            raise OSError(f"The figure {figure_path} could not be written.")

    # Rows of the kept ratio input that are black padding only
    padding_rows = int((kept_ratio_input.reshape(INPUT_SIZE, -1).max(axis=1) == 0).sum())
    return {"aspect_ratio.original_width": IMAGE_WIDTH, "aspect_ratio.original_height": IMAGE_HEIGHT,
            "aspect_ratio.input_size": INPUT_SIZE, "aspect_ratio.padding_rows": padding_rows}
