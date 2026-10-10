"""
OpenCV drawing helpers: bounding boxes and anchors for the detection debugging scripts, the input viewer of the
trainers, the frame grids of the sequence tasks, the caption bars of the evaluation error images, the class colours,
mask overlays, error maps and sample grids of the semantic segmentation, the instance mask overlays of the instance
segmentation, and the selection grids with their probability heat map of the frame selection. Used by the trainers and
the scripts.
"""

import os
import sys
import cv2
import numpy as np
import torch

from typing import Dict, List, Optional, Tuple

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.models.sequence.frame_selection_decoding import NO_FRAME
from vision_training.utilities.os_utilities import print_color

# Height in pixels of the caption bar written above the error images of the evaluation
CAPTION_BAR_HEIGHT = 24

# BGR colours of the captions of the evaluation sample images, by whether the prediction of the sample is right
CORRECT_PREDICTION_COLOUR = (0, 200, 0)
WRONG_PREDICTION_COLOUR = (0, 0, 255)

# Segmentation drawings: opacity of the class colours painted over an image, BGR colour of the ignored pixels (padding,
# unlabelled), and BGR colours of the error map (pixels of a class given their class, pixels given a wrong class,
# pixels of a class missed as background)
MASK_OVERLAY_OPACITY = 0.5
IGNORED_PIXEL_COLOUR = (128, 128, 128)
ERROR_MAP_COLOURS = {"Correct": (0, 200, 0), "Wrong Class": (0, 0, 255), "Missed": (0, 255, 255)}

# Brightness kept by the grayscale image under the error map, and opacity of the error colours painted over it
ERROR_MAP_BACKGROUND_BRIGHTNESS = 0.5
ERROR_MAP_OPACITY = 0.7

# Number of classes (background left out) whose evenly spaced hues stay distinct (at least 30 degrees apart), beyond
# which every other class is darkened so that neighbouring hues can still be told apart, and the brightness of these
# darker classes
MAXIMUM_CLASSES_WITH_DISTINCT_HUES = 12
DARK_CLASS_COLOUR_VALUE = 160

# Height in pixels of the legend bar written below the segmentation sample grids, and gap between their panels
LEGEND_BAR_HEIGHT = 24
GRID_GAP = 4

# Frame selection drawings: side in pixels of the thumbnails of the frames, thumbnails per row, width of the selection
# borders, length of the dashes of the missed true frames, brightness kept by the frames rightly unselected, height of
# the rows and width of the class names of the probability heat map, and width of a heat map cell from which its
# probability is written in it
SELECTION_THUMBNAIL_SIZE = 160
SELECTION_GRID_COLUMNS = 8
SELECTION_BORDER_WIDTH = 4
SELECTION_DASH_LENGTH = 10
UNSELECTED_FRAME_BRIGHTNESS = 0.35
HEAT_MAP_ROW_HEIGHT = 24
HEAT_MAP_LABEL_WIDTH = 120
HEAT_MAP_MINIMUM_VALUE_WIDTH = 36

# BGR colours of the frame selection borders, those of the segmentation error maps: a frame chosen for its true class,
# a frame chosen for a class it does not hold, and a true frame left unchosen
SELECTION_COLOURS = {"Chosen right": ERROR_MAP_COLOURS["Correct"], "Chosen wrong": ERROR_MAP_COLOURS["Wrong Class"],
                     "True not chosen": ERROR_MAP_COLOURS["Missed"]}


def add_bounding_boxes(bounding_boxes: np.ndarray, image: np.ndarray, input_image_size: int,
                       color: Tuple[int, int, int] = (0, 255, 0), captions: Optional[List[str]] = None) -> np.ndarray:
    """
    Draws one or more bounding boxes onto the provided image canvas, optionally each with a caption (e.g. its class
    and score in the detection sample images of the evaluation).

    This function processes bounding box coordinates, clamps them to the image boundaries
    to prevent out-of-bounds drawing errors, and renders the rectangles using OpenCV.
    It supports both single bounding boxes [4] and batches of bounding boxes [N, 4].

    Args:
        bounding_boxes (np.ndarray): The [x_min, y_min, x_max, y_max] coordinates. Shape (4,) or (N, 4).
        image (np.ndarray): The image canvas on which to draw the bounding boxes.
        input_image_size (int): The spatial size (height and width) of the image to clamp coordinates.
        color (Tuple[int, int, int]): The BGR color tuple for the bounding boxes. Defaults to green (0, 255, 0).
        captions (Optional[List[str]]): The text written above every box in its colour, on a black background.
            Defaults to None, in which case the boxes are drawn without text.

    Returns:
        np.ndarray: The updated image canvas containing the newly drawn bounding boxes.
    """
    # Standardize shape to [N, 4] for consistent iteration
    if bounding_boxes.ndim == 1:
        bounding_boxes = bounding_boxes.reshape((1, 4))

    for box_index, box in enumerate(bounding_boxes):
        x_min, y_min, x_max, y_max = box

        # Ensure coordinates are within image boundaries for clean visualization
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(input_image_size, int(x_max))
        y_max = min(input_image_size, int(y_max))

        # Draw the bounding box on the canvas using the provided color
        cv2.rectangle(img=image, pt1=(x_min, y_min), pt2=(x_max, y_max), color=color, thickness=2)

        # Write the caption of the box above it, or inside its top when the box touches the top of the image
        if captions is not None:
            (caption_width, caption_height), _ = cv2.getTextSize(text=captions[box_index],
                                                                 fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                                                                 thickness=1)
            caption_bottom = y_min - 3 if y_min - caption_height - 6 >= 0 else y_min + caption_height + 3
            cv2.rectangle(img=image, pt1=(x_min, caption_bottom - caption_height - 3),
                          pt2=(x_min + caption_width + 4, caption_bottom + 3), color=(0, 0, 0), thickness=-1)
            cv2.putText(img=image, text=captions[box_index], org=(x_min + 2, caption_bottom),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=color, thickness=1)

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


def add_caption_bar(image: np.ndarray, caption: str,
                    caption_colour: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Writes a caption in a black bar above an image, e.g. the true and predicted classes of a misclassified test sample
    in the error images of the evaluation, or of a fixed test sample in the sample images, coloured by whether its
    prediction is right. The bar is widened to the caption when the image is narrower, the image being padded with
    gray on its right.

    Args:
        image (np.ndarray): The [H, W, 3] uint8 BGR image.
        caption (str): The text written in the bar.
        caption_colour (Tuple[int, int, int]): The BGR colour of the text. Defaults to white.

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
                color=caption_colour, thickness=1)
    return captioned_image


def build_class_colours(number_classes: int) -> np.ndarray:
    """
    Gives every class of the semantic segmentation a BGR colour for the mask overlays and the sample grids: black for
    the background class 0, which is left unpainted, and fully saturated colours evenly spaced on the hue circle for
    the other classes, in class order, which keeps every two classes as far apart as the hue circle allows. Beyond 12
    classes, every other class is darkened, so that neighbouring hues can still be told apart.

    Args:
        number_classes (int): The number of classes K, background included.

    Returns:
        np.ndarray: The [K, 3] uint8 BGR colour of every class.
    """
    # Evenly spaced hues of the classes, OpenCV hues being within [0, 180)
    number_foreground_classes = number_classes - 1
    class_ranks = np.arange(number_foreground_classes)
    hues = (class_ranks * 180.0 / max(number_foreground_classes, 1)).astype(np.uint8)
    saturations = np.full(shape=hues.shape, fill_value=255, dtype=np.uint8)

    # Full brightness, or every other class darkened when the hues get too close
    values = np.full(shape=hues.shape, fill_value=255, dtype=np.uint8)
    if number_foreground_classes > MAXIMUM_CLASSES_WITH_DISTINCT_HUES:
        values[class_ranks % 2 == 1] = DARK_CLASS_COLOUR_VALUE
    class_colours = cv2.cvtColor(src=np.stack([hues, saturations, values], axis=-1)[np.newaxis],
                                 code=cv2.COLOR_HSV2BGR)[0]
    return np.concatenate([np.zeros(shape=(1, 3), dtype=np.uint8), class_colours], axis=0)


def overlay_segmentation_mask(image: np.ndarray, class_map: np.ndarray, class_colours: np.ndarray) -> np.ndarray:
    """
    Paints a class map over its image for the input viewer and the evaluation sample grids of the segmentation: every
    pixel of a class is blended with the colour of its class, the background stays as it is and the ignored pixels
    (padding, unlabelled) are grayed.

    Args:
        image (np.ndarray): The [H, W, 3] uint8 BGR image.
        class_map (np.ndarray): The [H, W] class index of every pixel, IGNORED_LABEL_INDEX on the ignored ones.
        class_colours (np.ndarray): The [K, 3] uint8 BGR colour of every class (build_class_colours).

    Returns:
        np.ndarray: The [H, W, 3] uint8 BGR image with its class map painted over it.
    """
    painted_image = image.copy()
    class_pixels = class_map > 0
    painted_image[class_pixels] = class_colours[class_map[class_pixels]]
    painted_image[class_map == IGNORED_LABEL_INDEX] = IGNORED_PIXEL_COLOUR
    return cv2.addWeighted(src1=image, alpha=1.0 - MASK_OVERLAY_OPACITY, src2=painted_image,
                           beta=MASK_OVERLAY_OPACITY, gamma=0.0)


def overlay_instance_masks(image: np.ndarray, instance_masks: np.ndarray, labels: np.ndarray,
                           class_colours: np.ndarray) -> np.ndarray:
    """
    Paints the binary masks of the objects of an image in the colour of their class, for the input viewer and the
    evaluation sample grids of the instance segmentation: the class map of the masks (a later object covering an
    earlier one) is blended over the image like a segmentation, then the contour of every object is drawn, so that two
    touching objects of the same class stay apart.

    Args:
        image (np.ndarray): The [H, W, 3] uint8 BGR image.
        instance_masks (np.ndarray): The [N, H, W] binary masks of the objects.
        labels (np.ndarray): The [N] class index of every object.
        class_colours (np.ndarray): The [K, 3] uint8 BGR colour of every class (build_class_colours).

    Returns:
        np.ndarray: The [H, W, 3] uint8 BGR image with the masks painted over it.
    """
    # Class map of the masks, the objects painted one after the other
    class_map = np.zeros(shape=image.shape[:2], dtype=np.int64)
    for instance_mask, label in zip(instance_masks, labels):
        class_map[instance_mask > 0] = int(label)
    painted_image = np.ascontiguousarray(overlay_segmentation_mask(image=image, class_map=class_map,
                                                                   class_colours=class_colours))

    # Contour of every object in the colour of its class
    for instance_mask, label in zip(instance_masks, labels):
        contours, _ = cv2.findContours(image=(instance_mask > 0).astype(np.uint8), mode=cv2.RETR_EXTERNAL,
                                       method=cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image=painted_image, contours=contours, contourIdx=-1,
                         color=tuple(int(component) for component in class_colours[int(label)]), thickness=1)
    return painted_image


def build_segmentation_error_map(image: np.ndarray, class_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Shows where a predicted class map is right or wrong, over the darkened grayscale image, for the evaluation sample
    grids of the segmentation: the pixels of a class given their class in green, the pixels given a wrong class
    (another class, or a class where the truth is the background) in red, the pixels of a class missed as background
    in yellow, and the ignored pixels in gray, the background predicted as background staying dark.

    Args:
        image (np.ndarray): The [H, W, 3] uint8 BGR image.
        class_map (np.ndarray): The [H, W] predicted class index of every pixel.
        mask (np.ndarray): The [H, W] true class index of every pixel, IGNORED_LABEL_INDEX on the ignored ones.

    Returns:
        np.ndarray: The [H, W, 3] uint8 BGR error map.
    """
    # Darkened grayscale image, so that the error colours stand out while the objects stay recognisable
    grayscale_image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    background = cv2.cvtColor(src=(grayscale_image * ERROR_MAP_BACKGROUND_BRIGHTNESS).astype(np.uint8),
                              code=cv2.COLOR_GRAY2BGR)

    # Colour of every labelled pixel by the kind of its prediction, and of the ignored pixels
    labelled_pixels = mask != IGNORED_LABEL_INDEX
    wrong_pixels = labelled_pixels & (class_map != mask)
    painted_map = background.copy()
    painted_map[labelled_pixels & (class_map == mask) & (mask > 0)] = ERROR_MAP_COLOURS["Correct"]
    painted_map[wrong_pixels & (class_map > 0)] = ERROR_MAP_COLOURS["Wrong Class"]
    painted_map[wrong_pixels & (class_map == 0)] = ERROR_MAP_COLOURS["Missed"]
    painted_map[~labelled_pixels] = IGNORED_PIXEL_COLOUR
    return cv2.addWeighted(src1=background, alpha=1.0 - ERROR_MAP_OPACITY, src2=painted_map, beta=ERROR_MAP_OPACITY,
                           gamma=0.0)


def draw_legend_bar(width: int, legend_entries: List[Tuple[str, Tuple[int, int, int]]]) -> np.ndarray:
    """
    Draws a black bar naming coloured entries one after the other, each as a colour square followed by its name, for
    the legend of the segmentation sample grids. The entries that do not fit in the width are left out.

    Args:
        width (int): The width of the bar in pixels.
        legend_entries (List[Tuple[str, Tuple[int, int, int]]]): The name and BGR colour of every entry.

    Returns:
        np.ndarray: The [LEGEND_BAR_HEIGHT, width, 3] uint8 BGR legend bar.
    """
    legend_bar = np.zeros(shape=(LEGEND_BAR_HEIGHT, width, 3), dtype=np.uint8)
    horizontal_position = 4
    for entry_name, entry_colour in legend_entries:
        # Width of the entry: its colour square, a space and its name
        (name_width, _), _ = cv2.getTextSize(text=entry_name, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45,
                                             thickness=1)
        if horizontal_position + 18 + name_width > width:
            break
        cv2.rectangle(img=legend_bar, pt1=(horizontal_position, 6), pt2=(horizontal_position + 12, 18),
                      color=tuple(int(component) for component in entry_colour), thickness=-1)
        cv2.putText(img=legend_bar, text=entry_name, org=(horizontal_position + 18, 17),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, color=(255, 255, 255), thickness=1)
        horizontal_position += 18 + name_width + 16
    return legend_bar


def place_panels_side_by_side(panels: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Lays out panels of the same size side by side on a gray canvas, each titled in its top left corner, for the
    sample images of the evaluation (e.g. the input, true mask, predicted mask and error map of a segmentation sample,
    or the ground truth and predicted boxes of a detection sample).

    Args:
        panels (Dict[str, np.ndarray]): The [H, W, 3] uint8 BGR panels keyed by title, in display order.

    Returns:
        np.ndarray: The [H, number of panels x W + gaps, 3] uint8 BGR grid, 4 gray pixels between the panels.
    """
    image_height, image_width = next(iter(panels.values())).shape[:2]
    grid = np.full(shape=(image_height, len(panels) * (image_width + GRID_GAP) - GRID_GAP, 3), fill_value=128,
                   dtype=np.uint8)
    for panel_index, (panel_title, panel) in enumerate(panels.items()):
        # Title written on a black block in the top left corner of a copy of the panel
        titled_panel = panel.copy()
        (title_width, _), _ = cv2.getTextSize(text=panel_title, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45,
                                              thickness=1)
        cv2.rectangle(img=titled_panel, pt1=(0, 0), pt2=(title_width + 8, 20), color=(0, 0, 0), thickness=-1)
        cv2.putText(img=titled_panel, text=panel_title, org=(4, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.45, color=(255, 255, 255), thickness=1)

        # Panel placed in its column of the canvas
        left = panel_index * (image_width + GRID_GAP)
        grid[:, left:left + image_width] = titled_panel
    return grid


def build_segmentation_sample_grid(image: np.ndarray, mask: np.ndarray, class_map: np.ndarray,
                                   class_names: List[str], class_colours: np.ndarray, caption: str) -> np.ndarray:
    """
    Lays out one evaluated test image of the segmentation in four titled panels side by side, saved and sent to
    TensorBoard at every evaluation so that the predictions of the same images can be followed over the training: the
    input, its true mask and its predicted mask painted over it, and the error map. A caption bar names the sample
    above the panels, and a legend bar below them names the classes painted in the image and the error colours.

    Args:
        image (np.ndarray): The [H, W, 3] uint8 BGR network input.
        mask (np.ndarray): The [H, W] true class index of every pixel, IGNORED_LABEL_INDEX on the ignored ones.
        class_map (np.ndarray): The [H, W] predicted class index of every pixel.
        class_names (List[str]): The name of every class, in class index order.
        class_colours (np.ndarray): The [K, 3] uint8 BGR colour of every class (build_class_colours).
        caption (str): The text of the caption bar, e.g. the pixel accuracy of the sample.

    Returns:
        np.ndarray: The [24 + H + 24, 4 x W + 3 x 4, 3] uint8 BGR grid.
    """
    grid = place_panels_side_by_side(panels={
        "Input": image,
        "Ground Truth": overlay_segmentation_mask(image=image, class_map=mask, class_colours=class_colours),
        "Prediction": overlay_segmentation_mask(image=image, class_map=class_map, class_colours=class_colours),
        "Errors": build_segmentation_error_map(image=image, class_map=class_map, mask=mask)})

    # Legend of the classes of the true and predicted masks (background left out), then of the error colours
    painted_class_indices = sorted(set(np.unique(ar=mask[mask > 0]).tolist()) |
                                   set(np.unique(ar=class_map[class_map > 0]).tolist()))
    legend_entries = [(class_names[class_index], tuple(class_colours[class_index].tolist()))
                      for class_index in painted_class_indices] + list(ERROR_MAP_COLOURS.items())
    legend_bar = draw_legend_bar(width=grid.shape[1], legend_entries=legend_entries)
    return add_caption_bar(image=np.concatenate([grid, legend_bar], axis=0), caption=caption)


def build_detection_sample_grid(image: np.ndarray, ground_truth_boxes: np.ndarray, ground_truth_labels: np.ndarray,
                                predicted_boxes: np.ndarray, predicted_labels: np.ndarray,
                                predicted_scores: np.ndarray, class_names: Dict[int, str], class_colours: np.ndarray,
                                caption: str, ground_truth_masks: Optional[np.ndarray] = None,
                                predicted_masks: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Lays out one evaluated test image of the object detection in two titled panels side by side, saved and sent to
    TensorBoard at every evaluation so that the detections of the same images can be followed over the training: the
    ground truth boxes captioned with their class, and the detections captioned with their class and score, every box
    in the colour of its class. A caption bar names the sample above the panels, and a legend bar below them names the
    classes drawn. For the instance segmentation, the masks of the objects are painted under their boxes.

    Args:
        image (np.ndarray): The [H, W, 3] uint8 BGR network input.
        ground_truth_boxes (np.ndarray): The [N, 4] true boxes in network input pixels (x_min, y_min, x_max, y_max).
        ground_truth_labels (np.ndarray): The [N] class index of every true box.
        predicted_boxes (np.ndarray): The [M, 4] detected boxes in network input pixels.
        predicted_labels (np.ndarray): The [M] class index of every detection.
        predicted_scores (np.ndarray): The [M] class probability of every detection.
        class_names (Dict[int, str]): The name of every class index.
        class_colours (np.ndarray): The [K, 3] uint8 BGR colour of every class (build_class_colours).
        caption (str): The text of the caption bar, e.g. the number of true boxes and of detections.
        ground_truth_masks (Optional[np.ndarray]): The [N, H, W] binary masks of the true objects, None for the object
            detection. Defaults to None.
        predicted_masks (Optional[np.ndarray]): The [M, H, W] binary masks of the detections, None for the object
            detection. Defaults to None.

    Returns:
        np.ndarray: The [24 + H + 24, 2 x W + 4, 3] uint8 BGR grid.
    """
    image_size = max(image.shape[:2])
    ground_truth_panel, prediction_panel = image.copy(), image.copy()

    # Masks of the instance segmentation painted first, under the boxes
    if ground_truth_masks is not None:
        ground_truth_panel = overlay_instance_masks(image=ground_truth_panel, instance_masks=ground_truth_masks,
                                                    labels=ground_truth_labels, class_colours=class_colours)
    if predicted_masks is not None:
        prediction_panel = overlay_instance_masks(image=prediction_panel, instance_masks=predicted_masks,
                                                  labels=predicted_labels, class_colours=class_colours)
    drawn_class_indices = sorted(set(ground_truth_labels.tolist()) | set(predicted_labels.tolist()))

    # Boxes of every class drawn in the colour of the class, the true ones with their class, the detections with their
    # class and score
    for class_index in drawn_class_indices:
        class_colour = tuple(int(component) for component in class_colours[class_index])
        ground_truth_mask, prediction_mask = ground_truth_labels == class_index, predicted_labels == class_index
        add_bounding_boxes(bounding_boxes=ground_truth_boxes[ground_truth_mask], image=ground_truth_panel,
                           input_image_size=image_size, color=class_colour,
                           captions=[class_names[class_index]] * int(ground_truth_mask.sum()))
        add_bounding_boxes(bounding_boxes=predicted_boxes[prediction_mask], image=prediction_panel,
                           input_image_size=image_size, color=class_colour,
                           captions=[f"{class_names[class_index]} {score:.2f}"
                                     for score in predicted_scores[prediction_mask]])
    grid = place_panels_side_by_side(panels={"Ground Truth": ground_truth_panel, "Prediction": prediction_panel})

    # Legend of the classes of the true and detected boxes
    legend_entries = [(class_names[class_index], tuple(class_colours[class_index].tolist()))
                      for class_index in drawn_class_indices]
    legend_bar = draw_legend_bar(width=grid.shape[1], legend_entries=legend_entries)
    return add_caption_bar(image=np.concatenate([grid, legend_bar], axis=0), caption=caption)


def draw_dashed_rectangle(image: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int],
                          colour: Tuple[int, int, int], thickness: int, dash_length: int) -> None:
    """
    Draws the outline of a rectangle as dashes, the border of the true frames left unchosen in the selection grids of
    the frame selection, which stays distinguishable from the solid border of a chosen frame drawn on the same frame.

    Args:
        image (np.ndarray): The [H, W, 3] uint8 BGR image drawn in place.
        top_left (Tuple[int, int]): The (x, y) top left corner.
        bottom_right (Tuple[int, int]): The (x, y) bottom right corner.
        colour (Tuple[int, int, int]): The BGR colour of the dashes.
        thickness (int): The thickness of the dashes in pixels.
        dash_length (int): The length of a dash, and of the gap after it, in pixels.
    """
    left, top = top_left
    right, bottom = bottom_right
    for side_start, side_end in [((left, top), (right, top)), ((right, top), (right, bottom)),
                                 ((right, bottom), (left, bottom)), ((left, bottom), (left, top))]:
        # Dashes along the side, every other piece of dash_length pixels
        side_length = max(abs(side_end[0] - side_start[0]), abs(side_end[1] - side_start[1]), 1)
        for dash_start in range(0, side_length, 2 * dash_length):
            dash_end = min(dash_start + dash_length, side_length)
            start_point = (int(round(side_start[0] + (side_end[0] - side_start[0]) * dash_start / side_length)),
                           int(round(side_start[1] + (side_end[1] - side_start[1]) * dash_start / side_length)))
            end_point = (int(round(side_start[0] + (side_end[0] - side_start[0]) * dash_end / side_length)),
                         int(round(side_start[1] + (side_end[1] - side_start[1]) * dash_end / side_length)))
            cv2.line(img=image, pt1=start_point, pt2=end_point, color=colour, thickness=thickness)


def draw_selection_thumbnail(frame: np.ndarray, frame_index: int,
                             chosen_entry: Optional[Tuple[str, Tuple[int, int, int]]],
                             missed_entry: Optional[Tuple[str, Tuple[int, int, int]]], dimmed: bool) -> np.ndarray:
    """
    Draws one frame of a sequence in the selection grid of the frame selection: a numbered thumbnail with a solid
    border when it was chosen (green for its true class, red for a class it does not hold), a dashed yellow border
    when it is a true frame left unchosen (inside the solid border when both apply), each labelled at the bottom with
    its class and probability, and dimmed when it was rightly left unselected so that the eye goes to the selections.

    Args:
        frame (np.ndarray): The [H, W, 3] uint8 BGR frame.
        frame_index (int): The position of the frame in its sequence, written in its top left corner.
        chosen_entry (Optional[Tuple[str, Tuple[int, int, int]]]): The label and BGR colour of the class the frame was
            chosen for, or None.
        missed_entry (Optional[Tuple[str, Tuple[int, int, int]]]): The label and BGR colour of the true class the frame
            was not chosen for, or None.
        dimmed (bool): Whether the frame is darkened, being rightly unselected.

    Returns:
        np.ndarray: The [SELECTION_THUMBNAIL_SIZE, SELECTION_THUMBNAIL_SIZE, 3] uint8 BGR thumbnail.
    """
    size, border_width = SELECTION_THUMBNAIL_SIZE, SELECTION_BORDER_WIDTH
    thumbnail = cv2.resize(src=frame, dsize=(size, size), interpolation=cv2.INTER_AREA)
    if dimmed:
        thumbnail = (thumbnail.astype(np.float32) * UNSELECTED_FRAME_BRIGHTNESS).astype(np.uint8)

    # Solid border of a chosen frame on the edge, dashed border of a missed true frame inside it
    if chosen_entry is not None:
        cv2.rectangle(img=thumbnail, pt1=(border_width // 2, border_width // 2),
                      pt2=(size - 1 - border_width // 2, size - 1 - border_width // 2), color=chosen_entry[1],
                      thickness=border_width)
    if missed_entry is not None:
        inset = border_width // 2 + (border_width + 2 if chosen_entry is not None else 0)
        draw_dashed_rectangle(image=thumbnail, top_left=(inset, inset),
                              bottom_right=(size - 1 - inset, size - 1 - inset), colour=missed_entry[1],
                              thickness=border_width, dash_length=SELECTION_DASH_LENGTH)

    # Number of the frame in the top left corner, and the label of every border at the bottom
    frame_text = str(frame_index)
    cv2.rectangle(img=thumbnail, pt1=(border_width, border_width),
                  pt2=(border_width + 8 + 9 * len(frame_text), border_width + 18), color=(0, 0, 0), thickness=-1)
    cv2.putText(img=thumbnail, text=frame_text, org=(border_width + 4, border_width + 14),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, color=(255, 255, 255), thickness=1)
    labelled_entries = [entry for entry in [missed_entry, chosen_entry] if entry is not None]
    for line_index, (entry_text, entry_colour) in enumerate(labelled_entries):
        (text_width, _), _ = cv2.getTextSize(text=entry_text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45,
                                             thickness=1)
        line_bottom = size - 2 * border_width - 2 - line_index * 18
        cv2.rectangle(img=thumbnail, pt1=(2 * border_width, line_bottom - 15),
                      pt2=(min(size - 2 * border_width, 2 * border_width + text_width + 8), line_bottom + 3),
                      color=(0, 0, 0), thickness=-1)
        cv2.putText(img=thumbnail, text=entry_text, org=(2 * border_width + 4, line_bottom - 2),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, color=entry_colour, thickness=1)
    return thumbnail


def draw_probability_heat_map(probabilities: np.ndarray, frame_labels: np.ndarray, selected_frames: np.ndarray,
                              wanted_class_indices: List[int], class_names: List[str], width: int) -> np.ndarray:
    """
    Draws the probability of every frame (columns) for every wanted class (rows) of a sequence under its selection
    grid, so that the frames competing for a class can be compared: every cell coloured by its probability (viridis,
    dark for 0 and yellow for 1, the value written in wide enough cells), the frames chosen for the class marked with a
    dot and its true frames outlined in white.

    Args:
        probabilities (np.ndarray): The [T, K] class probabilities of every frame.
        frame_labels (np.ndarray): The [T] true class of every frame.
        selected_frames (np.ndarray): The [W, N] chosen frames of every wanted class, NO_FRAME for no frame.
        wanted_class_indices (List[int]): The class indices of the wanted classes, one per row.
        class_names (List[str]): The name of every class, in class index order.
        width (int): The width of the heat map, the one of the grid above it.

    Returns:
        np.ndarray: The [(W + 1) x HEAT_MAP_ROW_HEIGHT + GRID_GAP, width, 3] uint8 BGR heat map.
    """
    number_frames, row_height = probabilities.shape[0], HEAT_MAP_ROW_HEIGHT
    cell_width = max(1, (width - HEAT_MAP_LABEL_WIDTH - GRID_GAP) // number_frames)
    heat_map = np.zeros(shape=((len(wanted_class_indices) + 1) * row_height + GRID_GAP, width, 3), dtype=np.uint8)

    # Frame numbers above the columns, one every few columns when the cells are narrow
    number_step = max(1, int(np.ceil(16 / cell_width)))
    for frame_index in range(0, number_frames, number_step):
        cv2.putText(img=heat_map, text=str(frame_index), org=(HEAT_MAP_LABEL_WIDTH + frame_index * cell_width + 2, 16),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255), thickness=1)

    for wanted_index, class_index in enumerate(wanted_class_indices):
        row_top = (wanted_index + 1) * row_height
        cv2.putText(img=heat_map, text=class_names[class_index][:14], org=(4, row_top + 16),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, color=(255, 255, 255), thickness=1)

        # Cell of every frame coloured by its probability for the class
        class_probabilities = probabilities[:, class_index]
        cell_colours = cv2.applyColorMap(src=np.round(class_probabilities * 255.0).astype(np.uint8)[None, :],
                                         colormap=cv2.COLORMAP_VIRIDIS)[0]
        chosen_frames = set(selected_frames[wanted_index][selected_frames[wanted_index] != NO_FRAME].tolist())
        for frame_index in range(number_frames):
            cell_left = HEAT_MAP_LABEL_WIDTH + frame_index * cell_width
            cell_top_left = (cell_left, row_top + 1)
            cell_bottom_right = (cell_left + cell_width - 2, row_top + row_height - 2)
            cv2.rectangle(img=heat_map, pt1=cell_top_left, pt2=cell_bottom_right,
                          color=tuple(int(component) for component in cell_colours[frame_index]), thickness=-1)

            # Probability written in the wide cells, dark on the bright colours
            if cell_width >= HEAT_MAP_MINIMUM_VALUE_WIDTH:
                text_colour = (0, 0, 0) if class_probabilities[frame_index] > 0.6 else (255, 255, 255)
                cv2.putText(img=heat_map, text=f"{class_probabilities[frame_index]:.2f}",
                            org=(cell_left + 2, row_top + 16), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.35,
                            color=text_colour, thickness=1)

            # True frames of the class outlined, chosen frames marked with a dot
            if frame_labels[frame_index] == class_index:
                cv2.rectangle(img=heat_map, pt1=cell_top_left, pt2=cell_bottom_right, color=(255, 255, 255),
                              thickness=2)
            if frame_index in chosen_frames:
                dot_center = (cell_left + cell_width // 2 - 1, row_top + row_height // 2)
                dot_radius = max(2, min(cell_width, row_height) // 5)
                cv2.circle(img=heat_map, center=dot_center, radius=dot_radius + 1, color=(255, 255, 255), thickness=-1)
                cv2.circle(img=heat_map, center=dot_center, radius=dot_radius, color=(0, 0, 0), thickness=-1)
    return heat_map


def build_frame_selection_image(frames: np.ndarray, probabilities: np.ndarray, frame_labels: np.ndarray,
                                selected_frames: np.ndarray, wanted_class_indices: List[int],
                                unselected_class_index: int, class_names: List[str]) -> np.ndarray:
    """
    Draws one evaluated test sequence of the frame selection, for its sample images and error images: the whole
    sequence as a grid of numbered thumbnails (8 per row) with the borders of the selection (solid green for a frame
    chosen for its true class, solid red for a frame chosen for a class it does not hold, dashed yellow for a true frame
    left unchosen, the rightly unselected frames dimmed), the probability heat map of every wanted class below it, a
    caption bar giving the number of right selections (green when all are right, red otherwise) and a legend bar.

    Args:
        frames (np.ndarray): The [T, H, W, 3] uint8 BGR frames of the sequence.
        probabilities (np.ndarray): The [T, K] class probabilities of every frame.
        frame_labels (np.ndarray): The [T] true class of every frame.
        selected_frames (np.ndarray): The [W, N] frames chosen for every wanted class, NO_FRAME for no frame.
        wanted_class_indices (List[int]): The class indices of the wanted classes, one per row of the selection.
        unselected_class_index (int): The class index of the unselected frames.
        class_names (List[str]): The name of every class, in class index order.

    Returns:
        np.ndarray: The [H', W', 3] uint8 BGR image of the sequence.
    """
    # Class each chosen frame was chosen for
    chosen_classes = {}
    for wanted_index, class_index in enumerate(wanted_class_indices):
        for frame_index in selected_frames[wanted_index][selected_frames[wanted_index] != NO_FRAME].tolist():
            chosen_classes[frame_index] = class_index

    # Thumbnail of every frame with the borders of its selection
    thumbnails, number_right_selections = [], 0
    for frame_index in range(len(frames)):
        true_class_index = int(frame_labels[frame_index])
        chosen_class_index = chosen_classes[frame_index] if frame_index in chosen_classes else None
        chosen_entry, missed_entry = None, None
        if chosen_class_index is not None:
            chosen_right = chosen_class_index == true_class_index
            number_right_selections += int(chosen_right)
            chosen_entry = (f"{class_names[chosen_class_index]} {probabilities[frame_index, chosen_class_index]:.2f}",
                            SELECTION_COLOURS["Chosen right" if chosen_right else "Chosen wrong"])
        if true_class_index != unselected_class_index and chosen_class_index != true_class_index:
            missed_entry = (f"{class_names[true_class_index]} {probabilities[frame_index, true_class_index]:.2f}",
                            SELECTION_COLOURS["True not chosen"])
        thumbnails.append(draw_selection_thumbnail(
            frame=frames[frame_index], frame_index=frame_index, chosen_entry=chosen_entry, missed_entry=missed_entry,
            dimmed=true_class_index == unselected_class_index and chosen_class_index is None))

    # Thumbnails in reading order on a gray canvas, with a gap around every thumbnail
    size, number_columns = SELECTION_THUMBNAIL_SIZE, min(SELECTION_GRID_COLUMNS, len(frames))
    number_rows = (len(frames) + number_columns - 1) // number_columns
    grid = np.full(shape=(number_rows * (size + GRID_GAP) + GRID_GAP, number_columns * (size + GRID_GAP) + GRID_GAP, 3),
                   fill_value=128, dtype=np.uint8)
    for frame_index, thumbnail in enumerate(thumbnails):
        top = GRID_GAP + (frame_index // number_columns) * (size + GRID_GAP)
        left = GRID_GAP + (frame_index % number_columns) * (size + GRID_GAP)
        grid[top:top + size, left:left + size] = thumbnail

    # Heat map and legend under the grid, caption above it
    heat_map = draw_probability_heat_map(probabilities=probabilities, frame_labels=frame_labels,
                                         selected_frames=selected_frames, wanted_class_indices=wanted_class_indices,
                                         class_names=class_names, width=grid.shape[1])
    legend_bar = draw_legend_bar(width=grid.shape[1], legend_entries=list(SELECTION_COLOURS.items()) +
                                 [("Map: dot chosen, box true", (255, 255, 255))])
    number_selections = int(selected_frames.size)
    return add_caption_bar(image=np.concatenate([grid, heat_map, legend_bar], axis=0),
                           caption=f"{number_right_selections}/{number_selections} selections right",
                           caption_colour=CORRECT_PREDICTION_COLOUR if number_right_selections == number_selections
                           else WRONG_PREDICTION_COLOUR)
