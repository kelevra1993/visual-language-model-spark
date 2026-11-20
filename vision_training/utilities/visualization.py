"""
OpenCV drawing helpers: the conversion of the network inputs into the arrays every drawing starts from, bounding boxes
of the detection sample and error images and anchors, the input viewer of the trainers (every image next to a black
information panel listing its target, with tables such as the objects per class, and its augmentations, the windows
centred on the screen, also saved as the input samples of every training launch), the frame grids of the sequence tasks,
the caption bars of the evaluation error images, the class colours, mask overlays, error maps and sample grids of the
semantic segmentation, the instance mask overlays of the instance segmentation, the keypoints of the objects of an image
with their skeleton and the keypoint sample grids of the keypoint detection, the selection grids with their probability
heat map of the frame selection, and the batch mosaic of the self-supervised input viewer. Used by the trainers.
"""

import importlib
import os
import sys
import cv2
import numpy as np
import torch

from functools import lru_cache
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.models.sequence.frame_selection_decoding import NO_FRAME
from vision_training.utilities.os_utilities import print_color

# Height in pixels of the caption bar written above the error images of the evaluation
CAPTION_BAR_HEIGHT = 24

# Information panel of the input viewer: its width and text margin in pixels, the font scale and height of its title
# and of its lines (the augmentations of the sample), the stroke thickness of its lines, sized to be read on a 4K
# screen at full scale, and the BGR colours of the title, of the group lines, of the indented lines and of the hint
INFORMATION_PANEL_WIDTH = 1000
PANEL_MARGIN = 20
PANEL_TITLE_FONT_SCALE = 1.2
PANEL_TITLE_HEIGHT = 56
PANEL_LINE_FONT_SCALE = 0.85
PANEL_LINE_HEIGHT = 36
PANEL_LINE_THICKNESS = 2
PANEL_TITLE_COLOUR = (255, 255, 255)
PANEL_GROUP_COLOUR = (0, 255, 0)
PANEL_LINE_COLOUR = (220, 220, 220)
PANEL_HINT_COLOUR = (150, 150, 150)
NAVIGATION_HINT = "Press SPACE For Next Image, Or ENTER To Quit"

# Two-row tables of the information panel (the objects of a detection sample per class): the height of a row, the
# padding around the text of a cell, the gap above a table and between two row pairs of a wrapped table, the grid colour
PANEL_TABLE_ROW_HEIGHT = 44
PANEL_TABLE_CELL_PADDING = 12
PANEL_TABLE_GAP = 10
PANEL_TABLE_GRID_COLOUR = (110, 110, 110)


class PanelTable(NamedTuple):
    """
    A two-row table written in the information panel of the input viewer among its lines, e.g. the objects of a
    detection sample: the titles of its two rows, written in a left column ("Class", "Count"), and its columns, each
    holding the text of both rows (a class name over its number of objects).
    """
    row_titles: Tuple[str, str]
    columns: List[Tuple[str, str]]


class ViewerWindow(NamedTuple):
    """
    One sample drawn as a window of the input viewer ('view_input_images'), and as an image of the input samples
    written at the start of every training launch ('<run>/Input_Samples/', trainers/input_samples.py): built by the
    build_input_windows hook of every task trainer and drawn by build_viewer_window. It holds the image fed to the
    network (its masks, keypoints or frame grid already painted on it), its boxes captioned with their classes, the
    title of its panel, the panel lines describing its target, and those describing its augmentations, kept apart so
    that the record of the input samples (samples.json) lists them apart.
    """
    image: torch.Tensor
    bounding_boxes: Optional[torch.Tensor]
    labels: Optional[torch.Tensor]
    label_names: Optional[Dict[int, str]]
    title: str
    target_lines: List[Union[str, PanelTable]]
    augmentation_lines: List[str]
    display_input_size: int


# Placement of the input viewer windows: the space in pixels between two windows, the height kept for the title bar of
# the window manager and the toolbar of the OpenCV window above every window, and the share of the screen the windows
# may cover before their content is scaled down
VIEWER_WINDOW_SPACING = 20
VIEWER_TITLE_BAR_HEIGHT = 80
VIEWER_SCREEN_FRACTION = 0.92

# Batch mosaic of the self-supervised input viewer: the side in pixels of the thumbnails of the images, the thumbnails
# per row (a block of more images going on several rows), the gray space in pixels around every block, the font scale
# of the image captions, and the navigation hint of its header
BATCH_MOSAIC_THUMBNAIL_SIZE = 128
BATCH_MOSAIC_IMAGES_PER_ROW = 16
BATCH_MOSAIC_BLOCK_SPACING = 16
BATCH_MOSAIC_CAPTION_FONT_SCALE = 0.5
BATCH_NAVIGATION_HINT = "Press SPACE For The Views Of Its Images, Or ENTER To Quit"

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

# Detection error images: pixels between a box in error and its dashed outline, length of the dashes, and the name of
# the outline in the legend
ERROR_OUTLINE_MARGIN = 4
ERROR_OUTLINE_DASH_LENGTH = 6
ERROR_OUTLINE_LEGEND = "Error"

# Keypoint drawings: BGR colours of the keypoints by the side of the object their name gives (blue left, orange right,
# yellow for the keypoints of neither side), radius in pixels of their discs, and thickness of the skeleton lines,
# drawn in the colour of the class of their object
KEYPOINT_SIDE_COLOURS = {"Left": (255, 160, 0), "Right": (0, 140, 255), "Centre": (0, 255, 255)}
KEYPOINT_RADIUS = 4
SKELETON_THICKNESS = 2

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


def convert_image_to_array(image: torch.Tensor) -> np.ndarray:
    """
    Turns a network input into the array OpenCV draws on, the first step of the sample images, error images, frame
    grids and input viewer of every trainer: the image is moved to the CPU, clamped to [0, 255] (the augmentations
    and views can leave values just outside, which a plain cast would wrap around), cast to uint8 and laid out with
    its channels last in contiguous memory, which the OpenCV drawing functions need.

    Args:
        image (torch.Tensor): The [3, H, W] BGR image in [0, 255], float or uint8, or the [T, 3, H, W] frames of a
            sequence, on any device.

    Returns:
        np.ndarray: The [H, W, 3] uint8 BGR image, or the [T, H, W, 3] frames.
    """
    uint8_image = image.detach().cpu().clamp(min=0, max=255).to(dtype=torch.uint8)
    return np.ascontiguousarray(a=uint8_image.movedim(source=-3, destination=-1).numpy())


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
        x_minimum, y_minimum, x_maximum, y_maximum = box

        # Ensure coordinates are within image boundaries for clean visualization
        x_minimum = max(0, int(x_minimum))
        y_minimum = max(0, int(y_minimum))
        x_maximum = min(input_image_size, int(x_maximum))
        y_maximum = min(input_image_size, int(y_maximum))

        # Draw the bounding box on the canvas using the provided color
        cv2.rectangle(img=image, pt1=(x_minimum, y_minimum), pt2=(x_maximum, y_maximum), color=color, thickness=2)

        # Write the caption of the box above it, or inside its top when the box touches the top of the image
        if captions is not None:
            (caption_width, caption_height), _ = cv2.getTextSize(text=captions[box_index],
                                                                 fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                                                                 thickness=1)
            caption_bottom = y_minimum - 3 if y_minimum - caption_height - 6 >= 0 else y_minimum + caption_height + 3
            cv2.rectangle(img=image, pt1=(x_minimum, caption_bottom - caption_height - 3),
                          pt2=(x_minimum + caption_width + 4, caption_bottom + 3), color=(0, 0, 0), thickness=-1)
            cv2.putText(img=image, text=captions[box_index], org=(x_minimum + 2, caption_bottom),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=color, thickness=1)

    return image


@lru_cache(maxsize=1)
def get_screen_size() -> Optional[Tuple[int, int]]:
    """
    Reads the size of the screen, for the input viewer ('view_input_images') to centre its windows on it. tkinter is
    loaded here only, the viewer being the only part of the project using it, and the size is read once per run.

    Returns:
        Optional[Tuple[int, int]]: The width and height of the screen in pixels, or None when tkinter is missing or
            cannot reach the display, the viewer then placing its windows from the top left of the screen.
    """
    try:
        tkinter_module = importlib.import_module(name="tkinter")
    except ImportError:
        return None

    # A hidden root window gives the size of the screen it opens on, and is closed right away
    try:
        root_window = tkinter_module.Tk()
    except tkinter_module.TclError:
        return None
    root_window.withdraw()
    screen_size = (root_window.winfo_screenwidth(), root_window.winfo_screenheight())
    root_window.destroy()
    return screen_size


def wrap_panel_line(line: str, maximum_width: int) -> List[str]:
    """
    Splits a line of the information panel of the input viewer into the lines fitting the panel, word by word, e.g.
    the line of a frame of a sequence listing every photometric augmentation drawn for it. The continuation lines keep
    the indentation of the line, four spaces deeper.

    Args:
        line (str): The line, its leading spaces giving its indentation.
        maximum_width (int): The width in pixels the text must fit in.

    Returns:
        List[str]: The wrapped lines, the line itself when it fits.
    """
    indentation = line[:len(line) - len(line.lstrip(" "))]
    wrapped_lines, current_line = [], indentation

    # Add the words one by one, starting a new line when a word does not fit, unless the line holds no word yet
    for word in line.split():
        candidate_line = f"{current_line} {word}" if current_line.strip() else f"{current_line}{word}"
        (text_width, _), _ = cv2.getTextSize(text=candidate_line, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                             fontScale=PANEL_LINE_FONT_SCALE, thickness=PANEL_LINE_THICKNESS)
        if text_width > maximum_width and current_line.strip():
            wrapped_lines.append(current_line)
            current_line = f"{indentation}    {word}"
        else:
            current_line = candidate_line
    wrapped_lines.append(current_line)
    return wrapped_lines


def measure_panel_text_width(text: str) -> int:
    """
    Gives the width of a text written in the font of the information panel lines, to size the cells of its tables.

    Args:
        text (str): The text.

    Returns:
        int: Its width in pixels.
    """
    (text_width, _), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=PANEL_LINE_FONT_SCALE,
                                         thickness=PANEL_LINE_THICKNESS)
    return text_width


def split_panel_table(table: PanelTable, maximum_width: int) -> Tuple[int, List[List[Tuple[str, str, int]]]]:
    """
    Lays out a table of the information panel of the input viewer: the width of its row title column, and its columns
    grouped into the row pairs fitting the panel, a new pair starting below when the next column does not fit (e.g.
    an image holding a dozen classes with long names). A column wider than the panel alone keeps a pair of its own.

    Args:
        table (PanelTable): The table.
        maximum_width (int): The width in pixels the table must fit in.

    Returns:
        Tuple[int, List[List[Tuple[str, str, int]]]]: The width of the row title column, and the row pairs, each a list
            of columns (top text, bottom text, column width), none for a table without columns.
    """
    title_width = max(measure_panel_text_width(text=row_title) for row_title in table.row_titles) + \
        2 * PANEL_TABLE_CELL_PADDING
    row_pairs, current_pair, current_width = [], [], title_width

    # Add the columns one by one, sized on their longer text, starting a new pair when a column does not fit
    for top_text, bottom_text in table.columns:
        column_width = max(measure_panel_text_width(text=top_text), measure_panel_text_width(text=bottom_text)) + \
            2 * PANEL_TABLE_CELL_PADDING
        if current_pair and current_width + column_width > maximum_width:
            row_pairs.append(current_pair)
            current_pair, current_width = [], title_width
        current_pair.append((top_text, bottom_text, column_width))
        current_width += column_width
    if current_pair:
        row_pairs.append(current_pair)
    return title_width, row_pairs


def draw_panel_table_cell(panel: np.ndarray, box: Tuple[int, int, int, int], text: str,
                          colour: Tuple[int, int, int]) -> None:
    """
    Draws one cell of a table of the information panel: its gray grid rectangle and its text centred in it.

    Args:
        panel (np.ndarray): The [H, W, 3] uint8 BGR panel, drawn on in place.
        box (Tuple[int, int, int, int]): The left, top, right and bottom of the cell in pixels.
        text (str): The text of the cell.
        colour (Tuple[int, int, int]): The BGR colour of the text.
    """
    left, top, right, bottom = box
    cv2.rectangle(img=panel, pt1=(left, top), pt2=(right, bottom), color=PANEL_TABLE_GRID_COLOUR, thickness=1)
    (text_width, text_height), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                   fontScale=PANEL_LINE_FONT_SCALE, thickness=PANEL_LINE_THICKNESS)
    text_origin = (left + (right - left - text_width) // 2, top + (bottom - top + text_height) // 2)
    cv2.putText(img=panel, text=text, org=text_origin, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=PANEL_LINE_FONT_SCALE, color=colour, thickness=PANEL_LINE_THICKNESS)


def draw_panel_table(panel: np.ndarray, row_titles: Tuple[str, str], title_width: int,
                     row_pairs: List[List[Tuple[str, str, int]]], top: int) -> int:
    """
    Draws a table of the information panel laid out by split_panel_table: every row pair under the previous one, its
    row titles in the left column in the colour of the group lines and its columns in the colour of the other lines.

    Args:
        panel (np.ndarray): The [H, W, 3] uint8 BGR panel, drawn on in place.
        row_titles (Tuple[str, str]): The titles of the two rows.
        title_width (int): The width of the row title column in pixels.
        row_pairs (List[List[Tuple[str, str, int]]]): The columns of every row pair (top text, bottom text, width).
        top (int): The top of the table in pixels.

    Returns:
        int: The bottom of the table in pixels, its top for a table without columns.
    """
    for row_pair in row_pairs:
        # Both rows of the pair: the row title, then the text of every column on that row
        for row_index, row_title in enumerate(row_titles):
            row_top = top + row_index * PANEL_TABLE_ROW_HEIGHT
            row_bottom = row_top + PANEL_TABLE_ROW_HEIGHT
            draw_panel_table_cell(panel=panel, box=(PANEL_MARGIN, row_top, PANEL_MARGIN + title_width, row_bottom),
                                  text=row_title, colour=PANEL_GROUP_COLOUR)
            column_left = PANEL_MARGIN + title_width
            for column in row_pair:
                column_width = column[2]
                draw_panel_table_cell(panel=panel, box=(column_left, row_top, column_left + column_width, row_bottom),
                                      text=column[row_index], colour=PANEL_LINE_COLOUR)
                column_left += column_width
        top += 2 * PANEL_TABLE_ROW_HEIGHT + PANEL_TABLE_GAP
    return top - PANEL_TABLE_GAP


def build_information_panel(title: str, lines: List[Union[str, PanelTable]], minimum_height: int,
                            width: int = INFORMATION_PANEL_WIDTH, hint: str = NAVIGATION_HINT) -> np.ndarray:
    """
    Draws the black information panel the input viewer ('view_input_images') shows next to an image: its title
    (Training, Validation, View 1...), the lines written by the trainer (the target of the sample, the augmentations
    drawn for it with their values, the objects they removed...) with the tables among them (the objects of a detection
    sample per class), and the navigation hint at the bottom, so that no text covers the image. The panel grows
    downwards when its lines need more room than the image height. The batch mosaic of the self-supervised viewer uses
    it as its header, as wide as the mosaic.

    Args:
        title (str): The title written at the top of the panel.
        lines (List[Union[str, PanelTable]]): The lines of the panel, the indented ones written in a lighter colour
            under their group, and the tables drawn in their place (PanelTable).
        minimum_height (int): The height of the displayed image, which the panel matches at least.
        width (int): The width of the panel in pixels. Defaults to INFORMATION_PANEL_WIDTH.
        hint (str): The navigation hint written at the bottom of the panel. Defaults to NAVIGATION_HINT.

    Returns:
        np.ndarray: The [H, width, 3] uint8 BGR panel.
    """
    # Lay out the entries first: every line wrapped to the panel width, every table split into the row pairs fitting it
    laid_out_entries: List[Union[str, Tuple[PanelTable, int, List[List[Tuple[str, str, int]]]]]] = []
    content_height = 0
    for line in lines:
        if isinstance(line, PanelTable):
            title_width, row_pairs = split_panel_table(table=line, maximum_width=width - 2 * PANEL_MARGIN)
            laid_out_entries.append((line, title_width, row_pairs))
            content_height += len(row_pairs) * (2 * PANEL_TABLE_ROW_HEIGHT + PANEL_TABLE_GAP)
        else:
            wrapped_lines = wrap_panel_line(line=line, maximum_width=width - 2 * PANEL_MARGIN)
            laid_out_entries.extend(wrapped_lines)
            content_height += len(wrapped_lines) * PANEL_LINE_HEIGHT

    # Room for the title, the lines and tables, an empty line and the navigation hint, at least the height of the image
    panel_height = max(minimum_height, 2 * PANEL_MARGIN + PANEL_TITLE_HEIGHT + content_height + 2 * PANEL_LINE_HEIGHT)
    panel = np.zeros(shape=(panel_height, width, 3), dtype=np.uint8)
    cv2.putText(img=panel, text=title, org=(PANEL_MARGIN, PANEL_MARGIN + PANEL_TITLE_HEIGHT - 14),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=PANEL_TITLE_FONT_SCALE, color=PANEL_TITLE_COLOUR,
                thickness=2)

    # Write the lines under the title, the group lines in green and the indented ones in light gray, and draw every
    # table below the line before it
    line_bottom = PANEL_MARGIN + PANEL_TITLE_HEIGHT
    for entry in laid_out_entries:
        if isinstance(entry, str):
            line_colour = PANEL_LINE_COLOUR if entry.startswith(" ") else PANEL_GROUP_COLOUR
            line_bottom += PANEL_LINE_HEIGHT
            cv2.putText(img=panel, text=entry, org=(PANEL_MARGIN, line_bottom), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=PANEL_LINE_FONT_SCALE, color=line_colour, thickness=PANEL_LINE_THICKNESS)
        else:
            table, title_width, row_pairs = entry
            line_bottom = draw_panel_table(panel=panel, row_titles=table.row_titles, title_width=title_width,
                                           row_pairs=row_pairs, top=line_bottom + PANEL_TABLE_GAP)

    # Write the navigation hint at the bottom of the panel
    cv2.putText(img=panel, text=hint, org=(PANEL_MARGIN, panel_height - PANEL_MARGIN),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=PANEL_LINE_FONT_SCALE, color=PANEL_HINT_COLOUR,
                thickness=PANEL_LINE_THICKNESS)
    return panel


def flatten_panel_lines(lines: List[Union[str, PanelTable]]) -> List[str]:
    """
    Writes the lines of an information panel of the input viewer as plain text, for the record of the input samples
    written at the start of every training launch (samples.json of trainers/input_samples.py): the empty separator
    lines are left out, and every table gives one line per row, its row title followed by the texts of its columns.

    Args:
        lines (List[Union[str, PanelTable]]): The lines and tables of the panel, as given to build_information_panel.

    Returns:
        List[str]: The text lines, e.g. ['Objects : 3', 'Class : circle, square', 'Count : 2, 1'].
    """
    text_lines = []
    for line in lines:
        # A table gives the texts of its top row, then those of its bottom row, each after its row title
        if isinstance(line, PanelTable):
            text_lines += [f"{row_title} : {', '.join(column[row_index] for column in line.columns)}"
                           for row_index, row_title in enumerate(line.row_titles)]
        elif line.strip():
            text_lines.append(line)
    return text_lines


def build_viewer_window(window: ViewerWindow, hint: str = NAVIGATION_HINT) -> np.ndarray:
    """
    Builds the content of one window of the input viewer ('view_input_images'), or one image of the input samples
    written at the start of every training launch (trainers/input_samples.py): the image fed to the network with its
    ground truth boxes captioned with their classes, resized so that its longest side is the display size, next to
    its black information panel holding the target lines then the augmentation lines of the window.

    Args:
        window (ViewerWindow): The sample drawn: its image, boxes, labels, class names, title, panel lines and display
            size.
        hint (str): The text written at the bottom of the panel: the navigation keys of the viewer, empty for a saved
            image. Defaults to NAVIGATION_HINT.

    Returns:
        np.ndarray: The [H, W, 3] uint8 BGR window content, the image on the left and the panel on the right.
    """
    image_numpy = convert_image_to_array(image=window.image)

    # Draw each object's bounding box rectangle and its class caption, for the tasks with boxes
    if window.bounding_boxes is not None:
        for box, label in zip(window.bounding_boxes.detach().cpu().numpy(), window.labels.detach().cpu().numpy()):
            box_left, box_top, box_right, box_bottom = map(int, box)
            caption = window.label_names.get(int(label), str(int(label))) if window.label_names else str(int(label))
            cv2.rectangle(img=image_numpy, pt1=(box_left, box_top), pt2=(box_right, box_bottom), color=(0, 255, 0),
                          thickness=2)
            cv2.putText(img=image_numpy, text=caption, org=(box_left, max(box_top - 10, 0)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0), thickness=2)

    # Resize the image so that its longest side is the display size, with sharp pixels when it is enlarged
    image_height, image_width, _ = image_numpy.shape
    display_scale = window.display_input_size / max(image_height, image_width)
    display_width = max(1, round(image_width * display_scale))
    display_height = max(1, round(image_height * display_scale))
    interpolation = cv2.INTER_NEAREST if display_scale >= 1.0 else cv2.INTER_AREA
    display_image = cv2.resize(src=image_numpy, dsize=(display_width, display_height), interpolation=interpolation)

    # Place the image at the top left of a black window and the panel on its right, behind a gray gap
    panel = build_information_panel(title=window.title, lines=window.target_lines + window.augmentation_lines,
                                    minimum_height=display_height, hint=hint)
    window_image = np.zeros(shape=(panel.shape[0], display_width + GRID_GAP + INFORMATION_PANEL_WIDTH, 3),
                            dtype=np.uint8)
    window_image[:display_height, :display_width] = display_image
    window_image[:, display_width:display_width + GRID_GAP] = 128
    window_image[:, display_width + GRID_GAP:] = panel
    return window_image


def compute_window_positions(window_sizes: List[Tuple[int, int]],
                             screen_size: Optional[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], float]:
    """
    Places the windows of the input viewer ('view_input_images') at the centre of the screen, so that the user never
    has to move them: side by side (the training window next to the validation one) when they fit the width of the
    screen, else stacked, and scaled down when they would not fit the screen.

    Args:
        window_sizes (List[Tuple[int, int]]): The width and height in pixels of the content of every window.
        screen_size (Optional[Tuple[int, int]]): The width and height of the screen, or None when it is unknown, the
            windows then being placed side by side from the top left of the screen, unscaled.

    Returns:
        Tuple[List[Tuple[int, int]], float]: The top left position of every window, and the scale of their content.
    """
    window_widths = [window_width for window_width, window_height in window_sizes]
    window_heights = [window_height for window_width, window_height in window_sizes]
    number_windows = len(window_sizes)

    # Without the screen size, the windows are placed side by side from the top left of the screen
    if screen_size is None:
        window_lefts = [sum(window_widths[:window_index]) + window_index * VIEWER_WINDOW_SPACING
                        for window_index in range(number_windows)]
        return [(window_left, VIEWER_TITLE_BAR_HEIGHT) for window_left in window_lefts], 1.0

    # Side by side when the windows fit the width of the screen, else stacked, the spacing and the title bars added by
    # the window manager keeping their size whatever the scale of the content
    screen_width, screen_height = screen_size
    side_by_side = (sum(window_widths) + (number_windows - 1) * VIEWER_WINDOW_SPACING <=
                    VIEWER_SCREEN_FRACTION * screen_width)
    if side_by_side:
        content_width, fixed_width = sum(window_widths), (number_windows - 1) * VIEWER_WINDOW_SPACING
        content_height, fixed_height = max(window_heights), VIEWER_TITLE_BAR_HEIGHT
    else:
        content_width, fixed_width = max(window_widths), 0
        content_height = sum(window_heights)
        fixed_height = number_windows * VIEWER_TITLE_BAR_HEIGHT + (number_windows - 1) * VIEWER_WINDOW_SPACING

    # Scale the content down when the block of windows would not fit the screen, then centre the block
    display_scale = min(1.0, (VIEWER_SCREEN_FRACTION * screen_width - fixed_width) / content_width,
                        (VIEWER_SCREEN_FRACTION * screen_height - fixed_height) / content_height)
    block_left = (screen_width - (content_width * display_scale + fixed_width)) / 2
    block_top = (screen_height - (content_height * display_scale + fixed_height)) / 2

    # Lay the windows out from the top left of the block, every stacked window centred horizontally
    window_positions = []
    for window_index in range(number_windows):
        if side_by_side:
            window_left = (block_left + sum(window_widths[:window_index]) * display_scale +
                           window_index * VIEWER_WINDOW_SPACING)
            window_top = block_top
        else:
            window_left = (screen_width - window_widths[window_index] * display_scale) / 2
            window_top = (block_top + sum(window_heights[:window_index]) * display_scale +
                          window_index * (VIEWER_TITLE_BAR_HEIGHT + VIEWER_WINDOW_SPACING))
        window_positions.append((max(0, round(window_left)), max(0, round(window_top))))
    return window_positions, display_scale


def view_input_data(windows: List[ViewerWindow]) -> bool:
    """
    Displays preprocessed input images with their ground truth boxes, one window per image, each next to a black
    information panel, so the user can check exactly what is fed to the neural network and how it was augmented.

    It is the 'view_input_images' diagnosis mode of the trainers (a Training and a Validation window, or the two views
    of the self-supervised task), whose windows the task trainers build (build_input_windows): the object detection
    trainer draws the boxes, and every trainer writes in the panels the target of the sample (its class, its
    objects...) and, for the training sample, the augmentations drawn for it with their values. The windows are
    centred on the screen. Execution halts until the user presses SPACE (next image) or ENTER (stop viewing).

    Args:
        windows (List[ViewerWindow]): The windows to show, side by side in their order, their titles naming them.

    Returns:
        bool: True if the user pressed ENTER to stop the visualization (or no display is available), False otherwise.
    """
    # Build every window: the image with its boxes next to its information panel
    window_images = [build_viewer_window(window=window) for window in windows]
    return show_viewer_windows(window_images=window_images, window_titles=[window.title for window in windows])


def join_viewer_windows(window_images: List[np.ndarray]) -> np.ndarray:
    """
    Places the windows of one sample side by side in a single image, for the input samples written at the start of
    every training launch (trainers/input_samples.py): one window for most tasks, the two views of the self-supervised
    task next to each other as the input viewer shows them, separated by a gray band as wide as the space between
    two windows of the viewer, the shorter windows padded with black at the bottom.

    Args:
        window_images (List[np.ndarray]): The [H, W, 3] uint8 BGR windows built by build_viewer_window, in order.

    Returns:
        np.ndarray: The [max H, sum W + gaps, 3] uint8 BGR image of the sample.
    """
    joined_height = max(window_image.shape[0] for window_image in window_images)
    joined_width = sum(window_image.shape[1] for window_image in window_images) + \
        (len(window_images) - 1) * VIEWER_WINDOW_SPACING
    joined_image = np.zeros(shape=(joined_height, joined_width, 3), dtype=np.uint8)

    # Copy every window at its place from the left, followed by the gray band separating it from the next one
    window_left = 0
    for window_image in window_images:
        window_height, window_width = window_image.shape[:2]
        joined_image[:window_height, window_left:window_left + window_width] = window_image
        joined_image[:, window_left + window_width:window_left + window_width + VIEWER_WINDOW_SPACING] = 128
        window_left += window_width + VIEWER_WINDOW_SPACING
    return joined_image


def show_viewer_windows(window_images: List[np.ndarray], window_titles: List[str]) -> bool:
    """
    Shows the windows of the input viewer ('view_input_images') centred on the screen, scaled down when they would not
    fit it, and waits for the user: the windows of the samples built by view_input_data, or the batch mosaic of the
    self-supervised viewer (build_batch_mosaic). Execution halts until the user presses SPACE (next) or ENTER (stop
    viewing).

    Args:
        window_images (List[np.ndarray]): The [H, W, 3] uint8 BGR content of every window.
        window_titles (List[str]): The title of every window.

    Returns:
        bool: True if the user pressed ENTER to stop the visualization (or no display is available), False otherwise.
    """
    # Skip the viewer when no graphical display is reachable (e.g. SSH without X forwarding) instead of crashing
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print_color(color="yellow",
                    output="No display available (DISPLAY is not set). Skipping the input image viewer.")
        return True

    # Centre the windows on the screen, scaled down when they would not fit it
    window_positions, display_scale = compute_window_positions(
        window_sizes=[(window_image.shape[1], window_image.shape[0]) for window_image in window_images],
        screen_size=get_screen_size())
    for window_image, window_title, (window_left, window_top) in zip(window_images, window_titles, window_positions):
        cv2.namedWindow(winname=window_title, flags=cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winname=window_title, width=round(window_image.shape[1] * display_scale),
                         height=round(window_image.shape[0] * display_scale))
        cv2.moveWindow(winname=window_title, x=window_left, y=window_top)
        cv2.imshow(winname=window_title, mat=window_image)

    # Enter a polling loop to restrict progression exclusively to the Spacebar or Enter keys
    while True:
        pressed_key = cv2.waitKey(delay=0) & 0xFF
        if pressed_key == 13:  # 13 represents the Enter/Return key
            cv2.destroyAllWindows()
            return True
        if pressed_key == ord(' '):
            break

    return False


def build_batch_mosaic(images: torch.Tensor, block_size: int, block_captions: List[str],
                       image_captions: Optional[List[str]], title: str, lines: List[str]) -> np.ndarray:
    """
    Draws the batch mosaic of the self-supervised input viewer ('view_input_images'), shown before the views of the
    images of a batch built by the GroupedBatchSampler: every image of the batch as a thumbnail, in the order of the
    batch, cut into consecutive blocks of block_size images, each block under a caption bar (its block value for a
    grouped batch) and surrounded by gray space, up to BATCH_MOSAIC_IMAGES_PER_ROW thumbnails per row, a block of more
    images going on several rows. A header above the blocks, drawn like the information panels, gives the title, the
    lines describing the batch and the navigation hint.

    Args:
        images (torch.Tensor): The [B, 3, H, W] BGR images of the batch in [0, 255], on any device.
        block_size (int): The number of images of a block.
        block_captions (List[str]): The caption of every block, written in its caption bar.
        image_captions (Optional[List[str]]): A caption written on every thumbnail (e.g. the group values of the images
            of a random batch), or None to leave the thumbnails bare.
        title (str): The title of the header.
        lines (List[str]): The lines of the header, the indented ones written in a lighter colour under their group.

    Returns:
        np.ndarray: The [H, W, 3] uint8 BGR mosaic, at least INFORMATION_PANEL_WIDTH pixels wide.
    """
    # Images of the batch as [B, H, W, 3] uint8 arrays, which OpenCV resizes and draws on
    image_arrays = convert_image_to_array(image=images)
    number_images, image_height, image_width, _ = image_arrays.shape

    # Thumbnails whose longest side is the thumbnail size
    thumbnail_scale = BATCH_MOSAIC_THUMBNAIL_SIZE / max(image_height, image_width)
    thumbnail_width = max(1, round(image_width * thumbnail_scale))
    thumbnail_height = max(1, round(image_height * thumbnail_scale))
    interpolation = cv2.INTER_NEAREST if thumbnail_scale >= 1.0 else cv2.INTER_AREA

    # Size of a block: its caption bar above rows of thumbnails separated by gaps, and the blocks per row of the mosaic
    columns_per_block = min(block_size, BATCH_MOSAIC_IMAGES_PER_ROW)
    rows_per_block = (block_size + columns_per_block - 1) // columns_per_block
    block_width = columns_per_block * (thumbnail_width + GRID_GAP) - GRID_GAP
    block_height = PANEL_LINE_HEIGHT + rows_per_block * (thumbnail_height + GRID_GAP)
    number_blocks = (number_images + block_size - 1) // block_size
    blocks_per_row = min(number_blocks, max(1, BATCH_MOSAIC_IMAGES_PER_ROW // block_size))
    number_block_rows = (number_blocks + blocks_per_row - 1) // blocks_per_row
    grid_width = blocks_per_row * (block_width + BATCH_MOSAIC_BLOCK_SPACING) + BATCH_MOSAIC_BLOCK_SPACING
    grid_height = number_block_rows * (block_height + BATCH_MOSAIC_BLOCK_SPACING) + BATCH_MOSAIC_BLOCK_SPACING

    # Header as wide as the mosaic, above a gray grid of blocks
    mosaic_width = max(grid_width, INFORMATION_PANEL_WIDTH)
    header = build_information_panel(title=title, lines=lines, minimum_height=0, width=mosaic_width,
                                     hint=BATCH_NAVIGATION_HINT)
    grid = np.full(shape=(grid_height, mosaic_width, 3), fill_value=64, dtype=np.uint8)
    for block_index in range(number_blocks):

        # Block with its black caption bar, the caption cut at the edge of the block when longer, above the thumbnails
        # separated by gray gaps
        block = np.full(shape=(block_height, block_width, 3), fill_value=128, dtype=np.uint8)
        block[:PANEL_LINE_HEIGHT] = 0
        cv2.putText(img=block, text=block_captions[block_index], org=(GRID_GAP, PANEL_LINE_HEIGHT - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=PANEL_LINE_FONT_SCALE, color=PANEL_GROUP_COLOUR,
                    thickness=PANEL_LINE_THICKNESS)

        # Thumbnails of the images of the block, row by row, each with its caption when given
        block_images = range(block_index * block_size, min((block_index + 1) * block_size, number_images))
        for position, image_index in enumerate(block_images):
            thumbnail = cv2.resize(src=image_arrays[image_index], dsize=(thumbnail_width, thumbnail_height),
                                   interpolation=interpolation)
            if image_captions is not None:
                (caption_width, caption_height), _ = cv2.getTextSize(
                    text=image_captions[image_index], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=BATCH_MOSAIC_CAPTION_FONT_SCALE, thickness=1)
                cv2.rectangle(img=thumbnail, pt1=(0, 0), pt2=(caption_width + 6, caption_height + 8),
                              color=(0, 0, 0), thickness=-1)
                cv2.putText(img=thumbnail, text=image_captions[image_index], org=(3, caption_height + 3),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=BATCH_MOSAIC_CAPTION_FONT_SCALE,
                            color=PANEL_TITLE_COLOUR, thickness=1)
            top = PANEL_LINE_HEIGHT + (position // columns_per_block) * (thumbnail_height + GRID_GAP)
            left = (position % columns_per_block) * (thumbnail_width + GRID_GAP)
            block[top:top + thumbnail_height, left:left + thumbnail_width] = thumbnail

        # Place the block in its cell of the grid, row by row
        block_top = BATCH_MOSAIC_BLOCK_SPACING + (block_index // blocks_per_row) * (block_height +
                                                                                    BATCH_MOSAIC_BLOCK_SPACING)
        block_left = BATCH_MOSAIC_BLOCK_SPACING + (block_index % blocks_per_row) * (block_width +
                                                                                    BATCH_MOSAIC_BLOCK_SPACING)
        grid[block_top:block_top + block_height, block_left:block_left + block_width] = block
    return np.concatenate([header, grid], axis=0)


def build_frame_grid(frames: torch.Tensor, sequence_mask: torch.Tensor, frame_captions: Optional[List[str]],
                     number_columns: int, first_frame_number: int = 0) -> torch.Tensor:
    """
    Lays out the real frames of a sequence in reading order on a grid, each numbered and optionally captioned (e.g.
    with its class), so that the input viewer of the sequence trainers shows a whole sequence in one window, and the
    mosaics of scripts/embedding/cluster_image_embeddings.py show the first images of a cluster numbered by their rank.

    Args:
        frames (torch.Tensor): The [T, 3, H, W] frames of the sequence, on CPU or GPU.
        sequence_mask (torch.Tensor): The [T] booleans, True for the real frames (padding frames are not shown).
        frame_captions (Optional[List[str]]): A text written on every frame, or None to only number them.
        number_columns (int): The number of frames per row of the grid.
        first_frame_number (int): The number written on the first real frame, the next ones counting on from it.
            Defaults to 0.

    Returns:
        torch.Tensor: The [3, rows x H, columns x W] uint8 grid, gray between the frames.
    """
    real_frames = convert_image_to_array(image=frames[sequence_mask.to(device=frames.device)])
    number_frames, frame_height, frame_width, _ = real_frames.shape
    number_rows = (number_frames + number_columns - 1) // number_columns

    # Gray canvas with a 4 pixel gap around every frame
    gap = 4
    grid = np.full(shape=(number_rows * (frame_height + gap) + gap, number_columns * (frame_width + gap) + gap, 3),
                   fill_value=128, dtype=np.uint8)
    for frame_index in range(number_frames):
        frame_numpy = real_frames[frame_index]

        # Number of the frame, followed by its caption when given
        frame_number = first_frame_number + frame_index
        frame_text = str(frame_number) if frame_captions is None else f"{frame_number} {frame_captions[frame_index]}"
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
                                predicted_masks: Optional[np.ndarray] = None,
                                highlighted_ground_truths: Optional[np.ndarray] = None,
                                highlighted_detections: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Lays out one evaluated test image of the object detection in two titled panels side by side, saved and sent to
    TensorBoard at every evaluation so that the detections of the same images can be followed over the training: the
    ground truth boxes captioned with their class, and the detections captioned with their class and score, every box
    in the colour of its class. A caption bar names the sample above the panels, and a legend bar below them names the
    classes drawn. For the instance segmentation, the masks of the objects are painted under their boxes. The error
    images of the evaluation outline the boxes in error (a missed object, a false positive, an object found with
    another class) with red dashes, named 'Error' in the legend.

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
        highlighted_ground_truths (Optional[np.ndarray]): The [N] flags of the true boxes outlined as errors, None for
            none. Defaults to None.
        highlighted_detections (Optional[np.ndarray]): The [M] flags of the detections outlined as errors, None for
            none. Defaults to None.

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

    # Boxes in error outlined with red dashes a few pixels around them, on the panel of their side
    for panel, boxes, highlighted_boxes in [(ground_truth_panel, ground_truth_boxes, highlighted_ground_truths),
                                            (prediction_panel, predicted_boxes, highlighted_detections)]:
        if highlighted_boxes is None:
            continue
        for x_minimum, y_minimum, x_maximum, y_maximum in boxes[highlighted_boxes]:
            draw_dashed_rectangle(image=panel, top_left=(max(0, int(x_minimum) - ERROR_OUTLINE_MARGIN),
                                                         max(0, int(y_minimum) - ERROR_OUTLINE_MARGIN)),
                                  bottom_right=(min(image_size - 1, int(x_maximum) + ERROR_OUTLINE_MARGIN),
                                                min(image_size - 1, int(y_maximum) + ERROR_OUTLINE_MARGIN)),
                                  colour=WRONG_PREDICTION_COLOUR, thickness=2, dash_length=ERROR_OUTLINE_DASH_LENGTH)
    grid = place_panels_side_by_side(panels={"Ground Truth": ground_truth_panel, "Prediction": prediction_panel})

    # Legend of the error outline when there is one, first so that it is never cut, then of the classes of the true
    # and detected boxes
    legend_entries = [(class_names[class_index], tuple(class_colours[class_index].tolist()))
                      for class_index in drawn_class_indices]
    if highlighted_ground_truths is not None or highlighted_detections is not None:
        legend_entries.insert(0, (ERROR_OUTLINE_LEGEND, WRONG_PREDICTION_COLOUR))
    legend_bar = draw_legend_bar(width=grid.shape[1], legend_entries=legend_entries)
    return add_caption_bar(image=np.concatenate([grid, legend_bar], axis=0), caption=caption)


def find_keypoint_side(keypoint_name: str) -> str:
    """
    Tells the side of the object a keypoint belongs to from its name, so that the keypoint drawings of the input
    viewer and of the evaluation sample grids colour the left and right keypoints differently: a prediction mixing them
    up then shows at a glance. The name is split on its underscores, 'left' or 'right' being read as a whole word.

    Args:
        keypoint_name (str): The name of the keypoint, e.g. 'left_elbow' or 'roof_apex'.

    Returns:
        str: "Left", "Right" or "Centre", a key of KEYPOINT_SIDE_COLOURS.
    """
    name_words = keypoint_name.split("_")
    if "left" in name_words:
        return "Left"
    if "right" in name_words:
        return "Right"
    return "Centre"


def draw_keypoint_objects(image: np.ndarray, keypoints: np.ndarray, visibilities: Optional[np.ndarray],
                          labels: np.ndarray, keypoint_categories: Dict[int, Dict[str, Any]],
                          class_colours: np.ndarray, captions: Optional[List[str]] = None) -> np.ndarray:
    """
    Draws the keypoints of every object of one image of the keypoint detection over a copy of the image, at its own
    resolution, for the input viewer of the trainer (the targets of an augmented image) and the panels of the
    evaluation sample grids (the true and the predicted objects of a test image): the skeleton of every object in the
    colour of its class, then every keypoint as a disc coloured by its side (find_keypoint_side), and the caption of
    the object above its highest keypoint. A labelled visible keypoint, or a found predicted one, is a filled disc, a
    labelled occluded keypoint a hollow ring, and an unlabelled keypoint or a predicted one not found is left out.

    Args:
        image (np.ndarray): The [H, W, 3] uint8 BGR network input.
        keypoints (np.ndarray): The [N, K, 2] keypoints (x, y) of the objects in continuous image pixels, a pixel
            centre being at its index plus 0.5, NaN for the predicted keypoints not found.
        visibilities (Optional[np.ndarray]): The [N, K] COCO visibilities (0 unlabelled, 1 labelled occluded, 2
            labelled visible) of the true keypoints, None for predicted keypoints, drawn as visible when found.
        labels (np.ndarray): The [N] class index of every object.
        keypoint_categories (Dict[int, Dict[str, Any]]): The 'keypoints' names and 'skeleton' (0-based pairs of
            keypoint positions) of every class index ('Data.keypoint_categories').
        class_colours (np.ndarray): The [C, 3] uint8 BGR colour of every class (build_class_colours).
        captions (Optional[List[str]]): The text written above every object in the colour of its class, on a black
            background. Defaults to None, in which case the objects are drawn without text.

    Returns:
        np.ndarray: The [H, W, 3] uint8 BGR drawing.
    """
    drawing = image.copy()
    for object_index, class_index in enumerate(labels.tolist()):
        definition = keypoint_categories[class_index]
        number_keypoints = len(definition["keypoints"])
        object_keypoints = keypoints[object_index, :number_keypoints]
        class_colour = tuple(int(component) for component in class_colours[class_index])

        # Keypoints drawn: the labelled true ones, the found predicted ones, at the pixel of their continuous position
        drawn_keypoints = (np.isfinite(object_keypoints).all(axis=1) if visibilities is None
                           else visibilities[object_index, :number_keypoints] > 0)
        if not drawn_keypoints.any():
            continue
        drawn_points = [(int(round(keypoint_x - 0.5)), int(round(keypoint_y - 0.5))) if drawn else (0, 0)
                        for (keypoint_x, keypoint_y), drawn in zip(object_keypoints.tolist(), drawn_keypoints.tolist(),
                                                                   strict=True)]

        # Skeleton lines between two drawn keypoints, in the colour of the class, under the keypoint discs
        for start_index, end_index in definition["skeleton"]:
            if drawn_keypoints[start_index] and drawn_keypoints[end_index]:
                cv2.line(img=drawing, pt1=drawn_points[start_index], pt2=drawn_points[end_index], color=class_colour,
                         thickness=SKELETON_THICKNESS, lineType=cv2.LINE_AA)

        # Keypoint discs coloured by their side with a black outline, the occluded true keypoints as rings
        for keypoint_index, keypoint_name in enumerate(definition["keypoints"]):
            if not drawn_keypoints[keypoint_index]:
                continue
            side_colour = KEYPOINT_SIDE_COLOURS[find_keypoint_side(keypoint_name=keypoint_name)]
            occluded = visibilities is not None and visibilities[object_index, keypoint_index] == 1
            cv2.circle(img=drawing, center=drawn_points[keypoint_index], radius=KEYPOINT_RADIUS + 1, color=(0, 0, 0),
                       thickness=-1 if not occluded else 4, lineType=cv2.LINE_AA)
            cv2.circle(img=drawing, center=drawn_points[keypoint_index], radius=KEYPOINT_RADIUS, color=side_colour,
                       thickness=-1 if not occluded else 2, lineType=cv2.LINE_AA)

        # Caption of the object above its highest keypoint, or below it when the keypoint touches the top of the image
        if captions is not None:
            caption_left = min(point[0] for point, drawn in zip(drawn_points, drawn_keypoints, strict=True) if drawn)
            highest_point = min(point[1] for point, drawn in zip(drawn_points, drawn_keypoints, strict=True) if drawn)
            (caption_width, caption_height), _ = cv2.getTextSize(text=captions[object_index],
                                                                 fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                                                                 thickness=1)
            caption_top = highest_point - KEYPOINT_RADIUS - caption_height - 6
            if caption_top < 0:
                caption_top = highest_point + KEYPOINT_RADIUS + 2
            caption_left = int(np.clip(caption_left, a_min=0, a_max=max(0, drawing.shape[1] - caption_width - 4)))
            cv2.rectangle(img=drawing, pt1=(caption_left, caption_top),
                          pt2=(caption_left + caption_width + 4, caption_top + caption_height + 6), color=(0, 0, 0),
                          thickness=-1)
            cv2.putText(img=drawing, text=captions[object_index],
                        org=(caption_left + 2, caption_top + caption_height + 3), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4, color=class_colour, thickness=1)
    return drawing


def build_keypoint_sample_grid(image: np.ndarray, ground_truth_keypoints: np.ndarray, visibilities: np.ndarray,
                               ground_truth_labels: np.ndarray, predicted_keypoints: np.ndarray,
                               predicted_labels: np.ndarray, predicted_scores: np.ndarray,
                               keypoint_categories: Dict[int, Dict[str, Any]], class_names: Dict[int, str],
                               class_colours: np.ndarray, caption: str) -> np.ndarray:
    """
    Lays out one evaluated test image of the keypoint detection in two titled panels side by side, saved and sent to
    TensorBoard at every evaluation so that the keypoints of the same images can be followed over the training: the
    true objects captioned with their class, and the predicted objects captioned with their class and score, all with
    the skeleton of their class (draw_keypoint_objects). A caption bar names the sample above the panels, and a legend
    bar below them names the keypoint sides, then the classes drawn.

    Args:
        image (np.ndarray): The [H, W, 3] uint8 BGR network input.
        ground_truth_keypoints (np.ndarray): The [N, K, 2] true keypoints of the objects in continuous image pixels.
        visibilities (np.ndarray): The [N, K] COCO visibilities of the true keypoints.
        ground_truth_labels (np.ndarray): The [N] class index of every true object.
        predicted_keypoints (np.ndarray): The [D, K, 2] predicted keypoints in continuous image pixels, NaN for the
            keypoints not found.
        predicted_labels (np.ndarray): The [D] class index of every predicted object.
        predicted_scores (np.ndarray): The [D] score of every predicted object.
        keypoint_categories (Dict[int, Dict[str, Any]]): The 'keypoints' names and 'skeleton' of every class index.
        class_names (Dict[int, str]): The name of every class index.
        class_colours (np.ndarray): The [C, 3] uint8 BGR colour of every class (build_class_colours).
        caption (str): The text of the caption bar, e.g. the number of true and predicted objects.

    Returns:
        np.ndarray: The [24 + H + 24, 2 x W + 4, 3] uint8 BGR grid.
    """
    grid = place_panels_side_by_side(panels={
        "Ground Truth": draw_keypoint_objects(image=image, keypoints=ground_truth_keypoints, visibilities=visibilities,
                                              labels=ground_truth_labels, keypoint_categories=keypoint_categories,
                                              class_colours=class_colours,
                                              captions=[class_names[label] for label in ground_truth_labels.tolist()]),
        "Prediction": draw_keypoint_objects(image=image, keypoints=predicted_keypoints, visibilities=None,
                                            labels=predicted_labels, keypoint_categories=keypoint_categories,
                                            class_colours=class_colours,
                                            captions=[f"{class_names[label]} {score:.2f}" for label, score in
                                                      zip(predicted_labels.tolist(), predicted_scores.tolist(),
                                                          strict=True)])})

    # Legend of the keypoint sides of the classes drawn, in the order of KEYPOINT_SIDE_COLOURS, then of the classes
    drawn_class_indices = sorted(set(ground_truth_labels.tolist()) | set(predicted_labels.tolist()))
    drawn_sides = {find_keypoint_side(keypoint_name=keypoint_name) for class_index in drawn_class_indices
                   for keypoint_name in keypoint_categories[class_index]["keypoints"]}
    legend_entries = [(side_name, side_colour) for side_name, side_colour in KEYPOINT_SIDE_COLOURS.items()
                      if side_name in drawn_sides]
    legend_entries += [(class_names[class_index], tuple(class_colours[class_index].tolist()))
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
