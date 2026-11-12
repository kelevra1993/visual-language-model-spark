"""
Drawing and saving of the confusion matrix of an evaluated checkpoint: the CSV table, always saved in the evaluation
folder of the checkpoint, and the matplotlib image (with the recall of every true class, the precision of every
predicted class and the accuracy), saved and logged to TensorBoard when the per-class report is on. Used by the
classification trainer and the trainers inheriting from it, by the segmentation trainer for its pixels, and by the
detection trainers for the detections matched to the objects, with a background row and column. The classification
image writes its shares, recalls, precisions and accuracy as the percentages of the reports ("94.03 %"), with larger
cells for their two decimals; the other tasks keep their earlier "94.0%" until they are moved to them.
"""

import csv
import cv2
import numpy as np
import torch

from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter
from pathlib import Path
from typing import List, Optional, Tuple

from vision_training.evaluation.report_formatting import format_percentage, format_percentage_tick

# Largest number of classes whose cells show their count and share of the true class, show their count only, and
# show the class names as tick labels (class indices above, matching the 'class_index' column of class_metrics.csv)
MAXIMUM_CLASSES_WITH_SHARES = 20
MAXIMUM_CLASSES_WITH_COUNTS = 40
MAXIMUM_CLASSES_WITH_NAMES = 60

# Number of characters of the class names kept in the tick labels
MAXIMUM_NAME_LENGTH = 24

# Side of the matrix in inches (0.4 inch per class, 0.65 inch with the two decimals of the percentages, clipped), width
# of the recall and precision strips in inches, and resolution of the image
MATRIX_SIDE_PER_CLASS = 0.4
PERCENT_MATRIX_SIDE_PER_CLASS = 0.65
MINIMUM_MATRIX_SIDE = 4.0
MAXIMUM_MATRIX_SIDE = 16.0
STRIP_WIDTH = 0.7
COLOUR_BAR_WIDTH = 0.25
IMAGE_DOTS_PER_INCH = 150

# Colour map of the shares of the true class (and of the recall, precision and accuracy strips, on the same scale),
# colour of the rows without any evaluated sample, and colours of the text written on the cells
COLOUR_MAP_NAME = "Blues"
EMPTY_ROW_COLOUR = "#e6e6e6"
LIGHT_TEXT_COLOUR = "#ffffff"
DARK_TEXT_COLOUR = "#1a1a1a"

# Relative luminance of a cell below which its text is written in the light colour (WCAG contrast)
LIGHT_TEXT_LUMINANCE = 0.4


def compute_text_colour(cell_colour: Tuple[float, float, float, float]) -> str:
    """
    Chooses the colour of the text written on a cell of the confusion matrix image, light on dark cells and dark on
    light cells, from the relative luminance of the cell colour (WCAG), so that every count stays readable.

    Args:
        cell_colour (Tuple[float, float, float, float]): The RGBA colour of the cell, components in [0, 1].

    Returns:
        str: The colour of the text.
    """
    linear_components = [component / 12.92 if component <= 0.03928 else ((component + 0.055) / 1.055) ** 2.4
                         for component in cell_colour[:3]]
    luminance = 0.2126 * linear_components[0] + 0.7152 * linear_components[1] + 0.0722 * linear_components[2]
    return LIGHT_TEXT_COLOUR if luminance < LIGHT_TEXT_LUMINANCE else DARK_TEXT_COLOUR


def format_share(share: float, in_percent: bool) -> str:
    """
    Writes a share of the confusion matrix image (the share of a cell in its true class, a recall, a precision or the
    accuracy): as the percentages of the reports for the classification image ("94.03 %"), with one decimal and no
    space for the tasks not yet moved to them ("94.0%").

    Args:
        share (float): The share, between 0 and 1.
        in_percent (bool): Whether the image writes the percentages of the reports.

    Returns:
        str: The written share.
    """
    return format_percentage(value=share) if in_percent else f"{100.0 * share:.1f}%"


def write_cell_texts(axes: Axes, counts: np.ndarray, shares: np.ndarray, colour_map: Colormap,
                     normalization: Normalize, show_counts: bool, background_index: Optional[int],
                     in_percent: bool) -> None:
    """
    Writes the count of every cell of the matrix of a confusion matrix image, with its share of the true class up to
    20 classes, the diagonal counts (the right predictions) in bold and the empty off-diagonal cells left blank.
    Without counts (e.g. the pixels of the segmentation, too many to print), the cells only show their share. The
    background cell of the detection matrices, the background found as background, is never counted and stays blank.

    Args:
        axes (Axes): The axes of the matrix.
        counts (np.ndarray): The [K, K] counts (rows true classes, columns predicted classes).
        shares (np.ndarray): The [K, K] shares of the true class of every cell, NaN for the rows without samples.
        colour_map (Colormap): The colour map of the shares, giving the colour behind every text.
        normalization (Normalize): The normalization of the shares into the colour map.
        show_counts (bool): Whether the cells show their count, or only their share of the true class.
        background_index (Optional[int]): The index of the background class of the detection matrices, None for the
            other tasks.
        in_percent (bool): Whether the shares are written as the percentages of the reports (format_share).
    """
    number_classes = counts.shape[0]
    show_shares = number_classes <= MAXIMUM_CLASSES_WITH_SHARES
    font_size = 8 if show_shares else 6
    for true_index in range(number_classes):
        for predicted_index in range(number_classes):
            count = int(counts[true_index, predicted_index])
            is_diagonal = true_index == predicted_index
            if (count == 0 and not is_diagonal) or (is_diagonal and true_index == background_index):
                continue

            # Count, with the share of the true class below it when the cells are large enough, or the share alone
            share = float(shares[true_index, predicted_index])
            if not show_counts:
                if np.isnan(share):
                    continue
                text = format_share(share=share, in_percent=in_percent)
            else:
                text = (f"{count}\n{format_share(share=share, in_percent=in_percent)}" if show_shares and
                        not np.isnan(share) else str(count))
            cell_colour = colour_map(normalization(share)) if not np.isnan(share) else (0.9, 0.9, 0.9, 1.0)
            axes.text(x=predicted_index, y=true_index, s=text, ha="center", va="center", fontsize=font_size,
                      fontweight="bold" if is_diagonal else "normal",
                      color=compute_text_colour(cell_colour=cell_colour))


def write_strip_texts(axes: Axes, values: np.ndarray, colour_map: Colormap, normalization: Normalize,
                      vertical: bool, in_percent: bool) -> None:
    """
    Writes the value of every cell of a strip of a confusion matrix image (the recall of every true class, or the
    precision of every predicted class) as a percentage, up to 40 classes.

    Args:
        axes (Axes): The axes of the strip.
        values (np.ndarray): The [K] values of the strip, NaN when undefined (a class never seen or never predicted).
        colour_map (Colormap): The colour map of the values, giving the colour behind every text.
        normalization (Normalize): The normalization of the values into the colour map.
        vertical (bool): True for the recall column, False for the precision row.
        in_percent (bool): Whether the values are written as the percentages of the reports (format_share).
    """
    font_size = 8 if len(values) <= MAXIMUM_CLASSES_WITH_SHARES else 6
    for class_index, value in enumerate(values.tolist()):
        if np.isnan(value):
            continue
        axes.text(x=0 if vertical else class_index, y=class_index if vertical else 0,
                  s=format_share(share=value, in_percent=in_percent), ha="center", va="center", fontsize=font_size,
                  color=compute_text_colour(cell_colour=colour_map(normalization(value))))


def set_class_ticks(axes: Axes, class_names: List[str], along_columns: bool) -> None:
    """
    Labels the classes of one side of the matrix of a confusion matrix image: their names (shortened, rotated on the
    predicted side) up to 60 classes, otherwise a few class indices.

    Args:
        axes (Axes): The axes of the matrix.
        class_names (List[str]): The name of every class.
        along_columns (bool): True to label the predicted classes (columns), False the true classes (rows).
    """
    number_classes = len(class_names)
    if number_classes <= MAXIMUM_CLASSES_WITH_NAMES:
        tick_positions = list(range(number_classes))
        tick_labels = [class_name[:MAXIMUM_NAME_LENGTH] for class_name in class_names]
        font_size = 9 if number_classes <= MAXIMUM_CLASSES_WITH_SHARES else 6
    else:
        tick_positions = list(range(0, number_classes, max(1, number_classes // 20)))
        tick_labels = [str(tick_position) for tick_position in tick_positions]
        font_size = 7
    if along_columns:
        axes.set_xticks(ticks=tick_positions, labels=tick_labels, rotation=45, ha="left", rotation_mode="anchor",
                        fontsize=font_size)
    else:
        axes.set_yticks(ticks=tick_positions, labels=tick_labels, fontsize=font_size)


def draw_confusion_matrix(confusion_matrix: torch.Tensor, class_names: List[str], title: str,
                          show_counts: bool = True, background_index: Optional[int] = None,
                          in_percent: bool = False) -> np.ndarray:
    """
    Draws the confusion matrix of an evaluated checkpoint as an image, saved in its evaluation folder and logged to
    TensorBoard by the per-class report: every cell coloured by its share of the samples of its true class, with its
    count, a recall column on the right (the diagonal share of every true class), a precision row at the bottom (the
    share of right predictions of every predicted class) and the accuracy in the corner, all on the same colour scale,
    so that the classes confused with each other stand out. It is drawn on a matplotlib figure of its own, without
    pyplot, its size growing with the number of classes (capped). The detection matrices have a background row (the
    false positives) and a background column (the missed objects): their background cell, the recall of the background
    and its precision are undefined and left blank, and the accuracy is the share of the right detections among the
    objects, the false positives and the confusions. The classification image writes the percentages of the reports
    ("94.03 %", colour bar "20 %") in cells of 0.65 inch per class, the other tasks keep "94.0%" and "20%".

    Args:
        confusion_matrix (torch.Tensor): The [K, K] confusion matrix (rows true classes, columns predicted classes).
        class_names (List[str]): The name of every class.
        title (str): The title of the image, e.g. 'Confusion Matrix · Iteration 4000 · Accuracy 81.23 %'.
        show_counts (bool): Whether the cells show their count, False for counts too large to print (pixels), the
            cells then showing their share of the true class only. Defaults to True.
        background_index (Optional[int]): The index of the background row and column of the detection matrices, None
            for a matrix without background. Defaults to None.
        in_percent (bool): Whether the shares, recalls, precisions, accuracy and colour bar are written as the
            percentages of the reports. Defaults to False.

    Returns:
        np.ndarray: The [H, W, 3] uint8 RGB image.
    """
    counts = confusion_matrix.to(dtype=torch.float64).numpy()
    number_classes = counts.shape[0]
    true_counts, predicted_counts, right_counts = counts.sum(axis=1), counts.sum(axis=0), np.diagonal(counts)

    # Shares of the true class (NaN for the rows without samples), recall, precision and accuracy
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = counts / true_counts[:, np.newaxis]
        recalls = right_counts / true_counts
        precisions = right_counts / predicted_counts
    accuracy = float(right_counts.sum() / counts.sum()) if counts.sum() > 0 else float("nan")

    # Background of the detection matrices: its cell, its recall and its precision are undefined
    if background_index is not None:
        shares[background_index, background_index] = np.nan
        recalls[background_index], precisions[background_index] = np.nan, np.nan

    # One colour scale for the matrix and its strips, the rows without samples in grey
    colour_map = colormaps[COLOUR_MAP_NAME].with_extremes(bad=EMPTY_ROW_COLOUR)
    normalization = Normalize(vmin=0.0, vmax=1.0)

    # Figure laid out as the matrix, the recall column, the colour bar, the precision row and the accuracy corner
    side_per_class = PERCENT_MATRIX_SIDE_PER_CLASS if in_percent else MATRIX_SIDE_PER_CLASS
    matrix_side = min(max(side_per_class * number_classes, MINIMUM_MATRIX_SIDE), MAXIMUM_MATRIX_SIDE)
    figure = Figure(figsize=(matrix_side + STRIP_WIDTH + COLOUR_BAR_WIDTH + 3.0, matrix_side + STRIP_WIDTH + 3.0),
                    dpi=IMAGE_DOTS_PER_INCH, layout="constrained")
    canvas = FigureCanvasAgg(figure=figure)
    grid_specification = figure.add_gridspec(nrows=2, ncols=3,
                                             width_ratios=[matrix_side, STRIP_WIDTH, COLOUR_BAR_WIDTH],
                                             height_ratios=[matrix_side, STRIP_WIDTH])
    matrix_axes = figure.add_subplot(grid_specification[0, 0])
    recall_axes = figure.add_subplot(grid_specification[0, 1], sharey=matrix_axes)
    colour_bar_axes = figure.add_subplot(grid_specification[0, 2])
    precision_axes = figure.add_subplot(grid_specification[1, 0], sharex=matrix_axes)
    accuracy_axes = figure.add_subplot(grid_specification[1, 1])

    # Matrix: shares of the true class, class names on both sides, predicted classes on top
    image = matrix_axes.imshow(X=np.ma.masked_invalid(shares), cmap=colour_map, norm=normalization,
                               interpolation="nearest", aspect="auto")
    matrix_axes.xaxis.tick_top()
    matrix_axes.xaxis.set_label_position(position="top")
    set_class_ticks(axes=matrix_axes, class_names=class_names, along_columns=True)
    set_class_ticks(axes=matrix_axes, class_names=class_names, along_columns=False)
    matrix_axes.set_xlabel(xlabel="Predicted Class", fontsize=10, fontweight="bold")
    matrix_axes.set_ylabel(ylabel="True Class", fontsize=10, fontweight="bold")
    if number_classes <= MAXIMUM_CLASSES_WITH_COUNTS:
        write_cell_texts(axes=matrix_axes, counts=counts, shares=shares, colour_map=colour_map,
                         normalization=normalization, show_counts=show_counts, background_index=background_index,
                         in_percent=in_percent)

        # Thin white separators between the cells
        matrix_axes.set_xticks(ticks=np.arange(-0.5, number_classes, 1.0), minor=True)
        matrix_axes.set_yticks(ticks=np.arange(-0.5, number_classes, 1.0), minor=True)
        matrix_axes.grid(which="minor", color="white", linewidth=1.0)
        matrix_axes.tick_params(which="minor", length=0)

    # Recall column, precision row and accuracy corner on the same colour scale
    recall_axes.imshow(X=np.ma.masked_invalid(recalls[:, np.newaxis]), cmap=colour_map, norm=normalization,
                       interpolation="nearest", aspect="auto")
    precision_axes.imshow(X=np.ma.masked_invalid(precisions[np.newaxis, :]), cmap=colour_map, norm=normalization,
                          interpolation="nearest", aspect="auto")
    accuracy_axes.imshow(X=np.ma.masked_invalid(np.array([[accuracy]])), cmap=colour_map, norm=normalization,
                         interpolation="nearest", aspect="auto")
    if number_classes <= MAXIMUM_CLASSES_WITH_COUNTS:
        write_strip_texts(axes=recall_axes, values=recalls, colour_map=colour_map, normalization=normalization,
                          vertical=True, in_percent=in_percent)
        write_strip_texts(axes=precision_axes, values=precisions, colour_map=colour_map, normalization=normalization,
                          vertical=False, in_percent=in_percent)
    if not np.isnan(accuracy):
        accuracy_colour = compute_text_colour(cell_colour=colour_map(normalization(accuracy)))
        accuracy_axes.text(x=0, y=0, s=format_share(share=accuracy, in_percent=in_percent), ha="center", va="center",
                           fontsize=8,
                           fontweight="bold", color=accuracy_colour)

    # Strip labels, without ticks of their own
    recall_axes.set_xticks(ticks=[0], labels=["Recall"], fontsize=9, fontweight="bold")
    recall_axes.xaxis.tick_top()
    recall_axes.tick_params(axis="y", which="both", left=False, labelleft=False)
    precision_axes.set_yticks(ticks=[0], labels=["Precision"], fontsize=9, fontweight="bold")
    precision_axes.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    accuracy_axes.set_xticks(ticks=[0], labels=["Accuracy"], fontsize=9, fontweight="bold")
    accuracy_axes.set_yticks(ticks=[])

    # Colour bar of the shares, as percentages
    colour_bar = figure.colorbar(mappable=image, cax=colour_bar_axes)
    colour_bar.ax.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick) if in_percent else
                                            PercentFormatter(xmax=1.0))
    colour_bar.set_label(label="Share Of The True Class", fontsize=9)
    figure.suptitle(t=title, fontsize=12, fontweight="bold")

    # Render the figure into an RGB array
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def save_confusion_matrix_table(confusion_matrix: torch.Tensor, class_names: List[str], output_folder: Path) -> Path:
    """
    Saves the confusion matrix of an evaluated checkpoint as a CSV table in its evaluation folder, whatever the
    per-class report, so that the confusions between classes can be inspected after the training.

    Args:
        confusion_matrix (torch.Tensor): The [K, K] confusion matrix (rows true classes, columns predicted classes).
        class_names (List[str]): The name of every class.
        output_folder (Path): The evaluation folder of the checkpoint, created if needed.

    Returns:
        Path: The path of the table.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    # Table: one row per true class, one column per predicted class
    table_path = output_folder / "confusion_matrix.csv"
    with open(table_path, "w", newline="") as table_file:
        table_writer = csv.writer(table_file)
        table_writer.writerow(["true \\ predicted"] + class_names)
        for class_name, row in zip(class_names, confusion_matrix.tolist()):
            table_writer.writerow([class_name] + row)
    return table_path


def save_confusion_matrix_image(confusion_matrix: torch.Tensor, class_names: List[str], output_folder: Path,
                                title: str, show_counts: bool = True, background_index: Optional[int] = None,
                                in_percent: bool = False) -> Tuple[Path, np.ndarray]:
    """
    Draws the confusion matrix of an evaluated checkpoint (draw_confusion_matrix) and saves it as
    'confusion_matrix.png' in its evaluation folder, for the per-class report, which also logs the returned image to
    TensorBoard: both come from the same array, so the saved image and the logged one are identical.

    Args:
        confusion_matrix (torch.Tensor): The [K, K] confusion matrix (rows true classes, columns predicted classes).
        class_names (List[str]): The name of every class.
        output_folder (Path): The evaluation folder of the checkpoint, created if needed.
        title (str): The title of the image.
        show_counts (bool): Whether the cells show their count, or only their share of the true class. Defaults to
            True.
        background_index (Optional[int]): The index of the background row and column of the detection matrices, None
            for a matrix without background. Defaults to None.
        in_percent (bool): Whether the image writes the percentages of the reports, True for the classification image.
            Defaults to False.

    Returns:
        Tuple[Path, np.ndarray]: The path of the image and the [H, W, 3] uint8 RGB image.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    image = draw_confusion_matrix(confusion_matrix=confusion_matrix, class_names=class_names, title=title,
                                  show_counts=show_counts, background_index=background_index, in_percent=in_percent)
    image_path = output_folder / "confusion_matrix.png"
    cv2.imwrite(filename=str(image_path), img=cv2.cvtColor(src=image, code=cv2.COLOR_RGB2BGR))
    return image_path, image
