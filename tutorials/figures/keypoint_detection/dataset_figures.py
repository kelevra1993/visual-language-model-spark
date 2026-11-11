"""
Figures of the first chapter of the keypoint detection series of the tutorial videos (labels, channels and targets),
made by tutorials/figures/keypoint_detection/figures.py from the test image of the chapters: its records of the COCO
labels.json written as a listing, the image with its labelled keypoints, the heatmap channels of every class read from
the channel table of the trained detector (the data of the channel table animation), and the targets that
build_keypoint_targets draws for the image at the strides 4 and 2, with the values of the cells around one keypoint
for the grid of the storyboard.
"""

import json
import operator

import matplotlib
import numpy

from matplotlib import pyplot
from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, POINTS_PER_PIXEL,
                                                SURFACE_COLOUR, TEXT_PRIMARY_COLOUR, TEXT_SECONDARY_COLOUR,
                                                register_fonts, round_fact, save_figure)
from tutorials.figures.keypoint_detection.keypoint_inference import build_sample_targets
from tutorials.figures.keypoint_detection.keypoint_sources import OBJECT_COLOURS, write_json
from tutorial_builder.layout import NAMED_COLOURS

# Size of the labelled test image of the labels.json scene, the image element of the diagram
LABELLED_IMAGE_SIZE = 660

# Half side of the window of target cells shown by the grid of the storyboard, around the cell of one keypoint
GRID_HALF_SIDE = 4

# Class and keypoint whose target cells the grid shows
GRID_CLASS_NAME = "trapezoid"
GRID_KEYPOINT_NAME = "top_left"

# Side of the three square panels of the targets figure, and the gap between them
TARGET_PANEL_SIDE = 560
TARGET_PANEL_GAP = 20
TARGET_PANEL_TOP = 110

# Colour map of the heatmaps: the card surface for 0, the blue of the palette, then the primary text grey for 1
HEATMAP_COLOUR_MAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    name="tutorial_heatmap", colors=[SURFACE_COLOUR, "#%02x%02x%02x" % NAMED_COLOURS["blue"], TEXT_PRIMARY_COLOUR])


def format_json_value(value: Any) -> str:
    """
    Writes one value of a labels.json record as JSON on one line, with a space after every comma, the way the listing
    of the storyboard shows it.

    Args:
        value (Any): The value.

    Returns:
        str: Its JSON text.
    """
    return json.dumps(obj=value, separators=(", ", ": "))


def wrap_json_list(prefix: str, values: List[Any], values_per_line: int, indentation: int) -> List[str]:
    """
    Writes a long JSON list of a labels.json record over several lines of the listing, a fixed number of values per
    line, the next lines aligned under the first value.

    Args:
        prefix (str): The text before the list, such as '"keypoints": '.
        values (List[Any]): The values of the list.
        values_per_line (int): The number of values on every line.
        indentation (int): The spaces before the prefix.

    Returns:
        List[str]: The lines of the list, the last one ending with the closing bracket and a comma.
    """
    lines = []
    for line_start in range(0, len(values), values_per_line):
        line_values = ", ".join(format_json_value(value=value)
                                for value in values[line_start:line_start + values_per_line])
        is_first_line = line_start == 0
        is_last_line = line_start + values_per_line >= len(values)
        opening = " " * indentation + prefix + "[" if is_first_line else " " * (indentation + len(prefix) + 1)
        lines.append(opening + line_values + ("]," if is_last_line else ","))
    return lines


def write_labels_listing(output_folder: Path, sample_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Writes the records of the test image in labels.json as the listing of the first chapter: the image, the trapezoid
    with its keypoint triplets, the circle distractor without keypoints, and the trapezoid category with its keypoint
    names, skeleton and extra field, all read from the manifest.

    Args:
        output_folder (Path): The figures folder of the series.
        sample_record (Dict[str, Any]): The test image of load_test_sample.

    Returns:
        Dict[str, Any]: The facts of the listing: the counts of the annotations of the image, the extra field and the
            number of distractor categories (without keypoints) of the manifest.
    """
    categories = {category["id"]: category for category in sample_record["categories"]}
    annotations = sample_record["annotations"]
    trapezoid = next(annotation for annotation in annotations if categories[annotation["category_id"]]["name"] ==
                     GRID_CLASS_NAME)
    distractor = next(annotation for annotation in annotations if not annotation["keypoints"])
    trapezoid_category = categories[trapezoid["category_id"]]
    extra_fields = [key for key in trapezoid_category if key not in ("id", "name", "keypoints", "skeleton",
                                                                       "supercategory")]

    # Image record, then the two annotations, then the category, every record with its own fields
    image_record = sample_record["image_record"]
    lines = ['"images": [',
             f'  {{"id": {image_record["id"]}, "file_name": {format_json_value(value=image_record["file_name"])},',
             f'   "width": {image_record["width"]}, "height": {image_record["height"]}}}, ...],',
             '"annotations": [',
             f'  {{"id": {trapezoid["id"]}, "image_id": {trapezoid["image_id"]}, '
             f'"category_id": {trapezoid["category_id"]},',
             f'   "bbox": {format_json_value(value=trapezoid["bbox"])}, "area": {trapezoid["area"]},']
    lines += wrap_json_list(prefix='"keypoints": ', values=trapezoid["keypoints"], values_per_line=6, indentation=3)
    lines += [f'   "num_keypoints": {trapezoid["num_keypoints"]}, "iscrowd": {trapezoid["iscrowd"]}}},',
              f'  {{"id": {distractor["id"]}, "image_id": {distractor["image_id"]}, '
              f'"category_id": {distractor["category_id"]},',
              f'   "keypoints": [], "num_keypoints": {distractor["num_keypoints"]}, ...}}, ...],',
              '"categories": [',
              f'  {{"id": {trapezoid_category["id"]}, "name": {format_json_value(value=trapezoid_category["name"])},']
    lines += wrap_json_list(prefix='"keypoints": ', values=trapezoid_category["keypoints"], values_per_line=2,
                            indentation=3)
    field_lines = [f'   "skeleton": {format_json_value(value=trapezoid_category["skeleton"])}']
    field_lines += [f'   {format_json_value(value=field_name)}: '
                    f'{format_json_value(value=trapezoid_category[field_name])}' for field_name in extra_fields]

    # Every field line but the last ends with a comma, the last one closing the category and the list
    lines += [field_line + "," for field_line in field_lines[:-1]] + [field_lines[-1] + "}, ...]"]

    # Listing file read by the listing element of the storyboard
    listing_path = output_folder / "dataset" / "labels_entry.txt"
    listing_path.parent.mkdir(parents=True, exist_ok=True)
    listing_path.write_text(data="\n".join(lines) + "\n", encoding="utf-8")
    return {"labels_image_annotations": len(annotations),
            "labels_image_objects_with_keypoints": sum(1 for annotation in annotations if annotation["keypoints"]),
            "labels_distractor_name": categories[distractor["category_id"]]["name"],
            "labels_distractor_count": sum(1 for category in categories.values() if not category["keypoints"]),
            "labels_extra_field": extra_fields[0] if extra_fields else "none",
            "labels_trapezoid_category_id": trapezoid["category_id"]}


def draw_labelled_image(output_folder: Path, sample_record: Dict[str, Any], objects: List[Dict[str, Any]]) -> None:
    """
    Draws the test image with the labelled keypoints of its objects, every object in its colour and the keypoints of
    the trapezoid named, next to the listing of its labels.json records.

    Args:
        output_folder (Path): The figures folder of the series.
        sample_record (Dict[str, Any]): The test image of load_test_sample.
        objects (List[Dict[str, Any]]): The objects of describe_sample_objects.
    """
    register_fonts()
    image = sample_record["sample"]["images"].permute(1, 2, 0).numpy()[..., ::-1]
    figure = pyplot.figure(figsize=(LABELLED_IMAGE_SIZE / DOTS_PER_INCH, LABELLED_IMAGE_SIZE / DOTS_PER_INCH),
                           dpi=DOTS_PER_INCH, facecolor=SURFACE_COLOUR)
    axes = figure.add_axes(rect=(0.0, 0.0, 1.0, 1.0))
    axes.imshow(X=image, interpolation="lanczos")
    axes.set_axis_off()

    # Keypoints of every object in its colour, the names of the trapezoid corners next to them
    for object_index, sample_object in enumerate(objects):
        positions = numpy.array(sample_object["positions"])
        axes.scatter(positions[:, 0], positions[:, 1], s=(9 * POINTS_PER_PIXEL) ** 2,
                     color=OBJECT_COLOURS[object_index], edgecolors=TEXT_PRIMARY_COLOUR,
                     linewidths=1.5 * POINTS_PER_PIXEL, zorder=3)
        if sample_object["class_name"] == GRID_CLASS_NAME:
            for keypoint_name, (position_x, position_y) in zip(sample_object["keypoint_names"], positions):
                is_top = keypoint_name.startswith("top")
                axes.annotate(text=keypoint_name, xy=(position_x, position_y),
                              xytext=(0, 14 if is_top else -14), textcoords="offset pixels",
                              ha="center", va="bottom" if is_top else "top", color=TEXT_PRIMARY_COLOUR,
                              fontsize=17 * POINTS_PER_PIXEL)

    # The distractor named on its box, without any keypoint
    categories = {category["id"]: category["name"] for category in sample_record["categories"]}
    for annotation in sample_record["annotations"]:
        if not annotation["keypoints"]:
            box_left, box_top, box_width, box_height = annotation["bbox"]
            axes.text(x=box_left + box_width / 2, y=box_top + box_height / 2,
                      s=f"{categories[annotation['category_id']]}\nno key points", ha="center", va="center",
                      color=TEXT_PRIMARY_COLOUR, fontsize=17 * POINTS_PER_PIXEL)
    save_figure(figure=figure, output_path=output_folder / "dataset" / "labelled_image.png")
    pyplot.close(fig=figure)


def write_channel_table(output_folder: Path, channel_table: numpy.ndarray,
                        keypoint_categories: Dict[int, Dict[str, Any]],
                        objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Writes the data of the channel table animation: the first channel and keypoint count of every class, from the
    channel table of the trained detector, and the classes of the test image highlighted by the storyboard.

    Args:
        output_folder (Path): The figures folder of the series.
        channel_table (numpy.ndarray): The channel of every keypoint of every class.
        keypoint_categories (Dict[int, Dict[str, Any]]): The keypoint definition of every class.
        objects (List[Dict[str, Any]]): The objects of describe_sample_objects.

    Returns:
        Dict[str, Any]: The facts of the channels: their number, the classes, the most keypoints of a class and the
            channels of the classes of the test image.
    """
    classes = []
    for class_index in sorted(keypoint_categories):
        class_channels = channel_table[class_index][channel_table[class_index] >= 0]
        classes.append({"index": int(class_index), "name": keypoint_categories[class_index]["name"],
                        "first_channel": int(class_channels[0]), "count": int(len(class_channels))})
    total_channels = int(channel_table.max()) + 1
    largest_class = max(classes, key=operator.itemgetter("count"))
    image_classes = sorted({sample_object["class_name"] for sample_object in objects})
    write_json(data={"classes": classes, "total_channels": total_channels, "table_rows": int(channel_table.shape[0]),
                     "table_columns": int(channel_table.shape[1]), "highlighted_classes": image_classes},
               output_path=output_folder / "dataset" / "channel_table.json")

    # Facts of the channels of every class of the image, by class name
    facts = {"channels_total": total_channels, "channels_classes": len(classes),
             "channels_table_rows": int(channel_table.shape[0]), "channels_table_columns": int(channel_table.shape[1]),
             "channels_largest_class": largest_class["name"].replace("_", " "),
             "channels_largest_count": largest_class["count"]}
    for class_record in classes:
        if class_record["name"] in image_classes or class_record["name"] in ("square", "house"):
            facts[f"channels_{class_record['name']}_first"] = class_record["first_channel"]
            facts[f"channels_{class_record['name']}_last"] = class_record["first_channel"] + class_record["count"] - 1
            facts[f"channels_{class_record['name']}_count"] = class_record["count"]
    return facts


def draw_heatmap_panel(figure: matplotlib.figure.Figure, left: int, image: numpy.ndarray, title: str,
                       subtitle: str) -> matplotlib.axes.Axes:
    """
    Draws one square panel of the targets figure, an image or a heatmap shown without smoothing, under its title and
    subtitle.

    Args:
        figure (matplotlib.figure.Figure): The figure.
        left (int): The left edge of the panel in pixels.
        image (numpy.ndarray): The RGB image, or the [h, w] heatmap within [0, 1].
        title (str): The title of the panel.
        subtitle (str): The line under the title.

    Returns:
        matplotlib.axes.Axes: The panel.
    """
    figure_width, figure_height = figure.get_figwidth() * DOTS_PER_INCH, figure.get_figheight() * DOTS_PER_INCH
    axes = figure.add_axes(rect=(left / figure_width, 1.0 - (TARGET_PANEL_TOP + TARGET_PANEL_SIDE) / figure_height,
                                 TARGET_PANEL_SIDE / figure_width, TARGET_PANEL_SIDE / figure_height))
    if image.ndim == 2:
        axes.imshow(X=image, cmap=HEATMAP_COLOUR_MAP, vmin=0.0, vmax=1.0, interpolation="nearest")
    else:
        axes.imshow(X=image, interpolation="lanczos")
    axes.set_xticks(ticks=[])
    axes.set_yticks(ticks=[])
    for spine in axes.spines.values():
        spine.set_color("#%02x%02x%02x" % NAMED_COLOURS["border"])

    # Title and subtitle above the panel
    figure.text(x=left / figure_width, y=1.0 - 34 / figure_height, s=title, color=TEXT_PRIMARY_COLOUR,
                fontsize=24 * POINTS_PER_PIXEL, ha="left", va="center")
    figure.text(x=left / figure_width, y=1.0 - 72 / figure_height, s=subtitle, color=TEXT_SECONDARY_COLOUR,
                fontsize=19 * POINTS_PER_PIXEL, ha="left", va="center")
    return axes


def make_target_figures(output_folder: Path, sample_record: Dict[str, Any], channel_table: numpy.ndarray,
                        objects: List[Dict[str, Any]], heatmap_sigma: float) -> Dict[str, Any]:
    """
    Builds the targets of the test image with build_keypoint_targets and draws them next to the image at both
    strides (the maximum over the channels, every peak cell at 1), and gives the values of the cells around the top
    left corner of the trapezoid at stride 4 for the grid of the storyboard, with the counts of positive and non-zero
    cells.

    Args:
        output_folder (Path): The figures folder of the series.
        sample_record (Dict[str, Any]): The test image of load_test_sample.
        channel_table (numpy.ndarray): The channel of every keypoint of every class.
        objects (List[Dict[str, Any]]): The objects of describe_sample_objects.
        heatmap_sigma (float): 'KeypointHead.heatmap_sigma' of the run.

    Returns:
        Dict[str, Any]: The facts of the targets.
    """
    register_fonts()
    sample = sample_record["sample"]
    number_channels = int(channel_table.max()) + 1
    targets = build_sample_targets(sample=sample, channel_table=channel_table, number_channels=number_channels,
                                   heatmap_sigma=heatmap_sigma)
    stride_heatmaps = {4: targets["heatmaps"][0][0].numpy(), 2: targets["heatmaps"][1][0].numpy()}

    # Cells of every stride: the keypoint cells at 1, the cells not 0, and all the cells of the 243 channels
    facts = {}
    for stride, heatmaps in stride_heatmaps.items():
        facts[f"targets_stride_{stride}_positive_cells"] = int((heatmaps == 1.0).sum())
        facts[f"targets_stride_{stride}_nonzero_cells"] = int((heatmaps > 0.0).sum())
        facts[f"targets_stride_{stride}_cells"] = int(heatmaps.size)
        facts[f"targets_stride_{stride}_nonzero_share"] = round_fact(value=(heatmaps > 0.0).mean())
        facts[f"targets_stride_{stride}_positive_share"] = round_fact(value=(heatmaps == 1.0).mean())
    facts["targets_trained_keypoints"] = int(targets["tag_valid"].sum())

    # Window of cells around the top left corner of the trapezoid at stride 4, its keypoint in its cell
    trapezoid = next(sample_object for sample_object in objects if sample_object["class_name"] == GRID_CLASS_NAME)
    keypoint_index = trapezoid["keypoint_names"].index(GRID_KEYPOINT_NAME)
    position_x, position_y = trapezoid["positions"][keypoint_index]
    channel = trapezoid["channels"][keypoint_index]
    cell_column, cell_row = int(position_x // 4), int(position_y // 4)
    window = stride_heatmaps[4][channel, cell_row - GRID_HALF_SIDE:cell_row + GRID_HALF_SIDE + 1,
                                cell_column - GRID_HALF_SIDE:cell_column + GRID_HALF_SIDE + 1]
    facts.update({"grid_values": [[round(float(value), 2) for value in row] for row in window],
                  "grid_column_labels": [str(column) for column in range(cell_column - GRID_HALF_SIDE,
                                                                         cell_column + GRID_HALF_SIDE + 1)],
                  "grid_row_labels": [str(row) for row in range(cell_row - GRID_HALF_SIDE,
                                                                    cell_row + GRID_HALF_SIDE + 1)],
                  "grid_keypoint_x": round_fact(value=position_x), "grid_keypoint_y": round_fact(value=position_y),
                  "grid_centre_x": round_fact(value=position_x / 4 - 0.5),
                  "grid_centre_y": round_fact(value=position_y / 4 - 0.5),
                  "grid_cell_column": cell_column, "grid_cell_row": cell_row, "grid_channel": int(channel),
                  "grid_left_value": round_fact(value=window[GRID_HALF_SIDE, GRID_HALF_SIDE - 1]),
                  "grid_right_value": round_fact(value=window[GRID_HALF_SIDE, GRID_HALF_SIDE + 1]),
                  "grid_upper_value": round_fact(value=window[GRID_HALF_SIDE - 1, GRID_HALF_SIDE]),
                  "grid_lower_value": round_fact(value=window[GRID_HALF_SIDE + 1, GRID_HALF_SIDE]),
                  "grid_radius_cells": int(numpy.ceil(3 * heatmap_sigma)), "grid_sigma": heatmap_sigma})

    # Figure of the image and of the targets of both strides, the maximum over the channels
    figure = pyplot.figure(figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH,
                           facecolor=SURFACE_COLOUR)
    image = sample["images"].permute(1, 2, 0).numpy()[..., ::-1]
    panel_lefts = [index * (TARGET_PANEL_SIDE + TARGET_PANEL_GAP) for index in range(3)]
    image_axes = draw_heatmap_panel(figure=figure, left=panel_lefts[0], image=image, title="Test image",
                                    subtitle=f"{facts['targets_trained_keypoints']} labelled key points, 512 x 512")
    for object_index, sample_object in enumerate(objects):
        positions = numpy.array(sample_object["positions"])
        image_axes.scatter(positions[:, 0], positions[:, 1], s=(6 * POINTS_PER_PIXEL) ** 2,
                           color=OBJECT_COLOURS[object_index], edgecolors=TEXT_PRIMARY_COLOUR,
                           linewidths=POINTS_PER_PIXEL, zorder=3)
    draw_heatmap_panel(figure=figure, left=panel_lefts[1], image=stride_heatmaps[4].max(axis=0),
                       title="Targets at stride 4", subtitle="128 x 128 cells, maximum over the 243 channels")
    draw_heatmap_panel(figure=figure, left=panel_lefts[2], image=stride_heatmaps[2].max(axis=0),
                       title="Targets at stride 2", subtitle="256 x 256 cells, maximum over the 243 channels")
    save_figure(figure=figure, output_path=output_folder / "dataset" / "targets.png")
    pyplot.close(fig=figure)

    # Boxes of the panels and of the trapezoid in every panel, in pixels of the figure, for the storyboard
    categories = {category["id"]: category["name"] for category in sample_record["categories"]}
    box_left, box_top, box_width, box_height = next(annotation["bbox"] for annotation in sample_record["annotations"]
                                                    if categories[annotation["category_id"]] == GRID_CLASS_NAME)
    pixel_scale = TARGET_PANEL_SIDE / sample["images"].shape[2]
    for panel_name, panel_left in zip(["image", "stride_4", "stride_2"], panel_lefts):
        facts[f"targets_{panel_name}_panel_box"] = [panel_left, TARGET_PANEL_TOP, panel_left + TARGET_PANEL_SIDE,
                                                    TARGET_PANEL_TOP + TARGET_PANEL_SIDE]
        facts[f"targets_{panel_name}_trapezoid_box"] = [
            round(panel_left + (box_left - 8) * pixel_scale), round(TARGET_PANEL_TOP + (box_top - 8) * pixel_scale),
            round(panel_left + (box_left + box_width + 8) * pixel_scale),
            round(TARGET_PANEL_TOP + (box_top + box_height + 8) * pixel_scale)]
    return facts
