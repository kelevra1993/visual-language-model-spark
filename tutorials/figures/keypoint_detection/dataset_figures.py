"""
Figures of the first chapter of the keypoint detection series of the tutorial videos (labels, channels and targets),
made by tutorials/figures/keypoint_detection/figures.py from the test image of the chapters: the records of its single
object (select_story_objects of keypoint_sources.py, the object alone in its class cut by the border) and of its
distractor in the COCO labels.json written as a listing, the image with its labelled keypoints, the heatmap channels
of every class read from the channel table of the trained detector (the data of the channel table animation), and the
targets that build_keypoint_targets draws for the image at the strides 4 and 2, with the values of the cells around
one keypoint of the single object for the grid of the storyboard.
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
from tutorials.figures.keypoint_detection.keypoint_sources import (OBJECT_COLOURS, find_labelled_positions,
                                                                  format_class_label, write_json)
from tutorial_builder.layout import NAMED_COLOURS
from vision_training.models.keypoints.hrnet import BRANCH_SCALE_FACTOR
from vision_training.models.keypoints.keypoint_detector import HEATMAP_STRIDE

# Size of the labelled test image of the labels.json scene, the image element of the diagram
LABELLED_IMAGE_SIZE = 660

# Texts drawn on the labelled test image: on a background of the card colour, so that they read on the textured and
# light backgrounds of the synthetic images, and aligned on the side of the image when nearer to it than this margin
# (image pixels)
LABEL_BACKGROUND = {"boxstyle": "round,pad=0.25", "facecolor": SURFACE_COLOUR, "edgecolor": "none", "alpha": 0.8}
LABEL_SIDE_MARGIN = 60

# Keypoint names closer than this along x (image pixels) on the same side of their keypoints are drawn one line apart,
# a line being this many pixels
LABEL_MINIMUM_SPACING = 150
LABEL_LINE_HEIGHT = 30

# Room (image pixels) the two lines of the distractor name need under its box, the name going above the box otherwise
DISTRACTOR_LABEL_ROOM = 60

# Half side of the window of target cells shown by the grid of the storyboard, around the cell of one keypoint of the
# single object, the first one whose window lies inside the heatmap and whose left and upper neighbours are higher than
# their opposite ones, as the narration says
GRID_HALF_SIDE = 4

# Largest distance between a keypoint of an annotation of labels.json and the same keypoint of the dataset sample, in
# pixels, the sample being the image at its own size
ANNOTATION_MATCH_TOLERANCE = 0.01

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


def find_object_annotation(sample_record: Dict[str, Any], sample_object: Dict[str, Any]) -> Dict[str, Any]:
    """
    Finds the labels.json annotation of an object of the dataset sample of the test image, the one of its class whose
    keypoint triplets are those of the object, so that the listing shows the records of the object the chapters follow.

    Args:
        sample_record (Dict[str, Any]): The test image of load_test_sample.
        sample_object (Dict[str, Any]): An object of describe_sample_objects.

    Returns:
        Dict[str, Any]: The annotation of the object.

    Raises:
        ValueError: If no annotation, or several, match the object.
    """
    categories = {category["id"]: category for category in sample_record["categories"]}
    object_keypoints = numpy.concatenate([numpy.array(sample_object["positions"]),
                                          numpy.array(sample_object["visibilities"])[:, None]], axis=1)
    matches = [annotation for annotation in sample_record["annotations"]
               if categories[annotation["category_id"]]["name"] == sample_object["class_name"]
               and len(annotation["keypoints"]) == object_keypoints.size
               and numpy.allclose(numpy.array(annotation["keypoints"]).reshape(-1, 3), object_keypoints,
                                  atol=ANNOTATION_MATCH_TOLERANCE)]
    if len(matches) != 1:
        raise ValueError(f"{len(matches)} annotations of {sample_record['image_record']['file_name']} match the "
                         f"{sample_object['class_name']} of the dataset sample, instead of one.")
    return matches[0]


def write_labels_listing(output_folder: Path, sample_record: Dict[str, Any], objects: List[Dict[str, Any]],
                         story_objects: Dict[str, Any]) -> Dict[str, Any]:
    """
    Writes the records of the test image in labels.json as the listing of the first chapter: the image, the single
    object (the object alone in its class cut by the border) with its keypoint triplets, the first distractor without
    keypoints, and the category of the single object with its keypoint names, skeleton and any extra field, all read
    from the manifest. The chapter says that a keypoint of the single object outside the image is the triplet of three
    zeros, which is checked.

    Args:
        output_folder (Path): The figures folder of the series.
        sample_record (Dict[str, Any]): The test image of load_test_sample.
        objects (List[Dict[str, Any]]): The objects of describe_sample_objects.
        story_objects (Dict[str, Any]): The objects named by their role, from select_story_objects.

    Returns:
        Dict[str, Any]: The facts of the listing: the counts of the annotations of the image, the distractor, the
            extra field and the number of distractor categories (without keypoints) of the manifest.

    Raises:
        ValueError: If the image holds no distractor, or if the unlabelled keypoint of the single object is not the
            triplet of zeros of labels.json.
    """
    categories = {category["id"]: category for category in sample_record["categories"]}
    annotations = sample_record["annotations"]
    single_annotation = find_object_annotation(sample_record=sample_record,
                                               sample_object=objects[story_objects["single"]])
    distractors = [annotation for annotation in annotations if not annotation["keypoints"]]
    if not distractors:
        raise ValueError(f"No distractor in {sample_record['image_record']['file_name']}, whose records the listing "
                         f"and the narration show.")
    distractor = distractors[0]
    single_category = categories[single_annotation["category_id"]]
    extra_fields = [key for key in single_category if key not in ("id", "name", "keypoints", "skeleton",
                                                                    "supercategory")]

    # Keypoints of the single object outside the image, written as three zeros (not labelled)
    triplets = numpy.array(single_annotation["keypoints"]).reshape(-1, 3)
    unlabelled_triplets = triplets[triplets[:, 2] == 0]
    if not len(unlabelled_triplets) or numpy.any(unlabelled_triplets != 0):
        raise ValueError(f"The unlabelled keypoints of the {single_category['name']} of the listing are "
                         f"{unlabelled_triplets.tolist()}, but the narration says they are triplets of zeros.")

    # Image record, then the two annotations, then the category, every record with its own fields
    image_record = sample_record["image_record"]
    lines = ['"images": [',
             f'  {{"id": {image_record["id"]}, "file_name": {format_json_value(value=image_record["file_name"])},',
             f'   "width": {image_record["width"]}, "height": {image_record["height"]}}}, ...],',
             '"annotations": [',
             f'  {{"id": {single_annotation["id"]}, "image_id": {single_annotation["image_id"]}, '
             f'"category_id": {single_annotation["category_id"]},',
             f'   "bbox": {format_json_value(value=single_annotation["bbox"])}, "area": {single_annotation["area"]},']
    lines += wrap_json_list(prefix='"keypoints": ', values=single_annotation["keypoints"], values_per_line=6,
                            indentation=3)
    lines += [f'   "num_keypoints": {single_annotation["num_keypoints"]}, '
              f'"iscrowd": {single_annotation["iscrowd"]}}},',
              f'  {{"id": {distractor["id"]}, "image_id": {distractor["image_id"]}, '
              f'"category_id": {distractor["category_id"]},',
              f'   "keypoints": [], "num_keypoints": {distractor["num_keypoints"]}, ...}}, ...],',
              '"categories": [',
              f'  {{"id": {single_category["id"]}, "name": {format_json_value(value=single_category["name"])},']
    lines += wrap_json_list(prefix='"keypoints": ', values=single_category["keypoints"], values_per_line=2,
                            indentation=3)
    field_lines = [f'   "skeleton": {format_json_value(value=single_category["skeleton"])}']
    field_lines += [f'   {format_json_value(value=field_name)}: '
                    f'{format_json_value(value=single_category[field_name])}' for field_name in extra_fields]

    # Every field line but the last ends with a comma, the last one closing the category and the list
    lines += [field_line + "," for field_line in field_lines[:-1]] + [field_lines[-1] + "}, ...]"]

    # Listing file read by the listing element of the storyboard
    listing_path = output_folder / "dataset" / "labels_entry.txt"
    listing_path.parent.mkdir(parents=True, exist_ok=True)
    listing_path.write_text(data="\n".join(lines) + "\n", encoding="utf-8")
    return {"labels_image_annotations": len(annotations),
            "labels_image_objects_with_keypoints": sum(1 for annotation in annotations if annotation["keypoints"]),
            "labels_distractor_name": format_class_label(class_name=categories[distractor["category_id"]]["name"]),
            "labels_distractor_count": sum(1 for category in categories.values() if not category["keypoints"]),
            "labels_extra_field": extra_fields[0] if extra_fields else "none",
            "labels_single_category_id": single_annotation["category_id"],
            "labels_single_keypoint_count": len(single_category["keypoints"]),
            "labels_single_labelled_keypoints": single_annotation["num_keypoints"]}


def draw_labelled_image(output_folder: Path, sample_record: Dict[str, Any], objects: List[Dict[str, Any]],
                        story_objects: Dict[str, Any]) -> None:
    """
    Draws the test image with the labelled keypoints of its objects, every object in its colour and the labelled
    keypoints of the single object named, next to the listing of its labels.json records; the keypoints outside the
    image, not labelled, are not drawn.

    Args:
        output_folder (Path): The figures folder of the series.
        sample_record (Dict[str, Any]): The test image of load_test_sample.
        objects (List[Dict[str, Any]]): The objects of describe_sample_objects.
        story_objects (Dict[str, Any]): The objects named by their role, from select_story_objects.
    """
    register_fonts()
    image = sample_record["sample"]["images"].permute(1, 2, 0).numpy()[..., ::-1]
    figure = pyplot.figure(figsize=(LABELLED_IMAGE_SIZE / DOTS_PER_INCH, LABELLED_IMAGE_SIZE / DOTS_PER_INCH),
                           dpi=DOTS_PER_INCH, facecolor=SURFACE_COLOUR)
    axes = figure.add_axes(rect=(0.0, 0.0, 1.0, 1.0))
    axes.imshow(X=image, interpolation="lanczos")
    axes.set_axis_off()

    # Labelled keypoints of every object in its colour, the names of those of the single object next to them
    for object_index, sample_object in enumerate(objects):
        positions = find_labelled_positions(sample_object=sample_object)
        axes.scatter(positions[:, 0], positions[:, 1], s=(9 * POINTS_PER_PIXEL) ** 2,
                     color=OBJECT_COLOURS[object_index], edgecolors=TEXT_PRIMARY_COLOUR,
                     linewidths=1.5 * POINTS_PER_PIXEL, zorder=3)
        if object_index == story_objects["single"]:
            labelled_names = [name for name, visibility in zip(sample_object["keypoint_names"],
                                                               sample_object["visibilities"]) if visibility > 0]
            placed_labels = []
            for keypoint_name, (position_x, position_y) in zip(labelled_names, positions):
                is_top = keypoint_name.startswith("top")
                # Names near a side of the image aligned on that side, so that they stay inside it, and a name moved
                # one line further from its keypoint when a name already placed on the same side is too close
                alignment = ("left" if position_x < LABEL_SIDE_MARGIN else
                             "right" if position_x > image.shape[1] - LABEL_SIDE_MARGIN else "center")
                line = sum(1 for placed_x, placed_y, placed_top in placed_labels
                           if placed_top == is_top and abs(placed_x - position_x) < LABEL_MINIMUM_SPACING
                           and abs(placed_y - position_y) < LABEL_LINE_HEIGHT)
                placed_labels.append((position_x, position_y, is_top))
                label_offset = 14 + line * LABEL_LINE_HEIGHT
                axes.annotate(text=keypoint_name, xy=(position_x, position_y),
                              xytext=(0, label_offset if is_top else -label_offset), textcoords="offset pixels",
                              ha=alignment, va="bottom" if is_top else "top", color=TEXT_PRIMARY_COLOUR,
                              fontsize=17 * POINTS_PER_PIXEL, bbox=LABEL_BACKGROUND)

    # The distractor named under its box, without any keypoint, or above it when the box ends near the bottom border,
    # so that the name leaves the shape visible
    categories = {category["id"]: category["name"] for category in sample_record["categories"]}
    for annotation in sample_record["annotations"]:
        if not annotation["keypoints"]:
            box_left, box_top, box_width, box_height = annotation["bbox"]
            is_below = box_top + box_height + DISTRACTOR_LABEL_ROOM < image.shape[0]
            axes.annotate(text=f"{format_class_label(class_name=categories[annotation['category_id']])}\nno key points",
                          xy=(box_left + box_width / 2, box_top + box_height if is_below else box_top),
                          xytext=(0, -8 if is_below else 8), textcoords="offset pixels", ha="center",
                          va="top" if is_below else "bottom", color=TEXT_PRIMARY_COLOUR,
                          fontsize=17 * POINTS_PER_PIXEL, bbox=LABEL_BACKGROUND)
    save_figure(figure=figure, output_path=output_folder / "dataset" / "labelled_image.png")
    pyplot.close(fig=figure)


def write_channel_table(output_folder: Path, channel_table: numpy.ndarray,
                        keypoint_categories: Dict[int, Dict[str, Any]], objects: List[Dict[str, Any]],
                        story_objects: Dict[str, Any]) -> Dict[str, Any]:
    """
    Writes the data of the channel table animation: the first channel and keypoint count of every class, from the
    channel table of the trained detector, and the classes of the single object and of the pair of the test image,
    highlighted by the storyboard, which names their channels.

    Args:
        output_folder (Path): The figures folder of the series.
        channel_table (numpy.ndarray): The channel of every keypoint of every class.
        keypoint_categories (Dict[int, Dict[str, Any]]): The keypoint definition of every class.
        objects (List[Dict[str, Any]]): The objects of describe_sample_objects.
        story_objects (Dict[str, Any]): The objects named by their role, from select_story_objects.

    Returns:
        Dict[str, Any]: The facts of the channels: their number, the classes, the most keypoints of a class and the
            channels of the classes of the single object and of the pair.
    """
    classes = []
    for class_index in sorted(keypoint_categories):
        class_channels = channel_table[class_index][channel_table[class_index] >= 0]
        classes.append({"index": int(class_index), "name": keypoint_categories[class_index]["name"],
                        "first_channel": int(class_channels[0]), "count": int(len(class_channels))})
    total_channels = int(channel_table.max()) + 1
    largest_class = max(classes, key=operator.itemgetter("count"))
    role_classes = {"single": objects[story_objects["single"]]["class_name"],
                    "pair": objects[story_objects["pair"][0]]["class_name"]}
    write_json(data={"classes": classes, "total_channels": total_channels, "table_rows": int(channel_table.shape[0]),
                     "table_columns": int(channel_table.shape[1]),
                     "highlighted_classes": sorted(role_classes.values())},
               output_path=output_folder / "dataset" / "channel_table.json")

    # Facts of the channels of the classes of the single object and of the pair, by role
    facts = {"channels_total": total_channels, "channels_classes": len(classes),
             "channels_table_rows": int(channel_table.shape[0]), "channels_table_columns": int(channel_table.shape[1]),
             "channels_largest_class": format_class_label(class_name=largest_class["name"]),
             "channels_largest_count": largest_class["count"]}
    for role, class_name in role_classes.items():
        class_record = next(record for record in classes if record["name"] == class_name)
        facts[f"channels_{role}_first"] = class_record["first_channel"]
        facts[f"channels_{role}_last"] = class_record["first_channel"] + class_record["count"] - 1
        facts[f"channels_{role}_count"] = class_record["count"]
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


def find_grid_keypoint(objects: List[Dict[str, Any]], story_objects: Dict[str, Any],
                       stride_4_heatmaps: numpy.ndarray) -> int:
    """
    Chooses the keypoint of the single object whose target cells the grid of the storyboard shows: the first labelled
    one whose window of GRID_HALF_SIDE cells around its cell lies inside the heatmap, and whose left and upper
    neighbours are higher than its right and lower ones, as the narration says (the keypoint left of and above the
    centre of its cell).

    Args:
        objects (List[Dict[str, Any]]): The objects of describe_sample_objects.
        story_objects (Dict[str, Any]): The objects named by their role, from select_story_objects.
        stride_4_heatmaps (numpy.ndarray): The [C, h, w] targets of the image at HEATMAP_STRIDE.

    Returns:
        int: The index of the keypoint in its class.

    Raises:
        ValueError: If no keypoint of the single object meets these conditions.
    """
    single_object = objects[story_objects["single"]]
    heatmap_height, heatmap_width = stride_4_heatmaps.shape[1:]
    for keypoint_index, ((position_x, position_y), visibility) in enumerate(zip(single_object["positions"],
                                                                                 single_object["visibilities"])):
        cell_column, cell_row = int(position_x // HEATMAP_STRIDE), int(position_y // HEATMAP_STRIDE)
        window_inside = (GRID_HALF_SIDE <= cell_column < heatmap_width - GRID_HALF_SIDE and
                         GRID_HALF_SIDE <= cell_row < heatmap_height - GRID_HALF_SIDE)
        if visibility == 0 or not window_inside:
            continue
        channel_map = stride_4_heatmaps[single_object["channels"][keypoint_index]]
        left_higher = channel_map[cell_row, cell_column - 1] > channel_map[cell_row, cell_column + 1]
        upper_higher = channel_map[cell_row - 1, cell_column] > channel_map[cell_row + 1, cell_column]
        if left_higher and upper_higher:
            return keypoint_index
    raise ValueError(f"No labelled keypoint of the {single_object['class_name']} of the test image lies left of and "
                     f"above the centre of its cell, {GRID_HALF_SIDE} cells inside the heatmap, as the grid scene "
                     f"says.")


def make_target_figures(output_folder: Path, sample_record: Dict[str, Any], channel_table: numpy.ndarray,
                        objects: List[Dict[str, Any]], story_objects: Dict[str, Any],
                        heatmap_sigma: float) -> Dict[str, Any]:
    """
    Builds the targets of the test image with build_keypoint_targets and draws them next to the image at both
    strides (the maximum over the channels, every peak cell at 1), and gives the values of the cells around one
    keypoint of the single object at HEATMAP_STRIDE (find_grid_keypoint) for the grid of the storyboard, with the
    counts of positive and non-zero cells. The facts of the two strides keep the names stride_4 and stride_2 of the
    other figure families, their strides being HEATMAP_STRIDE and BRANCH_SCALE_FACTOR finer.

    Args:
        output_folder (Path): The figures folder of the series.
        sample_record (Dict[str, Any]): The test image of load_test_sample.
        channel_table (numpy.ndarray): The channel of every keypoint of every class.
        objects (List[Dict[str, Any]]): The objects of describe_sample_objects.
        story_objects (Dict[str, Any]): The objects named by their role, from select_story_objects.
        heatmap_sigma (float): 'KeypointHead.heatmap_sigma' of the run.

    Returns:
        Dict[str, Any]: The facts of the targets.
    """
    register_fonts()
    sample = sample_record["sample"]
    number_channels = int(channel_table.max()) + 1
    targets = build_sample_targets(sample=sample, channel_table=channel_table, number_channels=number_channels,
                                   heatmap_sigma=heatmap_sigma)
    branch_stride = HEATMAP_STRIDE // BRANCH_SCALE_FACTOR
    stride_heatmaps = {"stride_4": (HEATMAP_STRIDE, targets["heatmaps"][0][0].numpy()),
                       "stride_2": (branch_stride, targets["heatmaps"][1][0].numpy())}

    # Cells of every stride: the keypoint cells at 1, the cells not 0, and all the cells of every channel
    facts = {}
    for stride_name, (_, heatmaps) in stride_heatmaps.items():
        facts[f"targets_{stride_name}_positive_cells"] = int((heatmaps == 1.0).sum())
        facts[f"targets_{stride_name}_nonzero_cells"] = int((heatmaps > 0.0).sum())
        facts[f"targets_{stride_name}_cells"] = int(heatmaps.size)
        facts[f"targets_{stride_name}_nonzero_share"] = round_fact(value=(heatmaps > 0.0).mean())
        facts[f"targets_{stride_name}_positive_share"] = round_fact(value=(heatmaps == 1.0).mean())
    facts["targets_trained_keypoints"] = int(targets["tag_valid"].sum())

    # Window of cells around one keypoint of the single object at HEATMAP_STRIDE, its keypoint in its cell
    single_object = objects[story_objects["single"]]
    keypoint_index = find_grid_keypoint(objects=objects, story_objects=story_objects,
                                        stride_4_heatmaps=stride_heatmaps["stride_4"][1])
    position_x, position_y = single_object["positions"][keypoint_index]
    channel = single_object["channels"][keypoint_index]
    cell_column, cell_row = int(position_x // HEATMAP_STRIDE), int(position_y // HEATMAP_STRIDE)
    window = stride_heatmaps["stride_4"][1][channel, cell_row - GRID_HALF_SIDE:cell_row + GRID_HALF_SIDE + 1,
                                            cell_column - GRID_HALF_SIDE:cell_column + GRID_HALF_SIDE + 1]
    facts.update({"grid_class_name": single_object["class_name"],
                  "grid_keypoint_name": single_object["keypoint_names"][keypoint_index],
                  "grid_keypoint_label": format_class_label(class_name=single_object["keypoint_names"][keypoint_index]),
                  "grid_values": [[round(float(value), 2) for value in row] for row in window],
                  "grid_column_labels": [str(column) for column in range(cell_column - GRID_HALF_SIDE,
                                                                         cell_column + GRID_HALF_SIDE + 1)],
                  "grid_row_labels": [str(row) for row in range(cell_row - GRID_HALF_SIDE,
                                                                    cell_row + GRID_HALF_SIDE + 1)],
                  "grid_keypoint_x": round_fact(value=position_x), "grid_keypoint_y": round_fact(value=position_y),
                  "grid_centre_x": round_fact(value=position_x / HEATMAP_STRIDE - 0.5),
                  "grid_centre_y": round_fact(value=position_y / HEATMAP_STRIDE - 0.5),
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
                                    subtitle=f"{facts['targets_trained_keypoints']} labelled key points, "
                                             f"{image.shape[1]} x {image.shape[0]}")
    for object_index, sample_object in enumerate(objects):
        positions = find_labelled_positions(sample_object=sample_object)
        image_axes.scatter(positions[:, 0], positions[:, 1], s=(6 * POINTS_PER_PIXEL) ** 2,
                           color=OBJECT_COLOURS[object_index], edgecolors=TEXT_PRIMARY_COLOUR,
                           linewidths=POINTS_PER_PIXEL, zorder=3)

    # Targets of both strides, the maximum over the channels, with their cell grid and channel count
    for panel_left, (stride, heatmaps) in zip(panel_lefts[1:], stride_heatmaps.values()):
        draw_heatmap_panel(figure=figure, left=panel_left, image=heatmaps.max(axis=0),
                           title=f"Targets at stride {stride}",
                           subtitle=f"{heatmaps.shape[2]} x {heatmaps.shape[1]} cells, maximum over the "
                                    f"{heatmaps.shape[0]} channels")
    save_figure(figure=figure, output_path=output_folder / "dataset" / "targets.png")
    pyplot.close(fig=figure)

    # Boxes of the panels and of the single object in every panel, in pixels of the figure, for the storyboard, the box
    # of the object kept inside its panel
    box_left, box_top, box_right, box_bottom = sample["bounding_boxes"][story_objects["single"]].tolist()
    image_size = sample["images"].shape[2]
    pixel_scale = TARGET_PANEL_SIDE / image_size
    for panel_name, panel_left in zip(["image", "stride_4", "stride_2"], panel_lefts):
        facts[f"targets_{panel_name}_panel_box"] = [panel_left, TARGET_PANEL_TOP, panel_left + TARGET_PANEL_SIDE,
                                                    TARGET_PANEL_TOP + TARGET_PANEL_SIDE]
        facts[f"targets_{panel_name}_single_box"] = [
            round(panel_left + max(box_left - 8, 0) * pixel_scale),
            round(TARGET_PANEL_TOP + max(box_top - 8, 0) * pixel_scale),
            round(panel_left + min(box_right + 8, image_size) * pixel_scale),
            round(TARGET_PANEL_TOP + min(box_bottom + 8, image_size) * pixel_scale)]
    return facts
