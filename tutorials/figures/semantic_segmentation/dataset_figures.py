"""
Figures of the data scenes of the semantic segmentation series of the tutorial videos, made by
tutorials/figures/semantic_segmentation/figures.py from the test split of the synthetic segmentation run: the layout of
a split and the manifest entry of the test image of the chapters written as a listing, with its categories and the
label mapping of the run, and the figure of its mask, the raw category values of the PNG file next to the class
indices the lookup table of the dataset gives, with the pixels of every value.
"""

import json

import cv2
import numpy

from matplotlib import pyplot
from pathlib import Path
from typing import Any, Dict

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, MONOSPACE_FONT_FAMILY,
                                                POINTS_PER_PIXEL, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR,
                                                TEXT_SECONDARY_COLOUR, register_fonts, round_fact, save_figure)
from tutorials.figures.semantic_segmentation.segmentation_sources import build_class_palette
from tutorial_builder.layout import NAMED_COLOURS
from vision_training.data.label_mapping import get_background_class_index
from vision_training.data.segmentation.segmentation_dataset import IGNORED_MASK_VALUE, build_class_lookup_table

# Layout of the mask figure: two square panels and the table of the values, under their titles
MASK_PANEL_SIDE = 560
MASK_PANEL_GAP = 24
MASK_PANEL_TOP = 100


def write_manifest_listing(output_folder: Path, sample_record: Dict[str, Any],
                           label_mapping_path: Path) -> Dict[str, Any]:
    """
    Writes the listing of the data scene: the folders of the test split, the manifest entry of the test image, the
    categories of the manifest and the label mapping file of the run, all read from the files.

    Args:
        output_folder (Path): The figures folder of the series.
        sample_record (Dict[str, Any]): The test image of load_test_sample.
        label_mapping_path (Path): The label mapping file of the run.

    Returns:
        Dict[str, Any]: The facts of the manifest.
    """
    entry = sample_record["manifest_entry"]
    categories = sample_record["categories"]
    mapping = json.loads(label_mapping_path.read_text(encoding="utf-8"))
    split_name = sample_record["image_path"].parents[1].name
    mapping_items = list(mapping.items())

    # Folders of the split, then the manifest entry, the categories and the mapping, shortened in the middle
    lines = [f"{split_name}/", f"├── data/{sample_record['image_path'].name}",
             f"├── masks/{sample_record['mask_path'].name}", "└── labels.json", "",
             "labels.json",
             '  "samples": [...,',
             f'    {{"id": {entry["id"]}, "file_name": {json.dumps(obj=entry["file_name"])},',
             f'     "mask_file_name": {json.dumps(obj=entry["mask_file_name"])}}}, ...]',
             f'  "categories": [{json.dumps(obj=categories[0])}, {json.dumps(obj=categories[1])},',
             f'                 ..., {json.dumps(obj=categories[-1])}]', "",
             f"{label_mapping_path.name}",
             "  {" + ", ".join(f'"{key}": {value}' for key, value in mapping_items[:3]) + ", ..., " +
             ", ".join(f'"{key}": {value}' for key, value in mapping_items[-1:]) + "}"]
    listing_path = output_folder / "dataset" / "manifest_entry.txt"
    listing_path.parent.mkdir(parents=True, exist_ok=True)
    listing_path.write_text(data="\n".join(lines) + "\n", encoding="utf-8")
    return {"manifest_samples": sample_record["number_samples"], "manifest_categories": len(categories),
            "manifest_entry_id": entry["id"], "manifest_mapping_entries": len(mapping)}


def draw_mask_figure(output_folder: Path, sample_record: Dict[str, Any], label_dictionary: Dict[Any, int],
                     label_mode: str) -> Dict[str, Any]:
    """
    Draws the mask of the test image as read from its PNG file (every pixel holding its category value) next to the
    class indices of the sample given by the dataset through build_class_lookup_table, and the table of every value
    of the mask with its class and its pixels.

    Args:
        output_folder (Path): The figures folder of the series.
        sample_record (Dict[str, Any]): The test image of load_test_sample.
        label_dictionary (Dict[Any, int]): The class of every category value.
        label_mode (str): 'Data.label_mapping.mode' of the run.

    Returns:
        Dict[str, Any]: The facts of the mask: its values, their pixels and classes, and the boxes of the panels.

    Raises:
        OSError: If the mask file cannot be read.
    """
    register_fonts()
    raw_mask = cv2.imread(filename=str(sample_record["mask_path"]), flags=cv2.IMREAD_UNCHANGED)
    if raw_mask is None:
        raise OSError(f"The mask {sample_record['mask_path']} cannot be read.")
    lookup_table = build_class_lookup_table(label_mode=label_mode, label_dictionary=label_dictionary)
    class_mask = sample_record["sample"]["masks"].numpy()
    class_names = sample_record["dataset"].class_names
    palette = build_class_palette(number_classes=len(class_names))
    values, counts = numpy.unique(ar=raw_mask, return_counts=True)

    # Raw values shown as grey levels, the classes in the colours of the sample images
    figure = pyplot.figure(figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH,
                           facecolor=SURFACE_COLOUR)
    panel_lefts = [index * (MASK_PANEL_SIDE + MASK_PANEL_GAP) for index in range(3)]
    titles = [(f"masks/{sample_record['mask_path'].name}", f"uint8 values, {raw_mask.shape[1]} x {raw_mask.shape[0]}"),
              ("masks of the sample", "class indices, through the lookup table"),
              ("value -> class", "pixels of the mask")]
    for panel_left, (title, subtitle) in zip(panel_lefts, titles):
        figure.text(x=panel_left / CHART_WIDTH, y=1.0 - 30 / CHART_HEIGHT, s=title, color=TEXT_PRIMARY_COLOUR,
                    fontsize=24 * POINTS_PER_PIXEL, ha="left", va="center")
        figure.text(x=panel_left / CHART_WIDTH, y=1.0 - 66 / CHART_HEIGHT, s=subtitle, color=TEXT_SECONDARY_COLOUR,
                    fontsize=19 * POINTS_PER_PIXEL, ha="left", va="center")
    panel_bottom = 1.0 - (MASK_PANEL_TOP + MASK_PANEL_SIDE) / CHART_HEIGHT
    raw_axes = figure.add_axes(rect=(panel_lefts[0] / CHART_WIDTH, panel_bottom, MASK_PANEL_SIDE / CHART_WIDTH,
                                     MASK_PANEL_SIDE / CHART_HEIGHT))
    raw_axes.imshow(X=numpy.where(raw_mask > 0, 0.3 + 0.07 * raw_mask.astype(numpy.float32), 0.0), cmap="gray",
                    vmin=0.0, vmax=1.0, interpolation="nearest")
    class_axes = figure.add_axes(rect=(panel_lefts[1] / CHART_WIDTH, panel_bottom, MASK_PANEL_SIDE / CHART_WIDTH,
                                       MASK_PANEL_SIDE / CHART_HEIGHT))
    class_axes.imshow(X=palette[numpy.clip(a=class_mask, a_min=0, a_max=None)], interpolation="nearest")

    # Value of every region written on it, and its class name next to the class map
    for value in values[values > 0]:
        rows, columns = numpy.nonzero(raw_mask == value)
        raw_axes.text(x=columns.mean(), y=rows.mean(), s=str(int(value)),
                      color="#%02x%02x%02x" % NAMED_COLOURS["accent"],
                      fontsize=26 * POINTS_PER_PIXEL, ha="center", va="center", fontweight="bold")
        class_axes.text(x=columns.mean(), y=rows.max() + 14, s=class_names[int(lookup_table[value])],
                        color=TEXT_PRIMARY_COLOUR, fontsize=17 * POINTS_PER_PIXEL, ha="center", va="top")
    for axes in [raw_axes, class_axes]:
        axes.set_xticks(ticks=[])
        axes.set_yticks(ticks=[])
        for spine in axes.spines.values():
            spine.set_color("#%02x%02x%02x" % NAMED_COLOURS["border"])

    # Table of the values of the mask, the ignored value last
    table_values = [int(value) for value in values] + ([IGNORED_MASK_VALUE] if IGNORED_MASK_VALUE not in values else [])
    for row_index, value in enumerate(table_values):
        class_index = int(lookup_table[value])
        pixels = int(counts[values == value][0]) if value in values else 0
        name = class_names[class_index] if class_index >= 0 else "ignored"
        figure.text(x=(panel_lefts[2] + 10) / CHART_WIDTH,
                    y=1.0 - (MASK_PANEL_TOP + 30 + row_index * 52) / CHART_HEIGHT,
                    s=f"{value:>3} -> {class_index:>4}  {name:<11}{pixels:>9,}", color=TEXT_PRIMARY_COLOUR,
                    fontsize=22 * POINTS_PER_PIXEL, family=MONOSPACE_FONT_FAMILY, ha="left", va="center")
    save_figure(figure=figure, output_path=output_folder / "dataset" / "mask_values.png")
    pyplot.close(fig=figure)

    # Facts of the values and of the panels, the shapes of the mask being its categories other than the background
    # and the ignored value
    facts = {"mask_values": ", ".join(str(int(value)) for value in values), "mask_size": int(raw_mask.shape[0]),
             "mask_background_class": get_background_class_index(label_dictionary=label_dictionary),
             "mask_shape_count": sum(1 for value in values if 0 < value < IGNORED_MASK_VALUE),
             "mask_pixels": int(raw_mask.size), "mask_ignored_value": IGNORED_MASK_VALUE,
             "mask_ignored_class": int(lookup_table[IGNORED_MASK_VALUE]),
             "mask_background_share": round_fact(value=float(counts[values == 0][0]) / raw_mask.size),
             "mask_shape_names": ", ".join(class_names[int(lookup_table[value])] for value in values if value > 0),
             "mask_lookup_table_size": int(len(lookup_table))}
    for value, count in zip(values, counts):
        facts[f"mask_pixels_{int(value)}"] = int(count)
    for panel_name, panel_left in zip(["raw", "classes", "table"], panel_lefts):
        facts[f"mask_{panel_name}_panel_box"] = [panel_left - 6, MASK_PANEL_TOP - 6, panel_left + MASK_PANEL_SIDE + 6,
                                                 MASK_PANEL_TOP + MASK_PANEL_SIDE + 6]
    return facts
