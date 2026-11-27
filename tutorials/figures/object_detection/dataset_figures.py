"""
Data figures of the object detection tutorials (the chapter on the data, the levels and the anchors), made at every
build by tutorials/figures/object_detection/figures.py from the DET run of tutorials/tutorial_sources.yaml: the
listing of the COCO manifest of its test split (the first image, the first annotation with the start of its compressed
run length encoding and the runs it stands for, the first and last categories) next to the label mapping file and the
label dictionary saved in the run folder, which adds the background, with the classes, the image side and the fewest
and most objects of an image that the narration quotes; the counts of the splits from the data record of the run
(data.json); and the anchor recall of the training objects (the anchor_recall.csv of the evaluation of the run made
again with the current code, which measured the training and test splits at its launch), by relative object size and
detection level, the values of the grid of the anchor recall scene, written in its cells as the percentages of the
reports of the repository and kept between 0 and 1 in the facts the narration quotes.
"""

import collections
import csv
import json
import math

from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import read_run_configuration, round_fact
from tutorials.figures.object_detection.detection_sources import find_test_folder, read_test_manifest, write_listing
from vision_training.data.detection.coco_masks import decode_compressed_counts
from vision_training.evaluation.checkpoint_evaluation import CHECKPOINT_EVALUATION_FOLDER_NAME
from vision_training.evaluation.detection.object_sizes import (OBJECT_SIZE_NAMES, OBJECT_SIZE_SIDE_SHARES,
                                                             describe_object_size_pixels)
from vision_training.evaluation.detection.region_recall import ALL_OBJECTS_GROUP
from vision_training.evaluation.report_formatting import round_percentage

# Characters of the compressed run length encoding of the first annotation kept in the listing (or values of an
# uncompressed one), the runs it stands for being counted
RUN_LENGTH_CHARACTERS_SHOWN = 12

# Rows of anchor_recall.csv in the order of the grid of the scene, all the objects first, then the relative object sizes
# of the evaluation from the smallest; its columns are those of the file, one per detection level of the run, then all
# the levels together; the grid shows the split the anchors are trained on
ANCHOR_RECALL_ROWS = [(ALL_OBJECTS_GROUP, "All objects")] + list(OBJECT_SIZE_NAMES.items())
ANCHOR_RECALL_SPLIT = "Training"
ANCHOR_RECALL_LEVEL_PREFIX = "recall_block_"
ANCHOR_RECALL_ALL_LEVELS_COLUMN = "recall_all_levels"


def decode_annotation_counts(annotation: Dict[str, Any]) -> List[int]:
    """
    Gives the run lengths of the run length encoding of an annotation, decoded from the compressed string the synthetic
    generator writes (decode_compressed_counts, as the loader decodes it), or read as they are from an uncompressed
    list, so that the listing and the facts of the first annotation count the runs and pixels the masks are built from.

    Args:
        annotation (Dict[str, Any]): The COCO annotation, with its run length encoded segmentation.

    Returns:
        List[int]: The run lengths, alternating background and object pixels, starting with background.
    """
    counts = annotation["segmentation"]["counts"]
    if isinstance(counts, str):
        return decode_compressed_counts(compressed_counts=counts).tolist()
    return list(counts)


def format_annotation(annotation: Dict[str, Any]) -> List[str]:
    """
    Writes the first annotation of the manifest on the lines of the listing: its identifiers, its bbox, and its run
    length encoding, the start of its compressed counts string (or the first values of an uncompressed list) followed
    by the number of runs it stands for and their sum, the pixels of the image.

    Args:
        annotation (Dict[str, Any]): The COCO annotation, with its bbox and its run length encoded segmentation.

    Returns:
        List[str]: The lines of the annotation.
    """
    stored_counts = annotation["segmentation"]["counts"]
    counts = decode_annotation_counts(annotation=annotation)
    if isinstance(stored_counts, str):
        shown_counts = f'"{stored_counts[:RUN_LENGTH_CHARACTERS_SHOWN]}..." compressed'
    else:
        shown_values = stored_counts[:RUN_LENGTH_CHARACTERS_SHOWN // 2]
        shown_counts = "[" + ", ".join(str(value) for value in shown_values) + ", ...]"
    return ["annotations[0]",
            f"  id {annotation['id']}   image_id {annotation['image_id']}   category_id {annotation['category_id']}",
            f"  bbox {annotation['bbox']}",
            f"  segmentation size {annotation['segmentation']['size']}",
            f"    counts {shown_counts}   {len(counts)} runs, sum {sum(counts):,}"]


def write_manifest_listing(run_folder: Path, output_path: Path) -> Dict[str, Any]:
    """
    Writes the listing of the COCO data of the run: the first image, annotation and categories of its test manifest,
    then its label mapping file and the label dictionary saved in the run folder, and gives the values the narration
    quotes: the counts of the manifest, the first box, the classes of the mapping and of the background, the side of
    the images, the fewest and most objects of an image of the test split (drawn like the training split) and the runs
    of the first annotation.

    Args:
        run_folder (Path): The folder of the DET run.
        output_path (Path): The text file of the listing.

    Returns:
        Dict[str, Any]: The facts of the manifest, named data.*.
    """
    run_configuration = read_run_configuration(run_folder=run_folder)
    test_folder = find_test_folder(run_configuration=run_configuration)
    manifest = read_test_manifest(test_folder=test_folder)
    mapping_path = Path(run_configuration["Data"]["label_mapping"]["mapping_file_path"])
    label_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    label_dictionary = json.loads((run_folder / "label_dictionary.json").read_text(encoding="utf-8"))

    # Manifest of the test split, then the mapping file and the dictionary of the run
    categories = manifest["categories"]
    first_annotation = manifest["annotations"][0]
    first_annotation_category = [category for category in categories if
                                 category["id"] == first_annotation["category_id"]][0]
    mapping_items = list(label_mapping.items())
    lines = [f"{test_folder.name}/labels.json   {len(manifest['images']):,} images   "
             f"{len(manifest['annotations']):,} annotations",
             f"images[0]       {json.dumps(obj=manifest['images'][0])}"]
    lines += format_annotation(annotation=first_annotation)
    lines += [f"categories      {json.dumps(obj=categories[0])}  ...",
              f"                {json.dumps(obj=first_annotation_category)}  ...",
              f"                {json.dumps(obj=categories[-1])}",
              "",
              f"label_mapping.json      {{\"{mapping_items[0][0]}\": {mapping_items[0][1]}, \"{mapping_items[1][0]}\": "
              f"{mapping_items[1][1]}, ..., \"{mapping_items[-1][0]}\": {mapping_items[-1][1]}}}",
              f"label_dictionary.json   the mapping + \"background\": {label_dictionary['background']}   "
              f"{len(label_dictionary)} classes"]
    write_listing(lines=lines, output_path=output_path)

    # Objects of every image of the test split, the images without object counting none
    objects_per_image = collections.Counter(annotation["image_id"] for annotation in manifest["annotations"])
    image_object_counts = [objects_per_image[image["id"]] for image in manifest["images"]]
    first_counts = decode_annotation_counts(annotation=first_annotation)
    return {"data.test_images": len(manifest["images"]), "data.test_annotations": len(manifest["annotations"]),
            "data.test_folder_name": test_folder.name,
            "data.minimum_objects_per_image": min(image_object_counts),
            "data.maximum_objects_per_image": max(image_object_counts),
            "data.image_size": manifest["images"][0]["width"],
            "data.background_class": label_dictionary["background"],
            "data.first_class": min(label_mapping.values()), "data.last_class": max(label_mapping.values()),
            "data.category_count": len(categories), "data.class_count": len(label_dictionary),
            "data.first_bbox": first_annotation["bbox"], "data.first_category": first_annotation_category["name"],
            "data.first_box_corners": [first_annotation["bbox"][0], first_annotation["bbox"][1],
                                       first_annotation["bbox"][0] + first_annotation["bbox"][2],
                                       first_annotation["bbox"][1] + first_annotation["bbox"][3]],
            "data.run_length_values": len(first_counts), "data.run_length_sum": sum(first_counts),
            "data.run_length_compressed": isinstance(first_annotation["segmentation"]["counts"], str)}


def read_split_counts(run_folder: Path) -> Dict[str, Any]:
    """
    Reads the images and objects of every split from the data record of the run (data.json), the counts the launch
    panel prints.

    Args:
        run_folder (Path): The folder of the DET run.

    Returns:
        Dict[str, Any]: The counts, named data.<split>_images and data.<split>_objects.
    """
    record = json.loads((run_folder / "data.json").read_text(encoding="utf-8"))
    facts = {}
    for split in ["train", "validation", "test"]:
        facts[f"data.{split}_images"] = record["splits"][split]["number_images"]
        facts[f"data.{split}_objects"] = sum(class_record["annotations"][split] for class_record in record["classes"])
    return facts


def read_anchor_recall(run_folder: Path, image_size: int) -> Dict[str, Any]:
    """
    Reads the anchor recall of the training objects from the anchor_recall.csv of the evaluation of the run made again
    with the current code (CHECKPOINT_EVALUATION_FOLDER_NAME), whose launch measured the training and test splits at
    the foreground threshold of the run: the share of the objects of every relative size whose best anchor of a level
    reaches the threshold, and of all the levels together, as the rows of the grid of the scene, with the number of
    objects of every size, the block of every level column, and the sides in pixels of every size at the network input.

    Args:
        run_folder (Path): The folder of the DET run.
        image_size (int): The side of the network input in pixels, which gives the sides of every size.

    Returns:
        Dict[str, Any]: The facts, named anchor_recall.*: the grid values as percentages rounded like the reports
            (rows all the objects, then every size from the smallest; a column per level of the run, then all the
            levels), the row and column labels, every value by name between 0 and 1, and the pixels of every size.

    Raises:
        ValueError: If the file holds no row of the training split, as written before the splits were recorded.
    """
    with open(file=run_folder / CHECKPOINT_EVALUATION_FOLDER_NAME / "anchor_recall.csv", mode="r",
              encoding="utf-8") as file_handler:
        reader = csv.DictReader(f=file_handler)
        rows = {row["objects"]: row for row in reader if row.get("split") == ANCHOR_RECALL_SPLIT}
        level_columns = [column for column in reader.fieldnames if column.startswith(ANCHOR_RECALL_LEVEL_PREFIX)]
    if not rows:
        raise ValueError(f"No {ANCHOR_RECALL_SPLIT} row in the anchor_recall.csv of {run_folder}: evaluate the run again "
                         f"with scripts/evaluation/evaluate_checkpoint.py.")
    columns = level_columns + [ANCHOR_RECALL_ALL_LEVELS_COLUMN]

    # Grid cells as the percentages of the reports, the values quoted by name staying between 0 and 1
    facts = {"anchor_recall.percentages": [[round_percentage(value=float(rows[row_key][column])) for column in
                                            columns] for row_key, _ in ANCHOR_RECALL_ROWS],
             "anchor_recall.column_labels": [f"block {column[len(ANCHOR_RECALL_LEVEL_PREFIX):]}"
                                             for column in level_columns] + ["all levels"],
             "anchor_recall.row_labels": [f"{label} · {int(rows[row_key]['count']):,}" for row_key, label in
                                          ANCHOR_RECALL_ROWS]}
    for row_key, _ in ANCHOR_RECALL_ROWS:
        facts[f"anchor_recall.{row_key}.count"] = int(rows[row_key]["count"])
        for column in columns:
            facts[f"anchor_recall.{row_key}.{column}"] = round_fact(value=float(rows[row_key][column]))

    # Name of every size and its sides in pixels at the network input of the run, e.g. "1/16 to 1/8" and "32 to 64",
    # with the side below which an object is in the size, none for the last size, open above
    for size_key, size_name in OBJECT_SIZE_NAMES.items():
        facts[f"anchor_recall.{size_key}.name"] = size_name
        facts[f"anchor_recall.{size_key}.pixels"] = describe_object_size_pixels(object_size=size_key,
                                                                                 image_side=image_size)
        upper_share = OBJECT_SIZE_SIDE_SHARES[size_key][1]
        if not math.isinf(upper_share):
            upper_side = upper_share * image_size
            facts[f"anchor_recall.{size_key}.upper_side"] = (int(upper_side) if float(upper_side).is_integer() else
                                                             round_fact(value=upper_side))
    return facts


def make_dataset_figures(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Writes the manifest listing of the DET run and gives the facts of its data and of its anchor recall.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the DET run of tutorial_sources.yaml.

    Returns:
        Dict[str, Any]: The facts, named data.* and anchor_recall.*.
    """
    facts = write_manifest_listing(run_folder=run_folder, output_path=output_folder / "listings" / "coco_manifest.txt")
    facts.update(read_split_counts(run_folder=run_folder))
    facts.update(read_anchor_recall(run_folder=run_folder, image_size=facts["data.image_size"]))
    return facts
