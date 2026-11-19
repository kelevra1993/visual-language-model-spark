"""
Data figures of the object detection tutorials (the chapter on the data, the levels and the anchors), made at every
build by tutorials/figures/object_detection/figures.py from the DET run of tutorials/tutorial_sources.yaml: the
listing of the COCO manifest of its test split (the first image, the first annotation with its run length encoding
shortened, the first and last categories) next to the label mapping file and the label dictionary saved in the run
folder, which adds the background, with the classes, the image side and the most objects of an image that the
narration quotes; the counts of the splits from the data record of the run (data.json); and the anchor recall of the
training objects the trainer wrote at launch (anchor_recall.csv), by object size and detection level, the values of
the grid of the anchor recall scene, written in its cells as the percentages of the reports of the repository and
kept between 0 and 1 in the facts the narration quotes.
"""

import collections
import csv
import json
import math

from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import read_run_configuration, round_fact
from tutorials.figures.object_detection.detection_sources import find_test_folder, read_test_manifest, write_listing
from vision_training.data.synthetic.create_synthetic_dataset import TASK_CATEGORIES
from vision_training.evaluation.detection.object_sizes import OBJECT_SIZE_RANGES
from vision_training.evaluation.report_formatting import round_percentage

# Values of the run length encoding of the first annotation kept in the listing, the rest being counted
RUN_LENGTH_VALUES_SHOWN = 6

# Task of the synthetic data set script that drew the data set of the run, whose number of categories a data set drawn
# today has
GENERATED_TASK = "instance_segmentation"

# Rows of anchor_recall.csv in the order of the grid of the scene, with the names of the object sizes of the evaluation
ANCHOR_RECALL_ROWS = [("small", "Small < 32²"), ("medium", "Medium 32² to 96²"), ("large", "Large >= 96²"),
                      ("all", "All objects")]
ANCHOR_RECALL_COLUMNS = ["recall_block_5", "recall_block_6", "recall_block_7", "recall_all_levels"]


def format_annotation(annotation: Dict[str, Any]) -> List[str]:
    """
    Writes the first annotation of the manifest on the lines of the listing, its run length encoding shortened to its
    first values followed by the number of values and their sum, the pixels of the image.

    Args:
        annotation (Dict[str, Any]): The COCO annotation, with its bbox and its uncompressed segmentation.

    Returns:
        List[str]: The lines of the annotation.
    """
    counts = annotation["segmentation"]["counts"]
    shown_counts = ", ".join(str(value) for value in counts[:RUN_LENGTH_VALUES_SHOWN])
    return ["annotations[0]",
            f"  id {annotation['id']}   image_id {annotation['image_id']}   category_id {annotation['category_id']}",
            f"  bbox {annotation['bbox']}",
            f"  segmentation size {annotation['segmentation']['size']}",
            f"    counts [{shown_counts}, ...]   {len(counts)} values, sum {sum(counts):,}"]


def write_manifest_listing(run_folder: Path, output_path: Path) -> Dict[str, Any]:
    """
    Writes the listing of the COCO data of the run: the first image, annotation and categories of its test manifest,
    then its label mapping file and the label dictionary saved in the run folder, and gives the values the narration
    quotes: the counts of the manifest, the first box, the classes of the mapping and of the background, the side of
    the images and the most objects of an image of the test split (drawn like the training split), and the number of
    categories of a data set drawn today by the synthetic data set script.

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
    lines = [f"test/labels.json   {len(manifest['images']):,} images   {len(manifest['annotations']):,} annotations",
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
    return {"data.test_images": len(manifest["images"]), "data.test_annotations": len(manifest["annotations"]),
            "data.maximum_objects_per_image": max(objects_per_image.values()),
            "data.image_size": manifest["images"][0]["width"],
            "data.background_class": label_dictionary["background"],
            "data.first_class": min(label_mapping.values()), "data.last_class": max(label_mapping.values()),
            "data.generated_category_count": len(TASK_CATEGORIES[GENERATED_TASK]),
            "data.category_count": len(categories), "data.class_count": len(label_dictionary),
            "data.first_bbox": first_annotation["bbox"], "data.first_category": first_annotation_category["name"],
            "data.first_box_corners": [first_annotation["bbox"][0], first_annotation["bbox"][1],
                                       first_annotation["bbox"][0] + first_annotation["bbox"][2],
                                       first_annotation["bbox"][1] + first_annotation["bbox"][3]],
            "data.run_length_values": len(first_annotation["segmentation"]["counts"]),
            "data.run_length_sum": sum(first_annotation["segmentation"]["counts"])}


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


def read_anchor_recall(run_folder: Path) -> Dict[str, Any]:
    """
    Reads the anchor recall of the training objects written by the trainer at launch (anchor_recall.csv): the share
    of the objects of every size whose best anchor of a level reaches the foreground threshold, and of all the levels
    together, as the rows of the grid of the scene, with the number of objects of every size and the box side from
    which an object is large (the square root of the smallest large area of OBJECT_SIZE_RANGES).

    Args:
        run_folder (Path): The folder of the DET run.

    Returns:
        Dict[str, Any]: The facts, named anchor_recall.*: the grid values as percentages rounded like the reports
            (rows small, medium, large, all; columns blocks 5, 6, 7 and all the levels), the row labels, every value
            by name between 0 and 1 and the large object side.
    """
    with open(file=run_folder / "anchor_recall.csv", mode="r", encoding="utf-8") as file_handler:
        rows = {row["objects"]: row for row in csv.DictReader(f=file_handler)}

    # Grid cells as the percentages of the reports, the values quoted by name staying between 0 and 1
    facts = {"anchor_recall.percentages": [[round_percentage(value=float(rows[row_key][column])) for column in
                                            ANCHOR_RECALL_COLUMNS] for row_key, _ in ANCHOR_RECALL_ROWS],
             "anchor_recall.row_labels": [f"{label} · {int(rows[row_key]['count']):,}" for row_key, label in
                                          ANCHOR_RECALL_ROWS],
             "anchor_recall.large_minimum_side": round(math.sqrt(OBJECT_SIZE_RANGES["large"][0]))}
    for row_key, _ in ANCHOR_RECALL_ROWS:
        facts[f"anchor_recall.{row_key}.count"] = int(rows[row_key]["count"])
        for column in ANCHOR_RECALL_COLUMNS:
            facts[f"anchor_recall.{row_key}.{column}"] = round_fact(value=float(rows[row_key][column]))
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
    facts.update(read_anchor_recall(run_folder=run_folder))
    return facts
