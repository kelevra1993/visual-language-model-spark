"""
Data listings of the classification tutorials (the chapter from labels to logits), written at every build by
tutorials/figures/classification/figures.py from the real files of the classification run of
tutorials/tutorial_sources.yaml: the start of the labels.json manifest of its training split (its categories and
first samples), its label mapping file, its first and last entries around the entry of the first sample, and the
classes of its data.json, the first and last classes on one compact line each with their category and image count in
every split. The listings are text files drawn by the listing element of the diagram scenes; the facts give the counts
the narration quotes, the category and class of the first sample of the manifest (the one the scene follows through
the mapping, with the points of the arrow joining its two lines), and the image size and the most shapes of a
multi-label image of the synthetic dataset generator of the repository. A build from the run snapshots reads a
training manifest reduced to a few samples, whose recorded sample count the listing quotes.
"""

import json

from pathlib import Path
from typing import Any, Dict, List, Tuple

from tutorials.figures.classification.evaluation_tables import format_class_name
from tutorials.figures.classification.snapshot_helpers import count_manifest_samples
from tutorials.figures.figure_utilities import read_run_configuration
from tutorials.figures.foundations.code_values import read_main_block_value
from tutorial_builder.figure_context import find_repository_root

# Synthetic dataset script, whose main block sets the image size and the most shapes of every task
SYNTHETIC_DATASET_SCRIPT = Path("vision_training") / "data" / "synthetic" / "create_synthetic_dataset.py"

# Samples of the training manifest shown in the labels.json listing
LISTED_SAMPLE_COUNT = 3

# Entries of the label mapping listing shown before the entry of the first sample, and classes of the data.json
# listing shown at its start and at its end, so that both listings fit their boxes whatever the number of classes
MAPPING_FIRST_ENTRIES = 3
FIRST_LISTED_CLASSES = 5
LAST_LISTED_CLASSES = 2

# Text of the lines standing for the entries left out of a listing
OMITTED_LINE = "  ..."

# Layout of the labels scene of 01_labels_to_logits.yaml, in pixels of its content area: the top of the boxes of the
# labels.json and label_mapping.json listings, the height of a line of their diagram_monospace_tiny font, the line of
# the first sample in the labels.json listing (counted from 0), and where the arrow joining it to its mapping entry
# starts (past the end of the sample line) and ends (the left edge of the mapping listing)
LISTING_TOP = 56
LISTING_LINE_HEIGHT = 22
FIRST_SAMPLE_LINE_INDEX = 8
ARROW_START_X = 700
ARROW_END_X = 800


def write_listing(lines: List[str], output_path: Path) -> None:
    """
    Writes the lines of a listing as a text file, the source of a listing element of a diagram scene.

    Args:
        lines (List[str]): The lines.
        output_path (Path): The text file, its folder created if missing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(data="\n".join(lines) + "\n", encoding="utf-8")


def list_labels_manifest(labels_path: Path) -> List[str]:
    """
    Gives the lines of the labels.json listing: the categories, then the first samples, each on one line, the other
    samples left out as an ellipsis, so that the layout of the manifest the dataset reads is seen whole.

    Args:
        labels_path (Path): The labels.json file of the training split.

    Returns:
        List[str]: The lines.
    """
    manifest = json.loads(labels_path.read_text(encoding="utf-8"))
    lines = ["{", '  "categories": [']
    for category in manifest["categories"][:3]:
        lines.append(f"    {json.dumps(obj=category)},")
    lines += [f"    ...  {len(manifest['categories'])} categories", "  ],", '  "samples": [']
    for sample in manifest["samples"][:LISTED_SAMPLE_COUNT]:
        lines.append(f"    {json.dumps(obj=sample)},")
    # Samples of the split, recorded by a manifest reduced for the run snapshots
    lines += [f"    ...  {count_manifest_samples(manifest=manifest):,} samples", "  ]", "}"]
    return lines


def list_label_mapping(mapping: Dict[str, int], followed_category: str) -> Tuple[List[str], int]:
    """
    Gives the lines of the label mapping listing: its first entries, the entry of the category the scene follows and
    the last entry, one per line as the file holds them, the entries in between left out as an ellipsis.

    Args:
        mapping (Dict[str, int]): The label mapping, category identifiers as strings to class indices.
        followed_category (str): The category of the first training sample, whose entry the scene highlights.

    Returns:
        Tuple[List[str], int]: The lines, and the index of the line of the followed category among them.
    """
    categories = list(mapping)
    followed_index = categories.index(followed_category)
    shown_indices = sorted(set(list(range(min(MAPPING_FIRST_ENTRIES, len(categories)))) +
                               [followed_index, len(categories) - 1]))

    # Shown entries with a comma but the last one, an ellipsis wherever entries are left out
    lines = ["{"]
    followed_line_index = 0
    for position, entry_index in enumerate(shown_indices):
        if position > 0 and entry_index > shown_indices[position - 1] + 1:
            lines.append(OMITTED_LINE)
        if entry_index == followed_index:
            followed_line_index = len(lines)
        category = categories[entry_index]
        separator = "" if entry_index == len(categories) - 1 else ","
        lines.append(f'  "{category}": {mapping[category]}{separator}')
    return lines + ["}"], followed_line_index


def list_data_classes(classes: List[Dict[str, Any]]) -> List[str]:
    """
    Gives the lines of the data.json listing: a header, then the first and last classes of the run, one compact line
    each with its index, name, first category and images per split, the classes in between left out as an ellipsis,
    the class column as wide as the longest class name.

    Args:
        classes (List[Dict[str, Any]]): The 'classes' records of the data.json of the run.

    Returns:
        List[str]: The lines.
    """
    class_width = max(len("class"), *(len(record["class_name"]) for record in classes))
    header = f"{'index':<5}  {'class':<{class_width}}  {'category':>8}  {'train':>7}  {'valid.':>6}  {'test':>6}"
    class_lines = [f"{record['class_index']:<5}  {record['class_name']:<{class_width}}  "
                   f"{record['category_ids'][0]:>8}  {record['images']['train']:>7,}  "
                   f"{record['images']['validation']:>6,}  {record['images']['test']:>6,}" for record in classes]
    if len(class_lines) <= FIRST_LISTED_CLASSES + LAST_LISTED_CLASSES + 1:
        return [header] + class_lines
    return [header] + class_lines[:FIRST_LISTED_CLASSES] + [OMITTED_LINE] + class_lines[-LAST_LISTED_CLASSES:]


def make_data_listings(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Writes the labels.json, label_mapping.json and data.json listings of the classification run into the listings
    folder of the series, and gives the counts of its data.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the classification run of tutorial_sources.yaml.

    Returns:
        Dict[str, Any]: The facts, named data.* and generator.*.

    Raises:
        ValueError: If the category of the first training sample is not in the label mapping of the run.
    """
    run_configuration = read_run_configuration(run_folder=run_folder)
    listings_folder = output_folder / "listings"

    # Start of the training manifest, and the category of its first sample, which the scene follows
    labels_path = Path(run_configuration["ExperimentConfiguration"]["train_data_folder"]) / "labels.json"
    write_listing(lines=list_labels_manifest(labels_path=labels_path), output_path=listings_folder / "labels_json.txt")
    manifest = json.loads(labels_path.read_text(encoding="utf-8"))
    first_category_id = manifest["samples"][0]["category_id"]
    category_names = {category["id"]: category["name"] for category in manifest["categories"]}

    # Label mapping file of the run around the entry of the first sample, one entry per line as the file holds it
    mapping_path = Path(run_configuration["Data"]["label_mapping"]["mapping_file_path"])
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if str(first_category_id) not in mapping:
        raise ValueError(f"The category {first_category_id} of the first sample of {labels_path} is not in the label "
                         f"mapping {mapping_path}.")
    mapping_lines, mapping_line_index = list_label_mapping(mapping=mapping, followed_category=str(first_category_id))
    write_listing(lines=mapping_lines, output_path=listings_folder / "label_mapping_json.txt")

    # Arrow from the line of the first sample to its mapping entry, at the middle height of both lines
    arrow_points = [[ARROW_START_X, LISTING_TOP + (FIRST_SAMPLE_LINE_INDEX + 0.5) * LISTING_LINE_HEIGHT],
                    [ARROW_END_X, LISTING_TOP + (mapping_line_index + 0.5) * LISTING_LINE_HEIGHT]]

    # Classes of the data record of the run, the first and last ones on one compact line each
    classes = json.loads((run_folder / "data.json").read_text(encoding="utf-8"))["classes"]
    write_listing(lines=list_data_classes(classes=classes), output_path=listings_folder / "data_json_classes.txt")
    facts = {"data.class_count": len(classes), "data.mapping_entries": len(mapping),
             "data.smallest_training_class": min(record["images"]["train"] for record in classes),
             "data.largest_training_class": max(record["images"]["train"] for record in classes),
             "data.first_sample.category_id": first_category_id,
             "data.first_sample.category_name": category_names[first_category_id],
             "data.first_sample.category_display_name": format_class_name(
                 class_name=category_names[first_category_id]),
             "data.first_sample.class_index": mapping[str(first_category_id)],
             "data.mapping_arrow_points": arrow_points}
    for split in ["train", "validation", "test"]:
        facts[f"data.{split}_images"] = sum(record["images"][split] for record in classes)

    # Image size of the classification images and most shapes of a multi-label image drawn by the generator
    script_path = find_repository_root() / SYNTHETIC_DATASET_SCRIPT
    image_sizes = read_main_block_value(script_path=script_path, variable_name="image_sizes")
    maximum_objects = read_main_block_value(script_path=script_path, variable_name="maximum_objects")
    facts.update({"generator.image_size": image_sizes["classification"],
                  "generator.multi_label_maximum_objects": maximum_objects["multi_label_classification"]})
    return facts
