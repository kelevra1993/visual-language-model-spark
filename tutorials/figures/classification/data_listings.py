"""
Data listings of the classification tutorials (the chapter from labels to logits), written at every build by
tutorials/figures/classification/figures.py from the real files of the classification run of
tutorials/tutorial_sources.yaml: the start of the labels.json manifest of its training split (its categories and
first samples), its label mapping file, and the classes of its data.json, one compact line per class with its
category and its image count in every split. The listings are text files drawn by the listing element of the diagram
scenes; the facts give the counts the narration quotes, the category and class of the first sample of the manifest
(the one the scene follows through the mapping), and the number of shape classes, the image size and the most shapes
of a multi-label image of the synthetic dataset generator of the repository as it draws them now. A build from the
run snapshots reads a training manifest reduced to a few samples, whose recorded sample count the listing quotes.
"""

import json

from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.classification.snapshot_helpers import count_manifest_samples
from tutorials.figures.figure_utilities import read_run_configuration
from tutorials.figures.foundations.code_values import read_main_block_value
from tutorial_builder.figure_context import find_repository_root
from vision_training.data.synthetic.create_synthetic_dataset import SHAPE_CATEGORIES

# Synthetic dataset script, whose main block sets the image size and the most shapes of every task
SYNTHETIC_DATASET_SCRIPT = Path("vision_training") / "data" / "synthetic" / "create_synthetic_dataset.py"

# Samples of the training manifest shown in the labels.json listing
LISTED_SAMPLE_COUNT = 3


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

    # Start of the training manifest
    labels_path = Path(run_configuration["ExperimentConfiguration"]["train_data_folder"]) / "labels.json"
    write_listing(lines=list_labels_manifest(labels_path=labels_path), output_path=listings_folder / "labels_json.txt")

    # Label mapping file of the run, one entry per line as the file holds it
    mapping_path = Path(run_configuration["Data"]["label_mapping"]["mapping_file_path"])
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping_lines = [f'  "{category}": {class_index},' for category, class_index in mapping.items()]
    mapping_lines[-1] = mapping_lines[-1].rstrip(",")
    write_listing(lines=["{"] + mapping_lines + ["}"], output_path=listings_folder / "label_mapping_json.txt")

    # Classes of the data record of the run, one compact line per class
    classes = json.loads((run_folder / "data.json").read_text(encoding="utf-8"))["classes"]
    class_lines = [f"{record['class_index']:<5}  {record['class_name']:<10}  {record['category_ids'][0]:>8}  "
                   f"{record['images']['train']:>7,}  {record['images']['validation']:>6,}  "
                   f"{record['images']['test']:>6,}" for record in classes]
    header = f"{'index':<5}  {'class':<10}  {'category':>8}  {'train':>7}  {'valid.':>6}  {'test':>6}"
    write_listing(lines=[header] + class_lines, output_path=listings_folder / "data_json_classes.txt")
    facts = {"data.class_count": len(classes), "data.mapping_entries": len(mapping),
             "data.smallest_training_class": min(record["images"]["train"] for record in classes),
             "data.largest_training_class": max(record["images"]["train"] for record in classes),
             "data.generator_class_count": len(SHAPE_CATEGORIES)}
    for split in ["train", "validation", "test"]:
        facts[f"data.{split}_images"] = sum(record["images"][split] for record in classes)

    # Category of the first training sample, highlighted in the manifest, and the class index the mapping gives it
    manifest = json.loads(labels_path.read_text(encoding="utf-8"))
    first_category_id = manifest["samples"][0]["category_id"]
    category_names = {category["id"]: category["name"] for category in manifest["categories"]}
    if str(first_category_id) not in mapping:
        raise ValueError(f"The category {first_category_id} of the first sample of {labels_path} is not in the label "
                         f"mapping {mapping_path}.")
    facts.update({"data.first_sample.category_id": first_category_id,
                  "data.first_sample.category_name": category_names[first_category_id],
                  "data.first_sample.class_index": mapping[str(first_category_id)]})

    # Image size of the classification images and most shapes of a multi-label image drawn by the generator now
    script_path = find_repository_root() / SYNTHETIC_DATASET_SCRIPT
    image_sizes = read_main_block_value(script_path=script_path, variable_name="image_sizes")
    maximum_objects = read_main_block_value(script_path=script_path, variable_name="maximum_objects")
    facts.update({"generator.image_size": image_sizes["classification"],
                  "generator.multi_label_maximum_objects": maximum_objects["multi_label_classification"]})
    return facts
