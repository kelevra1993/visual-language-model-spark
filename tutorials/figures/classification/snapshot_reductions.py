"""
Custom reduction of the run snapshots for the sample manifests of the synthetic shapes dataset shared by the
classification and self-supervised series of the tutorial videos: tutorials/make_run_snapshots.py calls it, as named in
the snapshot_files.yaml of both series, to write a labels.json of a split (too large for git with its 500,000 training
samples) holding only the samples the figure scripts read, with the sample count of the full manifest, which the
labels listing of the classification series quotes (snapshot_helpers.count_manifest_samples). It runs in the
repository environment of the writer, without the tutorial builder package the figure scripts import.
"""

import json

from pathlib import Path
from typing import Any, Dict, List

from vision_training.evaluation.self_supervised.representation_metrics import select_reference_samples

# Top-level key of a manifest reduced for the snapshots holding the number of samples of the full manifest, so that a
# listing quoting the size of the split stays right while the snapshot keeps only a few samples
SNAPSHOT_SAMPLE_COUNT_KEY = "snapshot_sample_count"


def select_category_reference_names(samples: List[Dict[str, Any]], number_images_per_category: int) -> List[str]:
    """
    Gives the file names of the samples the k-NN evaluation of a self-supervised run takes as reference images, picked
    by select_reference_samples of the repository with every category standing for one class (the synthetic shapes map
    every category to its own class), so that a reduced training manifest holds every reference image and the
    selection the figure scripts make from it is the one made from the full manifest.

    Args:
        samples (List[Dict[str, Any]]): The samples of the manifest, each with its 'id', 'file_name' and 'category_id'.
        number_images_per_category (int): The reference images of every category
            ('evaluation.number_reference_images_per_class' of the run).

    Returns:
        List[str]: The file names of the reference images.
    """
    records = [{"identifier": sample["id"], "file_name": sample["file_name"], "class_indices": [sample["category_id"]]}
               for sample in samples if "category_id" in sample]
    return [record["file_name"] for record in select_reference_samples(
        samples=records, number_images_per_class=number_images_per_category)]


def reduce_sample_manifest(source_path: Path, destination_path: Path, options: Dict[str, Any]) -> None:
    """
    Writes a sample manifest of a classification split (the 'info', 'categories' and 'samples' of a labels.json) for
    the run snapshots, called by tutorials/make_run_snapshots.py as a custom reduction: the other keys whole, the
    samples the figure scripts read in the order of the source, and the number of samples of the source under
    SNAPSHOT_SAMPLE_COUNT_KEY.

    Args:
        source_path (Path): The labels.json of the split.
        destination_path (Path): The reduced manifest.
        options (Dict[str, Any]): 'first_samples' (the first samples kept, 0 by default), 'samples' (file names kept),
            'reference_samples_per_category' (the k-NN reference images of every category kept, 0 by default).

    Raises:
        ValueError: If a listed file name is not in the manifest, or nothing would be kept.
    """
    manifest = json.loads(s=source_path.read_text(encoding="utf-8"))
    samples = manifest["samples"]

    # File names kept: the listed ones, the first ones of the file and the reference images of every category
    kept_names = set(options.get("samples", []))
    missing_names = kept_names - {sample["file_name"] for sample in samples}
    if missing_names:
        raise ValueError(f"{source_path} has no samples {sorted(missing_names)}.")
    kept_names.update(sample["file_name"] for sample in samples[:options.get("first_samples", 0)])
    if options.get("reference_samples_per_category", 0) > 0:
        kept_names.update(select_category_reference_names(
            samples=samples, number_images_per_category=options["reference_samples_per_category"]))
    if not kept_names:
        raise ValueError(f"No sample of {source_path} is kept by the options {options}.")

    # Every other key whole, the kept samples in the order of the source, and the count of the full manifest
    reduced_manifest = {key: value for key, value in manifest.items() if key != "samples"}
    reduced_manifest["samples"] = [sample for sample in samples if sample["file_name"] in kept_names]
    reduced_manifest[SNAPSHOT_SAMPLE_COUNT_KEY] = len(samples)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(data=json.dumps(obj=reduced_manifest, separators=(",", ":")), encoding="utf-8")
