"""
Custom reductions of the run snapshots for the classification files too large for git, called by
tutorials/make_run_snapshots.py as named in the snapshot_files.yaml of the series: the sample manifests of the synthetic
shapes dataset shared by the classification and self-supervised series of the tutorial videos, written as a labels.json
of a split (500,000 training samples) holding only the samples the figure scripts read, with the sample count of the
full manifest, which the labels listing of the classification series quotes (snapshot_helpers.count_manifest_samples);
and the predictions.csv of the classification run (one probability per class and test image, 10 megabytes gzipped at
35 classes), written with the probability of the predicted class of every image and the probabilities of the classes
the prediction confidence figure draws. They run in the repository environment of the writer, without the tutorial
builder package the figure scripts import.
"""

import csv
import json

from pathlib import Path
from typing import Any, Dict

# Top-level key of a manifest reduced for the snapshots holding the number of samples of the full manifest, so that a
# listing quoting the size of the split stays right while the snapshot keeps only a few samples
SNAPSHOT_SAMPLE_COUNT_KEY = "snapshot_sample_count"

# Columns of a predictions file kept whole by its reduction, the prefix of its probability columns, and the column the
# reduction adds with the probability of the predicted class of every row, copied as the file writes it
PREDICTION_KEPT_COLUMNS = ["sample", "true_class", "predicted_class", "correct"]
PROBABILITY_COLUMN_PREFIX = "probability_"
PREDICTED_PROBABILITY_COLUMN = "predicted_probability"


def reduce_sample_manifest(source_path: Path, destination_path: Path, options: Dict[str, Any]) -> None:
    """
    Writes a sample manifest of a classification split (the 'info', 'categories' and 'samples' of a labels.json) for
    the run snapshots, called by tutorials/make_run_snapshots.py as a custom reduction: the other keys whole, the
    samples the figure scripts read in the order of the source, and the number of samples of the source under
    SNAPSHOT_SAMPLE_COUNT_KEY.

    Args:
        source_path (Path): The labels.json of the split.
        destination_path (Path): The reduced manifest.
        options (Dict[str, Any]): 'first_samples' (the first samples kept, 0 by default), 'samples' (file names kept).

    Raises:
        ValueError: If a listed file name is not in the manifest, or nothing would be kept.
    """
    manifest = json.loads(s=source_path.read_text(encoding="utf-8"))
    samples = manifest["samples"]

    # File names kept: the listed ones and the first ones of the file
    kept_names = set(options.get("samples", []))
    missing_names = kept_names - {sample["file_name"] for sample in samples}
    if missing_names:
        raise ValueError(f"{source_path} has no samples {sorted(missing_names)}.")
    kept_names.update(sample["file_name"] for sample in samples[:options.get("first_samples", 0)])
    if not kept_names:
        raise ValueError(f"No sample of {source_path} is kept by the options {options}.")

    # Every other key whole, the kept samples in the order of the source, and the count of the full manifest
    reduced_manifest = {key: value for key, value in manifest.items() if key != "samples"}
    reduced_manifest["samples"] = [sample for sample in samples if sample["file_name"] in kept_names]
    reduced_manifest[SNAPSHOT_SAMPLE_COUNT_KEY] = len(samples)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(data=json.dumps(obj=reduced_manifest, separators=(",", ":")), encoding="utf-8")


def reduce_predictions_file(source_path: Path, destination_path: Path, options: Dict[str, Any]) -> None:
    """
    Writes the predictions.csv of an evaluation of the classification run for the run snapshots, called by
    tutorials/make_run_snapshots.py as a custom reduction: every row with its sample, true and predicted classes and
    whether it is right, the probability of its predicted class under PREDICTED_PROBABILITY_COLUMN, and the probability
    columns of the classes listed in the options, every value copied as the evaluation wrote it, so that the prediction
    confidence figure gives the same facts from the reduced file as from the original.

    Args:
        source_path (Path): The predictions.csv of the evaluation.
        destination_path (Path): The reduced file.
        options (Dict[str, Any]): 'kept_classes', the classes whose probability columns are kept (the pair of classes
            the prediction confidence figure draws), and 'samples', the samples whose rows are kept (the sample images
            of an evaluation), every row when absent.

    Raises:
        KeyError: If the file lacks a kept column or the probability column of a listed or predicted class.
        ValueError: If a listed sample is not in the file.
    """
    with open(file=source_path, newline="", encoding="utf-8") as source_file:
        reader = csv.DictReader(f=source_file)
        probability_columns = [PROBABILITY_COLUMN_PREFIX + class_name for class_name in options["kept_classes"]]
        missing_columns = [column for column in PREDICTION_KEPT_COLUMNS + probability_columns
                           if column not in reader.fieldnames]
        if missing_columns:
            raise KeyError(f"{source_path} has no columns {missing_columns}, its columns being {reader.fieldnames}.")
        kept_samples = set(options["samples"]) if "samples" in options else None
        rows = [row for row in reader if kept_samples is None or row["sample"] in kept_samples]
    if kept_samples is not None and len(rows) != len(kept_samples):
        raise ValueError(f"{source_path} lacks the samples {sorted(kept_samples - {row['sample'] for row in rows})}.")

    # Kept columns of every row, then the probability of its predicted class and those of the listed classes
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file=destination_path, mode="w", newline="", encoding="utf-8") as destination_file:
        writer = csv.writer(destination_file, lineterminator="\n")
        writer.writerow(PREDICTION_KEPT_COLUMNS + [PREDICTED_PROBABILITY_COLUMN] + probability_columns)
        for row in rows:
            writer.writerow([row[column] for column in PREDICTION_KEPT_COLUMNS] +
                            [row[PROBABILITY_COLUMN_PREFIX + row["predicted_class"]]] +
                            [row[column] for column in probability_columns])
