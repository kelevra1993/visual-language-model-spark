"""
Readers of the run snapshots for the classification figures of the tutorial videos, so that a build from
tutorials/run_snapshots gives the facts of a build from the real runs: the sample count of a training manifest reduced
by snapshot_reductions.reduce_sample_manifest (the labels listing), and the predictions file of the classification
run, kept gzipped in the snapshot with the size of the original recorded in the manifest of the series (the
prediction confidence figure and the predictions_file facts). On the real runs they read the files as before.
"""

import gzip
import json

from pathlib import Path
from typing import Any, Dict, Optional, TextIO

from tutorials.figures.classification.snapshot_reductions import SNAPSHOT_SAMPLE_COUNT_KEY
from tutorials.figures.figure_utilities import find_snapshot_root

# Suffix of the gzipped copy of a file in the snapshots ('compress: gzip' of snapshot_files.yaml)
GZIP_SUFFIX = ".gz"

# Folder of the snapshot folder holding the manifests of the series, and the series whose manifest records the
# predictions file of the classification run
MANIFESTS_FOLDER_NAME = "manifests"
CLASSIFICATION_SERIES = "classification"


def count_manifest_samples(manifest: Dict[str, Any]) -> int:
    """
    Gives the number of samples of a split for the labels.json listing of the classification tutorials: the samples of
    the manifest, or, for a manifest reduced for the run snapshots, the count of the full manifest it records.

    Args:
        manifest (Dict[str, Any]): The parsed labels.json.

    Returns:
        int: The number of samples of the split.
    """
    if SNAPSHOT_SAMPLE_COUNT_KEY in manifest:
        return int(manifest[SNAPSHOT_SAMPLE_COUNT_KEY])
    return len(manifest["samples"])


def find_predictions_file(predictions_path: Path) -> Path:
    """
    Finds the predictions file of an evaluation of the classification run: the CSV written by the evaluation, or, in
    a run read from the snapshots, its gzipped copy next to where the CSV would be.

    Args:
        predictions_path (Path): The predictions.csv path of the evaluation.

    Returns:
        Path: The CSV, or its gzipped copy when only the copy exists.

    Raises:
        FileNotFoundError: If neither exists (the evaluation ran without save_predictions).
    """
    compressed_path = predictions_path.with_name(predictions_path.name + GZIP_SUFFIX)
    if predictions_path.exists():
        return predictions_path
    if compressed_path.exists():
        return compressed_path
    raise FileNotFoundError(f"No predictions file {predictions_path} nor {compressed_path.name}: the evaluation needs "
                            f"save_predictions.")


def open_predictions_file(predictions_path: Path, newline: Optional[str]) -> TextIO:
    """
    Opens the predictions file of an evaluation as text, the CSV or its gzipped copy (find_predictions_file), with the
    newline handling of the caller (newline="" for the csv module).

    Args:
        predictions_path (Path): The predictions.csv path of the evaluation.
        newline (Optional[str]): The newline argument of open.

    Returns:
        TextIO: The open text file, to be closed by the caller.
    """
    file_path = find_predictions_file(predictions_path=predictions_path)
    if file_path.name.endswith(GZIP_SUFFIX):
        return gzip.open(filename=file_path, mode="rt", encoding="utf-8", newline=newline)
    return open(file=file_path, encoding="utf-8", newline=newline)


def measure_predictions_bytes(predictions_path: Path) -> int:
    """
    Gives the size in bytes of the predictions file of an evaluation, for the predictions_file facts: the size of the
    CSV, or, for its gzipped copy in the snapshots, the size of the original recorded in the manifest of the
    classification series, so that both builds give the same fact.

    Args:
        predictions_path (Path): The predictions.csv path of the evaluation.

    Returns:
        int: The size of the CSV in bytes.

    Raises:
        KeyError: If the manifest does not record the gzipped copy.
    """
    file_path = find_predictions_file(predictions_path=predictions_path)
    if file_path == predictions_path:
        return predictions_path.stat().st_size

    # Size of the source of the gzipped copy, recorded by tutorials/make_run_snapshots.py
    snapshot_root = find_snapshot_root().resolve()
    snapshot_path = file_path.resolve().relative_to(snapshot_root).as_posix()
    manifest_path = snapshot_root / MANIFESTS_FOLDER_NAME / f"{CLASSIFICATION_SERIES}.json"
    for record in json.loads(s=manifest_path.read_text(encoding="utf-8"))["files"]:
        if record["snapshot_path"] == snapshot_path:
            return int(record["source_bytes"])
    raise KeyError(f"The manifest {manifest_path} does not record {snapshot_path}.")
