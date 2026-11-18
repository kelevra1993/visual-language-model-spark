"""
Custom reduction of the run snapshots of the object detection series (tutorials/run_snapshots), named in
tutorials/figures/object_detection/snapshot_files.yaml and called by tutorials/make_run_snapshots.py: the data record
of the DET run (data.json, written by the trainer at launch) without the file name of every image of every split,
which makes most of its 4.8 megabytes, so that dataset_figures.py (read_split_counts) reads the same image and object
counts of the splits from a file small enough for git.
"""

import json

from pathlib import Path
from typing import Any, Dict


def reduce_data_record(source_path: Path, destination_path: Path, options: Dict[str, Any]) -> None:
    """
    Writes the data record of a run without the 'images' list of its splits, every other key kept (the creation date,
    the data folder, labels SHA-256 and number of images of every split, the images and annotations of every class),
    indented like the record the trainer writes; the object detection figures read only the number of images of every
    split and the annotations of every class from it.

    Args:
        source_path (Path): The data.json of the run.
        destination_path (Path): The reduced record in the snapshot folder.
        options (Dict[str, Any]): The entry options of snapshot_files.yaml, unused.

    Raises:
        KeyError: If the record has no 'splits', or a split has no 'images' list, a layout the figures do not read.
    """
    record = json.loads(source_path.read_text(encoding="utf-8"))

    # File names of the images of every split dropped, the counts staying in number_images
    for split_name, split_record in record["splits"].items():
        if "images" not in split_record:
            raise KeyError(f"The split {split_name} of {source_path} has no 'images' list, its keys being "
                           f"{sorted(split_record)}.")
        del split_record["images"]
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(data=json.dumps(obj=record, indent=2), encoding="utf-8")
