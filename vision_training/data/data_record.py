"""
Record of the data a run is trained on: the 'data.json' file the base trainer writes in the run folder next to
'training_configuration.yaml', giving the class names of the run in class index order and, for every split (train,
validation, test), its data folder, the SHA-256 of its 'labels.json' and the list of its images, so that the classes
and the images a model was trained and evaluated on stay known when the data folders change, and that a run adding
classes after those of its source run can check that the source classes come first (model_transfer.py). The images
are read from the four manifest formats of the framework: the COCO 'images' of the detection tasks, the 'samples' of
the classification and semantic segmentation, the frames of the 'sequences' of the sequence tasks and the frames of
both members of the 'pairs' of the sequence comparison. A resumed run compares its data with the saved record
(BaseTrainer.save_data_record) and keeps the previous record when the data changed.
"""

import hashlib

from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from vision_training.data.sequence.sequence_pair_dataset import PAIR_MEMBERS
from vision_training.utilities.os_utilities import read_json

# Name of the record of the data in the run folder
DATA_RECORD_FILE_NAME = "data.json"

# Entry of the record listing the class names of the run in class index order (background first for the detection and
# segmentation tasks), read by the transfer of the class layers to a run adding classes after those of its source run
CLASS_NAMES_RECORD_KEY = "class_names"

# Split of the record written for every data folder of the experiment configuration
DATA_RECORD_SPLITS = {"train": "train_data_folder", "validation": "validation_data_folder",
                      "test": "test_data_folder"}

# Entries of a split compared between two starts of a run, the labels file giving its images and their labels
COMPARED_SPLIT_ENTRIES = ["data_folder", "labels_sha256"]


class DataRecordChanges(TypedDict):
    """
    Changes of the data of a resumed run, found by the base trainer before it replaces the record saved in the run
    folder, and printed with the experiment information: the iteration of the checkpoint the run resumes from, the
    splits whose data folder or labels file changed, and the file keeping the previous record (None when nothing
    changed, the saved record being then kept).
    """
    resumed_iteration: int
    changed_splits: List[str]
    archived_data_record_path: Optional[Path]


def list_manifest_images(labels: Dict[str, Any]) -> List[str]:
    """
    Lists the images of a split from its 'labels.json', in the order of the manifest, whatever the task: the COCO
    'images' of the detection and instance segmentation, the 'samples' of the classification and semantic
    segmentation (the masks of the segmentation being given by the labels file), the frames of the 'sequences' of the
    sequence classification, sequence labelling and frame selection, and the frames of both members of the 'pairs'
    of the sequence comparison.

    Args:
        labels (Dict[str, Any]): The content of the 'labels.json' of the split.

    Returns:
        List[str]: The image file names, relative to the 'data' folder of the split.

    Raises:
        ValueError: If the labels file has none of the four formats.
    """
    # One list per manifest format, the frames of a sequence and of a pair member in their order
    if "images" in labels:
        return [image["file_name"] for image in labels["images"]]
    if "samples" in labels:
        return [sample["file_name"] for sample in labels["samples"]]
    if "sequences" in labels:
        return [frame for sequence in labels["sequences"] for frame in sequence["frames"]]
    if "pairs" in labels:
        return [frame for pair in labels["pairs"] for member_name in PAIR_MEMBERS for frame in pair[member_name]]
    raise ValueError(f"The labels file has none of the 'images', 'samples', 'sequences' or 'pairs' lists of the "
                     f"framework, its keys are {sorted(labels)}.")


def build_split_record(split_data_folder: str | Path) -> Dict[str, Any]:
    """
    Describes one split for the record of the data of a run: its absolute data folder, the SHA-256 of its
    'labels.json', which identifies exactly the images and labels the split held, and its images.

    Args:
        split_data_folder (str | Path): The root folder of the split, containing a 'labels.json'.

    Returns:
        Dict[str, Any]: The 'data_folder', 'labels_sha256', 'number_images' and 'images' of the split.
    """
    labels_path = Path(split_data_folder).resolve() / "labels.json"

    # Fingerprint of the labels file, read in blocks so that a large manifest is not held twice in memory
    with open(file=labels_path, mode="rb") as labels_file:
        labels_sha256 = hashlib.file_digest(labels_file, "sha256").hexdigest()

    images = list_manifest_images(labels=read_json(path=str(labels_path)))
    return {"data_folder": str(labels_path.parent), "labels_sha256": labels_sha256, "number_images": len(images),
            "images": images}


def find_changed_splits(previous_record: Dict[str, Any], current_record: Dict[str, Any]) -> List[str]:
    """
    Compares the record of the data saved by the previous start of a run with the record of the data it is resumed
    with: a split changed when its data folder or the content of its labels file changed, or when the previous record
    did not have it.

    Args:
        previous_record (Dict[str, Any]): The record saved in the run folder by the previous start.
        current_record (Dict[str, Any]): The record of the data the run is resumed with.

    Returns:
        List[str]: The names of the changed splits, in the order of the current record.
    """
    previous_splits = previous_record["splits"]
    return [split_name for split_name, current_split in current_record["splits"].items()
            if split_name not in previous_splits or
            any(previous_splits[split_name][entry] != current_split[entry] for entry in COMPARED_SPLIT_ENTRIES)]
