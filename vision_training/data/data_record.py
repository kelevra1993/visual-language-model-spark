"""
Record of the data a run is trained on: the 'data.json' file the base trainer writes in the run folder next to
'training_configuration.yaml', giving the classes of the run in class index order, each with its name, the
manifest categories mapped to it and its number of images (or annotations, sequences, frames, pairs) in every split,
and, for every split (train, validation, test), its data folder, the SHA-256 of its 'labels.json' and the list of its
images, so that the classes and the images a model was trained and evaluated on stay known when the data folders
change, and that a run adding classes after those of its source run can check that the source classes come first
(model_transfer.py). The images are read from the four manifest formats of the framework: the COCO 'images' of the
detection tasks, the 'samples' of the classification and semantic segmentation, the frames of the 'sequences' of the
sequence tasks and the frames of both members of the 'pairs' of the sequence comparison. A resumed run compares its
data with the saved record (BaseTrainer.save_data_record) and keeps the previous record when the data changed.
"""

import hashlib

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from vision_training.data.sequence.sequence_pair_dataset import PAIR_MEMBERS
from vision_training.utilities.os_utilities import read_json

# Name of the record of the data in the run folder
DATA_RECORD_FILE_NAME = "data.json"

# Entry of the record listing the classes of the run in class index order (background first for the detection and
# segmentation tasks), with their names read by the transfer of the class layers to a run adding classes after those
# of its source run
CLASSES_RECORD_KEY = "classes"

# Entry of the records written before the classes were counted, holding only their names, still read from a source run
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


def map_categories_to_classes(categories: List[Dict[str, Any]], label_dictionary: Dict[Any, int], label_mode: str,
                              unmapped_class_index: Optional[int]) -> Dict[Any, int]:
    """
    Gives the class index of every manifest category for the counts of the classes of the record of the data, the way
    the data loaders map them: through the label dictionary in multi-class mode, every category being the foreground
    in binary mode. The categories the label dictionary does not map are left out, as the loaders leave them out,
    except for the frame selection, whose loader gives their frames to the unselected class.

    Args:
        categories (List[Dict[str, Any]]): The 'id' and 'name' of every category of the manifests of the run.
        label_dictionary (Dict[Any, int]): The class index of every raw category identifier.
        label_mode (str): The 'Data.label_mapping.mode' of the run, "binary" or "multi_class".
        unmapped_class_index (Optional[int]): The class given to the categories the label dictionary does not map (the
            unselected class of the frame selection), None to leave them out.

    Returns:
        Dict[Any, int]: The class index of every counted category identifier.
    """
    # Binary mode: every category of the manifest is the foreground class
    if label_mode == "binary":
        return {category["id"]: label_dictionary["foreground"] for category in categories}

    # Multi-class mode: the class of the label dictionary, or the class of the unmapped categories when there is one
    category_classes = {}
    for category in categories:
        if category["id"] in label_dictionary:
            category_classes[category["id"]] = label_dictionary[category["id"]]
        elif unmapped_class_index is not None:
            category_classes[category["id"]] = unmapped_class_index
    return category_classes


def count_manifest_classes(labels: Dict[str, Any], category_classes: Dict[Any, int]) -> Dict[str, Counter]:
    """
    Counts the classes of one split from its 'labels.json', for the per-class counts of the record of the data, in the
    units of its manifest format: the 'images' holding the class and its 'annotations' for the COCO manifests of the
    detection, instance segmentation and keypoints, the 'images' of the class for the samples of the classification
    and self-supervised tasks (once per class of an image for the multi-label classification), the 'sequences' of
    the class (sequence classification) and the 'frames' of the class (sequence labelling and frame selection) for the
    sequences, and the 'pairs' of the class for the sequence comparison. The samples of the semantic segmentation
    count the masks holding the class from the 'category_ids' of every sample. A unit is only counted when the
    manifest gives its categories, so that a segmentation manifest without 'category_ids' and an unlabelled
    self-supervised dataset have no count.

    Args:
        labels (Dict[str, Any]): The content of the 'labels.json' of the split.
        category_classes (Dict[Any, int]): The class index of every mapped category (map_categories_to_classes).

    Returns:
        Dict[str, Counter]: The number of units of every class index, by unit name.

    Raises:
        ValueError: If the labels file has none of the four formats.
    """
    # COCO manifests: every annotation of a mapped category, and every image holding at least one of them
    if "images" in labels:
        annotation_classes = [(annotation["image_id"], category_classes[annotation["category_id"]])
                              for annotation in labels["annotations"]
                              if annotation["category_id"] in category_classes]
        return {"images": Counter(class_index for _, class_index in set(annotation_classes)),
                "annotations": Counter(class_index for _, class_index in annotation_classes)}

    # Samples: the class of every image, or each of its classes once for the multi-label classification and the masks
    # of the semantic segmentation
    if "samples" in labels:
        if not any("category_id" in sample or "category_ids" in sample for sample in labels["samples"]):
            return {}
        image_counts = Counter()
        for sample in labels["samples"]:
            sample_categories = sample["category_ids"] if "category_ids" in sample else [sample["category_id"]]
            image_counts.update({category_classes[category] for category in sample_categories
                                 if category in category_classes})
        return {"images": image_counts}

    # Sequences: the class of every sequence, and the class of every frame, for the categories the manifest gives
    if "sequences" in labels:
        class_counts = {}
        if any("category_id" in sequence for sequence in labels["sequences"]):
            class_counts["sequences"] = Counter(category_classes[sequence["category_id"]]
                                                for sequence in labels["sequences"]
                                                if sequence["category_id"] in category_classes)
        if any("frame_category_ids" in sequence for sequence in labels["sequences"]):
            class_counts["frames"] = Counter(category_classes[category] for sequence in labels["sequences"]
                                             for category in sequence["frame_category_ids"]
                                             if category in category_classes)
        return class_counts

    # Pairs: the comparison class of every pair
    if "pairs" in labels:
        return {"pairs": Counter(category_classes[pair["category_id"]] for pair in labels["pairs"]
                                 if pair["category_id"] in category_classes)}
    raise ValueError(f"The labels file has none of the 'images', 'samples', 'sequences' or 'pairs' lists of the "
                     f"framework, its keys are {sorted(labels)}.")


def build_class_records(class_names: List[str], label_dictionary: Dict[Any, int], label_mode: str,
                        unmapped_class_index: Optional[int],
                        split_data_folders: Dict[str, str | Path]) -> List[Dict[str, Any]]:
    """
    Describes the classes of a run for the record of its data ('classes' of 'data.json'): the index and name of every
    class, the manifest categories mapped to it (none for the background of the detection and segmentation tasks),
    and its number of images, annotations, sequences, frames or pairs in every split (count_manifest_classes), so that
    the class balance of the data a model was trained and evaluated on is read from its run folder.

    Args:
        class_names (List[str]): The class names of the run, in class index order (BaseTrainer.list_class_names).
        label_dictionary (Dict[Any, int]): The class index of every raw category identifier.
        label_mode (str): The 'Data.label_mapping.mode' of the run, "binary" or "multi_class".
        unmapped_class_index (Optional[int]): The class given to the categories the label dictionary does not map
            (BaseTrainer.get_unmapped_class_index), None to leave them out.
        split_data_folders (Dict[str, str | Path]): The root folder of every split, containing a 'labels.json'.

    Returns:
        List[Dict[str, Any]]: The 'class_index', 'class_name', 'category_ids' and the count of every split by unit of
            every class.
    """
    split_labels = {split_name: read_json(path=str(Path(split_data_folder) / "labels.json"))
                    for split_name, split_data_folder in split_data_folders.items()}

    # Categories of every manifest by identifier, the first split naming a category giving its name
    merged_categories = {}
    for labels in split_labels.values():
        for category in labels["categories"]:
            merged_categories.setdefault(category["id"], category)
    category_classes = map_categories_to_classes(categories=list(merged_categories.values()),
                                                 label_dictionary=label_dictionary, label_mode=label_mode,
                                                 unmapped_class_index=unmapped_class_index)

    # Counts of every split, the units of all the splits being given to every split
    split_counts = {split_name: count_manifest_classes(labels=labels, category_classes=category_classes)
                    for split_name, labels in split_labels.items()}
    units = list(dict.fromkeys(unit for class_counts in split_counts.values() for unit in class_counts))

    # One record per class, with the categories mapped to it and its count in every split for every unit
    class_records = []
    for class_index, class_name in enumerate(class_names):
        class_record = {"class_index": class_index, "class_name": class_name,
                        "category_ids": [category for category, mapped_index in category_classes.items()
                                         if mapped_index == class_index]}
        for unit in units:
            class_record[unit] = {split_name: class_counts[unit][class_index] if unit in class_counts else 0
                                  for split_name, class_counts in split_counts.items()}
        class_records.append(class_record)
    return class_records


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
