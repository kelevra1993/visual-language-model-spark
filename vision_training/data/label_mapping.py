"""
Label mapping of every task: builds the label dictionary injected into the configuration by the configuration loader
(raw dataset categories to predicted class indices, with the background class of the object detector, the instance
segmenter and the semantic segmenter), saves it in the run folder and reads it back for the resumed, evaluated and
exported runs, names the predicted classes for the evaluation reports and viewers, and writes the 'label_mapping.json'
file of the dataset creation tools.
"""

import json
import os

from pathlib import Path
from typing import Any, Dict, List, Optional

from vision_training.utilities.os_utilities import format_path_link, print_color

# Tasks whose model predicts a background class at index 0 (the object detector, the instance segmenter, the semantic
# segmenter), and the keypoint detection, which keeps the class indices of the detector giving its boxes; the models
# of the other tasks only predict the dataset classes, indexed from 0
TASKS_WITH_BACKGROUND_CLASS = ["object_detection", "instance_segmentation", "semantic_segmentation",
                               "keypoint_detection"]

# File of the label dictionary of a run, written in its folder by the trainer and read back when the run resumes, is
# evaluated or exported, so that its classes keep their meaning when the mapping file is edited or moved
RUN_LABEL_DICTIONARY_FILE_NAME = "label_dictionary.json"


def build_label_dictionary(label_mapping_configuration: Dict[str, Any], task: str) -> Optional[Dict[Any, int]]:
    """
    Builds the label dictionary injected into the configuration at loading time, mapping every raw dataset category to
    the class index the model predicts. The data loaders use it to convert the dataset annotations, and the model
    takes its number of classes from it. The object detector, the instance segmenter and the semantic segmenter
    predict a background class at index 0, named by the key "background", added when the mapping file does not name
    it, even when the file maps categories to 0 to leave them out; the classification and sequence models only
    predict the mapped classes, whose indices start at 0.

    Args:
        label_mapping_configuration (Dict[str, Any]): The 'Data.label_mapping' section, with its 'mode' ("binary" maps
            every object or labelled pixel to class 1, object detection, instance segmentation and semantic
            segmentation only; "multi_class" reads the JSON mapping file of 'mapping_file_path').
        task (str): The task of the configuration ('ExperimentConfiguration.task').

    Returns:
        Optional[Dict[Any, int]]: The class index of every raw category, or None for an unknown mode or a mode the task
            does not support, which the configuration checker then reports.

    Raises:
        FileNotFoundError: If the multi-class mapping file does not exist.
    """
    mode = label_mapping_configuration["mode"]

    # Binary mode: every annotated object is foreground, which only makes sense next to a background class
    if mode == "binary":
        return {"background": 0, "foreground": 1} if task in TASKS_WITH_BACKGROUND_CLASS else None

    # Multi-class mode: the JSON file maps every raw category identifier to its class index
    if mode == "multi_class":
        mapping_path = label_mapping_configuration["mapping_file_path"]
        if mapping_path and os.path.exists(mapping_path):
            with open(mapping_path, "r") as file:
                parsed_mapping = parse_label_mapping(raw_mapping=json.load(file))
                # The background class of the tasks predicting it is always named, the categories mapped to 0 being
                # left out of the training rather than naming it (get_class_names)
                if task in TASKS_WITH_BACKGROUND_CLASS and "background" not in parsed_mapping:
                    parsed_mapping["background"] = 0
                return parsed_mapping
        raise FileNotFoundError(f"Label mapping file not found at: {mapping_path}")

    return None


def parse_label_mapping(raw_mapping: Dict[str, int]) -> Dict[Any, int]:
    """
    Turns a label mapping read from a JSON file, whose keys are strings, into the keys of the label dictionary: the
    category identifiers written with digits become integers, as in the manifests, and the names stay strings. Used for
    the mapping file of the configuration and for the label dictionary saved in a run folder, which round-trips
    exactly.

    Args:
        raw_mapping (Dict[str, int]): The class index of every category key, as read from the JSON file.

    Returns:
        Dict[Any, int]: The class index of every category identifier (or name).
    """
    return {int(key) if str(key).isdigit() else key: class_index for key, class_index in raw_mapping.items()}


def write_run_label_dictionary(label_dictionary: Dict[Any, int], run_folder: Path) -> Path:
    """
    Saves the label dictionary of a run in its folder (RUN_LABEL_DICTIONARY_FILE_NAME), in the format of the mapping
    files, when the trainer creates or resumes the run folder, so that the classes of its checkpoints stay known
    whatever happens to the mapping file afterwards.

    Args:
        label_dictionary (Dict[Any, int]): The class index of every raw category identifier (or class name).
        run_folder (Path): The folder of the run.

    Returns:
        Path: The written file.
    """
    label_dictionary_path = run_folder / RUN_LABEL_DICTIONARY_FILE_NAME
    label_dictionary_path.write_text(data=json.dumps(obj={str(key): class_index
                                                          for key, class_index in label_dictionary.items()}, indent=4))
    return label_dictionary_path


def read_run_label_dictionary(run_folder: Path) -> Optional[Dict[Any, int]]:
    """
    Reads the label dictionary saved in a run folder (write_run_label_dictionary), for the configuration loader of the
    evaluation, export and model loading tools and for the trainer of a resumed run, with the keys of
    build_label_dictionary.

    Args:
        run_folder (Path): The folder of the run, or of a configuration file.

    Returns:
        Optional[Dict[Any, int]]: The saved label dictionary, or None when the folder holds none (not a run folder, or
            a run started before the file existed).
    """
    label_dictionary_path = run_folder / RUN_LABEL_DICTIONARY_FILE_NAME
    if not label_dictionary_path.is_file():
        return None
    return parse_label_mapping(raw_mapping=json.loads(label_dictionary_path.read_text()))


def get_background_class_index(label_dictionary: Dict[Any, int]) -> Optional[int]:
    """
    Gives the background class of a label dictionary, the class 0 named by the key "background" that
    build_label_dictionary adds for the tasks predicting a background: the categories mapped to it are left out of the
    training by the loaders of these tasks (their objects are not learnt, their pixels are background), so they name
    no class and are counted in no class (get_class_names, data_record.map_categories_to_classes).

    Args:
        label_dictionary (Dict[Any, int]): The class index of every raw category identifier (or class name).

    Returns:
        Optional[int]: 0 for a dictionary with a background class, None otherwise.
    """
    if "background" in label_dictionary and label_dictionary["background"] == 0:
        return 0
    return None


def get_class_names(categories: List[Dict[str, Any]], label_dictionary: Dict[Any, int]) -> Dict[int, str]:
    """
    Names the classes predicted by the classification, sequence and detection models after the categories of a
    'labels.json' manifest, for the evaluation reports, the confusion matrices and the viewer captions. The names of the
    categories merged into one class by the label dictionary are joined with '+', and a class named by no category
    keeps the name it has in the label dictionary (e.g. 'background' and 'foreground' of the detector), or else its
    index. The background class of the detection and segmentation tasks is always named 'background', the categories
    mapped to it being left out of the training (get_background_class_index).

    Args:
        categories (List[Dict[str, Any]]): The 'id' and 'name' of every category of the manifest.
        label_dictionary (Dict[Any, int]): The class index of every raw category identifier (or class name).

    Returns:
        Dict[int, str]: The name of every class index.
    """
    background_class_index = get_background_class_index(label_dictionary=label_dictionary)
    category_names = {}
    for category in categories:
        if category["id"] in label_dictionary and label_dictionary[category["id"]] != background_class_index:
            category_names.setdefault(label_dictionary[category["id"]], []).append(category["name"])

    # Classes named by no category: their name in the label dictionary when they have one, otherwise their index
    dictionary_names = {class_index: key for key, class_index in label_dictionary.items() if isinstance(key, str)}
    class_names = {}
    for class_index in range(len(set(label_dictionary.values()))):
        if class_index in category_names:
            class_names[class_index] = "+".join(category_names[class_index])
        else:
            class_names[class_index] = dictionary_names[class_index] if class_index in dictionary_names else str(
                class_index)
    return class_names


def write_label_mapping(categories: List[Dict[str, Any]], output_folder: str | Path) -> Path:
    """
    Writes the 'label_mapping.json' file of the 'multi_class' label mapping mode for manifests whose categories are
    numbered from 1, as built from the class folders of the ImageNet-22k download: every category maps to its
    identifier minus 1, the classification-like models predicting no background class.

    Args:
        categories (List[Dict[str, Any]]): The 'id' and 'name' of every category, numbered from 1.
        output_folder (str | Path): The folder receiving 'label_mapping.json'.

    Returns:
        Path: The absolute path of the written file.
    """
    label_mapping_path = Path(output_folder).absolute() / "label_mapping.json"
    label_mapping_path.write_text(data=json.dumps(obj={str(category["id"]): category["id"] - 1
                                                       for category in categories}, indent=4))
    print_color(color="green", output=f"Label Mapping File : {format_path_link(path=label_mapping_path)}",
                add_separators=True)
    return label_mapping_path
