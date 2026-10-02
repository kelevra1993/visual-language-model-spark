import json
import os

from typing import Any, Dict, List, Optional

# Tasks whose model predicts a background class at index 0 (the object detector); the models of the other tasks only
# predict the dataset classes, indexed from 0
TASKS_WITH_BACKGROUND_CLASS = ["object_detection"]


def build_label_dictionary(label_mapping_configuration: Dict[str, Any], task: str) -> Optional[Dict[Any, int]]:
    """
    Builds the label dictionary injected into the configuration at loading time, mapping every raw dataset category to
    the class index the model predicts. The data loaders use it to convert the dataset annotations, and the model
    takes its number of classes from it. The object detector predicts a background class at index 0, added when the
    mapping file does not name it; the classification and sequence models only predict the mapped classes, whose
    indices start at 0.

    Args:
        label_mapping_configuration (Dict[str, Any]): The 'Data.label_mapping' section, with its 'mode' ("binary" maps
            every object to class 1, object detection only; "multi_class" reads the JSON mapping file of
            'mapping_file_path').
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
                raw_mapping = json.load(file)
                # Convert string keys to int if they are digits for easier mapping during data loading
                parsed_mapping = {int(k) if str(k).isdigit() else k: v for k, v in raw_mapping.items()}
                # Ensure background is accounted for, for the tasks predicting it
                if task in TASKS_WITH_BACKGROUND_CLASS and "background" not in parsed_mapping and \
                        0 not in parsed_mapping.values():
                    parsed_mapping["background"] = 0
                return parsed_mapping
        raise FileNotFoundError(f"Label mapping file not found at: {mapping_path}")

    return None


def get_class_names(categories: List[Dict[str, Any]], label_dictionary: Dict[Any, int]) -> Dict[int, str]:
    """
    Names the classes predicted by the classification and sequence models after the categories of a 'labels.json'
    manifest, for the evaluation reports, the confusion matrices and the viewer captions. The names of the categories
    merged into one class by the label dictionary are joined with '+', and a class named by no category keeps its
    index as name.

    Args:
        categories (List[Dict[str, Any]]): The 'id' and 'name' of every category of the manifest.
        label_dictionary (Dict[Any, int]): The class index of every raw category identifier.

    Returns:
        Dict[int, str]: The name of every class index.
    """
    category_names = {}
    for category in categories:
        if category["id"] in label_dictionary:
            category_names.setdefault(label_dictionary[category["id"]], []).append(category["name"])
    return {class_index: "+".join(category_names.get(class_index, [str(class_index)]))
            for class_index in range(len(set(label_dictionary.values())))}
