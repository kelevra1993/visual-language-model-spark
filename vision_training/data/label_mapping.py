import json
import os

from typing import Any, Dict, Optional


def build_label_dictionary(label_mapping_configuration: Dict[str, Any]) -> Optional[Dict[Any, int]]:
    """
    Builds the label dictionary injected into the configuration at loading time, mapping every raw dataset category to
    the class index the model predicts. The data loaders use it to convert the dataset annotations, and the model
    takes its number of classes from it.

    Args:
        label_mapping_configuration (Dict[str, Any]): The 'Data.label_mapping' section, with its 'mode' ("binary" maps
            every object to class 1, "multi_class" reads the JSON mapping file of 'mapping_file_path').

    Returns:
        Optional[Dict[Any, int]]: The class index of every raw category (index 0 being the background), or None for
            an unknown mode, which the configuration checker then reports.

    Raises:
        FileNotFoundError: If the multi-class mapping file does not exist.
    """
    mode = label_mapping_configuration["mode"]

    # Binary mode: every annotated object is foreground
    if mode == "binary":
        return {"background": 0, "foreground": 1}

    # Multi-class mode: the JSON file maps every raw category identifier to its class index
    if mode == "multi_class":
        mapping_path = label_mapping_configuration["mapping_file_path"]
        if mapping_path and os.path.exists(mapping_path):
            with open(mapping_path, "r") as file:
                raw_mapping = json.load(file)
                # Convert string keys to int if they are digits for easier mapping during data loading
                parsed_mapping = {int(k) if str(k).isdigit() else k: v for k, v in raw_mapping.items()}
                # Ensure background is accounted for
                if "background" not in parsed_mapping and 0 not in parsed_mapping.values():
                    parsed_mapping["background"] = 0
                return parsed_mapping
        raise FileNotFoundError(f"Label mapping file not found at: {mapping_path}")

    return None
