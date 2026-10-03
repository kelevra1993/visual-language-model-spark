"""
Converts a sequence comparison dataset described by one pair list file per split into the split layout read by
SequencePairDataset ('data/' symbolic link and 'labels.json' manifest) and its 'label_mapping.json', without copying
any frame. Run from the sequence comparison example configuration instructions.
"""

import json

from pathlib import Path
from typing import Dict

from vision_training.data.label_mapping import write_label_mapping
from vision_training.data.sequence.sequence_pair_dataset import PAIR_MEMBERS
from vision_training.utilities.os_utilities import get_image_paths, link_folder, print_color, read_json


def create_pair_list_manifests(pair_list_files: Dict[str, str], output_folder: str) -> None:
    """
    Converts a sequence comparison dataset described by one pair list file per split into the split layout read by
    the sequence pair dataset, without copying any frame. A pair list file is a JSON file
    {"source": <folder holding one folder of frames per sequence>, "data": [{"before": <frame folder>,
    "after": <frame folder>, "label": <class name>, "sequence": <pair name>}]}: '<output_folder>/<split>/data' becomes
    a symbolic link to the source folder of the split, '<output_folder>/<split>/labels.json' lists every pair with the
    frames of its two members (sorted by file name) and its category, and '<output_folder>/label_mapping.json' maps
    every category to its class index for the 'multi_class' label mapping mode. The categories are the labels found in
    any split, numbered from 1 in alphabetical order, so that every split uses the same category identifiers.

    Args:
        pair_list_files (Dict[str, str]): The pair list file of every split, keyed by split name (e.g. 'train',
            'validation').
        output_folder (str): The root folder of the created splits.

    Raises:
        FileNotFoundError: If a member folder of a pair holds no image.
        FileExistsError: If the 'data' path of a created split exists and is not a symbolic link.
    """
    pair_lists = {split_name: read_json(path=pair_list_file) for split_name, pair_list_file in pair_list_files.items()}

    # Categories shared by every split: the labels of all the splits, numbered from 1 in alphabetical order
    class_names = sorted({entry["label"] for pair_list in pair_lists.values() for entry in pair_list["data"]})
    categories = [{"id": category_index + 1, "name": class_name} for category_index, class_name in
                  enumerate(class_names)]
    category_identifiers = {category["name"]: category["id"] for category in categories}

    for split_name, pair_list in pair_lists.items():
        output_split_folder = Path(output_folder).absolute() / split_name
        output_split_folder.mkdir(parents=True, exist_ok=True)

        # Link the source folder of the split as its data folder, so that the frames are read where they are
        source_folder = Path(pair_list["source"])
        link_folder(link_path=output_split_folder / "data", target_folder=source_folder)

        # One entry per pair, with the frames of its members relative to the data folder, sorted by file name
        pairs = []
        for entry in pair_list["data"]:
            pair = {"id": len(pairs) + 1, "name": entry["sequence"]}
            for member_name in PAIR_MEMBERS:
                pair[member_name] = [f"{entry[member_name]}/{image_path.name}" for image_path in
                                     get_image_paths(folder=source_folder / entry[member_name])]
            pair["category_id"] = category_identifiers[entry["label"]]
            pairs.append(pair)

        labels_path = output_split_folder / "labels.json"
        labels_path.write_text(data=json.dumps(obj={"categories": categories, "pairs": pairs}))
        print_color(color="green", output=f"{split_name} split : {len(pairs)} pairs : file://{labels_path}")

    # Class index of every category, from 0 since the comparison model predicts no background class
    write_label_mapping(categories=categories, output_folder=output_folder)


if __name__ == "__main__":
    # Pair list file of every split, giving the source folder of its frames and the before folder, after folder,
    # label and name of every pair. A test split can be added with its own pair list file
    pair_list_files = {"train": "/home/robert_kelevra/Data/sequence_data/sequence_comparison/train.json",
                       "validation": "/home/robert_kelevra/Data/sequence_data/sequence_comparison/valid.json"}
    # Root folder of the created splits, the one expected by
    # vision_training/configurations/sequence_comparison/configuration_example.yaml. Existing manifests are overwritten
    output_folder = "/home/robert_kelevra/Data/sequence_data/sequence_comparison/manifests"

    create_pair_list_manifests(pair_list_files=pair_list_files, output_folder=output_folder)
