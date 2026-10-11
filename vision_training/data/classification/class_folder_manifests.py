"""
Converts a classification dataset stored as one folder of images per class into the split layout read by
ClassificationDataset ('data/' symbolic link and 'labels.json' manifest) and its 'label_mapping.json', without
copying any image. Run from the classification example configuration instructions and by the ImageNet download.
"""

import json

from pathlib import Path
from typing import Dict

from vision_training.data.label_mapping import write_label_mapping
from vision_training.utilities.os_utilities import format_path_link, get_image_paths, link_folder, print_color


def create_class_folder_manifests(split_folders: Dict[str, str], output_folder: str) -> None:
    """
    Converts a classification dataset stored as one folder of images per class (e.g. 'training_data/circles/*.png')
    into the split layout read by the classification loader, without copying any image: '<output_folder>/<split>/data'
    is a symbolic link to the class folders of the split, '<output_folder>/<split>/labels.json' lists every image with
    its category, and '<output_folder>/label_mapping.json' maps every category to its class index (0 to K-1) for the
    'multi_class' label mapping mode of the configuration. The categories are the class folders found in any split,
    numbered from 1 in alphabetical order, so that every split uses the same category identifiers.

    Args:
        split_folders (Dict[str, str]): The folder of every split holding one subfolder of images per class, keyed by
            split name (e.g. 'train', 'validation').
        output_folder (str): The root folder of the created splits.

    Raises:
        FileExistsError: If the 'data' path of a created split exists and is not a symbolic link.
    """
    # Categories shared by every split: the class folders of all the splits, numbered from 1 in alphabetical order
    class_names = sorted({path.name for split_folder in split_folders.values() for path in Path(split_folder).iterdir()
                          if path.is_dir()})
    categories = [{"id": category_index + 1, "name": class_name} for category_index, class_name in
                  enumerate(class_names)]

    for split_name, split_folder in split_folders.items():
        output_split_folder = Path(output_folder).absolute() / split_name
        output_split_folder.mkdir(parents=True, exist_ok=True)

        # Link the class folders of the split as its data folder, so that the images are read where they are
        link_folder(link_path=output_split_folder / "data", target_folder=Path(split_folder))

        # One sample per image of every class folder of the split, named relative to the data folder
        samples = []
        for category in categories:
            class_folder = Path(split_folder) / category["name"]
            if not class_folder.is_dir():
                continue
            for image_path in get_image_paths(folder=class_folder):
                samples.append({"id": len(samples) + 1, "file_name": f"{category['name']}/{image_path.name}",
                                "category_id": category["id"]})

        labels_path = output_split_folder / "labels.json"
        labels_path.write_text(data=json.dumps(obj={"categories": categories, "samples": samples}))
        print_color(color="green", output=f"{split_name} split : {len(samples)} images : "
                                          f"{format_path_link(path=labels_path)}")

    # Class index of every category, from 0 since the classifiers predict no background class
    write_label_mapping(categories=categories, output_folder=output_folder)


if __name__ == "__main__":
    # Folder of every split, holding one subfolder of images per class
    split_folders = {"train": "/home/robert_kelevra/Data/classifier_data/training_data",
                     "validation": "/home/robert_kelevra/Data/classifier_data/validation_data"}
    # Root folder of the created splits, the one expected by
    # vision_training/configurations/classification/configuration_example.yaml. Existing manifests are overwritten
    output_folder = "/home/robert_kelevra/Data/classifier_data/manifests"

    create_class_folder_manifests(split_folders=split_folders, output_folder=output_folder)
