"""
Converts a classification dataset stored as one folder of images per class into the split layout read by
ClassificationDataset ('data/' symbolic link and 'labels.json' manifest) and its 'label_mapping.json', without
copying any image. A split can also gather several class-folder datasets (e.g. two collections of the same classes),
each linked under its own name in 'data/', and one image folder can serve two splits. Run by the ImageNet-22k download
(imagenet_download.py), whose validation and test splits read the same folder, or as a script with its main block.
"""

import json

from pathlib import Path
from typing import Dict, Optional, Set, Union

from vision_training.data.label_mapping import write_label_mapping
from vision_training.utilities.os_utilities import format_path_link, get_image_paths, link_folder, print_color


def get_split_sources(split_folder: Union[str, Dict[str, str]]) -> Dict[str, Path]:
    """
    Gives the source folders of a split in one form: a single class-folder tree, read from the 'data' link itself and
    keyed by the empty name, or several named trees, each read from its own link in the 'data' folder.

    Args:
        split_folder (Union[str, Dict[str, str]]): The folder holding one subfolder of images per class, or the folders
            of several such datasets keyed by source name (e.g. "first-collection" and "second-collection").

    Returns:
        Dict[str, Path]: The folder of every source, keyed by source name ("" for a single folder).

    Raises:
        ValueError: If a source name is empty or holds a '/', since it becomes a folder name of the file names.
    """
    if isinstance(split_folder, str):
        return {"": Path(split_folder)}
    for source_name in split_folder:
        if not source_name or "/" in source_name:
            raise ValueError(f"A source name must be a non-empty folder name without '/', got '{source_name}'")
    return {source_name: Path(source_folder) for source_name, source_folder in split_folder.items()}


def link_split_sources(data_path: Path, sources: Dict[str, Path]) -> None:
    """
    Links the source folders of a split under its 'data' path, so that the images are read where they are: the data
    path itself links to a single source, or becomes a real folder holding one link per named source.

    Args:
        data_path (Path): The 'data' path of the created split.
        sources (Dict[str, Path]): The folder of every source, keyed by source name ("" for a single folder).

    Raises:
        FileExistsError: If the data path is a real folder for a single source, or a link for several sources.
    """
    if "" in sources:
        link_folder(link_path=data_path, target_folder=sources[""])
        return

    # Several sources: a real folder of links, never a link left by an earlier single-source conversion
    if data_path.is_symlink():
        raise FileExistsError(f"The data path of the split is a link, expected a folder of source links : "
                              f"{format_path_link(path=data_path)}")
    data_path.mkdir(parents=True, exist_ok=True)
    for source_name, source_folder in sources.items():
        link_folder(link_path=data_path / source_name, target_folder=source_folder)


def create_class_folder_manifests(split_folders: Dict[str, Union[str, Dict[str, str]]], output_folder: str,
                                  excluded_file_names: Optional[Dict[str, Set[str]]]) -> None:
    """
    Converts a classification dataset stored as one folder of images per class (e.g. 'training_data/circles/*.png')
    into the split layout read by the classification loader, without copying any image: '<output_folder>/<split>/data'
    is a symbolic link to the class folders of the split, '<output_folder>/<split>/labels.json' lists every image with
    its category, and '<output_folder>/label_mapping.json' maps every category to its class index (0 to K-1) for the
    'multi_class' label mapping mode of the configuration. The categories are the class folders found in any split,
    numbered from 1 in alphabetical order, so that every split uses the same category identifiers. A split given
    several named source folders gets one link per source in its 'data' folder, its file names starting with the
    source name, so that several datasets of the same classes are merged without copying an image.

    Args:
        split_folders (Dict[str, Union[str, Dict[str, str]]]): For every split name (e.g. 'train', 'validation'), the
            folder holding one subfolder of images per class, or the folders of several such datasets keyed by
            source name.
        output_folder (str): The root folder of the created splits.
        excluded_file_names (Optional[Dict[str, Set[str]]]): For a split name, the file names (as written in its
            manifest, e.g. 'first-collection/n01440764/n01440764_10025.jpg') left out of it, or None to keep every
            image.

    Raises:
        FileExistsError: If the 'data' path of a created split exists and is not a symbolic link.
        ValueError: If an excluded file name is not an image of its split.
    """
    split_sources = {split_name: get_split_sources(split_folder=split_folder)
                     for split_name, split_folder in split_folders.items()}

    # Categories shared by every split: the class folders of all the sources, numbered from 1 in alphabetical order
    class_names = sorted({path.name for sources in split_sources.values() for source_folder in sources.values()
                          for path in source_folder.iterdir() if path.is_dir()})
    categories = [{"id": category_index + 1, "name": class_name} for category_index, class_name in
                  enumerate(class_names)]

    for split_name, sources in split_sources.items():
        output_split_folder = Path(output_folder).absolute() / split_name
        output_split_folder.mkdir(parents=True, exist_ok=True)

        # Link the class folders of every source of the split, so that the images are read where they are
        link_split_sources(data_path=output_split_folder / "data", sources=sources)

        # One sample per image of every class folder of every source, named relative to the data folder
        split_excluded_names = (excluded_file_names or {}).get(split_name, set())
        samples, number_excluded = [], 0
        for source_name, source_folder in sources.items():
            prefix = f"{source_name}/" if source_name else ""
            for category in categories:
                class_folder = source_folder / category["name"]
                if not class_folder.is_dir():
                    continue
                for image_path in get_image_paths(folder=class_folder):
                    file_name = f"{prefix}{category['name']}/{image_path.name}"
                    # Leave out the images asked for, counted to check that every one was found
                    if file_name in split_excluded_names:
                        number_excluded += 1
                        continue
                    samples.append({"id": len(samples) + 1, "file_name": file_name, "category_id": category["id"]})
        if number_excluded != len(split_excluded_names):
            raise ValueError(f"{len(split_excluded_names) - number_excluded} excluded file names of the {split_name} "
                             f"split are not images of it")

        labels_path = output_split_folder / "labels.json"
        labels_path.write_text(data=json.dumps(obj={"categories": categories, "samples": samples}))
        excluded_text = f", {number_excluded} left out" if number_excluded else ""
        print_color(color="green", output=f"{split_name} split : {len(samples)} images{excluded_text} : "
                                          f"{format_path_link(path=labels_path)}")

    # Class index of every category, from 0 since the classifiers predict no background class
    write_label_mapping(categories=categories, output_folder=output_folder)


if __name__ == "__main__":
    # Folder of every split, holding one subfolder of images per class
    split_folders = {"train": "/home/robert_kelevra/Data/classifier_data/training_data",
                     "validation": "/home/robert_kelevra/Data/classifier_data/validation_data"}
    # Root folder of the created splits, whose split folders and label_mapping.json a classification configuration
    # reads. Existing manifests are overwritten
    output_folder = "/home/robert_kelevra/Data/classifier_data/manifests"

    create_class_folder_manifests(split_folders=split_folders, output_folder=output_folder, excluded_file_names=None)
