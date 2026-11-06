"""
Builds the merged ImageNet manifests for the pre-training of the backbones on the classification task, from the
datasets written by imagenet_download.py (ImageNet-1k) and imagenet_22k_download.py (ImageNet-22k), without copying
any image: the train split gathers every split of ImageNet-22k and the train split of ImageNet-1k, the duplicates of
the two being kept, while the validation and test splits are those of ImageNet-1k, whose copies in ImageNet-22k are
left out of the training images so that the validation scores stay those of unseen images. Run as a script with the
dataset folders set in its main block.
"""

import os

from pathlib import Path
from typing import Dict, Set

from vision_training.data.classification.class_folder_manifests import create_class_folder_manifests
from vision_training.utilities.os_utilities import print_color

# Splits of ImageNet-22k all going to the merged train split, its own validation and test images being wasted otherwise
IMAGENET_22K_SPLITS = ["train", "validation", "test"]


def get_original_name(file_name: str) -> str:
    """
    Gives the original ImageNet name of an image file of either dataset, the key matching the same picture in both:
    ImageNet-1k appends its WordNet identifier ('n01440764_10026_n01440764.JPEG'), ImageNet-22k does not
    ('n01440764_10026.jpg').

    Args:
        file_name (str): The file name of an image, without folder.

    Returns:
        str: The original name, e.g. 'n01440764_10026'.
    """
    stem = file_name.rsplit(".", maxsplit=1)[0]
    image_key, _, suffix = stem.rpartition("_")
    # The ImageNet-1k suffix repeats the WordNet identifier starting the name
    if image_key and suffix == stem.split("_", maxsplit=1)[0]:
        return image_key
    return stem


def find_imagenet_22k_copies(imagenet_1k_split_folder: Path, imagenet_22k_images_folder: Path) -> Set[str]:
    """
    Finds the ImageNet-22k images having the original name of an image of an ImageNet-1k split (its validation split),
    as merged file names ('imagenet-22k-<split>/<WordNet identifier>/<name>.jpg'), so that they can be left out of the
    merged training images.

    Args:
        imagenet_1k_split_folder (Path): The class folders of the ImageNet-1k split.
        imagenet_22k_images_folder (Path): The folder of the split folders of ImageNet-22k.

    Returns:
        Set[str]: The merged file names of the ImageNet-22k copies.
    """
    copy_file_names = set()
    for class_entry in os.scandir(imagenet_1k_split_folder):
        if not class_entry.is_dir():
            continue
        original_names = {get_original_name(file_name=entry.name) for entry in os.scandir(class_entry.path)}
        # Look for every name in the same class folder of every ImageNet-22k split
        for split_name in IMAGENET_22K_SPLITS:
            for original_name in original_names:
                if (imagenet_22k_images_folder / split_name / class_entry.name / f"{original_name}.jpg").is_file():
                    copy_file_names.add(f"imagenet-22k-{split_name}/{class_entry.name}/{original_name}.jpg")
    return copy_file_names


def merge_imagenet_manifests(imagenet_1k_folder: str, imagenet_22k_folder: str, output_folder: str) -> None:
    """
    Writes the merged ImageNet splits read by the classification loader: 'train' (every ImageNet-22k split and the
    ImageNet-1k train split, without the ImageNet-22k copies of the ImageNet-1k validation images), 'validation' and
    'test' (those of ImageNet-1k), each with a 'data' folder linking to its sources and a 'labels.json' manifest, and
    'label_mapping.json' over the classes of both datasets (the ImageNet-22k classes and the ImageNet-1k class it
    lacks).

    Args:
        imagenet_1k_folder (str): The folder of ImageNet-1k, holding 'images/<split>/<WordNet identifier>/'.
        imagenet_22k_folder (str): The folder of ImageNet-22k, holding 'images/<split>/<WordNet identifier>/'.
        output_folder (str): The root folder of the merged splits.
    """
    images_1k = Path(imagenet_1k_folder) / "images"
    images_22k = Path(imagenet_22k_folder) / "images"

    # ImageNet-22k copies of the ImageNet-1k validation images, left out of the training images
    validation_copies = find_imagenet_22k_copies(imagenet_1k_split_folder=images_1k / "validation",
                                                 imagenet_22k_images_folder=images_22k)
    print_color(color="blue", output=f"{len(validation_copies)} ImageNet-22k copies of ImageNet-1k validation images "
                                     f"left out of the train split")

    # Every ImageNet-22k split and the ImageNet-1k train split for training, ImageNet-1k for the evaluations
    split_folders: Dict[str, Dict[str, str]] = {
        "train": {**{f"imagenet-22k-{split_name}": str(images_22k / split_name) for split_name in IMAGENET_22K_SPLITS},
                  "imagenet-1k-train": str(images_1k / "train")},
        "validation": {"imagenet-1k-validation": str(images_1k / "validation")},
        "test": {"imagenet-1k-test": str(images_1k / "test")}}
    create_class_folder_manifests(split_folders=split_folders, output_folder=output_folder,
                                  excluded_file_names={"train": validation_copies})


if __name__ == "__main__":
    # Datasets written by imagenet_download.py and imagenet_22k_download.py
    imagenet_1k_folder = "/home/robert_kelevra/Data/classifier_data/imagenet-1k"
    imagenet_22k_folder = "/home/robert_kelevra/Data/classifier_data/imagenet-22k"
    # Root folder of the merged splits, next to both datasets. Existing manifests are overwritten
    output_folder = "/home/robert_kelevra/Data/classifier_data/imagenet-merged"

    merge_imagenet_manifests(imagenet_1k_folder=imagenet_1k_folder, imagenet_22k_folder=imagenet_22k_folder,
                             output_folder=output_folder)
