import os

from pathlib import Path
from typing import Tuple

from vision_training.utilities.os_utilities import read_json


def count_split_images_and_annotations(split_data_folder: str | Path) -> Tuple[int, int]:
    """
    Counts the images and annotations of a dataset split from its COCO 'labels.json', printed by the Trainer at
    startup. The file is read directly because the DALI loader rounds its length up to a whole number of batches,
    so neither loader length gives the exact number of images.

    Args:
        split_data_folder (str | Path): The root folder of the split, containing a 'labels.json'.

    Returns:
        Tuple[int, int]: The number of images, and the number of raw annotations (before label mapping, 0 for an
            unlabelled split without an 'annotations' list).
    """
    labels = read_json(path=os.path.join(str(split_data_folder), "labels.json"))
    number_annotations = len(labels["annotations"]) if "annotations" in labels else 0
    return len(labels["images"]), number_annotations
