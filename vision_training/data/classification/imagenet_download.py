"""
Downloads ImageNet-1k from Hugging Face for the classification task: the parquet shards fetched with resume, extracted
by worker processes into class folders, a validation split held out from the training images by file name hash, the
official validation images as the test split, the split size checks, the class folder manifests and the class names.
Run as a script with the dataset folder and worker counts set in its main block.
"""

import ast
import hashlib
import json
import multiprocessing
import os
import re

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow.parquet as parquet

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import GatedRepoError
from tqdm import tqdm

from vision_training.data.classification.class_folder_manifests import create_class_folder_manifests
from vision_training.utilities.os_utilities import format_path_link, print_color

# Gated Hugging Face dataset holding ImageNet-1k (ILSVRC 2012 classification) as parquet shards of JPEG images and
# class indices. Its terms must be accepted on the dataset page, and this machine logged in with 'uv run hf auth login'
IMAGENET_REPOSITORY = "ILSVRC/imagenet-1k"

# Shards of the labelled splits of the repository, the test split (no public labels) being left out
IMAGENET_SHARD_PATTERNS = {"train": "data/train-*.parquet", "validation": "data/validation-*.parquet"}

# File of the repository giving the WordNet identifier and the names of every class, in class index order
IMAGENET_CLASSES_FILE = "classes.py"

# Split receiving the images of every downloaded split: the official validation images become the test split, the
# validation split being held out from the training images
SPLIT_DESTINATIONS = {"train": "train", "validation": "test"}

# Number of classes, of training images and of official validation images of ImageNet-1k
NUMBER_CLASSES = 1000
NUMBER_TRAINING_IMAGES = 1281167
NUMBER_OFFICIAL_VALIDATION_IMAGES = 50000

# Rows of a shard read at once by an extraction worker (about 55 MB of JPEG bytes)
EXTRACTION_BATCH_SIZE = 500

# WordNet identifier starting the training file names, e.g. 'n01440764' in 'n01440764_10026.JPEG'
WORDNET_IDENTIFIER_PATTERN = re.compile(pattern=r"n\d{8}")


def download_imagenet_shards(download_folder: Path, number_workers: int) -> Tuple[Dict[str, List[Path]], Path]:
    """
    Downloads the parquet shards of the ImageNet-1k training and validation splits (153 GB) and the file naming its
    classes from Hugging Face, the first step of the preparation of the classification dataset used to pre-train the
    backbones. Complete files are skipped, so an interrupted download resumes where it stopped.

    Args:
        download_folder (Path): The folder receiving the files, in the layout of the repository ('data/*.parquet' and
            'classes.py').
        number_workers (int): The number of files downloaded in parallel.

    Returns:
        Tuple[Dict[str, List[Path]], Path]: The sorted shard paths of the 'train' and 'validation' splits, and the
            path of the classes file.

    Raises:
        PermissionError: If this machine is not logged in to Hugging Face with an account that accepted the terms of
            the dataset.
    """
    # Stop before downloading anything when the gated dataset can't be read with the credentials of this machine
    try:
        HfApi().auth_check(repo_id=IMAGENET_REPOSITORY, repo_type="dataset")
    except GatedRepoError as error:
        raise PermissionError(f"No access to the gated dataset {IMAGENET_REPOSITORY} : accept its terms on "
                              f"https://huggingface.co/datasets/{IMAGENET_REPOSITORY} with your Hugging Face account, "
                              f"then log in on this machine with 'uv run hf auth login'.") from error

    # Download the shards of the labelled splits and the classes file, the complete files being skipped
    print_color(color="blue", output=f"Downloading {IMAGENET_REPOSITORY} : {format_path_link(path=download_folder)}")
    snapshot_download(repo_id=IMAGENET_REPOSITORY, repo_type="dataset", local_dir=download_folder,
                      allow_patterns=[*IMAGENET_SHARD_PATTERNS.values(), IMAGENET_CLASSES_FILE],
                      max_workers=number_workers)

    # Shards of every split, sorted so that the splits are always extracted in the same order
    shard_paths = {split_name: sorted(download_folder.glob(pattern=shard_pattern))
                   for split_name, shard_pattern in IMAGENET_SHARD_PATTERNS.items()}
    return shard_paths, download_folder / IMAGENET_CLASSES_FILE


def read_imagenet_class_names(classes_file: Path) -> Dict[str, str]:
    """
    Reads the WordNet identifier and the names of every ImageNet class, in class index order, from the 'classes.py'
    file of the dataset, to name the class folders of the extracted images. The file is parsed, not executed, since it
    comes from the internet.

    Args:
        classes_file (Path): The file assigning the class names, keyed by WordNet identifier, to an OrderedDict.

    Returns:
        Dict[str, str]: The names of every class keyed by WordNet identifier, in class index order (e.g.
            'n01440764': 'tench, Tinca tinca' first).

    Raises:
        ValueError: If the file does not hold exactly one dictionary, or if its WordNet identifiers are not in
            alphabetical order: the manifests number the class folders alphabetically, which must give the class
            indices of the dataset labels.
    """
    # The classes are the only dictionary literal of the file
    dictionary_nodes = [node for node in ast.walk(ast.parse(source=classes_file.read_text()))
                        if isinstance(node, ast.Dict)]
    if len(dictionary_nodes) != 1:
        raise ValueError(f"Expected one dictionary of classes, found {len(dictionary_nodes)} in : "
                         f"{format_path_link(path=classes_file)}")
    class_names = ast.literal_eval(node_or_string=dictionary_nodes[0])

    # The alphabetical numbering of the class folders by the manifests must keep the class indices of the labels
    class_identifiers = list(class_names)
    if class_identifiers != sorted(class_identifiers):
        raise ValueError(f"The WordNet identifiers are not in alphabetical order in : "
                         f"{format_path_link(path=classes_file)}")
    return class_names


def extract_shard(shard_path: Path, split_folder: Path, class_identifiers: List[str], marker_path: Path) -> int:
    """
    Writes the JPEG images of one parquet shard, unchanged, into the class folders of its split
    ('<split_folder>/<WordNet identifier>/<file name>'), the layout turned into manifests by
    create_class_folder_manifests. It runs in a worker process for every shard, and creates the marker of the shard
    once all its images are written, so that an interrupted extraction only redoes the shards without a marker.

    Args:
        shard_path (Path): The parquet shard, with an 'image' column (JPEG 'bytes' and original file 'path') and a
            'label' column (class index).
        split_folder (Path): The folder of the class folders of the split, which must already exist.
        class_identifiers (List[str]): The WordNet identifier of every class index.
        marker_path (Path): The empty file created once the images of the shard are written.

    Returns:
        int: The number of images written.

    Raises:
        ValueError: If a label is not a class index, or if a training file name starts with the WordNet identifier of
            another class than its label (a wrong class order).
    """
    number_images = 0

    # Read the shard by batches of rows, keeping the memory of the worker low
    parquet_file = parquet.ParquetFile(source=shard_path)
    for batch in parquet_file.iter_batches(batch_size=EXTRACTION_BATCH_SIZE, columns=["image", "label"]):
        for image, label in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
            if not 0 <= label < len(class_identifiers):
                raise ValueError(f"Label {label} is not a class index in : {format_path_link(path=shard_path)}")
            class_identifier = class_identifiers[label]

            # Original file name of the image, or a name built from the shard and the row when the shard has none
            file_name = Path(image["path"]).name if image["path"] else f"{shard_path.stem}_{number_images:06d}.JPEG"

            # The training file names start with the WordNet identifier of their class, which checks the class order
            file_identifier = file_name.split("_")[0]
            if WORDNET_IDENTIFIER_PATTERN.fullmatch(file_identifier) and file_identifier != class_identifier:
                raise ValueError(f"{file_name} is labelled {class_identifier} in : {format_path_link(path=shard_path)}")

            (split_folder / class_identifier / file_name).write_bytes(data=image["bytes"])
            number_images += 1

    marker_path.touch()
    return number_images


def extract_imagenet_shards(shard_paths: Dict[str, List[Path]], images_folder: Path, markers_folder: Path,
                            class_identifiers: List[str], number_workers: int) -> None:
    """
    Writes the images of every downloaded shard into the class folders of its split, one worker process per shard,
    the official validation shards going to the test split (SPLIT_DESTINATIONS). The shards already marked as
    extracted are skipped, so that the step can be run again after an interruption.

    Args:
        shard_paths (Dict[str, List[Path]]): The shards of the 'train' and 'validation' splits of the dataset.
        images_folder (Path): The folder receiving one folder of class folders per split.
        markers_folder (Path): The folder of the markers of the extracted shards.
        class_identifiers (List[str]): The WordNet identifier of every class index.
        number_workers (int): The number of worker processes.
    """
    # Create every class folder of every split before the workers write into them
    for split_name in shard_paths:
        for class_identifier in class_identifiers:
            (images_folder / SPLIT_DESTINATIONS[split_name] / class_identifier).mkdir(parents=True, exist_ok=True)
    markers_folder.mkdir(parents=True, exist_ok=True)

    # Shards without the marker of a completed extraction, with the folder of their split and their marker
    pending_shards = []
    for split_name, split_shard_paths in shard_paths.items():
        for shard_path in split_shard_paths:
            marker_path = markers_folder / f"{shard_path.stem}.extracted"
            if not marker_path.exists():
                pending_shards.append((shard_path, images_folder / SPLIT_DESTINATIONS[split_name], marker_path))
    print_color(color="blue", output=f"Extracting {len(pending_shards)} shards : "
                                     f"{format_path_link(path=images_folder)}")

    # One task per shard, the progress bar counting the extracted shards. The workers are started fresh (spawn)
    # instead of forked, since the download leaves threads running in this process and forking a multi-threaded
    # process can deadlock the children
    number_images = 0
    with ProcessPoolExecutor(max_workers=number_workers,
                             mp_context=multiprocessing.get_context(method="spawn")) as executor:
        futures = [executor.submit(extract_shard, shard_path=shard_path, split_folder=split_folder,
                                   class_identifiers=class_identifiers, marker_path=marker_path)
                   for shard_path, split_folder, marker_path in pending_shards]
        for future in tqdm(as_completed(fs=futures), total=len(futures), desc="Extracting Shards"):
            number_images += future.result()
    print_color(color="green", output=f"{number_images} images extracted from {len(pending_shards)} shards", indent=1)


def compute_file_name_hash(file_name: str) -> str:
    """
    Gives the MD5 hash of a file name, the order in which hold_out_validation_images picks the validation images of a
    class: spread over the class like a random draw, but the same on every run and machine without a seed.

    Args:
        file_name (str): The file name of an image.

    Returns:
        str: The hexadecimal MD5 hash of the file name.
    """
    return hashlib.md5(file_name.encode(), usedforsecurity=False).hexdigest()


def hold_out_validation_images(train_folder: Path, validation_folder: Path, images_per_class: int) -> int:
    """
    Builds the validation split out of the training images, the official validation images being the test split:
    'images_per_class' images of every class folder of 'train_folder' are moved into the same class folder of
    'validation_folder'. The moved images are the ones whose file names have the smallest MD5 hashes, so that running
    it again gives the same split, completes an interrupted run, and moves nothing more once done. A training image
    also found in the validation split (written again by a new extraction) is deleted, so that no image is in both.

    Args:
        train_folder (Path): The folder of the class folders of the training images.
        validation_folder (Path): The folder receiving the class folders of the validation images.
        images_per_class (int): The number of validation images of every class.

    Returns:
        int: The number of images moved by this call.
    """
    number_moved_images = 0
    for train_class_folder in sorted(path for path in train_folder.iterdir() if path.is_dir()):
        validation_class_folder = validation_folder / train_class_folder.name
        validation_class_folder.mkdir(parents=True, exist_ok=True)
        validation_file_names = {path.name for path in validation_class_folder.iterdir()}

        # Delete the training copies of the images already held out, keeping the other training images
        train_file_names = []
        for image_path in train_class_folder.iterdir():
            if image_path.name in validation_file_names:
                image_path.unlink()
            else:
                train_file_names.append(image_path.name)

        # Move the training images with the smallest hashes until the class has its number of validation images
        number_missing_images = max(images_per_class - len(validation_file_names), 0)
        for file_name in sorted(train_file_names, key=compute_file_name_hash)[:number_missing_images]:
            (train_class_folder / file_name).rename(target=validation_class_folder / file_name)
            number_moved_images += 1
    return number_moved_images


def check_split_sizes(images_folder: Path, expected_split_sizes: Dict[str, int]) -> None:
    """
    Verifies that every split holds its expected number of images before the manifests are built on them, which
    catches an incomplete download or extraction, e.g. two images of a class written to the same file name.

    Args:
        images_folder (Path): The folder holding one folder of class folders per split.
        expected_split_sizes (Dict[str, int]): The expected number of images of every split, keyed by split name.

    Raises:
        RuntimeError: If a split holds another number of images.
    """
    for split_name, expected_split_size in expected_split_sizes.items():
        # Count the images of every class folder of the split
        split_folder = images_folder / split_name
        class_folders = [path for path in split_folder.iterdir() if path.is_dir()]
        split_size = sum(len(os.listdir(path=class_folder)) for class_folder in class_folders)
        if split_size != expected_split_size:
            raise RuntimeError(f"The {split_name} split holds {split_size} images instead of {expected_split_size} : "
                               f"{format_path_link(path=split_folder)}")
        print_color(color="green", output=f"{split_name} split : {split_size} images in {len(class_folders)} classes",
                    indent=1)


def download_imagenet(dataset_folder: str, validation_images_per_class: int, number_download_workers: int,
                      number_extraction_workers: int) -> None:
    """
    Downloads ImageNet-1k (ILSVRC 2012 classification: 1000 classes, 1,281,167 training and 50,000 validation images)
    from Hugging Face and prepares it for the classification task, whose trained backbones initialise the other tasks
    through Backbone.pretrained. The dataset folder ends up holding:
    - 'parquet/': the downloaded shards (153 GB), which can be deleted once the manifests exist;
    - 'images/<split>/<WordNet identifier>/': the original JPEG images (153 GB) of the train split, of the validation
      split held out from the training images, and of the test split made of the official validation images;
    - 'manifests/': the splits read by the classification loader ('<split>/data' linking to the images and
      '<split>/labels.json'), and 'label_mapping.json', the class index of every WordNet identifier being its
      ImageNet class index;
    - 'class_names.json': the names of every WordNet identifier.
    Every step skips the work already done, so it can be run again after an interruption.

    Args:
        dataset_folder (str): The folder of the dataset.
        validation_images_per_class (int): The number of training images of every class moved to the validation split.
        number_download_workers (int): The number of files downloaded in parallel.
        number_extraction_workers (int): The number of processes writing the images of the shards.

    Raises:
        ValueError: If the classes file does not give 1000 classes.
    """
    dataset_path = Path(dataset_folder).absolute()
    images_folder = dataset_path / "images"

    # Download the shards and the classes file of the gated dataset
    shard_paths, classes_file = download_imagenet_shards(download_folder=dataset_path / "parquet",
                                                         number_workers=number_download_workers)

    # Names of every class in class index order, saved next to the manifests, which only give the WordNet identifiers
    class_names = read_imagenet_class_names(classes_file=classes_file)
    if len(class_names) != NUMBER_CLASSES:
        raise ValueError(f"Expected {NUMBER_CLASSES} classes, found {len(class_names)} in : "
                         f"{format_path_link(path=classes_file)}")
    class_names_path = dataset_path / "class_names.json"
    class_names_path.write_text(data=json.dumps(obj=class_names, indent=4))

    # Write the images of every shard into the class folders of its split
    extract_imagenet_shards(shard_paths=shard_paths, images_folder=images_folder,
                            markers_folder=dataset_path / "extracted_shards", class_identifiers=list(class_names),
                            number_workers=number_extraction_workers)

    # Hold out the validation split from the training images
    number_moved_images = hold_out_validation_images(train_folder=images_folder / "train",
                                                     validation_folder=images_folder / "validation",
                                                     images_per_class=validation_images_per_class)
    print_color(color="blue", output=f"{number_moved_images} training images moved to the validation split")

    # Check every split, then build the manifests read by the classification loader
    number_validation_images = NUMBER_CLASSES * validation_images_per_class
    check_split_sizes(images_folder=images_folder,
                      expected_split_sizes={"train": NUMBER_TRAINING_IMAGES - number_validation_images,
                                            "validation": number_validation_images,
                                            "test": NUMBER_OFFICIAL_VALIDATION_IMAGES})
    create_class_folder_manifests(split_folders={split_name: str(images_folder / split_name)
                                                 for split_name in ["train", "validation", "test"]},
                                  output_folder=str(dataset_path / "manifests"))
    print_color(color="green", output=f"Class Names : {format_path_link(path=class_names_path)}")


if __name__ == "__main__":
    # Folder of the dataset, next to the COCO dataset of the object detection configurations. It needs about 310 GB:
    # 153 GB of downloaded shards, which can be deleted once the manifests exist, and 153 GB of extracted images
    dataset_folder = "/home/robert_kelevra/Data/object_detection_data/imagenet-1k"
    # Training images of every class moved to the validation split (10,000 in total), the official validation images
    # (50 per class) becoming the test split
    validation_images_per_class = 10
    # Number of files downloaded in parallel
    number_download_workers = 8
    # Number of processes writing the images of the downloaded shards
    number_extraction_workers = 12

    download_imagenet(dataset_folder=dataset_folder, validation_images_per_class=validation_images_per_class,
                      number_download_workers=number_download_workers,
                      number_extraction_workers=number_extraction_workers)
