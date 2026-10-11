"""
Downloads Tencent ML-Images (17.6M multi-label images, 11,166 WordNet categories) for the pre-training of the backbones
on the classification task: the Tencent annotation lists, the ImageNet-22k images of Hugging Face and the Open Images
images of their public S3 bucket (training archives streamed, validation images as single files), rotated, resized and
re-encoded by worker processes, the duplicate images merged by perceptual hash, the ImageNet-1k test images left out,
and the multi-label manifests of the train, validation and test splits. Run as a script with the dataset folder and
worker counts set in its main block.
"""

import csv
import heapq
import json
import multiprocessing
import shutil
import tarfile
import urllib.request

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import boto3
import cv2
import gdown
import numpy as np

from botocore import UNSIGNED
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import ClientError
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError
from tqdm import tqdm

from vision_training.data.classification.imagenet_download import compute_file_name_hash
from vision_training.data.label_mapping import write_label_mapping
from vision_training.utilities.os_utilities import format_path_link, get_image_paths, link_folder, print_color

# Category dictionary of ML-Images: index, WordNet identifier, parent index and names of the 11,166 categories
TENCENT_DICTIONARY_URL = ("https://raw.githubusercontent.com/Tencent/tencent-ml-images/master/data/"
                          "dictionary_and_semantic_hierarchy.txt")
TENCENT_DICTIONARY_FILE = "dictionary_and_semantic_hierarchy.txt"

# Google Drive identifiers of the annotation lists of ML-Images, as given by the README of its repository
TENCENT_LIST_FILES = {"train_image_id_from_imagenet.txt": "1-7x4wPa764MJkjhhNj0PWPhgwMJOXziA",
                      "val_image_id_from_imagenet.txt": "1-1x1vJFZGesz-5R2W8DLWHaVEbIPjuJs",
                      "train_urls_from_openimages.txt": "1__HFVimF5yUwlyEjaUoSmBBfRQKJTYKW",
                      "val_urls_from_openimages.txt": "1JkTcEEkB1zYI6NtAM-vXpsv7uDZ3glEz"}

# Annotation list of every split of every source, the Tencent validation images becoming the test split
TENCENT_SPLIT_LISTS = {"imagenet": {"train": "train_image_id_from_imagenet.txt",
                                    "test": "val_image_id_from_imagenet.txt"},
                       "openimages": {"train": "train_urls_from_openimages.txt",
                                      "test": "val_urls_from_openimages.txt"}}

# Gated Hugging Face dataset holding the full ImageNet fall11 release (21,841 classes) as webdataset tar shards of
# about 0.29 GB, whose sample keys are the original file names (e.g. 'n04310904_8388'). Its terms must be accepted on
# the dataset page, and this machine logged in with 'uv run hf auth login'
IMAGENET_REPOSITORY = "timm/imagenet-22k-wds"

# Prefixes of the shards of both splits of the repository, a Tencent image being in either of them
IMAGENET_SHARD_PREFIXES = ("imagenet22k-train-", "imagenet22k-validation-")

# Metadata of every Open Images subset, giving the image identifier, the Flickr URL and the rotation of every image
OPEN_IMAGES_METADATA_URLS = {
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-images-with-labels-with-rotation.csv",
    "validation": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    "test": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv"}

# Public S3 bucket of the Open Images images, read without credentials
OPEN_IMAGES_BUCKET = "open-images-dataset"

# Subset whose images are only in the bucket as one gzip tar archive per first character of the image identifier
# ('tar/train_0.tar.gz' to 'tar/train_f.tar.gz', 30 to 49 GB each), streamed without being saved. The images of the
# other subsets are single files of the bucket ('<subset>/<image identifier>.jpg')
OPEN_IMAGES_ARCHIVE_SUBSET = "train"
OPEN_IMAGES_ARCHIVE_KEY_FORMAT = "tar/train_{}.tar.gz"

# Images stored by an archive worker between two flushes of its progress and two checks of the free disk space
ARCHIVE_CHECK_INTERVAL = 1000

# Single-file Open Images images downloaded by one worker task, the unit of resumption of their download
OPEN_IMAGES_BATCH_SIZE = 2000

# Attempts of every S3 request before the batch fails and is retried by the next run
S3_MAXIMUM_ATTEMPTS = 10

# Side of the grayscale thumbnail of the perceptual hash: 8 horizontal and 8 vertical differences per row and column,
# 128 bits in total
PERCEPTUAL_HASH_THUMBNAIL_SIDE = 9
PERCEPTUAL_HASH_BITS = 128

# Thumbnails with a lower grayscale standard deviation (blank or nearly uniform images) get no hash, since their hash
# would match unrelated images
MINIMUM_THUMBNAIL_STANDARD_DEVIATION = 2.0

# Differing bits up to which an image is a copy of an ImageNet-1k test image, found by splitting the hashes into
# one more chunk than this distance: two hashes this close share at least one identical chunk
EXCLUSION_MAXIMUM_DISTANCE = 3

# Factor applied to the number of workers to give the number of tasks queued in the process pool
QUEUED_TASKS_PER_WORKER = 2

# Splits of the created manifests
SPLIT_NAMES = ["train", "validation", "test"]


def download_file(url: str, output_path: Path) -> None:
    """
    Downloads a public file once, through a temporary '.part' file renamed at the end, so that an interrupted download
    never leaves a truncated file mistaken for a complete one by the next run. Used for the Tencent category dictionary
    and the Open Images metadata.

    Args:
        url (str): The URL of the file.
        output_path (Path): The path of the downloaded file, skipped when it already exists.
    """
    if output_path.exists():
        return
    print_color(color="blue", output=f"Downloading {url} : {format_path_link(path=output_path)}")
    temporary_path = output_path.with_name(f"{output_path.name}.part")
    with urllib.request.urlopen(url=url) as response, open(file=temporary_path, mode="wb") as output_file:
        shutil.copyfileobj(fsrc=response, fdst=output_file)
    temporary_path.rename(target=output_path)


def download_annotation_files(annotations_folder: Path) -> None:
    """
    Downloads the annotation files of ML-Images, the first step of the preparation: the category dictionary from GitHub
    and the four image lists (17.6M lines, 2 GB) from Google Drive, plus the Open Images metadata used to find the
    Open Images identifier of every Tencent URL. Complete files are skipped.

    Args:
        annotations_folder (Path): The folder receiving the files.

    Raises:
        RuntimeError: If Google Drive refuses a list file, usually after too many downloads of the file in a day: the
            message gives the link to download it by hand into the annotations folder.
    """
    annotations_folder.mkdir(parents=True, exist_ok=True)
    download_file(url=TENCENT_DICTIONARY_URL, output_path=annotations_folder / TENCENT_DICTIONARY_FILE)

    # Download every image list from Google Drive through a temporary file
    for file_name, drive_identifier in TENCENT_LIST_FILES.items():
        output_path = annotations_folder / file_name
        if output_path.exists():
            continue
        temporary_path = output_path.with_name(f"{file_name}.part")
        try:
            gdown.download(id=drive_identifier, output=str(temporary_path), quiet=False, resume=True)
        except Exception as error:
            raise RuntimeError(f"Google Drive refused {file_name} : download it from "
                               f"https://drive.google.com/file/d/{drive_identifier}/view in a browser, save it as "
                               f"{format_path_link(path=output_path)}, then run the script again.") from error
        temporary_path.rename(target=output_path)

    # Metadata of every Open Images subset, named after the subset
    for subset_name, url in OPEN_IMAGES_METADATA_URLS.items():
        download_file(url=url, output_path=annotations_folder / f"open_images_{subset_name}_metadata.csv")


def read_category_dictionary(dictionary_path: Path) -> List[Dict[str, Any]]:
    """
    Reads the 11,166 categories of ML-Images as the categories of the manifests, numbered from 1 so that
    write_label_mapping gives every category its Tencent index as class index. A category is named after the first
    synonym of its names, its WordNet identifier being appended when another category has the same name.

    Args:
        dictionary_path (Path): The tab-separated dictionary (index, WordNet identifier, parent index, names), with a
            header line.

    Returns:
        List[Dict[str, Any]]: The 'id' (Tencent index plus 1), 'name' and 'wordnet_identifier' of every category, in
            index order.

    Raises:
        ValueError: If the indices of the dictionary do not follow each other from 0.
    """
    # Index, WordNet identifier and first synonym of every line after the header
    entries = []
    for line in dictionary_path.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        category_index, wordnet_identifier, _, names = line.split(sep="\t", maxsplit=3)
        entries.append((int(category_index), wordnet_identifier, names.split(",")[0].strip()))
    if [entry[0] for entry in entries] != list(range(len(entries))):
        raise ValueError(f"The category indices do not follow each other from 0 in : "
                         f"{format_path_link(path=dictionary_path)}")

    # Names shared by several categories, which get their WordNet identifier appended
    name_counts = {}
    for _, _, name in entries:
        name_counts[name] = name_counts.get(name, 0) + 1
    return [{"id": category_index + 1, "wordnet_identifier": wordnet_identifier,
             "name": f"{name} ({wordnet_identifier})" if name_counts[name] > 1 else name}
            for category_index, wordnet_identifier, name in entries]


def parse_annotation_line(line: str) -> Tuple[str, List[int], List[float]]:
    """
    Splits a line of a Tencent image list into its image and its labels. The training lists give every label with its
    confidence ('2367:1', '5193:0.9'), the validation lists give the bare index, whose confidence is then 1.

    Args:
        line (str): The image (ImageNet file or Open Images URL) followed by its labels, separated by whitespace.

    Returns:
        Tuple[str, List[int], List[float]]: The image, the Tencent index of every label and its confidence.
    """
    image, *labels = line.split()
    category_indices, confidences = [], []
    for label in labels:
        category_index, _, confidence = label.partition(":")
        category_indices.append(int(category_index))
        confidences.append(float(confidence) if confidence else 1.0)
    return image, category_indices, confidences


def get_imagenet_key(image_identifier: str) -> str:
    """
    Gives the key of an ImageNet image of a Tencent list in the ImageNet-22k shards, its file name without folder nor
    extension.

    Args:
        image_identifier (str): The image of a Tencent list, e.g. 'n04310904/n04310904_8388.JPEG'.

    Returns:
        str: The key of the image, e.g. 'n04310904_8388'.
    """
    return image_identifier.rsplit("/", maxsplit=1)[-1].split(".")[0]


def encode_imagenet_key(imagenet_key: str) -> Optional[int]:
    """
    Turns an ImageNet key into an integer, so that the 10.7M keys of the Tencent lists fit in a sorted array of 86 MB
    shared by the shard workers through a memory map, instead of a Python set copied into every worker.

    Args:
        imagenet_key (str): The key of an image, e.g. 'n04310904_8388'.

    Returns:
        Optional[int]: The WordNet number times 10^10 plus the image number (e.g. 4310904 * 10^10 + 8388), or None when
            the key does not have this form.
    """
    wordnet_identifier, _, image_number = imagenet_key.partition("_")
    if len(wordnet_identifier) != 9 or wordnet_identifier[0] != "n" or not wordnet_identifier[1:].isdigit() or \
            not image_number.isdigit() or len(image_number) > 10:
        return None
    return int(wordnet_identifier[1:]) * 10 ** 10 + int(image_number)


def get_flickr_photo_number(url: str) -> str:
    """
    Gives the Flickr photo number of an image URL, the part of the file name before the first underscore. The Tencent
    URLs and the Open Images metadata point to the same photos through different Flickr hosts, so the photo number is
    the key joining them.

    Args:
        url (str): A Flickr image URL, e.g. 'https://c4.staticflickr.com/8/7239/6997334729_e5fb3938b1_o.jpg'.

    Returns:
        str: The photo number, e.g. '6997334729'.
    """
    return url.rsplit("/", maxsplit=1)[-1].split("_")[0]


def get_free_gigabytes(folder: Path) -> float:
    """
    Gives the free space of the disk of a folder, checked before every task of the downloads so that they stop before
    filling the disk.

    Args:
        folder (Path): A folder of the disk.

    Returns:
        float: The free space in gigabytes.
    """
    return shutil.disk_usage(path=folder).free / 1e9


def rotate_image(image: np.ndarray, rotation_degrees: int) -> np.ndarray:
    """
    Turns an Open Images image to the orientation intended by its Flickr author, given by the 'Rotation' column of the
    metadata as the counterclockwise rotation to apply (0, 90, 180 or 270).

    Args:
        image (np.ndarray): The BGR image as stored.
        rotation_degrees (int): The counterclockwise rotation in degrees.

    Returns:
        np.ndarray: The upright image.
    """
    # Counterclockwise rotations in OpenCV rotation codes, 270 degrees counterclockwise being 90 degrees clockwise
    rotation_codes = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}
    if rotation_degrees not in rotation_codes:
        return image
    return cv2.rotate(src=image, rotateCode=rotation_codes[rotation_degrees])


def resize_image(image: np.ndarray, shorter_side: int) -> np.ndarray:
    """
    Shrinks an image so that its shorter side is 'shorter_side', keeping its aspect ratio, which bounds the size of the
    stored dataset and the decoding time of the training loader. Smaller images are kept as they are, never enlarged.

    Args:
        image (np.ndarray): The image.
        shorter_side (int): The maximum length of the shorter side in pixels.

    Returns:
        np.ndarray: The resized image.
    """
    height, width = image.shape[:2]
    scale = shorter_side / min(height, width)
    if scale >= 1:
        return image
    return cv2.resize(src=image, dsize=(max(round(width * scale), 1), max(round(height * scale), 1)),
                      interpolation=cv2.INTER_AREA)


def compute_perceptual_hash(image: np.ndarray) -> Optional[str]:
    """
    Computes the perceptual hash of an image, which finds the same picture stored twice (ImageNet keeps an image under
    every class it shows, Open Images and ImageNet both come from Flickr) and the copies of the ImageNet-1k test images.
    Unlike a hash of the file bytes, it survives the re-encoding and the resizing of the image: the grayscale image is
    shrunk to a 9 x 9 thumbnail, and every bit tells whether a pixel is brighter than its left neighbour (64 bits) or
    its upper neighbour (64 bits).

    Args:
        image (np.ndarray): The BGR image.

    Returns:
        Optional[str]: The 128 bits as 32 hexadecimal characters, or None for a nearly uniform image, whose hash would
            match unrelated images.
    """
    # Grayscale thumbnail, averaged over the whole image by the area interpolation
    grayscale_image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    thumbnail = cv2.resize(src=grayscale_image, dsize=(PERCEPTUAL_HASH_THUMBNAIL_SIDE, PERCEPTUAL_HASH_THUMBNAIL_SIDE),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
    if thumbnail.std() < MINIMUM_THUMBNAIL_STANDARD_DEVIATION:
        return None

    # Brightness differences between horizontal neighbours, then between vertical neighbours
    horizontal_bits = thumbnail[:-1, 1:] > thumbnail[:-1, :-1]
    vertical_bits = thumbnail[1:, :-1] > thumbnail[:-1, :-1]
    hash_bits = np.concatenate([horizontal_bits.flatten(), vertical_bits.flatten()])
    return np.packbits(hash_bits).tobytes().hex()


def prepare_image(image_bytes: bytes, shorter_side: int, jpeg_quality: int,
                  rotation_degrees: int) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Turns a downloaded image into the stored one: decoded without applying any EXIF orientation (Open Images gives the
    rotation in its metadata instead), rotated, resized and re-encoded as JPEG, with its perceptual hash computed on
    the stored pixels.

    Args:
        image_bytes (bytes): The encoded image.
        shorter_side (int): The maximum length of the shorter side in pixels.
        jpeg_quality (int): The JPEG quality of the stored image (0 to 100).
        rotation_degrees (int): The counterclockwise rotation to apply, 0 for the ImageNet images.

    Returns:
        Tuple[Optional[bytes], Optional[str]]: The JPEG bytes and the perceptual hash, or (None, None) when the image
            cannot be decoded.
    """
    image = cv2.imdecode(buf=np.frombuffer(buffer=image_bytes, dtype=np.uint8),
                         flags=cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        return None, None
    image = resize_image(image=rotate_image(image=image, rotation_degrees=rotation_degrees), shorter_side=shorter_side)
    _, encoded_image = cv2.imencode(ext=".jpg", img=image, params=[cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return encoded_image.tobytes(), compute_perceptual_hash(image=image)


def write_records(records_path: Path, records: List[List[Optional[str]]], missing_identifiers: List[str]) -> None:
    """
    Saves the images written by one task (shard or batch) with their perceptual hashes, read back by the manifest step.
    The file is written last and renamed into place, so it also marks the task as done for the next run.

    Args:
        records_path (Path): The records file of the task.
        records (List[List[Optional[str]]]): The identifier and the perceptual hash of every written image.
        missing_identifiers (List[str]): The identifiers of the images the task could not get.
    """
    temporary_path = records_path.with_name(f"{records_path.name}.part")
    temporary_path.write_text(data=json.dumps(obj={"records": records, "missing": missing_identifiers}))
    temporary_path.rename(target=records_path)


def run_tasks_with_free_space_check(task_function: Callable[..., int], task_arguments: List[Dict[str, Any]],
                                    disk_folder: Path, minimum_free_gigabytes: float, number_workers: int,
                                    description: str) -> int:
    """
    Runs download tasks in worker processes, queuing a new task only while the disk keeps 'minimum_free_gigabytes'
    free, so that the ImageNet and Open Images downloads stop cleanly instead of filling the disk. When the space runs
    out, the running tasks finish, and the next run resumes from the tasks without records. The workers are started
    fresh (spawn) instead of forked, since the Hugging Face and S3 clients run threads.

    Args:
        task_function (Callable[..., int]): The task, a module-level function returning its number of written images.
        task_arguments (List[Dict[str, Any]]): The keyword arguments of every task.
        disk_folder (Path): A folder of the disk receiving the images.
        minimum_free_gigabytes (float): The free space below which no new task is started.
        number_workers (int): The number of worker processes.
        description (str): The label of the progress bar.

    Returns:
        int: The number of images written by all the tasks.

    Raises:
        RuntimeError: If the free space fell below the minimum before every task was started.
    """
    number_written_images = 0
    next_task_index = 0
    running_futures = set()
    low_free_space = False
    with ProcessPoolExecutor(max_workers=number_workers,
                             mp_context=multiprocessing.get_context(method="spawn")) as executor, \
            tqdm(total=len(task_arguments), desc=description) as progress_bar:
        while next_task_index < len(task_arguments) or running_futures:
            # Queue new tasks while the pool has room and the disk keeps enough free space
            while not low_free_space and next_task_index < len(task_arguments) and \
                    len(running_futures) < number_workers * QUEUED_TASKS_PER_WORKER:
                if get_free_gigabytes(folder=disk_folder) < minimum_free_gigabytes:
                    low_free_space = True
                    break
                running_futures.add(executor.submit(task_function, **task_arguments[next_task_index]))
                next_task_index += 1
            if not running_futures:
                break

            # Count the images of the finished tasks, a failed task stopping the step with its error
            completed_futures, running_futures = wait(fs=running_futures, return_when=FIRST_COMPLETED)
            for future in completed_futures:
                number_written_images += future.result()
                progress_bar.update(n=1)

    if low_free_space:
        raise RuntimeError(f"Less than {minimum_free_gigabytes} GB free on {format_path_link(path=disk_folder)} : "
                           f"{len(task_arguments) - next_task_index} tasks of '{description}' were not started. Free "
                           f"some space, then run the script again to resume.")
    return number_written_images


def write_imagenet_wanted_keys(annotations_folder: Path, wanted_keys_path: Path) -> int:
    """
    Saves the encoded keys of every ImageNet image of the Tencent lists as a sorted array, read by every shard worker
    to keep only these images out of the 14.1M images of ImageNet-22k.

    Args:
        annotations_folder (Path): The folder of the Tencent lists.
        wanted_keys_path (Path): The '.npy' file of the sorted keys, skipped when it already exists.

    Returns:
        int: The number of wanted keys.
    """
    if wanted_keys_path.exists():
        return len(np.load(file=wanted_keys_path, mmap_mode="r"))

    # Encoded key of the image of every line of both ImageNet lists
    encoded_keys = []
    for list_file_name in TENCENT_SPLIT_LISTS["imagenet"].values():
        with open(file=annotations_folder / list_file_name, mode="r") as list_file:
            for line in list_file:
                encoded_key = encode_imagenet_key(imagenet_key=get_imagenet_key(image_identifier=line.split()[0]))
                if encoded_key is not None:
                    encoded_keys.append(encoded_key)

    wanted_keys = np.unique(np.array(encoded_keys, dtype=np.int64))
    np.save(file=wanted_keys_path, arr=wanted_keys)
    return len(wanted_keys)


def process_imagenet_shard(shard_file_name: str, downloads_folder: Path, wanted_keys_path: Path, images_folder: Path,
                           records_path: Path, shorter_side: int, jpeg_quality: int) -> int:
    """
    Downloads one ImageNet-22k shard, writes its images listed by Tencent into 'imagenet/<WordNet identifier>/' of the
    images folder, resized and re-encoded, saves their records, then deletes the shard. It runs in a worker process for
    every shard without records, so the disk only holds the shards being processed.

    Args:
        shard_file_name (str): The tar shard in the repository.
        downloads_folder (Path): The folder receiving the shard while it is processed.
        wanted_keys_path (Path): The sorted encoded keys of the Tencent ImageNet images.
        images_folder (Path): The root folder of the stored images.
        records_path (Path): The records file of the shard, written once all its images are.
        shorter_side (int): The maximum length of the shorter side of the stored images.
        jpeg_quality (int): The JPEG quality of the stored images.

    Returns:
        int: The number of images written.
    """
    wanted_keys = np.load(file=wanted_keys_path, mmap_mode="r")
    shard_path = Path(hf_hub_download(repo_id=IMAGENET_REPOSITORY, filename=shard_file_name, repo_type="dataset",
                                      local_dir=downloads_folder))

    # Read the JPEG members of the shard, keeping the wanted keys
    records, undecodable_keys = [], []
    with tarfile.open(name=shard_path, mode="r") as shard:
        for member in shard:
            imagenet_key, _, extension = Path(member.name).name.rpartition(".")
            if extension != "jpg":
                continue
            encoded_key = encode_imagenet_key(imagenet_key=imagenet_key)
            if encoded_key is None:
                continue
            key_position = np.searchsorted(a=wanted_keys, v=encoded_key)
            if key_position >= len(wanted_keys) or wanted_keys[key_position] != encoded_key:
                continue

            # Store the resized image in the folder of its WordNet identifier
            image_bytes, perceptual_hash = prepare_image(image_bytes=shard.extractfile(member=member).read(),
                                                         shorter_side=shorter_side, jpeg_quality=jpeg_quality,
                                                         rotation_degrees=0)
            if image_bytes is None:
                undecodable_keys.append(imagenet_key)
                continue
            image_path = images_folder / "imagenet" / imagenet_key.split("_")[0] / f"{imagenet_key}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(data=image_bytes)
            records.append([imagenet_key, perceptual_hash])

    write_records(records_path=records_path, records=records, missing_identifiers=undecodable_keys)
    shard_path.unlink()
    return len(records)


def download_imagenet_images(dataset_path: Path, shorter_side: int, jpeg_quality: int, minimum_free_gigabytes: float,
                             number_workers: int) -> None:
    """
    Stores the 10.7M ImageNet images of the Tencent lists, read from the 4,608 shards of ImageNet-22k (1.3 TB, about
    0.29 GB each), one worker task per shard without records.

    Args:
        dataset_path (Path): The folder of the dataset.
        shorter_side (int): The maximum length of the shorter side of the stored images.
        jpeg_quality (int): The JPEG quality of the stored images.
        minimum_free_gigabytes (float): The free space below which no new shard is started.
        number_workers (int): The number of worker processes.
    """
    progress_folder = dataset_path / "progress"
    records_folder = progress_folder / "imagenet_shards"
    records_folder.mkdir(parents=True, exist_ok=True)

    # Sorted keys of the wanted images, shared by the workers
    wanted_keys_path = progress_folder / "imagenet_wanted_keys.npy"
    number_wanted_keys = write_imagenet_wanted_keys(annotations_folder=dataset_path / "annotations",
                                                    wanted_keys_path=wanted_keys_path)

    # Shards of both splits of the repository without records
    shard_file_names = sorted(file_name for file_name in HfApi().list_repo_files(repo_id=IMAGENET_REPOSITORY,
                                                                                 repo_type="dataset")
                              if file_name.startswith(IMAGENET_SHARD_PREFIXES) and file_name.endswith(".tar"))
    task_arguments = [{"shard_file_name": file_name, "downloads_folder": dataset_path / "shard_downloads",
                       "wanted_keys_path": wanted_keys_path, "images_folder": dataset_path / "images",
                       "records_path": records_folder / f"{Path(file_name).stem}.json", "shorter_side": shorter_side,
                       "jpeg_quality": jpeg_quality}
                      for file_name in shard_file_names
                      if not (records_folder / f"{Path(file_name).stem}.json").exists()]
    print_color(color="blue", output=f"ImageNet : {number_wanted_keys} Tencent images, {len(task_arguments)} of "
                                     f"{len(shard_file_names)} shards left : "
                                     f"{format_path_link(path=dataset_path / 'images')}")

    number_written_images = run_tasks_with_free_space_check(task_function=process_imagenet_shard,
                                                            task_arguments=task_arguments, disk_folder=dataset_path,
                                                            minimum_free_gigabytes=minimum_free_gigabytes,
                                                            number_workers=number_workers,
                                                            description="ImageNet Shards")
    print_color(color="green", output=f"{number_written_images} ImageNet images written", indent=1)


def match_open_images(annotations_folder: Path, matches_path: Path) -> int:
    """
    Finds the Open Images identifier, subset and rotation of every Tencent Open Images URL by joining its Flickr photo
    number with the URLs of the Open Images metadata, since the S3 bucket stores the images by identifier. The URLs of
    photos no longer in Open Images get no match, and are listed in the report of the manifest step.

    Args:
        annotations_folder (Path): The folder of the Tencent lists and of the Open Images metadata.
        matches_path (Path): The CSV of the matches (photo number, subset, image identifier, rotation), skipped when it
            already exists.

    Returns:
        int: The number of matched photos.
    """
    if matches_path.exists():
        with open(file=matches_path, mode="r", newline="") as matches_file:
            return sum(1 for _ in matches_file) - 1

    # Photo numbers of both Open Images lists
    wanted_photo_numbers = set()
    for list_file_name in TENCENT_SPLIT_LISTS["openimages"].values():
        with open(file=annotations_folder / list_file_name, mode="r") as list_file:
            for line in list_file:
                wanted_photo_numbers.add(get_flickr_photo_number(url=line.split(maxsplit=1)[0]))

    # Read the metadata of every subset, keeping the wanted photos with their rotation (0 when unknown)
    number_matches = 0
    temporary_path = matches_path.with_name(f"{matches_path.name}.part")
    with open(file=temporary_path, mode="w", newline="") as matches_file:
        matches_writer = csv.writer(matches_file)
        matches_writer.writerow(["photo_number", "subset", "image_identifier", "rotation"])
        for subset_name in OPEN_IMAGES_METADATA_URLS:
            metadata_path = annotations_folder / f"open_images_{subset_name}_metadata.csv"
            with open(file=metadata_path, mode="r", newline="") as metadata_file:
                for row in tqdm(csv.DictReader(metadata_file), desc=f"Matching Open Images {subset_name}"):
                    photo_number = get_flickr_photo_number(url=row["OriginalURL"])
                    if photo_number not in wanted_photo_numbers:
                        continue
                    rotation = int(float(row["Rotation"])) if row["Rotation"] not in ("", "nan") else 0
                    matches_writer.writerow([photo_number, row["Subset"], row["ImageID"], rotation])
                    wanted_photo_numbers.discard(photo_number)
                    number_matches += 1
    temporary_path.rename(target=matches_path)
    print_color(color="green", output=f"{number_matches} Open Images photos matched, {len(wanted_photo_numbers)} no "
                                      f"longer in Open Images", indent=1)
    return number_matches


def read_open_images_matches(matches_path: Path) -> Dict[str, Tuple[str, str, int]]:
    """
    Reads the matches written by match_open_images, to build the download batches and to name the Open Images images
    of the manifests.

    Args:
        matches_path (Path): The CSV of the matches.

    Returns:
        Dict[str, Tuple[str, str, int]]: The subset, image identifier and rotation of every matched photo number.
    """
    with open(file=matches_path, mode="r", newline="") as matches_file:
        return {row["photo_number"]: (row["subset"], row["image_identifier"], int(row["rotation"]))
                for row in csv.DictReader(matches_file)}


def get_open_images_file_name(image_identifier: str) -> str:
    """
    Gives the path of an Open Images image in the images folder, spread over 256 folders by the first two characters
    of its identifier so that no folder holds millions of files.

    Args:
        image_identifier (str): The Open Images identifier, 16 hexadecimal characters.

    Returns:
        str: The path relative to the images folder, e.g. 'openimages/4f/4fa8054781a4c382.jpg'.
    """
    return f"openimages/{image_identifier[:2]}/{image_identifier}.jpg"


def create_s3_client(max_pool_connections: int) -> BaseClient:
    """
    Creates the unsigned client of the public Open Images bucket, retrying the failed requests, in every Open Images
    worker process since a client can't be passed between processes.

    Args:
        max_pool_connections (int): The number of connections kept open, at least the number of threads using it.

    Returns:
        BaseClient: The S3 client.
    """
    s3_configuration = Config(signature_version=UNSIGNED, max_pool_connections=max_pool_connections,
                              retries={"max_attempts": S3_MAXIMUM_ATTEMPTS, "mode": "adaptive"})
    return boto3.client("s3", config=s3_configuration)


def fetch_open_images_object(s3_client: BaseClient, object_key: str) -> Optional[bytes]:
    """
    Downloads one image of the Open Images bucket, called by the download threads of a batch.

    Args:
        s3_client (BaseClient): The unsigned S3 client of the worker.
        object_key (str): The key of the image, '<subset>/<image identifier>.jpg'.

    Returns:
        Optional[bytes]: The image bytes, or None when the bucket has no such image.

    Raises:
        ClientError: For any other S3 error after the retries, which fails the batch so that the next run retries it.
    """
    try:
        return s3_client.get_object(Bucket=OPEN_IMAGES_BUCKET, Key=object_key)["Body"].read()
    except ClientError as error:
        if error.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def read_open_images_image(image_entry: Tuple[str, str, int], local_open_images_folder: Optional[Path],
                           s3_client: BaseClient) -> Optional[bytes]:
    """
    Gets the bytes of one Open Images image, from the local Open Images copy when it holds the image, otherwise from
    the S3 bucket, so that the images already on disk are not downloaded again.

    Args:
        image_entry (Tuple[str, str, int]): The subset, image identifier and rotation of the image.
        local_open_images_folder (Optional[Path]): The local copy ('<subset>/data/<image identifier>.jpg'), or None.
        s3_client (BaseClient): The unsigned S3 client of the worker.

    Returns:
        Optional[bytes]: The image bytes, or None when the bucket has no such image.
    """
    subset_name, image_identifier, _ = image_entry
    if local_open_images_folder is not None:
        local_path = local_open_images_folder / subset_name / "data" / f"{image_identifier}.jpg"
        if local_path.exists():
            return local_path.read_bytes()
    return fetch_open_images_object(s3_client=s3_client, object_key=f"{subset_name}/{image_identifier}.jpg")


def store_open_images_member(image_bytes: bytes, image_identifier: str, rotation_degrees: int, images_folder: Path,
                             shorter_side: int, jpeg_quality: int) -> Optional[List[Optional[str]]]:
    """
    Stores one Open Images image read from the bucket or an archive, rotated upright, resized and re-encoded, in the
    folder of the first characters of its identifier.

    Args:
        image_bytes (bytes): The encoded image.
        image_identifier (str): The Open Images identifier.
        rotation_degrees (int): The counterclockwise rotation to apply.
        images_folder (Path): The root folder of the stored images.
        shorter_side (int): The maximum length of the shorter side of the stored images.
        jpeg_quality (int): The JPEG quality of the stored images.

    Returns:
        Optional[List[Optional[str]]]: The record of the image (identifier and perceptual hash), or None when the image
            cannot be decoded.
    """
    stored_bytes, perceptual_hash = prepare_image(image_bytes=image_bytes, shorter_side=shorter_side,
                                                  jpeg_quality=jpeg_quality, rotation_degrees=rotation_degrees)
    if stored_bytes is None:
        return None
    image_path = images_folder / get_open_images_file_name(image_identifier=image_identifier)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(data=stored_bytes)
    return [image_identifier, perceptual_hash]


def process_open_images_batch(image_entries: List[Tuple[str, str, int]], local_open_images_folder: Optional[Path],
                              images_folder: Path, records_path: Path, shorter_side: int, jpeg_quality: int,
                              number_download_threads: int) -> int:
    """
    Stores one batch of single-file Open Images images (validation and test subsets), rotated upright, resized and
    re-encoded, and saves their records. It runs in a worker process for every batch without records, the images being
    fetched by threads since the S3 requests mostly wait on the network.

    Args:
        image_entries (List[Tuple[str, str, int]]): The subset, image identifier and rotation of every image.
        local_open_images_folder (Optional[Path]): The local Open Images copy read before the bucket, or None.
        images_folder (Path): The root folder of the stored images.
        records_path (Path): The records file of the batch, written once all its images are.
        shorter_side (int): The maximum length of the shorter side of the stored images.
        jpeg_quality (int): The JPEG quality of the stored images.
        number_download_threads (int): The number of images fetched at once.

    Returns:
        int: The number of images written.
    """
    s3_client = create_s3_client(max_pool_connections=number_download_threads)
    records, missing_identifiers = [], []
    with ThreadPoolExecutor(max_workers=number_download_threads) as executor:
        futures = {executor.submit(read_open_images_image, image_entry=image_entry,
                                   local_open_images_folder=local_open_images_folder, s3_client=s3_client): image_entry
                   for image_entry in image_entries}
        for future in as_completed(fs=futures):
            _, image_identifier, rotation_degrees = futures[future]
            downloaded_bytes = future.result()
            if downloaded_bytes is None:
                missing_identifiers.append(image_identifier)
                continue

            # Store the upright resized image, an undecodable image being listed as missing
            record = store_open_images_member(image_bytes=downloaded_bytes, image_identifier=image_identifier,
                                              rotation_degrees=rotation_degrees, images_folder=images_folder,
                                              shorter_side=shorter_side, jpeg_quality=jpeg_quality)
            if record is None:
                missing_identifiers.append(image_identifier)
                continue
            records.append(record)

    write_records(records_path=records_path, records=records, missing_identifiers=missing_identifiers)
    return len(records)


def process_open_images_archive(archive_key: str, wanted_rotations: Dict[str, int], images_folder: Path,
                                records_path: Path, shorter_side: int, jpeg_quality: int, disk_folder: Path,
                                minimum_free_gigabytes: float) -> int:
    """
    Streams one Open Images training archive from the bucket and stores its images listed by Tencent, without saving
    the archive. It runs in a worker process for every archive without records, for hours, so it appends every stored
    image to a progress file: after an interruption or a lack of disk space, the next run streams the archive again
    and only skips the images already stored. The progress file becomes the records file once the archive is done.

    Args:
        archive_key (str): The key of the archive in the bucket, e.g. 'tar/train_0.tar.gz'.
        wanted_rotations (Dict[str, int]): The rotation of every wanted image of the archive, keyed by identifier.
        images_folder (Path): The root folder of the stored images.
        records_path (Path): The records file of the archive, written once all its images are.
        shorter_side (int): The maximum length of the shorter side of the stored images.
        jpeg_quality (int): The JPEG quality of the stored images.
        disk_folder (Path): A folder of the disk receiving the images.
        minimum_free_gigabytes (float): The free space below which the archive stops.

    Returns:
        int: The number of images stored by this call.

    Raises:
        RuntimeError: If the free space falls below the minimum, the progress being kept for the next run.
    """
    # Images already stored by a previous run, one JSON record per line
    progress_path = records_path.with_name(f"{records_path.stem}.progress.jsonl")
    records, undecodable_identifiers = [], []
    if progress_path.exists():
        records = [json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()]
    stored_identifiers = {record[0] for record in records}
    number_stored_images = 0

    # Stream the archive member by member, keeping the wanted images not stored yet
    archive_stream = create_s3_client(max_pool_connections=1).get_object(Bucket=OPEN_IMAGES_BUCKET,
                                                                         Key=archive_key)["Body"]
    with tarfile.open(fileobj=archive_stream, mode="r|gz") as archive, \
            open(file=progress_path, mode="a") as progress_file:
        for member in archive:
            image_identifier = Path(member.name).stem
            if not member.isfile() or image_identifier not in wanted_rotations or \
                    image_identifier in stored_identifiers:
                continue
            record = store_open_images_member(image_bytes=archive.extractfile(member=member).read(),
                                              image_identifier=image_identifier,
                                              rotation_degrees=wanted_rotations[image_identifier],
                                              images_folder=images_folder, shorter_side=shorter_side,
                                              jpeg_quality=jpeg_quality)
            if record is None:
                undecodable_identifiers.append(image_identifier)
                continue
            progress_file.write(json.dumps(obj=record) + "\n")
            records.append(record)
            stored_identifiers.add(image_identifier)
            number_stored_images += 1

            # Save the progress regularly, and stop before filling the disk
            if number_stored_images % ARCHIVE_CHECK_INTERVAL == 0:
                progress_file.flush()
                if get_free_gigabytes(folder=disk_folder) < minimum_free_gigabytes:
                    raise RuntimeError(f"Less than {minimum_free_gigabytes} GB free on "
                                       f"{format_path_link(path=disk_folder)} : "
                                       f"{archive_key} stopped, its progress is kept. Free some space, then run the "
                                       f"script again to resume.")

    # Wanted images absent from the archive or undecodable, then the records replacing the progress file
    missing_identifiers = undecodable_identifiers + sorted(set(wanted_rotations) - stored_identifiers -
                                                           set(undecodable_identifiers))
    write_records(records_path=records_path, records=records, missing_identifiers=missing_identifiers)
    progress_path.unlink()
    return number_stored_images


def get_entry_identifier(image_entry: Tuple[str, str, int]) -> str:
    """
    Gives the image identifier of an Open Images entry, the sort key fixing the content of every download batch.

    Args:
        image_entry (Tuple[str, str, int]): The subset, image identifier and rotation of an image.

    Returns:
        str: The image identifier.
    """
    return image_entry[1]


def download_open_images(dataset_path: Path, local_open_images_folder: Optional[Path], shorter_side: int,
                         jpeg_quality: int, minimum_free_gigabytes: float, number_workers: int,
                         number_download_threads: int) -> None:
    """
    Stores the Open Images images of the Tencent lists found in the Open Images metadata. The training images are
    streamed from the 16 archives of the training subset (548 GB), one worker task per archive without records. The
    validation and test images are downloaded as single files by batches of OPEN_IMAGES_BATCH_SIZE images, one worker
    task per batch without records, cut from the images sorted by identifier so that they are the same on every run.

    Args:
        dataset_path (Path): The folder of the dataset.
        local_open_images_folder (Optional[Path]): The local Open Images copy read before the bucket for the single
            files, or None.
        shorter_side (int): The maximum length of the shorter side of the stored images.
        jpeg_quality (int): The JPEG quality of the stored images.
        minimum_free_gigabytes (float): The free space below which no new task is started.
        number_workers (int): The number of worker processes.
        number_download_threads (int): The number of single files fetched at once by every batch worker.
    """
    progress_folder = dataset_path / "progress"
    batches_folder = progress_folder / "open_images_batches"
    archives_folder = progress_folder / "open_images_archives"
    for records_folder in (batches_folder, archives_folder):
        records_folder.mkdir(parents=True, exist_ok=True)

    # Identifier of every matched Tencent URL, in a fixed order, split between the archives and the single files
    matches_path = progress_folder / "open_images_matches.csv"
    match_open_images(annotations_folder=dataset_path / "annotations", matches_path=matches_path)
    image_entries = sorted(set(read_open_images_matches(matches_path=matches_path).values()), key=get_entry_identifier)
    single_file_entries = [image_entry for image_entry in image_entries
                           if image_entry[0] != OPEN_IMAGES_ARCHIVE_SUBSET]
    wanted_rotations_by_archive = {}
    for subset_name, image_identifier, rotation_degrees in image_entries:
        if subset_name == OPEN_IMAGES_ARCHIVE_SUBSET:
            archive_key = OPEN_IMAGES_ARCHIVE_KEY_FORMAT.format(image_identifier[0])
            wanted_rotations_by_archive.setdefault(archive_key, {})[image_identifier] = rotation_degrees

    # Batches of single files without records
    batch_arguments = []
    for batch_index, first_entry_index in enumerate(range(0, len(single_file_entries), OPEN_IMAGES_BATCH_SIZE)):
        records_path = batches_folder / f"batch_{batch_index:05d}.json"
        if not records_path.exists():
            batch_arguments.append({"image_entries": single_file_entries[first_entry_index:
                                                                         first_entry_index + OPEN_IMAGES_BATCH_SIZE],
                                    "local_open_images_folder": local_open_images_folder,
                                    "images_folder": dataset_path / "images", "records_path": records_path,
                                    "shorter_side": shorter_side, "jpeg_quality": jpeg_quality,
                                    "number_download_threads": number_download_threads})

    # Archives without records
    archive_arguments = []
    for archive_key, wanted_rotations in sorted(wanted_rotations_by_archive.items()):
        records_path = archives_folder / f"{Path(archive_key).name.split('.')[0]}.json"
        if not records_path.exists():
            archive_arguments.append({"archive_key": archive_key, "wanted_rotations": wanted_rotations,
                                      "images_folder": dataset_path / "images", "records_path": records_path,
                                      "shorter_side": shorter_side, "jpeg_quality": jpeg_quality,
                                      "disk_folder": dataset_path, "minimum_free_gigabytes": minimum_free_gigabytes})
    print_color(color="blue", output=f"Open Images : {len(image_entries)} matched images, {len(batch_arguments)} "
                                     f"batches and {len(archive_arguments)} archives left : "
                                     f"{format_path_link(path=dataset_path / 'images')}")

    # Single files first (minutes), then the archives (hours)
    number_written_images = run_tasks_with_free_space_check(task_function=process_open_images_batch,
                                                            task_arguments=batch_arguments, disk_folder=dataset_path,
                                                            minimum_free_gigabytes=minimum_free_gigabytes,
                                                            number_workers=number_workers,
                                                            description="Open Images Batches")
    number_written_images += run_tasks_with_free_space_check(task_function=process_open_images_archive,
                                                             task_arguments=archive_arguments,
                                                             disk_folder=dataset_path,
                                                             minimum_free_gigabytes=minimum_free_gigabytes,
                                                             number_workers=number_workers,
                                                             description="Open Images Archives")
    print_color(color="green", output=f"{number_written_images} Open Images images written", indent=1)


def compute_image_file_hash(image_path: Path, shorter_side: int) -> Optional[str]:
    """
    Computes the perceptual hash of an ImageNet-1k test image after the resizing applied to the stored images, so that
    the copies of the test images found among the ML-Images images can be left out of the pre-training. It runs in a
    worker process for every test image.

    Args:
        image_path (Path): The test image.
        shorter_side (int): The maximum length of the shorter side of the stored images.

    Returns:
        Optional[str]: The perceptual hash, or None for an undecodable or nearly uniform image.
    """
    image = cv2.imread(filename=str(image_path), flags=cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        return None
    return compute_perceptual_hash(image=resize_image(image=image, shorter_side=shorter_side))


def hash_excluded_images(excluded_images_folder: Path, hashes_path: Path, shorter_side: int,
                         number_workers: int) -> List[int]:
    """
    Computes the perceptual hashes of the ImageNet-1k test images (one folder per class), the images whose copies are
    left out of the ML-Images splits, so that the backbones pre-trained here can still be evaluated on them fairly.

    Args:
        excluded_images_folder (Path): The folder of the class folders of the excluded images.
        hashes_path (Path): The JSON file caching the hashes, skipped when it already exists.
        shorter_side (int): The maximum length of the shorter side of the stored images.
        number_workers (int): The number of worker processes.

    Returns:
        List[int]: The hash of every excluded image that has one, as an integer.
    """
    if not hashes_path.exists():
        image_paths = [image_path for class_folder in sorted(excluded_images_folder.iterdir()) if class_folder.is_dir()
                       for image_path in get_image_paths(folder=class_folder)]
        with ProcessPoolExecutor(max_workers=number_workers,
                                 mp_context=multiprocessing.get_context(method="spawn")) as executor:
            futures = [executor.submit(compute_image_file_hash, image_path=image_path, shorter_side=shorter_side)
                       for image_path in image_paths]
            hashes = [future.result() for future in tqdm(futures, desc="Hashing Excluded Images")]
        hashes_path.write_text(data=json.dumps(obj=[perceptual_hash for perceptual_hash in hashes if perceptual_hash]))
    return [int(perceptual_hash, 16) for perceptual_hash in json.loads(hashes_path.read_text())]


def get_hash_chunks(hash_value: int) -> List[int]:
    """
    Splits a 128-bit perceptual hash into EXCLUSION_MAXIMUM_DISTANCE + 1 chunks, the keys of the index finding the near
    copies of the excluded images: two hashes differing by at most EXCLUSION_MAXIMUM_DISTANCE bits share at least one
    identical chunk.

    Args:
        hash_value (int): The hash as an integer.

    Returns:
        List[int]: The chunks, from the lowest bits.
    """
    number_chunks = EXCLUSION_MAXIMUM_DISTANCE + 1
    chunk_bits = PERCEPTUAL_HASH_BITS // number_chunks
    chunk_mask = (1 << chunk_bits) - 1
    return [(hash_value >> (chunk_index * chunk_bits)) & chunk_mask for chunk_index in range(number_chunks)]


def build_hash_chunk_index(hash_values: List[int]) -> List[Dict[int, List[int]]]:
    """
    Indexes the hashes of the excluded images by chunk, so that every ML-Images image is compared with the few excluded
    images sharing one of its chunks instead of all of them.

    Args:
        hash_values (List[int]): The hashes of the excluded images.

    Returns:
        List[Dict[int, List[int]]]: For every chunk position, the hashes having each chunk value.
    """
    chunk_index = [{} for _ in range(EXCLUSION_MAXIMUM_DISTANCE + 1)]
    for hash_value in hash_values:
        for chunk_position, chunk_value in enumerate(get_hash_chunks(hash_value=hash_value)):
            chunk_index[chunk_position].setdefault(chunk_value, []).append(hash_value)
    return chunk_index


def is_near_copy(hash_value: int, chunk_index: List[Dict[int, List[int]]]) -> bool:
    """
    Tells whether an image is a copy of an excluded image: its hash differs by at most EXCLUSION_MAXIMUM_DISTANCE bits
    from the hash of an excluded image sharing one of its chunks.

    Args:
        hash_value (int): The hash of the image.
        chunk_index (List[Dict[int, List[int]]]): The excluded hashes indexed by build_hash_chunk_index.

    Returns:
        bool: Whether the image is a copy of an excluded image.
    """
    for chunk_position, chunk_value in enumerate(get_hash_chunks(hash_value=hash_value)):
        for excluded_hash in chunk_index[chunk_position].get(chunk_value, []):
            if (hash_value ^ excluded_hash).bit_count() <= EXCLUSION_MAXIMUM_DISTANCE:
                return True
    return False


def read_records(records_folder: Path) -> Tuple[Dict[str, Optional[str]], Set[str]]:
    """
    Gathers the records of every finished task of a source, giving the stored images and their perceptual hashes to
    the manifest step.

    Args:
        records_folder (Path): The folder of the records files of the source.

    Returns:
        Tuple[Dict[str, Optional[str]], Set[str]]: The perceptual hash of every stored image keyed by identifier, and
            the identifiers of the images the tasks could not get.
    """
    stored_images, missing_identifiers = {}, set()
    for records_path in sorted(records_folder.glob(pattern="*.json")):
        task_records = json.loads(records_path.read_text())
        stored_images.update({identifier: perceptual_hash for identifier, perceptual_hash in task_records["records"]})
        missing_identifiers.update(task_records["missing"])
    return stored_images, missing_identifiers


def collect_samples(dataset_path: Path) -> List[List[Any]]:
    """
    Lists the samples of the Tencent lists whose image is stored, with their split, file name, labels and perceptual
    hash, and writes the reports of the images left out because they could not be found or downloaded. The labels are
    kept as the raw text of the list, parsed only when written, which keeps the 17.6M samples in memory small.

    Args:
        dataset_path (Path): The folder of the dataset.

    Returns:
        List[List[Any]]: The split name, file name, raw label text and perceptual hash (an integer, or None) of every
            sample.
    """
    annotations_folder = dataset_path / "annotations"
    reports_folder = dataset_path / "reports"
    reports_folder.mkdir(parents=True, exist_ok=True)

    # Stored images of both sources, and the photo number matches of Open Images
    imagenet_images, _ = read_records(records_folder=dataset_path / "progress" / "imagenet_shards")
    open_images_images, _ = read_records(records_folder=dataset_path / "progress" / "open_images_batches")
    open_images_images.update(read_records(records_folder=dataset_path / "progress" / "open_images_archives")[0])
    open_images_matches = read_open_images_matches(matches_path=dataset_path / "progress" / "open_images_matches.csv")

    samples = []
    for source_name, split_lists in TENCENT_SPLIT_LISTS.items():
        for split_name, list_file_name in split_lists.items():
            unavailable_images = []
            with open(file=annotations_folder / list_file_name, mode="r") as list_file:
                for line in tqdm(list_file, desc=f"Reading {list_file_name}"):
                    image, *label_parts = line.split(maxsplit=1)
                    label_text = label_parts[0].strip() if label_parts else ""

                    # Identifier of the stored image, from the key for ImageNet and from the matches for Open Images
                    if source_name == "imagenet":
                        identifier = get_imagenet_key(image_identifier=image)
                        file_name = f"imagenet/{identifier.split('_')[0]}/{identifier}.jpg"
                        stored_images = imagenet_images
                    else:
                        match = open_images_matches.get(get_flickr_photo_number(url=image))
                        identifier = match[1] if match else None
                        file_name = get_open_images_file_name(image_identifier=identifier) if match else None
                        stored_images = open_images_images
                    if identifier not in stored_images:
                        unavailable_images.append(image)
                        continue
                    perceptual_hash = stored_images[identifier]
                    samples.append([split_name, file_name, label_text,
                                    int(perceptual_hash, 16) if perceptual_hash else None])

            # Images of the list left out, for the summary
            report_path = reports_folder / f"unavailable_{Path(list_file_name).stem}.txt"
            report_path.write_text(data="\n".join(unavailable_images))
            if unavailable_images:
                print_color(color="yellow", output=f"{len(unavailable_images)} images of {list_file_name} not stored : "
                                                   f"{format_path_link(path=report_path)}", indent=1)
    return samples


def merge_label_texts(label_texts: List[str]) -> str:
    """
    Merges the labels of the copies of one image into the labels of the kept sample: every category of any copy, with
    its highest confidence.

    Args:
        label_texts (List[str]): The raw label text of every copy.

    Returns:
        str: The merged raw label text, in the 'index:confidence' format of the training lists.
    """
    merged_confidences = {}
    for label_text in label_texts:
        _, category_indices, confidences = parse_annotation_line(line=f"image {label_text}")
        for category_index, confidence in zip(category_indices, confidences):
            merged_confidences[category_index] = max(confidence, merged_confidences.get(category_index, 0.0))
    return "\t".join(f"{category_index}:{confidence:g}" for category_index, confidence in merged_confidences.items())


def merge_duplicate_samples(samples: List[List[Any]]) -> Tuple[List[List[Any]], int]:
    """
    Keeps one sample per picture among the samples with the same perceptual hash: the copies of a test image are left
    out of the training images, and the copies within a split become one sample with the labels of all of them, so
    that no picture weighs twice in the training nor appears in both the training and the test images.

    Args:
        samples (List[List[Any]]): The split name, file name, raw label text and perceptual hash of every sample.

    Returns:
        Tuple[List[List[Any]], int]: The kept samples, and the number of samples left out.
    """
    # Indices of the samples of every hash, kept only for the hashes of several samples
    sample_indices_by_hash = {}
    for sample_index, sample in enumerate(samples):
        if sample[3] is not None:
            sample_indices_by_hash.setdefault(sample[3], []).append(sample_index)
    removed_indices = set()
    for sample_indices in sample_indices_by_hash.values():
        if len(sample_indices) < 2:
            continue

        # Test copies win over training copies, then the first copy of the split carries the labels of all the copies
        test_indices = [sample_index for sample_index in sample_indices if samples[sample_index][0] == "test"]
        kept_split_indices = test_indices if test_indices else sample_indices
        samples[kept_split_indices[0]][2] = merge_label_texts(label_texts=[samples[sample_index][2]
                                                                           for sample_index in kept_split_indices])
        removed_indices.update(sample_index for sample_index in sample_indices
                               if sample_index != kept_split_indices[0])

    kept_samples = [sample for sample_index, sample in enumerate(samples) if sample_index not in removed_indices]
    return kept_samples, len(removed_indices)


def write_multi_label_manifest(manifest_path: Path, categories: List[Dict[str, Any]],
                               samples: List[List[Any]]) -> None:
    """
    Writes the 'labels.json' manifest of a split for the 'multi_label' output mode of ClassificationDataset: the
    categories, then one sample per line with its 'category_ids' (Tencent index plus 1) and its
    'category_confidences', which the loader ignores but keeps available for soft targets. The samples are written one
    by one instead of building the whole JSON text, which would take several gigabytes for the training split.

    Args:
        manifest_path (Path): The manifest file.
        categories (List[Dict[str, Any]]): The categories of the manifests.
        samples (List[List[Any]]): The split name, file name, raw label text and perceptual hash of every sample.
    """
    temporary_path = manifest_path.with_name(f"{manifest_path.name}.part")
    with open(file=temporary_path, mode="w") as manifest_file:
        manifest_file.write(f'{{"categories": {json.dumps(obj=categories)}, "samples": [\n')
        for sample_index, sample in enumerate(samples):
            _, category_indices, confidences = parse_annotation_line(line=f"image {sample[2]}")
            manifest_file.write(("" if sample_index == 0 else ",\n") +
                                json.dumps(obj={"id": sample_index + 1, "file_name": sample[1],
                                                "category_ids": [category_index + 1
                                                                 for category_index in category_indices],
                                                "category_confidences": confidences}))
        manifest_file.write("\n]}\n")
    temporary_path.rename(target=manifest_path)


def build_manifests(dataset_path: Path, excluded_hashes: List[int], validation_size: int) -> None:
    """
    Builds the multi-label manifests of the train, validation and test splits from the stored images: the Tencent
    training images give the train split, out of which the validation split is held out by file name hash (the same
    images on every run), and the Tencent validation images give the test split. The copies of a picture are merged
    and the copies of the excluded images left out first. It is rebuilt on every run, which takes a few minutes.

    Args:
        dataset_path (Path): The folder of the dataset.
        excluded_hashes (List[int]): The perceptual hashes of the images whose copies are left out.
        validation_size (int): The number of training images moved to the validation split.
    """
    categories = read_category_dictionary(dictionary_path=dataset_path / "annotations" / TENCENT_DICTIONARY_FILE)
    samples = collect_samples(dataset_path=dataset_path)
    number_collected_samples = len(samples)

    # One sample per picture, then no copy of an excluded image
    samples, number_duplicates = merge_duplicate_samples(samples=samples)
    chunk_index = build_hash_chunk_index(hash_values=excluded_hashes)
    samples_without_copies = [sample for sample in samples
                              if sample[3] is None or not is_near_copy(hash_value=sample[3], chunk_index=chunk_index)]
    number_excluded_copies = len(samples) - len(samples_without_copies)

    # Training images with the smallest file name hashes moved to the validation split
    training_hashes = [(compute_file_name_hash(file_name=sample[1]), sample_index)
                       for sample_index, sample in enumerate(samples_without_copies) if sample[0] == "train"]
    for _, sample_index in heapq.nsmallest(validation_size, training_hashes):
        samples_without_copies[sample_index][0] = "validation"

    # Manifests of every split reading the images folder through their 'data' link
    manifests_folder = dataset_path / "manifests"
    split_sizes = {}
    for split_name in SPLIT_NAMES:
        split_folder = manifests_folder / split_name
        split_folder.mkdir(parents=True, exist_ok=True)
        link_folder(link_path=split_folder / "data", target_folder=dataset_path / "images")
        split_samples = [sample for sample in samples_without_copies if sample[0] == split_name]
        write_multi_label_manifest(manifest_path=split_folder / "labels.json", categories=categories,
                                   samples=split_samples)
        split_sizes[split_name] = len(split_samples)
        print_color(color="green", output=f"{split_name} split : {len(split_samples)} images : "
                                          f"{format_path_link(path=split_folder / 'labels.json')}", indent=1)
    write_label_mapping(categories=categories, output_folder=manifests_folder)

    # Counts of every step, for the record of the dataset
    summary_path = dataset_path / "reports" / "summary.json"
    summary_path.write_text(data=json.dumps(obj={"collected_samples": number_collected_samples,
                                                 "merged_duplicates": number_duplicates,
                                                 "excluded_copies": number_excluded_copies,
                                                 "split_sizes": split_sizes, "categories": len(categories)}, indent=4))
    print_color(color="green", output=f"{number_duplicates} duplicates merged, {number_excluded_copies} copies of "
                                      f"excluded images left out : {format_path_link(path=summary_path)}")


def download_foundation_model_images(dataset_folder: str, local_open_images_folder: Optional[str],
                                     excluded_images_folder: Optional[str], shorter_side: int, jpeg_quality: int,
                                     minimum_free_gigabytes: float, validation_size: int, number_workers: int,
                                     number_download_threads: int) -> None:
    """
    Downloads Tencent ML-Images and prepares it for the multi-label classification task, whose trained backbones
    initialise the other tasks through Backbone.pretrained. The dataset folder ends up holding:
    - 'annotations/': the Tencent lists and category dictionary, and the Open Images metadata;
    - 'images/imagenet/<WordNet identifier>/' and 'images/openimages/<first two characters>/': the stored images;
    - 'progress/': the records of the finished shards and batches, from which every step resumes;
    - 'manifests/': the splits read by the classification loader ('<split>/data' linking to the images and
      '<split>/labels.json'), and 'label_mapping.json', the class index of every category being its Tencent index;
    - 'reports/': the images of the Tencent lists left out, and the counts of the manifest step.
    Every step skips the work already done, so it can be run again after an interruption or a lack of disk space.

    Args:
        dataset_folder (str): The folder of the dataset.
        local_open_images_folder (Optional[str]): A local Open Images copy ('<subset>/data/<image identifier>.jpg')
            read before the S3 bucket for the validation and test images, or None. The training images are always
            streamed from the archives, which hold them all.
        excluded_images_folder (Optional[str]): The class folders of the images whose copies are left out (the
            ImageNet-1k test split), or None to leave nothing out.
        shorter_side (int): The maximum length of the shorter side of the stored images.
        jpeg_quality (int): The JPEG quality of the stored images.
        minimum_free_gigabytes (float): The free space below which the downloads stop.
        validation_size (int): The number of training images moved to the validation split.
        number_workers (int): The number of worker processes.
        number_download_threads (int): The number of single files fetched at once by every Open Images batch worker.

    Raises:
        PermissionError: If this machine is not logged in to Hugging Face with an account that accepted the terms of
            ImageNet-22k.
        FileNotFoundError: If the folder of the excluded images does not exist.
    """
    dataset_path = Path(dataset_folder).absolute()
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Stop before downloading anything when ImageNet-22k or the excluded images can't be read
    try:
        HfApi().auth_check(repo_id=IMAGENET_REPOSITORY, repo_type="dataset")
    except GatedRepoError as error:
        raise PermissionError(f"No access to the gated dataset {IMAGENET_REPOSITORY} : accept its terms on "
                              f"https://huggingface.co/datasets/{IMAGENET_REPOSITORY} with your Hugging Face account, "
                              f"then log in on this machine with 'uv run hf auth login'.") from error
    if excluded_images_folder is not None and not Path(excluded_images_folder).is_dir():
        raise FileNotFoundError(f"The folder of the excluded images does not exist : "
                                f"{format_path_link(path=excluded_images_folder)}. "
                                f"Download ImageNet-1k with vision_training/data/classification/imagenet_download.py, "
                                f"or set it to None to leave no image out.")

    # Annotation files, then the images of both sources
    download_annotation_files(annotations_folder=dataset_path / "annotations")
    download_open_images(dataset_path=dataset_path,
                         local_open_images_folder=Path(local_open_images_folder) if local_open_images_folder else None,
                         shorter_side=shorter_side, jpeg_quality=jpeg_quality,
                         minimum_free_gigabytes=minimum_free_gigabytes, number_workers=number_workers,
                         number_download_threads=number_download_threads)
    download_imagenet_images(dataset_path=dataset_path, shorter_side=shorter_side, jpeg_quality=jpeg_quality,
                             minimum_free_gigabytes=minimum_free_gigabytes, number_workers=number_workers)

    # Hashes of the excluded images, then the manifests
    excluded_hashes = []
    if excluded_images_folder is not None:
        excluded_hashes = hash_excluded_images(excluded_images_folder=Path(excluded_images_folder),
                                               hashes_path=dataset_path / "progress" / "excluded_image_hashes.json",
                                               shorter_side=shorter_side, number_workers=number_workers)
    build_manifests(dataset_path=dataset_path, excluded_hashes=excluded_hashes, validation_size=validation_size)


if __name__ == "__main__":
    # Folder of the dataset, next to the other datasets. It needs about 1.7 TB: about 1.0 TB of ImageNet images and
    # 0.7 TB of Open Images images, plus the ImageNet shards being processed (the Open Images archives are streamed)
    dataset_folder = "/home/robert_kelevra/Data/object_detection_data/ml-images"
    # Open Images copy already on disk, whose validation images are read instead of downloaded again
    local_open_images_folder = "/home/robert_kelevra/Data/object_detection_data/open-images-v7"
    # ImageNet-1k test split (official validation images), written by imagenet_download.py: its copies are left out
    # so that the pre-trained backbones can be evaluated on it fairly. None leaves no image out
    excluded_images_folder = "/home/robert_kelevra/Data/object_detection_data/imagenet-1k/images/test"
    # Maximum length of the shorter side of the stored images (never enlarged), and their JPEG quality
    shorter_side = 512
    jpeg_quality = 85
    # Free disk space in GB below which no new shard or batch is started, the next run resuming from there
    minimum_free_gigabytes = 100
    # Training images moved to the validation split, the Tencent validation images being the test split
    validation_size = 50000
    # Number of worker processes downloading and storing the images (16 lets every Open Images training archive be
    # streamed at once), and of single files fetched at once by every Open Images batch worker
    number_workers = 16
    number_download_threads = 16

    download_foundation_model_images(dataset_folder=dataset_folder, local_open_images_folder=local_open_images_folder,
                                     excluded_images_folder=excluded_images_folder, shorter_side=shorter_side,
                                     jpeg_quality=jpeg_quality, minimum_free_gigabytes=minimum_free_gigabytes,
                                     validation_size=validation_size, number_workers=number_workers,
                                     number_download_threads=number_download_threads)
