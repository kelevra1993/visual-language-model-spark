"""
Downloads ImageNet-22k (the full ImageNet fall11 release: 14.1M images, 21,841 WordNet classes) from Hugging Face and
prepares it for the classification task, whose trained backbones initialise the other tasks through
Backbone.pretrained. The webdataset shards of a pinned revision of the repository are fetched one by one by worker
processes, each download retried after a network failure, every image written with its original bytes into its class
folder of the train split and the shard deleted, so that an interrupted run resumes shard by shard. Every shard, the
published validation shards included, goes to training; one image per class of at least 50 images is then held out in
the test split, which is also the validation split: 'manifests/validation' and 'manifests/test' describe the same
images. The split is redone from wherever the images sit, so a run on a folder split another way moves the images
instead of downloading them again. The class folder manifests and label mapping written at the end are read by the
classification configurations. Run as a script with the settings of its main block.
"""

import hashlib
import json
import multiprocessing
import os
import shutil
import tarfile
import time

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Dict, List, Set

import cv2
import httpx2
import numpy as np

from huggingface_hub import HfApi, constants as hugging_face_constants, hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError
from tqdm import tqdm

from vision_training.data.classification.class_folder_manifests import create_class_folder_manifests
from vision_training.utilities.os_utilities import format_path_link, print_color

# Gated Hugging Face dataset holding the full ImageNet fall11 release (21,841 classes) as webdataset tar shards of
# about 0.29 GB, whose sample keys are the original file names (e.g. 'n04310904_8388'), the shorter side of the images
# being at most 600 pixels. Its terms must be accepted on the dataset page, and this machine logged in with
# 'uv run hf auth login'
IMAGENET_22K_REPOSITORY = "timm/imagenet-22k-wds"

# Commit of the repository read by every run, so that a later change of the repository can't change the dataset: the
# last commit of its main branch (2024-01-29) when it was looked up on 2026-11-11
IMAGENET_22K_REVISION = "8bc63a33ad2e4b1207549248a3f2175fe4b6a546"

# Prefixes of the shards of the repository, all stored as training images: the 4,096 training shards (13,673,551
# images) and the 512 validation shards (472,840 images, 40 per class of the 11,821 classes of ImageNet-12k)
SHARD_PREFIXES = ("imagenet22k-train-", "imagenet22k-validation-")

# Split folders of 'images/': every shard writes its images to the train split, from which the held-out images are
# moved to the test split
TRAIN_SPLIT = "train"
HELD_OUT_SPLIT = "test"

# Image folder of every manifest split: the validation split is the test split, both manifests describing the same
# held-out images, so that every configuration keeps its three data folders
MANIFEST_SPLIT_FOLDERS = {"train": TRAIN_SPLIT, "validation": HELD_OUT_SPLIT, "test": HELD_OUT_SPLIT}

# Seconds without a received byte after which a request to the Hub fails instead of waiting on a dropped connection
DOWNLOAD_TIMEOUT_SECONDS = 60

# Attempts of every shard download, the wait between two attempts doubling from the first wait up to the maximum one,
# which rides out about one hour without network before the run stops
MAXIMUM_DOWNLOAD_ATTEMPTS = 10
FIRST_RETRY_WAIT_SECONDS = 30
MAXIMUM_RETRY_WAIT_SECONDS = 600

# HTTP statuses of the Hub retried besides the server errors: request timeout and rate limit
RETRIED_STATUS_CODES = (408, 429)

# Factor applied to the number of workers to give the number of tasks queued in the process pool
QUEUED_TASKS_PER_WORKER = 2


def get_free_gigabytes(folder: Path) -> float:
    """
    Gives the free space of the disk of a folder, checked before every shard of the download so that it stops before
    filling the disk.

    Args:
        folder (Path): A folder of the disk.

    Returns:
        float: The free space in gigabytes.
    """
    return shutil.disk_usage(path=folder).free / 1e9


def configure_hugging_face_downloads(download_timeout_seconds: int) -> None:
    """
    Makes the downloads of this process plain HTTP downloads that fail after a silent connection, so that a network
    reset raises an error retried by download_shard instead of a hang: the Xet client of the Hub hung for hours on a
    dropped Wi-Fi connection on 2026-11-06. huggingface_hub reads the HF_HUB_DISABLE_XET and HF_HUB_DOWNLOAD_TIMEOUT
    environment variables only once, into its constants module, when it is first imported, but reads these constants
    at every download: setting the constants takes effect after the imports at the top of this file, where setting the
    variables would be ignored. It is called by every shard task, since the spawned worker processes import the
    library afresh.

    Args:
        download_timeout_seconds (int): The seconds without a received byte after which a download fails.
    """
    hugging_face_constants.HF_HUB_DISABLE_XET = True
    hugging_face_constants.HF_HUB_DOWNLOAD_TIMEOUT = download_timeout_seconds


def is_transient_download_error(error: Exception) -> bool:
    """
    Tells whether a failed shard download is worth trying again: a dropped connection, a timeout or a server error,
    which pass after a while, unlike a missing file or a refused access, which stop the run at once.

    Args:
        error (Exception): The error raised by the download.

    Returns:
        bool: Whether the download should be tried again.
    """
    # Answers of the Hub: only the server errors, the request timeouts and the rate limits pass after a while
    if isinstance(error, HfHubHTTPError):
        return error.response.status_code >= 500 or error.response.status_code in RETRIED_STATUS_CODES

    # Connection errors of the HTTP client, and of the metadata request, raised as a file missing from the local cache
    return isinstance(error, (httpx2.TransportError, LocalEntryNotFoundError))


def download_shard(shard_file_name: str, downloads_folder: Path, maximum_attempts: int, first_retry_wait_seconds: float,
                   maximum_retry_wait_seconds: float) -> Path:
    """
    Downloads one shard of the pinned revision of ImageNet-22k for process_imagenet_22k_shard, trying again after a
    transient network failure with a doubling wait, a partial download being resumed by the Hugging Face client. This
    replaces the shell loop that restarted the whole run after every network reset.

    Args:
        shard_file_name (str): The tar shard in the repository, e.g. 'imagenet22k-train-0000.tar'.
        downloads_folder (Path): The folder receiving the shard.
        maximum_attempts (int): The number of attempts before the error is raised.
        first_retry_wait_seconds (float): The wait before the second attempt, doubled before every next one.
        maximum_retry_wait_seconds (float): The longest wait between two attempts.

    Returns:
        Path: The downloaded shard.

    Raises:
        Exception: The error of the last attempt, or at once an error that another attempt can't fix.
    """
    number_failed_attempts = 0
    while True:
        try:
            return Path(hf_hub_download(repo_id=IMAGENET_22K_REPOSITORY, filename=shard_file_name, repo_type="dataset",
                                        revision=IMAGENET_22K_REVISION, local_dir=downloads_folder,
                                        etag_timeout=DOWNLOAD_TIMEOUT_SECONDS))
        except Exception as error:
            # Raise the errors that another attempt can't fix, and the error of the last attempt
            number_failed_attempts += 1
            if not is_transient_download_error(error=error) or number_failed_attempts >= maximum_attempts:
                raise

            # Wait for the network to come back, longer after every failure
            wait_seconds = min(first_retry_wait_seconds * 2 ** (number_failed_attempts - 1), maximum_retry_wait_seconds)
            print_color(color="yellow", output=f"{shard_file_name} : attempt {number_failed_attempts} of "
                                               f"{maximum_attempts} failed ({type(error).__name__}), next one in "
                                               f"{wait_seconds:g} seconds")
            time.sleep(wait_seconds)


def write_records(records_path: Path, records: Dict[str, Any]) -> None:
    """
    Saves what one shard task did, read back by the split size check and the summary. The file is written last and
    renamed into place, so it also marks the shard as done for the next run.

    Args:
        records_path (Path): The records file of the shard.
        records (Dict[str, Any]): The keys of the 'stored' and 'undecodable' images of the shard.
    """
    temporary_path = records_path.with_name(f"{records_path.name}.part")
    temporary_path.write_text(data=json.dumps(obj=records))
    temporary_path.rename(target=records_path)


def process_imagenet_22k_shard(shard_file_name: str, downloads_folder: Path, images_folder: Path,
                               records_path: Path) -> int:
    """
    Downloads one ImageNet-22k shard, writes every image with its original bytes into
    'train/<WordNet identifier>/<key>.jpg' of the images folder, leaving out the images that can't be decoded, saves
    the records of the shard, then deletes the shard. It runs in a worker process for every shard without records, so
    the disk only holds the shards being processed. An image whose held-out copy is already in the test split is
    written to the train split again, the next split keeping one copy.

    Args:
        shard_file_name (str): The tar shard in the repository.
        downloads_folder (Path): The folder receiving the shard while it is processed.
        images_folder (Path): The root folder of the stored images.
        records_path (Path): The records file of the shard, written once all its images are.

    Returns:
        int: The number of images written.
    """
    configure_hugging_face_downloads(download_timeout_seconds=DOWNLOAD_TIMEOUT_SECONDS)
    shard_path = download_shard(shard_file_name=shard_file_name, downloads_folder=downloads_folder,
                                maximum_attempts=MAXIMUM_DOWNLOAD_ATTEMPTS,
                                first_retry_wait_seconds=FIRST_RETRY_WAIT_SECONDS,
                                maximum_retry_wait_seconds=MAXIMUM_RETRY_WAIT_SECONDS)

    # Read the JPEG members of the shard, the '.cls' and '.json' members repeating the class of the key
    records = {"stored": [], "undecodable": []}
    with tarfile.open(name=shard_path, mode="r") as shard:
        for member in shard:
            imagenet_key, _, extension = Path(member.name).name.rpartition(".")
            if extension != "jpg":
                continue
            image_bytes = shard.extractfile(member=member).read()

            # Leave out the images that the training loader could not read either
            image = cv2.imdecode(buf=np.frombuffer(buffer=image_bytes, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
            if image is None:
                records["undecodable"].append(imagenet_key)
                continue

            # Store the original bytes in the folder of the WordNet identifier starting the key
            image_path = images_folder / TRAIN_SPLIT / imagenet_key.split("_")[0] / f"{imagenet_key}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(data=image_bytes)
            records["stored"].append(imagenet_key)

    write_records(records_path=records_path, records=records)
    shard_path.unlink()
    return len(records["stored"])


def run_tasks_with_free_space_check(task_function: Callable[..., int], task_arguments: List[Dict[str, Any]],
                                    disk_folder: Path, minimum_free_gigabytes: float, number_workers: int,
                                    description: str) -> int:
    """
    Runs download tasks in worker processes, queuing a new task only while the disk keeps 'minimum_free_gigabytes'
    free, so that the download stops cleanly instead of filling the disk. When the space runs out, the running tasks
    finish, and the next run resumes from the tasks without records. The workers are started fresh (spawn) instead of
    forked, since the Hugging Face client runs threads.

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


def download_imagenet_22k_shards(dataset_path: Path, minimum_free_gigabytes: float, number_workers: int) -> None:
    """
    Stores the images of the 4,608 shards of ImageNet-22k (1.3 TB, about 0.29 GB each), one worker task per shard
    without records, the shards already recorded by an earlier run being skipped.

    Args:
        dataset_path (Path): The folder of the dataset.
        minimum_free_gigabytes (float): The free space below which no new shard is started.
        number_workers (int): The number of worker processes.
    """
    records_folder = dataset_path / "progress" / "shards"
    records_folder.mkdir(parents=True, exist_ok=True)

    # Shards of the pinned revision without records
    shard_file_names = sorted(file_name for file_name in HfApi().list_repo_files(repo_id=IMAGENET_22K_REPOSITORY,
                                                                                 repo_type="dataset",
                                                                                 revision=IMAGENET_22K_REVISION)
                              if file_name.startswith(SHARD_PREFIXES) and file_name.endswith(".tar"))
    task_arguments = [{"shard_file_name": file_name, "downloads_folder": dataset_path / "shard_downloads",
                       "images_folder": dataset_path / "images",
                       "records_path": records_folder / f"{Path(file_name).stem}.json"}
                      for file_name in shard_file_names
                      if not (records_folder / f"{Path(file_name).stem}.json").exists()]
    print_color(color="blue", output=f"ImageNet-22k : {len(task_arguments)} of {len(shard_file_names)} shards left : "
                                     f"{format_path_link(path=dataset_path / 'images')}")

    number_written_images = run_tasks_with_free_space_check(task_function=process_imagenet_22k_shard,
                                                            task_arguments=task_arguments, disk_folder=dataset_path,
                                                            minimum_free_gigabytes=minimum_free_gigabytes,
                                                            number_workers=number_workers,
                                                            description="ImageNet-22k Shards")
    print_color(color="green", output=f"{number_written_images} ImageNet-22k images written", indent=1)


def count_shard_records(records_folder: Path) -> Dict[str, int]:
    """
    Adds up the records of every finished shard, giving the number of images the split size check expects and the
    counts of the summary.

    Args:
        records_folder (Path): The folder of the records files of the shards.

    Returns:
        Dict[str, int]: The number of 'shards', and of 'stored' and 'undecodable' images.
    """
    counts = {"shards": 0, "stored": 0, "undecodable": 0}
    for records_path in sorted(records_folder.glob(pattern="*.json")):
        records = json.loads(records_path.read_text())
        counts["shards"] += 1
        counts["stored"] += len(records["stored"])
        counts["undecodable"] += len(records["undecodable"])
    return counts


def compute_file_name_hash(file_name: str) -> str:
    """
    Gives the MD5 hash of a file name, the order in which choose_held_out_file_names picks the held-out images of a
    class: spread over the class like a random draw, but the same on every run and machine without a seed.

    Args:
        file_name (str): The file name of an image.

    Returns:
        str: The hexadecimal MD5 hash of the file name.
    """
    return hashlib.md5(file_name.encode(), usedforsecurity=False).hexdigest()


def choose_held_out_file_names(file_names: List[str], images_per_class: int, minimum_class_size: int) -> Set[str]:
    """
    Applies the split rule to the images of one class, wherever they are stored: a class of at least
    'minimum_class_size' images gives its 'images_per_class' images with the smallest file name hashes to the held-out
    split, read as the validation and the test split, and a smaller class keeps all its images for training.

    Args:
        file_names (List[str]): The file names of every image of the class.
        images_per_class (int): The number of held-out images of a large enough class.
        minimum_class_size (int): The number of images below which a class gives no held-out image.

    Returns:
        Set[str]: The file names of the held-out images.
    """
    if len(file_names) < minimum_class_size:
        return set()
    return set(sorted(file_names, key=compute_file_name_hash)[:images_per_class])


def split_imagenet_22k_images(images_folder: Path, held_out_images_per_class: int, minimum_held_out_class_size: int,
                              dry_run: bool) -> Dict[str, Any]:
    """
    Puts every stored image into the split given by choose_held_out_file_names, from wherever it sits in the split
    folders of 'images/': after a download every image is in the train split, while a folder split another way (e.g.
    with a validation split held out from the training images) has its images moved, never downloaded again. A second
    copy of an image (written again by a shard downloaded after the split) is deleted, the copy already in its split
    being kept, and the split folders other than train and test are deleted once emptied. Running it again moves
    nothing.

    Args:
        images_folder (Path): The folder of the split folders of the stored images.
        held_out_images_per_class (int): The number of held-out images of every large enough class.
        minimum_held_out_class_size (int): The number of images below which a class gives no held-out image.
        dry_run (bool): Whether to only count the images that would be moved or deleted, leaving every file in place.

    Returns:
        Dict[str, Any]: The 'moved_images' of every '<source> to <destination>' pair of splits, the
            'deleted_duplicates', the number of 'held_out_classes' and the 'split_sizes' of the train and test splits
            once split.

    Raises:
        ValueError: If the held-out images could leave a class without training images.
    """
    if not 1 <= held_out_images_per_class < minimum_held_out_class_size:
        raise ValueError(f"The held-out images per class must be at least 1 and fewer than the minimum class size, got "
                         f"{held_out_images_per_class} and {minimum_held_out_class_size}")
    split_folders = sorted(path for path in images_folder.iterdir() if path.is_dir())
    class_identifiers = sorted({entry.name for split_folder in split_folders for entry in os.scandir(path=split_folder)
                                if entry.is_dir()})

    report = {"moved_images": {}, "deleted_duplicates": 0, "held_out_classes": 0,
              "split_sizes": {TRAIN_SPLIT: 0, HELD_OUT_SPLIT: 0}}
    for class_identifier in tqdm(class_identifiers, desc="Splitting Classes"):
        # Split folders holding every file name of the class, in alphabetical order of the splits
        file_splits: Dict[str, List[str]] = {}
        for split_folder in split_folders:
            class_folder = split_folder / class_identifier
            if class_folder.is_dir():
                for entry in os.scandir(path=class_folder):
                    file_splits.setdefault(entry.name, []).append(split_folder.name)
        held_out_names = choose_held_out_file_names(file_names=list(file_splits),
                                                    images_per_class=held_out_images_per_class,
                                                    minimum_class_size=minimum_held_out_class_size)
        if held_out_names:
            report["held_out_classes"] += 1

        for file_name, split_names in file_splits.items():
            # Keep the copy already in its split, or move the first one there
            destination_split = HELD_OUT_SPLIT if file_name in held_out_names else TRAIN_SPLIT
            kept_split = destination_split if destination_split in split_names else split_names[0]
            report["split_sizes"][destination_split] += 1
            if kept_split != destination_split:
                move_name = f"{kept_split} to {destination_split}"
                report["moved_images"][move_name] = report["moved_images"].get(move_name, 0) + 1
                if not dry_run:
                    destination_path = images_folder / destination_split / class_identifier / file_name
                    destination_path.parent.mkdir(parents=True, exist_ok=True)
                    (images_folder / kept_split / class_identifier / file_name).rename(target=destination_path)

            # Delete the other copies of the image
            for duplicate_split in split_names:
                if duplicate_split != kept_split:
                    report["deleted_duplicates"] += 1
                    if not dry_run:
                        (images_folder / duplicate_split / class_identifier / file_name).unlink()

        # Delete the class folders emptied by the moves, which the manifests would refuse
        if not dry_run:
            for split_folder in split_folders:
                class_folder = split_folder / class_identifier
                if class_folder.is_dir() and not any(class_folder.iterdir()):
                    class_folder.rmdir()

    # Delete the emptied split folders other than train and test, a folder still holding files raising an error
    if not dry_run:
        for split_folder in split_folders:
            if split_folder.name not in (TRAIN_SPLIT, HELD_OUT_SPLIT):
                split_folder.rmdir()
    return report


def check_split_sizes(images_folder: Path, expected_split_sizes: Dict[str, int]) -> None:
    """
    Verifies that every split holds its expected number of images before the manifests are built on them, which
    catches an incomplete download or split, e.g. an image deleted after its shard was recorded.

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


def build_imagenet_22k_splits(dataset_path: Path, held_out_images_per_class: int,
                              minimum_held_out_class_size: int) -> None:
    """
    Turns the stored images into the splits of the classification loader once every shard is done: the held-out
    images moved to the test split, the size of both splits checked against the shard records, the class folder
    manifests, the validation manifest describing the test images, and the counts of the dataset saved in
    'reports/summary.json'.

    Args:
        dataset_path (Path): The folder of the dataset.
        held_out_images_per_class (int): The number of held-out images of every large enough class.
        minimum_held_out_class_size (int): The number of images below which a class gives no held-out image.
    """
    images_folder = dataset_path / "images"

    # Move every image to its split, the test split existing even when no class is large enough
    split_report = split_imagenet_22k_images(images_folder=images_folder,
                                             held_out_images_per_class=held_out_images_per_class,
                                             minimum_held_out_class_size=minimum_held_out_class_size, dry_run=False)
    (images_folder / HELD_OUT_SPLIT).mkdir(parents=True, exist_ok=True)
    for move_name, number_moved_images in split_report["moved_images"].items():
        print_color(color="blue", output=f"{number_moved_images} images moved from {move_name}", indent=1)
    print_color(color="blue", output=f"{split_report['held_out_classes']} classes give held-out images, "
                                     f"{split_report['deleted_duplicates']} duplicates deleted", indent=1)

    # Every stored image is in either the train or the test split
    shard_counts = count_shard_records(records_folder=dataset_path / "progress" / "shards")
    number_held_out_images = split_report["split_sizes"][HELD_OUT_SPLIT]
    check_split_sizes(images_folder=images_folder,
                      expected_split_sizes={TRAIN_SPLIT: shard_counts["stored"] - number_held_out_images,
                                            HELD_OUT_SPLIT: number_held_out_images})
    create_class_folder_manifests(split_folders={split_name: str(images_folder / folder_name)
                                                 for split_name, folder_name in MANIFEST_SPLIT_FOLDERS.items()},
                                  output_folder=str(dataset_path / "manifests"), excluded_file_names=None)

    # Counts of every step, for the record of the dataset
    summary_path = dataset_path / "reports" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(data=json.dumps(obj={"revision": IMAGENET_22K_REVISION, "shard_counts": shard_counts,
                                                 "split_sizes": split_report["split_sizes"],
                                                 "held_out_classes": split_report["held_out_classes"],
                                                 "last_split_moves": split_report["moved_images"]}, indent=4))
    print_color(color="green", output=f"Summary : {format_path_link(path=summary_path)}")


def download_imagenet_22k(dataset_folder: Path, minimum_free_gigabytes: float, held_out_images_per_class: int,
                          minimum_held_out_class_size: int, number_workers: int) -> None:
    """
    Downloads ImageNet-22k and prepares it for the classification task. The dataset folder ends up holding:
    - 'images/train/<WordNet identifier>/': the original JPEG images of every shard but the held-out ones;
    - 'images/test/<WordNet identifier>/': the held-out images, one per class of at least 50 images by default;
    - 'progress/shards/': the records of the finished shards, from which a new run resumes;
    - 'manifests/': the splits read by the classification loader ('<split>/data' linking to the images and
      '<split>/labels.json'), 'validation' and 'test' both describing the held-out images, and 'label_mapping.json',
      the categories being the WordNet identifiers in alphabetical order;
    - 'reports/summary.json': the stored and undecodable images, and the size of both splits.
    Every step skips the work already done, so it can be run again after an interruption or a lack of disk space.

    Args:
        dataset_folder (Path): The folder of the dataset.
        minimum_free_gigabytes (float): The free space below which no new shard is started.
        held_out_images_per_class (int): The number of held-out images of every large enough class.
        minimum_held_out_class_size (int): The number of images below which a class gives no held-out image.
        number_workers (int): The number of worker processes.

    Raises:
        PermissionError: If this machine is not logged in to Hugging Face with an account that accepted the terms of
            ImageNet-22k.
    """
    dataset_path = dataset_folder.absolute()
    dataset_path.mkdir(parents=True, exist_ok=True)
    configure_hugging_face_downloads(download_timeout_seconds=DOWNLOAD_TIMEOUT_SECONDS)

    # Stop before downloading anything when ImageNet-22k can't be read with the credentials of this machine
    try:
        HfApi().auth_check(repo_id=IMAGENET_22K_REPOSITORY, repo_type="dataset")
    except GatedRepoError as error:
        raise PermissionError(f"No access to the gated dataset {IMAGENET_22K_REPOSITORY} : accept its terms on "
                              f"https://huggingface.co/datasets/{IMAGENET_22K_REPOSITORY} with your Hugging Face "
                              f"account, then log in on this machine with 'uv run hf auth login'.") from error

    # Images of every shard, then the splits and manifests
    download_imagenet_22k_shards(dataset_path=dataset_path, minimum_free_gigabytes=minimum_free_gigabytes,
                                 number_workers=number_workers)
    build_imagenet_22k_splits(dataset_path=dataset_path, held_out_images_per_class=held_out_images_per_class,
                              minimum_held_out_class_size=minimum_held_out_class_size)


if __name__ == "__main__":
    # Folder of the dataset, read by the classification configurations. It needs about 1.2 TB of original images
    # (1,100 GiB measured, 83 KB per image on average), plus the shards being processed
    dataset_folder = Path.home() / "Data" / "classifier_data" / "imagenet-22k"
    # Free disk space in GB below which no new shard is started, the next run resuming from there
    minimum_free_gigabytes = 100
    # Images of every class of at least 50 images held out as the validation and test split (the same images), the
    # smaller classes keeping all their images for training
    held_out_images_per_class = 1
    minimum_held_out_class_size = 50
    # Number of worker processes downloading the shards and writing their images
    number_workers = 16

    download_imagenet_22k(dataset_folder=dataset_folder, minimum_free_gigabytes=minimum_free_gigabytes,
                          held_out_images_per_class=held_out_images_per_class,
                          minimum_held_out_class_size=minimum_held_out_class_size, number_workers=number_workers)
