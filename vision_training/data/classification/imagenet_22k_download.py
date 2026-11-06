"""
Downloads ImageNet-22k (the full ImageNet fall11 release: 14.1M images, 21,841 WordNet classes) from Hugging Face for
the pre-training of the backbones on the classification task: the webdataset shards fetched one by one by worker
processes, every image written with its original bytes into the class folder of its split and the shard deleted, the
copies of the ImageNet-1k test images left out by perceptual hash, a validation split held out from the training
images, the split size checks and the class folder manifests read by the classification loader. Run as a script with
the dataset folder and worker count set in its main block.
"""

import json
import multiprocessing
import os
import shutil
import tarfile

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError
from tqdm import tqdm

from vision_training.data.classification.class_folder_manifests import create_class_folder_manifests
from vision_training.data.classification.imagenet_download import check_split_sizes, hold_out_validation_images
from vision_training.utilities.os_utilities import format_path_link, get_image_paths, print_color

# Gated Hugging Face dataset holding the full ImageNet fall11 release (21,841 classes) as webdataset tar shards of
# about 0.29 GB, whose sample keys are the original file names (e.g. 'n04310904_8388'), the shorter side of the images
# being at most 600 pixels. Its terms must be accepted on the dataset page, and this machine logged in with
# 'uv run hf auth login'
IMAGENET_22K_REPOSITORY = "timm/imagenet-22k-wds"

# Split receiving the images of the shards of every prefix: the 4,096 training shards (13,673,551 images) give the
# train split, and the 512 validation shards (472,840 images, 40 per class of the 11,821 classes of ImageNet-12k) the
# test split, the validation split being held out from the training images
SHARD_SPLITS = {"imagenet22k-train-": "train", "imagenet22k-validation-": "test"}

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


def compute_perceptual_hash(image: np.ndarray) -> Optional[str]:
    """
    Computes the perceptual hash of an image, which finds the copies of the ImageNet-1k test images among the
    ImageNet-22k images. Unlike a hash of the file bytes, it survives the re-encoding and the resizing of the image:
    the grayscale image is shrunk to a 9 x 9 thumbnail, and every bit tells whether a pixel is brighter than its left
    neighbour (64 bits) or its upper neighbour (64 bits).

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


def decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decodes a JPEG image of a shard or of the ImageNet-1k test split for its perceptual hash, without applying any EXIF
    orientation so that both are hashed the same way. An image that cannot be decoded is not stored, since the
    training loader could not read it either.

    Args:
        image_bytes (bytes): The encoded image.

    Returns:
        Optional[np.ndarray]: The BGR image, or None when it cannot be decoded.
    """
    return cv2.imdecode(buf=np.frombuffer(buffer=image_bytes, dtype=np.uint8),
                        flags=cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


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
    Indexes the hashes of the excluded images by chunk, so that every ImageNet-22k image is compared with the few
    excluded images sharing one of its chunks instead of all of them.

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


def compute_image_file_hash(image_path: Path) -> Optional[str]:
    """
    Computes the perceptual hash of an ImageNet-1k test image, so that its copies among the ImageNet-22k images can be
    left out of the pre-training. It runs in a worker process for every test image.

    Args:
        image_path (Path): The test image.

    Returns:
        Optional[str]: The perceptual hash, or None for an undecodable or nearly uniform image.
    """
    image = decode_image(image_bytes=image_path.read_bytes())
    if image is None:
        return None
    return compute_perceptual_hash(image=image)


def hash_excluded_images(excluded_images_folder: Path, hashes_path: Path, number_workers: int) -> int:
    """
    Computes the perceptual hashes of the ImageNet-1k test images (one folder per class), the images whose copies are
    left out of the ImageNet-22k splits, so that the backbones pre-trained here can still be evaluated on them fairly.
    The hashes are saved once, before the shards, and read by every shard worker.

    Args:
        excluded_images_folder (Path): The folder of the class folders of the excluded images.
        hashes_path (Path): The JSON file of the hashes, skipped when it already exists.
        number_workers (int): The number of worker processes.

    Returns:
        int: The number of excluded images that have a hash.
    """
    if not hashes_path.exists():
        image_paths = [image_path for class_folder in sorted(excluded_images_folder.iterdir()) if class_folder.is_dir()
                       for image_path in get_image_paths(folder=class_folder)]
        with ProcessPoolExecutor(max_workers=number_workers,
                                 mp_context=multiprocessing.get_context(method="spawn")) as executor:
            futures = [executor.submit(compute_image_file_hash, image_path=image_path) for image_path in image_paths]
            hashes = [future.result() for future in tqdm(futures, desc="Hashing Excluded Images")]
        # Write through a temporary file so that an interruption never leaves a partial list
        temporary_path = hashes_path.with_name(f"{hashes_path.name}.part")
        temporary_path.write_text(data=json.dumps(obj=[perceptual_hash for perceptual_hash in hashes
                                                       if perceptual_hash]))
        temporary_path.rename(target=hashes_path)
    return len(json.loads(hashes_path.read_text()))


def read_excluded_hash_index(hashes_path: Optional[Path]) -> List[Dict[int, List[int]]]:
    """
    Reads the hashes saved by hash_excluded_images into the chunk index of is_near_copy, built by every shard worker
    since an index can't be passed between processes cheaply (about 50,000 hashes, built in a fraction of a second).

    Args:
        hashes_path (Optional[Path]): The JSON file of the hashes, or None when no image is left out.

    Returns:
        List[Dict[int, List[int]]]: The excluded hashes indexed by chunk, empty for every chunk without hashes.
    """
    hash_values = [] if hashes_path is None else [int(perceptual_hash, 16)
                                                  for perceptual_hash in json.loads(hashes_path.read_text())]
    return build_hash_chunk_index(hash_values=hash_values)


def write_records(records_path: Path, records: Dict[str, Any]) -> None:
    """
    Saves what one shard task did, read back by the split size check and the summary. The file is written last and
    renamed into place, so it also marks the shard as done for the next run.

    Args:
        records_path (Path): The records file of the shard.
        records (Dict[str, Any]): The 'split' of the shard and the keys of its 'stored', 'excluded' and 'undecodable'
            images.
    """
    temporary_path = records_path.with_name(f"{records_path.name}.part")
    temporary_path.write_text(data=json.dumps(obj=records))
    temporary_path.rename(target=records_path)


def get_shard_split(shard_file_name: str) -> str:
    """
    Gives the split receiving the images of a shard, from the prefix of its file name.

    Args:
        shard_file_name (str): The tar shard in the repository, e.g. 'imagenet22k-train-0000.tar'.

    Returns:
        str: The split name, 'train' or 'test'.

    Raises:
        ValueError: If the file name has none of the prefixes of SHARD_SPLITS.
    """
    for shard_prefix, split_name in SHARD_SPLITS.items():
        if shard_file_name.startswith(shard_prefix):
            return split_name
    raise ValueError(f"The shard {shard_file_name} has none of the prefixes {list(SHARD_SPLITS)}")


def process_imagenet_22k_shard(shard_file_name: str, downloads_folder: Path, images_folder: Path, records_path: Path,
                               excluded_hashes_path: Optional[Path]) -> int:
    """
    Downloads one ImageNet-22k shard, writes every image with its original bytes into
    '<split>/<WordNet identifier>/<key>.jpg' of the images folder, leaving out the copies of the excluded images and
    the undecodable images, saves the records of the shard, then deletes the shard. It runs in a worker process for
    every shard without records, so the disk only holds the shards being processed.

    Args:
        shard_file_name (str): The tar shard in the repository.
        downloads_folder (Path): The folder receiving the shard while it is processed.
        images_folder (Path): The root folder of the stored images.
        records_path (Path): The records file of the shard, written once all its images are.
        excluded_hashes_path (Optional[Path]): The hashes of the excluded images, or None to leave nothing out.

    Returns:
        int: The number of images written.
    """
    split_name = get_shard_split(shard_file_name=shard_file_name)
    chunk_index = read_excluded_hash_index(hashes_path=excluded_hashes_path)
    shard_path = Path(hf_hub_download(repo_id=IMAGENET_22K_REPOSITORY, filename=shard_file_name, repo_type="dataset",
                                      local_dir=downloads_folder))

    # Read the JPEG members of the shard, the '.cls' and '.json' members repeating the class of the key
    records = {"split": split_name, "stored": [], "excluded": [], "undecodable": []}
    with tarfile.open(name=shard_path, mode="r") as shard:
        for member in shard:
            imagenet_key, _, extension = Path(member.name).name.rpartition(".")
            if extension != "jpg":
                continue
            image_bytes = shard.extractfile(member=member).read()

            # Leave out the undecodable images and the copies of the excluded images
            image = decode_image(image_bytes=image_bytes)
            if image is None:
                records["undecodable"].append(imagenet_key)
                continue
            perceptual_hash = compute_perceptual_hash(image=image)
            if perceptual_hash is not None and is_near_copy(hash_value=int(perceptual_hash, 16),
                                                            chunk_index=chunk_index):
                records["excluded"].append(imagenet_key)
                continue

            # Store the original bytes in the folder of the WordNet identifier starting the key
            image_path = images_folder / split_name / imagenet_key.split("_")[0] / f"{imagenet_key}.jpg"
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


def download_imagenet_22k_shards(dataset_path: Path, excluded_hashes_path: Optional[Path],
                                 minimum_free_gigabytes: float, number_workers: int) -> None:
    """
    Stores the images of the 4,608 shards of ImageNet-22k (1.3 TB, about 0.29 GB each), one worker task per shard
    without records.

    Args:
        dataset_path (Path): The folder of the dataset.
        excluded_hashes_path (Optional[Path]): The hashes of the excluded images, or None to leave nothing out.
        minimum_free_gigabytes (float): The free space below which no new shard is started.
        number_workers (int): The number of worker processes.
    """
    records_folder = dataset_path / "progress" / "shards"
    records_folder.mkdir(parents=True, exist_ok=True)

    # Shards of both splits of the repository without records
    shard_file_names = sorted(file_name for file_name in HfApi().list_repo_files(repo_id=IMAGENET_22K_REPOSITORY,
                                                                                 repo_type="dataset")
                              if file_name.startswith(tuple(SHARD_SPLITS)) and file_name.endswith(".tar"))
    task_arguments = [{"shard_file_name": file_name, "downloads_folder": dataset_path / "shard_downloads",
                       "images_folder": dataset_path / "images",
                       "records_path": records_folder / f"{Path(file_name).stem}.json",
                       "excluded_hashes_path": excluded_hashes_path}
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


def count_shard_records(records_folder: Path) -> Dict[str, Dict[str, int]]:
    """
    Adds up the records of every finished shard by split, giving the number of images the split check expects and
    the counts of the summary.

    Args:
        records_folder (Path): The folder of the records files of the shards.

    Returns:
        Dict[str, Dict[str, int]]: The number of 'stored', 'excluded' and 'undecodable' images of every split of the
            shards ('train' and 'test').
    """
    counts = {split_name: {"stored": 0, "excluded": 0, "undecodable": 0} for split_name in SHARD_SPLITS.values()}
    for records_path in sorted(records_folder.glob(pattern="*.json")):
        records = json.loads(records_path.read_text())
        for count_name in counts[records["split"]]:
            counts[records["split"]][count_name] += len(records[count_name])
    return counts


def build_imagenet_22k_splits(dataset_path: Path, validation_images_per_class: int,
                              minimum_validation_class_size: int) -> None:
    """
    Turns the stored images into the splits of the classification loader once every shard is done: the validation
    split held out from the training images, the size of every split checked against the shard records, the class
    folder manifests, and the counts of the dataset saved in 'reports/summary.json'.

    Args:
        dataset_path (Path): The folder of the dataset.
        validation_images_per_class (int): The number of training images of every large enough class moved to the
            validation split.
        minimum_validation_class_size (int): The number of images below which a class gives no validation image.
    """
    images_folder = dataset_path / "images"

    # Hold out the validation split from the training images, the small classes keeping all their images
    number_moved_images = hold_out_validation_images(train_folder=images_folder / "train",
                                                     validation_folder=images_folder / "validation",
                                                     images_per_class=validation_images_per_class,
                                                     minimum_class_size=minimum_validation_class_size)
    print_color(color="blue", output=f"{number_moved_images} training images moved to the validation split")

    # Every stored training image is in either the train or the validation split, which exists even when no class is
    # large enough to give validation images
    shard_counts = count_shard_records(records_folder=dataset_path / "progress" / "shards")
    validation_folder = images_folder / "validation"
    validation_folder.mkdir(parents=True, exist_ok=True)
    number_validation_images = sum(len(os.listdir(path=class_folder)) for class_folder in validation_folder.iterdir()
                                   if class_folder.is_dir())
    check_split_sizes(images_folder=images_folder,
                      expected_split_sizes={"train": shard_counts["train"]["stored"] - number_validation_images,
                                            "validation": number_validation_images,
                                            "test": shard_counts["test"]["stored"]})
    create_class_folder_manifests(split_folders={split_name: str(images_folder / split_name)
                                                 for split_name in SPLIT_NAMES},
                                  output_folder=str(dataset_path / "manifests"))

    # Counts of every step, for the record of the dataset
    summary_path = dataset_path / "reports" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(data=json.dumps(obj={"shard_counts": shard_counts,
                                                 "validation_images": number_validation_images}, indent=4))
    print_color(color="green", output=f"{shard_counts['train']['excluded'] + shard_counts['test']['excluded']} copies "
                                      f"of excluded images left out : {format_path_link(path=summary_path)}")


def download_imagenet_22k(dataset_folder: str, excluded_images_folder: Optional[str], minimum_free_gigabytes: float,
                          validation_images_per_class: int, minimum_validation_class_size: int,
                          number_workers: int) -> None:
    """
    Downloads ImageNet-22k and prepares it for the classification task, whose trained backbones initialise the other
    tasks through Backbone.pretrained. The dataset folder ends up holding:
    - 'images/<split>/<WordNet identifier>/': the original JPEG images of the train split, of the validation split
      held out from it, and of the test split made of the validation shards of the repository;
    - 'progress/': the hashes of the excluded images and the records of the finished shards, from which a new run
      resumes;
    - 'manifests/': the splits read by the classification loader ('<split>/data' linking to the images and
      '<split>/labels.json'), and 'label_mapping.json', the categories being the WordNet identifiers in alphabetical
      order;
    - 'reports/summary.json': the stored, excluded and undecodable images of every split.
    Every step skips the work already done, so it can be run again after an interruption or a lack of disk space.

    Args:
        dataset_folder (str): The folder of the dataset.
        excluded_images_folder (Optional[str]): The class folders of the images whose copies are left out (the
            ImageNet-1k test split), or None to leave nothing out.
        minimum_free_gigabytes (float): The free space below which no new shard is started.
        validation_images_per_class (int): The number of training images of every large enough class moved to the
            validation split.
        minimum_validation_class_size (int): The number of images below which a class gives no validation image.
        number_workers (int): The number of worker processes.

    Raises:
        PermissionError: If this machine is not logged in to Hugging Face with an account that accepted the terms of
            ImageNet-22k.
        FileNotFoundError: If the folder of the excluded images does not exist.
    """
    dataset_path = Path(dataset_folder).absolute()
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Stop before downloading anything when ImageNet-22k or the excluded images can't be read
    try:
        HfApi().auth_check(repo_id=IMAGENET_22K_REPOSITORY, repo_type="dataset")
    except GatedRepoError as error:
        raise PermissionError(f"No access to the gated dataset {IMAGENET_22K_REPOSITORY} : accept its terms on "
                              f"https://huggingface.co/datasets/{IMAGENET_22K_REPOSITORY} with your Hugging Face "
                              f"account, then log in on this machine with 'uv run hf auth login'.") from error
    if excluded_images_folder is not None and not Path(excluded_images_folder).is_dir():
        raise FileNotFoundError(f"The folder of the excluded images does not exist : "
                                f"{format_path_link(path=excluded_images_folder)}. "
                                f"Download ImageNet-1k with vision_training/data/classification/imagenet_download.py, "
                                f"or set it to None to leave no image out.")

    # Hashes of the excluded images first, since every shard worker leaves out their copies
    excluded_hashes_path = None
    if excluded_images_folder is not None:
        excluded_hashes_path = dataset_path / "progress" / "excluded_image_hashes.json"
        excluded_hashes_path.parent.mkdir(parents=True, exist_ok=True)
        number_excluded_hashes = hash_excluded_images(excluded_images_folder=Path(excluded_images_folder),
                                                      hashes_path=excluded_hashes_path, number_workers=number_workers)
        print_color(color="blue", output=f"{number_excluded_hashes} excluded images hashed : "
                                         f"{format_path_link(path=excluded_hashes_path)}")

    # Images of every shard, then the splits and manifests
    download_imagenet_22k_shards(dataset_path=dataset_path, excluded_hashes_path=excluded_hashes_path,
                                 minimum_free_gigabytes=minimum_free_gigabytes, number_workers=number_workers)
    build_imagenet_22k_splits(dataset_path=dataset_path, validation_images_per_class=validation_images_per_class,
                              minimum_validation_class_size=minimum_validation_class_size)


if __name__ == "__main__":
    # Folder of the dataset, next to ImageNet-1k. It needs about 1.15 TB of original images (81 KB on average,
    # measured on the first shard), plus the shards being processed
    dataset_folder = "/home/robert_kelevra/Data/classifier_data/imagenet-22k"
    # ImageNet-1k test split (official validation images), written by imagenet_download.py: its copies are left out
    # so that the pre-trained backbones can be evaluated on it fairly. None leaves no image out
    excluded_images_folder = "/home/robert_kelevra/Data/classifier_data/imagenet-1k/images/test"
    # Free disk space in GB below which no new shard is started, the next run resuming from there
    minimum_free_gigabytes = 100
    # Training images of every class of at least 50 images moved to the validation split, the smaller classes keeping
    # all their images
    validation_images_per_class = 5
    minimum_validation_class_size = 50
    # Number of worker processes downloading the shards and writing their images
    number_workers = 16

    download_imagenet_22k(dataset_folder=dataset_folder, excluded_images_folder=excluded_images_folder,
                          minimum_free_gigabytes=minimum_free_gigabytes,
                          validation_images_per_class=validation_images_per_class,
                          minimum_validation_class_size=minimum_validation_class_size, number_workers=number_workers)
