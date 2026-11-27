"""
Tests of vision_training/data/classification/imagenet_download.py on small files written by the tests, without
network: the Hugging Face client settings, the retried shard download, a fake ImageNet-22k shard written with its
original bytes into the train split, the free space stop, the split rule, the split moving the images from wherever they
sit, the checks, manifests (validation equal to test) and summary built from the stored images, and the class folder
manifests read by the classification dataset.
"""

import io
import json
import tarfile

import cv2
import httpx2
import numpy as np
import pytest

from huggingface_hub import constants as hugging_face_constants
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from pathlib import Path
from typing import Any, Dict, List

from vision_training.data.classification import imagenet_download
from vision_training.data.classification.classification_dataset import ClassificationDataset
from vision_training.data.classification.imagenet_download import (
    build_imagenet_22k_splits, choose_held_out_file_names, compute_file_name_hash, configure_hugging_face_downloads,
    download_shard, is_transient_download_error, process_imagenet_22k_shard, run_tasks_with_free_space_check,
    split_imagenet_22k_images, write_class_folder_manifests, write_records)


class ScriptedDownload:
    """
    Stands in for hf_hub_download in the retry tests: every call raises or returns the next of the given outcomes, and
    the keyword arguments of every call are kept.
    """

    def __init__(self, outcomes: List[Any]) -> None:
        """
        Keeps the outcomes of the calls to come.

        Args:
            outcomes (List[Any]): The error raised or the shard path returned by every call, in call order.
        """
        self.outcomes = outcomes
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **keyword_arguments: Any) -> str:
        """
        Raises or returns the outcome of this call.

        Args:
            **keyword_arguments (Any): The arguments of the download.

        Returns:
            str: The path of the downloaded shard.
        """
        self.calls.append(keyword_arguments)
        outcome = self.outcomes[len(self.calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def encode_image(seed: int) -> bytes:
    """
    Encodes a small random BGR image as JPEG, as the files of the shards are, different for every seed.

    Args:
        seed (int): The seed of the random pixels.

    Returns:
        bytes: The JPEG bytes.
    """
    image = np.random.default_rng(seed=seed).integers(low=0, high=256, size=(24, 32, 3), dtype=np.uint8)
    return cv2.imencode(ext=".jpg", img=image)[1].tobytes()


def write_tar_shard(shard_path: Path, members: Dict[str, bytes]) -> None:
    """
    Writes a tar shard in the webdataset layout of ImageNet-22k ('<key>.jpg', '<key>.cls', '<key>.json').

    Args:
        shard_path (Path): The tar file to write.
        members (Dict[str, bytes]): The content of every member, keyed by member name.
    """
    with tarfile.open(name=shard_path, mode="w") as shard:
        for member_name, member_bytes in members.items():
            member_information = tarfile.TarInfo(name=member_name)
            member_information.size = len(member_bytes)
            shard.addfile(tarinfo=member_information, fileobj=io.BytesIO(initial_bytes=member_bytes))


def write_class_images(images_folder: Path, split_name: str, class_identifier: str, file_names: List[str]) -> None:
    """
    Writes placeholder image files into the class folder of a split, as the shards store them.

    Args:
        images_folder (Path): The folder of the split folders.
        split_name (str): The split folder, e.g. 'train'.
        class_identifier (str): The WordNet identifier naming the class folder.
        file_names (List[str]): The file names of the images.
    """
    class_folder = images_folder / split_name / class_identifier
    class_folder.mkdir(parents=True, exist_ok=True)
    for file_name in file_names:
        (class_folder / file_name).write_bytes(data=file_name.encode())


def list_split_files(images_folder: Path) -> Dict[str, List[str]]:
    """
    Lists the files of every split folder as '<class>/<file name>', to compare two layouts of the stored images.

    Args:
        images_folder (Path): The folder of the split folders.

    Returns:
        Dict[str, List[str]]: The sorted files of every split folder, keyed by split name.
    """
    return {split_folder.name: sorted(f"{image_path.parent.name}/{image_path.name}"
                                      for image_path in split_folder.rglob(pattern="*") if image_path.is_file())
            for split_folder in sorted(images_folder.iterdir())}


def test_configure_hugging_face_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the Hugging Face client of the process is set to plain HTTP downloads with the given timeout, the
    constants it reads at every download.

    Args:
        monkeypatch (pytest.MonkeyPatch): Restores the constants of the library after the test.
    """
    monkeypatch.setattr(target=hugging_face_constants, name="HF_HUB_DISABLE_XET", value=False)
    monkeypatch.setattr(target=hugging_face_constants, name="HF_HUB_DOWNLOAD_TIMEOUT", value=10)
    configure_hugging_face_downloads(download_timeout_seconds=60)
    assert hugging_face_constants.HF_HUB_DISABLE_XET is True
    assert hugging_face_constants.HF_HUB_DOWNLOAD_TIMEOUT == 60


def test_download_shard_retries_transient_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that a shard download is tried again after the network errors that stopped the earlier runs (a reset
    connection, a failed metadata request) and a server error, that the last error is raised once the attempts are
    used up, and that an error another attempt can't fix (a missing file) is raised at once.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): Replaces the Hugging Face download by failures, then a local shard.
    """
    request = httpx2.Request(method="GET", url="https://huggingface.co")
    server_error = HfHubHTTPError("Service Unavailable", response=httpx2.Response(status_code=503, request=request))
    missing_file_error = HfHubHTTPError("Not Found", response=httpx2.Response(status_code=404, request=request))
    assert is_transient_download_error(error=server_error)
    assert not is_transient_download_error(error=missing_file_error)

    # Three failures, then the shard, downloaded from the pinned revision
    shard_path = tmp_path / "imagenet22k-train-0000.tar"
    scripted_download = ScriptedDownload(outcomes=[httpx2.ReadError("Connection reset by peer"),
                                                   LocalEntryNotFoundError("No connection"), server_error,
                                                   str(shard_path)])
    monkeypatch.setattr(target=imagenet_download, name="hf_hub_download", value=scripted_download)
    assert download_shard(shard_file_name=shard_path.name, downloads_folder=tmp_path, maximum_attempts=4,
                          first_retry_wait_seconds=0, maximum_retry_wait_seconds=0) == shard_path
    assert len(scripted_download.calls) == 4
    assert scripted_download.calls[0]["revision"] == imagenet_download.IMAGENET_22K_REVISION

    # The attempts used up raise the last error
    scripted_download = ScriptedDownload(outcomes=[httpx2.ReadError("Connection reset by peer")])
    monkeypatch.setattr(target=imagenet_download, name="hf_hub_download", value=scripted_download)
    with pytest.raises(expected_exception=httpx2.ReadError):
        download_shard(shard_file_name=shard_path.name, downloads_folder=tmp_path, maximum_attempts=1,
                       first_retry_wait_seconds=0, maximum_retry_wait_seconds=0)

    # An error that another attempt can't fix is raised at the first attempt
    scripted_download = ScriptedDownload(outcomes=[missing_file_error, str(shard_path)])
    monkeypatch.setattr(target=imagenet_download, name="hf_hub_download", value=scripted_download)
    with pytest.raises(expected_exception=HfHubHTTPError, match="Not Found"):
        download_shard(shard_file_name=shard_path.name, downloads_folder=tmp_path, maximum_attempts=4,
                       first_retry_wait_seconds=0, maximum_retry_wait_seconds=0)
    assert len(scripted_download.calls) == 1


def test_process_imagenet_22k_shard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that a shard, here a published validation shard, writes the original bytes of its images into the class
    folders of the train split, records the undecodable image without storing it, and is deleted once processed.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): Replaces the Hugging Face download by a local shard.
    """
    kept_bytes = {"n01440764_1": encode_image(seed=0), "n09999999_2": encode_image(seed=1)}
    downloaded_shard_path = tmp_path / "downloads" / "imagenet22k-validation-000.tar"
    downloaded_shard_path.parent.mkdir()
    write_tar_shard(shard_path=downloaded_shard_path,
                    members={"n01440764_1.jpg": kept_bytes["n01440764_1"], "n01440764_1.cls": b"0",
                             "n01440764_1.json": b'{"label": 0, "class_name": "n01440764"}',
                             "n09999999_2.jpg": kept_bytes["n09999999_2"], "n01440764_4.jpg": b"broken"})
    monkeypatch.setattr(target=imagenet_download, name="hf_hub_download", value=lambda **_: str(downloaded_shard_path))
    monkeypatch.setattr(target=hugging_face_constants, name="HF_HUB_DISABLE_XET", value=False)

    records_path = tmp_path / "records.json"
    number_images = process_imagenet_22k_shard(shard_file_name=downloaded_shard_path.name,
                                               downloads_folder=tmp_path / "downloads",
                                               images_folder=tmp_path / "images", records_path=records_path)

    # Original bytes in the train split, the broken image recorded but not stored, the client set to plain HTTP
    assert number_images == 2
    for imagenet_key, image_bytes in kept_bytes.items():
        image_path = tmp_path / "images" / "train" / imagenet_key.split("_")[0] / f"{imagenet_key}.jpg"
        assert image_path.read_bytes() == image_bytes
    assert len(list((tmp_path / "images").rglob(pattern="*.jpg"))) == 2
    assert json.loads(records_path.read_text()) == {"stored": ["n01440764_1", "n09999999_2"],
                                                    "undecodable": ["n01440764_4"]}
    assert not downloaded_shard_path.exists()
    assert hugging_face_constants.HF_HUB_DISABLE_XET is True


def test_run_tasks_stops_without_free_space(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that no task is started when the disk has less free space than the minimum, the error telling how many
    tasks are left for the next run.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): Reports a nearly full disk.
    """
    monkeypatch.setattr(target=imagenet_download, name="get_free_gigabytes", value=lambda **_: 50.0)
    with pytest.raises(expected_exception=RuntimeError, match="2 tasks of 'Shards' were not started"):
        run_tasks_with_free_space_check(task_function=write_records, task_arguments=[{}, {}], disk_folder=tmp_path,
                                        minimum_free_gigabytes=100, number_workers=1, description="Shards")


def test_choose_held_out_file_names() -> None:
    """
    Verifies the split rule: a class of at least the minimum size gives its images with the smallest file name hashes,
    whatever their order, and a smaller class gives none.
    """
    file_names = [f"n01440764_{image_index}.jpg" for image_index in range(6)]
    expected_names = set(sorted(file_names, key=compute_file_name_hash)[:2])
    assert choose_held_out_file_names(file_names=file_names, images_per_class=2, minimum_class_size=6) == expected_names
    assert choose_held_out_file_names(file_names=file_names[::-1], images_per_class=2,
                                      minimum_class_size=6) == expected_names
    assert choose_held_out_file_names(file_names=file_names, images_per_class=2, minimum_class_size=7) == set()


def test_split_imagenet_22k_images(tmp_path: Path) -> None:
    """
    Verifies that the split gives the same layout from a fresh download (every image in train) and from the earlier
    layout of this dataset (a validation split held out from the training images and the published validation images
    in test), moving the images from wherever they sit, deleting a second copy and the emptied validation split, that
    the dry run counts the same moves without touching a file, that a second run moves nothing, and that held-out
    settings leaving a class without training images are refused.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # A large class and a small one, the file names of the held-out image of the large class first
    large_class_names = [f"n01440764_{image_index}.jpg" for image_index in range(6)]
    small_class_names = ["n01443537_0.jpg", "n01443537_1.jpg"]
    held_out_name = sorted(large_class_names, key=compute_file_name_hash)[0]
    other_large_names = [file_name for file_name in large_class_names if file_name != held_out_name]

    # Fresh download: every image in the train split
    fresh_folder = tmp_path / "fresh" / "images"
    write_class_images(images_folder=fresh_folder, split_name="train", class_identifier="n01440764",
                       file_names=large_class_names)
    write_class_images(images_folder=fresh_folder, split_name="train", class_identifier="n01443537",
                       file_names=small_class_names)

    # Earlier layout: the held-out image and another one in validation, two published validation images in test,
    # one of them also written again in train
    earlier_folder = tmp_path / "earlier" / "images"
    write_class_images(images_folder=earlier_folder, split_name="validation", class_identifier="n01440764",
                       file_names=[held_out_name, other_large_names[0]])
    write_class_images(images_folder=earlier_folder, split_name="test", class_identifier="n01440764",
                       file_names=other_large_names[1:3])
    write_class_images(images_folder=earlier_folder, split_name="train", class_identifier="n01440764",
                       file_names=other_large_names[2:])
    write_class_images(images_folder=earlier_folder, split_name="train", class_identifier="n01443537",
                       file_names=small_class_names)

    # The dry run counts the moves without touching a file
    earlier_layout = list_split_files(images_folder=earlier_folder)
    expected_report = {"moved_images": {"validation to test": 1, "validation to train": 1, "test to train": 1},
                       "deleted_duplicates": 1, "held_out_classes": 1, "split_sizes": {"train": 7, "test": 1}}
    assert split_imagenet_22k_images(images_folder=earlier_folder, held_out_images_per_class=1,
                                     minimum_held_out_class_size=5, dry_run=True) == expected_report
    assert list_split_files(images_folder=earlier_folder) == earlier_layout

    # Both layouts give the held-out image of the large class in test, every other image in train
    expected_layout = {"test": [f"n01440764/{held_out_name}"],
                       "train": sorted([f"n01440764/{file_name}" for file_name in other_large_names] +
                                       [f"n01443537/{file_name}" for file_name in small_class_names])}
    assert split_imagenet_22k_images(images_folder=earlier_folder, held_out_images_per_class=1,
                                     minimum_held_out_class_size=5, dry_run=False) == expected_report
    assert list_split_files(images_folder=earlier_folder) == expected_layout
    assert split_imagenet_22k_images(images_folder=fresh_folder, held_out_images_per_class=1,
                                     minimum_held_out_class_size=5, dry_run=False)["moved_images"] == {
        "train to test": 1}
    assert list_split_files(images_folder=fresh_folder) == expected_layout

    # A second run moves nothing
    second_report = split_imagenet_22k_images(images_folder=earlier_folder, held_out_images_per_class=1,
                                              minimum_held_out_class_size=5, dry_run=False)
    assert second_report["moved_images"] == {} and second_report["deleted_duplicates"] == 0
    with pytest.raises(expected_exception=ValueError, match="fewer than the minimum class size"):
        split_imagenet_22k_images(images_folder=earlier_folder, held_out_images_per_class=5,
                                  minimum_held_out_class_size=5, dry_run=True)


def test_build_imagenet_22k_splits(tmp_path: Path) -> None:
    """
    Verifies the splits built from stored images: the held-out images moved from the large class only, the split
    sizes checked against the shard records, the validation and test manifests describing the same images, the
    categories of every split, and the summary; then that a missing image stops the build.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # Stored images of a large class and a small one, with the records of their shards
    images_folder = tmp_path / "images"
    write_class_images(images_folder=images_folder, split_name="train", class_identifier="n01440764",
                       file_names=[f"n01440764_{image_index}.jpg" for image_index in range(6)])
    write_class_images(images_folder=images_folder, split_name="train", class_identifier="n01443537",
                       file_names=["n01443537_0.jpg", "n01443537_1.jpg"])
    records_folder = tmp_path / "progress" / "shards"
    records_folder.mkdir(parents=True)
    write_records(records_path=records_folder / "imagenet22k-train-0000.json",
                  records={"stored": ["image"] * 5, "undecodable": []})
    write_records(records_path=records_folder / "imagenet22k-validation-000.json",
                  records={"stored": ["image"] * 3, "undecodable": ["broken"]})

    build_imagenet_22k_splits(dataset_path=tmp_path, held_out_images_per_class=2, minimum_held_out_class_size=5)

    # Two held-out images from the large class only, read by both the validation and the test manifests
    assert len(list((images_folder / "test" / "n01440764").iterdir())) == 2
    assert not (images_folder / "test" / "n01443537").exists()
    manifests = {split_name: json.loads((tmp_path / "manifests" / split_name / "labels.json").read_text())
                 for split_name in ["train", "validation", "test"]}
    assert {split_name: len(manifest["samples"]) for split_name, manifest in manifests.items()} == {
        "train": 6, "validation": 2, "test": 2}
    assert manifests["validation"] == manifests["test"]
    for split_name in ["validation", "test"]:
        assert (tmp_path / "manifests" / split_name / "data").resolve() == (images_folder / "test").resolve()
    assert [category["name"] for category in manifests["validation"]["categories"]] == ["n01440764", "n01443537"]
    summary = json.loads((tmp_path / "reports" / "summary.json").read_text())
    assert summary == {"revision": imagenet_download.IMAGENET_22K_REVISION,
                       "shard_counts": {"shards": 2, "stored": 8, "undecodable": 1},
                       "split_sizes": {"train": 6, "test": 2}, "held_out_classes": 1,
                       "last_split_moves": {"train to test": 2}}

    # A training image lost after its shard was recorded stops the build
    next((images_folder / "train" / "n01443537").iterdir()).unlink()
    with pytest.raises(expected_exception=RuntimeError, match="The train split holds 5 images instead of 6"):
        build_imagenet_22k_splits(dataset_path=tmp_path, held_out_images_per_class=2, minimum_held_out_class_size=5)


def test_write_class_folder_manifests(tmp_path: Path) -> None:
    """
    Verifies the conversion of a dataset stored as one folder per class: every split gets a 'data' link to its class
    folders and a manifest of its images, the categories (the class folders of all the splits, in alphabetical order)
    are the same for every split even when a split lacks a class, the label mapping gives the classes 0 to K-1, and
    the classification dataset reads the converted split.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # Training split with two classes, validation split without circles
    for split_name, class_images in [("training_data", {"squares": 2, "circles": 1}),
                                     ("validation_data", {"squares": 1})]:
        for class_name, number_images in class_images.items():
            class_folder = tmp_path / split_name / class_name
            class_folder.mkdir(parents=True)
            for image_index in range(number_images):
                cv2.imwrite(filename=str(class_folder / f"image_{image_index}.png"),
                            img=np.zeros(shape=(16, 16, 3), dtype=np.uint8))
            (class_folder / "notes.txt").write_text(data="not an image")

    # Convert both splits, the files that are not images being left out of the manifests
    output_folder = tmp_path / "manifests"
    write_class_folder_manifests(split_folders={"train": str(tmp_path / "training_data"),
                                                "validation": str(tmp_path / "validation_data")},
                                 output_folder=str(output_folder))

    # Both splits share the categories of every class folder in alphabetical order, mapped to the classes 0 and 1
    train_labels = json.loads((output_folder / "train" / "labels.json").read_text())
    validation_labels = json.loads((output_folder / "validation" / "labels.json").read_text())
    assert train_labels["categories"] == validation_labels["categories"] == [{"id": 1, "name": "circles"},
                                                                             {"id": 2, "name": "squares"}]
    assert [sample["file_name"] for sample in train_labels["samples"]] == ["circles/image_0.png",
                                                                           "squares/image_0.png",
                                                                           "squares/image_1.png"]
    assert [sample["category_id"] for sample in validation_labels["samples"]] == [2]
    assert (output_folder / "train" / "data").is_symlink()
    assert json.loads((output_folder / "label_mapping.json").read_text()) == {"1": 0, "2": 1}

    # A second conversion replaces the links and manifests
    write_class_folder_manifests(split_folders={"train": str(tmp_path / "training_data")},
                                 output_folder=str(output_folder))

    dataset = ClassificationDataset(data_directory=str(output_folder / "train" / "data"),
                                    labels_file=str(output_folder / "train" / "labels.json"), image_size=16,
                                    keep_ratio=True, label_dictionary={1: 0, 2: 1}, output_mode="multi_class",
                                    shuffle=False)
    assert [int(dataset[sample_index]["labels"]) for sample_index in range(len(dataset))] == [0, 1, 1]
