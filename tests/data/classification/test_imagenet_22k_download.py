"""
Tests of vision_training/data/classification/imagenet_22k_download.py on small files written by the tests, without
network: the perceptual hash, the near copy index, a fake ImageNet-22k shard written with its original bytes and the
copies of an excluded image left out, the free space stop, and the splits, checks and manifests built from the stored
images.
"""

import io
import json
import tarfile

import cv2
import numpy as np
import pytest

from pathlib import Path
from typing import Dict

from vision_training.data.classification import imagenet_22k_download
from vision_training.data.classification.imagenet_22k_download import (
    build_hash_chunk_index, build_imagenet_22k_splits, compute_perceptual_hash, get_shard_split, hash_excluded_images,
    is_near_copy, process_imagenet_22k_shard, run_tasks_with_free_space_check, write_records)


def create_textured_image(seed: int, height: int = 96, width: int = 128) -> np.ndarray:
    """
    Creates a BGR image of smooth random blobs, detailed enough to get a perceptual hash, different for every seed.

    Args:
        seed (int): The seed of the random blobs.
        height (int): The image height. Defaults to 96.
        width (int): The image width. Defaults to 128.

    Returns:
        np.ndarray: The uint8 BGR image.
    """
    random_generator = np.random.default_rng(seed=seed)
    coarse_image = random_generator.integers(low=0, high=256, size=(6, 8, 3), dtype=np.uint8)
    return cv2.resize(src=coarse_image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)


def encode_image(image: np.ndarray, jpeg_quality: int = 95) -> bytes:
    """
    Encodes an image as JPEG, as the files of the shards are.

    Args:
        image (np.ndarray): The BGR image.
        jpeg_quality (int): The JPEG quality. Defaults to 95.

    Returns:
        bytes: The JPEG bytes.
    """
    return cv2.imencode(ext=".jpg", img=image, params=[cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1].tobytes()


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


def test_perceptual_hash_survives_reencoding() -> None:
    """
    Verifies that the hash of an image stays within the exclusion distance after a resize and a JPEG re-encoding,
    that two different images are far apart, and that a uniform image gets no hash.
    """
    image = create_textured_image(seed=0)
    image_hash = int(compute_perceptual_hash(image=image), 16)
    resized_bytes = encode_image(image=cv2.resize(src=image, dsize=(64, 48), interpolation=cv2.INTER_AREA),
                                 jpeg_quality=70)
    resized_image = cv2.imdecode(buf=np.frombuffer(buffer=resized_bytes, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    chunk_index = build_hash_chunk_index(hash_values=[image_hash])
    assert is_near_copy(hash_value=int(compute_perceptual_hash(image=resized_image), 16), chunk_index=chunk_index)
    assert not is_near_copy(hash_value=int(compute_perceptual_hash(image=create_textured_image(seed=1)), 16),
                            chunk_index=chunk_index)
    assert compute_perceptual_hash(image=np.full(shape=(32, 32, 3), fill_value=128, dtype=np.uint8)) is None


def test_near_copy_index() -> None:
    """
    Verifies that a hash within EXCLUSION_MAXIMUM_DISTANCE bits of an excluded hash is a copy, even with a differing bit
    in every chunk but one, and that a hash one bit further is not.
    """
    excluded_hash = int("0123456789abcdef0123456789abcdef", 16)
    chunk_index = build_hash_chunk_index(hash_values=[excluded_hash])

    # Three flipped bits in three different chunks of 32 bits, then a fourth one in the last chunk
    three_bit_copy = excluded_hash ^ (1 << 0) ^ (1 << 40) ^ (1 << 80)
    assert is_near_copy(hash_value=three_bit_copy, chunk_index=chunk_index)
    assert not is_near_copy(hash_value=three_bit_copy ^ (1 << 120), chunk_index=chunk_index)


def test_get_shard_split() -> None:
    """
    Verifies that the training shards give the train split, the validation shards the test split, and that an unknown
    shard is refused.
    """
    assert get_shard_split(shard_file_name="imagenet22k-train-0000.tar") == "train"
    assert get_shard_split(shard_file_name="imagenet22k-validation-0511.tar") == "test"
    with pytest.raises(expected_exception=ValueError, match="none of the prefixes"):
        get_shard_split(shard_file_name="imagenet12k-train-0000.tar")


def test_process_imagenet_22k_shard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that a shard writes the original bytes of its images into the class folder of its split, leaves out a
    re-encoded copy of an excluded image and an undecodable image, records all three kinds, and is deleted once
    processed.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): Replaces the Hugging Face download by a local shard.
    """
    # Excluded image, hashed from its own folder as the ImageNet-1k test split is
    excluded_image = create_textured_image(seed=5)
    excluded_folder = tmp_path / "excluded" / "n01440764"
    excluded_folder.mkdir(parents=True)
    (excluded_folder / "ILSVRC2012_val_00000001.JPEG").write_bytes(data=encode_image(image=excluded_image))
    hashes_path = tmp_path / "excluded_image_hashes.json"
    assert hash_excluded_images(excluded_images_folder=tmp_path / "excluded", hashes_path=hashes_path,
                                number_workers=1) == 1

    # Shard of two kept images, a re-encoded copy of the excluded image and an undecodable image
    kept_bytes = {"n01440764_1": encode_image(image=create_textured_image(seed=0)),
                  "n09999999_2": encode_image(image=create_textured_image(seed=1))}
    downloaded_shard_path = tmp_path / "downloads" / "imagenet22k-validation-0000.tar"
    downloaded_shard_path.parent.mkdir()
    write_tar_shard(shard_path=downloaded_shard_path,
                    members={"n01440764_1.jpg": kept_bytes["n01440764_1"], "n01440764_1.cls": b"0",
                             "n01440764_1.json": b'{"label": 0, "class_name": "n01440764"}',
                             "n09999999_2.jpg": kept_bytes["n09999999_2"],
                             "n01440764_3.jpg": encode_image(image=excluded_image, jpeg_quality=60),
                             "n01440764_4.jpg": b"broken"})
    monkeypatch.setattr(target=imagenet_22k_download, name="hf_hub_download",
                        value=lambda **_: str(downloaded_shard_path))

    records_path = tmp_path / "records.json"
    number_images = process_imagenet_22k_shard(shard_file_name="imagenet22k-validation-0000.tar",
                                               downloads_folder=tmp_path / "downloads",
                                               images_folder=tmp_path / "images", records_path=records_path,
                                               excluded_hashes_path=hashes_path)

    # Original bytes in the test split, the copy and the broken image recorded but not stored
    assert number_images == 2
    for imagenet_key, image_bytes in kept_bytes.items():
        image_path = tmp_path / "images" / "test" / imagenet_key.split("_")[0] / f"{imagenet_key}.jpg"
        assert image_path.read_bytes() == image_bytes
    assert len(list((tmp_path / "images").rglob(pattern="*.jpg"))) == 2
    assert json.loads(records_path.read_text()) == {"split": "test", "stored": ["n01440764_1", "n09999999_2"],
                                                    "excluded": ["n01440764_3"], "undecodable": ["n01440764_4"]}
    assert not downloaded_shard_path.exists()


def test_run_tasks_stops_without_free_space(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that no task is started when the disk has less free space than the minimum, the error telling how many
    tasks are left for the next run.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): Reports a nearly full disk.
    """
    monkeypatch.setattr(target=imagenet_22k_download, name="get_free_gigabytes", value=lambda **_: 50.0)
    with pytest.raises(expected_exception=RuntimeError, match="2 tasks of 'Shards' were not started"):
        run_tasks_with_free_space_check(task_function=write_records, task_arguments=[{}, {}], disk_folder=tmp_path,
                                        minimum_free_gigabytes=100, number_workers=1, description="Shards")


def test_build_imagenet_22k_splits(tmp_path: Path) -> None:
    """
    Verifies the splits built from stored images: the validation images held out from the large classes only, the
    split sizes checked against the shard records, the manifests with the categories of every split, and the summary;
    then that a missing image stops the build.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # Stored images: a large training class, a small one, and one test class, with the records of their shards
    class_sizes = {("train", "n01440764"): 6, ("train", "n01443537"): 2, ("test", "n01440764"): 3}
    for (split_name, class_identifier), class_size in class_sizes.items():
        class_folder = tmp_path / "images" / split_name / class_identifier
        class_folder.mkdir(parents=True)
        for image_index in range(class_size):
            (class_folder / f"{class_identifier}_{image_index}.jpg").write_bytes(data=b"image")
    records_folder = tmp_path / "progress" / "shards"
    records_folder.mkdir(parents=True)
    write_records(records_path=records_folder / "imagenet22k-train-0000.json",
                  records={"split": "train", "stored": ["image"] * 8, "excluded": ["copy"], "undecodable": []})
    write_records(records_path=records_folder / "imagenet22k-validation-0000.json",
                  records={"split": "test", "stored": ["image"] * 3, "excluded": [], "undecodable": ["broken"]})

    build_imagenet_22k_splits(dataset_path=tmp_path, validation_images_per_class=2, minimum_validation_class_size=5)

    # Two validation images from the large class only, the small class keeping both its images
    assert len(list((tmp_path / "images" / "validation" / "n01440764").iterdir())) == 2
    assert not (tmp_path / "images" / "validation" / "n01443537").exists()
    manifests = {split_name: json.loads((tmp_path / "manifests" / split_name / "labels.json").read_text())
                 for split_name in ["train", "validation", "test"]}
    split_sizes = {split_name: len(manifest["samples"]) for split_name, manifest in manifests.items()}
    assert split_sizes == {"train": 6, "validation": 2, "test": 3}
    assert [category["name"] for category in manifests["validation"]["categories"]] == ["n01440764", "n01443537"]
    summary = json.loads((tmp_path / "reports" / "summary.json").read_text())
    assert summary == {"shard_counts": {"train": {"stored": 8, "excluded": 1, "undecodable": 0},
                                        "test": {"stored": 3, "excluded": 0, "undecodable": 1}},
                       "validation_images": 2}

    # A training image lost after its shard was recorded stops the build
    next((tmp_path / "images" / "train" / "n01443537").iterdir()).unlink()
    with pytest.raises(expected_exception=RuntimeError, match="The train split holds 5 images instead of 6"):
        build_imagenet_22k_splits(dataset_path=tmp_path, validation_images_per_class=2,
                                  minimum_validation_class_size=5)
