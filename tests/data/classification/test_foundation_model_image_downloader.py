"""
Tests of vision_training/data/classification/foundation_model_image_downloader.py on small files written by the tests,
without network: the parsing of the Tencent files, the image preparation and its perceptual hash, a fake ImageNet-22k
shard, the Open Images matching and batch with a local copy, the free space stop, the duplicate merging, the excluded
copies and the multi-label manifests.
"""

import csv
import io
import json
import tarfile

import cv2
import numpy as np
import pytest

from pathlib import Path
from typing import Dict, List

from vision_training.data.classification import foundation_model_image_downloader
from vision_training.data.classification.foundation_model_image_downloader import (
    build_hash_chunk_index, build_manifests, compute_perceptual_hash, encode_imagenet_key, get_flickr_photo_number,
    get_imagenet_key, is_near_copy, match_open_images, merge_duplicate_samples, parse_annotation_line, prepare_image,
    process_imagenet_shard, process_open_images_archive, process_open_images_batch, read_category_dictionary,
    resize_image, rotate_image,
    run_tasks_with_free_space_check, write_imagenet_wanted_keys, write_records)

# Category dictionary in the format of ML-Images, reduced to four categories, two of them sharing a first synonym
DICTIONARY_CONTENT = ("category_index\tcategory_id\tindex_of_parent_category\tcategory name\n"
                      "0\tn00002452\t-1\tthing\n"
                      "1\tn00002684\t-1\tobject, physical object\n"
                      "2\tn02012849\t1\tcrane\n"
                      "3\tn03126707\t1\tcrane\n")


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
    Encodes an image as JPEG, as the downloaded files are.

    Args:
        image (np.ndarray): The BGR image.
        jpeg_quality (int): The JPEG quality. Defaults to 95.

    Returns:
        bytes: The JPEG bytes.
    """
    return cv2.imencode(ext=".jpg", img=image, params=[cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1].tobytes()


def write_tar_shard(shard_path: Path, members: Dict[str, bytes], mode: str = "w") -> None:
    """
    Writes a tar shard in the webdataset layout of ImageNet-22k ('<key>.jpg', '<key>.cls'), or a gzip archive in the
    layout of the Open Images training archives ('train_0/<image identifier>.jpg').

    Args:
        shard_path (Path): The tar file to write.
        members (Dict[str, bytes]): The content of every member, keyed by member name.
        mode (str): The tarfile writing mode, "w:gz" for a gzip archive. Defaults to "w".
    """
    with tarfile.open(name=shard_path, mode=mode) as shard:
        for member_name, member_bytes in members.items():
            member_information = tarfile.TarInfo(name=member_name)
            member_information.size = len(member_bytes)
            shard.addfile(tarinfo=member_information, fileobj=io.BytesIO(initial_bytes=member_bytes))


def test_read_category_dictionary(tmp_path: Path) -> None:
    """
    Verifies that the categories are numbered from 1 in index order, named after their first synonym, with the
    WordNet identifier appended to the names shared by several categories, and that a gap in the indices is refused.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    dictionary_path = tmp_path / "dictionary.txt"
    dictionary_path.write_text(data=DICTIONARY_CONTENT)
    categories = read_category_dictionary(dictionary_path=dictionary_path)
    assert [category["id"] for category in categories] == [1, 2, 3, 4]
    assert [category["name"] for category in categories] == ["thing", "object", "crane (n02012849)",
                                                             "crane (n03126707)"]

    # The last line numbered 5 instead of 3
    dictionary_path.write_text(data=DICTIONARY_CONTENT.replace("3\tn03126707", "5\tn03126707"))
    with pytest.raises(expected_exception=ValueError, match="do not follow each other"):
        read_category_dictionary(dictionary_path=dictionary_path)


def test_parse_annotation_line() -> None:
    """
    Verifies that the labels with a confidence (training lists) and the bare labels (validation lists, confidence 1)
    are both read.
    """
    assert parse_annotation_line(line="https://a/b/1_c_o.jpg\t5193:0.9\t3:1\n") == ("https://a/b/1_c_o.jpg",
                                                                                     [5193, 3], [0.9, 1.0])
    assert parse_annotation_line(line="n04162706/n04162706_17303.JPEG\t3371\t2\n") == (
        "n04162706/n04162706_17303.JPEG", [3371, 2], [1.0, 1.0])


def test_image_identifiers() -> None:
    """
    Verifies the ImageNet key of a Tencent image, its integer encoding (None for another form), and the Flickr photo
    number of the Tencent and Open Images URLs.
    """
    assert get_imagenet_key(image_identifier="n04310904/n04310904_8388.JPEG") == "n04310904_8388"
    assert encode_imagenet_key(imagenet_key="n04310904_8388") == 4310904 * 10 ** 10 + 8388
    assert encode_imagenet_key(imagenet_key="ILSVRC2012_val_00000001") is None
    assert encode_imagenet_key(imagenet_key="n04310904_abc") is None
    assert get_flickr_photo_number(url="https://c4.staticflickr.com/8/7239/6997334729_e5fb3938b1_o.jpg") == \
        get_flickr_photo_number(url="https://farm8.staticflickr.com/7239/6997334729_e5fb3938b1_o.jpg") == "6997334729"


def test_rotate_and_resize_image() -> None:
    """
    Verifies that the rotation is counterclockwise, and that the resizing brings the shorter side to its maximum
    keeping the aspect ratio, without enlarging a smaller image.
    """
    # A white top-right pixel ends at the top left after 90 degrees counterclockwise, at the bottom right after 270
    image = np.zeros(shape=(4, 6, 3), dtype=np.uint8)
    image[0, 5] = 255
    assert rotate_image(image=image, rotation_degrees=90)[0, 0, 0] == 255
    assert rotate_image(image=image, rotation_degrees=270)[5, 3, 0] == 255
    assert rotate_image(image=image, rotation_degrees=0) is image

    assert resize_image(image=np.zeros(shape=(1000, 1500, 3), dtype=np.uint8), shorter_side=512).shape == (512, 768, 3)
    assert resize_image(image=np.zeros(shape=(300, 400, 3), dtype=np.uint8), shorter_side=512).shape == (300, 400, 3)


def test_perceptual_hash_and_prepare_image() -> None:
    """
    Verifies that the perceptual hash of an image survives its re-encoding and resizing, differs between images, is
    None for a uniform image, and that prepare_image stores a rotated, resized JPEG and refuses undecodable bytes.
    """
    image = create_textured_image(seed=0, height=600, width=800)
    image_hash = compute_perceptual_hash(image=image)
    assert len(image_hash) == 32

    # The same picture smaller and re-encoded keeps its hash within the exclusion distance, another picture does not
    copy_image = cv2.imdecode(buf=np.frombuffer(buffer=encode_image(image=cv2.resize(src=image, dsize=(400, 300)),
                                                                    jpeg_quality=80), dtype=np.uint8),
                              flags=cv2.IMREAD_COLOR)
    copy_distance = (int(image_hash, 16) ^ int(compute_perceptual_hash(image=copy_image), 16)).bit_count()
    other_distance = (int(image_hash, 16) ^
                      int(compute_perceptual_hash(image=create_textured_image(seed=1, height=600, width=800)),
                          16)).bit_count()
    assert copy_distance <= foundation_model_image_downloader.EXCLUSION_MAXIMUM_DISTANCE < other_distance
    assert compute_perceptual_hash(image=np.full(shape=(64, 64, 3), fill_value=128, dtype=np.uint8)) is None

    # Rotated by 90 degrees then resized: the 800 x 600 image becomes 512 wide and 683 high
    image_bytes, perceptual_hash = prepare_image(image_bytes=encode_image(image=image), shorter_side=512,
                                                 jpeg_quality=85, rotation_degrees=90)
    stored_image = cv2.imdecode(buf=np.frombuffer(buffer=image_bytes, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    assert stored_image.shape == (683, 512, 3) and perceptual_hash is not None
    assert prepare_image(image_bytes=b"not an image", shorter_side=512, jpeg_quality=85,
                         rotation_degrees=0) == (None, None)


def test_process_imagenet_shard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that a shard writes only the images of the Tencent lists into the folder of their WordNet identifier,
    records them with their hash, lists its undecodable images as missing, and is deleted once processed.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): Replaces the Hugging Face download by a local shard.
    """
    # Tencent lists naming two images of the shard and an image missing from it
    annotations_folder = tmp_path / "annotations"
    annotations_folder.mkdir()
    (annotations_folder / "train_image_id_from_imagenet.txt").write_text(
        data="n01440764/n01440764_1.JPEG\t2:1\nn01440764/n01440764_3.JPEG\t2:1\n")
    (annotations_folder / "val_image_id_from_imagenet.txt").write_text(
        data="n01443537/n01443537_2.JPEG\t2\nn01443537/n01443537_9.JPEG\t2\n")
    wanted_keys_path = tmp_path / "wanted_keys.npy"
    assert write_imagenet_wanted_keys(annotations_folder=annotations_folder, wanted_keys_path=wanted_keys_path) == 4

    # Shard of two wanted images, an unwanted image and an undecodable wanted image
    downloaded_shard_path = tmp_path / "downloads" / "imagenet22k-train-0000.tar"
    downloaded_shard_path.parent.mkdir()
    write_tar_shard(shard_path=downloaded_shard_path,
                    members={"n01440764_1.jpg": encode_image(image=create_textured_image(seed=0)),
                             "n01440764_1.cls": b"0",
                             "n01443537_2.jpg": encode_image(image=create_textured_image(seed=1)),
                             "n01440764_7.jpg": encode_image(image=create_textured_image(seed=2)),
                             "n01440764_3.jpg": b"broken"})
    monkeypatch.setattr(target=foundation_model_image_downloader, name="hf_hub_download",
                        value=lambda **_: str(downloaded_shard_path))

    records_path = tmp_path / "records.json"
    number_images = process_imagenet_shard(shard_file_name="imagenet22k-train-0000.tar",
                                           downloads_folder=tmp_path / "downloads", wanted_keys_path=wanted_keys_path,
                                           images_folder=tmp_path / "images", records_path=records_path,
                                           shorter_side=64, jpeg_quality=85)
    assert number_images == 2
    assert sorted(path.relative_to(tmp_path / "images").as_posix()
                  for path in (tmp_path / "images").rglob(pattern="*.jpg")) == \
        ["imagenet/n01440764/n01440764_1.jpg", "imagenet/n01443537/n01443537_2.jpg"]
    records = json.loads(records_path.read_text())
    assert sorted(record[0] for record in records["records"]) == ["n01440764_1", "n01443537_2"]
    assert records["missing"] == ["n01440764_3"]
    assert not downloaded_shard_path.exists()


def test_match_open_images(tmp_path: Path) -> None:
    """
    Verifies that the Tencent URLs are matched to the Open Images metadata by photo number across Flickr hosts, with
    their subset and rotation (0 when unknown), and that a photo no longer in Open Images gets no match.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    annotations_folder = tmp_path / "annotations"
    annotations_folder.mkdir()
    (annotations_folder / "train_urls_from_openimages.txt").write_text(
        data="https://c4.staticflickr.com/8/1/111_aa_o.jpg\t3:1\nhttps://c1.staticflickr.com/8/1/999_bb_o.jpg\t3:1\n")
    (annotations_folder / "val_urls_from_openimages.txt").write_text(
        data="https://c2.staticflickr.com/4/1/222_cc_o.jpg\t3\n")

    # Metadata of every subset with the columns used by the matching
    metadata_rows = {"train": [["aaaaaaaaaaaaaaaa", "train", "https://farm1.staticflickr.com/1/111_aa_o.jpg", "90.0"],
                               ["bbbbbbbbbbbbbbbb", "train", "https://farm1.staticflickr.com/1/333_dd_o.jpg", "0.0"]],
                     "validation": [["cccccccccccccccc", "validation", "https://farm1.staticflickr.com/1/222_cc_o.jpg",
                                     ""]],
                     "test": []}
    for subset_name, rows in metadata_rows.items():
        with open(file=annotations_folder / f"open_images_{subset_name}_metadata.csv", mode="w",
                  newline="") as metadata_file:
            metadata_writer = csv.writer(metadata_file)
            metadata_writer.writerow(["ImageID", "Subset", "OriginalURL", "Rotation"])
            metadata_writer.writerows(rows)

    matches_path = tmp_path / "matches.csv"
    assert match_open_images(annotations_folder=annotations_folder, matches_path=matches_path) == 2
    assert foundation_model_image_downloader.read_open_images_matches(matches_path=matches_path) == {
        "111": ("train", "aaaaaaaaaaaaaaaa", 90), "222": ("validation", "cccccccccccccccc", 0)}
    # The existing matches are read back instead of computed again
    assert match_open_images(annotations_folder=annotations_folder, matches_path=matches_path) == 2


def test_process_open_images_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that a batch reads the images of the local Open Images copy before the bucket, stores them rotated in the
    folder of their first characters, and lists the images missing from the bucket.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): Replaces the S3 download by a bucket without the image.
    """
    # Local copy holding the first image, the second one only being looked for in the bucket
    local_folder = tmp_path / "open-images-v7"
    (local_folder / "train" / "data").mkdir(parents=True)
    (local_folder / "train" / "data" / "aaaaaaaaaaaaaaaa.jpg").write_bytes(
        data=encode_image(image=create_textured_image(seed=0, height=60, width=80)))
    monkeypatch.setattr(target=foundation_model_image_downloader, name="fetch_open_images_object",
                        value=lambda **_: None)

    records_path = tmp_path / "batch_00000.json"
    number_images = process_open_images_batch(image_entries=[("train", "aaaaaaaaaaaaaaaa", 90),
                                                             ("train", "bbbbbbbbbbbbbbbb", 0)],
                                              local_open_images_folder=local_folder, images_folder=tmp_path / "images",
                                              records_path=records_path, shorter_side=512, jpeg_quality=85,
                                              number_download_threads=2)
    assert number_images == 1
    stored_image = cv2.imread(filename=str(tmp_path / "images" / "openimages" / "aa" / "aaaaaaaaaaaaaaaa.jpg"))
    assert stored_image.shape == (80, 60, 3)
    assert json.loads(records_path.read_text())["missing"] == ["bbbbbbbbbbbbbbbb"]


class FakeArchiveClient:
    """
    S3 client serving a local gzip archive as the streamed body of every requested object.
    """

    def __init__(self, archive_path: Path) -> None:
        """
        Keeps the archive served by get_object.

        Args:
            archive_path (Path): The local gzip archive.
        """
        self.archive_path = archive_path

    def get_object(self, Bucket: str, Key: str) -> Dict[str, io.BytesIO]:  # noqa: N803 (boto3 argument names)
        """
        Serves the archive as the body of the object.

        Args:
            Bucket (str): The bucket name, ignored.
            Key (str): The object key, ignored.

        Returns:
            Dict[str, io.BytesIO]: The body of the object.
        """
        return {"Body": io.BytesIO(initial_bytes=self.archive_path.read_bytes())}


def test_process_open_images_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that an archive stores only its wanted images, rotated, skips the images of a previous run's progress
    file, lists the wanted images it does not hold as missing, and replaces its progress file by its records.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): Replaces the S3 client by a local archive.
    """
    # Archive of two wanted images and an unwanted one, the image 'a1' being already stored by a previous run
    archive_path = tmp_path / "train_0.tar.gz"
    write_tar_shard(shard_path=archive_path, mode="w:gz",
                    members={"train_0/a1.jpg": encode_image(image=create_textured_image(seed=0, height=60, width=80)),
                             "train_0/a2.jpg": encode_image(image=create_textured_image(seed=1, height=60, width=80)),
                             "train_0/a3.jpg": encode_image(image=create_textured_image(seed=2, height=60, width=80))})
    monkeypatch.setattr(target=foundation_model_image_downloader, name="create_s3_client",
                        value=lambda **_: FakeArchiveClient(archive_path=archive_path))
    records_path = tmp_path / "train_0.json"
    (tmp_path / "train_0.progress.jsonl").write_text(data='["a1", null]\n')

    number_images = process_open_images_archive(archive_key="tar/train_0.tar.gz",
                                                wanted_rotations={"a1": 0, "a2": 90, "a9": 0},
                                                images_folder=tmp_path / "images", records_path=records_path,
                                                shorter_side=512, jpeg_quality=85, disk_folder=tmp_path,
                                                minimum_free_gigabytes=0)
    assert number_images == 1
    assert sorted(path.name for path in (tmp_path / "images").rglob(pattern="*.jpg")) == ["a2.jpg"]
    assert cv2.imread(filename=str(tmp_path / "images" / "openimages" / "a2" / "a2.jpg")).shape == (80, 60, 3)
    records = json.loads(records_path.read_text())
    assert [record[0] for record in records["records"]] == ["a1", "a2"] and records["missing"] == ["a9"]
    assert not (tmp_path / "train_0.progress.jsonl").exists()


def test_run_tasks_stops_without_free_space(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that no task is started when the disk has less free space than the minimum, the error telling how many
    tasks are left for the next run.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): Reports a nearly full disk.
    """
    monkeypatch.setattr(target=foundation_model_image_downloader, name="get_free_gigabytes", value=lambda **_: 50.0)
    with pytest.raises(expected_exception=RuntimeError, match="2 tasks of 'Shards' were not started"):
        run_tasks_with_free_space_check(task_function=write_records, task_arguments=[{}, {}], disk_folder=tmp_path,
                                        minimum_free_gigabytes=100, number_workers=1, description="Shards")


def test_merge_duplicate_samples() -> None:
    """
    Verifies that the copies of a test image are removed from the training images, that the copies within a split
    become one sample with all their labels at their highest confidence, and that samples without hash are kept.
    """
    samples = [["train", "a.jpg", "3:1\t5:0.8", 7], ["train", "b.jpg", "5:1\t6:0.9", 7],
               ["train", "c.jpg", "4:1", 9], ["test", "d.jpg", "4\t2", 9],
               ["train", "e.jpg", "1:1", None], ["train", "f.jpg", "1:1", None]]
    kept_samples, number_removed = merge_duplicate_samples(samples=samples)
    assert number_removed == 2
    assert [sample[1] for sample in kept_samples] == ["a.jpg", "d.jpg", "e.jpg", "f.jpg"]
    assert parse_annotation_line(line=f"image {kept_samples[0][2]}")[1:] == ([3, 5, 6], [1.0, 1.0, 0.9])
    assert kept_samples[1][2] == "4:1\t2:1"


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


def write_records_folder(records_folder: Path, records: List[List[str]], missing_identifiers: List[str]) -> None:
    """
    Writes the records file of one finished task, as the downloads do.

    Args:
        records_folder (Path): The folder of the records of a source.
        records (List[List[str]]): The identifier and the perceptual hash of every stored image.
        missing_identifiers (List[str]): The identifiers of the images the task could not get.
    """
    records_folder.mkdir(parents=True, exist_ok=True)
    write_records(records_path=records_folder / "task_0.json", records=records, missing_identifiers=missing_identifiers)


def test_build_manifests(tmp_path: Path) -> None:
    """
    Verifies the manifests built from stored images: the splits with the validation images held out from the training
    images, the categories and label mapping, a duplicate merged, the copy of an excluded image left out, and the
    images of the lists not stored reported.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # Annotation files of a dataset of five ImageNet and two Open Images images
    annotations_folder = tmp_path / "annotations"
    annotations_folder.mkdir()
    (annotations_folder / "dictionary_and_semantic_hierarchy.txt").write_text(data=DICTIONARY_CONTENT)
    (annotations_folder / "train_image_id_from_imagenet.txt").write_text(
        data="".join(f"n01440764/n01440764_{image_number}.JPEG\t2:1\t1:1\t0:1\n" for image_number in range(1, 6)))
    (annotations_folder / "val_image_id_from_imagenet.txt").write_text(data="n01443537/n01443537_1.JPEG\t3\t1\t0\n")
    (annotations_folder / "train_urls_from_openimages.txt").write_text(
        data="https://c4.staticflickr.com/8/1/111_aa_o.jpg\t3:0.9\t1:0.9\n"
             "https://c4.staticflickr.com/8/1/555_ee_o.jpg\t3:1\n")
    (annotations_folder / "val_urls_from_openimages.txt").write_text(data="")
    progress_folder = tmp_path / "progress"
    progress_folder.mkdir()
    (progress_folder / "open_images_matches.csv").write_text(
        data="photo_number,subset,image_identifier,rotation\n111,train,aaaaaaaaaaaaaaaa,0\n")

    # Stored images: ImageNet image 5 is a copy of image 1, image 4 a copy of an excluded image, image 3 was not stored
    excluded_hash = "0f" * 16
    write_records_folder(records_folder=progress_folder / "imagenet_shards",
                         records=[["n01440764_1", "01" * 16], ["n01440764_2", "02" * 16],
                                  ["n01440764_4", excluded_hash], ["n01440764_5", "01" * 16],
                                  ["n01443537_1", "03" * 16]], missing_identifiers=[])
    write_records_folder(records_folder=progress_folder / "open_images_batches",
                         records=[["aaaaaaaaaaaaaaaa", None]], missing_identifiers=[])
    (tmp_path / "images").mkdir()

    build_manifests(dataset_path=tmp_path, excluded_hashes=[int(excluded_hash, 16) ^ 1], validation_size=1)

    # Manifests of every split, the training images split between train and validation
    manifests = {split_name: json.loads((tmp_path / "manifests" / split_name / "labels.json").read_text())
                 for split_name in ["train", "validation", "test"]}
    training_file_names = sorted(sample["file_name"] for split_name in ["train", "validation"]
                                 for sample in manifests[split_name]["samples"])
    assert training_file_names == ["imagenet/n01440764/n01440764_1.jpg", "imagenet/n01440764/n01440764_2.jpg",
                                   "openimages/aa/aaaaaaaaaaaaaaaa.jpg"]
    assert len(manifests["validation"]["samples"]) == 1
    assert manifests["test"]["samples"] == [{"id": 1, "file_name": "imagenet/n01443537/n01443537_1.jpg",
                                             "category_ids": [4, 2, 1], "category_confidences": [1.0, 1.0, 1.0]}]
    assert [category["id"] for category in manifests["train"]["categories"]] == [1, 2, 3, 4]
    assert (tmp_path / "manifests" / "train" / "data").resolve() == (tmp_path / "images").resolve()
    assert json.loads((tmp_path / "manifests" / "label_mapping.json").read_text())["4"] == 3

    # Counts of the summary and images of the lists left out
    summary = json.loads((tmp_path / "reports" / "summary.json").read_text())
    assert (summary["merged_duplicates"], summary["excluded_copies"]) == (1, 1)
    assert (tmp_path / "reports" / "unavailable_train_image_id_from_imagenet.txt").read_text() == \
        "n01440764/n01440764_3.JPEG"
    assert (tmp_path / "reports" / "unavailable_train_urls_from_openimages.txt").read_text() == \
        "https://c4.staticflickr.com/8/1/555_ee_o.jpg"
