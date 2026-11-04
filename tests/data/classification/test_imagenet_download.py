"""
Tests of vision_training/data/classification/imagenet_download.py on small parquet shards written by the tests: the
extraction of the shards into class folders and the validation images held out by file name hash.
"""

import pyarrow
import pyarrow.parquet as parquet
import pytest

from pathlib import Path
from typing import Any, Dict, List

from vision_training.data.classification.imagenet_download import (compute_file_name_hash, extract_imagenet_shards,
                                                                   extract_shard, hold_out_validation_images)

# WordNet identifiers of the two classes of the test shards, in class index order
CLASS_IDENTIFIERS = ["n01440764", "n01443537"]


def write_shard(shard_path: Path, rows: List[Dict[str, Any]]) -> None:
    """
    Writes a parquet shard with the columns of the Hugging Face ImageNet-1k dataset: 'image' (JPEG 'bytes' and file
    'path') and 'label' (class index).

    Args:
        shard_path (Path): The parquet file to write.
        rows (List[Dict[str, Any]]): The 'bytes', 'path' and 'label' of every row.
    """
    image_type = pyarrow.struct([("bytes", pyarrow.binary()), ("path", pyarrow.string())])
    images = pyarrow.array([{"bytes": row["bytes"], "path": row["path"]} for row in rows], type=image_type)
    labels = pyarrow.array([row["label"] for row in rows], type=pyarrow.int64())
    parquet.write_table(table=pyarrow.table({"image": images, "label": labels}), where=shard_path)


def test_extract_shard(tmp_path: Path) -> None:
    """
    Verifies that the images of a shard are written unchanged into the class folder of their label, under their
    original file name or a name built from the shard when they have none, that the shard is marked as extracted, and
    that a label outside the classes or a training file name of another class than its label is refused.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # Class folders of the split, and a shard of two training images, an official validation image and an image
    # without a file name
    split_folder = tmp_path / "train"
    for class_identifier in CLASS_IDENTIFIERS:
        (split_folder / class_identifier).mkdir(parents=True)
    shard_path = tmp_path / "train-00000-of-00001.parquet"
    marker_path = tmp_path / "train-00000-of-00001.extracted"
    write_shard(shard_path=shard_path, rows=[{"bytes": b"first", "path": "n01440764_10026.JPEG", "label": 0},
                                             {"bytes": b"second", "path": "n01443537_7.JPEG", "label": 1},
                                             {"bytes": b"third", "path": "ILSVRC2012_val_00000001.JPEG", "label": 1},
                                             {"bytes": b"fourth", "path": None, "label": 0}])

    number_images = extract_shard(shard_path=shard_path, split_folder=split_folder,
                                  class_identifiers=CLASS_IDENTIFIERS, marker_path=marker_path)

    # Every image in the folder of its label, with the same bytes
    assert number_images == 4
    assert (split_folder / "n01440764" / "n01440764_10026.JPEG").read_bytes() == b"first"
    assert (split_folder / "n01443537" / "n01443537_7.JPEG").read_bytes() == b"second"
    assert (split_folder / "n01443537" / "ILSVRC2012_val_00000001.JPEG").read_bytes() == b"third"
    assert (split_folder / "n01440764" / "train-00000-of-00001_000003.JPEG").read_bytes() == b"fourth"
    assert marker_path.exists()

    # A label outside the class indices
    write_shard(shard_path=shard_path, rows=[{"bytes": b"fifth", "path": "n01440764_8.JPEG", "label": 2}])
    with pytest.raises(expected_exception=ValueError, match="is not a class index"):
        extract_shard(shard_path=shard_path, split_folder=split_folder, class_identifiers=CLASS_IDENTIFIERS,
                      marker_path=marker_path)

    # A training image whose file name gives another class than its label
    write_shard(shard_path=shard_path, rows=[{"bytes": b"sixth", "path": "n01443537_8.JPEG", "label": 0}])
    with pytest.raises(expected_exception=ValueError, match="is labelled n01440764"):
        extract_shard(shard_path=shard_path, split_folder=split_folder, class_identifiers=CLASS_IDENTIFIERS,
                      marker_path=marker_path)


def test_extract_imagenet_shards(tmp_path: Path) -> None:
    """
    Verifies that the worker processes extract the shards into the class folders of their split, the official
    validation shards going to the test split, that every class folder is created, and that the shards already
    extracted are skipped by a second extraction.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # One training shard and one official validation shard of one image each
    (tmp_path / "parquet").mkdir()
    train_shard_path = tmp_path / "parquet" / "train-00000-of-00001.parquet"
    validation_shard_path = tmp_path / "parquet" / "validation-00000-of-00001.parquet"
    write_shard(shard_path=train_shard_path, rows=[{"bytes": b"train", "path": "n01440764_1.JPEG", "label": 0}])
    write_shard(shard_path=validation_shard_path,
                rows=[{"bytes": b"validation", "path": "ILSVRC2012_val_00000001.JPEG", "label": 1}])
    shard_paths = {"train": [train_shard_path], "validation": [validation_shard_path]}
    images_folder = tmp_path / "images"
    markers_folder = tmp_path / "extracted_shards"

    extract_imagenet_shards(shard_paths=shard_paths, images_folder=images_folder, markers_folder=markers_folder,
                            class_identifiers=CLASS_IDENTIFIERS, number_workers=2)

    # The training image in the train split, the official validation image in the test split
    train_image_path = images_folder / "train" / "n01440764" / "n01440764_1.JPEG"
    assert train_image_path.read_bytes() == b"train"
    assert (images_folder / "test" / "n01443537" / "ILSVRC2012_val_00000001.JPEG").read_bytes() == b"validation"
    assert sorted(path.name for path in (images_folder / "test").iterdir()) == CLASS_IDENTIFIERS

    # The marked shards are skipped: an image deleted after the first extraction is not written again
    train_image_path.unlink()
    extract_imagenet_shards(shard_paths=shard_paths, images_folder=images_folder, markers_folder=markers_folder,
                            class_identifiers=CLASS_IDENTIFIERS, number_workers=2)
    assert not train_image_path.exists()


def test_hold_out_validation_images(tmp_path: Path) -> None:
    """
    Verifies that the training images with the smallest file name hashes of every class are moved to the validation
    split, that an interrupted hold-out is completed with the same images, that running it again moves nothing, and
    that a training copy of a validation image is deleted.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # Six training images per class, and the two of every class expected in the validation split
    train_folder = tmp_path / "train"
    validation_folder = tmp_path / "validation"
    class_file_names = {class_identifier: [f"{class_identifier}_{image_index}.JPEG" for image_index in range(6)]
                        for class_identifier in CLASS_IDENTIFIERS}
    for class_identifier, file_names in class_file_names.items():
        (train_folder / class_identifier).mkdir(parents=True)
        for file_name in file_names:
            (train_folder / class_identifier / file_name).write_bytes(data=file_name.encode())
    expected_validation_names = {class_identifier: sorted(file_names, key=compute_file_name_hash)[:2]
                                 for class_identifier, file_names in class_file_names.items()}

    # Interrupted hold-out: the first validation image of the first class was already moved
    first_class_identifier = CLASS_IDENTIFIERS[0]
    first_validation_name = expected_validation_names[first_class_identifier][0]
    (validation_folder / first_class_identifier).mkdir(parents=True)
    (train_folder / first_class_identifier / first_validation_name).rename(
        target=validation_folder / first_class_identifier / first_validation_name)

    number_moved_images = hold_out_validation_images(train_folder=train_folder, validation_folder=validation_folder,
                                                     images_per_class=2)

    # Three images moved to complete the two classes, the same ones as without the interruption
    assert number_moved_images == 3
    for class_identifier in CLASS_IDENTIFIERS:
        assert (sorted(path.name for path in (validation_folder / class_identifier).iterdir()) ==
                sorted(expected_validation_names[class_identifier]))
        assert len(list((train_folder / class_identifier).iterdir())) == 4

    # A new extraction writes a validation image again in the training split, deleted by the next hold-out
    duplicated_name = expected_validation_names[first_class_identifier][1]
    (train_folder / first_class_identifier / duplicated_name).write_bytes(data=b"extracted again")
    assert hold_out_validation_images(train_folder=train_folder, validation_folder=validation_folder,
                                      images_per_class=2) == 0
    assert not (train_folder / first_class_identifier / duplicated_name).exists()
    assert (validation_folder / first_class_identifier / duplicated_name).exists()
