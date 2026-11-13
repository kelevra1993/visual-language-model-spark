"""
Tests of the manifests parsed once per launch (vision_training/data/manifest_cache.py): within read_manifests_once a
'labels.json' is parsed once and shared by its readers, parsed again once rewritten, every read parsing its file
outside it; the shared manifests are freed and their heap memory given back when the launch ends, normally or by an
error; and the COCO dataset shuffles its images without modifying the shared manifest.
"""

import json
import pytest

from pathlib import Path
from typing import List

from vision_training.data import manifest_cache
from vision_training.data.detection.coco_dataset import CocoDetectionDataset
from vision_training.data.manifest_cache import read_manifest, read_manifests_once


def write_manifest(labels_path: Path, number_images: int) -> None:
    """
    Writes a COCO manifest of images without annotations, for the reads of the tests.

    Args:
        labels_path (Path): The 'labels.json' file written.
        number_images (int): The number of images of the manifest.
    """
    labels_path.write_text(data=json.dumps(obj={
        "images": [{"id": image_index, "file_name": f"{image_index}.png", "width": 40, "height": 20}
                   for image_index in range(number_images)],
        "annotations": [], "categories": [{"id": 1, "name": "first"}]}))


def count_parses(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """
    Records the name of every file parsed by read_manifest, the parse itself left unchanged.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the parser of the module.

    Returns:
        List[str]: The names of the parsed files, filled as they are parsed.
    """
    parsed_files = []
    original_load = json.load
    monkeypatch.setattr(target=manifest_cache.json, name="load",
                        value=lambda fp: parsed_files.append(Path(fp.name).name) or original_load(fp=fp))
    return parsed_files


def test_manifest_is_parsed_once_within_a_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that within a launch every reader of a manifest gets the one parsed by the first reader, while outside a
    launch every read parses its file into a new manifest.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture counting the parses.
    """
    labels_path = tmp_path / "labels.json"
    write_manifest(labels_path=labels_path, number_images=3)
    parsed_files = count_parses(monkeypatch=monkeypatch)

    # One parse shared by the readers of the launch, the same file named by another path included
    with read_manifests_once():
        first_manifest = read_manifest(labels_path=labels_path)
        assert read_manifest(labels_path=tmp_path / "." / "labels.json") is first_manifest
    assert parsed_files == ["labels.json"]

    # Every read parsing its file outside a launch
    assert read_manifest(labels_path=labels_path) is not read_manifest(labels_path=labels_path)
    assert parsed_files == ["labels.json"] * 3


def test_rewritten_manifest_is_parsed_again(tmp_path: Path) -> None:
    """
    Verifies that a manifest rewritten during a launch is parsed again, the shared manifest being the one of the file
    at its size and modification time.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    labels_path = tmp_path / "labels.json"
    write_manifest(labels_path=labels_path, number_images=2)
    with read_manifests_once():
        assert len(read_manifest(labels_path=labels_path)["images"]) == 2
        write_manifest(labels_path=labels_path, number_images=5)
        assert len(read_manifest(labels_path=labels_path)["images"]) == 5


def test_shared_manifests_are_freed_when_the_launch_ends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the shared manifests are freed and the freed heap memory given back when a launch ends normally and
    when it ends by an error, which is raised again.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture recording the release of the heap memory.
    """
    labels_path = tmp_path / "labels.json"
    write_manifest(labels_path=labels_path, number_images=2)
    releases = []
    monkeypatch.setattr(target=manifest_cache, name="release_freed_heap_memory", value=lambda: releases.append(True))

    # A launch ending normally, then one ending by an error
    with read_manifests_once():
        read_manifest(labels_path=labels_path)
        assert len(manifest_cache.shared_manifests) == 1
    assert manifest_cache.shared_manifests is None and releases == [True]
    with pytest.raises(expected_exception=RuntimeError, match="launch failed"):
        with read_manifests_once():
            read_manifest(labels_path=labels_path)
            raise RuntimeError("launch failed")
    assert manifest_cache.shared_manifests is None and releases == [True, True]


def test_coco_dataset_leaves_the_shared_manifest_unchanged(tmp_path: Path) -> None:
    """
    Verifies that the COCO dataset shuffles its images in a list of its own, the shared manifest keeping the order of
    its file for the record of the data read after it.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    labels_path = tmp_path / "labels.json"
    write_manifest(labels_path=labels_path, number_images=50)
    with read_manifests_once():
        dataset = CocoDetectionDataset(data_directory=str(tmp_path / "data"), labels_file=str(labels_path),
                                       image_size=80, keep_ratio=True, label_mode="multi_class",
                                       label_dictionary={1: 1}, shuffle=True)
        shared_images = read_manifest(labels_path=labels_path)["images"]
    assert [image["id"] for image in shared_images] == list(range(50))
    assert [dataset.get_image_record(image_index=image_index)[0]["id"] for image_index in range(50)] != list(range(50))
