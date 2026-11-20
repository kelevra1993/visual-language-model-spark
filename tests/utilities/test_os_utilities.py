"""
Tests of vision_training/utilities/os_utilities.py: the expandable segments of the CUDA allocator turned on by the
training launch, the images of a folder listed with or without its subfolders, the freed heap memory given back to
the system after a dataset packs its manifest and at the end of the trainer launch, on Linux only, and the JSON files
read whole, a malformed one stopping with its name.
"""

import ctypes
import os
import pytest
import sys
import torch

from pathlib import Path

from vision_training.utilities.os_utilities import (enable_expandable_cuda_segments, get_image_paths, read_json,
                                                    release_freed_heap_memory)


@pytest.mark.parametrize(argnames="settings_given, expected_settings",
                         argvalues=[("max_split_size_mb:128", "max_split_size_mb:128,expandable_segments:True")])
def test_expandable_segments_are_added_to_the_allocator_settings(
        monkeypatch: pytest.MonkeyPatch, settings_given: str | None, expected_settings: str) -> None:
    """
    Verifies that the training launch adds the expandable segments of the CUDA allocator to other allocator settings
    given by hand.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture setting the environment variable and the CUDA state.
        settings_given (str | None): The PYTORCH_CUDA_ALLOC_CONF set before the launch, None when it is not set.
        expected_settings (str): The PYTORCH_CUDA_ALLOC_CONF expected after it.
    """
    monkeypatch.setattr(target=torch.cuda, name="is_initialized", value=lambda: False)
    if settings_given is None:
        monkeypatch.delenv(name="PYTORCH_CUDA_ALLOC_CONF", raising=False)
    else:
        monkeypatch.setenv(name="PYTORCH_CUDA_ALLOC_CONF", value=settings_given)

    enable_expandable_cuda_segments()

    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == expected_settings


def test_expandable_segments_refused_once_cuda_is_initialised(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that turning on the expandable segments after CUDA is initialised raises an error instead of being
    silently ignored by the allocator, and leaves the environment variable unchanged.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture setting the environment variable and the CUDA state.
    """
    monkeypatch.setattr(target=torch.cuda, name="is_initialized", value=lambda: True)
    monkeypatch.delenv(name="PYTORCH_CUDA_ALLOC_CONF", raising=False)

    with pytest.raises(expected_exception=RuntimeError, match="before CUDA is initialised"):
        enable_expandable_cuda_segments()
    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ


def test_image_paths_of_the_subfolders_are_listed_on_request(tmp_path: Path) -> None:
    """
    Verifies that get_image_paths lists only the images of the folder itself by default, and with search_subfolders
    also those of its subfolders and of the folders it links to, as for a folder of plain images embedded by
    scripts/embedding/cluster_image_embeddings.py, the other files being left out in both cases.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # Images at the top, in a subfolder and in a linked folder, next to a file that is not an image
    image_folder = tmp_path / "images"
    (image_folder / "camera_1").mkdir(parents=True)
    linked_folder = tmp_path / "other_images"
    linked_folder.mkdir()
    for image_path in [image_folder / "top.jpg", image_folder / "camera_1" / "frame.PNG",
                       linked_folder / "linked.jpeg"]:
        image_path.write_bytes(data=b"image")
    (image_folder / "notes.txt").write_text(data="not an image")
    (image_folder / "camera_2").symlink_to(target=linked_folder, target_is_directory=True)

    assert get_image_paths(folder=image_folder) == [image_folder / "top.jpg"]
    assert get_image_paths(folder=image_folder, search_subfolders=True) == [
        image_folder / "camera_1" / "frame.PNG", image_folder / "camera_2" / "linked.jpeg", image_folder / "top.jpg"]


def read_resident_megabytes() -> float:
    """
    Reads the resident memory of the test process from the kernel, for the check of the release of the freed heap.

    Returns:
        float: The resident memory in megabytes.
    """
    with open(file="/proc/self/smaps_rollup", mode="r") as file_handler:
        resident_line = next(line for line in file_handler if line.startswith("Rss:"))
    return int(resident_line.split()[1]) / 1024


@pytest.mark.skipif(condition=not sys.platform.startswith("linux"), reason="malloc_trim is a GNU C library function")
def test_freed_heap_memory_is_given_back() -> None:
    """
    Verifies that the memory freed in the heap of the C allocator, which stays resident while live objects sit between
    the freed ones (like the strings of a parsed manifest among the objects allocated after it), is given back to the
    system: 200 MB of strings of 2 KB, one in 64 kept alive.
    """
    # Distinct strings above the 512 bytes of the Python object allocator, hence in the heap of the C allocator, freed
    strings = [string_index.to_bytes(length=2048, byteorder="little") for string_index in range(100_000)]
    kept_strings = strings[::64]
    del strings
    resident_before = read_resident_megabytes()

    release_freed_heap_memory()

    assert resident_before - read_resident_megabytes() > 100 and len(kept_strings) == 1563


def test_freed_heap_memory_is_left_outside_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the release of the freed heap memory does nothing outside Linux, without loading any C library.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture setting the platform and the C library loader.
    """
    monkeypatch.setattr(target=sys, name="platform", value="darwin")
    monkeypatch.setattr(target=ctypes, name="CDLL", value=lambda name: pytest.fail(reason=f"{name} loaded"))

    assert release_freed_heap_memory() is None


def test_malformed_json_file_is_refused_with_its_name(tmp_path: Path) -> None:
    """
    Verifies that read_json parses a valid file, raises FileNotFoundError on a missing one, and stops on a data
    record truncated by an interrupted write with an error naming the file and the position of the problem, instead
    of returning None to a caller that indexes it (the resume then failing on a 'NoneType' object).

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # A valid record is returned parsed
    valid_path = tmp_path / "data.json"
    valid_path.write_text(data='{"splits": {"train": {"images": 3}}}')
    assert read_json(path=str(valid_path)) == {"splits": {"train": {"images": 3}}}

    # A missing file keeps raising FileNotFoundError
    with pytest.raises(expected_exception=FileNotFoundError):
        read_json(path=str(tmp_path / "missing.json"))

    # A truncated record stops with its file name and the line and column of the error
    truncated_path = tmp_path / "truncated_data.json"
    truncated_path.write_text(data='{"splits": {"train": ')
    with pytest.raises(expected_exception=ValueError, match="truncated_data.json.* cannot be parsed: .*line 1 column"):
        read_json(path=str(truncated_path))
