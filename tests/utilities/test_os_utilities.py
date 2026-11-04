"""
Tests of vision_training/utilities/os_utilities.py: the expandable segments of the CUDA allocator turned on by the
training launch, and the images of a folder listed with or without its subfolders.
"""

import os
import pytest
import torch

from pathlib import Path

from vision_training.utilities.os_utilities import enable_expandable_cuda_segments, get_image_paths


@pytest.mark.parametrize(argnames="settings_given, expected_settings",
                         argvalues=[("max_split_size_mb:128", "max_split_size_mb:128,expandable_segments:True")])
def test_expandable_segments_are_added_to_the_allocator_settings(monkeypatch: pytest.MonkeyPatch,
                                                                 settings_given: str | None,
                                                                 expected_settings: str) -> None:
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
