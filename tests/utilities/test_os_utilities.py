"""
Tests of the console helpers of vision_training/utilities/os_utilities.py: the OSC 8 terminal hyperlinks that the
console output of the pipeline shows in place of the whole file:// paths, the separators of print_color, as long
as the messages show in the console, and the expandable segments of the CUDA allocator turned on by the training launch.
"""

import os
import pytest
import torch

from pathlib import Path

from vision_training.utilities.os_utilities import (color_text, enable_expandable_cuda_segments, format_path_link,
                                                    get_image_paths, print_color)


def test_path_link_opens_the_absolute_file_uri_under_a_short_label() -> None:
    """
    Verifies that a relative path links to the file URI of its absolute path, spaces percent-encoded, under the
    underlined name of the file by default or under the given underlined label.
    """
    relative_path = Path("experiments") / "run folder" / "training_configuration.yaml"
    file_uri = (Path(os.getcwd()) / relative_path).as_uri()

    assert file_uri.endswith("/experiments/run%20folder/training_configuration.yaml")
    assert format_path_link(path=relative_path) == (f"\033]8;;{file_uri}\033\\\033[4mtraining_configuration.yaml"
                                                    f"\033[24m\033]8;;\033\\")
    assert format_path_link(path=str(relative_path), label="Configuration") == (f"\033]8;;{file_uri}\033\\"
                                                                                f"\033[4mConfiguration\033[24m"
                                                                                f"\033]8;;\033\\")


def test_root_folder_link_shows_the_whole_path() -> None:
    """
    Verifies that the root folder, which has no name, is shown by its whole underlined path.
    """
    assert format_path_link(path="/") == "\033]8;;file:///\033\\\033[4m/\033[24m\033]8;;\033\\"


def test_separators_are_as_long_as_the_visible_message(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the separators of a message holding a hyperlink are as long as its visible text plus one, the
    hidden link target and the underline codes of its label taking no width.

    Args:
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
    """
    message = f"Label Mapping File : {format_path_link(path='/data/label_mapping.json')}"
    print_color(color="green", output=message, add_separators=True)

    separator = color_text(color="green", text=len("Label Mapping File : label_mapping.json-") * "-")
    assert capsys.readouterr().out.splitlines() == [separator, color_text(color="green", text=message), separator]


@pytest.mark.parametrize(argnames="settings_given, expected_settings",
                         argvalues=[(None, "expandable_segments:True"),
                                    ("max_split_size_mb:128", "max_split_size_mb:128,expandable_segments:True"),
                                    ("expandable_segments:False", "expandable_segments:False")])
def test_expandable_segments_are_added_to_the_allocator_settings(monkeypatch: pytest.MonkeyPatch,
                                                                 settings_given: str | None,
                                                                 expected_settings: str) -> None:
    """
    Verifies that the training launch turns on the expandable segments of the CUDA allocator when nothing is set, adds
    them to other allocator settings given by hand, and keeps an expandable segments setting given by hand.

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
