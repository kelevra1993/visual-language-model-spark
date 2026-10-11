"""
Tests of the console helpers of vision_training/utilities/os_utilities.py: the OSC 8 terminal hyperlinks that the
console output of the pipeline shows in place of the whole file:// paths, and the separators of print_color, as long
as the messages show in the console.
"""

import os
import pytest

from pathlib import Path

from vision_training.utilities.os_utilities import color_text, format_path_link, print_color


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
