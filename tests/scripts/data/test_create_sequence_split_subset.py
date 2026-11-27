"""
Tests of the command line of scripts/data/create_sequence_split_subset.py, which writes a fixed random subset of the
sequences of a split (test_10000 of the sequence labelling experiment plan): its help printed without arguments, the
size and the seed of the draw defaulting to DEFAULT_SUBSET_SETTINGS, every argument reaching its setting, and the
missing folders and malformed numbers refused before anything is written.
"""

import inspect

import pytest

from pathlib import Path
from typing import List

from scripts.data.create_sequence_split_subset import (DEFAULT_SUBSET_SETTINGS, create_sequence_split_subset,
                                                       parse_split_subset_arguments)

# Arguments of the command line of the script, every one described by its help
SUBSET_SCRIPT_ARGUMENTS = ["--source_split_folder", "--output_split_folder", "--number_sequences", "--seed"]


def test_subset_script_without_arguments_prints_its_help(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the subset script launched without any argument prints the help of every one of its arguments and
    stops without writing anything, instead of failing on the missing folders.

    Args:
        capsys (pytest.CaptureFixture): The captured console output.
    """
    with pytest.raises(expected_exception=SystemExit) as exit_information:
        parse_split_subset_arguments(arguments=[])
    assert exit_information.value.code == 0
    help_text = capsys.readouterr().out
    assert [argument for argument in SUBSET_SCRIPT_ARGUMENTS if argument not in help_text] == []


def test_subset_arguments_default_to_the_subset_settings() -> None:
    """
    Verifies that the subset script given only its two folders draws DEFAULT_SUBSET_SETTINGS (10,000 sequences with
    the seed 0, test_10000 of the sequence labelling experiment plan), and that the parsed arguments are exactly the
    parameters of create_sequence_split_subset, so that the main block passes them all.
    """
    parsed_arguments = parse_split_subset_arguments(arguments=["--source_split_folder", "test",
                                                               "--output_split_folder", "test_10000"])
    assert vars(parsed_arguments) == {"source_split_folder": Path("test"), "output_split_folder": Path("test_10000"),
                                      **DEFAULT_SUBSET_SETTINGS}
    assert set(vars(parsed_arguments)) == set(inspect.signature(create_sequence_split_subset).parameters)


def test_subset_arguments_reach_every_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that every setting of create_sequence_split_subset can be given on the command line of the subset script,
    the folders starting with a '~' the shell leaves after '=' expanded to the home folder.

    Args:
        tmp_path (Path): The pytest temporary directory, standing for the home folder.
        monkeypatch (pytest.MonkeyPatch): Sets the home folder.
    """
    monkeypatch.setenv(name="HOME", value=str(tmp_path))
    parsed_arguments = parse_split_subset_arguments(arguments=[
        "--source_split_folder=~/dataset/test", "--output_split_folder=~/dataset/test_50", "--number_sequences", "50",
        "--seed", "3"])
    assert vars(parsed_arguments) == {"source_split_folder": tmp_path / "dataset" / "test",
                                      "output_split_folder": tmp_path / "dataset" / "test_50", "number_sequences": 50,
                                      "seed": 3}


@pytest.mark.parametrize(argnames="arguments", argvalues=[
    ["--output_split_folder", "test_10000"],
    ["--source_split_folder", "test"],
    ["--source_split_folder", "test", "--output_split_folder", "test_10000", "--number_sequences", "many"],
    ["--source_split_folder", "test", "--output_split_folder", "test_10000", "--seed", "zero"]])
def test_subset_script_refuses_missing_folders_or_malformed_numbers(arguments: List[str],
                                                                    capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the subset script stops with its usage and the error, before any split is read, when a folder is
    missing or the size or the seed of the draw is not a whole number.

    Args:
        arguments (List[str]): The command line arguments.
        capsys (pytest.CaptureFixture): The captured console output.
    """
    with pytest.raises(expected_exception=SystemExit) as exit_information:
        parse_split_subset_arguments(arguments=arguments)
    assert exit_information.value.code == 2
    assert "error:" in capsys.readouterr().err
