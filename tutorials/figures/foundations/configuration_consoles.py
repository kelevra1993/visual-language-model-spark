"""
Consoles of the configuration tools for the first foundations chapter, right after the scene of the configuration
file: the check of configuration files without training (python -m vision_training.check_configuration), run for real
on the synthetic classification configuration and on a copy holding two mistakes, and the conversion of the files
written before a setting was removed (scripts/configuration_processing/convert_fixed_settings.py), whose file is first
refused by the same check and then listed by convert_fixed_settings with write_files False. The copies are written into
temporary folders from the synthetic configuration as it is in the repository, so that the consoles follow its keys
and the checks; their label mapping is the one the configuration names, read from the committed dataset files of the
run snapshots when the build reads them (resolve_dataset_path), so that the check, which loads the label mapping, never
reads a dataset of the machine. The console lines, without their colours and hyperlinks, are the facts the terminal
scenes print, with the exit code of the check.
"""

import subprocess
import sys
import tempfile

from pathlib import Path
from typing import Any, Dict, List, Tuple

from scripts.configuration_processing.convert_fixed_settings import (FIXED_SETTINGS, FIXED_VALUES, KEY_LINE_PATTERN,
                                                                     convert_fixed_settings, find_key_paths)
from tutorials.figures.figure_utilities import is_snapshot_folder, resolve_dataset_path
from tutorials.figures.foundations.code_values import read_main_block_value
from tutorials.figures.foundations.listings import capture_printed_text
from tutorial_builder.figure_context import find_repository_root
from vision_training.configuration_processing.common_checks import REMOVED_SETTINGS
from vision_training.configuration_processing.configuration_loader import load_configuration
from vision_training.utilities.os_utilities import INVISIBLE_SEQUENCE_PATTERN

# Key path of the label mapping file the loading of a classification configuration reads, pointed at the committed
# dataset files in the copies when the build reads the run snapshots
LABEL_MAPPING_KEY_PATH = "Data.label_mapping.mapping_file_path"

# Names of the copies, printed by the consoles: the copy holding two mistakes, and the file written before a setting
# was removed
MISTAKEN_CONFIGURATION_NAME = "configuration_with_mistakes.yaml"
OLD_CONFIGURATION_NAME = "old_configuration.yaml"

# Mistakes of the copy checked next to the synthetic configuration, those of the rejected configuration of the README
# (README/configuration_rejected.png): a negative learning rate, the opposite of the one of the file, and an image size
# that the poolings of the backbone cannot halve down to its last block
MISTAKEN_IMAGE_SIZE = 500
MISTAKEN_KEY_PATHS = {"learning_rate": "ExperimentConfiguration.learning_rate",
                      "image_size": "Data.image_settings.size"}

# Removed setting held by the old configuration, with the value the code now always uses (FIXED_VALUES), so that the
# conversion only removes it: mixed precision, always on since its removal
OLD_REMOVED_SETTING = "Optimization.mixed_precision"


def write_configuration_copy(source_text: str, destination_path: Path, replaced_values: Dict[str, str],
                             added_setting: Tuple[str, str] | None) -> None:
    """
    Writes a copy of a configuration file edited as text, its comments kept, for the consoles of the configuration
    tools: the value of some keys replaced, and optionally a setting added right under the line of its section, as a
    file written before the setting was removed would hold it.

    Args:
        source_text (str): The text of the configuration copied.
        destination_path (Path): The copy written.
        replaced_values (Dict[str, str]): The YAML text of the new value of every replaced key, by its key path.
        added_setting (Tuple[str, str] | None): The key path and YAML text of the value of the setting added, None for
            none.

    Raises:
        KeyError: If a replaced key, or the section of the added setting, is not found exactly once in the file.
    """
    lines = source_text.split("\n")
    key_paths = find_key_paths(lines=lines)

    # Value of every replaced key, the indentation and key of its line kept
    for key_path, value_text in replaced_values.items():
        line_indices = [line_index for line_index, line_key_path in enumerate(key_paths) if line_key_path == key_path]
        if len(line_indices) != 1:
            raise KeyError(f"Expected the key {key_path} once in the configuration copied to {destination_path}, found "
                           f"it {len(line_indices)} times.")
        key_match = KEY_LINE_PATTERN.match(lines[line_indices[0]])
        lines[line_indices[0]] = f"{key_match.group(1)}{key_match.group(2)}: {value_text}"

    # Setting added as the first line of its section, one indentation level below the section
    if added_setting is not None:
        setting_key_path, value_text = added_setting
        section_key_path, _, setting_key = setting_key_path.rpartition(".")
        section_indices = [line_index for line_index, line_key_path in enumerate(key_paths)
                           if line_key_path == section_key_path]
        if len(section_indices) != 1:
            raise KeyError(f"Expected the section {section_key_path} once in the configuration copied to "
                           f"{destination_path}, found it {len(section_indices)} times.")
        section_indentation = KEY_LINE_PATTERN.match(lines[section_indices[0]]).group(1)
        lines.insert(section_indices[0] + 1, f"{section_indentation}  {setting_key}: {value_text}")
    destination_path.write_text(data="\n".join(lines), encoding="utf-8")


def clean_console_lines(console_text: str, removed_prefix: str) -> List[str]:
    """
    Gives the lines of a console as a terminal scene prints them: without the colour codes and with the hyperlinks
    reduced to their labels, without the blank lines and the separator lines framing a red block, and without the
    temporary folder before the name of a copy.

    Args:
        console_text (str): The text printed by a tool.
        removed_prefix (str): The temporary folder written before the names of the copies, removed from the lines.

    Returns:
        List[str]: The lines of the console, right-stripped.
    """
    visible_text = INVISIBLE_SEQUENCE_PATTERN.sub(repl="", string=console_text).replace(removed_prefix, "")
    return [line.rstrip() for line in visible_text.splitlines() if line.strip() and set(line.strip()) != {"-"}]


def run_configuration_check(configuration_paths: List[Path], removed_prefix: str) -> Tuple[List[str], int]:
    """
    Runs the check of configuration files without training as a pipeline runs it, python -m
    vision_training.check_configuration with one --configuration_path per file, from the repository root with the
    Python of the repository, and gives its console and exit code.

    Args:
        configuration_paths (List[Path]): The configuration files checked, in order.
        removed_prefix (str): The temporary folder written before the names of the copies, removed from the lines.

    Returns:
        Tuple[List[str], int]: The lines of the console (clean_console_lines) and the exit code, 0 when every file is
            valid, 1 otherwise.

    Raises:
        RuntimeError: If the command exits with another code (a malformed command line or a crash), with its errors.
    """
    command = [sys.executable, "-m", "vision_training.check_configuration",
               *[f"--configuration_path={configuration_path}" for configuration_path in configuration_paths]]
    completed_process = subprocess.run(args=command, cwd=find_repository_root(), capture_output=True, text=True,
                                       check=False)
    if completed_process.returncode not in (0, 1):
        raise RuntimeError(f"The configuration check {' '.join(command)} exited with {completed_process.returncode}: "
                           f"{completed_process.stderr[-2000:]}")
    return (clean_console_lines(console_text=completed_process.stdout, removed_prefix=removed_prefix),
            completed_process.returncode)


def split_check_console(console_lines: List[str], file_name: str) -> Dict[str, Any]:
    """
    Finds the block of one file in the console of the configuration check: its green line when it is valid, or its red
    header and the problems listed under it.

    Args:
        console_lines (List[str]): The lines of the console (run_configuration_check).
        file_name (str): The name of the checked file, as the console prints it.

    Returns:
        Dict[str, Any]: {"valid": bool, "header": str, "problems": List[str]}, the header being the green line of a
            valid file.

    Raises:
        ValueError: If the console holds no line naming the file.
    """
    header_indices = [line_index for line_index, line in enumerate(console_lines)
                      if line.startswith(("Valid Configuration File : ", "Invalid Configuration File : "))
                      and f" : {file_name}" in line]
    if len(header_indices) != 1:
        raise ValueError(f"Expected one line of the check console naming {file_name}, found {len(header_indices)}: "
                         f"{console_lines}")
    header_index = header_indices[0]

    # Problems of an invalid file: the bulleted lines under its header, up to the next file
    problems = []
    for line in console_lines[header_index + 1:]:
        if not line.startswith("  - "):
            break
        problems.append(line)
    return {"valid": console_lines[header_index].startswith("Valid"), "header": console_lines[header_index],
            "problems": problems}


def make_configuration_consoles(run_folder: Path, configuration_path: Path) -> Dict[str, Any]:
    """
    Runs the configuration tools of the first foundations chapter on copies of the synthetic classification
    configuration and gives their consoles as facts: the check of the configuration and of a copy holding the mistakes
    of MISTAKEN_KEY_PATHS, then the check refusing a copy holding the removed setting OLD_REMOVED_SETTING and the
    listing of convert_fixed_settings with write_files False, searching a launch folder holding that copy alone.

    Args:
        run_folder (Path): The folder of the classification run, real or read from the snapshots, telling whether the
            label mapping is read from the committed dataset files.
        configuration_path (Path): The synthetic classification configuration of the repository (repository file
            data_pipeline_configuration of tutorial_sources.yaml), the file of the configuration scene.

    Returns:
        Dict[str, Any]: The facts configuration_check.* (the green line of the configuration, the red header and
            problems of the copy, the exit code) and configuration_conversion.* (the refusal of the old copy and the
            listing line).

    Raises:
        ValueError: If the configuration is invalid, the copy is not refused for its two mistakes, or the old copy is
            not refused for its removed setting alone and listed by the conversion.
    """
    configuration_text = configuration_path.read_text(encoding="utf-8")
    configuration = load_configuration(configuration_path=configuration_path)

    # Label mapping of the copies: the file the configuration names, or its committed copy for a build from the
    # snapshots
    label_mapping_path = resolve_dataset_path(path=configuration["Data"]["label_mapping"]["mapping_file_path"],
                                              snapshot_mode=is_snapshot_folder(folder=run_folder))
    label_mapping_value = {LABEL_MAPPING_KEY_PATH: f"\"{label_mapping_path}\""}

    # Removed setting of the old copy, with the value the code always uses so that the conversion only removes it
    if OLD_REMOVED_SETTING not in REMOVED_SETTINGS or OLD_REMOVED_SETTING not in FIXED_SETTINGS:
        raise ValueError(f"{OLD_REMOVED_SETTING} is no longer a removed fixed setting of the code: choose another one "
                         f"for the conversion console.")
    old_setting_value = str(FIXED_VALUES[OLD_REMOVED_SETTING])

    with tempfile.TemporaryDirectory(prefix="configuration_consoles_") as temporary_folder:
        # The configuration under its own name, the copy with its mistakes, and the old copy alone in its launch folder
        check_folder, launch_folder = Path(temporary_folder) / "check", Path(temporary_folder) / "launch"
        check_folder.mkdir()
        launch_folder.mkdir()
        checked_path = check_folder / configuration_path.name
        mistaken_path = check_folder / MISTAKEN_CONFIGURATION_NAME
        old_path = launch_folder / OLD_CONFIGURATION_NAME
        learning_rate = float(configuration["ExperimentConfiguration"]["learning_rate"])
        write_configuration_copy(source_text=configuration_text, destination_path=checked_path,
                                 replaced_values=label_mapping_value, added_setting=None)
        write_configuration_copy(source_text=configuration_text, destination_path=mistaken_path, replaced_values={
            **label_mapping_value, MISTAKEN_KEY_PATHS["learning_rate"]: repr(-learning_rate),
            MISTAKEN_KEY_PATHS["image_size"]: str(MISTAKEN_IMAGE_SIZE)}, added_setting=None)
        write_configuration_copy(source_text=configuration_text, destination_path=old_path,
                                 replaced_values=label_mapping_value,
                                 added_setting=(OLD_REMOVED_SETTING, old_setting_value))

        # Check of the configuration and of the copy with its mistakes, in one command as a pipeline gives them
        check_lines, check_exit_code = run_configuration_check(configuration_paths=[checked_path, mistaken_path],
                                                               removed_prefix=f"{check_folder}/")
        checked_block = split_check_console(console_lines=check_lines, file_name=checked_path.name)
        mistaken_block = split_check_console(console_lines=check_lines, file_name=mistaken_path.name)
        if not checked_block["valid"] or mistaken_block["valid"] or \
                len(mistaken_block["problems"]) != len(MISTAKEN_KEY_PATHS):
            raise ValueError(f"Expected {configuration_path.name} valid and {len(MISTAKEN_KEY_PATHS)} problems for "
                             f"{MISTAKEN_CONFIGURATION_NAME}, got: {check_lines}")

        # Refusal of the old copy, then the listing of the conversion searching its launch folder alone
        old_check_lines, _ = run_configuration_check(configuration_paths=[old_path],
                                                     removed_prefix=f"{launch_folder}/")
        old_block = split_check_console(console_lines=old_check_lines, file_name=old_path.name)
        if old_block["valid"] or len(old_block["problems"]) != 1 or f"'{OLD_REMOVED_SETTING}' was removed" not in \
                old_block["problems"][0]:
            raise ValueError(f"Expected {OLD_CONFIGURATION_NAME} refused for {OLD_REMOVED_SETTING} alone, got: "
                             f"{old_check_lines}")
        backup_suffix = read_main_block_value(
            script_path=find_repository_root() / "scripts" / "configuration_processing" / "convert_fixed_settings.py",
            variable_name="backup_suffix")
        conversion_text = capture_printed_text(printing_function=lambda: convert_fixed_settings(
            run_roots=[], configuration_folders=[launch_folder], write_files=False, backup_suffix=backup_suffix))
        conversion_lines = clean_console_lines(console_text=conversion_text, removed_prefix=f"{launch_folder}/")
        if len(conversion_lines) != 1 or not conversion_lines[0].startswith("To convert"):
            raise ValueError(f"Expected one line listing {OLD_CONFIGURATION_NAME} to convert, got: {conversion_lines}")
    return {"configuration_check.mistaken_file_name": MISTAKEN_CONFIGURATION_NAME,
            "configuration_check.valid_line": checked_block["header"],
            "configuration_check.invalid_line": mistaken_block["header"],
            "configuration_check.problem_lines": mistaken_block["problems"],
            "configuration_check.problem_count": len(mistaken_block["problems"]),
            "configuration_check.exit_code": check_exit_code,
            "configuration_conversion.old_file_name": OLD_CONFIGURATION_NAME,
            "configuration_conversion.invalid_line": old_block["header"],
            "configuration_conversion.problem_line": old_block["problems"][0],
            "configuration_conversion.listing_line": conversion_lines[0]}
