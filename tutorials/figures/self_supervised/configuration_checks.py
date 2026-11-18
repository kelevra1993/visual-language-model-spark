"""
Listing of the configuration checks of the self-supervised series of the tutorial videos (chapter 1), made by
tutorials/figures/self_supervised/figures.py: the saved configuration of the SimCLR run is written again with four
mistakes (the augmentation turned off, the crop of the views at a probability of 0.5, an unknown method and a batch of
one image), loaded by load_experiment_configuration of the repository, and the red block its checks print before the
launch stops is kept, its colours removed and its long lines wrapped to the width of the scene, as the listing of the
checks scene: the real messages of the current code, not a copy that could drift from it. For a run read from the
run snapshots, the file written keeps the dataset paths the run saved, and the checks read a copy pointing at the
dataset files committed in the snapshots (its label mapping file), so that the listing and the file stay those of a
build from the real run.
"""

import contextlib
import io
import re
import tempfile
import textwrap

import yaml

from pathlib import Path
from typing import Any, Dict

from tutorials.figures.figure_utilities import is_snapshot_folder, read_run_configuration
from vision_training.configuration_processing.configuration_loader import (load_configuration,
                                                                         load_experiment_configuration)

# Name of the configuration written with the mistakes, shown in the title of the printed block
WRONG_CONFIGURATION_NAME = "configuration_wrong.yaml"

# Width of the lines of the listing, in characters of the monospace font of the scene
LISTING_WIDTH = 104

# Pattern of the escape sequences of the colours and links of the console
ESCAPE_SEQUENCE_PATTERN = re.compile(pattern=r"\x1b\[[0-9;]*m|\x1b\]8;;[^\x1b]*\x1b\\")


def write_wrong_configuration(configuration: Dict[str, Any], output_path: Path) -> None:
    """
    Writes the saved configuration of the run with four mistakes the self-supervised checks refuse: the data
    augmentation turned off, the random crop at a probability of 0.5, the method "byol" and a batch size of 1.

    Args:
        configuration (Dict[str, Any]): The saved configuration of the self-supervised run, changed in place.
        output_path (Path): The YAML file written, its folder created if missing.
    """
    configuration["Data"]["data_augmentation"]["activated"] = False
    configuration["Data"]["data_augmentation"]["types"]["random_crop"]["probability"] = 0.5
    configuration["ProjectionHead"]["method"] = "byol"
    configuration["ExperimentConfiguration"]["batch_size"] = 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(data=yaml.safe_dump(data=configuration, sort_keys=False), encoding="utf-8")


def make_check_listing(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Loads the configuration with mistakes through the loader of every training and keeps what its checks print, as the
    listing listings/configuration_checks.txt, every error message wrapped and its continuation lines indented.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the self-supervised run.

    Returns:
        Dict[str, Any]: The number of errors listed, named checks.error_count.

    Raises:
        ValueError: If the loader accepted the configuration, the checks having changed.
    """
    # The configuration with mistakes, written with the dataset paths the run saved
    configuration_path = output_folder / "checks" / WRONG_CONFIGURATION_NAME
    write_wrong_configuration(configuration=load_configuration(configuration_path=run_folder /
                                                               "training_configuration.yaml"),
                              output_path=configuration_path)

    # The block printed by the loader before it stops the launch; a run read from the snapshots is checked from a copy
    # whose dataset paths point at the committed dataset files, under the same file name in a temporary folder, so that
    # the printed block names the same file and no dataset of the machine is read
    printed_output = io.StringIO()
    stopped = False
    with tempfile.TemporaryDirectory() as temporary_folder:
        checked_path = configuration_path
        if is_snapshot_folder(folder=run_folder):
            checked_path = Path(temporary_folder) / WRONG_CONFIGURATION_NAME
            write_wrong_configuration(configuration=read_run_configuration(run_folder=run_folder),
                                      output_path=checked_path)
        with contextlib.redirect_stdout(new_target=printed_output):
            try:
                load_experiment_configuration(configuration_path=checked_path)
            except SystemExit:
                stopped = True
    if not stopped:
        raise ValueError(f"The configuration {configuration_path} with mistakes was accepted: update "
                         f"{Path(__file__).name} to the current checks.")

    # Colours removed, the separators kept as long as the title, and every message wrapped under its dash
    printed_lines = ESCAPE_SEQUENCE_PATTERN.sub(repl="", string=printed_output.getvalue()).splitlines()
    listing_lines, error_count = [], 0
    for line in printed_lines:
        if set(line.strip()) == {"-"}:
            continue
        if line.lstrip().startswith("- "):
            error_count += 1
            listing_lines.append("")
            listing_lines += textwrap.wrap(text=line.strip(), width=LISTING_WIDTH, subsequent_indent="  ")
        elif line.strip():
            listing_lines.append(line.strip())
    listing_path = output_folder / "listings" / "configuration_checks.txt"
    listing_path.parent.mkdir(parents=True, exist_ok=True)
    listing_path.write_text(data="\n".join(listing_lines) + "\n", encoding="utf-8")
    return {"checks.error_count": error_count}
