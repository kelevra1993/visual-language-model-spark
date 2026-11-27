"""
When run, writes a smaller split of a sequence dataset (the 'data/' folder of frames and 'labels.json' manifest read by
SequenceDataset of vision_training/data/sequence/sequence_dataset.py) holding a fixed random subset of the sequences of
an existing split, without copying any frame: its 'data' is a symbolic link to the frames of the source split and its
'labels.json' lists the drawn sequences with the categories of the source. Used to shorten the test evaluations of a
hyperparameter experiment, the test_data_folder of its launches pointing to the subset. The source split and the
folder of the subset are required arguments, the size and the seed of the draw optional (DEFAULT_SUBSET_SETTINGS), the
help being printed when no argument is given.
"""

import argparse
import json
import random
import sys

from pathlib import Path
from typing import Any, Dict, List

from vision_training.utilities.os_utilities import format_path_link, link_folder, print_color, read_json

# Size of the subset and seed of the draw by default: test_10000 of the sequence labelling experiment plan, 10,000 of
# the 50,000 test sequences drawn with the seed 0, the same seed giving the same sequences
DEFAULT_SUBSET_SETTINGS = {"number_sequences": 10000, "seed": 0}


def create_sequence_split_subset(source_split_folder: Path, output_split_folder: Path, number_sequences: int,
                                 seed: int) -> Dict[str, Any]:
    """
    Draws 'number_sequences' sequences of a split once with a fixed seed and writes them as a new split, so that every
    launch of an experiment is evaluated on the same sequences, kept in the order of the source manifest.

    Args:
        source_split_folder (Path): The split to draw from, holding 'data/' and 'labels.json'.
        output_split_folder (Path): The folder of the new split, created if missing.
        number_sequences (int): The number of sequences drawn.
        seed (int): The seed of the draw, the same seed giving the same subset.

    Returns:
        Dict[str, Any]: The manifest written in '<output_split_folder>/labels.json'.

    Raises:
        ValueError: If the source split holds fewer than 'number_sequences' sequences.
        FileExistsError: If '<output_split_folder>/data' exists and is not a symbolic link.
    """
    source_labels = read_json(path=str(source_split_folder / "labels.json"))
    source_sequences = source_labels["sequences"]

    # Fail loudly instead of writing a subset as large as the source
    if number_sequences > len(source_sequences):
        raise ValueError(f"The split {format_path_link(path=source_split_folder)} holds {len(source_sequences)} "
                         f"sequences, fewer than the {number_sequences} asked for")

    # Draw the sequences once and keep them in the order of the source manifest
    drawn_indices = sorted(random.Random(seed).sample(population=range(len(source_sequences)), k=number_sequences))
    description = (f"{source_labels['info']['description']}, subset of {number_sequences} sequences drawn with the "
                   f"seed {seed}")
    subset_labels = {**source_labels, "info": {"description": description},
                     "sequences": [source_sequences[index] for index in drawn_indices]}

    # Link the frames of the source split and write the manifest of the subset
    output_split_folder.mkdir(parents=True, exist_ok=True)
    link_folder(link_path=output_split_folder / "data", target_folder=source_split_folder / "data")
    (output_split_folder / "labels.json").write_text(json.dumps(subset_labels), encoding="utf-8")
    return subset_labels


def parse_split_subset_arguments(arguments: List[str]) -> argparse.Namespace:
    """
    Reads the settings of a split subset from the command line of the script, so that a subset is written without
    editing the file: the source split and the folder of the subset are required, the size and the seed of the draw
    default to DEFAULT_SUBSET_SETTINGS. Launched without any argument, the script prints the help of every argument
    and stops. A '~' left by the shell at the start of a path ('--source_split_folder=~/Data/...') is expanded to the
    home folder.

    Args:
        arguments (List[str]): The command line arguments, without the script name.

    Returns:
        argparse.Namespace: The arguments of create_sequence_split_subset.

    Raises:
        SystemExit: After printing the help when no argument is given, or the usage and the error when a folder is
            missing or a number is not a whole number.
    """
    # The description and the example keep their line breaks, argparse wrapping the help of every argument
    parser = argparse.ArgumentParser(
        prog="uv run python -m scripts.data.create_sequence_split_subset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Writes a fixed random subset of the sequences of a split as a new split, its 'data' a link to\n"
                    "the frames of the source split, so that every launch of an experiment is tested on the same\n"
                    "shorter split.",
        epilog="Example, test_10000 of the synthetic changing shapes:\n"
               "  uv run python -m scripts.data.create_sequence_split_subset \\\n"
               "      --source_split_folder ~/Data/sequence_data/synthetic_changing_shapes/test \\\n"
               "      --output_split_folder ~/Data/sequence_data/synthetic_changing_shapes/test_10000")
    # The split drawn from and the folder of the subset
    parser.add_argument("--source_split_folder", type=Path, required=True,
                        help="Split the sequences are drawn from, holding 'data/' and 'labels.json' (required).")
    parser.add_argument("--output_split_folder", type=Path, required=True,
                        help="Folder of the subset, created if missing, its 'labels.json' replaced (required).")
    # The size of the subset and the seed of the draw
    parser.add_argument("--number_sequences", type=int, default=DEFAULT_SUBSET_SETTINGS["number_sequences"],
                        help="Number of sequences drawn (default: %(default)s).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SUBSET_SETTINGS["seed"],
                        help="Seed of the draw, the same seed giving the same sequences (default: %(default)s).")

    # Without any argument, the script describes its arguments instead of failing on the missing folders
    if not arguments:
        parser.print_help()
        parser.exit(status=0)
    parsed_arguments = parser.parse_args(args=arguments)

    # A '~' the shell leaves after '=' points to the home folder, as in the other scripts of the repository
    for path_name in ["source_split_folder", "output_split_folder"]:
        setattr(parsed_arguments, path_name, getattr(parsed_arguments, path_name).expanduser())
    return parsed_arguments


if __name__ == "__main__":
    # Source split, subset folder, size and seed read from the command line, the help printed when none is given
    subset_arguments = parse_split_subset_arguments(arguments=sys.argv[1:])

    subset_labels = create_sequence_split_subset(source_split_folder=subset_arguments.source_split_folder,
                                                 output_split_folder=subset_arguments.output_split_folder,
                                                 number_sequences=subset_arguments.number_sequences,
                                                 seed=subset_arguments.seed)
    number_frames = sum(len(sequence["frames"]) for sequence in subset_labels["sequences"])
    print_color(color="green", output=f"{len(subset_labels['sequences'])} sequences ({number_frames} frames) written "
                                      f"to {format_path_link(path=subset_arguments.output_split_folder)}")
