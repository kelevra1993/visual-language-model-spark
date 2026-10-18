"""
When run, writes a smaller split of a sequence dataset (the 'data/' folder of frames and 'labels.json' manifest read by
SequenceDataset of vision_training/data/sequence/sequence_dataset.py) holding a fixed random subset of the sequences of
an existing split, without copying any frame: its 'data' is a symbolic link to the frames of the source split and its
'labels.json' lists the drawn sequences with the categories of the source. Used to shorten the test evaluations of a
hyperparameter experiment, the test_data_folder of its launches pointing to the subset.
"""

import json
import random

from pathlib import Path
from typing import Any, Dict

from vision_training.utilities.os_utilities import format_path_link, link_folder, print_color, read_json


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


if __name__ == "__main__":
    # Split drawn from and folder of the subset, next to it
    source_split_folder = Path("/home/robert_kelevra/Data/sequence_data/synthetic_changing_shapes/test")
    output_split_folder = Path("/home/robert_kelevra/Data/sequence_data/synthetic_changing_shapes/test_10000")
    # Size of the subset and seed of the draw, the same seed giving the same sequences
    number_sequences = 10000
    seed = 0

    subset_labels = create_sequence_split_subset(source_split_folder=source_split_folder,
                                                 output_split_folder=output_split_folder,
                                                 number_sequences=number_sequences, seed=seed)
    number_frames = sum(len(sequence["frames"]) for sequence in subset_labels["sequences"])
    print_color(color="green", output=f"{len(subset_labels['sequences'])} sequences ({number_frames} frames) written "
                                      f"to {format_path_link(path=output_split_folder)}")
