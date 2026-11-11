"""
Dataset figures of the sequence classification tutorials (chapter 1), made by
tutorials/figures/sequence_classification/figures.py from the moving shapes dataset the main run trained on (its
training_configuration.yaml gives the split folders): the frames of the first training sequence copied for the HTML
scene of the task, the listings of the split layout and of its labels.json entry, and the chart of the sequence lengths
on disk against the lengths the current generator draws, with the share of padding frames each gives to batches of the
run. Its facts are the sizes of the splits, the classes and the padding shares the narration quotes.
"""

import json
import random

import cv2
import numpy

from collections import Counter
from matplotlib import pyplot
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, POINTS_PER_PIXEL,
                                                SERIES_COLOURS, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR, compute_pixel_box,
                                                register_fonts, round_fact, save_figure, style_chart_panel)
from tutorials.figures.sequence_classification.sequence_figure_utilities import (copy_frames, read_generator_setting,
                                                                                 write_text_lines)
from vision_training.data.synthetic.create_synthetic_dataset import MINIMUM_SEQUENCE_FRAMES

# Sequence shown on the scene of the task, the first of the training manifest
SHOWN_SEQUENCE_NAME = "train_000000"

# Lengths the generator draws for the sequence classification (the bounds of its main block), the training split
# drawn again with them from a fixed seed, and the seed of the shuffle making the batches of the run
GENERATOR_MAXIMUM_FRAMES = 12
LENGTH_DRAW_SEED = 0
BATCH_SHUFFLE_SEED = 0

# Splits of a dataset, in the order of the manifests
SPLIT_NAMES = ["train", "validation", "test"]

# Task of the synthetic generator whose main block settings the facts read
GENERATOR_TASK = "sequence_classification"


def read_manifest(split_folder: Path) -> Dict[str, Any]:
    """
    Reads the labels.json manifest of a split of the moving shapes dataset.

    Args:
        split_folder (Path): The split folder, holding labels.json and data/.

    Returns:
        Dict[str, Any]: The manifest, with its info, categories and sequences.

    Raises:
        FileNotFoundError: If the manifest is missing.
    """
    labels_path = split_folder / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"No manifest {labels_path}.")
    return json.loads(labels_path.read_text(encoding="utf-8"))


def write_dataset_listings(output_folder: Path, train_manifest: Dict[str, Any], label_mapping: Dict[str, int]) -> None:
    """
    Writes the two listings of the data scene: the layout of the training split with the frame folder of its first
    sequence, and its labels.json with the categories, the first sequence entry and the label mapping file, every
    count read from the manifest.

    Args:
        output_folder (Path): The figures folder of the series.
        train_manifest (Dict[str, Any]): The manifest of the training split.
        label_mapping (Dict[str, int]): The label_mapping.json of the dataset.
    """
    sequences = train_manifest["sequences"]
    first_sequence = sequences[0]
    frame_names = [Path(frame).name for frame in first_sequence["frames"]]
    tree_lines = ["train/", "├── data/", f"│   ├── {first_sequence['name']}/", f"│   │   ├── {frame_names[0]}",
                  "│   │   ├── ...", f"│   │   └── {frame_names[-1]}",
                  f"│   └── ... {len(sequences) - 1:,} more sequence folders", "└── labels.json"]
    write_text_lines(output_path=output_folder / "listings" / "split_tree.txt", lines=tree_lines)

    # The manifest with its categories, its first sequence and the mapping of the categories to class indices
    category_lines = [f'    {{"id": {category["id"]}, "name": "{category["name"]}"}}' +
                      ("," if category_index < len(train_manifest["categories"]) - 1 else "")
                      for category_index, category in enumerate(train_manifest["categories"])]
    mapping_text = ", ".join(f'"{key}": {value}' for key, value in label_mapping.items())
    manifest_lines = (["{", '  "categories": ['] + category_lines + ["  ],", '  "sequences": [',
                      f'    {{"id": {first_sequence["id"]}, "name": "{first_sequence["name"]}", '
                      f'"category_id": {first_sequence["category_id"]},',
                      f'     "frames": ["{first_sequence["frames"][0]}", ...,',
                      f'                "{first_sequence["frames"][-1]}"]}},',
                      f"    ... {len(sequences) - 1:,} more sequences", "  ]", "}", "",
                      f"label_mapping.json  {{{mapping_text}}}"])
    write_text_lines(output_path=output_folder / "listings" / "labels_entry.txt", lines=manifest_lines)


def read_single_length(lengths: List[int]) -> int:
    """
    Gives the number of frames shared by every sequence of the dataset on disk, which the narration quotes as the
    length of all its sequences, and refuses a dataset of several lengths, for which that sentence would be wrong.

    Args:
        lengths (List[int]): The frames of every training sequence.

    Returns:
        int: The common number of frames.

    Raises:
        ValueError: If the sequences have different numbers of frames.
    """
    if len(set(lengths)) != 1:
        raise ValueError(f"The training sequences have {sorted(set(lengths))} frames, while the storyboards of the "
                         f"series say they all have the same number: update their narration.")
    return lengths[0]


def compute_padding_share(lengths: List[int], batch_size: int, seed: int) -> float:
    """
    Gives the share of padding frames in the batches of a split, the sequences shuffled once like the dataset does
    and batched in order like the loader, every batch padded to its longest sequence by collate_padded_sequences.

    Args:
        lengths (List[int]): The frames of every sequence of the split.
        batch_size (int): The batch size of the run.
        seed (int): The seed of the shuffle.

    Returns:
        float: The padding frames over all the frames of the padded batches.
    """
    shuffled_lengths = list(lengths)
    random.Random(seed).shuffle(shuffled_lengths)
    padded_frames = 0
    for batch_start in range(0, len(shuffled_lengths), batch_size):
        batch_lengths = shuffled_lengths[batch_start:batch_start + batch_size]
        padded_frames += max(batch_lengths) * len(batch_lengths)
    return 1.0 - sum(shuffled_lengths) / padded_frames


def draw_lengths_figure(output_path: Path, disk_lengths: List[int], generator_lengths: List[int],
                        padding_shares: Dict[str, float], batch_size: int) -> Dict[str, Any]:
    """
    Draws the lengths chart: the number of training sequences of every length on disk and as the current generator
    draws them, and the share of padding frames in the batches of the run for both, and gives the boxes of its panels.

    Args:
        output_path (Path): The PNG file.
        disk_lengths (List[int]): The frames of every training sequence on disk.
        generator_lengths (List[int]): The frames of every training sequence drawn like the generator.
        padding_shares (Dict[str, float]): The padding share of the batches, 'disk' and 'generator'.
        batch_size (int): The batch size of the run, written on the padding panel.

    Returns:
        Dict[str, Any]: The pixel boxes of the panels, named lengths.*.
    """
    register_fonts()
    figure, (length_axes, padding_axes) = pyplot.subplots(
        nrows=1, ncols=2, figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH,
        gridspec_kw={"width_ratios": [2.2, 1.0]})
    figure.patch.set_facecolor(SURFACE_COLOUR)
    figure.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.12, wspace=0.28)

    # Training sequences of every length, side by side bars of the two sources
    all_lengths = list(range(MINIMUM_SEQUENCE_FRAMES, GENERATOR_MAXIMUM_FRAMES + 1))
    disk_counts, generator_counts = Counter(disk_lengths), Counter(generator_lengths)
    bar_width = 0.4
    length_axes.bar(x=numpy.array(all_lengths) - bar_width / 2, height=[disk_counts[length] for length in all_lengths],
                    width=bar_width - 0.04, color=SERIES_COLOURS[0], label="On disk")
    length_axes.bar(x=numpy.array(all_lengths) + bar_width / 2,
                    height=[generator_counts[length] for length in all_lengths], width=bar_width - 0.04,
                    color=SERIES_COLOURS[1], label="Current generator")
    style_chart_panel(axes=length_axes, title="Training sequences per number of frames", x_label="Frames",
                      y_label="Sequences")
    length_axes.set_xticks(ticks=all_lengths)
    length_axes.grid(visible=False, axis="x")
    length_axes.annotate(text=f"{disk_counts[8]:,}", xy=(8 - bar_width / 2, disk_counts[8]), xytext=(0, 8),
                         textcoords="offset points", ha="center", va="bottom", color=TEXT_PRIMARY_COLOUR,
                         fontsize=18 * POINTS_PER_PIXEL)

    # Padding share of the batches of the run, one bar per source with its value
    source_names = ["On disk", "Current generator"]
    share_values = [padding_shares["disk"], padding_shares["generator"]]
    padding_axes.bar(x=[0, 1], height=share_values, width=0.55, color=SERIES_COLOURS[:2])
    style_chart_panel(axes=padding_axes, title=f"Padding frames in batches of {batch_size}", x_label="",
                      y_label="Share of the frames")
    padding_axes.set_xticks(ticks=[0, 1], labels=source_names)
    padding_axes.set_ylim(bottom=0.0, top=0.6)
    padding_axes.grid(visible=False, axis="x")
    for bar_index, share_value in enumerate(share_values):
        padding_axes.annotate(text=f"{share_value:.1%}", xy=(bar_index, share_value), xytext=(0, 8),
                              textcoords="offset points", ha="center", va="bottom", color=TEXT_PRIMARY_COLOUR,
                              fontsize=20 * POINTS_PER_PIXEL)

    # Legend of the two sources above the panels
    legend_handles = [Line2D(xdata=[0], ydata=[0], color=SERIES_COLOURS[0], linewidth=10 * POINTS_PER_PIXEL),
                      Line2D(xdata=[0], ydata=[0], color=SERIES_COLOURS[1], linewidth=10 * POINTS_PER_PIXEL)]
    figure.legend(handles=legend_handles, labels=["On disk (synthetic_moving_shapes/train)",
                                                  f"Drawn like the current generator ({MINIMUM_SEQUENCE_FRAMES} to "
                                                  f"{GENERATOR_MAXIMUM_FRAMES} frames)"],
                  loc="center left", bbox_to_anchor=(0.05, 0.93), ncols=2, frameon=False,
                  labelcolor=TEXT_PRIMARY_COLOUR, fontsize=18 * POINTS_PER_PIXEL, handlelength=1.2)
    save_figure(figure=figure, output_path=output_path)
    facts = {"lengths.length_panel_box": compute_pixel_box(figure=figure, axes=length_axes, with_labels=True,
                                                           margin=6),
             "lengths.padding_panel_box": compute_pixel_box(figure=figure, axes=padding_axes, with_labels=True,
                                                            margin=6)}
    pyplot.close(fig=figure)
    return facts


def make_dataset_figures(output_folder: Path, train_folder: Path, batch_size: int) -> Dict[str, Any]:
    """
    Makes the dataset figures of the series from the moving shapes dataset of the main run, and gives their facts:
    the sequences, frames and class counts of every split, the frames of the sequences on disk, the class of the shown
    sequence, the mean length drawn like the generator and the padding shares of the batches.

    Args:
        output_folder (Path): The figures folder of the series.
        train_folder (Path): The training split of the run, 'ExperimentConfiguration.train_data_folder'.
        batch_size (int): The batch size of the run.

    Returns:
        Dict[str, Any]: The facts, named dataset.* and lengths.*.
    """
    dataset_folder = train_folder.parent
    manifests = {split_name: read_manifest(split_folder=dataset_folder / split_name) for split_name in SPLIT_NAMES}
    label_mapping = json.loads((dataset_folder / "label_mapping.json").read_text(encoding="utf-8"))
    category_names = {category["id"]: category["name"] for category in manifests["train"]["categories"]}

    # Frames of the shown sequence, copied for the HTML scene, and the listings of the split and its manifest
    shown_sequence = next(sequence for sequence in manifests["train"]["sequences"]
                          if sequence["name"] == SHOWN_SEQUENCE_NAME)
    copy_frames(frame_paths=[train_folder / "data" / frame for frame in shown_sequence["frames"]],
                series_folder=output_folder, subfolder="frames", name_prefix=SHOWN_SEQUENCE_NAME)
    write_dataset_listings(output_folder=output_folder, train_manifest=manifests["train"],
                           label_mapping=label_mapping)

    # Lengths on disk, lengths drawn like the generator (one draw per sequence of the training split), padding shares
    disk_lengths = [len(sequence["frames"]) for sequence in manifests["train"]["sequences"]]
    random_generator = numpy.random.default_rng(seed=LENGTH_DRAW_SEED)
    generator_lengths = [int(random_generator.integers(low=MINIMUM_SEQUENCE_FRAMES, high=GENERATOR_MAXIMUM_FRAMES + 1))
                         for _ in disk_lengths]
    padding_shares = {"disk": compute_padding_share(lengths=disk_lengths, batch_size=batch_size,
                                                    seed=BATCH_SHUFFLE_SEED),
                      "generator": compute_padding_share(lengths=generator_lengths, batch_size=batch_size,
                                                         seed=BATCH_SHUFFLE_SEED)}
    facts = draw_lengths_figure(output_path=output_folder / "dataset" / "lengths.png", disk_lengths=disk_lengths,
                                generator_lengths=generator_lengths, padding_shares=padding_shares,
                                batch_size=batch_size)

    # Sizes and classes of the splits, the lengths on disk and the shown sequence
    first_frame = cv2.imread(filename=str(train_folder / "data" / shown_sequence["frames"][0]))
    if first_frame is None:
        raise OSError(f"The frame {train_folder / 'data' / shown_sequence['frames'][0]} cannot be read.")
    facts.update({"dataset.frame_size": int(first_frame.shape[0]),
                  "dataset.category_count": len(manifests["train"]["categories"]),
                  "dataset.first_class_index": min(label_mapping.values()),
                  "dataset.last_class_index": max(label_mapping.values()),
                  "dataset.generator_image_size": read_generator_setting(setting_name="image_sizes",
                                                                         task=GENERATOR_TASK)})
    facts.update({"dataset.shown_sequence": SHOWN_SEQUENCE_NAME,
                  "dataset.shown_class": category_names[shown_sequence["category_id"]],
                  "dataset.shown_frames": len(shown_sequence["frames"]),
                  "dataset.disk_lengths": sorted(set(disk_lengths)),
                  "dataset.disk_frames": read_single_length(lengths=disk_lengths),
                  "dataset.generator_mean_length": round_fact(value=float(numpy.mean(generator_lengths))),
                  "dataset.generator_maximum_frames": GENERATOR_MAXIMUM_FRAMES,
                  "dataset.generator_minimum_frames": MINIMUM_SEQUENCE_FRAMES,
                  "lengths.padding_share_disk": round_fact(value=padding_shares["disk"]),
                  "lengths.padding_share_generator": round_fact(value=padding_shares["generator"])})
    for split_name, manifest in manifests.items():
        class_counts = Counter(category_names[sequence["category_id"]] for sequence in manifest["sequences"])
        facts[f"dataset.{split_name}_sequences"] = len(manifest["sequences"])
        facts[f"dataset.{split_name}_frames"] = sum(len(sequence["frames"]) for sequence in manifest["sequences"])
        facts[f"dataset.{split_name}_minimum_class"] = min(class_counts.values())
        facts[f"dataset.{split_name}_maximum_class"] = max(class_counts.values())
    return facts
