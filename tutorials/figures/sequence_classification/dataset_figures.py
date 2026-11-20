"""
Dataset figures of the sequence classification tutorials (chapter 1), made by
tutorials/figures/sequence_classification/figures.py from the moving shapes dataset the main run trained on (its
training_configuration.yaml gives the split folders): the frames of the first training sequence copied for the HTML
scene of the task, the listings of the split layout and of its labels.json entry, and the chart of the number of frames
of the training sequences with the share of padding frames their batches get for several batch sizes, the batch size of
the run marked. Its facts are the sizes of the splits, the classes, the range of lengths and the padding shares the
narration quotes. The padding shares are written as the percentages of the reports of the repository ("39.56 %", ticks
"20 %"), the facts keeping them between 0 and 1.
"""

import json
import random

import cv2
import numpy

from collections import Counter
from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH,
                                                POINTS_PER_PIXEL, SERIES_COLOURS, SURFACE_COLOUR,
                                                TEXT_PRIMARY_COLOUR, TEXT_SECONDARY_COLOUR, compute_pixel_box,
                                                register_fonts, round_fact, save_figure, style_chart_panel)
from tutorials.figures.sequence_classification.sequence_figure_utilities import (copy_frames, read_generator_setting,
                                                                                 write_text_lines)
from vision_training.evaluation.report_formatting import format_percentage, format_percentage_tick

# Sequence shown on the scene of the task, the first of the training manifest, and its number of frames, one image
# element per frame in the scene of 01_sequences_and_padded_batches.yaml: another number stops the build, the scene
# being laid out again for it
SHOWN_SEQUENCE_NAME = "train_000000"
SHOWN_FRAME_COUNT = 9

# Seed of the shuffle making the batches of the padding shares, the batch sizes of the padding panel, the batch size
# of the run being one of them, and the smaller batch size the narration compares the run with
BATCH_SHUFFLE_SEED = 0
PADDING_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
SMALL_PADDING_BATCH_SIZE = 8

# Splits of a dataset, in the order of the manifests
SPLIT_NAMES = ["train", "validation", "test"]

# Task of the synthetic generator whose main block settings the facts read
GENERATOR_TASK = "sequence_classification"


def read_generator_frame_range() -> Tuple[int, int]:
    """
    Reads the fewest and the most frames the synthetic generator draws for a sequence of the task, the minimum_frames
    and maximum_frames of its main block, so that the animated batch and the narration follow the script rather than a
    typed range.

    Returns:
        Tuple[int, int]: The fewest and the most frames of a generated sequence.
    """
    return (read_generator_setting(setting_name="minimum_frames", task=GENERATOR_TASK),
            read_generator_setting(setting_name="maximum_frames", task=GENERATOR_TASK))


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


def compute_padding_share(lengths: List[int], batch_size: int, seed: int) -> float:
    """
    Gives the share of padding frames in the batches of a split, the sequences shuffled once like the dataset does
    and batched in order like the loader, every batch padded to its longest sequence by collate_padded_sequences.

    Args:
        lengths (List[int]): The frames of every sequence of the split.
        batch_size (int): The number of sequences per batch.
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


def draw_lengths_figure(output_path: Path, disk_lengths: List[int], padding_shares: Dict[int, float],
                        batch_size: int) -> Dict[str, Any]:
    """
    Draws the lengths chart: the number of training sequences of every number of frames, and the share of padding
    frames in the batches of the training split for every batch size of PADDING_BATCH_SIZES, the batch size of the run
    in the colour of the series and the others dimmed; gives the boxes of its panels.

    Args:
        output_path (Path): The PNG file.
        disk_lengths (List[int]): The frames of every training sequence.
        padding_shares (Dict[int, float]): The padding share of the batches for every batch size.
        batch_size (int): The batch size of the run, whose bar is highlighted.

    Returns:
        Dict[str, Any]: The pixel boxes of the panels, named lengths.*.
    """
    register_fonts()
    figure, (length_axes, padding_axes) = pyplot.subplots(
        nrows=1, ncols=2, figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH,
        gridspec_kw={"width_ratios": [1.5, 1.0]})
    figure.patch.set_facecolor(SURFACE_COLOUR)
    figure.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.12, wspace=0.24)

    # Training sequences of every number of frames, the count written over every bar
    all_lengths = list(range(min(disk_lengths), max(disk_lengths) + 1))
    disk_counts = Counter(disk_lengths)
    length_axes.bar(x=all_lengths, height=[disk_counts[length] for length in all_lengths], width=0.7,
                    color=SERIES_COLOURS[0])
    style_chart_panel(axes=length_axes, title=f"{len(disk_lengths):,} training sequences per number of frames",
                      x_label="Frames of the sequence", y_label="Sequences")
    length_axes.set_xticks(ticks=all_lengths)
    length_axes.set_ylim(bottom=0, top=max(disk_counts.values()) * 1.15)
    length_axes.grid(visible=False, axis="x")
    for length in all_lengths:
        length_axes.annotate(text=f"{disk_counts[length]:,}", xy=(length, disk_counts[length]), xytext=(0, 6),
                             textcoords="offset points", ha="center", va="bottom", color=TEXT_PRIMARY_COLOUR,
                             fontsize=16 * POINTS_PER_PIXEL)

    # Padding share of the batches for every batch size, the batch size of the run highlighted, the value of every bar
    # written as the percentages of the reports
    batch_sizes = sorted(padding_shares)
    share_values = [padding_shares[size] for size in batch_sizes]
    bar_colours = [SERIES_COLOURS[1] if size == batch_size else TEXT_SECONDARY_COLOUR for size in batch_sizes]
    padding_axes.bar(x=range(len(batch_sizes)), height=share_values, width=0.7, color=bar_colours)
    style_chart_panel(axes=padding_axes, title="Padding frames per batch size", x_label="Sequences per batch",
                      y_label="Share of the frames")
    padding_axes.set_xticks(ticks=range(len(batch_sizes)), labels=[str(size) for size in batch_sizes])
    padding_axes.set_ylim(bottom=0.0, top=max(0.1, max(share_values) * 1.2))
    padding_axes.grid(visible=False, axis="x")
    padding_axes.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))
    for bar_index, share_value in enumerate(share_values):
        padding_axes.annotate(text=format_percentage(value=round_fact(value=share_value)),
                              xy=(bar_index, share_value), xytext=(0, 6), textcoords="offset points", ha="center",
                              va="bottom", color=TEXT_PRIMARY_COLOUR, fontsize=15 * POINTS_PER_PIXEL)

    # Note of the highlighted batch size above the panels
    figure.text(x=0.98, y=0.95, s=f"Highlighted: the batch size of the run, {batch_size} sequences", ha="right",
                va="center", color=TEXT_SECONDARY_COLOUR, fontsize=17 * POINTS_PER_PIXEL)
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
    the sequences, frames and class counts of every split, the class names in the order of their indices, the range
    of frames of the sequences on disk and of the generator, the class and frames of the shown sequence and the
    padding shares of the batches.

    Args:
        output_folder (Path): The figures folder of the series.
        train_folder (Path): The training split of the run, 'ExperimentConfiguration.train_data_folder'.
        batch_size (int): The batch size of the run.

    Returns:
        Dict[str, Any]: The facts, named dataset.* and lengths.*.

    Raises:
        ValueError: If the shown sequence does not have the frames its scene lays out, or the batch size of the run is
            not one of the padding panel.
    """
    dataset_folder = train_folder.parent
    manifests = {split_name: read_manifest(split_folder=dataset_folder / split_name) for split_name in SPLIT_NAMES}
    label_mapping = json.loads((dataset_folder / "label_mapping.json").read_text(encoding="utf-8"))
    category_names = {category["id"]: category["name"] for category in manifests["train"]["categories"]}

    # Frames of the shown sequence, copied for the HTML scene, whose image elements expect SHOWN_FRAME_COUNT frames
    shown_sequence = next(sequence for sequence in manifests["train"]["sequences"]
                          if sequence["name"] == SHOWN_SEQUENCE_NAME)
    if len(shown_sequence["frames"]) != SHOWN_FRAME_COUNT:
        raise ValueError(f"The shown sequence {SHOWN_SEQUENCE_NAME} has {len(shown_sequence['frames'])} frames, while "
                         f"the scene of the task lays out {SHOWN_FRAME_COUNT}: lay out the image elements of "
                         f"01_sequences_and_padded_batches.yaml again and update SHOWN_FRAME_COUNT.")
    copy_frames(frame_paths=[train_folder / "data" / frame for frame in shown_sequence["frames"]],
                series_folder=output_folder, subfolder="frames", name_prefix=SHOWN_SEQUENCE_NAME)
    write_dataset_listings(output_folder=output_folder, train_manifest=manifests["train"],
                           label_mapping=label_mapping)

    # Lengths of the training sequences and the padding share of their batches for every batch size
    disk_lengths = [len(sequence["frames"]) for sequence in manifests["train"]["sequences"]]
    if batch_size not in PADDING_BATCH_SIZES:
        raise ValueError(f"The batch size of the run, {batch_size}, is not among the batch sizes of the padding "
                         f"panel {PADDING_BATCH_SIZES}.")
    padding_shares = {size: compute_padding_share(lengths=disk_lengths, batch_size=size, seed=BATCH_SHUFFLE_SEED)
                      for size in PADDING_BATCH_SIZES}
    facts = draw_lengths_figure(output_path=output_folder / "dataset" / "lengths.png", disk_lengths=disk_lengths,
                                padding_shares=padding_shares, batch_size=batch_size)

    # Sizes and classes of the splits, the lengths on disk, the range the generator draws and the shown sequence
    first_frame = cv2.imread(filename=str(train_folder / "data" / shown_sequence["frames"][0]))
    if first_frame is None:
        raise OSError(f"The frame {train_folder / 'data' / shown_sequence['frames'][0]} cannot be read.")
    generator_frame_range = read_generator_frame_range()
    split_lengths = [len(sequence["frames"]) for manifest in manifests.values() for sequence in manifest["sequences"]]
    facts.update({"dataset.frame_size": int(first_frame.shape[0]),
                  "dataset.category_count": len(manifests["train"]["categories"]),
                  "dataset.first_class_index": min(label_mapping.values()),
                  "dataset.last_class_index": max(label_mapping.values()),
                  "dataset.class_names": [category_names[int(category_identifier)] for category_identifier, _ in
                                          sorted(label_mapping.items(), key=lambda item: item[1])],
                  "dataset.generator_image_size": read_generator_setting(setting_name="image_sizes",
                                                                         task=GENERATOR_TASK),
                  "dataset.generator_minimum_frames": generator_frame_range[0],
                  "dataset.generator_maximum_frames": generator_frame_range[1]})

    # Class and frames of the shown sequence, the lengths on disk, then the padding share of every batch size
    facts.update({"dataset.shown_sequence": SHOWN_SEQUENCE_NAME,
                  "dataset.shown_class": category_names[shown_sequence["category_id"]],
                  "dataset.shown_frames": len(shown_sequence["frames"]),
                  "dataset.disk_lengths": sorted(set(disk_lengths)),
                  "dataset.disk_minimum_frames": min(split_lengths),
                  "dataset.disk_maximum_frames": max(split_lengths),
                  "dataset.train_mean_frames": round_fact(value=float(numpy.mean(disk_lengths)))})
    for size, share in padding_shares.items():
        facts[f"lengths.padding_share_batch_{size}"] = round_fact(value=share)
    # The narration says that the smaller batch pads less than the batches of the run
    if not (SMALL_PADDING_BATCH_SIZE < batch_size and
            padding_shares[SMALL_PADDING_BATCH_SIZE] < padding_shares[batch_size]):
        raise ValueError(f"Batches of {SMALL_PADDING_BATCH_SIZE} sequences should be smaller and pad less than the "
                         f"batches of {batch_size} of the run, as the padding scene says.")
    facts.update({"lengths.padding_share_run": round_fact(value=padding_shares[batch_size]),
                  "lengths.padding_batch_small": SMALL_PADDING_BATCH_SIZE,
                  "lengths.padding_share_batch_small": round_fact(value=padding_shares[SMALL_PADDING_BATCH_SIZE])})

    # Sequences, frames and smallest and largest class of every split
    for split_name, manifest in manifests.items():
        class_counts = Counter(category_names[sequence["category_id"]] for sequence in manifest["sequences"])
        facts[f"dataset.{split_name}_sequences"] = len(manifest["sequences"])
        facts[f"dataset.{split_name}_frames"] = sum(len(sequence["frames"]) for sequence in manifest["sequences"])
        facts[f"dataset.{split_name}_minimum_class"] = min(class_counts.values())
        facts[f"dataset.{split_name}_maximum_class"] = max(class_counts.values())
    return facts
