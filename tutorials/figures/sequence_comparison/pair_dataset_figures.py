"""
Dataset figures of the sequence comparison tutorials (chapter 1), made by tutorials/figures/sequence_comparison/
figures.py from the object loss pairs dataset the comparison run trained on: the frames of a training pair and of a
test pair copied for the HTML scenes, the listings of the split layout and of its labels.json entry, the chart of the
member lengths of every class, the mask of one training batch built by the pair dataset and the collate of the loaders,
and the share of padding frames in the batches of the run. Its facts are the sizes of the splits, the classes and
their class indices, the size of the frames on disk, the lengths and the padding the narration quotes.
"""

import json
import random

import numpy
import torch

from collections import Counter
from matplotlib import pyplot
from matplotlib.lines import Line2D
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, POINTS_PER_PIXEL,
                                                SERIES_COLOURS, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR, compute_pixel_box,
                                                register_fonts, round_fact, save_figure, style_chart_panel)
from tutorials.figures.sequence_classification.sequence_figure_utilities import copy_frames, write_text_lines
from vision_training.data.sequence.sequence_dataset import collate_padded_sequences
from vision_training.data.sequence.sequence_pair_dataset import PAIR_MEMBERS, SequencePairDataset

# Pairs shown on the scenes: the first training pair, and a test pair of the other class that the run gets wrong
SHOWN_TRAINING_PAIR = "train_000000"
SHOWN_TEST_PAIR = "test_000003"

# Splits of a dataset, and the seed of the shuffle making the batches of the run
SPLIT_NAMES = ["train", "validation", "test"]
BATCH_SHUFFLE_SEED = 0


def read_pair_manifest(split_folder: Path) -> Dict[str, Any]:
    """
    Reads the labels.json manifest of a split of the object loss pairs dataset.

    Args:
        split_folder (Path): The split folder, holding labels.json and data/.

    Returns:
        Dict[str, Any]: The manifest, with its info, categories and pairs.

    Raises:
        FileNotFoundError: If the manifest is missing.
    """
    labels_path = split_folder / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"No manifest {labels_path}.")
    return json.loads(labels_path.read_text(encoding="utf-8"))


def write_pair_listings(output_folder: Path, train_manifest: Dict[str, Any], label_mapping: Dict[str, int]) -> None:
    """
    Writes the two listings of the data scene: the layout of the training split with the member folders of its first
    pair, and its labels.json with the categories, the first pair entry and the label mapping file.

    Args:
        output_folder (Path): The figures folder of the series.
        train_manifest (Dict[str, Any]): The manifest of the training split.
        label_mapping (Dict[str, int]): The label_mapping.json of the dataset.
    """
    pairs = train_manifest["pairs"]
    first_pair = pairs[0]
    tree_lines = ["train/", "├── data/", f"│   ├── {first_pair['name']}/",
                  f"│   │   ├── before/    {len(first_pair['before'])} frames",
                  f"│   │   └── after/     {len(first_pair['after'])} frames",
                  f"│   └── ... {len(pairs) - 1:,} more pair folders", "└── labels.json"]
    write_text_lines(output_path=output_folder / "listings" / "split_tree.txt", lines=tree_lines)
    category_lines = [f'    {{"id": {category["id"]}, "name": "{category["name"]}"}}' +
                      ("," if category_index < len(train_manifest["categories"]) - 1 else "")
                      for category_index, category in enumerate(train_manifest["categories"])]
    mapping_text = ", ".join(f'"{key}": {value}' for key, value in label_mapping.items())
    manifest_lines = (["{", '  "categories": ['] + category_lines + ["  ],", '  "pairs": [',
                      f'    {{"id": {first_pair["id"]}, "name": "{first_pair["name"]}", '
                      f'"category_id": {first_pair["category_id"]},',
                      f'     "before": ["{first_pair["before"][0]}", ... {len(first_pair["before"])} frames],',
                      f'     "after": ["{first_pair["after"][0]}", ... {len(first_pair["after"])} frames]}},',
                      f"    ... {len(pairs) - 1:,} more pairs", "  ]", "}", "",
                      f"label_mapping.json  {{{mapping_text}}}"])
    write_text_lines(output_path=output_folder / "listings" / "labels_entry.txt", lines=manifest_lines)


def compute_pair_padding_share(pairs: List[Dict[str, Any]], batch_size: int, seed: int) -> float:
    """
    Gives the share of padding frames in the batches of a split of pairs, the pairs shuffled once like the dataset
    does and batched in order like the loader: every member is padded to the longer member of its pair by the pair
    dataset, then to the longest member of the batch by the collate.

    Args:
        pairs (List[Dict[str, Any]]): The pairs of the split, with their before and after frames.
        batch_size (int): The batch size of the run.
        seed (int): The seed of the shuffle.

    Returns:
        float: The padding frames over all the frames of the padded batches.
    """
    shuffled_pairs = list(pairs)
    random.Random(seed).shuffle(shuffled_pairs)
    real_frames, padded_frames = 0, 0
    for batch_start in range(0, len(shuffled_pairs), batch_size):
        batch_pairs = shuffled_pairs[batch_start:batch_start + batch_size]
        longest_member = max(len(pair[member]) for pair in batch_pairs for member in PAIR_MEMBERS)
        padded_frames += len(batch_pairs) * len(PAIR_MEMBERS) * longest_member
        real_frames += sum(len(pair[member]) for pair in batch_pairs for member in PAIR_MEMBERS)
    return 1.0 - real_frames / padded_frames


def draw_member_length_figure(output_path: Path, pairs: List[Dict[str, Any]],
                              category_names: Dict[int, str]) -> Dict[str, Any]:
    """
    Draws the number of frames of the before and after members of the training pairs, one panel per class, so that
    the scene shows that the lengths do not give the class away; gives the boxes of the panels.

    Args:
        output_path (Path): The PNG file.
        pairs (List[Dict[str, Any]]): The training pairs.
        category_names (Dict[int, str]): The name of every category.

    Returns:
        Dict[str, Any]: The boxes, named lengths.*.
    """
    register_fonts()
    figure, panels = pyplot.subplots(nrows=1, ncols=len(category_names), figsize=(CHART_WIDTH / DOTS_PER_INCH,
                                                                                  CHART_HEIGHT / DOTS_PER_INCH),
                                     dpi=DOTS_PER_INCH, sharey=True)
    figure.patch.set_facecolor(SURFACE_COLOUR)
    figure.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.12, wspace=0.08)
    all_lengths = sorted({len(pair[member]) for pair in pairs for member in PAIR_MEMBERS})
    bar_width = 0.4
    facts = {}
    for panel_index, (axes, (category_identifier, category_name)) in enumerate(zip(panels, category_names.items())):
        class_pairs = [pair for pair in pairs if pair["category_id"] == category_identifier]
        for member_index, member in enumerate(PAIR_MEMBERS):
            counts = Counter(len(pair[member]) for pair in class_pairs)
            axes.bar(x=numpy.array(all_lengths) + (member_index - 0.5) * bar_width,
                     height=[counts[length] for length in all_lengths], width=bar_width - 0.04,
                     color=SERIES_COLOURS[1] if member_index == 0 else SERIES_COLOURS[3])
        style_chart_panel(axes=axes, title=f"{category_name} · {len(class_pairs):,} training pairs",
                          x_label="Frames of the member", y_label="Members" if panel_index == 0 else None)
        axes.set_xticks(ticks=all_lengths)
        axes.grid(visible=False, axis="x")
        facts[f"lengths.panel_{panel_index}_box"] = compute_pixel_box(figure=figure, axes=axes, with_labels=True,
                                                                      margin=6)
    legend_handles = [Line2D(xdata=[0], ydata=[0], color=SERIES_COLOURS[1], linewidth=10 * POINTS_PER_PIXEL),
                      Line2D(xdata=[0], ydata=[0], color=SERIES_COLOURS[3], linewidth=10 * POINTS_PER_PIXEL)]
    figure.legend(handles=legend_handles, labels=["before member", "after member"], loc="center left",
                  bbox_to_anchor=(0.05, 0.93), ncols=2, frameon=False, labelcolor=TEXT_PRIMARY_COLOUR,
                  fontsize=18 * POINTS_PER_PIXEL, handlelength=1.2)
    save_figure(figure=figure, output_path=output_path)
    pyplot.close(fig=figure)
    return facts


def build_training_batch_mask(train_folder: Path, label_dictionary: Dict[int, int], batch_size: int,
                              seed: int) -> torch.Tensor:
    """
    Builds the sequence mask of one training batch with the code of the loaders: the pair dataset (every member read
    and padded to the longer member of its pair) shuffled with a fixed seed, its first batch_size pairs batched by
    collate_padded_sequences with two members per sample. The frames are read at a small size, the mask not depending
    on the frame size.

    Args:
        train_folder (Path): The training split.
        label_dictionary (Dict[int, int]): The class index of every category.
        batch_size (int): The batch size of the run.
        seed (int): The seed of the shuffle.

    Returns:
        torch.Tensor: The [B, 2T] boolean mask, the before member then the after member of every pair.
    """
    dataset = SequencePairDataset(data_directory=str(train_folder / "data"),
                                  labels_file=str(train_folder / "labels.json"), image_size=32, keep_ratio=True,
                                  label_dictionary=label_dictionary,
                                  sequence_settings={"maximum_sequence_length": 10 ** 6}, split_name="Training",
                                  shuffle=False)
    pair_indices = list(range(len(dataset)))
    random.Random(seed).shuffle(pair_indices)
    batch = collate_padded_sequences(samples=[dataset[pair_index] for pair_index in pair_indices[:batch_size]],
                                     members_per_sample=2)
    return batch["sequence_mask"]


def make_pair_dataset_figures(output_folder: Path, train_folder: Path, label_dictionary: Dict[int, int],
                              batch_size: int) -> Dict[str, Any]:
    """
    Makes the dataset figures of the series from the object loss pairs dataset of the run, and gives their facts.

    Args:
        output_folder (Path): The figures folder of the series.
        train_folder (Path): The training split of the run.
        label_dictionary (Dict[int, int]): The class index of every category.
        batch_size (int): The batch size of the run.

    Returns:
        Dict[str, Any]: The facts, named dataset.*, lengths.* and mask.*.
    """
    dataset_folder = train_folder.parent
    manifests = {split_name: read_pair_manifest(split_folder=dataset_folder / split_name)
                 for split_name in SPLIT_NAMES}
    label_mapping = json.loads((dataset_folder / "label_mapping.json").read_text(encoding="utf-8"))
    category_names = {category["id"]: category["name"] for category in manifests["train"]["categories"]}

    # Frames of the shown pairs, copied for the HTML scenes, and the listings of the split and its manifest
    facts = {}
    for split_name, pair_name in [("train", SHOWN_TRAINING_PAIR), ("test", SHOWN_TEST_PAIR)]:
        pair = next(pair for pair in manifests[split_name]["pairs"] if pair["name"] == pair_name)
        for member in PAIR_MEMBERS:
            copy_frames(frame_paths=[dataset_folder / split_name / "data" / frame for frame in pair[member]],
                        series_folder=output_folder, subfolder="frames", name_prefix=f"{pair_name}_{member}")
            facts[f"dataset.{pair_name}.{member}_frames"] = len(pair[member])
        facts[f"dataset.{pair_name}.class"] = category_names[pair["category_id"]]
    write_pair_listings(output_folder=output_folder, train_manifest=manifests["train"], label_mapping=label_mapping)

    # Member lengths of every class, the mask of one training batch and the padding of the batches of the run
    train_pairs = manifests["train"]["pairs"]
    facts.update(draw_member_length_figure(output_path=output_folder / "dataset" / "member_lengths.png",
                                           pairs=train_pairs, category_names=category_names))
    for category_identifier, category_name in category_names.items():
        differences = [len(pair["after"]) - len(pair["before"]) for pair in train_pairs
                       if pair["category_id"] == category_identifier]
        facts[f"lengths.{category_name}.mean_difference"] = round_fact(value=float(numpy.mean(differences)))
    facts["lengths.equal_members"] = sum(int(len(pair["before"]) == len(pair["after"])) for pair in train_pairs)
    all_lengths = [len(pair[member]) for pair in train_pairs for member in PAIR_MEMBERS]
    facts.update({"lengths.minimum_member": min(all_lengths), "lengths.maximum_member": max(all_lengths),
                  "lengths.padding_share": round_fact(value=compute_pair_padding_share(
                      pairs=train_pairs, batch_size=batch_size, seed=BATCH_SHUFFLE_SEED))})
    batch_mask = build_training_batch_mask(train_folder=train_folder, label_dictionary=label_dictionary,
                                           batch_size=batch_size, seed=BATCH_SHUFFLE_SEED)
    facts.update({"mask.values": batch_mask.int().tolist(), "mask.shape": list(batch_mask.shape),
                  "mask.longest_member": int(batch_mask.shape[1]) // 2,
                  "mask.padding_share": round_fact(value=float((~batch_mask).float().mean()))})

    # Class index the label mapping gives every category, and the side of the square frames written on disk
    for category_identifier, class_index in label_mapping.items():
        facts[f"dataset.class_index.{category_names[int(category_identifier)]}"] = class_index
    with Image.open(fp=train_folder / "data" / train_pairs[0]["before"][0]) as first_frame:
        frame_width, frame_height = first_frame.size
    if frame_width != frame_height:
        raise ValueError(f"The frames of {train_folder} are expected square, the first one is {frame_width} x "
                         f"{frame_height} pixels.")
    facts["dataset.frame_size"] = frame_width

    # Sizes and classes of the splits
    for split_name, manifest in manifests.items():
        class_counts = Counter(category_names[pair["category_id"]] for pair in manifest["pairs"])
        facts[f"dataset.{split_name}_pairs"] = len(manifest["pairs"])
        facts[f"dataset.{split_name}_frames"] = sum(len(pair[member]) for pair in manifest["pairs"]
                                                    for member in PAIR_MEMBERS)
        for category_name, count in class_counts.items():
            facts[f"dataset.{split_name}.{category_name}"] = count
    return facts
