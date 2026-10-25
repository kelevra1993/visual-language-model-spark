"""
Data figures of the sequence labelling tutorials (the chapter on the task and its data), made at every build by
tutorials/figures/sequence_labelling/figures.py from the real sequence labelling run of tutorial_sources.yaml and its
synthetic changing shapes data: one test sequence drawn frame by frame with the label of every frame, the class balance
of the training split (data.json of the run) with the share of no_change at every frame position (the training
labels.json), the first test sequences padded into one batch by the collate function of the repository, and the
listings of a labels.json entry and of the label mapping. The class balance gives the facts of the position prior, the
predictor knowing only where a frame is in its sequence, whose cross entropy the training curves and the evaluation
chapter compare the run with.
"""

import json
import math
import operator

import numpy

from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter, PercentFormatter
from pathlib import Path
from PIL import Image, ImageDraw
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, MONOSPACE_FONT_FAMILY,
                                                POINTS_PER_PIXEL, SERIES_COLOURS, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR,
                                                TEXT_SECONDARY_COLOUR, compute_pixel_box, format_thousands,
                                                read_reused_facts, register_fonts, round_fact, save_figure,
                                                style_chart_panel, write_reused_facts)
from tutorials.figures.sequence_labelling.sequence_utilities import (build_test_dataset, collate_named_sequences,
                                                                     convert_frame_to_image, draw_captioned_frames,
                                                                     draw_figure_header, format_sequence_entry,
                                                                     load_thumbnail_fonts, read_run_sections,
                                                                     write_listing)
from tutorials.tutorial_builder.layout import ACCENT_COLOUR, CARD_COLOUR, DIM_TEXT_COLOUR, NAMED_COLOURS, TEXT_COLOUR
from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.losses.classification_losses import compute_balanced_class_weights

# Test sequence drawn frame by frame: a star appears twice, then a rectangle disappears, its frames redrawn every time
STRIP_SEQUENCE_NAME = "test_017424"

# First test sequences of the labels file, padded into the batch of the padding figure
BATCH_SEQUENCE_COUNT = 4

# Name of the class of the frames without change, always the class of the first frame
NO_CHANGE_CLASS_NAME = "no_change"

# Frame positions of the share of no_change per position, the synthetic sequences holding 2 to 12 frames
MAXIMUM_FRAME_POSITIONS = 12

# Size of the thumbnails of the sequence strip and of the padded batch
STRIP_FRAME_SIZE = 390
BATCH_FRAME_SIZE = 110


def draw_sequence_strip(dataset: Any, output_path: Path) -> Dict[str, Any]:
    """
    Draws the frames of the strip sequence as the test dataset reads them (512 pixels, shown at 390), every frame
    captioned with its position and its class, the changes in the accent colour, and gives the box of every frame for
    the views of its scene and the labels of the sequence.

    Args:
        dataset (Any): The SequenceDataset of the test split of the run, in the order of its labels file.
        output_path (Path): The PNG file.

    Returns:
        Dict[str, Any]: The facts of the strip, named dataset.strip_*.
    """
    batch = collate_named_sequences(dataset=dataset, sequence_names=[STRIP_SEQUENCE_NAME])
    class_names = dataset.class_names
    frame_labels = batch["labels"][0].tolist()
    fonts = load_thumbnail_fonts(caption_size=24)

    # Every frame captioned with its position, then its class, a change in the accent colour
    frames = [convert_frame_to_image(frame=frame, size=STRIP_FRAME_SIZE) for frame in batch["images"][0]]
    captions = [[(f"frame {frame_index}", TEXT_COLOUR),
                 (class_names[label], DIM_TEXT_COLOUR if class_names[label] == NO_CHANGE_CLASS_NAME else ACCENT_COLOUR)]
                for frame_index, label in enumerate(frame_labels)]
    canvas, frame_boxes = draw_captioned_frames(frames=frames, captions=captions, columns=len(frames),
                                                frame_size=STRIP_FRAME_SIZE, gap=36, caption_height=100,
                                                canvas_size=(CHART_WIDTH, CHART_HEIGHT), top=110,
                                                outlines=[None] * len(frames), fonts=fonts)
    draw_figure_header(canvas=canvas, text=f"synthetic_changing_shapes/test · {STRIP_SEQUENCE_NAME} · "
                                           f"{len(frames)} frames of 512 pixels", fonts=fonts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(fp=output_path)

    facts = {"dataset.strip_sequence": STRIP_SEQUENCE_NAME, "dataset.strip_frame_count": len(frames),
             "dataset.strip_labels": [class_names[label] for label in frame_labels]}
    for frame_index, frame_box in enumerate(frame_boxes):
        facts[f"dataset.strip_frame_{frame_index}_box"] = frame_box
    facts["dataset.strip_pair_01_box"] = [frame_boxes[0][0], frame_boxes[0][1], frame_boxes[1][2], frame_boxes[1][3]]
    facts["dataset.strip_pair_23_box"] = [frame_boxes[2][0], frame_boxes[2][1], frame_boxes[3][2], frame_boxes[3][3]]
    return facts


def compute_position_shares(training_labels: Dict[str, Any], no_change_category: int) -> Dict[str, Any]:
    """
    Counts, at every frame position of the training sequences, the frames and the share of them labelled no_change,
    the prior a model can learn from the position of a frame alone.

    Args:
        training_labels (Dict[str, Any]): The labels.json of the training split.
        no_change_category (int): The category identifier of no_change.

    Returns:
        Dict[str, Any]: The 'frames' and 'no_change_shares' of every position, and the number of 'sequences'.
    """
    position_frames = numpy.zeros(shape=(MAXIMUM_FRAME_POSITIONS,), dtype=numpy.int64)
    position_no_change = numpy.zeros(shape=(MAXIMUM_FRAME_POSITIONS,), dtype=numpy.int64)
    for sequence in training_labels["sequences"]:
        category_ids = numpy.array(sequence["frame_category_ids"][:MAXIMUM_FRAME_POSITIONS])
        position_frames[:len(category_ids)] += 1
        position_no_change[:len(category_ids)] += category_ids == no_change_category
    return {"frames": position_frames, "no_change_shares": position_no_change / numpy.maximum(position_frames, 1),
            "sequences": len(training_labels["sequences"])}


def compute_prior_facts(class_frames: List[int], sequence_count: int) -> Dict[str, Any]:
    """
    Computes the cross entropy of the position prior on the training frames: no_change with probability 1 on the first
    frame of every sequence (always no_change), and the class frequencies of the later frames on the later frames,
    whose cross entropy is their entropy. A model reaching this loss knows where a frame is, not what changed.

    Args:
        class_frames (List[int]): The training frames of every class, no_change first (data.json of the run).
        sequence_count (int): The number of training sequences, one first frame each.

    Returns:
        Dict[str, Any]: The facts of the prior, named dataset.prior_*.
    """
    later_frames = numpy.array(class_frames, dtype=numpy.float64)
    later_frames[0] -= sequence_count
    later_shares = later_frames / later_frames.sum()
    later_entropy = float(-(later_shares[later_shares > 0] * numpy.log(later_shares[later_shares > 0])).sum())
    total_frames = float(sum(class_frames))
    return {"dataset.prior_later_no_change_share": round_fact(value=later_shares[0]),
            "dataset.prior_later_entropy": round_fact(value=later_entropy),
            "dataset.prior_cross_entropy": round_fact(value=later_frames.sum() / total_frames * later_entropy),
            "dataset.prior_first_frame_share": round_fact(value=sequence_count / total_frames),
            "dataset.uniform_cross_entropy": round_fact(value=math.log(len(class_frames)))}


def draw_class_balance(class_frames: List[int], position_shares: Dict[str, Any], output_path: Path) -> Dict[str, Any]:
    """
    Draws the class balance of the training split: the frames of every class (no_change first, then the 70 change
    classes in class order) on the left, and the share of no_change at every frame position on the right, and gives
    the boxes of the two panels.

    Args:
        class_frames (List[int]): The training frames of every class, in class index order.
        position_shares (Dict[str, Any]): The frames and no_change shares of every position (compute_position_shares).
        output_path (Path): The PNG file.

    Returns:
        Dict[str, Any]: The boxes of the panels, named dataset.balance_*.
    """
    register_fonts()
    figure, (class_axes, position_axes) = pyplot.subplots(nrows=1, ncols=2, figsize=(CHART_WIDTH / DOTS_PER_INCH,
                                                                                    CHART_HEIGHT / DOTS_PER_INCH),
                                                          dpi=DOTS_PER_INCH, gridspec_kw={"width_ratios": [1.6, 1.0]})
    figure.patch.set_facecolor(SURFACE_COLOUR)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.83, bottom=0.13, wspace=0.22)
    figure.text(x=0.012, y=0.955, s="data.json of the run (training split) · labels.json of the training split",
                color=TEXT_SECONDARY_COLOUR, fontsize=16 * POINTS_PER_PIXEL, va="center", family=MONOSPACE_FONT_FAMILY)

    # Frames of every class: no_change in the accent colour of the series, the change classes in the second colour
    total_frames = sum(class_frames)
    bar_colours = [SERIES_COLOURS[2]] + [SERIES_COLOURS[0]] * (len(class_frames) - 1)
    class_axes.bar(x=numpy.arange(len(class_frames)), height=class_frames, width=0.8, color=bar_colours)
    style_chart_panel(axes=class_axes, title=f"Training frames of the {len(class_frames)} classes",
                      x_label="Class index (0 = no_change, then appearing_* and disappearing_*)", y_label="Frames")
    class_axes.set_xlim(left=-1, right=len(class_frames))
    class_axes.set_ylim(bottom=0, top=max(class_frames) * 1.18)
    class_axes.yaxis.set_major_formatter(formatter=FuncFormatter(func=format_thousands))
    class_axes.annotate(text=f"no_change  {class_frames[0]:,}  ({class_frames[0] / total_frames:.1%})",
                        xy=(0, class_frames[0]), xytext=(14 * POINTS_PER_PIXEL, 6 * POINTS_PER_PIXEL),
                        textcoords="offset points", ha="left", va="bottom", color=TEXT_PRIMARY_COLOUR,
                        fontsize=19 * POINTS_PER_PIXEL)
    change_frames = class_frames[1:]
    class_axes.annotate(text=f"70 change classes: {min(change_frames):,} to {max(change_frames):,} frames each",
                        xy=(len(class_frames) / 2, max(change_frames)), xytext=(0, 40 * POINTS_PER_PIXEL),
                        textcoords="offset points", ha="center", va="bottom", color=TEXT_PRIMARY_COLOUR,
                        fontsize=19 * POINTS_PER_PIXEL)

    # Share of no_change at every frame position: 1 on the first frame, the frequency of the later frames after it
    positions = numpy.arange(MAXIMUM_FRAME_POSITIONS)
    position_axes.bar(x=positions, height=position_shares["no_change_shares"], width=0.7,
                      color=[SERIES_COLOURS[2]] + [SERIES_COLOURS[1]] * (MAXIMUM_FRAME_POSITIONS - 1))
    style_chart_panel(axes=position_axes, title="Share of no_change by frame position", x_label="Frame position",
                      y_label="Share of the frames")
    position_axes.set_ylim(bottom=0, top=1.12)
    position_axes.set_xticks(ticks=positions)
    position_axes.yaxis.set_major_formatter(formatter=PercentFormatter(xmax=1.0, decimals=0))
    position_axes.annotate(text="first frame: always no_change", xy=(0, 1.0), xytext=(16 * POINTS_PER_PIXEL, 0),
                           textcoords="offset points", ha="left", va="center", color=TEXT_PRIMARY_COLOUR,
                           fontsize=18 * POINTS_PER_PIXEL)
    later_share = float(position_shares["no_change_shares"][1:].mean())
    position_axes.annotate(text=f"later frames: about {later_share:.0%}", xy=(6, later_share),
                           xytext=(0, 30 * POINTS_PER_PIXEL), textcoords="offset points", ha="center", va="bottom",
                           color=TEXT_PRIMARY_COLOUR, fontsize=18 * POINTS_PER_PIXEL)
    save_figure(figure=figure, output_path=output_path)
    facts = {"dataset.balance_class_panel_box": compute_pixel_box(figure=figure, axes=class_axes, with_labels=True,
                                                                  margin=0),
             "dataset.balance_position_panel_box": compute_pixel_box(figure=figure, axes=position_axes,
                                                                     with_labels=True, margin=0)}
    pyplot.close(fig=figure)
    return facts


def draw_padded_batch(dataset: Any, output_path: Path) -> Dict[str, Any]:
    """
    Pads the first test sequences into one batch with the collate function of the loaders and draws it: one row per
    sequence, every real frame as a thumbnail with its class index under it, every padding frame black with the label
    -100, and the shapes of the batch, and gives the box of the grid and the batch shapes.

    Args:
        dataset (Any): The SequenceDataset of the test split of the run.
        output_path (Path): The PNG file.

    Returns:
        Dict[str, Any]: The facts of the batch, named batch.*.
    """
    sequence_names = [sequence["sequence_name"] for sequence in dataset.sequences[:BATCH_SEQUENCE_COUNT]]
    batch = collate_named_sequences(dataset=dataset, sequence_names=sequence_names)
    fonts = load_thumbnail_fonts(caption_size=22)
    canvas = Image.new(mode="RGB", size=(CHART_WIDTH, CHART_HEIGHT), color=CARD_COLOUR)
    draw = ImageDraw.Draw(im=canvas)
    draw_figure_header(canvas=canvas, text=f"collate_padded_sequences of the first {BATCH_SEQUENCE_COUNT} test "
                                           f"sequences · images {list(batch['images'].shape)} uint8 · sequence_mask "
                                           f"{list(batch['sequence_mask'].shape)} · labels "
                                           f"{list(batch['labels'].shape)}", fonts=fonts)

    # One row per sequence: its name and real frames on the left, then its frames and their labels
    grid_left, grid_top, row_height, column_width = 300, 80, BATCH_FRAME_SIZE + 44, BATCH_FRAME_SIZE + 26
    padding_colour = NAMED_COLOURS["red"]
    for sequence_index, sequence_name in enumerate(sequence_names):
        row_top = grid_top + sequence_index * row_height
        real_frames = int(batch["sequence_mask"][sequence_index].sum())
        draw.text(xy=(grid_left - 24, row_top + BATCH_FRAME_SIZE / 2 - 14), text=sequence_name, fill=TEXT_COLOUR,
                  font=fonts["monospace"], anchor="rm")
        draw.text(xy=(grid_left - 24, row_top + BATCH_FRAME_SIZE / 2 + 16), text=f"{real_frames} real frames",
                  fill=DIM_TEXT_COLOUR, font=fonts["caption"], anchor="rm")
        for frame_index in range(batch["images"].shape[1]):
            frame_left = grid_left + frame_index * column_width
            label = int(batch["labels"][sequence_index, frame_index])
            is_real = bool(batch["sequence_mask"][sequence_index, frame_index])
            canvas.paste(im=convert_frame_to_image(frame=batch["images"][sequence_index, frame_index],
                                                   size=BATCH_FRAME_SIZE), box=(frame_left, row_top))
            if not is_real:
                draw.rectangle(xy=(frame_left, row_top, frame_left + BATCH_FRAME_SIZE - 1,
                                   row_top + BATCH_FRAME_SIZE - 1), outline=padding_colour, width=2)
                draw.text(xy=(frame_left + BATCH_FRAME_SIZE / 2, row_top + BATCH_FRAME_SIZE / 2), text="padding",
                          fill=DIM_TEXT_COLOUR, font=fonts["caption"], anchor="mm")
            draw.text(xy=(frame_left + BATCH_FRAME_SIZE / 2, row_top + BATCH_FRAME_SIZE + 8), text=str(label),
                      fill=TEXT_COLOUR if is_real else padding_colour, font=fonts["monospace"], anchor="ma")
    grid_right = grid_left + batch["images"].shape[1] * column_width
    grid_bottom = grid_top + BATCH_SEQUENCE_COUNT * row_height

    # Legend of the labels under the frames
    draw.text(xy=(grid_right + 30, grid_top + 10), text="under every frame:", fill=DIM_TEXT_COLOUR,
              font=fonts["caption"])
    draw.text(xy=(grid_right + 30, grid_top + 44), text="its class index", fill=TEXT_COLOUR, font=fonts["caption"])
    draw.text(xy=(grid_right + 30, grid_top + 78), text=f"{IGNORED_LABEL_INDEX}: padding, mask False",
              fill=padding_colour, font=fonts["caption"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(fp=output_path)

    # Shapes and labels of the batch, and the box of its grid for the views
    facts = {"batch.sequence_names": sequence_names, "batch.sequence_count": len(sequence_names),
             "batch.images_shape": str(list(batch["images"].shape)),
             "batch.length": int(batch["images"].shape[1]), "batch.labels": batch["labels"].tolist(),
             "batch.real_frames": batch["sequence_mask"].sum(dim=1).tolist(),
             "batch.real_frame_total": int(batch["sequence_mask"].sum()),
             "batch.frame_slots": int(batch["sequence_mask"].numel()),
             "batch.grid_box": [grid_left - 260, grid_top - 20, grid_right + 300, grid_bottom]}
    for sequence_index in range(BATCH_SEQUENCE_COUNT):
        row_top = grid_top + sequence_index * row_height
        facts[f"batch.row_{sequence_index}_box"] = [grid_left - 10, row_top - 10, grid_right, row_top + row_height - 4]
    return facts


def write_label_listings(training_labels: Dict[str, Any], mapping_path: Path, listings_folder: Path) -> Dict[str, Any]:
    """
    Writes the listings of the labels scene: the first sequence of the training labels.json with its categories, and
    the first and last lines of the label mapping file of the run.

    Args:
        training_labels (Dict[str, Any]): The labels.json of the training split.
        mapping_path (Path): The label_mapping.json of the run.
        listings_folder (Path): The folder of the listings.

    Returns:
        Dict[str, Any]: The facts of the listings, named listings.*.
    """
    first_sequence = training_labels["sequences"][0]
    categories = {category["id"]: category["name"] for category in training_labels["categories"]}
    entry_lines = ["train/labels.json, first sequence:"] + format_sequence_entry(sequence=first_sequence,
                                                                                 maximum_listed_frames=4)
    entry_lines += ["", "categories:"] + [f'  {{"id": {category_id}, "name": "{categories[category_id]}"}}'
                                          for category_id in sorted(set(first_sequence["frame_category_ids"]))]
    write_listing(lines=entry_lines, listing_path=listings_folder / "labels_entry.txt")

    # Label mapping: its first lines, then its last one
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping_items = list(mapping.items())
    mapping_lines = ["label_mapping.json:", "{"] + [f'  "{category}": {class_index},'
                                                    for category, class_index in mapping_items[:3]]
    mapping_lines += ["  ...", f'  "{mapping_items[-1][0]}": {mapping_items[-1][1]}', "}"]
    write_listing(lines=mapping_lines, listing_path=listings_folder / "label_mapping.txt")
    return {"listings.first_sequence_categories": first_sequence["frame_category_ids"],
            "listings.mapping_size": len(mapping_items)}


def make_dataset_figures(output_folder: Path, run_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Makes the data figures and listings of the sequence labelling series and gives their facts, the class balance and
    the listings (which read the whole training labels.json) being reused from the previous build unless forced.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the sequence labelling run.
        force (bool): Whether the reused figures are made again.

    Returns:
        Dict[str, Any]: The facts, named dataset.*, batch.* and listings.*.
    """
    dataset_folder = output_folder / "dataset"
    test_dataset = build_test_dataset(run_folder=run_folder, label_type="frame")
    facts = draw_sequence_strip(dataset=test_dataset, output_path=dataset_folder / "sequence_strip.png")
    facts.update(draw_padded_batch(dataset=test_dataset, output_path=dataset_folder / "padded_batch.png"))

    # Class balance and listings, from the training labels.json of 250,000 sequences, reused when already made
    balance_path = dataset_folder / "class_balance.png"
    listings_folder = output_folder / "listings"
    reused_facts_path = dataset_folder / "class_balance_facts.json"
    reused_facts = read_reused_facts(facts_path=reused_facts_path, required_paths=[
        listings_folder / "labels_entry.txt", listings_folder / "label_mapping.txt"], force=force)
    if reused_facts is not None:
        facts.update(reused_facts)
        return facts
    experiment_configuration, model_configuration = read_run_sections(run_folder=run_folder)
    data_record = json.loads((run_folder / "data.json").read_text(encoding="utf-8"))
    class_frames = [record["frames"]["train"] for record in sorted(data_record["classes"],
                                                                     key=operator.itemgetter("class_index"))]
    training_labels = json.loads((Path(experiment_configuration["train_data_folder"]) / "labels.json").read_text(
        encoding="utf-8"))
    no_change_category = [category["id"] for category in training_labels["categories"]
                          if category["name"] == NO_CHANGE_CLASS_NAME][0]
    position_shares = compute_position_shares(training_labels=training_labels, no_change_category=no_change_category)
    balance_facts = draw_class_balance(class_frames=class_frames, position_shares=position_shares,
                                       output_path=balance_path)

    # Counts of the training split, the prior of the frame position and the balanced weights it would give
    balanced_weights = compute_balanced_class_weights(class_counts=class_frames).tolist()
    balance_facts.update({"dataset.training_frames": int(sum(class_frames)),
                          "dataset.no_change_frames": class_frames[0],
                          "dataset.no_change_share": round_fact(value=class_frames[0] / sum(class_frames)),
                          "dataset.change_class_minimum_frames": min(class_frames[1:]),
                          "dataset.change_class_maximum_frames": max(class_frames[1:]),
                          "dataset.class_count": len(class_frames),
                          "dataset.training_sequences": position_shares["sequences"],
                          "dataset.validation_frames": sum(record["frames"]["validation"]
                                                           for record in data_record["classes"]),
                          "dataset.test_frames": sum(record["frames"]["test"] for record in data_record["classes"]),
                          "dataset.position_no_change_shares": [round_fact(value=share) for share in
                                                                position_shares["no_change_shares"]],
                          "dataset.balanced_weight_no_change": round_fact(value=balanced_weights[0]),
                          "dataset.balanced_weight_change_minimum": round_fact(value=min(balanced_weights[1:])),
                          "dataset.balanced_weight_change_maximum": round_fact(value=max(balanced_weights[1:])),
                          "dataset.image_size": model_configuration["Data"]["image_settings"]["size"]})
    balance_facts.update(compute_prior_facts(class_frames=class_frames, sequence_count=position_shares["sequences"]))
    balance_facts.update(write_label_listings(
        training_labels=training_labels, mapping_path=Path(model_configuration["Data"]["label_mapping"][
            "mapping_file_path"]), listings_folder=listings_folder))
    write_reused_facts(facts_path=reused_facts_path, figure_paths=[balance_path], facts=balance_facts)
    facts.update(balance_facts)
    return facts
