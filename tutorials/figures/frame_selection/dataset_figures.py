"""
Data figures of the frame selection tutorials (the chapter on the selection problem), made at every build by
tutorials/figures/frame_selection/figures.py from the real frame selection run of tutorial_sources.yaml and its
synthetic selected shapes data: one test sequence drawn frame by frame with the class of every frame, the frames to
select outlined, the listings of a labels.json entry and of the label mapping, and the error the SequenceDataset of the
repository raises when the sequences do not hold Selection.frames_per_class frames of every wanted class, captured by
building the training split with two frames per class. The class balance of data.json and the mean ln T of the
training sequences, the selection loss of a model giving every frame the same logit, are given as facts.
"""

import copy
import json
import math
import operator
import re

import numpy

from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import CHART_HEIGHT, CHART_WIDTH, round_fact
from tutorials.figures.sequence_labelling.sequence_utilities import (build_test_dataset, collate_named_sequences,
                                                                     convert_frame_to_image, draw_captioned_frames,
                                                                     draw_figure_header, format_sequence_entry,
                                                                     load_thumbnail_fonts, read_run_sections,
                                                                     write_listing)
from tutorial_builder.layout import ACCENT_COLOUR, DIM_TEXT_COLOUR, NAMED_COLOURS, TEXT_COLOUR
from vision_training.data.sequence.sequence_dataset import SequenceDataset

# Test sequence drawn frame by frame: twelve frames, the largest circle, star and triangle among smaller copies and
# other shapes
STRIP_SEQUENCE_NAME = "test_000057"

# Size of the thumbnails of the sequence strip and frames per row
STRIP_FRAME_SIZE = 210
STRIP_COLUMNS = 6

# Frames per class asked to the training split to capture the error of the selection check, one more than the run
CHECKED_FRAMES_PER_CLASS = 2

# Lines of the captured error kept in the listing: the message and its first offending sequences
LISTED_ERROR_LINES = 4

# Escape sequences of the console (colours, underlines, OSC 8 hyperlinks) removed from a captured message
CONSOLE_CODE_PATTERN = re.compile(pattern=r"\x1b\]8;;.*?\x1b\\|\x1b\[[0-9;]*m")


def draw_sequence_strip(dataset: SequenceDataset, output_path: Path) -> Dict[str, Any]:
    """
    Draws the frames of the strip sequence as the test dataset reads them (256 pixels, shown at 210), every frame
    captioned with its position and its class, the frames to select outlined, and gives the box of every frame for
    the views of its scene.

    Args:
        dataset (SequenceDataset): The test split of the run, in the order of its labels file.
        output_path (Path): The PNG file.

    Returns:
        Dict[str, Any]: The facts of the strip, named dataset.strip_*.
    """
    batch = collate_named_sequences(dataset=dataset, sequence_names=[STRIP_SEQUENCE_NAME])
    frame_labels = batch["labels"][0].tolist()
    fonts = load_thumbnail_fonts(caption_size=22)
    unselected_class_index = dataset.unselected_class_index

    # Every frame captioned with its position and class, the frames to select outlined in the aqua of the series
    frames = [convert_frame_to_image(frame=frame, size=STRIP_FRAME_SIZE) for frame in batch["images"][0]]
    captions = [[(f"frame {frame_index}", TEXT_COLOUR),
                 (dataset.class_names[label], DIM_TEXT_COLOUR if label == unselected_class_index else ACCENT_COLOUR)]
                for frame_index, label in enumerate(frame_labels)]
    outlines = [None if label == unselected_class_index else NAMED_COLOURS["aqua"] for label in frame_labels]
    canvas, frame_boxes = draw_captioned_frames(frames=frames, captions=captions, columns=STRIP_COLUMNS,
                                                frame_size=STRIP_FRAME_SIZE, gap=26, caption_height=70,
                                                canvas_size=(CHART_WIDTH, CHART_HEIGHT), top=62, outlines=outlines,
                                                fonts=fonts)
    draw_figure_header(canvas=canvas, text=f"synthetic_selected_shapes/test · {STRIP_SEQUENCE_NAME} · "
                                           f"{len(frames)} frames of 256 pixels · outlined: the frames to select",
                       fonts=fonts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(fp=output_path)

    # Classes and boxes of the frames, and the frames to select
    facts = {"dataset.strip_sequence": STRIP_SEQUENCE_NAME, "dataset.strip_frame_count": len(frames),
             "dataset.strip_labels": [dataset.class_names[label] for label in frame_labels],
             "dataset.strip_selected_frames": {dataset.class_names[label]: frame_index
                                               for frame_index, label in enumerate(frame_labels)
                                               if label != unselected_class_index}}
    for frame_index, frame_box in enumerate(frame_boxes):
        facts[f"dataset.strip_frame_{frame_index}_box"] = frame_box
    for class_name, frame_index in facts["dataset.strip_selected_frames"].items():
        facts[f"dataset.strip_{class_name}_frame"] = frame_index
        facts[f"dataset.strip_{class_name}_box"] = frame_boxes[frame_index]
    facts["dataset.strip_first_row_box"] = [frame_boxes[0][0], frame_boxes[0][1], frame_boxes[STRIP_COLUMNS - 1][2],
                                            frame_boxes[STRIP_COLUMNS - 1][3]]
    facts["dataset.strip_second_row_box"] = [frame_boxes[STRIP_COLUMNS][0], frame_boxes[STRIP_COLUMNS][1],
                                             frame_boxes[-1][2], frame_boxes[-1][3]]
    return facts


def capture_selection_error(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> List[str]:
    """
    Builds the training split of the run with the SequenceDataset of the repository, asking two frames of every wanted
    class instead of one, and captures the error it raises at the start of a training, the console codes removed.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration of the run.
        model_configuration (Dict[str, Any]): The model sections of the run.

    Returns:
        List[str]: The lines of the error message.

    Raises:
        RuntimeError: If the dataset raises no error, the check of the selection being gone.
    """
    training_folder = Path(experiment_configuration["train_data_folder"])
    selection_settings = copy.deepcopy(model_configuration["Selection"])
    selection_settings["frames_per_class"] = CHECKED_FRAMES_PER_CLASS
    image_settings = model_configuration["Data"]["image_settings"]
    try:
        SequenceDataset(data_directory=str(training_folder / "data"), labels_file=str(training_folder / "labels.json"),
                        image_size=image_settings["size"], keep_ratio=image_settings["keep_ratio"],
                        label_dictionary=model_configuration["Data"]["label_dictionary"],
                        sequence_settings=model_configuration["Data"]["sequence_settings"], label_type="selection",
                        split_name="Training", selection_settings=selection_settings, shuffle=False)
    except ValueError as selection_error:
        return CONSOLE_CODE_PATTERN.sub(repl="", string=str(selection_error)).splitlines()
    raise RuntimeError(f"The training split built with frames_per_class {CHECKED_FRAMES_PER_CLASS} raised no error: "
                       f"the selection check of SequenceDataset is no longer where the tutorials expect it.")


def write_label_listings(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                         listings_folder: Path) -> Dict[str, Any]:
    """
    Writes the listings of the labels scene: the first sequence of the training labels.json with the categories, the
    label mapping file of the run, and the error of the selection check.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration of the run.
        model_configuration (Dict[str, Any]): The model sections of the run.
        listings_folder (Path): The folder of the listings.

    Returns:
        Dict[str, Any]: The facts of the listings and of the training split, named listings.* and dataset.*.
    """
    training_labels = json.loads((Path(experiment_configuration["train_data_folder"]) / "labels.json").read_text(
        encoding="utf-8"))
    first_sequence = training_labels["sequences"][0]
    entry_lines = ["train/labels.json, first sequence:"] + format_sequence_entry(sequence=first_sequence,
                                                                                 maximum_listed_frames=4)
    entry_lines += ["", "categories:"] + [f'  {{"id": {category["id"]}, "name": "{category["name"]}"}}'
                                          for category in training_labels["categories"]]
    write_listing(lines=entry_lines, listing_path=listings_folder / "labels_entry.txt")

    # Label mapping of the run, and the error of the selection check cut to its first offending sequences
    mapping_path = Path(model_configuration["Data"]["label_mapping"]["mapping_file_path"])
    mapping_lines = ["label_mapping.json:"] + mapping_path.read_text(encoding="utf-8").strip().splitlines()
    write_listing(lines=mapping_lines, listing_path=listings_folder / "label_mapping.txt")
    error_lines = capture_selection_error(experiment_configuration=experiment_configuration,
                                          model_configuration=model_configuration)
    message_parts = error_lines[0].split(" : ")
    listed_error = [f"ValueError: {message_parts[0]} :", f"  {Path(message_parts[-1]).name}"] + \
        error_lines[1:LISTED_ERROR_LINES] + ["  ...", error_lines[-1]]
    write_listing(lines=listed_error, listing_path=listings_folder / "selection_error.txt")

    # Mean ln T of the training sequences: the selection loss of equal logits on the T frames of every sequence
    sequence_lengths = numpy.array([len(sequence["frames"]) for sequence in training_labels["sequences"]])
    return {"listings.first_sequence_categories": first_sequence["frame_category_ids"],
            "listings.selection_error_first_line": error_lines[0].split(" : ")[0],
            "dataset.training_sequences": len(training_labels["sequences"]),
            "dataset.minimum_frames": int(sequence_lengths.min()),
            "dataset.maximum_frames": int(sequence_lengths.max()),
            "dataset.mean_log_length": round_fact(value=numpy.log(sequence_lengths).mean()),
            "dataset.log_class_count": round_fact(value=math.log(len(training_labels["categories"])))}


def make_dataset_figures(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Makes the data figure and listings of the frame selection series and gives their facts, with the class balance
    of the record of the run.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the frame selection run.

    Returns:
        Dict[str, Any]: The facts, named dataset.* and listings.*.
    """
    experiment_configuration, model_configuration = read_run_sections(run_folder=run_folder)
    test_dataset = build_test_dataset(run_folder=run_folder, label_type="selection")
    facts = draw_sequence_strip(dataset=test_dataset, output_path=output_folder / "dataset" / "sequence_strip.png")
    facts.update(write_label_listings(experiment_configuration=experiment_configuration,
                                      model_configuration=model_configuration,
                                      listings_folder=output_folder / "listings"))

    # Frames of every class in every split, from the record of the run
    data_record = json.loads((run_folder / "data.json").read_text(encoding="utf-8"))
    classes = sorted(data_record["classes"], key=operator.itemgetter("class_index"))
    training_frames = sum(record["frames"]["train"] for record in classes)
    for record in classes:
        facts[f"dataset.{record['class_name']}_training_frames"] = record["frames"]["train"]
    facts.update({"dataset.training_frames": training_frames,
                  "dataset.unselected_share": round_fact(value=classes[0]["frames"]["train"] / training_frames),
                  "dataset.test_frames": sum(record["frames"]["test"] for record in classes),
                  "dataset.validation_sequences": classes[1]["frames"]["validation"],
                  "dataset.test_sequences": classes[1]["frames"]["test"],
                  "dataset.image_size": model_configuration["Data"]["image_settings"]["size"],
                  "dataset.class_names": [record["class_name"] for record in classes]})
    return facts
