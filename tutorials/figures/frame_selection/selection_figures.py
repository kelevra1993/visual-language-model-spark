"""
Selection figures of the frame selection tutorials (the chapters on the decoding and on the evaluation), made at every
build by tutorials/figures/frame_selection/figures.py: the checkpoint of iteration 1000 of the real frame selection
run, run again on the CPU on the three test sequences where it picks a wrong frame (all three for the triangle), gives
their probabilities and logits; the selection images of the repository (build_frame_selection_image) are drawn from
them, stacked into one figure, and the probability, gain and selection matrices of one of them, decoded by the Hungarian
and greedy decodings of the repository, are written as the data of the matrix animations, with the example of the
tasks reference (documentation/reference/tasks_reference.md, formerly in the README) decoded the same way. The
probabilities of the CPU run are checked against selections.csv of the evaluation of that iteration, so that the
figures show what the evaluation saw: the standalone evaluation of the run in its Checkpoint_Evaluation folder
(EVALUATION_FOLDER_NAME), whose report the evaluation chapter lists.
"""

import csv
import json

import numpy
import torch

from pathlib import Path
from PIL import Image, ImageDraw
from typing import Any, Dict, List, Tuple

from tutorials.figures.figure_utilities import read_reused_facts, round_fact, write_reused_facts
from tutorials.figures.sequence_labelling.sequence_utilities import (build_test_dataset, collate_named_sequences,
                                                                     count_model_parameters, load_sequence_model,
                                                                     load_thumbnail_fonts, predict_sequences)
from tutorial_builder.layout import CARD_COLOUR, TEXT_COLOUR
from vision_training.evaluation.checkpoint_evaluation import CHECKPOINT_EVALUATION_FOLDER_NAME
from vision_training.models.sequence.frame_encoder import encode_frames
from vision_training.models.sequence.frame_selection_decoding import (decode_frame_selection, flag_low_confidence,
                                                                      gather_frame_probabilities,
                                                                      get_wanted_class_indices)
from vision_training.utilities.visualization import build_frame_selection_image, convert_image_to_array

# Folder of the evaluations the figures read, inside the run folder: both checkpoints evaluated again with the code of
# the percent display by scripts/evaluation/evaluate_checkpoint.py, the Results.txt of the training having been written
# with every rate between 0 and 1
EVALUATION_FOLDER_NAME = CHECKPOINT_EVALUATION_FOLDER_NAME

# Checkpoint whose wrong picks are shown, and the test sequences where it picks a wrong frame (selections.csv)
SELECTION_ITERATION = 1000
WRONG_PICK_SEQUENCES = ["test_000004", "test_000134", "test_000052"]

# Sequence of the matrix animations of the decoding and of the selection loss
MATRIX_SEQUENCE_NAME = "test_000052"

# Largest difference allowed between a probability of the CPU run and the one of selections.csv (GPU evaluation)
PROBABILITY_TOLERANCE = 0.005

# Example of documentation/reference/tasks_reference.md (Frame Selection), formerly in the README, hence the names:
# two wanted classes A and B and three frames, the probabilities of every frame in the order A, B, unselected
README_CLASS_NAMES = ["A", "B", "unselected"]
README_PROBABILITIES = [[0.50, 0.45, 0.05], [0.45, 0.05, 0.50], [0.02, 0.08, 0.90]]

# Height of the heading band over every selection image of the stacked figure, and the gap between two images
HEADING_HEIGHT = 56
IMAGE_GAP = 24


def read_selection_rows(selections_path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Reads selections.csv of an evaluation, one row per test sequence and wanted class.

    Args:
        selections_path (Path): The selections.csv file.

    Returns:
        Dict[Tuple[str, str], Dict[str, str]]: The row of every sequence and class.
    """
    with selections_path.open(mode="r", encoding="utf-8", newline="") as selections_file:
        return {(row["sequence"], row["class"]): row for row in csv.DictReader(f=selections_file)}


def find_frame_index(frame_file: str) -> int:
    """
    Gives the position of a frame in its sequence from its file, 'test_000004/000008.png' giving 8.

    Args:
        frame_file (str): The frame file, relative to the data folder.

    Returns:
        int: The position of the frame.
    """
    return int(Path(frame_file).stem)


def find_sequence_number(sequence_name: str) -> int:
    """
    Gives the number of a test sequence from its name, 'test_000052' giving 52, so that the narration names the
    sequences of the wrong picks and of the matrix animations by the number the selection images caption.

    Args:
        sequence_name (str): The name of the sequence in the labels file, its number after the last underscore.

    Returns:
        int: The number of the sequence.
    """
    return int(sequence_name.rsplit("_", maxsplit=1)[-1])


def check_against_evaluation(probabilities: numpy.ndarray, sequence_name: str, class_names: Dict[int, str],
                             wanted_class_indices: List[int], selected_frames: numpy.ndarray,
                             selection_rows: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    """
    Checks that the CPU run of the checkpoint chooses the frames the evaluation of the run chose (selections.csv) with
    the same probabilities, within PROBABILITY_TOLERANCE, so that the figures show the evaluation as it was.

    Args:
        probabilities (numpy.ndarray): The [T, K] probabilities of the CPU run.
        sequence_name (str): The test sequence.
        class_names (Dict[int, str]): The name of every class index.
        wanted_class_indices (List[int]): The wanted classes, one per row of the selection.
        selected_frames (numpy.ndarray): The [W, 1] frames chosen by the Hungarian decoding of the CPU run.
        selection_rows (Dict[Tuple[str, str], Dict[str, str]]): The rows of selections.csv.

    Raises:
        ValueError: If a chosen frame or a probability differs from the evaluation.
    """
    for wanted_index, class_index in enumerate(wanted_class_indices):
        row = selection_rows[(sequence_name, class_names[class_index])]
        evaluated_frame = find_frame_index(frame_file=row["hungarian_frames"])
        evaluated_probability = float(row["hungarian_probabilities"])
        cpu_probability = probabilities[selected_frames[wanted_index, 0], class_index]
        if evaluated_frame != selected_frames[wanted_index, 0] or \
                abs(evaluated_probability - cpu_probability) > PROBABILITY_TOLERANCE:
            raise ValueError(f"The CPU run chooses frame {selected_frames[wanted_index, 0]} at {cpu_probability:.4f} "
                             f"for {class_names[class_index]} in {sequence_name}, the evaluation frame "
                             f"{evaluated_frame} at {evaluated_probability:.4f}.")


def stack_selection_images(selection_images: List[numpy.ndarray], headings: List[str],
                           output_path: Path) -> List[List[int]]:
    """
    Stacks the selection images of the repository (BGR arrays, shown as they are) into one figure on the card colour,
    every image under a heading band, and gives the box of every heading and image for the views of the scene.

    Args:
        selection_images (List[numpy.ndarray]): The [H, W, 3] uint8 BGR images of build_frame_selection_image.
        headings (List[str]): The heading of every image.
        output_path (Path): The PNG file.

    Returns:
        List[List[int]]: The left, top, right and bottom of every heading with its image.
    """
    width = max(image.shape[1] for image in selection_images) + 2 * IMAGE_GAP
    height = sum(image.shape[0] + HEADING_HEIGHT + IMAGE_GAP for image in selection_images) + IMAGE_GAP
    canvas = Image.new(mode="RGB", size=(width, height), color=CARD_COLOUR)
    draw = ImageDraw.Draw(im=canvas)
    fonts = load_thumbnail_fonts(caption_size=26)
    boxes, top = [], IMAGE_GAP

    # Every image under its heading, its colours turned from BGR to RGB, left aligned
    for selection_image, heading in zip(selection_images, headings):
        draw.text(xy=(IMAGE_GAP, top + 10), text=heading, fill=TEXT_COLOUR, font=fonts["caption_bold"])
        canvas.paste(im=Image.fromarray(obj=numpy.ascontiguousarray(selection_image[:, :, ::-1])),
                     box=(IMAGE_GAP, top + HEADING_HEIGHT))
        bottom = top + HEADING_HEIGHT + selection_image.shape[0]
        heading_width = draw.textlength(text=heading, font=fonts["caption_bold"])
        boxes.append([IMAGE_GAP - 8, top, IMAGE_GAP + int(max(selection_image.shape[1], heading_width)) + 8,
                      bottom + 8])
        top = bottom + IMAGE_GAP
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(fp=output_path)
    return boxes


def compute_column_softmax(logits: numpy.ndarray) -> numpy.ndarray:
    """
    Normalises every class column of the logits of a sequence over its frames, the distribution the selection loss
    trains towards the true frames of the class.

    Args:
        logits (numpy.ndarray): The [T, K] class logits of the frames of the sequence.

    Returns:
        numpy.ndarray: The [T, K] probabilities of every frame for every class, each column summing to 1.
    """
    shifted_logits = logits - logits.max(axis=0, keepdims=True)
    exponentials = numpy.exp(shifted_logits)
    return exponentials / exponentials.sum(axis=0, keepdims=True)


def build_matrix_data(probabilities: numpy.ndarray, logits: numpy.ndarray, frame_labels: List[int],
                      class_names: List[str], unselected_class_index: int) -> Dict[str, Any]:
    """
    Gathers the matrices of one real sequence for the animations of the decoding and of the selection loss: the
    per-frame probabilities, the selection distributions of the wanted classes over the frames, the Hungarian gains,
    the frames chosen by both decodings of the repository and the true frames.

    Args:
        probabilities (numpy.ndarray): The [T, K] class probabilities.
        logits (numpy.ndarray): The [T, K] class logits.
        frame_labels (List[int]): The true class of every frame.
        class_names (List[str]): The name of every class index.
        unselected_class_index (int): The class index of the unselected frames.

    Returns:
        Dict[str, Any]: The rows, columns, value sets, picks and true cells of the matrix.
    """
    wanted_class_indices = get_wanted_class_indices(number_classes=probabilities.shape[1],
                                                    unselected_class_index=unselected_class_index)
    log_probabilities = numpy.log(numpy.clip(a=probabilities, a_min=1e-12, a_max=None))
    gains = log_probabilities - log_probabilities[:, [unselected_class_index]]
    column_softmax = compute_column_softmax(logits=logits)
    picks = {}
    for decoding in ["hungarian", "greedy"]:
        selected_frames = decode_frame_selection(probabilities=probabilities,
                                                 unselected_class_index=unselected_class_index, frames_per_class=1,
                                                 decoding=decoding)
        picks[decoding] = [[int(selected_frames[wanted_index, 0]), int(class_index)]
                           for wanted_index, class_index in enumerate(wanted_class_indices)]

    # Values of the unselected column of the gains and selection distributions left empty (None)
    gain_rows = [[None if class_index == unselected_class_index else round(float(gains[frame_index, class_index]), 2)
                  for class_index in range(len(class_names))] for frame_index in range(len(frame_labels))]
    softmax_rows = [[None if class_index == unselected_class_index else
                     round(float(column_softmax[frame_index, class_index]), 3)
                     for class_index in range(len(class_names))] for frame_index in range(len(frame_labels))]
    return {"row_labels": [f"frame {frame_index}" for frame_index in range(len(frame_labels))],
            "column_labels": class_names,
            "values": {"probabilities": numpy.round(probabilities, 3).tolist(), "gains": gain_rows,
                       "selection": softmax_rows},
            "picks": picks,
            "true_cells": [[frame_index, label] for frame_index, label in enumerate(frame_labels)
                           if label != unselected_class_index]}


def decode_readme_example() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Decodes the example of the tasks reference with the Hungarian and greedy decodings of the repository and gives the
    probability of the labelling of the sequence each gives (the product, over the frames, of the probability of the
    class the decoding gives the frame), with the matrix data of its animation.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: The facts, named readme.*, and the matrix data of the example.
    """
    probabilities = numpy.array(README_PROBABILITIES)
    unselected_class_index = README_CLASS_NAMES.index("unselected")
    matrix_data = build_matrix_data(probabilities=probabilities, logits=numpy.log(probabilities),
                                    frame_labels=[unselected_class_index] * len(probabilities),
                                    class_names=README_CLASS_NAMES, unselected_class_index=unselected_class_index)
    matrix_data["true_cells"] = []
    matrix_data["row_labels"] = [f"frame {frame_index + 1}" for frame_index in range(len(probabilities))]
    facts = {}
    for decoding, picks in matrix_data["picks"].items():
        frame_classes = [unselected_class_index] * len(probabilities)
        for frame_index, class_index in picks:
            frame_classes[frame_index] = class_index
        labelling_probability = float(numpy.prod([probabilities[frame_index, class_index]
                                                  for frame_index, class_index in enumerate(frame_classes)]))
        facts[f"readme.{decoding}_labelling_probability"] = round_fact(value=labelling_probability)
        facts[f"readme.{decoding}_frame_classes"] = [README_CLASS_NAMES[class_index] for class_index in frame_classes]
        matrix_data[f"{decoding}_labelling_probability"] = round(labelling_probability, 4)
    facts["readme.probability_ratio"] = round_fact(value=facts["readme.hungarian_labelling_probability"] /
                                                   facts["readme.greedy_labelling_probability"])
    return facts, matrix_data


def count_decoding_differences(selection_rows: Dict[Tuple[str, str], Dict[str, str]]) -> Dict[str, int]:
    """
    Counts the rows of selections.csv where the Hungarian and greedy decodings chose other frames, and the rows each
    got wrong.

    Args:
        selection_rows (Dict[Tuple[str, str], Dict[str, str]]): The rows of selections.csv.

    Returns:
        Dict[str, int]: The 'differing', 'hungarian_wrong' and 'greedy_wrong' rows and the number of 'rows'.
    """
    return {"rows": len(selection_rows),
            "differing": sum(row["hungarian_frames"] != row["greedy_frames"] for row in selection_rows.values()),
            "hungarian_wrong": sum(row["hungarian_right_selections"] == "0" for row in selection_rows.values()),
            "greedy_wrong": sum(row["greedy_right_selections"] == "0" for row in selection_rows.values())}


def make_selection_figures(output_folder: Path, run_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Runs the checkpoint of SELECTION_ITERATION on the CPU on the sequences of its wrong picks, checks it against the
    evaluation, draws their selection images, writes the matrix data of the animations and gives the facts of the
    picks, of the decodings and of the model. Reused from the previous build unless forced.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the frame selection run.
        force (bool): Whether the figures are made again.

    Returns:
        Dict[str, Any]: The facts, named picks.*, matrix.*, readme.*, decodings.* and model.*.
    """
    selection_folder = output_folder / "selection"
    figure_path = selection_folder / "wrong_picks.png"
    data_paths = [selection_folder / "matrix_real.json", selection_folder / "matrix_readme.json"]
    reused_facts_path = selection_folder / "selection_facts.json"
    reused_facts = read_reused_facts(facts_path=reused_facts_path, required_paths=data_paths, force=force)
    if reused_facts is not None:
        return reused_facts

    # Checkpoint on the CPU and the sequences of its wrong picks, read and batched like the evaluation does
    model = load_sequence_model(run_folder=run_folder, iteration=SELECTION_ITERATION)
    dataset = build_test_dataset(run_folder=run_folder, label_type="selection")
    batch = collate_named_sequences(dataset=dataset, sequence_names=WRONG_PICK_SEQUENCES)
    outputs = predict_sequences(model=model, batch=batch)
    unselected_class_index = model.get_unselected_class_index()
    wanted_class_indices = get_wanted_class_indices(number_classes=outputs["class_probabilities"].shape[-1],
                                                    unselected_class_index=unselected_class_index)
    class_names = [dataset.class_names[class_index] for class_index in range(len(dataset.class_names))]
    thresholds = json.loads((run_folder / EVALUATION_FOLDER_NAME / f"Iteration_{SELECTION_ITERATION}" /
                             "class_thresholds.json").read_text(encoding="utf-8"))
    class_thresholds = [None if thresholds["classes"][class_names[class_index]] is None else
                        thresholds["classes"][class_names[class_index]]["threshold"]
                        for class_index in wanted_class_indices]
    selection_rows = read_selection_rows(selections_path=run_folder / EVALUATION_FOLDER_NAME /
                                         f"Iteration_{SELECTION_ITERATION}" / "selections.csv")

    # Every sequence decoded, checked against the evaluation and drawn by the selection image of the repository
    facts, selection_images, headings, matrix_data = {}, [], [], None
    for sample_index, sequence_name in enumerate(WRONG_PICK_SEQUENCES):
        frame_count = int(batch["sequence_mask"][sample_index].sum())
        probabilities = outputs["class_probabilities"][sample_index, :frame_count].double().numpy()
        logits = outputs["class_logits"][sample_index, :frame_count].double().numpy()
        frame_labels = batch["labels"][sample_index, :frame_count].numpy().astype(numpy.int64)
        selected_frames = decode_frame_selection(probabilities=probabilities,
                                                 unselected_class_index=unselected_class_index, frames_per_class=1,
                                                 decoding="hungarian")
        check_against_evaluation(probabilities=probabilities, sequence_name=sequence_name,
                                 class_names=dataset.class_names, wanted_class_indices=wanted_class_indices,
                                 selected_frames=selected_frames, selection_rows=selection_rows)
        selected_probabilities = gather_frame_probabilities(probabilities=probabilities, frames=selected_frames,
                                                            wanted_class_indices=wanted_class_indices)
        low_confidence = flag_low_confidence(selected_frames=selected_frames,
                                             selected_probabilities=selected_probabilities,
                                             class_thresholds=class_thresholds)
        selection_images.append(build_frame_selection_image(
            frames=convert_image_to_array(image=batch["images"][sample_index, :frame_count]),
            probabilities=probabilities, frame_labels=frame_labels, selected_frames=selected_frames,
            wanted_class_indices=wanted_class_indices, unselected_class_index=unselected_class_index,
            class_names=class_names))

        # Facts of the wrong pick of the sequence: the true and chosen frames of every class picked wrong
        for wanted_index, class_index in enumerate(wanted_class_indices):
            true_frame = int(numpy.flatnonzero(frame_labels == class_index)[0])
            chosen_frame = int(selected_frames[wanted_index, 0])
            if true_frame == chosen_frame:
                continue
            pick_name = f"picks.{sequence_name}"
            facts.update({f"{pick_name}.sequence_number": find_sequence_number(sequence_name=sequence_name),
                          f"{pick_name}.class": class_names[class_index], f"{pick_name}.true_frame": true_frame,
                          f"{pick_name}.true_probability": round_fact(value=probabilities[true_frame, class_index]),
                          f"{pick_name}.chosen_frame": chosen_frame,
                          f"{pick_name}.chosen_probability": round_fact(value=probabilities[chosen_frame, class_index]),
                          f"{pick_name}.low_confidence": bool(low_confidence[wanted_index, 0]),
                          f"{pick_name}.frame_count": frame_count})
            headings.append(f"{sequence_name} · {class_names[class_index]}: true frame {true_frame} at "
                            f"{probabilities[true_frame, class_index]:.3f}, chosen frame {chosen_frame} at "
                            f"{probabilities[chosen_frame, class_index]:.3f} · "
                            f"{'flagged low confidence' if low_confidence[wanted_index, 0] else 'not flagged'}")
        if sequence_name == MATRIX_SEQUENCE_NAME:
            matrix_data = build_matrix_data(probabilities=probabilities, logits=logits,
                                            frame_labels=frame_labels.tolist(), class_names=class_names,
                                            unselected_class_index=unselected_class_index)
            for frame_index, class_index in matrix_data["picks"]["hungarian"]:
                facts[f"matrix.{class_names[class_index]}_pick"] = frame_index
                facts[f"matrix.{class_names[class_index]}_pick_gain"] = matrix_data["values"]["gains"][frame_index][
                    class_index]
            facts["matrix.frame_count"] = frame_count
            facts["matrix.sequence_number"] = find_sequence_number(sequence_name=sequence_name)
            facts["matrix.true_frames"] = {class_names[label]: frame_index for frame_index, label in
                                           enumerate(frame_labels.tolist()) if label != unselected_class_index}
            for class_name, frame_index in facts["matrix.true_frames"].items():
                facts[f"matrix.true_{class_name}_frame"] = frame_index
            facts["matrix.decodings_agree"] = matrix_data["picks"]["hungarian"] == matrix_data["picks"]["greedy"]
    facts["picks.iteration"] = SELECTION_ITERATION
    box_list = stack_selection_images(selection_images=selection_images, headings=headings, output_path=figure_path)
    for sequence_name, box in zip(WRONG_PICK_SEQUENCES, box_list):
        facts[f"picks.{sequence_name}.box"] = box

    # Gains of the true and chosen triangle frames of the matrix sequence, and the selection distribution
    triangle_index = class_names.index("triangle")
    true_triangle = facts["matrix.true_frames"]["triangle"]
    facts["matrix.true_triangle_gain"] = matrix_data["values"]["gains"][true_triangle][triangle_index]
    facts["matrix.true_triangle_selection_share"] = matrix_data["values"]["selection"][true_triangle][triangle_index]
    facts["matrix.chosen_triangle_selection_share"] = matrix_data["values"]["selection"][
        facts["matrix.triangle_pick"]][triangle_index]
    facts["matrix.true_triangle_selection_loss"] = round_fact(
        value=-numpy.log(facts["matrix.true_triangle_selection_share"]))
    for frame_name, frame_index in [("true", true_triangle), ("chosen", facts["matrix.triangle_pick"])]:
        facts[f"matrix.{frame_name}_triangle_probability"] = matrix_data["values"]["probabilities"][frame_index][
            triangle_index]
        facts[f"matrix.{frame_name}_triangle_unselected_probability"] = matrix_data["values"]["probabilities"][
            frame_index][unselected_class_index]
    (selection_folder / "matrix_real.json").write_text(data=json.dumps(obj=matrix_data, indent=1), encoding="utf-8")
    readme_facts, readme_matrix = decode_readme_example()
    facts.update(readme_facts)
    (selection_folder / "matrix_readme.json").write_text(data=json.dumps(obj=readme_matrix, indent=1),
                                                          encoding="utf-8")

    # Agreement of the two decodings over every test sequence of the evaluations of the run
    for iteration in [SELECTION_ITERATION, 2000]:
        differences = count_decoding_differences(selection_rows=read_selection_rows(
            selections_path=run_folder / EVALUATION_FOLDER_NAME / f"Iteration_{iteration}" / "selections.csv"))
        for key, value in differences.items():
            facts[f"decodings.iteration_{iteration}_{key}"] = value

    # Parameters of the components and shapes of the stages of the model on the batch of the three sequences
    images = batch["images"].float()
    with torch.no_grad():
        backbone_maps, _ = model.backbone(input_tensor=images[batch["sequence_mask"]])
        frame_features = encode_frames(backbone=model.backbone, global_pooling=model.global_pooling,
                                       input_tensor=images, sequence_mask=batch["sequence_mask"])
        temporal_features = model.temporal_module(frame_features=frame_features, sequence_mask=batch["sequence_mask"])
    parameters = count_model_parameters(model=model)
    facts.update({"model.backbone_parameters": parameters["backbone"],
                  "model.temporal_parameters": parameters["temporal_module"],
                  "model.head_parameters": parameters["classification_head"],
                  "model.total_parameters": parameters["total"],
                  "model.total_megabytes": round_fact(value=parameters["total"] * 4 / 1e6),
                  "model.backbone_channels": int(backbone_maps.shape[1]),
                  "model.backbone_map_size": int(backbone_maps.shape[-1]),
                  "model.frame_feature_size": int(frame_features.shape[-1]),
                  "model.temporal_feature_size": int(temporal_features.shape[-1]),
                  "model.input_shape": str(list(batch["images"].shape)),
                  "model.backbone_shape": str(list(backbone_maps.shape)),
                  "model.frame_features_shape": str(list(frame_features.shape)),
                  "model.temporal_shape": str(list(temporal_features.shape)),
                  "model.logits_shape": str(list(outputs["class_logits"].shape)),
                  "model.wanted_class_mask": model.wanted_class_mask.tolist(),
                  "model.head": str(model.classification_head)})
    write_reused_facts(facts_path=reused_facts_path, figure_paths=[figure_path] + data_paths, facts=facts)
    return facts
