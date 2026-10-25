"""
Model figures of the sequence classification tutorials (chapters 2 and 3), made by
tutorials/figures/sequence_classification/figures.py from the checkpoints of the README runs, run on the CPU on real
test sequences of the moving shapes dataset: the cosine similarities between the frame vectors of one test sequence
for the average pooling run (64 features) and the flatten run (4,096 features), the probabilities of four test
sequences cut to different lengths alone and padded in one batch (the padding invariance), the accuracy of the
flatten run on the first frames of every test sequence, and the parameters and checkpoint sizes of the three runs.
The inference is reused from the previous build unless forced; its facts are the numbers the narration quotes.
"""

import json
import math

import numpy
import torch

from matplotlib import pyplot
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, POINTS_PER_PIXEL,
                                                SERIES_COLOURS, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR,
                                                compute_pixel_box, read_reused_facts, register_fonts, round_fact,
                                                save_figure, style_chart_panel, write_reused_facts)
from tutorials.figures.sequence_classification.sequence_figure_utilities import (compute_cosine_similarities,
                                                                                 load_sequence_model)
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.data.sequence.sequence_dataset import collate_padded_sequences, read_sequence_frames
from vision_training.models.model_registry import MODEL_DTYPE, TASK_MODEL_CLASSES
from vision_training.models.sequence.frame_encoder import encode_frames
from vision_training.models.task_model import TaskModel

# Test sequence whose frame vectors are compared, the first of the test manifest
SIMILARITY_SEQUENCE_INDEX = 0

# Lengths the first test sequences are cut to for the padding invariance, padded together to the longest
INVARIANCE_LENGTHS = [3, 5, 7, 8]

# Frames kept from the start of every test sequence for the accuracy on shorter sequences, and the size of the
# batches of the inference
TRUNCATION_LENGTHS = [2, 3, 4, 5, 6, 7, 8]
INFERENCE_BATCH_SIZE = 20


def read_test_split(test_folder: Path, label_dictionary: Dict[int, int], image_size: int,
                    sequence_count: int) -> Dict[str, Any]:
    """
    Reads the first test sequences of the moving shapes dataset the way the sequence dataset does (read_sequence_frames
    at the frame size of a run), with their class indices.

    Args:
        test_folder (Path): The test split, holding labels.json and data/.
        label_dictionary (Dict[int, int]): The class index of every category of the run.
        image_size (int): The frame size of the run.
        sequence_count (int): The number of sequences read from the start of the manifest.

    Returns:
        Dict[str, Any]: The 'names', the uint8 [S, T, 3, H, W] 'frames' (every sequence of the dataset has the same
            number of frames) and the 'labels'.

    Raises:
        ValueError: If the sequences read do not all have the same number of frames.
    """
    manifest = json.loads((test_folder / "labels.json").read_text(encoding="utf-8"))
    sequences = manifest["sequences"][:sequence_count]
    if len({len(sequence["frames"]) for sequence in sequences}) != 1:
        raise ValueError(f"The test sequences of {test_folder} have different numbers of frames: the figures stack "
                         f"them as one tensor.")
    frames = [read_sequence_frames(data_directory=str(test_folder / "data"), frame_files=sequence["frames"],
                                   number_frames=len(sequence["frames"]), image_size=image_size, keep_ratio=True)[0]
              for sequence in sequences]
    return {"names": [sequence["name"] for sequence in sequences], "frames": torch.stack(tensors=frames),
            "labels": torch.tensor(data=[label_dictionary[sequence["category_id"]] for sequence in sequences])}


def count_parameters(module: torch.nn.Module) -> int:
    """
    Counts the parameters of a module of a model.

    Args:
        module (torch.nn.Module): The module.

    Returns:
        int: Its number of parameters.
    """
    return sum(parameter.numel() for parameter in module.parameters())


def build_untrained_model(run_folder: Path) -> TaskModel:
    """
    Builds the model of a run from its saved configuration without reading its checkpoint, which is enough to count
    its parameters (the flatten run at 512 pixels has a checkpoint of about 600 MB).

    Args:
        run_folder (Path): The run folder.

    Returns:
        TaskModel: The model with random weights.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=run_folder / "training_configuration.yaml")
    return TASK_MODEL_CLASSES[experiment_configuration["task"]](configuration=model_configuration,
                                                                device=torch.device("cpu"), dtype=MODEL_DTYPE)


def compute_inference_facts(flatten_model: TaskModel, average_model: TaskModel, flatten_split: Dict[str, Any],
                            average_frames: torch.Tensor) -> Dict[str, Any]:
    """
    Runs the checkpoints on the test sequences: the frame vectors of one sequence for both poolings and their cosine
    similarities, the padding invariance of the flatten run, and its accuracy and mean probability of the true class
    on the first frames of every test sequence.

    Args:
        flatten_model (TaskModel): The flatten run at 128 pixels.
        average_model (TaskModel): The average pooling run at 512 pixels.
        flatten_split (Dict[str, Any]): The test sequences read at 128 pixels (read_test_split).
        average_frames (torch.Tensor): The uint8 [T, 3, 512, 512] frames of the compared sequence at 512 pixels.

    Returns:
        Dict[str, Any]: The facts, named similarity.*, invariance.* and truncation.*.
    """
    facts = {}
    with torch.no_grad():
        # Frame vectors of the compared sequence through both encoders, every frame real
        for model_name, model, frames in [("flatten", flatten_model,
                                           flatten_split["frames"][SIMILARITY_SEQUENCE_INDEX]),
                                          ("average", average_model, average_frames)]:
            vectors = encode_frames(backbone=model.backbone, global_pooling=model.global_pooling,
                                    input_tensor=frames[None].float(), sequence_mask=None)[0]
            similarities = compute_cosine_similarities(vectors=vectors)
            off_diagonal = [similarities[row][column] for row in range(len(similarities))
                            for column in range(len(similarities)) if row != column]
            facts.update({f"similarity.{model_name}.values": similarities,
                          f"similarity.{model_name}.features": int(vectors.shape[-1]),
                          f"similarity.{model_name}.minimum": min(off_diagonal),
                          f"similarity.{model_name}.mean": round_fact(value=float(numpy.mean(off_diagonal)))})

        # Four test sequences cut to different lengths, alone without mask, then padded together with their mask
        frames, labels = flatten_split["frames"], flatten_split["labels"]
        alone = torch.cat(tensors=[flatten_model(input_tensor=frames[index:index + 1, :length].float())[
            "class_probabilities"] for index, length in enumerate(INVARIANCE_LENGTHS)])
        samples = [{"images": frames[index, :length], "sequence_mask": torch.ones(size=(length,), dtype=torch.bool),
                    "labels": labels[index], "sample_identifiers": index}
                   for index, length in enumerate(INVARIANCE_LENGTHS)]
        batch = collate_padded_sequences(samples=samples, members_per_sample=1)
        padded = flatten_model(input_tensor=batch["images"].float(),
                               sequence_mask=batch["sequence_mask"])["class_probabilities"]
        maximum_difference = float((alone - padded).abs().max())
        facts.update({"invariance.lengths": INVARIANCE_LENGTHS, "invariance.maximum_difference": maximum_difference,
                      "invariance.inverse_bound": 10 ** -math.ceil(math.log10(max(maximum_difference, 1e-15))),
                      "invariance.padded_probabilities": [[round(value, 4) for value in row]
                                                          for row in padded.tolist()],
                      "invariance.sequences": flatten_split["names"][:len(INVARIANCE_LENGTHS)]})

        # Accuracy and probability of the true class on the first frames of every test sequence
        for length in TRUNCATION_LENGTHS:
            probabilities = torch.cat(tensors=[flatten_model(input_tensor=frames[start:start + INFERENCE_BATCH_SIZE,
                                                                                 :length].float())[
                "class_probabilities"] for start in range(0, len(labels), INFERENCE_BATCH_SIZE)])
            true_probabilities = probabilities[torch.arange(len(labels)), labels]
            facts[f"truncation.frames_{length}.accuracy"] = round_fact(
                value=float((probabilities.argmax(dim=1) == labels).float().mean()))
            facts[f"truncation.frames_{length}.mean_true_probability"] = round_fact(
                value=float(true_probabilities.mean()))
        facts["truncation.sequences"] = len(labels)
    return facts


def draw_truncation_figure(output_path: Path, facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Draws the accuracy and the mean probability of the true class of the flatten run against the number of first
    frames kept from every test sequence, both on the same 0 to 1 scale, the 8 frames of the training sequences
    marked; gives the box of the panel.

    Args:
        output_path (Path): The PNG file.
        facts (Dict[str, Any]): The truncation facts of compute_inference_facts.

    Returns:
        Dict[str, Any]: The box, named truncation.panel_box.
    """
    register_fonts()
    figure, axes = pyplot.subplots(figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH),
                                   dpi=DOTS_PER_INCH)
    figure.patch.set_facecolor(SURFACE_COLOUR)
    figure.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.13)
    series = [("accuracy", "Accuracy", SERIES_COLOURS[0]),
              ("mean_true_probability", "Mean probability of the true class", SERIES_COLOURS[1])]
    for fact_name, label, colour in series:
        values = [facts[f"truncation.frames_{length}.{fact_name}"] for length in TRUNCATION_LENGTHS]
        axes.plot(TRUNCATION_LENGTHS, values, color=colour, linewidth=3 * POINTS_PER_PIXEL, marker="o",
                  markersize=10 * POINTS_PER_PIXEL, markeredgecolor=SURFACE_COLOUR,
                  markeredgewidth=2 * POINTS_PER_PIXEL)
        for length, value in zip(TRUNCATION_LENGTHS, values):
            if fact_name == "accuracy":
                axes.annotate(text=f"{value:.2f}", xy=(length, value), xytext=(0, 12), textcoords="offset points",
                              ha="center", va="bottom", color=TEXT_PRIMARY_COLOUR, fontsize=17 * POINTS_PER_PIXEL)
    style_chart_panel(axes=axes, title=f"Flatten run, iteration 1,000, on the first frames of the "
                                       f"{facts['truncation.sequences']} test sequences",
                      x_label="Frames kept from the start of every sequence (training sequences: 8 frames)",
                      y_label="Share")
    axes.set_xticks(ticks=TRUNCATION_LENGTHS)
    axes.set_ylim(bottom=0.0, top=1.1)
    legend_handles = [Line2D(xdata=[0], ydata=[0], color=colour, linewidth=3 * POINTS_PER_PIXEL, marker="o",
                             markersize=8 * POINTS_PER_PIXEL) for _, _, colour in series]
    figure.legend(handles=legend_handles, labels=[label for _, label, _ in series], loc="center right",
                  bbox_to_anchor=(0.97, 0.93), ncols=2, frameon=False, labelcolor=TEXT_PRIMARY_COLOUR,
                  fontsize=18 * POINTS_PER_PIXEL)
    save_figure(figure=figure, output_path=output_path)
    box = compute_pixel_box(figure=figure, axes=axes, with_labels=True, margin=6)
    pyplot.close(fig=figure)
    return {"truncation.panel_box": box}


def make_model_figures(output_folder: Path, flatten_model: TaskModel, run_folders: Dict[str, Path],
                       checkpoint_iterations: Dict[str, int], test_folder: Path, label_dictionary: Dict[int, int],
                       force: bool) -> Dict[str, Any]:
    """
    Makes the model figures and gives their facts: the inference facts (reused from the previous build unless forced),
    the parameters of the backbone, temporal module and head of the three runs, and the size of their last checkpoint.

    Args:
        output_folder (Path): The figures folder of the series.
        flatten_model (TaskModel): The flatten run at 128 pixels, loaded on the CPU.
        run_folders (Dict[str, Path]): The folders of the runs 'flatten_128', 'average_512' and 'flatten_512'.
        checkpoint_iterations (Dict[str, int]): The last checkpoint of every run.
        test_folder (Path): The test split of the runs.
        label_dictionary (Dict[int, int]): The class index of every category.
        force (bool): Whether the inference runs again even when the previous build left its facts.

    Returns:
        Dict[str, Any]: The facts, named similarity.*, invariance.*, truncation.*, parameters.* and checkpoints.*.
    """
    model_folder = output_folder / "model"
    model_folder.mkdir(parents=True, exist_ok=True)
    facts_path = model_folder / "inference_facts.json"
    truncation_path = model_folder / "truncation.png"
    facts = read_reused_facts(facts_path=facts_path, required_paths=[truncation_path], force=force)
    if facts is None:
        flatten_split = read_test_split(test_folder=test_folder, label_dictionary=label_dictionary, image_size=128,
                                        sequence_count=10 ** 6)
        average_model = load_sequence_model(run_folder=run_folders["average_512"],
                                            iteration=checkpoint_iterations["average_512"])
        average_split = read_test_split(test_folder=test_folder, label_dictionary=label_dictionary, image_size=512,
                                        sequence_count=SIMILARITY_SEQUENCE_INDEX + 1)
        facts = compute_inference_facts(flatten_model=flatten_model, average_model=average_model,
                                        flatten_split=flatten_split,
                                        average_frames=average_split["frames"][SIMILARITY_SEQUENCE_INDEX])
        facts["similarity.sequence"] = flatten_split["names"][SIMILARITY_SEQUENCE_INDEX]
        facts.update(draw_truncation_figure(output_path=truncation_path, facts=facts))
        write_reused_facts(facts_path=facts_path, figure_paths=[truncation_path], facts=facts)

    # Parameters of every component of the three runs, and the size of their last checkpoint
    for run_name, run_folder in run_folders.items():
        model = flatten_model if run_name == "flatten_128" else build_untrained_model(run_folder=run_folder)
        for component_name in ["backbone", "temporal_module", "classification_head"]:
            parameter_count = count_parameters(module=getattr(model, component_name))
            facts[f"parameters.{run_name}.{component_name}"] = parameter_count
            facts[f"parameters.{run_name}.{component_name}_millions"] = round(parameter_count / 1e6, 2)
            facts[f"parameters.{run_name}.{component_name}_log10"] = round_fact(value=math.log10(parameter_count))
        facts[f"parameters.{run_name}.total"] = count_parameters(module=model)
        facts[f"parameters.{run_name}.total_millions"] = round(count_parameters(module=model) / 1e6, 2)
        facts[f"parameters.{run_name}.frame_features"] = int(model.temporal_module.recurrent_network.input_size)
        iteration = checkpoint_iterations[run_name]
        checkpoint_path = run_folder / "Weights" / f"Iteration_{iteration}" / f"model_{iteration:06d}.pt"
        facts[f"checkpoints.{run_name}.bytes"] = checkpoint_path.stat().st_size
        facts[f"checkpoints.{run_name}.megabytes"] = round_fact(value=checkpoint_path.stat().st_size / 1e6)
        facts[f"checkpoints.{run_name}.iteration"] = iteration
    return facts
