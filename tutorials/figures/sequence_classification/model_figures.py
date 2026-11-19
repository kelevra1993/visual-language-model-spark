"""
Model facts of the sequence classification tutorials (chapters 2 and 3), made by
tutorials/figures/sequence_classification/figures.py from the checkpoints of the tutorial runs, run on the CPU on real
test sequences of the moving shapes dataset: the cosine similarities between the frame vectors of one test sequence
for the average pooling run (64 features) and the flatten run (16,384 features at 256 pixels), the probabilities of
four test sequences cut to different lengths alone and padded in one batch (the padding invariance), and the
parameters and checkpoint sizes of the three runs. The inference is reused from the previous build unless forced; its
facts are the numbers the narration quotes, and the properties the narration states of them (average pooling giving
the same vector to every frame, flatten vectors drifting apart down to the first and last frames) are checked. A build
from the committed snapshots (tutorials/run_snapshots, without checkpoints) reuses the committed outputs of the
inference (figure_reuse/sequence_classification/model), counts the parameters on models built from the configurations
and reads the checkpoint sizes from the listings of the Weights folders.
"""

import json
import math

import numpy
import torch

from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (read_reused_facts, read_run_configuration, restore_committed_figures,
                                                round_fact, write_reused_facts)
from tutorials.figures.sequence_classification.sequence_figure_utilities import (build_untrained_model,
                                                                                 compute_cosine_similarities,
                                                                                 load_sequence_model,
                                                                                 read_checkpoint_bytes)
from vision_training.data.sequence.sequence_dataset import collate_padded_sequences, read_sequence_frames
from vision_training.deployment.export_specifications import build_inference_model, get_export_specification
from vision_training.models.sequence.frame_encoder import encode_frames
from vision_training.models.task_model import TaskModel
from vision_training.trainers.checkpointing import get_checkpoint_file

# Names of the main run (flatten pooling at 256 pixels) and of the average pooling run in the facts, checked against
# their configurations by figures.py
MAIN_RUN_NAME = "flatten_256"
AVERAGE_RUN_NAME = "average_512"

# Frames of the test sequence whose frame vectors are compared, the first test sequence of that length: the grids of
# the pooling scene of 02_frame_encoder_temporal_module_head.yaml are laid out for 8 x 8 cells
SIMILARITY_FRAMES = 8

# Smallest cosine similarity of the average pooling vectors of the compared sequence, between any two of its frames,
# for the narration saying that every pair rounds to 1.00
AVERAGE_SIMILARITY_FLOOR = 0.995

# Lengths the invariance test cuts four test sequences to, padded together to the longest, the sequences being the
# first four test sequences holding at least the longest of these lengths
INVARIANCE_LENGTHS = [3, 5, 7, 8]

# Version of the inference facts, written into the reused facts so that the outputs of an older version of this
# module are made again instead of being reused
MODEL_FACTS_VERSION = 2


def select_test_sequences(test_folder: Path) -> Dict[str, Any]:
    """
    Chooses the test sequences of the model facts by their number of frames, so that the choice follows the test
    split rather than typed names: the first test sequence of SIMILARITY_FRAMES frames for the similarity grids, and
    the first four holding at least the longest of INVARIANCE_LENGTHS for the invariance test.

    Args:
        test_folder (Path): The test split, holding labels.json and data/.

    Returns:
        Dict[str, Any]: The manifest entry of the 'similarity' sequence and the list of 'invariance' entries.

    Raises:
        ValueError: If the test split holds no such sequences.
    """
    sequences = json.loads((test_folder / "labels.json").read_text(encoding="utf-8"))["sequences"]
    similarity_sequence = next((sequence for sequence in sequences if len(sequence["frames"]) == SIMILARITY_FRAMES),
                               None)
    invariance_sequences = [sequence for sequence in sequences
                            if len(sequence["frames"]) >= max(INVARIANCE_LENGTHS)][:len(INVARIANCE_LENGTHS)]
    if similarity_sequence is None or len(invariance_sequences) < len(INVARIANCE_LENGTHS):
        raise ValueError(f"The test split {test_folder} needs a sequence of {SIMILARITY_FRAMES} frames and "
                         f"{len(INVARIANCE_LENGTHS)} sequences of at least {max(INVARIANCE_LENGTHS)} frames.")
    return {"similarity": similarity_sequence, "invariance": invariance_sequences}


def read_test_sequence(test_folder: Path, sequence: Dict[str, Any], image_size: int) -> torch.Tensor:
    """
    Reads the frames of one test sequence of the moving shapes dataset the way the sequence dataset does
    (read_sequence_frames at the frame size of a run, keeping the aspect ratio).

    Args:
        test_folder (Path): The test split, holding labels.json and data/.
        sequence (Dict[str, Any]): The manifest entry of the sequence.
        image_size (int): The frame size of the run.

    Returns:
        torch.Tensor: The uint8 [T, 3, H, W] frames.
    """
    return read_sequence_frames(data_directory=str(test_folder / "data"), frame_files=sequence["frames"],
                                number_frames=len(sequence["frames"]), image_size=image_size, keep_ratio=True)[0]


def count_parameters(module: torch.nn.Module) -> int:
    """
    Counts the parameters of a module of a model.

    Args:
        module (torch.nn.Module): The module.

    Returns:
        int: Its number of parameters.
    """
    return sum(parameter.numel() for parameter in module.parameters())


def compute_similarity_facts(model_name: str, model: TaskModel, frames: torch.Tensor) -> Dict[str, Any]:
    """
    Runs the frames of one sequence through the frame encoder of a run, every frame real, and gives the cosine
    similarity of the frame vectors of every pair of frames.

    Args:
        model_name (str): "flatten" or "average", the name of the facts.
        model (TaskModel): The trained run, on the CPU.
        frames (torch.Tensor): The uint8 [T, 3, H, W] frames of the sequence at the frame size of the run.

    Returns:
        Dict[str, Any]: The facts, named similarity.<model name>.*.
    """
    vectors = encode_frames(backbone=model.backbone, global_pooling=model.global_pooling,
                            input_tensor=frames[None].float(), sequence_mask=None)[0]
    similarities = compute_cosine_similarities(vectors=vectors)
    off_diagonal = [similarities[row][column] for row in range(len(similarities))
                    for column in range(len(similarities)) if row != column]
    return {f"similarity.{model_name}.values": similarities,
            f"similarity.{model_name}.features": int(vectors.shape[-1]),
            f"similarity.{model_name}.minimum": min(off_diagonal),
            f"similarity.{model_name}.first_last": similarities[0][-1],
            f"similarity.{model_name}.mean": round_fact(value=float(numpy.mean(off_diagonal)))}


def check_similarity_facts(facts: Dict[str, Any]) -> None:
    """
    Checks what the pooling scene says of the similarities: every pair of average pooling vectors rounds to 1.00 (at
    least AVERAGE_SIMILARITY_FLOOR), and the flatten vectors are furthest apart between the first and the last frame.

    Args:
        facts (Dict[str, Any]): The similarity facts of both runs.

    Raises:
        ValueError: If a property does not hold.
    """
    if facts["similarity.average.minimum"] < AVERAGE_SIMILARITY_FLOOR:
        raise ValueError(f"The average pooling vectors of the compared sequence reach a similarity of "
                         f"{facts['similarity.average.minimum']}, below {AVERAGE_SIMILARITY_FLOOR}: the pooling scene "
                         f"says that every pair rounds to 1.00.")
    if facts["similarity.flatten.first_last"] != facts["similarity.flatten.minimum"]:
        raise ValueError(f"The flatten vectors of the compared sequence are furthest apart at "
                         f"{facts['similarity.flatten.minimum']}, not between the first and the last frame "
                         f"({facts['similarity.flatten.first_last']}): the pooling scene says the opposite.")


def compute_inference_facts(flatten_model: TaskModel, average_model: TaskModel, test_folder: Path,
                            image_sizes: Dict[str, int], label_dictionary: Dict[int, int]) -> Dict[str, Any]:
    """
    Runs the checkpoints on test sequences: the frame vectors of one sequence for both poolings and their cosine
    similarities, and the padding invariance of the flatten run.

    Args:
        flatten_model (TaskModel): The flatten run at 256 pixels.
        average_model (TaskModel): The average pooling run at 512 pixels.
        test_folder (Path): The test split of the runs.
        image_sizes (Dict[str, int]): The frame size of the 'flatten' and 'average' runs.
        label_dictionary (Dict[int, int]): The class index of every category.

    Returns:
        Dict[str, Any]: The facts, named similarity.* and invariance.*.
    """
    sequences = select_test_sequences(test_folder=test_folder)
    facts = {"model.version": MODEL_FACTS_VERSION, "similarity.sequence": sequences["similarity"]["name"]}
    with torch.no_grad():
        # Frame vectors of the compared sequence through both encoders, every frame real, each run at its frame size
        for model_name, model in [("flatten", flatten_model), ("average", average_model)]:
            frames = read_test_sequence(test_folder=test_folder, sequence=sequences["similarity"],
                                        image_size=image_sizes[model_name])
            facts.update(compute_similarity_facts(model_name=model_name, model=model, frames=frames))
        check_similarity_facts(facts=facts)

        # Four test sequences cut to different lengths, alone without mask, then padded together with their mask
        cut_frames = [read_test_sequence(test_folder=test_folder, sequence=sequence,
                                         image_size=image_sizes["flatten"])[:length]
                      for sequence, length in zip(sequences["invariance"], INVARIANCE_LENGTHS)]
        labels = [label_dictionary[sequence["category_id"]] for sequence in sequences["invariance"]]
        alone = torch.cat(tensors=[flatten_model(input_tensor=frames[None].float())["class_probabilities"]
                                   for frames in cut_frames])
        samples = [{"images": frames, "sequence_mask": torch.ones(size=(len(frames),), dtype=torch.bool),
                    "labels": torch.tensor(data=label), "sample_identifiers": index}
                   for index, (frames, label) in enumerate(zip(cut_frames, labels))]
        batch = collate_padded_sequences(samples=samples, members_per_sample=1)
        padded = flatten_model(input_tensor=batch["images"].float(),
                               sequence_mask=batch["sequence_mask"])["class_probabilities"]
        maximum_difference = float((alone - padded).abs().max())
        facts.update({"invariance.lengths": INVARIANCE_LENGTHS, "invariance.maximum_difference": maximum_difference,
                      "invariance.inverse_bound": 10 ** -math.ceil(math.log10(max(maximum_difference, 1e-15))),
                      "invariance.padded_probabilities": [[round(value, 4) for value in row]
                                                          for row in padded.tolist()],
                      "invariance.sequences": [sequence["name"] for sequence in sequences["invariance"]]})
    return facts


def describe_export_input(model: TaskModel, task: str) -> Dict[str, Any]:
    """
    Gives the input and output of the export of the main run, read from get_export_specification of the deployment
    code on its inference model, for the export scene: the input shape with its batch and frame dimensions, the
    shapes of the batches the export is traced and checked on, the most frames of a request and the classes out.

    Args:
        model (TaskModel): The trained flatten run at 256 pixels, on the CPU.
        task (str): The task of the run.

    Returns:
        Dict[str, Any]: The facts, named export.*.
    """
    specification = get_export_specification(task=task, inference_model=build_inference_model(task=task, model=model))
    input_specification = specification["inputs"][0]

    # The dimensions of an input start with its number of frames (-1, dynamic), written T after the batch B
    frame_dimensions = [str(size) for size in input_specification["dimensions"][1:]]
    return {"export.input_shape": "[" + ", ".join(["B", "T"] + frame_dimensions) + "]",
            "export.example_shape": str(input_specification["example_shape"]),
            "export.check_shape": str(input_specification["check_shape"]),
            "export.maximum_frames": input_specification["maximum_frames"],
            "export.traced_frames": input_specification["example_shape"][1],
            "export.checked_frames": input_specification["check_shape"][1],
            "export.output_classes": specification["outputs"][0]["dimensions"][-1]}


def make_model_figures(output_folder: Path, flatten_model: TaskModel, run_folders: Dict[str, Path],
                       checkpoint_iterations: Dict[str, int], test_folder: Path, label_dictionary: Dict[int, int],
                       force: bool) -> Dict[str, Any]:
    """
    Makes the model facts: the inference facts (reused from the previous build unless forced, and restored from the
    committed snapshots when the checkpoints of the inference are missing), the parameters of the backbone, temporal
    module and head of the three runs, and the size of their last checkpoint.

    Args:
        output_folder (Path): The figures folder of the series.
        flatten_model (TaskModel): The flatten run at 256 pixels, loaded on the CPU.
        run_folders (Dict[str, Path]): The folders of the runs MAIN_RUN_NAME, AVERAGE_RUN_NAME and 'flatten_512'.
        checkpoint_iterations (Dict[str, int]): The last checkpoint of every run.
        test_folder (Path): The test split of the runs.
        label_dictionary (Dict[int, int]): The class index of every category.
        force (bool): Whether the inference runs again even when the previous build left its facts.

    Returns:
        Dict[str, Any]: The facts, named similarity.*, invariance.*, parameters.* and checkpoints.*.

    Raises:
        ValueError: If the reused inference facts were written by another version of this module.
    """
    model_folder = output_folder / "model"
    model_folder.mkdir(parents=True, exist_ok=True)
    facts_path = model_folder / "inference_facts.json"

    # The inference needs the checkpoints of the flatten and average runs: without them (a build from the committed
    # snapshots) the committed outputs of the family are restored and reused, --force having nothing to compute from
    checkpoints_present = all(get_checkpoint_file(weights_directory=run_folders[run_name] / "Weights",
                                                  iteration=checkpoint_iterations[run_name]).exists()
                              for run_name in [MAIN_RUN_NAME, AVERAGE_RUN_NAME])
    if not checkpoints_present:
        restore_committed_figures(output_folder=output_folder, family_folder="model")
    facts = read_reused_facts(facts_path=facts_path, required_paths=[], force=force and checkpoints_present)
    if facts is not None and facts.get("model.version") != MODEL_FACTS_VERSION:
        if not checkpoints_present:
            raise ValueError(f"The committed model facts {facts_path} are of version {facts.get('model.version')}, "
                             f"not {MODEL_FACTS_VERSION}: build with the real runs, then write the snapshots again.")
        facts = None
    if facts is None:
        average_model = load_sequence_model(run_folder=run_folders[AVERAGE_RUN_NAME],
                                            iteration=checkpoint_iterations[AVERAGE_RUN_NAME], allow_untrained=False)
        image_sizes = {model_name: read_run_configuration(run_folder=run_folders[run_name])["Data"]["image_settings"][
            "size"] for model_name, run_name in [("flatten", MAIN_RUN_NAME), ("average", AVERAGE_RUN_NAME)]}
        facts = compute_inference_facts(flatten_model=flatten_model, average_model=average_model,
                                        test_folder=test_folder, image_sizes=image_sizes,
                                        label_dictionary=label_dictionary)
        write_reused_facts(facts_path=facts_path, figure_paths=[], facts=facts)

    # Frames of the compared sequence and the lengths of the invariance test, with the labels of their grids, kept out
    # of the reused facts so that they follow the facts and constants of this module
    frame_count = len(facts["similarity.flatten.values"])
    facts.update({"similarity.frames": frame_count,
                  "similarity.frame_labels": [str(frame_index) for frame_index in range(frame_count)],
                  "invariance.sequence_count": len(facts["invariance.lengths"]),
                  "invariance.row_labels": [f"{length} frames" for length in facts["invariance.lengths"]]})

    # Parameters of every component of the three runs, and the size of their last checkpoint (from the listing of its
    # Weights folder for a run read from the committed snapshots)
    for run_name, run_folder in run_folders.items():
        model = flatten_model if run_name == MAIN_RUN_NAME else build_untrained_model(run_folder=run_folder)
        for component_name in ["backbone", "temporal_module", "classification_head"]:
            parameter_count = count_parameters(module=getattr(model, component_name))
            facts[f"parameters.{run_name}.{component_name}"] = parameter_count
            facts[f"parameters.{run_name}.{component_name}_millions"] = round(parameter_count / 1e6, 2)
            facts[f"parameters.{run_name}.{component_name}_log10"] = round_fact(value=math.log10(parameter_count))
        facts[f"parameters.{run_name}.total"] = count_parameters(module=model)
        facts[f"parameters.{run_name}.total_millions"] = round(count_parameters(module=model) / 1e6, 2)
        facts[f"parameters.{run_name}.frame_features"] = int(model.temporal_module.recurrent_network.input_size)
        iteration = checkpoint_iterations[run_name]
        checkpoint_bytes = read_checkpoint_bytes(run_folder=run_folder, iteration=iteration)
        facts[f"checkpoints.{run_name}.bytes"] = checkpoint_bytes
        facts[f"checkpoints.{run_name}.megabytes"] = round_fact(value=checkpoint_bytes / 1e6)
        facts[f"checkpoints.{run_name}.iteration"] = iteration
    return facts
