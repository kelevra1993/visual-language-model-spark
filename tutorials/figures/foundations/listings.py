"""
Text listings of the foundations tutorials, written by the figure series of tutorials/figures/foundations at every build
so that the videos show the repository as it is: the folder tree of the real classification run (its configuration
copy, records, checkpoints, logs, report and evaluation folders, one checkpoint and one evaluation expanded), and the
summary panels the trainer prints at launch for that run (backbone, receptive field, model size), captured from the
code of the repository on the CPU from the saved configuration of the run, without any training. The storyboards show
them as text excerpts (source file figures:foundations/listings/...). It also gives the facts of the chapters on the
data pipeline and on the backbone read from that model and configuration (the augmentation types activated and the
share of the training images they change, the receptive field, parameters and residual pairs of every block), and
the counts of the repository quoted by the closing chapter (tasks, test files), so that their narration and
drawings follow the run and the repository when they change.
"""

import contextlib
import io
import re

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.core.task_registry import TASK_TRAINER_CLASSES
from vision_training.data.augmentations import (AUGMENTATION_TYPES, GEOMETRIC_AUGMENTATION_TYPES,
                                                PHOTOMETRIC_AUGMENTATION_TYPES)
from vision_training.models.classification.image_classifier import ImageClassifier

# Colour codes of the terminal removed from the captured text
TERMINAL_CODE_PATTERN = re.compile(pattern=r"\x1b\[[0-9;]*[A-Za-z]")

def read_iteration_number(folder_name: str) -> int:
    """
    Gives the iteration of a checkpoint or evaluation folder of a run (Iteration_1500 gives 1500), to sort them.

    Args:
        folder_name (str): The name of the folder.

    Returns:
        int: Its iteration, -1 for a folder of another name.
    """
    match = re.fullmatch(pattern=r"Iteration_(\d+)", string=folder_name)
    return int(match.group(1)) if match else -1


def order_folder_entry(entry: Path) -> Tuple[bool, int, str]:
    """
    Gives the sort key of an entry of a run folder in the tree: the folders first, then the files, the iteration
    folders by their iteration and the others by name.

    Args:
        entry (Path): The file or folder.

    Returns:
        Tuple[bool, int, str]: Whether it is a file, its iteration and its name.
    """
    return entry.is_file(), read_iteration_number(folder_name=entry.name), entry.name


def list_run_folder_tree(run_folder: Path) -> List[str]:
    """
    Lists the folder of a run as a tree for the run folder scene: its folders then its files; in Weights and
    Evaluation the first and last iteration folders with "..." between them, the last checkpoint expanded (the last
    evaluation is listed apart by list_evaluation_folder); in Tensorboard the log folder of every split.

    Args:
        run_folder (Path): The folder of the run.

    Returns:
        List[str]: The lines of the tree.
    """
    lines = [f"{run_folder.name}/"]
    top_entries = sorted(run_folder.iterdir(), key=order_folder_entry)
    for top_index, top_entry in enumerate(top_entries):
        top_is_last = top_index == len(top_entries) - 1
        lines.append(f"{'└── ' if top_is_last else '├── '}{top_entry.name}{'/' if top_entry.is_dir() else ''}")
        if not top_entry.is_dir():
            continue
        child_prefix = "    " if top_is_last else "│   "
        children = sorted(top_entry.iterdir(), key=order_folder_entry)

        # Iteration folders: the first and the last, the last expanded; other folders: every entry
        iteration_children = [child for child in children if read_iteration_number(folder_name=child.name) >= 0]
        if len(iteration_children) > 2:
            other_children = [child for child in children if child not in iteration_children]
            shown_children = [iteration_children[0], None, iteration_children[-1]] + other_children
        else:
            shown_children = children
        for child_index, child in enumerate(shown_children):
            child_is_last = child_index == len(shown_children) - 1
            branch = "└── " if child_is_last else "├── "
            if child is None:
                lines.append(f"{child_prefix}├── ...")
                continue
            lines.append(f"{child_prefix}{branch}{child.name}{'/' if child.is_dir() else ''}")
            if child.is_dir() and iteration_children and child == iteration_children[-1] and \
                    top_entry.name != "Evaluation":
                grandchildren = sorted(child.iterdir(), key=order_folder_entry)
                for grandchild_index, grandchild in enumerate(grandchildren):
                    grandchild_branch = "└── " if grandchild_index == len(grandchildren) - 1 else "├── "
                    lines.append(f"{child_prefix}{'    ' if child_is_last else '│   '}{grandchild_branch}"
                                 f"{grandchild.name}{'/' if grandchild.is_dir() else ''}")
    return lines


def list_evaluation_folder(evaluation_folder: Path) -> List[str]:
    """
    Lists the files of one evaluation folder of a run, written for every evaluated checkpoint, as a tree under its
    name.

    Args:
        evaluation_folder (Path): The evaluation folder of one checkpoint.

    Returns:
        List[str]: The lines of the tree.
    """
    entries = sorted(evaluation_folder.iterdir(), key=order_folder_entry)
    return [f"Evaluation/{evaluation_folder.name}/"] + [
        f"{'└── ' if entry_index == len(entries) - 1 else '├── '}{entry.name}{'/' if entry.is_dir() else ''}"
        for entry_index, entry in enumerate(entries)]


def capture_printed_text(printing_function: Any) -> str:
    """
    Runs a printing function of the repository (a summary panel) and gives what it printed, without the colour codes.

    Args:
        printing_function (Any): The function, called without arguments.

    Returns:
        str: The printed text.
    """
    captured_output = io.StringIO()
    with contextlib.redirect_stdout(new_target=captured_output):
        printing_function()
    return TERMINAL_CODE_PATTERN.sub(repl="", string=captured_output.getvalue())


def join_numbers(values: List[int]) -> str:
    """
    Writes a list of numbers as the narration says it ("3 and 5", "2, 4 and 6"), for the facts quoted in a sentence.

    Args:
        values (List[int]): The numbers.

    Returns:
        str: The numbers joined with commas and a last "and", or "none" for an empty list.
    """
    if not values:
        return "none"
    if len(values) == 1:
        return str(values[0])
    return f"{', '.join(str(value) for value in values[:-1])} and {values[-1]}"


def describe_repository(repository_folder: Path) -> Dict[str, Any]:
    """
    Gives the facts of the closing chapter of the foundations series read from the repository itself, so that its
    narration follows the repository: the number of tasks of the task registry and the number of test files of the
    tests folder.

    Args:
        repository_folder (Path): The root folder of the repository.

    Returns:
        Dict[str, Any]: The facts, under the repository prefix.

    Raises:
        FileNotFoundError: If the repository has no tests folder.
    """
    tests_folder = repository_folder / "tests"
    if not tests_folder.is_dir():
        raise FileNotFoundError(f"No tests folder {tests_folder} in the repository.")
    return {"repository.task_count": len(TASK_TRAINER_CLASSES),
            "repository.test_file_count": len(list(tests_folder.rglob("test_*.py")))}


def describe_data_pipeline(data_configuration: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gives the facts of the data pipeline chapter of the foundations series from the Data section of the classification
    run: its image settings, the probability of augmenting a training image, the activated augmentation types in their
    order of application with their probabilities, and the expected share of the training images changed by at least
    one type (DataAugmenter.augment_batch draws the sample, then DataAugmenter.should_apply every activated type).

    Args:
        data_configuration (Dict[str, Any]): The Data section of the run configuration.

    Returns:
        Dict[str, Any]: The facts, under the data prefix.
    """
    augmentation = data_configuration["data_augmentation"]
    probability = float(augmentation["probability"]) if augmentation["activated"] else 0.0

    # Activated types in their order of application, the geometric types first
    activated_types = []
    for type_name in AUGMENTATION_TYPES:
        type_settings = augmentation["types"].get(type_name, {"activated": False})
        if type_settings["activated"]:
            activated_types.append({"name": type_name, "probability": float(type_settings["probability"]),
                                    "geometric": type_name in GEOMETRIC_AUGMENTATION_TYPES})

    # Share of the training images changed: augmented at all, then drawing at least one activated type
    unchanged_by_types = 1.0
    for activated_type in activated_types:
        unchanged_by_types *= 1.0 - activated_type["probability"]
    return {"data.image_size": int(data_configuration["image_settings"]["size"]),
            "data.keep_ratio": bool(data_configuration["image_settings"]["keep_ratio"]),
            "data.augmentation_probability": probability, "data.unaugmented_probability": 1.0 - probability,
            "data.activated_types": activated_types, "data.activated_type_count": len(activated_types),
            "data.changed_share": probability * (1.0 - unchanged_by_types),
            "data.type_count": len(AUGMENTATION_TYPES),
            "data.geometric_type_count": len(GEOMETRIC_AUGMENTATION_TYPES),
            "data.photometric_type_count": len(PHOTOMETRIC_AUGMENTATION_TYPES)}


def describe_backbone_blocks(model: ImageClassifier, example_block_number: int) -> Dict[str, Any]:
    """
    Gives the facts of the backbone chapter of the foundations series, read from the model of the classification run
    so that the narration follows the configuration when the run is trained again: the receptive field and stride
    after the last convolution of every block, the parameters of every block and of the Squeeze-and-Excitation modules,
    the blocks adding a residual shortcut, and the shapes of the block drawn layer by layer.

    Args:
        model (ImageClassifier): The model of the classification run, built on the CPU.
        example_block_number (int): The block drawn layer by layer, counted from 1.

    Returns:
        Dict[str, Any]: The facts, under the listings prefix.
    """
    backbone = model.backbone
    blocks = list(backbone.convolution_blocks)
    receptive_field_layers = backbone.compute_receptive_field_information()

    # Receptive field and stride after the last convolution of every block, and the output size of every block
    receptive_fields, block_output_sizes = [], []
    for block_number in range(1, len(blocks) + 1):
        block_layers = [layer for layer in receptive_field_layers if layer["block_number"] == block_number]
        convolution_layers = [layer for layer in block_layers if not layer["name"].startswith("Max Pooling")]
        if not convolution_layers:
            raise ValueError(f"Block {block_number} of the backbone has no convolution in its receptive field layers.")
        receptive_fields.append(int(convolution_layers[-1]["receptive_field"]))
        block_output_sizes.append(int(block_layers[-1]["output_size"]))

    # Parameters of every block, of the Squeeze-and-Excitation modules and of the head
    block_parameter_counts = [sum(parameter.numel() for parameter in block.parameters()) for block in blocks]
    backbone_parameter_count = sum(parameter.numel() for parameter in backbone.parameters())
    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    squeeze_excitation_parameter_count = sum(parameter.numel() for block in blocks
                                             for parameter in block.squeeze_and_excitation_layers.parameters())

    # Blocks adding at least one residual shortcut, from the residual pairs of every block, and the blocks after the
    # first that add none because their pairs are shifted by a change of channels
    residual_blocks = [block_number for block_number, block in enumerate(blocks, start=1)
                       if block.count_residual_additions() > 0]
    non_residual_blocks = [block_number for block_number, block in enumerate(blocks, start=1)
                           if block.residual and block.count_residual_additions() == 0]

    # Layers and residual pairs of every block for the drawing of the pairs: a pair closes at a layer of the parity
    # of ConvolutionBlock.compute_layers and starts one layer before it (the input of that layer being the shortcut)
    block_table = []
    for block_number, block in enumerate(blocks, start=1):
        closing_layers = [layer_index for layer_index in range(block.number_layers)
                          if block.residual and layer_index % 2 == (1 - block.residual_parity) and layer_index != 0]
        block_table.append({"number": block_number, "layer_count": block.number_layers,
                            "input_channels": block.input_channels, "output_channels": block.output_channels,
                            "pairs": [[closing_layer - 1, closing_layer] for closing_layer in closing_layers]})

    # Shapes of the block drawn layer by layer: its input is the output of the previous block
    example_block = blocks[example_block_number - 1]
    example_input_size = block_output_sizes[example_block_number - 2] if example_block_number > 1 else \
        int(backbone.input_image_size)
    return {"listings.block_count": len(blocks), "listings.receptive_fields": receptive_fields,
            "listings.receptive_field_labels": [str(value) for value in receptive_fields],
            "listings.final_receptive_field": receptive_fields[-1],
            "listings.final_stride": int(receptive_field_layers[-1]["stride"]),
            "listings.input_size": int(backbone.input_image_size),
            "listings.block_parameter_counts": block_parameter_counts,
            "listings.backbone_parameter_count": backbone_parameter_count,
            "listings.head_parameter_count": model_parameter_count - backbone_parameter_count,
            "listings.last_block_parameter_count": block_parameter_counts[-1],
            "listings.last_block_parameter_percent": round(100 * block_parameter_counts[-1] /
                                                           backbone_parameter_count),
            "listings.squeeze_excitation_parameter_count": squeeze_excitation_parameter_count,
            "listings.block_parameter_labels": [f"{count:,}" for count in block_parameter_counts],
            "listings.block_labels": [f"block {block_number}" for block_number in range(1, len(blocks) + 1)],
            "listings.block_table": block_table,
            "listings.residual_blocks": join_numbers(values=residual_blocks),
            "listings.residual_block_count": len(residual_blocks),
            "listings.non_residual_blocks": join_numbers(values=non_residual_blocks),
            "listings.example_block.number": example_block_number,
            "listings.example_block.layer_count": example_block.number_layers,
            "listings.example_block.input_channels": example_block.input_channels,
            "listings.example_block.output_channels": example_block.output_channels,
            "listings.example_block.input_size": example_input_size,
            "listings.example_block.output_size": block_output_sizes[example_block_number - 1],
            "listings.example_block.layer_weight_count": 9 * example_block.output_channels ** 2,
            "listings.example_block.residual_additions": example_block.count_residual_additions()}


def make_listings(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Writes the folder tree of the classification run and the summary panels of its model into the listings folder of
    the foundations figures, and gives the facts the storyboards quote from them.

    Args:
        output_folder (Path): The figures folder of the foundations series.
        run_folder (Path): The folder of the classification run of tutorial_sources.yaml.

    Returns:
        Dict[str, Any]: The facts: the number of checkpoints and evaluations of the run, the parameters of its model,
            the schedule of its training (batch size, summaries, checkpoints, evaluations, validation, iterations),
            the facts of its backbone blocks and data pipeline (describe_backbone_blocks, describe_data_pipeline) and
            those of the repository (describe_repository).
    """
    listings_folder = output_folder / "listings"
    listings_folder.mkdir(parents=True, exist_ok=True)

    # Tree of the run folder, its checkpoints and evaluations counted for the narration
    checkpoint_names = sorted((entry.name for entry in (run_folder / "Weights").iterdir() if entry.is_dir()),
                              key=read_iteration_number)
    evaluation_names = sorted((entry.name for entry in (run_folder / "Evaluation").iterdir() if entry.is_dir()),
                              key=read_iteration_number)
    tree_lines = list_run_folder_tree(run_folder=run_folder)
    (listings_folder / "run_folder_tree.txt").write_text(data="\n".join(tree_lines) + "\n", encoding="utf-8")
    evaluation_lines = list_evaluation_folder(evaluation_folder=run_folder / "Evaluation" / evaluation_names[-1])
    (listings_folder / "evaluation_folder_tree.txt").write_text(data="\n".join(evaluation_lines) + "\n",
                                                                encoding="utf-8")

    # Summary panels of the model of the run, built on the CPU from its saved configuration
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=run_folder / "training_configuration.yaml")
    model = ImageClassifier(configuration=model_configuration, device=torch.device(device="cpu"), dtype=torch.float32)
    (listings_folder / "backbone_summary.txt").write_text(
        data=capture_printed_text(printing_function=model.backbone.print_summary), encoding="utf-8")
    (listings_folder / "receptive_field_summary.txt").write_text(
        data=capture_printed_text(printing_function=model.print_receptive_field_summary), encoding="utf-8")
    (listings_folder / "model_size_summary.txt").write_text(
        data=capture_printed_text(printing_function=model.print_model_size_summary), encoding="utf-8")

    # Facts of the backbone chapter, the third block being the one drawn layer by layer
    backbone_facts = describe_backbone_blocks(model=model, example_block_number=3)
    data_facts = describe_data_pipeline(data_configuration=model_configuration["Data"])
    repository_facts = describe_repository(repository_folder=Path(__file__).absolute().parents[3])
    return {**backbone_facts, **data_facts, **repository_facts,
            "listings.checkpoint_count": len(checkpoint_names), "listings.evaluation_count": len(evaluation_names),
            "listings.run_name": run_folder.name, "listings.last_evaluation": evaluation_names[-1],
            "listings.last_checkpoint": checkpoint_names[-1] if checkpoint_names else None,
            "listings.parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "run.batch_size": int(experiment_configuration["batch_size"]),
            "run.information_dump": int(experiment_configuration["information_dump"]),
            "run.intermediate_weight_saving_iterations": int(
                experiment_configuration["intermediate_weight_saving_iterations"]),
            "run.weight_saving_iterations": int(experiment_configuration["weight_saving_iterations"]),
            "run.validation_iteration_ratio": int(experiment_configuration["validation_iteration_ratio"]),
            "run.number_iterations": int(experiment_configuration["number_iterations"]),
            "run.learning_rate": float(experiment_configuration["learning_rate"])}
