import os
import signal
import torch

from pathlib import Path
from torch import nn
from typing import Any, Dict, Optional

# Version of the checkpoint dictionaries: version 2 adds the task and the run name to the weights and optimizer state
CHECKPOINT_FORMAT_VERSION = 2

# Registry file of the 'Weights' folder of a run: its first line names the last checkpoint, the next lines list them all
CHECKPOINT_REGISTRY_NAME = "full-checkpoint"


def get_checkpoint_file(weights_directory: Path, iteration: int) -> Path:
    """
    Gives the path of the weight file saved at an iteration, in the layout written by the trainers:
    '<run>/Weights/Iteration_<iteration>/model_<iteration on 6 digits>.pt'.

    Args:
        weights_directory (Path): The 'Weights' folder of the run.
        iteration (int): The training iteration of the checkpoint.

    Returns:
        Path: The path of the weight file.
    """
    return weights_directory / f"Iteration_{iteration}" / f"model_{iteration:06}.pt"


def get_last_checkpoint_iteration(weights_directory: Path) -> int:
    """
    Reads the iteration of the most recent checkpoint of a run from its registry, which lets a training resume where it
    stopped and the deployment tools pick the last weights without being told the iteration.

    Args:
        weights_directory (Path): The 'Weights' folder of the run.

    Returns:
        int: The iteration of the last saved checkpoint, or 0 if the run has no checkpoint yet.
    """
    registry_path = weights_directory / CHECKPOINT_REGISTRY_NAME
    if not registry_path.exists():
        return 0

    # The first line names the last saved iteration, e.g. 'model_checkpoint_path: "Iteration_2000"'
    with open(registry_path, "r") as registry_file:
        first_line = registry_file.readline().strip()
    return int(first_line.split('"')[1].split("_")[-1])


def update_checkpoint_registry(weights_directory: Path, iteration: int) -> None:
    """
    Records a newly saved checkpoint in the registry of the run: the first line is updated to name it as the last
    checkpoint, and it is appended to the history of all the checkpoints.

    Args:
        weights_directory (Path): The 'Weights' folder of the run.
        iteration (int): The iteration of the newly saved checkpoint.
    """
    registry_path = weights_directory / CHECKPOINT_REGISTRY_NAME

    # Read the existing history of the run, if any
    try:
        with open(registry_path, "r") as registry_file:
            registry_lines = registry_file.readlines()
    except FileNotFoundError:
        registry_lines = []

    # Name the new checkpoint as the last one, then add it to the history
    last_checkpoint_line = f'model_checkpoint_path: "Iteration_{iteration}"\n'
    if len(registry_lines) == 0:
        registry_lines.append(last_checkpoint_line)
    else:
        registry_lines[0] = last_checkpoint_line
    registry_lines.append(f'all_model_checkpoint_paths: "Iteration_{iteration}"\n')

    with open(registry_path, "w") as registry_file:
        registry_file.writelines(registry_lines)


def save_checkpoint(weights_directory: Path, iteration: int, model: nn.Module, optimizer: torch.optim.Optimizer,
                    metadata: Dict[str, Any]) -> Path:
    """
    Persists the model weights and optimizer state of a training run, then records them in the checkpoint registry
    used to resume the training.

    The save cannot be interrupted: Ctrl+C is ignored until both the weight file and the registry are written, since
    copying every GPU tensor to the CPU can take a few seconds and a second Ctrl+C during that time would otherwise
    leave a corrupted checkpoint. The weights are first written to a temporary file and then renamed, so a checkpoint
    on disk is always either complete or absent.

    Args:
        weights_directory (Path): The 'Weights' folder of the run.
        iteration (int): The current training iteration, used for naming the weight file.
        model (nn.Module): The trained model.
        optimizer (torch.optim.Optimizer): The optimizer, whose state is needed to resume the training.
        metadata (Dict[str, Any]): Extra information stored with the weights, e.g. the task and the run name.

    Returns:
        Path: The folder of the saved checkpoint.
    """
    checkpoint_path = get_checkpoint_file(weights_directory=weights_directory, iteration=iteration)
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    temporary_checkpoint_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")

    # Ignore keyboard interrupts while saving, and restore the previous handler once the save is complete
    previous_interrupt_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        # Write to a temporary file, then atomically move it to its final name
        torch.save(obj={"iteration": iteration, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(), "format_version": CHECKPOINT_FORMAT_VERSION,
                        **metadata}, f=temporary_checkpoint_path)
        os.replace(src=temporary_checkpoint_path, dst=checkpoint_path)

        # Update the registry only once the weight file is complete
        update_checkpoint_registry(weights_directory=weights_directory, iteration=iteration)
    finally:
        signal.signal(signal.SIGINT, previous_interrupt_handler)

    return checkpoint_path.parent


def load_checkpoint(weights_directory: Path, iteration: int, device: torch.device) -> Dict[str, Any]:
    """
    Reads the checkpoint saved at an iteration, with its tensors mapped to the device of the current run.

    Args:
        weights_directory (Path): The 'Weights' folder of the run.
        iteration (int): The iteration of the checkpoint.
        device (torch.device): The device receiving the tensors.

    Returns:
        Dict[str, Any]: The checkpoint, with its 'model_state' and 'optimizer_state' (and, from format version 2, its
            'task' and 'run_name').
    """
    return torch.load(f=get_checkpoint_file(weights_directory=weights_directory, iteration=iteration),
                      map_location=device, weights_only=True)


def get_checkpoint_path(experiment_folder: Path, iteration: Optional[int]) -> Path:
    """
    Finds the weight file of a trained run, as saved by the trainers in '<run>/Weights', for the evaluation, export and
    inference tools.

    Args:
        experiment_folder (Path): The run folder holding 'Weights' and 'training_configuration.yaml'.
        iteration (Optional[int]): The iteration to load, or None for the last one recorded in the registry.

    Returns:
        Path: The path of the checkpoint file.

    Raises:
        FileNotFoundError: If the run has no registry (iteration None), or no checkpoint exists for the iteration.
    """
    weights_directory = experiment_folder / "Weights"

    # Without an explicit iteration, the registry gives the last saved one
    if iteration is None:
        registry_path = weights_directory / CHECKPOINT_REGISTRY_NAME
        if not registry_path.exists():
            raise FileNotFoundError(f"No checkpoint registry found : file://{registry_path.absolute()}")
        iteration = get_last_checkpoint_iteration(weights_directory=weights_directory)

    checkpoint_path = get_checkpoint_file(weights_directory=weights_directory, iteration=iteration)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found : file://{checkpoint_path.absolute()}")
    return checkpoint_path
