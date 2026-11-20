"""
Checkpoint files of the training runs: the 'Weights/Iteration_<iteration>' layout, the registry of saved
iterations, the save protected from keyboard interrupts and refusing non-finite weights, the load and the read of one
stored value. Used by the general trainer to save and resume, and by the checkpoint evaluation, the model transfer, the
exports and scripts/evaluation/evaluate_checkpoint.py to find and read the weights of a run.
"""

import os
import signal
import torch

from pathlib import Path
from torch import nn
from typing import Any, Dict, List, Optional

from vision_training.utilities.os_utilities import format_path_link, print_color

# Version of the checkpoint dictionaries: version 2 adds the task and the run name to the weights and optimizer state
CHECKPOINT_FORMAT_VERSION = 2

# Registry file of the 'Weights' folder of a run: its first line names the last checkpoint, the next lines list them all
CHECKPOINT_REGISTRY_NAME = "full-checkpoint"

# Key of the exponential moving average of the weights in the checkpoints of the runs that keep one
EXPONENTIAL_MOVING_AVERAGE_STATE_KEY = "exponential_moving_average_state"


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


def find_saved_checkpoint_iterations(weights_directory: Path) -> List[int]:
    """
    Lists the iterations whose checkpoint folder holds its complete weight file, the 'Iteration_<k>/model_<k>.pt'
    renamed from its temporary file once written, read instead of the registry when the registry cannot be read (a
    process killed while writing it before the registry was replaced atomically, or a file edited by hand).

    Args:
        weights_directory (Path): The 'Weights' folder of the run.

    Returns:
        List[int]: The iterations of the complete checkpoints, sorted, empty when there is none.
    """
    saved_iterations = []
    for checkpoint_folder in weights_directory.glob("Iteration_*"):
        iteration_text = checkpoint_folder.name.split("_")[-1]
        if iteration_text.isdigit() and get_checkpoint_file(weights_directory=weights_directory,
                                                            iteration=int(iteration_text)).is_file():
            saved_iterations.append(int(iteration_text))
    return sorted(saved_iterations)


def read_checkpoint_registry(weights_directory: Path) -> Optional[List[int]]:
    """
    Reads the iterations named by the registry of a run, its first line naming the last checkpoint and the next ones
    every checkpoint, e.g. 'model_checkpoint_path: "Iteration_2000"'. A registry that cannot be read (empty or cut
    lines) is announced in yellow, the callers then reading the checkpoint folders (find_saved_checkpoint_iterations).

    Args:
        weights_directory (Path): The 'Weights' folder of the run.

    Returns:
        Optional[List[int]]: The iterations of the registry lines in their order, the last checkpoint first, or None
            when the registry is missing or cannot be read.
    """
    registry_path = weights_directory / CHECKPOINT_REGISTRY_NAME
    if not registry_path.is_file():
        return None
    with open(registry_path, "r") as registry_file:
        registry_lines = [line.strip() for line in registry_file if line.strip()]

    # Every line ends with the quoted name of a checkpoint folder, whose last part is its iteration
    registry_iterations = []
    for registry_line in registry_lines:
        quoted_parts = registry_line.split('"')
        iteration_text = quoted_parts[1].split("_")[-1] if len(quoted_parts) == 3 else ""
        if not iteration_text.isdigit():
            registry_iterations = []
            break
        registry_iterations.append(int(iteration_text))
    if not registry_iterations:
        print_color(color="yellow", output=f"The Checkpoint Registry Cannot Be Read, The Checkpoint Folders Are Read "
                                           f"Instead : {format_path_link(path=registry_path)}")
        return None
    return registry_iterations


def get_last_checkpoint_iteration(weights_directory: Path) -> int:
    """
    Reads the iteration of the most recent checkpoint of a run from its registry, which lets a training resume where it
    stopped and the deployment tools pick the last weights without being told the iteration. Without a readable
    registry, the newest complete checkpoint folder gives it, so that a run whose registry was lost still resumes.

    Args:
        weights_directory (Path): The 'Weights' folder of the run.

    Returns:
        int: The iteration of the last saved checkpoint, or 0 if the run has no checkpoint yet.
    """
    registry_iterations = read_checkpoint_registry(weights_directory=weights_directory)
    if registry_iterations is not None:
        return registry_iterations[0]

    # No readable registry: the newest checkpoint whose weight file is complete
    saved_iterations = find_saved_checkpoint_iterations(weights_directory=weights_directory)
    return saved_iterations[-1] if saved_iterations else 0


def get_checkpoint_iterations(weights_directory: Path) -> List[int]:
    """
    Lists the iteration of every checkpoint recorded in the registry of a run, the intermediate checkpoints included,
    so that the evaluation tools can evaluate all of them, or of every complete checkpoint folder when the registry
    cannot be read.

    Args:
        weights_directory (Path): The 'Weights' folder of the run.

    Returns:
        List[int]: The recorded iterations, sorted and without repetition, empty if the run has no checkpoint.
    """
    registry_iterations = read_checkpoint_registry(weights_directory=weights_directory)
    if registry_iterations is None:
        return find_saved_checkpoint_iterations(weights_directory=weights_directory)
    return sorted(set(registry_iterations))


def synchronize_path(path: Path) -> None:
    """
    Flushes a file, or the entries of a folder, from the system cache to the disk (fsync), so that a power cut or a
    crash of the machine right after a checkpoint is saved keeps it (replace_durably).

    Args:
        path (Path): The file or folder to flush.
    """
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def replace_durably(temporary_file_path: Path, file_path: Path) -> None:
    """
    Moves a fully written temporary file of a checkpoint (weight file or registry) to its final name, so that a crash,
    a SIGKILL or a power cut never leaves a truncated file: the temporary file is flushed to the disk, then replaces
    the final file atomically, and the folder is flushed so that the rename itself is on the disk.

    Args:
        temporary_file_path (Path): The written temporary file, next to the final file.
        file_path (Path): The final file.
    """
    synchronize_path(path=temporary_file_path)
    os.replace(src=temporary_file_path, dst=file_path)
    synchronize_path(path=file_path.parent)


def update_checkpoint_registry(weights_directory: Path, iteration: int) -> None:
    """
    Records a newly saved checkpoint in the registry of the run: the first line is updated to name it as the last
    checkpoint, and it is appended to the history of all the checkpoints. The registry is written to a temporary file
    that replaces it atomically (replace_durably), so that a reader never finds it empty or cut.

    Args:
        weights_directory (Path): The 'Weights' folder of the run.
        iteration (int): The iteration of the newly saved checkpoint.
    """
    # History of the run, rebuilt from the checkpoint folders when the registry cannot be read
    registry_iterations = read_checkpoint_registry(weights_directory=weights_directory)
    if registry_iterations is None:
        saved_iterations = [saved_iteration for saved_iteration in find_saved_checkpoint_iterations(
            weights_directory=weights_directory) if saved_iteration != iteration]
        registry_iterations = [saved_iterations[-1]] + saved_iterations if saved_iterations else []

    # Name the new checkpoint as the last one, then add it to the history
    registry_lines = [f'model_checkpoint_path: "Iteration_{iteration}"\n']
    registry_lines += [f'all_model_checkpoint_paths: "Iteration_{recorded_iteration}"\n'
                       for recorded_iteration in registry_iterations[1:] + [iteration]]
    registry_path = weights_directory / CHECKPOINT_REGISTRY_NAME
    temporary_registry_path = registry_path.with_name(registry_path.name + ".tmp")

    # A failed write (a full disk) removes its partial temporary file before raising, leaving the registry as it was
    try:
        temporary_registry_path.write_text(data="".join(registry_lines))
        replace_durably(temporary_file_path=temporary_registry_path, file_path=registry_path)
    except Exception:
        temporary_registry_path.unlink(missing_ok=True)
        raise


def find_non_finite_tensors(state_dictionary: Dict[str, Any]) -> List[str]:
    """
    Names the floating point tensors of a model state holding NaN or infinite values, which save_checkpoint refuses to
    save, so that the registry of a run never names weights a resume or an evaluation could not use (e.g. after a step
    on NaN gradients, or batch normalization statistics updated on a NaN batch).

    Args:
        state_dictionary (Dict[str, Any]): The state of the model, its parameters and buffers by name.

    Returns:
        List[str]: The names of the non-finite tensors, in the order of the state, empty when every one is finite.
    """
    return [tensor_name for tensor_name, tensor in state_dictionary.items()
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point() and not torch.isfinite(tensor).all()]


def save_checkpoint(weights_directory: Path, iteration: int, model: nn.Module, optimizer: torch.optim.Optimizer,
                    metadata: Dict[str, Any], exponential_moving_average_state: Optional[Dict[str, Any]]) -> Path:
    """
    Persists the model weights and optimizer state of a training run, with the exponential moving average of its
    weights when it keeps one, then records them in the checkpoint registry used to resume the training.

    The save cannot be interrupted: Ctrl+C is ignored until both the weight file and the registry are written, since
    copying every GPU tensor to the CPU can take a few seconds and a second Ctrl+C during that time would otherwise
    leave a corrupted checkpoint. The weights and then the registry are first written to temporary files, flushed to
    the disk and renamed (replace_durably), so a checkpoint on disk is always either complete or absent, even after a
    SIGKILL, the out-of-memory killer or a power cut. A write that fails (e.g. on a full disk) removes its temporary
    file and raises its error. Weights holding NaN or infinite values are refused before anything is written, the
    registry keeping its last checkpoint.

    Args:
        weights_directory (Path): The 'Weights' folder of the run.
        iteration (int): The current training iteration, used for naming the weight file.
        model (nn.Module): The trained model.
        optimizer (torch.optim.Optimizer): The optimizer, whose state is needed to resume the training.
        metadata (Dict[str, Any]): Extra information stored with the weights, e.g. the task and the run name.
        exponential_moving_average_state (Optional[Dict[str, Any]]): The state of the moving average of the weights
            (vision_training/trainers/weight_averaging.py), stored under EXPONENTIAL_MOVING_AVERAGE_STATE_KEY, or None
            for a run without average.

    Returns:
        Path: The folder of the saved checkpoint.

    Raises:
        ValueError: If a tensor of the model holds NaN or infinite values, naming the first ones.
    """
    # Weights holding NaN or infinite values are never saved, so that the last checkpoint of the registry, which a
    # resume loads, stays the last finite one
    model_state = model.state_dict()
    non_finite_names = find_non_finite_tensors(state_dictionary=model_state)
    if non_finite_names:
        listed_names = ", ".join(non_finite_names[:5]) + (f" and {len(non_finite_names) - 5} more"
                                                          if len(non_finite_names) > 5 else "")
        raise ValueError(f"The checkpoint of iteration {iteration} is not saved: the weights of the model hold NaN or "
                         f"infinite values (in {listed_names}), so the registry keeps naming the last checkpoint "
                         f"saved.")

    checkpoint_path = get_checkpoint_file(weights_directory=weights_directory, iteration=iteration)
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    temporary_checkpoint_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")

    # Ignore keyboard interrupts while saving, and restore the previous handler once the save is complete
    previous_interrupt_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        # Checkpoint content, with the moving average of the weights of the runs that keep one
        checkpoint = {"iteration": iteration, "model_state": model_state,
                      "optimizer_state": optimizer.state_dict(), "format_version": CHECKPOINT_FORMAT_VERSION,
                      **metadata}
        if exponential_moving_average_state is not None:
            checkpoint[EXPONENTIAL_MOVING_AVERAGE_STATE_KEY] = exponential_moving_average_state

        # Write to a temporary file, then move it to its final name once it is on the disk; a failed write (e.g. a full
        # disk) removes its partial temporary file before raising, so that no orphan file is left in the folder
        try:
            torch.save(obj=checkpoint, f=temporary_checkpoint_path)
            replace_durably(temporary_file_path=temporary_checkpoint_path, file_path=checkpoint_path)
        except Exception:
            temporary_checkpoint_path.unlink(missing_ok=True)
            raise

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
            'task' and 'run_name', and the moving average of the weights of the runs that keep one).
    """
    return torch.load(f=get_checkpoint_file(weights_directory=weights_directory, iteration=iteration),
                      map_location=device, weights_only=True)


def read_checkpoint_value(weights_directory: Path, iteration: int, key_name: str) -> Any:
    """
    Reads one value stored with the weights of a checkpoint, the file being mapped in memory so that its tensors are
    not read, for the settings a resumed run needs before it loads its checkpoint: the backbone blocks it loaded from
    a shorter source backbone, which its freezing and its optimizer groups follow while the source run may be gone.

    Args:
        weights_directory (Path): The 'Weights' folder of the run.
        iteration (int): The iteration of the checkpoint.
        key_name (str): The key of the value in the checkpoint, e.g. 'pretrained_backbone_blocks'.

    Returns:
        Any: The value, None when the checkpoint was saved before the value was stored.
    """
    checkpoint = torch.load(f=get_checkpoint_file(weights_directory=weights_directory, iteration=iteration),
                            map_location="cpu", weights_only=True, mmap=True)
    return checkpoint[key_name] if key_name in checkpoint else None


def get_checkpoint_path(experiment_folder: Path, iteration: Optional[int]) -> Path:
    """
    Finds the weight file of a trained run, as saved by the trainers in '<run>/Weights', for the evaluation, export and
    inference tools.

    Args:
        experiment_folder (Path): The run folder holding 'Weights' and 'training_configuration.yaml'.
        iteration (Optional[int]): The iteration to load, or None for the last one recorded in the registry (or, without
            a readable registry, the newest complete checkpoint folder).

    Returns:
        Path: The path of the checkpoint file.

    Raises:
        FileNotFoundError: If the run has no checkpoint (iteration None), or no checkpoint exists for the iteration.
    """
    weights_directory = experiment_folder / "Weights"

    # Without an explicit iteration, the registry gives the last saved one
    if iteration is None:
        iteration = get_last_checkpoint_iteration(weights_directory=weights_directory)
        if iteration == 0:
            raise FileNotFoundError(f"No checkpoint found in the run : {format_path_link(path=weights_directory)}")

    checkpoint_path = get_checkpoint_file(weights_directory=weights_directory, iteration=iteration)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found : {format_path_link(path=checkpoint_path)}")
    return checkpoint_path
