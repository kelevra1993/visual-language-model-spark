from pathlib import Path
from typing import Optional


def get_checkpoint_path(experiment_folder: Path, iteration: Optional[int]) -> Path:
    """
    Finds the weight file of a trained experiment, as saved by the Trainer in '<experiment>/Weights'.

    Args:
        experiment_folder (Path): The experiment folder holding 'Weights' and 'training_configuration.yaml'.
        iteration (Optional[int]): The iteration to load, or None for the last one recorded in the 'full-checkpoint'
            registry.

    Returns:
        Path: The path of the checkpoint file.

    Raises:
        FileNotFoundError: If no checkpoint exists for the requested iteration.
    """
    weights_directory = experiment_folder / "Weights"

    # The first line of the registry names the last saved iteration, e.g. 'model_checkpoint_path: "Iteration_2000"'
    if iteration is None:
        registry_path = weights_directory / "full-checkpoint"
        if not registry_path.exists():
            raise FileNotFoundError(f"No checkpoint registry found : file://{registry_path.absolute()}")
        iteration = int(registry_path.read_text().splitlines()[0].split('"')[1].split("_")[-1])

    checkpoint_path = weights_directory / f"Iteration_{iteration}" / f"model_{iteration:06}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found : file://{checkpoint_path.absolute()}")
    return checkpoint_path
