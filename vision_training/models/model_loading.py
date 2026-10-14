"""
Rebuilds a trained task model in evaluation mode from the configuration and checkpoint saved in its run folder,
choosing the model class from the task through model_registry.py. Used by the deployment exports and the evaluation
and inference scripts.
"""

import torch

from pathlib import Path

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.models.model_registry import MODEL_DTYPE, TASK_MODEL_CLASSES
from vision_training.models.task_model import TaskModel
from vision_training.trainers.checkpointing import EXPONENTIAL_MOVING_AVERAGE_STATE_KEY
from vision_training.trainers.weight_averaging import AVERAGED_MODEL_STATE_KEY


def load_trained_model(configuration_path: Path, checkpoint_path: Path, device: torch.device,
                       use_averaged_weights: bool) -> TaskModel:
    """
    Rebuilds a trained model from its saved training configuration and checkpoint, in evaluation mode, as the starting
    point of every evaluation, deployment and inference script. The model class is the one of the task saved in the
    configuration ('ExperimentConfiguration.task').

    Args:
        configuration_path (Path): The 'training_configuration.yaml' saved in the run folder.
        checkpoint_path (Path): The checkpoint file written by the trainer.
        device (torch.device): The device on which to load the model.
        use_averaged_weights (bool): True to load the moving average of the weights stored in the checkpoint
            (Optimization.exponential_moving_average) instead of the trained weights, e.g. to export them.

    Returns:
        TaskModel: The trained model in evaluation mode.

    Raises:
        ValueError: If the averaged weights are asked for a checkpoint saved without a moving average.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)

    # Build the model of the task of the run, then load its trained weights
    model_class = TASK_MODEL_CLASSES[experiment_configuration["task"]]
    model = model_class(configuration=model_configuration, device=device, dtype=MODEL_DTYPE)
    checkpoint = torch.load(f=checkpoint_path, map_location=device, weights_only=True)
    if not use_averaged_weights:
        model.load_state_dict(state_dict=checkpoint["model_state"])
        return model.to(device=device).eval()

    # Averaged weights, stored only by the runs keeping a moving average of their weights
    if EXPONENTIAL_MOVING_AVERAGE_STATE_KEY not in checkpoint:
        raise ValueError(f"The checkpoint {checkpoint_path} has no moving average of the weights: it was saved by a "
                         f"run with Optimization.exponential_moving_average.activated False.")
    model.load_state_dict(state_dict=checkpoint[EXPONENTIAL_MOVING_AVERAGE_STATE_KEY][AVERAGED_MODEL_STATE_KEY])

    return model.to(device=device).eval()
