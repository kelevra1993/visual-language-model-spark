"""
Rebuilds a trained task model in evaluation mode from the configuration and checkpoint saved in its run folder,
choosing the model class from the task through model_registry.py. Used by the deployment exports and the evaluation
and inference scripts.
"""

import torch

from pathlib import Path

from vision_training.configuration.configuration_loader import load_experiment_configuration
from vision_training.models.model_registry import TASK_MODEL_CLASSES
from vision_training.models.task_model import TaskModel


def load_trained_model(configuration_path: Path, checkpoint_path: Path, device: torch.device) -> TaskModel:
    """
    Rebuilds a trained model from its saved training configuration and checkpoint, in evaluation mode, as the starting
    point of every evaluation, deployment and inference script. The model class is the one of the task saved in the
    configuration ('ExperimentConfiguration.task').

    Args:
        configuration_path (Path): The 'training_configuration.yaml' saved in the run folder.
        checkpoint_path (Path): The checkpoint file written by the trainer.
        device (torch.device): The device on which to load the model.

    Returns:
        TaskModel: The trained model in evaluation mode.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)

    # Build the model of the task of the run, then load its trained weights
    model_class = TASK_MODEL_CLASSES[experiment_configuration["task"]]
    model = model_class(configuration=model_configuration, device=device, dtype=experiment_configuration["dtype"])
    checkpoint = torch.load(f=checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict=checkpoint["model_state"])

    return model.to(device=device).eval()
