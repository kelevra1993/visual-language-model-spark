import torch

from pathlib import Path

from vision_training.models.detection.two_stage_detector import TwoStageDetector
from vision_training.configuration.configuration_loader import load_experiment_configuration


def load_trained_model(configuration_path: Path, checkpoint_path: Path, device: torch.device) -> TwoStageDetector:
    """
    Rebuilds a trained detector from its saved training configuration and checkpoint, in evaluation mode, as the
    starting point of every deployment and inference script.

    Args:
        configuration_path (Path): The 'training_configuration.yaml' saved in the experiment folder.
        checkpoint_path (Path): The checkpoint file written by the Trainer.
        device (torch.device): The device on which to load the model.

    Returns:
        TwoStageDetector: The trained detector in evaluation mode.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)

    model = TwoStageDetector(configuration=model_configuration, device=device, dtype=experiment_configuration["dtype"])
    checkpoint = torch.load(f=checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict=checkpoint["model_state"])

    return model.to(device=device).eval()
