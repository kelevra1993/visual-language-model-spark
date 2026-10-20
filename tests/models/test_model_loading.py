"""
Tests of vision_training/models/model_loading.py, which rebuilds the trained models of the export, evaluation and
inference scripts: the trained weights of a checkpoint, the moving average of the weights stored next to them, and
the refusal of the averaged weights for a checkpoint saved without them.
"""

import pytest
import torch

from pathlib import Path
from typing import Any, Dict, Tuple

from vision_training.models import model_loading
from vision_training.models.classification.image_classifier import ImageClassifier
from vision_training.models.model_loading import load_trained_model
from vision_training.trainers.checkpointing import get_checkpoint_file, save_checkpoint
from vision_training.trainers.weight_averaging import ExponentialMovingAverage
from tests.fixtures.configuration_fixtures import get_classification_model_configuration


def save_classifier_run(run_folder: Path, keep_average: bool,
                        monkeypatch: pytest.MonkeyPatch) -> Tuple[ImageClassifier, ImageClassifier, Path]:
    """
    Saves a checkpoint of a small classifier whose averaged weights differ from its trained weights, and points the
    configuration loader of model_loading.py to the configuration of the classifier, so that the run needs no
    configuration file.

    Args:
        run_folder (Path): The run folder receiving the 'Weights' folder.
        keep_average (bool): True to store the moving average of the weights in the checkpoint.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the configuration loader.

    Returns:
        Tuple[ImageClassifier, ImageClassifier, Path]: The trained classifier, the averaged classifier and the
            checkpoint file.
    """
    torch.manual_seed(seed=0)
    configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    classifier = ImageClassifier(configuration=configuration, device=device, dtype=dtype)

    # Average started from the weights of the classifier, then the trained weights moved away from it
    moving_average = ExponentialMovingAverage(model=classifier, decay=0.999)
    with torch.no_grad():
        for parameter in classifier.parameters():
            parameter.add_(other=torch.ones_like(input=parameter))

    save_checkpoint(weights_directory=run_folder / "Weights", iteration=3, model=classifier,
                    optimizer=torch.optim.Adam(params=classifier.parameters()), metadata={"task": "classification"},
                    exponential_moving_average_state=moving_average.state_dict() if keep_average else None,
                    auxiliary_heads_state=None)
    checkpoint_path = get_checkpoint_file(weights_directory=run_folder / "Weights", iteration=3)
    loaded_configuration: Tuple[Dict[str, Any], Dict[str, Any]] = ({"task": "classification"}, configuration)
    monkeypatch.setattr(target=model_loading, name="load_experiment_configuration",
                        value=lambda configuration_path: loaded_configuration)
    return classifier, moving_average.averaged_model, checkpoint_path


def test_trained_and_averaged_weights_are_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the trained weights of a checkpoint are loaded by default and its averaged weights with
    use_averaged_weights, both models in evaluation mode.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the configuration loader.
    """
    classifier, averaged_classifier, checkpoint_path = save_classifier_run(run_folder=tmp_path, keep_average=True,
                                                                           monkeypatch=monkeypatch)
    configuration_path = tmp_path / "training_configuration.yaml"

    # One model per kind of weights, compared with the saved state of the matching model
    for use_averaged_weights, expected_model in ((False, classifier), (True, averaged_classifier)):
        model = load_trained_model(configuration_path=configuration_path, checkpoint_path=checkpoint_path,
                                   device=torch.device("cpu"), use_averaged_weights=use_averaged_weights)
        assert not model.training
        for name, tensor in expected_model.state_dict().items():
            assert torch.equal(input=model.state_dict()[name], other=tensor), name


def test_averaged_weights_of_a_checkpoint_without_average_are_refused(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that asking for the averaged weights of a checkpoint saved without a moving average raises an error
    naming the setting, instead of exporting the trained weights silently.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the configuration loader.
    """
    _, _, checkpoint_path = save_classifier_run(run_folder=tmp_path, keep_average=False, monkeypatch=monkeypatch)
    with pytest.raises(expected_exception=ValueError, match="exponential_moving_average"):
        load_trained_model(configuration_path=tmp_path / "training_configuration.yaml",
                           checkpoint_path=checkpoint_path, device=torch.device("cpu"), use_averaged_weights=True)
