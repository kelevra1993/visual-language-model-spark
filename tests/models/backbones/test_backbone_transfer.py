import copy
import pytest
import torch
import yaml

from pathlib import Path
from typing import Any, Dict

from vision_training.models.backbones.backbone_transfer import freeze_backbone, load_pretrained_backbone
from vision_training.models.classification.image_classifier import ImageClassifier
from vision_training.models.detection.two_stage_detector import TwoStageDetector
from vision_training.trainers.checkpointing import save_checkpoint
from tests.fixtures.configuration_fixtures import get_detection_model_configuration


def get_classifier_configuration(detection_configuration: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds an image classifier configuration sharing the backbone of the detector configuration, without detection
    blocks, so that its trained backbone can initialise the detector.

    Args:
        detection_configuration (Dict[str, Any]): The detector configuration of the tests.

    Returns:
        Dict[str, Any]: The classifier configuration.
    """
    backbone_configuration = copy.deepcopy(detection_configuration["Backbone"])
    backbone_configuration["detection_convolution_block_indices"] = []
    return {"Data": {"image_settings": {"size": 512}, "label_dictionary": {1: 0, 2: 1}},
            "Backbone": backbone_configuration,
            "ClassificationHead": {"global_pooling": "average", "fully_connected": [], "dropout_rate": 0.0,
                                   "output_mode": "multi_class", "label_smoothing": 0.0, "class_weighting": "none"}}


def write_source_run(run_folder: Path, configuration: Dict[str, Any], model: torch.nn.Module) -> None:
    """
    Writes a trained run as the trainers save it: its configuration and one checkpoint of its model.

    Args:
        run_folder (Path): The run folder.
        configuration (Dict[str, Any]): The model configuration of the run.
        model (torch.nn.Module): The model whose weights are saved.
    """
    run_folder.mkdir(parents=True)
    (run_folder / "training_configuration.yaml").write_text(data=yaml.safe_dump(data=configuration))
    save_checkpoint(weights_directory=run_folder / "Weights", iteration=3, model=model,
                    optimizer=torch.optim.Adam(params=model.parameters()), metadata={"task": "classification"})


def get_pretrained_backbone_configuration(backbone_configuration: Dict[str, Any],
                                          source_folder: Path) -> Dict[str, Any]:
    """
    Adds the pretrained settings pointing to a source run to a backbone configuration.

    Args:
        backbone_configuration (Dict[str, Any]): The backbone configuration of the run receiving the backbone.
        source_folder (Path): The run folder of the source backbone.

    Returns:
        Dict[str, Any]: The backbone configuration with its 'pretrained' settings.
    """
    return {**backbone_configuration, "pretrained": {"experiment_folder": str(source_folder), "iteration": None,
                                                     "freeze": False, "learning_rate_multiplier": 1.0}}


def test_classification_backbone_initialises_a_detector(tmp_path: Path) -> None:
    """
    Verifies the transfer between two tasks: the backbone of a trained classifier initialises the backbone of a
    detector exactly, although the two runs differ by their detection blocks and their image size, while the heads of
    the detector keep their own weights.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    torch.manual_seed(seed=0)
    detection_configuration, device, dtype = get_detection_model_configuration()
    classifier_configuration = get_classifier_configuration(detection_configuration=detection_configuration)
    classifier = ImageClassifier(configuration=classifier_configuration, device=device, dtype=dtype)
    write_source_run(run_folder=tmp_path / "classification_run", configuration=classifier_configuration,
                     model=classifier)

    detector = TwoStageDetector(configuration=detection_configuration, device=device, dtype=dtype)
    detector_heads_before = copy.deepcopy(detector.detector_heads_dictionary.state_dict())
    checkpoint_path = load_pretrained_backbone(
        backbone=detector.backbone, backbone_configuration=get_pretrained_backbone_configuration(
            backbone_configuration=detection_configuration["Backbone"], source_folder=tmp_path / "classification_run"),
        input_image_size=1024, device=device)

    assert checkpoint_path.name == "model_000003.pt"
    for parameter_name, parameter_value in classifier.backbone.state_dict().items():
        assert torch.equal(input=detector.backbone.state_dict()[parameter_name], other=parameter_value)
    for parameter_name, parameter_value in detector.detector_heads_dictionary.state_dict().items():
        assert torch.equal(input=parameter_value, other=detector_heads_before[parameter_name])


def test_different_architectures_are_rejected(tmp_path: Path) -> None:
    """
    Verifies that a backbone is not transferred between runs whose backbone layers differ, the error listing every
    difference.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    detection_configuration, device, dtype = get_detection_model_configuration()
    classifier_configuration = get_classifier_configuration(detection_configuration=detection_configuration)
    write_source_run(run_folder=tmp_path / "classification_run", configuration=classifier_configuration,
                     model=ImageClassifier(configuration=classifier_configuration, device=device, dtype=dtype))

    # Wider last block and residual connections in the run receiving the backbone
    target_backbone_configuration = copy.deepcopy(detection_configuration["Backbone"])
    target_backbone_configuration["convolutions"]["9"] = [2, 128]
    target_backbone_configuration["modules"]["residual_nets"] = True
    target_configuration = {**detection_configuration, "Backbone": target_backbone_configuration}
    detector = TwoStageDetector(configuration=target_configuration, device=device, dtype=dtype)

    with pytest.raises(expected_exception=ValueError) as error_information:
        load_pretrained_backbone(
            backbone=detector.backbone, backbone_configuration=get_pretrained_backbone_configuration(
                backbone_configuration=target_backbone_configuration, source_folder=tmp_path / "classification_run"),
            input_image_size=1024, device=device)
    assert "'Backbone.convolutions' is" in str(error_information.value)
    assert "'Backbone.modules' is" in str(error_information.value)


def test_frozen_backbone_keeps_its_weights_and_statistics() -> None:
    """
    Verifies that a frozen backbone, kept in evaluation mode as the trainer does, receives no gradient and keeps its
    batch normalization statistics during a training step of the rest of the model.
    """
    torch.manual_seed(seed=0)
    detection_configuration, device, dtype = get_detection_model_configuration()
    classifier_configuration = get_classifier_configuration(detection_configuration=detection_configuration)
    classifier_configuration["Backbone"]["normalization"]["feature_map_normalization"] = "batch"
    classifier_configuration["Data"]["image_settings"]["size"] = 512
    classifier = ImageClassifier(configuration=classifier_configuration, device=device, dtype=dtype)
    freeze_backbone(backbone=classifier.backbone)
    backbone_state_before = copy.deepcopy(classifier.backbone.state_dict())

    # Training mode for the model, evaluation mode for the frozen backbone, as in BaseTrainer.set_training_mode
    classifier.train()
    classifier.backbone.eval()
    optimizer = torch.optim.Adam(params=classifier.parameters(), lr=0.1)
    loss = classifier(input_tensor=torch.rand(size=(2, 3, 512, 512)) * 255.0,
                      labels=torch.tensor(data=[0, 1]))["classification_loss"]
    loss.backward()
    optimizer.step()

    assert all(not parameter.requires_grad and parameter.grad is None for parameter in classifier.backbone.parameters())
    for parameter_name, parameter_value in classifier.backbone.state_dict().items():
        assert torch.equal(input=parameter_value, other=backbone_state_before[parameter_name]), parameter_name
