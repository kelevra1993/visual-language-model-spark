"""
Tests of vision_training/models/model_transfer.py: backbones, temporal modules, heads, segmentation decoders and
region proposal and detection heads of a saved run initialising other models and tasks, rejected architectures and
components (flattened heads, decoders and detection heads of other levels included), order-blind transformers moving
between window lengths, source runs missing a setting, frozen and fixed normalization parameters and backbones moving
between batch and sequence normalization.
"""

import copy
import pytest
import torch
import yaml

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from vision_training.models.classification.image_classifier import ImageClassifier
from vision_training.models.detection.two_stage_detector import TwoStageDetector
from vision_training.models.model_registry import TASK_MODEL_CLASSES
from vision_training.models.model_transfer import (NORMALIZATION_LAYER_TYPES, TRANSFERABLE_COMPONENTS,
                                                   compare_detection_heads, disable_pretrained_settings,
                                                   freeze_batch_normalization_statistics,
                                                   freeze_component, freeze_normalization_parameters,
                                                   load_pretrained_component)
from vision_training.models.task_model import TaskModel
from vision_training.trainers.checkpointing import save_checkpoint
from tests.fixtures.configuration_fixtures import (get_classification_model_configuration,
                                                   get_detection_model_configuration,
                                                   get_segmentation_model_configuration,
                                                   get_sequence_comparison_model_configuration,
                                                   get_sequence_model_configuration)

# Pretrained settings of a component trained from random weights, and the normalization flag of the backbone
NO_TRANSFER = {"experiment_folder": None, "iteration": None, "freeze": False, "learning_rate_multiplier": 1.0}
NO_BACKBONE_TRANSFER = {**NO_TRANSFER, "freeze_normalization_parameters": False}


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
            "ClassificationHead": {"global_pooling": "average", "fully_connected": [],
                                   "fully_connected_normalization": "none", "batch_normalization_momentum": 0.1,
                                   "dropout_rate": 0.0, "output_mode": "multi_class", "label_smoothing": 0.0,
                                   "class_weighting": "none"}}


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


def point_to_source_run(configuration: Dict[str, Any], section_name: str, source_folder: Path) -> Dict[str, Any]:
    """
    Copies a configuration with the pretrained settings of one section pointing to a source run, the component of
    that section being then initialised by the one of the source run.

    Args:
        configuration (Dict[str, Any]): The configuration of the run receiving the component.
        section_name (str): The section of the transferred component.
        source_folder (Path): The run folder of the source component.

    Returns:
        Dict[str, Any]: The configuration with the pretrained settings of the section.
    """
    target_configuration = copy.deepcopy(configuration)
    no_transfer = NO_BACKBONE_TRANSFER if section_name == "Backbone" else NO_TRANSFER
    target_configuration[section_name]["pretrained"] = {**no_transfer, "experiment_folder": str(source_folder)}
    return target_configuration


def build_task_model(task: str, configuration: Dict[str, Any], seed: int) -> TaskModel:
    """
    Builds the model of a task on the CPU with seeded weights, so that a source model and the model receiving its
    components start from different weights.

    Args:
        task (str): The task of the model, e.g. "sequence_classification" or "semantic_segmentation".
        configuration (Dict[str, Any]): The model configuration.
        seed (int): The seed of the random weights.

    Returns:
        TaskModel: The model of the task.
    """
    torch.manual_seed(seed=seed)
    return TASK_MODEL_CLASSES[task](configuration=configuration, device=torch.device("cpu"), dtype=torch.float32)


def write_sequence_classifier_run(run_folder: Path, temporal_type: str) -> Tuple[Dict[str, Any], TaskModel]:
    """
    Writes a trained sequence classifier run with the small sequence configuration of the tests, its temporal module
    being of the given type.

    Args:
        run_folder (Path): The run folder.
        temporal_type (str): The type of the temporal module of the run.

    Returns:
        Tuple[Dict[str, Any], TaskModel]: The configuration and the model of the run.
    """
    configuration, _, _ = get_sequence_model_configuration(temporal_type=temporal_type)
    source_model = build_task_model(task="sequence_classification", configuration=configuration, seed=0)
    write_source_run(run_folder=run_folder, configuration=configuration, model=source_model)
    return configuration, source_model


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
    checkpoint_path = load_pretrained_component(
        component=detector.backbone, section_name="Backbone", device=device,
        model_configuration=point_to_source_run(configuration=detection_configuration, section_name="Backbone",
                                                source_folder=tmp_path / "classification_run"))

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
        load_pretrained_component(
            component=detector.backbone, section_name="Backbone", device=device,
            model_configuration=point_to_source_run(configuration=target_configuration, section_name="Backbone",
                                                    source_folder=tmp_path / "classification_run"))
    assert "The Backbone Of The Source Run" in str(error_information.value)
    assert "'Backbone.convolutions' is" in str(error_information.value)
    assert "'Backbone.modules' is" in str(error_information.value)


def test_frozen_backbone_keeps_its_weights_and_statistics() -> None:
    """
    Verifies that a frozen backbone, whose batch normalization layers stay in evaluation mode as the trainer keeps
    them, receives no gradient and keeps its batch normalization statistics during a training step of the rest of
    the model.
    """
    torch.manual_seed(seed=0)
    detection_configuration, device, dtype = get_detection_model_configuration()
    classifier_configuration = get_classifier_configuration(detection_configuration=detection_configuration)
    classifier_configuration["Backbone"]["normalization"]["feature_map_normalization"] = "batch"
    classifier_configuration["Data"]["image_settings"]["size"] = 512
    classifier = ImageClassifier(configuration=classifier_configuration, device=device, dtype=dtype)
    freeze_component(component=classifier.backbone)
    backbone_state_before = copy.deepcopy(classifier.backbone.state_dict())

    # Training mode for the model, evaluation mode for the batch normalization of the frozen backbone, as in
    # BaseTrainer.set_training_mode
    classifier.train()
    freeze_batch_normalization_statistics(component=classifier.backbone)
    optimizer = torch.optim.Adam(params=classifier.parameters(), lr=0.1)
    loss = classifier(input_tensor=torch.rand(size=(2, 3, 512, 512)) * 255.0,
                      labels=torch.tensor(data=[0, 1]))["classification_loss"]
    loss.backward()
    optimizer.step()

    assert all(not parameter.requires_grad and parameter.grad is None for parameter in classifier.backbone.parameters())
    for parameter_name, parameter_value in classifier.backbone.state_dict().items():
        assert torch.equal(input=parameter_value, other=backbone_state_before[parameter_name]), parameter_name


def test_temporal_module_and_head_initialise_another_sequence_classifier(tmp_path: Path) -> None:
    """
    Verifies the transfer of the temporal module and of the classification head between two sequence classifiers:
    both components start from exactly the weights of the source run, while the backbone, not transferred, keeps its
    own weights.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration, source_model = write_sequence_classifier_run(run_folder=tmp_path / "source_run",
                                                                temporal_type="gru")
    target_model = build_task_model(task="sequence_classification", configuration=configuration, seed=1)
    target_backbone_before = copy.deepcopy(target_model.backbone.state_dict())

    for section_name, attribute_name in [("TemporalModel", "temporal_module"),
                                         ("ClassificationHead", "classification_head")]:
        load_pretrained_component(component=getattr(target_model, attribute_name), section_name=section_name,
                                  model_configuration=point_to_source_run(configuration=configuration,
                                                                          section_name=section_name,
                                                                          source_folder=tmp_path / "source_run"),
                                  device=torch.device("cpu"))
        for tensor_name, tensor_value in getattr(source_model, attribute_name).state_dict().items():
            assert torch.equal(input=getattr(target_model, attribute_name).state_dict()[tensor_name],
                               other=tensor_value), tensor_name
    for tensor_name, tensor_value in target_model.backbone.state_dict().items():
        assert torch.equal(input=tensor_value, other=target_backbone_before[tensor_name])


@pytest.mark.parametrize(argnames="target_task", argvalues=["sequence_labelling", "sequence_comparison"])
def test_sequence_classifier_temporal_module_initialises_other_sequence_tasks(tmp_path: Path,
                                                                              target_task: str) -> None:
    """
    Verifies that the temporal module of a sequence classifier initialises the temporal module of a sequence labeller
    (which keeps one vector per frame) and the temporal module shared by both members of a sequence comparator.

    Args:
        tmp_path (Path): The pytest temporary directory.
        target_task (str): The task of the run receiving the temporal module.
    """
    _, source_model = write_sequence_classifier_run(run_folder=tmp_path / "source_run", temporal_type="gru")
    if target_task == "sequence_comparison":
        target_configuration, _, _ = get_sequence_comparison_model_configuration(temporal_type="gru",
                                                                                 fusion_type="difference")
    else:
        target_configuration, _, _ = get_sequence_model_configuration(temporal_type="gru")
    target_model = build_task_model(task=target_task, configuration=target_configuration, seed=1)

    load_pretrained_component(component=target_model.temporal_module, section_name="TemporalModel",
                              model_configuration=point_to_source_run(configuration=target_configuration,
                                                                      section_name="TemporalModel",
                                                                      source_folder=tmp_path / "source_run"),
                              device=torch.device("cpu"))
    for tensor_name, tensor_value in source_model.temporal_module.state_dict().items():
        assert torch.equal(input=target_model.temporal_module.state_dict()[tensor_name], other=tensor_value)


def use_lstm(configuration: Dict[str, Any]) -> None:
    """
    Uses an LSTM instead of the GRU of the source run.

    Args:
        configuration (Dict[str, Any]): The configuration of the run receiving the component, modified in place.
    """
    configuration["TemporalModel"]["type"] = "lstm"


def double_hidden_size(configuration: Dict[str, Any]) -> None:
    """
    Doubles the hidden size of the temporal module.

    Args:
        configuration (Dict[str, Any]): The configuration of the run receiving the component, modified in place.
    """
    configuration["TemporalModel"]["hidden_size"] = 32


def read_backwards(configuration: Dict[str, Any]) -> None:
    """
    Makes the recurrent module bidirectional.

    Args:
        configuration (Dict[str, Any]): The configuration of the run receiving the component, modified in place.
    """
    configuration["TemporalModel"]["bidirectional"] = True


def double_attention_heads(configuration: Dict[str, Any]) -> None:
    """
    Doubles the number of heads of the transformer.

    Args:
        configuration (Dict[str, Any]): The configuration of the run receiving the component, modified in place.
    """
    configuration["TemporalModel"]["number_heads"] = 4


def widen_temporal_kernel(configuration: Dict[str, Any]) -> None:
    """
    Widens the kernel of the temporal convolution.

    Args:
        configuration (Dict[str, Any]): The configuration of the run receiving the component, modified in place.
    """
    configuration["TemporalModel"]["kernel_size"] = 5


def lengthen_windows(configuration: Dict[str, Any]) -> None:
    """
    Lengthens the windows of frames, and so the positional embedding of the transformer.

    Args:
        configuration (Dict[str, Any]): The configuration of the run receiving the component, modified in place.
    """
    configuration["Data"]["sequence_settings"]["sequence_length"] = 6


def remove_positional_embedding(configuration: Dict[str, Any]) -> None:
    """
    Removes the positional embedding of the transformer, one of its weights.

    Args:
        configuration (Dict[str, Any]): The configuration of the run receiving the component, modified in place.
    """
    configuration["TemporalModel"]["positional_embedding"] = False


def widen_last_backbone_block(configuration: Dict[str, Any]) -> None:
    """
    Widens the last backbone block, and so the frame vectors read by the temporal module.

    Args:
        configuration (Dict[str, Any]): The configuration of the run receiving the component, modified in place.
    """
    configuration["Backbone"]["convolutions"]["4"] = [2, 64]


def add_head_dropout(configuration: Dict[str, Any]) -> None:
    """
    Adds a dropout before the layers of the classification head, which keeps the names of its weights.

    Args:
        configuration (Dict[str, Any]): The configuration of the run receiving the component, modified in place.
    """
    configuration["ClassificationHead"]["dropout_rate"] = 0.3


def normalize_head_layers(configuration: Dict[str, Any]) -> None:
    """
    Normalizes the outputs of the hidden layers of the classification head, which adds batch normalization layers.

    Args:
        configuration (Dict[str, Any]): The configuration of the run receiving the component, modified in place.
    """
    configuration["ClassificationHead"]["fully_connected_normalization"] = "batch"


@pytest.mark.parametrize(argnames=("source_temporal_type", "modify_function", "section_name", "expected_message"),
                         argvalues=[
    ("gru", use_lstm, "TemporalModel", "'TemporalModel.type' is 'gru' in the source run and 'lstm' in this run."),
    ("gru", double_hidden_size, "TemporalModel",
     "'TemporalModel.hidden_size' is 16 in the source run and 32 in this run."),
    ("gru", read_backwards, "TemporalModel",
     "'TemporalModel.bidirectional' is False in the source run and True in this run."),
    ("transformer", double_attention_heads, "TemporalModel",
     "'TemporalModel.number_heads' is 2 in the source run and 4 in this run."),
    ("temporal_convolution", widen_temporal_kernel, "TemporalModel",
     "'TemporalModel.kernel_size' is 3 in the source run and 5 in this run."),
    ("transformer", lengthen_windows, "TemporalModel",
     "'Data.sequence_settings.sequence_length' is 4 in the source run and 6 in this run, the size of the positional "
     "embedding of the transformer."),
    ("transformer", remove_positional_embedding, "TemporalModel",
     "'TemporalModel.positional_embedding' is True in the source run and False in this run."),
    ("gru", widen_last_backbone_block, "TemporalModel",
     "The last backbone block has 32 channels in the source run and 64 in this run"),
    ("gru", normalize_head_layers, "ClassificationHead",
     "'ClassificationHead.fully_connected_normalization' is 'none' in the source run and 'batch' in this run.")])
def test_different_components_are_rejected(tmp_path: Path, source_temporal_type: str,
                                           modify_function: Callable[[Dict[str, Any]], None], section_name: str,
                                           expected_message: str) -> None:
    """
    Verifies that a temporal module or a classification head is not transferred when a setting defining its weights
    differs between the runs, the error naming the setting.

    Args:
        tmp_path (Path): The pytest temporary directory.
        source_temporal_type (str): The temporal module of the source sequence classifier.
        modify_function (Callable[[Dict[str, Any]], None]): Function changing the configuration of the run receiving
            the component.
        section_name (str): The section of the transferred component.
        expected_message (str): The expected description of the difference.
    """
    configuration, _ = write_sequence_classifier_run(run_folder=tmp_path / "source_run",
                                                     temporal_type=source_temporal_type)
    target_configuration = copy.deepcopy(configuration)
    modify_function(configuration=target_configuration)
    target_model = build_task_model(task="sequence_classification", configuration=target_configuration, seed=1)
    attribute_name = "temporal_module" if section_name == "TemporalModel" else "classification_head"

    with pytest.raises(expected_exception=ValueError) as error_information:
        load_pretrained_component(component=getattr(target_model, attribute_name), section_name=section_name,
                                  model_configuration=point_to_source_run(configuration=target_configuration,
                                                                          section_name=section_name,
                                                                          source_folder=tmp_path / "source_run"),
                                  device=torch.device("cpu"))
    assert "Cannot Initialise This Run, Their Architectures Differ" in str(error_information.value)
    assert expected_message in str(error_information.value)


def test_order_blind_transformer_initialises_runs_of_other_sequence_lengths(tmp_path: Path) -> None:
    """
    Verifies that a transformer without positional embedding, whose weights do not depend on the number of frames,
    initialises the transformer of a run with longer windows with exactly its weights.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration, _, _ = get_sequence_model_configuration(temporal_type="transformer")
    configuration["TemporalModel"]["positional_embedding"] = False
    source_model = build_task_model(task="sequence_classification", configuration=configuration, seed=0)
    write_source_run(run_folder=tmp_path / "source_run", configuration=configuration, model=source_model)
    target_configuration = copy.deepcopy(configuration)
    lengthen_windows(configuration=target_configuration)
    target_model = build_task_model(task="sequence_classification", configuration=target_configuration, seed=1)

    load_pretrained_component(component=target_model.temporal_module, section_name="TemporalModel",
                              model_configuration=point_to_source_run(configuration=target_configuration,
                                                                      section_name="TemporalModel",
                                                                      source_folder=tmp_path / "source_run"),
                              device=torch.device("cpu"))
    for tensor_name, tensor_value in source_model.temporal_module.state_dict().items():
        assert torch.equal(input=target_model.temporal_module.state_dict()[tensor_name], other=tensor_value), \
            tensor_name


def test_source_run_written_before_a_temporal_setting_is_rejected(tmp_path: Path) -> None:
    """
    Verifies that a transformer run written before the positional embedding flag existed is not read with a guessed
    value: the transfer stops with the missing setting and the script migrating the run.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration, source_model = write_sequence_classifier_run(run_folder=tmp_path / "source_run",
                                                                temporal_type="transformer")
    old_configuration = copy.deepcopy(configuration)
    del old_configuration["TemporalModel"]["positional_embedding"]
    (tmp_path / "source_run" / "training_configuration.yaml").write_text(data=yaml.safe_dump(data=old_configuration))
    target_model = build_task_model(task="sequence_classification", configuration=configuration, seed=1)

    with pytest.raises(expected_exception=ValueError, match="positional_embedding' is missing from the source run"):
        load_pretrained_component(component=target_model.temporal_module, section_name="TemporalModel",
                                  model_configuration=point_to_source_run(configuration=configuration,
                                                                          section_name="TemporalModel",
                                                                          source_folder=tmp_path / "source_run"),
                                  device=torch.device("cpu"))


@pytest.mark.parametrize(argnames=("old_normalization", "expected_message"), argvalues=[
    (None, "fully_connected_normalization' is missing from the source run"),
    (False, "fully_connected_normalization' is False in the source run, written before the layer normalization "
            "existed")])
def test_source_run_written_before_the_head_normalization_is_rejected(tmp_path: Path, old_normalization: bool | None,
                                                                      expected_message: str) -> None:
    """
    Verifies that a classification head of a run written before the normalization of the hidden layers existed, or
    with True or False before the layer normalization existed, is not read with a guessed value: the transfer stops
    with the setting and the script migrating the run.

    Args:
        tmp_path (Path): The pytest temporary directory.
        old_normalization (bool | None): The normalization of the source run, None when it is missing.
        expected_message (str): The start of the expected error.
    """
    configuration, _ = write_sequence_classifier_run(run_folder=tmp_path / "source_run", temporal_type="gru")
    old_configuration = copy.deepcopy(configuration)
    if old_normalization is None:
        del old_configuration["ClassificationHead"]["fully_connected_normalization"]
    else:
        old_configuration["ClassificationHead"]["fully_connected_normalization"] = old_normalization
    (tmp_path / "source_run" / "training_configuration.yaml").write_text(data=yaml.safe_dump(data=old_configuration))
    target_model = build_task_model(task="sequence_classification", configuration=configuration, seed=1)

    with pytest.raises(expected_exception=ValueError, match=expected_message):
        load_pretrained_component(component=target_model.classification_head, section_name="ClassificationHead",
                                  model_configuration=point_to_source_run(configuration=configuration,
                                                                          section_name="ClassificationHead",
                                                                          source_folder=tmp_path / "source_run"),
                                  device=torch.device("cpu"))


def test_components_of_another_batch_normalization_momentum_are_transferred(tmp_path: Path) -> None:
    """
    Verifies that the batch normalization momentum, which changes no tensor, does not stop the transfer: the backbone
    of a run written before the momentum existed and its normalized classification head initialise a run of other
    momentums with exactly their weights and batch statistics.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration, _, _ = get_sequence_model_configuration(temporal_type="gru")
    configuration["ClassificationHead"]["fully_connected_normalization"] = "batch"
    source_model = build_task_model(task="sequence_classification", configuration=configuration, seed=0)
    old_configuration = copy.deepcopy(configuration)
    del old_configuration["Backbone"]["normalization"]["batch_normalization_momentum"]
    write_source_run(run_folder=tmp_path / "source_run", configuration=old_configuration, model=source_model)

    # Run of other momentums receiving the backbone and the head of the source run
    target_configuration = copy.deepcopy(configuration)
    target_configuration["Backbone"]["normalization"]["batch_normalization_momentum"] = 0.01
    target_configuration["ClassificationHead"]["batch_normalization_momentum"] = 0.01
    target_model = build_task_model(task="sequence_classification", configuration=target_configuration, seed=1)
    for section_name in ["Backbone", "ClassificationHead"]:
        load_pretrained_component(component=getattr(target_model, TRANSFERABLE_COMPONENTS[section_name]),
                                  section_name=section_name, device=torch.device("cpu"),
                                  model_configuration=point_to_source_run(configuration=target_configuration,
                                                                          section_name=section_name,
                                                                          source_folder=tmp_path / "source_run"))
        source_state = getattr(source_model, TRANSFERABLE_COMPONENTS[section_name]).state_dict()
        target_state = getattr(target_model, TRANSFERABLE_COMPONENTS[section_name]).state_dict()
        for tensor_name, tensor_value in source_state.items():
            assert torch.equal(input=target_state[tensor_name], other=tensor_value), tensor_name


def test_head_with_another_dropout_rate_is_transferred(tmp_path: Path) -> None:
    """
    Verifies that a classification head trained without dropout initialises a head with dropout: the heads always
    hold their dropout layers, so the dropout rate changes neither the names nor the shapes of their weights.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration, source_model = write_sequence_classifier_run(run_folder=tmp_path / "source_run",
                                                                temporal_type="gru")
    target_configuration = copy.deepcopy(configuration)
    add_head_dropout(configuration=target_configuration)
    target_model = build_task_model(task="sequence_classification", configuration=target_configuration, seed=1)

    # Transfer the head, then compare every tensor with the one of the source run
    load_pretrained_component(component=target_model.classification_head, section_name="ClassificationHead",
                              model_configuration=point_to_source_run(configuration=target_configuration,
                                                                      section_name="ClassificationHead",
                                                                      source_folder=tmp_path / "source_run"),
                              device=torch.device("cpu"))
    for tensor_name, tensor_value in source_model.classification_head.state_dict().items():
        assert torch.equal(input=target_model.classification_head.state_dict()[tensor_name], other=tensor_value), \
            tensor_name


def test_head_tensors_of_another_shape_are_listed(tmp_path: Path) -> None:
    """
    Verifies that the head of a sequence classifier, whose settings match the head of a sequence comparator, is not
    loaded into it: the comparison head reads the three fused member vectors, and the mismatching tensor is named
    with both shapes.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    write_sequence_classifier_run(run_folder=tmp_path / "source_run", temporal_type="gru")
    target_configuration, _, _ = get_sequence_comparison_model_configuration(temporal_type="gru",
                                                                             fusion_type="difference")
    target_model = build_task_model(task="sequence_comparison", configuration=target_configuration, seed=1)

    with pytest.raises(expected_exception=ValueError) as error_information:
        load_pretrained_component(component=target_model.classification_head, section_name="ClassificationHead",
                                  model_configuration=point_to_source_run(configuration=target_configuration,
                                                                          section_name="ClassificationHead",
                                                                          source_folder=tmp_path / "source_run"),
                                  device=torch.device("cpu"))
    assert "Classification Head Tensors Of The Checkpoint" in str(error_information.value)
    assert ("'fully_connected_block.1.weight' has the shape [16, 16] in the source run and [16, 48] in this run."
            in str(error_information.value))


def write_flattened_classifier_run(run_folder: Path, global_pooling: str) -> Tuple[Dict[str, Any], TaskModel]:
    """
    Writes a trained image classifier run with the small classification configuration of the tests (64 pixel images),
    its head reading the pooled or flattened final feature map.

    Args:
        run_folder (Path): The run folder.
        global_pooling (str): The reduction of the final feature map of the run, "average", "max" or "flatten".

    Returns:
        Tuple[Dict[str, Any], TaskModel]: The configuration and the model of the run.
    """
    configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    configuration["ClassificationHead"]["global_pooling"] = global_pooling
    torch.manual_seed(seed=0)
    source_model = ImageClassifier(configuration=configuration, device=device, dtype=dtype)
    write_source_run(run_folder=run_folder, configuration=configuration, model=source_model)
    return configuration, source_model


def test_flattened_head_initialises_a_flattened_run(tmp_path: Path) -> None:
    """
    Verifies that the head of a classifier reading the flattened feature map initialises exactly the head of another
    flattened classifier of the same image size.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration, source_model = write_flattened_classifier_run(run_folder=tmp_path / "source_run",
                                                                 global_pooling="flatten")
    torch.manual_seed(seed=1)
    target_model = ImageClassifier(configuration=configuration, device=torch.device("cpu"), dtype=torch.float32)
    load_pretrained_component(component=target_model.classification_head, section_name="ClassificationHead",
                              model_configuration=point_to_source_run(configuration=configuration,
                                                                      section_name="ClassificationHead",
                                                                      source_folder=tmp_path / "source_run"),
                              device=torch.device("cpu"))

    for parameter_name, parameter_value in source_model.classification_head.state_dict().items():
        assert torch.equal(input=target_model.classification_head.state_dict()[parameter_name], other=parameter_value)


@pytest.mark.parametrize(argnames=("source_pooling", "target_pooling", "target_image_size", "expected_message"),
                         argvalues=[
    ("average", "flatten", 64, "'ClassificationHead.global_pooling' is 'average' in the source run and 'flatten' in "
                               "this run: a flattened feature map and a pooled one give input vectors of different "
                               "sizes."),
    ("flatten", "max", 64, "'ClassificationHead.global_pooling' is 'flatten' in the source run and 'max' in this run"),
    ("flatten", "flatten", 32, "'Data.image_settings.size' is 64 in the source run and 32 in this run: the flattened "
                               "feature map, the input of the head, has a size following the image size.")])
def test_heads_of_other_input_sizes_are_rejected(tmp_path: Path, source_pooling: str, target_pooling: str,
                                                 target_image_size: int, expected_message: str) -> None:
    """
    Verifies that a classification head is not transferred between a pooled and a flattened feature map, nor between
    two flattened maps of different image sizes, the error naming the setting to change.

    Args:
        tmp_path (Path): The pytest temporary directory.
        source_pooling (str): The reduction of the final feature map of the source run.
        target_pooling (str): The reduction of the final feature map of the run receiving the head.
        target_image_size (int): The image size of the run receiving the head.
        expected_message (str): The expected description of the difference.
    """
    configuration, _ = write_flattened_classifier_run(run_folder=tmp_path / "source_run", global_pooling=source_pooling)
    target_configuration = copy.deepcopy(configuration)
    target_configuration["ClassificationHead"]["global_pooling"] = target_pooling
    target_configuration["Data"]["image_settings"]["size"] = target_image_size
    target_model = ImageClassifier(configuration=target_configuration, device=torch.device("cpu"), dtype=torch.float32)

    with pytest.raises(expected_exception=ValueError) as error_information:
        load_pretrained_component(component=target_model.classification_head, section_name="ClassificationHead",
                                  model_configuration=point_to_source_run(configuration=target_configuration,
                                                                          section_name="ClassificationHead",
                                                                          source_folder=tmp_path / "source_run"),
                                  device=torch.device("cpu"))
    assert "Cannot Initialise This Run, Their Architectures Differ" in str(error_information.value)
    assert expected_message in str(error_information.value)


def test_source_run_without_the_component_is_rejected(tmp_path: Path) -> None:
    """
    Verifies that a temporal module is not looked for in a run whose task has none, here an image classifier.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    classifier_configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    write_source_run(run_folder=tmp_path / "classification_run", configuration=classifier_configuration,
                     model=ImageClassifier(configuration=classifier_configuration, device=device, dtype=dtype))
    target_configuration, _, _ = get_sequence_model_configuration(temporal_type="gru")
    target_model = build_task_model(task="sequence_classification", configuration=target_configuration, seed=1)

    target_configuration = point_to_source_run(configuration=target_configuration, section_name="TemporalModel",
                                               source_folder=tmp_path / "classification_run")

    with pytest.raises(expected_exception=ValueError, match="Has No 'TemporalModel' Section"):
        load_pretrained_component(component=target_model.temporal_module, section_name="TemporalModel",
                                  model_configuration=target_configuration, device=device)


@pytest.mark.parametrize(argnames="feature_map_normalization", argvalues=["batch", "layer"])
def test_normalization_parameters_stay_fixed(feature_map_normalization: str) -> None:
    """
    Verifies that the scale and shift of the batch or layer normalization layers of a backbone stay fixed during a
    training step while its convolutions learn, and that the batch normalization statistics keep following the data.

    Args:
        feature_map_normalization (str): The normalization of the backbone feature maps.
    """
    configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    configuration["Backbone"]["normalization"]["feature_map_normalization"] = feature_map_normalization
    torch.manual_seed(seed=0)
    classifier = ImageClassifier(configuration=configuration, device=device, dtype=dtype)
    freeze_normalization_parameters(component=classifier.backbone)
    backbone_state_before = copy.deepcopy(classifier.backbone.state_dict())

    # Normalization parameters frozen, every other parameter trainable
    normalization_parameter_names = {f"{module_name}.{parameter_name}"
                                     for module_name, module in classifier.backbone.named_modules()
                                     if isinstance(module, NORMALIZATION_LAYER_TYPES)
                                     for parameter_name, _ in module.named_parameters()}
    assert normalization_parameter_names
    for parameter_name, parameter in classifier.backbone.named_parameters():
        assert parameter.requires_grad == (parameter_name not in normalization_parameter_names), parameter_name

    # One training step of the whole model
    classifier.train()
    optimizer = torch.optim.Adam(params=[parameter for parameter in classifier.parameters()
                                         if parameter.requires_grad], lr=0.1)
    classifier(input_tensor=torch.rand(size=(2, 3, 64, 64)) * 255.0,
               labels=torch.tensor(data=[0, 1]))["classification_loss"].backward()
    optimizer.step()

    # Normalization parameters unchanged, convolutions changed, batch statistics updated
    backbone_state_after = classifier.backbone.state_dict()
    for parameter_name in normalization_parameter_names:
        assert torch.equal(input=backbone_state_after[parameter_name], other=backbone_state_before[parameter_name])
    convolution_weight_names = [tensor_name for tensor_name in backbone_state_after
                                if tensor_name.endswith(".weight") and backbone_state_after[tensor_name].dim() == 4]
    assert any(not torch.equal(input=backbone_state_after[tensor_name], other=backbone_state_before[tensor_name])
               for tensor_name in convolution_weight_names)
    running_mean_names = [tensor_name for tensor_name in backbone_state_after if tensor_name.endswith("running_mean")]
    assert (feature_map_normalization == "batch") == bool(running_mean_names)
    assert all(not torch.equal(input=backbone_state_after[tensor_name], other=backbone_state_before[tensor_name])
               for tensor_name in running_mean_names)


def test_pretrained_settings_are_disabled() -> None:
    """
    Verifies that turning off the transfers of a configuration removes the source run and the freezing of every
    transferable section, the backbone normalization flag included, and leaves the other settings unchanged.
    """
    configuration, _, _ = get_sequence_model_configuration(temporal_type="gru")
    configuration["Backbone"]["pretrained"] = {**NO_BACKBONE_TRANSFER, "experiment_folder": "/runs/source",
                                               "freeze_normalization_parameters": True,
                                               "learning_rate_multiplier": 0.5}
    configuration["TemporalModel"]["pretrained"] = {**NO_TRANSFER, "experiment_folder": "/runs/source",
                                                    "freeze": True, "iteration": 4}
    configuration["ClassificationHead"]["pretrained"] = {**NO_TRANSFER, "experiment_folder": "/runs/source"}

    disable_pretrained_settings(model_configuration=configuration)

    assert configuration["Backbone"]["pretrained"] == {**NO_BACKBONE_TRANSFER, "learning_rate_multiplier": 0.5}
    assert configuration["TemporalModel"]["pretrained"] == {**NO_TRANSFER, "iteration": 4}
    assert configuration["ClassificationHead"]["pretrained"] == NO_TRANSFER


@pytest.mark.parametrize(argnames=("source_normalization", "target_normalization"),
                         argvalues=[("batch", "sequence"), ("sequence", "batch")])
def test_backbone_moves_between_batch_and_sequence_normalization(tmp_path: Path, source_normalization: str,
                                                                 target_normalization: str) -> None:
    """
    Verifies that a backbone trained with batch normalization initialises a run with sequence normalization, and back,
    both layers holding the same tensors, the difference being reported.

    Args:
        tmp_path (Path): The pytest temporary directory.
        source_normalization (str): The feature map normalization of the source run.
        target_normalization (str): The feature map normalization of the run receiving the backbone.
    """
    configuration, _, _ = get_sequence_model_configuration(temporal_type="gru")
    configuration["Backbone"]["normalization"]["feature_map_normalization"] = source_normalization
    source_model = build_task_model(task="sequence_classification", configuration=configuration, seed=0)
    write_source_run(run_folder=tmp_path / "source_run", configuration=configuration, model=source_model)

    target_configuration = copy.deepcopy(configuration)
    target_configuration["Backbone"]["normalization"]["feature_map_normalization"] = target_normalization
    target_model = build_task_model(task="sequence_classification", configuration=target_configuration, seed=1)
    load_pretrained_component(component=target_model.backbone, section_name="Backbone",
                              model_configuration=point_to_source_run(configuration=target_configuration,
                                                                      section_name="Backbone",
                                                                      source_folder=tmp_path / "source_run"),
                              device=torch.device("cpu"))
    for tensor_name, tensor_value in source_model.backbone.state_dict().items():
        assert torch.equal(input=target_model.backbone.state_dict()[tensor_name], other=tensor_value), tensor_name


def test_classification_backbone_initialises_a_segmenter_and_back(tmp_path: Path) -> None:
    """
    Verifies the backbone transfers between the classification and the semantic segmentation: the backbone of a
    trained classifier initialises a segmenter exactly, and the backbone of a trained segmenter initialises a
    classifier exactly, both tasks sharing the same backbone layers.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    classifier_configuration, _, _ = get_classification_model_configuration(output_mode="multi_class")
    segmenter_configuration, _, _ = get_segmentation_model_configuration(upsampling="transposed")
    classifier = build_task_model(task="classification", configuration=classifier_configuration, seed=0)
    write_source_run(run_folder=tmp_path / "classification_run", configuration=classifier_configuration,
                     model=classifier)

    # Classifier backbone into a segmenter
    segmenter = build_task_model(task="semantic_segmentation", configuration=segmenter_configuration, seed=1)
    load_pretrained_component(component=segmenter.backbone, section_name="Backbone", device=torch.device("cpu"),
                              model_configuration=point_to_source_run(configuration=segmenter_configuration,
                                                                      section_name="Backbone",
                                                                      source_folder=tmp_path / "classification_run"))
    for tensor_name, tensor_value in classifier.backbone.state_dict().items():
        assert torch.equal(input=segmenter.backbone.state_dict()[tensor_name], other=tensor_value), tensor_name

    # Segmenter backbone, trained further, into a new classifier
    with torch.no_grad():
        for parameter in segmenter.backbone.parameters():
            parameter.add_(other=torch.ones_like(input=parameter))
    write_source_run(run_folder=tmp_path / "segmentation_run", configuration=segmenter_configuration, model=segmenter)
    new_classifier = build_task_model(task="classification", configuration=classifier_configuration, seed=2)
    load_pretrained_component(component=new_classifier.backbone, section_name="Backbone", device=torch.device("cpu"),
                              model_configuration=point_to_source_run(configuration=classifier_configuration,
                                                                      section_name="Backbone",
                                                                      source_folder=tmp_path / "segmentation_run"))
    for tensor_name, tensor_value in segmenter.backbone.state_dict().items():
        assert torch.equal(input=new_classifier.backbone.state_dict()[tensor_name], other=tensor_value), tensor_name


def test_decoder_initialises_a_segmenter_of_other_classes(tmp_path: Path) -> None:
    """
    Verifies the transfer of the segmentation decoder between two segmenters of different classes: the decoder starts
    from exactly the weights of the source run, while the segmentation head, giving the logits of other classes, keeps
    its own weights.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    source_configuration, _, _ = get_segmentation_model_configuration(upsampling="bilinear")
    source_segmenter = build_task_model(task="semantic_segmentation", configuration=source_configuration, seed=0)
    write_source_run(run_folder=tmp_path / "segmentation_run", configuration=source_configuration,
                     model=source_segmenter)

    # Segmenter of five classes, the background included, receiving the decoder of a segmenter of three classes
    target_configuration = copy.deepcopy(source_configuration)
    target_configuration["Data"]["label_dictionary"] = {"background": 0, 1: 1, 2: 2, 3: 3, 4: 4}
    target_segmenter = build_task_model(task="semantic_segmentation", configuration=target_configuration, seed=1)
    head_state_before = copy.deepcopy(target_segmenter.segmentation_head.state_dict())
    load_pretrained_component(component=target_segmenter.segmentation_decoder, section_name="SegmentationDecoder",
                              device=torch.device("cpu"),
                              model_configuration=point_to_source_run(configuration=target_configuration,
                                                                      section_name="SegmentationDecoder",
                                                                      source_folder=tmp_path / "segmentation_run"))

    for tensor_name, tensor_value in source_segmenter.segmentation_decoder.state_dict().items():
        assert torch.equal(input=target_segmenter.segmentation_decoder.state_dict()[tensor_name],
                           other=tensor_value), tensor_name
    for tensor_name, tensor_value in target_segmenter.segmentation_head.state_dict().items():
        assert torch.equal(input=tensor_value, other=head_state_before[tensor_name]), tensor_name
    assert target_segmenter.segmentation_head[1].out_channels == 5


def add_last_block_decoder_level(configuration: Dict[str, Any]) -> None:
    """
    Pools the last backbone block, whose skip connection gets one more decoder level.

    Args:
        configuration (Dict[str, Any]): The segmenter configuration, modified in place.
    """
    configuration["Backbone"]["last_max_pooling"] = True
    configuration["SegmentationDecoder"]["convolutions"]["4"] = [1, 32]


def use_bilinear_upsampling(configuration: Dict[str, Any]) -> None:
    """
    Upsamples the decoder levels bilinearly instead of with transposed convolutions.

    Args:
        configuration (Dict[str, Any]): The segmenter configuration, modified in place.
    """
    configuration["SegmentationDecoder"]["upsampling"] = "bilinear"


def widen_second_backbone_block(configuration: Dict[str, Any]) -> None:
    """
    Doubles the output channels of the second backbone block, which feeds a skip connection of the decoder.

    Args:
        configuration (Dict[str, Any]): The segmenter configuration, modified in place.
    """
    configuration["Backbone"]["convolutions"]["2"] = [1, 32]


@pytest.mark.parametrize(argnames=("modify_function", "expected_message"), argvalues=[
    (add_last_block_decoder_level, "in the source run and {'3': [1, 16], '2': [1, 16], '1': [1, 8], '4': [1, 32]} in "
                                   "this run."),
    (use_bilinear_upsampling, "'SegmentationDecoder.upsampling' is 'transposed' in the source run and 'bilinear' in "
                              "this run."),
    (widen_second_backbone_block, "The backbone blocks have [8, 16, 16, 32] output channels in the source run and "
                                  "[8, 32, 16, 32] in this run")])
def test_decoders_of_other_architectures_are_rejected(tmp_path: Path,
                                                      modify_function: Callable[[Dict[str, Any]], None],
                                                      expected_message: str) -> None:
    """
    Verifies that a segmentation decoder is not transferred to a decoder of other levels, of another upsampling, or
    reading backbone blocks of other channels, the error naming the difference.

    Args:
        tmp_path (Path): The pytest temporary directory.
        modify_function (Callable[[Dict[str, Any]], None]): Function changing the configuration receiving the decoder.
        expected_message (str): The fragment of the error message naming the difference.
    """
    source_configuration, _, _ = get_segmentation_model_configuration(upsampling="transposed")
    write_source_run(run_folder=tmp_path / "segmentation_run", configuration=source_configuration,
                     model=build_task_model(task="semantic_segmentation", configuration=source_configuration, seed=0))
    target_configuration = copy.deepcopy(source_configuration)
    modify_function(configuration=target_configuration)
    target_segmenter = build_task_model(task="semantic_segmentation", configuration=target_configuration, seed=1)

    with pytest.raises(expected_exception=ValueError) as error_information:
        load_pretrained_component(component=target_segmenter.segmentation_decoder, section_name="SegmentationDecoder",
                                  device=torch.device("cpu"),
                                  model_configuration=point_to_source_run(configuration=target_configuration,
                                                                          section_name="SegmentationDecoder",
                                                                          source_folder=tmp_path / "segmentation_run"))
    assert "The Segmentation Decoder Of The Source Run" in str(error_information.value)
    assert expected_message in str(error_information.value)


def write_detector_run(run_folder: Path) -> Tuple[Dict[str, Any], TaskModel]:
    """
    Writes a trained object detection run with the small detector configuration of the tests, whose region proposal
    and detection heads initialise the heads of other detection runs.

    Args:
        run_folder (Path): The run folder.

    Returns:
        Tuple[Dict[str, Any], TaskModel]: The configuration and the model of the run.
    """
    configuration, _, _ = get_detection_model_configuration()
    source_detector = build_task_model(task="object_detection", configuration=configuration, seed=0)
    write_source_run(run_folder=run_folder, configuration=configuration, model=source_detector)
    return configuration, source_detector


def test_detection_heads_initialise_an_instance_segmenter(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the transfer of the region proposal and detection heads of an object detector to an instance segmenter
    whose anchors have other scales of the same counts: both heads start from exactly the weights of the source run,
    the other anchor scales being reported, while the mask heads keep their own weights.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the transfer.
    """
    _, source_detector = write_detector_run(run_folder=tmp_path / "detection_run")
    target_configuration, _, _ = get_detection_model_configuration(mask_head=True)
    target_configuration["Anchors"]["scales_and_ratios"]["5"]["scales"] = [32, 64]
    instance_segmenter = build_task_model(task="instance_segmentation", configuration=target_configuration, seed=1)
    mask_heads_before = copy.deepcopy(instance_segmenter.mask_heads_dictionary.state_dict())

    # Region proposal heads, then detection heads, each from the checkpoint of the detection run
    for section_name in ["RegionProposal", "Detector"]:
        load_pretrained_component(component=getattr(instance_segmenter, TRANSFERABLE_COMPONENTS[section_name]),
                                  section_name=section_name, device=torch.device("cpu"),
                                  model_configuration=point_to_source_run(configuration=target_configuration,
                                                                          section_name=section_name,
                                                                          source_folder=tmp_path / "detection_run"))
        source_state = getattr(source_detector, TRANSFERABLE_COMPONENTS[section_name]).state_dict()
        target_state = getattr(instance_segmenter, TRANSFERABLE_COMPONENTS[section_name]).state_dict()
        for tensor_name, tensor_value in source_state.items():
            assert torch.equal(input=target_state[tensor_name], other=tensor_value), tensor_name
    for tensor_name, tensor_value in instance_segmenter.mask_heads_dictionary.state_dict().items():
        assert torch.equal(input=tensor_value, other=mask_heads_before[tensor_name]), tensor_name

    console_output = capsys.readouterr().out
    assert "The Pretrained Region Proposal Heads Were Trained With The Anchors" in console_output
    assert "Region Proposal Heads Initialised From" in console_output
    assert "Detection Heads Initialised From" in console_output


@pytest.mark.parametrize(argnames="section_name", argvalues=["RegionProposal", "Detector"])
def test_detection_heads_of_another_batch_normalization_momentum_match(section_name: str) -> None:
    """
    Verifies that the region proposal and detection heads of runs differing only by the batch normalization momentum
    of their convolution blocks, a source run written before the momentum existed included, have matching weights,
    while another feature map normalization is still reported.

    Args:
        section_name (str): The section of the compared heads.
    """
    source_configuration, _, _ = get_detection_model_configuration()
    del source_configuration[section_name]["normalization"]["batch_normalization_momentum"]
    target_configuration = copy.deepcopy(source_configuration)
    target_configuration[section_name]["normalization"]["batch_normalization_momentum"] = 0.01
    assert compare_detection_heads(section_name=section_name, source_configuration=source_configuration,
                                   target_configuration=target_configuration) == []

    target_configuration[section_name]["normalization"]["feature_map_normalization"] = "layer"
    differences = compare_detection_heads(section_name=section_name, source_configuration=source_configuration,
                                          target_configuration=target_configuration)
    assert len(differences) == 1 and differences[0].startswith(f"'{section_name}.normalization' is ")


def add_detection_level(configuration: Dict[str, Any]) -> None:
    """
    Adds the eighth backbone block to the detection levels, with the anchors and heads of the seventh one.

    Args:
        configuration (Dict[str, Any]): The detector configuration, modified in place.
    """
    configuration["Backbone"]["detection_convolution_block_indices"] = [5, 6, 7, 8]
    for level_configurations in [configuration["Anchors"]["scales_and_ratios"],
                                 configuration["RegionProposal"]["head_configurations"],
                                 configuration["Detector"]["head_configurations"]]:
        level_configurations["8"] = copy.deepcopy(level_configurations["7"])


def deepen_region_proposal_head(configuration: Dict[str, Any]) -> None:
    """
    Gives the region proposal head of the seventh level a third convolution layer.

    Args:
        configuration (Dict[str, Any]): The detector configuration, modified in place.
    """
    configuration["RegionProposal"]["head_configurations"]["7"] = [3, 128]


def add_anchor_scale(configuration: Dict[str, Any]) -> None:
    """
    Gives the anchors of the fifth level a third scale, 9 anchors per position instead of 6.

    Args:
        configuration (Dict[str, Any]): The detector configuration, modified in place.
    """
    configuration["Anchors"]["scales_and_ratios"]["5"]["scales"] = [32, 64, 128]


def add_detection_head_residuals(configuration: Dict[str, Any]) -> None:
    """
    Adds residual connections to the convolutions of the detection heads.

    Args:
        configuration (Dict[str, Any]): The detector configuration, modified in place.
    """
    configuration["Detector"]["modules"]["residual_nets"] = True


def remove_foreground_class(configuration: Dict[str, Any]) -> None:
    """
    Removes the fifth foreground class, so that the detection heads classify one class less.

    Args:
        configuration (Dict[str, Any]): The detector configuration, modified in place.
    """
    del configuration["Data"]["label_dictionary"]["foreground_5"]


@pytest.mark.parametrize(argnames=("section_name", "modify_function", "expected_message"), argvalues=[
    ("RegionProposal", add_detection_level, "'Backbone.detection_convolution_block_indices' is [5, 6, 7] in the "
                                            "source run and [5, 6, 7, 8] in this run, the levels of the region "
                                            "proposal heads."),
    ("RegionProposal", deepen_region_proposal_head, "'RegionProposal.head_configurations' is"),
    ("RegionProposal", add_anchor_scale, "The levels have {'5': 6, '6': 6, '7': 6} anchors per position in the source "
                                         "run and {'5': 9, '6': 6, '7': 6} in this run"),
    ("Detector", add_detection_level, "the levels of the detection heads."),
    ("Detector", add_detection_head_residuals, "'Detector.modules' is"),
    ("Detector", remove_foreground_class, "'5.classifier.weight' has the shape [6, 128] in the source run and "
                                          "[5, 128] in this run.")])
def test_detection_heads_of_other_architectures_are_rejected(tmp_path: Path, section_name: str,
                                                             modify_function: Callable[[Dict[str, Any]], None],
                                                             expected_message: str) -> None:
    """
    Verifies that region proposal or detection heads are not transferred to heads of other levels, layers, anchor
    counts, modules or classes, the error naming the difference: the settings for the architectures, the tensors for
    the number of classes.

    Args:
        tmp_path (Path): The pytest temporary directory.
        section_name (str): The section of the transferred heads.
        modify_function (Callable[[Dict[str, Any]], None]): Function changing the configuration receiving the heads.
        expected_message (str): The fragment of the error message naming the difference.
    """
    source_configuration, _ = write_detector_run(run_folder=tmp_path / "detection_run")
    target_configuration = copy.deepcopy(source_configuration)
    modify_function(configuration=target_configuration)
    target_detector = build_task_model(task="object_detection", configuration=target_configuration, seed=1)

    with pytest.raises(expected_exception=ValueError) as error_information:
        load_pretrained_component(component=getattr(target_detector, TRANSFERABLE_COMPONENTS[section_name]),
                                  section_name=section_name, device=torch.device("cpu"),
                                  model_configuration=point_to_source_run(configuration=target_configuration,
                                                                          section_name=section_name,
                                                                          source_folder=tmp_path / "detection_run"))
    assert expected_message in str(error_information.value)
