"""
Tests of vision_training/models/model_transfer.py: backbones, temporal modules, heads, segmentation decoders and region
proposal, detection and mask heads of a saved run initialising other models and tasks, rejected architectures and
components (flattened heads, decoders and detection heads of other levels included), order-blind transformers moving
between maximum sequence lengths, the keypoint heads of another hidden width, source runs missing a setting (the
pre-activation flag of the backbone counting as off), frozen and fixed normalization parameters and backbones moving
between batch and sequence normalization. The runs adding classes after those of their source run are covered too: the
class layers of the classification, detection and mask heads and of the semantic segmentation grown to the new classes,
which keep the random initialisation of the new run, the classes of another order or fewer classes rejected, the source
runs without class names, and the averaged or trained weights read from the source checkpoint.
"""

import copy
import json
import pytest
import torch
import yaml

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from vision_training.data.data_record import CLASS_NAMES_RECORD_KEY, DATA_RECORD_FILE_NAME
from vision_training.models.classification.image_classifier import ImageClassifier
from vision_training.models.detection.two_stage_detector import TwoStageDetector
from vision_training.models.model_registry import TASK_MODEL_CLASSES
from vision_training.models.model_transfer import (NORMALIZATION_LAYER_TYPES, TRANSFERABLE_COMPONENTS,
                                                   compare_detection_heads, compare_keypoint_heads,
                                                   disable_pretrained_settings,
                                                   freeze_batch_normalization_statistics, freeze_component,
                                                   freeze_normalization_parameters, load_pretrained_component,
                                                   load_segmentation_class_layers)
from vision_training.models.task_model import TaskModel
from vision_training.trainers.checkpointing import save_checkpoint
from vision_training.trainers.weight_averaging import AVERAGED_MODEL_STATE_KEY, NUMBER_UPDATES_KEY
from tests.fixtures.configuration_fixtures import (get_classification_model_configuration,
                                                   get_detection_model_configuration,
                                                   get_keypoint_model_configuration,
                                                   get_segmentation_model_configuration,
                                                   get_sequence_comparison_model_configuration,
                                                   get_sequence_model_configuration)

# Pretrained settings of a component trained from random weights, and the normalization flag of the backbone
NO_TRANSFER = {"experiment_folder": None, "iteration": None, "freeze": False, "learning_rate_multiplier": 1.0}
NO_BACKBONE_TRANSFER = {**NO_TRANSFER, "freeze_normalization_parameters": False}

# Class names of the classification-like source runs of 3 classes, and of the classes a run adds after them
SOURCE_CLASS_NAMES = ["square", "circle", "triangle"]
NEW_CLASS_NAMES = ["rhombus", "ellipse"]


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
    return {"Data": {"image_settings": {"size": 512}, "label_dictionary": {1: 0, 2: 1}},
            "Backbone": backbone_configuration,
            "ClassificationHead": {"global_pooling": "average", "fully_connected": [],
                                   "fully_connected_normalization": "none", "batch_normalization_momentum": 0.1,
                                   "dropout_rate": 0.0, "output_mode": "multi_class", "label_smoothing": 0.0,
                                   "class_weighting": "none"},
            "Optimization": {"exponential_moving_average": {"activated": False, "decay": 0.9998}}}


def write_source_run(run_folder: Path, configuration: Dict[str, Any], model: torch.nn.Module,
                     averaged_model: Optional[torch.nn.Module] = None, class_names: Optional[List[str]] = None) -> None:
    """
    Writes a trained run as the trainers save it: its configuration, one checkpoint of its model, holding the moving
    average of its weights when one is given, and the record of its data reduced to its class names when they are
    given (a run written before the class names were recorded otherwise).

    Args:
        run_folder (Path): The run folder.
        configuration (Dict[str, Any]): The model configuration of the run.
        model (torch.nn.Module): The model whose weights are saved.
        averaged_model (Optional[torch.nn.Module]): The model holding the moving average of the weights, None for a
            run without average. Defaults to None.
        class_names (Optional[List[str]]): The class names of the run in class index order, None to write no data
            record. Defaults to None.
    """
    run_folder.mkdir(parents=True)
    (run_folder / "training_configuration.yaml").write_text(data=yaml.safe_dump(data=configuration))

    # Moving average stored in the checkpoint as the trainers save it
    exponential_moving_average_state = None
    if averaged_model is not None:
        exponential_moving_average_state = {AVERAGED_MODEL_STATE_KEY: averaged_model.state_dict(),
                                            NUMBER_UPDATES_KEY: 3}
    save_checkpoint(weights_directory=run_folder / "Weights", iteration=3, model=model,
                    optimizer=torch.optim.Adam(params=model.parameters()), metadata={"task": "classification"},
                    exponential_moving_average_state=exponential_moving_average_state)

    # Class names of the data record written by BaseTrainer.save_data_record
    if class_names is not None:
        (run_folder / DATA_RECORD_FILE_NAME).write_text(data=json.dumps(obj={CLASS_NAMES_RECORD_KEY: class_names}))


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


def test_source_run_written_before_the_pre_activation_flag(tmp_path: Path) -> None:
    """
    Verifies that the backbone of a run written before the pre-activation placement of the residual shortcuts, whose
    configuration has no 'residual_nets_v2', counts as using the original placement: it initialises a backbone with
    the original residual shortcuts exactly, and is refused by a backbone with pre-activation shortcuts, whose layers
    differ.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    torch.manual_seed(seed=0)
    detection_configuration, device, dtype = get_detection_model_configuration()
    detection_configuration = copy.deepcopy(detection_configuration)
    detection_configuration["Backbone"]["modules"]["residual_nets"] = True
    classifier_configuration = get_classifier_configuration(detection_configuration=detection_configuration)
    classifier = ImageClassifier(configuration=classifier_configuration, device=device, dtype=dtype)

    # Source run saved without the pre-activation flag
    old_configuration = copy.deepcopy(classifier_configuration)
    del old_configuration["Backbone"]["modules"]["residual_nets_v2"]
    write_source_run(run_folder=tmp_path / "classification_run", configuration=old_configuration, model=classifier)

    # Original residual shortcuts in the run receiving the backbone
    detector = TwoStageDetector(configuration=detection_configuration, device=device, dtype=dtype)
    load_pretrained_component(component=detector.backbone, section_name="Backbone", device=device,
                              model_configuration=point_to_source_run(configuration=detection_configuration,
                                                                      section_name="Backbone",
                                                                      source_folder=tmp_path / "classification_run"))
    for parameter_name, parameter_value in classifier.backbone.state_dict().items():
        assert torch.equal(input=detector.backbone.state_dict()[parameter_name], other=parameter_value)

    # Pre-activation shortcuts in the run receiving the backbone
    pre_activation_configuration = copy.deepcopy(detection_configuration)
    pre_activation_configuration["Backbone"]["modules"]["residual_nets_v2"] = True
    pre_activation_detector = TwoStageDetector(configuration=pre_activation_configuration, device=device, dtype=dtype)
    with pytest.raises(expected_exception=ValueError, match="'Backbone.modules' is"):
        load_pretrained_component(component=pre_activation_detector.backbone, section_name="Backbone", device=device,
                                  model_configuration=point_to_source_run(configuration=pre_activation_configuration,
                                                                          section_name="Backbone",
                                                                          source_folder=tmp_path /
                                                                          "classification_run"))


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


def lengthen_sequences(configuration: Dict[str, Any]) -> None:
    """
    Lengthens the maximum sequence length, and so the positional embedding of the transformer.

    Args:
        configuration (Dict[str, Any]): The configuration of the run receiving the component, modified in place.
    """
    configuration["Data"]["sequence_settings"]["maximum_sequence_length"] = 6


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
    ("transformer", lengthen_sequences, "TemporalModel",
     "'Data.sequence_settings.maximum_sequence_length' is 4 in the source run and 6 in this run, the size of the "
     "positional embedding of the transformer."),
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
    initialises the transformer of a run with longer sequences with exactly its weights.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration, _, _ = get_sequence_model_configuration(temporal_type="transformer")
    configuration["TemporalModel"]["positional_embedding"] = False
    source_model = build_task_model(task="sequence_classification", configuration=configuration, seed=0)
    write_source_run(run_folder=tmp_path / "source_run", configuration=configuration, model=source_model)
    target_configuration = copy.deepcopy(configuration)
    lengthen_sequences(configuration=target_configuration)
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
    value: the transfer stops with the missing setting and asks for its value in the configuration of the run.

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


def test_source_run_written_before_the_maximum_sequence_length_is_rejected(tmp_path: Path) -> None:
    """
    Verifies that a transformer run written before the maximum sequence length existed, whose sequence settings gave
    the length of its windows instead, is not read with a guessed size of positional embedding: the transfer stops
    and asks for the setting in the configuration of the run.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration, _ = write_sequence_classifier_run(run_folder=tmp_path / "source_run", temporal_type="transformer")
    old_configuration = copy.deepcopy(configuration)
    old_configuration["Data"]["sequence_settings"] = {"sequence_length": 4, "window_stride": 4}
    (tmp_path / "source_run" / "training_configuration.yaml").write_text(data=yaml.safe_dump(data=old_configuration))
    target_model = build_task_model(task="sequence_classification", configuration=configuration, seed=1)

    with pytest.raises(expected_exception=ValueError, match="maximum_sequence_length' is missing from the source run"):
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
    with the setting and the value to write in the configuration of the run.

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
                               "feature map, the input vector, has a size following the image size.")])
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


def write_detector_run(run_folder: Path, mask_head: bool = False,
                       class_names: Optional[List[str]] = None) -> Tuple[Dict[str, Any], TaskModel]:
    """
    Writes a trained object detection run with the small detector configuration of the tests, whose region proposal
    and detection heads initialise the heads of other detection runs, or an instance segmentation run with mask_head,
    whose mask heads initialise those of other instance segmentation runs.

    Args:
        run_folder (Path): The run folder.
        mask_head (bool): Whether to write an instance segmentation run with mask heads. Defaults to False.
        class_names (Optional[List[str]]): The class names of the run in class index order, the background first,
            None to write no data record. Defaults to None.

    Returns:
        Tuple[Dict[str, Any], TaskModel]: The configuration and the model of the run.
    """
    configuration, _, _ = get_detection_model_configuration(mask_head=mask_head)
    task = "instance_segmentation" if mask_head else "object_detection"
    source_detector = build_task_model(task=task, configuration=configuration, seed=0)
    write_source_run(run_folder=run_folder, configuration=configuration, model=source_detector,
                     class_names=class_names)
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
    target_configuration["DetectionLevels"]["5"]["anchors"]["scales"] = [32, 64]
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


@pytest.mark.parametrize(argnames=("higher_resolution", "hidden_width", "normalization_reported"),
                         argvalues=[(False, 0, False), (False, 16, True), (True, 0, True)])
def test_keypoint_heads_compare_their_hidden_width(higher_resolution: bool, hidden_width: int,
                                                   normalization_reported: bool) -> None:
    """
    Verifies that the keypoint heads of two runs match when their settings are equal, that another width of the 3x3
    hidden convolution of the prediction heads is reported alone, and that another feature map normalization of the
    high resolution network is reported only when the head has layers it normalizes: the hidden convolutions or the
    refinement of the higher resolution branch.

    Args:
        higher_resolution (bool): Whether both runs have the higher resolution branch.
        hidden_width (int): The width of the hidden convolution of both runs, 0 for none.
        normalization_reported (bool): Whether another normalization changes the weights of the head.
    """
    source_configuration, _, _ = get_keypoint_model_configuration(higher_resolution=higher_resolution,
                                                                  hidden_width=hidden_width)
    target_configuration = copy.deepcopy(source_configuration)
    assert compare_keypoint_heads(source_configuration=source_configuration,
                                  target_configuration=target_configuration) == []

    # Another hidden width changes the layers of the heads
    target_configuration["KeypointHead"]["hidden_width"] = hidden_width + 32
    assert compare_keypoint_heads(source_configuration=source_configuration,
                                  target_configuration=target_configuration) == [
        f"'KeypointHead.hidden_width' is {hidden_width} in the source run and {hidden_width + 32} in this run."]

    # Another normalization of the high resolution network only matters to the normalized layers of the head
    target_configuration["KeypointHead"]["hidden_width"] = hidden_width
    target_configuration["HRNet"]["normalization"]["feature_map_normalization"] = "layer"
    differences = compare_keypoint_heads(source_configuration=source_configuration,
                                         target_configuration=target_configuration)
    assert [difference.split(" is ")[0] for difference in differences] == (
        ["'HRNet.normalization'"] if normalization_reported else [])


def add_detection_level(configuration: Dict[str, Any]) -> None:
    """
    Adds the eighth backbone block to the detection levels, with the anchors and heads of the seventh one.

    Args:
        configuration (Dict[str, Any]): The detector configuration, modified in place.
    """
    configuration["DetectionLevels"]["8"] = copy.deepcopy(configuration["DetectionLevels"]["7"])


def deepen_region_proposal_head(configuration: Dict[str, Any]) -> None:
    """
    Gives the region proposal head of the seventh level a third convolution layer.

    Args:
        configuration (Dict[str, Any]): The detector configuration, modified in place.
    """
    configuration["DetectionLevels"]["7"]["region_proposal"]["convolutions"] = [3, 128]


def add_anchor_scale(configuration: Dict[str, Any]) -> None:
    """
    Gives the anchors of the fifth level a third scale, 9 anchors per position instead of 6.

    Args:
        configuration (Dict[str, Any]): The detector configuration, modified in place.
    """
    configuration["DetectionLevels"]["5"]["anchors"]["scales"] = [32, 64, 128]


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
    ("RegionProposal", add_detection_level, "The detection levels are ['5', '6', '7'] in the source run and "
                                            "['5', '6', '7', '8'] in this run, the levels of the region "
                                            "proposal heads."),
    ("RegionProposal", deepen_region_proposal_head, "'DetectionLevels.7.region_proposal' is"),
    ("RegionProposal", add_anchor_scale, "The levels have {'5': 6, '6': 6, '7': 6} anchors per position in the source "
                                         "run and {'5': 9, '6': 6, '7': 6} in this run"),
    ("Detector", add_detection_level, "the levels of the detection heads."),
    ("Detector", add_detection_head_residuals, "'Detector.modules' is"),
    ("Detector", remove_foreground_class, "The source run predicts 6 classes and this run 5: a run can only add "
                                          "classes after those of its source run.")])
def test_detection_heads_of_other_architectures_are_rejected(tmp_path: Path, section_name: str,
                                                             modify_function: Callable[[Dict[str, Any]], None],
                                                             expected_message: str) -> None:
    """
    Verifies that region proposal or detection heads are not transferred to heads of other levels, layers, anchor
    counts, modules or fewer classes, the error naming the difference: the settings for the architectures, the class
    layers for the number of classes.

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


def get_class_configuration(task: str, number_classes: int) -> Dict[str, Any]:
    """
    Builds the small configuration of an image classifier or a sequence labeller (many-to-many) predicting the given
    number of classes, for the runs adding classes after those of their source run.

    Args:
        task (str): "classification" or "sequence_labelling".
        number_classes (int): The number of classes predicted by the head.

    Returns:
        Dict[str, Any]: The model configuration.
    """
    if task == "classification":
        configuration, _, _ = get_classification_model_configuration(output_mode="multi_class")
    else:
        configuration, _, _ = get_sequence_model_configuration(temporal_type="gru")
    configuration["Data"]["label_dictionary"] = {class_index + 1: class_index for class_index in range(number_classes)}
    return configuration


def list_detection_class_names(configuration: Dict[str, Any]) -> List[str]:
    """
    Lists the class names of a detector configuration in class index order, the background first, as
    DetectionTrainer.list_class_names gives them.

    Args:
        configuration (Dict[str, Any]): The detector configuration.

    Returns:
        List[str]: The class names.
    """
    label_dictionary = configuration["Data"]["label_dictionary"]
    return sorted(label_dictionary, key=lambda class_name: label_dictionary[class_name])


@pytest.mark.parametrize(argnames="task", argvalues=["classification", "sequence_labelling"])
def test_classification_head_adds_classes_after_the_source_classes(tmp_path: Path, task: str,
                                                                   capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the classification head of a run of 3 classes initialises the head of a run adding 2 classes after
    them, for an image classifier and a sequence labeller (many-to-many): the hidden layers and the classifier rows of
    the source classes are copied exactly, and the new classes keep the random weights and biases the head of the new
    run was built with.

    Args:
        tmp_path (Path): The pytest temporary directory.
        task (str): The task of both runs.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the transfer.
    """
    source_configuration = get_class_configuration(task=task, number_classes=3)
    source_model = build_task_model(task=task, configuration=source_configuration, seed=0)
    write_source_run(run_folder=tmp_path / "source_run", configuration=source_configuration, model=source_model,
                     class_names=SOURCE_CLASS_NAMES)
    target_configuration = get_class_configuration(task=task, number_classes=5)
    target_model = build_task_model(task=task, configuration=target_configuration, seed=1)
    initial_target_state = {tensor_name: tensor_value.clone() for tensor_name, tensor_value in
                            target_model.classification_head.state_dict().items()}
    load_pretrained_component(component=target_model.classification_head, section_name="ClassificationHead",
                              model_configuration=point_to_source_run(configuration=target_configuration,
                                                                      section_name="ClassificationHead",
                                                                      source_folder=tmp_path / "source_run"),
                              device=torch.device("cpu"), class_names=SOURCE_CLASS_NAMES + NEW_CLASS_NAMES)

    # Hidden layers and source classes copied exactly
    source_state = source_model.classification_head.state_dict()
    target_state = target_model.classification_head.state_dict()
    for tensor_name, tensor_value in source_state.items():
        if not tensor_name.startswith("classifier."):
            assert torch.equal(input=target_state[tensor_name], other=tensor_value), tensor_name
    assert torch.equal(input=target_state["classifier.weight"][:3], other=source_state["classifier.weight"])
    assert torch.equal(input=target_state["classifier.bias"][:3], other=source_state["classifier.bias"])

    # New classes keeping the random initialisation of the new run
    for tensor_name in ["classifier.weight", "classifier.bias"]:
        assert torch.equal(input=target_state[tensor_name][3:], other=initial_target_state[tensor_name][3:])
    assert torch.count_nonzero(input=target_state["classifier.weight"][3:]) > 0
    assert ("Classification Head : 3 Classes Of The Source Run Kept, 2 New Classes Added After Them, Randomly "
            "Initialised.") in capsys.readouterr().out


@pytest.mark.parametrize(argnames=("source_class_names", "target_class_names", "expected_message"), argvalues=[
    (SOURCE_CLASS_NAMES, ["square", "triangle", "circle", "rhombus"],
     "class 1 is 'circle' in the source run and 'triangle' in this run, class 2 is 'triangle' in the source run and "
     "'circle' in this run."),
    (SOURCE_CLASS_NAMES, ["square", "circle"], "The source run predicts 3 classes and this run 2: a run can only add "
                                               "classes after those of its source run."),
    (SOURCE_CLASS_NAMES + ["rhombus"], SOURCE_CLASS_NAMES + NEW_CLASS_NAMES,
     "The data record of the source run names 4 classes and its checkpoint predicts 3 classes.")])
def test_classes_of_another_order_or_fewer_classes_are_rejected(tmp_path: Path, source_class_names: List[str],
                                                                target_class_names: List[str],
                                                                expected_message: str) -> None:
    """
    Verifies that a classification head is not transferred to a run whose first classes are not those of the source
    run in the same order, to a run of fewer classes, or from a source run whose data record names other classes than
    its checkpoint predicts, the error naming the difference.

    Args:
        tmp_path (Path): The pytest temporary directory.
        source_class_names (List[str]): The class names recorded in the data record of the source run of 3 classes.
        target_class_names (List[str]): The class names of the run receiving the head.
        expected_message (str): The fragment of the error message naming the difference.
    """
    source_configuration = get_class_configuration(task="classification", number_classes=3)
    write_source_run(run_folder=tmp_path / "source_run", configuration=source_configuration,
                     model=build_task_model(task="classification", configuration=source_configuration, seed=0),
                     class_names=source_class_names)
    target_configuration = get_class_configuration(task="classification", number_classes=len(target_class_names))
    target_model = build_task_model(task="classification", configuration=target_configuration, seed=1)

    with pytest.raises(expected_exception=ValueError) as error_information:
        load_pretrained_component(component=target_model.classification_head, section_name="ClassificationHead",
                                  model_configuration=point_to_source_run(configuration=target_configuration,
                                                                          section_name="ClassificationHead",
                                                                          source_folder=tmp_path / "source_run"),
                                  device=torch.device("cpu"), class_names=target_class_names)
    assert "The Classification Head Of The Source Run" in str(error_information.value)
    assert expected_message in str(error_information.value)


def test_source_run_without_class_names_keeps_its_class_order(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a source run written before the class names were recorded (no data record) still initialises a run
    adding classes after its own, its classes being taken as the first classes of the new run with a warning.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the transfer.
    """
    source_configuration = get_class_configuration(task="classification", number_classes=3)
    source_model = build_task_model(task="classification", configuration=source_configuration, seed=0)
    write_source_run(run_folder=tmp_path / "source_run", configuration=source_configuration, model=source_model)
    target_configuration = get_class_configuration(task="classification", number_classes=5)
    target_model = build_task_model(task="classification", configuration=target_configuration, seed=1)
    load_pretrained_component(component=target_model.classification_head, section_name="ClassificationHead",
                              model_configuration=point_to_source_run(configuration=target_configuration,
                                                                      section_name="ClassificationHead",
                                                                      source_folder=tmp_path / "source_run"),
                              device=torch.device("cpu"), class_names=SOURCE_CLASS_NAMES + NEW_CLASS_NAMES)

    assert torch.equal(input=target_model.classification_head.classifier.weight[:3],
                       other=source_model.classification_head.classifier.weight)
    assert f"The Source Run Has No Class Names In Its {DATA_RECORD_FILE_NAME}" in capsys.readouterr().out


def test_detection_heads_add_classes_after_the_source_classes(tmp_path: Path) -> None:
    """
    Verifies that the detection heads of a detector of 5 foreground classes initialise the heads of a detector adding
    2 classes after them: on every level, the layers without classes are copied exactly, the classifier keeps the rows
    of the background and of the 5 source classes and the box regressor the 4 rows of every source class, while the
    rows of the 2 new classes keep the random weights and biases the heads of the new detector were built with.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    source_configuration, _, _ = get_detection_model_configuration()
    source_configuration, source_detector = write_detector_run(
        run_folder=tmp_path / "detection_run",
        class_names=list_detection_class_names(configuration=source_configuration))
    target_configuration = copy.deepcopy(source_configuration)
    target_configuration["Data"]["label_dictionary"].update({"foreground_6": 6, "foreground_7": 7})
    target_detector = build_task_model(task="object_detection", configuration=target_configuration, seed=1)
    initial_target_state = {tensor_name: tensor_value.clone() for tensor_name, tensor_value in
                            target_detector.detector_heads_dictionary.state_dict().items()}
    load_pretrained_component(component=target_detector.detector_heads_dictionary, section_name="Detector",
                              model_configuration=point_to_source_run(configuration=target_configuration,
                                                                      section_name="Detector",
                                                                      source_folder=tmp_path / "detection_run"),
                              device=torch.device("cpu"),
                              class_names=list_detection_class_names(configuration=target_configuration))

    source_state = source_detector.detector_heads_dictionary.state_dict()
    target_state = target_detector.detector_heads_dictionary.state_dict()
    for level in ["5", "6", "7"]:
        # Layers without classes copied exactly
        for tensor_name, tensor_value in source_state.items():
            if tensor_name.startswith(f"{level}.") and "classifier" not in tensor_name and \
                    "bounding_box_regressor" not in tensor_name:
                assert torch.equal(input=target_state[tensor_name], other=tensor_value), tensor_name

        # Rows of the background and source classes copied, rows of the new classes left as the new run built them,
        # 1 row per class for the classifier and 4 for the box regressor
        for layer_name, number_source_rows in [("classifier", 6), ("bounding_box_regressor", 24)]:
            for parameter_name in ["weight", "bias"]:
                tensor_name = f"{level}.{layer_name}.{parameter_name}"
                assert torch.equal(input=target_state[tensor_name][:number_source_rows],
                                   other=source_state[tensor_name]), tensor_name
                assert torch.equal(input=target_state[tensor_name][number_source_rows:],
                                   other=initial_target_state[tensor_name][number_source_rows:]), tensor_name
            assert torch.count_nonzero(input=target_state[f"{level}.{layer_name}.weight"][number_source_rows:]) > 0


def test_mask_heads_initialise_an_instance_segmenter_of_more_classes(tmp_path: Path,
                                                                     capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the transfer of the mask heads between two instance segmenters, the run receiving them adding 2 classes:
    the convolutions and upsamplings of every level are copied exactly, the mask predictor keeps the channels of the
    background and of the source classes, and the channels of the new classes keep the random weights the mask heads
    of the new run were built with.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the transfer.
    """
    source_configuration, _, _ = get_detection_model_configuration(mask_head=True)
    source_configuration, source_segmenter = write_detector_run(
        run_folder=tmp_path / "instance_run", mask_head=True,
        class_names=list_detection_class_names(configuration=source_configuration))
    target_configuration = copy.deepcopy(source_configuration)
    target_configuration["Data"]["label_dictionary"].update({"foreground_6": 6, "foreground_7": 7})
    target_segmenter = build_task_model(task="instance_segmentation", configuration=target_configuration, seed=1)
    initial_target_state = {tensor_name: tensor_value.clone() for tensor_name, tensor_value in
                            target_segmenter.mask_heads_dictionary.state_dict().items()}
    load_pretrained_component(component=target_segmenter.mask_heads_dictionary, section_name="MaskHead",
                              model_configuration=point_to_source_run(configuration=target_configuration,
                                                                      section_name="MaskHead",
                                                                      source_folder=tmp_path / "instance_run"),
                              device=torch.device("cpu"),
                              class_names=list_detection_class_names(configuration=target_configuration))

    source_state = source_segmenter.mask_heads_dictionary.state_dict()
    target_state = target_segmenter.mask_heads_dictionary.state_dict()
    for tensor_name, tensor_value in source_state.items():
        # Layers without classes copied exactly
        if "mask_predictor" not in tensor_name:
            assert torch.equal(input=target_state[tensor_name], other=tensor_value), tensor_name
            continue

        # Mask predictor: background and source classes copied, new classes as the new run built them
        assert torch.equal(input=target_state[tensor_name][:6], other=tensor_value), tensor_name
        assert torch.equal(input=target_state[tensor_name][6:], other=initial_target_state[tensor_name][6:])
    assert "Mask Heads Initialised From The Trained Weights Of" in capsys.readouterr().out


def test_mask_heads_of_another_mask_resolution_are_rejected(tmp_path: Path) -> None:
    """
    Verifies that mask heads are not transferred to mask heads of another mask resolution, whose upsampling layers
    differ, the error naming the setting.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    source_configuration, _ = write_detector_run(run_folder=tmp_path / "instance_run", mask_head=True)
    target_configuration = copy.deepcopy(source_configuration)
    target_configuration["MaskHead"]["mask_resolution"] = 28
    target_segmenter = build_task_model(task="instance_segmentation", configuration=target_configuration, seed=1)

    with pytest.raises(expected_exception=ValueError) as error_information:
        load_pretrained_component(component=target_segmenter.mask_heads_dictionary, section_name="MaskHead",
                                  model_configuration=point_to_source_run(configuration=target_configuration,
                                                                          section_name="MaskHead",
                                                                          source_folder=tmp_path / "instance_run"),
                                  device=torch.device("cpu"))
    assert "The Mask Heads Of The Source Run" in str(error_information.value)
    assert "'MaskHead.mask_resolution' is 14 in the source run and 28 in this run." in str(error_information.value)


def test_segmentation_class_layers_follow_the_decoder(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the loading of the class layers of a semantic segmenter with its decoder
    ('SegmentationDecoder.pretrained.load_class_layers'), the run receiving them adding 2 classes and a deep
    supervision head: the segmentation head and the deep supervision head of level 3, supervised by both runs, keep the
    channels of the background and of the source classes, the new classes keeping the random weights the new run was
    built with, while the deep supervision head of level 2, supervised by this run only, keeps its random weights,
    which is reported.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the transfer.
    """
    source_configuration, _, _ = get_segmentation_model_configuration(upsampling="transposed")
    source_configuration["SegmentationDecoder"]["deep_supervision_weights"] = {"3": 0.2}
    source_segmenter = build_task_model(task="semantic_segmentation", configuration=source_configuration, seed=0)
    write_source_run(run_folder=tmp_path / "segmentation_run", configuration=source_configuration,
                     model=source_segmenter, class_names=["background", "class_1", "class_2"])

    # Two more classes and a deep supervision head on level 2 in the run receiving the decoder and its class layers
    target_configuration = copy.deepcopy(source_configuration)
    target_configuration["Data"]["label_dictionary"].update({3: 3, 4: 4})
    target_configuration["SegmentationDecoder"]["deep_supervision_weights"] = {"3": 0.2, "2": 0.3}
    target_configuration = point_to_source_run(configuration=target_configuration, section_name="SegmentationDecoder",
                                               source_folder=tmp_path / "segmentation_run")
    target_configuration["SegmentationDecoder"]["pretrained"]["load_class_layers"] = True
    target_segmenter = build_task_model(task="semantic_segmentation", configuration=target_configuration, seed=1)
    initial_target_state = copy.deepcopy(target_segmenter.state_dict())
    level_2_state_before = copy.deepcopy(target_segmenter.deep_supervision_heads["2"].state_dict())
    target_class_names = ["background", "class_1", "class_2", "class_3", "class_4"]
    load_pretrained_component(component=target_segmenter.segmentation_decoder, section_name="SegmentationDecoder",
                              model_configuration=target_configuration, device=torch.device("cpu"),
                              class_names=target_class_names)
    load_segmentation_class_layers(segmenter=target_segmenter, model_configuration=target_configuration,
                                   device=torch.device("cpu"), class_names=target_class_names)

    # Class layers of both runs: background and source classes copied, new classes as the new run built them
    source_state = source_segmenter.state_dict()
    target_state = target_segmenter.state_dict()
    for module_name in ["segmentation_head.1", "deep_supervision_heads.3"]:
        for tensor_name in [f"{module_name}.weight", f"{module_name}.bias"]:
            assert torch.equal(input=target_state[tensor_name][:3], other=source_state[tensor_name])
            assert torch.equal(input=target_state[tensor_name][3:], other=initial_target_state[tensor_name][3:])

    # Deep supervision head of this run only kept random
    for tensor_name, tensor_value in target_segmenter.deep_supervision_heads["2"].state_dict().items():
        assert torch.equal(input=tensor_value, other=level_2_state_before[tensor_name]), tensor_name
    console_output = capsys.readouterr().out
    assert "The Source Run Has No Deep Supervision Head On The Levels ['2']" in console_output
    assert "Segmentation Class Layers Initialised From The Trained Weights Of" in console_output


@pytest.mark.parametrize(argnames=("source_has_average", "target_averages", "expected_weights"),
                         argvalues=[(True, True, "Averaged"), (True, False, "Trained"), (False, True, "Trained")])
def test_averaged_weights_initialise_a_run_keeping_a_moving_average(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                                    source_has_average: bool, target_averages: bool,
                                                                    expected_weights: str) -> None:
    """
    Verifies which weights of a source checkpoint initialise a component: the moving average of the weights when the
    checkpoint holds one and the new run keeps one too ('Optimization.exponential_moving_average.activated'), the
    trained weights otherwise, the transfer printing which.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the transfer.
        source_has_average (bool): Whether the source checkpoint holds a moving average of the weights.
        target_averages (bool): Whether the run receiving the backbone keeps a moving average.
        expected_weights (str): The weights expected in the backbone, "Averaged" or "Trained".
    """
    configuration = get_class_configuration(task="classification", number_classes=3)
    source_model = build_task_model(task="classification", configuration=configuration, seed=0)
    averaged_model = build_task_model(task="classification", configuration=configuration, seed=2) \
        if source_has_average else None
    write_source_run(run_folder=tmp_path / "source_run", configuration=configuration, model=source_model,
                     averaged_model=averaged_model)

    target_configuration = copy.deepcopy(configuration)
    target_configuration["Optimization"]["exponential_moving_average"]["activated"] = target_averages
    target_model = build_task_model(task="classification", configuration=target_configuration, seed=1)
    load_pretrained_component(component=target_model.backbone, section_name="Backbone", device=torch.device("cpu"),
                              model_configuration=point_to_source_run(configuration=target_configuration,
                                                                      section_name="Backbone",
                                                                      source_folder=tmp_path / "source_run"))

    expected_model = averaged_model if expected_weights == "Averaged" else source_model
    for tensor_name, tensor_value in expected_model.backbone.state_dict().items():
        assert torch.equal(input=target_model.backbone.state_dict()[tensor_name], other=tensor_value), tensor_name
    assert f"Backbone Initialised From The {expected_weights} Weights Of" in capsys.readouterr().out


def test_mask_heads_and_class_layers_transfers_are_disabled() -> None:
    """
    Verifies that turning off the transfers of the instance and semantic segmentation configurations also removes the
    source run and the freezing of the mask heads and the loading of the class layers of the segmentation decoder.
    """
    instance_configuration, _, _ = get_detection_model_configuration(mask_head=True)
    for section_name in ["Backbone", "RegionProposal", "Detector", "MaskHead"]:
        no_transfer = NO_BACKBONE_TRANSFER if section_name == "Backbone" else NO_TRANSFER
        instance_configuration[section_name]["pretrained"] = {**no_transfer, "experiment_folder": "/runs/source"}
    instance_configuration["MaskHead"]["pretrained"]["freeze"] = True
    segmentation_configuration, _, _ = get_segmentation_model_configuration(upsampling="transposed")
    segmentation_configuration["Backbone"]["pretrained"] = dict(NO_BACKBONE_TRANSFER)
    segmentation_configuration["SegmentationDecoder"]["pretrained"] = {**NO_TRANSFER, "load_class_layers": True,
                                                                       "experiment_folder": "/runs/source"}

    disable_pretrained_settings(model_configuration=instance_configuration)
    disable_pretrained_settings(model_configuration=segmentation_configuration)

    assert instance_configuration["MaskHead"]["pretrained"] == NO_TRANSFER
    assert segmentation_configuration["SegmentationDecoder"]["pretrained"] == {**NO_TRANSFER,
                                                                               "load_class_layers": False}
