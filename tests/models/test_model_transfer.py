"""
Tests of vision_training/models/model_transfer.py: backbones, temporal modules, heads, segmentation decoders, necks and
region proposal, detection and mask heads of a saved run initialising other models and tasks, backbones and necks of
other architectures rejected, source runs still holding the fixed settings, frozen backbones and the transfers turned
off. The heads of the detection tasks only move between runs of the same neck type and channels, with a warning when
their neck does not come from their checkpoint. The runs adding
classes after those of their source run are covered too: the class layers of the classification, detection and mask
heads and of the semantic segmentation grown to the new classes, which keep the random initialisation of the new run,
the source runs without class names, and the runs holding the source classes at other indices, whose rows land at the
index of their name, a run lacking a source class being refused. The keypoint head loaded with its keypoint channel
layers only moves between runs predicting the same keypoints on the same channels, read from the record of the data of
the source run, and the mask heads between runs sharing them by the levels or not alike. The classification,
detection, mask and keypoint heads loaded without their class layers initialise runs of unrelated classes or keypoints,
only their other layers being transferred, frozen and fine-tuned, and the class layer flags are turned off with the
transfers.
"""

import copy
import json
import pytest
import torch
import yaml

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from vision_training.configuration_processing.common_checks import (CLASS_LAYER_OPTION_SECTIONS,
                                                                   PRETRAINED_CLASS_LAYER_KEYS)
from vision_training.data.data_record import CLASSES_RECORD_KEY, DATA_RECORD_FILE_NAME, KEYPOINTS_RECORD_KEY
from vision_training.models.backbones.backbone import INCEPTION_STARTING_BLOCK
from vision_training.models.classification.image_classifier import ImageClassifier
from vision_training.models.detection.two_stage_detector import TwoStageDetector
from vision_training.models.model_registry import TASK_MODEL_CLASSES
from vision_training.models.model_transfer import (NORMALIZATION_LAYER_TYPES, TRANSFERABLE_COMPONENTS,
                                                   compare_class_names, compare_component_configurations,
                                                   count_loaded_backbone_blocks,
                                                   count_shared_backbone_blocks, disable_pretrained_settings,
                                                   find_backbone_block_index, freeze_batch_normalization_statistics,
                                                   freeze_component, freeze_normalization_parameters,
                                                   load_pretrained_component, load_segmentation_class_layers,
                                                   select_transferred_part)
from vision_training.models.task_model import TaskModel
from vision_training.trainers.checkpointing import save_checkpoint
from vision_training.trainers.weight_averaging import AVERAGED_MODEL_STATE_KEY, NUMBER_UPDATES_KEY
from tests.fixtures.configuration_fixtures import (get_classification_model_configuration,
                                                   get_detection_model_configuration,
                                                   get_keypoint_model_configuration,
                                                   get_segmentation_model_configuration,
                                                   get_sequence_model_configuration)

# Pretrained settings of a component trained from random weights, with the normalization flag of the backbone, the
# class layer flag of the classification head and the class layer flag of the detection, mask and keypoint heads and of
# the segmentation decoder
NO_TRANSFER = {"experiment_folder": None, "iteration": None, "freeze": False, "learning_rate_multiplier": 1.0}
NO_BACKBONE_TRANSFER = {**NO_TRANSFER, "freeze_normalization_parameters": False}
NO_HEAD_TRANSFER = {**NO_TRANSFER, "class_layer": False}
NO_CLASS_LAYERS_TRANSFER = {**NO_TRANSFER, "class_layers": False}

# Blocks of the shorter source backbone of the transfers into a deeper backbone, the first 7 of the 9 blocks of the
# detection fixture
SHORTER_SOURCE_BLOCKS = 7

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
                                   "fully_connected_normalization": "none", "dropout_rate": 0.0,
                                   "output_mode": "multi_class", "label_smoothing": 0.0, "class_weighting": "none",
                                   "decorrelation": {"activated": False, "weight": 0.001, "hidden_layer": 1}},
            "Optimization": {"exponential_moving_average": {"activated": False, "decay": 0.9998}}}


def write_source_run(run_folder: Path, configuration: Dict[str, Any], model: torch.nn.Module,
                     averaged_model: Optional[torch.nn.Module] = None, class_names: Optional[List[str]] = None,
                     class_keypoints: Optional[Dict[str, List[str]]] = None) -> None:
    """
    Writes a trained run as the trainers save it: its configuration, one checkpoint of its model, holding the moving
    average of its weights when one is given, and the record of its data reduced to its class names when they are
    given (a run written before the class names were recorded otherwise), with the keypoint names of the classes of a
    keypoint detection run.

    Args:
        run_folder (Path): The run folder.
        configuration (Dict[str, Any]): The model configuration of the run.
        model (torch.nn.Module): The model whose weights are saved.
        averaged_model (Optional[torch.nn.Module]): The model holding the moving average of the weights, None for a
            run without average. Defaults to None.
        class_names (Optional[List[str]]): The class names of the run in class index order, None to write no data
            record. Defaults to None.
        class_keypoints (Optional[Dict[str, List[str]]]): The keypoint names of the classes that define keypoints,
            keyed by class name, None for a record without keypoints. Defaults to None.
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

    # Classes of the data record written by BaseTrainer.save_data_record
    if class_names is not None:
        class_records = [{"class_index": class_index, "class_name": class_name, "category_ids": [class_index]}
                         for class_index, class_name in enumerate(class_names)]
        for class_record in class_records:
            if class_keypoints is not None and class_record["class_name"] in class_keypoints:
                class_record[KEYPOINTS_RECORD_KEY] = class_keypoints[class_record["class_name"]]
        (run_folder / DATA_RECORD_FILE_NAME).write_text(data=json.dumps(obj={CLASSES_RECORD_KEY: class_records}))


def point_to_source_run(configuration: Dict[str, Any], section_name: str, source_folder: Path,
                        class_layers: bool = True) -> Dict[str, Any]:
    """
    Copies a configuration with the pretrained settings of one section pointing to a source run, the component of
    that section being then initialised by the one of the source run, a classification, detection, mask or keypoint
    head with or without its class layers, a segmentation decoder without its class layers.

    Args:
        configuration (Dict[str, Any]): The configuration of the run receiving the component.
        section_name (str): The section of the transferred component.
        source_folder (Path): The run folder of the source component.
        class_layers (bool): The class layer flag of a transferred head of CLASS_LAYER_OPTION_SECTIONS ('class_layer'
            or 'class_layers'), ignored for the other sections. Defaults to True, the head being loaded with its class
            layers.

    Returns:
        Dict[str, Any]: The configuration with the pretrained settings of the section.
    """
    target_configuration = copy.deepcopy(configuration)
    no_transfer = NO_BACKBONE_TRANSFER if section_name == "Backbone" else NO_TRANSFER
    target_configuration[section_name]["pretrained"] = {**no_transfer, "experiment_folder": str(source_folder)}

    # Class layer flag of the sections holding one, False for the segmentation decoder, whose class layers lie outside
    if section_name in PRETRAINED_CLASS_LAYER_KEYS:
        target_configuration[section_name]["pretrained"][PRETRAINED_CLASS_LAYER_KEYS[section_name]] = \
            class_layers and section_name in CLASS_LAYER_OPTION_SECTIONS
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
    configuration, source_model = write_sequence_classifier_run(run_folder=tmp_path / "source_run", temporal_type="gru")
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


def test_components_of_a_run_holding_the_fixed_settings_are_transferred(tmp_path: Path) -> None:
    """
    Verifies that a source run written before the backbone and head settings were fixed, still holding their batch
    normalization momentums and the inception starting block, initialises a run with exactly the weights and batch
    statistics of its backbone and normalized classification head, the momentums changing no tensor, while a source
    run whose inception modules started at another block than the fixed one is refused.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration, _, _ = get_sequence_model_configuration(temporal_type="gru")
    configuration["ClassificationHead"]["fully_connected_normalization"] = "batch"
    source_model = build_task_model(task="sequence_classification", configuration=configuration, seed=0)

    # Source run written before the settings were fixed, of other momentums and of the fixed inception starting block
    old_configuration = copy.deepcopy(configuration)
    old_configuration["Backbone"]["normalization"]["batch_normalization_momentum"] = 0.01
    old_configuration["Backbone"]["modules"]["inception_starting_block"] = INCEPTION_STARTING_BLOCK
    old_configuration["ClassificationHead"]["batch_normalization_momentum"] = 0.01
    write_source_run(run_folder=tmp_path / "source_run", configuration=old_configuration, model=source_model)

    # Run receiving the backbone and the head of the source run, with exactly their tensors
    target_model = build_task_model(task="sequence_classification", configuration=configuration, seed=1)
    for section_name in ["Backbone", "ClassificationHead"]:
        load_pretrained_component(component=getattr(target_model, TRANSFERABLE_COMPONENTS[section_name]),
                                  section_name=section_name, device=torch.device("cpu"),
                                  model_configuration=point_to_source_run(configuration=configuration,
                                                                          section_name=section_name,
                                                                          source_folder=tmp_path / "source_run"))
        source_state = getattr(source_model, TRANSFERABLE_COMPONENTS[section_name]).state_dict()
        target_state = getattr(target_model, TRANSFERABLE_COMPONENTS[section_name]).state_dict()
        for tensor_name, tensor_value in source_state.items():
            assert torch.equal(input=target_state[tensor_name], other=tensor_value), tensor_name

    # A source run whose inception modules started at another block has other layers
    old_configuration["Backbone"]["modules"]["inception_starting_block"] = INCEPTION_STARTING_BLOCK + 1
    write_source_run(run_folder=tmp_path / "other_source_run", configuration=old_configuration, model=source_model)
    with pytest.raises(expected_exception=ValueError, match="'Backbone.modules' is"):
        load_pretrained_component(component=target_model.backbone, section_name="Backbone", device=torch.device("cpu"),
                                  model_configuration=point_to_source_run(configuration=configuration,
                                                                          section_name="Backbone",
                                                                          source_folder=tmp_path / "other_source_run"))


def build_deeper_backbone_configurations() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Builds the configurations of a classifier whose backbone of 9 residual blocks with batch normalization is deeper
    than the backbone of its source classifier, made of its first SHORTER_SOURCE_BLOCKS blocks, the max pooling after
    the last block differing (it has no weights).

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: The configuration of the source classifier, and the configuration of the
            deeper classifier receiving its backbone.
    """
    detection_configuration, _, _ = get_detection_model_configuration()
    target_configuration = get_classifier_configuration(detection_configuration=detection_configuration)
    target_configuration["Backbone"]["normalization"]["feature_map_normalization"] = "batch"
    target_configuration["Backbone"]["modules"]["residual_nets"] = True
    source_configuration = copy.deepcopy(target_configuration)
    source_configuration["Backbone"]["convolutions"] = dict(
        list(target_configuration["Backbone"]["convolutions"].items())[:SHORTER_SOURCE_BLOCKS])
    source_configuration["Backbone"]["last_max_pooling"] = not target_configuration["Backbone"]["last_max_pooling"]
    return source_configuration, target_configuration


def test_shorter_backbone_initialises_the_first_blocks_of_a_deeper_backbone(tmp_path: Path,
                                                                           capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a backbone deeper than the backbone of its source run takes the source blocks as its first blocks,
    exactly, while its next blocks keep the random initialisation it was built with, the console naming them, and that
    the trainer is told how many blocks were loaded.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the transfer.
    """
    source_configuration, target_configuration = build_deeper_backbone_configurations()
    device, dtype = torch.device("cpu"), torch.float32
    torch.manual_seed(seed=0)
    source_classifier = ImageClassifier(configuration=source_configuration, device=device, dtype=dtype)
    write_source_run(run_folder=tmp_path / "shorter_run", configuration=source_configuration, model=source_classifier)
    torch.manual_seed(seed=1)
    target_classifier = ImageClassifier(configuration=target_configuration, device=device, dtype=dtype)
    target_backbone_before = copy.deepcopy(target_classifier.backbone.state_dict())

    transfer_configuration = point_to_source_run(configuration=target_configuration, section_name="Backbone",
                                                 source_folder=tmp_path / "shorter_run")
    load_pretrained_component(component=target_classifier.backbone, section_name="Backbone", device=device,
                              model_configuration=transfer_configuration)

    # The loaded blocks hold the source tensors, the next blocks their own random tensors
    source_backbone_state = source_classifier.backbone.state_dict()
    for tensor_name, tensor_value in target_classifier.backbone.state_dict().items():
        block_index = find_backbone_block_index(tensor_name=tensor_name)
        assert block_index is not None, tensor_name
        expected_value = (source_backbone_state[tensor_name] if block_index < SHORTER_SOURCE_BLOCKS
                          else target_backbone_before[tensor_name])
        assert torch.equal(input=tensor_value, other=expected_value), tensor_name
    assert ("The Backbone Of This Run Starts With The 7 Blocks Of The Pretrained Backbone, Its Blocks 8 To 9 Keeping "
            "Random Weights" in capsys.readouterr().out)
    assert count_loaded_backbone_blocks(model_configuration=transfer_configuration) == SHORTER_SOURCE_BLOCKS

    # A backbone trained from random weights, or loaded whole from a source backbone of its depth, has no random block
    no_transfer_configuration = copy.deepcopy(transfer_configuration)
    no_transfer_configuration["Backbone"]["pretrained"] = NO_BACKBONE_TRANSFER
    assert count_loaded_backbone_blocks(model_configuration=no_transfer_configuration) is None
    write_source_run(run_folder=tmp_path / "same_depth_run", configuration=target_configuration,
                     model=target_classifier)
    assert count_loaded_backbone_blocks(model_configuration=point_to_source_run(
        configuration=target_configuration, section_name="Backbone", source_folder=tmp_path / "same_depth_run")) is None


@pytest.mark.parametrize(argnames=["source_blocks", "target_blocks", "changed_setting", "expected_shared_blocks",
                                   "expected_difference"],
                         argvalues=[(9, 9, None, None, None), (7, 9, None, 7, None), (9, 7, None, 7, None),
                                    (7, 9, "channels", None, "'Backbone.convolutions' is"),
                                    (7, 9, "modules", 7, "'Backbone.modules' is"),
                                    (9, 9, "last_max_pooling", None, "'Backbone.last_max_pooling' is")])
def test_backbones_sharing_their_first_blocks_are_compared_on_them(source_blocks: int, target_blocks: int,
                                                                   changed_setting: Optional[str],
                                                                   expected_shared_blocks: Optional[int],
                                                                   expected_difference: Optional[str]) -> None:
    """
    Verifies the architecture check of two backbones of different depths: a backbone shorter or deeper than its
    source backbone shares the blocks of the shorter one when they are the first blocks of the other, the pooling
    after the last block being free then, while backbones whose first blocks differ, whose modules differ, or of the
    same depth with another last pooling are refused.

    Args:
        source_blocks (int): The number of blocks of the source backbone, the first blocks of the fixture.
        target_blocks (int): The number of blocks of the receiving backbone.
        changed_setting (Optional[str]): The setting of the receiving backbone changed: "channels" for the channels of
            its second block, "modules" for its squeeze and excitation, "last_max_pooling", or None.
        expected_shared_blocks (Optional[int]): The number of shared blocks expected, None for none.
        expected_difference (Optional[str]): The start of the reported difference, None when the backbones fit.
    """
    # Backbones made of the first blocks of the fixture, the pooling after the last block differing between depths
    detection_configuration, _, _ = get_detection_model_configuration()
    blocks = list(detection_configuration["Backbone"]["convolutions"].items())
    source_backbone = copy.deepcopy(detection_configuration["Backbone"])
    target_backbone = copy.deepcopy(detection_configuration["Backbone"])
    source_backbone["convolutions"], target_backbone["convolutions"] = dict(blocks[:source_blocks]), dict(
        blocks[:target_blocks])
    if source_blocks != target_blocks:
        target_backbone["last_max_pooling"] = not source_backbone["last_max_pooling"]
    # One setting of the receiving backbone changed
    if changed_setting == "channels":
        target_backbone["convolutions"]["2"] = [2, 48]
    elif changed_setting == "modules":
        target_backbone["modules"]["squeeze_and_excite"] = True
    elif changed_setting == "last_max_pooling":
        target_backbone["last_max_pooling"] = not source_backbone["last_max_pooling"]

    assert count_shared_backbone_blocks(source_backbone_configuration=source_backbone,
                                        target_backbone_configuration=target_backbone) == expected_shared_blocks
    differences = compare_component_configurations(section_name="Backbone",
                                                   source_configuration={"Backbone": source_backbone},
                                                   target_configuration={"Backbone": target_backbone})
    if expected_difference is None:
        assert differences == []
    else:
        assert any(difference.startswith(expected_difference) for difference in differences), differences


def test_only_the_loaded_blocks_of_a_deeper_backbone_are_frozen() -> None:
    """
    Verifies the freezing of a backbone deeper than its source backbone, as the trainer applies it to its transferred
    part: the loaded blocks keep their weights and batch normalization statistics during a training step while the
    next blocks learn and update their statistics, the scale and shift of the normalization layers are fixed in the
    loaded blocks only, and a component loaded whole is its own transferred part.
    """
    _, target_configuration = build_deeper_backbone_configurations()
    device, dtype = torch.device("cpu"), torch.float32
    torch.manual_seed(seed=0)
    classifier = ImageClassifier(configuration=target_configuration, device=device, dtype=dtype)
    assert select_transferred_part(component=classifier.backbone, section_name="Backbone", number_loaded_blocks=None,
                                   with_class_layers=True) is classifier.backbone
    loaded_blocks = select_transferred_part(component=classifier.backbone, section_name="Backbone",
                                            number_loaded_blocks=SHORTER_SOURCE_BLOCKS, with_class_layers=True)
    freeze_component(component=loaded_blocks)
    backbone_state_before = copy.deepcopy(classifier.backbone.state_dict())

    # Training mode for the model, evaluation mode for the batch normalization of the loaded blocks, as in
    # BaseTrainer.set_training_mode
    classifier.train()
    freeze_batch_normalization_statistics(component=loaded_blocks)
    optimizer = torch.optim.Adam(params=classifier.parameters(), lr=0.1)
    loss = classifier(input_tensor=torch.rand(size=(2, 3, 512, 512)) * 255.0,
                      labels=torch.tensor(data=[0, 1]))["classification_loss"]
    loss.backward()
    optimizer.step()

    # Loaded blocks unchanged, next blocks learning, their statistics included
    for tensor_name, tensor_value in classifier.backbone.state_dict().items():
        unchanged = torch.equal(input=tensor_value, other=backbone_state_before[tensor_name])
        if find_backbone_block_index(tensor_name=tensor_name) < SHORTER_SOURCE_BLOCKS:
            assert unchanged, tensor_name
        elif tensor_name.endswith((".weight", ".running_mean")):
            assert not unchanged, tensor_name
    for block_index, block in enumerate(classifier.backbone.convolution_blocks):
        assert all(parameter.requires_grad == (block_index >= SHORTER_SOURCE_BLOCKS)
                   for parameter in block.parameters())

    # Fixed normalization parameters in the loaded blocks of a fresh backbone only, the convolutions learning
    torch.manual_seed(seed=0)
    classifier = ImageClassifier(configuration=target_configuration, device=device, dtype=dtype)
    freeze_normalization_parameters(component=select_transferred_part(component=classifier.backbone,
                                                                      section_name="Backbone",
                                                                      number_loaded_blocks=SHORTER_SOURCE_BLOCKS,
                                                                      with_class_layers=True))
    normalization_names = [module_name for module_name, module in classifier.backbone.named_modules()
                           if isinstance(module, NORMALIZATION_LAYER_TYPES)]
    for parameter_name, parameter_value in classifier.backbone.named_parameters():
        is_normalization = parameter_name.rsplit(".", 1)[0] in normalization_names
        is_loaded = find_backbone_block_index(tensor_name=parameter_name) < SHORTER_SOURCE_BLOCKS
        assert parameter_value.requires_grad == (not (is_normalization and is_loaded)), parameter_name


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
    configuration["ClassificationHead"]["pretrained"] = {**NO_HEAD_TRANSFER, "experiment_folder": "/runs/source",
                                                         "class_layer": True}

    disable_pretrained_settings(model_configuration=configuration)

    assert configuration["Backbone"]["pretrained"] == {**NO_BACKBONE_TRANSFER, "learning_rate_multiplier": 0.5}
    assert configuration["TemporalModel"]["pretrained"] == {**NO_TRANSFER, "iteration": 4}
    assert configuration["ClassificationHead"]["pretrained"] == NO_HEAD_TRANSFER


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


def write_detector_run(run_folder: Path, mask_head: bool = False, class_names: Optional[List[str]] = None,
                       neck_type: str = "none") -> Tuple[Dict[str, Any], TaskModel]:
    """
    Writes a trained object detection run with the small detector configuration of the tests, whose region proposal
    and detection heads (and neck) initialise those of other detection runs, or an instance segmentation run with
    mask_head, whose mask heads initialise those of other instance segmentation runs.

    Args:
        run_folder (Path): The run folder.
        mask_head (bool): Whether to write an instance segmentation run with mask heads. Defaults to False.
        class_names (Optional[List[str]]): The class names of the run in class index order, the background first,
            None to write no data record. Defaults to None.
        neck_type (str): The 'Neck.type' of the run. Defaults to "none", the heads reading the backbone blocks.

    Returns:
        Tuple[Dict[str, Any], TaskModel]: The configuration and the model of the run.
    """
    configuration, _, _ = get_detection_model_configuration(mask_head=mask_head, neck_type=neck_type)
    task = "instance_segmentation" if mask_head else "object_detection"
    source_detector = build_task_model(task=task, configuration=configuration, seed=0)
    write_source_run(run_folder=run_folder, configuration=configuration, model=source_detector, class_names=class_names)
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


@pytest.mark.parametrize(argnames=["source_neck_settings", "target_neck_settings", "refused_setting"], argvalues=[
    ({"type": "feature_pyramid"}, {"type": "none"}, "'Neck.type'"),
    ({"type": "none"}, {"type": "path_aggregation"}, "'Neck.type'"),
    ({"type": "feature_pyramid"}, {"type": "path_aggregation"}, "'Neck.type'"),
    ({"type": "feature_pyramid"}, {"type": "feature_pyramid", "output_channels": 64}, "'Neck.output_channels'"),
    ({"type": "path_aggregation"},
     {"type": "path_aggregation", "normalization": {"feature_map_normalization": "layer"}}, None),
    ({"type": "feature_pyramid"}, {"type": "feature_pyramid"}, None)])
def test_detection_heads_move_between_runs_of_the_same_neck(source_neck_settings: Dict[str, Any],
                                                            target_neck_settings: Dict[str, Any],
                                                            refused_setting: Optional[str]) -> None:
    """
    Verifies that the region proposal, detection and mask heads, which read the levels after the neck, move between
    runs without neck or with the same type of neck of the same output channels, the normalization of the neck changing
    no weight of the heads, and are refused when one run has a neck and the other none, when the necks are of other
    types or when their output channels differ. A source run written before the neck (without 'Neck' section) counts as
    a run without neck.

    Args:
        source_neck_settings (Dict[str, Any]): The 'Neck' settings of the source run changed from the fixture.
        target_neck_settings (Dict[str, Any]): The 'Neck' settings of the run receiving the heads changed from the
            fixture.
        refused_setting (Optional[str]): The setting named by the refusal of every head, None when the heads move.
    """
    configuration, _, _ = get_detection_model_configuration(mask_head=True)
    source_configuration = copy.deepcopy(configuration)
    source_configuration["Neck"].update(source_neck_settings)
    target_configuration = copy.deepcopy(configuration)
    target_configuration["Neck"].update(target_neck_settings)

    # Every head is refused by the neck setting differing between both runs, or accepted
    for section_name in ["RegionProposal", "Detector", "MaskHead"]:
        differences = compare_component_configurations(section_name=section_name,
                                                       source_configuration=source_configuration,
                                                       target_configuration=target_configuration)
        if refused_setting is None:
            assert differences == [], differences
        else:
            assert len(differences) == 1 and refused_setting in differences[0], differences

    # A source run written before the neck counts as a run without neck
    source_configuration.pop("Neck")
    for section_name in ["RegionProposal", "Detector", "MaskHead"]:
        differences = compare_component_configurations(section_name=section_name,
                                                       source_configuration=source_configuration,
                                                       target_configuration=target_configuration)
        assert (differences == []) == (target_configuration["Neck"]["type"] == "none"), differences


@pytest.mark.parametrize(argnames="neck_type", argvalues=["feature_pyramid", "path_aggregation"])
def test_neck_initialises_another_detection_run(tmp_path: Path, capsys: pytest.CaptureFixture, neck_type: str) -> None:
    """
    Verifies the transfer of the neck of an object detection run to an instance segmentation run with the same neck:
    every tensor of the neck (its convolutions, normalization scales and batch statistics) starts from exactly the
    weights of the source run, while the backbone and the heads keep their own weights.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the transfer.
        neck_type (str): The 'Neck.type' of both runs.
    """
    _, source_detector = write_detector_run(run_folder=tmp_path / "detection_run", neck_type=neck_type)
    target_configuration, _, _ = get_detection_model_configuration(mask_head=True, neck_type=neck_type)
    instance_segmenter = build_task_model(task="instance_segmentation", configuration=target_configuration, seed=1)
    source_neck_state = source_detector.neck.state_dict()
    other_states_before = {attribute_name: copy.deepcopy(getattr(instance_segmenter, attribute_name).state_dict())
                           for attribute_name in ["backbone", "region_proposer_dictionary",
                                                  "detector_heads_dictionary", "mask_heads_dictionary"]}
    assert any(not torch.equal(input=tensor_value, other=source_neck_state[tensor_name])
               for tensor_name, tensor_value in instance_segmenter.neck.state_dict().items()
               if tensor_value.is_floating_point())

    # The neck of the detection run, then every other component unchanged
    load_pretrained_component(component=instance_segmenter.neck, section_name="Neck", device=torch.device("cpu"),
                              model_configuration=point_to_source_run(configuration=target_configuration,
                                                                      section_name="Neck",
                                                                      source_folder=tmp_path / "detection_run"))
    assert sorted(instance_segmenter.neck.state_dict()) == sorted(source_neck_state)
    for tensor_name, tensor_value in instance_segmenter.neck.state_dict().items():
        assert torch.equal(input=tensor_value, other=source_neck_state[tensor_name]), tensor_name
    for attribute_name, state_before in other_states_before.items():
        for tensor_name, tensor_value in getattr(instance_segmenter, attribute_name).state_dict().items():
            assert torch.equal(input=tensor_value, other=state_before[tensor_name]), f"{attribute_name}.{tensor_name}"
    assert "Neck Initialised From The Trained Weights Of" in capsys.readouterr().out


def change_neck_and_level_channels(configuration: Dict[str, Any]) -> None:
    """
    Gives a detection configuration of the tests other neck output channels, another neck normalization and another
    width of the backbone block of level 5, read by the lateral of that level.

    Args:
        configuration (Dict[str, Any]): The detection configuration, modified in place.
    """
    configuration["Neck"].update({"output_channels": 16, "normalization": {"feature_map_normalization": "layer"}})
    configuration["Backbone"]["convolutions"]["5"] = [2, 32]


def change_neck_type(configuration: Dict[str, Any]) -> None:
    """
    Gives a detection configuration of the tests the path aggregation neck instead of the feature pyramid.

    Args:
        configuration (Dict[str, Any]): The detection configuration, modified in place.
    """
    configuration["Neck"]["type"] = "path_aggregation"


def remove_finest_level(configuration: Dict[str, Any]) -> None:
    """
    Removes level 5 from a detection configuration of the tests, its neck then having no lateral for it.

    Args:
        configuration (Dict[str, Any]): The detection configuration, modified in place.
    """
    del configuration["DetectionLevels"]["5"]


@pytest.mark.parametrize(argnames=["modify_function", "expected_messages"], argvalues=[
    (change_neck_type, ["'Neck.type' is 'feature_pyramid' in the source run and 'path_aggregation' in this run."]),
    (change_neck_and_level_channels, ["'Neck.output_channels' is 32 in the source run and 16 in this run.",
                                      "'Neck.normalization' is {'feature_map_normalization': 'batch'} in the source "
                                      "run and {'feature_map_normalization': 'layer'} in this run.",
                                      "The backbone blocks of the levels have {'5': 64, '6': 64, '7': 64} output "
                                      "channels in the source run and {'5': 32, '6': 64, '7': 64} in this run"]),
    (remove_finest_level, ["The detection levels are ['5', '6', '7'] in the source run and ['6', '7'] in this run"])])
def test_different_necks_are_rejected(tmp_path: Path, modify_function: Callable[[Dict[str, Any]], None],
                                      expected_messages: List[str]) -> None:
    """
    Verifies that a neck is refused by a run whose neck has another type, other output channels, another normalization,
    other detection levels or other channels of the backbone blocks of its levels, the error naming every difference,
    and by a run whose source run was written before the neck.

    Args:
        tmp_path (Path): The pytest temporary directory.
        modify_function (Callable[[Dict[str, Any]], None]): Function changing the configuration of the run receiving
            the neck.
        expected_messages (List[str]): The differences the refusal names.
    """
    source_configuration, _ = write_detector_run(run_folder=tmp_path / "detection_run", neck_type="feature_pyramid")
    target_configuration = copy.deepcopy(source_configuration)
    modify_function(configuration=target_configuration)
    detector = build_task_model(task="object_detection", configuration=target_configuration, seed=1)
    with pytest.raises(expected_exception=ValueError) as error_information:
        load_pretrained_component(component=detector.neck, section_name="Neck", device=torch.device("cpu"),
                                  model_configuration=point_to_source_run(configuration=target_configuration,
                                                                          section_name="Neck",
                                                                          source_folder=tmp_path / "detection_run"))
    assert "The Neck Of The Source Run" in str(error_information.value)
    for expected_message in expected_messages:
        assert expected_message in str(error_information.value)

    # A source run written before the neck has none to give
    old_configuration = copy.deepcopy(source_configuration)
    old_configuration.pop("Neck")
    (tmp_path / "detection_run" / "training_configuration.yaml").write_text(data=yaml.safe_dump(data=old_configuration))
    with pytest.raises(expected_exception=ValueError, match="Has No 'Neck' Section"):
        load_pretrained_component(component=detector.neck, section_name="Neck", device=torch.device("cpu"),
                                  model_configuration=point_to_source_run(configuration=target_configuration,
                                                                          section_name="Neck",
                                                                          source_folder=tmp_path / "detection_run"))


@pytest.mark.parametrize(argnames=["neck_source", "expected_warning"], argvalues=[
    (None, "But The Neck Of This Run Starts From Random Weights"),
    ("other_run", "But The Neck Of This Run Comes From Another Run"),
    ("other_checkpoint", "But The Neck Of This Run Comes From Another Checkpoint Of Their Run"),
    ("same_checkpoint", None)])
def test_heads_behind_a_neck_warn_when_the_neck_is_not_theirs(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                              neck_source: Optional[str],
                                                              expected_warning: Optional[str]) -> None:
    """
    Verifies that region proposal and detection heads trained behind a neck still initialise a run with the same neck
    when its neck is not loaded from their checkpoint, with a warning that they read other features than those they were
    trained on: a neck starting from random weights, the neck of another run, or the neck of another checkpoint of their
    run. The neck of their own checkpoint gives no warning.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the transfer.
        neck_source (Optional[str]): Where the neck of the run comes from: None (random weights), "other_run",
            "other_checkpoint" (iteration 2 of the run of the heads) or "same_checkpoint" (its last checkpoint).
        expected_warning (Optional[str]): The end of the warning naming the neck, None when nothing is printed.
    """
    source_configuration, source_detector = write_detector_run(run_folder=tmp_path / "detection_run",
                                                               neck_type="feature_pyramid")
    write_detector_run(run_folder=tmp_path / "other_detection_run", neck_type="feature_pyramid")
    save_checkpoint(weights_directory=tmp_path / "detection_run" / "Weights", iteration=2, model=source_detector,
                    optimizer=torch.optim.Adam(params=source_detector.parameters()),
                    metadata={"task": "object_detection"}, exponential_moving_average_state=None)

    # Run reading the heads of the last checkpoint of the source run, its neck from the run asked by the test
    neck_settings = {None: NO_TRANSFER,
                     "other_run": {**NO_TRANSFER, "experiment_folder": str(tmp_path / "other_detection_run")},
                     "other_checkpoint": {**NO_TRANSFER, "experiment_folder": str(tmp_path / "detection_run"),
                                          "iteration": 2},
                     "same_checkpoint": {**NO_TRANSFER, "experiment_folder": str(tmp_path / "detection_run"),
                                         "iteration": 3}}
    target_configuration = copy.deepcopy(source_configuration)
    target_configuration["Neck"]["pretrained"] = neck_settings[neck_source]
    detector = build_task_model(task="object_detection", configuration=target_configuration, seed=1)
    for section_name in ["RegionProposal", "Detector"]:
        heads_configuration = point_to_source_run(configuration=target_configuration, section_name=section_name,
                                                  source_folder=tmp_path / "detection_run")
        heads_configuration[section_name]["pretrained"]["iteration"] = 3
        load_pretrained_component(component=getattr(detector, TRANSFERABLE_COMPONENTS[section_name]),
                                  section_name=section_name, model_configuration=heads_configuration,
                                  device=torch.device("cpu"))

    # One warning per head section, or none when the neck comes from the checkpoint of the heads
    console_output = capsys.readouterr().out
    if expected_warning is None:
        assert "Were Trained Behind The Neck" not in console_output
    else:
        for component_name in ["Region Proposal Heads", "Detection Heads"]:
            assert (f"The Pretrained {component_name} Were Trained Behind The Neck Of Their Source Run, "
                    f"{expected_warning}") in console_output


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


@pytest.mark.parametrize(argnames="task", argvalues=["classification"])
def test_classification_head_adds_classes_after_the_source_classes(tmp_path: Path, task: str,
                                                                   capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the classification head of a run of 3 classes initialises the head of a run adding 2 classes after
    them, for an image classifier: the hidden layers and the classifier rows of the source classes are copied exactly,
    and the new classes keep the random weights and biases the head of the new run was built with.

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


@pytest.mark.parametrize(argnames=["target_class_names", "number_other_classes"],
                         argvalues=[(["rhombus", "triangle", "square", "ellipse", "circle"], 2),
                                    (["triangle", "square", "circle"], 0)])
def test_classification_head_finds_the_source_classes_by_name(tmp_path: Path, target_class_names: List[str],
                                                              number_other_classes: int,
                                                              capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the classification head of a run of 3 classes initialises the head of a run holding them at other
    indices, among new classes or alone in another order, as the 1,000 ImageNet-1k classes among the merged ImageNet
    classes sorted by WordNet identifier: the hidden layers are copied exactly, the row of every source class lands at
    the index of its name, and the other classes keep the random weights and biases the head of the new run was built
    with.

    Args:
        tmp_path (Path): The pytest temporary directory.
        target_class_names (List[str]): The class names of the run receiving the head, in class index order.
        number_other_classes (int): The number of classes of that run missing from the source run.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the transfer.
    """
    source_configuration = get_class_configuration(task="classification", number_classes=3)
    source_model = build_task_model(task="classification", configuration=source_configuration, seed=0)
    write_source_run(run_folder=tmp_path / "source_run", configuration=source_configuration, model=source_model,
                     class_names=SOURCE_CLASS_NAMES)
    target_configuration = get_class_configuration(task="classification", number_classes=len(target_class_names))
    target_model = build_task_model(task="classification", configuration=target_configuration, seed=1)
    initial_target_state = {tensor_name: tensor_value.clone() for tensor_name, tensor_value in
                            target_model.classification_head.state_dict().items()}
    load_pretrained_component(component=target_model.classification_head, section_name="ClassificationHead",
                              model_configuration=point_to_source_run(configuration=target_configuration,
                                                                      section_name="ClassificationHead",
                                                                      source_folder=tmp_path / "source_run"),
                              device=torch.device("cpu"), class_names=target_class_names)

    # Hidden layers copied exactly
    source_state = source_model.classification_head.state_dict()
    target_state = target_model.classification_head.state_dict()
    for tensor_name, tensor_value in source_state.items():
        if not tensor_name.startswith("classifier."):
            assert torch.equal(input=target_state[tensor_name], other=tensor_value), tensor_name

    # Row of every source class at the index of its name, the other rows as the new run built them
    for tensor_name in ["classifier.weight", "classifier.bias"]:
        for target_index, class_name in enumerate(target_class_names):
            expected_row = source_state[tensor_name][SOURCE_CLASS_NAMES.index(class_name)] if \
                class_name in SOURCE_CLASS_NAMES else initial_target_state[tensor_name][target_index]
            assert torch.equal(input=target_state[tensor_name][target_index], other=expected_row), class_name
    assert (f"Classification Head : 3 Classes Of The Source Run Moved To The Index Of Their Name In This Run, "
            f"{number_other_classes} Other Classes Randomly Initialised.") in capsys.readouterr().out


def test_classification_head_refuses_a_run_without_every_source_class(tmp_path: Path) -> None:
    """
    Verifies that the classification head of a run of 3 classes is refused by a run lacking one of them, even with
    more classes, the refusal naming the missing class, and that the head of the new run keeps its own weights.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    source_configuration = get_class_configuration(task="classification", number_classes=3)
    source_model = build_task_model(task="classification", configuration=source_configuration, seed=0)
    write_source_run(run_folder=tmp_path / "source_run", configuration=source_configuration, model=source_model,
                     class_names=SOURCE_CLASS_NAMES)
    target_class_names = ["square", "rhombus", "triangle", "ellipse"]
    target_configuration = get_class_configuration(task="classification", number_classes=len(target_class_names))
    target_model = build_task_model(task="classification", configuration=target_configuration, seed=1)
    initial_target_state = copy.deepcopy(target_model.classification_head.state_dict())

    with pytest.raises(expected_exception=ValueError, match=r"Their Classes Differ: The classes \['circle'\] of"):
        load_pretrained_component(component=target_model.classification_head, section_name="ClassificationHead",
                                  model_configuration=point_to_source_run(configuration=target_configuration,
                                                                          section_name="ClassificationHead",
                                                                          source_folder=tmp_path / "source_run"),
                                  device=torch.device("cpu"), class_names=target_class_names)
    for tensor_name, tensor_value in target_model.classification_head.state_dict().items():
        assert torch.equal(input=tensor_value, other=initial_target_state[tensor_name]), tensor_name


@pytest.mark.parametrize(argnames="task", argvalues=["classification", "sequence_labelling"])
def test_classification_head_without_class_layer_initialises_a_run_of_other_classes(
        tmp_path: Path, task: str, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the classification head of a run of 3 classes, loaded without its class layer
    ('ClassificationHead.pretrained.class_layer' False), initialises the head of a run of 4 unrelated classes, which
    the transfer with the class layer refuses, for an image classifier and a sequence labeller: the hidden layers are
    copied exactly, the class layer keeps the random weights and biases the head of the new run was built with, and
    the console says so.

    Args:
        tmp_path (Path): The pytest temporary directory.
        task (str): The task of both runs.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the transfer.
    """
    source_configuration = get_class_configuration(task=task, number_classes=3)
    source_model = build_task_model(task=task, configuration=source_configuration, seed=0)
    write_source_run(run_folder=tmp_path / "source_run", configuration=source_configuration, model=source_model,
                     class_names=SOURCE_CLASS_NAMES)
    target_class_names = ["cat", "dog", "parcel", "bicycle"]
    target_configuration = get_class_configuration(task=task, number_classes=len(target_class_names))
    target_model = build_task_model(task=task, configuration=target_configuration, seed=1)
    initial_target_state = copy.deepcopy(target_model.classification_head.state_dict())

    # Refused with its class layer, none of the source classes being a class of this run
    with pytest.raises(expected_exception=ValueError, match="Their Classes Differ"):
        load_pretrained_component(component=target_model.classification_head, section_name="ClassificationHead",
                                  model_configuration=point_to_source_run(configuration=target_configuration,
                                                                          section_name="ClassificationHead",
                                                                          source_folder=tmp_path / "source_run",
                                                                          class_layers=True),
                                  device=torch.device("cpu"), class_names=target_class_names)

    # Loaded without it: hidden layers of the source run, class layer of the new run
    load_pretrained_component(component=target_model.classification_head, section_name="ClassificationHead",
                              model_configuration=point_to_source_run(configuration=target_configuration,
                                                                      section_name="ClassificationHead",
                                                                      source_folder=tmp_path / "source_run",
                                                                      class_layers=False),
                              device=torch.device("cpu"), class_names=target_class_names)
    source_state = source_model.classification_head.state_dict()
    target_state = target_model.classification_head.state_dict()
    assert any(not tensor_name.startswith("classifier.") for tensor_name in source_state)
    for tensor_name, tensor_value in source_state.items():
        expected_value = initial_target_state[tensor_name] if tensor_name.startswith("classifier.") else tensor_value
        assert torch.equal(input=target_state[tensor_name], other=expected_value), tensor_name
    assert ("Classification Head : Loaded Without Its Class Layer, Which Keeps The Random Initialisation Of This Run "
            "For Its 4 Classes (The Source Run Predicts 3 Classes).") in capsys.readouterr().out


def test_classification_head_without_hidden_layer_is_refused_without_its_class_layer(tmp_path: Path) -> None:
    """
    Verifies that a classification head without hidden layer ('fully_connected' []) is refused when loaded without its
    class layer, which is all it holds, and that the head of the new run keeps its own weights.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    source_configuration = get_class_configuration(task="classification", number_classes=3)
    source_configuration["ClassificationHead"]["fully_connected"] = []
    source_model = build_task_model(task="classification", configuration=source_configuration, seed=0)
    write_source_run(run_folder=tmp_path / "source_run", configuration=source_configuration, model=source_model,
                     class_names=SOURCE_CLASS_NAMES)
    target_configuration = copy.deepcopy(source_configuration)
    target_model = build_task_model(task="classification", configuration=target_configuration, seed=1)
    initial_target_state = copy.deepcopy(target_model.classification_head.state_dict())

    with pytest.raises(expected_exception=ValueError, match="Has No Layer Besides Its Class Layers"):
        load_pretrained_component(component=target_model.classification_head, section_name="ClassificationHead",
                                  model_configuration=point_to_source_run(configuration=target_configuration,
                                                                          section_name="ClassificationHead",
                                                                          source_folder=tmp_path / "source_run",
                                                                          class_layers=False),
                                  device=torch.device("cpu"), class_names=SOURCE_CLASS_NAMES)
    for tensor_name, tensor_value in target_model.classification_head.state_dict().items():
        assert torch.equal(input=tensor_value, other=initial_target_state[tensor_name]), tensor_name


def test_head_loaded_without_its_class_layer_freezes_its_hidden_layers_only() -> None:
    """
    Verifies the transferred part of a classification head loaded without its class layer, which the trainer freezes
    and fine-tunes at the learning rate multiplier of the transfer: its hidden layers and their batch normalization,
    without the class layer, so that a training step of the frozen part changes the class layer only, the hidden
    layers keeping their weights and batch statistics, while a head loaded with its class layer is its own
    transferred part.
    """
    configuration = get_class_configuration(task="classification", number_classes=3)
    configuration["ClassificationHead"]["fully_connected_normalization"] = "batch"
    classifier = build_task_model(task="classification", configuration=configuration, seed=0)
    head = classifier.classification_head
    assert select_transferred_part(component=head, section_name="ClassificationHead", number_loaded_blocks=None,
                                   with_class_layers=True) is head

    # Parameters of the hidden layers only, the class layer staying out of the transferred part
    transferred_part = select_transferred_part(component=head, section_name="ClassificationHead",
                                               number_loaded_blocks=None, with_class_layers=False)
    transferred_identifiers = {id(parameter) for parameter in transferred_part.parameters()}
    assert transferred_identifiers == {id(parameter) for parameter in head.fully_connected_block.parameters()}
    assert not transferred_identifiers & {id(parameter) for parameter in head.classifier.parameters()}

    # Frozen part in evaluation mode during a training step, as BaseTrainer.set_training_mode applies it
    freeze_component(component=transferred_part)
    head_state_before = copy.deepcopy(head.state_dict())
    classifier.train()
    freeze_batch_normalization_statistics(component=transferred_part)
    optimizer = torch.optim.Adam(params=[parameter for parameter in classifier.parameters()
                                         if parameter.requires_grad], lr=0.1)
    loss = classifier(input_tensor=torch.rand(size=(2, 3, 64, 64)) * 255.0,
                      labels=torch.tensor(data=[0, 1]))["classification_loss"]
    loss.backward()
    optimizer.step()

    # Class layer learning, hidden layers and their statistics unchanged
    for tensor_name, tensor_value in head.state_dict().items():
        unchanged = torch.equal(input=tensor_value, other=head_state_before[tensor_name])
        assert unchanged == (not tensor_name.startswith("classifier.")), tensor_name


# Source class names, class names of the new run, expected index of every source class in the new run, and a text of
# the expected refusal (None when the classes are found)
CLASS_NAME_CASES = [(["a", "b"], ["a", "b", "c"], [0, 1], None),
                    (["a", "b"], ["c", "b", "a"], [2, 1], None),
                    (None, ["c", "b", "a"], [0, 1], None),
                    (["a", "b"], ["a", "c", "d"], [0, 1], "The classes ['b'] of the source run"),
                    ([f"source_{index}" for index in range(8)], [f"target_{index}" for index in range(8)],
                     list(range(8)), "'source_3', 'source_4'] (8 classes missing)"),
                    (["a", "b"], ["b", "a", "a"], [0, 1], "The class names ['a'] name several classes"),
                    (["a", "b", "c"], ["c", "b"], [0, 1, 2], "a run keeps every class of its source run")]


@pytest.mark.parametrize(argnames=["source_class_names", "target_class_names", "expected_indices", "expected_refusal"],
                         argvalues=CLASS_NAME_CASES)
def test_source_classes_are_found_first_or_by_name(source_class_names: Optional[List[str]],
                                                   target_class_names: List[str], expected_indices: List[int],
                                                   expected_refusal: Optional[str]) -> None:
    """
    Verifies where compare_class_names places the classes of a source run in a new run: first and in the same order
    when the names agree or are unknown, at the index of their name otherwise, and refused when a source class is
    missing (the first 5 named, all counted), when a name names several classes, or when the new run has fewer
    classes.

    Args:
        source_class_names (Optional[List[str]]): The class names of the source run, None when it has none.
        target_class_names (List[str]): The class names of the new run.
        expected_indices (List[int]): The expected index in the new run of every source class.
        expected_refusal (Optional[str]): A text of the expected refusal, None when the classes are found.
    """
    number_source_classes = len(expected_indices)
    target_class_indices, differences = compare_class_names(
        source_class_names=source_class_names, target_class_names=target_class_names,
        number_source_classes=number_source_classes, number_target_classes=len(target_class_names))
    assert target_class_indices == expected_indices
    if expected_refusal is None:
        assert differences == []
    else:
        assert len(differences) == 1 and expected_refusal in differences[0], differences


def test_detection_heads_find_the_source_classes_by_name(tmp_path: Path) -> None:
    """
    Verifies that the detection heads of a detector of 5 foreground classes initialise the heads of a detector holding
    them at other indices among 2 new classes: on every level, the classifier row of the background and of every
    source class and the 4 box regressor rows of every source class land at the index of its name, and the rows of the
    new classes keep the random weights and biases the heads of the new detector were built with.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    source_configuration, _, _ = get_detection_model_configuration()
    source_class_names = list_detection_class_names(configuration=source_configuration)
    source_configuration, source_detector = write_detector_run(run_folder=tmp_path / "detection_run",
                                                               class_names=source_class_names)
    target_configuration = copy.deepcopy(source_configuration)
    target_configuration["Data"]["label_dictionary"] = {"background": 0, "foreground_6": 1, "foreground_5": 2,
                                                        "foreground_1": 3, "foreground_2": 4, "foreground_4": 5,
                                                        "foreground_3": 6, "foreground_7": 7}
    target_class_names = list_detection_class_names(configuration=target_configuration)
    target_detector = build_task_model(task="object_detection", configuration=target_configuration, seed=1)
    initial_target_state = {tensor_name: tensor_value.clone() for tensor_name, tensor_value in
                            target_detector.detector_heads_dictionary.state_dict().items()}
    load_pretrained_component(component=target_detector.detector_heads_dictionary, section_name="Detector",
                              model_configuration=point_to_source_run(configuration=target_configuration,
                                                                      section_name="Detector",
                                                                      source_folder=tmp_path / "detection_run"),
                              device=torch.device("cpu"), class_names=target_class_names)

    # Rows of every class of the new run, from the class of the same name in the source run or as the new run built
    # them, 1 row per class for the classifier and 4 for the box regressor
    source_state = source_detector.detector_heads_dictionary.state_dict()
    target_state = target_detector.detector_heads_dictionary.state_dict()
    for level in target_configuration["DetectionLevels"]:
        for layer_name, rows_per_class in [("classifier", 1), ("bounding_box_regressor", 4)]:
            for parameter_name in ["weight", "bias"]:
                tensor_name = f"{level}.{layer_name}.{parameter_name}"
                for target_index, class_name in enumerate(target_class_names):
                    target_rows = target_state[tensor_name][target_index * rows_per_class:
                                                            (target_index + 1) * rows_per_class]
                    if class_name in source_class_names:
                        source_index = source_class_names.index(class_name)
                        expected_rows = source_state[tensor_name][source_index * rows_per_class:
                                                                  (source_index + 1) * rows_per_class]
                    else:
                        expected_rows = initial_target_state[tensor_name][target_index * rows_per_class:
                                                                          (target_index + 1) * rows_per_class]
                    assert torch.equal(input=target_rows, other=expected_rows), f"{tensor_name} {class_name}"


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


# Finer inputs of the mask heads read by their own layers: backbone skips at the pool size and at the mask resolution
# with fusion convolutions, the image branch staying off since it is not transferred with the mask heads
MASK_HEAD_SKIP_SETTINGS = {"skip_blocks_at_pool_size": [4], "image_branch_convolutions": [0, 0],
                           "skip_blocks_at_mask_resolution": [2], "fusion_convolutions": [1, 16]}


def test_mask_heads_with_fine_inputs_move_between_runs_of_the_same_inputs(tmp_path: Path) -> None:
    """
    Verifies the transfer of mask heads reading backbone skips through fusion convolutions: every tensor of the heads,
    the fusion convolutions and the convolution blocks widened by the skips included, is copied exactly to a run of
    the same finer inputs; a run of other finer inputs is refused with the differing settings, and a source run written
    before the finer inputs counts as a run without them.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration, _, _ = get_detection_model_configuration(mask_head=True)
    configuration = copy.deepcopy(configuration)
    configuration["MaskHead"].update(MASK_HEAD_SKIP_SETTINGS)
    source_segmenter = build_task_model(task="instance_segmentation", configuration=configuration, seed=0)
    write_source_run(run_folder=tmp_path / "instance_run", configuration=configuration, model=source_segmenter)
    target_segmenter = build_task_model(task="instance_segmentation", configuration=configuration, seed=1)
    load_pretrained_component(component=target_segmenter.mask_heads_dictionary, section_name="MaskHead",
                              model_configuration=point_to_source_run(configuration=configuration,
                                                                      section_name="MaskHead",
                                                                      source_folder=tmp_path / "instance_run"),
                              device=torch.device("cpu"))

    # Every tensor copied, the fusion convolutions included
    source_state = source_segmenter.mask_heads_dictionary.state_dict()
    target_state = target_segmenter.mask_heads_dictionary.state_dict()
    assert any("fusion_block" in tensor_name for tensor_name in source_state)
    assert source_state.keys() == target_state.keys()
    for tensor_name, tensor_value in source_state.items():
        assert torch.equal(input=target_state[tensor_name], other=tensor_value), tensor_name

    # Other finer inputs are refused, each differing setting being named
    other_configuration = copy.deepcopy(configuration)
    other_configuration["MaskHead"].update({"skip_blocks_at_pool_size": [3], "fusion_convolutions": [2, 16]})
    differences = compare_component_configurations(section_name="MaskHead", source_configuration=configuration,
                                                   target_configuration=other_configuration)
    assert len(differences) == 2
    assert "'MaskHead.skip_blocks_at_pool_size' is [4] in the source run and [3] in this run." in differences
    assert "'MaskHead.fusion_convolutions' is [1, 16] in the source run and [2, 16] in this run." in differences

    # A source run written before the finer inputs passes its heads to a run without them, not to a run with them
    former_configuration, _, _ = get_detection_model_configuration(mask_head=True)
    former_configuration = copy.deepcopy(former_configuration)
    for key_name in MASK_HEAD_SKIP_SETTINGS:
        former_configuration["MaskHead"].pop(key_name)
    target_configuration, _, _ = get_detection_model_configuration(mask_head=True)
    assert compare_component_configurations(section_name="MaskHead", source_configuration=former_configuration,
                                            target_configuration=target_configuration) == []
    assert len(compare_component_configurations(section_name="MaskHead", source_configuration=former_configuration,
                                                target_configuration=configuration)) == 3


def test_mask_heads_move_only_between_runs_of_the_same_sharing(tmp_path: Path) -> None:
    """
    Verifies that the mask head shared by the levels ('MaskHead.shared') moves to a run that also shares it, every
    tensor being copied, and that mask heads trained per level are refused by a run sharing one head and the reverse,
    a source run written before the sharing counting as one head per level.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    shared_configuration, _, _ = get_detection_model_configuration(mask_head=True)
    shared_configuration = copy.deepcopy(shared_configuration)
    shared_configuration["MaskHead"]["shared"] = True
    source_segmenter = build_task_model(task="instance_segmentation", configuration=shared_configuration, seed=0)
    write_source_run(run_folder=tmp_path / "shared_run", configuration=shared_configuration, model=source_segmenter)
    target_segmenter = build_task_model(task="instance_segmentation", configuration=shared_configuration, seed=1)
    load_pretrained_component(component=target_segmenter.mask_heads_dictionary, section_name="MaskHead",
                              model_configuration=point_to_source_run(configuration=shared_configuration,
                                                                      section_name="MaskHead",
                                                                      source_folder=tmp_path / "shared_run"),
                              device=torch.device("cpu"))

    # Every tensor of the single shared head copied
    source_state = source_segmenter.mask_heads_dictionary.state_dict()
    target_state = target_segmenter.mask_heads_dictionary.state_dict()
    assert {tensor_name.split(".")[0] for tensor_name in source_state} == {"shared"}
    assert source_state.keys() == target_state.keys()
    for tensor_name, tensor_value in source_state.items():
        assert torch.equal(input=target_state[tensor_name], other=tensor_value), tensor_name

    # Heads per level and a shared head are refused by each other, a source run written before the sharing being one
    # head per level
    level_configuration, _, _ = get_detection_model_configuration(mask_head=True)
    former_configuration = copy.deepcopy(level_configuration)
    former_configuration["MaskHead"].pop("shared")
    for source_configuration in [level_configuration, former_configuration]:
        assert compare_component_configurations(section_name="MaskHead", source_configuration=source_configuration,
                                                target_configuration=shared_configuration) == [
            "'MaskHead.shared' is False in the source run and True in this run."]
    assert compare_component_configurations(section_name="MaskHead", source_configuration=shared_configuration,
                                            target_configuration=level_configuration) == [
        "'MaskHead.shared' is True in the source run and False in this run."]
    assert compare_component_configurations(section_name="MaskHead", source_configuration=former_configuration,
                                            target_configuration=level_configuration) == []


def test_segmentation_class_layers_follow_the_decoder(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the loading of the class layers of a semantic segmenter with its decoder
    ('SegmentationDecoder.pretrained.class_layers'), the run receiving them adding 2 classes and a deep
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
    target_configuration["SegmentationDecoder"]["pretrained"]["class_layers"] = True
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


# Keypoint names of the classes of the keypoint fixture, as the record of the data of its run keeps them
KEYPOINT_CLASS_NAMES = ["background", "triangle", "rectangle"]
FIXTURE_CLASS_KEYPOINTS = {"triangle": ["top", "left", "right"],
                           "rectangle": ["top_left", "top_right", "bottom_right", "bottom_left"]}


@pytest.mark.parametrize(argnames=("target_triangle_keypoints", "source_class_keypoints", "expected_error"),
                         argvalues=[(["top", "left", "right"], FIXTURE_CLASS_KEYPOINTS, None),
                                    (["left", "right", "top"], FIXTURE_CLASS_KEYPOINTS,
                                     "Channel 0 Predicts 'triangle.top' In The Source Run And 'triangle.left' In This "
                                     "Run"),
                                    (["top", "left", "right"], None, "Records No Keypoint Names")])
def test_keypoint_head_moves_between_runs_of_the_same_keypoint_channels(
        tmp_path: Path, target_triangle_keypoints: List[str], source_class_keypoints: Optional[Dict[str, List[str]]],
        expected_error: Optional[str]) -> None:
    """
    Verifies that the keypoint head of a keypoint detection run initialises the head of a run predicting the same
    keypoints on the same channels, and is refused, before any weight is loaded, for a run whose triangle lists the
    same keypoints in another order, whose tensors have the same shapes although its channel 0 predicts another
    keypoint, and for a source run whose record of its data does not name the keypoints of its classes (written before
    they were recorded).

    Args:
        tmp_path (Path): The pytest temporary directory.
        target_triangle_keypoints (List[str]): The keypoint names of the triangle of the run receiving the head.
        source_class_keypoints (Optional[Dict[str, List[str]]]): The keypoint names of the classes in the record of
            the source run, None for a record without keypoints.
        expected_error (Optional[str]): A fragment of the refusal, None when the head must be loaded.
    """
    source_configuration, _, _ = get_keypoint_model_configuration(higher_resolution=True, hidden_width=16)
    source_model = build_task_model(task="keypoint_detection", configuration=source_configuration, seed=0)
    write_source_run(run_folder=tmp_path / "source_run", configuration=source_configuration, model=source_model,
                     class_names=KEYPOINT_CLASS_NAMES, class_keypoints=source_class_keypoints)
    target_configuration = copy.deepcopy(source_configuration)
    target_configuration["Data"]["keypoint_categories"][1]["keypoints"] = target_triangle_keypoints
    target_model = build_task_model(task="keypoint_detection", configuration=target_configuration, seed=1)
    target_configuration = point_to_source_run(configuration=target_configuration, section_name="KeypointHead",
                                               source_folder=tmp_path / "source_run")

    # Same channels: every tensor of the head loaded; other channels or unknown ones: refused, the head untouched
    initial_state = {key: tensor.clone() for key, tensor in target_model.keypoint_head.state_dict().items()}
    if expected_error is None:
        load_pretrained_component(component=target_model.keypoint_head, section_name="KeypointHead",
                                  model_configuration=target_configuration, device=torch.device("cpu"),
                                  class_names=KEYPOINT_CLASS_NAMES)
        loaded_state = target_model.keypoint_head.state_dict()
        assert all(torch.equal(input=loaded_state[key], other=tensor)
                   for key, tensor in source_model.keypoint_head.state_dict().items())
    else:
        with pytest.raises(expected_exception=ValueError, match=expected_error):
            load_pretrained_component(component=target_model.keypoint_head, section_name="KeypointHead",
                                      model_configuration=target_configuration, device=torch.device("cpu"),
                                      class_names=KEYPOINT_CLASS_NAMES)
        assert all(torch.equal(input=initial_state[key], other=tensor)
                   for key, tensor in target_model.keypoint_head.state_dict().items())


# Layers of the heads left out of a transfer without their class layers, by section, as fragments of their tensor
# names: the classifier and per-class box regressor of the detection heads, the mask predictor of the mask heads, and
# the last convolutions of the keypoint head (hidden width 16, so index 1), whose transposed convolution of the higher
# resolution branch is partly random
LEFT_OUT_LAYER_FRAGMENTS = {"Detector": [".classifier.", ".bounding_box_regressor."],
                            "MaskHead": [".mask_predictor."],
                            "KeypointHead": ["prediction_convolution.1.", "higher_resolution_branch.heatmap_head.1.",
                                             "higher_resolution_branch.transposed_convolution."]}


def is_left_out_tensor(section_name: str, tensor_name: str) -> bool:
    """
    Tells whether a tensor of a head belongs to the layers its transfer without class layers leaves out
    (LEFT_OUT_LAYER_FRAGMENTS), for the checks of the tests below.

    Args:
        section_name (str): A key of LEFT_OUT_LAYER_FRAGMENTS.
        tensor_name (str): The name of the tensor in the state of the head, e.g. '5.classifier.weight'.

    Returns:
        bool: True for a tensor of a left out layer.
    """
    return any(fragment in f".{tensor_name}" for fragment in LEFT_OUT_LAYER_FRAGMENTS[section_name])


def test_detection_and_mask_heads_without_class_layers_initialise_a_run_of_other_classes(
        tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the detection heads and the mask heads of an instance segmentation run of 5 foreground classes,
    loaded without their class layers ('Detector.pretrained.class_layers' and 'MaskHead.pretrained.class_layers'
    False), initialise the heads of a run of 2 unrelated classes, which the transfer with the class layers refuses:
    every other tensor is copied exactly, the classifier, box regressor and mask predictor keep the random weights the
    new run was built with, and the console says so.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the transfers.
    """
    source_configuration, _, _ = get_detection_model_configuration(mask_head=True)
    source_segmenter = build_task_model(task="instance_segmentation", configuration=source_configuration, seed=0)
    write_source_run(run_folder=tmp_path / "instance_run", configuration=source_configuration, model=source_segmenter,
                     class_names=list_detection_class_names(configuration=source_configuration))
    target_configuration = copy.deepcopy(source_configuration)
    target_configuration["Data"]["label_dictionary"] = {"background": 0, "cat": 1, "dog": 2}
    target_class_names = list_detection_class_names(configuration=target_configuration)
    target_segmenter = build_task_model(task="instance_segmentation", configuration=target_configuration, seed=1)

    for section_name, component_name in [("Detector", "Detection Heads"), ("MaskHead", "Mask Heads")]:
        component = getattr(target_segmenter, TRANSFERABLE_COMPONENTS[section_name])
        initial_target_state = copy.deepcopy(component.state_dict())

        # Refused with the class layers, none of the source classes being a class of this run
        with pytest.raises(expected_exception=ValueError, match="Their Classes Differ"):
            load_pretrained_component(component=component, section_name=section_name, device=torch.device("cpu"),
                                      model_configuration=point_to_source_run(
                                          configuration=target_configuration, section_name=section_name,
                                          source_folder=tmp_path / "instance_run", class_layers=True),
                                      class_names=target_class_names)

        # Loaded without them: other layers of the source run, class layers of the new run
        load_pretrained_component(component=component, section_name=section_name, device=torch.device("cpu"),
                                  model_configuration=point_to_source_run(
                                      configuration=target_configuration, section_name=section_name,
                                      source_folder=tmp_path / "instance_run", class_layers=False),
                                  class_names=target_class_names)
        source_state = getattr(source_segmenter, TRANSFERABLE_COMPONENTS[section_name]).state_dict()
        target_state = component.state_dict()
        assert any(is_left_out_tensor(section_name=section_name, tensor_name=name) for name in target_state)
        for tensor_name, tensor_value in target_state.items():
            left_out = is_left_out_tensor(section_name=section_name, tensor_name=tensor_name)
            expected_value = initial_target_state[tensor_name] if left_out else source_state[tensor_name]
            assert torch.equal(input=tensor_value, other=expected_value), tensor_name
        assert (f"{component_name} : Loaded Without Their Class Layers, Which Keep The Random Initialisation Of This "
                f"Run For Their 3 Classes (The Source Run Predicts 6 Classes).") in capsys.readouterr().out


@pytest.mark.parametrize(argnames=("higher_resolution", "hidden_width", "expected_error"),
                         argvalues=[(True, 16, None), (True, 0, None), (False, 16, None),
                                    (False, 0, "Has No Layer Besides Its Class Layers")])
def test_keypoint_head_without_class_layers_initialises_a_run_of_other_keypoints(
        tmp_path: Path, higher_resolution: bool, hidden_width: int, expected_error: Optional[str],
        capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the keypoint head of a run predicting 7 keypoint channels (a triangle and a rectangle), loaded without
    its keypoint channel layers ('KeypointHead.pretrained.class_layers' False), initialises the head of a run of one
    pentagon of 5 keypoints, which the transfer with them refuses: the hidden convolution and the higher resolution
    branch are copied exactly, the last convolutions of both heads keep the random weights the new run was built with,
    and the transposed convolution of the branch takes the source rows of the features and keeps its random rows of
    the predictions. A head without hidden convolution nor higher resolution branch, holding nothing else, is refused.

    Args:
        tmp_path (Path): The pytest temporary directory.
        higher_resolution (bool): Whether both runs have the higher resolution branch.
        hidden_width (int): The hidden width of both heads, 0 for none.
        expected_error (Optional[str]): A fragment of the refusal, None when the head must be loaded.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the transfer.
    """
    source_configuration, _, _ = get_keypoint_model_configuration(higher_resolution=higher_resolution,
                                                                   hidden_width=hidden_width)
    source_model = build_task_model(task="keypoint_detection", configuration=source_configuration, seed=0)
    write_source_run(run_folder=tmp_path / "source_run", configuration=source_configuration, model=source_model,
                     class_names=KEYPOINT_CLASS_NAMES, class_keypoints=FIXTURE_CLASS_KEYPOINTS)
    target_configuration = copy.deepcopy(source_configuration)
    target_configuration["Data"]["label_dictionary"] = {"background": 0, 1: 1}
    target_configuration["Data"]["keypoint_categories"] = {1: {
        "name": "pentagon", "keypoints": [f"corner_{index}" for index in range(5)],
        "skeleton": [[index, (index + 1) % 5] for index in range(5)]}}
    target_model = build_task_model(task="keypoint_detection", configuration=target_configuration, seed=1)
    initial_target_state = copy.deepcopy(target_model.keypoint_head.state_dict())
    target_class_names = ["background", "pentagon"]

    # Refused with the keypoint channel layers, the channels of both runs differing
    with pytest.raises(expected_exception=ValueError):
        load_pretrained_component(component=target_model.keypoint_head, section_name="KeypointHead",
                                  model_configuration=point_to_source_run(
                                      configuration=target_configuration, section_name="KeypointHead",
                                      source_folder=tmp_path / "source_run", class_layers=True),
                                  device=torch.device("cpu"), class_names=target_class_names)
    transfer_configuration = point_to_source_run(configuration=target_configuration, section_name="KeypointHead",
                                                 source_folder=tmp_path / "source_run", class_layers=False)
    if expected_error is not None:
        with pytest.raises(expected_exception=ValueError, match=expected_error):
            load_pretrained_component(component=target_model.keypoint_head, section_name="KeypointHead",
                                      model_configuration=transfer_configuration, device=torch.device("cpu"),
                                      class_names=target_class_names)
        return

    # Loaded without them: the last convolutions as the new run built them, the other layers from the source run
    load_pretrained_component(component=target_model.keypoint_head, section_name="KeypointHead",
                              model_configuration=transfer_configuration, device=torch.device("cpu"),
                              class_names=target_class_names)
    last_index = 1 if hidden_width else 0
    left_out_prefixes = [f"prediction_convolution.{last_index}.",
                         f"higher_resolution_branch.heatmap_head.{last_index}."]
    source_state = source_model.keypoint_head.state_dict()
    target_state = target_model.keypoint_head.state_dict()
    for tensor_name, tensor_value in target_state.items():
        if any(tensor_name.startswith(prefix) for prefix in left_out_prefixes):
            assert torch.equal(input=tensor_value, other=initial_target_state[tensor_name]), tensor_name
        elif tensor_name == "higher_resolution_branch.transposed_convolution.weight":
            # Input rows of the features from the source run, rows of the predictions as the new run built them
            feature_width = tensor_value.shape[1]
            assert torch.equal(input=tensor_value[:feature_width], other=source_state[tensor_name][:feature_width])
            assert torch.equal(input=tensor_value[feature_width:],
                               other=initial_target_state[tensor_name][feature_width:])
        else:
            assert torch.equal(input=tensor_value, other=source_state[tensor_name]), tensor_name
    assert ("Keypoint Head : Loaded Without Its Keypoint Channel Layers, Which Keep The Random Initialisation Of This "
            "Run For Its 5 Keypoint Channels (The Source Run Predicts 7 Keypoint Channels).") in capsys.readouterr().out


@pytest.mark.parametrize(argnames="section_name", argvalues=["Detector", "MaskHead", "KeypointHead"])
def test_heads_loaded_without_their_class_layers_transfer_their_other_layers_only(section_name: str) -> None:
    """
    Verifies the transferred part of the detection, mask and keypoint heads loaded without their class layers, which
    the trainer freezes and fine-tunes at the learning rate multiplier of the transfer: every parameter of the heads
    but those of their class layers (and of the partly random transposed convolution of the keypoint head), which learn
    at the learning rate of the run, while heads loaded with their class layers are their own transferred part.

    Args:
        section_name (str): The section of the heads.
    """
    if section_name == "KeypointHead":
        configuration, _, _ = get_keypoint_model_configuration(higher_resolution=True, hidden_width=16)
        model = build_task_model(task="keypoint_detection", configuration=configuration, seed=0)
    else:
        configuration, _, _ = get_detection_model_configuration(mask_head=True)
        model = build_task_model(task="instance_segmentation", configuration=configuration, seed=0)
    component = getattr(model, TRANSFERABLE_COMPONENTS[section_name])
    assert select_transferred_part(component=component, section_name=section_name, number_loaded_blocks=None,
                                   with_class_layers=True) is component

    # Parameters of every layer but the left out ones
    transferred_part = select_transferred_part(component=component, section_name=section_name,
                                               number_loaded_blocks=None, with_class_layers=False)
    transferred_identifiers = [id(parameter) for parameter in transferred_part.parameters()]
    expected_identifiers = [id(parameter) for parameter_name, parameter in component.named_parameters()
                            if not is_left_out_tensor(section_name=section_name, tensor_name=parameter_name)]
    assert sorted(transferred_identifiers) == sorted(expected_identifiers)
    assert len(expected_identifiers) < len(list(component.parameters()))


def test_class_layer_flags_are_disabled() -> None:
    """
    Verifies that turning off the transfers of a configuration sets the class layer flags of the detection heads, the
    mask heads, the keypoint head and the segmentation decoder back to False, with their source runs and freezing.
    """
    instance_configuration, _, _ = get_detection_model_configuration(mask_head=True)
    keypoint_configuration, _, _ = get_keypoint_model_configuration()
    segmentation_configuration, _, _ = get_segmentation_model_configuration(upsampling="transposed")
    for configuration in [instance_configuration, keypoint_configuration, segmentation_configuration]:
        # Every transferable section pointing to a source run, with its class layers and frozen
        transferred_sections = [section_name for section_name in TRANSFERABLE_COMPONENTS
                                if section_name in configuration]
        for section_name in transferred_sections:
            configuration.update(point_to_source_run(configuration=configuration, section_name=section_name,
                                                     source_folder=Path("/runs/source")))
            configuration[section_name]["pretrained"]["freeze"] = True
            if section_name in PRETRAINED_CLASS_LAYER_KEYS:
                configuration[section_name]["pretrained"][PRETRAINED_CLASS_LAYER_KEYS[section_name]] = True

        disable_pretrained_settings(model_configuration=configuration)

        for section_name in transferred_sections:
            expected_settings = NO_BACKBONE_TRANSFER if section_name == "Backbone" else NO_TRANSFER
            if section_name in PRETRAINED_CLASS_LAYER_KEYS:
                expected_settings = {**NO_TRANSFER, PRETRAINED_CLASS_LAYER_KEYS[section_name]: False}
            assert configuration[section_name]["pretrained"] == expected_settings, section_name
