"""
Tests of the model transfers through the trainers of vision_training/trainers on synthetic datasets: a frozen backbone,
the backbone, temporal module and head of a sequence classifier, the classification backbone and segmentation decoder
of the semantic segmentation runs, the backbone, region proposal and detection heads of an object detection run, and
the averaged backbone and head of a classifier of 9 classes initialising a run of 20 classes, initialise a new run that
trains, saves and resumes correctly.
"""

import copy
import json
import pytest
import torch
import yaml

from pathlib import Path
from typing import Any, Dict, Optional

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.core.task_registry import get_trainer_class
from vision_training.data.synthetic.create_synthetic_dataset import SHAPE_CATEGORIES, create_synthetic_dataset
from vision_training.models.model_transfer import NEW_CLASS_BIAS_OFFSET
from vision_training.trainers.base_trainer import BaseTrainer
from vision_training.trainers.checkpointing import EXPONENTIAL_MOVING_AVERAGE_STATE_KEY
from vision_training.trainers.weight_averaging import AVERAGED_MODEL_STATE_KEY
from vision_training.utilities.os_utilities import format_path_link

# Synthetic configuration of every task of the transfer tests
CONFIGURATIONS_FOLDER = Path(__file__).parents[2] / "vision_training" / "configurations"
SYNTHETIC_CONFIGURATION_PATHS = {
    "classification": CONFIGURATIONS_FOLDER / "classification" / "configuration_synthetic_shapes.yaml",
    "sequence_classification": (CONFIGURATIONS_FOLDER / "sequence_classification" /
                                "configuration_synthetic_sequences.yaml"),
    "semantic_segmentation": CONFIGURATIONS_FOLDER / "semantic_segmentation" / "configuration_synthetic_shapes.yaml",
    "object_detection": CONFIGURATIONS_FOLDER / "object_detection" / "configuration_synthetic_shapes.yaml",
    "instance_segmentation": CONFIGURATIONS_FOLDER / "instance_segmentation" / "configuration_synthetic_shapes.yaml"}

# Image size of the transfer tests and number of frames of their sequences, small enough for a few quick iterations
TRANSFER_IMAGE_SIZE = 64
TRANSFER_NUMBER_FRAMES = 4

# Learning rate of every transfer run, set here so that the expected learning rates of the parameter groups do not
# depend on the learning rate of the synthetic configuration files
TRANSFER_LEARNING_RATE = 1.0e-3

# Image size of the detection tasks, whose anchors must fit in the images, with anchor scales fitting it, and the small
# mask heads of the instance segmentation runs
DETECTION_TRANSFER_IMAGE_SIZE = 256
DETECTION_TRANSFER_ANCHOR_SCALES = {"5": [16, 32, 64], "6": [32, 64, 128], "7": [64, 128, 256]}
DETECTION_TRANSFER_MASK_HEAD = {"roi_align_pool_size": 7, "convolutions": [1, 16]}

# Components of a sequence classifier initialised by the run they come from, by section
SEQUENCE_COMPONENTS = {"Backbone": "backbone", "TemporalModel": "temporal_module",
                       "ClassificationHead": "classification_head"}

# Components of an object detector initialising an instance segmenter, by section
DETECTION_COMPONENTS = {"Backbone": "backbone", "RegionProposal": "region_proposer_dictionary",
                        "Detector": "detector_heads_dictionary"}


def build_transfer_trainer(temporary_directory: Path, dataset_folder: Path, number_iterations: int,
                           pretrained_settings: Dict[str, Dict[str, Any]], configuration_name: str, task: str,
                           label_mapping_path: Optional[Path] = None, moving_average: bool = False) -> BaseTrainer:
    """
    Writes a copy of the synthetic configuration of a task (synthetic shapes classification, segmentation, object
    detection or instance segmentation, or synthetic sequences classification with windows of 4 frames) for a quick
    run on the generated dataset, with the given pretrained settings, and builds its trainer exactly like main.py
    does. The detection tasks run on 256 pixel images with anchors fitting them, and the instance segmentation with
    small mask heads. Another label mapping trains the run on some of the classes of the dataset only, and the moving
    average of the weights can be turned on.

    Args:
        temporary_directory (Path): The pytest temporary directory receiving the configuration and the experiments.
        dataset_folder (Path): The root folder of the generated synthetic dataset.
        number_iterations (int): The number of training iterations of the run.
        pretrained_settings (Dict[str, Dict[str, Any]]): The 'pretrained' settings replaced for the run, by section.
        configuration_name (str): The file name of the written configuration.
        task (str): "classification", "sequence_classification", "semantic_segmentation", "object_detection" or
            "instance_segmentation".
        label_mapping_path (Optional[Path]): The label mapping of the run, None for the one of the generated dataset.
            Defaults to None.
        moving_average (bool): Whether the run keeps a moving average of its weights. Defaults to False.

    Returns:
        BaseTrainer: The trainer of the run.
    """
    configuration = yaml.safe_load(stream=SYNTHETIC_CONFIGURATION_PATHS[task].read_text())

    # Run in the temporary directory, on the generated splits, saving at iteration 2 and at the end of the run
    configuration["ExperimentConfiguration"].update({"experiment_folder": str(temporary_directory / "experiments"),
                                                     "train_data_folder": str(dataset_folder / "train"),
                                                     "validation_data_folder": str(dataset_folder / "validation"),
                                                     "test_data_folder": str(dataset_folder / "test"),
                                                     "batch_size": 4, "information_dump": 2,
                                                     "weight_saving_iterations": 4,
                                                     "intermediate_weight_saving_iterations": 2,
                                                     "number_iterations": number_iterations,
                                                     "validation_iteration_ratio": 2, "resume_training": True,
                                                     "learning_rate": TRANSFER_LEARNING_RATE})
    configuration["ExperimentConfiguration"]["diagnosis_modes"].update({"compute_receptive_field": False,
                                                                        "compute_model_size": False,
                                                                        "view_input_images": False})
    configuration["Data"]["image_settings"]["size"] = TRANSFER_IMAGE_SIZE
    if "Anchors" in configuration:
        configuration["Data"]["image_settings"]["size"] = DETECTION_TRANSFER_IMAGE_SIZE
        for level, scales in DETECTION_TRANSFER_ANCHOR_SCALES.items():
            configuration["Anchors"]["scales_and_ratios"][level]["scales"] = scales
    if "MaskHead" in configuration:
        configuration["MaskHead"]["mask_resolution"] = 14
        for level in DETECTION_TRANSFER_ANCHOR_SCALES:
            configuration["MaskHead"]["head_configurations"][level] = DETECTION_TRANSFER_MASK_HEAD
    # Two loader workers per split, enough for the small generated dataset
    configuration["Data"]["loading"]["number_workers"] = 2
    if task == "sequence_classification":
        configuration["Data"]["sequence_settings"].update({"sequence_length": TRANSFER_NUMBER_FRAMES,
                                                           "window_stride": TRANSFER_NUMBER_FRAMES})
    if label_mapping_path is None:
        label_mapping_path = dataset_folder / "label_mapping.json"
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(label_mapping_path)
    configuration["Optimization"]["exponential_moving_average"]["activated"] = moving_average
    for section_name, section_settings in pretrained_settings.items():
        configuration[section_name]["pretrained"].update(section_settings)

    configuration_path = temporary_directory / configuration_name
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    return get_trainer_class(task=task)(experiment_configuration=experiment_configuration,
                                        model_configuration=model_configuration,
                                        configuration_path=configuration_path)


def test_trained_backbone_initialises_a_frozen_run(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the backbone transfer through the trainers: a run initialised by the backbone of another run starts with
    exactly its weights in its own folder (named with PT and FRZ), its frozen backbone keeps its weights and batch
    normalization statistics while its head learns, its checkpoints record the source run, and resuming it does not
    load the source backbone again but keeps it frozen. A run fine-tuning the transferred backbone ten times slower
    gets a second optimizer parameter group, and resuming it with another learning rate multiplier keeps its folder
    and its groups while the backbone learns at the new rate.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainers.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=TRANSFER_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 4}, maximum_objects=1,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="classification")
    no_transfer = {"experiment_folder": None, "iteration": None, "freeze": False, "learning_rate_multiplier": 1.0}

    # Source run trained from random weights, saved at iterations 2 and 3
    source_trainer = build_transfer_trainer(temporary_directory=tmp_path, dataset_folder=dataset_folder,
                                            number_iterations=3, pretrained_settings={"Backbone": no_transfer},
                                            configuration_name="source.yaml", task="classification")
    source_trainer.run_training_loop()
    source_backbone_state = {parameter_name[len("backbone."):]: parameter_value for parameter_name, parameter_value in
                             torch.load(f=source_trainer.weights_directory / "Iteration_3" / "model_000003.pt",
                                        map_location="cpu", weights_only=True)["model_state"].items()
                             if parameter_name.startswith("backbone.")}

    # Frozen run initialised by the source backbone, in its own folder
    frozen_settings = {**no_transfer, "experiment_folder": str(source_trainer.experiment_folder), "freeze": True}
    frozen_trainer = build_transfer_trainer(temporary_directory=tmp_path, dataset_folder=dataset_folder,
                                            number_iterations=3, pretrained_settings={"Backbone": frozen_settings},
                                            configuration_name="frozen.yaml", task="classification")
    assert frozen_trainer.run_name == source_trainer.run_name.replace("2x64.2x128_", "2x64.2x128-PT-FRZ_")
    assert frozen_trainer.pretrained_checkpoint_paths == {
        "Backbone": source_trainer.weights_directory / "Iteration_3" / "model_000003.pt", "ClassificationHead": None}
    for parameter_name, parameter_value in frozen_trainer.model.backbone.state_dict().items():
        assert torch.equal(input=parameter_value.cpu(), other=source_backbone_state[parameter_name])

    # Training changes the head only, the frozen backbone keeping its weights and statistics
    head_state_before = copy.deepcopy(frozen_trainer.model.classification_head.state_dict())
    frozen_trainer.run_training_loop()
    for parameter_name, parameter_value in frozen_trainer.model.backbone.state_dict().items():
        assert torch.equal(input=parameter_value.cpu(), other=source_backbone_state[parameter_name]), parameter_name
    assert any(not torch.equal(input=parameter_value, other=head_state_before[parameter_name])
               for parameter_name, parameter_value in frozen_trainer.model.classification_head.state_dict().items())
    frozen_checkpoint = torch.load(f=frozen_trainer.weights_directory / "Iteration_3" / "model_000003.pt",
                                   map_location="cpu", weights_only=True)
    assert frozen_checkpoint["pretrained_backbone"] == str(source_trainer.experiment_folder)
    assert frozen_checkpoint["pretrained_classification_head"] is None

    # Resuming the frozen run keeps its own weights, without loading the source backbone again
    resumed_trainer = build_transfer_trainer(temporary_directory=tmp_path, dataset_folder=dataset_folder,
                                             number_iterations=1, pretrained_settings={"Backbone": frozen_settings},
                                             configuration_name="frozen.yaml", task="classification")
    assert resumed_trainer.start_iteration == 4 and resumed_trainer.pretrained_checkpoint_paths["Backbone"] is None
    assert "The Pretrained Backbone Is Not Loaded Again" in capsys.readouterr().out
    assert all(not parameter.requires_grad for parameter in resumed_trainer.model.backbone.parameters())

    # Fine-tuning run: the transferred backbone learns ten times slower than the head, saved at iteration 2
    fine_tuning_settings = {**no_transfer, "experiment_folder": str(source_trainer.experiment_folder),
                            "learning_rate_multiplier": 0.1}
    fine_tuning_trainer = build_transfer_trainer(temporary_directory=tmp_path, dataset_folder=dataset_folder,
                                                 number_iterations=2,
                                                 pretrained_settings={"Backbone": fine_tuning_settings},
                                                 configuration_name="fine_tuning.yaml", task="classification")
    assert "-PT_" in fine_tuning_trainer.run_name and "BLR" not in fine_tuning_trainer.run_name
    assert [parameter_group["lr"] for parameter_group in fine_tuning_trainer.optimizer.param_groups] == pytest.approx(
        expected=[1e-3, 1e-4])
    assert len(fine_tuning_trainer.optimizer.param_groups[1]["params"]) == len(
        list(fine_tuning_trainer.model.backbone.parameters()))
    fine_tuning_trainer.run_training_loop()

    # Resuming it with another multiplier keeps its folder and the layout of its optimizer groups, and the backbone
    # learns at the new rate from the next step on
    resumed_fine_tuning_trainer = build_transfer_trainer(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=3,
        pretrained_settings={"Backbone": {**fine_tuning_settings, "learning_rate_multiplier": 1.0}},
        configuration_name="fine_tuning.yaml", task="classification")
    assert resumed_fine_tuning_trainer.run_name == fine_tuning_trainer.run_name
    assert resumed_fine_tuning_trainer.start_iteration == 3
    assert resumed_fine_tuning_trainer.base_learning_rates == pytest.approx(expected=[1e-3, 1e-3])
    resumed_fine_tuning_trainer.run_training_loop()
    resumed_parameter_groups = resumed_fine_tuning_trainer.optimizer.param_groups
    assert len(resumed_parameter_groups) == 2
    assert resumed_parameter_groups[1]["lr"] == pytest.approx(expected=resumed_parameter_groups[0]["lr"])


def test_sequence_components_initialise_a_partly_frozen_run(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the transfer of every component of a sequence classifier through the trainers: a run initialised by the
    backbone (twice slower), the frozen GRU temporal module and the head (ten times slower) of another run starts
    with exactly their weights, in a folder named after the transfers, with one optimizer group per component
    learning at its own speed. Its training keeps the frozen GRU fixed while the backbone before it learns, which
    needs the recurrent layers in training mode on the GPU, its checkpoints record every source run, and resuming it
    loads nothing again. The launch information prints the settings of every transferred component, without learning
    rate for the frozen one, and the resume reports its changed number of iterations. A run keeping the
    normalization parameters of the backbone fixed trains every other backbone parameter, and a run freezing every
    component is refused.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainers.
    """
    dataset_folder = tmp_path / "synthetic_sequences"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=TRANSFER_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 4}, maximum_objects=1,
                             minimum_frames=TRANSFER_NUMBER_FRAMES, maximum_frames=TRANSFER_NUMBER_FRAMES,
                             number_workers=2, task="sequence_classification")

    # Source run trained from random weights, saved at iterations 2 and 3
    source_trainer = build_transfer_trainer(temporary_directory=tmp_path, dataset_folder=dataset_folder,
                                            number_iterations=3, pretrained_settings={},
                                            configuration_name="source.yaml", task="sequence_classification")
    source_trainer.run_training_loop()
    source_checkpoint_path = source_trainer.weights_directory / "Iteration_3" / "model_000003.pt"
    source_state = torch.load(f=source_checkpoint_path, map_location="cpu", weights_only=True)["model_state"]
    source_folder = str(source_trainer.experiment_folder)

    # Run initialised by every component of the source run, in its own folder named after the transfers
    transfer_settings = {"Backbone": {"experiment_folder": source_folder, "learning_rate_multiplier": 0.5},
                         "TemporalModel": {"experiment_folder": source_folder, "freeze": True},
                         "ClassificationHead": {"experiment_folder": source_folder, "learning_rate_multiplier": 0.1}}
    transfer_trainer = build_transfer_trainer(temporary_directory=tmp_path, dataset_folder=dataset_folder,
                                              number_iterations=3, pretrained_settings=transfer_settings,
                                              configuration_name="transfer.yaml", task="sequence_classification")
    model = transfer_trainer.model
    assert "-PT_T4_GRU1x128-BI-TPT-TFRZ_HAVG.64-HPT_" in transfer_trainer.run_name
    assert transfer_trainer.pretrained_checkpoint_paths == {section_name: source_checkpoint_path
                                                            for section_name in SEQUENCE_COMPONENTS}
    for tensor_name, tensor_value in model.state_dict().items():
        if tensor_name.split(".")[0] in SEQUENCE_COMPONENTS.values():
            assert torch.equal(input=tensor_value.cpu(), other=source_state[tensor_name]), tensor_name

    # The other parameters (the frozen temporal module) learn at the learning rate, the backbone and the head at
    # their own rates
    parameter_groups = transfer_trainer.optimizer.param_groups
    assert [parameter_group["lr"] for parameter_group in parameter_groups] == pytest.approx(expected=[1e-3, 5e-4, 1e-4])
    assert [len(parameter_group["params"]) for parameter_group in parameter_groups] == [
        len(list(getattr(model, attribute_name).parameters()))
        for attribute_name in ["temporal_module", "backbone", "classification_head"]]

    # Training keeps the frozen GRU fixed while the backbone before it learns
    temporal_state_before = copy.deepcopy(model.temporal_module.state_dict())
    backbone_state_before = copy.deepcopy(model.backbone.state_dict())
    transfer_trainer.run_training_loop()
    for tensor_name, tensor_value in model.temporal_module.state_dict().items():
        assert torch.equal(input=tensor_value, other=temporal_state_before[tensor_name]), tensor_name
    assert any(not torch.equal(input=tensor_value, other=backbone_state_before[tensor_name])
               for tensor_name, tensor_value in model.backbone.state_dict().items())
    transfer_checkpoint = torch.load(f=transfer_trainer.weights_directory / "Iteration_3" / "model_000003.pt",
                                     map_location="cpu", weights_only=True)
    assert [transfer_checkpoint[f"pretrained_{attribute_name}"] for attribute_name in SEQUENCE_COMPONENTS.values()] == [
        source_folder] * 3

    # Resuming loads nothing again and keeps the temporal module frozen
    resumed_trainer = build_transfer_trainer(temporary_directory=tmp_path, dataset_folder=dataset_folder,
                                             number_iterations=1, pretrained_settings=transfer_settings,
                                             configuration_name="transfer.yaml", task="sequence_classification")
    assert resumed_trainer.start_iteration == 4
    assert list(resumed_trainer.pretrained_checkpoint_paths.values()) == [None] * 3
    assert all(not parameter.requires_grad for parameter in resumed_trainer.model.temporal_module.parameters())
    console_output = capsys.readouterr().out
    assert "The Pretrained Temporal Module Is Not Loaded Again" in console_output
    assert "- Changed Settings        : Since Iteration 3" in console_output
    assert "    - ExperimentConfiguration.number_iterations : 3 -> 1" in console_output

    # Launch information of every transferred component: source run and iteration, freezes and learning rate
    console_lines = console_output.splitlines()
    source_link = format_path_link(path=source_folder)
    for expected_line in [f"- Pretrained Backbone     : {source_link}", "    - Norm. Layers Frozen : False",
                          "    - Learning Rate       : 0.0005 (multiplier 0.5 x 0.001)",
                          f"- Pretrained Class. Head  : {source_link}",
                          "    - Learning Rate       : 0.0001 (multiplier 0.1 x 0.001)"]:
        assert any(expected_line in console_line for console_line in console_lines), expected_line

    # The frozen temporal module does not learn, so its block has no learning rate line
    temporal_block_index = next(index for index, console_line in enumerate(console_lines)
                                if f"- Pretrained Temp. Module : {source_link}" in console_line)
    assert "    - Source Iteration    : last saved checkpoint" in console_lines[temporal_block_index + 1]
    assert "    - Frozen              : True" in console_lines[temporal_block_index + 2]
    assert "- Pretrained Class. Head" in console_lines[temporal_block_index + 3]

    # A new run without checkpoint says which components it took from other runs, the source run says it is new
    assert ("The New Model Starts From Pretrained Components Of Other Runs : Backbone, Temporal Module, "
            "Classification Head.") in console_output
    assert "We Generate A New Model That Will Be Trained From Scratch." in console_output
    assert "Error encountered during iteration" not in console_output and "Critical Error" not in console_output

    # Normalization parameters of the transferred backbone kept fixed while its other parameters learn
    normalization_trainer = build_transfer_trainer(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=2,
        pretrained_settings={"Backbone": {"experiment_folder": source_folder,
                                          "freeze_normalization_parameters": True}},
        configuration_name="normalization.yaml", task="sequence_classification")
    assert "-PT-FBN_" in normalization_trainer.run_name
    backbone = normalization_trainer.model.backbone
    batch_normalization_parameters = [parameter for module in backbone.modules()
                                      if isinstance(module, torch.nn.BatchNorm2d) for parameter in module.parameters()]
    assert batch_normalization_parameters and all(not parameter.requires_grad
                                                  for parameter in batch_normalization_parameters)
    assert sum(parameter.requires_grad for parameter in backbone.parameters()) == len(
        list(backbone.parameters())) - len(batch_normalization_parameters)

    # A run freezing every component would learn nothing
    with pytest.raises(expected_exception=ValueError, match="Every Parameter Of The Model Is Frozen"):
        build_transfer_trainer(temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=1,
                               pretrained_settings={section_name: {"experiment_folder": source_folder, "freeze": True}
                                                    for section_name in SEQUENCE_COMPONENTS},
                               configuration_name="all_frozen.yaml", task="sequence_classification")


def test_classification_backbone_and_segmentation_decoder_initialise_segmentation_runs(
        tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the transfers of the semantic segmentation through the trainers: the backbone of a classification run
    initialises a frozen segmentation run, which keeps it fixed while its decoder learns and records its source run in
    its checkpoints, and the decoder of that segmentation run initialises another segmentation run, which starts with
    exactly its weights and fine-tunes it ten times slower in its own optimizer group.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainers.
    """
    classification_dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(classification_dataset_folder), image_size=TRANSFER_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 4}, maximum_objects=1,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="classification")
    segmentation_dataset_folder = tmp_path / "synthetic_segmentation_shapes"
    create_synthetic_dataset(output_folder=str(segmentation_dataset_folder), image_size=TRANSFER_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 4}, maximum_objects=3,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="semantic_segmentation")

    # Classification source run trained from random weights, saved at iterations 2 and 3
    classification_trainer = build_transfer_trainer(temporary_directory=tmp_path,
                                                    dataset_folder=classification_dataset_folder, number_iterations=3,
                                                    pretrained_settings={}, configuration_name="classification.yaml",
                                                    task="classification")
    classification_trainer.run_training_loop()
    classification_folder = str(classification_trainer.experiment_folder)
    source_backbone_state = {
        tensor_name[len("backbone."):]: tensor_value for tensor_name, tensor_value in
        torch.load(f=classification_trainer.weights_directory / "Iteration_3" / "model_000003.pt",
                   map_location="cpu", weights_only=True)["model_state"].items() if tensor_name.startswith("backbone.")}

    # Segmentation run initialised by the frozen classification backbone, whose decoder learns
    segmentation_trainer = build_transfer_trainer(
        temporary_directory=tmp_path, dataset_folder=segmentation_dataset_folder, number_iterations=3,
        pretrained_settings={"Backbone": {"experiment_folder": classification_folder, "freeze": True}},
        configuration_name="segmentation.yaml", task="semantic_segmentation")
    assert "-PT-FRZ_UTC-" in segmentation_trainer.run_name
    model = segmentation_trainer.model
    decoder_state_before = copy.deepcopy(model.segmentation_decoder.state_dict())
    segmentation_trainer.run_training_loop()
    for tensor_name, tensor_value in model.backbone.state_dict().items():
        assert torch.equal(input=tensor_value.cpu(), other=source_backbone_state[tensor_name]), tensor_name
    assert any(not torch.equal(input=tensor_value, other=decoder_state_before[tensor_name])
               for tensor_name, tensor_value in model.segmentation_decoder.state_dict().items())
    segmentation_checkpoint = torch.load(f=segmentation_trainer.weights_directory / "Iteration_3" / "model_000003.pt",
                                         map_location="cpu", weights_only=True)
    assert segmentation_checkpoint["pretrained_backbone"] == classification_folder
    assert segmentation_checkpoint["pretrained_segmentation_decoder"] is None

    # Segmentation run initialised by the decoder of the previous one, fine-tuned ten times slower in its own group
    decoder_trainer = build_transfer_trainer(
        temporary_directory=tmp_path, dataset_folder=segmentation_dataset_folder, number_iterations=2,
        pretrained_settings={"SegmentationDecoder": {"experiment_folder": str(segmentation_trainer.experiment_folder),
                                                     "learning_rate_multiplier": 0.1}},
        configuration_name="decoder.yaml", task="semantic_segmentation")
    assert "-D2x64.2x64.2x32.2x32.1x16-DPT_" in decoder_trainer.run_name
    for tensor_name, tensor_value in decoder_trainer.model.segmentation_decoder.state_dict().items():
        assert torch.equal(input=tensor_value.cpu(),
                           other=segmentation_checkpoint["model_state"][f"segmentation_decoder.{tensor_name}"])
    parameter_groups = decoder_trainer.optimizer.param_groups
    assert [parameter_group["lr"] for parameter_group in parameter_groups] == pytest.approx(expected=[1e-3, 1e-4])
    assert len(parameter_groups[1]["params"]) == len(list(decoder_trainer.model.segmentation_decoder.parameters()))
    decoder_trainer.run_training_loop()

    console_output = capsys.readouterr().out
    assert "Segmentation Decoder Initialised From" in console_output
    assert "Error encountered during iteration" not in console_output and "Critical Error" not in console_output


def test_detection_run_initialises_a_frozen_instance_segmentation_run(tmp_path: Path,
                                                                     capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the transfers of the detection tasks through the trainers: the backbone, region proposal heads and
    detection heads of an object detection run initialise a frozen instance segmentation run (named with PT-FRZ,
    RPT-RFRZ and DPT-DFRZ), which starts with exactly their weights, keeps them and their batch normalization
    statistics while its mask heads learn, records its source run in its checkpoints, and holds every other parameter
    in a single optimizer group.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainers.
    """
    dataset_folder = tmp_path / "synthetic_instance_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=DETECTION_TRANSFER_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 4}, maximum_objects=3,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="instance_segmentation")

    # Object detection source run trained from random weights on the boxes of the dataset, saved at iterations 2 and 3
    detection_trainer = build_transfer_trainer(temporary_directory=tmp_path, dataset_folder=dataset_folder,
                                               number_iterations=3, pretrained_settings={},
                                               configuration_name="detection.yaml", task="object_detection")
    detection_trainer.run_training_loop()
    detection_folder = str(detection_trainer.experiment_folder)
    detection_state = torch.load(f=detection_trainer.weights_directory / "Iteration_3" / "model_000003.pt",
                                 map_location="cpu", weights_only=True)["model_state"]

    # Instance segmentation run initialised by the frozen backbone and heads of the detection run
    frozen_settings = {"experiment_folder": detection_folder, "freeze": True}
    segmentation_trainer = build_transfer_trainer(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=3,
        pretrained_settings={section_name: frozen_settings for section_name in DETECTION_COMPONENTS},
        configuration_name="instance_segmentation.yaml", task="instance_segmentation")
    assert "-PT-FRZ_L567_" in segmentation_trainer.run_name
    assert "-R2x64-RPT-RFRZ_" in segmentation_trainer.run_name and "-DPT-DFRZ_M" in segmentation_trainer.run_name
    assert len(segmentation_trainer.optimizer.param_groups) == 1
    model = segmentation_trainer.model
    mask_heads_before = copy.deepcopy(model.mask_heads_dictionary.state_dict())
    segmentation_trainer.run_training_loop()

    # Frozen components identical to those of the detection run, batch statistics included, the mask heads learning
    for attribute_name in DETECTION_COMPONENTS.values():
        for tensor_name, tensor_value in getattr(model, attribute_name).state_dict().items():
            assert torch.equal(input=tensor_value.cpu(), other=detection_state[f"{attribute_name}.{tensor_name}"]), \
                f"{attribute_name}.{tensor_name}"
    assert any(not torch.equal(input=tensor_value, other=mask_heads_before[tensor_name])
               for tensor_name, tensor_value in model.mask_heads_dictionary.state_dict().items())
    segmentation_checkpoint = torch.load(f=segmentation_trainer.weights_directory / "Iteration_3" / "model_000003.pt",
                                         map_location="cpu", weights_only=True)
    assert [segmentation_checkpoint[f"pretrained_{attribute_name}"] for attribute_name in
            DETECTION_COMPONENTS.values()] == [detection_folder] * 3

    console_output = capsys.readouterr().out
    assert "Pretrained RPN Heads" in console_output and "Pretrained Det. Heads" in console_output
    assert "Detection Heads Initialised From" in console_output
    assert "Error encountered during iteration" not in console_output and "Critical Error" not in console_output


def test_averaged_run_of_9_classes_initialises_a_run_of_20_classes(tmp_path: Path,
                                                                   capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the transfer to a run adding classes through the trainers: a source run trained on the first 9 classes of
    the synthetic shapes with a moving average of its weights records its class names in its data.json, and its
    backbone and classification head initialise a run of the 20 classes keeping a moving average too. Both components
    start from the averaged weights of the source checkpoint, the classifier keeping the rows of the 9 source classes
    and adding 11 classes after them with zero weights and a bias NEW_CLASS_BIAS_OFFSET below the mean source bias,
    and the run trains and saves.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainers.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=TRANSFER_IMAGE_SIZE,
                             split_sizes={"train": 40, "validation": 20, "test": 20}, maximum_objects=1,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="classification")
    no_transfer = {"experiment_folder": None, "iteration": None, "freeze": False, "learning_rate_multiplier": 1.0}

    # Source run on the first 9 categories, the images of the other categories being left out
    source_mapping_path = tmp_path / "label_mapping_9_classes.json"
    source_mapping_path.write_text(data=json.dumps(obj={str(category_identifier): category_identifier - 1
                                                        for category_identifier in range(1, 10)}))
    source_trainer = build_transfer_trainer(temporary_directory=tmp_path, dataset_folder=dataset_folder,
                                            number_iterations=3, configuration_name="source.yaml",
                                            pretrained_settings={"Backbone": no_transfer,
                                                                 "ClassificationHead": no_transfer},
                                            task="classification", label_mapping_path=source_mapping_path,
                                            moving_average=True)
    source_trainer.run_training_loop()
    source_data_record = json.loads(s=(source_trainer.experiment_folder / "data.json").read_text())
    assert source_data_record["class_names"] == list(SHAPE_CATEGORIES)[:9]
    averaged_state = torch.load(f=source_trainer.weights_directory / "Iteration_3" / "model_000003.pt",
                                map_location="cpu",
                                weights_only=True)[EXPONENTIAL_MOVING_AVERAGE_STATE_KEY][AVERAGED_MODEL_STATE_KEY]
    capsys.readouterr()

    # Run of the 20 classes initialised by the averaged backbone and head of the source run
    transfer_settings = {**no_transfer, "experiment_folder": str(source_trainer.experiment_folder)}
    target_trainer = build_transfer_trainer(temporary_directory=tmp_path, dataset_folder=dataset_folder,
                                            number_iterations=2, configuration_name="target.yaml",
                                            pretrained_settings={"Backbone": transfer_settings,
                                                                 "ClassificationHead": transfer_settings},
                                            task="classification", moving_average=True)
    assert source_trainer.run_name.endswith("_MC9") and target_trainer.run_name.endswith("-HPT_MC20")
    console_output = capsys.readouterr().out
    assert ("Classification Head : 9 Classes Of The Source Run Kept, 11 New Classes Added After Them, Built From The "
            "Source Classes.") in console_output
    assert "Backbone Initialised From The Averaged Weights Of" in console_output
    assert "Classification Head Initialised From The Averaged Weights Of" in console_output

    # Averaged weights copied, the classifier rows of the source classes included, and new classes starting quiet
    target_state = {tensor_name: tensor_value.cpu() for tensor_name, tensor_value in
                    target_trainer.model.state_dict().items()}
    for tensor_name, tensor_value in averaged_state.items():
        if tensor_name.startswith("classification_head.classifier."):
            assert torch.equal(input=target_state[tensor_name][:9], other=tensor_value), tensor_name
        elif tensor_name.startswith("backbone.") or tensor_name.startswith("classification_head."):
            assert torch.equal(input=target_state[tensor_name], other=tensor_value), tensor_name
    source_bias = averaged_state["classification_head.classifier.bias"]
    assert torch.count_nonzero(input=target_state["classification_head.classifier.weight"][9:]) == 0
    assert torch.allclose(input=target_state["classification_head.classifier.bias"][9:],
                          other=(source_bias.mean() - NEW_CLASS_BIAS_OFFSET).expand(11))

    # The run of 20 classes trains and saves
    target_trainer.run_training_loop()
    assert (target_trainer.weights_directory / "Iteration_2" / "model_000002.pt").is_file()
