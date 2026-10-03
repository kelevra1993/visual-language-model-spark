"""
Tests of the model transfers through the trainers of vision_training/trainers on synthetic datasets: a frozen backbone,
and the backbone, temporal module and head of a sequence classifier, initialise a new run that trains, saves and resumes
correctly.
"""

import copy
import pytest
import torch
import yaml

from pathlib import Path
from typing import Any, Dict

from vision_training.configuration.configuration_loader import load_experiment_configuration
from vision_training.core.task_registry import get_trainer_class
from vision_training.data.synthetic.create_synthetic_dataset import create_synthetic_dataset
from vision_training.trainers.base_trainer import BaseTrainer

# Synthetic configuration of every task of the transfer tests
CONFIGURATIONS_FOLDER = Path(__file__).parents[2] / "vision_training" / "configurations"
SYNTHETIC_CONFIGURATION_PATHS = {
    "classification": CONFIGURATIONS_FOLDER / "classification" / "configuration_synthetic_shapes.yaml",
    "sequence_classification": (CONFIGURATIONS_FOLDER / "sequence_classification" /
                                "configuration_synthetic_sequences.yaml")}

# Image size of the transfer tests and number of frames of their sequences, small enough for a few quick iterations
TRANSFER_IMAGE_SIZE = 64
TRANSFER_NUMBER_FRAMES = 4

# Components of a sequence classifier initialised by the run they come from, by section
SEQUENCE_COMPONENTS = {"Backbone": "backbone", "TemporalModel": "temporal_module",
                       "ClassificationHead": "classification_head"}


def build_transfer_trainer(temporary_directory: Path, dataset_folder: Path, number_iterations: int,
                           pretrained_settings: Dict[str, Dict[str, Any]], configuration_name: str,
                           task: str) -> BaseTrainer:
    """
    Writes a copy of the synthetic configuration of a task (synthetic shapes classification, or synthetic sequences
    classification with windows of 4 frames) for a quick run on the generated dataset, with the given pretrained
    settings, and builds its trainer exactly like main.py does.

    Args:
        temporary_directory (Path): The pytest temporary directory receiving the configuration and the experiments.
        dataset_folder (Path): The root folder of the generated synthetic dataset.
        number_iterations (int): The number of training iterations of the run.
        pretrained_settings (Dict[str, Dict[str, Any]]): The 'pretrained' settings replaced for the run, by section.
        configuration_name (str): The file name of the written configuration.
        task (str): "classification" or "sequence_classification".

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
                                                     "validation_iteration_ratio": 2, "resume_training": True})
    configuration["ExperimentConfiguration"]["diagnosis_modes"].update({"compute_receptive_field": False,
                                                                        "compute_model_size": False,
                                                                        "view_input_images": False})
    configuration["Data"]["image_settings"]["size"] = TRANSFER_IMAGE_SIZE
    # Two loader workers per split, enough for the small generated dataset
    configuration["Data"]["loading"]["number_workers"] = 2
    if task == "sequence_classification":
        configuration["Data"]["sequence_settings"].update({"sequence_length": TRANSFER_NUMBER_FRAMES,
                                                           "window_stride": TRANSFER_NUMBER_FRAMES})
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(dataset_folder / "label_mapping.json")
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
    gets a second optimizer parameter group.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainers.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=TRANSFER_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 4}, maximum_objects=1,
                             number_frames=1, number_workers=2, task="classification")
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
    assert "the pretrained backbone is not loaded again" in capsys.readouterr().out
    assert all(not parameter.requires_grad for parameter in resumed_trainer.model.backbone.parameters())

    # Fine-tuning run: the transferred backbone learns ten times slower than the head
    fine_tuning_settings = {**no_transfer, "experiment_folder": str(source_trainer.experiment_folder),
                            "learning_rate_multiplier": 0.1}
    fine_tuning_trainer = build_transfer_trainer(temporary_directory=tmp_path, dataset_folder=dataset_folder,
                                                 number_iterations=1,
                                                 pretrained_settings={"Backbone": fine_tuning_settings},
                                                 configuration_name="fine_tuning.yaml", task="classification")
    assert "-PT-BLR0.1_" in fine_tuning_trainer.run_name
    assert [parameter_group["lr"] for parameter_group in fine_tuning_trainer.optimizer.param_groups] == pytest.approx(
        expected=[1e-3, 1e-4])
    assert len(fine_tuning_trainer.optimizer.param_groups[1]["params"]) == len(
        list(fine_tuning_trainer.model.backbone.parameters()))


def test_sequence_components_initialise_a_partly_frozen_run(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the transfer of every component of a sequence classifier through the trainers: a run initialised by the
    backbone (twice slower), the frozen GRU temporal module and the head (ten times slower) of another run starts
    with exactly their weights, in a folder named after the transfers, with one optimizer group per component
    learning at its own speed. Its training keeps the frozen GRU fixed while the backbone before it learns, which
    needs the recurrent layers in training mode on the GPU, its checkpoints record every source run, and resuming it
    loads nothing again. A run keeping the normalization parameters of the backbone fixed trains every other
    backbone parameter, and a run freezing every component is refused.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainers.
    """
    dataset_folder = tmp_path / "synthetic_sequences"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=TRANSFER_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 4}, maximum_objects=1,
                             number_frames=TRANSFER_NUMBER_FRAMES, number_workers=2, task="sequence_classification")

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
    assert "-PT-BLR0.5_T4_GRU1x128-BI-TPT-TFRZ_HAVG.64-DO0.2-HPT-HLR0.1_" in transfer_trainer.run_name
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
    assert "the pretrained temporal module is not loaded again" in console_output
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
    with pytest.raises(expected_exception=ValueError, match="Every parameter of the model is frozen"):
        build_transfer_trainer(temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=1,
                               pretrained_settings={section_name: {"experiment_folder": source_folder, "freeze": True}
                                                    for section_name in SEQUENCE_COMPONENTS},
                               configuration_name="all_frozen.yaml", task="sequence_classification")
