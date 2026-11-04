"""
Tests of the model transfers through the trainers of vision_training/trainers on a synthetic dataset: the backbone of a
classification run, frozen or fine-tuned at its own learning rate, initialises a new run that trains, saves and resumes
correctly.
"""

import copy
import pytest
import torch
import yaml

from pathlib import Path
from typing import Any, Dict, Optional

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.core.task_registry import get_trainer_class
from vision_training.data.synthetic.create_synthetic_dataset import create_synthetic_dataset
from vision_training.trainers.base_trainer import BaseTrainer

# Synthetic configuration of every task of the transfer tests
CONFIGURATIONS_FOLDER = Path(__file__).parents[2] / "vision_training" / "configurations"
SYNTHETIC_CONFIGURATION_PATHS = {
    "classification": CONFIGURATIONS_FOLDER / "classification" / "configuration_synthetic_shapes.yaml",
    "sequence_classification": (CONFIGURATIONS_FOLDER / "sequence_classification" /
                                "configuration_synthetic_sequences.yaml"),
    "semantic_segmentation": CONFIGURATIONS_FOLDER / "semantic_segmentation" / "configuration_synthetic_shapes.yaml",
    "object_detection": CONFIGURATIONS_FOLDER / "object_detection" / "configuration_synthetic_shapes.yaml",
    "instance_segmentation": CONFIGURATIONS_FOLDER / "instance_segmentation" / "configuration_synthetic_shapes.yaml",
    "self_supervised": CONFIGURATIONS_FOLDER / "self_supervised" / "configuration_synthetic_shapes.yaml"}

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


def build_transfer_trainer(temporary_directory: Path, dataset_folder: Path, number_iterations: int,
                           pretrained_settings: Dict[str, Dict[str, Any]], configuration_name: str, task: str,
                           label_mapping_path: Optional[Path] = None, moving_average: bool = False) -> BaseTrainer:
    """
    Writes a copy of the synthetic configuration of a task (synthetic shapes classification, segmentation, object
    detection, instance segmentation or self-supervised learning, or synthetic sequences classification with
    sequences of at most 4 frames) for a quick run on the generated dataset, with the given pretrained settings, and
    builds its trainer exactly like main.py does. The detection tasks run on 256 pixel images with anchors fitting
    them, and the instance segmentation with small mask heads. Another label mapping trains the run on some of the
    classes of the dataset only, and the moving average of the weights can be turned on.

    Args:
        temporary_directory (Path): The pytest temporary directory receiving the configuration and the experiments.
        dataset_folder (Path): The root folder of the generated synthetic dataset.
        number_iterations (int): The number of training iterations of the run.
        pretrained_settings (Dict[str, Dict[str, Any]]): The 'pretrained' settings replaced for the run, by section.
        configuration_name (str): The file name of the written configuration.
        task (str): "classification", "sequence_classification", "semantic_segmentation", "object_detection",
            "instance_segmentation" or "self_supervised".
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
                                                     "validation_iteration_ratio": 2,
                                                     "learning_rate": TRANSFER_LEARNING_RATE})
    configuration["ExperimentConfiguration"]["diagnosis_modes"].update({"compute_receptive_field": False,
                                                                        "view_input_images": False})
    # Small images, the detectors taking their own size with anchor scales and mask heads fitted to it
    configuration["Data"]["image_settings"]["size"] = TRANSFER_IMAGE_SIZE
    if "DetectionLevels" in configuration:
        configuration["Data"]["image_settings"]["size"] = DETECTION_TRANSFER_IMAGE_SIZE
        for level, scales in DETECTION_TRANSFER_ANCHOR_SCALES.items():
            configuration["DetectionLevels"][level]["anchors"]["scales"] = scales
    if "MaskHead" in configuration:
        configuration["MaskHead"]["mask_resolution"] = 14
        for level in DETECTION_TRANSFER_ANCHOR_SCALES:
            configuration["DetectionLevels"][level]["mask_head"] = DETECTION_TRANSFER_MASK_HEAD
    if task == "sequence_classification":
        configuration["Data"]["sequence_settings"]["maximum_sequence_length"] = TRANSFER_NUMBER_FRAMES
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
