import copy
import pytest
import torch
import yaml

from pathlib import Path
from typing import Any, Dict

from vision_training.configuration.configuration_loader import load_experiment_configuration
from vision_training.data.synthetic.create_synthetic_dataset import create_synthetic_dataset
from vision_training.trainers.classification_trainer import ClassificationTrainer

SYNTHETIC_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" / "classification" /
                                "configuration_synthetic_shapes.yaml")

# Image size of the transfer test, small enough for a few quick iterations
TRANSFER_IMAGE_SIZE = 64


def build_transfer_trainer(temporary_directory: Path, dataset_folder: Path, number_iterations: int,
                           pretrained_settings: Dict[str, Any], configuration_name: str) -> ClassificationTrainer:
    """
    Writes a copy of the synthetic shapes classification configuration for a quick run on the generated dataset,
    with the given pretrained backbone settings, and builds its trainer exactly like main.py does.

    Args:
        temporary_directory (Path): The pytest temporary directory receiving the configuration and the experiments.
        dataset_folder (Path): The root folder of the generated synthetic dataset.
        number_iterations (int): The number of training iterations of the run.
        pretrained_settings (Dict[str, Any]): The 'Backbone.pretrained' settings replaced for the run.
        configuration_name (str): The file name of the written configuration.

    Returns:
        ClassificationTrainer: The trainer of the run.
    """
    configuration = yaml.safe_load(stream=SYNTHETIC_CONFIGURATION_PATH.read_text())

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
                                                                        "compute_model_size": False})
    configuration["Data"]["image_settings"]["size"] = TRANSFER_IMAGE_SIZE
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(dataset_folder / "label_mapping.json")
    configuration["Backbone"]["pretrained"].update(pretrained_settings)

    configuration_path = temporary_directory / configuration_name
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    return ClassificationTrainer(experiment_configuration=experiment_configuration,
                                 model_configuration=model_configuration, configuration_path=configuration_path)


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
                                            number_iterations=3, pretrained_settings=no_transfer,
                                            configuration_name="source.yaml")
    source_trainer.run_training_loop()
    source_backbone_state = {parameter_name[len("backbone."):]: parameter_value for parameter_name, parameter_value in
                             torch.load(f=source_trainer.weights_directory / "Iteration_3" / "model_000003.pt",
                                        map_location="cpu", weights_only=True)["model_state"].items()
                             if parameter_name.startswith("backbone.")}

    # Frozen run initialised by the source backbone, in its own folder
    frozen_settings = {**no_transfer, "experiment_folder": str(source_trainer.experiment_folder), "freeze": True}
    frozen_trainer = build_transfer_trainer(temporary_directory=tmp_path, dataset_folder=dataset_folder,
                                            number_iterations=3, pretrained_settings=frozen_settings,
                                            configuration_name="frozen.yaml")
    assert frozen_trainer.run_name == source_trainer.run_name.replace("2x64.2x128_", "2x64.2x128-PT-FRZ_")
    assert frozen_trainer.pretrained_backbone_path == (source_trainer.weights_directory / "Iteration_3" /
                                                       "model_000003.pt")
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

    # Resuming the frozen run keeps its own weights, without loading the source backbone again
    resumed_trainer = build_transfer_trainer(temporary_directory=tmp_path, dataset_folder=dataset_folder,
                                             number_iterations=1, pretrained_settings=frozen_settings,
                                             configuration_name="frozen.yaml")
    assert resumed_trainer.start_iteration == 4 and resumed_trainer.pretrained_backbone_path is None
    assert "the pretrained backbone is not loaded again" in capsys.readouterr().out
    assert all(not parameter.requires_grad for parameter in resumed_trainer.model.backbone.parameters())

    # Fine-tuning run: the transferred backbone learns ten times slower than the head
    fine_tuning_settings = {**no_transfer, "experiment_folder": str(source_trainer.experiment_folder),
                            "learning_rate_multiplier": 0.1}
    fine_tuning_trainer = build_transfer_trainer(temporary_directory=tmp_path, dataset_folder=dataset_folder,
                                                 number_iterations=1, pretrained_settings=fine_tuning_settings,
                                                 configuration_name="fine_tuning.yaml")
    assert "-PT-BLR0.1_" in fine_tuning_trainer.run_name
    assert [parameter_group["lr"] for parameter_group in fine_tuning_trainer.optimizer.param_groups] == pytest.approx(
        expected=[1e-3, 1e-4])
    assert len(fine_tuning_trainer.optimizer.param_groups[1]["params"]) == len(
        list(fine_tuning_trainer.model.backbone.parameters()))
