"""
Smoke test of DetectionTrainer of vision_training/trainers/detection_trainer.py on a synthetic dataset: training saves
checkpoints, evaluates and resumes.
"""

import pytest
import torch
import yaml

from pathlib import Path

from vision_training.data.synthetic.create_synthetic_dataset import create_synthetic_dataset
from vision_training.trainers.detection_trainer import DetectionTrainer
from vision_training.configuration.configuration_loader import load_experiment_configuration

SYNTHETIC_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" / "object_detection" /
                                "configuration_synthetic_shapes.yaml")

# Image size of the smoke test, small enough for a few quick iterations
SMOKE_IMAGE_SIZE = 256

# Anchor scales of the smoke test: the configuration checker requires every scale to fit in the image
SMOKE_ANCHOR_SCALES = {"5": [16, 32, 64], "6": [32, 64, 128], "7": [64, 128, 256]}


def write_smoke_configuration(temporary_directory: Path, dataset_folder: Path, number_iterations: int) -> Path:
    """
    Writes a copy of the synthetic shapes configuration adapted to a quick training smoke test: the generated
    dataset, 256 pixel images, the Native loader, batches of 2 images and a saving schedule that triggers an
    intermediate save, a save with test evaluation and a validation pass within a handful of iterations.

    Args:
        temporary_directory (Path): The pytest temporary directory receiving the configuration and the experiments.
        dataset_folder (Path): The root folder of the generated synthetic dataset.
        number_iterations (int): The number of training iterations of the run.

    Returns:
        Path: The path of the written configuration file.
    """
    configuration = yaml.safe_load(stream=SYNTHETIC_CONFIGURATION_PATH.read_text())

    # Run in the temporary directory, on the generated splits, with a saving schedule fitting a few iterations
    experiment_configuration = configuration["ExperimentConfiguration"]
    experiment_configuration.update({"experiment_folder": str(temporary_directory / "experiments"),
                                     "train_data_folder": str(dataset_folder / "train"),
                                     "validation_data_folder": str(dataset_folder / "validation"),
                                     "test_data_folder": str(dataset_folder / "test"), "batch_size": 2,
                                     "information_dump": 2, "weight_saving_iterations": 4,
                                     "intermediate_weight_saving_iterations": 2, "number_iterations": number_iterations,
                                     "validation_iteration_ratio": 2, "resume_training": True})
    experiment_configuration["diagnosis_modes"].update({"compute_receptive_field": False,
                                                        "compute_model_size": False, "view_input_images": False})

    # Read the small images with the Native loader, and map the generated categories to the model classes
    configuration["Data"]["DALI"]["activated"] = False
    configuration["Data"]["image_settings"]["size"] = SMOKE_IMAGE_SIZE
    # Two loader workers per split, enough for the small generated dataset
    configuration["Data"]["loading"]["number_workers"] = 2
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(dataset_folder / "label_mapping.json")
    for level, scales in SMOKE_ANCHOR_SCALES.items():
        configuration["Anchors"]["scales_and_ratios"][level]["scales"] = scales

    configuration_path = temporary_directory / f"configuration_{number_iterations}_iterations.yaml"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    return configuration_path


def build_trainer(configuration_path: Path) -> DetectionTrainer:
    """
    Loads a configuration file and builds the DetectionTrainer from it, exactly like main.py does.

    Args:
        configuration_path (Path): The configuration file of the run.

    Returns:
        DetectionTrainer: The trainer, resumed from the last checkpoint of its run folder when there is one.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    return DetectionTrainer(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                   configuration_path=configuration_path)


def test_training_saves_evaluates_and_resumes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the whole training pipeline on a generated synthetic dataset: 5 iterations save the intermediate and
    evaluated checkpoints and the final one, evaluate the test split into Results.txt and write the three TensorBoard
    folders. A second run with the same settings resolves to the same run folder, resumes from the last checkpoint
    with exactly its weights, and trains 2 more iterations. No training step may fail silently.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 4}, maximum_objects=4,
                             number_frames=1, number_workers=2, task="object_detection")

    # First run: saves at iterations 2 (intermediate), 4 (with test evaluation) and 5 (end of the run)
    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=5))
    assert trainer.start_iteration == 1
    trainer.run_training_loop()

    weights_directory = trainer.weights_directory
    assert sorted(path.name for path in weights_directory.glob("Iteration_*")) == ["Iteration_2", "Iteration_4",
                                                                                 "Iteration_5"]
    assert (weights_directory / "full-checkpoint").read_text().splitlines()[0] == 'model_checkpoint_path: "Iteration_5"'
    assert (trainer.experiment_folder / "Results.txt").read_text().startswith("Iteration 4 : 4 Test Images")
    assert sorted(path.name for path in trainer.tensorboard_directory.iterdir()) == ["Test", "Train", "Validation"]
    results = trainer.run_test_evaluation(iteration=5)
    assert results["headline_metric"] == "mAP@0.5/area" and results["headline_metric"] in results["scalars"]

    # Second run with the same settings: same run folder, resumed from iteration 5 with exactly its weights
    resumed_trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=2))
    assert resumed_trainer.experiment_folder == trainer.experiment_folder
    assert resumed_trainer.start_iteration == 6
    saved_model_state = torch.load(f=weights_directory / "Iteration_5" / "model_000005.pt", map_location="cpu",
                                   weights_only=True)["model_state"]
    for parameter_name, parameter_value in resumed_trainer.model.state_dict().items():
        assert torch.equal(input=parameter_value.cpu(), other=saved_model_state[parameter_name])

    # The resumed run trains iterations 6 (intermediate save) and 7 (end of the run)
    resumed_trainer.run_training_loop()
    assert sorted(path.name for path in weights_directory.glob("Iteration_*")) == ["Iteration_2", "Iteration_4",
                                                                                 "Iteration_5", "Iteration_6",
                                                                                 "Iteration_7"]
    assert (weights_directory / "full-checkpoint").read_text().splitlines()[0] == 'model_checkpoint_path: "Iteration_7"'

    # The training loop catches errors and keeps going, so make sure that no iteration failed along the way
    console_output = capsys.readouterr().out
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output
