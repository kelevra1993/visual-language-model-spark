"""
Tests of the validation loader left out of the trainings without validation ('validation_iteration_ratio' 0) for the
trainer of every task of vision_training/trainers: the smoke configuration of each task test of tests/trainers, set to
no validation, gives a trainer without validation loader (BaseTrainer.builds_validation_loader), so that no validation
dataset, batch or worker process holds memory during the training, while the training and test loaders are built and
the validation split is still described at the start of the run.
"""

import pytest
import yaml

from pathlib import Path
from typing import Any, Dict

from tests.trainers import (test_classification_trainer, test_detection_trainer, test_frame_selection_trainer,
                            test_instance_segmentation_trainer, test_keypoint_trainer, test_segmentation_trainer,
                            test_self_supervised_trainer, test_sequence_classification_trainer,
                            test_sequence_comparison_trainer, test_sequence_labelling_trainer)
from vision_training.data.synthetic.create_synthetic_dataset import create_synthetic_dataset

# For every task: the test module giving its smoke configuration (write_smoke_configuration, build_trainer and
# SMOKE_IMAGE_SIZE), the generated dataset of the task, a few samples per split, and the settings its
# write_smoke_configuration takes beyond the common ones, left as in the configuration of the task
TASK_SMOKE_SETTINGS = {
    "classification": {"module": test_classification_trainer,
                       "dataset_settings": {"task": "classification", "maximum_objects": 1, "minimum_frames": 1,
                                            "maximum_frames": 1, "split_sizes": {"train": 8, "validation": 4,
                                                                                 "test": 4}},
                       "configuration_settings": {"head_settings": {}, "evaluation_settings": {}}},
    "object_detection": {"module": test_detection_trainer,
                         "dataset_settings": {"task": "object_detection", "maximum_objects": 4, "minimum_frames": 1,
                                              "maximum_frames": 1, "split_sizes": {"train": 4, "validation": 2,
                                                                                   "test": 2}},
                         "configuration_settings": {}},
    "instance_segmentation": {"module": test_instance_segmentation_trainer,
                              "dataset_settings": {"task": "instance_segmentation", "maximum_objects": 4,
                                                   "minimum_frames": 1, "maximum_frames": 1,
                                                   "split_sizes": {"train": 4, "validation": 2, "test": 2}},
                              "configuration_settings": {"original_resolution": False}},
    "keypoint_detection": {"module": test_keypoint_trainer,
                           "dataset_settings": {"task": "keypoint_detection", "maximum_objects": 2,
                                                "minimum_frames": 1, "maximum_frames": 1,
                                                "split_sizes": {"train": 4, "validation": 2, "test": 2}},
                           "configuration_settings": {}},
    "semantic_segmentation": {"module": test_segmentation_trainer,
                              "dataset_settings": {"task": "semantic_segmentation", "maximum_objects": 3,
                                                   "minimum_frames": 1, "maximum_frames": 1,
                                                   "split_sizes": {"train": 8, "validation": 4, "test": 4}},
                              "configuration_settings": {"section_settings": {}, "evaluation_settings": {}}},
    "self_supervised": {"module": test_self_supervised_trainer,
                        "dataset_settings": {"task": "classification", "maximum_objects": 1, "minimum_frames": 1,
                                             "maximum_frames": 1, "split_sizes": {"train": 8, "validation": 4,
                                                                                  "test": 4}},
                        "configuration_settings": {"head_settings": {}}},
    "sequence_classification": {"module": test_sequence_classification_trainer,
                                "dataset_settings": {"task": "sequence_classification", "maximum_objects": 1,
                                                     "minimum_frames": 4, "maximum_frames": 4,
                                                     "split_sizes": {"train": 4, "validation": 2, "test": 2}},
                                "configuration_settings": {}},
    "sequence_comparison": {"module": test_sequence_comparison_trainer,
                            "dataset_settings": {"task": "sequence_comparison", "maximum_objects": 3,
                                                 "minimum_frames": 2, "maximum_frames": 4,
                                                 "split_sizes": {"train": 4, "validation": 2, "test": 2}},
                            "configuration_settings": {}},
    "sequence_labelling": {"module": test_sequence_labelling_trainer,
                           "dataset_settings": {"task": "sequence_labelling", "maximum_objects": 6,
                                                "minimum_frames": 4, "maximum_frames": 6,
                                                "split_sizes": {"train": 4, "validation": 2, "test": 2}},
                           "configuration_settings": {}},
    "frame_selection": {"module": test_frame_selection_trainer,
                        "dataset_settings": {"task": "frame_selection", "maximum_objects": 1, "minimum_frames": 4,
                                             "maximum_frames": 6, "split_sizes": {"train": 4, "validation": 2,
                                                                                  "test": 2}},
                        "configuration_settings": {}}}


@pytest.mark.parametrize(argnames="task_name", argvalues=list(TASK_SMOKE_SETTINGS))
def test_training_without_validation_builds_no_validation_loader(tmp_path: Path, task_name: str) -> None:
    """
    Verifies, for the trainer of every task, that a training without validation ('validation_iteration_ratio' 0)
    builds no validation loader: the trainer built from the smoke configuration of the task set to no validation has
    None in place of its validation loader and no validation loader among the loaders whose workers are stopped, while
    its training and test loaders are built and the validation split is still counted at the start of the run.

    Args:
        tmp_path (Path): The pytest temporary directory.
        task_name (str): The task whose trainer is built, a key of TASK_SMOKE_SETTINGS.
    """
    task_settings: Dict[str, Any] = TASK_SMOKE_SETTINGS[task_name]
    smoke_module = task_settings["module"]

    # Small generated dataset of the task, whose validation split the training never loads
    dataset_folder = tmp_path / "synthetic_dataset"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=smoke_module.SMOKE_IMAGE_SIZE,
                             number_workers=1, **task_settings["dataset_settings"])

    # The smoke configuration of the task, with the validation turned off
    configuration_path = smoke_module.write_smoke_configuration(temporary_directory=tmp_path,
                                                                dataset_folder=dataset_folder, number_iterations=2,
                                                                **task_settings["configuration_settings"])
    configuration = yaml.safe_load(stream=configuration_path.read_text())
    configuration["ExperimentConfiguration"]["validation_iteration_ratio"] = 0
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))

    # No validation loader, nor any to stop, the training and test loaders being built as with validation
    trainer = smoke_module.build_trainer(configuration_path=configuration_path)
    assert trainer.builds_validation_loader is False and trainer.validation_dataloader is None
    assert trainer.train_dataloader is not None and trainer.test_dataloader is not None
    assert all(dataloader is not None for dataloader in trainer.list_dataloaders())

    # The validation split is still described at the start of the run, from its manifest
    assert trainer.number_samples_per_split["validation_data_folder"] == (
        task_settings["dataset_settings"]["split_sizes"]["validation"])
