"""
Tests of the input samples written at the start of every training launch (BaseTrainer.save_input_samples of
vision_training/trainers/base_trainer.py and vision_training/trainers/input_samples.py): the object detection,
classification, sequence classification and self-supervised trainers draw the first 10 samples of every split by
identifier, as the input viewer draws them, into 'Input_Samples/<split>/sample_<position>.png' with their record
samples.json (the boxes of the detection, the training samples augmented on the device of the model, the two views of
the self-supervised task, the same validation and test images at every launch, the random states of the training left
as they were). On the dummy task, a training without validation loader writes no Validation folder and says so, a
resumed run keeps its folder when its configuration and data are unchanged and archives it as
'Input_Samples_until_iteration_<n>' when one of them changed, an archive of that name is never replaced, and an
unreadable sample is recorded without stopping the training.
"""

import cv2
import json
import numpy as np
import pytest
import random
import torch
import yaml

from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch

from tests.fixtures.dummy_task import DummyDataset
from tests.tracking.test_training_tracking import TRACKER_FACTORY_TARGET, RecordingTracker
from tests.trainers.test_base_trainer import build_dummy_trainer
from tests.trainers.test_optional_validation_loader import TASK_SMOKE_SETTINGS
from vision_training.data.synthetic.create_synthetic_dataset import create_synthetic_dataset
from vision_training.trainers.base_trainer import BaseTrainer
from vision_training.trainers.input_samples import (INPUT_SAMPLES_FOLDER_NAME, INPUT_SAMPLES_RECORD_FILE_NAME,
                                                    NUMBER_INPUT_SAMPLES, prepare_input_samples_folder,
                                                    read_input_samples)
from vision_training.trainers.sequence_classification_trainer import FRAME_GRID_DISPLAY_SIZE
from vision_training.utilities.visualization import GRID_GAP, INFORMATION_PANEL_WIDTH, VIEWER_WINDOW_SPACING

# Samples of every split of the generated datasets, more than the input samples drawn per split
INPUT_SAMPLE_SPLIT_SIZES = {"train": 12, "validation": 11, "test": 11}

# Split folders of the generated datasets, by split name of the input samples
SPLIT_FOLDER_NAMES = {"Training": "train", "Validation": "validation", "Test": "test"}

# Key naming every sample in its source description (describe_sample of its dataset), by task
SOURCE_KEYS = {"object_detection": "image", "classification": "sample", "sequence_classification": "sequence",
               "self_supervised": "sample"}


def build_task_trainer(temporary_directory: Path, task_name: str) -> BaseTrainer:
    """
    Builds the trainer of a task from the smoke configuration of its trainer test (TASK_SMOKE_SETTINGS) on a generated
    dataset of more samples per split than the input samples, every training sample being augmented (global
    augmentation probability 1) so that the augmentation lines of its record are known. The dataset is generated once
    per temporary directory.

    Args:
        temporary_directory (Path): The pytest temporary directory, receiving the dataset, the configuration and the
            run folder.
        task_name (str): The task, a key of TASK_SMOKE_SETTINGS.

    Returns:
        BaseTrainer: The trainer of a new run of the task.
    """
    task_settings = TASK_SMOKE_SETTINGS[task_name]
    smoke_module = task_settings["module"]
    dataset_folder = temporary_directory / "synthetic_dataset"
    if not dataset_folder.exists():
        create_synthetic_dataset(output_folder=str(dataset_folder), image_size=smoke_module.SMOKE_IMAGE_SIZE,
                                 number_workers=1, **{**task_settings["dataset_settings"],
                                                      "split_sizes": INPUT_SAMPLE_SPLIT_SIZES})

    # The smoke configuration of the task, every training sample augmented
    configuration_path = smoke_module.write_smoke_configuration(temporary_directory=temporary_directory,
                                                                dataset_folder=dataset_folder, number_iterations=1,
                                                                **task_settings["configuration_settings"])
    configuration = yaml.safe_load(stream=configuration_path.read_text())
    configuration["Data"]["data_augmentation"]["probability"] = 1.0
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    return smoke_module.build_trainer(configuration_path=configuration_path)


def list_first_manifest_names(task_name: str, split_folder: Path) -> List[str]:
    """
    Gives the names of the first NUMBER_INPUT_SAMPLES samples of a split by identifier, read from its manifest, which
    the input samples must show in that order: the image files of the classification and COCO manifests sorted by
    their 'id', and the sequences of a sequence manifest, numbered in its order.

    Args:
        task_name (str): The task of the split, a key of SOURCE_KEYS.
        split_folder (Path): The folder of the split, holding its labels.json.

    Returns:
        List[str]: The names of the first samples.
    """
    manifest = json.loads((split_folder / "labels.json").read_text())
    if task_name == "sequence_classification":
        return [sequence["name"] for sequence in manifest["sequences"]][:NUMBER_INPUT_SAMPLES]
    entries = manifest["images"] if task_name == "object_detection" else manifest["samples"]
    return [entry["file_name"] for entry in sorted(entries, key=lambda entry: entry["id"])][:NUMBER_INPUT_SAMPLES]


def read_folder_files(folder: Path) -> Dict[str, bytes]:
    """
    Reads every file of a folder, to compare the input samples of two launches.

    Args:
        folder (Path): The folder.

    Returns:
        Dict[str, bytes]: The content of every file, keyed by its path relative to the folder.
    """
    return {path.relative_to(folder).as_posix(): path.read_bytes() for path in sorted(folder.rglob(pattern="*"))
            if path.is_file()}


def write_record_marker(folder: Path, text: str) -> None:
    """
    Stands for the input samples of one launch in the folder tests of prepare_input_samples_folder: a record holding a
    text naming the launch, so that a test can tell which launch a kept folder comes from.

    Args:
        folder (Path): The input samples folder, created when missing.
        text (str): The text of the record.
    """
    folder.mkdir(parents=True, exist_ok=True)
    (folder / INPUT_SAMPLES_RECORD_FILE_NAME).write_text(data=text)


def read_random_states(device: torch.device) -> List[object]:
    """
    Reads the random states of Python and PyTorch (CPU and the device of the model), which the input samples must
    leave as they were.

    Args:
        device (torch.device): The device of the model.

    Returns:
        List[object]: The Python state, then the bytes of the PyTorch CPU state, then those of the CUDA state when the
            device is a GPU.
    """
    states = [random.getstate(), torch.get_rng_state().numpy().tobytes()]
    if device.type == "cuda":
        states.append(torch.cuda.get_rng_state(device=device).numpy().tobytes())
    return states


@pytest.mark.parametrize(argnames="task_name", argvalues=list(SOURCE_KEYS))
def test_input_samples_of_every_task_family(tmp_path: Path, task_name: str, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the input samples of a task of every family: the first 10 samples of the training, validation and test
    splits by identifier, drawn as the viewer windows of the task (the boxes of the detection, the frame grid of the
    sequences, the two views of the self-supervised training and validation images side by side), each PNG named after
    its position, with the record of every sample (split, file, source, labels, augmentations as the viewer writes
    them, size of the image fed to the network) and the console line giving the folder and the time taken. The
    training samples are augmented, on the device of the model, and the validation and test ones are not, except the
    validation views of the self-supervised task; the random states of the training are left as they were.

    Args:
        tmp_path (Path): The pytest temporary directory.
        task_name (str): The task whose trainer draws its input samples.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    trainer = build_task_trainer(temporary_directory=tmp_path, task_name=task_name)
    image_size = TASK_SMOKE_SETTINGS[task_name]["module"].SMOKE_IMAGE_SIZE
    random_states = read_random_states(device=trainer.device)
    capsys.readouterr()

    # The training batches augmented on the device of the model, as by the training step
    with patch.object(target=trainer, attribute="augment_input_batch",
                      wraps=trainer.augment_input_batch) as augment_mock:
        trainer.save_input_samples()
    assert [call.kwargs["batch"]["images"].device.type for call in augment_mock.call_args_list] == (
        [trainer.device.type] * NUMBER_INPUT_SAMPLES * len(trainer.AUGMENTED_INPUT_SPLITS))
    assert read_random_states(device=trainer.device) == random_states
    assert "(30 images written in" in capsys.readouterr().out

    # Ten images per split, named after their position, and their record
    input_samples_folder = trainer.experiment_folder / INPUT_SAMPLES_FOLDER_NAME
    assert sorted(path.name for path in input_samples_folder.iterdir()) == [
        "Test", "Training", "Validation", INPUT_SAMPLES_RECORD_FILE_NAME]
    record = json.loads((input_samples_folder / INPUT_SAMPLES_RECORD_FILE_NAME).read_text())
    assert record["first_iteration"] == 1 and record["samples_per_split"] == NUMBER_INPUT_SAMPLES
    for split_name, split_folder_name in SPLIT_FOLDER_NAMES.items():
        augmented = split_name in trainer.AUGMENTED_INPUT_SPLITS
        assert record["splits"][split_name] == {"written_samples": NUMBER_INPUT_SAMPLES, "augmented": augmented}
        assert sorted(path.name for path in (input_samples_folder / split_name).iterdir()) == [
            f"sample_{position:02d}.png" for position in range(NUMBER_INPUT_SAMPLES)]

        # The first samples of the manifest by identifier, in order, with the size of the image fed to the network
        split_records = [sample for sample in record["samples"] if sample["split"] == split_name]
        assert [sample["source"][SOURCE_KEYS[task_name]] for sample in split_records] == list_first_manifest_names(
            task_name=task_name, split_folder=tmp_path / "synthetic_dataset" / split_folder_name)
        assert [sample["file"] for sample in split_records] == [f"{split_name}/sample_{position:02d}.png"
                                                                for position in range(NUMBER_INPUT_SAMPLES)]
        assert all(sample["image_size"] == [image_size, image_size] for sample in split_records)

        # Every sample drawn as its viewer windows: one window, or the two views side by side
        window_titles = ["View 1", "View 2"] if task_name == "self_supervised" and augmented else [split_name]
        display_width = FRAME_GRID_DISPLAY_SIZE if task_name == "sequence_classification" else image_size
        expected_width = len(window_titles) * (display_width + GRID_GAP + INFORMATION_PANEL_WIDTH) + \
            (len(window_titles) - 1) * VIEWER_WINDOW_SPACING
        for sample in split_records:
            assert list(sample["augmentations"]) == window_titles
            assert cv2.imread(filename=str(input_samples_folder / sample["file"])).shape[1] == expected_width

            # The augmentations drawn for the augmented samples, every sample being augmented, or the note of the
            # samples fed as they are
            for augmentation_lines in sample["augmentations"].values():
                if augmented:
                    assert augmentation_lines[0] in ["Augmentations", "View augmentations"]
                    assert augmentation_lines[1] == "Geometric :"
                elif task_name == "self_supervised":
                    assert augmentation_lines == ["Views : none (test image, encoded whole)"]
                else:
                    assert augmentation_lines == [f"Augmentations : none ({split_name.lower()} sample)"]

        # The labels drawn: the objects of the detection, with their boxes in green on the image, or the class
        for sample in split_records:
            if task_name == "object_detection":
                assert sample["labels"][0].startswith("Objects : ")
                sample_image = cv2.imread(filename=str(input_samples_folder / sample["file"]))
                if sample["labels"][0] != "Objects : none":
                    assert np.all(sample_image[:, :image_size] == (0, 255, 0), axis=-1).any()
            else:
                assert sample["labels"][0].startswith("Class : ")


def test_validation_and_test_samples_are_the_same_at_every_launch(tmp_path: Path) -> None:
    """
    Verifies that two launches of a new classification run draw the same input samples: the same images of every
    split and the same validation and test images, byte for byte, the training images being augmented anew. The second
    launch, whose run has no checkpoint, replaces the folder of the first one.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    first_trainer = build_task_trainer(temporary_directory=tmp_path, task_name="classification")
    first_trainer.save_input_samples()
    input_samples_folder = first_trainer.experiment_folder / INPUT_SAMPLES_FOLDER_NAME
    first_files = read_folder_files(folder=input_samples_folder)
    first_record = json.loads(first_files[INPUT_SAMPLES_RECORD_FILE_NAME])

    second_trainer = build_task_trainer(temporary_directory=tmp_path, task_name="classification")
    second_trainer.save_input_samples()
    second_files = read_folder_files(folder=input_samples_folder)
    second_record = json.loads(second_files[INPUT_SAMPLES_RECORD_FILE_NAME])
    assert [sample["source"] for sample in second_record["samples"]] == [
        sample["source"] for sample in first_record["samples"]]
    assert {file_name: content for file_name, content in second_files.items()
            if file_name.startswith(("Validation/", "Test/"))} == {
        file_name: content for file_name, content in first_files.items()
        if file_name.startswith(("Validation/", "Test/"))}


def test_self_supervised_viewer_shows_the_two_views_of_a_training_image(tmp_path: Path) -> None:
    """
    Verifies the input viewer of the self-supervised task, which shares the windows of its input samples: the views
    drawn for a whole training batch on the CPU (augment_input_batch) are shown as two windows, "View 1" and "View 2",
    for one image of the batch, the validation image being left out.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    trainer = build_task_trainer(temporary_directory=tmp_path, task_name="self_supervised")
    training_batch = read_input_samples(dataloader=trainer.train_dataloader, number_samples=1)[0].batch
    viewed_batch = trainer.augment_input_batch(batch={**training_batch,
                                                      "images": training_batch["images"].to(dtype=torch.float32)})
    with patch(target="vision_training.trainers.self_supervised_trainer.view_input_data",
               return_value=False) as view_mock:
        assert trainer.show_input_samples(training_batch=viewed_batch, validation_batch=training_batch,
                                          sample_index=0) is False
    shown_windows = view_mock.call_args.kwargs["windows"]
    assert [window.title for window in shown_windows] == ["View 1", "View 2"]
    assert all(torch.equal(input=window.image, other=view[0])
               for window, view in zip(shown_windows, viewed_batch["views"]))


def test_viewer_shows_a_training_window_next_to_a_validation_window(tmp_path: Path) -> None:
    """
    Verifies the input viewer shared by the supervised tasks (BaseTrainer.show_input_samples): the windows of the
    training sample, then those of the validation sample, drawn by build_input_windows of the task.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=1, view_input_images=False,
                                  number_sample_images=0)
    training_batch = next(iter(trainer.train_dataloader))
    validation_batch = next(iter(trainer.validation_dataloader))
    with patch(target="vision_training.trainers.base_trainer.view_input_data", return_value=True) as view_mock:
        assert BaseTrainer.show_input_samples(self=trainer, training_batch=training_batch,
                                              validation_batch=validation_batch, sample_index=1) is True
    shown_windows = view_mock.call_args.kwargs["windows"]
    assert [window.title for window in shown_windows] == ["Training", "Validation"]
    assert torch.equal(input=shown_windows[1].image, other=validation_batch["images"][1])


def test_training_without_validation_loader_writes_no_validation_samples(tmp_path: Path,
                                                                        capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a training without validation loader ('validation_iteration_ratio' 0) writes the training and test
    input samples only, its record and one console line saying why the Validation folder is missing.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=2, view_input_images=False,
                                  number_sample_images=0, validation_iteration_ratio=0)
    trainer.run_training_loop()
    console_output = capsys.readouterr().out
    input_samples_folder = trainer.experiment_folder / INPUT_SAMPLES_FOLDER_NAME
    assert sorted(path.name for path in input_samples_folder.iterdir()) == [
        "Test", "Training", INPUT_SAMPLES_RECORD_FILE_NAME]
    record = json.loads((input_samples_folder / INPUT_SAMPLES_RECORD_FILE_NAME).read_text())
    assert record["splits"]["Validation"] == {
        "written_samples": 0, "augmented": False,
        "note": "No validation loader: 'ExperimentConfiguration.validation_iteration_ratio' is 0."}
    assert {sample["split"] for sample in record["samples"]} == {"Training", "Test"}
    assert console_output.count("No Validation Images, 'validation_iteration_ratio' Is 0 (No Validation Loader)") == 1
    assert "(14 images written in" in console_output


def test_resume_with_unchanged_settings_keeps_the_input_samples(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a resumed run whose configuration and data did not change keeps the input samples of its previous
    start as they are, without archive, and says so.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    first_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=2, view_input_images=False,
                                        number_sample_images=0)
    first_trainer.run_training_loop()
    input_samples_folder = first_trainer.experiment_folder / INPUT_SAMPLES_FOLDER_NAME
    first_files = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                   for path in input_samples_folder.rglob(pattern="*") if path.is_file()}
    capsys.readouterr()

    resumed_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=4, view_input_images=False,
                                          number_sample_images=0)
    resumed_trainer.run_training_loop()
    assert resumed_trainer.start_iteration == 3
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in input_samples_folder.rglob(pattern="*") if path.is_file()} == first_files
    assert not list(resumed_trainer.experiment_folder.glob(pattern=f"{INPUT_SAMPLES_FOLDER_NAME}_until_iteration_*"))
    assert "(kept, configuration and data unchanged since iteration 2)" in capsys.readouterr().out


@pytest.mark.parametrize(argnames="changed_part", argvalues=["configuration", "data"])
def test_resume_with_changed_settings_archives_the_input_samples(tmp_path: Path, changed_part: str,
                                                                 capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a resumed run whose configuration or data changed keeps the input samples of its previous start
    whole as 'Input_Samples_until_iteration_<n>', the iteration it resumes from, writes those of the new start in their
    place, says where the previous ones are, and sends the new ones to MLflow under 'Input_Samples_resume_<n>'.

    Args:
        tmp_path (Path): The pytest temporary directory.
        changed_part (str): "configuration" for a changed configuration file, or "data" for a training split of 9
            samples instead of 8, its change being allowed.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    first_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=2, view_input_images=False,
                                        number_sample_images=0)
    first_trainer.run_training_loop()
    input_samples_folder = first_trainer.experiment_folder / INPUT_SAMPLES_FOLDER_NAME
    first_files = read_folder_files(folder=input_samples_folder)
    capsys.readouterr()

    # The same run resumed from its checkpoint of iteration 2 with a changed configuration or changed data
    changed_arguments = {"configuration_text": "dummy: false\n"} if changed_part == "configuration" else {
        "allow_data_change_on_resume": True, "number_training_samples": 9}
    resumed_tracker = RecordingTracker()
    with patch(target=TRACKER_FACTORY_TARGET, return_value=resumed_tracker):
        resumed_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=4,
                                              view_input_images=False, number_sample_images=0, **changed_arguments)
        resumed_trainer.run_training_loop()
    console_output = capsys.readouterr().out

    # The previous samples kept whole, the new ones written for the training split of the new start
    archived_folder = resumed_trainer.experiment_folder / f"{INPUT_SAMPLES_FOLDER_NAME}_until_iteration_2"
    assert read_folder_files(folder=archived_folder) == first_files
    record = json.loads((input_samples_folder / INPUT_SAMPLES_RECORD_FILE_NAME).read_text())
    number_training_samples = 9 if changed_part == "data" else 8
    assert record["first_iteration"] == 3
    assert record["splits"]["Training"]["written_samples"] == number_training_samples
    assert "the previous ones kept in" in console_output and "Input_Samples_until_iteration_2" in console_output

    # The new samples sent to MLflow under the folder of the resumption, the split folders kept
    input_sample_artifacts = sorted((file_path, artifact_folder) for file_path, artifact_folder, _ in
                                    resumed_tracker.artifacts
                                    if str(artifact_folder).startswith(INPUT_SAMPLES_FOLDER_NAME))
    expected_artifacts = [(input_samples_folder / split_name / f"sample_{position:02d}.png",
                           f"{INPUT_SAMPLES_FOLDER_NAME}_resume_2/{split_name}")
                          for split_name, number_samples in [("Test", 6), ("Training", number_training_samples),
                                                             ("Validation", 4)]
                          for position in range(number_samples)]
    expected_artifacts.append((input_samples_folder / INPUT_SAMPLES_RECORD_FILE_NAME,
                               f"{INPUT_SAMPLES_FOLDER_NAME}_resume_2"))
    assert input_sample_artifacts == sorted(expected_artifacts)


def test_input_samples_folder_follows_the_launch(tmp_path: Path) -> None:
    """
    Verifies the decisions taken on the input samples folder at every launch (prepare_input_samples_folder): a run
    without folder writes it, a run without checkpoint replaces it, a resumed run with unchanged configuration and
    data keeps it, a resumed run whose configuration or data changed archives it, and an archive of the resumed
    checkpoint already there is never replaced, the current folder being replaced instead.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    input_samples_folder = tmp_path / INPUT_SAMPLES_FOLDER_NAME
    archived_folder = tmp_path / f"{INPUT_SAMPLES_FOLDER_NAME}_until_iteration_2"

    # Without folder, then without checkpoint, the samples are written, a previous folder being removed
    assert prepare_input_samples_folder(run_folder=tmp_path, resumed_iteration=None, launch_changed=False) == (
        True, None)
    write_record_marker(folder=input_samples_folder, text="first start")
    assert prepare_input_samples_folder(run_folder=tmp_path, resumed_iteration=None, launch_changed=False) == (
        True, None)
    assert not input_samples_folder.exists()

    # A resume without change keeps the folder, a resume with changes archives it
    write_record_marker(folder=input_samples_folder, text="first start")
    assert prepare_input_samples_folder(run_folder=tmp_path, resumed_iteration=2, launch_changed=False) == (
        False, None)
    assert (input_samples_folder / INPUT_SAMPLES_RECORD_FILE_NAME).read_text() == "first start"
    assert prepare_input_samples_folder(run_folder=tmp_path, resumed_iteration=2, launch_changed=True) == (
        True, archived_folder)
    assert not input_samples_folder.exists()
    assert (archived_folder / INPUT_SAMPLES_RECORD_FILE_NAME).read_text() == "first start"

    # A second resume from the same checkpoint keeps the archive of the first one and replaces the current folder
    write_record_marker(folder=input_samples_folder, text="second start")
    assert prepare_input_samples_folder(run_folder=tmp_path, resumed_iteration=2, launch_changed=True) == (
        True, archived_folder)
    assert not input_samples_folder.exists()
    assert (archived_folder / INPUT_SAMPLES_RECORD_FILE_NAME).read_text() == "first start"


def test_unreadable_input_sample_is_recorded_without_stopping_the_training(tmp_path: Path,
                                                                           capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a sample whose image cannot be read is recorded in samples.json with its error and named in the
    console instead of being drawn, the other samples being drawn at their positions and the training going on.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    training_dataset = DummyDataset(sample_identifiers=list(range(8)), seed=0, unreadable_identifiers=[3])
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=2, view_input_images=False,
                                  number_sample_images=0, training_dataset=training_dataset)
    trainer.run_training_loop()
    console_output = capsys.readouterr().out
    input_samples_folder = trainer.experiment_folder / INPUT_SAMPLES_FOLDER_NAME
    record = json.loads((input_samples_folder / INPUT_SAMPLES_RECORD_FILE_NAME).read_text())
    training_records = [sample for sample in record["samples"] if sample["split"] == "Training"]
    expected_files: List[Optional[str]] = [f"Training/sample_{position:02d}.png" for position in range(8)]
    expected_files[3] = None
    assert [sample["file"] for sample in training_records] == expected_files
    assert training_records[3]["error"] == "ValueError: The image cannot be read : 3.png"
    assert training_records[3]["source"] == {"sample": "sample_3"}
    assert record["splits"]["Training"]["written_samples"] == 7
    assert sorted(path.name for path in (input_samples_folder / "Training").iterdir()) == [
        f"sample_{position:02d}.png" for position in range(8) if position != 3]
    assert "- Input Sample Not Read   : Training sample 03" in console_output
    assert (trainer.weights_directory / "Iteration_2").is_dir()
