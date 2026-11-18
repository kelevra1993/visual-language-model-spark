"""
Input samples of a training launch: the first samples of the training, validation and test splits, drawn as the input
viewer ('view_input_images') shows them, written to '<run>/Input_Samples/' before the first iteration of every launch
that trains (BaseTrainer.save_input_samples of trainers/base_trainer.py), one folder per split and the record of what
every image shows (samples.json), so that the preprocessing, the label mapping and the augmentation a run trained with
can be checked afterwards, without a display. A resumed run keeps the folder of its previous start, unless its
configuration or its data changed, which keeps the previous folder as 'Input_Samples_until_iteration_<n>', like the
archived configuration and data record. The task trainers draw the samples (build_input_windows) and
utilities/visualization.py lays out their images; this module chooses, reads, writes, records and lists them.
"""

import cv2
import json
import shutil

from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from vision_training.utilities.visualization import (ViewerWindow, build_viewer_window, flatten_panel_lines,
                                                     join_viewer_windows)

# Number of samples drawn per split at every launch, the first ones of the split by identifier
NUMBER_INPUT_SAMPLES = 10

# Folder of the input samples in the run folder, holding one folder per split and the record of the samples, also the
# MLflow artifact folder they are sent to
INPUT_SAMPLES_FOLDER_NAME = "Input_Samples"
INPUT_SAMPLES_RECORD_FILE_NAME = "samples.json"


def prepare_input_samples_folder(run_folder: Path, resumed_iteration: Optional[int],
                                 launch_changed: bool) -> Tuple[bool, Optional[Path]]:
    """
    Decides, at the start of a training launch, whether its input samples are written, and makes room for them: a new
    run and a run started before the folder existed write them, a new run relaunched before its first checkpoint
    replaces the folder of its previous start, and a resumed run writes them only when its configuration or its data
    changed since its previous start, the previous folder being kept as 'Input_Samples_until_iteration_<n>', the samples
    that trained the run up to the checkpoint of iteration n it resumes from. As for the archived configuration, a
    folder of that name is never replaced: an earlier resume from the same checkpoint kept the samples that really
    trained the run, and the current folder is replaced instead.

    Args:
        run_folder (Path): The run folder.
        resumed_iteration (Optional[int]): The iteration of the checkpoint the run resumes from, None for a run without
            checkpoint.
        launch_changed (bool): Whether the configuration or the data of a resumed run changed since its previous
            start, as found by the comparisons of the base trainer with the saved configuration and data record.

    Returns:
        Tuple[bool, Optional[Path]]: Whether the input samples are written (the folder being then absent), and the
            folder keeping the samples of the previous start of a resumed run, None when there is none.
    """
    input_samples_folder = run_folder / INPUT_SAMPLES_FOLDER_NAME

    # A new run, or a run started before the input samples existed, writes them
    if not input_samples_folder.exists():
        return True, None

    # A run without checkpoint starts from scratch, its previous start having trained nothing worth keeping
    if resumed_iteration is None:
        shutil.rmtree(path=input_samples_folder)
        return True, None

    # A resumed run with the same configuration and data keeps the samples of its previous start
    if not launch_changed:
        return False, None

    # Keep the samples that trained the run up to the resumed checkpoint, unless an earlier resume kept them
    archived_folder = run_folder / f"{INPUT_SAMPLES_FOLDER_NAME}_until_iteration_{resumed_iteration}"
    if archived_folder.exists():
        shutil.rmtree(path=input_samples_folder)
    else:
        input_samples_folder.rename(target=archived_folder)
    return True, archived_folder


class InputSample(NamedTuple):
    """
    One input sample of a split listed by read_input_samples: its identifier in the batches, its source in the manifest
    (its image file, or its sequence or pair name, as the dataset describes it), and the batch of the sample alone as
    the loader gives it, None for a sample that could not be read, with the error that kept it from being read.
    """
    sample_identifier: int
    source: Dict[str, str]
    batch: Optional[Dict[str, Any]]
    error: Optional[str]


def read_input_samples(dataloader: DataLoader, number_samples: int) -> List[InputSample]:
    """
    Reads the input samples of a split in the main process: the samples of the smallest identifiers, the same at every
    launch whatever the shuffled order of the dataset, read from the dataset of the loader and each batched alone by
    its collate function, so that every batch is the one the loader gives for the sample, whatever the size of the
    image (the whole test images of the tiled segmentation) or of the sequence, without taking any batch from the
    loader whose order the training follows. A sample that cannot be read is recorded with its error, like the batches
    the training drops, the other samples being still read.

    Args:
        dataloader (DataLoader): The loader of the split, whose dataset lists its first samples (list_first_samples)
            and names them (describe_sample).
        number_samples (int): The number of samples read, all of them when the split holds fewer.

    Returns:
        List[InputSample]: The listed samples, in identifier order.
    """
    dataset = dataloader.dataset
    input_samples = []
    for sample_index, sample_identifier in dataset.list_first_samples(number_samples=number_samples):
        source = dataset.describe_sample(sample_identifier=sample_identifier)

        # An unreadable sample is recorded with its error, the other samples still being read
        try:
            batch = dataloader.collate_fn([dataset[sample_index]])
        except Exception as error:
            input_samples.append(InputSample(sample_identifier=sample_identifier, source=source, batch=None,
                                             error=f"{type(error).__name__}: {error}"))
            continue
        input_samples.append(InputSample(sample_identifier=sample_identifier, source=source, batch=batch, error=None))
    return input_samples


def write_input_sample_image(windows: List[ViewerWindow], image_path: Path) -> Dict[str, Any]:
    """
    Writes one input sample as the PNG image of the input viewer windows of the sample side by side (one window for
    most tasks, the two views of the self-supervised task), without the navigation hint of the viewer, and gives what
    the record of the input samples (samples.json) says of it.

    Args:
        windows (List[ViewerWindow]): The windows of the sample, built by the build_input_windows hook of the trainer.
        image_path (Path): The PNG file written, its split folder being created when missing.

    Returns:
        Dict[str, Any]: The 'labels' drawn (the target lines of the panel of the first window, shared by the windows of
            a sample) and the 'augmentations' of every window as its panel writes them, keyed by window title.

    Raises:
        OSError: If OpenCV could not write the image.
    """
    sample_image = join_viewer_windows(window_images=[build_viewer_window(window=window, hint="")
                                                      for window in windows])
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(filename=str(image_path), img=sample_image):
        raise OSError(f"OpenCV could not write the input sample image {image_path}.")
    return {"labels": flatten_panel_lines(lines=windows[0].target_lines),
            "augmentations": {window.title: flatten_panel_lines(lines=window.augmentation_lines)
                              for window in windows}}


def write_input_samples_record(input_samples_folder: Path, first_iteration: int, split_records: Dict[str, Any],
                               sample_records: List[Dict[str, Any]]) -> Path:
    """
    Writes the record of the input samples of a launch ('samples.json'), the only writer of the file: its date, the
    first iteration the launch trains, the number of samples drawn per split, what every split holds (the number of
    images written, whether they are augmented, and why a split has no images), and every listed sample with its
    split, its image file, its identifier, its source in the manifest, the labels drawn, its augmentations, the size of
    the image fed to the network, or the error that kept it from being drawn.

    Args:
        input_samples_folder (Path): The 'Input_Samples' folder of the run.
        first_iteration (int): The first iteration trained by the launch, 1 for a new run.
        split_records (Dict[str, Any]): The summary of every split, keyed by split name.
        sample_records (List[Dict[str, Any]]): The record of every listed sample, split by split.

    Returns:
        Path: The written record.
    """
    record = {"created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "first_iteration": first_iteration,
              "samples_per_split": NUMBER_INPUT_SAMPLES, "splits": split_records, "samples": sample_records}
    record_path = input_samples_folder / INPUT_SAMPLES_RECORD_FILE_NAME
    record_path.write_text(data=json.dumps(obj=record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record_path


def list_input_sample_artifacts(input_samples_folder: Path, artifact_folder: str) -> List[Tuple[Path, str]]:
    """
    Lists the files of the input samples sent to MLflow when a launch wrote them, each with its artifact folder, the
    split folders being kept under the artifact folder of the launch.

    Args:
        input_samples_folder (Path): The 'Input_Samples' folder of the run.
        artifact_folder (str): The MLflow artifact folder of the input samples of the launch, 'Input_Samples' for a new
            run or 'Input_Samples_resume_<n>' for a resumed run.

    Returns:
        List[Tuple[Path, str]]: Every file of the folder, in path order, with the artifact folder it is sent to.
    """
    return [(file_path, "/".join([artifact_folder, *file_path.parent.relative_to(input_samples_folder).parts]))
            for file_path in sorted(input_samples_folder.rglob(pattern="*")) if file_path.is_file()]
