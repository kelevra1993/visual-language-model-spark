"""
When run, benchmarks the data loaders of the classification, sequence classification and sequence comparison tasks
for implementation_dali_loaders.md (phase 1). For every case (a YAML of the repository and a dataset), the current
Native loader of the trainers, tuned Native loaders and the DALI prototypes are measured alone (stage A), against the
GPU ceiling of the model (B), feeding the training iteration (C) and the evaluation pass (D), with the parity of the
DALI outputs (E), after the cold read rate of the disk (F). Every case runs in its own process, and the measurements
are written to results.csv and summary.md. Run from the repository root with
'.venv/bin/python -m scripts.data.loader_benchmark.run_loader_benchmark'.
"""

import gc
import json
import multiprocessing
import os
import platform
import random
import subprocess
import time
import cv2
import torch
import nvidia.dali as dali

from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict, Iterable, List, Optional, Tuple

from scripts.data.loader_benchmark.benchmark_datasets import (create_benchmark_subset_manifest,
                                                              evict_from_page_cache, get_dataset_file_paths,
                                                              load_into_page_cache, write_benchmark_configuration)
from scripts.data.loader_benchmark.benchmark_measurements import (compare_loader_outputs, limit_opencv_threads,
                                                                  measure_disk_read_rate, measure_end_to_end,
                                                                  measure_evaluation_pass, measure_loader,
                                                                  measure_training_ceiling, summarise_rows,
                                                                  write_benchmark_reports)
from scripts.data.loader_benchmark.dali_prototypes import ClassificationDALILoader, SequenceDALILoader
from vision_training.core.task_registry import get_trainer_class
from vision_training.data.augmentations import DataAugmenter
from vision_training.data.classification.classification_dataset import ClassificationDataset
from vision_training.data.loaders import get_classification_dataloaders, get_sequence_dataloaders
from vision_training.data.sequence.sequence_dataset import SequenceDataset
from vision_training.data.sequence.sequence_pair_dataset import SequencePairDataset
from vision_training.models.model_registry import TASK_MODEL_CLASSES
from vision_training.models.task_model import TaskModel
from vision_training.utilities.os_utilities import print_color
from vision_training.utilities.tensor_utilities import get_device

# Folder of the YAML configurations of the repository, the templates of the benchmark cases
CONFIGURATIONS_FOLDER = Path(__file__).absolute().parents[3] / "vision_training" / "configurations"

# Number of worker processes of the current Native loaders, hard-coded in the trainers
CURRENT_NUMBER_WORKERS = 4

# Learning rate of the optimizer of the ceiling and training stages, whose timings do not depend on it
BENCHMARK_LEARNING_RATE = 1e-4

# Seed of the epoch permutations of the DALI sequence loaders
DALI_SEED = 0


def build_benchmark_cases(imagenet_folder: Path, sequence_data_folder: Path,
                          benchmark_data_folder: Path) -> Dict[str, Dict[str, Any]]:
    """
    Defines the benchmark cases of section 5.3 of the plan: the YAML of the repository giving the task, model, batch
    size and image settings of every case, its split folders and label mapping file (None keeps those of the YAML,
    for the synthetic datasets), whether it is also measured with a cold page cache, and whether the parity is also
    checked with frame labels (sequence labelling).

    Args:
        imagenet_folder (Path): The ImageNet folder prepared by imagenet_download.py.
        sequence_data_folder (Path): The folder of the sequence datasets, holding the benchmark sequences and pairs.
        benchmark_data_folder (Path): The folder of the ImageNet subset written by the benchmark.

    Returns:
        Dict[str, Dict[str, Any]]: The definition of every case, keyed by its name.
    """
    imagenet_manifests = imagenet_folder / "manifests"
    sequence_splits = {f"{split_name}_data_folder": sequence_data_folder / "imagenet_sequences" / split_name
                       for split_name in ["train", "validation", "test"]}
    pair_splits = {f"{split_name}_data_folder": sequence_data_folder / "imagenet_pairs" / split_name
                   for split_name in ["train", "validation", "test"]}
    return {
        "C-IN": {"template": CONFIGURATIONS_FOLDER / "classification" / "configuration_example.yaml",
                 "split_folders": {"train_data_folder": benchmark_data_folder / "imagenet_subset" / "train",
                                   "validation_data_folder": imagenet_manifests / "validation",
                                   "test_data_folder": imagenet_manifests / "test"},
                 "label_mapping_path": imagenet_manifests / "label_mapping.json", "cold_cache": True,
                 "frame_label_parity": False},
        "C-SYN": {"template": CONFIGURATIONS_FOLDER / "classification" / "configuration_synthetic_shapes.yaml",
                  "split_folders": None, "label_mapping_path": None, "cold_cache": False, "frame_label_parity": False},
        "S-SYN": {"template": CONFIGURATIONS_FOLDER / "sequence_classification" /
                  "configuration_synthetic_sequences.yaml", "split_folders": None, "label_mapping_path": None,
                  "cold_cache": False, "frame_label_parity": False},
        "S-JPG": {"template": CONFIGURATIONS_FOLDER / "sequence_classification" / "configuration_example.yaml",
                  "split_folders": sequence_splits,
                  "label_mapping_path": sequence_data_folder / "imagenet_sequences" / "label_mapping.json",
                  "cold_cache": True, "frame_label_parity": True},
        "P-SYN": {"template": CONFIGURATIONS_FOLDER / "sequence_comparison" / "configuration_synthetic_pairs.yaml",
                  "split_folders": None, "label_mapping_path": None, "cold_cache": False, "frame_label_parity": False},
        "P-JPG": {"template": CONFIGURATIONS_FOLDER / "sequence_comparison" / "configuration_example.yaml",
                  "split_folders": pair_splits,
                  "label_mapping_path": sequence_data_folder / "imagenet_pairs" / "label_mapping.json",
                  "cold_cache": True, "frame_label_parity": False}}


def build_current_native_loaders(experiment_configuration: Dict[str, Any],
                                 model_configuration: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds the training, validation and test loaders of a case exactly as the trainers do, through the loader
    functions of vision_training/data/loaders.py with their 4 worker processes and the label type of the trainer of
    the task.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration of the case.
        model_configuration (Dict[str, Any]): The processed model configuration of the case.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: The training, validation and test loaders.
    """
    task = experiment_configuration["task"]
    if task == "classification":
        return get_classification_dataloaders(experiment_configuration=experiment_configuration,
                                              model_configuration=model_configuration,
                                              batch_size=experiment_configuration["batch_size"],
                                              number_of_workers=CURRENT_NUMBER_WORKERS)
    return get_sequence_dataloaders(experiment_configuration=experiment_configuration,
                                    model_configuration=model_configuration,
                                    batch_size=experiment_configuration["batch_size"],
                                    number_of_workers=CURRENT_NUMBER_WORKERS,
                                    label_type=get_trainer_class(task=task).LABEL_TYPE)


def list_loader_variants(settings: Dict[str, Any], is_sequence_case: bool) -> List[Dict[str, Any]]:
    """
    Lists the loader variants of stage A (section 5.4 of the plan): the current Native loader, the tuned Native
    loaders of every worker count and prefetch factor, and the DALI prototypes of every thread count (and reader
    process count for the sequence tasks).

    Args:
        settings (Dict[str, Any]): The benchmark settings.
        is_sequence_case (bool): Whether the case is a sequence task.

    Returns:
        List[Dict[str, Any]]: The descriptor of every variant: its family ('variant'), its 'setting' label and its
            parameters.
    """
    variants = [{"variant": "native_current", "setting": f"workers={CURRENT_NUMBER_WORKERS}"}]

    # Tuned Native loaders
    for number_workers in settings["native_worker_counts"]:
        for prefetch_factor in settings["native_prefetch_factors"]:
            variants.append({"variant": "native_tuned",
                             "setting": f"workers={number_workers} prefetch={prefetch_factor}",
                             "number_workers": number_workers, "prefetch_factor": prefetch_factor})

    # DALI prototypes, the sequence ones with their reader processes
    reader_process_counts = settings["dali_reader_process_counts"] if is_sequence_case else [0]
    for number_threads in settings["dali_thread_counts"]:
        for number_reader_processes in reader_process_counts:
            reader_label = f" readers={number_reader_processes}" if is_sequence_case else ""
            variants.append({"variant": "dali", "setting": f"threads={number_threads}{reader_label}",
                             "number_threads": number_threads, "number_reader_processes": number_reader_processes})
    return variants


def build_dali_loader(dataset: Dataset, batch_size: int, number_threads: int, number_reader_processes: int,
                      prefetch_queue_depth: int, shuffle: bool) -> Iterable:
    """
    Builds the DALI prototype loader of a Native dataset: the classification loader, or the sequence loader for
    windows and pairs.

    Args:
        dataset (Dataset): The Native dataset of the split.
        batch_size (int): The number of samples of a batch.
        number_threads (int): The number of CPU threads of the pipeline.
        number_reader_processes (int): The number of processes reading the frame files (sequence tasks).
        prefetch_queue_depth (int): The number of batches prepared in advance by the pipeline.
        shuffle (bool): Whether to shuffle the samples at every epoch.

    Returns:
        Iterable: The DALI loader.
    """
    device_id = torch.cuda.current_device()
    if isinstance(dataset, ClassificationDataset):
        return ClassificationDALILoader(dataset=dataset, batch_size=batch_size, number_threads=number_threads,
                                        prefetch_queue_depth=prefetch_queue_depth, shuffle=shuffle, device_id=device_id)
    return SequenceDALILoader(dataset=dataset, batch_size=batch_size, number_threads=number_threads,
                              number_reader_processes=number_reader_processes,
                              prefetch_queue_depth=prefetch_queue_depth, shuffle=shuffle, seed=DALI_SEED,
                              device_id=device_id)


def build_variant_loader(variant: Dict[str, Any], dataset: Dataset, current_loader: DataLoader, batch_size: int,
                         settings: Dict[str, Any], shuffle: bool) -> Iterable:
    """
    Builds the loader of a variant for one split: the loader built by the trainers' functions for the current Native
    variant, a tuned DataLoader (persistent workers, pinned memory, one OpenCV thread per worker), or a DALI loader.

    Args:
        variant (Dict[str, Any]): The descriptor of the variant.
        dataset (Dataset): The Native dataset of the split.
        current_loader (DataLoader): The current Native loader of the split.
        batch_size (int): The number of samples of a batch.
        settings (Dict[str, Any]): The benchmark settings.
        shuffle (bool): Whether the tuned and DALI loaders shuffle the samples at every epoch.

    Returns:
        Iterable: The loader.
    """
    if variant["variant"] == "native_current":
        return current_loader
    if variant["variant"] == "native_tuned":
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=variant["number_workers"], prefetch_factor=variant["prefetch_factor"],
                          persistent_workers=True, pin_memory=True, worker_init_fn=limit_opencv_threads)
    return build_dali_loader(dataset=dataset, batch_size=batch_size, number_threads=variant["number_threads"],
                             number_reader_processes=variant["number_reader_processes"],
                             prefetch_queue_depth=settings["dali_prefetch_queue_depth"], shuffle=shuffle)


def make_row(case_name: str, stage: str, variant: Dict[str, Any], cache: str, repeat: int, metrics: Dict[str, Any],
             precision: str = "", label_type: str = "") -> Dict[str, Any]:
    """
    Builds one measurement row of results.csv, its identification columns followed by its metrics.

    Args:
        case_name (str): The case.
        stage (str): The stage letter.
        variant (Dict[str, Any]): The descriptor of the measured variant.
        cache (str): "warm", "cold" or "" when the page cache does not matter.
        repeat (int): The index of the repeat.
        metrics (Dict[str, Any]): The measured metrics.
        precision (str): The precision of the training stages. Defaults to "".
        label_type (str): The label type of the parity stage. Defaults to "".

    Returns:
        Dict[str, Any]: The row.
    """
    return {"case": case_name, "stage": stage, "variant": variant["variant"], "setting": variant["setting"],
            "cache": cache, "precision": precision, "label_type": label_type, "repeat": repeat, **metrics}


def run_loader_stage(case_name: str, variants: List[Dict[str, Any]], train_dataset: Dataset,
                     current_train_loader: DataLoader, batch_size: int, settings: Dict[str, Any],
                     device: torch.device, dtype: torch.dtype, frames_per_sample: Optional[int], cache: str,
                     train_file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Runs stage A for loader variants on the training split: every variant is built once (its build time recorded),
    then measured alone at every repeat, the files being evicted from the page cache before every cold repeat. The
    page cache is filled again after the cold runs, for the next warm stages.

    Args:
        case_name (str): The case.
        variants (List[Dict[str, Any]]): The measured variants.
        train_dataset (Dataset): The Native dataset of the training split.
        current_train_loader (DataLoader): The current Native training loader.
        batch_size (int): The number of samples of a batch.
        settings (Dict[str, Any]): The benchmark settings.
        device (torch.device): The device of the model.
        dtype (torch.dtype): The precision of the model.
        frames_per_sample (Optional[int]): The number of frames of a sample of the sequence tasks.
        cache (str): "warm" or "cold".
        train_file_paths (List[str]): The files of the training split.

    Returns:
        List[Dict[str, Any]]: The measurement rows.
    """
    rows = []
    for variant in variants:
        build_start_time = time.perf_counter()
        loader = build_variant_loader(variant=variant, dataset=train_dataset, current_loader=current_train_loader,
                                      batch_size=batch_size, settings=settings, shuffle=True)
        build_seconds = time.perf_counter() - build_start_time

        # Repeats of the measurement, the files read from the disk for the cold ones
        for repeat in range(settings["number_repeats"]):
            if cache == "cold":
                evict_from_page_cache(file_paths=train_file_paths)
            metrics = measure_loader(loader=loader, number_warmup_batches=settings["number_warmup_batches"],
                                     number_measured_batches=settings["number_measured_batches"], device=device,
                                     dtype=dtype, frames_per_sample=frames_per_sample)
            rows.append(make_row(case_name=case_name, stage="A", variant=variant, cache=cache, repeat=repeat,
                                 metrics={**metrics, "build_seconds": build_seconds}))
            print_color(color="blue", output=f"{case_name} A {cache} {variant['variant']} {variant['setting']} : "
                                             f"{metrics['samples_per_second']:.1f} samples/s", indent=1)
        del loader
        gc.collect()

    if cache == "cold":
        load_into_page_cache(file_paths=train_file_paths)
    return rows


def select_best_variant(rows: List[Dict[str, Any]], variants: List[Dict[str, Any]], family: str) -> Dict[str, Any]:
    """
    Chooses the fastest variant of a loader family from its warm stage A measurements (median of the repeats), the
    one measured in the cold, training and evaluation stages.

    Args:
        rows (List[Dict[str, Any]]): The measurement rows.
        variants (List[Dict[str, Any]]): The variants of the case.
        family (str): "native_tuned" or "dali".

    Returns:
        Dict[str, Any]: The descriptor of the fastest variant.
    """
    rates = {row["setting"]: row["samples_per_second"] for row in summarise_rows(rows=rows)
             if row["stage"] == "A" and row["cache"] == "warm" and row["variant"] == family}
    best_setting = max(rates, key=rates.get)
    return [variant for variant in variants if variant["variant"] == family and variant["setting"] == best_setting][0]


def run_ceiling_stage(case_name: str, model: TaskModel, optimizer: torch.optim.Optimizer, batch: Dict[str, Any],
                      task: str, augmenter: DataAugmenter, sequence_length: Optional[int], device: torch.device,
                      dtype: torch.dtype, settings: Dict[str, Any],
                      frames_per_sample: Optional[int]) -> List[Dict[str, Any]]:
    """
    Runs stage B: the GPU ceiling of the model of the case in float32 (the precision of the configurations) and in
    bfloat16, at every repeat.

    Args:
        case_name (str): The case.
        model (TaskModel): The model of the case, in training mode.
        optimizer (torch.optim.Optimizer): The optimizer of the model.
        batch (Dict[str, Any]): A loader batch of the case.
        task (str): The task of the case.
        augmenter (DataAugmenter): The augmenter of the case.
        sequence_length (Optional[int]): The sequence length of the sequence tasks.
        device (torch.device): The device of the model.
        dtype (torch.dtype): The precision of the model.
        settings (Dict[str, Any]): The benchmark settings.
        frames_per_sample (Optional[int]): The number of frames of a sample of the sequence tasks.

    Returns:
        List[Dict[str, Any]]: The measurement rows.
    """
    rows = []
    for precision in ["float32", "bfloat16"]:
        for repeat in range(settings["number_repeats"]):
            metrics = measure_training_ceiling(model=model, optimizer=optimizer, batch=batch, task=task,
                                               augmenter=augmenter, sequence_length=sequence_length,
                                               mixed_precision=precision == "bfloat16", device=device, dtype=dtype,
                                               number_warmup_iterations=settings["number_ceiling_warmup_iterations"],
                                               number_iterations=settings["number_ceiling_iterations"],
                                               frames_per_sample=frames_per_sample)
            rows.append(make_row(case_name=case_name, stage="B", variant={"variant": "gpu", "setting": "fixed batch"},
                                 cache="", repeat=repeat, metrics=metrics, precision=precision))
            print_color(color="blue", output=f"{case_name} B {precision} : {metrics['samples_per_second']:.1f} "
                                             f"samples/s", indent=1)
    return rows


def run_training_and_evaluation_stages(case_name: str, variant: Dict[str, Any], train_dataset: Dataset,
                                       test_dataset: Dataset, current_loaders: Tuple[DataLoader, ...],
                                       model: TaskModel, optimizer: torch.optim.Optimizer, task: str,
                                       augmenter: DataAugmenter, sequence_length: Optional[int],
                                       mixed_precision: bool, device: torch.device, dtype: torch.dtype,
                                       batch_size: int, settings: Dict[str, Any],
                                       frames_per_sample: Optional[int]) -> List[Dict[str, Any]]:
    """
    Runs stages C and D for one variant: the training iteration fed by its training loader, then the evaluation pass
    fed by its test loader, at every repeat. A DALI test loader is built again for every pass so that the pass starts
    at the beginning of an epoch (its build, done once in a training run, is not timed), and reads the test split in
    order like an evaluation would.

    Args:
        case_name (str): The case.
        variant (Dict[str, Any]): The descriptor of the variant.
        train_dataset (Dataset): The Native dataset of the training split.
        test_dataset (Dataset): The Native dataset of the test split.
        current_loaders (Tuple[DataLoader, ...]): The current Native training, validation and test loaders.
        model (TaskModel): The model of the case.
        optimizer (torch.optim.Optimizer): The optimizer of the model.
        task (str): The task of the case.
        augmenter (DataAugmenter): The augmenter of the case.
        sequence_length (Optional[int]): The sequence length of the sequence tasks.
        mixed_precision (bool): Whether the configuration trains in bfloat16.
        device (torch.device): The device of the model.
        dtype (torch.dtype): The precision of the model.
        batch_size (int): The number of samples of a batch.
        settings (Dict[str, Any]): The benchmark settings.
        frames_per_sample (Optional[int]): The number of frames of a sample of the sequence tasks.

    Returns:
        List[Dict[str, Any]]: The measurement rows.
    """
    rows = []
    precision = "bfloat16" if mixed_precision else "float32"

    # Training iteration
    train_loader = build_variant_loader(variant=variant, dataset=train_dataset, current_loader=current_loaders[0],
                                        batch_size=batch_size, settings=settings, shuffle=True)
    for repeat in range(settings["number_repeats"]):
        metrics = measure_end_to_end(loader=train_loader, model=model, optimizer=optimizer, task=task,
                                     augmenter=augmenter, sequence_length=sequence_length,
                                     mixed_precision=mixed_precision, device=device, dtype=dtype,
                                     number_warmup_iterations=settings["number_warmup_batches"],
                                     number_iterations=settings["number_end_to_end_iterations"],
                                     frames_per_sample=frames_per_sample)
        rows.append(make_row(case_name=case_name, stage="C", variant=variant, cache="warm", repeat=repeat,
                             metrics=metrics, precision=precision))
        print_color(color="blue", output=f"{case_name} C {variant['variant']} {variant['setting']} : "
                                         f"{metrics['samples_per_second']:.1f} samples/s", indent=1)
    del train_loader
    gc.collect()

    # Evaluation pass, the Native test loaders being kept between passes like in a training run
    test_loader = None
    for repeat in range(settings["number_repeats"]):
        if test_loader is None or variant["variant"] == "dali":
            test_loader = None
            gc.collect()
            test_loader = build_variant_loader(variant=variant, dataset=test_dataset, current_loader=current_loaders[2],
                                               batch_size=batch_size, settings=settings,
                                               shuffle=variant["variant"] != "dali")
        metrics = measure_evaluation_pass(loader=test_loader, model=model, task=task, sequence_length=sequence_length,
                                          mixed_precision=mixed_precision, device=device, dtype=dtype,
                                          maximum_batches=settings["number_evaluation_batches"],
                                          frames_per_sample=frames_per_sample)
        rows.append(make_row(case_name=case_name, stage="D", variant=variant, cache="warm", repeat=repeat,
                             metrics=metrics, precision=precision))
        print_color(color="blue", output=f"{case_name} D {variant['variant']} {variant['setting']} : "
                                         f"{metrics['samples_per_second']:.1f} samples/s", indent=1)
    del test_loader
    gc.collect()
    return rows


def get_parity_label_type(dataset: Dataset) -> str:
    """
    Names the labels of a dataset in the parity rows: the output mode of a classification dataset, the label type
    of a window dataset, or "pair".

    Args:
        dataset (Dataset): A Native dataset of the benchmark.

    Returns:
        str: The label type.
    """
    if isinstance(dataset, ClassificationDataset):
        return dataset.output_mode
    if isinstance(dataset, SequencePairDataset):
        return "pair"
    return dataset.label_type


def run_parity_stage(case_name: str, case_definition: Dict[str, Any], test_dataset: Dataset,
                     experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                     variant: Dict[str, Any], batch_size: int, settings: Dict[str, Any],
                     parity_folder: Path) -> List[Dict[str, Any]]:
    """
    Runs stage E: the samples of the DALI loader of the test split compared with the Native ones, and for the
    sequence classification case also with frame labels, the sequence labelling task reading the same windows.

    Args:
        case_name (str): The case.
        case_definition (Dict[str, Any]): The definition of the case.
        test_dataset (Dataset): The Native dataset of the test split.
        experiment_configuration (Dict[str, Any]): The processed experiment configuration of the case.
        model_configuration (Dict[str, Any]): The processed model configuration of the case.
        variant (Dict[str, Any]): The descriptor of the DALI variant compared.
        batch_size (int): The number of samples of a batch.
        settings (Dict[str, Any]): The benchmark settings.
        parity_folder (Path): The folder of the comparison images.

    Returns:
        List[Dict[str, Any]]: The measurement rows.
    """
    datasets = [test_dataset]

    # The windows of the test split with one class per frame, as read by the sequence labelling task
    if case_definition["frame_label_parity"]:
        data_configuration = model_configuration["Data"]
        test_folder = Path(experiment_configuration["test_data_folder"])
        datasets.append(SequenceDataset(data_directory=str(test_folder / "data"),
                                        labels_file=str(test_folder / "labels.json"),
                                        image_size=data_configuration["image_settings"]["size"],
                                        keep_ratio=data_configuration["image_settings"]["keep_ratio"],
                                        label_dictionary=data_configuration["label_dictionary"],
                                        sequence_settings=data_configuration["sequence_settings"], label_type="frame"))

    rows = []
    parity_folder.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        label_type = get_parity_label_type(dataset=dataset)
        loader = build_dali_loader(dataset=dataset, batch_size=batch_size, number_threads=variant["number_threads"],
                                   number_reader_processes=variant["number_reader_processes"],
                                   prefetch_queue_depth=settings["dali_prefetch_queue_depth"], shuffle=False)
        metrics = compare_loader_outputs(native_dataset=dataset, dali_loader=loader,
                                         number_samples=settings["number_parity_samples"],
                                         example_image_path=parity_folder / f"{case_name}_{label_type}.png")
        rows.append(make_row(case_name=case_name, stage="E", variant=variant, cache="", repeat=0, metrics=metrics,
                             label_type=label_type))
        print_color(color="blue", output=f"{case_name} E {label_type} : mean difference "
                                         f"{metrics['mean_pixel_difference']:.3f}, label mismatches "
                                         f"{metrics['label_mismatches']}", indent=1)
        del loader
        gc.collect()
    return rows


def run_benchmark_case(case_name: str, case_definition: Dict[str, Any], settings: Dict[str, Any],
                       case_output_folder: Path) -> None:
    """
    Measures one case in its own process: its configuration and current Native loaders, then the Native variants of
    stage A (their worker processes being forked before any DALI thread exists), the GPU ceiling, the training and
    evaluation stages of the current and best tuned Native loaders, the DALI variants of stage A, the training and
    evaluation stages of the best DALI loader, and the parity. The rows are written to '<case folder>/rows.json'.

    Args:
        case_name (str): The case.
        case_definition (Dict[str, Any]): The definition of the case (build_benchmark_cases).
        settings (Dict[str, Any]): The benchmark settings.
        case_output_folder (Path): The folder of the files of the case.
    """
    # Fork the DataLoader workers like a training process does: this process was spawned, which made "spawn" its
    # default start method and would start every worker as a new interpreter (about 0.8 s each, one after another)
    multiprocessing.set_start_method(method="fork", force=True)

    case_output_folder.mkdir(parents=True, exist_ok=True)
    experiment_configuration, model_configuration = write_benchmark_configuration(
        template_path=case_definition["template"], output_path=case_output_folder / "configuration.yaml",
        experiment_folder=case_output_folder / "experiments", split_folders=case_definition["split_folders"],
        label_mapping_path=case_definition["label_mapping_path"])

    # Settings of the case, as read by the trainer of its task
    task = experiment_configuration["task"]
    batch_size = experiment_configuration["batch_size"]
    device, dtype = get_device(), experiment_configuration["dtype"]
    mixed_precision = model_configuration["Optimization"]["mixed_precision"]
    is_sequence_case = task != "classification"
    sequence_length = model_configuration["Data"]["sequence_settings"]["sequence_length"] if is_sequence_case else None
    frames_per_sample = sequence_length * (2 if task == "sequence_comparison" else 1) if is_sequence_case else None

    # Current Native loaders and the files of the measured splits, loaded into the page cache
    current_loaders = build_current_native_loaders(experiment_configuration=experiment_configuration,
                                                   model_configuration=model_configuration)
    train_dataset, test_dataset = current_loaders[0].dataset, current_loaders[2].dataset
    train_file_paths = get_dataset_file_paths(dataset=train_dataset)
    load_into_page_cache(file_paths=train_file_paths + get_dataset_file_paths(dataset=test_dataset))
    variants = list_loader_variants(settings=settings, is_sequence_case=is_sequence_case)
    native_variants = [variant for variant in variants if variant["variant"] != "dali"]
    dali_variants = [variant for variant in variants if variant["variant"] == "dali"]
    stage_arguments = {"case_name": case_name, "train_dataset": train_dataset,
                       "current_train_loader": current_loaders[0], "batch_size": batch_size, "settings": settings,
                       "device": device, "dtype": dtype, "frames_per_sample": frames_per_sample,
                       "train_file_paths": train_file_paths}

    # Stage A of the Native loaders, warm then cold for the current and best tuned ones
    rows = run_loader_stage(variants=native_variants, cache="warm", **stage_arguments)
    best_tuned_variant = select_best_variant(rows=rows, variants=native_variants, family="native_tuned")
    if case_definition["cold_cache"]:
        rows += run_loader_stage(variants=[native_variants[0], best_tuned_variant], cache="cold", **stage_arguments)

    # Model of the case, its GPU ceiling, then stages C and D of the current and best tuned Native loaders
    model = TASK_MODEL_CLASSES[task](configuration=model_configuration, device=device, dtype=dtype, verbose=False)
    model.to(device=device, dtype=dtype)
    model.train()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=BENCHMARK_LEARNING_RATE)
    augmenter = DataAugmenter(augmentation_configuration=model_configuration["Data"]["data_augmentation"])
    rows += run_ceiling_stage(case_name=case_name, model=model, optimizer=optimizer,
                              batch=next(iter(current_loaders[0])), task=task, augmenter=augmenter,
                              sequence_length=sequence_length, device=device, dtype=dtype, settings=settings,
                              frames_per_sample=frames_per_sample)
    training_arguments = {"case_name": case_name, "train_dataset": train_dataset, "test_dataset": test_dataset,
                          "current_loaders": current_loaders, "model": model, "optimizer": optimizer, "task": task,
                          "augmenter": augmenter, "sequence_length": sequence_length,
                          "mixed_precision": mixed_precision, "device": device, "dtype": dtype,
                          "batch_size": batch_size, "settings": settings, "frames_per_sample": frames_per_sample}
    for variant in [native_variants[0], best_tuned_variant]:
        rows += run_training_and_evaluation_stages(variant=variant, **training_arguments)

    # Stages A, C and D of the DALI loaders
    rows += run_loader_stage(variants=dali_variants, cache="warm", **stage_arguments)
    best_dali_variant = select_best_variant(rows=rows, variants=dali_variants, family="dali")
    if case_definition["cold_cache"]:
        rows += run_loader_stage(variants=[best_dali_variant], cache="cold", **stage_arguments)
    rows += run_training_and_evaluation_stages(variant=best_dali_variant, **training_arguments)

    # Parity of the DALI outputs
    rows += run_parity_stage(case_name=case_name, case_definition=case_definition, test_dataset=test_dataset,
                             experiment_configuration=experiment_configuration,
                             model_configuration=model_configuration, variant=best_dali_variant,
                             batch_size=batch_size, settings=settings,
                             parity_folder=case_output_folder.parent / "Parity")
    (case_output_folder / "rows.json").write_text(data=json.dumps(obj=rows))


def run_case_in_process(case_name: str, case_definition: Dict[str, Any], settings: Dict[str, Any],
                        case_output_folder: Path) -> List[Dict[str, Any]]:
    """
    Runs a case in a new process (spawn), so that the worker processes, DALI pipelines and CUDA context of a case do
    not affect the next one, and reads its rows. A failed case gives no row and a red message.

    Args:
        case_name (str): The case.
        case_definition (Dict[str, Any]): The definition of the case.
        settings (Dict[str, Any]): The benchmark settings.
        case_output_folder (Path): The folder of the files of the case.

    Returns:
        List[Dict[str, Any]]: The rows of the case.
    """
    process = multiprocessing.get_context(method="spawn").Process(
        target=run_benchmark_case, kwargs={"case_name": case_name, "case_definition": case_definition,
                                           "settings": settings, "case_output_folder": case_output_folder})
    process.start()
    process.join()
    rows_path = case_output_folder / "rows.json"
    if process.exitcode != 0 or not rows_path.exists():
        print_color(color="red", output=f"Case {case_name} failed (exit code {process.exitcode}) : "
                                        f"file://{case_output_folder}")
        return []
    return json.loads(s=rows_path.read_text())


def run_disk_stage(imagenet_split_folder: Path, subset_manifest_path: Path, number_files: int, number_threads: int,
                   seed: int) -> List[Dict[str, Any]]:
    """
    Runs stage F: the cold read rate of the disk on ImageNet training files outside the ImageNet subset, the upper
    bound of a loader reading ImageNet, which is larger than the memory.

    Args:
        imagenet_split_folder (Path): The ImageNet training split.
        subset_manifest_path (Path): The manifest of the ImageNet subset, whose files are left out.
        number_files (int): The number of files read.
        number_threads (int): The number of reading threads.
        seed (int): The seed of the draw of the files.

    Returns:
        List[Dict[str, Any]]: The measurement row.
    """
    subset_file_names = {sample["file_name"] for sample in json.loads(s=subset_manifest_path.read_text())["samples"]}
    file_names = [sample["file_name"] for sample in json.loads(s=(imagenet_split_folder / "labels.json").read_text())[
        "samples"] if sample["file_name"] not in subset_file_names]
    file_paths = [str(imagenet_split_folder / "data" / file_name)
                  for file_name in random.Random(seed).sample(population=file_names, k=number_files)]
    metrics = measure_disk_read_rate(file_paths=file_paths, number_threads=number_threads)
    print_color(color="blue", output=f"Disk : {metrics['megabytes_per_second']:.0f} MB/s, "
                                     f"{metrics['files_per_second']:.0f} files/s", indent=1)
    return [make_row(case_name="DISK", stage="F", variant={"variant": "disk", "setting":
                     f"{number_files} ImageNet files, {number_threads} threads"}, cache="cold", repeat=0,
                     metrics=metrics)]


def describe_environment(settings: Dict[str, Any]) -> List[str]:
    """
    Describes the machine and the settings of a benchmark run, written to environment.txt and summary.md.

    Args:
        settings (Dict[str, Any]): The benchmark settings.

    Returns:
        List[str]: The Markdown lines of the description.
    """
    gpu_description = subprocess.run(args=["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.max.sm",
                                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    governor_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    governor = governor_path.read_text().strip() if governor_path.exists() else "unknown"
    return [f"- Date: {time.strftime('%Y-%m-%d %H:%M')}", f"- GPU: {gpu_description}",
            f"- CPU: {os.cpu_count()} cores ({platform.machine()}), frequency policy {governor}",
            f"- Libraries: torch {torch.__version__} (CUDA {torch.version.cuda}), DALI {dali.__version__}, "
            f"OpenCV {cv2.__version__}", f"- Settings: {json.dumps(obj=settings)}"]


def run_loader_benchmark(output_root: Path, imagenet_folder: Path, sequence_data_folder: Path, case_names: List[str],
                         settings: Dict[str, Any], imagenet_subset_size: int, number_disk_files: int,
                         seed: int) -> Path:
    """
    Runs the loader benchmark: the ImageNet subset of the classification case, the disk stage, then every case in its
    own process, the reports being written again after every case so that the finished cases survive a failure.

    Args:
        output_root (Path): The folder of the benchmark runs and of the ImageNet subset.
        imagenet_folder (Path): The ImageNet folder prepared by imagenet_download.py.
        sequence_data_folder (Path): The folder of the sequence datasets.
        case_names (List[str]): The measured cases, in order.
        settings (Dict[str, Any]): The benchmark settings.
        imagenet_subset_size (int): The number of training images of the ImageNet subset.
        number_disk_files (int): The number of files read by the disk stage.
        seed (int): The seed of the draws of the subset and of the disk files.

    Returns:
        Path: The folder of the run.
    """
    output_folder = output_root / time.strftime("%Y%m%d_%H%M%S")
    output_folder.mkdir(parents=True)
    environment_lines = describe_environment(settings=settings)
    (output_folder / "environment.txt").write_text(data="\n".join(environment_lines) + "\n")
    print_color(color="blue", output=f"Loader Benchmark : file://{output_folder}", add_separators=True)

    # ImageNet subset of the classification case, then the cold read rate of the disk
    subset_manifest_path = create_benchmark_subset_manifest(
        source_split_folder=imagenet_folder / "manifests" / "train",
        output_split_folder=output_root / "datasets" / "imagenet_subset" / "train", number_samples=imagenet_subset_size,
        seed=seed)
    rows = run_disk_stage(imagenet_split_folder=imagenet_folder / "manifests" / "train",
                          subset_manifest_path=subset_manifest_path, number_files=number_disk_files, number_threads=8,
                          seed=seed)

    # Every case in its own process
    cases = build_benchmark_cases(imagenet_folder=imagenet_folder, sequence_data_folder=sequence_data_folder,
                                  benchmark_data_folder=output_root / "datasets")
    for case_name in case_names:
        print_color(color="blue", output=f"Case {case_name}", add_separators=True)
        rows += run_case_in_process(case_name=case_name, case_definition=cases[case_name], settings=settings,
                                    case_output_folder=output_folder / case_name)
        write_benchmark_reports(rows=rows, output_folder=output_folder, environment_lines=environment_lines)
    return output_folder


if __name__ == "__main__":
    # Folder of the benchmark runs (one subfolder per run) and of the ImageNet subset
    output_root = Path("/home/robert_kelevra/Data/loader_benchmark")
    # ImageNet prepared by vision_training/data/classification/imagenet_download.py, and the folder of the sequence
    # datasets holding the benchmark sequences and pairs built by benchmark_datasets.py
    imagenet_folder = Path("/home/robert_kelevra/Data/object_detection_data/imagenet-1k")
    sequence_data_folder = Path("/home/robert_kelevra/Data/sequence_data")
    # Measured cases, in order (section 5.3 of the plan)
    case_names = ["C-IN", "C-SYN", "S-SYN", "S-JPG", "P-SYN", "P-JPG"]
    # Number of training images of the ImageNet subset (5.5 GB, held in memory), and number of untouched ImageNet
    # files read by the disk stage
    imagenet_subset_size = 50000
    number_disk_files = 20000
    # Measurement settings: repeats, batches and iterations of every stage, and the swept loader parameters. The
    # training stages use fewer iterations than the loader stage, the example models training at a few iterations
    # per second
    settings = {"number_repeats": 3, "number_warmup_batches": 20, "number_measured_batches": 200,
                "number_ceiling_warmup_iterations": 10, "number_ceiling_iterations": 30,
                "number_end_to_end_iterations": 100, "number_evaluation_batches": 100, "number_parity_samples": 256,
                "native_worker_counts": [8, 12, 16], "native_prefetch_factors": [2, 4],
                "dali_thread_counts": [4, 8, 12, 16], "dali_reader_process_counts": [2, 4],
                "dali_prefetch_queue_depth": 2}
    # Seed of the draws of the ImageNet subset and of the disk files
    seed = 0

    run_loader_benchmark(output_root=output_root, imagenet_folder=imagenet_folder,
                         sequence_data_folder=sequence_data_folder, case_names=case_names, settings=settings,
                         imagenet_subset_size=imagenet_subset_size, number_disk_files=number_disk_files, seed=seed)
