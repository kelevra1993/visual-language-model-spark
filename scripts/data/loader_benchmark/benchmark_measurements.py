"""
Measurements of the loader benchmark of implementation_dali_loaders.md (phase 1): the timing of a loader alone, of
the GPU training step without any loading (the ceiling a loader can feed), of the training iteration and of the
evaluation pass fed by a loader, the sampling of the CPU and GPU utilisation, the read rate of the disk, the parity of
the DALI outputs with the Native ones, and the CSV and Markdown reports. Used by run_loader_benchmark.py.
"""

import csv
import math
import statistics
import subprocess
import tempfile
import time
import cv2
import numpy as np
import psutil
import torch

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from torch.utils.data import Dataset
from typing import Any, Dict, Iterable, List, Optional, Tuple

from scripts.data.loader_benchmark.benchmark_datasets import evict_from_page_cache, read_file_bytes
from vision_training.core.contracts import LOSS_KEY_SUFFIX
from vision_training.data.augmentations import DataAugmenter
from vision_training.data.classification.classification_dataset import ClassificationDataset
from vision_training.data.sequence.sequence_pair_dataset import SequencePairDataset, split_pair_members
from vision_training.models.task_model import TaskModel
from vision_training.trainers.base_trainer import BaseTrainer
from vision_training.utilities.os_utilities import print_color

# Difference above which a DALI pixel counts as different from the Native one in the parity stage, in levels
PARITY_PIXEL_THRESHOLD = 8

# Interval between two samples of the GPU utilisation, in seconds
UTILISATION_SAMPLING_INTERVAL = 0.5

# Columns identifying a measurement in the reports, the other columns being its metrics
IDENTIFICATION_COLUMNS = ["case", "stage", "variant", "setting", "cache", "precision", "label_type", "repeat"]

# Loader families in the order of the reports, the first one being the reference of the speedups
VARIANT_ORDER = ["native_current", "native_tuned", "dali"]


def limit_opencv_threads(worker_index: int) -> None:
    """
    Initialises a worker process of the tuned Native loaders with a single OpenCV thread, OpenCV otherwise starting
    one thread per core in every worker (20 here), which oversubscribes the CPU once several workers decode at once.

    Args:
        worker_index (int): The index of the worker, given by the DataLoader.
    """
    cv2.setNumThreads(1)


def prepare_benchmark_batch(batch: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    """
    Moves a loader batch to the GPU exactly like the prepare_batch hooks of the classification and sequence trainers:
    the images in the precision of the model, the labels as int64 class indices (or multi-hot vectors in the
    precision of the model), the sequence mask as booleans.

    Args:
        batch (Dict[str, Any]): The batch yielded by a loader.
        device (torch.device): The device of the model.
        dtype (torch.dtype): The precision of the model.

    Returns:
        Dict[str, Any]: The prepared batch.
    """
    labels = batch["labels"]
    prepared_batch = {**batch, "images": batch["images"].to(device=device, dtype=dtype),
                      "labels": labels.to(device=device, dtype=torch.int64 if labels.dtype == torch.int64 else dtype)}
    if "sequence_mask" in batch:
        prepared_batch["sequence_mask"] = batch["sequence_mask"].to(device=device, dtype=torch.bool)
    return prepared_batch


def augment_benchmark_batch(batch: Dict[str, Any], augmenter: DataAugmenter,
                            frame_block_length: Optional[int]) -> Dict[str, Any]:
    """
    Augments a prepared training batch like the augment_batch hooks of the trainers: the DataAugmenter draws, then for
    the sequence tasks the frame shuffling of 'shuffle_frames', the frame labels following their frames.

    Args:
        batch (Dict[str, Any]): The prepared batch.
        augmenter (DataAugmenter): The augmenter built from the configuration of the case.
        frame_block_length (Optional[int]): The sequence length for the sequence tasks, None for classification.

    Returns:
        Dict[str, Any]: The augmented batch.
    """
    augmented_images, _, _ = augmenter.augment_batch(images=batch["images"])
    augmented_batch = {**batch, "images": augmented_images}
    if frame_block_length is not None:
        augmented_batch["images"], frame_orders = augmenter.shuffle_frames(images=augmented_images,
                                                                           sequence_mask=batch["sequence_mask"],
                                                                           frame_block_length=frame_block_length)
        if augmented_batch["labels"].dim() == 2:
            augmented_batch["labels"] = torch.gather(input=augmented_batch["labels"], dim=1, index=frame_orders)
    return augmented_batch


def run_model(model: TaskModel, batch: Dict[str, Any], task: str, sequence_length: Optional[int],
              with_labels: bool) -> Dict[str, Any]:
    """
    Calls the model of a case like the forward_pass hooks of the trainers (with the labels, to get the losses) or like
    their predict_batch hooks (without them, as in the evaluation).

    Args:
        model (TaskModel): The model of the case.
        batch (Dict[str, Any]): The prepared batch.
        task (str): The task of the case.
        sequence_length (Optional[int]): The sequence length of the sequence tasks.
        with_labels (bool): Whether to give the labels, so that the model returns its losses.

    Returns:
        Dict[str, Any]: The model outputs.
    """
    labels_argument = {"labels": batch["labels"]} if with_labels else {}

    # Pairs split into their before and after members
    if task == "sequence_comparison":
        before_images, after_images = split_pair_members(pair_tensor=batch["images"], sequence_length=sequence_length)
        before_mask, after_mask = split_pair_members(pair_tensor=batch["sequence_mask"],
                                                     sequence_length=sequence_length)
        return model(before_images=before_images, after_images=after_images, before_mask=before_mask,
                     after_mask=after_mask, **labels_argument)

    # Windows with their mask, or images
    if "sequence_mask" in batch:
        return model(input_tensor=batch["images"], sequence_mask=batch["sequence_mask"], **labels_argument)
    return model(input_tensor=batch["images"], **labels_argument)


def run_training_step(model: TaskModel, optimizer: torch.optim.Optimizer, batch: Dict[str, Any], task: str,
                      augmenter: Optional[DataAugmenter], sequence_length: Optional[int], mixed_precision: bool,
                      device: torch.device) -> torch.Tensor:
    """
    Runs one training step on a prepared batch like BaseTrainer.run_training_loop: augmentation, forward pass with
    the labels (in bfloat16 with mixed precision), sum of the losses, backward pass and optimizer step.

    Args:
        model (TaskModel): The model of the case, in training mode.
        optimizer (torch.optim.Optimizer): The optimizer of the model.
        batch (Dict[str, Any]): The prepared batch.
        task (str): The task of the case.
        augmenter (Optional[DataAugmenter]): The augmenter, or None to skip the augmentation.
        sequence_length (Optional[int]): The sequence length of the sequence tasks.
        mixed_precision (bool): Whether to run the forward pass in bfloat16.
        device (torch.device): The device of the model.

    Returns:
        torch.Tensor: The total loss, whose item() the caller reads like the metric tracker of the trainer.
    """
    optimizer.zero_grad()
    if augmenter is not None:
        batch = augment_benchmark_batch(batch=batch, augmenter=augmenter, frame_block_length=sequence_length)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=mixed_precision):
        model_outputs = run_model(model=model, batch=batch, task=task, sequence_length=sequence_length,
                                  with_labels=True)
    total_loss = sum(value for key, value in model_outputs.items() if key.endswith(LOSS_KEY_SUFFIX))
    total_loss.backward()
    optimizer.step()
    return total_loss


class UtilisationSampler:
    """
    Samples the GPU utilisation and the utilisation of its JPEG engine while a stage runs, to show whether the GPU
    waits for the loader: nvidia-smi samples them at a fixed interval in its own process, writing to a temporary
    file. A separate process and not a thread, since the DataLoader forks its workers during the stages, and a fork
    while another thread holds a lock can leave a worker blocked forever. The JPEG engine stays at 0 % on this machine,
    which has no usable hardware JPEG decoder (section 4 of the plan).
    """

    def __init__(self, interval: float) -> None:
        """
        Prepares the sampling.

        Args:
            interval (float): The interval between two samples, in seconds.
        """
        self.interval = interval
        self.output_file = tempfile.TemporaryFile(mode="w+")
        self.sampling_process: Optional[subprocess.Popen] = None

    def start(self) -> None:
        """
        Starts the nvidia-smi sampling process.
        """
        self.sampling_process = subprocess.Popen(args=["nvidia-smi", "--query-gpu=utilization.gpu,utilization.jpeg",
                                                       "--format=csv,noheader,nounits",
                                                       f"--loop-ms={int(self.interval * 1000)}"],
                                                 stdout=self.output_file, stderr=subprocess.DEVNULL)

    def stop_sampling(self) -> Dict[str, float]:
        """
        Stops the sampling process and averages its samples.

        Returns:
            Dict[str, float]: The mean 'gpu_utilisation' and 'jpeg_utilisation' in percent (NaN without samples).
        """
        self.sampling_process.terminate()
        self.sampling_process.wait()

        # One "gpu, jpeg" line per sample, the lines of an unavailable value being skipped
        gpu_samples, jpeg_samples = [], []
        self.output_file.seek(0)
        for line in self.output_file.read().splitlines():
            values = [value.strip() for value in line.split(",")]
            if len(values) == 2 and all(value.isdigit() for value in values):
                gpu_samples.append(float(values[0]))
                jpeg_samples.append(float(values[1]))
        self.output_file.close()
        return {"gpu_utilisation": statistics.fmean(gpu_samples) if gpu_samples else math.nan,
                "jpeg_utilisation": statistics.fmean(jpeg_samples) if jpeg_samples else math.nan}


def get_busy_cores(cpu_percent: float) -> float:
    """
    Converts the system-wide CPU utilisation measured by psutil over a stage into a number of busy cores.

    Args:
        cpu_percent (float): The utilisation of all the cores, in percent.

    Returns:
        float: The number of busy cores.
    """
    return cpu_percent / 100.0 * psutil.cpu_count()


def measure_loader(loader: Iterable, number_warmup_batches: int, number_measured_batches: int, device: torch.device,
                   dtype: torch.dtype, frames_per_sample: Optional[int]) -> Dict[str, float]:
    """
    Measures a loader alone (stage A): the time to its first batch from a new iterator, then the throughput of the
    measured batches after the warm-up ones, every batch being moved to the GPU like the trainers do. The epochs are
    chained like BaseTrainer.get_next_batch, a new iterator being created at the end of every epoch.

    Args:
        loader (Iterable): The loader.
        number_warmup_batches (int): The number of batches read before the measurement.
        number_measured_batches (int): The number of measured batches.
        device (torch.device): The device of the model.
        dtype (torch.dtype): The precision of the model.
        frames_per_sample (Optional[int]): The number of frames of a sample of the sequence tasks.

    Returns:
        Dict[str, float]: 'first_batch_seconds', 'samples_per_second', 'frames_per_second', 'batches_per_second' and
            'cpu_busy_cores'.
    """
    # First batch of a new iterator, then the warm-up batches
    start_time = time.perf_counter()
    loader_iterator = iter(loader)
    for batch_index in range(max(number_warmup_batches, 1)):
        batch, loader_iterator = BaseTrainer.get_next_batch(dataloader_iterator=loader_iterator, dataloader=loader)
        prepare_benchmark_batch(batch=batch, device=device, dtype=dtype)
        if batch_index == 0:
            torch.cuda.synchronize()
            first_batch_seconds = time.perf_counter() - start_time
    torch.cuda.synchronize()

    # Measured batches
    psutil.cpu_percent(interval=None)
    start_time = time.perf_counter()
    number_samples = 0
    for _ in range(number_measured_batches):
        batch, loader_iterator = BaseTrainer.get_next_batch(dataloader_iterator=loader_iterator, dataloader=loader)
        number_samples += prepare_benchmark_batch(batch=batch, device=device, dtype=dtype)["images"].shape[0]
    torch.cuda.synchronize()
    duration = time.perf_counter() - start_time
    busy_cores = get_busy_cores(cpu_percent=psutil.cpu_percent(interval=None))

    del loader_iterator
    return {"first_batch_seconds": first_batch_seconds, "samples_per_second": number_samples / duration,
            "frames_per_second": number_samples * frames_per_sample / duration if frames_per_sample else math.nan,
            "batches_per_second": number_measured_batches / duration, "cpu_busy_cores": busy_cores}


def time_training_steps(model: TaskModel, optimizer: torch.optim.Optimizer, batch: Dict[str, Any], task: str,
                        augmenter: Optional[DataAugmenter], sequence_length: Optional[int], mixed_precision: bool,
                        device: torch.device, number_warmup_iterations: int, number_iterations: int) -> float:
    """
    Times training steps repeated on the same prepared batch, for the GPU ceiling of stage B.

    Args:
        model (TaskModel): The model of the case, in training mode.
        optimizer (torch.optim.Optimizer): The optimizer of the model.
        batch (Dict[str, Any]): The prepared batch, on the GPU.
        task (str): The task of the case.
        augmenter (Optional[DataAugmenter]): The augmenter, or None to time the step without augmentation.
        sequence_length (Optional[int]): The sequence length of the sequence tasks.
        mixed_precision (bool): Whether to run the forward pass in bfloat16.
        device (torch.device): The device of the model.
        number_warmup_iterations (int): The number of steps run before the measurement.
        number_iterations (int): The number of measured steps.

    Returns:
        float: The number of steps per second.
    """
    for iteration in range(number_warmup_iterations + number_iterations):
        if iteration == number_warmup_iterations:
            torch.cuda.synchronize()
            start_time = time.perf_counter()
        run_training_step(model=model, optimizer=optimizer, batch=batch, task=task, augmenter=augmenter,
                          sequence_length=sequence_length, mixed_precision=mixed_precision, device=device).item()
    torch.cuda.synchronize()
    return number_iterations / (time.perf_counter() - start_time)


def measure_training_ceiling(model: TaskModel, optimizer: torch.optim.Optimizer, batch: Dict[str, Any], task: str,
                             augmenter: DataAugmenter, sequence_length: Optional[int], mixed_precision: bool,
                             device: torch.device, dtype: torch.dtype, number_warmup_iterations: int,
                             number_iterations: int, frames_per_sample: Optional[int]) -> Dict[str, float]:
    """
    Measures the GPU ceiling of a case (stage B): the training throughput with no loading at all, one batch of the
    case being kept on the GPU and used for every step, with the augmentation alone, the step alone and both. A loader
    faster than this ceiling cannot speed the training up.

    Args:
        model (TaskModel): The model of the case, in training mode.
        optimizer (torch.optim.Optimizer): The optimizer of the model.
        batch (Dict[str, Any]): A loader batch of the case.
        task (str): The task of the case.
        augmenter (DataAugmenter): The augmenter built from the configuration of the case.
        sequence_length (Optional[int]): The sequence length of the sequence tasks.
        mixed_precision (bool): Whether to run the forward pass in bfloat16.
        device (torch.device): The device of the model.
        dtype (torch.dtype): The precision of the model.
        number_warmup_iterations (int): The number of iterations run before every measurement.
        number_iterations (int): The number of measured iterations.
        frames_per_sample (Optional[int]): The number of frames of a sample of the sequence tasks.

    Returns:
        Dict[str, float]: 'augmentation_milliseconds' per batch, 'step_iterations_per_second' without augmentation,
            and 'iterations_per_second', 'samples_per_second' and 'frames_per_second' with it.
    """
    prepared_batch = prepare_benchmark_batch(batch=batch, device=device, dtype=dtype)
    batch_size = prepared_batch["images"].shape[0]

    # Augmentation alone
    for iteration in range(number_warmup_iterations + number_iterations):
        if iteration == number_warmup_iterations:
            torch.cuda.synchronize()
            start_time = time.perf_counter()
        augment_benchmark_batch(batch=prepared_batch, augmenter=augmenter, frame_block_length=sequence_length)
    torch.cuda.synchronize()
    augmentation_milliseconds = (time.perf_counter() - start_time) / number_iterations * 1000.0

    # Training step without and with the augmentation
    step_rate = time_training_steps(model=model, optimizer=optimizer, batch=prepared_batch, task=task, augmenter=None,
                                    sequence_length=sequence_length, mixed_precision=mixed_precision, device=device,
                                    number_warmup_iterations=number_warmup_iterations,
                                    number_iterations=number_iterations)
    iteration_rate = time_training_steps(model=model, optimizer=optimizer, batch=prepared_batch, task=task,
                                         augmenter=augmenter, sequence_length=sequence_length,
                                         mixed_precision=mixed_precision, device=device,
                                         number_warmup_iterations=number_warmup_iterations,
                                         number_iterations=number_iterations)
    return {"augmentation_milliseconds": augmentation_milliseconds, "step_iterations_per_second": step_rate,
            "iterations_per_second": iteration_rate, "samples_per_second": iteration_rate * batch_size,
            "frames_per_second": iteration_rate * batch_size * frames_per_sample if frames_per_sample else math.nan}


def measure_end_to_end(loader: Iterable, model: TaskModel, optimizer: torch.optim.Optimizer, task: str,
                       augmenter: DataAugmenter, sequence_length: Optional[int], mixed_precision: bool,
                       device: torch.device, dtype: torch.dtype, number_warmup_iterations: int,
                       number_iterations: int, frames_per_sample: Optional[int]) -> Dict[str, float]:
    """
    Measures the training iteration fed by a loader (stage C), like BaseTrainer.run_training_loop without its
    validation and logging: next batch (epochs chained like get_next_batch), preparation, augmentation, training step
    and the item() of the loss read by the metric tracker. It is the throughput a training run would reach.

    Args:
        loader (Iterable): The training loader.
        model (TaskModel): The model of the case, in training mode.
        optimizer (torch.optim.Optimizer): The optimizer of the model.
        task (str): The task of the case.
        augmenter (DataAugmenter): The augmenter built from the configuration of the case.
        sequence_length (Optional[int]): The sequence length of the sequence tasks.
        mixed_precision (bool): Whether to run the forward pass in bfloat16.
        device (torch.device): The device of the model.
        dtype (torch.dtype): The precision of the model.
        number_warmup_iterations (int): The number of iterations run before the measurement.
        number_iterations (int): The number of measured iterations.
        frames_per_sample (Optional[int]): The number of frames of a sample of the sequence tasks.

    Returns:
        Dict[str, float]: 'iterations_per_second', 'samples_per_second', 'frames_per_second', 'gpu_utilisation',
            'jpeg_utilisation' and 'cpu_busy_cores'.
    """
    loader_iterator = iter(loader)
    sampler = UtilisationSampler(interval=UTILISATION_SAMPLING_INTERVAL)
    number_samples = 0
    for iteration in range(number_warmup_iterations + number_iterations):
        # Start the measurement once the warm-up iterations are done
        if iteration == number_warmup_iterations:
            torch.cuda.synchronize()
            sampler.start()
            psutil.cpu_percent(interval=None)
            start_time = time.perf_counter()

        batch, loader_iterator = BaseTrainer.get_next_batch(dataloader_iterator=loader_iterator, dataloader=loader)
        prepared_batch = prepare_benchmark_batch(batch=batch, device=device, dtype=dtype)
        run_training_step(model=model, optimizer=optimizer, batch=prepared_batch, task=task, augmenter=augmenter,
                          sequence_length=sequence_length, mixed_precision=mixed_precision, device=device).item()
        if iteration >= number_warmup_iterations:
            number_samples += prepared_batch["images"].shape[0]
    torch.cuda.synchronize()
    duration = time.perf_counter() - start_time
    busy_cores = get_busy_cores(cpu_percent=psutil.cpu_percent(interval=None))
    utilisation = sampler.stop_sampling()

    del loader_iterator
    return {"iterations_per_second": number_iterations / duration, "samples_per_second": number_samples / duration,
            "frames_per_second": number_samples * frames_per_sample / duration if frames_per_sample else math.nan,
            **utilisation, "cpu_busy_cores": busy_cores}


def measure_evaluation_pass(loader: Iterable, model: TaskModel, task: str, sequence_length: Optional[int],
                            mixed_precision: bool, device: torch.device, dtype: torch.dtype, maximum_batches: int,
                            frames_per_sample: Optional[int]) -> Dict[str, float]:
    """
    Measures an evaluation pass on the test split fed by a loader (stage D), like BaseTrainer.run_test_evaluation:
    a new iterator (whose start is part of every evaluation), preparation, prediction without gradients and the class
    probabilities moved to the CPU like the evaluation accumulators do, for at most 'maximum_batches' batches.

    Args:
        loader (Iterable): The test loader, at the start of an epoch.
        model (TaskModel): The model of the case.
        task (str): The task of the case.
        sequence_length (Optional[int]): The sequence length of the sequence tasks.
        mixed_precision (bool): Whether to run the forward pass in bfloat16.
        device (torch.device): The device of the model.
        dtype (torch.dtype): The precision of the model.
        maximum_batches (int): The largest number of batches of the pass.
        frames_per_sample (Optional[int]): The number of frames of a sample of the sequence tasks.

    Returns:
        Dict[str, float]: 'first_batch_seconds', 'pass_seconds', 'samples_per_second' and 'frames_per_second' over the
            whole pass (start of the loader included), 'steady_samples_per_second' after the first batch (NaN for a
            pass of one batch), 'full_pass_seconds' (the estimated duration of an evaluation of the whole test split:
            the first batch, then the other samples at the steady rate) and 'gpu_utilisation'.
    """
    model.eval()
    sampler = UtilisationSampler(interval=UTILISATION_SAMPLING_INTERVAL)
    sampler.start()
    number_samples, number_first_batch_samples = 0, 0
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            prepared_batch = prepare_benchmark_batch(batch=batch, device=device, dtype=dtype)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=mixed_precision):
                predictions = run_model(model=model, batch=prepared_batch, task=task, sequence_length=sequence_length,
                                        with_labels=False)
            predictions["class_probabilities"].float().cpu()
            number_samples += prepared_batch["images"].shape[0]

            # Time and samples of the first batch, which holds the start of the loader
            if batch_index == 0:
                first_batch_seconds = time.perf_counter() - start_time
                number_first_batch_samples = number_samples
            if batch_index + 1 >= maximum_batches:
                break
    torch.cuda.synchronize()
    pass_seconds = time.perf_counter() - start_time
    utilisation = sampler.stop_sampling()
    model.train()

    # Steady rate after the first batch, and the duration of an evaluation of the whole test split at that rate
    steady_samples = number_samples - number_first_batch_samples
    steady_rate = steady_samples / (pass_seconds - first_batch_seconds) if steady_samples > 0 else math.nan
    remaining_samples = len(loader.dataset) - number_first_batch_samples
    return {"first_batch_seconds": first_batch_seconds, "pass_seconds": pass_seconds,
            "samples_per_second": number_samples / pass_seconds,
            "frames_per_second": number_samples * frames_per_sample / pass_seconds if frames_per_sample else math.nan,
            "steady_samples_per_second": steady_rate,
            "full_pass_seconds": first_batch_seconds + remaining_samples / steady_rate if remaining_samples > 0
            else pass_seconds, "gpu_utilisation": utilisation["gpu_utilisation"]}


def measure_disk_read_rate(file_paths: List[str], number_threads: int) -> Dict[str, float]:
    """
    Measures the cold read rate of the disk (stage F): the files are evicted from the page cache, then read whole with
    several threads. It bounds the throughput of any loader reading a dataset larger than the memory.

    Args:
        file_paths (List[str]): The files read, outside every other benchmark dataset.
        number_threads (int): The number of reading threads.

    Returns:
        Dict[str, float]: 'megabytes_per_second' and 'files_per_second'.
    """
    evict_from_page_cache(file_paths=file_paths)
    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=number_threads) as executor:
        number_bytes = sum(future.result() for future in [executor.submit(read_file_bytes, file_path=file_path)
                                                          for file_path in file_paths])
    duration = time.perf_counter() - start_time
    return {"megabytes_per_second": number_bytes / duration / 1e6, "files_per_second": len(file_paths) / duration}


def get_dataset_records(dataset: Dataset) -> List[Dict[str, Any]]:
    """
    Gives the sample records of a Native dataset of the benchmark, whose 'identifier' keys match the sample
    identifiers of the batches.

    Args:
        dataset (Dataset): A ClassificationDataset, SequenceDataset or SequencePairDataset.

    Returns:
        List[Dict[str, Any]]: The records, in dataset order.
    """
    if isinstance(dataset, ClassificationDataset):
        return dataset.samples
    if isinstance(dataset, SequencePairDataset):
        return dataset.pairs
    return dataset.windows


def compare_loader_outputs(native_dataset: Dataset, dali_loader: Iterable, number_samples: int,
                           example_image_path: Path) -> Dict[str, float]:
    """
    Compares the samples of a DALI loader with those of the Native dataset (stage E), matched by their identifiers:
    labels and sequence masks must be equal, and the pixels of the images (of the real frames for the sequence tasks)
    close, the GPU JPEG decoder rounding differently from libjpeg-turbo. Writes the first sample as a
    Native | DALI | difference x 10 image.

    Args:
        native_dataset (Dataset): The Native dataset of the split.
        dali_loader (Iterable): The DALI loader of the same split.
        number_samples (int): The number of compared samples.
        example_image_path (Path): The written comparison image.

    Returns:
        Dict[str, float]: 'compared_samples', 'label_mismatches', 'mask_mismatches', 'mean_pixel_difference',
            'maximum_pixel_difference' and 'large_difference_percentage' (pixels differing by more than
            PARITY_PIXEL_THRESHOLD levels).
    """
    position_by_identifier = {record["identifier"]: position
                              for position, record in enumerate(get_dataset_records(dataset=native_dataset))}
    totals = {"compared_samples": 0, "label_mismatches": 0, "mask_mismatches": 0}
    difference_sum, number_pixels, number_large_differences, maximum_difference = 0.0, 0, 0, 0

    for batch in dali_loader:
        for sample_index in range(batch["images"].shape[0]):
            native_sample = native_dataset[position_by_identifier[int(batch["sample_identifiers"][sample_index])]]
            dali_images, native_images = batch["images"][sample_index].cpu(), native_sample["images"]

            # Masks of the sequence samples, the pixels being compared on the real frames only
            if "sequence_mask" in batch:
                totals["mask_mismatches"] += int(not torch.equal(input=batch["sequence_mask"][sample_index].cpu(),
                                                                 other=native_sample["sequence_mask"]))
                dali_images = dali_images[native_sample["sequence_mask"]]
                native_images = native_images[native_sample["sequence_mask"]]
            totals["label_mismatches"] += int(not torch.equal(input=batch["labels"][sample_index].cpu(),
                                                              other=native_sample["labels"]))

            # Pixel differences in levels
            differences = (dali_images.to(dtype=torch.int16) - native_images.to(dtype=torch.int16)).abs()
            difference_sum += float(differences.sum())
            number_pixels += differences.numel()
            number_large_differences += int((differences > PARITY_PIXEL_THRESHOLD).sum())
            maximum_difference = max(maximum_difference, int(differences.max()))

            # First sample (first real frame of a sequence) written next to its Native version
            if totals["compared_samples"] == 0:
                native_image = native_images.reshape(-1, *native_images.shape[-3:])[0].permute(1, 2, 0).numpy()
                dali_image = dali_images.reshape(-1, *dali_images.shape[-3:])[0].permute(1, 2, 0).numpy()
                difference_image = np.clip(a=np.abs(native_image.astype(np.int16) - dali_image.astype(np.int16)) * 10,
                                           a_min=0, a_max=255).astype(np.uint8)
                cv2.imwrite(filename=str(example_image_path), img=np.hstack(tup=(native_image, dali_image,
                                                                                 difference_image)))

            totals["compared_samples"] += 1
            if totals["compared_samples"] >= number_samples:
                break
        if totals["compared_samples"] >= number_samples:
            break

    return {**totals, "mean_pixel_difference": difference_sum / max(number_pixels, 1),
            "maximum_pixel_difference": maximum_difference,
            "large_difference_percentage": 100.0 * number_large_differences / max(number_pixels, 1)}


def summarise_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Groups the measurements of the repeats of every measured configuration and keeps the median of every metric, the
    values written in summary.md.

    Args:
        rows (List[Dict[str, Any]]): The measurements, one per repeat.

    Returns:
        List[Dict[str, Any]]: One row per configuration, with the medians of its metrics and its number of repeats.
    """
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        group_key = tuple(row.get(column, "") for column in IDENTIFICATION_COLUMNS if column != "repeat")
        groups.setdefault(group_key, []).append(row)

    summary_rows = []
    for group_key, group_rows in groups.items():
        summary_row = dict(zip([column for column in IDENTIFICATION_COLUMNS if column != "repeat"], group_key))
        metric_keys = dict.fromkeys(key for row in group_rows for key in row if key not in IDENTIFICATION_COLUMNS)
        for metric_key in metric_keys:
            values = [row[metric_key] for row in group_rows if isinstance(row.get(metric_key), (int, float))
                      and not math.isnan(row[metric_key])]
            summary_row[metric_key] = statistics.median(values) if values else math.nan
        summary_row["repeats"] = len(group_rows)
        summary_rows.append(summary_row)
    return summary_rows


def format_number(value: Any, decimals: int = 0) -> str:
    """
    Formats a metric for the Markdown tables of summary.md, a missing value giving a dash.

    Args:
        value (Any): The metric.
        decimals (int): The number of decimals. Defaults to 0.

    Returns:
        str: The formatted value.
    """
    if not isinstance(value, (int, float)) or math.isnan(value):
        return "–"
    return f"{value:,.{decimals}f}"


def get_variant_rank(row: Dict[str, Any]) -> int:
    """
    Gives the rank of the loader family of a summary row, so that the tables of summary.md list the current Native
    loader first, then the tuned Native loaders, then DALI.

    Args:
        row (Dict[str, Any]): A summary row.

    Returns:
        int: The rank of its loader family.
    """
    return VARIANT_ORDER.index(row["variant"]) if row["variant"] in VARIANT_ORDER else len(VARIANT_ORDER)


def select_rows(summary_rows: List[Dict[str, Any]], case: str, stage: str,
                cache: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Selects the summary rows of a case and a stage (and a page cache state), in the order of the loader families,
    for the tables of summary.md.

    Args:
        summary_rows (List[Dict[str, Any]]): The summary rows.
        case (str): The case.
        stage (str): The stage letter.
        cache (Optional[str]): "warm", "cold" or None for any.

    Returns:
        List[Dict[str, Any]]: The selected rows.
    """
    selected_rows = [row for row in summary_rows if row["case"] == case and row["stage"] == stage
                     and (cache is None or row["cache"] == cache)]
    return sorted(selected_rows, key=get_variant_rank)


def get_reference_rate(rows: List[Dict[str, Any]]) -> float:
    """
    Gives the samples per second of the current Native loader among rows of one table, the reference of its
    speedups.

    Args:
        rows (List[Dict[str, Any]]): The rows of the table.

    Returns:
        float: The samples per second of the current Native loader, or NaN.
    """
    reference_rows = [row for row in rows if row["variant"] == "native_current"]
    return reference_rows[0]["samples_per_second"] if reference_rows else math.nan


def build_case_summary(summary_rows: List[Dict[str, Any]], case: str) -> List[str]:
    """
    Writes the Markdown section of one case in summary.md: the loader alone (warm and cold page cache), the GPU
    ceiling, the training iteration, the evaluation pass and the parity, the speedups being relative to the current
    Native loader.

    Args:
        summary_rows (List[Dict[str, Any]]): The summary rows.
        case (str): The case.

    Returns:
        List[str]: The lines of the section.
    """
    lines = [f"## {case}", ""]

    # Loader alone, warm and cold
    for cache in ["warm", "cold"]:
        rows = select_rows(summary_rows=summary_rows, case=case, stage="A", cache=cache)
        if not rows:
            continue
        reference_rate = get_reference_rate(rows=rows)
        lines += [f"**Loader alone ({cache} page cache)**", "",
                  "| Variant | Setting | Samples/s | Frames/s | Build (s) | First batch (s) | Busy cores | × current |",
                  "|---|---|---|---|---|---|---|---|"]
        for row in rows:
            lines.append(f"| {row['variant']} | {row['setting']} | {format_number(row['samples_per_second'])} | "
                         f"{format_number(row.get('frames_per_second'))} | {format_number(row['build_seconds'], 2)} | "
                         f"{format_number(row['first_batch_seconds'], 2)} | "
                         f"{format_number(row['cpu_busy_cores'], 1)} | "
                         f"{format_number(row['samples_per_second'] / reference_rate, 2)} |")
        lines.append("")

    # GPU ceiling
    ceiling_rows = select_rows(summary_rows=summary_rows, case=case, stage="B")
    if ceiling_rows:
        lines += ["**GPU ceiling (one batch kept on the GPU)**", "",
                  "| Precision | Augmentation (ms/batch) | Step alone (it/s) | Augmentation + step (it/s) | "
                  "Samples/s |", "|---|---|---|---|---|"]
        for row in ceiling_rows:
            lines.append(f"| {row['precision']} | {format_number(row['augmentation_milliseconds'], 1)} | "
                         f"{format_number(row['step_iterations_per_second'], 1)} | "
                         f"{format_number(row['iterations_per_second'], 1)} | "
                         f"{format_number(row['samples_per_second'])} |")
        lines.append("")
    float32_ceilings = [row["samples_per_second"] for row in ceiling_rows if row["precision"] == "float32"]
    ceiling_rate = float32_ceilings[0] if float32_ceilings else math.nan

    # Training iteration fed by the loaders
    rows = select_rows(summary_rows=summary_rows, case=case, stage="C")
    if rows:
        reference_rate = get_reference_rate(rows=rows)
        lines += ["**Training iteration (loader, augmentation and step)**", "",
                  "| Variant | Setting | It/s | Samples/s | Frames/s | GPU % | Busy cores | × current | % of ceiling |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for row in rows:
            lines.append(f"| {row['variant']} | {row['setting']} | {format_number(row['iterations_per_second'], 1)} | "
                         f"{format_number(row['samples_per_second'])} | "
                         f"{format_number(row.get('frames_per_second'))} | "
                         f"{format_number(row['gpu_utilisation'])} | {format_number(row['cpu_busy_cores'], 1)} | "
                         f"{format_number(row['samples_per_second'] / reference_rate, 2)} | "
                         f"{format_number(100.0 * row['samples_per_second'] / ceiling_rate)} |")
        lines.append("")

    # Evaluation pass
    rows = select_rows(summary_rows=summary_rows, case=case, stage="D")
    if rows:
        reference_rate = get_reference_rate(rows=rows)
        reference_seconds = [row["full_pass_seconds"] for row in rows if row["variant"] == "native_current"]
        reference_full_pass_seconds = reference_seconds[0] if reference_seconds else math.nan
        lines += ["**Evaluation pass (test split, start of the loader included)**", "",
                  "| Variant | Setting | First batch (s) | Samples/s | After the first batch | Frames/s | GPU % | "
                  "Whole test split (s) | × current |", "|---|---|---|---|---|---|---|---|---|"]
        for row in rows:
            lines.append(f"| {row['variant']} | {row['setting']} | {format_number(row['first_batch_seconds'], 2)} | "
                         f"{format_number(row['samples_per_second'])} | "
                         f"{format_number(row['steady_samples_per_second'])} | "
                         f"{format_number(row.get('frames_per_second'))} | {format_number(row['gpu_utilisation'])} | "
                         f"{format_number(row['full_pass_seconds'], 1)} | "
                         f"{format_number(reference_full_pass_seconds / row['full_pass_seconds'], 2)} |")
        lines.append("")

    # Parity of the DALI outputs
    rows = select_rows(summary_rows=summary_rows, case=case, stage="E")
    if rows:
        lines += ["**Parity of the DALI outputs with the Native ones**", "",
                  "| Labels | Samples | Label mismatches | Mask mismatches | Mean difference | Maximum difference | "
                  f"Pixels > {PARITY_PIXEL_THRESHOLD} levels (%) |", "|---|---|---|---|---|---|---|"]
        for row in rows:
            lines.append(f"| {row['label_type'] or 'default'} | {format_number(row['compared_samples'])} | "
                         f"{format_number(row['label_mismatches'])} | {format_number(row['mask_mismatches'])} | "
                         f"{format_number(row['mean_pixel_difference'], 3)} | "
                         f"{format_number(row['maximum_pixel_difference'])} | "
                         f"{format_number(row['large_difference_percentage'], 3)} |")
        lines.append("")
    return lines


def build_summary_markdown(summary_rows: List[Dict[str, Any]], environment_lines: List[str]) -> str:
    """
    Writes summary.md: the environment, an overview of every case (the training throughput of the current Native
    loader, the speedups of the best tuned Native and DALI loaders, the GPU ceiling), the disk read rate, then the
    section of every case.

    Args:
        summary_rows (List[Dict[str, Any]]): The summary rows.
        environment_lines (List[str]): The lines describing the machine and the settings.

    Returns:
        str: The Markdown text.
    """
    cases = list(dict.fromkeys(row["case"] for row in summary_rows if row["stage"] != "F"))
    lines = ["# Loader Benchmark", "", *environment_lines, "", "## Overview", "",
             "| Case | Current (samples/s) | Best tuned Native (×) | Best DALI (×) | GPU ceiling float32 (samples/s) | "
             "Current loader alone / ceiling |", "|---|---|---|---|---|---|"]

    # One overview line per case, from its training iterations, ceiling and loader alone
    for case in cases:
        training_rows = {row["variant"]: row for row in select_rows(summary_rows=summary_rows, case=case, stage="C")}
        ceiling_rows = [row for row in select_rows(summary_rows=summary_rows, case=case, stage="B")
                        if row["precision"] == "float32"]
        loader_rows = [row for row in select_rows(summary_rows=summary_rows, case=case, stage="A", cache="warm")
                       if row["variant"] == "native_current"]
        current_rate = training_rows["native_current"]["samples_per_second"] if "native_current" in training_rows \
            else math.nan
        ceiling_rate = ceiling_rows[0]["samples_per_second"] if ceiling_rows else math.nan
        speedups = [format_number(training_rows[variant]["samples_per_second"] / current_rate, 2)
                    if variant in training_rows else "–" for variant in ["native_tuned", "dali"]]
        loader_ratio = loader_rows[0]["samples_per_second"] / ceiling_rate if loader_rows else math.nan
        lines.append(f"| {case} | {format_number(current_rate)} | {speedups[0]} | {speedups[1]} | "
                     f"{format_number(ceiling_rate)} | {format_number(loader_ratio, 2)} |")
    lines.append("")

    # Disk read rate
    for row in [row for row in summary_rows if row["stage"] == "F"]:
        lines += [f"**Disk (cold reads, {row['setting']})**: {format_number(row['megabytes_per_second'])} MB/s, "
                  f"{format_number(row['files_per_second'])} files/s", ""]

    for case in cases:
        lines += build_case_summary(summary_rows=summary_rows, case=case)
    return "\n".join(lines) + "\n"


def write_benchmark_reports(rows: List[Dict[str, Any]], output_folder: Path,
                            environment_lines: List[str]) -> Tuple[Path, Path]:
    """
    Writes the reports of the benchmark: results.csv with every measurement (one row per case, stage, variant,
    setting, page cache state and repeat) and summary.md with the medians of the repeats.

    Args:
        rows (List[Dict[str, Any]]): The measurements.
        output_folder (Path): The folder of the reports.
        environment_lines (List[str]): The lines describing the machine and the settings.

    Returns:
        Tuple[Path, Path]: The paths of results.csv and summary.md.
    """
    # Every measurement, the identification columns first
    results_path = output_folder / "results.csv"
    metric_columns = list(dict.fromkeys(key for row in rows for key in row if key not in IDENTIFICATION_COLUMNS))
    with open(file=results_path, mode="w", newline="") as results_file:
        writer = csv.DictWriter(f=results_file, fieldnames=IDENTIFICATION_COLUMNS + metric_columns, restval="")
        writer.writeheader()
        writer.writerows(rowdicts=rows)

    # Medians of the repeats
    summary_path = output_folder / "summary.md"
    summary_path.write_text(data=build_summary_markdown(summary_rows=summarise_rows(rows=rows),
                                                        environment_lines=environment_lines))
    print_color(color="green", output=f"Results : file://{results_path}")
    print_color(color="green", output=f"Summary : file://{summary_path}")
    return results_path, summary_path
