"""
Helpers of the 'benchmark_speed' diagnosis mode (ExperimentConfiguration.diagnosis_modes), run by
BaseTrainer.benchmark_speed of trainers/base_trainer.py before it exits without training: they time the training step of
a task and the inference of its model in PyTorch and in every format served by Triton (TorchScript, ONNX Runtime CUDA,
ONNX Runtime with TensorRT in FP32 and FP16), on one batch of the training loader kept in memory and tiled to any batch
size, so that neither the loader nor the processor limits the measured speed. The module holds the memory cap keeping
a margin of the machine free, the tiling of the batch, the timing, the search of the largest batch size that fits, the
export and opening of every format, and the console output: the panel announcing the numbered steps, the progress
lines of every step, and the report (an overview panel, then a training panel and an inference panel, each naming the
question it answers and what its timings include).
"""

import gc
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
import numpy as np
import onnxruntime
import torch

from contextlib import contextmanager
from functools import partial
from pathlib import Path
from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from torch import nn
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypedDict

from vision_training.deployment.export_formats import export_onnx, export_torchscript, hide_known_export_warnings
from vision_training.deployment.export_specifications import create_random_inputs
from vision_training.evaluation.report_formatting import (HEADER_SEPARATOR, HEADER_STYLE, LEGEND_STYLE,
                                                          MISSING_VALUE_TEXT, NOTE_STYLE, REPORT_WIDTH,
                                                          TABLE_HEADER_STYLE, format_percentage,
                                                          format_percentage_label, print_report)
from vision_training.trainers.metric_tracking import MetricTracker

# Untimed run before every measurement, of at least this time and this number of iterations, so that the allocator,
# the choice of the cuDNN algorithms and the TensorRT engine builds are done before the timed iterations
WARM_UP_SECONDS = 2.0
MINIMUM_WARM_UP_ITERATIONS = 3

# Timed run of a measurement, split into windows of the same number of iterations: the median of the window means is
# the reported time, their spread tells how steady the GPU was
MEASURED_SECONDS = 5.0
NUMBER_TIMING_WINDOWS = 5

# Largest batch size the search tries, and the end of its bisection: the gap between the largest batch size that fits
# and the smallest one that does not is at most 1/16 of the first
MAXIMUM_BENCHMARK_BATCH_SIZE = 2048
SEARCH_RESOLUTION_DIVISOR = 16

# Training steps (or inference passes) run at every batch size the search tries: the first one allocates the gradients
# and the optimizer state, the second one runs with them in place
PROBE_ITERATIONS = 2

# Memory left free for the other processes: the integrated GPU of the DGX Spark shares one memory with the processor,
# where every training slows down once little memory is free, while a discrete GPU only holds its own memory
INTEGRATED_GPU_MEMORY_MARGIN_BYTES = 40 * 10 ** 9
DISCRETE_GPU_MEMORY_MARGIN_BYTES = 2 * 10 ** 9
BYTES_PER_GIGABYTE = 10 ** 9

# ONNX Runtime execution providers of the formats served by the ONNX Runtime backend of Triton
CUDA_PROVIDER = "CUDAExecutionProvider"
TENSORRT_PROVIDER = "TensorrtExecutionProvider"

# Groups of the inference formats in the report: the PyTorch passes run by the code of the repository, and the exported
# files the deployment serves with Triton
REPOSITORY_GROUP = "Inside this repository (Python, never served)"
TRITON_GROUP = "Served by Triton"

# Formats of the inference measured by the benchmark, in the order of the report: their identifier, the name the report
# and the console give them, their group, the precision of their computation and the ONNX Runtime provider running them
# (None for the PyTorch formats). The test evaluation is the predict_batch of the trainer, the network before export the
# inference model every export traces, the reference of the speed-up column of the report
REFERENCE_FORMAT_NAME = "PyTorch Inference Model"
INFERENCE_FORMATS = [{"name": "PyTorch Evaluation", "label": "Test evaluation (predict_batch)",
                      "group": REPOSITORY_GROUP, "precision": "FP32", "provider": None},
                     {"name": REFERENCE_FORMAT_NAME, "label": "Network before export (reference)",
                      "group": REPOSITORY_GROUP, "precision": "FP32", "provider": None},
                     {"name": "TorchScript", "label": "TorchScript file, PyTorch backend", "group": TRITON_GROUP,
                      "precision": "FP32", "provider": None},
                     {"name": "ONNX Runtime CUDA", "label": "ONNX file, ONNX Runtime CUDA", "group": TRITON_GROUP,
                      "precision": "FP32", "provider": CUDA_PROVIDER},
                     {"name": "ONNX Runtime TensorRT", "label": "ONNX file, TensorRT FP32", "group": TRITON_GROUP,
                      "precision": "FP32", "provider": TENSORRT_PROVIDER},
                     {"name": "ONNX Runtime TensorRT", "label": "ONNX file, TensorRT FP16 (half precision)",
                      "group": TRITON_GROUP, "precision": "FP16", "provider": TENSORRT_PROVIDER}]

# Status of a row of the report, written in its cells and in the progress lines of the console
MEASURED_STATUS = "done"
OUT_OF_MEMORY_STATUS = "out of memory"
NOT_AVAILABLE_STATUS = "not available"
FORMAT_FAILED_STATUS = "failed"

# Spread of the timing windows (slowest minus fastest, relative to their median) above which a timing is marked
# unsteady in the report, e.g. while other work shares the GPU
UNSTEADY_SPREAD = 0.10
UNSTEADY_MARK = " *"

# Smallest width of a value column of the inference table, that of its longest status ('not available')
STATUS_COLUMN_WIDTH = len(NOT_AVAILABLE_STATUS)

# Styles of the console output of the benchmark: its panels, the step numbers of the progress lines, the group rows of
# the inference table and the batch sizes that fit, and the indentation of the progress lines below their step
PANEL_BORDER_STYLE = "cyan"
PROGRESS_PANEL_BORDER_STYLE = "blue"
STEP_STYLE = "bold blue"
GROUP_STYLE = "italic"
SUCCESS_STYLE = "green"
PROGRESS_INDENT = "      "

# Daemon of the CUDA Multi-Process Service, listed by nvidia-smi among the GPU processes without running any work
MULTI_PROCESS_SERVICE_NAME = "nvidia-cuda-mps-server"

# Messages of the out of memory errors raised by PyTorch and by the arena of ONNX Runtime, in lower case
OUT_OF_MEMORY_MESSAGES = ["out of memory", "failed to allocate memory"]


class MemoryBudget(TypedDict):
    """
    Memory the speed benchmark may take, set by limit_process_memory when the benchmark starts and printed in its
    report: the free memory of the device, the margin left free for the other processes, the budget the benchmark may
    add to what the process already holds, and the total the allocator of PyTorch is capped at.
    """
    free_bytes: int
    margin_bytes: int
    budget_bytes: int
    allowed_bytes: int
    integrated: bool


class TimingMeasurement(TypedDict):
    """
    Time of one iteration of a training step or of an inference pass at one batch size, measured by
    measure_iteration_time: the median of the window means and the fastest and slowest windows in seconds, the number
    of timed iterations, and the peak memory reserved by PyTorch during the warm-up and the timed iterations (None for
    the ONNX Runtime formats, whose memory is outside PyTorch).
    """
    seconds_per_iteration: float
    fastest_seconds: float
    slowest_seconds: float
    measured_iterations: int
    peak_memory_bytes: Optional[int]


class BenchmarkRow(TypedDict):
    """
    Row of a table of the report: the configured or largest training batch ('configured', 'largest that fits'), or one
    inference format at one batch size (the label of the format), with its batch size (None for a format that could not
    be opened), its measurement (None when it could not be measured), its status, the real frames of its batch for the
    frame throughput of the sequence tasks (None for the image tasks), and the error of a failed format, written in the
    notes of the report (None otherwise).
    """
    label: str
    precision: str
    batch_size: Optional[int]
    measurement: Optional[TimingMeasurement]
    status: str
    frames_per_batch: Optional[int]
    failure_reason: Optional[str]


def get_device_index(device: torch.device) -> int:
    """
    Gives the index of the CUDA device of the benchmark, the current device for a device given without index
    ('cuda'), as the memory cap of PyTorch and the providers of ONNX Runtime need it.

    Args:
        device (torch.device): The CUDA device of the benchmark.

    Returns:
        int: The index of the device.
    """
    return torch.cuda.current_device() if device.index is None else device.index


def get_memory_margin(device: torch.device) -> Tuple[int, bool]:
    """
    Gives the memory the speed benchmark leaves free for the other processes, larger on an integrated GPU, whose memory
    is that of the whole machine, than on a discrete GPU.

    Args:
        device (torch.device): The CUDA device of the benchmark.

    Returns:
        Tuple[int, bool]: The margin in bytes, and whether the GPU is integrated.
    """
    integrated = bool(torch.cuda.get_device_properties(device=device).is_integrated)
    return (INTEGRATED_GPU_MEMORY_MARGIN_BYTES if integrated else DISCRETE_GPU_MEMORY_MARGIN_BYTES), integrated


@contextmanager
def limit_process_memory(device: torch.device) -> Iterator[MemoryBudget]:
    """
    Caps the memory the allocator of PyTorch may reserve during the speed benchmark at what the process already holds
    plus the free memory of the device minus the margin of get_memory_margin, so that the search of the largest batch
    size ends on a clean out of memory error of PyTorch instead of taking the memory the other trainings of the machine
    need (on the unified memory of the DGX Spark, the kernel would rather kill a process). The cap is lifted when the
    benchmark ends, even when it fails.

    Args:
        device (torch.device): The CUDA device of the benchmark.

    Yields:
        MemoryBudget: The free memory, the margin, the budget and the capped total, in bytes.

    Raises:
        RuntimeError: If the free memory is not above the margin.
    """
    free_bytes, total_bytes = torch.cuda.mem_get_info(device=device)
    margin_bytes, integrated = get_memory_margin(device=device)
    budget_bytes = free_bytes - margin_bytes
    if budget_bytes <= 0:
        memory_owner = ("the machine (an integrated GPU shares its memory with the processor)" if integrated
                        else "the GPU")
        raise RuntimeError(f"Only {free_bytes / BYTES_PER_GIGABYTE:.1f} GB of memory are free, and the speed benchmark "
                           f"leaves {margin_bytes / BYTES_PER_GIGABYTE:.0f} GB of {memory_owner} free for the other "
                           f"processes: stop or wait for the other trainings before benchmarking.")

    # The allocator may reserve what it already holds plus the budget, the fraction being of the whole device memory
    device_index = get_device_index(device=device)
    allowed_bytes = torch.cuda.memory_reserved(device=device_index) + budget_bytes
    torch.cuda.set_per_process_memory_fraction(fraction=min(1.0, allowed_bytes / total_bytes), device=device_index)
    try:
        yield {"free_bytes": free_bytes, "margin_bytes": margin_bytes, "budget_bytes": budget_bytes,
               "allowed_bytes": allowed_bytes, "integrated": integrated}
    finally:
        torch.cuda.set_per_process_memory_fraction(fraction=1.0, device=device_index)


def list_other_gpu_processes() -> Optional[List[str]]:
    """
    Lists the other processes computing on the GPUs when the speed benchmark starts (from nvidia-smi), whose work
    shares the GPU with the timed iterations and slows them down, so that the report warns that its times are those of
    a shared GPU. The benchmark process itself and the daemon of the CUDA Multi-Process Service are left out.

    Returns:
        Optional[List[str]]: Every other process as 'identifier (program, memory MiB)', empty when the GPU runs nothing
            else, or None when nvidia-smi cannot list them.
    """
    try:
        query_result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                       "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
                                      timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None

    # One line per process: its identifier, its program and its memory
    other_processes = []
    for line in query_result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not fields[0].isdigit():
            continue
        if int(fields[0]) == os.getpid() or MULTI_PROCESS_SERVICE_NAME in fields[1]:
            continue
        other_processes.append(f"{fields[0]} ({Path(fields[1]).name}, {fields[2]} MiB)")
    return other_processes


def release_cached_memory() -> None:
    """
    Gives the tensors freed by a measurement or by a failed batch size back to the device, the references held by a
    caught error being collected first, so that the next measurement starts from the memory of the model alone and its
    peak memory is its own.
    """
    gc.collect()
    torch.cuda.empty_cache()


@contextmanager
def capture_native_error_output() -> Iterator[Dict[str, bool]]:
    """
    Captures what the native libraries write to the error stream (file descriptor 2) during a probe or a timing of the
    speed benchmark: an out of memory error raised inside a torchvision operator makes libtorch write its whole C++
    stack trace there, outside Python, in the middle of the progress line of the search. The caller sets
    'out_of_memory' when the error was the expected lack of memory, whose output is then dropped, the batch size being
    reported as too large; any other output is written back to the error stream when the block ends.

    Yields:
        Dict[str, bool]: The outcome of the block, whose 'out_of_memory' the caller sets to True to drop the output.
    """
    sys.stderr.flush()
    saved_descriptor = os.dup(2)
    outcome = {"out_of_memory": False}
    with tempfile.TemporaryFile() as captured_file:
        os.dup2(captured_file.fileno(), 2)
        try:
            yield outcome
        finally:
            # Error stream given back, the captured output written to it unless it belongs to a lack of memory
            sys.stderr.flush()
            os.dup2(saved_descriptor, 2)
            os.close(saved_descriptor)
            if not outcome["out_of_memory"]:
                captured_file.seek(0)
                os.write(2, captured_file.read())


def is_out_of_memory_error(error: Exception) -> bool:
    """
    Tells whether an error raised by a training step or an inference pass is a lack of memory, which ends the batch
    sizes the search tries and marks a row of the report, any other error being raised again.

    Args:
        error (Exception): The raised error.

    Returns:
        bool: True for an out of memory error of PyTorch or of ONNX Runtime.
    """
    if isinstance(error, torch.OutOfMemoryError):
        return True
    return any(message in str(error).lower() for message in OUT_OF_MEMORY_MESSAGES)


def resize_batch(batch: Dict[str, Any], batch_size: int) -> Dict[str, Any]:
    """
    Repeats the samples of a loader batch, in their order and cyclically, into a batch of any size, without any code
    of the task: every tensor and every list whose first dimension is the number of samples of the batch (the images,
    the labels, the boxes of every image, the sequence masks) is indexed by sample, the other values (settings, tensors
    of another first dimension) being kept as they are. The tensors stay on their device and the tensors of the lists
    are copied, so that no two samples of the batch share a tensor.

    Args:
        batch (Dict[str, Any]): The batch yielded by the training loader, with its 'images'.
        batch_size (int): The number of samples of the returned batch.

    Returns:
        Dict[str, Any]: The batch of batch_size samples.

    Raises:
        ValueError: If the batch size is not positive.
    """
    if batch_size < 1:
        raise ValueError(f"A batch holds at least one sample, got a batch size of {batch_size}.")
    source_batch_size = batch["images"].shape[0]
    sample_indices = torch.arange(end=batch_size) % source_batch_size

    # Every per-sample value indexed by the repeated sample indices
    resized_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == source_batch_size:
            resized_batch[key] = value.index_select(dim=0, index=sample_indices.to(device=value.device))
        elif isinstance(value, list) and len(value) == source_batch_size:
            resized_batch[key] = [value[sample_index].clone() if isinstance(value[sample_index], torch.Tensor)
                                  else value[sample_index] for sample_index in sample_indices.tolist()]
        else:
            resized_batch[key] = value
    return resized_batch


def move_batch(batch: Dict[str, Any], device: torch.device, pin_memory: bool) -> Dict[str, Any]:
    """
    Moves the tensors of a loader batch, and the tensors of its lists, to where the benchmark keeps it: on the GPU, or
    in pinned host memory with pin_memory, as the loaders hand their batches to a discrete GPU.

    Args:
        batch (Dict[str, Any]): The batch yielded by the training loader.
        device (torch.device): The device receiving the tensors, the CPU for a batch in host memory.
        pin_memory (bool): Whether the tensors of a batch in host memory are pinned.

    Returns:
        Dict[str, Any]: The batch with its tensors moved.
    """
    moved_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = move_tensor(tensor=value, device=device, pin_memory=pin_memory)
        elif isinstance(value, list):
            moved_batch[key] = [move_tensor(tensor=element, device=device, pin_memory=pin_memory)
                                if isinstance(element, torch.Tensor) else element for element in value]
        else:
            moved_batch[key] = value
    return moved_batch


def move_tensor(tensor: torch.Tensor, device: torch.device, pin_memory: bool) -> torch.Tensor:
    """
    Moves one tensor of a batch for move_batch, pinning it when it stays in host memory and pin_memory is set.

    Args:
        tensor (torch.Tensor): The tensor.
        device (torch.device): The device receiving it.
        pin_memory (bool): Whether a tensor in host memory is pinned.

    Returns:
        torch.Tensor: The moved tensor.
    """
    moved_tensor = tensor.to(device=device)
    if moved_tensor.device.type == "cpu" and pin_memory:
        return moved_tensor.pin_memory()
    return moved_tensor


def measure_iteration_time(run_iteration: Callable[[], Any], device: torch.device) -> TimingMeasurement:
    """
    Times one training step or inference pass: the cache of the allocator is emptied and its peak reset, the iteration
    is run for at least WARM_UP_SECONDS and MINIMUM_WARM_UP_ITERATIONS untimed, then NUMBER_TIMING_WINDOWS windows of
    the same number of iterations, sized from the last warm-up iteration to last MEASURED_SECONDS in all, are timed
    between two synchronisations of the device, so that the time covers the GPU work queued by the iterations and not
    only their launch.

    Args:
        run_iteration (Callable[[], Any]): Runs one iteration on the batch of the measured batch size.
        device (torch.device): The CUDA device of the benchmark.

    Returns:
        TimingMeasurement: The median, fastest and slowest seconds per iteration, the timed iterations and the peak
            memory reserved by PyTorch.
    """
    release_cached_memory()
    torch.cuda.reset_peak_memory_stats(device=device)

    # Untimed warm-up, each iteration synchronised so that the time of the last one sizes the windows
    warm_up_start = time.perf_counter()
    warm_up_iterations, last_iteration_seconds = 0, 0.0
    while warm_up_iterations < MINIMUM_WARM_UP_ITERATIONS or time.perf_counter() - warm_up_start < WARM_UP_SECONDS:
        iteration_start = time.perf_counter()
        run_iteration()
        torch.cuda.synchronize(device=device)
        last_iteration_seconds = time.perf_counter() - iteration_start
        warm_up_iterations += 1

    # Timed windows of the same number of iterations, the device synchronised only at their ends
    window_iterations = max(1, math.ceil(MEASURED_SECONDS / NUMBER_TIMING_WINDOWS / max(last_iteration_seconds, 1e-6)))
    window_seconds = []
    for _ in range(NUMBER_TIMING_WINDOWS):
        window_start = time.perf_counter()
        for _ in range(window_iterations):
            run_iteration()
        torch.cuda.synchronize(device=device)
        window_seconds.append((time.perf_counter() - window_start) / window_iterations)
    return {"seconds_per_iteration": statistics.median(window_seconds), "fastest_seconds": min(window_seconds),
            "slowest_seconds": max(window_seconds), "measured_iterations": window_iterations * NUMBER_TIMING_WINDOWS,
            "peak_memory_bytes": torch.cuda.max_memory_reserved(device=device)}


def find_largest_batch_size(fits: Callable[[int], bool], starting_batch_size: int) -> Tuple[int, Optional[int]]:
    """
    Searches the largest batch size whose training step (or inference pass) fits in the memory budget, up to
    MAXIMUM_BENCHMARK_BATCH_SIZE: from the configured batch size, the size doubles while it fits (or halves while it
    does not), then the gap between the largest size that fits and the smallest one that does not is halved until it
    is at most 1/SEARCH_RESOLUTION_DIVISOR of the first.

    Args:
        fits (Callable[[int], bool]): Runs the probe iterations at its batch_size argument, True when they fit.
        starting_batch_size (int): The batch size tried first, the configured one.

    Returns:
        Tuple[int, Optional[int]]: The largest batch size that fits, 0 when even a single sample does not, and the
            smallest batch size found not to fit, None when the maximum batch size fits.
    """
    largest_fitting, smallest_failing = 0, None
    batch_size = min(starting_batch_size, MAXIMUM_BENCHMARK_BATCH_SIZE)
    if fits(batch_size=batch_size):
        largest_fitting = batch_size
    else:
        smallest_failing = batch_size

    # Doubling while the batch fits, up to the maximum batch size
    while smallest_failing is None and largest_fitting < MAXIMUM_BENCHMARK_BATCH_SIZE:
        batch_size = min(2 * largest_fitting, MAXIMUM_BENCHMARK_BATCH_SIZE)
        if fits(batch_size=batch_size):
            largest_fitting = batch_size
        else:
            smallest_failing = batch_size

    # Halving while the batch does not fit, down to a single sample
    while largest_fitting == 0 and smallest_failing > 1:
        batch_size = smallest_failing // 2
        if fits(batch_size=batch_size):
            largest_fitting = batch_size
        else:
            smallest_failing = batch_size

    # Bisection between the largest batch size that fits and the smallest one that does not
    while largest_fitting > 0 and smallest_failing is not None and \
            smallest_failing - largest_fitting > max(1, largest_fitting // SEARCH_RESOLUTION_DIVISOR):
        batch_size = (largest_fitting + smallest_failing) // 2
        if fits(batch_size=batch_size):
            largest_fitting = batch_size
        else:
            smallest_failing = batch_size
    return largest_fitting, smallest_failing


def export_benchmark_formats(inference_model: nn.Module, specification: Dict[str, Any], maximum_batch_size: int,
                             export_folder: Path, device: torch.device) -> Dict[str, Dict[str, Any]]:
    """
    Writes the inference model to the files the benchmark times, with the export functions of the deployment
    (deployment/export_formats.py), into a temporary folder: the TorchScript file, and the ONNX file only when this
    environment of ONNX Runtime has a GPU provider (onnxruntime-gpu), the CPU provider never running on the GPU. A
    failed export is recorded with its error instead of stopping the benchmark, its formats then being reported as
    failed.

    Args:
        inference_model (nn.Module): The inference model of the task, in evaluation mode.
        specification (Dict[str, Any]): The export specification of the task.
        maximum_batch_size (int): The largest batch size of the ONNX file, the largest inference batch measured.
        export_folder (Path): The temporary folder of the files.
        device (torch.device): The CUDA device of the benchmark.

    Returns:
        Dict[str, Dict[str, Any]]: For 'torchscript' and 'onnx', the 'path' of the written file (None when it is
            missing) and the 'error' explaining why it is missing (None when it was written).
    """
    hide_known_export_warnings()
    example_inputs = create_random_inputs(specification=specification, shape_name="example_shape",
                                          maximum_batch_size=maximum_batch_size, device=device, generator=None)
    exported_files = {}

    # TorchScript file of the Triton PyTorch backend
    torchscript_path = export_folder / "model.pt"
    try:
        export_torchscript(inference_model=inference_model, example_inputs=example_inputs, output_path=torchscript_path)
        exported_files["torchscript"] = {"path": torchscript_path, "error": None}
    except Exception as error:
        exported_files["torchscript"] = {"path": None,
                                         "error": f"TorchScript export failed: {describe_error(error=error)}"}

    # ONNX file of the Triton ONNX Runtime backend, only worth writing when a GPU provider can run it here
    available_providers = onnxruntime.get_available_providers()
    if CUDA_PROVIDER not in available_providers and TENSORRT_PROVIDER not in available_providers:
        missing_provider_error = (f"onnxruntime {onnxruntime.__version__} of this environment has no GPU provider "
                                  f"(only {', '.join(available_providers)}): the GPU build is the onnxruntime-gpu "
                                  f"package")
        exported_files["onnx"] = {"path": None, "error": missing_provider_error}
        return exported_files
    onnx_path = export_folder / "model.onnx"
    try:
        export_onnx(inference_model=inference_model, example_inputs=example_inputs, output_path=onnx_path,
                    specification=specification, maximum_batch_size=maximum_batch_size)
        exported_files["onnx"] = {"path": onnx_path, "error": None}
    except Exception as error:
        exported_files["onnx"] = {"path": None, "error": f"ONNX export failed: {describe_error(error=error)}"}
    return exported_files


def describe_error(error: Exception) -> str:
    """
    Gives the first line of an error raised by an export or a format, written in the legend of the report next to the
    format it stopped.

    Args:
        error (Exception): The raised error.

    Returns:
        str: The type of the error and the first line of its message.
    """
    message_lines = [line for line in str(error).splitlines() if line.strip()]
    return f"{type(error).__name__}: {message_lines[0] if message_lines else ''}".strip()


def open_export_format(inference_format: Dict[str, Any], inference_model: nn.Module,
                       exported_files: Dict[str, Dict[str, Any]], device: torch.device, memory_limit_bytes: int,
                       export_folder: Path) -> Tuple[Optional[Callable[..., Any]], Optional[str]]:
    """
    Opens one of the exported formats of INFERENCE_FORMATS for its measurement, right before it, so that the memory of
    one format is freed before the next one opens: the eager inference model, the TorchScript file loaded on the GPU
    as the Triton PyTorch backend loads it, or an ONNX Runtime session of the provider of the format (the TensorRT
    sessions keeping the CUDA provider for the operators TensorRT leaves, as the Triton ONNX Runtime backend does).

    Args:
        inference_format (Dict[str, Any]): The format, from INFERENCE_FORMATS.
        inference_model (nn.Module): The inference model of the task, in evaluation mode.
        exported_files (Dict[str, Dict[str, Any]]): The files written by export_benchmark_formats.
        device (torch.device): The CUDA device of the benchmark.
        memory_limit_bytes (int): The memory ONNX Runtime may take, the budget left to the benchmark, since the cap
            of PyTorch does not apply to it.
        export_folder (Path): The temporary folder of the exported files, holding the TensorRT engine cache.

    Returns:
        Tuple[Optional[Callable[..., Any]], Optional[str]]: The function running one pass of the format on
            (prepared_batch, inference_inputs) and None, or None and the reason why the format cannot be measured.
    """
    # Eager inference model, the network every export traces
    if inference_format["name"] == REFERENCE_FORMAT_NAME:
        return partial(run_torch_module, module=inference_model), None

    # TorchScript file, loaded on the GPU like the Triton PyTorch backend loads it
    if inference_format["name"] == "TorchScript":
        if exported_files["torchscript"]["path"] is None:
            return None, exported_files["torchscript"]["error"]
        traced_model = torch.jit.load(str(exported_files["torchscript"]["path"]), map_location=device)
        return partial(run_torch_module, module=traced_model), None

    # ONNX Runtime session of the provider of the format, refused when the provider cannot run here
    if exported_files["onnx"]["path"] is None:
        return None, exported_files["onnx"]["error"]
    provider = inference_format["provider"]
    if provider not in onnxruntime.get_available_providers():
        return None, f"onnxruntime {onnxruntime.__version__} of this environment has no {provider}"
    if memory_limit_bytes <= 0:
        return None, "PyTorch already holds the whole memory budget, none is left for ONNX Runtime"
    device_index = get_device_index(device=device)
    session = create_onnx_runtime_session(model_path=exported_files["onnx"]["path"], provider=provider,
                                          half_precision=inference_format["precision"] == "FP16",
                                          memory_limit_bytes=memory_limit_bytes, cache_folder=export_folder,
                                          device_index=device_index)
    if session.get_providers()[0] != provider:
        return None, (f"ONNX Runtime fell back to {session.get_providers()[0]}: the libraries of {provider} could "
                      f"not be loaded")
    return partial(run_onnx_runtime_session, session=session, device_index=device_index), None


def create_onnx_runtime_session(model_path: Path, provider: str, half_precision: bool, memory_limit_bytes: int,
                                cache_folder: Path, device_index: int) -> onnxruntime.InferenceSession:
    """
    Creates the ONNX Runtime session of an exported ONNX file on the GPU, with the CUDA provider capped at the memory
    left to the benchmark, preceded for the TensorRT formats by the TensorRT provider in FP32 or FP16 with its engine
    cache in the temporary folder of the exports.

    Args:
        model_path (Path): The exported 'model.onnx'.
        provider (str): CUDA_PROVIDER or TENSORRT_PROVIDER.
        half_precision (bool): Whether TensorRT may compute in FP16.
        memory_limit_bytes (int): The memory the CUDA provider and the TensorRT workspace may take.
        cache_folder (Path): The folder of the TensorRT engine cache.
        device_index (int): The index of the CUDA device.

    Returns:
        onnxruntime.InferenceSession: The session.
    """
    cuda_options = {"device_id": device_index, "gpu_mem_limit": memory_limit_bytes,
                    "arena_extend_strategy": "kSameAsRequested"}
    providers = [(CUDA_PROVIDER, cuda_options)]
    if provider == TENSORRT_PROVIDER:
        providers.insert(0, (TENSORRT_PROVIDER, {"device_id": device_index, "trt_fp16_enable": half_precision,
                                                 "trt_max_workspace_size": memory_limit_bytes,
                                                 "trt_engine_cache_enable": True,
                                                 "trt_engine_cache_path": str(cache_folder)}))

    # Errors only, the providers otherwise warning about every node they leave to the CPU for the shapes
    session_options = onnxruntime.SessionOptions()
    session_options.log_severity_level = 3
    return onnxruntime.InferenceSession(str(model_path), sess_options=session_options, providers=providers)


def run_torch_module(module: nn.Module, prepared_batch: Dict[str, Any],
                     inference_inputs: Dict[str, torch.Tensor]) -> Any:
    """
    Runs one inference pass of the eager inference model or of the loaded TorchScript file on the inputs already on the
    GPU, without gradients, as the export checks and the Triton PyTorch backend run them.

    Args:
        module (nn.Module): The inference model or the loaded TorchScript module.
        prepared_batch (Dict[str, Any]): The prepared batch, unused by the exported formats.
        inference_inputs (Dict[str, torch.Tensor]): The input of every argument of the inference model.

    Returns:
        Any: The outputs of the model.
    """
    with torch.no_grad():
        return module(**inference_inputs)


def run_onnx_runtime_session(session: onnxruntime.InferenceSession, device_index: int, prepared_batch: Dict[str, Any],
                             inference_inputs: Dict[str, torch.Tensor]) -> None:
    """
    Runs one inference pass of an ONNX Runtime session on the inputs already on the GPU, bound to the session by their
    device pointers and the outputs left on the GPU, so that no copy through the host memory enters the time.

    Args:
        session (onnxruntime.InferenceSession): The session of the format.
        device_index (int): The index of the CUDA device.
        prepared_batch (Dict[str, Any]): The prepared batch, unused by the exported formats.
        inference_inputs (Dict[str, torch.Tensor]): The contiguous float32 input of every argument of the inference
            model.
    """
    binding = session.io_binding()
    for input_name, input_tensor in inference_inputs.items():
        binding.bind_input(name=input_name, device_type="cuda", device_id=device_index, element_type=np.float32,
                           shape=tuple(input_tensor.shape), buffer_ptr=input_tensor.data_ptr())
    for output in session.get_outputs():
        binding.bind_output(name=output.name, device_type="cuda", device_id=device_index)
    session.run_with_iobinding(iobinding=binding)


def format_gigabytes(number_bytes: int) -> str:
    """
    Writes a memory size of the report in gigabytes with one decimal.

    Args:
        number_bytes (int): The size in bytes.

    Returns:
        str: The size, e.g. '12.3'.
    """
    return f"{number_bytes / BYTES_PER_GIGABYTE:.1f}"


def describe_timing() -> str:
    """
    Writes how every cell of the report was timed (measure_iteration_time), for the plan printed when the benchmark
    starts and for the overview of its report, from the timing constants read when it is called.

    Returns:
        str: The sentence, e.g. '2 seconds of warm-up, then 5 seconds timed in 5 windows; every cell is the median
            window'.
    """
    return (f"{WARM_UP_SECONDS:g} seconds of warm-up, then {MEASURED_SECONDS:g} seconds timed in "
            f"{NUMBER_TIMING_WINDOWS} windows; every cell is the median window")


def join_with_and(texts: List[str]) -> str:
    """
    Joins the parts of a sentence of the console output of the benchmark with commas and a final 'and', such as the
    batch sizes of a step ('1, 64 and 152') or the operations of a training step.

    Args:
        texts (List[str]): The parts, at least one.

    Returns:
        str: The joined parts.
    """
    return texts[0] if len(texts) == 1 else f"{', '.join(texts[:-1])} and {texts[-1]}"


def pad_format_label(label: str) -> str:
    """
    Pads the label of an inference format to the width of the longest one, so that the results of every format start
    in the same column of the progress lines.

    Args:
        label (str): The label of the format, from INFERENCE_FORMATS.

    Returns:
        str: The padded label.
    """
    return label.ljust(max(len(inference_format["label"]) for inference_format in INFERENCE_FORMATS) + 3)


def print_export_results(exported_files: Dict[str, Dict[str, Any]]) -> None:
    """
    Prints the progress lines of the export step of the benchmark: every file written, or the reason why it was not
    (e.g. no GPU provider of ONNX Runtime, whose formats are then not available).

    Args:
        exported_files (Dict[str, Dict[str, Any]]): The files written by export_benchmark_formats.
    """
    for file_key, file_name in [("torchscript", "TorchScript file"), ("onnx", "ONNX file")]:
        if exported_files[file_key]["path"] is not None:
            print_progress_line(text=f"{file_name} written", style=SUCCESS_STYLE)
        else:
            print_progress_line(text=f"{file_name} not written: {exported_files[file_key]['error']}", style=NOTE_STYLE)


def create_progress_console() -> Console:
    """
    Creates the console of the progress lines of the benchmark, writing to the current standard output without
    highlighting and without wrapping, so that the results of a step can be appended to its line as they come.

    Returns:
        Console: The console.
    """
    return Console(highlight=False, emoji=False, soft_wrap=True)


def print_benchmark_plan(introduction_lines: List[str], step_titles: List[str], warning_lines: List[str]) -> None:
    """
    Prints the panel opening the speed benchmark, once its memory budget is set: what the benchmark does, the numbered
    steps that the progress lines then follow, how every cell is timed, and the warnings (other processes on the GPU).

    Args:
        introduction_lines (List[str]): The lines saying what is measured, on which batch and in which memory.
        step_titles (List[str]): The steps, in order, numbered as the progress lines number them.
        warning_lines (List[str]): The warnings, written in the colour of the notes.
    """
    content: List[RenderableType] = [Text(text=line) for line in introduction_lines]
    content += [Text(text=" "), Text(text="Steps", style="bold")]
    content += [Text(text=f"  {step_number}. {step_title}")
                for step_number, step_title in enumerate(step_titles, start=1)]
    content += [Text(text=f"Every timing: {describe_timing()}.", style=LEGEND_STYLE)]
    content += [Text(text=line, style=NOTE_STYLE) for line in warning_lines]
    console = Console(width=REPORT_WIDTH, highlight=False, emoji=False)
    console.print(Panel(renderable=Group(*content), title=Text(text="Speed Benchmark", style="bold"),
                        title_align="left", border_style=PROGRESS_PANEL_BORDER_STYLE))


def print_step_title(step_number: int, number_steps: int, title: str) -> None:
    """
    Prints the title line of one step of the benchmark, numbered as in the opening panel, e.g. '[3/5] Timing the
    inference at batch 1, 64 and 152'.

    Args:
        step_number (int): The number of the step, from 1.
        number_steps (int): The number of steps of the benchmark.
        title (str): The title of the step.
    """
    create_progress_console().print(Text.assemble((f"[{step_number}/{number_steps}] ", STEP_STYLE), (title, "bold")))


def print_progress_text(text: str, style: str, indent: bool) -> None:
    """
    Appends a part to the current progress line of a step (a batch size that fits, a timed batch size), without ending
    the line, so that the results of the step show as they come.

    Args:
        text (str): The part.
        style (str): Its rich style, "" for none.
        indent (bool): Whether the part opens the line, indented below the title of its step.
    """
    create_progress_console().print(Text(text=(PROGRESS_INDENT if indent else "") + text, style=style), end="")


def end_progress_line() -> None:
    """
    Ends the current progress line of a step.
    """
    create_progress_console().print()


def print_progress_line(text: str, style: str) -> None:
    """
    Prints a whole progress line below the title of its step, e.g. the error of a failed format.

    Args:
        text (str): The line.
        style (str): Its rich style, "" for none.
    """
    create_progress_console().print(Text(text=PROGRESS_INDENT + text, style=style))


def get_status_style(status: str) -> str:
    """
    Gives the style of a status in the progress lines and in the cells of the report: green when measured, the colour
    of the notes otherwise (out of memory, failed, not available).

    Args:
        status (str): The status of the row.

    Returns:
        str: The rich style.
    """
    return SUCCESS_STYLE if status == MEASURED_STATUS else NOTE_STYLE


def measure_spread(measurement: TimingMeasurement) -> float:
    """
    Gives the spread of the timing windows of a measurement: the gap between its slowest and fastest windows relative
    to their median, above UNSTEADY_SPREAD when other work shared the GPU or its clock changed during the timing.

    Args:
        measurement (TimingMeasurement): The measurement.

    Returns:
        float: The spread, 0.1 for 10 %.
    """
    return (measurement["slowest_seconds"] - measurement["fastest_seconds"]) / measurement["seconds_per_iteration"]


def build_timing_text(value_text: str, measurement: TimingMeasurement) -> Text:
    """
    Writes a timed value of the report, followed by UNSTEADY_MARK when its timing windows spread more than
    UNSTEADY_SPREAD, so that an unreliable number is never read as a reliable one.

    Args:
        value_text (str): The value.
        measurement (TimingMeasurement): The measurement of the value.

    Returns:
        Text: The value, marked when unsteady.
    """
    text = Text(text=value_text)
    if measure_spread(measurement=measurement) > UNSTEADY_SPREAD:
        text.append(text=UNSTEADY_MARK, style=NOTE_STYLE)
    return text


def format_speed(value: float, decimals: int) -> str:
    """
    Writes a time or a throughput of the report with its digits grouped by three, as the console summaries of the
    training write them.

    Args:
        value (float): The value.
        decimals (int): The number of decimals.

    Returns:
        str: The value, e.g. '1 120.8'.
    """
    return MetricTracker.format_grouped_number(value=value, decimals=decimals)


def describe_steadiness(rows: List[BenchmarkRow]) -> Optional[str]:
    """
    Writes the note of a table of the report on the spread of its timing windows: the largest spread when every timing
    is steady, otherwise what the UNSTEADY_MARK of the unsteady cells means and how many there are.

    Args:
        rows (List[BenchmarkRow]): The rows of the table.

    Returns:
        Optional[str]: The note, None when the table has no measurement.
    """
    spreads = [measure_spread(measurement=row["measurement"]) for row in rows if row["measurement"] is not None]
    if not spreads:
        return None
    number_unsteady = sum(spread > UNSTEADY_SPREAD for spread in spreads)
    if number_unsteady == 0:
        return (f"Timings steady: the timing windows of every cell are within {format_percentage(value=max(spreads))} "
                f"of each other.")
    return (f"{UNSTEADY_MARK.strip()} Unsteady timing ({number_unsteady} of {len(spreads)}): its slowest and fastest "
            f"timing windows differ by more than {format_percentage_label(value=UNSTEADY_SPREAD)} (up to "
            f"{format_percentage(value=max(spreads))}), e.g. while other work shares the GPU; run again on an idle "
            f"GPU to trust it.")


def build_training_table(rows: List[BenchmarkRow], sample_name: str, counts_frames: bool) -> Table:
    """
    Builds the table of the training section of the report: one row per batch size (the configured one and the largest
    that fits), with the time of a training step, the throughput (and the frame throughput of the sequence tasks) and
    the peak memory, or the status of a batch size that could not be timed.

    Args:
        rows (List[BenchmarkRow]): The rows of the configured and largest batch sizes.
        sample_name (str): The kind of sample of the task, 'Images', 'Sequences' or 'Pairs'.
        counts_frames (bool): Whether a frame throughput column is added (sequence tasks).

    Returns:
        Table: The rich table.
    """
    columns = ["Batch Size", "Milliseconds\nPer Step", f"{sample_name}\nPer Second"] + \
        (["Frames\nPer Second"] if counts_frames else []) + ["Peak Memory\n(GB)"]
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, header_style=TABLE_HEADER_STYLE)
    for column_index, column_name in enumerate(columns):
        table.add_column(header=Text(text=column_name), justify="left" if column_index == 0 else "right", no_wrap=True)

    # Every batch size with its timing, or its status in the place of the timing
    for row in rows:
        label = Text(text=f"{row['batch_size']}  ({row['label']})")
        measurement = row["measurement"]
        if measurement is None:
            table.add_row(label, Text(text=row["status"], style=get_status_style(status=row["status"])),
                          *[Text(text=MISSING_VALUE_TEXT)] * (len(columns) - 2))
            continue
        seconds = measurement["seconds_per_iteration"]
        milliseconds_text = format_speed(value=1000.0 * seconds, decimals=2)
        cells = [build_timing_text(value_text=milliseconds_text, measurement=measurement),
                 Text(text=format_speed(value=row["batch_size"] / seconds, decimals=1))]
        if counts_frames:
            cells.append(Text(text=format_speed(value=row["frames_per_batch"] / seconds, decimals=1)))
        cells.append(Text(text=format_gigabytes(number_bytes=measurement["peak_memory_bytes"])))
        table.add_row(label, *cells)
    return table


def describe_inference_column(batch_size: int, configured_batch_size: int, largest_batch_size: int,
                              sample_name: str, counts_frames: bool) -> str:
    """
    Writes the header of one batch size column of the inference table on four lines, so that the columns stay narrow:
    the batch size, its role (configured, largest that fits, or none), then its unit, the milliseconds per sample at a
    batch of 1 (the latency of a request alone), the samples per second otherwise (with the frames per second in
    brackets for the sequence tasks).

    Args:
        batch_size (int): The batch size of the column.
        configured_batch_size (int): The batch size of the configuration.
        largest_batch_size (int): The largest inference batch size that fits.
        sample_name (str): The kind of sample of the task, 'Images', 'Sequences' or 'Pairs'.
        counts_frames (bool): Whether the frames per second are written in brackets (sequence tasks).

    Returns:
        str: The header, on four lines.
    """
    roles = (["Configured"] if batch_size == configured_batch_size else []) + \
        (["Largest"] if batch_size == largest_batch_size else [])
    if batch_size == 1:
        return f"Batch 1\n{', '.join(roles)}\nMilliseconds\nPer {sample_name[:-1]}"
    return f"Batch {batch_size}\n{', '.join(roles)}\n{sample_name}\nPer Second" + (" (Frames)" if counts_frames else "")


def build_inference_cell(row: Optional[BenchmarkRow], counts_frames: bool) -> Text:
    """
    Writes one cell of the inference table: at a batch of 1 the milliseconds per sample, otherwise the samples per
    second (and the frames per second in brackets for the sequence tasks), the status of a batch size that could not be
    timed, or '—' when the format has no row at this batch size.

    Args:
        row (Optional[BenchmarkRow]): The row of the format at the batch size of the column, None when there is none.
        counts_frames (bool): Whether the frames per second are added (sequence tasks).

    Returns:
        Text: The cell.
    """
    if row is None:
        return Text(text=MISSING_VALUE_TEXT)
    measurement = row["measurement"]
    if measurement is None:
        return Text(text=row["status"], style=get_status_style(status=row["status"]))
    seconds = measurement["seconds_per_iteration"]
    if row["batch_size"] == 1:
        return build_timing_text(value_text=format_speed(value=1000.0 * seconds, decimals=2), measurement=measurement)
    value_text = format_speed(value=row["batch_size"] / seconds, decimals=1)
    if counts_frames:
        value_text += f" ({format_speed(value=row['frames_per_batch'] / seconds, decimals=1)})"
    return build_timing_text(value_text=value_text, measurement=measurement)


def compute_speed_ups(rows: List[BenchmarkRow], batch_size: int) -> Dict[str, float]:
    """
    Gives the speed-up of every format measured at a batch size over the network before export (REFERENCE_FORMAT_NAME)
    at the same batch size, the throughput of the format divided by that of the reference.

    Args:
        rows (List[BenchmarkRow]): The rows of the inference table.
        batch_size (int): The batch size of the comparison, the configured one when it fits.

    Returns:
        Dict[str, float]: The speed-up of every measured format by its label, empty when the reference was not
            measured at this batch size.
    """
    reference_label = [inference_format["label"] for inference_format in INFERENCE_FORMATS
                       if inference_format["name"] == REFERENCE_FORMAT_NAME][0]
    throughputs = {row["label"]: row["batch_size"] / row["measurement"]["seconds_per_iteration"] for row in rows
                   if row["batch_size"] == batch_size and row["measurement"] is not None}
    if reference_label not in throughputs:
        return {}
    return {label: throughput / throughputs[reference_label] for label, throughput in throughputs.items()}


def build_inference_table(rows: List[BenchmarkRow], configured_batch_size: int, largest_batch_size: int,
                          sample_name: str, counts_frames: bool) -> Table:
    """
    Builds the table of the inference section of the report: one row per format, grouped into the passes of the
    repository and the files served by Triton, one column per batch size (1, the configured one, the largest that
    fits), and the speed-up of every format over the network before export at the configured batch size (the largest
    one when the configured one does not fit). A format that could not be opened gives its status in its first cell.

    Args:
        rows (List[BenchmarkRow]): The rows of every format and batch size.
        configured_batch_size (int): The batch size of the configuration.
        largest_batch_size (int): The largest inference batch size that fits, 0 when none does.
        sample_name (str): The kind of sample of the task, 'Images', 'Sequences' or 'Pairs'.
        counts_frames (bool): Whether the frames per second are added (sequence tasks).

    Returns:
        Table: The rich table.
    """
    batch_sizes = sorted({row["batch_size"] for row in rows if row["batch_size"] is not None})
    comparison_batch_size = configured_batch_size if configured_batch_size in batch_sizes else max(batch_sizes,
                                                                                                    default=0)
    speed_ups = compute_speed_ups(rows=rows, batch_size=comparison_batch_size)

    # Columns of the batch sizes (a single status column when not even one sample fits) and of the speed-up
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, header_style=TABLE_HEADER_STYLE)
    table.add_column(header=Text(text="What Runs"), justify="left", vertical="bottom", no_wrap=True)
    column_headers = [describe_inference_column(batch_size=batch_size, configured_batch_size=configured_batch_size,
                                                largest_batch_size=largest_batch_size, sample_name=sample_name,
                                                counts_frames=counts_frames) for batch_size in batch_sizes]
    column_headers = column_headers or ["Result"]
    if speed_ups:
        column_headers.append(f"Speed-Up\nAt Batch\n{comparison_batch_size}")
    for column_header in column_headers:
        table.add_column(header=Text(text=column_header), justify="right", vertical="bottom", no_wrap=True,
                         min_width=STATUS_COLUMN_WIDTH)

    # Every group with a title row, then its formats indented below it
    for group in dict.fromkeys(inference_format["group"] for inference_format in INFERENCE_FORMATS):
        group_formats = [inference_format for inference_format in INFERENCE_FORMATS
                         if inference_format["group"] == group and
                         any(row["label"] == inference_format["label"] for row in rows)]
        if not group_formats:
            continue
        table.add_row(Text(text=group, style=GROUP_STYLE), *[Text(text="")] * len(column_headers))
        for inference_format in group_formats:
            format_rows = {row["batch_size"]: row for row in rows if row["label"] == inference_format["label"]}

            # A format that could not be opened gives its status once, the others a cell per batch size
            if None in format_rows:
                status = format_rows[None]["status"]
                cells = [Text(text=status, style=get_status_style(status=status))] + \
                    [Text(text=MISSING_VALUE_TEXT)] * (len(column_headers) - 1)
            else:
                cells = [build_inference_cell(row=format_rows[batch_size] if batch_size in format_rows else None,
                                              counts_frames=counts_frames) for batch_size in batch_sizes]
                if speed_ups:
                    cells.append(Text(text=f"{speed_ups[inference_format['label']]:.2f} x"
                                      if inference_format["label"] in speed_ups else MISSING_VALUE_TEXT))
            table.add_row(Text(text=f"  {inference_format['label']}"), *cells)
    return table


def build_section_panel(title: str, description: str, table: Table, notes: List[str],
                        warning_notes: List[str]) -> Panel:
    """
    Lays out one section of the report (training or inference) in a panel: the question it answers as its title, what
    its timings include, its table, then its notes and its warnings (formats not available or failed).

    Args:
        title (str): The title of the panel, e.g. '1 · Training : how long will the training take?'.
        description (str): What the timings of the section include and leave out.
        table (Table): The table of the section.
        notes (List[str]): The notes written below the table.
        warning_notes (List[str]): The notes written after them in the colour of the notes.

    Returns:
        Panel: The panel.
    """
    content: List[RenderableType] = [Text(text=description), Text(text=" "), table, Text(text=" ")]
    content += [Text(text=note, style=LEGEND_STYLE) for note in notes]
    content += [Text(text=note, style=NOTE_STYLE) for note in warning_notes]
    return Panel(renderable=Group(*content), title=Text(text=title, style="bold"), title_align="left",
                 border_style=PANEL_BORDER_STYLE)


def build_overview_panel(introduction: str, overview_items: List[Tuple[str, str]], warning_lines: List[str]) -> Panel:
    """
    Lays out the overview opening the report in a panel: what the benchmark measures, then a line per setting of the
    measurement (GPU, memory, batch, weights, timing), and the warnings (other processes on the GPU, initial weights).

    Args:
        introduction (str): The sentence saying what the benchmark measures.
        overview_items (List[Tuple[str, str]]): The name and the value of every setting.
        warning_lines (List[str]): The warnings, written in the colour of the notes.

    Returns:
        Panel: The panel.
    """
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column()
    for item_name, item_value in overview_items:
        grid.add_row(Text(text=item_name), Text(text=item_value))
    content: List[RenderableType] = [Text(text=introduction), Text(text=" "), grid]
    content += [Text(text=line, style=NOTE_STYLE) for line in warning_lines]
    return Panel(renderable=Group(*content), title=Text(text="What Was Measured", style="bold"), title_align="left",
                 border_style=PANEL_BORDER_STYLE)


def print_speed_benchmark_report(header: str, panels: List[Panel]) -> None:
    """
    Prints the report of the speed benchmark at the width of the evaluation reports: a header rule naming the run, then
    the overview panel and the panels of the training and inference sections.

    Args:
        header (str): The text of the header rule.
        panels (List[Panel]): The overview panel, then the section panels.
    """
    print_report(renderables=[Rule(title=Text(text=header, style=HEADER_STYLE), align="left"), *panels])


def join_header_parts(parts: List[str]) -> str:
    """
    Joins the parts of the header of the report with the separator of the evaluation reports.

    Args:
        parts (List[str]): The parts, e.g. the title, the run name and the GPU.

    Returns:
        str: The header.
    """
    return HEADER_SEPARATOR.join(parts)
