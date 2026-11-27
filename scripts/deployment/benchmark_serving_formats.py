"""
Benchmarks the serving formats and precisions of exported models (documentation/architecture_benchmarking/
serving_benchmark_plan.md): it exports trained runs into one Triton model repository per precision setting with
vision_training/deployment/export_model.py (the FP32 TorchScript model with ONNX Runtime CUDA, the FP16 TorchScript
model alone in the torchscript_fp16 setting), times the inference model in the process (eager, TorchScript FP32 and
FP16), then serves every exported model alone in a Triton container and times it under several server settings, batch
sizes and concurrencies, through system shared memory with perf_analyzer of the Triton SDK image and over gRPC with a
tritonclient client (the client of run_inference.py), with the time of the first requests (TensorRT engine builds)
and the memory taken by the model. Every output stays outside the repository, under the benchmark folder given on the
command line, and every measurement is one line of its results.csv, read again at a relaunch so that a stopped sweep
restarts where it stopped. Run as a script from the repository root with the benchmark, runs and data folders of the
machine given as arguments (parse_benchmark_arguments), its help printed when none is given; the plan of the sweep
(models, precision settings, loads and calls) is in the constants of this module, which evaluate_served_formats.py
reads too.
"""

import argparse
import csv
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import numpy as np
import torch
import tritonclient.grpc as triton_client

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.deployment.deployment_utilities import (load_inference_inputs, load_local_inference_model,
                                                             run_local_inference, run_triton_inference)
from vision_training.deployment.export_formats import hide_known_export_warnings
from vision_training.deployment.export_model import export_experiment
from vision_training.deployment.export_specifications import TASK_MODEL_NAMES
from vision_training.deployment.triton_configuration import TORCHSCRIPT_FILE_NAMES
from vision_training.deployment.triton_server import (build_triton_server, load_served_model, send_server_request,
                                                      start_triton_server, stop_triton_server, unload_served_model)
from vision_training.utilities.os_utilities import format_path_link, print_color

# SDK image of the benchmark (perf_analyzer of Triton 2.65.0, the server image being TRITON_IMAGE of
# deployment/triton_server.py), and the name of its container, removed by name when a measurement fails or times out
SDK_IMAGE = "nvcr.io/nvidia/tritonserver:26.01-py3-sdk"
PERF_ANALYZER_CONTAINER_NAME = "serving-benchmark-perf-analyzer"

# Server of the benchmark, which shares the network of the machine on the default ports of Triton (HTTP for the
# repository and health requests, gRPC for the inference requests, metrics for perf_analyzer), and its addresses
BENCHMARK_SERVER = build_triton_server(container_name="serving-benchmark-triton", http_port=8000, grpc_port=8001,
                                       metrics_port=8002)
GRPC_ADDRESS = BENCHMARK_SERVER["grpc_address"]
METRICS_ADDRESS = BENCHMARK_SERVER["metrics_address"]

# Folders of the SDK container: the perf_analyzer input data and its reports
CONTAINER_INPUT_DATA_FOLDER = "/input_data"
CONTAINER_REPORT_FOLDER = "/reports"

# Size of the shared memory region perf_analyzer allocates per output, needed for the outputs of variable shape (the
# frame probabilities of the sequence models), above the largest output of a batch of 16
OUTPUT_SHARED_MEMORY_BYTES = 1 << 20

# Precision setting of the FP16 TorchScript network (torchscript_precision 'FP16' of export_model.py), served alone in
# its repository, the FP32 TorchScript model being exported and served once next to the ONNX model of ONNX Runtime CUDA
TORCHSCRIPT_HALF_PRECISION_SETTING = "torchscript_fp16"

# Processes allowed on the GPU during a measurement: the MPS daemon of the machine and the server being measured
ALLOWED_GPU_PROCESS_NAMES = ["nvidia-cuda-mps-server", "tritonserver"]

# Memory that must stay available on the machine for a measurement to start, and the waits for an idle machine
MINIMUM_AVAILABLE_GIGABYTES = 30.0
IDLE_POLL_SECONDS = 30
MAXIMUM_IDLE_WAIT_SECONDS = 4 * 3600

# Time a perf_analyzer run has to write its report
PERF_ANALYZER_TIMEOUT_SECONDS = 1800
# Memory limit of the perf_analyzer container: perf_analyzer 2.59.1 keeps about 10 MB per 512-pixel request sent
# without shared memory (gRPC or HTTP) until the end of its run, which filled the 128 GB of the machine in the pilot,
# so it only measures the shared memory loads, and a run above the limit is killed in its container
PERF_ANALYZER_MEMORY_LIMIT = "16g"
# Length of the measurement windows of perf_analyzer, three consecutive windows having to agree within 10 %, one run
# per length until one writes its report: the requests of the detector at batch 16 last several seconds, so that a
# window of 2 seconds holds a single request and perf_analyzer stops without a measurement
PERF_ANALYZER_WINDOWS_MILLISECONDS = [2000, 10000, 30000]

# gRPC loads timed by the tritonclient client, one process per request in flight: untimed requests first, then the
# requests ending within the window count, the GPU use and power of the server metrics being sampled during the
# window; the clients have CLIENT_START_TIMEOUT_SECONDS to start and to give back their latencies
CLIENT_WARMUP_SECONDS = 2.0
CLIENT_WINDOW_SECONDS = 10.0
METRICS_SAMPLE_SECONDS = 0.5
CLIENT_START_TIMEOUT_SECONDS = 600

# Server statistics whose average time per request gives the queue and compute times of a gRPC load
COMPUTE_STATISTICS = ["compute_input", "compute_infer", "compute_output"]

# Untimed requests sent after every load at batch sizes 1 and 16, so that the TensorRT engines of every batch size
# the dynamic batcher may form exist before the timed requests
WARMUP_REQUESTS_PER_BATCH_SIZE = 10

# Columns of results.csv, the first ones forming the key of a measurement
KEY_COLUMNS = ["model", "row", "kind", "dynamic_batching", "instance_count", "transport", "batch_size", "concurrency"]
RESULT_COLUMNS = KEY_COLUMNS + ["status", "load_client", "p50_milliseconds", "p99_milliseconds",
                                "average_milliseconds", "inferences_per_second", "images_per_second",
                                "server_queue_microseconds", "server_compute_microseconds", "gpu_utilization_percent",
                                "gpu_power_watts", "first_request_seconds", "memory_gigabytes",
                                "allocator_peak_gigabytes", "other_gpu_processes", "duration_seconds", "raw_file",
                                "error", "timestamp"]

# Plan of the sweep behind the committed results, kept in the code because the keys of results.csv hold neither the
# calls nor the loads, so that a relaunch resuming a sweep measures it as it started. Trained runs, their folders
# relative to --runs_folder and their test splits relative to --data_folder: the 9-class README classifier, the
# reference of the detection benchmark (run_9) and the sequence labelling tutorial run, with the served checkpoint and
# the frames of the sample sent to the models (None for images). The test splits were drawn again after the benchmark
# of 2026-11-17 (classification with 35 categories, of which the loaders keep the 9 of the run), so a new run measures
# the same shapes on other images
BENCHMARKED_MODELS = [
    {"name": "classification", "iteration": 1500, "sequence_length": None,
     "run": "readme_screenshots/CLS_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HFLAT.128-HLN-CWB_MC9",
     "test_split": "classifier_data/synthetic_shapes/test"},
    {"name": "object_detection", "iteration": 15000, "sequence_length": None,
     "run": "hyperparameter_experiments/object_detection/run_9/DET_IN-KAR-1024_SE-RES-BN-B1x16.2x16.2x16.3x32.3x32."
            "5x32.5x32_L4567-NFPN32BN_A6.16-768.d8ffba_RBN-R2x32_DSE-DRES-DBN-D7.2x32.64.32_MC35",
     "test_split": "object_detection_data/synthetic_hard_shapes/test_10000"},
    {"name": "sequence_labelling", "iteration": 2000, "sequence_length": 7,
     "run": "tutorials/M2M_IN-KAR-256_SE-RES-BN-B1x16.2x16.2x16.4x32.5x32.5x32-PT_T50_LSTM2x96-BI_HFLAT.96.96-HLN-"
            "CWB_MC71",
     "test_split": "sequence_data/synthetic_changing_shapes/test_10000"}]
# Precision settings, one model repository and one server each: the TensorRT precision of the export (None for ONNX
# Runtime CUDA and torchscript_fp16, the FP16 TorchScript model alone) and the server environment. BF16 is left out:
# the ONNX Runtime backend of Triton 26.01 refuses precision_mode "BF16" and ignores ORT_TENSORRT_BF16_ENABLE
PRECISION_SETTINGS = {"onnxruntime_cuda": {"tensorrt_precision": None, "environment": {}},
                      "tensorrt_fp32": {"tensorrt_precision": "FP32", "environment": {}},
                      "tensorrt_fp16": {"tensorrt_precision": "FP16", "environment": {}},
                      "torchscript_fp16": {"tensorrt_precision": None, "environment": {}}}
# Batch sizes of the in-process rows and of the first requests of the served rows
BATCH_SIZES = [1, 4, 16]
# Loads of the served rows (focused scope, user decision 2026-11-17): the exported configuration (dynamic batching on,
# 1 instance) for every row over the whole grid, and the other server settings (dynamic batching, instance count) for
# TensorRT FP16 only, at batch 1, where several clients sending at once is where they can matter
LOAD_GROUPS = [{"precision_settings": list(PRECISION_SETTINGS), "server_settings": [(True, 1)],
                "transports": ["grpc", "system_shared_memory"], "batch_sizes": [1, 4, 16],
                "concurrencies": [1, 4, 16]},
               {"precision_settings": ["tensorrt_fp16"], "server_settings": [(False, 1), (True, 2), (False, 2)],
                "transports": ["grpc", "system_shared_memory"], "batch_sizes": [1], "concurrencies": [1, 4, 16]}]
# Largest batch size of the exports and of the Triton configurations, also that of the test loaders of
# evaluate_served_formats.py
MAXIMUM_BATCH_SIZE = 16
# Timed and untimed calls of the in-process rows per batch size
IN_PROCESS_ITERATIONS = 200
IN_PROCESS_WARMUP_ITERATIONS = 20

# Phases of the benchmark, run in this order, each skipping what results.csv or the model repositories already hold
BENCHMARK_PHASES = ["exports", "in_process", "triton"]


def read_available_gigabytes() -> float:
    """
    Reads the memory available on the machine, the measure of the memory a model takes on the GB10, whose CPU and GPU
    share one memory (nvidia-smi gives no used memory there).

    Returns:
        float: MemAvailable of /proc/meminfo in gigabytes.

    Raises:
        RuntimeError: If /proc/meminfo has no MemAvailable line.
    """
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 ** 2)
    raise RuntimeError("Expected a MemAvailable line in /proc/meminfo.")


def list_other_gpu_processes() -> List[str]:
    """
    Lists the processes using the GPU besides the MPS daemon, the measured server and this script, so that a training
    started during the sweep is seen before it slows a measurement down.

    Returns:
        List[str]: '<pid>:<process name>' of every other process, empty when the GPU is idle.

    Raises:
        RuntimeError: If nvidia-smi fails.
    """
    query = subprocess.run(args=["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                           capture_output=True, text=True, check=False)
    if query.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {query.stderr.strip()}")
    other_processes = []
    for line in query.stdout.strip().splitlines():
        process_id, process_name = [field.strip() for field in line.split(",", maxsplit=1)]
        if int(process_id) == os.getpid() or any(name in process_name for name in ALLOWED_GPU_PROCESS_NAMES):
            continue
        other_processes.append(f"{process_id}:{process_name}")
    return other_processes


def wait_for_idle_machine() -> List[str]:
    """
    Waits before a measurement until no other process uses the GPU and enough memory is available, so that the
    numbers are those of the measured model alone; after MAXIMUM_IDLE_WAIT_SECONDS the measurement goes on and the
    processes seen are recorded with it.

    Returns:
        List[str]: The other GPU processes still running when the measurement starts, empty on an idle machine.
    """
    waited_seconds = 0
    while True:
        other_processes = list_other_gpu_processes()
        available_gigabytes = read_available_gigabytes()
        if not other_processes and available_gigabytes >= MINIMUM_AVAILABLE_GIGABYTES:
            return []
        if waited_seconds >= MAXIMUM_IDLE_WAIT_SECONDS:
            print_color(color="red", output=f"Machine still busy after {waited_seconds} s, measuring anyway : "
                                            f"{other_processes}, {available_gigabytes:.1f} GB available")
            return other_processes
        if waited_seconds == 0:
            print_color(color="yellow", output=f"Waiting for an idle machine : {other_processes}, "
                                               f"{available_gigabytes:.1f} GB available")
        time.sleep(IDLE_POLL_SECONDS)
        waited_seconds += IDLE_POLL_SECONDS


def read_completed_keys(results_path: Path) -> set:
    """
    Reads the keys of the measurements already in results.csv, so that a relaunch skips them and a stopped sweep
    restarts where it stopped.

    Args:
        results_path (Path): The results file, which may not exist yet.

    Returns:
        set: The tuple of KEY_COLUMNS values of every recorded measurement, as strings.
    """
    if not results_path.is_file():
        return set()
    with open(results_path, newline="") as results_file:
        return {tuple(row[column] for column in KEY_COLUMNS) for row in csv.DictReader(results_file)}


def build_result_key(result: Dict[str, Any]) -> Tuple[str, ...]:
    """
    Builds the key of a measurement as read back from results.csv, to compare a planned measurement with the recorded
    ones.

    Args:
        result (Dict[str, Any]): The measurement, holding every key column.

    Returns:
        Tuple[str, ...]: The KEY_COLUMNS values as strings.
    """
    return tuple(str(result[column]) for column in KEY_COLUMNS)


def append_result(results_path: Path, result: Dict[str, Any]) -> None:
    """
    Appends one measurement to results.csv, written at once so that a stopped sweep keeps every finished measurement,
    the header being written with the first line.

    Args:
        results_path (Path): The results file.
        result (Dict[str, Any]): The measurement, keyed by the columns of RESULT_COLUMNS (missing ones left empty).

    Raises:
        ValueError: If the measurement holds a column outside RESULT_COLUMNS.
    """
    unknown_columns = set(result) - set(RESULT_COLUMNS)
    if unknown_columns:
        raise ValueError(f"Expected the columns of RESULT_COLUMNS, got the unknown columns {sorted(unknown_columns)}.")
    write_header = not results_path.is_file()
    with open(results_path, "a", newline="") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({**result, "timestamp": datetime.now().isoformat(timespec="seconds")})


def select_test_sample(test_data_folder: Path, sample_folder: Path, sequence_length: Optional[int]) -> Path:
    """
    Links the test sample sent to the served models into a folder of its own, read by load_inference_inputs as one
    request: the first image of the manifest, or for the sequence tasks the frames of the first sequence holding
    sequence_length frames, named in their order.

    Args:
        test_data_folder (Path): The test split folder, holding 'labels.json' and the 'data' folder.
        sample_folder (Path): The folder receiving the links, emptied first.
        sequence_length (Optional[int]): The number of frames of the sequence, None for the image tasks.

    Returns:
        Path: The sample folder.

    Raises:
        ValueError: If the manifest holds no images, or no sequence of sequence_length frames.
    """
    with open(test_data_folder / "labels.json") as manifest_file:
        manifest = json.load(manifest_file)

    # File names of the sample, relative to the data folder: the frames of a sequence or a single image
    if sequence_length is not None:
        sequence = next((sequence for sequence in manifest["sequences"] if len(sequence["frames"]) == sequence_length),
                        None)
        if sequence is None:
            raise ValueError(f"Expected a test sequence of {sequence_length} frames in {test_data_folder}.")
        file_names = sequence["frames"]
    else:
        image_records = manifest["images"] if "images" in manifest else manifest["samples"]
        if not image_records:
            raise ValueError(f"Expected at least one test image in {test_data_folder}.")
        file_names = [image_records[0]["file_name"]]

    # One link per file, numbered so that the frames keep their order once sorted by name
    if sample_folder.exists():
        shutil.rmtree(sample_folder)
    sample_folder.mkdir(parents=True)
    for file_index, file_name in enumerate(file_names):
        source_path = test_data_folder / "data" / file_name
        (sample_folder / f"{file_index:06d}{source_path.suffix}").symlink_to(source_path)
    return sample_folder


def prepare_benchmark_inputs(model: Dict[str, Any], inputs_folder: Path) -> Dict[str, np.ndarray]:
    """
    Builds the request sent to the served models of a run from its test sample, preprocessed as the data loaders do
    (load_inference_inputs), and writes it for perf_analyzer: one raw float32 file per input, named after the input,
    holding the batch-1 request it repeats to every batch size.

    Args:
        model (Dict[str, Any]): The benchmarked model, with its 'name', 'experiment_folder', 'test_data_folder' and
            'sequence_length' (None for the image tasks).
        inputs_folder (Path): The folder of the inputs of the model, receiving 'source/' (links to the sample) and
            'perf_analyzer_data/'.

    Returns:
        Dict[str, np.ndarray]: The float32 batch-1 request, keyed by input name.
    """
    sample_folder = select_test_sample(test_data_folder=model["test_data_folder"],
                                       sample_folder=inputs_folder / "source", sequence_length=model["sequence_length"])
    request = load_inference_inputs(image_folder=sample_folder, experiment_folder=model["experiment_folder"])[0]

    # Raw data of every input without its batch dimension, as perf_analyzer reads a directory of input data
    data_folder = inputs_folder / "perf_analyzer_data"
    data_folder.mkdir(parents=True, exist_ok=True)
    for input_name, input_batch in request.items():
        np.ascontiguousarray(input_batch[0], dtype=np.float32).tofile(data_folder / input_name)
    return request


def read_served_model_name(experiment_folder: Path) -> str:
    """
    Gives the Triton name of the exported models of a run, followed by '_<export format>' in the repository, from the
    task of its training configuration.

    Args:
        experiment_folder (Path): The run folder.

    Returns:
        str: The Triton model name of the task, e.g. 'object_detector'.
    """
    experiment_configuration, _ = load_experiment_configuration(
        configuration_path=experiment_folder / "training_configuration.yaml")
    return TASK_MODEL_NAMES[experiment_configuration["task"]]


def read_served_output_names(client: triton_client.InferenceServerClient, model_name: str) -> List[str]:
    """
    Reads the names of every output of a served model from the server, as the TorchScript and ONNX models name them
    ('<name>__<index>' or '<name>'), so that the requests of the sweep ask for all of them.

    Args:
        client (triton_client.InferenceServerClient): The gRPC client of the server.
        model_name (str): The Triton model name.

    Returns:
        List[str]: The served output names.
    """
    return [output.name for output in client.get_model_metadata(model_name=model_name).outputs]


def repeat_request(request: Dict[str, np.ndarray], batch_size: int) -> Dict[str, np.ndarray]:
    """
    Repeats the batch-1 request of a model to a batch size, as perf_analyzer repeats its input data.

    Args:
        request (Dict[str, np.ndarray]): The batch-1 request, keyed by input name.
        batch_size (int): The batch size of the repeated request.

    Returns:
        Dict[str, np.ndarray]: The request of batch_size samples.
    """
    return {input_name: np.ascontiguousarray(np.repeat(input_batch, repeats=batch_size, axis=0))
            for input_name, input_batch in request.items()}


def count_frames_per_sample(request: Dict[str, np.ndarray]) -> int:
    """
    Counts the images of one sample of a request, 1 for the image tasks and the number of frames for the sequences,
    so that the throughput of every model is given in images per second.

    Args:
        request (Dict[str, np.ndarray]): The batch-1 request, [1, 3, H, W] or [1, T, 3, H, W] per input.

    Returns:
        int: The images of one sample, summed over the inputs (both members of a comparison).
    """
    return sum(input_batch.shape[1] if input_batch.ndim == 5 else 1 for input_batch in request.values())


def export_precision_setting(experiment_folder: Path, iteration: int, repository: Path, formats: List[str],
                             tensorrt_precision: Optional[str], torchscript_precision: str,
                             maximum_batch_size: int) -> None:
    """
    Exports one run into the model repository of one precision setting, run in a child process so that the CUDA
    context of the export never stays in the process of the sweep. The TensorRT engine check of the export is skipped
    (its FP16 overflow scan still runs): the benchmark serves every engine itself, and evaluate_served_formats.py scores
    it on the test split.

    Args:
        experiment_folder (Path): The run folder.
        iteration (int): The exported checkpoint.
        repository (Path): The Triton model repository of the precision setting.
        formats (List[str]): The export formats, among 'torchscript' and 'onnx'.
        tensorrt_precision (Optional[str]): The TensorRT precision written into the ONNX configuration, None for ONNX
            Runtime CUDA.
        torchscript_precision (str): The precision of the TorchScript network (get_torchscript_precision).
        maximum_batch_size (int): The largest batch size of the exported models.
    """
    hide_known_export_warnings()
    export_experiment(experiment_folder=experiment_folder, iteration=iteration, use_averaged_weights=False,
                      formats=formats, repository=repository, model_version=1, maximum_batch_size=maximum_batch_size,
                      tensorrt_precision=tensorrt_precision, torchscript_precision=torchscript_precision,
                      use_saved_temperature=False, saved_thresholds_path=None, test_split_parity=False,
                      skip_tensorrt_check=True)


def run_in_child_process(target: Any, keyword_arguments: Dict[str, Any]) -> None:
    """
    Runs a function in a new spawned process and waits for it, so that its CUDA context and memory are freed when it
    ends and the memory measured by the next one starts from the same state.

    Args:
        target (Any): The module-level function to run.
        keyword_arguments (Dict[str, Any]): Its keyword arguments.

    Raises:
        RuntimeError: If the child process fails.
    """
    child_process = multiprocessing.get_context(method="spawn").Process(target=target, kwargs=keyword_arguments)
    child_process.start()
    child_process.join()
    if child_process.exitcode != 0:
        raise RuntimeError(f"{target.__name__} failed in its child process (exit code {child_process.exitcode}).")


def get_repository_formats(precision_setting: str) -> List[str]:
    """
    Gives the formats exported into the repository of a precision setting: the FP32 TorchScript model, the same for
    every ONNX setting, is exported and served once, with ONNX Runtime CUDA, and the FP16 TorchScript model alone in
    TORCHSCRIPT_HALF_PRECISION_SETTING.

    Args:
        precision_setting (str): The precision setting, e.g. 'onnxruntime_cuda', 'tensorrt_fp16' or 'torchscript_fp16'.

    Returns:
        List[str]: The export formats of the repository.
    """
    if precision_setting == TORCHSCRIPT_HALF_PRECISION_SETTING:
        return ["torchscript"]
    return ["torchscript", "onnx"] if precision_setting == "onnxruntime_cuda" else ["onnx"]


def get_torchscript_precision(precision_setting: str) -> str:
    """
    Gives the precision of the TorchScript network exported into the repository of a precision setting, so that the
    FP16 TorchScript file (convert_to_half_precision, 'model_fp16.pt') is served by a setting of its own.

    Args:
        precision_setting (str): The precision setting.

    Returns:
        str: 'FP16' for TORCHSCRIPT_HALF_PRECISION_SETTING, 'FP32' for every other setting.
    """
    return "FP16" if precision_setting == TORCHSCRIPT_HALF_PRECISION_SETTING else "FP32"


def get_row_name(precision_setting: str, export_format: str) -> str:
    """
    Names the row of a served model in results.csv: 'triton_torchscript' for the FP32 TorchScript model, and
    'triton_<precision setting>' for the ONNX model and the FP16 TorchScript model ('triton_torchscript_fp16').

    Args:
        precision_setting (str): The precision setting of the repository.
        export_format (str): Either 'torchscript' or 'onnx'.

    Returns:
        str: The row name.
    """
    if export_format == "torchscript" and get_torchscript_precision(precision_setting=precision_setting) == "FP32":
        return "triton_torchscript"
    return f"triton_{precision_setting}"


def list_in_process_rows(model_repositories: Path, served_model_name: str, precision_settings: List[str]
                         ) -> List[Tuple[str, Optional[Path]]]:
    """
    Lists the in-process rows of a model with the TorchScript file each one loads: the eager model first, then the
    TorchScript file of every benchmarked setting whose repository holds one, 'in_process_torchscript' for the FP32
    'model.pt' of ONNX Runtime CUDA and 'in_process_torchscript_fp16' for the 'model_fp16.pt' of
    TORCHSCRIPT_HALF_PRECISION_SETTING, so that both precisions are also timed without a server.

    Args:
        model_repositories (Path): The folder of the model repositories of the benchmark, one per setting.
        served_model_name (str): The Triton name of the model, e.g. 'image_classifier'.
        precision_settings (List[str]): The benchmarked precision settings.

    Returns:
        List[Tuple[str, Optional[Path]]]: The row names with their TorchScript file, None for the eager model.
    """
    rows = [("in_process_eager", None)]
    for precision_setting in precision_settings:
        if "torchscript" not in get_repository_formats(precision_setting=precision_setting):
            continue
        # Row and file named after the precision of the TorchScript network of the setting
        torchscript_precision = get_torchscript_precision(precision_setting=precision_setting)
        row_name = "in_process_torchscript" if torchscript_precision == "FP32" else "in_process_torchscript_fp16"
        rows.append((row_name, model_repositories / precision_setting / f"{served_model_name}_torchscript" / "1" /
                     TORCHSCRIPT_FILE_NAMES[torchscript_precision]))
    return rows


def measure_in_process_row(experiment_folder: Path, iteration: int, torchscript_path: Optional[Path],
                           request: Dict[str, np.ndarray], batch_sizes: List[int], number_iterations: int,
                           number_warmup_iterations: int, output_path: Path) -> None:
    """
    Times the inference model of a run in this process, one call at a time, at every batch size: the eager model, or
    its TorchScript export loaded with torch.jit.load. Each call converts the request to the GPU and its outputs back
    (run_local_inference), as a request to a server would. Run in a child process, whose memory before the model is
    loaded (CUDA context included) is the reference of the memory it takes. The measures are written as JSON.

    Args:
        experiment_folder (Path): The run folder.
        iteration (int): The checkpoint of the eager model.
        torchscript_path (Optional[Path]): The TorchScript export to time, None for the eager model.
        request (Dict[str, np.ndarray]): The batch-1 request, keyed by input name.
        batch_sizes (List[int]): The batch sizes timed.
        number_iterations (int): The timed calls per batch size.
        number_warmup_iterations (int): The untimed calls made first at every batch size.
        output_path (Path): The JSON file receiving one measure per batch size.
    """
    # CUDA context created before the reference, so that the memory measured is the one of the model and its calls
    torch.zeros(1, device=torch.device("cuda"))
    torch.cuda.synchronize()
    available_before_gigabytes = read_available_gigabytes()

    inference_model, _ = load_local_inference_model(experiment_folder=experiment_folder, iteration=iteration)
    if torchscript_path is not None:
        inference_model = torch.jit.load(str(torchscript_path), map_location=torch.device("cuda"))
    inference_model.eval()

    measures = []
    for batch_size in batch_sizes:
        batched_request = repeat_request(request=request, batch_size=batch_size)

        # Untimed calls first, then every timed call ends with its outputs back on the CPU
        for _ in range(number_warmup_iterations):
            run_local_inference(inputs=batched_request, inference_model=inference_model, decoding_settings=None)
        call_seconds = []
        for _ in range(number_iterations):
            call_start = time.perf_counter()
            run_local_inference(inputs=batched_request, inference_model=inference_model, decoding_settings=None)
            torch.cuda.synchronize()
            call_seconds.append(time.perf_counter() - call_start)
        call_milliseconds = np.array(call_seconds) * 1000.0
        measures.append({"batch_size": batch_size, "p50_milliseconds": float(np.percentile(call_milliseconds, 50)),
                         "p99_milliseconds": float(np.percentile(call_milliseconds, 99)),
                         "average_milliseconds": float(call_milliseconds.mean()),
                         "inferences_per_second": batch_size * number_iterations / float(np.sum(call_seconds))})

    # Memory of the model and its calls, and the peak of the PyTorch allocator
    memory_gigabytes = available_before_gigabytes - read_available_gigabytes()
    allocator_peak_gigabytes = torch.cuda.max_memory_reserved() / (1024 ** 3)
    output_path.write_text(json.dumps({"measures": measures, "memory_gigabytes": memory_gigabytes,
                                       "allocator_peak_gigabytes": allocator_peak_gigabytes}))


def build_server_configuration(base_configuration: Dict[str, Any], dynamic_batching: bool,
                               instance_count: int) -> Dict[str, Any]:
    """
    Changes the configuration of a served model for one server setting, given to Triton at load time instead of
    editing config.pbtxt: the dynamic batcher written by the export kept or removed, and the number of instances of
    the model on the GPU.

    Args:
        base_configuration (Dict[str, Any]): The configuration of the model as Triton reads it from config.pbtxt
            (JSON of the model configuration endpoint).
        dynamic_batching (bool): Whether Triton groups the requests waiting for the model into larger batches.
        instance_count (int): The number of copies of the model running at the same time on the GPU.

    Returns:
        Dict[str, Any]: The configuration of the server setting.

    Raises:
        ValueError: If the base configuration has not exactly one instance group, or no dynamic batcher.
    """
    if len(base_configuration["instance_group"]) != 1 or "dynamic_batching" not in base_configuration:
        raise ValueError(f"Expected one instance group and a dynamic batcher in the exported configuration, got "
                         f"{base_configuration['instance_group']} and dynamic_batching "
                         f"{base_configuration.get('dynamic_batching')}.")
    configuration = json.loads(json.dumps(base_configuration))
    configuration["instance_group"][0]["count"] = instance_count
    if not dynamic_batching:
        del configuration["dynamic_batching"]
    return configuration


def time_first_requests(model_name: str, request: Dict[str, np.ndarray], batch_sizes: List[int]) -> Dict[int, float]:
    """
    Times the first request at every batch size, in increasing order, right after the model is loaded with an empty
    engine cache: ONNX Runtime builds a TensorRT engine at the first request and again at every larger batch, which
    the first clients of a new server wait for.

    Args:
        model_name (str): The Triton model name.
        request (Dict[str, np.ndarray]): The batch-1 request, keyed by input name.
        batch_sizes (List[int]): The batch sizes, in the order sent.

    Returns:
        Dict[int, float]: The seconds of the first request of every batch size.
    """
    client = triton_client.InferenceServerClient(url=GRPC_ADDRESS)
    served_output_names = read_served_output_names(client=client, model_name=model_name)
    first_request_seconds = {}
    for batch_size in batch_sizes:
        request_start = time.perf_counter()
        run_triton_inference(inputs=repeat_request(request=request, batch_size=batch_size), client=client,
                             model_name=model_name, output_names=served_output_names, shared_memory_handle=None,
                             region_name=None)
        first_request_seconds[batch_size] = time.perf_counter() - request_start
    return first_request_seconds


def warm_up_served_model(model_name: str, request: Dict[str, np.ndarray], maximum_batch_size: int) -> None:
    """
    Sends untimed requests at batch sizes 1 and maximum_batch_size after every load, so that the engines or kernels
    of every batch size the dynamic batcher may form exist before the timed requests.

    Args:
        model_name (str): The Triton model name.
        request (Dict[str, np.ndarray]): The batch-1 request, keyed by input name.
        maximum_batch_size (int): The largest batch size of the model.
    """
    client = triton_client.InferenceServerClient(url=GRPC_ADDRESS)
    served_output_names = read_served_output_names(client=client, model_name=model_name)
    for batch_size in [1, maximum_batch_size]:
        batched_request = repeat_request(request=request, batch_size=batch_size)
        for _ in range(WARMUP_REQUESTS_PER_BATCH_SIZE):
            run_triton_inference(inputs=batched_request, client=client, model_name=model_name,
                                 output_names=served_output_names, shared_memory_handle=None, region_name=None)


def parse_gpu_metric(cell: str) -> float:
    """
    Reads one GPU metric of a perf_analyzer report, written per GPU as 'GPU-<uuid>:<value>;', for the GPU use and
    power of a measurement (the memory columns read 0 on the GB10, whose memory is shared with the CPU).

    Args:
        cell (str): The cell of the report, e.g. 'GPU-2285...:0.375897;'.

    Returns:
        float: The mean value over the GPUs of the cell.

    Raises:
        ValueError: If the cell holds no GPU value.
    """
    values = [float(entry.rsplit(":", maxsplit=1)[1]) for entry in cell.split(";") if entry.strip()]
    if not values:
        raise ValueError(f"Expected 'GPU-<uuid>:<value>;' entries in a GPU metric of perf_analyzer, got '{cell}'.")
    return sum(values) / len(values)


def parse_perf_analyzer_report(report_path: Path) -> Dict[str, float]:
    """
    Reads the CSV report of one perf_analyzer measurement (one concurrency), whose latencies are in microseconds and
    whose GPU use is a fraction, read from the metrics of the server.

    Args:
        report_path (Path): The report written by perf_analyzer -f --verbose-csv --collect-metrics.

    Returns:
        Dict[str, float]: 'p50_milliseconds', 'p99_milliseconds', 'average_milliseconds', 'inferences_per_second',
            'server_queue_microseconds', 'server_compute_microseconds' (input, inference and output),
            'gpu_utilization_percent' and 'gpu_power_watts'.

    Raises:
        ValueError: If the report does not hold exactly one measurement.
    """
    with open(report_path, newline="") as report_file:
        rows = list(csv.DictReader(report_file))
    if len(rows) != 1:
        raise ValueError(f"Expected one measurement in {report_path}, got {len(rows)}.")
    row = rows[0]
    server_compute_microseconds = sum(float(row[column]) for column in
                                      ["Server Compute Input", "Server Compute Infer", "Server Compute Output"])
    return {"p50_milliseconds": float(row["p50 latency"]) / 1000.0,
            "p99_milliseconds": float(row["p99 latency"]) / 1000.0,
            "average_milliseconds": float(row["Avg latency"]) / 1000.0,
            "inferences_per_second": float(row["Inferences/Second"]),
            "server_queue_microseconds": float(row["Server Queue"]),
            "server_compute_microseconds": server_compute_microseconds,
            "gpu_utilization_percent": parse_gpu_metric(cell=row["Avg GPU Utilization"]) * 100.0,
            "gpu_power_watts": parse_gpu_metric(cell=row["Avg GPU Power Usage"])}


def run_perf_analyzer(model_name: str, batch_size: int, concurrency: int, transport: str, input_data_folder: Path,
                      shapes: Dict[str, List[int]], report_path: Path) -> Tuple[Optional[Dict[str, float]], str]:
    """
    Measures one load on a served model with perf_analyzer of the Triton SDK image: batch_size samples per request,
    concurrency requests in flight, the inputs sent through system shared memory (or inside the gRPC requests, which
    the sweep leaves to run_client_load, see PERF_ANALYZER_MEMORY_LIMIT), until three measurement windows agree within
    10 %. A run ending without a report is run again with the next window length of PERF_ANALYZER_WINDOWS_MILLISECONDS,
    the output and return code of every run kept in the log.

    Args:
        model_name (str): The Triton model name.
        batch_size (int): The samples per request.
        concurrency (int): The requests in flight.
        transport (str): 'grpc' or 'system_shared_memory'.
        input_data_folder (Path): The folder of the raw batch-1 inputs, one file per input.
        shapes (Dict[str, List[int]]): The shape of every input with a dynamic dimension (frames), batch excluded.
        report_path (Path): The CSV report written by perf_analyzer.

    Returns:
        Tuple[Optional[Dict[str, float]], str]: The measure (parse_perf_analyzer_report), None when perf_analyzer
            failed, and the end of its output when it failed (empty otherwise).
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Arguments of perf_analyzer: the input shapes, the transport, and its container with inputs and report mounted
    shape_arguments = [argument for input_name, shape in shapes.items()
                       for argument in ["--shape", f"{input_name}:{','.join(str(size) for size in shape)}"]]
    transport_arguments = (["--shared-memory", "system", "--output-shared-memory-size", str(OUTPUT_SHARED_MEMORY_BYTES)]
                           if transport == "system_shared_memory" else ["--shared-memory", "none"])
    command = ["docker", "run", "--rm", "--name", PERF_ANALYZER_CONTAINER_NAME, "--network", "host", "--ipc=host",
               "--memory", PERF_ANALYZER_MEMORY_LIMIT, "--memory-swap", PERF_ANALYZER_MEMORY_LIMIT,
               "--user", f"{os.getuid()}:{os.getgid()}",
               "--volume", f"{input_data_folder}:{CONTAINER_INPUT_DATA_FOLDER}:ro",
               "--volume", f"{report_path.parent}:{CONTAINER_REPORT_FOLDER}", "--entrypoint", "perf_analyzer",
               SDK_IMAGE, "-m", model_name, "-u", GRPC_ADDRESS, "-i", "grpc", "-b", str(batch_size),
               "--concurrency-range", str(concurrency), "--input-data", CONTAINER_INPUT_DATA_FOLDER,
               *shape_arguments, *transport_arguments, "-f", f"{CONTAINER_REPORT_FOLDER}/{report_path.name}",
               "--verbose-csv", "--collect-metrics", "--metrics-url", METRICS_ADDRESS]
    # Run it with a longer measurement window after every failure, keeping the end of the output of the last one
    log_texts = []
    error = ""
    for attempt_index, window_milliseconds in enumerate(PERF_ANALYZER_WINDOWS_MILLISECONDS):
        report_path.unlink(missing_ok=True)
        try:
            completed = subprocess.run(args=command + ["--measurement-interval", str(window_milliseconds)],
                                       capture_output=True, text=True, check=False,
                                       timeout=PERF_ANALYZER_TIMEOUT_SECONDS)
            output = completed.stdout + completed.stderr
            error = (f"return code {completed.returncode}: {output.strip()[-400:]}"
                     if completed.returncode != 0 or not report_path.is_file() else "")
        except subprocess.TimeoutExpired:
            # The container keeps running once its client is killed, so it is removed by name
            subprocess.run(args=["docker", "rm", "--force", PERF_ANALYZER_CONTAINER_NAME], capture_output=True,
                           check=False)
            output = ""
            error = f"timed out after {PERF_ANALYZER_TIMEOUT_SECONDS} s"

        # Output of every run kept in the log, the measurement read from the first run writing its report
        log_texts.append(f"=== Run {attempt_index + 1}, windows of {window_milliseconds} ms : "
                         f"{error or 'report written'}\n{output}")
        (report_path.parent / f"{report_path.stem}.log").write_text("\n".join(log_texts))
        if not error:
            return parse_perf_analyzer_report(report_path=report_path), ""
    return None, f"{len(PERF_ANALYZER_WINDOWS_MILLISECONDS)} runs failed, the last one with {error}"


def parse_triton_gpu_metrics(metrics_text: str) -> Dict[str, float]:
    """
    Reads the GPU use and power of the Prometheus metrics of the server (port 8002), sampled during a gRPC load as
    perf_analyzer samples them during its loads.

    Args:
        metrics_text (str): The text of the metrics endpoint.

    Returns:
        Dict[str, float]: 'gpu_utilization_percent' and 'gpu_power_watts', averaged over the GPUs.

    Raises:
        ValueError: If a metric is missing from the text.
    """
    metric_values = {"nv_gpu_utilization": [], "nv_gpu_power_usage": []}
    for line in metrics_text.splitlines():
        metric_name = line.split("{", maxsplit=1)[0]
        if metric_name in metric_values:
            metric_values[metric_name].append(float(line.rsplit(" ", maxsplit=1)[1]))
    missing_metrics = [metric_name for metric_name, values in metric_values.items() if not values]
    if missing_metrics:
        raise ValueError(f"Expected the metrics {list(metric_values)} from the server, missing {missing_metrics}.")
    return {"gpu_utilization_percent": 100.0 * float(np.mean(metric_values["nv_gpu_utilization"])),
            "gpu_power_watts": float(np.mean(metric_values["nv_gpu_power_usage"]))}


def read_inference_statistics(model_name: str) -> Dict[str, Tuple[int, int]]:
    """
    Reads the cumulative inference statistics of a served model (the statistics extension of Triton), whose
    differences over a gRPC load give its average queue and compute times.

    Args:
        model_name (str): The Triton model name.

    Returns:
        Dict[str, Tuple[int, int]]: The (count, nanoseconds) of 'queue' and of every part of COMPUTE_STATISTICS.
    """
    statistics = send_server_request(server=BENCHMARK_SERVER, path=f"v2/models/{model_name}/stats", body=None,
                                     timeout_seconds=30)
    inference_statistics = statistics["model_stats"][0]["inference_stats"]
    return {name: (int(inference_statistics[name]["count"]), int(inference_statistics[name]["ns"]))
            for name in ["queue"] + COMPUTE_STATISTICS}


def average_server_microseconds(statistics_before: Dict[str, Tuple[int, int]],
                                statistics_after: Dict[str, Tuple[int, int]]) -> Dict[str, float]:
    """
    Turns two readings of the inference statistics, taken at the start and at the end of a gRPC load, into the
    average queue time and compute time (input, inference and output) of the requests executed between them, in
    microseconds, the columns perf_analyzer gives for its loads.

    Args:
        statistics_before (Dict[str, Tuple[int, int]]): The statistics at the start of the window.
        statistics_after (Dict[str, Tuple[int, int]]): The statistics at its end.

    Returns:
        Dict[str, float]: 'server_queue_microseconds' and 'server_compute_microseconds'.

    Raises:
        ValueError: If the server executed no request between the two readings.
    """
    average_microseconds = {}
    for name in ["queue"] + COMPUTE_STATISTICS:
        number_requests = statistics_after[name][0] - statistics_before[name][0]
        if number_requests <= 0:
            raise ValueError(f"Expected requests executed by the server during the load, got {number_requests} for "
                             f"'{name}'.")
        average_microseconds[name] = (statistics_after[name][1] - statistics_before[name][1]) / number_requests / 1e3
    return {"server_queue_microseconds": average_microseconds["queue"],
            "server_compute_microseconds": sum(average_microseconds[name] for name in COMPUTE_STATISTICS)}


def send_requests_until(model_name: str, inputs: Dict[str, np.ndarray], output_names: List[str],
                        ready_queue: Any, start_event: Any, window_times: Any, results_queue: Any) -> None:
    """
    One of the concurrent clients of a gRPC load (run_client_load), run in a process of its own like a client of
    another pod, so that the clients do not share one Python interpreter lock: once its gRPC client is created it
    says so, waits for the window to be set, then sends the same request again and again, one at a time, built and
    sent like run_inference.py sends it, keeping the latency of every request ending within the window, and gives
    back its latencies and its error, if any.

    Args:
        model_name (str): The Triton model name.
        inputs (Dict[str, np.ndarray]): The batched request, keyed by input name.
        output_names (List[str]): The served output names.
        ready_queue (Any): The multiprocessing queue told when the client is ready.
        start_event (Any): The multiprocessing event set once the window is set.
        window_times (Any): The shared array of the time.monotonic() values starting and ending the window.
        results_queue (Any): The multiprocessing queue receiving the 'latencies_seconds' and 'error' of the client.
    """
    latencies_seconds, error = [], ""
    client = triton_client.InferenceServerClient(url=GRPC_ADDRESS)
    try:
        ready_queue.put(True)
        start_event.wait()
        window_start_time, window_end_time = window_times[0], window_times[1]
        while time.monotonic() < window_end_time:
            request_start = time.monotonic()
            run_triton_inference(inputs=inputs, client=client, model_name=model_name, output_names=output_names,
                                 shared_memory_handle=None, region_name=None)
            request_end = time.monotonic()
            if window_start_time <= request_end <= window_end_time:
                latencies_seconds.append(request_end - request_start)
    except Exception as exception:
        # The error is given back to run_client_load, which records the load as failed
        error = f"{type(exception).__name__}: {exception}"
    finally:
        client.close()
    results_queue.put({"latencies_seconds": latencies_seconds, "error": error})


def run_client_load(model_name: str, request: Dict[str, np.ndarray], batch_size: int, concurrency: int,
                    raw_path: Path) -> Tuple[Optional[Dict[str, float]], str]:
    """
    Measures one gRPC load on a served model with the tritonclient client of the repository instead of perf_analyzer
    (see PERF_ANALYZER_MEMORY_LIMIT): concurrency client processes send batch_size samples per request, the inputs
    inside the requests, during CLIENT_WARMUP_SECONDS of untimed requests then a window of CLIENT_WINDOW_SECONDS,
    started once every client is ready. The requests ending within the window give the latencies and the throughput,
    the statistics of the server their queue and compute times, and its metrics, sampled during the window, the GPU
    use and power. The latencies are saved as JSON.

    Args:
        model_name (str): The Triton model name.
        request (Dict[str, np.ndarray]): The batch-1 request, keyed by input name.
        batch_size (int): The samples per request.
        concurrency (int): The requests in flight, one client process each.
        raw_path (Path): The JSON file receiving the measure and the latencies.

    Returns:
        Tuple[Optional[Dict[str, float]], str]: The measure, with the keys of parse_perf_analyzer_report, None when
            a client failed or no request ended within the window, and the error (empty otherwise).

    Raises:
        RuntimeError: If the clients are not ready or do not answer within CLIENT_START_TIMEOUT_SECONDS.
    """
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    client = triton_client.InferenceServerClient(url=GRPC_ADDRESS)
    output_names = read_served_output_names(client=client, model_name=model_name)
    client.close()
    batched_request = repeat_request(request=request, batch_size=batch_size)

    # One client process per request in flight, the window set once all of them are ready
    context = multiprocessing.get_context(method="spawn")
    ready_queue, results_queue, start_event = context.Queue(), context.Queue(), context.Event()
    window_times = context.Array("d", 2)
    processes = [context.Process(target=send_requests_until, kwargs={
        "model_name": model_name, "inputs": batched_request, "output_names": output_names,
        "ready_queue": ready_queue, "start_event": start_event, "window_times": window_times,
        "results_queue": results_queue}) for _ in range(concurrency)]
    for process in processes:
        process.start()
    for _ in processes:
        ready_queue.get(timeout=CLIENT_START_TIMEOUT_SECONDS)
    window_start_time = time.monotonic() + CLIENT_WARMUP_SECONDS
    window_end_time = window_start_time + CLIENT_WINDOW_SECONDS
    window_times[0], window_times[1] = window_start_time, window_end_time
    start_event.set()

    # Statistics of the server at both ends of the window, and its GPU metrics sampled in between
    time.sleep(max(0.0, window_start_time - time.monotonic()))
    statistics_before = read_inference_statistics(model_name=model_name)
    metric_samples = []
    while time.monotonic() < window_end_time:
        with urllib.request.urlopen(f"http://{METRICS_ADDRESS}", timeout=5) as response:
            metric_samples.append(parse_triton_gpu_metrics(metrics_text=response.read().decode()))
        time.sleep(min(METRICS_SAMPLE_SECONDS, max(0.0, window_end_time - time.monotonic())))
    statistics_after = read_inference_statistics(model_name=model_name)

    # Latencies of every client, read before the processes are joined so that the queue is emptied
    client_results = [results_queue.get(timeout=CLIENT_START_TIMEOUT_SECONDS) for _ in processes]
    for process in processes:
        process.join()
    latencies_seconds = [latency for client_result in client_results for latency in client_result["latencies_seconds"]]
    errors = [client_result["error"] for client_result in client_results if client_result["error"]]
    if errors:
        return None, f"{len(errors)} clients failed, the first with {errors[0]}"
    if not latencies_seconds:
        return None, f"no request ended within the window of {CLIENT_WINDOW_SECONDS} s"

    # Latencies and throughput of the requests ending within the window
    latencies_milliseconds = np.array(latencies_seconds) * 1000.0
    measure = {"p50_milliseconds": float(np.percentile(latencies_milliseconds, 50)),
               "p99_milliseconds": float(np.percentile(latencies_milliseconds, 99)),
               "average_milliseconds": float(latencies_milliseconds.mean()),
               "inferences_per_second": len(latencies_seconds) * batch_size / CLIENT_WINDOW_SECONDS,
               **average_server_microseconds(statistics_before=statistics_before, statistics_after=statistics_after),
               **{metric_name: float(np.mean([sample[metric_name] for sample in metric_samples]))
                  for metric_name in ["gpu_utilization_percent", "gpu_power_watts"]}}
    raw_path.write_text(json.dumps({"measure": measure, "latencies_milliseconds": latencies_milliseconds.tolist()}))
    return measure, ""


def plan_served_loads(load_groups: List[Dict[str, Any]], precision_setting: str,
                      common_columns: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Lists the loads measured on one served row from the load groups of the sweep, each group giving the precision
    settings it applies to and the server settings, transports, batch sizes and concurrencies it crosses, e.g. the
    whole grid with the exported configuration for every row, and the other server settings for one row only. A load
    listed by two groups is measured once.

    Args:
        load_groups (List[Dict[str, Any]]): The groups, with 'precision_settings', 'server_settings' (dynamic
            batching, instance count), 'transports', 'batch_sizes' and 'concurrencies'.
        precision_setting (str): The precision setting of the row.
        common_columns (Dict[str, Any]): The 'model' and 'row' columns of the row.

    Returns:
        List[Dict[str, Any]]: The planned results (KEY_COLUMNS) of the row, in the order of the groups.
    """
    planned_loads, planned_keys = [], set()
    for load_group in load_groups:
        if precision_setting not in load_group["precision_settings"]:
            continue
        for dynamic_batching, instance_count in load_group["server_settings"]:
            for transport in load_group["transports"]:
                for batch_size in load_group["batch_sizes"]:
                    for concurrency in load_group["concurrencies"]:
                        planned_load = {**common_columns, "kind": "load", "dynamic_batching": dynamic_batching,
                                        "instance_count": instance_count, "transport": transport,
                                        "batch_size": batch_size, "concurrency": concurrency}
                        if build_result_key(result=planned_load) not in planned_keys:
                            planned_keys.add(build_result_key(result=planned_load))
                            planned_loads.append(planned_load)
    return planned_loads


def measure_in_process_rows(model: Dict[str, Any], request: Dict[str, np.ndarray], benchmark_folder: Path,
                            precision_settings: List[str], batch_sizes: List[int], number_iterations: int,
                            number_warmup_iterations: int, completed_keys: set, results_path: Path) -> None:
    """
    Times the in-process rows of a model (rows 1, 2 and 9 of the plan), the eager model and its TorchScript exports in
    FP32 and FP16 (list_in_process_rows), each in a child process, and records one line per batch size.

    Args:
        model (Dict[str, Any]): The benchmarked model.
        request (Dict[str, np.ndarray]): Its batch-1 request.
        benchmark_folder (Path): The folder of the outputs of the benchmark.
        precision_settings (List[str]): The benchmarked precision settings, whose repositories hold the TorchScript
            files.
        batch_sizes (List[int]): The batch sizes timed.
        number_iterations (int): The timed calls per batch size.
        number_warmup_iterations (int): The untimed calls made first.
        completed_keys (set): The keys already in results.csv, skipped.
        results_path (Path): The results file.
    """
    frames_per_sample = count_frames_per_sample(request=request)
    served_model_name = read_served_model_name(experiment_folder=model["experiment_folder"])
    in_process_rows = list_in_process_rows(model_repositories=benchmark_folder / "model_repositories",
                                           served_model_name=served_model_name, precision_settings=precision_settings)
    for row_name, row_torchscript_path in in_process_rows:
        planned_results = [{"model": model["name"], "row": row_name, "kind": "load", "dynamic_batching": "",
                            "instance_count": "", "transport": "in_process", "batch_size": batch_size,
                            "concurrency": 1} for batch_size in batch_sizes]
        if all(build_result_key(result=result) in completed_keys for result in planned_results):
            continue

        # One child process per row, after the machine is idle
        other_processes = wait_for_idle_machine()
        output_path = benchmark_folder / "in_process" / f"{model['name']}_{row_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print_color(color="blue", output=f"{model['name']} : {row_name}")
        measure_start = time.time()
        run_in_child_process(target=measure_in_process_row, keyword_arguments={
            "experiment_folder": model["experiment_folder"], "iteration": model["iteration"],
            "torchscript_path": row_torchscript_path, "request": request, "batch_sizes": batch_sizes,
            "number_iterations": number_iterations, "number_warmup_iterations": number_warmup_iterations,
            "output_path": output_path})
        # One result line per batch size, the images per second counting every frame of a sequence
        row_measures = json.loads(output_path.read_text())
        for planned_result, measure in zip(planned_results, row_measures["measures"]):
            append_result(results_path=results_path, result={
                **planned_result, "status": "ok",
                **{key: value for key, value in measure.items() if key != "batch_size"},
                "images_per_second": measure["inferences_per_second"] * frames_per_sample,
                "memory_gigabytes": row_measures["memory_gigabytes"],
                "allocator_peak_gigabytes": row_measures["allocator_peak_gigabytes"],
                "other_gpu_processes": ";".join(other_processes), "duration_seconds": time.time() - measure_start,
                "raw_file": str(output_path)})


def measure_served_model(model: Dict[str, Any], request: Dict[str, np.ndarray], precision_setting: str,
                         export_format: str, benchmark_folder: Path, load_groups: List[Dict[str, Any]],
                         first_request_batch_sizes: List[int], maximum_batch_size: int, completed_keys: set,
                         results_path: Path) -> None:
    """
    Measures one served model of a running server (one row of the plan for one model): the first requests after its
    first load with an empty engine cache, then for every server setting of its loads (plan_served_loads) its memory
    and every load, through system shared memory with perf_analyzer and over gRPC with the tritonclient client, each
    measurement recorded in results.csv as soon as it ends. The model is unloaded at the end.

    Args:
        model (Dict[str, Any]): The benchmarked model.
        request (Dict[str, np.ndarray]): Its batch-1 request.
        precision_setting (str): The precision setting of the server.
        export_format (str): Either 'torchscript' or 'onnx'.
        benchmark_folder (Path): The folder of the outputs of the benchmark.
        load_groups (List[Dict[str, Any]]): The load groups of the sweep (plan_served_loads).
        first_request_batch_sizes (List[int]): The batch sizes of the first requests, in increasing order.
        maximum_batch_size (int): The largest batch size of the model.
        completed_keys (set): The keys already in results.csv, skipped.
        results_path (Path): The results file.
    """
    row_name = get_row_name(precision_setting=precision_setting, export_format=export_format)
    model_name = f"{read_served_model_name(experiment_folder=model['experiment_folder'])}_{export_format}"
    frames_per_sample = count_frames_per_sample(request=request)
    shapes = {input_name: list(input_batch.shape[1:]) for input_name, input_batch in request.items()
              if input_batch.ndim == 5}
    common_columns = {"model": model["name"], "row": row_name}
    first_request_results = [{**common_columns, "kind": "first_request", "dynamic_batching": "", "instance_count": "",
                              "transport": "grpc", "batch_size": batch_size, "concurrency": 1}
                             for batch_size in first_request_batch_sizes]
    load_results = plan_served_loads(load_groups=load_groups, precision_setting=precision_setting,
                                     common_columns=common_columns)
    server_settings = list(dict.fromkeys((result["dynamic_batching"], result["instance_count"])
                                         for result in load_results))
    if all(build_result_key(result=result) in completed_keys for result in first_request_results + load_results):
        return

    # First load with an empty engine cache, its first requests timed (the engine builds of TensorRT)
    engine_cache_folder = benchmark_folder / "tensorrt_engine_cache" / precision_setting
    for cached_path in engine_cache_folder.iterdir():
        if cached_path.is_dir():
            shutil.rmtree(cached_path)
        else:
            cached_path.unlink()
    # Memory available before the model loads, so that the memory the served model takes is the difference
    available_without_model_gigabytes = read_available_gigabytes()
    load_served_model(server=BENCHMARK_SERVER, model_name=model_name, configuration=None)
    if not all(build_result_key(result=result) in completed_keys for result in first_request_results):
        other_processes = wait_for_idle_machine()
        first_request_seconds = time_first_requests(model_name=model_name, request=request,
                                                    batch_sizes=first_request_batch_sizes)
        for result in first_request_results:
            append_result(results_path=results_path, result={
                **result, "status": "ok", "first_request_seconds": first_request_seconds[result["batch_size"]],
                "other_gpu_processes": ";".join(other_processes)})
        print_color(color="blue", output=f"{model_name} ({precision_setting}) first requests : " +
                                         ", ".join(f"batch {batch_size} {seconds:.2f} s"
                                                   for batch_size, seconds in first_request_seconds.items()), indent=1)
    # Configuration exported with the model, the base of every server setting below
    base_configuration = send_server_request(server=BENCHMARK_SERVER, path=f"v2/models/{model_name}/config", body=None,
                                             timeout_seconds=30)

    for dynamic_batching, instance_count in server_settings:
        setting_results = [result for result in load_results if result["dynamic_batching"] == dynamic_batching
                           and result["instance_count"] == instance_count]
        if all(build_result_key(result=result) in completed_keys for result in setting_results):
            continue

        # Server setting given at load time, warmed up at the smallest and largest batch sizes, its memory read after
        load_served_model(server=BENCHMARK_SERVER, model_name=model_name, configuration=build_server_configuration(
            base_configuration=base_configuration, dynamic_batching=dynamic_batching, instance_count=instance_count))
        warm_up_served_model(model_name=model_name, request=request, maximum_batch_size=maximum_batch_size)
        memory_gigabytes = available_without_model_gigabytes - read_available_gigabytes()
        print_color(color="blue", output=f"{model_name} ({precision_setting}), dynamic batching {dynamic_batching}, "
                                         f"{instance_count} instance(s) : {memory_gigabytes:.2f} GB", indent=1)

        for result in setting_results:
            if build_result_key(result=result) in completed_keys:
                continue
            other_processes = wait_for_idle_machine()
            raw_stem = (benchmark_folder / "loads" / model["name"] /
                        f"{row_name}_batching-{dynamic_batching}_instances-{instance_count}_{result['transport']}_"
                        f"batch-{result['batch_size']}_concurrency-{result['concurrency']}")
            measure_start = time.time()

            # gRPC loads timed by the tritonclient client, shared memory loads by perf_analyzer
            if result["transport"] == "grpc":
                load_client, raw_path = "tritonclient", raw_stem.with_suffix(".json")
                measure, error = run_client_load(model_name=model_name, request=request,
                                                 batch_size=result["batch_size"], concurrency=result["concurrency"],
                                                 raw_path=raw_path)
            else:
                load_client, raw_path = "perf_analyzer", raw_stem.with_suffix(".csv")
                measure, error = run_perf_analyzer(model_name=model_name, batch_size=result["batch_size"],
                                                   concurrency=result["concurrency"], transport=result["transport"],
                                                   input_data_folder=benchmark_folder / "inputs" / model["name"] /
                                                   "perf_analyzer_data", shapes=shapes, report_path=raw_path)
            # Result line of the load, also written when it failed, with the error
            measured_columns = {} if measure is None else {
                **measure, "images_per_second": measure["inferences_per_second"] * frames_per_sample}
            append_result(results_path=results_path, result={
                **result, "status": "ok" if measure is not None else "failed", "load_client": load_client,
                **measured_columns, "memory_gigabytes": memory_gigabytes,
                "other_gpu_processes": ";".join(other_processes), "duration_seconds": time.time() - measure_start,
                "raw_file": str(raw_path), "error": error})
            measure_text = "failed" if measure is None else (f"p50 {measure['p50_milliseconds']:.2f} ms, "
                                                             f"{measured_columns['images_per_second']:.1f} Images "
                                                             f"Per Second")
            print_color(color="green" if measure is not None else "red",
                        output=f"{result['transport']}, batch {result['batch_size']}, concurrency "
                               f"{result['concurrency']} : {measure_text}", indent=2)
    unload_served_model(server=BENCHMARK_SERVER, model_name=model_name)


def run_serving_benchmark(models: List[Dict[str, Any]], benchmark_folder: Path,
                          precision_settings: Dict[str, Dict[str, Any]], load_groups: List[Dict[str, Any]],
                          batch_sizes: List[int], maximum_batch_size: int, number_in_process_iterations: int,
                          number_in_process_warmup_iterations: int, run_exports: bool, run_in_process: bool,
                          run_triton: bool) -> None:
    """
    Runs the phases of the benchmark in order, each skipping what results.csv already holds: the inputs of every
    model, the exports into one repository per precision setting, the in-process rows, then one Triton server per
    precision setting measuring its models one at a time.

    Args:
        models (List[Dict[str, Any]]): The benchmarked models, with 'name', 'experiment_folder', 'iteration',
            'test_data_folder' and 'sequence_length'.
        benchmark_folder (Path): The folder of every output, outside the repository.
        precision_settings (Dict[str, Dict[str, Any]]): The 'tensorrt_precision' and server 'environment' of every
            precision setting, keyed by its name, the precision of its TorchScript network following from its name
            (get_torchscript_precision).
        load_groups (List[Dict[str, Any]]): The loads of the served rows (plan_served_loads).
        batch_sizes (List[int]): The batch sizes of the in-process rows and of the first requests of the served rows.
        maximum_batch_size (int): The largest batch size of the exported models.
        number_in_process_iterations (int): The timed calls of the in-process rows per batch size.
        number_in_process_warmup_iterations (int): Their untimed calls made first.
        run_exports (bool): Whether to export the models whose repository folders are missing.
        run_in_process (bool): Whether to measure the in-process rows.
        run_triton (bool): Whether to measure the served rows.
    """
    results_path = benchmark_folder / "results.csv"
    benchmark_folder.mkdir(parents=True, exist_ok=True)
    print_color(color="blue", output=f"Serving Benchmark : {format_path_link(path=results_path)}", add_separators=True)

    # Inputs of every model, and its Triton name and outputs read once from the export of the first setting
    requests = {}
    for model in models:
        requests[model["name"]] = prepare_benchmark_inputs(model=model,
                                                           inputs_folder=benchmark_folder / "inputs" / model["name"])

    # Exports of every model into the repository of every precision setting, skipped when already there
    if run_exports:
        for precision_setting, precision_values in precision_settings.items():
            repository = benchmark_folder / "model_repositories" / precision_setting
            formats = get_repository_formats(precision_setting=precision_setting)
            for model in models:
                served_model_name = read_served_model_name(experiment_folder=model["experiment_folder"])
                if all((repository / f"{served_model_name}_{export_format}" / "1").is_dir()
                       for export_format in formats):
                    continue
                print_color(color="blue", output=f"Export {model['name']} : {precision_setting}")
                run_in_child_process(target=export_precision_setting, keyword_arguments={
                    "experiment_folder": model["experiment_folder"], "iteration": model["iteration"],
                    "repository": repository, "formats": formats,
                    "tensorrt_precision": precision_values["tensorrt_precision"],
                    "torchscript_precision": get_torchscript_precision(precision_setting=precision_setting),
                    "maximum_batch_size": maximum_batch_size})

    # In-process rows of every model (eager and TorchScript in FP32 and FP16), each timed in a child process
    if run_in_process:
        for model in models:
            measure_in_process_rows(model=model, request=requests[model["name"]], benchmark_folder=benchmark_folder,
                                    precision_settings=list(precision_settings), batch_sizes=batch_sizes,
                                    number_iterations=number_in_process_iterations,
                                    number_warmup_iterations=number_in_process_warmup_iterations,
                                    completed_keys=read_completed_keys(results_path=results_path),
                                    results_path=results_path)

    if not run_triton:
        return
    for precision_setting, precision_values in precision_settings.items():
        log_path = benchmark_folder / "logs" / f"triton_{precision_setting}_{datetime.now():%Y%m%d_%H%M%S}.log"
        print_color(color="blue", output=f"Triton : {precision_setting}", add_separators=True)
        start_triton_server(server=BENCHMARK_SERVER,
                            repository=benchmark_folder / "model_repositories" / precision_setting,
                            engine_cache_folder=benchmark_folder / "tensorrt_engine_cache" / precision_setting,
                            environment=precision_values["environment"])
        # The server log is kept and the container removed whatever ends the measurements
        try:
            for model in models:
                for export_format in get_repository_formats(precision_setting=precision_setting):
                    measure_served_model(model=model, request=requests[model["name"]],
                                         precision_setting=precision_setting, export_format=export_format,
                                         benchmark_folder=benchmark_folder, load_groups=load_groups,
                                         first_request_batch_sizes=batch_sizes, maximum_batch_size=maximum_batch_size,
                                         completed_keys=read_completed_keys(results_path=results_path),
                                         results_path=results_path)
        finally:
            stop_triton_server(server=BENCHMARK_SERVER, log_path=log_path)


def build_benchmarked_models(runs_folder: Path, data_folder: Path) -> List[Dict[str, Any]]:
    """
    Places the models of BENCHMARKED_MODELS on this machine, their run folders under the runs folder and their test
    splits under the data folder given on the command line, so that the plan of the sweep stays in the repository and
    only the folders change from one machine to another; both drivers of the benchmark measure these models.

    Args:
        runs_folder (Path): The folder holding the trained runs, outside the repository.
        data_folder (Path): The folder holding the data sets of their test splits.

    Returns:
        List[Dict[str, Any]]: The models with their 'name', 'iteration', 'sequence_length', 'experiment_folder' and
            'test_data_folder'.
    """
    return [{"name": model["name"], "iteration": model["iteration"], "sequence_length": model["sequence_length"],
             "experiment_folder": runs_folder / model["run"], "test_data_folder": data_folder / model["test_split"]}
            for model in BENCHMARKED_MODELS]


def add_benchmark_folder_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Adds the folders of the machine to the command line of a driver of the benchmark, the only settings that leave the
    code, so that both drivers read the same model repositories, runs and test splits.

    Args:
        parser (argparse.ArgumentParser): The parser of the driver, extended.
    """
    parser.add_argument("--benchmark_folder", type=Path, required=True,
                        help="Folder of the benchmark, outside the repository: the model repositories, the TensorRT "
                             "engine caches, results.csv and test_metrics.csv (required).")
    parser.add_argument("--runs_folder", type=Path, required=True,
                        help="Folder the run folders of BENCHMARKED_MODELS are relative to (required).")
    parser.add_argument("--data_folder", type=Path, required=True,
                        help="Folder the test splits of BENCHMARKED_MODELS are relative to (required).")


def parse_benchmark_arguments(arguments: List[str]) -> argparse.Namespace:
    """
    Reads the folders of the machine and the phases of the sweep from the command line of the script, the plan of the
    sweep staying in the constants of this module: the three folders are required, every phase runs by default.
    Launched without any argument, the script prints the help of every argument and stops. A '~' left by the shell at
    the start of a path ('--benchmark_folder=~/Data/...') is expanded to the home folder.

    Args:
        arguments (List[str]): The command line arguments, without the script name.

    Returns:
        argparse.Namespace: The benchmark_folder, runs_folder, data_folder and phases.

    Raises:
        SystemExit: After printing the help when no argument is given, or the usage and the error when a folder is
            missing or a phase is unknown.
    """
    # The description and the example keep their line breaks, argparse wrapping the help of every argument
    parser = argparse.ArgumentParser(
        prog="uv run python -m scripts.deployment.benchmark_serving_formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Exports the runs of BENCHMARKED_MODELS once per precision setting, times them in the process,\n"
                    "then served by Triton, one line per measurement in <benchmark_folder>/results.csv.",
        epilog="Example, the whole sweep, resumed where a stopped one stopped:\n"
               "  uv run python -m scripts.deployment.benchmark_serving_formats \\\n"
               "      --benchmark_folder ~/Data/serving_benchmark --runs_folder ~/Data/tutorial_runs \\\n"
               "      --data_folder ~/Data")
    # The folders of this machine, then the phases run
    add_benchmark_folder_arguments(parser=parser)
    parser.add_argument("--phases", nargs="+", choices=BENCHMARK_PHASES, default=list(BENCHMARK_PHASES),
                        help="Phases run, each skipping what results.csv or the model repositories already hold "
                             f"(default: {' '.join(BENCHMARK_PHASES)}).")

    # Without any argument, the script describes its arguments instead of failing on the missing folders
    if not arguments:
        parser.print_help()
        parser.exit(status=0)
    parsed_arguments = parser.parse_args(args=arguments)

    # A '~' the shell leaves after '=' points to the home folder, as in the other scripts of the repository
    for path_name in ["benchmark_folder", "runs_folder", "data_folder"]:
        setattr(parsed_arguments, path_name, getattr(parsed_arguments, path_name).expanduser())
    return parsed_arguments


if __name__ == "__main__":
    # Folders of this machine and phases read from the command line, the help printed when none is given
    benchmark_arguments = parse_benchmark_arguments(arguments=sys.argv[1:])

    # The plan of the sweep from the constants of this module, its models placed in the folders of this machine
    run_serving_benchmark(models=build_benchmarked_models(runs_folder=benchmark_arguments.runs_folder,
                                                          data_folder=benchmark_arguments.data_folder),
                          benchmark_folder=benchmark_arguments.benchmark_folder, precision_settings=PRECISION_SETTINGS,
                          load_groups=LOAD_GROUPS, batch_sizes=BATCH_SIZES, maximum_batch_size=MAXIMUM_BATCH_SIZE,
                          number_in_process_iterations=IN_PROCESS_ITERATIONS,
                          number_in_process_warmup_iterations=IN_PROCESS_WARMUP_ITERATIONS,
                          run_exports="exports" in benchmark_arguments.phases,
                          run_in_process="in_process" in benchmark_arguments.phases,
                          run_triton="triton" in benchmark_arguments.phases)
