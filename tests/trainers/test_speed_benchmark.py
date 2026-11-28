"""
Tests of the 'benchmark_speed' diagnosis mode: the helpers of vision_training/trainers/speed_benchmark.py (tiling of a
loader batch, search of the largest batch size, out of memory errors, memory cap) and BaseTrainer.benchmark_speed of
vision_training/trainers/base_trainer.py, run on the dummy task (training step and test evaluation only, the task
having no export) and on the synthetic classification task (TorchScript and ONNX Runtime formats too), checking that
the benchmark exits without training nor run folder and prints its numbered progress lines and its report, plus the
tables and notes of the report and the capture of the native error output of the probes.
"""

import os
import subprocess
import onnxruntime
import pytest
import torch
import yaml

from functools import partial
from io import StringIO
from pathlib import Path
from rich.console import Console
from typing import Any, Dict, List, Optional

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.data.synthetic.create_synthetic_dataset import create_synthetic_dataset
from vision_training.trainers import base_trainer, speed_benchmark
from vision_training.trainers.classification_trainer import ClassificationTrainer
from vision_training.trainers.sequence_comparison_trainer import SequenceComparisonTrainer
from tests.fixtures.dummy_task import DummyDataset, DummyTrainer, build_dummy_configurations

SYNTHETIC_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" / "classification" /
                                "configuration_synthetic_shapes.yaml")

# Image size of the classification benchmark test, small enough for a quick export and timing
SMOKE_IMAGE_SIZE = 64

# Largest batch size of the searches of the tests, so that they end after a few probes
TEST_MAXIMUM_BATCH_SIZE = 16

# Mebibytes, for the memory cap test
BYTES_PER_MEBIBYTE = 2 ** 20

requires_cuda = pytest.mark.skipif(condition=not torch.cuda.is_available(),
                                   reason="the speed benchmark times CUDA GPUs")


def fits_below(batch_size: int, limit: int, tried_batch_sizes: List[int]) -> bool:
    """
    Stands for the probe passes of a batch size in the tests of find_largest_batch_size: a batch fits when it is not
    larger than the limit, every tried batch size being recorded.

    Args:
        batch_size (int): The tried batch size.
        limit (int): The largest batch size that fits.
        tried_batch_sizes (List[int]): The tried batch sizes, appended in place.

    Returns:
        bool: True when the batch size is at most the limit.
    """
    tried_batch_sizes.append(batch_size)
    return batch_size <= limit


def shorten_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Shortens the timing of the speed benchmark and caps its searches at TEST_MAXIMUM_BATCH_SIZE for the tests, the
    memory margins set to 0 since the tested models take a few megabytes, so that the benchmark runs on a GPU shared
    with other trainings.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the constants.
    """
    monkeypatch.setattr(target=speed_benchmark, name="WARM_UP_SECONDS", value=0.01)
    monkeypatch.setattr(target=speed_benchmark, name="MINIMUM_WARM_UP_ITERATIONS", value=1)
    monkeypatch.setattr(target=speed_benchmark, name="MEASURED_SECONDS", value=0.05)
    monkeypatch.setattr(target=speed_benchmark, name="MAXIMUM_BENCHMARK_BATCH_SIZE", value=TEST_MAXIMUM_BATCH_SIZE)
    monkeypatch.setattr(target=base_trainer, name="MAXIMUM_BENCHMARK_BATCH_SIZE", value=TEST_MAXIMUM_BATCH_SIZE)
    monkeypatch.setattr(target=speed_benchmark, name="INTEGRATED_GPU_MEMORY_MARGIN_BYTES", value=0)
    monkeypatch.setattr(target=speed_benchmark, name="DISCRETE_GPU_MEMORY_MARGIN_BYTES", value=0)


def build_benchmarking_dummy_trainer(temporary_directory: Path, benchmark_speed: bool) -> DummyTrainer:
    """
    Builds the trainer of the dummy task with batches of 2 samples, the speed benchmark on or off.

    Args:
        temporary_directory (Path): The pytest temporary directory holding the experiments folder.
        benchmark_speed (bool): Whether the speed benchmark runs (and the trainer exits) at construction.

    Returns:
        DummyTrainer: The trainer, when the benchmark is off.
    """
    configuration_path = temporary_directory / "configuration.yaml"
    configuration_path.write_text(data="dummy: configuration\n")
    experiment_configuration, model_configuration = build_dummy_configurations(
        experiment_folder=temporary_directory / "experiments", number_iterations=2)
    experiment_configuration["diagnosis_modes"]["benchmark_speed"] = benchmark_speed
    datasets = {"train": DummyDataset(sample_identifiers=list(range(8)), seed=0),
                "validation": DummyDataset(sample_identifiers=list(range(4)), seed=1),
                "test": DummyDataset(sample_identifiers=list(range(4)), seed=2)}
    return DummyTrainer(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                        configuration_path=configuration_path, datasets=datasets)


def write_benchmark_configuration(temporary_directory: Path, dataset_folder: Path) -> Path:
    """
    Writes a copy of the synthetic shapes classification configuration with the speed benchmark on, on the generated
    dataset, with 64 pixel images and batches of 4 images.

    Args:
        temporary_directory (Path): The pytest temporary directory receiving the configuration and the experiments.
        dataset_folder (Path): The root folder of the generated synthetic dataset.

    Returns:
        Path: The written configuration file.
    """
    configuration = yaml.safe_load(stream=SYNTHETIC_CONFIGURATION_PATH.read_text())
    experiment_configuration = configuration["ExperimentConfiguration"]
    experiment_configuration.update({"experiment_folder": str(temporary_directory / "experiments"),
                                     "train_data_folder": str(dataset_folder / "train"),
                                     "validation_data_folder": str(dataset_folder / "validation"),
                                     "test_data_folder": str(dataset_folder / "test"), "batch_size": 4})
    experiment_configuration["diagnosis_modes"].update({"compute_receptive_field": False, "view_input_images": False,
                                                        "benchmark_speed": True})
    configuration["Data"]["image_settings"]["size"] = SMOKE_IMAGE_SIZE
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(dataset_folder / "label_mapping.json")
    configuration_path = temporary_directory / "configuration_benchmark.yaml"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    return configuration_path


def test_resize_batch_repeats_every_per_sample_value() -> None:
    """
    Verifies that resize_batch repeats the samples of a batch cyclically into a larger batch and keeps the first ones
    for a smaller batch, the per-sample tensors and lists alike, the tensors of the lists copied so that no two samples
    share one, and the values of another first dimension kept as they are; a batch of no sample is refused.
    """
    batch = {"images": torch.arange(end=3).view(3, 1, 1, 1).expand(3, 3, 2, 2).clone(),
             "labels": torch.tensor(data=[10, 11, 12]),
             "bounding_boxes": [torch.full(size=(sample_index + 1, 4), fill_value=float(sample_index))
                                for sample_index in range(3)],
             "class_weights": torch.ones(size=(5,)), "setting": "kept"}

    # A larger batch repeats the samples in their order
    resized_batch = speed_benchmark.resize_batch(batch=batch, batch_size=7)
    assert resized_batch["images"][:, 0, 0, 0].tolist() == [0, 1, 2, 0, 1, 2, 0]
    assert resized_batch["labels"].tolist() == [10, 11, 12, 10, 11, 12, 10]
    assert [boxes.shape[0] for boxes in resized_batch["bounding_boxes"]] == [1, 2, 3, 1, 2, 3, 1]
    assert resized_batch["bounding_boxes"][0] is not resized_batch["bounding_boxes"][3]
    assert resized_batch["bounding_boxes"][0] is not batch["bounding_boxes"][0]
    assert resized_batch["class_weights"] is batch["class_weights"] and resized_batch["setting"] == "kept"

    # A smaller batch keeps the first samples
    assert speed_benchmark.resize_batch(batch=batch, batch_size=2)["labels"].tolist() == [10, 11]
    with pytest.raises(expected_exception=ValueError, match="at least one sample"):
        speed_benchmark.resize_batch(batch=batch, batch_size=0)


@pytest.mark.parametrize(argnames=("limit", "starting_batch_size", "expected_largest", "expected_failing"),
                         argvalues=[(2048, 64, 2048, None), (5000, 64, 2048, None), (0, 64, 0, 1), (5, 64, 5, 6),
                                    (64, 64, 64, 68)])
def test_find_largest_batch_size(limit: int, starting_batch_size: int, expected_largest: int,
                                 expected_failing: Any) -> None:
    """
    Verifies the search of the largest batch size: it doubles from the configured batch size while the batch fits, up
    to 2048 at most, halves while it does not fit, then bisects until the gap between the largest batch size that fits
    and the smallest one that does not is at most 1/16 of the first (or a single sample).

    Args:
        limit (int): The largest batch size that fits.
        starting_batch_size (int): The configured batch size, tried first.
        expected_largest (int): The expected largest batch size found.
        expected_failing (Any): The expected smallest batch size found not to fit, None when the maximum fits.
    """
    tried_batch_sizes = []
    largest_fitting, smallest_failing = speed_benchmark.find_largest_batch_size(
        fits=partial(fits_below, limit=limit, tried_batch_sizes=tried_batch_sizes),
        starting_batch_size=starting_batch_size)
    assert (largest_fitting, smallest_failing) == (expected_largest, expected_failing)
    assert tried_batch_sizes[0] == starting_batch_size
    assert max(tried_batch_sizes) <= speed_benchmark.MAXIMUM_BENCHMARK_BATCH_SIZE


def test_find_largest_batch_size_stops_within_its_resolution() -> None:
    """
    Verifies that the bisection stops once the gap between the largest batch size that fits and the smallest one that
    does not is at most 1/16 of the first, the found batch size never exceeding the limit.
    """
    largest_fitting, smallest_failing = speed_benchmark.find_largest_batch_size(
        fits=partial(fits_below, limit=1000, tried_batch_sizes=[]), starting_batch_size=48)
    assert largest_fitting <= 1000 < smallest_failing
    assert smallest_failing - largest_fitting <= largest_fitting // speed_benchmark.SEARCH_RESOLUTION_DIVISOR


def test_out_of_memory_errors_are_recognised() -> None:
    """
    Verifies that the out of memory errors of PyTorch and of the arena of ONNX Runtime end a batch size of the search,
    and that any other error does not.
    """
    assert speed_benchmark.is_out_of_memory_error(error=torch.OutOfMemoryError("CUDA out of memory."))
    assert speed_benchmark.is_out_of_memory_error(error=RuntimeError("CUDA error: out of memory"))
    assert speed_benchmark.is_out_of_memory_error(
        error=RuntimeError("[ONNXRuntimeError] : 6 : Failed to allocate memory for requested buffer of size 1024"))
    assert not speed_benchmark.is_out_of_memory_error(error=ValueError("shapes do not match"))


def list_fake_gpu_processes(arguments: List[str], **keyword_arguments: Any) -> subprocess.CompletedProcess:
    """
    Stands for nvidia-smi in the test of list_other_gpu_processes: the benchmark process itself, the daemon of the
    Multi-Process Service, another training and a line without process identifier.

    Args:
        arguments (List[str]): The command, checked to be the query of the compute processes.
        **keyword_arguments (Any): The options of subprocess.run, unused.

    Returns:
        subprocess.CompletedProcess: The output of the query.
    """
    assert arguments[0] == "nvidia-smi" and arguments[1] == "--query-compute-apps=pid,process_name,used_memory"
    output_lines = [f"{os.getpid()}, /usr/bin/python3, 512", "184241, nvidia-cuda-mps-server, 61",
                    "1635003, /home/user/.venv/bin/python, 21201", "No running processes found"]
    return subprocess.CompletedProcess(args=arguments, returncode=0, stdout="\n".join(output_lines), stderr="")


def raise_missing_program(arguments: List[str], **keyword_arguments: Any) -> subprocess.CompletedProcess:
    """
    Stands for a machine without nvidia-smi in the test of list_other_gpu_processes.

    Args:
        arguments (List[str]): The command.
        **keyword_arguments (Any): The options of subprocess.run, unused.

    Raises:
        FileNotFoundError: Always, as subprocess.run does for a missing program.
    """
    raise FileNotFoundError(arguments[0])


def test_other_gpu_processes_are_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the processes sharing the GPU are listed from nvidia-smi without the benchmark process itself and the
    daemon of the Multi-Process Service, and that a machine without nvidia-smi gives None, which the report announces.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing subprocess.run.
    """
    monkeypatch.setattr(target=speed_benchmark.subprocess, name="run", value=list_fake_gpu_processes)
    assert speed_benchmark.list_other_gpu_processes() == ["1635003 (python, 21201 MiB)"]
    monkeypatch.setattr(target=speed_benchmark.subprocess, name="run", value=raise_missing_program)
    assert speed_benchmark.list_other_gpu_processes() is None


@requires_cuda
def test_memory_cap_refuses_a_budget_below_the_margin(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the speed benchmark refuses to start when less memory than the margin is free, naming the free
    memory and the margin, and leaves the allocator uncapped.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the margins.
    """
    device = torch.device("cuda")
    monkeypatch.setattr(target=speed_benchmark, name="INTEGRATED_GPU_MEMORY_MARGIN_BYTES", value=10 ** 15)
    monkeypatch.setattr(target=speed_benchmark, name="DISCRETE_GPU_MEMORY_MARGIN_BYTES", value=10 ** 15)
    with pytest.raises(expected_exception=RuntimeError, match="GB of memory are free"):
        with speed_benchmark.limit_process_memory(device=device):
            pass
    assert torch.cuda.get_per_process_memory_fraction(device=torch.cuda.current_device()) == 1.0


@requires_cuda
def test_memory_cap_stops_allocations_beyond_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the memory cap of the speed benchmark makes an allocation larger than the budget fail with the out of
    memory error of PyTorch, an allocation within it succeeding, and that the cap is lifted at the end.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the margins.
    """
    device = torch.device("cuda")
    free_bytes, _ = torch.cuda.mem_get_info(device=device)
    budget_bytes = 512 * BYTES_PER_MEBIBYTE
    for margin_name in ["INTEGRATED_GPU_MEMORY_MARGIN_BYTES", "DISCRETE_GPU_MEMORY_MARGIN_BYTES"]:
        monkeypatch.setattr(target=speed_benchmark, name=margin_name, value=free_bytes - budget_bytes)
    with speed_benchmark.limit_process_memory(device=device) as memory_budget:
        assert memory_budget["budget_bytes"] <= budget_bytes + 64 * BYTES_PER_MEBIBYTE
        allowed_tensor = torch.empty(size=(64 * BYTES_PER_MEBIBYTE,), dtype=torch.uint8, device=device)
        with pytest.raises(expected_exception=torch.OutOfMemoryError):
            torch.empty(size=(2048 * BYTES_PER_MEBIBYTE,), dtype=torch.uint8, device=device)
        del allowed_tensor
    assert torch.cuda.get_per_process_memory_fraction(device=torch.cuda.current_device()) == 1.0
    speed_benchmark.release_cached_memory()


@requires_cuda
def test_dummy_benchmark_exits_without_run_folder(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the speed benchmark on the dummy task, which has no export: the trainer exits at construction without
    creating the run folder, after printing the opening panel and the four numbered steps (no export step), the
    searches of the largest batch sizes (both reaching the capped maximum), the inference panel with the test evaluation
    timed at a batch of 1, at the configured batch size and at the largest one, the exported formats reported as not
    available, the training panel with the configured and largest batch sizes timed, the training time of the run, and
    the initial weights; the memory cap is lifted.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture shortening the benchmark.
    """
    shorten_benchmark(monkeypatch=monkeypatch)
    with pytest.raises(expected_exception=SystemExit):
        build_benchmarking_dummy_trainer(temporary_directory=tmp_path, benchmark_speed=True)
    printed_output = capsys.readouterr().out

    # Nothing written to the experiments folder, and the allocator uncapped
    assert not (tmp_path / "experiments").exists()
    assert torch.cuda.get_per_process_memory_fraction(device=torch.cuda.current_device()) == 1.0

    # Numbered steps of the progress lines, four without the export step
    assert "[1/4] Searching the largest inference batch size" in printed_output
    assert "[4/4] Timing the training step at batch 2 and 16" in printed_output
    assert "Export of the network" not in printed_output
    assert f"{TEST_MAXIMUM_BATCH_SIZE} fits   → largest: {TEST_MAXIMUM_BATCH_SIZE}" in printed_output

    # Panels, tables and notes of the report
    assert "1 · Training: how long will the training take?" in printed_output
    assert "2 · Inference: how fast does the model answer?" in printed_output
    assert (f"Largest inference batch size: {TEST_MAXIMUM_BATCH_SIZE}, the maximum the search tries "
            f"({TEST_MAXIMUM_BATCH_SIZE}).") in printed_output
    assert (f"Largest training batch size: {TEST_MAXIMUM_BATCH_SIZE}, the maximum the search tries "
            f"({TEST_MAXIMUM_BATCH_SIZE}).") in printed_output
    assert "2  (configured)" in printed_output and f"{TEST_MAXIMUM_BATCH_SIZE}  (largest that fits)" in printed_output
    assert "not available: no export specification for the dummy_task task" in printed_output
    assert "2 steps (number_iterations) at batch 2: at least" in printed_output
    assert "initial weights, no checkpoint in the run folder" in printed_output
    assert printed_output.count("Test evaluation (predict_batch)") == 2


@requires_cuda
def test_dummy_benchmark_reads_the_last_checkpoint_without_writing(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the speed benchmark of a run that already has checkpoints times the inference with the weights of
    the last one, and leaves every file of the run folder unchanged.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture shortening the benchmark.
    """
    build_benchmarking_dummy_trainer(temporary_directory=tmp_path, benchmark_speed=False).run_training_loop()
    run_folder = tmp_path / "experiments" / "dummy_run"
    files_before = {path: path.stat().st_mtime_ns for path in run_folder.rglob("*") if path.is_file()}
    capsys.readouterr()

    shorten_benchmark(monkeypatch=monkeypatch)
    with pytest.raises(expected_exception=SystemExit):
        build_benchmarking_dummy_trainer(temporary_directory=tmp_path, benchmark_speed=True)
    printed_output = capsys.readouterr().out
    assert "checkpoint of iteration 2 of the run folder" in printed_output
    assert "no checkpoint in the run folder" not in printed_output
    assert {path: path.stat().st_mtime_ns for path in run_folder.rglob("*") if path.is_file()} == files_before


@requires_cuda
def test_classification_benchmark_times_the_exported_formats(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the speed benchmark of a classifier on a generated synthetic dataset: the TorchScript export is timed at
    every inference batch size, and the ONNX Runtime formats are timed when this environment has a GPU provider, or
    reported as not available with the reason otherwise; no run folder is created.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture shortening the benchmark.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 4}, maximum_objects=1,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="classification")
    configuration_path = write_benchmark_configuration(temporary_directory=tmp_path, dataset_folder=dataset_folder)
    experiment_configuration, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    capsys.readouterr()

    shorten_benchmark(monkeypatch=monkeypatch)
    with pytest.raises(expected_exception=SystemExit):
        ClassificationTrainer(experiment_configuration=experiment_configuration,
                              model_configuration=model_configuration, configuration_path=configuration_path)
    printed_output = capsys.readouterr().out
    assert not (tmp_path / "experiments").exists()

    # Five numbered steps with the export, and TorchScript timed at the batch of 1, the configured batch of 4 and the
    # largest batch of the search, on its progress line
    assert "[2/5] Exporting the network to TorchScript and ONNX" in printed_output
    assert "TorchScript file written" in printed_output
    torchscript_lines = [line for line in printed_output.splitlines()
                         if line.strip().startswith("TorchScript file, PyTorch backend")]
    assert len(torchscript_lines) == 1 and torchscript_lines[0].count(" done") == 3

    # ONNX Runtime formats timed with a GPU provider, reported with their reason without one
    if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        assert "has no GPU provider" in printed_output
        assert "not available: no ONNX file, see the export step" in printed_output
        assert "not available (same reason as above)" in printed_output


def test_sequence_comparison_gives_both_members_to_the_inference_model() -> None:
    """
    Verifies that the sequence comparison trainer gives the inference comparator the before and after members of the
    pairs of a prepared batch, each with half of its frames.
    """
    trainer = object.__new__(SequenceComparisonTrainer)
    frames = torch.arange(end=2 * 6).float().view(2, 6, 1, 1, 1).expand(2, 6, 3, 4, 4)
    batch = {"images": frames, "sequence_mask": torch.ones(size=(2, 6), dtype=torch.bool)}
    inference_inputs = trainer.build_inference_inputs(batch=batch)
    assert list(inference_inputs) == ["before_images", "after_images"]
    assert inference_inputs["before_images"].shape == (2, 3, 3, 4, 4)
    assert inference_inputs["before_images"][0, :, 0, 0, 0].tolist() == [0.0, 1.0, 2.0]
    assert inference_inputs["after_images"][0, :, 0, 0, 0].tolist() == [3.0, 4.0, 5.0]


def test_base_trainer_gives_the_images_to_the_inference_model() -> None:
    """
    Verifies that the trainers of the image and sequence tasks give the inference model the images of the prepared
    batch, its only input.
    """
    images = torch.zeros(size=(2, 3, 8, 8))
    inference_inputs = DummyTrainer.build_inference_inputs(self=object.__new__(DummyTrainer),
                                                           batch={"images": images, "labels": torch.zeros(size=(2,))})
    assert list(inference_inputs) == ["images"] and inference_inputs["images"] is images


def render_text(renderable: Any) -> str:
    """
    Renders a table of the report of the speed benchmark as plain text, wide enough that no cell is cut, so that the
    tests read its cells.

    Args:
        renderable (Any): The rich table.

    Returns:
        str: The rendered text.
    """
    console = Console(file=StringIO(), width=200, color_system=None)
    console.print(renderable)
    return console.file.getvalue()


def build_row(label: str, batch_size: Optional[int], seconds: Optional[float], status: str,
              failure_reason: Optional[str] = None) -> Dict[str, Any]:
    """
    Builds a row of a table of the report for the tests, measured in 'seconds' per iteration with windows 20 % apart
    (an unsteady timing) and 3 GB of peak memory, or without measurement when 'seconds' is None.

    Args:
        label (str): The label of the row, a format label or 'configured' / 'largest that fits'.
        batch_size (Optional[int]): The batch size, None for a format that could not be opened.
        seconds (Optional[float]): The seconds per iteration, None for a row without measurement.
        status (str): The status of the row.
        failure_reason (Optional[str]): The error of a failed format. Defaults to None.

    Returns:
        Dict[str, Any]: The row.
    """
    measurement = None if seconds is None else {"seconds_per_iteration": seconds, "fastest_seconds": 0.9 * seconds,
                                                "slowest_seconds": 1.1 * seconds, "measured_iterations": 10,
                                                "peak_memory_bytes": 3 * 10 ** 9}
    return {"label": label, "precision": "FP32", "batch_size": batch_size, "measurement": measurement,
            "status": status, "frames_per_batch": None if batch_size is None else 5 * batch_size,
            "failure_reason": failure_reason}


def test_training_table_writes_every_batch_size() -> None:
    """
    Verifies the training table of the report: a timed batch size gives its milliseconds per step (marked unsteady
    with windows 20 % apart), its throughput, its frame throughput and its peak memory, and a batch size that does not
    fit gives its status.
    """
    rows = [build_row(label="configured", batch_size=8, seconds=0.5, status=speed_benchmark.MEASURED_STATUS),
            build_row(label="largest that fits", batch_size=64, seconds=None,
                      status=speed_benchmark.OUT_OF_MEMORY_STATUS)]
    lines = render_text(renderable=speed_benchmark.build_training_table(rows=rows, sample_name="Sequences",
                                                                       counts_frames=True)).splitlines()
    assert "Milliseconds" in lines[0] and "Frames" in lines[0] and "Peak Memory" in lines[0]
    assert [line.split() for line in lines if "(configured)" in line] == [["8", "(configured)", "500.00", "*", "16.0",
                                                                          "80.0", "3.0"]]
    assert [line.split() for line in lines if "(largest that fits)" in line] == [
        ["64", "(largest", "that", "fits)", "out", "of", "memory", "—", "—", "—"]]


def test_inference_table_writes_every_format_and_speed_up() -> None:
    """
    Verifies the inference table of the report: one column per batch size with its role and unit (milliseconds per
    image at a batch of 1, images per second otherwise), the formats under their group, the speed-up over the network
    before export at the configured batch size, a failed batch size written with its status and without speed-up, and a
    format that could not be opened written once with its status.
    """
    evaluation_label, reference_label, torchscript_label, onnx_label = [
        inference_format["label"] for inference_format in speed_benchmark.INFERENCE_FORMATS[:4]]
    rows = [build_row(label=evaluation_label, batch_size=1, seconds=0.002, status=speed_benchmark.MEASURED_STATUS),
            build_row(label=evaluation_label, batch_size=8, seconds=0.25, status=speed_benchmark.MEASURED_STATUS),
            build_row(label=reference_label, batch_size=1, seconds=0.004, status=speed_benchmark.MEASURED_STATUS),
            build_row(label=reference_label, batch_size=8, seconds=0.5, status=speed_benchmark.MEASURED_STATUS),
            build_row(label=torchscript_label, batch_size=1, seconds=0.003, status=speed_benchmark.MEASURED_STATUS),
            build_row(label=torchscript_label, batch_size=8, seconds=None, status=speed_benchmark.FORMAT_FAILED_STATUS,
                      failure_reason="RuntimeError: broken"),
            build_row(label=onnx_label, batch_size=None, seconds=None, status=speed_benchmark.NOT_AVAILABLE_STATUS)]
    text = render_text(renderable=speed_benchmark.build_inference_table(
        rows=rows, configured_batch_size=8, largest_batch_size=8, sample_name="Images", counts_frames=False))
    lines = text.splitlines()

    # Headers of the batch sizes, their roles and units, and of the speed-up at the configured batch size
    assert "Batch 1" in lines[0] and "Batch 8" in lines[0] and "Speed-Up" in lines[1]
    assert "Configured, Largest" in lines[1]
    assert "Per Image" in text and "Per Second" in text and "At Batch" in text

    # Groups, then every format with its cells and its speed-up over the reference
    assert text.index(speed_benchmark.REPOSITORY_GROUP) < text.index(evaluation_label) < text.index(
        speed_benchmark.TRITON_GROUP) < text.index(torchscript_label)
    cells_by_label = {label: line.split(label)[1].split() for line in lines for label in
                      [evaluation_label, reference_label, torchscript_label, onnx_label] if label in line}
    assert cells_by_label[evaluation_label] == ["2.00", "*", "32.0", "*", "2.00", "x"]
    assert cells_by_label[reference_label] == ["4.00", "*", "16.0", "*", "1.00", "x"]
    assert cells_by_label[torchscript_label] == ["3.00", "*", "failed", "—"]
    assert cells_by_label[onnx_label] == ["not", "available", "—", "—"]


def test_steadiness_notes_count_the_unsteady_timings() -> None:
    """
    Verifies the note on the spread of the timing windows: the largest spread when every timing is steady, the number
    of unsteady timings otherwise, and no note for a table without measurement.
    """
    steady_row = build_row(label="configured", batch_size=8, seconds=0.5, status=speed_benchmark.MEASURED_STATUS)
    steady_row["measurement"] = {**steady_row["measurement"], "fastest_seconds": 0.49, "slowest_seconds": 0.51}
    unsteady_row = build_row(label="largest that fits", batch_size=16, seconds=1.0,
                             status=speed_benchmark.MEASURED_STATUS)
    assert speed_benchmark.describe_steadiness(rows=[steady_row]) == (
        "Timings steady: the timing windows of every cell are within 4.00 % of each other.")
    assert speed_benchmark.describe_steadiness(rows=[steady_row, unsteady_row]).startswith(
        "* Unsteady timing (1 of 2): its slowest and fastest timing windows differ by more than 10 % (up to 20.00 %)")
    assert speed_benchmark.describe_steadiness(rows=[build_row(
        label="configured", batch_size=8, seconds=None, status=speed_benchmark.OUT_OF_MEMORY_STATUS)]) is None


def test_native_error_output_is_dropped_only_for_a_lack_of_memory(capfd: pytest.CaptureFixture) -> None:
    """
    Verifies that what a native library writes to the error stream during a probe of the benchmark (the C++ stack trace
    of libtorch) is dropped when the probe ran out of memory, and written back otherwise.

    Args:
        capfd (pytest.CaptureFixture): The pytest fixture capturing the file descriptors 1 and 2.
    """
    with speed_benchmark.capture_native_error_output() as native_output:
        os.write(2, b"stack trace of an out of memory error\n")
        native_output["out_of_memory"] = True
    with speed_benchmark.capture_native_error_output():
        os.write(2, b"warning of another library\n")
    captured_error = capfd.readouterr().err
    assert "stack trace of an out of memory error" not in captured_error
    assert "warning of another library" in captured_error
