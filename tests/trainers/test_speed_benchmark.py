"""
Tests of the 'benchmark_speed' diagnosis mode: the helpers of vision_training/trainers/speed_benchmark.py (tiling of a
loader batch, search of the largest batch size, out of memory errors, memory cap) and BaseTrainer.benchmark_speed of
vision_training/trainers/base_trainer.py, run on the dummy task (training step and test evaluation only, the task
having no export) and on the synthetic classification task (TorchScript and ONNX Runtime formats too), checking that
the benchmark exits without training nor run folder and prints its report.
"""

import os
import subprocess
import onnxruntime
import pytest
import torch
import yaml

from functools import partial
from pathlib import Path
from typing import Any, List

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
    creating the run folder, after printing the searches of the largest batch sizes (both reaching the capped maximum),
    the inference table with the test evaluation measured at a batch of 1, at the configured batch size and at the
    largest one, the exported formats reported as not available, the training table with the configured and largest
    batch sizes measured, the training time of the run, and the note on the initial weights; the memory cap is lifted.

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

    # Searches, tables and notes of the report
    assert f"Batch Size {TEST_MAXIMUM_BATCH_SIZE} : Fits" in printed_output
    assert (f"Largest inference batch size : {TEST_MAXIMUM_BATCH_SIZE}, the largest the search tries "
            f"({TEST_MAXIMUM_BATCH_SIZE})") in printed_output
    assert (f"Largest training batch size : {TEST_MAXIMUM_BATCH_SIZE}, the largest the search tries "
            f"({TEST_MAXIMUM_BATCH_SIZE})") in printed_output
    assert "Training Step" in printed_output and "Inference" in printed_output
    assert "Configured" in printed_output and "Largest Found" in printed_output
    assert "no export specification for the dummy_task task" in printed_output
    assert "2 iterations (number_iterations) at the configured batch size" in printed_output
    assert "No checkpoint in the run folder" in printed_output
    assert printed_output.count("PyTorch Evaluation") >= 3


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
    assert "Weights : checkpoint of iteration 2 of the run folder" in printed_output
    assert "No checkpoint in the run folder" not in printed_output
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
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    capsys.readouterr()

    shorten_benchmark(monkeypatch=monkeypatch)
    with pytest.raises(expected_exception=SystemExit):
        ClassificationTrainer(experiment_configuration=experiment_configuration,
                              model_configuration=model_configuration, configuration_path=configuration_path)
    printed_output = capsys.readouterr().out
    assert not (tmp_path / "experiments").exists()

    # TorchScript measured at the batch of 1, the configured batch of 4 and the largest batch of the search
    assert "Timing TorchScript FP32" in printed_output
    torchscript_section = printed_output.split("Timing TorchScript FP32")[1].split("Timing ")[0]
    assert torchscript_section.count(": Measured") == 3

    # ONNX Runtime formats timed with a GPU provider, reported with their reason without one
    if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        assert "has no GPU provider" in printed_output
        assert "Not Available" in printed_output


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


def test_report_tables_write_every_row() -> None:
    """
    Verifies the cells of the tables of the report: a measured row gives its milliseconds, throughput, frame
    throughput, window spread and peak memory, a row without measurement or without batch size gives empty cells and
    its status, and a format whose memory is outside PyTorch gives no peak memory.
    """
    measurement = {"seconds_per_iteration": 0.5, "fastest_seconds": 0.45, "slowest_seconds": 0.55,
                   "measured_iterations": 10, "peak_memory_bytes": 3 * 10 ** 9}
    rows = [{"label": "Configured", "precision": "BF16", "batch_size": 8, "measurement": measurement,
             "status": speed_benchmark.MEASURED_STATUS, "frames_per_batch": 40},
            {"label": "Largest Found", "precision": "BF16", "batch_size": 64, "measurement": None,
             "status": speed_benchmark.OUT_OF_MEMORY_STATUS, "frames_per_batch": None}]
    training_table = speed_benchmark.build_training_table(rows=rows, sample_name="Sequences", counts_frames=True,
                                                          legend="legend")
    assert training_table["columns"] == ["Batch", "Batch\nSize", "Milliseconds\nPer Iteration",
                                         "Sequences\nPer Second", "Frames\nPer Second", "Window\nSpread",
                                         "Peak Memory\n(GB)", "Status"]
    assert training_table["rows"][0] == ["Configured", 8, "500.00", "16.0", "80.0", pytest.approx(0.2), "3.0",
                                         "Measured"]
    assert training_table["rows"][1] == ["Largest Found", 64, None, None, None, None, None, "Out Of Memory"]

    # Inference rows, one of a format measured without peak memory and one of a format that could not be opened
    inference_rows = [{**rows[0], "label": "ONNX Runtime CUDA", "precision": "FP32",
                       "measurement": {**measurement, "peak_memory_bytes": None}},
                      {"label": "TorchScript", "precision": "FP32", "batch_size": None, "measurement": None,
                       "status": speed_benchmark.NOT_AVAILABLE_STATUS, "frames_per_batch": None}]
    inference_table = speed_benchmark.build_inference_table(rows=inference_rows, sample_name="Images",
                                                            counts_frames=False, legend="legend")
    assert inference_table["rows"][0] == ["ONNX Runtime CUDA", "FP32", 8, "500.00", "16.0", pytest.approx(0.2), None,
                                          "Measured"]
    assert inference_table["rows"][1] == ["TorchScript", "FP32", None, None, None, None, None, "Not Available"]
