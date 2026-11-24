"""
Tests of scripts/deployment/check_export_size_estimates.py, the check of the Export Sizes panel against real exports:
the engines a model adds to the TensorRT cache are summed and checked for the cache prefix of its configuration, the
report gives the error of every estimate, places a real engine inside or outside its range and gives the time of the
engine builds or the failure of the server, a model Triton fails to serve is returned as its error, the calibration
fits the TensorRT overhead per operator from every engine built, the results file keeps one line per run with its
columns, and the optimizer of a run is of the class the trainer builds.
"""

import csv
import numpy as np
import pytest
import torch

from pathlib import Path
from torch import nn
from typing import Dict, List
from tritonclient.utils import InferenceServerException

import scripts.deployment.check_export_size_estimates as check_module
from scripts.deployment.check_export_size_estimates import (RESULT_COLUMNS, build_run_optimizer, compute_relative_error,
                                                            fit_calibration, measure_new_engines, print_check_report,
                                                            read_engine_cache_prefix, serve_model_for_engines,
                                                            write_results)
from vision_training.deployment.export_size_estimates import (TENSORRT_BYTES_PER_OTHER_OPERATOR,
                                                              TENSORRT_BYTES_PER_WEIGHTED_OPERATOR)

# One run of the check as check_run gives it, with the engines of both precisions measured: 100 weighted and 1000 other
# operators whose engines hold 2,000,000 bytes of overhead besides 4,000,000 bytes of FP32 weights
CHECKED_RUN = {"run": "DET_test", "task": "object_detection", "device": "cuda", "estimate_seconds": 1.2,
               "total_parameters": 1000000, "exported_parameters": 1000000, "state_dictionary_tensors": 400,
               "state_dictionary_key_characters": 24000, "constant_bytes": 4000000, "batch_normalization_bytes": 0,
               "weighted_operators": 100, "other_operators": 1000, "pytorch_fp32_estimated": 4100000,
               "pytorch_fp32_real": 4120000, "pytorch_fp16_estimated": 2100000, "pytorch_fp16_real": 2090000,
               "torchscript_measured": 4500000, "onnx_estimated": 4180000, "onnx_graph_real": 170000,
               "onnx_data_real": 4000000, "onnx_real": 4170000, "checkpoint_estimated": 16000000,
               "checkpoint_real": 16100000, "tensorrt_fp32_lowest": 5500000, "tensorrt_fp32_highest": 6500000,
               "tensorrt_fp32_real": 6000000, "tensorrt_fp32_build_seconds": 12.3,
               "tensorrt_fp32_named_after_prefix": True, "tensorrt_fp16_lowest": 3500000,
               "tensorrt_fp16_highest": 3900000, "tensorrt_fp16_real": 4000000, "tensorrt_fp16_build_seconds": 14.06,
               "tensorrt_fp16_named_after_prefix": False, "tensorrt_largest_batch": 16, "tensorrt_fp32_error": None,
               "tensorrt_fp16_error": None}
# The same run before its engines were built
UNMEASURED_ENGINES = {"tensorrt_fp32_real": None, "tensorrt_fp32_build_seconds": None,
                      "tensorrt_fp32_named_after_prefix": None, "tensorrt_fp16_real": None,
                      "tensorrt_fp16_build_seconds": None, "tensorrt_fp16_named_after_prefix": None,
                      "tensorrt_largest_batch": None, "tensorrt_fp32_error": None, "tensorrt_fp16_error": None}
# Error of a server crashing at the first TensorRT request of a model, as the Triton client raises it
SERVER_CRASH_ERROR = "[StatusCode.UNAVAILABLE] Stream removed (Socket closed)"


def test_new_engines_are_measured_with_their_prefix(tmp_path: Path) -> None:
    """
    Verifies that the engines of a model are the '.engine' files it added to the cache, one per part of the graph, the
    profiles and the engines already there being left out, that an engine not starting with the cache prefix of the
    model is reported as such (ONNX Runtime naming it after the graph only), and that a model adding no engine is
    refused.

    Args:
        tmp_path (Path): The pytest temporary directory, standing for the engine cache.
    """
    prefix = "image_classifier_onnx_1_fp32_0123456789ab"
    (tmp_path / "object_detector_onnx_1_fp32_ba9876543210_1_0_0_sm121.engine").write_bytes(b"0" * 999)
    engines_before = set(tmp_path.glob("*.engine"))
    (tmp_path / f"{prefix}_7477124481917795400_0_0_sm121.engine").write_bytes(b"0" * 100)
    (tmp_path / f"{prefix}_7477124481917795400_1_1_sm121.engine").write_bytes(b"0" * 50)
    (tmp_path / f"{prefix}_7477124481917795400_0_0_sm121.profile").write_bytes(b"0" * 7)
    assert measure_new_engines(engine_cache_folder=tmp_path, engines_before=engines_before,
                               engine_cache_prefix=prefix) == (150, True)

    # An engine named after the graph only, then no engine at all
    engines_before = set(tmp_path.glob("*.engine"))
    (tmp_path / "TensorrtExecutionProvider_TRTKernel_graph_main_graph_14681384001524542084_0_0_sm121.engine"
     ).write_bytes(b"0" * 30)
    assert measure_new_engines(engine_cache_folder=tmp_path, engines_before=engines_before,
                               engine_cache_prefix="sequence_labeller_onnx_1_fp32_0123456789ab") == (30, False)
    with pytest.raises(RuntimeError, match="No engine written"):
        measure_new_engines(engine_cache_folder=tmp_path, engines_before=set(tmp_path.glob("*.engine")),
                            engine_cache_prefix=prefix)


def test_engine_cache_prefix_is_read_from_the_configuration(tmp_path: Path) -> None:
    """
    Verifies that the engine cache prefix is read from the TensorRT parameters of a config.pbtxt, and that a
    configuration without one is refused.

    Args:
        tmp_path (Path): The pytest temporary directory, standing for the folder of a model.
    """
    (tmp_path / "config.pbtxt").write_text(
        'name: "image_classifier_onnx"\n'
        '  parameters { key: "trt_engine_cache_enable" value: "1" }\n'
        '  parameters { key: "trt_engine_cache_prefix" value: "image_classifier_onnx_1_fp16_0123456789ab" }\n')
    assert read_engine_cache_prefix(model_directory=tmp_path) == "image_classifier_onnx_1_fp16_0123456789ab"
    (tmp_path / "config.pbtxt").write_text('name: "image_classifier_onnx"\n')
    with pytest.raises(ValueError, match="trt_engine_cache_prefix"):
        read_engine_cache_prefix(model_directory=tmp_path)


def crash_server(model_name: str, request: Dict[str, np.ndarray], batch_sizes: List[int]) -> None:
    """
    Stands for time_first_requests when Triton crashes at the first TensorRT request of a model, as the keypoint
    detector of the tutorials makes Triton 26.01 do.

    Args:
        model_name (str): The Triton model name.
        request (Dict[str, np.ndarray]): The batch-1 request, keyed by input name.
        batch_sizes (List[int]): The batch sizes of the first requests.

    Raises:
        InferenceServerException: Always, with the error of a closed gRPC stream on two lines.
    """
    raise InferenceServerException(msg="Stream removed (Socket closed)\nsecond line", status="StatusCode.UNAVAILABLE")


def test_failed_serving_is_returned_as_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that a model served without error gives the seconds of its first requests, and that a server crashing at
    the first TensorRT request of a model gives the first line of its error instead of stopping the check.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the calls to the Triton server.
    """
    calls = []
    monkeypatch.setattr(check_module, "load_served_model",
                        lambda server, model_name, configuration: calls.append("load"))
    monkeypatch.setattr(check_module, "unload_served_model", lambda server, model_name: calls.append("unload"))
    monkeypatch.setattr(check_module, "time_first_requests",
                        lambda model_name, request, batch_sizes: {batch_size: 2.0 for batch_size in batch_sizes})
    assert serve_model_for_engines(model_name="image_classifier_onnx", request={}, batch_sizes=[1, 16]) == (
        {1: 2.0, 16: 2.0}, None)
    assert calls == ["load", "unload"]

    # The server crashing at the first request: no seconds, the error, no unload sent to the crashed server
    monkeypatch.setattr(check_module, "time_first_requests", crash_server)
    assert serve_model_for_engines(model_name="keypoint_detector_onnx", request={}, batch_sizes=[1, 16]) == (
        None, SERVER_CRASH_ERROR)
    assert calls == ["load", "unload", "load"]


def test_report_gives_the_errors_and_the_range_positions(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the report gives the relative error of every single estimate, a real engine inside its range as
    'inside' and one above it with its distance, the time of the engine builds and the naming of the engines, an
    engine not measured as such, and the error of a model Triton failed to serve.

    Args:
        capsys (pytest.CaptureFixture): Captures the printed report.
    """
    unmeasured_run = {**CHECKED_RUN, **UNMEASURED_ENGINES, "run": "CLS_test"}
    failed_run = {**CHECKED_RUN, "run": "KPT_test", "tensorrt_fp32_build_seconds": None,
                  "tensorrt_fp32_error": SERVER_CRASH_ERROR}
    print_check_report(results=[CHECKED_RUN, unmeasured_run, failed_run])
    report_lines = capsys.readouterr().out.splitlines()
    detector_lines = report_lines[report_lines.index(next(line for line in report_lines if "DET_test" in line)):]
    assert any("PyTorch FP32 Weights" in line and "-0.49 %" in line for line in detector_lines)
    assert any("ONNX" in line and "+0.24 %" in line for line in detector_lines)
    assert any("TensorRT FP32 Engine" in line and "inside" in line for line in detector_lines)
    assert any("TensorRT FP16 Engine" in line and "outside by 2.5 %" in line for line in detector_lines)
    assert "  TensorRT FP32 engines built in 12.3 seconds (batch 1 then 16), named after their cache prefix: yes" in (
        detector_lines)
    assert "  TensorRT FP16 engines built in 14.1 seconds (batch 1 then 16), named after their cache prefix: no" in (
        detector_lines)
    assert any("TensorRT FP32 Engine" in line and "not measured" in line for line in report_lines)
    assert sum("engines built in" in line for line in report_lines) == 3
    failed_lines = report_lines[report_lines.index(next(line for line in report_lines if "KPT_test" in line)):]
    assert any(f"TensorRT FP32 serving failed: {SERVER_CRASH_ERROR}" in line for line in failed_lines)
    assert sum("serving failed" in line for line in report_lines) == 1
    assert compute_relative_error(estimated_bytes=110, real_bytes=100) == pytest.approx(0.1)
    assert compute_relative_error(estimated_bytes=110, real_bytes=None) is None


def test_calibration_fits_the_tensorrt_overhead(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the calibration recovers the TensorRT bytes per weighted and other operator of engines built from
    known overheads, the engine of a model Triton failed to serve included, gives the spread of the real overheads
    around the current constants, and prints the largest error of every single estimate.

    Args:
        capsys (pytest.CaptureFixture): Captures the printed calibration.
    """
    # Two runs whose engine overheads are 15,000 bytes per weighted and 500 per other operator exactly
    first_run = {**CHECKED_RUN, "tensorrt_fp32_real": 4000000 + 100 * 15000 + 1000 * 500,
                 "tensorrt_fp16_real": 2000000 + 100 * 15000 + 1000 * 500}
    second_run = {**CHECKED_RUN, "run": "CLS_test", "weighted_operators": 40, "other_operators": 120,
                  "tensorrt_fp32_real": 4000000 + 40 * 15000 + 120 * 500,
                  "tensorrt_fp16_real": 2000000 + 40 * 15000 + 120 * 500}
    # A third run whose server crashed after its first engine, the engine built with the same overheads: still fitted
    failed_run = {**CHECKED_RUN, "run": "KPT_test", "tensorrt_fp32_real": 4000000 + 100 * 15000 + 1000 * 500,
                  "tensorrt_fp16_real": None, "tensorrt_fp32_error": SERVER_CRASH_ERROR,
                  "tensorrt_fp16_error": SERVER_CRASH_ERROR}
    fit_calibration(results=[first_run, second_run, failed_run])
    printed_text = capsys.readouterr().out
    assert f"TENSORRT_BYTES_PER_WEIGHTED_OPERATOR current {TENSORRT_BYTES_PER_WEIGHTED_OPERATOR}, fitted 15000" in (
        printed_text)
    assert f"TENSORRT_BYTES_PER_OTHER_OPERATOR    current {TENSORRT_BYTES_PER_OTHER_OPERATOR}, fitted 500" in (
        printed_text)
    # Real overheads against the current constants: the larger and the smaller run
    current_ratios = [(100 * 15000 + 1000 * 500) / (100 * TENSORRT_BYTES_PER_WEIGHTED_OPERATOR
                                                    + 1000 * TENSORRT_BYTES_PER_OTHER_OPERATOR),
                      (40 * 15000 + 120 * 500) / (40 * TENSORRT_BYTES_PER_WEIGHTED_OPERATOR
                                                  + 120 * TENSORRT_BYTES_PER_OTHER_OPERATOR)]
    assert (f"from 1.00 to 1.00 times the fitted one, from {min(current_ratios):.2f} to {max(current_ratios):.2f} "
            f"times the current one (5 engines)") in printed_text
    assert "Training Checkpoint    largest error 0.62 % (3 runs)" in printed_text


def test_results_file_keeps_one_line_per_run(tmp_path: Path) -> None:
    """
    Verifies that export_size_check.csv has the columns of RESULT_COLUMNS and one line per run, an engine not measured
    being left empty.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    unmeasured_run = {**CHECKED_RUN, **UNMEASURED_ENGINES, "run": "CLS_test"}
    write_results(results=[CHECKED_RUN, unmeasured_run], results_path=tmp_path / "export_size_check.csv")
    with (tmp_path / "export_size_check.csv").open() as results_file:
        rows = list(csv.DictReader(results_file))
    assert list(rows[0]) == RESULT_COLUMNS
    assert [row["run"] for row in rows] == ["DET_test", "CLS_test"]
    assert rows[1]["tensorrt_fp32_real"] == "" and rows[0]["tensorrt_fp32_real"] == "6000000"
    assert rows[0]["tensorrt_fp16_named_after_prefix"] == "False" and rows[1]["tensorrt_fp16_build_seconds"] == ""
    assert rows[0]["tensorrt_fp32_error"] == ""


def test_run_optimizer_follows_the_configuration() -> None:
    """
    Verifies that the optimizer of a run is of the class BaseTrainer.create_optimizer builds for its configuration,
    SGD with its momentum.
    """
    model = nn.Linear(in_features=3, out_features=2)
    assert type(build_run_optimizer(model=model, optimizer_name="adam")) is torch.optim.Adam
    assert type(build_run_optimizer(model=model, optimizer_name="adamw")) is torch.optim.AdamW
    sgd_optimizer = build_run_optimizer(model=model, optimizer_name="sgd")
    assert type(sgd_optimizer) is torch.optim.SGD and sgd_optimizer.param_groups[0]["momentum"] == 0.9
