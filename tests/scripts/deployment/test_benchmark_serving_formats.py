"""
Tests of scripts/deployment/benchmark_serving_formats.py, the serving benchmark of the exported models: the settings
of the server given at load time (dynamic batching and instances) leave the exported configuration unchanged, the
perf_analyzer report is read in milliseconds, the test sample is linked in its frame order, results.csv keeps every
measurement so that a relaunch skips the recorded ones, every row gets the loads of its groups, the gRPC loads read
the metrics and statistics of the server, and the fixed frame count of an exported sequence model is made dynamic.
"""

import json
import numpy as np
import onnx
import pytest

from onnx import TensorProto, helper
from pathlib import Path

from scripts.deployment.benchmark_serving_formats import (RESULT_COLUMNS, append_result,
                                                          average_server_microseconds, build_result_key,
                                                          build_server_configuration, count_frames_per_sample,
                                                          get_repository_formats, get_row_name,
                                                          parse_perf_analyzer_report, parse_triton_gpu_metrics,
                                                          plan_served_loads, read_completed_keys, repeat_request,
                                                          repair_onnx_output_frames, select_test_sample)

# Configuration of an exported model as the configuration endpoint of Triton returns it, reduced to the fields the
# server settings change
EXPORTED_CONFIGURATION = {"name": "image_classifier_onnx", "max_batch_size": 16,
                          "dynamic_batching": {"max_queue_delay_microseconds": 1000},
                          "instance_group": [{"name": "image_classifier_onnx", "kind": "KIND_GPU", "count": 1}]}

# Report of one perf_analyzer measurement written with --verbose-csv --collect-metrics, latencies in microseconds,
# GPU use as a fraction and power in watts per GPU, the memory columns at 0 as on the GB10
PERF_ANALYZER_REPORT = ("Concurrency,Inferences/Second,Client Send,Network+Server Send/Recv,Server Queue,"
                        "Server Compute Input,Server Compute Infer,Server Compute Output,Client Recv,p50 latency,"
                        "p90 latency,p95 latency,p99 latency,Avg latency,request/response,response wait,"
                        "Avg GPU Utilization,Avg GPU Power Usage,Max GPU Memory Usage,Total GPU Memory\n"
                        "4,812.5,15,300,120,40,4500,30,5,4900,5200,5300,6100,4950,0,0,GPU-22852066:0.375897;,"
                        "GPU-22852066:26.306;,GPU-22852066:0;,GPU-22852066:0;\n")

# Lines of the Prometheus metrics of the server read during a gRPC load, GPU use as a fraction and power in watts
TRITON_METRICS = ("# HELP nv_gpu_utilization GPU utilization rate [0.0 - 1.0)\n"
                  "# TYPE nv_gpu_utilization gauge\n"
                  'nv_gpu_utilization{gpu_uuid="GPU-22852066"} 0.75\n'
                  'nv_gpu_power_usage{gpu_uuid="GPU-22852066"} 41.5\n'
                  'nv_gpu_memory_used_bytes{gpu_uuid="GPU-22852066"} 0\n')

# Load groups of the focused scope: the whole grid with the exported configuration for every row, and the other
# server settings at batch 1 for TensorRT FP16 only, the exported configuration listed again to check it is not doubled
FOCUSED_LOAD_GROUPS = [{"precision_settings": ["onnxruntime_cuda", "tensorrt_fp32", "tensorrt_fp16"],
                        "server_settings": [(True, 1)], "transports": ["grpc", "system_shared_memory"],
                        "batch_sizes": [1, 4, 16], "concurrencies": [1, 4, 16]},
                       {"precision_settings": ["tensorrt_fp16"], "server_settings": [(True, 1), (False, 1), (True, 2)],
                        "transports": ["grpc"], "batch_sizes": [1], "concurrencies": [4]}]


def test_server_settings_change_a_copy_of_the_exported_configuration() -> None:
    """
    Turning the dynamic batcher off removes it and the instance count follows the setting, while the exported
    configuration, read once and reused for every setting, stays as it was.
    """
    configuration = build_server_configuration(base_configuration=EXPORTED_CONFIGURATION, dynamic_batching=False,
                                               instance_count=2)
    assert "dynamic_batching" not in configuration
    assert configuration["instance_group"][0]["count"] == 2
    assert EXPORTED_CONFIGURATION["dynamic_batching"] == {"max_queue_delay_microseconds": 1000}
    assert EXPORTED_CONFIGURATION["instance_group"][0]["count"] == 1

    # The dynamic batcher kept as exported
    configuration = build_server_configuration(base_configuration=EXPORTED_CONFIGURATION, dynamic_batching=True,
                                               instance_count=1)
    assert configuration == EXPORTED_CONFIGURATION


def test_server_settings_refuse_a_configuration_without_dynamic_batcher() -> None:
    """
    A configuration without the dynamic batcher written by the export cannot be measured with and without it.
    """
    configuration = {key: value for key, value in EXPORTED_CONFIGURATION.items() if key != "dynamic_batching"}
    with pytest.raises(ValueError, match="dynamic batcher"):
        build_server_configuration(base_configuration=configuration, dynamic_batching=True, instance_count=1)


def test_perf_analyzer_report_is_read_in_milliseconds(tmp_path: Path) -> None:
    """
    The latencies of the report, in microseconds, are given in milliseconds, the server compute time sums its
    input, inference and output parts, and the GPU use, a fraction, is given in percent.

    Args:
        tmp_path (Path): The temporary folder of the test.
    """
    report_path = tmp_path / "report.csv"
    report_path.write_text(PERF_ANALYZER_REPORT)
    measure = parse_perf_analyzer_report(report_path=report_path)
    assert measure["p50_milliseconds"] == pytest.approx(4.9)
    assert measure["p99_milliseconds"] == pytest.approx(6.1)
    assert measure["average_milliseconds"] == pytest.approx(4.95)
    assert measure["inferences_per_second"] == pytest.approx(812.5)
    assert measure["server_queue_microseconds"] == pytest.approx(120.0)
    assert measure["server_compute_microseconds"] == pytest.approx(4570.0)
    assert measure["gpu_utilization_percent"] == pytest.approx(37.5897)
    assert measure["gpu_power_watts"] == pytest.approx(26.306)


def test_test_sample_keeps_the_frame_order(tmp_path: Path) -> None:
    """
    The frames of the first sequence of the asked length are linked under names sorting in their order, the shorter
    sequences before it being skipped, and an image manifest gives its first image.

    Args:
        tmp_path (Path): The temporary folder of the test.
    """
    # Sequence split: a sequence of 2 frames, then one of 3 frames whose frames are not named in their order
    sequence_folder = tmp_path / "sequences"
    frames = ["test_000001/000009.png", "test_000001/000002.png", "test_000001/000005.png"]
    for frame in ["test_000000/000000.png", "test_000000/000001.png"] + frames:
        (sequence_folder / "data" / frame).parent.mkdir(parents=True, exist_ok=True)
        (sequence_folder / "data" / frame).write_bytes(b"")
    manifest = {"sequences": [{"frames": ["test_000000/000000.png", "test_000000/000001.png"]}, {"frames": frames}]}
    (sequence_folder / "labels.json").write_text(json.dumps(manifest))
    sample_folder = select_test_sample(test_data_folder=sequence_folder, sample_folder=tmp_path / "sample",
                                       sequence_length=3)
    linked_paths = sorted(sample_folder.iterdir())
    assert [path.resolve() for path in linked_paths] == [(sequence_folder / "data" / frame).resolve()
                                                         for frame in frames]

    # No sequence of the asked length
    with pytest.raises(ValueError, match="4 frames"):
        select_test_sample(test_data_folder=sequence_folder, sample_folder=tmp_path / "sample", sequence_length=4)

    # Image split, linked again into the same folder, emptied first
    image_folder = tmp_path / "images"
    (image_folder / "data").mkdir(parents=True)
    (image_folder / "data" / "test_000000.png").write_bytes(b"")
    (image_folder / "labels.json").write_text(json.dumps({"images": [{"file_name": "test_000000.png"}]}))
    sample_folder = select_test_sample(test_data_folder=image_folder, sample_folder=tmp_path / "sample",
                                       sequence_length=None)
    assert [path.name for path in sample_folder.iterdir()] == ["000000.png"]


def test_results_file_gives_back_the_recorded_keys(tmp_path: Path) -> None:
    """
    Every measurement appended to results.csv is found again by its key at a relaunch, its booleans and numbers read
    back as the strings build_result_key compares, and a column outside RESULT_COLUMNS is refused.

    Args:
        tmp_path (Path): The temporary folder of the test.
    """
    results_path = tmp_path / "results.csv"
    assert read_completed_keys(results_path=results_path) == set()
    result = {"model": "classification", "row": "triton_tensorrt_fp16", "kind": "load", "dynamic_batching": False,
              "instance_count": 2, "transport": "grpc", "batch_size": 16, "concurrency": 4, "status": "ok",
              "p50_milliseconds": 3.2}
    append_result(results_path=results_path, result=result)
    append_result(results_path=results_path, result={**result, "concurrency": 16})
    assert read_completed_keys(results_path=results_path) == {build_result_key(result=result),
                                                              build_result_key(result={**result, "concurrency": 16})}
    assert results_path.read_text().splitlines()[0].split(",") == RESULT_COLUMNS

    with pytest.raises(ValueError, match="unknown columns"):
        append_result(results_path=results_path, result={**result, "throughput": 1.0})


def test_requests_count_frames_and_repeat_to_the_batch_size() -> None:
    """
    A request is repeated to the batch size along its first dimension, and its images per sample are the frames of a
    sequence, or 1 for an image.
    """
    image_request = {"images": np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)}
    sequence_request = {"images": np.zeros((1, 7, 3, 2, 2), dtype=np.float32)}
    repeated_request = repeat_request(request=image_request, batch_size=4)
    assert repeated_request["images"].shape == (4, 3, 2, 2)
    assert np.array_equal(repeated_request["images"][3], image_request["images"][0])
    assert count_frames_per_sample(request=image_request) == 1
    assert count_frames_per_sample(request=sequence_request) == 7


def test_rows_of_the_precision_settings() -> None:
    """
    The TorchScript model is served once, from the ONNX Runtime CUDA repository, and every ONNX model is the row of its
    precision setting.
    """
    assert get_repository_formats(precision_setting="onnxruntime_cuda") == ["torchscript", "onnx"]
    assert get_repository_formats(precision_setting="tensorrt_fp16") == ["onnx"]
    assert get_row_name(precision_setting="onnxruntime_cuda", export_format="torchscript") == "triton_torchscript"
    assert get_row_name(precision_setting="tensorrt_fp16", export_format="onnx") == "triton_tensorrt_fp16"


def test_loads_of_a_row_follow_its_load_groups() -> None:
    """
    Every row gets the whole grid of the exported configuration, only the TensorRT FP16 row gets the other server
    settings, and a load listed by two groups is planned once.
    """
    common_columns = {"model": "classification", "row": "triton_tensorrt_fp32"}
    fp32_loads = plan_served_loads(load_groups=FOCUSED_LOAD_GROUPS, precision_setting="tensorrt_fp32",
                                   common_columns=common_columns)
    assert len(fp32_loads) == 18
    assert {(load["dynamic_batching"], load["instance_count"]) for load in fp32_loads} == {(True, 1)}
    fp16_loads = plan_served_loads(load_groups=FOCUSED_LOAD_GROUPS, precision_setting="tensorrt_fp16",
                                   common_columns={**common_columns, "row": "triton_tensorrt_fp16"})
    assert len(fp16_loads) == 20
    assert fp16_loads[-1] == {"model": "classification", "row": "triton_tensorrt_fp16", "kind": "load",
                              "dynamic_batching": True, "instance_count": 2, "transport": "grpc", "batch_size": 1,
                              "concurrency": 4}


def test_gpu_metrics_and_server_times_of_a_grpc_load() -> None:
    """
    The GPU use of the server metrics, a fraction, is given in percent with its power, the comment lines and the
    other metrics being skipped, and two readings of the inference statistics give the average queue and compute
    times of the requests executed between them, in microseconds.
    """
    assert parse_triton_gpu_metrics(metrics_text=TRITON_METRICS) == {"gpu_utilization_percent": 75.0,
                                                                     "gpu_power_watts": 41.5}
    with pytest.raises(ValueError, match="nv_gpu_power_usage"):
        parse_triton_gpu_metrics(metrics_text=TRITON_METRICS.replace("nv_gpu_power_usage", "nv_energy"))

    # 10 requests between the readings: 2 ms of queue and 5 ms of compute (1 + 3 + 1) on average
    statistics_before = {"queue": (5, 1_000_000), "compute_input": (5, 0), "compute_infer": (5, 0),
                         "compute_output": (5, 0)}
    statistics_after = {"queue": (15, 21_000_000), "compute_input": (15, 10_000_000),
                        "compute_infer": (15, 30_000_000), "compute_output": (15, 10_000_000)}
    server_times = average_server_microseconds(statistics_before=statistics_before, statistics_after=statistics_after)
    assert server_times == {"server_queue_microseconds": pytest.approx(2000.0),
                            "server_compute_microseconds": pytest.approx(5000.0)}
    with pytest.raises(ValueError, match="requests executed"):
        average_server_microseconds(statistics_before=statistics_before, statistics_after=statistics_before)


def write_served_onnx_model(model_folder: Path, input_shape: list, output_shape: list,
                            configured_output_dimensions: str) -> None:
    """
    Writes a served ONNX model whose output shape is declared like the export writes it, with its config.pbtxt
    output line, the frame count of a sequence model possibly fixed in the file while the configuration keeps it -1.

    Args:
        model_folder (Path): The folder of the served model, receiving 'config.pbtxt' and '1/model.onnx'.
        input_shape (list): The declared input shape, names for the dynamic dimensions.
        output_shape (list): The declared output shape, names for the dynamic dimensions.
        configured_output_dimensions (str): The output dimensions of config.pbtxt, batch excluded, e.g. '-1, 3'.
    """
    # Mean over the pixels of every frame, its output and an intermediate shape declared as given
    reduced_axes = [3, 4] if len(input_shape) == 5 else [2, 3]
    node = helper.make_node(op_type="ReduceMean", inputs=["images"], outputs=["class_probabilities"],
                            axes=reduced_axes, keepdims=0)
    graph = helper.make_graph(nodes=[node], name="served_model",
                              inputs=[helper.make_tensor_value_info("images", TensorProto.FLOAT, input_shape)],
                              outputs=[helper.make_tensor_value_info("class_probabilities", TensorProto.FLOAT,
                                                                     output_shape)],
                              value_info=[helper.make_tensor_value_info("intermediate", TensorProto.FLOAT,
                                                                        output_shape)])
    (model_folder / "1").mkdir(parents=True)
    onnx.save(helper.make_model(graph=graph, opset_imports=[helper.make_opsetid(domain="", version=13)]),
              str(model_folder / "1" / "model.onnx"))
    output_line = f'{{ name: "class_probabilities" data_type: TYPE_FP32 dims: [ {configured_output_dimensions} ] }}'
    (model_folder / "config.pbtxt").write_text(f"output [\n  {output_line}\n]\n")


def test_fixed_output_frames_of_a_sequence_model_become_dynamic(tmp_path: Path) -> None:
    """
    An output whose frame count the export fixed (50) while config.pbtxt declares it -1 gets the frame dimension of
    the sequence input, the intermediate shapes are dropped, and a second repair changes nothing; an image model
    whose outputs match its configuration is left as it is.

    Args:
        tmp_path (Path): The temporary folder of the test.
    """
    sequence_folder = tmp_path / "sequence_labeller_onnx"
    write_served_onnx_model(model_folder=sequence_folder, input_shape=["batch", "images_frames", 3, 2, 2],
                            output_shape=["batch", 50, 3], configured_output_dimensions="-1, 3")
    assert repair_onnx_output_frames(model_folder=sequence_folder) == ["class_probabilities axis 1: 50"]
    repaired_model = onnx.load(str(sequence_folder / "1" / "model.onnx"))
    output_dimensions = repaired_model.graph.output[0].type.tensor_type.shape.dim
    assert [dimension.dim_param or dimension.dim_value for dimension in output_dimensions] == ["batch",
                                                                                             "images_frames", 3]
    assert len(repaired_model.graph.value_info) == 0
    assert repair_onnx_output_frames(model_folder=sequence_folder) == []

    # Image model: its fixed number of classes is the configured one
    image_folder = tmp_path / "image_classifier_onnx"
    write_served_onnx_model(model_folder=image_folder, input_shape=["batch", 3, 2, 2], output_shape=["batch", 3],
                            configured_output_dimensions="3")
    assert repair_onnx_output_frames(model_folder=image_folder) == []
    assert len(onnx.load(str(image_folder / "1" / "model.onnx")).graph.value_info) == 1
