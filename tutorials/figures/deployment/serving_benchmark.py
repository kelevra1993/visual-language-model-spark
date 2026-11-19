"""
Figures and facts of the serving benchmark chapter of the deployment tutorials, made at every build by
tutorials/figures/deployment/figures.py from the results of the serving benchmark
(documentation/architecture_benchmarking/serving_benchmark_plan.md): the committed copies of its speed results and test
metrics (serving_benchmark_results.csv and serving_benchmark_test_metrics.csv, the repository_files of
tutorials/tutorial_sources.yaml), the table of the false positives of the detector by request size written in the plan,
and the committed copies of the outputs of the detector exported again with its NMS without offsets
(serving_benchmark_nms_fix_test_metrics.csv and the perf_analyzer reports of serving_benchmark_nms_fix_loads/). It
draws the latency and throughput of every row of every model, the test metric of every served format against the
eager model in points, and the images per second of TensorRT FP16 under the four server settings, and gives the
numbers the narration quotes as {{deployment.<name>}}: the rows as the plan reads them, the first requests building the
TensorRT engines, the cost of the dynamic batcher, the batches of the eager detector, and the float16 overflow of the
group offsets of its NMS.
"""

import csv
import json
import math
import re

import matplotlib.lines
import matplotlib.ticker
import numpy

from matplotlib import pyplot
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tutorials.figures.figure_utilities import (CHART_WIDTH, DOTS_PER_INCH, GRID_COLOUR, POINTS_PER_PIXEL,
                                                SURFACE_COLOUR, TEXT_PRIMARY_COLOUR, TEXT_SECONDARY_COLOUR,
                                                compute_pixel_box, find_repository_file, find_run_folder,
                                                format_thousands, read_run_configuration, register_fonts, round_fact,
                                                save_figure, style_chart_panel)
from tutorial_builder.figure_context import find_repository_root
from tutorial_builder.layout import NAMED_COLOURS
from tutorials.figures.foundations.code_values import read_main_block_value

# Models of the benchmark in the order of the plan, with their short name on the figures and in the facts, and the
# run of tutorial_sources.yaml every model was exported from: the classifier measured is the run of the 9-class
# version of the synthetic shapes (serving_benchmark_plan.md), not the classification run of the tutorials
MODEL_NAMES = {"classification": "CLS", "object_detection": "DET", "sequence_labelling": "M2M"}
MODEL_RUN_KEYS = {"CLS": "classification_nine_classes", "DET": "object_detection_benchmark_joint_a",
                  "M2M": "sequence_labelling"}

# Driver of the benchmark, whose main block sets the calls of the rows in the process
BENCHMARK_SCRIPT = Path("scripts") / "deployment" / "benchmark_serving_formats.py"

# Class index of the background in the label dictionary of a detector, not a class of its detections
BACKGROUND_CLASS_INDEX = 0

# Rows of the benchmark in the order of the plan, with their label on the figures and their colour, the rows in the
# process in greys, the served ones in the series colours
ROW_LABELS = {"in_process_eager": "Eager, in the process", "in_process_torchscript": "TorchScript, in the process",
              "triton_torchscript": "Triton TorchScript", "triton_onnxruntime_cuda": "Triton ONNX Runtime CUDA",
              "triton_tensorrt_fp32": "Triton TensorRT FP32", "triton_tensorrt_fp16": "Triton TensorRT FP16"}
ROW_COLOURS = {"in_process_eager": "dim", "in_process_torchscript": "text", "triton_torchscript": "blue",
               "triton_onnxruntime_cuda": "aqua", "triton_tensorrt_fp32": "violet", "triton_tensorrt_fp16": "yellow"}

# Short name of every row in the facts
ROW_FACT_NAMES = {"in_process_eager": "eager", "in_process_torchscript": "torchscript_in_process",
                  "triton_torchscript": "torchscript", "triton_onnxruntime_cuda": "onnxruntime_cuda",
                  "triton_tensorrt_fp32": "tensorrt_fp32", "triton_tensorrt_fp16": "tensorrt_fp16"}

# Reference row of the test metric, and the served rows compared with it
EAGER_ROW = "in_process_eager"
SERVED_ROWS = ["triton_torchscript", "triton_onnxruntime_cuda", "triton_tensorrt_fp32", "triton_tensorrt_fp16"]

# Rows of the detector measured again with its NMS without offsets, with the perf_analyzer report of every load
FIXED_DETECTOR_ROWS = {"triton_tensorrt_fp32": "tensorrt_fp32", "triton_tensorrt_fp16": "tensorrt_fp16"}

# Transports of the measurements: the rows in the process are called in the process, the served ones are read over
# system shared memory (perf_analyzer), gRPC (the tritonclient client) being the second transport of the plan
IN_PROCESS_TRANSPORT = "in_process"
SHARED_MEMORY_TRANSPORT = "system_shared_memory"
GRPC_TRANSPORT = "grpc"

# Server settings of the configuration the export writes (dynamic batching on, one instance), and the four settings
# measured for TensorRT FP16 at batch 1, with their labels
EXPORTED_SERVER_SETTING = ("True", "1")
SERVER_SETTINGS = {("True", "1"): "batching on, 1 instance (exported)", ("False", "1"): "batching off, 1 instance",
                   ("True", "2"): "batching on, 2 instances", ("False", "2"): "batching off, 2 instances"}
SERVER_SETTING_COLOURS = {("True", "1"): "yellow", ("False", "1"): "blue", ("True", "2"): "violet",
                          ("False", "2"): "aqua"}

# Headline metric of the detector on the figures, and the recall the narration compares with it
DETECTOR_RECALL_METRIC = "mAR@0.5/all"
DETECTOR_STRICT_METRIC = "mAP@[0.5:0.95]/area"

# Columns of the perf_analyzer reports: concurrency, inferences per second and the p50 latency in microseconds
PERF_ANALYZER_THROUGHPUT_COLUMN = "Inferences/Second"
PERF_ANALYZER_P50_COLUMN = "p50 latency"
MICROSECONDS_PER_MILLISECOND = 1000.0

# Heading of the table of the plan giving the false positives of the detector by request size, and its row labels
FALSE_POSITIVE_TABLE_HEADER = "| Row | Request | True positives | False positives |"

# Gap of the test metric, in rates between 0 and 1, above which a served format is drawn as a loss on the parity table
PARITY_TOLERANCE = 0.001

# Height of the speed figure, two charts of the height of a card stacked, viewed one after the other by the camera
SPEED_FIGURE_HEIGHT = 1400

# Headroom of the bar charts the scenes draw from the facts, above their largest value, for the value labels
BAR_HEADROOM = 1.15


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """
    Reads every row of a results file of the serving benchmark.

    Args:
        csv_path (Path): The CSV file.

    Returns:
        List[Dict[str, str]]: The rows, keyed by column.
    """
    with open(file=csv_path, encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def find_single_row(rows: List[Dict[str, str]], **column_values: str) -> Dict[str, str]:
    """
    Finds the one measurement of the results matching every given column value, so that a missing or repeated
    measurement stops the build instead of quoting another one.

    Args:
        rows (List[Dict[str, str]]): The rows of serving_benchmark_results.csv.
        **column_values (str): The value of every column the measurement must have.

    Returns:
        Dict[str, str]: The measurement.

    Raises:
        ValueError: If no measurement or several match.
    """
    matching_rows = [row for row in rows if all(row[column] == value for column, value in column_values.items())]
    if len(matching_rows) != 1:
        raise ValueError(f"{len(matching_rows)} measurements of the serving benchmark match {column_values}, "
                         f"expected one.")
    return matching_rows[0]


def find_load(rows: List[Dict[str, str]], model: str, row_name: str, transport: str, batch_size: int,
              concurrency: int, server_setting: Tuple[str, str]) -> Dict[str, str]:
    """
    Finds a load measurement of the speed sweep, the rows in the process having no server setting.

    Args:
        rows (List[Dict[str, str]]): The rows of serving_benchmark_results.csv.
        model (str): The model, a key of MODEL_NAMES.
        row_name (str): The row, a key of ROW_LABELS.
        transport (str): The transport of the measurement.
        batch_size (int): The client batch size.
        concurrency (int): The requests in flight.
        server_setting (Tuple[str, str]): The dynamic batching and instance count of the served rows.

    Returns:
        Dict[str, str]: The measurement.
    """
    is_in_process = row_name.startswith("in_process")
    return find_single_row(rows=rows, model=model, row=row_name, kind="load",
                           transport=IN_PROCESS_TRANSPORT if is_in_process else transport, batch_size=str(batch_size),
                           concurrency=str(concurrency), dynamic_batching="" if is_in_process else server_setting[0],
                           instance_count="" if is_in_process else server_setting[1])


def read_perf_analyzer_report(report_path: Path) -> Dict[str, float]:
    """
    Reads the throughput and the p50 latency of a perf_analyzer report of one load (one line of measurements).

    Args:
        report_path (Path): The CSV report of perf_analyzer.

    Returns:
        Dict[str, float]: 'images_per_second' and 'p50_milliseconds'.

    Raises:
        ValueError: If the report does not hold exactly one measurement.
    """
    report_rows = read_csv_rows(csv_path=report_path)
    if len(report_rows) != 1:
        raise ValueError(f"{report_path} holds {len(report_rows)} measurements, expected one.")
    return {"images_per_second": float(report_rows[0][PERF_ANALYZER_THROUGHPUT_COLUMN]),
            "p50_milliseconds": float(report_rows[0][PERF_ANALYZER_P50_COLUMN]) / MICROSECONDS_PER_MILLISECOND}


def read_markdown_table(text: str, header: str) -> List[Dict[str, str]]:
    """
    Reads the table of a markdown document whose header line starts with the given text, its cells stripped.

    Args:
        text (str): The markdown document.
        header (str): The start of the header line of the table.

    Returns:
        List[Dict[str, str]]: The rows of the table, keyed by the header cells.

    Raises:
        ValueError: If no line starts with the header.
    """
    lines = text.split("\n")
    header_indices = [line_index for line_index, line in enumerate(lines) if line.startswith(header)]
    if len(header_indices) != 1:
        raise ValueError(f"{len(header_indices)} tables of the plan start with {header!r}, expected one.")

    # Cells of the header, then of every row until the first line that is not a table line, the separator skipped
    header_cells = [cell.strip() for cell in lines[header_indices[0]].strip("|").split("|")]
    table_rows = []
    for line in lines[header_indices[0] + 2:]:
        if not line.startswith("|"):
            break
        table_rows.append(dict(zip(header_cells, [cell.strip() for cell in line.strip("|").split("|")])))
    return table_rows


def parse_count(cell: str) -> int:
    """
    Reads a count written with thousands separators in a table of the plan, such as '1,531'.

    Args:
        cell (str): The cell.

    Returns:
        int: The count.
    """
    return int(cell.replace(",", ""))


def describe_speed(result_rows: List[Dict[str, str]], batch_sizes: List[int]) -> Dict[str, Any]:
    """
    Gives the speed facts of every model and row with the exported server configuration and one client: the p50 of
    every batch size and the images per second at the largest batch over shared memory (in the process for the rows
    in the process), the images per second at the largest batch over gRPC, the memory, and the first requests.

    Args:
        result_rows (List[Dict[str, str]]): The rows of serving_benchmark_results.csv.
        batch_sizes (List[int]): The client batch sizes of the sweep, in increasing order.

    Returns:
        Dict[str, Any]: The facts speed.<model>.<row>.* and first_request.<model>.<row>.batch_<size>.
    """
    facts = {}
    largest_batch_size = batch_sizes[-1]
    for model, model_name in MODEL_NAMES.items():
        for row_name, row_fact_name in ROW_FACT_NAMES.items():
            prefix = f"speed.{model_name}.{row_fact_name}"

            # Latency of every batch size and throughput at the largest one, one request at a time
            for batch_size in batch_sizes:
                load = find_load(rows=result_rows, model=model, row_name=row_name, transport=SHARED_MEMORY_TRANSPORT,
                                 batch_size=batch_size, concurrency=1, server_setting=EXPORTED_SERVER_SETTING)
                facts[f"{prefix}.p50_batch_{batch_size}"] = round_fact(value=float(load["p50_milliseconds"]))
                facts[f"{prefix}.images_per_second_batch_{batch_size}"] = round_fact(
                    value=float(load["images_per_second"]))
            facts[f"{prefix}.memory_gigabytes"] = round_fact(value=float(load["memory_gigabytes"]))

            # Served rows: gRPC at the largest batch, and the first requests building the engines
            if row_name.startswith("triton"):
                grpc_load = find_load(rows=result_rows, model=model, row_name=row_name, transport=GRPC_TRANSPORT,
                                      batch_size=largest_batch_size, concurrency=1,
                                      server_setting=EXPORTED_SERVER_SETTING)
                facts[f"{prefix}.grpc_images_per_second_batch_{largest_batch_size}"] = round_fact(
                    value=float(grpc_load["images_per_second"]))
                for batch_size in batch_sizes:
                    first_request = find_single_row(rows=result_rows, model=model, row=row_name, kind="first_request",
                                                    batch_size=str(batch_size))
                    facts[f"first_request.{model_name}.{row_fact_name}.batch_{batch_size}"] = round_fact(
                        value=float(first_request["first_request_seconds"]))

        # Top of the axis of the first requests drawn by the serving chapter, above the longest wait of the model
        facts[f"first_request.{model_name}.axis_maximum"] = round_fact(value=BAR_HEADROOM * max(
            value for name, value in facts.items() if name.startswith(f"first_request.{model_name}.")))

        # Gains of TensorRT FP16 at the largest batch over the eager model and over TensorRT FP32
        fp16_throughput = facts[f"speed.{model_name}.tensorrt_fp16.images_per_second_batch_{largest_batch_size}"]
        for reference_name in ["eager", "tensorrt_fp32"]:
            reference_throughput = facts[f"speed.{model_name}.{reference_name}.images_per_second_batch_"
                                         f"{largest_batch_size}"]
            facts[f"speed.{model_name}.fp16_over_{reference_name}"] = round_fact(
                value=fp16_throughput / reference_throughput)
    return facts


def describe_server_settings(result_rows: List[Dict[str, str]], concurrencies: List[int]) -> Dict[str, Any]:
    """
    Gives the images per second of TensorRT FP16 at batch 1 under the four server settings and every concurrency, over
    both transports, and the queue time of the server with one client with and without the dynamic batcher.

    Args:
        result_rows (List[Dict[str, str]]): The rows of serving_benchmark_results.csv.
        concurrencies (List[int]): The concurrencies of the sweep, in increasing order.

    Returns:
        Dict[str, Any]: The facts settings.<model>.<transport>.<setting>.concurrency_<n> and
            settings.<model>.queue_milliseconds_batching_<on|off>.
    """
    facts = {}
    for model, model_name in MODEL_NAMES.items():
        for transport_name, transport in [("shared_memory", SHARED_MEMORY_TRANSPORT), ("grpc", GRPC_TRANSPORT)]:
            for server_setting in SERVER_SETTINGS:
                setting_name = (f"batching_{'on' if server_setting[0] == 'True' else 'off'}_instances_"
                                f"{server_setting[1]}")
                for concurrency in concurrencies:
                    load = find_load(rows=result_rows, model=model, row_name="triton_tensorrt_fp16",
                                     transport=transport, batch_size=1, concurrency=concurrency,
                                     server_setting=server_setting)
                    facts[f"settings.{model_name}.{transport_name}.{setting_name}.concurrency_{concurrency}"] = (
                        round_fact(value=float(load["images_per_second"])))

        # Queue time of one client over shared memory, the delay of the dynamic batcher paid by every request
        for batching_name, server_setting in [("on", ("True", "1")), ("off", ("False", "1"))]:
            load = find_load(rows=result_rows, model=model, row_name="triton_tensorrt_fp16",
                             transport=SHARED_MEMORY_TRANSPORT, batch_size=1, concurrency=1,
                             server_setting=server_setting)
            facts[f"settings.{model_name}.queue_milliseconds_batching_{batching_name}"] = round_fact(
                value=float(load["server_queue_microseconds"]) / MICROSECONDS_PER_MILLISECOND)
            facts[f"settings.{model_name}.p50_milliseconds_batching_{batching_name}"] = round_fact(
                value=float(load["p50_milliseconds"]))
    return facts


def describe_test_metrics(metric_rows: List[Dict[str, str]], fixed_metric_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Gives the headline test metric of every model and row, its gap to the eager model as a rate difference, and for
    the detector its recall and strict mAP, before and after its NMS without offsets.

    Args:
        metric_rows (List[Dict[str, str]]): The rows of serving_benchmark_test_metrics.csv.
        fixed_metric_rows (List[Dict[str, str]]): The rows of serving_benchmark_nms_fix_test_metrics.csv, the detector
            exported again.

    Returns:
        Dict[str, Any]: The facts metric.<model>.* and metric.DET_fixed.*.
    """
    facts = {}
    for model, model_name in MODEL_NAMES.items():
        headline_rows = {row["row"]: row for row in metric_rows if row["model"] == model and row["headline"] == "True"}
        facts[f"metric.{model_name}.name"] = headline_rows[EAGER_ROW]["metric"]
        facts[f"metric.{model_name}.evaluated"] = headline_rows[EAGER_ROW]["evaluated_samples"]
        eager_value = float(headline_rows[EAGER_ROW]["value"])
        facts[f"metric.{model_name}.eager"] = round_fact(value=eager_value)
        for row_name in SERVED_ROWS:
            row_value = float(headline_rows[row_name]["value"])
            facts[f"metric.{model_name}.{ROW_FACT_NAMES[row_name]}"] = round_fact(value=row_value)
            facts[f"metric.{model_name}.{ROW_FACT_NAMES[row_name]}_gap"] = round_fact(value=row_value - eager_value)

    # Detector: recall and strict mAP of the eager model and of TensorRT FP16 before the fix
    for metric_key, metric_name in [("recall", DETECTOR_RECALL_METRIC), ("strict", DETECTOR_STRICT_METRIC)]:
        values = {row["row"]: float(row["value"]) for row in metric_rows
                  if row["model"] == "object_detection" and row["metric"] == metric_name}
        facts[f"metric.DET.{metric_key}_name"] = metric_name
        facts[f"metric.DET.{metric_key}.eager"] = round_fact(value=values[EAGER_ROW])
        facts[f"metric.DET.{metric_key}.tensorrt_fp16_gap"] = round_fact(
            value=values["triton_tensorrt_fp16"] - values[EAGER_ROW])

    # Detector exported again: its headline metric against the same eager reference
    headline_name = facts["metric.DET.name"]
    fixed_values = {row["row"]: float(row["value"]) for row in fixed_metric_rows if row["metric"] == headline_name}
    if EAGER_ROW not in fixed_values or round_fact(value=fixed_values[EAGER_ROW]) != facts["metric.DET.eager"]:
        raise ValueError(f"The test metrics of the detector exported again do not hold the eager {headline_name} of "
                         f"the benchmark ({facts['metric.DET.eager']}).")
    for row_name, row_fact_name in [(name, ROW_FACT_NAMES[name]) for name in FIXED_DETECTOR_ROWS]:
        facts[f"metric.DET_fixed.{row_fact_name}"] = round_fact(value=fixed_values[row_name])
        facts[f"metric.DET_fixed.{row_fact_name}_gap"] = round_fact(
            value=fixed_values[row_name] - fixed_values[EAGER_ROW])
    return facts


def describe_fixed_detector_speed(loads_folder: Path) -> Dict[str, Any]:
    """
    Gives the speed of the detector exported again with its NMS without offsets, from the perf_analyzer reports of
    its loads over shared memory: the p50 at batch 1 and the images per second at batch 16 of both precisions with one
    client, and the images per second of TensorRT FP16 with single images at every concurrency, batching on and off.

    Args:
        loads_folder (Path): The serving_benchmark_nms_fix_loads folder, the loads of the detector exported again.

    Returns:
        Dict[str, Any]: The facts fixed.<row>.* and fixed.settings.batching_<on|off>.concurrency_<n>.
    """
    facts = {}
    for row_fact_name in FIXED_DETECTOR_ROWS.values():
        single_image = read_perf_analyzer_report(report_path=loads_folder / f"{row_fact_name}_batch_1.csv")
        full_batch = read_perf_analyzer_report(report_path=loads_folder / f"{row_fact_name}_batch_16.csv")
        facts[f"fixed.{row_fact_name}.p50_batch_1"] = round_fact(value=single_image["p50_milliseconds"])
        facts[f"fixed.{row_fact_name}.images_per_second_batch_16"] = round_fact(value=full_batch["images_per_second"])
        facts[f"fixed.{row_fact_name}.p50_batch_16"] = round_fact(value=full_batch["p50_milliseconds"])
    for batching_name, batching_value in [("on", "True"), ("off", "False")]:
        for report_path in sorted(loads_folder.glob(f"tensorrt_fp16_batching_{batching_value}_concurrency_*.csv")):
            concurrency = int(re.search(pattern=r"concurrency_(\d+)", string=report_path.name).group(1))
            facts[f"fixed.settings.batching_{batching_name}.concurrency_{concurrency}"] = round_fact(
                value=read_perf_analyzer_report(report_path=report_path)["images_per_second"])
    return facts


def describe_group_offsets(detection_run_folder: Path) -> Dict[str, Any]:
    """
    Gives the numbers of the float16 overflow of the group offsets of the NMS of the detector: every group of image and
    class moved by its index times the largest coordinate plus one (suppress_groups_apart), the largest coordinate
    being the input side of the clipped boxes, so that the moved coordinates pass the largest float16 value from the
    group whose offset alone passes it, among the groups of the image given by the classes of the detector; and the
    side of the smallest square box whose area in pixels passes that value, the overflow of the DIoU-NMS areas.

    Args:
        detection_run_folder (Path): The reference run of the detection benchmark, served by the benchmark.

    Returns:
        Dict[str, Any]: The facts offsets.*.
    """
    # Input side and foreground classes of the detector, one group of the exported NMS per image and class
    run_configuration = read_run_configuration(run_folder=detection_run_folder)
    image_size = run_configuration["Data"]["image_settings"]["size"]
    label_dictionary_path = detection_run_folder / "label_dictionary.json"
    label_dictionary = json.loads(label_dictionary_path.read_text(encoding="utf-8"))
    foreground_class_count = len({class_index for class_index in label_dictionary.values()
                                  if class_index != BACKGROUND_CLASS_INDEX})

    # Offset of every group and the first group whose offset passes the largest float16 value
    float16_maximum = float(numpy.finfo(numpy.float16).max)
    group_offset = image_size + 1
    first_overflowing_group = math.floor(float16_maximum / group_offset) + 1
    return {"offsets.image_size": image_size, "offsets.group_offset": group_offset,
            "offsets.float16_maximum": int(float16_maximum), "offsets.groups_per_image": foreground_class_count,
            "offsets.first_overflowing_group": first_overflowing_group,
            "offsets.first_overflowing_image": math.ceil(first_overflowing_group / foreground_class_count),
            "offsets.area_overflow_side": math.ceil(math.sqrt(float16_maximum))}


def describe_benchmark_models(experiments_folder: Path) -> Dict[str, Any]:
    """
    Gives the input size and the classes of the three models of the benchmark, read from their runs (the background of
    the detector left out), and the calls of the rows in the process from the main block of the driver.

    Args:
        experiments_folder (Path): The folder of the real runs.

    Returns:
        Dict[str, Any]: The facts models.<model>.image_size, models.<model>.class_count and the in_process.* calls.
    """
    facts = {}
    for model_name, run_key in MODEL_RUN_KEYS.items():
        run_folder = find_run_folder(experiments_folder=experiments_folder, run_key=run_key)
        label_dictionary = json.loads((run_folder / "label_dictionary.json").read_text(encoding="utf-8"))
        class_indices = set(label_dictionary.values())
        if model_name == "DET":
            class_indices.discard(BACKGROUND_CLASS_INDEX)
        facts[f"models.{model_name}.image_size"] = read_run_configuration(run_folder=run_folder)["Data"][
            "image_settings"]["size"]
        facts[f"models.{model_name}.class_count"] = len(class_indices)
    script_path = find_repository_root() / BENCHMARK_SCRIPT
    facts["in_process.timed_calls"] = read_main_block_value(script_path=script_path,
                                                            variable_name="number_in_process_iterations")
    facts["in_process.warmup_calls"] = read_main_block_value(script_path=script_path,
                                                             variable_name="number_in_process_warmup_iterations")
    return facts


def describe_false_positives(plan_text: str) -> Dict[str, Any]:
    """
    Reads the false positives of the eager detector and of its TensorRT engines on the first test images, sent one
    image per request and sixteen per request, from the table of the plan.

    Args:
        plan_text (str): The text of serving_benchmark_plan.md.

    Returns:
        Dict[str, Any]: The facts false_positives.<row>_<request>.
    """
    facts = {}
    for table_row in read_markdown_table(text=plan_text, header=FALSE_POSITIVE_TABLE_HEADER):
        row_name = next(row_name for row_name, label in [("eager", "eager"), ("tensorrt_fp16", "FP16"),
                                                         ("tensorrt_fp32", "FP32")] if label in table_row["Row"])
        request_size = parse_count(cell=table_row["Request"].split(" ")[0])
        facts[f"false_positives.{row_name}_request_{request_size}"] = parse_count(cell=table_row["False positives"])
    return facts


def style_bar_panel(axes: Any, title: str, x_label: str, maximum: float) -> None:
    """
    Styles a horizontal bar panel of the speed figure, its value axis starting at zero with room for the value labels.

    Args:
        axes (Any): The panel.
        title (str): The title of the panel.
        x_label (str): The name of the value axis.
        maximum (float): The largest value drawn.
    """
    style_chart_panel(axes=axes, title=title, x_label=x_label, y_label=None)
    axes.set_xlim(left=0.0, right=maximum * 1.32)
    axes.grid(axis="y", visible=False)
    axes.invert_yaxis()


def draw_speed_figure(facts: Dict[str, Any], output_path: Path, largest_batch_size: int) -> Dict[str, Any]:
    """
    Draws the speed of every row of every model with the exported configuration and one client, over shared memory:
    the p50 of one image per request on top, the images per second of batches of the largest size below, the
    detector exported again with its NMS without offsets drawn hatched after its rows.

    Args:
        facts (Dict[str, Any]): The facts of describe_speed and describe_fixed_detector_speed.
        output_path (Path): The PNG file.
        largest_batch_size (int): The largest client batch size.

    Returns:
        Dict[str, Any]: The pixel boxes of the panels, figures.speed.<metric>.<model>_box, and the views of the two
            charts.
    """
    register_fonts()
    figure, axes_grid = pyplot.subplots(nrows=2, ncols=3, figsize=(CHART_WIDTH / DOTS_PER_INCH,
                                                                   SPEED_FIGURE_HEIGHT / DOTS_PER_INCH),
                                        dpi=DOTS_PER_INCH, facecolor=SURFACE_COLOUR)
    figure.subplots_adjust(left=0.17, right=0.985, top=0.94, bottom=0.06, wspace=0.12, hspace=0.32)
    boxes = {}
    metrics = [("p50", "p50_batch_1", "p50 in milliseconds, one image per request", "{:.1f}"),
               ("throughput", f"images_per_second_batch_{largest_batch_size}",
                f"images per second, batches of {largest_batch_size}", "{:,.0f}")]
    for metric_index, (metric_key, fact_suffix, axis_label, value_format) in enumerate(metrics):
        for model_index, model_name in enumerate(MODEL_NAMES.values()):
            axes = axes_grid[metric_index][model_index]

            # Bars of every row, then the detector exported again hatched after its rows
            labels, values, colours, hatches = [], [], [], []
            for row_name, row_fact_name in ROW_FACT_NAMES.items():
                labels.append(ROW_LABELS[row_name])
                values.append(facts[f"speed.{model_name}.{row_fact_name}.{fact_suffix}"])
                colours.append(NAMED_COLOURS[ROW_COLOURS[row_name]])
                hatches.append("")
            if model_name == "DET":
                for row_name, row_fact_name in FIXED_DETECTOR_ROWS.items():
                    labels.append(f"{ROW_LABELS[row_name]}, NMS fixed")
                    values.append(facts[f"fixed.{row_fact_name}.{fact_suffix}"])
                    colours.append(NAMED_COLOURS[ROW_COLOURS[row_name]])
                    hatches.append("//")
            positions = numpy.arange(len(values), dtype=float)
            bar_colours = ["#%02x%02x%02x" % colour for colour in colours]
            bars = axes.barh(y=positions, width=values, height=0.7, color=bar_colours, edgecolor=SURFACE_COLOUR)
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)

            # Value at the end of every bar, the panel styled with the rows named on the first column only
            for position, value in zip(positions, values):
                axes.text(x=value, y=position, s="  " + value_format.format(value), va="center", ha="left",
                          color=TEXT_PRIMARY_COLOUR, fontsize=16 * POINTS_PER_PIXEL)
            # The sequence labeller takes one sequence per request and counts frames
            if model_name == "M2M":
                axis_label = axis_label.replace("one image", "one sequence").replace("images", "frames")
            style_bar_panel(axes=axes, title=model_name, x_label=axis_label, maximum=max(values))
            axes.xaxis.set_major_formatter(pyplot.FuncFormatter(format_thousands) if metric_key == "throughput" else
                                           matplotlib.ticker.StrMethodFormatter(fmt="{x:g}"))
            axes.set_yticks(ticks=positions)
            axes.set_yticklabels(labels=labels if model_index == 0 else [""] * len(labels),
                                 fontsize=15 * POINTS_PER_PIXEL)
            if model_name == "DET":
                for position in positions[len(ROW_FACT_NAMES):]:
                    axes.text(x=0.0, y=position, s="NMS fixed  ", va="center", ha="right",
                              color=TEXT_SECONDARY_COLOUR, fontsize=15 * POINTS_PER_PIXEL)
            boxes[f"figures.speed.{metric_key}.{model_name}_box"] = compute_pixel_box(
                figure=figure, axes=axes, with_labels=True, margin=10)
    save_figure(figure=figure, output_path=output_path)
    pyplot.close(fig=figure)

    # Views of the camera: the latency chart, then the throughput chart, each the height of a card
    half_height = SPEED_FIGURE_HEIGHT // 2
    boxes["figures.speed.p50_view"] = [0, 0, CHART_WIDTH, half_height]
    boxes["figures.speed.throughput_view"] = [0, half_height, CHART_WIDTH, SPEED_FIGURE_HEIGHT]
    return boxes


def format_gap(gap: Optional[float]) -> str:
    """
    Writes the gap of a served format to the eager model in points, as the parity table shows it.

    Args:
        gap (Optional[float]): The gap as a rate difference, or None when the format was not measured.

    Returns:
        str: The gap such as '-7.88', '+0.01' or '0.00', or a dash.
    """
    if gap is None:
        return "-"
    points = 100.0 * gap
    return "0.00" if abs(points) < 0.005 else f"{points:+.2f}"


def convert_pixels_to_figure(chart_height: int, x_pixels: float, y_pixels: float) -> Tuple[float, float]:
    """
    Converts a position in pixels from the top left corner of a chart of the width of a card into the figure
    coordinates of matplotlib, from the bottom left corner, for the texts of the parity table.

    Args:
        chart_height (int): The height of the chart in pixels.
        x_pixels (float): The distance from the left edge.
        y_pixels (float): The distance from the top edge.

    Returns:
        Tuple[float, float]: The figure coordinates.
    """
    return x_pixels / CHART_WIDTH, 1.0 - y_pixels / chart_height


def draw_parity_table(facts: Dict[str, Any], output_path: Path, chart_height: int,
                      maximum_request_size: int) -> Dict[str, Any]:
    """
    Draws the test metric of every served format against the eager model, in points, one column per model and one for
    the detector exported again with its NMS without offsets, the eager value in the first row and every gap larger
    than PARITY_TOLERANCE in red.

    Args:
        facts (Dict[str, Any]): The facts of describe_test_metrics.
        output_path (Path): The PNG file.
        chart_height (int): The height of the figure in pixels.
        maximum_request_size (int): The most samples of a request of the test metric, the largest batch size.

    Returns:
        Dict[str, Any]: The pixel boxes of the columns of the table, figures.parity.<column>_box.
    """
    register_fonts()
    figure = pyplot.figure(figsize=(CHART_WIDTH / DOTS_PER_INCH, chart_height / DOTS_PER_INCH), dpi=DOTS_PER_INCH,
                           facecolor=SURFACE_COLOUR)

    # Columns: a model and its metric, the facts of its values
    columns = [("CLS", f"CLS · {facts['metric.CLS.name']}", "CLS"),
               ("DET", f"DET · {facts['metric.DET.name']}", "DET"),
               ("DET_fixed", "DET · NMS fixed", "DET_fixed"),
               ("M2M", f"M2M · {facts['metric.M2M.name']}", "M2M")]
    row_names = [EAGER_ROW] + SERVED_ROWS
    label_width, top_margin, row_height = 470, 120, 92
    column_width = (CHART_WIDTH - label_width - 30) / len(columns)
    boxes = {}

    # Header of every column and the label of every row, placed in pixels from the top left corner
    for column_index, (_, column_title, _) in enumerate(columns):
        x_centre = label_width + (column_index + 0.5) * column_width
        figure.text(*convert_pixels_to_figure(chart_height=chart_height, x_pixels=x_centre, y_pixels=top_margin - 40),
                    s=column_title, ha="center", va="center", color=TEXT_PRIMARY_COLOUR,
                    fontsize=19 * POINTS_PER_PIXEL, fontweight="semibold")
    for row_index, row_name in enumerate(row_names):
        y_centre = top_margin + (row_index + 0.5) * row_height
        figure.text(*convert_pixels_to_figure(chart_height=chart_height, x_pixels=24, y_pixels=y_centre),
                    s=ROW_LABELS[row_name], ha="left", va="center", color=TEXT_PRIMARY_COLOUR,
                    fontsize=19 * POINTS_PER_PIXEL)
        _, line_height = convert_pixels_to_figure(chart_height=chart_height, x_pixels=0,
                                                  y_pixels=y_centre + row_height / 2)
        figure.add_artist(matplotlib.lines.Line2D(xdata=[0.01, 0.99], ydata=[line_height, line_height],
                                                  color=GRID_COLOUR, linewidth=POINTS_PER_PIXEL))

        # Cells: the eager value as a percentage, the gaps in points, a dash where the row was not measured
        for column_index, (column_key, _, fact_model) in enumerate(columns):
            x_centre = label_width + (column_index + 0.5) * column_width
            if row_name == EAGER_ROW:
                eager_model = "DET" if column_key == "DET_fixed" else fact_model
                cell_text, cell_colour = f"{100.0 * facts[f'metric.{eager_model}.eager']:.2f} %", TEXT_SECONDARY_COLOUR
            else:
                # The detector exported again was only measured with TensorRT, every other cell must have its gap
                is_measured = column_key != "DET_fixed" or row_name in FIXED_DETECTOR_ROWS
                gap = facts[f"metric.{fact_model}.{ROW_FACT_NAMES[row_name]}_gap"] if is_measured else None
                cell_text = format_gap(gap=gap) + ("" if gap is None else " points")
                is_loss = gap is not None and gap < -PARITY_TOLERANCE
                cell_colour = "#%02x%02x%02x" % NAMED_COLOURS["red"] if is_loss else TEXT_PRIMARY_COLOUR
            figure.text(*convert_pixels_to_figure(chart_height=chart_height, x_pixels=x_centre, y_pixels=y_centre),
                        s=cell_text, ha="center", va="center", color=cell_colour, fontsize=22 * POINTS_PER_PIXEL,
                        family="Hack")
    for column_index, (column_key, _, _) in enumerate(columns):
        left = int(label_width + column_index * column_width)
        boxes[f"figures.parity.{column_key}_box"] = [left, top_margin - 75, int(left + column_width),
                                                     int(top_margin + len(row_names) * row_height + 5)]

    # Note under the table
    figure.text(*convert_pixels_to_figure(chart_height=chart_height, x_pixels=24, y_pixels=chart_height - 40),
                s=f"test split through each row, requests of at most {maximum_request_size} samples · gap to the eager "
                  f"model in points · red: a loss above {100.0 * PARITY_TOLERANCE:g} points",
                ha="left", va="center", color=TEXT_SECONDARY_COLOUR, fontsize=16 * POINTS_PER_PIXEL)
    save_figure(figure=figure, output_path=output_path)
    pyplot.close(fig=figure)
    return boxes


def draw_server_settings_figure(facts: Dict[str, Any], output_path: Path, concurrencies: List[int],
                                chart_height: int) -> Dict[str, Any]:
    """
    Draws the images per second of TensorRT FP16 with one image per request over shared memory under the four server
    settings, one panel per model, grouped by the requests in flight.

    Args:
        facts (Dict[str, Any]): The facts of describe_server_settings.
        output_path (Path): The PNG file.
        concurrencies (List[int]): The concurrencies of the sweep.
        chart_height (int): The height of the figure in pixels.

    Returns:
        Dict[str, Any]: The pixel boxes of the panels, figures.settings.<model>_box.
    """
    register_fonts()
    figure, axes_row = pyplot.subplots(nrows=1, ncols=3, figsize=(CHART_WIDTH / DOTS_PER_INCH,
                                                                  chart_height / DOTS_PER_INCH),
                                       dpi=DOTS_PER_INCH, facecolor=SURFACE_COLOUR)
    figure.subplots_adjust(left=0.06, right=0.99, top=0.78, bottom=0.12, wspace=0.22)
    boxes = {}
    bar_width = 0.2
    for model_index, model_name in enumerate(MODEL_NAMES.values()):
        axes = axes_row[model_index]
        group_positions = numpy.arange(len(concurrencies), dtype=float)
        maximum = 0.0
        for setting_index, (server_setting, setting_label) in enumerate(SERVER_SETTINGS.items()):
            setting_name = f"batching_{'on' if server_setting[0] == 'True' else 'off'}_instances_{server_setting[1]}"
            values = [facts[f"settings.{model_name}.shared_memory.{setting_name}.concurrency_{concurrency}"]
                      for concurrency in concurrencies]
            maximum = max(maximum, max(values))
            axes.bar(x=group_positions + (setting_index - 1.5) * bar_width, height=values, width=bar_width * 0.92,
                     color="#%02x%02x%02x" % NAMED_COLOURS[SERVER_SETTING_COLOURS[server_setting]],
                     label=setting_label if model_index == 0 else None)
        style_chart_panel(axes=axes, title=model_name, x_label="requests in flight, one image or sequence each",
                          y_label=None)
        # Title of the panel: frames for the sequence labeller, the detector measured before its NMS fix
        panel_titles = {"CLS": "CLS · images per second", "DET": "DET · images per second, unfixed NMS",
                        "M2M": "M2M · frames per second"}
        axes.set_title(label=panel_titles[model_name], color=TEXT_PRIMARY_COLOUR, fontsize=24 * POINTS_PER_PIXEL,
                       loc="left", pad=14)
        axes.set_xticks(ticks=group_positions)
        axes.set_xticklabels(labels=[str(concurrency) for concurrency in concurrencies])
        axes.set_ylim(bottom=0.0, top=maximum * 1.1)
        axes.yaxis.set_major_formatter(pyplot.FuncFormatter(format_thousands))
        axes.grid(axis="x", visible=False)
        boxes[f"figures.settings.{model_name}_box"] = compute_pixel_box(figure=figure, axes=axes, with_labels=True,
                                                                        margin=10)
    legend = figure.legend(loc="upper center", ncol=4, frameon=False, fontsize=17 * POINTS_PER_PIXEL,
                           bbox_to_anchor=(0.5, 0.995))
    for legend_text in legend.get_texts():
        legend_text.set_color(TEXT_PRIMARY_COLOUR)
    save_figure(figure=figure, output_path=output_path)
    pyplot.close(fig=figure)
    return boxes


def make_serving_benchmark_figures(output_folder: Path, experiments_folder: Path) -> Dict[str, Any]:
    """
    Makes the figures and facts of the serving benchmark chapter.

    Args:
        output_folder (Path): The figures folder of the series.
        experiments_folder (Path): The folder of the real runs.

    Returns:
        Dict[str, Any]: The facts of every family of this module.
    """
    result_rows = read_csv_rows(csv_path=find_repository_file(file_key="serving_benchmark_results"))
    metric_rows = read_csv_rows(csv_path=find_repository_file(file_key="serving_benchmark_test_metrics"))
    plan_text = find_repository_file(file_key="serving_benchmark_plan").read_text(encoding="utf-8")
    fixed_metric_rows = read_csv_rows(csv_path=find_repository_file(file_key="serving_benchmark_nms_fix_test_metrics"))
    detection_run_folder = find_run_folder(experiments_folder=experiments_folder,
                                           run_key="object_detection_benchmark_joint_a")

    # Grid of the sweep, read from the results
    batch_sizes = sorted({int(row["batch_size"]) for row in result_rows})
    concurrencies = sorted({int(row["concurrency"]) for row in result_rows})
    frames_per_sequence = {round(float(row["images_per_second"]) / float(row["inferences_per_second"]))
                           for row in result_rows if row["model"] == "sequence_labelling" and row["kind"] == "load"
                           and row["transport"] == SHARED_MEMORY_TRANSPORT}
    if len(frames_per_sequence) != 1:
        raise ValueError(f"The sequence labelling loads count {sorted(frames_per_sequence)} frames per sequence.")
    facts = {"sweep.measurement_count": len(result_rows),
             "sweep.load_count": sum(row["kind"] == "load" for row in result_rows),
             "sweep.batch_sizes": batch_sizes, "sweep.largest_batch_size": batch_sizes[-1],
             "sweep.concurrencies": concurrencies, "sweep.largest_concurrency": concurrencies[-1],
             "sweep.frames_per_sequence": frames_per_sequence.pop(), "sweep.model_count": len(MODEL_NAMES),
             "sweep.served_row_count": len(SERVED_ROWS),
             "sweep.largest_instance_count": max(int(row["instance_count"]) for row in result_rows
                                                 if row["instance_count"]),
             "sweep.failed_count": sum(row["status"] != "ok" for row in result_rows)}

    # Facts of every family, then the figures drawn from them
    facts.update(describe_speed(result_rows=result_rows, batch_sizes=batch_sizes))
    facts.update(describe_server_settings(result_rows=result_rows, concurrencies=concurrencies))
    facts.update(describe_test_metrics(metric_rows=metric_rows, fixed_metric_rows=fixed_metric_rows))
    fixed_loads_folder = find_repository_file(file_key="serving_benchmark_nms_fix_loads")
    facts.update(describe_fixed_detector_speed(loads_folder=fixed_loads_folder))
    facts.update(describe_group_offsets(detection_run_folder=detection_run_folder))
    facts.update(describe_false_positives(plan_text=plan_text))
    facts.update(describe_benchmark_models(experiments_folder=experiments_folder))
    facts.update(draw_speed_figure(facts=facts, output_path=output_folder / "benchmark" / "serving_speed.png",
                                   largest_batch_size=batch_sizes[-1]))
    facts.update(draw_parity_table(facts=facts, output_path=output_folder / "benchmark" / "test_metric_parity.png",
                                   chart_height=700, maximum_request_size=batch_sizes[-1]))
    facts.update(draw_server_settings_figure(facts=facts,
                                             output_path=output_folder / "benchmark" / "server_settings.png",
                                             concurrencies=concurrencies, chart_height=700))
    return facts
