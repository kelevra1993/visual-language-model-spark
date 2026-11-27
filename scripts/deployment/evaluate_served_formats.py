"""
Measures the test metric of every served format of the serving benchmark (documentation/architecture_benchmarking/
serving_benchmark_plan.md) against the eager FP32 model: the test split of each benchmarked model goes through the
eager inference model in this process, then through every model of the Triton repositories written by
scripts/deployment/benchmark_serving_formats.py (TorchScript FP32, ONNX Runtime CUDA, TensorRT FP32 and FP16,
TorchScript FP16), each scored
by the test evaluation of the trainer of its task, whose predict_batch is replaced by the requests to the model (the
functions of the test split parity of vision_training/deployment/export_parity.py, for every task). Every row sends
the same requests (the real frames of each sequence alone for a sequence model) and keeps the detections the deployed
detector returns, so that the gap to the eager row measures the format and the precision only. Every
output stays outside the repository, in the benchmark folder: the evaluation files of every row and test_metrics.csv,
one line per scalar of every row, read again at a relaunch so that the rows already measured are skipped. Run as a
script from the repository root, on a machine where the speed sweep is not running, with the benchmark, runs and data
folders of the machine given as arguments (parse_evaluation_arguments), its help printed when none is given; the
models, precision settings and largest request are the constants of benchmark_serving_formats.py.
"""

import argparse
import csv
import functools
import sys
import time
import numpy as np
import torch
import tritonclient.grpc as triton_client

from pathlib import Path
from typing import Any, Callable, Dict, List

from scripts.deployment.benchmark_serving_formats import (BENCHMARK_SERVER, MAXIMUM_BATCH_SIZE, PRECISION_SETTINGS,
                                                          add_benchmark_folder_arguments, build_benchmarked_models,
                                                          get_repository_formats, get_row_name,
                                                          read_served_model_name, wait_for_idle_machine)
from vision_training.deployment.deployment_utilities import (check_served_input_names, get_triton_output_names,
                                                             load_local_inference_model, run_local_inference,
                                                             run_triton_inference)
from vision_training.deployment.export_parity import build_evaluation_trainer, evaluate_parity_row
from vision_training.deployment.export_specifications import TASK_INPUT_NAMES
from vision_training.deployment.triton_server import (load_served_model, start_triton_server, stop_triton_server,
                                                      unload_served_model)
from vision_training.evaluation.checkpoint_evaluation import CHECKPOINT_EVALUATION_FOLDER_NAME
from vision_training.trainers.base_trainer import BaseTrainer
from vision_training.utilities.os_utilities import format_path_link, print_color

# Columns of test_metrics.csv, one line per scalar of the evaluation of every row of every model
METRIC_COLUMNS = ["model", "row", "metric", "value", "headline", "evaluated_samples", "duration_seconds", "timestamp"]

# Row of the eager inference model in the process, the reference of every served row
EAGER_ROW_NAME = "in_process_eager"


def read_measured_rows(metrics_path: Path) -> set:
    """
    Gives the (model, row) pairs already in test_metrics.csv, skipped at a relaunch.

    Args:
        metrics_path (Path): The metrics file, possibly missing.

    Returns:
        set: The measured (model, row) pairs.
    """
    if not metrics_path.is_file():
        return set()
    with open(metrics_path, newline="") as metrics_file:
        return {(line["model"], line["row"]) for line in csv.DictReader(metrics_file)}


def append_row_metrics(metrics_path: Path, model_name: str, row_name: str, results: Dict[str, Any],
                       duration_seconds: float) -> None:
    """
    Appends every scalar of the evaluation of one row to test_metrics.csv as soon as the row ends, the headline metric
    of the task flagged, so that a stopped evaluation keeps the rows already measured.

    Args:
        metrics_path (Path): The metrics file, its header written when it is created.
        model_name (str): The benchmarked model, e.g. 'classification'.
        row_name (str): The row of the plan, e.g. 'triton_tensorrt_fp16'.
        results (Dict[str, Any]): The results of evaluate_parity_row.
        duration_seconds (float): The time of the evaluation of the row.
    """
    write_header = not metrics_path.is_file()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(metrics_path, mode="a", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=METRIC_COLUMNS)
        if write_header:
            writer.writeheader()
        for metric, value in results["scalars"].items():
            writer.writerow({"model": model_name, "row": row_name, "metric": metric, "value": float(value),
                             "headline": metric == results["headline_metric"],
                             "evaluated_samples": results["evaluated_samples"], "duration_seconds": duration_seconds,
                             "timestamp": timestamp})


def evaluate_row(trainer: BaseTrainer, model: Dict[str, Any], row_name: str, run_model: Callable[..., List[np.ndarray]],
                 output_names: List[str], maximum_request_size: int, evaluation_folder: Path,
                 metrics_path: Path) -> None:
    """
    Scores one row of one model on the whole test split with the evaluation of the trainer (evaluate_parity_row of
    the test split parity of the export), its predict_batch running the model of the row, its evaluation files written
    to their own folder, and appends its scalars to test_metrics.csv. The detectors keep their deployment score
    threshold (the one written into their export) instead of the lower threshold of the training evaluation, as the
    served detector returns only those detections.

    Args:
        trainer (BaseTrainer): The trainer of the model, from build_evaluation_trainer.
        model (Dict[str, Any]): The benchmarked model, with its 'name' and 'iteration'.
        row_name (str): The row of the plan.
        run_model (Callable[..., List[np.ndarray]]): Runs one request ('inputs' keyword), the eager or served model.
        output_names (List[str]): The output names of the model, in the order run_model returns them.
        maximum_request_size (int): The largest batch size of a request.
        evaluation_folder (Path): The evaluation folder of the model.
        metrics_path (Path): The metrics file.
    """
    other_processes = wait_for_idle_machine()
    if other_processes:
        print_color(color="yellow", output=f"Other GPU processes : {', '.join(other_processes)}", indent=1)
    print_color(color="blue", output=f"{model['name']} : {row_name}", add_separators=True)
    results = evaluate_parity_row(trainer=trainer, run_model=run_model, output_names=output_names,
                                  maximum_request_size=maximum_request_size, iteration=model["iteration"],
                                  row_folder=evaluation_folder / row_name, comparison=None)
    append_row_metrics(metrics_path=metrics_path, model_name=model["name"], row_name=row_name, results=results,
                       duration_seconds=results["duration_seconds"])


def run_served_evaluations(models: List[Dict[str, Any]], benchmark_folder: Path, precision_settings: List[str],
                           maximum_request_size: int) -> None:
    """
    Evaluates every row of every model in turn, skipping the rows already in test_metrics.csv: the eager inference
    model in the process first, the reference, then one Triton server per precision setting serving the models of
    its repository one at a time, with the configuration of their export (dynamic batching on, one instance) and the
    TensorRT engine cache of the speed sweep.

    Args:
        models (List[Dict[str, Any]]): The benchmarked models, with 'name', 'experiment_folder', 'iteration' and
            'test_data_folder'.
        benchmark_folder (Path): The folder of the benchmark, holding the model repositories, outside the repository.
        precision_settings (List[str]): The precision settings of the served rows, e.g. 'tensorrt_fp16'.
        maximum_request_size (int): The largest batch size of the served models, also the batch size of the test
            loader.
    """
    metrics_path = benchmark_folder / "test_metrics.csv"
    print_color(color="blue", output=f"Served Formats Test Metrics : {format_path_link(path=metrics_path)}",
                add_separators=True)
    for model in models:
        evaluation_folder = benchmark_folder / "test_metric" / model["name"]
        measured_rows = read_measured_rows(metrics_path=metrics_path)
        served_model_name = read_served_model_name(experiment_folder=model["experiment_folder"])
        planned_rows = [EAGER_ROW_NAME] + [
            get_row_name(precision_setting=precision_setting, export_format=export_format)
            for precision_setting in precision_settings
            for export_format in get_repository_formats(precision_setting=precision_setting)]
        if all((model["name"], row_name) in measured_rows for row_name in planned_rows):
            continue
        trainer = build_evaluation_trainer(experiment_folder=model["experiment_folder"],
                                           test_data_folder=model["test_data_folder"],
                                           evaluation_folder=evaluation_folder,
                                           standalone_evaluation_folder=CHECKPOINT_EVALUATION_FOLDER_NAME,
                                           batch_size=maximum_request_size)

        # Reference: the eager inference model of the checkpoint (its trained weights, as exported) in the process
        inference_model, export_specification = load_local_inference_model(
            experiment_folder=model["experiment_folder"], iteration=model["iteration"])
        output_names = [output["name"] for output in export_specification["outputs"]]
        if (model["name"], EAGER_ROW_NAME) not in measured_rows:
            evaluate_row(trainer=trainer, model=model, row_name=EAGER_ROW_NAME,
                         run_model=functools.partial(run_local_inference, inference_model=inference_model,
                                                     decoding_settings=None),
                         output_names=output_names, maximum_request_size=maximum_request_size,
                         evaluation_folder=evaluation_folder, metrics_path=metrics_path)
        del inference_model
        torch.cuda.empty_cache()

        # Served rows, one server per precision setting, its models loaded one at a time
        for precision_setting in precision_settings:
            setting_rows = {export_format: get_row_name(precision_setting=precision_setting,
                                                        export_format=export_format)
                            for export_format in get_repository_formats(precision_setting=precision_setting)}
            if all((model["name"], row_name) in measured_rows for row_name in setting_rows.values()):
                continue
            start_triton_server(server=BENCHMARK_SERVER,
                                repository=benchmark_folder / "model_repositories" / precision_setting,
                                engine_cache_folder=benchmark_folder / "tensorrt_engine_cache" / precision_setting,
                                environment={})
            # The server log is kept and the container removed whatever ends the evaluations
            try:
                client = triton_client.InferenceServerClient(url=BENCHMARK_SERVER["grpc_address"])
                for export_format, row_name in setting_rows.items():
                    if (model["name"], row_name) in measured_rows:
                        continue
                    model_name = f"{served_model_name}_{export_format}"
                    load_served_model(server=BENCHMARK_SERVER, model_name=model_name, configuration=None)
                    # A model exported before an input of its task (the image windows of the detectors) is refused
                    check_served_input_names(client=client, model_name=model_name,
                                             input_names=TASK_INPUT_NAMES[trainer.TASK_NAME])
                    # Served names of the outputs ('<name>__<index>' for TorchScript), in the order of output_names
                    served_output_names = get_triton_output_names(client=client, model_name=model_name,
                                                                  output_names=output_names)
                    evaluate_row(trainer=trainer, model=model, row_name=row_name,
                                 run_model=functools.partial(run_triton_inference, client=client,
                                                             model_name=model_name, output_names=served_output_names,
                                                             shared_memory_handle=None, region_name=None),
                                 output_names=output_names, maximum_request_size=maximum_request_size,
                                 evaluation_folder=evaluation_folder, metrics_path=metrics_path)
                    unload_served_model(server=BENCHMARK_SERVER, model_name=model_name)
            finally:
                log_name = f"test_metric_{model['name']}_{precision_setting}_{time.strftime('%Y%m%d_%H%M%S')}.log"
                stop_triton_server(server=BENCHMARK_SERVER, log_path=benchmark_folder / "logs" / log_name)
        del trainer
        torch.cuda.empty_cache()


def parse_evaluation_arguments(arguments: List[str]) -> argparse.Namespace:
    """
    Reads the folders of the machine from the command line of the script, the same as those of
    benchmark_serving_formats.py, whose constants give the models (BENCHMARKED_MODELS), the precision settings and the
    largest request of the evaluation, so that it scores the model repositories the speed sweep wrote. Launched without
    any argument, the script prints the help of every argument and stops. A '~' left by the shell at the start of a
    path ('--benchmark_folder=~/Data/...') is expanded to the home folder.

    Args:
        arguments (List[str]): The command line arguments, without the script name.

    Returns:
        argparse.Namespace: The benchmark_folder, runs_folder and data_folder.

    Raises:
        SystemExit: After printing the help when no argument is given, or the usage and the error when a folder is
            missing.
    """
    # The description and the example keep their line breaks, argparse wrapping the help of every argument
    parser = argparse.ArgumentParser(
        prog="uv run python -m scripts.deployment.evaluate_served_formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Scores the test split of every run of the serving benchmark through the eager model, then\n"
                    "through every served format, one line per scalar in <benchmark_folder>/test_metrics.csv.",
        epilog="Example, after benchmark_serving_formats.py, with the same folders:\n"
               "  uv run python -m scripts.deployment.evaluate_served_formats \\\n"
               "      --benchmark_folder ~/Data/serving_benchmark --runs_folder ~/Data/tutorial_runs \\\n"
               "      --data_folder ~/Data")
    # The folders of this machine, those of the speed sweep
    add_benchmark_folder_arguments(parser=parser)

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
    # Folders of this machine read from the command line, the help printed when none is given
    evaluation_arguments = parse_evaluation_arguments(arguments=sys.argv[1:])

    # The models, precision settings and largest request of the speed sweep, its models placed in the folders of this
    # machine
    run_served_evaluations(models=build_benchmarked_models(runs_folder=evaluation_arguments.runs_folder,
                                                           data_folder=evaluation_arguments.data_folder),
                           benchmark_folder=evaluation_arguments.benchmark_folder,
                           precision_settings=list(PRECISION_SETTINGS), maximum_request_size=MAXIMUM_BATCH_SIZE)
