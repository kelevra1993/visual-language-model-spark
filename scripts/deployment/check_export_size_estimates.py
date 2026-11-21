"""
When run, checks the export sizes printed at the launch of every training (the Export Sizes panel of
vision_training/deployment/export_size_estimates.py) against the files of real exports of trained runs, and gives the
constants of its formulas fitted on them, so that the panel can be calibrated again after a change of the exports or of
the libraries. For every run it loads the last checkpoint, computes the estimates with the function of the panel,
writes the PyTorch weights in FP32 and FP16 with torch.save and the ONNX files with export_onnx, and reads the size of
the real checkpoint of the run. With --build-tensorrt-engines it also serves the ONNX files in the Triton container of
the serving benchmark (scripts/deployment/benchmark_serving_formats.py), once in FP32 and once in FP16, sends one
request at batch 1 and one at the largest batch so that ONNX Runtime builds the TensorRT engines, and reads their size
in the engine cache: about a minute of GPU per run and precision, to run when the GPU is free. Every output stays
outside the repository, under --output-folder: one folder of exported files per run, the Triton logs and
'export_size_check.csv', one line per run.

Run from the repository root:
    uv run python -m scripts.deployment.check_export_size_estimates
    uv run python -m scripts.deployment.check_export_size_estimates --build-tensorrt-engines
"""

import argparse
import csv
import shutil
import statistics
import numpy as np
import torch

from pathlib import Path
from torch import nn
from typing import Any, Dict, List, Optional

from scripts.deployment.benchmark_serving_formats import (load_served_model, start_triton_server, stop_triton_server,
                                                          time_first_requests, unload_served_model)
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.deployment.export_formats import export_onnx, hide_known_export_warnings
from vision_training.deployment.export_size_estimates import (EXPORT_MAXIMUM_BATCH_SIZE, ONNX_GRAPH_BYTES_PER_OPERATOR,
                                                              STATE_DICTIONARY_BYTES_PER_TENSOR,
                                                              TENSORRT_BYTES_PER_OTHER_OPERATOR,
                                                              TENSORRT_BYTES_PER_WEIGHTED_OPERATOR,
                                                              estimate_export_sizes)
from vision_training.deployment.export_specifications import (TASK_MODEL_NAMES, build_inference_model,
                                                              create_random_inputs, get_export_specification)
from vision_training.deployment.triton_configuration import write_triton_configuration
from vision_training.models.model_loading import load_trained_model
from vision_training.trainers.checkpointing import get_checkpoint_path
from vision_training.utilities.os_utilities import format_path_link, print_color

# Trained runs of every task the check exports, their last checkpoint being read: the tutorial runs, from their copies
# outside the repository, with the reference detector of the detection benchmark (run_9)
TUTORIAL_RUNS_FOLDER = Path.home() / "Data" / "tutorial_runs"
DEFAULT_EXPERIMENT_FOLDERS = [
    TUTORIAL_RUNS_FOLDER / "readme_screenshots" / "CLS_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HFLAT.128-HLN-CWB_MC9",
    TUTORIAL_RUNS_FOLDER / "hyperparameter_experiments" / "object_detection" / "run_9" /
    "DET_IN-KAR-1024_SE-RES-BN-B1x16.2x16.2x16.3x32.3x32.5x32.5x32_L4567-NFPN32BN_A6.16-768.d8ffba_RBN-R2x32_DSE-DRES-DBN-D7.2x32.64.32_MC35",
    TUTORIAL_RUNS_FOLDER / "readme_screenshots" / "INS_IN-KAR-512_SE-RES-BN-B1x16.2x16.2x16.3x32.3x32.5x32.5x32_L4567-NFPN32BN_A6+6+6+9.16-384.986117_RBN-R2x32-RPL_DSE-DRES-DBN-D7.2x32.64.32_MRES-MBN-MUTC-M14.4x64-MR28_MC35",
    TUTORIAL_RUNS_FOLDER / "readme_screenshots" / "SEG_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_UTC-DBN-D2x64.2x64.2x32.2x32.1x16_MC9",
    TUTORIAL_RUNS_FOLDER / "keypoint_focal_synthetic" / "experiments" / "KPT_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x64_HRSE-HRRES-HRBN-HR16.1.2.2x2_K243-KH256-HH_MC33",
    TUTORIAL_RUNS_FOLDER / "tutorials" / "SSL_IN-KAR-256_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_PAVG.512.128-PBN_SIMCLR-T0.2",
    TUTORIAL_RUNS_FOLDER / "readme_screenshots" / "M2O_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128-BI_HAVG.64_MC4",
    TUTORIAL_RUNS_FOLDER / "tutorials" / "M2M_IN-KAR-256_SE-RES-BN-B1x16.2x16.2x16.4x32.5x32.5x32-PT_T50_LSTM2x96-BI_HFLAT.96.96-HLN-CWB_MC71",
    TUTORIAL_RUNS_FOLDER / "readme_screenshots" / "SEL_IN-KAR-256_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128-BI_HAVG.64_MC4_FPC1-SLW1",
    TUTORIAL_RUNS_FOLDER / "readme_screenshots" / "CMP_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_MEAN_FDIF_HAVG.64_MC2"]

# Optimizer of every 'Optimization.optimizer' of the configurations, as BaseTrainer.create_optimizer builds it, of
# which the checkpoint estimate only reads the class and the momentum
OPTIMIZER_CLASSES = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}

# Columns of export_size_check.csv: the run, the time of the estimates, the forward pass of the probe, then the
# estimated and real size of every format, in bytes (empty when not measured)
RESULT_COLUMNS = ["run", "task", "device", "estimate_seconds", "total_parameters", "exported_parameters",
                  "state_dictionary_tensors", "state_dictionary_key_characters", "constant_bytes",
                  "batch_normalization_bytes", "weighted_operators", "other_operators",
                  "pytorch_fp32_estimated", "pytorch_fp32_real", "pytorch_fp16_estimated", "pytorch_fp16_real",
                  "torchscript_measured", "onnx_estimated", "onnx_graph_real", "onnx_data_real", "onnx_real",
                  "checkpoint_estimated", "checkpoint_real", "tensorrt_fp32_lowest", "tensorrt_fp32_highest",
                  "tensorrt_fp32_real", "tensorrt_fp16_lowest", "tensorrt_fp16_highest", "tensorrt_fp16_real"]

# Rows of the printed report: the label, the estimated and real columns (a range for TensorRT)
REPORT_FORMATS = [("PyTorch FP32 Weights", ["pytorch_fp32_estimated"], "pytorch_fp32_real"),
                  ("PyTorch FP16 Weights", ["pytorch_fp16_estimated"], "pytorch_fp16_real"),
                  ("ONNX", ["onnx_estimated"], "onnx_real"),
                  ("TensorRT FP32 Engine", ["tensorrt_fp32_lowest", "tensorrt_fp32_highest"], "tensorrt_fp32_real"),
                  ("TensorRT FP16 Engine", ["tensorrt_fp16_lowest", "tensorrt_fp16_highest"], "tensorrt_fp16_real"),
                  ("Training Checkpoint", ["checkpoint_estimated"], "checkpoint_real")]


def build_run_optimizer(model: nn.Module, optimizer_name: str) -> torch.optim.Optimizer:
    """
    Builds the optimizer of a run over its parameters, of the class the trainer builds for its configuration, the
    checkpoint estimate reading the class and momentum of the optimizer and the trained parameters only.

    Args:
        model (nn.Module): The model of the run.
        optimizer_name (str): 'Optimization.optimizer' of the run, 'adam', 'adamw' or 'sgd'.

    Returns:
        torch.optim.Optimizer: The optimizer.
    """
    if optimizer_name == "sgd":
        return torch.optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    return OPTIMIZER_CLASSES[optimizer_name](params=model.parameters(), lr=1e-3)


def measure_saved_state_dictionaries(state_dictionary: Dict[str, torch.Tensor], output_folder: Path) -> Dict[str, int]:
    """
    Writes a state dictionary with torch.save in FP32 and with its floating tensors in FP16, and gives the size of
    both files, the real sizes of the PyTorch weights of the panel.

    Args:
        state_dictionary (Dict[str, torch.Tensor]): The state dictionary of the inference model.
        output_folder (Path): The folder of the run, receiving 'weights_fp32.pt' and 'weights_fp16.pt'.

    Returns:
        Dict[str, int]: The bytes of 'pytorch_fp32_real' and 'pytorch_fp16_real'.
    """
    half_state_dictionary = {key: tensor.half() if tensor.is_floating_point() else tensor
                             for key, tensor in state_dictionary.items()}
    torch.save(obj=state_dictionary, f=output_folder / "weights_fp32.pt")
    torch.save(obj=half_state_dictionary, f=output_folder / "weights_fp16.pt")
    return {"pytorch_fp32_real": (output_folder / "weights_fp32.pt").stat().st_size,
            "pytorch_fp16_real": (output_folder / "weights_fp16.pt").stat().st_size}


def check_run(experiment_folder: Path, device: torch.device, output_folder: Path) -> Dict[str, Any]:
    """
    Computes the estimates of the Export Sizes panel for the last checkpoint of a run, then measures the real files:
    the PyTorch weights, the ONNX files exported as deployment/export_model.py does (kept in the folder of the run for
    the TensorRT engines) and the checkpoint of the run.

    Args:
        experiment_folder (Path): The run folder, holding 'Weights' and 'training_configuration.yaml'.
        device (torch.device): The device of the model during the estimates (the time of the panel at launch is that
            of the GPU), the exports running on the CPU.
        output_folder (Path): The folder of the files of the run.

    Returns:
        Dict[str, Any]: The line of the run in export_size_check.csv, without its TensorRT engines.
    """
    configuration_path = experiment_folder / "training_configuration.yaml"
    experiment_configuration, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    checkpoint_path = get_checkpoint_path(experiment_folder=experiment_folder, iteration=None)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Estimates of the panel, on the model in training mode as at launch
    model = load_trained_model(configuration_path=configuration_path, checkpoint_path=checkpoint_path, device=device,
                               use_averaged_weights=False).train()
    optimization_configuration = model_configuration["Optimization"]
    export_sizes = estimate_export_sizes(
        task=experiment_configuration["task"], model=model,
        optimizer=build_run_optimizer(model=model, optimizer_name=optimization_configuration["optimizer"]),
        keeps_moving_average=optimization_configuration["exponential_moving_average"]["activated"])
    if export_sizes["errors"]:
        raise RuntimeError(f"The estimates of {experiment_folder.name} failed:\n" + "\n".join(export_sizes["errors"]))
    estimated = {row["label"]: row for row in export_sizes["rows"]}
    probe = export_sizes["probe"]

    # Real files, exported from the model loaded again on the CPU (a model built on the GPU keeps tensor attributes
    # there), in evaluation mode
    model = load_trained_model(configuration_path=configuration_path, checkpoint_path=checkpoint_path,
                               device=torch.device("cpu"), use_averaged_weights=False)
    inference_model = build_inference_model(task=experiment_configuration["task"], model=model)
    state_dictionary = inference_model.state_dict()
    result = {"run": experiment_folder.name, "task": experiment_configuration["task"], "device": str(device),
              "estimate_seconds": round(export_sizes["seconds"], 3),
              "total_parameters": export_sizes["total_parameters"],
              "exported_parameters": export_sizes["exported_parameters"],
              "state_dictionary_tensors": len(state_dictionary),
              "state_dictionary_key_characters": sum(len(key) for key in state_dictionary),
              "constant_bytes": probe["constant_bytes"], "batch_normalization_bytes": probe["batch_normalization_bytes"],
              "weighted_operators": probe["weighted_operators"], "other_operators": probe["other_operators"],
              "pytorch_fp32_estimated": estimated["PyTorch FP32 Weights"]["lowest_bytes"],
              "pytorch_fp16_estimated": estimated["PyTorch FP16 Weights"]["lowest_bytes"],
              "torchscript_measured": estimated["TorchScript"]["lowest_bytes"],
              "onnx_estimated": estimated["ONNX"]["lowest_bytes"],
              "checkpoint_estimated": estimated["Training Checkpoint"]["lowest_bytes"],
              "checkpoint_real": checkpoint_path.stat().st_size,
              "tensorrt_fp32_lowest": estimated["TensorRT FP32 Engine"]["lowest_bytes"],
              "tensorrt_fp32_highest": estimated["TensorRT FP32 Engine"]["highest_bytes"],
              "tensorrt_fp16_lowest": estimated["TensorRT FP16 Engine"]["lowest_bytes"],
              "tensorrt_fp16_highest": estimated["TensorRT FP16 Engine"]["highest_bytes"],
              "tensorrt_fp32_real": None, "tensorrt_fp16_real": None}
    result.update(measure_saved_state_dictionaries(state_dictionary=state_dictionary, output_folder=output_folder))

    # ONNX files of the export, traced on the example batch of the export
    specification = get_export_specification(task=experiment_configuration["task"], inference_model=inference_model)
    example_inputs = create_random_inputs(specification=specification, shape_name="example_shape",
                                          maximum_batch_size=EXPORT_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                          generator=torch.Generator().manual_seed(0))
    onnx_path = output_folder / "onnx" / "model.onnx"
    export_onnx(inference_model=inference_model, example_inputs=example_inputs, output_path=onnx_path,
                specification=specification, maximum_batch_size=EXPORT_MAXIMUM_BATCH_SIZE)
    result.update({"onnx_graph_real": onnx_path.stat().st_size,
                   "onnx_data_real": Path(f"{onnx_path}.data").stat().st_size})
    result["onnx_real"] = result["onnx_graph_real"] + result["onnx_data_real"]
    return result


def build_tensorrt_repository(results: List[Dict[str, Any]], experiment_folders: List[Path], output_folder: Path,
                              tensorrt_precision: str) -> Path:
    """
    Builds the Triton model repository of one TensorRT precision from the ONNX files of check_run: every run as its
    task model '<model name>_onnx', version 1, with the config.pbtxt the export writes for that precision.

    Args:
        results (List[Dict[str, Any]]): The lines of the runs, from check_run.
        experiment_folders (List[Path]): The run folders, in the order of the results.
        output_folder (Path): The folder of every output of the check.
        tensorrt_precision (str): 'FP32' or 'FP16'.

    Returns:
        Path: The repository.
    """
    repository = output_folder / f"model_repository_{tensorrt_precision.lower()}"
    shutil.rmtree(path=repository, ignore_errors=True)
    for result, experiment_folder in zip(results, experiment_folders):
        model_name = f"{TASK_MODEL_NAMES[result['task']]}_onnx"
        model_directory = repository / model_name
        (model_directory / "1").mkdir(parents=True)
        for file_name in ["model.onnx", "model.onnx.data"]:
            shutil.copyfile(src=output_folder / result["run"] / "onnx" / file_name,
                            dst=model_directory / "1" / file_name)
        specification = read_export_specification(experiment_folder=experiment_folder)
        write_triton_configuration(model_directory=model_directory, model_name=model_name, model_version=1,
                                   export_format="onnx", specification=specification,
                                   maximum_batch_size=EXPORT_MAXIMUM_BATCH_SIZE, tensorrt_precision=tensorrt_precision)
    return repository


def read_export_specification(experiment_folder: Path) -> Dict[str, Any]:
    """
    Gives the export specification of a run, from its inference model built on the CPU without trained weights (the
    specification depends on the configuration only).

    Args:
        experiment_folder (Path): The run folder.

    Returns:
        Dict[str, Any]: The export specification of the task of the run.
    """
    configuration_path = experiment_folder / "training_configuration.yaml"
    experiment_configuration, _ = load_experiment_configuration(configuration_path=configuration_path)
    model = load_trained_model(configuration_path=configuration_path,
                               checkpoint_path=get_checkpoint_path(experiment_folder=experiment_folder, iteration=None),
                               device=torch.device("cpu"), use_averaged_weights=False)
    inference_model = build_inference_model(task=experiment_configuration["task"], model=model)
    return get_export_specification(task=experiment_configuration["task"], inference_model=inference_model)


def sum_engine_bytes(engine_cache_folder: Path, engine_cache_prefix: str) -> Optional[int]:
    """
    Sums the TensorRT engines ONNX Runtime wrote into the cache for one model, one file per part of the graph handed
    to TensorRT, named after the cache prefix of its config.pbtxt.

    Args:
        engine_cache_folder (Path): The engine cache folder mounted into the container.
        engine_cache_prefix (str): The 'trt_engine_cache_prefix' of the model.

    Returns:
        Optional[int]: The bytes of its engines, None when it has none.
    """
    engine_paths = sorted(engine_cache_folder.glob(f"{engine_cache_prefix}*.engine"))
    return sum(engine_path.stat().st_size for engine_path in engine_paths) if engine_paths else None


def read_engine_cache_prefix(model_directory: Path) -> str:
    """
    Reads the 'trt_engine_cache_prefix' written by write_triton_configuration into the config.pbtxt of a model.

    Args:
        model_directory (Path): The folder of the model in the repository.

    Returns:
        str: The prefix of its engine files.

    Raises:
        ValueError: If the configuration has no engine cache prefix.
    """
    for line in (model_directory / "config.pbtxt").read_text().splitlines():
        if '"trt_engine_cache_prefix"' in line:
            return line.split('value: "')[1].split('"')[0]
    raise ValueError(f"No trt_engine_cache_prefix in {model_directory / 'config.pbtxt'}.")


def measure_tensorrt_engines(results: List[Dict[str, Any]], experiment_folders: List[Path],
                             output_folder: Path) -> None:
    """
    Builds the TensorRT engines of every run in FP32 and FP16 through Triton and records their size in the results:
    for every precision, one server on the repository of build_tensorrt_repository with an empty engine cache, every
    model loaded alone and sent a random request at batch 1 then at the largest batch (ONNX Runtime building its
    engines at the first request and again for a larger batch), then unloaded.

    Args:
        results (List[Dict[str, Any]]): The lines of the runs, from check_run, completed in place.
        experiment_folders (List[Path]): The run folders, in the order of the results.
        output_folder (Path): The folder of every output of the check.
    """
    for tensorrt_precision in ["FP32", "FP16"]:
        repository = build_tensorrt_repository(results=results, experiment_folders=experiment_folders,
                                               output_folder=output_folder, tensorrt_precision=tensorrt_precision)
        engine_cache_folder = output_folder / f"tensorrt_engine_cache_{tensorrt_precision.lower()}"
        shutil.rmtree(path=engine_cache_folder, ignore_errors=True)
        start_triton_server(repository=repository, engine_cache_folder=engine_cache_folder, environment={})
        try:
            for result, experiment_folder in zip(results, experiment_folders):
                model_name = f"{TASK_MODEL_NAMES[result['task']]}_onnx"
                print_color(color="blue", output=f"TensorRT {tensorrt_precision} : {result['run']}", indent=1)
                # Random batch-1 request of the shapes of the traced example, the engine depending on shapes only
                specification = read_export_specification(experiment_folder=experiment_folder)
                request = {name: batch.numpy().astype(np.float32) for name, batch in create_random_inputs(
                    specification=specification, shape_name="example_shape", maximum_batch_size=1,
                    device=torch.device("cpu"), generator=torch.Generator().manual_seed(0)).items()}
                load_served_model(model_name=model_name, configuration=None)
                time_first_requests(model_name=model_name, request=request,
                                    batch_sizes=[1, EXPORT_MAXIMUM_BATCH_SIZE])
                unload_served_model(model_name=model_name)
                engine_bytes = sum_engine_bytes(engine_cache_folder=engine_cache_folder, engine_cache_prefix=(
                    read_engine_cache_prefix(model_directory=repository / model_name)))
                if engine_bytes is None:
                    raise RuntimeError(f"No engine of {model_name} in {engine_cache_folder}: "
                                       f"{sorted(path.name for path in engine_cache_folder.iterdir())}")
                result[f"tensorrt_{tensorrt_precision.lower()}_real"] = engine_bytes
        finally:
            stop_triton_server(log_path=output_folder / f"triton_{tensorrt_precision.lower()}.log")


def compute_relative_error(estimated_bytes: Optional[int], real_bytes: Optional[int]) -> Optional[float]:
    """
    Gives the relative error of an estimate against the real size.

    Args:
        estimated_bytes (Optional[int]): The estimate, None when missing.
        real_bytes (Optional[int]): The real size, None when not measured.

    Returns:
        Optional[float]: The error, e.g. 0.012 for 1.2 % above, None when either size is missing.
    """
    if estimated_bytes is None or real_bytes is None:
        return None
    return estimated_bytes / real_bytes - 1.0


def print_check_report(results: List[Dict[str, Any]]) -> None:
    """
    Prints, for every run, every format with its estimate (a range for TensorRT), its real size and the relative error
    of the estimate, a range being reported by how far the real size lies outside it (0 inside).

    Args:
        results (List[Dict[str, Any]]): The lines of the runs.
    """
    for result in results:
        print_color(color="blue", output=f"{result['run']} ({result['estimate_seconds']:.2f} seconds on "
                                         f"{result['device']}, TorchScript {result['torchscript_measured']:,} bytes)",
                    add_separators=True)
        for label, estimated_columns, real_column in REPORT_FORMATS:
            estimates = [result[column] for column in estimated_columns]
            real_bytes = result[real_column]
            if real_bytes is None:
                error_text = "not measured"
            elif len(estimates) == 2:
                outside_bytes = max(estimates[0] - real_bytes, real_bytes - estimates[1], 0)
                error_text = "inside" if outside_bytes == 0 else f"outside by {outside_bytes / real_bytes * 100:.1f} %"
            else:
                error_text = f"{compute_relative_error(estimated_bytes=estimates[0], real_bytes=real_bytes) * 100:+.2f} %"
            estimate_text = " to ".join(f"{estimate:,}" for estimate in estimates)
            real_text = f"{real_bytes:,}" if real_bytes is not None else "-"
            print(f"  {label:<22}{estimate_text:>27}  {real_text:>12}  {error_text}")


def fit_calibration(results: List[Dict[str, Any]]) -> None:
    """
    Prints the constants of export_size_estimates.py fitted on the runs next to their current values, and the largest
    errors of the current estimates, the margins of the panel: the bytes per tensor of a state dictionary, the bytes of
    the ONNX graph per operator, and, once engines were built, the TensorRT bytes per weighted and other operator by
    least squares on the overhead of every engine (its size less its weights in its precision).

    Args:
        results (List[Dict[str, Any]]): The lines of the runs.
    """
    print_color(color="blue", output="Calibration", add_separators=True)
    # Bytes per tensor of the state dictionaries, besides the values of the tensors and the characters of their keys:
    # the current constant corrected by the error of the estimate spread over the tensors
    per_tensor_bytes = [STATE_DICTIONARY_BYTES_PER_TENSOR + (result["pytorch_fp32_real"] -
                                                             result["pytorch_fp32_estimated"])
                        / result["state_dictionary_tensors"] for result in results]
    print(f"  STATE_DICTIONARY_BYTES_PER_TENSOR    current {STATE_DICTIONARY_BYTES_PER_TENSOR}, fitted "
          f"{statistics.median(per_tensor_bytes):.0f} (median of the runs)")

    # Bytes of the ONNX graph per operator of the forward pass
    graph_bytes_per_operator = [result["onnx_graph_real"] / (result["weighted_operators"] + result["other_operators"])
                                for result in results]
    print(f"  ONNX_GRAPH_BYTES_PER_OPERATOR        current {ONNX_GRAPH_BYTES_PER_OPERATOR}, fitted "
          f"{statistics.median(graph_bytes_per_operator):.0f} (median, from {min(graph_bytes_per_operator):.0f} to "
          f"{max(graph_bytes_per_operator):.0f})")

    # TensorRT overhead per weighted and other operator, by least squares over the engines built
    overhead_rows, overhead_bytes = [], []
    for result in results:
        folded_bytes = result["constant_bytes"] - result["batch_normalization_bytes"] * 3 // 4
        for precision_name, bytes_per_value in [("fp32", 4), ("fp16", 2)]:
            if result[f"tensorrt_{precision_name}_real"] is not None:
                overhead_rows.append([result["weighted_operators"], result["other_operators"]])
                overhead_bytes.append(result[f"tensorrt_{precision_name}_real"] - folded_bytes * bytes_per_value // 4)
    if len(overhead_rows) >= 2:
        fitted_constants = np.linalg.lstsq(np.array(overhead_rows, dtype=np.float64),
                                           np.array(overhead_bytes, dtype=np.float64), rcond=None)[0]
        predicted_overheads = np.array(overhead_rows, dtype=np.float64) @ fitted_constants
        overhead_ratios = np.array(overhead_bytes, dtype=np.float64) / predicted_overheads
        print(f"  TENSORRT_BYTES_PER_WEIGHTED_OPERATOR current {TENSORRT_BYTES_PER_WEIGHTED_OPERATOR}, fitted "
              f"{fitted_constants[0]:.0f}")
        print(f"  TENSORRT_BYTES_PER_OTHER_OPERATOR    current {TENSORRT_BYTES_PER_OTHER_OPERATOR}, fitted "
              f"{fitted_constants[1]:.0f}")
        print(f"  TENSORRT_OVERHEAD_RANGE              real overhead from {overhead_ratios.min():.2f} to "
              f"{overhead_ratios.max():.2f} times the fitted one ({len(overhead_rows)} engines)")

    # Largest error of every estimated format, the margins printed by the panel
    for label, estimated_columns, real_column in REPORT_FORMATS:
        if len(estimated_columns) == 1:
            errors = [compute_relative_error(estimated_bytes=result[estimated_columns[0]], real_bytes=result[real_column])
                      for result in results]
            errors = [error for error in errors if error is not None]
            if errors:
                print(f"  {label:<22} largest error {max(abs(error) for error in errors) * 100:.2f} % "
                      f"({len(errors)} runs)")


def write_results(results: List[Dict[str, Any]], results_path: Path) -> None:
    """
    Writes export_size_check.csv, one line per run with the columns of RESULT_COLUMNS.

    Args:
        results (List[Dict[str, Any]]): The lines of the runs.
        results_path (Path): The CSV file.
    """
    with results_path.open(mode="w", newline="") as results_file:
        writer = csv.DictWriter(f=results_file, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def main() -> None:
    """
    Checks the export size estimates of every run, writes the results and prints the report and the calibration.
    """
    parser = argparse.ArgumentParser(description="Checks the Export Sizes panel against real exports.")
    parser.add_argument("--experiment-folders", type=Path, nargs="+", default=DEFAULT_EXPERIMENT_FOLDERS,
                        help="Run folders to check (default: the tutorial runs of every task).")
    parser.add_argument("--output-folder", type=Path, default=Path.home() / "Data" / "export_size_check",
                        help="Folder of the exported files, the Triton logs and export_size_check.csv.")
    parser.add_argument("--device", default="cpu", help="Device of the model during the estimates, e.g. 'cuda'.")
    parser.add_argument("--build-tensorrt-engines", action="store_true",
                        help="Also build the TensorRT engines through Triton (about a minute of GPU per run and "
                             "precision).")
    arguments = parser.parse_args()

    hide_known_export_warnings()
    arguments.output_folder.mkdir(parents=True, exist_ok=True)
    results = []
    for experiment_folder in arguments.experiment_folders:
        print_color(color="blue", output=f"Checking {format_path_link(path=experiment_folder)}")
        results.append(check_run(experiment_folder=experiment_folder, device=torch.device(arguments.device),
                                 output_folder=arguments.output_folder / experiment_folder.name))
    if arguments.build_tensorrt_engines:
        measure_tensorrt_engines(results=results, experiment_folders=arguments.experiment_folders,
                                 output_folder=arguments.output_folder)
    write_results(results=results, results_path=arguments.output_folder / "export_size_check.csv")
    print_check_report(results=results)
    fit_calibration(results=results)


if __name__ == "__main__":
    main()
