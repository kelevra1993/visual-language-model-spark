"""
Checks of the TensorRT engine that Triton builds from an exported 'model.onnx' (deployment/export_model.py with
tensorrt_precision 'FP32' or 'FP16'), run in the staging folder of the export before its models are installed, so that
an engine giving other outputs than the eager model blocks the installation like a failed format check. For FP16, a CPU
scan runs the ONNX graph with ONNX Runtime on the check batches with every intermediate node output exposed and
reports the finite values above 65,504, the largest FP16 value, which the engine would turn into infinities (the NMS
of the detectors overflowed this way before 2026-11-17, losing 7.88 points of mAP). Then, unless skipped, the staged
model is served with its TensorRT config.pbtxt by a temporary Triton container (deployment/triton_server.py, as the
test split parity of deployment/export_parity.py serves it), the check batches are sent and the outputs compared with
the eager model by the comparison of the export specification of the task.
"""

import operator
import tempfile
import time
import urllib.error
import numpy as np
import onnx
import onnxruntime
import tritonclient.grpc as triton_client

from pathlib import Path
from tritonclient.utils import InferenceServerException
from typing import Any, Dict, List

from vision_training.deployment.deployment_utilities import get_triton_output_names, run_triton_inference
from vision_training.deployment.triton_server import (TritonServerError, load_served_model,
                                                      start_temporary_triton_server, stop_triton_server)

# Largest finite FP16 value: a larger value computed by an FP16 TensorRT engine becomes an infinity
HALF_PRECISION_LARGEST_VALUE = 65504.0

# ONNX element types of the node outputs the overflow scan reads, the ones an FP16 engine computes in FP16
FLOATING_ELEMENT_TYPES = [onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE, onnx.TensorProto.FLOAT16]

# Nodes named in the console and the error of a failed overflow scan, the largest magnitudes first
OVERFLOWING_NODES_SHOWN = 10

# Container of the engine check, followed by the process identifier, so that it never collides with the server of
# deployment/triton/docker-compose.yml, a benchmark or the test split parity, the Triton log lines quoted in the
# mismatch of an engine Triton fails to build or run, and the words marking them (a crash of the backend is logged as
# 'Signal (11) received', without the word error)
TENSORRT_CHECK_CONTAINER_PREFIX = "export-tensorrt-check-triton"
SERVER_LOG_LINES_SHOWN = 10
SERVER_LOG_FAILURE_WORDS = ["error", "signal"]


def list_floating_node_outputs(model_proto: onnx.ModelProto) -> List[Dict[str, str]]:
    """
    Lists the floating point outputs of every node of the main graph of an exported model, with the node producing
    them, the values the overflow scan exposes as graph outputs. The element types come from the ONNX shape inference,
    so that the integer and boolean values (indices, masks), which an FP16 engine keeps in their own type, are left
    out.

    Args:
        model_proto (onnx.ModelProto): The exported model, its weights loaded or not.

    Returns:
        List[Dict[str, str]]: The 'output' name, 'node' name and 'op_type' of every floating point node output, in
            the order of the nodes.

    Raises:
        ValueError: If the shape inference gives no element type for a node output, whose values could otherwise be
            left out of the scan unnoticed.
    """
    # Element type of every value of the graph, read from the shape inference of the graph without its weights
    inferred_model = onnx.shape_inference.infer_shapes(model=model_proto)
    element_types = {value.name: value.type.tensor_type.elem_type
                     for value in [*inferred_model.graph.value_info, *inferred_model.graph.output]}

    # Floating point outputs of every node, an output without a type stopping the scan instead of being skipped
    node_outputs = []
    for node in model_proto.graph.node:
        for output_name in node.output:
            if not output_name:
                continue
            if element_types.get(output_name, onnx.TensorProto.UNDEFINED) == onnx.TensorProto.UNDEFINED:
                raise ValueError(f"The ONNX shape inference gives no element type for the output {output_name} of the "
                                 f"{node.op_type} node {node.name}: the FP16 overflow scan cannot tell whether it is "
                                 f"a floating point value.")
            if element_types[output_name] in FLOATING_ELEMENT_TYPES:
                node_outputs.append({"output": output_name, "node": node.name, "op_type": node.op_type})
    return node_outputs


def find_half_precision_overflows(model_path: Path, check_inputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Finds the values of an exported ONNX model that an FP16 TensorRT engine cannot hold, before the engine is built: a
    copy of the graph exposing every floating point node output as a graph output is run by ONNX Runtime on the CPU
    on the check batches, one sample at a time so that the exposed values of a single sample are held in memory at
    once (about the memory of the ONNX Runtime check of the whole batch), and the largest finite magnitude of every
    output is compared with HALF_PRECISION_LARGEST_VALUE. The infinities and NaNs of the graph are left out: they are
    not created by the reduced precision.

    Args:
        model_path (Path): The exported 'model.onnx', its weights in 'model.onnx.data' next to it.
        check_inputs (Dict[str, np.ndarray]): The float32 check batch of every input, keyed by input name, every
            input holding the same number of samples.

    Returns:
        Dict[str, Any]: The number of 'scanned_outputs' and the 'overflows': the 'node', 'op_type', 'output' and
            'largest_magnitude' of every node output with a finite value above HALF_PRECISION_LARGEST_VALUE, the
            largest magnitudes first, empty when every value fits.
    """
    # Copy of the graph exposing every floating point node output, written with its weights into a temporary folder
    # so that the exported version folder, whose files name the TensorRT engines, stays untouched
    model_proto = onnx.load(f=str(model_path))
    node_outputs = list_floating_node_outputs(model_proto=model_proto)
    exposed_names = {output.name for output in model_proto.graph.output}
    model_proto.graph.output.extend([onnx.ValueInfoProto(name=node_output["output"]) for node_output in node_outputs
                                     if node_output["output"] not in exposed_names])
    largest_magnitudes = {node_output["output"]: 0.0 for node_output in node_outputs}
    with tempfile.TemporaryDirectory(prefix="half_precision_scan_") as scan_folder:
        scan_model_path = Path(scan_folder) / "model.onnx"
        onnx.save_model(proto=model_proto, f=str(scan_model_path), save_as_external_data=True,
                        all_tensors_to_one_file=True, location="model.onnx.data")
        del model_proto
        session = onnxruntime.InferenceSession(str(scan_model_path), providers=["CPUExecutionProvider"])
        output_names = [session_output.name for session_output in session.get_outputs()]

        # Every sample of the check batches alone, the largest finite magnitude of every output kept across samples
        number_samples = next(iter(check_inputs.values())).shape[0]
        for sample_index in range(number_samples):
            sample_inputs = {input_name: input_batch[sample_index:sample_index + 1]
                             for input_name, input_batch in check_inputs.items()}
            for output_name, output_value in zip(output_names, session.run(None, sample_inputs)):
                if output_name not in largest_magnitudes:
                    continue
                finite_magnitudes = np.abs(output_value[np.isfinite(output_value)])
                if finite_magnitudes.size:
                    largest_magnitudes[output_name] = max(largest_magnitudes[output_name],
                                                          float(finite_magnitudes.max()))
        del session

    # Node outputs above the largest FP16 value, the largest first
    overflows = [{**node_output, "largest_magnitude": largest_magnitudes[node_output["output"]]}
                 for node_output in node_outputs
                 if largest_magnitudes[node_output["output"]] > HALF_PRECISION_LARGEST_VALUE]
    return {"scanned_outputs": len(node_outputs),
            "overflows": sorted(overflows, key=operator.itemgetter("largest_magnitude"), reverse=True)}


def describe_half_precision_overflows(overflows: List[Dict[str, Any]]) -> List[str]:
    """
    Writes the mismatches of a failed FP16 overflow scan, printed in red by the export and raised with its error: one
    line per overflowing node (up to OVERFLOWING_NODES_SHOWN, the largest magnitudes first) naming its op type and its
    largest magnitude, then the settings exporting the model without the FP16 engine.

    Args:
        overflows (List[Dict[str, Any]]): The 'overflows' of find_half_precision_overflows, not empty.

    Returns:
        List[str]: The lines of the mismatch.
    """
    lines = [f"TensorRT FP16 overflow: {len(overflows)} node outputs of model.onnx exceed "
             f"{HALF_PRECISION_LARGEST_VALUE:,.0f}, the largest FP16 value, on the check batches"]
    lines += [f"  {overflow['node']} ({overflow['op_type']}): largest magnitude {overflow['largest_magnitude']:.6g}"
              for overflow in overflows[:OVERFLOWING_NODES_SHOWN]]
    if len(overflows) > OVERFLOWING_NODES_SHOWN:
        lines.append(f"  and {len(overflows) - OVERFLOWING_NODES_SHOWN} more")
    lines.append("  Export with --tensorrt_precision FP32, or None for ONNX Runtime CUDA without TensorRT")
    return lines


def compare_engine_outputs(reference_outputs: List[np.ndarray], engine_outputs: List[np.ndarray],
                           output_names: List[str], specification: Dict[str, Any],
                           tolerances: Dict[str, float]) -> List[str]:
    """
    Compares the outputs of the TensorRT engine served by Triton with the eager reference on the check batches: any
    infinity or NaN of a floating point output fails, whatever the tolerances (an FP16 overflow gives infinities, which
    a comparison of differences may not see), then the comparison of the export specification of the task runs with
    the given tolerances.

    Args:
        reference_outputs (List[np.ndarray]): The eager outputs, the FP32 model for an FP32 engine, its FP16 copy for an
            FP16 engine.
        engine_outputs (List[np.ndarray]): The outputs of the served engine, in the same order.
        output_names (List[str]): The output names of the export specification, in order.
        specification (Dict[str, Any]): The export specification of the task, giving its 'compare_outputs'.
        tolerances (Dict[str, float]): The tolerances of the comparison.

    Returns:
        List[str]: A description of every mismatch, empty when the engine agrees with the reference.
    """
    # Infinities and NaNs of the engine always fail, named with their output
    mismatches = [f"{output_name}: {int((~np.isfinite(engine_output)).sum())} non-finite values"
                  for output_name, engine_output in zip(output_names, engine_outputs)
                  if np.issubdtype(engine_output.dtype, np.floating) and not np.isfinite(engine_output).all()]
    if mismatches:
        return mismatches
    return specification["compare_outputs"](reference=reference_outputs, candidate=engine_outputs, **tolerances)


def read_server_log_errors(log_path: Path) -> List[str]:
    """
    Gives the last lines of a saved Triton log reporting an error or a signal, quoted in the mismatch of an engine
    Triton fails to build or run (a node without a TensorRT implementation, a crash of the backend), since the staging
    folder holding the log is removed with the export.

    Args:
        log_path (Path): The log saved by stop_triton_server.

    Returns:
        List[str]: Up to SERVER_LOG_LINES_SHOWN lines containing one of SERVER_LOG_FAILURE_WORDS, the last ones of
            the log.
    """
    error_lines = [line.strip() for line in log_path.read_text(errors="replace").splitlines()
                   if any(word in line.lower() for word in SERVER_LOG_FAILURE_WORDS)]
    return error_lines[-SERVER_LOG_LINES_SHOWN:]


def check_tensorrt_engine(staged_repository: Path, engine_cache_folder: Path, log_path: Path, model_name: str,
                          check_inputs: Dict[str, np.ndarray], reference_outputs: List[np.ndarray],
                          specification: Dict[str, Any], tolerances: Dict[str, float]) -> Dict[str, Any]:
    """
    Serves the staged ONNX model of an export with its TensorRT config.pbtxt by a temporary Triton container on free
    ports, sends the check batches in one request, which builds the engine (ONNX Runtime builds it at the first request
    of a shape), and compares the outputs with the eager reference (compare_engine_outputs). A model Triton fails to
    load, build or run is a mismatch quoting the error and the error lines of the server log. The container is removed
    whatever happens.

    Args:
        staged_repository (Path): The model repository of the staged export, served read only.
        engine_cache_folder (Path): The folder receiving the TensorRT engines, shared with the test split parity.
        log_path (Path): The file receiving the server log.
        model_name (str): The Triton model name of the ONNX export.
        check_inputs (Dict[str, np.ndarray]): The float32 check batch of every input, keyed by input name.
        reference_outputs (List[np.ndarray]): The eager outputs on the check batches.
        specification (Dict[str, Any]): The export specification of the task.
        tolerances (Dict[str, float]): The tolerances of the comparison.

    Returns:
        Dict[str, Any]: The 'mismatches' (empty when the engine agrees), the 'server_start_seconds' and the
            'engine_check_seconds' (model load, engine build, request and comparison).
    """
    output_names = [output["name"] for output in specification["outputs"]]
    start_time = time.time()
    server = start_temporary_triton_server(container_name_prefix=TENSORRT_CHECK_CONTAINER_PREFIX,
                                           repository=staged_repository, engine_cache_folder=engine_cache_folder)
    server_start_seconds = time.time() - start_time

    # The request builds the engine and runs it, a failure of Triton being recorded with its first line
    check_start_time = time.time()
    server_error = None
    try:
        try:
            load_served_model(server=server, model_name=model_name, configuration=None)
            client = triton_client.InferenceServerClient(url=server["grpc_address"])
            engine_outputs = run_triton_inference(
                inputs=check_inputs, client=client, model_name=model_name,
                output_names=get_triton_output_names(client=client, model_name=model_name, output_names=output_names),
                shared_memory_handle=None, region_name=None)
            mismatches = compare_engine_outputs(reference_outputs=reference_outputs, engine_outputs=engine_outputs,
                                                output_names=output_names, specification=specification,
                                                tolerances=tolerances)
        except (TritonServerError, InferenceServerException, urllib.error.URLError) as error:
            server_error = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
    finally:
        stop_triton_server(server=server, log_path=log_path)
    engine_check_seconds = time.time() - check_start_time

    # An engine Triton could not build or run, with the errors of its log, the staging folder holding it being removed
    if server_error is not None:
        mismatches = [f"Triton failed to build or run the engine: {server_error}"]
        mismatches += [f"  log: {line}" for line in read_server_log_errors(log_path=log_path)]
    return {"mismatches": mismatches, "server_start_seconds": server_start_seconds,
            "engine_check_seconds": engine_check_seconds}
