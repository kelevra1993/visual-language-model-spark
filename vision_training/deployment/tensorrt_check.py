"""
Checks of the TensorRT engine that Triton builds from an exported 'model.onnx' (deployment/export_model.py with
tensorrt_precision 'FP32' or 'FP16'), run in the staging folder of the export before its models are installed, so that
an engine giving other outputs than the eager model blocks the installation like a failed format check. For FP16, a CPU
scan runs the ONNX graph with ONNX Runtime on the check batches with every intermediate node output exposed and
reports the finite values above 65,504, the largest FP16 value, which the engine would turn into infinities (the NMS
of the detectors overflowed this way before 2026-11-17, losing 7.88 points of mAP). Then, unless skipped, the staged
model is served with its TensorRT config.pbtxt by a temporary Triton container (deployment/triton_server.py, as the
test split parity of deployment/export_parity.py serves it), the check batches are sent and the outputs compared with
the eager model by the comparison of the export specification of the task. An engine missing those tolerances is
compared again with the FP32 eager model within a factor of the differences its reduced precision makes in the eager
model on the same batch: TF32 for an FP32 engine, which TensorRT runs with TF32 allowed, FP16 for an FP16 engine.
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
from typing import Any, Dict, List, Optional

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

# An engine missing its tolerances is compared again with the FP32 eager model within the roundings of its reduced
# precision, measured on the eager model on the same check batch: TF32 for an FP32 engine (which TensorRT runs with
# TF32 allowed, ONNX Runtime unable to switch it off, against the eager reference with TF32 off), FP16 for an FP16
# engine (the eager FP16 copy of convert_to_half_precision against the FP32 eager model, both FP16 roundings, the copy's
# and the engine's, adding up in a direct comparison of the two). It is accepted when its differences from the FP32
# eager model stay within REDUCED_PRECISION_DIFFERENCE_FACTOR times those of the eager model in the reduced precision,
# and its share of differing outputs (1 - minimum_equal_share) within REDUCED_PRECISION_DISAGREEMENT_FACTOR times that
# of the eager model, the tolerances of the specification staying the floor
REDUCED_PRECISION_DIFFERENCE_FACTOR = 4
REDUCED_PRECISION_DISAGREEMENT_FACTOR = 2

# Eager comparison measuring the roundings of every reduced precision, named in the console line of the comparison
REDUCED_PRECISION_MEASURES = {"TF32": "eager TF32 on vs off", "FP16": "eager FP16 vs FP32"}

# Range and bisection steps of the search of the factor of the tolerances within which the reduced precision moves the
# eager outputs, from 1 / PRECISION_SEARCH_RANGE to PRECISION_SEARCH_RANGE; a difference beyond the range (another
# number of detections, a shape) leaves the tolerances of the specification unchanged
PRECISION_SEARCH_RANGE = 1e4
PRECISION_SEARCH_STEPS = 40


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


def compare_at_tolerances(reference_outputs: List[np.ndarray], candidate_outputs: List[np.ndarray],
                          specification: Dict[str, Any], tolerances: Dict[str, float]) -> bool:
    """
    Tells whether two sets of outputs agree by the comparison of the export specification at the given tolerances, the
    test repeated by the searches of the reduced precision differences of the eager model (find_precision_scale,
    find_precision_share).

    Args:
        reference_outputs (List[np.ndarray]): The FP32 eager outputs.
        candidate_outputs (List[np.ndarray]): The compared outputs, the eager ones in the reduced precision.
        specification (Dict[str, Any]): The export specification of the task, giving its 'compare_outputs'.
        tolerances (Dict[str, float]): The tolerances of the comparison.

    Returns:
        bool: True when the comparison reports no mismatch.
    """
    return not specification["compare_outputs"](reference=reference_outputs, candidate=candidate_outputs, **tolerances)


def scale_tolerances(tolerances: Dict[str, float], scale: float) -> Dict[str, float]:
    """
    Multiplies every difference tolerance of an export specification (the keys ending with 'tolerance') by one scale,
    the minimum shares of equal outputs set to zero, the tolerances of the search of find_precision_scale.

    Args:
        tolerances (Dict[str, float]): The comparison tolerances of the export specification.
        scale (float): The factor of every difference tolerance.

    Returns:
        Dict[str, float]: The scaled tolerances, with the keys of the given ones.
    """
    return {name: value * scale if name.endswith("tolerance") else 0.0 for name, value in tolerances.items()}


def find_precision_scale(reference_outputs: List[np.ndarray], reduced_precision_outputs: List[np.ndarray],
                         specification: Dict[str, Any], tolerances: Dict[str, float]) -> Optional[float]:
    """
    Measures how far a reduced precision (TF32, FP16) moves the eager outputs on the check batch, in multiples of the
    tolerances of the export specification, for derive_precision_tolerances: the smallest factor of every difference
    tolerance at once at which the eager outputs in the reduced precision agree with the FP32 ones, the shares of equal
    outputs set to zero, found by bisection in log scale. One factor for all the tolerances keeps those that define
    what is compared (the position pairing the keypoint candidates, the score pairing the detections) as tight as the
    others, which a search of one tolerance with the others loosened would not.

    Args:
        reference_outputs (List[np.ndarray]): The FP32 eager outputs (TF32 off), the reference of the engine check.
        reduced_precision_outputs (List[np.ndarray]): The eager outputs in the reduced precision, on the same check
            batch.
        specification (Dict[str, Any]): The export specification of the task, giving its 'compare_outputs'.
        tolerances (Dict[str, float]): The comparison tolerances of the export specification.

    Returns:
        Optional[float]: The factor, None when the outputs differ beyond PRECISION_SEARCH_RANGE times the tolerances or
            by something no tolerance covers (a shape, a number of detections).
    """
    lower_scale, upper_scale = 1.0 / PRECISION_SEARCH_RANGE, PRECISION_SEARCH_RANGE
    if not compare_at_tolerances(reference_outputs=reference_outputs, candidate_outputs=reduced_precision_outputs,
                                 specification=specification,
                                 tolerances=scale_tolerances(tolerances=tolerances, scale=upper_scale)):
        return None
    if compare_at_tolerances(reference_outputs=reference_outputs, candidate_outputs=reduced_precision_outputs,
                             specification=specification,
                             tolerances=scale_tolerances(tolerances=tolerances, scale=lower_scale)):
        return lower_scale

    # Bisection in log scale between a failing and a passing factor
    for _ in range(PRECISION_SEARCH_STEPS):
        middle_scale = float(np.sqrt(lower_scale * upper_scale))
        if compare_at_tolerances(reference_outputs=reference_outputs, candidate_outputs=reduced_precision_outputs,
                                 specification=specification,
                                 tolerances=scale_tolerances(tolerances=tolerances, scale=middle_scale)):
            upper_scale = middle_scale
        else:
            lower_scale = middle_scale
    return upper_scale


def find_precision_share(reference_outputs: List[np.ndarray], reduced_precision_outputs: List[np.ndarray],
                         specification: Dict[str, Any], tolerances: Dict[str, float],
                         share_name: str) -> Optional[float]:
    """
    Measures the share of outputs a reduced precision leaves equal in the eager model on the check batch (keypoint
    candidates at the same position with the same tag, pixels of the same class), for derive_precision_tolerances: the
    largest value of one minimum share at which the eager outputs in the reduced precision agree with the FP32 ones, at
    the given difference tolerances (those within which the reduced precision agrees, since they define what counts as
    equal) and the other shares set to zero, found by bisection.

    Args:
        reference_outputs (List[np.ndarray]): The FP32 eager outputs (TF32 off), the reference of the engine check.
        reduced_precision_outputs (List[np.ndarray]): The eager outputs in the reduced precision, on the same check
            batch.
        specification (Dict[str, Any]): The export specification of the task, giving its 'compare_outputs'.
        tolerances (Dict[str, float]): The difference tolerances the share is measured at, with the shares.
        share_name (str): The share measured, a key not ending with 'tolerance' ('minimum_equal_share').

    Returns:
        Optional[float]: The share of equal outputs between the reduced precision and FP32, None when the outputs
            differ by something no share covers.
    """
    # Tolerances kept, every other share dropped
    unshared = {name: value if name.endswith("tolerance") else 0.0 for name, value in tolerances.items()}
    if not compare_at_tolerances(reference_outputs=reference_outputs, candidate_outputs=reduced_precision_outputs,
                                 specification=specification, tolerances={**unshared, share_name: 0.0}):
        return None
    lower_share, upper_share = 0.0, 1.0
    if compare_at_tolerances(reference_outputs=reference_outputs, candidate_outputs=reduced_precision_outputs,
                             specification=specification, tolerances={**unshared, share_name: upper_share}):
        return upper_share

    # Bisection between a passing and a failing share
    for _ in range(PRECISION_SEARCH_STEPS):
        middle_share = (lower_share + upper_share) / 2.0
        if compare_at_tolerances(reference_outputs=reference_outputs, candidate_outputs=reduced_precision_outputs,
                                 specification=specification, tolerances={**unshared, share_name: middle_share}):
            lower_share = middle_share
        else:
            upper_share = middle_share
    return lower_share


def derive_precision_tolerances(reference_outputs: List[np.ndarray], reduced_precision_outputs: List[np.ndarray],
                                specification: Dict[str, Any], tolerances: Dict[str, float]) -> Dict[str, Any]:
    """
    Gives the tolerances at which an engine that missed its tolerances is still accepted against the FP32 eager model,
    within the roundings of its reduced precision (TF32 for an FP32 engine, FP16 for an FP16 engine): every difference
    tolerance multiplied by REDUCED_PRECISION_DIFFERENCE_FACTOR times the factor the reduced precision needs in the
    eager model on the same check batch (find_precision_scale), every minimum share of equal outputs allowing
    REDUCED_PRECISION_DISAGREEMENT_FACTOR times the share the reduced precision changes at those tolerances
    (find_precision_share), the values of the specification staying the floor, and outputs the reduced precision
    changes beyond any tolerance keeping them.

    Args:
        reference_outputs (List[np.ndarray]): The FP32 eager outputs (TF32 off), the reference of the engine check.
        reduced_precision_outputs (List[np.ndarray]): The eager outputs in the reduced precision, on the same check
            batch.
        specification (Dict[str, Any]): The export specification of the task, giving its 'compare_outputs'.
        tolerances (Dict[str, float]): The comparison tolerances of the export specification.

    Returns:
        Dict[str, Any]: The 'precision_scale' (None when not measurable), the 'precision_shares' measured for every
            minimum share (None when not measurable) and the 'tolerances' of the comparison, with the keys of the
            given ones.
    """
    # Difference tolerances: REDUCED_PRECISION_DIFFERENCE_FACTOR times the factor the reduced precision needs, never
    # below those of the specification
    precision_scale = find_precision_scale(reference_outputs=reference_outputs,
                                           reduced_precision_outputs=reduced_precision_outputs,
                                           specification=specification, tolerances=tolerances)
    difference_scale = 1.0 if precision_scale is None else max(1.0,
                                                               REDUCED_PRECISION_DIFFERENCE_FACTOR * precision_scale)
    measured_scale = 1.0 if precision_scale is None else max(1.0, precision_scale)
    precision_tolerances = {name: value * difference_scale for name, value in tolerances.items()
                            if name.endswith("tolerance")}

    # Minimum shares: REDUCED_PRECISION_DISAGREEMENT_FACTOR times the share the reduced precision changes, measured
    # within the tolerances it needs
    precision_shares = {}
    for name, value in tolerances.items():
        if name.endswith("tolerance"):
            continue
        precision_shares[name] = find_precision_share(
            reference_outputs=reference_outputs, reduced_precision_outputs=reduced_precision_outputs,
            specification=specification,
            tolerances={**scale_tolerances(tolerances=tolerances, scale=measured_scale), name: value}, share_name=name)
        precision_tolerances[name] = (value if precision_shares[name] is None else min(value, max(
            0.0, 1.0 - REDUCED_PRECISION_DISAGREEMENT_FACTOR * (1.0 - precision_shares[name]))))
    return {"precision_scale": precision_scale, "precision_shares": precision_shares,
            "tolerances": {name: precision_tolerances[name] for name in tolerances}}


def describe_precision_tolerances(precision_comparison: Dict[str, Any]) -> str:
    """
    Writes the console line of an engine compared with the FP32 eager model within the roundings of its reduced
    precision: the factor the reduced precision needs in the eager model, then every tolerance or share the comparison
    used, a share with the one the reduced precision leaves.

    Args:
        precision_comparison (Dict[str, Any]): The 'precision_comparison' of check_tensorrt_engine, from
            derive_precision_tolerances, with its 'reduced_precision' ('TF32' or 'FP16').

    Returns:
        str: The line, such as 'eager TF32 on vs off within 2.37 x the tolerances, compared at 4 x that and 2 x its
            disagreement: tolerance 9.48e-04'.
    """
    measure = REDUCED_PRECISION_MEASURES[precision_comparison["reduced_precision"]]
    scale = precision_comparison["precision_scale"]
    measured = "beyond any tolerance" if scale is None else f"within {scale:.2f} x the tolerances"
    descriptions = []
    for name, value in precision_comparison["tolerances"].items():
        share_description = ""
        if name in precision_comparison["precision_shares"]:
            share = precision_comparison["precision_shares"][name]
            share_description = f" ({measure} not measurable)" if share is None else f" ({measure} {share:.3f})"
        descriptions.append(f"{name} {value:.2e}{share_description}")
    return (f"{measure} {measured}, compared at {REDUCED_PRECISION_DIFFERENCE_FACTOR} x that and "
            f"{REDUCED_PRECISION_DISAGREEMENT_FACTOR} x its disagreement: " + ", ".join(descriptions))


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
                          specification: Dict[str, Any], tolerances: Dict[str, float],
                          precision_fallback: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Serves the staged ONNX model of an export with its TensorRT config.pbtxt by a temporary Triton container on free
    ports, sends the check batches in one request, which builds the engine (ONNX Runtime builds it at the first request
    of a shape), and compares the outputs with the eager reference (compare_engine_outputs). An engine missing the
    tolerances is compared again with the FP32 eager model within the roundings of its reduced precision measured on
    the eager model (derive_precision_tolerances), TF32 for an FP32 engine, FP16 for an FP16 engine, when the fallback
    is given. A model Triton fails to load, build or run is a mismatch quoting the error and the error lines of the
    server log. The container is removed whatever happens.

    Args:
        staged_repository (Path): The model repository of the staged export, served read only.
        engine_cache_folder (Path): The folder receiving the TensorRT engines, shared with the test split parity.
        log_path (Path): The file receiving the server log.
        model_name (str): The Triton model name of the ONNX export.
        check_inputs (Dict[str, np.ndarray]): The float32 check batch of every input, keyed by input name.
        reference_outputs (List[np.ndarray]): The eager outputs on the check batches, the FP32 model for an FP32
            engine, its FP16 copy for an FP16 engine.
        specification (Dict[str, Any]): The export specification of the task.
        tolerances (Dict[str, float]): The tolerances of the comparison with reference_outputs.
        precision_fallback (Optional[Dict[str, Any]]): The comparison of an engine missing the tolerances: its
            'reduced_precision' ('TF32' or 'FP16'), the FP32 eager outputs 'reference_outputs' and the eager outputs
            in the reduced precision 'reduced_precision_outputs' (TF32 on, or the FP16 copy), compared at the
            'comparison_tolerances' of the specification; None for an engine compared at the tolerances only.

    Returns:
        Dict[str, Any]: The 'mismatches' (empty when the engine agrees), the 'precision_comparison' (None when the
            tolerances decided, otherwise its 'reduced_precision', the 'tolerances_mismatches' at the tolerances and
            the 'precision_scale', 'precision_shares' and 'tolerances' of derive_precision_tolerances), the
            'server_start_seconds' and the 'engine_check_seconds' (model load, engine build, request and
            comparison).
    """
    output_names = [output["name"] for output in specification["outputs"]]
    start_time = time.time()
    server = start_temporary_triton_server(container_name_prefix=TENSORRT_CHECK_CONTAINER_PREFIX,
                                           repository=staged_repository, engine_cache_folder=engine_cache_folder)
    server_start_seconds = time.time() - start_time

    # The request builds the engine and runs it, a failure of Triton being recorded with its first line
    check_start_time = time.time()
    server_error, precision_comparison = None, None
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

            # An engine missing the tolerances, compared again with the FP32 eager model within the roundings of its
            # reduced precision in the eager model
            if mismatches and precision_fallback is not None:
                precision_comparison = derive_precision_tolerances(
                    reference_outputs=precision_fallback["reference_outputs"],
                    reduced_precision_outputs=precision_fallback["reduced_precision_outputs"],
                    specification=specification, tolerances=specification["comparison_tolerances"])
                precision_comparison["reduced_precision"] = precision_fallback["reduced_precision"]
                precision_comparison["tolerances_mismatches"] = mismatches
                mismatches = compare_engine_outputs(reference_outputs=precision_fallback["reference_outputs"],
                                                    engine_outputs=engine_outputs, output_names=output_names,
                                                    specification=specification,
                                                    tolerances=precision_comparison["tolerances"])
        except (TritonServerError, InferenceServerException, urllib.error.URLError) as error:
            server_error = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
    finally:
        stop_triton_server(server=server, log_path=log_path)
    engine_check_seconds = time.time() - check_start_time

    # An engine Triton could not build or run, with the errors of its log, the staging folder holding it being removed
    if server_error is not None:
        mismatches = [f"Triton failed to build or run the engine: {server_error}"]
        mismatches += [f"  log: {line}" for line in read_server_log_errors(log_path=log_path)]
    return {"mismatches": mismatches, "precision_comparison": precision_comparison,
            "server_start_seconds": server_start_seconds,
            "engine_check_seconds": engine_check_seconds}
