"""
Writes an inference model to the files of the export formats served by Triton: a traced TorchScript file for the
PyTorch (libtorch) backend and an ONNX file with dynamic batch (and frame) dimensions for the ONNX Runtime backend,
optionally accelerated by TensorRT, with the warnings of the libraries hidden during the export. Used by
deployment/export_model.py, which checks the exported files and installs them in the Triton model repository, and by
the speed benchmark of the trainers (trainers/speed_benchmark.py), which times every format in the training process.
"""

import logging
import warnings
import onnx
import torch

from onnxscript import opset18
from onnxscript.onnx_types import FLOAT, INT64
from pathlib import Path
from torch import nn
from typing import Any, Dict, List

from vision_training.models.detection.postprocessing import STACKED_GROUP_SUPPRESSION_OPERATOR
from vision_training.utilities.os_utilities import format_path_link

# Largest number of boxes NonMaxSuppression keeps per batch and class, no limit, as in the translation of the
# torchvision nms by onnxscript
MAXIMUM_SUPPRESSION_OUTPUTS = 2 ** 63 - 1


def translate_stacked_group_suppression(boxes: FLOAT, scores: FLOAT, iou_threshold: float) -> INT64:
    """
    Writes the custom operator suppress_stacked_groups of the detectors (vision_training/models/detection/
    postprocessing.py) into the ONNX graph as one NonMaxSuppression whose batches are the stacked groups, its single
    class holding their scores, so that no box is moved by an offset that TensorRT FP16 would overflow; the selected
    (group, class, box) triplets become the flat indices group times N plus box that the operator returns.

    Args:
        boxes (FLOAT): The [G, N, 4] boxes of every group.
        scores (FLOAT): Their [G, N] scores.
        iou_threshold (float): The overlap above which the lower scoring box of a pair of the same group is suppressed.

    Returns:
        INT64: The flat indices of the kept boxes.
    """
    # Selected [K, 3] triplets of group, class (always 0) and box
    selected_triplets = opset18.NonMaxSuppression(boxes=boxes, scores=opset18.Unsqueeze(data=scores, axes=[1]),
                                                  max_output_boxes_per_class=MAXIMUM_SUPPRESSION_OUTPUTS,
                                                  iou_threshold=iou_threshold)
    group_indices = opset18.Reshape(data=opset18.Slice(data=selected_triplets, starts=[0], ends=[1], axes=[1]),
                                    shape=[-1])
    box_indices = opset18.Reshape(data=opset18.Slice(data=selected_triplets, starts=[2], ends=[3], axes=[1]),
                                  shape=[-1])

    # Flat index of every kept box in the [G, N] stack
    number_boxes = opset18.Shape(data=scores, start=1, end=2)
    return opset18.Add(A=opset18.Mul(A=group_indices, B=number_boxes), B=box_indices)


def hide_known_export_warnings() -> None:
    """
    Hides the warnings printed by PyTorch, torchvision and the ONNX library during the export, which come from inside
    these libraries and cannot be fixed in the project code. They are harmless for this export: the deprecation
    notices of TorchScript, of a torchvision helper and of a size check emitted by torch.export for the dynamic
    dimensions, the tracer notice raised inside torchvision's RoIAlign, the notice of torch.export about the weights
    the GRU and LSTM layers flatten while running, the notice of the ONNX exporter naming the batch dimension shared
    by the inputs of the comparison model only once, the ONNX notice about an empty attribute defaulting to integers,
    and the notice of the ONNX optimizer leaving the operators with several outputs unfolded (the split of the weights
    of the attention layers). The exports are checked against the eager model anyway.
    """
    # Deprecation notices of torch.jit.trace / trace_method / load, of a torchvision export helper and of the size
    # checks written by torch.export
    warnings.filterwarnings(action="ignore", message=r"`torch\.jit\.", category=FutureWarning)
    warnings.filterwarnings(action="ignore", message=r"`create_unbacked_symint` is deprecated", category=FutureWarning)
    warnings.filterwarnings(action="ignore", message=r"_check_is_size will be removed", category=FutureWarning)

    # Weights flattened by the GRU and LSTM layers at every call, reported by torch.export as assigned attributes
    warnings.filterwarnings(action="ignore", message=r"The tensor attributes .*_flat_weights", category=UserWarning)

    # Batch dimension shared by several inputs, which the ONNX exporter names after the first input only
    warnings.filterwarnings(action="ignore", message=r"# The axis name: .* will not be used, since it shares the same",
                            category=UserWarning)

    # Tracer notice raised by the shape assertion inside torchvision's RoIAlign
    warnings.filterwarnings(action="ignore", message=r"Converting a tensor to a Python boolean",
                            category=torch.jit.TracerWarning)

    # The ONNX libraries report the empty attribute and the unfolded operators with several outputs through the logging
    # module rather than the warnings module
    logging.getLogger(name="onnx_ir._convenience").setLevel(level=logging.ERROR)
    logging.getLogger(name="onnxscript.optimizer._constant_folding").setLevel(level=logging.ERROR)


def export_torchscript(inference_model: nn.Module, example_inputs: Dict[str, torch.Tensor], output_path: Path) -> None:
    """
    Traces an inference model into a self-contained TorchScript file served by Triton's PyTorch (libtorch) backend.
    The traced graph keeps the batch dimension (and the numbers of frames of the sequence models) dynamic, and the
    names of the forward arguments, which the Triton PyTorch backend matches with the input names of the model
    configuration.

    Args:
        inference_model (nn.Module): The inference model to export, in evaluation mode.
        example_inputs (Dict[str, torch.Tensor]): The example batch of every input, keyed by name in the order of the
            forward arguments (from create_random_inputs), used to trace the graph.
        output_path (Path): The path of the written 'model.pt' file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        traced_model = torch.jit.trace(inference_model, example_inputs=tuple(example_inputs.values()),
                                       check_trace=False)
    traced_model.save(str(output_path))


def reset_recurrent_dispatch_caches() -> None:
    """
    Empties the dispatch caches PyTorch keeps for the GRU and LSTM operators. The first ONNX export of a recurrent
    model caches the decomposition of these operators, and every later export in the same process reuses it with the
    sizes of the first one, silently writing a fixed number of frames into the exported model. Emptying the caches
    before every export makes each export trace the operators again. The caches are a private attribute of PyTorch,
    so their absence after an upgrade is tolerated, the dimensions of every export being checked afterwards by
    find_fixed_dynamic_dimensions.
    """
    for recurrent_operator in [torch.ops.aten.gru.input, torch.ops.aten.lstm.input]:
        if hasattr(recurrent_operator, "_dispatch_cache"):
            recurrent_operator._dispatch_cache.clear()


def find_fixed_dynamic_dimensions(model_proto: onnx.ModelProto, specification: Dict[str, Any]) -> List[str]:
    """
    Lists the dimensions of an exported ONNX model that the export specification declares dynamic (the batch
    dimension of every input and output, the number of frames of the sequence inputs and of the outputs per frame) but
    that the export wrote as fixed values. Such a model would only accept requests of the traced sizes, or Triton would
    refuse to load it, errors that would only show once serving.

    Args:
        model_proto (onnx.ModelProto): The exported model, loaded without its external weights.
        specification (Dict[str, Any]): The export specification of the task, giving the inputs and their dynamic
            dimensions, and the output dimensions without the batch (-1 for the frames).

    Returns:
        List[str]: A description of every fixed dimension, empty when every declared dynamic dimension is dynamic.
    """
    graph_inputs = {graph_input.name: graph_input for graph_input in model_proto.graph.input}
    fixed_dimensions = []
    for input_specification in specification["inputs"]:
        input_dimensions = graph_inputs[input_specification["name"]].type.tensor_type.shape.dim

        # The batch dimension is always dynamic, the number of frames only for the sequence inputs
        dynamic_axes = {0: "batch"}
        if input_specification["dynamic_frames"]:
            dynamic_axes[1] = "frames"
        for axis, axis_name in dynamic_axes.items():
            if input_dimensions[axis].HasField("dim_value"):
                fixed_dimensions.append(f"the {axis_name} dimension of '{input_specification['name']}' is fixed to "
                                        f"{input_dimensions[axis].dim_value}")

    # The batch dimension of every output, and its frame dimension where the specification declares -1
    graph_outputs = {graph_output.name: graph_output for graph_output in model_proto.graph.output}
    for output_specification in specification["outputs"]:
        output_dimensions = graph_outputs[output_specification["name"]].type.tensor_type.shape.dim
        dynamic_axes = [0] + [axis + 1 for axis, size in enumerate(output_specification["dimensions"]) if size == -1]
        for axis in dynamic_axes:
            if output_dimensions[axis].HasField("dim_value"):
                fixed_dimensions.append(f"axis {axis} of the output '{output_specification['name']}' is fixed to "
                                        f"{output_dimensions[axis].dim_value}")
    return fixed_dimensions


def restore_dynamic_output_frames(model_proto: onnx.ModelProto, specification: Dict[str, Any]) -> List[str]:
    """
    Gives back their dynamic number of frames to the outputs per frame of an exported sequence model (the sequence
    labeller and the frame selector): the dynamo export of a recurrent layer writes the frame count of its trace into
    the shapes after it, the graph still computing one row per input frame, so that Triton refuses an output declared
    [-1, 50, 71] in the file against [-1, -1, 71] in config.pbtxt. Every output axis that the specification declares -1
    but the file fixes takes the frame dimension of the first sequence input, and the intermediate shapes written by
    the export are dropped (ONNX Runtime and TensorRT infer them again). Called by export_onnx before its dimension
    check.

    Args:
        model_proto (onnx.ModelProto): The exported model, loaded without its external weights, changed in place.
        specification (Dict[str, Any]): The export specification of the task, giving the output dimensions.

    Returns:
        List[str]: The restored outputs, as '<name> axis <axis>: <fixed size>', empty when nothing was changed.

    Raises:
        ValueError: If an output to restore has no sequence input to take its frame dimension from.
    """
    # Name of the frame dimension of the first sequence input, [batch, frames, channels, height, width]
    frame_dimension_names = [graph_input.type.tensor_type.shape.dim[1].dim_param for graph_input in
                             model_proto.graph.input if len(graph_input.type.tensor_type.shape.dim) == 5]

    # Output axes declared -1 by the specification but fixed in the file
    graph_outputs = {graph_output.name: graph_output for graph_output in model_proto.graph.output}
    restored_outputs = []
    for output_specification in specification["outputs"]:
        output_dimensions = graph_outputs[output_specification["name"]].type.tensor_type.shape.dim
        for axis, size in enumerate(output_specification["dimensions"], start=1):
            if size != -1 or not output_dimensions[axis].HasField("dim_value"):
                continue
            if not frame_dimension_names or not frame_dimension_names[0]:
                raise ValueError(f"Expected a sequence input with a dynamic frame dimension to restore the fixed axis "
                                 f"{axis} of the output '{output_specification['name']}'.")
            restored_outputs.append(f"{output_specification['name']} axis {axis}: {output_dimensions[axis].dim_value}")
            output_dimensions[axis].dim_param = frame_dimension_names[0]

    # Intermediate shapes of the export dropped once an output changed
    if restored_outputs:
        del model_proto.graph.value_info[:]
    return restored_outputs


def export_onnx(inference_model: nn.Module, example_inputs: Dict[str, torch.Tensor], output_path: Path,
                specification: Dict[str, Any], maximum_batch_size: int) -> None:
    """
    Exports an inference model to an ONNX file served by Triton's ONNX Runtime backend (optionally accelerated by
    TensorRT), with one dynamic batch dimension shared by every input up to the maximum batch size and, for the
    sequence models, a dynamic number of frames per input up to the maximum sequence length of the training, so that
    the before and after members of the comparison model may have different numbers of frames.

    Args:
        inference_model (nn.Module): The inference model to export, in evaluation mode.
        example_inputs (Dict[str, torch.Tensor]): The example batch of every input, keyed by name in the order of the
            forward arguments (from create_random_inputs), used to export the graph.
        output_path (Path): The path of the written 'model.onnx' file.
        specification (Dict[str, Any]): The export specification of the task, giving the inputs with their dynamic
            dimensions and the output names.
        maximum_batch_size (int): The largest batch size the exported model accepts.

    Raises:
        RuntimeError: If a dimension declared dynamic was written as a fixed value.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # One batch dimension shared by every input, and one number of frames per sequence input
    batch_dimension = torch.export.Dim("batch", min=1, max=maximum_batch_size)
    dynamic_shapes = {}
    for input_specification in specification["inputs"]:
        dynamic_dimensions = {0: batch_dimension}
        if input_specification["dynamic_frames"]:
            dynamic_dimensions[1] = torch.export.Dim(f"{input_specification['name']}_frames", min=1,
                                                     max=input_specification["maximum_frames"])
        dynamic_shapes[input_specification["name"]] = dynamic_dimensions

    # The recurrent operators are traced again for this export, the stacked group suppression of the detectors written
    # as a batched NonMaxSuppression
    reset_recurrent_dispatch_caches()
    torch.onnx.export(inference_model, args=tuple(example_inputs.values()), f=str(output_path),
                      input_names=list(example_inputs),
                      output_names=[output["name"] for output in specification["outputs"]],
                      dynamic_shapes=dynamic_shapes, opset_version=18, dynamo=True, verbose=False,
                      custom_translation_table={STACKED_GROUP_SUPPRESSION_OPERATOR:
                                                translate_stacked_group_suppression})

    # Frames of the outputs per frame made dynamic again where the export fixed them, the external weights file left
    # untouched, then the dynamic dimensions of the file checked
    model_proto = onnx.load(f=str(output_path), load_external_data=False)
    if restore_dynamic_output_frames(model_proto=model_proto, specification=specification):
        onnx.save(proto=model_proto, f=str(output_path))
    fixed_dimensions = find_fixed_dynamic_dimensions(model_proto=model_proto, specification=specification)
    if fixed_dimensions:
        raise RuntimeError(f"The ONNX Export Wrote Fixed Dimensions : {format_path_link(path=output_path)}\n" +
                           "\n".join(fixed_dimensions))
