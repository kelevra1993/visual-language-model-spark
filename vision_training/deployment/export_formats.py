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

from pathlib import Path
from torch import nn
from typing import Any, Dict, List

from vision_training.utilities.os_utilities import format_path_link


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
    Lists the input dimensions of an exported ONNX model that the export specification declares dynamic (the batch
    dimension of every input, and the number of frames of the sequence inputs) but that the export wrote as fixed
    values. Such a model would only accept requests of the traced sizes, an error that Triton would only report once
    serving.

    Args:
        model_proto (onnx.ModelProto): The exported model, loaded without its external weights.
        specification (Dict[str, Any]): The export specification of the task, giving the inputs and their dynamic
            dimensions.

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
    return fixed_dimensions


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

    # The recurrent operators are traced again for this export, then the dynamic dimensions of the file are checked
    reset_recurrent_dispatch_caches()
    torch.onnx.export(inference_model, args=tuple(example_inputs.values()), f=str(output_path),
                      input_names=list(example_inputs),
                      output_names=[output["name"] for output in specification["outputs"]],
                      dynamic_shapes=dynamic_shapes, opset_version=18, dynamo=True, verbose=False)
    fixed_dimensions = find_fixed_dynamic_dimensions(
        model_proto=onnx.load(f=str(output_path), load_external_data=False), specification=specification)
    if fixed_dimensions:
        raise RuntimeError(f"The ONNX Export Wrote Fixed Dimensions : {format_path_link(path=output_path)}\n" +
                           "\n".join(fixed_dimensions))
