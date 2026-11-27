"""
Writes an inference model to the files of the export formats served by Triton: a traced TorchScript file for the
PyTorch (libtorch) backend, in FP32 or with its network in FP16 (convert_to_half_precision), and an ONNX file with
dynamic batch (and frame) dimensions for the ONNX Runtime backend, optionally accelerated by TensorRT, without the
debugging metadata of the exporter, with the warnings of the libraries hidden during the export. Used by
deployment/export_model.py, which checks the exported files and installs them in the Triton model repository, by the
speed benchmark of the trainers (trainers/speed_benchmark.py), which times every format in the training process, and by
the Export Sizes panel of the launch (deployment/export_size_estimates.py), which measures the TorchScript file.
"""

import copy
import logging
import warnings
import onnx
import torch

from onnxscript import opset18
from onnxscript.onnx_types import FLOAT, INT64
from pathlib import Path
from torch import nn
from typing import Any, Dict, List, Tuple

from vision_training.models.detection.postprocessing import STACKED_GROUP_SUPPRESSION_OPERATOR
from vision_training.utilities.os_utilities import format_path_link

# Largest number of boxes NonMaxSuppression keeps per batch and class, no limit, as in the translation of the
# torchvision nms by onnxscript
MAXIMUM_SUPPRESSION_OUTPUTS = 2 ** 63 - 1

# Modules whose weights the FP16 TorchScript export holds in FP16, run on the tensor cores of the GPU; the normalization
# layers, the recurrent layers, the other parameters and every buffer (running statistics, anchors, logit temperature)
# stay FP32
HALF_PRECISION_MODULE_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d,
                               nn.ConvTranspose3d, nn.Linear, nn.MultiheadAttention)

# Recurrent layers of the FP16 TorchScript export, kept FP32 and run without autocast (FullPrecisionRecurrentLayer)
RECURRENT_MODULE_TYPES = (nn.LSTM, nn.GRU, nn.RNN)

# Inputs of the comparison model (TASK_INPUT_NAMES of export_specifications.py)
PAIR_INPUT_NAMES = ["before_images", "after_images"]

# Inputs of the detectors and instance segmenters, the images and the window of every image (TASK_INPUT_NAMES of
# export_specifications.py)
WINDOWED_IMAGE_INPUT_NAMES = ["images", "image_windows"]


def cast_to_full_precision(value: Any) -> Any:
    """
    Casts a value given to a recurrent layer of the FP16 TorchScript export to FP32: a floating tensor, the data of a
    PackedSequence (its batch sizes and indices left as integers), every member of a tuple (the hidden and cell states
    of an LSTM), any other value being returned as it is.

    Args:
        value (Any): The argument of the recurrent layer.

    Returns:
        Any: The value with its floating tensors in FP32.
    """
    # A PackedSequence is a tuple, told apart first so that only its data is cast
    if isinstance(value, nn.utils.rnn.PackedSequence):
        return value._replace(data=value.data.float())
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(cast_to_full_precision(value=item) for item in value)
    return value


class FullPrecisionRecurrentLayer(nn.Module):
    """
    Recurrent layer of the FP16 TorchScript export (convert_to_half_precision), run in FP32 with autocast turned off
    around it: the tracer does not record the casts autocast makes around an LSTM or GRU, so that the traced graph
    would replay it in FP32 while the layers after it were traced on its FP16 outputs. Its inputs are cast to FP32,
    the precision of its weights, and its FP32 outputs are cast by autocast again in the layers after it, every cast
    being written into the graph.
    """

    def __init__(self, recurrent_layer: nn.Module) -> None:
        """
        Wraps an FP32 recurrent layer of the copied inference model.

        Args:
            recurrent_layer (nn.Module): The LSTM, GRU or RNN layer.
        """
        super(FullPrecisionRecurrentLayer, self).__init__()
        self.recurrent_layer = recurrent_layer

    def forward(self, *arguments: Any, **keyword_arguments: Any) -> Any:
        """
        Runs the recurrent layer in FP32 with the arguments of its own forward.

        Args:
            *arguments (Any): The positional arguments of the recurrent layer (the sequence, the initial state).
            **keyword_arguments (Any): Its keyword arguments ('input', 'hx').

        Returns:
            Any: The outputs of the recurrent layer, in FP32.
        """
        with torch.autocast(device_type="cuda", enabled=False):
            return self.recurrent_layer(*[cast_to_full_precision(value=argument) for argument in arguments],
                                        **{name: cast_to_full_precision(value=value)
                                           for name, value in keyword_arguments.items()})


def restore_full_precision_outputs(outputs: torch.Tensor | Tuple[torch.Tensor, ...]) -> torch.Tensor | Tuple[
        torch.Tensor, ...]:
    """
    Casts the floating outputs of the FP16 network of a TorchScript export back to FP32, the type of the outputs of
    its Triton configuration and of the eager model, the integer outputs (labels, counts, class maps) being left as
    they are.

    Args:
        outputs (torch.Tensor | Tuple[torch.Tensor, ...]): The output, or the tuple of outputs, of the inference model.

    Returns:
        torch.Tensor | Tuple[torch.Tensor, ...]: The outputs, floating ones in FP32, in the same structure.
    """
    if isinstance(outputs, torch.Tensor):
        return outputs.float() if outputs.is_floating_point() else outputs
    return tuple(output.float() if output.is_floating_point() else output for output in outputs)


class HalfPrecisionImageModel(nn.Module):
    """
    FP16 TorchScript version of the inference model of an image or sequence task, traced by export_torchscript: its
    forward runs the inference model under the automatic mixed precision of PyTorch (FP16 autocast on the GPU), whose
    casts the tracer writes into the graph, so that the served file computes the convolutions, linear and attention
    layers in FP16 while the operators autocast keeps in FP32 (softmax, normalizations, exponentials, sums) and the
    operations meeting an FP32 buffer (anchors, logit temperature) run in FP32, then returns FP32 outputs. Its forward
    argument keeps the input name of the task, which the Triton PyTorch backend matches with the model configuration.
    """

    def __init__(self, inference_model: nn.Module) -> None:
        """
        Wraps the inference model whose weights convert_to_half_precision converted.

        Args:
            inference_model (nn.Module): The inference model, its HALF_PRECISION_MODULE_TYPES modules in FP16.
        """
        super(HalfPrecisionImageModel, self).__init__()
        self.inference_model = inference_model

    def forward(self, images: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        """
        Runs the inference model in mixed precision on FP32 images or frames.

        Args:
            images (torch.Tensor): The FP32 input of the task, as the FP32 export takes it.

        Returns:
            torch.Tensor | Tuple[torch.Tensor, ...]: The outputs of the inference model, floating ones in FP32.
        """
        with torch.autocast(device_type="cuda", dtype=torch.float16, cache_enabled=False):
            outputs = self.inference_model(images=images)
        return restore_full_precision_outputs(outputs=outputs)


class HalfPrecisionWindowedImageModel(nn.Module):
    """
    FP16 TorchScript version of the inference detector or instance segmenter, the counterpart of HalfPrecisionImageModel
    for their two inputs, its forward arguments keeping the names of the images and of their windows. The windows stay
    FP32 and the boxes are clipped to them in the type of the boxes.
    """

    def __init__(self, inference_model: nn.Module) -> None:
        """
        Wraps the inference detector whose weights convert_to_half_precision converted.

        Args:
            inference_model (nn.Module): The inference detector, its HALF_PRECISION_MODULE_TYPES modules in FP16.
        """
        super(HalfPrecisionWindowedImageModel, self).__init__()
        self.inference_model = inference_model

    def forward(self, images: torch.Tensor, image_windows: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Runs the inference detector in mixed precision on FP32 images and their FP32 windows.

        Args:
            images (torch.Tensor): The FP32 [B, 3, H, W] images.
            image_windows (torch.Tensor): The FP32 [B, 4] window of every image in the network input.

        Returns:
            Tuple[torch.Tensor, ...]: The outputs of the inference detector, floating ones in FP32.
        """
        with torch.autocast(device_type="cuda", dtype=torch.float16, cache_enabled=False):
            outputs = self.inference_model(images=images, image_windows=image_windows)
        return restore_full_precision_outputs(outputs=outputs)


class HalfPrecisionPairModel(nn.Module):
    """
    FP16 TorchScript version of the inference model of the sequence comparison, the two-input counterpart of
    HalfPrecisionImageModel, its forward arguments keeping the names of the before and after members.
    """

    def __init__(self, inference_model: nn.Module) -> None:
        """
        Wraps the inference comparator whose weights convert_to_half_precision converted.

        Args:
            inference_model (nn.Module): The inference comparator, its HALF_PRECISION_MODULE_TYPES modules in FP16.
        """
        super(HalfPrecisionPairModel, self).__init__()
        self.inference_model = inference_model

    def forward(self, before_images: torch.Tensor, after_images: torch.Tensor) -> torch.Tensor:
        """
        Runs the inference comparator in mixed precision on the FP32 frames of both members.

        Args:
            before_images (torch.Tensor): The FP32 frames of the before member.
            after_images (torch.Tensor): The FP32 frames of the after member.

        Returns:
            torch.Tensor: The FP32 class probabilities of every pair.
        """
        with torch.autocast(device_type="cuda", dtype=torch.float16, cache_enabled=False):
            outputs = self.inference_model(before_images=before_images, after_images=after_images)
        return restore_full_precision_outputs(outputs=outputs)


def convert_to_half_precision(inference_model: nn.Module, input_names: List[str]) -> nn.Module:
    """
    Builds the FP16 version of an inference model of any task for the FP16 TorchScript export
    (export_model.py with torchscript_precision 'FP16'): a copy whose convolution, linear and attention weights are
    converted to FP16, about halving the file, the normalization layers and the buffers staying FP32 as in the automatic
    mixed precision of PyTorch and the recurrent layers running in FP32 without autocast
    (FullPrecisionRecurrentLayer), wrapped so that it runs under FP16 autocast and returns FP32 outputs. The
    inputs stay the FP32 batches of the input contract, so that a client sends the same requests to both precisions.
    The eager model of the export is left untouched; the copy runs on the GPU only, as Triton serves it.

    Args:
        inference_model (nn.Module): The FP32 inference model of the export, on the GPU, with its logit temperature.
        input_names (List[str]): The input names of the task, in the order of the forward arguments.

    Returns:
        nn.Module: The FP16 version, in evaluation mode, with the forward arguments of the inference model.

    Raises:
        ValueError: If a weight of the inference model is not on the GPU, where the FP16 autocast of the copy runs.
    """
    # The autocast of the copy runs on CUDA only: a CPU model would reach FP16 weights with FP32 inputs
    if any(parameter.device.type != "cuda" for parameter in inference_model.parameters()):
        raise ValueError("Expected the inference model on the GPU for its FP16 TorchScript export: the FP16 autocast "
                         "of its copy runs on CUDA only.")
    half_precision_model = copy.deepcopy(x=inference_model)
    # Weights of the layers computed on the tensor cores in FP16, their own buffers following (none for these types)
    for module in half_precision_model.modules():
        if isinstance(module, HALF_PRECISION_MODULE_TYPES):
            module.half()

    # Recurrent layers replaced in their parent by their FP32 wrapper, the list of children read before any change,
    # their weights gathered again into the single block cuDNN reads, which the copy splits
    recurrent_children = [(parent, child_name, child) for parent in half_precision_model.modules()
                          for child_name, child in parent.named_children() if isinstance(child, RECURRENT_MODULE_TYPES)]
    for parent, child_name, child in recurrent_children:
        child.flatten_parameters()
        setattr(parent, child_name, FullPrecisionRecurrentLayer(recurrent_layer=child))
    # Wrapper keeping the forward arguments of the task: the two members of the comparison, the images and their
    # windows of the detectors, or the images
    wrapper_classes = {tuple(PAIR_INPUT_NAMES): HalfPrecisionPairModel,
                       tuple(WINDOWED_IMAGE_INPUT_NAMES): HalfPrecisionWindowedImageModel}
    wrapper_class = wrapper_classes.get(tuple(input_names), HalfPrecisionImageModel)
    return wrapper_class(inference_model=half_precision_model).eval()


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


def list_graph_metadata_holders(graph: onnx.GraphProto) -> List[Any]:
    """
    Lists every element of an ONNX graph that can carry metadata_props (the graph, its nodes, inputs, outputs,
    intermediate shapes and initializers), the graphs nested in the attributes of its nodes (the branches and bodies of
    control flow operators) included, for strip_exporter_metadata.

    Args:
        graph (onnx.GraphProto): The graph to walk.

    Returns:
        List[Any]: The graph and its elements, then those of the nested graphs.
    """
    metadata_holders = [graph, *graph.node, *graph.input, *graph.output, *graph.value_info, *graph.initializer]
    # Branches and loop bodies are graphs of their own, stored in the attributes of their node
    for node in graph.node:
        for attribute in node.attribute:
            nested_graphs = ([attribute.g] if attribute.HasField("g") else []) + list(attribute.graphs)
            for nested_graph in nested_graphs:
                metadata_holders += list_graph_metadata_holders(graph=nested_graph)
    return metadata_holders


def strip_exporter_metadata(model_proto: onnx.ModelProto) -> int:
    """
    Removes the debugging metadata the dynamo exporter writes into the ONNX graph: on every node its Python stack trace
    (with the absolute paths of the files of this machine), its module path and class hierarchy and the text of its FX
    node, on the inputs, shapes and weights their original names, and on the graph the signature of the traced
    program, which together make 87 to 94 % of 'model.onnx' (1.37 MB for the run_9 detector, whose graph alone takes
    0.16 MB) and which no runtime reads. The metadata of the model itself (the input contract written by
    write_onnx_input_contract after the export) is kept. Called by export_onnx, so that the shipped files carry no
    local path and the ONNX size estimates of export_size_estimates.py hold within their calibrated margin.

    Args:
        model_proto (onnx.ModelProto): The exported model, loaded without its external weights, changed in place.

    Returns:
        int: The number of metadata entries removed.
    """
    removed_entries = 0
    # The main graph and its local functions, every element with metadata emptied
    metadata_holders = list_graph_metadata_holders(graph=model_proto.graph)
    for function in model_proto.functions:
        metadata_holders += [function, *function.node]
    for metadata_holder in metadata_holders:
        removed_entries += len(metadata_holder.metadata_props)
        del metadata_holder.metadata_props[:]
    return removed_entries


def export_onnx(inference_model: nn.Module, example_inputs: Dict[str, torch.Tensor], output_path: Path,
                specification: Dict[str, Any], maximum_batch_size: int) -> None:
    """
    Exports an inference model to an ONNX file served by Triton's ONNX Runtime backend (optionally accelerated by
    TensorRT), with one dynamic batch dimension shared by every input up to the maximum batch size and, for the
    sequence models, a dynamic number of frames per input up to the maximum sequence length of the training, so that
    the before and after members of the comparison model may have different numbers of frames. The weights go to
    'model.onnx.data' next to it, and the debugging metadata of the exporter is removed from the graph
    (strip_exporter_metadata).

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

    # Frames of the outputs per frame made dynamic again where the export fixed them and the debugging metadata of the
    # exporter removed, the graph file written again with the external weights file left untouched, then the dynamic
    # dimensions of the file checked
    model_proto = onnx.load(f=str(output_path), load_external_data=False)
    restore_dynamic_output_frames(model_proto=model_proto, specification=specification)
    strip_exporter_metadata(model_proto=model_proto)
    onnx.save(proto=model_proto, f=str(output_path))
    fixed_dimensions = find_fixed_dynamic_dimensions(model_proto=model_proto, specification=specification)
    if fixed_dimensions:
        raise RuntimeError(f"The ONNX Export Wrote Fixed Dimensions : {format_path_link(path=output_path)}\n" +
                           "\n".join(fixed_dimensions))
