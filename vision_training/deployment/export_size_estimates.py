"""
Size of every file a model of the training is turned into, printed in the Export Sizes panel at the launch of every
training of a task Triton serves (BaseTrainer, after the Model Size panel of TaskModel.print_model_size_summary): the
PyTorch weights in FP32 and FP16, the TorchScript file in FP32 and FP16, the ONNX graph and weights, the TensorRT
engines in FP32 and FP16 built by Triton (deployment/export_model.py and triton_configuration.py) and the training
checkpoint of every saved iteration (trainers/checkpointing.py). Each size is estimated from a formula when the formula
stays within 5 % of the real file on the tutorial runs, and measured otherwise: the FP32 TorchScript trace is written
to a temporary file, the FP16 file being that size less the bytes its FP16 weights save, while the ONNX files and the
TensorRT engines are estimated from what one forward pass of the inference model reads (the weights it uses and the
constants it computes) and from its number of operators. The constants of the formulas
and their margins were fitted on the files of real exports of the tutorial runs.
"""

import tempfile
import time
import warnings
import torch

from pathlib import Path
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

from vision_training.deployment.export_formats import (HALF_PRECISION_MODULE_TYPES, export_torchscript,
                                                       hide_known_export_warnings)
from vision_training.deployment.export_specifications import (build_inference_model, create_random_inputs,
                                                              get_export_specification)

# Largest batch size of the exports of deployment/export_model.py, which caps the batch of the traced example
EXPORT_MAXIMUM_BATCH_SIZE = 16

# Attribute marking, during the probe of the forward pass, the tensors computed from the inputs of the model
INPUT_DEPENDENCE_ATTRIBUTE = "_depends_on_export_input"

# Operators holding weights, whose TensorRT layers take more room in an engine than the other operators
WEIGHTED_OPERATORS = {"aten.convolution", "aten._convolution", "aten.cudnn_convolution", "aten.mkldnn_convolution",
                      "aten.convolution_overrideable", "aten.mm", "aten.addmm", "aten.bmm", "aten.baddbmm",
                      "aten.matmul", "aten.linear"}
# Batch normalization operators, folded by the exports into the convolution before them
BATCH_NORMALIZATION_OPERATORS = {"aten.batch_norm", "aten._batch_norm_impl_index", "aten.native_batch_norm",
                                 "aten._native_batch_norm_legit", "aten._native_batch_norm_legit_no_training",
                                 "aten.cudnn_batch_norm", "aten.miopen_batch_norm"}

# Constants of the estimates, fitted on the real exports of the ten tutorial runs of every task on 2026-11-21
# (PyTorch 2.14, onnxscript 0.7.2), the TensorRT ones on the engines of the serving benchmark and of the tutorial runs
# (Triton 26.01 with TensorRT 10.14).
# Bytes torch.save writes for a state dictionary besides the values of its tensors: per tensor (its zip records
# aligned to 64 bytes and its entry in the pickled dictionary, the characters of its key added), and once (the pickle,
# version and byte order records and the zip directory)
STATE_DICTIONARY_BYTES_PER_TENSOR = 300
STATE_DICTIONARY_FIXED_BYTES = 1300
# Bytes of the optimizer state per parameter besides its moment tensors (its 'step' tensor, its records and entry)
OPTIMIZER_BYTES_PER_PARAMETER = 550
# Bytes of a checkpoint besides the states of the model, the optimizer and the moving average (iteration, task, run
# name, label dictionary and the other metadata of the trainer)
CHECKPOINT_METADATA_BYTES = 20000
# Bytes of 'model.onnx', the graph without its weights, per operator of the forward pass, once the debugging metadata
# of the exporter is removed (from 131 to 246 bytes on the tutorial runs, probed on the CPU or the GPU)
ONNX_GRAPH_BYTES_PER_OPERATOR = 165
# Bytes of a TensorRT engine besides its weights (layer descriptions, chosen kernels, fused kernels compiled by the
# builder) per weighted operator or recurrent layer and per other operator, fitted on the relative errors of the 20
# engines Triton 26.01 built on the GB10 for the ten tutorial runs on 2026-11-22, and the range around this overhead
# holding all of them (from 0.52 to 2.12 times it: the overhead follows the operators loosely, and two builds of the
# same keypoint detector file gave engines 7 % apart), the engine of a precision holding its weights in that precision
TENSORRT_BYTES_PER_WEIGHTED_OPERATOR = 7700
TENSORRT_BYTES_PER_OTHER_OPERATOR = 2300
TENSORRT_OVERHEAD_RANGE = (0.5, 2.2)
# Margin printed next to every other estimate, the largest relative error on the tutorial runs rounded up (0.70 % for
# the PyTorch weights, 0.87 % for the checkpoints, 2.30 % for the ONNX files, on the CPU and on the GPU)
STATE_DICTIONARY_MARGIN = 0.01
CHECKPOINT_MARGIN = 0.01
ONNX_MARGIN = 0.03
# Margin of the FP16 TorchScript file estimated from the FP32 trace, the largest relative error against the real FP16
# files of the ten tutorial runs (convert_to_half_precision then export_torchscript on the GPU, 2026-11-25) rounded up:
# from 0.15 to 1.54 % below them, the casts autocast adds to the traced code being left out
TORCHSCRIPT_HALF_PRECISION_MARGIN = 0.02

# Megabytes of the panel, as in the Model Size panel, and the spaces indenting its table of formats
BYTES_PER_MEGABYTE = 1e6
TABLE_INDENT = 3


class GraphProbe(TypedDict):
    """
    What one forward pass of an inference model reads and runs, the parts of its exported graph (probe_inference_graph).
    """
    constant_bytes: int
    batch_normalization_bytes: int
    weighted_operators: int
    other_operators: int
    constant_storages: Set[int]


class ExportSizeRow(TypedDict):
    """
    One format of the Export Sizes panel: its size, or the range of its size, and how it was obtained.
    """
    label: str
    lowest_bytes: Optional[int]
    highest_bytes: Optional[int]
    method: str


class ExportSizes(TypedDict):
    """
    The sizes of the Export Sizes panel (estimate_export_sizes), with the forward pass they were estimated from (None
    when it failed).
    """
    exported_parameters: int
    total_parameters: int
    rows: List[ExportSizeRow]
    seconds: float
    errors: List[str]
    probe: Optional[GraphProbe]


class ExportGraphProbe(TorchDispatchMode):
    """
    Dispatch mode recording, during one forward pass of an inference model, what its exported graph holds: the
    operators of the pass, as the nodes of the graph, by kind, and the memory of the tensors that do not depend on the
    inputs but are read by an operator that does (the weights the pass uses and the constants it computes, such as the
    anchors of the detectors), which the exports write as their weights. A tensor computed by an operator reading a
    tensor computed from the inputs is marked by INPUT_DEPENDENCE_ATTRIBUTE, the inputs being marked before the pass.
    The memory read is measured per storage as the union of the byte ranges of the tensors read, so that a weight read
    through several views (transposed, expanded over the batch, flattened by a recurrent layer) counts once. A recurrent
    module, run as many operators on the CPU but exported as one node per stacked layer (a two-layer LSTM gives two
    LSTM nodes, each holding both directions), counts as one weighted operator per layer (its forward hooks set
    recurrent_depth).
    """

    def __init__(self) -> None:
        """
        Starts the probe with no operator nor constant recorded.
        """
        super().__init__()
        self.weighted_operators = 0
        self.other_operators = 0
        self.recurrent_depth = 0
        # Byte ranges read in every storage, keyed by the address of the storage, the storages kept alive until the end
        # of the pass so that no address is reused by another tensor
        self.constant_ranges: Dict[int, List[Tuple[int, int]]] = {}
        self.batch_normalization_ranges: Dict[int, List[Tuple[int, int]]] = {}
        self.kept_storages: List[Any] = []

    def __torch_dispatch__(self, func: Any, types: Any, args: Tuple = (), kwargs: Optional[Dict] = None) -> Any:
        """
        Runs one operator of the pass, recording it and the constants it reads when it depends on the inputs.

        Args:
            func (Any): The ATen operator.
            types (Any): The tensor subclasses of the arguments, unused.
            args (Tuple): The positional arguments of the operator.
            kwargs (Optional[Dict]): The keyword arguments of the operator.

        Returns:
            Any: The outputs of the operator, marked when they depend on the inputs.
        """
        kwargs = kwargs or {}
        input_tensors = [value for value in tree_flatten((args, kwargs))[0] if isinstance(value, torch.Tensor)]
        depends_on_input = any(getattr(tensor, INPUT_DEPENDENCE_ATTRIBUTE, False) for tensor in input_tensors)
        operator_name = str(func.overloadpacket)

        # Constants read by an operator of the graph: their byte ranges, those of the batch normalization kept apart
        if depends_on_input:
            for tensor in input_tensors:
                if not getattr(tensor, INPUT_DEPENDENCE_ATTRIBUTE, False):
                    self.record_constant(tensor=tensor, is_batch_normalization=operator_name in
                                         BATCH_NORMALIZATION_OPERATORS)
        outputs = func(*args, **kwargs)

        # Every operator outside a recurrent layer is one node of the graph
        if self.recurrent_depth == 0:
            if operator_name in WEIGHTED_OPERATORS:
                self.weighted_operators += 1
            else:
                self.other_operators += 1

        # Outputs of an operator reading the inputs depend on them in turn
        if depends_on_input:
            for output in tree_flatten(outputs)[0]:
                if isinstance(output, torch.Tensor):
                    setattr(output, INPUT_DEPENDENCE_ATTRIBUTE, True)
        return outputs

    def record_constant(self, tensor: torch.Tensor, is_batch_normalization: bool) -> None:
        """
        Records the byte range a constant tensor reads in its storage: from its first element to its last one, so that
        a view expanded over the batch (stride 0) reads its storage once.

        Args:
            tensor (torch.Tensor): The constant tensor read by an operator depending on the inputs.
            is_batch_normalization (bool): Whether the operator is a batch normalization, whose constants the exports
                fold into the convolution before it.
        """
        storage = tensor.untyped_storage()
        start_byte = tensor.storage_offset() * tensor.element_size()
        last_element = sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
        end_byte = start_byte + (last_element + 1) * tensor.element_size() if tensor.numel() > 0 else start_byte
        self.constant_ranges.setdefault(storage.data_ptr(), []).append((start_byte, end_byte))
        if is_batch_normalization:
            self.batch_normalization_ranges.setdefault(storage.data_ptr(), []).append((start_byte, end_byte))
        self.kept_storages.append(storage)

    def enter_recurrent_layer(self, module: nn.Module, arguments: Tuple) -> None:
        """
        Forward pre-hook of the recurrent modules: the module counts as one weighted operator per stacked layer, the
        ONNX node of each layer, the operators it runs being left out of the count until it returns.

        Args:
            module (nn.Module): The recurrent module (nn.RNNBase), with its 'num_layers'.
            arguments (Tuple): Its positional arguments, unused.
        """
        if self.recurrent_depth == 0:
            self.weighted_operators += module.num_layers
        self.recurrent_depth += 1

    def leave_recurrent_layer(self, module: nn.Module, arguments: Tuple, outputs: Any) -> None:
        """
        Forward hook of the recurrent layers, counting the operators again after the layer.

        Args:
            module (nn.Module): The recurrent layer.
            arguments (Tuple): Its positional arguments, unused.
            outputs (Any): Its outputs, unused.
        """
        self.recurrent_depth -= 1


def measure_range_union(byte_ranges: Dict[int, List[Tuple[int, int]]]) -> int:
    """
    Sums, over the storages, the bytes covered by the union of the byte ranges read in each, so that overlapping reads
    of the same memory count once.

    Args:
        byte_ranges (Dict[int, List[Tuple[int, int]]]): The (start, end) byte ranges read in every storage.

    Returns:
        int: The bytes covered.
    """
    covered_bytes = 0
    for storage_ranges in byte_ranges.values():
        # Merge the sorted ranges, a range starting inside the current one extending it
        current_start, current_end = None, None
        for start_byte, end_byte in sorted(storage_ranges):
            if current_end is None or start_byte > current_end:
                covered_bytes += 0 if current_end is None else current_end - current_start
                current_start, current_end = start_byte, end_byte
            else:
                current_end = max(current_end, end_byte)
        covered_bytes += 0 if current_end is None else current_end - current_start
    return covered_bytes


def probe_inference_graph(inference_model: nn.Module, inputs: Dict[str, torch.Tensor]) -> GraphProbe:
    """
    Runs one forward pass of an inference model under ExportGraphProbe and gives what its exports hold: the bytes of
    the constants its operators read (the weights the exports write), of which those of the batch normalization, and
    its numbers of operators. Called by estimate_export_sizes on a batch of one sample, the constants not depending on
    the batch size.

    Args:
        inference_model (nn.Module): The inference model, in evaluation mode.
        inputs (Dict[str, torch.Tensor]): The batch of every input, keyed by forward argument.

    Returns:
        GraphProbe: The constants and operators of the pass, with the addresses of the storages read.
    """
    probe = ExportGraphProbe()
    # Inputs marked as the start of the graph, on copies so that the batch of the caller stays unmarked
    marked_inputs = {name: batch.clone() for name, batch in inputs.items()}
    for batch in marked_inputs.values():
        setattr(batch, INPUT_DEPENDENCE_ATTRIBUTE, True)

    # Recurrent layers counted as one operator each through their forward hooks, removed after the pass
    hook_handles = []
    for module in inference_model.modules():
        if isinstance(module, nn.RNNBase):
            hook_handles.append(module.register_forward_pre_hook(hook=probe.enter_recurrent_layer))
            hook_handles.append(module.register_forward_hook(hook=probe.leave_recurrent_layer))
    try:
        with torch.no_grad(), probe:
            inference_model(**marked_inputs)
    finally:
        for hook_handle in hook_handles:
            hook_handle.remove()
    return {"constant_bytes": measure_range_union(byte_ranges=probe.constant_ranges),
            "batch_normalization_bytes": measure_range_union(byte_ranges=probe.batch_normalization_ranges),
            "weighted_operators": probe.weighted_operators, "other_operators": probe.other_operators,
            "constant_storages": set(probe.constant_ranges)}


def estimate_state_dictionary_bytes(state_dictionary: Dict[str, torch.Tensor], bytes_per_floating_value: Optional[int]
                                    ) -> int:
    """
    Estimates the size of the file torch.save writes for a state dictionary, from the bytes of its tensors and of its
    keys, without writing it: the PyTorch weights of the panel, and the model, optimizer and averaged states of a
    checkpoint.

    Args:
        state_dictionary (Dict[str, torch.Tensor]): The tensors, keyed by name.
        bytes_per_floating_value (Optional[int]): The bytes of every floating value once converted (2 for FP16), or
            None to keep the type of every tensor.

    Returns:
        int: The estimated size in bytes.
    """
    estimated_bytes = STATE_DICTIONARY_FIXED_BYTES
    for key, tensor in state_dictionary.items():
        value_bytes = (bytes_per_floating_value if bytes_per_floating_value is not None and tensor.is_floating_point()
                       else tensor.element_size())
        estimated_bytes += tensor.numel() * value_bytes + STATE_DICTIONARY_BYTES_PER_TENSOR + len(key)
    return estimated_bytes


def count_half_precision_saved_bytes(inference_model: nn.Module) -> int:
    """
    Counts the bytes the FP16 TorchScript export (torchscript_precision 'FP16' of deployment/export_model.py) saves on
    the FP32 file of an inference model, so that the panel gives the size of the FP16 file without converting the model
    nor tracing it again: convert_to_half_precision writes the floating tensors of the HALF_PRECISION_MODULE_TYPES
    modules (convolution, linear and attention weights) in 2 bytes instead of their own, every other tensor keeping its
    type. A tensor shared by several modules is stored once and counted once.

    Args:
        inference_model (nn.Module): The FP32 inference model of the export.

    Returns:
        int: The bytes saved on the FP32 TorchScript file.
    """
    converted_tensors = {}
    for module in inference_model.modules():
        if isinstance(module, HALF_PRECISION_MODULE_TYPES):
            for tensor in [*module.parameters(), *module.buffers()]:
                if tensor.is_floating_point():
                    converted_tensors[id(tensor)] = tensor
    return sum(tensor.numel() * (tensor.element_size() - 2) for tensor in converted_tensors.values())


def count_optimizer_moments(optimizer: torch.optim.Optimizer) -> int:
    """
    Gives the number of tensors the optimizer keeps per parameter, the size of a parameter each: the two moment
    averages of Adam and AdamW (three with AMSGrad), the momentum buffer of SGD with momentum.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer of the training (BaseTrainer.build_optimizer).

    Returns:
        int: The number of tensors per parameter.

    Raises:
        ValueError: If the optimizer is not one the trainer builds.
    """
    if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        return 3 if any(parameter_group["amsgrad"] for parameter_group in optimizer.param_groups) else 2
    if isinstance(optimizer, torch.optim.SGD):
        return 1 if any(parameter_group["momentum"] for parameter_group in optimizer.param_groups) else 0
    raise ValueError(f"Expected an Adam, AdamW or SGD optimizer to estimate its state, got {type(optimizer).__name__}.")


def estimate_checkpoint_bytes(model: nn.Module, optimizer: torch.optim.Optimizer, keeps_moving_average: bool) -> int:
    """
    Estimates the size of the checkpoint the trainer saves at every saved iteration (save_checkpoint): the weights of
    the model, the state of the optimizer for every trained parameter, the averaged weights of a run keeping a moving
    average of its weights, and the metadata of the trainer.

    Args:
        model (nn.Module): The trained model.
        optimizer (torch.optim.Optimizer): The optimizer of the training, whose state is still empty at launch.
        keeps_moving_average (bool): Whether the run keeps a moving average of its weights.

    Returns:
        int: The estimated size in bytes.
    """
    model_bytes = estimate_state_dictionary_bytes(state_dictionary=model.state_dict(), bytes_per_floating_value=None)

    # Every trained parameter gets its moment tensors once it has received a gradient
    number_moments = count_optimizer_moments(optimizer=optimizer)
    optimizer_bytes = sum(number_moments * parameter.numel() * parameter.element_size() + OPTIMIZER_BYTES_PER_PARAMETER
                          for parameter_group in optimizer.param_groups for parameter in parameter_group["params"]
                          if parameter.requires_grad)
    return model_bytes * (2 if keeps_moving_average else 1) + optimizer_bytes + CHECKPOINT_METADATA_BYTES


def measure_torchscript_bytes(inference_model: nn.Module, example_inputs: Dict[str, torch.Tensor]) -> int:
    """
    Measures the TorchScript file of an inference model by exporting it as deployment/export_model.py does
    (export_torchscript, the trace of the example batch) to a temporary 'model.pt', the name of the file being part of
    the names of its records, the file being removed afterwards: no formula holds its code within 5 % for every task.

    Args:
        inference_model (nn.Module): The inference model, in evaluation mode.
        example_inputs (Dict[str, torch.Tensor]): The traced batch of every input (create_random_inputs).

    Returns:
        int: The size of 'model.pt' in bytes.
    """
    with tempfile.TemporaryDirectory(prefix="export_size_") as temporary_folder:
        torchscript_path = Path(temporary_folder) / "model.pt"
        export_torchscript(inference_model=inference_model, example_inputs=example_inputs, output_path=torchscript_path)
        return torchscript_path.stat().st_size


def estimate_onnx_bytes(probe: GraphProbe) -> int:
    """
    Estimates the ONNX files of an export (export_onnx): 'model.onnx.data', the constants read by the forward pass with
    every batch normalization folded into the convolution before it (of its four vectors only the bias stays), and
    'model.onnx', its operators.

    Args:
        probe (GraphProbe): The forward pass of the inference model (probe_inference_graph).

    Returns:
        int: The estimated size of both files in bytes.
    """
    folded_constant_bytes = probe["constant_bytes"] - probe["batch_normalization_bytes"] * 3 // 4
    number_operators = probe["weighted_operators"] + probe["other_operators"]
    return folded_constant_bytes + number_operators * ONNX_GRAPH_BYTES_PER_OPERATOR


def estimate_tensorrt_bytes(probe: GraphProbe, bytes_per_value: int) -> Tuple[int, int]:
    """
    Estimates the range of the size of the TensorRT engine Triton builds from the ONNX file of an export: its weights,
    folded as in the ONNX file, in the precision of the engine, and the overhead of its layers, larger for the
    convolutions, matrix products and recurrent layers than for the other operators, within TENSORRT_OVERHEAD_RANGE,
    which is wide since the overhead follows the number of operators only loosely.

    Args:
        probe (GraphProbe): The forward pass of the inference model (probe_inference_graph).
        bytes_per_value (int): The bytes of every weight in the engine, 4 in FP32 and 2 in FP16.

    Returns:
        Tuple[int, int]: The lowest and highest estimated sizes in bytes.
    """
    weight_bytes = (probe["constant_bytes"] - probe["batch_normalization_bytes"] * 3 // 4) * bytes_per_value // 4
    overhead_bytes = (probe["weighted_operators"] * TENSORRT_BYTES_PER_WEIGHTED_OPERATOR
                      + probe["other_operators"] * TENSORRT_BYTES_PER_OTHER_OPERATOR)
    return (weight_bytes + int(overhead_bytes * TENSORRT_OVERHEAD_RANGE[0]),
            weight_bytes + int(overhead_bytes * TENSORRT_OVERHEAD_RANGE[1]))


def format_margin(margin: float) -> str:
    """
    Writes the margin of an estimated size for the Method column of the panel.

    Args:
        margin (float): The largest relative error of the estimate, e.g. 0.03.

    Returns:
        str: The margin, e.g. '± 3 %' or '± 0.1 %'.
    """
    return f"± {margin * 100:g} %"


def estimate_export_sizes(task: str, model: nn.Module, optimizer: torch.optim.Optimizer, keeps_moving_average: bool
                          ) -> ExportSizes:
    """
    Gives the size of every file the model of a training is turned into, for the Export Sizes panel printed at launch:
    the inference model of the task (build_inference_model) runs once on one random sample under ExportGraphProbe and
    is traced to a temporary TorchScript file on the example batch of the exports, on the device of the model, then
    every module of the model gets back its training mode. The FP16 TorchScript file is the FP32 trace less the bytes
    its FP16 weights save (count_half_precision_saved_bytes), estimated on the CPU as well, where the FP16 export
    itself cannot run. A failed trace or probe leaves its rows without size, its error being listed under the panel, so
    that a model the exports cannot trace still trains.

    Args:
        task (str): The task of the training, one of the tasks Triton serves (INFERENCE_MODEL_CLASSES).
        model (nn.Module): The model of the training.
        optimizer (torch.optim.Optimizer): The optimizer of the training.
        keeps_moving_average (bool): Whether the run keeps a moving average of its weights, saved in its checkpoints.

    Returns:
        ExportSizes: The rows of the panel, the exported and total parameters, the time taken, the errors and the
            forward pass of the probe.
    """
    start_time = time.perf_counter()
    errors = []
    probe, torchscript_bytes, torchscript_half_bytes = None, None, None
    # Training mode of every module, restored after the inference model has run in evaluation mode, so that the frozen
    # batch normalization layers of a transfer stay as the trainer set them
    training_modes = {module: module.training for module in model.modules()}
    try:
        inference_model = build_inference_model(task=task, model=model)
        specification = get_export_specification(task=task, inference_model=inference_model)
        device = next(model.parameters()).device
        # Random inputs drawn from a generator of their own, so that the random stream of the training is unchanged
        example_inputs = create_random_inputs(specification=specification, shape_name="example_shape",
                                              maximum_batch_size=EXPORT_MAXIMUM_BATCH_SIZE, device=device,
                                              generator=torch.Generator(device=device).manual_seed(0))
        state_dictionary = inference_model.state_dict()
        half_precision_saved_bytes = count_half_precision_saved_bytes(inference_model=inference_model)

        # One forward pass on one sample for the ONNX and TensorRT estimates
        try:
            probe = probe_inference_graph(inference_model=inference_model,
                                          inputs={name: batch[:1] for name, batch in example_inputs.items()})
        except Exception as error:
            errors.append(f"Forward pass of the probe: {type(error).__name__}: {error}")

        # TorchScript file traced as the export does, the warnings of the tracer hidden for this trace only: the export
        # itself prints them and checks the traced file against the eager model
        try:
            with warnings.catch_warnings():
                hide_known_export_warnings()
                warnings.filterwarnings(action="ignore", category=torch.jit.TracerWarning)
                torchscript_bytes = measure_torchscript_bytes(inference_model=inference_model,
                                                              example_inputs=example_inputs)
            # FP16 file: the same trace with its convolution, linear and attention weights in 2 bytes
            torchscript_half_bytes = torchscript_bytes - half_precision_saved_bytes
        except Exception as error:
            errors.append(f"TorchScript trace: {type(error).__name__}: {error}")
    finally:
        for module, training_mode in training_modes.items():
            module.training = training_mode

    # Rows of the panel, the TensorRT ranges printed as such and every other estimate with its margin
    float_bytes = estimate_state_dictionary_bytes(state_dictionary=state_dictionary, bytes_per_floating_value=None)
    half_bytes = estimate_state_dictionary_bytes(state_dictionary=state_dictionary, bytes_per_floating_value=2)
    onnx_bytes = estimate_onnx_bytes(probe=probe) if probe is not None else None
    tensorrt_float_bytes = estimate_tensorrt_bytes(probe=probe, bytes_per_value=4) if probe is not None else (None,
                                                                                                             None)
    tensorrt_half_bytes = estimate_tensorrt_bytes(probe=probe, bytes_per_value=2) if probe is not None else (None,
                                                                                                            None)
    checkpoint_bytes = estimate_checkpoint_bytes(model=model, optimizer=optimizer,
                                                 keeps_moving_average=keeps_moving_average)
    rows = [{"label": "PyTorch FP32 Weights", "lowest_bytes": float_bytes, "highest_bytes": float_bytes,
             "method": format_margin(margin=STATE_DICTIONARY_MARGIN)},
            {"label": "PyTorch FP16 Weights", "lowest_bytes": half_bytes, "highest_bytes": half_bytes,
             "method": format_margin(margin=STATE_DICTIONARY_MARGIN)},
            {"label": "TorchScript FP32", "lowest_bytes": torchscript_bytes, "highest_bytes": torchscript_bytes,
             "method": "traced"},
            {"label": "TorchScript FP16", "lowest_bytes": torchscript_half_bytes,
             "highest_bytes": torchscript_half_bytes,
             "method": format_margin(margin=TORCHSCRIPT_HALF_PRECISION_MARGIN)},
            {"label": "ONNX", "lowest_bytes": onnx_bytes, "highest_bytes": onnx_bytes,
             "method": format_margin(margin=ONNX_MARGIN)},
            {"label": "TensorRT FP32 Engine", "lowest_bytes": tensorrt_float_bytes[0],
             "highest_bytes": tensorrt_float_bytes[1], "method": "range"},
            {"label": "TensorRT FP16 Engine", "lowest_bytes": tensorrt_half_bytes[0],
             "highest_bytes": tensorrt_half_bytes[1], "method": "range"},
            {"label": "Training Checkpoint", "lowest_bytes": checkpoint_bytes, "highest_bytes": checkpoint_bytes,
             "method": format_margin(margin=CHECKPOINT_MARGIN)}]

    # Parameters the forward pass reads, those of the training-only parts (the projection head of a self-supervised
    # encoder) staying out
    exported_parameters = (sum(parameter.numel() for parameter in model.parameters()
                               if parameter.untyped_storage().data_ptr() in probe["constant_storages"])
                           if probe is not None else 0)
    return {"exported_parameters": exported_parameters,
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()), "rows": rows,
            "seconds": time.perf_counter() - start_time, "errors": errors, "probe": probe}


def format_size(row: ExportSizeRow) -> str:
    """
    Writes the size of a row of the panel in megabytes, a range as 'lowest - highest', a missing size as 'failed'.

    Args:
        row (ExportSizeRow): The row.

    Returns:
        str: The size, e.g. '18.59 MB' or '18.94 - 19.24 MB'.
    """
    if row["lowest_bytes"] is None:
        return "failed"
    if row["lowest_bytes"] == row["highest_bytes"]:
        return f"{row['lowest_bytes'] / BYTES_PER_MEGABYTE:.2f} MB"
    return f"{row['lowest_bytes'] / BYTES_PER_MEGABYTE:.2f} - {row['highest_bytes'] / BYTES_PER_MEGABYTE:.2f} MB"


def print_export_size_summary(export_sizes: ExportSizes) -> None:
    """
    Prints the Export Sizes panel after the Model Size panel at the launch of a training: the parameters the exports
    keep, then, indented under them, the size of every format with how it was obtained, the time taken in the subtitle
    and the errors of a failed trace or probe below the table.

    Args:
        export_sizes (ExportSizes): The sizes, from estimate_export_sizes.
    """
    console = Console(width=60)
    table = Table(box=None, show_header=True, header_style="bold", pad_edge=False, padding=(0, 2, 0, 0))
    table.add_column(header="Format")
    table.add_column(header="Size", justify="right")
    table.add_column(header="Method")
    for row in export_sizes["rows"]:
        table.add_row(row["label"], format_size(row=row), row["method"],
                      style="red" if row["lowest_bytes"] is None else None)
    parameters_line = Text(text=f"Exported Parameters : {export_sizes['exported_parameters']:,} of "
                                f"{export_sizes['total_parameters']:,}\n")
    error_lines = [Text(text=f"\n{error}", style="red") for error in export_sizes["errors"]]
    # Table indented under the parameters line, so that the formats do not run along the border of the panel
    indented_table = Padding(renderable=table, pad=(0, 0, 0, TABLE_INDENT))
    panel = Panel(renderable=Group(parameters_line, indented_table, *error_lines), title="[bold]Export Sizes[/bold]",
                  subtitle=f"{export_sizes['seconds']:.1f} seconds", border_style="cyan")
    console.print(panel)
