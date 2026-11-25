"""
Tests of vision_training/deployment/export_size_estimates.py, the Export Sizes panel printed at the launch of every
training of a task Triton serves: the probe of the forward pass counts once the memory of the weights it reads through
several views, counts the constants it computes and leaves out the weights it never reads, counts a recurrent layer as
one operator whatever the number of frames, and a stacked recurrent module as one operator per layer; the state dictionary and checkpoint formulas stay within their margins of
the files torch.save and save_checkpoint write; the FP16 TorchScript estimate counts once the weights the FP16 export
converts and none it keeps in FP32; the estimates of a classifier give the size of its TorchScript file (up to the
numbering of its classes by TorchScript), its FP16 file as that size less the converted weights, and the size of its
ONNX files within their margin, and leave the training modes and the random stream of the training as they were; on
the GPU, the FP16 estimate of a classifier follows its real FP16 export; the model of every task Triton serves is
probed and traced without error; a failed trace is printed in the panel without stopping the estimates.
"""

import pytest
import torch

from pathlib import Path
from torch import nn
from typing import Dict, Optional

from tests.deployment.test_export_specifications import build_test_model
from tests.testing_utilities import build_test_model as build_test_detector
from vision_training.deployment import export_size_estimates
from vision_training.deployment.export_formats import (convert_to_half_precision, export_onnx, export_torchscript,
                                                       hide_known_export_warnings)
from vision_training.deployment.export_size_estimates import (CHECKPOINT_MARGIN, EXPORT_MAXIMUM_BATCH_SIZE, ONNX_MARGIN,
                                                              STATE_DICTIONARY_MARGIN, count_half_precision_saved_bytes,
                                                              count_optimizer_moments, estimate_checkpoint_bytes,
                                                              estimate_export_sizes,
                                                              estimate_state_dictionary_bytes, measure_range_union,
                                                              print_export_size_summary, probe_inference_graph)
from vision_training.deployment.export_specifications import (build_inference_model, create_random_inputs,
                                                              get_export_specification)
from vision_training.trainers.checkpointing import save_checkpoint
from vision_training.trainers.weight_averaging import ExponentialMovingAverage

# Bytes of a float32 value
FLOAT_BYTES = 4


class ProbedModule(nn.Module):
    """
    Module read by the probe tests: a convolution followed by a batch normalization, a linear layer called twice
    (each call reading its transposed weight), a linear layer never called, a constant computed during the forward
    pass from Python numbers and a buffer expanded over the batch.
    """

    def __init__(self) -> None:
        """
        Builds the layers and the buffer of the module.
        """
        super().__init__()
        self.convolution = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, bias=False)
        self.normalization = nn.BatchNorm2d(num_features=8)
        self.linear = nn.Linear(in_features=8, out_features=8)
        self.unused_linear = nn.Linear(in_features=8, out_features=100)
        self.register_buffer(name="offsets", tensor=torch.arange(8, dtype=torch.float32))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Computes 8 features per image.

        Args:
            images (torch.Tensor): Images of shape [B, 3, H, W].

        Returns:
            torch.Tensor: The features [B, 8].
        """
        features = self.normalization(self.convolution(images)).mean(dim=(2, 3))
        features = self.linear(self.linear(features))
        # A constant computed from Python numbers, then a buffer read once whatever the batch size
        scales = torch.arange(1, 9, dtype=torch.float32, device=images.device)
        return features * scales + self.offsets.expand(images.shape[0], -1)


class RecurrentModule(nn.Module):
    """
    Module of the recurrent probe tests: a bidirectional GRU of one or more stacked layers over sequences of features,
    summed over the frames.
    """

    def __init__(self, number_layers: int = 1) -> None:
        """
        Builds the recurrent layers.

        Args:
            number_layers (int): The stacked layers of the GRU, each exported as one ONNX node.
        """
        super().__init__()
        self.recurrent = nn.GRU(input_size=4, hidden_size=6, num_layers=number_layers, batch_first=True,
                                bidirectional=True)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Encodes every sequence.

        Args:
            sequences (torch.Tensor): Sequences of shape [B, T, 4].

        Returns:
            torch.Tensor: The encodings [B, 12].
        """
        outputs, _ = self.recurrent(sequences)
        return outputs.sum(dim=1)


class HalfPrecisionCountedModule(nn.Module):
    """
    Module of the FP16 TorchScript estimate test, never run: a convolution, a batch normalization, an attention layer
    (whose output projection is a linear layer of its own), two linear layers sharing their weight, an LSTM and a
    buffer, of which the FP16 export converts the convolution, attention and linear weights only.
    """

    def __init__(self) -> None:
        """
        Builds the layers, the shared weight and the buffer of the module.
        """
        super().__init__()
        self.convolution = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.normalization = nn.BatchNorm2d(num_features=8)
        self.attention = nn.MultiheadAttention(embed_dim=8, num_heads=2)
        self.first_linear = nn.Linear(in_features=8, out_features=8)
        self.second_linear = nn.Linear(in_features=8, out_features=8)
        # Weight shared by both linear layers, stored once in the TorchScript file
        self.second_linear.weight = self.first_linear.weight
        self.recurrent = nn.LSTM(input_size=8, hidden_size=8)
        self.register_buffer(name="offsets", tensor=torch.zeros(8))


def build_sized_model() -> nn.Sequential:
    """
    Builds a model of about 1.1 million parameters, the size order of the smallest models of the repository, with
    batch normalization buffers, for the checkpoint test.

    Returns:
        nn.Sequential: The model.
    """
    torch.manual_seed(seed=0)
    return nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3), nn.BatchNorm2d(num_features=64),
                         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3), nn.BatchNorm2d(num_features=64),
                         nn.AdaptiveAvgPool2d(output_size=4), nn.Flatten(),
                         nn.Linear(in_features=1024, out_features=1024))


def raise_trace_error(inference_model: nn.Module, example_inputs: Dict[str, torch.Tensor]) -> int:
    """
    Stands for a TorchScript trace that fails, in place of measure_torchscript_bytes in the panel test.

    Args:
        inference_model (nn.Module): The inference model, unused.
        example_inputs (Dict[str, torch.Tensor]): The traced batch, unused.

    Raises:
        RuntimeError: Always.
    """
    raise RuntimeError("the trace failed")


def test_probe_counts_the_memory_read_once() -> None:
    """
    Verifies that the probe counts the weights of the convolution, the batch normalization and the linear layer once
    although the linear layer is called twice through its transposed weight, the constant computed during the forward
    pass and the expanded buffer at the size of its storage, leaves out the linear layer never called, keeps apart the
    four vectors of the batch normalization, and counts the convolution and the two matrix products as weighted
    operators.
    """
    module = ProbedModule().eval()
    probe = probe_inference_graph(inference_model=module, inputs={"images": torch.rand(4, 3, 10, 10)})

    # Convolution 8x3x3x3, batch normalization 4 vectors of 8, linear 8x8 + 8, the computed scales and the offsets
    expected_values = 8 * 3 * 3 * 3 + 4 * 8 + 8 * 8 + 8 + 8 + 8
    assert probe["constant_bytes"] == expected_values * FLOAT_BYTES
    assert probe["batch_normalization_bytes"] == 4 * 8 * FLOAT_BYTES
    assert probe["weighted_operators"] == 3
    assert module.convolution.weight.untyped_storage().data_ptr() in probe["constant_storages"]
    assert module.unused_linear.weight.untyped_storage().data_ptr() not in probe["constant_storages"]


def test_probe_counts_a_recurrent_layer_as_one_operator() -> None:
    """
    Verifies that a GRU counts as one weighted operator whatever the number of frames, although it runs operators
    frame by frame on the CPU, and that its weights are counted once whatever the number of frames, the only other
    constants being the initial hidden state of one sample and its first products with the weights (computed from
    the batch size, 120 bytes per direction here).
    """
    module = RecurrentModule().eval()
    probes = [probe_inference_graph(inference_model=module, inputs={"sequences": torch.rand(1, frames, 4)})
              for frames in [3, 9]]
    weight_bytes = sum(parameter.numel() for parameter in module.parameters()) * FLOAT_BYTES
    for probe in probes:
        assert probe["weighted_operators"] == 1
        assert all(parameter.untyped_storage().data_ptr() in probe["constant_storages"]
                   for parameter in module.parameters())
        assert weight_bytes < probe["constant_bytes"] < 1.2 * weight_bytes
    assert probes[0]["constant_bytes"] == probes[1]["constant_bytes"]
    assert probes[0]["other_operators"] == probes[1]["other_operators"]


def test_probe_counts_one_operator_per_stacked_recurrent_layer() -> None:
    """
    Verifies that a GRU of three stacked layers counts as three weighted operators, the ONNX export writing one GRU
    node per layer (two LSTM nodes in model.onnx of the two-layer sequence labelling tutorial run), with the weights of
    every layer read.
    """
    module = RecurrentModule(number_layers=3).eval()
    probe = probe_inference_graph(inference_model=module, inputs={"sequences": torch.rand(1, 5, 4)})
    assert probe["weighted_operators"] == 3
    assert all(parameter.untyped_storage().data_ptr() in probe["constant_storages"]
               for parameter in module.parameters())


def test_half_precision_estimate_counts_the_converted_weights_once() -> None:
    """
    Verifies that the bytes the FP16 TorchScript file saves are those of the convolution, attention and linear weights
    the FP16 export converts (convert_to_half_precision), 2 bytes per FP32 value, the output projection of the
    attention and the weight shared by two linear layers counted once, the batch normalization, the LSTM and the
    buffer staying FP32.
    """
    # Values converted: convolution 8 x 3 x 3 x 3 + 8, attention 3 x 8 x 8 + 3 x 8 and its output projection 8 x 8 + 8,
    # the shared linear weight 8 x 8 and both biases 8
    converted_values = (8 * 3 * 3 * 3 + 8) + (3 * 8 * 8 + 3 * 8 + 8 * 8 + 8) + (8 * 8 + 8 + 8)
    assert count_half_precision_saved_bytes(inference_model=HalfPrecisionCountedModule()) == converted_values * 2


def test_range_union_counts_overlapping_reads_once() -> None:
    """
    Verifies that measure_range_union merges the overlapping and nested ranges of a storage and adds the storages.
    """
    assert measure_range_union(byte_ranges={1: [(5, 20), (0, 10), (30, 40), (32, 35)], 2: [(8, 12)]}) == 34
    assert measure_range_union(byte_ranges={}) == 0


@pytest.mark.parametrize(argnames="bytes_per_floating_value", argvalues=[None, 2])
def test_state_dictionary_estimate_is_within_its_margin(tmp_path: Path, bytes_per_floating_value: int) -> None:
    """
    Verifies that the estimated size of a state dictionary of large and small tensors (convolution weights, batch
    normalization vectors and an integer counter), in FP32 and with its floating tensors in FP16, is within
    STATE_DICTIONARY_MARGIN of the file torch.save writes.

    Args:
        tmp_path (Path): The pytest temporary directory.
        bytes_per_floating_value (int): None for the FP32 weights, 2 for the FP16 weights.
    """
    torch.manual_seed(seed=0)
    state_dictionary = {}
    for block_index in range(12):
        state_dictionary[f"backbone.convolution_blocks.{block_index}.convolution.weight"] = torch.randn(64, 64, 3, 3)
        for vector_name in ["weight", "bias", "running_mean", "running_var"]:
            state_dictionary[f"backbone.convolution_blocks.{block_index}.normalization.{vector_name}"] = torch.randn(64)
        state_dictionary[f"backbone.convolution_blocks.{block_index}.normalization.num_batches_tracked"] = torch.tensor(7)
    saved_dictionary = {key: tensor.half() if bytes_per_floating_value == 2 and tensor.is_floating_point() else tensor
                        for key, tensor in state_dictionary.items()}
    torch.save(obj=saved_dictionary, f=tmp_path / "model.pt")
    estimated_bytes = estimate_state_dictionary_bytes(state_dictionary=state_dictionary,
                                                      bytes_per_floating_value=bytes_per_floating_value)
    assert abs(estimated_bytes / (tmp_path / "model.pt").stat().st_size - 1.0) < STATE_DICTIONARY_MARGIN


@pytest.mark.parametrize(argnames="keeps_moving_average", argvalues=[False, True])
def test_checkpoint_estimate_is_within_its_margin(tmp_path: Path, keeps_moving_average: bool) -> None:
    """
    Verifies that the estimated checkpoint of a model trained with Adam, with and without the moving average of its
    weights, is within CHECKPOINT_MARGIN of the file save_checkpoint writes after a first step, the estimate being
    made before the step as at launch, when the optimizer has no state yet.

    Args:
        tmp_path (Path): The pytest temporary directory.
        keeps_moving_average (bool): Whether the checkpoint holds the averaged weights.
    """
    model = build_sized_model()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    estimated_bytes = estimate_checkpoint_bytes(model=model, optimizer=optimizer,
                                                keeps_moving_average=keeps_moving_average)

    # One step creates the moment tensors of every parameter, saved with the metadata of a run
    model(torch.rand(2, 3, 12, 12)).sum().backward()
    optimizer.step()
    moving_average = ExponentialMovingAverage(model=model, decay=0.99) if keeps_moving_average else None
    checkpoint_folder = save_checkpoint(
        weights_directory=tmp_path / "Weights", iteration=10, model=model, optimizer=optimizer,
        metadata={"task": "classification", "run_name": "CLS_test", "label_dictionary": {"cat": 0, "dog": 1}},
        exponential_moving_average_state=moving_average.state_dict() if keeps_moving_average else None)
    checkpoint_path = next(checkpoint_folder.glob("*.pt"))
    assert abs(estimated_bytes / checkpoint_path.stat().st_size - 1.0) < CHECKPOINT_MARGIN


def test_optimizer_moments_follow_the_optimizer() -> None:
    """
    Verifies the number of tensors kept per parameter by the optimizers the trainer builds, and that another optimizer
    is refused rather than estimated wrongly.
    """
    parameters = [nn.Parameter(torch.zeros(3))]
    assert count_optimizer_moments(optimizer=torch.optim.Adam(params=parameters)) == 2
    assert count_optimizer_moments(optimizer=torch.optim.AdamW(params=parameters)) == 2
    assert count_optimizer_moments(optimizer=torch.optim.Adam(params=parameters, amsgrad=True)) == 3
    assert count_optimizer_moments(optimizer=torch.optim.SGD(params=parameters, lr=0.1, momentum=0.9)) == 1
    assert count_optimizer_moments(optimizer=torch.optim.SGD(params=parameters, lr=0.1)) == 0
    with pytest.raises(ValueError, match="RMSprop"):
        count_optimizer_moments(optimizer=torch.optim.RMSprop(params=parameters))


def test_classifier_estimates_match_its_exports(tmp_path: Path) -> None:
    """
    Verifies on a small classifier in training mode, one of its batch normalization layers frozen in evaluation mode
    as a transferred component is, that the estimates give the size of the TorchScript file of the export (up to the
    class names TorchScript numbers with a counter of the process, a few kilobytes for this small model whose code
    outweighs its weights), the FP16 file as that size less the bytes of the converted weights, and the size of its
    ONNX files within ONNX_MARGIN, count every parameter as exported, and leave the training mode of every module and
    the global random stream as they were.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    hide_known_export_warnings()
    model = build_test_model(task="classification", temporal_type=None, fusion_type=None,
                             global_pooling="average").train()
    frozen_normalization = next(module for module in model.modules() if isinstance(module, nn.BatchNorm2d))
    frozen_normalization.eval()
    training_modes = {name: module.training for name, module in model.named_modules()}
    random_state = torch.random.get_rng_state()

    export_sizes = estimate_export_sizes(task="classification", model=model,
                                         optimizer=torch.optim.Adam(params=model.parameters()),
                                         keeps_moving_average=False)

    # The model and the random stream are left as they were, and nothing failed
    assert {name: module.training for name, module in model.named_modules()} == training_modes
    assert torch.equal(torch.random.get_rng_state(), random_state)
    assert export_sizes["errors"] == []
    assert export_sizes["exported_parameters"] == export_sizes["total_parameters"]
    rows = {row["label"]: row for row in export_sizes["rows"]}

    # Files of the export of the same inference model and example batch
    inference_model = build_inference_model(task="classification", model=model)
    specification = get_export_specification(task="classification", inference_model=inference_model)
    example_inputs = create_random_inputs(specification=specification, shape_name="example_shape",
                                          maximum_batch_size=EXPORT_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                          generator=torch.Generator().manual_seed(0))
    export_torchscript(inference_model=inference_model, example_inputs=example_inputs, output_path=tmp_path / "model.pt")
    export_onnx(inference_model=inference_model, example_inputs=example_inputs, output_path=tmp_path / "model.onnx",
                specification=specification, maximum_batch_size=EXPORT_MAXIMUM_BATCH_SIZE)
    onnx_bytes = (tmp_path / "model.onnx").stat().st_size + (tmp_path / "model.onnx.data").stat().st_size
    assert abs(rows["TorchScript FP32"]["lowest_bytes"] / (tmp_path / "model.pt").stat().st_size - 1.0) < 0.03
    saved_bytes = count_half_precision_saved_bytes(inference_model=inference_model)
    assert rows["TorchScript FP16"]["lowest_bytes"] == rows["TorchScript FP32"]["lowest_bytes"] - saved_bytes
    assert abs(rows["ONNX"]["lowest_bytes"] / onnx_bytes - 1.0) < ONNX_MARGIN

    # The TensorRT engines are ranges around weights in their precision, the FP16 one below the FP32 one
    float_engine, half_engine = rows["TensorRT FP32 Engine"], rows["TensorRT FP16 Engine"]
    assert float_engine["lowest_bytes"] < float_engine["highest_bytes"]
    assert half_engine["highest_bytes"] < float_engine["highest_bytes"]


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="The FP16 copy of the network runs on the GPU.")
def test_half_precision_torchscript_estimate_follows_the_fp16_export(tmp_path: Path) -> None:
    """
    Verifies on the GPU that the FP16 TorchScript row of a small classifier follows the file of the FP16 export
    (convert_to_half_precision traced by export_torchscript into 'model_fp16.pt'): within 6 % for this model of 0.2
    MB, whose code and records, which the estimate takes from the FP32 trace without the casts of autocast, outweigh
    its weights (the files of the tutorial runs, from 0.6 to 53 MB, lie within TORCHSCRIPT_HALF_PRECISION_MARGIN in
    scripts/deployment/check_export_size_estimates.py), against about 23 % for the FP32 trace itself.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    hide_known_export_warnings()
    device = torch.device("cuda")
    model = build_test_model(task="classification", temporal_type=None, fusion_type=None,
                             global_pooling="average").to(device)
    export_sizes = estimate_export_sizes(task="classification", model=model.train(),
                                         optimizer=torch.optim.Adam(params=model.parameters()),
                                         keeps_moving_average=False)
    rows = {row["label"]: row for row in export_sizes["rows"]}

    # FP16 file of the export, traced on its example batch
    inference_model = build_inference_model(task="classification", model=model)
    specification = get_export_specification(task="classification", inference_model=inference_model)
    example_inputs = create_random_inputs(specification=specification, shape_name="example_shape",
                                          maximum_batch_size=EXPORT_MAXIMUM_BATCH_SIZE, device=device,
                                          generator=torch.Generator(device=device).manual_seed(0))
    half_precision_model = convert_to_half_precision(inference_model=inference_model, input_names=["images"])
    export_torchscript(inference_model=half_precision_model, example_inputs=example_inputs,
                       output_path=tmp_path / "model_fp16.pt")
    half_precision_bytes = (tmp_path / "model_fp16.pt").stat().st_size
    assert abs(rows["TorchScript FP16"]["lowest_bytes"] / half_precision_bytes - 1.0) < 0.06
    assert rows["TorchScript FP32"]["lowest_bytes"] / half_precision_bytes - 1.0 > 0.15


@pytest.mark.parametrize(argnames=("task", "temporal_type", "fusion_type"),
                         argvalues=[("classification", None, None), ("semantic_segmentation", None, None),
                                    ("self_supervised", None, None), ("keypoint_detection", None, None),
                                    ("sequence_classification", "gru", None), ("sequence_labelling", "gru", None),
                                    ("frame_selection", "gru", None), ("sequence_comparison", "gru", "difference"),
                                    ("object_detection", None, None), ("instance_segmentation", None, None)])
def test_every_task_is_estimated_without_error(task: str, temporal_type: Optional[str],
                                               fusion_type: Optional[str]) -> None:
    """
    Verifies that the probe and the trace of the panel run on the inference model of every task Triton serves, so
    that every row gets a size: the panel catches their errors to let the training go on, so a task whose model the
    probe or the trace cannot run would only print red rows at its launch. The exported parameters of the
    self-supervised model leave out its projection head, used by the training only.

    Args:
        task (str): The task of the model.
        temporal_type (Optional[str]): The temporal module of the sequence models, None for the other tasks.
        fusion_type (Optional[str]): The fusion of the members of the comparison model, None for the other tasks.
    """
    hide_known_export_warnings()
    if task in ["object_detection", "instance_segmentation"]:
        model = build_test_detector(dtype=torch.float32, instance_segmentation=task == "instance_segmentation")
    else:
        model = build_test_model(task=task, temporal_type=temporal_type, fusion_type=fusion_type,
                                 global_pooling="average")
    export_sizes = estimate_export_sizes(task=task, model=model.train(), optimizer=torch.optim.SGD(
        params=model.parameters(), lr=0.1, momentum=0.9), keeps_moving_average=True)
    assert export_sizes["errors"] == []
    assert all(row["lowest_bytes"] is not None for row in export_sizes["rows"])
    if task == "self_supervised":
        assert 0 < export_sizes["exported_parameters"] < export_sizes["total_parameters"]
    else:
        assert export_sizes["exported_parameters"] == export_sizes["total_parameters"]


def test_failed_trace_is_printed_without_stopping_the_estimates(monkeypatch: pytest.MonkeyPatch,
                                                                capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a TorchScript trace failing leaves both TorchScript rows without size and its error listed, the
    other formats still being estimated, and that the panel prints every format with 'failed' for the missing sizes and
    the error.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the TorchScript measure by a failing one.
        capsys (pytest.CaptureFixture): Captures the printed panel.
    """
    monkeypatch.setattr(target=export_size_estimates, name="measure_torchscript_bytes", value=raise_trace_error)
    model = build_test_model(task="classification", temporal_type=None, fusion_type=None, global_pooling="average")
    export_sizes = estimate_export_sizes(task="classification", model=model,
                                         optimizer=torch.optim.Adam(params=model.parameters()),
                                         keeps_moving_average=True)
    rows = {row["label"]: row for row in export_sizes["rows"]}
    assert rows["TorchScript FP32"]["lowest_bytes"] is None and rows["TorchScript FP16"]["lowest_bytes"] is None
    assert rows["ONNX"]["lowest_bytes"] is not None
    assert export_sizes["errors"] == ["TorchScript trace: RuntimeError: the trace failed"]

    print_export_size_summary(export_sizes=export_sizes)
    printed_text = capsys.readouterr().out
    for label in rows:
        assert label in printed_text
    assert "failed" in printed_text and "the trace failed" in printed_text
    assert f"Exported Parameters : {export_sizes['total_parameters']:,}" in printed_text
