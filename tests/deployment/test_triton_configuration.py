"""
Tests of write_triton_configuration of vision_training/deployment/triton_configuration.py: the comparison configuration
has one input line per member, each with its own varying number of frames, and the output names of the PyTorch backend,
and a misspelled export format or TensorRT precision is refused instead of being served as plain ONNX Runtime.
"""

import pytest

from pathlib import Path
from typing import Any, List

from vision_training.deployment.export_specifications import build_inference_model, get_export_specification
from vision_training.deployment.triton_configuration import get_served_output_names, write_triton_configuration
from vision_training.models.sequence.sequence_comparator import SequenceComparator
from tests.fixtures.configuration_fixtures import get_sequence_comparison_model_configuration

# Export specification of a classifier of three classes on 32 pixel images, as get_export_specification gives it
CLASSIFIER_SPECIFICATION = {"inputs": [{"name": "images", "data_type": "TYPE_FP32", "dimensions": [3, 32, 32]}],
                            "outputs": [{"name": "class_probabilities", "data_type": "TYPE_FP32", "dimensions": [3],
                                         "reshape_to_scalar": False}]}


def test_comparison_triton_configuration_has_one_line_per_member(tmp_path: Path) -> None:
    """
    Verifies the Triton configuration of a sequence comparator: one input line per member, in the order of the
    forward arguments of the model and named like them, each with its own varying number of frames, and one vector of
    class probabilities per pair, named after the convention of the PyTorch backend.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration, device, dtype = get_sequence_comparison_model_configuration(temporal_type="gru",
                                                                               fusion_type="difference")
    inference_model = build_inference_model(task="sequence_comparison", model=SequenceComparator(
        configuration=configuration, device=device, dtype=dtype))
    specification = get_export_specification(task="sequence_comparison", inference_model=inference_model)
    assert specification["model_name"] == "sequence_comparator"

    write_triton_configuration(model_directory=tmp_path, model_name="sequence_comparator_torchscript",
                               model_version=1, export_format="torchscript", specification=specification,
                               maximum_batch_size=8, tensorrt_precision=None)
    triton_configuration = (tmp_path / "config.pbtxt").read_text()
    assert ('input [\n'
            '  { name: "before_images" data_type: TYPE_FP32 dims: [ -1, 3, 64, 64 ] },\n'
            '  { name: "after_images" data_type: TYPE_FP32 dims: [ -1, 3, 64, 64 ] }\n'
            ']\n') in triton_configuration
    assert '{ name: "class_probabilities__0" data_type: TYPE_FP32 dims: [ 3 ] }' in triton_configuration


@pytest.mark.parametrize(argnames=("export_format", "tensorrt_precision", "expected_message"), argvalues=[
    ("tensorrt", "FP16", "Expected an export format among"),
    ("TorchScript", None, "Expected an export format among"),
    ("onnx", "fp16", "Expected a TensorRT precision among"),
    ("onnx", "INT8", "Expected a TensorRT precision among")])
def test_unknown_export_format_or_precision_is_refused(tmp_path: Path, export_format: str, tensorrt_precision: Any,
                                                       expected_message: str) -> None:
    """
    Verifies that a format other than 'torchscript' and 'onnx' and a TensorRT precision other than 'FP32', 'FP16' and
    None are refused without writing any configuration, instead of giving an ONNX Runtime model whose TensorRT block
    is silently left out, and that the served output names refuse the misspelled format too.

    Args:
        tmp_path (Path): The pytest temporary directory.
        export_format (str): The export format given.
        tensorrt_precision (Any): The TensorRT precision given.
        expected_message (str): The pattern of the expected error.
    """
    with pytest.raises(expected_exception=ValueError, match=expected_message):
        write_triton_configuration(model_directory=tmp_path, model_name="image_classifier_onnx", model_version=1,
                                   export_format=export_format, specification=CLASSIFIER_SPECIFICATION,
                                   maximum_batch_size=8, tensorrt_precision=tensorrt_precision)
    assert not (tmp_path / "config.pbtxt").exists()
    if export_format not in ["torchscript", "onnx"]:
        with pytest.raises(expected_exception=ValueError, match=expected_message):
            get_served_output_names(export_format=export_format, specification=CLASSIFIER_SPECIFICATION)


@pytest.mark.parametrize(argnames=("export_format", "tensorrt_precision", "expected_lines", "absent_lines"), argvalues=[
    ("onnx", "FP16", ['platform: "onnxruntime_onnx"', 'value: "FP16"'], []),
    ("onnx", None, ['platform: "onnxruntime_onnx"'], ['name: "tensorrt"']),
    ("torchscript", "FP32", ['platform: "pytorch_libtorch"', '"class_probabilities__0"'], ['name: "tensorrt"'])])
def test_known_export_formats_get_their_platform(tmp_path: Path, export_format: str, tensorrt_precision: Any,
                                                 expected_lines: List[str], absent_lines: List[str]) -> None:
    """
    Verifies the platform of the two known formats, the TensorRT block being written for an ONNX model with a
    precision only, and never for the TorchScript model.

    Args:
        tmp_path (Path): The pytest temporary directory.
        export_format (str): The export format given.
        tensorrt_precision (Any): The TensorRT precision given.
        expected_lines (List[str]): The texts the configuration must hold.
        absent_lines (List[str]): The texts the configuration must not hold.
    """
    write_triton_configuration(model_directory=tmp_path, model_name=f"image_classifier_{export_format}",
                               model_version=1, export_format=export_format, specification=CLASSIFIER_SPECIFICATION,
                               maximum_batch_size=8, tensorrt_precision=tensorrt_precision)
    triton_configuration = (tmp_path / "config.pbtxt").read_text()
    assert all(expected_line in triton_configuration for expected_line in expected_lines)
    assert not any(absent_line in triton_configuration for absent_line in absent_lines)
