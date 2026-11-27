"""
Tests of write_triton_configuration of vision_training/deployment/triton_configuration.py: the comparison configuration
has one input line per member, each with its own varying number of frames, and the output names of the PyTorch backend,
a misspelled export format or TensorRT precision is refused instead of being served as plain ONNX Runtime, the
TensorRT block sets no workspace limit, and the TensorRT engines of an ONNX export are cached under a prefix naming its
files, so that an export of other weights of the same graph never loads the engines built from an earlier export.
"""

import re
import numpy as np
import onnx
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

# Line of a written config.pbtxt giving the prefix of the cached TensorRT engines of the export
ENGINE_CACHE_PREFIX_PATTERN = r'key: "trt_engine_cache_prefix" value: "([^"]+)"'


def write_onnx_export(version_directory: Path, weight_value: float) -> None:
    """
    Writes a stand-in ONNX export into a version folder as export_formats.export_onnx lays it out, the graph in
    'model.onnx' and its weights in 'model.onnx.data', so that two exports of the same graph with other weights differ
    only by their weights file, the case of another checkpoint of the same run exported into the same version.

    Args:
        version_directory (Path): The version folder of the export, created if missing.
        weight_value (float): The value of every weight of the stand-in model.
    """
    version_directory.mkdir(parents=True, exist_ok=True)
    weights = onnx.numpy_helper.from_array(np.full(shape=(3, 3), fill_value=weight_value, dtype=np.float32),
                                           name="weights")
    image_input = onnx.helper.make_tensor_value_info(name="images", elem_type=onnx.TensorProto.FLOAT, shape=[1, 3])
    probabilities_output = onnx.helper.make_tensor_value_info(name="class_probabilities",
                                                              elem_type=onnx.TensorProto.FLOAT, shape=[1, 3])
    graph = onnx.helper.make_graph(nodes=[onnx.helper.make_node(op_type="MatMul", inputs=["images", "weights"],
                                                                outputs=["class_probabilities"])],
                                   name="main_graph", inputs=[image_input], outputs=[probabilities_output],
                                   initializer=[weights])
    onnx.save_model(proto=onnx.helper.make_model(graph=graph), f=str(version_directory / "model.onnx"),
                    save_as_external_data=True, all_tensors_to_one_file=True, location="model.onnx.data",
                    size_threshold=0)


def read_engine_cache_prefix(model_directory: Path, model_version: int, weight_value: float,
                             tensorrt_precision: str) -> str:
    """
    Writes a stand-in ONNX export of the given weights into a model folder, then its Triton configuration with
    TensorRT, as export_experiment does, and reads back the prefix of its cached engines.

    Args:
        model_directory (Path): The model folder of the export.
        model_version (int): The Triton version of the export.
        weight_value (float): The value of every weight of the stand-in model.
        tensorrt_precision (str): 'FP16' or 'FP32'.

    Returns:
        str: The trt_engine_cache_prefix of the written config.pbtxt.
    """
    write_onnx_export(version_directory=model_directory / str(model_version), weight_value=weight_value)
    write_triton_configuration(model_directory=model_directory, model_name="image_classifier_onnx",
                               model_version=model_version, export_format="onnx",
                               specification=CLASSIFIER_SPECIFICATION, maximum_batch_size=8,
                               tensorrt_precision=tensorrt_precision, torchscript_precision="FP32")
    return re.findall(pattern=ENGINE_CACHE_PREFIX_PATTERN, string=(model_directory / "config.pbtxt").read_text())[0]


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
                               maximum_batch_size=8, tensorrt_precision=None, torchscript_precision="FP32")
    triton_configuration = (tmp_path / "config.pbtxt").read_text()
    assert ('input [\n'
            '  { name: "before_images" data_type: TYPE_FP32 dims: [ -1, 3, 64, 64 ] },\n'
            '  { name: "after_images" data_type: TYPE_FP32 dims: [ -1, 3, 64, 64 ] }\n'
            ']\n') in triton_configuration
    assert '{ name: "class_probabilities__0" data_type: TYPE_FP32 dims: [ 3 ] }' in triton_configuration


@pytest.mark.parametrize(argnames=("export_format", "tensorrt_precision", "torchscript_precision", "expected_message"),
                         argvalues=[("tensorrt", "FP16", "FP32", "Expected an export format among"),
                                    ("TorchScript", None, "FP32", "Expected an export format among"),
                                    ("onnx", "fp16", "FP32", "Expected a TensorRT precision among"),
                                    ("onnx", "INT8", "FP32", "Expected a TensorRT precision among"),
                                    ("torchscript", None, "fp16", "Expected a TorchScript precision among")])
def test_unknown_export_format_or_precision_is_refused(tmp_path: Path, export_format: str, tensorrt_precision: Any,
                                                       torchscript_precision: Any, expected_message: str) -> None:
    """
    Verifies that a format other than 'torchscript' and 'onnx', a TensorRT precision other than 'FP32', 'FP16' and
    None and a TorchScript precision other than 'FP32' and 'FP16' are refused without writing any configuration,
    instead of giving an ONNX Runtime model whose TensorRT block is silently left out or a TorchScript model served
    from a file of another name, and that the served output names refuse the misspelled format too.

    Args:
        tmp_path (Path): The pytest temporary directory.
        export_format (str): The export format given.
        tensorrt_precision (Any): The TensorRT precision given.
        torchscript_precision (Any): The TorchScript precision given.
        expected_message (str): The pattern of the expected error.
    """
    with pytest.raises(expected_exception=ValueError, match=expected_message):
        write_triton_configuration(model_directory=tmp_path, model_name="image_classifier_onnx", model_version=1,
                                   export_format=export_format, specification=CLASSIFIER_SPECIFICATION,
                                   maximum_batch_size=8, tensorrt_precision=tensorrt_precision,
                                   torchscript_precision=torchscript_precision)
    assert not (tmp_path / "config.pbtxt").exists()
    if export_format not in ["torchscript", "onnx"]:
        with pytest.raises(expected_exception=ValueError, match=expected_message):
            get_served_output_names(export_format=export_format, specification=CLASSIFIER_SPECIFICATION)


@pytest.mark.parametrize(argnames=("export_format", "tensorrt_precision", "torchscript_precision", "expected_lines",
                                   "absent_lines"), argvalues=[
    ("onnx", "FP16", "FP16", ['platform: "onnxruntime_onnx"', 'value: "FP16"'],
     ["default_model_filename", "max_workspace_size_bytes"]),
    ("onnx", None, "FP32", ['platform: "onnxruntime_onnx"'], ['name: "tensorrt"']),
    ("torchscript", "FP32", "FP32", ['platform: "pytorch_libtorch"', '"class_probabilities__0"'],
     ['name: "tensorrt"', "default_model_filename"]),
    ("torchscript", None, "FP16", ['platform: "pytorch_libtorch"\ndefault_model_filename: "model_fp16.pt"\n'],
     ['name: "tensorrt"'])])
def test_known_export_formats_get_their_platform(tmp_path: Path, export_format: str, tensorrt_precision: Any,
                                                 torchscript_precision: str, expected_lines: List[str],
                                                 absent_lines: List[str]) -> None:
    """
    Verifies the platform of the two known formats, the TensorRT block being written for an ONNX model with a
    precision only, without a workspace limit (the 4 GiB limit left no kernel for the batches of 16 sequences of 50
    frames), and never for the TorchScript model, whose FP16 file 'model_fp16.pt' is named by
    'default_model_filename', the FP32 'model.pt' being the default of the backend.

    Args:
        tmp_path (Path): The pytest temporary directory.
        export_format (str): The export format given.
        tensorrt_precision (Any): The TensorRT precision given.
        torchscript_precision (str): The TorchScript precision given.
        expected_lines (List[str]): The texts the configuration must hold.
        absent_lines (List[str]): The texts the configuration must not hold.
    """
    # The model files are written before the configuration, as the export does
    write_onnx_export(version_directory=tmp_path / "1", weight_value=1.0)
    write_triton_configuration(model_directory=tmp_path, model_name=f"image_classifier_{export_format}",
                               model_version=1, export_format=export_format, specification=CLASSIFIER_SPECIFICATION,
                               maximum_batch_size=8, tensorrt_precision=tensorrt_precision,
                               torchscript_precision=torchscript_precision)
    triton_configuration = (tmp_path / "config.pbtxt").read_text()
    assert all(expected_line in triton_configuration for expected_line in expected_lines)
    assert not any(absent_line in triton_configuration for absent_line in absent_lines)


def test_export_of_other_weights_gets_other_tensorrt_engines(tmp_path: Path) -> None:
    """
    Verifies that the TensorRT engines of an ONNX export are cached under a prefix naming its model, version, precision
    and files: an export of other weights of the same graph into the same version, whose 'model.onnx' is unchanged,
    gets another prefix, so that a server keeping its engine cache never serves the engine built from the weights of
    the earlier export; the same files exported again keep their prefix (and their engines), and each precision and
    version gets its own.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    first_prefix = read_engine_cache_prefix(model_directory=tmp_path / "first", model_version=1, weight_value=1.0,
                                            tensorrt_precision="FP16")
    other_weights_prefix = read_engine_cache_prefix(model_directory=tmp_path / "other_weights", model_version=1,
                                                    weight_value=2.0, tensorrt_precision="FP16")

    # Only the weights file differs between the two exports, the graph file being the same
    assert ((tmp_path / "first" / "1" / "model.onnx").read_bytes() ==
            (tmp_path / "other_weights" / "1" / "model.onnx").read_bytes())
    assert re.fullmatch(pattern=r"image_classifier_onnx_1_fp16_[0-9a-f]{16}", string=first_prefix)
    assert other_weights_prefix != first_prefix

    # The same files keep their prefix, another precision or version gets its own (each export in a new folder, as in
    # the staging folder of export_experiment, the ONNX writer appending to an existing weights file)
    assert read_engine_cache_prefix(model_directory=tmp_path / "same_files", model_version=1, weight_value=1.0,
                                    tensorrt_precision="FP16") == first_prefix
    assert read_engine_cache_prefix(model_directory=tmp_path / "other_precision", model_version=1, weight_value=1.0,
                                    tensorrt_precision="FP32") == first_prefix.replace("_fp16_", "_fp32_")
    assert read_engine_cache_prefix(model_directory=tmp_path / "other_version", model_version=2, weight_value=1.0,
                                    tensorrt_precision="FP16").startswith("image_classifier_onnx_2_fp16_")


def test_tensorrt_configuration_before_the_model_file_is_refused(tmp_path: Path) -> None:
    """
    Verifies that the configuration of an ONNX model accelerated by TensorRT is refused while its version folder holds
    no 'model.onnx', no configuration being written, its engine cache prefix being the hash of the exported files,
    while the same model without TensorRT needs no file.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    with pytest.raises(expected_exception=FileNotFoundError, match="before its Triton configuration"):
        write_triton_configuration(model_directory=tmp_path, model_name="image_classifier_onnx", model_version=1,
                                   export_format="onnx", specification=CLASSIFIER_SPECIFICATION, maximum_batch_size=8,
                                   tensorrt_precision="FP16", torchscript_precision="FP32")
    assert not (tmp_path / "config.pbtxt").exists()
    write_triton_configuration(model_directory=tmp_path, model_name="image_classifier_onnx", model_version=1,
                               export_format="onnx", specification=CLASSIFIER_SPECIFICATION, maximum_batch_size=8,
                               tensorrt_precision=None, torchscript_precision="FP32")
    assert "trt_engine_cache" not in (tmp_path / "config.pbtxt").read_text()
