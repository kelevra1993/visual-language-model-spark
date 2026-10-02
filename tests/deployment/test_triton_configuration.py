import pytest

from pathlib import Path

from vision_training.deployment.export_specifications import build_inference_model, get_export_specification
from vision_training.deployment.triton_configuration import write_triton_configuration
from vision_training.models.detection.two_stage_detector import TwoStageDetector
from vision_training.models.sequence.sequence_labeller import SequenceLabeller
from tests.fixtures.configuration_fixtures import get_detection_model_configuration, get_sequence_model_configuration

# Triton configurations of the detector written by the export before it was generalised to every task, for 1024
# pixel images, 100 detections, batches of up to 16 images and version 1
DETECTION_TORCHSCRIPT_CONFIGURATION = (
    'name: "object_detector_torchscript"\n'
    'platform: "pytorch_libtorch"\n'
    'max_batch_size: 16\n'
    'input [\n'
    '  { name: "images" data_type: TYPE_FP32 dims: [ 3, 1024, 1024 ] }\n'
    ']\n'
    'output [\n'
    '  { name: "detection_boxes__0" data_type: TYPE_FP32 dims: [ 100, 4 ] },\n'
    '  { name: "detection_scores__1" data_type: TYPE_FP32 dims: [ 100 ] },\n'
    '  { name: "detection_labels__2" data_type: TYPE_INT64 dims: [ 100 ] },\n'
    '  { name: "number_detections__3" data_type: TYPE_INT64 dims: [ 1 ] reshape: { shape: [ ] } }\n'
    ']\n'
    'dynamic_batching { max_queue_delay_microseconds: 1000 }\n'
    'instance_group [ { kind: KIND_GPU count: 1 } ]\n'
    'version_policy: { specific: { versions: [ 1 ] } }\n'
)
DETECTION_ONNX_FP16_CONFIGURATION = (
    'name: "object_detector_onnx"\n'
    'platform: "onnxruntime_onnx"\n'
    'max_batch_size: 16\n'
    'input [\n'
    '  { name: "images" data_type: TYPE_FP32 dims: [ 3, 1024, 1024 ] }\n'
    ']\n'
    'output [\n'
    '  { name: "detection_boxes" data_type: TYPE_FP32 dims: [ 100, 4 ] },\n'
    '  { name: "detection_scores" data_type: TYPE_FP32 dims: [ 100 ] },\n'
    '  { name: "detection_labels" data_type: TYPE_INT64 dims: [ 100 ] },\n'
    '  { name: "number_detections" data_type: TYPE_INT64 dims: [ 1 ] reshape: { shape: [ ] } }\n'
    ']\n'
    'dynamic_batching { max_queue_delay_microseconds: 1000 }\n'
    'instance_group [ { kind: KIND_GPU count: 1 } ]\n'
    'version_policy: { specific: { versions: [ 1 ] } }\n'
    'optimization { execution_accelerators { gpu_execution_accelerator: [ {\n'
    '  name: "tensorrt"\n'
    '  parameters { key: "precision_mode" value: "FP16" }\n'
    '  parameters { key: "max_workspace_size_bytes" value: "4294967296" }\n'
    '  parameters { key: "trt_engine_cache_enable" value: "1" }\n'
    '  parameters { key: "trt_engine_cache_path" value: "/tmp/tensorrt_engine_cache" }\n'
    '} ] } }\n'
)


@pytest.mark.parametrize(argnames=("export_format", "tensorrt_precision", "expected_configuration"), argvalues=[
    ("torchscript", None, DETECTION_TORCHSCRIPT_CONFIGURATION), ("onnx", "FP16", DETECTION_ONNX_FP16_CONFIGURATION)])
def test_detection_triton_configuration_is_unchanged(tmp_path: Path, export_format: str,
                                                     tensorrt_precision: str | None,
                                                     expected_configuration: str) -> None:
    """
    Verifies that the Triton configuration written from the export specification of the detector is exactly the one
    written before the deployment was generalised to every task, so that the served detectors keep working.

    Args:
        tmp_path (Path): The pytest temporary directory.
        export_format (str): The export format.
        tensorrt_precision (str | None): The TensorRT precision of the ONNX model, or None.
        expected_configuration (str): The expected content of config.pbtxt.
    """
    configuration, device, dtype = get_detection_model_configuration()
    inference_model = build_inference_model(task="object_detection", model=TwoStageDetector(
        configuration=configuration, device=device, dtype=dtype))
    specification = get_export_specification(task="object_detection", inference_model=inference_model)
    assert f"{specification['model_name']}_{export_format}" == f"object_detector_{export_format}"

    write_triton_configuration(model_directory=tmp_path, model_name=f"object_detector_{export_format}",
                               model_version=1, export_format=export_format, specification=specification,
                               maximum_batch_size=16, tensorrt_precision=tensorrt_precision)
    assert (tmp_path / "config.pbtxt").read_text() == expected_configuration


def test_sequence_triton_configuration_has_dynamic_frames(tmp_path: Path) -> None:
    """
    Verifies the Triton configuration of a sequence labeller: windows with a varying number of frames, and one vector
    of class probabilities per frame.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration, device, dtype = get_sequence_model_configuration(temporal_type="gru")
    inference_model = build_inference_model(task="sequence_labelling", model=SequenceLabeller(
        configuration=configuration, device=device, dtype=dtype))
    specification = get_export_specification(task="sequence_labelling", inference_model=inference_model)

    write_triton_configuration(model_directory=tmp_path, model_name="sequence_labeller_onnx", model_version=2,
                               export_format="onnx", specification=specification, maximum_batch_size=4,
                               tensorrt_precision=None)
    triton_configuration = (tmp_path / "config.pbtxt").read_text()
    assert '{ name: "images" data_type: TYPE_FP32 dims: [ -1, 3, 64, 64 ] }' in triton_configuration
    assert '{ name: "class_probabilities" data_type: TYPE_FP32 dims: [ -1, 3 ] }' in triton_configuration
    assert "max_batch_size: 4" in triton_configuration and "versions: [ 2 ]" in triton_configuration
