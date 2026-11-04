"""
Tests of write_triton_configuration of vision_training/deployment/triton_configuration.py: the comparison configuration
has one input line per member, each with its own varying number of frames, and the output names of the PyTorch backend.
"""

from pathlib import Path

from vision_training.deployment.export_specifications import build_inference_model, get_export_specification
from vision_training.deployment.triton_configuration import write_triton_configuration
from vision_training.models.sequence.sequence_comparator import SequenceComparator
from tests.fixtures.configuration_fixtures import get_sequence_comparison_model_configuration


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
