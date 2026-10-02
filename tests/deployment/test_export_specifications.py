import onnxruntime
import pytest
import torch

from pathlib import Path
from typing import Optional

from vision_training.deployment.export_model import export_onnx, export_torchscript, hide_known_export_warnings
from vision_training.deployment.export_specifications import (build_inference_model, convert_outputs_to_numpy,
                                                              get_export_specification)
from vision_training.models.model_registry import TASK_MODEL_CLASSES
from vision_training.models.task_model import TaskModel
from tests.fixtures.configuration_fixtures import (get_classification_model_configuration,
                                                   get_sequence_model_configuration)


def build_test_model(task: str, temporal_type: Optional[str]) -> TaskModel:
    """
    Builds a small untrained model of a task on the CPU, in evaluation mode, with seeded weights: bidirectional
    recurrent modules, and a causal transformer for the sequence labeller.

    Args:
        task (str): "classification", "sequence_classification" or "sequence_labelling".
        temporal_type (Optional[str]): The temporal module of the sequence models, None for the classifier.

    Returns:
        TaskModel: The model of the task.
    """
    torch.manual_seed(seed=0)
    if temporal_type is None:
        configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    else:
        configuration, device, dtype = get_sequence_model_configuration(temporal_type=temporal_type)
        configuration["TemporalModel"]["bidirectional"] = temporal_type in ["gru", "lstm"]
        configuration["TemporalModel"]["causal"] = task == "sequence_labelling" and temporal_type == "transformer"
    return TASK_MODEL_CLASSES[task](configuration=configuration, device=device, dtype=dtype).eval()


@pytest.mark.parametrize(argnames=("task", "temporal_type"), argvalues=[
    ("classification", None), ("sequence_classification", "gru"), ("sequence_classification", "transformer"),
    ("sequence_labelling", "lstm"), ("sequence_labelling", "transformer"),
    ("sequence_labelling", "temporal_convolution")])
def test_exports_match_the_eager_model(tmp_path: Path, task: str, temporal_type: Optional[str]) -> None:
    """
    Verifies the TorchScript and ONNX exports of the inference models of the classification and sequence tasks: both
    reproduce the eager class probabilities on a batch of another size than the traced one and, for the sequence
    models, with another number of frames, so that Triton can batch requests and receive windows of any length.

    Args:
        tmp_path (Path): The pytest temporary directory.
        task (str): The task of the exported model.
        temporal_type (Optional[str]): The temporal module of the sequence models, None for the classifier.
    """
    hide_known_export_warnings()
    inference_model = build_inference_model(task=task, model=build_test_model(task=task, temporal_type=temporal_type))
    specification = get_export_specification(task=task, inference_model=inference_model)
    example_images = torch.rand(size=specification["example_shape"], generator=torch.Generator().manual_seed(1)) * 255.0
    check_images = torch.rand(size=specification["check_shape"], generator=torch.Generator().manual_seed(2)) * 255.0
    with torch.no_grad():
        reference_outputs = convert_outputs_to_numpy(outputs=inference_model(images=check_images))

    # One vector of class probabilities per sample, or per frame for the sequence labeller
    expected_shape = tuple(specification["check_shape"][:2]) + (3,) if task == "sequence_labelling" else (3, 3)
    assert reference_outputs[0].shape == expected_shape

    # TorchScript export, traced on the example batch and run on the check batch
    export_torchscript(inference_model=inference_model, example_images=example_images,
                       output_path=tmp_path / "model.pt")
    with torch.no_grad():
        torchscript_outputs = convert_outputs_to_numpy(outputs=torch.jit.load(str(tmp_path / "model.pt"))(check_images))
    assert specification["compare_outputs"](reference=reference_outputs, candidate=torchscript_outputs,
                                            **specification["comparison_tolerances"]) == []

    # ONNX export with dynamic batch (and frames), run by ONNX Runtime on the check batch
    export_onnx(inference_model=inference_model, example_images=example_images, output_path=tmp_path / "model.onnx",
                specification=specification, maximum_batch_size=4)
    session = onnxruntime.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {"images": check_images.numpy()})
    assert specification["compare_outputs"](reference=reference_outputs, candidate=onnx_outputs,
                                            **specification["comparison_tolerances"]) == []


@pytest.mark.parametrize(argnames=("task", "temporal_type"), argvalues=[
    ("sequence_classification", "gru"), ("sequence_classification", "attention_pooling"),
    ("sequence_labelling", "transformer"), ("sequence_labelling", "temporal_convolution")])
def test_inference_path_matches_the_training_path_on_full_windows(task: str, temporal_type: str) -> None:
    """
    Verifies that the path of the sequence models without sequence mask, run by the inference models, gives the
    predictions of the masked path used in training and evaluation when every frame of the windows is real.

    Args:
        task (str): The sequence task of the model.
        temporal_type (str): The temporal module of the model.
    """
    model = build_test_model(task=task, temporal_type=temporal_type)
    frames = torch.rand(size=(2, 4, 3, 64, 64), generator=torch.Generator().manual_seed(3)) * 255.0
    with torch.no_grad():
        masked_probabilities = model(input_tensor=frames, sequence_mask=torch.ones(size=(2, 4), dtype=torch.bool))[
            "class_probabilities"]
        inference_probabilities = build_inference_model(task=task, model=model)(images=frames)
    assert torch.allclose(input=inference_probabilities, other=masked_probabilities, atol=1e-6)
