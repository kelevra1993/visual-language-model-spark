"""
Tests of the deployment of the frame selection, vision_training/deployment/frame_selection_deployment.py and
frame_selection_decoder_backend.py: the decoding settings read the thresholds of the evaluated checkpoint, the decoder
model and the ensembles are written to the Triton repository, the Python backend model decodes like the evaluation
(run with a stand-in for the Triton backend utilities, which only exist in the Triton image), and the local inference
returns the outputs of the ensembles.
"""

import importlib.util
import json
import sys
import types
import numpy as np
import pytest
import torch

from pathlib import Path
from typing import Any, Dict, List

from vision_training.deployment.deployment_utilities import get_served_model, run_local_inference
from vision_training.deployment.export_specifications import build_inference_model, get_export_specification
from vision_training.deployment.frame_selection_deployment import (DECODER_OUTPUT_TYPES, build_decoding_settings,
                                                                   export_frame_selection_decoding)
from vision_training.models.sequence.frame_selection_decoding import run_frame_selection
from vision_training.models.sequence.frame_selector import FrameSelector
from tests.fixtures.configuration_fixtures import get_sequence_model_configuration

# Decoding settings of the test frame selector: classes 0 and 2 wanted, class 1 unselected
TEST_DECODING_SETTINGS = {"decoding": "hungarian", "frames_per_class": 1, "unselected_class_index": 1,
                          "number_candidates": 2, "class_names": ["A", "C"], "class_thresholds": [0.5, None]}


class FakeTritonTensor:
    """
    Stand-in for the tensors of the Triton Python backend, holding a named array.
    """

    def __init__(self, name: str, array: np.ndarray) -> None:
        """
        Keeps the name and the array of the tensor.

        Args:
            name (str): The name of the tensor.
            array (np.ndarray): The values of the tensor.
        """
        self.name = name
        self.array = array

    def as_numpy(self) -> np.ndarray:
        """
        Gives the values of the tensor, like the Triton tensors.

        Returns:
            np.ndarray: The values of the tensor.
        """
        return self.array


class FakeInferenceRequest:
    """
    Stand-in for an inference request of the Triton Python backend, holding its inputs by name.
    """

    def __init__(self, inputs: Dict[str, np.ndarray]) -> None:
        """
        Keeps the inputs of the request.

        Args:
            inputs (Dict[str, np.ndarray]): The batch of every input, keyed by input name.
        """
        self.inputs = inputs


class FakeInferenceResponse:
    """
    Stand-in for an inference response of the Triton Python backend, holding its output tensors.
    """

    def __init__(self, output_tensors: List[FakeTritonTensor]) -> None:
        """
        Keeps the outputs of the response.

        Args:
            output_tensors (List[FakeTritonTensor]): The output tensors, in order.
        """
        self.output_tensors = output_tensors


def get_fake_input_tensor_by_name(inference_request: FakeInferenceRequest, name: str) -> FakeTritonTensor:
    """
    Stand-in for get_input_tensor_by_name of the Triton backend utilities, with the same argument names.

    Args:
        inference_request (FakeInferenceRequest): The request.
        name (str): The name of the input.

    Returns:
        FakeTritonTensor: The input tensor.
    """
    return FakeTritonTensor(name=name, array=inference_request.inputs[name])


def build_test_frame_selector() -> FrameSelector:
    """
    Builds a small untrained frame selector on the CPU, in evaluation mode, its class 1 marked as unselected.

    Returns:
        FrameSelector: The frame selector, predicting 3 classes for sequences of up to 4 frames of 64 pixels.
    """
    torch.manual_seed(seed=0)
    configuration, device, dtype = get_sequence_model_configuration(temporal_type="gru")
    configuration["Selection"] = {"unselected_category": "unselected", "decoding": "hungarian",
                                  "frames_per_class": 1, "selection_loss_weight": 1.0}
    model = FrameSelector(configuration=configuration, device=device, dtype=dtype).eval()
    model.wanted_class_mask[1] = 0.0
    return model


@pytest.mark.parametrize(argnames="evaluated", argvalues=[True, False])
def test_decoding_settings_read_the_thresholds_of_the_checkpoint(tmp_path: Path, evaluated: bool) -> None:
    """
    Verifies the decoding settings of a frame selection run: the selection settings, the number of candidates, and the
    thresholds and class names of the evaluation of the exported checkpoint, or no low-confidence flag and the class
    indices as names when the checkpoint was not evaluated.

    Args:
        tmp_path (Path): The pytest temporary directory, holding the run folder.
        evaluated (bool): Whether the exported checkpoint was evaluated.
    """
    if evaluated:
        evaluation_folder = tmp_path / "Evaluation" / "Iteration_4"
        evaluation_folder.mkdir(parents=True)
        (evaluation_folder / "class_thresholds.json").write_text(data=json.dumps(obj={
            "target_precision": 0.99, "decoding": "greedy", "unselected_class": "unselected",
            "classes": {"A": {"threshold": 0.7, "recall": 0.9}, "C": None}}))
    decoding_settings = build_decoding_settings(
        experiment_configuration={"evaluation": {"top_k": 3}},
        model_configuration={"Data": {"label_dictionary": {1: 0, 2: 1, 3: 2}},
                             "Selection": {"decoding": "greedy", "frames_per_class": 2}},
        unselected_class_index=1, checkpoint_path=tmp_path / "Weights" / "Iteration_4" / "model_000004.pt",
        experiment_folder=tmp_path)

    assert {key: decoding_settings[key] for key in ["decoding", "frames_per_class", "unselected_class_index",
                                                    "number_candidates"]} == {
        "decoding": "greedy", "frames_per_class": 2, "unselected_class_index": 1, "number_candidates": 3}
    if evaluated:
        assert decoding_settings["class_names"] == ["A", "C"] and decoding_settings["class_thresholds"] == [0.7, None]
    else:
        assert decoding_settings["class_names"] == ["0", "2"] and decoding_settings["class_thresholds"] == [0.0, 0.0]


def test_decoder_and_ensembles_are_written(tmp_path: Path) -> None:
    """
    Verifies the Triton repository entries of the decoding: the decoder model with its Python model, decoding module
    and settings, run on the CPU on the per-frame probabilities, and one ensemble per format chaining the exported frame
    selector (its TorchScript output named after the PyTorch backend convention) with the decoder, with an empty
    version folder and the outputs of the decoder.

    Args:
        tmp_path (Path): The pytest temporary directory, used as the Triton repository.
    """
    inference_model = build_inference_model(task="frame_selection", model=build_test_frame_selector())
    specification = get_export_specification(task="frame_selection", inference_model=inference_model)
    export_frame_selection_decoding(repository=tmp_path, model_version=1, formats=["torchscript", "onnx"],
                                    specification=specification, decoding_settings=TEST_DECODING_SETTINGS,
                                    number_classes=3, maximum_batch_size=4)

    # Decoder: Python model, decoding module and settings in its version folder
    decoder_folder = tmp_path / "frame_selector_decoder"
    assert sorted(path.name for path in (decoder_folder / "1").iterdir()) == [
        "decoding_settings.json", "frame_selection_decoding.py", "model.py"]
    assert json.loads(s=(decoder_folder / "1" / "decoding_settings.json").read_text()) == TEST_DECODING_SETTINGS
    decoder_configuration = (decoder_folder / "config.pbtxt").read_text()
    assert 'backend: "python"' in decoder_configuration and "kind: KIND_CPU" in decoder_configuration
    assert '{ name: "class_probabilities" data_type: TYPE_FP32 dims: [ -1, 3 ] }' in decoder_configuration
    assert '{ name: "low_confidence" data_type: TYPE_BOOL dims: [ 2, 1 ] }' in decoder_configuration
    assert '{ name: "candidate_frames" data_type: TYPE_INT64 dims: [ 2, 2 ] }' in decoder_configuration

    # Ensembles: the frame selector of the format followed by the decoder
    for export_format, network_output_name in [("torchscript", "class_probabilities__0"),
                                               ("onnx", "class_probabilities")]:
        ensemble_folder = tmp_path / f"frame_selector_{export_format}_ensemble"
        assert (ensemble_folder / "1").is_dir() and not any((ensemble_folder / "1").iterdir())
        ensemble_configuration = (ensemble_folder / "config.pbtxt").read_text()
        assert 'platform: "ensemble"' in ensemble_configuration and "max_batch_size: 4" in ensemble_configuration
        assert '{ name: "images" data_type: TYPE_FP32 dims: [ -1, 3, 64, 64 ] }' in ensemble_configuration
        assert f'model_name: "frame_selector_{export_format}"' in ensemble_configuration
        assert (f'output_map {{ key: "{network_output_name}" value: "class_probabilities" }}' in
                ensemble_configuration)
        assert 'model_name: "frame_selector_decoder"' in ensemble_configuration
        for output_name in DECODER_OUTPUT_TYPES:
            assert f'output_map {{ key: "{output_name}" value: "{output_name}" }}' in ensemble_configuration


def test_python_backend_decodes_like_the_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the Python backend model as Triton runs it, loaded from the version folder of the decoder written to the
    repository with a stand-in for the Triton backend utilities: every request of a batch of sequences is answered with
    the outputs of run_frame_selection for every sequence, stacked in the order of the decoder outputs.

    Args:
        tmp_path (Path): The pytest temporary directory, used as the Triton repository.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture installing the stand-in utilities module.
    """
    inference_model = build_inference_model(task="frame_selection", model=build_test_frame_selector())
    specification = get_export_specification(task="frame_selection", inference_model=inference_model)
    export_frame_selection_decoding(repository=tmp_path, model_version=1, formats=["onnx"],
                                    specification=specification, decoding_settings=TEST_DECODING_SETTINGS,
                                    number_classes=3, maximum_batch_size=4)

    # Stand-in for the backend utilities of the Triton image, then the Python model loaded as Triton does
    fake_utilities = types.ModuleType(name="triton_python_backend_utils")
    fake_utilities.Tensor = FakeTritonTensor
    fake_utilities.InferenceResponse = FakeInferenceResponse
    fake_utilities.get_input_tensor_by_name = get_fake_input_tensor_by_name
    monkeypatch.setitem(dic=sys.modules, name="triton_python_backend_utils", value=fake_utilities)
    module_specification = importlib.util.spec_from_file_location(
        name="frame_selector_decoder_model", location=tmp_path / "frame_selector_decoder" / "1" / "model.py")
    backend_module = importlib.util.module_from_spec(spec=module_specification)
    module_specification.loader.exec_module(module=backend_module)
    backend_model = backend_module.TritonPythonModel()
    backend_model.initialize(args={})

    # Two requests of two sequences of 4 frames
    probabilities = np.random.default_rng(seed=0).dirichlet(alpha=np.ones(shape=3),
                                                            size=(2, 2, 4)).astype(dtype=np.float32)
    responses = backend_model.execute(requests=[FakeInferenceRequest(inputs={"class_probabilities": batch})
                                                for batch in probabilities])
    assert len(responses) == 2
    for response, batch in zip(responses, probabilities):
        assert [tensor.name for tensor in response.output_tensors] == list(DECODER_OUTPUT_TYPES)
        for sequence_index, sequence_probabilities in enumerate(batch):
            expected_outputs = run_frame_selection(probabilities=sequence_probabilities,
                                                   decoding_settings=TEST_DECODING_SETTINGS)
            for tensor in response.output_tensors:
                assert np.array_equal(a1=tensor.as_numpy()[sequence_index], a2=expected_outputs[tensor.name])


def test_served_model_of_a_frame_selector_is_its_ensemble() -> None:
    """
    Verifies the Triton model the clients query: the exported model with its outputs for the other tasks, the
    ensemble with the outputs of the decoder for a frame selector.
    """
    specification = {"model_name": "frame_selector", "outputs": [{"name": "class_probabilities"}]}
    assert get_served_model(specification=specification, export_format="onnx", decoding_settings=None) == (
        "frame_selector_onnx", ["class_probabilities"])
    assert get_served_model(specification=specification, export_format="onnx",
                            decoding_settings=TEST_DECODING_SETTINGS) == ("frame_selector_onnx_ensemble",
                                                                          list(DECODER_OUTPUT_TYPES))


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="The local inference runs on the GPU.")
def test_local_inference_decodes_the_selection() -> None:
    """
    Verifies the local inference of a frame selector: the network on the GPU followed by the decoding of every
    sequence, giving the outputs of the ensembles stacked over the sequences of the request.
    """
    inference_model = build_inference_model(task="frame_selection", model=build_test_frame_selector().cuda())
    images = np.random.default_rng(seed=0).uniform(low=0.0, high=255.0, size=(2, 4, 3, 64, 64)).astype(dtype=np.float32)
    outputs = run_local_inference(inputs={"images": images}, inference_model=inference_model,
                                  decoding_settings=TEST_DECODING_SETTINGS)
    assert [output.shape for output in outputs] == [(2, 2, 1), (2, 2, 1), (2, 2, 1), (2, 2, 2), (2, 2, 2)]

    # The same selection as the decoding of the probabilities of the network
    with torch.no_grad():
        probabilities = inference_model(images=torch.from_numpy(images).cuda()).cpu().numpy()
    assert np.array_equal(a1=outputs[0][1], a2=run_frame_selection(probabilities=probabilities[1],
                                                             decoding_settings=TEST_DECODING_SETTINGS)[
        "selected_frames"])
