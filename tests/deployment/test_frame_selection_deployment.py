"""
Tests of the deployment of the frame selection, vision_training/deployment/frame_selection_deployment.py and
frame_selection_decoder_backend.py: the decoder model and the ensembles are written to the Triton repository, the
Python backend model decodes like the evaluation (run with a stand-in for the Triton backend utilities, which only exist
in the Triton image), the local inference returns the outputs of the ensembles, and the decoding settings take the
thresholds of the evaluation of the exported checkpoint from the training or from
scripts/evaluation/evaluate_checkpoint.py, the latter when both hold them; the export of a tiny frame selector
installs its networks, decoder and ensembles together, or nothing when its thresholds cannot be used.
"""

import importlib.util
import json
import re
import sys
import types
import numpy as np
import pytest
import torch

from pathlib import Path
from typing import Any, Dict, List

from vision_training.deployment.deployment_utilities import run_local_inference
from vision_training.deployment.export_specifications import build_inference_model, get_export_specification
from vision_training.deployment.frame_selection_deployment import (DECODER_OUTPUT_TYPES, THRESHOLDS_FILE_NAME,
                                                                   build_decoding_settings,
                                                                   export_frame_selection_decoding)
from vision_training.evaluation.checkpoint_evaluation import CHECKPOINT_EVALUATION_FOLDER_NAME
from vision_training.models.sequence.frame_selection_decoding import run_frame_selection
from vision_training.models.sequence.frame_selector import FrameSelector
from vision_training.trainers.base_trainer import EVALUATION_FOLDER_NAME
from vision_training.trainers.checkpointing import get_checkpoint_file
from tests.deployment import test_input_contract
from tests.deployment.test_export_model import read_repository_files
from tests.fixtures.configuration_fixtures import get_sequence_model_configuration

# Colour codes and opening or closing terminal hyperlinks of the console lines, removed before a line is read
CONSOLE_CODE_PATTERN = r"\x1b\[[0-9;]*m|\x1b\]8;;[^\x1b]*\x1b\\"

# Decoding settings of the test frame selector: classes 0 and 2 wanted, class 1 unselected
TEST_DECODING_SETTINGS = {"decoding": "hungarian", "frames_per_class": 1, "unselected_class_index": 1,
                          "number_candidates": 2, "class_names": ["A", "C"], "class_thresholds": [0.5, None]}


def strip_console_codes(text: str) -> str:
    """
    Removes the colour codes and the terminal hyperlinks of the console output of the export (print_color,
    format_path_link), so that a test reads the line as plain text, the label of every link kept.

    Args:
        text (str): The captured console output.

    Returns:
        str: The output without escape sequences.
    """
    return re.sub(pattern=CONSOLE_CODE_PATTERN, repl="", string=text)


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


def write_test_thresholds(experiment_folder: Path, evaluation_folder_name: str, threshold: float,
                          iteration_folder_name: str) -> None:
    """
    Writes the class_thresholds.json of the evaluation of one iteration of a frame selection run, in the format of the
    frame selection trainer, so that build_decoding_settings finds it in the given evaluation folder.

    Args:
        experiment_folder (Path): The run folder.
        evaluation_folder_name (str): 'Evaluation' (training) or 'Checkpoint_Evaluation' (evaluate_checkpoint.py).
        threshold (float): The threshold of the wanted class 'A', the class 'C' never reaching the target precision.
        iteration_folder_name (str): The folder of the evaluated iteration, named after its checkpoint folder.
    """
    iteration_folder = experiment_folder / evaluation_folder_name / iteration_folder_name
    iteration_folder.mkdir(parents=True)
    (iteration_folder / THRESHOLDS_FILE_NAME).write_text(data=json.dumps(obj={
        "target_precision": 0.9, "decoding": "hungarian", "unselected_class": "B",
        "classes": {"A": {"threshold": threshold, "recall": 0.8}, "C": None}}))


def build_test_decoding_settings(experiment_folder: Path) -> Dict[str, Any]:
    """
    Builds the decoding settings of the export of iteration 500 of a frame selection run with classes A, B and C, B
    being the unselected class, as export_model.py does for a frame selector.

    Args:
        experiment_folder (Path): The run folder, holding the evaluation folders written by write_test_thresholds.

    Returns:
        Dict[str, Any]: The decoding settings of the run.
    """
    model_configuration = {"Selection": {"decoding": "hungarian", "frames_per_class": 1},
                           "Data": {"label_dictionary": {"A": 0, "B": 1, "C": 2}}}
    return build_decoding_settings(experiment_configuration={"evaluation": {"top_k": 2}},
                                   model_configuration=model_configuration, unselected_class_index=1,
                                   checkpoint_path=experiment_folder / "Weights" / "Iteration_500" / "model_500.pt",
                                   experiment_folder=experiment_folder)


@pytest.mark.parametrize(argnames="evaluation_folder_name",
                         argvalues=[EVALUATION_FOLDER_NAME, CHECKPOINT_EVALUATION_FOLDER_NAME])
def test_decoding_settings_read_the_thresholds_of_either_evaluation(tmp_path: Path,
                                                                    evaluation_folder_name: str) -> None:
    """
    Verifies that the export of a frame selector takes the thresholds of the evaluation of the exported checkpoint from
    the evaluation folder of the training or from the one of scripts/evaluation/evaluate_checkpoint.py, its class names
    replacing the class indices.

    Args:
        tmp_path (Path): The pytest temporary directory, used as the run folder.
        evaluation_folder_name (str): The evaluation folder holding the thresholds.
    """
    write_test_thresholds(experiment_folder=tmp_path, evaluation_folder_name=evaluation_folder_name, threshold=0.7,
                          iteration_folder_name="Iteration_500")
    decoding_settings = build_test_decoding_settings(experiment_folder=tmp_path)
    assert decoding_settings["class_names"] == ["A", "C"]
    assert decoding_settings["class_thresholds"] == [0.7, None]


def test_decoding_settings_take_the_checkpoint_evaluation_of_two(tmp_path: Path,
                                                                  capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the export of a frame selector takes the thresholds of the standalone evaluation
    (Checkpoint_Evaluation) when both evaluation folders hold thresholds for the exported checkpoint, and names both
    files and the one used in a yellow console line, instead of stopping the export.

    Args:
        tmp_path (Path): The pytest temporary directory, used as the run folder.
        capsys (pytest.CaptureFixture): Captures the console line naming the two files.
    """
    write_test_thresholds(experiment_folder=tmp_path, evaluation_folder_name=EVALUATION_FOLDER_NAME, threshold=0.7,
                          iteration_folder_name="Iteration_500")
    write_test_thresholds(experiment_folder=tmp_path, evaluation_folder_name=CHECKPOINT_EVALUATION_FOLDER_NAME,
                          threshold=0.6, iteration_folder_name="Iteration_500")
    decoding_settings = build_test_decoding_settings(experiment_folder=tmp_path)
    assert decoding_settings["class_thresholds"] == [0.6, None]
    console_output = strip_console_codes(text=capsys.readouterr().out)
    assert (f"Two {THRESHOLDS_FILE_NAME} for the exported checkpoint, the one of Checkpoint_Evaluation is used : "
            f"Checkpoint_Evaluation/Iteration_500/{THRESHOLDS_FILE_NAME} (not Evaluation/Iteration_500/"
            f"{THRESHOLDS_FILE_NAME})") in console_output


def test_decoding_settings_without_evaluation_flag_nothing(tmp_path: Path) -> None:
    """
    Verifies that a frame selector exported from a checkpoint evaluated by neither the training nor the evaluation
    script gets a zero threshold for every wanted class, named by its class index, so that no chosen frame is flagged.

    Args:
        tmp_path (Path): The pytest temporary directory, used as the run folder.
    """
    decoding_settings = build_test_decoding_settings(experiment_folder=tmp_path)
    assert decoding_settings["class_names"] == ["0", "2"]
    assert decoding_settings["class_thresholds"] == [0.0, 0.0]


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="The export runs the model on the GPU.")
@pytest.mark.parametrize(argnames="checkpoint_evaluation_readable", argvalues=[True, False])
def test_frame_selector_export_installs_its_decoder_or_nothing(tmp_path: Path,
                                                               checkpoint_evaluation_readable: bool) -> None:
    """
    Verifies the export of a tiny frame selector whose exported checkpoint was evaluated by its training and by the
    evaluation script: it completes, installing the networks, the decoder with the thresholds of Checkpoint_Evaluation
    and the ensembles; and when the only thresholds it finds cannot be used (a file without its 'decoding'), it stops
    before anything is installed, instead of leaving the networks served without their decoder and ensembles.

    Args:
        tmp_path (Path): The pytest temporary directory, holding the run and the repository.
        checkpoint_evaluation_readable (bool): False to write the thresholds of Checkpoint_Evaluation alone, without
            their 'decoding' entry.
    """
    experiment_folder, repository = tmp_path / "SEL_TEST_MC3", tmp_path / "served_models"
    test_input_contract.write_trained_model(experiment_folder=experiment_folder, task="frame_selection",
                                            keep_ratio=True, tiling_configuration=None)
    checkpoint_path = get_checkpoint_file(weights_directory=experiment_folder / "Weights",
                                          iteration=test_input_contract.EXPORTED_ITERATION)
    iteration_folder_name = checkpoint_path.parent.name

    # Class 1 marked as the unselected class in the saved weights, as the frame selection trainer sets it
    checkpoint = torch.load(f=checkpoint_path, weights_only=True)
    checkpoint["model_state"]["wanted_class_mask"][1] = 0.0
    torch.save(obj=checkpoint, f=checkpoint_path)
    evaluated_folders = [(EVALUATION_FOLDER_NAME, 0.7), (CHECKPOINT_EVALUATION_FOLDER_NAME, 0.6)]
    for evaluation_folder_name, threshold in evaluated_folders[0 if checkpoint_evaluation_readable else 1:]:
        write_test_thresholds(experiment_folder=experiment_folder, evaluation_folder_name=evaluation_folder_name,
                              threshold=threshold, iteration_folder_name=iteration_folder_name)
    if not checkpoint_evaluation_readable:
        thresholds_path = experiment_folder / CHECKPOINT_EVALUATION_FOLDER_NAME / iteration_folder_name / (
            THRESHOLDS_FILE_NAME)
        thresholds = json.loads(s=thresholds_path.read_text())
        del thresholds["decoding"]
        thresholds_path.write_text(data=json.dumps(obj=thresholds))

        # The unreadable thresholds stop the export with the repository left empty
        with pytest.raises(expected_exception=KeyError, match="decoding"):
            test_input_contract.export_test_run(experiment_folder=experiment_folder, repository=repository,
                                                formats=["onnx"])
        assert read_repository_files(repository=repository) == {}
        return

    # Both evaluations: the networks, the decoder with the thresholds of Checkpoint_Evaluation and the ensembles
    test_input_contract.export_test_run(experiment_folder=experiment_folder, repository=repository,
                                        formats=["torchscript", "onnx"])
    installed_models = sorted(model_folder.name for model_folder in repository.iterdir())
    assert installed_models == ["frame_selector_decoder", "frame_selector_onnx", "frame_selector_onnx_ensemble",
                                "frame_selector_torchscript", "frame_selector_torchscript_ensemble"]
    decoding_settings = json.loads(s=(repository / "frame_selector_decoder" / "1" /
                                      "decoding_settings.json").read_text())
    assert decoding_settings["class_thresholds"] == [0.6, None]
    assert (repository / "frame_selector_onnx_ensemble" / "1").is_dir()
    assert not list(tmp_path.glob(pattern=".export_staging_*"))
