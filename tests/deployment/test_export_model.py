"""
Tests of find_saved_thresholds_path, check_saved_thresholds_location and read_saved_temperature of
vision_training/deployment/export_model.py, used by the export with use_saved_temperature: the thresholds saved from
the temperature explorer of the exported checkpoint are found in its evaluation folder in the Checkpoint_Evaluation
folder of scripts/evaluation/evaluate_checkpoint.py (its 'EMA' subfolder for the averaged weights), the export stopping
when the Evaluation folder of the training holds them too, thresholds given by the export button of the page are
accepted only from an evaluation of the exported checkpoint and weights in the run folder, and their temperature is
read after checking them against the classes of the exported model. An export failing its check against the eager
model leaves the served models of the repository unchanged, a checked one replacing them. A misspelled format,
TensorRT or TorchScript precision stops export_experiment before anything is read or written. The random batch check
of an FP16 TorchScript export loosens the difference tolerances only, and a run exported with an FP16 TorchScript file
installs an FP16 network with FP32 outputs close to those of the FP32 export.
"""

import json
import re
import pytest
import torch

from pathlib import Path
from typing import Any, Dict, List

from tests.deployment import test_input_contract
from vision_training.deployment import export_model
from vision_training.deployment.export_model import (HALF_PRECISION_TOLERANCE_FACTOR, STAGING_FOLDER_PREFIX,
                                                     check_saved_thresholds_location, export_experiment,
                                                     find_saved_thresholds_path, read_saved_temperature,
                                                     scale_half_precision_tolerances)
from vision_training.evaluation.classification.temperature_explorer import SAVED_THRESHOLDS_FILE_NAME
from vision_training.trainers.checkpointing import get_checkpoint_file

# Exported run and checkpoint of the tests
RUN_NAME = "CLS_TEST_MC3"
EXPORTED_ITERATION = 6

# Mismatch reported by the comparison of get_mismatching_specification whatever the outputs, and the specification
# of the task it stands in for
FORCED_MISMATCH = "class_probabilities differ by 1.0 (forced by the test)"
ORIGINAL_GET_EXPORT_SPECIFICATION = export_model.get_export_specification


def report_forced_mismatch(reference: Any, candidate: Any, **comparison_tolerances: Any) -> List[str]:
    """
    Stand-in for the comparison of an export specification, reporting a mismatch whatever the outputs, as the check of
    a wrong export does.

    Args:
        reference (Any): The eager outputs, ignored.
        candidate (Any): The outputs of the exported model, ignored.
        **comparison_tolerances (Any): The tolerances of the specification, ignored.

    Returns:
        List[str]: The forced mismatch.
    """
    return [FORCED_MISMATCH]


def get_mismatching_specification(task: str, inference_model: torch.nn.Module) -> Dict[str, Any]:
    """
    Stand-in for get_export_specification giving the export specification of the task with report_forced_mismatch as
    its comparison, so that every exported format fails its check.

    Args:
        task (str): The task of the exported run.
        inference_model (torch.nn.Module): The inference model of the run.

    Returns:
        Dict[str, Any]: The specification of the task, its comparison failing.
    """
    return (ORIGINAL_GET_EXPORT_SPECIFICATION(task=task, inference_model=inference_model) |
            {"compare_outputs": report_forced_mismatch})


def read_repository_files(repository: Path) -> Dict[str, bytes]:
    """
    Reads every file of a model repository, so that a test tells which files an export changed.

    Args:
        repository (Path): The model repository.

    Returns:
        Dict[str, bytes]: The content of every file, keyed by its path relative to the repository.
    """
    return {path.relative_to(repository).as_posix(): path.read_bytes() for path in sorted(repository.rglob(pattern="*"))
            if path.is_file()}


def build_thresholds(logit_temperature: float, number_classes: int) -> Dict[str, Any]:
    """
    Builds thresholds in the format of class_thresholds.json, the file of the evaluation and the saved file sharing it.

    Args:
        logit_temperature (float): The temperature of the thresholds.
        number_classes (int): The number of classes of the thresholds.

    Returns:
        Dict[str, Any]: The thresholds.
    """
    return {"logit_temperature": logit_temperature, "target_precision": 0.99,
            "classes": {f"class_{class_index}": {"threshold": 0.8, "recall": 0.5}
                        for class_index in range(number_classes)}}


def write_evaluation(evaluation_folder: Path, saved_thresholds: Dict[str, Any], number_classes: int) -> Path:
    """
    Writes the class_thresholds.json of an evaluation, as the classification evaluation writes it, and the thresholds
    saved from its temperature explorer next to it.

    Args:
        evaluation_folder (Path): The evaluation folder of the checkpoint, created.
        saved_thresholds (Dict[str, Any]): The saved thresholds.
        number_classes (int): The number of classes of class_thresholds.json.

    Returns:
        Path: The saved thresholds file.
    """
    evaluation_folder.mkdir(parents=True, exist_ok=True)
    (evaluation_folder / "class_thresholds.json").write_text(
        data=json.dumps(obj=build_thresholds(logit_temperature=1.0, number_classes=number_classes)))
    saved_path = evaluation_folder / SAVED_THRESHOLDS_FILE_NAME
    saved_path.write_text(data=json.dumps(obj=saved_thresholds))
    return saved_path


def get_checkpoint_path(experiment_folder: Path) -> Path:
    """
    Gives the checkpoint of the exported iteration of the test run, the export naming its evaluation folder after the
    folder of the checkpoint.

    Args:
        experiment_folder (Path): The exported run folder.

    Returns:
        Path: The checkpoint of the exported iteration.
    """
    return experiment_folder / "Weights" / f"Iteration_{EXPORTED_ITERATION}" / "model_000006.pt"


@pytest.mark.parametrize(argnames=("evaluation_subfolder", "use_averaged_weights"),
                         argvalues=[(f"{RUN_NAME}/Checkpoint_Evaluation/Iteration_6/EMA", True)])
def test_saved_temperature_is_read_from_the_evaluation_of_the_checkpoint(
        tmp_path: Path, evaluation_subfolder: str, use_averaged_weights: bool) -> None:
    """
    Verifies that the saved thresholds are found in the evaluation of the exported iteration and averaged weights in
    the Checkpoint_Evaluation folder of the run, and that their temperature is read.

    Args:
        tmp_path (Path): The pytest temporary folder holding the run and its evaluations.
        evaluation_subfolder (str): The evaluation folder, relative to the temporary folder.
        use_averaged_weights (bool): Whether the moving average of the weights is expected.
    """
    saved_path = write_evaluation(evaluation_folder=tmp_path / evaluation_subfolder,
                                  saved_thresholds=build_thresholds(logit_temperature=2.5, number_classes=3),
                                  number_classes=3)
    experiment_folder = tmp_path / RUN_NAME
    found_path = find_saved_thresholds_path(experiment_folder=experiment_folder,
                                            checkpoint_path=get_checkpoint_path(experiment_folder=experiment_folder),
                                            use_averaged_weights=use_averaged_weights)
    assert found_path == saved_path
    assert read_saved_temperature(saved_thresholds_path=found_path, number_classes=3) == 2.5


def test_thresholds_saved_in_both_folders_are_refused(tmp_path: Path) -> None:
    """
    Verifies that thresholds saved for the exported checkpoint in both the Evaluation and the Checkpoint_Evaluation
    folders of the run stop the export, so that the deployed temperature is never picked silently between two saves.

    Args:
        tmp_path (Path): The pytest temporary folder holding the run and its evaluations.
    """
    for evaluation_subfolder in [f"{RUN_NAME}/Evaluation/Iteration_6",
                                 f"{RUN_NAME}/Checkpoint_Evaluation/Iteration_6"]:
        write_evaluation(evaluation_folder=tmp_path / evaluation_subfolder,
                         saved_thresholds=build_thresholds(logit_temperature=2.5, number_classes=3), number_classes=3)
    experiment_folder = tmp_path / RUN_NAME
    with pytest.raises(expected_exception=ValueError, match="Two saved_temperature_thresholds.json"):
        find_saved_thresholds_path(experiment_folder=experiment_folder,
                                   checkpoint_path=get_checkpoint_path(experiment_folder=experiment_folder),
                                   use_averaged_weights=False)


@pytest.mark.parametrize(argnames=("saved_thresholds", "evaluated_classes", "exported_classes", "expected_message"),
                         argvalues=[(build_thresholds(logit_temperature=2.5, number_classes=2), 2, 3,
                                     "the exported classifier predicts 3")])
def test_saved_thresholds_of_another_evaluation_or_model_are_refused(
        tmp_path: Path, saved_thresholds: Dict[str, Any], evaluated_classes: int, exported_classes: int,
        expected_message: str) -> None:
    """
    Verifies that saved thresholds holding another number of classes than the exported classifier stop the export, the
    error naming the mismatch.

    Args:
        tmp_path (Path): The pytest temporary folder holding the run and its evaluations.
        saved_thresholds (Dict[str, Any]): The content of the saved thresholds file.
        evaluated_classes (int): The number of classes of the evaluated model.
        exported_classes (int): The number of classes of the exported model.
        expected_message (str): The start of the expected error message.
    """
    saved_path = write_evaluation(evaluation_folder=tmp_path / RUN_NAME / "Evaluation" / "Iteration_6",
                                  saved_thresholds=saved_thresholds, number_classes=evaluated_classes)
    with pytest.raises(expected_exception=ValueError, match=expected_message):
        read_saved_temperature(saved_thresholds_path=saved_path, number_classes=exported_classes)


@pytest.mark.parametrize(argnames=("saved_subfolder", "use_averaged_weights", "accepted"), argvalues=[
    (f"{RUN_NAME}/Checkpoint_Evaluation/Iteration_6/EMA", True, True),
    (f"{RUN_NAME}/Evaluation/Iteration_6", False, True),
    (f"{RUN_NAME}/Evaluation/Iteration_7", False, False),
    (f"{RUN_NAME}/Checkpoint_Evaluation/Iteration_6/EMA", False, False),
    (f"{RUN_NAME}/Evaluation/Iteration_6", True, False),
    ("CLS_OTHER_MC3/Checkpoint_Evaluation/Iteration_6", False, False),
    (f"evaluations/{RUN_NAME}/Evaluation/Iteration_6", False, False),
    (f"{RUN_NAME}/Checkpoint_Evaluation/Evaluation/Iteration_6", False, False)])
def test_given_saved_thresholds_must_be_those_of_the_exported_checkpoint(tmp_path: Path, saved_subfolder: str,
                                                                         use_averaged_weights: bool,
                                                                         accepted: bool) -> None:
    """
    Verifies that the thresholds given by the export button of the page are accepted only from the evaluation of the
    exported run, iteration and weights, in its Evaluation or Checkpoint_Evaluation folder, and refused from the
    evaluation of another iteration, of the other weights, of another run or from the former evaluations folder next to
    the run.

    Args:
        tmp_path (Path): The pytest temporary folder holding the run and its evaluations.
        saved_subfolder (str): The evaluation folder the thresholds are saved in.
        use_averaged_weights (bool): Whether the moving average of the weights is expected.
        accepted (bool): Whether the export accepts the saved thresholds.
    """
    experiment_folder = tmp_path / RUN_NAME
    saved_path = tmp_path / saved_subfolder / SAVED_THRESHOLDS_FILE_NAME
    location_settings = {"saved_thresholds_path": saved_path, "experiment_folder": experiment_folder,
                         "checkpoint_path": get_checkpoint_path(experiment_folder=experiment_folder),
                         "use_averaged_weights": use_averaged_weights}
    if accepted:
        check_saved_thresholds_location(**location_settings)
    else:
        with pytest.raises(expected_exception=ValueError, match="for the exported checkpoint"):
            check_saved_thresholds_location(**location_settings)


@pytest.mark.parametrize(argnames=("export_settings", "expected_message"), argvalues=[
    ({"formats": ["tensorrt"]}, "Expected distinct export formats"),
    ({"formats": ["TorchScript", "onnx"]}, "Expected distinct export formats"),
    ({"formats": []}, "Expected distinct export formats"),
    ({"formats": ["onnx", "onnx"]}, "Expected distinct export formats"),
    ({"tensorrt_precision": "fp16"}, "Expected a TensorRT precision"),
    ({"tensorrt_precision": "INT8"}, "Expected a TensorRT precision"),
    ({"torchscript_precision": "fp16"}, "Expected a TorchScript precision"),
    ({"torchscript_precision": "BF16"}, "Expected a TorchScript precision")])
def test_unknown_export_settings_stop_the_export_before_anything_is_read(tmp_path: Path,
                                                                        export_settings: Dict[str, Any],
                                                                        expected_message: str) -> None:
    """
    Verifies that export_experiment refuses a misspelled format, which would otherwise be exported as an ONNX model
    served without its TensorRT setting, and a TensorRT precision Triton does not know, before it reads the run
    (whose folder does not even exist here) or creates the model repository.

    Args:
        tmp_path (Path): The pytest temporary directory.
        export_settings (Dict[str, Any]): The settings replacing the valid ones of the export.
        expected_message (str): The pattern of the expected error.
    """
    repository = tmp_path / "served_models"
    with pytest.raises(expected_exception=ValueError, match=expected_message):
        export_experiment(**{"experiment_folder": tmp_path / RUN_NAME, "iteration": EXPORTED_ITERATION,
                             "use_averaged_weights": False, "formats": ["torchscript", "onnx"],
                             "repository": repository, "model_version": 1, "maximum_batch_size": 4,
                             "tensorrt_precision": "FP16", "torchscript_precision": "FP32",
                             "use_saved_temperature": False, "saved_thresholds_path": None,
                             "test_split_parity": False, **export_settings})
    assert not repository.exists()


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="The export runs the model on the GPU.")
def test_failed_export_check_leaves_the_served_models_unchanged(tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that an export of new weights failing its check against the eager model raises and leaves every file of
    the served models unchanged, both formats included, without any staging folder left, and that the checked export
    of the same weights then replaces the version folders of both formats while keeping another version.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture making the comparison of the export fail.
    """
    experiment_folder, repository = tmp_path / "CLS_TEST_MC3", tmp_path / "served_models"
    test_input_contract.write_trained_model(experiment_folder=experiment_folder, task="classification",
                                            keep_ratio=True, tiling_configuration=None)
    test_input_contract.export_test_run(experiment_folder=experiment_folder, repository=repository,
                                        formats=["torchscript", "onnx"])

    # Another version of the served classifier, which an export of version 1 must keep
    other_version_path = repository / "image_classifier_torchscript" / "2" / "model.pt"
    other_version_path.parent.mkdir()
    other_version_path.write_bytes(data=b"version 2")
    served_files = read_repository_files(repository=repository)

    # New weights for the run, so that its next export writes other model files
    checkpoint_path = get_checkpoint_file(weights_directory=experiment_folder / "Weights",
                                          iteration=test_input_contract.EXPORTED_ITERATION)
    checkpoint = torch.load(f=checkpoint_path, weights_only=True)
    checkpoint["model_state"] = {name: tensor + 0.5 if tensor.is_floating_point() else tensor
                                 for name, tensor in checkpoint["model_state"].items()}
    torch.save(obj=checkpoint, f=checkpoint_path)

    # An export failing its check raises, the served files staying as they were
    monkeypatch.setattr(target=export_model, name="get_export_specification", value=get_mismatching_specification)
    with pytest.raises(expected_exception=RuntimeError, match=re.escape(pattern=FORCED_MISMATCH)):
        test_input_contract.export_test_run(experiment_folder=experiment_folder, repository=repository,
                                            formats=["torchscript", "onnx"])
    assert read_repository_files(repository=repository) == served_files
    assert not list(tmp_path.glob(pattern=f"{STAGING_FOLDER_PREFIX}*"))

    # The checked export replaces version 1 of both formats and keeps version 2
    monkeypatch.undo()
    test_input_contract.export_test_run(experiment_folder=experiment_folder, repository=repository,
                                        formats=["torchscript", "onnx"])
    exported_files = read_repository_files(repository=repository)
    assert exported_files.keys() == served_files.keys()
    for model_file in ["image_classifier_torchscript/1/model.pt", "image_classifier_onnx/1/model.onnx"]:
        assert exported_files[model_file] != served_files[model_file]
    assert exported_files["image_classifier_torchscript/2/model.pt"] == b"version 2"
    assert not list(tmp_path.glob(pattern=f"{STAGING_FOLDER_PREFIX}*"))


def test_half_precision_tolerances_loosen_the_differences_only() -> None:
    """
    Verifies that the tolerances of the random batch check of an FP16 TorchScript export multiply every difference
    tolerance of the export specification and keep the smallest share of equal outputs.
    """
    keypoint_tolerances = {"position_tolerance": 1e-2, "score_tolerance": 1e-4, "tag_tolerance": 1e-3,
                           "minimum_equal_share": 0.95}
    assert scale_half_precision_tolerances(comparison_tolerances=keypoint_tolerances) == {
        "position_tolerance": 1e-2 * HALF_PRECISION_TOLERANCE_FACTOR,
        "score_tolerance": 1e-4 * HALF_PRECISION_TOLERANCE_FACTOR,
        "tag_tolerance": 1e-3 * HALF_PRECISION_TOLERANCE_FACTOR, "minimum_equal_share": 0.95}


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="The export runs the model on the GPU.")
def test_half_precision_torchscript_export_installs_an_fp16_network(tmp_path: Path) -> None:
    """
    Verifies that a run exported with torchscript_precision 'FP16' installs 'model_fp16.pt', named by its
    config.pbtxt and alone in its version folder, whose convolution kernels are FP16 and whose normalization statistics
    stay FP32, taking FP32 images and returning FP32 probabilities within 1e-2 of those of the FP32 export of the same
    checkpoint, written to 'model.pt'.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    experiment_folder = tmp_path / RUN_NAME
    test_input_contract.write_trained_model(experiment_folder=experiment_folder, task="classification",
                                            keep_ratio=True, tiling_configuration=None)

    # The same checkpoint exported with both TorchScript precisions, each into its own repository
    traced_models = {}
    for torchscript_precision, file_name in [("FP32", "model.pt"), ("FP16", "model_fp16.pt")]:
        repository = tmp_path / f"served_models_{torchscript_precision.lower()}"
        export_experiment(experiment_folder=experiment_folder, iteration=test_input_contract.EXPORTED_ITERATION,
                          use_averaged_weights=False, formats=["torchscript"], repository=repository, model_version=1,
                          maximum_batch_size=test_input_contract.TEST_MAXIMUM_BATCH_SIZE,
                          tensorrt_precision=None, torchscript_precision=torchscript_precision,
                          use_saved_temperature=False, saved_thresholds_path=None, test_split_parity=False)
        model_directory = repository / "image_classifier_torchscript"
        assert sorted(path.name for path in (model_directory / "1").glob(pattern="*.pt")) == [file_name]
        assert (('default_model_filename: "model_fp16.pt"' in (model_directory / "config.pbtxt").read_text()) ==
                (torchscript_precision == "FP16"))
        traced_models[torchscript_precision] = torch.jit.load(str(model_directory / "1" / file_name),
                                                              map_location="cuda")

    # FP16 convolution kernels (the 4-dimensional weights), FP32 normalization statistics
    state_dictionary = traced_models["FP16"].state_dict()
    assert {tensor.dtype for tensor in state_dictionary.values() if tensor.ndim == 4} == {torch.float16}
    assert {tensor.dtype for name, tensor in state_dictionary.items() if name.endswith("running_mean")} == {
        torch.float32}

    # FP32 probabilities of the same FP32 images, close to those of the FP32 export
    images = torch.rand(size=(3, 3, 32, 32), generator=torch.Generator().manual_seed(0)).cuda() * 255
    with torch.no_grad():
        probabilities = {precision: traced_model(images=images) for precision, traced_model in traced_models.items()}
    assert probabilities["FP16"].dtype == torch.float32
    assert torch.allclose(input=probabilities["FP16"], other=probabilities["FP32"], atol=1e-2)
