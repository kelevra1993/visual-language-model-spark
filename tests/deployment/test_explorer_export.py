"""
Tests of vision_training/deployment/explorer_export.py, the export launched from the temperature explorer page: the
run, iteration and weights read from an evaluation folder, the export folder of every checkpoint, weights and
temperature, the exports of a checkpoint listed without the unfinished ones, the settings of the page refused with their
reason, and on the GPU the export of a tiny classifier at two saved temperatures, each in its own folder with its
thresholds, an export of the same temperature replacing the earlier one.
"""

import json
import pytest
import torch

from pathlib import Path
from typing import Any, Dict

from tests.deployment.test_input_contract import EXPORTED_ITERATION, write_trained_model
from vision_training.deployment.explorer_export import (EXPORT_LOG_FILE_NAME, EXPORT_RECORD_FILE_NAME,
                                                        build_export_folder, check_export_settings,
                                                        describe_evaluation_folder, list_checkpoint_exports,
                                                        prepare_export_job, run_export_job)
from vision_training.deployment.export_model import hide_known_export_warnings
from vision_training.evaluation.classification.temperature_explorer import SAVED_THRESHOLDS_FILE_NAME

# Run of the tests
RUN_NAME = "CLS_TEST_MC3"


@pytest.mark.parametrize(argnames=("evaluation_subfolder", "run_subfolder", "iteration", "use_averaged_weights"),
                         argvalues=[(f"{RUN_NAME}/Evaluation/Iteration_800", RUN_NAME, 800, False),
                                    (f"{RUN_NAME}/Evaluation/Iteration_800/EMA", RUN_NAME, 800, True),
                                    (f"evaluations/{RUN_NAME}/Evaluation/Iteration_3500/EMA", RUN_NAME, 3500, True),
                                    (f"group/evaluations/{RUN_NAME}/Evaluation/Iteration_5", f"group/{RUN_NAME}", 5,
                                     False)])
def test_evaluation_folder_gives_its_run_iteration_and_weights(tmp_path: Path, evaluation_subfolder: str,
                                                               run_subfolder: str, iteration: int,
                                                               use_averaged_weights: bool) -> None:
    """
    Verifies that the evaluation of the training and that of evaluate_checkpoint.py (in the evaluations folder next to
    the run) both give the run folder holding the weights, the iteration and the weights of the checkpoint.
    """
    assert describe_evaluation_folder(evaluation_folder=tmp_path / evaluation_subfolder) == {
        "run_folder": tmp_path / run_subfolder, "iteration": iteration, "use_averaged_weights": use_averaged_weights}


@pytest.mark.parametrize(argnames="evaluation_subfolder", argvalues=[f"{RUN_NAME}/Evaluation/Iteration_last",
                                                                     f"{RUN_NAME}/Results/Iteration_800",
                                                                     f"{RUN_NAME}/Evaluation/Iteration_800/Errors"])
def test_folders_other_than_an_evaluation_are_refused(tmp_path: Path, evaluation_subfolder: str) -> None:
    """
    Verifies that a folder not laid out as the evaluation of an iteration is refused with the expected layout.
    """
    with pytest.raises(expected_exception=ValueError, match="Expected an evaluation folder"):
        describe_evaluation_folder(evaluation_folder=tmp_path / evaluation_subfolder)


def test_exports_of_a_checkpoint_get_their_own_folders(tmp_path: Path) -> None:
    """
    Verifies the export folder of every weights and temperature, and that the exports of a checkpoint are listed
    without those of the other weights or iteration and without an unfinished export.
    """
    run_folder = tmp_path / RUN_NAME
    assert build_export_folder(run_folder=run_folder, iteration=800, use_averaged_weights=True,
                               logit_temperature=1.83) == run_folder / "Exports" / "Iteration_800_EMA_T1.83"
    trained_folder = build_export_folder(run_folder=run_folder, iteration=800, use_averaged_weights=False,
                                         logit_temperature=2.0)
    assert trained_folder == run_folder / "Exports" / "Iteration_800_T2"

    # Exports of both weights, of another iteration and one unfinished, only the finished trained one listed
    for folder_name in ["Iteration_800_T2", "Iteration_800_EMA_T1.83", "Iteration_8000_T2", "Iteration_800_T3.partial"]:
        (run_folder / "Exports" / folder_name).mkdir(parents=True)
    (trained_folder / EXPORT_RECORD_FILE_NAME).write_text(data=json.dumps(obj={"logit_temperature": 2.0}))
    assert list_checkpoint_exports(run_folder=run_folder, iteration=800, use_averaged_weights=False) == [
        {"folder": trained_folder, "record": {"logit_temperature": 2.0}}]
    assert [export["folder"].name for export in list_checkpoint_exports(
        run_folder=run_folder, iteration=800, use_averaged_weights=True)] == ["Iteration_800_EMA_T1.83"]


@pytest.mark.parametrize(argnames=("settings", "expected_message"), argvalues=[
    ({"formats": []}, "Expected distinct export formats"),
    ({"formats": ["onnx", "onnx"]}, "Expected distinct export formats"),
    ({"formats": ["tensorflow"]}, "Expected distinct export formats"),
    ({"model_version": 0}, "Expected model_version"),
    ({"maximum_batch_size": 1.5}, "Expected maximum_batch_size"),
    ({"maximum_batch_size": True}, "Expected maximum_batch_size"),
    ({"tensorrt_precision": "INT8"}, "Expected a TensorRT precision")])
def test_export_settings_of_the_page_are_checked(settings: Dict[str, Any], expected_message: str) -> None:
    """
    Verifies that every wrong setting of the export form is refused with its reason before any export starts.
    """
    with pytest.raises(expected_exception=ValueError, match=expected_message):
        check_export_settings(**{"formats": ["torchscript"], "model_version": 1, "maximum_batch_size": 4,
                                 "tensorrt_precision": None, **settings})


def write_saved_evaluation(evaluation_folder: Path, logit_temperature: float) -> Dict[str, Any]:
    """
    Writes the class_thresholds.json of the evaluation of the test classifier of three classes and the thresholds saved
    on its page at a temperature.

    Args:
        evaluation_folder (Path): The evaluation folder of the checkpoint, created.
        logit_temperature (float): The saved temperature.

    Returns:
        Dict[str, Any]: The saved thresholds.
    """
    evaluation_folder.mkdir(parents=True, exist_ok=True)
    class_names = [f"class_{class_index}" for class_index in range(3)]
    for file_name, temperature in [("class_thresholds.json", 1.0), (SAVED_THRESHOLDS_FILE_NAME, logit_temperature)]:
        thresholds = {"logit_temperature": temperature, "target_precision": 0.99,
                      "classes": {class_name: {"threshold": 0.9 / temperature, "recall": 0.5}
                                  for class_name in class_names}}
        (evaluation_folder / file_name).write_text(data=json.dumps(obj=thresholds))
    return json.loads(s=(evaluation_folder / SAVED_THRESHOLDS_FILE_NAME).read_text())


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="The export runs the model on the GPU.")
def test_exports_at_two_temperatures_keep_their_own_folders(tmp_path: Path) -> None:
    """
    Verifies that a tiny classifier exported from its evaluation at a saved temperature of 2.5 is written to
    Exports/Iteration_1_T2.5 with its record, its console output and the saved thresholds next to every model, that a
    temperature of 3 saved afterwards goes to its own folder, the first export staying, and that exporting 2.5 again
    replaces its folder.
    """
    hide_known_export_warnings()
    run_folder = tmp_path / RUN_NAME
    write_trained_model(experiment_folder=run_folder, task="classification", keep_ratio=False,
                        tiling_configuration=None)
    evaluation_folder = tmp_path / "evaluations" / RUN_NAME / "Evaluation" / f"Iteration_{EXPORTED_ITERATION}"
    export_settings = {"evaluation_folder": evaluation_folder, "formats": ["onnx"], "model_version": 1,
                       "maximum_batch_size": 4, "tensorrt_precision": None}

    # Export at every saved temperature, the console output written where the serve script writes it
    exported_records = {}
    for logit_temperature in [2.5, 3.0, 2.5]:
        saved_thresholds = write_saved_evaluation(evaluation_folder=evaluation_folder,
                                                  logit_temperature=logit_temperature)
        job = prepare_export_job(**export_settings)
        job["log_path"].write_text(data="console output of the export\n")
        export_folder = run_export_job(job_path=job["job_path"])
        assert export_folder == job["export_folder"] == build_export_folder(
            run_folder=run_folder, iteration=EXPORTED_ITERATION, use_averaged_weights=False,
            logit_temperature=logit_temperature)
        assert not job["work_folder"].exists()
        assert sorted(path.name for path in export_folder.iterdir()) == [EXPORT_LOG_FILE_NAME, EXPORT_RECORD_FILE_NAME,
                                                                        "image_classifier_onnx"]
        deployed_thresholds = export_folder / "image_classifier_onnx" / "1" / "class_thresholds.json"
        assert json.loads(s=deployed_thresholds.read_text()) == saved_thresholds
        record = json.loads(s=(export_folder / EXPORT_RECORD_FILE_NAME).read_text())
        assert (record["logit_temperature"], record["iteration"], record["formats"]) == (
            logit_temperature, EXPORTED_ITERATION, ["onnx"])
        exported_records[export_folder.name] = record

    # Both temperatures listed, the export of 2.5 being the second one
    listed_exports = list_checkpoint_exports(run_folder=run_folder, iteration=EXPORTED_ITERATION,
                                             use_averaged_weights=False)
    assert sorted(export["folder"].name for export in listed_exports) == ["Iteration_1_T2.5", "Iteration_1_T3"]
    assert {export["folder"].name: export["record"] for export in listed_exports} == exported_records
