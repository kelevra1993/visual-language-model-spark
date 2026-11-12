"""
Tests of scripts/evaluation/serve_temperature_explorer.py, the local server of the temperature explorers of the
experiments folders: the index lists the evaluations holding a page, every page is served with the current template
and names its evaluation, a typed temperature is described from the class logits, the thresholds saved from a page
with logits are found by the server for every class whatever was posted, those of a page without logits are checked
against its class_thresholds.json, the status gives the saved temperature and the export folder, and an export without
saved thresholds or without its run is refused with the reason, nothing being written by a refused request.
"""

import json
import threading
import urllib.error
import urllib.request

from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple
from urllib.parse import quote

import pytest

from scripts.evaluation.serve_temperature_explorer import TemperatureExplorerServer
from tests.evaluation.classification.test_temperature_explorer import (EXPLORER_CLASS_NAMES, list_file_thresholds,
                                                                       write_explorer_evaluation)
from vision_training.evaluation.classification.temperature_explorer import (EXPLORER_DATA_PATTERN,
                                                                            SAVED_THRESHOLDS_FILE_NAME,
                                                                            compute_temperature_thresholds,
                                                                            describe_explorer_temperature)

# Evaluations of the tests: one with its class logits, evaluated by the training of its run, one without, evaluated by
# evaluate_checkpoint.py for the averaged weights, in the Checkpoint_Evaluation folder of a run without configuration
LIVE_EVALUATION = "experiments/CLS_LIVE_MC3/Evaluation/Iteration_6"
GRID_EVALUATION = "experiments/CLS_GRID_MC3/Checkpoint_Evaluation/Iteration_3/EMA"


@pytest.fixture
def served_experiments(tmp_path: Path) -> Iterator[Tuple[Dict[str, Any], str]]:
    """
    Serves an experiments folder holding the two evaluations of the tests on a free local port, in a thread stopped
    after the test, the run of the live evaluation holding no configuration.

    Args:
        tmp_path (Path): The temporary folder of the test.

    Returns:
        Iterator[Tuple[Dict[str, Any], str]]: The folders and class logits of the evaluations with the server, and the
            address of the server.
    """
    live_folder, grid_folder = tmp_path / LIVE_EVALUATION, tmp_path / GRID_EVALUATION
    evaluations = {"live_folder": live_folder, "grid_folder": grid_folder,
                   "live_logits": write_explorer_evaluation(evaluation_folder=live_folder, seed=1, with_logits=True),
                   "grid_logits": write_explorer_evaluation(evaluation_folder=grid_folder, seed=2, with_logits=False)}
    server = TemperatureExplorerServer(server_address=("127.0.0.1", 0), experiments_roots=[tmp_path / "experiments"])
    evaluations["server"] = server
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield evaluations, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def request_server(url: str, content: Optional[Dict[str, Any]]) -> Tuple[int, bytes]:
    """
    Sends a request as the page does: a GET without content, a POST of a JSON body with it.

    Args:
        url (str): The address of the request.
        content (Optional[Dict[str, Any]]): The posted body, None for a GET.

    Returns:
        Tuple[int, bytes]: The status of the answer and its body.
    """
    data = None if content is None else json.dumps(obj=content).encode(encoding="utf-8")
    request = urllib.request.Request(url=url, data=data, method="GET" if content is None else "POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(url=request) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_index_and_pages_are_served_with_their_evaluation(served_experiments: Tuple[Dict[str, Any], str]) -> None:
    """
    Verifies that the index lists both evaluations, the training one and the checkpoint evaluation, any temperature for
    the one with logits and the grid for the other, and that every page is built with the current template, its data
    naming the evaluation and the export settings.
    """
    _, address = served_experiments
    status, index = request_server(url=f"{address}/", content=None)
    assert status == 200
    assert index.count(b"<tr><td>") == 2 and b"CLS_LIVE_MC3" in index and b"CLS_GRID_MC3" in index
    assert b"<td>Any</td>" in index and b"<td>Grid only</td>" in index
    assert b"<td>Training</td>" in index and b"<td>Checkpoint evaluation</td>" in index

    for identifier, live_temperatures in [(LIVE_EVALUATION, True), (GRID_EVALUATION, False)]:
        status, page = request_server(url=f"{address}/explorer?evaluation={quote(identifier)}", content=None)
        page_data = json.loads(s=EXPLORER_DATA_PATTERN.search(string=page.decode(encoding="utf-8")).group(1))
        assert status == 200 and b'id="deployment"' in page
        assert (page_data["evaluation"], page_data["live_temperatures"]) == (identifier, live_temperatures)
        assert page_data["export_defaults"]["formats"] == ["torchscript", "onnx"]


@pytest.mark.parametrize(argnames=("query", "expected_status", "expected_message"), argvalues=[
    (f"evaluation={quote(GRID_EVALUATION)}&temperature=2.3", 400, "only the temperatures of its slider"),
    (f"evaluation={quote(LIVE_EVALUATION)}&temperature=0", 400, "Expected a logit temperature"),
    (f"evaluation={quote(LIVE_EVALUATION)}&temperature=hot", 400, "could not convert"),
    ("evaluation=experiments/../CLS_LIVE_MC3/Evaluation/Iteration_6&temperature=2", 404, "Unknown evaluation"),
    ("evaluation=other/CLS_LIVE_MC3/Evaluation/Iteration_6&temperature=2", 404, "Unknown evaluation")])
def test_wrong_temperature_requests_are_refused(served_experiments: Tuple[Dict[str, Any], str], query: str,
                                                expected_status: int, expected_message: str) -> None:
    """
    Verifies that a typed temperature is refused for an evaluation without logits, outside the calibration range or not
    a number, and that an evaluation outside the experiments folders is unknown.
    """
    _, address = served_experiments
    status, body = request_server(url=f"{address}/api/temperature?{query}", content=None)
    assert status == expected_status and expected_message in json.loads(s=body)["error"]


def test_typed_temperature_is_described_from_the_logits(served_experiments: Tuple[Dict[str, Any], str]) -> None:
    """
    Verifies that a typed temperature of 2.3 is described as the explorer describes it, for the classes of the page.
    """
    evaluations, address = served_experiments
    status, body = request_server(url=f"{address}/api/temperature?evaluation={quote(LIVE_EVALUATION)}&temperature=2.3",
                                  content=None)
    assert status == 200
    assert json.loads(s=body) == describe_explorer_temperature(class_logits_record=evaluations["live_logits"],
                                                               shown_class_indices=[1, 0, 2], logit_temperature=2.3)


def test_saved_thresholds_are_found_by_the_server_or_checked(served_experiments: Tuple[Dict[str, Any], str]) -> None:
    """
    Verifies that a save of the page with logits writes the thresholds found by the server from the logits at the
    posted temperature, whatever thresholds were posted, that the page without logits saves its posted thresholds once
    checked, those of other classes being refused without writing, and that the status gives the saved temperature,
    the export folder of the checkpoint and why it cannot be exported.
    """
    evaluations, address = served_experiments
    status, body = request_server(url=f"{address}/api/save", content={
        "evaluation": LIVE_EVALUATION, "temperature": 2.3, "thresholds": {"logit_temperature": 2.3, "classes": {}}})
    saved_path = evaluations["live_folder"] / SAVED_THRESHOLDS_FILE_NAME
    expected_thresholds = compute_temperature_thresholds(class_logits_record=evaluations["live_logits"],
                                                         logit_temperature=2.3)
    assert status == 200 and json.loads(s=body) == {"path": str(saved_path), "temperature": 2.3}
    assert json.loads(s=saved_path.read_text()) == expected_thresholds

    # Status of the saved temperature, the run folder of the evaluation holding no configuration
    status, body = request_server(url=f"{address}/api/status?evaluation={quote(LIVE_EVALUATION)}", content=None)
    run_status = json.loads(s=body)
    assert status == 200 and run_status["saved"]["temperature"] == 2.3
    assert run_status["next_export_folder"].endswith("experiments/CLS_LIVE_MC3/Exports/Iteration_6_T2.3")
    assert "run folder" in run_status["export_unavailable"] and run_status["exports"] == []
    assert run_status["export_job"] is None and not run_status["busy"]

    # Page without logits: the posted thresholds of other classes refused, then those of its classes saved
    grid_folder = evaluations["grid_folder"]
    posted_thresholds = compute_temperature_thresholds(class_logits_record=evaluations["grid_logits"],
                                                       logit_temperature=2.0)
    wrong_thresholds = {**posted_thresholds, "classes": {class_name: None for class_name in ["dog", "cat", "bird"]}}
    status, body = request_server(url=f"{address}/api/save", content={
        "evaluation": GRID_EVALUATION, "temperature": 2.0, "thresholds": wrong_thresholds})
    assert status == 400 and "Expected the class names" in json.loads(s=body)["error"]
    assert not (grid_folder / SAVED_THRESHOLDS_FILE_NAME).exists()
    status, _ = request_server(url=f"{address}/api/save", content={
        "evaluation": GRID_EVALUATION, "temperature": 2.0, "thresholds": posted_thresholds})
    assert status == 200
    assert list_file_thresholds(thresholds=json.loads(s=(grid_folder / SAVED_THRESHOLDS_FILE_NAME).read_text())) == \
        list_file_thresholds(thresholds=posted_thresholds)
    assert list(posted_thresholds["classes"]) == EXPLORER_CLASS_NAMES


@pytest.mark.parametrize(argnames=("identifier", "save_first", "expected_message"), argvalues=[
    (LIVE_EVALUATION, False, "save the thresholds of a temperature first"),
    (LIVE_EVALUATION, True, "No training_configuration.yaml in the run folder")])
def test_exports_without_saved_thresholds_or_run_are_refused(served_experiments: Tuple[Dict[str, Any], str],
                                                             identifier: str, save_first: bool,
                                                             expected_message: str) -> None:
    """
    Verifies that the export button is refused with the reason before thresholds are saved, and when the run folder of
    the evaluation holds no configuration, no export folder being created.
    """
    evaluations, address = served_experiments
    if save_first:
        request_server(url=f"{address}/api/save", content={"evaluation": identifier, "temperature": 2.0})
    status, body = request_server(url=f"{address}/api/export", content={
        "evaluation": identifier, "formats": ["onnx"], "model_version": 1, "maximum_batch_size": 4,
        "tensorrt_precision": None})
    assert status == 400 and expected_message in json.loads(s=body)["error"]
    assert not (evaluations["live_folder"].parents[2] / "Exports").exists()


class RunningProcess:
    """
    Stand-in of the process of an export still running, for the refusals of the server while it runs.
    """

    def poll(self) -> Optional[int]:
        """
        Tells that the export has not ended, as subprocess.Popen.poll does.

        Returns:
            Optional[int]: None, the process still running.
        """
        return None


def test_saves_and_exports_wait_for_the_running_export(served_experiments: Tuple[Dict[str, Any], str],
                                                       tmp_path: Path) -> None:
    """
    Verifies that while the checkpoint of a page is being exported its saved thresholds cannot change, and that a second
    export is refused while one runs, the status of the page showing the running export.
    """
    evaluations, address = served_experiments
    request_server(url=f"{address}/api/save", content={"evaluation": LIVE_EVALUATION, "temperature": 2.0})
    log_path = tmp_path / "export_log.txt"
    log_path.write_text(data="\x1b[34mExporting Checkpoint\x1b[0m\n")
    evaluations["server"].export_job = {"evaluation": LIVE_EVALUATION, "process": RunningProcess(),
                                        "log_path": log_path, "export_folder": tmp_path / "Iteration_6_T2",
                                        "temperature": 2.0, "started_at": "2026-11-12 15:00:00", "reported": False}

    # The saved temperature of the exported checkpoint is kept, and no second export starts
    status, body = request_server(url=f"{address}/api/save", content={"evaluation": LIVE_EVALUATION,
                                                                      "temperature": 3.0})
    assert status == 409 and "being exported" in json.loads(s=body)["error"]
    assert json.loads(s=(evaluations["live_folder"] / SAVED_THRESHOLDS_FILE_NAME).read_text())[
        "logit_temperature"] == 2.0
    status, body = request_server(url=f"{address}/api/export", content={
        "evaluation": GRID_EVALUATION, "formats": ["onnx"], "model_version": 1, "maximum_batch_size": 4,
        "tensorrt_precision": None})
    assert status == 409 and "An export is running" in json.loads(s=body)["error"]

    # The page of the exported checkpoint shows the running export, its console colours removed, the other one waits
    status, body = request_server(url=f"{address}/api/status?evaluation={quote(LIVE_EVALUATION)}", content=None)
    export_job = json.loads(s=body)["export_job"]
    assert (export_job["state"], export_job["log"], json.loads(s=body)["busy"]) == ("running", "Exporting Checkpoint",
                                                                                    True)
    status, body = request_server(url=f"{address}/api/status?evaluation={quote(GRID_EVALUATION)}", content=None)
    assert json.loads(s=body)["export_job"] is None and json.loads(s=body)["busy"]
