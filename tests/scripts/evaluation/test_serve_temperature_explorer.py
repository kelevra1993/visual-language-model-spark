"""
Tests of scripts/evaluation/serve_temperature_explorer.py, the local server of the temperature explorers of the
experiments folders: the index groups the evaluations holding a page by run folder, then by iteration (newest first,
trained weights before averaged ones), every page is served with the current template
and names its evaluation, with the global metrics of its checkpoint and the best evaluation of its summary, a typed
temperature is described from the class logits, the score ranges are counted from them, from default cut points before
any are typed (wrong requests refused), the thresholds saved from a page
with logits are found by the server for every class whatever was posted, those of a page without logits are checked
against its class_thresholds.json, the status gives the saved temperature and the export folder, and an export without
saved thresholds or without its run is refused with the reason, nothing being written by a refused request.
"""

import json
import re
import threading
import urllib.error
import urllib.request

from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple
from urllib.parse import quote

import pytest

from scripts.evaluation.serve_temperature_explorer import (TemperatureExplorerServer, build_index_groups,
                                                           find_explorer_evaluations)
from tests.evaluation.classification.test_temperature_explorer import (EXPLORER_CLASS_NAMES, list_file_thresholds,
                                                                       write_explorer_evaluation)
from vision_training.evaluation.classification.temperature_explorer import (DEFAULT_SCORE_CUT_POINTS,
                                                                            EXPLORER_DATA_PATTERN,
                                                                            SAVED_THRESHOLDS_FILE_NAME,
                                                                            check_score_cut_points,
                                                                            compute_score_ranges,
                                                                            compute_temperature_thresholds,
                                                                            describe_explorer_temperature,
                                                                            write_saved_thresholds)
from vision_training.evaluation.evaluation_curves import EVALUATION_SUMMARY_FILE_NAME, write_evaluation_summary

# Evaluations of the tests: one with its class logits, evaluated by the training of its run, one without, evaluated by
# evaluate_checkpoint.py for the averaged weights, in the Checkpoint_Evaluation folder of a run without configuration
LIVE_EVALUATION = "experiments/CLS_LIVE_MC3/Evaluation/Iteration_6"
GRID_EVALUATION = "experiments/CLS_GRID_MC3/Checkpoint_Evaluation/Iteration_3/EMA"

# Columns of the evaluation summaries of the tests, as the classification trainer writes them, the top-5 accuracy left
# empty (a column the page does not show)
SUMMARY_COLUMNS = ["iteration", "Accuracy", "Balanced Accuracy", "Macro F1", "Top-5 Accuracy", "ema_Accuracy",
                   "ema_Balanced Accuracy", "ema_Macro F1", "ema_Top-5 Accuracy"]


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


def request_page_data(address: str, identifier: str) -> Dict[str, Any]:
    """
    Requests the page of an evaluation and reads the data the server built it with, as the page script reads it.

    Args:
        address (str): The address of the server.
        identifier (str): The identifier of the evaluation.

    Returns:
        Dict[str, Any]: The data of the page.
    """
    status, page = request_server(url=f"{address}/explorer?evaluation={quote(identifier)}", content=None)
    assert status == 200
    return json.loads(s=EXPLORER_DATA_PATTERN.search(string=page.decode(encoding="utf-8")).group(1))


def test_index_and_pages_are_served_with_their_evaluation(served_experiments: Tuple[Dict[str, Any], str]) -> None:
    """
    Verifies that the index lists both evaluations, the training one and the checkpoint evaluation, any temperature for
    the one with logits and the grid for the other, and that every page is built with the current template, its data
    naming the evaluation and the export settings.

    Args:
        served_experiments (Tuple[Dict[str, Any], str]): The served evaluations and the address of the server.
    """
    _, address = served_experiments
    status, index = request_server(url=f"{address}/", content=None)
    assert status == 200
    assert index.count(b'<details class="run">') == 2 and index.count(b"<tr><td>") == 2
    assert all(f'<span class="run-name">{run_name}</span>'.encode() in index for run_name in ["CLS_LIVE_MC3",
                                                                                             "CLS_GRID_MC3"])
    assert b"<td>Any</td>" in index and b"<td>Grid only</td>" in index
    assert b"<td>Training</td>" in index and b"<td>Checkpoint evaluation</td>" in index
    for identifier, weights_label in [(LIVE_EVALUATION, b"Trained"), (GRID_EVALUATION, b"EMA")]:
        assert f'<a href="/explorer?evaluation={quote(identifier)}">'.encode() + weights_label + b"</a>" in index

    for identifier, live_temperatures in [(LIVE_EVALUATION, True), (GRID_EVALUATION, False)]:
        status, page = request_server(url=f"{address}/explorer?evaluation={quote(identifier)}", content=None)
        page_data = json.loads(s=EXPLORER_DATA_PATTERN.search(string=page.decode(encoding="utf-8")).group(1))
        assert status == 200 and b'id="deployment"' in page
        assert (page_data["evaluation"], page_data["live_temperatures"]) == (identifier, live_temperatures)
        assert page_data["export_defaults"]["formats"] == ["torchscript", "onnx"]


def test_index_groups_the_evaluations_by_run_then_iteration(tmp_path: Path) -> None:
    """
    Verifies that the index gives one block per run folder, a run of a subfolder included, whose iterations are listed
    newest first with only the newest open, and whose iteration blocks list the trained weights before the averaged
    ones, the training evaluation before the checkpoint evaluation, with the temperature saved for the averaged weights
    in the title of the iteration and the count of the run.

    Args:
        tmp_path (Path): The temporary folder of the test, holding the experiments folder.
    """
    experiments_root = tmp_path / "experiments"
    for relative_path, seed in [
            ("CLS_A_MC3/Evaluation/Iteration_6/EMA", 1), ("CLS_A_MC3/Evaluation/Iteration_6", 2),
            ("CLS_A_MC3/Checkpoint_Evaluation/Iteration_6", 3), ("CLS_A_MC3/Evaluation/Iteration_12", 4),
            ("group/CLS_B_MC3/Evaluation/Iteration_3", 5)]:
        class_logits_record = write_explorer_evaluation(evaluation_folder=experiments_root / relative_path, seed=seed,
                                                        with_logits=True)
        if relative_path.endswith("EMA"):
            write_saved_thresholds(evaluation_folder=experiments_root / relative_path,
                                   saved_thresholds=compute_temperature_thresholds(
                                       class_logits_record=class_logits_record, logit_temperature=2.5))
    index_body = build_index_groups(evaluations=find_explorer_evaluations(
        experiments_roots={"experiments": experiments_root}))

    # One block per run, in the order of their identifiers, the run of the subfolder naming its folder
    run_blocks = index_body.split('<details class="run">')[1:]
    assert [block.split("</span>")[0] for block in run_blocks] == ['<summary><span class="run-name">CLS_A_MC3',
                                                                   '<summary><span class="run-name">CLS_B_MC3']
    assert "experiments · 2 iterations (newest 12) · 1 saved T · 0 exports" in run_blocks[0]
    assert "experiments/group · 1 iteration (newest 3)" in run_blocks[1]

    # Iterations of the first run newest first, the newest open, then the evaluations of iteration 6 in their order
    iteration_blocks = run_blocks[0].split('<details class="iteration"')[1:]
    assert [block.split("</summary>")[0] for block in iteration_blocks] == [
        " open><summary>Iteration 12 · Trained", "><summary>Iteration 6 · Trained, EMA · saved EMA T = 2.5"]
    iteration_links = re.findall(pattern=r'evaluation=([^"]+)">(Trained|EMA)</a></td><td>([^<]+)</td>',
                                 string=iteration_blocks[1])
    assert iteration_links == [("experiments/CLS_A_MC3/Evaluation/Iteration_6", "Trained", "Training"),
                               ("experiments/CLS_A_MC3/Checkpoint_Evaluation/Iteration_6", "Trained",
                                "Checkpoint evaluation"),
                               ("experiments/CLS_A_MC3/Evaluation/Iteration_6/EMA", "EMA", "Training")]


@pytest.mark.parametrize(argnames=("query", "expected_status", "expected_message"), argvalues=[
    (f"evaluation={quote(GRID_EVALUATION)}&temperature=2.3", 400, "only the temperatures of its slider"),
    (f"evaluation={quote(LIVE_EVALUATION)}&temperature=0", 400, "Expected a logit temperature"),
    (f"evaluation={quote(LIVE_EVALUATION)}&temperature=hot", 400, "could not convert"),
    ("evaluation=experiments/../CLS_LIVE_MC3/Evaluation/Iteration_6&temperature=2", 404, "Unknown evaluation"),
    ("evaluation=other/CLS_LIVE_MC3/Evaluation/Iteration_6&temperature=2", 404, "Unknown evaluation")])
def test_wrong_temperature_requests_are_refused(served_experiments: Tuple[Dict[str, Any], str], query: str,
                                                expected_status: int, expected_message: str) -> None:
    """
    Verifies that a typed temperature is refused for an evaluation without logits, outside the temperature range or not
    a number, and that an evaluation outside the experiments folders is unknown.

    Args:
        served_experiments (Tuple[Dict[str, Any], str]): The served evaluations and the address of the server.
        query (str): The query of the refused request.
        expected_status (int): The status of the refusal.
        expected_message (str): A part of the reason of the refusal.
    """
    _, address = served_experiments
    status, body = request_server(url=f"{address}/api/temperature?{query}", content=None)
    assert status == expected_status and expected_message in json.loads(s=body)["error"]


def test_typed_temperature_is_described_from_the_logits(served_experiments: Tuple[Dict[str, Any], str]) -> None:
    """
    Verifies that a typed temperature of 2.3 is described as the explorer describes it, for the classes of the page.

    Args:
        served_experiments (Tuple[Dict[str, Any], str]): The served evaluations and the address of the server.
    """
    evaluations, address = served_experiments
    status, body = request_server(url=f"{address}/api/temperature?evaluation={quote(LIVE_EVALUATION)}&temperature=2.3",
                                  content=None)
    assert status == 200
    assert json.loads(s=body) == describe_explorer_temperature(class_logits_record=evaluations["live_logits"],
                                                               shown_class_indices=[0, 1, 2], logit_temperature=2.3)


def test_score_ranges_are_counted_from_the_logits(served_experiments: Tuple[Dict[str, Any], str]) -> None:
    """
    Verifies that the score ranges asked by the page at the temperature 2.3, with cut points typed in any order, are
    those the explorer counts for the classes of the page, and that the served page holds the score ranges section.

    Args:
        served_experiments (Tuple[Dict[str, Any], str]): The served evaluations and the address of the server.
    """
    evaluations, address = served_experiments
    status, body = request_server(url=f"{address}/api/score_ranges?evaluation={quote(LIVE_EVALUATION)}"
                                      f"&temperature=2.3&cut_points=0.8,0.4", content=None)
    assert status == 200
    assert json.loads(s=body) == compute_score_ranges(class_logits_record=evaluations["live_logits"],
                                                      shown_class_indices=[0, 1, 2], logit_temperature=2.3,
                                                      cut_points=[0.4, 0.8])
    status, page = request_server(url=f"{address}/explorer?evaluation={quote(LIVE_EVALUATION)}", content=None)
    assert status == 200 and b'id="score-ranges"' in page


def test_pages_show_the_global_metrics_of_their_checkpoint(served_experiments: Tuple[Dict[str, Any], str]) -> None:
    """
    Verifies that the page of an evaluation carries the global metrics of its checkpoint read from the evaluation
    summary of its folder (the 'ema_' columns for the averaged weights), with the best evaluation of that summary by
    balanced accuracy over the trained and averaged weights, linked when it holds a page, and no metric while the
    summary lacks the row of its iteration.

    Args:
        served_experiments (Tuple[Dict[str, Any], str]): The served evaluations and the address of the server.
    """
    evaluations, address = served_experiments
    assert request_page_data(address=address, identifier=LIVE_EVALUATION)["model_performance"] is None

    # Training evaluations of the live run: its trained weights at iteration 6, the best being the averaged weights of
    # iteration 3, which hold no page
    write_evaluation_summary(summary_path=evaluations["live_folder"].parent / EVALUATION_SUMMARY_FILE_NAME,
                             column_names=SUMMARY_COLUMNS, rows_by_iteration={
                                 3: {"Accuracy": 0.91, "Balanced Accuracy": 0.9, "Macro F1": 0.89,
                                     "ema_Accuracy": 0.96, "ema_Balanced Accuracy": 0.95, "ema_Macro F1": 0.94},
                                 6: {"Accuracy": 0.93, "Balanced Accuracy": 0.92, "Macro F1": 0.91,
                                     "ema_Accuracy": 0.92, "ema_Balanced Accuracy": 0.91, "ema_Macro F1": 0.9}})
    assert request_page_data(address=address, identifier=LIVE_EVALUATION)["model_performance"] == {
        "run": "experiments/CLS_LIVE_MC3", "summary_folder": "Evaluation", "number_iterations": 2,
        "metrics": [{"name": "Balanced Accuracy", "value": 0.92}, {"name": "Accuracy", "value": 0.93},
                    {"name": "Macro F1", "value": 0.91}],
        "best": {"metric": "Balanced Accuracy", "value": 0.95, "iteration": 3, "use_averaged_weights": True,
                 "evaluation": None, "reached": False}}

    # Checkpoint evaluation of the averaged weights, the best of its own summary, linked to its own page
    write_evaluation_summary(summary_path=evaluations["grid_folder"].parent.parent / EVALUATION_SUMMARY_FILE_NAME,
                             column_names=SUMMARY_COLUMNS, rows_by_iteration={
                                 3: {"Accuracy": 0.81, "Balanced Accuracy": 0.8, "Macro F1": 0.79, "ema_Accuracy": 0.86,
                                     "ema_Balanced Accuracy": 0.85, "ema_Macro F1": 0.84}})
    assert request_page_data(address=address, identifier=GRID_EVALUATION)["model_performance"] == {
        "run": "experiments/CLS_GRID_MC3", "summary_folder": "Checkpoint_Evaluation", "number_iterations": 1,
        "metrics": [{"name": "Balanced Accuracy", "value": 0.85}, {"name": "Accuracy", "value": 0.86},
                    {"name": "Macro F1", "value": 0.84}],
        "best": {"metric": "Balanced Accuracy", "value": 0.85, "iteration": 3, "use_averaged_weights": True,
                 "evaluation": GRID_EVALUATION, "reached": True}}


def test_score_ranges_start_from_the_default_cut_points(served_experiments: Tuple[Dict[str, Any], str]) -> None:
    """
    Verifies that the served page gives the cut points its score ranges start from before any are typed, valid cut
    points that the server counts the score ranges with.

    Args:
        served_experiments (Tuple[Dict[str, Any], str]): The served evaluations and the address of the server.
    """
    evaluations, address = served_experiments
    page_data = request_page_data(address=address, identifier=LIVE_EVALUATION)
    assert page_data["default_score_cut_points"] == DEFAULT_SCORE_CUT_POINTS
    assert check_score_cut_points(cut_points=DEFAULT_SCORE_CUT_POINTS) == DEFAULT_SCORE_CUT_POINTS
    cut_points_text = ",".join(str(cut_point) for cut_point in DEFAULT_SCORE_CUT_POINTS)
    status, body = request_server(url=f"{address}/api/score_ranges?evaluation={quote(LIVE_EVALUATION)}"
                                      f"&temperature=1&cut_points={cut_points_text}", content=None)
    assert status == 200
    assert json.loads(s=body) == compute_score_ranges(class_logits_record=evaluations["live_logits"],
                                                      shown_class_indices=[0, 1, 2], logit_temperature=1,
                                                      cut_points=DEFAULT_SCORE_CUT_POINTS)


@pytest.mark.parametrize(argnames=("query", "expected_message"), argvalues=[
    (f"evaluation={quote(GRID_EVALUATION)}&temperature=2.3&cut_points=0.5", "the score ranges are counted from"),
    (f"evaluation={quote(LIVE_EVALUATION)}&temperature=0&cut_points=0.5", "Expected a logit temperature"),
    (f"evaluation={quote(LIVE_EVALUATION)}&temperature=2.3&cut_points=0.5,1", "Expected 1 to 10 distinct scores"),
    (f"evaluation={quote(LIVE_EVALUATION)}&temperature=2.3&cut_points=0.5,0.5", "Expected 1 to 10 distinct scores"),
    (f"evaluation={quote(LIVE_EVALUATION)}&temperature=2.3&cut_points=0.5,high", "could not convert"),
    (f"evaluation={quote(LIVE_EVALUATION)}&temperature=2.3", "cut_points")])
def test_wrong_score_range_requests_are_refused(served_experiments: Tuple[Dict[str, Any], str], query: str,
                                                expected_message: str) -> None:
    """
    Verifies that score ranges are refused for an evaluation without logits, at a temperature outside the temperature
    range, and for cut points outside ]0, 1[, repeated, not numbers or missing.

    Args:
        served_experiments (Tuple[Dict[str, Any], str]): The served evaluations and the address of the server.
        query (str): The query of the refused request.
        expected_message (str): A part of the reason of the refusal.
    """
    _, address = served_experiments
    status, body = request_server(url=f"{address}/api/score_ranges?{query}", content=None)
    assert status == 400 and expected_message in json.loads(s=body)["error"]


def test_saved_thresholds_are_found_by_the_server_or_checked(served_experiments: Tuple[Dict[str, Any], str]) -> None:
    """
    Verifies that a save of the page with logits writes the thresholds found by the server from the logits at the
    posted temperature, whatever thresholds were posted, that the page without logits saves its posted thresholds once
    checked, those of other classes being refused without writing, and that the status gives the saved temperature,
    the export folder of the checkpoint and why it cannot be exported.

    Args:
        served_experiments (Tuple[Dict[str, Any], str]): The served evaluations and the address of the server.
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

    Args:
        served_experiments (Tuple[Dict[str, Any], str]): The served evaluations and the address of the server.
        identifier (str): The evaluation whose export is asked.
        save_first (bool): Whether thresholds are saved before the export is asked.
        expected_message (str): A part of the reason of the refusal.
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

    Args:
        served_experiments (Tuple[Dict[str, Any], str]): The served evaluations and the address of the server.
        tmp_path (Path): The temporary folder of the test, holding the console output of the running export.
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
