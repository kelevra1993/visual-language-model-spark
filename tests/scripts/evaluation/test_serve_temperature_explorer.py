"""
Tests of scripts/evaluation/serve_temperature_explorer.py, the local server of the temperature explorer of an evaluated
checkpoint: the page is served from its evaluation folder, the thresholds posted by its save button are written next
to it as saved_temperature_thresholds.json, and thresholds of another evaluation are refused without writing
anything.
"""

import json
import threading
import urllib.error
import urllib.request

from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import pytest

from scripts.evaluation.serve_temperature_explorer import TemperatureExplorerHandler
from vision_training.evaluation.classification.temperature_explorer import (EXPLORER_FILE_NAME,
                                                                            SAVE_THRESHOLDS_ROUTE,
                                                                            SAVED_THRESHOLDS_FILE_NAME)

# Thresholds of the evaluation at the temperature 1, and those of the temperature 2 posted by the save button
CLASS_THRESHOLDS = {"logit_temperature": 1.0, "target_precision": 0.99,
                    "classes": {"bad": {"threshold": 0.9999, "recall": 0.58}, "good": None},
                    "additional_targets": [{"target_precision": 0.9,
                                            "classes": {"bad": {"threshold": 0.03, "recall": 0.93},
                                                        "good": {"threshold": 0.82, "recall": 0.91}}}]}
SAVED_THRESHOLDS = {"logit_temperature": 2.0, "target_precision": 0.99,
                    "classes": {"bad": {"threshold": 0.96, "recall": 0.58}, "good": None},
                    "additional_targets": [{"target_precision": 0.9,
                                            "classes": {"bad": {"threshold": 0.31, "recall": 0.93},
                                                        "good": {"threshold": 0.71, "recall": 0.91}}}]}


@pytest.fixture
def served_evaluation_folder(tmp_path: Path) -> Iterator[Tuple[Path, str]]:
    """
    Serves an evaluation folder holding a temperature explorer and its class thresholds on a free local port, in a
    thread stopped after the test.

    Args:
        tmp_path (Path): The temporary folder of the test.

    Returns:
        Iterator[Tuple[Path, str]]: The evaluation folder and the address of the server.
    """
    evaluation_folder = tmp_path / "Evaluation" / "Iteration_3500"
    evaluation_folder.mkdir(parents=True)
    (evaluation_folder / EXPLORER_FILE_NAME).write_text(data="<html>explorer</html>")
    (evaluation_folder / "class_thresholds.json").write_text(data=json.dumps(obj=CLASS_THRESHOLDS))
    server = ThreadingHTTPServer(server_address=("127.0.0.1", 0),
                                 RequestHandlerClass=partial(TemperatureExplorerHandler,
                                                             directory=str(evaluation_folder)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield evaluation_folder, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def post_json(url: str, content: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """
    Posts a JSON body as the save button of the page does, and reads the JSON answer of the server.

    Args:
        url (str): The address posted to.
        content (Dict[str, Any]): The posted body.

    Returns:
        Tuple[int, Dict[str, Any]]: The status of the answer and its body.
    """
    request = urllib.request.Request(url=url, data=json.dumps(obj=content).encode(encoding="utf-8"), method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(url=request) as response:
            return response.status, json.loads(s=response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(s=error.read())


def test_page_is_served_and_saved_thresholds_are_written(served_evaluation_folder: Tuple[Path, str]) -> None:
    """
    Verifies that the page is served from its evaluation folder and that the thresholds posted at the temperature 2
    are written next to it, as posted, the answer giving the written file and the temperature.
    """
    evaluation_folder, address = served_evaluation_folder
    with urllib.request.urlopen(url=f"{address}/{EXPLORER_FILE_NAME}") as response:
        assert response.read() == b"<html>explorer</html>"

    status, answer = post_json(url=f"{address}{SAVE_THRESHOLDS_ROUTE}", content=SAVED_THRESHOLDS)
    saved_path = evaluation_folder / SAVED_THRESHOLDS_FILE_NAME
    assert status == 200 and answer == {"path": str(saved_path), "temperature": 2.0}
    assert json.loads(s=saved_path.read_text()) == SAVED_THRESHOLDS


@pytest.mark.parametrize(argnames=("route", "content", "expected_status", "expected_message"), argvalues=[
    (SAVE_THRESHOLDS_ROUTE, {**SAVED_THRESHOLDS, "classes": {"bad": None, "other": None}}, 400, "class names")])
def test_invalid_saves_are_refused_without_writing(served_evaluation_folder: Tuple[Path, str], route: str,
                                                   content: Dict[str, Any], expected_status: int,
                                                   expected_message: str) -> None:
    """
    Verifies that thresholds of other classes are refused with the reason, no thresholds file being written.
    """
    evaluation_folder, address = served_evaluation_folder
    status, answer = post_json(url=f"{address}{route}", content=content)
    assert status == expected_status and expected_message in answer["error"]
    assert not (evaluation_folder / SAVED_THRESHOLDS_FILE_NAME).exists()
