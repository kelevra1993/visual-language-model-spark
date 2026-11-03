"""
When run, serves the temperature explorer of one evaluated checkpoint of a classifier on the local machine, so that
the "Save Thresholds At This Temperature" button of 'temperature_explorer.html' writes
'saved_temperature_thresholds.json' next to the page: the thresholds of every class at the temperature shown, which
vision_training/deployment/export_model.py builds into the exported classifier with use_saved_temperature. The page is
written by the classification trainer or scripts/evaluation/evaluate_checkpoint.py with the per-class report; opened as
a plain file, its button downloads the file instead.
"""

import json

from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from vision_training.evaluation.classification.temperature_explorer import (EXPLORER_FILE_NAME,
                                                                            SAVE_THRESHOLDS_ROUTE,
                                                                            write_saved_thresholds)
from vision_training.utilities.os_utilities import format_path_link, print_color


class TemperatureExplorerHandler(SimpleHTTPRequestHandler):
    """
    Request handler of the served evaluation folder: the files of the folder (the page) are served as they are, and the
    thresholds posted by the save button of the page are checked and written next to it.
    """

    def send_json(self, status: HTTPStatus, content: Dict[str, Any]) -> None:
        """
        Answers a request of the save button with a JSON body, read by the page to tell whether the thresholds were
        saved.

        Args:
            status (HTTPStatus): The status of the answer.
            content (Dict[str, Any]): The body of the answer.
        """
        body = json.dumps(obj=content).encode(encoding="utf-8")
        self.send_response(code=status)
        self.send_header(keyword="Content-Type", value="application/json")
        self.send_header(keyword="Content-Length", value=str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        """
        Saves the thresholds posted by the save button of the page in the served evaluation folder, after checking them
        against its 'class_thresholds.json' (write_saved_thresholds), the answer giving the written file or the reason
        of the refusal.
        """
        if self.path != SAVE_THRESHOLDS_ROUTE:
            self.send_json(status=HTTPStatus.NOT_FOUND, content={"error": f"Unknown route {self.path}."})
            return

        # Thresholds posted by the page, refused with the reason when they are not those of this evaluation
        try:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            saved_thresholds = json.loads(s=body)
            saved_path = write_saved_thresholds(evaluation_folder=Path(self.directory),
                                                saved_thresholds=saved_thresholds)
        except (ValueError, TypeError, FileNotFoundError) as error:
            self.send_json(status=HTTPStatus.BAD_REQUEST, content={"error": str(error)})
            return
        print_color(color="green", output=f"Thresholds At T = {saved_thresholds['logit_temperature']:g} Saved : "
                                          f"{format_path_link(path=saved_path)}", indent=1)
        self.send_json(status=HTTPStatus.OK, content={"path": str(saved_path),
                                                      "temperature": saved_thresholds["logit_temperature"]})


def serve_temperature_explorer(evaluation_folder: Path, port: int) -> None:
    """
    Serves the evaluation folder of a checkpoint on 127.0.0.1 until the script is stopped, so that the temperature
    explorer opened at the printed address saves its thresholds next to it.

    Args:
        evaluation_folder (Path): The evaluation folder of the checkpoint, 'Evaluation/Iteration_<n>' ('EMA' inside it
            for the averaged weights), holding 'temperature_explorer.html' and 'class_thresholds.json'.
        port (int): The local port of the server.

    Raises:
        FileNotFoundError: If the folder holds no temperature explorer.
    """
    explorer_path = evaluation_folder / EXPLORER_FILE_NAME
    if not explorer_path.is_file():
        raise FileNotFoundError(f"No {EXPLORER_FILE_NAME} in {evaluation_folder}: it is written by the classification "
                                f"evaluations with evaluation.per_class_report.")

    # The handler serves the files of the evaluation folder, whatever the folder the script is run from
    server = ThreadingHTTPServer(server_address=("127.0.0.1", port),
                                 RequestHandlerClass=partial(TemperatureExplorerHandler,
                                                             directory=str(evaluation_folder)))
    print_color(color="blue", output=f"Temperature Explorer : http://127.0.0.1:{port}/{EXPLORER_FILE_NAME}",
                add_separators=True)
    print_color(color="blue", output=f"Evaluation Folder : {format_path_link(path=evaluation_folder)}", indent=1)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print_color(color="yellow", output="Temperature Explorer Stopped", indent=1)
    finally:
        server.server_close()


if __name__ == "__main__":
    # Evaluation folder of the checkpoint whose temperature is chosen, in the run folder or in the evaluations folder of
    # scripts/evaluation/evaluate_checkpoint.py, e.g. <runs>/evaluations/<run>/Evaluation/Iteration_3500, with 'EMA'
    # after it for the averaged weights
    evaluation_folder = Path(__file__).absolute().parents[2] / "experiments" / "evaluations" / "<run name>" / \
        "Evaluation" / "Iteration_3500"
    # Local port of the page, opened at http://127.0.0.1:<port>/temperature_explorer.html
    port = 8090

    serve_temperature_explorer(evaluation_folder=evaluation_folder, port=port)
