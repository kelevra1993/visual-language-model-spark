"""
When run, serves on the local machine the temperature explorers of every classifier evaluated in the experiments folders
of the repository, to choose the logit temperature of a checkpoint, save its thresholds and export it from the page.
The index lists the evaluations (run, iteration, weights, saved temperature, exports); every page is built again with
the current template, from the 'class_logits.npz' of the evaluation (any temperature can be typed in, and the images of
every class are counted in the score ranges of the cut points typed on the page, at the temperature shown) or from the
data of its written 'temperature_explorer.html' (the temperatures of its slider). The save button writes
'saved_temperature_thresholds.json' next to the page, the thresholds being found here from the logits; the export button
exports the checkpoint at the saved temperature into '<run>/Exports/Iteration_<n>[_EMA]_T<temperature>/'
(vision_training/deployment/explorer_export.py), in a separate process whose console output the page shows. The pages
are written by the classification trainer or scripts/evaluation/evaluate_checkpoint.py with the per-class report.
"""

import html
import json
import os
import re
import subprocess
import sys
import threading

from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, urlparse

from vision_training.deployment.explorer_export import (DEFAULT_EXPORT_SETTINGS, EXPORT_FORMATS,
                                                        EXPORT_LOG_FILE_NAME, TENSORRT_PRECISIONS,
                                                        build_export_folder, describe_evaluation_folder,
                                                        list_checkpoint_exports, prepare_export_job)
from vision_training.evaluation.classification.temperature_explorer import (CLASS_LOGITS_FILE_NAME,
                                                                            EXPLORER_FILE_NAME,
                                                                            SAVED_THRESHOLDS_FILE_NAME,
                                                                            build_explorer_data_from_logits,
                                                                            check_logit_temperature,
                                                                            check_score_cut_points,
                                                                            compute_score_ranges,
                                                                            compute_temperature_thresholds,
                                                                            describe_explorer_temperature,
                                                                            load_class_logits,
                                                                            read_explorer_page_data,
                                                                            render_explorer_page,
                                                                            write_saved_thresholds)
from vision_training.evaluation.checkpoint_evaluation import CHECKPOINT_EVALUATION_FOLDER_NAME
from vision_training.trainers.base_trainer import AVERAGED_WEIGHTS_FOLDER_NAME, EVALUATION_FOLDER_NAME
from vision_training.trainers.checkpointing import get_checkpoint_path
from vision_training.utilities.os_utilities import format_path_link, print_color

# Repository root, from which the export process imports the project, and the module of the export process
REPOSITORY_ROOT = Path(__file__).absolute().parents[2]
EXPORT_MODULE = "vision_training.deployment.explorer_export"

# Evaluation folders looked for under every experiments folder, those of the training ('<run>/Evaluation/Iteration_<n>')
# and of scripts/evaluation/evaluate_checkpoint.py ('<run>/Checkpoint_Evaluation/Iteration_<n>'), the runs up to two
# folders deeper (grouped in subfolders), without walking into the image folders of the evaluations
ITERATION_FOLDER_PATTERNS = [f"{run_depth}/{evaluation_folder_name}/Iteration_*" for run_depth in ["*", "*/*", "*/*/*"]
                             for evaluation_folder_name in [EVALUATION_FOLDER_NAME, CHECKPOINT_EVALUATION_FOLDER_NAME]]

# Template of the index page, its rows replacing the placeholder
INDEX_TEMPLATE_PATH = (REPOSITORY_ROOT / "vision_training" / "evaluation" / "classification" /
                       "temperature_explorer_index_template.html")
INDEX_ROWS_PLACEHOLDER = "__INDEX_ROWS__"

# Last lines of the console output of an export shown on the page, and the escape sequences of the colours and links of
# print_color and format_path_link, removed from it
NUMBER_LOG_LINES = 80
TERMINAL_ESCAPE_PATTERN = re.compile(pattern=r"\x1b\]8;;[^\x1b]*\x1b\\|\x1b\[[0-9;]*m")


class TemperatureExplorerServer(ThreadingHTTPServer):
    """
    Local server of the temperature explorers: the experiments folders it serves, the page data already built (keyed by
    the evaluation and the date of its logits), and the single export it runs at a time.
    """

    def __init__(self, server_address: Tuple[str, int], experiments_roots: List[Path]) -> None:
        """
        Starts the server on a local address for the evaluations of the experiments folders.

        Args:
            server_address (Tuple[str, int]): The host and port.
            experiments_roots (List[Path]): The experiments folders, of distinct folder names (the first part of the
                identifiers of their evaluations).

        Raises:
            ValueError: If two experiments folders share a name, or one is missing.
        """
        root_names = [experiments_root.name for experiments_root in experiments_roots]
        if len(set(root_names)) != len(root_names) or not all(root.is_dir() for root in experiments_roots):
            raise ValueError(f"Expected existing experiments folders of distinct names, got {experiments_roots}.")
        super().__init__(server_address=server_address, RequestHandlerClass=TemperatureExplorerHandler)
        self.experiments_roots = {experiments_root.name: experiments_root for experiments_root in experiments_roots}
        self.page_data_cache: Dict[Tuple[str, float], Dict[str, Any]] = {}
        self.export_lock = threading.Lock()
        self.export_job: Optional[Dict[str, Any]] = None


def is_explorer_evaluation(evaluation_folder: Path) -> bool:
    """
    Tells whether a folder is the evaluation of a classifier served by the explorer: its class_thresholds.json with its
    class logits or its written page.

    Args:
        evaluation_folder (Path): The candidate folder.

    Returns:
        bool: True for a served evaluation.
    """
    return (evaluation_folder / "class_thresholds.json").is_file() and (
        (evaluation_folder / CLASS_LOGITS_FILE_NAME).is_file() or (evaluation_folder / EXPLORER_FILE_NAME).is_file())


def find_explorer_evaluations(experiments_roots: Dict[str, Path]) -> Dict[str, Path]:
    """
    Finds the evaluations of the experiments folders that hold a temperature explorer, the trained and averaged weights
    of every evaluated iteration, for the index page; looked for again at every visit, so that a new evaluation appears.

    Args:
        experiments_roots (Dict[str, Path]): The experiments folders, keyed by name.

    Returns:
        Dict[str, Path]: The evaluation folders, keyed by their identifier '<experiments folder name>/<relative path>'.
    """
    evaluations = {}
    for root_name, experiments_root in experiments_roots.items():
        for pattern in ITERATION_FOLDER_PATTERNS:
            for iteration_folder in experiments_root.glob(pattern=pattern):
                if not re.fullmatch(pattern=r"Iteration_\d+", string=iteration_folder.name):
                    continue
                # The trained weights in the iteration folder, the averaged ones in its 'EMA' subfolder
                for evaluation_folder in [iteration_folder, iteration_folder / AVERAGED_WEIGHTS_FOLDER_NAME]:
                    if is_explorer_evaluation(evaluation_folder=evaluation_folder):
                        relative_path = evaluation_folder.relative_to(experiments_root).as_posix()
                        evaluations[f"{root_name}/{relative_path}"] = evaluation_folder
    return dict(sorted(evaluations.items()))


def resolve_evaluation_folder(identifier: str, experiments_roots: Dict[str, Path]) -> Path:
    """
    Finds the evaluation folder of an identifier given by the page, kept inside its experiments folder.

    Args:
        identifier (str): The identifier, '<experiments folder name>/<relative path>'.
        experiments_roots (Dict[str, Path]): The experiments folders, keyed by name.

    Returns:
        Path: The evaluation folder.

    Raises:
        ValueError: If the identifier names no served evaluation.
    """
    root_name, _, relative_path = identifier.partition("/")
    if root_name not in experiments_roots or not relative_path:
        raise ValueError(f"Unknown evaluation {identifier!r}.")
    experiments_root = experiments_roots[root_name]
    evaluation_folder = experiments_root / relative_path
    if (not evaluation_folder.resolve().is_relative_to(experiments_root.resolve()) or
            not is_explorer_evaluation(evaluation_folder=evaluation_folder)):
        raise ValueError(f"Unknown evaluation {identifier!r}.")
    return evaluation_folder


def read_saved_temperature_state(evaluation_folder: Path) -> Optional[Dict[str, Any]]:
    """
    Reads the temperature saved on the page of an evaluation and when it was saved, for the page and the index.

    Args:
        evaluation_folder (Path): The evaluation folder.

    Returns:
        Optional[Dict[str, Any]]: The saved 'temperature' and 'saved_at', or None when nothing was saved.
    """
    saved_path = evaluation_folder / SAVED_THRESHOLDS_FILE_NAME
    if not saved_path.is_file():
        return None
    return {"temperature": json.loads(s=saved_path.read_text())["logit_temperature"],
            "saved_at": datetime.fromtimestamp(saved_path.stat().st_mtime).isoformat(sep=" ", timespec="minutes")}


def describe_export_job(export_job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Describes the export run by the server for the page: its state, its folder and the last lines of its console
    output, the colours and links of the console removed.

    Args:
        export_job (Dict[str, Any]): The export started by the server.

    Returns:
        Dict[str, Any]: The 'evaluation', 'state' ('running', 'succeeded' or 'failed'), 'temperature', 'folder',
            'started_at' and 'log' of the export.
    """
    return_code = export_job["process"].poll()
    state = "running" if return_code is None else "succeeded" if return_code == 0 else "failed"

    # The console output moves with the folder once the export succeeded
    log_path = (export_job["export_folder"] / EXPORT_LOG_FILE_NAME if state == "succeeded"
                else export_job["log_path"])
    log_lines = log_path.read_text(errors="replace").splitlines() if log_path.is_file() else []
    log_text = TERMINAL_ESCAPE_PATTERN.sub(repl="", string="\n".join(log_lines[-NUMBER_LOG_LINES:]))

    # Outcome printed once on the console of the server
    if state != "running" and not export_job["reported"]:
        export_job["reported"] = True
        outcome_path = export_job["export_folder"] if state == "succeeded" else log_path
        print_color(color="green" if state == "succeeded" else "red",
                    output=f"Export {state.capitalize()} At T = {export_job['temperature']:g} : "
                           f"{format_path_link(path=outcome_path)}", indent=1)
    return {"evaluation": export_job["evaluation"], "state": state, "temperature": export_job["temperature"],
            "folder": str(export_job["export_folder"]), "started_at": export_job["started_at"], "log": log_text}


def describe_export_availability(evaluation_folder: Path) -> Dict[str, Any]:
    """
    Tells whether the checkpoint of an evaluation can be exported from its page (its run folder and checkpoint still on
    disk), and lists its exports, for the page and the index.

    Args:
        evaluation_folder (Path): The evaluation folder.

    Returns:
        Dict[str, Any]: The 'unavailable' reason (None when the export is possible) and the 'exports' of the checkpoint
            (folder name, path and record).
    """
    checkpoint = describe_evaluation_folder(evaluation_folder=evaluation_folder)
    unavailable = None
    if not (checkpoint["run_folder"] / "training_configuration.yaml").is_file():
        unavailable = f"The run folder {checkpoint['run_folder']} of the evaluation is missing."
    else:
        try:
            get_checkpoint_path(experiment_folder=checkpoint["run_folder"], iteration=checkpoint["iteration"])
        except FileNotFoundError:
            unavailable = f"The checkpoint of iteration {checkpoint['iteration']} is missing from its run folder."
    exports = [{"name": export["folder"].name, "folder": str(export["folder"]), "record": export["record"]}
               for export in list_checkpoint_exports(run_folder=checkpoint["run_folder"],
                                                     iteration=checkpoint["iteration"],
                                                     use_averaged_weights=checkpoint["use_averaged_weights"])]
    return {"checkpoint": checkpoint, "unavailable": unavailable, "exports": exports}


def build_index_rows(evaluations: Dict[str, Path]) -> str:
    """
    Builds the table rows of the index page: for every served evaluation, its run, its evaluation folder (training or
    checkpoint evaluation), iteration, weights, temperatures (any one from its logits, or those of its written page),
    saved temperature and exports, with the link to its page.

    Args:
        evaluations (Dict[str, Path]): The evaluation folders, keyed by identifier.

    Returns:
        str: The HTML rows.
    """
    rows = []
    for identifier, evaluation_folder in evaluations.items():
        checkpoint = describe_evaluation_folder(evaluation_folder=evaluation_folder)
        saved_state = read_saved_temperature_state(evaluation_folder=evaluation_folder)
        exports = list_checkpoint_exports(run_folder=checkpoint["run_folder"], iteration=checkpoint["iteration"],
                                          use_averaged_weights=checkpoint["use_averaged_weights"])
        evaluation_name = (evaluation_folder.parent if checkpoint["use_averaged_weights"] else
                           evaluation_folder).parent.name
        cells = [f'<a href="/explorer?evaluation={quote(identifier)}">{html.escape(checkpoint["run_folder"].name)}</a>',
                 "Checkpoint evaluation" if evaluation_name == CHECKPOINT_EVALUATION_FOLDER_NAME else "Training",
                 str(checkpoint["iteration"]), "EMA" if checkpoint["use_averaged_weights"] else "Trained",
                 "Any" if (evaluation_folder / CLASS_LOGITS_FILE_NAME).is_file() else "Grid only",
                 "—" if saved_state is None else f"{saved_state['temperature']:g}",
                 ", ".join(html.escape(export["folder"].name) for export in exports) or "—",
                 html.escape(identifier.partition("/")[0])]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    return "\n".join(rows)


class TemperatureExplorerHandler(BaseHTTPRequestHandler):
    """
    Requests of the index, the explorer pages and their buttons: the pages are built again for every visit, the saved
    thresholds and exports are checked before anything is written, and a refused request gets its reason.
    """

    server: TemperatureExplorerServer

    def send_body(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        """
        Answers a request with a body of the given type.

        Args:
            status (HTTPStatus): The status of the answer.
            body (bytes): The body.
            content_type (str): The media type of the body.
        """
        self.send_response(code=status)
        self.send_header(keyword="Content-Type", value=content_type)
        self.send_header(keyword="Content-Length", value=str(len(body)))
        self.send_header(keyword="Cache-Control", value="no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: HTTPStatus, content: Dict[str, Any]) -> None:
        """
        Answers a request of the page with a JSON body, read by the page to show the result or the reason of a refusal.

        Args:
            status (HTTPStatus): The status of the answer.
            content (Dict[str, Any]): The body of the answer.
        """
        self.send_body(status=status, body=json.dumps(obj=content).encode(encoding="utf-8"),
                       content_type="application/json")

    def read_query(self) -> Dict[str, str]:
        """
        Reads the parameters of the address of a request, each given once.

        Returns:
            Dict[str, str]: The parameters.
        """
        return {name: values[0] for name, values in parse_qs(qs=urlparse(url=self.path).query).items()}

    def read_json_body(self) -> Dict[str, Any]:
        """
        Reads the JSON body posted by the page.

        Returns:
            Dict[str, Any]: The posted object.

        Raises:
            ValueError: If the body is not a JSON object.
        """
        body = json.loads(s=self.rfile.read(int(self.headers["Content-Length"])))
        if not isinstance(body, dict):
            raise ValueError("Expected a JSON object.")
        return body

    def build_page_data(self, identifier: str, evaluation_folder: Path) -> Dict[str, Any]:
        """
        Builds the data of the page of an evaluation, from its class logits (kept until they change) or from its written
        page, with what the served page needs: its identifier, whether any temperature can be shown, and the export
        settings it offers.

        Args:
            identifier (str): The identifier of the evaluation.
            evaluation_folder (Path): The evaluation folder.

        Returns:
            Dict[str, Any]: The data of the page.
        """
        logits_path = evaluation_folder / CLASS_LOGITS_FILE_NAME
        if logits_path.is_file():
            cache_key = (identifier, logits_path.stat().st_mtime)
            if cache_key not in self.server.page_data_cache:
                self.server.page_data_cache[cache_key] = build_explorer_data_from_logits(
                    class_logits_record=load_class_logits(evaluation_folder=evaluation_folder))
            page_data = dict(self.server.page_data_cache[cache_key])
        else:
            page_data = read_explorer_page_data(explorer_path=evaluation_folder / EXPLORER_FILE_NAME)
        return {**page_data, "evaluation": identifier, "live_temperatures": logits_path.is_file(),
                "export_formats": EXPORT_FORMATS, "tensorrt_precisions": TENSORRT_PRECISIONS,
                "export_defaults": DEFAULT_EXPORT_SETTINGS}

    def describe_status(self, identifier: str, evaluation_folder: Path) -> Dict[str, Any]:
        """
        Describes the deployment state of an evaluation for its page: the saved temperature, the export folder it would
        be written to, the exports of the checkpoint, the export running for it, and whether the export button is
        available.

        Args:
            identifier (str): The identifier of the evaluation.
            evaluation_folder (Path): The evaluation folder.

        Returns:
            Dict[str, Any]: The status of the evaluation.
        """
        saved_state = read_saved_temperature_state(evaluation_folder=evaluation_folder)
        availability = describe_export_availability(evaluation_folder=evaluation_folder)
        checkpoint = availability["checkpoint"]
        next_export_folder = None if saved_state is None else str(build_export_folder(
            run_folder=checkpoint["run_folder"], iteration=checkpoint["iteration"],
            use_averaged_weights=checkpoint["use_averaged_weights"], logit_temperature=saved_state["temperature"]))

        # Export of this evaluation, or another one keeping the server busy
        with self.server.export_lock:
            export_job = self.server.export_job
            job_description = None if export_job is None else describe_export_job(export_job=export_job)
        return {"saved": saved_state, "next_export_folder": next_export_folder, "exports": availability["exports"],
                "export_unavailable": availability["unavailable"],
                "export_job": job_description if job_description and job_description["evaluation"] == identifier
                else None,
                "busy": job_description is not None and job_description["state"] == "running"}

    def is_exporting(self, identifier: str) -> bool:
        """
        Tells whether the export of an evaluation is running, during which its saved thresholds must not change.

        Args:
            identifier (str): The identifier of the evaluation.

        Returns:
            bool: True while its export runs.
        """
        with self.server.export_lock:
            export_job = self.server.export_job
            return (export_job is not None and export_job["evaluation"] == identifier and
                    export_job["process"].poll() is None)

    def do_GET(self) -> None:
        """
        Answers the index ('/'), the page of an evaluation ('/explorer'), the description of a typed temperature
        ('/api/temperature'), the images of every class counted in score ranges at a temperature ('/api/score_ranges')
        and the deployment status of an evaluation ('/api/status').
        """
        route, query = urlparse(url=self.path).path, self.read_query()
        try:
            if route == "/":
                evaluations = find_explorer_evaluations(experiments_roots=self.server.experiments_roots)
                page = INDEX_TEMPLATE_PATH.read_text().replace(INDEX_ROWS_PLACEHOLDER,
                                                               build_index_rows(evaluations=evaluations))
                self.send_body(status=HTTPStatus.OK, body=page.encode(encoding="utf-8"),
                               content_type="text/html; charset=utf-8")
                return
            if route not in ["/explorer", "/api/temperature", "/api/score_ranges", "/api/status"]:
                self.send_json(status=HTTPStatus.NOT_FOUND, content={"error": f"Unknown route {route}."})
                return
            identifier = query["evaluation"] if "evaluation" in query else ""
            evaluation_folder = resolve_evaluation_folder(identifier=identifier,
                                                          experiments_roots=self.server.experiments_roots)
        except ValueError as error:
            self.send_json(status=HTTPStatus.NOT_FOUND, content={"error": str(error)})
            return

        # Page of the evaluation, built with the current template
        if route == "/explorer":
            page_data = self.build_page_data(identifier=identifier, evaluation_folder=evaluation_folder)
            page = render_explorer_page(explorer_data=page_data)
            self.send_body(status=HTTPStatus.OK, body=page.encode(encoding="utf-8"),
                           content_type="text/html; charset=utf-8")
        # Temperature typed on the page, described from the class logits for the classes the page draws
        elif route == "/api/temperature":
            try:
                if not (evaluation_folder / CLASS_LOGITS_FILE_NAME).is_file():
                    raise ValueError(f"No {CLASS_LOGITS_FILE_NAME} for this evaluation: only the temperatures of its "
                                     f"slider can be shown. Evaluate the checkpoint again to show any temperature.")
                logit_temperature = check_logit_temperature(logit_temperature=float(query["temperature"]))
                page_data = self.build_page_data(identifier=identifier, evaluation_folder=evaluation_folder)
                state = describe_explorer_temperature(
                    class_logits_record=load_class_logits(evaluation_folder=evaluation_folder),
                    shown_class_indices=page_data["class_indices"], logit_temperature=logit_temperature)
            except (ValueError, KeyError) as error:
                self.send_json(status=HTTPStatus.BAD_REQUEST, content={"error": str(error)})
                return
            self.send_json(status=HTTPStatus.OK, content=state)
        # Score ranges typed on the page, the images of every drawn class counted from the class logits at the
        # temperature shown, the cut points given as '0.5,0.8'
        elif route == "/api/score_ranges":
            try:
                if not (evaluation_folder / CLASS_LOGITS_FILE_NAME).is_file():
                    raise ValueError(f"No {CLASS_LOGITS_FILE_NAME} for this evaluation: the score ranges are counted "
                                     f"from the logits. Evaluate the checkpoint again to count them.")
                logit_temperature = check_logit_temperature(logit_temperature=float(query["temperature"]))
                cut_points = check_score_cut_points(cut_points=[float(cut_point) for cut_point in
                                                                query["cut_points"].split(",")])
                page_data = self.build_page_data(identifier=identifier, evaluation_folder=evaluation_folder)
                score_ranges = compute_score_ranges(
                    class_logits_record=load_class_logits(evaluation_folder=evaluation_folder),
                    shown_class_indices=page_data["class_indices"], logit_temperature=logit_temperature,
                    cut_points=cut_points)
            except (ValueError, KeyError) as error:
                self.send_json(status=HTTPStatus.BAD_REQUEST, content={"error": str(error)})
                return
            self.send_json(status=HTTPStatus.OK, content=score_ranges)
        else:
            self.send_json(status=HTTPStatus.OK,
                           content=self.describe_status(identifier=identifier, evaluation_folder=evaluation_folder))

    def do_POST(self) -> None:
        """
        Saves the thresholds of a temperature ('/api/save') or starts the export of the checkpoint at its saved
        temperature ('/api/export'), refused with the reason when the request cannot be carried out.
        """
        route = urlparse(url=self.path).path
        if route not in ["/api/save", "/api/export"]:
            self.send_json(status=HTTPStatus.NOT_FOUND, content={"error": f"Unknown route {route}."})
            return
        try:
            body = self.read_json_body()
            identifier = body["evaluation"]
            evaluation_folder = resolve_evaluation_folder(identifier=identifier,
                                                          experiments_roots=self.server.experiments_roots)
            if route == "/api/save":
                self.save_thresholds(identifier=identifier, evaluation_folder=evaluation_folder, body=body)
            else:
                self.start_export(identifier=identifier, evaluation_folder=evaluation_folder, body=body)
        except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
            self.send_json(status=HTTPStatus.BAD_REQUEST,
                           content={"error": TERMINAL_ESCAPE_PATTERN.sub(repl="", string=str(error))})

    def save_thresholds(self, identifier: str, evaluation_folder: Path, body: Dict[str, Any]) -> None:
        """
        Saves the thresholds of the temperature shown on the page next to it: found here from the class logits for
        every class, or, for a page without logits, the thresholds it posted, checked against its
        class_thresholds.json. Refused while the checkpoint is being exported.

        Args:
            identifier (str): The identifier of the evaluation.
            evaluation_folder (Path): The evaluation folder.
            body (Dict[str, Any]): The posted 'temperature', and the 'thresholds' of a page without logits.
        """
        if self.is_exporting(identifier=identifier):
            self.send_json(status=HTTPStatus.CONFLICT,
                           content={"error": "The checkpoint is being exported: save once the export has ended."})
            return
        logit_temperature = check_logit_temperature(logit_temperature=body["temperature"])
        if (evaluation_folder / CLASS_LOGITS_FILE_NAME).is_file():
            saved_thresholds = compute_temperature_thresholds(
                class_logits_record=load_class_logits(evaluation_folder=evaluation_folder),
                logit_temperature=logit_temperature)
        else:
            saved_thresholds = body["thresholds"]
            if (not isinstance(saved_thresholds, dict) or "logit_temperature" not in saved_thresholds or
                    saved_thresholds["logit_temperature"] != logit_temperature):
                raise ValueError("Expected the posted thresholds of the posted temperature.")
        saved_path = write_saved_thresholds(evaluation_folder=evaluation_folder, saved_thresholds=saved_thresholds)
        print_color(color="green", output=f"Thresholds At T = {logit_temperature:g} Saved : "
                                          f"{format_path_link(path=saved_path)}", indent=1)
        self.send_json(status=HTTPStatus.OK, content={"path": str(saved_path), "temperature": logit_temperature})

    def start_export(self, identifier: str, evaluation_folder: Path, body: Dict[str, Any]) -> None:
        """
        Starts the export of the checkpoint of the page at its saved temperature, with the settings chosen on the page,
        as a separate process writing its console output next to the export; one export runs at a time.

        Args:
            identifier (str): The identifier of the evaluation.
            evaluation_folder (Path): The evaluation folder.
            body (Dict[str, Any]): The posted 'formats', 'model_version', 'maximum_batch_size' and
                'tensorrt_precision'.
        """
        with self.server.export_lock:
            export_job = self.server.export_job
            if export_job is not None and export_job["process"].poll() is None:
                self.send_json(status=HTTPStatus.CONFLICT,
                               content={"error": f"An export is running: {export_job['export_folder']}."})
                return

            # Job written next to the export, then the export process, its console output in the log file
            job = prepare_export_job(evaluation_folder=evaluation_folder, formats=body["formats"],
                                     model_version=body["model_version"],
                                     maximum_batch_size=body["maximum_batch_size"],
                                     tensorrt_precision=body["tensorrt_precision"])
            with open(file=job["log_path"], mode="w", encoding="utf-8") as log_file:
                process = subprocess.Popen(args=[sys.executable, "-u", "-m", EXPORT_MODULE, str(job["job_path"])],
                                           cwd=REPOSITORY_ROOT, stdout=log_file, stderr=subprocess.STDOUT,
                                           env={**os.environ, "PYTHONUNBUFFERED": "1"})
            self.server.export_job = {"evaluation": identifier, "process": process, "log_path": job["log_path"],
                                      "export_folder": job["export_folder"], "temperature": job["logit_temperature"],
                                      "started_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
                                      "reported": False}
        print_color(color="blue", output=f"Export Started At T = {job['logit_temperature']:g} : "
                                         f"{format_path_link(path=job['export_folder'])}", indent=1)
        self.send_json(status=HTTPStatus.OK, content={"folder": str(job["export_folder"]),
                                                      "temperature": job["logit_temperature"]})

    def log_message(self, message_format: str, *args: Any) -> None:
        """
        Keeps the console of the server to the saves and exports, without a line for every request of the page.

        Args:
            message_format (str): The format of the request line.
            *args (Any): Its values.
        """


def serve_temperature_explorers(experiments_roots: List[Path], port: int) -> None:
    """
    Serves the temperature explorers of the experiments folders on 127.0.0.1 until the script is stopped, the index at
    the printed address.

    Args:
        experiments_roots (List[Path]): The experiments folders, of distinct folder names.
        port (int): The local port of the server.
    """
    server = TemperatureExplorerServer(server_address=("127.0.0.1", port), experiments_roots=experiments_roots)
    print_color(color="blue", output=f"Temperature Explorers : http://127.0.0.1:{port}/", add_separators=True)
    for experiments_root in experiments_roots:
        print_color(color="blue", output=f"Experiments Folder : {format_path_link(path=experiments_root)}", indent=1)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print_color(color="yellow", output="Temperature Explorers Stopped", indent=1)
    finally:
        server.server_close()


if __name__ == "__main__":
    # Experiments folders whose evaluations are served: the folders of the repository root whose name holds
    # 'experiments', with the evaluations of the trainings and of evaluate_checkpoint.py in their run folders
    experiments_roots = sorted(folder for folder in REPOSITORY_ROOT.iterdir()
                               if folder.is_dir() and "experiments" in folder.name)
    # Local port of the index, opened at http://127.0.0.1:<port>/, not 8090 so that the server runs next to the
    # inference explorer of the ficha repositories, which takes 8090
    port = 8091

    serve_temperature_explorers(experiments_roots=experiments_roots, port=port)
