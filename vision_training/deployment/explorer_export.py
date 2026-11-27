"""
Export of a classifier launched from the export button of its temperature explorer, served by
scripts/evaluation/serve_temperature_explorer.py. The evaluation folder of the page gives the run, the iteration and the
weights, the thresholds saved on the page give the temperature, and export_experiment (export_model.py) writes the
classifier into a Triton model repository of its own, '<run>/Exports/Iteration_<n>[_EMA]_T<temperature>/', so that the
exports of several classifiers and temperatures never overwrite one another, an export of the same temperature replacing
the earlier one but for its TorchScript file of the other precision ('model.pt' or 'model_fp16.pt'), kept next to the
new one. The serve script prepares the job (prepare_export_job) and runs it as a separate process
('python -m vision_training.deployment.explorer_export <job file>'), so that the GPU memory of the export is freed when
it ends and its console output, written to 'export_log.txt', is shown on the page.
"""

import json
import re
import shutil
import sys

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# The settings the page can choose (EXPORT_FORMATS, TENSORRT_PRECISIONS and TORCHSCRIPT_PRECISIONS, the export refusing
# any other value) and those it shows first (DEFAULT_EXPORT_SETTINGS, the defaults of the arguments of export_model.py)
# are imported here for the serve script too
from vision_training.deployment.export_model import (DEFAULT_EXPORT_SETTINGS, check_export_settings, export_experiment,
                                                     hide_known_export_warnings)
from vision_training.deployment.triton_configuration import (EXPORT_FORMATS, TENSORRT_PRECISIONS,
                                                             TORCHSCRIPT_FILE_NAMES, TORCHSCRIPT_PRECISIONS)
from vision_training.evaluation.checkpoint_evaluation import CHECKPOINT_EVALUATION_FOLDER_NAME
from vision_training.evaluation.classification.temperature_explorer import (SAVED_THRESHOLDS_FILE_NAME,
                                                                            check_logit_temperature)
from vision_training.trainers.base_trainer import AVERAGED_WEIGHTS_FOLDER_NAME, EVALUATION_FOLDER_NAME
from vision_training.trainers.checkpointing import get_checkpoint_path
from vision_training.utilities.os_utilities import format_path_link, print_color

# Folder of the exports of a run, and the files of every export: the job written by the serve script, the console
# output of the export and the record of what was exported, the export being written to '<folder>.partial' first
EXPORTS_FOLDER_NAME = "Exports"
PARTIAL_EXPORT_SUFFIX = ".partial"
EXPORT_JOB_FILE_NAME = "export_job.json"
EXPORT_LOG_FILE_NAME = "export_log.txt"
EXPORT_RECORD_FILE_NAME = "export_record.json"


def describe_evaluation_folder(evaluation_folder: Path) -> Dict[str, Any]:
    """
    Reads the run, iteration and weights of an evaluation folder of a classifier, the checkpoint exported from its
    temperature explorer: '<run>/Evaluation/Iteration_<n>' (evaluation of the training) or
    '<run>/Checkpoint_Evaluation/Iteration_<n>' (scripts/evaluation/evaluate_checkpoint.py), followed by 'EMA' for the
    moving average of the weights.

    Args:
        evaluation_folder (Path): The evaluation folder of the checkpoint.

    Returns:
        Dict[str, Any]: The 'run_folder' (holding the weights), the 'iteration' and 'use_averaged_weights'.

    Raises:
        ValueError: If the folder is not laid out as an evaluation folder.
    """
    # The averaged weights are evaluated in the 'EMA' subfolder of the iteration
    use_averaged_weights = evaluation_folder.name == AVERAGED_WEIGHTS_FOLDER_NAME
    iteration_folder = evaluation_folder.parent if use_averaged_weights else evaluation_folder
    iteration_match = re.fullmatch(pattern=r"Iteration_(\d+)", string=iteration_folder.name)
    if iteration_match is None or iteration_folder.parent.name not in [EVALUATION_FOLDER_NAME,
                                                                       CHECKPOINT_EVALUATION_FOLDER_NAME]:
        raise ValueError(f"Expected an evaluation folder <run>/[{EVALUATION_FOLDER_NAME} or "
                         f"{CHECKPOINT_EVALUATION_FOLDER_NAME}]/Iteration_<n>[/{AVERAGED_WEIGHTS_FOLDER_NAME}], got "
                         f"{evaluation_folder}.")

    # Both evaluation folders sit in the run folder
    return {"run_folder": iteration_folder.parent.parent, "iteration": int(iteration_match.group(1)),
            "use_averaged_weights": use_averaged_weights}


def build_export_folder(run_folder: Path, iteration: int, use_averaged_weights: bool,
                        logit_temperature: float) -> Path:
    """
    Names the Triton model repository receiving the export of a checkpoint at a temperature,
    '<run>/Exports/Iteration_<n>[_EMA]_T<temperature>', so that every checkpoint, weights and temperature gets its own
    folder.

    Args:
        run_folder (Path): The exported run folder.
        iteration (int): The exported iteration.
        use_averaged_weights (bool): True when the moving average of the weights is exported.
        logit_temperature (float): The temperature built into the exported classifier.

    Returns:
        Path: The export folder.
    """
    weights_suffix = f"_{AVERAGED_WEIGHTS_FOLDER_NAME}" if use_averaged_weights else ""
    return run_folder / EXPORTS_FOLDER_NAME / f"Iteration_{iteration}{weights_suffix}_T{logit_temperature:g}"


def list_checkpoint_exports(run_folder: Path, iteration: int, use_averaged_weights: bool) -> List[Dict[str, Any]]:
    """
    Lists the finished exports of a checkpoint and weights in the exports folder of its run, newest first, for the page
    and the index of the serve script, every export described by its record.

    Args:
        run_folder (Path): The run folder.
        iteration (int): The iteration of the checkpoint.
        use_averaged_weights (bool): True for the exports of the moving average of the weights.

    Returns:
        List[Dict[str, Any]]: The 'folder' (path) and 'record' (export_record.json, None when missing) of every export.
    """
    weights_suffix = f"_{AVERAGED_WEIGHTS_FOLDER_NAME}" if use_averaged_weights else ""
    exports_folder = run_folder / EXPORTS_FOLDER_NAME
    if not exports_folder.is_dir():
        return []

    # Finished exports of the checkpoint and weights only, the folders still written ending with '.partial'
    exports = []
    for export_folder in exports_folder.glob(pattern=f"Iteration_{iteration}{weights_suffix}_T*"):
        if not export_folder.is_dir() or export_folder.name.endswith(PARTIAL_EXPORT_SUFFIX):
            continue
        record_path = export_folder / EXPORT_RECORD_FILE_NAME
        exports.append({"folder": export_folder,
                        "record": json.loads(s=record_path.read_text()) if record_path.is_file() else None})
    return sorted(exports, key=lambda export: export["folder"].stat().st_mtime, reverse=True)


def prepare_export_job(evaluation_folder: Path, formats: List[str], model_version: int, maximum_batch_size: int,
                       tensorrt_precision: Optional[str], torchscript_precision: str) -> Dict[str, Any]:
    """
    Prepares the export of the checkpoint of a temperature explorer at its saved temperature: the settings and the saved
    thresholds are checked, the checkpoint is found in the run folder, an unfinished folder of an earlier export of the
    same temperature is removed, and the job is written to '<export folder>.partial/export_job.json', for run_export_job
    run as a separate process by the serve script.

    Args:
        evaluation_folder (Path): The evaluation folder of the exported checkpoint, holding the saved thresholds.
        formats (List[str]): The export formats, among 'torchscript' and 'onnx'.
        model_version (int): The Triton version of the exported models.
        maximum_batch_size (int): The largest batch size of the exported models.
        tensorrt_precision (Optional[str]): 'FP16' or 'FP32' to accelerate the ONNX model with TensorRT, or None.
        torchscript_precision (str): 'FP32' or 'FP16', the precision of the network traced into the TorchScript file.

    Returns:
        Dict[str, Any]: The job: 'job_path', 'work_folder', 'export_folder', 'log_path' and 'logit_temperature'.

    Raises:
        FileNotFoundError: If no thresholds were saved on the page, or the run or its checkpoint is missing.
        ValueError: If a setting or the saved temperature is not one the export accepts.
    """
    check_export_settings(formats=formats, model_version=model_version, maximum_batch_size=maximum_batch_size,
                          tensorrt_precision=tensorrt_precision, torchscript_precision=torchscript_precision)
    saved_thresholds_path = evaluation_folder / SAVED_THRESHOLDS_FILE_NAME
    if not saved_thresholds_path.is_file():
        raise FileNotFoundError(f"No {SAVED_THRESHOLDS_FILE_NAME} in {evaluation_folder}: save the thresholds of a "
                                f"temperature first.")
    logit_temperature = check_logit_temperature(
        logit_temperature=json.loads(s=saved_thresholds_path.read_text())["logit_temperature"])

    # Run, checkpoint and weights of the evaluation, the run folder holding the configuration and the weights
    checkpoint = describe_evaluation_folder(evaluation_folder=evaluation_folder)
    if not (checkpoint["run_folder"] / "training_configuration.yaml").is_file():
        raise FileNotFoundError(f"No training_configuration.yaml in the run folder {checkpoint['run_folder']} of the "
                                f"evaluation.")
    get_checkpoint_path(experiment_folder=checkpoint["run_folder"], iteration=checkpoint["iteration"])

    # Folder written by the export, an unfinished one of the same temperature removed, then the job
    export_folder = build_export_folder(run_folder=checkpoint["run_folder"], iteration=checkpoint["iteration"],
                                        use_averaged_weights=checkpoint["use_averaged_weights"],
                                        logit_temperature=logit_temperature)
    work_folder = export_folder.with_name(export_folder.name + PARTIAL_EXPORT_SUFFIX)
    if work_folder.exists():
        shutil.rmtree(path=work_folder)
    work_folder.mkdir(parents=True)
    job = {"evaluation_folder": str(evaluation_folder), "saved_thresholds_path": str(saved_thresholds_path),
           "run_folder": str(checkpoint["run_folder"]), "iteration": checkpoint["iteration"],
           "use_averaged_weights": checkpoint["use_averaged_weights"], "logit_temperature": logit_temperature,
           "formats": formats, "model_version": model_version, "maximum_batch_size": maximum_batch_size,
           "tensorrt_precision": tensorrt_precision, "torchscript_precision": torchscript_precision,
           "export_folder": str(export_folder)}
    job_path = work_folder / EXPORT_JOB_FILE_NAME
    job_path.write_text(data=json.dumps(obj=job, indent=4))
    return {"job_path": job_path, "work_folder": work_folder, "export_folder": export_folder,
            "log_path": work_folder / EXPORT_LOG_FILE_NAME, "logit_temperature": logit_temperature}


def keep_other_torchscript_files(previous_folder: Path, work_folder: Path, formats: List[str], model_version: int,
                                 torchscript_precision: str) -> List[str]:
    """
    Copies into a new export of a temperature (run_export_job) the TorchScript files of the other precision that the
    earlier export of the same temperature holds in the same version folder, so that 'model.pt' (FP32) and
    'model_fp16.pt' (FP16) of one checkpoint, weights and temperature sit side by side, the config.pbtxt of the new
    export naming the file Triton serves. Nothing is kept when the new export writes no TorchScript model or another
    version, the earlier folder being replaced whole as before.

    Args:
        previous_folder (Path): The folder of the earlier export of the temperature, which may not exist.
        work_folder (Path): The '.partial' folder of the new export, its models written.
        formats (List[str]): The formats of the new export.
        model_version (int): The Triton version of the new export.
        torchscript_precision (str): The TorchScript precision of the new export, whose file it wrote itself.

    Returns:
        List[str]: The TorchScript precisions whose file was kept from the earlier export.
    """
    if "torchscript" not in formats or not previous_folder.is_dir():
        return []

    # Every TorchScript model of the new export, its file of the other precision taken from the same version folder
    kept_precisions = []
    for model_directory in sorted(work_folder.glob(pattern="*_torchscript")):
        previous_version_directory = previous_folder / model_directory.name / str(model_version)
        for precision, file_name in TORCHSCRIPT_FILE_NAMES.items():
            previous_file = previous_version_directory / file_name
            if precision != torchscript_precision and previous_file.is_file():
                shutil.copyfile(src=previous_file, dst=model_directory / str(model_version) / file_name)
                kept_precisions.append(precision)
    return kept_precisions


def run_export_job(job_path: Path) -> Path:
    """
    Runs an export prepared by prepare_export_job: the classifier is exported into the folder of the job with the
    saved thresholds of its page (export_experiment with use_saved_temperature), the TorchScript file of the other
    precision is kept from the earlier export of the same temperature (keep_other_torchscript_files), the record of
    the export is written next to the models, and the folder replaces the export of the same temperature, if any. A
    failed export leaves the '.partial' folder with its console output, removed by the next export of that
    temperature, the earlier export staying as it was.

    Args:
        job_path (Path): The 'export_job.json' of the export, in its '.partial' folder.

    Returns:
        Path: The export folder.

    Raises:
        ValueError: If the thresholds saved on the page changed temperature since the job was prepared.
    """
    job = json.loads(s=job_path.read_text())
    work_folder, export_folder = job_path.parent, Path(job["export_folder"])
    saved_thresholds_path = Path(job["saved_thresholds_path"])

    # The temperature of the folder name must be the one exported
    saved_temperature = json.loads(s=saved_thresholds_path.read_text())["logit_temperature"]
    if saved_temperature != job["logit_temperature"]:
        raise ValueError(f"The thresholds of {saved_thresholds_path} were saved at T = {saved_temperature:g} after the "
                         f"export at T = {job['logit_temperature']:g} was prepared: export again.")

    # Export into the folder of the job, with the thresholds of the page
    hide_known_export_warnings()
    export_experiment(experiment_folder=Path(job["run_folder"]), iteration=job["iteration"],
                      use_averaged_weights=job["use_averaged_weights"], formats=job["formats"], repository=work_folder,
                      model_version=job["model_version"], maximum_batch_size=job["maximum_batch_size"],
                      tensorrt_precision=job["tensorrt_precision"],
                      torchscript_precision=job["torchscript_precision"], use_saved_temperature=True,
                      saved_thresholds_path=saved_thresholds_path, test_split_parity=False)

    # TorchScript file of the other precision of the same checkpoint, weights and temperature kept next to the new one
    kept_torchscript_precisions = keep_other_torchscript_files(
        previous_folder=export_folder, work_folder=work_folder, formats=job["formats"],
        model_version=job["model_version"], torchscript_precision=job["torchscript_precision"])

    # Record of what was exported, then the finished folder in place of an earlier export of the same temperature
    record = {**{key: job[key] for key in ["evaluation_folder", "saved_thresholds_path", "run_folder", "iteration",
                                           "use_averaged_weights", "logit_temperature", "formats", "model_version",
                                           "maximum_batch_size", "tensorrt_precision", "torchscript_precision"]},
              "kept_torchscript_precisions": kept_torchscript_precisions,
              "exported_at": datetime.now().isoformat(timespec="seconds")}
    (work_folder / EXPORT_RECORD_FILE_NAME).write_text(data=json.dumps(obj=record, indent=4))
    job_path.unlink()
    if export_folder.exists():
        shutil.rmtree(path=export_folder)
    work_folder.rename(target=export_folder)
    print_color(color="green", output=f"Exported At T = {job['logit_temperature']:g} : "
                                      f"{format_path_link(path=export_folder)}", indent=1)
    return export_folder


if __name__ == "__main__":
    # Job file written by prepare_export_job, given by scripts/evaluation/serve_temperature_explorer.py
    run_export_job(job_path=Path(sys.argv[1]))
