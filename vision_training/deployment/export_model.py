"""
Exports a trained experiment of any task to the Triton model repository as TorchScript (its network in FP32 or FP16)
and ONNX (optionally run with TensorRT), writes the Triton configurations and checks every export against the eager
inference model, in a staging folder moved into the repository only when every format matches, adding for the frame
selection the decoder and the ensembles returning the chosen frames, and for the classification the logit temperature
and the class thresholds saved from the temperature explorer of the exported checkpoint. Every exported model carries
the input contract of the run (deployment/input_contract.py), in the metadata of the ONNX file and in
'preprocessing.json' next to it, so that a server builds the input from a photo as the training did. An ONNX model
served with TensorRT is also checked as Triton will run it (deployment/tensorrt_check.py): in FP16, a CPU scan of its
graph for values above the largest FP16 value, then, unless skip_tensorrt_check, its engine built and run by a
temporary Triton container on the check batches, compared with the eager model. With test_split_parity, every
format is also served by a temporary Triton container and scored on the whole test split against the eager model
(deployment/export_parity.py), the summary written to '<run>/Export_Parity/Iteration_<n>[_EMA][_T<temperature>]_
<precisions>/' and next to the installed models. Run as a script with the run folder and the export settings given as
arguments (parse_export_arguments), its help printed when none is given.
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
import numpy as np
import onnxruntime
import torch

from pathlib import Path
from torch import nn
from typing import Any, Dict, List, Optional

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.deployment.export_formats import (convert_to_half_precision, export_onnx, export_torchscript,
                                                       hide_known_export_warnings)
from vision_training.deployment.export_parity import run_test_split_parity, write_parity_summary
from vision_training.deployment.export_specifications import (build_inference_model, convert_outputs_to_numpy,
                                                              create_random_inputs, get_export_specification)
from vision_training.deployment.frame_selection_deployment import (build_decoding_settings,
                                                                   export_frame_selection_decoding)
from vision_training.deployment.input_contract import (build_input_contract, write_onnx_input_contract,
                                                       write_preprocessing_file)
from vision_training.deployment.tensorrt_check import (HALF_PRECISION_LARGEST_VALUE, check_tensorrt_engine,
                                                       describe_half_precision_overflows,
                                                       find_half_precision_overflows)
from vision_training.deployment.triton_configuration import (EXPORT_FORMATS, TENSORRT_PRECISIONS,
                                                             TORCHSCRIPT_FILE_NAMES, TORCHSCRIPT_PRECISIONS,
                                                             write_triton_configuration)
from vision_training.deployment.triton_server import check_triton_available
from vision_training.evaluation.checkpoint_evaluation import CHECKPOINT_EVALUATION_FOLDER_NAME
from vision_training.evaluation.classification.temperature_explorer import (SAVED_THRESHOLDS_FILE_NAME,
                                                                            check_saved_thresholds)
from vision_training.trainers.base_trainer import AVERAGED_WEIGHTS_FOLDER_NAME, EVALUATION_FOLDER_NAME
from vision_training.trainers.checkpointing import get_checkpoint_path, get_last_checkpoint_iteration
from vision_training.models.model_loading import load_trained_model
from vision_training.utilities.os_utilities import format_path_link, print_color

# Default Triton model repository, mounted by deployment/triton/docker-compose.yml
DEFAULT_REPOSITORY = Path(__file__).absolute().parent / "triton" / "served_models"

# Export settings taken when the command line of the script (parse_export_arguments) or the export form of the
# temperature explorer (explorer_export.py) does not give them
DEFAULT_EXPORT_SETTINGS = {"formats": ["torchscript", "onnx"], "model_version": 1, "maximum_batch_size": 16,
                           "tensorrt_precision": "FP32", "torchscript_precision": "FP32"}

# Staging folder of an export, created next to the model repository, where every format is exported and checked
# before install_staged_models moves the models into the repository, and its folders of the exported models and of
# the version folders they replace, removed with it
STAGING_FOLDER_PREFIX = ".export_staging_"
STAGED_MODELS_FOLDER_NAME = "models"
REPLACED_VERSIONS_FOLDER_NAME = "replaced_versions"

# Folder of the staging folder receiving the TensorRT engines built by the Triton containers of the TensorRT engine
# check and of the test split parity, which reuses the engines of the check, removed with it, and the file of the
# staging folder receiving the server log of the engine check
STAGED_ENGINE_CACHE_FOLDER_NAME = "tensorrt_engine_cache"
TENSORRT_CHECK_LOG_NAME = "tensorrt_check_triton_server.log"

# Sentence ending the error of an export whose TensorRT engine check cannot start a Triton container
TENSORRT_CHECK_REMEDY = ("Export with --skip_tensorrt_check to install the ONNX model without running its TensorRT "
                         "engine, or with --tensorrt_precision None to serve it with ONNX Runtime CUDA without "
                         "TensorRT.")

# Factor of the difference tolerances of the random batch check of an FP16 TorchScript export, compared with the eager
# FP16 copy it was traced from: both run the same FP16 network but may pick other GPU kernels, whose FP16 roundings
# differ by up to about 1e-3 (7.3e-4 on the candidate scores of the keypoint test model) against tolerances of 1e-4, so
# that the check still catches a broken trace without failing on rounding; its cost against FP32 is that of the parity
HALF_PRECISION_TOLERANCE_FACTOR = 100


def find_saved_thresholds_path(experiment_folder: Path, checkpoint_path: Path, use_averaged_weights: bool) -> Path:
    """
    Finds the thresholds saved from the temperature explorer of the exported checkpoint, whose temperature is built
    into an exported classifier with use_saved_temperature: 'saved_temperature_thresholds.json' in the evaluation
    folder of the exported iteration and weights, '<run>/Evaluation/Iteration_<n>' (evaluation of the training) or
    '<run>/Checkpoint_Evaluation/Iteration_<n>' (scripts/evaluation/evaluate_checkpoint.py), its 'EMA' subfolder for
    the averaged weights. Exactly one of them must hold the file, so that the deployed temperature is never taken from
    another evaluation than the one chosen.

    Args:
        experiment_folder (Path): The exported run folder.
        checkpoint_path (Path): The exported checkpoint, '<run>/Weights/Iteration_<n>/model_<n>.pt'.
        use_averaged_weights (bool): True when the moving average of the weights is exported.

    Returns:
        Path: The saved thresholds of the exported checkpoint and weights.

    Raises:
        FileNotFoundError: If none of the two evaluation folders holds saved thresholds.
        ValueError: If both hold saved thresholds.
    """
    # Evaluation folder of the exported iteration and weights, of the training then of the standalone evaluation
    weights_folders = [AVERAGED_WEIGHTS_FOLDER_NAME] if use_averaged_weights else []
    candidate_paths = [Path(experiment_folder, evaluation_folder_name, checkpoint_path.parent.name, *weights_folders,
                            SAVED_THRESHOLDS_FILE_NAME)
                       for evaluation_folder_name in [EVALUATION_FOLDER_NAME, CHECKPOINT_EVALUATION_FOLDER_NAME]]
    found_paths = [candidate_path for candidate_path in candidate_paths if candidate_path.is_file()]
    if not found_paths:
        raise FileNotFoundError(f"No {SAVED_THRESHOLDS_FILE_NAME} for the exported checkpoint: save the thresholds of "
                                f"a temperature with the button of its temperature explorer, served by "
                                f"scripts/evaluation/serve_temperature_explorer.py. Looked for: " +
                                ", ".join(str(candidate_path) for candidate_path in candidate_paths) + ".")
    if len(found_paths) > 1:
        raise ValueError(f"Two {SAVED_THRESHOLDS_FILE_NAME} for the exported checkpoint, remove the one not to "
                         f"deploy: " + ", ".join(str(found_path) for found_path in found_paths) + ".")
    return found_paths[0]


def check_saved_thresholds_location(saved_thresholds_path: Path, experiment_folder: Path, checkpoint_path: Path,
                                    use_averaged_weights: bool) -> None:
    """
    Checks that thresholds given to the export (by the export button of the temperature explorer) are those saved in
    the evaluation folder of the exported run, iteration and weights, '<run>/Evaluation/Iteration_<n>/[EMA/]' or
    '<run>/Checkpoint_Evaluation/Iteration_<n>/[EMA/]', so that a classifier is never exported with the
    temperature chosen on the evaluation of another checkpoint or of the other weights.

    Args:
        saved_thresholds_path (Path): The given 'saved_temperature_thresholds.json'.
        experiment_folder (Path): The exported run folder.
        checkpoint_path (Path): The exported checkpoint, '<run>/Weights/Iteration_<n>/model_<n>.pt'.
        use_averaged_weights (bool): True when the moving average of the weights is exported.

    Raises:
        ValueError: If the file is not the saved thresholds of the evaluation of the exported checkpoint and weights.
    """
    # Evaluation folder of the iteration, the averaged weights in its 'EMA' subfolder, in one of the two evaluation
    # folders of the exported run folder itself
    evaluation_folder = saved_thresholds_path.parent
    iteration_folder = evaluation_folder.parent if use_averaged_weights else evaluation_folder
    weights_folder_name = evaluation_folder.name if use_averaged_weights else None
    found_location = (saved_thresholds_path.name, weights_folder_name, iteration_folder.name,
                      iteration_folder.parent.name, iteration_folder.parent.parent.resolve())
    expected_locations = [(SAVED_THRESHOLDS_FILE_NAME, AVERAGED_WEIGHTS_FOLDER_NAME if use_averaged_weights else None,
                           checkpoint_path.parent.name, evaluation_folder_name, experiment_folder.resolve())
                          for evaluation_folder_name in [EVALUATION_FOLDER_NAME, CHECKPOINT_EVALUATION_FOLDER_NAME]]
    if found_location not in expected_locations or (not use_averaged_weights and
                                                    evaluation_folder.name == AVERAGED_WEIGHTS_FOLDER_NAME):
        weights_description = f"{AVERAGED_WEIGHTS_FOLDER_NAME}/" if use_averaged_weights else ""
        raise ValueError(f"Expected the thresholds saved in {experiment_folder.name}/[{EVALUATION_FOLDER_NAME} or "
                         f"{CHECKPOINT_EVALUATION_FOLDER_NAME}]/{checkpoint_path.parent.name}/{weights_description}"
                         f"{SAVED_THRESHOLDS_FILE_NAME} for the exported checkpoint, got {saved_thresholds_path}.")


def read_saved_temperature(saved_thresholds_path: Path, number_classes: int) -> float:
    """
    Reads the logit temperature built into an exported classifier from the thresholds saved from its temperature
    explorer, after checking them against the class_thresholds.json next to them and the classes of the exported
    model, so that the exported probabilities are those the deployed thresholds were found on: an export with another
    temperature would make them accept nearly every image, or almost none.

    Args:
        saved_thresholds_path (Path): The saved thresholds of the exported checkpoint, from find_saved_thresholds_path.
        number_classes (int): The number of classes of the exported classifier.

    Returns:
        float: The logit temperature of the saved thresholds.

    Raises:
        FileNotFoundError: If no class_thresholds.json is next to the saved thresholds.
        ValueError: If the saved thresholds are not those of the evaluation next to them, or of another number of
            classes than the exported classifier.
    """
    saved_thresholds = json.loads(s=saved_thresholds_path.read_text())
    check_saved_thresholds(saved_thresholds=saved_thresholds, evaluation_folder=saved_thresholds_path.parent)
    if len(saved_thresholds["classes"]) != number_classes:
        raise ValueError(f"The saved thresholds {saved_thresholds_path} hold {len(saved_thresholds['classes'])} "
                         f"classes, the exported classifier predicts {number_classes}.")
    return float(saved_thresholds["logit_temperature"])


def install_staged_models(staging_folder: Path, repository: Path, model_version: int) -> None:
    """
    Moves the models export_experiment exported and checked in its staging folder into the Triton model repository,
    once every format matched the eager model, so that a failed check leaves the models served before untouched
    (Triton would otherwise load the mismatching model at its next start, or one format would be updated and the other
    not). The version folder of every model replaces the one of an earlier export of that version whole, files of the
    earlier export included, its other versions being kept, and its config.pbtxt replaces the earlier one.

    Args:
        staging_folder (Path): The staging folder of the export, holding one checked model folder per format in its
            STAGED_MODELS_FOLDER_NAME folder.
        repository (Path): The Triton model repository.
        model_version (int): The Triton version of the exported models, the name of their version folder.
    """
    replaced_versions_folder = staging_folder / REPLACED_VERSIONS_FOLDER_NAME
    replaced_versions_folder.mkdir()
    for staged_model_directory in sorted((staging_folder / STAGED_MODELS_FOLDER_NAME).iterdir()):
        model_directory = repository / staged_model_directory.name
        version_directory = model_directory / str(model_version)
        model_directory.mkdir(exist_ok=True)

        # The version folder of an earlier export moves into the staging folder, removed with it, its class
        # thresholds being reported when the new export deploys none (a classifier exported at a temperature of 1)
        if version_directory.exists():
            replaced_version_directory = replaced_versions_folder / staged_model_directory.name
            shutil.move(src=version_directory, dst=replaced_version_directory)
            if ((replaced_version_directory / "class_thresholds.json").exists() and
                    not (staged_model_directory / str(model_version) / "class_thresholds.json").exists()):
                print_color(color="yellow", output=f"Class thresholds of an earlier export removed : "
                                                   f"{format_path_link(path=version_directory)}", indent=1)

        # The checked version folder and its configuration take their place in the repository
        shutil.move(src=staged_model_directory / str(model_version), dst=version_directory)
        shutil.move(src=staged_model_directory / "config.pbtxt", dst=model_directory / "config.pbtxt")
        print_color(color="green", output=f"Installed : {format_path_link(path=model_directory)}", indent=1)


def check_export_settings(formats: Any, model_version: Any, maximum_batch_size: Any, tensorrt_precision: Any,
                          torchscript_precision: Any) -> None:
    """
    Checks the settings of an export before anything is loaded or written, at the start of export_experiment and by
    the export button of the temperature explorer before its job is prepared (explorer_export.py), so that a wrong
    value is refused with its reason instead of failing in the middle of the export, and a misspelled format
    ('TorchScript', 'tensorrt') is never exported as an ONNX model served without its TensorRT setting.

    Args:
        formats (Any): The export formats, a non-empty list of distinct values of EXPORT_FORMATS.
        model_version (Any): The Triton version of the exported models, a whole number from 1.
        maximum_batch_size (Any): The largest batch size of the exported models, a whole number from 1.
        tensorrt_precision (Any): One of TENSORRT_PRECISIONS.
        torchscript_precision (Any): One of TORCHSCRIPT_PRECISIONS.

    Raises:
        ValueError: If a setting is not one the export accepts.
    """
    # Known formats are checked before the repeated ones, so that a value that cannot be hashed is reported too
    if (not isinstance(formats, list) or not formats or
            not all(export_format in EXPORT_FORMATS for export_format in formats) or
            len(set(formats)) != len(formats)):
        raise ValueError(f"Expected distinct export formats among {EXPORT_FORMATS}, got {formats!r}.")
    for setting_name, setting_value in [("model_version", model_version), ("maximum_batch_size", maximum_batch_size)]:
        if not isinstance(setting_value, int) or isinstance(setting_value, bool) or setting_value < 1:
            raise ValueError(f"Expected {setting_name} to be a whole number from 1, got {setting_value!r}.")
    if tensorrt_precision not in TENSORRT_PRECISIONS:
        raise ValueError(f"Expected a TensorRT precision among {TENSORRT_PRECISIONS}, got {tensorrt_precision!r}.")
    if torchscript_precision not in TORCHSCRIPT_PRECISIONS:
        raise ValueError(f"Expected a TorchScript precision among {TORCHSCRIPT_PRECISIONS}, got "
                         f"{torchscript_precision!r}.")


def scale_half_precision_tolerances(comparison_tolerances: Dict[str, float]) -> Dict[str, float]:
    """
    Gives the tolerances of the random batch check of an FP16 TorchScript export, which export_experiment compares
    with the eager FP16 copy it was traced from: every difference tolerance of the export specification ('tolerance',
    'score_tolerance', 'box_tolerance'...) multiplied by HALF_PRECISION_TOLERANCE_FACTOR, the smallest shares of
    equal outputs ('minimum_equal_share') kept.

    Args:
        comparison_tolerances (Dict[str, float]): The 'comparison_tolerances' of the export specification of the task.

    Returns:
        Dict[str, float]: The tolerances of the FP16 check, with the same keys.
    """
    return {name: value * HALF_PRECISION_TOLERANCE_FACTOR if name.endswith("tolerance") else value
            for name, value in comparison_tolerances.items()}


def check_tensorrt_export(model_path: Path, staging_folder: Path, model_name: str, tensorrt_precision: str,
                          skip_tensorrt_check: bool, inference_model: nn.Module, check_inputs: Dict[str, torch.Tensor],
                          reference_outputs: List[np.ndarray], specification: Dict[str, Any]) -> List[str]:
    """
    Checks the staged ONNX model of an export as Triton will run it with TensorRT, once its ONNX Runtime check passed,
    so that an engine giving other outputs than the eager model blocks the installation like a failed format check
    (deployment/tensorrt_check.py). In FP16, the graph is first scanned on the CPU for values above the largest FP16
    value, a value above it stopping the checks. Then, unless skip_tensorrt_check, the engine is built and run on the
    check batches by a temporary Triton container and compared with the eager model: an FP32 engine with the FP32 model
    at the tolerances of the export specification, an FP16 engine with the FP16 copy of the network
    (convert_to_half_precision) at the tolerances of the FP16 TorchScript check (scale_half_precision_tolerances). Every
    step prints what it checked and how long it took.

    Args:
        model_path (Path): The staged 'model.onnx', its config.pbtxt written next to its version folder.
        staging_folder (Path): The staging folder of the export, holding the staged models, the engine cache and the
            server log.
        model_name (str): The Triton model name of the ONNX export.
        tensorrt_precision (str): 'FP32' or 'FP16', the precision of the engine.
        skip_tensorrt_check (bool): True to leave the engine unchecked (the FP16 overflow scan still runs).
        inference_model (nn.Module): The eager inference model of the export, on the GPU.
        check_inputs (Dict[str, torch.Tensor]): The check batch of every input, keyed by input name.
        reference_outputs (List[np.ndarray]): The outputs of the FP32 eager model on the check batches.
        specification (Dict[str, Any]): The export specification of the task.

    Returns:
        List[str]: The mismatches of the checks, empty when the engine (or, skipped, the FP16 scan) passed.
    """
    numpy_inputs = {input_name: input_batch.cpu().numpy() for input_name, input_batch in check_inputs.items()}
    batch_description = f"{next(iter(numpy_inputs.values())).shape[0]} samples"

    # FP16: every floating point node output of the graph against the largest FP16 value, on the CPU
    if tensorrt_precision == "FP16":
        scan_start_time = time.time()
        scan = find_half_precision_overflows(model_path=model_path, check_inputs=numpy_inputs)
        if scan["overflows"]:
            overflow_lines = describe_half_precision_overflows(overflows=scan["overflows"])
            print_color(color="red", output="\n".join(overflow_lines), indent=1)
            return overflow_lines
        print_color(color="green", output=f"TensorRT FP16 Overflow Scan : none of the {scan['scanned_outputs']:,} "
                                          f"floating point node outputs above {HALF_PRECISION_LARGEST_VALUE:,.0f} on "
                                          f"{batch_description} (ONNX Runtime CPU, "
                                          f"{time.time() - scan_start_time:.1f} seconds)", indent=1)

    # Engine left unchecked on request, Triton building it at its first request once installed
    if skip_tensorrt_check:
        print_color(color="yellow", output=f"TensorRT {tensorrt_precision} Engine Not Checked (--skip_tensorrt_check) "
                                           f": built by Triton at its first request, never compared with the eager "
                                           f"model", indent=1)
        return []

    # Reference of the engine: the FP32 eager model, or its FP16 copy at the loosened tolerances of FP16
    reference_name, engine_reference_outputs = "Eager Model", reference_outputs
    tolerances = specification["comparison_tolerances"]
    if tensorrt_precision == "FP16":
        input_names = [input_specification["name"] for input_specification in specification["inputs"]]
        half_precision_model = convert_to_half_precision(inference_model=inference_model, input_names=input_names)
        with torch.no_grad():
            engine_reference_outputs = convert_outputs_to_numpy(outputs=half_precision_model(**check_inputs))
        del half_precision_model
        reference_name = "Eager FP16 Model"
        tolerances = scale_half_precision_tolerances(comparison_tolerances=tolerances)

    # Engine built and run by a temporary Triton container on the check batches
    engine_check = check_tensorrt_engine(staged_repository=staging_folder / STAGED_MODELS_FOLDER_NAME,
                                         engine_cache_folder=staging_folder / STAGED_ENGINE_CACHE_FOLDER_NAME,
                                         log_path=staging_folder / TENSORRT_CHECK_LOG_NAME, model_name=model_name,
                                         check_inputs=numpy_inputs, reference_outputs=engine_reference_outputs,
                                         specification=specification, tolerances=tolerances)
    timing_description = (f"{batch_description} served by Triton, engine built and checked in "
                          f"{engine_check['engine_check_seconds']:.1f} seconds, Triton started in "
                          f"{engine_check['server_start_seconds']:.1f} seconds")
    if engine_check["mismatches"]:
        mismatch_lines = ([f"TensorRT {tensorrt_precision} Engine Fails Its Check Against The {reference_name} : "
                           f"{timing_description}"] + engine_check["mismatches"])
        print_color(color="red", output="\n".join(mismatch_lines), indent=1)
        return mismatch_lines
    print_color(color="green", output=f"TensorRT {tensorrt_precision} Engine Matches The {reference_name} : "
                                      f"{timing_description}", indent=1)
    return []


def export_experiment(experiment_folder: Path, iteration: Optional[int], use_averaged_weights: bool,
                      formats: List[str], repository: Path, model_version: int, maximum_batch_size: int,
                      tensorrt_precision: Optional[str], torchscript_precision: str, use_saved_temperature: bool,
                      saved_thresholds_path: Optional[Path], test_split_parity: bool,
                      skip_tensorrt_check: bool) -> None:
    """
    Exports a trained experiment of any task to the Triton model repository in every requested format, writes the
    Triton configurations, and checks each export against the eager inference model on a random batch of another
    size (and, for the sequence models, of another number of frames) than the traced one. The inference model, the
    Triton model names, the outputs and the comparison come from the export specification of the task of the run.
    With torchscript_precision 'FP16', the TorchScript file holds the FP16 copy of the network
    (convert_to_half_precision), with the same FP32 inputs and outputs, and its random batch check compares it with
    that eager FP16 copy at tolerances multiplied by HALF_PRECISION_TOLERANCE_FACTOR, its differences from the FP32
    eager model being printed for information.
    A frame selector also gets its decoder and one ensemble per format, with the decoding settings of the run and
    the confidence thresholds of the evaluation of the exported checkpoint. With use_saved_temperature, a classifier
    divides its class logits by the temperature saved from the temperature explorer of the exported checkpoint
    (find_saved_thresholds_path, or saved_thresholds_path given by the export button of the page, then
    read_saved_temperature), the saved thresholds being copied next to every exported model as 'class_thresholds.json';
    without it, by 1, without thresholds. Every exported model carries the input contract of the run
    (build_input_contract), written into the 'metadata_props' of the ONNX file and as 'preprocessing.json' next to the
    model of every format, the export stopping before any file is written when the contract cannot describe the input
    of the run. Every format is exported and checked in a staging folder next to the repository, and the models
    replace those of the repository (install_staged_models) only once all of them match the eager model, so that a
    failed check leaves the served models as they were. An ONNX model served with TensorRT is checked as Triton will
    run it once its ONNX Runtime check passed (check_tensorrt_export): in FP16, a CPU scan of its graph for values above
    the largest FP16 value, then, unless skip_tensorrt_check, its engine built by a temporary Triton container and
    compared with the eager model on the check batches, a failure blocking the installation like a mismatch, its
    mismatches recorded with those of the random batch check of the format. With test_split_parity, every format is
    then served from the staging folder by a temporary Triton container, as it will be served (its TensorRT precision
    included, reusing the engines of the TensorRT check), and scored on the whole test split of the run with the eager
    model (run_test_split_parity), even when the random batch check of a format failed; the summary is written to
    '<run>/Export_Parity/Iteration_<n>[_EMA][_T<temperature>]_<precisions>/' (get_parity_folder) and copied next to
    every installed model. The parity only reports: the random batch checks (with the TensorRT check) alone decide the
    installation, an expected drop of a reduced precision (TensorRT FP16, FP16 TorchScript) on the test split never
    stopping an export.

    Args:
        experiment_folder (Path): The experiment folder holding 'Weights' and 'training_configuration.yaml'.
        iteration (Optional[int]): The iteration to export, or None for the last saved one.
        use_averaged_weights (bool): True to export the moving average of the weights stored in the checkpoint
            instead of the trained weights.
        formats (List[str]): The export formats, distinct values among 'torchscript' and 'onnx'.
        repository (Path): The Triton model repository.
        model_version (int): The Triton version of the exported models, i.e. the name of their version folder.
        maximum_batch_size (int): The largest batch size of the exported models.
        tensorrt_precision (Optional[str]): 'FP16' or 'FP32' to accelerate the ONNX model with TensorRT, or None.
        torchscript_precision (str): 'FP32' to trace the network as it is, 'FP16' to trace its FP16 copy.
        use_saved_temperature (bool): Classification only: True builds the temperature saved from the temperature
            explorer of the exported checkpoint and weights into the model and deploys its thresholds, False exports
            with a temperature of 1 and without thresholds.
        saved_thresholds_path (Optional[Path]): With use_saved_temperature, the saved thresholds to deploy, checked to
            be those of the evaluation of the exported checkpoint and weights (check_saved_thresholds_location), or
            None to look for them in its evaluation folders (find_saved_thresholds_path).
        test_split_parity (bool): True to score every exported format, served by Triton, on the whole test split of
            the run against the eager model and write the parity summary, False for the random batch check only.
        skip_tensorrt_check (bool): True to install an ONNX model served with TensorRT without building and running
            its engine in a temporary Triton container, the FP16 overflow scan still running; not read without the
            'onnx' format or with tensorrt_precision None.

    Raises:
        RuntimeError: If an exported model does not reproduce the eager outputs, or its TensorRT engine fails its
            checks, the repository being left unchanged (with test_split_parity, after the parity summary is written),
            or if test_split_parity or the TensorRT engine check needs a Triton container on a machine where none can
            be started (check_triton_available, before anything is read).
        FileNotFoundError: If use_saved_temperature finds no saved thresholds for the exported checkpoint.
        ValueError: If a format, the model version, the maximum batch size, the TensorRT or the TorchScript precision
            is not one the export accepts (check_export_settings, before anything is loaded), if
            use_saved_temperature is set for another task than the classification, or the saved thresholds are
            ambiguous or do not belong to the exported checkpoint, or saved_thresholds_path is given without
            use_saved_temperature, or the input contract cannot describe the input of the run (another
            interpolation, padding, placement or channel order).
    """
    # A misspelled format or precision stops the export here, before the checkpoint is loaded
    check_export_settings(formats=formats, model_version=model_version, maximum_batch_size=maximum_batch_size,
                          tensorrt_precision=tensorrt_precision, torchscript_precision=torchscript_precision)
    if saved_thresholds_path is not None and not use_saved_temperature:
        raise ValueError(f"saved_thresholds_path {saved_thresholds_path} is only read with use_saved_temperature.")
    # The parity and the TensorRT engine check serve the exports with Triton in Docker: a machine without it stops
    # here, before the export
    if test_split_parity:
        check_triton_available(purpose="The test split parity", remedy="Export without --test_split_parity.")
    if "onnx" in formats and tensorrt_precision is not None and not skip_tensorrt_check:
        check_triton_available(purpose=f"The TensorRT {tensorrt_precision} engine check", remedy=TENSORRT_CHECK_REMEDY)

    # PyTorch runs GPU convolutions in reduced-precision TF32 by default (about 3 significant digits instead of 7),
    # while ONNX Runtime computes the exported model in exact FP32, so the two differ even for a correct export, by up
    # to about 1e-3 on the scores of a detector (5e-5 on the probabilities of a classifier). TF32 is switched off so
    # that the reference is computed at full precision too.
    # The switch is global to the whole Python process: the previous values are saved here and restored at the end,
    # so that code calling this function does not silently keep slower full-precision GPU computations afterwards
    previous_cudnn_tf32, previous_matmul_tf32 = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = False, False
    try:
        device = torch.device("cuda")
        configuration_path = experiment_folder / "training_configuration.yaml"
        experiment_configuration, model_configuration = load_experiment_configuration(
            configuration_path=configuration_path)
        checkpoint_path = get_checkpoint_path(experiment_folder=experiment_folder, iteration=iteration)
        model = load_trained_model(configuration_path=configuration_path, checkpoint_path=checkpoint_path,
                                   device=device, use_averaged_weights=use_averaged_weights)
        inference_model = build_inference_model(task=experiment_configuration["task"], model=model)
        specification = get_export_specification(task=experiment_configuration["task"],
                                                 inference_model=inference_model)
        weights_description = " (EMA Weights)" if use_averaged_weights else ""
        print_color(color="blue", output=f"Exporting Checkpoint{weights_description} : "
                                         f"{format_path_link(path=checkpoint_path)}", add_separators=True)

        # Input contract of every exported model, read from the configuration and the preprocessing code before any
        # file is written, so that a run whose input the contract cannot describe stops here
        input_contract = build_input_contract(task=experiment_configuration["task"],
                                              model_configuration=model_configuration, specification=specification)
        print_color(color="blue", output=f"Input Contract : {input_contract['input_resize']}, "
                                         f"{input_contract['input_size']} pixels, {input_contract['input_color_order']}"
                                         f" {input_contract['input_layout']}", indent=1)

        # Logit temperature of the classifier, from the thresholds saved from its temperature explorer and deployed
        # with it, set before the reference outputs so that the comparison checks the exported temperature too
        if use_saved_temperature and experiment_configuration["task"] != "classification":
            raise ValueError(f"use_saved_temperature is only read for the classification task, not for the "
                             f"{experiment_configuration['task']} task.")
        deployed_thresholds_path = None
        logit_temperature = 1.0
        if experiment_configuration["task"] == "classification":
            if not use_saved_temperature:
                print_color(color="yellow", output="use_saved_temperature off : exported with a logit temperature of 1 "
                                                   "and without class thresholds", indent=1)
            else:
                # Thresholds given by the export button of the page, or looked for in the evaluation folders
                if saved_thresholds_path is None:
                    deployed_thresholds_path = find_saved_thresholds_path(experiment_folder=experiment_folder,
                                                                          checkpoint_path=checkpoint_path,
                                                                          use_averaged_weights=use_averaged_weights)
                else:
                    check_saved_thresholds_location(saved_thresholds_path=saved_thresholds_path,
                                                    experiment_folder=experiment_folder,
                                                    checkpoint_path=checkpoint_path,
                                                    use_averaged_weights=use_averaged_weights)
                    deployed_thresholds_path = saved_thresholds_path
                logit_temperature = read_saved_temperature(saved_thresholds_path=deployed_thresholds_path,
                                                           number_classes=inference_model.number_classes)
                inference_model.set_logit_temperature(logit_temperature=logit_temperature)
                print_color(color="blue", output=f"Logit Temperature {logit_temperature:g} : "
                                                 f"{format_path_link(path=deployed_thresholds_path)}", indent=1)

        # Random example and check batches of different sizes, to confirm the batch dimension stays dynamic
        example_inputs = create_random_inputs(specification=specification, shape_name="example_shape",
                                              maximum_batch_size=maximum_batch_size, device=device, generator=None)
        check_inputs = create_random_inputs(specification=specification, shape_name="check_shape",
                                            maximum_batch_size=maximum_batch_size, device=device, generator=None)
        with torch.no_grad():
            reference_outputs = convert_outputs_to_numpy(outputs=inference_model(**check_inputs))

        # Every format is exported and checked in a staging folder next to the repository (outside the folder Triton
        # reads), its models moved into the repository only once all of them match the eager model
        repository.mkdir(parents=True, exist_ok=True)
        staging_folder = Path(tempfile.mkdtemp(prefix=STAGING_FOLDER_PREFIX, dir=repository.parent))
        try:
            (staging_folder / STAGED_MODELS_FOLDER_NAME).mkdir()
            random_batch_checks = {}
            for export_format in formats:
                model_name = f"{specification['model_name']}_{export_format}"
                model_directory = staging_folder / STAGED_MODELS_FOLDER_NAME / model_name
                # Reference of the random batch check of the format, the eager model unless its network is in FP16
                check_reference_name, check_reference_outputs = "Eager Model", reference_outputs
                check_tolerances = specification["comparison_tolerances"]

                # Export the model, then run the exported file on the check batch, an FP16 TorchScript file being
                # named 'model_fp16.pt' so that it can sit next to the 'model.pt' of the FP32 export
                if export_format == "torchscript":
                    model_path = model_directory / str(model_version) / TORCHSCRIPT_FILE_NAMES[torchscript_precision]
                    traced_model = inference_model
                    # FP16: the FP16 copy of the network is traced and is the reference of the check, at tolerances
                    # loosened for the FP16 roundings of other GPU kernels
                    if torchscript_precision == "FP16":
                        input_names = [input_specification["name"] for input_specification in specification["inputs"]]
                        traced_model = convert_to_half_precision(inference_model=inference_model,
                                                                 input_names=input_names)
                        with torch.no_grad():
                            check_reference_outputs = convert_outputs_to_numpy(outputs=traced_model(**check_inputs))
                        check_reference_name = "Eager FP16 Model"
                        check_tolerances = scale_half_precision_tolerances(comparison_tolerances=check_tolerances)
                    export_torchscript(inference_model=traced_model, example_inputs=example_inputs,
                                       output_path=model_path)
                    del traced_model
                    with torch.no_grad():
                        exported_outputs = convert_outputs_to_numpy(
                            outputs=torch.jit.load(str(model_path), map_location=device)(**check_inputs))
                else:
                    model_path = model_directory / str(model_version) / "model.onnx"
                    export_onnx(inference_model=inference_model, example_inputs=example_inputs, output_path=model_path,
                                specification=specification, maximum_batch_size=maximum_batch_size)
                    write_onnx_input_contract(model_path=model_path, input_contract=input_contract)
                    session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                    exported_outputs = session.run(None, {input_name: input_batch.cpu().numpy()
                                                          for input_name, input_batch in check_inputs.items()})

                # Configuration written once the model files are final, the TensorRT engines of an ONNX model being
                # cached under a prefix hashing them, so that an export of other weights never loads older engines
                write_triton_configuration(model_directory=model_directory, model_name=model_name,
                                           model_version=model_version, export_format=export_format,
                                           specification=specification, maximum_batch_size=maximum_batch_size,
                                           tensorrt_precision=tensorrt_precision,
                                           torchscript_precision=torchscript_precision)

                # Input contract next to the model of every format, with the values of the metadata of the ONNX file
                write_preprocessing_file(model_folder=model_path.parent, input_contract=input_contract)

                # Class thresholds of the classifier next to the model, found at the temperature built into it; without
                # them the installed version folder holds none, so that no thresholds of another temperature stay there
                if deployed_thresholds_path is not None:
                    shutil.copyfile(src=deployed_thresholds_path, dst=model_path.parent / "class_thresholds.json")

                # A mismatch stops the export at once, or with the parity once every format is exported and scored
                mismatches = specification["compare_outputs"](reference=check_reference_outputs,
                                                              candidate=exported_outputs, **check_tolerances)
                random_batch_checks[export_format] = mismatches
                mismatch_message = (f"The {export_format} Export Does Not Match The {check_reference_name}, The "
                                    f"Served Models Left Unchanged:\n" + "\n".join(mismatches))
                if mismatches and not test_split_parity:
                    raise RuntimeError(mismatch_message)
                print_color(color="red" if mismatches else "green",
                            output=mismatch_message if mismatches else f"{export_format} Export Matches The "
                                                                       f"{check_reference_name}", indent=1)

                # ONNX model served with TensorRT: checked as Triton will run it, once its ONNX Runtime check passed, a
                # failure stopping the export like a mismatch
                if export_format == "onnx" and tensorrt_precision is not None and not mismatches:
                    tensorrt_mismatches = check_tensorrt_export(
                        model_path=model_path, staging_folder=staging_folder, model_name=model_name,
                        tensorrt_precision=tensorrt_precision, skip_tensorrt_check=skip_tensorrt_check,
                        inference_model=inference_model, check_inputs=check_inputs,
                        reference_outputs=reference_outputs, specification=specification)
                    random_batch_checks[export_format] = tensorrt_mismatches
                    if tensorrt_mismatches and not test_split_parity:
                        raise RuntimeError(f"The onnx Export Fails Its TensorRT {tensorrt_precision} Check, The Served "
                                           f"Models Left Unchanged:\n" + "\n".join(tensorrt_mismatches))

                # FP16 network: its differences from the FP32 eager model on the check batch, for information only,
                # the test split parity measuring what they cost
                if export_format == "torchscript" and torchscript_precision == "FP16":
                    precision_differences = specification["compare_outputs"](
                        reference=reference_outputs, candidate=exported_outputs,
                        **specification["comparison_tolerances"])
                    print_color(color="blue", output="FP16 TorchScript Against The FP32 Eager Model : " + (
                        "; ".join(precision_differences) if precision_differences else "within the FP32 tolerances"),
                                indent=1)

            # Test split parity of the staged models, written whatever the random batch check found, the installation
            # recorded in it before the models are installed or refused
            mismatching_formats = [export_format for export_format, mismatches in random_batch_checks.items()
                                   if mismatches]
            summary_paths = []
            if test_split_parity:
                parity_summary, sample_records = run_test_split_parity(
                    experiment_folder=experiment_folder, checkpoint_path=checkpoint_path,
                    iteration=iteration if iteration is not None else get_last_checkpoint_iteration(
                        weights_directory=experiment_folder / "Weights"),
                    use_averaged_weights=use_averaged_weights, inference_model=inference_model,
                    specification=specification, staged_repository=staging_folder / STAGED_MODELS_FOLDER_NAME,
                    engine_cache_folder=staging_folder / STAGED_ENGINE_CACHE_FOLDER_NAME, formats=formats,
                    tensorrt_precision=tensorrt_precision, torchscript_precision=torchscript_precision,
                    maximum_batch_size=maximum_batch_size,
                    model_version=model_version, logit_temperature=logit_temperature,
                    random_batch_checks=random_batch_checks)
                parity_summary["installation"] = (
                    f"refused, the random batch check of {', '.join(mismatching_formats)} failed" if mismatching_formats
                    else f"installed into {repository}, version {model_version}")
                summary_paths = write_parity_summary(parity_folder=Path(parity_summary["parity_folder"]),
                                                     summary=parity_summary, sample_records=sample_records)
                print_color(color="blue", output=f"Test Split Parity Summary : "
                                                 f"{format_path_link(path=summary_paths[0])}", indent=1)
            if mismatching_formats:
                raise RuntimeError("\n".join(f"The {export_format} Export Does Not Match Its Eager Reference, The "
                                             f"Served Models Left Unchanged:\n" +
                                             "\n".join(random_batch_checks[export_format])
                                             for export_format in mismatching_formats))

            # Every format matched: the parity summary next to every model, then the checked models replace the version
            # folders and configurations served before
            for export_format in formats:
                version_directory = (staging_folder / STAGED_MODELS_FOLDER_NAME /
                                     f"{specification['model_name']}_{export_format}" / str(model_version))
                for summary_path in summary_paths:
                    shutil.copyfile(src=summary_path, dst=version_directory / summary_path.name)
            install_staged_models(staging_folder=staging_folder, repository=repository, model_version=model_version)

        # The staging folder is removed whether the export succeeded or raised, with the version folders replaced
        finally:
            shutil.rmtree(path=staging_folder)

        # Frame selector: the decoder of the selection and the ensembles the clients query
        if experiment_configuration["task"] == "frame_selection":
            decoding_settings = build_decoding_settings(
                experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                unselected_class_index=model.get_unselected_class_index(), checkpoint_path=checkpoint_path,
                experiment_folder=experiment_folder)
            export_frame_selection_decoding(repository=repository, model_version=model_version, formats=formats,
                                            specification=specification, decoding_settings=decoding_settings,
                                            number_classes=inference_model.number_classes,
                                            maximum_batch_size=maximum_batch_size)
    # 'finally' runs whether the export succeeds or raises an error (missing checkpoint, export mismatch...), so the
    # TF32 settings are always restored, and any error still reaches the caller afterwards
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = (previous_cudnn_tf32,
                                                                                   previous_matmul_tf32)


def read_tensorrt_precision_argument(value: str) -> Optional[str]:
    """
    Converts the --tensorrt_precision argument of the export script into a value of TENSORRT_PRECISIONS, 'None'
    giving the ONNX model to ONNX Runtime CUDA without TensorRT, so that the command line reaches every TensorRT
    setting of export_experiment.

    Args:
        value (str): The text given on the command line.

    Returns:
        Optional[str]: None for 'None', otherwise the text, which argparse then checks against TENSORRT_PRECISIONS.
    """
    return None if value == "None" else value


def parse_export_arguments(arguments: List[str]) -> argparse.Namespace:
    """
    Reads the settings of an export from the command line of the script, so that a run is exported without editing the
    file: the run folder is required, the other settings default to DEFAULT_EXPORT_SETTINGS (the values the export
    form of the temperature explorer starts from too) and DEFAULT_REPOSITORY. Launched without any argument, the
    script prints the help of every argument and stops. A '~' left by the shell at the start of a path
    ('--experiment_folder=~/Data/...') is expanded to the home folder.

    Args:
        arguments (List[str]): The command line arguments, without the script name.

    Returns:
        argparse.Namespace: The arguments of export_experiment, with show_warnings.

    Raises:
        SystemExit: After printing the help when no argument is given, or the usage and the error when the run folder
            is missing or a value is not one the export accepts.
    """
    # The description and the example keep their line breaks, argparse wrapping the help of every argument
    parser = argparse.ArgumentParser(
        prog="uv run python -m vision_training.deployment.export_model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Exports a trained run to the Triton model repository as TorchScript and ONNX, checks every\n"
                    "format against the eager model and replaces the served models only when all of them match.",
        epilog="Example, iteration 2000 of a run with TensorRT in FP16:\n"
               "  uv run python -m vision_training.deployment.export_model --experiment_folder ~/Data/<run> \\\n"
               "      --iteration 2000 --tensorrt_precision FP16")
    # The run and the checkpoint to export
    parser.add_argument("--experiment_folder", type=Path, required=True,
                        help="Run folder holding 'Weights' and 'training_configuration.yaml' (required).")
    parser.add_argument("--iteration", type=int, default=None,
                        help="Iteration of the exported checkpoint (default: the last saved one).")
    parser.add_argument("--use_averaged_weights", action="store_true",
                        help="Export the moving average of the weights saved in the checkpoint instead of the "
                             "trained weights.")
    # The formats, where they are installed and the batches they accept
    parser.add_argument("--formats", nargs="+", choices=EXPORT_FORMATS,
                        default=list(DEFAULT_EXPORT_SETTINGS["formats"]),
                        help=f"Export formats (default: {' '.join(DEFAULT_EXPORT_SETTINGS['formats'])}).")
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY,
                        help="Triton model repository (default: the served_models folder that "
                             "vision_training/deployment/triton/docker-compose.yml mounts).")
    parser.add_argument("--model_version", type=int, default=DEFAULT_EXPORT_SETTINGS["model_version"],
                        help="Triton version of the models, the name of their version folder (default: %(default)s).")
    parser.add_argument("--maximum_batch_size", type=int, default=DEFAULT_EXPORT_SETTINGS["maximum_batch_size"],
                        help="Largest batch the models accept and Triton sends them (default: %(default)s).")
    # The precisions of the served models, their inputs and outputs staying FP32
    parser.add_argument("--tensorrt_precision", type=read_tensorrt_precision_argument, choices=TENSORRT_PRECISIONS,
                        default=DEFAULT_EXPORT_SETTINGS["tensorrt_precision"],
                        help="TensorRT precision of the ONNX model, None for ONNX Runtime CUDA "
                             "without TensorRT (default: %(default)s).")
    parser.add_argument("--torchscript_precision", choices=TORCHSCRIPT_PRECISIONS,
                        default=DEFAULT_EXPORT_SETTINGS["torchscript_precision"],
                        help="Precision of the traced network: FP32 writes 'model.pt', FP16 'model_fp16.pt', about "
                             "half the size (default: %(default)s).")
    # The temperature of a classifier, the test split parity, the TensorRT engine check and the warnings
    parser.add_argument("--use_saved_temperature", action="store_true",
                        help="Classification only: build the temperature saved from the temperature explorer of the "
                             "exported checkpoint into the model and copy its thresholds next to it as "
                             "class_thresholds.json (default: a temperature of 1, without thresholds).")
    parser.add_argument("--saved_thresholds_path", type=Path, default=None,
                        help="With --use_saved_temperature, the saved_temperature_thresholds.json to deploy (default: "
                             "the one in <run>/Evaluation/Iteration_<n>/ or <run>/Checkpoint_Evaluation/Iteration_<n>/"
                             ", in their EMA folder for the averaged weights).")
    parser.add_argument("--test_split_parity", action="store_true",
                        help="Also serve every format with a temporary Triton container (Docker needed) and score it "
                             "on the whole test split against the eager model, into <run>/Export_Parity/; a report "
                             "only, the random batch check alone deciding the installation.")
    parser.add_argument("--skip_tensorrt_check", action="store_true",
                        help="Install the ONNX model without building and running its TensorRT engine in a temporary "
                             "Triton container (Docker and a GPU needed otherwise); the FP16 overflow scan still runs.")
    parser.add_argument("--show_warnings", action="store_true",
                        help="Show the harmless library warnings printed during the export.")

    # Without any argument, the script describes its arguments instead of failing on the missing run folder
    if not arguments:
        parser.print_help()
        parser.exit(status=0)
    parsed_arguments = parser.parse_args(args=arguments)

    # A '~' the shell leaves after '=' points to the home folder, as in the other scripts of the repository
    for path_name in ["experiment_folder", "repository", "saved_thresholds_path"]:
        if getattr(parsed_arguments, path_name) is not None:
            setattr(parsed_arguments, path_name, getattr(parsed_arguments, path_name).expanduser())
    return parsed_arguments


if __name__ == "__main__":
    # Run folder and export settings read from the command line, the help printed when none is given
    export_arguments = parse_export_arguments(arguments=sys.argv[1:])

    # The harmless library warnings printed during the export are hidden unless --show_warnings is given
    if not export_arguments.show_warnings:
        hide_known_export_warnings()

    export_experiment(experiment_folder=export_arguments.experiment_folder, iteration=export_arguments.iteration,
                      use_averaged_weights=export_arguments.use_averaged_weights, formats=export_arguments.formats,
                      repository=export_arguments.repository, model_version=export_arguments.model_version,
                      maximum_batch_size=export_arguments.maximum_batch_size,
                      tensorrt_precision=export_arguments.tensorrt_precision,
                      torchscript_precision=export_arguments.torchscript_precision,
                      use_saved_temperature=export_arguments.use_saved_temperature,
                      saved_thresholds_path=export_arguments.saved_thresholds_path,
                      test_split_parity=export_arguments.test_split_parity,
                      skip_tensorrt_check=export_arguments.skip_tensorrt_check)
