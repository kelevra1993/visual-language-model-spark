"""
Exports a trained experiment of any task to the Triton model repository as TorchScript and ONNX (optionally run with
TensorRT), writes the Triton configurations and checks every export against the eager inference model, in a staging
folder moved into the repository only when every format matches, adding for the frame selection the decoder and the
ensembles returning the chosen frames, and for the classification the logit temperature and the class thresholds
saved from the temperature explorer of the exported checkpoint. Every exported model carries the input contract of the
run (deployment/input_contract.py), in the metadata of the ONNX file and in 'preprocessing.json' next to it, so that a
server builds the input from a photo as the training did. Run as a script with the experiment, iteration and formats
set in its main block.
"""

import json
import shutil
import tempfile
import onnxruntime
import torch

from pathlib import Path
from typing import Any, List, Optional

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.deployment.export_formats import export_onnx, export_torchscript, hide_known_export_warnings
from vision_training.deployment.export_specifications import (build_inference_model, convert_outputs_to_numpy,
                                                              create_random_inputs, get_export_specification)
from vision_training.deployment.frame_selection_deployment import (build_decoding_settings,
                                                                   export_frame_selection_decoding)
from vision_training.deployment.input_contract import (build_input_contract, write_onnx_input_contract,
                                                       write_preprocessing_file)
from vision_training.deployment.triton_configuration import (EXPORT_FORMATS, TENSORRT_PRECISIONS,
                                                             write_triton_configuration)
from vision_training.evaluation.checkpoint_evaluation import CHECKPOINT_EVALUATION_FOLDER_NAME
from vision_training.evaluation.classification.temperature_explorer import (SAVED_THRESHOLDS_FILE_NAME,
                                                                            check_saved_thresholds)
from vision_training.trainers.base_trainer import AVERAGED_WEIGHTS_FOLDER_NAME, EVALUATION_FOLDER_NAME
from vision_training.trainers.checkpointing import get_checkpoint_path
from vision_training.models.model_loading import load_trained_model
from vision_training.utilities.os_utilities import format_path_link, print_color

# Default Triton model repository, mounted by deployment/triton/docker-compose.yml
DEFAULT_REPOSITORY = Path(__file__).absolute().parent / "triton" / "served_models"

# Staging folder of an export, created next to the model repository, where every format is exported and checked
# before install_staged_models moves the models into the repository, and its folders of the exported models and of
# the version folders they replace, removed with it
STAGING_FOLDER_PREFIX = ".export_staging_"
STAGED_MODELS_FOLDER_NAME = "models"
REPLACED_VERSIONS_FOLDER_NAME = "replaced_versions"


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


def check_export_settings(formats: Any, model_version: Any, maximum_batch_size: Any, tensorrt_precision: Any) -> None:
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


def export_experiment(experiment_folder: Path, iteration: Optional[int], use_averaged_weights: bool,
                      formats: List[str], repository: Path, model_version: int, maximum_batch_size: int,
                      tensorrt_precision: Optional[str], use_saved_temperature: bool,
                      saved_thresholds_path: Optional[Path]) -> None:
    """
    Exports a trained experiment of any task to the Triton model repository in every requested format, writes the
    Triton configurations, and checks each export against the eager inference model on a random batch of another
    size (and, for the sequence models, of another number of frames) than the traced one. The inference model, the
    Triton model names, the outputs and the comparison come from the export specification of the task of the run.
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
    failed check leaves the served models as they were.

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
        use_saved_temperature (bool): Classification only: True builds the temperature saved from the temperature
            explorer of the exported checkpoint and weights into the model and deploys its thresholds, False exports
            with a temperature of 1 and without thresholds.
        saved_thresholds_path (Optional[Path]): With use_saved_temperature, the saved thresholds to deploy, checked to
            be those of the evaluation of the exported checkpoint and weights (check_saved_thresholds_location), or
            None to look for them in its evaluation folders (find_saved_thresholds_path).

    Raises:
        RuntimeError: If an exported model does not reproduce the eager outputs, the repository being left unchanged.
        FileNotFoundError: If use_saved_temperature finds no saved thresholds for the exported checkpoint.
        ValueError: If a format, the model version, the maximum batch size or the TensorRT precision is not one the
            export accepts (check_export_settings, before anything is loaded), if use_saved_temperature is set for
            another task than the classification, or the saved thresholds are ambiguous or do not belong to the
            exported checkpoint, or saved_thresholds_path is given without use_saved_temperature, or the input
            contract cannot describe the input of the run (another interpolation, padding, placement or channel
            order).
    """
    # A misspelled format or precision stops the export here, before the checkpoint is loaded
    check_export_settings(formats=formats, model_version=model_version, maximum_batch_size=maximum_batch_size,
                          tensorrt_precision=tensorrt_precision)
    if saved_thresholds_path is not None and not use_saved_temperature:
        raise ValueError(f"saved_thresholds_path {saved_thresholds_path} is only read with use_saved_temperature.")

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
            for export_format in formats:
                model_name = f"{specification['model_name']}_{export_format}"
                model_directory = staging_folder / STAGED_MODELS_FOLDER_NAME / model_name

                # Export the model, then run the exported file on the check batch
                if export_format == "torchscript":
                    model_path = model_directory / str(model_version) / "model.pt"
                    export_torchscript(inference_model=inference_model, example_inputs=example_inputs,
                                       output_path=model_path)
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

                write_triton_configuration(model_directory=model_directory, model_name=model_name,
                                           model_version=model_version, export_format=export_format,
                                           specification=specification, maximum_batch_size=maximum_batch_size,
                                           tensorrt_precision=tensorrt_precision)

                # Input contract next to the model of every format, with the values of the metadata of the ONNX file
                write_preprocessing_file(model_folder=model_path.parent, input_contract=input_contract)

                # Class thresholds of the classifier next to the model, found at the temperature built into it; without
                # them the installed version folder holds none, so that no thresholds of another temperature stay there
                if deployed_thresholds_path is not None:
                    shutil.copyfile(src=deployed_thresholds_path, dst=model_path.parent / "class_thresholds.json")

                mismatches = specification["compare_outputs"](reference=reference_outputs, candidate=exported_outputs,
                                                              **specification["comparison_tolerances"])
                if mismatches:
                    raise RuntimeError(f"The {export_format} Export Does Not Match The Eager Model, The Served Models "
                                       f"Left Unchanged:\n" + "\n".join(mismatches))
                print_color(color="green", output=f"{export_format} Export Matches The Eager Model", indent=1)

            # Every format matched: the checked models replace the version folders and configurations served before
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

if __name__ == "__main__":
    # Trained experiment to export, holding 'Weights' and 'training_configuration.yaml' (the 35-class synthetic shapes
    # classifier of the README, kept with the tutorial runs)
    run_name = "CLS_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128-PT_HFLAT.128-HLN-CWB_MC35"
    experiment_folder = Path.home() / "Data" / "tutorial_runs" / "readme_screenshots" / run_name
    # Iteration to export, None for the last one recorded in the 'full-checkpoint' registry
    iteration = 2000
    # True exports the moving average of the weights stored in the checkpoint (runs with
    # Optimization.exponential_moving_average activated), False the trained weights
    use_averaged_weights = False
    # Export formats, among 'torchscript' and 'onnx', any other value being refused
    formats = ["torchscript", "onnx"]
    # Triton model repository, mounted by deployment/triton/docker-compose.yml
    repository = DEFAULT_REPOSITORY
    # Triton version of the exported models: they are written to '<model>/<version>/' and Triton serves this version
    model_version = 1
    # Largest batch size the exported models accept, and that Triton may send them
    maximum_batch_size = 16
    # TensorRT precision of the ONNX model ('FP16' or 'FP32'), None to run it with ONNX Runtime CUDA only
    tensorrt_precision = "FP32"
    # False hides the harmless library warnings printed during the export (see hide_known_export_warnings)
    show_warnings = False
    # Classification only: True builds the temperature saved with the button of the temperature explorer of the exported
    # iteration and weights (served by scripts/evaluation/serve_temperature_explorer.py) into the exported model and
    # copies its thresholds next to it as class_thresholds.json; the file 'saved_temperature_thresholds.json' is looked
    # for in '<run>/Evaluation/Iteration_<n>/' and '<run>/Checkpoint_Evaluation/Iteration_<n>/' ('EMA' after them for
    # the averaged weights). False exports with a temperature of 1 and without thresholds
    use_saved_temperature = False
    # With use_saved_temperature, the saved_temperature_thresholds.json to deploy, None to look for it in the two
    # folders above (the export button of the temperature explorer gives the file of its page)
    saved_thresholds_path = None

    if not show_warnings:
        hide_known_export_warnings()

    export_experiment(experiment_folder=experiment_folder, iteration=iteration,
                      use_averaged_weights=use_averaged_weights, formats=formats, repository=repository,
                      model_version=model_version, maximum_batch_size=maximum_batch_size,
                      tensorrt_precision=tensorrt_precision, use_saved_temperature=use_saved_temperature,
                      saved_thresholds_path=saved_thresholds_path)
