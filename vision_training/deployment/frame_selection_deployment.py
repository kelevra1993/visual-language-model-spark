"""
Deployment of the frame selection task (SEL): the decoding settings of an exported run (decoding, frames per class,
unselected class, confidence thresholds of the evaluated checkpoint, number of candidates), the Triton decoder model
running the decoding of models/sequence/frame_selection_decoding.py with the Python backend
(deployment/frame_selection_decoder_backend.py), and the Triton ensembles chaining every exported frame selector with
it, so that the clients receive the chosen frames rather than the per-frame probabilities. Called by
deployment/export_model.py and deployment/deployment_utilities.py.
"""

import json
import shutil

from pathlib import Path
from typing import Any, Dict, List

from vision_training.deployment.triton_configuration import format_dimensions, get_served_output_names
from vision_training.models.sequence import frame_selection_decoding
from vision_training.models.sequence.frame_selection_decoding import get_wanted_class_indices
from vision_training.utilities.os_utilities import format_path_link, print_color

# Suffixes of the Triton names of the decoder model, after the model name of the task, and of the ensembles, after the
# name of the exported model of every format
DECODER_MODEL_SUFFIX = "_decoder"
ENSEMBLE_MODEL_SUFFIX = "_ensemble"

# Python backend model copied as 'model.py' into the decoder, with the decoding module it loads and its settings file
DECODER_BACKEND_PATH = Path(__file__).absolute().parent / "frame_selection_decoder_backend.py"
DECODING_MODULE_PATH = Path(frame_selection_decoding.__file__).absolute()
DECODING_SETTINGS_FILE = "decoding_settings.json"

# Input of the decoder (the output of the exported frame selector), and its outputs with their Triton data types, in
# the order of the outputs of the Python backend model
DECODER_INPUT_NAME = "class_probabilities"
DECODER_OUTPUT_TYPES = {"selected_frames": "TYPE_INT64", "selected_probabilities": "TYPE_FP32",
                        "low_confidence": "TYPE_BOOL", "candidate_frames": "TYPE_INT64",
                        "candidate_probabilities": "TYPE_FP32"}


def build_decoding_settings(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                            unselected_class_index: int, checkpoint_path: Path,
                            experiment_folder: Path) -> Dict[str, Any]:
    """
    Gathers the settings of the decoding of a frame selection run, written next to the Triton decoder and used by the
    local inference: the decoding and the frames per class of the 'Selection' section, the unselected class saved with
    the weights, the number of candidate frames of every class ('evaluation.top_k'), and the confidence thresholds and
    class names of the evaluation of the checkpoint ('class_thresholds.json'). A checkpoint that was not evaluated has
    no thresholds, and no chosen frame is then flagged as low confidence.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration of the run.
        model_configuration (Dict[str, Any]): The processed model sections of the run, with its 'Selection' section.
        unselected_class_index (int): The class index of the unselected frames, from the wanted class mask of the
            trained frame selector.
        checkpoint_path (Path): The checkpoint file, '<run>/Weights/Iteration_<n>/model_<n>.pt'.
        experiment_folder (Path): The run folder, holding the 'Evaluation' folder of the checkpoints.

    Returns:
        Dict[str, Any]: The 'decoding', 'frames_per_class', 'unselected_class_index', 'number_candidates', and for
            every wanted class (in class index order) its name in 'class_names' and its threshold in
            'class_thresholds' (None when the target precision was never reached, 0 without evaluation).
    """
    selection_configuration = model_configuration["Selection"]
    number_classes = len(set(model_configuration["Data"]["label_dictionary"].values()))
    wanted_class_indices = get_wanted_class_indices(number_classes=number_classes,
                                                    unselected_class_index=unselected_class_index)

    # Thresholds of the evaluation of the checkpoint, in the evaluation folder of its iteration
    thresholds_path = experiment_folder / "Evaluation" / checkpoint_path.parent.name / "class_thresholds.json"
    if thresholds_path.exists():
        thresholds = json.loads(s=thresholds_path.read_text())
        class_names = list(thresholds["classes"])
        class_thresholds = [None if class_threshold is None else class_threshold["threshold"]
                            for class_threshold in thresholds["classes"].values()]
        if thresholds["decoding"] != selection_configuration["decoding"]:
            print_color(color="yellow", output=f"The thresholds were found for the {thresholds['decoding']} decoding, "
                                               f"the run now uses the {selection_configuration['decoding']} decoding : "
                                               f"{format_path_link(path=thresholds_path)}")
    else:
        print_color(color="yellow", output=f"The checkpoint was not evaluated, no chosen frame is flagged as low "
                                           f"confidence : {format_path_link(path=thresholds_path)}")
        class_names = [str(class_index) for class_index in wanted_class_indices]
        class_thresholds = [0.0] * len(wanted_class_indices)

    return {"decoding": selection_configuration["decoding"],
            "frames_per_class": selection_configuration["frames_per_class"],
            "unselected_class_index": unselected_class_index,
            "number_candidates": experiment_configuration["evaluation"]["top_k"], "class_names": class_names,
            "class_thresholds": class_thresholds}


def get_decoder_outputs(decoding_settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Describes the outputs of the Triton decoder and of the ensembles: the chosen frames, their probabilities and their
    low-confidence flags [W, N], and the candidate frames and their probabilities [W, k], one row per wanted class.

    Args:
        decoding_settings (Dict[str, Any]): The decoding settings of the run (build_decoding_settings).

    Returns:
        List[Dict[str, Any]]: The 'name', 'dimensions' (without the batch one) and Triton 'data_type' of every output.
    """
    number_wanted_classes = len(decoding_settings["class_thresholds"])
    selection_dimensions = [number_wanted_classes, decoding_settings["frames_per_class"]]
    candidate_dimensions = [number_wanted_classes, decoding_settings["number_candidates"]]
    return [{"name": output_name, "data_type": data_type,
             "dimensions": candidate_dimensions if output_name.startswith("candidate") else selection_dimensions}
            for output_name, data_type in DECODER_OUTPUT_TYPES.items()]


def format_tensor_lines(tensors: List[Dict[str, Any]]) -> str:
    """
    Writes the input or output lines of a Triton model configuration, one line per tensor.

    Args:
        tensors (List[Dict[str, Any]]): The 'name', 'data_type' and 'dimensions' of every tensor.

    Returns:
        str: The lines, separated by commas.
    """
    return ",\n".join(f'  {{ name: "{tensor["name"]}" data_type: {tensor["data_type"]} '
                      f'dims: {format_dimensions(dimensions=tensor["dimensions"])} }}' for tensor in tensors)


def write_frame_selection_decoder(repository: Path, model_name: str, model_version: int,
                                  decoding_settings: Dict[str, Any], number_classes: int,
                                  maximum_batch_size: int) -> Path:
    """
    Writes the decoder model of the frame selection into the Triton repository: the Python backend model as
    'model.py', the decoding module it loads and the decoding settings of the run in its version folder, and its
    'config.pbtxt', run on the CPU and reading the [T, K] class probabilities of every sequence of a batch.

    Args:
        repository (Path): The Triton model repository.
        model_name (str): The Triton name of the decoder model.
        model_version (int): The Triton version of the model, the name of its version folder.
        decoding_settings (Dict[str, Any]): The decoding settings of the run (build_decoding_settings).
        number_classes (int): The number of classes K of the frame selector, the unselected class included.
        maximum_batch_size (int): The largest batch size of the exported frame selector.

    Returns:
        Path: The folder of the decoder model.
    """
    model_directory = repository / model_name
    version_directory = model_directory / str(model_version)
    version_directory.mkdir(parents=True, exist_ok=True)

    # Python model, decoding module and settings, read by the Triton Python backend from the version folder
    shutil.copyfile(src=DECODER_BACKEND_PATH, dst=version_directory / "model.py")
    shutil.copyfile(src=DECODING_MODULE_PATH, dst=version_directory / DECODING_MODULE_PATH.name)
    (version_directory / DECODING_SETTINGS_FILE).write_text(data=json.dumps(obj=decoding_settings, indent=4))

    decoder_input = {"name": DECODER_INPUT_NAME, "data_type": "TYPE_FP32", "dimensions": [-1, number_classes]}
    decoder_outputs = get_decoder_outputs(decoding_settings=decoding_settings)
    configuration = (f'name: "{model_name}"\n'
                     f'backend: "python"\n'
                     f'max_batch_size: {maximum_batch_size}\n'
                     f'input [\n{format_tensor_lines(tensors=[decoder_input])}\n]\n'
                     f'output [\n{format_tensor_lines(tensors=decoder_outputs)}\n]\n'
                     f'instance_group [ {{ kind: KIND_CPU count: 1 }} ]\n'
                     f'version_policy: {{ specific: {{ versions: [ {model_version} ] }} }}\n')
    (model_directory / "config.pbtxt").write_text(data=configuration)
    return model_directory


def write_frame_selection_ensemble(repository: Path, ensemble_name: str, network_name: str, decoder_name: str,
                                   model_version: int, export_format: str, specification: Dict[str, Any],
                                   decoding_settings: Dict[str, Any], maximum_batch_size: int) -> Path:
    """
    Writes the Triton ensemble of the frame selection for one export format: the exported frame selector gives the
    class probabilities of every frame of the sequences, passed to the decoder, so that a client sends the frames and
    receives the chosen frames of every wanted class in one request. Triton needs an empty version folder for the
    ensemble.

    Args:
        repository (Path): The Triton model repository.
        ensemble_name (str): The Triton name of the ensemble.
        network_name (str): The Triton name of the exported frame selector of the format.
        decoder_name (str): The Triton name of the decoder model.
        model_version (int): The Triton version of the composing models and of the ensemble.
        export_format (str): Either 'torchscript' or 'onnx', naming the output of the exported frame selector.
        specification (Dict[str, Any]): The export specification of the frame selector, giving its inputs.
        decoding_settings (Dict[str, Any]): The decoding settings of the run (build_decoding_settings).
        maximum_batch_size (int): The largest batch size of the composing models.

    Returns:
        Path: The folder of the ensemble.
    """
    model_directory = repository / ensemble_name
    (model_directory / str(model_version)).mkdir(parents=True, exist_ok=True)
    decoder_outputs = get_decoder_outputs(decoding_settings=decoding_settings)
    network_output_name = get_served_output_names(export_format=export_format, specification=specification)[0]

    # Inputs of the frame selector given to it, its probabilities given to the decoder, the decoder outputs returned
    network_input_maps = "".join(f'      input_map {{ key: "{input_specification["name"]}" value: '
                                 f'"{input_specification["name"]}" }}\n'
                                 for input_specification in specification["inputs"])
    decoder_output_maps = "".join(f'      output_map {{ key: "{output["name"]}" value: "{output["name"]}" }}\n'
                                  for output in decoder_outputs)
    # The inputs of the frame selector and the outputs of the decoder, then the step running the frame selector
    configuration = (f'name: "{ensemble_name}"\n'
                     f'platform: "ensemble"\n'
                     f'max_batch_size: {maximum_batch_size}\n'
                     f'input [\n{format_tensor_lines(tensors=specification["inputs"])}\n]\n'
                     f'output [\n{format_tensor_lines(tensors=decoder_outputs)}\n]\n'
                     f'ensemble_scheduling {{\n'
                     f'  step [\n'
                     f'    {{\n'
                     f'      model_name: "{network_name}"\n'
                     f'      model_version: {model_version}\n'
                     f'{network_input_maps}'
                     f'      output_map {{ key: "{network_output_name}" value: "{DECODER_INPUT_NAME}" }}\n'
                     f'    }},\n'
                     # The step running the decoder on the probabilities, its outputs returned by the ensemble
                     f'    {{\n'
                     f'      model_name: "{decoder_name}"\n'
                     f'      model_version: {model_version}\n'
                     f'      input_map {{ key: "{DECODER_INPUT_NAME}" value: "{DECODER_INPUT_NAME}" }}\n'
                     f'{decoder_output_maps}'
                     f'    }}\n'
                     f'  ]\n'
                     f'}}\n'
                     f'version_policy: {{ specific: {{ versions: [ {model_version} ] }} }}\n')
    (model_directory / "config.pbtxt").write_text(data=configuration)
    return model_directory


def export_frame_selection_decoding(repository: Path, model_version: int, formats: List[str],
                                    specification: Dict[str, Any], decoding_settings: Dict[str, Any],
                                    number_classes: int, maximum_batch_size: int) -> None:
    """
    Adds the decoding of the selection to the exported frame selector in the Triton repository: one decoder model,
    shared by the formats, and one ensemble per export format chaining the exported frame selector with it, named
    '<frame selector>_<format>_ensemble', the model the clients query.

    Args:
        repository (Path): The Triton model repository.
        model_version (int): The Triton version of the exported models.
        formats (List[str]): The export formats of the frame selector, among 'torchscript' and 'onnx'.
        specification (Dict[str, Any]): The export specification of the frame selector.
        decoding_settings (Dict[str, Any]): The decoding settings of the run (build_decoding_settings).
        number_classes (int): The number of classes K of the frame selector, the unselected class included.
        maximum_batch_size (int): The largest batch size of the exported frame selector.
    """
    decoder_name = f"{specification['model_name']}{DECODER_MODEL_SUFFIX}"
    decoder_directory = write_frame_selection_decoder(repository=repository, model_name=decoder_name,
                                                      model_version=model_version,
                                                      decoding_settings=decoding_settings,
                                                      number_classes=number_classes,
                                                      maximum_batch_size=maximum_batch_size)
    print_color(color="green", output=f"Frame Selection Decoder ({decoding_settings['decoding']}) : "
                                      f"{format_path_link(path=decoder_directory)}", indent=1)

    # One ensemble per export format, chaining the exported frame selector of the format with the decoder
    for export_format in formats:
        network_name = f"{specification['model_name']}_{export_format}"
        ensemble_directory = write_frame_selection_ensemble(
            repository=repository, ensemble_name=f"{network_name}{ENSEMBLE_MODEL_SUFFIX}", network_name=network_name,
            decoder_name=decoder_name, model_version=model_version, export_format=export_format,
            specification=specification, decoding_settings=decoding_settings, maximum_batch_size=maximum_batch_size)
        print_color(color="green", output=f"{export_format} Selection Ensemble : "
                                          f"{format_path_link(path=ensemble_directory)}", indent=1)
