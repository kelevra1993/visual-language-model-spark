"""
Values and listings of the export and serving chapters of the deployment tutorials, made at every build by
tutorials/figures/deployment/figures.py from the code of the repository and from the real runs and exports named in
tutorials/tutorial_sources.yaml: the settings of the main block of vision_training/deployment/export_model.py, the
batches the exports are traced and checked on and the comparison tolerances of every task (from
get_export_specification of export_specifications.py, given stand-ins holding the attributes it reads), the config.pbtxt
files the export writes for the README classifier and for the benchmark detector (write_triton_configuration on their
export specifications, into a temporary folder), the input contract of the README classifier (build_input_contract,
no file written), the Export Sizes panel its launch prints (estimate_export_sizes on its model built on the CPU), the
size of the float32 requests, the ports and image of the Triton docker compose file, and the
layout of a real model repository of the serving benchmark (its file list committed next to the plan of the benchmark,
serving_benchmark_model_repository_files.txt). The listings are text files drawn by the listing elements of the diagram
scenes; the facts are quoted as {{deployment.<name>}}.
"""

import functools
import json
import re
import tempfile

import torch
import yaml

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from scripts.deployment.check_export_size_estimates import build_run_optimizer
from tutorial_builder.figure_context import find_repository_root
from tutorials.figures.classification.data_listings import write_listing
from tutorials.figures.figure_utilities import find_repository_file, find_run_folder, read_run_configuration
from tutorials.figures.foundations.code_values import read_main_block_value, read_triton_host_ports
from tutorials.figures.foundations.listings import capture_printed_text
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.deployment.export_size_estimates import estimate_export_sizes, print_export_size_summary
from vision_training.deployment.export_specifications import build_input_specification, get_export_specification
from vision_training.deployment.frame_selection_deployment import DECODER_OUTPUT_TYPES
from vision_training.deployment.input_contract import INPUT_CONTRACT_KEYS, build_input_contract
from vision_training.deployment.triton_configuration import (format_dimensions, get_served_output_names,
                                                             write_triton_configuration)
from vision_training.models.classification.image_classifier import ImageClassifier
from vision_training.models.model_transfer import TRANSFERABLE_COMPONENTS

# Export script, whose main block sets the formats, the version, the largest batch size and the TensorRT precision
EXPORT_SCRIPT = Path("vision_training") / "deployment" / "export_model.py"

# Script measuring the throughput of a run in the process and through Triton, whose main block sets its requests
RUN_INFERENCE_SCRIPT = Path("vision_training") / "deployment" / "run_inference.py"

# Docker compose file starting Triton on the model repository
COMPOSE_FILE = Path("vision_training") / "deployment" / "triton" / "docker-compose.yml"

# Microseconds per millisecond, for the queue delay of the dynamic batcher
MICROSECONDS_PER_MILLISECOND = 1000.0

# Colour channels of every input image, bytes of a float32 value (the type of every input of the served models), and
# bytes per megabyte of the request sizes
IMAGE_CHANNELS = 3
FLOAT32_BYTES = 4
BYTES_PER_MEGABYTE = 1e6

# Values the stand-in inference models give get_export_specification for the tasks the tutorials only quote the
# tolerances of: the tolerances do not depend on them, they only fill the shapes of the outputs (mask_scoring adds the
# 'mask_scores' output of the instance segmenter, compared within the score tolerance)
STAND_IN_ATTRIBUTES = {"input_image_size": 64, "number_classes": 2, "maximum_detections": 10, "mask_size": 28,
                       "mask_scoring": False, "maximum_sequence_length": 8, "number_channels": 3,
                       "number_candidates": 5, "representation_size": 16}

# Bottom border of the Export Sizes panel, whose subtitle gives the seconds the estimates took, different at every
# build: the listing writes PANEL_SECONDS_SUBTITLE in their place, so that it only changes with the sizes
PANEL_SECONDS_PATTERN = re.compile(pattern=r"^╰─+ [0-9.]+ seconds ─+╯$")
PANEL_SECONDS_SUBTITLE = " <seconds> seconds "

# Content of the stand-in model file of the config.pbtxt listings, no checkpoint being exported
STAND_IN_MODEL_BYTES = b"stand-in for the exported model file"

# Tasks whose comparison tolerances the export chapter quotes
TOLERANCE_TASKS = ["classification", "object_detection", "instance_segmentation", "semantic_segmentation",
                   "keypoint_detection", "self_supervised", "sequence_labelling"]

# Model folders of the real model repository of the serving benchmark (its onnxruntime_cuda setting) listed as a tree
LISTED_MODEL_FOLDERS = ["image_classifier_onnx", "image_classifier_torchscript", "object_detector_onnx"]

# Lines of a config.pbtxt holding a numeric setting the narration quotes, with the pattern of its value
CONFIGURATION_VALUE_PATTERNS = {"queue_delay_microseconds": r"max_queue_delay_microseconds: (\d+)",
                                "instance_count": r"kind: KIND_GPU count: (\d+)",
                                "workspace_bytes": r'key: "max_workspace_size_bytes" value: "(\d+)"',
                                "maximum_batch_size": r"max_batch_size: (\d+)"}


def describe_export_settings(repository_folder: Path) -> Dict[str, Any]:
    """
    Reads the settings of the main block of the export script that the export chapter quotes, so that a sentence
    naming the default batch size, version or precision follows the script.

    Args:
        repository_folder (Path): The repository root.

    Returns:
        Dict[str, Any]: The facts export.maximum_batch_size, export.model_version, export.tensorrt_precision and
            export.format_count.
    """
    script_path = repository_folder / EXPORT_SCRIPT
    return {"export.maximum_batch_size": read_main_block_value(script_path=script_path,
                                                               variable_name="maximum_batch_size"),
            "export.model_version": read_main_block_value(script_path=script_path, variable_name="model_version"),
            "export.tensorrt_precision": read_main_block_value(script_path=script_path,
                                                               variable_name="tensorrt_precision"),
            "export.format_count": len(read_main_block_value(script_path=script_path, variable_name="formats"))}


def describe_run_inference_settings(repository_folder: Path) -> Dict[str, Any]:
    """
    Reads the requests of the main block of run_inference.py, the timed ones and the untimed ones sent first while
    TensorRT and CUDA initialise, which the serving chapter quotes.

    Args:
        repository_folder (Path): The repository root.

    Returns:
        Dict[str, Any]: The facts run_inference.timed_requests and run_inference.warmup_requests.
    """
    script_path = repository_folder / RUN_INFERENCE_SCRIPT
    return {"run_inference.timed_requests": read_main_block_value(script_path=script_path,
                                                                  variable_name="number_iterations"),
            "run_inference.warmup_requests": read_main_block_value(script_path=script_path,
                                                                   variable_name="number_warmup_iterations")}


def describe_check_batches(sequence_run_folder: Path) -> Dict[str, Any]:
    """
    Gives the batches every export is traced on and checked on, from build_input_specification: the image batches,
    and the sequences of the sequence labelling run of the tutorials, traced with its maximum sequence length and
    checked with fewer frames, so that a dynamic dimension frozen by the export fails the check.

    Args:
        sequence_run_folder (Path): The sequence labelling run, whose maximum sequence length the sequences take.

    Returns:
        Dict[str, Any]: The facts export.traced_batch_size, export.checked_batch_size, export.traced_frames and
            export.checked_frames.
    """
    image_specification = build_input_specification(input_name="images", image_size=STAND_IN_ATTRIBUTES[
        "input_image_size"], maximum_sequence_length=None, input_index=0)
    maximum_sequence_length = read_run_configuration(run_folder=sequence_run_folder)["Data"]["sequence_settings"][
        "maximum_sequence_length"]
    sequence_specification = build_input_specification(input_name="images", image_size=STAND_IN_ATTRIBUTES[
        "input_image_size"], maximum_sequence_length=maximum_sequence_length, input_index=0)
    return {"export.traced_batch_size": image_specification["example_shape"][0],
            "export.checked_batch_size": image_specification["check_shape"][0],
            "export.traced_frames": sequence_specification["example_shape"][1],
            "export.checked_frames": sequence_specification["check_shape"][1]}


def describe_comparison_tolerances() -> Dict[str, Any]:
    """
    Reads the comparison tolerances of every quoted task from get_export_specification, given a stand-in inference
    model holding the attributes it reads, the tolerances being written in the specification and not in a constant.

    Returns:
        Dict[str, Any]: The facts tolerances.<task>.<tolerance name>.
    """
    facts = {}
    for task in TOLERANCE_TASKS:
        specification = get_export_specification(task=task, inference_model=SimpleNamespace(**STAND_IN_ATTRIBUTES))
        for tolerance_name, tolerance_value in specification["comparison_tolerances"].items():
            facts[f"tolerances.{task}.{tolerance_name}"] = tolerance_value
    return facts


def read_configuration_values(configuration_text: str) -> Dict[str, int]:
    """
    Reads the numeric settings of a written config.pbtxt that the narration quotes.

    Args:
        configuration_text (str): The text of the config.pbtxt.

    Returns:
        Dict[str, int]: The value of every setting of CONFIGURATION_VALUE_PATTERNS found in the text.

    Raises:
        ValueError: If the maximum batch size, the queue delay or the instance count is missing.
    """
    values = {}
    for value_name, value_pattern in CONFIGURATION_VALUE_PATTERNS.items():
        match = re.search(pattern=value_pattern, string=configuration_text)
        if match is not None:
            values[value_name] = int(match.group(1))
    missing_names = [name for name in ["maximum_batch_size", "queue_delay_microseconds", "instance_count"]
                     if name not in values]
    if missing_names:
        raise ValueError(f"The written config.pbtxt has no {missing_names}:\n{configuration_text}")
    return values


def write_configuration_listing(model_name: str, export_format: str, specification: Dict[str, Any],
                                maximum_batch_size: int, model_version: int, tensorrt_precision: Any,
                                output_path: Path) -> str:
    """
    Writes the config.pbtxt the export writes for a model with write_triton_configuration, in a temporary model folder
    holding a stand-in model file, whose hash gives the TensorRT engine cache prefix of an ONNX model, then copies its
    text as a listing.

    Args:
        model_name (str): The Triton model name, '<task model name>_<format>'.
        export_format (str): 'torchscript' or 'onnx'.
        specification (Dict[str, Any]): The export specification of the model.
        maximum_batch_size (int): The largest batch size of the export.
        model_version (int): The served version.
        tensorrt_precision (Any): 'FP16', 'FP32' or None.
        output_path (Path): The listing file.

    Returns:
        str: The text of the configuration.
    """
    with tempfile.TemporaryDirectory() as model_folder:
        # Stand-in for the exported model file, written before the configuration as the export does, so that the
        # listing holds a fixed engine cache prefix from one build to the next
        version_folder = Path(model_folder) / str(model_version)
        version_folder.mkdir()
        (version_folder / ("model.onnx" if export_format == "onnx" else "model.pt")).write_bytes(STAND_IN_MODEL_BYTES)
        write_triton_configuration(model_directory=Path(model_folder), model_name=model_name,
                                   model_version=model_version, export_format=export_format,
                                   specification=specification, maximum_batch_size=maximum_batch_size,
                                   tensorrt_precision=tensorrt_precision)
        configuration_text = (Path(model_folder) / "config.pbtxt").read_text(encoding="utf-8")
    write_listing(lines=configuration_text.rstrip("\n").split("\n"), output_path=output_path)
    return configuration_text


def write_served_outputs_listing(specification: Dict[str, Any], export_format: str, output_path: Path) -> None:
    """
    Writes the outputs of a served model as Triton names them, one short line per output (served name, data type,
    dimensions without the batch one, the scalar reshape), narrower than the output lines of its config.pbtxt.

    Args:
        specification (Dict[str, Any]): The export specification of the model.
        export_format (str): 'torchscript' or 'onnx'.
        output_path (Path): The listing file.
    """
    served_names = get_served_output_names(export_format=export_format, specification=specification)
    name_width = max(len(served_name) for served_name in served_names)
    lines = [f"{served_name:<{name_width}}  {output['data_type']:<10} "
             f"{format_dimensions(dimensions=output['dimensions'])}{' -> [ ]' if output['reshape_to_scalar'] else ''}"
             for served_name, output in zip(served_names, specification["outputs"])]
    write_listing(lines=lines, output_path=output_path)


def write_decoder_outputs_listing(output_path: Path) -> None:
    """
    Writes the outputs of the frame selection decoder with their Triton data types, one line per output, the outputs
    the clients of a frame selection ensemble receive.

    Args:
        output_path (Path): The listing file.
    """
    name_width = max(len(output_name) for output_name in DECODER_OUTPUT_TYPES)
    write_listing(lines=[f"{output_name:<{name_width}}  {data_type}"
                         for output_name, data_type in DECODER_OUTPUT_TYPES.items()], output_path=output_path)


def describe_triton_configurations(output_folder: Path, classification_run_folder: Path,
                                   detection_run_folder: Path, export_settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Writes the config.pbtxt listings of the export chapter: the README classifier exported as ONNX with TensorRT FP16
    (a stand-in holding the input size and class count of its configuration) and the benchmark detector as TorchScript
    (a stand-in holding the input size and maximum detections of its configuration), the attributes their export
    specifications read, so that no checkpoint is loaded; and reads the settings the narration quotes from them.

    Args:
        output_folder (Path): The figures folder of the series.
        classification_run_folder (Path): The classification run of the tutorials (the README classifier).
        detection_run_folder (Path): The reference run of the detection benchmark, served by the serving benchmark.
        export_settings (Dict[str, Any]): The facts of describe_export_settings.

    Returns:
        Dict[str, Any]: The facts triton.*, the image size and classes of the classifier (classifier.*), and the image
            size, maximum detections and output count of the detector (detector.*).
    """
    # Classifier: a stand-in holding the input size and class count of its configuration and label dictionary, all its
    # export specification reads (InferenceClassifier takes them from the backbone and the label dictionary), so that
    # the listing needs no checkpoint and builds from the run snapshots
    _, classification_model_configuration = load_experiment_configuration(
        configuration_path=classification_run_folder / "training_configuration.yaml")
    classification_inference_model = SimpleNamespace(
        input_image_size=classification_model_configuration["Data"]["image_settings"]["size"],
        number_classes=len(set(classification_model_configuration["Data"]["label_dictionary"].values())))
    classification_specification = get_export_specification(task="classification",
                                                            inference_model=classification_inference_model)
    classification_text = write_configuration_listing(
        model_name=f"{classification_specification['model_name']}_onnx", export_format="onnx",
        specification=classification_specification, maximum_batch_size=export_settings["export.maximum_batch_size"],
        model_version=export_settings["export.model_version"], tensorrt_precision="FP16",
        output_path=output_folder / "listings" / "config_pbtxt_classifier_onnx_fp16.txt")
    configuration_values = read_configuration_values(configuration_text=classification_text)

    # Detector: the attributes of its export specification from its configuration
    _, detection_model_configuration = load_experiment_configuration(
        configuration_path=detection_run_folder / "training_configuration.yaml")
    detection_stand_in = SimpleNamespace(
        input_image_size=detection_model_configuration["Data"]["image_settings"]["size"],
        maximum_detections=detection_model_configuration["Detector"]["inference"]["maximum_detections"])
    detection_specification = get_export_specification(task="object_detection", inference_model=detection_stand_in)
    write_configuration_listing(
        model_name=f"{detection_specification['model_name']}_torchscript", export_format="torchscript",
        specification=detection_specification, maximum_batch_size=export_settings["export.maximum_batch_size"],
        model_version=export_settings["export.model_version"], tensorrt_precision=None,
        output_path=output_folder / "listings" / "config_pbtxt_detector_torchscript.txt")
    write_served_outputs_listing(specification=detection_specification, export_format="torchscript",
                                 output_path=output_folder / "listings" / "detector_torchscript_outputs.txt")

    # Settings read back from the written classifier configuration, then the attributes of both stand-ins
    return {"triton.queue_delay_microseconds": configuration_values["queue_delay_microseconds"],
            "triton.queue_delay_milliseconds": configuration_values["queue_delay_microseconds"] /
                                                   MICROSECONDS_PER_MILLISECOND,
            "triton.instance_count": configuration_values["instance_count"],
            "triton.workspace_gigabytes": configuration_values["workspace_bytes"] / 2 ** 30,
            "classifier.image_size": classification_inference_model.input_image_size,
            "classifier.number_classes": classification_inference_model.number_classes,
            "detector.image_size": detection_stand_in.input_image_size,
            "detector.maximum_detections": detection_stand_in.maximum_detections,
            "detector.output_count": len(detection_specification["outputs"])}


def describe_input_contract(output_folder: Path, classification_run_folder: Path) -> Dict[str, Any]:
    """
    Builds the input contract of the README classifier as its export would (build_input_contract, no file written)
    and writes it as the preprocessing.json listing, with the number of keys of the contract.

    Args:
        output_folder (Path): The figures folder of the series.
        classification_run_folder (Path): The classification run of the tutorials.

    Returns:
        Dict[str, Any]: The facts contract.key_count, contract.input_size and contract.input_resize.
    """
    _, model_configuration = load_experiment_configuration(
        configuration_path=classification_run_folder / "training_configuration.yaml")
    image_size = model_configuration["Data"]["image_settings"]["size"]
    specification = {"inputs": [build_input_specification(input_name="images", image_size=image_size,
                                                          maximum_sequence_length=None, input_index=0)]}
    input_contract = build_input_contract(task="classification", model_configuration=model_configuration,
                                          specification=specification)
    write_listing(lines=json.dumps(obj=input_contract, indent=2).split("\n"),
                  output_path=output_folder / "listings" / "preprocessing_json_classifier.txt")
    return {"contract.key_count": len(INPUT_CONTRACT_KEYS), "contract.input_size": input_contract["input_size"],
            "contract.input_resize": input_contract["input_resize"]}


def describe_export_sizes(output_folder: Path, classification_run_folder: Path) -> Dict[str, Any]:
    """
    Writes the Export Sizes panel the launch of the README classifier prints after its Model Size panel
    (estimate_export_sizes and print_export_size_summary of vision_training/deployment/export_size_estimates.py), so
    that the export chapter shows that the size of every exported file is known before the training. The model is
    built on the CPU from the saved configuration of the run, its random weights giving the same sizes as the trained
    ones, with the optimizer and moving average of that configuration (build_run_optimizer of
    scripts/deployment/check_export_size_estimates.py, as the calibration of the panel builds them); the seconds of the
    subtitle, different at every build, are written '<seconds>'.

    Args:
        output_folder (Path): The figures folder of the series.
        classification_run_folder (Path): The classification run of the tutorials (the README classifier).

    Returns:
        Dict[str, Any]: The facts export_sizes.exported_parameters and export_sizes.total_parameters.

    Raises:
        ValueError: If a component of the run is frozen (the trainer leaves its parameters out of the optimizer, this
            model trains all of them) or if the panel has no subtitle of seconds.
        RuntimeError: If the forward pass or the trace of the estimates failed.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=classification_run_folder / "training_configuration.yaml")
    frozen_sections = [section_name for section_name in TRANSFERABLE_COMPONENTS if section_name in model_configuration
                       and model_configuration[section_name]["pretrained"]["freeze"]]
    if frozen_sections:
        raise ValueError(f"{classification_run_folder.name} freezes {frozen_sections}, whose parameters the optimizer "
                         f"built here would count in the checkpoint estimate.")

    # Estimates of the panel for the model in training mode, as at launch, with the optimizer of its configuration
    model = ImageClassifier(configuration=model_configuration, device=torch.device(device="cpu"), dtype=torch.float32)
    optimization_configuration = model_configuration["Optimization"]
    export_sizes = estimate_export_sizes(
        task=experiment_configuration["task"], model=model.train(),
        optimizer=build_run_optimizer(model=model, optimizer_name=optimization_configuration["optimizer"]),
        keeps_moving_average=optimization_configuration["exponential_moving_average"]["activated"])
    if export_sizes["errors"]:
        raise RuntimeError(f"The export size estimates of {classification_run_folder.name} failed:\n" +
                           "\n".join(export_sizes["errors"]))

    # Printed panel, its bottom border written again around '<seconds>' at the same width
    panel_lines = capture_printed_text(printing_function=functools.partial(
        print_export_size_summary, export_sizes=export_sizes)).rstrip("\n").split("\n")
    if PANEL_SECONDS_PATTERN.match(panel_lines[-1]) is None:
        raise ValueError(f"The Export Sizes panel ends with '{panel_lines[-1]}', expected its seconds subtitle.")
    border_width = len(panel_lines[-1]) - 2 - len(PANEL_SECONDS_SUBTITLE)
    panel_lines[-1] = (f"╰{'─' * (border_width // 2)}{PANEL_SECONDS_SUBTITLE}"
                       f"{'─' * (border_width - border_width // 2)}╯")
    write_listing(lines=panel_lines, output_path=output_folder / "listings" / "export_sizes_classifier.txt")
    return {"export_sizes.exported_parameters": export_sizes["exported_parameters"],
            "export_sizes.total_parameters": export_sizes["total_parameters"]}


def describe_request_sizes(classification_image_size: int, detection_image_size: int) -> Dict[str, Any]:
    """
    Gives the size of one float32 image of the classifier and of the detector inside a request, the tensor a gRPC
    request carries unless the images go through system shared memory.

    Args:
        classification_image_size (int): The input side of the classifier.
        detection_image_size (int): The input side of the detector.

    Returns:
        Dict[str, Any]: The facts request.classifier_image_megabytes and request.detector_image_megabytes.
    """
    return {f"request.{name}_image_megabytes":
            IMAGE_CHANNELS * image_size * image_size * FLOAT32_BYTES / BYTES_PER_MEGABYTE
            for name, image_size in [("classifier", classification_image_size), ("detector", detection_image_size)]}


def describe_compose_file(repository_folder: Path) -> Dict[str, Any]:
    """
    Reads the ports and the image of the Triton docker compose file, which the serving chapter quotes.

    Args:
        repository_folder (Path): The repository root.

    Returns:
        Dict[str, Any]: The facts compose.<endpoint>_port and compose.image.
    """
    compose_path = repository_folder / COMPOSE_FILE
    facts = {f"compose.{endpoint}_port": port for endpoint, port in
             read_triton_host_ports(compose_path=compose_path).items()}
    compose = yaml.safe_load(stream=compose_path.read_text(encoding="utf-8"))
    facts["compose.image"] = next(iter(compose["services"].values()))["image"]
    return facts


def list_path_tree(relative_paths: List[Tuple[str, ...]], prefix: str) -> List[str]:
    """
    Gives the lines of a folder tree drawn with box characters from the file paths it holds, the files and folders of
    every level sorted by name, so that the serving chapter shows the layout of a real model repository from its
    committed file list rather than from a folder that only exists on the machine of the benchmark.

    Args:
        relative_paths (List[Tuple[str, ...]]): The parts of every file path, relative to the folder drawn.
        prefix (str): The characters drawn before every line of this level.

    Returns:
        List[str]: The lines of the tree under the folder.
    """
    # Entries of this level: a name is a folder when a path continues under it
    entry_children = {}
    for path_parts in relative_paths:
        entry_children.setdefault(path_parts[0], [])
        if len(path_parts) > 1:
            entry_children[path_parts[0]].append(path_parts[1:])
    lines = []
    entry_names = sorted(entry_children)
    for entry_index, entry_name in enumerate(entry_names):
        # One line per entry, its sub-tree indented under it when it is a folder
        is_last = entry_index == len(entry_names) - 1
        children = entry_children[entry_name]
        lines.append(f"{prefix}{'└── ' if is_last else '├── '}{entry_name}{'/' if children else ''}")
        if children:
            lines += list_path_tree(relative_paths=children, prefix=prefix + ("    " if is_last else "│   "))
    return lines


def describe_model_repository(output_folder: Path) -> Dict[str, Any]:
    """
    Writes the tree of a real model repository written by the export for the serving benchmark (its ONNX Runtime CUDA
    setting: the classifier in both formats and the detector in ONNX), from its file list committed next to the plan of
    the benchmark (serving_benchmark_model_repository_files of the repository_files of tutorial_sources.yaml).

    Args:
        output_folder (Path): The figures folder of the series.

    Returns:
        Dict[str, Any]: The fact repository.listed_model_count.

    Raises:
        ValueError: If the file list does not hold exactly the listed model folders.
    """
    # File paths of the model repository, one per line, grouped by their model folder
    files_path = find_repository_file(file_key="serving_benchmark_model_repository_files")
    relative_paths = [tuple(line.split("/")) for line in files_path.read_text(encoding="utf-8").splitlines() if line]
    listed_folders = sorted({path_parts[0] for path_parts in relative_paths})
    if listed_folders != sorted(LISTED_MODEL_FOLDERS):
        raise ValueError(f"{files_path} lists the model folders {listed_folders}, expected {LISTED_MODEL_FOLDERS}.")
    lines = ["served_models/"] + list_path_tree(relative_paths=relative_paths, prefix="")
    write_listing(lines=lines, output_path=output_folder / "listings" / "model_repository_tree.txt")
    return {"repository.listed_model_count": len(LISTED_MODEL_FOLDERS)}


def make_export_values(output_folder: Path, experiments_folder: Path) -> Dict[str, Any]:
    """
    Makes the listings and facts of the export and serving chapters of the deployment series, from the code of the
    repository, the classification (its configuration, input contract and Export Sizes panel), detection benchmark and
    sequence labelling runs of tutorial_sources.yaml and the committed file list of the model repository of the serving
    benchmark.

    Args:
        output_folder (Path): The figures folder of the series.
        experiments_folder (Path): The folder of the real runs.

    Returns:
        Dict[str, Any]: The facts of every family of this module.
    """
    repository_folder = find_repository_root()
    classification_run_folder = find_run_folder(experiments_folder=experiments_folder, run_key="classification")
    detection_run_folder = find_run_folder(experiments_folder=experiments_folder,
                                           run_key="object_detection_benchmark_joint_a")
    sequence_run_folder = find_run_folder(experiments_folder=experiments_folder, run_key="sequence_labelling")

    # Outputs of the frame selection decoder, then the settings of the export script and everything derived from them
    write_decoder_outputs_listing(output_path=output_folder / "listings" / "decoder_outputs.txt")
    export_settings = describe_export_settings(repository_folder=repository_folder)
    triton_facts = describe_triton_configurations(
        output_folder=output_folder, classification_run_folder=classification_run_folder,
        detection_run_folder=detection_run_folder, export_settings=export_settings)
    return {**export_settings, **triton_facts,
            **describe_check_batches(sequence_run_folder=sequence_run_folder),
            **describe_comparison_tolerances(),
            **describe_input_contract(output_folder=output_folder, classification_run_folder=classification_run_folder),
            **describe_export_sizes(output_folder=output_folder, classification_run_folder=classification_run_folder),
            **describe_request_sizes(classification_image_size=triton_facts["classifier.image_size"],
                                     detection_image_size=triton_facts["detector.image_size"]),
            **describe_compose_file(repository_folder=repository_folder),
            **describe_run_inference_settings(repository_folder=repository_folder),
            **describe_model_repository(output_folder=output_folder)}
