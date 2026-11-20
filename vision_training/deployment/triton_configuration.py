"""
Writes the 'config.pbtxt' Triton model configuration of an exported model of any task (platform, inputs, outputs,
batching, TensorRT acceleration with an engine cache named after the exported files) from its export specification,
and names its outputs as Triton serves them, refusing any export format or TensorRT precision other than those of
EXPORT_FORMATS and TENSORRT_PRECISIONS. Called by deployment/export_model.py once the model files are written, and by
the frame selection ensembles of deployment/frame_selection_deployment.py.
"""

import hashlib

from pathlib import Path
from typing import Any, Dict, List, Optional

# Export formats served by Triton with the platform of the backend reading each, and the TensorRT precisions of an ONNX
# model, None running it with ONNX Runtime CUDA only; any other value is refused before a model is exported
EXPORT_PLATFORMS = {"torchscript": "pytorch_libtorch", "onnx": "onnxruntime_onnx"}
EXPORT_FORMATS = list(EXPORT_PLATFORMS)
TENSORRT_PRECISIONS = ["FP32", "FP16", None]

# Folder of the Triton container where ONNX Runtime caches the TensorRT engines it builds at the first requests, not
# mounted by deployment/triton/docker-compose.yml, and the hexadecimal characters of the SHA-256 of the exported files
# kept in the engine cache prefix of an export (build_tensorrt_engine_cache_prefix)
TENSORRT_ENGINE_CACHE_FOLDER = "/tmp/tensorrt_engine_cache"
EXPORT_HASH_LENGTH = 16


def check_export_format(export_format: Any) -> None:
    """
    Refuses an export format the Triton configuration does not know, so that a misspelled format ('TorchScript',
    'tensorrt', 'trt') stops the export with its reason instead of being served as an ONNX model without its TensorRT
    setting, every other function of the deployment telling the formats apart by their exact name.

    Args:
        export_format (Any): The export format, one of EXPORT_FORMATS.

    Raises:
        ValueError: If the format is not one of EXPORT_FORMATS.
    """
    if export_format not in EXPORT_FORMATS:
        raise ValueError(f"Expected an export format among {EXPORT_FORMATS}, got {export_format!r}.")


def format_dimensions(dimensions: List[int]) -> str:
    """
    Writes tensor dimensions in the syntax of the Triton model configurations, e.g. '[ 3, 1024, 1024 ]'.

    Args:
        dimensions (List[int]): The dimensions, -1 for a dimension that changes from one request to the next.

    Returns:
        str: The formatted dimensions.
    """
    return "[ " + ", ".join(str(dimension) for dimension in dimensions) + " ]"


def get_served_output_names(export_format: str, specification: Dict[str, Any]) -> List[str]:
    """
    Names the outputs of an exported model as Triton serves them, for its configuration and for the ensembles reading
    them: '<name>__<index>' for TorchScript, following the convention of the Triton PyTorch backend, and the plain
    names of the export specification for ONNX.

    Args:
        export_format (str): Either 'torchscript' or 'onnx'.
        specification (Dict[str, Any]): The export specification of the task, giving the outputs in order.

    Returns:
        List[str]: The served name of every output, in order.

    Raises:
        ValueError: If the format is not one of EXPORT_FORMATS.
    """
    check_export_format(export_format=export_format)
    if export_format == "torchscript":
        return [f"{output['name']}__{index}" for index, output in enumerate(specification["outputs"])]
    return [output["name"] for output in specification["outputs"]]


def build_tensorrt_engine_cache_prefix(model_name: str, model_version: int, tensorrt_precision: str,
                                       version_directory: Path) -> str:
    """
    Names the TensorRT engines of one export of an ONNX model in the engine cache of the Triton container (the
    'trt_engine_cache_prefix' of its config.pbtxt), '<model name>_<version>_<precision>_<hash>', the hash being the
    first EXPORT_HASH_LENGTH hexadecimal characters of the SHA-256 of every file of its version folder: the graph
    'model.onnx' with its input contract and the weights 'model.onnx.data' the ONNX exporter writes next to it.
    ONNX Runtime names a cached engine after a fingerprint of the graph, its documentation asking to delete the cache
    whenever the model changes, so that with a fixed name an export of another checkpoint of the same architecture
    into the same version could be served with the engine built from the weights of the earlier export whenever the
    container kept its cache (after 'docker compose restart', or a model reloaded by a running server). With this
    prefix ONNX Runtime only loads the engines built from the same files, and builds new ones for any other export.

    Args:
        model_name (str): The Triton model name, '<task model name>_onnx'.
        model_version (int): The Triton version of the export, the name of its version folder.
        tensorrt_precision (str): 'FP16' or 'FP32', kept in the prefix so that each precision has its own engines.
        version_directory (Path): The version folder of the export, holding the written 'model.onnx' and its weights.

    Returns:
        str: The engine cache prefix, made of letters, digits and underscores.

    Raises:
        FileNotFoundError: If the version folder holds no 'model.onnx', the configuration being written before the
            model it names the engines of.
    """
    model_path = version_directory / "model.onnx"
    if not model_path.is_file():
        raise FileNotFoundError(f"Expected the exported {model_path} before its Triton configuration: the TensorRT "
                                f"engine cache prefix is the hash of the files of its version folder.")

    # Every file of the version folder in the order of their names, each hashed with its name, so that a change of
    # the graph, of its input contract or of its weights alone gives another prefix
    export_hash = hashlib.sha256()
    for file_path in sorted(path for path in version_directory.rglob(pattern="*") if path.is_file()):
        export_hash.update(file_path.relative_to(version_directory).as_posix().encode() + b"\0")
        with open(file_path, "rb") as exported_file:
            export_hash.update(hashlib.file_digest(exported_file, "sha256").digest())
    return f"{model_name}_{model_version}_{tensorrt_precision.lower()}_{export_hash.hexdigest()[:EXPORT_HASH_LENGTH]}"


def write_triton_configuration(model_directory: Path, model_name: str, model_version: int, export_format: str,
                               specification: Dict[str, Any], maximum_batch_size: int,
                               tensorrt_precision: Optional[str]) -> None:
    """
    Writes the 'config.pbtxt' describing an exported model to Triton, for every task: platform, inputs ('images', or
    'before_images' and 'after_images' for the comparison) and outputs of the export specification, dynamic
    batching, GPU execution and, for ONNX, the optional TensorRT accelerator, whose cached engines are named after the
    files of the export (build_tensorrt_engine_cache_prefix), so that the model files must be written first.

    Args:
        model_directory (Path): The model folder of the Triton repository (holding the version folder).
        model_name (str): The Triton model name.
        model_version (int): The version served by Triton, which is also the name of the folder holding the model file.
        export_format (str): Either 'torchscript' or 'onnx'.
        specification (Dict[str, Any]): The export specification of the task, giving the inputs and the outputs (see
            get_export_specification).
        maximum_batch_size (int): The largest batch size Triton may send to the model.
        tensorrt_precision (Optional[str]): 'FP16' or 'FP32' to accelerate the ONNX model with TensorRT, or None.

    Raises:
        ValueError: If the format is not one of EXPORT_FORMATS or the precision not one of TENSORRT_PRECISIONS, no
            configuration being written.
        FileNotFoundError: If an ONNX model accelerated by TensorRT has no 'model.onnx' in its version folder yet, no
            configuration being written.
    """
    # A misspelled format or precision is refused instead of giving an ONNX model served without TensorRT
    check_export_format(export_format=export_format)
    if tensorrt_precision not in TENSORRT_PRECISIONS:
        raise ValueError(f"Expected a TensorRT precision among {TENSORRT_PRECISIONS}, got {tensorrt_precision!r}.")

    # Backend of the format, the TorchScript outputs following the '<name>__<index>' naming convention of the Triton
    # PyTorch backend
    platform = EXPORT_PLATFORMS[export_format]
    output_names = get_served_output_names(export_format=export_format, specification=specification)

    # One line per input, in the order of the forward arguments of the model, whose names the inputs carry
    inputs = [f'  {{ name: "{input_specification["name"]}" data_type: {input_specification["data_type"]} '
              f'dims: {format_dimensions(dimensions=input_specification["dimensions"])} }}'
              for input_specification in specification["inputs"]]

    # One line per output, a single value per sample being exposed as [1] and reshaped to a scalar for the model
    outputs = []
    for output_name, output in zip(output_names, specification["outputs"]):
        reshape = " reshape: { shape: [ ] }" if output["reshape_to_scalar"] else ""
        outputs.append(f'  {{ name: "{output_name}" data_type: {output["data_type"]} '
                       f'dims: {format_dimensions(dimensions=output["dimensions"])}{reshape} }}')

    configuration = (f'name: "{model_name}"\n'
                     f'platform: "{platform}"\n'
                     f'max_batch_size: {maximum_batch_size}\n'
                     f'input [\n' + ",\n".join(inputs) + '\n]\n'
                     f'output [\n' + ",\n".join(outputs) + '\n]\n'
                     f'dynamic_batching {{ max_queue_delay_microseconds: 1000 }}\n'
                     f'instance_group [ {{ kind: KIND_GPU count: 1 }} ]\n'
                     f'version_policy: {{ specific: {{ versions: [ {model_version} ] }} }}\n')

    # Let ONNX Runtime run the supported parts of the graph with TensorRT, the other operators stay on CUDA; its
    # engines are cached in the container under a prefix naming this export, so that an export of other weights never
    # loads the engines built from those of an earlier export
    if export_format == "onnx" and tensorrt_precision is not None:
        engine_cache_prefix = build_tensorrt_engine_cache_prefix(
            model_name=model_name, model_version=model_version, tensorrt_precision=tensorrt_precision,
            version_directory=model_directory / str(model_version))
        configuration += ('optimization { execution_accelerators { gpu_execution_accelerator: [ {\n'
                          '  name: "tensorrt"\n'
                          f'  parameters {{ key: "precision_mode" value: "{tensorrt_precision}" }}\n'
                          '  parameters { key: "max_workspace_size_bytes" value: "4294967296" }\n'
                          '  parameters { key: "trt_engine_cache_enable" value: "1" }\n'
                          f'  parameters {{ key: "trt_engine_cache_path" value: "{TENSORRT_ENGINE_CACHE_FOLDER}" }}\n'
                          f'  parameters {{ key: "trt_engine_cache_prefix" value: "{engine_cache_prefix}" }}\n'
                          '} ] } }\n')

    (model_directory / "config.pbtxt").write_text(configuration)
