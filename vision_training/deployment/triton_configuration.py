from pathlib import Path
from typing import Any, Dict, List, Optional


def format_dimensions(dimensions: List[int]) -> str:
    """
    Writes tensor dimensions in the syntax of the Triton model configurations, e.g. '[ 3, 1024, 1024 ]'.

    Args:
        dimensions (List[int]): The dimensions, -1 for a dimension that changes from one request to the next.

    Returns:
        str: The formatted dimensions.
    """
    return "[ " + ", ".join(str(dimension) for dimension in dimensions) + " ]"


def write_triton_configuration(model_directory: Path, model_name: str, model_version: int, export_format: str,
                               specification: Dict[str, Any], maximum_batch_size: int,
                               tensorrt_precision: Optional[str]) -> None:
    """
    Writes the 'config.pbtxt' describing an exported model to Triton, for every task: platform, inputs ('images', or
    'before_images' and 'after_images' for the comparison) and outputs of the export specification, dynamic
    batching, GPU execution and, for ONNX, the optional TensorRT accelerator.

    Args:
        model_directory (Path): The model folder of the Triton repository (holding the version folder).
        model_name (str): The Triton model name.
        model_version (int): The version served by Triton, which is also the name of the folder holding the model file.
        export_format (str): Either 'torchscript' or 'onnx'.
        specification (Dict[str, Any]): The export specification of the task, giving the inputs and the outputs (see
            get_export_specification).
        maximum_batch_size (int): The largest batch size Triton may send to the model.
        tensorrt_precision (Optional[str]): 'FP16' or 'FP32' to accelerate the ONNX model with TensorRT, or None.
    """
    # TorchScript outputs follow the '<name>__<index>' naming convention of the Triton PyTorch backend
    if export_format == "torchscript":
        platform = "pytorch_libtorch"
        output_names = [f"{output['name']}__{index}" for index, output in enumerate(specification["outputs"])]
    else:
        platform = "onnxruntime_onnx"
        output_names = [output["name"] for output in specification["outputs"]]

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

    # Let ONNX Runtime run the supported parts of the graph with TensorRT, the other operators stay on CUDA
    if export_format == "onnx" and tensorrt_precision is not None:
        configuration += ('optimization { execution_accelerators { gpu_execution_accelerator: [ {\n'
                          '  name: "tensorrt"\n'
                          f'  parameters {{ key: "precision_mode" value: "{tensorrt_precision}" }}\n'
                          '  parameters { key: "max_workspace_size_bytes" value: "4294967296" }\n'
                          '  parameters { key: "trt_engine_cache_enable" value: "1" }\n'
                          '  parameters { key: "trt_engine_cache_path" value: "/tmp/tensorrt_engine_cache" }\n'
                          '} ] } }\n')

    (model_directory / "config.pbtxt").write_text(configuration)
