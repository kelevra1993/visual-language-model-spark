"""
Values of the code and of the files of the repository that the foundations chapters quote but that no
{{constant:...}} or {{config:...}} placeholder of a storyboard can read, gathered at every build by
tutorials/figures/foundations/figures.py into the facts of the series: a keyword argument of a call (the momentum of
SGD in BaseTrainer.create_optimizer, the data type of the mixed precision autocast in BaseTrainer.run_model_iteration),
a default argument of a script (the PCA dimensions of cluster_image_embeddings.py, the free disk space and held-out
images of imagenet_download.py), the host ports of the Triton docker compose file (strings such as
"8000:8000"), the strengths of the supervised and self-supervised augmentation (tables holding computed values, which
ast.literal_eval refuses), the pixel range given by the input normalization, and values of the synthetic
classification configuration that need a conversion or a computation (its number of iterations written 1e10, the steps
averaged by its moving average). The storyboards quote them as {{foundations.<name>}}, so that a sentence follows the
code when it changes. The readers of main blocks are also used by the classification figures for the settings of the
synthetic dataset script.
"""

import ast

import torch
import yaml

from pathlib import Path
from typing import Any, Dict

from scripts.embedding.cluster_image_embeddings import DEFAULT_CLUSTERING_SETTINGS
from vision_training.configuration_processing.configuration_loader import load_configuration
from vision_training.data.augmentations import (SELF_SUPERVISED_AUGMENTATION_STRENGTHS,
                                                SUPERVISED_AUGMENTATION_STRENGTHS)
from vision_training.data.classification.imagenet_download import DEFAULT_DOWNLOAD_SETTINGS
from vision_training.models.components.input_normalization import InputNormalizer

# Ports of the Triton container for its HTTP, gRPC and metrics endpoints, the defaults of Triton, which the docker
# compose file of the repository maps to ports of the machine
TRITON_CONTAINER_PORTS = {"http": 8000, "grpc": 8001, "metrics": 8002}

# Lowest and highest raw pixel values the data loaders deliver, given to the input normalization to measure its range
RAW_PIXEL_RANGE = (0.0, 255.0)


def find_called_name(call: ast.Call) -> str:
    """
    Gives the name a call of a Python source calls, so that read_call_keyword finds a call by its name whatever its
    module prefix: "SGD" for torch.optim.SGD(...), "print" for print(...).

    Args:
        call (ast.Call): The call node.

    Returns:
        str: The called name, or an empty string for a call of an expression without a name.
    """
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return ""


def read_keyword_value(value: ast.expr) -> Any:
    """
    Gives the value of a keyword argument of a call read by read_call_keyword: a literal (the momentum 0.9 of SGD), or
    an attribute of a module (torch.bfloat16, the data type of the autocast) given by its last name, so that a fact
    names the data type the code passes without importing the module.

    Args:
        value (ast.expr): The node of the value of the keyword argument.

    Returns:
        Any: The literal value, or the last name of the attribute of a module ("bfloat16").

    Raises:
        ValueError: If the value is neither a literal nor an attribute of a module (e.g. an attribute of self).
    """
    # Attribute of a module, such as torch.bfloat16, the attributes of an object (self.momentum) being refused below
    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and value.value.id != "self":
        return value.attr
    return ast.literal_eval(node_or_string=value)


def read_call_keyword(module_path: Path, function_name: str, called_name: str, keyword: str) -> Any:
    """
    Reads a keyword argument written as a literal, or as a dotted name, in a call inside a function of a Python module,
    without importing it, for the facts of the values the code passes inline (the momentum of SGD in
    BaseTrainer.create_optimizer, the data type torch.bfloat16 of the autocast in BaseTrainer.run_model_iteration),
    which no constant placeholder of a storyboard can read.

    Args:
        module_path (Path): The Python module.
        function_name (str): The function or method holding the call, at any depth of the module.
        called_name (str): The name of the called function, without its module prefix (e.g. "SGD").
        keyword (str): The keyword argument read.

    Returns:
        Any: The literal value of the keyword argument, or the last name of a dotted name ("bfloat16" for
            torch.bfloat16).

    Raises:
        ValueError: If the function, the call or the keyword is missing, if the call is written twice with the keyword,
            or if the value is neither a literal nor a dotted name.
    """
    tree = ast.parse(source=module_path.read_text(encoding="utf-8"), filename=str(module_path))
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
                 node.name == function_name]
    if len(functions) != 1:
        raise ValueError(f"Expected one function {function_name} in {module_path}, found {len(functions)}.")

    # Keyword of every call of that name inside the function
    values = [read_keyword_value(value=argument.value) for node in ast.walk(functions[0])
              if isinstance(node, ast.Call) and find_called_name(call=node) == called_name
              for argument in node.keywords if argument.arg == keyword]
    if len(values) != 1:
        raise ValueError(f"Expected one call of {called_name} with the keyword {keyword} in {function_name} of "
                         f"{module_path}, found {len(values)}.")
    return values[0]


def read_main_block_value(script_path: Path, variable_name: str) -> Any:
    """
    Reads a setting assigned as a literal in the main block of a script (if __name__ == "__main__":), without running
    it, for the facts of the settings the scripts of the repository keep in their main block (the image size of the
    synthetic dataset script), which no constant placeholder can read.

    Args:
        script_path (Path): The Python script.
        variable_name (str): The variable assigned in its main block.

    Returns:
        Any: The literal value of its last assignment in the main block.

    Raises:
        ValueError: If the script has no main block, the variable is not assigned in it, or the value is not a literal.
    """
    tree = ast.parse(source=script_path.read_text(encoding="utf-8"), filename=str(script_path))
    main_blocks = [node for node in tree.body if isinstance(node, ast.If) and
                   ast.unparse(ast_obj=node.test).replace("'", '"') == '__name__ == "__main__"']
    if not main_blocks:
        raise ValueError(f"No main block in {script_path}.")

    # Last assignment of the variable at the top of the main block, as the script would leave it before its calls
    assignments = [node for node in main_blocks[-1].body if isinstance(node, ast.Assign) and
                   any(isinstance(target, ast.Name) and target.id == variable_name for target in node.targets)]
    if not assignments:
        raise ValueError(f"No assignment of {variable_name} in the main block of {script_path}.")
    return ast.literal_eval(node_or_string=assignments[-1].value)


def read_triton_host_ports(compose_path: Path) -> Dict[str, int]:
    """
    Reads the ports of the machine that the docker compose file of Triton maps to the HTTP, gRPC and metrics ports of
    the container, so that the serving scene names the ports a client reaches.

    Args:
        compose_path (Path): The docker compose file of vision_training/deployment/triton.

    Returns:
        Dict[str, int]: The host port of every endpoint of TRITON_CONTAINER_PORTS.

    Raises:
        ValueError: If an endpoint of the container is not mapped by the file.
    """
    compose = yaml.safe_load(stream=compose_path.read_text(encoding="utf-8"))

    # Every "host:container" mapping of every service, keyed by its container port
    host_ports = {}
    for service in compose["services"].values():
        for mapping in (service["ports"] if "ports" in service else []):
            host_port, container_port = str(mapping).rsplit(sep=":", maxsplit=1)
            host_ports[int(container_port)] = int(host_port.rsplit(sep=":", maxsplit=1)[-1])
    missing_endpoints = [name for name, port in TRITON_CONTAINER_PORTS.items() if port not in host_ports]
    if missing_endpoints:
        raise ValueError(f"The endpoints {missing_endpoints} of Triton are not mapped by {compose_path}.")
    return {name: host_ports[port] for name, port in TRITON_CONTAINER_PORTS.items()}


def describe_code_values(repository_folder: Path, configuration_path: Path) -> Dict[str, Any]:
    """
    Gives the facts of the foundations chapters read from the code and the files of the repository rather than from a
    run: the momentum of SGD, the data type of the mixed precision forward passes, the PCA dimensions of the clustering
    script, the free disk space and held-out images of the ImageNet-22k download, the host ports of Triton, the
    strengths of the supervised and self-supervised augmentation the data pipeline chapter quotes, the range of the
    input normalization, and the number of iterations and the averaging horizon of the moving average of the synthetic
    classification configuration the chapters show.

    Args:
        repository_folder (Path): The root folder of the repository.
        configuration_path (Path): The synthetic classification configuration the chapters show (repository file
            data_pipeline_configuration of tutorial_sources.yaml).

    Returns:
        Dict[str, Any]: The facts, named optimization.*, scripts.*, triton.*, data.* and configuration.*.
    """
    # Values passed inline by the code and default PCA dimensions of the clustering script
    base_trainer_path = repository_folder / "vision_training" / "trainers" / "base_trainer.py"
    facts = {"optimization.sgd_momentum": read_call_keyword(
                 module_path=base_trainer_path, function_name="create_optimizer", called_name="SGD",
                 keyword="momentum"),
             "optimization.mixed_precision_dtype": read_call_keyword(
                 module_path=base_trainer_path, function_name="run_model_iteration", called_name="autocast",
                 keyword="dtype"),
             "scripts.cluster_pca_dimensions": DEFAULT_CLUSTERING_SETTINGS["pca_dimensions"]}

    # Default arguments of the ImageNet-22k download quoted by the first chapter: the free disk space below which no new
    # shard starts, and the held-out images of every large enough class, both the validation and the test split
    facts.update({f"scripts.imagenet_{setting_name}": DEFAULT_DOWNLOAD_SETTINGS[setting_name]
                  for setting_name in ["minimum_free_gigabytes", "held_out_images_per_class",
                                       "minimum_held_out_class_size"]})
    triton_ports = read_triton_host_ports(
        compose_path=repository_folder / "vision_training" / "deployment" / "triton" / "docker-compose.yml")
    facts.update({f"triton.{endpoint}_port": port for endpoint, port in triton_ports.items()})

    # Strengths of the supervised augmentation quoted by the data pipeline chapter, the jigsaw cutting its image into
    # a square grid of cells, and the smallest share of the area kept by the crop of the self-supervised views
    jigsaw_grid_size = int(SUPERVISED_AUGMENTATION_STRENGTHS["jigsaw_grid_size"])
    facts.update({"data.supervised_strengths.crop_side_minimum": float(
                      SUPERVISED_AUGMENTATION_STRENGTHS["crop_side_range"][0]),
                  "data.supervised_strengths.brightness_maximum_delta": float(
                      SUPERVISED_AUGMENTATION_STRENGTHS["brightness_maximum_delta"]),
                  "data.supervised_strengths.noise_standard_deviation": float(
                      SUPERVISED_AUGMENTATION_STRENGTHS["noise_standard_deviation"]),
                  "data.supervised_strengths.jigsaw_cell_count": jigsaw_grid_size ** 2,
                  "data.self_supervised_strengths.crop_area_minimum": float(
                      SELF_SUPERVISED_AUGMENTATION_STRENGTHS["crop_area_range"][0])})

    # Range of the pixels after the input normalization, measured on the lowest and highest raw pixel values
    input_normalizer = InputNormalizer(input_normalization=True, per_sample_normalization=False)
    normalized_range = input_normalizer(input_tensor=torch.tensor(data=[[[RAW_PIXEL_RANGE]]])).flatten().tolist()
    facts.update({"data.normalized_pixel_minimum": float(normalized_range[0]),
                  "data.normalized_pixel_maximum": float(normalized_range[1])})

    # Number of iterations of the synthetic configuration, written 1e10 (a string for YAML) so that the run goes on
    # until stopped, and the number of steps its moving average of the weights averages, one over one minus its decay
    configuration = load_configuration(configuration_path=configuration_path)
    decay = float(configuration["Optimization"]["exponential_moving_average"]["decay"])
    facts.update({"configuration.number_iterations": int(float(
                      configuration["ExperimentConfiguration"]["number_iterations"])),
                  "configuration.moving_average_horizon": round(1.0 / (1.0 - decay))})
    return facts
