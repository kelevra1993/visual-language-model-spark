"""
Measures the inference throughput of a trained experiment of any task, run locally in this process and served by
Triton as TorchScript and ONNX (TensorRT), on the images of deployment/inference_data, a frame selector being measured
with the decoding of its selection (its Triton ensembles), a semantic segmentation run trained on tiles being refused
(load_inference_inputs). Run as a script once the models are exported by deployment/export_model.py and the Triton
server is started, with the run folder and the request settings given as arguments (parse_inference_arguments), its
help printed when none is given.
"""

import argparse
import sys

from pathlib import Path
from typing import List, Optional

from vision_training.deployment.deployment_utilities import (get_served_model, load_decoding_settings,
                                                             load_inference_inputs, load_local_inference_model,
                                                             measure_local_throughput, measure_triton_throughput)

# Folder of the images sent to the models when the command line gives none, one image per request (the whole folder as
# one sequence for the sequence models, its first half as the before member and its second half as the after one for
# the comparison)
DEFAULT_IMAGE_FOLDER = Path(__file__).absolute().parent / "inference_data"

# Request settings taken when the command line does not give them, also quoted by the deployment tutorials: the Triton
# gRPC address of deployment/triton/docker-compose.yml, the timed requests per run and the untimed ones sent first
DEFAULT_INFERENCE_SETTINGS = {"url": "localhost:8001", "number_iterations": 1000, "number_warmup_iterations": 20}


def measure_inference_throughput(experiment_folder: Path, image_folder: Path, url: str, use_shared_memory: bool,
                                 number_iterations: int, number_warmup_iterations: int,
                                 iteration: Optional[int]) -> None:
    """
    Measures the throughput of an exported run on the same requests three ways, in this process, then served by Triton
    as TorchScript and as ONNX, so that the cost of serving a model is known before it is deployed (the Triton Server
    section of the README and the serving chapter of the deployment tutorials quote it).

    Args:
        experiment_folder (Path): The run folder, giving the input size, resizing mode and maximum sequence length, the
            weights of the local run and the names of its served models.
        image_folder (Path): The folder of the images sent to the models.
        url (str): The Triton gRPC address of the server started with deployment/triton/docker-compose.yml.
        use_shared_memory (bool): Whether the images reach Triton through system shared memory instead of inside the
            gRPC requests, which needs the 'ipc: host' setting of the docker compose file.
        number_iterations (int): The timed requests per run.
        number_warmup_iterations (int): The untimed requests made first while TensorRT and CUDA initialise.
        iteration (Optional[int]): The checkpoint iteration of the local run, None for the last saved one.
    """
    inputs = load_inference_inputs(image_folder=image_folder, experiment_folder=experiment_folder)

    # Local run, the inference model in this process without Triton, followed by the decoding of the selection for a
    # frame selector
    inference_model, specification = load_local_inference_model(experiment_folder=experiment_folder,
                                                                iteration=iteration)
    decoding_settings = load_decoding_settings(experiment_folder=experiment_folder, iteration=iteration,
                                               inference_model=inference_model)
    measure_local_throughput(inputs=inputs, inference_model=inference_model, decoding_settings=decoding_settings,
                             number_iterations=number_iterations, number_warmup_iterations=number_warmup_iterations)

    # Triton runs, the TorchScript model and the ONNX model accelerated by TensorRT (their ensembles with the decoder
    # for a frame selector)
    for export_format in ["torchscript", "onnx"]:
        model_name, output_names = get_served_model(specification=specification, export_format=export_format,
                                                    decoding_settings=decoding_settings)
        measure_triton_throughput(inputs=inputs, model_name=model_name, output_names=output_names, url=url,
                                  use_shared_memory=use_shared_memory, number_iterations=number_iterations,
                                  number_warmup_iterations=number_warmup_iterations)


def parse_inference_arguments(arguments: List[str]) -> argparse.Namespace:
    """
    Reads the settings of a throughput measurement from the command line of the script, so that an exported run is
    measured without editing the file: the run folder is required, the images default to DEFAULT_IMAGE_FOLDER and the
    requests to DEFAULT_INFERENCE_SETTINGS. Launched without any argument, the script prints the help of every argument
    and stops. A '~' left by the shell at the start of a path ('--experiment_folder=~/Data/...') is expanded to the
    home folder.

    Args:
        arguments (List[str]): The command line arguments, without the script name.

    Returns:
        argparse.Namespace: The arguments of measure_inference_throughput.

    Raises:
        SystemExit: After printing the help when no argument is given, or the usage and the error when the run folder
            is missing or a value is not a number.
    """
    # The description and the example keep their line breaks, argparse wrapping the help of every argument
    parser = argparse.ArgumentParser(
        prog="uv run python -m vision_training.deployment.run_inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Measures the throughput of an exported run in the process, then served by Triton as TorchScript\n"
                    "and ONNX, one image per request, once the Triton server is started.",
        epilog="Example, the last checkpoint of a run, the images sent through system shared memory:\n"
               "  uv run python -m vision_training.deployment.run_inference --experiment_folder ~/Data/<run> \\\n"
               "      --use_shared_memory")
    # The run measured and the images sent to its models
    parser.add_argument("--experiment_folder", type=Path, required=True,
                        help="Run folder holding 'Weights' and 'training_configuration.yaml', exported by "
                             "export_model.py (required).")
    parser.add_argument("--image_folder", type=Path, default=DEFAULT_IMAGE_FOLDER,
                        help="Images sent to the models, one per request, the whole folder as one sequence for the "
                             "sequence tasks (default: vision_training/deployment/inference_data).")
    # The server and the requests
    parser.add_argument("--url", default=DEFAULT_INFERENCE_SETTINGS["url"],
                        help="Triton gRPC address of deployment/triton/docker-compose.yml (default: %(default)s).")
    parser.add_argument("--use_shared_memory", action="store_true",
                        help="Send the images through system shared memory instead of inside the gRPC requests, "
                             "which needs the 'ipc: host' setting of the docker compose file.")
    parser.add_argument("--number_iterations", type=int, default=DEFAULT_INFERENCE_SETTINGS["number_iterations"],
                        help="Timed requests per run (default: %(default)s).")
    parser.add_argument("--number_warmup_iterations", type=int,
                        default=DEFAULT_INFERENCE_SETTINGS["number_warmup_iterations"],
                        help="Untimed requests sent first while TensorRT and CUDA initialise (default: %(default)s).")
    # The checkpoint of the run in the process
    parser.add_argument("--iteration", type=int, default=None,
                        help="Iteration of the checkpoint run in the process (default: the last saved one).")

    # Without any argument, the script describes its arguments instead of failing on the missing run folder
    if not arguments:
        parser.print_help()
        parser.exit(status=0)
    parsed_arguments = parser.parse_args(args=arguments)

    # A '~' the shell leaves after '=' points to the home folder, as in the other scripts of the repository
    for path_name in ["experiment_folder", "image_folder"]:
        setattr(parsed_arguments, path_name, getattr(parsed_arguments, path_name).expanduser())
    return parsed_arguments


if __name__ == "__main__":
    # Run folder and request settings read from the command line, the help printed when none is given
    inference_arguments = parse_inference_arguments(arguments=sys.argv[1:])
    measure_inference_throughput(experiment_folder=inference_arguments.experiment_folder,
                                 image_folder=inference_arguments.image_folder, url=inference_arguments.url,
                                 use_shared_memory=inference_arguments.use_shared_memory,
                                 number_iterations=inference_arguments.number_iterations,
                                 number_warmup_iterations=inference_arguments.number_warmup_iterations,
                                 iteration=inference_arguments.iteration)
