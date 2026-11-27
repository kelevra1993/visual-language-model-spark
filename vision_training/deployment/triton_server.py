"""
Starts, drives and stops a Triton Inference Server container serving a model repository written by
deployment/export_model.py, so that the exported models are measured exactly as Triton serves them (TorchScript with
the PyTorch backend, ONNX with ONNX Runtime CUDA or TensorRT, the TensorRT engines built at the first requests). A
server is described by its container name and ports (build_triton_server), so that the test split parity of an export
(deployment/export_parity.py) and its TensorRT engine check (deployment/tensorrt_check.py) run their own container on
free ports (start_temporary_triton_server) next to the server of deployment/triton/docker-compose.yml or a benchmark,
while the serving benchmark scripts of scripts/deployment keep their fixed container and ports. The models are loaded
and unloaded one at a time through the repository extension (explicit model control).
"""

import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
import torch

from pathlib import Path
from typing import Any, Dict, List, Optional

from vision_training.deployment.triton_configuration import TENSORRT_ENGINE_CACHE_FOLDER

# Triton server image (Triton 2.65.0, ONNX Runtime 1.23.2, TensorRT 10.14.1, arm64 build for the DGX Spark), the one of
# deployment/triton/docker-compose.yml, and the folder of its container holding the served model repository
TRITON_IMAGE = "nvcr.io/nvidia/tritonserver:26.01-py3"
CONTAINER_REPOSITORY_FOLDER = "/models"

# Waits of the server: start, model load and unload (a TensorRT engine of a large detector takes minutes to build),
# and the memory left to settle after an unload
SERVER_START_TIMEOUT_SECONDS = 300
MODEL_LOAD_TIMEOUT_SECONDS = 1800
MEMORY_SETTLE_SECONDS = 5


class TritonServerError(RuntimeError):
    """
    Error answered by the HTTP endpoint of a Triton server (a model it cannot load, an unknown model), told apart from
    the errors of the code so that the test split parity records a model Triton refuses as a failed row and goes on.
    """


def build_triton_server(container_name: str, http_port: int, grpc_port: int, metrics_port: int) -> Dict[str, Any]:
    """
    Describes a Triton container by its name and the ports it listens on in the network of the machine, read by every
    function of this module, so that two servers (the test split parity of an export and the deployed server, or a
    benchmark) never share a container name or a port.

    Args:
        container_name (str): The name of the Docker container, removed by name when the server stops or fails.
        http_port (int): The port of the HTTP endpoint (health, repository and statistics requests).
        grpc_port (int): The port of the gRPC endpoint (inference requests of tritonclient).
        metrics_port (int): The port of the Prometheus metrics endpoint.

    Returns:
        Dict[str, Any]: The 'container_name', the three ports, and the 'http_address', 'grpc_address' and
            'metrics_address' of the server on the machine.
    """
    return {"container_name": container_name, "http_port": http_port, "grpc_port": grpc_port,
            "metrics_port": metrics_port, "http_address": f"localhost:{http_port}",
            "grpc_address": f"localhost:{grpc_port}", "metrics_address": f"localhost:{metrics_port}/metrics"}


def find_free_ports(number_ports: int) -> List[int]:
    """
    Asks the system for ports no process listens on, all held open together so that they are distinct, for a server
    started next to others on the network of the machine. A port can in principle be taken again between this call and
    the start of the server, which then fails to start with the name of the port in its log.

    Args:
        number_ports (int): The number of ports.

    Returns:
        List[int]: The free ports.
    """
    sockets = [socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM) for _ in range(number_ports)]
    try:
        for open_socket in sockets:
            open_socket.bind(("", 0))
        return [open_socket.getsockname()[1] for open_socket in sockets]
    finally:
        for open_socket in sockets:
            open_socket.close()


def check_triton_available(purpose: str, remedy: str) -> None:
    """
    Refuses to go on when no Triton container can be started on this machine (no GPU for the container, Docker
    missing, or the Triton image not pulled), before an export spends time on the models, so that the test split parity
    or the TensorRT engine check of an export (deployment/export_model.py) never stops halfway for a machine setting.

    Args:
        purpose (str): What needs the container, starting the error, e.g. 'The test split parity'.
        remedy (str): The sentence ending the error, telling how to export without the container.

    Raises:
        RuntimeError: If no GPU is visible, the docker command is not found, or the Triton image is not on the machine.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(f"{purpose} serves the exported models with Triton in Docker on the GPU, but no GPU is "
                           f"visible on this machine. {remedy}")
    if shutil.which(cmd="docker") is None:
        raise RuntimeError(f"{purpose} serves the exported models with Triton in Docker, but the docker command is not "
                           f"found on this machine. {remedy}")
    image_check = subprocess.run(args=["docker", "image", "inspect", TRITON_IMAGE], capture_output=True, text=True,
                                 check=False)
    if image_check.returncode != 0:
        raise RuntimeError(f"{purpose} needs the Triton image {TRITON_IMAGE}, not on this machine (docker image "
                           f"inspect: {image_check.stderr.strip()}): pull it with 'docker pull {TRITON_IMAGE}'. "
                           f"{remedy}")


def send_server_request(server: Dict[str, Any], path: str, body: Optional[Dict[str, Any]],
                        timeout_seconds: float) -> Dict[str, Any]:
    """
    Sends a request to the HTTP endpoint of a server (health, model configuration, statistics, load and unload of the
    repository extension) and returns its JSON answer.

    Args:
        server (Dict[str, Any]): The server, from build_triton_server.
        path (str): The path after the address, e.g. 'v2/repository/models/image_classifier_onnx/load'.
        body (Optional[Dict[str, Any]]): The JSON body of a POST request, None for a GET request.
        timeout_seconds (float): The time the server has to answer.

    Returns:
        Dict[str, Any]: The JSON answer, empty when the server answers without a body.

    Raises:
        TritonServerError: If the server answers with an error, its message included.
    """
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url=f"http://{server['http_address']}/{path}", data=data,
                                     method="GET" if body is None else "POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            answer = response.read()
    except urllib.error.HTTPError as error:
        raise TritonServerError(f"Triton refused {path}: {error.read().decode(errors='replace')}") from error
    return json.loads(answer) if answer else {}


def is_server_ready(server: Dict[str, Any]) -> bool:
    """
    Tells whether a server answers its readiness request, while it starts.

    Args:
        server (Dict[str, Any]): The server, from build_triton_server.

    Returns:
        bool: True once the server is ready.
    """
    try:
        send_server_request(server=server, path="v2/health/ready", body=None, timeout_seconds=5)
        return True
    except (urllib.error.URLError, ConnectionError, RuntimeError, TimeoutError):
        return False


def start_triton_server(server: Dict[str, Any], repository: Path, engine_cache_folder: Path,
                        environment: Dict[str, str]) -> None:
    """
    Starts the Triton container of a server with no model loaded (the models are loaded one at a time), on the network
    and the shared memory of the machine and listening on the ports of the server, as the user of this process so
    that the TensorRT engine cache it writes can be emptied from here. The repository is mounted read only.

    Args:
        server (Dict[str, Any]): The server, from build_triton_server.
        repository (Path): The model repository served.
        engine_cache_folder (Path): The folder receiving the TensorRT engines ONNX Runtime builds (the
            'trt_engine_cache_path' written into the ONNX configurations by deployment/triton_configuration.py).
        environment (Dict[str, str]): The environment variables of the server.

    Raises:
        RuntimeError: If the server is not ready within SERVER_START_TIMEOUT_SECONDS.
    """
    subprocess.run(args=["docker", "rm", "--force", server["container_name"]], capture_output=True, check=False)
    engine_cache_folder.mkdir(parents=True, exist_ok=True)
    environment_arguments = [argument for name, value in environment.items()
                             for argument in ["--env", f"{name}={value}"]]
    subprocess.run(args=["docker", "run", "--detach", "--name", server["container_name"], "--gpus", "all",
                         "--ipc=host", "--network", "host", "--user", f"{os.getuid()}:{os.getgid()}",
                         "--volume", f"{repository}:{CONTAINER_REPOSITORY_FOLDER}:ro",
                         "--volume", f"{engine_cache_folder}:{TENSORRT_ENGINE_CACHE_FOLDER}",
                         *environment_arguments, TRITON_IMAGE, "tritonserver",
                         f"--model-repository={CONTAINER_REPOSITORY_FOLDER}", "--model-control-mode=explicit",
                         f"--http-port={server['http_port']}", f"--grpc-port={server['grpc_port']}",
                         f"--metrics-port={server['metrics_port']}"],
                   capture_output=True, text=True, check=True)
    start_time = time.time()
    while not is_server_ready(server=server):
        if time.time() - start_time > SERVER_START_TIMEOUT_SECONDS:
            raise RuntimeError(f"Triton not ready after {SERVER_START_TIMEOUT_SECONDS} s, see docker logs "
                               f"{server['container_name']}.")
        time.sleep(2)


def start_temporary_triton_server(container_name_prefix: str, repository: Path,
                                  engine_cache_folder: Path) -> Dict[str, Any]:
    """
    Starts a Triton container of this process on free ports, named '<prefix>-<process identifier>', for the checks of
    an export run next to the deployed server or a benchmark: the test split parity (deployment/export_parity.py) and
    the TensorRT engine check (deployment/tensorrt_check.py), each serving the staged models of the export.

    Args:
        container_name_prefix (str): The start of the container name, telling which check started it.
        repository (Path): The model repository served, read only.
        engine_cache_folder (Path): The folder receiving the TensorRT engines ONNX Runtime builds.

    Returns:
        Dict[str, Any]: The started server, from build_triton_server, to stop with stop_triton_server.
    """
    server_ports = find_free_ports(number_ports=3)
    server = build_triton_server(container_name=f"{container_name_prefix}-{os.getpid()}", http_port=server_ports[0],
                                 grpc_port=server_ports[1], metrics_port=server_ports[2])
    start_triton_server(server=server, repository=repository, engine_cache_folder=engine_cache_folder, environment={})
    return server


def stop_triton_server(server: Dict[str, Any], log_path: Path) -> None:
    """
    Saves the log of the Triton container of a server and removes the container, at the end of its measurements or
    when they fail.

    Args:
        server (Dict[str, Any]): The server, from build_triton_server.
        log_path (Path): The file receiving the server log.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logs = subprocess.run(args=["docker", "logs", server["container_name"]], capture_output=True, text=True,
                          check=False)
    log_path.write_text(logs.stdout + logs.stderr)
    subprocess.run(args=["docker", "rm", "--force", server["container_name"]], capture_output=True, check=False)


def load_served_model(server: Dict[str, Any], model_name: str, configuration: Optional[Dict[str, Any]]) -> None:
    """
    Loads a model into a server, replacing it if it is already loaded, with its config.pbtxt or with the configuration
    of a server setting.

    Args:
        server (Dict[str, Any]): The server, from build_triton_server.
        model_name (str): The Triton model name.
        configuration (Optional[Dict[str, Any]]): The configuration given at load time, None for config.pbtxt.
    """
    body = {} if configuration is None else {"parameters": {"config": json.dumps(configuration)}}
    send_server_request(server=server, path=f"v2/repository/models/{model_name}/load", body=body,
                        timeout_seconds=MODEL_LOAD_TIMEOUT_SECONDS)


def unload_served_model(server: Dict[str, Any], model_name: str) -> None:
    """
    Unloads a model and waits until the server no longer serves it, then leaves the memory time to settle, so that
    the next model starts from a server holding no model.

    Args:
        server (Dict[str, Any]): The server, from build_triton_server.
        model_name (str): The Triton model name.
    """
    send_server_request(server=server, path=f"v2/repository/models/{model_name}/unload", body={},
                        timeout_seconds=MODEL_LOAD_TIMEOUT_SECONDS)
    while True:
        try:
            send_server_request(server=server, path=f"v2/models/{model_name}/ready", body=None, timeout_seconds=5)
        except RuntimeError:
            break
        time.sleep(1)
    time.sleep(MEMORY_SETTLE_SECONDS)
