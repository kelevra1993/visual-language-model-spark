import os
import cv2
import numpy as np
import torch
import tritonclient.grpc as triton_client
import tritonclient.utils.shared_memory as shared_memory

from pathlib import Path
from torch import nn
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple

from vision_training.deployment.export_specifications import (SEQUENCE_TASKS, build_inference_model,
                                                              convert_outputs_to_numpy, get_export_specification)
from vision_training.trainers.checkpointing import get_checkpoint_path
from vision_training.models.model_loading import load_trained_model
from vision_training.data.preprocessing import preprocess_image
from vision_training.utilities.os_utilities import get_image_paths
from vision_training.configuration.configuration_loader import load_experiment_configuration


def load_inference_inputs(image_folder: Path, experiment_folder: Path) -> List[np.ndarray]:
    """
    Reads and preprocesses the images of the inference folder exactly like the data loaders (resize and centred
    padding to the square network input), so that every deployed model receives the same kind of input as in
    training, and turns them into the unit sent to the models of the run: one image, or for the sequence tasks one
    window of consecutive images of the folder (cycling through them) as long as the windows of the training.

    Args:
        image_folder (Path): The folder holding the images.
        experiment_folder (Path): The trained experiment, whose configuration gives the task, the input size, the
            resizing mode and the sequence length.

    Returns:
        List[np.ndarray]: One float32 BGR batch in [0, 255] per image of the folder: [1, 3, image_size, image_size],
            or [1, T, 3, image_size, image_size] for the sequence tasks.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=experiment_folder / "training_configuration.yaml")
    image_size = model_configuration["Data"]["image_settings"]["size"]
    keep_ratio = model_configuration["Data"]["image_settings"]["keep_ratio"]

    # Every image in the channels first layout of the network
    images = []
    for image_path in get_image_paths(folder=image_folder):
        image = preprocess_image(image=cv2.imread(filename=str(image_path)), image_size=image_size,
                                 keep_ratio=keep_ratio)
        images.append(image.transpose(2, 0, 1).astype(np.float32))

    # Every image becomes a batch of one element, or starts a window of one element for the sequence tasks
    if experiment_configuration["task"] not in SEQUENCE_TASKS:
        return [image[None] for image in images]
    sequence_length = model_configuration["Data"]["sequence_settings"]["sequence_length"]
    return [np.stack([images[(window_start + frame_index) % len(images)]
                      for frame_index in range(sequence_length)])[None] for window_start in range(len(images))]


def load_local_inference_model(experiment_folder: Path, iteration: Optional[int]) -> Tuple[nn.Module,
                                                                                            Dict[str, Any]]:
    """
    Loads the trained weights of an experiment of any task into the inference model of its task on the GPU, the same
    network as the exported models but run in this process, as the reference for the speed of the served models,
    with the export specification giving the Triton model names and outputs.

    Args:
        experiment_folder (Path): The trained experiment holding 'Weights' and 'training_configuration.yaml'.
        iteration (Optional[int]): The checkpoint iteration to load, None for the last saved one.

    Returns:
        Tuple[nn.Module, Dict[str, Any]]: The inference model in evaluation mode, and the export specification of
            its task.
    """
    configuration_path = experiment_folder / "training_configuration.yaml"
    experiment_configuration, _ = load_experiment_configuration(configuration_path=configuration_path)
    checkpoint_path = get_checkpoint_path(experiment_folder=experiment_folder, iteration=iteration)
    model = load_trained_model(configuration_path=configuration_path, checkpoint_path=checkpoint_path,
                               device=torch.device("cuda"))
    inference_model = build_inference_model(task=experiment_configuration["task"], model=model)
    return inference_model, get_export_specification(task=experiment_configuration["task"],
                                                     inference_model=inference_model)


def get_triton_output_names(client: triton_client.InferenceServerClient, model_name: str,
                            output_names: List[str]) -> List[str]:
    """
    Finds the served names of the model outputs, read once per model before the timed requests.

    Args:
        client (triton_client.InferenceServerClient): The gRPC client of the Triton server.
        model_name (str): The Triton model name.
        output_names (List[str]): The output names of the export specification, in order.

    Returns:
        List[str]: The served output names, in the order of output_names.
    """
    # TorchScript outputs are named '<name>__<index>', ONNX outputs use the plain names
    served_output_names = [output.name for output in client.get_model_metadata(model_name=model_name).outputs]
    return [next(name for name in served_output_names if name.split("__")[0] == output_name)
            for output_name in output_names]


def create_triton_input_shared_memory(client: triton_client.InferenceServerClient, byte_size: int) -> Tuple[
        Any, str]:
    """
    Creates a system shared memory region holding one input batch and registers it with Triton, so that the images
    are written once to memory shared with the server instead of being copied into every gRPC request (12 MB per
    image of 1024 pixels, times the number of frames of a window for the sequence models). Triton must share the
    memory of the machine ('ipc: host' in deployment/triton/docker-compose.yml).

    Args:
        client (triton_client.InferenceServerClient): The gRPC client of the Triton server.
        byte_size (int): The size of the input batch in bytes.

    Returns:
        Tuple[Any, str]: The handle of the region, used to write the images, and its name registered with Triton.
    """
    # The process identifier keeps the region of every running client distinct
    region_name = f"inference_input_{os.getpid()}"
    shared_memory_handle = shared_memory.create_shared_memory_region(triton_shm_name=region_name,
                                                                     shm_key=f"/{region_name}", byte_size=byte_size)
    client.register_system_shared_memory(name=region_name, key=f"/{region_name}", byte_size=byte_size)
    return shared_memory_handle, region_name


def release_triton_input_shared_memory(client: triton_client.InferenceServerClient, shared_memory_handle: Any,
                                       region_name: str) -> None:
    """
    Unregisters the input shared memory region from Triton and frees it, once the requests are done or when they
    fail, so that no region is left behind on the machine.

    Args:
        client (triton_client.InferenceServerClient): The gRPC client of the Triton server.
        shared_memory_handle (Any): The handle of the region, from create_triton_input_shared_memory.
        region_name (str): The name of the region registered with Triton.
    """
    client.unregister_system_shared_memory(name=region_name)
    shared_memory.destroy_shared_memory_region(shm_handle=shared_memory_handle)


def run_triton_inference(batch: np.ndarray, client: triton_client.InferenceServerClient, model_name: str,
                         output_names: List[str], shared_memory_handle: Optional[Any],
                         region_name: Optional[str]) -> List[np.ndarray]:
    """
    Sends a preprocessed batch to a model served by Triton over gRPC and returns its outputs. With a shared memory
    region, the batch is written to it and the request only tells Triton where to read it.

    Args:
        batch (np.ndarray): The float32 [B, 3, H, W] (or [B, T, 3, H, W]) batch.
        client (triton_client.InferenceServerClient): The gRPC client of the Triton server.
        model_name (str): The Triton model name.
        output_names (List[str]): The served output names, from get_triton_output_names.
        shared_memory_handle (Optional[Any]): The handle of the input shared memory region, None to send the batch
            inside the request.
        region_name (Optional[str]): The name of the input shared memory region registered with Triton, None to send
            the batch inside the request.

    Returns:
        List[np.ndarray]: The outputs, in the order of output_names.
    """
    inference_input = triton_client.InferInput(name="images", shape=list(batch.shape), datatype="FP32")
    if shared_memory_handle is None or region_name is None:
        inference_input.set_data_from_numpy(batch)
    else:
        # Write the batch at the start of the region and point the request to it
        shared_memory.set_shared_memory_region(shm_handle=shared_memory_handle, input_values=[batch])
        inference_input.set_shared_memory(region_name=region_name, byte_size=batch.nbytes)

    result = client.infer(model_name=model_name, inputs=[inference_input],
                          outputs=[triton_client.InferRequestedOutput(name=name) for name in output_names])
    return [result.as_numpy(name=name) for name in output_names]


def run_local_inference(batch: np.ndarray, inference_model: nn.Module) -> List[np.ndarray]:
    """
    Runs a preprocessed batch through the inference model in this process, without Triton.

    Args:
        batch (np.ndarray): The float32 [B, 3, H, W] (or [B, T, 3, H, W]) batch.
        inference_model (nn.Module): The inference model on the GPU.

    Returns:
        List[np.ndarray]: The outputs of the inference model, in the order of its export specification.
    """
    with torch.no_grad():
        outputs = inference_model(images=torch.from_numpy(batch).to(device=torch.device("cuda")))
    return convert_outputs_to_numpy(outputs=outputs)


def measure_local_throughput(inputs: List[np.ndarray], inference_model: nn.Module, number_iterations: int,
                             number_warmup_iterations: int) -> None:
    """
    Measures the raw throughput of the local inference model, one image (or window) per call, the tqdm bar showing
    the calls per second. The first calls are left out, while CUDA is still initialising.

    Args:
        inputs (List[np.ndarray]): The preprocessed [1, 3, H, W] images (or [1, T, 3, H, W] windows), sent in turn.
        inference_model (nn.Module): The inference model on the GPU.
        number_iterations (int): The number of timed calls.
        number_warmup_iterations (int): The number of untimed calls made first.
    """
    for iteration_index in range(number_warmup_iterations):
        run_local_inference(batch=inputs[iteration_index % len(inputs)], inference_model=inference_model)

    for iteration_index in tqdm(range(number_iterations), desc="local", unit="sample"):
        run_local_inference(batch=inputs[iteration_index % len(inputs)], inference_model=inference_model)


def measure_triton_throughput(inputs: List[np.ndarray], model_name: str, output_names: List[str], url: str,
                              use_shared_memory: bool, number_iterations: int, number_warmup_iterations: int) -> None:
    """
    Measures the raw throughput of a model served by Triton, one image (or window) per request, the tqdm bar showing
    the requests per second. The first requests are left out, while TensorRT and CUDA are still initialising.

    Args:
        inputs (List[np.ndarray]): The preprocessed [1, 3, H, W] images (or [1, T, 3, H, W] windows), sent in turn.
        model_name (str): The Triton model name, e.g. 'object_detector_torchscript' or 'image_classifier_onnx'.
        output_names (List[str]): The output names of the export specification of the model, in order.
        url (str): The Triton gRPC address, e.g. 'localhost:8001'.
        use_shared_memory (bool): Whether Triton reads the images from system shared memory instead of receiving them
            inside the gRPC requests.
        number_iterations (int): The number of timed requests.
        number_warmup_iterations (int): The number of untimed requests made first.
    """
    client = triton_client.InferenceServerClient(url=url)
    served_output_names = get_triton_output_names(client=client, model_name=model_name, output_names=output_names)

    # One shared memory region sized for one input serves every request, when shared memory is used
    shared_memory_handle, region_name = None, None
    if use_shared_memory:
        shared_memory_handle, region_name = create_triton_input_shared_memory(client=client,
                                                                              byte_size=inputs[0].nbytes)

    # The region is always released, even when a request fails
    try:
        for iteration_index in range(number_warmup_iterations):
            run_triton_inference(batch=inputs[iteration_index % len(inputs)], client=client, model_name=model_name,
                                 output_names=served_output_names, shared_memory_handle=shared_memory_handle,
                                 region_name=region_name)

        for iteration_index in tqdm(range(number_iterations), desc=model_name, unit="sample"):
            run_triton_inference(batch=inputs[iteration_index % len(inputs)], client=client, model_name=model_name,
                                 output_names=served_output_names, shared_memory_handle=shared_memory_handle,
                                 region_name=region_name)
    finally:
        if shared_memory_handle is not None and region_name is not None:
            release_triton_input_shared_memory(client=client, shared_memory_handle=shared_memory_handle,
                                               region_name=region_name)
