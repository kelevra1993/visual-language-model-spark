"""
Inference helpers of the deployment stage: preprocessing the images of the inference folder into the requests of the
model of any task (the whole image being the object crop of the keypoint detection, a predicted segmentation being
restored to its original image by data/preprocessing.py:restore_prediction_to_original_image), loading the local
inference model of a run (with the decoding settings of a frame selector, decoded after it like in the Triton
ensembles), sending requests to Triton over gRPC (optionally through shared memory), and measuring the throughput of
the local and served models. Used by deployment/run_inference.py.
"""

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

from vision_training.deployment.export_specifications import (SEQUENCE_TASKS, TASK_INPUT_NAMES,
                                                              build_inference_model, convert_outputs_to_numpy,
                                                              get_export_specification)
from vision_training.deployment.frame_selection_deployment import (DECODER_OUTPUT_TYPES, ENSEMBLE_MODEL_SUFFIX,
                                                                   build_decoding_settings)
from vision_training.models.sequence.frame_selection_decoding import run_frame_selection
from vision_training.trainers.checkpointing import get_checkpoint_path
from vision_training.models.model_loading import load_trained_model
from vision_training.data.keypoints.crop_transform import (build_crop_affines, compute_crop_regions,
                                                           create_identity_augmentations, warp_crops)
from vision_training.data.preprocessing import preprocess_boxes, preprocess_image
from vision_training.utilities.os_utilities import get_image_paths
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration


def build_frame_window(images: List[np.ndarray], window_start: int, sequence_length: int) -> np.ndarray:
    """
    Builds one window of consecutive images of the inference folder for the sequence models, cycling back to the
    first image at the end of the folder, so that any folder gives as many full windows as it holds images.

    Args:
        images (List[np.ndarray]): The preprocessed [3, H, W] images of the folder, in order.
        window_start (int): The index of the first image of the window, which may go past the end of the folder.
        sequence_length (int): The number of frames of the window.

    Returns:
        np.ndarray: The [T, 3, H, W] window.
    """
    return np.stack([images[(window_start + frame_index) % len(images)] for frame_index in range(sequence_length)])


def cut_inference_crop(image: np.ndarray, bounding_box: List[float], crop_settings: Dict[str, Any]) -> np.ndarray:
    """
    Cuts the crop of one object out of a preprocessed image for the keypoint detector, exactly like the unaugmented
    validation and test crops of the keypoint trainer (the box centred, widened to the aspect ratio of the crop and
    enlarged), so that the deployed model receives the same kind of crop as in its evaluation.

    Args:
        image (np.ndarray): The preprocessed [3, H, W] float32 BGR image in [0, 255].
        bounding_box (List[float]): The (x1, y1, x2, y2) box of the object in preprocessed image pixels.
        crop_settings (Dict[str, Any]): The Data.crop_settings section, giving the 'height', 'width' and
            'box_enlargement' of the crops.

    Returns:
        np.ndarray: The [1, 3, crop height, crop width] float32 crop.
    """
    crop_height, crop_width = crop_settings["height"], crop_settings["width"]
    centres, sizes = compute_crop_regions(bounding_boxes=torch.tensor(data=[bounding_box], dtype=torch.float32),
                                          crop_height=crop_height, crop_width=crop_width,
                                          box_enlargement=crop_settings["box_enlargement"])

    # The crop without augmentation, sampled from the image through the crop to image matrix
    _, crop_to_image = build_crop_affines(centres=centres, sizes=sizes,
                                          augmentations=create_identity_augmentations(number_samples=1),
                                          crop_height=crop_height, crop_width=crop_width)
    crops = warp_crops(images=torch.from_numpy(image)[None], crop_to_image=crop_to_image, crop_height=crop_height,
                       crop_width=crop_width)
    return crops.numpy()


def load_inference_inputs(image_folder: Path, experiment_folder: Path) -> List[Dict[str, np.ndarray]]:
    """
    Reads and preprocesses the images of the inference folder exactly like the data loaders (resize and centred
    padding to the square network input), so that every deployed model receives the same kind of input as in
    training, and turns them into the requests sent to the models of the run, keyed by input name: one image, or for
    the sequence tasks one window of consecutive images of the folder (cycling through them) as long as the windows
    of the training. For the comparison, the after window starts one window later than the before window, so that the
    two members of every pair hold different images. For the keypoint detection, every image is the box of one object
    (a folder of object images cut by a detector), turned into its crop.

    Args:
        image_folder (Path): The folder holding the images.
        experiment_folder (Path): The trained experiment, whose configuration gives the task, the input size, the
            resizing mode, the sequence length and the crop settings.

    Returns:
        List[Dict[str, np.ndarray]]: One request per image of the folder, the float32 BGR batch in [0, 255] of every
            input of the task: {'images': [1, 3, image_size, image_size]}, {'images': [1, T, 3, image_size,
            image_size]} for the sequence classification, labelling and frame selection, the 'before_images' and
            'after_images' windows for the comparison, and {'crops': [1, 3, crop height, crop width]} for the
            keypoint detection.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=experiment_folder / "training_configuration.yaml")
    task = experiment_configuration["task"]
    image_size = model_configuration["Data"]["image_settings"]["size"]
    keep_ratio = model_configuration["Data"]["image_settings"]["keep_ratio"]

    # Every image in the channels first layout of the network, with the place of the original image in it
    images, image_boxes = [], []
    for image_path in get_image_paths(folder=image_folder):
        original_image = cv2.imread(filename=str(image_path))
        image = preprocess_image(image=original_image, image_size=image_size, keep_ratio=keep_ratio)
        images.append(image.transpose(2, 0, 1).astype(np.float32))
        original_height, original_width = original_image.shape[:2]
        image_boxes.append(preprocess_boxes(bounding_boxes=[[0.0, 0.0, float(original_width), float(original_height)]],
                                            image_width=original_width, image_height=original_height,
                                            image_size=image_size, keep_ratio=keep_ratio)[0])

    # Keypoint detection: the whole image is the box of the object of the crop
    if task == "keypoint_detection":
        return [{"crops": cut_inference_crop(image=image, bounding_box=image_box,
                                             crop_settings=model_configuration["Data"]["crop_settings"])}
                for image, image_box in zip(images, image_boxes)]

    # Every image becomes a batch of one element, or starts a window of one element for the sequence tasks, every
    # further input starting one window later
    if task not in SEQUENCE_TASKS:
        return [{"images": image[None]} for image in images]
    sequence_length = model_configuration["Data"]["sequence_settings"]["sequence_length"]
    return [{input_name: build_frame_window(images=images, window_start=window_start + input_index * sequence_length,
                                            sequence_length=sequence_length)[None]
             for input_index, input_name in enumerate(TASK_INPUT_NAMES[task])} for window_start in range(len(images))]


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
                               device=torch.device("cuda"), use_averaged_weights=False)
    inference_model = build_inference_model(task=experiment_configuration["task"], model=model)
    return inference_model, get_export_specification(task=experiment_configuration["task"],
                                                     inference_model=inference_model)


def load_decoding_settings(experiment_folder: Path, iteration: Optional[int],
                           inference_model: nn.Module) -> Optional[Dict[str, Any]]:
    """
    Gives the decoding settings of a frame selection run, so that its local inference decodes the chosen frames after
    the network exactly like its Triton ensembles, or None for the runs of the other tasks.

    Args:
        experiment_folder (Path): The trained experiment holding 'Weights' and 'training_configuration.yaml'.
        iteration (Optional[int]): The checkpoint iteration, None for the last saved one.
        inference_model (nn.Module): The inference model of the run, wrapping the trained model.

    Returns:
        Optional[Dict[str, Any]]: The decoding settings (frame_selection_deployment.build_decoding_settings), or None.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=experiment_folder / "training_configuration.yaml")
    if experiment_configuration["task"] != "frame_selection":
        return None
    return build_decoding_settings(experiment_configuration=experiment_configuration,
                                   model_configuration=model_configuration,
                                   unselected_class_index=inference_model.model.get_unselected_class_index(),
                                   checkpoint_path=get_checkpoint_path(experiment_folder=experiment_folder,
                                                                       iteration=iteration),
                                   experiment_folder=experiment_folder)


def get_served_model(specification: Dict[str, Any], export_format: str,
                     decoding_settings: Optional[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """
    Names the Triton model the clients query for an export format, with its outputs: the exported model itself, or
    for a frame selector its ensemble returning the chosen frames.

    Args:
        specification (Dict[str, Any]): The export specification of the task.
        export_format (str): Either 'torchscript' or 'onnx'.
        decoding_settings (Optional[Dict[str, Any]]): The decoding settings of a frame selector, None otherwise.

    Returns:
        Tuple[str, List[str]]: The Triton model name and the names of its outputs, in order.
    """
    model_name = f"{specification['model_name']}_{export_format}"
    if decoding_settings is None:
        return model_name, [output["name"] for output in specification["outputs"]]
    return f"{model_name}{ENSEMBLE_MODEL_SUFFIX}", list(DECODER_OUTPUT_TYPES)


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
    Creates a system shared memory region holding the inputs of one request and registers it with Triton, so that the
    images are written once to memory shared with the server instead of being copied into every gRPC request (12 MB
    per image of 1024 pixels, times the number of frames of a window for the sequence models, times two windows for
    the comparison). Triton must share the memory of the machine ('ipc: host' in deployment/triton/docker-compose.yml).

    Args:
        client (triton_client.InferenceServerClient): The gRPC client of the Triton server.
        byte_size (int): The size of all the inputs of a request in bytes.

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


def run_triton_inference(inputs: Dict[str, np.ndarray], client: triton_client.InferenceServerClient,
                         model_name: str, output_names: List[str], shared_memory_handle: Optional[Any],
                         region_name: Optional[str]) -> List[np.ndarray]:
    """
    Sends the preprocessed inputs of a request to a model served by Triton over gRPC and returns its outputs. With a
    shared memory region, the inputs are written to it one after the other, and the request only tells Triton where
    to read each of them.

    Args:
        inputs (Dict[str, np.ndarray]): The float32 [B, 3, H, W] (or [B, T, 3, H, W]) batch of every input, keyed by
            the input name of the served model.
        client (triton_client.InferenceServerClient): The gRPC client of the Triton server.
        model_name (str): The Triton model name.
        output_names (List[str]): The served output names, from get_triton_output_names.
        shared_memory_handle (Optional[Any]): The handle of the input shared memory region, None to send the inputs
            inside the request.
        region_name (Optional[str]): The name of the input shared memory region registered with Triton, None to send
            the inputs inside the request.

    Returns:
        List[np.ndarray]: The outputs, in the order of output_names.
    """
    # One input per name, sent inside the request or written to the shared memory region after the previous inputs
    inference_inputs = []
    byte_offset = 0
    for input_name, input_batch in inputs.items():
        inference_input = triton_client.InferInput(name=input_name, shape=list(input_batch.shape), datatype="FP32")
        if shared_memory_handle is None or region_name is None:
            inference_input.set_data_from_numpy(input_batch)
        else:
            # Write the input right after the previous one in the region and point the request to it
            shared_memory.set_shared_memory_region(shm_handle=shared_memory_handle, input_values=[input_batch],
                                                   offset=byte_offset)
            inference_input.set_shared_memory(region_name=region_name, byte_size=input_batch.nbytes,
                                              offset=byte_offset)
            byte_offset += input_batch.nbytes
        inference_inputs.append(inference_input)

    result = client.infer(model_name=model_name, inputs=inference_inputs,
                          outputs=[triton_client.InferRequestedOutput(name=name) for name in output_names])
    return [result.as_numpy(name=name) for name in output_names]


def run_local_inference(inputs: Dict[str, np.ndarray], inference_model: nn.Module,
                        decoding_settings: Optional[Dict[str, Any]]) -> List[np.ndarray]:
    """
    Runs the preprocessed inputs of a request through the inference model in this process, without Triton, every
    input being passed to the forward argument of the same name. A frame selector then decodes the chosen frames of
    every sequence on the CPU, with the function run by its Triton decoder.

    Args:
        inputs (Dict[str, np.ndarray]): The float32 [B, 3, H, W] (or [B, T, 3, H, W]) batch of every input, keyed by
            input name.
        inference_model (nn.Module): The inference model on the GPU.
        decoding_settings (Optional[Dict[str, Any]]): The decoding settings of a frame selector, None otherwise.

    Returns:
        List[np.ndarray]: The outputs of the inference model, in the order of its export specification, or the
            outputs of the decoder (DECODER_OUTPUT_TYPES) stacked over the sequences for a frame selector.
    """
    with torch.no_grad():
        outputs = inference_model(**{input_name: torch.from_numpy(input_batch).to(device=torch.device("cuda"))
                                     for input_name, input_batch in inputs.items()})
    if decoding_settings is None:
        return convert_outputs_to_numpy(outputs=outputs)

    # Selection of every sequence of the request, decoded from its [T, K] class probabilities
    sequence_outputs = [run_frame_selection(probabilities=sequence_probabilities,
                                            decoding_settings=decoding_settings)
                        for sequence_probabilities in convert_outputs_to_numpy(outputs=outputs)[0]]
    return [np.stack([sequence_output[output_name] for sequence_output in sequence_outputs])
            for output_name in DECODER_OUTPUT_TYPES]


def measure_local_throughput(inputs: List[Dict[str, np.ndarray]], inference_model: nn.Module,
                             decoding_settings: Optional[Dict[str, Any]], number_iterations: int,
                             number_warmup_iterations: int) -> None:
    """
    Measures the raw throughput of the local inference model, one image (or window, or pair of windows) per call, the
    tqdm bar showing the calls per second, the decoding of a frame selector included as in its Triton ensembles. The
    first calls are left out, while CUDA is still initialising.

    Args:
        inputs (List[Dict[str, np.ndarray]]): The requests, from load_inference_inputs, sent in turn.
        inference_model (nn.Module): The inference model on the GPU.
        decoding_settings (Optional[Dict[str, Any]]): The decoding settings of a frame selector, None otherwise.
        number_iterations (int): The number of timed calls.
        number_warmup_iterations (int): The number of untimed calls made first.
    """
    for iteration_index in range(number_warmup_iterations):
        run_local_inference(inputs=inputs[iteration_index % len(inputs)], inference_model=inference_model,
                            decoding_settings=decoding_settings)

    for iteration_index in tqdm(range(number_iterations), desc="local", unit="sample"):
        run_local_inference(inputs=inputs[iteration_index % len(inputs)], inference_model=inference_model,
                            decoding_settings=decoding_settings)


def measure_triton_throughput(inputs: List[Dict[str, np.ndarray]], model_name: str, output_names: List[str], url: str,
                              use_shared_memory: bool, number_iterations: int, number_warmup_iterations: int) -> None:
    """
    Measures the raw throughput of a model served by Triton, one image (or window, or pair of windows) per request,
    the tqdm bar showing the requests per second. The first requests are left out, while TensorRT and CUDA are still
    initialising.

    Args:
        inputs (List[Dict[str, np.ndarray]]): The requests, from load_inference_inputs, sent in turn.
        model_name (str): The Triton model name, e.g. 'object_detector_torchscript', 'image_classifier_onnx' or
            'frame_selector_onnx_ensemble'.
        output_names (List[str]): The output names of the export specification of the model, in order.
        url (str): The Triton gRPC address, e.g. 'localhost:8001'.
        use_shared_memory (bool): Whether Triton reads the images from system shared memory instead of receiving them
            inside the gRPC requests.
        number_iterations (int): The number of timed requests.
        number_warmup_iterations (int): The number of untimed requests made first.
    """
    client = triton_client.InferenceServerClient(url=url)
    served_output_names = get_triton_output_names(client=client, model_name=model_name, output_names=output_names)

    # One shared memory region sized for all the inputs of one request serves every request, when shared memory is
    # used
    shared_memory_handle, region_name = None, None
    if use_shared_memory:
        shared_memory_handle, region_name = create_triton_input_shared_memory(
            client=client, byte_size=sum(input_batch.nbytes for input_batch in inputs[0].values()))

    # The region is always released, even when a request fails
    try:
        for iteration_index in range(number_warmup_iterations):
            run_triton_inference(inputs=inputs[iteration_index % len(inputs)], client=client, model_name=model_name,
                                 output_names=served_output_names, shared_memory_handle=shared_memory_handle,
                                 region_name=region_name)

        for iteration_index in tqdm(range(number_iterations), desc=model_name, unit="sample"):
            run_triton_inference(inputs=inputs[iteration_index % len(inputs)], client=client, model_name=model_name,
                                 output_names=served_output_names, shared_memory_handle=shared_memory_handle,
                                 region_name=region_name)
    finally:
        if shared_memory_handle is not None and region_name is not None:
            release_triton_input_shared_memory(client=client, shared_memory_handle=shared_memory_handle,
                                               region_name=region_name)
