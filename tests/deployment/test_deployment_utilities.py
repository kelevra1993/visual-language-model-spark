import json
import os
import cv2
import numpy as np
import pytest
import tritonclient.grpc as triton_client
import tritonclient.utils.shared_memory as shared_memory
import yaml

from pathlib import Path
from typing import Dict, List

from vision_training.deployment.deployment_utilities import (build_frame_window, load_inference_inputs,
                                                             run_triton_inference)

CONFIGURATIONS_FOLDER = Path(__file__).parents[2] / "vision_training" / "configurations"

# Synthetic configuration of every task whose requests are tested
TASK_CONFIGURATION_FILES = {"classification": "classification/configuration_synthetic_shapes.yaml",
                            "sequence_classification": "sequence_classification/configuration_synthetic_sequences.yaml",
                            "sequence_comparison": "sequence_comparison/configuration_synthetic_pairs.yaml"}

# Side of the network input of the test experiments, number of frames of their windows, and number of images of the
# inference folder, the image of index i being filled with the value 10 * (i + 1)
TEST_IMAGE_SIZE = 32
TEST_SEQUENCE_LENGTH = 4
TEST_NUMBER_IMAGES = 5


class RecordingResult:
    """
    Stand-in for the result of a Triton request, giving zero class probabilities for every requested output.
    """

    def as_numpy(self, name: str) -> np.ndarray:
        """
        Returns the class probabilities of the single sample of the request.

        Args:
            name (str): The served output name.

        Returns:
            np.ndarray: The [1, 2] zero class probabilities.
        """
        return np.zeros(shape=(1, 2), dtype=np.float32)


class RecordingClient:
    """
    Stand-in for the gRPC client of the Triton server, recording the inputs of every request instead of sending them,
    so that the requests built by the deployment client are tested without a running server.
    """

    def __init__(self) -> None:
        """
        Starts with no recorded request.
        """
        self.recorded_inputs = []

    def infer(self, model_name: str, inputs: List[triton_client.InferInput],
              outputs: List[triton_client.InferRequestedOutput]) -> RecordingResult:
        """
        Records the inputs of a request.

        Args:
            model_name (str): The Triton model name.
            inputs (List[triton_client.InferInput]): The inputs of the request.
            outputs (List[triton_client.InferRequestedOutput]): The requested outputs.

        Returns:
            RecordingResult: The stand-in result of the request.
        """
        self.recorded_inputs.append(inputs)
        return RecordingResult()


def write_test_experiment(experiment_folder: Path, task: str) -> None:
    """
    Writes the training configuration of a trained experiment of a task, as read by the deployment client: the
    synthetic configuration of the task with 32 pixel inputs, windows of 4 frames for the sequence tasks, and a label
    mapping of 3 classes.

    Args:
        experiment_folder (Path): The experiment folder receiving 'training_configuration.yaml'.
        task (str): The task of the experiment, a key of TASK_CONFIGURATION_FILES.
    """
    experiment_folder.mkdir(parents=True)
    mapping_path = experiment_folder / "label_mapping.json"
    mapping_path.write_text(data=json.dumps(obj={"1": 0, "2": 1, "3": 2}))

    # Small inputs, short windows and the written label mapping
    configuration = yaml.safe_load(stream=(CONFIGURATIONS_FOLDER / TASK_CONFIGURATION_FILES[task]).read_text())
    configuration["Data"]["image_settings"]["size"] = TEST_IMAGE_SIZE
    if "sequence_settings" in configuration["Data"]:
        configuration["Data"]["sequence_settings"]["sequence_length"] = TEST_SEQUENCE_LENGTH
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(mapping_path)
    (experiment_folder / "training_configuration.yaml").write_text(data=yaml.safe_dump(data=configuration))


@pytest.mark.parametrize(argnames=("task", "expected_images"), argvalues=[
    ("classification", {"images": [2]}), ("sequence_classification", {"images": [2, 3, 4, 0]}),
    ("sequence_comparison", {"before_images": [2, 3, 4, 0], "after_images": [1, 2, 3, 4]})])
def test_requests_hold_the_inputs_of_the_task(tmp_path: Path, task: str,
                                              expected_images: Dict[str, List[int]]) -> None:
    """
    Verifies the requests built from the inference folder for every kind of task: one request per image of the
    folder, keyed by the input names of the task, holding one image, one window of consecutive images cycling through
    the folder, or for the comparison a before window and an after window starting one window later.

    Args:
        tmp_path (Path): The pytest temporary directory.
        task (str): The task of the experiment.
        expected_images (Dict[str, List[int]]): The indices of the images held by every input of the third request.
    """
    # Inference folder of uniform images, and the trained experiment of the task
    image_folder = tmp_path / "inference_data"
    image_folder.mkdir()
    for image_index in range(TEST_NUMBER_IMAGES):
        cv2.imwrite(filename=str(image_folder / f"image_{image_index:06d}.png"),
                    img=np.full(shape=(16, 16, 3), fill_value=10 * (image_index + 1), dtype=np.uint8))
    write_test_experiment(experiment_folder=tmp_path / "experiment", task=task)

    requests = load_inference_inputs(image_folder=image_folder, experiment_folder=tmp_path / "experiment")

    # One request per image, keyed by the input names of the task, the third one checked image by image
    assert len(requests) == TEST_NUMBER_IMAGES
    assert all(list(request) == list(expected_images) for request in requests)
    for input_name, image_indices in expected_images.items():
        input_batch = requests[2][input_name]
        assert input_batch.dtype == np.float32
        # Images are [1, 3, H, W] batches, windows [1, T, 3, H, W] batches
        if task == "classification":
            assert input_batch.shape == (1, 3, TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)
            centre_values = [input_batch[0, 0, TEST_IMAGE_SIZE // 2, TEST_IMAGE_SIZE // 2]]
        else:
            assert input_batch.shape == (1, TEST_SEQUENCE_LENGTH, 3, TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)
            centre_values = input_batch[0, :, 0, TEST_IMAGE_SIZE // 2, TEST_IMAGE_SIZE // 2].tolist()
        assert centre_values == [10 * (image_index + 1) for image_index in image_indices]


def test_frame_windows_cycle_through_the_folder() -> None:
    """
    Verifies that a window going past the last image of the folder continues from its first image.
    """
    # Window of 4 frames starting at the last of 3 images
    images = [np.full(shape=(3, 2, 2), fill_value=image_index, dtype=np.float32) for image_index in range(3)]
    window = build_frame_window(images=images, window_start=2, sequence_length=4)
    assert window.shape == (4, 3, 2, 2)
    assert window[:, 0, 0, 0].tolist() == [2, 0, 1, 2]


def test_triton_requests_write_every_input_at_its_offset() -> None:
    """
    Verifies the requests sent to Triton for a model with several inputs: one input per name in the order of the
    request, and with shared memory every input written right after the previous one in the region, each input of the
    request pointing to its own offset; without shared memory, the same named inputs travel inside the request.
    """
    before_images = np.full(shape=(1, 4, 3, 8, 8), fill_value=1.0, dtype=np.float32)
    after_images = np.full(shape=(1, 2, 3, 8, 8), fill_value=2.0, dtype=np.float32)
    inputs = {"before_images": before_images, "after_images": after_images}
    client = RecordingClient()

    # One region holding both inputs, read back at the offset of every input before it is freed
    region_name = f"test_inference_input_{os.getpid()}"
    shared_memory_handle = shared_memory.create_shared_memory_region(
        triton_shm_name=region_name, shm_key=f"/{region_name}", byte_size=before_images.nbytes + after_images.nbytes)
    try:
        outputs = run_triton_inference(inputs=inputs, client=client, model_name="sequence_comparator_onnx",
                                       output_names=["class_probabilities"], shared_memory_handle=shared_memory_handle,
                                       region_name=region_name)
        written_inputs = [shared_memory.get_contents_as_numpy(shm_handle=shared_memory_handle, datatype=np.float32,
                                                              shape=list(input_batch.shape), offset=offset).copy()
                          for input_batch, offset in [(before_images, 0), (after_images, before_images.nbytes)]]
    finally:
        shared_memory.destroy_shared_memory_region(shm_handle=shared_memory_handle)

    # Both inputs written one after the other, each input of the request pointing to its own offset
    assert np.array_equal(a1=written_inputs[0], a2=before_images) and np.array_equal(a1=written_inputs[1],
                                                                                     a2=after_images)
    assert outputs[0].shape == (1, 2)
    recorded_inputs = client.recorded_inputs[0]
    assert [recorded_input.name() for recorded_input in recorded_inputs] == ["before_images", "after_images"]
    assert [recorded_input.shape() for recorded_input in recorded_inputs] == [[1, 4, 3, 8, 8], [1, 2, 3, 8, 8]]
    # The offsets are only exposed by the request message of every input
    assert [recorded_input._get_tensor().parameters["shared_memory_offset"].int64_param
            for recorded_input in recorded_inputs] == [0, before_images.nbytes]

    # Without shared memory, the same named inputs are sent inside the request
    run_triton_inference(inputs=inputs, client=client, model_name="sequence_comparator_onnx",
                         output_names=["class_probabilities"], shared_memory_handle=None, region_name=None)
    assert [recorded_input.name() for recorded_input in client.recorded_inputs[1]] == ["before_images",
                                                                                      "after_images"]
