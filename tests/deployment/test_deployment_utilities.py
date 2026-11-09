"""
Tests of the Triton deployment client of vision_training/deployment/deployment_utilities.py, with a recording stand-in
for the gRPC client: the request built for the comparison (the two halves of the folder), the folders longer than the
sequences of the training refused, and the inputs written in shared memory at their offsets.
"""

import json
import os
import cv2
import numpy as np
import pytest
import tritonclient.grpc as triton_client
import tritonclient.utils.shared_memory as shared_memory
import yaml

from pathlib import Path
from typing import Dict, List, Optional

from vision_training.data.synthetic.create_synthetic_dataset import build_keypoint_label_mapping, build_keypoint_labels
from vision_training.deployment.deployment_utilities import load_inference_inputs, run_triton_inference

CONFIGURATIONS_FOLDER = Path(__file__).parents[2] / "vision_training" / "configurations"

# Synthetic configuration of every task whose requests are tested
TASK_CONFIGURATION_FILES = {"classification": "classification/configuration_synthetic_shapes.yaml",
                            "semantic_segmentation": "semantic_segmentation/configuration_synthetic_shapes.yaml",
                            "sequence_classification": "sequence_classification/configuration_synthetic_sequences.yaml",
                            "sequence_comparison": "sequence_comparison/configuration_synthetic_pairs.yaml",
                            "frame_selection": "frame_selection/configuration_synthetic_sequences.yaml",
                            "keypoint_detection": "keypoint_detection/configuration_synthetic_shapes.yaml"}

# Side of the network input of the test experiments, maximum number of frames of their sequences, and number of images
# of the inference folder, the image of index i being filled with the value 10 * (i + 1)
TEST_IMAGE_SIZE = 32
TEST_MAXIMUM_SEQUENCE_LENGTH = 5
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


def write_test_experiment(experiment_folder: Path, task: str,
                          maximum_sequence_length: int = TEST_MAXIMUM_SEQUENCE_LENGTH,
                          keep_ratio: Optional[bool] = None) -> None:
    """
    Writes the training configuration of a trained experiment of a task, as read by the deployment client and the
    export: the synthetic configuration of the task with 32 pixel inputs, sequences of at most 5 frames, and a label
    mapping of 3 classes. The keypoint detection gets the training labels of the synthetic shapes whose keypoint
    definitions the configuration loader reads, with the label mapping of the synthetic dataset.

    Args:
        experiment_folder (Path): The experiment folder receiving 'training_configuration.yaml'.
        task (str): The task of the experiment, a key of TASK_CONFIGURATION_FILES.
        maximum_sequence_length (int): The maximum sequence length of the sequence tasks. Defaults to
            TEST_MAXIMUM_SEQUENCE_LENGTH.
        keep_ratio (Optional[bool]): The 'Data.image_settings.keep_ratio' of the experiment, None to keep the one of
            the synthetic configuration. Defaults to None.
    """
    experiment_folder.mkdir(parents=True)
    mapping_path = experiment_folder / "label_mapping.json"
    mapping_path.write_text(data=json.dumps(obj={"1": 0, "2": 1, "3": 2}))

    # Small inputs, the resize asked for, short sequences and the written label mapping
    configuration = yaml.safe_load(stream=(CONFIGURATIONS_FOLDER / TASK_CONFIGURATION_FILES[task]).read_text())
    configuration["Data"]["image_settings"]["size"] = TEST_IMAGE_SIZE
    if keep_ratio is not None:
        configuration["Data"]["image_settings"]["keep_ratio"] = keep_ratio
    if "sequence_settings" in configuration["Data"]:
        configuration["Data"]["sequence_settings"]["maximum_sequence_length"] = maximum_sequence_length

    # Keypoint detection: the synthetic shape categories as the training labels
    if task == "keypoint_detection":
        categories = build_keypoint_labels(image_records=[], image_size=TEST_IMAGE_SIZE,
                                           split_name="train")[0]["categories"]
        train_folder = experiment_folder / "train"
        train_folder.mkdir()
        (train_folder / "labels.json").write_text(data=json.dumps(obj={"images": [], "annotations": [],
                                                                       "categories": categories}))
        configuration["ExperimentConfiguration"]["train_data_folder"] = str(train_folder)
        mapping_path.write_text(data=json.dumps(obj=build_keypoint_label_mapping()))
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(mapping_path)
    (experiment_folder / "training_configuration.yaml").write_text(data=yaml.safe_dump(data=configuration))


def write_inference_folder(image_folder: Path, number_images: int) -> None:
    """
    Writes the inference folder read by the deployment client, the uniform image of index i being filled with the
    value 10 * (i + 1) so that the tests can tell which images a request holds.

    Args:
        image_folder (Path): The inference folder to create.
        number_images (int): The number of images of the folder.
    """
    image_folder.mkdir()
    for image_index in range(number_images):
        cv2.imwrite(filename=str(image_folder / f"image_{image_index:06d}.png"),
                    img=np.full(shape=(16, 16, 3), fill_value=10 * (image_index + 1), dtype=np.uint8))


@pytest.mark.parametrize(argnames=("task", "expected_requests"), argvalues=[
    ("sequence_comparison", [{"before_images": [0, 1, 2], "after_images": [3, 4]}])])
def test_requests_hold_the_inputs_of_the_task(tmp_path: Path, task: str,
                                              expected_requests: List[Dict[str, List[int]]]) -> None:
    """
    Verifies the requests built from the inference folder for the comparison, keyed by the input names of the task: a
    single request whose before member is the first half of the folder and whose after member is the rest.

    Args:
        tmp_path (Path): The pytest temporary directory.
        task (str): The task of the experiment.
        expected_requests (List[Dict[str, List[int]]]): The indices of the images held by every input of every request.
    """
    write_inference_folder(image_folder=tmp_path / "inference_data", number_images=TEST_NUMBER_IMAGES)
    write_test_experiment(experiment_folder=tmp_path / "experiment", task=task)

    requests = load_inference_inputs(image_folder=tmp_path / "inference_data",
                                     experiment_folder=tmp_path / "experiment")

    # Every request keyed by the input names of the task, checked image by image
    assert len(requests) == len(expected_requests)
    for request, expected_images in zip(requests, expected_requests):
        assert list(request) == list(expected_images)
        for input_name, image_indices in expected_images.items():
            input_batch = request[input_name]
            assert input_batch.dtype == np.float32
            # Images are [1, 3, H, W] batches, sequences [1, T, 3, H, W] batches
            if task in ["classification", "semantic_segmentation", "keypoint_detection"]:
                assert input_batch.shape == (1, 3, TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)
                centre_values = [input_batch[0, 0, TEST_IMAGE_SIZE // 2, TEST_IMAGE_SIZE // 2]]
            else:
                assert input_batch.shape == (1, len(image_indices), 3, TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)
                centre_values = input_batch[0, :, 0, TEST_IMAGE_SIZE // 2, TEST_IMAGE_SIZE // 2].tolist()
            assert centre_values == [10 * (image_index + 1) for image_index in image_indices]


@pytest.mark.parametrize(argnames=("task", "number_images", "expected_message"), argvalues=[
    ("sequence_classification", 5, "gives sequences of up to 5 images, more than the maximum sequence length of the "
                                   "training \\(4\\)")])
def test_sequence_folders_that_do_not_fit_are_refused(tmp_path: Path, task: str, number_images: int,
                                                      expected_message: str) -> None:
    """
    Verifies that the deployment client refuses an inference folder longer than the sequences of the training, which
    the training leaves out.

    Args:
        tmp_path (Path): The pytest temporary directory.
        task (str): The task of the experiment.
        number_images (int): The number of images of the inference folder.
        expected_message (str): The pattern of the expected error.
    """
    write_inference_folder(image_folder=tmp_path / "inference_data", number_images=number_images)
    write_test_experiment(experiment_folder=tmp_path / "experiment", task=task, maximum_sequence_length=4)
    with pytest.raises(expected_exception=ValueError, match=expected_message):
        load_inference_inputs(image_folder=tmp_path / "inference_data", experiment_folder=tmp_path / "experiment")


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
