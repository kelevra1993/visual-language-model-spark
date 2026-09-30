import json
import cv2
import numpy as np
import torch
import tritonclient.grpc as triton_client

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from model.inference_model import InferenceDetector, get_checkpoint_path, load_trained_model
from utilities.data_utilities import preprocess_image, restore_boxes_to_original_image
from utilities.model.model_utilities import add_bounding_boxes
from utilities.os_utilities import load_experiment_configuration, print_color

# Image file extensions read from an input folder
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# Names of the detector outputs, in the order returned by the exported models
OUTPUT_NAMES = ["detection_boxes", "detection_scores", "detection_labels", "number_detections"]


def collect_image_paths(inputs: List[Path]) -> List[Path]:
    """
    Lists the images to run the detector on, from image files and folders given on the command line.

    Args:
        inputs (List[Path]): Image files and folders of images.

    Returns:
        List[Path]: The sorted image paths.
    """
    image_paths = []
    for input_path in inputs:
        if input_path.is_dir():
            image_paths += sorted(path for path in input_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        else:
            image_paths.append(input_path)
    return image_paths


def load_class_names(labels_file: Optional[Path]) -> Dict[int, str]:
    """
    Reads the class names from the categories of a COCO annotation file, to caption the detections; without a file
    the detections are captioned with their class index.

    Args:
        labels_file (Optional[Path]): A COCO 'labels.json' file, or None.

    Returns:
        Dict[int, str]: The class name of every class index.
    """
    if labels_file is None:
        return {}
    categories = json.loads(labels_file.read_text())["categories"]
    return {category["id"]: category["name"] for category in categories}


def preprocess_batch(image_paths: List[Path], image_size: int, keep_ratio: bool) -> Tuple[
        np.ndarray, List[np.ndarray]]:
    """
    Reads and preprocesses images exactly like the data loaders (resize and centred padding to the square network
    input), so that the deployed model receives the same kind of input as during training.

    Args:
        image_paths (List[Path]): The images of the batch.
        image_size (int): The size of the square network input.
        keep_ratio (bool): Whether to keep the aspect ratio and pad the images.

    Returns:
        Tuple[np.ndarray, List[np.ndarray]]: The float32 BGR batch [B, 3, image_size, image_size] in [0, 255] and the
            original images.
    """
    original_images = [cv2.imread(filename=str(image_path)) for image_path in image_paths]
    preprocessed_images = [preprocess_image(image=image, image_size=image_size, keep_ratio=keep_ratio)
                           for image in original_images]
    batch = np.stack([image.transpose(2, 0, 1) for image in preprocessed_images]).astype(np.float32)
    return batch, original_images


def run_triton_inference(batch: np.ndarray, model_name: str, url: str) -> List[np.ndarray]:
    """
    Sends a preprocessed batch to a detector served by Triton over gRPC and returns its outputs.

    Args:
        batch (np.ndarray): The float32 [B, 3, H, W] batch.
        model_name (str): The Triton model name ('object_detector_torchscript' or 'object_detector_onnx').
        url (str): The Triton gRPC address, e.g. 'localhost:8001'.

    Returns:
        List[np.ndarray]: The boxes [B, K, 4], scores [B, K], labels [B, K] and numbers of detections [B].
    """
    client = triton_client.InferenceServerClient(url=url)

    # TorchScript outputs are named '<name>__<index>', ONNX outputs use the plain names
    served_output_names = [output.name for output in client.get_model_metadata(model_name=model_name).outputs]
    output_names = [next(name for name in served_output_names if name.split("__")[0] == output_name)
                    for output_name in OUTPUT_NAMES]

    inference_input = triton_client.InferInput(name="images", shape=list(batch.shape), datatype="FP32")
    inference_input.set_data_from_numpy(batch)
    result = client.infer(model_name=model_name, inputs=[inference_input],
                          outputs=[triton_client.InferRequestedOutput(name=name) for name in output_names])
    return [result.as_numpy(name=name) for name in output_names]


def run_local_inference(batch: np.ndarray, detector: InferenceDetector) -> List[np.ndarray]:
    """
    Runs the same preprocessed batch through the inference detector in this process, without Triton, to check or
    replace the served models.

    Args:
        batch (np.ndarray): The float32 [B, 3, H, W] batch.
        detector (InferenceDetector): The inference detector on the GPU.

    Returns:
        List[np.ndarray]: The boxes [B, K, 4], scores [B, K], labels [B, K] and numbers of detections [B].
    """
    with torch.no_grad():
        outputs = detector(images=torch.from_numpy(batch).to(device=torch.device("cuda")))
    return [tensor.cpu().numpy() for tensor in outputs]


def annotate_image(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
                   class_names: Dict[int, str]) -> np.ndarray:
    """
    Draws the detections, with their class name and score, on a copy of the original image.

    Args:
        image (np.ndarray): The original BGR image.
        boxes (np.ndarray): The [N, 4] detection boxes in original image pixels.
        scores (np.ndarray): The [N] detection scores.
        labels (np.ndarray): The [N] detection labels.
        class_names (Dict[int, str]): The class name of every class index.

    Returns:
        np.ndarray: The annotated image.
    """
    annotated_image = image.copy()
    if len(boxes) > 0:
        annotated_image = add_bounding_boxes(bounding_boxes=boxes, image=annotated_image,
                                             input_image_size=max(image.shape[:2]), color=(0, 0, 255))

    # Caption every box with its class and score above its top left corner
    for box, score, label in zip(boxes, scores, labels):
        caption = f"{class_names.get(int(label), str(int(label)))} {score:.2f}"
        cv2.putText(img=annotated_image, text=caption, org=(int(box[0]), max(int(box[1]) - 5, 12)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1)
    return annotated_image


def detect_images(image_paths: List[Path], experiment_folder: Path, backend: str, model_name: str, url: str,
                  output_folder: Path, labels_file: Optional[Path], batch_size: int,
                  iteration: Optional[int]) -> Dict[str, List[Dict[str, object]]]:
    """
    Detects the objects of a list of images with the Triton-served detector or the local inference detector, maps
    the detections back to the original images, saves annotated copies and prints the detections.

    Args:
        image_paths (List[Path]): The images to process.
        experiment_folder (Path): The trained experiment, whose configuration gives the input size and resizing mode
            (and whose weights are used by the local backend).
        backend (str): 'triton' or 'local'.
        model_name (str): The Triton model name, used by the Triton backend.
        url (str): The Triton gRPC address, used by the Triton backend.
        output_folder (Path): The folder receiving the annotated images.
        labels_file (Optional[Path]): A COCO 'labels.json' giving the class names, or None.
        batch_size (int): The number of images sent per request.
        iteration (Optional[int]): The checkpoint iteration used by the local backend, None for the last one.

    Returns:
        Dict[str, List[Dict[str, object]]]: For every image name, its detections with 'label', 'score' and 'box'.
    """
    _, model_configuration = load_experiment_configuration(
        configuration_path=experiment_folder / "training_configuration.yaml")
    image_size = model_configuration["Data"]["image_settings"]["size"]
    keep_ratio = model_configuration["Data"]["image_settings"]["keep_ratio"]
    class_names = load_class_names(labels_file=labels_file)

    # The local backend loads the trained weights once for all batches
    detector = None
    if backend == "local":
        checkpoint_path = get_checkpoint_path(experiment_folder=experiment_folder, iteration=iteration)
        detector = InferenceDetector(model=load_trained_model(
            configuration_path=experiment_folder / "training_configuration.yaml", checkpoint_path=checkpoint_path,
            device=torch.device("cuda"))).eval()

    output_folder.mkdir(parents=True, exist_ok=True)
    all_detections = {}

    # Process the images batch by batch
    for batch_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_start:batch_start + batch_size]
        batch, original_images = preprocess_batch(image_paths=batch_paths, image_size=image_size,
                                                  keep_ratio=keep_ratio)
        if backend == "triton":
            boxes, scores, labels, numbers = run_triton_inference(batch=batch, model_name=model_name, url=url)
        else:
            boxes, scores, labels, numbers = run_local_inference(batch=batch, detector=detector)

        # Keep the valid detections of every image and express them in original image pixels
        for image_index, (image_path, original_image) in enumerate(zip(batch_paths, original_images)):
            number_detections = int(np.array(numbers[image_index]).reshape(-1)[0])
            image_height, image_width = original_image.shape[:2]
            image_boxes = restore_boxes_to_original_image(bounding_boxes=boxes[image_index, :number_detections],
                                                          image_width=image_width, image_height=image_height,
                                                          image_size=image_size, keep_ratio=keep_ratio)
            image_scores = scores[image_index, :number_detections]
            image_labels = labels[image_index, :number_detections]

            annotated_image = annotate_image(image=original_image, boxes=image_boxes, scores=image_scores,
                                             labels=image_labels, class_names=class_names)
            output_path = output_folder / f"{image_path.stem}_detections.png"
            cv2.imwrite(filename=str(output_path), img=annotated_image)

            all_detections[image_path.name] = [
                {"label": class_names.get(int(label), str(int(label))), "score": round(float(score), 4),
                 "box": [round(float(value), 1) for value in box]}
                for box, score, label in zip(image_boxes, image_scores, image_labels)]
            print_color(color="blue", output=f"{image_path.name} : {number_detections} detections : "
                                             f"file://{output_path.absolute()}")
            for detection in all_detections[image_path.name]:
                print_color(color="green", output=f"{detection['label']:<10} {detection['score']:.2f} "
                                                  f"{detection['box']}", indent=1)

    return all_detections


if __name__ == "__main__":
    # Trained experiment giving the input size and resizing mode (and the weights of the local backend)
    experiment_folder = Path(__file__).absolute().parents[2] / "experiments" / "synthetic_shapes"
    # Image files and folders of images to run the detector on
    image_inputs = [Path("/home/robert_kelevra/Data/object_detection_data/synthetic_shapes/test/data")]
    # 'triton' to query the Triton server started with deployment/triton/docker-compose.yml, 'local' to run in process
    backend = "triton"
    # Triton model to query, 'object_detector_torchscript' or 'object_detector_onnx'
    model_name = "object_detector_torchscript"
    # Triton gRPC address
    url = "localhost:8001"
    # Folder receiving the annotated images
    output_folder = experiment_folder / "Detections"
    # COCO annotation file giving the class names, None to caption the detections with their class index
    labels_file = Path("/home/robert_kelevra/Data/object_detection_data/synthetic_shapes/test/labels.json")
    # Number of images sent per request
    batch_size = 8
    # Checkpoint iteration used by the local backend, None for the last saved one
    iteration = None

    detect_images(image_paths=collect_image_paths(inputs=image_inputs), experiment_folder=experiment_folder,
                  backend=backend, model_name=model_name, url=url, output_folder=output_folder,
                  labels_file=labels_file, batch_size=batch_size, iteration=iteration)
