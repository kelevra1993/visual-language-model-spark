import csv
import json
import time
import cv2
import numpy as np
import torch
import tritonclient.grpc as triton_client

from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from model.inference_model import InferenceDetector, get_checkpoint_path, load_trained_model
from utilities.data_utilities import preprocess_image, restore_boxes_to_original_image
from utilities.os_utilities import get_image_paths, load_experiment_configuration, print_color

# Names of the detector outputs, in the order returned by the exported models
OUTPUT_NAMES = ["detection_boxes", "detection_scores", "detection_labels", "number_detections"]

# Triton model served for every backend name, the 'local' backend runs the inference detector in this process instead
TRITON_MODEL_NAMES = {"torchscript": "object_detector_torchscript", "onnx": "object_detector_onnx"}

# Minimum intersection over union for two detections of the same class, from two backends, to be the same object
MATCHING_INTERSECTION_OVER_UNION = 0.5

# Names of the box coordinates, in the order of the [x1, y1, x2, y2] boxes
COORDINATE_NAMES = ["x_1", "y_1", "x_2", "y_2"]


def load_class_names(labels_file: Path) -> Dict[int, str]:
    """
    Reads the class names from the categories of a COCO annotation file, which give the class columns of the
    comparison files.

    Args:
        labels_file (Path): A COCO 'labels.json' file.

    Returns:
        Dict[int, str]: The class name of every class index.
    """
    categories = json.loads(labels_file.read_text())["categories"]
    return {category["id"]: category["name"] for category in categories}


def preprocess_batch(image_paths: List[Path], image_size: int, keep_ratio: bool) -> Tuple[
    np.ndarray, List[Tuple[int, int]]]:
    """
    Reads and preprocesses images exactly like the data loaders (resize and centred padding to the square network
    input), so that the deployed models receive the same kind of input as during training.

    Args:
        image_paths (List[Path]): The images of the batch.
        image_size (int): The size of the square network input.
        keep_ratio (bool): Whether to keep the aspect ratio and pad the images.

    Returns:
        Tuple[np.ndarray, List[Tuple[int, int]]]: The float32 BGR batch [B, 3, image_size, image_size] in [0, 255] and
            the (height, width) of every original image.
    """
    original_images = [cv2.imread(filename=str(image_path)) for image_path in image_paths]
    preprocessed_images = [preprocess_image(image=image, image_size=image_size, keep_ratio=keep_ratio)
                           for image in original_images]
    batch = np.stack([image.transpose(2, 0, 1) for image in preprocessed_images]).astype(np.float32)
    return batch, [image.shape[:2] for image in original_images]


def get_triton_output_names(client: triton_client.InferenceServerClient, model_name: str) -> List[str]:
    """
    Finds the served names of the detector outputs, read once per model before the timed requests.

    Args:
        client (triton_client.InferenceServerClient): The gRPC client of the Triton server.
        model_name (str): The Triton model name.

    Returns:
        List[str]: The served output names, in the order of OUTPUT_NAMES.
    """
    # TorchScript outputs are named '<name>__<index>', ONNX outputs use the plain names
    served_output_names = [output.name for output in client.get_model_metadata(model_name=model_name).outputs]
    return [next(name for name in served_output_names if name.split("__")[0] == output_name)
            for output_name in OUTPUT_NAMES]


def run_triton_inference(batch: np.ndarray, client: triton_client.InferenceServerClient, model_name: str,
                         output_names: List[str]) -> List[np.ndarray]:
    """
    Sends a preprocessed batch to a detector served by Triton over gRPC and returns its outputs.

    Args:
        batch (np.ndarray): The float32 [B, 3, H, W] batch.
        client (triton_client.InferenceServerClient): The gRPC client of the Triton server.
        model_name (str): The Triton model name.
        output_names (List[str]): The served output names, from get_triton_output_names.

    Returns:
        List[np.ndarray]: The boxes [B, K, 4], scores [B, K], labels [B, K] and numbers of detections [B].
    """
    inference_input = triton_client.InferInput(name="images", shape=list(batch.shape), datatype="FP32")
    inference_input.set_data_from_numpy(batch)
    result = client.infer(model_name=model_name, inputs=[inference_input],
                          outputs=[triton_client.InferRequestedOutput(name=name) for name in output_names])
    return [result.as_numpy(name=name) for name in output_names]


def run_local_inference(batch: np.ndarray, detector: InferenceDetector) -> List[np.ndarray]:
    """
    Runs the same preprocessed batch through the inference detector in this process, without Triton, as the
    reference the served models are compared to.

    Args:
        batch (np.ndarray): The float32 [B, 3, H, W] batch.
        detector (InferenceDetector): The inference detector on the GPU.

    Returns:
        List[np.ndarray]: The boxes [B, K, 4], scores [B, K], labels [B, K] and numbers of detections [B].
    """
    with torch.no_grad():
        outputs = detector(images=torch.from_numpy(batch).to(device=torch.device("cuda")))
    return [tensor.cpu().numpy() for tensor in outputs]


def get_intersection_over_union_matrix(first_boxes: np.ndarray, second_boxes: np.ndarray) -> np.ndarray:
    """
    Computes the intersection over union of every pair of boxes from two backends, used to find which detections of
    two backends are the same object.

    Args:
        first_boxes (np.ndarray): The [N, 4] [x1, y1, x2, y2] boxes of the first backend.
        second_boxes (np.ndarray): The [M, 4] [x1, y1, x2, y2] boxes of the second backend.

    Returns:
        np.ndarray: The [N, M] intersection over union matrix.
    """
    intersection_widths = (np.minimum(first_boxes[:, None, 2], second_boxes[None, :, 2]) -
                           np.maximum(first_boxes[:, None, 0], second_boxes[None, :, 0])).clip(min=0.0)
    intersection_heights = (np.minimum(first_boxes[:, None, 3], second_boxes[None, :, 3]) -
                            np.maximum(first_boxes[:, None, 1], second_boxes[None, :, 1])).clip(min=0.0)
    intersections = intersection_widths * intersection_heights

    # Union of every pair, kept strictly positive for degenerate boxes
    first_areas = (first_boxes[:, 2] - first_boxes[:, 0]) * (first_boxes[:, 3] - first_boxes[:, 1])
    second_areas = (second_boxes[:, 2] - second_boxes[:, 0]) * (second_boxes[:, 3] - second_boxes[:, 1])
    unions = np.maximum(first_areas[:, None] + second_areas[None, :] - intersections, 1e-9)
    return intersections / unions


def match_detections(reference_detections: Dict[str, np.ndarray],
                     other_detections: Dict[str, np.ndarray]) -> Dict[int, int]:
    """
    Pairs the detections of a backend with those of the reference backend, greedily from the most overlapping pair,
    between detections of the same class overlapping by at least MATCHING_INTERSECTION_OVER_UNION. Detections are
    paired by overlap rather than by their position in the output, since two objects with almost equal scores can be
    listed in a different order by two backends.

    Args:
        reference_detections (Dict[str, np.ndarray]): The 'boxes' [N, 4] and 'labels' [N] of the reference backend.
        other_detections (Dict[str, np.ndarray]): The 'boxes' [M, 4] and 'labels' [M] of the compared backend.

    Returns:
        Dict[int, int]: The index of the matched compared detection for every matched reference detection.
    """
    intersection_over_union = get_intersection_over_union_matrix(first_boxes=reference_detections["boxes"],
                                                                 second_boxes=other_detections["boxes"])
    # Detections of different classes are never the same object
    same_class = reference_detections["labels"][:, None] == other_detections["labels"][None, :]
    intersection_over_union = np.where(same_class, intersection_over_union, 0.0)

    # Take the pairs from the most to the least overlapping, each detection being matched at most once
    matches, matched_others = {}, set()
    for flat_index in np.argsort(-intersection_over_union, axis=None):
        reference_index, other_index = np.unravel_index(flat_index, intersection_over_union.shape)
        if intersection_over_union[reference_index, other_index] < MATCHING_INTERSECTION_OVER_UNION:
            break
        if reference_index not in matches and other_index not in matched_others:
            matches[int(reference_index)] = int(other_index)
            matched_others.add(other_index)
    return matches


def build_detection_rows(image_name: str, image_detections: Dict[str, Dict[str, np.ndarray]], backends: List[str],
                         class_names: Dict[int, str]) -> List[Dict[str, object]]:
    """
    Builds the rows of the detection comparison file for one image: one row per object, with its score and box for
    every backend side by side. Objects found by only some backends leave the cells of the others empty.

    Args:
        image_name (str): The image file name.
        image_detections (Dict[str, Dict[str, np.ndarray]]): The 'boxes', 'scores' and 'labels' of every backend.
        backends (List[str]): The compared backends, the first one being the reference.
        class_names (Dict[int, str]): The class name of every class index.

    Returns:
        List[Dict[str, object]]: The rows of the image, keyed by column name.
    """
    reference_backend = backends[0]
    reference_detections = image_detections[reference_backend]

    # One object per reference detection, holding the index of the matching detection of every backend
    objects = [{reference_backend: index} for index in range(len(reference_detections["labels"]))]
    for backend in backends[1:]:
        matches = match_detections(reference_detections=reference_detections,
                                   other_detections=image_detections[backend])
        for reference_index, other_index in matches.items():
            objects[reference_index][backend] = other_index

        # Detections of this backend matching no reference detection are objects of their own
        objects += [{backend: index} for index in range(len(image_detections[backend]["labels"]))
                    if index not in matches.values()]

    # Fill the score and coordinate columns of every backend that found the object
    rows = []
    for detection_object in objects:
        first_backend, first_index = next(iter(detection_object.items()))
        label = int(image_detections[first_backend]["labels"][first_index])
        row = {"image": image_name, "class": class_names.get(label, str(label))}
        row.update({f"score_{backend}": "" for backend in backends})
        row.update({f"{name}_{backend}": "" for name in COORDINATE_NAMES for backend in backends})
        for backend, index in detection_object.items():
            row[f"score_{backend}"] = round(float(image_detections[backend]["scores"][index]), 4)
            for name, value in zip(COORDINATE_NAMES, image_detections[backend]["boxes"][index]):
                row[f"{name}_{backend}"] = round(float(value), 1)
        rows.append(row)
    return rows


def build_image_row(image_name: str, image_detections: Dict[str, Dict[str, np.ndarray]],
                    detection_rows: List[Dict[str, object]], backends: List[str],
                    class_names: Dict[int, str]) -> Dict[str, object]:
    """
    Builds the row of the image comparison file for one image: the number of detections and mean score of every
    class for every backend, and whether the backends agree.

    Args:
        image_name (str): The image file name.
        image_detections (Dict[str, Dict[str, np.ndarray]]): The 'boxes', 'scores' and 'labels' of every backend.
        detection_rows (List[Dict[str, object]]): The detection comparison rows of the image.
        backends (List[str]): The compared backends, the first one being the reference.
        class_names (Dict[int, str]): The class name of every class index.

    Returns:
        Dict[str, object]: The row of the image, keyed by column name.
    """
    row = {"image": image_name}

    # Number of detections and mean score of every class, for every backend
    for label, class_name in class_names.items():
        for backend in backends:
            row[f"{class_name}_count_{backend}"] = int(np.sum(image_detections[backend]["labels"] == label))
        for backend in backends:
            class_scores = image_detections[backend]["scores"][image_detections[backend]["labels"] == label]
            row[f"{class_name}_mean_score_{backend}"] = (round(float(class_scores.mean()), 4)
                                                          if len(class_scores) > 0 else "")

    # The backends agree when every object is found by all of them, the differences are measured on those objects
    complete_rows = [detection_row for detection_row in detection_rows
                     if all(detection_row[f"score_{backend}"] != "" for backend in backends)]
    row["all_objects_match"] = len(complete_rows) == len(detection_rows)
    score_differences = [max(detection_row[f"score_{backend}"] for backend in backends) -
                         min(detection_row[f"score_{backend}"] for backend in backends)
                         for detection_row in complete_rows]
    box_differences = [max(detection_row[f"{name}_{backend}"] for backend in backends) -
                       min(detection_row[f"{name}_{backend}"] for backend in backends)
                       for detection_row in complete_rows for name in COORDINATE_NAMES]
    row["maximum_score_difference"] = round(max(score_differences), 4) if score_differences else ""
    row["maximum_box_difference"] = round(max(box_differences), 1) if box_differences else ""
    return row


def build_speed_rows(batch_durations: Dict[str, List[float]], batch_sizes: List[int],
                     number_warmup_batches: int) -> List[Dict[str, object]]:
    """
    Summarises the timed model calls of every backend, leaving out the first batches, during which TensorRT and CUDA
    are still initialising.

    Args:
        batch_durations (Dict[str, List[float]]): The duration in seconds of every batch, for every backend.
        batch_sizes (List[int]): The number of images of every batch.
        number_warmup_batches (int): The number of first batches left out of the timings.

    Returns:
        List[Dict[str, object]]: One row per backend, keyed by column name.
    """
    rows = []
    for backend, durations in batch_durations.items():
        timed_durations = np.array(durations[number_warmup_batches:]) * 1000.0
        rows.append({"backend": backend, "timed_images": sum(batch_sizes[number_warmup_batches:]),
                     "images_per_second": round(sum(batch_sizes[number_warmup_batches:]) /
                                                (timed_durations.sum() / 1000.0), 1),
                     "mean_batch_milliseconds": round(float(timed_durations.mean()), 2),
                     "median_batch_milliseconds": round(float(np.median(timed_durations)), 2),
                     "percentile_95_batch_milliseconds": round(float(np.percentile(timed_durations, 95)), 2)})
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    """
    Writes rows sharing the same columns to a CSV file, one of the backend comparison files.

    Args:
        path (Path): The CSV file to write.
        rows (List[Dict[str, object]]): The rows, keyed by column name.
    """
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def compare_backends(image_paths: List[Path], experiment_folder: Path, backends: List[str], url: str,
                     labels_file: Path, output_folder: Path, batch_size: int, number_warmup_batches: int,
                     iteration: Optional[int]) -> None:
    """
    Runs the same images through several deployments of the detector (the local inference detector and the models
    served by Triton), times their model calls and writes three comparison files: the detections of every class per
    image, every object side by side, and the speed of every backend.

    Args:
        image_paths (List[Path]): The images to process.
        experiment_folder (Path): The trained experiment, whose configuration gives the input size and resizing mode
            (and whose weights are used by the local backend).
        backends (List[str]): The compared backends among 'local', 'torchscript' and 'onnx', the first one being the
            reference the others are matched to.
        url (str): The Triton gRPC address, used by the Triton backends.
        labels_file (Path): A COCO 'labels.json' giving the class names.
        output_folder (Path): The folder receiving the comparison files.
        batch_size (int): The number of images sent per request.
        number_warmup_batches (int): The number of first batches left out of the timings.
        iteration (Optional[int]): The checkpoint iteration used by the local backend, None for the last one.

    Raises:
        ValueError: If there are not more batches than warm-up batches.
    """
    number_batches = -(-len(image_paths) // batch_size)
    if number_batches <= number_warmup_batches:
        raise ValueError(f"{number_batches} batches for {number_warmup_batches} warm-up batches, use more images.")

    _, model_configuration = load_experiment_configuration(
        configuration_path=experiment_folder / "training_configuration.yaml")
    image_size = model_configuration["Data"]["image_settings"]["size"]
    keep_ratio = model_configuration["Data"]["image_settings"]["keep_ratio"]
    class_names = load_class_names(labels_file=labels_file)

    # The local backend loads the trained weights once for all batches
    detector = None
    if "local" in backends:
        checkpoint_path = get_checkpoint_path(experiment_folder=experiment_folder, iteration=iteration)
        detector = InferenceDetector(model=load_trained_model(
            configuration_path=experiment_folder / "training_configuration.yaml", checkpoint_path=checkpoint_path,
            device=torch.device("cuda"))).eval()

    # The Triton backends share one client, and their output names are read once before the timed requests
    client = triton_client.InferenceServerClient(url=url) if any(backend != "local" for backend in backends) else None
    output_names = {backend: get_triton_output_names(client=client, model_name=TRITON_MODEL_NAMES[backend])
                    for backend in backends if backend != "local"}

    detection_rows, image_rows = [], []
    batch_durations, batch_sizes = {backend: [] for backend in backends}, []

    # Preprocess every batch once and send the same batch to every backend, timing the model calls only
    progress_bar = tqdm(total=len(image_paths), desc=f"Comparing {', '.join(backends)}", unit="image")
    for batch_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_start:batch_start + batch_size]
        batch, image_shapes = preprocess_batch(image_paths=batch_paths, image_size=image_size, keep_ratio=keep_ratio)
        batch_sizes.append(len(batch_paths))

        batch_outputs = {}
        for backend in backends:
            start_time = time.perf_counter()
            if backend == "local":
                batch_outputs[backend] = run_local_inference(batch=batch, detector=detector)
            else:
                batch_outputs[backend] = run_triton_inference(batch=batch, client=client,
                                                              model_name=TRITON_MODEL_NAMES[backend],
                                                              output_names=output_names[backend])
            batch_durations[backend].append(time.perf_counter() - start_time)

        # Keep the valid detections of every image and backend, expressed in original image pixels
        for image_index, (image_path, (image_height, image_width)) in enumerate(zip(batch_paths, image_shapes)):
            image_detections = {}
            for backend, (boxes, scores, labels, numbers) in batch_outputs.items():
                number_detections = int(np.array(numbers[image_index]).reshape(-1)[0])
                image_detections[backend] = {
                    "boxes": restore_boxes_to_original_image(bounding_boxes=boxes[image_index, :number_detections],
                                                             image_width=image_width, image_height=image_height,
                                                             image_size=image_size, keep_ratio=keep_ratio),
                    "scores": scores[image_index, :number_detections],
                    "labels": labels[image_index, :number_detections]}

            image_detection_rows = build_detection_rows(image_name=image_path.name, image_detections=image_detections,
                                                        backends=backends, class_names=class_names)
            detection_rows += image_detection_rows
            image_rows.append(build_image_row(image_name=image_path.name, image_detections=image_detections,
                                              detection_rows=image_detection_rows, backends=backends,
                                              class_names=class_names))
        progress_bar.update(n=len(batch_paths))
    progress_bar.close()

    # Write the three comparison files
    speed_rows = build_speed_rows(batch_durations=batch_durations, batch_sizes=batch_sizes,
                                  number_warmup_batches=number_warmup_batches)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_paths = {"image_comparison.csv": image_rows, "detection_comparison.csv": detection_rows,
                    "speed_comparison.csv": speed_rows}
    for file_name, rows in output_paths.items():
        write_csv(path=output_folder / file_name, rows=rows)

    # Summarise the agreement of the backends and their speed
    number_mismatches = sum(not image_row["all_objects_match"] for image_row in image_rows)
    score_differences = [row["maximum_score_difference"] for row in image_rows if row["maximum_score_difference"] != ""]
    box_differences = [row["maximum_box_difference"] for row in image_rows if row["maximum_box_difference"] != ""]
    print_color(color="green" if number_mismatches == 0 else "yellow",
                output=f"{len(image_rows) - number_mismatches} / {len(image_rows)} images with the same objects for "
                       f"all backends, maximum score difference {max(score_differences, default=0.0)}, maximum box "
                       f"difference {max(box_differences, default=0.0)} px", add_separators=True)
    for speed_row in speed_rows:
        print_color(color="blue", output=f"{speed_row['backend']:<12} {speed_row['images_per_second']:>7} images/s, "
                                         f"median batch {speed_row['median_batch_milliseconds']} ms", indent=1)
    for file_name in output_paths:
        print_color(color="green", output=f"{file_name} : file://{(output_folder / file_name).absolute()}", indent=1)


if __name__ == "__main__":
    # Trained experiment giving the input size and resizing mode (and the weights of the local backend)
    experiment_folder = Path(__file__).absolute().parents[2] / "experiments" / "poseidon_experiment_4"
    split_folder = Path("/home/robert_kelevra/Data/object_detection_data/synthetic_shapes/test")
    # Folder of the images to run the detectors on
    image_folder = split_folder / "data"
    # COCO annotation file giving the class names of the comparison columns
    labels_file = split_folder / "labels.json"
    # Compared backends: 'local' runs the detector in this process, 'torchscript' and 'onnx' query the Triton models
    # started with deployment/triton/docker-compose.yml. The first one is the reference the others are matched to
    backends = ["local", "torchscript", "onnx"]
    # Triton gRPC address
    url = "localhost:8001"
    # Folder receiving image_comparison.csv, detection_comparison.csv and speed_comparison.csv
    output_folder = experiment_folder / "Backend_Comparison"
    # Number of images sent per request
    batch_size = 8
    # Number of first batches left out of the timings, while TensorRT and CUDA are still initialising
    number_warmup_batches = 3
    # Checkpoint iteration used by the local backend, None for the last saved one
    iteration = None

    compare_backends(image_paths=get_image_paths(folder=image_folder), experiment_folder=experiment_folder,
                     backends=backends, url=url, labels_file=labels_file, output_folder=output_folder,
                     batch_size=batch_size, number_warmup_batches=number_warmup_batches, iteration=iteration)
