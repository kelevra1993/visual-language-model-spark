"""
Object keypoint similarity and keypoint accuracy of the keypoint detection task, computed on the objects of the test
split once the keypoint trainer has brought the predicted keypoints of every crop back to the network input image: the
per-keypoint standard deviations of the similarity (COCO sigmas, from the evaluation section of the configuration),
the object keypoint similarity (OKS) of every predicted object with every ground truth object of the same class and
image, which replaces the intersection over union of the average precision of mean_average_precision.py, the
percentage of correct keypoints (PCK, a keypoint within a fraction of the longest box side) and the pixel errors of
every class and keypoint, and their tables for the report and rows for the metrics files. The objects are measured by
their box areas in network input pixels.
"""

import torch

from typing import Any, Dict, List, Tuple

from vision_training.core.contracts import EvaluationTable
from vision_training.evaluation.detection.object_sizes import compute_box_areas

# Fraction of the longest side of the object box within which a predicted keypoint is correct (PCK@0.05)
CORRECT_KEYPOINT_FRACTION = 0.05

# Name of the percentage of correct keypoints in the tables, the scalars and the report
CORRECT_KEYPOINTS_NAME = f"PCK@{CORRECT_KEYPOINT_FRACTION:g}"

# Smallest box area dividing the squared distances, so that a degenerate box gives a similarity instead of a NaN
MINIMUM_OBJECT_AREA = torch.finfo(torch.float64).eps


def build_keypoint_sigma_table(keypoint_categories: Dict[int, Dict[str, Any]], default_sigma: float,
                               keypoint_sigmas: Dict[str, Dict[str, float]], maximum_keypoints: int) -> torch.Tensor:
    """
    Builds the standard deviation of the object keypoint similarity of every keypoint position of every class, the
    tolerance of the evaluation of the keypoint trainer: the COCO sigma of the keypoint, a fraction of the square root
    of the object area (its per-keypoint constant being twice the sigma), given by category and keypoint name in
    'evaluation.keypoint_sigmas', the default for the keypoints left out. A class merging several categories takes the
    value given under any of their names.

    Args:
        keypoint_categories (Dict[int, Dict[str, Any]]): The 'name' (merged categories joined by '+') and 'keypoints'
            names of every foreground class, as injected in 'Data.keypoint_categories'.
        default_sigma (float): The sigma of the keypoints without their own value.
        keypoint_sigmas (Dict[str, Dict[str, float]]): The sigma of some keypoints, by category name then keypoint
            name.
        maximum_keypoints (int): The most keypoints of a class, the number of keypoint positions of every object.

    Returns:
        torch.Tensor: The float64 [C, K] sigmas, C being the largest class index plus one, the rows of the classes
            without keypoints and the padding positions holding the default.

    Raises:
        ValueError: If the categories merged into one class give different sigmas to one keypoint.
    """
    sigma_table = torch.full(size=(max(keypoint_categories) + 1, maximum_keypoints), fill_value=float(default_sigma),
                             dtype=torch.float64)
    for class_index, definition in keypoint_categories.items():
        for keypoint_position, keypoint_name in enumerate(definition["keypoints"]):
            # Values given to the keypoint under the names of the categories of the class
            given_sigmas = {float(keypoint_sigmas[category_name][keypoint_name])
                            for category_name in definition["name"].split("+")
                            if category_name in keypoint_sigmas and keypoint_name in keypoint_sigmas[category_name]}
            if len(given_sigmas) > 1:
                raise ValueError(f"The categories merged into the class '{definition['name']}' give different sigmas "
                                 f"{sorted(given_sigmas)} to the keypoint '{keypoint_name}' in "
                                 f"'evaluation.keypoint_sigmas'.")
            if given_sigmas:
                sigma_table[class_index, keypoint_position] = given_sigmas.pop()
    return sigma_table


def compute_object_keypoint_similarity(squared_distances: torch.Tensor, visibilities: torch.Tensor,
                                       areas: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """
    Computes the COCO object keypoint similarity of predicted keypoints with ground truth keypoints, the overlap of the
    average precision of the keypoint detection: the mean over the labelled ground truth keypoints of
    exp(-d² / (2 · area · (2σ)²)), d being the distance of the predicted keypoint to the ground truth one. A ground
    truth without labelled keypoint has a similarity of 0. The inputs broadcast together, so that the same function
    gives the similarity of every object with its own ground truth and the matrix of every prediction with every ground
    truth of an image.

    Args:
        squared_distances (torch.Tensor): The [..., K] squared distances of the predicted keypoints to the ground truth
            keypoints, in squared pixels.
        visibilities (torch.Tensor): The [..., K] COCO visibilities of the ground truth keypoints, 0 for the unlabelled
            ones and the padding positions.
        areas (torch.Tensor): The [...] areas of the ground truth objects, in squared pixels.
        sigmas (torch.Tensor): The [..., K] sigmas of the ground truth keypoints.

    Returns:
        torch.Tensor: The [...] float64 object keypoint similarities, within [0, 1].
    """
    keypoint_errors = squared_distances.to(dtype=torch.float64) / (
        2.0 * (2.0 * sigmas.to(dtype=torch.float64)) ** 2 *
        areas.to(dtype=torch.float64).clamp(min=MINIMUM_OBJECT_AREA).unsqueeze(dim=-1))
    labelled_keypoints = (visibilities > 0).to(dtype=torch.float64)
    labelled_counts = labelled_keypoints.sum(dim=-1).clamp(min=1.0)
    return (torch.exp(-keypoint_errors) * labelled_keypoints).sum(dim=-1) / labelled_counts


def group_image_objects(objects: Dict[str, torch.Tensor], sigma_table: torch.Tensor) -> Tuple[
        List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
    """
    Groups the evaluated objects of the test split by image into the detections and ground truths of the average
    precision (evaluate_detections with the 'keypoints' overlap), the keypoint dataset spreading the objects of one
    image over several batches. Every ground truth object gives one detection, the keypoints predicted on its crop, with
    its box and class and the score of its keypoints; the overlap of a detection with a ground truth of the same class
    is their object keypoint similarity, 0 across classes since their keypoints differ.

    Args:
        objects (Dict[str, torch.Tensor]): The 'image_identifiers' [N], 'labels' [N], 'bounding_boxes' [N, 4],
            ground truth 'keypoints' [N, K, 2] and 'visibilities' [N, K], 'predicted_keypoints' [N, K, 2] and 'scores'
            [N] of every evaluated object, in network input pixels.
        sigma_table (torch.Tensor): The [C, K] sigmas of every class (build_keypoint_sigma_table).

    Returns:
        Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]: The detections of every image ('boxes',
            'scores', 'labels' and the [N_i, N_i] 'keypoint_overlaps') and its ground truths ('boxes' and 'labels').
    """
    detections, ground_truths = [], []
    image_identifiers, image_positions = torch.unique(input=objects["image_identifiers"], return_inverse=True)
    for image_position in range(len(image_identifiers)):
        object_indices = torch.nonzero(input=image_positions == image_position).squeeze(dim=1)
        labels = objects["labels"][object_indices]
        boxes = objects["bounding_boxes"][object_indices].to(dtype=torch.float64)

        # Similarity of the keypoints predicted on every crop with the keypoints of every object of the image
        predicted_keypoints = objects["predicted_keypoints"][object_indices].to(dtype=torch.float64)
        ground_truth_keypoints = objects["keypoints"][object_indices].to(dtype=torch.float64)
        squared_distances = ((predicted_keypoints.unsqueeze(dim=1) - ground_truth_keypoints.unsqueeze(dim=0)) ** 2).sum(
            dim=-1)
        keypoint_overlaps = compute_object_keypoint_similarity(
            squared_distances=squared_distances, visibilities=objects["visibilities"][object_indices].unsqueeze(dim=0),
            areas=compute_box_areas(boxes=boxes).unsqueeze(dim=0), sigmas=sigma_table[labels].unsqueeze(dim=0))

        # Objects of different classes have different keypoints, so they never match
        keypoint_overlaps = keypoint_overlaps * (labels.unsqueeze(dim=1) == labels.unsqueeze(dim=0))
        detections.append({"boxes": boxes, "scores": objects["scores"][object_indices].to(dtype=torch.float64),
                           "labels": labels, "keypoint_overlaps": keypoint_overlaps})
        ground_truths.append({"boxes": boxes, "labels": labels})
    return detections, ground_truths


def measure_keypoint_accuracy(objects: Dict[str, torch.Tensor], sigma_table: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Measures how close the keypoints predicted on the crop of every object are to its own ground truth keypoints,
    independently of the score ranking of the average precision: the pixel distance of every keypoint, whether it is
    within CORRECT_KEYPOINT_FRACTION of the longest box side (the PCK) and the object keypoint similarity of the
    object, the unlabelled keypoints being left out.

    Args:
        objects (Dict[str, torch.Tensor]): The evaluated objects (group_image_objects), in network input pixels.
        sigma_table (torch.Tensor): The [C, K] sigmas of every class (build_keypoint_sigma_table).

    Returns:
        Dict[str, torch.Tensor]: The [N, K] 'distances' in pixels, the [N, K] booleans 'labelled' and 'correct', and
            the [N] 'similarities' of every object.
    """
    boxes = objects["bounding_boxes"].to(dtype=torch.float64)
    differences = objects["predicted_keypoints"].to(dtype=torch.float64) - objects["keypoints"].to(dtype=torch.float64)
    squared_distances = (differences ** 2).sum(dim=-1)
    distances = squared_distances.sqrt()

    # Correct keypoints within a fraction of the longest side of the box of their object
    longest_sides = torch.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    labelled = objects["visibilities"] > 0
    correct = labelled & (distances <= CORRECT_KEYPOINT_FRACTION * longest_sides.unsqueeze(dim=1))
    similarities = compute_object_keypoint_similarity(squared_distances=squared_distances,
                                                      visibilities=objects["visibilities"],
                                                      areas=compute_box_areas(boxes=boxes),
                                                      sigmas=sigma_table[objects["labels"]])
    return {"distances": distances, "labelled": labelled, "correct": correct, "similarities": similarities}


def compute_mean_or_none(values: torch.Tensor) -> float | None:
    """
    Averages the values of a group of keypoints or objects for a cell of the keypoint tables, a group without values
    (a class or keypoint never labelled in the test split) having no mean.

    Args:
        values (torch.Tensor): The values of the group.

    Returns:
        float | None: Their mean, None without values.
    """
    return float(values.to(dtype=torch.float64).mean()) if values.numel() > 0 else None


def summarise_keypoint_accuracy(objects: Dict[str, torch.Tensor], sigma_table: torch.Tensor,
                                keypoint_categories: Dict[int, Dict[str, Any]],
                                class_names: Dict[int, str]) -> Dict[str, Any]:
    """
    Summarises the keypoint accuracy of the test split for the report of the keypoint trainer, next to the average
    precisions on the object keypoint similarity: the PCK, the mean object keypoint similarity and the mean pixel error
    of all the labelled keypoints as global metrics and TensorBoard scalars, the same values for every class (the
    per-class table and the rows of class_metrics.csv) and for every keypoint of every class (the per-keypoint table
    and the rows of keypoint_metrics.csv), telling which shapes and which keypoints the detector places badly.

    Args:
        objects (Dict[str, torch.Tensor]): The evaluated objects (group_image_objects), in network input pixels.
        sigma_table (torch.Tensor): The [C, K] sigmas of every class (build_keypoint_sigma_table).
        keypoint_categories (Dict[int, Dict[str, Any]]): The 'keypoints' names of every foreground class.
        class_names (Dict[int, str]): The name of every class index.

    Returns:
        Dict[str, Any]: The global 'metrics', the 'scalars', the 'class_rows' and 'keypoint_rows', and the 'tables'
            ('classes' and 'keypoints', both per-class tables).
    """
    accuracy = measure_keypoint_accuracy(objects=objects, sigma_table=sigma_table)
    labelled, correct, distances = accuracy["labelled"], accuracy["correct"], accuracy["distances"]
    # Objects without labelled keypoint have no similarity to average (their OKS of 0 means nothing)
    measured_objects = labelled.any(dim=1)
    metrics = {CORRECT_KEYPOINTS_NAME: compute_mean_or_none(values=correct[labelled]),
               "Mean OKS": compute_mean_or_none(values=accuracy["similarities"][measured_objects]),
               "Mean Keypoint Error (pixels)": compute_mean_or_none(values=distances[labelled])}

    # Accuracy of every class, in class index order, and of every keypoint of the class
    class_rows, keypoint_rows = [], []
    for class_index, definition in keypoint_categories.items():
        class_objects = objects["labels"] == class_index
        class_labelled = labelled[class_objects]
        class_rows.append({"class_index": class_index, "class": class_names[class_index],
                           "objects": int(class_objects.sum()), "labelled_keypoints": int(class_labelled.sum()),
                           "percentage_correct_keypoints": compute_mean_or_none(
                               values=correct[class_objects][class_labelled]),
                           "mean_object_keypoint_similarity": compute_mean_or_none(
                               values=accuracy["similarities"][class_objects & measured_objects]),
                           "mean_error_pixels": compute_mean_or_none(values=distances[class_objects][class_labelled])})
        for keypoint_position, keypoint_name in enumerate(definition["keypoints"]):
            keypoint_labelled = class_labelled[:, keypoint_position]
            keypoint_rows.append({
                "class_index": class_index, "class": class_names[class_index], "keypoint": keypoint_name,
                "labelled": int(keypoint_labelled.sum()),
                "percentage_correct_keypoints": compute_mean_or_none(
                    values=correct[class_objects][:, keypoint_position][keypoint_labelled]),
                "mean_error_pixels": compute_mean_or_none(
                    values=distances[class_objects][:, keypoint_position][keypoint_labelled])})

    # Tables of the classes and of the keypoints, the worst ones first
    class_table: EvaluationTable = {
        "title": "Keypoint Accuracy Per Class", "rows": [
            [row["class"], row["objects"], row["labelled_keypoints"], row["percentage_correct_keypoints"],
             row["mean_object_keypoint_similarity"], row["mean_error_pixels"]] for row in class_rows],
        "columns": ["Class", "Objects", "Labelled\nKeypoints", CORRECT_KEYPOINTS_NAME, "Mean\nOKS",
                    "Mean Error\n(pixels)"],
        "sort_column": CORRECT_KEYPOINTS_NAME, "per_class": True,
        "legend": f"Worst {CORRECT_KEYPOINTS_NAME} first. {CORRECT_KEYPOINTS_NAME} : labelled keypoints within "
                  f"{CORRECT_KEYPOINT_FRACTION:g} of the longest box side, errors in network input pixels. — : no "
                  f"labelled keypoint."}
    keypoint_table: EvaluationTable = {
        "title": "Keypoint Accuracy Per Keypoint", "rows": [
            [row["class"], row["keypoint"], row["labelled"], row["percentage_correct_keypoints"],
             row["mean_error_pixels"]] for row in keypoint_rows],
        "columns": ["Class", "Keypoint", "Labelled", CORRECT_KEYPOINTS_NAME, "Mean Error\n(pixels)"],
        "sort_column": CORRECT_KEYPOINTS_NAME, "per_class": True,
        "legend": f"Worst {CORRECT_KEYPOINTS_NAME} first, errors in network input pixels. — : keypoint never "
                  f"labelled."}

    # Global values as TensorBoard scalars, a split without labelled keypoint having none
    scalars = {f"Keypoint Accuracy/{metric_name}": value for metric_name, value in metrics.items() if value is not None}
    return {"metrics": metrics, "scalars": scalars, "class_rows": class_rows, "keypoint_rows": keypoint_rows,
            "tables": {"classes": class_table, "keypoints": keypoint_table}}
