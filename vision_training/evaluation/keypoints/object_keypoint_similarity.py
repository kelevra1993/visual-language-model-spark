"""
Object keypoint similarity and keypoint accuracy of the keypoint detection task, computed on the test images once the
keypoint trainer has grouped the keypoint candidates of every image into objects
(models/keypoints/keypoint_grouping.py): the per-keypoint standard deviations of the similarity (COCO sigmas, from the
evaluation section of the configuration), the object keypoint similarity (OKS) of every predicted object with every
ground truth object of the same class and image, which replaces the intersection over union of the average precision of
mean_average_precision.py, the similarity of every predicted object with the objects without labelled keypoint (the
COCO crowds and persons without keypoints, against their box extended by its size on every side as pycocotools), which
the average precision ignores, the matching of every ground truth object to one predicted object reaching an OKS of
0.5, and, on the matched keypoints, the percentage of correct keypoints (PCK, a keypoint within a fraction of the
longest box side) and the pixel errors of every class and keypoint, with their tables for the report and rows for the
metrics files.
The objects are measured by their box areas in network input pixels. The report writes the matched share, the PCK and
the mean OKS as percentages ("94.03 %"), the counts and pixel errors keeping their values, and the metrics files and
scalars keep them between 0 and 1.
"""

import torch

from scipy.optimize import linear_sum_assignment
from typing import Any, Dict, List

from vision_training.core.contracts import EvaluationTable
from vision_training.evaluation.detection.object_sizes import compute_box_areas

# Fraction of the longest side of the object box within which a predicted keypoint is correct (PCK@0.05)
CORRECT_KEYPOINT_FRACTION = 0.05

# Name of the percentage of correct keypoints in the tables, the scalars and the report
CORRECT_KEYPOINTS_NAME = f"PCK@{CORRECT_KEYPOINT_FRACTION:g}"

# Smallest box area dividing the squared distances, so that a degenerate box gives a similarity instead of a NaN
MINIMUM_OBJECT_AREA = torch.finfo(torch.float64).eps

# Smallest object keypoint similarity at which a predicted object matches a ground truth object for the keypoint
# accuracy, the overlap of the OKS AP@0.5, so that a prediction far from the object (whose similarity only reaches 0
# on an underflow, about 4 box sides away) leaves it missed rather than adding its distance to the pixel errors
MINIMUM_MATCH_SIMILARITY = 0.5

# Values of every ground truth object of the test split kept by evaluate_image_keypoints for the keypoint accuracy
EVALUATED_OBJECT_KEYS = ["labels", "bounding_boxes", "keypoints", "visibilities", "predicted_keypoints", "matched"]


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
            ones and the padding positions; for an object without labelled keypoint, 1 on the keypoints found by the
            predicted object, the keypoints averaged by evaluate_image_keypoints.
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


def compute_squared_distances(predicted_keypoints: torch.Tensor, ground_truth_keypoints: torch.Tensor) -> torch.Tensor:
    """
    Computes the squared pixel distances of predicted keypoints to ground truth keypoints for the object keypoint
    similarity and the keypoint accuracy, a keypoint the grouping did not find (NaN) being infinitely far, so that it
    adds 0 to the similarity and is never correct. The inputs broadcast together.

    Args:
        predicted_keypoints (torch.Tensor): The [..., K, 2] (x, y) predicted keypoints, NaN for the keypoints not found.
        ground_truth_keypoints (torch.Tensor): The [..., K, 2] (x, y) ground truth keypoints.

    Returns:
        torch.Tensor: The [..., K] float64 squared distances, infinite for the keypoints not found.
    """
    differences = predicted_keypoints.to(dtype=torch.float64) - ground_truth_keypoints.to(dtype=torch.float64)
    squared_distances = (differences ** 2).sum(dim=-1)
    return torch.nan_to_num(input=squared_distances, nan=float("inf"))


def compute_squared_extended_box_distances(predicted_keypoints: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes the squared pixel distances of predicted keypoints to the objects without labelled keypoint (the COCO
    crowds and persons without keypoints) for their object keypoint similarity, the rule of pycocotools for such an
    object: the distance to its box extended by its width on the left and on the right and by its height above and
    below, 0 for a keypoint inside, so that an object found on it reaches a similarity of 1 and is ignored by the
    average precision instead of being a false positive. A keypoint not found (NaN) is infinitely far. The inputs
    broadcast together.

    Args:
        predicted_keypoints (torch.Tensor): The [..., K, 2] (x, y) predicted keypoints, NaN for the keypoints not found.
        boxes (torch.Tensor): The [..., 4] (x1, y1, x2, y2) boxes of the objects without labelled keypoint.

    Returns:
        torch.Tensor: The [..., K] float64 squared distances, infinite for the keypoints not found.
    """
    keypoints = predicted_keypoints.to(dtype=torch.float64)
    boxes = boxes.to(dtype=torch.float64).unsqueeze(dim=-2)
    widths, heights = boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]

    # Distance of every keypoint outside the extended box along each axis, 0 inside it
    horizontal_distances = ((boxes[..., 0] - widths - keypoints[..., 0]).clamp(min=0.0) +
                            (keypoints[..., 0] - boxes[..., 2] - widths).clamp(min=0.0))
    vertical_distances = ((boxes[..., 1] - heights - keypoints[..., 1]).clamp(min=0.0) +
                          (keypoints[..., 1] - boxes[..., 3] - heights).clamp(min=0.0))
    return torch.nan_to_num(input=horizontal_distances ** 2 + vertical_distances ** 2, nan=float("inf"))


def compute_keypoint_boxes(keypoints: torch.Tensor) -> torch.Tensor:
    """
    Gives the box of the keypoints found for every predicted object, which sizes the detections of the keypoint average
    precision that match no ground truth (the matched ones being sized by their ground truth box).

    Args:
        keypoints (torch.Tensor): The [D, K, 2] (x, y) predicted keypoints, NaN for the keypoints not found.

    Returns:
        torch.Tensor: The [D, 4] float64 (x1, y1, x2, y2) boxes.

    Raises:
        ValueError: If a predicted object has no keypoint.
    """
    keypoints = keypoints.to(dtype=torch.float64)
    found = ~torch.isnan(keypoints).any(dim=-1)
    if not bool(found.any(dim=1).all()):
        raise ValueError("A predicted object has no keypoint, the grouping must give at least one keypoint per object.")
    smallest = torch.where(found[..., None], keypoints, torch.full_like(input=keypoints, fill_value=float("inf")))
    largest = torch.where(found[..., None], keypoints, torch.full_like(input=keypoints, fill_value=float("-inf")))
    return torch.cat(tensors=(smallest.amin(dim=1), largest.amax(dim=1)), dim=1)


def match_ground_truth_objects(keypoint_overlaps: torch.Tensor) -> torch.Tensor:
    """
    Matches every ground truth object of an image to at most one predicted object for the keypoint accuracy, whatever
    the scores: the one-to-one matching of the largest total object keypoint similarity over the pairs reaching
    MINIMUM_MATCH_SIMILARITY, a pair below it (other class, or keypoints too far) never matching nor taking part in the
    assignment.

    Args:
        keypoint_overlaps (torch.Tensor): The [D, G] object keypoint similarity of every predicted object with every
            ground truth object, 0 across classes.

    Returns:
        torch.Tensor: The [G] int64 index of the predicted object of every ground truth object, -1 when it is missed.
    """
    matched_predictions = torch.full(size=(keypoint_overlaps.shape[1],), fill_value=-1, dtype=torch.int64)
    if keypoint_overlaps.numel() == 0:
        return matched_predictions

    # Leave the pairs below the minimum similarity out of the assignment, then keep the assigned pairs reaching it
    eligible_overlaps = torch.where(keypoint_overlaps >= MINIMUM_MATCH_SIMILARITY, keypoint_overlaps,
                                    torch.zeros_like(input=keypoint_overlaps))
    prediction_indices, ground_truth_indices = linear_sum_assignment(cost_matrix=eligible_overlaps.numpy(),
                                                                     maximize=True)
    for prediction_index, ground_truth_index in zip(prediction_indices, ground_truth_indices, strict=True):
        if eligible_overlaps[prediction_index, ground_truth_index] > 0:
            matched_predictions[ground_truth_index] = int(prediction_index)
    return matched_predictions


def evaluate_image_keypoints(images: List[Dict[str, torch.Tensor]], sigma_table: torch.Tensor) -> Dict[str, Any]:
    """
    Compares the objects found by the keypoint detector in every test image with its ground truth objects, for the
    average precision (evaluate_detections with the 'keypoints' overlap) and the keypoint accuracy: the object keypoint
    similarity of every predicted object with every ground truth object of the same class, 0 across classes since their
    keypoints differ, measured with the ground truth box area; the similarity of every predicted object with every
    object without labelled keypoint of the same class (compute_squared_extended_box_distances, averaged over the
    keypoints found), which the average precision ignores as pycocotools does; then every ground truth object matched
    to one predicted object (match_ground_truth_objects), its keypoints compared with the matched ones, NaN for a missed
    object, the objects without labelled keypoint taking no part in the keypoint accuracy.

    Args:
        images (List[Dict[str, torch.Tensor]]): For every image, its ground truth 'labels' [G], 'bounding_boxes'
            [G, 4], 'keypoints' [G, K, 2] and 'visibilities' [G, K], its objects without labelled keypoint
            'ignored_bounding_boxes' [U, 4], 'ignored_labels' [U] and 'ignored_crowds' [U], and its 'predicted_labels'
            [D], 'predicted_keypoints' [D, K, 2] (NaN for the keypoints not found) and 'predicted_scores' [D], in
            network input pixels, with the 'image_area' of the network input, which sizes the objects.
        sigma_table (torch.Tensor): The [C, K] sigmas of every class (build_keypoint_sigma_table).

    Returns:
        Dict[str, Any]: The 'detections' of every image ('boxes' of their keypoints, 'scores', 'labels', the [D, G]
            'keypoint_overlaps' and the [D, U] 'ignored_overlaps'), its 'ground_truths' ('boxes', 'labels',
            'image_area', 'ignored_labels' and 'ignored_crowds'), and the ground truth 'objects' of all the images
            ('labels', 'bounding_boxes', 'keypoints', 'visibilities', the 'predicted_keypoints' of their matched
            object, NaN when missed, and the 'matched' booleans).
    """
    detections, ground_truths, objects = [], [], {object_key: [] for object_key in EVALUATED_OBJECT_KEYS}
    for image in images:
        labels, predicted_labels = image["labels"], image["predicted_labels"]
        boxes = image["bounding_boxes"].to(dtype=torch.float64)

        # Similarity of every predicted object with every ground truth object of the image, of the same class only
        squared_distances = compute_squared_distances(predicted_keypoints=image["predicted_keypoints"][:, None],
                                                      ground_truth_keypoints=image["keypoints"][None])
        keypoint_overlaps = compute_object_keypoint_similarity(
            squared_distances=squared_distances, visibilities=image["visibilities"][None],
            areas=compute_box_areas(boxes=boxes)[None], sigmas=sigma_table[labels][None])
        keypoint_overlaps = keypoint_overlaps * (predicted_labels[:, None] == labels[None, :])

        # Similarity of every predicted object with every object without labelled keypoint of the same class, its
        # keypoints found against the extended box of the object, averaged over those keypoints
        ignored_boxes, ignored_labels = image["ignored_bounding_boxes"].to(dtype=torch.float64), image["ignored_labels"]
        found_keypoints = (~torch.isnan(image["predicted_keypoints"]).any(dim=-1)).to(dtype=torch.int64)
        ignored_distances = compute_squared_extended_box_distances(
            predicted_keypoints=image["predicted_keypoints"][:, None], boxes=ignored_boxes[None])
        ignored_overlaps = compute_object_keypoint_similarity(
            squared_distances=ignored_distances, visibilities=found_keypoints[:, None],
            areas=compute_box_areas(boxes=ignored_boxes)[None], sigmas=sigma_table[ignored_labels][None])
        ignored_overlaps = ignored_overlaps * (predicted_labels[:, None] == ignored_labels[None, :])
        detections.append({"boxes": compute_keypoint_boxes(keypoints=image["predicted_keypoints"]),
                           "scores": image["predicted_scores"].to(dtype=torch.float64), "labels": predicted_labels,
                           "keypoint_overlaps": keypoint_overlaps, "ignored_overlaps": ignored_overlaps})
        ground_truths.append({"boxes": boxes, "labels": labels, "image_area": image["image_area"],
                              "ignored_labels": ignored_labels, "ignored_crowds": image["ignored_crowds"]})

        # Keypoints of the predicted object matched to every ground truth object, NaN for the missed ones
        matched_predictions = match_ground_truth_objects(keypoint_overlaps=keypoint_overlaps)
        matched = matched_predictions >= 0
        matched_keypoints = torch.full_like(input=image["keypoints"], fill_value=float("nan"), dtype=torch.float32)
        matched_keypoints[matched] = image["predicted_keypoints"][matched_predictions[matched]].to(dtype=torch.float32)
        image_objects = {"labels": labels, "bounding_boxes": boxes, "keypoints": image["keypoints"],
                         "visibilities": image["visibilities"], "predicted_keypoints": matched_keypoints,
                         "matched": matched}
        for object_key in EVALUATED_OBJECT_KEYS:
            objects[object_key].append(image_objects[object_key])
    return {"detections": detections, "ground_truths": ground_truths,
            "objects": {object_key: torch.cat(tensors=values, dim=0) for object_key, values in objects.items()}}


def measure_keypoint_accuracy(objects: Dict[str, torch.Tensor], sigma_table: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Measures how close the keypoints of the predicted object matched to every ground truth object are to its ground
    truth keypoints, independently of the score ranking of the average precision: the pixel distance of every keypoint,
    whether it is within CORRECT_KEYPOINT_FRACTION of the longest box side (the PCK) and the object keypoint similarity
    of the object, the unlabelled keypoints being left out. A keypoint not found, alone or with its missed object, is
    infinitely far: never correct, adding 0 to the similarity.

    Args:
        objects (Dict[str, torch.Tensor]): The ground truth objects (evaluate_image_keypoints), in network input
            pixels.
        sigma_table (torch.Tensor): The [C, K] sigmas of every class (build_keypoint_sigma_table).

    Returns:
        Dict[str, torch.Tensor]: The [N, K] 'distances' in pixels (infinite for the keypoints not found), the [N, K]
            booleans 'labelled', 'found' and 'correct', and the [N] 'similarities' of every object.
    """
    boxes = objects["bounding_boxes"].to(dtype=torch.float64)
    squared_distances = compute_squared_distances(predicted_keypoints=objects["predicted_keypoints"],
                                                  ground_truth_keypoints=objects["keypoints"])
    distances = squared_distances.sqrt()

    # Correct keypoints within a fraction of the longest side of the box of their object
    longest_sides = torch.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    labelled = objects["visibilities"] > 0
    correct = labelled & (distances <= CORRECT_KEYPOINT_FRACTION * longest_sides.unsqueeze(dim=1))
    similarities = compute_object_keypoint_similarity(
        squared_distances=squared_distances, visibilities=objects["visibilities"], areas=compute_box_areas(boxes=boxes),
        sigmas=sigma_table[objects["labels"]])
    return {"distances": distances, "labelled": labelled, "found": torch.isfinite(distances), "correct": correct,
            "similarities": similarities}


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
    precisions on the object keypoint similarity: the fraction of the ground truth objects matched to a predicted
    object, the PCK and the mean object keypoint similarity of all the labelled keypoints (a missed object or keypoint
    counting as wrong) and the mean pixel error of the keypoints found as global metrics and TensorBoard scalars, the
    same values for every class (the per-class table and the rows of class_metrics.csv) and for every keypoint of every
    class (the per-keypoint table and the rows of keypoint_metrics.csv), telling which shapes and which keypoints the
    detector misses or places badly. The matched share, the PCK and the mean OKS are rates, written as percentages in
    the metrics grid and the tables, the mean pixel error keeping its 4 decimals.

    Args:
        objects (Dict[str, torch.Tensor]): The ground truth objects (evaluate_image_keypoints), in network input
            pixels.
        sigma_table (torch.Tensor): The [C, K] sigmas of every class (build_keypoint_sigma_table).
        keypoint_categories (Dict[int, Dict[str, Any]]): The 'keypoints' names of every foreground class.
        class_names (Dict[int, str]): The name of every class index.

    Returns:
        Dict[str, Any]: The global 'metrics' with the names of those written as percentages ('percent_metrics'), the
            'scalars', the 'class_rows' and 'keypoint_rows', and the 'tables' ('classes' and 'keypoints', both
            per-class tables).
    """
    accuracy = measure_keypoint_accuracy(objects=objects, sigma_table=sigma_table)
    labelled, correct, distances = accuracy["labelled"], accuracy["correct"], accuracy["distances"]
    matched = objects["matched"]
    # Objects without labelled keypoint have no similarity to average (their OKS of 0 means nothing), and the pixel
    # errors are those of the labelled keypoints found
    measured_objects = labelled.any(dim=1)
    measured_errors = labelled & accuracy["found"]
    metrics = {"Matched Objects": compute_mean_or_none(values=matched),
               CORRECT_KEYPOINTS_NAME: compute_mean_or_none(values=correct[labelled]),
               "Mean OKS": compute_mean_or_none(values=accuracy["similarities"][measured_objects]),
               "Mean Keypoint Error (pixels)": compute_mean_or_none(values=distances[measured_errors])}
    # Rates of the grid written as percentages, the pixel error keeping its 4 decimals
    percent_metrics = ["Matched Objects", CORRECT_KEYPOINTS_NAME, "Mean OKS"]

    # Accuracy of every class, in class index order, and of every keypoint of the class
    class_rows, keypoint_rows = [], []
    for class_index, definition in keypoint_categories.items():
        class_objects = objects["labels"] == class_index
        class_labelled, class_errors = labelled[class_objects], measured_errors[class_objects]
        class_rows.append({"class_index": class_index, "class": class_names[class_index],
                           "objects": int(class_objects.sum()), "matched_objects": int(matched[class_objects].sum()),
                           "labelled_keypoints": int(class_labelled.sum()),
                           "percentage_correct_keypoints": compute_mean_or_none(
                               values=correct[class_objects][class_labelled]),
                           "mean_object_keypoint_similarity": compute_mean_or_none(
                               values=accuracy["similarities"][class_objects & measured_objects]),
                           "mean_error_pixels": compute_mean_or_none(values=distances[class_objects][class_errors])})
        # The same measures for every keypoint of the class, from its column of the objects of the class
        for keypoint_position, keypoint_name in enumerate(definition["keypoints"]):
            keypoint_labelled = class_labelled[:, keypoint_position]
            keypoint_rows.append({
                "class_index": class_index, "class": class_names[class_index], "keypoint": keypoint_name,
                "labelled": int(keypoint_labelled.sum()),
                "found": int(class_errors[:, keypoint_position].sum()),
                "percentage_correct_keypoints": compute_mean_or_none(
                    values=correct[class_objects][:, keypoint_position][keypoint_labelled]),
                "mean_error_pixels": compute_mean_or_none(
                    values=distances[class_objects][:, keypoint_position][class_errors[:, keypoint_position]])})

    # Tables of the classes and of the keypoints, the worst ones first, the PCK and mean OKS written as percentages and
    # the counts and pixel errors kept as they are
    class_table: EvaluationTable = {
        "title": "Keypoint Accuracy Per Class", "rows": [
            [row["class"], row["objects"], row["matched_objects"], row["labelled_keypoints"],
             row["percentage_correct_keypoints"], row["mean_object_keypoint_similarity"], row["mean_error_pixels"]]
            for row in class_rows],
        "columns": ["Class", "Objects", "Matched\nObjects", "Labelled\nKeypoints", CORRECT_KEYPOINTS_NAME,
                    "Mean\nOKS", "Mean Error\n(pixels)"],
        "sort_column": CORRECT_KEYPOINTS_NAME, "per_class": True,
        "percent_columns": [CORRECT_KEYPOINTS_NAME, "Mean\nOKS"],
        "legend": f"Worst {CORRECT_KEYPOINTS_NAME} first. {CORRECT_KEYPOINTS_NAME} : labelled keypoints within "
                  f"{CORRECT_KEYPOINT_FRACTION:g} of the longest box side, a missed object or keypoint counting as "
                  f"wrong; errors of the keypoints found, in network input pixels. — : no labelled keypoint."}

    # The same table for every keypoint of every class
    keypoint_table: EvaluationTable = {
        "title": "Keypoint Accuracy Per Keypoint", "rows": [
            [row["class"], row["keypoint"], row["labelled"], row["found"], row["percentage_correct_keypoints"],
             row["mean_error_pixels"]] for row in keypoint_rows],
        "columns": ["Class", "Keypoint", "Labelled", "Found", CORRECT_KEYPOINTS_NAME, "Mean Error\n(pixels)"],
        "sort_column": CORRECT_KEYPOINTS_NAME, "per_class": True, "percent_columns": [CORRECT_KEYPOINTS_NAME],
        "legend": f"Worst {CORRECT_KEYPOINTS_NAME} first, a missed keypoint counting as wrong; errors of the keypoints "
                  f"found, in network input pixels. — : keypoint never labelled or never found."}

    # Global values as TensorBoard scalars, a split without labelled keypoint having none
    scalars = {f"Keypoint Accuracy/{metric_name}": value for metric_name, value in metrics.items() if value is not None}
    return {"metrics": metrics, "percent_metrics": percent_metrics, "scalars": scalars, "class_rows": class_rows,
            "keypoint_rows": keypoint_rows, "tables": {"classes": class_table, "keypoints": keypoint_table}}
