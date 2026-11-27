"""
Deployment of the object detection and instance segmentation tasks (DET, INS): the score threshold of every class built
into their exports. The evaluation of a checkpoint writes the score from which every class reaches the target precision
into 'class_thresholds.json' (DetectionTrainer.save_error_analysis, the mask matching for the instance segmenter); the
export reads the file of the exported checkpoint and weights (find_class_thresholds_path), gives every class its
threshold, or 'Detector.inference.score_threshold' when it never reaches the target, sets them into the
'class_score_thresholds' buffer of the inference model, prints them and records them in
'detection_score_thresholds.json' next to every exported model. Called by deployment/export_model.py and by the local
inference of deployment/deployment_utilities.py, so that the local model keeps the detections of the served ones.
"""

import json

from pathlib import Path
from typing import Any, Dict, Optional

from vision_training.deployment.frame_selection_deployment import THRESHOLDS_FILE_NAME, find_class_thresholds_path
from vision_training.evaluation.checkpoint_evaluation import CHECKPOINT_EVALUATION_FOLDER_NAME
from vision_training.models.detection.inference_detector import InferenceDetector
from vision_training.trainers.base_trainer import AVERAGED_WEIGHTS_FOLDER_NAME, EVALUATION_FOLDER_NAME
from vision_training.utilities.os_utilities import format_path_link, print_color

# Record of the score thresholds built into an exported detector or instance segmenter, written next to the model of
# every format, for the clients and the reviews of the export: the detections are already kept by these thresholds
DEPLOYED_THRESHOLDS_FILE_NAME = "detection_score_thresholds.json"

# Source of the threshold of a class that has none in the evaluation of the checkpoint
CONFIGURED_THRESHOLD_SOURCE = "Detector.inference.score_threshold"


def build_class_score_thresholds(thresholds_path: Optional[Path], configured_score_threshold: float,
                                 number_classes: int) -> Dict[str, Any]:
    """
    Gives the score threshold of every foreground class of an exported detector or instance segmenter: the threshold
    of the class in the 'class_thresholds.json' of the evaluation of the checkpoint, the classes listed in class index
    order as save_error_analysis writes them, or the configured threshold for a class that never reached the target
    precision (null in the file) and for every class without the file (a checkpoint never evaluated).

    Args:
        thresholds_path (Optional[Path]): The 'class_thresholds.json' of the evaluation of the exported checkpoint and
            weights (find_class_thresholds_path), None when there is none.
        configured_score_threshold (float): 'Detector.inference.score_threshold' of the run.
        number_classes (int): The number of classes of the detector, background included.

    Returns:
        Dict[str, Any]: The record of the thresholds: the 'configured_score_threshold', the 'class_thresholds_file'
            (its path as text, None without file), its 'target_precision' (None without file), and the 'classes', a
            list of one {'class_index', 'class_name' (None without file), 'score_threshold', 'source'} per
            foreground class in class index order, 'source' being THRESHOLDS_FILE_NAME or CONFIGURED_THRESHOLD_SOURCE.

    Raises:
        ValueError: If the file does not hold one entry per foreground class.
    """
    # Without evaluation, every class keeps the configured threshold
    if thresholds_path is None:
        return {"configured_score_threshold": configured_score_threshold, "class_thresholds_file": None,
                "target_precision": None,
                "classes": [{"class_index": class_index, "class_name": None,
                             "score_threshold": configured_score_threshold, "source": CONFIGURED_THRESHOLD_SOURCE}
                            for class_index in range(1, number_classes)]}

    # Entries of the file in class index order, one per foreground class
    thresholds = json.loads(s=thresholds_path.read_text())
    class_entries = list(thresholds["classes"].items())
    if len(class_entries) != number_classes - 1:
        raise ValueError(f"{thresholds_path} holds {len(class_entries)} classes, the detector has "
                         f"{number_classes - 1} foreground classes: the file comes from another run.")

    # Threshold of the class reaching the target precision, or the configured one
    classes = []
    for class_index, (class_name, class_threshold) in enumerate(class_entries, start=1):
        if class_threshold is None:
            classes.append({"class_index": class_index, "class_name": class_name,
                            "score_threshold": configured_score_threshold, "source": CONFIGURED_THRESHOLD_SOURCE})
        else:
            classes.append({"class_index": class_index, "class_name": class_name,
                            "score_threshold": float(class_threshold["threshold"]), "source": THRESHOLDS_FILE_NAME})
    return {"configured_score_threshold": configured_score_threshold, "class_thresholds_file": str(thresholds_path),
            "target_precision": thresholds["target_precision"], "classes": classes}


def apply_class_score_thresholds(inference_model: InferenceDetector, experiment_folder: Path, checkpoint_path: Path,
                                 use_averaged_weights: bool) -> Dict[str, Any]:
    """
    Builds the score thresholds of the evaluation of the exported checkpoint and weights into the inference detector or
    instance segmenter (set_class_score_thresholds), before its reference outputs and its export, so that the exported
    files keep the detections class by class, and prints them: the number of classes taking the threshold of the
    evaluation, with the target precision and the file, or a warning naming the folders looked into when the checkpoint
    was never evaluated.

    Args:
        inference_model (InferenceDetector): The inference detector or instance segmenter of the export.
        experiment_folder (Path): The exported run folder.
        checkpoint_path (Path): The exported checkpoint, '<run>/Weights/Iteration_<n>/model_<n>.pt'.
        use_averaged_weights (bool): True when the moving average of the weights is exported, whose evaluation sits in
            the 'EMA' subfolder of the iteration folder.

    Returns:
        Dict[str, Any]: The record of the thresholds (build_class_score_thresholds).

    Raises:
        ValueError: If both evaluation folders hold thresholds for the exported checkpoint, or the file does not hold
            one entry per foreground class.
    """
    thresholds_path = find_class_thresholds_path(experiment_folder=experiment_folder, checkpoint_path=checkpoint_path,
                                                 use_averaged_weights=use_averaged_weights)
    record = build_class_score_thresholds(thresholds_path=thresholds_path,
                                          configured_score_threshold=inference_model.score_threshold,
                                          number_classes=inference_model.number_classes)
    inference_model.set_class_score_thresholds(
        class_score_thresholds=[class_record["score_threshold"] for class_record in record["classes"]])

    # Console line of the thresholds built into the export
    configured_text = f"{record['configured_score_threshold']:g} ({CONFIGURED_THRESHOLD_SOURCE})"
    if thresholds_path is None:
        weights_folder = f"/{AVERAGED_WEIGHTS_FOLDER_NAME}" if use_averaged_weights else ""
        print_color(color="yellow", output=f"The checkpoint was not evaluated, every class keeps the score threshold "
                                           f"{configured_text}, no {THRESHOLDS_FILE_NAME} in [{EVALUATION_FOLDER_NAME}"
                                           f" or {CHECKPOINT_EVALUATION_FOLDER_NAME}]/{checkpoint_path.parent.name}"
                                           f"{weights_folder} of {format_path_link(path=experiment_folder)}", indent=1)
        return record
    number_evaluated = sum(class_record["source"] == THRESHOLDS_FILE_NAME for class_record in record["classes"])
    others_text = "" if number_evaluated == len(record["classes"]) else f", the others at {configured_text}"
    print_color(color="blue", output=f"Class Score Thresholds : {number_evaluated} of {len(record['classes'])} classes "
                                     f"at their threshold of {100.0 * record['target_precision']:g} % precision"
                                     f"{others_text} : {format_path_link(path=thresholds_path)}", indent=1)
    return record


def write_deployed_thresholds_file(model_folder: Path, record: Dict[str, Any]) -> Path:
    """
    Writes the record of the score thresholds built into an exported detector or instance segmenter as
    DEPLOYED_THRESHOLDS_FILE_NAME in the version folder of the model, next to its 'preprocessing.json', so that a
    client or a review of the export reads which threshold keeps the detections of every class and where it comes
    from. The detections the model returns are already kept by these thresholds.

    Args:
        model_folder (Path): The version folder of the exported model, '<model name>/<version>/'.
        record (Dict[str, Any]): The record of the thresholds (apply_class_score_thresholds).

    Returns:
        Path: The written file.
    """
    thresholds_path = model_folder / DEPLOYED_THRESHOLDS_FILE_NAME
    thresholds_path.write_text(data=json.dumps(obj=record, indent=2) + "\n")
    return thresholds_path

