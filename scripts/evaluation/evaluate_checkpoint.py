"""
When run, evaluates the last, a given or every checkpoint of a trained experiment on its test split (or another
labelled split) through vision_training.evaluation.checkpoint_evaluation.evaluate_checkpoints, the trainer of the task
being built from the copied configuration, the results written inside the run folder, in
'<run>/Checkpoint_Evaluation/' next to the 'Evaluation' folder of the training (another split in a folder of its own,
e.g. '<run>/Checkpoint_Evaluation_split_synthetic_shapes_validation/', so that the evaluations of two splits never
mix), and, when the tracking is on, tracked in an MLflow run of their own. A run without checkpoint stops the script
before anything is written. The precisions targeted by the class thresholds can be replaced, e.g. by
'--target_precisions 0.99 0.95 0.90' to read the thresholds of a research report without training again; with the
per-class report, a classifier gets its 'temperature_explorer.html' and 'class_logits.npz', served by
scripts/evaluation/serve_temperature_explorer.py to choose, save and export the temperature of the checkpoint, and the
predictions of every evaluated sample can be saved, read by the groups chart of
scripts/evaluation/draw_report_figures.py, which draws the figures of a research report from this folder. The
final non-maximum suppression of a detection run can be replaced, e.g. by DIoU-NMS for stage 5b of the detection
benchmark, its evaluations going to a folder of their own, '<run>/Checkpoint_Evaluation_nms_<name>/'. Run as a script
with the run folder and the evaluation settings given as arguments (parse_evaluation_arguments), its help printed when
none is given, the evaluation itself being evaluate_run.
"""

import argparse
import json
import sys

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from vision_training.core.task_registry import get_trainer_class
from vision_training.configuration_processing.classification_checks import check_target_precisions
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.configuration_processing.detection_checks import check_nms_criterion_configuration
from vision_training.data.data_record import DATA_RECORD_FILE_NAME
from vision_training.data.manifest_cache import read_manifests_once
from vision_training.evaluation.checkpoint_evaluation import CHECKPOINT_EVALUATION_FOLDER_NAME, evaluate_checkpoints
from vision_training.models.detection.postprocessing import NMS_CRITERIA
from vision_training.models.model_transfer import disable_pretrained_settings
from vision_training.trainers.checkpointing import (get_checkpoint_file, get_checkpoint_iterations,
                                                    get_last_checkpoint_iteration)
from vision_training.utilities.os_utilities import format_path_link, print_color

# Tasks whose evaluation gives the thresholds of every class at several target precisions (the classification trainer
# and the sequence trainers sharing its evaluation), the frame selection and the detection tasks keeping one
SEVERAL_TARGET_PRECISIONS_TASKS = ["classification", "sequence_classification", "sequence_labelling",
                                   "sequence_comparison"]

# Evaluation settings taken when the command line of the script (parse_evaluation_arguments) does not give them: a
# batch larger than the training one, since no gradient is kept, and every error image of a test split saved
DEFAULT_EVALUATION_SETTINGS = {"batch_size": 128, "maximum_error_images_per_class_pair": 100000}


def apply_error_image_limit(experiment_configuration: Dict[str, Any],
                            maximum_error_images_per_class_pair: Optional[int]) -> None:
    """
    Turns the error images of the evaluated checkpoints on with their own limit per pair of true and predicted classes,
    for the tasks saving their errors as images (an 'evaluation.error_dumps' section: classification, detection),
    e.g. to look at every error of a checkpoint. The tasks without error images (semantic segmentation, keypoint
    detection, self-supervised learning) are evaluated without them, a warning telling that the limit was skipped.

    Args:
        experiment_configuration (Dict[str, Any]): The loaded experiment configuration of the evaluated run, changed in
            place.
        maximum_error_images_per_class_pair (Optional[int]): The largest number of error images of every pair of
            classes, None to keep the experiment's 'evaluation.error_dumps' settings.

    Raises:
        KeyError: If the configuration has no 'evaluation' section, which a loaded configuration always has.
    """
    if maximum_error_images_per_class_pair is None:
        return

    # The tasks without an error_dumps section save no error image, so the limit has nothing to apply to; every task
    # has an 'evaluation' section, which the configuration checks require
    evaluation_configuration = experiment_configuration["evaluation"]
    if "error_dumps" not in evaluation_configuration:
        print_color(color="yellow", output=f"maximum_error_images_per_class_pair is skipped: the "
                                           f"{experiment_configuration['task']} task saves no error image (no "
                                           f"'evaluation.error_dumps' section).")
        return
    error_dumps_configuration = evaluation_configuration["error_dumps"]
    error_dumps_configuration["activated"] = True
    error_dumps_configuration["maximum_per_class_pair"] = maximum_error_images_per_class_pair


def apply_save_predictions(experiment_configuration: Dict[str, Any], save_predictions: Optional[bool]) -> None:
    """
    Turns on or off the file of the predictions of every evaluated sample, written in every evaluated iteration folder
    ('predictions.csv' for CLS, M2O, M2M and CMP, 'selections.csv' for SEL), e.g. on for a kept classifier, whose
    predictions.csv gives the chart of the groups seen and never seen in training of
    scripts/evaluation/draw_report_figures.py. The tasks without the setting (detection, segmentation, keypoint
    detection, self-supervised learning) are evaluated without it, a warning telling that it was skipped.

    Args:
        experiment_configuration (Dict[str, Any]): The loaded experiment configuration of the evaluated run, changed in
            place.
        save_predictions (Optional[bool]): Whether the predictions are saved, None to keep the experiment's
            'evaluation.save_predictions'.

    Raises:
        KeyError: If the configuration has no 'evaluation' section, which a loaded configuration always has.
    """
    if save_predictions is None:
        return

    # The tasks without a predictions file have no 'evaluation.save_predictions' setting to replace
    evaluation_configuration = experiment_configuration["evaluation"]
    if "save_predictions" not in evaluation_configuration:
        print_color(color="yellow", output=f"save_predictions is skipped: the {experiment_configuration['task']} task "
                                           f"writes no predictions file (no 'evaluation.save_predictions' setting).")
        return
    evaluation_configuration["save_predictions"] = save_predictions


def apply_target_precisions(experiment_configuration: Dict[str, Any],
                            target_precisions: Optional[Union[float, List[float]]]) -> None:
    """
    Replaces the precision targeted by the class thresholds of the evaluated checkpoints, e.g. by [0.99, 0.95, 0.90]
    to read the thresholds of a research report from class_thresholds.json, class_metrics.csv and
    precision_recall_thresholds.png without training again. A list is accepted by the tasks of
    SEVERAL_TARGET_PRECISIONS_TASKS, a single precision also by the frame selection and the detection tasks, and the
    tasks without class thresholds (semantic segmentation, keypoint detection, self-supervised learning) are evaluated
    without it, a warning telling that it was skipped.

    Args:
        experiment_configuration (Dict[str, Any]): The loaded experiment configuration of the evaluated run, changed in
            place.
        target_precisions (Optional[Union[float, List[float]]]): The target precision, or the list of distinct target
            precisions whose first one is the main target, None to keep the experiment's 'evaluation.target_precision'.

    Raises:
        ValueError: If a target precision is not within [0, 1], or if a list is given to a task keeping a single target
            precision.
        KeyError: If the configuration has no 'evaluation' section, which a loaded configuration always has.
    """
    if target_precisions is None:
        return

    # The tasks without class thresholds have no 'evaluation.target_precision' setting to replace
    evaluation_configuration = experiment_configuration["evaluation"]
    if "target_precision" not in evaluation_configuration:
        print_color(color="yellow", output=f"target_precisions is skipped: the {experiment_configuration['task']} task "
                                           f"gives no class threshold (no 'evaluation.target_precision' setting).")
        return

    # Checked like the configuration file, which was checked before this replacement, a list being refused by the tasks
    # keeping one target precision
    errors = check_target_precisions(
        value=target_precisions, key_path="target_precisions",
        allow_several_target_precisions=experiment_configuration["task"] in SEVERAL_TARGET_PRECISIONS_TASKS)
    if errors:
        raise ValueError(" ".join(errors))
    evaluation_configuration["target_precision"] = target_precisions


def name_final_nms(detector_configuration: Dict[str, Any]) -> str:
    """
    Names the final non-maximum suppression of a 'Detector' section, the criterion followed by the exponent of its
    distance term for DIoU-NMS (e.g. 'distance_iou_0.6'), the exponent changing nothing with the greedy NMS ('iou'),
    so that apply_nms_criterion tells whether an evaluated NMS differs from the one of the run.

    Args:
        detector_configuration (Dict[str, Any]): The 'Detector' section, with its 'nms_criterion' and
            'nms_distance_exponent'.

    Returns:
        str: The name of the final NMS.
    """
    if detector_configuration["nms_criterion"] == "iou":
        return "iou"
    return f"{detector_configuration['nms_criterion']}_{float(detector_configuration['nms_distance_exponent'])}"


def apply_nms_criterion(model_configuration: Dict[str, Any], task: str, nms_criterion: Optional[str],
                        nms_distance_exponent: Optional[float]) -> str:
    """
    Replaces the criterion and the distance exponent of the final non-maximum suppression of the evaluated checkpoints
    of a detection or instance segmentation run, e.g. "distance_iou" with 1.0, then 0.6, to compare DIoU-NMS with the
    greedy NMS of the run at stage 5b of the detection benchmark without
    training again, the final NMS changing no trained weight. The tasks without a 'Detector' section are evaluated
    without it, a warning telling that it was skipped.

    Args:
        model_configuration (Dict[str, Any]): The loaded model configuration of the evaluated run, changed in place.
        task (str): The task of the run, named in the warning.
        nms_criterion (Optional[str]): One of NMS_CRITERIA of models/detection/postprocessing.py, None to keep the
            run's 'Detector.nms_criterion'.
        nms_distance_exponent (Optional[float]): The exponent of the distance term of "distance_iou", None to keep the
            run's 'Detector.nms_distance_exponent'.

    Returns:
        str: The suffix of the output subfolder of the evaluation, '_nms_<name>' (name_final_nms) when the evaluated
            NMS differs from the one of the run, so that the evaluations of two NMS of a run never share a folder, and
            an empty suffix otherwise.

    Raises:
        ValueError: If the criterion is not one of NMS_CRITERIA, or the exponent not a finite positive number.
    """
    if nms_criterion is None and nms_distance_exponent is None:
        return ""

    # The tasks without detector have no final non-maximum suppression to replace
    if "Detector" not in model_configuration:
        print_color(color="yellow", output=f"nms_criterion and nms_distance_exponent are skipped: the {task} task has "
                                           f"no final non-maximum suppression (no 'Detector' section).")
        return ""

    # Checked like the configuration file, which was checked before this replacement, the run's section being changed
    # only once the evaluated settings are valid
    evaluated_detector_configuration = dict(model_configuration["Detector"])
    if nms_criterion is not None:
        evaluated_detector_configuration["nms_criterion"] = nms_criterion
    if nms_distance_exponent is not None:
        evaluated_detector_configuration["nms_distance_exponent"] = nms_distance_exponent
    errors = check_nms_criterion_configuration(model_configuration={"Detector": evaluated_detector_configuration})
    if errors:
        raise ValueError(" ".join(errors))

    # Subfolder suffix naming the evaluated NMS when it is not the one of the run, then the replaced settings
    run_nms_name = name_final_nms(detector_configuration=model_configuration["Detector"])
    evaluated_nms_name = name_final_nms(detector_configuration=evaluated_detector_configuration)
    model_configuration["Detector"].update(evaluated_detector_configuration)
    return "" if evaluated_nms_name == run_nms_name else f"_nms_{evaluated_nms_name}"


def apply_test_data_folder(experiment_configuration: Dict[str, Any],
                           test_data_folder: Optional[Union[str, Path]]) -> str:
    """
    Replaces the test split of the evaluated run by another labelled split, e.g. the validation split of its dataset or
    the test split of a newer dataset, and names the output subfolder of its evaluations after it, so that the
    evaluations of two splits of a run never share a folder: their class thresholds, class metrics, data.json and
    summary rows would otherwise replace each other, their error images and TensorBoard values mix, and the export, the
    temperature explorer and compare_best_evaluations.py read whichever split ran last. The test split of the run,
    given again, keeps the 'Checkpoint_Evaluation' folder of the run.

    Args:
        experiment_configuration (Dict[str, Any]): The loaded experiment configuration of the evaluated run, changed in
            place.
        test_data_folder (Optional[Union[str, Path]]): The evaluated labelled split, holding its 'labels.json', None to
            keep the test split of the run.

    Returns:
        str: The suffix of the output subfolder of the evaluation, '_split_<parent folder>_<split folder>' of the
            resolved split (e.g. '_split_synthetic_shapes_validation') when it is not the test split of the run, and
            an empty suffix otherwise.
    """
    if test_data_folder is None:
        return ""

    # The test split of the run given again keeps its folder, both folders being compared once resolved
    run_test_data_folder = Path(experiment_configuration["test_data_folder"]).resolve()
    evaluated_test_data_folder = Path(test_data_folder).resolve()
    experiment_configuration["test_data_folder"] = Path(test_data_folder)
    if evaluated_test_data_folder == run_test_data_folder:
        return ""

    # Suffix naming the split by its folder and the folder holding it, e.g. the dataset of a 'validation' split
    split_name_parts = [part for part in [evaluated_test_data_folder.parent.name, evaluated_test_data_folder.name]
                        if part]
    return "_split_" + "_".join(split_name_parts)


def check_evaluation_folder_split(evaluation_folder: Path, test_data_folder: Union[str, Path]) -> None:
    """
    Refuses to write the evaluations of a split into a folder holding those of another split, before the trainer
    replaces the data.json of the folder: two splits whose folders give the same subfolder name
    (apply_test_data_folder), or a 'Checkpoint_Evaluation' folder written by the evaluation of another split before
    every split had a folder of its own. The test split recorded in the data.json of the folder must be the evaluated
    one; a new folder, or one written before the record of the data existed, holds no record and is accepted.

    Args:
        evaluation_folder (Path): The output folder of the evaluation, '<run>/Checkpoint_Evaluation<suffix>/'.
        test_data_folder (Union[str, Path]): The evaluated split.

    Raises:
        ValueError: If the folder holds the evaluations of another split.
    """
    data_record_path = evaluation_folder / DATA_RECORD_FILE_NAME
    if not data_record_path.is_file():
        return

    # Data folder of the test split of the record, resolved like the evaluated one (build_split_record)
    data_record = json.loads(data_record_path.read_text(encoding="utf-8"))
    recorded_test_data_folder = Path(data_record["splits"]["test"]["data_folder"]).resolve()
    evaluated_test_data_folder = Path(test_data_folder).resolve()
    if recorded_test_data_folder != evaluated_test_data_folder:
        raise ValueError(f"{evaluation_folder} holds the evaluations of the split {recorded_test_data_folder} (test "
                         f"split of its {DATA_RECORD_FILE_NAME}), not of the evaluated split "
                         f"{evaluated_test_data_folder}: move or rename that folder, then launch again, so that the "
                         f"evaluations of two splits never mix.")


def list_evaluated_iterations(weights_directory: Path, iteration: Optional[Union[int, str]]) -> List[int]:
    """
    Lists the iterations of the evaluated checkpoints before the trainer is built, so that a run without checkpoint
    stops the script before anything is written, as it stops the export tools, instead of an evaluation of nothing
    ending normally: every checkpoint of the registry from the last saved one ("all"), so that the most trained
    weights are evaluated first, the last saved one (None), or the given one. A checkpoint listed by the registry but
    deleted from the disk is skipped by the evaluation (evaluate_checkpoints), as long as another one is evaluated.

    Args:
        weights_directory (Path): The 'Weights' folder of the evaluated run.
        iteration (Optional[Union[int, str]]): The iteration to evaluate, None for the last saved one, or "all" for
            every checkpoint of the registry, the intermediate ones included.

    Returns:
        List[int]: The iterations to evaluate, in the evaluation order.

    Raises:
        FileNotFoundError: If none of them has its weight file on the disk: a run stopped before its first save (no
            registry nor complete checkpoint folder, the last iteration being then 0), or a given iteration without
            checkpoint.
    """
    if iteration == "all":
        evaluated_iterations = get_checkpoint_iterations(weights_directory=weights_directory)[::-1]
    elif iteration is None:
        last_iteration = get_last_checkpoint_iteration(weights_directory=weights_directory)
        evaluated_iterations = [last_iteration] if last_iteration else []
    else:
        evaluated_iterations = [iteration]

    # At least one weight file on the disk, the saved checkpoints being listed to choose another iteration
    if not any(get_checkpoint_file(weights_directory=weights_directory, iteration=evaluated_iteration).is_file()
               for evaluated_iteration in evaluated_iterations):
        saved_iterations = get_checkpoint_iterations(weights_directory=weights_directory)
        raise FileNotFoundError(f"No checkpoint to evaluate for iteration = {iteration!r} (checkpoints of the "
                                f"registry: {saved_iterations}, iterations evaluated: {evaluated_iterations}) in : "
                                f"{format_path_link(path=weights_directory)}")
    return evaluated_iterations


def point_to_checkpoint_evaluation(experiment_configuration: Dict[str, Any], experiment_folder: Path,
                                   folder_suffix: str) -> Path:
    """
    Points the configuration copied from an evaluated run to the folder of its standalone evaluation,
    '<run>/Checkpoint_Evaluation<suffix>/', inside the run folder next to the 'Evaluation' folder of its training, so
    that every evaluation of a run stays with it: the trainer writes its Results.txt, Tensorboard/Test, the copied
    configuration and the iteration folders there, directly (standalone_evaluation_folder), saves no weights and
    writes nothing else in the run folder. The run name stays the one of the evaluated run, for the launch panel and the
    tags of the MLflow run of the evaluation, and 'tracking_root' keeps the experiments root folder of the run, so that
    this MLflow run goes to the backend of the evaluated experiment.

    Args:
        experiment_configuration (Dict[str, Any]): The loaded experiment configuration of the evaluated run, changed in
            place.
        experiment_folder (Path): The evaluated run folder.
        folder_suffix (str): The suffix of the evaluation folder, '_split_<name>' for another split
            (apply_test_data_folder) followed by '_nms_<name>' for a detection run evaluated with another final NMS
            (apply_nms_criterion), empty otherwise.

    Returns:
        Path: The folder of the evaluation.
    """
    experiment_configuration["experiment_folder"] = experiment_folder.parent
    experiment_configuration["run_name"] = experiment_folder.name
    experiment_configuration["standalone_evaluation_folder"] = CHECKPOINT_EVALUATION_FOLDER_NAME + folder_suffix
    return experiment_folder / experiment_configuration["standalone_evaluation_folder"]


def evaluate_run(experiment_folder: Path, iteration: Optional[Union[int, str]], test_data_folder: Optional[Path],
                 batch_size: Optional[int], track_evaluation: Optional[bool], save_predictions: Optional[bool],
                 maximum_error_images_per_class_pair: Optional[int],
                 target_precisions: Optional[Union[float, List[float]]], nms_criterion: Optional[str],
                 nms_distance_exponent: Optional[float]) -> None:
    """
    Evaluates checkpoints of a trained run again with the settings given on the command line of the script, to choose
    the checkpoint to export, or to score a run on another labelled split or with another final NMS, without training:
    the iterations are listed before anything is written, the configuration copied into the run is loaded and changed
    by the given settings (None keeping those of the run), and the trainer of the task, pointed to
    '<run>/Checkpoint_Evaluation<suffix>/', scores every listed checkpoint there through evaluate_checkpoints, the files
    of the run being only read.

    Args:
        experiment_folder (Path): The evaluated run folder, holding 'Weights' and 'training_configuration.yaml'.
        iteration (Optional[Union[int, str]]): The iteration to evaluate, None for the last saved one, or "all" for
            every checkpoint of the registry from the last saved one.
        test_data_folder (Optional[Path]): The labelled split evaluated instead of the test split of the run, None to
            keep the test split of the run.
        batch_size (Optional[int]): The evaluation batch size, None to keep the one of the run.
        track_evaluation (Optional[bool]): Whether the evaluation is tracked in an MLflow run of its own, None to keep
            the 'tracking.activated' of the run.
        save_predictions (Optional[bool]): Whether the predictions of every evaluated sample are saved, None to keep
            the 'evaluation.save_predictions' of the run.
        maximum_error_images_per_class_pair (Optional[int]): The largest number of error images of every pair of
            classes, the error images being turned on, None to keep the 'evaluation.error_dumps' of the run.
        target_precisions (Optional[Union[float, List[float]]]): The precision targeted by the class thresholds, or the
            list of distinct precisions whose first one is the main target, None to keep the one of the run.
        nms_criterion (Optional[str]): The criterion of the final NMS of a detection run, one of NMS_CRITERIA, None to
            keep the one of the run.
        nms_distance_exponent (Optional[float]): The exponent of the distance term of "distance_iou", None to keep the
            one of the run.

    Raises:
        FileNotFoundError: If none of the listed checkpoints has its weight file, before anything is written.
        ValueError: If a target precision or the final NMS is refused, or if the evaluation folder holds the
            evaluations of another split.
    """
    # Every checkpoint of the registry from the last saved one, so that the most trained weights are evaluated first,
    # the last one, or the given one, listed before anything is written so that a run without checkpoint stops here
    evaluated_iterations = list_evaluated_iterations(weights_directory=experiment_folder / "Weights",
                                                     iteration=iteration)

    # Copied configuration of the run pointed to the evaluated split and settings, the files of the run being only read
    configuration_path = experiment_folder / "training_configuration.yaml"
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    # Another labelled split replacing the test split of the run, its evaluations written to a folder of their own,
    # named after the split, when it is not the one of the run
    split_folder_suffix = apply_test_data_folder(experiment_configuration=experiment_configuration,
                                                 test_data_folder=test_data_folder)
    if batch_size is not None:
        experiment_configuration["batch_size"] = batch_size
    if track_evaluation is not None:
        experiment_configuration["tracking"]["activated"] = track_evaluation
    # Predictions file of every evaluated sample turned on or off, for the tasks writing one
    apply_save_predictions(experiment_configuration=experiment_configuration, save_predictions=save_predictions)
    # Error images turned on with their own limit per pair of classes, for the tasks saving their errors as images
    apply_error_image_limit(experiment_configuration=experiment_configuration,
                            maximum_error_images_per_class_pair=maximum_error_images_per_class_pair)
    # Target precisions of the class thresholds replaced, for the tasks giving class thresholds
    apply_target_precisions(experiment_configuration=experiment_configuration, target_precisions=target_precisions)
    # Final non-maximum suppression replaced for the detection tasks, its evaluations written to a folder of their
    # own, named after the NMS, when it is not the one of the run
    nms_folder_suffix = apply_nms_criterion(model_configuration=model_configuration,
                                            task=experiment_configuration["task"], nms_criterion=nms_criterion,
                                            nms_distance_exponent=nms_distance_exponent)
    # Results.txt, Tensorboard/Test and the iteration folders written in '<run>/Checkpoint_Evaluation/', next to the
    # 'Evaluation' folder of the training, where the export also looks for the thresholds saved from the temperature
    # explorer of a classifier, a folder holding the evaluations of another split being refused
    evaluation_folder = point_to_checkpoint_evaluation(experiment_configuration=experiment_configuration,
                                                       experiment_folder=experiment_folder,
                                                       folder_suffix=split_folder_suffix + nms_folder_suffix)
    check_evaluation_folder_split(evaluation_folder=evaluation_folder,
                                  test_data_folder=experiment_configuration["test_data_folder"])

    # The weights come from the evaluated checkpoints, so the pretrained components the run started from are not loaded
    disable_pretrained_settings(model_configuration=model_configuration)

    # The trainer of the task, every 'labels.json' parsed once for all its readers and freed before the evaluation
    trainer_class = get_trainer_class(task=experiment_configuration["task"])
    with read_manifests_once():
        trainer = trainer_class(experiment_configuration=experiment_configuration,
                                model_configuration=model_configuration, configuration_path=configuration_path)
    evaluate_checkpoints(trainer=trainer, experiment_folder=experiment_folder, iterations=evaluated_iterations)


def read_iteration_argument(value: str) -> Union[int, str]:
    """
    Converts the --iteration argument of the evaluation script into the iteration list_evaluated_iterations takes,
    'all' for every checkpoint of the registry, so that the command line reaches every choice of the evaluated
    checkpoints, the last saved one being the default of the argument.

    Args:
        value (str): The text given on the command line.

    Returns:
        Union[int, str]: "all", or the iteration as an integer.

    Raises:
        argparse.ArgumentTypeError: If the text is neither 'all' nor a whole number, argparse then printing the usage
            and the error.
    """
    # Only 'all' and whole numbers name checkpoints, any other word (e.g. 'last') being refused
    if value == "all":
        return value
    if not value.isdecimal():
        raise argparse.ArgumentTypeError(f"expected an iteration or all, got {value!r}")
    return int(value)


def read_optional_positive_integer_argument(value: str) -> Optional[int]:
    """
    Converts the --batch_size and --maximum_error_images_per_class_pair arguments of the evaluation script into a
    positive integer, 'None' keeping the setting of the evaluated run, so that the command line reaches the three
    states of these settings although their defaults are numbers (DEFAULT_EVALUATION_SETTINGS).

    Args:
        value (str): The text given on the command line.

    Returns:
        Optional[int]: None for 'None', otherwise the number.

    Raises:
        argparse.ArgumentTypeError: If the text is neither 'None' nor a whole number of at least 1, argparse then
            printing the usage and the error.
    """
    # A batch or a limit of zero images evaluates nothing, so only whole numbers from 1 are accepted
    if value == "None":
        return None
    if not value.isdecimal() or int(value) < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer or None, got {value!r}")
    return int(value)


def read_optional_boolean_argument(value: str) -> Optional[bool]:
    """
    Converts the --track_evaluation and --save_predictions arguments of the evaluation script, the words True or
    False, into a boolean, so that the command line reaches the three states of these settings without a pair of
    opposite flags, their default None keeping the setting of the evaluated run.

    Args:
        value (str): The text given on the command line.

    Returns:
        Optional[bool]: True for 'True', False for 'False'.

    Raises:
        argparse.ArgumentTypeError: If the text is neither 'True' nor 'False', argparse then printing the usage and
            the error.
    """
    if value not in ["True", "False"]:
        raise argparse.ArgumentTypeError(f"expected True or False, got {value!r}")
    return value == "True"


def parse_evaluation_arguments(arguments: List[str]) -> argparse.Namespace:
    """
    Reads the settings of an evaluation from the command line of the script, so that the checkpoints of a run are
    evaluated again without editing the file: the run folder is required, the batch size and the error image limit
    default to DEFAULT_EVALUATION_SETTINGS and every other setting to None, which keeps the one of the run. Launched
    without any argument, the script prints the help of every argument and stops. A '~' left by the shell at the start
    of a path ('--experiment_folder=~/Data/...') is expanded to the home folder, and a single target precision is
    given as a number, since the frame selection and the detection tasks refuse a list.

    Args:
        arguments (List[str]): The command line arguments, without the script name.

    Returns:
        argparse.Namespace: The arguments of evaluate_run.

    Raises:
        SystemExit: After printing the help when no argument is given, or the usage and the error when the run folder
            is missing or a value is not one the evaluation accepts.
    """
    # The description and the example keep their line breaks, argparse wrapping the help of every argument
    parser = argparse.ArgumentParser(
        prog="uv run python -m scripts.evaluation.evaluate_checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Evaluates checkpoints of a trained run again into <run>/Checkpoint_Evaluation/, the files of the\n"
                    "training being only read: on another labelled split, at other target precisions or another NMS.",
        epilog="Example, every checkpoint of a run with the class thresholds at three precisions:\n"
               "  uv run python -m scripts.evaluation.evaluate_checkpoint --experiment_folder ~/Data/<run> \\\n"
               "      --iteration all --target_precisions 0.99 0.95 0.90")
    # The run and the checkpoints to evaluate
    parser.add_argument("--experiment_folder", type=Path, required=True,
                        help="Run folder holding 'Weights' and 'training_configuration.yaml', only read (required).")
    parser.add_argument("--iteration", type=read_iteration_argument, default=None,
                        help="Iteration of the evaluated checkpoint, or all for every checkpoint from the last saved "
                             "one (default: the last saved one).")
    parser.add_argument("--test_data_folder", type=Path, default=None,
                        help="Labelled split evaluated instead, written to <run>/Checkpoint_Evaluation_split_<parent "
                             "folder>_<split folder>/ (default: the test split of the run).")
    # The settings of the run replaced for the evaluation
    parser.add_argument("--batch_size", type=read_optional_positive_integer_argument,
                        default=DEFAULT_EVALUATION_SETTINGS["batch_size"],
                        help="Evaluation batch size, larger than the training one since no gradient is kept, None for "
                             "the one of the run (default: %(default)s).")
    parser.add_argument("--track_evaluation", type=read_optional_boolean_argument, default=None,
                        metavar="{True,False}",
                        help="Track the evaluation in an MLflow run of its own, '<run name>-evaluation', tagged with "
                             "its training run (default: the tracking.activated of the run).")
    parser.add_argument("--save_predictions", type=read_optional_boolean_argument, default=None,
                        metavar="{True,False}",
                        help="Write predictions.csv (selections.csv for the frame selection) in every evaluated "
                             "iteration folder (default: the evaluation.save_predictions of the run).")
    # The error images and the class thresholds, skipped with a warning by the tasks without them
    parser.add_argument("--maximum_error_images_per_class_pair", type=read_optional_positive_integer_argument,
                        default=DEFAULT_EVALUATION_SETTINGS["maximum_error_images_per_class_pair"],
                        help="Error images turned on with at most this number per pair of true and predicted "
                             "classes, None for the evaluation.error_dumps of the run (default: %(default)s).")
    parser.add_argument("--target_precisions", type=float, nargs="+", default=None,
                        help="Precision targeted by the class thresholds, or several, the first the main one, a list "
                             "refused by the frame selection and the detection tasks (default: the "
                             "evaluation.target_precision of the run).")
    # The final non-maximum suppression of the detection tasks, skipped with a warning by the other tasks
    parser.add_argument("--nms_criterion", choices=NMS_CRITERIA, default=None,
                        help="Criterion of the final NMS, another one than the run's written to "
                             "<run>/Checkpoint_Evaluation_nms_<criterion>[_<exponent>]/ (default: the "
                             "Detector.nms_criterion of the run).")
    parser.add_argument("--nms_distance_exponent", type=float, default=None,
                        help="Exponent of the distance term of distance_iou, 1.0 in the paper, 0.6 in YOLOv4 "
                             "(default: the Detector.nms_distance_exponent of the run).")

    # Without any argument, the script describes its arguments instead of failing on the missing run folder
    if not arguments:
        parser.print_help()
        parser.exit(status=0)
    parsed_arguments = parser.parse_args(args=arguments)

    # A '~' the shell leaves after '=' points to the home folder, as in the other scripts of the repository
    for path_name in ["experiment_folder", "test_data_folder"]:
        if getattr(parsed_arguments, path_name) is not None:
            setattr(parsed_arguments, path_name, getattr(parsed_arguments, path_name).expanduser())

    # One target precision is given as a number, the tasks keeping a single target precision refusing a list
    if parsed_arguments.target_precisions is not None and len(parsed_arguments.target_precisions) == 1:
        parsed_arguments.target_precisions = parsed_arguments.target_precisions[0]
    return parsed_arguments


if __name__ == "__main__":
    # Run folder and evaluation settings read from the command line, the help printed when none is given
    evaluation_arguments = parse_evaluation_arguments(arguments=sys.argv[1:])
    evaluate_run(experiment_folder=evaluation_arguments.experiment_folder, iteration=evaluation_arguments.iteration,
                 test_data_folder=evaluation_arguments.test_data_folder, batch_size=evaluation_arguments.batch_size,
                 track_evaluation=evaluation_arguments.track_evaluation,
                 save_predictions=evaluation_arguments.save_predictions,
                 maximum_error_images_per_class_pair=evaluation_arguments.maximum_error_images_per_class_pair,
                 target_precisions=evaluation_arguments.target_precisions,
                 nms_criterion=evaluation_arguments.nms_criterion,
                 nms_distance_exponent=evaluation_arguments.nms_distance_exponent)
