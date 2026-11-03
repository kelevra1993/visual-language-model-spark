"""
When run, evaluates the last, a given or every checkpoint of a trained experiment on its test split (or another
labelled split) through vision_training.evaluation.checkpoint_evaluation.evaluate_checkpoints, the trainer of the task
being built from the copied configuration, the results written to a separate evaluations folder and, when the tracking
is on, tracked in an MLflow run of their own. The precisions targeted by the class thresholds can be replaced, e.g. by
[0.99, 0.95, 0.90] to read the thresholds of a research report without training again; with the per-class report, a
classifier gets its 'temperature_explorer.html', served by scripts/evaluation/serve_temperature_explorer.py to save the
thresholds of the temperature of its export.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from vision_training.core.task_registry import get_trainer_class
from vision_training.configuration_processing.classification_checks import check_target_precisions
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.evaluation.checkpoint_evaluation import EVALUATIONS_FOLDER_NAME, evaluate_checkpoints
from vision_training.models.model_transfer import disable_pretrained_settings
from vision_training.trainers.checkpointing import get_checkpoint_iterations, get_last_checkpoint_iteration
from vision_training.utilities.os_utilities import print_color

# Tasks whose evaluation gives the thresholds of every class at several target precisions (the classification trainer
# and the sequence trainers sharing its evaluation), the frame selection and the detection tasks keeping one
SEVERAL_TARGET_PRECISIONS_TASKS = ["classification", "sequence_classification", "sequence_labelling",
                                   "sequence_comparison"]


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
    """
    if maximum_error_images_per_class_pair is None:
        return

    # The tasks without an error_dumps section save no error image, so the limit has nothing to apply to
    error_dumps_configuration = experiment_configuration.get("evaluation", {}).get("error_dumps")
    if error_dumps_configuration is None:
        print_color(color="yellow", output=f"maximum_error_images_per_class_pair is skipped: the "
                                           f"{experiment_configuration['task']} task saves no error image (no "
                                           f"'evaluation.error_dumps' section).")
        return
    error_dumps_configuration["activated"] = True
    error_dumps_configuration["maximum_per_class_pair"] = maximum_error_images_per_class_pair


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
    """
    if target_precisions is None:
        return

    # The tasks without class thresholds have no 'evaluation.target_precision' setting to replace
    evaluation_configuration = experiment_configuration.get("evaluation", {})
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


if __name__ == "__main__":
    # Run name of the trained experiment whose configuration and checkpoints are evaluated, it is only read, in the
    # experiments root folder of its configuration ('experiment_folder')
    run_name = "CLS_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HFLAT.128-HLN-CWB_MC9"
    experiment_folder = Path(__file__).absolute().parents[2] / "experiments" / "readme_screenshots" / run_name
    # Checkpoint iteration to evaluate, None for the last saved one, or "all" for every checkpoint of the registry
    # (the intermediate ones included, the last saved first), compared in 'Evaluation/evaluation_summary.csv'
    iteration = 1500
    # Labelled split evaluated instead of the test split of the experiment, None to keep the experiment's one
    test_data_folder = None
    # Batch size of the evaluation, which can be larger than the training one since no gradient is kept, None to
    # keep the experiment's one
    batch_size = 128
    # Tracks the evaluation in an MLflow run of its own, named '<run name>-evaluation' in the backend of the evaluated
    # experiment and tagged with its training run ('evaluation_of'), True or False, None to keep the experiment's
    # 'tracking.activated'. The MLflow run of the training is never modified
    track_evaluation = None
    # Largest number of error images saved for every pair of true and predicted classes, the first ones met, in
    # 'Evaluation/Iteration_<n>/Errors/<TRUE>-TO-<PREDICTED>/', the error images being turned on. A number above the
    # number of test samples saves every error. None keeps the experiment's 'evaluation.error_dumps' settings. Skipped
    # with a warning for the tasks without error images (semantic segmentation, keypoint detection, self-supervised)
    maximum_error_images_per_class_pair = 100000
    # Precision targeted by the class thresholds, or a list of distinct precisions whose first one is the main target,
    # e.g. [0.99, 0.95, 0.90] to read the thresholds of every class at each of them in class_thresholds.json,
    # class_metrics.csv and precision_recall_thresholds.png. None keeps the experiment's 'evaluation.target_precision'.
    # A list is refused by the frame selection and the detection tasks; skipped with a warning for the tasks without
    # class thresholds (semantic segmentation, keypoint detection, self-supervised)
    target_precisions = None
    # Root folder of the evaluations, kept apart from the experiment so that its files are untouched. Like a training
    # run, Results.txt, Tensorboard/Test and the Evaluation folder are written in a subfolder of it with the name of
    # the evaluated run folder. The export looks for the thresholds saved from the temperature explorer of a classifier
    # in this default folder
    output_folder = experiment_folder.parent / EVALUATIONS_FOLDER_NAME

    # Point the copied configuration to the output folder and to the evaluated split, its 'tracking_root' keeping the
    # experiments root folder of the run so that the panel names the MLflow backend the evaluation is tracked in
    configuration_path = experiment_folder / "training_configuration.yaml"
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    experiment_configuration["experiment_folder"] = output_folder
    # The output subfolder keeps the name of the evaluated run folder, which the settings no longer give once the
    # transfer of pretrained components is turned off below
    experiment_configuration["run_name"] = experiment_folder.name
    if test_data_folder is not None:
        experiment_configuration["test_data_folder"] = test_data_folder
    if batch_size is not None:
        experiment_configuration["batch_size"] = batch_size
    if track_evaluation is not None:
        experiment_configuration["tracking"]["activated"] = track_evaluation
    # Error images turned on with their own limit per pair of classes, for the tasks saving their errors as images
    apply_error_image_limit(experiment_configuration=experiment_configuration,
                            maximum_error_images_per_class_pair=maximum_error_images_per_class_pair)
    # Target precisions of the class thresholds replaced, for the tasks giving class thresholds
    apply_target_precisions(experiment_configuration=experiment_configuration, target_precisions=target_precisions)

    # The weights come from the evaluated checkpoints, so the pretrained components the run started from are not loaded
    disable_pretrained_settings(model_configuration=model_configuration)

    trainer_class = get_trainer_class(task=experiment_configuration["task"])
    trainer = trainer_class(experiment_configuration=experiment_configuration,
                            model_configuration=model_configuration, configuration_path=configuration_path)

    # Every checkpoint of the registry from the last saved one, so that the most trained weights are evaluated first,
    # the last one, or the given one
    weights_directory = experiment_folder / "Weights"
    if iteration == "all":
        evaluated_iterations = get_checkpoint_iterations(weights_directory=weights_directory)[::-1]
    elif iteration is None:
        evaluated_iterations = [get_last_checkpoint_iteration(weights_directory=weights_directory)]
    else:
        evaluated_iterations = [iteration]
    evaluate_checkpoints(trainer=trainer, experiment_folder=experiment_folder, iterations=evaluated_iterations)
