from typing import Any, Dict, List

from vision_training.configuration.common_checks import (check_image_task_augmentations,
                                                         check_pretrained_configuration)
from vision_training.configuration.configuration_values import (check_boolean_flags, check_positive_integers,
                                                                check_probability_value, get_configuration_value)

# Global pooling of the final feature map before the classification head
VALID_GLOBAL_POOLINGS = ["average", "max"]

# Output modes of the classification head: one class per sample (softmax) or several (sigmoid)
VALID_OUTPUT_MODES = ["multi_class", "multi_label"]

# Weighting of the classes in the classification loss: uniform, or inversely proportional to the training class counts
VALID_CLASS_WEIGHTINGS = ["none", "balanced"]


def check_classification_data_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the data settings of the classification-like tasks (classification and both sequence tasks): the Native
    loaders, a multi-class label mapping whose class indices are 0 to K-1 (the classifiers predict no background
    class), and no detection block, their heads using the final feature map of the backbone.

    Args:
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []

    # The DALI pipelines only read the COCO detection format
    if get_configuration_value(configuration=model_configuration, key_path="Data.DALI.activated") is True:
        errors.append("'Data.DALI.activated' must be False: the DALI loaders are only available for the object "
                      "detection task, the other tasks use the Native loaders.")

    # The classifiers predict the mapped classes only, indexed from 0 without gaps
    label_mode = get_configuration_value(configuration=model_configuration, key_path="Data.label_mapping.mode")
    if label_mode != "multi_class":
        errors.append(f"'Data.label_mapping.mode' must be 'multi_class' for the classification tasks (the mapping "
                      f"file gives the class index of every category), got {label_mode!r}.")
    else:
        class_indices = sorted(set(get_configuration_value(configuration=model_configuration,
                                                           key_path="Data.label_dictionary").values()))
        if len(class_indices) < 2 or class_indices != list(range(len(class_indices))):
            errors.append(f"The classes of the label mapping file must be at least two, indexed 0 to K-1 without gaps, "
                          f"got {class_indices}.")

    # The heads use the final feature map of the backbone, so no block feeds them
    detection_block_indices = get_configuration_value(configuration=model_configuration,
                                                      key_path="Backbone.detection_convolution_block_indices")
    if detection_block_indices != []:
        errors.append(f"'Backbone.detection_convolution_block_indices' must be [] for the classification tasks, their "
                      f"heads use the final feature map, got {detection_block_indices!r}.")

    return errors


def check_classification_head_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'ClassificationHead' section shared by the classification-like tasks: the global pooling, the hidden
    fully connected layers, the dropout, the output mode, the label smoothing, the class weighting of the loss, and
    the 'pretrained' settings initialising the head with the one of another run.

    Args:
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []

    global_pooling = get_configuration_value(configuration=model_configuration,
                                             key_path="ClassificationHead.global_pooling")
    if global_pooling not in VALID_GLOBAL_POOLINGS:
        errors.append(f"'ClassificationHead.global_pooling' must be one of {VALID_GLOBAL_POOLINGS}, "
                      f"got {global_pooling!r}.")

    # Hidden layers, possibly none, each with a positive number of units
    fully_connected = get_configuration_value(configuration=model_configuration,
                                              key_path="ClassificationHead.fully_connected")
    if not isinstance(fully_connected, list) or not all(isinstance(units, int) and not isinstance(units, bool) and
                                                        units >= 1 for units in fully_connected):
        errors.append(f"'ClassificationHead.fully_connected' must be a list of positive integers, "
                      f"got {fully_connected!r}.")

    # Dropout and label smoothing rates, strictly below 1 so that some signal always remains
    for key_name in ["dropout_rate", "label_smoothing"]:
        key_path = f"ClassificationHead.{key_name}"
        value = get_configuration_value(configuration=model_configuration, key_path=key_path)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value < 1.0:
            errors.append(f"'{key_path}' must be a number within [0, 1), got {value!r}.")

    output_mode = get_configuration_value(configuration=model_configuration, key_path="ClassificationHead.output_mode")
    if output_mode not in VALID_OUTPUT_MODES:
        errors.append(f"'ClassificationHead.output_mode' must be one of {VALID_OUTPUT_MODES}, got {output_mode!r}.")
    class_weighting = get_configuration_value(configuration=model_configuration,
                                              key_path="ClassificationHead.class_weighting")
    if class_weighting not in VALID_CLASS_WEIGHTINGS:
        errors.append(f"'ClassificationHead.class_weighting' must be one of {VALID_CLASS_WEIGHTINGS}, "
                      f"got {class_weighting!r}.")

    # Head possibly initialised by the one of another run
    errors += check_pretrained_configuration(model_configuration=model_configuration,
                                             section_name="ClassificationHead")
    return errors


def check_classification_evaluation_configuration(experiment_configuration: Dict[str, Any],
                                                  model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'ExperimentConfiguration.evaluation' settings of the classification-like tasks: the k of the top-k
    accuracy, the precision targeted by the per-class threshold finder, whether the predictions of every test
    sample are saved, and the error images of the misclassified samples, which need one class per sample.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        model_configuration (Dict[str, Any]): The processed model sections, giving the output mode of the head.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    # Nest the section under its name so that every reported key path matches its location in the YAML file
    configuration = {"ExperimentConfiguration": experiment_configuration}
    errors = check_positive_integers(configuration=configuration,
                                     key_paths=["ExperimentConfiguration.evaluation.top_k"],
                                     allow_zero=False)
    target_precision = get_configuration_value(configuration=configuration,
                                               key_path="ExperimentConfiguration.evaluation.target_precision")
    errors += check_probability_value(value=target_precision,
                                      key_path="ExperimentConfiguration.evaluation.target_precision")
    errors += check_boolean_flags(configuration=configuration,
                                  key_paths=["ExperimentConfiguration.evaluation.save_predictions"])

    # Error images, at most a positive number per class pair, for one class per sample only
    error_flag_errors = check_boolean_flags(configuration=configuration,
                                            key_paths=["ExperimentConfiguration.evaluation.error_dumps.activated"])
    errors += error_flag_errors
    errors += check_positive_integers(configuration=configuration, allow_zero=False,
                                      key_paths=["ExperimentConfiguration.evaluation.error_dumps."
                                                 "maximum_per_class_pair"])
    output_mode = get_configuration_value(configuration=model_configuration, key_path="ClassificationHead.output_mode")
    if not error_flag_errors and output_mode == "multi_label" and get_configuration_value(
            configuration=configuration, key_path="ExperimentConfiguration.evaluation.error_dumps.activated"):
        errors.append("'ExperimentConfiguration.evaluation.error_dumps.activated' must be False with "
                      "'ClassificationHead.output_mode' 'multi_label': a sample with several classes has no single "
                      "true and predicted class pair.")
    return errors


def check_classification_configuration(experiment_configuration: Dict[str, Any],
                                       model_configuration: Dict[str, Any]) -> List[str]:
    """
    Runs every check specific to the classification task: its data settings, without the sequence augmentations,
    its classification head and its evaluation settings.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    errors += check_classification_data_configuration(model_configuration=model_configuration)
    errors += check_image_task_augmentations(model_configuration=model_configuration)
    errors += check_classification_head_configuration(model_configuration=model_configuration)
    errors += check_classification_evaluation_configuration(experiment_configuration=experiment_configuration,
                                                            model_configuration=model_configuration)
    return errors
