"""
Checks of the configuration sections specific to the image classification task: the data settings (label mode,
class weighting), the classification head (global pooling, hidden layers and their normalization, output mode,
dropout, label smoothing), the decorrelation of its hidden features and the evaluation settings. Run by
configuration_checker.py for the classification task and reused by the sequence task checks, except the decorrelation,
a setting of the image classifier only.
"""

from typing import Any, Dict, List

from vision_training.configuration_processing.common_checks import (check_error_dumps_configuration,
                                                                    check_image_task_sequence_settings,
                                                                    check_pretrained_configuration)
from vision_training.configuration_processing.configuration_values import (VALID_FEATURE_MAP_NORMALIZATIONS,
                                                                           check_batch_normalization_momentum,
                                                                           check_boolean_flags, check_positive_integers,
                                                                           check_probability_value,
                                                                           get_configuration_value, is_number)

# Global pooling of the final feature map before the classification head (or the temporal module of the sequence
# models), or its flattening
VALID_GLOBAL_POOLINGS = ["average", "max", "flatten"]

# Output modes of the classification head: one class per sample (softmax) or several (sigmoid)
VALID_OUTPUT_MODES = ["multi_class", "multi_label"]

# Weighting of the classes in the classification loss: uniform, or inversely proportional to the training class counts
VALID_CLASS_WEIGHTINGS = ["none", "balanced"]

# Normalization of the hidden layers of the classification head, the values of the feature map normalizations
VALID_HEAD_NORMALIZATIONS = VALID_FEATURE_MAP_NORMALIZATIONS

# Value replacing every True or False normalization of the hidden layers written before the layer normalization
BOOLEAN_HEAD_NORMALIZATIONS = {True: "batch", False: "none"}


def check_classification_data_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the data settings of the classification-like tasks (classification and both sequence tasks): a
    multi-class label mapping whose class indices are 0 to K-1 (the classifiers predict no background class).

    Args:
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []

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

    return errors


def check_classification_head_configuration(experiment_configuration: Dict[str, Any],
                                            model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'ClassificationHead' section shared by the classification-like tasks: the global pooling, the hidden
    fully connected layers, their normalization and its batch momentum (a batch normalization needing batches of at
    least two samples), the dropout, the output mode, the label smoothing, the class weighting of the loss, and the
    'pretrained' settings initialising the head with the one of another run.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration, giving the batch size.
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

    # Normalization of the hidden layers, which needs at least one of them, a True or False of the configurations
    # written before the layer normalization being reported with its new value
    fully_connected_normalization = get_configuration_value(
        configuration=model_configuration, key_path="ClassificationHead.fully_connected_normalization")
    if isinstance(fully_connected_normalization, bool):
        errors.append(f"'ClassificationHead.fully_connected_normalization' must be one of {VALID_HEAD_NORMALIZATIONS}, "
                      f"got {fully_connected_normalization!r}, written before the layer normalization: it is now "
                      f"{BOOLEAN_HEAD_NORMALIZATIONS[fully_connected_normalization]!r}.")
    elif fully_connected_normalization not in VALID_HEAD_NORMALIZATIONS:
        errors.append(f"'ClassificationHead.fully_connected_normalization' must be one of "
                      f"{VALID_HEAD_NORMALIZATIONS}, got {fully_connected_normalization!r}.")
    elif fully_connected_normalization != "none" and fully_connected == []:
        errors.append("'ClassificationHead.fully_connected_normalization' must be 'none' when "
                      "'ClassificationHead.fully_connected' is []: the head has no hidden layer to normalize.")
    errors += check_batch_normalization_momentum(configuration=model_configuration,
                                                 key_path="ClassificationHead.batch_normalization_momentum")

    # A batch normalization of the hidden layers trains on the statistics of a batch, which one sample cannot give
    batch_size = get_configuration_value(configuration={"ExperimentConfiguration": experiment_configuration},
                                         key_path="ExperimentConfiguration.batch_size")
    if fully_connected_normalization == "batch" and isinstance(batch_size, int) and not isinstance(batch_size, bool) \
            and batch_size == 1:
        errors.append("'ExperimentConfiguration.batch_size' must be at least 2 with "
                      "'ClassificationHead.fully_connected_normalization' 'batch': the batch normalization of the "
                      "hidden layers needs the statistics of several samples.")

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


def check_classification_decorrelation_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'ClassificationHead.decorrelation' settings of the image classifier, which adds to its loss the
    squared off-diagonal correlations of the features of one hidden layer of its head: the on/off flag, a weight above
    0, and the hidden layer, a positive integer counted from 1 in 'ClassificationHead.fully_connected', which must
    exist when the decorrelation is on. The sequence tasks sharing the classification head have no such setting.

    Args:
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = check_boolean_flags(configuration=model_configuration,
                                 key_paths=["ClassificationHead.decorrelation.activated"])
    weight = get_configuration_value(configuration=model_configuration,
                                     key_path="ClassificationHead.decorrelation.weight")
    if not is_number(value=weight) or weight <= 0:
        errors.append(f"'ClassificationHead.decorrelation.weight' must be a number > 0, got {weight!r}.")
    hidden_layer_errors = check_positive_integers(configuration=model_configuration,
                                                  key_paths=["ClassificationHead.decorrelation.hidden_layer"],
                                                  allow_zero=False)
    errors += hidden_layer_errors

    # The decorrelated layer must be one of the hidden layers of the head when the decorrelation is on
    activated = get_configuration_value(configuration=model_configuration,
                                        key_path="ClassificationHead.decorrelation.activated")
    hidden_layer = get_configuration_value(configuration=model_configuration,
                                           key_path="ClassificationHead.decorrelation.hidden_layer")
    fully_connected = get_configuration_value(configuration=model_configuration,
                                              key_path="ClassificationHead.fully_connected")
    if activated is True and not hidden_layer_errors and isinstance(fully_connected, list) and \
            hidden_layer > len(fully_connected):
        errors.append(f"'ClassificationHead.decorrelation.hidden_layer' must be at most the number of hidden layers of "
                      f"'ClassificationHead.fully_connected' ({len(fully_connected)}) when "
                      f"'ClassificationHead.decorrelation.activated' is True, got {hidden_layer}.")
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
    errors += check_error_dumps_configuration(experiment_configuration=experiment_configuration)
    output_mode = get_configuration_value(configuration=model_configuration, key_path="ClassificationHead.output_mode")
    if output_mode == "multi_label" and get_configuration_value(
            configuration=configuration, key_path="ExperimentConfiguration.evaluation.error_dumps.activated") is True:
        errors.append("'ExperimentConfiguration.evaluation.error_dumps.activated' must be False with "
                      "'ClassificationHead.output_mode' 'multi_label': a sample with several classes has no single "
                      "true and predicted class pair.")
    return errors


def check_classification_configuration(experiment_configuration: Dict[str, Any],
                                       model_configuration: Dict[str, Any]) -> List[str]:
    """
    Runs every check specific to the classification task: its data settings, without the sequence settings,
    its classification head with the decorrelation of its hidden features, and its evaluation settings.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    errors += check_classification_data_configuration(model_configuration=model_configuration)
    errors += check_image_task_sequence_settings(model_configuration=model_configuration)
    errors += check_classification_head_configuration(experiment_configuration=experiment_configuration,
                                                      model_configuration=model_configuration)
    errors += check_classification_decorrelation_configuration(model_configuration=model_configuration)
    errors += check_classification_evaluation_configuration(experiment_configuration=experiment_configuration,
                                                            model_configuration=model_configuration)
    return errors
