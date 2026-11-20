"""
Entry point of the configuration checks: refuses the removed settings, then the keys the code does not read, on the
file as written (check_configuration_keys), then runs the checks of the sections common to every task and those of
the task of 'ExperimentConfiguration.task' on the processed configuration (check_configuration), each step raising a
ConfigurationError listing every problem it found. Called by configuration_loader.py when a configuration file is
loaded, before any model or data loader is built.
"""

from typing import Any, Callable, Dict, List

from vision_training.configuration_processing.classification_checks import check_classification_configuration
from vision_training.configuration_processing.common_checks import (check_backbone_configuration,
                                                                    check_data_configuration,
                                                                    check_experiment_configuration,
                                                                    check_optimization_configuration,
                                                                    check_removed_settings)
from vision_training.configuration_processing.configuration_keys import check_unknown_keys
from vision_training.configuration_processing.configuration_values import ConfigurationError
from vision_training.configuration_processing.detection_checks import check_detection_configuration
from vision_training.configuration_processing.instance_segmentation_checks import (
    check_instance_segmentation_configuration)
from vision_training.configuration_processing.keypoint_checks import check_keypoint_configuration
from vision_training.configuration_processing.segmentation_checks import check_segmentation_configuration
from vision_training.configuration_processing.self_supervised_checks import check_self_supervised_configuration
from vision_training.configuration_processing.sequence_checks import (check_frame_selection_configuration,
                                                                      check_sequence_classification_configuration,
                                                                      check_sequence_comparison_configuration,
                                                                      check_sequence_labelling_configuration)

# Checks of the sections specific to every task, run after the checks of the sections common to every task. The keys
# are the tasks that can be trained, i.e. the valid values of 'ExperimentConfiguration.task'
TASK_CONFIGURATION_CHECKS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], List[str]]] = {
    "object_detection": check_detection_configuration,
    "instance_segmentation": check_instance_segmentation_configuration,
    "classification": check_classification_configuration,
    "semantic_segmentation": check_segmentation_configuration,
    "sequence_classification": check_sequence_classification_configuration,
    "sequence_labelling": check_sequence_labelling_configuration,
    "sequence_comparison": check_sequence_comparison_configuration,
    "frame_selection": check_frame_selection_configuration,
    "self_supervised": check_self_supervised_configuration,
    "keypoint_detection": check_keypoint_configuration}


def check_configuration_keys(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Verifies the keys of a configuration file as written, before the loader converts its values and adds its own keys
    (the run name, the label dictionary...): the settings and sections removed from the configuration files are
    refused first, with what replaced them, then every key the code of the task does not read (a misspelled key, a
    setting no code reads, a section of another task), whose value would otherwise be silently ignored. Run by
    'prepare_experiment_configuration' before check_configuration.

    Args:
        experiment_configuration (Dict[str, Any]): The 'ExperimentConfiguration' section as written.
        model_configuration (Dict[str, Any]): The other sections as written.

    Raises:
        ConfigurationError: If the file holds removed settings, listing all of them, or else unknown keys, listing
            every one of them by its path.
    """
    # Refuse the removed settings on their own: a file written before their removal also lacks the keys added with it,
    # whose first missing key would stop the other checks without naming the removed setting
    removed_setting_errors = check_removed_settings(experiment_configuration=experiment_configuration,
                                                    model_configuration=model_configuration)
    if removed_setting_errors:
        raise ConfigurationError(errors=removed_setting_errors)

    # Then the unknown keys on their own, a misspelled key leaving its correct key missing, whose first missing key
    # would stop the other checks without naming the misspelled one
    unknown_key_errors = check_unknown_keys(experiment_configuration=experiment_configuration,
                                            model_configuration=model_configuration)
    if unknown_key_errors:
        raise ConfigurationError(errors=unknown_key_errors)


def check_configuration(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Verifies that a loaded configuration file is complete and consistent before any model or data loader is built.
    It is run by 'prepare_experiment_configuration', so an inconsistent file is rejected at startup with the full list
    of problems instead of failing later inside model construction or, worse, training silently with wrong values.
    The keys of the file having been checked as written (check_configuration_keys), the sections common to every task
    (experiment, data, backbone, optimization) are checked, then the sections of the task.

    Args:
        experiment_configuration (Dict[str, Any]): The processed 'ExperimentConfiguration' section.
        model_configuration (Dict[str, Any]): The processed model sections (Data, Backbone and the task sections).

    Raises:
        ConfigurationError: If a required configuration key is missing, or if one or more configuration values are
            inconsistent, listing all of them.
    """
    errors = []

    # Run every check of the common sections and collect all the problems so they can be fixed in a single pass
    errors += check_experiment_configuration(experiment_configuration=experiment_configuration,
                                             valid_tasks=list(TASK_CONFIGURATION_CHECKS))
    errors += check_data_configuration(model_configuration=model_configuration)
    errors += check_backbone_configuration(model_configuration=model_configuration)
    errors += check_optimization_configuration(model_configuration=model_configuration)

    # Then the checks of the sections of the task, when the task itself is valid
    task = experiment_configuration["task"]
    if task in TASK_CONFIGURATION_CHECKS:
        errors += TASK_CONFIGURATION_CHECKS[task](experiment_configuration=experiment_configuration,
                                                  model_configuration=model_configuration)

    if errors:
        raise ConfigurationError(errors=errors)
