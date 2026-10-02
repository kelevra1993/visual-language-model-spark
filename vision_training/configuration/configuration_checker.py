from typing import Any, Callable, Dict, List

from vision_training.configuration.classification_checks import check_classification_configuration
from vision_training.configuration.common_checks import (check_backbone_configuration,
                                                         check_data_configuration,
                                                         check_experiment_configuration,
                                                         check_optimization_configuration)
from vision_training.configuration.configuration_values import ConfigurationError
from vision_training.configuration.detection_checks import check_detection_configuration
from vision_training.configuration.sequence_checks import (check_sequence_classification_configuration,
                                                           check_sequence_labelling_configuration)

# Checks of the sections specific to every task, run after the checks of the sections common to every task. The keys
# are the tasks that can be trained, i.e. the valid values of 'ExperimentConfiguration.task'
TASK_CONFIGURATION_CHECKS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], List[str]]] = {
    "object_detection": check_detection_configuration, "classification": check_classification_configuration,
    "sequence_classification": check_sequence_classification_configuration,
    "sequence_labelling": check_sequence_labelling_configuration}


def check_configuration(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Verifies that a loaded configuration file is complete and consistent before any model or data loader is built.
    It is run by 'load_experiment_configuration', so an inconsistent file is rejected at startup with the full list
    of problems instead of failing later inside model construction or, worse, training silently with wrong values.
    The sections common to every task (experiment, data, backbone, optimization) are checked first, then the sections
    of the task.

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
    errors += check_optimization_configuration(experiment_configuration=experiment_configuration,
                                               model_configuration=model_configuration)

    # Then the checks of the sections of the task, when the task itself is valid
    task = experiment_configuration["task"]
    if task in TASK_CONFIGURATION_CHECKS:
        errors += TASK_CONFIGURATION_CHECKS[task](experiment_configuration=experiment_configuration,
                                                  model_configuration=model_configuration)

    if errors:
        raise ConfigurationError(errors=errors)
