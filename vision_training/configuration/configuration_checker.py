from typing import Any, Dict

from vision_training.configuration.common_checks import (check_backbone_configuration,
                                                         check_data_configuration,
                                                         check_experiment_configuration)
from vision_training.configuration.configuration_values import ConfigurationError
from vision_training.configuration.detection_checks import (check_anchors_configuration,
                                                            check_detection_levels,
                                                            check_detector_configuration,
                                                            check_region_proposal_configuration)


def check_configuration(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Verifies that a loaded configuration file is complete and consistent before any model or data loader is built.
    It is run by 'load_experiment_configuration', so an inconsistent file is rejected at startup with the full list
    of problems instead of failing later inside model construction or, worse, training silently with wrong values.

    Args:
        experiment_configuration (Dict[str, Any]): The processed 'ExperimentConfiguration' section.
        model_configuration (Dict[str, Any]): The processed model sections (Data, Backbone, RegionProposal, Anchors,
            Detector).

    Raises:
        ConfigurationError: If a required configuration key is missing, or if one or more configuration values are
            inconsistent, listing all of them.
    """
    errors = []

    # Run every section check and collect all the problems so they can be fixed in a single pass
    errors += check_experiment_configuration(experiment_configuration=experiment_configuration)
    errors += check_data_configuration(model_configuration=model_configuration)
    errors += check_backbone_configuration(model_configuration=model_configuration)
    errors += check_detection_levels(model_configuration=model_configuration)
    errors += check_anchors_configuration(model_configuration=model_configuration)
    errors += check_region_proposal_configuration(model_configuration=model_configuration)
    errors += check_detector_configuration(model_configuration=model_configuration)

    if errors:
        raise ConfigurationError(errors=errors)
