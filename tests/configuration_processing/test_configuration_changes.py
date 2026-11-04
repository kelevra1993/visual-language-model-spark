"""
Tests of vision_training/configuration_processing/configuration_changes.py: the settings changed, added and removed
between the configuration saved in a run folder and the one a run is resumed with are listed as dotted keys for the
launch information of the base trainer.
"""

from vision_training.configuration_processing.configuration_changes import MISSING_SETTING, find_changed_settings

# Configuration saved in a run folder by its previous start, as read from its YAML file
PREVIOUS_CONFIGURATION = {"ExperimentConfiguration": {"learning_rate": 2.0e-4, "batch_size": 32,
                                                      "tracking": {"activated": False, "experiment_name": None}},
                          "Backbone": {"convolutions": {1: [1, 64], 2: [2, 64]}, "modules": {},
                                       "pretrained": {"experiment_folder": "/runs/source", "freeze": False,
                                                      "learning_rate_multiplier": 0.5}},
                          "Data": {"removed_setting": True}}


def test_changed_added_and_removed_settings_are_listed() -> None:
    """
    Verifies that the settings changed and added by the new configuration are listed in its order, then the removed
    ones, a missing value being MISSING_SETTING, and that a changed list counts as one changed setting.
    """
    current_configuration = {"ExperimentConfiguration": {"learning_rate": 1.0e-4, "batch_size": 32,
                                                         "tracking": {"activated": False, "experiment_name": None}},
                             "Backbone": {"convolutions": {1: [1, 64], 2: [3, 64]}, "modules": {},
                                          "pretrained": {"experiment_folder": "/runs/source", "freeze": False,
                                                         "learning_rate_multiplier": 1.0}},
                             "Data": {"added_setting": "value"}}
    assert find_changed_settings(previous_configuration=PREVIOUS_CONFIGURATION,
                                 current_configuration=current_configuration) == [
        ("ExperimentConfiguration.learning_rate", 2.0e-4, 1.0e-4), ("Backbone.convolutions.2", [2, 64], [3, 64]),
        ("Backbone.pretrained.learning_rate_multiplier", 0.5, 1.0), ("Data.added_setting", MISSING_SETTING, "value"),
        ("Data.removed_setting", True, MISSING_SETTING)]
