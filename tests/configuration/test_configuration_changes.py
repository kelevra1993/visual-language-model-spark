"""
Tests of vision_training/configuration/configuration_changes.py: the YAML configurations are flattened into dotted
keys with their exact values, and the settings changed, added and removed between the configuration saved in a run
folder and the one a run is resumed with are listed and formatted for the launch information of the base trainer.
"""

from vision_training.configuration.configuration_changes import (MISSING_SETTING, find_changed_settings,
                                                                  flatten_settings, format_setting_value)

# Configuration saved in a run folder by its previous start, as read from its YAML file
PREVIOUS_CONFIGURATION = {"ExperimentConfiguration": {"learning_rate": 2.0e-4, "batch_size": 32,
                                                      "tracking": {"activated": False, "experiment_name": None}},
                          "Backbone": {"convolutions": {1: [1, 64], 2: [2, 64]}, "modules": {},
                                       "pretrained": {"experiment_folder": "/runs/source", "freeze": False,
                                                      "learning_rate_multiplier": 0.5}},
                          "Data": {"removed_setting": True}}


def test_flatten_settings_gives_dotted_keys_with_exact_values() -> None:
    """
    Verifies that every non-empty section gives one dotted key per entry, integer keys included, while the lists, the
    empty sections and the null values stay single settings with their exact values (no string conversion nor cut).
    """
    assert flatten_settings(configuration=PREVIOUS_CONFIGURATION) == {
        "ExperimentConfiguration.learning_rate": 2.0e-4, "ExperimentConfiguration.batch_size": 32,
        "ExperimentConfiguration.tracking.activated": False, "ExperimentConfiguration.tracking.experiment_name": None,
        "Backbone.convolutions.1": [1, 64], "Backbone.convolutions.2": [2, 64], "Backbone.modules": {},
        "Backbone.pretrained.experiment_folder": "/runs/source", "Backbone.pretrained.freeze": False,
        "Backbone.pretrained.learning_rate_multiplier": 0.5, "Data.removed_setting": True}


def test_identical_configurations_have_no_changed_setting() -> None:
    """
    Verifies that a run resumed with the configuration saved in its folder has no changed setting, whatever the order
    of the keys.
    """
    reordered_configuration = dict(reversed(list(PREVIOUS_CONFIGURATION.items())))
    assert find_changed_settings(previous_configuration=PREVIOUS_CONFIGURATION,
                                 current_configuration=reordered_configuration) == []


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


def test_setting_values_are_formatted_like_the_yaml_files() -> None:
    """
    Verifies the spelling of the printed values: '<absent>' for a missing setting, 'null' for None, and the plain
    value otherwise.
    """
    assert [format_setting_value(value=value) for value in [MISSING_SETTING, None, 0.5, False, [1, 64], "adamw"]] == [
        "<absent>", "null", "0.5", "False", "[1, 64]", "adamw"]
