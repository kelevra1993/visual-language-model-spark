"""
Comparison of the configuration a run is resumed with against the configuration saved in its run folder: the settings
that give no token in the run name (learning rate, learning rate multipliers, batch size, dropout, augmentation...)
can change between two resumptions of a run, so the base trainer lists every changed setting at launch and keeps the
previous configuration file, which would otherwise be replaced by the copy of the new one.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

# Value of a setting missing from one of the two compared configurations, a key added or removed between two starts
MISSING_SETTING = object()


class ConfigurationChanges(TypedDict):
    """
    Changes of the configuration of a resumed run, found by the base trainer before it replaces the configuration saved
    in the run folder, and printed with the experiment information: the iteration of the checkpoint the run resumes
    from, every changed setting as (dotted key, previous value, new value), and the file keeping the previous
    configuration (None when nothing changed, the saved file being then identical).
    """
    resumed_iteration: int
    changed_settings: List[Tuple[str, Any, Any]]
    archived_configuration_path: Optional[Path]


def flatten_settings(configuration: Dict[Any, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Flattens the nested sections of a YAML configuration into dotted keys with their exact values, e.g.
    'Backbone.pretrained.learning_rate_multiplier', so that two configurations can be compared setting by setting. A
    list or an empty section is one setting.

    Args:
        configuration (Dict[Any, Any]): The nested settings, as read from the YAML file.
        prefix (str): The dotted key of the section within the whole configuration, empty at the top.

    Returns:
        Dict[str, Any]: The value of every setting, keyed by its dotted key, in the order of the file.
    """
    settings = {}
    for key, value in configuration.items():
        setting_key = f"{prefix}{key}"

        # A non-empty section gives one setting per entry, through its own sub-sections
        if isinstance(value, dict) and value:
            settings.update(flatten_settings(configuration=value, prefix=f"{setting_key}."))
        else:
            settings[setting_key] = value
    return settings


def find_changed_settings(previous_configuration: Dict[Any, Any],
                          current_configuration: Dict[Any, Any]) -> List[Tuple[str, Any, Any]]:
    """
    Lists the settings whose value differs between the configuration saved in a run folder and the one the run is
    resumed with, for the launch information of the base trainer: the changed and added settings in the order of the
    new configuration, then the removed ones, a missing value being MISSING_SETTING.

    Args:
        previous_configuration (Dict[Any, Any]): The configuration saved in the run folder by the previous start.
        current_configuration (Dict[Any, Any]): The configuration the run is resumed with.

    Returns:
        List[Tuple[str, Any, Any]]: Every changed setting as (dotted key, previous value, new value), empty when the
            two configurations hold the same settings.
    """
    previous_settings = flatten_settings(configuration=previous_configuration)
    current_settings = flatten_settings(configuration=current_configuration)

    # Settings changed or added by the new configuration, in its order
    changed_settings = []
    for setting_key, current_value in current_settings.items():
        previous_value = previous_settings.get(setting_key, MISSING_SETTING)
        if previous_value is MISSING_SETTING or previous_value != current_value:
            changed_settings.append((setting_key, previous_value, current_value))

    # Settings removed by the new configuration
    for setting_key, previous_value in previous_settings.items():
        if setting_key not in current_settings:
            changed_settings.append((setting_key, previous_value, MISSING_SETTING))
    return changed_settings


def format_setting_value(value: Any) -> str:
    """
    Writes the value of a changed setting for the launch information of the base trainer, in the spelling of the YAML
    files for the empty values.

    Args:
        value (Any): The value of the setting, or MISSING_SETTING when one of the configurations does not hold it.

    Returns:
        str: The value as printed, '<absent>' for a missing setting and 'null' for None.
    """
    if value is MISSING_SETTING:
        return "<absent>"
    if value is None:
        return "null"
    return str(value)
