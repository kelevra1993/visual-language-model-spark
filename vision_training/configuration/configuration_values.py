from typing import Any, Dict, List


# Feature map normalization strategies supported by the ConvolutionBlock, and those of the backbone, which also accepts
# the sequence normalization (batch normalization with the statistics of each sequence, sequence tasks only)
VALID_FEATURE_MAP_NORMALIZATIONS = ["batch", "layer", "none"]
BACKBONE_FEATURE_MAP_NORMALIZATIONS = VALID_FEATURE_MAP_NORMALIZATIONS + ["sequence"]


class ConfigurationError(Exception):
    """
    Raised when the configuration file is incomplete or inconsistent. It carries the plain list of problems so that
    'load_experiment_configuration' can display them as short messages instead of a full traceback.
    """

    def __init__(self, errors: List[str]) -> None:
        """
        Stores the configuration problems found by the checker.

        Args:
            errors (List[str]): One short description per configuration problem.
        """
        super(ConfigurationError, self).__init__("\n".join(errors))
        self.errors = errors


def get_configuration_value(configuration: Dict[str, Any], key_path: str) -> Any:
    """
    Retrieves a nested configuration value from a dotted key path (e.g. 'Backbone.modules.inception').

    The configuration checker relies on this lookup so that a key missing from the YAML file is reported with its
    full location in the file, rather than as a bare KeyError raised deep inside the model construction.

    Args:
        configuration (Dict[str, Any]): The configuration dictionary to search.
        key_path (str): The dot-separated path of keys leading to the value.

    Returns:
        Any: The value stored at the given key path.

    Raises:
        ConfigurationError: If any key along the path is missing from the configuration.
    """
    current_value = configuration

    # Walk down the nested dictionaries one key at a time
    for key in key_path.split("."):
        if not isinstance(current_value, dict) or key not in current_value:
            raise ConfigurationError(errors=[f"Missing configuration key '{key_path}'."])
        current_value = current_value[key]

    return current_value


def check_boolean_flags(configuration: Dict[str, Any], key_paths: List[str]) -> List[str]:
    """
    Verifies that on/off flags are real booleans. A quoted "False" in the YAML is a non-empty string, which Python
    evaluates as True, so it would silently enable a feature of the pipeline.

    Args:
        configuration (Dict[str, Any]): The configuration dictionary containing the flags.
        key_paths (List[str]): The dotted key paths of the flags to verify.

    Returns:
        List[str]: The description of every flag that is not a boolean.
    """
    errors = []

    # Check the type of every listed flag
    for key_path in key_paths:
        value = get_configuration_value(configuration=configuration, key_path=key_path)
        if not isinstance(value, bool):
            errors.append(f"'{key_path}' must be True or False, got {value!r}.")

    return errors


def check_positive_integers(configuration: Dict[str, Any], key_paths: List[str], allow_zero: bool) -> List[str]:
    """
    Verifies that counts used by the pipeline (iterations, batch sizes, proposal numbers...) are integers and
    strictly positive, or non-negative when zero is meaningful.

    Args:
        configuration (Dict[str, Any]): The configuration dictionary containing the values.
        key_paths (List[str]): The dotted key paths of the values to verify.
        allow_zero (bool): Whether zero is an accepted value.

    Returns:
        List[str]: The description of every invalid value.
    """
    errors = []
    minimum_value = 0 if allow_zero else 1

    # Booleans are integers in Python, so they are rejected explicitly
    for key_path in key_paths:
        value = get_configuration_value(configuration=configuration, key_path=key_path)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum_value:
            errors.append(f"'{key_path}' must be an integer >= {minimum_value}, got {value!r}.")

    return errors


def check_probability_value(value: Any, key_path: str) -> List[str]:
    """
    Verifies that a probability read from the configuration (e.g. of applying a data augmentation) is a number
    within [0, 1].

    Args:
        value (Any): The probability read from the configuration.
        key_path (str): The dotted key path of the value, used in the error message.

    Returns:
        List[str]: A single error description if the value is invalid, otherwise an empty list.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        return [f"'{key_path}' must be a probability within [0, 1], got {value!r}."]
    return []


def check_intersection_over_union_value(value: Any, key_path: str) -> List[str]:
    """
    Verifies that an intersection over union threshold is a number within [0, 1], the only range an IoU can take
    during anchor assignment, proposal assignment and non-maximum suppression.

    Args:
        value (Any): The threshold value read from the configuration.
        key_path (str): The dotted key path of the value, used in the error message.

    Returns:
        List[str]: A single error description if the value is invalid, otherwise an empty list.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        return [f"'{key_path}' must be a number within [0, 1], got {value!r}."]
    return []
