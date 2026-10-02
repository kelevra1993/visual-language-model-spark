from typing import Any, Dict, List

from vision_training.configuration.classification_checks import (check_classification_data_configuration,
                                                                 check_classification_evaluation_configuration,
                                                                 check_classification_head_configuration)
from vision_training.configuration.configuration_values import (check_boolean_flags, check_positive_integers,
                                                                get_configuration_value)
from vision_training.models.sequence.temporal_modules import ORDERED_TEMPORAL_TYPES, POOLING_TEMPORAL_TYPES

# Handling of the sequences shorter than one window: padded with black frames, or left out
VALID_SHORT_SEQUENCES = ["pad", "skip"]


def check_sequence_settings(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'Data.sequence_settings' section of the sequence tasks, which cuts the sequences into the windows of
    frames fed to the models, and that the classification head predicts one class per sequence or per frame (the
    sequence manifests give a single category per sequence or per frame).

    Args:
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    # Frames per window, frames skipped between two kept frames, and frames between the starts of two windows
    errors = check_positive_integers(configuration=model_configuration,
                                     key_paths=["Data.sequence_settings.sequence_length",
                                                "Data.sequence_settings.frame_stride",
                                                "Data.sequence_settings.window_stride"], allow_zero=False)
    short_sequences = get_configuration_value(configuration=model_configuration,
                                              key_path="Data.sequence_settings.short_sequences")
    if short_sequences not in VALID_SHORT_SEQUENCES:
        errors.append(f"'Data.sequence_settings.short_sequences' must be one of {VALID_SHORT_SEQUENCES}, "
                      f"got {short_sequences!r}.")

    # One class per sequence or per frame
    output_mode = get_configuration_value(configuration=model_configuration, key_path="ClassificationHead.output_mode")
    if output_mode != "multi_class":
        errors.append(f"'ClassificationHead.output_mode' must be 'multi_class' for the sequence tasks, the manifests "
                      f"giving one category per sequence or per frame, got {output_mode!r}.")
    return errors


def check_temporal_model_configuration(model_configuration: Dict[str, Any], valid_types: List[str],
                                       allow_causal: bool) -> List[str]:
    """
    Verifies the 'TemporalModel' section of the sequence tasks: the module type allowed by the task, its sizes, its
    dropout, and the consistency of the causal and bidirectional flags (a causal model never looks at the future
    frames, so it cannot read the sequence backwards).

    Args:
        model_configuration (Dict[str, Any]): The processed model sections.
        valid_types (List[str]): The temporal module types the task can use.
        allow_causal (bool): Whether the task can use causal modules (many to many only).

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    temporal_type = get_configuration_value(configuration=model_configuration, key_path="TemporalModel.type")
    if temporal_type not in valid_types:
        errors.append(f"'TemporalModel.type' must be one of {valid_types} for this task, got {temporal_type!r}.")

    # Sizes of the modules, all required even when the type does not use them
    size_errors = check_positive_integers(configuration=model_configuration,
                                          key_paths=["TemporalModel.hidden_size", "TemporalModel.number_layers",
                                                     "TemporalModel.number_heads", "TemporalModel.kernel_size"],
                                          allow_zero=False)
    errors += size_errors
    dropout_rate = get_configuration_value(configuration=model_configuration, key_path="TemporalModel.dropout_rate")
    if isinstance(dropout_rate, bool) or not isinstance(dropout_rate, (int, float)) or not 0.0 <= dropout_rate < 1.0:
        errors.append(f"'TemporalModel.dropout_rate' must be a number within [0, 1), got {dropout_rate!r}.")

    # The transformer splits its model size between its heads, the temporal convolution centres its kernels
    hidden_size = get_configuration_value(configuration=model_configuration, key_path="TemporalModel.hidden_size")
    number_heads = get_configuration_value(configuration=model_configuration, key_path="TemporalModel.number_heads")
    kernel_size = get_configuration_value(configuration=model_configuration, key_path="TemporalModel.kernel_size")
    if not size_errors and temporal_type == "transformer" and hidden_size % number_heads != 0:
        errors.append(f"'TemporalModel.hidden_size' ({hidden_size}) must be a multiple of "
                      f"'TemporalModel.number_heads' ({number_heads}) for the transformer.")
    if not size_errors and temporal_type == "temporal_convolution" and kernel_size % 2 == 0:
        errors.append(f"'TemporalModel.kernel_size' must be odd for the temporal convolution, got {kernel_size}.")

    # Causal modules only see the past frames, which excludes reading the sequence backwards
    flag_errors = check_boolean_flags(configuration=model_configuration,
                                      key_paths=["TemporalModel.bidirectional", "TemporalModel.causal"])
    errors += flag_errors
    causal = get_configuration_value(configuration=model_configuration, key_path="TemporalModel.causal")
    bidirectional = get_configuration_value(configuration=model_configuration, key_path="TemporalModel.bidirectional")
    if not flag_errors and causal and not allow_causal:
        errors.append("'TemporalModel.causal' must be False for the sequence classification task, which classifies a "
                      "whole sequence (causal modules are for the sequence labelling task).")
    if not flag_errors and causal and bidirectional:
        errors.append("'TemporalModel.bidirectional' must be False when 'TemporalModel.causal' is True: reading the "
                      "sequence backwards looks at the future frames.")
    return errors


def check_sequence_classification_configuration(experiment_configuration: Dict[str, Any],
                                                model_configuration: Dict[str, Any]) -> List[str]:
    """
    Runs every check specific to the sequence classification task (many to one): the data settings, classification
    head and evaluation settings shared with the classification task, the sequence settings, and a temporal module
    summarising a sequence (any type, not causal).

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    errors += check_classification_data_configuration(model_configuration=model_configuration)
    errors += check_classification_head_configuration(model_configuration=model_configuration)
    errors += check_classification_evaluation_configuration(experiment_configuration=experiment_configuration)
    errors += check_sequence_settings(model_configuration=model_configuration)
    errors += check_temporal_model_configuration(model_configuration=model_configuration,
                                                 valid_types=POOLING_TEMPORAL_TYPES + ORDERED_TEMPORAL_TYPES,
                                                 allow_causal=False)
    return errors
