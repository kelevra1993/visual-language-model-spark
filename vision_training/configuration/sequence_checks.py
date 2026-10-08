"""
Checks of the configuration sections specific to the sequence tasks: the window settings, the temporal module and
the fusion of the pairs, for sequence classification (M2O), sequence labelling (M2M) and sequence comparison (CMP).
Run by configuration_checker.py for these tasks, on top of the classification head and data checks.
"""

from typing import Any, Dict, List

from vision_training.configuration.classification_checks import (check_classification_data_configuration,
                                                                 check_classification_evaluation_configuration,
                                                                 check_classification_head_configuration)
from vision_training.configuration.common_checks import check_pretrained_configuration
from vision_training.configuration.configuration_values import (check_boolean_flags, check_positive_integers,
                                                                get_configuration_value)
from vision_training.models.sequence.member_fusion import FUSION_TYPES, SEQUENCE_NORMALIZATION_SCOPES
from vision_training.models.sequence.temporal_modules import ORDERED_TEMPORAL_TYPES, POOLING_TEMPORAL_TYPES

# Handling of the sequences shorter than one window: padded with black frames, or left out
VALID_SHORT_SEQUENCES = ["pad", "skip"]

# Number of frames the windows of a batch are padded to: the sequence length, or the longest window of the batch
VALID_BATCH_PADDINGS = ["sequence_length", "longest"]


def check_sequence_settings(model_configuration: Dict[str, Any], cuts_windows: bool) -> List[str]:
    """
    Verifies the 'Data.sequence_settings' section of the sequence tasks, which gives the windows of frames fed to the
    models (cut from the sequences, or one window per member of a pair) and the number of frames their batches are
    padded to, that the classification head predicts one class per sequence, per frame or per pair (the manifests
    give a single category per sequence, per frame or per pair), and that every frame is pooled to one vector rather
    than flattened.

    Args:
        model_configuration (Dict[str, Any]): The processed model sections.
        cuts_windows (bool): Whether the task cuts the sequences into windows (sequence classification and labelling),
            which needs 'window_stride'. The comparison task feeds one window per member and does not read it.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    # Frames per window, and frames between the starts of two windows when the sequences are cut into windows
    key_paths = ["Data.sequence_settings.sequence_length"]
    if cuts_windows:
        key_paths.append("Data.sequence_settings.window_stride")
    errors = check_positive_integers(configuration=model_configuration, key_paths=key_paths, allow_zero=False)
    short_sequences = get_configuration_value(configuration=model_configuration,
                                              key_path="Data.sequence_settings.short_sequences")
    if short_sequences not in VALID_SHORT_SEQUENCES:
        errors.append(f"'Data.sequence_settings.short_sequences' must be one of {VALID_SHORT_SEQUENCES}, "
                      f"got {short_sequences!r}.")

    # Windows of a batch padded to the sequence length, or to the longest window of the batch
    batch_padding = get_configuration_value(configuration=model_configuration,
                                            key_path="Data.sequence_settings.batch_padding")
    if batch_padding not in VALID_BATCH_PADDINGS:
        errors.append(f"'Data.sequence_settings.batch_padding' must be one of {VALID_BATCH_PADDINGS}, "
                      f"got {batch_padding!r}.")

    # One class per sequence, per frame or per pair
    output_mode = get_configuration_value(configuration=model_configuration, key_path="ClassificationHead.output_mode")
    if output_mode != "multi_class":
        errors.append(f"'ClassificationHead.output_mode' must be 'multi_class' for the sequence tasks, the manifests "
                      f"giving one category per sequence, per frame or per pair, got {output_mode!r}.")

    # Frames pooled to vectors as wide as the last backbone block, which the temporal module and the fusion read
    global_pooling = get_configuration_value(configuration=model_configuration,
                                             key_path="ClassificationHead.global_pooling")
    if global_pooling == "flatten":
        errors.append("'ClassificationHead.global_pooling' cannot be 'flatten' for the sequence tasks: the temporal "
                      "module and the fusion read frame vectors as wide as the last backbone block, use 'average' or "
                      "'max'.")
    return errors


def check_temporal_model_configuration(model_configuration: Dict[str, Any], valid_types: List[str],
                                       allow_causal: bool) -> List[str]:
    """
    Verifies the 'TemporalModel' section of the sequence tasks: the module type allowed by the task, its sizes, its
    dropout, the consistency of the causal and bidirectional flags (a causal model never looks at the future frames,
    so it cannot read the sequence backwards), and the 'pretrained' settings initialising the module with the one of
    another run, which the poolings without weights cannot use.

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
        errors.append("'TemporalModel.causal' must be False for this task, which summarises whole sequences (causal "
                      "modules are for the sequence labelling task).")
    if not flag_errors and causal and bidirectional:
        errors.append("'TemporalModel.bidirectional' must be False when 'TemporalModel.causal' is True: reading the "
                      "sequence backwards looks at the future frames.")

    # Module possibly initialised by the one of another run, which needs weights to transfer
    errors += check_pretrained_configuration(model_configuration=model_configuration, section_name="TemporalModel")
    pretrained_experiment_folder = get_configuration_value(configuration=model_configuration,
                                                           key_path="TemporalModel.pretrained.experiment_folder")
    if temporal_type in ["mean_pooling", "max_pooling"] and pretrained_experiment_folder is not None:
        errors.append(f"'TemporalModel.pretrained.experiment_folder' must be null for the {temporal_type!r} temporal "
                      f"module, which has no weights to transfer.")
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
    errors += check_classification_evaluation_configuration(experiment_configuration=experiment_configuration,
                                                            model_configuration=model_configuration)
    errors += check_sequence_settings(model_configuration=model_configuration, cuts_windows=True)
    errors += check_temporal_model_configuration(model_configuration=model_configuration,
                                                 valid_types=POOLING_TEMPORAL_TYPES + ORDERED_TEMPORAL_TYPES,
                                                 allow_causal=False)
    return errors


def check_sequence_labelling_configuration(experiment_configuration: Dict[str, Any],
                                           model_configuration: Dict[str, Any]) -> List[str]:
    """
    Runs every check specific to the sequence labelling task (many to many): the data settings, classification head
    and evaluation settings shared with the classification task, the sequence settings, and a temporal module keeping
    one vector per frame (the order-blind poolings summarise a whole sequence), possibly causal.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    errors += check_classification_data_configuration(model_configuration=model_configuration)
    errors += check_classification_head_configuration(model_configuration=model_configuration)
    errors += check_classification_evaluation_configuration(experiment_configuration=experiment_configuration,
                                                            model_configuration=model_configuration)
    errors += check_sequence_settings(model_configuration=model_configuration, cuts_windows=True)
    errors += check_temporal_model_configuration(model_configuration=model_configuration,
                                                 valid_types=ORDERED_TEMPORAL_TYPES, allow_causal=True)
    return errors


def check_fusion_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'Fusion' section of the sequence comparison task, which combines the before and after members of a
    pair: its type, the number of heads and the dropout of the cross attention (all required whatever the type), that
    the cross attention can split the frame vectors between its heads, and the frames sharing the statistics of the
    sequence normalization (required whatever the normalization of the backbone).

    Args:
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    fusion_type = get_configuration_value(configuration=model_configuration, key_path="Fusion.type")
    if fusion_type not in FUSION_TYPES:
        errors.append(f"'Fusion.type' must be one of {FUSION_TYPES}, got {fusion_type!r}.")

    # Heads and dropout of the cross attention
    head_errors = check_positive_integers(configuration=model_configuration, key_paths=["Fusion.number_heads"],
                                          allow_zero=False)
    errors += head_errors
    dropout_rate = get_configuration_value(configuration=model_configuration, key_path="Fusion.dropout_rate")
    if isinstance(dropout_rate, bool) or not isinstance(dropout_rate, (int, float)) or not 0.0 <= dropout_rate < 1.0:
        errors.append(f"'Fusion.dropout_rate' must be a number within [0, 1), got {dropout_rate!r}.")
    normalization_scope = get_configuration_value(configuration=model_configuration,
                                                  key_path="Fusion.sequence_normalization_scope")
    if normalization_scope not in SEQUENCE_NORMALIZATION_SCOPES:
        errors.append(f"'Fusion.sequence_normalization_scope' must be one of {SEQUENCE_NORMALIZATION_SCOPES}, "
                      f"got {normalization_scope!r}.")

    # The cross attention splits the frame vectors, as wide as the last backbone block, between its heads (a
    # malformed block being reported by the backbone check)
    convolutions = get_configuration_value(configuration=model_configuration, key_path="Backbone.convolutions")
    if fusion_type == "cross_attention" and not head_errors and isinstance(convolutions, dict) and convolutions:
        last_block = list(convolutions.values())[-1]
        number_heads = get_configuration_value(configuration=model_configuration, key_path="Fusion.number_heads")
        if isinstance(last_block, list) and len(last_block) == 2 and isinstance(last_block[1], int) and \
                not isinstance(last_block[1], bool) and last_block[1] % number_heads != 0:
            errors.append(f"The channels of the last backbone block ({last_block[1]}) must be a multiple of "
                          f"'Fusion.number_heads' ({number_heads}) for the cross attention.")
    return errors


def check_sequence_comparison_configuration(experiment_configuration: Dict[str, Any],
                                            model_configuration: Dict[str, Any]) -> List[str]:
    """
    Runs every check specific to the sequence comparison task: the data settings, classification head and evaluation
    settings shared with the classification task, the sequence settings of the member windows (no window stride, a
    member being one window), a temporal module summarising a member (any type, not causal), and the fusion of the
    members.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    errors += check_classification_data_configuration(model_configuration=model_configuration)
    errors += check_classification_head_configuration(model_configuration=model_configuration)
    errors += check_classification_evaluation_configuration(experiment_configuration=experiment_configuration,
                                                            model_configuration=model_configuration)
    errors += check_sequence_settings(model_configuration=model_configuration, cuts_windows=False)
    errors += check_temporal_model_configuration(model_configuration=model_configuration,
                                                 valid_types=POOLING_TEMPORAL_TYPES + ORDERED_TEMPORAL_TYPES,
                                                 allow_causal=False)
    errors += check_fusion_configuration(model_configuration=model_configuration)
    return errors
