"""
Checks of the configuration sections specific to the sequence tasks: the sequence settings, the temporal module, the
fusion of the pairs and the selection settings, for sequence classification (M2O), sequence labelling (M2M), sequence
comparison (CMP) and frame selection (SEL). Run by configuration_checker.py for these tasks, on top of the
classification head and data checks.
"""

from typing import Any, Dict, List

from vision_training.configuration_processing.classification_checks import (
    check_classification_data_configuration, check_classification_evaluation_configuration,
    check_classification_head_configuration)
from vision_training.configuration_processing.common_checks import check_pretrained_configuration
from vision_training.configuration_processing.configuration_values import (check_boolean_flags, check_positive_integers,
                                                                           check_probability_value,
                                                                           get_configuration_value)
from vision_training.models.sequence.frame_selection_decoding import VALID_DECODINGS
from vision_training.models.sequence.member_fusion import FUSION_TYPES, SEQUENCE_NORMALIZATION_SCOPES
from vision_training.models.sequence.temporal_modules import ORDERED_TEMPORAL_TYPES, POOLING_TEMPORAL_TYPES


def check_sequence_settings(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'Data.sequence_settings' section of the sequence tasks, which gives the largest number of frames of a
    sequence (or of a member of a pair) fed whole to the models, the longer ones being left out of the splits, the
    sequence augmentation settings held by the sequence tasks only (the photometric types drawn frame by frame and
    the shuffling of the frames), and that the classification head predicts one class per sequence, per frame or per
    pair (the manifests give a single category per sequence, per frame or per pair).

    Args:
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    # Largest number of frames of a sequence or member, also the size of the positional embedding of the transformer
    errors = check_positive_integers(configuration=model_configuration,
                                     key_paths=["Data.sequence_settings.maximum_sequence_length"], allow_zero=False)

    # Sequence augmentations: the photometric types drawn frame by frame, and the shuffling of the frames of a sequence
    errors += check_boolean_flags(configuration=model_configuration,
                                  key_paths=["Data.data_augmentation.independent_frames",
                                             "Data.data_augmentation.shuffle_frames.activated"])
    errors += check_probability_value(
        value=get_configuration_value(configuration=model_configuration,
                                      key_path="Data.data_augmentation.shuffle_frames.probability"),
        key_path="Data.data_augmentation.shuffle_frames.probability")

    # One class per sequence, per frame or per pair
    output_mode = get_configuration_value(configuration=model_configuration, key_path="ClassificationHead.output_mode")
    if output_mode != "multi_class":
        errors.append(f"'ClassificationHead.output_mode' must be 'multi_class' for the sequence tasks, the manifests "
                      f"giving one category per sequence, per frame or per pair, got {output_mode!r}.")
    return errors


def check_temporal_model_configuration(model_configuration: Dict[str, Any], valid_types: List[str],
                                       allow_causal: bool) -> List[str]:
    """
    Verifies the 'TemporalModel' section of the sequence tasks: the module type allowed by the task, its sizes, its
    dropout, the consistency of the causal and bidirectional flags (a causal model never looks at the future frames,
    so it cannot read the sequence backwards), the positional embedding flag of the transformer, and the 'pretrained'
    settings initialising the module with the one of another run, which the poolings without weights cannot use.

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

    # Causal modules only see the past frames, which excludes reading the sequence backwards. The positional
    # embedding flag is required for every type, like the sizes, and only read by the transformer
    flag_errors = check_boolean_flags(configuration=model_configuration,
                                      key_paths=["TemporalModel.bidirectional", "TemporalModel.causal",
                                                 "TemporalModel.positional_embedding"])
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
    errors += check_classification_head_configuration(experiment_configuration=experiment_configuration,
                                                      model_configuration=model_configuration)
    errors += check_classification_evaluation_configuration(experiment_configuration=experiment_configuration,
                                                            model_configuration=model_configuration)
    errors += check_sequence_settings(model_configuration=model_configuration)
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
    errors += check_classification_head_configuration(experiment_configuration=experiment_configuration,
                                                      model_configuration=model_configuration)
    errors += check_classification_evaluation_configuration(experiment_configuration=experiment_configuration,
                                                            model_configuration=model_configuration)
    errors += check_sequence_settings(model_configuration=model_configuration)
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

    # The cross attention splits the frame vectors between its heads: requiring the channels of the last backbone
    # block to be a multiple of the heads covers the flattened maps too, whose width is a multiple of these channels
    # (a malformed block being reported by the backbone check)
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
    settings shared with the classification task, the sequence settings (the largest number of frames of a member),
    a temporal module summarising a member (any type, not causal), and the fusion of the
    members.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    errors += check_classification_data_configuration(model_configuration=model_configuration)
    errors += check_classification_head_configuration(experiment_configuration=experiment_configuration,
                                                      model_configuration=model_configuration)
    errors += check_classification_evaluation_configuration(experiment_configuration=experiment_configuration,
                                                            model_configuration=model_configuration)
    errors += check_sequence_settings(model_configuration=model_configuration)
    errors += check_temporal_model_configuration(model_configuration=model_configuration,
                                                 valid_types=POOLING_TEMPORAL_TYPES + ORDERED_TEMPORAL_TYPES,
                                                 allow_causal=False)
    errors += check_fusion_configuration(model_configuration=model_configuration)
    return errors


def check_selection_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'Selection' section of the frame selection task: the name of the manifest category of the unselected
    frames (its presence in the manifests being checked by the dataset), the decoding of the selection, the number of
    frames chosen for every wanted class and the weight of the selection loss.

    Args:
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    unselected_category = get_configuration_value(configuration=model_configuration,
                                                  key_path="Selection.unselected_category")
    if not isinstance(unselected_category, str) or not unselected_category.strip():
        errors.append(f"'Selection.unselected_category' must be the non-empty name of a manifest category, got "
                      f"{unselected_category!r}.")
    decoding = get_configuration_value(configuration=model_configuration, key_path="Selection.decoding")
    if decoding not in VALID_DECODINGS:
        errors.append(f"'Selection.decoding' must be one of {VALID_DECODINGS}, got {decoding!r}.")
    errors += check_positive_integers(configuration=model_configuration, key_paths=["Selection.frames_per_class"],
                                      allow_zero=False)

    # Weight of the selection loss, 0 training with the per-frame loss alone
    loss_weight = get_configuration_value(configuration=model_configuration, key_path="Selection.selection_loss_weight")
    if isinstance(loss_weight, bool) or not isinstance(loss_weight, (int, float)) or loss_weight < 0:
        errors.append(f"'Selection.selection_loss_weight' must be a number >= 0, got {loss_weight!r}.")
    return errors


def check_frame_selection_configuration(experiment_configuration: Dict[str, Any],
                                        model_configuration: Dict[str, Any]) -> List[str]:
    """
    Runs every check specific to the frame selection task: the data settings, classification head and evaluation
    settings shared with the classification task (with a single target precision, the deployment keeping one threshold
    per class), the number of candidate frames of every class given by its deployment ('evaluation.top_k', also the k
    of its top-k selection accuracy), the sequence settings (the largest number of frames of a sequence), a temporal
    module keeping one vector per frame, possibly causal, and the selection settings.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    errors += check_classification_data_configuration(model_configuration=model_configuration)
    errors += check_classification_head_configuration(experiment_configuration=experiment_configuration,
                                                      model_configuration=model_configuration)
    errors += check_classification_evaluation_configuration(experiment_configuration=experiment_configuration,
                                                            model_configuration=model_configuration,
                                                            allow_several_target_precisions=False)
    errors += check_positive_integers(configuration={"ExperimentConfiguration": experiment_configuration},
                                      key_paths=["ExperimentConfiguration.evaluation.top_k"], allow_zero=False)
    errors += check_sequence_settings(model_configuration=model_configuration)
    errors += check_temporal_model_configuration(model_configuration=model_configuration,
                                                 valid_types=ORDERED_TEMPORAL_TYPES, allow_causal=True)
    errors += check_selection_configuration(model_configuration=model_configuration)
    return errors
