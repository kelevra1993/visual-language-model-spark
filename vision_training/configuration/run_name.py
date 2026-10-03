from typing import Any, Callable, Dict, List


# Run name code of every feature map normalization strategy
NORMALIZATION_CODES = {"batch": "BN", "layer": "LN", "none": "NN"}

# Run name code of every task, first section of the run names, so that runs of different tasks never share a folder
TASK_CODES = {"object_detection": "DET", "classification": "CLS", "sequence_classification": "M2O",
              "sequence_labelling": "M2M", "sequence_comparison": "CMP"}

# Run name code of every temporal module type of the sequence tasks
TEMPORAL_TYPE_CODES = {"mean_pooling": "MEAN", "max_pooling": "MAX", "attention_pooling": "ATT", "gru": "GRU",
                       "lstm": "LSTM", "transformer": "TRF", "temporal_convolution": "TCN"}

# Run name code of every fusion of the members of a pair (sequence comparison task)
FUSION_TYPE_CODES = {"concatenation": "FCAT", "difference": "FDIF", "cross_attention": "FXA"}

# Prefixes of the tokens of a component initialised by the one of another run: '<prefix>PT' when it is transferred,
# followed by '<prefix>FRZ' when it is frozen or '<learning rate prefix>LR<multiplier>' when it learns at its own
# speed, i.e. PT, FRZ and BLR for the backbone, TPT, TFRZ and TLR for the temporal module, HPT, HFRZ and HLR for the
# classification head
PRETRAINED_TOKEN_PREFIXES = {"Backbone": ("", "B"), "TemporalModel": ("T", "T"), "ClassificationHead": ("H", "H")}


def join_level_tokens(level_tokens: List[str]) -> str:
    """
    Joins the per-level descriptions of a head type (region proposal heads, detector heads, anchor counts) for the
    run name: written once when every detection level uses the same settings, otherwise listed in level order and
    joined with '+'.

    Args:
        level_tokens (List[str]): The description of every detection level, in level order.

    Returns:
        str: The joined description.
    """
    return level_tokens[0] if len(set(level_tokens)) == 1 else "+".join(level_tokens)


def build_input_tokens(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Builds the input section of the run name, shared by every task: IN (input normalization), PSN (per-sample
    normalization), KAR (keep aspect ratio), then the input size.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration.

    Returns:
        List[str]: The tokens of the section.
    """
    backbone_normalization = model_configuration["Backbone"]["normalization"]
    image_settings = model_configuration["Data"]["image_settings"]

    # Normalization of the pixel values, resizing mode and input size
    input_tokens = []
    if backbone_normalization["input_normalization"]:
        input_tokens.append("IN")
    if backbone_normalization["per_sample_normalization"]:
        input_tokens.append("PSN")
    if image_settings["keep_ratio"]:
        input_tokens.append("KAR")
    input_tokens.append(str(image_settings["size"]))
    return input_tokens


def build_pretrained_tokens(model_configuration: Dict[str, Any], section_name: str) -> List[str]:
    """
    Builds the tokens of a component initialised by the trained one of another run, placed at the end of the section
    of the component in the run name (see PRETRAINED_TOKEN_PREFIXES): e.g. PT, then FRZ when it is frozen or
    BLR<learning rate multiplier> when it learns at another speed than the new layers (a frozen component does not
    learn at all), then FBN when the scale and shift of the normalization layers of the backbone are kept fixed. A
    component trained from random weights gives no token.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration.
        section_name (str): The section holding the 'pretrained' settings, a key of PRETRAINED_TOKEN_PREFIXES.

    Returns:
        List[str]: The tokens of the transfer, empty without transfer.
    """
    pretrained_configuration = model_configuration[section_name]["pretrained"]
    if pretrained_configuration["experiment_folder"] is None:
        return []

    # Transferred component, frozen or learning at its own speed
    token_prefix, learning_rate_prefix = PRETRAINED_TOKEN_PREFIXES[section_name]
    pretrained_tokens = [f"{token_prefix}PT"]
    if pretrained_configuration["freeze"]:
        pretrained_tokens.append(f"{token_prefix}FRZ")
    elif pretrained_configuration["learning_rate_multiplier"] != 1.0:
        pretrained_tokens.append(f"{learning_rate_prefix}LR{pretrained_configuration['learning_rate_multiplier']:g}")

    # Normalization parameters kept fixed in a backbone that still learns
    if section_name == "Backbone" and pretrained_configuration["freeze_normalization_parameters"]:
        pretrained_tokens.append("FBN")
    return pretrained_tokens


def build_backbone_tokens(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Builds the backbone section of the run name, shared by every task: the modules (SE, RES, INC<starting block>),
    the feature map normalization (BN, LN or NN), B<layers>x<filters> of every block in order, then PT when the
    backbone is initialised by the one of another run, followed by BLR<learning rate multiplier> when it learns at
    another speed than the heads or FRZ when it is frozen, and FBN when its normalization parameters are kept fixed
    (see build_pretrained_tokens).

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration.

    Returns:
        List[str]: The tokens of the section.
    """
    backbone_configuration = model_configuration["Backbone"]
    backbone_modules = backbone_configuration["modules"]

    # Modules and feature map normalization of the convolution blocks
    backbone_tokens = []
    if backbone_modules["squeeze_and_excite"]:
        backbone_tokens.append("SE")
    if backbone_modules["residual_nets"]:
        backbone_tokens.append("RES")
    if backbone_modules["inception"]:
        backbone_tokens.append(f"INC{backbone_modules['inception_starting_block']}")
    backbone_tokens.append(NORMALIZATION_CODES[
        str(backbone_configuration["normalization"]["feature_map_normalization"]).lower().strip()])

    # Layers and filters of every block, in order
    convolutions = backbone_configuration["convolutions"]
    backbone_tokens.append("B" + ".".join(f"{convolutions[block][0]}x{convolutions[block][1]}"
                                          for block in sorted(convolutions, key=int)))

    # Backbone transferred from another run, frozen or learning at its own speed
    backbone_tokens += build_pretrained_tokens(model_configuration=model_configuration, section_name="Backbone")
    return backbone_tokens


def format_scientific(value: float) -> str:
    """
    Writes a number in the short scientific notation of the run names, without trailing zeros, e.g. 2e-4 or 1.5e-3.

    Args:
        value (float): The number.

    Returns:
        str: The short scientific notation.
    """
    mantissa, exponent = f"{value:e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"

def build_training_tokens(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> List[str]:
    """
    Builds the training section of the run name, shared by every task: AUG when augmenting, followed by SHUF when the
    frames of the windows are shuffled and IND when the photometric augmentations are drawn frame by frame, the batch
    size and the learning rate in short scientific notation (e.g. 2e-4, 1.5e-3), then the optimization settings that
    differ from plain Adam at a constant learning rate: ADAMW or SGD, WD<weight decay>, WU<warm-up iterations>,
    COS<decay iterations> for the cosine schedule, GC<gradient clipping norm> and BF16 for mixed precision.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        model_configuration (Dict[str, Any]): The processed model configuration.

    Returns:
        List[str]: The tokens of the section.
    """
    augmentation_configuration = model_configuration["Data"]["data_augmentation"]
    training_tokens = ["AUG"] if augmentation_configuration["activated"] else []

    # Sequence augmentations, only applied when augmenting: shuffled frames, and photometric draws per frame
    if augmentation_configuration["activated"] and augmentation_configuration["shuffle_frames"]["activated"]:
        training_tokens.append("SHUF")
    if augmentation_configuration["activated"] and augmentation_configuration["independent_frames"]:
        training_tokens.append("IND")
    training_tokens.append(f"BS{experiment_configuration['batch_size']}")
    training_tokens.append(f"LR{format_scientific(value=experiment_configuration['learning_rate'])}")

    # Optimization settings, written only when they differ from plain Adam at a constant learning rate
    optimization_configuration = model_configuration["Optimization"]
    if optimization_configuration["optimizer"] != "adam":
        training_tokens.append(optimization_configuration["optimizer"].upper())
    if optimization_configuration["weight_decay"] > 0:
        training_tokens.append(f"WD{format_scientific(value=optimization_configuration['weight_decay'])}")
    if optimization_configuration["warmup_iterations"] > 0:
        training_tokens.append(f"WU{optimization_configuration['warmup_iterations']}")
    if optimization_configuration["learning_rate_schedule"] == "cosine":
        training_tokens.append(f"COS{optimization_configuration['decay_iterations']}")
    if optimization_configuration["gradient_clipping_norm"] is not None:
        training_tokens.append(f"GC{optimization_configuration['gradient_clipping_norm']:g}")
    if optimization_configuration["mixed_precision"]:
        training_tokens.append("BF16")
    return training_tokens



def build_detection_sections(model_configuration: Dict[str, Any]) -> List[List[str]]:
    """
    Builds the sections of the run name specific to object detection: L<blocks feeding the detection heads>, the
    anchors A<anchors per position>.<smallest scale>-<largest scale>, the region proposal heads (R<normalization>,
    R<layers>x<filters>), the detector heads (modules DSE, DRES, DINC, D<normalization>, D<RoIAlign pool size>
    .<layers>x<filters>.<fully connected sizes>) and the classes (BIN or MC<foreground classes>). Per-level settings
    are written once when identical on every level, otherwise joined with '+' in level order.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration, with its label dictionary.

    Returns:
        List[List[str]]: The tokens of every section.
    """
    data_configuration = model_configuration["Data"]
    region_proposal_configuration = model_configuration["RegionProposal"]
    detector_configuration = model_configuration["Detector"]
    level_names = [str(index) for index in model_configuration["Backbone"]["detection_convolution_block_indices"]]

    # Detection levels, and the number of anchors per position (scales x ratios) with the range of anchor scales
    levels_token = "L" + "".join(level_names)
    scales_and_ratios = model_configuration["Anchors"]["scales_and_ratios"]
    anchor_counts = [str(len(scales_and_ratios[level]["scales"]) * len(scales_and_ratios[level]["aspect_ratios"]))
                     for level in level_names]
    anchor_scales = [scale for level in level_names for scale in scales_and_ratios[level]["scales"]]
    anchors_token = f"A{join_level_tokens(level_tokens=anchor_counts)}.{min(anchor_scales)}-{max(anchor_scales)}"

    # Region proposal heads: normalization, then the layers and filters of the head of every level
    region_proposal_heads = region_proposal_configuration["head_configurations"]
    region_proposal_tokens = [
        "R" + NORMALIZATION_CODES[
            str(region_proposal_configuration["normalization"]["feature_map_normalization"]).lower().strip()],
        "R" + join_level_tokens(level_tokens=[f"{region_proposal_heads[level][0]}x{region_proposal_heads[level][1]}"
                                              for level in level_names])]

    # Detector heads: modules, normalization, then the RoIAlign pool size, convolutions and fully connected layers
    detector_modules = detector_configuration["modules"]
    detector_tokens = []
    if detector_modules["squeeze_and_excite"]:
        detector_tokens.append("DSE")
    if detector_modules["residual_nets"]:
        detector_tokens.append("DRES")
    if detector_modules["inception"]:
        detector_tokens.append("DINC")
    detector_tokens.append("D" + NORMALIZATION_CODES[
        str(detector_configuration["normalization"]["feature_map_normalization"]).lower().strip()])
    detector_heads = detector_configuration["head_configurations"]
    detector_tokens.append("D" + join_level_tokens(level_tokens=[
        f"{detector_heads[level]['roi_align_pool_size']}.{detector_heads[level]['convolutions'][0]}x"
        f"{detector_heads[level]['convolutions'][1]}." + ".".join(str(size) for size in
                                                                   detector_heads[level]["fully_connected"])
        for level in level_names]))

    # Classes, the label dictionary including the background class
    if data_configuration["label_mode"] == "binary":
        classes_token = "BIN"
    else:
        classes_token = f"MC{len(data_configuration['label_dictionary']) - 1}"

    return [[levels_token], [anchors_token], region_proposal_tokens, detector_tokens, [classes_token]]


def build_classification_sections(model_configuration: Dict[str, Any]) -> List[List[str]]:
    """
    Builds the sections of the run name describing the classification head, shared by the classification-like tasks:
    the head H<global pooling AVG or MAX>.<fully connected sizes> with the dropout DO<rate>, LS<label smoothing> when
    smoothing, CWB when the classes are balanced in the loss and HPT, HFRZ or HLR<multiplier> when the head is
    initialised by the one of another run (see build_pretrained_tokens), then the classes MC<classes> (one class per
    sample) or ML<classes> (several classes per sample), e.g. 'HAVG.256-DO0.5_MC4'.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration, with its label dictionary.

    Returns:
        List[List[str]]: The tokens of every section.
    """
    head_configuration = model_configuration["ClassificationHead"]

    # Pooling and hidden layers, dropout, then the label smoothing and class weighting of the loss when used
    pooling_code = {"average": "AVG", "max": "MAX"}[head_configuration["global_pooling"]]
    head_tokens = ["H" + ".".join([pooling_code] + [str(size) for size in head_configuration["fully_connected"]]),
                   f"DO{head_configuration['dropout_rate']:g}"]
    if head_configuration["label_smoothing"] > 0:
        head_tokens.append(f"LS{head_configuration['label_smoothing']:g}")
    if head_configuration["class_weighting"] == "balanced":
        head_tokens.append("CWB")
    head_tokens += build_pretrained_tokens(model_configuration=model_configuration, section_name="ClassificationHead")

    # Classes, without background class
    output_code = "MC" if head_configuration["output_mode"] == "multi_class" else "ML"
    classes_token = f"{output_code}{len(set(model_configuration['Data']['label_dictionary'].values()))}"

    return [head_tokens, [classes_token]]


def build_window_tokens(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Builds the token of the run name describing the windows of frames fed to the sequence models: T<frames per window>
    (frames per member of a pair for the comparison task), followed by S<frame stride> when frames are skipped, e.g.
    'T16S2'.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration.

    Returns:
        List[str]: The window token.
    """
    sequence_settings = model_configuration["Data"]["sequence_settings"]
    sequence_token = f"T{sequence_settings['sequence_length']}"
    if sequence_settings["frame_stride"] > 1:
        sequence_token += f"S{sequence_settings['frame_stride']}"
    return [sequence_token]


def build_temporal_tokens(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Builds the tokens of the run name describing the temporal module of the sequence models: MEAN, MAX, ATT<hidden
    size>, GRU or LSTM<layers>x<hidden size> with BI when bidirectional, TRF<layers>x<hidden size>.<heads> or
    TCN<layers>x<hidden size>.<kernel size> with CAU when causal, TDO<dropout> when the module uses a dropout, and
    TPT, TFRZ or TLR<multiplier> when the module is initialised by the one of another run (see
    build_pretrained_tokens), e.g. 'GRU2x256-BI'.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration.

    Returns:
        List[str]: The tokens of the temporal module.
    """
    temporal_configuration = model_configuration["TemporalModel"]
    temporal_type = temporal_configuration["type"]

    # Temporal module with the settings its type uses
    temporal_code = TEMPORAL_TYPE_CODES[temporal_type]
    layers_and_size = f"{temporal_configuration['number_layers']}x{temporal_configuration['hidden_size']}"
    if temporal_type in ["mean_pooling", "max_pooling"]:
        temporal_tokens = [temporal_code]
    elif temporal_type == "attention_pooling":
        temporal_tokens = [f"{temporal_code}{temporal_configuration['hidden_size']}"]
    elif temporal_type in ["gru", "lstm"]:
        temporal_tokens = [temporal_code + layers_and_size] + (
            ["BI"] if temporal_configuration["bidirectional"] else [])
    else:
        type_parameter = temporal_configuration["number_heads" if temporal_type == "transformer" else "kernel_size"]
        temporal_tokens = [f"{temporal_code}{layers_and_size}.{type_parameter}"] + (
            ["CAU"] if temporal_configuration["causal"] else [])
    if temporal_type not in ["mean_pooling", "max_pooling", "attention_pooling"] and \
            temporal_configuration["dropout_rate"] > 0:
        temporal_tokens.append(f"TDO{temporal_configuration['dropout_rate']:g}")

    # Temporal module transferred from another run, frozen or learning at its own speed
    temporal_tokens += build_pretrained_tokens(model_configuration=model_configuration, section_name="TemporalModel")
    return temporal_tokens


def build_sequence_sections(model_configuration: Dict[str, Any]) -> List[List[str]]:
    """
    Builds the sections of the run name specific to the sequence tasks: the window (build_window_tokens), the temporal
    module (build_temporal_tokens), then the classification head and classes sections, e.g.
    'T16_GRU2x256-BI_HAVG.256-DO0.5_MC4'.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration, with its label dictionary.

    Returns:
        List[List[str]]: The tokens of every section.
    """
    return ([build_window_tokens(model_configuration=model_configuration),
             build_temporal_tokens(model_configuration=model_configuration)] +
            build_classification_sections(model_configuration=model_configuration))


def build_fusion_tokens(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Builds the tokens of the run name describing the fusion of the members of a pair in the sequence comparison model:
    FCAT (concatenation), FDIF (difference) or FXA<heads> (cross attention), the latter followed by FDO<dropout> when
    its attention uses a dropout, e.g. 'FXA4-FDO0.1'.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration.

    Returns:
        List[str]: The tokens of the fusion.
    """
    fusion_configuration = model_configuration["Fusion"]
    fusion_type = fusion_configuration["type"]
    if fusion_type != "cross_attention":
        return [FUSION_TYPE_CODES[fusion_type]]

    # Cross attention with its number of heads, and its dropout when used
    fusion_tokens = [f"{FUSION_TYPE_CODES[fusion_type]}{fusion_configuration['number_heads']}"]
    if fusion_configuration["dropout_rate"] > 0:
        fusion_tokens.append(f"FDO{fusion_configuration['dropout_rate']:g}")
    return fusion_tokens


def build_sequence_comparison_sections(model_configuration: Dict[str, Any]) -> List[List[str]]:
    """
    Builds the sections of the run name specific to the sequence comparison task: the member windows
    (build_window_tokens), the temporal module (build_temporal_tokens), the fusion of the members
    (build_fusion_tokens), then the classification head and classes sections, e.g. 'T8_MEAN_FDIF_HAVG.64-DO0.2_MC2'.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration, with its label dictionary.

    Returns:
        List[List[str]]: The tokens of every section.
    """
    return ([build_window_tokens(model_configuration=model_configuration),
             build_temporal_tokens(model_configuration=model_configuration),
             build_fusion_tokens(model_configuration=model_configuration)] +
            build_classification_sections(model_configuration=model_configuration))


# Builder of the task-specific sections of the run name, placed between the backbone and training sections
TASK_RUN_NAME_SECTIONS: Dict[str, Callable[[Dict[str, Any]], List[List[str]]]] = {
    "object_detection": build_detection_sections, "classification": build_classification_sections,
    "sequence_classification": build_sequence_sections, "sequence_labelling": build_sequence_sections,
    "sequence_comparison": build_sequence_comparison_sections}


def build_run_name(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> str:
    """
    Builds the name of the folder a training run is saved in, from the settings that define the run, so that a run
    always trains in a folder matching its configuration: the same settings resume in the same folder, and changed
    settings, in particular every setting changing the model size, start a new one instead of mixing with a folder
    whose saved configuration no longer matches.

    The name is made of '_' separated sections of '-' separated tokens, a flag token being left out when its setting
    is off, e.g. 'DET_IN-KAR-1024_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512_RBN-R2x64_
    DSE-DRES-DBN-D7.2x64.256.128_BIN_AUG-BS40-LR2e-4' (without the line break): the task code (DET, CLS, M2O, M2M or
    CMP), the input and backbone sections shared by every task (build_input_tokens, build_backbone_tokens), the
    sections of the task (e.g. build_detection_sections), then the training section (build_training_tokens).

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration, with its task.
        model_configuration (Dict[str, Any]): The processed model configuration, with its label dictionary.

    Returns:
        str: The run name.
    """
    task = experiment_configuration["task"]
    sections = ([[TASK_CODES[task]], build_input_tokens(model_configuration=model_configuration),
                 build_backbone_tokens(model_configuration=model_configuration)] +
                TASK_RUN_NAME_SECTIONS[task](model_configuration=model_configuration) +
                [build_training_tokens(experiment_configuration=experiment_configuration,
                                       model_configuration=model_configuration)])
    return "_".join("-".join(section) for section in sections)
