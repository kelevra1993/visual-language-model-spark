"""
Building of the run names: the name of the folder a training run is saved in, made of short tokens describing the
settings of the run (task, input, backbone, task and optimizer sections), followed by the suffix chosen in the
configuration (ExperimentConfiguration.run_name_suffix). Called by the base trainer to create or resume a run folder.
"""

import hashlib
import json

from typing import Any, Callable, Dict, List


# Run name code of every feature map normalization strategy, also giving the HBN and HLN tokens of the normalized
# hidden layers of the classification head
NORMALIZATION_CODES = {"batch": "BN", "layer": "LN", "none": "NN", "sequence": "SN"}

# Run name code of every task, first section of the run names, so that runs of different tasks never share a folder
TASK_CODES = {"object_detection": "DET", "instance_segmentation": "INS", "classification": "CLS",
              "sequence_classification": "M2O", "sequence_labelling": "M2M", "sequence_comparison": "CMP",
              "semantic_segmentation": "SEG", "frame_selection": "SEL", "self_supervised": "SSL",
              "keypoint_detection": "KPT"}

# Run name code of every temporal module type of the sequence tasks
TEMPORAL_TYPE_CODES = {"mean_pooling": "MEAN", "max_pooling": "MAX", "attention_pooling": "ATT", "gru": "GRU",
                       "lstm": "LSTM", "transformer": "TRF", "temporal_convolution": "TCN"}

# Run name code of every fusion of the members of a pair (sequence comparison task)
FUSION_TYPE_CODES = {"concatenation": "FCAT", "difference": "FDIF", "cross_attention": "FXA"}

# Run name code of every reduction of the final feature map before the classification head: its average or maximum,
# or all its values flattened
GLOBAL_POOLING_CODES = {"average": "AVG", "max": "MAX", "flatten": "FLAT"}

# Run name code of every method of the self-supervised task, followed by its setting: the temperature of SimCLR
# (T<temperature>) or the off-diagonal weight of Barlow Twins (OD<weight>)
SELF_SUPERVISED_METHOD_CODES = {"simclr": "SIMCLR", "barlow_twins": "BT"}

# Run name code of every upsampling of the segmentation decoder levels and of the mask heads (prefixed with M): a
# transposed convolution, or a bilinear upsampling followed by a convolution
UPSAMPLING_CODES = {"transposed": "UTC", "bilinear": "UBL"}

# Prefix of the tokens of a component initialised by the one of another run: '<prefix>PT' when it is transferred,
# followed by '<prefix>FRZ' when it is frozen, i.e. PT and FRZ for the backbone, TPT and TFRZ for the temporal module,
# HPT and HFRZ for the classification head, DPT and DFRZ for the segmentation decoder, RPT and RFRZ for the region
# proposal heads, DPT and DFRZ for the detection heads, MPT and MFRZ for the mask heads, HRPT and HRFRZ for the high
# resolution network of the keypoint detector, and KHPT and KHFRZ for its keypoint head. The learning rate multiplier
# of the component gives no token, so that it can change between two resumptions of a run
PRETRAINED_TOKEN_PREFIXES = {"Backbone": "", "TemporalModel": "T", "ClassificationHead": "H",
                             "SegmentationDecoder": "D", "RegionProposal": "R", "Detector": "D", "MaskHead": "M",
                             "HRNet": "HR", "KeypointHead": "KH"}


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
    normalization), KAR (keep aspect ratio, padded to a square) or SQC (keep aspect ratio, cut to a square, the square
    crop of the classification and self-supervised tasks), then the input size.

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
        input_tokens.append("SQC" if image_settings["square_crop"] else "KAR")
    input_tokens.append(str(image_settings["size"]))
    return input_tokens


def build_pretrained_tokens(model_configuration: Dict[str, Any], section_name: str) -> List[str]:
    """
    Builds the tokens of a component initialised by the trained one of another run, placed at the end of the section of
    the component in the run name (see PRETRAINED_TOKEN_PREFIXES): e.g. PT, then DCL when the segmentation decoder also
    loads the class layers of the source run, then FRZ when it is frozen, then FBN when the scale and shift of the
    normalization layers of the backbone are kept fixed. A component trained from random weights gives no token. The
    learning rate multiplier gives no token, the optimizer giving every transferred component that learns its own
    parameter group whatever its multiplier, so that it can change between two resumptions of a run.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration.
        section_name (str): The section holding the 'pretrained' settings, a key of PRETRAINED_TOKEN_PREFIXES.

    Returns:
        List[str]: The tokens of the transfer, empty without transfer.
    """
    pretrained_configuration = model_configuration[section_name]["pretrained"]
    if pretrained_configuration["experiment_folder"] is None:
        return []

    # Transferred component, frozen or not
    token_prefix = PRETRAINED_TOKEN_PREFIXES[section_name]
    pretrained_tokens = [f"{token_prefix}PT"]

    # Segmentation head and deep supervision heads loaded with the decoder, which changes what the run starts from
    if section_name == "SegmentationDecoder" and pretrained_configuration["load_class_layers"]:
        pretrained_tokens.append("DCL")
    if pretrained_configuration["freeze"]:
        pretrained_tokens.append(f"{token_prefix}FRZ")

    # Normalization parameters kept fixed in a backbone that still learns
    if section_name == "Backbone" and pretrained_configuration["freeze_normalization_parameters"]:
        pretrained_tokens.append("FBN")
    return pretrained_tokens


def build_backbone_tokens(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Builds the backbone section of the run name, shared by every task: the modules (SE, RES or RES2 for the
    pre-activation shortcuts, INC<starting block>), the feature map normalization (BN, LN or NN), B<layers>x<filters>
    of every block in order, LMP when a max pooling follows the last block ('last_max_pooling', which changes the
    size of the final feature map and of a flattening head), then PT when the backbone is initialised by the one of
    another run, followed by FRZ when it is frozen and FBN when its normalization parameters are kept fixed (see
    build_pretrained_tokens).

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
        backbone_tokens.append("RES2" if backbone_modules["residual_nets_v2"] else "RES")
    if backbone_modules["inception"]:
        backbone_tokens.append(f"INC{backbone_modules['inception_starting_block']}")
    backbone_tokens.append(NORMALIZATION_CODES[
        str(backbone_configuration["normalization"]["feature_map_normalization"]).lower().strip()])

    # Layers and filters of every block, in order
    convolutions = backbone_configuration["convolutions"]
    backbone_tokens.append("B" + ".".join(f"{convolutions[block][0]}x{convolutions[block][1]}"
                                          for block in sorted(convolutions, key=int)))

    # Max pooling after the last block, halving the final feature map
    if backbone_configuration["last_max_pooling"]:
        backbone_tokens.append("LMP")

    # Backbone transferred from another run, frozen or not
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

def build_optimizer_tokens(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Builds the optimizer section of the run name, shared by every task, left out with plain Adam without weight decay:
    ADAMW or SGD, whose state cannot be resumed by another optimizer, then WD<weight decay>. The other training settings
    (augmentation, batch size, learning rate and its schedule, learning rate multipliers of the transferred
    components, gradient clipping, mixed precision, batch normalization momentums) give no token, so that a run can be
    resumed in its folder with changed values.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration.

    Returns:
        List[str]: The tokens of the section, empty with plain Adam without weight decay.
    """
    optimization_configuration = model_configuration["Optimization"]

    # Optimizer and weight decay, written only when they differ from plain Adam without weight decay
    optimizer_tokens = []
    if optimization_configuration["optimizer"] != "adam":
        optimizer_tokens.append(optimization_configuration["optimizer"].upper())
    if optimization_configuration["weight_decay"] > 0:
        optimizer_tokens.append(f"WD{format_scientific(value=optimization_configuration['weight_decay'])}")
    return optimizer_tokens


def build_anchors_hash(detection_levels: Dict[str, Any], level_names: List[str]) -> str:
    """
    Writes the anchors of every detection level as a short hash for the anchor token of the run names: the count and
    the range of the scales alone keep the name when the aspect ratios or the middle scales change, and a run would
    then resume heads trained on other anchors. The scales and ratios are taken in their order, which is the order of
    the anchors in the outputs of the heads, as floats so that 32 and 32.0 give the same hash.

    Args:
        detection_levels (Dict[str, Any]): The 'DetectionLevels' section, with the 'anchors' of every level.
        level_names (List[str]): The detection levels, in level order.

    Returns:
        str: The first 6 hexadecimal characters of the SHA-256 of the anchors of every level.
    """
    anchor_settings = [[level_name, [float(scale) for scale in detection_levels[level_name]["anchors"]["scales"]],
                        [float(ratio) for ratio in detection_levels[level_name]["anchors"]["aspect_ratios"]]]
                       for level_name in level_names]
    return hashlib.sha256(json.dumps(obj=anchor_settings).encode()).hexdigest()[:6]


def build_detection_sections(model_configuration: Dict[str, Any]) -> List[List[str]]:
    """
    Builds the sections of the run name specific to object detection: L<blocks feeding the detection heads>, the
    anchors A<anchors per position>.<smallest scale>-<largest scale>.<hash of every scale and aspect ratio of every
    level> (build_anchors_hash), so that other ratios or middle scales start a new run, the region proposal heads
    (R<normalization>, R<layers>x<filters>, then RPT and RFRZ when they are transferred and frozen), the detector heads
    (modules DSE, DRES, DINC, D<normalization>, D<RoIAlign pool size>.<layers>x<filters>.<fully connected sizes>, then
    DPT and DFRZ when they are transferred and frozen) and the classes (BIN or MC<foreground classes>, the categories
    merged into one class counting once). The levels are the keys of 'DetectionLevels', whose anchor counts, region
    proposal and detector settings are written once when identical on every level, otherwise joined with '+' in level
    order.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration, with its label dictionary.

    Returns:
        List[List[str]]: The tokens of every section.
    """
    data_configuration = model_configuration["Data"]
    region_proposal_configuration = model_configuration["RegionProposal"]
    detector_configuration = model_configuration["Detector"]
    detection_levels = model_configuration["DetectionLevels"]
    level_names = sorted(detection_levels, key=int)

    # Detection levels, and the number of anchors per position (scales x ratios) with the range of anchor scales
    levels_token = "L" + "".join(level_names)
    anchor_counts = [str(len(detection_levels[level]["anchors"]["scales"]) *
                         len(detection_levels[level]["anchors"]["aspect_ratios"])) for level in level_names]
    anchor_scales = [scale for level in level_names for scale in detection_levels[level]["anchors"]["scales"]]
    anchors_hash = build_anchors_hash(detection_levels=detection_levels, level_names=level_names)
    anchors_token = (f"A{join_level_tokens(level_tokens=anchor_counts)}.{min(anchor_scales)}-{max(anchor_scales)}."
                     f"{anchors_hash}")

    # Region proposal heads: normalization, then the layers and filters of the head of every level
    region_proposal_heads = {level: detection_levels[level]["region_proposal"]["convolutions"] for level in level_names}
    region_proposal_tokens = [
        "R" + NORMALIZATION_CODES[
            str(region_proposal_configuration["normalization"]["feature_map_normalization"]).lower().strip()],
        "R" + join_level_tokens(level_tokens=[f"{region_proposal_heads[level][0]}x{region_proposal_heads[level][1]}"
                                              for level in level_names])]
    region_proposal_tokens += build_pretrained_tokens(model_configuration=model_configuration,
                                                      section_name="RegionProposal")

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
    detector_heads = {level: detection_levels[level]["detector"] for level in level_names}
    detector_tokens.append("D" + join_level_tokens(level_tokens=[
        f"{detector_heads[level]['roi_align_pool_size']}.{detector_heads[level]['convolutions'][0]}x"
        f"{detector_heads[level]['convolutions'][1]}." + ".".join(str(size) for size in
                                                                   detector_heads[level]["fully_connected"])
        for level in level_names]))
    detector_tokens += build_pretrained_tokens(model_configuration=model_configuration, section_name="Detector")

    # Classes, the label dictionary including the background class
    if data_configuration["label_mode"] == "binary":
        classes_token = "BIN"
    else:
        classes_token = f"MC{len(set(data_configuration['label_dictionary'].values())) - 1}"

    return [[levels_token], [anchors_token], region_proposal_tokens, detector_tokens, [classes_token]]


def build_instance_segmentation_sections(model_configuration: Dict[str, Any]) -> List[List[str]]:
    """
    Builds the sections of the run name specific to instance segmentation: the sections of object detection
    (build_detection_sections), with the mask heads section before the classes: the modules MSE, MRES, MINC, the
    normalization MBN, MLN or MNN, the upsampling MUTC or MUBL, then M<RoIAlign pool size>.<layers>x<channels> of the
    mask head of every level of 'DetectionLevels', written once when identical on every level, otherwise joined with
    '+' in level order, MR<mask resolution>, and MPT, followed by MFRZ when frozen, when the mask heads are initialised
    by those of another run (see build_pretrained_tokens), e.g. 'MRES-MBN-MUTC-M14.4x64-MR28'. The mask threshold gives
    no token, so that it can change between two resumptions of a run.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration, with its label dictionary.

    Returns:
        List[List[str]]: The tokens of every section.
    """
    mask_head_configuration = model_configuration["MaskHead"]
    detection_levels = model_configuration["DetectionLevels"]
    level_names = sorted(detection_levels, key=int)

    # Modules and normalization of the mask head convolution blocks
    mask_modules = mask_head_configuration["modules"]
    mask_tokens = []
    if mask_modules["squeeze_and_excite"]:
        mask_tokens.append("MSE")
    if mask_modules["residual_nets"]:
        mask_tokens.append("MRES")
    if mask_modules["inception"]:
        mask_tokens.append("MINC")
    mask_tokens.append("M" + NORMALIZATION_CODES[
        str(mask_head_configuration["normalization"]["feature_map_normalization"]).lower().strip()])

    # Upsampling, then the RoIAlign pool size, layers and channels of the head of every level, then the mask resolution
    mask_tokens.append("M" + UPSAMPLING_CODES[mask_head_configuration["upsampling"]])
    mask_heads = {level: detection_levels[level]["mask_head"] for level in level_names}
    mask_tokens.append("M" + join_level_tokens(level_tokens=[
        f"{mask_heads[level]['roi_align_pool_size']}.{mask_heads[level]['convolutions'][0]}x"
        f"{mask_heads[level]['convolutions'][1]}" for level in level_names]))
    mask_tokens.append(f"MR{mask_head_configuration['mask_resolution']}")
    mask_tokens += build_pretrained_tokens(model_configuration=model_configuration, section_name="MaskHead")

    # The mask heads section comes before the classes section of the detection sections
    detection_sections = build_detection_sections(model_configuration=model_configuration)
    return detection_sections[:-1] + [mask_tokens] + detection_sections[-1:]


def build_classification_sections(model_configuration: Dict[str, Any]) -> List[List[str]]:
    """
    Builds the sections of the run name describing the classification head, shared by the classification-like tasks:
    the head H<global pooling AVG, MAX or FLAT>.<fully connected sizes>, HBN or HLN when its hidden layers are batch
    or layer normalized, LS<label smoothing> when smoothing, CWB when the classes are balanced in the loss and HPT,
    followed by HFRZ when frozen, when the head is initialised by the one of another run (see build_pretrained_tokens),
    then the classes MC<classes> (one class per sample) or ML<classes> (several classes per sample), e.g.
    'HAVG.256-LS0.1_MC4', 'HFLAT.256-HBN_MC4', 'HFLAT.256-HLN_MC4' or 'HFLAT.256_MC4'. The dropout and the batch
    normalization momentum give no token, so that they can change between two resumptions of a run.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration, with its label dictionary.

    Returns:
        List[List[str]]: The tokens of every section.
    """
    head_configuration = model_configuration["ClassificationHead"]

    # Pooling and hidden layers with their normalization, then the label smoothing and class weighting of the loss
    # when used
    pooling_code = GLOBAL_POOLING_CODES[head_configuration["global_pooling"]]
    head_tokens = ["H" + ".".join([pooling_code] + [str(size) for size in head_configuration["fully_connected"]])]
    if head_configuration["fully_connected_normalization"] != "none":
        head_tokens.append("H" + NORMALIZATION_CODES[head_configuration["fully_connected_normalization"]])
    if head_configuration["label_smoothing"] > 0:
        head_tokens.append(f"LS{head_configuration['label_smoothing']:g}")
    if head_configuration["class_weighting"] == "balanced":
        head_tokens.append("CWB")
    head_tokens += build_pretrained_tokens(model_configuration=model_configuration, section_name="ClassificationHead")

    # Classes, without background class
    output_code = "MC" if head_configuration["output_mode"] == "multi_class" else "ML"
    classes_token = f"{output_code}{len(set(model_configuration['Data']['label_dictionary'].values()))}"

    return [head_tokens, [classes_token]]


def build_sequence_length_tokens(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Builds the token of the run name giving the largest number of frames of the sequences fed whole to the sequence
    models, the longer ones being left out: T<maximum sequence length> (of a member of a pair for the comparison
    task), e.g. 'T50'.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration.

    Returns:
        List[str]: The maximum sequence length token.
    """
    return [f"T{model_configuration['Data']['sequence_settings']['maximum_sequence_length']}"]


def build_temporal_tokens(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Builds the tokens of the run name describing the temporal module of the sequence models: MEAN, MAX, ATT<hidden
    size>, GRU or LSTM<layers>x<hidden size> with BI when bidirectional, TRF<layers>x<hidden size>.<heads> or
    TCN<layers>x<hidden size>.<kernel size> with CAU when causal, NPE for a transformer without positional embedding,
    and TPT, followed by TFRZ when frozen, when the module is initialised by the one of another run (see
    build_pretrained_tokens), e.g. 'GRU2x256-BI'. The dropout gives no token, so that it can change between two
    resumptions of a run.

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

        # Transformer treating a sequence as a set of frames, whose weights hold no positional embedding
        if temporal_type == "transformer" and not temporal_configuration["positional_embedding"]:
            temporal_tokens.append("NPE")

    # Temporal module transferred from another run, frozen or not
    temporal_tokens += build_pretrained_tokens(model_configuration=model_configuration, section_name="TemporalModel")
    return temporal_tokens


def build_sequence_sections(model_configuration: Dict[str, Any]) -> List[List[str]]:
    """
    Builds the sections of the run name specific to the sequence tasks: the maximum sequence length
    (build_sequence_length_tokens), the temporal module (build_temporal_tokens), then the classification head and
    classes sections, e.g. 'T50_GRU2x256-BI_HAVG.256_MC4'.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration, with its label dictionary.

    Returns:
        List[List[str]]: The tokens of every section.
    """
    return ([build_sequence_length_tokens(model_configuration=model_configuration),
             build_temporal_tokens(model_configuration=model_configuration)] +
            build_classification_sections(model_configuration=model_configuration))


def build_frame_selection_sections(model_configuration: Dict[str, Any]) -> List[List[str]]:
    """
    Builds the sections of the run name specific to the frame selection task: the sections of the sequence tasks
    (build_sequence_sections), then the selection section FPC<frames per class>, followed by SLW<weight> when the
    selection loss is used, e.g. 'T40_GRU1x128_HAVG.64_MC4_FPC1-SLW1'. The decoding gives no token: both decodings
    are scored at every evaluation, and it only chooses the one drawn in the images and deployed.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration, with its label dictionary.

    Returns:
        List[List[str]]: The tokens of every section.
    """
    selection_configuration = model_configuration["Selection"]
    selection_tokens = [f"FPC{selection_configuration['frames_per_class']}"]
    if selection_configuration["selection_loss_weight"] > 0:
        selection_tokens.append(f"SLW{selection_configuration['selection_loss_weight']:g}")
    return build_sequence_sections(model_configuration=model_configuration) + [selection_tokens]


def build_fusion_tokens(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Builds the tokens of the run name describing the fusion of the members of a pair in the sequence comparison model:
    FCAT (concatenation), FDIF (difference) or FXA<heads> (cross attention), then SNP when both members share the
    statistics of the sequence normalization of the backbone, e.g. 'FXA4' or 'FDIF-SNP'. The dropout of the cross
    attention gives no token, so that it can change between two resumptions of a run.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration.

    Returns:
        List[str]: The tokens of the fusion.
    """
    fusion_configuration = model_configuration["Fusion"]
    fusion_type = fusion_configuration["type"]

    # Cross attention with its number of heads, or the type of the other fusions
    if fusion_type == "cross_attention":
        fusion_tokens = [f"{FUSION_TYPE_CODES[fusion_type]}{fusion_configuration['number_heads']}"]
    else:
        fusion_tokens = [FUSION_TYPE_CODES[fusion_type]]

    # Statistics of the sequence normalization shared by both members of a pair
    if model_configuration["Backbone"]["normalization"]["feature_map_normalization"] == "sequence" and \
            fusion_configuration["sequence_normalization_scope"] == "pair":
        fusion_tokens.append("SNP")
    return fusion_tokens


def build_sequence_comparison_sections(model_configuration: Dict[str, Any]) -> List[List[str]]:
    """
    Builds the sections of the run name specific to the sequence comparison task: the maximum member length
    (build_sequence_length_tokens), the temporal module (build_temporal_tokens), the fusion of the members
    (build_fusion_tokens), then the classification head and classes sections, e.g. 'T50_MEAN_FDIF_HAVG.64_MC2'.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration, with its label dictionary.

    Returns:
        List[List[str]]: The tokens of every section.
    """
    return ([build_sequence_length_tokens(model_configuration=model_configuration),
             build_temporal_tokens(model_configuration=model_configuration),
             build_fusion_tokens(model_configuration=model_configuration)] +
            build_classification_sections(model_configuration=model_configuration))


def build_segmentation_sections(model_configuration: Dict[str, Any]) -> List[List[str]]:
    """
    Builds the sections of the run name specific to the semantic segmentation task: TL<scale> when the run trains on
    tiles of the images scaled by <scale>, the decoder (UTC or UBL for the transposed convolution or bilinear
    upsampling, the modules DSE and DRES, the normalization DBN, DLN or DNN, then D<layers>x<channels> of every level
    from the deepest one to the input resolution, AUX<levels> for the deep supervision heads of these levels, and DPT,
    followed by DCL when the class layers are loaded too and DFRZ when frozen, when the decoder is initialised by the
    one of another run, see build_pretrained_tokens),
    then the classes (BIN or MC<foreground classes>), e.g. 'UTC-DBN-D2x64.2x64.2x32.2x32.1x16-AUX5.4_MC4'. The dropout
    of the segmentation head, the weights of the deep supervision losses, the overlap of the evaluation tiles and the
    'SegmentationLoss' settings give no token, so that they can change between two resumptions of a run.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration, with its label dictionary.

    Returns:
        List[List[str]]: The tokens of every section.
    """
    data_configuration = model_configuration["Data"]
    decoder_configuration = model_configuration["SegmentationDecoder"]

    # Upsampling, modules and normalization of the decoder blocks
    decoder_tokens = [UPSAMPLING_CODES[decoder_configuration["upsampling"]]]
    if decoder_configuration["modules"]["squeeze_and_excite"]:
        decoder_tokens.append("DSE")
    if decoder_configuration["modules"]["residual_nets"]:
        decoder_tokens.append("DRES")
    decoder_tokens.append("D" + NORMALIZATION_CODES[decoder_configuration["feature_map_normalization"]])

    # Layers and channels of every level, from the deepest one to the input resolution, then the transfer
    convolutions = decoder_configuration["convolutions"]
    decoder_tokens.append("D" + ".".join(f"{convolutions[level][0]}x{convolutions[level][1]}"
                                         for level in sorted(convolutions, key=int, reverse=True)))

    # Levels holding a deep supervision head, whose weights are part of the model, from the deepest one
    deep_supervision_weights = decoder_configuration["deep_supervision_weights"]
    if deep_supervision_weights:
        decoder_tokens.append("AUX" + ".".join(sorted(deep_supervision_weights, key=int, reverse=True)))
    decoder_tokens += build_pretrained_tokens(model_configuration=model_configuration,
                                              section_name="SegmentationDecoder")

    # Classes, the label dictionary including the background class
    if data_configuration["label_mode"] == "binary":
        classes_token = "BIN"
    else:
        classes_token = f"MC{len(set(data_configuration['label_dictionary'].values())) - 1}"

    # Training on tiles of the scaled images, which changes what the network learns from
    tiling_configuration = data_configuration["tiling"]
    tiling_sections = [[f"TL{tiling_configuration['scale']:g}"]] if tiling_configuration["activated"] else []
    return tiling_sections + [decoder_tokens, [classes_token]]


def build_self_supervised_sections(model_configuration: Dict[str, Any]) -> List[List[str]]:
    """
    Builds the sections of the run name specific to the self-supervised task: the projection head
    P<global pooling AVG, MAX or FLAT>.<layer sizes, the embedding size last>, PBN or PLN when its hidden layers are
    batch or layer normalized, then the method with its setting, SIMCLR-T<temperature> or BT-OD<off-diagonal weight>,
    e.g. 'PAVG.2048.128-PBN_SIMCLR-T0.1' or 'PAVG.2048.2048.2048-PBN_BT-OD0.005'. The data augmentation drawing the
    views and the batch normalization momentum give no token, so that they can change between two resumptions of a
    run.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration.

    Returns:
        List[List[str]]: The tokens of every section.
    """
    head_configuration = model_configuration["ProjectionHead"]

    # Pooling and layers of the projection head with the normalization of its hidden layers
    pooling_code = GLOBAL_POOLING_CODES[head_configuration["global_pooling"]]
    head_tokens = ["P" + ".".join([pooling_code] + [str(size) for size in head_configuration["fully_connected"]])]
    if head_configuration["fully_connected_normalization"] != "none":
        head_tokens.append("P" + NORMALIZATION_CODES[head_configuration["fully_connected_normalization"]])

    # Method with the setting changing its loss
    method = head_configuration["method"]
    if method == "simclr":
        method_tokens = [SELF_SUPERVISED_METHOD_CODES[method], f"T{head_configuration['simclr']['temperature']:g}"]
    else:
        method_tokens = [SELF_SUPERVISED_METHOD_CODES[method],
                         f"OD{head_configuration['barlow_twins']['off_diagonal_weight']:g}"]
    return [head_tokens, method_tokens]


def build_keypoint_sections(model_configuration: Dict[str, Any]) -> List[List[str]]:
    """
    Builds the sections of the run name specific to the keypoint detection: the high resolution network (modules HRSE
    and HRRES, HR<normalization>, then HR<base width>.<modules of every stage>x<blocks per branch>, then HRPT and
    HRFRZ when it is transferred and frozen), the keypoint head K<heatmap channels> (the keypoints of every class, each
    with its heatmap and its tag map, then KH<hidden width> with a hidden convolution, HH with the higher resolution
    branch, then KHPT and KHFRZ when it is transferred and frozen) and the classes MC<foreground classes>, e.g.
    'HRSE-HRRES-HRBN-HR32.1.4.3x4_K107-KH256-HH_MC22'.
    The heatmap sigma, the tag loss weights and the grouping settings give no token, so that they can change between
    two resumptions of a run.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration, with its label dictionary and the
            keypoint categories read from the training split.

    Returns:
        List[List[str]]: The tokens of every section.
    """
    data_configuration = model_configuration["Data"]
    network_configuration = model_configuration["HRNet"]
    network_modules = network_configuration["modules"]

    # High resolution network: modules, normalization, then the base width, modules of every stage and blocks per branch
    network_tokens = []
    if network_modules["squeeze_and_excite"]:
        network_tokens.append("HRSE")
    if network_modules["residual_nets"]:
        network_tokens.append("HRRES")
    network_tokens.append("HR" + NORMALIZATION_CODES[
        str(network_configuration["normalization"]["feature_map_normalization"]).lower().strip()])
    network_tokens.append(f"HR{network_configuration['base_width']}." +
                          ".".join(str(modules) for modules in network_configuration["modules_per_stage"]) +
                          f"x{network_configuration['blocks_per_branch']}")
    network_tokens += build_pretrained_tokens(model_configuration=model_configuration, section_name="HRNet")

    # Keypoint head: one heatmap channel per keypoint of every class, then the hidden convolution and the higher
    # resolution branch
    keypoint_categories = data_configuration["keypoint_categories"]
    heatmap_channels = sum(len(category["keypoints"]) for category in keypoint_categories.values())
    head_tokens = [f"K{heatmap_channels}"]
    if model_configuration["KeypointHead"]["hidden_width"] > 0:
        head_tokens.append(f"KH{model_configuration['KeypointHead']['hidden_width']}")
    if model_configuration["KeypointHead"]["higher_resolution"]:
        head_tokens.append("HH")
    head_tokens += build_pretrained_tokens(model_configuration=model_configuration, section_name="KeypointHead")

    # Classes, the label dictionary including the background class
    classes_token = f"MC{len(set(data_configuration['label_dictionary'].values())) - 1}"
    return [network_tokens, head_tokens, [classes_token]]


# Builder of the task-specific sections of the run name, placed between the backbone and training sections
TASK_RUN_NAME_SECTIONS: Dict[str, Callable[[Dict[str, Any]], List[List[str]]]] = {
    "object_detection": build_detection_sections, "instance_segmentation": build_instance_segmentation_sections,
    "classification": build_classification_sections,
    "sequence_classification": build_sequence_sections, "sequence_labelling": build_sequence_sections,
    "sequence_comparison": build_sequence_comparison_sections, "semantic_segmentation": build_segmentation_sections,
    "frame_selection": build_frame_selection_sections, "self_supervised": build_self_supervised_sections,
    "keypoint_detection": build_keypoint_sections}


def build_run_name(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> str:
    """
    Builds the name of the folder a training run is saved in, from the settings that define the run, so that a run
    always trains in a folder matching its configuration: the same settings resume in the same folder, and changed
    settings, in particular every setting changing the model size, start a new one instead of mixing with a folder
    whose saved configuration no longer matches. The training settings that can change between two resumptions of a
    run (augmentation, batch size, learning rate and its schedule, learning rate multipliers of the transferred
    components, gradient clipping, mixed precision, dropout rates, batch normalization momentums) give no token.
    Neither do the source run of a transferred component (only whether it is transferred, a resumed run never loading
    it again) nor the content of the label mapping file (only the number of classes): a run resumed with another
    mapping of as many classes keeps the label dictionary saved in its folder
    (BaseTrainer.restore_run_label_dictionary).

    The name is made of '_' separated sections of '-' separated tokens, a flag token being left out when its setting
    is off, e.g. 'DET_IN-KAR-1024_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512.598295_RBN-R2x64_
    DSE-DRES-DBN-D7.2x64.256.128_BIN' (without the line break): the task code (DET, INS, CLS, M2O, M2M, CMP, SEG,
    SEL, SSL or KPT), the input and backbone sections shared by every task (build_input_tokens,
    build_backbone_tokens), the sections of the task (e.g. build_detection_sections), then the optimizer section
    (build_optimizer_tokens), left out when empty, and last the suffix of 'ExperimentConfiguration.run_name_suffix'
    when it is not null (e.g. '_v1'), which gives a run of the same settings with another batch size, augmentation or
    learning rate a folder of its own instead of resuming the run without suffix.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration, with its task and run
            name suffix.
        model_configuration (Dict[str, Any]): The processed model configuration, with its label dictionary.

    Returns:
        str: The run name.
    """
    task = experiment_configuration["task"]
    sections = ([[TASK_CODES[task]], build_input_tokens(model_configuration=model_configuration),
                 build_backbone_tokens(model_configuration=model_configuration)] +
                TASK_RUN_NAME_SECTIONS[task](model_configuration=model_configuration))

    # Optimizer section, only written when the run does not use plain Adam without weight decay
    optimizer_tokens = build_optimizer_tokens(model_configuration=model_configuration)
    if optimizer_tokens:
        sections.append(optimizer_tokens)

    # Suffix chosen in the configuration, last section of the name, left out when null
    run_name_suffix = experiment_configuration["run_name_suffix"]
    if run_name_suffix is not None:
        sections.append([run_name_suffix])
    return "_".join("-".join(section) for section in sections)
