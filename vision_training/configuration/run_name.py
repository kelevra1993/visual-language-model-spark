from typing import Any, Dict, List


# Run name code of every feature map normalization strategy
NORMALIZATION_CODES = {"batch": "BN", "layer": "LN", "none": "NN"}


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


def build_run_name(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> str:
    """
    Builds the name of the folder a training run is saved in, from the settings that define the run, so that a run
    always trains in a folder matching its configuration: the same settings resume in the same folder, and changed
    settings, in particular every setting changing the model size, start a new one instead of mixing with a folder
    whose saved configuration no longer matches.

    The name is made of '_' separated sections of '-' separated tokens, a flag token being left out when its setting
    is off, e.g. 'IN-KAR-1024_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512_RBN-R2x64_
    DSE-DRES-DBN-D7.2x64.256.128_BIN_AUG-BS40-LR2e-4' (without the line break):
    - input: IN (input normalization), PSN (per-sample normalization), KAR (keep aspect ratio), the input size;
    - backbone: modules (SE, RES, INC<starting block>), feature map normalization (BN, LN or NN), then
      B<layers>x<filters> of every block;
    - L<blocks feeding the detection heads>;
    - anchors: A<anchors per position>.<smallest scale>-<largest scale>;
    - region proposal heads: R<normalization>, then R<layers>x<filters>;
    - detector heads: modules (DSE, DRES, DINC), D<normalization>, then D<RoIAlign pool size>.<layers>x<filters>
      .<fully connected sizes>;
    - classes: BIN or MC<foreground classes>;
    - training: AUG, BS<batch size>, LR<learning rate>.
    Per-level settings are written once when identical on every level, otherwise joined with '+' in level order.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        model_configuration (Dict[str, Any]): The processed model configuration, with its label dictionary.

    Returns:
        str: The run name.
    """
    data_configuration = model_configuration["Data"]
    backbone_configuration = model_configuration["Backbone"]
    region_proposal_configuration = model_configuration["RegionProposal"]
    detector_configuration = model_configuration["Detector"]
    level_names = [str(index) for index in backbone_configuration["detection_convolution_block_indices"]]

    # Input: normalization of the pixel values, resizing mode and input size
    input_tokens = []
    if backbone_configuration["normalization"]["input_normalization"]:
        input_tokens.append("IN")
    if backbone_configuration["normalization"]["per_sample_normalization"]:
        input_tokens.append("PSN")
    if data_configuration["image_settings"]["keep_ratio"]:
        input_tokens.append("KAR")
    input_tokens.append(str(data_configuration["image_settings"]["size"]))

    # Backbone: modules, feature map normalization, then the layers and filters of every block in order
    backbone_modules = backbone_configuration["modules"]
    backbone_tokens = []
    if backbone_modules["squeeze_and_excite"]:
        backbone_tokens.append("SE")
    if backbone_modules["residual_nets"]:
        backbone_tokens.append("RES")
    if backbone_modules["inception"]:
        backbone_tokens.append(f"INC{backbone_modules['inception_starting_block']}")
    backbone_tokens.append(NORMALIZATION_CODES[
        str(backbone_configuration["normalization"]["feature_map_normalization"]).lower().strip()])
    convolutions = backbone_configuration["convolutions"]
    backbone_tokens.append("B" + ".".join(f"{convolutions[block][0]}x{convolutions[block][1]}"
                                          for block in sorted(convolutions, key=int)))

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

    # Training: augmentation, batch size and learning rate in short scientific notation (e.g. 2e-4, 1.5e-3)
    training_tokens = ["AUG"] if data_configuration["data_augmentation"]["activated"] else []
    training_tokens.append(f"BS{experiment_configuration['batch_size']}")
    mantissa, exponent = f"{experiment_configuration['learning_rate']:e}".split("e")
    training_tokens.append(f"LR{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}")

    sections = [input_tokens, backbone_tokens, [levels_token], [anchors_token], region_proposal_tokens,
                detector_tokens, [classes_token], training_tokens]
    return "_".join("-".join(section) for section in sections)
