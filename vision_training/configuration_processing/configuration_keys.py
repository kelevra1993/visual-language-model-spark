"""
Keys of the configuration files that the code reads, with the tasks reading each of them, and the check refusing every
other key: a misspelled key, a key no code reads (a setting reverted from the code, for example) or a section of
another task would otherwise be silently ignored, the training running with the value the code reads instead. The keys
are those read by the loader (configuration_loader.py: the folders and the label mapping file) and by the checks of
every task (the key paths of common_checks.py and of the task check modules, which read every setting the models,
datasets and trainers take). Run by configuration_checker.py on the file as written, after the refusal of the removed
settings (REMOVED_SETTINGS of common_checks.py) and before every other check.
"""

import difflib

from typing import Any, Dict, List, Optional, Tuple

from vision_training.configuration_processing.common_checks import PRETRAINED_CLASS_LAYER_KEYS, SEQUENCE_TASKS
from vision_training.data.augmentations import AUGMENTATION_TYPES

# Groups of the tasks of 'ExperimentConfiguration.task' sharing sections: the two-stage detectors, the tasks with a
# classification head (the image classifier and the sequence tasks), and every task (the keys of
# TASK_CONFIGURATION_CHECKS of configuration_checker.py)
DETECTION_TASKS = ("object_detection", "instance_segmentation")
CLASSIFICATION_LIKE_TASKS = ("classification",) + tuple(SEQUENCE_TASKS)
ALL_TASKS = DETECTION_TASKS + CLASSIFICATION_LIKE_TASKS + ("semantic_segmentation", "self_supervised",
                                                           "keypoint_detection")

# Key standing for any key of a section keyed by names, e.g. the block numbers of 'DetectionLevels'
ANY_KEY = "*"

# Settings of the 'pretrained' block of every section initialising its component with the one of another run, with the
# tasks holding the section (PRETRAINED_COMPONENT_NAMES of common_checks.py)
PRETRAINED_KEYS = ["experiment_folder", "iteration", "freeze", "learning_rate_multiplier"]
PRETRAINED_SECTION_TASKS = {"Backbone": ALL_TASKS, "TemporalModel": tuple(SEQUENCE_TASKS),
                            "ClassificationHead": CLASSIFICATION_LIKE_TASKS,
                            "SegmentationDecoder": ("semantic_segmentation",), "RegionProposal": DETECTION_TASKS,
                            "Detector": DETECTION_TASKS, "MaskHead": ("instance_segmentation",),
                            "HRNet": ("keypoint_detection",), "KeypointHead": ("keypoint_detection",)}

# Every key path the code reads, by the tasks reading it. A path ends at a value: a value that is a dictionary keyed
# by names checked by the task (the blocks of 'Backbone.convolutions', the losses of 'Optimization.loss_weights', the
# classes and keypoints of 'evaluation.keypoint_sigmas') is not looked into, and ANY_KEY matches any key of a section
# whose keys are names, e.g. the levels of 'DetectionLevels'
KNOWN_KEY_PATHS: Dict[str, Tuple[str, ...]] = {
    # Experiment settings of every task: the task, the folders, the training loop, the evaluation and the tracking
    **{f"ExperimentConfiguration.{key_name}": ALL_TASKS for key_name in [
        "task", "experiment_folder", "run_name_suffix", "train_data_folder", "validation_data_folder",
        "test_data_folder", "batch_size", "information_dump", "weight_saving_iterations",
        "intermediate_weight_saving_iterations", "number_iterations", "validation_iteration_ratio", "learning_rate",
        "allow_data_change_on_resume", "evaluation.per_class_report", "evaluation.number_sample_images",
        "tracking.activated", "tracking.tracking_uri", "tracking.experiment_name",
        "diagnosis_modes.compute_receptive_field", "diagnosis_modes.view_input_images",
        "diagnosis_modes.benchmark_speed"]},

    # Evaluation and diagnosis settings of some tasks only
    "ExperimentConfiguration.diagnosis_modes.compute_anchor_coverage.activated": DETECTION_TASKS,
    "ExperimentConfiguration.diagnosis_modes.compute_anchor_coverage.iou_threshold": DETECTION_TASKS,
    "ExperimentConfiguration.evaluation.score_threshold": DETECTION_TASKS,
    **{f"ExperimentConfiguration.evaluation.{key_name}": DETECTION_TASKS + CLASSIFICATION_LIKE_TASKS for key_name in [
        "target_precision", "error_dumps.activated", "error_dumps.maximum_per_class_pair"]},
    "ExperimentConfiguration.evaluation.save_predictions": CLASSIFICATION_LIKE_TASKS,
    "ExperimentConfiguration.evaluation.original_resolution": ("instance_segmentation", "semantic_segmentation"),
    "ExperimentConfiguration.evaluation.save_predicted_masks": ("semantic_segmentation",),
    "ExperimentConfiguration.evaluation.boundary_tolerance": ("semantic_segmentation",),
    "ExperimentConfiguration.evaluation.top_k": ("frame_selection",),
    "ExperimentConfiguration.evaluation.number_neighbours": ("self_supervised",),
    "ExperimentConfiguration.evaluation.number_reference_images_per_class": ("self_supervised",),
    "ExperimentConfiguration.evaluation.default_keypoint_sigma": ("keypoint_detection",),
    "ExperimentConfiguration.evaluation.keypoint_sigmas": ("keypoint_detection",),

    # Optimizer, schedule, moving average of the weights, and the loss weights of the detectors
    **{f"Optimization.{key_name}": ALL_TASKS for key_name in [
        "optimizer", "weight_decay", "learning_rate_schedule", "warmup_iterations", "decay_iterations",
        "minimum_learning_rate_ratio", "exponential_moving_average.activated", "exponential_moving_average.decay"]},
    "Optimization.loss_weights": DETECTION_TASKS,

    # Images, labels and augmentations of every task, every augmentation type with its switch and probability
    **{f"Data.{key_name}": ALL_TASKS for key_name in [
        "image_settings.size", "image_settings.keep_ratio", "label_mapping.mode", "label_mapping.mapping_file_path",
        "data_augmentation.activated", "data_augmentation.probability",
        "data_augmentation.types.random_black_out.grid_size"]},
    **{f"Data.data_augmentation.types.{augmentation_type}.{key_name}": ALL_TASKS
       for augmentation_type in AUGMENTATION_TYPES for key_name in ["activated", "probability"]},

    # Data settings of some tasks only: the frames of the sequence tasks, the index maps of the instance segmentation,
    # the tiles of the semantic segmentation and the grouped batches of the self-supervised task
    **{f"Data.{key_name}": tuple(SEQUENCE_TASKS) for key_name in [
        "sequence_settings.maximum_sequence_length", "data_augmentation.independent_frames",
        "data_augmentation.shuffle_frames.activated", "data_augmentation.shuffle_frames.probability"]},
    "Data.instance_index_maps": ("instance_segmentation",),
    **{f"Data.tiling.{key_name}": ("semantic_segmentation",) for key_name in ["activated", "scale", "overlap"]},
    **{f"Data.grouped_batches.{key_name}": ("self_supervised",) for key_name in [
        "activated", "batch_key", "block_key", "images_per_block"]},

    # Backbone of every task, its blocks keyed by their numbers
    **{f"Backbone.{key_name}": ALL_TASKS for key_name in [
        "input_channels", "convolutions", "last_max_pooling", "normalization.input_normalization",
        "normalization.per_sample_normalization", "normalization.feature_map_normalization", "modules.residual_nets",
        "modules.residual_nets_v2", "modules.inception", "modules.squeeze_and_excite",
        "pretrained.freeze_normalization_parameters"]},

    # 'pretrained' settings of every transferable section, with the class layer flag of the sections with class layers
    **{f"{section_name}.pretrained.{key_name}": section_tasks
       for section_name, section_tasks in PRETRAINED_SECTION_TASKS.items() for key_name in PRETRAINED_KEYS},
    **{f"{section_name}.pretrained.{key_name}": PRETRAINED_SECTION_TASKS[section_name]
       for section_name, key_name in PRETRAINED_CLASS_LAYER_KEYS.items()},

    # Classification head of the image classifier and of the sequence tasks, the decorrelation being the classifier's
    **{f"ClassificationHead.{key_name}": CLASSIFICATION_LIKE_TASKS for key_name in [
        "global_pooling", "fully_connected", "fully_connected_normalization", "dropout_rate", "output_mode",
        "label_smoothing", "class_weighting"]},
    **{f"ClassificationHead.decorrelation.{key_name}": ("classification",) for key_name in [
        "activated", "weight", "hidden_layer"]},

    # Detection levels keyed by their backbone block numbers, the neck, the region proposal and detection heads
    **{f"DetectionLevels.{ANY_KEY}.{key_name}": DETECTION_TASKS for key_name in [
        "anchors.scales", "anchors.aspect_ratios", "region_proposal.convolutions", "detector.roi_align_pool_size",
        "detector.convolutions", "detector.fully_connected"]},
    **{f"Neck.{key_name}": DETECTION_TASKS for key_name in [
        "type", "output_channels", "normalization.feature_map_normalization"]},
    **{f"RegionProposal.{key_name}": DETECTION_TASKS for key_name in [
        "normalization.feature_map_normalization", "proposal_selection", "training.pre_nms_proposals",
        "training.post_nms_proposals", "inference.pre_nms_proposals", "inference.post_nms_proposals",
        "foreground_iou_threshold.min", "foreground_iou_threshold.max", "background_iou_threshold.min",
        "background_iou_threshold.max", "nms_iou_threshold", "number_training_positives", "total_training_samples"]},
    **{f"Detector.{key_name}": DETECTION_TASKS for key_name in [
        "normalization.feature_map_normalization", "modules.residual_nets", "modules.inception",
        "modules.squeeze_and_excite", "inference.score_threshold", "inference.maximum_detections",
        "foreground_iou_threshold", "nms_iou_threshold", "nms_criterion", "nms_distance_exponent",
        "number_training_positives", "total_training_samples", "box_loss"]},

    # Mask heads of the instance segmentation, with their settings in every detection level
    **{f"DetectionLevels.{ANY_KEY}.mask_head.{key_name}": ("instance_segmentation",) for key_name in [
        "roi_align_pool_size", "convolutions"]},
    **{f"MaskHead.{key_name}": ("instance_segmentation",) for key_name in [
        "shared", "mask_resolution", "upsampling", "fusion_convolutions", "image_branch_convolutions",
        "skip_blocks_at_pool_size", "skip_blocks_at_mask_resolution", "normalization.feature_map_normalization",
        "modules.squeeze_and_excite", "modules.residual_nets", "modules.inception", "inference.mask_threshold"]},

    # Decoder and losses of the semantic segmentation, the decoder levels and deep supervision keyed by block numbers
    **{f"SegmentationDecoder.{key_name}": ("semantic_segmentation",) for key_name in [
        "upsampling", "convolutions", "modules.residual_nets", "modules.squeeze_and_excite",
        "feature_map_normalization", "dropout_rate", "deep_supervision_weights"]},
    **{f"SegmentationLoss.{key_name}": ("semantic_segmentation",) for key_name in [
        "cross_entropy_weight", "focal_gamma", "label_smoothing", "class_weighting", "dice_weight", "lovasz_weight",
        "boundary_weight", "boundary_distance"]},

    # Temporal module of the sequence tasks, fusion of the comparison and selection of the frame selection
    **{f"TemporalModel.{key_name}": tuple(SEQUENCE_TASKS) for key_name in [
        "type", "hidden_size", "number_layers", "number_heads", "kernel_size", "dropout_rate", "bidirectional",
        "causal", "positional_embedding"]},
    **{f"Fusion.{key_name}": ("sequence_comparison",) for key_name in [
        "type", "number_heads", "dropout_rate", "sequence_normalization_scope"]},
    **{f"Selection.{key_name}": ("frame_selection",) for key_name in [
        "decoding", "frames_per_class", "selection_loss_weight", "unselected_category"]},

    # Projection head of the self-supervised task
    **{f"ProjectionHead.{key_name}": ("self_supervised",) for key_name in [
        "method", "global_pooling", "fully_connected", "fully_connected_normalization", "simclr.temperature",
        "barlow_twins.off_diagonal_weight"]},

    # High resolution network and keypoint head of the keypoint detection
    **{f"HRNet.{key_name}": ("keypoint_detection",) for key_name in [
        "base_width", "blocks_per_branch", "modules_per_stage", "modules.residual_nets", "modules.squeeze_and_excite",
        "normalization.feature_map_normalization"]},
    **{f"KeypointHead.{key_name}": ("keypoint_detection",) for key_name in [
        "heatmap_sigma", "hidden_width", "higher_resolution", "ignore_unlabelled_objects", "tag_pull_weight",
        "tag_push_weight", "inference.keypoint_threshold", "inference.maximum_candidates",
        "inference.object_score_threshold", "inference.tag_threshold"]}}


def build_known_key_tree(known_key_paths: Dict[str, Tuple[str, ...]]) -> Dict[str, Any]:
    """
    Arranges the known key paths into a tree walked alongside a configuration by find_unknown_keys: every node holds
    the tasks of the paths going through it, whether it ends a path (its value is not looked into) and its child
    nodes by key, so that a section is known to a task when one of its keys is.

    Args:
        known_key_paths (Dict[str, Tuple[str, ...]]): The tasks reading every key path, e.g. KNOWN_KEY_PATHS.

    Returns:
        Dict[str, Any]: The root node, with 'tasks' (set), 'value' (bool, True when it ends a path) and 'children'
            (the nodes of its keys).

    Raises:
        ValueError: If a key path ends where another one goes on, the node being both a value and a section.
    """
    root = {"tasks": set(), "value": False, "children": {}}
    for key_path, tasks in known_key_paths.items():
        # Every node along the path is read by the tasks of the path, the last one ending it
        node = root
        for key in key_path.split("."):
            node = node["children"].setdefault(key, {"tasks": set(), "value": False, "children": {}})
            node["tasks"].update(tasks)
        node["value"] = True

    # A node ending a path is not looked into, so it cannot hold other known keys
    nodes_to_visit = [("", root)]
    while nodes_to_visit:
        node_path, node = nodes_to_visit.pop()
        if node["value"] and node["children"]:
            raise ValueError(f"The known key path '{node_path}' ends where other known key paths go on.")
        nodes_to_visit += [(f"{node_path}.{key}" if node_path else key, child_node)
                           for key, child_node in node["children"].items()]
    return root


# Tree of KNOWN_KEY_PATHS, built once
KNOWN_KEY_TREE = build_known_key_tree(known_key_paths=KNOWN_KEY_PATHS)


def describe_unknown_key(key_path: str, key: Any, sibling_nodes: Dict[str, Any], task: Optional[str]) -> str:
    """
    Writes the problem of a key no code reads at its place, for the list of check_unknown_keys, with the closest key
    the task reads at the same place when one is close enough to be a misspelling of it.

    Args:
        key_path (str): The dotted path of the key in the file.
        key (Any): The key itself, as written.
        sibling_nodes (Dict[str, Any]): The known nodes at the place of the key, by key.
        task (Optional[str]): The task of the configuration, None when it is missing or not a task.

    Returns:
        str: The description of the problem.
    """
    # Keys read at this place by the task, the wildcard of the sections keyed by names left out
    known_keys = [known_key for known_key, node in sibling_nodes.items()
                  if known_key != ANY_KEY and (task is None or task in node["tasks"])]
    close_keys = difflib.get_close_matches(word=str(key), possibilities=known_keys, n=1)
    suggestion = f" (did you mean '{close_keys[0]}'?)" if close_keys else ""
    return (f"'{key_path}' is not a configuration key{suggestion}: no code reads it, so its value would be ignored. "
            f"Correct it or delete it from the configuration file.")


def find_unknown_keys(configuration: Dict[str, Any], node: Dict[str, Any], key_prefix: str,
                      task: Optional[str]) -> List[str]:
    """
    Walks one section of a configuration alongside its node of KNOWN_KEY_TREE and describes every key of the section
    that no code reads, or that only other tasks read, then walks the known sections below it, so that a whole unknown
    section is reported once, by its own path, and not key by key.

    Args:
        configuration (Dict[str, Any]): The section of the configuration.
        node (Dict[str, Any]): The node of KNOWN_KEY_TREE at the place of the section.
        key_prefix (str): The dotted path of the section, empty for the whole file.
        task (Optional[str]): The task of the configuration, None when it is missing or not a task, every known key
            being accepted then (the task itself being refused by the experiment checks).

    Returns:
        List[str]: The description of every unknown key, in the order of the file.
    """
    errors = []
    for key, value in configuration.items():
        key_path = f"{key_prefix}.{key}" if key_prefix else str(key)

        # The node of the key, or of any key in a section keyed by names
        children = node["children"]
        key_node = children[str(key)] if str(key) in children else children[ANY_KEY] if ANY_KEY in children else None
        if key_node is None:
            errors.append(describe_unknown_key(key_path=key_path, key=key, sibling_nodes=children, task=task))
            continue

        # A key read by other tasks only, e.g. a section copied from the configuration of another task
        if task is not None and task not in key_node["tasks"]:
            errors.append(f"'{key_path}' is not read by the task '{task}', only by {sorted(key_node['tasks'])}: its "
                          f"value would be ignored. Delete it from the configuration file.")
            continue

        # The keys of a known section, its value being reported by the other checks when it is not a section
        if not key_node["value"] and isinstance(value, dict):
            errors += find_unknown_keys(configuration=value, node=key_node, key_prefix=key_path, task=task)
    return errors


def check_unknown_keys(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> List[str]:
    """
    Refuses every key of a configuration file that the code of its task does not read (KNOWN_KEY_PATHS): a misspelled
    key, whose correct key is then missing too, a setting no code reads any longer, or a section of another task.
    Without this check their values would be silently ignored, the training running with other values than written.
    Run by configuration_checker.py on the file as written, after the removed settings, which are refused with what
    replaced them, and before every other check.

    Args:
        experiment_configuration (Dict[str, Any]): The 'ExperimentConfiguration' section as written, giving the task.
        model_configuration (Dict[str, Any]): The other sections as written.

    Returns:
        List[str]: The description of every unknown key, by its dotted path.
    """
    # Nest the experiment section under its name so that every key path matches its location in the YAML file
    configuration = {"ExperimentConfiguration": experiment_configuration, **model_configuration}
    task = experiment_configuration["task"] if "task" in experiment_configuration else None
    return find_unknown_keys(configuration=configuration, node=KNOWN_KEY_TREE, key_prefix="",
                             task=task if task in ALL_TASKS else None)
