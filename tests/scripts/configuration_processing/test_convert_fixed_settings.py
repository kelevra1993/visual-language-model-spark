"""
Tests of scripts/configuration_processing/convert_fixed_settings.py, the conversion of the configuration files written
before the fixed settings and the added sections and settings: the values changed by a conversion, the neck section
added to the detection files, the box loss added to their detector section, the box loss of their region proposal
section removed (a CIoU one leaving the file as it is), the turned off gradient clipping norm removed (a clipping file
left as it is), the final NMS settings added to their detector section, the proposal selection added to their region
proposal section, and the refusals of a conversion changing another setting or the backbone.
"""

import pytest

from pathlib import Path

from scripts.configuration_processing.convert_fixed_settings import (NECK_SECTION_TEXT, add_required_sections,
                                                                      add_required_settings, check_conversion,
                                                                      convert_fixed_settings, remove_fixed_settings)
from vision_training.data.loaders import NUMBER_LOADER_WORKERS
from vision_training.evaluation.classification.classification_metrics import TOP_K_ACCURACY_CLASSES

# Configuration of a task given by its placeholder, written before the fixed settings, with their comments
CONFIGURATION_TEXT = """ExperimentConfiguration:
  task: {task}
  # Resumes the run from its last checkpoint
  resume_training: True
  evaluation:
    # Classes counted by the top-k accuracy
    top_k: 2
    target_precision: 0.99
Data:
  # Loader settings
  loading:
    # Worker processes of every loader
    number_workers: 8
    prefetch_batches: 2
  data_augmentation:
    activated: True
    independent_frames: False
    shuffle_frames:
      activated: False
      probability: 0.0
Backbone:
  modules:
    inception: False

    # Block from which inception modules replace the plain convolutions
    inception_starting_block: 3

    squeeze_and_excite: True
"""

# The configuration of an image task once converted, the block removed between two blank lines leaving a single one
CONVERTED_CLASSIFICATION_TEXT = """ExperimentConfiguration:
  task: classification
  evaluation:
    target_precision: 0.99
Data:
  data_augmentation:
    activated: True
Backbone:
  modules:
    inception: False

    squeeze_and_excite: True
"""


def test_conversion_lists_the_changed_values_and_refuses_other_changes(tmp_path: Path) -> None:
    """
    Verifies the check of a conversion: the text without the fixed settings is accepted with the list of the values
    its runs change to, a converted text whose other settings changed is refused, and so is a backbone whose inception
    modules start at another block than the fixed one, whose layers the code no longer builds.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    original_text = CONFIGURATION_TEXT.format(task="classification")
    changed_values = check_conversion(original_text=original_text, converted_text=CONVERTED_CLASSIFICATION_TEXT,
                                      task="classification", configuration_path=tmp_path / "configuration.yaml")
    assert changed_values == [f"ExperimentConfiguration.evaluation.top_k 2 -> {TOP_K_ACCURACY_CLASSES}",
                              f"Data.loading.number_workers 8 -> {NUMBER_LOADER_WORKERS}"]
    with pytest.raises(expected_exception=ValueError, match="ExperimentConfiguration.evaluation.target_precision"):
        check_conversion(original_text=original_text, task="classification",
                         converted_text=CONVERTED_CLASSIFICATION_TEXT.replace("0.99", "0.95"),
                         configuration_path=tmp_path / "configuration.yaml")

    # Inception modules from block 2, refused, while a starting block without inception changes no layer
    inception_text = original_text.replace("inception: False", "inception: True").replace(
        "inception_starting_block: 3", "inception_starting_block: 2")
    with pytest.raises(expected_exception=ValueError, match="starts its inception modules at block 2"):
        check_conversion(original_text=inception_text, task="classification",
                         converted_text=remove_fixed_settings(configuration_text=inception_text, task="classification"),
                         configuration_path=tmp_path / "configuration.yaml")
    unused_block_text = original_text.replace("inception_starting_block: 3", "inception_starting_block: 2")
    assert "Backbone.modules.inception_starting_block 2 -> 3" in check_conversion(
        original_text=unused_block_text, task="classification", configuration_path=tmp_path / "configuration.yaml",
        converted_text=remove_fixed_settings(configuration_text=unused_block_text, task="classification"))


# Detection sections written before the head settings were fixed, with a strict fallback assignment turned on
DETECTION_CONFIGURATION_TEXT = """ExperimentConfiguration:
  task: object_detection
RegionProposal:
  normalization:
    feature_map_normalization: "batch"
    # Momentum of the batch normalization running statistics
    batch_normalization_momentum: 0.1
  # Assigns the best-overlapping anchors as foreground even below the minimum threshold.
  strict_fallback_assignment: True
  number_training_positives: 128
  total_training_samples: 256
  # Smooth L1 beta of the box regression loss.
  localisation_loss_beta: 1/9
Detector:
  normalization:
    feature_map_normalization: "batch"
    batch_normalization_momentum: 0.1
  nms_iou_threshold: 0.5
  total_training_samples: 128
  localisation_loss_beta: 1/9
"""

# The detection sections once converted
CONVERTED_DETECTION_TEXT = """ExperimentConfiguration:
  task: object_detection
RegionProposal:
  normalization:
    feature_map_normalization: "batch"
  number_training_positives: 128
  total_training_samples: 256
Detector:
  normalization:
    feature_map_normalization: "batch"
  nms_iou_threshold: 0.5
  total_training_samples: 128
"""


def test_conversion_removes_the_fixed_head_settings(tmp_path: Path) -> None:
    """
    Verifies that the conversion removes the batch normalization momentums of the detection heads, the strict fallback
    assignment and the localisation loss betas with their comments, the check of the whole conversion listing the
    strict assignment turned on as a value its runs change to, the momentums of 0.1 and the betas of 1/9 being those
    of the code.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    converted_text = remove_fixed_settings(configuration_text=DETECTION_CONFIGURATION_TEXT, task="object_detection")
    assert converted_text == CONVERTED_DETECTION_TEXT
    converted_text = add_required_sections(configuration_text=converted_text, task="object_detection",
                                           configuration_path=tmp_path / "configuration.yaml")
    converted_text = add_required_settings(configuration_text=converted_text, task="object_detection",
                                           configuration_path=tmp_path / "configuration.yaml")
    changed_values = check_conversion(original_text=DETECTION_CONFIGURATION_TEXT, converted_text=converted_text,
                                      task="object_detection", configuration_path=tmp_path / "configuration.yaml")
    assert changed_values == ["RegionProposal.strict_fallback_assignment True -> False"]


# The converted detection sections with the neck section added before the region proposal, and its comment
CONVERTED_DETECTION_TEXT_WITH_NECK = CONVERTED_DETECTION_TEXT.replace(
    "RegionProposal:\n", f"{NECK_SECTION_TEXT}\n# Region proposal heads\nRegionProposal:\n")


def test_conversion_adds_the_neck_to_the_detection_files(tmp_path: Path) -> None:
    """
    Verifies that the conversion writes the 'Neck' section without neck before the region proposal section and its
    comment in the detection files written before the neck, and only there: a classification file and a detection file
    holding a neck are left as they are, a detection file without region proposal section is refused, and the check
    refuses a converted file lacking the section or holding other values than those reproducing its runs.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = tmp_path / "configuration.yaml"
    original_text = CONVERTED_DETECTION_TEXT.replace("RegionProposal:\n", "# Region proposal heads\nRegionProposal:\n")
    converted_text = add_required_sections(configuration_text=original_text, task="object_detection",
                                           configuration_path=configuration_path)
    assert converted_text == CONVERTED_DETECTION_TEXT_WITH_NECK
    converted_text = add_required_settings(configuration_text=converted_text, task="object_detection",
                                           configuration_path=configuration_path)
    assert check_conversion(original_text=original_text, converted_text=converted_text, task="object_detection",
                            configuration_path=configuration_path) == []

    # Files of another task or already holding the section are left as they are
    classification_text = CONVERTED_CLASSIFICATION_TEXT
    assert add_required_sections(configuration_text=classification_text, task="classification",
                                 configuration_path=configuration_path) == classification_text
    assert add_required_sections(configuration_text=converted_text, task="object_detection",
                                 configuration_path=configuration_path) == converted_text

    # A detection file without region proposal section, and conversions without the section or with another neck
    with pytest.raises(expected_exception=ValueError, match="has no 'RegionProposal' section"):
        add_required_sections(configuration_text="ExperimentConfiguration:\n  task: object_detection\n",
                              task="object_detection", configuration_path=configuration_path)
    with pytest.raises(expected_exception=ValueError, match="Neck.type"):
        check_conversion(original_text=original_text, task="object_detection", configuration_path=configuration_path,
                         converted_text=add_required_settings(configuration_text=original_text, task="object_detection",
                                                              configuration_path=configuration_path))
    with pytest.raises(expected_exception=ValueError, match="Neck.type"):
        check_conversion(original_text=original_text, task="object_detection", configuration_path=configuration_path,
                         converted_text=converted_text.replace('type: "none"', 'type: "feature_pyramid"'))


# Lines the conversion of 2026-11-08 wrote after the sampling budget of the region proposal section, removed since the
# region proposal network always uses the smooth L1, and the lines it writes after that of the detector section
REGION_PROPOSAL_BOX_LOSS_TEXT = (
    '  # Box loss of the positive anchors: "smooth_l1" on the offsets, or "complete_iou" (1 - CIoU of the\n'
    '  # decoded boxes, https://arxiv.org/abs/2005.03572).\n'
    '  box_loss: "smooth_l1"\n')
DETECTOR_BOX_LOSS_TEXT = REGION_PROPOSAL_BOX_LOSS_TEXT.replace("positive anchors", "positive proposals")

# The converted detection sections with the neck, once the box loss is added after the sampling budget of the
# detector, before the final NMS settings
CONVERTED_DETECTION_TEXT_WITH_BOX_LOSS = CONVERTED_DETECTION_TEXT_WITH_NECK.replace(
    "  total_training_samples: 128\n", f"  total_training_samples: 128\n{DETECTOR_BOX_LOSS_TEXT}")

# Lines the conversion writes after the IoU threshold of the final NMS of the detector section
DETECTOR_NMS_TEXT = (
    '  # Criterion of the final NMS: "iou", or "distance_iou" (the IoU minus the normalized distance of the centres,\n'
    '  # https://arxiv.org/abs/1911.08287), compared with nms_iou_threshold.\n'
    '  nms_criterion: "iou"\n'
    '  # Exponent of the distance term of "distance_iou": 1.0 as in the paper, 0.6 in YOLOv4 (more boxes kept).\n'
    '  nms_distance_exponent: 1.0\n')

# The converted detection sections with the neck and the final NMS settings, without and with the box loss, the
# latter holding every added setting
CONVERTED_DETECTION_TEXT_WITH_NECK_AND_NMS = CONVERTED_DETECTION_TEXT_WITH_NECK.replace(
    "  nms_iou_threshold: 0.5\n", f"  nms_iou_threshold: 0.5\n{DETECTOR_NMS_TEXT}")
CONVERTED_DETECTION_TEXT_WITH_NMS = CONVERTED_DETECTION_TEXT_WITH_BOX_LOSS.replace(
    "  nms_iou_threshold: 0.5\n", f"  nms_iou_threshold: 0.5\n{DETECTOR_NMS_TEXT}")

# Lines the conversion writes after the normalization of the region proposal section, the selection of the proposals
# over all the levels at once, and the region proposal section the text of every file starts with
REGION_PROPOSAL_SELECTION_TEXT = (
    '  # "joint" (one top-k and NMS over all the levels) or "per_level" (top-k and NMS of every level, then the best\n'
    '  # over all the levels, as torchvision, empty proposals dropped), which makes pre_nms_proposals a count per '
    'level.\n'
    '  proposal_selection: "joint"\n')
REGION_PROPOSAL_START_TEXT = 'RegionProposal:\n  normalization:\n    feature_map_normalization: "batch"\n'


def add_proposal_selection_text(configuration_text: str) -> str:
    """
    Writes the lines of the proposal selection after the normalization of the region proposal section of a detection
    text, as the conversion does, for the texts of the files converted after 2026-11-11.

    Args:
        configuration_text (str): The detection text, holding the region proposal section of the texts above.

    Returns:
        str: The text with the proposal selection.
    """
    return configuration_text.replace(REGION_PROPOSAL_START_TEXT,
                                      f"{REGION_PROPOSAL_START_TEXT}{REGION_PROPOSAL_SELECTION_TEXT}")


# The converted detection sections holding every added setting: the neck, the box loss, the final NMS settings and the
# proposal selection
CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS = add_proposal_selection_text(
    configuration_text=CONVERTED_DETECTION_TEXT_WITH_NMS)

# The detection sections of the files converted between 2026-11-08 and 2026-11-10, holding the box loss of the region
# proposal section as well
DETECTION_TEXT_WITH_REGION_PROPOSAL_BOX_LOSS = CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS.replace(
    "  total_training_samples: 256\n", f"  total_training_samples: 256\n{REGION_PROPOSAL_BOX_LOSS_TEXT}")


def test_conversion_adds_the_box_loss_to_the_detection_files(tmp_path: Path) -> None:
    """
    Verifies that the conversion writes the smooth L1 box loss of the runs written before the CIoU loss, with its
    comment lines, right after the sampling budget of the detector section of the detection and instance segmentation
    files holding the final NMS settings, and only there, the region proposal section getting none: a classification
    file and a file holding the box loss are left as they are, a file without sampling budget is refused, and the check
    refuses a converted file lacking the box loss or holding the CIoU loss, which would change its runs.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = tmp_path / "configuration.yaml"
    for task in ["object_detection", "instance_segmentation"]:
        original_text = CONVERTED_DETECTION_TEXT_WITH_NECK_AND_NMS.replace("task: object_detection", f"task: {task}")
        converted_text = add_required_settings(configuration_text=original_text, task=task,
                                               configuration_path=configuration_path)
        assert converted_text == CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS.replace("task: object_detection",
                                                                                      f"task: {task}")
        assert check_conversion(original_text=original_text, converted_text=converted_text, task=task,
                                configuration_path=configuration_path) == []

    # Files of another task or already holding the box loss are left as they are
    assert add_required_settings(configuration_text=CONVERTED_CLASSIFICATION_TEXT, task="classification",
                                 configuration_path=configuration_path) == CONVERTED_CLASSIFICATION_TEXT
    assert add_required_settings(configuration_text=CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS,
                                 task="object_detection", configuration_path=configuration_path) == \
        CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS

    # A detector section without sampling budget, and conversions without the box loss or with the CIoU loss
    with pytest.raises(expected_exception=ValueError, match="has no 'Detector.total_training_samples' setting"):
        add_required_settings(configuration_text=CONVERTED_DETECTION_TEXT_WITH_NECK_AND_NMS.replace(
            "  total_training_samples: 128\n", ""), task="object_detection", configuration_path=configuration_path)
    with pytest.raises(expected_exception=ValueError, match=r"than the fixed ones: Detector\.box_loss\.$"):
        check_conversion(original_text=CONVERTED_DETECTION_TEXT_WITH_NECK_AND_NMS, task="object_detection",
                         converted_text=add_proposal_selection_text(
                             configuration_text=CONVERTED_DETECTION_TEXT_WITH_NECK_AND_NMS),
                         configuration_path=configuration_path)
    complete_iou_text = CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS.replace(
        DETECTOR_BOX_LOSS_TEXT, DETECTOR_BOX_LOSS_TEXT.replace('box_loss: "smooth_l1"', 'box_loss: "complete_iou"'))
    with pytest.raises(expected_exception=ValueError, match=r"than the fixed ones: Detector\.box_loss\.$"):
        check_conversion(original_text=CONVERTED_DETECTION_TEXT_WITH_NECK_AND_NMS, converted_text=complete_iou_text,
                         task="object_detection", configuration_path=configuration_path)


def test_conversion_removes_the_box_loss_of_the_region_proposal_section(tmp_path: Path,
                                                                        capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the conversion removes the smooth L1 box loss of the region proposal section, with its comment lines,
    from the detection and instance segmentation files written between the CIoU loss and its removal from the region
    proposal network, the detector keeping its box loss, and that convert_fixed_settings converts such a launch file
    with its backup while it reports and leaves untouched a file whose region proposal section holds the CIoU loss, its
    runs staying as records, without stopping on it.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    configuration_path = tmp_path / "configuration.yaml"
    for task in ["object_detection", "instance_segmentation"]:
        original_text = DETECTION_TEXT_WITH_REGION_PROPOSAL_BOX_LOSS.replace("task: object_detection", f"task: {task}")
        converted_text = remove_fixed_settings(configuration_text=original_text, task=task)
        assert converted_text == CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS.replace("task: object_detection",
                                                                                      f"task: {task}")
        assert check_conversion(original_text=original_text, converted_text=converted_text, task=task,
                                configuration_path=configuration_path) == []

    # A launch folder holding a file of each region proposal box loss, the CIoU one listed first
    configuration_folder = tmp_path / "launch_files"
    configuration_folder.mkdir()
    complete_iou_text = DETECTION_TEXT_WITH_REGION_PROPOSAL_BOX_LOSS.replace(
        REGION_PROPOSAL_BOX_LOSS_TEXT, REGION_PROPOSAL_BOX_LOSS_TEXT.replace('"smooth_l1"', '"complete_iou"'))
    (configuration_folder / "run-1.yaml").write_text(data=complete_iou_text, encoding="utf-8")
    (configuration_folder / "run-2.yaml").write_text(data=DETECTION_TEXT_WITH_REGION_PROPOSAL_BOX_LOSS,
                                                     encoding="utf-8")
    converted_paths = convert_fixed_settings(run_roots=[], configuration_folders=[configuration_folder],
                                             write_files=True, backup_suffix="_before_region_proposal_box_loss")

    # The smooth L1 file converted with its backup, the CIoU file reported and left as it was
    assert converted_paths == [configuration_folder / "run-2.yaml"]
    assert (configuration_folder / "run-2.yaml").read_text(encoding="utf-8") == \
        CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS
    assert (configuration_folder / "previous_configurations" / "run-2_before_region_proposal_box_loss.yaml").read_text(
        encoding="utf-8") == DETECTION_TEXT_WITH_REGION_PROPOSAL_BOX_LOSS
    assert (configuration_folder / "run-1.yaml").read_text(encoding="utf-8") == complete_iou_text
    backup_folder = configuration_folder / "previous_configurations"
    assert not (backup_folder / "run-1_before_region_proposal_box_loss.yaml").exists()
    printed_output = capsys.readouterr().out
    assert "Not converted, its runs staying as records" in printed_output
    assert "RegionProposal.box_loss 'complete_iou'" in printed_output


# Configuration of a task given by its placeholder, written before the removal of the gradient clipping, its turned
# off norm and comment line between the cosine minimum and the moving average of the weights
CLIPPING_CONFIGURATION_TEXT = """ExperimentConfiguration:
  task: {task}
Optimization:
  # Final cosine learning rate, as a fraction of the learning rate (cosine only).
  minimum_learning_rate_ratio: 0.01
  # Maximum global gradient norm, clipped before every step, null to disable.
  gradient_clipping_norm: null
  # Exponential moving average of the weights.
  exponential_moving_average:
    activated: False
    decay: 0.9998
"""

# The same configuration once converted, the moving average directly after the cosine minimum
CONVERTED_CLIPPING_CONFIGURATION_TEXT = """ExperimentConfiguration:
  task: {task}
Optimization:
  # Final cosine learning rate, as a fraction of the learning rate (cosine only).
  minimum_learning_rate_ratio: 0.01
  # Exponential moving average of the weights.
  exponential_moving_average:
    activated: False
    decay: 0.9998
"""


def test_conversion_removes_the_gradient_clipping_norm(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the conversion removes the turned off gradient clipping norm, with its comment line, from the files of
    the tasks written before its removal, and that convert_fixed_settings converts such a launch file with its backup
    while it reports and leaves untouched a file clipping its gradients, its runs staying as records, without stopping
    on it.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    configuration_path = tmp_path / "configuration.yaml"
    for task in ["classification", "sequence_labelling", "self_supervised"]:
        original_text = CLIPPING_CONFIGURATION_TEXT.format(task=task)
        converted_text = remove_fixed_settings(configuration_text=original_text, task=task)
        assert converted_text == CONVERTED_CLIPPING_CONFIGURATION_TEXT.format(task=task)
        assert check_conversion(original_text=original_text, converted_text=converted_text, task=task,
                                configuration_path=configuration_path) == []

    # A launch folder holding a file clipping its gradients, listed first, and a file without clipping
    configuration_folder = tmp_path / "launch_files"
    configuration_folder.mkdir()
    original_text = CLIPPING_CONFIGURATION_TEXT.format(task="classification")
    clipping_text = original_text.replace("gradient_clipping_norm: null", "gradient_clipping_norm: 1.0")
    (configuration_folder / "run-1.yaml").write_text(data=clipping_text, encoding="utf-8")
    (configuration_folder / "run-2.yaml").write_text(data=original_text, encoding="utf-8")
    converted_paths = convert_fixed_settings(run_roots=[], configuration_folders=[configuration_folder],
                                             write_files=True, backup_suffix="_before_gradient_clipping_norm")

    # The file without clipping converted with its backup, the clipping file reported and left as it was
    backup_folder = configuration_folder / "previous_configurations"
    assert converted_paths == [configuration_folder / "run-2.yaml"]
    assert (configuration_folder / "run-2.yaml").read_text(encoding="utf-8") == \
        CONVERTED_CLIPPING_CONFIGURATION_TEXT.format(task="classification")
    assert (backup_folder / "run-2_before_gradient_clipping_norm.yaml").read_text(encoding="utf-8") == original_text
    assert (configuration_folder / "run-1.yaml").read_text(encoding="utf-8") == clipping_text
    assert not (backup_folder / "run-1_before_gradient_clipping_norm.yaml").exists()
    printed_output = capsys.readouterr().out
    assert "Not converted, its runs staying as records" in printed_output
    assert "Optimization.gradient_clipping_norm 1.0" in printed_output


def test_conversion_adds_the_final_nms_settings_to_the_detection_files(tmp_path: Path) -> None:
    """
    Verifies that the conversion writes the greedy final NMS of the runs written before DIoU-NMS, the criterion "iou"
    and the exponent 1.0 with their comment lines, right after the IoU threshold of the final NMS of the detector
    section of the detection and instance segmentation files converted for the box loss, and only there: a
    classification file and a file holding the settings are left as they are, a detector section without IoU threshold
    is refused, and the check refuses a converted file lacking the settings or holding DIoU-NMS, which would change the
    evaluations of its runs.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = tmp_path / "configuration.yaml"
    for task in ["object_detection", "instance_segmentation"]:
        original_text = CONVERTED_DETECTION_TEXT_WITH_BOX_LOSS.replace("task: object_detection", f"task: {task}")
        converted_text = add_required_settings(configuration_text=original_text, task=task,
                                               configuration_path=configuration_path)
        assert converted_text == CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS.replace("task: object_detection",
                                                                                      f"task: {task}")
        assert check_conversion(original_text=original_text, converted_text=converted_text, task=task,
                                configuration_path=configuration_path) == []

    # Files of another task or already holding the settings are left as they are
    assert add_required_settings(configuration_text=CONVERTED_CLASSIFICATION_TEXT, task="classification",
                                 configuration_path=configuration_path) == CONVERTED_CLASSIFICATION_TEXT
    assert add_required_settings(configuration_text=CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS,
                                 task="instance_segmentation", configuration_path=configuration_path) == \
        CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS

    # A detector section without IoU threshold, and conversions without the settings or with DIoU-NMS
    with pytest.raises(expected_exception=ValueError, match="has no 'Detector.nms_iou_threshold' setting"):
        add_required_settings(configuration_text=CONVERTED_DETECTION_TEXT_WITH_BOX_LOSS.replace(
            "  nms_iou_threshold: 0.5\n", ""), task="object_detection", configuration_path=configuration_path)
    with pytest.raises(expected_exception=ValueError,
                       match=r"than the fixed ones: Detector\.nms_criterion, Detector\.nms_distance_exponent\.$"):
        check_conversion(original_text=CONVERTED_DETECTION_TEXT_WITH_BOX_LOSS, task="object_detection",
                         converted_text=add_proposal_selection_text(
                             configuration_text=CONVERTED_DETECTION_TEXT_WITH_BOX_LOSS),
                         configuration_path=configuration_path)
    distance_text = CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS.replace('nms_criterion: "iou"',
                                                                         'nms_criterion: "distance_iou"')
    with pytest.raises(expected_exception=ValueError, match=r"than the fixed ones: Detector\.nms_criterion\.$"):
        check_conversion(original_text=CONVERTED_DETECTION_TEXT_WITH_BOX_LOSS, converted_text=distance_text,
                         task="object_detection", configuration_path=configuration_path)


def test_conversion_adds_the_proposal_selection_to_the_detection_files(tmp_path: Path) -> None:
    """
    Verifies that the conversion writes the selection of the proposals over all the levels at once of the runs written
    before the selection level by level, "joint" with its comment lines, right after the normalization of the region
    proposal section of the detection and instance segmentation files, and only there: a classification file and a
    file holding the selection are left as they are, a region proposal section without normalization is refused, the
    check refuses a converted file lacking the selection or selecting level by level, which would change its runs, and
    convert_fixed_settings converts a launch file with its backup.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = tmp_path / "configuration.yaml"
    for task in ["object_detection", "instance_segmentation"]:
        original_text = CONVERTED_DETECTION_TEXT_WITH_NMS.replace("task: object_detection", f"task: {task}")
        converted_text = add_required_settings(configuration_text=original_text, task=task,
                                               configuration_path=configuration_path)
        assert converted_text == CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS.replace("task: object_detection",
                                                                                      f"task: {task}")
        assert check_conversion(original_text=original_text, converted_text=converted_text, task=task,
                                configuration_path=configuration_path) == []

    # Files of another task or already holding the selection are left as they are
    assert add_required_settings(configuration_text=CONVERTED_CLASSIFICATION_TEXT, task="classification",
                                 configuration_path=configuration_path) == CONVERTED_CLASSIFICATION_TEXT
    per_level_text = CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS.replace('proposal_selection: "joint"',
                                                                          'proposal_selection: "per_level"')
    assert add_required_settings(configuration_text=per_level_text, task="object_detection",
                                 configuration_path=configuration_path) == per_level_text

    # A region proposal section without normalization, and conversions without the selection or level by level
    with pytest.raises(expected_exception=ValueError, match="has no 'RegionProposal.normalization' setting"):
        add_required_settings(configuration_text=CONVERTED_DETECTION_TEXT_WITH_NMS.replace(
            REGION_PROPOSAL_START_TEXT, "RegionProposal:\n"), task="object_detection",
            configuration_path=configuration_path)
    with pytest.raises(expected_exception=ValueError,
                       match=r"than the fixed ones: RegionProposal\.proposal_selection\.$"):
        check_conversion(original_text=CONVERTED_DETECTION_TEXT_WITH_NMS, task="object_detection",
                         converted_text=CONVERTED_DETECTION_TEXT_WITH_NMS, configuration_path=configuration_path)
    with pytest.raises(expected_exception=ValueError,
                       match=r"than the fixed ones: RegionProposal\.proposal_selection\.$"):
        check_conversion(original_text=CONVERTED_DETECTION_TEXT_WITH_NMS, converted_text=per_level_text,
                         task="object_detection", configuration_path=configuration_path)

    # A launch file converted with its backup, the original kept in it
    configuration_folder = tmp_path / "launch_files"
    configuration_folder.mkdir()
    (configuration_folder / "run-1.yaml").write_text(data=CONVERTED_DETECTION_TEXT_WITH_NMS, encoding="utf-8")
    converted_paths = convert_fixed_settings(run_roots=[], configuration_folders=[configuration_folder],
                                             write_files=True, backup_suffix="_before_proposal_selection")
    assert converted_paths == [configuration_folder / "run-1.yaml"]
    assert (configuration_folder / "run-1.yaml").read_text(encoding="utf-8") == \
        CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS
    assert (configuration_folder / "previous_configurations" / "run-1_before_proposal_selection.yaml").read_text(
        encoding="utf-8") == CONVERTED_DETECTION_TEXT_WITH_NMS
