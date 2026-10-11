"""
Migration of configuration files and run folders written by older versions of the code: inserts as text, keeping
the comments, the keys and sections added since (task, pretrained settings, evaluation outputs, per-class report,
sample images, MLflow tracking, sequence settings and batch padding, positional embedding of the transformer,
segmentation losses, deep supervision, tiles and evaluation settings, Optimization), and renames the run folder to its
current run name (without the training, learning rate multiplier and dropout tokens no longer written in the run
names). Called by scripts/configuration/migrate_configuration.py.
"""

import yaml

from pathlib import Path

from vision_training.configuration.configuration_loader import load_experiment_configuration
from vision_training.configuration.run_name import build_run_name
from vision_training.utilities.os_utilities import format_path_link, print_color

# Section header under which the task key is inserted, and the comment written above it
EXPERIMENT_SECTION_HEADER = "ExperimentConfiguration:\n"
TASK_KEY_COMMENT = ("  # Task trained by this configuration, selecting the trainer, the model and the checks of "
                    "the task sections.\n")

# Pretrained settings of a run trained from random weights, inserted below the header of every section whose component
# can be initialised by the one of another run
PRETRAINED_SETTINGS_BLOCKS = {
    "Backbone": ("  # Backbone of another training run initialising this run, null to start from random weights.\n"
                 "  pretrained:\n"
                 "    experiment_folder: null\n"
                 "    iteration: null\n"
                 "    freeze: False\n"
                 "    freeze_normalization_parameters: False\n"
                 "    learning_rate_multiplier: 1.0\n"),
    "TemporalModel": ("  # Temporal module of another training run initialising this run, null to start from random "
                      "weights.\n"
                      "  pretrained:\n"
                      "    experiment_folder: null\n"
                      "    iteration: null\n"
                      "    freeze: False\n"
                      "    learning_rate_multiplier: 1.0\n"),
    "ClassificationHead": ("  # Classification head of another training run initialising this run, null to start "
                           "from random weights.\n"
                           "  pretrained:\n"
                           "    experiment_folder: null\n"
                           "    iteration: null\n"
                           "    freeze: False\n"
                           "    learning_rate_multiplier: 1.0\n"),
    "RegionProposal": ("  # Region proposal heads of another training run initialising this run, null to start from "
                       "random weights.\n"
                       "  pretrained:\n"
                       "    experiment_folder: null\n"
                       "    iteration: null\n"
                       "    freeze: False\n"
                       "    learning_rate_multiplier: 1.0\n"),
    "Detector": ("  # Detection heads of another training run initialising this run, null to start from random "
                 "weights.\n"
                 "  pretrained:\n"
                 "    experiment_folder: null\n"
                 "    iteration: null\n"
                 "    freeze: False\n"
                 "    learning_rate_multiplier: 1.0\n")}

# Section header of the backbone, start of the line of the freeze flag of its pretrained settings, and the flag keeping
# its normalization parameters fixed inserted after it, off in every run written before it existed
BACKBONE_SECTION_HEADER = "Backbone:\n"
PRETRAINED_FREEZE_LINE_START = "    freeze:"
FREEZE_NORMALIZATION_LINES = ("    # Keeps the scale and shift of the normalization layers of the loaded backbone "
                              "fixed while its convolutions learn.\n"
                              "    freeze_normalization_parameters: False\n")

# Start of the lines of the target precision and of the predictions setting of the evaluation settings of the
# classification-like tasks, and the settings of the predictions file and of the error images inserted after them,
# off in every run written before they existed
TARGET_PRECISION_LINE_START = "    target_precision:"
SAVE_PREDICTIONS_LINE_START = "    save_predictions:"
SAVE_PREDICTIONS_LINES = ("    # Saves the predictions of every test sample in the evaluation folder of every "
                          "evaluated checkpoint.\n"
                          "    save_predictions: False\n")
ERROR_DUMPS_LINES = ("    # Saves the misclassified test samples as images at every evaluation.\n"
                     "    error_dumps:\n"
                     "      activated: False\n"
                     "      maximum_per_class_pair: 20\n")

# Line of the evaluation settings of the experiment section, and the switch of the per-class part of the evaluation
# reports inserted as their first setting, on as in every run written before it existed
EVALUATION_SECTION_LINE = "  evaluation:\n"
PER_CLASS_REPORT_LINES = ("    # Reports every class at every evaluation: the table of the classes, their TensorBoard "
                          "scalars and the\n"
                          "    # confusion matrix image. False keeps the global metrics only.\n"
                          "    per_class_report: True\n")

# Start of the line of the per-class report switch, and the number of sample images inserted after it, none as in the
# runs of every task but the segmentation written before it was shared by every task
PER_CLASS_REPORT_LINE_START = "    per_class_report:"
NUMBER_SAMPLE_IMAGES_LINES = ("    # Number of fixed test samples drawn with their targets and predictions at every "
                              "evaluation, 0 for none.\n"
                              "    number_sample_images: 0\n")

# MLflow tracking section inserted after the evaluation settings of the experiment section, off as in every run
# written before it existed
TRACKING_SECTION_LINES = ("  # Tracking of the run in MLflow, next to TensorBoard and Results.txt: the "
                          "hyperparameters, the averages of the\n"
                          "  # training and validation metrics of every information_dump interval, the learning rate, "
                          "the test metrics of every\n"
                          "  # evaluated checkpoint and a few small artifacts (configuration, confusion matrices, "
                          "class metrics and thresholds,\n"
                          "  # Results.txt). The tracking never blocks nor stops the training: any MLflow error prints "
                          "one warning and the\n"
                          "  # training continues without it.\n"
                          "  tracking:\n"
                          "    # Turns the MLflow tracking on.\n"
                          "    activated: False\n"
                          "    # Address of an MLflow tracking server, e.g. \"http://192.168.1.20:5000\", or null for "
                          "the local database of the\n"
                          "    # experiments root folder (sqlite:///<experiment_folder>/mlflow.db, artifacts in "
                          "<experiment_folder>/mlartifacts),\n"
                          "    # browsed with \"mlflow ui --backend-store-uri "
                          "sqlite:///<experiment_folder>/mlflow.db\". The credentials of a\n"
                          "    # server are read from the MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD "
                          "environment variables, never from\n"
                          "    # this file.\n"
                          "    tracking_uri: null\n"
                          "    # MLflow experiment grouping the runs, null for the task name (e.g. "
                          "\"classification\").\n"
                          "    experiment_name: null\n"
                          "    # Sends the configuration, the confusion matrices, the class metrics and thresholds and "
                          "Results.txt as artifacts.\n"
                          "    log_artifacts: True\n")

# Lines of the augmentation section and of its types, and the sequence augmentations inserted before the types, off in
# every run written before they existed
DATA_AUGMENTATION_LINE = "  data_augmentation:\n"
AUGMENTATION_TYPES_LINE = "    types:\n"
SEQUENCE_AUGMENTATION_LINES = ("    # Draws the photometric augmentations frame by frame (sequence tasks only).\n"
                               "    independent_frames: False\n"
                               "    # Shuffles the real frames of the training windows (sequence tasks only).\n"
                               "    shuffle_frames: { activated: False, probability: 0.5 }\n")

# Header of the fusion section of the comparison configurations, and the scope of the sequence normalization inserted
# below it, every member normalised by its own frames, the only behaviour of the runs written before it existed
FUSION_SECTION_HEADER = "Fusion:\n"
SEQUENCE_NORMALIZATION_SCOPE_LINES = ("  # Frames sharing the statistics of the sequence normalization of the "
                                      "backbone.\n"
                                      "  sequence_normalization_scope: \"member\"\n")

# Header of the data section, and the loading settings inserted below it, the 4 worker processes of the runs written
# before they existed
DATA_SECTION_HEADER = "Data:\n"
LOADING_SETTINGS_LINES = ("  # Worker processes of the loaders and batches prepared in advance by each of them.\n"
                          "  loading:\n"
                          "    number_workers: 4\n"
                          "    prefetch_batches: 2\n")

# Header and last line of the block of the NVIDIA DALI loader in the data section, removed from the runs written
# when it existed
DALI_SECTION_LINE = "  DALI:\n"
DALI_ACTIVATED_LINE_START = "    activated:"

# Start of the line of the handling of the short sequences of the sequence settings, and the padding of the batches
# inserted after it, every window padded to the sequence length as in every run written before it existed
SHORT_SEQUENCES_LINE_START = "    short_sequences:"
BATCH_PADDING_LINES = ("    # Frames the windows of a batch are padded to: \"sequence_length\" or \"longest\" (the longest "
                       "window of the batch).\n"
                       "    batch_padding: \"sequence_length\"\n")

# Header of the temporal module section of the sequence configurations, and the positional embedding flag inserted
# below it, the transformer of every run written before the flag existed giving its frames their position
TEMPORAL_SECTION_HEADER = "TemporalModel:\n"
POSITIONAL_EMBEDDING_LINES = ("  # Gives every frame its position with a learned embedding (transformer).\n"
                              "  positional_embedding: True\n")

# Settings of the semantic segmentation added after its first runs, all off as in these runs: the path of the section
# holding each one, its name, the header line below which it is inserted, and its inserted lines
SEGMENTATION_SETTINGS_INSERTIONS = [
    (["Data"], "tiling", DATA_SECTION_HEADER,
     "  # Training on tiles of the scaled images instead of the images resized to the network input (off).\n"
     "  tiling:\n"
     "    activated: False\n"
     "    scale: 1.0\n"
     "    overlap: 128\n"),
    (["SegmentationDecoder"], "deep_supervision_weights", "SegmentationDecoder:\n",
     "  # Deep supervision heads of the intermediate decoder levels (none).\n"
     "  deep_supervision_weights: {}\n"),
    (["SegmentationLoss"], "lovasz_weight", "SegmentationLoss:\n",
     "  # Weight of the Lovasz-Softmax loss (off).\n"
     "  lovasz_weight: 0.0\n"),
    (["SegmentationLoss"], "boundary_weight", "SegmentationLoss:\n",
     "  # Weight of the boundary loss (off).\n"
     "  boundary_weight: 0.0\n"),
    (["SegmentationLoss"], "boundary_distance", "SegmentationLoss:\n",
     "  # Distance in pixels from the true edges over which the boundary loss weighs the pixels.\n"
     "  boundary_distance: 3\n"),
    (["ExperimentConfiguration", "evaluation"], "original_resolution", EVALUATION_SECTION_LINE,
     "    # Metrics at the original resolution of the test images (off, at the network resolution).\n"
     "    original_resolution: False\n"),
    (["ExperimentConfiguration", "evaluation"], "boundary_tolerance", EVALUATION_SECTION_LINE,
     "    # Distance in pixels within which a predicted contour counts as right for the boundary metrics.\n"
     "    boundary_tolerance: 2\n")]

# Optimization section of a run trained with plain Adam at a constant learning rate, the training of every run
# written before the section existed
OPTIMIZATION_SECTION = ("\n# Optimization of a run trained with Adam at a constant learning rate.\n"
                        "Optimization:\n"
                        "  optimizer: \"adam\"\n"
                        "  weight_decay: 0.0\n"
                        "  learning_rate_schedule: \"constant\"\n"
                        "  warmup_iterations: 0\n"
                        "  decay_iterations: 100000\n"
                        "  minimum_learning_rate_ratio: 0.01\n"
                        "  gradient_clipping_norm: null\n"
                        "  mixed_precision: False\n")


def add_task_to_configuration(configuration_path: Path, task: str) -> bool:
    """
    Upgrades a configuration file written before the task-aware configuration by inserting its
    'ExperimentConfiguration.task' key right below the section header. The key is inserted as text, so that every
    comment of the file is kept, and a file that already has a task is left untouched.

    Args:
        configuration_path (Path): The configuration file, usually the 'training_configuration.yaml' of a run.
        task (str): The task the run was trained for, e.g. "object_detection".

    Returns:
        bool: True if the task was added, False if the file already had one.

    Raises:
        ValueError: If the file has no 'ExperimentConfiguration:' section header.
    """
    text = configuration_path.read_text()
    if "task" in yaml.safe_load(stream=text)["ExperimentConfiguration"]:
        return False
    if EXPERIMENT_SECTION_HEADER not in text:
        raise ValueError(f"No '{EXPERIMENT_SECTION_HEADER.strip()}' section header in "
                         f"{format_path_link(path=configuration_path)}")

    configuration_path.write_text(data=text.replace(
        EXPERIMENT_SECTION_HEADER, f'{EXPERIMENT_SECTION_HEADER}{TASK_KEY_COMMENT}  task: "{task}"\n', 1))
    return True


def add_pretrained_settings_to_configuration(configuration_path: Path, section_name: str) -> bool:
    """
    Upgrades a configuration file written before the transfer of a component by inserting the 'pretrained' settings of
    its section (Backbone, TemporalModel, ClassificationHead, RegionProposal or Detector), those of a run trained from
    random weights, right below the section header. The settings are inserted as text, so that every comment of the file
    is kept, and a file that already has them, or that has no such section (e.g. no temporal module in a detection
    configuration), is left untouched.

    Args:
        configuration_path (Path): The configuration file, usually the 'training_configuration.yaml' of a run.
        section_name (str): The section receiving the settings, a key of PRETRAINED_SETTINGS_BLOCKS.

    Returns:
        bool: True if the settings were added, False if the file already had them or has no such section.

    Raises:
        ValueError: If the section exists but its header is not a line of its own.
    """
    text = configuration_path.read_text()
    configuration = yaml.safe_load(stream=text)
    if section_name not in configuration or "pretrained" in configuration[section_name]:
        return False
    section_header = f"{section_name}:\n"
    lines = text.splitlines(keepends=True)
    if section_header not in lines:
        raise ValueError(f"No '{section_name}:' section header in {format_path_link(path=configuration_path)}")

    # Insert the settings below the header line, the only line made of the header alone
    header_index = lines.index(section_header)
    configuration_path.write_text(data="".join(lines[:header_index + 1] + [PRETRAINED_SETTINGS_BLOCKS[section_name]] +
                                               lines[header_index + 1:]))
    return True


def add_freeze_normalization_parameters_to_configuration(configuration_path: Path) -> bool:
    """
    Upgrades a configuration file written before the 'Backbone.pretrained.freeze_normalization_parameters' flag by
    inserting it, off, right after the freeze flag of the backbone pretrained settings: the first freeze flag after
    the backbone header, whether the settings close the section (the shipped configurations) or open it (the
    settings inserted by add_pretrained_settings_to_configuration). The flag is inserted as text, so that every
    comment of the file is kept, and a file that already has it is left untouched.

    Args:
        configuration_path (Path): The configuration file, usually the 'training_configuration.yaml' of a run.

    Returns:
        bool: True if the flag was added, False if the file already had it.

    Raises:
        ValueError: If the backbone has no pretrained settings with a freeze flag.
    """
    text = configuration_path.read_text()
    backbone_configuration = yaml.safe_load(stream=text)["Backbone"]
    if "pretrained" in backbone_configuration and \
            "freeze_normalization_parameters" in backbone_configuration["pretrained"]:
        return False

    # Freeze flag of the pretrained settings, searched from the backbone header to the next section header
    lines = text.splitlines(keepends=True)
    freeze_index = None
    for line_index in range(lines.index(BACKBONE_SECTION_HEADER) + 1, len(lines)):
        if lines[line_index][:1].isalpha():
            break
        if lines[line_index].startswith(PRETRAINED_FREEZE_LINE_START):
            freeze_index = line_index
            break
    if "pretrained" not in backbone_configuration or freeze_index is None:
        raise ValueError(f"No 'Backbone.pretrained.freeze' flag in {format_path_link(path=configuration_path)}, "
                         f"add the pretrained settings first with add_pretrained_settings_to_configuration.")

    configuration_path.write_text(data="".join(lines[:freeze_index + 1] + [FREEZE_NORMALIZATION_LINES] +
                                               lines[freeze_index + 1:]))
    return True


def add_evaluation_outputs_to_configuration(configuration_path: Path) -> bool:
    """
    Upgrades the configuration file of a classification-like run written before the evaluation outputs by inserting
    the settings it lacks, off as in the runs of that time: 'ExperimentConfiguration.evaluation.save_predictions' right
    after the target precision of its evaluation settings, then 'ExperimentConfiguration.evaluation.error_dumps' right
    after it. The settings are inserted as text, so that every comment of the file is kept, and a file that already
    has them, or whose evaluation settings have no target precision (object detection), is left untouched.

    Args:
        configuration_path (Path): The configuration file, usually the 'training_configuration.yaml' of a run.

    Returns:
        bool: True if a setting was added, False if the file already had them or is not a classification-like run.
    """
    text = configuration_path.read_text()
    experiment_configuration = yaml.safe_load(stream=text)["ExperimentConfiguration"]
    if "evaluation" not in experiment_configuration or \
            "target_precision" not in experiment_configuration["evaluation"]:
        return False
    evaluation_settings = experiment_configuration["evaluation"]
    lines = text.splitlines(keepends=True)
    added_settings = False

    # Predictions setting after the line of the target precision, the only one of the file
    if "save_predictions" not in evaluation_settings:
        precision_index = next(line_index for line_index, line in enumerate(lines)
                               if line.startswith(TARGET_PRECISION_LINE_START))
        lines = "".join(lines[:precision_index + 1] + [SAVE_PREDICTIONS_LINES] +
                        lines[precision_index + 1:]).splitlines(keepends=True)
        added_settings = True

    # Error images setting after the line of the predictions setting
    if "error_dumps" not in evaluation_settings:
        predictions_index = next(line_index for line_index, line in enumerate(lines)
                                 if line.startswith(SAVE_PREDICTIONS_LINE_START))
        lines = lines[:predictions_index + 1] + [ERROR_DUMPS_LINES] + lines[predictions_index + 1:]
        added_settings = True

    if added_settings:
        configuration_path.write_text(data="".join(lines))
    return added_settings


def add_per_class_report_to_configuration(configuration_path: Path) -> bool:
    """
    Upgrades the configuration file of a run of any task written before
    'ExperimentConfiguration.evaluation.per_class_report' by inserting the switch, on, as the first setting of its
    evaluation settings: the runs of that time reported every class. The switch is inserted as text, so that every
    comment of the file is kept, and a file that already has it, or has no evaluation settings, is left untouched.

    Args:
        configuration_path (Path): The configuration file, usually the 'training_configuration.yaml' of a run.

    Returns:
        bool: True if the switch was added, False if the file already had it or has no evaluation settings.

    Raises:
        ValueError: If the evaluation settings have no line of their own in the experiment section.
    """
    text = configuration_path.read_text()
    experiment_configuration = yaml.safe_load(stream=text)["ExperimentConfiguration"]
    if "evaluation" not in experiment_configuration or "per_class_report" in experiment_configuration["evaluation"]:
        return False

    # Line of the evaluation settings, searched from the experiment header to the next section header
    lines = text.splitlines(keepends=True)
    evaluation_index = None
    for line_index in range(lines.index(EXPERIMENT_SECTION_HEADER) + 1, len(lines)):
        if lines[line_index][:1].isalpha():
            break
        if lines[line_index] == EVALUATION_SECTION_LINE:
            evaluation_index = line_index
            break
    if evaluation_index is None:
        raise ValueError(f"No '  evaluation:' line in the experiment section of "
                         f"{format_path_link(path=configuration_path)}")

    configuration_path.write_text(data="".join(lines[:evaluation_index + 1] + [PER_CLASS_REPORT_LINES] +
                                               lines[evaluation_index + 1:]))
    return True


def add_sample_images_to_configuration(configuration_path: Path) -> bool:
    """
    Upgrades the configuration file of a run written before 'ExperimentConfiguration.evaluation.number_sample_images'
    was shared by every task by inserting it, at 0 (no sample images, as in the runs of that time), right after the
    per-class report switch of its evaluation settings. The setting is inserted as text, so that every comment of the
    file is kept, and a file that already has it (the segmentation runs), or has no evaluation settings, is left
    untouched.

    Args:
        configuration_path (Path): The configuration file, usually the 'training_configuration.yaml' of a run.

    Returns:
        bool: True if the setting was added, False if the file already had it or has no evaluation settings.

    Raises:
        ValueError: If the evaluation settings have no per-class report switch, add it first with
            add_per_class_report_to_configuration.
    """
    text = configuration_path.read_text()
    experiment_configuration = yaml.safe_load(stream=text)["ExperimentConfiguration"]
    if "evaluation" not in experiment_configuration or \
            "number_sample_images" in experiment_configuration["evaluation"]:
        return False

    # Line of the per-class report switch, the only one of the file
    lines = text.splitlines(keepends=True)
    report_index = next((line_index for line_index, line in enumerate(lines)
                         if line.startswith(PER_CLASS_REPORT_LINE_START)), None)
    if report_index is None:
        raise ValueError(f"No 'per_class_report' line in the evaluation settings of "
                         f"{format_path_link(path=configuration_path)}, add it first with "
                         f"add_per_class_report_to_configuration.")

    configuration_path.write_text(data="".join(lines[:report_index + 1] + [NUMBER_SAMPLE_IMAGES_LINES] +
                                               lines[report_index + 1:]))
    return True


def add_tracking_to_configuration(configuration_path: Path) -> bool:
    """
    Upgrades the configuration file of a run written before the MLflow tracking by inserting the commented
    'ExperimentConfiguration.tracking' section, off as in the runs of that time, right after the evaluation settings
    of the experiment section, so that the run can be resumed and its checkpoints evaluated. The section is inserted as
    text, so that every comment of the file is kept, and a file that already has it is left untouched.

    Args:
        configuration_path (Path): The configuration file, usually the 'training_configuration.yaml' of a run.

    Returns:
        bool: True if the section was added, False if the file already had it.

    Raises:
        ValueError: If the experiment section has no '  evaluation:' line, add it first with
            add_per_class_report_to_configuration.
    """
    text = configuration_path.read_text()
    if "tracking" in yaml.safe_load(stream=text)["ExperimentConfiguration"]:
        return False

    # Line of the evaluation settings, searched within the experiment section only
    lines = text.splitlines(keepends=True)
    evaluation_index = None
    for line_index in range(lines.index(EXPERIMENT_SECTION_HEADER) + 1, len(lines)):
        if lines[line_index][:1].isalpha():
            break
        if lines[line_index] == EVALUATION_SECTION_LINE:
            evaluation_index = line_index
            break
    if evaluation_index is None:
        raise ValueError(f"No '  evaluation:' line in the experiment section of "
                         f"{format_path_link(path=configuration_path)}, add it first with "
                         f"add_per_class_report_to_configuration.")

    # Last line of the evaluation settings: the section ends at the first line indented as the experiment keys (a key or
    # the comment of the next key), the blank lines before it staying after the inserted section
    insertion_index = evaluation_index + 1
    for line_index in range(evaluation_index + 1, len(lines)):
        if not lines[line_index].strip():
            continue
        if len(lines[line_index]) - len(lines[line_index].lstrip(" ")) <= 2:
            break
        insertion_index = line_index + 1

    configuration_path.write_text(data="".join(lines[:insertion_index] + [TRACKING_SECTION_LINES] +
                                               lines[insertion_index:]))
    return True


def add_sequence_augmentation_to_configuration(configuration_path: Path) -> bool:
    """
    Upgrades a configuration file written before the sequence augmentations by inserting
    'Data.data_augmentation.independent_frames' and 'Data.data_augmentation.shuffle_frames', off as in the runs of
    that time, right before the augmentation types. The settings are inserted as text, so that every comment of the
    file is kept, and a file that already has them is left untouched.

    Args:
        configuration_path (Path): The configuration file, usually the 'training_configuration.yaml' of a run.

    Returns:
        bool: True if the settings were added, False if the file already had them.

    Raises:
        ValueError: If the augmentation section has no 'types:' line of its own.
    """
    text = configuration_path.read_text()
    augmentation_configuration = yaml.safe_load(stream=text)["Data"]["data_augmentation"]
    if "independent_frames" in augmentation_configuration and "shuffle_frames" in augmentation_configuration:
        return False

    # Insert the settings right before the types of the augmentation section
    lines = text.splitlines(keepends=True)
    augmentation_index = lines.index(DATA_AUGMENTATION_LINE)
    types_index = next((line_index for line_index in range(augmentation_index, len(lines))
                        if lines[line_index] == AUGMENTATION_TYPES_LINE), None)
    if types_index is None:
        raise ValueError(f"No 'types:' line in the augmentation section of {format_path_link(path=configuration_path)}")
    configuration_path.write_text(data="".join(lines[:types_index] + [SEQUENCE_AUGMENTATION_LINES] +
                                               lines[types_index:]))
    return True


def add_sequence_normalization_scope_to_configuration(configuration_path: Path) -> bool:
    """
    Upgrades the configuration file of a comparison run written before the sequence normalization by inserting
    'Fusion.sequence_normalization_scope', every member normalised by its own frames, right below the header of its
    fusion section. The setting is inserted as text, so that every comment of the file is kept, and a file that
    already has it, or that has no fusion section (the other tasks), is left untouched.

    Args:
        configuration_path (Path): The configuration file, usually the 'training_configuration.yaml' of a run.

    Returns:
        bool: True if the setting was added, False if the file already had it or is not a comparison run.

    Raises:
        ValueError: If the fusion section exists but its header is not a line of its own.
    """
    text = configuration_path.read_text()
    configuration = yaml.safe_load(stream=text)
    if "Fusion" not in configuration or "sequence_normalization_scope" in configuration["Fusion"]:
        return False
    lines = text.splitlines(keepends=True)
    if FUSION_SECTION_HEADER not in lines:
        raise ValueError(f"No 'Fusion:' section header in {format_path_link(path=configuration_path)}")

    # Insert the setting below the header line
    header_index = lines.index(FUSION_SECTION_HEADER)
    configuration_path.write_text(data="".join(lines[:header_index + 1] + [SEQUENCE_NORMALIZATION_SCOPE_LINES] +
                                               lines[header_index + 1:]))
    return True


def add_loading_settings_to_configuration(configuration_path: Path) -> bool:
    """
    Upgrades a configuration file written before the loading settings by inserting 'Data.loading', with the 4 worker
    processes of the loaders of that time, right below the header of its data section. The settings are inserted as
    text, so that every comment of the file is kept, and a file that already has them is left untouched.

    Args:
        configuration_path (Path): The configuration file, usually the 'training_configuration.yaml' of a run.

    Returns:
        bool: True if the settings were added, False if the file already had them.

    Raises:
        ValueError: If the data section has no header line of its own.
    """
    text = configuration_path.read_text()
    if "loading" in yaml.safe_load(stream=text)["Data"]:
        return False
    lines = text.splitlines(keepends=True)
    if DATA_SECTION_HEADER not in lines:
        raise ValueError(f"No 'Data:' section header in {format_path_link(path=configuration_path)}")

    # Insert the settings below the header line
    header_index = lines.index(DATA_SECTION_HEADER)
    configuration_path.write_text(data="".join(lines[:header_index + 1] + [LOADING_SETTINGS_LINES] +
                                               lines[header_index + 1:]))
    return True


def add_batch_padding_to_configuration(configuration_path: Path) -> bool:
    """
    Upgrades the configuration file of a sequence run written before the padding of the batches to their longest
    window by inserting 'Data.sequence_settings.batch_padding', every window padded to the sequence length as in the
    runs of that time, right after the 'short_sequences' setting. The setting is inserted as text, so that every
    comment of the file is kept, and a file that already has it, or that has no sequence settings (the image tasks), is
    left untouched.

    Args:
        configuration_path (Path): The configuration file, usually the 'training_configuration.yaml' of a run.

    Returns:
        bool: True if the setting was added, False if the file already had it or is not a sequence run.

    Raises:
        ValueError: If the sequence settings have no 'short_sequences' line of their own.
    """
    text = configuration_path.read_text()
    data_configuration = yaml.safe_load(stream=text)["Data"]
    if "sequence_settings" not in data_configuration or "batch_padding" in data_configuration["sequence_settings"]:
        return False

    # Insert the setting right after the line of the short sequences
    lines = text.splitlines(keepends=True)
    short_sequences_index = next((line_index for line_index, line in enumerate(lines)
                                  if line.startswith(SHORT_SEQUENCES_LINE_START)), None)
    if short_sequences_index is None:
        raise ValueError(f"No 'short_sequences:' line in the sequence settings of "
                         f"{format_path_link(path=configuration_path)}")
    configuration_path.write_text(data="".join(lines[:short_sequences_index + 1] + [BATCH_PADDING_LINES] +
                                               lines[short_sequences_index + 1:]))
    return True


def add_positional_embedding_to_configuration(configuration_path: Path) -> bool:
    """
    Upgrades the configuration file of a sequence run written before the order-blind transformer by inserting
    'TemporalModel.positional_embedding', the transformer giving every frame its position as in the runs of that time,
    right below the header of its temporal module section. The setting is inserted as text, so that every comment of the
    file is kept, and a file that already has it, or that has no temporal module (the image tasks), is left untouched.

    Args:
        configuration_path (Path): The configuration file, usually the 'training_configuration.yaml' of a run.

    Returns:
        bool: True if the setting was added, False if the file already had it or is not a sequence run.

    Raises:
        ValueError: If the temporal module section exists but its header is not a line of its own.
    """
    text = configuration_path.read_text()
    configuration = yaml.safe_load(stream=text)
    if "TemporalModel" not in configuration or "positional_embedding" in configuration["TemporalModel"]:
        return False
    lines = text.splitlines(keepends=True)
    if TEMPORAL_SECTION_HEADER not in lines:
        raise ValueError(f"No 'TemporalModel:' section header in {format_path_link(path=configuration_path)}")

    # Insert the setting below the header line
    header_index = lines.index(TEMPORAL_SECTION_HEADER)
    configuration_path.write_text(data="".join(lines[:header_index + 1] + [POSITIONAL_EMBEDDING_LINES] +
                                               lines[header_index + 1:]))
    return True


def add_segmentation_settings_to_configuration(configuration_path: Path) -> bool:
    """
    Upgrades the configuration file of a semantic segmentation run written before the settings added since its first
    runs (SEGMENTATION_SETTINGS_INSERTIONS: the training on tiles, the deep supervision, the Lovász and boundary
    losses, the evaluation at the original resolution and the boundary tolerance) by inserting every missing one, off
    as in the runs of that time, right below the header of its section. The settings are inserted as text, so that
    every comment of the file is kept, and a file that already has them, or that is not a segmentation run, is left
    untouched.

    Args:
        configuration_path (Path): The configuration file, usually the 'training_configuration.yaml' of a run.

    Returns:
        bool: True if at least one setting was added, False if the file had them all or is not a segmentation run.

    Raises:
        ValueError: If the section of a missing setting has no header line of its own.
    """
    text = configuration_path.read_text()
    if "SegmentationDecoder" not in yaml.safe_load(stream=text):
        return False

    # Insert every missing setting below the header of its section, reading the file again after every insertion
    settings_added = False
    for section_path, key_name, header_line, inserted_lines in SEGMENTATION_SETTINGS_INSERTIONS:
        section = yaml.safe_load(stream=text)
        for section_name in section_path:
            section = section[section_name]
        if key_name in section:
            continue
        lines = text.splitlines(keepends=True)
        if header_line not in lines:
            raise ValueError(f"No {header_line.strip()!r} header line for '{key_name}' in "
                             f"{format_path_link(path=configuration_path)}")
        header_index = lines.index(header_line)
        text = "".join(lines[:header_index + 1] + [inserted_lines] + lines[header_index + 1:])
        settings_added = True

    if settings_added:
        configuration_path.write_text(data=text)
    return settings_added


def add_optimization_section_to_configuration(configuration_path: Path) -> bool:
    """
    Upgrades a configuration file written before the 'Optimization' section by appending the section of a run
    trained with plain Adam at a constant learning rate, the only optimization of the code of that time. The section
    is appended as text, so that every comment of the file is kept, and a file that already has it is left untouched.

    Args:
        configuration_path (Path): The configuration file, usually the 'training_configuration.yaml' of a run.

    Returns:
        bool: True if the section was added, False if the file already had it.
    """
    text = configuration_path.read_text()
    if "Optimization" in yaml.safe_load(stream=text):
        return False
    configuration_path.write_text(data=text.rstrip("\n") + "\n" + OPTIMIZATION_SECTION)
    return True


def remove_dali_settings_from_configuration(configuration_path: Path) -> bool:
    """
    Upgrades a configuration file written when the NVIDIA DALI loader existed by removing its 'Data.DALI' block (the
    header, its comments and its 'activated' flag), every task now reading its data with its PyTorch loader. The block
    is removed as text, so that every other comment of the file is kept, and a file without it is left untouched.

    Args:
        configuration_path (Path): The configuration file, usually the 'training_configuration.yaml' of a run.

    Returns:
        bool: True if the block was removed, False if the file had none.

    Raises:
        ValueError: If the block has no header line of its own or no 'activated' line.
    """
    text = configuration_path.read_text()
    if "DALI" not in yaml.safe_load(stream=text)["Data"]:
        return False
    lines = text.splitlines(keepends=True)
    if DALI_SECTION_LINE not in lines:
        raise ValueError(f"No '  DALI:' line in {format_path_link(path=configuration_path)}")

    # Remove the lines from the header to the activation flag, its last line
    start_index = lines.index(DALI_SECTION_LINE)
    end_index = next((index for index in range(start_index + 1, len(lines))
                      if lines[index].startswith(DALI_ACTIVATED_LINE_START)), None)
    if end_index is None:
        raise ValueError(f"No 'activated' line in the DALI block of {format_path_link(path=configuration_path)}")
    configuration_path.write_text(data="".join(lines[:start_index] + lines[end_index + 1:]))
    return True


def migrate_run_folder(run_folder: Path, task: str, rename_folder: bool) -> Path:
    """
    Upgrades a run trained with an older version of the code so that it can be resumed, evaluated and exported with
    the current code: its saved configuration gets its task, the 'pretrained' settings of its backbone, temporal
    module and classification head (those of a run trained from random weights), the
    'Backbone.pretrained.freeze_normalization_parameters' flag (off), the evaluation outputs of the classification-like
    tasks (no predictions file nor error images), the per-class report switch of every task (on, the per-class
    report of that time), the number of sample images (none), the MLflow tracking settings (off), the sequence
    augmentations (off), the scope of the sequence normalization of the comparison task (per member), its
    'Optimization' section (plain Adam at a constant learning rate), its loading settings (4 worker processes), the
    padding of the batches of the sequence tasks (every window padded to the sequence length), the positional
    embedding of their transformer (on) and the settings of the segmentation added since its first runs (off), it loses
    the 'Data.DALI' block of the removed DALI loader, and its
    folder is renamed to the run name the current code builds from that configuration (the task code being now the
    first section of the run names, and the augmentation, batch size, learning rate, schedule, gradient clipping, mixed
    precision, dropout and learning rate multiplier (BLR, TLR, HLR, DLR) tokens being no longer written). The
    configuration is fully checked on the way, any other missing key being reported by the configuration checker. Two
    old folders differing only by these settings get the same new name, the second one being refused.

    Args:
        run_folder (Path): The run folder, holding 'training_configuration.yaml', 'Weights' and 'Tensorboard'.
        task (str): The task the run was trained for, e.g. "object_detection".
        rename_folder (bool): True to rename the folder, False to only print the new name (a folder must never be
            renamed while a training writes into it).

    Returns:
        Path: The run folder after the migration (renamed or not).

    Raises:
        FileExistsError: If a folder already has the new run name.
    """
    configuration_path = run_folder / "training_configuration.yaml"
    if add_task_to_configuration(configuration_path=configuration_path, task=task):
        print_color(color="green", output=f"Task {task!r} Added To : {format_path_link(path=configuration_path)}")
    else:
        print_color(color="blue", output=f"Task Already Set In : {format_path_link(path=configuration_path)}")
    for section_name in PRETRAINED_SETTINGS_BLOCKS:
        if add_pretrained_settings_to_configuration(configuration_path=configuration_path, section_name=section_name):
            print_color(color="green", output=f"Pretrained {section_name} Settings Added To : "
                                              f"{format_path_link(path=configuration_path)}")
    if add_freeze_normalization_parameters_to_configuration(configuration_path=configuration_path):
        print_color(color="green", output=f"Backbone Normalization Freeze Flag Added To : "
                                          f"{format_path_link(path=configuration_path)}")
    if add_evaluation_outputs_to_configuration(configuration_path=configuration_path):
        print_color(color="green", output=f"Evaluation Outputs Settings Added To : "
                                          f"{format_path_link(path=configuration_path)}")
    if add_per_class_report_to_configuration(configuration_path=configuration_path):
        print_color(color="green", output=f"Per-Class Report Switch Added To : "
                                          f"{format_path_link(path=configuration_path)}")
    if add_sample_images_to_configuration(configuration_path=configuration_path):
        print_color(color="green", output=f"Number Of Sample Images Added To : "
                                          f"{format_path_link(path=configuration_path)}")
    if add_tracking_to_configuration(configuration_path=configuration_path):
        print_color(color="green", output=f"MLflow Tracking Settings Added To : "
                                          f"{format_path_link(path=configuration_path)}")
    if add_sequence_augmentation_to_configuration(configuration_path=configuration_path):
        print_color(color="green", output=f"Sequence Augmentation Settings Added To : "
                                          f"{format_path_link(path=configuration_path)}")
    if add_sequence_normalization_scope_to_configuration(configuration_path=configuration_path):
        print_color(color="green", output=f"Sequence Normalization Scope Added To : "
                                          f"{format_path_link(path=configuration_path)}")
    if add_optimization_section_to_configuration(configuration_path=configuration_path):
        print_color(color="green", output=f"Optimization Section Added To : "
                                          f"{format_path_link(path=configuration_path)}")
    if add_loading_settings_to_configuration(configuration_path=configuration_path):
        print_color(color="green", output=f"Loading Settings Added To : {format_path_link(path=configuration_path)}")
    if add_batch_padding_to_configuration(configuration_path=configuration_path):
        print_color(color="green", output=f"Batch Padding Of The Sequence Windows Added To : "
                                          f"{format_path_link(path=configuration_path)}")
    if add_positional_embedding_to_configuration(configuration_path=configuration_path):
        print_color(color="green", output=f"Positional Embedding Flag Of The Transformer Added To : "
                                          f"{format_path_link(path=configuration_path)}")
    if add_segmentation_settings_to_configuration(configuration_path=configuration_path):
        print_color(color="green", output=f"Segmentation Losses, Deep Supervision, Tiles And Evaluation Settings "
                                          f"Added To : {format_path_link(path=configuration_path)}")
    if remove_dali_settings_from_configuration(configuration_path=configuration_path):
        print_color(color="green", output=f"DALI Loader Settings Removed From : "
                                          f"{format_path_link(path=configuration_path)}")

    # The run name the current code gives to this configuration
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    new_run_folder = run_folder.parent / build_run_name(experiment_configuration=experiment_configuration,
                                                        model_configuration=model_configuration)
    if new_run_folder == run_folder:
        print_color(color="blue",
                    output=f"Run Folder Already Named After Its Settings : {format_path_link(path=run_folder)}")
        return run_folder

    if not rename_folder:
        print_color(color="yellow", output=f"Run Folder To Rename (rename_folder is False) : "
                                           f"{format_path_link(path=run_folder)}\n  - New Name : {new_run_folder.name}")
        return run_folder
    if new_run_folder.exists():
        raise FileExistsError(f"A folder already has the new run name : {format_path_link(path=new_run_folder)}")

    run_folder.rename(new_run_folder)
    print_color(color="green", output=f"Run Folder Renamed : {format_path_link(path=new_run_folder)}")
    return new_run_folder
