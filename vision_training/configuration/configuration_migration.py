import yaml

from pathlib import Path

from vision_training.configuration.configuration_loader import load_experiment_configuration
from vision_training.configuration.run_name import build_run_name
from vision_training.utilities.os_utilities import print_color

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
                           "    learning_rate_multiplier: 1.0\n")}

# Section header of the backbone, start of the line of the freeze flag of its pretrained settings, and the flag keeping
# its normalization parameters fixed inserted after it, off in every run written before it existed
BACKBONE_SECTION_HEADER = "Backbone:\n"
PRETRAINED_FREEZE_LINE_START = "    freeze:"
FREEZE_NORMALIZATION_LINES = ("    # Keeps the scale and shift of the normalization layers of the loaded backbone "
                              "fixed while its convolutions learn.\n"
                              "    freeze_normalization_parameters: False\n")

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
                         f"file://{configuration_path.absolute()}")

    configuration_path.write_text(data=text.replace(
        EXPERIMENT_SECTION_HEADER, f'{EXPERIMENT_SECTION_HEADER}{TASK_KEY_COMMENT}  task: "{task}"\n', 1))
    return True


def add_pretrained_settings_to_configuration(configuration_path: Path, section_name: str) -> bool:
    """
    Upgrades a configuration file written before the transfer of a component by inserting the 'pretrained' settings
    of its section (Backbone, TemporalModel or ClassificationHead), those of a run trained from random weights, right
    below the section header. The settings are inserted as text, so that every comment of the file is kept, and a
    file that already has them, or that has no such section (e.g. no temporal module in a detection configuration),
    is left untouched.

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
        raise ValueError(f"No '{section_name}:' section header in file://{configuration_path.absolute()}")

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
        raise ValueError(f"No 'Backbone.pretrained.freeze' flag in file://{configuration_path.absolute()}, add the "
                         f"pretrained settings first with add_pretrained_settings_to_configuration.")

    configuration_path.write_text(data="".join(lines[:freeze_index + 1] + [FREEZE_NORMALIZATION_LINES] +
                                               lines[freeze_index + 1:]))
    return True


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


def migrate_run_folder(run_folder: Path, task: str, rename_folder: bool) -> Path:
    """
    Upgrades a run trained with an older version of the code so that it can be resumed, evaluated and exported with
    the current code: its saved configuration gets its task, the 'pretrained' settings of its backbone, temporal
    module and classification head (those of a run trained from random weights), the
    'Backbone.pretrained.freeze_normalization_parameters' flag (off) and its 'Optimization' section (plain Adam at a
    constant learning rate), and its folder is renamed to the run name the current code builds from that
    configuration (the task code being now the first section of the run names). The configuration is fully checked
    on the way, any other missing key being reported by the configuration checker.

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
        print_color(color="green", output=f"Task {task!r} Added To : file://{configuration_path.absolute()}")
    else:
        print_color(color="blue", output=f"Task Already Set In : file://{configuration_path.absolute()}")
    for section_name in PRETRAINED_SETTINGS_BLOCKS:
        if add_pretrained_settings_to_configuration(configuration_path=configuration_path, section_name=section_name):
            print_color(color="green", output=f"Pretrained {section_name} Settings Added To : "
                                              f"file://{configuration_path.absolute()}")
    if add_freeze_normalization_parameters_to_configuration(configuration_path=configuration_path):
        print_color(color="green", output=f"Backbone Normalization Freeze Flag Added To : "
                                          f"file://{configuration_path.absolute()}")
    if add_optimization_section_to_configuration(configuration_path=configuration_path):
        print_color(color="green", output=f"Optimization Section Added To : file://{configuration_path.absolute()}")

    # The run name the current code gives to this configuration
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    new_run_folder = run_folder.parent / build_run_name(experiment_configuration=experiment_configuration,
                                                        model_configuration=model_configuration)
    if new_run_folder == run_folder:
        print_color(color="blue",
                    output=f"Run Folder Already Named After Its Settings : file://{run_folder.absolute()}")
        return run_folder

    if not rename_folder:
        print_color(color="yellow", output=f"Run Folder To Rename (rename_folder is False) : "
                                           f"file://{run_folder.absolute()}\n  - New Name : {new_run_folder.name}")
        return run_folder
    if new_run_folder.exists():
        raise FileExistsError(f"A folder already has the new run name : file://{new_run_folder.absolute()}")

    run_folder.rename(new_run_folder)
    print_color(color="green", output=f"Run Folder Renamed : file://{new_run_folder.absolute()}")
    return new_run_folder
