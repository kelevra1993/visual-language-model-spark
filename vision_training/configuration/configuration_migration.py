import yaml

from pathlib import Path

from vision_training.configuration.configuration_loader import load_experiment_configuration
from vision_training.configuration.run_name import build_run_name
from vision_training.utilities.os_utilities import print_color

# Section header under which the task key is inserted, and the comment written above it
EXPERIMENT_SECTION_HEADER = "ExperimentConfiguration:\n"
TASK_KEY_COMMENT = ("  # Task trained by this configuration, selecting the trainer, the model and the checks of "
                    "the task sections.\n")

# Section header under which the pretrained backbone settings are inserted, and the settings of a run trained from
# random weights
BACKBONE_SECTION_HEADER = "Backbone:\n"
PRETRAINED_BACKBONE_BLOCK = ("  # Backbone of another training run initialising this run, null to start from random "
                             "weights.\n"
                             "  pretrained:\n"
                             "    experiment_folder: null\n"
                             "    iteration: null\n"
                             "    freeze: False\n"
                             "    learning_rate_multiplier: 1.0\n")

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


def add_pretrained_backbone_to_configuration(configuration_path: Path) -> bool:
    """
    Upgrades a configuration file written before the backbone transfer by inserting its 'Backbone.pretrained'
    settings, those of a run trained from random weights, right below the section header. The settings are inserted
    as text, so that every comment of the file is kept, and a file that already has them is left untouched.

    Args:
        configuration_path (Path): The configuration file, usually the 'training_configuration.yaml' of a run.

    Returns:
        bool: True if the settings were added, False if the file already had them.

    Raises:
        ValueError: If the file has no 'Backbone:' section header.
    """
    text = configuration_path.read_text()
    if "pretrained" in yaml.safe_load(stream=text)["Backbone"]:
        return False
    if BACKBONE_SECTION_HEADER not in text.splitlines(keepends=True):
        raise ValueError(f"No '{BACKBONE_SECTION_HEADER.strip()}' section header in "
                         f"file://{configuration_path.absolute()}")

    # Insert the settings below the header line, the only line made of the header alone
    lines = text.splitlines(keepends=True)
    header_index = lines.index(BACKBONE_SECTION_HEADER)
    configuration_path.write_text(data="".join(lines[:header_index + 1] + [PRETRAINED_BACKBONE_BLOCK] +
                                               lines[header_index + 1:]))
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
    Upgrades a run trained before the task-aware configuration so that it can be resumed, evaluated and exported with
    the current code: its saved configuration gets its task, its 'Backbone.pretrained' settings (those of a run
    trained from random weights) and its 'Optimization' section (plain Adam at a constant learning rate), and its
    folder is renamed to the run name the current code builds from that configuration (the task code being now the
    first section of the run names). The configuration is fully checked
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
    if add_pretrained_backbone_to_configuration(configuration_path=configuration_path):
        print_color(color="green", output=f"Pretrained Backbone Settings Added To : "
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
