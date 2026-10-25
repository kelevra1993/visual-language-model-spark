"""
Reference check of the storyboards, run first by every build and on its own in seconds (build_tutorials.py
--check-references): every storyboard is loaded without its figures, which reads every configuration and code excerpt
it quotes and checks every highlight pattern, then the file and function of every "where in the code" chip, the modules
and configuration files of the terminal commands and the extra references of the storyboard are looked up in the
repository. All the problems of all the storyboards are listed at once, so that one run after a change of the
repository shows every tutorial it made wrong.
"""

import ast
import re

from pathlib import Path
from typing import Any, Dict, List

from tutorial_builder.settings import REPOSITORY_ROOT, BuildSettings
from tutorial_builder.storyboard_loading import find_python_definition, find_yaml_key_lines, load_storyboard

# Separator of a file and the name inside it, in the chips and references: "path/to/file.py · Class.method"
REFERENCE_SEPARATOR = " · "

# Module run by a terminal command, and configuration file given to it
MODULE_PATTERN = re.compile(pattern=r"python(?:3)?\s+-m\s+([\w.]+)")
CONFIGURATION_PATTERN = re.compile(pattern=r"--configuration_path=\"?([^\"\s]+)\"?")


def check_named_reference(reference: str, context: str) -> List[str]:
    """
    Checks a reference to a file of the repository, optionally followed by a name inside it: a function or class
    (Class.method) of a Python file, a key path (Optimization.learning_rate_schedule) of a YAML file.

    Args:
        reference (str): The reference, "path" or "path · name".
        context (str): Where the reference is, for the messages.

    Returns:
        List[str]: The problems found, empty when the reference holds.
    """
    path_text, _, name = reference.partition(REFERENCE_SEPARATOR)
    file_path = REPOSITORY_ROOT / path_text.strip()
    if not file_path.exists():
        return [f"{context}: no file {path_text.strip()} in the repository (reference {reference!r})."]
    name = name.strip()
    if not name or not re.fullmatch(pattern=r"[A-Za-z_][\w.]*(\(\))?", string=name):
        return []
    name = name.removesuffix("()")

    # A Python definition by its qualified name, or a YAML key by its path
    if file_path.suffix == ".py":
        try:
            find_python_definition(tree=ast.parse(source=file_path.read_text(encoding="utf-8")), qualified_name=name)
        except KeyError:
            return [f"{context}: no definition {name} in {path_text.strip()}."]
    elif file_path.suffix in (".yaml", ".yml"):
        paths = {key_line["path"] for key_line in find_yaml_key_lines(
            lines=file_path.read_text(encoding="utf-8").splitlines())}
        if name not in paths:
            return [f"{context}: no key {name} in {path_text.strip()}."]
    return []


def check_terminal_commands(scene: Dict[str, Any], context: str) -> List[str]:
    """
    Checks the commands typed in a terminal scene: every module run with python -m exists as a file or package of the
    repository, and every configuration given to --configuration_path exists.

    Args:
        scene (Dict[str, Any]): The terminal scene.
        context (str): Where the scene is, for the messages.

    Returns:
        List[str]: The problems found.
    """
    problems = []
    for line in scene["lines"]:
        if line.get("role") != "command":
            continue
        for module_name in MODULE_PATTERN.findall(string=line["text"]):
            module_path = REPOSITORY_ROOT / Path(*module_name.split("."))
            if not module_path.with_suffix(".py").exists() and not module_path.is_dir() and \
                    not module_name.startswith(("pytest", "tutorial_builder")):
                problems.append(f"{context}: the command runs the module {module_name}, which is not in the "
                                f"repository.")
        for configuration_path in CONFIGURATION_PATTERN.findall(string=line["text"]):
            if not (REPOSITORY_ROOT / configuration_path).exists() and not Path(configuration_path).exists():
                problems.append(f"{context}: the command gives the configuration {configuration_path}, which does "
                                f"not exist.")
    return problems


def check_storyboard_references(storyboard_paths: List[Path], settings: BuildSettings) -> List[str]:
    """
    Checks every reference of the storyboards to the repository and lists the problems of all of them.

    Args:
        storyboard_paths (List[Path]): The storyboards to check.
        settings (BuildSettings): The settings of the build.

    Returns:
        List[str]: The problems found, empty when every storyboard holds.
    """
    problems = []
    for storyboard_path in storyboard_paths:
        # Loading reads every excerpt and checks every highlight, page and repository image
        try:
            storyboard = load_storyboard(storyboard_path=storyboard_path, settings=settings, figures_required=False)
        except (KeyError, ValueError, FileNotFoundError, SyntaxError) as error:
            problems.append(f"{storyboard_path.name}: {error}")
            continue

        # Chips of the scenes, commands of the terminals and extra references of the storyboard
        for scene_index, scene in enumerate(storyboard["scenes"]):
            context = f"{storyboard_path.name}, scene {scene_index + 1} ({scene.get('title', scene['kind'])})"
            if scene.get("code_reference"):
                problems += check_named_reference(reference=scene["code_reference"], context=context)
            if scene["kind"] == "terminal":
                problems += check_terminal_commands(scene=scene, context=context)
        for reference in storyboard.get("references", []):
            problems += check_named_reference(reference=reference, context=storyboard_path.name)
    return problems
