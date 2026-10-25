"""
Tests of tutorials/tutorial_builder/storyboard_loading.py and reference_check.py, the part of the tutorial build that
keeps the videos in step with the repository: the configuration and code excerpts read from the files (YAML blocks
under their parent keys, Python definitions by their name, narrowed by patterns), the highlights that must match a
line, the facts of the figure scripts filled in the texts, and the reference check listing what a storyboard quotes
that no longer exists.
"""

from pathlib import Path

import pytest
import yaml

from tutorial_builder.reference_check import check_storyboard_references
from tutorial_builder.settings import BuildSettings, create_build_settings
from tutorial_builder.storyboard_loading import (extract_python_excerpt, extract_yaml_excerpt, fill_facts,
                                                 load_storyboard, resolve_excerpt)

# Smoke storyboard quoting real files of the repository, one scene of every kind
SMOKE_STORYBOARD_PATH = Path(__file__).parent / "fixtures" / "smoke_storyboard.yaml"

# Configuration excerpted by the tests, with comments, nested keys, a flow mapping and skipped keys
TEST_CONFIGURATION = """# Header comment of the file
Experiment:
  # Comment of the task
  task: "classification"
  batch_size: 64
  number_iterations: 1e10
Backbone:
  convolutions:
    {
      "1": [ 1, 16 ],
      "2": [ 2, 32 ],
    }
  modules:
    residual_nets: True
"""

# Python module excerpted by the tests
TEST_MODULE = '''"""Module docstring."""


class Block:
    """A block."""

    def count(self) -> int:
        """
        Counts something.
        """
        first = 1
        second = 2
        return first + second
'''


def create_test_settings(cache_folder: Path) -> BuildSettings:
    """
    Gives build settings whose cache is a temporary folder, so that the tests read no figure of a real cache.

    Args:
        cache_folder (Path): The temporary cache folder.

    Returns:
        BuildSettings: The settings.
    """
    return create_build_settings(output_folder=str(cache_folder / "output"), cache_folder=str(cache_folder),
                                 experiments_folder=str(cache_folder / "experiments"), repository_python=None,
                                 chrome_executable=None)


def test_yaml_excerpt_keeps_parents_and_marks_skipped_lines(tmp_path: Path) -> None:
    """
    Verifies that a YAML excerpt shows every block under its parent keys, the comment of a key above it when comments
    are kept, "..." where lines are skipped, and a flow mapping as part of its key, not as keys of its own.
    """
    configuration_path = tmp_path / "configuration.yaml"
    configuration_path.write_text(data=TEST_CONFIGURATION, encoding="utf-8")

    lines = extract_yaml_excerpt(file_path=configuration_path, keys=["Experiment.task", "Experiment.number_iterations",
                                                                     "Backbone.modules"], keep_comments=True)
    assert lines == ["Experiment:", "  # Comment of the task", "  task: \"classification\"", "  ...",
                     "  number_iterations: 1e10", "Backbone:", "  ...", "  modules:", "    residual_nets: True"]

    # Without comments, and a flow mapping kept whole under its key
    lines = extract_yaml_excerpt(file_path=configuration_path, keys=["Backbone.convolutions"], keep_comments=False)
    assert lines == ["Backbone:", "  convolutions:", "    {", "      \"1\": [ 1, 16 ],", "      \"2\": [ 2, 32 ],",
                     "    }"]
    with pytest.raises(expected_exception=KeyError):
        extract_yaml_excerpt(file_path=configuration_path, keys=["Experiment.missing"], keep_comments=False)


def test_yaml_excerpt_without_comments_drops_blank_lines(tmp_path: Path) -> None:
    """
    Verifies that a YAML excerpt without its comments also leaves out the blank lines that separated the commented
    keys of a block, as in the configuration files of the repository, so that the excerpt fits the card, while the
    excerpt with its comments keeps them as in the file.
    """
    configuration_path = tmp_path / "configuration.yaml"
    configuration_path.write_text(data="Backbone:\n  modules:\n    # Shortcuts\n    residual_nets: True\n\n"
                                       "    # Gates\n    squeeze_and_excite: True\n", encoding="utf-8")
    lines = extract_yaml_excerpt(file_path=configuration_path, keys=["Backbone.modules"], keep_comments=False)
    assert lines == ["Backbone:", "  modules:", "    residual_nets: True", "    squeeze_and_excite: True"]
    lines = extract_yaml_excerpt(file_path=configuration_path, keys=["Backbone.modules"], keep_comments=True)
    assert lines == ["Backbone:", "  modules:", "    # Shortcuts", "    residual_nets: True", "",
                     "    # Gates", "    squeeze_and_excite: True"]


def test_python_excerpt_by_name(tmp_path: Path) -> None:
    """
    Verifies that a Python excerpt is found by its qualified name with its docstring left out, and narrowed between two
    patterns after its signature, "..." standing for the lines skipped before and after.
    """
    module_path = tmp_path / "module.py"
    module_path.write_text(data=TEST_MODULE, encoding="utf-8")

    lines = extract_python_excerpt(file_path=module_path, qualified_name="Block.count", keep_docstring=False,
                                   start_pattern=None, end_pattern=None)
    assert lines == ["def count(self) -> int:", "    first = 1", "    second = 2", "    return first + second"]

    # Narrowed to one line, with the signature and the skipped lines marked
    lines = extract_python_excerpt(file_path=module_path, qualified_name="Block.count", keep_docstring=False,
                                   start_pattern="second", end_pattern="second")
    assert lines == ["def count(self) -> int:", "    ...", "    second = 2", "    ..."]
    with pytest.raises(expected_exception=KeyError):
        extract_python_excerpt(file_path=module_path, qualified_name="Block.missing", keep_docstring=False,
                               start_pattern=None, end_pattern=None)


def test_python_excerpt_and_reference_of_a_constant(tmp_path: Path) -> None:
    """
    Verifies that a constant of a module (a table assigned to a single name, with or without annotation) is found by
    its name like a function, whole or narrowed between two patterns, so that the excerpts and chips can name the type
    lists and strength tables of the repository.
    """
    module_path = tmp_path / "module.py"
    module_path.write_text(data='"""Module."""\n\nLIMITS: dict = {\n    "low": 1,\n    "high": 2}\n\n'
                                'NAMES = ["first", "second"]\n', encoding="utf-8")
    lines = extract_python_excerpt(file_path=module_path, qualified_name="LIMITS", keep_docstring=False,
                                   start_pattern=None, end_pattern=None)
    assert lines == ["LIMITS: dict = {", "    \"low\": 1,", "    \"high\": 2}"]
    lines = extract_python_excerpt(file_path=module_path, qualified_name="LIMITS", keep_docstring=False,
                                   start_pattern="high", end_pattern="high")
    assert lines == ["LIMITS: dict = {", "    ...", "    \"high\": 2}"]
    lines = extract_python_excerpt(file_path=module_path, qualified_name="NAMES", keep_docstring=False,
                                   start_pattern=None, end_pattern=None)
    assert lines == ["NAMES = [\"first\", \"second\"]"]


def test_highlight_must_match_a_line(tmp_path: Path) -> None:
    """
    Verifies that the highlights of an excerpt mark the lines they match for their sentences, and that a highlight
    matching no line stops the loading, so that a renamed key cannot silently highlight nothing.
    """
    configuration_path = tmp_path / "configuration.yaml"
    configuration_path.write_text(data=TEST_CONFIGURATION, encoding="utf-8")
    settings = create_test_settings(cache_folder=tmp_path / "cache")
    source = {"file": str(configuration_path), "keys": ["Experiment.batch_size"], "comments": False}

    lines = resolve_excerpt(source=source, highlights=[{"match": "batch_size", "sentences": [1]}], settings=settings,
                            context="test", figures_required=True)
    assert [line["text"] for line in lines] == ["Experiment:", "  ...", "  batch_size: 64"]
    assert [line["highlight_sentences"] for line in lines] == [[], [], [1]]
    with pytest.raises(expected_exception=KeyError):
        resolve_excerpt(source=source, highlights=[{"match": "learning_rate", "sentences": [0]}], settings=settings,
                        context="test", figures_required=True)


def test_facts_are_filled_in_the_texts() -> None:
    """
    Verifies that the placeholders of the facts are formatted in a text, that a placeholder alone takes the fact as it
    is (a box), that a missing fact stops the loading, and that the facts of figures not built stay as placeholders.
    """
    facts = {"classification": {"test_accuracy": 0.98765, "panel_box": [1, 2, 3, 4]}}
    assert fill_facts(value="accuracy {{classification.test_accuracy:.1%}}", facts=facts,
                      context="test") == "accuracy 98.8%"
    assert fill_facts(value="{{classification.panel_box}}", facts=facts, context="test") == [1, 2, 3, 4]
    assert fill_facts(value=["{{classification.test_accuracy:.2f}}"], facts=facts, context="test") == ["0.99"]
    with pytest.raises(expected_exception=KeyError):
        fill_facts(value="{{classification.missing}}", facts=facts, context="test")
    assert fill_facts(value="{{detection.mean_average_precision:.2f}}", facts={"detection": None},
                      context="test") == "{{detection.mean_average_precision:.2f}}"


def test_facts_index_one_item_of_a_list() -> None:
    """
    Verifies that a placeholder can name one item of a list fact by its indices, alone (the box of one panel of a
    figure, kept as a list) or formatted in a text, and that an index outside the list stops the loading.
    """
    facts = {"classification": {"panel_boxes": [[1, 2, 3, 4], [5, 6, 7, 8]], "matrix": [[0.1, 0.2], [0.3, 0.4]]}}
    assert fill_facts(value="{{classification.panel_boxes[1]}}", facts=facts, context="test") == [5, 6, 7, 8]
    assert fill_facts(value="cell {{classification.matrix[1][0]:.2f}}", facts=facts, context="test") == "cell 0.30"
    with pytest.raises(expected_exception=IndexError):
        fill_facts(value="{{classification.panel_boxes[2]}}", facts=facts, context="test")
    assert fill_facts(value="{{detection.boxes[0]}}", facts={"detection": None}, context="test") == \
        "{{detection.boxes[0]}}"


def test_smoke_storyboard_loads_with_its_excerpts(tmp_path: Path) -> None:
    """
    Verifies that the smoke storyboard, one scene of every kind, loads with the excerpts of its yaml and code scenes
    read from the repository, their highlights placed, and its series label completed.
    """
    storyboard = load_storyboard(storyboard_path=SMOKE_STORYBOARD_PATH,
                                 settings=create_test_settings(cache_folder=tmp_path), figures_required=False)
    scenes_by_kind = {scene["kind"]: scene for scene in storyboard["scenes"]}
    assert set(scenes_by_kind) == {"title_card", "yaml", "code", "terminal", "html", "text_panel", "image", "summary"}
    code_lines = [line["text"] for line in scenes_by_kind["code"]["lines"]]
    assert code_lines[0] == "def count_residual_additions(self) -> int:"
    assert any(line["highlight_sentences"] == [0] for line in scenes_by_kind["code"]["lines"])
    assert storyboard["series_label"].startswith("Smoke Test · 1/")


def test_reference_check_lists_what_no_longer_exists(tmp_path: Path) -> None:
    """
    Verifies that the reference check accepts the smoke storyboard and lists, for a copy whose function, chip and
    command no longer exist in the repository, every one of these problems at once.
    """
    settings = create_test_settings(cache_folder=tmp_path)
    assert check_storyboard_references(storyboard_paths=[SMOKE_STORYBOARD_PATH], settings=settings) == []

    # A copy with a chip naming a missing function and a command running a missing module
    storyboard = yaml.safe_load(SMOKE_STORYBOARD_PATH.read_text(encoding="utf-8"))
    for scene in storyboard["scenes"]:
        if scene["kind"] == "code":
            scene["code_reference"] = "vision_training/models/components/convolution_block.py · ConvolutionBlock.gone"
        if scene["kind"] == "terminal":
            scene["lines"].append({"role": "command", "text": "uv run python -m vision_training.missing_module"})
    broken_path = tmp_path / "broken_storyboard.yaml"
    broken_path.write_text(data=yaml.safe_dump(data=storyboard), encoding="utf-8")
    problems = check_storyboard_references(storyboard_paths=[broken_path], settings=settings)
    assert len(problems) == 2
    assert any("ConvolutionBlock.gone" in problem for problem in problems)
    assert any("vision_training.missing_module" in problem for problem in problems)

    # A copy whose excerpt names a function that no longer exists stops at its loading
    for scene in storyboard["scenes"]:
        if scene["kind"] == "code":
            scene["source"]["function"] = "ConvolutionBlock.gone"
    broken_path.write_text(data=yaml.safe_dump(data=storyboard), encoding="utf-8")
    problems = check_storyboard_references(storyboard_paths=[broken_path], settings=settings)
    assert len(problems) == 1 and "ConvolutionBlock.gone" in problems[0]
