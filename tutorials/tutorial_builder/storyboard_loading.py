"""
Loading of the storyboards of tutorials/storyboards, the YAML files describing every scene of a chapter and its
narration, into the plain dictionaries the renderers draw. Everything a storyboard quotes from the repository is read
at build time, so that a video rebuilt after the code changed shows the code as it is: the excerpts of configurations
(the blocks of their keys, under their parent keys) and of the code (a function or class found by its name), the lines
being explained highlighted by a pattern rather than by their number; the numbers of the real runs, written by the
figure scripts into facts.json and named in the texts as {{series.name}}; the figures and files, named with a prefix
(figures:, repository:, runs:, animations:). Anything missing stops the build with a message naming it, which is also
how reference_check.py finds the storyboards a change of the repository broke.
"""

import ast
import functools
import json
import re
import textwrap

import yaml

from pathlib import Path
from typing import Any, Dict, List, Optional

from tutorial_builder.settings import ANIMATIONS_FOLDER, REPOSITORY_ROOT, STORYBOARDS_FOLDER, BuildSettings

# Kinds of scenes and the keys every one of them needs, besides its narration
SCENE_REQUIRED_KEYS = {"title_card": ["heading", "subheading"], "image": ["title", "image"],
                       "text_panel": ["title", "columns"], "yaml": ["title", "source"], "code": ["title", "source"],
                       "terminal": ["title", "lines"], "summary": ["title", "items"], "html": ["title", "page"]}

# Keys of a storyboard
STORYBOARD_REQUIRED_KEYS = ["series", "series_title", "chapter", "title", "output_name", "pace", "voice", "scenes"]

# Placeholders of the facts of the runs in the texts: {{series.name}} or {{series.name:format}}, the name followed by
# indices to take one item of a list fact ({{series.panel_boxes[2]}}, {{series.matrix[1][0]:.2f}})
FACT_PATTERN = re.compile(pattern=r"\{\{\s*([A-Za-z_]\w*)\.([A-Za-z_][\w.]*)((?:\[\d+\])*)\s*(?::([^}]*))?\}\}")

# Roots of the prefixes of the paths named in the storyboards, the others being found in the settings of the build
PATH_PREFIXES = ("figures:", "repository:", "runs:", "animations:")

# Key of a YAML line: optional list dash, then the key (quoted or not) before its colon
YAML_KEY_PATTERN = re.compile(pattern=r"^(\s*)(?:-\s+)?[\"']?([^\"':#\s{}\[\]][^\"':#{}\[\]]*?)[\"']?\s*:(?:\s|$)")

# Keys of the scenes holding texts shown or spoken, in which the facts are filled
TEXT_KEYS = {"narration", "heading", "subheading", "footnote", "title", "text", "label", "caption", "items", "lines",
             "segments", "value_labels", "labels", "columns", "heading_text"}


def resolve_path(path_text: str, settings: BuildSettings) -> Path:
    """
    Turns a path of a storyboard into an absolute path: figures: in the figures folder of the cache, repository: in the
    repository, runs: in the experiments folder, animations: in tutorials/animations.

    Args:
        path_text (str): The path, with its prefix.
        settings (BuildSettings): The settings of the build.

    Returns:
        Path: The absolute path.

    Raises:
        ValueError: If the path has none of the prefixes.
    """
    prefix, _, relative_path = path_text.partition(":")
    roots = {"figures": settings.figures_folder, "repository": REPOSITORY_ROOT, "runs": settings.experiments_folder,
             "animations": ANIMATIONS_FOLDER}
    if prefix not in roots:
        raise ValueError(f"The path {path_text!r} has none of the prefixes {PATH_PREFIXES}.")
    return roots[prefix] / relative_path


def find_fact_value(match: re.Match, facts: Dict[str, Dict[str, Any]], context: str) -> Any:
    """
    Gives the value named by one placeholder: the fact written by the figure scripts of the series, then the item of
    every index of the placeholder, so that a highlight or a label can name one panel of a list of panel boxes.

    Args:
        match (re.Match): The placeholder matched by FACT_PATTERN.
        facts (Dict[str, Dict[str, Any]]): The facts of every figure series, keyed by series then name.
        context (str): Where the text is, for the error message.

    Returns:
        Any: The fact, or its item.

    Raises:
        KeyError: If the placeholder names a fact the figure scripts did not write.
        IndexError: If an index is outside its list, or follows a value that is not a list.
    """
    series_name, fact_name, indices = match.group(1), match.group(2), match.group(3)
    if series_name not in facts or fact_name not in facts[series_name]:
        raise KeyError(f"{context}: no fact {series_name}.{fact_name}, the facts of {series_name} being "
                       f"{sorted(facts.get(series_name, {}))}.")
    value = facts[series_name][fact_name]

    # Every index of the placeholder taken in turn, from the outer list inwards
    for index in [int(index) for index in re.findall(pattern=r"\d+", string=indices)]:
        if not isinstance(value, list) or index >= len(value):
            length = f"a list of {len(value)} items" if isinstance(value, list) else f"not a list but {value!r}"
            raise IndexError(f"{context}: {match.group(0)} takes item {index} of {series_name}.{fact_name}, which is "
                             f"{length}.")
        value = value[index]
    return value


def format_fact_placeholder(match: re.Match, facts: Dict[str, Dict[str, Any]], context: str) -> str:
    """
    Gives the text of one placeholder {{series.name:format}}: the fact written by the figure scripts of the series (or
    the item its indices name), formatted with the format of the placeholder (such as .1% or ,d), called by fill_facts
    for every placeholder. A series whose facts are None (figures not built, for the reference check) keeps its
    placeholders.

    Args:
        match (re.Match): The placeholder matched by FACT_PATTERN.
        facts (Dict[str, Dict[str, Any]]): The facts of every figure series, keyed by series then name.
        context (str): Where the text is, for the error message.

    Returns:
        str: The formatted fact.
    """
    if match.group(1) in facts and facts[match.group(1)] is None:
        return match.group(0)
    return format(find_fact_value(match=match, facts=facts, context=context), (match.group(4) or "").strip())


def fill_facts(value: Any, facts: Dict[str, Dict[str, Any]], context: str) -> Any:
    """
    Fills the placeholders of the facts in a text of a scene, or in every text of a list or dictionary of texts, so
    that a narration or a label quotes the numbers of the real runs read at build time. A text that is one placeholder
    without format, such as "{{foundations.loss_panel_box}}", is replaced by the fact itself, a list or a number, so
    that the boxes of the panels of a figure come from the script that drew it.

    Args:
        value (Any): A text, or a list or dictionary holding texts.
        facts (Dict[str, Dict[str, Any]]): The facts of every figure series.
        context (str): Where the value is, for the error message.

    Returns:
        Any: The value with its facts.
    """
    if isinstance(value, str):
        # A text made of one placeholder without format takes the value of its fact or item as it is (a box, a number)
        whole_match = FACT_PATTERN.fullmatch(string=value.strip())
        if whole_match is not None and whole_match.group(4) is None and facts.get(whole_match.group(1)) is not None:
            return find_fact_value(match=whole_match, facts=facts, context=context)
        return FACT_PATTERN.sub(repl=functools.partial(format_fact_placeholder, facts=facts, context=context),
                                string=value)
    if isinstance(value, list):
        return [fill_facts(value=item, facts=facts, context=context) for item in value]
    if isinstance(value, dict):
        return {key: fill_facts(value=item, facts=facts, context=context) for key, item in value.items()}
    return value


def find_yaml_key_lines(lines: List[str]) -> List[Dict[str, Any]]:
    """
    Lists the key lines of a YAML file with their indentation and full key path (Optimization.learning_rate_schedule),
    the structure the configuration excerpts are cut from without parsing away the comments.

    Args:
        lines (List[str]): The lines of the file.

    Returns:
        List[Dict[str, Any]]: For every key line, its index, indentation and dotted path.
    """
    key_lines = []
    stack = []
    flow_depth = 0
    for line_index, line in enumerate(lines):
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue

        # Keys inside a flow mapping or list written over several lines are values of their parent, not keys
        match = YAML_KEY_PATTERN.match(string=line) if flow_depth == 0 else None
        code_part = line.split(" #", 1)[0]
        flow_depth += code_part.count("{") + code_part.count("[") - code_part.count("}") - code_part.count("]")
        if match is None:
            continue
        indentation, key = len(match.group(1)), match.group(2).strip()
        while stack and stack[-1][0] >= indentation:
            stack.pop()
        stack.append((indentation, key))
        key_lines.append({"index": line_index, "indentation": indentation,
                          "path": ".".join(stack_key for _, stack_key in stack)})
    return key_lines


def extract_yaml_excerpt(file_path: Path, keys: List[str], keep_comments: bool) -> List[str]:
    """
    Cuts the blocks of some keys out of a configuration file, each block under the lines of its parent keys and the
    blocks separated by "..." where lines are skipped, so that the excerpt reads like the file: a key line, its comments
    and every line indented under it. The comments describing a key, just above it, come with it when kept; without
    them, the blank lines that separated the commented keys go too, so that the excerpt stays compact.

    Args:
        file_path (Path): The YAML file.
        keys (List[str]): The dotted paths of the keys, in the order of the file.
        keep_comments (bool): Whether the comment lines, and the blank lines between them, are kept.

    Returns:
        List[str]: The lines of the excerpt, dedented.

    Raises:
        KeyError: If a key is not in the file.
    """
    lines = file_path.read_text(encoding="utf-8").splitlines()
    key_lines = find_yaml_key_lines(lines=lines)
    paths = {key_line["path"]: key_line for key_line in key_lines}
    selected_indices = set()
    for key in keys:
        if key not in paths:
            raise KeyError(f"No key {key} in {file_path}, whose keys include "
                           f"{[path for path in paths if path.count('.') <= key.count('.')][:40]}.")
        key_line = paths[key]

        # The block: the key line and every following line indented under it (blank lines inside included)
        block_end = key_line["index"] + 1
        while block_end < len(lines):
            next_line = lines[block_end]
            next_indentation = len(next_line) - len(next_line.lstrip())
            if next_line.strip() and next_indentation <= key_line["indentation"]:
                break
            block_end += 1
        while block_end > key_line["index"] + 1 and not lines[block_end - 1].strip():
            block_end -= 1
        selected_indices.update(range(key_line["index"], block_end))

        # The comments just above the key, at its indentation
        comment_index = key_line["index"] - 1
        while comment_index >= 0 and lines[comment_index].strip().startswith("#") and \
                len(lines[comment_index]) - len(lines[comment_index].lstrip()) == key_line["indentation"]:
            selected_indices.add(comment_index)
            comment_index -= 1

        # The lines of the parent keys, so that the block keeps its place in the file
        for parent_line in key_lines:
            if key.startswith(parent_line["path"] + ".") and parent_line["index"] < key_line["index"]:
                selected_indices.add(parent_line["index"])

    # Lines in the order of the file, "..." where lines are skipped, comments and blank lines dropped when asked
    excerpt_lines = []
    previous_index = None
    for line_index in sorted(selected_indices):
        line = lines[line_index]
        if not keep_comments and (line.strip().startswith("#") or not line.strip()):
            continue
        if previous_index is not None and line_index > previous_index + 1:
            skipped_lines = [lines[skipped] for skipped in range(previous_index + 1, line_index)]
            if any(skipped.strip() and (keep_comments or not skipped.strip().startswith("#"))
                   for skipped in skipped_lines):
                indentation = len(line) - len(line.lstrip())
                excerpt_lines.append(" " * indentation + "...")
        excerpt_lines.append(line)
        previous_index = line_index
    return textwrap.dedent(text="\n".join(excerpt_lines)).splitlines()


def get_definition_name(node: ast.AST) -> Optional[str]:
    """
    Gives the name a statement of a module or class defines, for the excerpts and chips naming it: a function, a class,
    or a constant assigned to a single name (the type lists and strength tables of the augmentation code, for instance).

    Args:
        node (ast.AST): The statement.

    Returns:
        Optional[str]: The name, or None for a statement defining no single name.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def find_python_definition(tree: ast.Module, qualified_name: str) -> ast.AST:
    """
    Finds a function, class or constant of a module by its qualified name: name for a module-level definition,
    Class.method for a method.

    Args:
        tree (ast.Module): The parsed module.
        qualified_name (str): The name of the definition.

    Returns:
        ast.AST: The FunctionDef, AsyncFunctionDef, ClassDef, Assign or AnnAssign node.

    Raises:
        KeyError: If no definition has that name.
    """
    nodes = tree.body
    found_node = None
    for name_part in qualified_name.split("."):
        found_node = next((node for node in nodes if get_definition_name(node=node) == name_part), None)
        if found_node is None:
            raise KeyError(f"No definition {qualified_name} (missing {name_part}).")
        nodes = getattr(found_node, "body", [])
    return found_node


def extract_python_excerpt(file_path: Path, qualified_name: str, keep_docstring: bool,
                           start_pattern: Optional[str], end_pattern: Optional[str]) -> List[str]:
    """
    Cuts a function, class or constant out of a Python file of the repository, found by its name, its docstring dropped
    unless asked; narrowed to the lines from the first one matching start_pattern to the first following one matching
    end_pattern when given, after the lines of its signature and "..." for the lines skipped.

    Args:
        file_path (Path): The Python file.
        qualified_name (str): The name of the function, class or constant, Class.method for a method.
        keep_docstring (bool): Whether its docstring is kept.
        start_pattern (Optional[str]): The pattern of the first line of the narrowed excerpt, or None.
        end_pattern (Optional[str]): The pattern of its last line, or None for the end of the definition.

    Returns:
        List[str]: The lines of the excerpt, dedented.

    Raises:
        KeyError: If the definition, or a line matching a pattern, is not found.
    """
    source = file_path.read_text(encoding="utf-8")
    source_lines = source.splitlines()
    node = find_python_definition(tree=ast.parse(source=source), qualified_name=qualified_name)
    first_line = min([node.lineno] + [decorator.lineno for decorator in getattr(node, "decorator_list", [])]) - 1
    body_lines = list(range(first_line, node.end_lineno))

    # Lines of the docstring left out unless asked; a constant has no docstring, its first line standing for the
    # signature of a narrowed excerpt
    body = getattr(node, "body", [])
    docstring_lines = set()
    if not keep_docstring and body and isinstance(body[0], ast.Expr) and \
            isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        docstring_lines = set(range(body[0].lineno - 1, body[0].end_lineno))
    signature_end = body[0].lineno - 1 if body else first_line + 1
    kept_lines = [line_index for line_index in body_lines if line_index not in docstring_lines]

    # Narrowed excerpt: the signature, "...", the lines between the two patterns, and "..." when the body goes on
    if start_pattern is not None:
        start_index = next((line_index for line_index in kept_lines if line_index >= signature_end and
                            re.search(pattern=start_pattern, string=source_lines[line_index])), None)
        if start_index is None:
            raise KeyError(f"No line of {qualified_name} in {file_path} matches {start_pattern!r}.")
        end_index = kept_lines[-1]
        if end_pattern is not None:
            end_index = next((line_index for line_index in kept_lines if line_index >= start_index and
                              re.search(pattern=end_pattern, string=source_lines[line_index])), None)
            if end_index is None:
                raise KeyError(f"No line of {qualified_name} in {file_path} after {start_pattern!r} matches "
                               f"{end_pattern!r}.")
        body_indentation = " " * (len(source_lines[start_index]) - len(source_lines[start_index].lstrip()))
        excerpt_lines = [source_lines[line_index] for line_index in kept_lines if line_index < signature_end]
        if start_index > signature_end and any(line_index < start_index and line_index >= signature_end
                                               for line_index in kept_lines):
            excerpt_lines.append(body_indentation + "...")
        excerpt_lines += [source_lines[line_index] for line_index in kept_lines if start_index <= line_index <=
                          end_index]
        if end_index < kept_lines[-1]:
            excerpt_lines.append(body_indentation + "...")
    else:
        excerpt_lines = [source_lines[line_index] for line_index in kept_lines]
    return textwrap.dedent(text="\n".join(excerpt_lines)).splitlines()


def extract_text_excerpt(file_path: Path, start_pattern: Optional[str], end_pattern: Optional[str],
                         line_count: Optional[int]) -> List[str]:
    """
    Cuts lines out of a text file (a labels.json entry written by a figure script, a log of a run): from the first
    line matching start_pattern (the first line when None) to the first following one matching end_pattern, or for
    line_count lines.

    Args:
        file_path (Path): The text file.
        start_pattern (Optional[str]): The pattern of the first line, or None.
        end_pattern (Optional[str]): The pattern of the last line, or None.
        line_count (Optional[int]): The number of lines when no end pattern is given, None for the rest of the file.

    Returns:
        List[str]: The lines of the excerpt.

    Raises:
        KeyError: If no line matches a pattern.
    """
    lines = file_path.read_text(encoding="utf-8").splitlines()
    start_index = 0
    if start_pattern is not None:
        start_index = next((index for index, line in enumerate(lines) if re.search(pattern=start_pattern,
                                                                                    string=line)), None)
        if start_index is None:
            raise KeyError(f"No line of {file_path} matches {start_pattern!r}.")
    end_index = len(lines) - 1 if line_count is None else min(len(lines) - 1, start_index + line_count - 1)
    if end_pattern is not None:
        end_index = next((index for index in range(start_index, len(lines)) if re.search(pattern=end_pattern,
                                                                                       string=lines[index])), None)
        if end_index is None:
            raise KeyError(f"No line of {file_path} after {start_pattern!r} matches {end_pattern!r}.")
    return lines[start_index:end_index + 1]


def resolve_excerpt(source: Dict[str, Any], highlights: List[Dict[str, Any]], settings: BuildSettings, context: str,
                    figures_required: bool) -> List[Dict[str, Any]]:
    """
    Reads the lines of an excerpt (a yaml or code scene, a listing element of an HTML diagram) from its source and
    marks the lines of its highlights: every highlight gives a pattern and the sentences during which its matching
    lines are highlighted (or always), and must match at least one line, so that a renamed key or function breaks the
    build instead of silently highlighting nothing.

    Args:
        source (Dict[str, Any]): The source of the excerpt: its file and keys, function or start/end/count.
        highlights (List[Dict[str, Any]]): The highlights (match, sentences or always, first_only).
        settings (BuildSettings): The settings of the build.
        context (str): Where the excerpt is, for the error messages.
        figures_required (bool): Whether a source among the figures of the cache must exist; when False, an excerpt
            of a figure not built yet is one placeholder line, for the reference check.

    Returns:
        List[Dict[str, Any]]: The lines of the excerpt, with their text, highlight and highlight_sentences.

    Raises:
        FileNotFoundError: If the source file does not exist.
        KeyError: If a highlight matches no line.
    """
    file_path = resolve_path(path_text=source["file"], settings=settings) if source["file"].startswith(PATH_PREFIXES) \
        else REPOSITORY_ROOT / source["file"]
    if not file_path.exists() and not figures_required and file_path.is_relative_to(settings.figures_folder):
        return [{"text": f"({file_path.name}, figure not built)", "highlight": False, "highlight_sentences": []}]
    if not file_path.exists():
        raise FileNotFoundError(f"{context}: no file {file_path}.")

    # Lines of the excerpt from the kind of the source
    if "keys" in source:
        texts = extract_yaml_excerpt(file_path=file_path, keys=source["keys"],
                                     keep_comments=source.get("comments", True))
    elif "function" in source:
        texts = extract_python_excerpt(file_path=file_path, qualified_name=source["function"],
                                       keep_docstring=source.get("docstring", False),
                                       start_pattern=source.get("start"), end_pattern=source.get("end"))
    else:
        texts = extract_text_excerpt(file_path=file_path, start_pattern=source.get("start"),
                                     end_pattern=source.get("end"), line_count=source.get("count"))
    lines = [{"text": text, "highlight": False, "highlight_sentences": []} for text in texts]

    # Highlights by pattern, every one matching at least one line
    for highlight in highlights:
        matched_lines = [line for line in lines if re.search(pattern=highlight["match"], string=line["text"])]
        if not matched_lines:
            raise KeyError(f"{context}: the highlight {highlight['match']!r} matches no line of the excerpt of "
                           f"{file_path}.")
        if highlight.get("first_only", False):
            matched_lines = matched_lines[:1]
        for line in matched_lines:
            line["highlight"] = line["highlight"] or highlight.get("always", False)
            line["highlight_sentences"] = sorted(set(line["highlight_sentences"]) | set(highlight.get("sentences",
                                                                                                    [])))
    return lines


def resolve_scene_paths(value: Any, settings: BuildSettings) -> Any:
    """
    Turns every prefixed path of a scene (figures:, repository:, runs:, animations:) into an absolute path, wherever it
    is in the scene: its image, overlay, the images of its diagram elements, the data of its HTML page.

    Args:
        value (Any): A value of the scene.
        settings (BuildSettings): The settings of the build.

    Returns:
        Any: The value with absolute paths.
    """
    if isinstance(value, str) and value.startswith(PATH_PREFIXES):
        return str(resolve_path(path_text=value, settings=settings))
    if isinstance(value, list):
        return [resolve_scene_paths(value=item, settings=settings) for item in value]
    if isinstance(value, dict):
        return {key: resolve_scene_paths(value=item, settings=settings) for key, item in value.items()}
    return value


def load_facts(figure_series: List[str], settings: BuildSettings,
               figures_required: bool) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Reads the facts.json written by the figure scripts of every series a storyboard uses: the numbers of the real
    runs (accuracies, counts, shapes) its texts quote.

    Args:
        figure_series (List[str]): The figure series of the storyboard.
        settings (BuildSettings): The settings of the build.
        figures_required (bool): Whether the figures must be built; when False, a series without facts.json gets None
            and its placeholders stay unfilled.

    Returns:
        Dict[str, Optional[Dict[str, Any]]]: The facts keyed by series then name.

    Raises:
        FileNotFoundError: If a required series has no facts.json, its figures not being made yet.
    """
    facts = {}
    for series_name in figure_series:
        facts_path = settings.figures_folder / series_name / "facts.json"
        if not facts_path.exists() and not figures_required:
            facts[series_name] = None
            continue
        if not facts_path.exists():
            raise FileNotFoundError(f"No {facts_path}: make the figures of {series_name} first (build_tutorials.py "
                                    f"--figures-only --series ...).")
        facts[series_name] = json.loads(facts_path.read_text(encoding="utf-8"))
    return facts


def load_storyboard(storyboard_path: Path, settings: BuildSettings, figures_required: bool = True) -> Dict[str, Any]:
    """
    Loads a storyboard and prepares it for the renderers: its keys checked, the facts of its figure series filled in
    its texts, its paths made absolute, the excerpts of its yaml and code scenes read from the repository, every image
    and HTML page checked on disk, its series label completed.

    Args:
        storyboard_path (Path): The YAML storyboard.
        settings (BuildSettings): The settings of the build.
        figures_required (bool): Whether the figures of the cache must be built; False for the reference check, which
            then leaves the facts unfilled and the figures unchecked.

    Returns:
        Dict[str, Any]: The storyboard ready to render.

    Raises:
        ValueError: If a key is missing, a scene kind is unknown or a scene has no narration.
        FileNotFoundError: If an image or an HTML page is missing.
    """
    storyboard = yaml.safe_load(storyboard_path.read_text(encoding="utf-8"))
    missing_keys = [key for key in STORYBOARD_REQUIRED_KEYS if key not in storyboard]
    if missing_keys:
        raise ValueError(f"{storyboard_path}: missing keys {missing_keys}.")
    facts = load_facts(figure_series=storyboard.get("figures", []), settings=settings,
                       figures_required=figures_required)
    # Numbered folder of the series holding the storyboard, for its chapter count here and for the output folder of
    # its videos
    storyboard["series_folder"] = storyboard_path.parent.name
    chapter_count = len(sorted(storyboard_path.parent.glob("*.yaml")))
    storyboard["series_label"] = f"{storyboard['series_title']} · {storyboard['chapter']}/{chapter_count}"
    storyboard["storyboard_path"] = str(storyboard_path)

    # Every scene checked, its facts filled, its paths made absolute and its excerpt read
    for scene_index, scene in enumerate(storyboard["scenes"]):
        context = f"{storyboard_path.name}, scene {scene_index + 1} ({scene.get('title', scene.get('kind'))})"
        if scene.get("kind") not in SCENE_REQUIRED_KEYS:
            raise ValueError(f"{context}: unknown kind {scene.get('kind')!r}, expected one of "
                             f"{sorted(SCENE_REQUIRED_KEYS)}.")
        missing_keys = [key for key in SCENE_REQUIRED_KEYS[scene["kind"]] + ["narration"] if not scene.get(key)]
        if missing_keys:
            raise ValueError(f"{context}: missing or empty keys {missing_keys}.")
        for key in list(scene):
            if key in TEXT_KEYS or key in ("elements", "data", "views", "highlights", "reveals"):
                scene[key] = fill_facts(value=scene[key], facts=facts, context=context)
        if scene["kind"] in ("yaml", "code"):
            scene.setdefault("file_label", scene["source"]["file"].split(":", 1)[-1])
        storyboard["scenes"][scene_index] = scene = resolve_scene_paths(value=scene, settings=settings)
        if scene["kind"] in ("yaml", "code"):
            scene["lines"] = resolve_excerpt(source=scene["source"], highlights=scene.get("highlights", []),
                                             settings=settings, context=context, figures_required=figures_required)
            scene.setdefault("syntax", "yaml" if scene["kind"] == "yaml" else "python")

        # Images of image scenes and HTML pages on disk
        if scene["kind"] == "image":
            scene["image_path"] = scene["image"]
            if "overlay" in scene:
                scene["overlay_path"] = scene["overlay"]
            for image_path in [scene["image_path"], scene.get("overlay_path")]:
                is_unbuilt_figure = not figures_required and Path(image_path or "").is_relative_to(
                    settings.figures_folder)
                if image_path is not None and not is_unbuilt_figure and not Path(image_path).exists():
                    raise FileNotFoundError(f"{context}: no image {image_path}.")
        if scene["kind"] == "html":
            page_path = ANIMATIONS_FOLDER / f"{scene['page']}.html"
            if not page_path.exists():
                raise FileNotFoundError(f"{context}: no HTML page {page_path}.")
            scene["page_path"] = str(page_path)

            # Listing elements of the diagram, their lines read like the excerpts of the code scenes
            for element in scene.get("elements", []):
                if element.get("type") == "listing" and "source" in element:
                    element["lines"] = resolve_excerpt(source=element["source"],
                                                       highlights=element.get("highlights", []), settings=settings,
                                                       context=f"{context}, listing of {element['source']['file']}",
                                                       figures_required=figures_required)
    return storyboard


def find_series_folder(series_name: str) -> str:
    """
    Gives the folder of a series, its name after its two-digit position in tutorials/storyboards/series.yaml
    (01_foundations), so that the storyboards, the animations and the videos of the output folder sort in viewing
    order, while --series and the facts keep the plain name.

    Args:
        series_name (str): The name of the series in series.yaml.

    Returns:
        str: The folder name of the series.

    Raises:
        ValueError: If the series is not in series.yaml.
    """
    series_names = [series["name"] for series in
                    yaml.safe_load((STORYBOARDS_FOLDER / "series.yaml").read_text(encoding="utf-8"))["series"]]
    if series_name not in series_names:
        raise ValueError(f"Unknown series {series_name}, expected one of {series_names}.")
    return f"{series_names.index(series_name):02d}_{series_name}"


def list_storyboards(series_names: Optional[List[str]], chapters: Optional[List[int]]) -> List[Path]:
    """
    Lists the storyboards to build, in the order of their series (the order of tutorials/series.yaml) and chapters.

    Args:
        series_names (Optional[List[str]]): The series to build, None for every series.
        chapters (Optional[List[int]]): The chapters to build within them, None for every chapter.

    Returns:
        List[Path]: The storyboard files.

    Raises:
        ValueError: If a series is unknown.
    """
    series_order = yaml.safe_load((STORYBOARDS_FOLDER / "series.yaml").read_text(encoding="utf-8"))["series"]
    known_series = [series["name"] for series in series_order]
    unknown_series = [name for name in series_names or [] if name not in known_series]
    if unknown_series:
        raise ValueError(f"Unknown series {unknown_series}, expected some of {known_series}.")
    storyboard_paths = []
    for series_name in known_series:
        if series_names and series_name not in series_names:
            continue
        series_folder = STORYBOARDS_FOLDER / find_series_folder(series_name=series_name)
        for storyboard_path in sorted(series_folder.glob("*.yaml")):
            chapter = yaml.safe_load(storyboard_path.read_text(encoding="utf-8"))["chapter"]
            if chapters is None or chapter in chapters:
                storyboard_paths.append(storyboard_path)
    return storyboard_paths
