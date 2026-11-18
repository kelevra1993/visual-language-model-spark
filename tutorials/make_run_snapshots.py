"""
Writer of the committed run snapshots of the tutorials (tutorials/run_snapshots, the 'snapshots' section of
tutorials/tutorial_sources.yaml): it copies from the experiments folder of the runs, the datasets of the machine and the
figures folder of the build cache the small files every series declares in
tutorials/figures/<series>/snapshot_files.yaml, reduced where a whole file would be too large for git (scalars of the TensorBoard logs, keys of a JSON file, lines of a
console log, records of a dataset manifest), into the snapshot folder at the paths the figure scripts and storyboards
read, and writes one manifest per series (source, sizes and SHA-256 of every file). A machine without the runs then
builds every series with `tutorials/build_tutorials.sh --experiments-folder tutorials/run_snapshots`.

Run it from the repository root with the Python of the repository environment (it reads the TensorBoard logs):
    .venv/bin/python -m tutorials.make_run_snapshots                # every series
    .venv/bin/python -m tutorials.make_run_snapshots --series classification foundations
    .venv/bin/python -m tutorials.make_run_snapshots --check        # compare the snapshot with its manifests only
"""

import argparse
import csv
import fnmatch
import glob
import gzip
import hashlib
import importlib
import json
import re
import shutil

import yaml

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto import event_pb2
from tensorboard.summary.writer.record_writer import RecordWriter

# Files of the tutorial folder read by the writer, relative to the repository root
TUTORIALS_FOLDER = Path("tutorials")
SOURCES_FILE_NAME = "tutorial_sources.yaml"
PROJECT_FILE_NAME = "tutorial_project.yaml"
SPECIFICATION_FILE_NAME = "snapshot_files.yaml"

# Folders of the snapshot folder holding the dataset files, the committed figures and the manifests
DATASETS_FOLDER_NAME = "datasets"
FIGURE_REUSE_FOLDER_NAME = "figure_reuse"
MANIFESTS_FOLDER_NAME = "manifests"

# Reductions a specification entry can name, every other one being a "module:function" custom reduction
BUILT_IN_REDUCTIONS = ["copy", "tensorboard_scalars", "json_keys", "text_lines", "csv_columns", "manifest_subset",
                       "folder_listing"]

# Options of an entry that are not passed to its reduction: "compress: gzip" writes the reduced file gzipped, its name
# ending with .gz, with a fixed header (no name, no time) so that writing it again gives the same bytes
COMPRESSION_SUFFIXES = {"gzip": ".gz"}

# Plugin name of the scalar summaries written as tensors by the summary writer of PyTorch
SCALARS_PLUGIN_NAME = "scalars"

BYTES_PER_MEGABYTE = 1024 * 1024


def parse_arguments() -> argparse.Namespace:
    """
    Reads the series to snapshot and the folders to read from, the defaults being those of the tutorial project file.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Writes the committed run snapshots of the tutorials.")
    parser.add_argument("--series", nargs="*", default=None,
                        help="Series whose snapshot_files.yaml is written, all by default.")
    parser.add_argument("--experiments-folder", default=None,
                        help="Folder of the real runs, the experiments_folder of tutorial_project.yaml by default.")
    parser.add_argument("--figures-folder", default=None,
                        help="Figures folder of a build, <cache_folder of tutorial_project.yaml>/figures by default.")
    parser.add_argument("--check", action="store_true",
                        help="Only compare the files of the snapshot folder with the SHA-256 of the manifests.")
    parser.add_argument("--prune", action="store_true",
                        help="Delete the snapshot files no manifest lists (only with every series written).")
    return parser.parse_args()


def compute_sha256(file_path: Path) -> str:
    """
    Gives the SHA-256 of a file, recorded in the manifests so that a snapshot file changed by hand, or a source changed
    since the snapshot, shows in --check and in a diff of the manifests.

    Args:
        file_path (Path): The file.

    Returns:
        str: The hexadecimal digest.
    """
    digest = hashlib.sha256()
    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(BYTES_PER_MEGABYTE), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_run_entry(runs: Dict[str, Any], run_key: str) -> Dict[str, Any]:
    """
    Finds a run of tutorial_sources.yaml from the key a specification gives, a dotted path reaching the comparison runs
    of a run (e.g. "sequence_classification.comparison_runs.flatten_512").

    Args:
        runs (Dict[str, Any]): The 'runs' section of tutorial_sources.yaml.
        run_key (str): The key.

    Returns:
        Dict[str, Any]: The run entry, with its 'folder' or 'folders'.

    Raises:
        KeyError: If the key names no run with a folder.
    """
    entry: Any = runs
    for key_part in run_key.split("."):
        if not isinstance(entry, dict) or key_part not in entry:
            raise KeyError(f"No run {run_key} in {SOURCES_FILE_NAME} (no {key_part!r} under it).")
        entry = entry[key_part]
    if not isinstance(entry, dict) or ("folder" not in entry and "folders" not in entry):
        raise KeyError(f"The run {run_key} of {SOURCES_FILE_NAME} has no 'folder' nor 'folders'.")
    return entry


def expand_source_files(root_folder: Path, pattern: str) -> List[Path]:
    """
    Gives the files a specification path names, a path relative to a folder that may hold glob characters and '..'
    (the console log next to a run), refusing a path leaving the root it is snapshotted under.

    Args:
        root_folder (Path): The folder the path is relative to (a run folder, the data root, a figures folder).
        pattern (str): The path or glob.

    Returns:
        List[Path]: The matching paths, sorted (files and, for a folder listing, folders).

    Raises:
        FileNotFoundError: If nothing matches.
    """
    matches = sorted(Path(match) for match in glob.glob(str(root_folder / pattern), recursive=True))
    if not matches:
        raise FileNotFoundError(f"Nothing matches {pattern!r} under {root_folder}.")
    return matches


def reduce_tensorboard_scalars(source_path: Path, destination_path: Path, options: Dict[str, Any]) -> None:
    """
    Writes a TensorBoard event file holding only the scalars of the source (the sample images, histograms and texts
    make most of the bytes of an evaluation log), and only the tags listed in the options when it lists some (exact
    tags or fnmatch patterns); the events without a summary (the file version) are kept, so that EventAccumulator reads
    the same scalar curves.

    Args:
        source_path (Path): The event file of the run.
        destination_path (Path): The reduced event file.
        options (Dict[str, Any]): The entry options, with the optional 'tags' list.
    """
    tag_patterns = options.get("tags", [])
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    writer = RecordWriter(open(destination_path, "wb"))
    for raw_record in RawEventFileLoader(str(source_path)).Load():
        event = event_pb2.Event.FromString(raw_record)

        # Events without a summary (file version, session logs) are written unchanged
        if not event.HasField("summary"):
            writer.write(raw_record)
            continue

        # Scalar values only, written as simple values or as scalar tensors, of the tags asked for
        kept_values = []
        for value in event.summary.value:
            value_kind = value.WhichOneof("value")
            is_scalar = value_kind == "simple_value" or (
                value_kind == "tensor" and value.metadata.plugin_data.plugin_name == SCALARS_PLUGIN_NAME)
            is_kept_tag = not tag_patterns or any(fnmatch.fnmatchcase(value.tag, pattern) for pattern in tag_patterns)
            if is_scalar and is_kept_tag:
                kept_values.append(value)
        if kept_values:
            reduced_event = event_pb2.Event(wall_time=event.wall_time, step=event.step)
            reduced_event.summary.value.extend(kept_values)
            writer.write(reduced_event.SerializeToString())
    writer.close()


def reduce_json_keys(source_path: Path, destination_path: Path, options: Dict[str, Any]) -> None:
    """
    Writes a JSON file holding only the top-level keys listed in the options (the class counts of a data.json without
    the file names of every split).

    Args:
        source_path (Path): The JSON file.
        destination_path (Path): The reduced file.
        options (Dict[str, Any]): The entry options, with the 'keys' list.

    Raises:
        KeyError: If a listed key is missing from the source.
    """
    content = json.loads(source_path.read_text(encoding="utf-8"))
    missing_keys = [key for key in options["keys"] if key not in content]
    if missing_keys:
        raise KeyError(f"{source_path} has no keys {missing_keys}, its keys being {sorted(content)}.")
    reduced_content = {key: content[key] for key in options["keys"]}
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(data=json.dumps(obj=reduced_content, indent=2), encoding="utf-8")


def reduce_text_lines(source_path: Path, destination_path: Path, options: Dict[str, Any]) -> None:
    """
    Writes a text file holding only the lines matching the regular expression of the options (the lines of a console
    log a figure script parses), in their order.

    Args:
        source_path (Path): The text file.
        destination_path (Path): The reduced file.
        options (Dict[str, Any]): The entry options, with the 'pattern' regular expression.

    Raises:
        ValueError: If no line matches.
    """
    pattern = re.compile(pattern=options["pattern"])
    lines = source_path.read_text(encoding="utf-8", errors="strict").splitlines(keepends=True)
    kept_lines = [line for line in lines if pattern.search(line)]
    if not kept_lines:
        raise ValueError(f"No line of {source_path} matches {options['pattern']!r}.")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(data="".join(kept_lines), encoding="utf-8")


def reduce_csv_columns(source_path: Path, destination_path: Path, options: Dict[str, Any]) -> None:
    """
    Writes a CSV file holding only the columns listed in the options, in the order of the source, every row kept (the
    columns of a predictions.csv a figure reads, without the probability of every class).

    Args:
        source_path (Path): The CSV file.
        destination_path (Path): The reduced file.
        options (Dict[str, Any]): The entry options, with the 'columns' list.

    Raises:
        KeyError: If a listed column is missing from the source.
    """
    with open(source_path, newline="", encoding="utf-8") as source_file:
        reader = csv.reader(source_file)
        header = next(reader)
        missing_columns = [column for column in options["columns"] if column not in header]
        if missing_columns:
            raise KeyError(f"{source_path} has no columns {missing_columns}, its columns being {header}.")
        kept_indices = [index for index, column in enumerate(header) if column in options["columns"]]
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with open(destination_path, "w", newline="", encoding="utf-8") as destination_file:
            writer = csv.writer(destination_file, lineterminator="\n")
            writer.writerow([header[index] for index in kept_indices])
            for row in reader:
                writer.writerow([row[index] for index in kept_indices])


def compress_file(file_path: Path) -> None:
    """
    Gzips a written snapshot file in place, with no file name and a zero time in the header, so that the same content
    always gives the same bytes and SHA-256.

    Args:
        file_path (Path): The file, replaced by its gzipped bytes.
    """
    content = file_path.read_bytes()
    with open(file_path, "wb") as raw_file:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_file, mtime=0, compresslevel=9) as gzip_file:
            gzip_file.write(content)


def reduce_manifest_subset(source_path: Path, destination_path: Path, options: Dict[str, Any]) -> None:
    """
    Writes a dataset manifest (a labels.json of images and annotations) keeping whole the records of the images listed
    in the options and their annotations, and the other records slimmed to some fields ('slim', the counts staying
    right), kept ('keep') or dropped ('drop'); the other top-level keys (info, categories) are kept.

    Args:
        source_path (Path): The manifest.
        destination_path (Path): The reduced manifest.
        options (Dict[str, Any]): 'images' (file names kept whole), 'other_images' and 'other_annotations' (keep,
            slim or drop, slim by default), 'image_fields' and 'annotation_fields' (the fields of the slimmed records,
            [id, file_name] and [id, image_id] by default).

    Raises:
        ValueError: If a listed image is not in the manifest or a mode is unknown.
    """
    content = json.loads(source_path.read_text(encoding="utf-8"))
    kept_names = set(options.get("images", []))
    image_fields = options.get("image_fields", ["id", "file_name"])
    annotation_fields = options.get("annotation_fields", ["id", "image_id"])
    modes = {"other_images": options.get("other_images", "slim"),
             "other_annotations": options.get("other_annotations", "slim")}
    for mode_name, mode in modes.items():
        if mode not in ["keep", "slim", "drop"]:
            raise ValueError(f"Unknown {mode_name} {mode!r} for {source_path}: keep, slim or drop.")

    # Images kept whole, by file name, the others reduced as asked
    kept_image_identifiers = {image["id"] for image in content["images"] if image["file_name"] in kept_names}
    found_names = {image["file_name"] for image in content["images"] if image["file_name"] in kept_names}
    if found_names != kept_names:
        raise ValueError(f"{source_path} has no images {sorted(kept_names - found_names)}.")
    reduced_content = {key: value for key, value in content.items() if key not in ["images", "annotations"]}
    reduced_content["images"] = [image if image["id"] in kept_image_identifiers or modes["other_images"] == "keep"
                                 else {field: image[field] for field in image_fields if field in image}
                                 for image in content["images"]
                                 if image["id"] in kept_image_identifiers or modes["other_images"] != "drop"]

    # Annotations of the kept images whole, the others reduced as asked
    if "annotations" in content:
        reduced_content["annotations"] = [
            annotation if annotation["image_id"] in kept_image_identifiers or modes["other_annotations"] == "keep"
            else {field: annotation[field] for field in annotation_fields if field in annotation}
            for annotation in content["annotations"]
            if annotation["image_id"] in kept_image_identifiers or modes["other_annotations"] != "drop"]
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(data=json.dumps(obj=reduced_content, separators=(",", ":")), encoding="utf-8")


def write_folder_listing(source_path: Path, destination_path: Path, options: Dict[str, Any]) -> None:
    """
    Writes the listing of a folder of the run (one "type, size in bytes, relative path" line per entry, tab
    separated, sorted), for the figures that draw the tree of a run whose large files the snapshot leaves out; the
    names listed in the 'exclude' option (fnmatch patterns on the relative path) are left out.

    Args:
        source_path (Path): The folder.
        destination_path (Path): The listing file.
        options (Dict[str, Any]): The entry options, with the optional 'exclude' list.
    """
    exclude_patterns = options.get("exclude", [])
    lines = []
    for entry_path in sorted(source_path.rglob("*")):
        relative_path = entry_path.relative_to(source_path).as_posix()
        if any(fnmatch.fnmatchcase(relative_path, pattern) for pattern in exclude_patterns):
            continue
        entry_type = "folder" if entry_path.is_dir() else "file"
        entry_size = 0 if entry_path.is_dir() else entry_path.stat().st_size
        lines.append(f"{entry_type}\t{entry_size}\t{relative_path}\n")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(data="".join(lines), encoding="utf-8")


def copy_file(source_path: Path, destination_path: Path, options: Dict[str, Any]) -> None:
    """
    Copies a file unchanged.

    Args:
        source_path (Path): The file.
        destination_path (Path): The copy.
        options (Dict[str, Any]): The entry options, unused.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src=source_path, dst=destination_path)


def find_reduction(reduction_name: str) -> Callable[[Path, Path, Dict[str, Any]], None]:
    """
    Gives the function of a reduction: a built-in one, or a "module:function" of the repository for a reduction only
    one series needs, called with the source path, the destination path and the entry options.

    Args:
        reduction_name (str): The reduction of the entry.

    Returns:
        Callable[[Path, Path, Dict[str, Any]], None]: The function.

    Raises:
        ValueError: If the name is neither built in nor a "module:function".
    """
    built_in_functions = {"copy": copy_file, "tensorboard_scalars": reduce_tensorboard_scalars,
                          "json_keys": reduce_json_keys, "text_lines": reduce_text_lines,
                          "csv_columns": reduce_csv_columns, "manifest_subset": reduce_manifest_subset,
                          "folder_listing": write_folder_listing}
    if reduction_name in built_in_functions:
        return built_in_functions[reduction_name]
    if ":" not in reduction_name:
        raise ValueError(f"Unknown reduction {reduction_name!r}: one of {BUILT_IN_REDUCTIONS} or a module:function.")
    module_name, function_name = reduction_name.split(":")
    return getattr(importlib.import_module(name=module_name), function_name)


def plan_series_entries(series_name: str, specification: Dict[str, Any], sources: Dict[str, Any],
                        experiments_folder: Path, data_root: Path, figures_folder: Path,
                        snapshot_root: Path) -> List[Dict[str, Any]]:
    """
    Turns the specification of a series into the list of files to write: for every entry, every source file it
    matches, the snapshot path it is written to (the same path relative to the experiments folder, datasets/ for the
    dataset files, figure_reuse/<series>/ for the committed figures), its reduction and options.

    Args:
        series_name (str): The series.
        specification (Dict[str, Any]): Its snapshot_files.yaml.
        sources (Dict[str, Any]): tutorial_sources.yaml.
        experiments_folder (Path): The folder of the real runs.
        data_root (Path): The data root of the machine, the recorded_data_root of the snapshots by default.
        figures_folder (Path): The figures folder of the build cache.
        snapshot_root (Path): The snapshot folder.

    Returns:
        List[Dict[str, Any]]: One planned file per source file, with 'source', 'source_kind', 'destination',
            'reduction' and 'options'.

    Raises:
        ValueError: If a source path leaves its root folder.
    """
    planned_files = []

    # Files of the runs, at their path relative to the experiments folder
    for run_specification in specification.get("runs", []):
        run_entry = resolve_run_entry(runs=sources["runs"], run_key=run_specification["run"])
        run_folders = [run_entry["folder"]] if "folder" in run_entry else run_entry["folders"]
        for run_folder_text in run_folders:
            for file_specification in run_specification["files"]:
                for source_path in expand_source_files(root_folder=experiments_folder / run_folder_text,
                                                       pattern=file_specification["path"]):
                    planned_files.append(build_planned_file(source_path=source_path, root_folder=experiments_folder,
                                                            destination_root=snapshot_root, source_kind="run",
                                                            file_specification=file_specification))

    # Dataset files, at their path relative to the data root, under datasets/
    for file_specification in specification.get("datasets", []):
        for source_path in expand_source_files(root_folder=data_root, pattern=file_specification["path"]):
            planned_files.append(build_planned_file(source_path=source_path, root_folder=data_root,
                                                    destination_root=snapshot_root / DATASETS_FOLDER_NAME,
                                                    source_kind="dataset", file_specification=file_specification))

    # Committed figures of the families that need a checkpoint, under figure_reuse/<series>/
    for family_specification in specification.get("figure_reuse", []):
        family_folder = figures_folder / series_name / family_specification["family"]
        if not family_folder.is_dir():
            raise FileNotFoundError(f"No figures folder {family_folder}: build the figures of {series_name} first.")
        for source_path in sorted(path for path in family_folder.rglob("*") if path.is_file()):
            planned_files.append(build_planned_file(
                source_path=source_path, root_folder=figures_folder / series_name,
                destination_root=snapshot_root / FIGURE_REUSE_FOLDER_NAME / series_name, source_kind="figures",
                file_specification={"path": str(source_path.relative_to(figures_folder / series_name))}))
    return planned_files


def build_planned_file(source_path: Path, root_folder: Path, destination_root: Path, source_kind: str,
                       file_specification: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the record of one file to write, its destination keeping its path relative to its root folder (a folder
    listing being written under the name of its 'output' option instead).

    Args:
        source_path (Path): The source file or folder.
        root_folder (Path): The experiments folder, data root or figures folder it is relative to.
        destination_root (Path): The folder of the snapshot it goes under.
        source_kind (str): "run", "dataset" or "figures", recorded in the manifest.
        file_specification (Dict[str, Any]): The entry of the specification, with its reduction and options.

    Returns:
        Dict[str, Any]: The planned file.

    Raises:
        ValueError: If the source leaves the root folder.
    """
    resolved_source = source_path.resolve()
    resolved_root = root_folder.resolve()
    if not resolved_source.is_relative_to(resolved_root):
        raise ValueError(f"{source_path} is outside {root_folder}, which the snapshot mirrors.")
    relative_path = resolved_source.relative_to(resolved_root)
    reduction = file_specification.get("reduction", "copy")
    if reduction == "folder_listing":
        relative_path = relative_path / file_specification["output"]
    if "compress" in file_specification:
        compression = file_specification["compress"]
        if compression not in COMPRESSION_SUFFIXES:
            raise ValueError(f"Unknown compression {compression!r}: one of {list(COMPRESSION_SUFFIXES)}.")
        relative_path = relative_path.with_name(relative_path.name + COMPRESSION_SUFFIXES[compression])
    options ={key: value for key, value in file_specification.items() if key not in ["path", "reduction"]}
    return {"source": resolved_source, "source_kind": source_kind, "source_relative": str(relative_path),
            "destination": destination_root / relative_path, "reduction": reduction, "options": options}


def merge_planned_files(planned_by_series: Dict[str, List[Dict[str, Any]]]) -> Dict[Path, Dict[str, Any]]:
    """
    Merges the files planned by every series, a file read by several series (the classification run of the
    foundations, classification and deployment series) being written once; two series asking for the same file with
    different reductions stop the writer, since one of them would read a file it did not ask for.

    Args:
        planned_by_series (Dict[str, List[Dict[str, Any]]]): The planned files of every series.

    Returns:
        Dict[Path, Dict[str, Any]]: The planned files by destination, with the 'series' reading them.

    Raises:
        ValueError: If two series plan the same destination differently.
    """
    merged_files: Dict[Path, Dict[str, Any]] = {}
    for series_name, planned_files in planned_by_series.items():
        for planned_file in planned_files:
            destination = planned_file["destination"]
            if destination in merged_files:
                existing_file = merged_files[destination]
                if (existing_file["source"], existing_file["reduction"], existing_file["options"]) != (
                        planned_file["source"], planned_file["reduction"], planned_file["options"]):
                    raise ValueError(f"{destination} is planned with {existing_file['reduction']} "
                                     f"{existing_file['options']} by {existing_file['series']} and with "
                                     f"{planned_file['reduction']} {planned_file['options']} by {series_name}: "
                                     f"give both series the same entry.")
                existing_file["series"].append(series_name)
                continue
            merged_files[destination] = {**planned_file, "series": [series_name]}
    return merged_files


def write_planned_file(planned_file: Dict[str, Any], maximum_file_bytes: int, allow_replace: bool) -> Dict[str, Any]:
    """
    Writes one snapshot file through its reduction, into a temporary file first, and checks its size against the
    git-friendly maximum; a file already in the snapshot with other bytes is replaced only when every series reading
    it is written in this call, so that writing one series never changes the file of another one.

    Args:
        planned_file (Dict[str, Any]): The merged planned file.
        maximum_file_bytes (int): The largest size of a snapshot file.
        allow_replace (bool): Whether a different existing file may be replaced.

    Returns:
        Dict[str, Any]: Its manifest record.

    Raises:
        ValueError: If the written file is too large, or would replace a different file without allow_replace.
    """
    destination = planned_file["destination"]
    temporary_path = destination.with_name(destination.name + ".snapshot_tmp")
    reduction_options = {key: value for key, value in planned_file["options"].items() if key != "compress"}
    find_reduction(reduction_name=planned_file["reduction"])(planned_file["source"], temporary_path, reduction_options)
    if planned_file["options"].get("compress") == "gzip":
        compress_file(file_path=temporary_path)
    written_bytes = temporary_path.stat().st_size
    if written_bytes > maximum_file_bytes:
        temporary_path.unlink()
        raise ValueError(f"{destination} would be {written_bytes / BYTES_PER_MEGABYTE:.2f} megabytes, above the "
                         f"{maximum_file_bytes / BYTES_PER_MEGABYTE:.0f} of the snapshots: reduce it in "
                         f"snapshot_files.yaml of {planned_file['series']}.")
    written_sha256 = compute_sha256(file_path=temporary_path)
    if destination.exists() and compute_sha256(file_path=destination) != written_sha256 and not allow_replace:
        temporary_path.unlink()
        raise ValueError(f"{destination} exists with other bytes and is also read by series not written now: write "
                         f"every series (no --series) to replace it.")
    temporary_path.replace(target=destination)
    source = planned_file["source"]
    return {"snapshot_path": None, "source_kind": planned_file["source_kind"],
            "source_path": planned_file["source_relative"], "reduction": planned_file["reduction"],
            "options": planned_file["options"],
            "source_bytes": source.stat().st_size if source.is_file() else None,
            "source_sha256": compute_sha256(file_path=source) if source.is_file() else None,
            "snapshot_bytes": written_bytes, "snapshot_sha256": written_sha256}


def read_settings(repository_root: Path, arguments: argparse.Namespace) -> Tuple[Dict[str, Any], Path, Path, Path]:
    """
    Reads tutorial_sources.yaml and the folders of the writer: the experiments folder and figures folder given on the
    command line or by the tutorial project file, and the snapshot folder.

    Args:
        repository_root (Path): The repository root.
        arguments (argparse.Namespace): The parsed arguments.

    Returns:
        Tuple[Dict[str, Any], Path, Path, Path]: The sources, the experiments folder, the figures folder and the
            snapshot folder.
    """
    sources = yaml.safe_load((repository_root / TUTORIALS_FOLDER / SOURCES_FILE_NAME).read_text(encoding="utf-8"))
    project = yaml.safe_load((repository_root / TUTORIALS_FOLDER / PROJECT_FILE_NAME).read_text(encoding="utf-8"))
    experiments_folder = Path(arguments.experiments_folder or project["experiments_folder"]).expanduser()
    figures_folder = Path(arguments.figures_folder or Path(project["cache_folder"]) / "figures").expanduser()
    snapshot_root = repository_root / sources["snapshots"]["folder"]
    return sources, experiments_folder, figures_folder, snapshot_root


def check_snapshot(snapshot_root: Path) -> None:
    """
    Compares every file of the manifests with the snapshot folder (presence, size and SHA-256), and lists the snapshot
    files no manifest names, without reading the runs.

    Args:
        snapshot_root (Path): The snapshot folder.

    Raises:
        ValueError: If a file is missing or differs from its manifest.
    """
    problems = []
    listed_paths = set()
    for manifest_path in sorted((snapshot_root / MANIFESTS_FOLDER_NAME).glob("*.json")):
        for record in json.loads(manifest_path.read_text(encoding="utf-8"))["files"]:
            snapshot_path = snapshot_root / record["snapshot_path"]
            listed_paths.add(snapshot_path)
            if not snapshot_path.exists():
                problems.append(f"missing {record['snapshot_path']} ({manifest_path.name})")
            elif compute_sha256(file_path=snapshot_path) != record["snapshot_sha256"]:
                problems.append(f"changed {record['snapshot_path']} ({manifest_path.name})")
    unlisted_paths = [path for path in snapshot_root.rglob("*") if path.is_file() and path not in listed_paths
                      and MANIFESTS_FOLDER_NAME not in path.relative_to(snapshot_root).parts
                      and path.name != "README.md"]
    for unlisted_path in sorted(unlisted_paths):
        print(f"Not in any manifest: {unlisted_path.relative_to(snapshot_root)}")
    if problems:
        raise ValueError("The snapshot differs from its manifests:\n" + "\n".join(problems))
    print(f"{len(listed_paths)} snapshot files match their manifests")


def main() -> None:
    """
    Writes the snapshot files of the series asked for and their manifests, then prints the size of the snapshot folder.
    """
    arguments = parse_arguments()
    repository_root = Path.cwd()
    sources, experiments_folder, figures_folder, snapshot_root = read_settings(repository_root=repository_root,
                                                                               arguments=arguments)
    if arguments.check:
        check_snapshot(snapshot_root=snapshot_root)
        return

    # Specifications of the series asked for, every series by default
    specification_paths = sorted((repository_root / TUTORIALS_FOLDER / "figures").glob(f"*/{SPECIFICATION_FILE_NAME}"))
    all_series = [path.parent.name for path in specification_paths]
    series_names = arguments.series or all_series
    unknown_series = sorted(set(series_names) - set(all_series))
    if unknown_series:
        raise ValueError(f"No {SPECIFICATION_FILE_NAME} for {unknown_series}; series with one: {all_series}.")

    # Files planned by every series, merged by destination
    data_root = Path(sources["snapshots"]["recorded_data_root"])
    planned_by_series = {}
    for series_name in series_names:
        specification = yaml.safe_load((repository_root / TUTORIALS_FOLDER / "figures" / series_name /
                                        SPECIFICATION_FILE_NAME).read_text(encoding="utf-8")) or {}
        planned_by_series[series_name] = plan_series_entries(
            series_name=series_name, specification=specification, sources=sources,
            experiments_folder=experiments_folder, data_root=data_root, figures_folder=figures_folder,
            snapshot_root=snapshot_root)
    merged_files = merge_planned_files(planned_by_series=planned_by_series)

    # Files written, then one manifest per series listing the files it reads
    maximum_file_bytes = int(sources["snapshots"]["maximum_file_megabytes"] * BYTES_PER_MEGABYTE)
    allow_replace = arguments.series is None or set(series_names) == set(all_series)
    records_by_series: Dict[str, List[Dict[str, Any]]] = {series_name: [] for series_name in series_names}
    for destination, planned_file in sorted(merged_files.items()):
        record = write_planned_file(planned_file=planned_file, maximum_file_bytes=maximum_file_bytes,
                                    allow_replace=allow_replace)
        record["snapshot_path"] = str(destination.relative_to(snapshot_root))
        for series_name in planned_file["series"]:
            records_by_series[series_name].append(record)
    (snapshot_root / MANIFESTS_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
    for series_name, records in records_by_series.items():
        manifest = {"series": series_name, "experiments_folder_label": sources["snapshots"]["experiments_label"],
                    "files": records, "snapshot_bytes": sum(record["snapshot_bytes"] for record in records)}
        (snapshot_root / MANIFESTS_FOLDER_NAME / f"{series_name}.json").write_text(
            data=json.dumps(obj=manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(f"{series_name}: {len(records)} files, {manifest['snapshot_bytes'] / BYTES_PER_MEGABYTE:.2f} megabytes")

    # Stale files removed when every series was written, then the size of the whole snapshot folder
    if arguments.prune:
        if not allow_replace:
            raise ValueError("--prune needs every series written (no --series).")
        written_paths = set(merged_files)
        for snapshot_path in sorted(snapshot_root.rglob("*")):
            relative_parts = snapshot_path.relative_to(snapshot_root).parts
            if (snapshot_path.is_file() and snapshot_path not in written_paths
                    and relative_parts[0] != MANIFESTS_FOLDER_NAME and snapshot_path.name != "README.md"):
                print(f"Pruned {snapshot_path.relative_to(snapshot_root)}")
                snapshot_path.unlink()
    total_bytes = sum(path.stat().st_size for path in snapshot_root.rglob("*") if path.is_file())
    maximum_total_megabytes = sources["snapshots"]["maximum_total_megabytes"]
    print(f"Snapshot folder: {total_bytes / BYTES_PER_MEGABYTE:.2f} megabytes "
          f"(at most {maximum_total_megabytes} wanted)")
    if total_bytes > maximum_total_megabytes * BYTES_PER_MEGABYTE:
        print(f"Warning: the snapshot folder is above {maximum_total_megabytes} megabytes")


if __name__ == "__main__":
    main()
