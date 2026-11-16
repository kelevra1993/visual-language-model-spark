"""
Helpers shared by the figure scripts of the tutorial videos (tutorials/figures/<series>/), which make_figures.py runs
with the Python of the repository at every build: the real runs and repository files the figures are drawn from, named
in tutorials/tutorial_sources.yaml; the palette, the chart size and the Inter and Hack fonts of the video frames
(tutorial_builder/layout.py and its fonts folder, of the tutorial-video-builder package the build puts on the import
path), so that a chart sits on the card of an image scene like the text around
it; the PNG files written at their exact pixel size and the pixel boxes of their panels, which the storyboards use as
highlight boxes and reveal regions through the facts; the trained model of the classification run, read on the CPU with
its input images, for the model figures of the foundations and classification series; the metric lines of a console
summary cut from the log of a run written again by the console formatter of the current code with their logged values,
for the console summary scenes; and the reuse of the expensive figures between builds unless forced, their facts kept
in a small JSON file next to them.
"""

import functools
import json
import math
import re

import cv2
import matplotlib
import numpy
import torch
import yaml

from matplotlib import font_manager, pyplot
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Type

from tutorial_builder.layout import (CARD_BORDER_COLOUR, CARD_COLOUR, DIM_TEXT_COLOUR,
                                     MONOSPACE_BOLD_FONT_PATH, MONOSPACE_FONT_PATH, NAMED_COLOURS,
                                     SANS_BOLD_FONT_PATH, SANS_FONT_PATH, SANS_SEMIBOLD_FONT_PATH,
                                     TEXT_COLOUR, VIEWPORT_HEIGHT, VIEWPORT_WIDTH)
from tutorial_builder.figure_context import find_repository_root, find_sources_file, read_thread_count
from vision_training.configuration_processing.configuration_loader import load_configuration
from vision_training.data.preprocessing import preprocess_image, read_image
from vision_training.models.model_loading import load_trained_model
from vision_training.models.task_model import TaskModel
from vision_training.trainers.base_trainer import BaseTrainer
from vision_training.trainers.checkpointing import get_checkpoint_path, get_last_checkpoint_iteration
from vision_training.trainers.metric_tracking import BEST_MARKER, MetricTracker

# Charts drawn without a window: the figure scripts run in a subprocess of the build, a display being there or not
matplotlib.use(backend="Agg")

# Size of the charts in pixels: the viewport of an image scene, so that a chart is shown without scaling, at 100 dots
# per inch, the sizes of the texts being given in pixels and converted to points
CHART_WIDTH = VIEWPORT_WIDTH
CHART_HEIGHT = VIEWPORT_HEIGHT
DOTS_PER_INCH = 100
POINTS_PER_PIXEL = 72 / DOTS_PER_INCH

# Colours of the charts in matplotlib notation, those of the video frames: the card surface the figures are shown on,
# its border for the grids and axes, the primary and secondary text greys, and the series in the order of the
# categorical palette of the frames (blue, aqua, yellow, magenta, violet), validated against the card surface
SURFACE_COLOUR = "#%02x%02x%02x" % CARD_COLOUR
GRID_COLOUR = "#%02x%02x%02x" % CARD_BORDER_COLOUR
TEXT_PRIMARY_COLOUR = "#%02x%02x%02x" % TEXT_COLOUR
TEXT_SECONDARY_COLOUR = "#%02x%02x%02x" % DIM_TEXT_COLOUR
SERIES_COLOURS = ["#%02x%02x%02x" % NAMED_COLOURS[colour_name] for colour_name in
                  ["blue", "aqua", "yellow", "magenta", "violet"]]

# Font families of the charts, registered from the fonts of the builder: Inter for the text, Hack for the file names
# and the configuration excerpts
SANS_FONT_FAMILY = "Inter"
MONOSPACE_FONT_FAMILY = "Hack"

# Pixels added on every side of the plot area of a panel to make its reveal region, wide enough for the end dots drawn
# over the edges of the axes
REVEAL_MARGIN = 14

# Test image of the model figures of the foundations and classification series, a star of the synthetic shapes test
# split read from the test folder of the classification run
STAR_IMAGE_NAME = "test_005452.png"

# Significant digits kept in the facts, so that a change of a fact shows in a diff of facts.json without the noise of
# the last digits of a float
FACT_SIGNIFICANT_DIGITS = 6

# Prefix of the metric lines of a console summary, and their pattern once their colours are removed, whatever code
# printed the log: the split, the display name, the value (a rate between 0 and 1 in the runs trained before
# 2026-11-12, with its percent sign since), its change since the previous summary (that of a rate between 0 and 1, then
# in "points", then with the percent sign) and the best marker
CONSOLE_METRIC_LINE_PREFIX = "Moving Average of "
CONSOLE_METRIC_LINE_PATTERN = re.compile(
    pattern=rf"^{CONSOLE_METRIC_LINE_PREFIX}(?P<split_name>Train|Valid) (?P<display_name>.+?) +: +"
            rf"(?P<value>-?\d+\.\d+)(?P<value_percent> ?%)?(?: +(?P<change>[+-]\d+\.\d+)(?P<change_unit> ?%| points)?)?"
            rf"(?: +(?P<best_marker>{BEST_MARKER}))?$")

# Colour codes of the console, removed from the metric lines the current formatter writes again
CONSOLE_COLOUR_PATTERN = re.compile(pattern=r"\x1b\[[0-9;]*m")


def read_tutorial_sources() -> Dict[str, Any]:
    """
    Reads tutorials/tutorial_sources.yaml, the real runs and repository files the figure scripts draw from, so that a
    run trained again under another name only changes that file.

    Returns:
        Dict[str, Any]: The 'runs' and 'repository_files' sections.

    Raises:
        FileNotFoundError: If the sources file is missing.
    """
    if not find_sources_file().exists():
        raise FileNotFoundError(f"No sources file {find_sources_file()}: it names the runs read by the figure scripts.")
    return yaml.safe_load(find_sources_file().read_text(encoding="utf-8"))


def find_run_folder(experiments_folder: Path, run_key: str) -> Path:
    """
    Finds the folder of a real run of tutorial_sources.yaml in the experiments folder of the build, so that a missing
    run stops the build with what it takes to make it again (its 'trained_with') rather than with a bare missing file.

    Args:
        experiments_folder (Path): The experiments folder of the build (--experiments-folder of make_figures.py).
        run_key (str): The name of the run under 'runs', e.g. "classification".

    Returns:
        Path: The run folder.

    Raises:
        KeyError: If the run is not in the sources file or has no single 'folder'.
        FileNotFoundError: If the run folder does not exist.
    """
    runs = read_tutorial_sources()["runs"]
    if run_key not in runs or "folder" not in runs[run_key]:
        raise KeyError(f"No run {run_key} with a 'folder' in {find_sources_file()}, whose runs are {sorted(runs)}.")
    run_folder = experiments_folder / runs[run_key]["folder"]
    if not run_folder.is_dir():
        raise FileNotFoundError(f"No run folder {run_folder} for the {run_key} run of {find_sources_file()}: make it "
                                f"again with {runs[run_key]['trained_with']}, then update its folder in "
                                f"{find_sources_file().name} if its run name changed.")
    return run_folder


def find_repository_file(file_key: str) -> Path:
    """
    Finds a file of the repository named under 'repository_files' of tutorial_sources.yaml (the benchmark results, the
    configuration of the learning rate schedule), relative to the repository root.

    Args:
        file_key (str): The name of the file under 'repository_files', e.g. "benchmark_results".

    Returns:
        Path: The absolute path of the file.

    Raises:
        KeyError: If the sources file does not name the file.
        FileNotFoundError: If the file does not exist.
    """
    repository_files = read_tutorial_sources()["repository_files"]
    if file_key not in repository_files:
        raise KeyError(f"No repository file {file_key} in {find_sources_file()}, whose files are "
                       f"{sorted(repository_files)}.")
    file_path = find_repository_root() / repository_files[file_key]
    if not file_path.exists():
        raise FileNotFoundError(f"No file {file_path} for the repository file {file_key} of {find_sources_file()}.")
    return file_path


def read_run_configuration(run_folder: Path) -> Dict[str, Any]:
    """
    Reads the training configuration saved in a run folder, which gives its dataset folders, its image settings and
    the settings of its model, as the run used them.

    Args:
        run_folder (Path): The run folder.

    Returns:
        Dict[str, Any]: The configuration, with its ExperimentConfiguration, Optimization, Data and model sections.
    """
    return load_configuration(configuration_path=run_folder / "training_configuration.yaml")


def build_console_metric_tracker(run_folder: Path, trainer_class: Type[BaseTrainer],
                                 trainer_attributes: Dict[str, Any]) -> MetricTracker:
    """
    Builds the console metric tracker of a run as its trainer builds it, with the tracked metrics and rates of the
    current code, so that the console summary scenes of the series show a summary cut from the log of an older run in
    the columns and units a run trained today prints (reformat_console_summary). The trainer is made without its
    constructor, which would build the loaders and the model, and receives the attributes its metric lists read.

    Args:
        run_folder (Path): The run, whose saved configuration gives the batch size and the validation ratio.
        trainer_class (Type[BaseTrainer]): The trainer of the run, e.g. DetectionTrainer.
        trainer_attributes (Dict[str, Any]): The attributes read by get_tracked_metrics and get_percent_tracked_metrics
            of the trainer, e.g. {"method": "simclr"} for the self-supervised trainer,
            {"decorrelation_activated": False} for the sequence trainers, {} for the detection trainers.

    Returns:
        MetricTracker: The tracker, without previous or best values.
    """
    # Trainer holding only the attributes its metric lists read
    trainer = object.__new__(trainer_class)
    for attribute_name, attribute_value in trainer_attributes.items():
        setattr(trainer, attribute_name, attribute_value)

    # Tracker set up with the settings of the run, as BaseTrainer sets it up
    experiment_configuration = read_run_configuration(run_folder=run_folder)["ExperimentConfiguration"]
    return MetricTracker(tracked_metrics_mapping=trainer.get_tracked_metrics(),
                         batch_size=int(experiment_configuration["batch_size"]),
                         validation_iteration_ratio=int(experiment_configuration["validation_iteration_ratio"]),
                         sample_name=trainer_class.SAMPLE_NAME, counts_frames=trainer_class.COUNTS_FRAMES,
                         percent_metric_keys=trainer.get_percent_tracked_metrics())


def reformat_console_metric_line(line: str, metric_tracker: MetricTracker) -> str:
    """
    Writes a metric line of a console summary read from the log of a run again with the console formatter of the
    current code (MetricTracker.format_metric_line), keeping its logged value, change and best marker, so that a scene
    shows the columns and units a run trained today prints even from a log of an older code: rates logged between 0
    and 1 or with their percent sign, their changes between 0 and 1, in "points" or with the percent sign.

    Args:
        line (str): The metric line of the log, its colours and links removed.
        metric_tracker (MetricTracker): The tracker of the run (build_console_metric_tracker), whose previous and best
            values of the metric of the line are set from the log before it writes the line.

    Returns:
        str: The line as the current formatter prints it, its colours removed.

    Raises:
        ValueError: If the line is not a metric line, names a metric the trainer does not track, writes a value the
            trainer does not track as a rate with a percent sign, writes a change in another unit than its value, or
            holds a best marker without a change.
    """
    line_match = CONSOLE_METRIC_LINE_PATTERN.match(string=line.rstrip())
    if line_match is None:
        raise ValueError(f"Expected a console summary metric line matching {CONSOLE_METRIC_LINE_PATTERN.pattern!r}, "
                         f"got {line!r}.")
    split_name, display_name = line_match.group("split_name"), line_match.group("display_name")
    metric_keys = {name: metric_key for metric_key, name in metric_tracker.tracked_metrics_mapping.items()}
    if display_name not in metric_keys:
        raise ValueError(f"Expected one of the tracked metrics {list(metric_keys)} in the console line {line!r}.")
    metric_key = metric_keys[display_name]

    # A logged percent sign only on a rate, and a change in the unit of its value
    logged_in_percent = line_match.group("value_percent") is not None
    change_text = line_match.group("change")
    if logged_in_percent and metric_key not in metric_tracker.percent_metric_keys:
        raise ValueError(f"Expected {display_name} without a percent sign, as the trainer does not track it as a rate, "
                         f"in the console line {line!r}.")
    if change_text is not None and (line_match.group("change_unit") is not None) != logged_in_percent:
        raise ValueError(f"Expected the change of {display_name} in the unit of its value in the console line "
                         f"{line!r}.")
    if change_text is None and line_match.group("best_marker") is not None:
        raise ValueError(f"Expected a change before the best marker in the console line {line!r}.")

    # The tracker receives the rates between 0 and 1, a log written in percent giving them and their changes times 100
    scale = 100.0 if logged_in_percent else 1.0
    current_value = float(line_match.group("value")) / scale

    # A logged change comes back from a previous value set that far from the value, and a logged best marker from a
    # best value any value beats, the value itself standing as the best when the log marked none
    if change_text is None:
        metric_tracker.previous_values[split_name].pop(metric_key, None)
    else:
        metric_tracker.previous_values[split_name][metric_key] = current_value - float(change_text) / scale
        unbeaten_value = math.inf if metric_tracker.is_lower_better(metric_key=metric_key) else -math.inf
        metric_tracker.best_values[split_name][metric_key] = (unbeaten_value if line_match.group("best_marker") else
                                                              current_value)
    formatted_line = metric_tracker.format_metric_line(split_name=split_name, metric_key=metric_key,
                                                       display_name=display_name, current_value=current_value)
    return CONSOLE_COLOUR_PATTERN.sub(repl="", string=formatted_line)


def reformat_console_summary(lines: List[str], metric_tracker: MetricTracker) -> List[str]:
    """
    Writes the metric lines of a console summary read from the log of a run again with the console formatter of the
    current code (reformat_console_metric_line), the header, rules and interval lines kept as logged since the current
    code prints them the same way, for the console summary scenes of the series.

    Args:
        lines (List[str]): The lines of the summary, their colours and links removed.
        metric_tracker (MetricTracker): The tracker of the run (build_console_metric_tracker).

    Returns:
        List[str]: The lines of the summary, its metric lines as the current formatter prints them.
    """
    return [reformat_console_metric_line(line=line, metric_tracker=metric_tracker)
            if line.startswith(CONSOLE_METRIC_LINE_PREFIX) else line for line in lines]


@functools.lru_cache(maxsize=1)
def register_fonts() -> None:
    """
    Registers the fonts of the video frames for matplotlib, once per process: Inter (regular, semibold and bold) for
    the text of the charts and Hack for their file names and configuration excerpts, read from the fonts folder of the
    builder (tutorial_builder.layout) so that every machine draws the same charts, Inter becoming the default family
    of the charts.

    Raises:
        FileNotFoundError: If a font file of the builder is missing.
    """
    for font_path in [SANS_FONT_PATH, SANS_SEMIBOLD_FONT_PATH, SANS_BOLD_FONT_PATH, MONOSPACE_FONT_PATH,
                      MONOSPACE_BOLD_FONT_PATH]:
        if not font_path.exists():
            raise FileNotFoundError(f"Missing font file {font_path}: the fonts of the videos live in "
                                    f"{font_path.parent}.")
        font_manager.fontManager.addfont(path=str(font_path))
    pyplot.rcParams["font.family"] = SANS_FONT_FAMILY


def format_thousands(value: float, position: int) -> str:
    """
    Formats an iteration tick of a chart with a thousands separator (1,500), the tick formatter of the training curves
    and benchmark panels.

    Args:
        value (float): The tick value.
        position (int): The position of the tick, given by matplotlib.

    Returns:
        str: The tick label.
    """
    return f"{value:,.0f}"


def style_chart_panel(axes: matplotlib.axes.Axes, title: str, x_label: str, y_label: Optional[str]) -> None:
    """
    Gives a chart panel the quiet anatomy of the reference charts of the videos: the card surface, a solid hairline
    grid and the left and bottom axes one step off the surface, secondary grey ticks and axis labels, and the title in
    the primary grey on the left. The limits, ticks and tick formats are set by every chart.

    Args:
        axes (matplotlib.axes.Axes): The panel.
        title (str): The title of the panel.
        x_label (str): The name of the horizontal axis.
        y_label (Optional[str]): The name of the vertical axis, or None for a panel sharing the scale of its left
            neighbour.
    """
    # Surface, hairline grid and the two axes kept, the others removed
    axes.set_facecolor(SURFACE_COLOUR)
    axes.grid(visible=True, color=GRID_COLOUR, linewidth=POINTS_PER_PIXEL, linestyle="-")
    axes.set_axisbelow(True)
    for side in ["top", "right"]:
        axes.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        axes.spines[side].set_color(GRID_COLOUR)
        axes.spines[side].set_linewidth(POINTS_PER_PIXEL)

    # Ticks, axis labels and title
    axes.tick_params(axis="both", colors=TEXT_SECONDARY_COLOUR, labelsize=18 * POINTS_PER_PIXEL, length=0, pad=8)
    axes.set_xlabel(xlabel=x_label, color=TEXT_SECONDARY_COLOUR, fontsize=19 * POINTS_PER_PIXEL, labelpad=8)
    if y_label is not None:
        axes.set_ylabel(ylabel=y_label, color=TEXT_SECONDARY_COLOUR, fontsize=19 * POINTS_PER_PIXEL, labelpad=10)
    axes.set_title(label=title, color=TEXT_PRIMARY_COLOUR, fontsize=24 * POINTS_PER_PIXEL, loc="left", pad=14)


def save_figure(figure: matplotlib.figure.Figure, output_path: Path) -> None:
    """
    Writes a chart as a PNG file of its exact pixel size (its size in inches at DOTS_PER_INCH), on the card colour,
    the size the storyboards place it at and the frame of the pixel boxes of its facts, and refuses a file of another
    size, which would shift every box.

    Args:
        figure (matplotlib.figure.Figure): The chart.
        output_path (Path): The PNG file, its folder created if missing.

    Raises:
        ValueError: If the written file does not have the pixel size of the figure.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(fname=output_path, dpi=DOTS_PER_INCH, facecolor=SURFACE_COLOUR)

    # Size of the written file against the size of the figure
    expected_size = (round(figure.get_figwidth() * DOTS_PER_INCH), round(figure.get_figheight() * DOTS_PER_INCH))
    with Image.open(fp=output_path) as written_image:
        written_size = written_image.size
    if written_size != expected_size:
        raise ValueError(f"{output_path} was written at {written_size[0]}x{written_size[1]} pixels instead of the "
                         f"{expected_size[0]}x{expected_size[1]} of its figure.")


def compute_pixel_box(figure: matplotlib.figure.Figure, axes: matplotlib.axes.Axes, with_labels: bool,
                      margin: int) -> List[int]:
    """
    Gives the box of a chart panel in pixels of its PNG file from the top left corner, the box format of the
    storyboards: with its title and tick labels for a highlight box, or for a reveal region its plot area joined with
    the labels drawn on it (which may pass its edges), widened by a margin. It is measured on the figure as drawn, so
    that the boxes follow the fonts and the data.

    Args:
        figure (matplotlib.figure.Figure): The drawn figure.
        axes (matplotlib.axes.Axes): The panel.
        with_labels (bool): Whether the box holds the title and tick labels, the plot area and its labels otherwise.
        margin (int): The pixels added on every side.

    Returns:
        List[int]: The left, top, right and bottom of the box.
    """
    renderer = figure.canvas.get_renderer()
    if with_labels:
        box = axes.get_tightbbox(renderer=renderer)
    else:
        box = matplotlib.transforms.Bbox.union(bboxes=[axes.get_window_extent(renderer=renderer)] +
                                               [text.get_window_extent(renderer=renderer) for text in axes.texts])

    # Matplotlib measures from the bottom left corner, the storyboards from the top left corner
    figure_height = figure.bbox.height
    return [int(numpy.floor(box.x0)) - margin, int(numpy.floor(figure_height - box.y1)) - margin,
            int(numpy.ceil(box.x1)) + margin, int(numpy.ceil(figure_height - box.y0)) + margin]


def round_fact(value: float) -> float:
    """
    Rounds a measured value of a fact (an accuracy, a logit, a learning rate) to FACT_SIGNIFICANT_DIGITS significant
    digits, as a plain float that facts.json can hold, the storyboards formatting it further when they quote it.

    Args:
        value (float): The value, a Python, NumPy or PyTorch scalar.

    Returns:
        float: The rounded value.
    """
    return float(f"{float(value):.{FACT_SIGNIFICANT_DIGITS}g}")


def merge_facts(fact_groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Gathers the facts of the figure families of a series into the dictionary make_figures.py writes as facts.json,
    refusing a name given by two families, which would silently replace one of the two numbers.

    Args:
        fact_groups (List[Dict[str, Any]]): The facts of every family, keyed by their dotted names.

    Returns:
        Dict[str, Any]: Every fact of the series.

    Raises:
        ValueError: If two families give a fact of the same name.
    """
    merged_facts = {}
    for fact_group in fact_groups:
        repeated_names = sorted(set(merged_facts) & set(fact_group))
        if repeated_names:
            raise ValueError(f"The facts {repeated_names} are given by two figure families of the series.")
        merged_facts.update(fact_group)
    return merged_facts


def compute_file_fingerprint(file_paths: List[Path]) -> str:
    """
    Describes the files an expensive figure family reads (a predictions file, a checkpoint, a labels file) by their
    name, size in bytes and modification time in nanoseconds, the key a reused family is compared on, so that a run
    trained or evaluated again under the same name and the same paths makes its figures again instead of reusing those
    of the files it replaced.

    Args:
        file_paths (List[Path]): The files read by the family.

    Returns:
        str: The fingerprint, one "name:size:modification time" entry per file, joined by semicolons.

    Raises:
        FileNotFoundError: If one of the files is missing.
    """
    entries = []
    for file_path in file_paths:
        file_status = file_path.stat()
        entries.append(f"{file_path.name}:{file_status.st_size}:{file_status.st_mtime_ns}")
    return ";".join(entries)


def read_reused_facts(facts_path: Path, required_paths: List[Path], force: bool) -> Optional[Dict[str, Any]]:
    """
    Decides whether the expensive figures of a family (a checkpoint inference, the augmentation examples) are reused
    from a previous build: not when forced, nor when the JSON file of their facts is missing, nor when a figure it
    lists or a required path is missing; their facts are then read back from that file.

    Args:
        facts_path (Path): The JSON file written next to the figures by write_reused_facts.
        required_paths (List[Path]): The files that must exist besides those the JSON file lists, such as the figure
            of an augmentation type added to the repository since the last build.
        force (bool): Whether the figures are made again whatever is on disk (--force of make_figures.py).

    Returns:
        Optional[Dict[str, Any]]: The facts of the reused figures, or None when the figures must be made again.
    """
    if force or not facts_path.exists():
        return None
    record = json.loads(facts_path.read_text(encoding="utf-8"))
    listed_paths = [facts_path.parent / figure_name for figure_name in record["figures"]]
    if not all(path.exists() for path in listed_paths + required_paths):
        return None
    print(f"Reusing the figures of {facts_path.parent} listed in {facts_path.name} (--force makes them again)")
    return record["facts"]


def write_reused_facts(facts_path: Path, figure_paths: List[Path], facts: Dict[str, Any]) -> None:
    """
    Writes the JSON file that lets the next build reuse the expensive figures of a family: the figures it made,
    relative to the folder of the file, and their facts. It is written last and replaced in one step, so that a build
    stopped halfway never leaves a file naming figures that were not all written.

    Args:
        facts_path (Path): The JSON file, next to the figures.
        figure_paths (List[Path]): The figures of the family, in the folder of the file or below it.
        facts (Dict[str, Any]): The facts of the figures.
    """
    record = {"figures": sorted(str(path.relative_to(facts_path.parent)) for path in figure_paths), "facts": facts}
    temporary_path = facts_path.with_suffix(".json.tmp")
    temporary_path.write_text(data=json.dumps(obj=record, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(target=facts_path)


def load_classification_model(run_folder: Path) -> Tuple[TaskModel, int]:
    """
    Rebuilds the trained model of the classification run at its last checkpoint, on the CPU and in evaluation mode,
    with the loader of the evaluation and deployment scripts of the repository, so that the model figures show what
    the trained network computes (its trained weights, not their moving average). The threads of PyTorch are limited
    to those of the build, the trainings keeping the processors.

    Args:
        run_folder (Path): The folder of the classification run.

    Returns:
        Tuple[TaskModel, int]: The model and the iteration of its checkpoint.

    Raises:
        FileNotFoundError: If the run has no checkpoint.
    """
    torch.set_num_threads(read_thread_count())
    checkpoint_iteration = get_last_checkpoint_iteration(weights_directory=run_folder / "Weights")

    # Iteration 0 means no checkpoint, for which get_checkpoint_path raises the error naming the folder
    checkpoint_path = get_checkpoint_path(experiment_folder=run_folder, iteration=checkpoint_iteration or None)
    model = load_trained_model(configuration_path=run_folder / "training_configuration.yaml",
                               checkpoint_path=checkpoint_path, device=torch.device("cpu"), use_averaged_weights=False)
    return model, checkpoint_iteration


def find_star_image(run_configuration: Dict[str, Any]) -> Path:
    """
    Finds the test star of the model figures in the test split of the classification run, the image whose feature
    maps, gates, logits and probabilities the videos show.

    Args:
        run_configuration (Dict[str, Any]): The saved configuration of the classification run.

    Returns:
        Path: The image file.

    Raises:
        FileNotFoundError: If the image is not in the test split of the run.
    """
    image_path = Path(run_configuration["ExperimentConfiguration"]["test_data_folder"]) / "data" / STAR_IMAGE_NAME
    if not image_path.exists():
        raise FileNotFoundError(f"No test star {image_path} in the test split of the classification run.")
    return image_path


def read_network_input(image_path: Path, image_size: int, keep_ratio: bool) -> torch.Tensor:
    """
    Reads an image the way the data loaders of the repository do (read_image and preprocess_image, BGR, resized to
    the square input of the network), as the [1, 3, H, W] float tensor in [0, 255] the model expects.

    Args:
        image_path (Path): The image file.
        image_size (int): The input size of the network, 'Data.image_settings.size' of the run.
        keep_ratio (bool): Whether the aspect ratio is kept by padding, 'Data.image_settings.keep_ratio' of the run.

    Returns:
        torch.Tensor: The input tensor of the image.
    """
    image = preprocess_image(image=read_image(image_path=image_path), image_size=image_size, keep_ratio=keep_ratio)
    return torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(dim=0)


def write_network_input(input_tensor: torch.Tensor, output_path: Path) -> None:
    """
    Writes the input of the network for an image (the first of a batch) as a PNG file, the image the model figures
    show next to what the network computes from it.

    Args:
        input_tensor (torch.Tensor): The [B, 3, H, W] float BGR input in [0, 255].
        output_path (Path): The PNG file, its folder created if missing.

    Raises:
        OSError: If the file cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(filename=str(output_path), img=input_tensor[0].permute(1, 2, 0).numpy().astype(numpy.uint8)):
        raise OSError(f"The figure {output_path} could not be written.")
