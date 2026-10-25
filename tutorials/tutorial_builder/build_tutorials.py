"""
Entry point of the tutorial build (tutorials/build_tutorials.sh runs it with the right environment), rebuilding the
tutorial videos from their storyboards, the repository and its real runs, in six steps, each stopping the build on a
problem: the figures and facts of the runs made by tutorials/figures in the repository environment, the reference check
of the storyboards against the repository and those facts (reference_check.py), the previews with their layout check
(previews.py), the videos (video_encoding.py), the check of their narration by speech recognition
(transcription_check.py), then the verification report and the index page (index_page.py) of the output folder.
Options select the series and chapters and stop after a step, so that a storyboard can be edited and previewed in
seconds.
"""

import argparse
import json
import os
import subprocess
import sys

from pathlib import Path
from typing import Any, Dict, List

import yaml

from tutorial_builder.index_page import write_index_page
from tutorial_builder.model_files import ensure_model_files
from tutorial_builder.previews import render_storyboard_previews
from tutorial_builder.reference_check import check_storyboard_references
from tutorial_builder.settings import PROCESS_NICENESS, REPOSITORY_ROOT, BuildSettings, create_build_settings
from tutorial_builder.storyboard_loading import list_storyboards
from tutorial_builder.transcription_check import check_video_narration
from tutorial_builder.video_encoding import make_video

# Word error rate of the narration above which a chapter is flagged in the verification report
WORD_ERROR_RATE_WARNING = 0.05


def parse_arguments() -> argparse.Namespace:
    """
    Reads the options of the build from the command line.

    Returns:
        argparse.Namespace: The options.
    """
    parser = argparse.ArgumentParser(description="Rebuilds the tutorial videos of visual-language-model-spark.")
    parser.add_argument("--series", nargs="+", default=None,
                        help="Series to build (names of tutorials/storyboards/series.yaml), every series by default.")
    parser.add_argument("--chapters", nargs="+", type=int, default=None,
                        help="Chapters to build within the series, every chapter by default.")
    parser.add_argument("--output-folder", default=None,
                        help="Folder receiving the videos and the index page (default ~/Data/tutorial_videos, or "
                             "VLMS_TUTORIALS_OUTPUT).")
    parser.add_argument("--cache-folder", default=None,
                        help="Folder of the models, narration, figures and previews (default ~/.cache/vlms_tutorials, "
                             "or VLMS_TUTORIALS_CACHE).")
    parser.add_argument("--experiments-folder", default=None,
                        help="Folder of the real runs read by the figures (default the experiments folder of the "
                             "repository, or VLMS_TUTORIALS_EXPERIMENTS).")
    parser.add_argument("--repository-python", default=None,
                        help="Python of the repository environment running the figure scripts (default .venv/bin/python "
                             "of the repository, or VLMS_TUTORIALS_REPOSITORY_PYTHON).")
    parser.add_argument("--chrome-executable", default=None,
                        help="Chrome capturing the HTML scenes (default the first Chrome on the PATH, or "
                             "VLMS_TUTORIALS_CHROME, else the Chromium of Playwright).")
    parser.add_argument("--check-references", action="store_true",
                        help="Only check the references of the storyboards to the repository.")
    parser.add_argument("--figures-only", action="store_true", help="Stop after making the figures.")
    parser.add_argument("--previews-only", action="store_true", help="Stop after the previews and their layout check.")
    parser.add_argument("--skip-figures", action="store_true", help="Reuse the figures already in the cache.")
    parser.add_argument("--force-figures", action="store_true",
                        help="Make again the figures that the figure scripts would otherwise reuse.")
    parser.add_argument("--skip-transcription", action="store_true", help="Do not transcribe the narration back.")
    parser.add_argument("--allow-layout-problems", action="store_true",
                        help="Render the videos even when the previews list layout problems.")
    return parser.parse_args()


def collect_figure_series(storyboard_paths: List[Path]) -> List[str]:
    """
    Lists the figure series the selected storyboards use, in the order they are first named.

    Args:
        storyboard_paths (List[Path]): The storyboards of the build.

    Returns:
        List[str]: The names of the figure series.
    """
    figure_series = []
    for storyboard_path in storyboard_paths:
        for series_name in yaml.safe_load(storyboard_path.read_text(encoding="utf-8")).get("figures", []):
            if series_name not in figure_series:
                figure_series.append(series_name)
    return figure_series


def make_series_figures(figure_series: List[str], settings: BuildSettings, force: bool) -> None:
    """
    Makes the figures and facts of every figure series with tutorials/figures/make_figures.py, run as a module of the
    repository by the Python of its environment (PyTorch and the repository code being needed to read the runs and run
    their checkpoints), on the CPU and at the lowest priority.

    Args:
        figure_series (List[str]): The figure series to make.
        settings (BuildSettings): The settings of the build.
        force (bool): Whether the figure scripts make again the figures they would otherwise reuse.

    Raises:
        FileNotFoundError: If the Python of the repository environment does not exist.
        subprocess.CalledProcessError: If a figure script fails.
    """
    if not settings.repository_python.exists():
        raise FileNotFoundError(f"No Python of the repository environment at {settings.repository_python}: run "
                                f"`uv sync` in {REPOSITORY_ROOT} or give --repository-python.")
    # The hints of transparent huge pages of numpy and PyTorch turned off: with the memory fragmented by the trainings,
    # every large allocation stalled in the kernel (a 16-image CPU inference took 124 s instead of 1.6 s)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NUMPY_MADVISE_HUGEPAGE": "0",
                   "THP_MEM_ALLOC_ENABLE": "0"}
    for series_name in figure_series:
        print(f"Making the figures of {series_name}")
        command = [str(settings.repository_python), "-m", "tutorials.figures.make_figures", "--series", series_name,
                   "--output-folder", str(settings.figures_folder / series_name), "--experiments-folder",
                   str(settings.experiments_folder)] + (["--force"] if force else [])
        subprocess.run(args=command, cwd=REPOSITORY_ROOT, env=environment, check=True)


def write_verification_report(records: List[Dict[str, Any]], layout_problems: List[str],
                              output_folder: Path) -> Path:
    """
    Writes the verification report of the build next to the videos: every chapter with its duration, the counts of its
    cached and synthesized sentences and the check of its narration, flagged above WORD_ERROR_RATE_WARNING, and the
    layout problems accepted by --allow-layout-problems.

    Args:
        records (List[Dict[str, Any]]): The records of the chapters built.
        layout_problems (List[str]): The layout problems of the previews.
        output_folder (Path): The output folder.

    Returns:
        Path: The Markdown report, written with its JSON version.
    """
    lines = ["# Tutorial build verification", "", "| Chapter | Duration | Sentences synthesized | Word error rate "
             "(spelling differences excluded) | Flag |", "|---|---|---|---|---|"]
    for record in records:
        check = record.get("narration_check")
        rate = check["word_error_rate_without_spellings"] if check else None
        flag = "check the narration" if rate is not None and rate > WORD_ERROR_RATE_WARNING else ""
        lines.append(f"| {record['series']} {record['chapter']}: {record['title']} | {record['duration']:.0f} s | "
                     f"{record['sentence_counts']['synthesized']} of "
                     f"{sum(record['sentence_counts'].values())} | "
                     f"{'not checked' if rate is None else f'{rate:.1%}'} | {flag} |")
    if layout_problems:
        lines += ["", "## Layout problems accepted", ""] + [f"- {problem}" for problem in layout_problems]
    for record in records:
        if record.get("narration_check") and record["narration_check"]["differences"]:
            lines += ["", f"## Narration differences of {record['output_name']}", ""]
            lines += [f"- {difference}" for difference in record["narration_check"]["differences"]]
    report_path = output_folder / "verification_report.md"
    report_path.write_text(data="\n".join(lines) + "\n", encoding="utf-8")

    # JSON version of the report, one entry per chapter
    chapter_entries = []
    for record in records:
        chapter_entry = {key: record[key] for key in ("series", "chapter", "title", "duration", "sentence_counts",
                                                      "build_seconds")}
        chapter_entry["narration_check"] = record.get("narration_check")
        chapter_entries.append(chapter_entry)
    (output_folder / "verification_report.json").write_text(
        data=json.dumps(obj={"chapters": chapter_entries, "layout_problems": layout_problems}, indent=2),
        encoding="utf-8")
    return report_path


def main() -> None:
    """
    Runs the steps of the build selected by the options, each stopping the build on a problem.
    """
    arguments = parse_arguments()
    os.nice(max(0, PROCESS_NICENESS - os.nice(0)))
    if os.environ.get("NUMPY_MADVISE_HUGEPAGE") != "0":
        print("NUMPY_MADVISE_HUGEPAGE is not 0: frames may stall when the memory is fragmented, run the build with "
              "tutorials/build_tutorials.sh.")
    settings = create_build_settings(output_folder=arguments.output_folder, cache_folder=arguments.cache_folder,
                                     experiments_folder=arguments.experiments_folder,
                                     repository_python=arguments.repository_python,
                                     chrome_executable=arguments.chrome_executable)
    storyboard_paths = list_storyboards(series_names=arguments.series, chapters=arguments.chapters)
    print(f"{len(storyboard_paths)} storyboards, videos in {settings.output_folder}, cache in {settings.cache_folder}")

    # Figures and facts of the runs, made before the reference check so that it reads the facts of this build (a fact
    # added to a figure script exists only once its figures are made again); a reference check alone makes none
    if not arguments.skip_figures and not arguments.check_references:
        make_series_figures(figure_series=collect_figure_series(storyboard_paths=storyboard_paths), settings=settings,
                            force=arguments.force_figures)

    # Reference check of the storyboards against the repository and the facts
    reference_problems = check_storyboard_references(storyboard_paths=storyboard_paths, settings=settings)
    if reference_problems:
        print("\n".join(["References of the storyboards no longer found in the repository:"] + reference_problems))
        sys.exit(1)
    print("References of the storyboards found in the repository")
    if arguments.check_references or arguments.figures_only:
        return

    # Previews and their layout check
    layout_problems = []
    for storyboard_path in storyboard_paths:
        preview_result = render_storyboard_previews(storyboard_path=storyboard_path, settings=settings,
                                                    preview_folder=settings.cache_folder / "previews" /
                                                    storyboard_path.parent.name / storyboard_path.stem)
        layout_problems += preview_result["problems"]
        print(f"Previews of {storyboard_path.name}: {len(preview_result['previews'])} frames, contact sheets "
              f"{[str(sheet_path) for sheet_path in preview_result['contact_sheets']]}")
    if layout_problems:
        print("\n".join(["Layout problems of the previews:"] + layout_problems))
        if not arguments.allow_layout_problems:
            sys.exit(1)
    if arguments.previews_only:
        return

    # Videos, the narration of every one checked by speech recognition
    model_folders = ensure_model_files(model_folder=settings.model_folder)
    records = []
    for storyboard_path in storyboard_paths:
        record = make_video(storyboard_path=storyboard_path, settings=settings)
        if not arguments.skip_transcription:
            record["narration_check"] = check_video_narration(
                video_path=settings.output_folder / record["series"] / record["video"], record=record,
                whisper_folder=model_folders["faster-whisper-base.en"])
            (settings.output_folder / record["series"] / f"{record['output_name']}.json").write_text(
                data=json.dumps(obj=record, indent=2), encoding="utf-8")
            print(f"Narration of {record['output_name']}: word error rate "
                  f"{record['narration_check']['word_error_rate_without_spellings']:.1%} without the spelling "
                  f"differences")
        records.append(record)

    # Verification report and index page of the output folder
    report_path = write_verification_report(records=records, layout_problems=layout_problems,
                                            output_folder=settings.output_folder)
    index_path = write_index_page(output_folder=settings.output_folder)
    print(f"Verification report {report_path}, index page {index_path}")


if __name__ == "__main__":
    main()
