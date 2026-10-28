"""
Paths and defaults of a tutorial build, shared by every step of tutorial_builder: where the repository and its tutorials
folder are, where the videos are written (the output folder, outside the repository by default), where the downloaded
models, the narration of every sentence and the figures are cached between builds, where the real runs read by the
figure scripts live, which Python runs those scripts (the environment of the repository, with PyTorch) and which Chrome
captures the HTML scenes. Every default can be changed by an environment variable or by an argument of
build_tutorials.py, so that a later build on another machine only changes the settings, not the code.
"""

import os
import shutil

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Folders of the repository, found from this file: tutorials/tutorial_builder/settings.py
TUTORIALS_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TUTORIALS_ROOT.parent
STORYBOARDS_FOLDER = TUTORIALS_ROOT / "storyboards"
ANIMATIONS_FOLDER = TUTORIALS_ROOT / "animations"
SOURCES_FILE = TUTORIALS_ROOT / "tutorial_sources.yaml"

# Environment variables overriding the defaults of the build folders and programs
OUTPUT_FOLDER_VARIABLE = "VLMS_TUTORIALS_OUTPUT"
CACHE_FOLDER_VARIABLE = "VLMS_TUTORIALS_CACHE"
EXPERIMENTS_FOLDER_VARIABLE = "VLMS_TUTORIALS_EXPERIMENTS"
REPOSITORY_PYTHON_VARIABLE = "VLMS_TUTORIALS_REPOSITORY_PYTHON"
CHROME_VARIABLE = "VLMS_TUTORIALS_CHROME"
THREADS_VARIABLE = "VLMS_TUTORIALS_THREADS"

# Chrome executables tried in this order when none is given, the bundled Chromium of Playwright being the last resort
CHROME_CANDIDATES = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]

# Lowest CPU priority, and the default threads of the synthesis, the transcription, the encoding and the figure
# scripts, so that the trainings keep the machine; --threads raises them when the machine is free
PROCESS_NICENESS = 19
DEFAULT_THREAD_COUNT = 4

# Threads of the figure scripts, run in their own processes, which receive the thread count of the build in
# VLMS_TUTORIALS_THREADS (make_series_figures of build_tutorials.py)
THREAD_COUNT = int(os.environ.get(THREADS_VARIABLE) or DEFAULT_THREAD_COUNT)


@dataclass(frozen=True)
class BuildSettings:
    """
    Folders and programs of one tutorial build, given to every step of the builder.

    Attributes:
        output_folder (Path): The folder receiving the videos, their subtitles, the verification report and the index.
        cache_folder (Path): The folder keeping the downloaded models, the narration of every sentence and the figures.
        experiments_folder (Path): The folder of the real runs read by the figure scripts.
        repository_python (Path): The Python of the repository environment, running the figure scripts.
        chrome_executable (Optional[Path]): The Chrome capturing the HTML scenes, None for the Chromium of Playwright.
        thread_count (int): The threads of the synthesis, the transcription, the encoding and the figure scripts.
    """
    output_folder: Path
    cache_folder: Path
    experiments_folder: Path
    repository_python: Path
    chrome_executable: Optional[Path]
    thread_count: int

    @property
    def model_folder(self) -> Path:
        """
        Gives the folder of the downloaded voice and speech recognition models.

        Returns:
            Path: The models folder of the cache.
        """
        return self.cache_folder / "models"

    @property
    def narration_folder(self) -> Path:
        """
        Gives the folder of the synthesized sentences, one WAV file per sentence, voice and pace.

        Returns:
            Path: The narration folder of the cache.
        """
        return self.cache_folder / "narration"

    @property
    def figures_folder(self) -> Path:
        """
        Gives the folder of the figures made by the figure scripts, one subfolder per series with its facts.json.

        Returns:
            Path: The figures folder of the cache.
        """
        return self.cache_folder / "figures"


def find_chrome_executable(requested_path: Optional[str]) -> Optional[Path]:
    """
    Finds the Chrome that captures the HTML scenes: the requested one, else the one of the environment variable, else
    the first Chrome or Chromium on the PATH, else None so that Playwright uses its own Chromium (installed with
    `playwright install chromium`).

    Args:
        requested_path (Optional[str]): The executable given on the command line, or None.

    Returns:
        Optional[Path]: The Chrome executable, or None for the Chromium of Playwright.

    Raises:
        FileNotFoundError: If a requested executable does not exist.
    """
    requested_path = requested_path or os.environ.get(CHROME_VARIABLE)
    if requested_path:
        executable_path = Path(requested_path).expanduser()
        if not executable_path.exists():
            raise FileNotFoundError(f"No Chrome executable at {executable_path} (argument --chrome-executable or "
                                    f"environment variable {CHROME_VARIABLE}).")
        return executable_path

    # First Chrome or Chromium found on the PATH
    for candidate_name in CHROME_CANDIDATES:
        candidate_path = shutil.which(candidate_name)
        if candidate_path is not None:
            return Path(candidate_path)
    return None


def create_build_settings(output_folder: Optional[str], cache_folder: Optional[str], experiments_folder: Optional[str],
                          repository_python: Optional[str], chrome_executable: Optional[str],
                          thread_count: Optional[int]) -> BuildSettings:
    """
    Gathers the settings of a build from the command line, the environment variables and the defaults, in this order
    of priority: videos in ~/Data/tutorial_videos, caches in ~/.cache/vlms_tutorials, runs in the experiments folder of
    the repository, figure scripts run by the Python of the repository environment (.venv).

    Args:
        output_folder (Optional[str]): The output folder given on the command line, or None.
        cache_folder (Optional[str]): The cache folder given on the command line, or None.
        experiments_folder (Optional[str]): The experiments folder given on the command line, or None.
        repository_python (Optional[str]): The Python of the repository given on the command line, or None.
        chrome_executable (Optional[str]): The Chrome executable given on the command line, or None.
        thread_count (Optional[int]): The threads given on the command line, or None for VLMS_TUTORIALS_THREADS or
            DEFAULT_THREAD_COUNT.

    Returns:
        BuildSettings: The settings of the build.

    Raises:
        ValueError: If the thread count is not a positive integer.
    """
    # Threads from the argument, else the environment variable, else the default leaving the machine to the trainings
    resolved_thread_count = (thread_count if thread_count is not None else
                             int(os.environ.get(THREADS_VARIABLE) or DEFAULT_THREAD_COUNT))
    if resolved_thread_count < 1:
        raise ValueError(f"The thread count must be a positive integer (argument --threads or environment variable "
                         f"{THREADS_VARIABLE}), got {resolved_thread_count}.")

    # Every folder from its argument, else its environment variable, else its default
    resolved_output_folder = Path(output_folder or os.environ.get(OUTPUT_FOLDER_VARIABLE) or
                                  Path.home() / "Data" / "tutorial_videos").expanduser()
    resolved_cache_folder = Path(cache_folder or os.environ.get(CACHE_FOLDER_VARIABLE) or
                                 Path.home() / ".cache" / "vlms_tutorials").expanduser()
    resolved_experiments_folder = Path(experiments_folder or os.environ.get(EXPERIMENTS_FOLDER_VARIABLE) or
                                       REPOSITORY_ROOT / "experiments").expanduser()
    resolved_repository_python = Path(repository_python or os.environ.get(REPOSITORY_PYTHON_VARIABLE) or
                                      REPOSITORY_ROOT / ".venv" / "bin" / "python").expanduser()
    return BuildSettings(output_folder=resolved_output_folder, cache_folder=resolved_cache_folder,
                         experiments_folder=resolved_experiments_folder, repository_python=resolved_repository_python,
                         chrome_executable=find_chrome_executable(requested_path=chrome_executable),
                         thread_count=resolved_thread_count)
