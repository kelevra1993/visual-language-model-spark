"""
Runner of the figure scripts of the tutorials, started by the tutorial build (build-tutorials of tutorial-video-builder)
with the Python of the repository environment, from the repository root, as `python -m tutorials.figures.make_figures`
(figures_module of tutorials/tutorial_project.yaml): the figures of a series are made by the make_figures function of
tutorials/figures/<series>/figures.py from the real runs, datasets and checkpoints (read on the CPU), into the figures
folder of the cache, and the numbers its storyboards quote are written next to them in facts.json. The figure scripts
use the code of the repository, so that a figure made after the code changed shows what the code does now; they read
the palette and fonts of the frames from tutorial_builder.layout and the repository, sources file and threads of the
build from tutorial_builder.figure_context, both put on their import path by the builder.
"""

import argparse
import importlib
import json

from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    """
    Reads the series to make and the folders given by the tutorial build.

    Returns:
        argparse.Namespace: The series, output folder, experiments folder and force flag.
    """
    parser = argparse.ArgumentParser(description="Makes the figures and facts of one tutorial series.")
    parser.add_argument("--series", required=True, help="Series, the name of a folder of tutorials/figures.")
    parser.add_argument("--output-folder", required=True, help="Folder receiving the figures and facts.json.")
    parser.add_argument("--experiments-folder", required=True, help="Folder of the real runs.")
    parser.add_argument("--force", action="store_true", help="Make again the figures the scripts would reuse.")
    return parser.parse_args()


def main() -> None:
    """
    Makes the figures of the series and writes its facts.json, sorted so that a change of a fact shows in a diff.
    """
    arguments = parse_arguments()
    output_folder = Path(arguments.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    figure_module = importlib.import_module(name=f"tutorials.figures.{arguments.series}.figures")
    facts = figure_module.make_figures(output_folder=output_folder,
                                       experiments_folder=Path(arguments.experiments_folder), force=arguments.force)
    (output_folder / "facts.json").write_text(data=json.dumps(obj=facts, indent=2, sort_keys=True), encoding="utf-8")
    print(f"{arguments.series}: {len(list(output_folder.rglob('*.png')))} figures, {len(facts)} facts in "
          f"{output_folder}")


if __name__ == "__main__":
    main()
