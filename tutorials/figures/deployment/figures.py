"""
Figures and facts of the deployment series of the tutorial videos, made by make_figures.py (python -m
tutorials.figures.make_figures --series deployment) at every build: the values and listings of the export and of
Triton read from the code of the repository, the README classifier, the reference detector of the detection benchmark
and the sequence labelling run of tutorials/tutorial_sources.yaml, and the real model repository of the serving
benchmark (export_values.py), then the figures and numbers of the serving benchmark, from the committed copies of its
results next to documentation/architecture_benchmarking/serving_benchmark_plan.md and the raw outputs of its detector
exported again (serving_benchmark.py), gathered into the facts.json the storyboards quote as {{deployment.<name>}}.
"""

from pathlib import Path
from typing import Any, Dict

from tutorials.figures.deployment.export_values import make_export_values
from tutorials.figures.deployment.serving_benchmark import make_serving_benchmark_figures
from tutorials.figures.figure_utilities import merge_facts


def make_figures(output_folder: Path, experiments_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Makes every figure and listing of the deployment series into its output folder and gives their facts; nothing is
    expensive enough to be reused between builds, so force changes nothing.

    Args:
        output_folder (Path): The figures folder of the series in the cache of the build.
        experiments_folder (Path): The folder of the real runs.
        force (bool): Whether the expensive figures are made again, unused by this series.

    Returns:
        Dict[str, Any]: The facts of every figure family, keyed by their dotted names.
    """
    return merge_facts(fact_groups=[make_export_values(output_folder=output_folder,
                                                       experiments_folder=experiments_folder),
                                    make_serving_benchmark_figures(output_folder=output_folder,
                                                                   experiments_folder=experiments_folder)])
