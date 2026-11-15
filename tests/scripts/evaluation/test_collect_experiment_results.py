"""
Tests of scripts/evaluation/collect_experiment_results.py, which gathers the launches of a hyperparameter experiment
into the results file read by the experiment plans and their figures: the settings given as lists (the [layers,
channels] of the mask heads and the backbone blocks of their skips in the instance segmentation experiment) group the
launches by their values like the scalar settings, one mean row per setting, and are written to the file as they are
in the configuration.
"""

import csv

from pathlib import Path
from typing import Any, Dict, List

from scripts.evaluation.collect_experiment_results import build_result_rows, make_hashable, write_results

# Swept settings of the fake experiment: a scalar one, the convolutions of the mask heads and the skips of their
# finer inputs, both lists, and the averaged weights flag every experiment reads
SWEPT_SETTINGS = {"learning_rate": ("ExperimentConfiguration", "learning_rate"),
                  "mask_convolutions": ("DetectionLevels", "4", "mask_head", "convolutions"),
                  "mask_skip_blocks_at_pool_size": ("MaskHead", "skip_blocks_at_pool_size"),
                  "exponential_moving_average": ("Optimization", "exponential_moving_average", "activated")}


def make_launch(launch_name: str, mask_convolutions: List[int], skip_blocks: List[int],
                evaluations: Dict[int, float]) -> Dict[str, Any]:
    """
    Builds a fake launch as collect_launches returns it, so that the grouping of build_result_rows can be checked
    without training folders.

    Args:
        launch_name (str): The launch, e.g. "run_7".
        mask_convolutions (List[int]): The [layers, channels] of its mask heads.
        skip_blocks (List[int]): The backbone blocks of its skips at the pool size, empty for none.
        evaluations (Dict[int, float]): Its test metric at every evaluated iteration.

    Returns:
        Dict[str, Any]: The launch, finished, with its trained weights.
    """
    settings = {"learning_rate": 0.001, "mask_convolutions": mask_convolutions,
                "mask_skip_blocks_at_pool_size": skip_blocks, "exponential_moving_average": True,
                "evaluated_weights": "trained"}
    return {"launch": launch_name, "evaluations": evaluations, "finished": True, "settings": settings,
            "run_folder": f"INS_{launch_name}"}


# Two launches of the reference mask heads, two of larger heads (the lists built anew, as every configuration file
# gives its own), and one of the larger heads with skips
LAUNCHES = [make_launch(launch_name="run_1", mask_convolutions=[3, 32], skip_blocks=[],
                        evaluations={1000: 0.70, 2000: 0.80}),
            make_launch(launch_name="run_2", mask_convolutions=[3, 32], skip_blocks=[],
                        evaluations={1000: 0.72, 2000: 0.82}),
            make_launch(launch_name="run_7", mask_convolutions=[4, 64], skip_blocks=[],
                        evaluations={1000: 0.74, 2000: 0.84}),
            make_launch(launch_name="run_8", mask_convolutions=[4, 64], skip_blocks=[],
                        evaluations={1000: 0.76, 2000: 0.86}),
            make_launch(launch_name="run_11", mask_convolutions=[4, 64], skip_blocks=[4],
                        evaluations={1000: 0.78, 2000: 0.88})]


def test_list_settings_compare_by_value() -> None:
    """
    The hashable form of a setting is equal for equal lists or mappings built separately, differs for other values,
    and leaves the scalar settings as they are, so that the grouping of the launches only changes for list settings.
    """
    assert make_hashable(value=[4, 64]) == make_hashable(value=[4, 64]) == (4, 64)
    assert hash(make_hashable(value=[[1, 2], [3]])) == hash(((1, 2), (3,)))
    assert make_hashable(value=[4, 64]) != make_hashable(value=[3, 32])
    assert make_hashable(value=[]) != make_hashable(value=[4])
    assert make_hashable(value={"convolutions": [4, 64], "roi_align_pool_size": 14}) == \
        make_hashable(value={"roi_align_pool_size": 14, "convolutions": [4, 64]})
    assert make_hashable(value=0.001) == 0.001 and make_hashable(value="batch") == "batch"
    assert make_hashable(value=None) is None and make_hashable(value=True) is True


def test_launches_with_list_settings_get_one_mean_row_per_setting() -> None:
    """
    The launches of the same list settings share one mean row whatever the list objects, those of other lists get
    their own, in the order of their first launch, with the mean of their launches and their settings unchanged.
    """
    columns, rows = build_result_rows(launches=LAUNCHES, metric_column="mask_map_50_95", number_last_evaluations=1,
                                      swept_settings=SWEPT_SETTINGS)
    assert [row["launch"] for row in rows] == ["run_1", "run_2", "mean", "run_7", "run_8", "mean", "run_11", "mean"]

    # Mean rows of the three settings, carrying the settings as lists
    mean_rows = [row for row in rows if row["launch"] == "mean"]
    assert [row["mask_convolutions"] for row in mean_rows] == [[3, 32], [4, 64], [4, 64]]
    assert [row["mask_skip_blocks_at_pool_size"] for row in mean_rows] == [[], [], [4]]
    assert [round(row["mask_map_50_95_2000"], 6) for row in mean_rows] == [0.81, 0.85, 0.88]
    assert [round(row["mask_map_50_95_last_1_mean"], 6) for row in mean_rows] == [0.81, 0.85, 0.88]
    assert columns[-5:] == [*SWEPT_SETTINGS, "run_folder"]


def test_results_file_writes_list_settings_as_in_the_configuration(tmp_path: Path) -> None:
    """
    The results file writes the list settings as the configuration gives them, so that a spreadsheet, the experiment
    plan and its figures read them back unchanged.

    Args:
        tmp_path (Path): The temporary folder of the test, holding the results file.
    """
    columns, rows = build_result_rows(launches=LAUNCHES, metric_column="mask_map_50_95", number_last_evaluations=1,
                                      swept_settings=SWEPT_SETTINGS)
    output_path = tmp_path / "results.csv"
    write_results(columns=columns, rows=rows, output_path=output_path, swept_settings=SWEPT_SETTINGS)
    with output_path.open(encoding="utf-8", newline="") as results_file:
        written_rows = list(csv.DictReader(f=results_file))
    assert [row["mask_convolutions"] for row in written_rows] == ["[3, 32]"] * 3 + ["[4, 64]"] * 5
    assert [row["mask_skip_blocks_at_pool_size"] for row in written_rows] == ["[]"] * 6 + ["[4]"] * 2
    assert [row["mask_map_50_95_2000"] for row in written_rows][:3] == ["0.800000", "0.820000", "0.810000"]
