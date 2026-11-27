"""
Tests of scripts/deployment/check_half_precision_parity.py, the trial of an FP16 TorchScript classifier against the
eager FP32 one: the image comparison counts the predicted classes and the deployed decisions that moved and the
probability differences, the table says when no threshold is deployed, the deployed thresholds are read in the class
order of the run (a file of other classes refused). The FP16 conversion the trial shares with the export is tested by
tests/deployment/test_export_specifications.py. The script prints its help when launched without arguments, reads every
setting of check_half_precision_parity from its command line, under the names and defaults of the arguments of
export_model.py, and refuses a missing run folder or a value that is not a number before anything is loaded.
"""

import inspect
import json
import numpy as np
import pytest

from pathlib import Path
from typing import List

from scripts.deployment.check_half_precision_parity import (build_parity_table, check_half_precision_parity,
                                                            compare_half_precision_outputs,
                                                            parse_half_precision_parity_arguments,
                                                            read_deployed_thresholds)
from vision_training.deployment.export_model import DEFAULT_EXPORT_SETTINGS, parse_export_arguments

# Run folder given on the command line, never read by the parsers
RUN_NAME = "CLS_TEST_MC3"

# Every argument of the trial script, which its help must describe
PARITY_SCRIPT_ARGUMENTS = ["--experiment_folder", "--iteration", "--use_averaged_weights", "--use_saved_temperature",
                           "--saved_thresholds_path", "--maximum_batch_size"]


def test_comparison_counts_the_moved_predictions_and_decisions() -> None:
    """
    Verifies that the comparison counts an image whose predicted class changes, an image whose deployed decision
    changes for one class without its predicted class changing, and gives the largest and mean probability differences.
    """
    eager_probabilities = np.array([[0.95, 0.05], [0.40, 0.60], [0.10, 0.90], [0.80, 0.20]], dtype=np.float32)
    half_probabilities = np.array([[0.95, 0.05], [0.55, 0.45], [0.10, 0.90], [0.70, 0.30]], dtype=np.float32)
    comparison = compare_half_precision_outputs(eager_probabilities=eager_probabilities,
                                                half_probabilities=half_probabilities,
                                                thresholds=np.array([0.75, 0.5]))
    assert comparison["compared_images"] == 4 and comparison["predicted_class_changes"] == 1
    assert comparison["decision_changes"] == 2 and comparison["decision_changes_per_class"] == [1, 1]
    assert comparison["largest_difference"] == pytest.approx(0.15, abs=1e-6)
    assert comparison["mean_difference"] == pytest.approx((0.15 * 2 + 0.1 * 2) / 8, abs=1e-6)
    with pytest.raises(expected_exception=ValueError, match="of the same shape"):
        compare_half_precision_outputs(eager_probabilities=eager_probabilities,
                                       half_probabilities=half_probabilities[:2], thresholds=np.array([0.75, 0.5]))


def test_table_without_deployed_thresholds() -> None:
    """
    Verifies that without deployed thresholds (use_saved_temperature off) the comparison counts no decision and the
    table says that no threshold is deployed, the metrics of both rows and their change in points still given.
    """
    comparison = compare_half_precision_outputs(eager_probabilities=np.array([[0.7, 0.3]], dtype=np.float32),
                                                half_probabilities=np.array([[0.69, 0.31]], dtype=np.float32),
                                                thresholds=None)
    assert comparison["decision_changes"] is None and comparison["decision_changes_per_class"] is None
    rows = [{"scalars": {"Balanced Accuracy": balanced_accuracy, "Accuracy": 0.9, "Macro F1": 0.8},
             "evaluated_samples": "1 Test Images"} for balanced_accuracy in [0.95, 0.9448]]
    table = build_parity_table(summary={
        "run": "/runs/CLS_TEST_MC2", "checkpoint": "/runs/CLS_TEST_MC2/Weights/Iteration_10/model_000010.pt",
        "iteration": 10, "weights": "trained weights", "test_data_folder": "/data/test", "logit_temperature": 1.0,
        "thresholds_path": None, "random_batch_check": [], "rows": rows, "comparison": comparison,
        "agreement": {"within_tolerance": 1, "compared_samples": 1}, "file_sizes": {"fp32": 2e6, "fp16": 1e6}})
    assert "thresholds not deployed" in table and "| Deployed decision changed | | no threshold deployed | |" in table
    assert "| Balanced Accuracy | 95.00 % | 94.48 % | -0.52 points |" in table
    assert "| TorchScript file | 2.0 MB | 1.0 MB | |" in table


def test_deployed_thresholds_follow_the_class_order(tmp_path: Path) -> None:
    """
    Verifies that the deployed thresholds are read in the class order of the run, whatever the order of the file, and
    that a file of other classes is refused.

    Args:
        tmp_path (Path): The temporary folder of the test.
    """
    thresholds_path = tmp_path / "saved_temperature_thresholds.json"
    thresholds_path.write_text(data=json.dumps(obj={"logit_temperature": 3.0, "target_precision": 0.99, "classes": {
        "no_event": {"threshold": 0.93, "recall": 0.67}, "event": {"threshold": 0.9, "recall": 0.87}}}))
    assert read_deployed_thresholds(thresholds_path=thresholds_path,
                                    class_names=["event", "no_event"]).tolist() == [0.9, 0.93]
    with pytest.raises(expected_exception=ValueError, match="Expected the thresholds of the classes"):
        read_deployed_thresholds(thresholds_path=thresholds_path, class_names=["event", "other"])


def test_parity_script_without_arguments_prints_its_help(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the trial script launched without any argument prints the help of every one of its arguments and
    stops without tracing, instead of failing on the missing run folder.

    Args:
        capsys (pytest.CaptureFixture): The captured console output.
    """
    with pytest.raises(expected_exception=SystemExit) as exit_information:
        parse_half_precision_parity_arguments(arguments=[])
    assert exit_information.value.code == 0
    help_text = capsys.readouterr().out
    assert [argument for argument in PARITY_SCRIPT_ARGUMENTS if argument not in help_text] == []


def test_parity_arguments_default_as_the_export_arguments() -> None:
    """
    Verifies that the trial script given only the run folder traces the trained weights of the last checkpoint at a
    temperature of 1 with the maximum batch size of DEFAULT_EXPORT_SETTINGS, that every argument has the name and the
    default of an argument of the export script, so that the trial judges the file the export would write, and that
    the parsed arguments are exactly the parameters of check_half_precision_parity, so that the main block passes them
    all.
    """
    parsed_arguments = vars(parse_half_precision_parity_arguments(arguments=["--experiment_folder", RUN_NAME]))
    assert parsed_arguments == {"experiment_folder": Path(RUN_NAME), "iteration": None, "use_averaged_weights": False,
                                "use_saved_temperature": False, "saved_thresholds_path": None,
                                "maximum_batch_size": DEFAULT_EXPORT_SETTINGS["maximum_batch_size"]}
    export_arguments = vars(parse_export_arguments(arguments=["--experiment_folder", RUN_NAME]))
    assert {name: export_arguments[name] for name in parsed_arguments} == parsed_arguments
    assert set(parsed_arguments) == set(inspect.signature(check_half_precision_parity).parameters)


def test_parity_arguments_reach_every_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that every setting of check_half_precision_parity can be given on the command line of the trial script,
    and that paths starting with a '~' the shell leaves after '=' are expanded to the home folder.

    Args:
        tmp_path (Path): The pytest temporary directory, standing for the home folder.
        monkeypatch (pytest.MonkeyPatch): Sets the home folder.
    """
    monkeypatch.setenv(name="HOME", value=str(tmp_path))
    parsed_arguments = parse_half_precision_parity_arguments(arguments=[
        f"--experiment_folder=~/{RUN_NAME}", "--iteration", "2000", "--use_averaged_weights", "--use_saved_temperature",
        "--saved_thresholds_path=~/saved_temperature_thresholds.json", "--maximum_batch_size", "8"])
    assert vars(parsed_arguments) == {"experiment_folder": tmp_path / RUN_NAME, "iteration": 2000,
                                      "use_averaged_weights": True, "use_saved_temperature": True,
                                      "saved_thresholds_path": tmp_path / "saved_temperature_thresholds.json",
                                      "maximum_batch_size": 8}


@pytest.mark.parametrize(argnames="arguments", argvalues=[
    ["--iteration", "2000"],
    ["--experiment_folder", RUN_NAME, "--iteration", "last"],
    ["--experiment_folder", RUN_NAME, "--maximum_batch_size", "large"],
    ["--experiment_folder", RUN_NAME, "--tensorrt_precision", "FP16"]])
def test_parity_script_refuses_a_missing_run_or_unknown_values(arguments: List[str],
                                                              capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the trial script stops with its usage and the error, before anything is loaded, when the run folder
    is missing, a number is not a whole number, or an argument of the export the trial does not take is given.

    Args:
        arguments (List[str]): The command line arguments.
        capsys (pytest.CaptureFixture): The captured console output.
    """
    with pytest.raises(expected_exception=SystemExit) as exit_information:
        parse_half_precision_parity_arguments(arguments=arguments)
    assert exit_information.value.code == 2
    assert "error:" in capsys.readouterr().err
