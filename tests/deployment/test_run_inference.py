"""
Tests of vision_training/deployment/run_inference.py, the throughput of an exported run in the process and through
Triton: the script prints its help when launched without arguments, reads every setting of
measure_inference_throughput from its command line, DEFAULT_IMAGE_FOLDER and DEFAULT_INFERENCE_SETTINGS for those not
given (the values the serving chapter of the deployment tutorials quotes), and refuses a missing run folder or a value
that is not a number before anything is loaded. No Triton server is started.
"""

import inspect
import pytest

from pathlib import Path
from typing import List

from vision_training.deployment.run_inference import (DEFAULT_IMAGE_FOLDER, DEFAULT_INFERENCE_SETTINGS,
                                                      measure_inference_throughput, parse_inference_arguments)

# Run folder given on the command line, never read by the parser
RUN_NAME = "CLS_TEST_MC3"

# Every argument of the script, which its help must describe
INFERENCE_SCRIPT_ARGUMENTS = ["--experiment_folder", "--image_folder", "--url", "--use_shared_memory",
                              "--number_iterations", "--number_warmup_iterations", "--iteration"]


def test_inference_script_without_arguments_prints_its_help(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the throughput script launched without any argument prints the help of every one of its arguments,
    with the default address and request counts, and stops without measuring, instead of failing on the missing run
    folder.

    Args:
        capsys (pytest.CaptureFixture): The captured console output.
    """
    with pytest.raises(expected_exception=SystemExit) as exit_information:
        parse_inference_arguments(arguments=[])
    assert exit_information.value.code == 0
    # The help joined on one line, argparse wrapping it to the width of the terminal
    help_text = " ".join(capsys.readouterr().out.split())
    assert [argument for argument in INFERENCE_SCRIPT_ARGUMENTS if argument not in help_text] == []
    assert all(f"(default: {value})" in help_text for value in DEFAULT_INFERENCE_SETTINGS.values())


def test_inference_arguments_default_to_the_inference_settings() -> None:
    """
    Verifies that the throughput script given only the run folder sends the images of DEFAULT_IMAGE_FOLDER inside the
    gRPC requests with DEFAULT_INFERENCE_SETTINGS to the last saved checkpoint, the images being those of the
    repository, and that the parsed arguments are exactly the parameters of measure_inference_throughput, so that the
    main block passes them all.
    """
    parsed_arguments = parse_inference_arguments(arguments=["--experiment_folder", RUN_NAME])
    assert vars(parsed_arguments) == {"experiment_folder": Path(RUN_NAME), "image_folder": DEFAULT_IMAGE_FOLDER,
                                      "use_shared_memory": False, "iteration": None, **DEFAULT_INFERENCE_SETTINGS}
    assert DEFAULT_IMAGE_FOLDER.parts[-3:] == ("vision_training", "deployment", "inference_data")
    assert DEFAULT_IMAGE_FOLDER.is_dir()
    assert set(vars(parsed_arguments)) == set(inspect.signature(measure_inference_throughput).parameters)


def test_inference_arguments_reach_every_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that every setting of measure_inference_throughput can be given on the command line of the script, and
    that paths starting with a '~' the shell leaves after '=' are expanded to the home folder.

    Args:
        tmp_path (Path): The pytest temporary directory, standing for the home folder.
        monkeypatch (pytest.MonkeyPatch): Sets the home folder.
    """
    monkeypatch.setenv(name="HOME", value=str(tmp_path))
    parsed_arguments = parse_inference_arguments(arguments=[
        f"--experiment_folder=~/{RUN_NAME}", "--image_folder=~/photos", "--url", "triton-host:9001",
        "--use_shared_memory", "--number_iterations", "50", "--number_warmup_iterations", "5", "--iteration", "2000"])
    assert vars(parsed_arguments) == {"experiment_folder": tmp_path / RUN_NAME, "image_folder": tmp_path / "photos",
                                      "url": "triton-host:9001", "use_shared_memory": True, "number_iterations": 50,
                                      "number_warmup_iterations": 5, "iteration": 2000}


@pytest.mark.parametrize(argnames="arguments", argvalues=[
    ["--iteration", "2000"],
    ["--experiment_folder", RUN_NAME, "--number_iterations", "many"],
    ["--experiment_folder", RUN_NAME, "--number_warmup_iterations", "2.5"],
    ["--experiment_folder", RUN_NAME, "--iteration", "last"],
    ["--experiment_folder", RUN_NAME, "--batch_size", "16"]])
def test_inference_script_refuses_a_missing_run_or_unknown_values(arguments: List[str],
                                                                 capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the throughput script stops with its usage and the error, before anything is loaded, when the run
    folder is missing, a count or the iteration is not a whole number, or an argument is unknown.

    Args:
        arguments (List[str]): The command line arguments.
        capsys (pytest.CaptureFixture): The captured console output.
    """
    with pytest.raises(expected_exception=SystemExit) as exit_information:
        parse_inference_arguments(arguments=arguments)
    assert exit_information.value.code == 2
    assert "error:" in capsys.readouterr().err
