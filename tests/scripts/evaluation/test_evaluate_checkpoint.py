"""
Tests of scripts/evaluation/evaluate_checkpoint.py, the standalone evaluation of the checkpoints of a trained run: the
target precisions of the class thresholds replace the experiment's one (a list for the classification), and a list
given to a task keeping a single precision (the frame selection) is refused. The final non-maximum suppression of a
detection run is replaced by the evaluated one, whose evaluations get an output subfolder of their own when it differs
from the one of the run, invalid settings being refused and the tasks without detector skipped. The predictions file
of every evaluated sample is turned on or off for the tasks writing one, the others being skipped, and the error image
limit turns the error images on for the tasks saving them, a configuration without 'evaluation' section being refused.
Another labelled split is evaluated into a folder of its own, a folder holding the evaluations of another split being
refused, and a run without checkpoint, or a given iteration without one, stops the script before anything is written.
The script prints its help when launched without arguments, reads every setting of evaluate_run from its command line,
DEFAULT_EVALUATION_SETTINGS or None for those not given, and refuses a missing run folder or a value the evaluation
does not accept before anything is loaded.
"""

import inspect
import json
import pytest

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from scripts.evaluation.evaluate_checkpoint import (DEFAULT_EVALUATION_SETTINGS, apply_error_image_limit,
                                                    apply_nms_criterion, apply_save_predictions,
                                                    apply_target_precisions, apply_test_data_folder,
                                                    check_evaluation_folder_split, evaluate_run,
                                                    list_evaluated_iterations, parse_evaluation_arguments,
                                                    point_to_checkpoint_evaluation)
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.data.data_record import DATA_RECORD_FILE_NAME
from vision_training.evaluation.checkpoint_evaluation import CHECKPOINT_EVALUATION_FOLDER_NAME
from vision_training.trainers.checkpointing import get_checkpoint_file, update_checkpoint_registry
from tests.trainers.test_base_trainer import build_dummy_trainer

# Detection configuration of the run whose final NMS is replaced, set to the greedy NMS ("iou", exponent 1.0) of the
# runs trained before DIoU-NMS was kept
DETECTION_CONFIGURATION_PATH = "vision_training/configurations/object_detection/configuration_example.yaml"

# Run folder given to the command line of the script, never read by the parser
RUN_NAME = "CLS_TEST_MC3"

# Every argument of the evaluation script, which its help must describe
EVALUATION_SCRIPT_ARGUMENTS = ["--experiment_folder", "--iteration", "--test_data_folder", "--batch_size",
                               "--track_evaluation", "--save_predictions", "--maximum_error_images_per_class_pair",
                               "--target_precisions", "--nms_criterion", "--nms_distance_exponent"]


def load_greedy_detection_configuration() -> Dict[str, Any]:
    """
    Loads the model configuration of the example detection run with the greedy final NMS ("iou", exponent 1.0), the
    one of the runs trained before stage 5b of the detection benchmark kept DIoU-NMS, so that the tests replace the
    NMS of such a run whatever the example configuration holds.

    Returns:
        Dict[str, Any]: The model configuration, its 'Detector' section with the greedy NMS.
    """
    _, model_configuration = load_experiment_configuration(configuration_path=DETECTION_CONFIGURATION_PATH)
    model_configuration["Detector"].update({"nms_criterion": "iou", "nms_distance_exponent": 1.0})
    return model_configuration


@pytest.mark.parametrize(argnames=("configuration_name", "target_precisions"),
                         argvalues=[("classification/configuration_synthetic_shapes.yaml", [0.99, 0.95, 0.90])])
def test_target_precisions_replace_the_experiment_one(configuration_name: str,
                                                      target_precisions: float | list[float]) -> None:
    """
    Verifies that the target precisions replace the 'evaluation.target_precision' of the experiment, a list for the
    classification, and that None leaves the experiment's one untouched.

    Args:
        configuration_name (str): The configuration of the evaluated run, below vision_training/configurations.
        target_precisions (float | list[float]): The target precisions replacing the experiment's one.
    """
    experiment_configuration, _ = load_experiment_configuration(
        configuration_path=f"vision_training/configurations/{configuration_name}")
    apply_target_precisions(experiment_configuration=experiment_configuration, target_precisions=None)
    assert experiment_configuration["evaluation"]["target_precision"] == 0.99

    apply_target_precisions(experiment_configuration=experiment_configuration, target_precisions=target_precisions)
    assert experiment_configuration["evaluation"]["target_precision"] == target_precisions


@pytest.mark.parametrize(argnames=("configuration_name", "target_precisions", "expected_message"), argvalues=[
    ("frame_selection/configuration_synthetic_sequences.yaml", [0.99, 0.95], "must be a single probability")])
def test_invalid_target_precisions_are_refused(configuration_name: str, target_precisions: list[float],
                                               expected_message: str) -> None:
    """
    Verifies that a list given to a task keeping one target precision stops the script before any evaluation, the
    configuration of the experiment being left as it was.

    Args:
        configuration_name (str): The configuration of the evaluated run, below vision_training/configurations.
        target_precisions (list[float]): The refused target precisions.
        expected_message (str): A part of the message of the refusal.
    """
    experiment_configuration, _ = load_experiment_configuration(
        configuration_path=f"vision_training/configurations/{configuration_name}")
    with pytest.raises(expected_exception=ValueError, match=expected_message):
        apply_target_precisions(experiment_configuration=experiment_configuration, target_precisions=target_precisions)
    assert experiment_configuration["evaluation"]["target_precision"] == 0.99


@pytest.mark.parametrize(argnames=("nms_criterion", "nms_distance_exponent", "expected_suffix", "expected_settings"),
                         argvalues=[(None, None, "", ("iou", 1.0)),
                                    ("distance_iou", None, "_nms_distance_iou_1.0", ("distance_iou", 1.0)),
                                    ("distance_iou", 0.6, "_nms_distance_iou_0.6", ("distance_iou", 0.6)),
                                    ("iou", None, "", ("iou", 1.0)),
                                    (None, 0.6, "", ("iou", 0.6))])
def test_nms_criterion_replaces_the_run_one(nms_criterion: Optional[str], nms_distance_exponent: Optional[float],
                                            expected_suffix: str, expected_settings: Tuple[str, float]) -> None:
    """
    Verifies that the criterion and the distance exponent of the evaluation replace those of the 'Detector' section of
    a run evaluated with the greedy NMS, None keeping the run's, and that the output subfolder gets a suffix naming the
    evaluated NMS only when it differs from the one of the run, an exponent changing nothing with the greedy NMS.

    Args:
        nms_criterion (Optional[str]): The evaluated criterion, None to keep the run's.
        nms_distance_exponent (Optional[float]): The evaluated exponent, None to keep the run's.
        expected_suffix (str): The expected suffix of the output subfolder.
        expected_settings (Tuple[str, float]): The criterion and exponent of the evaluation.
    """
    model_configuration = load_greedy_detection_configuration()
    assert apply_nms_criterion(model_configuration=model_configuration, task="object_detection",
                               nms_criterion=nms_criterion, nms_distance_exponent=nms_distance_exponent) == \
        expected_suffix
    assert (model_configuration["Detector"]["nms_criterion"],
            model_configuration["Detector"]["nms_distance_exponent"]) == expected_settings


def test_invalid_or_skipped_nms_criterion(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that an unknown criterion or a negative exponent stops the script before any evaluation, the 'Detector'
    section of the run being left as it was, that a run evaluated with DIoU-NMS gets a subfolder of its own for the
    greedy NMS, and that a task without detector is evaluated without the replacement, a warning telling it.

    Args:
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed warning.
    """
    model_configuration = load_greedy_detection_configuration()
    for nms_criterion, nms_distance_exponent, expected_message in [
            ("soft_nms", None, "'Detector.nms_criterion' must be one of"),
            (None, -1.0, "'Detector.nms_distance_exponent' must be a finite positive number")]:
        with pytest.raises(expected_exception=ValueError, match=expected_message):
            apply_nms_criterion(model_configuration=model_configuration, task="object_detection",
                                nms_criterion=nms_criterion, nms_distance_exponent=nms_distance_exponent)
        assert (model_configuration["Detector"]["nms_criterion"],
                model_configuration["Detector"]["nms_distance_exponent"]) == ("iou", 1.0)

    # A run evaluated with DIoU-NMS, then with the greedy NMS
    model_configuration["Detector"].update({"nms_criterion": "distance_iou", "nms_distance_exponent": 0.6})
    assert apply_nms_criterion(model_configuration=model_configuration, task="object_detection", nms_criterion="iou",
                               nms_distance_exponent=None) == "_nms_iou"

    # The classification has no final non-maximum suppression
    _, classification_configuration = load_experiment_configuration(
        configuration_path="vision_training/configurations/classification/configuration_synthetic_shapes.yaml")
    assert apply_nms_criterion(model_configuration=classification_configuration, task="classification",
                               nms_criterion="distance_iou", nms_distance_exponent=1.0) == ""
    assert "Detector" not in classification_configuration
    assert "nms_criterion and nms_distance_exponent are skipped" in capsys.readouterr().out


def test_error_image_limit_turns_the_error_images_on_or_is_skipped(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the error image limit turns on the 'evaluation.error_dumps' of a classification run with its limit,
    None leaving them untouched, that a task without error images (the self-supervised task) is evaluated without
    them, a warning telling it, and that a configuration without 'evaluation' section stops the script instead of
    being taken for a task without error images.

    Args:
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed warning.
    """
    experiment_configuration, _ = load_experiment_configuration(
        configuration_path="vision_training/configurations/classification/configuration_synthetic_shapes.yaml")
    error_dumps_before = dict(experiment_configuration["evaluation"]["error_dumps"])
    apply_error_image_limit(experiment_configuration=experiment_configuration,
                            maximum_error_images_per_class_pair=None)
    assert experiment_configuration["evaluation"]["error_dumps"] == error_dumps_before
    apply_error_image_limit(experiment_configuration=experiment_configuration, maximum_error_images_per_class_pair=7)
    assert experiment_configuration["evaluation"]["error_dumps"] == {**error_dumps_before, "activated": True,
                                                                     "maximum_per_class_pair": 7}

    # The self-supervised task saves no error image
    self_supervised_configuration, _ = load_experiment_configuration(
        configuration_path="vision_training/configurations/self_supervised/configuration_synthetic_shapes.yaml")
    apply_error_image_limit(experiment_configuration=self_supervised_configuration,
                            maximum_error_images_per_class_pair=7)
    assert "error_dumps" not in self_supervised_configuration["evaluation"]
    assert "maximum_error_images_per_class_pair is skipped" in capsys.readouterr().out

    # A configuration without its 'evaluation' section is refused, for every replaced evaluation setting
    del experiment_configuration["evaluation"]
    with pytest.raises(expected_exception=KeyError, match="evaluation"):
        apply_error_image_limit(experiment_configuration=experiment_configuration,
                                maximum_error_images_per_class_pair=7)
    with pytest.raises(expected_exception=KeyError, match="evaluation"):
        apply_save_predictions(experiment_configuration=experiment_configuration, save_predictions=True)
    with pytest.raises(expected_exception=KeyError, match="evaluation"):
        apply_target_precisions(experiment_configuration=experiment_configuration, target_precisions=0.9)
    assert capsys.readouterr().out == ""


def test_save_predictions_replaces_the_experiment_one_or_is_skipped(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that save_predictions replaces the 'evaluation.save_predictions' of a classification run, None leaving it
    untouched, and that a task without the setting (the detection) is evaluated without it, a warning telling it.

    Args:
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed warning.
    """
    experiment_configuration, _ = load_experiment_configuration(
        configuration_path="vision_training/configurations/classification/configuration_synthetic_shapes.yaml")
    apply_save_predictions(experiment_configuration=experiment_configuration, save_predictions=None)
    assert experiment_configuration["evaluation"]["save_predictions"] is True
    apply_save_predictions(experiment_configuration=experiment_configuration, save_predictions=False)
    assert experiment_configuration["evaluation"]["save_predictions"] is False

    # The detection writes no predictions file
    detection_configuration, _ = load_experiment_configuration(configuration_path=DETECTION_CONFIGURATION_PATH)
    apply_save_predictions(experiment_configuration=detection_configuration, save_predictions=True)
    assert "save_predictions" not in detection_configuration["evaluation"]
    assert "save_predictions is skipped" in capsys.readouterr().out


def test_another_split_is_evaluated_into_a_folder_of_its_own(tmp_path: Path) -> None:
    """
    Verifies that another labelled split replaces the test split of the run and names the output subfolder of its
    evaluations after its folder and the folder holding it, so that the evaluations of two splits never share a folder,
    while None or the test split of the run given again (written differently) keep the Checkpoint_Evaluation folder.

    Args:
        tmp_path (Path): The pytest temporary directory, holding the run folder and the dataset folders.
    """
    run_test_data_folder = tmp_path / "synthetic_shapes" / "test"
    run_folder = tmp_path / "experiments" / "CLS_RUN_MC3"
    expected_folder_names = {None: CHECKPOINT_EVALUATION_FOLDER_NAME,
                             f"{run_test_data_folder}/": CHECKPOINT_EVALUATION_FOLDER_NAME,
                             tmp_path / "synthetic_shapes" / "validation":
                                 f"{CHECKPOINT_EVALUATION_FOLDER_NAME}_split_synthetic_shapes_validation",
                             tmp_path / "newer_shapes" / "test":
                                 f"{CHECKPOINT_EVALUATION_FOLDER_NAME}_split_newer_shapes_test"}
    for test_data_folder, expected_folder_name in expected_folder_names.items():
        experiment_configuration = {"test_data_folder": run_test_data_folder}
        folder_suffix = apply_test_data_folder(experiment_configuration=experiment_configuration,
                                               test_data_folder=test_data_folder)
        evaluation_folder = point_to_checkpoint_evaluation(experiment_configuration=experiment_configuration,
                                                           experiment_folder=run_folder, folder_suffix=folder_suffix)
        assert evaluation_folder == run_folder / expected_folder_name
        expected_test_data_folder = run_test_data_folder if test_data_folder is None else Path(test_data_folder)
        assert experiment_configuration["test_data_folder"] == expected_test_data_folder


def write_evaluation_data_record(evaluation_folder: Path, test_data_folder: Union[str, Path]) -> None:
    """
    Writes the data.json of an evaluation folder, as the trainer of a standalone evaluation records the data it
    evaluated, reduced to the data folder of its test split read by check_evaluation_folder_split.

    Args:
        evaluation_folder (Path): The evaluation folder, created.
        test_data_folder (Union[str, Path]): The evaluated split recorded as the test split.
    """
    evaluation_folder.mkdir(parents=True)
    data_record = {"splits": {"test": {"data_folder": str(test_data_folder)}}}
    (evaluation_folder / DATA_RECORD_FILE_NAME).write_text(data=json.dumps(obj=data_record), encoding="utf-8")


def test_evaluation_folder_of_another_split_is_refused(tmp_path: Path) -> None:
    """
    Verifies that the evaluations of a split are refused in a folder whose data.json records another test split, e.g.
    a Checkpoint_Evaluation folder written by the evaluation of another split before every split had a folder of its
    own, while the same split, a new folder and a folder without data record are accepted.

    Args:
        tmp_path (Path): The pytest temporary directory, holding the evaluation folders.
    """
    run_test_data_folder = tmp_path / "synthetic_shapes" / "test"
    other_data_folder = tmp_path / "synthetic_shapes" / "validation"

    # A new folder, then a folder written before the record of the data existed
    check_evaluation_folder_split(evaluation_folder=tmp_path / "new", test_data_folder=run_test_data_folder)
    (tmp_path / "without_record").mkdir()
    check_evaluation_folder_split(evaluation_folder=tmp_path / "without_record", test_data_folder=run_test_data_folder)

    # A folder holding the evaluations of the validation split, accepted for it only
    evaluation_folder = tmp_path / CHECKPOINT_EVALUATION_FOLDER_NAME
    write_evaluation_data_record(evaluation_folder=evaluation_folder, test_data_folder=other_data_folder)
    check_evaluation_folder_split(evaluation_folder=evaluation_folder, test_data_folder=f"{other_data_folder}/")
    with pytest.raises(expected_exception=ValueError, match="holds the evaluations of the split"):
        check_evaluation_folder_split(evaluation_folder=evaluation_folder, test_data_folder=run_test_data_folder)


def test_standalone_evaluation_records_the_split_its_folder_is_checked_against(tmp_path: Path) -> None:
    """
    Verifies on the dummy task that the data.json a standalone evaluation writes in its Checkpoint_Evaluation folder
    records the split check_evaluation_folder_split compares, so that a later evaluation of the same split is accepted
    in the folder and one of another split is refused before its trainer replaces that record.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=5, view_input_images=False,
                                  number_sample_images=0)
    trainer.run_training_loop()
    evaluation_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=5,
                                             view_input_images=False, number_sample_images=0,
                                             evaluated_run_folder=trainer.experiment_folder)

    # The dummy splits are named after their folders, 'test' being the one of the run
    evaluation_folder = evaluation_trainer.experiment_folder
    check_evaluation_folder_split(evaluation_folder=evaluation_folder, test_data_folder=Path("test"))
    with pytest.raises(expected_exception=ValueError, match="holds the evaluations of the split"):
        check_evaluation_folder_split(evaluation_folder=evaluation_folder, test_data_folder=Path("validation"))


def save_empty_checkpoints(weights_directory: Path, iterations: list[int]) -> None:
    """
    Saves empty weight files at the given iterations, recorded in the registry in that order as the trainer records
    its checkpoints, the listing of the evaluated iterations reading only the registry and the presence of the files.

    Args:
        weights_directory (Path): The 'Weights' folder of the run, created.
        iterations (list[int]): The iterations of the checkpoints, in their saving order.
    """
    for iteration in iterations:
        checkpoint_path = get_checkpoint_file(weights_directory=weights_directory, iteration=iteration)
        checkpoint_path.parent.mkdir(parents=True)
        checkpoint_path.write_bytes(data=b"")
        update_checkpoint_registry(weights_directory=weights_directory, iteration=iteration)


def test_evaluated_iterations_need_a_checkpoint_on_the_disk(tmp_path: Path) -> None:
    """
    Verifies that a run stopped before its first save (no registry nor checkpoint) stops the script for the last, every
    or a given checkpoint, that a run with checkpoints gives every checkpoint of the registry from the last saved one,
    the last one or the given one, a given iteration without checkpoint being refused, and that a checkpoint of the
    registry deleted from the disk is left to the evaluation, which skips it, as long as another one is still saved.

    Args:
        tmp_path (Path): The pytest temporary directory, holding the 'Weights' folder.
    """
    weights_directory = tmp_path / "Weights"
    weights_directory.mkdir()
    for iteration in [None, "all", 4]:
        with pytest.raises(expected_exception=FileNotFoundError, match="No checkpoint to evaluate"):
            list_evaluated_iterations(weights_directory=weights_directory, iteration=iteration)

    # Checkpoints of iterations 2, 4 and 5, the last saved first for "all"
    save_empty_checkpoints(weights_directory=weights_directory, iterations=[2, 4, 5])
    expected_iterations = {"all": [5, 4, 2], None: [5], 4: [4]}
    for iteration, evaluated_iterations in expected_iterations.items():
        assert list_evaluated_iterations(weights_directory=weights_directory, iteration=iteration) == \
            evaluated_iterations
    with pytest.raises(expected_exception=FileNotFoundError, match=r"checkpoints of the registry: \[2, 4, 5\]"):
        list_evaluated_iterations(weights_directory=weights_directory, iteration=3)

    # Deleted checkpoints: skipped later while one is left, refused once none is
    get_checkpoint_file(weights_directory=weights_directory, iteration=2).unlink()
    assert list_evaluated_iterations(weights_directory=weights_directory, iteration="all") == [5, 4, 2]
    for iteration in [4, 5]:
        get_checkpoint_file(weights_directory=weights_directory, iteration=iteration).unlink()
    with pytest.raises(expected_exception=FileNotFoundError, match="No checkpoint to evaluate"):
        list_evaluated_iterations(weights_directory=weights_directory, iteration="all")


def test_evaluation_script_without_arguments_prints_its_help(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the evaluation script launched without any argument prints the help of every one of its arguments
    and stops without evaluating, instead of failing on the missing run folder.

    Args:
        capsys (pytest.CaptureFixture): The captured console output.
    """
    with pytest.raises(expected_exception=SystemExit) as exit_information:
        parse_evaluation_arguments(arguments=[])
    assert exit_information.value.code == 0
    help_text = capsys.readouterr().out
    assert [argument for argument in EVALUATION_SCRIPT_ARGUMENTS if argument not in help_text] == []


def test_evaluation_arguments_default_to_the_evaluation_settings() -> None:
    """
    Verifies that the evaluation script given only the run folder evaluates its last checkpoint on its test split with
    DEFAULT_EVALUATION_SETTINGS, every other setting None to keep the one of the run, and that the parsed arguments are
    exactly the parameters of evaluate_run, so that the main block passes them all.
    """
    parsed_arguments = parse_evaluation_arguments(arguments=["--experiment_folder", RUN_NAME])
    assert vars(parsed_arguments) == {"experiment_folder": Path(RUN_NAME), "iteration": None, "test_data_folder": None,
                                      "track_evaluation": None, "save_predictions": None, "target_precisions": None,
                                      "nms_criterion": None, "nms_distance_exponent": None,
                                      **DEFAULT_EVALUATION_SETTINGS}
    assert set(vars(parsed_arguments)) == set(inspect.signature(evaluate_run).parameters)


@pytest.mark.parametrize(argnames=("arguments", "expected_settings"), argvalues=[
    (["--iteration", "all", "--batch_size", "None", "--maximum_error_images_per_class_pair", "None",
      "--track_evaluation", "False", "--save_predictions", "True", "--target_precisions", "0.95"],
     {"iteration": "all", "batch_size": None, "maximum_error_images_per_class_pair": None, "track_evaluation": False,
      "save_predictions": True, "target_precisions": 0.95}),
    (["--iteration", "2000", "--batch_size", "64", "--maximum_error_images_per_class_pair", "10",
      "--track_evaluation", "True", "--save_predictions", "False", "--target_precisions", "0.99", "0.95", "0.90",
      "--nms_criterion", "distance_iou", "--nms_distance_exponent", "0.6"],
     {"iteration": 2000, "batch_size": 64, "maximum_error_images_per_class_pair": 10, "track_evaluation": True,
      "save_predictions": False, "target_precisions": [0.99, 0.95, 0.90], "nms_criterion": "distance_iou",
      "nms_distance_exponent": 0.6})])
def test_evaluation_arguments_reach_every_setting(arguments: List[str], expected_settings: Dict[str, Any],
                                                  tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that every setting of evaluate_run can be given on the command line of the evaluation script: 'all' or a
    number as iteration, 'None' keeping the batch size or the error images of the run, the words True and False, one
    target precision given as a number and several as a list, and paths starting with a '~' the shell leaves after '='
    expanded to the home folder.

    Args:
        arguments (List[str]): The command line arguments after the run folder and the split.
        expected_settings (Dict[str, Any]): The parsed settings they give, the others keeping their defaults.
        tmp_path (Path): The pytest temporary directory, standing for the home folder.
        monkeypatch (pytest.MonkeyPatch): Sets the home folder.
    """
    monkeypatch.setenv(name="HOME", value=str(tmp_path))
    parsed_arguments = parse_evaluation_arguments(arguments=[f"--experiment_folder=~/{RUN_NAME}",
                                                             "--test_data_folder=~/synthetic_shapes/validation",
                                                             *arguments])
    assert vars(parsed_arguments) == {"experiment_folder": tmp_path / RUN_NAME,
                                      "test_data_folder": tmp_path / "synthetic_shapes" / "validation",
                                      "nms_criterion": None, "nms_distance_exponent": None, **expected_settings}


@pytest.mark.parametrize(argnames="arguments", argvalues=[
    ["--iteration", "all"],
    ["--experiment_folder", RUN_NAME, "--nms_criterion", "soft_nms"],
    ["--experiment_folder", RUN_NAME, "--iteration", "last"],
    ["--experiment_folder", RUN_NAME, "--batch_size", "0"],
    ["--experiment_folder", RUN_NAME, "--track_evaluation", "yes"],
    ["--experiment_folder", RUN_NAME, "--target_precisions", "high"]])
def test_evaluation_script_refuses_a_missing_run_or_unknown_values(arguments: List[str],
                                                                  capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the evaluation script stops with its usage and the error, before anything is loaded, when the run
    folder is missing or an iteration, an NMS criterion, a batch size, a boolean or a precision is not one the
    evaluation accepts.

    Args:
        arguments (List[str]): The command line arguments.
        capsys (pytest.CaptureFixture): The captured console output.
    """
    with pytest.raises(expected_exception=SystemExit) as exit_information:
        parse_evaluation_arguments(arguments=arguments)
    assert exit_information.value.code == 2
    assert "error:" in capsys.readouterr().err
