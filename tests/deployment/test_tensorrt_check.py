"""
Tests of the TensorRT checks of an export (vision_training/deployment/tensorrt_check.py and their use by
export_experiment of vision_training/deployment/export_model.py): the FP16 overflow scan flags a node output above
65,504 with its node and op type, and lets pass a graph whose values stay below it, its infinities and integer values
being left out; the engine check served by Triton (its server and client stood in for) fails on non-finite outputs and
on differences above the tolerances, passes within them, records a model Triton cannot build or run with the errors of
its log, and always stops its container; the TF32 tolerances follow the differences TF32 makes in the eager model, the
tolerances of the specification staying the floor, and an FP32 engine missing those tolerances passes within the TF32
ones; an export whose TensorRT check fails leaves the repository unchanged, a passing one installs the models, an FP32
engine being given the eager outputs with TF32 on, an FP16 engine being compared with the eager FP16 copy at the
loosened tolerances, and --skip_tensorrt_check leaving the engine unchecked; an export needing the engine check on a
machine without Docker stops before anything is written, naming --skip_tensorrt_check and --tensorrt_precision None.
"""

import re
import numpy as np
import onnx
import pytest
import torch

from onnx import TensorProto, helper
from pathlib import Path
from typing import Any, Dict, List, Optional

from tests.deployment import test_input_contract
from tests.deployment.test_export_model import read_repository_files
from vision_training.deployment import export_model, tensorrt_check, triton_server
from vision_training.deployment.export_model import (HALF_PRECISION_TOLERANCE_FACTOR, STAGING_FOLDER_PREFIX,
                                                     export_experiment)
from vision_training.deployment.export_specifications import compare_probabilities, compare_segmentations
from vision_training.deployment.tensorrt_check import (HALF_PRECISION_LARGEST_VALUE, TF32_DIFFERENCE_FACTOR,
                                                       TF32_DISAGREEMENT_FACTOR, check_tensorrt_engine,
                                                       derive_tf32_tolerances, describe_half_precision_overflows,
                                                       describe_tf32_tolerances, find_half_precision_overflows)
from vision_training.deployment.triton_server import TritonServerError

# Factor of the node of the overflow test graph, which takes its input above 65,504 when the input exceeds 65.504
SCALE_FACTOR = 1000.0

# Specification of a classifier of three classes, as get_export_specification gives the comparison of its outputs
PROBABILITY_SPECIFICATION = {"outputs": [{"name": "class_probabilities"}], "compare_outputs": compare_probabilities,
                             "comparison_tolerances": {"tolerance": 1e-4}}

# Specification of a segmenter, its class maps compared by the share of equal pixels, for the TF32 share tests
SEGMENTATION_SPECIFICATION = {"outputs": [{"name": "class_map"}, {"name": "class_probabilities"}],
                              "compare_outputs": compare_segmentations,
                              "comparison_tolerances": {"tolerance": 1e-4, "minimum_equal_share": 0.999}}

# Eager probabilities of the engine check tests, a batch of two images
REFERENCE_PROBABILITIES = np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]], dtype=np.float32)

# Error and crash lines of the Triton log of an engine that cannot be built or run, quoted by the mismatch
ENGINE_LOG_ERROR = "E1127 IBuilder::buildSerializedNetwork: Error Code 10: Could not find any implementation"
ENGINE_LOG_CRASH = "Signal (11) received."


def write_overflow_graph(model_path: Path) -> None:
    """
    Writes a small ONNX graph whose intermediate values exceed the largest FP16 value for large enough inputs: its
    'scale_up' Mul node multiplies the input by SCALE_FACTOR and its 'scale_down' Mul node brings it back, so that the
    output stays as small as the input while the intermediate value does not, the case of the NMS offsets of the
    detectors. It also holds an infinity ('reciprocal_of_zero') and an integer copy of the large value ('to_integer'),
    which the scan must leave out.

    Args:
        model_path (Path): The written 'model.onnx'.
    """
    nodes = [helper.make_node("Mul", ["images", "scale_factor"], ["scaled_up"], name="scale_up"),
             helper.make_node("Mul", ["scaled_up", "inverse_factor"], ["scaled_down"], name="scale_down"),
             helper.make_node("Sub", ["images", "images"], ["zeros"], name="subtract_itself"),
             helper.make_node("Reciprocal", ["zeros"], ["infinities"], name="reciprocal_of_zero"),
             helper.make_node("Cast", ["scaled_up"], ["integer_values"], name="to_integer", to=TensorProto.INT64),
             helper.make_node("Add", ["scaled_down", "zeros"], ["outputs"], name="add_zeros")]
    graph = helper.make_graph(
        nodes=nodes, name="overflow_graph",
        inputs=[helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 2])],
        outputs=[helper.make_tensor_value_info("outputs", TensorProto.FLOAT, ["batch", 2]),
                 helper.make_tensor_value_info("infinities", TensorProto.FLOAT, ["batch", 2]),
                 helper.make_tensor_value_info("integer_values", TensorProto.INT64, ["batch", 2])],
        initializer=[helper.make_tensor("scale_factor", TensorProto.FLOAT, [], [SCALE_FACTOR]),
                     helper.make_tensor("inverse_factor", TensorProto.FLOAT, [], [1.0 / SCALE_FACTOR])])
    # IR version 10, read by the installed ONNX Runtime, as the exporter writes it
    onnx.save_model(proto=helper.make_model(graph=graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10),
                    f=str(model_path))


@pytest.mark.parametrize(argnames=("largest_input", "overflowing"), argvalues=[(66.0, True), (65.0, False)])
def test_overflow_scan_flags_the_values_above_the_largest_half_precision_value(tmp_path: Path, largest_input: float,
                                                                                overflowing: bool) -> None:
    """
    Verifies that the FP16 overflow scan flags the 'scale_up' node once one sample of the check batch takes it above
    65,504 (66,000), with its op type and largest magnitude, and nothing at 65,000, the infinity and the integer copy
    of the graph never being flagged, and that the description of an overflow names the node and the settings to
    export with instead.

    Args:
        tmp_path (Path): The pytest temporary directory holding the graph.
        largest_input (float): The largest input value of the check batch, in the second sample.
        overflowing (bool): Whether the scan must flag the 'scale_up' node.
    """
    model_path = tmp_path / "model.onnx"
    write_overflow_graph(model_path=model_path)
    check_inputs = {"images": np.array([[1.0, -2.0], [-largest_input, 3.0]], dtype=np.float32)}
    scan = find_half_precision_overflows(model_path=model_path, check_inputs=check_inputs)

    # The five floating point node outputs are scanned, the integer one left out
    assert scan["scanned_outputs"] == 5
    if not overflowing:
        assert scan["overflows"] == []
        return
    assert scan["overflows"] == [{"output": "scaled_up", "node": "scale_up", "op_type": "Mul",
                                  "largest_magnitude": largest_input * SCALE_FACTOR}]
    assert largest_input * SCALE_FACTOR > HALF_PRECISION_LARGEST_VALUE
    description = "\n".join(describe_half_precision_overflows(overflows=scan["overflows"]))
    assert "scale_up (Mul): largest magnitude 66000" in description
    assert "--tensorrt_precision FP32, or None" in description


class StandInTritonServer:
    """
    Stand-in for the Triton server and client of the engine check, recording the start and stop of its container and
    answering the request with the given outputs, or failing to load the model.
    """

    def __init__(self, engine_outputs: List[np.ndarray], load_error: str) -> None:
        """
        Sets what the stand-in server answers.

        Args:
            engine_outputs (List[np.ndarray]): The outputs of the request.
            load_error (str): The error of the model load, empty for a model that loads.
        """
        self.engine_outputs = engine_outputs
        self.load_error = load_error
        self.events = []

    def start(self, container_name_prefix: str, repository: Path, engine_cache_folder: Path) -> Dict[str, Any]:
        """
        Records the start of the container, standing in for start_temporary_triton_server.

        Args:
            container_name_prefix (str): The start of the container name.
            repository (Path): The served repository.
            engine_cache_folder (Path): The engine cache folder.

        Returns:
            Dict[str, Any]: The server, with its gRPC address.
        """
        self.events.append(f"start {container_name_prefix}")
        return {"grpc_address": "localhost:1"}

    def load(self, server: Dict[str, Any], model_name: str, configuration: Any) -> None:
        """
        Loads the model, or fails as Triton does for an engine it cannot build, standing in for load_served_model.

        Args:
            server (Dict[str, Any]): The server.
            model_name (str): The Triton model name.
            configuration (Any): The configuration given at load time, None for config.pbtxt.

        Raises:
            TritonServerError: If the stand-in has a load error.
        """
        self.events.append(f"load {model_name}")
        if self.load_error:
            raise TritonServerError(self.load_error)

    def infer(self, inputs: Dict[str, np.ndarray], **request_settings: Any) -> List[np.ndarray]:
        """
        Answers the request with the outputs of the stand-in, standing in for run_triton_inference.

        Args:
            inputs (Dict[str, np.ndarray]): The check batches.
            **request_settings (Any): The client, model and output names, ignored.

        Returns:
            List[np.ndarray]: The outputs of the stand-in.
        """
        self.events.append(f"infer {list(inputs)}")
        return self.engine_outputs

    def stop(self, server: Dict[str, Any], log_path: Path) -> None:
        """
        Records the stop of the container and writes its log, standing in for stop_triton_server.

        Args:
            server (Dict[str, Any]): The server.
            log_path (Path): The file receiving the log.
        """
        self.events.append("stop")
        log_path.write_text(f"I1127 loading model\n{ENGINE_LOG_ERROR}\nI1127 unloading model\n{ENGINE_LOG_CRASH}\n")


def run_engine_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine_outputs: List[np.ndarray],
                     load_error: str, tolerances: Dict[str, float],
                     tf32_reference_outputs: Optional[List[np.ndarray]]) -> Dict[str, Any]:
    """
    Runs check_tensorrt_engine on the classifier specification with the Triton server and client stood in for.

    Args:
        tmp_path (Path): The pytest temporary directory, receiving the server log.
        monkeypatch (pytest.MonkeyPatch): Replaces the Triton functions of the engine check.
        engine_outputs (List[np.ndarray]): The outputs the stand-in server answers.
        load_error (str): The error of the model load, empty for a model that loads.
        tolerances (Dict[str, float]): The tolerances of the comparison.
        tf32_reference_outputs (Optional[List[np.ndarray]]): The eager outputs with TF32 on of an FP32 engine, None
            for a comparison at the tolerances only.

    Returns:
        Dict[str, Any]: The result of the engine check, with the 'events' of the stand-in server.
    """
    stand_in = StandInTritonServer(engine_outputs=engine_outputs, load_error=load_error)
    monkeypatch.setattr(target=tensorrt_check, name="start_temporary_triton_server", value=stand_in.start)
    monkeypatch.setattr(target=tensorrt_check, name="load_served_model", value=stand_in.load)
    monkeypatch.setattr(target=tensorrt_check, name="run_triton_inference", value=stand_in.infer)
    monkeypatch.setattr(target=tensorrt_check, name="stop_triton_server", value=stand_in.stop)
    monkeypatch.setattr(target=tensorrt_check.triton_client, name="InferenceServerClient", value=lambda url: None)
    monkeypatch.setattr(target=tensorrt_check, name="get_triton_output_names",
                        value=lambda client, model_name, output_names: output_names)
    engine_check = check_tensorrt_engine(staged_repository=tmp_path / "models",
                                         engine_cache_folder=tmp_path / "engines", log_path=tmp_path / "triton.log",
                                         model_name="image_classifier_onnx",
                                         check_inputs={"images": np.zeros(shape=(2, 3, 4, 4), dtype=np.float32)},
                                         reference_outputs=[REFERENCE_PROBABILITIES],
                                         specification=PROBABILITY_SPECIFICATION, tolerances=tolerances,
                                         tf32_reference_outputs=tf32_reference_outputs)
    return {**engine_check, "events": stand_in.events}


@pytest.mark.parametrize(argnames=("engine_probabilities", "tolerances", "expected_mismatch"), argvalues=[
    (REFERENCE_PROBABILITIES + 5e-5, {"tolerance": 1e-4}, None),
    (REFERENCE_PROBABILITIES + 1e-3, {"tolerance": 1e-4}, "class probabilities differ by up to 1.00e-03"),
    (REFERENCE_PROBABILITIES + 1e-3, {"tolerance": 1e-4 * HALF_PRECISION_TOLERANCE_FACTOR}, None),
    (np.where(REFERENCE_PROBABILITIES > 0.55, np.inf, REFERENCE_PROBABILITIES), {"tolerance": 1e3},
     "class_probabilities: 1 non-finite values")])
def test_engine_check_compares_the_served_outputs_with_the_eager_model(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch,
                                                                       engine_probabilities: np.ndarray,
                                                                       tolerances: Dict[str, float],
                                                                       expected_mismatch: Any) -> None:
    """
    Verifies that the engine check passes outputs within the tolerances of the comparison, fails differences above
    them (the FP32 tolerance), passes them at the loosened FP16 tolerance, and fails an infinity whatever the
    tolerances, its container being started, given the check batches and stopped every time.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): Replaces the Triton functions of the engine check.
        engine_probabilities (np.ndarray): The probabilities the stand-in engine answers.
        tolerances (Dict[str, float]): The tolerances of the comparison.
        expected_mismatch (Any): The expected mismatch, None when the engine must pass.
    """
    engine_check = run_engine_check(tmp_path=tmp_path, monkeypatch=monkeypatch,
                                    engine_outputs=[engine_probabilities.astype(np.float32)], load_error="",
                                    tolerances=tolerances, tf32_reference_outputs=None)
    assert engine_check["mismatches"] == ([] if expected_mismatch is None else [expected_mismatch])
    assert engine_check["tf32_comparison"] is None
    assert engine_check["events"] == ["start export-tensorrt-check-triton", "load image_classifier_onnx",
                                      "infer ['images']", "stop"]
    assert engine_check["engine_check_seconds"] >= 0 and engine_check["server_start_seconds"] >= 0


def test_engine_triton_cannot_build_is_a_mismatch_with_its_log(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that a model Triton fails to load (an engine it cannot build) is a mismatch quoting the error and the error
    and crash lines of the server log, the container being stopped and no request sent.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): Replaces the Triton functions of the engine check.
    """
    engine_check = run_engine_check(tmp_path=tmp_path, monkeypatch=monkeypatch, engine_outputs=[],
                                    load_error="Triton refused load: onnx runtime error 11\nsecond line",
                                    tolerances={"tolerance": 1e-4}, tf32_reference_outputs=[REFERENCE_PROBABILITIES])
    assert engine_check["mismatches"] == ["Triton failed to build or run the engine: Triton refused load: onnx "
                                          "runtime error 11", f"  log: {ENGINE_LOG_ERROR}",
                                          f"  log: {ENGINE_LOG_CRASH}"]
    assert engine_check["events"] == ["start export-tensorrt-check-triton", "load image_classifier_onnx", "stop"]


def test_tf32_tolerances_follow_the_differences_of_the_eager_model() -> None:
    """
    Verifies that derive_tf32_tolerances measures in multiples of the tolerances how far TF32 moves the eager outputs
    and multiplies the tolerances by TF32_DIFFERENCE_FACTOR times that, that a minimum share of equal outputs allows
    TF32_DISAGREEMENT_FACTOR times the share TF32 changes, measured within the tolerances TF32 needs (the probabilities
    moved too, which the tolerances of the specification would refuse before the share is read), that the tolerances
    of the specification stay the floor, and that outputs TF32 changes in shape keep them.
    """
    # Probabilities moved by 2e-4 by TF32: 2 x the tolerance of 1e-4, so a tolerance of 4 x 2e-4
    tf32_probabilities = REFERENCE_PROBABILITIES + np.float32(2e-4)
    probabilities_comparison = derive_tf32_tolerances(reference_outputs=[REFERENCE_PROBABILITIES],
                                                      tf32_outputs=[tf32_probabilities],
                                                      specification=PROBABILITY_SPECIFICATION,
                                                      tolerances={"tolerance": 1e-4})
    measured_difference = float(np.abs(tf32_probabilities - REFERENCE_PROBABILITIES).max())
    assert probabilities_comparison["tf32_scale"] == pytest.approx(measured_difference / 1e-4, rel=1e-3)
    assert probabilities_comparison["tolerances"]["tolerance"] == pytest.approx(
        TF32_DIFFERENCE_FACTOR * measured_difference, rel=1e-3)

    # Class maps of ten pixels, one changed by TF32, the probabilities moved by 3e-4: the share is read within 3 x the
    # tolerance and drops to 1 - 2 x 0.1, the tolerance becoming 4 x 3e-4
    class_map = np.zeros(shape=(1, 10), dtype=np.int64)
    tf32_class_map = class_map.copy()
    tf32_class_map[0, 3] = 1
    probabilities = np.full(shape=(1, 2, 10), fill_value=0.5, dtype=np.float32)
    segmentation_comparison = derive_tf32_tolerances(
        reference_outputs=[class_map, probabilities], tf32_outputs=[tf32_class_map, probabilities + np.float32(3e-4)],
        specification=SEGMENTATION_SPECIFICATION, tolerances={"tolerance": 1e-4, "minimum_equal_share": 0.999})
    assert segmentation_comparison["tf32_shares"]["minimum_equal_share"] == pytest.approx(0.9, abs=1e-6)
    assert segmentation_comparison["tolerances"]["minimum_equal_share"] == pytest.approx(
        1.0 - TF32_DISAGREEMENT_FACTOR * 0.1, abs=1e-6)
    assert segmentation_comparison["tolerances"]["tolerance"] == pytest.approx(TF32_DIFFERENCE_FACTOR * 3e-4, rel=1e-2)

    # A TF32 difference far below the tolerance: the tolerance of the specification kept as the floor
    floor_comparison = derive_tf32_tolerances(reference_outputs=[REFERENCE_PROBABILITIES],
                                              tf32_outputs=[REFERENCE_PROBABILITIES + np.float32(1e-6)],
                                              specification=PROBABILITY_SPECIFICATION, tolerances={"tolerance": 1e-4})
    assert floor_comparison["tolerances"] == {"tolerance": 1e-4}

    # Outputs of another shape: nothing measurable, the tolerances of the specification kept
    shape_comparison = derive_tf32_tolerances(reference_outputs=[REFERENCE_PROBABILITIES],
                                              tf32_outputs=[REFERENCE_PROBABILITIES[:1]],
                                              specification=PROBABILITY_SPECIFICATION, tolerances={"tolerance": 1e-4})
    assert shape_comparison == {"tf32_scale": None, "tf32_shares": {}, "tolerances": {"tolerance": 1e-4}}


@pytest.mark.parametrize(argnames=("engine_shift", "expected_mismatches"), argvalues=[
    (5e-4, []), (1e-2, ["class probabilities differ by up to 1.00e-02"])])
def test_fp32_engine_is_accepted_within_the_tf32_differences(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                             engine_shift: float,
                                                             expected_mismatches: List[str]) -> None:
    """
    Verifies that an FP32 engine missing the tolerance of the specification (1e-4) is compared again within
    TF32_DIFFERENCE_FACTOR times the difference TF32 makes in the eager model (2e-4 here): an engine 5e-4 away passes,
    one 1e-2 away still fails, and both record what they missed at the tolerance of the specification.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): Replaces the Triton functions of the engine check.
        engine_shift (float): The difference of the stand-in engine probabilities from the eager ones.
        expected_mismatches (List[str]): The mismatches expected within the TF32 differences.
    """
    engine_probabilities = (REFERENCE_PROBABILITIES + np.float32(engine_shift)).astype(np.float32)
    engine_check = run_engine_check(tmp_path=tmp_path, monkeypatch=monkeypatch, engine_outputs=[engine_probabilities],
                                    load_error="", tolerances={"tolerance": 1e-4},
                                    tf32_reference_outputs=[REFERENCE_PROBABILITIES + np.float32(2e-4)])
    assert engine_check["mismatches"] == expected_mismatches
    engine_difference = float(np.abs(engine_probabilities - REFERENCE_PROBABILITIES).max())
    assert engine_check["tf32_comparison"]["tolerances_mismatches"] == [
        f"class probabilities differ by up to {engine_difference:.2e}"]
    assert engine_check["tf32_comparison"]["tolerances"]["tolerance"] == pytest.approx(TF32_DIFFERENCE_FACTOR * 2e-4,
                                                                                       rel=1e-2)
    assert describe_tf32_tolerances(tf32_comparison=engine_check["tf32_comparison"]).startswith(
        "eager TF32 on vs off within 2.00 x the tolerances")


class EngineCheckRecorder:
    """
    Stand-in for check_tensorrt_engine in export_experiment, recording the reference and tolerances it is given and
    answering with the given mismatches.
    """

    def __init__(self, mismatches: List[str]) -> None:
        """
        Sets the mismatches of the stand-in engine check.

        Args:
            mismatches (List[str]): The mismatches answered.
        """
        self.mismatches = mismatches
        self.calls = []

    def __call__(self, **check_settings: Any) -> Dict[str, Any]:
        """
        Records the settings of the engine check and answers with the mismatches of the stand-in.

        Args:
            **check_settings (Any): The arguments of check_tensorrt_engine.

        Returns:
            Dict[str, Any]: The result of the stand-in engine check.
        """
        self.calls.append(check_settings)
        return {"mismatches": self.mismatches, "tf32_comparison": None, "server_start_seconds": 1.0,
                "engine_check_seconds": 2.0}


def export_classifier(experiment_folder: Path, repository: Path, tensorrt_precision: str,
                      skip_tensorrt_check: bool) -> None:
    """
    Exports the ONNX model of the test classifier with TensorRT, as export_model.py does from its command line.

    Args:
        experiment_folder (Path): The run folder, from write_trained_model.
        repository (Path): The model repository receiving the export.
        tensorrt_precision (str): 'FP32' or 'FP16'.
        skip_tensorrt_check (bool): True to leave the engine unchecked.
    """
    export_experiment(experiment_folder=experiment_folder, iteration=test_input_contract.EXPORTED_ITERATION,
                      use_averaged_weights=False, formats=["onnx"], repository=repository, model_version=1,
                      maximum_batch_size=test_input_contract.TEST_MAXIMUM_BATCH_SIZE,
                      tensorrt_precision=tensorrt_precision, torchscript_precision="FP32", use_saved_temperature=False,
                      saved_thresholds_path=None, test_split_parity=False, skip_tensorrt_check=skip_tensorrt_check)


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="The export runs the model on the GPU.")
def test_failed_engine_check_blocks_the_installation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that an export whose TensorRT engine check fails raises with the mismatch and leaves the repository
    without any model nor staging folder, that a passing FP32 check installs the ONNX model, the engine being compared
    with the FP32 eager outputs at the FP32 tolerances, that an FP16 engine is compared with the eager FP16 copy at the
    FP32 tolerances times HALF_PRECISION_TOLERANCE_FACTOR after the overflow scan of its graph passed, and that
    skip_tensorrt_check installs the model without any engine check.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): Stands in for the Triton parts of the engine check.
    """
    experiment_folder, repository = tmp_path / "CLS_TEST_MC3", tmp_path / "served_models"
    test_input_contract.write_trained_model(experiment_folder=experiment_folder, task="classification",
                                            keep_ratio=True, tiling_configuration=None)
    monkeypatch.setattr(target=export_model, name="check_triton_available", value=lambda purpose, remedy: None)

    # A failing engine check raises, the repository staying empty
    failing_check = EngineCheckRecorder(mismatches=["class probabilities differ by up to 3.00e-02"])
    monkeypatch.setattr(target=export_model, name="check_tensorrt_engine", value=failing_check)
    with pytest.raises(expected_exception=RuntimeError, match=re.escape(pattern="Fails Its TensorRT FP32 Check")):
        export_classifier(experiment_folder=experiment_folder, repository=repository, tensorrt_precision="FP32",
                          skip_tensorrt_check=False)
    assert read_repository_files(repository=repository) == {}
    assert not list(tmp_path.glob(pattern=f"{STAGING_FOLDER_PREFIX}*"))

    # A passing FP32 engine check installs the model, its reference being the FP32 eager model
    passing_check = EngineCheckRecorder(mismatches=[])
    monkeypatch.setattr(target=export_model, name="check_tensorrt_engine", value=passing_check)
    export_classifier(experiment_folder=experiment_folder, repository=repository, tensorrt_precision="FP32",
                      skip_tensorrt_check=False)
    assert "image_classifier_onnx/1/model.onnx" in read_repository_files(repository=repository)
    fp32_call = passing_check.calls[0]
    assert fp32_call["model_name"] == "image_classifier_onnx" and fp32_call["tolerances"] == {"tolerance": 1e-4}
    assert fp32_call["staged_repository"].name == "models"

    # The FP32 engine also gets the eager outputs with TF32 on, for the comparison within the TF32 roundings
    assert [output.shape for output in fp32_call["tf32_reference_outputs"]] == [
        output.shape for output in fp32_call["reference_outputs"]]

    # FP16: the scan of the graph passes, then the engine is compared with the eager FP16 copy at loosened tolerances
    export_classifier(experiment_folder=experiment_folder, repository=repository, tensorrt_precision="FP16",
                      skip_tensorrt_check=False)
    fp16_call = passing_check.calls[1]
    assert fp16_call["tolerances"] == {"tolerance": 1e-4 * HALF_PRECISION_TOLERANCE_FACTOR}
    assert fp16_call["tf32_reference_outputs"] is None
    reference_difference = np.abs(fp16_call["reference_outputs"][0] - fp32_call["reference_outputs"][0]).max()
    assert 0 < reference_difference < 1e-2

    # Skipped: no engine check, the model installed
    export_classifier(experiment_folder=experiment_folder, repository=repository, tensorrt_precision="FP16",
                      skip_tensorrt_check=True)
    assert len(passing_check.calls) == 2


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="The export runs the model on the GPU.")
def test_overflowing_graph_blocks_an_fp16_installation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that an FP16 export whose overflow scan finds a value above 65,504 raises with the overflowing node and
    leaves the repository empty, without starting the engine check, even with skip_tensorrt_check.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): Makes the overflow scan find an overflow.
    """
    experiment_folder, repository = tmp_path / "CLS_TEST_MC3", tmp_path / "served_models"
    test_input_contract.write_trained_model(experiment_folder=experiment_folder, task="classification",
                                            keep_ratio=True, tiling_configuration=None)
    engine_check = EngineCheckRecorder(mismatches=[])
    monkeypatch.setattr(target=export_model, name="check_tensorrt_engine", value=engine_check)
    overflow = {"output": "scaled", "node": "node_Mul_7", "op_type": "Mul", "largest_magnitude": 1.5e5}
    monkeypatch.setattr(target=export_model, name="find_half_precision_overflows",
                        value=lambda model_path, check_inputs: {"scanned_outputs": 9, "overflows": [overflow]})
    with pytest.raises(expected_exception=RuntimeError, match=re.escape(pattern="node_Mul_7 (Mul): largest magnitude")):
        export_classifier(experiment_folder=experiment_folder, repository=repository, tensorrt_precision="FP16",
                          skip_tensorrt_check=True)
    assert read_repository_files(repository=repository) == {}
    assert engine_check.calls == []


def test_engine_check_without_docker_stops_before_the_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that an export needing the TensorRT engine check on a machine without Docker stops before the run is read
    or the repository created, the error naming --skip_tensorrt_check and --tensorrt_precision None.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): Hides the docker command and shows a GPU.
    """
    monkeypatch.setattr(triton_server.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(triton_server.shutil, "which", lambda cmd: None)
    repository = tmp_path / "served_models"
    with pytest.raises(expected_exception=RuntimeError,
                       match="TensorRT FP16 engine check .* docker command is not found .* --skip_tensorrt_check .* "
                             "--tensorrt_precision None"):
        export_classifier(experiment_folder=tmp_path / "missing_run", repository=repository, tensorrt_precision="FP16",
                          skip_tensorrt_check=False)
    assert not repository.exists()
