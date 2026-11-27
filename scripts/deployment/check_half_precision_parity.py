"""
Test split parity of a classifier traced in half precision (an FP16 TorchScript '.pt') without Triton, a lighter
companion of the FP16 TorchScript export of vision_training/deployment/export_model.py (torchscript_precision 'FP16'
with test_split_parity) that also gives the deployed threshold decisions that move. The classifier of a run is loaded
as the export loads it (load_trained_model, build_inference_model, and with use_saved_temperature the temperature
saved from its temperature explorer), converted to FP16 as the export converts it (convert_to_half_precision of
deployment/export_formats.py: convolution and linear weights in FP16, normalization layers, temperature and softmax in
FP32), traced to '<run>/Half_Precision_Parity/Iteration_<n>[_EMA]/model_fp16.pt', then scored with the eager FP32
model on the whole test split of the run by the test evaluation of its trainer (evaluate_parity_row of
deployment/export_parity.py). The summary gives the metrics of both rows, the predicted classes and the deployed
threshold decisions that moved, and the sizes of the files, so that the worth of an FP16 TorchScript export of a
classifier can be judged before it is deployed. Run as a script with the run folder and the settings of the trial
given as arguments (parse_half_precision_parity_arguments), named as those of export_model.py, its help printed when
none is given.
"""

import argparse
import functools
import json
import shutil
import sys
import numpy as np
import torch

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.deployment.deployment_utilities import run_local_inference
from vision_training.deployment.export_formats import (convert_to_half_precision, export_torchscript,
                                                       hide_known_export_warnings)
from vision_training.deployment.export_model import (DEFAULT_EXPORT_SETTINGS, find_saved_thresholds_path,
                                                     read_saved_temperature)
from vision_training.deployment.export_parity import (ROW_EVALUATIONS_FOLDER_NAME, build_evaluation_trainer,
                                                      create_output_comparison, evaluate_parity_row,
                                                      summarise_sample_agreement)
from vision_training.deployment.export_specifications import (build_inference_model, convert_outputs_to_numpy,
                                                              create_random_inputs, get_export_specification)
from vision_training.models.model_loading import load_trained_model
from vision_training.evaluation.report_formatting import format_points, format_report_value
from vision_training.trainers.checkpointing import get_checkpoint_path, get_last_checkpoint_iteration
from vision_training.utilities.os_utilities import format_path_link, print_color

# Folder of the run receiving the trial, one subfolder per checkpoint and weights, as Export_Parity
HALF_PRECISION_PARITY_FOLDER_NAME = "Half_Precision_Parity"
HALF_PRECISION_MODEL_NAME = "model_fp16.pt"
REFERENCE_MODEL_NAME = "model_fp32.pt"
SUMMARY_MARKDOWN_NAME = "half_precision_parity.md"
SUMMARY_JSON_NAME = "half_precision_parity.json"

# Global metrics of the table, the first one the headline of the classification trainer for one class per image
TABLE_METRICS = ["Balanced Accuracy", "Accuracy", "Macro F1"]


def record_request_outputs(inputs: Dict[str, np.ndarray], run_model: Callable[..., List[np.ndarray]],
                           recorded_probabilities: List[np.ndarray]) -> List[np.ndarray]:
    """
    Runs one request of the parity through a model and keeps its class probabilities, so that the eager and FP16
    probabilities of the same requests, recorded in the same order, can be compared image by image after the row.

    Args:
        inputs (Dict[str, np.ndarray]): The inputs of the request, keyed by input name.
        run_model (Callable[..., List[np.ndarray]]): Runs one request ('inputs' keyword) through the model.
        recorded_probabilities (List[np.ndarray]): The probabilities of the earlier requests, extended.

    Returns:
        List[np.ndarray]: The outputs of the model.
    """
    outputs = run_model(inputs=inputs)
    recorded_probabilities.append(outputs[0])
    return outputs


def read_deployed_thresholds(thresholds_path: Path, class_names: List[str]) -> np.ndarray:
    """
    Reads the class thresholds deployed with the exported classifier, in the class order of the model, to count the
    images whose deployed decision (score at least the threshold of a class) changes in half precision.

    Args:
        thresholds_path (Path): The saved_temperature_thresholds.json deployed with the export.
        class_names (List[str]): The class names of the run, in class index order.

    Returns:
        np.ndarray: The [K] thresholds.

    Raises:
        ValueError: If the file does not hold the threshold of every class of the run.
    """
    thresholds_file = json.loads(thresholds_path.read_text())
    if sorted(thresholds_file["classes"]) != sorted(class_names):
        raise ValueError(f"Expected the thresholds of the classes {class_names} in {thresholds_path}, got "
                         f"{sorted(thresholds_file['classes'])}.")
    return np.array([thresholds_file["classes"][class_name]["threshold"] for class_name in class_names],
                    dtype=np.float64)


def compare_half_precision_outputs(eager_probabilities: np.ndarray, half_probabilities: np.ndarray,
                                   thresholds: Optional[np.ndarray]) -> Dict[str, Any]:
    """
    Compares the eager FP32 and FP16 probabilities of every test image: the predicted classes that changed, the images
    whose deployed decision changed for at least one class, and the largest and mean probability differences.

    Args:
        eager_probabilities (np.ndarray): The [N, K] eager probabilities.
        half_probabilities (np.ndarray): The [N, K] FP16 probabilities of the same images, in the same order.
        thresholds (Optional[np.ndarray]): The [K] deployed thresholds, None when the export deploys none.

    Returns:
        Dict[str, Any]: The 'compared_images', 'predicted_class_changes', 'decision_changes' (with the
            'decision_changes_per_class', both None without thresholds) and the 'largest_difference' and
            'mean_difference' of the probabilities.

    Raises:
        ValueError: If the two arrays do not have the same shape.
    """
    if eager_probabilities.shape != half_probabilities.shape:
        raise ValueError(f"Expected eager and FP16 probabilities of the same shape, got {eager_probabilities.shape} "
                         f"and {half_probabilities.shape}.")
    differences = np.abs(eager_probabilities.astype(np.float64) - half_probabilities.astype(np.float64))
    comparison = {"compared_images": int(eager_probabilities.shape[0]),
                  "predicted_class_changes": int((eager_probabilities.argmax(axis=1)
                                                  != half_probabilities.argmax(axis=1)).sum()),
                  "decision_changes": None, "decision_changes_per_class": None,
                  "largest_difference": float(differences.max(initial=0.0)),
                  "mean_difference": float(differences.mean())}

    # Images whose decision at the deployed threshold of at least one class changed
    if thresholds is not None:
        changed_decisions = (eager_probabilities >= thresholds) != (half_probabilities >= thresholds)
        comparison["decision_changes"] = int(changed_decisions.any(axis=1).sum())
        comparison["decision_changes_per_class"] = changed_decisions.sum(axis=0).tolist()
    return comparison


def build_parity_table(summary: Dict[str, Any]) -> str:
    """
    Writes the summary of the trial read by a person: the checkpoint, temperature and thresholds, then one table of
    the eager FP32 and FP16 TorchScript rows with their metrics, changes, image agreement and file sizes.

    Args:
        summary (Dict[str, Any]): The summary of check_half_precision_parity.

    Returns:
        str: The Markdown text.
    """
    eager_row, half_row = summary["rows"]
    comparison, agreement = summary["comparison"], summary["agreement"]
    number_images = comparison["compared_images"]
    lines = [f"# Half Precision Parity · {Path(summary['run']).name}", "",
             f"- Checkpoint: `{summary['checkpoint']}` (iteration {summary['iteration']}, {summary['weights']})",
             f"- Test split: `{summary['test_data_folder']}`, {eager_row['evaluated_samples']}",
             f"- Logit temperature {summary['logit_temperature']:g}, thresholds "
             f"{'not deployed' if summary['thresholds_path'] is None else '`' + summary['thresholds_path'] + '`'}",
             f"- Random batch check of the FP16 file against the eager model: "
             f"{'; '.join(summary['random_batch_check']) or 'within the FP32 tolerances of the export'}", "",
             "| Metric | Eager FP32 | TorchScript FP16 | Change |", "|---|---|---|---|"]

    # Global metrics of both rows, the change of the FP16 row in points
    for metric_name in TABLE_METRICS:
        eager_value, half_value = eager_row["scalars"][metric_name], half_row["scalars"][metric_name]
        lines.append(f"| {metric_name} | {format_report_value(value=eager_value, in_percent=True)} | "
                     f"{format_report_value(value=half_value, in_percent=True)} | "
                     f"{format_points(difference=half_value - eager_value)} |")

    # Image agreement and file sizes
    lines += [f"| Predicted class changed | | {comparison['predicted_class_changes']} of {number_images} images | |",
              f"| Deployed decision changed | | "
              f"{'no threshold deployed' if comparison['decision_changes'] is None else comparison['decision_changes']}"
              f"{'' if comparison['decision_changes'] is None else f' of {number_images} images'} | |",
              f"| Within the FP32 tolerances of the export | | {agreement['within_tolerance']} of "
              f"{agreement['compared_samples']} images | |",
              f"| Largest probability difference | | {comparison['largest_difference']:.2e} | |",
              f"| Mean probability difference | | {comparison['mean_difference']:.2e} | |",
              f"| TorchScript file | {summary['file_sizes']['fp32'] / 1e6:.1f} MB | "
              f"{summary['file_sizes']['fp16'] / 1e6:.1f} MB | |"]
    return "\n".join(lines) + "\n"


def check_half_precision_parity(experiment_folder: Path, iteration: Optional[int], use_averaged_weights: bool,
                                use_saved_temperature: bool, saved_thresholds_path: Optional[Path],
                                maximum_batch_size: int) -> Dict[str, Any]:
    """
    Traces the classifier of a run in half precision and scores it with the eager FP32 model on the whole test split
    of the run, as the test split parity of an export does, with the temperature and thresholds saved from its
    temperature explorer when the export deploys them (use_saved_temperature), then writes the summary into
    '<run>/Half_Precision_Parity/Iteration_<n>[_EMA]/' next to the traced FP16 and FP32 files (an earlier trial of the
    same checkpoint and weights replaced).

    Args:
        experiment_folder (Path): The run folder holding 'Weights' and 'training_configuration.yaml'.
        iteration (Optional[int]): The iteration of the checkpoint, None for the last one.
        use_averaged_weights (bool): Whether the moving average of the weights is traced.
        use_saved_temperature (bool): True traces the temperature saved from the temperature explorer of the
            checkpoint and compares the decisions at its thresholds, as an export with use_saved_temperature; False
            traces a temperature of 1 without thresholds.
        saved_thresholds_path (Optional[Path]): With use_saved_temperature, the saved_temperature_thresholds.json of
            the deployed temperature and thresholds, None to look for it in the evaluation folders of the checkpoint
            (find_saved_thresholds_path).
        maximum_batch_size (int): The largest batch size of a request, that of the export.

    Returns:
        Dict[str, Any]: The summary of the trial.

    Raises:
        ValueError: If the run is not a classification run, or saved_thresholds_path is given without
            use_saved_temperature.
    """
    # Full precision reference as in the export: the TF32 convolutions of PyTorch turned off, restored at the end
    previous_cudnn_tf32, previous_matmul_tf32 = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = False, False
    try:
        device = torch.device("cuda")
        configuration_path = experiment_folder / "training_configuration.yaml"
        experiment_configuration, _ = load_experiment_configuration(configuration_path=configuration_path)
        if experiment_configuration["task"] != "classification":
            raise ValueError(f"Expected a classification run, got the {experiment_configuration['task']} task.")
        if saved_thresholds_path is not None and not use_saved_temperature:
            raise ValueError(f"saved_thresholds_path {saved_thresholds_path} is only read with use_saved_temperature.")
        checkpoint_path = get_checkpoint_path(experiment_folder=experiment_folder, iteration=iteration)
        checkpoint_iteration = iteration if iteration is not None else get_last_checkpoint_iteration(
            weights_directory=experiment_folder / "Weights")

        # Eager FP32 inference classifier of the export, with the saved temperature when the export deploys it
        model = load_trained_model(configuration_path=configuration_path, checkpoint_path=checkpoint_path,
                                   device=device, use_averaged_weights=use_averaged_weights)
        inference_model = build_inference_model(task="classification", model=model)
        specification = get_export_specification(task="classification", inference_model=inference_model)
        thresholds_path, logit_temperature = None, 1.0
        if use_saved_temperature:
            thresholds_path = saved_thresholds_path if saved_thresholds_path is not None else (
                find_saved_thresholds_path(experiment_folder=experiment_folder, checkpoint_path=checkpoint_path,
                                           use_averaged_weights=use_averaged_weights))
            logit_temperature = read_saved_temperature(saved_thresholds_path=thresholds_path,
                                                       number_classes=inference_model.number_classes)
            inference_model.set_logit_temperature(logit_temperature=logit_temperature)
        print_color(color="blue", output=f"Half Precision Parity : {format_path_link(path=checkpoint_path)}, logit "
                                         f"temperature {logit_temperature:g}", add_separators=True)

        # Folder of the trial, an earlier one of the same checkpoint and weights replaced
        weights_suffix = "_EMA" if use_averaged_weights else ""
        parity_folder = (experiment_folder / HALF_PRECISION_PARITY_FOLDER_NAME /
                         f"{checkpoint_path.parent.name}{weights_suffix}")
        if parity_folder.exists():
            shutil.rmtree(path=parity_folder)
        parity_folder.mkdir(parents=True)

        # FP32 and FP16 TorchScript files traced on the same random batch, the FP16 one checked on another batch
        example_inputs = create_random_inputs(specification=specification, shape_name="example_shape",
                                              maximum_batch_size=maximum_batch_size, device=device, generator=None)
        check_inputs = create_random_inputs(specification=specification, shape_name="check_shape",
                                            maximum_batch_size=maximum_batch_size, device=device, generator=None)
        half_model = convert_to_half_precision(inference_model=inference_model, input_names=["images"])
        export_torchscript(inference_model=inference_model, example_inputs=example_inputs,
                           output_path=parity_folder / REFERENCE_MODEL_NAME)
        export_torchscript(inference_model=half_model, example_inputs=example_inputs,
                           output_path=parity_folder / HALF_PRECISION_MODEL_NAME)
        traced_half_model = torch.jit.load(str(parity_folder / HALF_PRECISION_MODEL_NAME), map_location=device)
        del half_model
        with torch.no_grad():
            random_batch_check = specification["compare_outputs"](
                reference=convert_outputs_to_numpy(outputs=inference_model(**check_inputs)),
                candidate=convert_outputs_to_numpy(outputs=traced_half_model(**check_inputs)),
                **specification["comparison_tolerances"])

        # Trainer of the run for its test evaluation, its files written into the folder of every row
        test_data_folder = Path(experiment_configuration["test_data_folder"])
        trainer = build_evaluation_trainer(experiment_folder=experiment_folder, test_data_folder=test_data_folder,
                                           evaluation_folder=parity_folder,
                                           standalone_evaluation_folder=ROW_EVALUATIONS_FOLDER_NAME,
                                           batch_size=maximum_batch_size)
        output_names = [output["name"] for output in specification["outputs"]]
        run_eager_model = functools.partial(run_local_inference, inference_model=inference_model,
                                            decoding_settings=None)

        # Eager FP32 row, the reference
        print_color(color="blue", output="Row : Eager FP32", indent=1)
        eager_results = evaluate_parity_row(trainer=trainer, run_model=run_eager_model, output_names=output_names,
                                            maximum_request_size=maximum_batch_size, iteration=checkpoint_iteration,
                                            row_folder=parity_folder / ROW_EVALUATIONS_FOLDER_NAME / "eager_fp32",
                                            comparison=None)

        # FP16 row, every request also run by the eager model, both probabilities recorded in the order of the requests
        print_color(color="blue", output="Row : TorchScript FP16", indent=1)
        eager_probabilities, half_probabilities = [], []
        comparison = create_output_comparison(
            task="classification", run_reference=functools.partial(
                record_request_outputs, run_model=run_eager_model, recorded_probabilities=eager_probabilities),
            specification=specification, name_sample=trainer.get_test_sample_name)
        run_half_model = functools.partial(
            record_request_outputs, run_model=functools.partial(run_local_inference, inference_model=traced_half_model,
                                                                decoding_settings=None),
            recorded_probabilities=half_probabilities)
        half_results = evaluate_parity_row(trainer=trainer, run_model=run_half_model, output_names=output_names,
                                           maximum_request_size=maximum_batch_size, iteration=checkpoint_iteration,
                                           row_folder=parity_folder / ROW_EVALUATIONS_FOLDER_NAME / "torchscript_fp16",
                                           comparison=comparison)

        # Image by image comparison at the deployed thresholds, then the summary
        image_comparison = compare_half_precision_outputs(
            eager_probabilities=np.concatenate(eager_probabilities, axis=0),
            half_probabilities=np.concatenate(half_probabilities, axis=0),
            thresholds=None if thresholds_path is None else read_deployed_thresholds(thresholds_path=thresholds_path,
                                                                                   class_names=trainer.class_names))
        summary = {"run": str(experiment_folder), "checkpoint": str(checkpoint_path), "iteration": checkpoint_iteration,
                   "weights": "EMA weights" if use_averaged_weights else "trained weights",
                   "test_data_folder": str(test_data_folder), "logit_temperature": logit_temperature,
                   "thresholds_path": None if thresholds_path is None else str(thresholds_path),
                   "maximum_batch_size": maximum_batch_size,
                   "random_batch_check": random_batch_check,
                   "rows": [{"label": "Eager FP32", **eager_results}, {"label": "TorchScript FP16", **half_results}],
                   "comparison": image_comparison,
                   "agreement": {key: value for key, value in summarise_sample_agreement(
                       records=comparison["records"]).items() if key != "worst_samples"},
                   "file_sizes": {"fp32": (parity_folder / REFERENCE_MODEL_NAME).stat().st_size,
                                  "fp16": (parity_folder / HALF_PRECISION_MODEL_NAME).stat().st_size},
                   "created_at": datetime.now().isoformat(timespec="seconds"), "parity_folder": str(parity_folder)}
        (parity_folder / SUMMARY_MARKDOWN_NAME).write_text(data=build_parity_table(summary=summary))
        (parity_folder / SUMMARY_JSON_NAME).write_text(data=json.dumps(obj=summary, indent=2, default=str))
        print(build_parity_table(summary=summary))
        print_color(color="green", output=f"Summary : {format_path_link(path=parity_folder / SUMMARY_MARKDOWN_NAME)}")
        del trainer
        torch.cuda.empty_cache()
        return summary
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = (previous_cudnn_tf32,
                                                                                   previous_matmul_tf32)


def parse_half_precision_parity_arguments(arguments: List[str]) -> argparse.Namespace:
    """
    Reads the settings of the trial from the command line of the script, named and defaulting as the arguments of
    export_model.py, so that the trial judges the FP16 file the export would write: the run folder is required, the
    largest request defaults to the maximum batch size of DEFAULT_EXPORT_SETTINGS. Launched without any argument, the
    script prints the help of every argument and stops. A '~' left by the shell at the start of a path
    ('--experiment_folder=~/Data/...') is expanded to the home folder.

    Args:
        arguments (List[str]): The command line arguments, without the script name.

    Returns:
        argparse.Namespace: The arguments of check_half_precision_parity.

    Raises:
        SystemExit: After printing the help when no argument is given, or the usage and the error when the run folder
            is missing or a value is not a number.
    """
    # The description and the example keep their line breaks, argparse wrapping the help of every argument
    parser = argparse.ArgumentParser(
        prog="uv run python -m scripts.deployment.check_half_precision_parity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Traces the classifier of a run in FP16 as the export does and scores it with the eager FP32\n"
                    "model on the whole test split of the run, without Triton.",
        epilog="Example, iteration 2000 of a classifier with the temperature saved from its explorer:\n"
               "  uv run python -m scripts.deployment.check_half_precision_parity \\\n"
               "      --experiment_folder ~/Data/<run> --iteration 2000 --use_saved_temperature")
    # The run, the checkpoint and the weights traced
    parser.add_argument("--experiment_folder", type=Path, required=True,
                        help="Classification run folder holding 'Weights' and 'training_configuration.yaml' "
                             "(required).")
    parser.add_argument("--iteration", type=int, default=None,
                        help="Iteration of the traced checkpoint (default: the last saved one).")
    parser.add_argument("--use_averaged_weights", action="store_true",
                        help="Trace the moving average of the weights saved in the checkpoint instead of the trained "
                             "weights.")
    # The temperature and thresholds deployed with the classifier, and the largest request
    parser.add_argument("--use_saved_temperature", action="store_true",
                        help="Trace the temperature saved from the temperature explorer of the checkpoint and compare "
                             "the decisions at its thresholds (default: a temperature of 1, without thresholds).")
    parser.add_argument("--saved_thresholds_path", type=Path, default=None,
                        help="With --use_saved_temperature, the saved_temperature_thresholds.json to use (default: the "
                             "one in the evaluation folders of the checkpoint, looked for as the export does).")
    parser.add_argument("--maximum_batch_size", type=int, default=DEFAULT_EXPORT_SETTINGS["maximum_batch_size"],
                        help="Largest batch of a request, that of the export (default: %(default)s).")

    # Without any argument, the script describes its arguments instead of failing on the missing run folder
    if not arguments:
        parser.print_help()
        parser.exit(status=0)
    parsed_arguments = parser.parse_args(args=arguments)

    # A '~' the shell leaves after '=' points to the home folder, as in the other scripts of the repository
    for path_name in ["experiment_folder", "saved_thresholds_path"]:
        if getattr(parsed_arguments, path_name) is not None:
            setattr(parsed_arguments, path_name, getattr(parsed_arguments, path_name).expanduser())
    return parsed_arguments


if __name__ == "__main__":
    # Run folder and settings of the trial read from the command line, the help printed when none is given
    parity_arguments = parse_half_precision_parity_arguments(arguments=sys.argv[1:])

    # The harmless library warnings printed while tracing are always hidden
    hide_known_export_warnings()
    check_half_precision_parity(experiment_folder=parity_arguments.experiment_folder,
                                iteration=parity_arguments.iteration,
                                use_averaged_weights=parity_arguments.use_averaged_weights,
                                use_saved_temperature=parity_arguments.use_saved_temperature,
                                saved_thresholds_path=parity_arguments.saved_thresholds_path,
                                maximum_batch_size=parity_arguments.maximum_batch_size)
