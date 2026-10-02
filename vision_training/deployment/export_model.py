import logging
import warnings
import onnxruntime
import torch

from pathlib import Path
from torch import nn
from typing import Any, Dict, List, Optional

from vision_training.configuration.configuration_loader import load_experiment_configuration
from vision_training.deployment.export_specifications import (build_inference_model, convert_outputs_to_numpy,
                                                              get_export_specification)
from vision_training.deployment.triton_configuration import write_triton_configuration
from vision_training.trainers.checkpointing import get_checkpoint_path
from vision_training.models.model_loading import load_trained_model
from vision_training.utilities.os_utilities import print_color

# Default Triton model repository, mounted by deployment/triton/docker-compose.yml
DEFAULT_REPOSITORY = Path(__file__).absolute().parent / "triton" / "served_models"


def hide_known_export_warnings() -> None:
    """
    Hides the warnings printed by PyTorch, torchvision and the ONNX library during the export, which come from inside
    these libraries and cannot be fixed in the project code. They are harmless for this export: the deprecation
    notices of TorchScript, of a torchvision helper and of a size check emitted by torch.export for the dynamic
    dimensions, the tracer notice raised inside torchvision's RoIAlign, the notice of torch.export about the weights
    the GRU and LSTM layers flatten while running, and the ONNX notice about an empty attribute defaulting to
    integers. The exports are checked against the eager model anyway.
    """
    # Deprecation notices of torch.jit.trace / trace_method / load, of a torchvision export helper and of the size
    # checks written by torch.export
    warnings.filterwarnings(action="ignore", message=r"`torch\.jit\.", category=FutureWarning)
    warnings.filterwarnings(action="ignore", message=r"`create_unbacked_symint` is deprecated", category=FutureWarning)
    warnings.filterwarnings(action="ignore", message=r"_check_is_size will be removed", category=FutureWarning)

    # Weights flattened by the GRU and LSTM layers at every call, reported by torch.export as assigned attributes
    warnings.filterwarnings(action="ignore", message=r"The tensor attributes .*_flat_weights", category=UserWarning)

    # Tracer notice raised by the shape assertion inside torchvision's RoIAlign
    warnings.filterwarnings(action="ignore", message=r"Converting a tensor to a Python boolean",
                            category=torch.jit.TracerWarning)

    # The ONNX library reports the empty attribute through the logging module rather than the warnings module
    logging.getLogger(name="onnx_ir._convenience").setLevel(level=logging.ERROR)


def export_torchscript(inference_model: nn.Module, example_images: torch.Tensor, output_path: Path) -> None:
    """
    Traces an inference model into a self-contained TorchScript file served by Triton's PyTorch (libtorch) backend.
    The traced graph keeps the batch dimension (and the number of frames of the sequence models) dynamic.

    Args:
        inference_model (nn.Module): The inference model to export, in evaluation mode.
        example_images (torch.Tensor): An example [B, 3, H, W] (or [B, T, 3, H, W]) batch used to trace the graph.
        output_path (Path): The path of the written 'model.pt' file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        traced_model = torch.jit.trace(inference_model, example_inputs=(example_images,), check_trace=False)
    traced_model.save(str(output_path))


def export_onnx(inference_model: nn.Module, example_images: torch.Tensor, output_path: Path,
                specification: Dict[str, Any], maximum_batch_size: int) -> None:
    """
    Exports an inference model to an ONNX file served by Triton's ONNX Runtime backend (optionally accelerated by
    TensorRT), with a dynamic batch dimension up to the maximum batch size and, for the sequence models, a dynamic
    number of frames up to the sequence length of the training.

    Args:
        inference_model (nn.Module): The inference model to export, in evaluation mode.
        example_images (torch.Tensor): An example [B, 3, H, W] (or [B, T, 3, H, W]) batch used to export the graph.
        output_path (Path): The path of the written 'model.onnx' file.
        specification (Dict[str, Any]): The export specification of the task, giving the output names and whether
            the number of frames is dynamic.
        maximum_batch_size (int): The largest batch size the exported model accepts.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_dimensions = {0: torch.export.Dim("batch", min=1, max=maximum_batch_size)}
    if specification["dynamic_frames"]:
        dynamic_dimensions[1] = torch.export.Dim("frames", min=1, max=specification["maximum_frames"])
    torch.onnx.export(inference_model, args=(example_images,), f=str(output_path), input_names=["images"],
                      output_names=[output["name"] for output in specification["outputs"]],
                      dynamic_shapes={"images": dynamic_dimensions}, opset_version=18, dynamo=True, verbose=False)


def export_experiment(experiment_folder: Path, iteration: Optional[int], formats: List[str], repository: Path,
                      model_version: int, maximum_batch_size: int, tensorrt_precision: Optional[str]) -> None:
    """
    Exports a trained experiment of any task to the Triton model repository in every requested format, writes the
    Triton configurations, and checks each export against the eager inference model on a random batch of another
    size (and, for the sequence models, of another number of frames) than the traced one. The inference model, the
    Triton model names, the outputs and the comparison come from the export specification of the task of the run.

    Args:
        experiment_folder (Path): The experiment folder holding 'Weights' and 'training_configuration.yaml'.
        iteration (Optional[int]): The iteration to export, or None for the last saved one.
        formats (List[str]): The export formats, among 'torchscript' and 'onnx'.
        repository (Path): The Triton model repository.
        model_version (int): The Triton version of the exported models, i.e. the name of their version folder.
        maximum_batch_size (int): The largest batch size of the exported models.
        tensorrt_precision (Optional[str]): 'FP16' or 'FP32' to accelerate the ONNX model with TensorRT, or None.

    Raises:
        RuntimeError: If an exported model does not reproduce the eager outputs.
    """
    # PyTorch runs GPU convolutions in reduced-precision TF32 by default (about 3 significant digits instead of 7),
    # while ONNX Runtime computes the exported model in exact FP32, so the two differ by about 1e-3 on scores even
    # for a correct export. TF32 is switched off so that the reference is computed at full precision too.
    # The switch is global to the whole Python process: the previous values are saved here and restored at the end,
    # so that code calling this function does not silently keep slower full-precision GPU computations afterwards
    previous_cudnn_tf32, previous_matmul_tf32 = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = False, False
    try:
        device = torch.device("cuda")
        configuration_path = experiment_folder / "training_configuration.yaml"
        experiment_configuration, _ = load_experiment_configuration(configuration_path=configuration_path)
        checkpoint_path = get_checkpoint_path(experiment_folder=experiment_folder, iteration=iteration)
        model = load_trained_model(configuration_path=configuration_path, checkpoint_path=checkpoint_path,
                                   device=device)
        inference_model = build_inference_model(task=experiment_configuration["task"], model=model)
        specification = get_export_specification(task=experiment_configuration["task"],
                                                 inference_model=inference_model)
        print_color(color="blue", output=f"Exporting Checkpoint : file://{checkpoint_path.absolute()}",
                    add_separators=True)

        # Random example and check batches of different sizes, to confirm the batch dimension stays dynamic
        example_images = torch.rand(size=specification["example_shape"], device=device) * 255.0
        check_shape = [min(specification["check_shape"][0], maximum_batch_size)] + specification["check_shape"][1:]
        check_images = torch.rand(size=check_shape, device=device) * 255.0
        with torch.no_grad():
            reference_outputs = convert_outputs_to_numpy(outputs=inference_model(images=check_images))

        for export_format in formats:
            model_name = f"{specification['model_name']}_{export_format}"
            model_directory = repository / model_name

            # Export the model, then run the exported file on the check batch
            if export_format == "torchscript":
                model_path = model_directory / str(model_version) / "model.pt"
                export_torchscript(inference_model=inference_model, example_images=example_images,
                                   output_path=model_path)
                with torch.no_grad():
                    exported_outputs = convert_outputs_to_numpy(
                        outputs=torch.jit.load(str(model_path), map_location=device)(check_images))
            else:
                model_path = model_directory / str(model_version) / "model.onnx"
                export_onnx(inference_model=inference_model, example_images=example_images, output_path=model_path,
                            specification=specification, maximum_batch_size=maximum_batch_size)
                session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                exported_outputs = session.run(None, {"images": check_images.cpu().numpy()})

            write_triton_configuration(model_directory=model_directory, model_name=model_name,
                                       model_version=model_version, export_format=export_format,
                                       specification=specification, maximum_batch_size=maximum_batch_size,
                                       tensorrt_precision=tensorrt_precision)

            mismatches = specification["compare_outputs"](reference=reference_outputs, candidate=exported_outputs,
                                                          **specification["comparison_tolerances"])
            if mismatches:
                raise RuntimeError(f"The {export_format} Export Does Not Match The Eager Model:\n" +
                                   "\n".join(mismatches))
            print_color(color="green", output=f"{export_format} Export Matches The Eager Model : ", indent=1)
            print_color(color="green", output=f"file://{model_directory.absolute()}", indent=2)
    # 'finally' runs whether the export succeeds or raises an error (missing checkpoint, export mismatch...), so the
    # TF32 settings are always restored, and any error still reaches the caller afterwards
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = (previous_cudnn_tf32,
                                                                                   previous_matmul_tf32)

if __name__ == "__main__":
    # Trained experiment to export, holding 'Weights' and 'training_configuration.yaml'
    experiment_folder = Path(__file__).absolute().parents[2] / "experiments" / "poseidon_experiment_4"
    # Iteration to export, None for the last one recorded in the 'full-checkpoint' registry
    iteration = 6000
    # Export formats, among 'torchscript' and 'onnx'
    formats = ["torchscript", "onnx"]
    # Triton model repository, mounted by deployment/triton/docker-compose.yml
    repository = DEFAULT_REPOSITORY
    # Triton version of the exported models: they are written to '<model>/<version>/' and Triton serves this version
    model_version = 1
    # Largest batch size the exported models accept, and that Triton may send them
    maximum_batch_size = 16
    # TensorRT precision of the ONNX model ('FP16' or 'FP32'), None to run it with ONNX Runtime CUDA only
    tensorrt_precision = "FP32"
    # False hides the harmless library warnings printed during the export (see hide_known_export_warnings)
    show_warnings = False

    if not show_warnings:
        hide_known_export_warnings()

    export_experiment(experiment_folder=experiment_folder, iteration=iteration, formats=formats, repository=repository,
                      model_version=model_version, maximum_batch_size=maximum_batch_size,
                      tensorrt_precision=tensorrt_precision)
