import logging
import warnings
import numpy as np
import onnxruntime
import torch

from pathlib import Path
from typing import List, Optional, Sequence

from model.inference_model import InferenceDetector, get_checkpoint_path, load_trained_model
from utilities.os_utilities import print_color

# Default Triton model repository, mounted by deployment/triton/docker-compose.yml
DEFAULT_REPOSITORY = Path(__file__).absolute().parent / "triton" / "served_models"

# Names of the model outputs, in the order returned by InferenceDetector
OUTPUT_NAMES = ["detection_boxes", "detection_scores", "detection_labels", "number_detections"]

# Triton model names of every export format
MODEL_NAMES = {"torchscript": "object_detector_torchscript", "onnx": "object_detector_onnx"}


def hide_known_export_warnings() -> None:
    """
    Hides the warnings printed by PyTorch, torchvision and the ONNX library during the export, which come from inside
    these libraries and cannot be fixed in the project code. They are harmless for this export: the deprecation
    notices of TorchScript and of a torchvision helper, the tracer notice raised inside torchvision's RoIAlign, and
    the ONNX notice about an empty attribute defaulting to integers. The exports are checked against the eager model
    anyway.
    """
    # Deprecation notices of torch.jit.trace / trace_method / load and of a torchvision export helper
    warnings.filterwarnings(action="ignore", message=r"`torch\.jit\.", category=FutureWarning)
    warnings.filterwarnings(action="ignore", message=r"`create_unbacked_symint` is deprecated", category=FutureWarning)

    # Tracer notice raised by the shape assertion inside torchvision's RoIAlign
    warnings.filterwarnings(action="ignore", message=r"Converting a tensor to a Python boolean",
                            category=torch.jit.TracerWarning)

    # The ONNX library reports the empty attribute through the logging module rather than the warnings module
    logging.getLogger(name="onnx_ir._convenience").setLevel(level=logging.ERROR)


def export_torchscript(detector: InferenceDetector, example_images: torch.Tensor, output_path: Path) -> None:
    """
    Traces the inference detector into a self-contained TorchScript file served by Triton's PyTorch (libtorch)
    backend. The traced graph keeps the batch dimension dynamic.

    Args:
        detector (InferenceDetector): The inference detector to export, in evaluation mode.
        example_images (torch.Tensor): An example [B, 3, H, W] batch used to trace the graph.
        output_path (Path): The path of the written 'model.pt' file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        traced_detector = torch.jit.trace(detector, example_inputs=(example_images,), check_trace=False)
    traced_detector.save(str(output_path))


def export_onnx(detector: InferenceDetector, example_images: torch.Tensor, output_path: Path,
                maximum_batch_size: int) -> None:
    """
    Exports the inference detector to an ONNX file served by Triton's ONNX Runtime backend (optionally accelerated
    by TensorRT), with a dynamic batch dimension up to the maximum batch size.

    Args:
        detector (InferenceDetector): The inference detector to export, in evaluation mode.
        example_images (torch.Tensor): An example [B, 3, H, W] batch used to export the graph.
        output_path (Path): The path of the written 'model.onnx' file.
        maximum_batch_size (int): The largest batch size the exported model accepts.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch_dimension = torch.export.Dim("batch", min=1, max=maximum_batch_size)
    torch.onnx.export(detector, args=(example_images,), f=str(output_path), input_names=["images"],
                      output_names=OUTPUT_NAMES, dynamic_shapes={"images": {0: batch_dimension}}, opset_version=18,
                      dynamo=True, verbose=False)


def write_triton_configuration(model_directory: Path, model_name: str, model_version: int, export_format: str,
                               image_size: int, maximum_detections: int, maximum_batch_size: int,
                               tensorrt_precision: Optional[str]) -> None:
    """
    Writes the 'config.pbtxt' describing an exported detector to Triton: platform, input and output tensors,
    dynamic batching, GPU execution and, for ONNX, the optional TensorRT accelerator.

    Args:
        model_directory (Path): The model folder of the Triton repository (holding the version folder).
        model_name (str): The Triton model name.
        model_version (int): The version served by Triton, which is also the name of the folder holding the model file.
        export_format (str): Either 'torchscript' or 'onnx'.
        image_size (int): The size of the square network input.
        maximum_detections (int): The fixed number of detections returned per image.
        maximum_batch_size (int): The largest batch size Triton may send to the model.
        tensorrt_precision (Optional[str]): 'FP16' or 'FP32' to accelerate the ONNX model with TensorRT, or None.
    """
    # TorchScript outputs follow the '<name>__<index>' naming convention of the Triton PyTorch backend
    if export_format == "torchscript":
        platform = "pytorch_libtorch"
        output_names = [f"{name}__{index}" for index, name in enumerate(OUTPUT_NAMES)]
    else:
        platform = "onnxruntime_onnx"
        output_names = OUTPUT_NAMES

    output_shapes = [(f"[ {maximum_detections}, 4 ]", "TYPE_FP32"), (f"[ {maximum_detections} ]", "TYPE_FP32"),
                     (f"[ {maximum_detections} ]", "TYPE_INT64"), ("[ 1 ]", "TYPE_INT64")]
    outputs = []
    for output_name, (dimensions, data_type) in zip(output_names, output_shapes):
        # The number of detections is one value per image, exposed as [1] and reshaped to a scalar for the model
        reshape = " reshape: { shape: [ ] }" if dimensions == "[ 1 ]" else ""
        outputs.append(f'  {{ name: "{output_name}" data_type: {data_type} dims: {dimensions}{reshape} }}')

    configuration = (f'name: "{model_name}"\n'
                     f'platform: "{platform}"\n'
                     f'max_batch_size: {maximum_batch_size}\n'
                     f'input [\n  {{ name: "images" data_type: TYPE_FP32 '
                     f'dims: [ 3, {image_size}, {image_size} ] }}\n]\n'
                     f'output [\n' + ",\n".join(outputs) + '\n]\n'
                     f'dynamic_batching {{ max_queue_delay_microseconds: 1000 }}\n'
                     f'instance_group [ {{ kind: KIND_GPU count: 1 }} ]\n'
                     f'version_policy: {{ specific: {{ versions: [ {model_version} ] }} }}\n')

    # Let ONNX Runtime run the supported parts of the graph with TensorRT, the other operators stay on CUDA
    if export_format == "onnx" and tensorrt_precision is not None:
        configuration += ('optimization { execution_accelerators { gpu_execution_accelerator: [ {\n'
                          '  name: "tensorrt"\n'
                          f'  parameters {{ key: "precision_mode" value: "{tensorrt_precision}" }}\n'
                          '  parameters { key: "max_workspace_size_bytes" value: "4294967296" }\n'
                          '  parameters { key: "trt_engine_cache_enable" value: "1" }\n'
                          '  parameters { key: "trt_engine_cache_path" value: "/tmp/tensorrt_engine_cache" }\n'
                          '} ] } }\n')

    (model_directory / "config.pbtxt").write_text(configuration)


def sort_detections(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Orders the detections of one image canonically (by score, then label and box coordinates) so that two exports
    can be compared even when detections with exactly the same score come out in a different order.

    Args:
        boxes (np.ndarray): The [K, 4] detection boxes.
        scores (np.ndarray): The [K] detection scores.
        labels (np.ndarray): The [K] detection labels.

    Returns:
        np.ndarray: The [K, 6] detections as (score, label, x1, y1, x2, y2) rows in canonical order.
    """
    detections = np.concatenate([scores[:, None], labels[:, None].astype(np.float64), boxes], axis=1)
    order = np.lexsort(keys=tuple(detections[:, column] for column in reversed(range(detections.shape[1]))))
    return detections[order]


def compare_detections(reference: Sequence[np.ndarray], candidate: Sequence[np.ndarray], box_tolerance: float,
                       score_tolerance: float) -> List[str]:
    """
    Compares the outputs of an exported model with the eager inference detector, image by image, independently of
    the order of detections with tied scores.

    Args:
        reference (Sequence[np.ndarray]): The eager boxes, scores, labels and numbers of detections.
        candidate (Sequence[np.ndarray]): The exported model outputs, in the same order.
        box_tolerance (float): The largest accepted box coordinate difference in pixels.
        score_tolerance (float): The largest accepted score difference.

    Returns:
        List[str]: A description of every mismatch, empty when the outputs agree.
    """
    mismatches = []
    reference_boxes, reference_scores, reference_labels, reference_numbers = reference
    candidate_boxes, candidate_scores, candidate_labels, candidate_numbers = candidate

    if not np.array_equal(reference_numbers.reshape(-1), candidate_numbers.reshape(-1)):
        mismatches.append(f"numbers of detections differ: {reference_numbers.tolist()} vs {candidate_numbers.tolist()}")

    # Compare the canonically ordered detections of every image
    for image_index in range(reference_boxes.shape[0]):
        reference_detections = sort_detections(boxes=reference_boxes[image_index], scores=reference_scores[image_index],
                                               labels=reference_labels[image_index])
        candidate_detections = sort_detections(boxes=candidate_boxes[image_index], scores=candidate_scores[image_index],
                                               labels=candidate_labels[image_index])
        score_difference = np.abs(reference_detections[:, 0] - candidate_detections[:, 0]).max()
        box_difference = np.abs(reference_detections[:, 2:] - candidate_detections[:, 2:]).max()
        labels_equal = np.array_equal(reference_detections[:, 1], candidate_detections[:, 1])
        if score_difference > score_tolerance or box_difference > box_tolerance or not labels_equal:
            mismatches.append(f"image {image_index}: score difference {score_difference:.2e}, box difference "
                              f"{box_difference:.2e} px, labels equal {labels_equal}")
    return mismatches


def export_experiment(experiment_folder: Path, iteration: Optional[int], formats: List[str], repository: Path,
                      model_version: int, maximum_batch_size: int, tensorrt_precision: Optional[str]) -> None:
    """
    Exports a trained experiment to the Triton model repository in every requested format, writes the Triton
    configurations, and checks each export against the eager inference detector on a random batch.

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
        checkpoint_path = get_checkpoint_path(experiment_folder=experiment_folder, iteration=iteration)
        model = load_trained_model(configuration_path=experiment_folder / "training_configuration.yaml",
                                   checkpoint_path=checkpoint_path, device=device)
        detector = InferenceDetector(model=model).eval()
        print_color(color="blue", output=f"Exporting Checkpoint : file://{checkpoint_path.absolute()}",
                    add_separators=True)

        # Random example and check batches of different sizes, to confirm the batch dimension stays dynamic
        image_size = detector.input_image_size
        example_images = torch.rand(size=(2, 3, image_size, image_size), device=device) * 255.0
        check_images = torch.rand(size=(min(3, maximum_batch_size), 3, image_size, image_size), device=device) * 255.0
        with torch.no_grad():
            reference_outputs = [tensor.cpu().numpy() for tensor in detector(images=check_images)]

        for export_format in formats:
            model_directory = repository / MODEL_NAMES[export_format]

            # Export the model, then run the exported file on the check batch
            if export_format == "torchscript":
                model_path = model_directory / str(model_version) / "model.pt"
                export_torchscript(detector=detector, example_images=example_images, output_path=model_path)
                with torch.no_grad():
                    exported_outputs = [tensor.cpu().numpy() for tensor in
                                        torch.jit.load(str(model_path), map_location=device)(check_images)]
            else:
                model_path = model_directory / str(model_version) / "model.onnx"
                export_onnx(detector=detector, example_images=example_images, output_path=model_path,
                            maximum_batch_size=maximum_batch_size)
                session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                exported_outputs = session.run(None, {"images": check_images.cpu().numpy()})

            write_triton_configuration(model_directory=model_directory, model_name=MODEL_NAMES[export_format],
                                       model_version=model_version, export_format=export_format, image_size=image_size,
                                       maximum_detections=detector.maximum_detections,
                                       maximum_batch_size=maximum_batch_size, tensorrt_precision=tensorrt_precision)

            mismatches = compare_detections(reference=reference_outputs, candidate=exported_outputs, box_tolerance=1e-2,
                                            score_tolerance=1e-4)
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
