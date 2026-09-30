import copy
import torch

from pathlib import Path
from typing import Any, Dict

from model.model import Model
from model.inference_model import InferenceDetector
from scripts.utilities.debugging_utilities import get_model_configuration
from utilities.model.model_utilities import postprocess_detections

DEVICE = torch.device(device="cpu")


def build_test_model() -> Model:
    """
    Builds a small untrained detector from the debug configuration, with 512 pixel inputs (the smallest size fitting
    its 9 pooling blocks) and a score threshold of zero so that the compared outputs always contain detections.

    Returns:
        Model: The detector in evaluation mode, on the CPU.
    """
    torch.manual_seed(seed=0)
    configuration, _, dtype = get_model_configuration()
    configuration: Dict[str, Any] = copy.deepcopy(configuration)
    configuration["Data"]["image_settings"]["size"] = 512
    configuration["Detector"]["inference"]["score_threshold"] = 0.0
    return Model(configuration=configuration, device=DEVICE, dtype=dtype).eval()


def test_postprocess_detections_threshold_nms_and_padding() -> None:
    """
    Verifies the final post-processing on hand-made predictions: the background class is ignored, low scores are
    removed, overlapping boxes of the same class are suppressed while other classes are kept, boxes are clipped to
    the image, padding proposals never produce detections, and the output is padded to a fixed size.
    """
    # One image, four proposals, classes (background, A, B); the last proposal is padding
    probabilities = torch.tensor(data=[[[0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.05, 0.1, 0.85], [0.0, 0.9, 0.1]]])
    first_box, shifted_box = [10.0, 10.0, 50.0, 50.0], [12.0, 12.0, 52.0, 52.0]
    far_box, padding_box = [100.0, 100.0, 150.0, 150.0], [0.0, 0.0, 40.0, 40.0]
    boxes = torch.tensor(data=[[[box] * 3 for box in [first_box, shifted_box, far_box, padding_box]]])

    detections = postprocess_detections(classification_scores=torch.log(input=probabilities), detection_boxes=boxes,
                                        valid_proposal_mask=torch.tensor(data=[[True, True, True, False]]),
                                        score_threshold=0.5, nms_iou_threshold=0.5, maximum_detections=4,
                                        input_image_size=128)

    assert detections["number_detections"].tolist() == [2]
    assert detections["final_detection_labels"].tolist() == [[2, 1, 0, 0]]
    assert torch.allclose(input=detections["final_detection_scores"], other=torch.tensor(data=[[0.85, 0.8, 0.0, 0.0]]))
    assert detections["final_detection_boxes"][0, :2].tolist() == [[100.0, 100.0, 128.0, 128.0], first_box]
    assert detections["final_detection_boxes"][0, 2:].abs().sum() == 0.0


def test_inference_detector_matches_the_eager_model() -> None:
    """
    Verifies that the export-friendly inference detector reproduces exactly the detections of the eager Model in
    evaluation mode, on the same weights and images.
    """
    model = build_test_model()
    detector = InferenceDetector(model=model).eval()
    images = torch.rand(size=(2, 3, 512, 512)) * 255.0

    with torch.no_grad():
        eager_outputs = model(input_tensor=images)
        boxes, scores, labels, numbers = detector(images=images)

    assert torch.equal(input=numbers, other=eager_outputs["number_detections"])
    assert torch.equal(input=labels, other=eager_outputs["final_detection_labels"])
    assert torch.allclose(input=scores, other=eager_outputs["final_detection_scores"])
    assert torch.allclose(input=boxes, other=eager_outputs["final_detection_boxes"])


def test_torchscript_round_trip(tmp_path: Path) -> None:
    """
    Verifies that the traced TorchScript detector, saved and reloaded like the Triton PyTorch backend does,
    reproduces the eager inference detector, including for a batch size different from the traced one.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    detector = InferenceDetector(model=build_test_model()).eval()
    model_path = tmp_path / "model.pt"

    with torch.no_grad():
        torch.jit.trace(detector, example_inputs=(torch.rand(size=(2, 3, 512, 512)) * 255.0,),
                        check_trace=False).save(str(model_path))
        images = torch.rand(size=(3, 3, 512, 512)) * 255.0
        reference_outputs = detector(images=images)
        traced_outputs = torch.jit.load(str(model_path))(images)

    for reference, traced in zip(reference_outputs, traced_outputs):
        assert torch.allclose(input=traced.to(dtype=reference.dtype), other=reference)
