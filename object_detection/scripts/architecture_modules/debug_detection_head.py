import torch

from rich.console import Console
from rich.panel import Panel

from architecture_modules.detector import DetectionHead
from scripts.utilities.debugging_utilities import (get_model_configuration,
                                                   get_dummy_ground_truth_boxes,
                                                   get_dummy_ground_truth_labels)
from utilities.tensor_utilities import print_tensor_list


def debug_detection_head() -> None:
    """
    Instantiates the Detection Head and runs a forward pass using mock feature maps and labels.
    """
    console = Console()

    # Fetch parameters
    configuration, device, dtype = get_model_configuration()

    # Extract configuration specifically for scale '4' as an example
    detector_configuration = configuration.get("Detector")
    detector_head_configurations = detector_configuration["head_configurations"]
    specific_configuration = detector_head_configurations["5"]
    feature_map_normalization = detector_configuration["normalization"]["feature_map_normalization"]

    input_image_size = 1024
    feature_map_size = 32  # Block 5 corresponds to 32x32 feature map
    number_classes = 21  # Standard (20 objects + 1 background)
    input_channels = 64  # Output channels from the feature extraction layers
    batch_size = 2

    # Mocking the number of ROI proposals we feed into the detection head
    number_of_proposals_per_image = 128
    total_proposals = batch_size * number_of_proposals_per_image

    # Generate a raw backbone feature map instead of pre-pooled ROIs
    mock_feature_map = torch.randn(size=(batch_size, input_channels, feature_map_size, feature_map_size),
                                   device=device, dtype=dtype)

    # Generate the mock [K, 5] proposal boxes tensor
    batch_indices = torch.arange(start=0, end=batch_size, device=device, dtype=dtype).view(-1, 1)
    batch_indices = batch_indices.expand(batch_size, number_of_proposals_per_image).reshape(-1, 1)

    boxes = torch.rand(size=(total_proposals, 4), device=device, dtype=dtype) * (input_image_size / 2)
    boxes[:, 2:] += boxes[:, :2]  # Ensure x2 > x1 and y2 > y1
    mock_proposal_boxes = torch.cat([batch_indices, boxes], dim=-1)

    # Instantiate the module
    console.print(Panel(renderable="[bold green]Instantiating DetectionHead Module...[/bold green]"))
    detection_head = DetectionHead(detector_configuration=specific_configuration,
                                   number_classes=number_classes,
                                   input_channels=input_channels,
                                   mode="training",
                                   feature_map_normalization=feature_map_normalization,
                                   input_image_size=input_image_size,
                                   feature_map_size=feature_map_size,
                                   dtype=dtype,
                                   device=device)

    # Get mock ground truth boxes and labels to validate our new utility function
    mock_ground_truth_boxes = get_dummy_ground_truth_boxes(batch_size=batch_size,
                                                           input_image_size=input_image_size,
                                                           configuration=configuration,
                                                           device=device, dtype=dtype)

    mock_ground_truth_labels = get_dummy_ground_truth_labels(ground_truth_boxes=mock_ground_truth_boxes,
                                                             number_classes=number_classes,
                                                             device=device, dtype=dtype)

    console.print("[bold blue]Generated Ground Truths:[/bold blue]")
    for index, (boxes, labels) in enumerate(zip(mock_ground_truth_boxes, mock_ground_truth_labels)):
        console.print(f"  - Image {index}: {boxes.shape[0]} boxes, {labels.shape[0]} labels.")

    # Forward Pass
    console.print(Panel(renderable="[bold green]Running Forward Pass...[/bold green]"))
    class_scores, box_regressions = detection_head(proposal_boxes=mock_proposal_boxes,
                                                   input_tensor=mock_feature_map,
                                                   tensor_to_concatenate=None)

    # Analyze outputs
    console.print("[bold blue]DetectionHead Outputs:[/bold blue]")
    console.print(f"  - Input Feature Map  : {list(mock_feature_map.shape)}")
    console.print(f"  - Input Proposals    : {list(mock_proposal_boxes.shape)}")
    console.print(f"  - Class Scores       : {list(class_scores.shape)}")
    console.print(f"  - Box Regressions    : {list(box_regressions.shape)}")


if __name__ == '__main__':
    debug_detection_head()
