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
    specific_configuration = detector_head_configurations["4"]
    feature_map_normalization = detector_configuration["normalization"]["feature_map_normalization"]

    input_image_size = 1024
    number_classes = 21  # Standard (20 objects + 1 background)
    input_channels = 64  # Output channels from the feature extraction layers
    batch_size = 2
    roi_align_pool_size = specific_configuration.get("roi_align_pool_size", 7)

    # Mocking the number of ROI proposals we feed into the detection head
    number_of_proposals_per_image = 500
    total_proposals = batch_size * number_of_proposals_per_image

    # Generate mock pooled features (simulating output of RoIAlign)
    mock_roi_features = torch.randn(size=(total_proposals, input_channels, roi_align_pool_size, roi_align_pool_size),
                                    device=device, dtype=dtype)

    # Instantiate the module
    console.print(Panel(renderable="[bold green]Instantiating DetectionHead Module...[/bold green]"))
    detection_head = DetectionHead(detector_configuration=specific_configuration,
                                   number_classes=number_classes,
                                   input_channels=input_channels,
                                   feature_map_normalization=feature_map_normalization,
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
    class_scores, box_regressions = detection_head(input_tensor=mock_roi_features)

    # Analyze outputs
    console.print("[bold blue]DetectionHead Outputs:[/bold blue]")
    console.print(f"  - Input RoI Features : {list(mock_roi_features.shape)}")
    console.print(f"  - Class Scores       : {list(class_scores.shape)}")
    console.print(f"  - Box Regressions    : {list(box_regressions.shape)}")


if __name__ == '__main__':
    debug_detection_head()
