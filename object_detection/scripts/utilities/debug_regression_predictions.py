import cv2
import numpy as np
import torch

from utilities.model.model_utilities import apply_regression_predictions, add_bounding_box
from utilities.os_utilities import print_green, print_blue, get_random_color


def debug_regression_predictions() -> None:
    """
    Visually debugs the bounding box regression logic.
    
    This script generates sample bounding boxes and dummy regression predictions for multiple classes,
    applies the predictions using `apply_regression_predictions`, and displays the original anchors
    (green) alongside their class-specific predictions (random colors) to verify that the regression 
    math correctly handles multi-class outputs per anchor.
    
    Args:
        None
    """
    input_image_size = 1024
    number_of_boxes = 5
    number_classes = 2

    # Create a completely black background image
    canvas = np.zeros(shape=(input_image_size, input_image_size, 3), dtype=np.uint8)

    # 1. Generate number_of_boxes dummy bounding boxes in [x_min, y_min, x_max, y_max] format
    # Create boxes of size 50x50 spread out diagonally for clear visibility
    boxes = []
    for index in range(number_of_boxes):
        offset = index * 45
        x_min = 20 + offset
        y_min = 20 + offset
        x_max = x_min + 50
        y_max = y_min + 50
        boxes.append([x_min, y_min, x_max, y_max])

    boxes_tensor = torch.tensor(data=boxes, dtype=torch.float32)

    # 2. Generate dummy regression predictions of shape (N, k, 4)
    # dx, dy: shift the center right and down (positive values)
    # dw, dh: slightly expand the boxes (positive values)
    regression_predictions = torch.ones(size=(number_of_boxes, number_classes, 4), dtype=torch.float32) * 0.1

    # Add some randomness to each prediction so they don't all look exactly identical
    regression_predictions += torch.randn(size=(number_of_boxes, number_classes, 4)) * 0.1

    # 3. Apply the regressions
    predicted_boxes_tensor = apply_regression_predictions(
        regression_predictions=regression_predictions,
        boxes=boxes_tensor)

    # Convert to numpy arrays for OpenCV compatibility
    original_boxes_array = boxes_tensor.numpy().astype(dtype=np.int32)
    predicted_boxes_array = predicted_boxes_tensor.numpy().astype(dtype=np.int32)

    # Generate a unique random color for each class
    class_colors = [get_random_color() for _ in range(number_classes)]

    # 4. Visualize the results
    print_blue(output="Visualizing regression predictions...", add_separators=True)
    print_green(output="- GREEN boxes: Original Reference Anchors", add_separators=False)
    print_blue(output="- RANDOM colors: Predicted/Regressed Boxes for each class", add_separators=False)

    for original_box, predicted_classes_boxes in zip(original_boxes_array, predicted_boxes_array):
        # Draw Original Anchor Box (GREEN) using the utility function
        canvas = add_bounding_box(
            bounding_box=original_box, 
            image=canvas, 
            input_image_size=input_image_size, 
            color=(0, 255, 0)
        )

        # Draw Predicted Boxes for each class in its respective random color
        for class_index in range(number_classes):
            predicted_box = predicted_classes_boxes[class_index]
            canvas = add_bounding_box(
                bounding_box=predicted_box, 
                image=canvas, 
                input_image_size=input_image_size, 
                color=class_colors[class_index]
            )

    window_name = "Regression Predictions Debugger"
    cv2.imshow(winname=window_name, mat=canvas)

    print_blue(output="Press any key on the image window to close it.", add_separators=True)
    cv2.waitKey(delay=0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    debug_regression_predictions()
