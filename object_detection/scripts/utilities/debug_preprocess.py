import os
import cv2
from pathlib import Path
from utilities.data_utilities import preprocess_image_and_boxes


def debug_preprocessing() -> None:
    """
    Debug script to visualize the image and bounding box preprocessing transformations.
    
    This function reads a random image, applies a hardcoded bounding box in [x, y, w, h] format,
    runs the preprocessing pipeline to scale the image and bounding box, and displays the 
    original and transformed results in separate OpenCV windows.
    """
    image_path = os.path.join(str(Path(__file__).resolve().parent.parent.parent), "README", "random_image.png")

    # Read the image using OpenCV
    original_image = cv2.imread(filename=image_path)
    if original_image is None:
        print(f"Failed to load image at {image_path}")
        return

    # Hardcoded bounding box in [x_1, y_1, x_2, y_2] format
    original_bounding_boxes = [[86.0, 56.0, 650.0, 450.0]]

    # Preprocess the image and bounding boxes
    preprocessed_image, preprocessed_bounding_boxes = preprocess_image_and_boxes(
        image=original_image, bounding_boxes=original_bounding_boxes, image_size=1024, keep_ratio=True)

    # Draw bounding box on original image
    # Note: original box is [x_1, y_1, x_2, y_2]
    image_with_original_box = original_image.copy()
    box_x_1, box_y_1, box_x_2, box_y_2 = original_bounding_boxes[0]
    cv2.rectangle(img=image_with_original_box,
                  pt1=(int(box_x_1), int(box_y_1)), pt2=(int(box_x_2), int(box_y_2)), color=(0, 255, 0), thickness=2)

    # Draw bounding box on resized image
    # Note: preprocessed box is [x_1, y_1, x_2, y_2]
    image_with_preprocessed_box = preprocessed_image.copy()
    new_x_1, new_y_1, new_x_2, new_y_2 = preprocessed_bounding_boxes[0]
    cv2.rectangle(img=image_with_preprocessed_box,
                  pt1=(int(new_x_1), int(new_y_1)), pt2=(int(new_x_2), int(new_y_2)), color=(50, 200, 150), thickness=2)

    # Display the results using OpenCV windows
    cv2.imshow(winname="Original Image & Bounding Box", mat=image_with_original_box)
    cv2.imshow(winname="Preprocessed Image & Bounding Box", mat=image_with_preprocessed_box)

    print("Press any key to close the windows...")
    cv2.waitKey(delay=0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    debug_preprocessing()
