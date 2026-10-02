import csv
import cv2
import numpy as np
import torch

from pathlib import Path
from typing import Dict, List

# Side in pixels of a cell of the confusion matrix image, and width of the class name margins
CELL_SIZE = 70
NAME_MARGIN = 160


def draw_confusion_matrix(confusion_matrix: torch.Tensor, class_names: List[str]) -> np.ndarray:
    """
    Draws a confusion matrix as an image: every cell shows its count, coloured by its share of the samples of its true
    class (the recall on the diagonal), so that the classes confused with each other stand out.

    Args:
        confusion_matrix (torch.Tensor): The [K, K] confusion matrix (rows true classes, columns predicted classes).
        class_names (List[str]): The name of every class.

    Returns:
        np.ndarray: The BGR image of the matrix.
    """
    number_classes = len(class_names)
    image = np.full(shape=(NAME_MARGIN + number_classes * CELL_SIZE, NAME_MARGIN + number_classes * CELL_SIZE, 3),
                    fill_value=255, dtype=np.uint8)
    counts = confusion_matrix.to(dtype=torch.float64)
    row_shares = (counts / counts.sum(dim=1, keepdim=True).clamp(min=1.0)).numpy()

    # Class names: true classes on the rows, predicted classes on the columns (shortened to fit their cell)
    cv2.putText(img=image, text="true \\ predicted", org=(5, NAME_MARGIN - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.45, color=(0, 0, 0), thickness=1)
    for class_index, class_name in enumerate(class_names):
        position = NAME_MARGIN + class_index * CELL_SIZE
        cv2.putText(img=image, text=class_name[:16], org=(5, position + CELL_SIZE // 2 + 5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, color=(0, 0, 0), thickness=1)
        cv2.putText(img=image, text=class_name[:7], org=(position + 4, NAME_MARGIN - 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, color=(0, 0, 0), thickness=1)

    # Cells coloured from white (no sample) to dark blue (all the samples of the true class), with their count
    for true_index in range(number_classes):
        for predicted_index in range(number_classes):
            share = float(row_shares[true_index, predicted_index])
            colour = (255, int(255 * (1.0 - 0.8 * share)), int(255 * (1.0 - share)))
            top_left = (NAME_MARGIN + predicted_index * CELL_SIZE, NAME_MARGIN + true_index * CELL_SIZE)
            bottom_right = (top_left[0] + CELL_SIZE - 2, top_left[1] + CELL_SIZE - 2)
            cv2.rectangle(img=image, pt1=top_left, pt2=bottom_right, color=colour, thickness=-1)
            cv2.putText(img=image, text=str(int(counts[true_index, predicted_index])),
                        org=(top_left[0] + 8, top_left[1] + CELL_SIZE // 2 + 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(255, 255, 255) if share > 0.6 else (0, 0, 0), thickness=1)
    return image


def save_confusion_matrix(confusion_matrix: torch.Tensor, class_names: List[str],
                          output_folder: Path) -> Dict[str, Path]:
    """
    Saves the confusion matrix of an evaluated checkpoint as a CSV table and as an image, in the evaluation folder of
    the checkpoint, so that the confusions between classes can be inspected after the training.

    Args:
        confusion_matrix (torch.Tensor): The [K, K] confusion matrix (rows true classes, columns predicted classes).
        class_names (List[str]): The name of every class.
        output_folder (Path): The evaluation folder of the checkpoint, created if needed.

    Returns:
        Dict[str, Path]: The paths of the 'Confusion Matrix Table' and the 'Confusion Matrix Image'.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    # Table: one row per true class, one column per predicted class
    table_path = output_folder / "confusion_matrix.csv"
    with open(table_path, "w", newline="") as table_file:
        table_writer = csv.writer(table_file)
        table_writer.writerow(["true \\ predicted"] + class_names)
        for class_name, row in zip(class_names, confusion_matrix.tolist()):
            table_writer.writerow([class_name] + row)

    image_path = output_folder / "confusion_matrix.png"
    cv2.imwrite(filename=str(image_path), img=draw_confusion_matrix(confusion_matrix=confusion_matrix,
                                                                    class_names=class_names))
    return {"Confusion Matrix Table": table_path, "Confusion Matrix Image": image_path}
