"""
Decoding of the COCO 'segmentation' field of an annotation into a binary instance mask at the network resolution,
read by CocoDetectionDataset for the instance segmentation task: polygons are rasterised directly in the network input,
run length encodings (uncompressed or compressed) are decoded at the original resolution then resized and padded like
their image. The masks feed the mask targets of the instance segmenter and the mask evaluation.
"""

import cv2
import numpy as np

from typing import Any, Dict, List, Union

from vision_training.data.preprocessing import preprocess_boxes, preprocess_image


def fill_polygon_part(mask: np.ndarray, vertices: np.ndarray) -> None:
    """
    Sets to 1 the pixels of the mask whose centre lies inside one polygon part, so that a polygon part covers exactly
    the pixels of its area as rasterize_polygons needs (cv2.fillPoly also fills the pixels whose centre lies half a
    pixel outside the edges, one extra column and row for a square). Every pixel row is crossed by the edges of the part
    at its centre, the crossings are sorted along the row, and the pixels between every pair of crossings are filled
    (even-odd rule), all rows at once through a difference array.

    Args:
        mask (np.ndarray): The [S, S] uint8 mask, modified in place.
        vertices (np.ndarray): The [V, 2] x, y vertices of the part in mask pixels, whose integer coordinates are pixel
            edges.
    """
    mask_size = mask.shape[0]

    # Only the rows within the vertical span of the part can hold its pixels
    first_row = int(np.clip(np.floor(vertices[:, 1].min()), 0, mask_size))
    end_row = int(np.clip(np.ceil(vertices[:, 1].max()), 0, mask_size))
    if end_row <= first_row:
        return
    number_rows = end_row - first_row
    row_centres = np.arange(first_row, end_row, dtype=np.float64)[:, None] + 0.5
    start_points, end_points = vertices, np.roll(a=vertices, shift=-1, axis=0)

    # Every edge crosses the rows whose centre lies within its vertical span (lower end included, upper end excluded)
    lower_y = np.minimum(start_points[:, 1], end_points[:, 1])
    upper_y = np.maximum(start_points[:, 1], end_points[:, 1])
    crosses_row = (row_centres >= lower_y[None, :]) & (row_centres < upper_y[None, :])
    vertical_extent = np.where(end_points[:, 1] != start_points[:, 1], end_points[:, 1] - start_points[:, 1], 1.0)
    crossing_x = start_points[None, :, 0] + (row_centres - start_points[None, :, 1]) * (
        (end_points[:, 0] - start_points[:, 0]) / vertical_extent)[None, :]

    # Sort the crossings of every row, the edges missing the row pushed to the end
    crossing_x = np.sort(a=np.where(crosses_row, crossing_x, np.inf), axis=1)
    if crossing_x.shape[1] % 2 == 1:
        crossing_x = np.concatenate([crossing_x, np.full(shape=(number_rows, 1), fill_value=np.inf)], axis=1)
    span_starts, span_ends = crossing_x[:, 0::2], crossing_x[:, 1::2]

    # Pixel columns whose centre lies within [start, end) of every span, clipped to the mask
    valid_spans = np.isfinite(span_ends)
    first_columns = np.clip(np.ceil(span_starts - 0.5), 0, mask_size).astype(np.int64)
    end_columns = np.clip(np.ceil(np.where(valid_spans, span_ends, 0.0) - 0.5), 0, mask_size).astype(np.int64)
    row_indices = np.broadcast_to(np.arange(number_rows)[:, None], shape=span_starts.shape)
    keep = valid_spans & (end_columns > first_columns)

    # Mark the start and end of every span in a difference array, whose running sum along the row covers the spans
    differences = np.zeros(shape=(number_rows, mask_size + 1), dtype=np.int32)
    np.add.at(differences, (row_indices[keep], first_columns[keep]), 1)
    np.add.at(differences, (row_indices[keep], end_columns[keep]), -1)
    mask[first_row:end_row][np.cumsum(a=differences, axis=1)[:, :mask_size] > 0] = 1


def rasterize_polygons(polygons: List[List[float]], image_width: int, image_height: int, image_size: int,
                       keep_ratio: bool) -> np.ndarray:
    """
    Rasterises the polygon parts of one COCO object into its binary mask in the network input, the vertices being
    placed by preprocess_boxes exactly like the boxes of the image. Every part is filled on its own so that the mask is
    the union of the parts, as in COCO, a single even-odd fill over several parts cutting holes where they overlap. A
    pixel belongs to a part when its centre lies inside it.

    Args:
        polygons (List[List[float]]): The parts of the object, each a flat [x1, y1, x2, y2, ...] list of vertices in
            original image pixels, whose integer coordinates are pixel edges.
        image_width (int): The width of the original image.
        image_height (int): The height of the original image.
        image_size (int): The size of the square network input.
        keep_ratio (bool): Whether the image is letterboxed (aspect ratio kept) or stretched.

    Returns:
        np.ndarray: The [image_size, image_size] uint8 mask, 1 on the object.
    """
    mask = np.zeros(shape=(image_size, image_size), dtype=np.uint8)

    # A part needs 3 vertices to enclose any area, the shorter ones are skipped
    valid_polygons = [polygon for polygon in polygons if len(polygon) >= 6 and len(polygon) % 2 == 0]
    transformed_polygons = preprocess_boxes(bounding_boxes=valid_polygons, image_width=image_width,
                                            image_height=image_height, image_size=image_size, keep_ratio=keep_ratio)

    # Fill every part on its own into the same mask, giving the union of the parts
    for polygon in transformed_polygons:
        fill_polygon_part(mask=mask, vertices=np.array(object=polygon, dtype=np.float64).reshape(-1, 2))
    return mask


def decode_compressed_counts(compressed_counts: str) -> List[int]:
    """
    Decodes the string of a compressed COCO run length encoding into its list of run lengths, so that
    decode_run_length_encoding reads both encodings the same way: every run is written in groups of 5 bits (offset by
    48 into printable characters, the sixth bit flagging another group, the fifth bit of the last group the sign), and
    from the third run on as a difference with the run two places before.

    Args:
        compressed_counts (str): The 'counts' string of the encoding.

    Returns:
        List[int]: The run lengths, alternating background and object pixels, starting with background.
    """
    counts = []
    position = 0

    # Read one run per iteration, group by group
    while position < len(compressed_counts):
        value, shift, has_more_groups = 0, 0, True
        while has_more_groups:
            group = ord(compressed_counts[position]) - 48
            value |= (group & 0x1F) << (5 * shift)
            has_more_groups = bool(group & 0x20)
            position, shift = position + 1, shift + 1

            # Extend the sign bit of the last group into a negative value
            if not has_more_groups and group & 0x10:
                value |= -1 << (5 * shift)

        # From the third run on, the value is the difference with the run two places before
        if len(counts) > 2:
            value += counts[-2]
        counts.append(value)
    return counts


def decode_run_length_encoding(segmentation: Dict[str, Any]) -> np.ndarray:
    """
    Decodes a COCO run length encoding into its binary mask at the original image resolution, for the RLE annotations
    of COCO (crowds) and the visible parts of the synthetic instance segmentation objects.

    Args:
        segmentation (Dict[str, Any]): The encoding, with 'size' [height, width] and 'counts', a list of run lengths
            (uncompressed) or a string (compressed), the runs following the pixels column by column.

    Returns:
        np.ndarray: The [height, width] uint8 mask, 1 on the object.

    Raises:
        ValueError: If the runs do not cover the image exactly.
    """
    height, width = segmentation["size"]
    counts = segmentation["counts"]
    if isinstance(counts, str):
        counts = decode_compressed_counts(compressed_counts=counts)

    # The runs alternate between background (even positions) and object pixels (odd positions)
    if sum(counts) != height * width:
        raise ValueError(f"The run length encoding covers {sum(counts)} pixels instead of {height} x {width}.")
    run_values = np.arange(len(counts), dtype=np.int64) % 2
    flat_mask = np.repeat(a=run_values.astype(np.uint8), repeats=np.array(object=counts, dtype=np.int64))

    # The pixels are listed column by column (Fortran order)
    return np.ascontiguousarray(flat_mask.reshape((height, width), order="F"))


def encode_run_length_encoding(mask: np.ndarray) -> Dict[str, Any]:
    """
    Encodes a binary mask into an uncompressed COCO run length encoding, the inverse of decode_run_length_encoding,
    used by the synthetic instance segmentation generator to write the visible part of every object (holes included)
    in its labels.

    Args:
        mask (np.ndarray): The [height, width] mask, nonzero on the object.

    Returns:
        Dict[str, Any]: The encoding, with 'size' [height, width] and the 'counts' list of run lengths, alternating
            background and object pixels column by column, starting with background.
    """
    height, width = mask.shape
    flat_mask = (np.asarray(mask).reshape(-1, order="F") > 0).astype(np.int8)

    # Run boundaries where the value changes, a leading empty background run when the first pixel is on the object
    change_positions = np.flatnonzero(np.diff(flat_mask)) + 1
    boundaries = np.concatenate([[0], change_positions, [flat_mask.size]])
    counts = np.diff(boundaries).tolist()
    if flat_mask.size > 0 and flat_mask[0] == 1:
        counts = [0] + counts
    return {"size": [int(height), int(width)], "counts": [int(count) for count in counts]}


def build_instance_mask(segmentation: Union[List[List[float]], Dict[str, Any]], image_width: int, image_height: int,
                        image_size: int, keep_ratio: bool) -> np.ndarray:
    """
    Builds the binary mask of one COCO object at the network resolution, aligned with its preprocessed image, for the
    instance masks returned by CocoDetectionDataset: polygons are rasterised directly in the network input, run length
    encodings are decoded at the original resolution then resized with the nearest interpolation and padded with 0.

    Args:
        segmentation (Union[List[List[float]], Dict[str, Any]]): The 'segmentation' field of the annotation, a list of
            polygon parts or a run length encoding.
        image_width (int): The width of the original image.
        image_height (int): The height of the original image.
        image_size (int): The size of the square network input.
        keep_ratio (bool): Whether the image is letterboxed (aspect ratio kept) or stretched.

    Returns:
        np.ndarray: The [image_size, image_size] uint8 mask, 1 on the object.
    """
    # Polygon parts, placed like the boxes then filled at the network resolution
    if isinstance(segmentation, list):
        return rasterize_polygons(polygons=segmentation, image_width=image_width, image_height=image_height,
                                  image_size=image_size, keep_ratio=keep_ratio)

    # Run length encoding, decoded at the original resolution then resized like the segmentation masks
    original_mask = decode_run_length_encoding(segmentation=segmentation)
    return preprocess_image(image=original_mask, image_size=image_size, keep_ratio=keep_ratio,
                            interpolation=cv2.INTER_NEAREST, padding_value=0)
