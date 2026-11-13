"""
Decoding of the COCO 'segmentation' field of an annotation into a binary instance mask at the network resolution or
at the original image resolution, read by CocoDetectionDataset for the instance segmentation task: polygons are
rasterised directly in the network input, run length encodings (uncompressed or compressed) are decoded at the
original resolution then resized and padded like their image. The masks feed the mask targets of the instance
segmenter and the mask evaluation, which can also read the masks at the original image resolution
(evaluation.original_resolution). The boxes of the instance segmentation come from the masks too: from the
segmentation field without decoding it (selection of the objects, anchor recall) and from the masks at the network
resolution (loader, geometric augmentations). The encoding functions write the visible parts of the synthetic instance
segmentation objects, compressed so that the labels of the large hard datasets fit in memory.
"""

import cv2
import numpy as np
import torch

from typing import Any, Dict, List, Optional, Union

from vision_training.data.preprocessing import preprocess_boxes, preprocess_image


def fill_polygon_part(mask: np.ndarray, vertices: np.ndarray) -> None:
    """
    Sets to 1 the pixels of the mask whose centre lies inside one polygon part, so that a polygon part covers exactly
    the pixels of its area as rasterize_polygons needs (cv2.fillPoly also fills the pixels whose centre lies half a
    pixel outside the edges, one extra column and row for a square). Every pixel row is crossed by the edges of the part
    at its centre, the crossings are sorted along the row, and the pixels between every pair of crossings are filled
    (even-odd rule), all rows at once through a difference array.

    Args:
        mask (np.ndarray): The [H, W] uint8 mask, modified in place.
        vertices (np.ndarray): The [V, 2] x, y vertices of the part in mask pixels, whose integer coordinates are pixel
            edges.
    """
    mask_height, mask_width = mask.shape

    # Only the rows within the vertical span of the part can hold its pixels
    first_row = int(np.clip(np.floor(vertices[:, 1].min()), 0, mask_height))
    end_row = int(np.clip(np.ceil(vertices[:, 1].max()), 0, mask_height))
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
    first_columns = np.clip(np.ceil(span_starts - 0.5), 0, mask_width).astype(np.int64)
    end_columns = np.clip(np.ceil(np.where(valid_spans, span_ends, 0.0) - 0.5), 0, mask_width).astype(np.int64)
    row_indices = np.broadcast_to(np.arange(number_rows)[:, None], shape=span_starts.shape)
    keep = valid_spans & (end_columns > first_columns)

    # Mark the start and end of every span in a difference array, whose running sum along the row covers the spans
    differences = np.zeros(shape=(number_rows, mask_width + 1), dtype=np.int32)
    np.add.at(differences, (row_indices[keep], first_columns[keep]), 1)
    np.add.at(differences, (row_indices[keep], end_columns[keep]), -1)
    mask[first_row:end_row][np.cumsum(a=differences, axis=1)[:, :mask_width] > 0] = 1


def rasterize_polygons(polygons: List[List[float]], image_width: int, image_height: int, image_size: Optional[int],
                       keep_ratio: bool) -> np.ndarray:
    """
    Rasterises the polygon parts of one COCO object into its binary mask in the network input, the vertices being
    placed by preprocess_boxes exactly like the boxes of the image, or at the original image resolution for the mask
    evaluation at that resolution, the vertices then being kept as they are. Every part is filled on its own so that
    the mask is the union of the parts, as in COCO, a single even-odd fill over several parts cutting holes where they
    overlap. A pixel belongs to a part when its centre lies inside it.

    Args:
        polygons (List[List[float]]): The parts of the object, each a flat [x1, y1, x2, y2, ...] list of vertices in
            original image pixels, whose integer coordinates are pixel edges.
        image_width (int): The width of the original image.
        image_height (int): The height of the original image.
        image_size (Optional[int]): The size of the square network input, None for the original image resolution.
        keep_ratio (bool): Whether the image is letterboxed (aspect ratio kept) or stretched, unused at the original
            resolution.

    Returns:
        np.ndarray: The [image_size, image_size] uint8 mask, or [image_height, image_width] at the original
            resolution, 1 on the object.
    """
    # A part needs 3 vertices to enclose any area, the shorter ones are skipped
    valid_polygons = [polygon for polygon in polygons if len(polygon) >= 6 and len(polygon) % 2 == 0]

    # Vertices kept in original pixels at the original resolution, else placed in the network input like the boxes
    if image_size is None:
        mask = np.zeros(shape=(image_height, image_width), dtype=np.uint8)
        transformed_polygons = valid_polygons
    else:
        mask = np.zeros(shape=(image_size, image_size), dtype=np.uint8)
        transformed_polygons = preprocess_boxes(bounding_boxes=valid_polygons, image_width=image_width,
                                                image_height=image_height, image_size=image_size,
                                                keep_ratio=keep_ratio)

    # Fill every part on its own into the same mask, giving the union of the parts
    for polygon in transformed_polygons:
        fill_polygon_part(mask=mask, vertices=np.array(object=polygon, dtype=np.float64).reshape(-1, 2))
    return mask


def decode_compressed_counts(compressed_counts: str) -> np.ndarray:
    """
    Decodes the string of a compressed COCO run length encoding into its run lengths, so that
    decode_run_length_encoding reads both encodings the same way: every run is written in groups of 5 bits (offset by
    48 into printable characters, the sixth bit flagging another group, the fifth bit of the last group the sign), and
    from the fourth run on as a difference with the run two places before. The groups are decoded with numpy at once,
    since the training loader of the instance segmentation decodes every mask of every batch.

    Args:
        compressed_counts (str): The 'counts' string of the encoding.

    Returns:
        np.ndarray: The int64 run lengths, alternating background and object pixels, starting with background.
    """
    # Groups of 5 bits, the sixth bit telling that another group of the same run follows
    groups = np.frombuffer(buffer=compressed_counts.encode("ascii"), dtype=np.uint8).astype(np.int64) - 48
    if groups.size == 0:
        return np.zeros(shape=(0,), dtype=np.int64)
    is_last_group = (groups & 0x20) == 0
    run_indices = np.concatenate(([0], np.cumsum(a=is_last_group)[:-1]))

    # Position of every group within its run, giving the shift of its 5 bits
    run_starts = np.flatnonzero(np.concatenate(([True], is_last_group[:-1])))
    group_positions = np.arange(groups.size) - run_starts[run_indices]
    values = np.zeros(shape=(run_starts.size,), dtype=np.int64)
    np.add.at(values, run_indices, (groups & 0x1F) << (5 * group_positions))

    # Extend the sign bit of the last group of every run into a negative value
    run_lengths_in_groups = np.diff(np.concatenate((run_starts, [groups.size])))
    is_negative = (groups[is_last_group] & 0x10) != 0
    values[is_negative] |= -1 << (5 * run_lengths_in_groups[is_negative])

    # From the fourth run on (index 3), the value is the difference with the run two places before: a cumulative
    # sum within the odd runs from run 1 and within the even runs from run 2
    counts = values.copy()
    counts[1::2] = np.cumsum(a=values[1::2])
    counts[2::2] = np.cumsum(a=values[2::2])
    return counts


def decode_run_length_encoding(segmentation: Dict[str, Any]) -> np.ndarray:
    """
    Decodes a COCO run length encoding into its binary mask at the original image resolution, for the RLE annotations
    of COCO (mostly its crowd annotations) and the visible parts of the synthetic instance segmentation objects.

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
    counts = np.asarray(a=counts, dtype=np.int64)
    if int(counts.sum()) != height * width:
        raise ValueError(f"The run length encoding covers {int(counts.sum())} pixels instead of {height} x {width}.")
    run_values = np.arange(len(counts), dtype=np.int64) % 2
    flat_mask = np.repeat(a=run_values.astype(np.uint8), repeats=np.array(object=counts, dtype=np.int64))

    # The pixels are listed column by column: rows of the [width, height] transpose, turned back by the OpenCV
    # transpose, about ten times faster than the copy of a Fortran ordered numpy view
    return cv2.transpose(src=flat_mask.reshape(width, height))


def compress_counts(counts: List[int]) -> str:
    """
    Writes run lengths as the string of a compressed COCO run length encoding, the inverse of decode_compressed_counts,
    so that the synthetic instance segmentation labels take about a third of the disk space and a seventh of the
    memory of the uncompressed list once loaded by CocoDetectionDataset (the hard 1024 pixel datasets).

    Args:
        counts (List[int]): The run lengths, alternating background and object pixels.

    Returns:
        str: The compressed 'counts' string.
    """
    characters = []
    for count_index, count in enumerate(counts):
        # From the third run on, the difference with the run two places before is written
        value = count - counts[count_index - 2] if count_index > 2 else count

        # Groups of 5 bits, the sixth bit flagging another group, the fifth bit of the last group the sign
        has_more_groups = True
        while has_more_groups:
            group = value & 0x1F
            value >>= 5
            has_more_groups = value != -1 if group & 0x10 else value != 0
            characters.append(chr((group | 0x20 if has_more_groups else group) + 48))
    return "".join(characters)


def encode_run_length_encoding(mask: np.ndarray, compressed: bool = False) -> Dict[str, Any]:
    """
    Encodes a binary mask into a COCO run length encoding, the inverse of decode_run_length_encoding, used by the
    synthetic instance segmentation generator to write the visible part of every object (holes included) in its labels,
    compressed there, and by the tests to build masks of known pixels.

    Args:
        mask (np.ndarray): The [height, width] mask, nonzero on the object.
        compressed (bool): Whether the 'counts' are written as the compressed COCO string (compress_counts) instead of
            the list of run lengths. Defaults to False.

    Returns:
        Dict[str, Any]: The encoding, with 'size' [height, width] and the 'counts' run lengths, alternating background
            and object pixels column by column, starting with background, as a list or as a compressed string.
    """
    height, width = mask.shape
    flat_mask = (np.asarray(mask).reshape(-1, order="F") > 0).astype(np.int8)

    # Run boundaries where the value changes, a leading empty background run when the first pixel is on the object
    change_positions = np.flatnonzero(np.diff(flat_mask)) + 1
    boundaries = np.concatenate([[0], change_positions, [flat_mask.size]])
    counts = np.diff(boundaries).tolist()
    if flat_mask.size > 0 and flat_mask[0] == 1:
        counts = [0] + counts
    counts = [int(count) for count in counts]
    return {"size": [int(height), int(width)], "counts": compress_counts(counts=counts) if compressed else counts}


def compute_segmentation_bounding_box(segmentation: Union[List[List[float]], Dict[str, Any]]) -> List[float]:
    """
    Gives the box of one COCO object from its 'segmentation' field, so that the instance segmentation needs no 'bbox'
    in its annotations: select_image_objects reads it for the loader and for the anchor recall of the trainer at
    launch, which measures the training objects without loading any image, so the masks are never decoded here.
    Polygons give the extent of the vertices of their valid parts (at least 3 vertices); a run length encoding gives
    the extent of its object runs, read from their positions in the column by column pixel order.

    Args:
        segmentation (Union[List[List[float]], Dict[str, Any]]): The 'segmentation' field of the annotation, a list of
            polygon parts or a run length encoding (uncompressed or compressed).

    Returns:
        List[float]: The [x1, y1, x2, y2] box in original image pixels, x2 and y2 one past the last pixel of a run
            length encoding, [0, 0, 0, 0] for an object without any valid part or object pixel.
    """
    # Polygon parts: the extent of the vertices of the parts enclosing an area, whose coordinates are pixel edges
    if isinstance(segmentation, list):
        valid_polygons = [polygon for polygon in segmentation if len(polygon) >= 6 and len(polygon) % 2 == 0]
        if len(valid_polygons) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        vertices = np.concatenate([np.asarray(a=polygon, dtype=np.float64).reshape(-1, 2)
                                   for polygon in valid_polygons])
        return [float(vertices[:, 0].min()), float(vertices[:, 1].min()), float(vertices[:, 0].max()),
                float(vertices[:, 1].max())]

    # Run length encoding: the runs alternate background (even positions) and object pixels (odd positions)
    height = int(segmentation["size"][0])
    counts = segmentation["counts"]
    if isinstance(counts, str):
        counts = decode_compressed_counts(compressed_counts=counts)
    counts = np.asarray(a=counts, dtype=np.int64)
    run_ends = np.cumsum(a=counts)
    object_runs = (np.arange(stop=len(counts)) % 2 == 1) & (counts > 0)
    if not object_runs.any():
        return [0.0, 0.0, 0.0, 0.0]

    # First and last pixel of every object run, as column and row of the column by column order
    first_pixels = run_ends[object_runs] - counts[object_runs]
    last_pixels = run_ends[object_runs] - 1
    first_columns, first_rows = first_pixels // height, first_pixels % height
    last_columns, last_rows = last_pixels // height, last_pixels % height

    # A run within one column spans its own rows, a run crossing to the next column reaches the top and bottom rows
    within_column = first_columns == last_columns
    top_rows = np.where(within_column, first_rows, 0)
    bottom_rows = np.where(within_column, last_rows, height - 1)
    return [float(first_columns.min()), float(top_rows.min()), float(last_columns.max() + 1),
            float(bottom_rows.max() + 1)]


def compute_mask_bounding_boxes(instance_masks: torch.Tensor) -> torch.Tensor:
    """
    Gives the tight box of every instance mask, the box the instance segmentation learns and is evaluated on, at the
    network resolution or, for the evaluation at the original resolution, at that resolution: CocoDetectionDataset
    takes the box of every object from its mask once resized, and the data augmenter replaces the moved box of every
    object by the box of its moved mask after a geometric augmentation (a rotated box would become the larger
    enclosing box of its corners). An empty mask, whose object left the image, gives an empty box, removed by
    clean_bounding_boxes.

    Args:
        instance_masks (torch.Tensor): The [N, H, W] binary masks.

    Returns:
        torch.Tensor: The [N, 4] float [x1, y1, x2, y2] boxes, x2 and y2 one past the last pixel of the mask.
    """
    mask_height, mask_width = instance_masks.shape[-2], instance_masks.shape[-1]
    occupied_pixels = instance_masks > 0
    occupied_rows, occupied_columns = occupied_pixels.any(dim=2), occupied_pixels.any(dim=1)

    # First occupied row and column from the start, last ones from the end
    y_1 = occupied_rows.int().argmax(dim=1)
    y_2 = mask_height - occupied_rows.flip(dims=(1,)).int().argmax(dim=1)
    x_1 = occupied_columns.int().argmax(dim=1)
    x_2 = mask_width - occupied_columns.flip(dims=(1,)).int().argmax(dim=1)
    boxes = torch.stack(tensors=[x_1, y_1, x_2, y_2], dim=1).float()

    # The empty masks give an empty box
    return torch.where(occupied_rows.any(dim=1)[:, None], boxes, torch.zeros_like(boxes))


def build_instance_mask(segmentation: Union[List[List[float]], Dict[str, Any]], image_width: int, image_height: int,
                        image_size: Optional[int], keep_ratio: bool) -> np.ndarray:
    """
    Builds the binary mask of one COCO object at the network resolution, aligned with its preprocessed image, for the
    instance masks returned by CocoDetectionDataset: polygons are rasterised directly in the network input, run length
    encodings are decoded at the original resolution then resized with the nearest interpolation and padded with 0.
    Without image size, the mask stays at the original image resolution, for the mask evaluation at that resolution
    (evaluation.original_resolution of the instance segmentation).

    Args:
        segmentation (Union[List[List[float]], Dict[str, Any]]): The 'segmentation' field of the annotation, a list of
            polygon parts or a run length encoding.
        image_width (int): The width of the original image.
        image_height (int): The height of the original image.
        image_size (Optional[int]): The size of the square network input, None for the original image resolution.
        keep_ratio (bool): Whether the image is letterboxed (aspect ratio kept) or stretched, unused at the original
            resolution.

    Returns:
        np.ndarray: The [image_size, image_size] uint8 mask, or [image_height, image_width] at the original
            resolution, 1 on the object.

    Raises:
        ValueError: If a run length encoding at the original resolution does not have the size of its image.
    """
    # Polygon parts, placed like the boxes then filled at the network resolution, or filled at the original one
    if isinstance(segmentation, list):
        return rasterize_polygons(polygons=segmentation, image_width=image_width, image_height=image_height,
                                  image_size=image_size, keep_ratio=keep_ratio)

    # Run length encoding at the original resolution, which must cover its image exactly to stay aligned with it
    original_mask = decode_run_length_encoding(segmentation=segmentation)
    if image_size is None:
        if original_mask.shape != (image_height, image_width):
            raise ValueError(f"The run length encoding has the size {original_mask.shape[0]} x "
                             f"{original_mask.shape[1]} instead of the {image_height} x {image_width} of its image.")
        return original_mask

    # Resized on the pixel centres like the image, the plain nearest neighbour shifting the mask, and the box taken
    # from it, by half a pixel when the image shrinks
    return preprocess_image(image=original_mask, image_size=image_size, keep_ratio=keep_ratio,
                            interpolation=cv2.INTER_NEAREST_EXACT, padding_value=0)
