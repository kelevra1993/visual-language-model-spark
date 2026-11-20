"""
Anchor generation of the object detection task: the Anchors module computes the reference boxes of one feature map
level from its scales and aspect ratios, used by the region proposal heads and the target assignment of the
TwoStageDetector, and the best anchor overlap of every object, read by the anchor recall of the detection trainers.
"""

import torch
from torch import nn
from typing import List

# Number of boxes whose best anchor overlap is computed at once: millions of objects reuse a few megabytes of buffers
# instead of touching about a gigabyte of fresh memory (25 s with the trainings holding the memory), and at 32768, the
# size up to which PyTorch runs an element-wise operation on one thread, the many small operations skip the
# synchronisation of the thread pool (12 s on busy cores); 4.35 million boxes then take about 2 s
BEST_OVERLAP_CHUNK_SIZE = 32768


class Anchors(nn.Module):
    """
    Generates bounding box anchors for a specific feature map level.

    This module computes the reference bounding boxes (anchors) across a spatial grid
    corresponding to a specific feature map level. These anchors serve as the foundational
    coordinate references for the downstream Region Proposal Network and detector modules
    to predict offsets and classes.
    """

    def __init__(self, scales: List[float], aspect_ratios: List[float], input_image_size: int, feature_map_size: int,
                 dtype: torch.dtype, device: torch.device) -> None:
        """
        Initializes the Anchors module and pre-computes the anchor grid.

        Args:
            scales (List[float]): A list of base scales for the anchors.
            aspect_ratios (List[float]): A list of aspect ratios (height/width) for the anchors.
            input_image_size (int): The spatial resolution of the original input image.
            feature_map_size (int): The spatial resolution of the feature map at this level.
            dtype (torch.dtype): The tensor data type.
            device (torch.device): The device to allocate tensors on.
        """
        super(Anchors, self).__init__()

        self.dtype = dtype
        self.device = device

        # Set up scales, ratios and number of anchors per spatial location
        self.scales = torch.as_tensor(data=scales, dtype=self.dtype, device=self.device)
        self.aspect_ratios = torch.as_tensor(data=aspect_ratios, dtype=self.dtype, device=self.device)
        self.number_anchors_per_location = len(scales) * len(aspect_ratios)

        self.input_image_size = input_image_size
        self.feature_map_size = feature_map_size

        # Pre-generate the complete grid of anchors for this feature map level
        self.anchors = self.generate_anchors()

    def generate_anchors(self) -> torch.Tensor:
        """
        Computes the Cartesian product of scales and ratios and shifts them across the feature map.

        Returns:
            torch.Tensor: A tensor of shape [feature_map_size * feature_map_size * number_anchors, 4]
                          containing the absolute [x_min, y_min, x_max, y_max] coordinates for all anchors.
        """
        # Calculate the spatial stride (how many pixels in the original image correspond to 1 pixel in the feature map)
        stride = torch.tensor(data=self.input_image_size // self.feature_map_size,
                              dtype=self.dtype, device=self.device)

        # Compute different ratios for height and widths
        height_ratios = torch.sqrt(input=self.aspect_ratios)
        width_ratios = 1.0 / height_ratios

        # Compute all combinations of scales and aspect ratios using broadcasting
        anchor_heights = (height_ratios.unsqueeze(dim=-1) * self.scales.unsqueeze(dim=0)).reshape(shape=(-1,))
        anchor_widths = (width_ratios.unsqueeze(dim=-1) * self.scales.unsqueeze(dim=0)).reshape(shape=(-1,))

        # Create the base anchors centered at [0, 0] in the format [x_min, y_min, x_max, y_max]
        base_anchors = torch.stack(tensors=[-anchor_widths, -anchor_heights, anchor_widths, anchor_heights],
                                   dim=1) / 2.0

        # Get the shifts along the height and width across the entire feature map grid
        shifts = torch.arange(start=0, end=self.feature_map_size, dtype=self.dtype, device=self.device) * stride

        # Generate a 2D meshgrid of spatial coordinates (x, y)
        shifts_y, shifts_x = torch.meshgrid(shifts, shifts, indexing="ij")

        shifts_y = shifts_y.reshape(shape=(-1,))
        shifts_x = shifts_x.reshape(shape=(-1,))

        # Stack the shifts to match the [x_min, y_min, x_max, y_max] format
        shifts = torch.stack(tensors=(shifts_x, shifts_y, shifts_x, shifts_y), dim=1)

        # Add the base anchors to the spatial shifts to broadcast the anchors across the entire grid
        anchors = shifts.unsqueeze(dim=-2) + base_anchors.unsqueeze(dim=0)

        # Flatten the anchors to a 2D tensor of shape [total_anchors, 4]
        anchors = anchors.reshape(shape=(-1, 4))

        return anchors

    def compute_best_overlaps(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Gives the intersection over union of every box with its best anchor of this level, for the anchor recall of the
        detection trainers (at launch and in the 'compute_anchor_coverage' diagnosis mode), without comparing the box
        with every anchor. For one anchor shape, the intersection only shrinks as the anchor centre moves away from the
        box centre along x or along y, so the best anchor of a shape is the grid anchor nearest the box centre on each
        axis, and the best anchor of the level the best of these, one per scale and aspect ratio. Computed in float64
        on the CPU, BEST_OVERLAP_CHUNK_SIZE boxes at a time, about 2 seconds for the 4.35 million objects of a training
        split.

        Args:
            boxes (torch.Tensor): The [N, 4] [x_min, y_min, x_max, y_max] boxes, in network input pixels.

        Returns:
            torch.Tensor: The [N] float32 best overlap of every box, on the CPU.
        """
        best_overlaps = torch.zeros(size=(boxes.shape[0],), dtype=torch.float32)
        for chunk_start in range(0, boxes.shape[0], BEST_OVERLAP_CHUNK_SIZE):
            chunk_end = chunk_start + BEST_OVERLAP_CHUNK_SIZE
            best_overlaps[chunk_start:chunk_end] = self.compute_chunk_best_overlaps(boxes=boxes[chunk_start:chunk_end])
        return best_overlaps

    def compute_chunk_best_overlaps(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Computes the best anchor overlap of one chunk of boxes for compute_best_overlaps, from the grid anchor nearest
        every box centre for every scale and aspect ratio of the level.

        Args:
            boxes (torch.Tensor): The [M, 4] [x_min, y_min, x_max, y_max] boxes of the chunk, in network input pixels.

        Returns:
            torch.Tensor: The [M] float32 best overlap of every box of the chunk, on the CPU.
        """
        boxes = boxes.to(device="cpu", dtype=torch.float64)
        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Nearest anchor centre of the grid on each axis, the grid of generate_anchors starting at 0 with the stride
        stride = self.input_image_size // self.feature_map_size
        nearest_centres = []
        for minimum_column, maximum_column in [(0, 2), (1, 3)]:
            box_centres = (boxes[:, minimum_column] + boxes[:, maximum_column]) / 2
            nearest_centres.append(torch.clamp(input=torch.round(input=box_centres / stride), min=0,
                                               max=self.feature_map_size - 1) * stride)
        nearest_centre_x, nearest_centre_y = nearest_centres

        # Overlap with the nearest anchor of every scale and aspect ratio, sized like in generate_anchors
        best_overlaps = torch.zeros(size=(boxes.shape[0],), dtype=torch.float64)
        for aspect_ratio in self.aspect_ratios.to(device="cpu", dtype=torch.float64):
            for scale in self.scales.to(device="cpu", dtype=torch.float64):
                half_height = torch.sqrt(input=aspect_ratio) * scale / 2
                half_width = scale / torch.sqrt(input=aspect_ratio) / 2
                intersection_width = (torch.minimum(input=boxes[:, 2], other=nearest_centre_x + half_width) -
                                      torch.maximum(input=boxes[:, 0], other=nearest_centre_x - half_width))
                intersection_height = (torch.minimum(input=boxes[:, 3], other=nearest_centre_y + half_height) -
                                       torch.maximum(input=boxes[:, 1], other=nearest_centre_y - half_height))
                intersections = intersection_width.clamp(min=0) * intersection_height.clamp(min=0)
                unions = box_areas + 4 * half_width * half_height - intersections
                best_overlaps = torch.maximum(input=best_overlaps, other=intersections / unions)
        return best_overlaps.to(dtype=torch.float32)
