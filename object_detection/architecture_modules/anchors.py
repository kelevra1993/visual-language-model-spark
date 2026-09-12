import torch
from torch import nn
from typing import List


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
            aspect_ratios (List[float]): A list of aspect ratios (width/height) for the anchors.
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
