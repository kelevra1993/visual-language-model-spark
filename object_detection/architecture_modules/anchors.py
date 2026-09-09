import torch
from torch import nn

from utilities.tensor_utilities import print_tensor_list, print_tensor_shape, print_tensor_status


class Anchors(nn.Module):
    def __init__(self, scales, aspect_ratios, input_image_size, feature_map_size, dtype, device):
        super(Anchors, self).__init__()

        self.dtype = dtype
        self.device = device

        # Set up scales, ratios and number of anchors
        self.scales = torch.as_tensor(data=scales, dtype=self.dtype, device=self.device)
        self.aspect_ratios = torch.as_tensor(data=aspect_ratios, dtype=self.dtype, device=self.device)
        self.number_anchors = len(scales) * len(aspect_ratios)

        self.input_image_size = input_image_size
        self.feature_map_size = feature_map_size

        self.anchors = self.generate_anchors()

    def generate_anchors(self):
        stride = torch.tensor(self.input_image_size // self.feature_map_size,
                              dtype=torch.int32, device=self.device)

        # Compute different ratios for height and widths
        height_ratios = torch.sqrt(self.aspect_ratios)
        width_ratios = 1 / height_ratios

        anchor_heights = (height_ratios.unsqueeze(-1) * self.scales.unsqueeze(0)).reshape(-1)
        anchor_widths = (width_ratios.unsqueeze(-1) * self.scales.unsqueeze(0)).reshape(-1)

        base_anchors = torch.stack([-anchor_widths, -anchor_heights, anchor_widths, anchor_heights], dim=1) / 2

        # Get the shifts along the height and width
        shifts = torch.arange(0, self.feature_map_size, dtype=torch.int32, device=self.device) * stride

        shifts_y, shifts_x = torch.meshgrid(shifts, shifts, indexing="ij")

        shifts_y = shifts_y.reshape(-1)
        shifts_x = shifts_x.reshape(-1)

        shifts = torch.stack((shifts_x, shifts_y, shifts_x, shifts_y), dim=1)

        anchors = shifts.unsqueeze(-2) + base_anchors.unsqueeze(0)

        anchors = anchors.reshape(-1, 4)

        return anchors
