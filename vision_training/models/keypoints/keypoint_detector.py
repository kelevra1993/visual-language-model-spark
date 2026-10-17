"""
Model of the keypoint detection task: KeypointDetector chains the first blocks of the shared Backbone down to 1/4 of
the object crop (the stem, transferable from the other tasks), the high resolution network of hrnet.py and a 1x1
convolution giving one heatmap per keypoint of every class (the union of the keypoints of the classes). The keypoints
are decoded from the heatmaps by their maximum, shifted by a quarter of a cell towards the higher neighbour, and the
channels of the class of every crop are selected for the loss and the evaluation. Registered in model_registry.py for
the "keypoint_detection" task and wrapped by inference_keypoint_detector.py for the exports.
"""

import torch

from rich.console import Console
from rich.panel import Panel
from torch import nn
from typing import Any, Dict, List, Optional, Tuple

from vision_training.losses.keypoint_losses import compute_keypoint_heatmap_loss
from vision_training.models.backbones.backbone import build_backbone
from vision_training.models.keypoints.hrnet import HighResolutionNetwork
from vision_training.models.task_model import TaskModel

# Crop pixels per heatmap cell, the resolution of the highest resolution branch of the high resolution network
HEATMAP_STRIDE = 4

# Shift of a decoded keypoint towards the higher of its two neighbouring cells, in heatmap cells
SUBPIXEL_SHIFT = 0.25

# Standard deviation of the initial weights of the heatmap head, small so that the first heatmaps are close to zero
HEAD_INITIALISATION_STANDARD_DEVIATION = 0.001


def build_keypoint_channel_table(keypoint_categories: Dict[int, Dict[str, Any]]) -> Tuple[torch.Tensor, List[str]]:
    """
    Assigns the heatmap channels of the keypoint detector to the keypoints of every class, the channels of a class
    following each other in the order of its keypoints and the classes in increasing order, so that the model, the
    export and the evaluation agree on what every channel predicts.

    Args:
        keypoint_categories (Dict[int, Dict[str, Any]]): The 'Data.keypoint_categories' definition of every class
            index, with its 'name' and 'keypoints' names.

    Returns:
        Tuple[torch.Tensor, List[str]]: The [number of classes + 1, most keypoints of a class] int64 channel of every
            keypoint position of every class (-1 for the padding positions and the classes without keypoints, such as
            the background 0), and the 'class.keypoint' name of every channel.

    Raises:
        ValueError: If no class defines keypoints.
    """
    if not keypoint_categories:
        raise ValueError("No class defines keypoints: 'Data.keypoint_categories' is empty, the training labels.json "
                         "must give the 'keypoints' names of its categories.")
    maximum_class = max(int(class_index) for class_index in keypoint_categories)
    maximum_keypoints = max(len(definition["keypoints"]) for definition in keypoint_categories.values())

    # Consecutive channels for the keypoints of every class, in increasing class order
    channel_table = torch.full(size=(maximum_class + 1, maximum_keypoints), fill_value=-1, dtype=torch.int64)
    channel_names = []
    for class_index in sorted(keypoint_categories, key=int):
        definition = keypoint_categories[class_index]
        for keypoint_index, keypoint_name in enumerate(definition["keypoints"]):
            channel_table[int(class_index), keypoint_index] = len(channel_names)
            channel_names.append(f"{definition['name']}.{keypoint_name}")
    return channel_table, channel_names


def gather_heatmap_values(flat_heatmaps: torch.Tensor, rows: torch.Tensor, columns: torch.Tensor, heatmap_height: int,
                          heatmap_width: int) -> torch.Tensor:
    """
    Reads the value of every heatmap at one cell, the cell being clamped inside the heatmap, for the neighbours of the
    maxima read by decode_heatmaps.

    Args:
        flat_heatmaps (torch.Tensor): The [B, K, h * w] flattened heatmaps.
        rows (torch.Tensor): The [B, K] int64 row of the cell read in every heatmap.
        columns (torch.Tensor): The [B, K] int64 column of the cell read in every heatmap.
        heatmap_height (int): The number of rows h of the heatmaps.
        heatmap_width (int): The number of columns w of the heatmaps.

    Returns:
        torch.Tensor: The [B, K] values of the cells.
    """
    flat_indices = rows.clamp(min=0, max=heatmap_height - 1) * heatmap_width + columns.clamp(min=0,
                                                                                            max=heatmap_width - 1)
    return flat_heatmaps.gather(dim=2, index=flat_indices.unsqueeze(dim=2)).squeeze(dim=2)


def decode_heatmaps(heatmaps: torch.Tensor, heatmap_stride: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decodes the keypoint of every heatmap of the keypoint detector, in the model and in its exports: the cell of the
    maximum, shifted by a quarter of a cell towards the higher of its two neighbours along each axis (no shift on the
    border of the heatmap), brought back to crop pixels with the centre of cell j at (j + 0.5) * stride, the convention
    of build_heatmap_targets. The score of the keypoint is the value of the maximum. It is computed in float32.

    Args:
        heatmaps (torch.Tensor): The [B, K, h, w] predicted heatmaps.
        heatmap_stride (int): The crop pixels per heatmap cell.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The [B, K, 2] float32 keypoints (x, y) in continuous crop pixels and their
            [B, K] float32 scores.
    """
    heatmaps = heatmaps.to(dtype=torch.float32)
    batch_size, number_channels, heatmap_height, heatmap_width = heatmaps.shape
    flat_heatmaps = heatmaps.reshape(batch_size, number_channels, heatmap_height * heatmap_width)

    # Cell of the maximum of every heatmap
    keypoint_scores, flat_indices = flat_heatmaps.max(dim=2)
    columns = flat_indices % heatmap_width
    rows = torch.div(flat_indices, heatmap_width, rounding_mode="floor")

    # Differences between the two neighbours along each axis, giving the direction of the quarter cell shift
    horizontal_difference = (gather_heatmap_values(flat_heatmaps=flat_heatmaps, rows=rows, columns=columns + 1,
                                                   heatmap_height=heatmap_height, heatmap_width=heatmap_width) -
                             gather_heatmap_values(flat_heatmaps=flat_heatmaps, rows=rows, columns=columns - 1,
                                                   heatmap_height=heatmap_height, heatmap_width=heatmap_width))
    vertical_difference = (gather_heatmap_values(flat_heatmaps=flat_heatmaps, rows=rows + 1, columns=columns,
                                                 heatmap_height=heatmap_height, heatmap_width=heatmap_width) -
                           gather_heatmap_values(flat_heatmaps=flat_heatmaps, rows=rows - 1, columns=columns,
                                                 heatmap_height=heatmap_height, heatmap_width=heatmap_width))

    # The maxima on the border of the heatmap have a single neighbour along that axis and are not shifted
    horizontal_inside = ((columns > 0) & (columns < heatmap_width - 1)).to(dtype=torch.float32)
    vertical_inside = ((rows > 0) & (rows < heatmap_height - 1)).to(dtype=torch.float32)
    shift_x = SUBPIXEL_SHIFT * torch.sign(horizontal_difference) * horizontal_inside
    shift_y = SUBPIXEL_SHIFT * torch.sign(vertical_difference) * vertical_inside

    # Shifted cell indices brought back to crop pixels
    keypoint_x = (columns.to(dtype=torch.float32) + shift_x + 0.5) * heatmap_stride
    keypoint_y = (rows.to(dtype=torch.float32) + shift_y + 0.5) * heatmap_stride
    return torch.stack(tensors=(keypoint_x, keypoint_y), dim=2), keypoint_scores


class KeypointDetector(TaskModel):
    """
    Model of the keypoint detection task, top-down: the object crop goes through the stem of the shared Backbone down
    to 1/4 of its size, the high resolution network keeps that resolution while exchanging features with its lower
    resolution branches, and a 1x1 convolution gives one heatmap per keypoint of every class. Every crop is decoded on
    all the channels, then the channels of its class (given by the box of the object) are selected.

    Outputs: 'heatmaps' [B, C, h, w], 'keypoints' [B, C, 2] in crop pixels and 'keypoint_scores' [B, C] of every
    channel; with the labels, 'category_keypoints' [B, K, 2] and 'category_keypoint_scores' [B, K] of the keypoints of
    the class of every crop (zero for the padding positions); with the targets, 'keypoint_loss'.
    """

    def __init__(self, configuration: Dict[str, Any], verbose: bool = False, device: torch.device = None,
                 dtype: torch.dtype = None) -> None:
        """
        Builds the stem, the high resolution network and the heatmap head from the configuration.

        Args:
            configuration (Dict[str, Any]): The model configuration (Data with its 'crop_settings' and injected
                'keypoint_categories', Backbone, HRNet and KeypointHead sections).
            verbose (bool): Unused, kept so that every task model is built with the same arguments.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(KeypointDetector, self).__init__()
        self.device = device
        self.dtype = dtype
        self.crop_height = configuration["Data"]["crop_settings"]["height"]
        self.crop_width = configuration["Data"]["crop_settings"]["width"]
        self.high_resolution_configuration = configuration["HRNet"]
        self.head_configuration = configuration["KeypointHead"]

        # Heatmap channel of every keypoint of every class, moved with the model to its device
        channel_table, self.channel_names = build_keypoint_channel_table(
            keypoint_categories=configuration["Data"]["keypoint_categories"])
        self.register_buffer(name="channel_table", tensor=channel_table.to(device=device))

        # Stem of the shared feature extractor, its last block at 1/4 of the crop (checked by keypoint_checks.py)
        self.backbone = build_backbone(backbone_configuration=configuration["Backbone"],
                                       input_image_size=self.crop_height, device=device, dtype=dtype)
        stem_channels = list(configuration["Backbone"]["convolutions"].values())[-1][1]

        # High resolution body on the stem feature map
        self.high_resolution_network = HighResolutionNetwork(
            input_channels=stem_channels, base_width=self.high_resolution_configuration["base_width"],
            modules_per_stage=self.high_resolution_configuration["modules_per_stage"],
            blocks_per_branch=self.high_resolution_configuration["blocks_per_branch"],
            modules=self.high_resolution_configuration["modules"],
            normalization=self.high_resolution_configuration["normalization"], device=device, dtype=dtype)

        # One heatmap per keypoint of every class, starting close to zero everywhere
        self.keypoint_head = nn.Conv2d(in_channels=self.high_resolution_configuration["base_width"],
                                       out_channels=len(self.channel_names), kernel_size=1, device=device, dtype=dtype)
        nn.init.normal_(tensor=self.keypoint_head.weight, mean=0.0, std=HEAD_INITIALISATION_STANDARD_DEVIATION)
        nn.init.zeros_(tensor=self.keypoint_head.bias)

    def forward(self, input_tensor: torch.Tensor, labels: Optional[torch.Tensor] = None,
                target_heatmaps: Optional[torch.Tensor] = None,
                heatmap_weights: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Predicts the heatmaps of a batch of object crops and decodes their keypoints, selects the keypoints of the
        class of every crop when the labels are given, and computes the heatmap loss when the targets are given too.

        Args:
            input_tensor (torch.Tensor): The [B, 3, crop height, crop width] float BGR crops in [0, 255].
            labels (Optional[torch.Tensor]): The [B] class index of every crop, or None.
            target_heatmaps (Optional[torch.Tensor]): The [B, K, h, w] target heatmaps in the keypoint order of the
                class of every crop, or None.
            heatmap_weights (Optional[torch.Tensor]): The [B, K] weight of every target keypoint, or None.

        Returns:
            Dict[str, Any]: The 'heatmaps', 'keypoints' and 'keypoint_scores' of every channel, plus
                'category_keypoints' and 'category_keypoint_scores' with the labels, and 'keypoint_loss' with the
                targets.
        """
        stem_features, _ = self.backbone(input_tensor=input_tensor)
        heatmaps = self.keypoint_head(self.high_resolution_network(input_tensor=stem_features))
        keypoints, keypoint_scores = decode_heatmaps(heatmaps=heatmaps, heatmap_stride=HEATMAP_STRIDE)
        outputs = {"heatmaps": heatmaps, "keypoints": keypoints, "keypoint_scores": keypoint_scores}
        if labels is None:
            return outputs

        # Channels of the keypoints of the class of every crop, the padding positions reading channel 0 and zeroed
        crop_channels = self.channel_table[labels]
        valid_positions = crop_channels >= 0
        gather_channels = crop_channels.clamp(min=0)
        category_keypoints = keypoints.gather(dim=1, index=gather_channels.unsqueeze(dim=2).expand(-1, -1, 2))
        outputs["category_keypoints"] = category_keypoints * valid_positions.unsqueeze(dim=2)
        outputs["category_keypoint_scores"] = keypoint_scores.gather(dim=1, index=gather_channels) * valid_positions
        if target_heatmaps is None:
            return outputs

        # Heatmap loss on the channels of the class of every crop only
        heatmap_height, heatmap_width = heatmaps.shape[2:]
        category_heatmaps = heatmaps.gather(dim=1, index=gather_channels[:, :, None, None].expand(
            -1, -1, heatmap_height, heatmap_width))
        outputs["keypoint_loss"] = compute_keypoint_heatmap_loss(predicted_heatmaps=category_heatmaps,
                                                                 target_heatmaps=target_heatmaps,
                                                                 heatmap_weights=heatmap_weights * valid_positions)
        return outputs

    def print_summary(self) -> None:
        """
        Prints the stem summary traced on a crop, then the branches of every stage of the high resolution network and
        the heatmap head: number of channels, heatmap size and Gaussian sigma of the targets.
        """
        self.backbone.print_summary(input_width=self.crop_width)

        # Shapes of the branches of every stage, traced in evaluation mode to keep the batch normalization statistics
        stem_height, stem_width = self.crop_height // HEATMAP_STRIDE, self.crop_width // HEATMAP_STRIDE
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy_crop = torch.zeros(size=(1, 3, self.crop_height, self.crop_width), device=self.device,
                                     dtype=self.dtype)
            stem_features, _ = self.backbone(input_tensor=dummy_crop)
            stage_shapes = self.high_resolution_network.trace_branch_shapes(input_tensor=stem_features)
        self.train(mode=was_training)

        network_configuration = self.high_resolution_configuration
        summary_text = (f"[bold]Base Width[/bold]          : {network_configuration['base_width']}\n"
                        f"[bold]Modules Per Stage[/bold]   : {network_configuration['modules_per_stage']}\n"
                        f"[bold]Blocks Per Branch[/bold]   : {network_configuration['blocks_per_branch']}\n"
                        f"[bold]Stem Output[/bold]         : {list(stem_features.shape)}\n\n")
        for stage_index, branch_shapes in enumerate(stage_shapes, start=1):
            summary_text += f"[bold] - Stage {stage_index}[/bold] ({len(branch_shapes)} branches)\n"
            for branch_index, branch_shape in enumerate(branch_shapes):
                summary_text += f"   - Branch 1/{HEATMAP_STRIDE * 2 ** branch_index:<3}: {branch_shape}\n"
        summary_text += (f"\n[bold]Heatmap Channels[/bold]    : {len(self.channel_names)} "
                         f"(the keypoints of {int((self.channel_table >= 0).any(dim=1).sum())} classes)\n"
                         f"[bold]Heatmap Size[/bold]        : {stem_height}x{stem_width} (stride {HEATMAP_STRIDE})\n"
                         f"[bold]Heatmap Sigma[/bold]       : {self.head_configuration['heatmap_sigma']} cells")
        Console(width=90).print(Panel(summary_text, title="[bold]High Resolution Network[/bold]",
                                      border_style="green"))

    def get_size_components(self) -> Dict[str, nn.Module]:
        """
        Lists the stem, the high resolution network, the heatmap head and the whole detector for the model size
        summary.

        Returns:
            Dict[str, nn.Module]: The display name and module of every component.
        """
        return {"Backbone": self.backbone, "High Resolution Network": self.high_resolution_network,
                "Keypoint Head": self.keypoint_head, "Total Model": self}
