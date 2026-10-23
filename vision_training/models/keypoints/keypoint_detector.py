"""
Model of the keypoint detection task, bottom-up (HigherHRNet): KeypointDetector chains the first blocks of the shared
Backbone down to 1/4 of the whole image (the stem, transferable from the other tasks), the high resolution network of
hrnet.py and a prediction head (a 1x1 convolution, after a hidden 3x3 convolution with KeypointHead.hidden_width)
giving, for every keypoint of every class (the union of the keypoints of the classes), one heatmap and one tag map at
1/4 of the image, plus, with the higher resolution branch, a second heatmap at 1/2 of the image, every heatmap being
the sigmoid of a predicted logit, the probability of a keypoint at every cell. Trained with the focal heatmap loss and
the associative embedding losses of losses/keypoint_losses.py on the targets of data/keypoints/heatmap_targets.py, it
gives at inference the keypoint candidates of every channel (local maxima of the averaged heatmaps placed within their
cell by the log-heatmap, with their score and tag), grouped into objects by models/keypoints/keypoint_grouping.py.
Registered in model_registry.py for the "keypoint_detection" task and wrapped by inference_keypoint_detector.py for the
exports.
"""

import math
import torch

from rich.console import Console
from rich.panel import Panel
from torch import nn
from torch.nn import functional as torch_functional
from typing import Any, Dict, List, Optional, Tuple

from vision_training.data.keypoints.heatmap_targets import build_keypoint_targets
from vision_training.losses.keypoint_losses import compute_heatmap_loss, compute_tag_losses
from vision_training.models.backbones.backbone import build_backbone
from vision_training.models.keypoints.hrnet import (BRANCH_SCALE_FACTOR, HigherResolutionBranch, HighResolutionNetwork,
                                                    build_prediction_head)
from vision_training.models.task_model import TaskModel

# Image pixels per heatmap cell, the resolution of the highest resolution branch of the high resolution network and of
# the tag maps
HEATMAP_STRIDE = 4

# Lowest heatmap value taken by the logarithm of the peak refinement, the heatmaps being probabilities that can be 0
LOG_HEATMAP_EPSILON = 1e-10

# Largest move of a keypoint candidate from the centre of its cell, in heatmap cells: the vertex of the parabola through
# a local maximum and its two neighbours lies within half a cell of it, the bound only holding back the zero-score
# candidates that are not local maxima
MAXIMUM_PEAK_OFFSET = 0.5

# Standard deviation of the initial weights of the last convolution of the heatmap and tag heads, small so that the
# first heatmaps and tags are the same at every cell
HEAD_INITIALISATION_STANDARD_DEVIATION = 0.001

# Probability of a keypoint at every cell of the first heatmaps, the sigmoid of the initial bias of the heatmap
# channels. The focal loss starts on a few cells only out of the many empty ones (RetinaNet "Focal Loss for Dense
# Object Detection" https://arxiv.org/abs/1708.02002 uses 0.01), so that the empty cells do not swamp the first
# iterations with a large loss.
HEATMAP_PRIOR_PROBABILITY = 0.01

# Side of the window of the local maxima of the heatmaps, a keypoint candidate being the highest cell of its window
CANDIDATE_WINDOW_SIZE = 3


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
    Reads the value of every heatmap at several cells, the cells being clamped inside the heatmap, for the neighbours
    of the keypoint candidates read by extract_keypoint_candidates.

    Args:
        flat_heatmaps (torch.Tensor): The [B, C, h * w] flattened heatmaps.
        rows (torch.Tensor): The [B, C, N] int64 rows of the cells read in every heatmap.
        columns (torch.Tensor): The [B, C, N] int64 columns of the cells read in every heatmap.
        heatmap_height (int): The number of rows h of the heatmaps.
        heatmap_width (int): The number of columns w of the heatmaps.

    Returns:
        torch.Tensor: The [B, C, N] values of the cells.
    """
    flat_indices = rows.clamp(min=0, max=heatmap_height - 1) * heatmap_width + columns.clamp(min=0,
                                                                                            max=heatmap_width - 1)
    return flat_heatmaps.gather(dim=2, index=flat_indices)


def compute_peak_offsets(centre_values: torch.Tensor, next_values: torch.Tensor,
                         previous_values: torch.Tensor) -> torch.Tensor:
    """
    Places the keypoint candidates of extract_keypoint_candidates within their cell along one axis, at the vertex of
    the parabola through the log-heatmap at the candidate cell and at its two neighbours: the second order Taylor
    expansion of the log-heatmap around its maximum of DARK ("Distribution-Aware Coordinate Representation for Human
    Pose Estimation" https://arxiv.org/abs/1910.06278), taken along each axis, which is exact for a round Gaussian peak.
    With a and b the drops of the log-heatmap from the cell to its next and previous neighbours, the offset is
    (b - a) / (2 (a + b)), towards the higher neighbour and within half a cell at a local maximum (both drops >= 0),
    and 0 on a flat cell. DARK also uses the cross derivative, which can push the offset out of the cell, and blurs
    the heatmaps first, which brought the peaks of a trained synthetic run no closer to their keypoints.

    Args:
        centre_values (torch.Tensor): The [B, C, N] heatmap values of the candidate cells.
        next_values (torch.Tensor): The [B, C, N] heatmap values of their next neighbours along the axis.
        previous_values (torch.Tensor): The [B, C, N] heatmap values of their previous neighbours along the axis.

    Returns:
        torch.Tensor: The [B, C, N] offsets in heatmap cells, positive towards the next neighbour, within
            [-MAXIMUM_PEAK_OFFSET, MAXIMUM_PEAK_OFFSET].
    """
    # Drops of the log-heatmap from the candidate cell to its two neighbours
    log_centre_values = torch.log(input=centre_values.clamp(min=LOG_HEATMAP_EPSILON))
    next_drops = log_centre_values - torch.log(input=next_values.clamp(min=LOG_HEATMAP_EPSILON))
    previous_drops = log_centre_values - torch.log(input=previous_values.clamp(min=LOG_HEATMAP_EPSILON))

    # Vertex of the parabola, none on a flat cell, bounded for the zero-score candidates that are not local maxima
    total_drops = next_drops + previous_drops
    safe_total_drops = torch.where(total_drops > 0.0, total_drops, torch.ones_like(input=total_drops))
    offsets = torch.where(total_drops > 0.0, (previous_drops - next_drops) / (2.0 * safe_total_drops),
                          torch.zeros_like(input=total_drops))
    return offsets.clamp(min=-MAXIMUM_PEAK_OFFSET, max=MAXIMUM_PEAK_OFFSET)


def extract_keypoint_candidates(heatmaps: torch.Tensor, tags: torch.Tensor,
                                higher_resolution_heatmaps: Optional[torch.Tensor],
                                maximum_candidates: int) -> Dict[str, torch.Tensor]:
    """
    Extracts the keypoint candidates of every channel of the keypoint detector, in the model and in its exports, before
    their grouping into objects (keypoint_grouping.py). With the higher resolution branch, the quarter resolution
    heatmaps and tags are upsampled bilinearly to half the image resolution and the heatmaps averaged with the higher
    resolution ones. The candidates of a channel are its local maxima over a 3x3 window, the maximum_candidates highest
    ones kept, each placed within its cell along each axis at the vertex of the parabola through the log-heatmap at its
    cell and at its two neighbours (compute_peak_offsets, no move on the border), brought back to image pixels with the
    centre of cell j at (j + 0.5) x stride, the convention of build_keypoint_targets. Its score is the value of its cell
    and its tag the tag of its channel at its cell. It is computed in float32.

    Args:
        heatmaps (torch.Tensor): The [B, C, h, w] heatmaps at a quarter of the image resolution.
        tags (torch.Tensor): The [B, C, h, w] tag maps at a quarter of the image resolution.
        higher_resolution_heatmaps (Optional[torch.Tensor]): The [B, C, 2h, 2w] heatmaps of the higher resolution
            branch, or None without it.
        maximum_candidates (int): The number of candidates kept per channel, the highest local maxima (padded with
            zero scores when a heatmap has fewer).

    Returns:
        Dict[str, torch.Tensor]: The 'candidate_positions' [B, C, N, 2] (x, y) in continuous image pixels, the
            'candidate_scores' [B, C, N] and the 'candidate_tags' [B, C, N], N being maximum_candidates (at most the
            number of cells), the candidates of every channel by decreasing score.
    """
    heatmaps = heatmaps.to(dtype=torch.float32)
    tags = tags.to(dtype=torch.float32)

    # Heatmaps and tags at half the image resolution with the higher resolution branch, averaging both heatmaps
    heatmap_stride = HEATMAP_STRIDE
    if higher_resolution_heatmaps is not None:
        heatmaps = (torch_functional.interpolate(input=heatmaps, scale_factor=BRANCH_SCALE_FACTOR, mode="bilinear",
                                                 align_corners=False) +
                    higher_resolution_heatmaps.to(dtype=torch.float32)) / 2.0
        tags = torch_functional.interpolate(input=tags, scale_factor=BRANCH_SCALE_FACTOR, mode="bilinear",
                                            align_corners=False)
        heatmap_stride = HEATMAP_STRIDE // BRANCH_SCALE_FACTOR
    batch_size, number_channels, heatmap_height, heatmap_width = heatmaps.shape
    flat_heatmaps = heatmaps.reshape(batch_size, number_channels, heatmap_height * heatmap_width)

    # Local maxima of every heatmap, the other cells set to zero, and the highest ones kept
    window_maxima = torch_functional.max_pool2d(input=heatmaps, kernel_size=CANDIDATE_WINDOW_SIZE, stride=1,
                                                padding=CANDIDATE_WINDOW_SIZE // 2)
    local_maxima = torch.where(heatmaps == window_maxima, heatmaps, torch.zeros_like(input=heatmaps))
    number_candidates = min(maximum_candidates, heatmap_height * heatmap_width)
    candidate_scores, flat_indices = local_maxima.reshape(batch_size, number_channels, -1).topk(k=number_candidates,
                                                                                                dim=2)
    columns = flat_indices % heatmap_width
    rows = torch.div(flat_indices, heatmap_width, rounding_mode="floor")

    # Offsets of every candidate within its cell along each axis, from the values of its cell and of its two neighbours
    # read on the heatmaps before the suppression
    centre_values = flat_heatmaps.gather(dim=2, index=flat_indices)
    offset_x = compute_peak_offsets(
        centre_values=centre_values,
        next_values=gather_heatmap_values(flat_heatmaps=flat_heatmaps, rows=rows, columns=columns + 1,
                                          heatmap_height=heatmap_height, heatmap_width=heatmap_width),
        previous_values=gather_heatmap_values(flat_heatmaps=flat_heatmaps, rows=rows, columns=columns - 1,
                                              heatmap_height=heatmap_height, heatmap_width=heatmap_width))
    offset_y = compute_peak_offsets(
        centre_values=centre_values,
        next_values=gather_heatmap_values(flat_heatmaps=flat_heatmaps, rows=rows + 1, columns=columns,
                                          heatmap_height=heatmap_height, heatmap_width=heatmap_width),
        previous_values=gather_heatmap_values(flat_heatmaps=flat_heatmaps, rows=rows - 1, columns=columns,
                                              heatmap_height=heatmap_height, heatmap_width=heatmap_width))

    # The candidates on the border of the heatmap have a single neighbour along that axis and are not moved
    horizontal_inside = ((columns > 0) & (columns < heatmap_width - 1)).to(dtype=torch.float32)
    vertical_inside = ((rows > 0) & (rows < heatmap_height - 1)).to(dtype=torch.float32)

    # Moved cell indices brought back to image pixels, and the tag of every candidate at its cell
    candidate_x = (columns.to(dtype=torch.float32) + offset_x * horizontal_inside + 0.5) * heatmap_stride
    candidate_y = (rows.to(dtype=torch.float32) + offset_y * vertical_inside + 0.5) * heatmap_stride
    candidate_tags = tags.reshape(batch_size, number_channels, -1).gather(dim=2, index=flat_indices)
    return {"candidate_positions": torch.stack(tensors=(candidate_x, candidate_y), dim=3),
            "candidate_scores": candidate_scores, "candidate_tags": candidate_tags}


class KeypointDetector(TaskModel):
    """
    Model of the keypoint detection task, bottom-up: the whole image goes through the stem of the shared Backbone down
    to 1/4 of its size, the high resolution network keeps that resolution while exchanging features with its lower
    resolution branches, and the prediction head (a 1x1 convolution, after a hidden 3x3 convolution with a hidden
    width) gives one heatmap and one tag map per keypoint of every class. With the higher resolution branch, a second
    set of heatmaps is predicted at 1/2 of the image from the features and these predictions.

    Outputs: 'heatmaps' (the sigmoid of the heatmap logits) and 'tags' [B, C, H / 4, W / 4], with the higher resolution
    branch 'higher_resolution_heatmaps' [B, C, H / 2, W / 2]; with the ground truth, the 'heatmap_loss' (computed on
    the logits, without the cells of the ignored regions), 'tag_pull_loss' and 'tag_push_loss'; without it, the
    keypoint candidates of every channel of extract_keypoint_candidates ('candidate_positions', 'candidate_scores',
    'candidate_tags').
    """

    def __init__(self, configuration: Dict[str, Any], verbose: bool = False, device: torch.device = None,
                 dtype: torch.dtype = None) -> None:
        """
        Builds the stem, the high resolution network, the heatmap and tag head and the optional higher resolution
        branch from the configuration.

        Args:
            configuration (Dict[str, Any]): The model configuration (Data with its 'image_settings' and injected
                'keypoint_categories', Backbone, HRNet and KeypointHead sections).
            verbose (bool): Unused, kept so that every task model is built with the same arguments.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(KeypointDetector, self).__init__()
        self.device = device
        self.dtype = dtype
        self.image_size = configuration["Data"]["image_settings"]["size"]
        self.high_resolution_configuration = configuration["HRNet"]
        self.head_configuration = configuration["KeypointHead"]
        self.higher_resolution = bool(self.head_configuration["higher_resolution"])
        self.hidden_width = self.head_configuration["hidden_width"]
        self.heatmap_sigma = self.head_configuration["heatmap_sigma"]
        self.tag_pull_weight = self.head_configuration["tag_pull_weight"]
        self.tag_push_weight = self.head_configuration["tag_push_weight"]
        self.maximum_candidates = self.head_configuration["inference"]["maximum_candidates"]

        # Heatmap channel of every keypoint of every class, moved with the model to its device
        channel_table, self.channel_names = build_keypoint_channel_table(
            keypoint_categories=configuration["Data"]["keypoint_categories"])
        self.register_buffer(name="channel_table", tensor=channel_table.to(device=device))
        self.number_channels = len(self.channel_names)

        # Stem of the shared feature extractor, its last block at 1/4 of the image (checked by keypoint_checks.py)
        self.backbone = build_backbone(backbone_configuration=configuration["Backbone"],
                                       input_image_size=self.image_size, device=device, dtype=dtype)
        stem_channels = list(configuration["Backbone"]["convolutions"].values())[-1][1]

        # High resolution body on the stem feature map
        base_width = self.high_resolution_configuration["base_width"]
        self.high_resolution_network = HighResolutionNetwork(
            input_channels=stem_channels, base_width=base_width,
            modules_per_stage=self.high_resolution_configuration["modules_per_stage"],
            blocks_per_branch=self.high_resolution_configuration["blocks_per_branch"],
            modules=self.high_resolution_configuration["modules"],
            normalization=self.high_resolution_configuration["normalization"], device=device, dtype=dtype)

        # One heatmap then one tag map per keypoint of every class, in the dictionary of the whole head so that the
        # KeypointHead section transfers, freezes and sets the learning rate of the higher resolution branch with it
        self.keypoint_head = nn.ModuleDict({"prediction_convolution": build_prediction_head(
            input_channels=base_width, hidden_width=self.hidden_width, output_channels=2 * self.number_channels,
            normalization=self.high_resolution_configuration["normalization"], device=device, dtype=dtype)})
        prediction_heads = [self.keypoint_head["prediction_convolution"]]

        # Higher resolution heatmaps from the features and the quarter resolution predictions
        if self.higher_resolution:
            self.keypoint_head["higher_resolution_branch"] = HigherResolutionBranch(
                feature_channels=base_width, prediction_channels=2 * self.number_channels,
                heatmap_channels=self.number_channels, hidden_width=self.hidden_width,
                blocks_per_branch=self.high_resolution_configuration["blocks_per_branch"],
                modules=self.high_resolution_configuration["modules"],
                normalization=self.high_resolution_configuration["normalization"], device=device, dtype=dtype)
            prediction_heads.append(self.keypoint_head["higher_resolution_branch"].heatmap_head)

        # Last convolution of every head starting with the same predictions at every cell: the heatmaps at the prior
        # probability and the tags at zero, the tags following the heatmaps in the quarter resolution head
        prior_logit = math.log(HEATMAP_PRIOR_PROBABILITY / (1.0 - HEATMAP_PRIOR_PROBABILITY))
        for prediction_head in prediction_heads:
            nn.init.normal_(tensor=prediction_head[-1].weight, mean=0.0, std=HEAD_INITIALISATION_STANDARD_DEVIATION)
            nn.init.zeros_(tensor=prediction_head[-1].bias)
            nn.init.constant_(tensor=prediction_head[-1].bias[:self.number_channels], val=prior_logit)

    def forward(self, input_tensor: torch.Tensor, ground_truth_keypoints: Optional[List[torch.Tensor]] = None,
                ground_truth_labels: Optional[List[torch.Tensor]] = None,
                ignored_regions: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Predicts the heatmaps and tags of a batch of whole images, the heatmaps being the sigmoid of the predicted
        logits. With the ground truth, builds the targets on the device and returns the heatmap loss of the logits
        (summed over the strides, leaving out the cells of the ignored regions) and the weighted tag losses; without
        it, extracts the keypoint candidates of every channel.

        Args:
            input_tensor (torch.Tensor): The [B, 3, H, W] float BGR images in [0, 255].
            ground_truth_keypoints (Optional[List[torch.Tensor]]): One [N_i, K, 3] tensor per image of the (x, y,
                visibility) keypoints of its objects in image pixels, or None.
            ground_truth_labels (Optional[List[torch.Tensor]]): One [N_i] int64 tensor per image of the class of its
                objects, or None.
            ignored_regions (Optional[torch.Tensor]): The [B, H, W] uint8 pixels of the objects without labelled
                keypoint, left out of the heatmap loss ('KeypointHead.ignore_unlabelled_objects'), or None to train
                every cell. Unused without the ground truth. Defaults to None.

        Returns:
            Dict[str, Any]: The 'heatmaps' and 'tags' (and 'higher_resolution_heatmaps'), plus the three losses with
                the ground truth, or the keypoint candidates without it.
        """
        stem_features, _ = self.backbone(input_tensor=input_tensor)
        features = self.high_resolution_network(input_tensor=stem_features)
        predictions = self.keypoint_head["prediction_convolution"](features)
        heatmap_logits = [predictions[:, :self.number_channels]]
        outputs = {"heatmaps": torch.sigmoid(heatmap_logits[0]), "tags": predictions[:, self.number_channels:]}

        # Higher resolution heatmaps from the features and the logits and tags of the quarter resolution
        higher_resolution_heatmaps = None
        if self.higher_resolution:
            heatmap_logits.append(self.keypoint_head["higher_resolution_branch"](features=features,
                                                                                  predictions=predictions))
            higher_resolution_heatmaps = torch.sigmoid(heatmap_logits[1])
            outputs["higher_resolution_heatmaps"] = higher_resolution_heatmaps

        # Keypoint candidates of every channel at inference
        if ground_truth_keypoints is None:
            candidates = extract_keypoint_candidates(heatmaps=outputs["heatmaps"], tags=outputs["tags"],
                                                     higher_resolution_heatmaps=higher_resolution_heatmaps,
                                                     maximum_candidates=self.maximum_candidates)
            return {**outputs, **candidates}

        # Targets of every heatmap stride and of the tags, then the losses
        image_height, image_width = input_tensor.shape[2:]
        heatmap_strides = [HEATMAP_STRIDE // BRANCH_SCALE_FACTOR ** stride_index
                           for stride_index in range(len(heatmap_logits))]
        targets = build_keypoint_targets(keypoints=ground_truth_keypoints, labels=ground_truth_labels,
                                         channel_table=self.channel_table, number_channels=self.number_channels,
                                         image_height=image_height, image_width=image_width,
                                         heatmap_strides=heatmap_strides, tag_stride=HEATMAP_STRIDE,
                                         heatmap_sigma=self.heatmap_sigma, ignored_regions=ignored_regions)

        # Heatmap loss of every stride, without the cells of the ignored regions when there are some
        ignored_cells = targets["ignored_cells"] if ignored_regions is not None else [None] * len(heatmap_logits)
        outputs["heatmap_loss"] = sum(compute_heatmap_loss(predicted_logits=stride_logits,
                                                           target_heatmaps=target_heatmaps,
                                                           ignored_cells=stride_ignored_cells)
                                      for stride_logits, target_heatmaps, stride_ignored_cells
                                      in zip(heatmap_logits, targets["heatmaps"], ignored_cells, strict=True))
        pull_loss, push_loss = compute_tag_losses(tag_maps=outputs["tags"], tag_indices=targets["tag_indices"],
                                                  tag_valid=targets["tag_valid"],
                                                  object_labels=targets["object_labels"])
        outputs["tag_pull_loss"] = self.tag_pull_weight * pull_loss
        outputs["tag_push_loss"] = self.tag_push_weight * push_loss
        return outputs

    def print_summary(self) -> None:
        """
        Prints the stem summary traced on an image, then the branches of every stage of the high resolution network and
        the heads: number of heatmap and tag channels, hidden width, heatmap sizes and Gaussian sigma of the targets,
        tag loss weights.
        """
        self.backbone.print_summary()

        # Shapes of the branches of every stage, traced in evaluation mode to keep the batch normalization statistics
        heatmap_size = self.image_size // HEATMAP_STRIDE
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy_image = torch.zeros(size=(1, 3, self.image_size, self.image_size), device=self.device,
                                      dtype=self.dtype)
            stem_features, _ = self.backbone(input_tensor=dummy_image)
            stage_shapes = self.high_resolution_network.trace_branch_shapes(input_tensor=stem_features)
        self.train(mode=was_training)

        # The settings of the network and the stem output, then the shape of every branch of every stage
        network_configuration = self.high_resolution_configuration
        summary_text = (f"[bold]Base Width[/bold]          : {network_configuration['base_width']}\n"
                        f"[bold]Modules Per Stage[/bold]   : {network_configuration['modules_per_stage']}\n"
                        f"[bold]Blocks Per Branch[/bold]   : {network_configuration['blocks_per_branch']}\n"
                        f"[bold]Stem Output[/bold]         : {list(stem_features.shape)}\n\n")
        for stage_index, branch_shapes in enumerate(stage_shapes, start=1):
            summary_text += f"[bold] - Stage {stage_index}[/bold] ({len(branch_shapes)} branches)\n"
            for branch_index, branch_shape in enumerate(branch_shapes):
                summary_text += f"   - Branch 1/{HEATMAP_STRIDE * 2 ** branch_index:<3}: {branch_shape}\n"
        # The settings of the heads and of their targets, printed in a panel below the branches
        higher_resolution_size = heatmap_size * BRANCH_SCALE_FACTOR
        higher_resolution_line = (f"{higher_resolution_size}x{higher_resolution_size} (stride "
                                  f"{HEATMAP_STRIDE // BRANCH_SCALE_FACTOR})" if self.higher_resolution else "Off")
        hidden_width_line = (f"{self.hidden_width} (3x3 convolution before the predictions)" if self.hidden_width > 0
                             else "Off (1x1 predictions on the features)")
        summary_text += (f"\n[bold]Heatmap Channels[/bold]    : {self.number_channels} "
                         f"(the keypoints of {int((self.channel_table >= 0).any(dim=1).sum())} classes)\n"
                         f"[bold]Tag Channels[/bold]        : {self.number_channels}\n"
                         f"[bold]Hidden Width[/bold]        : {hidden_width_line}\n"
                         f"[bold]Heatmap Size[/bold]        : {heatmap_size}x{heatmap_size} (stride {HEATMAP_STRIDE})\n"
                         f"[bold]Higher Resolution[/bold]   : {higher_resolution_line}\n"
                         f"[bold]Heatmap Sigma[/bold]       : {self.heatmap_sigma} cells\n"
                         f"[bold]Tag Weights[/bold]         : pull {self.tag_pull_weight:g}, push "
                         f"{self.tag_push_weight:g}")
        Console(width=90).print(Panel(summary_text, title="[bold]High Resolution Network[/bold]",
                                      border_style="green"))

    def get_size_components(self) -> Dict[str, nn.Module]:
        """
        Lists the stem, the high resolution network, the heatmap and tag head, the higher resolution branch when it is
        on and the whole detector for the model size summary.

        Returns:
            Dict[str, nn.Module]: The display name and module of every component.
        """
        components = {"Backbone": self.backbone, "High Resolution Network": self.high_resolution_network,
                      "Keypoint Head": self.keypoint_head["prediction_convolution"]}
        if self.higher_resolution:
            components["Higher Resolution Branch"] = self.keypoint_head["higher_resolution_branch"]
        return {**components, "Total Model": self}
