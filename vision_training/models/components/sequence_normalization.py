"""
Sequence batch normalization ('feature_map_normalization: "sequence"') used in the ConvolutionBlocks of the backbone
of the sequence models: statistics computed over the frames of each window (or pair) at training and at inference,
with the same tensors as nn.BatchNorm2d so that a backbone transfers between batch and sequence normalization.
"""

import torch

from torch import nn
from typing import Optional

from vision_training.core.contracts import SequenceGrouping


class SequenceBatchNormalization(nn.BatchNorm2d):
    """
    Batch normalization whose statistics are computed over the frames of each sequence, at training and at inference,
    for the backbone of the sequence models ('feature_map_normalization: "sequence"'): every channel of every frame
    is normalised with the mean and variance of that channel over the frames of its sequence (its window, or its pair
    for the comparison task with the pair scope) and their spatial positions, so that a frame is normalised by the
    frames it is seen with, whatever the lighting of the sequence.

    It holds the same parameters and buffers as nn.BatchNorm2d, so that a backbone can move between batch and sequence
    normalization through the transfer between runs. The running statistics keep following the training batches as
    in batch normalization, but they are never used.
    """

    def update_running_statistics(self, input_tensor: torch.Tensor) -> None:
        """
        Updates the running mean and variance with the statistics of the whole training batch, exactly as batch
        normalization does (unbiased variance, momentum or cumulative average), so that the backbone remains usable
        with batch normalization.

        Args:
            input_tensor (torch.Tensor): The [N, C, H, W] feature maps of the batch.
        """
        with torch.no_grad():
            self.num_batches_tracked += 1
            average_factor = 1.0 / float(self.num_batches_tracked) if self.momentum is None else self.momentum
            batch_mean = input_tensor.mean(dim=(0, 2, 3))
            batch_variance = input_tensor.var(dim=(0, 2, 3),
                                              correction=1 if input_tensor.numel() > input_tensor.shape[1] else 0)
            self.running_mean.mul_(other=1.0 - average_factor).add_(
                other=batch_mean.to(dtype=self.running_mean.dtype), alpha=average_factor)
            self.running_var.mul_(other=1.0 - average_factor).add_(
                other=batch_variance.to(dtype=self.running_var.dtype), alpha=average_factor)

    def forward(self, input_tensor: torch.Tensor,
                sequence_grouping: Optional[SequenceGrouping] = None) -> torch.Tensor:
        """
        Normalises every frame with the statistics of its sequence, then applies the learned scale and shift. The
        statistics are computed in float32 (float64 for float64 inputs), also under bfloat16 autocast.

        Args:
            input_tensor (torch.Tensor): The [N, C, H, W] feature maps of the frames of the batch.
            sequence_grouping (Optional[SequenceGrouping]): The sequence of every frame: equal consecutive groups
                ('frames_per_sequence', reshaped, export friendly) or a group index per frame (segment sums), or None
                to treat the whole batch as one sequence (backbone summary, receptive field).

        Returns:
            torch.Tensor: The [N, C, H, W] normalised feature maps, in the precision of the input.
        """
        computation_dtype = torch.float64 if input_tensor.dtype == torch.float64 else torch.float32
        statistics_input = input_tensor.to(dtype=computation_dtype)

        # Running statistics of the whole batch, never used by the normalization itself
        if self.training and self.track_running_stats:
            self.update_running_statistics(input_tensor=statistics_input)

        # Whole batch as one sequence
        if sequence_grouping is None:
            sequence_means = statistics_input.mean(dim=(0, 2, 3), keepdim=True)
            centred_input = statistics_input - sequence_means
            sequence_variances = centred_input.square().mean(dim=(0, 2, 3), keepdim=True)
            normalized_input = centred_input / torch.sqrt(input=sequence_variances + self.eps)

        # Equal consecutive groups of frames, gathered along a frame axis
        elif sequence_grouping["frames_per_sequence"] is not None:
            channels, height, width = statistics_input.shape[1:]
            grouped_input = statistics_input.reshape(-1, sequence_grouping["frames_per_sequence"], channels, height,
                                                     width)
            sequence_means = grouped_input.mean(dim=(1, 3, 4), keepdim=True)
            centred_input = grouped_input - sequence_means
            sequence_variances = centred_input.square().mean(dim=(1, 3, 4), keepdim=True)
            normalized_input = (centred_input / torch.sqrt(input=sequence_variances + self.eps)).reshape(
                statistics_input.shape)

        # Groups of any size, summed frame by frame into their group
        else:
            frame_sequence_indices = sequence_grouping["frame_sequence_indices"]
            number_sequences = sequence_grouping["number_sequences"]
            channels, height, width = statistics_input.shape[1:]
            frame_ones = statistics_input.new_ones(size=(frame_sequence_indices.shape[0],))
            frame_counts = statistics_input.new_zeros(size=(number_sequences,)).index_add_(
                dim=0, index=frame_sequence_indices, source=frame_ones)
            values_per_sequence = (frame_counts * height * width).clamp(min=1.0)[:, None]
            sequence_means = statistics_input.new_zeros(size=(number_sequences, channels)).index_add_(
                dim=0, index=frame_sequence_indices, source=statistics_input.sum(dim=(2, 3))) / values_per_sequence
            centred_input = statistics_input - sequence_means[frame_sequence_indices][:, :, None, None]
            sequence_variances = statistics_input.new_zeros(size=(number_sequences, channels)).index_add_(
                dim=0, index=frame_sequence_indices,
                source=centred_input.square().sum(dim=(2, 3))) / values_per_sequence
            normalized_input = centred_input / torch.sqrt(
                input=sequence_variances[frame_sequence_indices][:, :, None, None] + self.eps)

        # Learned scale and shift of every channel
        if self.affine:
            normalized_input = (normalized_input * self.weight.to(dtype=computation_dtype)[None, :, None, None] +
                                self.bias.to(dtype=computation_dtype)[None, :, None, None])
        return normalized_input.to(dtype=input_tensor.dtype)
