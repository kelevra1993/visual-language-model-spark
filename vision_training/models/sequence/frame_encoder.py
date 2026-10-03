import torch

from torch import nn
from typing import Optional

from vision_training.core.contracts import SequenceGrouping
from vision_training.models.backbones.backbone import Backbone


def encode_frames(backbone: Backbone, global_pooling: nn.Module, input_tensor: torch.Tensor,
                  sequence_mask: Optional[torch.Tensor],
                  sequence_groups: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Turns every frame of a batch of sequences into one feature vector, the input of the temporal module of the
    sequence models: the shared Backbone extracts the final feature map of every real frame, and the global pooling
    reduces it to a vector. The padding frames are not run through the backbone, which saves their computation and
    keeps them out of the batch normalization statistics; their feature vectors are zeros. Without mask, every frame
    is real (deployment windows), and the frames are run as one fixed-shape batch, which the exports require. The
    sequence of every frame is given to the backbone, whose sequence normalization layers (if any) compute their
    statistics over the frames of each sequence, or of each group of sequences (both members of a pair).

    Args:
        backbone (Backbone): The backbone of the sequence model.
        global_pooling (nn.Module): The global pooling mapping [N, C, H, W] feature maps to [N, C] vectors.
        input_tensor (torch.Tensor): The [B, T, 3, H, W] float BGR frames in [0, 255].
        sequence_mask (Optional[torch.Tensor]): The [B, T] booleans, True for the real frames, or None when every
            frame is real.
        sequence_groups (Optional[torch.Tensor]): The [B] normalization group of every sequence, in [0, B), with
            masks only, or None for one group per sequence.

    Returns:
        torch.Tensor: The [B, T, F] frame feature vectors, zeros for the padding frames.
    """
    batch_size, sequence_length = input_tensor.shape[0], input_tensor.shape[1]

    # Every frame is real: run the frames of all the sequences as one batch of images, in groups of T frames
    if sequence_mask is None:
        sequence_grouping: SequenceGrouping = {"frames_per_sequence": sequence_length, "frame_sequence_indices": None,
                                               "number_sequences": None}
        final_feature_map, _ = backbone(input_tensor=input_tensor.flatten(start_dim=0, end_dim=1),
                                        sequence_grouping=sequence_grouping)
        return global_pooling(input=final_feature_map).reshape(batch_size, sequence_length, -1)

    # Run the backbone on the real frames only, gathered into one batch of images, with the group of every frame
    if sequence_groups is None:
        sequence_groups = torch.arange(batch_size, device=input_tensor.device)
    sequence_grouping = {"frames_per_sequence": None, "number_sequences": batch_size,
                         "frame_sequence_indices": sequence_groups[:, None].expand(-1, sequence_length)[sequence_mask]}
    final_feature_map, _ = backbone(input_tensor=input_tensor[sequence_mask], sequence_grouping=sequence_grouping)
    frame_vectors = global_pooling(input=final_feature_map)

    # Put every vector back at the position of its frame
    frame_features = frame_vectors.new_zeros(size=(batch_size, sequence_length, frame_vectors.shape[-1]))
    frame_features[sequence_mask] = frame_vectors
    return frame_features
