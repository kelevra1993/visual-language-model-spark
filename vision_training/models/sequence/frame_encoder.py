import torch

from torch import nn

from vision_training.models.backbones.backbone import Backbone


def encode_frames(backbone: Backbone, global_pooling: nn.Module, input_tensor: torch.Tensor,
                  sequence_mask: torch.Tensor) -> torch.Tensor:
    """
    Turns every frame of a batch of sequences into one feature vector, the input of the temporal module of the
    sequence models: the shared Backbone extracts the final feature map of every real frame, and the global pooling
    reduces it to a vector. The padding frames are not run through the backbone, which saves their computation and
    keeps them out of the batch normalization statistics; their feature vectors are zeros.

    Args:
        backbone (Backbone): The backbone of the sequence model.
        global_pooling (nn.Module): The global pooling mapping [N, C, H, W] feature maps to [N, C] vectors.
        input_tensor (torch.Tensor): The [B, T, 3, H, W] float BGR frames in [0, 255].
        sequence_mask (torch.Tensor): The [B, T] booleans, True for the real frames.

    Returns:
        torch.Tensor: The [B, T, F] frame feature vectors, zeros for the padding frames.
    """
    batch_size, sequence_length = sequence_mask.shape

    # Run the backbone on the real frames only, gathered into one batch of images
    final_feature_map, _ = backbone(input_tensor=input_tensor[sequence_mask])
    frame_vectors = global_pooling(input=final_feature_map)

    # Put every vector back at the position of its frame
    frame_features = frame_vectors.new_zeros(size=(batch_size, sequence_length, frame_vectors.shape[-1]))
    frame_features[sequence_mask] = frame_vectors
    return frame_features
