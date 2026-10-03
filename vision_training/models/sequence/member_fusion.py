import torch

from torch import nn
from typing import Optional, Tuple

# Combinations of the before and after vectors of a pair given to the classification head: their concatenation
# [a, b], with their difference [a, b, b - a], or with their difference after a cross attention between the frames of
# the two members
FUSION_TYPES = ["concatenation", "difference", "cross_attention"]

# Frames sharing the statistics of the sequence normalization of the backbone in the comparison model: the frames of
# each member window, or the frames of both members of a pair
SEQUENCE_NORMALIZATION_SCOPES = ["member", "pair"]


class MemberFusion(nn.Module):
    """
    Fusion of the two members of a pair in the sequence comparison model. The member vectors given by the temporal
    module are combined into the input of the classification head: [before, after] for "concatenation", and
    [before, after, after - before] for "difference" and "cross_attention", the difference making the change between
    the members explicit. With "cross_attention", the frames of each member also attend to the frames of the other
    member before the temporal module (one attention layer shared by both directions, with a residual connection and
    a layer normalization), so that the frames are matched across the members before being summarised.
    """

    def __init__(self, fusion_type: str, frame_features: int, member_features: int, number_heads: int,
                 dropout_rate: float, device: torch.device, dtype: torch.dtype) -> None:
        """
        Builds the cross attention layer when the fusion uses it.

        Args:
            fusion_type (str): "concatenation", "difference" or "cross_attention".
            frame_features (int): The number of features of the frame vectors, a multiple of number_heads for the cross
                attention.
            member_features (int): The number of features of the member vectors given by the temporal module.
            number_heads (int): The number of attention heads of the cross attention.
            dropout_rate (float): The dropout of the cross attention weights.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(MemberFusion, self).__init__()
        self.fusion_type = fusion_type
        self.output_features = member_features * (2 if fusion_type == "concatenation" else 3)

        # Attention of the frames of every member to the frames of the other member, shared by both directions
        if fusion_type == "cross_attention":
            self.cross_attention = nn.MultiheadAttention(embed_dim=frame_features, num_heads=number_heads,
                                                         dropout=dropout_rate, batch_first=True, device=device,
                                                         dtype=dtype)
            self.attention_normalization = nn.LayerNorm(normalized_shape=frame_features, device=device, dtype=dtype)

    def attend_to_member(self, query_features: torch.Tensor, other_member_features: torch.Tensor,
                         other_member_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Updates the frame vectors of one member with the frames of the other member they attend to, the padding frames
        of the other member being ignored.

        Args:
            query_features (torch.Tensor): The [B, T1, F] frame vectors of the updated member.
            other_member_features (torch.Tensor): The [B, T2, F] frame vectors of the other member.
            other_member_mask (Optional[torch.Tensor]): The [B, T2] booleans of the other member, True for its real
                frames, or None when every frame is real.

        Returns:
            torch.Tensor: The [B, T1, F] updated frame vectors.
        """
        key_padding_mask = None if other_member_mask is None else ~other_member_mask
        attended_features, _ = self.cross_attention(query=query_features, key=other_member_features,
                                                    value=other_member_features, key_padding_mask=key_padding_mask,
                                                    need_weights=False)
        return self.attention_normalization(input=query_features + attended_features)

    def attend_across_members(self, before_features: torch.Tensor, after_features: torch.Tensor,
                              before_mask: Optional[torch.Tensor],
                              after_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lets the frames of every member attend to the frames of the other member (cross attention fusion only, the
        frame vectors being returned unchanged otherwise). The padding frames are set back to zero vectors, so that
        the temporal module still ignores them.

        Args:
            before_features (torch.Tensor): The [B, T1, F] frame vectors of the before member.
            after_features (torch.Tensor): The [B, T2, F] frame vectors of the after member.
            before_mask (Optional[torch.Tensor]): The [B, T1] booleans of the before member, True for its real frames,
                or None when every frame is real.
            after_mask (Optional[torch.Tensor]): The [B, T2] booleans of the after member, or None when every frame is
                real.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The [B, T1, F] and [B, T2, F] frame vectors of the before and after
                members.
        """
        if self.fusion_type != "cross_attention":
            return before_features, after_features

        # Both directions computed from the frame vectors before any update
        attended_before = self.attend_to_member(query_features=before_features, other_member_features=after_features,
                                                other_member_mask=after_mask)
        attended_after = self.attend_to_member(query_features=after_features, other_member_features=before_features,
                                               other_member_mask=before_mask)

        # Padding frames back to zero vectors
        if before_mask is not None:
            attended_before = attended_before.masked_fill(mask=~before_mask.unsqueeze(dim=-1), value=0.0)
        if after_mask is not None:
            attended_after = attended_after.masked_fill(mask=~after_mask.unsqueeze(dim=-1), value=0.0)
        return attended_before, attended_after

    def combine_member_vectors(self, before_vectors: torch.Tensor, after_vectors: torch.Tensor) -> torch.Tensor:
        """
        Combines the vectors summarising the members of the pairs into the input of the classification head.

        Args:
            before_vectors (torch.Tensor): The [B, F'] vectors of the before members.
            after_vectors (torch.Tensor): The [B, F'] vectors of the after members.

        Returns:
            torch.Tensor: The [B, 2F'] concatenation, or the [B, 3F'] concatenation followed by the difference
                after - before.
        """
        if self.fusion_type == "concatenation":
            return torch.cat(tensors=[before_vectors, after_vectors], dim=-1)
        return torch.cat(tensors=[before_vectors, after_vectors, after_vectors - before_vectors], dim=-1)
