import torch

from torch import nn
from typing import Any, Dict, Optional, Tuple

from vision_training.losses.classification_losses import compute_classification_loss, convert_logits_to_probabilities
from vision_training.models.backbones.backbone import build_backbone
from vision_training.models.classification.classification_head import (ClassificationHead, build_global_pooling,
                                                                       print_classification_head_summary)
from vision_training.models.sequence.frame_encoder import encode_frames
from vision_training.models.sequence.member_fusion import MemberFusion
from vision_training.models.sequence.temporal_modules import build_temporal_module, print_sequence_module_summary
from vision_training.models.task_model import TaskModel


class SequenceComparator(TaskModel):
    """
    Model of the sequence comparison task: two sequences of frames, a before member and an after member, classified
    together (e.g. whether an object was lost between them). The shared Backbone and global pooling turn every frame
    of both members into a feature vector, the member fusion lets the frames of each member attend to the other member
    (cross attention fusion only), the shared temporal module summarises every member into one vector, ignoring its
    padding frames, the member fusion combines the two member vectors, and the classification head gives the logits
    of the classes of the pair.

    With sequence masks (training and evaluation, both members padded to the same number of frames), the two members
    are encoded and summarised together as one batch, so that the batch normalization statistics are computed over the
    frames of both members. Without masks (deployment, every frame real), each member is encoded and summarised on its
    own, so that the two members may have different numbers of frames.

    Outputs: 'class_logits' [B, K] and 'class_probabilities' [B, K], plus 'classification_loss' when labels are given.
    """

    def __init__(self, configuration: Dict[str, Any], verbose: bool = False, device: torch.device = None,
                 dtype: torch.dtype = None) -> None:
        """
        Builds the backbone, the global pooling, the temporal module, the member fusion and the classification head
        from the configuration.

        Args:
            configuration (Dict[str, Any]): The model configuration (Data with its label dictionary and sequence
                settings, Backbone, TemporalModel, Fusion and ClassificationHead sections).
            verbose (bool): Unused, kept so that every task model is built with the same arguments.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(SequenceComparator, self).__init__()
        self.device = device
        self.dtype = dtype
        self.head_configuration = configuration["ClassificationHead"]
        self.temporal_configuration = configuration["TemporalModel"]
        self.fusion_configuration = configuration["Fusion"]
        self.output_mode = self.head_configuration["output_mode"]
        self.label_smoothing = self.head_configuration["label_smoothing"]
        self.number_classes = len(set(configuration["Data"]["label_dictionary"].values()))
        self.sequence_length = configuration["Data"]["sequence_settings"]["sequence_length"]

        # Shared feature extractor applied to every frame of both members, and the pooling of its final feature map
        backbone_configuration = configuration["Backbone"]
        self.backbone = build_backbone(backbone_configuration=backbone_configuration,
                                       input_image_size=configuration["Data"]["image_settings"]["size"],
                                       device=device, dtype=dtype)
        self.global_pooling = build_global_pooling(global_pooling=self.head_configuration["global_pooling"])
        frame_features = list(backbone_configuration["convolutions"].values())[-1][1]

        # Shared summary of the frame vectors of a member, then the fusion of the two members and the class logits
        self.temporal_module = build_temporal_module(
            temporal_configuration=self.temporal_configuration, input_features=frame_features,
            maximum_length=self.sequence_length, keep_time_axis=False, device=device, dtype=dtype)
        self.member_fusion = MemberFusion(fusion_type=self.fusion_configuration["type"], frame_features=frame_features,
                                          member_features=self.temporal_module.output_features,
                                          number_heads=self.fusion_configuration["number_heads"],
                                          dropout_rate=self.fusion_configuration["dropout_rate"], device=device,
                                          dtype=dtype)
        self.classification_head = ClassificationHead(input_features=self.member_fusion.output_features,
                                                      fully_connected=self.head_configuration["fully_connected"],
                                                      dropout_rate=self.head_configuration["dropout_rate"],
                                                      number_classes=self.number_classes, device=device, dtype=dtype)

        # Weight of every class in the loss, uniform unless the trainer sets the balanced weights of its training split
        self.register_buffer(name="class_weights",
                             tensor=torch.ones(size=(self.number_classes,), device=device, dtype=dtype))

    def encode_members(self, before_images: torch.Tensor, after_images: torch.Tensor,
                       before_mask: Optional[torch.Tensor],
                       after_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Turns every frame of both members into a feature vector: the two members as one batch when masks are given,
        so that the batch normalization statistics are computed over the frames of both members, or member by member
        without masks, for members of different numbers of frames.

        Args:
            before_images (torch.Tensor): The [B, T1, 3, H, W] frames of the before members.
            after_images (torch.Tensor): The [B, T2, 3, H, W] frames of the after members.
            before_mask (Optional[torch.Tensor]): The [B, T1] booleans of the before members, or None.
            after_mask (Optional[torch.Tensor]): The [B, T2] booleans of the after members, or None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The [B, T1, F] and [B, T2, F] frame vectors, zero for the padding frames.

        Raises:
            ValueError: If the two members have different numbers of frames while masks are given.
        """
        # Every frame real: each member encoded on its own, whatever its number of frames
        if before_mask is None:
            before_features = encode_frames(backbone=self.backbone, global_pooling=self.global_pooling,
                                            input_tensor=before_images, sequence_mask=None)
            after_features = encode_frames(backbone=self.backbone, global_pooling=self.global_pooling,
                                           input_tensor=after_images, sequence_mask=None)
            return before_features, after_features

        # Both members padded to the same number of frames, encoded as one batch of sequences
        if before_images.shape[1] != after_images.shape[1]:
            raise ValueError(f"With sequence masks, both members must be padded to the same number of frames, got "
                             f"{before_images.shape[1]} and {after_images.shape[1]}.")
        frame_features = encode_frames(backbone=self.backbone, global_pooling=self.global_pooling,
                                       input_tensor=torch.cat(tensors=[before_images, after_images], dim=0),
                                       sequence_mask=torch.cat(tensors=[before_mask, after_mask], dim=0))
        before_features, after_features = frame_features.chunk(chunks=2, dim=0)
        return before_features, after_features

    def summarise_members(self, before_features: torch.Tensor, after_features: torch.Tensor,
                          before_mask: Optional[torch.Tensor],
                          after_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Summarises the frame vectors of every member into one vector with the shared temporal module: the two members
        as one batch when masks are given, member by member without masks.

        Args:
            before_features (torch.Tensor): The [B, T1, F] frame vectors of the before members.
            after_features (torch.Tensor): The [B, T2, F] frame vectors of the after members.
            before_mask (Optional[torch.Tensor]): The [B, T1] booleans of the before members, or None.
            after_mask (Optional[torch.Tensor]): The [B, T2] booleans of the after members, or None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The [B, F'] vectors of the before and after members.
        """
        # Every frame real: each member summarised on its own, whatever its number of frames
        if before_mask is None:
            return (self.temporal_module(frame_features=before_features, sequence_mask=None),
                    self.temporal_module(frame_features=after_features, sequence_mask=None))

        # Both members summarised as one batch of sequences
        member_features = torch.cat(tensors=[before_features, after_features], dim=0)
        member_vectors = self.temporal_module(frame_features=member_features,
                                              sequence_mask=torch.cat(tensors=[before_mask, after_mask], dim=0))
        before_vectors, after_vectors = member_vectors.chunk(chunks=2, dim=0)
        return before_vectors, after_vectors

    def forward(self, before_images: torch.Tensor, after_images: torch.Tensor,
                before_mask: Optional[torch.Tensor] = None, after_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Classifies a batch of pairs, and computes the classification loss when the labels are given.

        Args:
            before_images (torch.Tensor): The [B, T1, 3, H, W] float BGR frames of the before members in [0, 255].
            after_images (torch.Tensor): The [B, T2, 3, H, W] float BGR frames of the after members in [0, 255].
            before_mask (Optional[torch.Tensor]): The [B, T1] booleans, True for the real frames, which come first, or
                None when every frame is real (deployment).
            after_mask (Optional[torch.Tensor]): The [B, T2] booleans of the after members, given with before_mask.
            labels (Optional[torch.Tensor]): The [B] class indices, or None.

        Returns:
            Dict[str, Any]: The 'class_logits' and 'class_probabilities' [B, K], plus 'classification_loss' with labels.

        Raises:
            ValueError: If only one of the two masks is given.
        """
        if (before_mask is None) != (after_mask is None):
            raise ValueError("The masks of the before and after members must be given together, or both left out.")

        # Frame vectors of both members, matched across the members by the cross attention fusion, then summarised
        before_features, after_features = self.encode_members(before_images=before_images, after_images=after_images,
                                                              before_mask=before_mask, after_mask=after_mask)
        before_features, after_features = self.member_fusion.attend_across_members(
            before_features=before_features, after_features=after_features, before_mask=before_mask,
            after_mask=after_mask)
        before_vectors, after_vectors = self.summarise_members(before_features=before_features,
                                                               after_features=after_features,
                                                               before_mask=before_mask, after_mask=after_mask)

        # Fusion of the member vectors, then the class logits of the pairs
        pair_features = self.member_fusion.combine_member_vectors(before_vectors=before_vectors,
                                                                  after_vectors=after_vectors)
        class_logits = self.classification_head(features=pair_features)

        outputs = {"class_logits": class_logits,
                   "class_probabilities": convert_logits_to_probabilities(class_logits=class_logits,
                                                                          output_mode=self.output_mode)}
        if labels is not None:
            outputs["classification_loss"] = compute_classification_loss(class_logits=class_logits, labels=labels,
                                                                         output_mode=self.output_mode,
                                                                         label_smoothing=self.label_smoothing,
                                                                         class_weights=self.class_weights)
        return outputs

    def print_summary(self) -> None:
        """
        Prints the backbone summary, then the temporal module, the member fusion and the classification head.
        """
        self.backbone.print_summary()
        print_sequence_module_summary(title="Temporal Module", module_configuration=self.temporal_configuration,
                                      output_features=self.temporal_module.output_features, border_style="blue")
        print_sequence_module_summary(title="Member Fusion", module_configuration=self.fusion_configuration,
                                      output_features=self.member_fusion.output_features, border_style="magenta")
        print_classification_head_summary(head_configuration=self.head_configuration,
                                          number_classes=self.number_classes)

    def get_size_components(self) -> Dict[str, nn.Module]:
        """
        Lists the backbone, the temporal module, the member fusion, the classification head and the whole model for
        the model size summary.

        Returns:
            Dict[str, nn.Module]: The display name and module of every component.
        """
        return {"Backbone": self.backbone, "Temporal Module": self.temporal_module,
                "Member Fusion": self.member_fusion, "Classification Head": self.classification_head,
                "Total Model": self}
