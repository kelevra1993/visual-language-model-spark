"""
Shared base of the sequence classifier (many to one, M2O) and of the sequence labeller (many to many, M2M, and the
frame selector built on it): the frame encoder, the temporal module and the classification head giving the class
logits of every sequence or of every frame. The two models only differ by the time axis kept by the temporal module and
by their loss, defined in sequence_classifier.py and sequence_labeller.py.
"""

import torch

from torch import nn
from typing import Any, Dict, Optional

from vision_training.losses.classification_losses import convert_logits_to_probabilities
from vision_training.models.backbones.backbone import build_backbone
from vision_training.models.classification.classification_head import (ClassificationHead, build_global_pooling,
                                                                       compute_pooled_features,
                                                                       print_classification_head_summary)
from vision_training.models.components.convolution_block import BATCH_NORMALIZATION_MOMENTUM
from vision_training.models.sequence.frame_encoder import encode_frames
from vision_training.models.sequence.temporal_modules import build_temporal_module, print_sequence_module_summary
from vision_training.models.task_model import TaskModel


class SequenceFrameModel(TaskModel):
    """
    Base of the sequence models classifying the frames of a sequence: the shared Backbone and a global pooling turn
    every frame into a feature vector, the temporal module summarises the frame vectors of a sequence into one vector
    or relates every frame to the frames it may see while keeping the time axis, and the classification head gives the
    class logits of the sequence or of every frame. The subclasses give the time axis and compute their loss in their
    forward pass.
    """

    def __init__(self, configuration: Dict[str, Any], keep_time_axis: bool, device: torch.device = None,
                 dtype: torch.dtype = None) -> None:
        """
        Builds the backbone, the global pooling, the temporal module and the classification head from the
        configuration, with the same attribute names for both subclasses so that their checkpoints keep their keys.

        Args:
            configuration (Dict[str, Any]): The model configuration (Data with its label dictionary and sequence
                settings, Backbone, TemporalModel and ClassificationHead sections).
            keep_time_axis (bool): Whether the temporal module keeps one vector per frame (sequence labeller) instead of
                one vector per sequence (sequence classifier).
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(SequenceFrameModel, self).__init__()
        self.device = device
        self.dtype = dtype
        self.keep_time_axis = keep_time_axis
        self.head_configuration = configuration["ClassificationHead"]
        self.temporal_configuration = configuration["TemporalModel"]
        self.output_mode = self.head_configuration["output_mode"]
        self.label_smoothing = self.head_configuration["label_smoothing"]
        self.number_classes = len(set(configuration["Data"]["label_dictionary"].values()))
        self.maximum_sequence_length = configuration["Data"]["sequence_settings"]["maximum_sequence_length"]

        # Shared feature extractor applied to every frame, and the pooling of its final feature map
        backbone_configuration = configuration["Backbone"]
        self.backbone = build_backbone(backbone_configuration=backbone_configuration,
                                       input_image_size=configuration["Data"]["image_settings"]["size"],
                                       device=device, dtype=dtype)
        self.global_pooling = build_global_pooling(global_pooling=self.head_configuration["global_pooling"])
        frame_features = compute_pooled_features(global_pooling=self.head_configuration["global_pooling"],
                                                 backbone=self.backbone)

        # Summary of the frame vectors of a sequence, or the frame vectors in their temporal context, then the logits
        self.temporal_module = build_temporal_module(
            temporal_configuration=self.temporal_configuration, input_features=frame_features,
            maximum_length=self.maximum_sequence_length, keep_time_axis=keep_time_axis, device=device, dtype=dtype)
        self.classification_head = ClassificationHead(input_features=self.temporal_module.output_features,
                                                      fully_connected=self.head_configuration["fully_connected"],
                                                      fully_connected_normalization=self.head_configuration[
                                                          "fully_connected_normalization"],
                                                      batch_normalization_momentum=BATCH_NORMALIZATION_MOMENTUM,
                                                      dropout_rate=self.head_configuration["dropout_rate"],
                                                      number_classes=self.number_classes, device=device, dtype=dtype)

        # Weight of every class in the loss, uniform unless the trainer sets the balanced weights of its training split
        self.register_buffer(name="class_weights",
                             tensor=torch.ones(size=(self.number_classes,), device=device, dtype=dtype))

    def compute_class_outputs(self, input_tensor: torch.Tensor,
                              sequence_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Encodes the frames of a batch of sequences, passes them through the temporal module and the classification
        head, and gives the class logits and probabilities, to which the forward pass of the subclasses adds the loss.

        Args:
            input_tensor (torch.Tensor): The [B, T, 3, H, W] float BGR frames in [0, 255].
            sequence_mask (Optional[torch.Tensor]): The [B, T] booleans, True for the real frames, which come first, or
                None when every frame is real (deployment sequences).

        Returns:
            Dict[str, torch.Tensor]: The 'class_logits' and 'class_probabilities', [B, K] for the sequence classifier
                and [B, T, K] for the sequence labeller.
        """
        frame_features = encode_frames(backbone=self.backbone, global_pooling=self.global_pooling,
                                       input_tensor=input_tensor, sequence_mask=sequence_mask)
        temporal_features = self.temporal_module(frame_features=frame_features, sequence_mask=sequence_mask)

        # The padding frames are left out of the batch statistics of the head only when one vector per frame is kept
        class_logits = self.classification_head(features=temporal_features,
                                                feature_mask=sequence_mask if self.keep_time_axis else None)
        return {"class_logits": class_logits,
                "class_probabilities": convert_logits_to_probabilities(class_logits=class_logits,
                                                                       output_mode=self.output_mode)}

    def print_summary(self) -> None:
        """
        Prints the backbone summary, then the temporal module and the classification head.
        """
        self.backbone.print_summary()
        print_sequence_module_summary(title="Temporal Module", module_configuration=self.temporal_configuration,
                                      output_features=self.temporal_module.output_features, border_style="blue")
        print_classification_head_summary(head_configuration=self.head_configuration,
                                          input_features=self.classification_head.input_features,
                                          number_classes=self.number_classes)

    def get_size_components(self) -> Dict[str, nn.Module]:
        """
        Lists the backbone, the temporal module, the classification head and the whole model for the model size
        summary.

        Returns:
            Dict[str, nn.Module]: The display name and module of every component.
        """
        return {"Backbone": self.backbone, "Temporal Module": self.temporal_module,
                "Classification Head": self.classification_head, "Total Model": self}
