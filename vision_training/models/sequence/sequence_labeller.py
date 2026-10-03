"""
Model of the sequence labelling task (many to many, M2M): the frame encoder, a temporal module keeping the time axis
and the classification head give the class logits, probabilities and loss of every frame. Registered in
model_registry.py for the "sequence_labelling" task and wrapped by inference_sequence_model.py for the exports.
"""

import torch

from torch import nn
from typing import Any, Dict, Optional

from vision_training.losses.classification_losses import convert_logits_to_probabilities
from vision_training.losses.sequence_losses import compute_sequence_labelling_loss
from vision_training.models.backbones.backbone import build_backbone
from vision_training.models.classification.classification_head import (ClassificationHead, build_global_pooling,
                                                                       print_classification_head_summary)
from vision_training.models.sequence.frame_encoder import encode_frames
from vision_training.models.sequence.temporal_modules import build_temporal_module, print_sequence_module_summary
from vision_training.models.task_model import TaskModel


class SequenceLabeller(TaskModel):
    """
    Model of the sequence labelling task (many to many): the shared Backbone and a global pooling turn every frame
    into a feature vector, the temporal module relates every frame to the frames it may see while keeping the time
    axis (only the past frames when causal), and the classification head gives the class logits of every frame.

    Outputs: 'class_logits' [B, T, K] and 'class_probabilities' [B, T, K], plus 'classification_loss' when labels are
    given.
    """

    def __init__(self, configuration: Dict[str, Any], verbose: bool = False, device: torch.device = None,
                 dtype: torch.dtype = None) -> None:
        """
        Builds the backbone, the global pooling, the temporal module and the classification head from the
        configuration.

        Args:
            configuration (Dict[str, Any]): The model configuration (Data with its label dictionary and sequence
                settings, Backbone, TemporalModel and ClassificationHead sections).
            verbose (bool): Unused, kept so that every task model is built with the same arguments.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(SequenceLabeller, self).__init__()
        self.device = device
        self.dtype = dtype
        self.head_configuration = configuration["ClassificationHead"]
        self.temporal_configuration = configuration["TemporalModel"]
        self.output_mode = self.head_configuration["output_mode"]
        self.label_smoothing = self.head_configuration["label_smoothing"]
        self.number_classes = len(set(configuration["Data"]["label_dictionary"].values()))
        self.sequence_length = configuration["Data"]["sequence_settings"]["sequence_length"]

        # Shared feature extractor applied to every frame, and the pooling of its final feature map
        backbone_configuration = configuration["Backbone"]
        self.backbone = build_backbone(backbone_configuration=backbone_configuration,
                                       input_image_size=configuration["Data"]["image_settings"]["size"],
                                       device=device, dtype=dtype)
        self.global_pooling = build_global_pooling(global_pooling=self.head_configuration["global_pooling"])
        frame_features = list(backbone_configuration["convolutions"].values())[-1][1]

        # Frame vectors in their temporal context, then the class logits of every frame
        self.temporal_module = build_temporal_module(
            temporal_configuration=self.temporal_configuration, input_features=frame_features,
            maximum_length=self.sequence_length, keep_time_axis=True,
            device=device, dtype=dtype)
        self.classification_head = ClassificationHead(input_features=self.temporal_module.output_features,
                                                      fully_connected=self.head_configuration["fully_connected"],
                                                      dropout_rate=self.head_configuration["dropout_rate"],
                                                      number_classes=self.number_classes, device=device, dtype=dtype)

        # Weight of every class in the loss, uniform unless the trainer sets the balanced weights of its training split
        self.register_buffer(name="class_weights",
                             tensor=torch.ones(size=(self.number_classes,), device=device, dtype=dtype))

    def forward(self, input_tensor: torch.Tensor, sequence_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Classifies every frame of a batch of sequences, and computes the loss over the labelled frames when the labels
        are given.

        Args:
            input_tensor (torch.Tensor): The [B, T, 3, H, W] float BGR frames in [0, 255].
            sequence_mask (Optional[torch.Tensor]): The [B, T] booleans, True for the real frames, which come first, or
                None when every frame is real (deployment windows).
            labels (Optional[torch.Tensor]): The [B, T] class index of every frame (-100 for the frames without class),
                or None.

        Returns:
            Dict[str, Any]: The 'class_logits' and 'class_probabilities' [B, T, K], plus 'classification_loss' with
                labels.
        """
        frame_features = encode_frames(backbone=self.backbone, global_pooling=self.global_pooling,
                                       input_tensor=input_tensor, sequence_mask=sequence_mask)
        contextual_features = self.temporal_module(frame_features=frame_features, sequence_mask=sequence_mask)
        class_logits = self.classification_head(features=contextual_features)

        outputs = {"class_logits": class_logits,
                   "class_probabilities": convert_logits_to_probabilities(class_logits=class_logits,
                                                                          output_mode=self.output_mode)}
        if labels is not None:
            outputs["classification_loss"] = compute_sequence_labelling_loss(class_logits=class_logits, labels=labels,
                                                                             label_smoothing=self.label_smoothing,
                                                                             class_weights=self.class_weights)
        return outputs

    def print_summary(self) -> None:
        """
        Prints the backbone summary, then the temporal module and the classification head applied to every frame.
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
