import torch

from torch import nn
from typing import Any, Dict, Optional

from vision_training.losses.classification_losses import compute_classification_loss, convert_logits_to_probabilities
from vision_training.models.backbones.backbone import build_backbone
from vision_training.models.classification.classification_head import (ClassificationHead, build_global_pooling,
                                                                       print_classification_head_summary)
from vision_training.models.sequence.frame_encoder import encode_frames
from vision_training.models.sequence.temporal_modules import (build_temporal_module,
                                                              print_temporal_module_summary)
from vision_training.models.task_model import TaskModel


class SequenceClassifier(TaskModel):
    """
    Model of the sequence classification task (many to one): the shared Backbone and a global pooling turn every
    frame into a feature vector, the temporal module summarises the frame vectors of a sequence into one vector,
    ignoring the padding frames, and the classification head gives the logits of the classes of the sequence.

    Outputs: 'class_logits' [B, K] and 'class_probabilities' [B, K], plus 'classification_loss' when labels are given.
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
        super(SequenceClassifier, self).__init__()
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

        # Summary of the frame vectors of a sequence, then the class logits
        self.temporal_module = build_temporal_module(
            temporal_configuration=self.temporal_configuration, input_features=frame_features,
            maximum_length=self.sequence_length, keep_time_axis=False,
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
        Classifies a batch of sequences, and computes the classification loss when the labels are given.

        Args:
            input_tensor (torch.Tensor): The [B, T, 3, H, W] float BGR frames in [0, 255].
            sequence_mask (Optional[torch.Tensor]): The [B, T] booleans, True for the real frames, which come first, or
                None when every frame is real (deployment windows).
            labels (Optional[torch.Tensor]): The [B] class indices, or None.

        Returns:
            Dict[str, Any]: The 'class_logits' and 'class_probabilities' [B, K], plus 'classification_loss' with labels.
        """
        frame_features = encode_frames(backbone=self.backbone, global_pooling=self.global_pooling,
                                       input_tensor=input_tensor, sequence_mask=sequence_mask)
        sequence_features = self.temporal_module(frame_features=frame_features, sequence_mask=sequence_mask)
        class_logits = self.classification_head(features=sequence_features)

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
        Prints the backbone summary, then the temporal module and the classification head.
        """
        self.backbone.print_summary()
        print_temporal_module_summary(temporal_configuration=self.temporal_configuration,
                                      output_features=self.temporal_module.output_features)
        print_classification_head_summary(head_configuration=self.head_configuration,
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
