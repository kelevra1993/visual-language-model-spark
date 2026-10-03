"""
Model of the image classification task: ImageClassifier chains the shared Backbone, a global pooling and the
classification head, and returns the class logits, probabilities and loss. Registered in model_registry.py for the
"classification" task and wrapped by inference_classifier.py for the exports.
"""

import torch

from torch import nn
from typing import Any, Dict, Optional

from vision_training.losses.classification_losses import compute_classification_loss, convert_logits_to_probabilities
from vision_training.models.backbones.backbone import build_backbone
from vision_training.models.classification.classification_head import (ClassificationHead, build_global_pooling,
                                                                       print_classification_head_summary)
from vision_training.models.task_model import TaskModel


class ImageClassifier(TaskModel):
    """
    Model of the classification task: the shared Backbone extracts the final feature map of the image, a global
    pooling reduces it to one feature vector, and the classification head gives the logits of the classes. It
    predicts one class per image (softmax) or any number of classes (sigmoid), depending on
    'ClassificationHead.output_mode'.

    Outputs: 'class_logits' [B, K] and 'class_probabilities' [B, K], plus 'classification_loss' when labels are given.
    """

    def __init__(self, configuration: Dict[str, Any], verbose: bool = False, device: torch.device = None,
                 dtype: torch.dtype = None) -> None:
        """
        Builds the backbone, the global pooling and the classification head from the configuration.

        Args:
            configuration (Dict[str, Any]): The model configuration (Data with its label dictionary, Backbone and
                ClassificationHead sections).
            verbose (bool): Unused, kept so that every task model is built with the same arguments.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(ImageClassifier, self).__init__()
        self.device = device
        self.dtype = dtype
        self.head_configuration = configuration["ClassificationHead"]
        self.output_mode = self.head_configuration["output_mode"]
        self.label_smoothing = self.head_configuration["label_smoothing"]
        self.number_classes = len(set(configuration["Data"]["label_dictionary"].values()))

        # Shared feature extractor, whose last block gives the number of features of the pooled vector
        backbone_configuration = configuration["Backbone"]
        self.backbone = build_backbone(backbone_configuration=backbone_configuration,
                                       input_image_size=configuration["Data"]["image_settings"]["size"],
                                       device=device, dtype=dtype)
        final_channels = list(backbone_configuration["convolutions"].values())[-1][1]

        # Global pooling of the final feature map, then the class logits
        self.global_pooling = build_global_pooling(global_pooling=self.head_configuration["global_pooling"])
        self.classification_head = ClassificationHead(input_features=final_channels,
                                                      fully_connected=self.head_configuration["fully_connected"],
                                                      dropout_rate=self.head_configuration["dropout_rate"],
                                                      number_classes=self.number_classes, device=device, dtype=dtype)

        # Weight of every class in the loss, uniform unless the trainer sets the balanced weights of its training split
        self.register_buffer(name="class_weights",
                             tensor=torch.ones(size=(self.number_classes,), device=device, dtype=dtype))

    def forward(self, input_tensor: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Classifies a batch of images, and computes the classification loss when the labels are given.

        Args:
            input_tensor (torch.Tensor): The [B, 3, H, W] float BGR images in [0, 255].
            labels (Optional[torch.Tensor]): The [B] class indices or [B, K] multi-hot targets, or None.

        Returns:
            Dict[str, Any]: The 'class_logits' and 'class_probabilities' [B, K], plus 'classification_loss' with labels.
        """
        final_feature_map, _ = self.backbone(input_tensor=input_tensor)
        class_logits = self.classification_head(features=self.global_pooling(input=final_feature_map))

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
        Prints the backbone summary, then the classification head: pooling, hidden layers, dropout, output mode and
        number of classes.
        """
        self.backbone.print_summary()
        print_classification_head_summary(head_configuration=self.head_configuration,
                                          number_classes=self.number_classes)

    def get_size_components(self) -> Dict[str, nn.Module]:
        """
        Lists the backbone, the classification head and the whole classifier for the model size summary.

        Returns:
            Dict[str, nn.Module]: The display name and module of every component.
        """
        return {"Backbone": self.backbone, "Classification Head": self.classification_head, "Total Model": self}
