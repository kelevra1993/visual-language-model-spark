"""
Model of the image classification task: ImageClassifier chains the shared Backbone, a global pooling (or the flattening
of the feature map) and the classification head, and returns the class logits, probabilities and losses: the
classification loss, and the decorrelation loss of one hidden layer of the head when 'ClassificationHead.decorrelation'
is on. Registered in model_registry.py for the "classification" task and wrapped by inference_classifier.py for the
exports.
"""

import torch

from torch import nn
from typing import Any, Dict, Optional

from vision_training.losses.classification_losses import (compute_classification_loss, compute_decorrelation_loss,
                                                          convert_logits_to_probabilities)
from vision_training.models.backbones.backbone import build_backbone
from vision_training.models.classification.classification_head import (ClassificationHead, build_global_pooling,
                                                                       compute_pooled_features,
                                                                       print_classification_head_summary)
from vision_training.models.task_model import TaskModel


class ImageClassifier(TaskModel):
    """
    Model of the classification task: the shared Backbone extracts the final feature map of the image, a global
    pooling reduces it to one feature vector (or the map is flattened, keeping where the features are), and the
    classification head gives the logits of the classes. It predicts one class per image (softmax) or any number of
    classes (sigmoid), depending on 'ClassificationHead.output_mode'.

    Outputs: 'class_logits' [B, K] and 'class_probabilities' [B, K], plus 'classification_loss' when labels are given,
    and 'decorrelation_loss' with the tracked 'hidden_feature_correlation' when the decorrelation is also on.
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

        # Decorrelation of the features of one hidden layer of the head, added to the loss when the labels are given
        self.decorrelation_configuration = self.head_configuration["decorrelation"]
        self.number_classes = len(set(configuration["Data"]["label_dictionary"].values()))

        # Shared feature extractor, whose final feature map gives the number of features of the pooled vector: its
        # channels for the poolings, its channels times its height and width when flattened
        self.backbone = build_backbone(backbone_configuration=configuration["Backbone"],
                                       input_image_size=configuration["Data"]["image_settings"]["size"],
                                       device=device, dtype=dtype)
        pooled_features = compute_pooled_features(global_pooling=self.head_configuration["global_pooling"],
                                                  backbone=self.backbone)

        # Global pooling (or flattening) of the final feature map, then the class logits
        self.global_pooling = build_global_pooling(global_pooling=self.head_configuration["global_pooling"])
        self.classification_head = ClassificationHead(input_features=pooled_features,
                                                      fully_connected=self.head_configuration["fully_connected"],
                                                      fully_connected_normalization=self.head_configuration[
                                                          "fully_connected_normalization"],
                                                      batch_normalization_momentum=self.head_configuration[
                                                          "batch_normalization_momentum"],
                                                      dropout_rate=self.head_configuration["dropout_rate"],
                                                      number_classes=self.number_classes, device=device, dtype=dtype)

        # Weight of every class in the loss, uniform unless the trainer sets the balanced weights of its training split
        self.register_buffer(name="class_weights",
                             tensor=torch.ones(size=(self.number_classes,), device=device, dtype=dtype))

    def forward(self, input_tensor: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Classifies a batch of images, and computes the classification loss when the labels are given, with the
        decorrelation loss of the features of one hidden layer of the head when 'ClassificationHead.decorrelation' is
        on (training and validation batches; the predictions without labels never compute it).

        Args:
            input_tensor (torch.Tensor): The [B, 3, H, W] float BGR images in [0, 255].
            labels (Optional[torch.Tensor]): The [B] class indices or [B, K] multi-hot targets, or None.

        Returns:
            Dict[str, Any]: The 'class_logits' and 'class_probabilities' [B, K], plus 'classification_loss' with labels,
                and 'decorrelation_loss' and 'hidden_feature_correlation' with labels and the decorrelation on.
        """
        final_feature_map, _ = self.backbone(input_tensor=input_tensor)
        pooled_features = self.global_pooling(input=final_feature_map)

        # Keep the features of the decorrelated hidden layer, before its ReLU, only when its loss is computed
        decorrelation_on = labels is not None and self.decorrelation_configuration["activated"]
        if decorrelation_on:
            class_logits, hidden_features = self.classification_head.compute_logits(
                features=pooled_features, kept_hidden_layer=self.decorrelation_configuration["hidden_layer"])
        else:
            class_logits = self.classification_head(features=pooled_features)

        outputs = {"class_logits": class_logits,
                   "class_probabilities": convert_logits_to_probabilities(class_logits=class_logits,
                                                                          output_mode=self.output_mode)}
        if labels is not None:
            outputs["classification_loss"] = compute_classification_loss(class_logits=class_logits, labels=labels,
                                                                         output_mode=self.output_mode,
                                                                         label_smoothing=self.label_smoothing,
                                                                         class_weights=self.class_weights)

        # Decorrelation loss of the kept hidden features, summed into the total loss with the classification loss
        if decorrelation_on:
            outputs.update(compute_decorrelation_loss(hidden_features=hidden_features,
                                                      weight=self.decorrelation_configuration["weight"]))
        return outputs

    def print_summary(self, auxiliary_heads: Optional[Dict[str, nn.Module]] = None) -> None:
        """
        Prints the backbone summary, then the classification head: pooling, size of the pooled or flattened vector,
        hidden layers, dropout, output mode and number of classes.

        Args:
            auxiliary_heads (Optional[Dict[str, nn.Module]]): Unused, the trainer of the task having no auxiliary
                head; kept so that every task model prints its summary with the same arguments.
        """
        self.backbone.print_summary()
        print_classification_head_summary(head_configuration=self.head_configuration,
                                          input_features=self.classification_head.input_features,
                                          number_classes=self.number_classes)

    def get_size_components(self) -> Dict[str, nn.Module]:
        """
        Lists the backbone, the classification head and the whole classifier for the model size summary.

        Returns:
            Dict[str, nn.Module]: The display name and module of every component.
        """
        return {"Backbone": self.backbone, "Classification Head": self.classification_head, "Total Model": self}
