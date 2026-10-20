"""
Model of the sequence classification task (many to one, M2O): the frame encoder, the temporal module and the
classification head of sequence_model.py give the class logits, probabilities and loss of every sequence. Registered in
model_registry.py for the "sequence_classification" task and wrapped by inference_sequence_model.py for the exports.
"""

import torch

from typing import Any, Dict, Optional

from vision_training.losses.classification_losses import compute_classification_loss
from vision_training.models.sequence.sequence_model import SequenceFrameModel


class SequenceClassifier(SequenceFrameModel):
    """
    Model of the sequence classification task (many to one): the shared Backbone and a global pooling turn every
    frame into a feature vector, the temporal module summarises the frame vectors of a sequence into one vector,
    ignoring the padding frames, and the classification head gives the logits of the classes of the sequence.

    Outputs: 'class_logits' [B, K] and 'class_probabilities' [B, K], plus 'classification_loss' when labels are given.
    """

    def __init__(self, configuration: Dict[str, Any], verbose: bool = False, device: torch.device = None,
                 dtype: torch.dtype = None) -> None:
        """
        Builds the backbone, the global pooling, the temporal module summarising every sequence into one vector and
        the classification head from the configuration.

        Args:
            configuration (Dict[str, Any]): The model configuration (Data with its label dictionary and sequence
                settings, Backbone, TemporalModel and ClassificationHead sections).
            verbose (bool): Unused, kept so that every task model is built with the same arguments.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(SequenceClassifier, self).__init__(configuration=configuration, keep_time_axis=False, device=device,
                                                 dtype=dtype)

    def forward(self, input_tensor: torch.Tensor, sequence_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Classifies a batch of sequences, and computes the classification loss when the labels are given.

        Args:
            input_tensor (torch.Tensor): The [B, T, 3, H, W] float BGR frames in [0, 255].
            sequence_mask (Optional[torch.Tensor]): The [B, T] booleans, True for the real frames, which come first, or
                None when every frame is real (deployment sequences).
            labels (Optional[torch.Tensor]): The [B] class indices, or None.

        Returns:
            Dict[str, Any]: The 'class_logits' and 'class_probabilities' [B, K], plus 'classification_loss' with labels.
        """
        outputs = self.compute_class_outputs(input_tensor=input_tensor, sequence_mask=sequence_mask)

        # Classification loss of every sequence when the labels are given
        if labels is not None:
            outputs["classification_loss"] = compute_classification_loss(class_logits=outputs["class_logits"],
                                                                         labels=labels, output_mode=self.output_mode,
                                                                         label_smoothing=self.label_smoothing,
                                                                         class_weights=self.class_weights)
        return outputs
