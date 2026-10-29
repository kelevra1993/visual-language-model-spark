"""
Model of the sequence labelling task (many to many, M2M): the frame encoder, a temporal module keeping the time axis
and the classification head of sequence_model.py give the class logits, probabilities and loss of every frame.
Registered in model_registry.py for the "sequence_labelling" task, extended by frame_selector.py and wrapped by
inference_sequence_model.py for the exports.
"""

import torch

from typing import Any, Dict, Optional

from vision_training.losses.sequence_losses import compute_sequence_labelling_loss
from vision_training.models.sequence.sequence_model import SequenceFrameModel


class SequenceLabeller(SequenceFrameModel):
    """
    Model of the sequence labelling task (many to many): the shared Backbone and a global pooling turn every frame
    into a feature vector, the temporal module relates every frame to the frames it may see while keeping the time
    axis (only the past frames when causal), and the classification head gives the class logits of every frame.

    Outputs: 'class_logits' [B, T, K] and 'class_probabilities' [B, T, K], plus 'classification_loss' when labels are
    given.
    """

    def __init__(self, configuration: Dict[str, Any], device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        Builds the backbone, the global pooling, the temporal module keeping one vector per frame and the
        classification head from the configuration.

        Args:
            configuration (Dict[str, Any]): The model configuration (Data with its label dictionary and sequence
                settings, Backbone, TemporalModel and ClassificationHead sections).
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(SequenceLabeller, self).__init__(configuration=configuration, keep_time_axis=True, device=device,
                                               dtype=dtype)

    def forward(self, input_tensor: torch.Tensor, sequence_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Classifies every frame of a batch of sequences, and computes the loss over the labelled frames when the labels
        are given.

        Args:
            input_tensor (torch.Tensor): The [B, T, 3, H, W] float BGR frames in [0, 255].
            sequence_mask (Optional[torch.Tensor]): The [B, T] booleans, True for the real frames, which come first, or
                None when every frame is real (deployment sequences).
            labels (Optional[torch.Tensor]): The [B, T] class index of every frame (-100 for the frames without class),
                or None.

        Returns:
            Dict[str, Any]: The 'class_logits' and 'class_probabilities' [B, T, K], plus 'classification_loss' with
                labels.
        """
        outputs = self.compute_class_outputs(input_tensor=input_tensor, sequence_mask=sequence_mask)

        # Classification loss over the labelled frames when the labels are given
        if labels is not None:
            outputs["classification_loss"] = compute_sequence_labelling_loss(class_logits=outputs["class_logits"],
                                                                             labels=labels,
                                                                             label_smoothing=self.label_smoothing,
                                                                             class_weights=self.class_weights)
        return outputs
