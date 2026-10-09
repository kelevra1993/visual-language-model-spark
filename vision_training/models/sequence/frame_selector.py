"""
Model of the frame selection task (SEL): the sequence labeller (frame encoder, temporal module keeping the time axis,
classification head giving the class probabilities of every frame) trained with the selection loss on top of the
per-frame loss, the frames of every wanted class being chosen afterwards by models/sequence/frame_selection_decoding.py.
Registered in model_registry.py for the "frame_selection" task and wrapped by inference_sequence_model.py for the
exports.
"""

import torch

from typing import Any, Dict, Optional

from vision_training.losses.sequence_losses import compute_frame_selection_loss
from vision_training.models.sequence.sequence_labeller import SequenceLabeller


class FrameSelector(SequenceLabeller):
    """
    Model of the frame selection task: the sequence labeller giving the probability of every class (the wanted classes
    and the unselected class) for every frame of a whole sequence, trained with the per-frame cross entropy plus the
    selection loss weighted by 'Selection.selection_loss_weight', which makes the frames of a sequence compete for
    every wanted class.

    The 'wanted_class_mask' buffer, set by the frame selection trainer from the unselected category of its manifests,
    marks the wanted classes with 1 and the unselected class with 0. Saved with the weights, it gives the unselected
    class to the exports and the local inference without the manifests.

    Outputs: 'class_logits' [B, T, K] and 'class_probabilities' [B, T, K], plus 'classification_loss' and
    'selection_loss' when labels are given.
    """

    def __init__(self, configuration: Dict[str, Any], verbose: bool = False, device: torch.device = None,
                 dtype: torch.dtype = None) -> None:
        """
        Builds the sequence labeller and reads the weight of the selection loss.

        Args:
            configuration (Dict[str, Any]): The model configuration (Data with its label dictionary and sequence
                settings, Backbone, TemporalModel, ClassificationHead and Selection sections).
            verbose (bool): Unused, kept so that every task model is built with the same arguments.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(FrameSelector, self).__init__(configuration=configuration, verbose=verbose, device=device, dtype=dtype)
        self.selection_loss_weight = configuration["Selection"]["selection_loss_weight"]

        # Wanted classes (1) and unselected class (0), every class being wanted until the trainer sets the unselected
        # one
        self.register_buffer(name="wanted_class_mask",
                             tensor=torch.ones(size=(self.number_classes,), device=device, dtype=dtype))

    def get_unselected_class_index(self) -> int:
        """
        Gives the class index of the unselected frames, saved with the weights in the wanted class mask, for the
        decoding of the exported and locally run models.

        Returns:
            int: The class index of the unselected frames.

        Raises:
            ValueError: If the mask does not mark exactly one unselected class, e.g. a model never set by the trainer.
        """
        unselected_class_indices = torch.nonzero(input=self.wanted_class_mask == 0).flatten().tolist()
        if len(unselected_class_indices) != 1:
            raise ValueError(f"The wanted class mask of the frame selector must mark exactly one unselected class, "
                             f"got {unselected_class_indices}.")
        return unselected_class_indices[0]

    def forward(self, input_tensor: torch.Tensor, sequence_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Classifies every frame of a batch of sequences, and computes the per-frame loss and the weighted selection loss
        when the labels are given.

        Args:
            input_tensor (torch.Tensor): The [B, T, 3, H, W] float BGR frames in [0, 255].
            sequence_mask (Optional[torch.Tensor]): The [B, T] booleans, True for the real frames, which come first, or
                None when every frame is real (deployment sequences).
            labels (Optional[torch.Tensor]): The [B, T] class index of every frame (-100 for the padding frames), or
                None.

        Returns:
            Dict[str, Any]: The 'class_logits' and 'class_probabilities' [B, T, K], plus 'classification_loss' and
                'selection_loss' with labels.
        """
        outputs = super(FrameSelector, self).forward(input_tensor=input_tensor, sequence_mask=sequence_mask,
                                                     labels=labels)

        # Selection loss, skipped when its weight is 0 so that only the per-frame loss trains the model
        if labels is not None:
            if self.selection_loss_weight > 0:
                outputs["selection_loss"] = self.selection_loss_weight * compute_frame_selection_loss(
                    class_logits=outputs["class_logits"], labels=labels, wanted_class_mask=self.wanted_class_mask)
            else:
                outputs["selection_loss"] = torch.zeros(size=(), device=outputs["class_logits"].device)
        return outputs
