"""
Abstract base class of the model of every task (detector, image classifier, semantic segmenter, sequence models): the
backbone stored in 'self.backbone' and the architecture diagnostics printed by the trainer at the start of a run
(summary, receptive field and size of every component, with the auxiliary heads the trainer keeps outside the model).
Used by the model registry, the trainers, model loading and the exports.
"""

from abc import ABC, abstractmethod
from rich.console import Console
from rich.panel import Panel
from torch import nn
from typing import Dict, List, Optional, Tuple

# Name of the last line of the model size summary when the trainer has auxiliary heads: the model with its auxiliary
# heads, everything the training updates and saves in the checkpoints
TOTAL_WITH_AUXILIARY_HEADS_NAME = "Total With Auxiliary Heads"


def measure_modules(modules: List[nn.Module]) -> Tuple[int, int, int]:
    """
    Measures modules together for a line of the model size summary (TaskModel.print_model_size_summary): one component
    of the model, an auxiliary head of the trainer, or the model with its auxiliary heads.

    Args:
        modules (List[nn.Module]): The modules measured together, without shared tensors between them.

    Returns:
        Tuple[int, int, int]: The number of parameters, the number of trainable parameters, and the bytes of the
            parameters and buffers (e.g. batch normalization running statistics).
    """
    parameters = [parameter for module in modules for parameter in module.parameters()]
    buffers = [buffer for module in modules for buffer in module.buffers()]
    number_parameters = sum(parameter.numel() for parameter in parameters)
    number_trainable_parameters = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in parameters + buffers)
    return number_parameters, number_trainable_parameters, total_bytes


class TaskModel(nn.Module, ABC):
    """
    Base class of the networks of every task of the repository (object detector, image classifier, semantic
    segmenter, sequence models).

    Every task model is built around the shared Backbone, always stored in 'self.backbone' so that a backbone trained
    for one task can initialise another one, and its forward pass returns a dictionary whose keys ending in '_loss'
    are summed by the trainer into the total loss. This class provides the architecture diagnostics the trainer
    prints at the start of a run, the backbone receptive field and the size of every component, so that every task
    model gets them without copying them.
    """

    @abstractmethod
    def print_summary(self, auxiliary_heads: Optional[Dict[str, nn.Module]] = None) -> None:
        """
        Prints the architecture summary of the model at the start of a training run, so that the user can check the
        feature map sizes and head structures before the training starts, with the auxiliary heads the trainer of the
        task trains with the model (none for the current tasks).

        Args:
            auxiliary_heads (Optional[Dict[str, nn.Module]]): The auxiliary heads of the trainer keyed by name, built
                by its build_auxiliary_heads, empty or None without any.
        """

    @abstractmethod
    def get_size_components(self) -> Dict[str, nn.Module]:
        """
        Lists the components whose parameter count and memory size are reported by print_model_size_summary, the
        whole model being given last.

        Returns:
            Dict[str, nn.Module]: The display name and module of every component.
        """

    def get_detection_block_annotations(self) -> Dict[str, str]:
        """
        Gives, for the backbone blocks feeding task heads, an extra line printed below them in the receptive field
        summary (for instance the anchor scales assigned to a detection block). Models without such information keep
        this default, which annotates nothing.

        Returns:
            Dict[str, str]: The annotation of every annotated block, keyed by block number.
        """
        return {}

    def print_receptive_field_summary(self) -> None:
        """
        Prints the receptive field of every spatial layer of the Backbone, highlighting the blocks feeding the task
        heads with their annotation, if any.

        This diagnostic (enabled by 'diagnosis_modes.compute_receptive_field') shows, for each such block, the
        receptive field next to the information given by get_detection_block_annotations: for the object detector,
        the anchor scales assigned to the block, since anchors much larger than the receptive field cannot be fully
        observed by their feature map cell.
        """
        console = Console(width=100)

        receptive_field_information = self.backbone.compute_receptive_field_information()
        block_annotations = self.get_detection_block_annotations()
        input_image_size = self.backbone.input_image_size

        summary_text = f"[bold]Input Image Size : {input_image_size}x{input_image_size}[/bold]\n\n"

        # Describe every spatial layer, adding the annotation below each annotated block output
        previous_block_number = None
        for layer_information in receptive_field_information:
            # Separate consecutive blocks with an empty line so that each block stands out
            if previous_block_number is not None and layer_information["block_number"] != previous_block_number:
                summary_text += "\n"
            previous_block_number = layer_information["block_number"]

            output_size = layer_information["output_size"]
            layer_text = (f"Block {layer_information['block_number']} - {layer_information['name']:<14} : "
                          f"Output {output_size}x{output_size}x{layer_information['output_channels']}"
                          f" | Stride {layer_information['stride']}"
                          f" | Receptive Field {layer_information['receptive_field']}")

            if layer_information["is_detection_block_output"]:
                block_string = str(layer_information["block_number"])
                summary_text += f"[bold green]{layer_text}[/bold green]\n"
                if block_string in block_annotations:
                    summary_text += f"[bold green]  - {block_annotations[block_string]}[/bold green]\n"
            else:
                summary_text += f"{layer_text}\n"

        # Squeeze-and-Excitation pools over the whole feature map, which the convolutional receptive field ignores
        if self.backbone.squeeze_and_excite:
            summary_text += ("\n[bold yellow]Squeeze-and-Excitation adds global image context.\n"
                             "This is not captured by the convolutional receptive field above.[/bold yellow]")

        panel = Panel(summary_text.rstrip("\n"), title="[bold]Backbone Receptive Field[/bold]", border_style="magenta")
        console.print(panel)

    def print_model_size_summary(self, auxiliary_heads: Optional[Dict[str, nn.Module]] = None) -> None:
        """
        Prints the parameter count and memory size of every component of the model, then of every auxiliary head of
        the trainer and of the model with its auxiliary heads: the heads are trained and saved in the checkpoints with
        the model, but never exported.

        This diagnostic (enabled by 'diagnosis_modes.compute_model_size') reports the size of every component given by
        get_size_components. Sizes include parameters and buffers (e.g. batch normalization running statistics) and
        are expressed in megabytes (1 MB = 1e6 bytes).

        Args:
            auxiliary_heads (Optional[Dict[str, nn.Module]]): The auxiliary heads of the trainer keyed by name (e.g.
                'auxiliary_projector', shown as 'Auxiliary Projector'), empty or None without any.
        """
        console = Console(width=60)

        # Modules measured on every line: the components of the model, then the auxiliary heads and the model with
        # them, only when the trainer has auxiliary heads
        measured_modules = {component_name: [component]
                            for component_name, component in self.get_size_components().items()}
        if auxiliary_heads:
            for head_name, auxiliary_head in auxiliary_heads.items():
                measured_modules[head_name.replace("_", " ").title()] = [auxiliary_head]
            measured_modules[TOTAL_WITH_AUXILIARY_HEADS_NAME] = [self, *auxiliary_heads.values()]

        summary_text = ""

        # Measure every line by summing the element counts and byte sizes of the tensors of its modules
        for component_name, modules in measured_modules.items():
            number_parameters, number_trainable_parameters, total_bytes = measure_modules(modules=modules)

            summary_text += f"[bold]{component_name}[/bold]\n"
            summary_text += f"  - Parameters           : {number_parameters:,}\n"
            summary_text += f"  - Trainable Parameters : {number_trainable_parameters:,}\n"
            summary_text += f"  - Estimated Size       : {total_bytes / 1e6:.2f} MB\n\n"

        panel = Panel(summary_text.rstrip("\n"), title="[bold]Model Size[/bold]", border_style="cyan")
        console.print(panel)
