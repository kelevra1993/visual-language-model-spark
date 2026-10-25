"""
Run of the trained semantic segmenter on the CPU for the figures of the semantic segmentation series of the tutorial
videos, the expensive step of tutorials/figures/semantic_segmentation/figures.py (about a minute on a busy machine),
reused between builds unless forced: the checkpoint of iteration 2000 of the synthetic segmentation run, on the test
image of the chapters with its mask, gives in one forward pass the skip connections of the encoder and the output of
every decoder level (kept as the mean absolute activation of their channels, through forward hooks), the class logits
and probabilities, the losses of the image, and the shapes and parameters of the model.
"""

import numpy
import torch

from pathlib import Path
from torch import nn
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import read_reused_facts, round_fact, write_reused_facts
from tutorials.figures.semantic_segmentation.segmentation_sources import CHECKPOINT_ITERATION
from tutorials.tutorial_builder.settings import THREAD_COUNT
from vision_training.models.model_loading import load_trained_model
from vision_training.trainers.checkpointing import get_checkpoint_path

# Files of the inference in the figures folder, the arrays and the facts that let the next build reuse them
INFERENCE_ARRAYS_NAME = "inference/segmentation_inference.npz"
INFERENCE_FACTS_NAME = "inference/segmentation_inference_facts.json"


class ActivationRecorder:
    """
    Forward hooks keeping, for the encoder and every decoder level of the segmenter, the shape of its output and the
    mean absolute activation of its channels, the maps the figure of the levels shows.

    Attributes:
        shapes (Dict[str, List[int]]): The output shape of every recorded part.
        maps (Dict[str, numpy.ndarray]): The [H, W] mean absolute activation of every recorded part.
    """

    def __init__(self) -> None:
        """
        Starts with nothing recorded.
        """
        self.shapes: Dict[str, List[int]] = {}
        self.maps: Dict[str, numpy.ndarray] = {}

    def attach(self, module: nn.Module, name: str) -> torch.utils.hooks.RemovableHandle:
        """
        Registers the hook on a module under a name.

        Args:
            module (nn.Module): The module whose output is recorded.
            name (str): The name of the part in the facts.

        Returns:
            torch.utils.hooks.RemovableHandle: The handle removing the hook.
        """
        module.recorded_name = name
        return module.register_forward_hook(hook=self.record)

    def keep(self, name: str, tensor: torch.Tensor) -> None:
        """
        Keeps the shape and the mean absolute activation map of one output tensor.

        Args:
            name (str): The name of the part.
            tensor (torch.Tensor): Its [1, C, H, W] output.
        """
        self.shapes[name] = list(tensor.shape)
        self.maps[name] = tensor[0].detach().abs().mean(dim=0).to(dtype=torch.float32).numpy()

    def record(self, module: nn.Module, inputs: Any, output: Any) -> None:
        """
        Keeps the output of a module: the final map and every skip connection of the encoder (its output with
        return_pre_pooling_outputs), or the output tensor of a decoder level.

        Args:
            module (nn.Module): The module that ran.
            inputs (Any): Its positional inputs.
            output (Any): Its output.
        """
        if isinstance(output, tuple):
            final_map, skip_tensors = output
            self.keep(name="bottleneck", tensor=final_map)
            for block_number, skip_tensor in skip_tensors.items():
                self.keep(name=f"skip_{block_number}", tensor=skip_tensor)
            return
        self.keep(name=module.recorded_name, tensor=output)


def run_segmentation_inference(output_folder: Path, run_folder: Path, sample_record: Dict[str, Any],
                               force: bool) -> Dict[str, Any]:
    """
    Runs the trained segmenter on the test image once, or reuses the previous run, and gives its facts: the shapes of
    the encoder outputs and of every decoder level, the parameters of every component, the receptive field of the
    encoder, the losses and the pixel accuracy of the image. The maps are saved in INFERENCE_ARRAYS_NAME.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the segmentation run.
        sample_record (Dict[str, Any]): The test image of load_test_sample.
        force (bool): Whether the inference runs again even when the previous build left its files.

    Returns:
        Dict[str, Any]: The facts of the inference.
    """
    arrays_path = output_folder / INFERENCE_ARRAYS_NAME
    facts_path = output_folder / INFERENCE_FACTS_NAME
    reused_facts = read_reused_facts(facts_path=facts_path, required_paths=[arrays_path], force=force)
    if reused_facts is not None:
        return reused_facts

    # Trained segmenter of the checkpoint on the CPU, its threads limited so that the trainings keep the processors
    torch.set_num_threads(THREAD_COUNT)
    checkpoint_path = get_checkpoint_path(experiment_folder=run_folder, iteration=CHECKPOINT_ITERATION)
    model = load_trained_model(configuration_path=run_folder / "training_configuration.yaml",
                               checkpoint_path=checkpoint_path, device=torch.device("cpu"), use_averaged_weights=False)
    recorder = ActivationRecorder()
    handles = [recorder.attach(module=model.backbone, name="encoder")]
    for level_key, convolution_block in model.segmentation_decoder.convolution_blocks.items():
        handles.append(recorder.attach(module=convolution_block, name=f"level_{level_key}"))

    # One forward pass with the mask: the logits, the probabilities of the evaluation mode and the losses
    sample = sample_record["sample"]
    with torch.no_grad():
        outputs = model(input_tensor=sample["images"][None].to(dtype=torch.float32), masks=sample["masks"][None])
    for handle in handles:
        handle.remove()
    class_map = outputs["class_logits"][0].argmax(dim=0)
    labelled = sample["masks"] >= 0

    # Maps and class map of the figures
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    numpy.savez_compressed(file=arrays_path, class_map=class_map.numpy().astype(numpy.uint8),
                           top_probability=outputs["class_probabilities"][0].max(dim=0).values.numpy().astype(
                               numpy.float16),
                           **{f"map_{name}": activation_map.astype(numpy.float16)
                              for name, activation_map in recorder.maps.items()})
    receptive_field = model.backbone.compute_receptive_field_information()[-1]["receptive_field"]
    facts = {"inference_shapes": recorder.shapes,
             "inference_parameters": {name: sum(parameter.numel() for parameter in component.parameters())
                                      for name, component in model.get_size_components().items()},
             "inference_receptive_field": int(receptive_field),
             "inference_cross_entropy": round_fact(value=outputs["segmentation_loss"]),
             "inference_dice_loss": round_fact(value=outputs["dice_loss"]),
             "inference_pixel_accuracy": round_fact(value=(class_map[labelled] == sample["masks"][labelled]).float()
                                                    .mean()),
             "inference_upsampling_layers": {level_key: str(layer) for level_key, layer in
                                             model.segmentation_decoder.upsampling_layers.items()},
             "inference_head": str(model.segmentation_head),
             "inference_shortcuts": {"backbone": [block.count_residual_additions()
                                                  for block in model.backbone.convolution_blocks],
                                     "decoder": [block.count_residual_additions() for block in
                                                 model.segmentation_decoder.convolution_blocks.values()]},
             "inference_checkpoint_iteration": CHECKPOINT_ITERATION}
    write_reused_facts(facts_path=facts_path, figure_paths=[arrays_path], facts=facts)
    return facts
