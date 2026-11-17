"""
Console of a backbone deeper than its source backbone, for the transfer scene of the foundations chapter on the
backbone: the trained classification run of tutorial_sources.yaml is loaded into the same backbone with more blocks
by the real transfer code of the repository (load_pretrained_component, count_loaded_backbone_blocks), on the CPU and
without any training, and the messages it prints at launch, with the pretrained backbone block of the experiment
information (BaseTrainer.print_pretrained_components and its Loaded Blocks line), are written as a text listing that
the storyboard shows (figures:foundations/listings/backbone_transfer_console.txt). The facts give the blocks of the
source and of the deeper backbone, so that the narration and the drawing follow the run.
"""

import copy
import types

from pathlib import Path
from typing import Any, Dict

import torch

from tutorials.figures.foundations.listings import capture_printed_text
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.models.backbones.backbone import build_backbone
from vision_training.models.model_transfer import count_loaded_backbone_blocks, load_pretrained_component
from vision_training.trainers.base_trainer import BaseTrainer
from vision_training.utilities.os_utilities import INVISIBLE_SEQUENCE_PATTERN

# Blocks added after those of the classification run, each a copy of its last block, so that the deeper backbone
# starts with every block of the run, like the run of 6 blocks started from a run of 4 of
# tests/trainers/test_backbone_transfer_training.py
ADDED_BLOCK_COUNT = 2

# Learning rate multiplier of the loaded blocks shown, fine-tuning them ten times slower than the new layers, as the
# example of the Transfer Between Runs section of the README
LEARNING_RATE_MULTIPLIER = 0.1


def make_backbone_transfer_listing(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Loads the backbone of the classification run into a copy of its backbone with ADDED_BLOCK_COUNT more blocks, as a
    new run with Backbone.pretrained naming the run would at launch, and writes the console it prints into the
    listings folder of the foundations figures, for the transfer scene of the backbone chapter.

    Args:
        output_folder (Path): The figures folder of the foundations series.
        run_folder (Path): The folder of the classification run of tutorial_sources.yaml, the source run.

    Returns:
        Dict[str, Any]: The facts: the blocks of the source backbone, of the deeper backbone and the loaded ones, the
            first and last random blocks, and the learning rate multiplier of the loaded blocks.
    """
    listings_folder = output_folder / "listings"
    listings_folder.mkdir(parents=True, exist_ok=True)

    # Configuration of the deeper run: the saved configuration of the source run with its last block repeated, its
    # backbone pretrained from the source run and fine-tuned at the multiplier
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=run_folder / "training_configuration.yaml")
    deeper_configuration = copy.deepcopy(model_configuration)
    convolutions = deeper_configuration["Backbone"]["convolutions"]
    source_block_count = len(convolutions)
    last_block = list(convolutions.values())[-1]
    for added_index in range(ADDED_BLOCK_COUNT):
        convolutions[str(source_block_count + added_index + 1)] = list(last_block)
    deeper_configuration["Backbone"]["pretrained"].update({
        "experiment_folder": str(run_folder), "iteration": None, "freeze": False,
        "freeze_normalization_parameters": False, "learning_rate_multiplier": LEARNING_RATE_MULTIPLIER})

    # Backbone of the deeper run built on the CPU and loaded by the transfer code, its launch messages captured
    backbone = build_backbone(backbone_configuration=deeper_configuration["Backbone"],
                              input_image_size=deeper_configuration["Data"]["image_settings"]["size"],
                              device=torch.device(device="cpu"), dtype=torch.float32)
    loading_text = capture_printed_text(printing_function=lambda: load_pretrained_component(
        component=backbone, section_name="Backbone", model_configuration=deeper_configuration,
        device=torch.device(device="cpu")))
    loaded_block_count = count_loaded_backbone_blocks(model_configuration=deeper_configuration)
    if loaded_block_count != source_block_count:
        raise ValueError(f"The deeper backbone loads {loaded_block_count} blocks of {run_folder}, expected its "
                         f"{source_block_count} blocks.")

    # Pretrained backbone block of the experiment information, printed by the trainer from the attributes it reads
    trainer_attributes = types.SimpleNamespace(
        pretrained_configurations={"Backbone": deeper_configuration["Backbone"]["pretrained"]},
        loaded_backbone_blocks=loaded_block_count, model_configuration=deeper_configuration,
        learning_rate=experiment_configuration["learning_rate"])
    information_text = capture_printed_text(
        printing_function=lambda: BaseTrainer.print_pretrained_components(self=trainer_attributes))

    # Hyperlinks reduced to their labels, the run name, and the separator lines of the loading message left out
    console_lines = [line.rstrip() for line in INVISIBLE_SEQUENCE_PATTERN.sub(
        repl="", string=loading_text + information_text).splitlines()
        if line.strip() and set(line.strip()) != {"-"}]
    (listings_folder / "backbone_transfer_console.txt").write_text(data="\n".join(console_lines) + "\n",
                                                                   encoding="utf-8")
    deeper_block_count = len(convolutions)
    return {"backbone_transfer.source_block_count": source_block_count,
            "backbone_transfer.deeper_block_count": deeper_block_count,
            "backbone_transfer.loaded_block_count": loaded_block_count,
            "backbone_transfer.first_random_block": loaded_block_count + 1,
            "backbone_transfer.last_random_block": deeper_block_count,
            "backbone_transfer.learning_rate_multiplier": LEARNING_RATE_MULTIPLIER}
