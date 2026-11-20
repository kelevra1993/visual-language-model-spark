"""
Figures and facts of the sequence comparison (CMP) series of the tutorial videos, made by make_figures.py
(python -m tutorials.figures.make_figures --series sequence_comparison) at every build, from the README run of
tutorials/tutorial_sources.yaml, its object loss pairs dataset and the code of the repository: the dataset figures and
listings, the animated pair batch encoded by the trained comparator on the CPU, the sizes of the components of the
comparator, and the training and evaluation figures. Every figure family is a module of this folder writing into a
subfolder of the output folder (dataset/, batch/, evaluation/, ...), its facts gathered here into the facts.json the
storyboards quote as {{sequence_comparison.<name>}}, with the settings the run was trained with, the counts of fusion
and temporal module types of the code and the settings of the generator. The helpers are those of the sequence
classification series. A build from the committed snapshots (tutorials/run_snapshots, without the checkpoint) counts
the components on the comparator built from its configuration and reuses the committed outputs of the animated batch.
"""

import torch

from pathlib import Path
from typing import Any, Dict

from tutorials.figures.figure_utilities import find_run_folder, merge_facts, read_run_configuration
from tutorials.figures.sequence_classification.sequence_figure_utilities import (add_number_words, find_console_log,
                                                                                 load_sequence_model)
from tutorials.figures.sequence_comparison.pair_batch_animation import make_pair_batch_animation
from tutorials.figures.sequence_comparison.pair_dataset_figures import make_pair_dataset_figures
from tutorials.figures.sequence_comparison.pair_evaluation_figures import make_pair_evaluation_figures
from tutorials.figures.sequence_labelling.sequence_utilities import read_main_block_settings
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.data.augmentations import GEOMETRIC_AUGMENTATION_TYPES
from vision_training.models.sequence.member_fusion import FUSION_TYPES, MemberFusion
from vision_training.models.sequence.temporal_modules import ORDERED_TEMPORAL_TYPES, POOLING_TEMPORAL_TYPES
from vision_training.models.task_model import TaskModel
from vision_training.trainers.checkpointing import get_checkpoint_file, get_last_checkpoint_iteration
from vision_training.trainers.sequence_comparison_trainer import SequenceComparisonTrainer

# Run of the series in tutorial_sources.yaml
RUN_KEY = "sequence_comparison"

# Generator whose main block settings the narration quotes (what a dataset generated with it holds), and its task
GENERATOR_SCRIPT = "vision_training/data/synthetic/create_synthetic_dataset.py"
GENERATOR_TASK = "sequence_comparison"

# Only geometric augmentation of the run, as the pitfalls of the third chapter say: the crops, zooms, shifts and
# rotations can cut or push out a border object and change the count, the mirrored pair keeping its objects
NARRATED_GEOMETRIC_AUGMENTATIONS = ["random_left_right_flip"]


def count_component_parameters(model: TaskModel, image_size: int, input_channels: int) -> Dict[str, Any]:
    """
    Counts the parameters of every component of the trained comparator, gives the features of its member and pair
    vectors and those of every fusion type for its member vectors, and runs its backbone on one black frame to give
    the channels and side of its last feature map, the sizes the comparator scenes quote.

    Args:
        model (TaskModel): The trained comparator.
        image_size (int): The frame size of the run.
        input_channels (int): The channels of a frame of the run.

    Returns:
        Dict[str, Any]: The facts, named parameters.*, features.* and backbone.*.
    """
    facts = {}
    for component_name in ["backbone", "temporal_module", "member_fusion", "classification_head"]:
        facts[f"parameters.{component_name}"] = sum(parameter.numel() for parameter in
                                                    getattr(model, component_name).parameters())
    facts["parameters.total"] = sum(parameter.numel() for parameter in model.parameters())
    facts["parameters.total_megabytes"] = round(facts["parameters.total"] * 4 / 1e6, 2)
    member_features = int(model.temporal_module.output_features)
    facts["features.member"] = member_features
    facts["features.pair"] = int(model.member_fusion.output_features)

    # Pair vector of every fusion type for the member vectors of the run, and its multiple of a member vector, the
    # fusions built with one head, a divisor of any channel count, since the head count leaves the sizes unchanged
    with torch.no_grad():
        backbone_maps, _ = model.backbone(input_tensor=torch.zeros(size=(1, input_channels, image_size, image_size)))
    for fusion_type in FUSION_TYPES:
        fusion = MemberFusion(fusion_type=fusion_type, frame_features=int(backbone_maps.shape[1]),
                              member_features=member_features, number_heads=1, dropout_rate=0.0,
                              device=torch.device("cpu"), dtype=torch.float32)
        facts[f"features.{fusion_type}"] = int(fusion.output_features)
        facts[f"features.{fusion_type}_multiple"] = int(fusion.output_features) // member_features

    # Channels and side of the last feature map of the backbone at the frame size of the run
    facts["backbone.channels"] = int(backbone_maps.shape[1])
    facts["backbone.map_size"] = int(backbone_maps.shape[2])
    return facts


def read_run_setting_facts(run_configuration: Dict[str, Any], label_dictionary: Dict[int, int]) -> Dict[str, Any]:
    """
    Reads the settings the comparison run was trained with from its saved training configuration, those the narration
    quotes for the README run (batch, frame and sequence sizes, the intervals of its console summary and of its
    evaluations, its head and its fusion), so that the narration describes that run even when the shipped
    configuration changes, and checks that its only geometric augmentation is the one the pitfalls name.

    Args:
        run_configuration (Dict[str, Any]): The saved training configuration of the run.
        label_dictionary (Dict[int, int]): The class index of every category of the run.

    Returns:
        Dict[str, Any]: The facts, named run.*.
    """
    experiment_configuration = run_configuration["ExperimentConfiguration"]
    head_configuration = run_configuration["ClassificationHead"]

    # Geometric augmentations of the run, the left-right flip alone as the narration says
    augmentation_configuration = run_configuration["Data"]["data_augmentation"]
    activated_geometric_types = [augmentation_type for augmentation_type in GEOMETRIC_AUGMENTATION_TYPES
                                 if augmentation_configuration["activated"]
                                 and augmentation_configuration["types"][augmentation_type]["activated"]]
    if activated_geometric_types != NARRATED_GEOMETRIC_AUGMENTATIONS:
        raise ValueError(f"The geometric augmentations of the run are {activated_geometric_types}, the narration says "
                         f"{NARRATED_GEOMETRIC_AUGMENTATIONS}: update the pitfalls of the third chapter.")
    return {"run.information_dump": experiment_configuration["information_dump"],
            "run.weight_saving_iterations": experiment_configuration["weight_saving_iterations"],
            "run.maximum_sequence_length": run_configuration["Data"]["sequence_settings"]["maximum_sequence_length"],
            "run.input_channels": run_configuration["Backbone"]["input_channels"],
            "run.head_hidden_units": head_configuration["fully_connected"][0],
            "run.head_dropout_rate": head_configuration["dropout_rate"],
            "run.class_count": len(set(label_dictionary.values())),
            "run.fusion_number_heads": run_configuration["Fusion"]["number_heads"]}


def read_code_facts() -> Dict[str, Any]:
    """
    Counts what the code of the comparison offers and imposes: the members of a sample of its trainer, the fusion types
    and the temporal modules, and reads from the main block of the synthetic dataset generator the members it writes
    (frame range, frame size, objects of the before member), the settings the dataset of the run was generated with.

    Returns:
        Dict[str, Any]: The facts, named code.* and generator.*.
    """
    generator_settings = read_main_block_settings(script_path=GENERATOR_SCRIPT, setting_names=[
        "minimum_frames", "maximum_frames", "image_sizes", "maximum_objects"])
    return {"code.members_per_sample": SequenceComparisonTrainer.MEMBERS_PER_SAMPLE,
            "code.fusion_type_count": len(FUSION_TYPES),
            "code.temporal_type_count": len(POOLING_TEMPORAL_TYPES) + len(ORDERED_TEMPORAL_TYPES),
            "generator.minimum_frames": generator_settings["minimum_frames"][GENERATOR_TASK],
            "generator.maximum_frames": generator_settings["maximum_frames"][GENERATOR_TASK],
            "generator.image_size": generator_settings["image_sizes"][GENERATOR_TASK],
            "generator.maximum_objects": generator_settings["maximum_objects"][GENERATOR_TASK]}


def make_figures(output_folder: Path, experiments_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Makes every figure of the sequence comparison series into its output folder and gives their facts, the encoding
    of the animated batch being reused from the previous build unless forced.

    Args:
        output_folder (Path): The figures folder of the series in the cache of the build.
        experiments_folder (Path): The folder of the real runs, or the committed snapshots.
        force (bool): Whether the expensive figures are made again even when the previous build left them.

    Returns:
        Dict[str, Any]: The facts of every figure family, keyed by their dotted names, with the words of their small
            whole numbers.
    """
    run_folder = find_run_folder(experiments_folder=experiments_folder, run_key=RUN_KEY)
    log_path = find_console_log(experiments_folder=experiments_folder, run_key=RUN_KEY, comparison_key=None)
    iteration = get_last_checkpoint_iteration(weights_directory=run_folder / "Weights")
    run_configuration = read_run_configuration(run_folder=run_folder)
    experiment_configuration = run_configuration["ExperimentConfiguration"]
    evaluation_configuration = experiment_configuration["evaluation"]
    _, model_configuration = load_experiment_configuration(configuration_path=run_folder /
                                                           "training_configuration.yaml")
    label_dictionary = model_configuration["Data"]["label_dictionary"]

    # Comparator with its trained weights, or built from its configuration when the run is read from the committed
    # snapshots, its components having the same sizes and the animated batch being restored from the snapshots
    model = load_sequence_model(run_folder=run_folder, iteration=iteration, allow_untrained=True)
    has_trained_weights = get_checkpoint_file(weights_directory=run_folder / "Weights", iteration=iteration).exists()
    train_folder = Path(experiment_configuration["train_data_folder"])
    image_settings = run_configuration["Data"]["image_settings"]
    image_size = image_settings["size"]
    facts = {"run.name": run_folder.name, "run.batch_size": experiment_configuration["batch_size"],
             "run.image_size": image_size, "run.iteration": iteration}
    return add_number_words(facts=merge_facts(fact_groups=[
        facts, count_component_parameters(model=model, image_size=image_size,
                                          input_channels=run_configuration["Backbone"]["input_channels"]),
        read_run_setting_facts(run_configuration=run_configuration, label_dictionary=label_dictionary),
        read_code_facts(),
        # Figure families of the series: the dataset, the animated pair batch, then the training and evaluation
        make_pair_dataset_figures(output_folder=output_folder, train_folder=train_folder,
                                  label_dictionary=label_dictionary, batch_size=experiment_configuration["batch_size"]),
        make_pair_batch_animation(output_folder=output_folder, model=model, train_folder=train_folder,
                                  label_dictionary=label_dictionary, image_size=image_size,
                                  keep_ratio=image_settings["keep_ratio"], has_trained_weights=has_trained_weights,
                                  force=force),
        make_pair_evaluation_figures(output_folder=output_folder, run_folder=run_folder, log_path=log_path,
                                     iteration=iteration, class_count=len(set(label_dictionary.values())),
                                     test_folder=Path(experiment_configuration["test_data_folder"]),
                                     information_dump=experiment_configuration["information_dump"],
                                     image_settings=image_settings,
                                     sequence_settings=run_configuration["Data"]["sequence_settings"],
                                     label_dictionary=label_dictionary,
                                     number_sample_images=evaluation_configuration["number_sample_images"])]))
