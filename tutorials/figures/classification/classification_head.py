"""
Model figures of the classification tutorials (the head, output and loss chapters), made by
tutorials/figures/classification/figures.py from the last checkpoint of the classification run, read on the CPU, so
that the scenes quote what the trained classifier really computes: the class logits and probabilities of a test star,
the correlation matrix of the units of the first hidden layer of the head over validation images (the matrix the
decorrelation loss of the repository penalizes, drawn on the card colour), the balanced class weights of the training
split, the parameters of every block of the backbone and of the head, and the shapes and parameters of every layer of
the head with those of the same head after an average pooling. The checkpoint inference is reused from the previous
build unless forced.
"""

import json

import numpy
import torch

from matplotlib import pyplot
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (DOTS_PER_INCH, SERIES_COLOURS, STAR_IMAGE_NAME, SURFACE_COLOUR,
                                                TEXT_SECONDARY_COLOUR, find_star_image, load_classification_model,
                                                read_network_input, read_reused_facts, read_run_configuration,
                                                register_fonts, round_fact, save_figure, write_network_input,
                                                write_reused_facts)
from tutorials.tutorial_builder.layout import NAMED_COLOURS
from vision_training.losses.classification_losses import (compute_balanced_class_weights,
                                                          compute_decorrelation_loss)
from vision_training.losses.self_supervised_losses import compute_feature_cross_correlation
from vision_training.models.classification.classification_head import ClassificationHead
from vision_training.models.components.convolution_block import BATCH_NORMALIZATION_MOMENTUM
from vision_training.models.task_model import TaskModel

# Validation images of the correlation matrix of the hidden layer of the head, and the size of their batches: one
# image at a time, so that every activation stays small enough to be reused from the heap; batches of 32 images map
# activations of hundreds of megabytes afresh at every batch, which spent minutes in the kernel faulting their pages
# with the memory fragmented by the trainings
CORRELATION_IMAGE_COUNT = 256
BATCH_SIZE = 1

# Diverging ramp of the correlations, from blue for -1 through a neutral grey for 0 to red for 1, validated against
# the card surface
CORRELATION_COLOURS = [SERIES_COLOURS[0], "#4a4e58", "#%02x%02x%02x" % NAMED_COLOURS["red"]]

# Size of the correlation figure in pixels, placed in a box of a diagram scene, and the unit numbers labelled on its
# axes, every UNIT_TICK_STEP units
CORRELATION_FIGURE_WIDTH = 760
CORRELATION_FIGURE_HEIGHT = 680
UNIT_TICK_STEP = 32

# Correlation above which two hidden units count as strongly correlated in the facts
STRONG_CORRELATION = 0.5

# Name of the JSON file keeping the facts of the figures for their reuse
FACTS_FILE_NAME = "classification_head_facts.json"


def compute_hidden_features(model: TaskModel, image_paths: List[Path], image_size: int, keep_ratio: bool,
                            hidden_layer: int) -> torch.Tensor:
    """
    Computes the features of one hidden layer of the classification head, before its ReLU, for validation images,
    the features the decorrelation loss of the repository is computed on: the backbone, the global pooling (or the
    flattening) and the head up to that layer, by batches.

    Args:
        model (TaskModel): The trained classifier, in evaluation mode.
        image_paths (List[Path]): The validation images.
        image_size (int): The input size of the network.
        keep_ratio (bool): Whether the aspect ratio is kept by padding.
        hidden_layer (int): The hidden layer, counted from 1 ('ClassificationHead.decorrelation.hidden_layer').

    Returns:
        torch.Tensor: The [N, D] hidden features.
    """
    hidden_feature_batches = []
    for batch_start in range(0, len(image_paths), BATCH_SIZE):
        batch_tensor = torch.cat(tensors=[read_network_input(image_path=image_path, image_size=image_size,
                                                             keep_ratio=keep_ratio)
                                          for image_path in image_paths[batch_start:batch_start + BATCH_SIZE]])
        with torch.no_grad():
            final_feature_map, _ = model.backbone(input_tensor=batch_tensor)
            _, hidden_features = model.classification_head.compute_logits(
                features=model.global_pooling(input=final_feature_map), kept_hidden_layer=hidden_layer)
        hidden_feature_batches.append(hidden_features)
    return torch.cat(tensors=hidden_feature_batches)


def draw_correlation_figure(correlation: numpy.ndarray, output_path: Path) -> None:
    """
    Draws the correlation matrix of the units of the hidden layer of the head over the validation images, on the card
    colour of the videos, with the diverging ramp of the videos centred on zero and its colour bar.

    Args:
        correlation (numpy.ndarray): The [D, D] correlation matrix.
        output_path (Path): The PNG file of the figure.
    """
    register_fonts()
    figure, axes = pyplot.subplots(figsize=(CORRELATION_FIGURE_WIDTH / DOTS_PER_INCH,
                                            CORRELATION_FIGURE_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH)
    figure.patch.set_facecolor(SURFACE_COLOUR)
    axes.set_facecolor(SURFACE_COLOUR)
    colour_map = LinearSegmentedColormap.from_list(name="correlation", colors=CORRELATION_COLOURS)
    image = axes.imshow(X=correlation, cmap=colour_map, vmin=-1.0, vmax=1.0, interpolation="nearest")

    # Unit numbers on both axes, every UNIT_TICK_STEP units and the last one, in the secondary text colour
    ticks = list(range(0, correlation.shape[0], UNIT_TICK_STEP)) + [correlation.shape[0] - 1]
    axes.set_xticks(ticks=ticks, labels=[str(tick + 1) for tick in ticks])
    axes.set_yticks(ticks=ticks, labels=[str(tick + 1) for tick in ticks])
    axes.tick_params(colors=TEXT_SECONDARY_COLOUR, labelsize=13)
    axes.set_xlabel(xlabel="hidden unit", color=TEXT_SECONDARY_COLOUR, fontsize=14)
    axes.set_ylabel(ylabel="hidden unit", color=TEXT_SECONDARY_COLOUR, fontsize=14)
    for spine in axes.spines.values():
        spine.set_visible(False)

    # Colour bar of the correlations, from -1 to 1
    colour_bar = figure.colorbar(mappable=image, ax=axes, fraction=0.046, pad=0.04, ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    colour_bar.ax.tick_params(colors=TEXT_SECONDARY_COLOUR, labelsize=13)
    colour_bar.outline.set_visible(False)
    colour_bar.set_label(label="correlation", color=TEXT_SECONDARY_COLOUR, fontsize=14)
    figure.tight_layout()
    save_figure(figure=figure, output_path=output_path)
    pyplot.close(fig=figure)


def describe_star_outputs(class_names: List[str], logits: numpy.ndarray,
                          probabilities: numpy.ndarray) -> Dict[str, Any]:
    """
    Gives the facts of the class outputs of the test star: the logit and probability of every class, and the classes
    with their logits and probabilities from the most probable one, the order of the bars of the scene.

    Args:
        class_names (List[str]): The classes, in the order of the outputs.
        logits (numpy.ndarray): The [K] class logits of the star.
        probabilities (numpy.ndarray): The [K] class probabilities of the star.

    Returns:
        Dict[str, Any]: The facts, named star_image.*.
    """
    facts = {"star_image.file_name": STAR_IMAGE_NAME}
    for class_name, logit, probability in zip(class_names, logits, probabilities):
        facts[f"star_image.logits.{class_name}"] = round_fact(value=logit)
        facts[f"star_image.probabilities.{class_name}"] = round_fact(value=probability)

    # Classes from the most probable one, for the bars of the scene
    class_order = numpy.argsort(-probabilities, kind="stable").tolist()
    facts.update({"star_image.predicted_class": class_names[class_order[0]],
                  "star_image.sorted_class_names": [class_names[class_index] for class_index in class_order],
                  "star_image.sorted_logits": [round_fact(value=logits[class_index]) for class_index in class_order],
                  "star_image.sorted_probabilities": [round_fact(value=probabilities[class_index])
                                                      for class_index in class_order],
                  "star_image.sorted_probability_labels": [f"{probabilities[class_index]:.4f}"
                                                           for class_index in class_order],
                  "star_image.sorted_logit_rows": [[round_fact(value=logits[class_index])]
                                                   for class_index in class_order]})
    return facts


def count_parameters(module: torch.nn.Module) -> int:
    """
    Counts the parameters of a module of the classifier, the numbers of the parameter facts of the head scenes.

    Args:
        module (torch.nn.Module): The module.

    Returns:
        int: Its number of parameters.
    """
    return sum(parameter.numel() for parameter in module.parameters())


def describe_head_layers(model: TaskModel, head_configuration: Dict[str, Any], image_size: int) -> Dict[str, Any]:
    """
    Gives the shapes and parameters of the layers of the trained head, the numbers of the pooling and head scenes: the
    final feature map of the backbone and the vector the pooling makes of it, the parameters of the first hidden
    layer, of its normalization and of the class layer, the parameters the same head would hold after an average
    pooling (built here with the head settings of the run, never trained), and the size of the flattened vector at
    half the image size.

    Args:
        model (TaskModel): The trained classifier.
        head_configuration (Dict[str, Any]): The 'ClassificationHead' section of the run.
        image_size (int): The input size of the network, 'Data.image_settings.size' of the run.

    Returns:
        Dict[str, Any]: The facts, named head.*.
    """
    head = model.classification_head
    final_layer = model.backbone.compute_receptive_field_information()[-1]
    channels, map_size = final_layer["output_channels"], final_layer["output_size"]

    # First linear layer and normalization of the hidden block (dropout, linear, normalization, ReLU)
    hidden_layers = list(head.fully_connected_block)
    first_linear = [layer for layer in hidden_layers if isinstance(layer, torch.nn.Linear)][0]
    normalization = [layer for layer in hidden_layers if isinstance(layer, torch.nn.LayerNorm)][0]

    # The same head after an average pooling, which gives one value per channel
    average_head = ClassificationHead(input_features=channels, fully_connected=head_configuration["fully_connected"],
                                      fully_connected_normalization=head_configuration["fully_connected_normalization"],
                                      batch_normalization_momentum=BATCH_NORMALIZATION_MOMENTUM,
                                      dropout_rate=head_configuration["dropout_rate"],
                                      number_classes=model.number_classes, device=torch.device("cpu"),
                                      dtype=torch.float32)

    # The last map shrinks with the image, its side being the image side over the stride of the backbone
    half_size_map = map_size * (image_size // 2) // image_size
    return {"head.final_map_channels": channels, "head.final_map_size": map_size,
            "head.input_features": head.input_features, "head.hidden_units": first_linear.out_features,
            "head.number_classes": model.number_classes,
            "head.first_linear_parameters": count_parameters(module=first_linear),
            "head.normalization_parameters": count_parameters(module=normalization),
            "head.classifier_parameters": count_parameters(module=head.classifier),
            "head.average_pooling_parameters": count_parameters(module=average_head),
            "head.half_image_size": image_size // 2,
            "head.flatten_features_at_half_size": channels * half_size_map * half_size_map,
            "head.first_linear_share": round_fact(value=count_parameters(module=first_linear) /
                                                  count_parameters(module=model))}


def make_classification_head_figures(output_folder: Path, run_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Runs the last checkpoint of the classification run on the test star and on validation images, writes the input of
    the star and the correlation figure of the hidden features into the model folder of the series, and gives the
    outputs of the star, the correlation of the hidden features, the class weights and the parameters of the model;
    the figures of the previous build are reused unless forced or incomplete.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the classification run of tutorial_sources.yaml.
        force (bool): Whether the checkpoint is run again even when the previous build left the figures.

    Returns:
        Dict[str, Any]: The facts of the figures, named star_image.*, hidden_features.*, class_weights.* and
            parameters.*.
    """
    # Figures of the previous build
    model_folder = output_folder / "model"
    facts_path = model_folder / FACTS_FILE_NAME
    figure_paths = [model_folder / "input_star.png", model_folder / "hidden_feature_correlation.png"]
    reused_facts = read_reused_facts(facts_path=facts_path, required_paths=figure_paths, force=force)
    if reused_facts is not None:
        return reused_facts

    # Trained model of the run, its classes and the settings of its decorrelation
    model, checkpoint_iteration = load_classification_model(run_folder=run_folder)
    run_configuration = read_run_configuration(run_folder=run_folder)
    image_settings = run_configuration["Data"]["image_settings"]
    decorrelation_configuration = run_configuration["ClassificationHead"]["decorrelation"]
    class_records = json.loads((run_folder / "data.json").read_text(encoding="utf-8"))["classes"]
    class_names = [class_record["class_name"] for class_record in class_records]

    # Class outputs of the test star, written as the input of the network
    star_tensor = read_network_input(image_path=find_star_image(run_configuration=run_configuration),
                                     image_size=image_settings["size"], keep_ratio=image_settings["keep_ratio"])
    with torch.no_grad():
        star_outputs = model(input_tensor=star_tensor)
    write_network_input(input_tensor=star_tensor, output_path=model_folder / "input_star.png")
    facts = describe_star_outputs(class_names=class_names, logits=star_outputs["class_logits"][0].numpy(),
                                  probabilities=star_outputs["class_probabilities"][0].numpy())
    facts["star_image.checkpoint_iteration"] = checkpoint_iteration

    # Hidden features of the head over the first validation images, their correlation matrix as the decorrelation loss
    # computes it, and that loss at the weight of the configuration
    validation_folder = Path(run_configuration["ExperimentConfiguration"]["validation_data_folder"])
    validation_samples = json.loads((validation_folder / "labels.json").read_text(encoding="utf-8"))["samples"]
    hidden_features = compute_hidden_features(
        model=model, image_paths=[validation_folder / "data" / sample["file_name"]
                                  for sample in validation_samples[:CORRELATION_IMAGE_COUNT]],
        image_size=image_settings["size"], keep_ratio=image_settings["keep_ratio"],
        hidden_layer=decorrelation_configuration["hidden_layer"])
    correlation = compute_feature_cross_correlation(first_embeddings=hidden_features,
                                                    second_embeddings=hidden_features).numpy()
    draw_correlation_figure(correlation=correlation, output_path=model_folder / "hidden_feature_correlation.png")
    decorrelation = compute_decorrelation_loss(hidden_features=hidden_features,
                                               weight=decorrelation_configuration["weight"])
    image_count, unit_count = hidden_features.shape
    off_diagonal = numpy.abs(correlation[~numpy.eye(unit_count, dtype=bool)])
    facts.update({"hidden_features.image_count": image_count, "hidden_features.unit_count": unit_count,
                  "hidden_features.hidden_layer": decorrelation_configuration["hidden_layer"],
                  "hidden_features.mean_absolute_correlation": round_fact(
                      value=decorrelation["hidden_feature_correlation"]),
                  "hidden_features.decorrelation_weight": decorrelation_configuration["weight"],
                  "hidden_features.decorrelation_loss": round_fact(value=decorrelation["decorrelation_loss"]),
                  "hidden_features.strong_pair_fraction": round_fact(value=(off_diagonal > STRONG_CORRELATION).mean()),
                  "hidden_features.noise_floor": round_fact(value=unit_count * (unit_count - 1) / (image_count - 1))})

    # Floor of the sum of the squared correlations over one training batch of the run, from the sampling noise alone
    batch_size = run_configuration["ExperimentConfiguration"]["batch_size"]
    facts.update({"hidden_features.batch_size": batch_size,
                  "hidden_features.batch_noise_floor": round_fact(value=unit_count * (unit_count - 1) /
                                                                  (batch_size - 1))})

    # Balanced class weights of the training split, from the training images of every class recorded by the run
    class_counts = [class_record["images"]["train"] for class_record in class_records]
    class_weights = [round_fact(value=weight) for weight in
                     compute_balanced_class_weights(class_counts=class_counts).tolist()]
    for class_name, class_count, class_weight in zip(class_names, class_counts, class_weights):
        facts[f"class_weights.{class_name}.training_images"] = class_count
        facts[f"class_weights.{class_name}.weight"] = class_weight
    facts.update({"class_weights.class_names": class_names, "class_weights.training_images": class_counts,
                  "class_weights.weights": class_weights, "class_weights.minimum": min(class_weights),
                  "class_weights.maximum": max(class_weights),
                  "class_weights.weight_rows": [class_weights]})

    # Parameters of every block of the backbone, of the backbone, of the head and of the whole model
    block_parameters = [sum(parameter.numel() for parameter in block.parameters())
                        for block in model.backbone.convolution_blocks]
    for block_number, parameter_count in enumerate(block_parameters, start=1):
        facts[f"parameters.block_{block_number}"] = parameter_count

    # Blocks after the first adding no residual shortcut, those changing the channels with two layers
    facts["backbone.blocks_without_shortcut"] = [
        block_number for block_number, block in enumerate(model.backbone.convolution_blocks, start=1)
        if block_number > 1 and block.count_residual_additions() == 0]
    block_numbers = [str(block_number) for block_number in facts["backbone.blocks_without_shortcut"]]
    facts["backbone.blocks_without_shortcut_text"] = (", ".join(block_numbers[:-1]) + " and " + block_numbers[-1]
                                                      if len(block_numbers) > 1 else "".join(block_numbers))
    facts.update({"parameters.blocks": block_parameters,
                  "parameters.backbone": sum(parameter.numel() for parameter in model.backbone.parameters()),
                  "parameters.head": sum(parameter.numel() for parameter in model.classification_head.parameters()),
                  "parameters.total": sum(parameter.numel() for parameter in model.parameters())})
    facts.update(describe_head_layers(model=model, head_configuration=run_configuration["ClassificationHead"],
                                      image_size=image_settings["size"]))
    facts["star_image.cross_entropy"] = round_fact(value=-numpy.log(facts["star_image.probabilities.star"]))
    write_reused_facts(facts_path=facts_path, figure_paths=figure_paths, facts=facts)
    return facts
