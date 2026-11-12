"""
Representation figures of the self-supervised series of the tutorial videos (chapters 2 and 3), made by
tutorials/figures/self_supervised/figures.py from the checkpoints of the SimCLR run rebuilt on the CPU: the
neighbourhood of the star of the sample images, a test image, among the reference training images of the k-NN
evaluation (select_reference_samples), encoded by the exported InferenceEncoder at the first and last evaluated
checkpoints, as the evaluation classifies every labelled test image (its nearest reference images by cosine similarity,
the weighted vote of its neighbours, checked against compute_nearest_neighbour_predictions, and a list of its nearest
images for the animated page animations/11_self_supervised/knn_vote.html); a mosaic of its nearest reference images at
both checkpoints; the cosine similarities of the embeddings of the two views of four sample images, the InfoNCE matrix
of chapter 2; the views similarity of the star at every evaluated checkpoint; and the positive pair accuracy of an
untrained and of the trained network on colour and on grayscale views, the cue the two views share. The encoding takes
minutes on a shared CPU, so the figures are reused unless forced.
"""

import json
import math

import numpy
import torch

from matplotlib import pyplot
from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, POINTS_PER_PIXEL,
                                                SURFACE_COLOUR, TEXT_PRIMARY_COLOUR, TEXT_SECONDARY_COLOUR,
                                                read_reused_facts, read_run_configuration, register_fonts,
                                                round_fact, save_figure, write_reused_facts)
from tutorials.figures.self_supervised.ssl_sources import (build_view_augmenter, draw_sample_views, encode_images,
                                                          find_sample, list_evaluated_iterations,
                                                          load_self_supervised_model, read_image_settings,
                                                          read_images, read_reference_samples, read_test_samples)
from tutorials.figures.self_supervised.view_figures import VIEW_SAMPLE_NAMES
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.evaluation.self_supervised.representation_metrics import (
    NEAREST_NEIGHBOUR_TEMPERATURE, compute_nearest_neighbour_predictions)
from vision_training.losses.self_supervised_losses import compute_info_nce_loss
from vision_training.models.self_supervised.self_supervised_model import SelfSupervisedModel

# Test image whose neighbourhood among the reference training images the figures show: the star of the sample images
QUERY_NAME = "test_000000.png"

# Nearest reference images of the query shown by the radial layout of the animated page, and by the mosaic
RADIAL_POINT_COUNT = 150
MOSAIC_NEIGHBOUR_COUNT = 5

# Seed of the untrained network and of the first draw of the views the positive pair accuracy of the colour cue is
# measured on (the first test images, as many as a batch of the run), and the number of draws of the views it is
# averaged over, a single draw of a batch being too noisy to compare colour and grayscale views
CUE_SEED = 0
CUE_VIEW_DRAWS = 3

# Weights of the blue, green and red channels in the luminance of the grayscale views (ITU-R BT.601, as OpenCV)
GRAYSCALE_WEIGHTS = (0.114, 0.587, 0.299)

# Colours of the nine classes in the animated page and the mosaic captions, the series colours of the frames first
CLASS_COLOURS = ["#3987e5", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767", "#7dc4ff", "#f0c878", "#78dc78"]

# Name of the JSON file keeping the facts of the figures for their reuse
FACTS_FILE_NAME = "representation_facts.json"

# Fact written only since the neighbourhood is drawn among the reference training images, so that facts kept by a
# build made when it was drawn among the test images are made again instead of being reused
REFERENCE_FACT_NAME = "knn.reference_count"


def describe_neighbourhood(query_representation: numpy.ndarray, reference_representations: numpy.ndarray,
                           reference_samples: List[Dict[str, Any]], number_neighbours: int,
                           number_classes: int) -> Dict[str, Any]:
    """
    Gives the neighbourhood of a test image among the reference training images, as the k-NN evaluation classifies it:
    the cosine similarity of every reference image to the query (both normalized and multiplied as the evaluation
    does), the k nearest chosen by the same topk (k reduced to the number of reference images) so that equal
    similarities pick the same neighbours, then the other reference images from the nearest, and the vote of the k
    nearest, each weighing exp(similarity / NEAREST_NEIGHBOUR_TEMPERATURE). The predicted class must be the one
    compute_nearest_neighbour_predictions of the repository gives, so that the scene shows the vote of the current code.

    Args:
        query_representation (numpy.ndarray): The [F] representation of the test image.
        reference_representations (numpy.ndarray): The [M, F] representations of the reference images.
        reference_samples (List[Dict[str, Any]]): The reference images, in the order of their representations.
        number_neighbours (int): The number k of voting neighbours of the run.
        number_classes (int): The number of classes of the run.

    Returns:
        Dict[str, Any]: The 'similarities' [M], the 'order' of the reference images (the voting neighbours first), the
            'number_voting_neighbours', the 'votes' of every class and the predicted 'class_index'.

    Raises:
        ValueError: If the vote differs from compute_nearest_neighbour_predictions, the evaluation having changed.
    """
    # Similarities and voting neighbours computed like compute_nearest_neighbour_predictions
    normalized_query = torch.nn.functional.normalize(input=torch.from_numpy(query_representation[None, :]).float(),
                                                     dim=1)
    normalized_references = torch.nn.functional.normalize(input=torch.from_numpy(reference_representations).float(),
                                                           dim=1)
    similarities = (normalized_query @ normalized_references.T)[0]
    number_voting_neighbours = min(number_neighbours, len(reference_samples))
    neighbour_similarities, neighbour_indices = similarities.topk(k=number_voting_neighbours)

    # Weighted vote of the neighbours per class, the class of the largest weight (the first one on a tie) predicted
    reference_labels = torch.tensor(data=[sample["class_index"] for sample in reference_samples], dtype=torch.int64)
    votes = torch.zeros(size=(number_classes,))
    votes.scatter_add_(dim=0, index=reference_labels[neighbour_indices],
                       src=torch.exp(input=neighbour_similarities / NEAREST_NEIGHBOUR_TEMPERATURE))
    predicted_class_index = int(votes.argmax())

    # The prediction of the evaluation code itself, which the vote drawn by the scene must give
    evaluated_class_index = int(compute_nearest_neighbour_predictions(
        query_representations=torch.from_numpy(query_representation[None, :]),
        reference_representations=torch.from_numpy(reference_representations), reference_labels=reference_labels,
        number_neighbours=number_neighbours, number_classes=number_classes)[0])
    if evaluated_class_index != predicted_class_index:
        raise ValueError(f"The vote of the figures gives the class {predicted_class_index} and "
                         f"compute_nearest_neighbour_predictions the class {evaluated_class_index}: update "
                         f"describe_neighbourhood to the k-NN of the evaluation.")

    # Voting neighbours first, then the other reference images from the nearest
    voting_indices = neighbour_indices.tolist()
    voting_index_set = set(voting_indices)
    other_indices = [index for index in torch.argsort(input=similarities, descending=True, stable=True).tolist()
                     if index not in voting_index_set]
    return {"similarities": similarities.numpy(), "order": numpy.array(voting_indices + other_indices),
            "number_voting_neighbours": number_voting_neighbours, "votes": votes.double().numpy(),
            "class_index": evaluated_class_index}


def write_radial_layout(output_path: Path, reference_samples: List[Dict[str, Any]], query_sample: Dict[str, Any],
                        neighbourhood: Dict[str, Any], class_names: List[str]) -> None:
    """
    Writes the data of the animated page of the k-NN vote: the RADIAL_POINT_COUNT nearest reference images of the
    query in order, each with its cosine similarity to the query, its class and its rank, the page placing every image
    at a distance from the centre growing with 1 - similarity, the k voting neighbours inside the ring of the k-th one.

    Args:
        output_path (Path): The JSON file written.
        reference_samples (List[Dict[str, Any]]): The reference images.
        query_sample (Dict[str, Any]): The test image at the centre.
        neighbourhood (Dict[str, Any]): The neighbourhood of the query, from describe_neighbourhood.
        class_names (List[str]): The class names in class index order.
    """
    shown_indices = neighbourhood["order"][:RADIAL_POINT_COUNT]
    points = [{"similarity": round(float(neighbourhood["similarities"][image_index]), 5),
               "class_index": reference_samples[image_index]["class_index"], "rank": rank}
              for rank, image_index in enumerate(shown_indices)]
    layout = {"points": points, "number_neighbours": neighbourhood["number_voting_neighbours"],
              "query_class_index": query_sample["class_index"], "class_names": class_names,
              "class_colours": CLASS_COLOURS}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(data=json.dumps(obj=layout), encoding="utf-8")


def draw_neighbour_mosaic(output_path: Path, rows: List[Dict[str, Any]], query_sample: Dict[str, Any],
                          reference_samples: List[Dict[str, Any]], image_size: int, keep_ratio: bool) -> None:
    """
    Draws the mosaic of the nearest reference images of the query at several checkpoints: one row per checkpoint, the
    query then its nearest reference images, every image captioned with its class and its cosine similarity to the
    query, the class in its colour.

    Args:
        output_path (Path): The PNG file written.
        rows (List[Dict[str, Any]]): Every row: its 'iteration' and its 'neighbours' (their 'index' among the reference
            images and their 'similarity' to the query), nearest first.
        query_sample (Dict[str, Any]): The test image whose neighbours are shown.
        reference_samples (List[Dict[str, Any]]): The reference images.
        image_size (int): The input size of the run.
        keep_ratio (bool): Whether the run keeps the aspect ratio.
    """
    register_fonts()
    figure = pyplot.figure(figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH,
                           facecolor=SURFACE_COLOUR)
    tile_pixels, gap_pixels, left_pixels = 228, 24, 150
    tile_width, tile_height = tile_pixels / CHART_WIDTH, tile_pixels / CHART_HEIGHT
    for row_index, row in enumerate(rows):
        bottom = 1.0 - (50 + row_index * 330 + tile_pixels) / CHART_HEIGHT
        figure.text(x=20 / CHART_WIDTH, y=bottom + tile_height / 2, s=f"iteration\n{row['iteration']:,}",
                    color=TEXT_PRIMARY_COLOUR, fontsize=22 * POINTS_PER_PIXEL, va="center", ha="left")
        shown_images = [{"sample": query_sample, "similarity": None}] + [
            {"sample": reference_samples[neighbour["index"]], "similarity": neighbour["similarity"]}
            for neighbour in row["neighbours"]]

        # The query, then its nearest reference images with their class and similarity
        for column_index, shown_image in enumerate(shown_images):
            left = (left_pixels + column_index * (tile_pixels + gap_pixels) +
                    (2 * gap_pixels if column_index > 0 else 0)) / CHART_WIDTH
            axes = figure.add_axes(rect=(left, bottom, tile_width, tile_height))
            sample = shown_image["sample"]
            image = read_images(samples=[sample], image_size=image_size, keep_ratio=keep_ratio)
            axes.imshow(X=image[0].permute(1, 2, 0).numpy()[:, :, ::-1].astype(numpy.uint8))
            axes.set_xticks(ticks=[])
            axes.set_yticks(ticks=[])
            for spine in axes.spines.values():
                spine.set_color(TEXT_SECONDARY_COLOUR if column_index > 0 else TEXT_PRIMARY_COLOUR)
                spine.set_linewidth(2 * POINTS_PER_PIXEL)
            caption = "query" if shown_image["similarity"] is None else f"{shown_image['similarity']:.3f}"
            figure.text(x=left, y=bottom - 10 / CHART_HEIGHT, s=sample["class_name"].replace("_", " "), va="top",
                        ha="left", color=CLASS_COLOURS[sample["class_index"]], fontsize=19 * POINTS_PER_PIXEL,
                        fontweight="semibold")
            figure.text(x=left + tile_width, y=bottom - 10 / CHART_HEIGHT, s=caption, va="top", ha="right",
                        color=TEXT_SECONDARY_COLOUR, fontsize=19 * POINTS_PER_PIXEL)
    save_figure(figure=figure, output_path=output_path)
    pyplot.close(fig=figure)


def make_neighbour_mosaic(output_folder: Path, run_folder: Path, facts: Dict[str, Any]) -> None:
    """
    Draws the mosaic of the nearest reference images of the query at the first and last evaluated checkpoints from the
    neighbours kept in the facts of the representation figures, so that its layout can change without encoding the
    images again.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the self-supervised run.
        facts (Dict[str, Any]): The facts of the representation figures, with the knn.nearest_<iteration> neighbours
            of both checkpoints.
    """
    samples, class_names = read_test_samples(run_folder=run_folder)
    reference_samples = read_reference_samples(run_folder=run_folder, class_names=class_names)
    image_size, keep_ratio = read_image_settings(run_folder=run_folder)
    rows = [{"iteration": facts[f"knn.{position}_iteration"],
             "neighbours": facts[f"knn.nearest_{facts[f'knn.{position}_iteration']}"]}
            for position in ["first", "last"]]
    draw_neighbour_mosaic(output_path=output_folder / "representations" / "neighbour_mosaic.png", rows=rows,
                          query_sample=find_sample(samples=samples, file_name=QUERY_NAME),
                          reference_samples=reference_samples, image_size=image_size, keep_ratio=keep_ratio)


def measure_info_nce_matrix(model: SelfSupervisedModel, augmenter: Any, samples: List[Dict[str, Any]],
                            image_size: int, keep_ratio: bool, temperature: float) -> Dict[str, Any]:
    """
    Encodes the two views of the sample images of VIEW_SAMPLE_NAMES (drawn as the evaluation draws them) with the
    trained model in evaluation mode, as the validation batches are, and gives what InfoNCE computes on them: the
    cosine similarity of every pair of their embeddings, the first views first, the loss of the repository and its
    positive pair accuracy, and the row of the first view.

    Args:
        model (SelfSupervisedModel): The trained model, in evaluation mode.
        augmenter (Any): The DataAugmenter of the views of the run.
        samples (List[Dict[str, Any]]): The test samples.
        image_size (int): The input size of the run.
        keep_ratio (bool): Whether the run keeps the aspect ratio.
        temperature (float): The temperature of the run.

    Returns:
        Dict[str, Any]: The facts of the matrix, named matrix.*.
    """
    first_views, second_views = [], []
    for file_name in VIEW_SAMPLE_NAMES:
        sample = find_sample(samples=samples, file_name=file_name)
        image = read_images(samples=[sample], image_size=image_size, keep_ratio=keep_ratio)
        first_view, second_view, _, _ = draw_sample_views(augmenter=augmenter, image=image,
                                                          sample_identifier=sample["identifier"])
        first_views.append(first_view)
        second_views.append(second_view)
    with torch.no_grad():
        embeddings = model.encode(images=torch.cat(tensors=first_views + second_views, dim=0))["embeddings"]
    number_images = len(first_views)
    losses = compute_info_nce_loss(first_embeddings=embeddings[:number_images],
                                   second_embeddings=embeddings[number_images:], temperature=temperature)

    # Cosine similarities of the views, and the softmax of the row of the first view over the other views
    normalized = torch.nn.functional.normalize(input=embeddings, dim=1)
    cosines = (normalized @ normalized.T).numpy()
    row_logits = cosines[0] / temperature
    row_logits[0] = -numpy.inf
    row_probabilities = numpy.exp(row_logits - row_logits.max())
    row_probabilities /= row_probabilities.sum()
    return {"matrix.cosines": [[round(float(value), 2) for value in row] for row in cosines],
            "matrix.view_count": 2 * number_images, "matrix.other_view_count": 2 * number_images - 1,
            "matrix.info_nce": round_fact(value=float(losses["info_nce_loss"])),
            "matrix.random_info_nce": round_fact(value=math.log(2 * number_images - 1)),
            "matrix.positive_pair_accuracy": round_fact(value=float(losses["positive_pair_accuracy"])),
            "matrix.first_partner_cosine": round_fact(value=float(cosines[0, number_images])),
            "matrix.first_partner_probability": round_fact(value=float(row_probabilities[number_images])),
            "matrix.first_row_probabilities": [round_fact(value=float(value)) for value in row_probabilities],
            "matrix.first_row_probability_labels": [f"{value:.2f}" for value in row_probabilities],
            "matrix.first_row_loss": round_fact(value=float(-math.log(row_probabilities[number_images])))}


def convert_to_grayscale(images: torch.Tensor) -> torch.Tensor:
    """
    Turns BGR images into grayscale images kept on three channels, their luminance with GRAYSCALE_WEIGHTS, so that
    two views no longer share the colour of their shape.

    Args:
        images (torch.Tensor): The [B, 3, H, W] float BGR images.

    Returns:
        torch.Tensor: The [B, 3, H, W] grayscale images.
    """
    weights = torch.tensor(data=GRAYSCALE_WEIGHTS, dtype=images.dtype).view(1, 3, 1, 1)
    return (images * weights).sum(dim=1, keepdim=True).repeat(1, 3, 1, 1)


def measure_colour_cue(run_folder: Path, trained_model: SelfSupervisedModel, augmenter: Any,
                       samples: List[Dict[str, Any]], image_size: int, keep_ratio: bool) -> Dict[str, Any]:
    """
    Measures the positive pair accuracy of the views of a batch of test images in training mode, as the loss sees
    them (as many images as a batch of the run, the views drawn CUE_VIEW_DRAWS times and the accuracies averaged), for
    an untrained network (seeded) and for the trained one, on the colour views and on the same views turned to
    grayscale: a network finding the partner of a view above chance before any training uses a cue both views share,
    and the colour of the shape is one when it falls on grayscale views.

    Args:
        run_folder (Path): The folder of the self-supervised run.
        trained_model (SelfSupervisedModel): The model of the last checkpoint.
        augmenter (Any): The DataAugmenter of the views of the run.
        samples (List[Dict[str, Any]]): The test samples.
        image_size (int): The input size of the run.
        keep_ratio (bool): Whether the run keeps the aspect ratio.

    Returns:
        Dict[str, Any]: The facts of the cue, named cue.*.

    Raises:
        ValueError: If the untrained network does not fall on grayscale views, the colour cue that the scene of the
            second chapter shows having disappeared with the views of the current code.
    """
    # Untrained network of the run settings, seeded
    _, model_configuration = load_experiment_configuration(configuration_path=run_folder /
                                                           "training_configuration.yaml")
    torch.manual_seed(seed=CUE_SEED)
    untrained_model = SelfSupervisedModel(configuration=model_configuration, device=torch.device("cpu"),
                                          dtype=torch.float32)

    # Views of the first test images of a batch of the run, drawn several times with seeded generators
    batch_size = int(read_run_configuration(run_folder=run_folder)["ExperimentConfiguration"]["batch_size"])
    images = read_images(samples=samples[:batch_size], image_size=image_size, keep_ratio=keep_ratio)
    view_draws = [draw_sample_views(augmenter=augmenter, image=images, sample_identifier=CUE_SEED + draw_index)[:2]
                  for draw_index in range(CUE_VIEW_DRAWS)]
    facts = {"cue.batch_size": batch_size, "cue.chance": round_fact(value=1.0 / (2 * batch_size - 1)),
             "cue.other_view_count": 2 * batch_size - 1, "cue.view_draws": CUE_VIEW_DRAWS}

    # Accuracy and loss of both networks on the colour views and on the same views in grayscale, averaged over the draws
    for model_name, model in [("untrained", untrained_model), ("trained", trained_model)]:
        model.train()
        for view_name in ["colour", "grayscale"]:
            accuracies, losses = [], []
            for first_views, second_views in view_draws:
                if view_name == "grayscale":
                    first_views = convert_to_grayscale(images=first_views)
                    second_views = convert_to_grayscale(images=second_views)
                with torch.no_grad():
                    outputs = model(input_tensor=first_views, second_view_tensor=second_views)
                accuracies.append(float(outputs["positive_pair_accuracy"]))
                losses.append(float(outputs["info_nce_loss"]))
            facts[f"cue.{model_name}_{view_name}_accuracy"] = round_fact(value=float(numpy.mean(accuracies)))
            facts[f"cue.{model_name}_{view_name}_loss"] = round_fact(value=float(numpy.mean(losses)))
        model.eval()
    if facts["cue.untrained_colour_accuracy"] <= facts["cue.untrained_grayscale_accuracy"]:
        raise ValueError(f"The untrained network finds {facts['cue.untrained_colour_accuracy']:.4f} of the partners "
                         f"on colour views and {facts['cue.untrained_grayscale_accuracy']:.4f} on grayscale views: "
                         f"the colour is no longer a cue of the views, reword the colour cue scene of the second "
                         f"self-supervised chapter.")
    facts["cue.untrained_colour_over_chance"] = round_fact(value=facts["cue.untrained_colour_accuracy"] /
                                                           facts["cue.chance"])
    return facts


def measure_views_similarity(model: SelfSupervisedModel, augmenter: Any, sample: Dict[str, Any], image_size: int,
                             keep_ratio: bool) -> float:
    """
    Computes the views similarity of a sample image as the caption of the evaluation does: the cosine similarity of
    the representations of its two seeded views, the model in evaluation mode.

    Args:
        model (SelfSupervisedModel): The model of a checkpoint, in evaluation mode.
        augmenter (Any): The DataAugmenter of the views of the run.
        sample (Dict[str, Any]): The test sample.
        image_size (int): The input size of the run.
        keep_ratio (bool): Whether the run keeps the aspect ratio.

    Returns:
        float: The views similarity.
    """
    image = read_images(samples=[sample], image_size=image_size, keep_ratio=keep_ratio)
    first_view, second_view, _, _ = draw_sample_views(augmenter=augmenter, image=image,
                                                      sample_identifier=sample["identifier"])
    with torch.no_grad():
        representations = model.represent(images=torch.cat(tensors=[first_view, second_view], dim=0))
    return float(torch.nn.functional.cosine_similarity(x1=representations[0:1], x2=representations[1:2], dim=1))


def add_first_and_last_names(facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gives the facts measured at the first and last evaluated checkpoints a second name ending in _first or _last
    instead of the iteration, so that the storyboards quote them whatever the length of the run.

    Args:
        facts (Dict[str, Any]): The facts of the representation figures, with knn.first_iteration and
            knn.last_iteration.

    Returns:
        Dict[str, Any]: The facts with their second names.
    """
    named_facts = dict(facts)
    for position in ["first", "last"]:
        iteration_suffix = f"_{facts[f'knn.{position}_iteration']}"
        for name, value in facts.items():
            if name.endswith(iteration_suffix):
                named_facts[name[:-len(iteration_suffix)] + f"_{position}"] = value
    return named_facts


def make_representation_figures(output_folder: Path, run_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Makes the representation figures of the series from the checkpoints of the run on the CPU and gives their facts,
    the figures of the previous build being reused unless forced or incomplete, or made when the neighbourhood was
    drawn among the test images.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the self-supervised run.
        force (bool): Whether the figures are made again even when the previous build left them.

    Returns:
        Dict[str, Any]: The facts, named knn.*, matrix.*, cue.* and similarity.*.
    """
    representation_folder = output_folder / "representations"
    facts_path = representation_folder / FACTS_FILE_NAME
    reused_facts = read_reused_facts(facts_path=facts_path, required_paths=[], force=force)
    if reused_facts is not None and REFERENCE_FACT_NAME in reused_facts:
        make_neighbour_mosaic(output_folder=output_folder, run_folder=run_folder, facts=reused_facts)
        return add_first_and_last_names(facts=reused_facts)

    # Settings of the run, its star and the reference training images of its k-NN evaluation
    run_configuration = read_run_configuration(run_folder=run_folder)
    number_neighbours = int(run_configuration["ExperimentConfiguration"]["evaluation"]["number_neighbours"])
    temperature = float(run_configuration["ProjectionHead"]["simclr"]["temperature"])
    samples, class_names = read_test_samples(run_folder=run_folder)
    query_sample = find_sample(samples=samples, file_name=QUERY_NAME)
    reference_samples = read_reference_samples(run_folder=run_folder, class_names=class_names)
    image_size, keep_ratio = read_image_settings(run_folder=run_folder)
    augmenter = build_view_augmenter(run_folder=run_folder)
    evaluated_iterations = list_evaluated_iterations(run_folder=run_folder)
    facts = {"knn.reference_count": len(reference_samples),
             "knn.reference_images_per_class": int(run_configuration["ExperimentConfiguration"]["evaluation"][
                 "number_reference_images_per_class"]),
             "knn.query_class": query_sample["class_name"].replace("_", " "),
             "knn.number_neighbours": number_neighbours, "knn.temperature": NEAREST_NEIGHBOUR_TEMPERATURE}
    figure_paths = []

    # Every evaluated checkpoint: the views similarity of the star; the first and last also encode the reference images
    for iteration in evaluated_iterations:
        print(f"Self-supervised figures: checkpoint {iteration} on the CPU")
        model = load_self_supervised_model(run_folder=run_folder, iteration=iteration)
        facts[f"similarity.star_{iteration}"] = round_fact(value=measure_views_similarity(
            model=model, augmenter=augmenter, sample=query_sample, image_size=image_size, keep_ratio=keep_ratio))
        if iteration not in (evaluated_iterations[0], evaluated_iterations[-1]):
            continue
        query_representation = encode_images(model=model, samples=[query_sample], image_size=image_size,
                                             keep_ratio=keep_ratio)[0]
        reference_representations = encode_images(model=model, samples=reference_samples, image_size=image_size,
                                                  keep_ratio=keep_ratio)
        neighbourhood = describe_neighbourhood(query_representation=query_representation,
                                               reference_representations=reference_representations,
                                               reference_samples=reference_samples,
                                               number_neighbours=number_neighbours, number_classes=len(class_names))
        mosaic_indices = neighbourhood["order"][:MOSAIC_NEIGHBOUR_COUNT]
        facts[f"knn.nearest_{iteration}"] = [
            {"index": int(index), "similarity": round_fact(value=float(neighbourhood["similarities"][index]))}
            for index in mosaic_indices]
        facts[f"knn.mosaic_same_class_neighbours_{iteration}"] = int(sum(
            reference_samples[index]["class_index"] == query_sample["class_index"] for index in mosaic_indices))

        # Voting neighbours of the query at this checkpoint: their classes, similarities and vote
        neighbour_indices = neighbourhood["order"][:neighbourhood["number_voting_neighbours"]]
        facts[f"knn.query_same_class_neighbours_{iteration}"] = int(sum(
            reference_samples[index]["class_index"] == query_sample["class_index"] for index in neighbour_indices))
        facts[f"knn.query_prediction_{iteration}"] = class_names[neighbourhood["class_index"]].replace("_", " ")
        facts[f"knn.query_nearest_similarity_{iteration}"] = round_fact(
            value=float(neighbourhood["similarities"][neighbour_indices[0]]))
        facts[f"knn.query_last_neighbour_similarity_{iteration}"] = round_fact(
            value=float(neighbourhood["similarities"][neighbour_indices[-1]]))

        # The last checkpoint: the vote shares of the bars, the weights of the extreme neighbours and the radial layout
        if iteration == evaluated_iterations[-1]:
            vote_shares = neighbourhood["votes"] / neighbourhood["votes"].sum()
            facts["knn.vote_shares"] = [round_fact(value=share) for share in vote_shares]
            facts["knn.vote_labels"] = [name.replace("_", " ") for name in class_names]
            facts["knn.vote_value_labels"] = [f"{share:.0%}" if share > 0 else "" for share in vote_shares]
            facts["knn.vote_colours"] = CLASS_COLOURS[:len(class_names)]
            facts["knn.weight_ratio"] = round_fact(value=math.exp(
                (facts[f"knn.query_nearest_similarity_{iteration}"] -
                 facts[f"knn.query_last_neighbour_similarity_{iteration}"]) / NEAREST_NEIGHBOUR_TEMPERATURE))
            radial_path = representation_folder / "knn_radial_layout.json"
            write_radial_layout(output_path=radial_path, reference_samples=reference_samples,
                                query_sample=query_sample, neighbourhood=neighbourhood, class_names=class_names)
            figure_paths.append(radial_path)

            # What InfoNCE computes on the views of the sample images, and the cue both views share
            facts.update(measure_info_nce_matrix(model=model, augmenter=augmenter, samples=samples,
                                                 image_size=image_size, keep_ratio=keep_ratio,
                                                 temperature=temperature))
            facts.update(measure_colour_cue(run_folder=run_folder, trained_model=model, augmenter=augmenter,
                                            samples=samples, image_size=image_size, keep_ratio=keep_ratio))

    facts["knn.first_iteration"] = evaluated_iterations[0]
    facts["knn.last_iteration"] = evaluated_iterations[-1]
    write_reused_facts(facts_path=facts_path, figure_paths=figure_paths, facts=facts)
    make_neighbour_mosaic(output_folder=output_folder, run_folder=run_folder, facts=facts)
    return add_first_and_last_names(facts=facts)
