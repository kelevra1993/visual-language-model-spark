"""
Representation figures of the self-supervised series of the tutorial videos (chapters 2 and 3), made by
tutorials/figures/self_supervised/figures.py from the checkpoints of the SimCLR run rebuilt on the CPU: the
representations of the first test images encoded by the exported InferenceEncoder at the first and last evaluated
checkpoints, scored by their effective rank; the neighbourhood of the star of the sample images among these test
images, as the k-NN compared the test images with each other when the run was evaluated (its nearest test images by
cosine similarity, the weighted vote of its 20 neighbours, and a
list of its nearest images for the animated page animations/11_self_supervised/knn_vote.html); a mosaic of its
five nearest images at both checkpoints; the cosine similarities of the embeddings of the two views of four sample
images, the InfoNCE matrix of chapter 2; the views similarity of the star at every evaluated checkpoint; and the
positive pair accuracy of an untrained and of the trained network on colour and on grayscale views, the cue the two
views share. The encoding takes minutes on a shared CPU, so the figures are reused unless forced.
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
from tutorials.figures.self_supervised.ssl_sources import (build_view_augmenter, draw_sample_views,
                                                          encode_test_images, find_sample, list_evaluated_iterations,
                                                          load_self_supervised_model, read_image_settings,
                                                          read_test_images, read_test_samples)
from tutorials.figures.self_supervised.view_figures import VIEW_SAMPLE_NAMES
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.evaluation.self_supervised.representation_metrics import (NEAREST_NEIGHBOUR_TEMPERATURE,
                                                                               compute_effective_rank)
from vision_training.losses.self_supervised_losses import compute_info_nce_loss
from vision_training.models.self_supervised.self_supervised_model import SelfSupervisedModel

# Number of first test images encoded at both checkpoints, a subset of the 50,000 the run evaluates
SUBSET_SIZE = 1000

# Test image whose neighbourhood the figures show: the star of the sample images
QUERY_NAME = "test_000000.png"

# Nearest images of the query shown by the radial layout of the animated page, and by the mosaic
RADIAL_POINT_COUNT = 150
MOSAIC_NEIGHBOUR_COUNT = 5

# Images of the batch whose views the positive pair accuracy of the colour cue is measured on, and the seed of the
# untrained network and of the views
CUE_BATCH_SIZE = 64
CUE_SEED = 0

# Weights of the blue, green and red channels in the luminance of the grayscale views (ITU-R BT.601, as OpenCV)
GRAYSCALE_WEIGHTS = (0.114, 0.587, 0.299)

# Colours of the nine classes in the animated page and the mosaic captions, the series colours of the frames first
CLASS_COLOURS = ["#3987e5", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767", "#7dc4ff", "#f0c878", "#78dc78"]

# Name of the JSON file keeping the facts of the figures for their reuse
FACTS_FILE_NAME = "representation_facts.json"


def describe_neighbourhood(representations: numpy.ndarray, samples: List[Dict[str, Any]], query_index: int,
                           number_neighbours: int) -> Dict[str, Any]:
    """
    Gives the neighbourhood of a query image among the encoded test images, as the k-NN evaluation sees it: the cosine
    similarity of every image to the query, the query itself left out, its nearest images in order, and the vote of
    its neighbours, each weighing exp(similarity / NEAREST_NEIGHBOUR_TEMPERATURE).

    Args:
        representations (numpy.ndarray): The [N, F] normalized representations.
        samples (List[Dict[str, Any]]): The encoded test samples, in the order of the representations.
        query_index (int): The index of the query among them.
        number_neighbours (int): The number k of voting neighbours of the run.

    Returns:
        Dict[str, Any]: The 'similarities' [N] (the query at minus infinity), the 'order' of the other images from the
            nearest, the 'votes' of every class and the predicted 'class_index'.
    """
    similarities = representations @ representations[query_index]
    similarities[query_index] = -numpy.inf
    order = numpy.argsort(-similarities)
    neighbour_indices = order[:number_neighbours]
    number_classes = max(sample["class_index"] for sample in samples) + 1
    votes = numpy.zeros(shape=number_classes, dtype=numpy.float64)
    for neighbour_index in neighbour_indices:
        votes[samples[neighbour_index]["class_index"]] += math.exp(similarities[neighbour_index] /
                                                                   NEAREST_NEIGHBOUR_TEMPERATURE)
    return {"similarities": similarities, "order": order, "votes": votes, "class_index": int(numpy.argmax(votes))}


def write_radial_layout(output_path: Path, samples: List[Dict[str, Any]], query_index: int,
                        neighbourhood: Dict[str, Any], number_neighbours: int, class_names: List[str]) -> None:
    """
    Writes the data of the animated page of the k-NN vote: the RADIAL_POINT_COUNT nearest images of the query in
    order, each with its cosine similarity to the query, its class and its rank, the page placing every image at a
    distance from the centre growing with 1 - similarity, the k neighbours inside the ring of the k-th one.

    Args:
        output_path (Path): The JSON file written.
        samples (List[Dict[str, Any]]): The encoded test samples.
        query_index (int): The index of the query.
        neighbourhood (Dict[str, Any]): The neighbourhood of the query, from describe_neighbourhood.
        number_neighbours (int): The number k of voting neighbours.
        class_names (List[str]): The class names in class index order.
    """
    shown_indices = neighbourhood["order"][:RADIAL_POINT_COUNT]
    points = [{"similarity": round(float(neighbourhood["similarities"][image_index]), 5),
               "class_index": samples[image_index]["class_index"], "rank": rank}
              for rank, image_index in enumerate(shown_indices)]
    layout = {"points": points, "number_neighbours": number_neighbours,
              "query_class_index": samples[query_index]["class_index"], "class_names": class_names,
              "class_colours": CLASS_COLOURS}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(data=json.dumps(obj=layout), encoding="utf-8")


def draw_neighbour_mosaic(output_path: Path, rows: List[Dict[str, Any]], samples: List[Dict[str, Any]],
                          image_size: int, keep_ratio: bool) -> None:
    """
    Draws the mosaic of the nearest test images of the query at several checkpoints: one row per checkpoint, the query
    then its nearest images, every image captioned with its class and its cosine similarity to the query, the class in
    its colour.

    Args:
        output_path (Path): The PNG file written.
        rows (List[Dict[str, Any]]): Every row: its 'iteration', the 'query_index' and its 'neighbours' (their 'index'
            among the samples and their 'similarity' to the query), nearest first.
        samples (List[Dict[str, Any]]): The encoded test samples.
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
        shown_images = [{"index": row["query_index"], "similarity": None}] + row["neighbours"]

        # The query, then its nearest images with their class and similarity
        for column_index, shown_image in enumerate(shown_images):
            left = (left_pixels + column_index * (tile_pixels + gap_pixels) +
                    (2 * gap_pixels if column_index > 0 else 0)) / CHART_WIDTH
            axes = figure.add_axes(rect=(left, bottom, tile_width, tile_height))
            sample = samples[shown_image["index"]]
            image = read_test_images(samples=[sample], image_size=image_size, keep_ratio=keep_ratio)
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
    Draws the mosaic of the nearest test images of the query at the first and last evaluated checkpoints from the
    neighbours kept in the facts of the representation figures, so that its layout can change without encoding the
    test images again.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the self-supervised run.
        facts (Dict[str, Any]): The facts of the representation figures, with knn.query_index and the
            knn.nearest_<iteration> neighbours of both checkpoints.
    """
    samples, _ = read_test_samples(run_folder=run_folder)
    image_size, keep_ratio = read_image_settings(run_folder=run_folder)
    rows = [{"iteration": facts[f"knn.{position}_iteration"], "query_index": facts["knn.query_index"],
             "neighbours": facts[f"knn.nearest_{facts[f'knn.{position}_iteration']}"]}
            for position in ["first", "last"]]
    draw_neighbour_mosaic(output_path=output_folder / "representations" / "neighbour_mosaic.png", rows=rows,
                          samples=samples[:facts["knn.subset_size"]], image_size=image_size, keep_ratio=keep_ratio)


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
        image = read_test_images(samples=[sample], image_size=image_size, keep_ratio=keep_ratio)
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
    them, for an untrained network (seeded) and for the trained one, on the colour views and on the same views turned
    to grayscale: a network finding the partner of a view far above chance before any training uses a cue both views
    share, and the colour of the shape is one when it falls on grayscale views.

    Args:
        run_folder (Path): The folder of the self-supervised run.
        trained_model (SelfSupervisedModel): The model of the last checkpoint.
        augmenter (Any): The DataAugmenter of the views of the run.
        samples (List[Dict[str, Any]]): The test samples.
        image_size (int): The input size of the run.
        keep_ratio (bool): Whether the run keeps the aspect ratio.

    Returns:
        Dict[str, Any]: The facts of the cue, named cue.*.
    """
    # Untrained network of the run settings, seeded
    _, model_configuration = load_experiment_configuration(configuration_path=run_folder /
                                                           "training_configuration.yaml")
    torch.manual_seed(seed=CUE_SEED)
    untrained_model = SelfSupervisedModel(configuration=model_configuration, device=torch.device("cpu"),
                                          dtype=torch.float32)

    # Views of the first test images, drawn with seeded generators
    images = read_test_images(samples=samples[:CUE_BATCH_SIZE], image_size=image_size, keep_ratio=keep_ratio)
    first_views, second_views, _, _ = draw_sample_views(augmenter=augmenter, image=images, sample_identifier=CUE_SEED)
    facts = {"cue.batch_size": CUE_BATCH_SIZE, "cue.chance": round_fact(value=1.0 / (2 * CUE_BATCH_SIZE - 1)),
             "cue.other_view_count": 2 * CUE_BATCH_SIZE - 1}
    for model_name, model in [("untrained", untrained_model), ("trained", trained_model)]:
        model.train()
        for view_name, (first, second) in [("colour", (first_views, second_views)),
                                           ("grayscale", (convert_to_grayscale(images=first_views),
                                                          convert_to_grayscale(images=second_views)))]:
            with torch.no_grad():
                outputs = model(input_tensor=first, second_view_tensor=second)
            facts[f"cue.{model_name}_{view_name}_accuracy"] = round_fact(value=float(outputs["positive_pair_accuracy"]))
            facts[f"cue.{model_name}_{view_name}_loss"] = round_fact(value=float(outputs["info_nce_loss"]))
        model.eval()
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
    image = read_test_images(samples=[sample], image_size=image_size, keep_ratio=keep_ratio)
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
    the figures of the previous build being reused unless forced or incomplete.

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
    if reused_facts is not None:
        make_neighbour_mosaic(output_folder=output_folder, run_folder=run_folder, facts=reused_facts)
        return add_first_and_last_names(facts=reused_facts)

    # Settings of the run and the encoded subset of its test split
    run_configuration = read_run_configuration(run_folder=run_folder)
    number_neighbours = int(run_configuration["ExperimentConfiguration"]["evaluation"]["number_neighbours"])
    temperature = float(run_configuration["ProjectionHead"]["simclr"]["temperature"])
    samples, class_names = read_test_samples(run_folder=run_folder)
    subset = samples[:SUBSET_SIZE]
    image_size, keep_ratio = read_image_settings(run_folder=run_folder)
    augmenter = build_view_augmenter(run_folder=run_folder)
    evaluated_iterations = list_evaluated_iterations(run_folder=run_folder)
    query_index = [sample["path"].name for sample in subset].index(QUERY_NAME)
    facts = {"knn.subset_size": SUBSET_SIZE, "knn.query_class": subset[query_index]["class_name"].replace("_", " "),
             "knn.number_neighbours": number_neighbours,
             "knn.temperature": NEAREST_NEIGHBOUR_TEMPERATURE}
    figure_paths = []

    # Every evaluated checkpoint: the views similarity of the star; the first and last also encode the subset
    for iteration in evaluated_iterations:
        print(f"Self-supervised figures: checkpoint {iteration} on the CPU")
        model = load_self_supervised_model(run_folder=run_folder, iteration=iteration)
        facts[f"similarity.star_{iteration}"] = round_fact(value=measure_views_similarity(
            model=model, augmenter=augmenter, sample=subset[query_index], image_size=image_size,
            keep_ratio=keep_ratio))
        if iteration not in (evaluated_iterations[0], evaluated_iterations[-1]):
            continue
        representations = encode_test_images(model=model, samples=subset, image_size=image_size,
                                             keep_ratio=keep_ratio)
        facts[f"knn.subset_effective_rank_{iteration}"] = round_fact(value=compute_effective_rank(
            representations=torch.from_numpy(representations)))
        neighbourhood = describe_neighbourhood(representations=representations, samples=subset,
                                               query_index=query_index, number_neighbours=number_neighbours)
        facts[f"knn.nearest_{iteration}"] = [
            {"index": int(index), "similarity": round_fact(value=float(neighbourhood["similarities"][index]))}
            for index in neighbourhood["order"][:MOSAIC_NEIGHBOUR_COUNT]]

        # Neighbours of the query at this checkpoint: their classes, similarities and vote
        neighbour_indices = neighbourhood["order"][:number_neighbours]
        neighbour_classes = [subset[index]["class_index"] for index in neighbour_indices]
        facts[f"knn.query_same_class_neighbours_{iteration}"] = int(sum(
            class_index == subset[query_index]["class_index"] for class_index in neighbour_classes))
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
            write_radial_layout(output_path=radial_path, samples=subset, query_index=query_index,
                                neighbourhood=neighbourhood, number_neighbours=number_neighbours,
                                class_names=class_names)
            figure_paths.append(radial_path)

            # What InfoNCE computes on the views of the sample images, and the cue both views share
            facts.update(measure_info_nce_matrix(model=model, augmenter=augmenter, samples=samples,
                                                 image_size=image_size, keep_ratio=keep_ratio,
                                                 temperature=temperature))
            facts.update(measure_colour_cue(run_folder=run_folder, trained_model=model, augmenter=augmenter,
                                            samples=samples, image_size=image_size, keep_ratio=keep_ratio))

    facts["knn.query_index"] = query_index
    facts["knn.first_iteration"] = evaluated_iterations[0]
    facts["knn.last_iteration"] = evaluated_iterations[-1]
    write_reused_facts(facts_path=facts_path, figure_paths=figure_paths, facts=facts)
    make_neighbour_mosaic(output_folder=output_folder, run_folder=run_folder, facts=facts)
    return add_first_and_last_names(facts=facts)
