"""
Figures and facts of the keypoint detection series of the tutorial videos, made by make_figures.py (python -m
tutorials.figures.make_figures --series keypoint_detection) at every build, from the synthetic keypoint run of
tutorials/tutorial_sources.yaml, its dataset and the code of the repository: the labels, channels and targets of the
first chapter (dataset_figures.py), the shapes, parameters, losses, tags and training curves of the second
(network_figures.py), and the heatmaps, grouping, similarity curves, metrics and sample boxes of the third
(evaluation_figures.py). The forward pass of the trained detector on the CPU (keypoint_inference.py) is the only
expensive step, reused between builds unless forced. Every family writes its figures into a subfolder of the output
folder and gives its facts, gathered here into the facts.json the storyboards quote as {{keypoint_detection.<name>}}.
"""

from pathlib import Path
from typing import Any, Dict

from tutorials.figures.figure_utilities import find_run_folder, merge_facts
from tutorials.figures.keypoint_detection.dataset_figures import (draw_labelled_image, make_target_figures,
                                                                  write_channel_table, write_labels_listing)
from tutorials.figures.keypoint_detection.evaluation_figures import (describe_metrics,
                                                                     describe_object_keypoint_similarity,
                                                                     describe_sample_image, draw_heatmap_figure,
                                                                     draw_worst_figure, write_grouping_data)
from tutorials.figures.keypoint_detection.keypoint_inference import run_keypoint_inference
from tutorials.figures.keypoint_detection.keypoint_sources import (describe_sample_objects, find_test_split,
                                                                  load_test_sample, read_channel_table,
                                                                  read_label_dictionary, read_repository_configuration)
from tutorials.figures.keypoint_detection.network_figures import (describe_heatmap_losses, describe_network,
                                                                  describe_stem_and_run, describe_tags,
                                                                  draw_training_curves)
from vision_training.configuration_processing.configuration_loader import load_configuration


def make_figures(output_folder: Path, experiments_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Makes every figure of the keypoint detection series into its output folder and gives their facts, the inference
    of the trained detector being reused from the previous build unless forced.

    Args:
        output_folder (Path): The figures folder of the series in the cache of the build.
        experiments_folder (Path): The folder of the real runs.
        force (bool): Whether the inference runs again even when the previous build left its files.

    Returns:
        Dict[str, Any]: The facts of every figure family, keyed by their names.
    """
    run_folder = find_run_folder(experiments_folder=experiments_folder, run_key="keypoint_detection")
    configuration = read_repository_configuration(run_folder=run_folder)
    test_split = find_test_split(configuration=configuration)
    label_dictionary = read_label_dictionary(configuration=configuration)
    channel_table, channel_names, keypoint_categories = read_channel_table(run_folder=run_folder,
                                                                           test_split=test_split,
                                                                           label_dictionary=label_dictionary)
    sample_record = load_test_sample(configuration=configuration, test_split=test_split,
                                     label_dictionary=label_dictionary, maximum_keypoints=int(channel_table.shape[1]))
    objects = describe_sample_objects(sample=sample_record["sample"], keypoint_categories=keypoint_categories,
                                      channel_table=channel_table)
    head_configuration = configuration["KeypointHead"]

    # The expensive forward pass of the trained detector, then every family drawn from it or from the run files
    inference_facts = run_keypoint_inference(output_folder=output_folder, run_folder=run_folder,
                                             sample_record=sample_record, channel_table=channel_table,
                                             keypoint_categories=keypoint_categories, configuration=configuration,
                                             force=force)
    draw_labelled_image(output_folder=output_folder, sample_record=sample_record, objects=objects)
    draw_training_curves(output_folder=output_folder, run_folder=run_folder, with_curves=True)
    run_configuration = load_configuration(configuration_path=run_folder / "training_configuration.yaml")
    run_label = f"{experiments_folder.name}/{run_folder.relative_to(experiments_folder)}"
    return merge_facts(fact_groups=[
        {"run_label": run_label, "run_iterations": run_configuration["ExperimentConfiguration"]["number_iterations"],
         "run_batch_size": run_configuration["ExperimentConfiguration"]["batch_size"],
         "run_warmup_iterations": run_configuration["Optimization"]["warmup_iterations"],
         "run_decay_iterations": run_configuration["Optimization"]["decay_iterations"],
         "sample_image_name": sample_record["image_record"]["file_name"],
         "checkpoint_iteration": inference_facts["inference_checkpoint_iteration"]},
        write_labels_listing(output_folder=output_folder, sample_record=sample_record),
        write_channel_table(output_folder=output_folder, channel_table=channel_table,
                            keypoint_categories=keypoint_categories, objects=objects),
        make_target_figures(output_folder=output_folder, sample_record=sample_record, channel_table=channel_table,
                            objects=objects, heatmap_sigma=head_configuration["heatmap_sigma"]),
        describe_network(inference_facts=inference_facts, configuration=configuration, output_folder=output_folder),
        describe_heatmap_losses(inference_facts=inference_facts, sample_record=sample_record,
                                channel_table=channel_table, heatmap_sigma=head_configuration["heatmap_sigma"]),
        describe_tags(inference_facts=inference_facts, output_folder=output_folder),
        describe_stem_and_run(configuration=configuration, run_folder=run_folder),
        draw_training_curves(output_folder=output_folder, run_folder=run_folder, with_curves=False),
        draw_heatmap_figure(output_folder=output_folder, sample_record=sample_record, inference_facts=inference_facts,
                            channel_table=channel_table,
                            keypoint_threshold=head_configuration["inference"]["keypoint_threshold"]),
        write_grouping_data(output_folder=output_folder, inference_facts=inference_facts, channel_table=channel_table,
                            inference_configuration=head_configuration["inference"]),
        describe_object_keypoint_similarity(sample_record=sample_record, inference_facts=inference_facts,
                                            default_sigma=configuration["ExperimentConfiguration"]["evaluation"][
                                                "default_keypoint_sigma"]),
        describe_metrics(run_folder=run_folder),
        draw_worst_figure(output_folder=output_folder, run_folder=run_folder),
        describe_sample_image(run_folder=run_folder, inference_facts=inference_facts)])
