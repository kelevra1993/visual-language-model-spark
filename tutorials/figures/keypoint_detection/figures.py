"""
Figures and facts of the keypoint detection series of the tutorial videos, made by make_figures.py (python -m
tutorials.figures.make_figures --series keypoint_detection) at every build, from the synthetic keypoint run of
tutorials/tutorial_sources.yaml, its dataset and the code of the repository: the labels, channels and targets of the
first chapter (dataset_figures.py), the shapes, parameters, losses, tags and training curves of the second
(network_figures.py), and the heatmaps, grouping, similarity curves, metrics and sample boxes of the third
(evaluation_figures.py). The forward pass of the trained detector on the CPU (keypoint_inference.py) is the only
expensive step, reused between builds unless forced, and taken from the committed snapshots (tutorials/run_snapshots,
snapshot_files.yaml) when the run read has no checkpoint. Every family writes its figures into a subfolder of the
output folder and gives its facts, gathered here into the facts.json the storyboards quote as
{{keypoint_detection.<name>}}.
"""

from pathlib import Path
from typing import Any, Dict

from tutorials.figures.figure_utilities import describe_run_label, find_run_folder, merge_facts
from tutorials.figures.keypoint_detection.dataset_figures import (draw_labelled_image, make_target_figures,
                                                                  write_channel_table, write_labels_listing)
from tutorials.figures.keypoint_detection.evaluation_figures import (describe_metrics,
                                                                     describe_object_keypoint_similarity,
                                                                     describe_sample_image, draw_heatmap_figure,
                                                                     draw_worst_figure, write_grouping_data)
from tutorials.figures.keypoint_detection.keypoint_inference import run_keypoint_inference
from tutorials.figures.keypoint_detection.keypoint_sources import (describe_sample_objects, find_test_split,
                                                                  load_test_sample, read_channel_table,
                                                                  read_label_dictionary, read_repository_configuration,
                                                                  select_story_objects)
from tutorials.figures.keypoint_detection.network_figures import (describe_architecture_constants,
                                                                  describe_heatmap_losses, describe_network,
                                                                  describe_stem_and_run, describe_tags,
                                                                  draw_training_curves)
from vision_training.configuration_processing.configuration_loader import load_configuration
from vision_training.data.augmentations import SUPERVISED_AUGMENTATION_STRENGTHS


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
    channel_table, channel_names, keypoint_categories = read_channel_table(run_folder=run_folder, test_split=test_split,
                                                                           label_dictionary=label_dictionary)
    sample_record = load_test_sample(configuration=configuration, test_split=test_split,
                                     label_dictionary=label_dictionary, maximum_keypoints=int(channel_table.shape[1]))
    objects = describe_sample_objects(sample=sample_record["sample"], keypoint_categories=keypoint_categories,
                                      channel_table=channel_table)
    story_objects = select_story_objects(objects=objects, sample=sample_record["sample"])
    head_configuration = configuration["KeypointHead"]
    architecture_facts = describe_architecture_constants(configuration=configuration)

    # The expensive forward pass of the trained detector, then every family drawn from it or from the run files
    inference_facts = run_keypoint_inference(output_folder=output_folder, run_folder=run_folder,
                                             sample_record=sample_record, channel_table=channel_table,
                                             keypoint_categories=keypoint_categories, configuration=configuration,
                                             force=force)
    draw_labelled_image(output_folder=output_folder, sample_record=sample_record, objects=objects,
                        story_objects=story_objects)
    draw_training_curves(output_folder=output_folder, run_folder=run_folder, with_curves=True,
                         tag_loss_weight=architecture_facts["tag_loss_weight"])
    run_configuration = load_configuration(configuration_path=run_folder / "training_configuration.yaml")
    run_label = describe_run_label(experiments_folder=experiments_folder, run_folder=run_folder)

    # Facts of every family, with the objects named by their role, the settings of the run and the largest rotation of
    # the augmentations of the supervised tasks (SUPERVISED_AUGMENTATION_STRENGTHS, whose computed values a constant
    # placeholder cannot read)
    return merge_facts(fact_groups=[
        architecture_facts, story_objects["facts"],
        {"run_label": run_label, "run_iterations": run_configuration["ExperimentConfiguration"]["number_iterations"],
         "run_batch_size": run_configuration["ExperimentConfiguration"]["batch_size"],
         "run_evaluation_interval": run_configuration["ExperimentConfiguration"]["weight_saving_iterations"],
         "run_validation_iteration_ratio": run_configuration["ExperimentConfiguration"]["validation_iteration_ratio"],
         "run_warmup_iterations": run_configuration["Optimization"]["warmup_iterations"],
         "run_decay_iterations": run_configuration["Optimization"]["decay_iterations"],
         "sample_image_name": sample_record["image_record"]["file_name"],
         "checkpoint_iteration": inference_facts["inference_checkpoint_iteration"],
         "augmentation_maximum_rotation_degrees": SUPERVISED_AUGMENTATION_STRENGTHS["maximum_rotation_degrees"]},

        # Labels listing, channel table and targets of the first chapter
        write_labels_listing(output_folder=output_folder, sample_record=sample_record, objects=objects,
                             story_objects=story_objects),
        write_channel_table(output_folder=output_folder, channel_table=channel_table,
                            keypoint_categories=keypoint_categories, objects=objects, story_objects=story_objects),
        make_target_figures(output_folder=output_folder, sample_record=sample_record, channel_table=channel_table,
                            objects=objects, story_objects=story_objects,
                            heatmap_sigma=head_configuration["heatmap_sigma"]),

        # Shapes, parameters, heatmap losses, tags, stem and training curve axes of the second chapter
        describe_network(inference_facts=inference_facts, configuration=configuration, output_folder=output_folder),
        describe_heatmap_losses(inference_facts=inference_facts, sample_record=sample_record,
                                channel_table=channel_table, heatmap_sigma=head_configuration["heatmap_sigma"]),
        describe_tags(inference_facts=inference_facts, story_objects=story_objects, output_folder=output_folder),
        describe_stem_and_run(configuration=configuration, run_folder=run_folder),
        draw_training_curves(output_folder=output_folder, run_folder=run_folder, with_curves=False,
                             tag_loss_weight=architecture_facts["tag_loss_weight"]),

        # Heatmaps, grouping, similarity, metrics, worst keypoints and sample image of the third chapter
        draw_heatmap_figure(output_folder=output_folder, sample_record=sample_record, inference_facts=inference_facts,
                            channel_table=channel_table,
                            keypoint_threshold=head_configuration["inference"]["keypoint_threshold"],
                            story_objects=story_objects),
        write_grouping_data(output_folder=output_folder, inference_facts=inference_facts, channel_table=channel_table,
                            inference_configuration=head_configuration["inference"], story_objects=story_objects),
        describe_object_keypoint_similarity(sample_record=sample_record, inference_facts=inference_facts,
                                            default_sigma=configuration["ExperimentConfiguration"]["evaluation"][
                                                "default_keypoint_sigma"], story_objects=story_objects),
        describe_metrics(run_folder=run_folder),
        draw_worst_figure(output_folder=output_folder, run_folder=run_folder),
        describe_sample_image(run_folder=run_folder, sample_record=sample_record, inference_facts=inference_facts,
                              story_objects=story_objects)])
