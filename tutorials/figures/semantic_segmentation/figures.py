"""
Figures and facts of the semantic segmentation series of the tutorial videos, made by make_figures.py (python -m
tutorials.figures.make_figures --series semantic_segmentation) at every build, from the synthetic segmentation run of
tutorials/tutorial_sources.yaml, its dataset and the code of the repository: the manifest and mask figures of the data
scenes (dataset_figures.py), the shapes, parameters, shortcuts and level maps of the model scenes, the decoder
shortcuts of the shipped synthetic configuration and the export specification of the segmenter (model_figures.py),
the curves, pixel shares and throughput of the training scenes (training_figures.py), and the samples, confusion
matrix, boundary metrics and headline metrics of the evaluation scenes (evaluation_figures.py). The forward pass of
the trained segmenter on the CPU (segmentation_inference.py) is the only expensive step, reused between builds unless
forced, and taken from the committed snapshots (tutorials/run_snapshots, snapshot_files.yaml) when the run read has no
checkpoint. Every family writes its figures into a subfolder of the output folder and gives its facts, gathered here
into the facts.json the storyboards quote as {{semantic_segmentation.<name>}}.
"""

from pathlib import Path
from typing import Any, Dict

from tutorials.figures.figure_utilities import (describe_run_label, find_run_folder, merge_facts,
                                                read_run_configuration)
from tutorials.figures.semantic_segmentation.dataset_figures import draw_mask_figure, write_manifest_listing
from tutorials.figures.semantic_segmentation.evaluation_figures import (describe_confusion_matrix,
                                                                        describe_headline_metrics,
                                                                        draw_boundary_figure, stack_sample_images,
                                                                        write_evaluation_listing)
from tutorials.figures.semantic_segmentation.model_figures import (count_configuration_decoder_shortcuts,
                                                                   describe_export_specification, describe_model,
                                                                   draw_level_figure)
from tutorials.figures.semantic_segmentation.segmentation_inference import run_segmentation_inference
from tutorials.figures.semantic_segmentation.segmentation_sources import (CHECKPOINT_ITERATION, EVALUATED_ITERATIONS,
                                                                         SAMPLE_IMAGE_NAME, TASK_NAME, find_test_split,
                                                                         load_test_sample, read_label_dictionary,
                                                                         read_run_dataset_configuration)
from tutorials.figures.semantic_segmentation.training_figures import (describe_pixel_shares, draw_training_curves,
                                                                      read_training_throughput)
from vision_training.data.synthetic.create_synthetic_dataset import TASK_CATEGORIES
from vision_training.evaluation.segmentation.boundary_metrics import BOUNDARY_COUNT_COLUMNS


def make_figures(output_folder: Path, experiments_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Makes every figure of the semantic segmentation series into its output folder and gives their facts, the
    inference of the trained segmenter being reused from the previous build unless forced.

    Args:
        output_folder (Path): The figures folder of the series in the cache of the build.
        experiments_folder (Path): The folder of the real runs.
        force (bool): Whether the inference runs again even when the previous build left its files.

    Returns:
        Dict[str, Any]: The facts of every figure family, keyed by their names.
    """
    # Configuration of the run, its dataset paths leading to the files it was trained on
    run_folder = find_run_folder(experiments_folder=experiments_folder, run_key="semantic_segmentation")
    configuration = read_run_dataset_configuration(run_folder=run_folder,
                                                   configuration=read_run_configuration(run_folder=run_folder))
    test_split = find_test_split(configuration=configuration)
    label_dictionary = read_label_dictionary(configuration=configuration)
    sample_record = load_test_sample(configuration=configuration, test_split=test_split,
                                     label_dictionary=label_dictionary)
    number_classes = len(sample_record["dataset"].class_names)

    # The expensive forward pass of the trained segmenter, then every family drawn from it or from the run files
    inference_facts = run_segmentation_inference(output_folder=output_folder, run_folder=run_folder,
                                                 sample_record=sample_record, force=force)
    draw_training_curves(output_folder=output_folder, run_folder=run_folder, with_curves=True)
    experiment_configuration = configuration["ExperimentConfiguration"]

    # Settings of the run, the checkpoints the figures read, the shapes the generator of the repository draws for the
    # task and the boundary counts the evaluation keeps per class
    return merge_facts(fact_groups=[
        {"run_label": describe_run_label(experiments_folder=experiments_folder, run_folder=run_folder),
         "run_iterations": experiment_configuration["number_iterations"],
         "run_batch_size": experiment_configuration["batch_size"],
         "run_validation_ratio": experiment_configuration["validation_iteration_ratio"],
         "run_learning_rate": experiment_configuration["learning_rate"],
         "run_learning_rate_schedule": configuration["Optimization"]["learning_rate_schedule"],
         "run_boundary_tolerance": experiment_configuration["evaluation"]["boundary_tolerance"],
         "run_image_size": configuration["Data"]["image_settings"]["size"],
         "run_checkpoint_iteration": CHECKPOINT_ITERATION, "run_first_evaluated_iteration": EVALUATED_ITERATIONS[0],
         "run_number_classes": number_classes, "sample_image_name": sample_record["manifest_entry"]["file_name"],
         "sample_image_number": int(Path(SAMPLE_IMAGE_NAME).stem.split("_")[-1]),
         "generator_shape_categories": len(TASK_CATEGORIES[TASK_NAME]),
         "evaluation_boundary_counts": len(BOUNDARY_COUNT_COLUMNS)},
        write_manifest_listing(output_folder=output_folder, sample_record=sample_record,
                               label_mapping_path=Path(configuration["Data"]["label_mapping"]["mapping_file_path"])),
        draw_mask_figure(output_folder=output_folder, sample_record=sample_record, label_dictionary=label_dictionary,
                         label_mode=configuration["Data"]["label_mapping"]["mode"]),
        describe_model(inference_facts=inference_facts, configuration=configuration),
        count_configuration_decoder_shortcuts(),
        describe_export_specification(run_folder=run_folder),
        draw_level_figure(output_folder=output_folder, inference_facts=inference_facts, number_classes=number_classes),
        draw_training_curves(output_folder=output_folder, run_folder=run_folder, with_curves=False),
        describe_pixel_shares(run_folder=run_folder, number_classes=number_classes),
        read_training_throughput(run_folder=run_folder),
        stack_sample_images(output_folder=output_folder, run_folder=run_folder),
        describe_confusion_matrix(run_folder=run_folder),
        draw_boundary_figure(output_folder=output_folder, run_folder=run_folder,
                             boundary_tolerance=experiment_configuration["evaluation"]["boundary_tolerance"]),
        describe_headline_metrics(run_folder=run_folder),
        write_evaluation_listing(output_folder=output_folder, run_folder=run_folder)])
