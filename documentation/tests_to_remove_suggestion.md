# Tests To Remove: Suggestion

In order to make the test suite quick enough to launch often, this document proposes which tests to remove or to cut down, chosen both for their run time and for their low stake (a behaviour already covered by another test, trivial code, wording, formatting, PyTorch behaviour, or near-identical parameter cases). The proposal was applied on 2026-11-04, as the section below records.

## Applied On 2026-11-04

Since the proposal was applied as written, this section records what changed and what the reduced suite no longer protects.

- **Result**: 526 test cases left (505 kept and 21 cases of the 16 restored tests below), all passing in 325 seconds (1673 cases in 655 seconds before), 50% less time. The slowest tests left are the end-to-end trainings of the sequence labelling (105 to 135 seconds), the keypoint detection (67 seconds) and the restored self-supervised run on mixed labelled and unlabelled images (33 seconds, 2.5 seconds when the other self-supervised trainings ran before it).
- **Files deleted**, every test of them removed: `tests/evaluation/detection/test_object_sizes.py`, `tests/evaluation/test_evaluation_report.py`, `tests/models/backbones/test_backbone_summary.py`, `tests/models/keypoints/test_hrnet.py`, `tests/trainers/test_input_viewer_panels.py`. `tests/evaluation/classification/test_class_metrics_report.py`, `tests/scripts/configuration_processing/test_convert_fixed_settings.py` and `tests/scripts/data/test_visualize_augmentation_strengths.py` lost every test too but each holds one restored test (below).
- **Functions left without any test**: `build_keypoint_channel_table`, `compute_skip_block_numbers`, `compare_instance_segmentations`, and the headless (no pyplot) check of the confusion matrix image.
- **Reference files no longer read**: 12 `region_proposal_*.pt` and 8 `convolution_block_modules_*.pt` files in `tests/models/components/reference_values/`, deleted with the user's approval.
- **Kept cases chosen when the proposal named none**: `[frame]` for `test_collate_pads_the_batches_to_their_longest_sequence` (the only case checking the -100 label padding), the first case elsewhere; `test_inconsistent_configuration_is_rejected` keeps the 6 named cases although its reason listed 6 cases to drop.

The two passes did not see each other's verdicts, so the 20 removals below relied on a test that the other pass removed too. So that every behaviour keeps one test, the user chose to restore the 16 tests they relied on (the second name of each line), back with their original cases:

- `tests/trainers/test_self_supervised_trainer.py::test_dataset_without_labels_is_evaluated_on_its_representations`, relying on `test_dataset_mixing_labelled_and_unlabelled_images_is_trained_and_evaluated`
- `tests/trainers/test_metric_tracking.py::test_metric_changes_and_best_marker`, relying on `test_correlations_improve_when_they_decrease`
- `tests/configuration_processing/test_run_name.py::test_sequence_augmentation_keeps_run_name`, relying on `test_dropout_rates_keep_run_name`
- `tests/data/test_augmentations.py::test_report_lines_of_sequences_and_of_unaugmented_samples`, relying on `test_reports_name_the_drawn_types_their_values_and_the_removed_objects`
- `tests/data/classification/test_imagenet_download.py::test_read_imagenet_class_names`, relying on `test_extract_imagenet_shards`
- `tests/data/test_self_supervised_views.py::test_views_without_any_type_are_the_images`, relying on `test_disabled_augmentation_is_identity`
- `tests/evaluation/test_evaluation_report.py::test_class_metrics_file`, relying on `test_file_values_are_formatted`
- `tests/evaluation/test_recall_intervals.py::test_bootstrap_interval_widens_with_correlated_samples`, relying on `test_bootstrap_interval_widens_for_few_groups`
- `tests/evaluation/classification/test_class_metrics_report.py::test_class_metrics_table`, relying on `test_class_metrics_rows`
- `tests/evaluation/classification/test_classification_metrics.py::test_class_absent_from_the_evaluation_is_left_out_of_the_averages`, relying on `test_class_predicted_but_absent_lowers_the_macro_f1`
- `tests/evaluation/classification/test_classification_metrics.py::test_multi_label_class_predicted_but_absent_lowers_the_macro_f1`, relying on `test_class_predicted_but_absent_lowers_the_macro_f1`
- `tests/evaluation/detection/test_mean_average_precision.py::test_perfect_detections_give_one`, relying on `test_per_class_average_precisions_and_ground_truths`
- `tests/evaluation/detection/test_region_recall.py::test_group_recalls_of_every_size`, relying on `test_anchor_recall_rows_and_table`
- `tests/evaluation/keypoints/test_keypoint_average_precision.py::test_keypoint_report_prefixes_its_tables_scalars_and_columns`, relying on `test_box_and_mask_tables_are_told_apart_by_their_prefixes`
- `tests/scripts/data/test_visualize_augmentation_strengths.py::test_example_images_come_from_different_trucks_and_categories`, relying on `test_figures_are_written_for_every_type`
- `tests/scripts/data/test_visualize_augmentation_strengths.py::test_type_figure_shows_the_image_then_both_strengths`, relying on `test_figures_are_written_for_every_type`
- `tests/scripts/data/test_visualize_augmentation_strengths.py::test_crop_ratio_figure_frames_and_resizes_the_region_of_each_ratio`, relying on `test_figures_are_written_for_every_type`
- `tests/scripts/data/test_visualize_augmentation_strengths.py::test_noise_figure_adds_the_noise_of_every_level`, relying on `test_figures_are_written_for_every_type`
- `tests/scripts/configuration_processing/test_convert_fixed_settings.py::test_tasks_holding_a_setting_keep_it`, relying on `test_conversion_lists_the_changed_values_and_refuses_other_changes`
- `tests/utilities/test_visualization.py::test_class_colours_are_distinct_with_a_black_background`, relying on `test_overlay_paints_the_classes_and_grays_the_ignored_pixels`

## How The Proposal Was Made

Should you wonder where the numbers come from, the suite of main d6e9d43 (2026-11-04, 1673 test cases, all passing) was run once with `pytest --durations=0 --junitxml`, on an idle GPU, in 655 seconds (651 seconds measured inside the tests, collection taking only 4.3 seconds, so the wait is the test execution and not the launch). Six agents then read every test of one area each, in two passes, and gave each test function a verdict: REMOVE, SHRINK to N cases (a parametrization cut down to the cases named next to the test), or KEEP. A SHRINK saves the time of its removed cases in proportion to their number. Every test not listed below is kept.

## Summary

To see at a glance what each area gives up, the table counts the cases removed and the measured time saved.

| Area | Cases | Cases Removed | Run Time (seconds) | Time Saved (seconds) |
|---|---|---|---|---|
| Trainers | 91 | 55 (60%) | 534 | 224 (42%) |
| Configuration, tracking, core | 330 | 235 (71%) | 25 | 19 (75%) |
| Data | 547 | 377 (69%) | 12 | 6 (48%) |
| Evaluation and losses | 202 | 138 (68%) | 4 | 3 (89%) |
| Models | 353 | 250 (71%) | 34 | 25 (74%) |
| Deployment, scripts, utilities | 150 | 113 (75%) | 43 | 33 (79%) |
| **Total** | **1673** | **1168 (70%)** | **651** | **310 (48%)** |

- **1168 of 1673 cases (70%)** are proposed, saving **310 of 651 seconds (48%)**.
- **The trainers take 534 of the 651 seconds (82%)**: the time saved comes from them, the other areas holding many cheap, low stake cases.
- **One test alone takes 200 seconds**: `tests/trainers/test_sequence_labelling_trainer.py` end-to-end training (about 31% of the suite), kept because it is the only end-to-end test of its trainer, while the other sequence trainers take about 4 seconds each. Shortening it (fewer iterations or a smaller dataset) would save more time than any removal below.
- **What stays**: one end-to-end training per task, the numerical tests of the losses, metrics, anchors, boxes, masks and keypoints, one exported-equals-trained check per export format, the label correctness tests of every dataset, the tests guarding past bugs, and the two tests added on 2026-11-04 for the hard instance segmentation dataset.

## Slow Tests

If you only remove a few tests, take these: each saves at least one second, 61 cases saving 289 seconds in all.

| Test | Verdict | Cases | Time (seconds) | Saved (seconds) | Reason |
|---|---|---|---|---|---|
| `tests/trainers/test_classification_trainer.py::test_explorer_thresholds_are_saved_for_the_export` | REMOVE | 1 | 64.2 | 64.2 | Covered by evaluation/classification/test_temperature_explorer.py::test_explorer_page_describes_every_temperature and ::test_saved_thresholds_are_written_next_to_the_page. |
| `tests/trainers/test_self_supervised_trainer.py::test_training_on_grouped_batches` | REMOVE | 1 | 59.5 | 59.5 | Covered by data/test_grouped_batch_sampler.py::test_self_supervised_loaders_give_grouped_batches; iterations per epoch by test_iterations_per_epoch_leave_out_the_last_incomplete_batch. |
| `tests/deployment/test_export_specifications.py::test_exports_match_the_eager_model` (keep [sequence_comparison-gru-cross_attention-average-False]) | SHRINK to 1 | 14 | 20.7 | 19.2 | Kept case checks TorchScript and ONNX against the eager model (multi-input, recurrent); the classifier exports are also checked against eager by test_classifier_exports_divide_the_logits_by_the_logit_temperature. |
| `tests/trainers/test_self_supervised_trainer.py::test_self_supervised_training_saves_evaluates_and_resumes` ([keep barlow_twins-_BT-OD0.005]) | SHRINK to 1 | 2 | 36.5 | 18.3 | Pipeline is method independent; SimCLR loss covered by losses/test_self_supervised_losses.py. |
| `tests/tracking/test_training_tracking.py::test_classification_training_and_evaluation_are_tracked` | REMOVE | 1 | 15.2 | 15.2 | Generates a dataset, trains, resumes and evaluates for real. Already covered by test_training_loop_feeds_the_tracker (trainer hooks) and test_mlflow_tracker.py::test_run_is_written_to_the_local_database / test_resumed_training_continues_its_run (database side). |
| `tests/models/instance_segmentation/test_instance_segmenter.py::test_exports_match_the_eager_inference_instance_segmenter` | REMOVE | 2 | 10.0 | 10.0 | Slowest test of the area; the export model is still checked by test_inference_instance_segmenter_matches_the_eager_model and detection/test_inference_detector.py::test_torchscript_round_trip. |
| `tests/trainers/test_classification_trainer.py::test_flattened_feature_map_training` | REMOVE | 1 | 7.2 | 7.2 | Covered by models/components/test_fully_connected_block.py, models/test_model_transfer.py::test_flattened_head_initialises_a_flattened_run and configuration_processing/test_run_name.py (HFLAT). |
| `tests/trainers/test_optimization.py::test_mixed_precision_training` | REMOVE | 1 | 7.2 | 7.2 | Mixed precision is always on, so every classification training (test_classification_training_saves_evaluates_and_resumes) already runs in bfloat16. |
| `tests/trainers/test_classification_trainer.py::test_intermediate_weight_saving_can_be_turned_off` | REMOVE | 1 | 7.1 | 7.1 | Trivial configuration switch; checkpoint schedule covered by test_base_trainer.py::test_training_saves_evaluates_and_resumes. |
| `tests/trainers/test_classification_trainer.py::test_decorrelation_turned_on_when_a_run_resumes` | REMOVE | 1 | 7.1 | 7.1 | Loss covered by losses/test_classification_losses.py, tracking by test_metric_tracking.py, changed configuration on resume by test_base_trainer.py::test_resumed_run_reports_and_keeps_its_changed_configuration. |
| `tests/trainers/test_backbone_transfer_training.py::test_classification_backbone_and_segmentation_decoder_initialise_segmentation_runs` | REMOVE | 1 | 5.1 | 5.1 | Covered by models/test_model_transfer.py::test_classification_backbone_initialises_a_segmenter_and_back and ::test_decoder_initialises_a_segmenter_of_other_classes; frozen training covered by test_trained_backbone_initialises_a_frozen_run. |
| `tests/trainers/test_segmentation_trainer.py::test_tiled_segmentation_training_evaluates_at_the_original_resolution` | REMOVE | 1 | 5.0 | 5.0 | Tiled prediction covered by models/segmentation/test_sliding_window.py::test_tiled_segmentation_equals_the_whole_image_prediction; original resolution by data/segmentation/test_segmentation_dataset.py. |
| `tests/trainers/test_classification_trainer.py::test_multi_label_classification_training` | REMOVE | 1 | 4.9 | 4.9 | Multi-label metrics, loss and dataset covered by evaluation/classification/test_classification_metrics.py::test_multi_label_metrics, losses/test_classification_losses.py and data/classification/test_classification_dataset.py. |
| `tests/deployment/test_export_specifications.py::test_sequence_normalization_exports_match_the_eager_model` (keep [sequence_comparison-pair]) | SHRINK to 1 | 3 | 6.2 | 4.2 | The pair scope runs the same per-sequence statistics in the exported graph. |
| `tests/trainers/test_backbone_transfer_training.py::test_detection_run_initialises_a_frozen_instance_segmentation_run` | REMOVE | 1 | 3.9 | 3.9 | Covered by models/test_model_transfer.py::test_detection_heads_initialise_an_instance_segmenter and ::test_frozen_backbone_keeps_its_weights_and_statistics. |
| `tests/trainers/test_backbone_transfer_training.py::test_sequence_components_initialise_a_partly_frozen_run` | REMOVE | 1 | 3.7 | 3.7 | Component transfer covered by models/test_model_transfer.py::test_temporal_module_and_head_initialise_another_sequence_classifier and ::test_normalization_parameters_stay_fixed; frozen training covered by test_trained_backbone_initialises_a_frozen_run. |
| `tests/trainers/test_backbone_transfer_training.py::test_averaged_run_of_9_classes_initialises_a_run_of_35_classes` | REMOVE | 1 | 3.5 | 3.5 | Covered by models/test_model_transfer.py::test_averaged_weights_initialise_a_run_keeping_a_moving_average and ::test_classification_head_adds_classes_after_the_source_classes. |
| `tests/trainers/test_backbone_transfer_training.py::test_self_supervised_backbone_initialises_a_classification_run` | REMOVE | 1 | 3.3 | 3.3 | Backbone transfer through the trainers covered by test_trained_backbone_initialises_a_frozen_run, and backbone loading by models/test_model_transfer.py::test_classification_backbone_initialises_a_detector. |
| `tests/trainers/test_segmentation_trainer.py::test_dice_only_binary_segmentation_training` | REMOVE | 1 | 2.9 | 2.9 | Dice loss covered by losses tests; per-class report off and missing optional files are option switches; main segmentation test covers the pipeline. |
| `tests/models/test_model_transfer.py::test_detection_heads_of_other_architectures_are_rejected` | REMOVE | 6 | 2.9 | 2.9 | Rejection message; the architecture comparison is kept in test_different_architectures_are_rejected. |
| `tests/trainers/test_self_supervised_trainer.py::test_averaged_weights_are_evaluated_on_the_same_views` | REMOVE | 1 | 2.8 | 2.8 | EMA evaluation covered by test_base_trainer.py::test_moving_average_is_saved_evaluated_and_resumed, fixed views by test_sample_views_are_fixed_and_leave_the_training_random_state. |
| `tests/models/detection/test_inference_detector.py::test_detector_with_fewer_anchors_than_proposals` | REMOVE | 1 | 2.8 | 2.8 | A single top-k edge case on a whole detector; costs 2.8 seconds. |
| `tests/trainers/test_classification_trainer.py::test_predictions_file_per_class_report_and_sample_images_can_be_turned_off` | REMOVE | 1 | 2.7 | 2.7 | Option switches of the report (save_predictions, per_class_report, number_sample_images), whose report side is covered by evaluation/test_evaluation_report.py and evaluation/test_checkpoint_evaluation.py. |
| `tests/trainers/test_self_supervised_trainer.py::test_run_name_suffix_starts_a_run_of_its_own` | REMOVE | 1 | 2.5 | 2.5 | Suffix in the run name covered by configuration_processing/test_run_name.py; the folder follows the name. |
| `tests/trainers/test_self_supervised_trainer.py::test_dataset_mixing_labelled_and_unlabelled_images_is_trained_and_evaluated` | REMOVE | 1 | 2.5 | 2.5 | Edge case of the dataset layout; labelled counts covered by data/test_data_record.py and k-NN evaluation by the kept test_self_supervised_training_saves_evaluates_and_resumes. |
| `tests/trainers/test_self_supervised_trainer.py::test_dataset_without_labels_is_evaluated_on_its_representations` | REMOVE | 1 | 2.4 | 2.4 | Unlabelled images trained and reported covered by test_dataset_mixing_labelled_and_unlabelled_images_is_trained_and_evaluated. |
| `tests/scripts/evaluation/test_evaluate_checkpoint.py::test_target_precisions_are_skipped_without_class_thresholds` | REMOVE | 3 | 2.4 | 2.4 | Warning branch of a CLI override, low stake (the keypoint case alone took 2.4 s). |
| `tests/trainers/test_keypoint_trainer.py::test_ignored_regions_go_from_the_loader_to_the_heatmap_loss` | REMOVE | 1 | 2.4 | 2.4 | Each step covered: data/detection/test_coco_dataset.py (regions), data/test_augmentations.py (flip), models/keypoints/test_keypoint_detector.py (loss with regions), data/keypoints/test_heatmap_targets.py. |
| `tests/scripts/evaluation/test_evaluate_checkpoint.py::test_error_image_limit_is_skipped_without_error_images` | REMOVE | 3 | 2.4 | 2.4 | Warning branch of a CLI override, low stake (the keypoint case alone took 2.4 s). |
| `tests/scripts/embedding/test_cluster_image_embeddings.py::test_layer_requests_are_checked_against_the_run` | REMOVE | 1 | 2.0 | 2.0 | Input validation of an analysis script. Most of its time is probably the module fixture, which will move to the next test using embedding_run. |
| `tests/models/classification/test_image_classifier.py::test_minimising_the_decorrelation_loss_decorrelates_the_hidden_features` | REMOVE | 1 | 1.9 | 1.9 | Optimisation loop; the loss value is checked by test_decorrelation_loss_of_the_hidden_features. |
| `tests/trainers/test_self_supervised_trainer.py::test_sample_views_are_fixed_and_leave_the_training_random_state` | REMOVE | 1 | 1.9 | 1.9 | Low stake: only the sample images and the random draw order are affected, not the correctness of the training. |
| `tests/models/instance_segmentation/test_instance_segmenter.py::test_instance_segmenter_trains_and_predicts_masks` ((keep [mask_head_settings1])) | SHRINK to 1 | 2 | 2.8 | 1.4 | The non-default mask head settings exercise the same loss and prediction path. |
| `tests/trainers/test_detection_trainer.py::test_anchor_coverage_mode_prints_every_split_and_exits` | REMOVE | 1 | 1.3 | 1.3 | Anchor recall covered by evaluation/detection/test_region_recall.py::test_anchor_recall_rows_and_table and printed at launch in test_detection_trainer.py::test_training_saves_evaluates_and_resumes. |
| `tests/trainers/test_self_supervised_trainer.py::test_iterations_per_epoch_leave_out_the_last_incomplete_batch` | REMOVE | 1 | 1.3 | 1.3 | Low stake: only the printed and logged iterations per epoch, not the training. |
| `tests/data/classification/test_imagenet_download.py::test_extract_imagenet_shards` | REMOVE | 1 | 1.1 | 1.1 | Slowest test of the area; per-shard extraction kept by test_extract_shard. |
| `tests/scripts/evaluation/test_serve_temperature_explorer.py::test_invalid_saves_are_refused_without_writing` (keep [/save_temperature_thresholds-content0-400-class names]) | SHRINK to 1 | 3 | 1.5 | 1.0 | The temperature check is the same check_saved_thresholds refusal path. |

## Low Stake Tests

So that the suite also keeps fewer tests to maintain, the cheap tests below (under one second saved each) are proposed for their low protection, area by area. Together they hold 1107 cases but save only 21 seconds.

### Trainers

| Test | Verdict | Cases | Time (seconds) | Saved (seconds) | Reason |
|---|---|---|---|---|---|
| `tests/trainers/test_input_viewer_panels.py::test_self_supervised_batch_mosaic_describes_the_grouped_batches` | REMOVE | 1 | 0.9 | 0.9 | Viewer captions; grouped batches covered by data/test_grouped_batch_sampler.py. |
| `tests/trainers/test_optimization.py::test_optimizer_schedule_and_clipping_in_training` ([keep adamw-AdamW]) | SHRINK to 1 | 3 | 0.3 | 0.2 | Schedule and clipping are optimizer independent; the name to class mapping is trivial. |
| `tests/trainers/test_base_trainer.py::test_moving_average_turned_on_when_a_run_resumes` | REMOVE | 1 | 0.0 | 0.0 | Niche branch; EMA save, evaluation and resume covered by test_moving_average_is_saved_evaluated_and_resumed. |
| `tests/trainers/test_base_trainer.py::test_resumed_run_reports_and_keeps_its_changed_configuration` | REMOVE | 1 | 0.0 | 0.0 | The configuration comparison is covered by configuration_processing/test_configuration_changes.py; only the archive file is left uncovered. |
| `tests/trainers/test_checkpointing.py::test_registry_lists_every_checkpoint_once` | REMOVE | 1 | 0.0 | 0.0 | Covered by test_registry_names_the_last_checkpoint_and_keeps_the_history. |
| `tests/trainers/test_base_trainer.py::test_evaluation_draws_the_fixed_sample_images` | REMOVE | 1 | 0.0 | 0.0 | Sample images are drawn and checked by every task's saves_evaluates_and_resumes test. |
| `tests/trainers/test_checkpointing.py::test_saved_checkpoint_is_found_and_loaded` | REMOVE | 1 | 0.0 | 0.0 | Covered by test_base_trainer.py::test_training_saves_evaluates_and_resumes, which resumes with exactly the saved weights. |
| `tests/trainers/test_base_trainer.py::test_model_iteration_sums_the_losses_and_tracks_the_metrics` | REMOVE | 1 | 0.0 | 0.0 | The loss sum is covered by test_instance_segmentation_trainer.py::test_loss_weights_weight_the_total_loss and every training test. |
| `tests/trainers/test_input_viewer_panels.py::test_keypoint_panels_count_the_keypoints_of_the_flipped_image` | REMOVE | 1 | 0.0 | 0.0 | Keypoint flip covered by data/test_augmentations.py; the rest only checks the debug viewer's wording. |
| `tests/trainers/test_base_trainer.py::test_input_viewer_creates_no_run_folder` | REMOVE | 1 | 0.0 | 0.0 | Low stake: the debug viewer only. |
| `tests/trainers/test_input_viewer_panels.py::test_pair_panels_give_the_real_frames_and_the_shuffled_order` | REMOVE | 1 | 0.0 | 0.0 | Shuffle covered by test_sequence_augmentation.py::test_pair_members_are_shuffled_within_their_blocks. |
| `tests/trainers/test_metric_tracking.py::test_interval_summary_header` | REMOVE (was SHRINK to 1) | 3 | 0.0 | 0.0 | Only checks the console header wording. |
| `tests/trainers/test_input_viewer_panels.py::test_detection_panels_count_the_objects_and_name_the_removed_ones` | REMOVE | 1 | 0.0 | 0.0 | Wording of a debug viewer panel; shift augmentation covered by data/test_augmentations.py. |
| `tests/trainers/test_weight_averaging.py::test_state_round_trip_and_reset` | REMOVE | 1 | 0.0 | 0.0 | EMA state save and restore covered by test_base_trainer.py::test_moving_average_is_saved_evaluated_and_resumed. |
| `tests/trainers/test_input_viewer_panels.py::test_self_supervised_panels_describe_the_augmentations_of_each_view` | REMOVE | 1 | 0.0 | 0.0 | Wording of a debug viewer panel; view augmentations covered by data/test_self_supervised_views.py. |
| `tests/trainers/test_checkpointing.py::test_missing_checkpoints_are_reported` | REMOVE | 1 | 0.0 | 0.0 | Only checks the wording of an error message. |
| `tests/trainers/test_optimization.py::test_learning_rate_factor` ([keep 1-constant-0.25, 9-cosine-0.55]) | SHRINK to 2 (was 3) | 8 | 0.0 | 0.0 | The warm-up point and the cosine midpoint are enough; test_optimizer_schedule_and_clipping_in_training checks the schedule step by step. |
| `tests/trainers/test_metric_tracking.py::test_time_interval_format` | REMOVE | 1 | 0.0 | 0.0 | Trivial string formatting. |
| `tests/trainers/test_metric_tracking.py::test_summary_header_bars_keep_their_place` | REMOVE (was SHRINK to 1) | 2 | 0.0 | 0.0 | Only checks the console layout. |
| `tests/trainers/test_metric_tracking.py::test_correlations_improve_when_they_decrease` | REMOVE (was SHRINK to 1) | 2 | 0.0 | 0.0 | Only the best marker of the console; no effect on training or saving. |
| `tests/trainers/test_metric_tracking.py::test_metric_changes_and_best_marker` | REMOVE | 1 | 0.0 | 0.0 | Console marker wording; improvement direction covered by test_correlations_improve_when_they_decrease. |

### Configuration, Tracking, Core

| Test | Verdict | Cases | Time (seconds) | Saved (seconds) | Reason |
|---|---|---|---|---|---|
| `tests/tracking/test_mlflow_tracker.py::test_full_queue_drops_events` | REMOVE | 1 | 0.5 | 0.5 | Low-stake overflow path. Non-blocking behaviour stays covered by test_hanging_backend_close_is_bounded and test_unreachable_server_never_blocks_the_training. |
| `tests/configuration_processing/test_configuration_checker.py::test_inconsistent_configuration_is_rejected` | SHRINK to 6 (keep set_image_size, set_overlapping_thresholds, set_invalid_detection_level_settings, set_multiple_errors, set_invalid_loss_weights, activate_image_only_augmentation_types) | 25 | 0.5 | 0.4 | Drops remove_level_anchors, set_single_input_channel, set_inception_channels, set_unknown_task, add_mask_loss_weight and zero_every_loss_weight. The kept detection-level and loss weight cases cover those families. |
| `tests/configuration_processing/test_configuration_checker.py::test_inconsistent_classification_configuration_is_rejected` | SHRINK to 5 (keep use_binary_labels, set_invalid_head, list_invalid_target_precisions, set_invalid_decorrelation, normalize_the_head_of_single_image_batches) | 15 | 0.2 | 0.1 | Drops set_invalid_classification_evaluation, repeat_a_target_precision and write_per_class_report_as_text: simple type checks or a family already covered. |
| `tests/configuration_processing/test_configuration_checker.py::test_image_size_must_keep_every_pooled_feature_map_even` | SHRINK to 1 (keep [224-True-expected_messages6]) | 8 | 0.2 | 0.1 | The multiple-of rule is also checked by set_image_size. Keeps the last max pooling variant. |
| `tests/configuration_processing/test_configuration_checker.py::test_inconsistent_segmentation_configuration_is_rejected` | SHRINK to 3 (keep remove_decoder_level, set_invalid_decoder, remove_both_losses) | 12 | 0.2 | 0.1 | Drops use_sequence_normalization, use_last_max_pooling_without_its_decoder_level and set_invalid_tiling: the family is covered by three cases. |
| `tests/configuration_processing/test_configuration_checker.py::test_run_name_suffix_is_checked` | SHRINK to 2 (keep [None-True], [runs/v1-False]) | 9 | 0.2 | 0.1 | One accepted and one refused value of a single regular expression. |
| `tests/configuration_processing/test_configuration_checker.py::test_inconsistent_instance_segmentation_configuration_is_rejected` | SHRINK to 2 (keep remove_level_mask_head, set_invalid_mask_head) | 7 | 0.1 | 0.1 | Drops the pool size and mask loss weight cases. |
| `tests/configuration_processing/test_run_name.py::test_changed_settings_change_run_name` | SHRINK to 2 (keep enable_inception, change_detection_levels) | 12 | 0.1 | 0.1 | String tokens of one builder. |
| `tests/configuration_processing/test_configuration_checker.py::test_inconsistent_keypoint_configuration_is_rejected` | SHRINK to 2 (keep set_image_size_not_matching_the_branches, set_unknown_keypoint_sigmas) | 8 | 0.1 | 0.1 | Drops the head and high resolution network cases. |
| `tests/configuration_processing/test_configuration_checker.py::test_jigsaw_image_size_is_checked` | SHRINK to 1 (keep [3-256-True-True-False]) | 5 | 0.1 | 0.1 | Keeps the refusal case. |
| `tests/configuration_processing/test_configuration_checker.py::test_inconsistent_pretrained_backbone_is_rejected` | SHRINK to 2 (keep freeze_random_backbone, set_invalid_pretrained_run) | 6 | 0.1 | 0.1 | The frozen-normalization variant is part of the same rule family. |
| `tests/configuration_processing/test_configuration_checker.py::test_inconsistent_self_supervised_configuration_is_rejected` | SHRINK to 2 (keep set_invalid_projection_head, set_invalid_grouped_batches) | 7 | 0.1 | 0.1 | Drops the view augmentation and evaluation cases. |
| `tests/configuration_processing/test_run_name.py::test_segmentation_run_names` | SHRINK to 2 (keep None, train_on_half_scale_tiles) | 10 | 0.1 | 0.1 | String tokens of one builder. |
| `tests/configuration_processing/test_run_name.py::test_keypoint_run_names` | SHRINK to 1 (keep None) | 9 | 0.1 | 0.1 | String tokens of one builder. |
| `tests/configuration_processing/test_configuration_checker.py::test_flattened_feature_map_is_accepted_for_the_sequence_tasks` | REMOVE | 4 | 0.1 | 0.1 | Acceptance side. test_flattened_feature_map_is_accepted_for_classification is kept. |
| `tests/configuration_processing/test_configuration_checker.py::test_several_target_precisions_are_accepted` | REMOVE | 4 | 0.1 | 0.1 | Acceptance side. The target precision refusal is kept. |
| `tests/configuration_processing/test_configuration_checker.py::test_intermediate_weight_saving_iterations_is_checked` | SHRINK to 1 (keep [5000-...]) | 4 | 0.1 | 0.1 | Keeps the refusal. Accepted values load in every other test. |
| `tests/configuration_processing/test_configuration_checker.py::test_detection_label_mapping_indexes_its_classes_from_the_background` | SHRINK to 1 (keep [label_mapping1-...]) | 4 | 0.1 | 0.1 | Keeps the refusal. |
| `tests/configuration_processing/test_run_name.py::test_sequence_run_names` | SHRINK to 1 (keep None) | 8 | 0.1 | 0.1 | String tokens of one builder. |
| `tests/configuration_processing/test_run_name.py::test_classification_run_names` | SHRINK to 2 (keep None, use_multi_label_outputs) | 9 | 0.1 | 0.0 | String tokens of one builder. |
| `tests/configuration_processing/test_configuration_checker.py::test_inconsistent_frame_selection_configuration_is_rejected` | SHRINK to 1 (keep set_invalid_selection) | 4 | 0.1 | 0.0 | The kept case bundles the selection settings. |
| `tests/configuration_processing/test_configuration_checker.py::test_removed_settings_are_refused_alone` | SHRINK to 1 (keep keep_the_fixed_settings) | 4 | 0.1 | 0.0 | Same refusal mechanism for every removed setting. |
| `tests/configuration_processing/test_configuration_checker.py::test_sequence_tasks_keep_their_settings` | SHRINK to 1 (keep [sequence_classification]) | 4 | 0.1 | 0.0 | One shared code path for the four sequence tasks. |
| `tests/configuration_processing/test_configuration_checker.py::test_inconsistent_sequence_classification_configuration_is_rejected` | SHRINK to 1 (keep set_invalid_transformer_and_settings) | 4 | 0.1 | 0.0 | The kept case bundles several temporal setting errors. |
| `tests/configuration_processing/test_configuration_checker.py::test_sequence_tasks_require_the_maximum_sequence_length` | SHRINK to 1 (keep [sequence_classification-...]) | 4 | 0.1 | 0.0 | One shared required-key check. |
| `tests/configuration_processing/test_configuration_checker.py::test_inconsistent_sequence_comparison_configuration_is_rejected` | SHRINK to 1 (keep set_unknown_fusion) | 4 | 0.1 | 0.0 | The main task-specific refusal. |
| `tests/configuration_processing/test_configuration_checker.py::test_segmentation_label_mapping_gives_mask_values` | SHRINK to 1 (keep [label_mapping0-...]) | 4 | 0.1 | 0.0 | Keeps the gap refusal. |
| `tests/configuration_processing/test_configuration_checker.py::test_image_only_augmentation_types_are_accepted_without_targets` | REMOVE | 3 | 0.0 | 0.0 | Acceptance side. The refusal activate_image_only_augmentation_types stays in test_inconsistent_configuration_is_rejected. |
| `tests/configuration_processing/test_run_name.py::test_pretrained_temporal_module_and_head_run_names` | SHRINK to 1 (keep [TemporalModel-True-1.0-...TPT-TFRZ_]) | 6 | 0.0 | 0.0 | String tokens of one builder. |
| `tests/configuration_processing/test_run_name.py::test_dropout_rates_keep_run_name` | REMOVE | 5 | 0.0 | 0.0 | Invariance of a token that never existed. test_reference_run_name already checks name stability. |
| `tests/configuration_processing/test_run_name.py::test_pretrained_backbone_run_names` | SHRINK to 1 (keep [True-1.0-False--PT-FRZ_]) | 6 | 0.0 | 0.0 | String tokens of one builder. |
| `tests/configuration_processing/test_run_name.py::test_sequence_augmentation_keeps_run_name` | REMOVE | 4 | 0.0 | 0.0 | Same reason as test_dropout_rates_keep_run_name. |
| `tests/configuration_processing/test_run_name.py::test_pretrained_segmentation_decoder_run_names` | SHRINK to 1 (keep [True-True-1.0--DPT-DCL-DFRZ_]) | 5 | 0.0 | 0.0 | The kept case holds every token. |
| `tests/configuration_processing/test_configuration_checker.py::test_inconsistent_sequence_labelling_configuration_is_rejected` | SHRINK to 1 (keep use_pooling_for_labelling) | 3 | 0.0 | 0.0 | The main task-specific refusal. |
| `tests/configuration_processing/test_configuration_checker.py::test_sequence_normalization_is_accepted_for_the_sequence_tasks` | REMOVE | 1 | 0.0 | 0.0 | Acceptance side. The refusal use_sequence_normalization stays in the segmentation list. |
| `tests/configuration_processing/test_configuration_checker.py::test_inconsistent_detection_levels_are_rejected` | SHRINK to 2 (keep empty_detection_levels, reverse_detection_levels) | 5 | 0.0 | 0.0 | The rule family is covered by two cases. |
| `tests/configuration_processing/test_run_name.py::test_frame_selection_run_names` | SHRINK to 1 (keep [1-1.0-hungarian-_FPC1-SLW1]) | 4 | 0.0 | 0.0 | String tokens of one builder. |
| `tests/configuration_processing/test_run_name.py::test_sequence_comparison_run_names` | SHRINK to 1 (keep use_cross_attention) | 4 | 0.0 | 0.0 | String tokens of one builder. |
| `tests/configuration_processing/test_configuration_checker.py::test_instance_segmentation_configurations_are_consistent` | SHRINK to 1 (keep [configuration_example.yaml]) | 2 | 0.0 | 0.0 | Same reason as the other consistency pairs. |
| `tests/configuration_processing/test_run_name.py::test_self_supervised_run_names` | SHRINK to 1 (keep head_settings0) | 4 | 0.0 | 0.0 | String tokens of one builder. |
| `tests/configuration_processing/test_configuration_checker.py::test_segmentation_configurations_are_consistent` | SHRINK to 1 (keep [configuration_example.yaml]) | 2 | 0.0 | 0.0 | Same reason as the other consistency pairs. |
| `tests/configuration_processing/test_run_name.py::test_instance_segmentation_run_names` | SHRINK to 1 (keep change_mask_heads) | 3 | 0.0 | 0.0 | String tokens of one builder. |
| `tests/configuration_processing/test_run_name.py::test_pretrained_detection_head_run_names` | SHRINK to 1 (keep [True-True-...RPT-RFRZ...DPT-DFRZ...]) | 3 | 0.0 | 0.0 | The kept case holds every token. |
| `tests/configuration_processing/test_configuration_checker.py::test_configuration_without_run_name_suffix_is_refused` | REMOVE | 1 | 0.0 | 0.0 | The missing-key path is covered by test_missing_configuration_key_is_reported_with_its_path. |
| `tests/configuration_processing/test_configuration_checker.py::test_classification_configurations_are_consistent` | SHRINK to 1 (keep [configuration_example.yaml]) | 2 | 0.0 | 0.0 | The synthetic configuration is also loaded by every classification synthetic smoke training in tests/trainers. |
| `tests/tracking/test_training_tracking.py::test_run_status_follows_the_end_of_the_training` | SHRINK to 1 (keep [save_model-RuntimeError-FAILED-RuntimeError]) | 4 | 0.0 | 0.0 | One case still checks that the run is closed with a status and that the error is raised again. |
| `tests/configuration_processing/test_configuration_checker.py::test_transferred_mask_heads_and_class_layers_are_accepted` | SHRINK to 1 (keep [instance_segmentation]) | 2 | 0.0 | 0.0 | Same transfer acceptance path for both tasks. |
| `tests/configuration_processing/test_configuration_checker.py::test_frame_selection_configurations_are_consistent` | SHRINK to 1 (keep [configuration_example.yaml]) | 2 | 0.0 | 0.0 | Same reason as the other consistency pairs. |
| `tests/configuration_processing/test_run_name.py::test_sequence_normalization_run_name_tokens` | SHRINK to 1 (keep [sequence-member-...]) | 3 | 0.0 | 0.0 | String tokens of one builder. |
| `tests/configuration_processing/test_configuration_checker.py::test_sequence_classification_configurations_are_consistent` | SHRINK to 1 (keep [configuration_example.yaml]) | 2 | 0.0 | 0.0 | The synthetic file is loaded by the sequence smoke trainings. |
| `tests/configuration_processing/test_configuration_checker.py::test_sequence_labelling_configurations_are_consistent` | SHRINK to 1 (keep [configuration_example.yaml]) | 2 | 0.0 | 0.0 | Same reason as the other consistency pairs. |
| `tests/configuration_processing/test_configuration_checker.py::test_sequence_comparison_configurations_are_consistent` | SHRINK to 1 (keep [configuration_example.yaml]) | 2 | 0.0 | 0.0 | Same reason as the other consistency pairs. |
| `tests/configuration_processing/test_configuration_checker.py::test_transferred_sequence_components_are_accepted` | REMOVE | 1 | 0.0 | 0.0 | Acceptance side. The transfer refusal (test_inconsistent_temporal_and_head_transfers_are_rejected) is kept. |
| `tests/configuration_processing/test_configuration_checker.py::test_inconsistent_temporal_and_head_transfers_are_rejected` | SHRINK to 1 (keep set_invalid_temporal_and_head_transfers) | 2 | 0.0 | 0.0 | The kept case bundles the invalid transfer settings. |
| `tests/configuration_processing/test_configuration_checker.py::test_segmentation_last_max_pooling_and_binary_labels_are_accepted` | REMOVE | 1 | 0.0 | 0.0 | Acceptance side of a rule family covered by remove_decoder_level. |
| `tests/configuration_processing/test_configuration_checker.py::test_keypoint_geometric_augmentations_are_accepted` | REMOVE | 1 | 0.0 | 0.0 | Acceptance side. |
| `tests/configuration_processing/test_run_name.py::test_run_name_suffix_ends_the_run_name` | SHRINK to 1 (keep [v1-...]) | 3 | 0.0 | 0.0 | The kept case also proves the suffix is the last section. |
| `tests/configuration_processing/test_configuration_checker.py::test_self_supervised_configurations_are_consistent` | SHRINK to 1 (keep [configuration_example.yaml]) | 2 | 0.0 | 0.0 | Same reason as the other consistency pairs. |
| `tests/configuration_processing/test_configuration_checker.py::test_decorrelation_of_a_hidden_layer_is_accepted` | REMOVE | 1 | 0.0 | 0.0 | Acceptance side of a rule whose refusal (set_invalid_decorrelation) is kept. |
| `tests/configuration_processing/test_configuration_checker.py::test_views_without_crop_are_accepted` | REMOVE | 1 | 0.0 | 0.0 | Acceptance side. |
| `tests/tracking/test_mlflow_tracker.py::test_unavailable_backend_never_blocks_the_training` | REMOVE | 1 | 0.0 | 0.0 | Same "backend down never blocks" property as test_unreachable_server_never_blocks_the_training, which tests it on a real closed port. |
| `tests/configuration_processing/test_configuration_checker.py::test_example_configuration_is_consistent` | REMOVE | 1 | 0.0 | 0.0 | Loads the same example file as test_background_minimum_above_zero_only_warns, which also asserts it loads cleanly. |
| `tests/configuration_processing/test_run_name.py::test_pretrained_mask_head_run_names` | SHRINK to 1 (keep [True--MR28-MPT-MFRZ_]) | 2 | 0.0 | 0.0 | The kept case holds both tokens. |
| `tests/tracking/test_mlflow_tracker.py::test_tracker_does_nothing_outside_its_run` | REMOVE | 1 | 0.0 | 0.0 | Trivial no-op guard. The disabled and closed paths are covered by test_experiment_tracker.py::test_factory_gives_a_disabled_tracker_without_tracking. |
| `tests/tracking/test_experiment_tracker.py::test_disabled_tracker_does_nothing` | REMOVE | 1 | 0.0 | 0.0 | Tests a trivial null object. The factory test below already returns and uses DisabledTracker. |
| `tests/tracking/test_experiment_tracker.py::test_tracking_uri_is_the_local_database_unless_a_server_is_set` | REMOVE | 1 | 0.0 | 0.0 | A two-branch URI choice. The local database path is exercised by test_run_is_written_to_the_local_database. |
| `tests/test_main.py::test_absolute_configuration_path_is_returned` | REMOVE | 1 | 0.0 | 0.0 | Trivial identity branch. The relative-path test covers the resolution. |
| `tests/test_main.py::test_missing_file_or_argument_stops_the_launch` | SHRINK to 1 (keep [arguments0]) | 2 | 0.0 | 0.0 | Both cases go through the same stop path. |
| `tests/tracking/test_mlflow_tracker.py::test_error_is_described_by_its_deepest_cause` | REMOVE | 1 | 0.0 | 0.0 | Wording of a warning message. |
| `tests/core/test_task_registry.py::test_keypoint_detection_is_registered` | REMOVE | 1 | 0.0 | 0.0 | Per-task identity checks that test_every_task_is_registered_everywhere covers set-wise (trainer TASK_NAME match included). |
| `tests/core/test_task_registry.py::test_unknown_task_has_no_trainer` | REMOVE | 1 | 0.0 | 0.0 | Error wording of a KeyError. Unknown tasks are refused by the configuration checker first. |
| `tests/configuration_processing/test_configuration_changes.py::test_flatten_settings_gives_dotted_keys_with_exact_values` | REMOVE | 1 | 0.0 | 0.0 | The flattening is exercised by test_changed_added_and_removed_settings_are_listed, which compares the dotted keys. |
| `tests/configuration_processing/test_configuration_changes.py::test_identical_configurations_have_no_changed_setting` | REMOVE | 1 | 0.0 | 0.0 | A trivial empty-diff edge case, implied by the listing test. |
| `tests/configuration_processing/test_configuration_changes.py::test_setting_values_are_formatted_like_the_yaml_files` | REMOVE | 1 | 0.0 | 0.0 | Formatting of displayed values. |

### Data

| Test | Verdict | Cases | Time (seconds) | Saved (seconds) | Reason |
|---|---|---|---|---|---|
| `tests/data/synthetic/test_create_synthetic_dataset.py::test_textured_backgrounds_change_the_colour_by_the_texture_amplitude` | REMOVE | 1 | 0.9 | 0.9 | Slow, low stake: background texture amplitude, not labels. |
| `tests/data/synthetic/test_create_synthetic_dataset.py::test_hollow_shapes_are_closed_rings_inside_their_filled_shape` | SHRINK to 3: [False-MINIMUM_SHAPE_SIZE-square], [-triangle], [-capsule] | 90 | 0.7 | 0.6 | Shape drawing check per class. |
| `tests/data/synthetic/test_create_synthetic_dataset.py::test_every_shape_stays_inside_its_square` | SHRINK to 2: [parallelogram], [capsule] | 35 | 0.6 | 0.6 | Shape drawing check per class. |
| `tests/data/test_grouped_batch_sampler.py::test_self_supervised_loaders_give_grouped_batches` | REMOVE | 1 | 0.5 | 0.5 | Loader wiring; grouping kept by test_grouped_batches_are_full_and_share_their_values and test_dataset_keeps_the_groups_of_its_samples. |
| `tests/data/synthetic/test_create_synthetic_dataset.py::test_created_instance_segmentation_dataset_encodes_the_visible_parts` | REMOVE | 1 | 0.4 | 0.4 | Worker-pool generation; encoding of visible parts covered by test_created_hard_instance_segmentation_dataset_encodes_small_and_large_visible_parts and test_instance_map_gives_the_visible_part_of_every_object. |
| `tests/data/sequence/test_sequence_dataset.py::test_loaders_needing_batch_statistics_drop_the_last_training_batch` | REMOVE | 1 | 0.3 | 0.3 | Same rule as test_classifier_needing_batch_statistics_drops_its_last_training_batch. |
| `tests/data/synthetic/test_create_synthetic_dataset.py::test_shape_boxes_are_tight` | SHRINK to 1: [triangle] | 35 | 0.2 | 0.2 | Generic get_mask_bounding_box, one class is enough. |
| `tests/data/test_augmentations.py::test_device_computed_types_on_the_gpu_never_make_the_host_wait` | SHRINK to 1: [clahe_augment] | 5 | 0.3 | 0.2 | Near-identical sync checks. |
| `tests/data/test_augmentations.py::test_clahe_and_luminance_equalization_give_the_values_of_kornia` | SHRINK to 1: [cpu] | 2 | 0.3 | 0.2 | Same comparison on the second device; test_clahe_gives_the_values_of_kornia keeps cuda. |
| `tests/data/synthetic/test_create_synthetic_dataset.py::test_created_hard_detection_dataset_holds_small_large_and_cut_objects` | REMOVE | 1 | 0.1 | 0.1 | Worker-pool generation; hard path covered by test_created_hard_instance_segmentation_dataset_encodes_small_and_large_visible_parts and the hard_images unit tests. |
| `tests/data/test_device_image_operations.py::test_clahe_gives_the_values_of_kornia` | SHRINK to 1: [shape0-cuda] | 8 | 0.1 | 0.1 | Same comparison on the second device; cpu kept by the augmentation CLAHE test. |
| `tests/data/test_augmentations.py::test_plasma_map_is_the_diamond_square_map_of_kornia` | SHRINK to 1: [cpu] | 2 | 0.1 | 0.0 | Same comparison on the second device. |
| `tests/data/test_augmentations.py::test_a_single_histogram_equalization_is_applied_per_image` | REMOVE | 1 | 0.0 | 0.0 | Low stake photometric draw rule. |
| `tests/data/test_augmentations.py::test_black_out_cells_follow_the_rounded_edges_of_the_grid` | REMOVE | 1 | 0.0 | 0.0 | Covered by test_black_out_hides_whole_cells_of_the_configured_grid_drawn_one_by_one. |
| `tests/data/test_augmentations.py::test_view_types_on_the_gpu_never_make_the_host_wait` | SHRINK to 1: [random_perspective] | 8 | 0.0 | 0.0 | Near-identical sync checks; test_views_on_the_gpu_never_make_the_host_wait covers the view path. |
| `tests/data/test_augmentations.py::test_geometric_augmentation_moves_instance_masks_and_recomputes_boxes` | SHRINK to 2: [random_crop], [random_rotation] | 8 | 0.1 | 0.0 | Near-identical resampling cases. |
| `tests/data/test_augmentations.py::test_photometric_augmentation_keeps_boxes_and_range` | SHRINK to 1: [brightness_contrast_saturation] | 18 | 0.0 | 0.0 | Near-identical range checks. |
| `tests/data/test_device_image_operations.py::test_rotation_gives_the_values_of_torchvision` | SHRINK to 1: [shape0-cuda] | 6 | 0.0 | 0.0 | Same comparison on the second device. |
| `tests/data/synthetic/test_create_synthetic_dataset.py::test_keypoint_drawing_functions_give_one_point_per_keypoint_name` | SHRINK to 4: [False-circle], [False-capsule], [False-square], [False-annulus] (or the first 4 ids) | 74 | 0.0 | 0.0 | Per-category count check of a constant table; keypoint labels kept by test_created_keypoint_dataset_is_valid_coco_and_loads. |
| `tests/data/test_augmentations.py::test_geometric_augmentation_moves_the_ignored_regions_with_the_keypoints` | SHRINK to 2 cases: [random_rotation], [random_crop] | 8 | 0.0 | 0.0 | Ignored regions follow the box path tested per type by test_geometric_augmentation_moves_boxes_with_the_image. |
| `tests/data/test_augmentations.py::test_geometric_augmentation_moves_masks_with_the_image` | SHRINK to 2: [random_crop], [random_rotation] | 8 | 0.0 | 0.0 | Near-identical resampling cases. |
| `tests/data/test_augmentations.py::test_geometric_augmentation_moves_boxes_with_the_image` | SHRINK to 4: [random_left_right_flip], [random_crop], [random_zoom_out], [random_rotation] | 8 | 0.1 | 0.0 | Translation and zoom out also have dedicated tests; 90 rotation and up-down flip repeat the permutation path. |
| `tests/data/synthetic/test_create_synthetic_dataset.py::test_quadrilaterals_keep_their_defining_geometry` | REMOVE | 1 | 0.0 | 0.0 | Shape drawing geometry, low stake. |
| `tests/data/test_augmentations.py::test_geometric_augmentation_moves_keypoints_with_the_image` | SHRINK to 3: [random_left_right_flip], [random_crop], [random_rotation] | 8 | 0.0 | 0.0 | One exact permutation and two resampling types still guard the keypoint labels. |
| `tests/data/synthetic/test_create_synthetic_dataset.py::test_hollow_shapes_are_drawn_at_their_minimum_size` | SHRINK to 1: [False] | 2 | 0.0 | 0.0 | Near-identical cases. |
| `tests/data/sequence/test_sequence_pair_dataset.py::test_split_description_counts_pairs` | REMOVE | 1 | 0.0 | 0.0 | Counting kept by test_split_images_and_annotations_are_counted. |
| `tests/data/synthetic/test_create_synthetic_dataset.py::test_low_contrast_colours_stay_in_their_range` | REMOVE | 3 | 0.0 | 0.0 | Colour range of the hard images, not labels. |
| `tests/data/test_augmentations.py::test_channel_equalization_changes_the_drawn_channels_only` | REMOVE | 1 | 0.0 | 0.0 | Low stake photometric detail. |
| `tests/data/test_device_image_operations.py::test_mask_rotation_gives_the_values_of_torchvision` | SHRINK to 1: [cuda] | 2 | 0.0 | 0.0 | Same comparison on the second device. |
| `tests/data/synthetic/test_create_synthetic_dataset.py::test_pair_frames_fail_loudly_when_objects_do_not_fit` | REMOVE | 1 | 0.0 | 0.0 | Error guard, low stake. |
| `tests/data/synthetic/test_create_synthetic_dataset.py::test_shape_change_shapes_get_the_minimum_size` | SHRINK to 1: [512-...] | 2 | 0.0 | 0.0 | Near-identical size cases. |
| `tests/data/test_augmentations.py::test_image_only_types_refuse_the_targets` | REMOVE | 2 | 0.0 | 0.0 | Error guard, low stake. |
| `tests/data/classification/test_classification_dataset.py::test_unreadable_image_is_named` | REMOVE | 1 | 0.0 | 0.0 | Wording of an error message. |
| `tests/data/synthetic/test_create_synthetic_dataset.py::test_counted_shapes_get_the_sequence_labelling_size` | SHRINK to 1: [512-...] | 2 | 0.0 | 0.0 | Near-identical size cases. |
| `tests/data/synthetic/test_create_synthetic_dataset.py::test_sequence_frame_range_is_checked` | SHRINK to 1: [sequence_classification-1-4] | 3 | 0.0 | 0.0 | Error guard, low stake. |
| `tests/data/test_device_image_operations.py::test_clahe_of_a_grid_that_is_not_square_follows_its_transposition` | SHRINK to 1 case: the first grid | 3 | 0.0 | 0.0 | The three grids repeat one transposition check. |
| `tests/data/synthetic/test_create_synthetic_dataset.py::test_image_shapes_get_the_minimum_size` | SHRINK to 1: [512-...] | 3 | 0.0 | 0.0 | Near-identical size cases. |
| `tests/data/sequence/test_sequence_dataset.py::test_collate_pads_the_batches_to_their_longest_sequence` | SHRINK to 1 case | 2 | 0.0 | 0.0 | Both cases pad the same way; pairs padding kept by test_collate_pads_every_member_of_the_pairs. |
| `tests/data/sequence/test_sequence_dataset.py::test_selection_needs_a_mapped_unselected_category` | REMOVE | 2 | 0.0 | 0.0 | Error guard, low stake. |
| `tests/data/test_device_image_operations.py::test_colour_conversions_give_the_values_of_kornia` | SHRINK to 1: first case | 2 | 0.0 | 0.0 | Near-identical conversions. |
| `tests/data/test_data_record.py::test_classes_counted_in_every_manifest_format` | SHRINK to 1: [labels0-expected_counts0] | 9 | 0.0 | 0.0 | Near-identical label lists. |
| `tests/data/test_preprocessing.py::test_restored_boxes_invert_the_preprocessing` | SHRINK to 1: [True-640-480] | 6 | 0.0 | 0.0 | Near-identical size cases. |
| `tests/data/classification/test_foundation_model_image_downloader.py::test_read_category_dictionary` | REMOVE | 1 | 0.0 | 0.0 | Trivial file reading, used again by the build manifests and shard processing tests. |
| `tests/data/classification/test_imagenet_download.py::test_read_imagenet_class_names` | REMOVE | 1 | 0.0 | 0.0 | Trivial file reading; the names are used by test_extract_imagenet_shards. |
| `tests/data/test_augmentations.py::test_objects_translated_out_of_the_image_are_removed` | REMOVE | 1 | 0.0 | 0.0 | Covered by test_boxes_leaving_the_image_are_removed_with_their_labels and test_translation_shifts_the_image_boxes_and_masks_by_whole_pixels. |
| `tests/data/test_augmentations.py::test_report_lines_of_sequences_and_of_unaugmented_samples` | REMOVE | 1 | 0.0 | 0.0 | Wording of report lines; the report content is covered by test_reports_name_the_drawn_types_their_values_and_the_removed_objects. |
| `tests/data/test_self_supervised_views.py::test_views_without_any_type_are_the_images` | REMOVE | 1 | 0.0 | 0.0 | Trivial identity path; disabled augmentation identity covered by test_disabled_augmentation_is_identity. |
| `tests/data/test_split_description.py::test_split_images_and_annotations_are_counted` | SHRINK to 1 case | 2 | 0.0 | 0.0 | Both cases count the same way. |
| `tests/data/classification/test_imagenet_download.py::test_check_split_sizes` | REMOVE | 1 | 0.0 | 0.0 | Error guard of a download tool. |
| `tests/data/detection/test_coco_dataset.py::test_annotation_without_segmentation_is_refused` | REMOVE | 1 | 0.0 | 0.0 | Error guard, low stake. |
| `tests/data/keypoints/test_keypoint_categories.py::test_inconsistent_categories_are_refused` | REMOVE | 1 | 0.0 | 0.0 | Error guard, low stake. |
| `tests/data/segmentation/test_segmentation_dataset.py::test_invalid_masks_are_refused` | REMOVE | 1 | 0.0 | 0.0 | Error guard, low stake. |
| `tests/data/sequence/test_sequence_dataset.py::test_selection_sequences_that_do_not_fit_are_all_listed` | REMOVE | 1 | 0.0 | 0.0 | Error listing, low stake. |
| `tests/data/sequence/test_sequence_dataset.py::test_split_description_counts_kept_sequences` | REMOVE | 1 | 0.0 | 0.0 | Counting kept by test_split_images_and_annotations_are_counted. |
| `tests/data/synthetic/test_create_synthetic_dataset.py::test_annulus_is_a_closed_ring_around_a_hole` | REMOVE | 1 | 0.0 | 0.0 | Ring check kept by test_hollow_shapes_are_closed_rings_inside_their_filled_shape. |
| `tests/data/synthetic/test_create_synthetic_dataset.py::test_counted_shapes_image_refuses_a_shape_count_outside_its_classes` | REMOVE | 2 | 0.0 | 0.0 | Error guard, low stake. |
| `tests/data/synthetic/test_create_synthetic_dataset.py::test_hollow_shape_too_small_for_a_hole_raises` | REMOVE | 1 | 0.0 | 0.0 | Error guard, low stake. |
| `tests/data/test_augmentations.py::test_disabled_augmentation_is_identity` | REMOVE | 1 | 0.0 | 0.0 | Trivial identity path. |
| `tests/data/test_augmentations.py::test_images_without_boxes_for_classification` | REMOVE | 1 | 0.0 | 0.0 | Box-free path also run by the sequence and view tests. |
| `tests/data/test_augmentations.py::test_invalid_inputs_are_rejected` | REMOVE | 1 | 0.0 | 0.0 | Error guard, low stake. |
| `tests/data/test_augmentations.py::test_keypoints_need_their_boxes` | REMOVE | 1 | 0.0 | 0.0 | Error guard, low stake. |
| `tests/data/test_augmentations.py::test_reports_name_the_drawn_types_their_values_and_the_removed_objects` | REMOVE | 1 | 0.0 | 0.0 | Wording of the augmentation report. |
| `tests/data/test_data_record.py::test_split_record_identifies_its_labels_file` | REMOVE | 1 | 0.0 | 0.0 | Low stake record field. |
| `tests/data/test_label_mapping.py::test_detector_classes_keep_their_dictionary_names` | REMOVE | 1 | 0.0 | 0.0 | Naming only. |
| `tests/data/test_self_supervised_views.py::test_crop_of_the_views_falls_back_on_the_whole_image` | REMOVE | 1 | 0.0 | 0.0 | Rare fallback, low stake. |
| `tests/data/test_self_supervised_views.py::test_views_are_refused_to_a_supervised_augmenter_and_for_a_single_image` | REMOVE | 1 | 0.0 | 0.0 | Error guard, low stake. |
| `tests/data/test_augmentations.py::test_sharpening_kernels_are_those_of_the_sharpening` | SHRINK to 1 case: [torch.float32] | 2 | 0.0 | 0.0 | bfloat16 repeats the same kernel comparison. |
| `tests/data/test_preprocessing.py::test_class_map_is_restored_to_the_original_image` | SHRINK to 1: first case | 2 | 0.0 | 0.0 | Near-identical cases. |
| `tests/data/test_augmentations.py::test_fragments_of_objects_are_removed_with_their_labels` | REMOVE | 1 | 0.0 | 0.0 | Covered end to end by test_crop_removes_the_fragments_of_boxes_and_instance_masks. |
| `tests/data/test_augmentations.py::test_image_tasks_keep_the_sequence_settings_off` | REMOVE | 1 | 0.0 | 0.0 | Trivial configuration flag check. |
| `tests/data/classification/test_foundation_model_image_downloader.py::test_image_identifiers` | REMOVE | 1 | 0.0 | 0.0 | Trivial string building. |
| `tests/data/classification/test_foundation_model_image_downloader.py::test_parse_annotation_line` | REMOVE | 1 | 0.0 | 0.0 | Trivial parsing, run again by test_process_open_images_archive. |
| `tests/data/detection/test_coco_masks.py::test_run_length_encoding_must_cover_the_image` | REMOVE | 1 | 0.0 | 0.0 | Error guard; decoding kept by test_run_length_encodings_decode_column_by_column. |
| `tests/data/synthetic/test_create_synthetic_dataset.py::test_hollow_inner_keypoints_without_outer_keypoints_raise` | REMOVE | 1 | 0.0 | 0.0 | Error guard, low stake. |
| `tests/data/synthetic/test_create_synthetic_dataset.py::test_shape_change_sequence_needs_five_objects` | REMOVE | 1 | 0.0 | 0.0 | Error guard, low stake. |
| `tests/data/test_data_record.py::test_images_of_every_manifest_format` | SHRINK to 2: first two formats | 4 | 0.0 | 0.0 | Near-identical format cases. |
| `tests/data/test_data_record.py::test_unknown_manifest_format_fails_loudly` | REMOVE | 1 | 0.0 | 0.0 | Error guard, low stake. |
| `tests/data/test_preprocessing.py::test_polygons_are_transformed_like_the_boxes` | SHRINK to 1: first case | 2 | 0.0 | 0.0 | Near-identical cases. |
| `tests/data/test_self_supervised_views.py::test_strength_sets_hold_every_strength_and_the_augmenter_takes_the_set_of_its_task` | REMOVE | 1 | 0.0 | 0.0 | Constant tables. |

### Evaluation And Losses

| Test | Verdict | Cases | Time (seconds) | Saved (seconds) | Reason |
|---|---|---|---|---|---|
| `tests/evaluation/classification/test_score_curves.py::test_curve_images_of_a_few_and_of_many_classes` | REMOVE | 1 | 0.8 | 0.8 | Renders several matplotlib images to check layout. |
| `tests/evaluation/classification/test_score_curves.py::test_thresholds_image_zooms_on_the_marked_thresholds` | REMOVE | 1 | 0.7 | 0.7 | Renders images to check axis zoom; cosmetic. |
| `tests/evaluation/classification/test_classification_metrics.py::test_confusion_matrix_image_grows_with_the_classes_and_stays_headless` | REMOVE | 1 | 0.6 | 0.6 | Draws several matrix images to check figure sizes; cosmetic (the headless guard is lost, flag it). |
| `tests/evaluation/test_checkpoint_evaluation.py::test_every_checkpoint_is_evaluated_and_summarised` ([keep (True, True)]) | SHRINK to 1 | 4 | 0.6 | 0.5 | Each case builds a dummy trainer, trains and evaluates every checkpoint; (True, True) runs the per-class report and EMA paths, the other cases only switch them off. |
| `tests/evaluation/classification/test_classification_metrics.py::test_detection_confusion_matrix_leaves_its_background_cell_blank` | REMOVE | 1 | 0.1 | 0.1 | Draws two images to check one blank cell; cosmetic. |
| `tests/evaluation/classification/test_classification_metrics.py::test_confusion_matrix_files` | REMOVE | 1 | 0.1 | 0.1 | PNG and CSV writing of a matrix already checked by test_confusion_matrix_and_metrics_match_hand_computed_values. |
| `tests/evaluation/classification/test_score_curves.py::test_curve_image_titles_stay_inside_the_image` | REMOVE | 1 | 0.1 | 0.1 | Renders images to check title ink; cosmetic. |
| `tests/evaluation/test_checkpoint_evaluation.py::test_tied_checkpoints_are_all_reported` | REMOVE | 1 | 0.1 | 0.1 | Full dummy training and evaluation to check the wording of one console line. |
| `tests/evaluation/classification/test_temperature_explorer.py::test_explorer_page_describes_every_temperature` | REMOVE | 3 | 0.1 | 0.1 | HTML page of an interactive tool; the thresholds it saves are still checked by test_saved_thresholds_are_written_next_to_the_page. |
| `tests/evaluation/test_checkpoint_evaluation.py::test_evaluation_folder_keeps_the_name_of_the_evaluated_run` | REMOVE | 1 | 0.1 | 0.1 | Folder naming of the rarely used standalone evaluation; trainer-backed and slow. |
| `tests/evaluation/classification/test_classification_metrics.py::test_class_predicted_but_absent_lowers_the_macro_f1` | REMOVE | 1 | 0.0 | 0.0 | Macro F1 values pinned by test_confusion_matrix_and_metrics_match_hand_computed_values. |
| `tests/evaluation/detection/test_average_precision_report.py::test_average_recall_table_and_resolution_of_the_sizes` | REMOVE | 1 | 0.0 | 0.0 | Formatting; values covered by test_mean_average_recall_of_all_the_objects_and_of_every_size. |
| `tests/evaluation/detection/test_mean_average_precision.py::test_point_average_precision_matches_its_definition` ([keep "101_point"]) | SHRINK to 1 | 2 | 0.0 | 0.0 | Same vectorised code for both point counts; the 101-point case (COCO) is the one in use. |
| `tests/evaluation/test_evaluation_report.py::test_report_follows_the_per_class_report` | REMOVE | 2 | 0.0 | 0.0 | Per-class part covered by test_every_checkpoint_is_evaluated_and_summarised (True, True), hiding covered by test_per_class_tables_are_hidden_without_the_per_class_report. |
| `tests/evaluation/detection/test_mean_average_precision.py::test_class_score_thresholds_of_the_worked_example` ([keep 0.99]) | SHRINK to 1 | 3 | 0.0 | 0.0 | Second target precision near-identical; thresholds numerically kept by test_class_thresholds_reach_the_target_precision. |
| `tests/evaluation/detection/test_average_precision_report.py::test_object_size_table` | REMOVE | 1 | 0.0 | 0.0 | Formatting; size values covered by test_average_precision_by_object_size. |
| `tests/evaluation/classification/test_temperature_explorer.py::test_saved_thresholds_of_another_evaluation_are_refused` ([keep classes {"bad": None}]) | SHRINK to 1 | 12 | 0.0 | 0.0 | One refusal case keeps the guard against thresholds of another evaluation. |
| `tests/evaluation/classification/test_temperature_explorer.py::test_explorer_draws_the_lowest_f1_first_and_at_most_the_maximum` | REMOVE | 1 | 0.0 | 0.0 | Writes a page to check class order and HTML escaping; display only. |
| `tests/evaluation/keypoints/test_keypoint_average_precision.py::test_missed_objects_and_false_detections_lower_the_recall_and_precision` | REMOVE | 1 | 0.0 | 0.0 | OKS AP numerically kept by test_partially_correct_keypoints_lower_the_strict_thresholds_first; AP matching shared with the box AP tests. |
| `tests/evaluation/detection/test_average_precision_report.py::test_mean_average_precision_table` | REMOVE | 1 | 0.0 | 0.0 | Console table formatting. |
| `tests/evaluation/detection/test_average_precision_report.py::test_box_and_mask_tables_are_told_apart_by_their_prefixes` | REMOVE | 1 | 0.0 | 0.0 | Report table and column prefixes, formatting. |
| `tests/evaluation/detection/test_mean_average_precision.py::test_per_class_average_precisions_and_ground_truths` | REMOVE | 1 | 0.0 | 0.0 | Per-threshold AP covered by test_coco_style_mean_over_thresholds and per-class rows by test_class_average_precision_rows_and_table. |
| `tests/evaluation/keypoints/test_keypoint_average_precision.py::test_perfect_keypoints_give_an_average_precision_of_one_at_every_threshold` | REMOVE | 1 | 0.0 | 0.0 | Covered by test_partially_correct_keypoints_lower_the_strict_thresholds_first and test_missed_objects_and_false_detections_lower_the_recall_and_precision (perfect objects). |
| `tests/evaluation/detection/test_mean_average_precision.py::test_worked_example_average_precision` ([keep "area"]) | SHRINK to 1 | 3 | 0.0 | 0.0 | 11 and 101-point methods pinned exactly by test_point_average_precision_matches_its_definition. |
| `tests/evaluation/detection/test_average_precision_report.py::test_class_thresholds_join_the_rows_and_the_table` | REMOVE | 1 | 0.0 | 0.0 | Column joining; threshold values covered by test_class_score_thresholds_of_the_worked_example. |
| `tests/evaluation/detection/test_mean_average_precision.py::test_class_without_ground_truth_or_detection_has_no_threshold` | REMOVE | 1 | 0.0 | 0.0 | Covered by test_class_thresholds_reach_the_target_precision (class without sample) and test_recall_counts_the_members_never_predicted (class never predicted). |
| `tests/evaluation/detection/test_mean_average_precision.py::test_perfect_detections_give_one` | REMOVE | 1 | 0.0 | 0.0 | Covered by test_per_class_average_precisions_and_ground_truths (exact box, AP 1 at every threshold). |
| `tests/evaluation/keypoints/test_keypoint_average_precision.py::test_keypoint_report_prefixes_its_tables_scalars_and_columns` | REMOVE | 1 | 0.0 | 0.0 | Prefix formatting, same mechanism as test_box_and_mask_tables_are_told_apart_by_their_prefixes. |
| `tests/evaluation/test_recall_intervals.py::test_bootstrap_interval_is_reproducible_and_holds_the_recall` | REMOVE | 1 | 0.0 | 0.0 | Bootstrap kept by test_bootstrap_interval_matches_the_exact_interval_for_independent_samples. |
| `tests/evaluation/test_recall_intervals.py::test_bootstrap_interval_widens_for_few_groups` | REMOVE | 1 | 0.0 | 0.0 | Bootstrap kept by test_bootstrap_interval_matches_the_exact_interval_for_independent_samples. |
| `tests/evaluation/detection/test_mean_average_precision.py::test_no_detection_gives_zero` | REMOVE | 1 | 0.0 | 0.0 | Covered by test_wrong_class_never_matches (object class AP 0 counted in the mean). |
| `tests/evaluation/keypoints/test_keypoint_average_precision.py::test_object_sizes_are_measured_by_the_box_areas` | REMOVE | 1 | 0.0 | 0.0 | Size buckets covered by test_average_precision_by_object_size and test_keypoint_boxes_span_the_keypoints_found. |
| `tests/losses/test_self_supervised_losses.py::test_embedding_standard_deviation_tells_the_collapse` | REMOVE | 1 | 0.0 | 0.0 | Monitoring statistic, not a loss; light property check. |
| `tests/evaluation/test_recall_intervals.py::test_bootstrap_interval_without_variance_is_exact_over_the_groups` | REMOVE | 1 | 0.0 | 0.0 | Zero-variance edge of the bootstrap. |
| `tests/evaluation/classification/test_score_curves.py::test_drawn_curves_keep_at_most_the_maximum_points` | REMOVE | 1 | 0.0 | 0.0 | Decimation of drawn points; metrics covered by test_curve_metrics_match_scikit_learn_with_ties. |
| `tests/evaluation/detection/test_mean_average_precision.py::test_class_without_ground_truth_is_left_out` | REMOVE | 1 | 0.0 | 0.0 | Covered by test_wrong_class_never_matches (predicted class without ground truth left out). |
| `tests/evaluation/detection/test_mean_average_precision.py::test_detection_matched_to_an_object_of_another_size_is_ignored` | REMOVE | 1 | 0.0 | 0.0 | Edge of the size breakdown; size AP kept by test_average_precision_by_object_size. |
| `tests/evaluation/test_report_formatting.py::test_report_values_are_formatted` | REMOVE | 6 | 0.0 | 0.0 | Remaining case covered by test_report_block_is_rendered_as_plain_text (4 decimals, missing values). |
| `tests/evaluation/test_report_formatting.py::test_unknown_report_values_are_refused` | REMOVE | 1 | 0.0 | 0.0 | Trivial type guard of a formatter. |
| `tests/evaluation/test_report_formatting.py::test_per_class_tables_are_hidden_without_the_per_class_report` | REMOVE | 1 | 0.0 | 0.0 | Report formatting option. |
| `tests/evaluation/detection/test_mean_average_precision.py::test_duplicate_detection_is_a_false_positive` | REMOVE | 1 | 0.0 | 0.0 | False positives in score order covered by test_worked_example_average_precision. |
| `tests/losses/test_keypoint_losses.py::test_heatmap_loss_is_divided_by_the_number_of_keypoints` | REMOVE | 1 | 0.0 | 0.0 | Heatmap loss numerically kept by test_heatmap_loss_matches_a_hand_computed_case. |
| `tests/evaluation/test_evaluation_report.py::test_class_metrics_file` | REMOVE | 1 | 0.0 | 0.0 | CSV writing; value formatting covered by test_file_values_are_formatted, file written by the checkpoint evaluation test. |
| `tests/evaluation/test_report_formatting.py::test_table_is_laid_out_with_its_title_and_legend` | REMOVE | 1 | 0.0 | 0.0 | Layout wording, covered by test_report_block_is_rendered_as_plain_text. |
| `tests/evaluation/detection/test_mean_average_precision.py::test_overlap_below_threshold_is_a_false_positive` | REMOVE | 2 | 0.0 | 0.0 | Covered by test_coco_style_mean_over_thresholds (found below, missed above the overlap). |
| `tests/evaluation/self_supervised/test_representation_metrics.py::test_nearest_neighbours_classify_separated_clusters` | REMOVE | 1 | 0.0 | 0.0 | Basic case covered by test_nearest_neighbours_leave_the_image_out and test_nearest_neighbour_vote_is_weighted_and_ties_go_to_the_lowest_class. |
| `tests/losses/test_segmentation_losses.py::test_ignored_pixels_give_no_loss_nor_gradient` | REMOVE | 1 | 0.0 | 0.0 | Ignore index of the cross entropy kept by test_cross_entropy_without_focal_factor_is_the_weighted_smoothed_cross_entropy. |
| `tests/losses/test_self_supervised_losses.py::test_info_nce_loss_of_identical_and_random_views` | REMOVE | 1 | 0.0 | 0.0 | Reference values implied by test_info_nce_loss_matches_the_cross_entropy_written_by_hand. |
| `tests/evaluation/test_report_formatting.py::test_report_header` | REMOVE | 1 | 0.0 | 0.0 | Header covered by test_report_block_is_rendered_as_plain_text. |
| `tests/evaluation/test_sample_selection.py::test_samples_without_class_are_evenly_spread` | REMOVE | 1 | 0.0 | 0.0 | Choice of the images shown in TensorBoard, low stake. |
| `tests/evaluation/test_sample_selection.py::test_samples_are_grouped_by_their_most_frequent_class` | REMOVE | 1 | 0.0 | 0.0 | Edge grouping of multi-class samples for the shown images only. |
| `tests/evaluation/test_recall_intervals.py::test_clopper_pearson_bounds_of_fractional_counts` | REMOVE | 1 | 0.0 | 0.0 | Exact interval kept by test_clopper_pearson_bounds_match_their_closed_forms. |
| `tests/evaluation/test_recall_intervals.py::test_clopper_pearson_rejects_inconsistent_counts` | REMOVE | 1 | 0.0 | 0.0 | Input validation only. |
| `tests/evaluation/test_recall_intervals.py::test_bootstrap_interval_widens_with_correlated_samples` | REMOVE | 1 | 0.0 | 0.0 | Motivation property; the group bootstrap is already pinned by test_bootstrap_interval_widens_for_few_groups and the exact-interval match. |
| `tests/evaluation/classification/test_class_metrics_report.py::test_class_metrics_rows` | REMOVE | 1 | 0.0 | 0.0 | Report rows; per-class rows kept by test_class_average_precision_rows_and_table. |
| `tests/evaluation/classification/test_classification_metrics.py::test_class_absent_from_the_evaluation_is_left_out_of_the_averages` | REMOVE | 1 | 0.0 | 0.0 | Covered by test_class_predicted_but_absent_lowers_the_macro_f1 (balanced accuracy over the present classes). |
| `tests/evaluation/classification/test_classification_metrics.py::test_multi_label_class_predicted_but_absent_lowers_the_macro_f1` | REMOVE | 1 | 0.0 | 0.0 | Same rule as test_class_predicted_but_absent_lowers_the_macro_f1 on the multi-label path. |
| `tests/evaluation/classification/test_classification_metrics.py::test_tied_scores_are_cut_together` | REMOVE | 1 | 0.0 | 0.0 | Tie edge; thresholds kept by test_class_thresholds_reach_the_target_precision. |
| `tests/evaluation/classification/test_score_curves.py::test_separated_reversed_and_undefined_classes` | REMOVE | 1 | 0.0 | 0.0 | Edge values; curve metrics kept by test_curve_metrics_match_scikit_learn_with_ties. |
| `tests/evaluation/classification/test_temperature_explorer.py::test_explorer_without_sample_is_refused` | REMOVE | 1 | 0.0 | 0.0 | Trivial refusal. |
| `tests/evaluation/classification/test_temperature_explorer.py::test_class_histograms_separate_scores_near_one` | REMOVE | 1 | 0.0 | 0.0 | Checks a bin width constant (0.02). |
| `tests/evaluation/classification/test_temperature_explorer.py::test_reliability_bins_separate_confidences_two_hundredths_apart` | REMOVE | 1 | 0.0 | 0.0 | Checks a bin width constant (0.02). |
| `tests/evaluation/classification/test_temperature_explorer.py::test_saved_thresholds_without_evaluation_are_refused` | REMOVE | 1 | 0.0 | 0.0 | Trivial refusal. |
| `tests/evaluation/detection/test_detection_confusion.py::test_overlap_matrix_of_boxes_and_masks` | REMOVE | 1 | 0.0 | 0.0 | Box IoU is torchvision; mask overlaps covered by test_mask_average_precision_equals_box_average_precision_for_box_masks. |
| `tests/evaluation/detection/test_mean_average_precision.py::test_worked_example_true_positives_and_curve` | REMOVE | 1 | 0.0 | 0.0 | Intermediate curve of the worked example whose AP value test_worked_example_average_precision pins. |
| `tests/evaluation/detection/test_region_recall.py::test_object_boxes_follow_the_preprocessing` | REMOVE | 1 | 0.0 | 0.0 | Diagnosis mode input; low stake for training. |
| `tests/evaluation/detection/test_region_recall.py::test_group_recalls_of_every_size` | REMOVE | 1 | 0.0 | 0.0 | Covered by test_anchor_recall_rows_and_table (recall of every size). |
| `tests/evaluation/detection/test_region_recall.py::test_anchor_recall_rows_and_table` | REMOVE | 1 | 0.0 | 0.0 | Rows and table of a diagnosis mode; anchor overlaps kept by test_best_anchor_overlaps_of_every_level. |
| `tests/evaluation/keypoints/test_object_keypoint_similarity.py::test_measure_keypoint_accuracy_counts_the_keypoints_within_the_box_fraction` | REMOVE | 1 | 0.0 | 0.0 | PCK kept by test_summarise_keypoint_accuracy_gives_the_global_class_and_keypoint_values. |
| `tests/evaluation/keypoints/test_object_keypoint_similarity.py::test_summarise_keypoint_accuracy_without_labelled_keypoint_has_no_value` | REMOVE | 1 | 0.0 | 0.0 | Empty edge case. |
| `tests/evaluation/keypoints/test_object_keypoint_similarity.py::test_summarise_keypoint_accuracy_of_a_split_without_objects` | REMOVE | 1 | 0.0 | 0.0 | Empty edge case. |
| `tests/evaluation/segmentation/test_boundary_metrics.py::test_perfect_prediction_has_boundary_metrics_of_one` | REMOVE | 1 | 0.0 | 0.0 | Covered by test_contours_further_than_the_tolerance_are_wrong (F1 1 within tolerance). |
| `tests/evaluation/segmentation/test_boundary_metrics.py::test_ignored_pixels_draw_no_contour` | REMOVE | 1 | 0.0 | 0.0 | Ignored pixel edge; boundary metric kept by test_contours_further_than_the_tolerance_are_wrong. |
| `tests/evaluation/segmentation/test_segmentation_metrics.py::test_curve_metrics_join_the_class_rows_table_and_scalars` | REMOVE | 1 | 0.0 | 0.0 | Column and legend formatting. |
| `tests/evaluation/segmentation/test_segmentation_metrics.py::test_class_rows_table_and_scalars_leave_out_the_absent_classes` | REMOVE | 1 | 0.0 | 0.0 | Report formatting; absent class rule covered by test_metrics_of_a_known_confusion_matrix. |
| `tests/evaluation/self_supervised/test_representation_metrics.py::test_nearest_neighbours_leave_the_image_out` | REMOVE | 1 | 0.0 | 0.0 | k-NN kept by test_nearest_neighbour_vote_is_weighted_and_ties_go_to_the_lowest_class. |
| `tests/evaluation/self_supervised/test_representation_metrics.py::test_nearest_neighbours_give_the_same_predictions_by_chunks` | REMOVE | 1 | 0.0 | 0.0 | Memory chunking equivalence; k-NN kept by test_nearest_neighbour_vote_is_weighted_and_ties_go_to_the_lowest_class. |
| `tests/evaluation/sequence/test_frame_selection_metrics.py::test_ranking_metrics_before_decoding` | REMOVE | 1 | 0.0 | 0.0 | Secondary ranking metric; selection metrics kept by test_selection_metrics_of_a_decoding. |
| `tests/evaluation/sequence/test_frame_selection_metrics.py::test_selection_thresholds_keep_the_target_precision` | REMOVE | 1 | 0.0 | 0.0 | Threshold search covered by test_class_thresholds_reach_the_target_precision. |
| `tests/losses/test_classification_losses.py::test_probabilities_follow_the_output_mode` | REMOVE | 1 | 0.0 | 0.0 | Softmax or sigmoid dispatch, trivial. |
| `tests/losses/test_instance_segmentation_losses.py::test_mask_loss_without_positive_proposal_is_a_zero_in_the_graph` | REMOVE | 1 | 0.0 | 0.0 | Empty edge; mask loss kept by test_mask_loss_reads_the_channel_of_the_ground_truth_class. |
| `tests/losses/test_keypoint_losses.py::test_heatmap_loss_leaves_out_the_ignored_cells` | REMOVE | 1 | 0.0 | 0.0 | Ignored-cell edge of the heatmap loss; loss kept by test_heatmap_loss_matches_a_hand_computed_case. |
| `tests/losses/test_keypoint_losses.py::test_objects_of_one_keypoint_and_empty_batches_give_finite_losses` | REMOVE | 1 | 0.0 | 0.0 | Edge cases; tag losses kept by test_tag_losses_match_a_hand_computed_case. |
| `tests/losses/test_segmentation_losses.py::test_dice_loss_of_a_perfect_prediction_is_zero` | REMOVE | 1 | 0.0 | 0.0 | Sanity bound; Dice covered by test_dice_loss_leaves_out_the_absent_classes and test_ignored_pixels_give_no_loss_nor_gradient. |
| `tests/losses/test_segmentation_losses.py::test_lovasz_loss_gives_a_gradient_to_the_wrong_pixels` | REMOVE | 1 | 0.0 | 0.0 | Gradient existence; values pinned by test_lovasz_loss_of_hard_predictions_is_one_minus_the_iou. |
| `tests/evaluation/test_report_formatting.py::test_file_values_are_formatted` | REMOVE | 1 | 0.0 | 0.0 | CSV value formatting, low stake. |
| `tests/evaluation/test_report_formatting.py::test_table_rows_are_shown_worst_first` | REMOVE | 1 | 0.0 | 0.0 | Display order, covered by test_report_block_is_rendered_as_plain_text (per-class table worst first). |
| `tests/evaluation/test_recall_intervals.py::test_counts_are_added_up_per_group` | REMOVE | 1 | 0.0 | 0.0 | Covered by test_image_class_pixel_counts_add_up_to_the_confusion_matrix and test_selection_metrics_of_a_decoding (per-group counts). |
| `tests/evaluation/test_recall_intervals.py::test_recall_interval_table` | REMOVE | 1 | 0.0 | 0.0 | Console table formatting. |
| `tests/evaluation/classification/test_class_metrics_report.py::test_class_metrics_table` | REMOVE | 1 | 0.0 | 0.0 | Console table formatting; values covered by test_class_metrics_rows. |
| `tests/evaluation/classification/test_class_metrics_report.py::test_class_metric_scalars` | REMOVE | 1 | 0.0 | 0.0 | TensorBoard tag names only. |
| `tests/evaluation/classification/test_class_metrics_report.py::test_curve_metrics_of_the_per_class_report` | REMOVE | 1 | 0.0 | 0.0 | Column order and legend; metric values covered by test_curve_metrics_match_scikit_learn_with_ties. |
| `tests/evaluation/classification/test_class_metrics_report.py::test_threshold_columns_of_several_target_precisions` | REMOVE | 1 | 0.0 | 0.0 | Column naming of the rarely used additional target precisions. |
| `tests/evaluation/classification/test_class_metrics_report.py::test_thresholds_need_one_target_precision_each` | REMOVE | 1 | 0.0 | 0.0 | Input validation only. |
| `tests/evaluation/classification/test_classification_metrics.py::test_recall_counts_the_members_never_predicted` | REMOVE | 1 | 0.0 | 0.0 | Detection variant of the thresholds kept by test_class_thresholds_reach_the_target_precision. |
| `tests/evaluation/detection/test_detection_confusion.py::test_wrong_class_detection_finds_an_object_left_unfound` | REMOVE | 1 | 0.0 | 0.0 | The confusion of the second pass is in test_confusion_matrix_counts_every_detection_and_object_once. |
| `tests/evaluation/detection/test_detection_confusion.py::test_most_confident_detection_takes_the_object` | REMOVE | 1 | 0.0 | 0.0 | Matching order rule; confusion matrix kept by test_confusion_matrix_counts_every_detection_and_object_once. |
| `tests/evaluation/detection/test_detection_confusion.py::test_overlap_threshold_is_inclusive` | REMOVE | 2 | 0.0 | 0.0 | Boundary edge of the matching; remaining case of the first-pass shrink dropped. |
| `tests/evaluation/detection/test_detection_confusion.py::test_errors_are_grouped_by_folder` | REMOVE | 1 | 0.0 | 0.0 | Grouping of error images for display only. |
| `tests/evaluation/detection/test_mean_average_precision.py::test_unknown_method_is_rejected` | REMOVE | 1 | 0.0 | 0.0 | Trivial guard. |
| `tests/evaluation/detection/test_object_sizes.py::test_box_areas_of_valid_and_degenerate_boxes` | REMOVE | 1 | 0.0 | 0.0 | Trivial area formula, used through test_average_precision_by_object_size. |
| `tests/evaluation/detection/test_object_sizes.py::test_every_area_has_one_size` | REMOVE | 1 | 0.0 | 0.0 | Size bounds exercised by test_average_precision_by_object_size. |
| `tests/evaluation/detection/test_object_sizes.py::test_unknown_size_is_refused` | REMOVE | 1 | 0.0 | 0.0 | Trivial guard. |
| `tests/evaluation/detection/test_region_recall.py::test_proposal_recall_without_any_image` | REMOVE | 1 | 0.0 | 0.0 | Empty edge case. |
| `tests/evaluation/keypoints/test_object_keypoint_similarity.py::test_sigma_table_accepts_equal_sigmas_and_rejects_conflicting_ones_of_merged_categories` | REMOVE | 1 | 0.0 | 0.0 | Configuration validation of merged categories. |
| `tests/evaluation/keypoints/test_object_keypoint_similarity.py::test_object_keypoint_similarity_of_perfect_and_unlabelled_objects` | REMOVE | 1 | 0.0 | 0.0 | Covered by test_object_keypoint_similarity_matches_a_hand_computed_case (perfect keypoint 1, unlabelled left out). |
| `tests/evaluation/keypoints/test_object_keypoint_similarity.py::test_keypoint_boxes_span_the_keypoints_found` | REMOVE | 1 | 0.0 | 0.0 | Box of the predicted keypoints, only used for the size buckets. |
| `tests/evaluation/keypoints/test_object_keypoint_similarity.py::test_ground_truth_objects_need_the_minimum_similarity_to_match` | REMOVE | 1 | 0.0 | 0.0 | Covered by test_a_far_prediction_is_neither_matched_nor_measured (B51, below the minimum). |
| `tests/evaluation/segmentation/test_segmentation_metrics.py::test_pixel_score_histograms_count_the_labelled_pixels` | REMOVE | 1 | 0.0 | 0.0 | Histogram curves kept by test_histogram_curves_equal_the_exact_curves_of_the_same_pixels. |
| `tests/evaluation/segmentation/test_segmentation_metrics.py::test_metrics_without_labelled_pixel_are_zero` | REMOVE | 1 | 0.0 | 0.0 | Empty edge case. |
| `tests/evaluation/sequence/test_segment_metrics.py::test_perfect_prediction_and_no_sequence` | REMOVE | 1 | 0.0 | 0.0 | Trivial perfect and empty cases. |
| `tests/evaluation/sequence/test_segment_metrics.py::test_unlabelled_frames_end_a_segment` | REMOVE | 1 | 0.0 | 0.0 | Edge of the segment split; segment metrics kept by test_fragmented_prediction_scores_low_on_segments. |
| `tests/losses/test_segmentation_losses.py::test_focal_factor_lowers_the_loss_of_well_classified_pixels` | REMOVE | 1 | 0.0 | 0.0 | Property check; segmentation cross entropy kept by test_cross_entropy_without_focal_factor_is_the_weighted_smoothed_cross_entropy. |
| `tests/losses/test_sequence_losses.py::test_frame_selection_loss_is_small_when_the_true_frames_stand_out` | REMOVE | 1 | 0.0 | 0.0 | Limit behaviour covered by test_frame_selection_loss_makes_the_frames_compete_for_every_wanted_class. |

### Models

| Test | Verdict | Cases | Time (seconds) | Saved (seconds) | Reason |
|---|---|---|---|---|---|
| `tests/models/sequence/test_sequence_comparator.py::test_sequence_comparator_outputs_and_padding_frames` ((keep [gru-cross_attention])) | SHRINK to 1 | 18 | 0.7 | 0.6 | 6x3 grid; module shapes covered by test_temporal_modules output test, fusions by test_member_fusion, padding by test_batch_padding. |
| `tests/models/instance_segmentation/test_instance_segmenter.py::test_inference_instance_segmenter_matches_the_eager_model` ((keep [mask_head_settings1])) | SHRINK to 1 | 2 | 1.2 | 0.6 | Same equivalence check on two head settings. |
| `tests/models/classification/test_image_classifier.py::test_classifier_learns_a_tiny_batch` | REMOVE | 1 | 0.6 | 0.6 | Optimisation smoke test; training loop covered by the trainer tests. |
| `tests/models/test_model_transfer.py::test_source_run_written_before_the_pre_activation_flag` | REMOVE | 1 | 0.5 | 0.5 | Rejects legacy runs; low stake. |
| `tests/models/test_model_transfer.py::test_mask_heads_of_another_mask_resolution_are_rejected` | REMOVE | 1 | 0.5 | 0.5 | Rejection message on two instance segmenters. |
| `tests/models/keypoints/test_keypoint_detector.py::test_detector_trains_under_the_bfloat16_autocast` | REMOVE | 2 | 0.3 | 0.3 | PyTorch autocast behaviour; training covered by test_losses_train_every_heatmap_and_the_tags_of_the_keypoints. |
| `tests/models/sequence/test_batch_padding.py::test_sequence_comparator_outputs_do_not_depend_on_the_batch_padding` | REMOVE | 21 | 0.3 | 0.3 | Batch padding still checked for the classifier, which shares the encoder. |
| `tests/models/keypoints/test_hrnet.py::test_branches_halve_the_resolution_and_double_the_width` | REMOVE | 6 | 0.2 | 0.2 | The keypoint detector tests build and run the same HRNet. |
| `tests/models/instance_segmentation/test_instance_segmenter.py::test_instance_segmenter_summaries_list_the_mask_heads` | REMOVE | 1 | 0.2 | 0.2 | Summary wording on a full model. |
| `tests/models/sequence/test_sequence_comparator.py::test_mask_free_path_matches_the_masked_path` ((keep [transformer-cross_attention])) | SHRINK to 1 | 15 | 0.2 | 0.2 | One case keeps the deployment-path equivalence. |
| `tests/models/test_model_transfer.py::test_different_components_are_rejected` | REMOVE | 9 | 0.2 | 0.2 | Rejection message. |
| `tests/models/sequence/test_sequence_classifier.py::test_sequence_classifier_outputs_and_padding_frames` ((keep [transformer])) | SHRINK to 1 | 6 | 0.1 | 0.1 | Per-module shapes and padding covered by test_temporal_modules::test_output_shapes_and_padding_frames_are_ignored and test_batch_padding. |
| `tests/models/test_model_transfer.py::test_averaged_weights_initialise_a_run_keeping_a_moving_average` | REMOVE | 3 | 0.1 | 0.1 | Averaged loading covered by test_model_loading.py::test_trained_and_averaged_weights_are_loaded. |
| `tests/models/test_model_transfer.py::test_decoders_of_other_architectures_are_rejected` | REMOVE | 3 | 0.1 | 0.1 | Rejection message. |
| `tests/models/components/test_architecture_modules.py::test_backbone_modules_forward_shapes` | REMOVE | 4 | 0.1 | 0.1 | Backbones are built and run by the model tests of every task. |
| `tests/models/test_model_transfer.py::test_classes_of_another_order_or_fewer_classes_are_rejected` | REMOVE | 3 | 0.1 | 0.1 | Rejection message. |
| `tests/models/test_model_transfer.py::test_heads_of_other_input_sizes_are_rejected` | REMOVE | 3 | 0.1 | 0.1 | Rejection message. |
| `tests/models/test_model_transfer.py::test_backbone_moves_between_batch_and_sequence_normalization` | REMOVE | 2 | 0.1 | 0.1 | Low-stake key mapping between two normalization types. |
| `tests/models/detection/test_anchors.py::test_best_overlaps_equal_the_best_of_every_anchor` ((keep [8-scales2-aspect_ratios2])) | SHRINK to 1 | 3 | 0.1 | 0.1 | Same brute-force comparison on three anchor grids; known-box test kept. |
| `tests/models/sequence/test_sequence_labeller.py::test_sequence_labeller_outputs_one_prediction_per_frame` ((keep [transformer])) | SHRINK to 1 | 4 | 0.1 | 0.1 | Per-module time-axis shapes covered by test_temporal_modules output test (keep_time_axis cases). |
| `tests/models/sequence/test_sequence_comparator.py::test_batch_normalization_sees_both_members` | REMOVE | 1 | 0.1 | 0.1 | Shared statistics also shown by the pair scope case. |
| `tests/models/keypoints/test_keypoint_detector.py::test_losses_train_every_heatmap_and_the_tags_of_the_keypoints` ((keep [True-16])) | SHRINK to 1 | 3 | 0.1 | 0.0 | The higher-resolution case with tags trains every heatmap. |
| `tests/models/sequence/test_batch_padding.py::test_sequence_classifier_outputs_do_not_depend_on_the_batch_padding` ((keep [transformer])) | SHRINK to 1 | 7 | 0.1 | 0.0 | One module is enough for the padding guard. |
| `tests/models/keypoints/test_keypoint_detector.py::test_detector_gives_candidates_without_ground_truth` ((keep [True-16])) | SHRINK to 1 | 4 | 0.1 | 0.0 | 2x2 grid; the higher-resolution case with tags runs every head. |
| `tests/models/keypoints/test_hrnet.py::test_higher_resolution_branch_doubles_the_resolution` | REMOVE | 3 | 0.0 | 0.0 | Covered by test_keypoint_detector.py::test_higher_resolution_heatmaps_are_averaged_at_half_the_resolution. |
| `tests/models/test_model_transfer.py::test_sequence_classifier_temporal_module_initialises_other_sequence_tasks` | REMOVE | 2 | 0.0 | 0.0 | Temporal module transfer covered by test_temporal_module_and_head_initialise_another_sequence_classifier. |
| `tests/models/sequence/test_sequence_labeller.py::test_causal_sequence_labeller_does_not_look_ahead` | REMOVE | 3 | 0.0 | 0.0 | Covered by test_temporal_modules::test_causal_modules_do_not_look_ahead on the same modules. |
| `tests/models/segmentation/test_semantic_segmenter.py::test_logits_at_the_input_resolution` ((keep [bilinear-layer-True-True])) | SHRINK to 1 | 3 | 0.1 | 0.0 | Transposed upsampling covered by test_last_max_pooling_adds_a_decoder_level and test_deep_supervision_heads_add_the_losses_of_the_intermediate_levels. |
| `tests/models/test_model_transfer.py::test_source_run_written_before_the_head_normalization_is_rejected` | REMOVE | 2 | 0.0 | 0.0 | Rejects legacy runs (message). |
| `tests/models/sequence/test_sequence_comparator.py::test_sequence_normalization_scope_of_the_pairs` ((keep [pair-True])) | SHRINK to 1 | 2 | 0.1 | 0.0 | The member scope repeats test_only_the_cross_attention_mixes_the_members. |
| `tests/models/test_model_transfer.py::test_classification_head_adds_classes_after_the_source_classes` ((keep [classification])) | SHRINK to 1 | 2 | 0.1 | 0.0 | Sequence labelling head uses the same class-extension code. |
| `tests/models/classification/test_image_classifier.py::test_head_input_follows_the_pooled_feature_map` ((keep [flatten-64-True-512])) | SHRINK to 1 | 4 | 0.0 | 0.0 | Same size arithmetic; pooling shapes also in test_global_pooling_and_head_shapes. |
| `tests/models/sequence/test_batch_padding.py::test_sequence_labeller_outputs_do_not_depend_on_the_batch_padding` | REMOVE | 4 | 0.0 | 0.0 | Same padding guard as the classifier case. |
| `tests/models/self_supervised/test_self_supervised_model.py::test_model_computes_the_loss_of_its_method` ((keep [barlow_twins])) | SHRINK to 1 | 2 | 0.1 | 0.0 | One self-supervised method. |
| `tests/models/sequence/test_temporal_modules.py::test_output_shapes_and_padding_frames_are_ignored` ((keep [attention_pooling-False-False], [transformer-True-False])) | SHRINK to 2 | 13 | 0.0 | 0.0 | One pooling and one time-axis case. |
| `tests/models/test_model_transfer.py::test_normalization_parameters_stay_fixed` | REMOVE | 2 | 0.0 | 0.0 | Freezing covered by test_frozen_backbone_keeps_its_weights_and_statistics. |
| `tests/models/test_model_transfer.py::test_order_blind_transformer_initialises_runs_of_other_sequence_lengths` | REMOVE | 1 | 0.0 | 0.0 | Narrow exception of the transfer rules. |
| `tests/models/test_model_transfer.py::test_flattened_head_initialises_a_flattened_run` | REMOVE | 1 | 0.0 | 0.0 | Head transfer covered by test_classification_head_adds_classes_after_the_source_classes. |
| `tests/models/test_model_transfer.py::test_head_tensors_of_another_shape_are_listed` | REMOVE | 1 | 0.0 | 0.0 | Rejection message. |
| `tests/models/test_model_loading.py::test_averaged_weights_of_a_checkpoint_without_average_are_refused` | REMOVE | 1 | 0.0 | 0.0 | Rejection message. |
| `tests/models/test_model_transfer.py::test_head_with_another_dropout_rate_is_transferred` | REMOVE | 1 | 0.0 | 0.0 | Low-stake key naming. |
| `tests/models/components/test_architecture_modules.py::test_summary_names_the_shortcuts_of_blocks_adding_one` ((keep [3-2-0])) | SHRINK to 1 | 6 | 0.0 | 0.0 | The case of the fixed summary bug. |
| `tests/models/test_model_transfer.py::test_source_run_written_before_a_temporal_setting_is_rejected` | REMOVE | 1 | 0.0 | 0.0 | Rejects legacy runs (message). |
| `tests/models/test_model_transfer.py::test_source_run_written_before_the_maximum_sequence_length_is_rejected` | REMOVE | 1 | 0.0 | 0.0 | Rejects legacy runs (message). |
| `tests/models/test_model_transfer.py::test_source_run_without_the_component_is_rejected` | REMOVE | 1 | 0.0 | 0.0 | Trivial error path (component absent from task). |
| `tests/models/keypoints/test_keypoint_detector.py::test_heatmap_loss_leaves_out_the_ignored_regions` ((keep [True])) | SHRINK to 1 | 2 | 0.0 | 0.0 | Same mask logic for both settings. |
| `tests/models/detection/test_region_proposer.py::test_region_proposer` | REMOVE | 1 | 0.0 | 0.0 | Region proposer forward and loss covered by test_detector_baseline.py::test_training_losses_match_the_baseline. |
| `tests/models/sequence/test_frame_selector.py::test_frame_selector_outputs_and_losses` ((keep [2.5])) | SHRINK to 1 | 2 | 0.0 | 0.0 | The non-zero weight case exercises the same outputs plus the extra loss term. |
| `tests/models/components/test_convolution_block.py::test_convolution_block_modules` | REMOVE | 1 | 0.0 | 0.0 | Block layout covered by test_architecture_modules.py::test_plain_convolution_block_state_dictionary_keys. |
| `tests/models/components/test_fully_connected_block.py::test_head_keeps_the_features_of_a_hidden_layer_before_its_relu` ((keep [batch])) | SHRINK to 1 | 3 | 0.0 | 0.0 | Normalization choice does not change where the features are taken. |
| `tests/models/sequence/test_sequence_classifier.py::test_sequence_normalization_ignores_the_padding_frames` | REMOVE | 1 | 0.0 | 0.0 | Covered by components/test_sequence_normalization.py and the batch padding test. |
| `tests/models/self_supervised/test_self_supervised_model.py::test_model_gives_representations_and_embeddings` ((keep [flatten-2048])) | SHRINK to 1 | 3 | 0.0 | 0.0 | Pooling variants covered by classification/test_image_classifier.py::test_global_pooling_and_head_shapes. |
| `tests/models/classification/test_image_classifier.py::test_global_pooling_and_head_shapes` ((keep [max-expected_vector1])) | SHRINK to 1 | 3 | 0.0 | 0.0 | Average and flatten shapes covered by test_head_input_follows_the_pooled_feature_map and the self-supervised representations test. |
| `tests/models/classification/test_image_classifier.py::test_classifier_output_shapes_and_probabilities` ((keep [multi_label])) | SHRINK to 1 | 2 | 0.0 | 0.0 | Multi-class output is run by most model tests. |
| `tests/models/instance_segmentation/test_mask_operations.py::test_mask_head_doubles_its_pooled_features_up_to_the_mask_resolution` ((keep [7-28-bilinear-2])) | SHRINK to 1 | 4 | 0.0 | 0.0 | One output shape case per head. |
| `tests/models/sequence/test_sequence_classifier.py::test_frames_are_encoded_without_the_padding_frames` | REMOVE | 1 | 0.0 | 0.0 | Covered by test_sequence_comparator.py::test_padding_frames_never_reach_the_backbone. |
| `tests/models/segmentation/test_semantic_segmenter.py::test_summary_and_size_components` | REMOVE | 1 | 0.0 | 0.0 | Summary wording. |
| `tests/models/segmentation/test_semantic_segmenter.py::test_last_max_pooling_adds_a_decoder_level` | REMOVE | 1 | 0.0 | 0.0 | Architecture option; output shape kept in test_logits_at_the_input_resolution. |
| `tests/models/backbones/test_backbone_outputs.py::test_pre_pooling_outputs_of_every_block` ((keep [True-4])) | SHRINK to 1 | 2 | 0.0 | 0.0 | Pre-activation case runs the same output collection. |
| `tests/models/sequence/test_frame_selection_decoding.py::test_hungarian_decoding_gives_the_most_probable_labelling` ((keep [0])) | SHRINK to 1 | 4 | 0.0 | 0.0 | Random seeds of the same brute-force comparison. |
| `tests/models/self_supervised/test_self_supervised_model.py::test_projection_head_without_hidden_layer` | REMOVE | 1 | 0.0 | 0.0 | Low-stake head option. |
| `tests/models/sequence/test_temporal_modules.py::test_non_causal_modules_look_ahead` | REMOVE | 3 | 0.0 | 0.0 | Counterpart of the causal test; low stake. |
| `tests/models/backbones/test_backbone_summary.py::test_summary_leaves_the_backbone_untouched` | REMOVE | 1 | 0.0 | 0.0 | Low-stake summary side effect. |
| `tests/models/sequence/test_frame_selector.py::test_unselected_class_is_saved_with_the_weights` | REMOVE | 1 | 0.0 | 0.0 | Checks one buffer in the checkpoint; low stake. |
| `tests/models/test_model_transfer.py::test_source_class_names_of_every_record_format` | REMOVE | 4 | 0.0 | 0.0 | The record reader is covered by test_source_run_without_class_names_keeps_its_class_order and the class-extension tests. |
| `tests/models/segmentation/test_semantic_segmenter.py::test_weight_names_do_not_depend_on_the_dropout_rate` | REMOVE | 1 | 0.0 | 0.0 | Low-stake key naming. |
| `tests/models/sequence/test_temporal_modules.py::test_ordered_modules_depend_on_the_frame_order` ((keep [transformer])) | SHRINK to 1 | 4 | 0.0 | 0.0 | Recurrent order dependence is PyTorch behaviour. |
| `tests/models/components/test_architecture_modules.py::test_pre_activation_shortcut_placement` ((keep [3-3])) | SHRINK to 1 | 2 | 0.0 | 0.0 | Channel-changing case is the past bug; [8-2] repeats test_residual_shortcut_placement. |
| `tests/models/sequence/test_temporal_modules.py::test_poolings_ignore_the_frame_order` | REMOVE | 3 | 0.0 | 0.0 | Order invariance of the poolings; low stake. |
| `tests/models/components/test_fully_connected_block.py::test_batch_normalization_momentums_reach_every_layer` | REMOVE | 2 | 0.0 | 0.0 | Low-stake setting propagation. |
| `tests/models/backbones/test_backbone_summary.py::test_summary_lists_the_shortcuts_of_the_blocks_adding_one` | REMOVE | 1 | 0.0 | 0.0 | Covered by test_architecture_modules.py::test_summary_names_the_shortcuts_of_blocks_adding_one [3-2-0]. |
| `tests/models/sequence/test_temporal_modules.py::test_causal_modules_do_not_look_ahead` ((keep [transformer-True], [temporal_convolution-True])) | SHRINK to 2 | 4 | 0.0 | 0.0 | The unidirectional GRU not looking ahead is PyTorch behaviour. |
| `tests/models/sequence/test_sequence_comparator.py::test_a_single_mask_is_rejected` | REMOVE | 1 | 0.0 | 0.0 | Trivial argument validation. |
| `tests/models/test_model_transfer.py::test_detection_heads_of_another_batch_normalization_momentum_match` | REMOVE | 2 | 0.0 | 0.0 | Low-stake comparison option. |
| `tests/models/detection/test_inference_detector.py::test_postprocess_detections_with_fewer_candidates_than_detections` | REMOVE | 2 | 0.0 | 0.0 | Padding edge case of the postprocess kept in test_postprocess_detections_threshold_nms_and_padding. |
| `tests/models/components/test_architecture_modules.py::test_pre_activation_convolution_block_layers` | REMOVE | 1 | 0.0 | 0.0 | Layer order also fixed by test_pre_activation_shortcut_placement. |
| `tests/models/components/test_architecture_modules.py::test_residual_shortcut_placement` ((keep [True])) | SHRINK to 1 | 2 | 0.0 | 0.0 | One placement mode is enough. |
| `tests/models/components/test_fully_connected_block.py::test_fully_connected_layers` ((keep [batch-expected_layer_types1])) | SHRINK to 1 | 3 | 0.0 | 0.0 | Same layer building for every normalization. |
| `tests/models/components/test_fully_connected_block.py::test_layer_normalization_normalizes_every_vector_alone` | REMOVE | 1 | 0.0 | 0.0 | PyTorch LayerNorm behaviour. |
| `tests/models/sequence/test_member_fusion.py::test_member_vectors_are_combined` ((keep [cross_attention-9])) | SHRINK to 1 | 3 | 0.0 | 0.0 | Concatenation and difference are trivial. |
| `tests/models/sequence/test_temporal_modules.py::test_module_summary_leaves_out_the_pretrained_settings` | REMOVE | 1 | 0.0 | 0.0 | Summary wording. |
| `tests/models/sequence/test_frame_selection_decoding.py::test_documented_example_is_decoded_as_described` | REMOVE | 1 | 0.0 | 0.0 | Documentation example. |
| `tests/models/test_model_transfer.py::test_keypoint_heads_compare_their_hidden_width` | REMOVE | 3 | 0.0 | 0.0 | Rejection message. |
| `tests/models/instance_segmentation/test_instance_segmenter.py::test_instance_segmentation_comparison_reports_mask_differences` | REMOVE | 1 | 0.0 | 0.0 | Export comparison tool; its export test is removed above. |
| `tests/models/detection/test_inference_detector.py::test_postprocess_detections_threshold_nms_and_padding` ((keep [True])) | SHRINK to 1 | 2 | 0.0 | 0.0 | Boolean variant of the same postprocess path. |
| `tests/models/detection/test_target_assignment.py::test_ground_truth_boxes_overwrite_the_last_proposal_slots` | REMOVE | 1 | 0.0 | 0.0 | Ground truth proposals covered by test_every_object_gives_a_positive_proposal_with_its_ground_truth_box. |
| `tests/models/detection/test_target_assignment.py::test_ground_truths_are_padded_to_the_largest_image` | REMOVE | 1 | 0.0 | 0.0 | Images with several object counts covered by test_proposal_targets_of_images_with_different_object_counts. |
| `tests/models/detection/test_two_stage_detector.py::test_proposals_give_the_index_of_their_matched_ground_truth` | REMOVE | 1 | 0.0 | 0.0 | Matched ground truths are also checked by test_detection_stages_carry_levels_and_matched_ground_truths. |
| `tests/models/keypoints/test_keypoint_detector.py::test_channel_table_gives_consecutive_channels_per_class` | REMOVE | 1 | 0.0 | 0.0 | Trivial index table. |
| `tests/models/segmentation/test_sliding_window.py::test_overlapping_tiles_are_averaged` | REMOVE | 1 | 0.0 | 0.0 | Covered by test_tiled_segmentation_equals_the_whole_image_prediction. |
| `tests/models/self_supervised/test_self_supervised_model.py::test_unknown_method_is_refused` | REMOVE | 1 | 0.0 | 0.0 | Trivial argument validation. |
| `tests/models/components/test_sequence_normalization.py::test_statistics_are_those_of_every_sequence` | REMOVE | 1 | 0.0 | 0.0 | Covered by test_a_frame_depends_on_its_sequence_only. |
| `tests/models/sequence/test_member_fusion.py::test_fusions_without_attention_leave_the_frames_unchanged` | REMOVE | 2 | 0.0 | 0.0 | Covered by test_sequence_comparator.py::test_only_the_cross_attention_mixes_the_members [difference-False]. |
| `tests/models/sequence/test_frame_selection_decoding.py::test_unknown_decoding_is_rejected` | REMOVE | 1 | 0.0 | 0.0 | Trivial argument validation. |
| `tests/models/sequence/test_frame_selection_decoding.py::test_sequences_shorter_than_the_selections_leave_missing_frames` | REMOVE | 1 | 0.0 | 0.0 | Low-stake edge case. |
| `tests/models/test_model_transfer.py::test_mask_heads_and_class_layers_transfers_are_disabled` | REMOVE | 1 | 0.0 | 0.0 | Configuration switch; same mechanism kept in test_pretrained_settings_are_disabled. |
| `tests/models/keypoints/test_keypoint_grouping.py::test_image_without_candidates_gives_no_object` | REMOVE | 1 | 0.0 | 0.0 | Low-stake empty case. |

### Deployment, Scripts, Utilities

| Test | Verdict | Cases | Time (seconds) | Saved (seconds) | Reason |
|---|---|---|---|---|---|
| `tests/deployment/test_export_specifications.py::test_fixed_dynamic_dimensions_are_reported` | REMOVE | 1 | 0.9 | 0.9 | Export guard only. Dynamic frames are checked by test_second_recurrent_export_keeps_dynamic_frames. |
| `tests/deployment/test_triton_configuration.py::test_detection_triton_configuration_is_unchanged` | REMOVE | 2 | 0.4 | 0.4 | The configuration writer is covered by test_comparison_triton_configuration_has_one_line_per_member. |
| `tests/scripts/data/test_visualize_augmentation_strengths.py::test_figures_are_written_for_every_type` | REMOVE | 1 | 0.1 | 0.1 | Explanatory figure script main run, low stake. |
| `tests/scripts/embedding/test_cluster_image_embeddings.py::test_relaunch_loads_the_stored_features_and_computes_only_the_missing_layers` | REMOVE | 1 | 0.1 | 0.1 | Reuse of stored features covered by test_interrupted_extraction_continues_from_its_last_finished_batch. |
| `tests/scripts/embedding/test_cluster_image_embeddings.py::test_clustering_algorithms_separate_distinct_groups` | REMOVE | 1 | 0.1 | 0.1 | Mostly scikit-learn / HDBSCAN behaviour; clusters checked end to end in test_clustering_writes_clusters_links_mosaics_and_kept_images. |
| `tests/scripts/embedding/test_cluster_image_embeddings.py::test_kept_images_of_plain_image_folders_and_coco_splits` | REMOVE | 1 | 0.1 | 0.1 | Second end-to-end run; listing covered by test_images_of_every_manifest_format_and_of_plain_folders_are_listed, selection by test_representatives_and_baselines_are_kept. |
| `tests/deployment/test_deployment_utilities.py::test_requests_hold_the_inputs_of_the_task` (keep [sequence_comparison-expected_requests5]) | SHRINK to 1 | 6 | 0.1 | 0.1 | The per-image and whole-folder requests are simpler variants of the kept case. |
| `tests/deployment/test_export_specifications.py::test_inference_path_matches_the_training_path_on_full_sequences` (keep [sequence_labelling-transformer]) | SHRINK to 1 | 4 | 0.1 | 0.1 | Near-identical unmasked-path cases. |
| `tests/deployment/test_export_specifications.py::test_inference_comparator_matches_the_training_path_on_full_sequences` (keep [transformer-cross_attention]) | SHRINK to 1 | 2 | 0.1 | 0.0 | The gru-concatenation case runs the same per-member encoding path. |
| `tests/scripts/embedding/test_cluster_image_embeddings.py::test_mosaics_keep_the_ratio_of_the_images` | REMOVE | 1 | 0.0 | 0.0 | Visual layout of the mosaics; mosaics written in test_clustering_writes_clusters_links_mosaics_and_kept_images. |
| `tests/scripts/evaluation/test_evaluate_checkpoint.py::test_target_precisions_replace_the_experiment_one` (keep [classification/configuration_synthetic_shapes.yaml-target_precisions0]) | SHRINK to 1 | 4 | 0.0 | 0.0 | The single-precision branch is simpler than the kept list case. |
| `tests/scripts/evaluation/test_evaluate_checkpoint.py::test_invalid_target_precisions_are_refused` (keep [frame_selection/configuration_synthetic_sequences.yaml-target_precisions0-must be a single probability]) | SHRINK to 1 | 4 | 0.0 | 0.0 | Other cases are plain value checks. |
| `tests/scripts/embedding/test_cluster_image_embeddings.py::test_released_files_leave_the_page_cache` | REMOVE | 1 | 0.0 | 0.0 | Tests Linux page cache behaviour (memory optimisation), not results. |
| `tests/utilities/test_visualization.py::test_batch_mosaic_lays_out_the_blocks_under_their_header` | REMOVE | 1 | 0.0 | 0.0 | Input viewer mosaic layout. |
| `tests/scripts/data/test_visualize_augmentation_strengths.py::test_both_strengths_of_a_type_draw_the_same_random_numbers` | REMOVE | 1 | 0.0 | 0.0 | Explanatory figure script, low stake. |
| `tests/scripts/evaluation/test_evaluate_checkpoint.py::test_error_image_limit_turns_on_the_error_images` | REMOVE | 2 | 0.0 | 0.0 | Small CLI override of one setting, low stake. |
| `tests/deployment/test_deployment_utilities.py::test_sequence_folders_that_do_not_fit_are_refused` (keep [sequence_classification-5-...]) | SHRINK to 1 | 2 | 0.0 | 0.0 | Second refusal is a plain size check. |
| `tests/utilities/test_visualization.py::test_information_panel_matches_the_image_and_grows_with_its_lines` | REMOVE | 1 | 0.0 | 0.0 | Input viewer panel colours and height. |
| `tests/scripts/embedding/test_cluster_image_embeddings.py::test_vectors_are_normalised_and_reduced` | REMOVE | 1 | 0.0 | 0.0 | Normalisation and PCA run in test_clustering_writes_clusters_links_mosaics_and_kept_images; PCA itself is third party. |
| `tests/scripts/data/test_visualize_augmentation_strengths.py::test_type_figure_shows_the_image_then_both_strengths` | REMOVE | 1 | 0.0 | 0.0 | Figure layout, built by test_figures_are_written_for_every_type. |
| `tests/scripts/data/test_visualize_augmentation_strengths.py::test_noise_figure_adds_the_noise_of_every_level` | REMOVE | 1 | 0.0 | 0.0 | Explanatory figure, built by test_figures_are_written_for_every_type. |
| `tests/scripts/configuration_processing/test_convert_fixed_settings.py::test_files_are_converted_with_their_backups` | REMOVE | 1 | 0.0 | 0.0 | One-time migration, already applied. |
| `tests/deployment/test_triton_configuration.py::test_keypoint_triton_configuration_has_whole_images_and_candidates` | REMOVE | 2 | 0.0 | 0.0 | Same writer as test_comparison_triton_configuration_has_one_line_per_member, which also checks the PyTorch backend output names. |
| `tests/scripts/data/test_visualize_augmentation_strengths.py::test_crop_ratio_figure_frames_and_resizes_the_region_of_each_ratio` | REMOVE | 1 | 0.0 | 0.0 | Figure layout, built by test_figures_are_written_for_every_type. |
| `tests/scripts/evaluation/test_compare_best_evaluations.py::test_missing_curves_and_counts_fail_loudly` | REMOVE | 3 | 0.0 | 0.0 | Error wording of a comparison script. |
| `tests/scripts/embedding/test_cluster_image_embeddings.py::test_kept_images_table_gives_every_layer_and_category_with_their_total` | REMOVE | 1 | 0.0 | 0.0 | Console table formatting. |
| `tests/scripts/embedding/test_cluster_image_embeddings.py::test_data_sources_are_resolved_once_each` | REMOVE | 1 | 0.0 | 0.0 | Source resolution runs in test_clustering_writes_clusters_links_mosaics_and_kept_images. |
| `tests/deployment/test_export_model.py::test_thresholds_saved_for_another_checkpoint_are_not_found` | REMOVE | 4 | 0.0 | 0.0 | Not-found error path, low stake (the export stops anyway). |
| `tests/scripts/configuration_processing/test_convert_fixed_settings.py::test_conversion_lists_the_changed_values_and_refuses_other_changes` | REMOVE | 1 | 0.0 | 0.0 | One-time migration, already applied (2026-10-29, missed files 2026-11-03). |
| `tests/scripts/evaluation/test_compare_best_evaluations.py::test_older_records_are_counted_from_their_labels_files` | REMOVE | 1 | 0.0 | 0.0 | Legacy data.json format of old runs only. |
| `tests/scripts/evaluation/test_compare_best_evaluations.py::test_runs_of_the_same_columns_share_a_table` | REMOVE | 1 | 0.0 | 0.0 | Console table grouping. |
| `tests/deployment/test_export_model.py::test_saved_temperature_is_read_from_the_evaluation_of_the_checkpoint` (keep [evaluations/CLS_TEST_MC3/Evaluation/Iteration_6/EMA-True]) | SHRINK to 1 | 4 | 0.0 | 0.0 | Both folders are searched by the same path list, and test_thresholds_saved_in_both_folders_are_refused also uses them. |
| `tests/deployment/test_export_model.py::test_saved_thresholds_of_another_evaluation_or_model_are_refused` (keep [saved_thresholds2-2-3-the exported classifier predicts 3]) | SHRINK to 1 | 3 | 0.0 | 0.0 | Class names and temperature cases go through check_saved_thresholds, already covered by test_serve_temperature_explorer.py::test_invalid_saves_are_refused_without_writing. |
| `tests/deployment/test_triton_configuration.py::test_sequence_triton_configuration_has_dynamic_frames` | REMOVE | 1 | 0.0 | 0.0 | Dynamic frame dimensions covered by test_comparison_triton_configuration_has_one_line_per_member. |
| `tests/deployment/test_frame_selection_deployment.py::test_decoding_settings_read_the_thresholds_of_the_checkpoint` | REMOVE | 2 | 0.0 | 0.0 | Settings are used by test_python_backend_decodes_like_the_evaluation and test_decoder_and_ensembles_are_written. |
| `tests/scripts/embedding/test_cluster_image_embeddings.py::test_images_of_every_manifest_format_and_of_plain_folders_are_listed` | REMOVE | 1 | 0.0 | 0.0 | The classification manifest listing runs end to end in test_clustering_writes_clusters_links_mosaics_and_kept_images. |
| `tests/scripts/data/test_visualize_augmentation_strengths.py::test_example_images_come_from_different_trucks_and_categories` | REMOVE | 1 | 0.0 | 0.0 | Image choice of an explanatory figure script, run by test_figures_are_written_for_every_type. |
| `tests/scripts/evaluation/test_serve_temperature_explorer.py::test_folder_without_explorer_is_refused` | REMOVE | 1 | 0.0 | 0.0 | Trivial existence guard with an error wording check. |
| `tests/utilities/test_visualization.py::test_sample_grid_lays_out_four_panels` | REMOVE | 1 | 0.0 | 0.0 | Image size of a debug grid. |
| `tests/utilities/test_visualization.py::test_viewer_window_places_the_resized_image_next_to_its_panel` | REMOVE | 1 | 0.0 | 0.0 | Input viewer window layout. |
| `tests/utilities/test_os_utilities.py::test_expandable_segments_are_added_to_the_allocator_settings` (keep [max_split_size_mb:128-max_split_size_mb:128,expandable_segments:True]) | SHRINK to 1 | 3 | 0.0 | 0.0 | The kept case covers the add path; the empty case is simpler. |
| `tests/scripts/embedding/test_cluster_image_embeddings.py::test_clustering_settings_are_checked` | REMOVE | 1 | 0.0 | 0.0 | Settings validation of an analysis script, low stake. |
| `tests/utilities/test_mask_morphology.py::test_dilation_reaches_the_chessboard_neighbours` | REMOVE | 1 | 0.0 | 0.0 | Dilation is used by test_contour_bands_ignore_the_image_border_and_the_neutral_pixels. |
| `tests/utilities/test_os_utilities.py::test_path_link_opens_the_absolute_file_uri_under_a_short_label` | REMOVE | 1 | 0.0 | 0.0 | Console hyperlink formatting. |
| `tests/utilities/test_os_utilities.py::test_separators_are_as_long_as_the_visible_message` | REMOVE | 1 | 0.0 | 0.0 | Console separator width. |
| `tests/utilities/test_visualization.py::test_class_colours_are_distinct_with_a_black_background` | REMOVE | 1 | 0.0 | 0.0 | Trivial colour table; colours used by test_overlay_paints_the_classes_and_grays_the_ignored_pixels. |
| `tests/utilities/test_visualization.py::test_caption_bar_is_written_in_its_colour` | REMOVE | 1 | 0.0 | 0.0 | Text colour of a caption bar. |
| `tests/utilities/test_visualization.py::test_panel_lines_wrap_within_the_panel` | REMOVE | 1 | 0.0 | 0.0 | Text wrapping of the input viewer panel. |
| `tests/utilities/test_visualization.py::test_frame_grid_numbers_the_frames_from_the_given_number` | REMOVE | 1 | 0.0 | 0.0 | Frame numbering of a debug grid. |
| `tests/utilities/test_visualization.py::test_overlay_paints_the_classes_and_grays_the_ignored_pixels` | REMOVE | 1 | 0.0 | 0.0 | Colours of a debug overlay. |
| `tests/utilities/test_visualization.py::test_detection_grid_outlines_the_boxes_in_error` | REMOVE | 1 | 0.0 | 0.0 | Cosmetic outline of debug error images. |
| `tests/utilities/test_visualization.py::test_keypoint_grid_lays_out_the_true_and_predicted_objects` | REMOVE | 1 | 0.0 | 0.0 | Debug grid layout. Drawing is kept in test_keypoint_objects_draw_the_labelled_keypoints_by_side. |
| `tests/deployment/test_frame_selection_deployment.py::test_served_model_of_a_frame_selector_is_its_ensemble` | REMOVE | 1 | 0.0 | 0.0 | Trivial name mapping, also exercised by test_local_inference_decodes_the_selection. |
| `tests/scripts/data/test_visualize_augmentation_strengths.py::test_noise_columns_name_the_levels_of_both_strength_sets` | REMOVE | 1 | 0.0 | 0.0 | Column wording of a figure. |
| `tests/scripts/configuration_processing/test_convert_fixed_settings.py::test_fixed_settings_are_removed_with_their_comments` | REMOVE | 1 | 0.0 | 0.0 | One-time migration (done 2026-10-29); comment and blank-line handling is cosmetic, conversion exercised by the two kept tests. |
| `tests/scripts/configuration_processing/test_convert_fixed_settings.py::test_tasks_holding_a_setting_keep_it` | REMOVE | 1 | 0.0 | 0.0 | One-time migration; task-specific settings are part of the text checked by test_conversion_lists_the_changed_values_and_refuses_other_changes. |
| `tests/utilities/test_os_utilities.py::test_root_folder_link_shows_the_whole_path` | REMOVE | 1 | 0.0 | 0.0 | Edge case of console link wording. |
| `tests/utilities/test_visualization.py::test_keypoint_side_is_read_from_the_whole_words_of_its_name` | REMOVE | 7 | 0.0 | 0.0 | Side colour of debug drawings, low stake. |
| `tests/utilities/test_visualization.py::test_windows_are_centred_on_the_screen` | REMOVE | 1 | 0.0 | 0.0 | Input viewer window placement on screen. |
| `tests/utilities/test_visualization.py::test_error_map_colours_every_kind_of_pixel` | REMOVE | 1 | 0.0 | 0.0 | Colours of a debug error map. |

## Points To Check Before Removing

Before deleting anything, these points deserve a look:

- **Kept parameter cases chosen by identifier**: for `tests/data/test_data_record.py::test_classes_counted_in_every_manifest_format` and other parametrizations whose case identifiers are opaque (`labels0`, `labels4`, ...), the kept cases were picked without seeing which format each covers; check them in the parametrize list.
- **Shared module fixtures**: removing the first test of a module that builds a run or a dataset in a module-scoped fixture moves that cost to the next test of the module, so the real saving can be smaller than shown (noted in `tests/scripts/embedding/test_cluster_image_embeddings.py`).
- **Saved time of the SHRINK rows** is estimated in proportion to the number of cases removed, not measured case by case.
- **A removal is final only after the full suite still passes** and the remaining tests are checked to still cover the behaviours named in the Reason column.
