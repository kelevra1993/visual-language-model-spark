# Configuration Reference

Keys, values and outputs of the configurations, completing the [Configurations](../../README.md#configurations), [Run Names](../../README.md#run-names) and [MLflow Tracking](../../README.md#mlflow-tracking) sections of the README. Every key is also explained by its comment in the configuration files of `vision_training/configurations/<task>/`.

## Table Of Content
- [Configurations](#configurations)
  - [Keys Per Task](#keys-per-task)
  - [Values Per Task](#values-per-task)
  - [Fixed Settings](#fixed-settings)
  - [Checking Without Training](#checking-without-training)
  - [Evaluation Outputs](#evaluation-outputs)
- [Run Names](#run-names)
- [MLflow Tracking](#mlflow-tracking)

## Configurations
Every task has a `configuration_example.yaml` and a `configuration_synthetic_*.yaml` with the same keys. A configuration file holds only the keys of its task, all required: the check stops at the first missing key, naming its path. The settings fixed in the code are refused ([Fixed Settings](#fixed-settings)), then every key the code of the task does not read, whose value would otherwise be silently ignored: a misspelled key, a setting no code reads any longer, or a section or key of another task ([Keys Per Task](#keys-per-task)).

The tasks are written with their codes: DET `object_detection`, INS `instance_segmentation`, CLS `classification`, M2O `sequence_classification`, M2M `sequence_labelling`, CMP `sequence_comparison`, SEG `semantic_segmentation`, SEL `frame_selection`, SSL `self_supervised`, KPT `keypoint_detection`.

### Keys Per Task
Every task has `ExperimentConfiguration` (with `evaluation.per_class_report`, `evaluation.number_sample_images` and `tracking`), `Optimization`, `Data` and `Backbone`. The other keys belong to some tasks only, `evaluation` standing for `ExperimentConfiguration.evaluation`:

| Key | Tasks |
|---|---|
| `DetectionLevels`, `Neck`, `RegionProposal`, `Detector`, `Optimization.loss_weights`, `evaluation.score_threshold` | DET, INS |
| `MaskHead`, `Data.instance_index_maps` | INS |
| `evaluation.original_resolution` | INS, SEG |
| `evaluation.target_precision`, `evaluation.error_dumps` | DET, INS, CLS, M2O, M2M, CMP, SEL (`target_precision` a list of precisions, the first one the main target, for CLS, M2O, M2M, CMP) |
| `ClassificationHead`, `evaluation.save_predictions` | CLS, M2O, M2M, CMP, SEL |
| `ClassificationHead.decorrelation` | CLS |
| `TemporalModel`, `Data.sequence_settings`, `Data.data_augmentation.independent_frames` and `shuffle_frames` | M2O, M2M, CMP, SEL |
| `Fusion` | CMP |
| `SegmentationDecoder`, `SegmentationLoss`, `Data.tiling`, `evaluation.save_predicted_masks`, `evaluation.boundary_tolerance` | SEG |
| `Selection`, `evaluation.top_k` | SEL |
| `ProjectionHead`, `Data.grouped_batches`, `evaluation.number_neighbours`, `evaluation.number_reference_images_per_class` | SSL |
| `HRNet`, `KeypointHead`, `evaluation.default_keypoint_sigma`, `evaluation.keypoint_sigmas` | KPT |

In case a file holds a key the code of its task does not read, a misspelled key or a section copied from another task, the loading refuses it after the fixed settings and before the other checks, every such key listed by its path in one error. The keys of every task, the common sections and those of this table, are listed one by one in `KNOWN_KEY_PATHS` of `vision_training/configuration_processing/configuration_keys.py` (the keys read by the loader and by the checks of every task). A misspelled key also leaves its correct key missing, which the next launch then reports.
- A key no code reads: `'ExperimentConfiguration.bach_size' is not a configuration key (did you mean 'batch_size'?): no code reads it, so its value would be ignored. Correct it or delete it from the configuration file.`, the closest key read at the same place being named when one is close.
- A key of another task: `'RegionProposal' is not read by the task 'classification', only by ['instance_segmentation', 'object_detection']: its value would be ignored. Delete it from the configuration file.`, a whole section being named once.
- The keys of the sections keyed by names are left to the checks of their task: the blocks of `Backbone.convolutions` and `SegmentationDecoder.convolutions`, the levels of `DetectionLevels` (whose own keys are checked), the losses of `Optimization.loss_weights`, the levels of `SegmentationDecoder.deep_supervision_weights`, and the categories and keypoints of `evaluation.keypoint_sigmas`.

### Values Per Task
The check of a task refuses a value it cannot use. Every key not listed accepts the same values in every task.

**All tasks**
- `ExperimentConfiguration.information_dump`, `weight_saving_iterations`, `intermediate_weight_saving_iterations`, `number_iterations`, `Optimization.warmup_iterations` and `decay_iterations`: integers, also written in scientific notation when the value is integral (`1e10`, which YAML reads as a text, or `4.0e3`); a non-integer value such as `31.7` is refused with its key, not truncated.
- `Data.image_settings.size`: a positive even integer, multiple of 2 to the power of the number of max poolings of `Backbone` (every block but the last, and the last with `last_max_pooling: True`), so that no pooling drops a row or a column. With 5 poolings, 96, 224, 384 and 512 are accepted and 100 is refused. The output of the last pooling may be odd (7x7 for 224).
- `Data.label_mapping.mode`: `binary` or `multi_class` for DET, INS and SEG, `multi_class` for the others.
- `Data.data_augmentation.activated`: `True` for SSL, whose views need it.
- `Data.data_augmentation.types.random_perspective` and `random_jigsaw`, which move the image without its targets: off for DET, INS, SEG and KPT.
- `Data.data_augmentation.types.random_crop.probability`: `1.0` when activated for SSL.
- `Backbone.normalization.feature_map_normalization`: `sequence` only for M2O, M2M, CMP and SEL.

**Classification heads and temporal modules**
- `ClassificationHead.global_pooling`: `average`, `max` or `flatten`.
- `ClassificationHead.output_mode`: `multi_label` for CLS only.
- `TemporalModel.type`: `mean_pooling`, `max_pooling`, `attention_pooling`, `gru`, `lstm`, `transformer` or `temporal_convolution` for M2O and CMP; the last four only for M2M and SEL, which need one output per frame.
- `TemporalModel.causal`: `True` for M2M and SEL only. In these tasks a causal module (`causal: True`, or a GRU or LSTM that is not bidirectional) refuses `Backbone.normalization.feature_map_normalization: "sequence"`, whose statistics come from the whole sequence.

**Task rules**
- DET: every level of `DetectionLevels` holds exactly `anchors`, `region_proposal` and `detector`. INS adds `mask_head`, whose `roi_align_pool_size` must not exceed `MaskHead.mask_resolution`, and `Data.instance_index_maps` (`True` or `False`): one uint8 map per image numbering its objects instead of one mask per object, expanded into the masks on the GPU; it needs run length encodings whose masks never overlap and at most 255 objects per image, an image breaking this stopping the loading (`True` in the synthetic configuration, `False` for COCO). The finer inputs of the mask heads, all off by default: `MaskHead.fusion_convolutions` and `MaskHead.image_branch_convolutions` are `[layers, channels]` of positive integers, or `[0, 0]` when unused; `MaskHead.skip_blocks_at_pool_size` and `skip_blocks_at_mask_resolution` list distinct backbone blocks from 0 (the normalized input image) to the last block, read before their pooling, block b at stride 2 ** (b - 1): every skip must be finer than the finest detection level (block 4 at most with levels 4 to 7), every skip at the mask resolution finer than every skip at the pool size, and no block listed in both; the image branch must be `[0, 0]` when `MaskHead.pretrained.experiment_folder` is set, the branch not being transferred with the mask heads. `MaskHead.shared` is `True` or `False`; `True` (one mask head read by every level) needs the same `mask_head` on every level and the same channels on every level, which a neck gives and the backbone blocks of the levels must give without one. `MaskHead.mask_scoring.activated` is `True` or `False`, `mask_scoring.convolutions` a `[layers, channels]` of positive integers and `mask_scoring.fully_connected` a non-empty list of positive integers, checked even when off; `True` needs `MaskHead.mask_resolution` to be a multiple of the `roi_align_pool_size` of every level and a positive `Optimization.loss_weights.mask_iou_loss`, a weight refused while the scoring is off. `Neck.type` is `none`, `feature_pyramid` or `path_aggregation`, `Neck.output_channels` a positive integer, and with a neck every level must be a whole multiple of the next coarser one (checked when the model is built). `Detector.box_loss` is `smooth_l1` or `complete_iou` (`BOX_LOSSES` of `vision_training/losses/detection_losses.py`), the region proposal network always using the smooth L1. `Detector.nms_criterion` is `iou` or `distance_iou` (`NMS_CRITERIA` of `vision_training/models/detection/postprocessing.py`) and `Detector.nms_distance_exponent` a finite positive number; neither enters the run name, the final NMS running in evaluation mode only. `RegionProposal.proposal_selection` is `joint` or `per_level` (`PROPOSAL_SELECTIONS` of `vision_training/models/detection/region_proposer.py`); `per_level` makes `pre_nms_proposals` a count per level and adds the `RPL` token.
- SEG: the keys of `SegmentationDecoder.convolutions` are the backbone blocks whose output is a skip connection, blocks 1 to N-1 of N blocks (and N with `last_max_pooling: True`). The keys of `deep_supervision_weights` are keys of `convolutions` other than the finest level. The categories of a `multi_class` mapping are the mask values 1 to 254.
- KPT: the classes of the label mapping are indexed 0 (background) to K-1 without gaps. `Backbone` ends at 1/4 of the image (3 blocks, or 2 with `last_max_pooling: True`), and `Data.image_settings.size` is a multiple of 4 x 2 to the power of the length of `HRNet.modules_per_stage`. `hidden_width` and the tag loss weights are at least 0, `inference.maximum_candidates` at least 1, `inference.keypoint_threshold` within ]0, 1[, `inference.tag_threshold` positive and `inference.object_score_threshold` within [0, 1]. The categories and keypoints of `evaluation.keypoint_sigmas` must exist in the training split.

### Fixed Settings
These settings were removed from the configuration files, the code always using the behaviour below. A file still holding one is refused before any other check, with a message naming what replaced it (`REMOVED_SETTINGS` of `vision_training/configuration_processing/common_checks.py`): `'<key>' was removed: <what replaced it>. Delete it from the configuration file.`

| Removed setting | Fixed behaviour |
|---|---|
| `ExperimentConfiguration.resume_training` | A run always resumes from the last checkpoint of its folder. To train the same settings from scratch, set another `run_name_suffix` or `experiment_folder`, or move the run folder away. |
| `ExperimentConfiguration.evaluation.top_k` | The top-k accuracy counts the 5 most probable classes (`TOP_K_ACCURACY_CLASSES`). SEL keeps the key, the number of candidate frames of every class. |
| `ExperimentConfiguration.tracking.log_artifacts` | A tracked run always sends its files as MLflow artifacts. |
| `ExperimentConfiguration.diagnosis_modes.compute_model_size` | The model size is always printed at launch. |
| `ExperimentConfiguration.diagnosis_modes.verbose` | Removed. |
| `Optimization.mixed_precision` | Mixed precision is always on: the training and validation forward passes run in bfloat16, the weights and the test evaluation in float32. |
| `Optimization.gradient_clipping_norm` | The gradients are never clipped: the optimizer steps with the gradients of the backward pass as they are. Every saved run had it at `null`. |
| `Data.loading` (`number_workers`, `prefetch_batches`) | Every loader (training, validation and test) uses 1 worker process (`NUMBER_LOADER_WORKERS` of `vision_training/data/loaders.py`), preparing 2 batches in advance (`PREFETCH_BATCHES`). The validation loader is built only when `validation_iteration_ratio` is above 0 (or for `view_input_images`, `compute_anchor_coverage` and `benchmark_speed`), so that a training without validation holds no validation dataset, worker or batch: about 2.3 GB in a 1024 pixel instance segmentation training of batch 64 (2026-11-13: 1.14 GB private to the idle worker, 0.68 GB for the parsed manifest of the 50,000 validation images, 0.31 GB since the dataset packs it, 0.50 GB for the 2 batches prepared in advance). 1 keeps the memory down at no cost in speed: a 1024 pixel detection pair trained at 2.10 and 2.36 seconds per iteration with 1 worker against 2.18 and 2.42 with 2 (2026-11-05), the 12 workers of the three loaders of such a training having held about 26 GB with 4, about 5 GB per training loader worker. On the ImageNet classifier of blocks 1x32.2x32.2x32.16x32.20x32.20x32 (the small model before 2026-11-21), 2, 4, 6 and 8 workers gave 455, 471, 471 and 456 images per second. |
| `Backbone.modules.inception_starting_block` | The inception modules start at block 3 (`INCEPTION_STARTING_BLOCK`). |
| `batch_normalization_momentum` of `Backbone.normalization`, `ClassificationHead`, `RegionProposal.normalization`, `Detector.normalization`, `MaskHead.normalization`, `SegmentationDecoder`, `HRNet.normalization` and `ProjectionHead` | 0.1 (`BATCH_NORMALIZATION_MOMENTUM`) for every batch normalization. |
| `RegionProposal.strict_fallback_assignment` | The anchor made foreground for an object it does not overlap the most targets the object it overlaps the most, as the region proposal network of torchvision. |
| `RegionProposal.localisation_loss_beta`, `Detector.localisation_loss_beta` | The smooth L1 box losses of the region proposal network and of the detection heads (`Detector.box_loss: "smooth_l1"`) use a beta of 1/9 (`LOCALISATION_LOSS_BETA` of `vision_training/losses/detection_losses.py`). |
| `RegionProposal.box_loss` | The region proposal network always uses the smooth L1 box loss on the offsets (`REGION_PROPOSAL_BOX_LOSS` of `vision_training/losses/detection_losses.py`), the CIoU box loss being an option of the detection heads only (`Detector.box_loss`). |
| `RegionProposal.anchor_centres` | The anchors are centred on the cell corners (column x stride, as the `AnchorGenerator` of torchvision), the cell centres having tied with them in stage 8 of the detection benchmark. |
| `Data.data_augmentation.independent_frames`, `shuffle_frames` | Kept in M2O, M2M, CMP and SEL only, off for the other tasks. |

The sections `Data.view_augmentation` and `ClassificationHead.view_losses`, removed earlier, are refused the same way, and so is `SegmentationDecoder.pretrained.load_class_layers`, renamed `class_layers` like the flags of the detection, mask and keypoint heads.

`scripts/configuration_processing/convert_fixed_settings.py` removes the fixed settings from the files written before, the `training_configuration.yaml` of the run folders and the launch files set in its main block, keeping their comments. It also adds what a task now requires with the values its runs trained with: the `Neck` section (`type: "none"`), the `box_loss` (`"smooth_l1"`) of `Detector`, the `nms_criterion` (`"iou"`) and `nms_distance_exponent` (`1.0`) of `Detector`, the `proposal_selection` (`"joint"`) of `RegionProposal` in the DET and INS files, the `instance_index_maps` (`False`, one mask per object) of `Data` in the INS files, the finer inputs of their `MaskHead` after its `upsampling` (`skip_blocks_at_pool_size: []`, `image_branch_convolutions: [0, 0]`, `skip_blocks_at_mask_resolution: []`, `fusion_convolutions: [0, 0]`, none), the `shared` of their `MaskHead` after its `fusion_convolutions` (`False`, one mask head per level) and its `mask_scoring` section after it (off), the `benchmark_speed` (`False`) of `ExperimentConfiguration.diagnosis_modes` in the files of every task, the `class_layer` of `ClassificationHead.pretrained` after its `freeze` in the CLS, M2O, M2M, CMP and SEL files, and the `class_layers` of `Detector.pretrained` (DET, INS), `MaskHead.pretrained` (INS) and `KeypointHead.pretrained` (KPT) after their `freeze`, `True` for heads transferred from another run, which loaded their class layers, and `False` otherwise, so that the run names stay (`ADDED_SECTIONS`, `ADDED_SETTINGS`). It renames `SegmentationDecoder.pretrained.load_class_layers` to `class_layers` in the SEG files, with its value and the mentions in the comments of the section (`RENAMED_SETTINGS`):
- Every converted file is first copied to a backup named with the `backup_suffix` of the main block, one per conversion (`_before_fixed_settings`, `_before_fixed_head_settings`, `_before_neck`, `_before_box_loss`, `_before_nms_criterion`, `_before_region_proposal_box_loss`, `_before_gradient_clipping_norm`, `_before_proposal_selection`, `_before_anchor_centres`, `_before_anchor_centres_removal`, `_before_instance_index_maps`, `_before_mask_fine_inputs`, `_before_benchmark_speed`, `_before_shared_mask_head`, `_before_mask_scoring`, `_before_class_layer`, `_before_class_layers`): `training_configuration<suffix>.yaml` next to a run configuration, `previous_configurations/<subfolder>/<name><suffix>.yaml` for a launch file. The copies of MLflow (`mlartifacts`) are left as logged.
- A file holding another value than the fixed one is listed with the values its runs change to (e.g. `ExperimentConfiguration.resume_training False -> True`), and a backbone whose inception modules start at another block than 3 is refused, the code no longer building its layers.
- A file holding another value than the fixed one of a setting the code no longer trains (`RECORD_ONLY_SETTINGS`: `RegionProposal.box_loss` `"complete_iou"`, run_13 to run_16 of the detection benchmark, an `Optimization.gradient_clipping_norm` other than `null`, held by no saved run, and `RegionProposal.anchor_centres` `"cell_centres"`, run_23 and run_24) is reported and left as it is, its runs staying as records, the other files being converted.
- `write_files = False` only lists the files to convert.

### Checking Without Training
Whenever a pipeline has to decide whether a configuration file can start a training, `vision_training/check_configuration.py` runs the loading and the checks of a training (`prepare_experiment_configuration` of `vision_training/configuration_processing/configuration_loader.py`) on every file given, without building a model or a loader:

```bash
uv run python -m vision_training.check_configuration --configuration_path=<file> [--configuration_path=<file> ...] [--check_paths]
```

- Every file is checked, the next ones too after an invalid one. An invalid file prints the red block of the trainings, `Invalid Configuration File : <file>` and one line per problem; a valid one prints `Valid Configuration File : <file> | Run Name : <run name>`, the folder it would train in inside `experiment_folder` ([Run Names](#run-names)).
- The exit code is 0 when every file is valid, 1 otherwise, and 2 for a malformed command line (no `--configuration_path`, an unknown option).
- Without `--check_paths`, the check reads what the loading reads: the label mapping file of the `multi_class` mode (or the `label_dictionary.json` next to the configuration of a run folder) and, for KPT, the `labels.json` of the training split. A file the loading cannot read is reported with the name of its error, e.g. `FileNotFoundError: Label mapping file not found at: <path>` (also a missing or unparsable file, no `ExperimentConfiguration`, a null folder or learning rate); a null count is refused by the checks with its key.
- `--check_paths` then requires, once the settings pass the checks, the paths a new training reads: the three split folders with their `labels.json` and `data/` folder (and `masks/` for SEG), the mapping file of the `multi_class` mode, and for every section whose `pretrained.experiment_folder` is set, that run folder, its `training_configuration.yaml` and the checkpoint of `pretrained.iteration` (null: its last checkpoint, found as the transfer finds it). `experiment_folder` may not exist yet, the training creating it. A run resuming from its own checkpoints does not load its pretrained runs again, but the check requires them all.

### Evaluation Outputs
Every evaluated checkpoint is reported in the console and in `<run>/Results.txt`, headed `Iteration <n> · <evaluated samples> · <date and time>`, with its global metrics, notes and tables, the per-class table and the per-class recall table sorted worst first. Its files go to `<run>/Evaluation/Iteration_<n>/`, and those of the moving average to its `EMA/` subfolder; `scripts/evaluation/evaluate_checkpoint.py` writes its `Results.txt` and iteration folders in `<run>/Checkpoint_Evaluation/` instead. Every task writes the rates of the report as percentages (`94.03 %`, a target precision `99 %`, a confidence level `95 %`), the thresholds, ranks, counts, pixel errors, effective rank and standard deviations keeping their values, and the files below keep every rate between 0 and 1. `ExperimentConfiguration.evaluation.per_class_report: False` keeps the global metrics only, for runs with many classes, the files keeping the details.

| Output | Tasks | Written |
|---|---|---|
| Global metrics and notes, global Test scalars | all | every evaluation |
| Per-class table and per-class Test scalars | all | with `per_class_report` |
| `Per Class Recall` table: the recall of every class with its 95 % confidence interval, worst recall first (`Per Class k-NN Recall` for SSL, `Mask Per Class Recall` and `Box Per Class Recall` for INS, `OKS Per Class Recall` for KPT, the selection accuracy of `Selection.decoding` for SEL) | all, SSL with labelled test images | with `per_class_report` |
| `Macro ROC AUC`, `Macro Average Precision`, and the ROC AUC and average precision of every class | CLS, M2O, M2M, CMP, SEG | with `per_class_report` |
| `class_metrics.csv`, one row per class (table below) | all, SSL with labelled test images | every evaluation |
| `keypoint_metrics.csv`: labelled keypoints, PCK and mean error of every keypoint | KPT | every evaluation |
| OKS size and recall tables; `Keypoint Accuracy Per Keypoint` table | KPT | every evaluation; the keypoint table with `per_class_report` |
| `Mean Average Precision By Object Size (Area)`, `Mean Average Recall` and `Proposal Recall` tables, `proposal_recall.csv` | DET, INS | every evaluation |
| `class_thresholds.json` | DET, INS, CLS, M2O, M2M, CMP, SEL | every evaluation |
| `confusion_matrix.csv` | DET, INS, CLS multi-class, M2O, M2M, CMP, SEG, SEL | every evaluation, up to 1,000 classes for CLS, M2O, M2M, CMP |
| `confusion_cells.csv`: the non-zero cells of the confusion matrix, one row `true_class,predicted_class,count` each, sorted by true then predicted class | CLS multi-class, M2O, M2M, CMP above 1,000 classes (`MAXIMUM_DENSE_CONFUSION_CLASSES`) | every evaluation |
| `confusion_matrix.png` and its Test image | the tasks of `confusion_matrix.csv` | with `per_class_report`, never with `confusion_cells.csv` |
| `roc_curves.png`, `precision_recall_curves.png` and their Test images `ROC Curves` and `Precision Recall Curves` | CLS, M2O, M2M, CMP, SEG | with `per_class_report` |
| `precision_recall_thresholds.png` and its Test image `Precision Recall Thresholds`, the precision-recall curves zoomed on the thresholds of every target precision, each threshold written in the colour of its curve | CLS, M2O, M2M, CMP | with `per_class_report` |
| `predictions.csv` | CLS, M2O, M2M, CMP | with `save_predictions` |
| `selections.csv` | SEL | with `save_predictions` |
| `Errors/<TRUE>-TO-<PREDICTED>/` | CLS multi-class, M2O, M2M, CMP | with `error_dumps.activated` |
| `Errors/<CLASS>/<sequence>.png` | SEL | with `error_dumps.activated` |
| `Errors/FALSE-POSITIVE/<class>/`, `Errors/MISSED/<class>/`, `Errors/<TRUE>-TO-<PREDICTED>/` | DET, INS | with `error_dumps.activated` |
| `Samples/<sample>.png` and its Test image `Samples/<sample>` (table below) | all | with `number_sample_images` above 0 |
| `Predicted_Masks/<image>.png` | SEG | with `save_predicted_masks` |
| `evaluation_summary.csv` in `Evaluation/` (`Checkpoint_Evaluation/` for `scripts/evaluation/evaluate_checkpoint.py`), one row per evaluated iteration with every scalar, `ema_` columns for the averaged weights with their number of updates `ema_number_updates`, and `evaluation_curves.png` next to `Results.txt`, the headline metric of every evaluation ([Run Folder](training_reference.md#run-folder)) | all | every evaluation; per-class columns with `per_class_report` |

| Tasks | Columns of `class_metrics.csv` |
|---|---|
| CLS, M2O, M2M, CMP | samples, precision, recall with its 95 % confidence interval (`recall_lower`, `recall_upper`), F1, ROC AUC and average precision (with `per_class_report`), threshold and recall at the target precision (`threshold_at_target_precision`, `recall_at_target_precision`), then at every other target precision of a list (`threshold_at_precision_<p>`, `recall_at_precision_<p>`) |
| SEL | true frames, accuracy with both decodings, that of `Selection.decoding` followed by its 95 % confidence interval (`<decoding>_accuracy_lower`, `<decoding>_accuracy_upper`), top-k accuracy, mean rank, threshold and recall at the target precision |
| DET, INS | ground truths, average precisions of all the objects and of every size, recall of all the detections at an overlap of 0.5 and 0.75 with its 95 % confidence interval (`recall_50`, `recall_50_lower`, `recall_50_upper`, `recall_75`, `recall_75_lower`, `recall_75_upper`), both also of the masks for INS (prefixed `mask_`), threshold and recall at the target precision |
| SEG | pixels, IoU, Dice, boundary IoU, boundary F1, precision, recall with its 95 % confidence interval (`recall_lower`, `recall_upper`), ROC AUC and average precision (with `per_class_report`) |
| SSL | labelled test images, k-NN precision, recall with its 95 % confidence interval (`recall_lower`, `recall_upper`) and F1 |
| KPT | objects, OKS average precisions, OKS recall at 0.5 and 0.75 with its 95 % confidence interval (`oks_recall_50` to `oks_recall_75_upper`), matched objects, labelled keypoints, PCK, mean OKS, mean error |

The bounds of the recall intervals are empty for a class without true samples. They come from `vision_training/evaluation/recall_intervals.py`: an exact (Clopper-Pearson) interval when the counted samples are independent (the images of CLS and SSL, the sequences of M2O, the pairs of CMP, the ground truth objects of DET, INS and KPT), and a Korn-Graubard interval over the test sequences for M2M and SEL (the frames of a sequence being correlated) and over the test images for SEG (the pixels of an image being correlated): the exact interval at the effective sample size given by the variance of the recall over 1000 bootstrap resamples of the sequences or images (fixed seed), reduced for the degrees of freedom of those holding the class, or the exact interval over the sequences or images holding the class when the recall is 0 % or 100 % or only one of them holds the class.

| Tasks | Sample image |
|---|---|
| CLS, M2O, CMP | the input (frame grid for a sequence, before member above after member for CMP), captioned `true -> predicted probability`, green when right and red when wrong |
| M2M | the frame grid, every frame captioned `true/predicted`, `X` on the wrong ones |
| SEL | the thumbnail grid with the borders of the selection, and the probability heat map |
| DET, INS | ground truth boxes next to the detections above `Detector.inference.score_threshold`, with the masks for INS |
| SEG | input, true mask, predicted mask and error map |
| SSL | the image next to two random views, with the cosine similarity of their representations |
| KPT | ground truth objects next to the objects found scoring at least `KeypointHead.inference.object_score_threshold`, with their skeletons |

The sample images are chosen once from the test split, so every evaluation shows the same samples, even after a resume. CLS, M2O, M2M, CMP, SSL and KPT spread them over the classes first, DET, INS and SEG over the images, and SEL over the sequences.

## Run Names
`vision_training/configuration_processing/run_name.py` builds the run name from the configuration, so the same settings resume in the same folder and changed settings start a new one.
- The sections are joined with `_` and their tokens with `-`, in the order of the table below. `<...>` is read from the configuration.
- A flag token is left out when its setting is off. The optimizer section is left out for plain Adam without weight decay, and the suffix section when `run_name_suffix` is null.
- A setting identical on every detection level is written once; otherwise the levels are joined with `+`.
- The class counts give the foreground classes of the label dictionary, merged categories counting once.

**Settings without a token**, which may change between two resumptions of a run (two runs differing only by them share a folder, unless one has a `run_name_suffix`):
- the augmentation (with `shuffle_frames` and `independent_frames`), batch size, learning rate, warm-up and cosine schedule, `pretrained.learning_rate_multiplier`, `Optimization.exponential_moving_average`, the `batch_normalization_momentum` of the sections holding one, and the dropout rates (`ClassificationHead.dropout_rate`, `TemporalModel.dropout_rate`, `Fusion.dropout_rate`, `SegmentationDecoder.dropout_rate`);
- `Optimization.loss_weights`, `Detector.nms_criterion` and `nms_distance_exponent`, and the IoU thresholds, sample and proposal counts, NMS thresholds and `inference` settings of `RegionProposal` and `Detector` (DET, INS), `ClassificationHead.decorrelation` (CLS), `SegmentationLoss`, the deep supervision weights and `Data.tiling.overlap` (SEG), `Selection.decoding` (SEL), `Data.grouped_batches` (SSL), `MaskHead.inference.mask_threshold` (INS), and `KeypointHead.heatmap_sigma`, `ignore_unlabelled_objects`, `tag_pull_weight`, `tag_push_weight` and `inference` (KPT);
- the evaluation settings.

The optimizer keeps its token, as the state of one optimizer cannot be resumed by another.

| Section | Token | Setting | Tasks |
|---|---|---|---|
| Task | `DET`, `INS`, `CLS`, `M2O`, `M2M`, `CMP`, `SEG`, `SEL`, `SSL`, `KPT` | `ExperimentConfiguration.task` | all |
| Input | `IN` | `Backbone.normalization.input_normalization` | all |
| Input | `PSN` | `Backbone.normalization.per_sample_normalization` | all |
| Input | `KAR` | `Data.image_settings.keep_ratio` | all |
| Input | `<size>` | `Data.image_settings.size` | all |
| Backbone | `SE`, `RES`, or `RES2` for pre-activation shortcuts | `Backbone.modules.squeeze_and_excite`, `residual_nets`, `residual_nets_v2` | all |
| Backbone | `INC3` | `Backbone.modules.inception`, the inception modules starting at block 3 | all |
| Backbone | `BN`, `LN`, `NN`, `SN` | `Backbone.normalization.feature_map_normalization`: batch, layer, none, sequence | all (`SN`: M2O, M2M, CMP, SEL) |
| Backbone | `B<layers>x<filters>.…` | `Backbone.convolutions`, one per block | all |
| Backbone | `LMP` | `Backbone.last_max_pooling` | all |
| Backbone | `PT`, `FRZ`, `FBN` | `Backbone.pretrained`: set, `freeze`, `freeze_normalization_parameters` | all |
| Levels | `L<blocks>` | the keys of `DetectionLevels` | DET, INS |
| Levels | `NFPN<channels><BN, LN or NN>`, `NPAN…` | `Neck.type` (no token for `none`), `output_channels`, `normalization` | DET, INS |
| Anchors | `A<anchors>.<smallest>-<largest>.<hash>` | anchors per position, range of the scales, first 6 characters of the SHA-256 of the scales and ratios | DET, INS |
| Region proposal | `RBN`, `RLN`, `RNN` | `RegionProposal.normalization.feature_map_normalization` | DET, INS |
| Region proposal | `R<layers>x<filters>` | `DetectionLevels.<level>.region_proposal.convolutions` | DET, INS |
| Region proposal | `RPL` | `RegionProposal.proposal_selection` `per_level` (no token for `joint`) | DET, INS |
| Region proposal | `RPT`, `RFRZ` | `RegionProposal.pretrained`, frozen | DET, INS |
| Detector | `DSE`, `DRES`, `DINC` | `Detector.modules` | DET, INS |
| Detector | `DBN`, `DLN`, `DNN` | `Detector.normalization.feature_map_normalization` | DET, INS |
| Detector | `D<pool>.<layers>x<filters>.<fully connected>` | `DetectionLevels.<level>.detector` | DET, INS |
| Detector | `DCIOU` | `Detector.box_loss` `complete_iou` (no token for `smooth_l1`) | DET, INS |
| Detector | `DPT`, `DRCL`, `DFRZ` | `Detector.pretrained`, loaded without their class layers (`class_layers` False), frozen | DET, INS |
| Mask heads | `MSE`, `MRES`, `MINC` | `MaskHead.modules` | INS |
| Mask heads | `MBN`, `MLN`, `MNN` | `MaskHead.normalization.feature_map_normalization` | INS |
| Mask heads | `MUTC`, `MUBL` | `MaskHead.upsampling`: transposed, bilinear | INS |
| Mask heads | `M<pool>.<layers>x<channels>` | `DetectionLevels.<level>.mask_head` | INS |
| Mask heads | `MSH` | `MaskHead.shared`, one mask head read by every level (no token for `False`) | INS |
| Mask heads | `MR<resolution>` | `MaskHead.mask_resolution` | INS |
| Mask heads | `MSP<blocks>` | `MaskHead.skip_blocks_at_pool_size`, blocks joined with `.` (no token for `[]`) | INS |
| Mask heads | `MIB<layers>x<channels>` | `MaskHead.image_branch_convolutions` (no token for `[0, 0]`) | INS |
| Mask heads | `MSM<blocks>` | `MaskHead.skip_blocks_at_mask_resolution`, blocks joined with `.` (no token for `[]`) | INS |
| Mask heads | `MF<layers>x<channels>` | `MaskHead.fusion_convolutions` (no token for `[0, 0]`) | INS |
| Mask heads | `MIOU<layers>x<channels>.<fully connected>` | `MaskHead.mask_scoring`, the mask IoU heads of Mask Scoring R-CNN (no token when not activated) | INS |
| Mask heads | `MPT`, `MRCL`, `MFRZ` | `MaskHead.pretrained`, loaded without their class layers (`class_layers` False), frozen | INS |
| Tiling | `TL<scale>` | `Data.tiling.activated`, `Data.tiling.scale` | SEG |
| Decoder | `UTC`, `UBL` | `SegmentationDecoder.upsampling`: transposed, bilinear | SEG |
| Decoder | `DSE`, `DRES` | `SegmentationDecoder.modules` | SEG |
| Decoder | `DBN`, `DLN`, `DNN` | `SegmentationDecoder.feature_map_normalization` | SEG |
| Decoder | `D<layers>x<channels>.…` | `SegmentationDecoder.convolutions`, deepest level first | SEG |
| Decoder | `AUX<level>.…` | the levels of `SegmentationDecoder.deep_supervision_weights`, deepest first | SEG |
| Decoder | `DPT`, `DCL`, `DFRZ` | `SegmentationDecoder.pretrained`, loaded with its class layers (`class_layers` True), frozen | SEG |
| Classes | `BIN` or `MC<classes>` | `Data.label_mapping.mode` | DET, INS, SEG |
| Sequence length | `T<frames>` | `Data.sequence_settings.maximum_sequence_length` (of a member for CMP) | M2O, M2M, CMP, SEL |
| Temporal module | `MEAN`, `MAX`, `ATT<hidden size>` | `TemporalModel.type`: mean, max, attention pooling | M2O, CMP |
| Temporal module | `GRU<layers>x<hidden size>`, `LSTM<layers>x<hidden size>` | `TemporalModel.type`: gru, lstm | M2O, M2M, CMP, SEL |
| Temporal module | `TRF<layers>x<hidden size>.<heads>` | `TemporalModel.type`: transformer | M2O, M2M, CMP, SEL |
| Temporal module | `TCN<layers>x<hidden size>.<kernel size>` | `TemporalModel.type`: temporal_convolution | M2O, M2M, CMP, SEL |
| Temporal module | `BI` | `TemporalModel.bidirectional` (gru, lstm) | M2O, M2M, CMP, SEL |
| Temporal module | `CAU` | `TemporalModel.causal` (transformer, temporal_convolution) | M2M, SEL |
| Temporal module | `NPE` | `TemporalModel.positional_embedding` is False (transformer) | M2O, M2M, CMP, SEL |
| Temporal module | `TPT`, `TFRZ` | `TemporalModel.pretrained`, frozen | M2O, M2M, CMP, SEL |
| Fusion | `FCAT`, `FDIF`, `FXA<heads>` | `Fusion.type`: concatenation, difference, cross_attention | CMP |
| Fusion | `SNP` | `Fusion.sequence_normalization_scope` is pair | CMP |
| Head | `HAVG.<fully connected>`, `HMAX.<…>`, `HFLAT.<…>` | `ClassificationHead.global_pooling`, `fully_connected` | CLS, M2O, M2M, CMP, SEL |
| Head | `HBN`, `HLN` | `ClassificationHead.fully_connected_normalization` | CLS, M2O, M2M, CMP, SEL |
| Head | `LS<rate>` | `ClassificationHead.label_smoothing` above 0 | CLS, M2O, M2M, CMP, SEL |
| Head | `CWB` | `ClassificationHead.class_weighting` balanced | CLS, M2O, M2M, CMP, SEL |
| Head | `HPT`, `HRCL`, `HFRZ` | `ClassificationHead.pretrained`, loaded without its class layer (`class_layer` False), frozen | CLS, M2O, M2M, CMP, SEL |
| Classes | `MC<classes>`, or `ML<classes>` for multi-label | `ClassificationHead.output_mode` | CLS, M2O, M2M, CMP, SEL (`ML`: CLS) |
| Selection | `FPC<frames>` | `Selection.frames_per_class` | SEL |
| Selection | `SLW<weight>` | `Selection.selection_loss_weight` above 0 | SEL |
| Projection head | `PAVG.<layers>`, `PMAX.<…>`, `PFLAT.<…>` | `ProjectionHead.global_pooling`, `fully_connected` | SSL |
| Projection head | `PBN`, `PLN` | `ProjectionHead.fully_connected_normalization` | SSL |
| Method | `SIMCLR-T<temperature>`, `BT-OD<weight>` | `ProjectionHead.method` and its parameter | SSL |
| High resolution network | `HRSE`, `HRRES` | `HRNet.modules` | KPT |
| High resolution network | `HRBN`, `HRLN`, `HRNN` | `HRNet.normalization.feature_map_normalization` | KPT |
| High resolution network | `HR<width>.<modules of every stage>x<blocks>` | `HRNet.base_width`, `modules_per_stage`, `blocks_per_branch` | KPT |
| High resolution network | `HRPT`, `HRFRZ` | `HRNet.pretrained`, frozen | KPT |
| Keypoint head | `K<channels>` | the heatmap channels | KPT |
| Keypoint head | `KH<width>` | `KeypointHead.hidden_width` above 0 | KPT |
| Keypoint head | `HH` | `KeypointHead.higher_resolution` | KPT |
| Keypoint head | `KHPT`, `KHRCL`, `KHFRZ` | `KeypointHead.pretrained`, loaded without its keypoint channel layers (`class_layers` False), frozen | KPT |
| Classes | `MC<classes>` | the foreground classes | KPT |
| Optimizer | `ADAMW`, `SGD` | `Optimization.optimizer` other than adam | all |
| Optimizer | `WD<decay>` | `Optimization.weight_decay` above 0 | all |
| Suffix | `<suffix>` | `ExperimentConfiguration.run_name_suffix` | all |

The suffix starts a run of its own among runs of the same settings, and resuming it needs the same suffix: with `run_name_suffix: "v1"`, `self_supervised/configuration_synthetic_shapes.yaml` trains in `SSL_IN-KAR-256_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_PAVG.512.128-PBN_SIMCLR-T0.2_v1`. It holds letters, digits, `-` and `.` only.

**Example of every task**, the run name of its example configuration for DET, INS and KPT, and of its synthetic configuration for the other tasks

Each name is the one `build_run_name()` gives the configuration of its row (in `vision_training/configurations/`), the class count of a synthetic file being that of the label mapping `create_synthetic_dataset.py` writes.

| Task | Configuration | Run name |
|---|---|---|
| DET | `object_detection/configuration_example.yaml` | `DET_IN-KAR-1024_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567-NFPN64BN_A9.32-512.598295_RBN-R2x64-RPL_DSE-DRES-DBN-D7.2x64.256.128_BIN` |
| INS | `instance_segmentation/configuration_example.yaml` | `INS_IN-KAR-1024_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512.598295_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128_MRES-MBN-MUTC-M14.4x64-MR28_BIN` |
| CLS | `classification/configuration_synthetic_shapes.yaml` | `CLS_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HFLAT.128-HLN-CWB_MC35` |
| M2O | `sequence_classification/configuration_synthetic_sequences.yaml` | `M2O_IN-KAR-128_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128-BI_HFLAT.64_MC4` |
| M2M | `sequence_labelling/configuration_synthetic_sequences.yaml` | `M2M_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128_HAVG.64_MC71` |
| CMP | `sequence_comparison/configuration_synthetic_pairs.yaml` | `CMP_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_MEAN_FDIF_HAVG.64_MC2` |
| SEG | `semantic_segmentation/configuration_synthetic_shapes.yaml` | `SEG_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_UTC-DBN-D2x64.2x64.2x32.2x32.1x16_MC35` |
| SEL | `frame_selection/configuration_synthetic_sequences.yaml` | `SEL_IN-KAR-256_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128-BI_HAVG.64_MC4_FPC1-SLW1` |
| SSL | `self_supervised/configuration_synthetic_shapes.yaml` | `SSL_IN-KAR-256_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_PAVG.512.128-PBN_SIMCLR-T0.2` |
| KPT | `keypoint_detection/configuration_example.yaml` | `KPT_IN-KAR-512_SE-RES-BN-B1x64.2x64.2x64_HRSE-HRRES-HRBN-HR32.1.4.3x4_K17-HH_MC1` |

## MLflow Tracking
With `ExperimentConfiguration.tracking.activated: True`, every run is recorded in `<experiment_folder>/mlflow.db`, its files in `<experiment_folder>/mlartifacts/`, next to TensorBoard and `Results.txt`.
- **Kept**: the full configuration and the git commit of every run, in one table sorted and filtered by any setting or metric. A resumed training continues its MLflow run and records the changed settings.
- **Safe**: the values are written by a background thread, and an unavailable database only prints a warning.
- **Limits**: the curves are averages over `information_dump` iterations, the per-iteration curves staying in TensorBoard. The checkpoints, predictions and sample images of the evaluations are not sent, only the run folder path. The package takes about 1 second to import, and a new database a few seconds to create.

| Logged | Tasks |
|---|---|
| The configuration as parameters, the number of model parameters and of samples | all |
| `train/<metric>`, `validation/<metric>`, `learning_rate`, `train/throughput` | all |
| `train/frame_throughput`, the real frames per second | M2O, M2M, CMP, SEL |
| `test/best_<headline metric>`, `Results.txt` and the configuration file; `evaluation_summary.csv` and `evaluation_curves.png`, sent again after every evaluation | all |
| The input samples written by the launch, under `Input_Samples/<split>/` with `samples.json` (`Input_Samples_resume_<n>/` for a resume from iteration n) | all |
| `test_ema/` and `test_ema_class/` metrics, and the files under `evaluation/iteration_<n>/ema/` | all, with the moving average on |
| `test/mAP@0.5`, `mAP@0.75`, `mAP@0.95`, `mAP@[0.5:0.95]` per method (`area`, `11_point`, `101_point`); `test/mAP@0.5 By Size/<size>`, `mAP@[0.5:0.95] By Size/<size>`; `test/mAR@0.5/<all or size>`, `mAR@0.75/<all or size>`, `mAR@0.95/<all or size>`, `mAR@[0.5:0.95]/<all or size>`; `test/Proposal Recall@0.5/<group>`, `Proposal Recall@0.7/<group>`, `Proposals Per Image/<group>`; `test_class/AP@0.5/<class>`; `class_metrics.csv`, `class_thresholds.json`, `confusion_matrix.png`, `proposal_recall.csv`; `anchor_recall.csv` under `analysis/` (`anchor_recall_resume_<n>.csv` for a resume from iteration n) | DET, INS (with the `Mask` metrics) |
| `test/Accuracy`, `Balanced Accuracy`, `Macro F1`, `Top-5 Accuracy` (multi-label: `Subset Accuracy`, `Micro F1`, `Macro F1`), `Macro ROC AUC`, `Macro Average Precision`; `test_class/Precision`, `Recall`, `F1`, `ROC AUC`, `Average Precision`; `class_metrics.csv`, `class_thresholds.json`, `confusion_matrix.png`, `roc_curves.png`, `precision_recall_curves.png`, `precision_recall_thresholds.png` | CLS, M2O, M2M, CMP |
| `test/Sequence Exact Match`, `Edit Score`, `Segment F1@10`, `Segment F1@25`, `Segment F1@50` | M2M |
| `test/Hungarian Selection Accuracy`, `Greedy Selection Accuracy`, `<Decoding> Fully Right Sequences`, `<Decoding> Background Picks`, `<Decoding> Swapped Picks`, `Top-<k> Selection Accuracy`, `Mean True Frame Rank`, `Frame Accuracy`, `Frame Balanced Accuracy`, `Frame Macro F1`; `test_class/Hungarian Selection Accuracy`, `Greedy Selection Accuracy`, `Mean True Frame Rank`; `class_metrics.csv`, `class_thresholds.json`, `confusion_matrix.png` | SEL |
| `test/Mean IoU`, `Mean Dice`, `Mean Boundary IoU`, `Mean Boundary F1`, `Pixel Accuracy`, `Frequency Weighted IoU`, `Macro ROC AUC`, `Macro Average Precision`; `test_class/IoU`, `Dice`, `Boundary IoU`, `Boundary F1`, `Precision`, `Recall`, `ROC AUC`, `Average Precision`; `class_metrics.csv`, `confusion_matrix.png`, `roc_curves.png`, `precision_recall_curves.png` | SEG |
| `test/k-NN Accuracy`, `k-NN Balanced Accuracy`, `k-NN Macro F1`, `Effective Rank`, `Representation Standard Deviation`; `test_class/k-NN Precision`, `k-NN Recall`, `k-NN F1`; `class_metrics.csv` | SSL |
| `test/OKS mAP@0.5`, `OKS mAP@0.75`, `OKS mAP@0.95`, `OKS mAP@[0.5:0.95]` per method, by size, the OKS mean average recalls, `Keypoint Accuracy/PCK@0.05`, `Keypoint Accuracy/Mean OKS`, `Keypoint Accuracy/Mean Keypoint Error (pixels)`; `test_class/OKS AP@0.5/<class>`; `class_metrics.csv`, `keypoint_metrics.csv` | KPT |

- The `test_class/` metrics, the macro ROC AUC and average precision, and the curve and confusion images need `evaluation.per_class_report: True`. The files are always sent.
- MLflow writes `@` as `_at_` and `[0.5:0.95]` as `0.5-0.95` in the metric names.
- `scripts/evaluation/evaluate_checkpoint.py` logs its evaluations in a run of their own, `<run name>-evaluation`.
