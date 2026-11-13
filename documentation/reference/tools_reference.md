# Tools Reference

Settings and outputs of the scripts run after a training, completing the [Evaluating Checkpoints](../../README.md#evaluating-checkpoints), [Comparing Launches](../../README.md#comparing-launches), [Clustering Image Embeddings](../../README.md#clustering-image-embeddings) and [Deployment](../../README.md#deployment) sections of the README. Every setting is a variable of the main block of its script.

## Table Of Content
- [Evaluating Checkpoints](#evaluating-checkpoints)
  - [Logit Temperature](#logit-temperature)
- [Comparing Launches](#comparing-launches)
  - [Comparing The Best Evaluations Of Runs](#comparing-the-best-evaluations-of-runs)
  - [Drawing The Comparison Of Every Stage](#drawing-the-comparison-of-every-stage)
- [Clustering Image Embeddings](#clustering-image-embeddings)
  - [Stored Features And Relaunch](#stored-features-and-relaunch)
  - [Clusters And Kept Images](#clusters-and-kept-images)
  - [Group Pattern](#group-pattern)
- [Deployment](#deployment)
  - [Export](#export)
  - [Throughput Measurement](#throughput-measurement)

## Evaluating Checkpoints
Whenever the checkpoints of a trained run must be judged again, on another split, with other target precisions or another NMS, `scripts/evaluation/evaluate_checkpoint.py` evaluates them into the `Checkpoint_Evaluation/` folder of the run, next to the `Evaluation/` folder of the training, without touching its other files.

| Setting | Meaning |
|---|---|
| `run_name`, `experiment_folder` | the evaluated run, `<experiments>/<run name>` |
| `iteration` | `None` for the last checkpoint, an iteration, or `"all"` for every checkpoint |
| `test_data_folder` | another labelled split, `None` for the test split of the run |
| `batch_size` | the evaluation batch size, `None` keeping the run's |
| `track_evaluation` | sends the evaluation to MLflow, in a run named `<run name>-evaluation`, `None` keeping the `tracking.activated` of the run |
| `maximum_error_images_per_class_pair` | turns the error images on with at most this number per pair of classes, `None` keeping the `evaluation.error_dumps` of the run |
| `target_precisions` | the precision, or list of precisions such as `[0.99, 0.95, 0.90]` (CLS, M2O, M2M, CMP), targeted by the class thresholds, `None` keeping the `evaluation.target_precision` of the run; skipped with a warning for the tasks without class thresholds |
| `nms_criterion`, `nms_distance_exponent` | the final NMS of a DET or INS run replaced, `"iou"` (greedy) or `"distance_iou"` (DIoU-NMS) with the exponent of its distance term, `None` keeping the run's; an NMS other than the run's is written to `<run>/Checkpoint_Evaluation_nms_<criterion>[_<exponent>]/` (e.g. `Checkpoint_Evaluation_nms_distance_iou_0.6`), and the setting is skipped with a warning for the tasks without detector |

It writes `Results.txt`, `Tensorboard/Test` (and `Tensorboard/Test_EMA` for the averaged weights when the run kept a moving average), `training_configuration.yaml`, `label_dictionary.json`, `data.json` (its test split being the evaluated one) and one `Iteration_<n>/[EMA/]` folder per checkpoint, laid out as in the `Evaluation/` folder of the training, directly in `<run>/Checkpoint_Evaluation/`. With `iteration = "all"`, `Checkpoint_Evaluation/evaluation_summary.csv` compares every checkpoint (the averaged weights in `ema_<scalar>` columns, every value between 0 and 1), and the best iteration is printed for both weights, its value written like in the report: `Best Checkpoint : iteration 1500 (Accuracy 82.73 %)`, the rates of CLS, M2O, M2M and CMP as percentages.

### Logit Temperature
A classifier trained with the cross entropy is often overconfident: its probabilities are squeezed against 0 and 1, and its thresholds at a high precision end up at 0.999 or above. Dividing the class logits by a temperature T above 1 before the softmax spreads them without changing the predicted class, so the accuracy and the confusion matrix stay the same. With two classes (or in multi-label mode) the recall at a target precision stays the same and only the threshold moves; with more classes it can move a little.

1. With `per_class_report`, every evaluation writes `Evaluation/Iteration_<n>/temperature_explorer.html` (`Checkpoint_Evaluation/Iteration_<n>/` for the evaluation script), a self-contained page whose slider divides the logits by temperatures from 0.5 to 10. For every temperature it shows, for every class in class index order, the histogram of its score over its own images, to see how far the temperature spreads the scores squeezed against 1, with the thresholds of every target precision, and a table of the thresholds and recalls (columns `Threshold @ P99 %` and `Recall @ P99 %`, the recall written as a percentage with its change from T = 1 in points below it, e.g. `(+0.36 points from T = 1)`). The thresholds and scores stay between 0 and 1. Up to 100 classes, the evaluation also keeps the logits next to the page, `class_logits.npz` (with the labels, class names, target precisions, class F1 and title; about 0.3 MB for 40,000 images of 2 classes, 20 MB for 50,000 images of 100 classes).
2. Start `scripts/evaluation/serve_temperature_explorer.py` and open `http://127.0.0.1:8091/`. Its main block sets `experiments_roots` (the folders of the repository root whose name holds `experiments`) and `port`. The index lists the evaluations holding a page, `<run>/Evaluation/Iteration_<n>[/EMA]` of the training and `<run>/Checkpoint_Evaluation/Iteration_<n>[/EMA]` of the evaluation script, with the run folder up to three folders below an experiments folder, grouped by run folder (with its number of iterations, saved temperatures and exports), then by iteration, newest first and the newest open, every iteration giving its evaluations, trained weights first, with their evaluation, a link per weights (`Trained`, `EMA`), temperatures (`Any` with `class_logits.npz`, `Grid only` without), saved temperature and exports. Every page is built again with the current template: from `class_logits.npz`, where any temperature from 0.05 to 100 can be typed in (**Show T**), or from the data of the written page, with the temperatures of its slider. With `class_logits.npz`, the **Score Ranges** section takes a list of 1 to 10 (`MAXIMUM_SCORE_CUT_POINTS`) distinct cut points strictly between 0 and 1, in any order (`0.5, 0.8`), and gives for every class of the page the share and the number of its images whose score for their own class is below 0.5, from 0.5 to below 0.8 and from 0.8, a score equal to a cut point counting in the range above, as an image scoring its threshold is given the class. The server counts them from the logits at the temperature shown (`/api/score_ranges`, `compute_score_ranges`), again whenever the temperature changes, and the browser keeps the cut points of every evaluation for its next visit.
3. Choose the temperature and press **Save Thresholds At This Temperature**: `saved_temperature_thresholds.json` is written next to the page, in the format of `class_thresholds.json`, with the temperature as `logit_temperature` and the threshold and recall of every class at every target precision, found by the server from the logits (the thresholds the page shows), or, without logits, those posted by the page after a check against the `class_thresholds.json` of the folder. A save replaces the earlier one, and is refused while the checkpoint is being exported. Opened as a plain file, the page downloads the file instead; the button needs every class on the page (at most 100 classes).
4. Choose the formats, version, maximum batch size and TensorRT precision of the **Export** panel (those of the main block of `export_model.py` first) and press **Export At T = …**: `vision_training/deployment/explorer_export.py` exports the checkpoint and weights of the page at the saved temperature, in a separate process, one export at a time, into `<run>/Exports/Iteration_<n>[_EMA]_T<temperature>/` (written as `<folder>.partial` until it succeeds), a Triton model repository holding `image_classifier_<format>/<version>/` with its `class_thresholds.json`, `export_record.json` (run, iteration, weights, temperature, settings, date) and `export_log.txt`, the console output the page shows while it runs. Another temperature gets its own folder, the same temperature replaces its folder, and a failed export leaves its `.partial` folder with its console output until the next export of that temperature. The run folder and checkpoint of the evaluation must still be on disk.
5. Without the page, export the checkpoint with `use_saved_temperature` ([Export](#export)).

## Comparing Launches
To decide which setting of a stage of a hyperparameter experiment to keep, `scripts/evaluation/collect_experiment_results.py` gathers every launch of the experiment into one CSV file. Every launch trains in its own `run_<n>` folder, a repeat launch in `run_<n><letters>` (`run_23b`).

| Setting | Meaning |
|---|---|
| `task` | the experiment, `"classification"`, `"sequence_labelling"` or `"object_detection"`, which chooses the folder, the metric and the swept settings below |
| `experiment_folders` | the folder of the `run_<n>` folders of every experiment: `<repository>/hyperparameter_experiments` for the classification, its `sequence_labelling/` and `object_detection/` subfolders for the others |
| `metrics`, `metric_columns` | the compared test curve of every experiment, `"Accuracy"`, `"Macro F1"` and `"mAP@[0.5:0.95]/area"`, and the prefix of its columns, `accuracy`, `macro_f1` and `map_50_95` |
| `number_last_evaluations` | the last evaluations averaged in a second mean column, 4 |
| `output_path` | the CSV file, `<experiment_folder>/results.csv` |

Every row is a launch and its weights (`evaluated_weights`: `trained`, or `ema` from `Tensorboard/Test_EMA` for a launch keeping a moving average). It gives the metric at every evaluated iteration, its mean, the mean of the last evaluations (`<metric column>_last_<n>_mean`, which ignores how fast a launch starts), whether the launch finished, the swept settings of the experiment (`TASK_SWEPT_SETTINGS`) and the run folder. The launches of the same settings and weights are grouped in the order of their first launch, followed by a `mean` row. A launch still training leaves its future cells empty, so the file can be written at any time.

### Comparing The Best Evaluations Of Runs
`scripts/evaluation/compare_best_evaluations.py` prints Markdown tables comparing the best test evaluation of runs, over the trained (`Tensorboard/Test`) and averaged (`Tensorboard/Test_EMA`) weights.

| Setting | Meaning |
|---|---|
| `experiment_folders` | experiment folders, whose subfolders holding a `training_configuration.yaml` are runs, or run folders |
| `best_metric`, `higher_is_better` | the test curve choosing the best evaluation |
| `evaluated_weights` | `["trained", "ema"]` or one of them; a win of the averaged weights is marked ` (EMA)` |
| `columns` | `experiment`, `run`, `best_iteration`; a test curve (`"Recall/{class}"` gives one column per class); a setting (`"configuration:Data.image_settings.size"`); a count of `data.json`, `"<split>_<unit>"` or `"<split>_<unit>/{class}"` (unit `images`, `annotations`, `sequences`, `frames` or `pairs`) |
| `column_titles` | the titles of the columns, `{class}` being replaced by the class name, e.g. `{"Recall/{class}": "Recall {class}"}` |
| `table_grouping` | `"run"` for one table per run, `"columns"` for one table per set of column titles |
| `curves_in_percent`, `number_decimals`, `output_path` | whether the curves are written as the percentages of the reports, `89.80 %` (`True` in the main block, the classification curves being rates), the decimals of the curves otherwise (3), and a Markdown file receiving the tables (`None` to print only) |

```text
| Image Size | Best Iteration | Balanced Accuracy | Recall empty_bin | Recall not_empty | Validation Images | Train Images |
|---|---|---|---|---|---|---|
| 512 | 200 | 89.80 % | 94.60 % | 85.00 % | 18,702 | 43,577 |
```

A run not evaluated yet is printed in yellow and left out. A missing curve or count stops the script, listing the available ones.

### Drawing The Comparison Of Every Stage
After a stage of a hyperparameter experiment, `scripts/evaluation/plot_experiment_comparisons.py` draws its figure next to its experiment plan, from the copy of its results file in `documentation/architecture_benchmarking` (`results.csv` for classification, `<task>_results.csv` for the others) and its entry in `experiment_comparisons.yaml`.

| Setting | Meaning |
|---|---|
| `registry_path` | the registry of the compared stages, `documentation/architecture_benchmarking/experiment_comparisons.yaml` by default, the results files being read from its folder |
| `output_folder` | the folder of the figures, one subfolder per experiment, `documentation/architecture_benchmarking/figures` by default |
| `experiment`, `stage` | the experiment (`"classification"`, `"sequence_labelling"`, `"object_detection"`) and stage (`"3"`, `"2b"`, `"repeat"`) to draw alone, `None` for every stage |

Every entry of the registry gives the results file, the stage and its title, the iteration and weights the plan compares (`last_iteration`, `compared_weights`), the verdict, and the compared settings (`arms`): their label, their launches as written in the results file, whether they are the reference, and `trained_as_averaged` for a setting trained without moving average whose trained weights stand in the column of the averaged weights. A figure has a column per kind of weights evaluated for every launch of the stage; every launch is a thin line, the mean of the launches of every setting a thick one labelled at the compared iteration, and the lower panel gives the gap of every setting to the reference. The accuracy of the classification and the macro F1 of the sequence labelling (`PERCENT_METRICS`) are written as the percentages of the reports: labels such as `85.81 %` on axes such as `86 %`, gaps in points such as `+0.95 points`, the gap axis named `Gap to <reference>` with `(points)` on a line of its own; the detection mAP keeps 4 decimals. The verdict under the title is wrapped by `wrap_verdict`, which keeps every number with its unit (`85.81 %`, `+0.95 points`) on one line. A stage without exactly one reference stops the script.

## Clustering Image Embeddings
`scripts/embedding/cluster_image_embeddings.py` embeds images with a trained run and clusters them, to choose diverse images from a dataset of near-identical video frames.

| Setting | Meaning |
|---|---|
| `experiment_folder`, `iteration`, `use_averaged_weights` | the embedding run and checkpoint: any task for the convolution blocks, a classification run for the fully connected layers |
| `data_sources` | `"run"` (the splits of the run), a configuration file (its splits), a split folder of any task, or a folder of plain images; a folder given twice is embedded once |
| `convolution_block_indices` | the blocks embedded, counted from 1, their whole output before pooling; a block of more than 32,768 values per image (`MAXIMUM_BLOCK_VALUES`) is refused |
| `fully_connected_layer_indices` | the hidden layers of the classification head embedded, after their ReLU |
| `batch_size`, `number_workers` | the extraction loader |
| `normalize_vectors` | divides every vector by its length, so the distances compare which features are active |
| `pca_dimensions` | the dimensions kept by the PCA before clustering (200 by default), with the variance they hold printed |
| `clustering_algorithm`, `clustering_parameters` | `"hdbscan"` (`min_cluster_size`, `min_samples`, `cluster_selection_method` `"eom"` or `"leaf"`), `"dbscan"` (`eps`, `min_samples`) or `"kmeans"` (`number_clusters`, `number_initializations`) |
| `number_representatives_per_cluster`, `isolated_image_selection` | the images kept per cluster, and the isolated images kept (`"all"`, `"none"` or a number drawn at random) |
| `group_pattern` | the regular expression giving the group of every image ([Group Pattern](#group-pattern)), or `None` |
| `seed`, `mosaic_grid_size`, `output_root` | the draws, the images per mosaic row, and the output root, `<experiments>/embeddings` by default |

An asked layer outside the run stops the script before any inference, with the valid range. The images are read with the preprocessing of the run.

HDBSCAN runs on a single core, its time growing with the square of the images and with the dimensions. On 31,805 images of block 6, it took 116 seconds at 200 dimensions and 487 at 800, the clusters agreeing closely (adjusted Rand index of 0.81 and more). The timings of every layer and setting are at the top of the script.

### Stored Features And Relaunch
The features of a source are stored in `<output_root>/<run name>/Iteration_<n>[_EMA]/<source>/`:
- `images.csv`: the file name and category of every image;
- `features_block_<b>.npy` and `features_fully_connected_<l>.npy`, one float32 row per image, and `predictions.npy` for a classification run;
- `extraction.json`: what produced the features (checkpoint, weights, preprocessing, source and its SHA-256) and the progress of every layer.

A relaunch with the same layers only clusters again, a new layer runs the model for that layer only, and an interrupted extraction continues where it stopped. Another checkpoint or changed images stop the script without overwriting anything: delete the folder or change `output_root`. The images and features are released from the Linux page cache once read (`posix_fadvise` with `POSIX_FADV_DONTNEED`, no `sudo`), so they do not take the unified memory of the DGX Spark from the GPU.

### Clusters And Kept Images
Every layer is clustered in `<source>/clusters/<layer>/<clustering>/`, written again at every launch. A classification split is clustered category by category. The isolated images are those in no cluster.
- `clusters.csv`: every image with its cluster (-1 when isolated), its membership probability, its distance to the centre and rank, why it is kept, its predicted class and its group.
- `cluster_summary.csv`: every cluster with its images, kept images, error rate and, with a group pattern, its groups.
- `<category>/cluster_<k>/` and `<category>/isolated_images/`: links to the images, named by their rank, with a mosaic `.jpg` of the first ones.
- Three folders of kept images, with the same number per category, to compare by training on each:
  - `kept_images_by_clustering/`: the images closest to the centre of every cluster, and the chosen isolated images;
  - `kept_images_at_random/`;
  - `kept_images_evenly_spaced_per_group/`, with a group pattern: the images spread evenly over every group, in name order.

Every folder of kept images lists them in `kept_images.csv`. The kept images of a labelled split form a split to train on, `data` linking to the source and `labels.json` holding the kept samples. A table of the initial, clustered, isolated and kept images of every layer and category ends the run.

### Group Pattern
`group_pattern` is a regular expression searched in the file name of every image, relative to its data folder, e.g. to find the video a frame comes from. The group is the first capturing group that matched, or the whole match without capturing group; the images it does not match form `(no group)`.

| Pattern | File name | Group |
|---|---|---|
| `r"^(video_t\d+_[\d-]+_[\d-]+)_"` | `video_t0053_2026-09-25_05-51-30_0037018ms.jpg` | `video_t0053_2026-09-25_05-51-30` |
| `r"^([^/]+)/"`, the subfolder | `camera_1/frame_0012.png` | `camera_1` |
| `r"^(video_\d+)_\|^(camera_\d+)/"` | `video_3_frame_1.png`, `camera_2/frame_1.png` | `video_3`, `camera_2` |
| `r"video_\d+"` | `camera_2/video_7_frame_1.png` | `video_7` |

## Deployment

### Export
`vision_training/deployment/export_model.py` writes every model with its `config.pbtxt` and checks it against the PyTorch model. `use_averaged_weights` exports the moving average of the weights, raising an error for a checkpoint without it. For a classifier, `use_saved_temperature` builds the temperature of the `saved_temperature_thresholds.json` of the exported iteration and weights ([Logit Temperature](#logit-temperature)) into the model, a constant `logit_temperature` of the TorchScript and ONNX graphs dividing the logits before the softmax, and copies the file to `<model>/<version>/class_thresholds.json`. The file is looked for in `<run>/Evaluation/Iteration_<n>/[EMA/]` of the training and in `<run>/Checkpoint_Evaluation/Iteration_<n>/[EMA/]` of the evaluation script: the export stops when it is in neither or in both, or when it does not match the `class_thresholds.json` next to it or the number of classes of the model. `saved_thresholds_path` gives the file instead (the export button of the page gives the file of its evaluation), accepted only in one of these two folders of the exported run, iteration and weights. Without it, a classifier is exported with a temperature of 1, and the copy of an earlier export is removed. It is refused for the other tasks. Every model also gets the input contract of the run, in the `metadata_props` of `model.onnx` and as `<model>/<version>/preprocessing.json` for both formats, the export stopping before writing any file when the contract cannot describe the input of the run ([README](../../README.md#input-contract)). The inputs and outputs of every model are listed in the [README](../../README.md#export).

**Semantic segmentation (SEG).** The outputs are at the network resolution: `restore_prediction_to_original_image` (`vision_training/data/preprocessing.py`) brings them back to the original image. A run trained on tiles takes one tile: the client scales the image by `Data.tiling.scale` and segments it with `segment_image_by_tiles` (`vision_training/models/segmentation/sliding_window.py`), given a function sending a batch of tiles to Triton; the scale, overlap, interpolation and padding of the tiles are in the input contract of the model (`input_resize` `tiles`, [README](../../README.md#input-contract)).

**Instance segmentation (INS).** Every mask holds the probabilities of the detected class inside its box. The client brings the boxes back to the original image with `restore_boxes_to_original_image` (`vision_training/data/preprocessing.py`), pastes the masks with `paste_instance_masks` (`vision_training/models/instance_segmentation/mask_operations.py`), and keeps the pixels above `MaskHead.inference.mask_threshold`.

**Keypoint detection (KPT).** The model gives the keypoint candidates of every channel, and the client groups them:
1. Preprocess the image with `preprocess_image` (`vision_training/data/preprocessing.py`) and send it.
2. Build the channel table with `build_keypoint_channel_table` (`vision_training/models/keypoints/keypoint_detector.py`), from the `Data.keypoint_categories` read by `load_experiment_configuration` (`vision_training/configuration_processing/configuration_loader.py`).
3. Group the candidates with `group_keypoint_candidates` (`vision_training/models/keypoints/keypoint_grouping.py`) and the `KeypointHead.inference` settings of the run.
4. Bring the keypoints back to the original image with `restore_boxes_to_original_image`, as boxes of zero size.

**Frame selection (SEL).** The export adds the CPU decoder `frame_selector_decoder`, a Triton Python backend running the decoding of `Selection.decoding` with the thresholds of the `class_thresholds.json` of the exported iteration, in `Evaluation/` or `Checkpoint_Evaluation/` (0 when the checkpoint was not evaluated, an error when both folders hold one), and one ensemble per format, `frame_selector_<format>_ensemble`, chaining network and decoder. The decoder folder holds a copy of `vision_training/models/sequence/frame_selection_decoding.py` and `decoding_settings.json` (the wanted classes in output order), and uses the numpy and scipy of the Triton image:

| Output | Type | Shape | Content |
|---|---|---|---|
| `selected_frames` | INT64 | [B, W, N] | frames chosen for every wanted class, -1 when the sequence is too short |
| `selected_probabilities` | FP32 | [B, W, N] | probability of every chosen frame |
| `low_confidence` | BOOL | [B, W, N] | choice below the threshold of its class, or no frame |
| `candidate_frames` | INT64 | [B, W, k] | the k most probable frames of every wanted class |
| `candidate_probabilities` | FP32 | [B, W, k] | their probabilities |

W is the number of wanted classes, N `Selection.frames_per_class` and k `ExperimentConfiguration.evaluation.top_k`.

### Throughput Measurement
`vision_training/deployment/run_inference.py` times the images of `vision_training/deployment/inference_data/`. The image tasks send one request per image. The sequence tasks send the whole folder as one sequence, of 2 to `maximum_sequence_length` images, CMP splitting it into a before and an after half. SEL decodes every sequence locally, and queries Triton through its ensembles.
