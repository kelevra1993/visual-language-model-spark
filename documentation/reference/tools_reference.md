# Tools Reference

Settings and outputs of the scripts run after a training, completing the [Evaluating Checkpoints](../../README.md#evaluating-checkpoints), [Comparing Launches](../../README.md#comparing-launches), [Clustering Image Embeddings](../../README.md#clustering-image-embeddings) and [Deployment](../../README.md#deployment) sections of the README. Every setting is a variable of the main block of its script.

## Table Of Content
- [Evaluating Checkpoints](#evaluating-checkpoints)
  - [Logit Temperature](#logit-temperature)
- [Comparing Launches](#comparing-launches)
  - [Comparing The Best Evaluations Of Runs](#comparing-the-best-evaluations-of-runs)
- [Clustering Image Embeddings](#clustering-image-embeddings)
  - [Stored Features And Relaunch](#stored-features-and-relaunch)
  - [Clusters And Kept Images](#clusters-and-kept-images)
  - [Group Pattern](#group-pattern)
- [Deployment](#deployment)
  - [Export](#export)
  - [Throughput Measurement](#throughput-measurement)

## Evaluating Checkpoints
`scripts/evaluation/evaluate_checkpoint.py` evaluates the checkpoints of a run again, without modifying its folder.

| Setting | Meaning |
|---|---|
| `run_name`, `experiment_folder` | the evaluated run, `<experiments>/<run name>` |
| `iteration` | `None` for the last checkpoint, an iteration, or `"all"` for every checkpoint |
| `test_data_folder` | another labelled split, `None` for the test split of the run |
| `batch_size` | the evaluation batch size |
| `track_evaluation` | sends the evaluation to MLflow, in a run named `<run name>-evaluation` |
| `maximum_error_images_per_class_pair` | turns the error images on with at most this number per pair of classes, `None` keeping the `evaluation.error_dumps` of the run |
| `target_precisions` | the precision, or list of precisions such as `[0.99, 0.95, 0.90]` (CLS, M2O, M2M, CMP), targeted by the class thresholds, `None` keeping the `evaluation.target_precision` of the run; skipped with a warning for the tasks without class thresholds |
| `logit_temperature` | CLS only: the temperature dividing the class logits before the softmax, 1 for the probabilities of the model; the thresholds of `class_thresholds.json` are found at this temperature, which the file records ([Logit Temperature](#logit-temperature)); another value than 1 is refused for the other tasks |
| `output_folder` | the root of the evaluations, `<experiments>/evaluations` by default |

It writes `Results.txt`, `Tensorboard/Test` and `Evaluation/` in `<output_folder>/<run name>/`, as the training does, and the averaged weights when the run kept a moving average. With `iteration = "all"`, `Evaluation/evaluation_summary.csv` compares every checkpoint (the averaged weights in `ema_<scalar>` columns), and the best iteration is printed for both weights.

### Logit Temperature
A classifier trained with the cross entropy is often overconfident: its probabilities are squeezed against 0 and 1, and its thresholds at a high precision end up at 0.999 or above. Dividing the class logits by a temperature T above 1 before the softmax spreads them without changing the predicted class, so the accuracy and the confusion matrix stay the same. With two classes (or in multi-label mode) the recall at a target precision stays the same and only the threshold moves; with more classes it can move a little.

1. With `per_class_report`, every evaluation writes `Evaluation/Iteration_<n>/temperature_explorer.html`, a self-contained page whose slider divides the logits by temperatures from 0.5 to 10. For every temperature it shows the histograms of the probability of every class over its images (above) and the other images (below) with the thresholds of every target precision, a table of the thresholds and recalls (with the change of the recall from T = 1), the negative log likelihood, the calibration error and the reliability diagram. The calibrated temperature, of lowest negative log likelihood on the evaluated split, is suggested.
2. Set the chosen temperature as `logit_temperature` and evaluate the checkpoint again: `class_thresholds.json` records `logit_temperature` with the thresholds found at it, and the report notes the temperature.
3. Export the checkpoint with `class_thresholds_path` set to that file ([Export](#export)).

## Comparing Launches
`scripts/evaluation/collect_experiment_results.py` gathers the launches of a hyperparameter experiment into one CSV file. Every launch trains in its own `run_<n>` folder, a repeat launch in `run_<n><letters>` (`run_23b`).

| Setting | Meaning |
|---|---|
| `experiment_folder` | the folder of the `run_<n>` folders, `<repository>/hyperparameter_experiments` by default |
| `metric` | the compared test curve, e.g. `"Accuracy"`, `"Balanced Accuracy"`, `"Macro F1"`, `"Top-5 Accuracy"` |
| `number_last_evaluations` | the last evaluations averaged in a second mean column, 4 by default |
| `output_path` | the CSV file, `<experiment_folder>/results.csv` by default |

Every row is a launch, in launch order with its repeats right after it. It gives the metric at every evaluated iteration, its mean, the mean of the last evaluations (`<metric>_last_<n>_mean`, which ignores how fast a launch starts), whether the launch finished, the swept settings (`SWEPT_SETTINGS`) and the run folder. The launches of the same settings are followed by a `mean` row. A launch still training leaves its future cells empty, so the file can be written at any time.

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
| `number_decimals`, `output_path` | the decimals of the curves, and a Markdown file receiving the tables (`None` to print only) |

```text
| Image Size | Best Iteration | Balanced Accuracy | Recall empty_bin | Recall not_empty | Validation Images | Train Images |
|---|---|---|---|---|---|---|
| 512 | 200 | 0.898 | 0.946 | 0.850 | 18,702 | 43,577 |
```

A run not evaluated yet is printed in yellow and left out. A missing curve or count stops the script, listing the available ones.

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
`vision_training/deployment/export_model.py` writes every model with its `config.pbtxt` and checks it against the PyTorch model. `use_averaged_weights` exports the moving average of the weights, raising an error for a checkpoint without it. For a classifier, `class_thresholds_path` names the `class_thresholds.json` of the evaluation of the exported checkpoint, in the run or in the evaluations folder of the evaluation script: its `logit_temperature` (1 for a file written before it was recorded) is built into the model, a constant `logit_temperature` of the TorchScript and ONNX graphs dividing the logits before the softmax, and the file is copied to `<model>/<version>/class_thresholds.json`. The file must be `<run>/Evaluation/Iteration_<n>/[EMA/]class_thresholds.json` of the exported run, iteration and weights, with the number of classes of the model, otherwise the export stops. Without it, a classifier is exported with a temperature of 1, and the copy of an earlier export is removed. It is refused for the other tasks. The inputs and outputs of every model are listed in the [README](../../README.md#export).

**Semantic segmentation (SEG).** The outputs are at the network resolution: `restore_prediction_to_original_image` (`vision_training/data/preprocessing.py`) brings them back to the original image. A run trained on tiles takes one tile: the client scales the image by `Data.tiling.scale` and segments it with `segment_image_by_tiles` (`vision_training/models/segmentation/sliding_window.py`), given a function sending a batch of tiles to Triton.

**Instance segmentation (INS).** Every mask holds the probabilities of the detected class inside its box. The client brings the boxes back to the original image with `restore_boxes_to_original_image` (`vision_training/data/preprocessing.py`), pastes the masks with `paste_instance_masks` (`vision_training/models/instance_segmentation/mask_operations.py`), and keeps the pixels above `MaskHead.inference.mask_threshold`.

**Keypoint detection (KPT).** The model gives the keypoint candidates of every channel, and the client groups them:
1. Preprocess the image with `preprocess_image` (`vision_training/data/preprocessing.py`) and send it.
2. Build the channel table with `build_keypoint_channel_table` (`vision_training/models/keypoints/keypoint_detector.py`), from the `Data.keypoint_categories` read by `load_experiment_configuration` (`vision_training/configuration_processing/configuration_loader.py`).
3. Group the candidates with `group_keypoint_candidates` (`vision_training/models/keypoints/keypoint_grouping.py`) and the `KeypointHead.inference` settings of the run.
4. Bring the keypoints back to the original image with `restore_boxes_to_original_image`, as boxes of zero size.

**Frame selection (SEL).** The export adds the CPU decoder `frame_selector_decoder`, a Triton Python backend running the decoding of `Selection.decoding` with the thresholds of `class_thresholds.json` (0 when the checkpoint was not evaluated), and one ensemble per format, `frame_selector_<format>_ensemble`, chaining network and decoder. The decoder folder holds a copy of `vision_training/models/sequence/frame_selection_decoding.py` and `decoding_settings.json` (the wanted classes in output order), and uses the numpy and scipy of the Triton image:

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
