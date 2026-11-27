# Tools Reference

Settings and outputs of the scripts run after a training, completing the [Evaluating Checkpoints](../../README.md#evaluating-checkpoints) and [Deployment](../../README.md#deployment) sections of the README. Every setting is a variable of the main block of its script.

## Table Of Content
- [Evaluating Checkpoints](#evaluating-checkpoints)
  - [Logit Temperature](#logit-temperature)
  - [Drawing The Figures Of A Research Report](#drawing-the-figures-of-a-research-report)
  - [Scoring A Large Split](#scoring-a-large-split)
- [Comparing Launches](#comparing-launches)
  - [Comparing The Best Evaluations Of Runs](#comparing-the-best-evaluations-of-runs)
  - [Drawing The Comparison Of Every Stage](#drawing-the-comparison-of-every-stage)
  - [Drawing The Mask Head Designs](#drawing-the-mask-head-designs)
- [Clustering Image Embeddings](#clustering-image-embeddings)
  - [Stored Features And Relaunch](#stored-features-and-relaunch)
  - [Clusters And Kept Images](#clusters-and-kept-images)
  - [Group Pattern](#group-pattern)
- [Deployment](#deployment)
  - [Export](#export)
  - [Input Contract](#input-contract)
  - [Test Split Parity](#test-split-parity)
  - [FP16 TorchScript Trial](#fp16-torchscript-trial)
  - [Throughput Measurement](#throughput-measurement)
  - [Serving Benchmark](#serving-benchmark)
  - [Export Size Check](#export-size-check)

## Evaluating Checkpoints
Whenever the checkpoints of a trained run must be judged again, on another split, with other target precisions or another NMS, `scripts/evaluation/evaluate_checkpoint.py` evaluates them into the `Checkpoint_Evaluation/` folder of the run, next to the `Evaluation/` folder of the training, without touching its other files.

| Setting | Meaning |
|---|---|
| `run_name`, `experiment_folder` | the evaluated run, `<experiments>/<run name>` |
| `iteration` | `None` for the last checkpoint, an iteration, or `"all"` for every checkpoint; a run without checkpoint, or an iteration without one, stops the script with a `FileNotFoundError` naming its `Weights/` folder before anything is written |
| `test_data_folder` | another labelled split, `None` for the test split of the run; another split is written to `<run>/Checkpoint_Evaluation_split_<parent folder>_<split folder>/` (e.g. `Checkpoint_Evaluation_split_synthetic_shapes_validation`), and a folder whose `data.json` records another test split is refused, so that the evaluations of two splits never mix |
| `batch_size` | the evaluation batch size, `None` keeping the run's |
| `track_evaluation` | sends the evaluation to MLflow, in a run named `<run name>-evaluation` in the backend the launch panel names (that of the `experiment_folder` of the configuration of the run, even when the run folder was moved since), `None` keeping the `tracking.activated` of the run |
| `save_predictions` | writes `predictions.csv` (CLS, M2O, M2M, CMP) or `selections.csv` (SEL) in every evaluated iteration folder when `True`, `None` keeping the `evaluation.save_predictions` of the run; skipped with a warning for the tasks without the setting |
| `maximum_error_images_per_class_pair` | turns the error images on with at most this number per pair of classes, `None` keeping the `evaluation.error_dumps` of the run; skipped with a warning for the tasks without error images (SEG, KPT, SSL) |
| `target_precisions` | the precision, or list of precisions such as `[0.99, 0.95, 0.90]` (CLS, M2O, M2M, CMP), targeted by the class thresholds, `None` keeping the `evaluation.target_precision` of the run; skipped with a warning for the tasks without class thresholds |
| `nms_criterion`, `nms_distance_exponent` | the final NMS of a DET or INS run replaced, `"iou"` (greedy) or `"distance_iou"` (DIoU-NMS) with the exponent of its distance term, `None` keeping the run's; an NMS other than the run's is written to `<run>/Checkpoint_Evaluation_nms_<criterion>[_<exponent>]/` (e.g. `Checkpoint_Evaluation_nms_distance_iou_0.6`), and the setting is skipped with a warning for the tasks without detector |

It writes `Results.txt`, `Tensorboard/Test` (and `Tensorboard/Test_EMA` for the averaged weights when the run kept a moving average), `training_configuration.yaml`, `label_dictionary.json`, `data.json` (its test split being the evaluated one) and one `Iteration_<n>/[EMA/]` folder per checkpoint, laid out as in the `Evaluation/` folder of the training, directly in `<run>/Checkpoint_Evaluation/`. A checkpoint evaluated again with error images replaces the `Errors/` folder of its iteration and weights. Every evaluated checkpoint adds its row to `Checkpoint_Evaluation/evaluation_summary.csv` (the averaged weights in `ema_<scalar>` columns with their number of updates restored with the checkpoint in `ema_number_updates`, every rate between 0 and 1, the rows of the earlier launches into the folder kept, in increasing iteration order) and draws `Checkpoint_Evaluation/evaluation_curves.png` again, as during the training, the window of the moving average ending at the highest evaluated iteration ([Run Folder](training_reference.md#run-folder)). `iteration = "all"` evaluates every checkpoint, and the best of the evaluated iterations is printed for both weights, its value written like in the report: `Best Checkpoint : iteration 2000 (Accuracy 81.76 %)`, the headline metric of every task (`mAP@0.5/area` of DET, `Mean IoU` of SEG, `Hungarian Selection Accuracy` of SEL, ...) being a rate written as a percentage, except the effective rank of an SSL run without labelled test images, which keeps 4 decimals.

To find where a classifier fails, read its confusion matrix (counts coloured by their share of the true class, precision and recall in the margins; above 1,000 classes no image is drawn, and `confusion_cells.csv` lists the non-zero cells as `true_class,predicted_class,count` in place of `confusion_matrix.csv`), then the misclassified images saved in `Errors/<true>-TO-<predicted>/` (at most `maximum_error_images_per_class_pair` per pair), named after the sample and the probability of the predicted class.

<p align="center">
  <img src="../../README/confusion_matrix.png" width="80%" />
</p>

<p align="center">
  <img src="../../README/evaluation_error_images.png" width="100%" />
</p>
<p align="center"><em>The confusion matrix of the 35 classes of the synthetic shapes classifier of the README screenshots, evaluated at iteration 2000 (accuracy 81.76 %), and the error folders of its three largest confusions: hollow parallelograms read as hollow trapezoids, trapezoids as parallelograms and rectangles as squares.</em></p>

### Logit Temperature
A classifier trained with the cross entropy is often overconfident: its probabilities are squeezed against 0 and 1, and its thresholds at a high precision end up at 0.999 or above. Dividing the class logits by a temperature T above 1 before the softmax spreads them without changing the predicted class, so the accuracy and the confusion matrix stay the same. With two classes (or in multi-label mode) the recall at a target precision stays the same and only the threshold moves; with more classes it can move a little.

1. With `per_class_report`, every evaluation writes `Evaluation/Iteration_<n>/temperature_explorer.html` (`Checkpoint_Evaluation/Iteration_<n>/` for the evaluation script), a self-contained page whose slider divides the logits by temperatures from 0.5 to 10. For every temperature it shows, for every class in class index order, the histogram of its score over its own images, to see how far the temperature spreads the scores squeezed against 1, with the thresholds of every target precision, and a table of the thresholds and recalls (columns `Threshold @ P99 %` and `Recall @ P99 %`, the recall written as a percentage with its change from T = 1 in points below it, e.g. `(+0.36 points from T = 1)`). The thresholds and scores stay between 0 and 1. Up to 100 classes, the evaluation also keeps the logits next to the page, `class_logits.npz` (with the labels, class names, target precisions, class F1 and title; about 0.3 MB for 40,000 images of 2 classes, 20 MB for 50,000 images of 100 classes).
2. Start `scripts/evaluation/serve_temperature_explorer.py` and open `http://127.0.0.1:8091/`. Its main block sets `experiments_roots` (the folders of the repository root whose name holds `experiments`) and `port`. The index lists the evaluations holding a page, `<run>/Evaluation/Iteration_<n>[/EMA]` of the training and `<run>/Checkpoint_Evaluation/Iteration_<n>[/EMA]` of the evaluation script, with the run folder up to three folders below an experiments folder, grouped by run folder (with its number of iterations, saved temperatures and exports), then by iteration, newest first and the newest open, every iteration giving its evaluations, trained weights first, with their evaluation, a link per weights (`Trained`, `EMA`), temperatures (`Any` with `class_logits.npz`, `Grid only` without), saved temperature and exports. Every page is built again with the current template: from `class_logits.npz`, where any temperature from 0.05 to 100 can be typed in (**Show T**), or from the data of the written page, with the temperatures of its slider. So that the exported checkpoint is known by its performance, every page opens on its global metrics, read from the `evaluation_summary.csv` of its `Evaluation/` or `Checkpoint_Evaluation/` folder (`describe_model_performance`, `PERFORMANCE_METRICS`: Balanced Accuracy, Accuracy and Macro F1, or Subset Accuracy, Micro F1 and Macro F1 in multi-label mode, the `ema_` columns for the EMA weights), with a last tile for the best iteration of that summary by the first metric over the trained and EMA weights, linked to its page, or "This checkpoint", then its experiment folder and run name; a summary without the row of the checkpoint shows no metric. With `class_logits.npz`, the **Score Ranges** section takes a list of 1 to 10 (`MAXIMUM_SCORE_CUT_POINTS`) distinct cut points strictly between 0 and 1, in any order (`0.5, 0.8`), and gives for every class of the page the share and the number of its images whose score for their own class is below 0.5, from 0.5 to below 0.8 and from 0.8, a score equal to a cut point counting in the range above, as an image scoring its threshold is given the class. The server counts them from the logits at the temperature shown (`/api/score_ranges`, `compute_score_ranges`), again whenever the temperature changes, and the browser keeps the cut points of every evaluation for its next visit, a first visit counting with `0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99` (`DEFAULT_SCORE_CUT_POINTS`).
3. Choose the temperature and press **Save Thresholds At This Temperature**: `saved_temperature_thresholds.json` is written next to the page, in the format of `class_thresholds.json`, with the temperature as `logit_temperature` and the threshold and recall of every class at every target precision, found by the server from the logits (the thresholds the page shows), or, without logits, those posted by the page after a check against the `class_thresholds.json` of the folder. A save replaces the earlier one, and is refused while the checkpoint is being exported. Opened as a plain file, the page downloads the file instead; the button needs every class on the page (at most 100 classes).
4. Choose the formats, version, maximum batch size, TorchScript precision and TensorRT precision of the **Export** panel (those of the main block of `export_model.py` first) and press **Export At T = …**: `vision_training/deployment/explorer_export.py` exports the checkpoint and weights of the page at the saved temperature, in a separate process, one export at a time, into `<run>/Exports/Iteration_<n>[_EMA]_T<temperature>/` (written as `<folder>.partial` until it succeeds), a Triton model repository holding `image_classifier_<format>/<version>/` with its `class_thresholds.json`, `export_record.json` (run, iteration, weights, temperature, settings, the TorchScript precisions kept, date) and `export_log.txt`, the console output the page shows while it runs. Another temperature gets its own folder, the same temperature replaces its folder (an export in the other TorchScript precision first copying the earlier `model.pt` or `model_fp16.pt` of the same version next to its own file, which its `config.pbtxt` serves, the TorchScript column of the table then reading e.g. `FP16, FP32 kept`), and a failed export leaves its `.partial` folder with its console output until the next export of that temperature. The run folder and checkpoint of the evaluation must still be on disk.
5. Without the page, export the checkpoint with `use_saved_temperature` ([Export](#export)).

### Drawing The Figures Of A Research Report
Should a research report need its figures again, after a new evaluation of its kept checkpoint or a new saved temperature, `scripts/evaluation/draw_report_figures.py` draws them from the run folder into `<run>/Checkpoint_Evaluation/Iteration_<n>[/EMA]/Report_Figures/`. It reads every input and checks it before the first figure is drawn, so that a wrong input stops the run without writing anything.

| Setting | Meaning |
|---|---|
| `run_name`, `experiment_folder` | the run of the kept classifier, `<experiments>/<run name>`, the 35-class README classifier of `~/Data/tutorial_runs/readme_screenshots` in the main block |
| `iteration`, `weights` | the kept checkpoint, its weights `"trained"` or `"ema"` |
| `evaluation_folder_name` | `Checkpoint_Evaluation` (`evaluate_checkpoint.py`, where the temperature explorer saves the temperature) or `Evaluation` (the training) |
| `class_display_names` | the name shown for every class, in the order of the curves or panels, `None` for the class names in class index order |
| `sample_kind`, `split_name` | written in the titles, e.g. "on the 50,000 test images" |
| `minimum_recall` | the left end of the recall axis, 0, or close to 1 when every threshold keeps nearly every sample (e.g. 0.98) |
| `group_manifest_path` | a JSON Lines manifest, one object per sample, giving the group of every training and evaluated sample (such as the camera or vehicle of every frame), `None` to skip the groups chart |
| `file_name_key`, `group_key`, `group_plural` | the keys of the file name and of the group in the manifest, and the groups in the plural (`"file_name"`, `"camera"`, `"cameras"` in the main block) |
| `training_labels_path` | the `labels.json` of the training split, `None` for the one of the `train_data_folder` of the run |
| `predictions_path` | the predictions of the evaluation, `None` for its `predictions.csv` (`save_predictions = True` in `evaluate_checkpoint.py`) |

It writes, drawn in the Inter font of the reports when it is installed in `/usr/share/fonts/opentype/inter`:
- `confusion_matrix.png`: the trainer's drawing, its title giving the accuracy and the balanced accuracy in percent.
- `precision_recall_curves.png`: the curve of every class with the point of its threshold at every target precision, the threshold rounded down to the fewest decimals (four at least) selecting the same samples, at the temperature of `saved_temperature_thresholds.json` (the scores of the exported model). The saved thresholds must be exactly those found again from `class_logits.npz` at that temperature; without saved file, the curves are drawn at a temperature of 1 with a warning. Two classes share one plot (the first one in red), more classes get one panel each, three per row.
- `training_curves.png`: the balanced accuracy of the trained and averaged weights at every checkpoint evaluated during the training (`Tensorboard/Test`, `Tensorboard/Test_EMA`), the kept checkpoint ringed with its value, drawn like the `evaluation_curves.png` of a run (`save_evaluation_curves_figure` of `vision_training/evaluation/evaluation_curves.py`) without the window of the moving average.
- `seen_and_never_seen_groups.png`, with a manifest: the balanced accuracy on the samples of the groups having at least one training sample and on those of the groups never seen, with their numbers of groups and samples. The predictions must give the confusion matrix of the evaluation exactly, and every training and evaluated sample needs its group.
- `report_values.json`: the numbers shown, to fill the tables of the report: accuracy, balanced accuracy and recall of every class, the temperature, the written and exact threshold, recall and precision of every class at every target precision, the balanced accuracy of the kept checkpoint during the training, and the groups, classes, samples and recall of the seen, never seen and all samples.

<p align="center">
  <img src="../../README/report_precision_recall_curves.png" width="90%" />
</p>

<p align="center">
  <img src="../../README/report_training_curves.png" width="80%" />
</p>
<p align="center"><em>The precision-recall curves and the training curves drawn for the classifier above at iteration 2000 (test balanced accuracy 81.65 %), the curves shown for the first nine of its 35 classes (the script draws one panel per class): a dot marks the threshold reaching 99 % precision, reached by these nine and never by trapezoid, hollow parallelogram and hollow kite, at a logit temperature of 1 since no temperature was saved. The confusion matrix of the report is the one shown above; the chart of the groups needs a group manifest, which the synthetic shapes do not have.</em></p>

### Scoring A Large Split
Say a rare class must be mined from a split far larger than a test split: `scripts/evaluation/score_image_split.py` scores every image of it with one checkpoint, without the metrics, error images and second weights of `evaluate_checkpoint.py`. Its settings are command line options, not variables of its main block.

| Option | Meaning |
|---|---|
| `--experiment_folder`, `--iteration` | the run folder and the scored checkpoint |
| `--use_averaged_weights` | scores with the moving average of the weights saved with the checkpoint, refused for a run without one |
| `--test_data_folder` | the scored split, a `labels.json` and its `data/` folder, every image given a class of the run (a placeholder class for unlabelled images) |
| `--output_path` | the predictions file, appended to when it exists |
| `--precision` | `float32` (default) or `bfloat16` autocast, the precision of the training |
| `--batch_size`, `--number_workers` | the images per forward pass (128) and the loader worker processes (12) |
| `--time_budget_hours` | the time after which no new batch is scored, `None` (default) for none |
| `--compare_precisions` | first scores the labelled test split of the run in both precisions and prints their time, the largest difference of probability, the number of images whose predicted class changes and the balanced accuracy of each; without `--test_data_folder` it stops there |

The predictions file has the columns of the `predictions.csv` of `evaluate_checkpoint.py`, `sample` (the file name of the image relative to `data/`) and one `probability_<class>` per class written with nine significant digits, one row per image in the order of the split, written batch by batch. A launch on an existing file scores only the images it does not hold, after checking that it names the same columns and that its last row is complete: a file of other classes, a row lacking a value or a file ending without a line break, as left by an interrupted write, is refused before anything is written. The header is written only by the launch that creates the file. The trainer of the run is built on the split and writes its `training_configuration.yaml`, `label_dictionary.json`, `data.json` and `Tensorboard/` into `<run>/Checkpoint_Evaluation_split_<parent folder>_<split folder>/` (`Checkpoint_Evaluation_score_image_split_test/` for the precision comparison), so that no file of the run or of its evaluations changes.

## Comparing Launches
To decide which setting of a stage of a hyperparameter experiment to keep, `scripts/evaluation/collect_experiment_results.py` gathers every launch of the experiment into one CSV file. Every launch trains in its own `run_<n>` folder, a repeat launch in `run_<n><letters>` (`run_23b`).

| Setting | Meaning |
|---|---|
| `task` | the experiment, `"classification"`, `"sequence_labelling"` or `"object_detection"`, which chooses the folder, the metric and the swept settings below |
| `experiment_folders` | the folder of the `run_<n>` folders of every experiment: `<repository>/hyperparameter_experiments` for the classification, its `sequence_labelling/` and `object_detection/` subfolders for the others |
| `metrics`, `metric_columns` | the compared test curve of every experiment, `"Accuracy"`, `"Macro F1"` and `"mAP@[0.5:0.95]/area"`, and the prefix of its columns, `accuracy`, `macro_f1` and `map_50_95` |
| `number_last_evaluations` | the last evaluations averaged in a second mean column, 4 |
| `output_path` | the CSV file, `<experiment_folder>/results.csv` |

Every row is a launch and its weights (`evaluated_weights`: `trained`, or `ema` from `Tensorboard/Test_EMA` for a launch keeping a moving average). It gives the metric at every evaluated iteration, its mean, the mean of the last evaluations (`<metric column>_last_<n>_mean`, which ignores how fast a launch starts), whether the launch finished, the swept settings of the experiment (`TASK_SWEPT_SETTINGS`) and the run folder. The launches of the same settings and weights are grouped in the order of their first launch, followed by a `mean` row. Settings given as lists, such as the `[layers, channels]` of the mask heads or the backbone blocks of their skips, which the driver of the instance segmentation experiment adds to the swept settings of the detection, are compared by their values (`make_hashable`) and written as the configuration gives them (`[4, 64]`). A launch still training leaves its future cells empty, so the file can be written at any time.

<p align="center">
  <img src="../../README/comparing_launches_flow.png" width="100%" />
</p>

Before trusting a gap between two settings, compare it with the spread of the launches of each: in the repeated squeeze-and-excitation stage below, six launches per setting overlap, and the means differ by 0.21 points for a standard error of 1.11 points.

<p align="center">
  <img src="../../README/launch_comparison_stage_figure.png" width="80%" />
</p>

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
| `curves_in_percent`, `number_decimals`, `output_path` | whether the curves that are rates are written as the percentages of the reports, `89.80 %` (`True` in the main block), the decimals of the other curves (3): the mean true frame rank, effective rank, standard deviation, proposals per image and keypoint pixel error of `NON_RATE_CURVE_NAMES`, or every curve with `False`; and a Markdown file receiving the tables (`None` to print only) |

Say you compare two of the four runs of the main block, the 9-class synthetic shapes classifier of the README screenshots trained at two learning rates, with `table_grouping = "columns"`, the learning rate as first column and the recalls of two of its classes:

```text
| Learning Rate | Best Iteration | Balanced Accuracy | Recall square | Recall circle | Validation Images | Train Images |
|---|---|---|---|---|---|---|
| 0.0002 | 1500 | 82.64 % | 62.11 % | 70.23 % | 50,000 | 500,000 |
| 0.001 | 600 | 41.61 % | 87.35 % | 100.00 % | 50,000 | 500,000 |
```

A run not evaluated yet is printed in yellow and left out. A missing curve or count stops the script, listing the available ones.

### Drawing The Comparison Of Every Stage
After a stage of a hyperparameter experiment, `scripts/evaluation/plot_experiment_comparisons.py` draws its figure next to its experiment plan, from the copy of its results file in `documentation/architecture_benchmarking` (`results.csv` for classification, `<task>_results.csv` for the others) and its entry in `experiment_comparisons.yaml`.

| Setting | Meaning |
|---|---|
| `registry_path` | the registry of the compared stages, `documentation/architecture_benchmarking/experiment_comparisons.yaml` by default, the results files being read from its folder |
| `output_folder` | the folder of the figures, one subfolder per experiment, `documentation/architecture_benchmarking/figures` by default |
| `experiment`, `stage` | the experiment (`"classification"`, `"sequence_labelling"`, `"object_detection"`, `"instance_segmentation"`) and stage (`"3"`, `"2b"`, `"repeat"`) to draw alone, `None` for every stage |

Every entry of the registry gives the results file, the stage and its title, the iteration and weights the plan compares (`last_iteration`, `compared_weights`), the verdict, and the compared settings (`arms`): their label, their launches as written in the results file, whether they are the reference, and `trained_as_averaged` for a setting trained without moving average whose trained weights stand in the column of the averaged weights. A figure has a column per kind of weights evaluated for every launch of the stage; every launch is a thin line, the mean of the launches of every setting a thick one labelled at the compared iteration, and the lower panel gives the gap of every setting to the reference. Every compared metric (`PERCENT_METRICS`: the accuracy of the classification, the macro F1 of the sequence labelling, the mAP of the detection and the mask mAP of the instance segmentation) is written as the percentages of the reports: labels such as `85.81 %` on axes such as `86 %`, gaps in points such as `+0.95 points`, the gap axis named `Gap to <reference>` with `(points)` on a line of its own. The verdict under the title is wrapped by `wrap_verdict`, which keeps every number with its unit (`85.81 %`, `+0.95 points`) on one line. A stage without exactly one reference stops the script.

### Drawing The Mask Head Designs
If one wishes to see every convolution before the mask of the designs of stages 3 and 3b of the instance segmentation benchmark, with their cost, `scripts/models/instance_segmentation/draw_mask_head_designs.py` draws them one under the other:

```bash
uv run python -m scripts.models.instance_segmentation.draw_mask_head_designs --configuration_path=vision_training/configurations/instance_segmentation/configuration_synthetic_shapes.yaml --output_path=/tmp/mask_head_designs.png
```

Any instance segmentation configuration can be given, the counts following its settings. The figure of the [instance segmentation plan](../architecture_benchmarking/instance_segmentation_experiment_plan.md), `documentation/architecture_benchmarking/figures/instance_segmentation/mask_head_designs.png`, was drawn from the launch file of the benchmark, a configuration not tracked by git. Every design of its `DESIGNS` (the reference `[3, 32]`, stage 3 `[4, 64]`, and on it the image branch of design A, the U-Net skips of design B and the fusion convolution alone of the control) is built from the configuration file with its settings, as the training builds it, the finer inputs a design leaves out being turned off whatever the configuration holds; the figure reads its layers, channels, strides and residual shortcuts from the built mask heads, and its multiply-accumulates per box are counted on those layers by forward hooks on one box, printed with the parameters of the mask heads and the fine inputs.

## Clustering Image Embeddings
`scripts/embedding/cluster_image_embeddings.py` embeds images with a trained run and clusters them, to choose diverse images from a dataset of near-identical video frames. Its main block embeds the 1,417 frames of the test split of the synthetic moving shapes (`~/Data/sequence_data/synthetic_moving_shapes/test`, one group per sequence) with block 6 and the hidden layer of the 35-class README classifier at iteration 2000, and clusters them with DBSCAN.

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
| `seed`, `mosaic_grid_size`, `output_root` | the draws, the images per mosaic row, and the output root, `~/Data/embeddings` by default |

An asked layer outside the run stops the script before any inference, with the valid range. The images are read with the preprocessing of the run.

HDBSCAN runs on a single core, its time growing with the square of the images and with the dimensions. On 31,805 images of block 6, it took 116 seconds at 200 dimensions and 487 at 800, the clusters agreeing closely (adjusted Rand index of 0.81 and more). The timings of every layer and setting are at the top of the script.

<p align="center">
  <img src="../../README/embedding_clusters.png" width="100%" />
</p>
<p align="center"><em>A run on 720 synthetic shapes, one folder for each of nine of the 35 classes: the clusters follow what the layer of the classifier separates, one cluster for each of six shapes, the stars split in two and the rectangles shared between a cluster of their own and the cluster of the squares.</em></p>

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
| `r"^(camera_\d+_\d{4}-\d{2}-\d{2})_"` | `camera_3_2026-09-25_frame_0040.jpg` | `camera_3_2026-09-25` |
| `r"^([^/]+)/"`, the subfolder (the default, the sequence of a frame) | `test_000012/000003.png` | `test_000012` |
| `r"^(video_\d+)_\|^(camera_\d+)/"` | `video_3_frame_1.png`, `camera_2/frame_1.png` | `video_3`, `camera_2` |
| `r"video_\d+"` | `camera_2/video_7_frame_1.png` | `video_7` |

## Deployment

### Export
`vision_training/deployment/export_model.py` first refuses a format other than `torchscript` and `onnx`, a repeated format, a `tensorrt_precision` other than "FP32", "FP16" and None and a `torchscript_precision` other than "FP32" and "FP16" (`check_export_settings`), so that a misspelled format such as `tensorrt` is never served as plain ONNX Runtime. With `torchscript_precision` "FP16", `convert_to_half_precision` of `export_formats.py` copies the inference model on the GPU, converts the weights of its convolution, linear and attention layers to FP16, keeps its normalization layers, other parameters and buffers in FP32, runs its LSTM, GRU and RNN layers in FP32 without autocast (`FullPrecisionRecurrentLayer`, the tracer recording none of the casts autocast makes around them), and wraps it to run under FP16 autocast and return FP32 outputs; the traced file, `model_fp16.pt` (`TORCHSCRIPT_FILE_NAMES` of `triton_configuration.py`) so that it can sit next to the `model.pt` of an FP32 export, keeps the input names and the FP32 inputs and outputs, its `config.pbtxt` naming it in a `default_model_filename` line, the only difference from the FP32 configuration. Its random batch check compares it with that eager FP16 copy, whose GPU kernels may round otherwise, every tolerance of the task multiplied by `HALF_PRECISION_TOLERANCE_FACTOR` (100) and the shares of equal outputs kept, then prints its differences from the FP32 eager model for information. It writes every model with its `config.pbtxt` into a staging folder next to the model repository (`.export_staging_<random>/`, removed at the end), checks it against the PyTorch model, and moves the models into the repository only once every format matches: a failed check raises and leaves the served models as they were, and a checked export replaces the `<model>/<version>/` folder of an earlier export of that version whole, the other versions kept. With a TensorRT precision, the `config.pbtxt` of the ONNX model, written once its files are, caches the TensorRT engines under `trt_engine_cache_prefix` `<model>_<version>_<precision>_<hash>`, the hash of the files of its version folder (`build_tensorrt_engine_cache_prefix` of `triton_configuration.py`), so that an export of other weights builds its own engines instead of being served with those of an earlier export by a container that kept its cache. `use_averaged_weights` exports the moving average of the weights, raising an error for a checkpoint without it. For a classifier, `use_saved_temperature` builds the temperature of the `saved_temperature_thresholds.json` of the exported iteration and weights ([Logit Temperature](#logit-temperature)) into the model, a constant `logit_temperature` of the TorchScript and ONNX graphs dividing the logits before the softmax, and copies the file to `<model>/<version>/class_thresholds.json`. The file is looked for in `<run>/Evaluation/Iteration_<n>/[EMA/]` of the training and in `<run>/Checkpoint_Evaluation/Iteration_<n>/[EMA/]` of the evaluation script: the export stops when it is in neither or in both, or when it does not match the `class_thresholds.json` next to it or the number of classes of the model. `saved_thresholds_path` gives the file instead (the export button of the page gives the file of its evaluation), accepted only in one of these two folders of the exported run, iteration and weights. Without it, a classifier is exported with a temperature of 1, and the copy of an earlier export is removed. It is refused for the other tasks. The ONNX graph keeps none of the debugging metadata of the dynamo exporter (`strip_exporter_metadata` of `export_formats.py`: the stack traces, module paths and FX node text of every node, the original names of the values and weights and the signature of the traced program), so that a shipped `model.onnx` holds no path of the exporting machine and is 8 to 17 times smaller (the ten tutorial runs). Every model also gets the input contract of the run, in the `metadata_props` of `model.onnx` and as `<model>/<version>/preprocessing.json` for both formats, the export stopping before writing any file when the contract cannot describe the input of the run ([Input Contract](#input-contract)). The inputs and outputs of every model are listed in the [README](../../README.md#export).

**Semantic segmentation (SEG).** The outputs are at the network resolution: `restore_prediction_to_original_image` (`vision_training/data/preprocessing.py`) brings them back to the original image. A run trained on tiles takes one tile: the client scales the image by `Data.tiling.scale` and segments it with `segment_image_by_tiles` (`vision_training/models/segmentation/sliding_window.py`), given a function sending a batch of tiles to Triton; the scale, overlap, interpolation and padding of the tiles are in the input contract of the model (`input_resize` `tiles`, [Input Contract](#input-contract)).

**Instance segmentation (INS).** Every mask holds the probabilities of the detected class inside its box. The client brings the boxes back to the original image with `restore_boxes_to_original_image(..., clip=False)` (`vision_training/data/preprocessing.py`), since a box clipped at the border would squeeze its mask, pastes the masks with `paste_instance_masks` (`vision_training/models/instance_segmentation/mask_operations.py`), and keeps the pixels at or above `MaskHead.inference.mask_threshold`.

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

### Input Contract
Should a server need to build the input of an exported model from a photo, it reads the input contract written with the model instead of assuming a resize or a channel order. Every export writes it into the `metadata_props` of `model.onnx` and as `preprocessing.json` next to the model of both formats (`<model name>_<format>/<version>/`), with the keys and values defined in `vision_training/deployment/input_contract.py`:

| Key | Values | Meaning |
|---|---|---|
| `input_contract_version` | `2` | Version of the keys and values; a server refuses a version it does not know. |
| `input_color_order` | `BGR` | Channel order of the photo decoded as `cv2.imread` (`read_image`) decodes it: 3 channels of 8 bits, rotated by its EXIF orientation; a file of more bits per channel (a 16 bit PNG) has its values divided by 256, with a yellow warning the first time. |
| `input_resize` | `stretch`, `keep_ratio`, `tiles` | `stretch` (`Data.image_settings.keep_ratio` False): both sides resized to the input size. `keep_ratio` (True): the longest side resized to the input size and the other one to `max(1, round(side * size / longest side))` (Python `round`, half to even), then padded to the square. `tiles` (SEG with `Data.tiling.activated`): the photo scaled by `input_tile_scale`, padded to at least a tile, then cut into overlapping tiles of the input size, one input per tile. |
| `input_size` | a positive integer | Side of the square input in pixels, the side of a tile for `tiles`, `Data.image_settings.size`. |
| `input_interpolation` | `linear`, `area`, `none` | OpenCV resize of the uint8 photo: `cv2.INTER_LINEAR` (`stretch` and `keep_ratio`, `tiles` enlarged), `cv2.INTER_AREA` (`tiles` shrunk), or no resize (`tiles` at the scale 1). The bilinear resize of Pillow gives other pixels (58 % of them, by up to 49 levels, for a 600 x 400 photo resized to 384 x 256). |
| `input_padding_value` | `0` | Value of every channel of the padding pixels (`keep_ratio` and `tiles`). |
| `input_padding_placement` | `centred`, `bottom_right` | `centred` (`keep_ratio`): resized photo at the row `(size - resized height) // 2` and the column `(size - resized width) // 2`, an odd padding pixel going to the bottom or the right. `bottom_right` (`tiles`): the scaled photo padded on its bottom and right sides to at least a tile. |
| `input_tile_scale` | a positive number, `none` | `Data.tiling.scale`: every side of the photo scaled to `max(1, round(side * scale))`, half to even (`tiles` only). |
| `input_tile_overlap` | a non-negative integer, `none` | `Data.tiling.overlap`, the pixels shared by two neighbouring tiles: along a padded side of length L, the tiles start at 0, size - overlap, 2 (size - overlap)... below L - size, then at L - size, cut row by row (`tiles` only). |
| `input_layout` | `NCHW`, `NTCHW` | Batch, channels, height and width, the frames coming after the batch for the sequence tasks (M2O, M2M, CMP, SEL); the tiles of a photo form the batch. |
| `input_value_range` | `0-255` | The uint8 pixels cast to float32 without scaling. |

Every value is read from the configuration of the run or from the preprocessing code of the data loaders (`SegmentationDataset.cut_tile` and `segment_image_by_tiles` for the tiles), never written by hand. In case a run uses an input the contract cannot describe, the export stops before writing any file and names the setting: a change to `vision_training/data/preprocessing.py` (another default interpolation or padding value of `preprocess_image`, photos read as RGB by `read_image`, another placement in `compute_letterbox_region`), or to the scaling or padding of the tiles. The probabilities of the tiles are put back together by the client, averaged where tiles overlap ([Export](#export)).

To read the contract of an ONNX model, ONNX Runtime gives its entries as the custom metadata of the model, and `check_input_contract` refuses a missing key, an unknown key or a value the version does not allow:

```python
session = onnxruntime.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input_contract = session.get_modelmeta().custom_metadata_map
check_input_contract(input_contract=input_contract)
```

If you are looking for a server-side implementation, `build_input_from_contract` in `tests/deployment/test_input_contract.py` builds the inputs of a photo from the contract alone, and the tests check them pixel for pixel against `preprocess_image` on a landscape and a portrait photo for both resizes, and against the tiles of the evaluation of a segmenter trained on tiles.

### Test Split Parity
Should you need to know what an export costs in test metrics before it is served, `test_split_parity = True` in the main block of `export_model.py` scores every exported format on the whole test split of the run (`run_test_split_parity` of `vision_training/deployment/export_parity.py`), for every task. The random batch check of the export compares the FP32 formats with the eager model on random inputs only (TorchScript, and ONNX with ONNX Runtime on the CPU), never the TensorRT engines Triton builds.

- **Before anything is read**: the export stops when no Triton container can be started (`check_triton_available` of `vision_training/deployment/triton_server.py`: the `docker` command and the image `TRITON_IMAGE`, the one of `deployment/triton/docker-compose.yml`).
- **Rows**: the eager inference model of the export in the process first (its weights, averaged or not, and its logit temperature), the reference, then every format of the staging folder, served by a container on free ports (`export-parity-triton-<process>`, so that it never meets the deployed server or a benchmark) with its exported `config.pbtxt`, the ONNX model with its TensorRT precision and the TorchScript model in its precision (`TorchScript (FP16)` for an FP16 file). Every row runs even when the random batch check of a format failed.
- **Requests**: the test evaluation of the trainer of the task (`build_evaluation_trainer`, its tracking, predictions file, error images, per-class report and sample images off) has its `predict_batch` replaced by `predict_exported_batch`: batches of `maximum_batch_size`, the real frames of every sequence alone (M2O, M2M, SEL, the frame probabilities of the padding left at 0), the real frames of both members of every pair alone (CMP), the tiles of every image of a SEG run trained on tiles in requests of at most `maximum_batch_size` tiles. The outputs become the predictions of the trainer (`convert_inference_outputs`): the class probabilities, the representations, the detections with no proposal (the proposal scalars read 0, left out of `parity_summary.md`), the masks pasted by the INS trainer, and the keypoint candidates grouped by the KPT trainer (`group_candidates`). The detectors are scored at their deployment score threshold (`Detector.inference.score_threshold`, the one of the export), not at `evaluation.score_threshold`.
- **Samples**: every request of a served row also goes through the eager model, and every sample is compared with the comparison and FP32 tolerances of the random batch check (`compare_request_outputs`), an FP16 TorchScript row included, with its largest difference: of a class probability, of a representation value, or of the detection or candidate scores sorted by value, so that two detections swapped by a rounding are no difference. Every sample is named as in the evaluation outputs (`get_test_sample_name`: the image, sequence or pair name), every tile of a tiled SEG image under the name of its image, and the k-NN reference images of SSL, training images also sent to the served model, under their own `reference` split (`encode_named_reference_images`).
- **Stored evaluation**: the `evaluation_summary.csv` of `<run>/Checkpoint_Evaluation/`, else of `<run>/Evaluation/`, at the exported iteration (`ema_` columns for the averaged weights), shown next to the rows, the eager row being expected to reproduce it. It is left out for DET and INS (another score threshold) and for a classifier exported at a temperature other than 1; the rank metrics of SSL differ by construction, the encoder returning normalized representations.
- **Failed rows**: a model Triton refuses to load or fails to run (the TensorRT keypoint detector of the tutorials crashed the server) is recorded with its error, the log of the server up to the failure kept as `triton_server_<format>_failure.log`, and the server is started again for the other rows.
- **Outputs**: `<run>/Export_Parity/Iteration_<n>[_EMA][_T<temperature>]_<precisions>/` (`get_parity_folder`), the temperature built into a classifier (`T1` without a saved temperature, no temperature for the other tasks) and the precision of every exported format in the order TorchScript, ONNX (`TorchScript-FP32` or `TorchScript-FP16`, then `TensorRT-FP32`, `TensorRT-FP16` or `ONNXRuntime-CUDA`), e.g. `Iteration_5000_EMA_T3_TorchScript-FP16_TensorRT-FP32`, replaced only by the next parity of the same export: `parity_summary.md` (installation, random batch check, every metric of every row with its change against the eager row, in points for a rate, the stored evaluation, the agreement of the samples and the 10 samples of largest difference of every row, the failed rows), `parity_summary.json`, `parity_samples.csv` (one line per compared sample of every served row: split, identifier, name, agreement, largest difference, mismatches), `row_evaluations/<row>/` (the `Results.txt`, summary and iteration folder of every row) and `triton_server.log` (with a `triton_server_<format>_failure.log` per failed row). The console prints the headline metric of every row, a change of at least 0.5 points in yellow. Both summaries are copied into the version folder of every installed model, which Triton ignores.
- **Installation**: decided by the random batch check alone, as without the parity: a failed check raises once the summary is written, and an expected drop of a reduced precision never stops an export.

### FP16 TorchScript Trial
In case you want the cost of an FP16 TorchScript file of a classifier without starting Triton, with the deployed threshold decisions it moves, `scripts/deployment/check_half_precision_parity.py` converts it as the export does (`convert_to_half_precision`, `torchscript_precision` "FP16" of [Export](#export)), traces it and scores it on the whole test split against the eager FP32 model, classification only, without timing it; the export with `test_split_parity` scores the file served by Triton. Its main block sets the run, `iteration`, `use_averaged_weights`, `use_saved_temperature`, `saved_thresholds_path` and `maximum_batch_size`, as in `export_model.py`.

```bash
uv run python -m scripts.deployment.check_half_precision_parity
```

- **Model**: the inference classifier of the export, with the temperature of its saved thresholds when `use_saved_temperature` is set, converted by `convert_to_half_precision`; the input stays the FP32 batch of the input contract and the FP16 logits meet the FP32 temperature, so that the division and the softmax run in FP32, FP16 values near 1 lying about 0.0005 apart.
- **Rows**: the eager FP32 model, then the traced FP16 file in the process, each scored by `evaluate_parity_row` of the [test split parity](#test-split-parity), every FP16 request also going through the eager model.
- **Outputs**: `<run>/Half_Precision_Parity/Iteration_<n>[_EMA]/`, replaced by the next trial of the same checkpoint and weights: `model_fp16.pt`, `model_fp32.pt` (the size reference), the evaluation files of both rows, and `half_precision_parity.md` and `.json`: Balanced Accuracy, Accuracy and Macro F1 of both rows with the change in points, the images whose predicted class or deployed threshold decision changed, those within the FP32 tolerances of the export, the largest and mean probability differences and the sizes of both files.

### Throughput Measurement
`vision_training/deployment/run_inference.py` times the images of `vision_training/deployment/inference_data/`. The image tasks send one request per image, letterboxed like the loaders of a run trained on whole images; a SEG run trained on tiles is refused, a photo needing many tiles per request. The sequence tasks send the whole folder as one sequence, of 2 to `maximum_sequence_length` images, CMP splitting it into a before and an after half. SEL decodes every sequence locally, and queries Triton through its ensembles.

### Serving Benchmark
Should you need to compare the serving paths of trained runs on an idle GPU, two drivers of `scripts/deployment/`, run one after the other from the repository root, measure them; their settings are variables of their main blocks, their outputs stay outside the repository in `benchmark_folder` (`~/Data/serving_benchmark`), and the [serving benchmark plan](../architecture_benchmarking/serving_benchmark_plan.md) records the runs, the method and the results.

```bash
uv run python -m scripts.deployment.benchmark_serving_formats
uv run python -m scripts.deployment.evaluate_served_formats
```

| Script | What it does | Settings | Output |
|---|---|---|---|
| `benchmark_serving_formats.py` | exports every run of `models` with `export_experiment` into one Triton model repository per precision setting, times the inference model in the process (eager, TorchScript FP32 and FP16), then serves every exported model alone in a Triton container (`TRITON_IMAGE` of `vision_training/deployment/triton_server.py`) and times it through system shared memory with `perf_analyzer` of `SDK_IMAGE` and over gRPC with the client of `run_inference.py`, with the first requests (TensorRT engine builds) and the memory taken | `models` (run folder, iteration, test split, frames of a sequence), `precision_settings` (ONNX Runtime CUDA with the FP32 TorchScript model, TensorRT FP32 and FP16, and `torchscript_fp16`, the FP16 TorchScript model alone, exported with `torchscript_precision` "FP16" and named `triton_torchscript_fp16` and `in_process_torchscript_fp16` in the results; BF16 is refused by the ONNX Runtime backend of Triton), `batch_sizes`, `load_groups` (server settings, transports, batch sizes and requests in flight), `maximum_batch_size`, the in-process iterations, and `run_exports`, `run_in_process`, `run_triton` to run some phases only | `results.csv`, one line per measurement, read again at a relaunch so that a stopped sweep restarts where it stopped |
| `evaluate_served_formats.py` | sends the test split of every run through the eager inference model, then through every model of the repositories of the first script, each scored by the test evaluation of the trainer of its task with the requests of the [test split parity](#test-split-parity) (`evaluate_parity_row`, every task), so that the gap to the eager row measures the format and the precision only | `models`, `precision_settings`, `maximum_request_size` (the largest batch of the served models and of the test loaders) | the evaluation files of every row (its own `Results.txt` included) and `test_metrics.csv`, one line per scalar of every row, the rows already measured skipped at a relaunch |

In case you rerun them to compare with the plan, the three test splits of their main blocks were drawn again after the benchmark of 2026-11-17 (the classification split now holds 35 categories, of which the loaders keep the 9 of the run): a new run measures the same models on other images, so its scores are not those of the plan.

### Export Size Check
Should the exports, their libraries or the formulas of `vision_training/deployment/export_size_estimates.py` change, `scripts/deployment/check_export_size_estimates.py` compares the [Export Sizes](../../README.md#export-sizes) panel with real files and prints its constants fitted again. For every run it loads the last checkpoint, computes the estimates of the panel on `--device` with the optimizer and moving average of the run, writes the PyTorch weights in FP32 and FP16 with `torch.save` and the ONNX files with `export_onnx` from the model loaded again on the CPU, traces the FP16 TorchScript file as the export does (`convert_to_half_precision`, from the model loaded on the GPU, not measured without one), and reads the size of the checkpoint. With `--build_tensorrt_engines` it then serves the ONNX files in the Triton container of the [serving benchmark](#serving-benchmark), once with the `config.pbtxt` of TensorRT FP32 and once with that of FP16, serves every model alone with a request at batch 1 and one at the largest batch up to 16 whose request stays below 1 GB (3 for the comparator of two sequences of 50 frames of 512 pixels, gRPC refusing messages above 2 GB), and sums the engines ONNX Runtime added to the cache, the check stopping when a model added none; it records the seconds of these first requests, which build the engines, and whether every engine starts with the `trt_engine_cache_prefix` of the model, a red line naming a model whose engines do not. A model whose load or first requests fail is reported in red with the error, the log of the failure kept (`triton_<precision>_<model>_failure.log`), the engines it wrote before failing still measured, and the server started again for the next models: on 2026-11-22 the keypoint detector crashed Triton 26.01 in both precisions and the four sequence models failed to build their FP32 engine beyond batch 1 (item 41 of `documentation/suggestions.md`). The first requests take from 10 seconds to a minute per model and precision, but 49 minutes for the instance segmenter (43 for its batch-16 engine): about two hours for the ten runs, to run on an idle GPU.

```bash
uv run python -m scripts.deployment.check_export_size_estimates
uv run python -m scripts.deployment.check_export_size_estimates --build_tensorrt_engines
```

| Option | Default | Meaning |
|---|---|---|
| `--experiment_folders` | the ten runs of `~/Data/tutorial_runs` of `DEFAULT_EXPERIMENT_FOLDERS`, one per task | run folders to check |
| `--output_folder` | `~/Data/export_size_check` | the files of every run (`weights_fp32.pt`, `weights_fp16.pt`, `onnx/`), the Triton repositories, engine caches and logs, and `export_size_check.csv` |
| `--device` | `cpu` | device of the model during the estimates, `cuda` giving the time of a launch |
| `--build_tensorrt_engines` | off | also build and measure the TensorRT engines |

For every run it prints each format with its estimate (a range for TensorRT), its real size and the error, an engine being `inside` its range or outside by a percentage, then the calibration: `STATE_DICTIONARY_BYTES_PER_TENSOR` and `ONNX_GRAPH_BYTES_PER_OPERATOR` fitted on the runs (medians), the TensorRT bytes per weighted and other operator fitted by least squares on the relative errors of the overhead of every engine built (its size less its weights; the engines of a model Triton failed to serve included), with the spread of the real overheads around the fit and around the current constants, which gives `TENSORRT_OVERHEAD_RANGE`, and the largest error of every format, the margins of the panel. `export_size_check.csv` holds one line per run: the forward pass of the probe (constants read, of which batch normalization, weighted and other operators), then every estimated and real size in bytes, with the seconds of the engine builds, whether the engines carry their prefix, the largest batch sent and the error of a precision Triton failed to serve, empty when not measured. The constants and margins of `export_size_estimates.py` are changed by hand from this output, their comment giving the date and the library versions of the fit (2026-11-21, PyTorch 2.14 and onnxscript 0.7.2; the TensorRT constants on 2026-11-22 from the 20 engines Triton 26.01 built for the ten runs).
