# Visual Language Model Spark

## Table Of Content
- Description
- Prerequisites
- Configurations
- Run Names
- MLflow Tracking
- TODO

## Description
This repository contains an object detection pipeline built with PyTorch, supporting NVIDIA DALI and native PyTorch data loading backends.

## Prerequisites
- PyTorch
- NVIDIA DALI
- OpenCV
- MLflow, only for the tracking of the runs (`ExperimentConfiguration.tracking`)

## Configurations
Every task has its own configuration files in `vision_training/configurations/<task>/`: a `configuration_example.yaml` and a `configuration_synthetic_*.yaml`. Both files of a task have the same keys, only their values differ.

A configuration file holds only the keys its task reads, and every one of them is required. To start a new run, copy the file of the right task, not the file of another task. The check of the task lists any missing key, and a section used only by another task is ignored.

The tables below list only the keys that are not in every task. Every other key (the rest of `ExperimentConfiguration`, including `evaluation.per_class_report`, `evaluation.number_sample_images` and `tracking`, `Optimization`, `Data` and `Backbone`) is the same in all tasks. A section name alone (e.g. `RegionProposal`) means the whole section with all its keys. Tasks are written with their run name codes:

| Code | Task |
|---|---|
| DET | object_detection |
| CLS | classification |
| M2O | sequence_classification |
| M2M | sequence_labelling |
| CMP | sequence_comparison |
| SEG | semantic_segmentation |

### Keys Per Task
| Key | DET | CLS | M2O | M2M | CMP | SEG |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `RegionProposal` | ✓ | | | | | |
| `Anchors` | ✓ | | | | | |
| `Detector` | ✓ | | | | | |
| `ExperimentConfiguration.evaluation.score_threshold` | ✓ | | | | | |
| `ExperimentConfiguration.evaluation.top_k` | | ✓ | ✓ | ✓ | ✓ | |
| `ExperimentConfiguration.evaluation.target_precision` | | ✓ | ✓ | ✓ | ✓ | |
| `ExperimentConfiguration.evaluation.save_predictions` | | ✓ | ✓ | ✓ | ✓ | |
| `ExperimentConfiguration.evaluation.error_dumps` | | ✓ | ✓ | ✓ | ✓ | |
| `ClassificationHead` | | ✓ | ✓ | ✓ | ✓ | |
| `TemporalModel` | | | ✓ | ✓ | ✓ | |
| `Data.sequence_settings` | | | ✓ | ✓ | ✓ | |
| `Data.sequence_settings.window_stride` | | | ✓ | ✓ | | |
| `Fusion` | | | | | ✓ | |
| `SegmentationDecoder` | | | | | | ✓ |
| `SegmentationLoss` | | | | | | ✓ |
| `ExperimentConfiguration.evaluation.save_predicted_masks` | | | | | | ✓ |

### Keys Per Group Of Tasks
| Tasks | Keys they share that the other tasks do not have |
|---|---|
| DET | `RegionProposal`, `Anchors`, `Detector`, `ExperimentConfiguration.evaluation.score_threshold` |
| CLS, M2O, M2M, CMP | `ClassificationHead`, `ExperimentConfiguration.evaluation.top_k`, `target_precision`, `save_predictions`, `error_dumps` |
| M2O, M2M, CMP | `TemporalModel`, `Data.sequence_settings` (`sequence_length`, `frame_stride`, `short_sequences`, `batch_padding`) |
| M2O, M2M | `Data.sequence_settings.window_stride` |
| CMP | `Fusion` |
| SEG | `SegmentationDecoder`, `SegmentationLoss`, `ExperimentConfiguration.evaluation.save_predicted_masks` |

### Values Per Task
Some keys are in several tasks but do not accept the same values in each. The check of the task refuses a value that its task cannot use. Everything not listed here accepts the same values in every task that has the key.

| Key | DET | CLS | M2O | M2M | CMP | SEG |
|---|---|---|---|---|---|---|
| `Data.DALI.activated` | `True` or `False` | `False` | `False` | `False` | `False` | `False` |
| `Data.label_mapping.mode` | `binary` or `multi_class` | `multi_class` | `multi_class` | `multi_class` | `multi_class` | `binary` or `multi_class` |
| `Data.data_augmentation.independent_frames` | `False` | `False` | `True` or `False` | `True` or `False` | `True` or `False` | `False` |
| `Data.data_augmentation.shuffle_frames.activated` | `False` | `False` | `True` or `False` | `True` or `False` | `True` or `False` | `False` |
| `Backbone.detection_convolution_block_indices` | at least one block | `[]` | `[]` | `[]` | `[]` | `[]` |
| `Backbone.normalization.feature_map_normalization` | any except `sequence` | any except `sequence` | any | any | any | any except `sequence` |
| `ClassificationHead.global_pooling` | | `average`, `max` or `flatten` | `average` or `max` | `average` or `max` | `average` or `max` | |
| `ClassificationHead.output_mode` | | `multi_class` or `multi_label` | `multi_class` | `multi_class` | `multi_class` | |
| `TemporalModel.type` | | | all 7 types | `gru`, `lstm`, `transformer`, `temporal_convolution` | all 7 types | |
| `TemporalModel.causal` | | | `False` | `True` or `False` | `False` | |

An empty cell means that the task does not have the key. The 7 temporal types are `mean_pooling`, `max_pooling`, `attention_pooling`, `gru`, `lstm`, `transformer` and `temporal_convolution`. M2M needs one output per frame, which the three poolings do not give.

For SEG, the keys of `SegmentationDecoder.convolutions` must be the backbone blocks whose output before pooling is a skip connection: blocks 1 to N-1 of a backbone of N blocks, and block N too with `Backbone.last_max_pooling: True`. The categories of a `multi_class` mapping file are the pixel values of the masks (1 to 254, 0 being the background and 255 the unlabelled pixels).

M2O and M2M have exactly the same keys. The number of blocks in `Backbone.convolutions` changes from one example to another, but this is a value, not a different key.

### Evaluation Outputs
Every evaluated checkpoint is reported in the console and in a block of `<run>/Results.txt` with the same layout: a header `Iteration <n> · <evaluated samples> · <date and time>`, the global metrics, the notes (e.g. the error images) and the tables of the task, the per-class table being sorted worst first. Its files are written in `<run>/Evaluation/Iteration_<n>/`. `ExperimentConfiguration.evaluation.per_class_report` set to `False` keeps the global metrics only, for runs with many classes: the files written in every case keep the details.

| Output | Tasks | Written |
|---|---|---|
| Global metrics and notes (console, `Results.txt`), global Test scalars (TensorBoard) | all | every evaluation |
| Per-class table (console, `Results.txt`), per-class Test scalars (TensorBoard) | all | with `per_class_report` |
| `class_metrics.csv`: one row per class (samples, precision, recall, F1, threshold and recall at the target precision; detection: ground truths and average precisions; segmentation: pixels, IoU, Dice, precision, recall) | all | every evaluation |
| `class_thresholds.json` | CLS, M2O, M2M, CMP | every evaluation |
| `confusion_matrix.csv` (pixels for SEG) | CLS (multi-class), M2O, M2M, CMP, SEG | every evaluation |
| `confusion_matrix.png` and its Test image (TensorBoard), the shares of the true class only for SEG | CLS (multi-class), M2O, M2M, CMP, SEG | with `per_class_report` |
| `predictions.csv` | CLS, M2O, M2M, CMP | with `save_predictions` |
| `Errors/<TRUE>-TO-<PREDICTED>/` | CLS (multi-class), M2O, M2M, CMP | with `error_dumps.activated` |
| `Samples/<sample>.png` and its Test image `Samples/<sample>` (TensorBoard): the same test samples at every evaluation with their targets and predictions. CLS, M2O, CMP: the input (frame grid of the window, before member above after member for CMP) captioned `true -> predicted probability`, green when right and red when wrong. M2M: the frame grid, every frame captioned `true/predicted` with `X` on the wrong ones. DET: ground truth boxes next to the detections above `Detector.inference.score_threshold`. SEG: input, true mask, predicted mask and error map | all | with `number_sample_images` above 0, whatever `per_class_report` |
| `Predicted_Masks/<image>.png`: class index of every pixel at the network resolution | SEG | with `save_predicted_masks` |
| `Evaluation/evaluation_summary.csv` (`scripts/evaluation/evaluate_checkpoint.py` with `iteration = "all"`) | all | per-class columns with `per_class_report` |

The `number_sample_images` test samples are chosen once from the test split, so that the same samples are shown at every evaluation and after a resume. For CLS, M2O, M2M and CMP they are spread over the classes first, one per class in turn (a window of M2M counts for the class of most of its frames), then evenly over the samples of each class. For DET and SEG they are evenly spread over the test images. A test split smaller than `number_sample_images` shows all its samples.

## Run Names
A training run is saved in `<ExperimentConfiguration.experiment_folder>/<run name>`. The run name is built from the configuration by `vision_training/configuration/run_name.py`: the same settings resume in the same folder, changed settings start a new folder.

The name is made of sections joined with `_`, the tokens of a section being joined with `-`. A flag token is left out when its setting is off, and the optimizer section is left out with plain Adam without weight decay. The sections always come in the order of the table below. Values in `<...>` are read from the configuration.

The training settings that can change between two resumptions of a run give no token: augmentation (including `shuffle_frames` and `independent_frames`), batch size, learning rate, warm-up, cosine schedule, gradient clipping, mixed precision, the dropout rates (`ClassificationHead.dropout_rate`, `TemporalModel.dropout_rate`, `Fusion.dropout_rate`, `SegmentationDecoder.dropout_rate`) and the losses of SEG (`SegmentationLoss`, whose class weights are set again from the configuration when a run resumes). A run can be resumed in its folder with changed values, and two runs differing only by these settings share a folder. The optimizer keeps its token, because the state of one optimizer cannot be resumed by another.

| Section | Token | Meaning (configuration key) | DET | CLS | M2O | M2M | CMP | SEG |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Task | `DET`, `CLS`, `M2O`, `M2M`, `CMP`, `SEG` | `ExperimentConfiguration.task` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Input | `IN` | `Backbone.normalization.input_normalization` is True | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Input | `PSN` | `Backbone.normalization.per_sample_normalization` is True | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Input | `KAR` | `Data.image_settings.keep_ratio` is True | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Input | `<size>` | `Data.image_settings.size` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Backbone | `SE`, `RES` | `Backbone.modules.squeeze_and_excite`, `residual_nets` are True | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Backbone | `INC<block>` | `Backbone.modules.inception` is True, from `inception_starting_block` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Backbone | `BN`, `LN`, `NN` | `Backbone.normalization.feature_map_normalization`: batch, layer, none | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Backbone | `SN` | `Backbone.normalization.feature_map_normalization`: sequence | | | ✓ | ✓ | ✓ | |
| Backbone | `B<layers>x<filters>.…` | `Backbone.convolutions`, one `<layers>x<filters>` per block | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Backbone | `PT` | `Backbone.pretrained.experiment_folder` is set | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Backbone | `FRZ` or `BLR<multiplier>` | `Backbone.pretrained.freeze` is True, or `learning_rate_multiplier` is not 1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Backbone | `FBN` | `Backbone.pretrained.freeze_normalization_parameters` is True | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Levels | `L<blocks>` | `Backbone.detection_convolution_block_indices` | ✓ | | | | | |
| Anchors | `A<anchors>.<smallest>-<largest>` | `Anchors.scales_and_ratios`: anchors per position, range of the scales | ✓ | | | | | |
| Region proposal | `RBN`, `RLN`, `RNN` | `RegionProposal.normalization.feature_map_normalization` | ✓ | | | | | |
| Region proposal | `R<layers>x<filters>` | `RegionProposal.head_configurations` | ✓ | | | | | |
| Detector | `DSE`, `DRES`, `DINC` | `Detector.modules` | ✓ | | | | | |
| Detector | `DBN`, `DLN`, `DNN` | `Detector.normalization.feature_map_normalization` | ✓ | | | | | |
| Detector | `D<pool>.<layers>x<filters>.<fully connected>` | `Detector.head_configurations` | ✓ | | | | | |
| Decoder | `UTC` or `UBL` | `SegmentationDecoder.upsampling`: transposed, bilinear | | | | | | ✓ |
| Decoder | `DSE`, `DRES` | `SegmentationDecoder.modules` | | | | | | ✓ |
| Decoder | `DBN`, `DLN`, `DNN` | `SegmentationDecoder.feature_map_normalization` | | | | | | ✓ |
| Decoder | `D<layers>x<channels>.…` | `SegmentationDecoder.convolutions`, one `<layers>x<channels>` per level from the deepest one | | | | | | ✓ |
| Decoder | `DPT`, then `DFRZ` or `DLR<multiplier>` | `SegmentationDecoder.pretrained` | | | | | | ✓ |
| Classes | `BIN` or `MC<classes>` | `Data.label_mapping.mode`: binary or multi_class, foreground classes of the label dictionary | ✓ | | | | | ✓ |
| Window | `T<frames>` | `Data.sequence_settings.sequence_length` (frames per member for CMP) | | | ✓ | ✓ | ✓ | |
| Window | `S<stride>` | `Data.sequence_settings.frame_stride` is above 1 | | | ✓ | ✓ | ✓ | |
| Temporal module | `MEAN`, `MAX`, `ATT<hidden size>` | `TemporalModel.type`: mean_pooling, max_pooling, attention_pooling | | | ✓ | | ✓ | |
| Temporal module | `GRU<layers>x<hidden size>`, `LSTM<layers>x<hidden size>` | `TemporalModel.type`: gru, lstm | | | ✓ | ✓ | ✓ | |
| Temporal module | `TRF<layers>x<hidden size>.<heads>` | `TemporalModel.type`: transformer | | | ✓ | ✓ | ✓ | |
| Temporal module | `TCN<layers>x<hidden size>.<kernel size>` | `TemporalModel.type`: temporal_convolution | | | ✓ | ✓ | ✓ | |
| Temporal module | `BI` | `TemporalModel.bidirectional` is True (gru, lstm) | | | ✓ | ✓ | ✓ | |
| Temporal module | `CAU` | `TemporalModel.causal` is True (transformer, temporal_convolution) | | | | ✓ | | |
| Temporal module | `TPT`, then `TFRZ` or `TLR<multiplier>` | `TemporalModel.pretrained` (not for mean or max pooling) | | | ✓ | ✓ | ✓ | |
| Fusion | `FCAT`, `FDIF`, `FXA<heads>` | `Fusion.type`: concatenation, difference, cross_attention | | | | | ✓ | |
| Fusion | `SNP` | `Fusion.sequence_normalization_scope` is pair, with the `SN` backbone | | | | | ✓ | |
| Head | `HAVG.<fully connected>` or `HMAX.<fully connected>` | `ClassificationHead.global_pooling` average or max, `fully_connected` | | ✓ | ✓ | ✓ | ✓ | |
| Head | `HFLAT.<fully connected>` | `ClassificationHead.global_pooling` flatten, `fully_connected` | | ✓ | | | | |
| Head | `LS<rate>` | `ClassificationHead.label_smoothing` is above 0 | | ✓ | ✓ | ✓ | ✓ | |
| Head | `CWB` | `ClassificationHead.class_weighting` is balanced | | ✓ | ✓ | ✓ | ✓ | |
| Head | `HPT`, then `HFRZ` or `HLR<multiplier>` | `ClassificationHead.pretrained` | | ✓ | ✓ | ✓ | ✓ | |
| Classes | `MC<classes>` | `ClassificationHead.output_mode` multi_class, classes of the label dictionary | | ✓ | ✓ | ✓ | ✓ | |
| Classes | `ML<classes>` | `ClassificationHead.output_mode` multi_label | | ✓ | | | | |
| Optimizer | `ADAMW` or `SGD` | `Optimization.optimizer` is not adam | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Optimizer | `WD<decay>` | `Optimization.weight_decay` is above 0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

For the detection heads and the anchors, a setting identical on every detection level is written once, otherwise the levels are joined with `+` in level order. The classes of SEG count the foreground classes. The loading settings (`Data.loading`) and the evaluation settings add no token either.

Examples, built from the configurations of the repository:

| Task | Run name |
|---|---|
| DET | `DET_IN-KAR-1024_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128_BIN` |
| CLS | `CLS_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HAVG.128_MC4` |
| M2O | `M2O_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T8_GRU1x128-BI_HAVG.64_MC4` |
| CMP | `CMP_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T8_MEAN_FDIF_HAVG.64_MC2` |
| SEG | `SEG_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_UTC-DBN-D2x64.2x64.2x32.2x32.1x16_MC9` |

M2M names have the same layout as M2O names.

## MLflow Tracking
MLflow keeps every training run in one database, so that the runs can be listed, compared and searched by their settings and results in a web page. It is used next to TensorBoard and `Results.txt`, not instead of them.

To launch it, set `ExperimentConfiguration.tracking.activated: True` in the configuration and train as usual. The runs are written to `<experiment_folder>/mlflow.db` and their files to `<experiment_folder>/mlartifacts/`. To browse them, start the interface and open http://localhost:5000:

```bash
uv run mlflow ui --backend-store-uri sqlite:///<experiment_folder>/mlflow.db
```

Advantages:
- All the runs of an experiments root folder in one table, sorted and filtered by any setting or metric (e.g. the best test accuracy).
- The full configuration of every run is kept, with the git commit it was trained from.
- A resumed training continues the same MLflow run, and the changed settings are recorded.
- It never slows down nor stops the training: the values are written by a background thread, and an unavailable database only prints a warning.

Disadvantages:
- An extra package, about 1 s to import, and a few seconds to create a new database.
- The training curves are interval averages (every `information_dump` iterations): the per-iteration curves stay in TensorBoard.
- The checkpoints, predictions, error images and sample images are not sent, only the folder of the run is recorded.
- The copied files (configuration, evaluation files) take extra disk space.

What is logged:

| Output | Tasks |
|---|---|
| The configuration as parameters, the number of model parameters and of samples | all |
| `train/<metric>`, `validation/<metric>`, `learning_rate`, `train/throughput` | all |
| `train/frame_throughput` (real frames per second, padding frames excluded) | M2O, M2M, CMP |
| `test/best_<headline metric>`, `Results.txt` and the configuration file | all |
| `test/mAP@0.5`, `mAP@0.75`, `mAP@[0.5:0.95]` per method (`area`, `11_point`, `101_point`); `test_class/AP@0.5/<class>`; `class_metrics.csv` | DET |
| `test/Accuracy`, `Balanced Accuracy`, `Macro F1`, `Top-<k> Accuracy` (multi-label: `Subset Accuracy`, `Micro F1`, `Macro F1`); `test_class/Precision`, `Recall`, `F1` per class; `class_metrics.csv`, `class_thresholds.json`, `confusion_matrix.png` | CLS, M2O, M2M, CMP |
| `test/Sequence Exact Match` | M2M |
| `test/Mean IoU`, `Mean Dice`, `Pixel Accuracy`, `Frequency Weighted IoU`; `test_class/IoU`, `Dice`, `Precision`, `Recall` per class; `class_metrics.csv`, `confusion_matrix.png` | SEG |

The `test_class/` metrics and `confusion_matrix.png` need `evaluation.per_class_report: True`, and the files need `tracking.log_artifacts: True`. MLflow writes `@` as `_at_` and `[0.5:0.95]` as `0.5-0.95` in the metric names. `scripts/evaluation/evaluate_checkpoint.py` logs its evaluations in a separate run, `<run name>-evaluation`.

## TODO
