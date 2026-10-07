# Visual Language Model Spark

## Table Of Content
- Description
- Prerequisites
- Configurations
- Run Names
- TODO

## Description
This repository contains an object detection pipeline built with PyTorch, supporting NVIDIA DALI and native PyTorch data loading backends.

## Prerequisites
- PyTorch
- NVIDIA DALI
- OpenCV

## Configurations
Every task has its own configuration files in `vision_training/configurations/<task>/`: a `configuration_example.yaml` and a `configuration_synthetic_*.yaml`. Both files of a task have the same keys, only their values differ.

A configuration file holds only the keys its task reads, and every one of them is required. To start a new run, copy the file of the right task, not the file of another task. The check of the task lists any missing key, and a section used only by another task is ignored.

The tables below list only the keys that are not in every task. Every other key (the rest of `ExperimentConfiguration`, including `evaluation.per_class_report`, `Optimization`, `Data` and `Backbone`) is the same in all tasks. A section name alone (e.g. `RegionProposal`) means the whole section with all its keys. Tasks are written with their run name codes:

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
| `ExperimentConfiguration.evaluation.number_sample_images` | | | | | | ✓ |
| `ExperimentConfiguration.evaluation.save_predicted_masks` | | | | | | ✓ |

### Keys Per Group Of Tasks
| Tasks | Keys they share that the other tasks do not have |
|---|---|
| DET | `RegionProposal`, `Anchors`, `Detector`, `ExperimentConfiguration.evaluation.score_threshold` |
| CLS, M2O, M2M, CMP | `ClassificationHead`, `ExperimentConfiguration.evaluation.top_k`, `target_precision`, `save_predictions`, `error_dumps` |
| M2O, M2M, CMP | `TemporalModel`, `Data.sequence_settings` (`sequence_length`, `frame_stride`, `short_sequences`) |
| M2O, M2M | `Data.sequence_settings.window_stride` |
| CMP | `Fusion` |
| SEG | `SegmentationDecoder`, `SegmentationLoss`, `ExperimentConfiguration.evaluation.number_sample_images`, `save_predicted_masks` |

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
| `Samples/<image>.png` and its Test image (TensorBoard): input, true mask, predicted mask and error map of the same test images at every evaluation | SEG | with `number_sample_images` above 0, whatever `per_class_report` |
| `Predicted_Masks/<image>.png`: class index of every pixel at the network resolution | SEG | with `save_predicted_masks` |
| `Evaluation/evaluation_summary.csv` (`scripts/evaluation/evaluate_checkpoint.py` with `iteration = "all"`) | all | per-class columns with `per_class_report` |

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

## TODO
- Sample grids for every task, as for SEG: a few fixed test samples drawn with their targets and predictions at every evaluation, saved in `Evaluation/Iteration_<n>/Samples/` and sent to TensorBoard through `EvaluationResults["sample_images"]` (CLS, DET, M2O, M2M, CMP).
