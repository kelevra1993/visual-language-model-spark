# Visual Language Model Spark

## Table Of Content
- Description
- Prerequisites
- Configurations
- Run Names

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

### Keys Per Task
| Key | DET | CLS | M2O | M2M | CMP |
|---|:---:|:---:|:---:|:---:|:---:|
| `RegionProposal` | ✓ | | | | |
| `Anchors` | ✓ | | | | |
| `Detector` | ✓ | | | | |
| `ExperimentConfiguration.evaluation.score_threshold` | ✓ | | | | |
| `ExperimentConfiguration.evaluation.top_k` | | ✓ | ✓ | ✓ | ✓ |
| `ExperimentConfiguration.evaluation.target_precision` | | ✓ | ✓ | ✓ | ✓ |
| `ExperimentConfiguration.evaluation.save_predictions` | | ✓ | ✓ | ✓ | ✓ |
| `ExperimentConfiguration.evaluation.error_dumps` | | ✓ | ✓ | ✓ | ✓ |
| `ClassificationHead` | | ✓ | ✓ | ✓ | ✓ |
| `TemporalModel` | | | ✓ | ✓ | ✓ |
| `Data.sequence_settings` | | | ✓ | ✓ | ✓ |
| `Data.sequence_settings.window_stride` | | | ✓ | ✓ | |
| `Fusion` | | | | | ✓ |

### Keys Per Group Of Tasks
| Tasks | Keys they share that the other tasks do not have |
|---|---|
| DET | `RegionProposal`, `Anchors`, `Detector`, `ExperimentConfiguration.evaluation.score_threshold` |
| CLS, M2O, M2M, CMP | `ClassificationHead`, `ExperimentConfiguration.evaluation.top_k`, `target_precision`, `save_predictions`, `error_dumps` |
| M2O, M2M, CMP | `TemporalModel`, `Data.sequence_settings` (`sequence_length`, `frame_stride`, `short_sequences`) |
| M2O, M2M | `Data.sequence_settings.window_stride` |
| CMP | `Fusion` |

### Values Per Task
Some keys are in several tasks but do not accept the same values in each. The check of the task refuses a value that its task cannot use. Everything not listed here accepts the same values in every task that has the key.

| Key | DET | CLS | M2O | M2M | CMP |
|---|---|---|---|---|---|
| `Data.DALI.activated` | `True` or `False` | `False` | `False` | `False` | `False` |
| `Data.label_mapping.mode` | `binary` or `multi_class` | `multi_class` | `multi_class` | `multi_class` | `multi_class` |
| `Data.data_augmentation.independent_frames` | `False` | `False` | `True` or `False` | `True` or `False` | `True` or `False` |
| `Data.data_augmentation.shuffle_frames.activated` | `False` | `False` | `True` or `False` | `True` or `False` | `True` or `False` |
| `Backbone.detection_convolution_block_indices` | at least one block | `[]` | `[]` | `[]` | `[]` |
| `Backbone.normalization.feature_map_normalization` | any except `sequence` | any except `sequence` | any | any | any |
| `ClassificationHead.global_pooling` | | `average`, `max` or `flatten` | `average` or `max` | `average` or `max` | `average` or `max` |
| `ClassificationHead.output_mode` | | `multi_class` or `multi_label` | `multi_class` | `multi_class` | `multi_class` |
| `TemporalModel.type` | | | all 7 types | `gru`, `lstm`, `transformer`, `temporal_convolution` | all 7 types |
| `TemporalModel.causal` | | | `False` | `True` or `False` | `False` |

An empty cell means that the task does not have the key. The 7 temporal types are `mean_pooling`, `max_pooling`, `attention_pooling`, `gru`, `lstm`, `transformer` and `temporal_convolution`. M2M needs one output per frame, which the three poolings do not give.

M2O and M2M have exactly the same keys. The number of blocks in `Backbone.convolutions` changes from one example to another, but this is a value, not a different key.

### Evaluation Outputs
Every evaluated checkpoint is reported in the console and in a block of `<run>/Results.txt` with the same layout: a header `Iteration <n> · <evaluated samples> · <date and time>`, the global metrics, the notes (e.g. the error images) and the tables of the task, the per-class table being sorted worst first. Its files are written in `<run>/Evaluation/Iteration_<n>/`. `ExperimentConfiguration.evaluation.per_class_report` set to `False` keeps the global metrics only, for runs with many classes: the files written in every case keep the details.

| Output | Tasks | Written |
|---|---|---|
| Global metrics and notes (console, `Results.txt`), global Test scalars (TensorBoard) | all | every evaluation |
| Per-class table (console, `Results.txt`), per-class Test scalars (TensorBoard) | all | with `per_class_report` |
| `class_metrics.csv`: one row per class (samples, precision, recall, F1, threshold and recall at the target precision; detection: ground truths and average precisions) | all | every evaluation |
| `class_thresholds.json` | CLS, M2O, M2M, CMP | every evaluation |
| `confusion_matrix.csv` | CLS (multi-class), M2O, M2M, CMP | every evaluation |
| `confusion_matrix.png` and its Test image (TensorBoard) | CLS (multi-class), M2O, M2M, CMP | with `per_class_report` |
| `predictions.csv` | CLS, M2O, M2M, CMP | with `save_predictions` |
| `Errors/<TRUE>-TO-<PREDICTED>/` | CLS (multi-class), M2O, M2M, CMP | with `error_dumps.activated` |
| `Evaluation/evaluation_summary.csv` (`scripts/evaluation/evaluate_checkpoint.py` with `iteration = "all"`) | all | per-class columns with `per_class_report` |

## Run Names
A training run is saved in `<ExperimentConfiguration.experiment_folder>/<run name>`. The run name is built from the configuration by `vision_training/configuration/run_name.py`: the same settings resume in the same folder, changed settings start a new folder.

The name is made of sections joined with `_`, the tokens of a section being joined with `-`. A flag token is left out when its setting is off, and the optimizer section is left out with plain Adam without weight decay. The sections always come in the order of the table below. Values in `<...>` are read from the configuration.

The training settings that can change between two resumptions of a run give no token: augmentation (including `shuffle_frames` and `independent_frames`), batch size, learning rate, warm-up, cosine schedule, gradient clipping and mixed precision. A run can be resumed in its folder with changed values, and two runs differing only by these settings share a folder. The optimizer keeps its token, because the state of one optimizer cannot be resumed by another.

| Section | Token | Meaning (configuration key) | DET | CLS | M2O | M2M | CMP |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| Task | `DET`, `CLS`, `M2O`, `M2M`, `CMP` | `ExperimentConfiguration.task` | ✓ | ✓ | ✓ | ✓ | ✓ |
| Input | `IN` | `Backbone.normalization.input_normalization` is True | ✓ | ✓ | ✓ | ✓ | ✓ |
| Input | `PSN` | `Backbone.normalization.per_sample_normalization` is True | ✓ | ✓ | ✓ | ✓ | ✓ |
| Input | `KAR` | `Data.image_settings.keep_ratio` is True | ✓ | ✓ | ✓ | ✓ | ✓ |
| Input | `<size>` | `Data.image_settings.size` | ✓ | ✓ | ✓ | ✓ | ✓ |
| Backbone | `SE`, `RES` | `Backbone.modules.squeeze_and_excite`, `residual_nets` are True | ✓ | ✓ | ✓ | ✓ | ✓ |
| Backbone | `INC<block>` | `Backbone.modules.inception` is True, from `inception_starting_block` | ✓ | ✓ | ✓ | ✓ | ✓ |
| Backbone | `BN`, `LN`, `NN` | `Backbone.normalization.feature_map_normalization`: batch, layer, none | ✓ | ✓ | ✓ | ✓ | ✓ |
| Backbone | `SN` | `Backbone.normalization.feature_map_normalization`: sequence | | | ✓ | ✓ | ✓ |
| Backbone | `B<layers>x<filters>.…` | `Backbone.convolutions`, one `<layers>x<filters>` per block | ✓ | ✓ | ✓ | ✓ | ✓ |
| Backbone | `PT` | `Backbone.pretrained.experiment_folder` is set | ✓ | ✓ | ✓ | ✓ | ✓ |
| Backbone | `FRZ` or `BLR<multiplier>` | `Backbone.pretrained.freeze` is True, or `learning_rate_multiplier` is not 1 | ✓ | ✓ | ✓ | ✓ | ✓ |
| Backbone | `FBN` | `Backbone.pretrained.freeze_normalization_parameters` is True | ✓ | ✓ | ✓ | ✓ | ✓ |
| Levels | `L<blocks>` | `Backbone.detection_convolution_block_indices` | ✓ | | | | |
| Anchors | `A<anchors>.<smallest>-<largest>` | `Anchors.scales_and_ratios`: anchors per position, range of the scales | ✓ | | | | |
| Region proposal | `RBN`, `RLN`, `RNN` | `RegionProposal.normalization.feature_map_normalization` | ✓ | | | | |
| Region proposal | `R<layers>x<filters>` | `RegionProposal.head_configurations` | ✓ | | | | |
| Detector | `DSE`, `DRES`, `DINC` | `Detector.modules` | ✓ | | | | |
| Detector | `DBN`, `DLN`, `DNN` | `Detector.normalization.feature_map_normalization` | ✓ | | | | |
| Detector | `D<pool>.<layers>x<filters>.<fully connected>` | `Detector.head_configurations` | ✓ | | | | |
| Classes | `BIN` or `MC<classes>` | `Data.label_mapping.mode`: binary or multi_class, foreground classes of the label dictionary | ✓ | | | | |
| Window | `T<frames>` | `Data.sequence_settings.sequence_length` (frames per member for CMP) | | | ✓ | ✓ | ✓ |
| Window | `S<stride>` | `Data.sequence_settings.frame_stride` is above 1 | | | ✓ | ✓ | ✓ |
| Temporal module | `MEAN`, `MAX`, `ATT<hidden size>` | `TemporalModel.type`: mean_pooling, max_pooling, attention_pooling | | | ✓ | | ✓ |
| Temporal module | `GRU<layers>x<hidden size>`, `LSTM<layers>x<hidden size>` | `TemporalModel.type`: gru, lstm | | | ✓ | ✓ | ✓ |
| Temporal module | `TRF<layers>x<hidden size>.<heads>` | `TemporalModel.type`: transformer | | | ✓ | ✓ | ✓ |
| Temporal module | `TCN<layers>x<hidden size>.<kernel size>` | `TemporalModel.type`: temporal_convolution | | | ✓ | ✓ | ✓ |
| Temporal module | `BI` | `TemporalModel.bidirectional` is True (gru, lstm) | | | ✓ | ✓ | ✓ |
| Temporal module | `CAU` | `TemporalModel.causal` is True (transformer, temporal_convolution) | | | | ✓ | |
| Temporal module | `TDO<rate>` | `TemporalModel.dropout_rate` is above 0 (gru, lstm, transformer, temporal_convolution) | | | ✓ | ✓ | ✓ |
| Temporal module | `TPT`, then `TFRZ` or `TLR<multiplier>` | `TemporalModel.pretrained` (not for mean or max pooling) | | | ✓ | ✓ | ✓ |
| Fusion | `FCAT`, `FDIF`, `FXA<heads>` | `Fusion.type`: concatenation, difference, cross_attention | | | | | ✓ |
| Fusion | `FDO<rate>` | `Fusion.dropout_rate` is above 0 (cross_attention) | | | | | ✓ |
| Fusion | `SNP` | `Fusion.sequence_normalization_scope` is pair, with the `SN` backbone | | | | | ✓ |
| Head | `HAVG.<fully connected>` or `HMAX.<fully connected>` | `ClassificationHead.global_pooling` average or max, `fully_connected` | | ✓ | ✓ | ✓ | ✓ |
| Head | `HFLAT.<fully connected>` | `ClassificationHead.global_pooling` flatten, `fully_connected` | | ✓ | | | |
| Head | `DO<rate>` | `ClassificationHead.dropout_rate` | | ✓ | ✓ | ✓ | ✓ |
| Head | `LS<rate>` | `ClassificationHead.label_smoothing` is above 0 | | ✓ | ✓ | ✓ | ✓ |
| Head | `CWB` | `ClassificationHead.class_weighting` is balanced | | ✓ | ✓ | ✓ | ✓ |
| Head | `HPT`, then `HFRZ` or `HLR<multiplier>` | `ClassificationHead.pretrained` | | ✓ | ✓ | ✓ | ✓ |
| Classes | `MC<classes>` | `ClassificationHead.output_mode` multi_class, classes of the label dictionary | | ✓ | ✓ | ✓ | ✓ |
| Classes | `ML<classes>` | `ClassificationHead.output_mode` multi_label | | ✓ | | | |
| Optimizer | `ADAMW` or `SGD` | `Optimization.optimizer` is not adam | ✓ | ✓ | ✓ | ✓ | ✓ |
| Optimizer | `WD<decay>` | `Optimization.weight_decay` is above 0 | ✓ | ✓ | ✓ | ✓ | ✓ |

For the detection heads and the anchors, a setting identical on every detection level is written once, otherwise the levels are joined with `+` in level order. The loading settings (`Data.loading`) and the evaluation settings add no token either.

Examples, built from the configurations of the repository:

| Task | Run name |
|---|---|
| DET | `DET_IN-KAR-1024_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128_BIN` |
| CLS | `CLS_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HAVG.128-DO0.2_MC4` |
| M2O | `M2O_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T8_GRU1x128-BI_HAVG.64-DO0.2_MC4` |
| CMP | `CMP_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T8_MEAN_FDIF_HAVG.64-DO0.2_MC2` |

M2M names have the same layout as M2O names.

## Implementation Plan

### Semantic Segmentation Task (`semantic_segmentation`, run name code `SEG`)

#### Goal
Add a sixth task, semantic segmentation (one class per pixel), built like the other tasks: a U-Net whose downscaling
path is the existing `Backbone`, a new decoder section, its own trainer, checks, run name section, dataset, losses,
evaluation, synthetic dataset, deployment wrapper and tests. The backbone can be initialised by any trained run
(classification, detection, sequence tasks or segmentation) through `Backbone.pretrained`, as for the other tasks.

The previous U-Net project was read for reference only: nothing is copied as is, and no code comment refers to it.

#### What The Previous Project Did, And What Changes
| Topic | Previous project | Plan |
|---|---|---|
| Encoder | Its own U-Net encoder (`[layers, channels]` per level, channel LayerNorm, max pooling between levels) | The repository `Backbone`, with every option it has (SE, residual, inception, batch / layer / no normalization, input normalization) and its transfer |
| Skip connections | Block outputs before pooling, the first one at full resolution | The same, which needs a small `Backbone` change: it only returns block outputs after their max pooling today (see "Backbone Change") |
| Decoder | Transposed convolution ×2 to the skip channels, concatenation (skip first), convolution block, bilinear resize if the sizes differ, final 1×1 convolution | Kept, configurable (transposed convolution or bilinear upsampling), built from the repository `ConvolutionBlock` so the decoder gets the same normalization / SE / residual options |
| Masks on disk | One binary PNG per class, background computed, resized with bilinear interpolation (soft edges) | One single-channel class-index PNG per image, 0 background, 255 ignored, resized with nearest interpolation |
| Targets | One-hot float maps | Integer class maps `[B, H, W]`, padding and unlabelled pixels at `IGNORED_LABEL_INDEX` (-100) |
| Losses | Cross entropy, weighted cross entropy or focal, one at a time, no ignore index | Cross entropy (ignore index, label smoothing, class weighting) with an optional focal factor, plus an optional Dice loss, each a separate `_loss` output summed by the trainer |
| Metrics | IoU per image averaged per batch, a class absent and not predicted counting 1 | A pixel confusion matrix accumulated over the whole test split: pixel accuracy, per-class IoU / Dice / precision / recall, mean IoU, frequency-weighted IoU; absent classes shown as "—" and left out of the means |
| Augmentation | Horizontal flip only | Every geometric augmentation of `DataAugmenter`, applied to the masks with nearest interpolation, and every photometric one on the images only |
| Visual checks | 20 fixed test samples, per-class grid with TP / FP / FN error map and boundary overlay, GIF script | Kept: fixed test samples with input, ground truth overlay, prediction overlay and error map, saved per evaluation and sent to TensorBoard |
| Deployment | None | An inference wrapper, its export specification and the restoration of a mask to the original image size |

#### Backbone Change: Skip Features Before Pooling
Today `Backbone.forward` returns `(final_tensor, {block: output})` for the blocks of
`detection_convolution_block_indices`, every output being taken after the max pooling that ends the block
(`ConvolutionBlock` appends `nn.MaxPool2d` last). The finest map is then at stride 2, and a U-Net would lose the
full-resolution details of block 1.

Change, without any new weight (so the transfer between tasks is unaffected):
1. `ConvolutionBlock.forward(input_tensor, sequence_grouping=None, return_pre_pooling_tensor=False)`: when asked,
   also returns the tensor right before its pooling layer (`self.pooling_index` already marks it), i.e.
   `(output_tensor, pre_pooling_tensor)`.
2. New `Backbone.extract_block_features(input_tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]`: runs the input
   normalizer and every block, and returns the final tensor and the pre-pooling output of every block, block 1 at the
   input resolution, block b at `S / 2**(b-1)`.
3. `Backbone.forward` and every existing caller stay unchanged.
4. `compute_receptive_field_information` / `print_summary` stay as they are; the segmentation model prints its own
   summary with the skip sizes.

Levels of the U-Net with N backbone blocks:
- the skips are the pre-pooling outputs of blocks 1 to N-1;
- with `last_max_pooling: False` the bottleneck is the output of block N (at `S / 2**(N-1)`);
- with `last_max_pooling: True` the bottleneck is the pooled output of block N (at `S / 2**N`) and the pre-pooling
  output of block N is one more skip;
- so the decoder has N-1 levels (N with `last_max_pooling: True`), each doubling the resolution, the last one
  ending at the input resolution.

`detection_convolution_block_indices` must be `[]` for segmentation (checked), as for classification: the decoder
always reads every block.

#### Configuration
New section `SegmentationDecoder`, and evaluation keys of the task; every other section is the one of the other
tasks. Example for 6 backbone blocks with `last_max_pooling: False` (5 decoder levels):

```yaml
SegmentationDecoder:
  # Upsampling of every decoder level: "transposed" (2x2 transposed convolution with stride 2, learned) or "bilinear"
  # (bilinear upsampling followed by a 1x1 convolution to the channels of the level).
  upsampling: "transposed"
  # Convolution block of every decoder level, keyed by the backbone block whose skip connection it merges, as
  # [number of 3x3 convolutions, output channels]. The upsampled tensor is concatenated with the skip (skip first),
  # then goes through the block. Every skip level must be listed: blocks 1 to N-1, and N with last_max_pooling.
  convolutions:
    "5": [2, 64]
    "4": [2, 64]
    "3": [2, 32]
    "2": [2, 32]
    "1": [1, 16]
  # Same options as the backbone blocks, applied to the decoder blocks.
  modules:
    squeeze_and_excite: False
    residual_nets: False
  # Normalization of the decoder feature maps: "batch", "layer" or "none".
  feature_map_normalization: "batch"
  # Dropout before the final 1x1 convolution giving the class logits, 0 to disable.
  dropout_rate: 0.0
  # Initialisation of the decoder by the decoder of another segmentation run, same keys as Backbone.pretrained
  # (without freeze_normalization_parameters). The final 1x1 convolution is never transferred, so the classes may
  # differ.
  pretrained:
    experiment_folder: null
    iteration: null
    freeze: False
    learning_rate_multiplier: 1.0

SegmentationLoss:
  # Weight of the pixel cross entropy (ignoring the pixels at the ignore index), 0 to disable.
  cross_entropy_weight: 1.0
  # Focusing parameter of the focal loss applied to the cross entropy, 0 for the plain cross entropy.
  focal_gamma: 0.0
  # Label smoothing of the cross entropy, within [0, 1).
  label_smoothing: 0.0
  # Class weights of the cross entropy: "none", or "balanced" (inverse pixel frequency of the training split).
  class_weighting: "none"
  # Weight of the soft Dice loss (mean over the classes present in the batch), 0 to disable.
  dice_weight: 0.0
```

Evaluation keys (`ExperimentConfiguration.evaluation`): `per_class_report` (common), `number_sample_images` (fixed
test samples drawn at every evaluation, 0 to disable), `save_predicted_masks` (class-index PNG of every test image,
False by default since it grows with the test set).

Data: `Data.label_mapping.mode` is `"binary"` (background / foreground) or `"multi_class"` (the mapping file gives the
class of every category id, class 0 being the background); `Data.DALI.activated` must be False (native loader only).

Checker `configuration/segmentation_checks.py::check_segmentation_configuration`:
- DALI False, label mode binary or multi_class, classes indexed 0 to K-1 with 0 the background;
- `detection_convolution_block_indices == []`, `check_image_task_sequence_settings`;
- `SegmentationDecoder.convolutions` keys exactly the skip levels computed from the backbone blocks and
  `last_max_pooling`, every value `[positive, positive]`, `upsampling` valid, normalization valid, dropout in [0, 1);
- `check_pretrained_configuration(model_configuration, "SegmentationDecoder")` (added to `PRETRAINED_COMPONENT_NAMES`);
- loss weights non-negative and not both 0, `focal_gamma >= 0`, label smoothing in [0, 1), class weighting valid;
- evaluation keys.

Run name (`build_segmentation_sections`, code `SEG`): decoder tokens `U<TC|BL>`, `DSE` / `DRES` / normalization code,
`D<layers>x<channels>.…` in decoder order, `DPT` / `DFRZ` / `DLR<m>` for its transfer (`PRETRAINED_TOKEN_PREFIXES`
gains `"SegmentationDecoder": ("D", "D")`), loss tokens following the head precedent (`LS<smoothing>` and the class
weighting token, `FG<gamma>`, `DICE<weight>`), and the classes `BIN` / `MC<K>`. Example:
`SEG_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_UTC-DBN-D2x64.2x64.2x32.2x32.1x16_MC5`.

#### Data
- Split layout: `<split>/data/<image>`, `<split>/masks/<mask>.png` (single channel, uint8, pixel = category id, 0
  background, 255 ignored), `<split>/labels.json`:
  `{"categories": [{"id", "name"}], "samples": [{"id", "file_name", "mask_file_name"}]}`.
- `data/segmentation/segmentation_dataset.py::SegmentationDataset`: reads the BGR image and the mask, maps the
  category ids to class indices with a lookup table (255 → ignore), resizes the image with `preprocess_image` and the
  mask with a new `preprocessing.preprocess_mask(mask, image_size, keep_ratio)` (nearest interpolation, same scale and
  centred offsets as the image, padding with 255), returns `uint8 [3, S, S]` and `int64 [S, S]` with
  `IGNORED_LABEL_INDEX` on padding and ignored pixels.
- `loaders.get_segmentation_dataloaders(...)` with `build_native_loader`; `split_description` counts the images and
  the classes.
- `core/contracts.py`: `SegmentationBatch(images, masks, sample_identifiers)` and its line in the module docstring.
- Label mapping: `binary` gives `{"background": 0, "foreground": 1}` for segmentation too (`TASKS_WITH_BACKGROUND_CLASS`
  gains `"semantic_segmentation"`); `multi_class` reads the mapping file and adds the background at 0.
- Class weights for `class_weighting: "balanced"`: pixel counts per class of the training masks, computed once by the
  trainer (reusing `compute_balanced_class_weights`), cached in `<run folder>/class_pixel_counts.json` so that a resume
  does not read every mask again.

#### Augmentation
`DataAugmenter.augment_batch(images, bounding_boxes=None, labels=None, masks=None)`: every geometric method gets a
`masks` argument, the masks wrapped as `tv_tensors.Mask` so that the torchvision v2 functional ops use nearest
interpolation (no antialias); `random_crop` crops and resizes them, the flips flip them, `random_rotation` fills the
corners with `IGNORED_LABEL_INDEX`, `random_90_rotation` uses `torch.rot90`. The photometric ops only touch the
images. `SegmentationTrainer.augment_batch` overrides the default like `DetectionTrainer` does for the boxes.

#### Model
- `models/segmentation/segmentation_decoder.py::SegmentationDecoder(nn.Module)`: one upsampling layer and one
  `ConvolutionBlock` (`add_pooling=False`) per level, built from the skip channels read from
  `Backbone.convolutions`; `forward(bottleneck_tensor, skip_tensors) -> torch.Tensor`, with the bilinear size fallback
  when an upsampled tensor does not match its skip.
- `models/segmentation/semantic_segmenter.py::SemanticSegmenter(TaskModel)`: attributes `backbone`,
  `segmentation_decoder` and `segmentation_head` (dropout + 1×1 convolution to K classes);
  `forward(input_tensor, masks=None)` returns `class_logits [B, K, S, S]`, `class_probabilities` only in evaluation
  mode (a softmax over every pixel is not needed for the training), and `segmentation_loss` / `dice_loss` when masks
  are given; `print_summary` (backbone, skip sizes, decoder levels, head), `get_size_components`
  (`Backbone`, `Segmentation Decoder`, `Segmentation Head`, `Total Model`).
- `models/segmentation/inference_segmenter.py::InferenceSegmenter`: `forward(images) -> class probabilities` or class
  map (see Deployment).
- `models/model_registry.py`: `TASK_MODEL_CLASSES["semantic_segmentation"] = SemanticSegmenter`.

#### Transfer
- Backbone: nothing to change in `model_transfer.py`, `ARCHITECTURE_BACKBONE_KEYS` already ignores
  `detection_convolution_block_indices` and the image size, so a classifier, detector or sequence backbone initialises
  the segmentation backbone (a different image size only prints its warning), and a segmentation backbone initialises
  the other tasks.
- Decoder: `TRANSFERABLE_COMPONENTS` gains `"SegmentationDecoder": "segmentation_decoder"`, and
  `compare_component_configurations` a branch comparing `upsampling`, `convolutions`, `modules`,
  `feature_map_normalization` and the backbone block channels feeding the skips (today any unknown section falls
  through to the classification head comparison, so the branch is required). `freeze`, `learning_rate_multiplier`,
  the optimizer parameter groups, `set_training_mode` and the checkpoint metadata (`pretrained_segmentation_decoder`)
  then work through the existing loops over `TRANSFERABLE_COMPONENTS`. `disable_pretrained_settings` covers it too.
- Migration: `PRETRAINED_SETTINGS_BLOCKS` gains the decoder block; no existing run needs migrating (new task).

#### Losses
`losses/segmentation_losses.py`:
- `compute_segmentation_cross_entropy(class_logits, masks, label_smoothing, class_weights, focal_gamma)`: pixel cross
  entropy with `ignore_index=IGNORED_LABEL_INDEX`, the focal factor `(1 - p_t) ** gamma` when `gamma > 0`, normalised
  by the number of labelled pixels (or by the sum of their weights with class weights).
- `compute_dice_loss(class_logits, masks)`: soft Dice on the softmax, ignored pixels left out, mean over the classes
  present in the batch, in float32 (stable under mixed precision).
- The model multiplies them by their weights and returns `segmentation_loss` and `dice_loss` (a loss with weight 0 is
  not computed nor returned), so the console and TensorBoard track them separately.

#### Trainer
`trainers/segmentation_trainer.py::SegmentationTrainer(BaseTrainer)`, `TASK_NAME = "semantic_segmentation"`,
`SAMPLE_NAME = "Images"`:
- `build_dataloaders`, `build_model` (through `TASK_MODEL_CLASSES`, class weights copied into the model),
  `get_tracked_metrics` (total, segmentation and Dice losses, pixel accuracy, batch mean IoU), `describe_split`,
  `prepare_batch` (images to the model dtype, masks to int64 on the device), `forward_pass`, `predict_batch`,
  `augment_batch` (masks), `compute_batch_metrics`;
- `create_evaluation_accumulator` / `accumulate_evaluation_batch`: a `[K, K]` int64 confusion matrix on the device
  filled with `torch.bincount` on the labelled pixels of every batch (no per-pixel data kept on the CPU), plus the
  fixed sample images;
- `compute_evaluation_results`: global metrics (pixel accuracy, mean IoU, frequency-weighted IoU, mean Dice), a
  per-class table (Class, Pixels, IoU, Dice, Precision, Recall, worst IoU first, "—" for classes absent from the test
  split), Test scalars and per-class scalars, `class_metrics.csv`, `confusion_matrix.csv` and, with the per-class
  report, the confusion matrix image (shares per true class, the pixel counts being too large to print), the sample
  grids in `Evaluation/Iteration_<n>/Samples/` and as TensorBoard images, `headline_metric = "Mean IoU"`;
- `show_input_samples`: the image with its mask overlaid (training augmented, validation), for `view_input_images`.
- `core/task_registry.py`: `TASK_TRAINER_CLASSES["semantic_segmentation"] = SegmentationTrainer`.

`evaluation/segmentation/segmentation_metrics.py` (pure functions on the confusion matrix, reusing the classification
precision / recall helpers where they fit) and `evaluation/segmentation/segmentation_report.py` (rows, table,
scalars), `utilities/visualization.py`: `overlay_segmentation_mask` and `build_segmentation_error_map` (TP green,
FP red, FN yellow), with one colour per class.

#### Synthetic Dataset
`create_synthetic_dataset(task="semantic_segmentation")`: `place_shape` also paints the class index into an `[H, W]`
class map at the same pixels and in the same order as the image (later shapes overwrite earlier ones, as in the
image), several shapes per image (`maximum_objects`), the mask written next to the image in `masks/`, the
`labels.json` of the task, and the label mapping with the background at 0. New
`configurations/semantic_segmentation/configuration_example.yaml` and `configuration_synthetic_shapes.yaml`, and the
commented path in `main.py`.

#### Deployment
- `deployment/export_specifications.py`: `INFERENCE_MODEL_CLASSES`, `TASK_MODEL_NAMES` (`semantic_segmenter`),
  `TASK_INPUT_NAMES`, and a segmentation branch in `get_export_specification`: output `class_map` `[S, S]` INT64 and
  `class_probabilities` `[K, S, S]` FP32 (or only the class map for many classes), compared with
  `compare_probabilities` and the share of equal class map pixels.
- `deployment/deployment_utilities.py`: `restore_mask_to_original_image(class_map, original_height, original_width,
  keep_ratio)` removing the centred padding and resizing with nearest interpolation, for `run_inference.py`.

#### Every Place Listing The Tasks
`core/task_registry.py`, `models/model_registry.py`, `configuration/configuration_checker.py`,
`configuration/run_name.py` (`TASK_CODES`, `TASK_RUN_NAME_SECTIONS`), `deployment/export_specifications.py`,
`data/label_mapping.py`, `data/synthetic/create_synthetic_dataset.py` (`TASK_CATEGORIES`, `TASK_MANIFEST_KEYS`,
`TASK_SAMPLE_NAMES`, `generate_split`, `class_index_offset`), `main.py`, README tables (task codes, keys per task,
values per task, evaluation outputs, run names), `tests/configuration/test_configuration_checker.py` (the list of
valid tasks in the expected error message), `tests/core/test_task_registry.py` (checks every registry holds the same
tasks), `tests/deployment/test_deployment_utilities.py` (`TASK_CONFIGURATION_FILES`).

#### Tests
- `tests/models/test_backbone.py` (or the existing backbone tests): `extract_block_features` sizes and channels with
  and without `last_max_pooling`, `forward` unchanged.
- `tests/models/segmentation/test_semantic_segmenter.py`: output `[B, K, S, S]`, losses only with masks, ignored pixels
  without gradient, both upsampling modes, SE / residual / normalization options, `print_summary`.
- `tests/models/test_model_transfer.py`: a classifier backbone initialising a segmenter and back, a decoder transferred
  between segmenters with different classes, a decoder of other levels rejected with its message.
- `tests/losses/test_segmentation_losses.py`: ignore index, focal with gamma 0 equal to the cross entropy, Dice of a
  perfect prediction equal to 0, class weights.
- `tests/data/segmentation/test_segmentation_dataset.py`: nearest resizing (only class values in the mask), padding at
  the ignore index, keep ratio offsets equal to the image ones.
- `tests/data/test_augmentations.py`: masks follow every geometric augmentation, keep only class or ignore values,
  photometric ops leave them unchanged.
- `tests/evaluation/segmentation/test_segmentation_metrics.py`: IoU / Dice / accuracy from a known confusion matrix,
  absent classes excluded from the means.
- `tests/configuration`: checker (every key, the decoder levels against the backbone), run name tokens, migration of
  the decoder pretrained block.
- `tests/data/synthetic/test_create_synthetic_dataset.py`: masks match the drawn shapes.
- `tests/trainers/test_segmentation_trainer.py`: a short synthetic training, `Results.txt`, TensorBoard, evaluation
  files, resume; `tests/trainers/test_backbone_transfer_training.py`: a classification backbone initialising a
  frozen segmentation run.
- `tests/deployment`: export specification, TorchScript / ONNX export and comparison, mask restoration.

#### Phases
1. Backbone pre-pooling features and their tests.
2. Data: manifest, dataset, mask preprocessing, label mapping, augmentation of the masks, synthetic generator.
3. Model, decoder, head, losses, model registry.
4. Configuration: checks, run name, YAMLs, README tables.
5. Trainer, evaluation metrics, report, sample grids, task registry, trainer smoke test.
6. Transfer: backbone tests, decoder as a transferable section.
7. Deployment.
8. Verification (full test suite, a synthetic training looked at in TensorBoard, a transfer from the ImageNet
   classifier backbone), then removal of this plan.

#### Suggestions
- **Pre-pooling skips over post-pooling skips:** using the existing outputs (after pooling) would need one more ×2
  upsampling at the end without any full-resolution skip, which blurs the boundaries; the small `ConvolutionBlock`
  change keeps the classic U-Net quality and adds no weight.
- **Rename `detection_convolution_block_indices`** to `output_convolution_block_indices` in a later, separate change
  (with its migration): the name is detection-specific while three tasks now read or check it.
- **Pretraining:** start from the ImageNet classification backbone; the image size may differ (a warning only).
  Train first with `freeze: True` for a few thousand iterations, then resume with the backbone trainable at
  `learning_rate_multiplier: 0.1` (the run name changes with these tokens, so plan the second phase as a transfer from
  the first run).
- **Small batches:** segmentation images are large, so batches are small and batch normalization statistics become
  noisy. Prefer `layer` normalization in the decoder, and for a pretrained batch-normalized backbone keep its
  statistics with `freeze_normalization_parameters` or freezing.
- **Loss choice for imbalanced classes:** cross entropy plus Dice (e.g. 1.0 / 1.0) is a robust default for small
  structures; the focal factor or `balanced` weights are the alternatives, compare them on the synthetic dataset first.
- **Evaluation at network resolution first:** the metrics are computed at the network resolution with the padding
  ignored; an option to evaluate at the original image resolution (masks restored) can come later.
- **Boundary metrics and losses later:** boundary IoU, deep supervision (auxiliary losses at the decoder levels),
  Lovász loss, sliding-window inference for images larger than the network input.
- **Converter for per-class binary masks:** a script in `scripts/data/segmentation/` turning datasets stored as one
  binary PNG per class into the class-index masks and `labels.json` of the task, so older datasets (e.g. the ultrasound
  one) can be reused.
- **Fix the last checkpoint overwrite** (evaluated checkpoint rewritten by the final save) before or together with
  this task, since the segmentation evaluation goes through the same loop.
- **MLflow:** the segmentation scalars go through `results["scalars"]`, so the MLflow plan covers them without change.
