# Visual Language Model Spark

## Table Of Content
- Description
- Prerequisites
- Configurations

## Description
This repository contains an object detection pipeline built with PyTorch, supporting NVIDIA DALI and native PyTorch data loading backends.

## Prerequisites
- PyTorch
- NVIDIA DALI
- OpenCV

## Configurations
Every task has its own configuration files in `vision_training/configurations/<task>/`: a `configuration_example.yaml` and a `configuration_synthetic_*.yaml`. Both files of a task have the same keys, only their values differ.

A configuration file holds only the keys its task reads, and every one of them is required. To start a new run, copy the file of the right task, not the file of another task. The check of the task lists any missing key, and a section used only by another task is ignored.

The tables below list only the keys that are not in every task. Every other key (the rest of `ExperimentConfiguration`, `Optimization`, `Data` and `Backbone`) is the same in all tasks. A section name alone (e.g. `RegionProposal`) means the whole section with all its keys. Tasks are written with their run name codes:

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
| `ClassificationHead.output_mode` | | `multi_class` or `multi_label` | `multi_class` | `multi_class` | `multi_class` |
| `TemporalModel.type` | | | all 7 types | `gru`, `lstm`, `transformer`, `temporal_convolution` | all 7 types |
| `TemporalModel.causal` | | | `False` | `True` or `False` | `False` |

An empty cell means that the task does not have the key. The 7 temporal types are `mean_pooling`, `max_pooling`, `attention_pooling`, `gru`, `lstm`, `transformer` and `temporal_convolution`. M2M needs one output per frame, which the three poolings do not give.

M2O and M2M have exactly the same keys. The number of blocks in `Backbone.convolutions` changes from one example to another, but this is a value, not a different key.
