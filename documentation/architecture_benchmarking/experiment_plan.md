# Experiment Plan

- **Configurations:** `vision_training/configurations/classification/run-<n>.yaml`, one per launch (`run-1.yaml` to `run-10.yaml` for stage 1), launched with `uv run python -m vision_training.main --configuration_path=vision_training/configurations/classification/run-<n>.yaml`; the files of the later stages are copies of the kept one with the changed parameter
- **Dataset:** synthetic shapes classification, 9 classes (20 classes for stage 14, see Class Extension Stage), 500,000 train / 50,000 validation / 50,000 test images, resized to 256 pixels
- **Launch folders:** `hyperparameter_experiments/run_1` to `run_44` at the repository root, set as `ExperimentConfiguration.experiment_folder` before every launch

## Strategy

One parameter changes at a time. The best value of a stage is kept for all the following stages, and its launches serve as the reference of the next stage, so they are not trained again.

Every setting is launched twice, A and B, to smooth out the noise of a single training. The training sets no random seed, so the two launches start from different weights, data order and augmentations.

## Fixed Settings

- `batch_size`: 256
- `number_iterations`: 2000 (512,000 images, about one epoch)
- `weight_saving_iterations`: 250, the test evaluation running at iterations 250, 500, 750, 1000, 1250, 1500, 1750 and 2000 (8 tested models)
- `intermediate_weight_saving_iterations`: 125 (must be smaller than `weight_saving_iterations`), 8 more checkpoints saved without a test evaluation
- `data_augmentation`: on, not tested
- Everything else as in the configuration file (backbone blocks, head layers `[256, 128]`, Adam, constant learning rate)

## Starting Values

| Parameter | Value |
|---|---|
| `learning_rate` | 1e-4 |
| `fully_connected_normalization` | none |
| `batch_normalization_momentum` (backbone and head) | 0.1 |
| `feature_map_normalization` | batch |
| `global_pooling` | flatten |
| `residual_nets` | True |
| `squeeze_and_excite` | True |
| `dtype` | float32 (`mixed_precision: True`) |
| `class_weighting` | balanced |
| `dropout_rate` | 0.2 |

## Launches

Every launch gives its test accuracy at iterations 250, 500, 750, 1000, 1250, 1500, 1750 and 2000 (`Results.txt` of its run folder) and the mean of these eight values, all gathered in `hyperparameter_experiments/results.csv` by `uv run python -m scripts.evaluation.collect_experiment_results`, launches and Mean rows as in the tables below. `results.csv`, next to this plan, keeps every launch of the experiment: run_1 to run_32, gathered before their experiment folder was deleted, then run_33 to run_44 with their trained and averaged weights (`evaluated_weights`), a setting not recorded by a stage being left empty and run_41 to run_44 split by their `new_class_initialisation` (`source` or `random`, a setting removed since). The Mean row of a setting averages its two launches, and the setting with the highest mean of the Mean row is kept.

### Stage 1: Learning Rate (10 launches)

| `learning_rate` | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 1e-4 | run_1 | 0.2897 | 0.7904 | 0.8143 | 0.9777 | 0.9782 | 0.9879 | 0.9868 | 0.9895 | 0.8518 |
| 1e-4 | run_2 | 0.2809 | 0.7465 | 0.8767 | 0.9597 | 0.9763 | 0.9743 | 0.9790 | 0.9696 | 0.8454 |
| 1e-4 | **Mean** | 0.2853 | 0.7684 | 0.8455 | 0.9687 | 0.9773 | 0.9811 | 0.9829 | 0.9796 | 0.8486 |
| 2e-4 | run_3 | 0.4177 | 0.6613 | 0.9242 | 0.9804 | 0.9338 | 0.9839 | 0.9809 | 0.9382 | 0.8526 |
| 2e-4 | run_4 | 0.4032 | 0.7234 | 0.9329 | 0.9324 | 0.9784 | 0.9628 | 0.9838 | 0.9930 | 0.8637 |
| 2e-4 | **Mean** | 0.4104 | 0.6923 | 0.9285 | 0.9564 | 0.9561 | 0.9733 | 0.9824 | 0.9656 | **0.8581** |
| 5e-4 | run_9 | 0.1107 | 0.1129 | 0.1126 | 0.1268 | 0.1628 | 0.1455 | 0.3061 | 0.7757 | 0.2317 |
| 5e-4 | run_10 | 0.1123 | 0.1202 | 0.1326 | 0.1601 | 0.2100 | 0.7831 | 0.7790 | 0.8769 | 0.3968 |
| 5e-4 | **Mean** | 0.1115 | 0.1166 | 0.1226 | 0.1434 | 0.1864 | 0.4643 | 0.5425 | 0.8263 | 0.3142 |
| 1e-3 | run_5 | 0.1118 | 0.1112 | 0.1114 | 0.1107 | 0.1101 | 0.1098 | 0.1122 | 0.1568 | 0.1168 |
| 1e-3 | run_6 | 0.1096 | 0.1126 | 0.1109 | 0.1268 | 0.1164 | 0.1284 | 0.2103 | 0.3356 | 0.1563 |
| 1e-3 | **Mean** | 0.1107 | 0.1119 | 0.1112 | 0.1187 | 0.1132 | 0.1191 | 0.1612 | 0.2462 | 0.1365 |
| 2e-3 | run_7 | 0.1090 | 0.1476 | 0.1235 | 0.3402 | 0.6182 | 0.7419 | 0.9702 | 0.8994 | 0.4937 |
| 2e-3 | run_8 | 0.1127 | 0.1132 | 0.1117 | 0.1133 | 0.1131 | 0.1103 | 0.1126 | 0.1111 | 0.1122 |
| 2e-3 | **Mean** | 0.1109 | 0.1304 | 0.1176 | 0.2267 | 0.3656 | 0.4261 | 0.5414 | 0.5052 | 0.3030 |

Kept: **2e-4** (run_3 and run_4), the highest mean (0.8581 against 0.8486 for 1e-4); the configuration files of the next stages are copies of `run-3.yaml` with the changed parameter.

- 2e-4 learns faster than 1e-4 (0.4104 against 0.2853 at iteration 250, 0.9285 against 0.8455 at iteration 750), which gives it the higher mean.
- 1e-4 ends higher and steadier over iterations 1250 to 2000 (mean 0.9802 against 0.9694), 2e-4 dropping to 0.9338 at iteration 1250 (run_3) and 0.9382 at iteration 2000 (run_3). The two launches of 2e-4 differ by 0.0548 at iteration 2000 (0.9382 and 0.9930), more than the gap between the two learning rates, so this difference is within the noise of two launches.
- From 5e-4 up, the launches stay at about 0.111 (the accuracy of predicting a single class, 1 in 9) for at least the first 750 iterations, and run_8 never leaves it: the largest usable learning rate lies between 2e-4 and 5e-4.

### Stage 2: Fully Connected Normalization (4 launches)

`ClassificationHead.fully_connected_normalization`: batch normalization (statistics of every unit over the vectors of the batch, running statistics used in evaluation) or layer normalization (statistics of every vector over its units, the same in training and evaluation) between every hidden linear layer and its ReLU.

| `fully_connected_normalization` | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| none | run_3 (stage 1) | 0.4177 | 0.6613 | 0.9242 | 0.9804 | 0.9338 | 0.9839 | 0.9809 | 0.9382 | 0.8526 |
| none | run_4 (stage 1) | 0.4032 | 0.7234 | 0.9329 | 0.9324 | 0.9784 | 0.9628 | 0.9838 | 0.9930 | 0.8637 |
| none | **Mean** | 0.4104 | 0.6923 | 0.9285 | 0.9564 | 0.9561 | 0.9733 | 0.9824 | 0.9656 | 0.8581 |
| batch | run_11 | 0.1268 | 0.1480 | 0.2659 | 0.9210 | 0.9630 | 0.9740 | 0.9807 | 0.9810 | 0.6700 |
| batch | run_12 | 0.1900 | 0.8069 | 0.9428 | 0.9487 | 0.9705 | 0.9825 | 0.9893 | 0.9917 | 0.8528 |
| batch | **Mean** | 0.1584 | 0.4774 | 0.6044 | 0.9349 | 0.9667 | 0.9782 | 0.9850 | 0.9864 | 0.7614 |
| layer | run_13 | 0.6055 | 0.8918 | 0.9343 | 0.9663 | 0.9352 | 0.9787 | 0.9915 | 0.9909 | 0.9118 |
| layer | run_14 | 0.4637 | 0.7247 | 0.9480 | 0.9823 | 0.9849 | 0.9879 | 0.9799 | 0.9856 | 0.8821 |
| layer | **Mean** | 0.5346 | 0.8083 | 0.9411 | 0.9743 | 0.9601 | 0.9833 | 0.9857 | 0.9882 | **0.8969** |

Kept: **layer** (run_13 and run_14), the highest mean (0.8969 against 0.8581 for none and 0.7614 for batch) and the highest mean of the last four evaluations (`accuracy_last_4_mean`, 0.9793 against 0.9791 for batch and 0.9694 for none); the configuration files of the next stages are copies of `run-13.yaml` with the changed parameter.

- layer learns fastest: 0.5346 at iteration 250 and 0.8083 at iteration 500 against 0.4104 and 0.6923 for none, with train accuracies of 0.8004 and 0.8197 at iteration 500 against 0.6535 and 0.7652. Both of its launches start well, where batch is slower: run_11 stays near 0.111 (single class) up to iteration 500 (train accuracy 0.1429) and takes off between iterations 750 and 1000, which gives batch the lowest mean.
- layer and batch end at the same level, above none: 0.9793 and 0.9791 over iterations 1250 to 2000 against 0.9694, 0.9882 and 0.9864 at iteration 2000 against 0.9656. run_13 dips once to 0.9352 at iteration 1250, as run_3 does twice with none (0.9338 and 0.9382), where both launches of batch rise at every evaluation from iteration 1000.
- The final train accuracies of the three settings are close (0.9574 and 0.9556 for layer, 0.9547 and 0.9555 for batch, 0.9523 and 0.9505 for none).

### Stage 3: Backbone Batch Normalization Momentum (2 launches)

Changed in `Backbone.normalization.batch_normalization_momentum` only: stage 2 kept the layer normalization of the head, which has no running statistics, so `ClassificationHead.batch_normalization_momentum` stays at 0.1 and has no effect. 0.01 is the decay 0.99 of the running statistics of the older TensorFlow classifiers.

| Backbone `batch_normalization_momentum` | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | run_13 (stage 2) | 0.6055 | 0.8918 | 0.9343 | 0.9663 | 0.9352 | 0.9787 | 0.9915 | 0.9909 | 0.9118 |
| 0.1 | run_14 (stage 2) | 0.4637 | 0.7247 | 0.9480 | 0.9823 | 0.9849 | 0.9879 | 0.9799 | 0.9856 | 0.8821 |
| 0.1 | **Mean** | 0.5346 | 0.8083 | 0.9411 | 0.9743 | 0.9601 | 0.9833 | 0.9857 | 0.9882 | **0.8969** |
| 0.01 | run_15 | 0.1109 | 0.7721 | 0.9426 | 0.8553 | 0.9480 | 0.7789 | 0.9841 | 0.9148 | 0.7883 |
| 0.01 | run_16 | 0.1109 | 0.5686 | 0.7023 | 0.9675 | 0.9507 | 0.9697 | 0.9849 | 0.9511 | 0.7757 |
| 0.01 | **Mean** | 0.1109 | 0.6703 | 0.8225 | 0.9114 | 0.9494 | 0.8743 | 0.9845 | 0.9329 | 0.7820 |

Kept: **0.1** (run_13 and run_14), the highest mean (0.8969 against 0.7820) and the highest mean of the last four evaluations (0.9793 against 0.9353); the configuration files of the next stages stay copies of `run-13.yaml` and `run-14.yaml`.

- The running statistics of 0.01 lag behind the weights: both launches are at 0.1109 (single class) at iteration 250, where their train accuracies match those of 0.1 (0.74 and 0.68 at iteration 400 against 0.74 and 0.70), the momentum changing only the statistics used in evaluation.
- The lag lasts to the end with the constant learning rate, Adam moving the weights by about the learning rate at every step: run_15 drops to 0.7789 at iteration 1500 and 0.9148 at iteration 2000, run_16 to 0.9511 at iteration 2000, where 0.1 ends at 0.9909 and 0.9856. The final train accuracies are the same (0.9520 and 0.9489 against 0.9574 and 0.9556).

### Stage 4: Feature Map Normalization (2 launches)

`Backbone.normalization.feature_map_normalization`: batch normalization of the feature maps (statistics of every channel over the images and positions of the batch, running statistics used in evaluation) or layer normalization over the channels (statistics of every position over its channels, the same in training and evaluation, without running statistics, so that the backbone momentum no longer acts). `run-17.yaml` and `run-18.yaml` are copies of `run-13.yaml` and `run-14.yaml` (kept by stage 3) with `"layer"`.

| `feature_map_normalization` | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| batch | run_13 (stage 2) | 0.6055 | 0.8918 | 0.9343 | 0.9663 | 0.9352 | 0.9787 | 0.9915 | 0.9909 | 0.9118 |
| batch | run_14 (stage 2) | 0.4637 | 0.7247 | 0.9480 | 0.9823 | 0.9849 | 0.9879 | 0.9799 | 0.9856 | 0.8821 |
| batch | **Mean** | 0.5346 | 0.8083 | 0.9411 | 0.9743 | 0.9601 | 0.9833 | 0.9857 | 0.9882 | **0.8969** |
| layer | run_17 (stopped) | 0.1107 | 0.1115 | | | | | | | 0.1111 |
| layer | run_18 (stopped) | 0.1105 | 0.1086 | | | | | | | 0.1096 |
| layer | **Mean** | 0.1106 | 0.1101 | | | | | | | 0.1103 |

Kept: **batch** (run_13 and run_14), layer being stopped at iteration 500; the configuration files of the next stages stay copies of `run-13.yaml` and `run-14.yaml`.

- layer does not learn: both launches stay at the accuracy of a single class up to iteration 500 (train loss 2.20, the log of 9 classes, test accuracy 0.11 at iterations 250 and 500), where batch is at a train accuracy of 0.80 and a train loss of 0.52 at iteration 500.
- layer trains about three times slower (100 iterations in 13 minutes 52 seconds to 16 minutes 15 seconds against 4 minutes 34 seconds to 5 minutes 14 seconds, 28 to 31 Images Per Second against 93) and takes 44.8 gigabytes per launch against 33.4: `ChannelLayerNormalizer` permutes every feature map to channels last, `nn.LayerNorm` makes it contiguous and runs in float32 under autocast, and the next convolution casts it back.
- Both launches were stopped by hand at iteration 500, reached in 72 minutes (about 3 hours 20 minutes more for the pair), their run folders keeping the evaluations of iterations 250 and 500.

### Stage 5: Global Pooling (2 launches)

`ClassificationHead.global_pooling`: "flatten" gives the head every position of the last feature map, "average" the mean of every channel over the positions (128 features). `run-19.yaml` and `run-20.yaml` are copies of `run-13.yaml` and `run-14.yaml` (kept by stage 4) with `"average"`.

| `global_pooling` | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| flatten | run_13 (stage 2) | 0.6055 | 0.8918 | 0.9343 | 0.9663 | 0.9352 | 0.9787 | 0.9915 | 0.9909 | 0.9118 |
| flatten | run_14 (stage 2) | 0.4637 | 0.7247 | 0.9480 | 0.9823 | 0.9849 | 0.9879 | 0.9799 | 0.9856 | 0.8821 |
| flatten | **Mean** | 0.5346 | 0.8083 | 0.9411 | 0.9743 | 0.9601 | 0.9833 | 0.9857 | 0.9882 | **0.8969** |
| average | run_19 | 0.4872 | 0.5882 | 0.8437 | 0.9266 | 0.9883 | 0.9300 | 0.9908 | 0.9848 | 0.8425 |
| average | run_20 | 0.6939 | 0.8891 | 0.9646 | 0.9807 | 0.9832 | 0.9812 | 0.9794 | 0.9626 | 0.9293 |
| average | **Mean** | 0.5905 | 0.7387 | 0.9042 | 0.9536 | 0.9857 | 0.9556 | 0.9851 | 0.9737 | 0.8859 |

Kept: **flatten** (run_13 and run_14); the configuration files of the next stages stay copies of `run-13.yaml` and `run-14.yaml`.

- average reaches a mean of 0.8859 against 0.8969 for flatten, and a mean of the last 4 evaluations (iterations 1250 to 2000) of 0.9750 against 0.9793.
- The two average launches are far apart: run_20 is ahead of both flatten launches at iterations 250 to 750 (0.6939, 0.8891 and 0.9646), while run_19 starts slowly (0.5882 at iteration 500) and drops to 0.9300 at iteration 1500; run_20 drops to 0.9626 at iteration 2000. The gap between the two launches of a setting is larger than the gap between the two settings.
- The training time does not change: the pair took 2 hours 19 minutes (100 iterations in 4 minutes 33 seconds to 10 minutes 15 seconds), against 2 hours 19 to 2 hours 25 minutes for run_13 and run_14.

### Stage 6: Residual Connections (2 launches)

`Backbone.modules.residual_nets`: True adds the input of every pair of convolution layers to its output (a shortcut every two layers, from the second convolution block on), False keeps the convolution layers stacked without shortcuts. `run-21.yaml` and `run-22.yaml` are copies of `run-13.yaml` and `run-14.yaml` (kept by stage 5) with `False`.

| `residual_nets` | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| True | run_13 (stage 2) | 0.6055 | 0.8918 | 0.9343 | 0.9663 | 0.9352 | 0.9787 | 0.9915 | 0.9909 | 0.9118 |
| True | run_14 (stage 2) | 0.4637 | 0.7247 | 0.9480 | 0.9823 | 0.9849 | 0.9879 | 0.9799 | 0.9856 | 0.8821 |
| True | **Mean** | 0.5346 | 0.8083 | 0.9411 | 0.9743 | 0.9601 | 0.9833 | 0.9857 | 0.9882 | **0.8969** |
| False | run_21 | 0.1098 | 0.1123 | 0.1108 | 0.1122 | 0.1103 | 0.1299 | 0.1087 | 0.1122 | 0.1133 |
| False | run_22 | 0.1123 | 0.1107 | 0.1109 | 0.1109 | 0.1109 | 0.1114 | 0.1115 | 0.1107 | 0.1111 |
| False | **Mean** | 0.1111 | 0.1115 | 0.1109 | 0.1115 | 0.1106 | 0.1207 | 0.1101 | 0.1114 | 0.1122 |

Kept: **True** (run_13 and run_14); the configuration files of the next stages stay copies of `run-13.yaml` and `run-14.yaml`.

- False does not learn: both launches stay at the accuracy of a single class up to iteration 2000 (test accuracy 0.11 at every evaluation, train loss 2.20, the log of 9 classes, train accuracy 0.11), the 61 convolution layers of the backbone being stacked without shortcuts.
- The training time does not change: the pair took 2 hours 14 minutes (100 iterations in about 4 minutes 50 seconds), against 2 hours 19 to 2 hours 25 minutes for run_13 and run_14.

### Stage 7: Squeeze And Excitation (2 launches)

`Backbone.modules.squeeze_and_excite`: True recalibrates the channels after every convolution layer (each channel scaled by a weight computed from the mean of every channel over the positions, before the residual shortcut is added), False keeps the convolution outputs as they are. `run-23.yaml` and `run-24.yaml` are copies of `run-13.yaml` and `run-14.yaml` (kept by stage 6) with `False`.

| `squeeze_and_excite` | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| True | run_13 (stage 2) | 0.6055 | 0.8918 | 0.9343 | 0.9663 | 0.9352 | 0.9787 | 0.9915 | 0.9909 | 0.9118 |
| True | run_14 (stage 2) | 0.4637 | 0.7247 | 0.9480 | 0.9823 | 0.9849 | 0.9879 | 0.9799 | 0.9856 | 0.8821 |
| True | **Mean** | 0.5346 | 0.8083 | 0.9411 | 0.9743 | 0.9601 | 0.9833 | 0.9857 | 0.9882 | 0.8969 |
| False | run_23 | 0.4871 | 0.8713 | 0.9362 | 0.9721 | 0.9807 | 0.9744 | 0.9826 | 0.9916 | 0.8995 |
| False | run_24 | 0.5368 | 0.7841 | 0.9642 | 0.9788 | 0.9808 | 0.9879 | 0.9900 | 0.9892 | 0.9015 |
| False | **Mean** | 0.5119 | 0.8277 | 0.9502 | 0.9755 | 0.9807 | 0.9811 | 0.9863 | 0.9904 | **0.9005** |

Kept: **False** (run_23 and run_24); the configuration files of the next stages are copies of `run-23.yaml` and `run-24.yaml`.

- False reaches a mean of 0.9005 against 0.8969 for True, and a mean of the last 4 evaluations (iterations 1250 to 2000) of 0.9847 against 0.9793, both launches being at 0.97 or more from iteration 1000 on.
- False trains faster and lighter: the pair took 1 hour 49 minutes (100 iterations in 3 minutes 20 seconds to 8 minutes 17 seconds) against 2 hours 19 to 2 hours 25 minutes for run_13 and run_14, with 20.4 gigabytes per launch against 33.4.

### Stage 8: Data Type (2 launches)

`ExperimentConfiguration.dtype` and `TrainingConfiguration.mixed_precision`: float32 with mixed precision runs the training and validation forward passes in bfloat16, the weights and the test evaluation staying in float32; float32 without mixed precision runs everything in float32. `run-25.yaml` and `run-26.yaml` are copies of `run-23.yaml` and `run-24.yaml` (kept by stage 7) with `mixed_precision: False`. float64 was removed: a float64 launch trains at 9.4 Images Per Second and takes 64 gigabytes (measured over the first 10 iterations of run_27), that is about 15 hours of training per launch, two float64 launches not fitting in memory together.

| `dtype` | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| float32 (`mixed_precision: True`) | run_23 (stage 7) | 0.4871 | 0.8713 | 0.9362 | 0.9721 | 0.9807 | 0.9744 | 0.9826 | 0.9916 | 0.8995 |
| float32 (`mixed_precision: True`) | run_24 (stage 7) | 0.5368 | 0.7841 | 0.9642 | 0.9788 | 0.9808 | 0.9879 | 0.9900 | 0.9892 | 0.9015 |
| float32 (`mixed_precision: True`) | **Mean** | 0.5119 | 0.8277 | 0.9502 | 0.9755 | 0.9807 | 0.9811 | 0.9863 | 0.9904 | **0.9005** |
| float32 (`mixed_precision: False`) | run_25 | 0.3961 | 0.6566 | 0.9656 | 0.9765 | 0.9840 | 0.9761 | 0.9757 | 0.9838 | 0.8643 |
| float32 (`mixed_precision: False`) | run_26 | 0.4550 | 0.8413 | 0.9358 | 0.9672 | 0.9783 | 0.9901 | 0.9909 | 0.9857 | 0.8930 |
| float32 (`mixed_precision: False`) | **Mean** | 0.4255 | 0.7490 | 0.9507 | 0.9718 | 0.9812 | 0.9831 | 0.9833 | 0.9848 | 0.8787 |

Kept: **float32 (`mixed_precision: True`)** (run_23 and run_24), chosen by the user whatever the results of run_25 and run_26: mixed precision trains about 1.8 times faster with comparable accuracy; the configuration files of the next stages stay copies of `run-23.yaml` and `run-24.yaml`.

- Without mixed precision the mean is 0.8787 against 0.9005 and the mean of the last 4 evaluations (iterations 1250 to 2000) 0.9831 against 0.9847: float32 starts slower (0.4255 against 0.5119 at iteration 250, 0.7490 against 0.8277 at iteration 500), then both data types are level from iteration 750 on.
- Without mixed precision the pair took 2 hours 48 minutes against 1 hour 49 minutes (100 iterations in 6 minutes 16 seconds to 7 minutes 1 second at iteration 100, 68.0 and 60.8 Images Per Second, against 3 minutes 20 seconds to 3 minutes 57 seconds), with 38.7 gigabytes per launch against 20.4.

### Stage 9: Class Weighting (2 launches)

`ClassificationHead.class_weighting`: "balanced" makes every class weigh the same in the loss (the weight of a class being inversely proportional to its number of training samples), "none" makes every sample weigh the same. `run-29.yaml` and `run-30.yaml` are copies of `run-23.yaml` and `run-24.yaml` (kept by stage 8) with `"none"`. The training classes are almost balanced (55,010 to 56,166 of the 500,000 images), so the balanced weights only range from 0.989 to 1.010 and the two settings should differ by the gap between launches only.

| `class_weighting` | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| balanced | run_23 (stage 7) | 0.4871 | 0.8713 | 0.9362 | 0.9721 | 0.9807 | 0.9744 | 0.9826 | 0.9916 | 0.8995 |
| balanced | run_24 (stage 7) | 0.5368 | 0.7841 | 0.9642 | 0.9788 | 0.9808 | 0.9879 | 0.9900 | 0.9892 | 0.9015 |
| balanced | **Mean** | 0.5119 | 0.8277 | 0.9502 | 0.9755 | 0.9807 | 0.9811 | 0.9863 | 0.9904 | **0.9005** |
| none | run_29 | 0.5482 | 0.9276 | 0.9726 | 0.9341 | 0.9887 | 0.9841 | 0.9905 | 0.9922 | 0.9173 |
| none | run_30 | 0.4099 | 0.5849 | 0.9438 | 0.9807 | 0.8896 | 0.9864 | 0.9901 | 0.9792 | 0.8456 |
| none | **Mean** | 0.4790 | 0.7562 | 0.9582 | 0.9574 | 0.9392 | 0.9853 | 0.9903 | 0.9857 | 0.8814 |

Kept: **balanced** (run_23 and run_24); the configuration files of the next stage stay copies of `run-23.yaml` and `run-24.yaml`.

- none reaches a mean of 0.8815 against 0.9005 for balanced, and a mean of the last 4 evaluations (iterations 1250 to 2000) of 0.9751 against 0.9847.
- The two none launches are far apart: run_29 has the second highest mean of all launches (0.9173 after 0.9293 for run_20, 0.9276 at iteration 500), run_30 a mean of 0.8456 (0.5849 at iteration 500 and 0.8896 at iteration 1250). With almost balanced classes, the gap between the two settings is the gap between launches.
- The training time does not change: the pair took 1 hour 49 minutes (100 iterations in 3 minutes 20 seconds to 3 minutes 59 seconds at iteration 100), with 20.4 gigabytes per launch.

### Stage 10: Dropout (2 launches)

`ClassificationHead.dropout_rate`: probability of zeroing every input of the fully connected layers of the head during training, 0.0 disabling the dropout. `run-31.yaml` and `run-32.yaml` are copies of `run-23.yaml` and `run-24.yaml` (kept by stage 9) with `0.0`.

| `dropout_rate` | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | run_23 (stage 7) | 0.4871 | 0.8713 | 0.9362 | 0.9721 | 0.9807 | 0.9744 | 0.9826 | 0.9916 | 0.8995 |
| 0.2 | run_24 (stage 7) | 0.5368 | 0.7841 | 0.9642 | 0.9788 | 0.9808 | 0.9879 | 0.9900 | 0.9892 | 0.9015 |
| 0.2 | **Mean** | 0.5119 | 0.8277 | 0.9502 | 0.9755 | 0.9807 | 0.9811 | 0.9863 | 0.9904 | **0.9005** |
| 0.0 | run_31 | 0.4204 | 0.8280 | 0.9477 | 0.9666 | 0.9728 | 0.9678 | 0.9749 | 0.9906 | 0.8836 |
| 0.0 | run_32 | 0.5555 | 0.8835 | 0.9639 | 0.9856 | 0.9739 | 0.9862 | 0.9928 | 0.9908 | 0.9165 |
| 0.0 | **Mean** | 0.4880 | 0.8558 | 0.9558 | 0.9761 | 0.9734 | 0.9770 | 0.9839 | 0.9907 | 0.9001 |

Kept: **0.2** (run_23 and run_24).

- 0.0 reaches a mean of 0.9001 against 0.9005 for 0.2 (0.900061 against 0.900480), and a mean of the last 4 evaluations (iterations 1250 to 2000) of 0.9812 against 0.9847: the gap between the two settings is far smaller than the gap between the two launches of 0.0 (0.8836 and 0.9165).
- 0.0 is behind at iteration 250 (0.4880 against 0.5119), ahead at iteration 500 (0.8558 against 0.8277) and level at iteration 2000 (0.9907 against 0.9904).
- The training time does not change: the pair took 1 hour 49 minutes (100 iterations in 3 minutes 20 seconds to 3 minutes 59 seconds at iteration 100), with 20.4 gigabytes per launch.

The settings kept by the ten stages are those of `run-23.yaml` and `run-24.yaml`: learning rate 2e-4, fully connected normalization "layer", backbone batch normalization momentum 0.1, feature map batch normalization, flatten pooling, residual connections, no squeeze and excitation, float32 with mixed precision, balanced class weighting and dropout 0.2 (mean 0.9005, mean of the last 4 evaluations 0.9847).

### Repeat Launches: Squeeze And Excitation (8 launches)

Four more launches of each squeeze and excitation setting of stage 7, the other settings being those kept by the ten stages, to measure the gap between launches of the same settings: `run-23b.yaml`, `run-23c.yaml`, `run-24b.yaml` and `run-24c.yaml` are copies of `run-23.yaml` and `run-24.yaml` (squeeze and excitation off), `run-13b.yaml`, `run-13c.yaml`, `run-14b.yaml` and `run-14c.yaml` copies of `run-13.yaml` and `run-14.yaml` (squeeze and excitation on), only their experiment folder changing. They run two at a time, the off repeats first.

| `squeeze_and_excite` | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| False | run_23 | 0.4871 | 0.8713 | 0.9362 | 0.9721 | 0.9807 | 0.9744 | 0.9826 | 0.9916 | 0.8995 |
| False | run_24 | 0.5368 | 0.7841 | 0.9642 | 0.9788 | 0.9808 | 0.9879 | 0.9900 | 0.9892 | 0.9015 |
| False | run_23b | 0.6023 | 0.9428 | 0.9659 | 0.9157 | 0.9812 | 0.9908 | 0.9906 | 0.9848 | 0.9218 |
| False | run_23c | 0.4003 | 0.7537 | 0.9323 | 0.9642 | 0.9834 | 0.9843 | 0.9894 | 0.9927 | 0.8750 |
| False | run_24b | 0.3542 | 0.9186 | 0.9544 | 0.8746 | 0.9849 | 0.9750 | 0.9859 | 0.9816 | 0.8787 |
| False | run_24c | 0.5093 | 0.7220 | 0.9228 | 0.9705 | 0.9576 | 0.9893 | 0.9908 | 0.9856 | 0.8810 |
| False | **Mean** | 0.4816 | 0.8321 | 0.9460 | 0.9460 | 0.9781 | 0.9836 | 0.9882 | 0.9876 | 0.8929 |
| True | run_13 | 0.6055 | 0.8918 | 0.9343 | 0.9663 | 0.9352 | 0.9787 | 0.9915 | 0.9909 | 0.9118 |
| True | run_14 | 0.4637 | 0.7247 | 0.9480 | 0.9823 | 0.9849 | 0.9879 | 0.9799 | 0.9856 | 0.8821 |
| True | run_13b | 0.5505 | 0.8785 | 0.8610 | 0.9793 | 0.9686 | 0.9910 | 0.9893 | 0.9759 | 0.8993 |
| True | run_13c | 0.3752 | 0.7032 | 0.9187 | 0.9271 | 0.9515 | 0.9858 | 0.9821 | 0.9929 | 0.8546 |
| True | run_14b | 0.5184 | 0.9182 | 0.8943 | 0.9322 | 0.9852 | 0.9908 | 0.9921 | 0.9889 | 0.9025 |
| True | run_14c | 0.5222 | 0.8817 | 0.9511 | 0.9678 | 0.9038 | 0.9813 | 0.9648 | 0.9845 | 0.8946 |
| True | **Mean** | 0.5059 | 0.8330 | 0.9179 | 0.9592 | 0.9549 | 0.9859 | 0.9833 | 0.9864 | 0.8908 |

- Over six launches each, squeeze and excitation off and on are level: mean 0.8929 against 0.8908 (standard deviation between launches 0.0180 and 0.0203, a difference of 0.0021 for a standard error of 0.0111), accuracy at iteration 2000 0.9876 against 0.9864.
- The mean of the last 4 evaluations (iterations 1250 to 2000) leans towards off, 0.9844 against 0.9776 (a difference of 0.0068 for a standard error of 0.0045, not significant): the six off launches all end between 0.9808 and 0.9874, the on launches between 0.9586 and 0.9892, run_14c dropping to 0.9038 at iteration 1250 and 0.9648 at iteration 1750.
- The mean of a single launch varies by about 0.02 between launches of the same settings (0.8750 to 0.9218 off, 0.8546 to 0.9118 on), mostly through iterations 250 to 1000; with two launches per setting, the gap between two settings is noise below about 0.036, which covers stages 5, 7, 9 and 10. The mean of the last 4 evaluations varies about six times less for off (0.0030).
- Off stays the faster and lighter setting: the off pairs took 1 hour 51 to 1 hour 55 minutes and the on pairs 2 hours 28 to 2 hours 33 minutes (with another process of the machine on the GPU), with 20.4 gigabytes per launch against 25.2, both read at iteration 100.

## New Settings Stages

Stages 11 to 13 test the settings added after the ten stages, one at a time on top of the settings they kept, with the same fixed settings. The launches run one at a time, next to the training of the large model on ImageNet, which shares the GPU: their speed and memory are not comparable with the earlier stages, their accuracy is. The configuration files are copies of `run-23.yaml` and `run-24.yaml` with the changed parameter, completed with the settings added since (`weight_initialization: "default"`, `exponential_moving_average` off) and without `dtype`, removed since (every model is built in float32). The reference of stage 11 is run_23 and run_24, the six launches of the repeat stage giving the spread of the kept settings (mean 0.8929, standard deviation 0.0180 between launches).

### Stage 11: Pre-Activation Residual Shortcuts (2 launches)

`Backbone.modules.residual_nets_v2`: True places the residual shortcuts of blocks 2 to 6 before the activations (paper "Identity Mappings in Deep Residual Networks"): every layer applies its batch normalization and ReLU before its convolution, nothing follows the addition of the shortcut, so that the shortcuts form an identity path through the block, and every block ends with one batch normalization and ReLU. False keeps the original placement: convolution, batch normalization and ReLU, the shortcut being added before the ReLU of the second layer of every pair. Block 1, without shortcut, keeps the original order in both. `run-33.yaml` and `run-34.yaml` are copies of `run-23.yaml` and `run-24.yaml` with `True` (run name token `RES2` instead of `RES`).

| `residual_nets_v2` | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| False | run_23 (stage 7) | 0.4871 | 0.8713 | 0.9362 | 0.9721 | 0.9807 | 0.9744 | 0.9826 | 0.9916 | 0.8995 |
| False | run_24 (stage 7) | 0.5368 | 0.7841 | 0.9642 | 0.9788 | 0.9808 | 0.9879 | 0.9900 | 0.9892 | 0.9015 |
| False | **Mean** | 0.5119 | 0.8277 | 0.9502 | 0.9755 | 0.9807 | 0.9811 | 0.9863 | 0.9904 | 0.9005 |
| False | **Mean of the 6 launches** (repeat stage) | 0.4816 | 0.8321 | 0.9460 | 0.9460 | 0.9781 | 0.9836 | 0.9882 | 0.9876 | 0.8929 |
| True | run_33 | 0.5694 | 0.8862 | 0.9477 | 0.9676 | 0.9430 | 0.9450 | 0.9908 | 0.9876 | 0.9047 |
| True | run_34 | 0.5536 | 0.8735 | 0.7272 | 0.9786 | 0.9756 | 0.9882 | 0.9567 | 0.9890 | 0.8803 |
| True | **Mean** | 0.5615 | 0.8798 | 0.8375 | 0.9731 | 0.9593 | 0.9666 | 0.9738 | 0.9883 | 0.8925 |

**Kept: False.** The pre-activation mean (0.8925) is 0.008 under the reference pair and level with the mean of the 6 repeat launches (0.8929), a gap well inside the noise between two-launch means (about 0.036). Its curves drop more often between evaluations (run_34 at 750 and 1750, run_33 at 1250 and 1500), and a launch takes 28.1 GB of GPU memory instead of 20.4 GB, so the original placement stays.

### Stage 12: Weight Initialisation (2 launches)

`Optimization.weight_initialization`: "he" draws the convolutions of the convolution blocks and the hidden fully connected layers of the head from the He normal distribution (standard deviation √(2 / fan_in), biases at 0), the class layer keeping the PyTorch initialisation; "default" keeps the PyTorch initialisation of every layer (uniform, standard deviation √(1 / (3 × fan_in)), about 2.45 times smaller). `run-35.yaml` and `run-36.yaml` are copies of the files kept by stage 11 with `"he"`, the launches kept by stage 11 being the "default" reference.

| `weight_initialization` | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| default | run_23 (stage 7) | 0.4871 | 0.8713 | 0.9362 | 0.9721 | 0.9807 | 0.9744 | 0.9826 | 0.9916 | 0.8995 |
| default | run_24 (stage 7) | 0.5368 | 0.7841 | 0.9642 | 0.9788 | 0.9808 | 0.9879 | 0.9900 | 0.9892 | 0.9015 |
| default | **Mean** | 0.5119 | 0.8277 | 0.9502 | 0.9755 | 0.9807 | 0.9811 | 0.9863 | 0.9904 | 0.9005 |
| he | run_35 | 0.3929 | 0.7284 | 0.8430 | 0.6248 | 0.9223 | 0.9834 | 0.9401 | 0.9732 | 0.8010 |
| he | run_36 | 0.3097 | 0.7188 | 0.8806 | 0.9063 | 0.9716 | 0.9744 | 0.9767 | 0.9859 | 0.8405 |
| he | **Mean** | 0.3513 | 0.7236 | 0.8618 | 0.7656 | 0.9470 | 0.9789 | 0.9584 | 0.9796 | 0.8208 |

**Kept: default.** The He mean (0.8208) is 0.080 under the reference pair and 0.072 under the mean of the 6 repeat launches, twice the noise between two-launch means (about 0.036). Both launches start slower (0.35 against 0.51 at iteration 250), run_35 drops to 0.6248 at iteration 1000, and the final accuracy (0.9796) stays under the reference (0.9904), so the default initialisation stays. The `weight_initialization` setting and the He initialisation were then removed from the code; `run-35.yaml` and `run-36.yaml` were deleted with them, the results above staying as the record of the stage.

### Stage 13: Moving Average Of The Weights (2 launches)

`Optimization.exponential_moving_average.activated`: True keeps a moving average of the weights, `ema_weights = decay × ema_weights + (1 − decay) × current_weights` after every optimizer step, the configured decay being 0.9998. `run-37.yaml` and `run-38.yaml` are copies of the files kept by stage 12 with `True`. Every launch evaluates its trained weights, then its averaged weights, at every checkpoint (`Tensorboard/Test` and `Tensorboard/Test_EMA`), so that it gives both a repeat launch of the kept settings and the curve of the averaged weights; `collect_experiment_results.py` writes both, its `evaluated_weights` column telling them apart.

In 2,000 iterations, the decay of the n-th update, min(decay, (1 + n) / (10 + n)), never exceeds 0.9955 (iteration 2000): every decay above 0.9955 gives the same average, which follows about the last (10 + n) / 9 iterations (about 220 at iteration 2000). This stage tests the average with its warm-up; the decay itself needs longer trainings, tested later.

| Evaluated weights | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| trained (EMA off) | run_23 (stage 7) | 0.4871 | 0.8713 | 0.9362 | 0.9721 | 0.9807 | 0.9744 | 0.9826 | 0.9916 | 0.8995 |
| trained (EMA off) | run_24 (stage 7) | 0.5368 | 0.7841 | 0.9642 | 0.9788 | 0.9808 | 0.9879 | 0.9900 | 0.9892 | 0.9015 |
| trained (EMA off) | **Mean** | 0.5119 | 0.8277 | 0.9502 | 0.9755 | 0.9807 | 0.9811 | 0.9863 | 0.9904 | 0.9005 |
| trained (EMA on) | run_37 | 0.4896 | 0.8186 | 0.9571 | 0.9757 | 0.9794 | 0.9854 | 0.9913 | 0.9906 | 0.8985 |
| trained (EMA on) | run_38 | 0.4672 | 0.8123 | 0.9752 | 0.9083 | 0.9611 | 0.9853 | 0.9908 | 0.9908 | 0.8864 |
| trained (EMA on) | **Mean** | 0.4784 | 0.8155 | 0.9662 | 0.9420 | 0.9702 | 0.9854 | 0.9911 | 0.9907 | 0.8924 |
| EMA | run_37, source | 0.5675 | 0.8761 | 0.9832 | 0.9899 | 0.9928 | 0.9953 | 0.9962 | 0.9968 | 0.9247 |
| EMA | run_38 | 0.5074 | 0.9295 | 0.9841 | 0.9904 | 0.9925 | 0.9940 | 0.9955 | 0.9957 | 0.9236 |
| EMA | **Mean** | 0.5374 | 0.9028 | 0.9836 | 0.9901 | 0.9927 | 0.9946 | 0.9959 | 0.9962 | 0.9242 |

**Kept: True.** The averaged weights are ahead of the trained weights at all 16 evaluations of the two launches, with a mean of 0.9242 against 0.8924 (0.9005 for the reference pair without average) and 0.9962 against 0.9907 at iteration 2000, the errors falling from 0.9 % to 0.4 %. Their two curves stay within 0.001 of each other from iteration 750, where the trained weights of the two launches differ by up to 0.067 (run_38 dropping to 0.9083 at iteration 1000), and the trained weights of the launches keeping the average stay within the noise of the reference, the average never feeding back into the training. An audit of run_37 found no leakage: no test image is identical to a training image, the evaluation leaves every buffer of the averaged model unchanged, and the checkpoints of iterations 500, 1000 and 2000, scored again outside the trainer, give the logged accuracies. Swapping the batch normalization statistics between the two sets of weights shows that the averaged parameters bring most of the gain from iteration 1000 (0.9899 and 0.9955 with the statistics of the trained weights at iterations 1000 and 2000), the averaged statistics mattering early (0.8212 instead of 0.8760 at iteration 500). The launches keep a constant learning rate, whose noise the average removes: a decaying learning rate would leave a smaller gap, and the decay of 0.9998, never reached in 2,000 iterations, needs longer trainings.

## Class Extension Stage

Stage 14 tests the class extension, which lets a trained run initialise a run with more classes (`pretrained` blocks of `Backbone` and `ClassificationHead`, README section Adding Classes). Its launches train on `synthetic_shapes_20`, a dataset of 20 classes with the same split sizes (500,000 train / 50,000 validation / 50,000 test images): the 9 classes of the earlier stages keep their identifiers 1 to 9 and come first, followed by rhombus, parallelogram, trapezoid, kite, heptagon, nonagon, ellipse, semicircle, annulus, capsule and l_shape, every shape measuring at least 64 pixels. A first launch on 21 classes, with a right triangle, confused the triangles with the right triangles (triangle recall 0.12 with the trained weights at iteration 1500), so the right triangle was removed and the dataset generated again. The launches use the settings of `run-37.yaml` (moving average on) with `per_class_report: True`, and ran two at a time on the GPU, the ImageNet training being stopped: their duration is not comparable with the earlier stages.

### Stage 14: Adding Classes (6 launches)

Every launch starts from iteration 2000 of run_37, read from its averaged weights since the launches keep a moving average, nothing being frozen. `run-39.yaml` and `run-40.yaml` take the backbone of run_37 and a random classification head of 20 classes (run name token `PT`); `run-41.yaml` and `run-42.yaml` take its backbone and its classification head (token `HPT`), the hidden layers `[256, 128]` and the 9 class rows of the head being copied, the 11 new rows starting at zero weights with a bias 3 below the mean of the 9 copied biases (`new_class_initialisation: "source"`). `run-43.yaml` and `run-44.yaml` copy the same backbone, hidden layers and 9 class rows, the 11 new rows keeping the random initialisation of the new run (`new_class_initialisation: "random"`, token `HRND`). The random head is the reference: the transfer of the head succeeds if it keeps the 9 classes without slowing the 11 new ones. The first table gives the accuracy of the 20 classes (`results.csv`), the two others the Mean rows of the 9 classes of run_37 and of the 11 new classes, computed from the recall and the number of test images of every class in the `class_metrics.csv` of every evaluation.

All 20 classes:

| Evaluated weights | Head | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| trained | random | run_39 | 0.8495 | 0.9091 | 0.9512 | 0.9779 | 0.9847 | 0.9844 | 0.9934 | 0.9881 | 0.9548 |
| trained | random | run_40 | 0.8622 | 0.7718 | 0.9073 | 0.9758 | 0.9909 | 0.9655 | 0.9910 | 0.9937 | 0.9323 |
| trained | random | **Mean** | 0.8558 | 0.8405 | 0.9292 | 0.9769 | 0.9878 | 0.9749 | 0.9922 | 0.9909 | 0.9435 |
| trained | run_37, source | run_41 | 0.6039 | 0.7743 | 0.8013 | 0.9114 | 0.9371 | 0.9656 | 0.9638 | 0.9789 | 0.8670 |
| trained | run_37, source | run_42 | 0.6101 | 0.7161 | 0.8149 | 0.9003 | 0.9240 | 0.9418 | 0.9595 | 0.9846 | 0.8564 |
| trained | run_37, source | **Mean** | 0.6070 | 0.7452 | 0.8081 | 0.9058 | 0.9305 | 0.9537 | 0.9616 | 0.9818 | 0.8617 |
| trained | run_37, random | run_43 | 0.8553 | 0.9470 | 0.9490 | 0.9692 | 0.9868 | 0.9824 | 0.9564 | 0.9966 | 0.9553 |
| trained | run_37, random | run_44 | 0.8245 | 0.9168 | 0.9484 | 0.9721 | 0.9579 | 0.9907 | 0.9954 | 0.9955 | 0.9502 |
| trained | run_37, random | **Mean** | 0.8399 | 0.9319 | 0.9487 | 0.9707 | 0.9723 | 0.9866 | 0.9759 | 0.9961 | 0.9528 |
| EMA | random | run_39 | 0.9156 | 0.9701 | 0.9916 | 0.9971 | 0.9989 | 0.9994 | 0.9993 | 0.9996 | 0.9839 |
| EMA | random | run_40 | 0.9034 | 0.9449 | 0.9816 | 0.9935 | 0.9984 | 0.9990 | 0.9995 | 0.9997 | 0.9775 |
| EMA | random | **Mean** | 0.9095 | 0.9575 | 0.9866 | 0.9953 | 0.9986 | 0.9992 | 0.9994 | 0.9997 | 0.9807 |
| EMA | run_37, source | run_41 | 0.6630 | 0.7951 | 0.8630 | 0.9162 | 0.9444 | 0.9641 | 0.9889 | 0.9959 | 0.8913 |
| EMA | run_37, source | run_42 | 0.6629 | 0.7943 | 0.8778 | 0.9202 | 0.9435 | 0.9527 | 0.9706 | 0.9910 | 0.8892 |
| EMA | run_37, source | **Mean** | 0.6630 | 0.7947 | 0.8704 | 0.9182 | 0.9440 | 0.9584 | 0.9798 | 0.9935 | 0.8902 |
| EMA | run_37, random | run_43 | 0.9093 | 0.9712 | 0.9875 | 0.9959 | 0.9986 | 0.9994 | 0.9996 | 0.9996 | 0.9826 |
| EMA | run_37, random | run_44 | 0.8899 | 0.9491 | 0.9773 | 0.9917 | 0.9968 | 0.9990 | 0.9995 | 0.9998 | 0.9754 |
| EMA | run_37, random | **Mean** | 0.8996 | 0.9601 | 0.9824 | 0.9938 | 0.9977 | 0.9992 | 0.9996 | 0.9997 | 0.9790 |

The 9 classes of run_37:

| Evaluated weights | Head | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| trained | random | 0.9569 | 0.9194 | 0.9559 | 0.9903 | 0.9938 | 0.9958 | 0.9960 | 0.9902 | 0.9748 |
| trained | run_37, source | 0.9665 | 0.9596 | 0.9927 | 0.9943 | 0.9969 | 0.9958 | 0.9958 | 0.9976 | 0.9874 |
| trained | run_37, random | 0.9488 | 0.9837 | 0.9907 | 0.9736 | 0.9762 | 0.9939 | 0.9554 | 0.9985 | 0.9776 |
| EMA | random | 0.9980 | 0.9992 | 0.9996 | 0.9997 | 0.9997 | 0.9999 | 0.9998 | 0.9999 | 0.9995 |
| EMA | run_37, source | 0.9997 | 0.9997 | 0.9995 | 0.9998 | 0.9999 | 0.9998 | 0.9998 | 0.9999 | 0.9998 |
| EMA | run_37, random | 0.9986 | 0.9994 | 0.9996 | 0.9998 | 0.9999 | 0.9998 | 0.9999 | 0.9999 | 0.9996 |

The 11 new classes:

| Evaluated weights | Head | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| trained | random | 0.7736 | 0.7762 | 0.9076 | 0.9660 | 0.9829 | 0.9579 | 0.9892 | 0.9915 | 0.9181 |
| trained | run_37, source | 0.3144 | 0.5707 | 0.6579 | 0.8339 | 0.8765 | 0.9194 | 0.9338 | 0.9689 | 0.7594 |
| trained | run_37, random | 0.7513 | 0.8898 | 0.9145 | 0.9683 | 0.9692 | 0.9805 | 0.9926 | 0.9941 | 0.9325 |
| EMA | random | 0.8375 | 0.9236 | 0.9761 | 0.9916 | 0.9977 | 0.9987 | 0.9991 | 0.9994 | 0.9655 |
| EMA | run_37, source | 0.3890 | 0.6279 | 0.7654 | 0.8518 | 0.8985 | 0.9247 | 0.9634 | 0.9882 | 0.8011 |
| EMA | run_37, random | 0.8190 | 0.9282 | 0.9684 | 0.9889 | 0.9960 | 0.9987 | 0.9993 | 0.9995 | 0.9623 |

**Kept: run_37 head, random new classes.** With the source initialisation, the transferred head keeps the 9 classes of run_37: its averaged weights classify them at 0.9997 from iteration 250 (0.9980 for the random head), its trained weights at 0.9665 against 0.9569, a gap within the spread of the launches (run_42 dropping to 0.9279 at iteration 500, run_40 to 0.8544). But it slows the 11 new classes, at 0.3144 against 0.7736 with the trained weights at iteration 250 and 0.3890 against 0.8375 with the averaged weights, and still at 0.9689 against 0.9915 and 0.9882 against 0.9994 at iteration 2000. The 20 classes stay behind the random head at every evaluation, mean 0.8902 against 0.9807 with the averaged weights and 0.8617 against 0.9435 with the trained ones, the two launches of each head being within 0.007 of each other. In run_41 at iteration 250 the ellipses (0.99), the annuli (1.00) and most semicircles (0.62) are predicted as capsules, the rhombi, parallelograms and kites as l_shapes or trapezoids, the nonagons as octagons (0.99), and the octagons lose 38 % of their images to the heptagons, while the random head of run_39 already gives the annuli a recall of 0.998, the ellipses 0.96 and the nonagons 0.87.

run_43 and run_44 copy the same hidden layers and 9 class rows as run_41 and run_42, only the 11 new rows keeping the random initialisation of the new run, and their new classes learn as fast as with the random head: 0.7513 at iteration 250 with the trained weights (0.7736 for the random head, 0.3144 for the source initialisation), a mean of 0.9325 (0.9181, 0.7594), and 0.8190 at iteration 250 and a mean of 0.9623 with the averaged weights (0.8375 and 0.9655 for the random head). The hidden layers trained on 9 classes are therefore not what slowed the new classes, but the initialisation of their rows, the likely cause being that the zero rows of the source initialisation all start in the same direction with quiet biases, while random rows split the new classes along different directions of the hidden features from the first iteration. The 9 classes of run_37 stay at 0.9986 with the averaged weights at iteration 250 (0.9997 for the source initialisation, 0.9980 for the random head) and at a mean of 0.9776 with the trained weights (0.9874, 0.9748), the trained weights of run_44 starting lower (0.9141 at iteration 250) within the spread of the launches. On the 20 classes, the random new classes match the random head, a mean of 0.9528 against 0.9435 with the trained weights and 0.9790 against 0.9807 with the averaged ones, the launches of each head differing by as much (0.9553 and 0.9502, 0.9548 and 0.9323), and they beat the source initialisation at every evaluation (0.9790 against 0.8902 with the averaged weights). On this dataset, whose backbone already separates the shapes, keeping the classification head gains nothing measurable over a random head once its new classes start random; with "source", it costs the new classes 1,000 to 1,500 iterations. The source initialisation was removed after this stage, with the `new_class_initialisation` key and the `HRND`, `DRND` and `MRND` tokens: the new classes of a component with class layers now always keep the random initialisation of the new run. The run names of run_41 and run_42 are unchanged, so that `run-41.yaml` and `run-42.yaml` launched again into their folders resume them, and in other folders train as `run-43.yaml` and `run-44.yaml`; the folders of run_43 and run_44 keep the `HRND` token their run name had when they were trained, which a launch of `run-43.yaml` or `run-44.yaml` no longer gives, starting a new run beside them.

## Total

18 settings, 36 launches (44 with the repeat launches), against 30720 combinations (61440 launches) for every combination.

## Notes

- Every launch has its own `run_<n>` folder, the run folder named after the settings being created inside it, so no launch resumes the checkpoints of another one. `resume_training: true` only resumes a launch restarted in its own `run_<n>` folder, e.g. after a crash.
- The mean of the eight evaluations rewards the settings that learn fast as well as the ones that end high. The 2000 column alone gives the final accuracy.
- If the two launches of a setting differ by about as much as two settings differ from each other, the difference between those settings is noise, and more launches are needed to separate them.
