# Experiment Plan

- **Configurations:** `vision_training/configurations/classification/run-<n>.yaml`, one per launch (`run-1.yaml` to `run-10.yaml` for stage 1), launched with `--task="classification" --name="run-<n>.yaml"`; the files of the later stages are copies of the kept one with the changed parameter
- **Dataset:** synthetic shapes classification, 9 classes, 500,000 train / 50,000 validation / 50,000 test images, resized to 256 pixels
- **Launch folders:** `hyperparameter_experiments/run_1` to `run_32` at the repository root, set as `ExperimentConfiguration.experiment_folder` before every launch

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

Every launch gives its test accuracy at iterations 250, 500, 750, 1000, 1250, 1500, 1750 and 2000 (`Results.txt` of its run folder) and the mean of these eight values, all gathered in `hyperparameter_experiments/results.csv` by `uv run python -m scripts.evaluation.collect_experiment_results`, launches and Mean rows as in the tables below. The Mean row of a setting averages its two launches, and the setting with the highest mean of the Mean row is kept.

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
| 0.1 | **Mean** | 0.5346 | 0.8083 | 0.9411 | 0.9743 | 0.9601 | 0.9833 | 0.9857 | 0.9882 | 0.8969 |
| 0.01 | run_15 | | | | | | | | | |
| 0.01 | run_16 | | | | | | | | | |
| 0.01 | **Mean** | | | | | | | | | |

Kept:

### Stage 4: Feature Map Normalization (2 launches)

| `feature_map_normalization` | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| batch | best so far, A | | | | | | | | | |
| batch | best so far, B | | | | | | | | | |
| batch | **Mean** | 0.1584 | 0.4774 | 0.6044 | 0.9349 | 0.9667 | 0.9782 | 0.9850 | 0.9864 | 0.7614 |
| layer | run_17 | | | | | | | | | |
| layer | run_18 | | | | | | | | | |
| layer | **Mean** | | | | | | | | | |

Kept:

### Stage 5: Global Pooling (2 launches)

| `global_pooling` | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| flatten | best so far, A | | | | | | | | | |
| flatten | best so far, B | | | | | | | | | |
| flatten | **Mean** | | | | | | | | | |
| average | run_19 | | | | | | | | | |
| average | run_20 | | | | | | | | | |
| average | **Mean** | | | | | | | | | |

Kept:

### Stage 6: Residual Connections (2 launches)

| `residual_nets` | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| True | best so far, A | | | | | | | | | |
| True | best so far, B | | | | | | | | | |
| True | **Mean** | | | | | | | | | |
| False | run_21 | | | | | | | | | |
| False | run_22 | | | | | | | | | |
| False | **Mean** | | | | | | | | | |

Kept:

### Stage 7: Squeeze And Excitation (2 launches)

| `squeeze_and_excite` | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| True | best so far, A | | | | | | | | | |
| True | best so far, B | | | | | | | | | |
| True | **Mean** | | | | | | | | | |
| False | run_23 | | | | | | | | | |
| False | run_24 | | | | | | | | | |
| False | **Mean** | | | | | | | | | |

Kept:

### Stage 8: Data Type (4 launches)

| `dtype` | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| float32 (`mixed_precision: True`) | best so far, A | | | | | | | | | |
| float32 (`mixed_precision: True`) | best so far, B | | | | | | | | | |
| float32 (`mixed_precision: True`) | **Mean** | | | | | | | | | |
| float32 (`mixed_precision: False`) | run_25 | | | | | | | | | |
| float32 (`mixed_precision: False`) | run_26 | | | | | | | | | |
| float32 (`mixed_precision: False`) | **Mean** | | | | | | | | | |
| float64 (`mixed_precision: False`) | run_27 | | | | | | | | | |
| float64 (`mixed_precision: False`) | run_28 | | | | | | | | | |
| float64 (`mixed_precision: False`) | **Mean** | | | | | | | | | |

Kept:

### Stage 9: Class Weighting (2 launches)

| `class_weighting` | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| balanced | best so far, A | | | | | | | | | |
| balanced | best so far, B | | | | | | | | | |
| balanced | **Mean** | | | | | | | | | |
| none | run_29 | | | | | | | | | |
| none | run_30 | | | | | | | | | |
| none | **Mean** | | | | | | | | | |

Kept:

### Stage 10: Dropout (2 launches)

| `dropout_rate` | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | best so far, A | | | | | | | | | |
| 0.2 | best so far, B | | | | | | | | | |
| 0.2 | **Mean** | | | | | | | | | |
| 0.0 | run_31 | | | | | | | | | |
| 0.0 | run_32 | | | | | | | | | |
| 0.0 | **Mean** | | | | | | | | | |

Kept:

## Total

16 settings, 32 launches, against 5760 combinations (11520 launches) for every combination.

## Notes

- Every launch has its own `run_<n>` folder, the run folder named after the settings being created inside it, so no launch resumes the checkpoints of another one. `resume_training: true` only resumes a launch restarted in its own `run_<n>` folder, e.g. after a crash.
- The mean of the eight evaluations rewards the settings that learn fast as well as the ones that end high. The 2000 column alone gives the final accuracy.
- If the two launches of a setting differ by about as much as two settings differ from each other, the difference between those settings is noise, and more launches are needed to separate them.
