# Sequence Labelling Experiment Plan

- **Configurations:** `vision_training/configurations/sequence_labelling/run-<n>.yaml`, one per launch (`run-1.yaml` to `run-8.yaml` for stage 1), launched with `uv run python -m vision_training.main --configuration_path=vision_training/configurations/sequence_labelling/run-<n>.yaml`; the files of the later stages are copies of `run-1.yaml` with the settings kept by the previous stages and the changed parameter
- **Dataset:** synthetic changing shapes (`create_synthetic_dataset(task="sequence_labelling")`), frames of 5 to 10 shapes drawn again at every frame, every frame from the second one adding a shape, removing one or keeping them; 71 labels (`appearing_<class>` and `disappearing_<class>` for the 35 shape classes, and `no_change`, always the label of the first frame); 250,000 train / 50,000 validation / 50,000 test sequences of 2 to 12 frames of 512 pixels, resized to 256 pixels
- **Launch folders:** `hyperparameter_experiments/sequence_labelling/run_<n>` at the repository root, set as `ExperimentConfiguration.experiment_folder` before every launch

## Strategy

One parameter changes at a time. The best value of a stage is kept for all the following stages, and its launches serve as the reference of the next stage, so they are not trained again when that stage trains as many iterations. A stage training another number of iterations trains its reference again from scratch (two new launches, never a resumed one), so that every compared launch starts from a new initialisation. The short learning rate stage (2000 iterations) chooses the learning rate, and the following stages train 6000 iterations: the class weighting stage trains both of its values, and the later stages reuse its kept launches.

Every setting is launched twice, A and B, to smooth out the noise of a single training, the two launches of a setting training at the same time on the GPU. The training sets no random seed, so the two launches start from different weights, data order and augmentations.

Every launch keeps a moving average of its weights, and the settings are compared on the averaged weights: the mean of the test Macro F1 of the averaged weights over all the evaluations of the launch, at iterations 250 to 2000 for the learning rate stage and 250 to 6000 for the following stages (`Tensorboard/Test_EMA`), averaged over the two launches. The Macro F1 weighs the frames of every change as much as the far more frequent frames without change, whose share makes the frame accuracy high for a model that never finds a change.

## Fixed Settings

- `batch_size`: 64 sequences
- `number_iterations`: 2000 for the learning rate stage (128,000 sequences, about half an epoch of the 250,000 training sequences), 6000 for the following stages (384,000 sequences, about 1.5 epochs)
- `weight_saving_iterations`: 250, the test evaluation running every 250 iterations (8 tested models at 2000 iterations, 24 at 6000) on `test_10000`, a fixed subset of 10,000 of the 50,000 test sequences (69,918 frames, every label present) drawn with the seed 0 by `scripts/data/create_sequence_split_subset.py`, the evaluation of the whole test split taking 26 minutes
- `intermediate_weight_saving_iterations`: 125, as many more checkpoints saved without a test evaluation
- `number_workers`: 2 per loader, `prefetch_batches`: 2, the expandable segments of the CUDA allocator turned on by the training launch (two launches side by side stay at about 55 Gigabytes of the 121 of the machine); `Data.loading` has since been removed from the configurations, every loader now using 4 workers each prefetching 2 batches
- `information_dump`: 250, `validation_iteration_ratio`: 0 (no validation during the training)
- `maximum_sequence_length`: 50, every sequence fed whole, the sequences of a batch padded to the longest one
- Backbone `{"1": [1, 16], "2": [2, 16], "3": [2, 16], "4": [4, 32], "5": [5, 32], "6": [5, 32]}`, residual blocks with squeeze and excite, input normalization
- Temporal module: hidden size 96, dropout 0, positional embedding (transformer), 4 heads (transformer), kernel size 3 (temporal convolution)
- Classification head `[96, 96]` with layer normalization, dropout 0.2, no label smoothing
- Adam, no weight decay, constant learning rate without warm-up, no gradient clipping
- Mixed precision (bfloat16), moving average of the weights with a decay of 0.9998
- Data augmentation with a probability of 0.4: left-right flip (0.5), and with a probability of 0.3 each brightness, contrast and saturation, CLAHE, sharpening, luminance histogram equalization, histogram equalization of the BGR channels, noise, grayscale and the swap of the blue and red channels, the same for all the frames of a sequence
- Every augmentation type added after the start of the experiment kept off, the haze included (on by default since 2026-10-20), so that the launches from run_11 on, restarted on 2026-10-29, train with the augmentations of run_1 to run_10
- `top_k`: 5 (now fixed in the code, no longer a setting), no per-class report, no saved predictions, 8 sample sequences

## Starting Values

| Parameter | Value |
|---|---|
| `learning_rate` | 1e-3 |
| Direction | unidirectional (`causal: True`, `bidirectional: False`) |
| `TemporalModel.type` | lstm |
| `TemporalModel.number_layers` | 1 |
| `global_pooling` | flatten |
| `feature_map_normalization` | batch |
| `class_weighting` | balanced |

## Stages

| Stage | Parameter | Values | Iterations |
|---|---|---|---|
| 1 | `learning_rate` | 1e-3, 1e-4, 2e-4, 2e-3 | 2000 |
| 2 | `class_weighting` | balanced, none | 6000 |
| 3 | Direction | unidirectional, bidirectional | 6000 |
| 4 | `TemporalModel.type` | lstm, gru, transformer, temporal_convolution | 6000 |
| 5 | `TemporalModel.number_layers` | 1, 2 | 6000 |
| 6 | `global_pooling` | flatten, average, max | 6000 |
| 7 | `feature_map_normalization` | batch, sequence | 6000 |

The direction is written as the flags of the temporal module: unidirectional is `causal: True` and `bidirectional: False` for every type; bidirectional is `bidirectional: True` and `causal: False` for the GRU and the LSTM, and `causal: False` (attention over every frame, centred convolutions) for the transformer and the temporal convolution.

## Launches

Every launch gives its test Macro F1 at every evaluation (every 250 iterations up to the last one of its stage) for its trained and averaged weights, all gathered in `hyperparameter_experiments/sequence_labelling/results.csv` by `uv run python -m scripts.evaluation.collect_experiment_results` (`task = "sequence_labelling"`), launches and mean rows as in the tables below. The mean row of a setting averages its two launches, and the setting with the highest mean of the averaged weights is kept.

### Stage 1: Learning Rate (8 launches, 2000 iterations)

| `learning_rate` | Launch | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 1e-4 | run_3 | 0.0032 | 0.0034 | 0.0066 | 0.0089 | 0.0103 | 0.0114 | 0.0123 | 0.0132 | 0.0087 |
| 1e-4 | run_4 | 0.0042 | 0.0050 | 0.0069 | 0.0092 | 0.0102 | 0.0107 | 0.0103 | 0.0118 | 0.0086 |
| 1e-4 | **Mean** | 0.0037 | 0.0042 | 0.0067 | 0.0091 | 0.0102 | 0.0111 | 0.0113 | 0.0125 | 0.0086 |
| 2e-4 | run_5 | 0.0024 | 0.0091 | 0.0108 | 0.0116 | 0.0112 | 0.0117 | 0.0185 | 0.0193 | 0.0118 |
| 2e-4 | run_6 | 0.0026 | 0.0028 | 0.0093 | 0.0102 | 0.0145 | 0.0131 | 0.0146 | 0.0174 | 0.0106 |
| 2e-4 | **Mean** | 0.0025 | 0.0059 | 0.0101 | 0.0109 | 0.0128 | 0.0124 | 0.0166 | 0.0183 | 0.0112 |
| 1e-3 | run_1 | 0.0012 | 0.0074 | 0.0069 | 0.0138 | 0.0173 | 0.0207 | 0.0226 | 0.0274 | 0.0147 |
| 1e-3 | run_2 | 0.0073 | 0.0071 | 0.0070 | 0.0080 | 0.0138 | 0.0174 | 0.0183 | 0.0215 | 0.0125 |
| 1e-3 | **Mean** | 0.0042 | 0.0073 | 0.0069 | 0.0109 | 0.0155 | 0.0191 | 0.0205 | 0.0244 | **0.0136** |
| 2e-3 | run_7 | 0.0022 | 0.0063 | 0.0133 | 0.0177 | 0.0202 | 0.0242 | 0.0250 | 0.0294 | 0.0173 |
| 2e-3 | run_8 | 0.0006 | 0.0069 | 0.0067 | 0.0068 | 0.0068 | 0.0070 | 0.0068 | 0.0068 | 0.0061 |
| 2e-3 | **Mean** | 0.0014 | 0.0066 | 0.0100 | 0.0122 | 0.0135 | 0.0156 | 0.0159 | 0.0181 | 0.0117 |

Kept: **1e-3** (run_1 and run_2), the highest mean (0.0136 against 0.0117 for 2e-3, 0.0112 for 2e-4 and 0.0086 for 1e-4); the configuration files of the next stages are copies of `run-1.yaml` with the changed parameter and 6000 iterations.

- The higher the learning rate, the faster the launches learn up to 1e-3: at iteration 2000 the means are 0.0125 (1e-4), 0.0183 (2e-4) and 0.0244 (1e-3).
- 2e-3 gives the best single launch (run_7, 0.0294 at iteration 2000), but run_8 stays at about 0.0068 from iteration 500 to 2000, the score of a model that only learns that the first frame of a sequence is `no_change`: 2e-3 lies at the edge of the usable learning rates.
- Every launch that learns is still rising at iteration 2000, so the following stages train 6000 iterations.

### Stage 2: Class Weighting (4 launches, 6000 iterations)

The columns give every second evaluation, the mean is taken over all 24 evaluations (250 to 6000).

| `class_weighting` | Launch | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 | 5500 | 6000 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| balanced | run_9 | 0.0068 | 0.0164 | 0.0221 | 0.0262 | 0.0341 | 0.0536 | 0.0624 | 0.0629 | 0.0714 | 0.0745 | 0.0760 | 0.0686 | 0.0465 |
| balanced | run_10 | 0.0069 | 0.0154 | 0.0194 | 0.0302 | 0.0430 | 0.0657 | 0.0885 | 0.1131 | 0.1374 | 0.1593 | 0.1712 | 0.1727 | 0.0824 |
| balanced | **Mean** | 0.0068 | 0.0159 | 0.0208 | 0.0282 | 0.0386 | 0.0596 | 0.0754 | 0.0880 | 0.1044 | 0.1169 | 0.1236 | 0.1207 | 0.0645 |
| none | run_11 | 0.0090 | 0.0090 | 0.0090 | 0.0155 | 0.0168 | 0.0189 | 0.0200 | 0.0189 | 0.0208 | 0.0254 | 0.0298 | 0.0375 | 0.0187 |
| none | run_12 | 0.0090 | 0.0090 | 0.0090 | 0.0090 | 0.0090 | 0.0090 | 0.0090 | 0.0090 | 0.0090 | 0.0090 | 0.0090 | 0.0090 | 0.0090 |
| none | **Mean** | 0.0090 | 0.0090 | 0.0090 | 0.0123 | 0.0129 | 0.0140 | 0.0145 | 0.0140 | 0.0149 | 0.0172 | 0.0194 | 0.0233 | 0.0139 |

Kept: **balanced** (run_9 and run_10), the highest mean (0.0645 against 0.0139 for none); the direction stage trains bidirectional (run_13 and run_14) against the unidirectional run_9 and run_10.

- Without class weighting the launches barely learn a change: run_12 stays at 0.0090 from the first to the last evaluation, the score of a model that only predicts `no_change`, and run_11 only starts rising after iteration 1500 (0.0375 at iteration 6000).
- The two balanced launches part after iteration 3000 (0.0686 and 0.1727 at iteration 6000), run_10 still rising at iteration 6000 while run_9 levels off at about 0.07.
