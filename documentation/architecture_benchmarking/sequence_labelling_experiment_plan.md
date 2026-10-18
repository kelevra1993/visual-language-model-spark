# Sequence Labelling Experiment Plan

- **Configurations:** `vision_training/configurations/sequence_labelling/run-<n>.yaml`, one per launch (`run-1.yaml` to `run-8.yaml` for stage 1), launched with `uv run python -m vision_training.main --configuration_path=vision_training/configurations/sequence_labelling/run-<n>.yaml`; the files of the later stages are copies of `run-1.yaml` with the settings kept by the previous stages and the changed parameter
- **Dataset:** synthetic changing shapes (`create_synthetic_dataset(task="sequence_labelling")`), frames of 5 to 10 shapes drawn again at every frame, every frame from the second one adding a shape, removing one or keeping them; 71 labels (`appearing_<class>` and `disappearing_<class>` for the 35 shape classes, and `no_change`, always the label of the first frame); 250,000 train / 50,000 validation / 50,000 test sequences of 2 to 12 frames of 512 pixels, resized to 256 pixels
- **Launch folders:** `hyperparameter_experiments/sequence_labelling/run_<n>` at the repository root, set as `ExperimentConfiguration.experiment_folder` before every launch

## Strategy

One parameter changes at a time. The best value of a stage is kept for all the following stages, and its launches serve as the reference of the next stage, so they are not trained again.

Every setting is launched twice, A and B, to smooth out the noise of a single training, the two launches of a setting training at the same time on the GPU. The training sets no random seed, so the two launches start from different weights, data order and augmentations.

Every launch keeps a moving average of its weights, and the settings are compared on the averaged weights: the mean of the test Macro F1 of the averaged weights at iterations 250 to 2000 (`Tensorboard/Test_EMA`), averaged over the two launches. The Macro F1 weighs the frames of every change as much as the far more frequent frames without change, whose share makes the frame accuracy high for a model that never finds a change.

## Fixed Settings

- `batch_size`: 64 sequences
- `number_iterations`: 2000 (128,000 sequences, about half an epoch of the 250,000 training sequences)
- `weight_saving_iterations`: 250, the test evaluation running every 250 iterations from 250 to 2000 (8 tested models) on `test_10000`, a fixed subset of 10,000 of the 50,000 test sequences (69,918 frames, every label present) drawn with the seed 0 by `scripts/data/create_sequence_split_subset.py`, the evaluation of the whole test split taking 26 minutes
- `intermediate_weight_saving_iterations`: 125, 8 more checkpoints saved without a test evaluation
- `number_workers`: 2 per loader, `prefetch_batches`: 2, the expandable segments of the CUDA allocator turned on by the training launch (two launches side by side stay at about 55 Gigabytes of the 121 of the machine)
- `information_dump`: 250, `validation_iteration_ratio`: 0 (no validation during the training)
- `maximum_sequence_length`: 50, every sequence fed whole, the sequences of a batch padded to the longest one
- Backbone `{"1": [1, 16], "2": [2, 16], "3": [2, 16], "4": [4, 32], "5": [5, 32], "6": [5, 32]}`, residual blocks with squeeze and excite, input normalization
- Temporal module: hidden size 96, dropout 0, positional embedding (transformer), 4 heads (transformer), kernel size 3 (temporal convolution)
- Classification head `[96, 96]` with layer normalization, dropout 0.2, no label smoothing
- Adam, no weight decay, constant learning rate without warm-up, no gradient clipping
- Mixed precision (bfloat16), moving average of the weights with a decay of 0.9998
- Data augmentation with a probability of 0.4: left-right flip (0.5), and with a probability of 0.3 each brightness, contrast and saturation, CLAHE, sharpening, luminance histogram equalization, histogram equalization of the BGR channels, noise, grayscale and the swap of the blue and red channels, the same for all the frames of a sequence
- `top_k`: 5, no per-class report, no saved predictions, 8 sample sequences

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

| Stage | Parameter | Values |
|---|---|---|
| 1 | `learning_rate` | 1e-3, 1e-4, 2e-4, 2e-3 |
| 2 | Direction | unidirectional, bidirectional |
| 3 | `TemporalModel.type` | lstm, gru, transformer, temporal_convolution |
| 4 | `TemporalModel.number_layers` | 1, 2 |
| 5 | `global_pooling` | flatten, average, max |
| 6 | `feature_map_normalization` | batch, sequence |
| 7 | `class_weighting` | balanced, none |

The direction is written as the flags of the temporal module: unidirectional is `causal: True` and `bidirectional: False` for every type; bidirectional is `bidirectional: True` and `causal: False` for the GRU and the LSTM, and `causal: False` (attention over every frame, centred convolutions) for the transformer and the temporal convolution.

## Launches

Every launch gives its test Macro F1 at iterations 250 to 2000 for its trained and averaged weights, all gathered in `hyperparameter_experiments/sequence_labelling/results.csv` by `uv run python -m scripts.evaluation.collect_experiment_results` (`task = "sequence_labelling"`), launches and mean rows as in the tables below. The mean row of a setting averages its two launches, and the setting with the highest mean of the averaged weights is kept.
