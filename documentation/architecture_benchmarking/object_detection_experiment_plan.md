# Object Detection Experiment Plan

- **Configurations:** `vision_training/configurations/object_detection/run-<n>.yaml`, one per launch, launched with `uv run python -m vision_training.main --configuration_path=vision_training/configurations/object_detection/run-<n>.yaml`; the files after `run-1.yaml` are copies of it with the settings kept by the previous stages and the changed parameter, written by the driver `hyperparameter_experiments/object_detection/run_benchmark_stages.py`
- **Dataset:** synthetic hard shapes (`create_synthetic_dataset(task="object_detection", hard_objects=True)`), `/home/robert_kelevra/Data/object_detection_data/synthetic_hard_shapes`: 512,000 train / 50,000 validation / 50,000 test images of 1024 pixels, 5 to 12 shapes of the 35 shape classes per image (8.5 on average), from 16 pixels to 80% of the image side with log-uniform sizes (as many objects per scale octave), drawn from the largest to the smallest so that small shapes lie over large ones, up to 30% of a shape outside the image (box clipped, 19% of the boxes touching the border), on a flat background (20% of its shapes in a colour 20 to 40 levels from the background) or, for half of the images, a textured one (gradient, smooth noise or clutter lines changing the background by up to 40 levels). Below about 64 pixels the round polygons (heptagon, octagon, nonagon) are hard to tell from each other and from the circle
- **Launch folders:** `hyperparameter_experiments/object_detection/run_<n>` at the repository root, set as `ExperimentConfiguration.experiment_folder` before every launch

## Strategy

One parameter changes at a time. The best value of a stage is kept for all the following stages, and its launches serve as the reference of the next stage, reused when an earlier stage trained launches with the same settings for as many iterations (or more, under the constant learning rate schedule).

Every setting is launched twice, A and B, to smooth out the noise of a single training. Two launches train at the same time, the next launch starting as soon as one ends, the kernels of the two launches sharing the GPU through the MPS daemon. The training sets no random seed, so the two launches start from different weights, data order and augmentations.

Every launch keeps a moving average of its weights, and the settings are compared on the averaged weights (`Tensorboard/Test_EMA`): the score of a setting is the test mean average precision over the intersection over union thresholds 0.5 to 0.95 (`mAP@[0.5:0.95]/area`, the area under the precision recall curves) at the last iteration of the stage, averaged over its two launches. The mean average precision by object size (small, medium, large) and the proposal recall by detection level are read next to it.

## Fixed Settings

- `batch_size`: 64 images of 1024 pixels (two launches side by side reach about 63 Gigabytes above the idle machine, of its 121; one launch at 128 images would take about 49 alone)
- `number_iterations`: 10,000 (640,000 images, 1.25 epochs of the 512,000 training images), lowered from 20,000 by the user on 2026-11-02 at iteration 2500 of run_1 and run_2 (a pair at 20,000 took about 26 hours), the constant learning rate making their launches the same as launches planned for 10,000
- `weight_saving_iterations`: 500, the test evaluation running every 500 iterations (20 tested models) on `test_10000`, the first 10,000 of the 50,000 test images (85,143 objects), its images read from the `test` split through a link; `intermediate_weight_saving_iterations`: 250
- `information_dump`: 1000, `validation_iteration_ratio`: 0 (no validation during the training)
- Backbone `{"1": [1, 16], "2": [2, 16], "3": [2, 16], "4": [3, 32], "5": [3, 32], "6": [5, 32], "7": [5, 32]}`, residual blocks with squeeze and excite, input normalization, batch normalization
- Detection levels 4 to 7 (strides 16 to 128) with anchors of 16 and 32, 48 and 96, 128 and 256, 384 and 768 pixels, aspect ratios 0.5, 1 and 2, so that every size of the dataset has an anchor; region proposal heads `[2, 32]`, detection heads with a 7 x 7 RoIAlign grid, `[2, 32]` convolutions and `[64, 32]` fully connected layers, each level with its own heads (no feature pyramid network)
- Region proposals: 1000 before and 500 after the non-maximum suppression in training, 500 and 250 in inference, foreground intersection over union 0.5, background below 0.3, 128 positive of 256 sampled anchors
- Adam, no weight decay, constant learning rate without warm-up, no gradient clipping, moving average of the weights with a decay of 0.9998
- Data augmentation with a probability of 0.4: the haze (0.3), every other type off

## Starting Values

| Parameter | Value |
|---|---|
| `learning_rate` | 1e-3 |
| Feature pyramid network | none |

## Stages

| Stage | Parameter | Values | Iterations |
|---|---|---|---|
| 1 | `learning_rate` | 1e-3, 1e-4, 2e-4, 2e-3 | 10,000 |
| 2 | Feature pyramid network | none, feature pyramid network (to implement) | 10,000 |

## Launches

Every launch gives its test mean average precision at every evaluation (every 500 iterations up to 10,000) for its trained and averaged weights, all gathered in `hyperparameter_experiments/object_detection/results.csv` by `uv run python -m scripts.evaluation.collect_experiment_results` (`task = "object_detection"`), launches and mean rows as in the tables below.

### Stage 1: Learning Rate (8 launches, 10,000 iterations)

| Value | Launches |
|---|---|
| 1e-3 | run_1, run_2 |
| 1e-4 | run_3, run_4 |
| 2e-4 | run_5, run_6 |
| 2e-3 | run_7, run_8 |

A pair trains in about 13 hours (2500 iterations of run_1 and run_2 in about 3 hours, test evaluations included), the stage in about 2 days.
