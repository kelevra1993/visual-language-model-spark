# Keypoint Detection Experiment Plan

- **Configurations:** `vision_training/configurations/keypoint_detection/run-<n>.yaml`, one per launch, `run-1.yaml` written before the first launch from `configuration_synthetic_shapes.yaml`
- **Dataset:** synthetic keypoint shapes, `/home/robert_kelevra/Data/keypoint_data/synthetic_shapes` (512 x 512 images, 250,000 train, 50,000 validation, 50,000 test, 2 to 5 objects per image, 3.5 on average, 37 shape types with 0 to 18 keypoints each, 33 of them in the label mapping)
- **Launch folders:** `hyperparameter_experiments/keypoint_detection/run_<n>` at the repository root

## Table Of Content
- [Strategy](#strategy)
- [Fixed Settings](#fixed-settings)
- [Starting Values](#starting-values)
- [Stages](#stages)
- [Not Planned](#not-planned)

## Strategy

So that the stages read like those of the detection and instance segmentation benchmarks, the method is the same:

- one parameter changes at a time, and the best value of a stage is kept for all the following stages;
- every value is trained twice, A and B, two launches side by side sharing the GPU through the MPS daemon, without random seed;
- the score of a value is the test OKS mean average precision over the thresholds 0.5 to 0.95 of the averaged weights (`Test_EMA`, OKS `mAP@[0.5:0.95]`, area method) at the last iteration, averaged over its two launches; the mAP by object size, the mean keypoint error and the PCK@0.05 explain the gaps;
- a challenger whose pair mean differs from the best value so far by less than the larger gap between the two launches of either value is a tie, which keeps the earlier value, except in the stages that only make the model cheaper (stages 6 and 7), where a tie keeps the cheaper value;
- a stage that needs code has its code merged right before its pair and reverted when it is not better, unless the user keeps it;
- no stage starts before the user says so.

## Fixed Settings

For every launch to see the same data and be judged the same way, these settings stay fixed (the user's choice of 2026-11-20):

- `number_iterations` 10,000 with `batch_size` 16 images of 512 pixels, the cosine decay over the 9,500 iterations after a warm-up of 500;
- one evaluation every 1000 iterations (`weight_saving_iterations` 1000), 10 per launch, on the first 5,000 test images (a `test_first_5000` folder made like `test_first_1024` before stage 0), the full 50,000 taking about an hour per evaluation;
- Adam without weight decay, a moving average of the weights with a decay of 0.999;
- the augmentation of `configuration_synthetic_shapes.yaml` (probability 0.4: crop, zoom-out, translation, rotation and the photometric types), one loader worker per loader;
- the backbone stem `[1, 16]`, `[2, 32]`, `[2, 64]` with squeeze and excitation, residual blocks and batch normalization.

## Starting Values

Before stage 0 is launched, `run-1.yaml` takes the values of `configuration_synthetic_shapes.yaml`, those of the tutorial run (87.59 % at 6,000 on 1,024 test images, large objects 81.28 % against 93.83 % for the medium ones):

| Parameter | Value |
|---|---|
| `learning_rate` | 1e-3 |
| `HRNet` | `base_width` 16, `modules_per_stage` [1, 2, 2] (branches at strides 4 to 32), `blocks_per_branch` 2 |
| `KeypointHead.higher_resolution` | True (heatmaps at stride 2 averaged with those at stride 4) |
| `KeypointHead.hidden_width` | 256 |
| `KeypointHead.heatmap_sigma` | 2 cells |
| `KeypointHead.tag_pull_weight`, `tag_push_weight` | 0.1, 0.1 |
| `KeypointHead.inference` | `keypoint_threshold` 0.1, `tag_threshold` 1.0, 30 candidates per channel |

## Stages

So that the order and the reason of every stage are seen at once, the table lists them; each challenger is one pair of launches, the reference pair being that of the value kept so far:

| Stage | Short name | What changes, and why it may help |
|---|---|---|
| 0 | Reference and preflight | The starting values, one pair. Gives the score to beat, the gap between two launches of the same settings (the noise every later gap is judged against), the time of a pair and its memory. |
| 0b | Grouping thresholds (evaluation only) | `keypoint_threshold` 0.05, 0.1, 0.2 and `tag_threshold` 0.5, 1.0, 2.0 on the checkpoints of stage 0, no training. They decide which heatmap peaks become keypoints and which keypoints join the same object; a bad pair of thresholds would hide what later trainings gain. |
| 1 | Learning rate | 1e-3 against 5e-4 and 2e-3. The first stage of every benchmark: with 10,000 iterations instead of the 20,000 of the shipped configuration, a higher rate may converge sooner, while a lower one may train the tags more steadily. |
| 2 | Heatmap sigma | 2 cells against 1.5 and 3. A narrower Gaussian marks the keypoint more sharply, which may lower the keypoint error (4.5 pixels in the tutorial run) and raise the mAP@0.95 (72.89 %); a wider one gives more positive cells per keypoint and an easier target, which may help the large objects. |
| 3 | Tag loss weights | pull and push 0.1 against 0.5. The large objects score the lowest; their keypoints lie far apart, so their grouping relies most on the tags, which these weights train. A stronger tag loss may group them better, at the risk of a weaker heatmap loss. |
| 4 | HRNet depth | `modules_per_stage` [1, 2, 2] against [1, 4, 3], the module counts of HRNet-W32. More exchanges between the branches carry more context from the coarse branches to the stride 4 branch that feeds the head, which the large objects may need. |
| 5 | HRNet width | `base_width` 16 against 32, the width of HRNet-W32. Twice the channels in every branch for more capacity, for about four times the computation; kept only if it gains beyond the launch gap. |
| 6 | Higher-resolution branch | on against off. The HigherHRNet branch helps the small objects in its paper, but they already score 88.86 % here; turning it off saves its upsampling and refining blocks. A tie keeps it off. |
| 7 | Head hidden width | 256 against 128 channels before the 1 x 1 heatmap and tag convolution. 256 covers the 243 heatmaps comfortably; 128 is cheaper and may suffice. A tie keeps 128. |

Every training pair is expected to take about 6 to 8 hours (the tutorial run trained about 8 images per second alone), which stage 0 measures; stage 0b only evaluates saved checkpoints.

## Not Planned

In case a later benchmark wants them, these were left out:

- The left-right flip: it would need the left and right keypoints of the symmetric shapes (the stick figure's elbows, hands, knees and feet) swapped with the image, code the dataset does not describe yet.
- A larger batch or image size: at a fixed 10,000 iterations both change the images seen and the time of a pair together, which would mix their effect with the time budget.
- A top-down model (detector then one keypoint head per box): another architecture, not a hyperparameter of this one.
- Sub-pixel decoding of the heatmap peaks: already in the code, every peak moved within its cell by a parabola fitted to the log-heatmap at the peak and its two neighbours on each axis (`compute_peak_offsets` of `vision_training/models/keypoints/keypoint_detector.py`).
