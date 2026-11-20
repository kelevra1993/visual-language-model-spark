# Keypoint Detection Experiment Plan

- **Configurations:** `vision_training/configurations/keypoint_detection/run-<n>.yaml`, one per launch, `run-1.yaml` written before the first launch from `configuration_synthetic_shapes.yaml` and the architecture agreed with the user
- **Dataset:** synthetic keypoint shapes, `/home/robert_kelevra/Data/keypoint_data/synthetic_shapes` (512 x 512 images, 250,000 train, 50,000 validation, 50,000 test, 2 to 5 objects per image, 3.5 on average, 37 shape types with 0 to 18 keypoints each, 33 of them in the label mapping)
- **Launch folders:** `hyperparameter_experiments/keypoint_detection/run_<n>` at the repository root

## Table Of Content
- [Strategy](#strategy)
- [Fixed Settings](#fixed-settings)
- [Before Stage 0](#before-stage-0)
  - [The Architecture](#the-architecture)
  - [The Test Split](#the-test-split)
- [Starting Values](#starting-values)
- [Stages](#stages)
- [Not Planned](#not-planned)

## Strategy

So that the stages read like those of the detection and instance segmentation benchmarks, the method is the same:

- one parameter changes at a time, and the best value of a stage is kept for all the following stages;
- every value is trained twice, A and B, two launches side by side sharing the GPU through the MPS daemon, without random seed;
- the score of a value is the test OKS mean average precision over the thresholds 0.5 to 0.95 of the averaged weights (`Test_EMA`, OKS `mAP@[0.5:0.95]`, area method) at the last iteration, averaged over its two launches; the mAP by object size, the mean keypoint error and the PCK@0.05 explain the gaps;
- a challenger whose pair mean differs from the best value so far by less than the larger gap between the two launches of either value is a tie, which keeps the earlier value, except in the stages that only make the model cheaper (stages 4 and 5), where a tie keeps the cheaper value;
- a stage that needs code has its code merged right before its pair and reverted when it is not better, unless the user keeps it;
- no stage starts before the user says so.

## Fixed Settings

For every launch to see the same data and be judged the same way, these settings stay fixed (the user's choices of 2026-11-20):

- `number_iterations` 10,000 with `batch_size` 64 images of 512 pixels (640,000 images, 2.6 passes over the 250,000 training images), the cosine decay over the 9,500 iterations after a warm-up of 500; stage 0 measures whether 64 images fit in memory next to a second launch;
- `HRNet.base_width` 32, the width of HRNet-W32, not tested;
- one evaluation every 1000 iterations (`weight_saving_iterations` 1000), 10 per launch, on the test split as fixed below;
- Adam without weight decay, a moving average of the weights with a decay of 0.999;
- the augmentation of `configuration_synthetic_shapes.yaml` (probability 0.4: crop, zoom-out, translation, rotation of at most 5 degrees and the photometric types), one loader worker per loader;
- both flips off: a flip moves the keypoints but never renames them (`augmentations.py`), while the dataset names them on the upright shape (a square's `top_left`, the stick figure's `left_elbow`), so a flipped image would teach wrong names.

## Before Stage 0

Two choices are made before the first launch, the first with the user.

### The Architecture

Before any stage, the architecture is defined together with the user (2026-11-20): the backbone stem, the HRNet stages and the keypoint head, then written into `run-1.yaml`. The HRNet is set by three numbers:

- `base_width` C (32, fixed): the channels of the stride 4 branch, each lower branch having twice those of the branch above (32, 64, 128, 256);
- `modules_per_stage`: one number per stage, each stage adding one lower-resolution branch, so [1, 2, 2] gives 3 stages and the branches at strides 4 and 8 (stage 1), 4 to 16 (stage 2), 4 to 32 (stage 3); each number counts the exchange modules of its stage, where every branch runs its convolutions then receives the features of all the other branches, resized to its own resolution. [1, 4, 3], the counts of HRNet-W32, keeps the same branches with 8 modules instead of 5, so more convolutions and more exchanges;
- `blocks_per_branch`: the pairs of 3 x 3 convolutions every branch runs in every module, 2 in the shipped configuration and 4 in HRNet-W32.

The head reads the stride 4 branch only, its `hidden_width` 3 x 3 convolution (256 in the shipped configuration) giving one heatmap and one tag map per keypoint of every class (243 of each).

### The Test Split

So that an evaluation stays short without changing its score, the test split is reduced in two steps, keeping one folder named `test` (no `test_10000` or similar):

- the `test` folder keeps its first 20,000 images, its other 30,000 deleted, before stage 0;
- the checkpoints of stage 0 are evaluated again on the first 10,000 of those images (from a temporary folder, removed afterwards); if the scores of the 10,000 and the 20,000 images differ by less than the gap between the two launches of stage 0 at every evaluation, the `test` folder keeps its first 10,000 images, the others deleted, and every later stage uses those.

Before any deletion, the tutorials that read this `test` folder are checked, as the deleted images cannot come back without generating the dataset again: the keypoint figures (`tutorials/figures/keypoint_detection/snapshot_files.yaml`) read `test/data/test_000409.png`, which stays, and `test/labels.json`, whose snapshot then needs refreshing; the tutorial run evaluates on its own `test_first_1024` folder, left as it is.

## Starting Values

Before stage 0 is launched, `run-1.yaml` takes the values of `configuration_synthetic_shapes.yaml`, those of the tutorial run (87.59 % at 6,000 on 1,024 test images, large objects 81.28 % against 93.83 % for the medium ones), with the fixed settings above and the agreed architecture:

| Parameter | Value |
|---|---|
| `learning_rate` | 1e-3 |
| `HRNet` | `base_width` 32, the stages and blocks agreed with the user |
| `KeypointHead.higher_resolution` | True (heatmaps at stride 2 averaged with those at stride 4) |
| `KeypointHead.hidden_width` | 256, unless the agreed architecture sets it |
| `KeypointHead.heatmap_sigma` | 2 cells |
| `KeypointHead.tag_pull_weight`, `tag_push_weight` | 0.1, 0.1 |
| `KeypointHead.inference` | `keypoint_threshold` 0.1, `tag_threshold` 1.0, 30 candidates per channel |

## Stages

So that the order and the reason of every stage are seen at once, the table lists them; each challenger is one pair of launches, the reference pair being that of the value kept so far:

| Stage | Short name | What changes, and why it may help |
|---|---|---|
| 0 | Reference and preflight | The agreed architecture with the starting values, one pair. Gives the score to beat, the gap between two launches of the same settings (the noise every later gap is judged against), the time of a pair and its memory at a batch of 64. |
| 0a | Test split size (evaluation only) | The checkpoints of stage 0 on the first 10,000 test images against the 20,000. Decides whether the test split is cut to 10,000 (see The Test Split). |
| 0b | Grouping thresholds (evaluation only) | `keypoint_threshold` 0.05, 0.1, 0.2 and `tag_threshold` 0.5, 1.0, 2.0 on the checkpoints of stage 0, no training. They decide which heatmap peaks become keypoints and which keypoints join the same object; a bad pair of thresholds would hide what later trainings gain. |
| 1 | Learning rate | 1e-3 against 1e-4, 5e-4 and 2e-3. The first stage of every benchmark: with 10,000 iterations, a higher rate may converge sooner, while a lower one may train the tags more steadily; 1e-4 covers the case where the batch of 64 and the width of 32 train better slowly. |
| 2 | Heatmap sigma | 2 cells against 1.5 and 3. A narrower Gaussian marks the keypoint more sharply, which may lower the keypoint error (4.5 pixels in the tutorial run) and raise the mAP@0.95 (72.89 %); a wider one gives more positive cells per keypoint and an easier target, which may help the large objects. |
| 3 | Tag loss weights | pull and push 0.1 against 0.5. The large objects score the lowest; their keypoints lie far apart, so their grouping relies most on the tags, which these weights train. A stronger tag loss may group them better, at the risk of a weaker heatmap loss. |
| 4 | Higher-resolution branch | on against off. The HigherHRNet branch helps the small objects in its paper, but they already score 88.86 % here; turning it off saves its upsampling and refining blocks. A tie keeps it off. |
| 5 | Head hidden width | 256 against 128 channels before the 1 x 1 heatmap and tag convolution, to be confirmed once the architecture is agreed. 256 covers the 243 heatmaps comfortably; 128 is cheaper and may suffice. A tie keeps 128. |

The time of a pair is measured by stage 0: the tutorial run trained about 8 images per second alone with a width of 16, so the 640,000 images of a launch at a width of 32 may take a day or more, which stage 0 tells before the other stages are planned in time; stages 0a and 0b only evaluate saved checkpoints.

## Not Planned

In case a later benchmark wants them, these were left out:

- The HRNet width: fixed at 32 by the user.
- The HRNet depth: part of the architecture agreed before stage 0.
- The left-right and up-down flips: the code moves keypoints without renaming them (Fixed Settings), and renaming them would need the left and right keypoints of every shape, which neither the code nor the dataset describes.
- A larger image size: it would change the images seen and the time of a pair together.
- A top-down model (detector then one keypoint head per box): another architecture, not a hyperparameter of this one.
- Sub-pixel decoding of the heatmap peaks: already in the code, every peak moved within its cell by a parabola fitted to the log-heatmap at the peak and its two neighbours on each axis (`compute_peak_offsets` of `vision_training/models/keypoints/keypoint_detector.py`).
