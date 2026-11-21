# Shape Pretrainers

## Table Of Content
- [Description](#description)
- [Prerequisites](#prerequisites)
- [Pretrainers](#pretrainers)
- [Configurations Each One Fits](#configurations-each-one-fits)
- [Using A Pretrainer](#using-a-pretrainer)
- [Training One Again](#training-one-again)

## Description
Whenever a tutorial run has to be trained again, its backbone can start from one of these folders instead of random
weights, so that it starts from features of shapes and converges sooner. Every folder holds a backbone trained on the multi-label
classification of the 35 counted synthetic shapes (`vision_training/configurations/classification/configuration_synthetic_counted_shapes.yaml`,
256 pixels), one folder per architecture, named after the backbone part of the run names (`SE-RES-BN-B<convolutions>x<channels>…`
for squeeze and excitation, residual shortcuts, batch normalization and the blocks). A folder keeps only what the
transfer reads, `training_configuration.yaml` and the weights of the last checkpoint with its registry
(`Weights/full-checkpoint`, naming only the kept checkpoints), and `Results.txt`, the test scores of its evaluations;
`SE-RES-BN-B1x16.2x16.2x16.4x32.5x32.5x32` also keeps its checkpoint of iteration 10000.

## Prerequisites
Before loading one, the run needs the environment of the repository (`uv sync` at its root); nothing else, the synthetic
shapes data set being needed only to train a pretrainer again.

## Pretrainers
To pick a folder, match the blocks of the run with the folder name; the scores are those of the last evaluation in
`Results.txt` (50,000 test images, threshold 0.5 for the F1), the start telling whether the backbone was trained from
random weights or from the backbone of another run.

| Folder | Blocks | Start | Iterations × images | Test macro F1 | Test macro ROC AUC | Kept weights |
|---|---|---|---|---|---|---|
| `SE-RES-BN-B1x16.2x32.2x64` | 3 | random weights | 7,500 × 64 | 17.58 % | 74.75 % | `Weights/Iteration_7500/model_007500.pt` |
| `SE-RES-BN-B1x16.2x32.2x64.2x64.2x64` | 5 | random weights | 30,000 × 64 | 88.77 % | 97.89 % | `Weights/Iteration_30000/model_030000.pt` |
| `SE-RES-BN-B1x16.2x16.2x16.4x32.5x32.5x32` | 6 | random weights | 15,000 × 256 | 71.95 % | 93.56 % | `Weights/Iteration_15000/model_015000.pt`, `Weights/Iteration_10000/model_010000.pt` |
| `SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128` | 6 | backbone of the classification tutorial run | 25,000 × 64 | 89.19 % | 97.83 % | `Weights/Iteration_25000/model_025000.pt` |
| `SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x64` | 6 | random weights | 30,000 × 64 | 88.78 % | 97.87 % | `Weights/Iteration_30000/model_030000.pt` |
| `SE-RES-BN-B1x32.2x32.2x32.16x32.20x32.20x32` | 6 | Small ImageNet classifier | 20,000 × 64 | 64.77 % | 91.56 % | `Weights/Iteration_20000/model_020000.pt` |
| `SE-RES-BN-B1x64.2x64.2x64.4x64.5x64.5x64` | 6 | Classic ImageNet classifier | 20,000 × 64 | 84.80 % | 96.69 % | `Weights/Iteration_20000/model_020000.pt` |
| `SE-RES-BN-B1x16.2x16.2x16.3x32.3x32.5x32.5x32` | 7 | random weights | 40,000 × 64 | 67.32 % | 91.18 % | `Weights/Iteration_40000/model_040000.pt` |

Two warm starts of 1,000 iterations were also trained and removed, the pretrainer of the same blocks from random
weights scoring higher: the 7 blocks from the backbone of detection benchmark run_21 (19.72 % test macro F1)
and the 3 blocks from the backbone of the keypoint tutorial run (2.30 %). `SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128`
scored its best at 22,500 iterations (90.08 % test macro F1), 0.89 points above the kept last checkpoint.

`SE-RES-BN-B1x16.2x16.2x16.4x32.5x32.5x32`, the counted shapes classifier, trained 10,000 iterations at a constant
learning rate, then 5,000 more under a cosine schedule over the whole run, the rate falling from 26 % of its value to
1 %. It keeps the checkpoint of 10,000 (71.82 % test macro F1, 93.37 % ROC AUC), which
`tutorials/run_configurations/sequence_labelling_pretrained_tutorial.yaml` loads, next to that of 15,000. Each
checkpoint file also holds the averaged weights of the training (decay 0.999), which a run keeping a moving average
loads instead of the trained ones: 71.40 % test macro F1 and 93.36 % ROC AUC at 10,000, 71.95 % and 93.55 % at 15,000.
The other pretrainers trained without one.

Whatever their score, the pretrainers with 32 channels at most still miss most triangles and quadrilaterals at their
last evaluation, the main part of their lower macro F1: 10 of the 35 classes stay under 5 % test recall for
`SE-RES-BN-B1x16.2x16.2x16.3x32.3x32.5x32.5x32` (13 at 27,500 iterations) and
`SE-RES-BN-B1x16.2x16.2x16.4x32.5x32.5x32` (triangle, parallelogram, trapezoid, kite, rhombus and their hollow forms),
9 for `SE-RES-BN-B1x32.2x32.2x32.16x32.20x32.20x32`. The 3 blocks miss 25 classes, while the 5 and 6 blocks ending with
64 channels or more keep every class above 5 %.

## Configurations Each One Fits
If you are looking for the pretrainer of a configuration of the repository, this list gives every configuration whose
backbone the transfer code accepts from each folder (`compare_backbone_architectures`): the same blocks, or the first
blocks of the pretrainer.

- `SE-RES-BN-B1x16.2x32.2x64`: `vision_training/configurations/keypoint_detection/configuration_synthetic_shapes.yaml`.
- `SE-RES-BN-B1x16.2x32.2x64.2x64.2x64`: `vision_training/configurations/keypoint_detection/configuration_synthetic_shapes.yaml`, through its first 3 blocks.
- `SE-RES-BN-B1x16.2x16.2x16.4x32.5x32.5x32`: `vision_training/configurations/classification/configuration_synthetic_counted_shapes.yaml`, `vision_training/configurations/self_supervised/configuration_counted_shapes_barlow_twins.yaml`, `vision_training/configurations/self_supervised/configuration_counted_shapes_simclr.yaml`, `tutorials/run_configurations/sequence_labelling_pretrained_tutorial.yaml`.
- `SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128`: `vision_training/configurations/classification/configuration_synthetic_shapes.yaml`, `vision_training/configurations/frame_selection/configuration_synthetic_sequences.yaml`, `vision_training/configurations/self_supervised/configuration_synthetic_shapes.yaml`, `vision_training/configurations/semantic_segmentation/configuration_synthetic_shapes.yaml`, `vision_training/configurations/sequence_classification/configuration_synthetic_sequences.yaml`, `vision_training/configurations/sequence_comparison/configuration_synthetic_pairs.yaml`, `vision_training/configurations/sequence_labelling/configuration_synthetic_sequences.yaml`, `tutorials/run_configurations/self_supervised_synthetic_shapes_tutorial.yaml`.
- `SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x64`: `vision_training/configurations/frame_selection/configuration_synthetic_sequences.yaml`, `vision_training/configurations/sequence_classification/configuration_synthetic_sequences.yaml`, `vision_training/configurations/sequence_comparison/configuration_synthetic_pairs.yaml`, `vision_training/configurations/sequence_labelling/configuration_synthetic_sequences.yaml`.
- `SE-RES-BN-B1x32.2x32.2x32.16x32.20x32.20x32`: none, the Small ImageNet configuration having 64 filters in blocks 4 to 6.
- `SE-RES-BN-B1x64.2x64.2x64.4x64.5x64.5x64`: `vision_training/configurations/classification/configuration_example.yaml`, `vision_training/configurations/keypoint_detection/configuration_example.yaml`, `vision_training/configurations/self_supervised/configuration_example.yaml`.
- `SE-RES-BN-B1x16.2x16.2x16.3x32.3x32.5x32.5x32`: `vision_training/configurations/instance_segmentation/configuration_synthetic_shapes.yaml`, `vision_training/configurations/object_detection/configuration_synthetic_shapes.yaml`, `tutorials/run_configurations/object_detection_synthetic_shapes_tutorial.yaml`.

For the keypoint configuration, both pretrainers give stems of the same quality, the higher score of the 5 blocks
(88.77 % against 17.58 %) not carrying over to the 3 blocks the stem takes: a linear probe of their 3 blocks (a logistic
regression per class on the average and maximum of every channel of the stem output, 3,000 training and 2,000 test
images of the counted shapes at 256 pixels) scores 77.02 % macro ROC AUC from `SE-RES-BN-B1x16.2x32.2x64` and 76.76 %
from `SE-RES-BN-B1x16.2x32.2x64.2x64.2x64`, against 53.84 % from random weights.

## Using A Pretrainer
Suppose a configuration with the blocks of a folder: point its `Backbone.pretrained` section at the folder and leave
the iteration null for the last checkpoint, the one the registry names first; only
`SE-RES-BN-B1x16.2x16.2x16.4x32.5x32.5x32` keeps a second one, iteration 10000.

```yaml
Backbone:
  pretrained:
    experiment_folder: "/home/robert_kelevra/Data/tutorial_runs/pretrained/SE-RES-BN-B1x16.2x32.2x64"
    iteration: null
```

The training then loads the backbone before its first iteration and announces it; another image size than 256 pixels
is accepted and reported, and a run resuming from its own checkpoints does not load it again.

## Training One Again
Should a pretrainer be trained again, `tutorials/run_configurations/pretraining/<folder name>.yaml` of the repository
holds its launch file, whose header says how it was trained. `SE-RES-BN-B1x16.2x16.2x16.4x32.5x32.5x32` trained its
first 10,000 iterations with `vision_training/configurations/classification/configuration_synthetic_counted_shapes.yaml`
as shipped (256 images, constant learning rate, averaged weights), its launch file only resuming it to 15,000: launched
from random weights, that file trains the 15,000 iterations under the cosine schedule. The training writes its run
under the full run name, which is renamed to the folder name once the unused files are removed.

```bash
.venv/bin/python -m vision_training.main --configuration_path=tutorials/run_configurations/pretraining/SE-RES-BN-B1x16.2x32.2x64.yaml
```
