# Pretraining Reference

If one wishes to start a backbone from pre-trained weights, this reference covers the ImageNet-22k download and the pretraining plan. It completes the [Tasks](../../README.md#tasks) section of the README. A run loads the backbone through `Backbone.pretrained` ([Transfer Between Runs](../../README.md#transfer-between-runs)).

## Table Of Content
To find your way in this reference, the sections are:
- [ImageNet-22k](#imagenet-22k)
  - [Downloading ImageNet-22k](#downloading-imagenet-22k)
  - [Validation And Test Splits](#validation-and-test-splits)
- [Pretraining Strategy](#pretraining-strategy)
  - [Model Sizes](#model-sizes)
  - [Classification Pretraining On ImageNet-22k](#classification-pretraining-on-imagenet-22k)
  - [Object Detection](#object-detection)
    - [Open Images V7 Splits](#open-images-v7-splits)
    - [Detection Configurations](#detection-configurations)
  - [Instance Segmentation](#instance-segmentation)
  - [Order And Resources](#order-and-resources)
  - [Open Questions](#open-questions)

## ImageNet-22k
Should you need the pre-trained backbones of the classification or self-supervised configurations, download ImageNet-22k with `vision_training/data/classification/imagenet_download.py`. It reads `timm/imagenet-22k-wds` from Hugging Face (fall11 release, 21,841 classes, 4,608 shards) and writes the `labels.json` manifests and `label_mapping.json`.

### Downloading ImageNet-22k
Before the first shard is fetched, accept the terms of [timm/imagenet-22k-wds](https://huggingface.co/datasets/timm/imagenet-22k-wds) (non-commercial research and education only). Keep about 1.3 TB free on the disk of `--dataset_folder`, the only required argument; the download stops below 100 GB free (`--minimum_free_gigabytes`). It resumes after an interruption and fetches about 1,500 shards an hour with 16 workers on a DGX Spark (`--number_workers`). Without arguments, it prints its help.

```bash
# Log in with the account that accepted the terms
uv run hf auth login

# Download the shards into the folder the example classification configurations read
uv run python -m vision_training.data.classification.imagenet_download --dataset_folder ~/Data/classifier_data/imagenet-22k
```

The figure shows the download in progress.

<p align="center">
  <img src="../../README/imagenet_22k_download.png" width="100%" />
</p>

So that a configuration trains on the download, point `train_data_folder`, `validation_data_folder`, `test_data_folder` and `Data.label_mapping.mapping_file_path` to `manifests/train`, `manifests/validation`, `manifests/test` and `manifests/label_mapping.json` of `--dataset_folder`.

### Validation And Test Splits
In case you compare scores with other results, note that the 18,846 held-out images are both the validation and the test split. The scores do not compare with published ImageNet-1k accuracies. Split sizes are in the [tasks reference](tasks_reference.md#classification-cls).

## Pretraining Strategy
Before training the production models, this plan sets the order of the stages: three backbone sizes pretrained on classification, then fine-tuned for detection and instance segmentation.

### Model Sizes
If one wishes to pick the size of a production model, the table lists the three candidates, with blocks written as convolutions x filters. Each size doubles the filters, about 4 times the cost (1.3, 4.9 and 19.4 billion multiply-accumulates per 256 pixel image).

| Size | Backbone | Head | Parameters (backbone + head, 21,841 classes) | Configuration |
|---|---|---|---|---|
| S | 1x32 · 2x32 · 2x32 · 16x64 · 20x64 · 20x64 | [256, 128] | 2.3 M + 3.9 M | `configuration_small_model.yaml` |
| M | 1x64 · 2x64 · 2x64 · 16x128 · 20x128 · 20x128 | [512, 256] | 9.3 M + 9.9 M | `configuration_medium_model.yaml` |
| L | 1x128 · 2x128 · 2x128 · 16x256 · 20x256 · 20x256 | [512, 256] | 37.1 M + 14.1 M | `configuration_large_model.yaml` |

The files are in `vision_training/configurations/classification/`.

### Classification Pretraining On ImageNet-22k
For the three sizes to compare, each trains on ImageNet-22k at 256 pixels with the same recipe: learning rate 2e-4, the best of stage 1 of the [classification benchmark](../architecture_benchmarking/experiment_plan.md#stage-1-learning-rate-10-launches).

1. Start from random weights, or from an earlier backbone of the same design (`Backbone.pretrained`).
2. Keep the moving average of the weights.
3. Choose the checkpoint on test accuracy, with the same rule for every size.
4. Record accuracy, size and speed (`diagnosis_modes.benchmark_speed`).

### Object Detection
Whenever a size has its pretrained backbone, the detector loads it into its 7-block backbone, the 7th block starting random. It keeps the settings of the detection benchmark.

1. Convert Open Images V7 (1.74 million training images, 601 classes, boxes in `detections.csv`) into COCO `labels.json` ([Open Images V7 Splits](#open-images-v7-splits)).
2. Pretrain the detector of each size on Open Images ([Detection Configurations](#detection-configurations)).
3. Fine-tune it on COCO 2017 (80 classes) and compare its validation box mAP with published results. The backbone, neck and heads load from the Open Images run, the heads with `class_layers: False`.
4. Keep the Open Images detector as the start of projects whose classes are not in COCO.

#### Open Images V7 Splits
In case you rebuild the detection data, note that the splits were converted once from the FiftyOne download of `vision_training/data/data_retriever.py`, by a script not kept. They are in `~/Data/object_detection_data/open-images-v7/coco_format/`:
- `train/`: 1,872,515 images, 15,734,499 boxes, including group-of boxes (6 %) and rotated images (about 1 %).
- `validation_test_20000/`: 20,000 images, 116,880 boxes, drawn rarest classes first (585 classes present), without group-of box or rotation.
- Boxes are in pixels `[x, y, width, height]`; the 601 classes are numbered from 1 alphabetically, parents (Vehicle above Car) kept.
- Images without box and 27 images with an EXIF orientation are dropped.
- 29 % of validation boxes share their object with another class (Man, Person; Tire, Wheel), so detecting one leaves the others missed.

#### Detection Configurations
Should you launch the Open Images pretraining of a size, use the `configuration_<size>_model.yaml` files of `vision_training/configurations/object_detection/`. Each takes the classification backbone of its size with a block 7 equal to block 6, and a path aggregation neck of 32, 64 or 128 channels. They have 4.5 M, 17.1 M and 66.4 M parameters. Path aggregation tied with the feature pyramid in stage 4 of the detection benchmark; it is kept here for the context its bottom-up path brings to a large dataset. They share these settings:
- 1024 pixel images, aspect ratio kept, batch size 16, GPU memory about 12, 22 and 42 GB for S, M and L.
- Levels 3 to 7 (strides 8, 16, 32, 64, 64), 3 anchor scales and 5 ratios (0.25 to 4) per level.
- Adam at 1e-3, constant until stopped (`number_iterations: 1e10`), moving average 0.9998.
- Test on `validation_test_20000/` every 5,000 iterations, MLflow on, runs in `open_images_experiments/<size>/`.
- Blocks 1 to 6 come from the last averaged checkpoint of the ImageNet run of the same size, in `imagenet_experiments/<size>/`. Until the Large run exists, `check_configuration.py --check_paths` refuses the Large file.

Before choosing the anchors, the `compute_anchor_coverage` diagnosis measured the share of objects with an anchor at IoU 0.5 or more. The kept anchors cover 90.7 % of the objects and 71.9 % of those under 32 pixels a side, against 79.8 % and 40.7 % for the detection benchmark anchors. Level 3 brings most of the small-object gain, for +20 % GPU memory (S) and about 14 % fewer images per second.

### Instance Segmentation
So that the backbone, pyramid and proposal network are not trained twice, each model starts from the COCO detector of its size. The `pretrained` blocks of `Backbone`, `Neck`, `RegionProposal` and `Detector` name that run. Only the mask head starts random.

1. Fine-tune on the COCO masks, with the mask head kept by the instance segmentation benchmark.
2. Record the mask and box mAP on the COCO validation set.

### Order And Resources
To find problems on cheap runs first, S goes through every stage before M and L start.

1. S: classification, Open Images detection, COCO detection, instance segmentation, then a check of the whole chain.
2. M and L: the same chain, at most 2 trainings at a time on a DGX Spark, 25 GB of memory left free.
3. Final table per size: accuracy, box and mask mAP, speed per export format, memory and parameters.

### Open Questions
In case you pick up the plan, one choice is still open: the number of iterations of each stage and size. The classification and Open Images detection files train with a constant learning rate until stopped (`number_iterations: 1e10`).
