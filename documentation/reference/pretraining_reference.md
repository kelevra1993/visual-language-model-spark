# Pretraining Reference

If one wishes to start a backbone from pre-trained weights instead of random ones, this reference describes the download of the dataset the classification configurations train on, completing the [Data Requirements](../../README.md#data-requirements) section of the README. A classification, object detection or segmentation run then loads the trained backbone through `Backbone.pretrained` ([Transfer Between Runs](../../README.md#transfer-between-runs)), and the [pretraining strategy](../pretraining_strategy.md) plans the order in which the backbone sizes are pre-trained.

## Table Of Content
To find your way in this reference, the sections are:
- [ImageNet-22k](#imagenet-22k)
  - [Downloading ImageNet-22k](#downloading-imagenet-22k)
  - [Validation And Test Splits](#validation-and-test-splits)

## ImageNet-22k
Should you need the pre-trained backbones of the classification configurations (`configuration_example.yaml` and the `*_model.yaml` files of `vision_training/configurations/classification/`) or of the self-supervised example, download ImageNet-22k with `vision_training/data/classification/imagenet_download.py`. It reads `timm/imagenet-22k-wds` from Hugging Face at a pinned revision (fall11 release, 21,841 classes, 4,608 shards of about 0.29 GB), and writes the `labels.json` manifests and the `label_mapping.json` these configurations read.

### Downloading ImageNet-22k
Before the first shard is fetched, the script checks that this machine is logged in with an account that accepted the terms of the dataset, and it stops when the disk runs low:
1. Accept the terms of [timm/imagenet-22k-wds](https://huggingface.co/datasets/timm/imagenet-22k-wds) (non-commercial research and education only), then log in with `uv run hf auth login`.
2. Keep about 1.3 TB free on the disk of `dataset_folder` (main block of the script, `~/Data/classifier_data/imagenet-22k` by default); the download stops below 100 GB free.
3. Run it. It resumes shard by shard after an interruption, at about 1,500 shards an hour with 16 workers on a DGX Spark.

```bash
uv run python -m vision_training.data.classification.imagenet_download
```

<p align="center">
  <img src="../../README/imagenet_22k_download.png" width="100%" />
</p>

So that a configuration trains on the download, point its `train_data_folder`, `validation_data_folder`, `test_data_folder` and `Data.label_mapping.mapping_file_path` to `manifests/train`, `manifests/validation`, `manifests/test` and `manifests/label_mapping.json` of `dataset_folder`, and its `experiment_folder` to the folder of your runs.

### Validation And Test Splits
In case you compare its scores with other results, the held-out images (18,846) are both the validation and the test split, so a setting chosen on the validation curves is chosen on the test images, and the scores cannot be compared with published ImageNet-1k accuracies (shorter side of the images at most 600 pixels in the source, person classes kept). The split sizes and the retries of the download are in the [tasks reference](tasks_reference.md#classification-cls).
