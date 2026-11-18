# Pretraining Strategy

Before training the production models, this plan sets the order in which five backbone sizes are pretrained on classification, then fine-tuned for object detection and instance segmentation, so that every size ends with the same three models and one comparison table.

## Contents
- [Model Sizes](#model-sizes)
- [Classification Pretraining On ImageNet-Merged](#classification-pretraining-on-imagenet-merged)
- [Object Detection](#object-detection)
- [Instance Segmentation](#instance-segmentation)
- [Order And Resources](#order-and-resources)
- [Open Questions](#open-questions)

## Model Sizes
To keep the earlier ImageNet-1k runs useful, sizes S, M and XL reuse their backbones, L reuses the long model and only T is new; every backbone has 6 blocks, written as convolutions x filters per block.

| Size | Backbone | Configuration (`vision_training/configurations/classification/imagenet_merged/`) |
|---|---|---|
| T | 1x16 · 2x16 · 2x16 · 4x32 · 5x32 · 5x32 | New `configuration_tiny_model.yaml` |
| S | 1x32 · 2x32 · 2x32 · 16x32 · 20x32 · 20x32 | `configuration_small_model.yaml` |
| M | 1x64 · 2x64 · 2x64 · 4x64 · 5x64 · 5x64 | `configuration_classic_model.yaml` |
| L | 1x64 · 2x64 · 2x64 · 16x64 · 20x64 · 20x64 | `configuration_long_model.yaml` |
| XL | 1x64 · 2x64 · 2x64 · 16x128 · 20x128 · 20x128 | `configuration_large_model.yaml` |

## Classification Pretraining On ImageNet-Merged
Each size first learns general features on ImageNet-merged (21,842 classes, 256 pixels), with the same recipe for all five so that their scores compare.

1. Start S, M and XL from the backbone of their ImageNet-1k runs in `experiments/` (Small, Classic, Large), with a new head for the 21,842 classes; start T and L from random weights.
2. Train with the moving average of the weights, and keep the averaged weights.
3. Choose the checkpoint on the accuracy of the test evaluations (the 50,000 ImageNet-1k images of `test_data_folder`), the same for every size; a k-NN check would need new code, the k-NN evaluation belonging to the self-supervised trainer only.
4. Record accuracy, model size and speed (`diagnosis_modes.benchmark_speed`) of each size in the final table.

## Object Detection
Whenever a size has its pretrained backbone, the detector loads it into its 7-block backbone, the 7th block starting random (a deeper backbone loaded from a shorter one), with the settings kept by the detection benchmark (feature pyramid, learning rate 1e-3, DIoU-NMS, per-level proposal selection).

1. Convert Open Images V7 (1.74 million training images, 601 classes, boxes in `detections.csv`) into the COCO `labels.json` the loader reads, and decide how its group-of boxes are handled: the loader reads every box as an ordinary object, as it does for the COCO crowd boxes.
2. Pretrain the detector of each size on Open Images.
3. Fine-tune it on COCO 2017 (118,287 training images, 80 classes, every object labelled), the score compared with published results being the COCO validation box mAP.
4. Keep the Open Images detector as well, as the start of detection projects whose classes are not in COCO.

## Instance Segmentation
So that the backbone, pyramid and proposal network are not trained twice, each instance segmentation model starts from the COCO detector of its size, only the mask head starting random.

1. Fine-tune on the COCO masks, with the mask head kept by the instance segmentation benchmark.
2. Record the mask and box mAP on the COCO validation set.

## Order And Resources
To find problems on cheap runs first, T and S go through every stage before M, L and XL start.

1. T and S: classification, Open Images detection, COCO detection, instance segmentation, then a check of the whole chain.
2. M, L and XL: the same chain, at most 2 trainings at a time, with 25 GB of memory left free on the DGX Spark.
3. Final table per size: classification accuracy, Open Images and COCO box mAP, COCO mask mAP, speed per export format, memory and parameters.

## Open Questions
- Number of iterations of each stage and size.
- Image size of the detection stages (1024 pixels in the detection benchmark).
- Whether Open Images detection keeps its 601 classes or a subset close to the production use cases.
- Whether COCO crowd boxes and Open Images group-of boxes should become ignored regions instead of objects.
