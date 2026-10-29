# Tasks Reference

Details of every task, completing the [Tasks](../../README.md#tasks) section of the README. The [task workflows](../task%20workflows/) follow one run of every task function by function.

## Table Of Content
- [Synthetic Datasets](#synthetic-datasets)
- [Common Dataset Layout](#common-dataset-layout)
- [Classification (CLS)](#classification-cls)
- [Object Detection (DET)](#object-detection-det)
- [Instance Segmentation (INS)](#instance-segmentation-ins)
- [Keypoint Detection (KPT)](#keypoint-detection-kpt)
- [Semantic Segmentation (SEG)](#semantic-segmentation-seg)
- [Sequence Tasks (M2O, M2M, CMP, SEL)](#sequence-tasks-m2o-m2m-cmp-sel)
- [Sequence Classification (M2O)](#sequence-classification-m2o)
- [Sequence Labelling (M2M)](#sequence-labelling-m2m)
- [Sequence Comparison (CMP)](#sequence-comparison-cmp)
- [Frame Selection (SEL)](#frame-selection-sel)
- [Self-Supervised Learning (SSL)](#self-supervised-learning-ssl)

## Synthetic Datasets
`vision_training/data/synthetic/create_synthetic_dataset.py` generates the dataset of every task. Its configuration is listed in the [Quick Start](../../README.md#quick-start-with-synthetic-data).

| Task | Content | Classes |
|---|---|---|
| CLS | one shape per image | the 35 shapes |
| DET | the INS dataset (`task = "instance_segmentation"`), boxes around the visible part of every shape | the 35 shapes |
| INS | up to 6 shapes per image, with the run length encoding of their visible part; a shape under 64 visible pixels is left out | the 35 shapes |
| KPT | 2 to 4 upright objects per image, some keypoints hidden (v = 1) or outside the image | 33 classes: the shapes with keypoints, `stick_figure`, `house` |
| SEG | up to 6 shapes per image, with the class mask | the 35 shapes and the background |
| M2O | one shape crossing 2 to 12 frames | `moving_left`, `moving_right`, `moving_up`, `moving_down` |
| M2M | 5 to 10 shapes redrawn every frame over 2 to 12 frames, at most one appearing or disappearing per frame | `no_change`, `appearing_<shape>`, `disappearing_<shape>` (71 labels) |
| CMP | before and after members of 2 to 12 frames, 2 to 6 shapes before | `loss_of_object`, `same_number_of_objects` |
| SEL | one shape per frame over 4 to 12 frames, the largest circle, triangle and star to select | `unselected`, `circle`, `triangle`, `star` |
| SSL | the CLS dataset, its classes only used by the k-NN evaluation | the 35 shapes |
| SSL | 5 to 10 shapes per image labelled by their count (`task = "self_supervised"`, 500,000 training images, the shape categories in `shape_category_ids`), with the backbone of the M2M launches (`sequence_labelling/run-9.yaml`) so it can initialise them | `5_shapes` to `10_shapes` |

- **The 35 shapes**: square, circle, triangle, rectangle, pentagon, hexagon, plus sign, star, octagon, rhombus, parallelogram, trapezoid, kite, heptagon, nonagon, ellipse, semicircle, annulus, capsule, L-shape, and the hollow `hollow_<shape>` of square, triangle, rectangle, pentagon, hexagon, octagon, rhombus, parallelogram, trapezoid, kite, heptagon, nonagon, ellipse, semicircle and capsule.
- **Sizes**: from 64 pixels (`MINIMUM_SHAPE_SIZE`) to a third of the image side. A hollow shape measures at least 16 pixels (`HOLLOW_MINIMUM_SIZE`); its hole is the shape scaled by 0.4 to 0.6 (`HOLLOW_INNER_SIZE_RATIO_RANGE`) inside walls of at least 2 pixels (`HOLLOW_MINIMUM_WALL_PIXELS`).
- **Distinct classes**: rhombus angles of 44° to 70°, parallelogram sides in a ratio of 1.5 to 2.5, trapezoids with parallel sides of different lengths, ellipses and capsules 1.5 to 3 times as long as wide.
- **KPT keypoints**: the corners of the polygons, the tips of the star and plus sign, the ends and corners of the capsule, the 11 joints of the stick figure, and for hollow shapes the corners of the hole after the outer ones. `circle`, `ellipse`, `annulus` and `hollow_ellipse` have none: left out of `label_mapping.json`, they only distract the network.
- **Category identifiers**: the hollow shapes are 21 to 35, `stick_figure` and `house` 36 and 37. A dataset generated before shapes were added keeps its own classes.

## Common Dataset Layout

```text
<dataset>/
├── label_mapping.json
├── train/
│   ├── data/
│   └── labels.json
├── validation/
│   ├── data/
│   └── labels.json
└── test/
    ├── data/
    └── labels.json
```

- `train_data_folder`, `validation_data_folder` and `test_data_folder` point to the splits. The validation split gives the validation curves (`validation_iteration_ratio`), and the test split is evaluated whole at every `weight_saving_iterations` checkpoint. Both may be the same folder.
- `labels.json` names the images relative to `data/`, and its categories have integer identifiers.
- `label_mapping.json` (`Data.label_mapping.mapping_file_path`, `mode: "multi_class"`) maps every category identifier, as a string, to a class index. The converters and the synthetic generator write it next to the splits.

```json
{"1": 0, "2": 1, "3": 1}
```

- CLS, M2O, M2M, CMP and SEL predict the classes 0 to K-1. DET, INS, KPT and SEG predict 1 to K, 0 being the `background`. The class indices must have no gaps (checked for every task but KPT and SSL).
- Categories mapped to one class are merged, the class being named `name_2+name_3` and counted once.
- In DET, INS, KPT and SEG, a category mapped to 0 is left out. DET and INS warn about it, since a mapping numbered from 0, as written for the classifiers, drops its first category.
- DET, INS and SEG accept `mode: "binary"`, which maps every category to the foreground class 1 without a mapping file.

A category missing from the mapping is left out:

| Task | Effect |
|---|---|
| CLS multi-class, M2O, CMP | its samples are skipped |
| CLS multi-label | it is dropped from the classes of its samples |
| M2M | its frames are ignored |
| SEL | its frames become unselected |
| DET, INS, KPT | its objects are ignored (DET and INS list these categories) |
| SEG | its pixels become background |

## Classification (CLS)
Predicts the class of an image (`ClassificationHead.output_mode: "multi_class"`, softmax) or every class present in it (`"multi_label"`, sigmoid).

**Data.** Every sample gives its `category_id`, or its `category_ids` when multi-label:

```json
{
  "categories": [{"id": 1, "name": "circle"}, {"id": 2, "name": "square"}],
  "samples": [{"id": 1, "file_name": "circle/image_0001.png", "category_id": 1}]
}
```

**Converting a dataset**
- One folder of images per class: `vision_training/data/classification/class_folder_manifests.py` writes the manifests and the label mapping, `data/` linking to the class folders without copying them. Set `split_folders` and `output_folder` in its main block, then run `uv run python -m vision_training.data.classification.class_folder_manifests`.
- ImageNet-1k: `imagenet_download.py` downloads it from Hugging Face (about 310 GB).
- Tencent ML-Images (17.6M multi-label images, 11,166 categories, about 1.7 TB), to pre-train the backbones: `foundation_model_image_downloader.py`.

**Configuration** (`ClassificationHead`)
- `global_pooling`: `average`, `max` or `flatten`.
- `fully_connected`: the hidden layers. With `fully_connected_normalization` set to `"batch"` or `"layer"`, every hidden layer is linear, normalization, ReLU, and at least one hidden layer is needed. The batch normalization needs a `batch_size` of at least 2 and drops the last incomplete training batch of every epoch.
- `output_mode`, `label_smoothing`, `class_weighting`.
- `decorrelation`: adds the DeCov penalty ([Cogswell et al., 2016](https://arxiv.org/abs/1511.06068)) to the loss, `weight` times the sum of the squared correlations between the different units of layer `hidden_layer` (counted from 1, before its ReLU) over the batch, so the units carry different information. It is tracked as `Decorrelation Loss` and `Hidden Feature Correlation`, and cannot reach 0 with more units than images in a batch. It gives no run name token and never runs at inference.

**Evaluation**
- Accuracy, balanced accuracy, macro F1 and top-5 accuracy (multi-label: subset accuracy, micro and macro F1), and the precision, recall and F1 of every class.
- `class_thresholds.json`: the score threshold of every class reaching `target_precision`, with its recall, the lowest threshold keeping that precision (the highest recall). With a list of precisions such as `[0.99, 0.95, 0.90]`, the first one is the main target (`target_precision` and `classes`, read by the deployment) and the others are in `additional_targets`, with their `threshold_at_precision_<p>` and `recall_at_precision_<p>` columns in `class_metrics.csv`. The per-class table of the report keeps the main target, and a `Per Class Thresholds` table gives the threshold and recall of every class at every target (with `per_class_report`).
- With `per_class_report`: the ROC AUC and average precision of every class, their means `Macro ROC AUC` and `Macro Average Precision`, and `roc_curves.png` and `precision_recall_curves.png`, with a dotted line per target precision and a point at the threshold of every class. The curves share one plot up to 10 classes, then a grid of panels, keeping the 100 worst classes above 100. `precision_recall_thresholds.png` zooms on these points and writes every threshold in the colour of its curve, rounded down to the fewest decimals (at least 4, at most 9 significant digits) selecting the same samples as the exact threshold, so that a threshold can be chosen from the image.
- `predictions.csv` gives the probabilities with 9 significant digits, enough to find the thresholds of `class_thresholds.json` again.
- The confusion matrix, `predictions.csv` (`save_predictions`), and the misclassified images in `Errors/<TRUE>-TO-<PREDICTED>/` (`error_dumps`).

## Object Detection (DET)
A two-stage detector. Every entry of `DetectionLevels` is a backbone level with its own anchors, region proposal head and detection head. In training, the ground truth boxes are added to the proposals, so every object gives a positive proposal even when the region proposal network misses it.

**Data.** A COCO `labels.json`, the boxes being `[x, y, width, height]` in pixels of the original image. The crowd annotations (`"iscrowd": 1`) are ordinary objects.

```json
{
  "images": [{"id": 1, "file_name": "000001.png", "width": 1024, "height": 1024}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [120, 340, 64, 48], "area": 3072, "iscrowd": 0}],
  "categories": [{"id": 3, "name": "triangle"}]
}
```

**Augmentation**
- An object keeping less than 25 % of its box inside the image after a crop, translation or rotation is removed (`MINIMUM_VISIBLE_AREA_FRACTION`).
- `random_zoom_out`, applied after the crop, shrinks the image to 50 to 100 % of each side on a black canvas (`zoom_out_side_range`), so the finest levels also see the large objects. It is on in the example configurations of DET and INS (probability 0.3).
- `random_translation` (up to `maximum_translation_fraction` of each side) leaves a border that is black in the image, ignored in the class masks and background in the instance masks.

**Converting a dataset.** `vision_training/data/data_retriever.py` downloads COCO 2017 with FiftyOne (about 25 GB). Set the dataset folder and `maximum_samples` in its main block, then run `uv run python -m vision_training.data.data_retriever`.

**Configuration**
- `DetectionLevels`: keyed by the quoted number of the backbone block feeding the level, finest first. Every level holds its `anchors` (scales in input pixels, aspect ratios height / width), its `region_proposal` convolutions and its `detector` head.
- `RegionProposal` and `Detector`: the settings shared by the levels (normalization, modules, proposal counts, IoU thresholds, sampling, pretrained heads).
- `Optimization.loss_weights`: the weights of `region_proposal_classification_loss`, `region_proposal_localisation_loss`, `detection_classification_loss` and `detection_regression_loss`, at least 0 and not all 0. The curves show the losses unweighted and the total weighted.

```yaml
DetectionLevels:
  "5":
    anchors: { scales: [32, 64, 128], aspect_ratios: [0.5, 1.0, 2.0] }
    region_proposal: { convolutions: [2, 64] }
    detector: { roi_align_pool_size: 7, convolutions: [2, 64], fully_connected: [256, 128] }
  "6":
    anchors: { scales: [64, 128, 256], aspect_ratios: [0.5, 1.0, 2.0] }
    region_proposal: { convolutions: [2, 64] }
    detector: { roi_align_pool_size: 7, convolutions: [2, 64], fully_connected: [256, 128] }
```

**Evaluation**
- **mAP** at IoU 0.5, 0.75, 0.95 and 0.5:0.95, with the area, 11-point and 101-point methods, and the AP@0.5 of every class. The detections below `evaluation.score_threshold` are ignored.
- **By object size**: small (below 32²), medium (32² to 96²) and large objects, by box area in network input pixels as in COCO, also per class in `class_metrics.csv` (`average_precision_50_<size>`, `average_precision_50_95_<size>`).
- **Mean average recall**: the share of the objects found by any kept detection, per IoU and size. It separates the objects never found from those found with a low score.
- **Anchor recall**, at every launch (`anchor_recall.csv`): the share of the training objects whose best anchor overlaps them by at least `RegionProposal.foreground_iou_threshold.min`, per level and size.
- **Proposal recall**, at every evaluation (`proposal_recall.csv`): the share of the test objects covered by a proposal at IoU 0.5 and 0.7, per size and level, with the proposals per image.
- **Class thresholds**: `class_thresholds.json` gives the score threshold of every class reaching `target_precision`.
- **Confusion matrix** of the deployed detections (score from `Detector.inference.score_threshold`), matched to the objects at IoU 0.5, with a `background` row for the false positives and a `background` column for the missed objects.
- **Error images** (`error_dumps.activated`): `Errors/FALSE-POSITIVE/<class>/`, `Errors/MISSED/<class>/` and `Errors/<TRUE>-TO-<PREDICTED>/`, at most `maximum_per_class_pair` per folder.
- **TensorBoard tags**: `mAP@0.5 By Size/<size>`, `mAP@[0.5:0.95] By Size/<size>`, `mAR@0.5/<all or size>`, `mAR@0.75/<all or size>`, `mAR@[0.5:0.95]/<all or size>`, `Proposal Recall@0.5/<group>`, `Proposal Recall@0.7/<group>`, `Proposals Per Image/<group>`.

## Instance Segmentation (INS)
The detector of DET with one mask head per detection level, as in Mask R-CNN ([He et al., 2017](https://arxiv.org/abs/1703.06870)).
- The mask head of a level pools every box with RoI Align (`mask_head.roi_align_pool_size`, 14 by default) and refines it with convolutions.
- It doubles the resolution up to `MaskHead.mask_resolution` (28 by default), with transposed convolutions or bilinear upsamplings (`MaskHead.upsampling`), and gives one mask per class, the detected class being kept.
- It trains on the positive proposals, and runs on the final detections at inference.

**Data.** The COCO `labels.json` of DET, every object having a `segmentation`: polygons (flat `[x1, y1, x2, y2, ...]` lists in pixels of the original image) or a run length encoding `{"size": [height, width], "counts": ...}`, compressed or not.

```json
{"id": 1, "image_id": 1, "category_id": 3, "iscrowd": 0, "segmentation": [[120, 388, 152, 340, 184, 388]]}
```

- The masks are built at the network resolution. The box of every object is the tight box of its mask, so no `bbox` is needed.
- An object without `segmentation` stops the loading. An object whose mask vanishes at the network resolution is removed, except from the test objects with `evaluation.original_resolution`.
- After augmentation, an object keeping less than 25 % of its mask pixels inside the image is removed.

**Configuration.** The keys of DET, and:
- a `mask_head` in every level, e.g. `mask_head: { roi_align_pool_size: 14, convolutions: [4, 64] }`;
- `MaskHead`: mask resolution, upsampling, normalization, modules, and `inference.mask_threshold`, the probability from which a pixel belongs to its object;
- the weight of the `mask_loss` in `Optimization.loss_weights`, raised when the masks lag behind the boxes;
- `evaluation.original_resolution`.

**Evaluation**
- The mask mAP and the box mAP of the same detections. `Mask mAP@0.5/area` compares the checkpoints. The box metrics keep the names of DET, so INS and DET runs compare; the mask columns of `class_metrics.csv` start with `mask_`.
- The masks are compared at the network resolution or, with `evaluation.original_resolution`, at the original resolution of every test image as COCO does.
- The size tables, mean average recall, thresholds, confusion matrix and error images are those of DET, given for the masks and the boxes, the detections being matched by their masks. The mask tags are `Mask mAP@0.5 By Size/<size>` and `Mask mAR@0.5/<all or size>`.

## Keypoint Detection (KPT)
A bottom-up keypoint detector, as in HigherHRNet ([Cheng et al., 2020](https://arxiv.org/abs/1908.10357)): it finds the keypoints of all the objects of an image at once, and groups them into objects with the tags of associative embedding ([Newell et al., 2017](https://arxiv.org/abs/1611.05424)). It needs no box.

**Network**
- **Stem**: the first blocks of the backbone, down to 1/4 of the image: 3 blocks with `Backbone.last_max_pooling: False`, or 2 with True.
- **`HRNet`**: keeps a branch at 1/4 of the image throughout, adds a branch at half the resolution and twice the width at every stage, and mixes the branches at the end of every module. `base_width: 32`, `modules_per_stage: [1, 4, 3]` and `blocks_per_branch: 4` is HRNet-W32.
- **`KeypointHead`**: a 1x1 convolution giving one heatmap and one tag map per keypoint of every class, at 1/4 of the image. `higher_resolution: True` adds the HigherHRNet branch, giving second heatmaps at 1/2 of the image for the small objects.
- **`hidden_width`**: above 0, adds a 3x3 convolution of that width before the 1x1 convolutions. Set it to at least the number of heatmap channels when they outnumber `HRNet.base_width` (243 channels and `hidden_width: 256` in `configuration_synthetic_shapes.yaml`); 0 follows the paper.
- **Channels**: the keypoints of every class, so classes can have different keypoints (a person 17, a car 12).

**Training**
- The heatmap targets are Gaussians of `KeypointHead.heatmap_sigma` cells centred on the labelled keypoints. The heatmap loss is the focal loss of CornerNet ([Law and Deng, 2018](https://arxiv.org/abs/1808.01244)), since the keypoint cells are a tiny share of all cells.
- With `ignore_unlabelled_objects`, the pixels of the objects without labelled keypoints (COCO crowds, persons without keypoints) are left out of the heatmap loss, so the network is not taught that nothing is there.
- The tags have no target value. The pull loss gathers the tags of the keypoints of an object, and the push loss separates the objects of the same class.
- The loss is the heatmap loss plus `tag_pull_weight` x the pull loss plus `tag_push_weight` x the push loss.

**Grouping**, at inference, by `group_keypoint_candidates` (`vision_training/models/keypoints/keypoint_grouping.py`):
1. The candidates of every channel are its `inference.maximum_candidates` highest local maxima, refined below the cell size as in DARK ([Zhang et al., 2020](https://arxiv.org/abs/1910.06278)).
2. Class by class and keypoint by keypoint, the candidates above `keypoint_threshold` are matched to the objects already formed by the distance between their tags (Hungarian matching). A candidate farther than `tag_threshold` from every object starts a new one.
3. The score of an object is the sum of its keypoint scores divided by the keypoints of its class. The evaluation keeps every object; the sample images and the deployment keep those scoring at least `object_score_threshold`.

**Data.** The COCO `labels.json` of DET. Every object has its `keypoints`, a flat `[x1, y1, v1, ...]` list in pixels of the original image, v being 0 (not labelled), 1 (hidden) or 2 (visible). Every category has its `keypoints` names and its `skeleton`, the 1-based keypoint pairs linked in the drawings.

```json
{
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0, "bbox": [120, 40, 80, 200],
                   "keypoints": [160, 60, 2, 140, 120, 2, 180, 120, 1], "num_keypoints": 3}],
  "categories": [{"id": 1, "name": "person", "keypoints": ["head", "left_hand", "right_hand"], "skeleton": [[1, 2], [1, 3]]}]
}
```

- The objects without any labelled keypoint are left out of the labels. With `ignore_unlabelled_objects`, each of them needs its `segmentation`, as in INS.
- The categories merged into one class must have the same keypoint names.
- COCO 2017 person keypoints (`annotations_trainval2017.zip`): every split folder links the COCO images as `data/` and `person_keypoints_<split>2017.json` as `labels.json`. The test keypoints being withheld, the example configuration evaluates on the validation split.

**Augmentation**
- The geometric types move the keypoints; `random_perspective` and `random_jigsaw` are refused.
- A keypoint leaving the image becomes unlabelled. An object is removed when less than 25 % of its box and none of its keypoints stay inside.
- A keypoint keeps its name whatever the augmentation: a flipped `left_wrist` stays `left_wrist` on the right side. The flips and rotations only suit keypoints whose names depend neither on the side nor on the orientation, and both keypoint configurations turn the flips and 90 degree rotations off.

**Configuration.** `keypoint_detection/configuration_example.yaml` (COCO persons, HigherHRNet-W32 on 512x512 images). `Data.image_settings.size` must be a multiple of 4 x 2 to the power of the number of stages (32 for 3 stages). `Backbone.pretrained` loads the first blocks of any backbone, as many as the stem has; `HRNet.pretrained` and `KeypointHead.pretrained` load those of another KPT run.

```yaml
HRNet:
  base_width: 32
  modules_per_stage: [1, 4, 3]
  blocks_per_branch: 4
  modules: { squeeze_and_excite: True, residual_nets: True }
  normalization: { feature_map_normalization: "batch", batch_normalization_momentum: 0.1 }
KeypointHead:
  higher_resolution: True
  hidden_width: 0
  heatmap_sigma: 2.0
  ignore_unlabelled_objects: True
  tag_pull_weight: 0.1
  tag_push_weight: 0.1
  inference: { maximum_candidates: 30, keypoint_threshold: 0.1, tag_threshold: 1.0, object_score_threshold: 0.2 }
```

**Evaluation.** Every object found is matched to the ground truth objects of its class with the object keypoint similarity (OKS) of COCO, which replaces the IoU:

OKS = Σ exp(-d² / (2 s² k²)) δ(v > 0) / Σ δ(v > 0)

d is the distance to the ground truth keypoint, s² the ground truth box area, k = 2 x sigma, over the labelled keypoints. The sigma of a keypoint is `evaluation.keypoint_sigmas.<category>.<keypoint>`, or `default_keypoint_sigma` (0.05); a loosely placed keypoint (a hip) deserves a larger one than a sharp corner.
- **OKS mAP**, by size and per class, and the OKS mean average recall, as the box metrics of DET. `OKS mAP@0.5/area` compares the checkpoints.
- **Matched Objects**: the share of the ground truth objects matched to an object found.
- **PCK@0.05**: the share of the labelled keypoints within 0.05 of the longest box side of their object.
- **Mean OKS** of the ground truth objects, and **mean keypoint error** in network input pixels.
- The tables `Keypoint Accuracy Per Class` and `Keypoint Accuracy Per Keypoint`, saved in `class_metrics.csv` and `keypoint_metrics.csv`.
- TensorBoard: the DET tags prefixed by `OKS ` (`OKS mAP@0.5 By Size/<size>`, `OKS mAR@0.5/<all or size>`), and `Keypoint Accuracy/Matched Objects`, `Keypoint Accuracy/PCK@0.05`, `Keypoint Accuracy/Mean OKS` and `Keypoint Accuracy/Mean Keypoint Error (pixels)`.

In the sample images, the left keypoints are blue, the right ones orange and the others yellow, the hidden ones being drawn as rings.

## Semantic Segmentation (SEG)
A U-Net whose encoder is the backbone, the decoder merging the skip connection of a backbone block at every level.

**Data.** Every split holds its images in `data/` and their masks in `masks/`. A mask is a single-channel uint8 PNG of the size of its image holding the category of every pixel, 0 for the background and 255 for the unlabelled pixels, which are ignored. The optional `category_ids` of a sample list its mask categories for the class counts of `data.json`.

```json
{
  "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "road"}],
  "samples": [{"id": 1, "file_name": "image_001.png", "mask_file_name": "image_001.png", "category_ids": [1, 2]}]
}
```

**Configuration.** `SegmentationDecoder` (upsampling, the convolutions of every level keyed by the backbone block it merges, modules, normalization, deep supervision), `SegmentationLoss`, `Data.tiling`, and the evaluation keys `save_predicted_masks`, `original_resolution` and `boundary_tolerance`.

**Losses.** Every loss has its weight in `SegmentationLoss`, 0 turning it off; the cross entropy, Dice or Lovász loss must be on.
- `cross_entropy_weight`: cross entropy of the labelled pixels, with `focal_gamma`, `label_smoothing` and `class_weighting`.
- `dice_weight`: one minus the mean soft Dice of the classes in the batch, every class counting the same whatever its area.
- `lovasz_weight`: the Lovász-Softmax loss, which optimizes the mean IoU directly, usually to fine-tune a run trained with the cross entropy.
- `boundary_weight`: cross entropy of the pixels within `boundary_distance` pixels of an edge between two classes, to sharpen the contours.

**Deep supervision.** `SegmentationDecoder.deep_supervision_weights`, e.g. `{ "5": 0.1, "4": 0.2, "3": 0.3, "2": 0.4 }`, adds a class layer to the listed decoder levels, trained with the losses times the weight of the level (`deep_supervision_loss`), to help deep decoders. These layers only serve the training, and are transferred only with `load_class_layers` ([Adding Classes](training_reference.md#adding-classes)). `{}` turns it off.

**Training on tiles.** With `Data.tiling.activated: True`, the images and masks are scaled by `Data.tiling.scale` and cut into tiles of `Data.image_settings.size` pixels, so large images keep their small objects:
- the training tiles are cut at random places at every epoch, and the validation tiles at a place fixed per image;
- the test images are segmented whole, tile by tile, the probabilities being averaged where the tiles overlap by `Data.tiling.overlap` pixels.

**Evaluation**
- Mean IoU, mean Dice, mean boundary IoU and boundary F1, pixel accuracy and frequency weighted IoU, and the per-class metrics.
- The boundary metrics only score the pixels within `boundary_tolerance` pixels of the contours, where the IoU hardly sees an error.
- With `per_class_report`: the ROC AUC and average precision of every class, from histograms of the pixel probabilities, and their curves.
- With `original_resolution: True`, the predictions are brought back to the original size of every test image and compared with its original mask, as the deployed output is used.
- `save_predicted_masks` writes the predicted masks to `Predicted_Masks/`.

## Sequence Tasks (M2O, M2M, CMP, SEL)
- Every sequence holds its frames in `data/<sequence name>/`, listed in order in `labels.json`.
- The backbone runs on every frame, `ClassificationHead.global_pooling` turns every feature map into a frame vector (`flatten` keeping its whole content), and the temporal module (`TemporalModel.type`) combines the frames. `mean_pooling`, `max_pooling` and `attention_pooling` ignore their order; `gru`, `lstm`, `transformer` and `temporal_convolution` follow it.
- Every sequence is fed whole. A sequence (or CMP member) longer than `Data.sequence_settings.maximum_sequence_length` is left out with a red warning. The batches are padded with masked black frames, which never change the outputs.
- `Data.data_augmentation.independent_frames` draws the photometric augmentations frame by frame. `shuffle_frames` shuffles the frames of sequences whose order carries no information, with `TemporalModel.positional_embedding: False` for the transformer to treat them as sets.
- `Backbone.normalization.feature_map_normalization: "sequence"` normalizes over the real frames of every sequence.

## Sequence Classification (M2O)
Predicts one class per sequence (many to one).

```json
{
  "categories": [{"id": 1, "name": "moving_left"}, {"id": 2, "name": "moving_right"}],
  "sequences": [{"id": 1, "name": "sequence_0001", "category_id": 1, "frames": ["sequence_0001/000000.png", "sequence_0001/000001.png"]}]
}
```

- **Configuration**: `Data.sequence_settings`, `TemporalModel` (all 7 types) and `ClassificationHead`, with the evaluation keys of CLS. A class given by a motion needs `global_pooling: flatten`, as in both configurations: `average` and `max` remove the position of the object, and the synthetic moving shapes then stay at chance.
- **Evaluation**: the metrics and files of CLS, on the sequences.

## Sequence Labelling (M2M)
Predicts one class per frame (many to many). With `TemporalModel.causal: True` (transformer, temporal convolution), every frame only sees the frames before it, as in a live stream.

```json
{"id": 1, "name": "sequence_0001", "frame_category_ids": [1, 1, 2], "frames": ["sequence_0001/000000.png", "sequence_0001/000001.png", "sequence_0001/000002.png"]}
```

- **Configuration**: the keys of M2O, with the temporal types `gru`, `lstm`, `transformer` and `temporal_convolution` only.
- **Evaluation**: the metrics of CLS on the frames, the sequence exact match (every frame right), and two segment metrics, a segment being a run of frames of one class:
  - **Edit Score**: 1 minus the segments to insert, remove or replace to turn the predicted order of segments into the true one, divided by the larger number of segments. It is low when the prediction is fragmented.
  - **Segment F1@10, @25 and @50**: a predicted segment is right when it overlaps an unmatched true segment of its class by at least 10, 25 or 50 % IoU.

For example, true `A A A A A B B B B B` predicted `A A B A A B B A B B` has 80 % of its frames right, an edit score of 0.33 (`A B` against `A B A B A B`), a segment F1@10 of 0.50 and a segment F1@50 of 0. The report note `Segments` counts the true and predicted segments.

## Sequence Comparison (CMP)
Predicts one class per pair of a before and an after sequence, e.g. whether an object was lost between two visits. Both members go through the same backbone and temporal module, combined by `Fusion.type`: `concatenation`, `difference` or `cross_attention`.

```json
{
  "categories": [{"id": 1, "name": "loss_of_object"}, {"id": 2, "name": "same_number_of_objects"}],
  "pairs": [{"id": 1, "name": "pair_0001", "category_id": 1,
             "before": ["pair_0001/before/000000.png", "pair_0001/before/000001.png"], "after": ["pair_0001/after/000000.png"]}]
}
```

- **Data**: the members may hold different numbers of frames. `vision_training/data/sequence/pair_list_manifests.py` converts pair list files `{"source": <folder>, "data": [{"before": ..., "after": ..., "label": ..., "sequence": ...}]}` (set `pair_list_files` and `output_folder` in its main block, then run `uv run python -m vision_training.data.sequence.pair_list_manifests`).
- **Configuration**: the keys of M2O and `Fusion`. `Fusion.sequence_normalization_scope` normalizes every member on its own (`"member"`) or both together (`"pair"`).
- **Evaluation**: the metrics and files of CLS, on the pairs.

## Frame Selection (SEL)
Chooses in every sequence the `Selection.frames_per_class` frames of every wanted class, e.g. the sharpest view of each side of an object. Every class except `Selection.unselected_category` is wanted, and the other frames are unselected.

**Model.** The model of M2M, trained with the per-frame loss and a selection loss (`Selection.selection_loss_weight`) that makes the frames of a sequence compete for every wanted class.

**Decoding**
- `hungarian`: the most probable labelling of the sequence with every wanted class chosen `frames_per_class` times, every frame at most once.
- `greedy`: every frame takes its most probable class; a class with too many frames keeps the most probable, one with too few is completed with the best unselected frames.

For example, with classes A and B and three frames of probabilities (A, B, unselected) (0.50, 0.45, 0.05), (0.45, 0.05, 0.50) and (0.02, 0.08, 0.90), the greedy rule gives A to frame 1 and B to frame 3, while the Hungarian assignment gives B to frame 1 and A to frame 2, a labelling ten times more probable. Both are scored at every evaluation; `Selection.decoding` is the one drawn, thresholded and deployed.

**Data.** The layout of M2M, `frame_category_ids` giving the unselected or a wanted category for every frame. The mapping must map the unselected category, and a sequence without exactly `frames_per_class` frames of every wanted class stops the training at startup.

**Configuration.** The keys of M2M and `Selection` (`unselected_category`, `decoding`, `frames_per_class`, `selection_loss_weight`). A bidirectional or non-causal temporal module compares every frame with the whole sequence.

**Evaluation**
- `Hungarian Selection Accuracy` and `Greedy Selection Accuracy`: the share of the true frames chosen for their class, the one of `Selection.decoding` being the headline metric.
- `<Decoding> Fully Right Sequences`, `<Decoding> Unselected Precision` and `<Decoding> Unselected Recall`.
- `Top-<k> Selection Accuracy` and `Mean True Frame Rank`: whether a wrong choice was a close one.
- `Frame Accuracy`, `Frame Balanced Accuracy` and `Frame Macro F1`, before decoding.
- `class_thresholds.json`: the probability from which the choices of every class reach `target_precision`; the deployed model flags the choices below it as low confidence.
- `selections.csv` (`save_predictions`): the true and chosen frames of every sequence and wanted class, and `Errors/<CLASS>/<sequence>.png` (`error_dumps.activated`).

The images show the sequence as a grid of thumbnails: green for a right choice, red for a wrong one, dashed yellow for a missed true frame, with a heat map of the probabilities of every wanted class below.

## Self-Supervised Learning (SSL)
Trains the backbone without labels. Two random views of every image go through the backbone, the global pooling and a projection head, and the loss pulls the two views together:
- `ProjectionHead.method: "simclr"`: InfoNCE, every view finding its partner among the other views of the batch by cosine similarity over `simclr.temperature`. Larger batches help.
- `ProjectionHead.method: "barlow_twins"`: the cross-correlation of the embedding features of both views is pushed to the identity, the off-diagonal terms weighted by `barlow_twins.off_diagonal_weight`. It works with smaller batches and wider embeddings.

The projection head only serves the training: the evaluated and exported representation is the pooled feature map before it.

**Data.** The layout of CLS. Every image is trained on, labelled or not; the categories only serve the k-NN evaluation, which uses the test images with exactly one mapped category. A dataset without labels uses an empty label mapping file (`{}`).

**Grouped batches.** `Data.grouped_batches` builds every batch from groups of related images (same camera, site or device type), so that the negatives of an image are hard ones. The groups come from the optional `groups` of every sample:

```json
{"id": 17, "file_name": "data/abc.jpg", "groups": {"camera": "c042", "camera_type": "type_a", "site": "s3"}}
```

- `activated`: grouped batches when True, random batches when False.
- `block_key`: the attribute shared by the images of a block, of `images_per_block` images (at least 2, dividing the batch size).
- `batch_key`: the attribute shared by every image of a batch, or `null` for blocks of any value.

`GroupedBatchSampler` (`vision_training/data/grouped_batch_sampler.py`) rebuilds the batches at every pass, the images that fill no block or batch forming random batches. A split whose samples hold none of the keys keeps random batches with a warning; a split where only some hold them stops the launch. The test split is not grouped, and the section can change when a run resumes.

**Views.** Both views are drawn by `Data.data_augmentation`, which must be on, at the strengths of `SELF_SUPERVISED_AUGMENTATION_STRENGTHS` ([Data Augmentation](training_reference.md#data-augmentation)); every type is drawn per view with its own `probability`.
- `random_crop` is the random resized crop of SimCLR, at probability 1.0 when it is on. The counted shapes configurations turn it off, a crop changing the number of shapes.
- `brightness_contrast_saturation` and `random_hue` are the colour jitter, milder than SimCLR on ImageNet.
- `random_solarize` and `random_black_out` only apply to the second view (`SECOND_VIEW_ONLY_TYPES`), as in BYOL and Barlow Twins.
- A cue shared by both views lets the backbone match them without learning the content: the colour of an object when the colour types are off, its position when the crops are too large.

**Configuration.** `self_supervised/configuration_example.yaml` (Barlow Twins) or `configuration_synthetic_shapes.yaml` (SimCLR).
- `ProjectionHead`: global pooling, `fully_connected` with the embedding size last, normalization, `method`, `simclr.temperature`, `barlow_twins.off_diagonal_weight`. SimCLR uses one hidden layer and a small embedding (`[2048, 128]`), Barlow Twins wide layers (`[2048, 2048, 2048]` or more).
- `batch_size` must be at least 2, and the training and validation splits at least one batch.

**Training metrics**, next to the loss (`InfoNCE Loss` or `Barlow Twins Loss`):
- `Positive Pair Accuracy` (SimCLR): the share of the views whose most similar view is their partner. Near 1 quickly while the evaluation stays poor, it shows a shortcut.
- `Diagonal Correlation` and `Off-Diagonal Correlation` (Barlow Twins), towards 1 and 0.
- `Embedding Standard Deviation`: about 1 when the embeddings spread, 0 when they collapse.

**Evaluation**, without training any classifier:
- `k-NN Accuracy` (headline), `k-NN Balanced Accuracy` and `k-NN Macro F1`: every labelled test image is classified by its `number_neighbours` nearest labelled test images. Every class needs several labelled test images.
- `Effective Rank`: the number of directions the representations spread in, about 1 when they collapse; the headline metric of a test split without labels.
- `Representation Standard Deviation`: its fall towards 0 shows a collapse (about 0.6 to 0.8 when the representations spread).

The sample images show every test image next to two views of it, with the cosine similarity of their representations.

**Using the backbone.** A run of any task sets `Backbone.pretrained.experiment_folder` to the SSL run ([Transfer Between Runs](training_reference.md#transfer-between-runs)). To measure the gain, compare it with a run from random weights on 1 %, 10 % and 100 % of the labels: the gap is expected to be the largest with the fewest labels.
