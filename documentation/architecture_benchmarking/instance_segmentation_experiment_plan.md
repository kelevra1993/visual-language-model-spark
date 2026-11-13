# Instance Segmentation Experiment Plan

- **Configurations:** `vision_training/configurations/instance_segmentation/run-<n>.yaml`, one per launch. `run-1.yaml` is written before the first launch from the base of the detection benchmark (`vision_training/configurations/object_detection/run-1.yaml`) with the values its stages kept, the `MaskHead` section and a `mask_head` in every detection level. Each run is launched with `uv run python -m vision_training.main --configuration_path=vision_training/configurations/instance_segmentation/run-<n>.yaml`, the later files written by a copy of the detection driver (`hyperparameter_experiments/object_detection/run_benchmark_stages.py`) whose setting keys include the mask head settings
- **Dataset:** synthetic hard shapes for instance segmentation (`create_synthetic_dataset(task="instance_segmentation", hard_objects=True)`, generated on 2026-11-04 with the settings of the detection dataset), `/home/robert_kelevra/Data/instance_segmentation_data/synthetic_hard_shapes`: 512,000 train / 50,000 validation / 50,000 test images of 1024 pixels, 5 to 12 of the 35 shape classes per image (8.5 on average), the boxes being those of the visible part of every shape, 11.8 % of the visible parts in several pieces. The masks are compressed COCO RLE strings (a train `labels.json` of 3.8 GB, 10.9 GB at the peak of its loading). `test_10000` holds the first 10,000 test images (84,922 objects), read from the `test` split through a link
- **Launch folders:** `hyperparameter_experiments/instance_segmentation/run_<n>` at the repository root, set as `ExperimentConfiguration.experiment_folder` before every launch

## Strategy

So that the stages can be read like those of the detection benchmark (`object_detection_experiment_plan.md`), the method is the same, the user's choice of 2026-11-12 ("we will proceed in the same manner but the defaults will be our best one for the object detection"):

- one parameter changes at a time, and the best value of a stage is kept for all the following stages;
- every value is trained twice, A and B, two launches side by side sharing the GPU through the MPS daemon, without random seed;
- the score of a value is the test mask mean average precision over the IoU thresholds 0.5 to 0.95 of the averaged weights (`Tensorboard/Test_EMA`, `Mask mAP@[0.5:0.95]/area`) at the last iteration, averaged over its two launches. The mask mAP by object size, the mask mAP@0.75 and the box `mAP@[0.5:0.95]/area` are read next to it, so that a mask gain bought with worse boxes is seen;
- the values are searched one pair at a time. A challenger whose pair mean differs from the best value so far by less than the larger gap between the two launches of either value is a tie, which keeps the earlier value;
- a stage that needs code has its code merged right before its pair and reverted when it is not better, unless the user keeps it.

## Fixed Settings

To compare the instance segmentation with the detection benchmark, the settings of that benchmark stay fixed (its Fixed Settings section):

- `batch_size` 64 images of 1024 pixels, `number_iterations` 15,000, `weight_saving_iterations` 500 (the evaluation of `test_10000` every 500 iterations, unless stage 0 finds it too slow), `intermediate_weight_saving_iterations` 250, `information_dump` 1000, `validation_iteration_ratio` 0;
- the backbone, the detection levels 4 to 7 (strides 16, 32, 64 and 64), their anchors (16 and 32, 48 and 96, 128 and 256, 384 and 768 pixels, aspect ratios 0.5, 1 and 2) and their region proposal and detection heads;
- the region proposal counts and thresholds, 128 positives of 256 sampled anchors, 64 positive proposals of 128 for the detection heads, the ground truth boxes written into the proposals;
- Adam without weight decay, a constant learning rate without warm-up, a moving average of the weights with a decay of 0.9998, MLflow tracking;
- the augmentation of the user's `detection-run-configuration-defaults.yaml` (probability 0.25, the zoom-out, the left-right flip at 0.5 and the photometric types), the masks moving with the geometric ones;
- one loader worker per loader, the loss weights 1 for the five losses (`mask_loss` included);
- `Data.instance_index_maps: True`: the masks of an image travel as one uint8 map numbering its objects, expanded on the GPU, the same masks, labels and boxes as one mask per object on 1,000 test images. The first reference launch, with one mask per object, was loader-bound (3.16 s per iteration, the GPU at 3 to 7 %), and the user stopped it on 2026-11-13 ("I really want something that's fast"); with the faster decoding of the run lengths and the maps, a batch of 64 reaches the GPU in 1.20 s instead of 2.22 s with one worker.

## Starting Values

Before stage 0 is launched, every detection setting is set at the value kept by the detection benchmark, and the mask heads at those of the shipped instance segmentation configurations:

| Parameter | Value |
|---|---|
| `learning_rate` | 1e-3 (detection stage 1) |
| `RegionProposal.background_iou_threshold.min` | 0.1 (detection stage 2) |
| `Neck` | `feature_pyramid`, 32 channels, batch normalization (detection stage 3), read by the mask heads as well |
| `Detector.box_loss` | `smooth_l1` (detection stage 5) |
| `Detector.nms_criterion`, `nms_distance_exponent` | `distance_iou`, 0.6 (detection stage 5b, never measured on masks: stage 0b checks it) |
| `Detector.normalization.feature_map_normalization` | batch (detection stage 6) |
| `RegionProposal.proposal_selection` | `per_level` (detection stage 7) |
| Anchor centres | cell corners, the centring removed after the tie of detection stage 8 |
| `DetectionLevels.<4, 5, 6, 7>.mask_head` | `roi_align_pool_size: 14`, `convolutions: [3, 32]` |
| `MaskHead.mask_resolution` | 28 (one doubling of the pool of 14) |
| `MaskHead.upsampling` | `transposed` |
| `MaskHead.normalization`, `modules` | batch, residual |
| `MaskHead.inference.mask_threshold` | 0.5 |

## Stages

So that the order and the cost of the benchmark are seen at once, the table lists every stage:

| Stage | Parameter | Values | Code | Status |
|---|---|---|---|---|
| 0 | Reference and preflight | the starting values | none | planned |
| 0b | Mask threshold and final NMS, evaluation only | `mask_threshold` 0.4, 0.5, 0.6; greedy NMS against DIoU-NMS 0.6 | none | planned |
| 1 | Mask RoIAlign pool size (the user's stage) | 14, 22, 18, with `mask_resolution` twice the pool (28, 44, 36) | none | planned |
| 2 | Mask resolution | one doubling against two (`mask_resolution` 2 or 4 times the pool kept by stage 1) | none | planned |
| 3 | Mask head size | `convolutions` `[3, 32]`, `[4, 64]` | none | planned |
| 4 | Upsampling | `transposed`, `bilinear` | none | planned |
| 5 | One mask head shared by the levels (`suggestions.md` item 7) | per level, shared | `MaskHead.shared` | planned |
| 6 | Mask level chosen by the box size (`suggestions.md` item 2) | level of the proposal, level of the box size | level rule | planned |
| 7 | A finer map for the masks (`suggestions.md` item 5) | the levels of the detection, a level 3 of the feature pyramid (stride 8) for the masks | neck level | planned |
| 8 | Mask scoring (Mask Scoring R-CNN) | class score, class score times the predicted mask IoU | mask IoU head | planned |

Stages 0 to 4 change settings only; stages 5 to 8 need code, merged right before their pair. Every training pair is expected to take at least the 14 hours of a detection pair, the mask heads and targets adding a share that stage 0 measures. On 2026-11-12 the user kept every stage of the table (stage 4 trains both upsamplings, the transposed convolution as the reference), the pools of 18 and 22 for stage 1, and the launch of stage 0 as soon as stage 8 of the detection benchmark ends.

## The Stages

### Stage 0: Reference And Preflight

Before any comparison, the reference pair (run_1 and run_2) trains with the starting values, and the launch is checked first:

- 40 iterations alone, then two launches side by side: images per second, peak GPU memory and the free memory of the machine, which must stay above 25 GB (the trainings slow down two to three times below it), the 3.8 GB `labels.json` being loaded by every launch;
- the two evaluations of one checkpoint (trained and averaged weights, 10,000 images each, every mask pasted at 1024 pixels): if they take much more than the 8.3 minutes of a detection checkpoint, the evaluation runs every 1000 iterations instead of 500;
- the anchor recall of the dataset (`anchor_recall.csv`, written at the launch), against that of the detection dataset.

The box mAP of the pair can be set beside that of the detection reference (run_21 and run_22, 0.8183), as an indication only: the images and the boxes (those of the visible parts) differ.

### Stage 0b: Mask Threshold And Final NMS (evaluation only)

Since `MaskHead.inference.mask_threshold` and the final NMS act in evaluation only, `scripts/evaluation/evaluate_checkpoint.py` evaluates the 15,000 checkpoints of run_1 and run_2 again, as for detection stage 5b, without training:

- `mask_threshold` 0.4 and 0.6 against 0.5, the probability from which a pixel of a pasted mask belongs to its object;
- the greedy NMS against DIoU-NMS 0.6, which the detection benchmark kept on boxes (+0.0034 on its occlusion split) but never measured on masks.

### Stage 1: Mask RoIAlign Pool Size

At the user's request ("increasing the size of the 11x11 feature map for segmentation trying out 15x15 and 19x19"), the RoIAlign grid of the mask heads grows from its current side. Every instance segmentation configuration pools 14 x 14 (`DetectionLevels.<level>.mask_head.roi_align_pool_size: 14`, the size of Mask R-CNN), and no setting of the repository holds 11, so on 2026-11-12 the user chose the same steps from 14: pools of 18 and 22 against 14.

The pool size is tied to `MaskHead.mask_resolution` by `count_mask_upsamplings` (`mask_head.py`): the pooled features are doubled only while twice their side fits in the mask resolution. At a resolution of 28, a pool of 18 or 22 would get no doubling, its 18 x 18 or 22 x 22 logits being resized bilinearly to 28 and the learned transposed convolution lost. Every pool therefore trains with a mask resolution of twice its side (36 for 18, 44 for 22, the user's choice), keeping its one learned doubling, the mask targets following the resolution (`build_mask_targets`), the run name its `M<pool>` and `MR<resolution>` tokens.

Both pools are trained, 22 first, the larger change, then 18, and the best by the rule of the strategy is kept. The cost grows with the square of the side: the mask head convolutions, and the logits of the 35 classes of up to 4096 positive proposals per batch (64 images, 64 positives each), 0.45 GB in float32 at 28 x 28, 0.74 GB at 36 x 36 and 1.11 GB at 44 x 44, before their gradients.

A larger grid may bring little: RoIAlign samples the feature map of the level of the proposal, and the square anchors span one to four strides on levels 4 to 6 (16 to 256 pixels at strides 16 to 64) and six to twelve on level 7, so the box of an object covers a few cells of its map, rarely as many as the 14 of the grid that already samples it. Mask R-CNN assigns a box of 224 pixels to its stride 16 level (the level rule of the feature pyramid paper), where it covers 14 cells, the side of its grid. Stages 6 and 7 act on that limit.

### Stage 2: Mask Resolution

With the pool kept by stage 1, a second doubling (`mask_resolution` four times the pool, 56 for 14) gives the large objects finer borders: a 28 x 28 mask pasted on an object of 800 pixels is stretched about 29 times. The pooled features stay the same, so a gain would show on the large objects and at the high IoU thresholds (mask mAP@0.75 to @0.95). The logits take four times the memory of one doubling (1.8 GB in float32 at 56 x 56 for 4096 proposals, 4.4 GB at 88 x 88 after a pool of 22), measured before the pair.

### Stage 3: Mask Head Size

To find out whether small heads limit the masks, a larger head is trained: the mask heads of the synthetic configuration have three convolutions of 32 channels, those of `configuration_example.yaml` four of 64, and those of Mask R-CNN four of 256 (Detectron2 `configs/Base-RCNN-FPN.yaml`: `ROI_MASK_HEAD.NUM_CONV: 4`, `POOLER_RESOLUTION: 14`, with `CONV_DIM = 256` of `detectron2/config/defaults.py`). The stage trains `[4, 64]` against `[3, 32]`.

### Stage 4: Upsampling

In case the transposed convolution marks the borders, the stage trains `MaskHead.upsampling: "bilinear"` (a bilinear doubling followed by a 3 x 3 convolution) against the transposed convolution of Mask R-CNN, whose 2 x 2 kernel of stride 2 can leave a checkerboard pattern on the borders. A tie is likely, but the user kept the stage ("we try transposed and bilinear upsampling right ?").

### Stage 5: One Mask Head Shared By The Levels

Whenever a level holds few positives, its own mask head learns from few examples (`suggestions.md` item 7): with the feature pyramid, every level already has 32 channels, so a single mask head can read every level and train on every positive of the batch, where each per-level head sees only the positives of its level (`compute_proposal_mask_loss`), and the export runs that single head once on all the detections instead of every head on every detection. Code: a `MaskHead.shared` switch, its run name token, and a transfer that refuses to load per-level heads into a shared one.

### Stage 6: Mask Level Chosen By The Box Size

For the medium and large boxes to be pooled from enough cells, the level can follow the box size (`suggestions.md` item 2): every mask is pooled from the level of its proposal, whose anchors match the box, so a box covers a few cells of its map (stage 1). The rule of the feature pyramid paper, level k = k0 + log2(sqrt(w h) / 224) with k0 = 4, pools every box from the level where it covers about 14 cells: the finest level (stride 16) for the boxes up to about 224 pixels, level 5 (stride 32) around 448 pixels. A box of 128 to 256 pixels, now pooled from 2 to 4 cells of level 6, would get 8 to 16 cells of level 4. It works best with the shared head of stage 5, every head otherwise seeing boxes of other sizes than before. Code: the level rule in the mask loss, the mask prediction and the export, and a setting to choose it.

### Stage 7: A Finer Map For The Masks

If one wishes to give the small objects finer masks, a finer map is needed (`suggestions.md` item 5): the finest level of the detection has a stride of 16, where Mask R-CNN pools its small boxes from a stride of 4. The feature pyramid can add a level 3 (stride 8, block 3 of the backbone) read by the mask heads only, the detection keeping its levels and anchors. The small objects, 1 to 2 cells wide on level 4, would get 2 to 4 cells. Code: a neck level outside `DetectionLevels`, the mask head of that level, its memory and time measured before the pair.

### Stage 8: Mask Scoring

Suppose a detection with a sure class has a poor mask: the score of a mask is the class score of its box, whatever the quality of the mask. Mask Scoring R-CNN (Huang et al., CVPR 2019, https://arxiv.org/abs/1903.00241) adds a head predicting the IoU of every mask with its object and scores the mask with the class score times that IoU, so that the better masks rank first in the mask AP; its abstract reports a "consistent and noticeable gain" on COCO. Code: the mask IoU head, its loss against the IoU of the predicted and target masks, the score in the evaluation and the export.

### Not Planned

Should the time allow more stages later, these were considered and set aside:

- A boundary refinement in the style of PointRend (https://arxiv.org/abs/1912.08193), whose abstract reports crisp borders and significant gains on COCO, the largest code of all, kept for after stages 2 and 7.
- Ground truth masks cropped to their boxes (`suggestions.md` item 8): the instance index maps already cut the masks through the loader to a ninth on this dataset.
- Detections clipped to the image window (`suggestions.md` item 9): the images of the dataset are square, so it changes nothing here.
- The normalization of the mask heads: detection stage 6 kept batch normalization for its stability.
