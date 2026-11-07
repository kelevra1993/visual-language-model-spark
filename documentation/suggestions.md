# Suggestions

> Proposed improvements of the object detection (DET) and instance segmentation (INS) tasks that are not implemented
> yet, kept in `documentation/` so that we can come back to them and decide what to implement. They were checked
> against the code on 2026-10-15, together with the findings of the
> [Repository Audit For Instance Segmentation](#repository-audit-for-instance-segmentation), which they do not repeat.
> The repository audit is kept at the end of this file. The items
> taken from Mask R-CNN come from a comparison with its Keras / TensorFlow 1 implementation (matterport, commit
> `3deaec5`) written on 2026-10-12 and merged here: "Mask R-CNN" means that repository, and "the paper" He et al.,
> 2017. Its file references (`model.py`, `utils.py`, `config.py`, `coco.py`, `nucleus.py`, `visualize.py`) are those of
> `mrcnn/` and of its samples. An item is removed from this list once it is implemented, or as soon as we decide not
> to do it.
>
> The items implemented since the first version of this list were removed:
> - mAP evaluation, checkpoint evaluation script, run names, dataset sizes at startup, README;
> - split of the detector and loader code;
> - AdamW / warm-up / cosine / clipping, mixed precision;
> - He initialisation (removed after stage 12 of the experiment), pre-activation residual pairs;
> - moving average of the weights, its export, best value and summary columns;
> - confusion matrix, class thresholds and error images of the detector;
> - batched target assignment, sampling and proposal selection of the training, exact box overlap;
> - crowd annotations kept as ordinary objects by DET and INS, which no longer leave them out;
> - record of the images of every split (`data.json`);
> - configurable mask heads;
> - covered objects left out of the synthetic DET images;
> - average precision by object size;
> - recall of the anchors at launch and of the proposals at every evaluation;
> - object fragments left by the crop and the rotation removed (25 % of their area);
> - per-level settings grouped in `DetectionLevels`;
> - ground truth boxes added to the proposals of the detection heads;
> - instance segmentation boxes taken from the masks, without the annotation `bbox`;
> - detector with fewer anchors than proposals or candidates than detections, in training and in the exports;
> - zoom-out augmentation (`random_zoom_out`), on in the DET and INS `configuration_example.yaml`;
> - translation augmentation (`random_translation`) along the width, the height or both, up to 25 % of each side, for
>   every task, off in every configuration file;
> - region proposal negatives down to IoU 0 (audit H1), the check warning when `background_iou_threshold.min` is
>   above 0;
> - NMS at 0.7 for the region proposals and 0.5 for the detections (audit M4);
> - box regression weights of both stages kept at (1, 1, 1, 1) (`BOX_REGRESSION_WEIGHTS`),
>   the (10, 10, 5, 5) of torchvision giving about 2.5 times lower box and mask mAP@0.5 on the synthetic shapes;
> - INS evaluation at the original resolution of the test images (`evaluation.original_resolution`) and mean
>   average recall of the boxes and masks by object size, for DET and INS (item 12), not compared with pycocotools;
> - a weight per loss of DET and INS in the total loss (`Optimization.loss_weights`), tested: the weights of 1
>   everywhere gave the best results and are kept (2026-11-05);
> - the decorrelation of the hidden features of the classification head (`ClassificationHead.decorrelation`, item 18,
>   whose experiment is left);
> - the keypoint candidates placed within their cell by the parabola of the log-heatmap (item 17, whose measure on
>   COCO is left);
> - the objects without labelled keypoint left out of the keypoint heatmap loss
>   (`KeypointHead.ignore_unlabelled_objects`, item 16, whose experiment is left);
> - the ImageNet-22k download (`imagenet_22k_download.py`, which replaced the Tencent ML-Images downloader) and its
>   merge with ImageNet-1k (`merge_imagenet_manifests.py`, `~/Data/classifier_data/imagenet-merged`), on 2026-11-06;
> - the neck of DET and INS (`Neck.type`: `feature_pyramid`, `path_aggregation`, items 1 and 37, `none` in every
>   configuration), on 2026-11-07, whose stages 3 and 4 are left to run.
>
> Dropped without being implemented:
> - the masked proposal padding of the training;
> - the partial freezing of a pretrained backbone;
> - `torch.compile`, after a measurement on the GB10 showed no speed-up;
> - the predictions file of the detector;
> - the dropout and the global average pooling of the detection head;
> - the watch for the CUDA crash seen with DALI;
> - the crowd regions ignored by the targets and the evaluation (audit M12), the crowds being ordinary objects;
> - the COCO matching of the mAP (audit L6), the VOC matching being kept;
> - the box-aware mosaic / copy-paste augmentation, the tiles at native resolution for DET and INS, and the projection
>   shortcuts of the channel-changing residual blocks (audit H4), dropped on 2026-11-05.
>
> Tested and reverted:
> - the region proposal negatives mixing a share of easy anchors below `background_iou_threshold.min` with the hard
>   ones above it (`RegionProposal.easy_negative_fraction`, c3068e3): tested in stage 2b of
>   `architecture_benchmarking/object_detection_experiment_plan.md` (share 0.25, run_7 and run_8 against run_3 and
>   run_4), not conclusive, 0.25 being 0.0190 behind at 12,500 with boxes about half as tight at mAP@0.95, so the code
>   was reverted on 2026-11-07.

## Table Of Content
- [A. Detector architecture](#a-detector-architecture)
- [B. Instance segmentation](#b-instance-segmentation)
- [C. Evaluation and deployment](#c-evaluation-and-deployment)
- [E. Mask R-CNN on the open audit items](#e-mask-r-cnn-on-the-open-audit-items)
- [F. Reference: Mask R-CNN compared stage by stage](#f-reference-mask-r-cnn-compared-stage-by-stage)
- [G. Keypoint detection](#g-keypoint-detection)
- [H. Classification](#h-classification)
- [I. Selection of training images by embedding clustering](#i-selection-of-training-images-by-embedding-clustering)
- [J. Self-supervised counted shapes: tests left to run](#j-self-supervised-counted-shapes-tests-left-to-run)
- [K. README screenshots and synthetic runs: left to redo](#k-readme-screenshots-and-synthetic-runs-left-to-redo)
- [Suggested Order](#suggested-order)
- [Repository Audit For Instance Segmentation](#repository-audit-for-instance-segmentation)
  - [Summary By Task](#summary-by-task)
  - [Critical](#critical)
  - [High](#high)
  - [Medium](#medium)
  - [Low](#low)
  - [Statements Contradicting The Code](#statements-contradicting-the-code)
  - [Checked And Correct](#checked-and-correct)
  - [Suggested Order Of Fixes](#suggested-order-of-fixes)

## A. Detector architecture
1. **A top-down feature pyramid (FPN), implemented, stage 3 left to run** (`Neck.type: "feature_pyramid"`, 1x1
   laterals to `Neck.output_channels`, nearest upsampling and addition, the 16 x 16 blocks 6 and 7 added without
   resampling, then a 3x3 convolution; the heads stay one per level): lateral 1×1 convolutions and upsampling give
   the finest detection level the context of the deeper ones. All levels then share the same channel count, and
   could share one detection head.
   - **Mask R-CNN's recipe** (`model.py:1902-1925`): 1×1 laterals to 256 channels on C2–C5, nearest ×2 upsampling and
     addition top-down, and a 3×3 convolution on every sum; P6, made by subsampling P5, serves the region proposal
     network only.
   - **Two adaptations here:** the 1×1 laterals would also bring the levels of different channel counts (64, 128, 128
     in the examples) to a common count, and blocks 6 and 7 share the same 16×16 resolution at 1024 pixels, so they
     would be added without upsampling, or only one of them kept.
   - What matters most for the masks is a finer level, such as block 3 or 4 (stride 8 or 16) at 1024 pixels, which
     gets the context of the deeper blocks through the top-down path (item 5).
2. **Choosing the detection level by box size** (the FPN rule `k = k₀ + log₂(√(wh)/224)`), instead of the level of
   the anchor the proposal came from.
   - **Mask R-CNN** uses `k = 4 + round(log2(√(wh) / 224))`, clamped to [2, 5], with w and h in network input pixels
     (`model.py:380-393`). The rule chooses the level of the box head and of the mask head: in training on the
     sampled proposals, at inference on the final refined boxes, so the mask of a detection is pooled from the level
     matching its final size, not the size of its anchor.
   - With levels at strides 32 / 64 / 64, the reference size (224 at P4, stride 16) has to be re-derived, for example
     from the middle anchor scale of every level.
   - The ground truth boxes added to the training proposals take the level of their best anchor
     (`add_ground_truth_proposals`); they would follow the same rule.
   - The export already runs the mask head of every level on every detection and selects per detection with
     `torch.where` (`inference_instance_segmenter.py`). Only the selector would change.
3. **NMS per level in the proposal filter**: `batched_nms` with the level as the group, so that one strong level does
   not suppress the proposals of the others (audit M3). Mask R-CNN does not do this either: it runs one top-k and one
   NMS over all levels (`model.py:287-328`). torchvision and Detectron2 (top-k per level, NMS grouped by level) are
   the reference.
36. **A box loss on the overlap (CIoU)**, scheduled in `object_detection_experiment_plan.md`:
    - **Why:** the smooth L1 on the offsets gives the same loss to predictions of very different overlap and shrinks
      with the size of the boxes, while 1 - IoU is flat for every box that does not overlap. GIoU adds the empty area of
      the smallest enclosing box, but falls back to the IoU when one box contains the other; DIoU adds the squared
      distance between the centres over the squared diagonal of the enclosing box (faster convergence); CIoU (the loss
      of YOLOv4) adds a term on the difference of aspect ratios.
    - **Here:** the region proposal and detection losses are smooth L1 on the offsets (`LOCALISATION_LOSS_BETA` 1/9),
      while the synthetic hard shapes reach mAP@0.5 0.97 but mAP@0.75 0.93 and mAP@0.95 0.05: the boxes are found but
      not tight. torchvision 0.29 has `complete_box_iou_loss`, `distance_box_iou_loss` and `generalized_box_iou_loss`.
    - **Change:** decode the predicted offsets of the sampled positives into boxes (`apply_regression_predictions`)
      and take 1 - CIoU against their target boxes, for both stages, the smooth L1 staying the other value of the
      setting during the stage.
37. **Path aggregation after the feature pyramid (PANet), implemented, stage 4 left to run**
    (`Neck.type: "path_aggregation"`, with item 1, the bottom-up convolutions followed by a ReLU), scheduled in
    `object_detection_experiment_plan.md`:
    - **Why:** the top-down path of the FPN gives the fine levels the context of the deep ones, but the localisation
      detail of the fine levels reaches the coarse ones only through the whole backbone. Path aggregation adds a
      bottom-up path after the top-down one: every finer output, downsampled by a strided 3x3 convolution, is merged
      into the next coarser level, and the heads read the outputs of that path (the neck of YOLOv4, which concatenates
      where PANet adds).
    - **Here:** one neck between the backbone blocks of `DetectionLevels` (4 to 7) and the heads: 1x1 laterals to a
      common channel count, the top-down path, then optionally the bottom-up one. Blocks 6 and 7 have the same 16 x 16
      resolution at 1024 pixels, so they are merged without resampling. The neck changes the run name, the export and
      the transfer of pretrained components.
38. **DIoU non-maximum suppression**: suppress a box by its IoU minus the centre distance term of DIoU with the kept
    box, so that two distinct objects overlapping a lot but with distant centres are not merged (occlusions). For
    the final detections, and possibly the proposals.
39. **Spatial pyramid pooling (SPP) after the backbone**: the last feature map concatenated with its max pools of 5, 9
    and 13, widening the receptive field of the deepest level at little cost (YOLOv4).

## B. Instance segmentation
5. **Sharper masks for small objects**: the mask heads pool every box from the backbone feature map of its level
   (RoIAlign 14 × 14, upsampled to 28 × 28), never from the image, so a small object on a coarse level (blocks 5–7
   of the examples, strides 32, 64 and 64 at 1024 pixels) is interpolated from a handful of feature cells and gets
   blurry borders: a 32-pixel object, the smallest anchor, spans one feature cell of its level. Mask R-CNN pools from
   P2–P5, at strides 4 to 32, with the context of the FPN. In order of cost:
   - an earlier, finer level in `DetectionLevels` (more anchors and heads to train);
   - the FPN of item 1, whose finest level gets the context of the deeper ones (the stride 4 level of Mask R-CNN);
   - a mask head reading a finer block than its detection level, e.g. the box pooled from block 4 for the masks
     while the detection head reads block 5;
   - larger masks (`MaskHead.mask_resolution` 56, a second doubling of the pooled features) or a larger pool size on
     the coarse levels (`DetectionLevels.<level>.mask_head.roi_align_pool_size`), useful only once the pooled
     features themselves are fine enough;
   - a boundary refinement in the style of PointRend, predicting the uncertain border points again from finer
     features.
7. **One mask head shared by the levels**:
   - **Mask R-CNN:** `build_fpn_mask_graph` is built once (`model.py:956-1005`) and receives the RoIs of every level
     through `PyramidROIAlign`. The box head and the region proposal head are shared the same way
     (`model.py:1939-1944`).
   - **Here:** there is one `MaskHead` per level, and `compute_proposal_mask_loss` splits the positives by level, so
     each head trains on a fraction of the positives of the batch. The export runs the mask head of every level on
     every detection, which triples the mask computation for three levels.
   - **Change:** a `MaskHead.shared` switch. When shared, a per-level 1×1 convolution with normalization brings every
     level to the input channel count of one `MaskHead`, which then sees every positive, and the export runs that
     single head once on all the detections. Without the FPN (item 1), the levels still differ in depth and context,
     so this works best together with item 1 or after it. The detection heads could be shared the same way.
   - **Cost and risk:** medium: a run name token, and the model transfer must refuse to load per-level heads into a
     shared one. A shared head may fit the coarse levels worse than dedicated heads do, which is the point of
     measuring it.
   - **Measure:** mask mAP overall and by object size, and the export throughput.
8. **Ground truth masks cropped to their boxes ("mini masks")**:
   - **Mask R-CNN:** `USE_MINI_MASK` and `MINI_MASK_SHAPE` (56 × 56) (`config.py:103-104`): each mask is cropped to its
     box and resized (`minimize_mask`, `utils.py:517-534`), and the mask target of a RoI is cropped from that small
     mask, after expressing the RoI in the coordinates of the ground truth box (`model.py:584-600`).
   - **Here:** one `[N, S, S]` uint8 mask per object goes through the loader workers to the GPU. Audit L19 estimates
     about 4.7 GiB in flight at 1024 pixels, batch 40, 8 workers and prefetch 2.
   - **Change:** an option to load the masks as `[N, m, m]` crops with their boxes (`m` configurable, 56 in Mask
     R-CNN, 112 for large objects). `build_mask_targets` would crop with RoIAlign in the coordinates of the crop (box
     offset and scale). The flips, the crop and the zoom-out act exactly on a box-relative crop; the ±5° rotation
     needs the crop resampled in the rotated box. The evaluation and the sample images paste the crops back with
     `paste_instance_masks`. The other fix that L19 suggests (sending polygons or RLE and rasterising on the device)
     keeps the full precision.
   - **Cost and risk:** medium to high (dataset, augmentations, targets, evaluation), with a small loss of precision
     for large objects. Only worth doing if memory or loader throughput becomes the limit.
9. **Detections clipped to the image window before the mask head**:
   - **Mask R-CNN:** `refine_detections_graph` clips the refined boxes to the image window, the letterbox without its
     padding (`model.py:711`), and the mask head then runs on those boxes (`model.py:2043-2052`).
   - **Here:** `postprocess_detections` clamps the boxes to the whole input, padding included. With `keep_ratio`, a
     box can reach into the padding: its mask is predicted over that box, and the box is clipped again when restored
     to the original image, which squeezes the mask (audit L22). In the network-resolution evaluation, such boxes also
     lose IoU.
   - **Change:** clamp to the letterbox window (`compute_letterbox_region`, `preprocessing.py`) before the mask head;
     the trainer knows the original sizes from `labels.json`. The exported model would need the windows as a second
     input (`[B, 4]`); otherwise the clients keep the L22 fix, pasting with the unclipped boxes.
   - **Cost and risk:** small in the trainer, changing the export signature being the expensive part. A small effect,
     limited to non-square images.
10. **Datasets annotating different class subsets**:
    - **Mask R-CNN:** `active_class_ids` (`model.py:1076-1109, 1271-1276`). When the images come from several datasets
      that each annotate a subset of the classes, the classification loss ignores the predictions of the classes that
      the source of the image does not annotate.
    - **Here:** one label mapping for every image.
    - **Change:** only if INS is ever trained on merged datasets: a per-image list of annotated categories (a COCO
      extension, or one per source folder), with the classification loss masked accordingly. Medium cost, low priority
      until such a dataset exists.

## C. Evaluation and deployment
11. **Per-class score thresholds in the exported detector**: the evaluation finds the score threshold of every class
    reaching `evaluation.target_precision` (`class_thresholds.json`), but `InferenceDetector` applies the single
    `Detector.inference.score_threshold` to every class, whatever its precision at that score. `export_model.py`
    could read the thresholds of the exported checkpoint, as `frame_selection_deployment.py` does for SEL, and
    `postprocess_detections` could take one threshold per class, a class that never reaches the target precision
    keeping the global one. The confusion matrix, filled batch by batch before the thresholds are found, could be
    built again from the kept detections with them, so that it keeps describing the deployed detector.
12. **Visual checks of the training targets and of the matches**:
    - **Mask R-CNN:** `inspect_model.ipynb` walks through the region proposal targets (positive anchors with their
      refinement), the proposals before and after NMS, the sampled RoIs with their positive ratio (`draw_rois`,
      `visualize.py:208`), the refinement, the per-class NMS, and the 28 × 28 mask targets next to the predicted masks.
      `display_differences` (`visualize.py:170-205`) draws the ground truth (green) and the detections (red) on one
      image, each detection captioned "score / IoU", and `plot_overlaps` (`visualize.py:320`) the grid of IoUs between
      ground truths and detections.
    - **Here:** `view_input_images` shows the augmented inputs with their boxes and masks. The sample grids show the
      ground truth next to the detections, without the IoU of each detection. The detection debugging scripts were
      removed in `6225b4a`; `visualise_anchors` remains.
    - **Change:** (a) a `diagnosis_modes.view_training_targets` flag, shown like `view_input_images` on a training
      batch, drawing the positive anchors and a sample of the negatives, the top proposals after the filter, the
      sampled RoIs (the added ground truth boxes included) with their matched ground truth, and for a few positive
      RoIs the mask target next to the predicted mask, generalising `view_input_data` and reusing `visualise_anchors`.
      This makes failures such as audit H1 (proposals on plain background) or M5 (shifted mask targets) visible at a
      glance. (b) In `build_detection_sample_grid`, caption every detection with its score and its IoU to the matched
      ground truth.
    - **Cost and risk:** (b) is small, (a) medium. No number changes.

## E. Mask R-CNN on the open audit items
What Mask R-CNN does for the items of the [repository audit](#repository-audit-for-instance-segmentation) that are still open:
- **M10 (weight decay):** Mask R-CNN skips the BN scales and offsets but still decays the biases
  (`model.py:2179-2183`). It also divides each L2 term by the tensor size, an unusual scaling not to copy.
- **L1 (anchor centres):** Mask R-CNN has the same corner offset (`utils.py:603-604`), so it is no argument against
  the fix.
- **L5 (box loss normalisation):** Mask R-CNN averages the smooth L1 over the coordinates of the positives
  (`model.py:1070-1073, 1135-1139`). That is per coordinate, a quarter of a per-box sum, on targets scaled by 5 to
  10 as here.
- **L7 (zero-area boxes):** Mask R-CNN removes the zero-area detections when restoring them to the image
  (`model.py:2460-2469`). It keeps the small proposals on purpose, citing a loss on small objects
  (`model.py:313-315`).
- **L19 (full-resolution masks through the loaders):** Mask R-CNN's answer is the mini masks (item 8).
- **L22 (masks squeezed at the border):** Mask R-CNN clips the refined boxes to the image window before the mask head
  runs on them (`model.py:711, 2043-2052`), so the mask is predicted on the box that is later pasted (item 9).

## F. Reference: Mask R-CNN compared stage by stage
**The mask branch is the same design.** Both use RoIAlign 14 × 14, a stack of 3 × 3 convolutions, a 2 × 2 transposed
convolution to 28 × 28 and one sigmoid mask per class. The loss is a binary cross entropy on the channel of the ground
truth class, over the positive proposals only, and at inference the masks are predicted on the final refined boxes. The
differences lie in what feeds the mask branch (resolution and context of the levels, items 1, 2 and 5) and in how the
heads are shared (item 7).

**Where this repository does more than Mask R-CNN:** the export (fixed-shape TorchScript / ONNX), the mask pasting
(bilinear sampling at pixel centres, instead of resizing to an integer box then thresholding), RoIAlign (aligned,
adaptive sampling, instead of one bilinear sample per bin), the detection candidates (every class of every RoI,
instead of the argmax class only), the GPU augmentations, and the evaluation at every checkpoint with the AP and the
average recall by object size and the recall of the anchors and of the proposals.

| Stage | Mask R-CNN | This repository |
|---|---|---|
| Input | Resize so the short side reaches 800 (up only) and the long side stays ≤ 1024, then pad to 1024 × 1024 centred. Modes `crop` (random crops, training) and `pad64` (native size, inference) (`utils.py:388-492`). RGB minus the mean pixel. | Fixed square input, letterboxed centred (`keep_ratio`) or stretched. BGR in [0, 255], then `input_normalization` / `per_sample_normalization`. |
| Backbone | ResNet-50/101 bottleneck stages C2–C5, pretrained on ImageNet or COCO (`model.py:171-206`). | Stack of `ConvolutionBlock`s (residual, SE, inception options). The detection levels are the outputs of the blocks numbered by the keys of `DetectionLevels`, after their pooling. Pretrained from another run of any task. |
| Feature pyramid | FPN, P6 for the region proposals only (`model.py:1902-1925`). | `Neck` (items 1 and 37): none by default, the levels keeping the channels of their blocks, or a feature pyramid or path aggregation to `output_channels`, without P6. |
| Anchors | One scale per level (32 on P2 to 512 on P6), ratios 0.5 / 1 / 2, corner-aligned (`utils.py:583-619`). | Several scales and ratios per level (`DetectionLevels.<level>.anchors`), corner-aligned (audit L1). |
| Region proposal head | One head shared by every level: 3 × 3 conv of 512, then 1 × 1 convs for a 2-class softmax and the offsets (`model.py:830-871`). | One head per level (`DetectionLevels.<level>.region_proposal`): `ConvolutionBlock`, 3 × 3 conv, then 1 × 1 convs for a sigmoid objectness and the offsets. |
| Region proposal targets | Positive at IoU ≥ 0.7, plus the best anchor(s) of every ground truth. Negative below 0.3. Anchors touching a crowd are never negatives. 256 sampled, at most half positive (`model.py:1445-1553`). | Positive at IoU ≥ 0.5, plus the best anchor(s). Negative below 0.3, 0 included. 256 sampled, at most 128 positive. |
| Proposal filter | One top-k (6000) and one NMS (0.7) over all levels, 2000 kept in training and 1000 at inference, padded with zeros (`model.py:255-329`). | One top-k (1000 / 500) and one NMS (0.7) over all levels (audit M3), 500 / 250 kept, padded with copies of random survivors in training and evaluation (masked slots in deployment), at most one per anchor. |
| RoI sampling | 200 RoIs, 33 % positive (IoU ≥ 0.5). Negatives below 0.5 that touch no crowd, in proportion to the positives, so none without positives (`model.py:486-619`). The ground truth boxes are left out on purpose, a comment citing a sampling study that found no gain (`model.py:1317-1318`). | The ground truth boxes of every image are added to its proposals at the level of their best anchor (`add_ground_truth_proposals`, as in torchvision and Detectron2). 128 RoIs, up to 64 positive (IoU ≥ 0.5), the negatives filling the rest of the budget. |
| RoI pooling | `PyramidROIAlign`: level `4 + round(log2(√(wh) / 224))`, clamped to P2–P5, then `crop_and_resize`, one bilinear sample per bin (`model.py:344-447`). | `roi_align` with `aligned=True` and adaptive sampling, on the level of the anchor that produced the proposal (item 2). |
| Box head | Shared: 7 × 7 conv to 1024 + BN, 1 × 1 conv 1024 + BN, then class and per-class box layers (`model.py:900-953`). | Per level (`DetectionLevels.<level>.detector`): `ConvolutionBlock`, flattened, fully connected layers without normalization, then class and per-class box layers. |
| Box regression | Targets divided by [0.1, 0.1, 0.2, 0.2] on both stages. Smooth L1 with β = 1, averaged over the coordinates of the positives (`model.py:575, 1047-1073, 1112-1140`). | Targets left unscaled, fixed weights (1, 1, 1, 1) on both stages (`BOX_REGRESSION_WEIGHTS`), (10, 10, 5, 5) giving about 2.5 times lower mAP@0.5 on the synthetic shapes. Smooth L1 with β = 1/9, summed per box and divided by the positives (audit L5). |
| Mask head | Shared: 4 × (3 × 3 conv 256 + BN + ReLU), 2 × 2 transposed conv 256, 1 × 1 conv to one sigmoid per class (`model.py:956-1005`). | Per level (`DetectionLevels.<level>.mask_head`, item 7): `ConvolutionBlock` of `[layers, channels]` with module options, configurable pool size, mask resolution and upsampling, 1 × 1 conv per class. |
| Mask targets | `crop_and_resize` of the matched mask (full or mini) inside the positive RoI, rounded at 0.5 (`model.py:577-606`). | `roi_align` (aligned) of the matched full-resolution mask inside the positive proposal, thresholded at 0.5. |
| Mask loss | BCE of the ground truth class channel, mean over the positives (`model.py:1143-1179`). | Same, computed in float32. |
| Postprocessing | Class = argmax of each RoI. Score ≥ 0.7, per-class NMS 0.3, top 100. Boxes clipped to the image window, padding excluded (`model.py:684-779`). | Every foreground class of every RoI is a candidate. Score ≥ 0.5 (0.05 in evaluation), batched per-class NMS 0.5, fixed top 100, boxes clamped to the whole input, padding included (item 9). |
| Masks at inference | Mask head on the final boxes. Each 28 × 28 mask is resized to its integer box in the original image, then thresholded at 0.5 (`model.py:2417-2480`, `utils.py:560-576`). | Mask head on the final boxes. `paste_instance_masks` samples at pixel centres with sub-pixel boxes. The INS evaluation pastes them at the original resolution with `evaluation.original_resolution` (at the network resolution otherwise), and the clients paste at the original resolution. |
| Ground truth boxes | Always the tight box of the mask, the dataset boxes being ignored (`model.py:1269`). | The same: the tight box of the mask at the network resolution, the INS annotations needing no `bbox`. |
| Ground truth masks in memory | Optional 56 × 56 crops in the box (`config.py:103-104`, `utils.py:517-552`). | One `[N, S, S]` uint8 mask per object at the network resolution (item 8, audit L19). |
| Crowds | Kept with a negative class id, kept out of the negatives, ignored by pycocotools (audit M12). | Ordinary objects, learnt and scored like the others. |
| Objects lost by the resizing | Skipped at original resolution when under 1 pixel (`coco.py:248-250`). Masks emptied by the resize are dropped from the training targets (`model.py:1263-1265`). The evaluation keeps them, since it runs on the original images. | INS masks emptied by the resize are dropped from the training and validation targets, and from the test ground truth unless `evaluation.original_resolution` keeps them, so that they count as missed. |
| Augmentation | imgaug pipeline with a list of mask-safe augmenters (flips, crop-and-pad, Affine, PiecewiseAffine; `model.py:1232-1259`). COCO uses horizontal flips only, nucleus flips, 90° rotations, brightness and blur. | `DataAugmenter` on the GPU: crop, aspect crop, zoom-out, flips, ±5° and 90° rotations, twelve photometric types, masks moved and boxes recomputed, fragments of objects removed. |
| Optimisation | SGD momentum 0.9, learning rate 0.001. Each gradient tensor clipped at norm 5 (Keras `clipnorm`). L2 1e-4 except the BN scales and offsets. Weight per loss (`config.py:177-211`, `model.py:2153-2199`). | Adam / AdamW / SGD Nesterov, warm-up, cosine, optional global-norm clipping. One weight decay for every parameter (audit M10). One weight per loss (`Optimization.loss_weights`), 1 in every configuration file (item 6). |
| Training schedule | Stages selected by layer regular expressions: `heads`, `3+`, `4+`, `5+`, `all` (`model.py:2313-2322`, `coco.py:498-521`). | One stage per run. A loaded component can be frozen whole or given a learning rate multiplier; a schedule needs successive runs. |
| Batch normalization | `TRAIN_BN: False` by default: the stored statistics are used and never updated (batches of 2). | Trained, with a configurable momentum. `layer` normalization is available for small batches. |
| Evaluation | `compute_ap` with mask IoU and COCO matching (`utils.py:656-751`). COCO sample: pycocotools `COCOeval` for boxes and masks at the original resolution (`coco.py:342-385`). | Own mAP (area, 11 and 101 points) at 0.5, 0.75 and 0.5:0.95 with VOC matching (kept, audit L6), mean average recall, both by object size, box and mask results per class, proposal recall, at every checkpoint, at the network resolution or, for INS, at the original resolution (`evaluation.original_resolution`). |
| Visual checks | Notebooks walking through every stage, and `display_differences` (item 12). | `view_input_images`, sample grids and the task workflow notes (item 12). |
| Deployment | Keras weights only, batch size fixed at build time. | TorchScript / ONNX exports with fixed shapes, served by Triton, with a comparison check. |

Default hyperparameters of Mask R-CNN (`config.py`) against `instance_segmentation/configuration_example.yaml` (the
synthetic shapes configuration uses the same values at 512 pixels, with smaller heads and a batch of 64):

| Setting | Mask R-CNN | INS example | Note |
|---|---|---|---|
| Input | 800–1024, padded to 1024 | 1024 letterboxed | |
| Strides of the head levels | 4, 8, 16, 32 (region proposals also 64) | 32, 64, 64 | Item 5 |
| Anchor scales | 32, 64, 128, 256, 512 (one per level) | 32–128, 64–256, 128–512 | |
| Region proposal positive IoU | ≥ 0.7 | ≥ 0.5 | |
| Region proposal negative IoU | < 0.3 | < 0.3 | Same |
| Anchors sampled | 256, ≤ 50 % positive | 256, ≤ 128 positive | Same |
| Proposals before NMS | 6000 | 1000 / 500 | |
| Region proposal NMS | 0.7 | 0.7 | Same |
| Proposals after NMS | 2000 / 1000 | 500 / 250 | |
| RoIs sampled per image | 200, 33 % positive, without the ground truth boxes | 128, ≤ 64 positive, with the ground truth boxes | The paper: 512, 25 % |
| Box head | Shared, 2 × FC 1024 with BN | Per level, 2 conv × 64, FC 256 and 128 | |
| Mask head | Shared, 4 conv × 256 with BN | Per level, 4 conv × 64, residual, BN | Item 7 |
| Box target scaling | ÷ [0.1, 0.1, 0.2, 0.2] | × (1, 1, 1, 1), fixed | Different |
| Smooth L1 β | 1 (on scaled targets) | 1/9 (on unscaled targets) | torchvision: 1/9 |
| Detection score threshold | 0.7 | 0.5 (0.05 in evaluation) | |
| Detection NMS | 0.3 | 0.5 | |
| Maximum detections | 100 | 100 | Same |
| Mask threshold | 0.5 | 0.5 | Same |
| Optimiser | SGD momentum 0.9, learning rate 0.001 | Adam, learning rate 2e-4 | The paper: 0.02. Mask R-CNN lowered it, reporting exploding weights. |
| Weight decay | 1e-4, BN parameters excluded | 0 | Audit M10 |
| Gradient clipping | Norm 5 per gradient tensor | None | Mask R-CNN always clips, and its README suspects that clipping is what keeps the higher learning rate of the paper stable. `gradient_clipping_norm: 5.0` is a cheap first INS experiment. |
| Batch | 2 images per GPU, BN frozen | 40, BN trained | |
| Loss weights | 1 each, configurable | 1 each, configurable (`Optimization.loss_weights`) | Same; tested, the weights of 1 kept |

## G. Keypoint detection
14. **Keypoint names next to the exported model**: the export of a KPT run writes no channel names. The clients
    rebuild the channel table with `build_keypoint_channel_table` from the `keypoint_categories` that
    `load_experiment_configuration` reads from the `labels.json` of the training data, so a client without the
    training data cannot name the channels. `export_model.py` could write a `keypoint_channels.json` (class, keypoint
    name and skeleton of every channel) next to the exported files, read by the clients instead of the training labels.
    Cost and risk: small.
15. **Mixed precision for the COCO training, checked on the CPU, experiment left to run**: `configuration_example.yaml`
    trains in float32 (`mixed_precision: False`, a setting since removed, every task now training under the bfloat16
    autocast), and a 3000 iteration run on the synthetic shapes with a W16 network
    and 243 heatmap channels processed about 16.5 images per second, the W32 network of the COCO configuration not
    being timed. Under the bfloat16 autocast used by the other tasks, the network predicts its heatmap logits and tags
    in bfloat16, and the heatmap and tag losses and the candidates are computed from them in float32. On the CPU, the
    untrained detector of the tests gives a heatmap loss within 0.06 % of float32 (5 seeds, with and without the
    higher resolution branch) and a finite gradient for every parameter, at a cosine similarity of 0.9996 to the
    float32 gradient (`test_detector_trains_under_the_bfloat16_autocast`, removed on 2026-11-04 with the test suite
    reduction). The tags are rounded to 8 significant bits, within about 0.004 of a tag near 1, which bounds how
    close the pull loss brings the tags of an object, far below the `tag_threshold` of 1.0 of the grouping.
    - **Left:** the OKS AP and the images per second of a short COCO run in each precision, on the GPU.
16. **Ignored regions, implemented, experiment left to run** (`KeypointHead.ignore_unlabelled_objects`): the objects
    left out of the labels (the COCO crowds and the persons without any labelled keypoint) stayed in the images, where
    the heatmaps learned to find no keypoint on persons that are there. With the setting, as in HigherHRNet, the
    dataset unites the masks of these objects, read from their `segmentation`, into the ignored regions of the
    training and validation images, which the augmentation moves with the image, and the heatmap loss leaves out the
    cells at least half covered by them, at both strides and in every channel, the keypoint cells of the labelled
    persons inside them staying trained. On in the COCO `configuration_example.yaml`, off in the synthetic one, whose
    shapes have no `segmentation`.
    - **Left:** the OKS AP of a COCO run with and without the setting, the crowds and the unlabelled persons being
      4652 of the 11004 persons of the validation split.
17. **Finer peaks, implemented, measure on COCO left to run** (`compute_peak_offsets`): `extract_keypoint_candidates`
    moved every local maximum a quarter of a cell towards its higher neighbour. It now places every candidate within
    its cell, along each axis, at the vertex of the parabola through the log-heatmap at its cell and at its two
    neighbours, in the eager and the exported models: the Taylor expansion of DARK without its cross derivative,
    exact for a round Gaussian peak and always within half a cell.
    - **Measured** on 200 test images of the synthetic run (`keypoint_focal_synthetic`, iteration 6000, moving
      average, stride 2, 4290 matched keypoints): the mean error per axis is 0.363 pixel instead of 0.458 on the
      coordinates off the pixel centres (-21 %), and 0.293 instead of 0.127 on the pixel centres. At stride 2 a pixel
      centre lies a quarter of a cell from a cell centre, where the quarter shift was exact. 59 % of the synthetic
      coordinates are pixel centres, so the mean distance over all the synthetic keypoints rises from 0.43 to 0.50
      pixel, while the resized COCO coordinates are continuous.
    - **Variants measured** (off and on the pixel centres): DARK with its Gaussian modulation 0.392 and 0.300, the
      modulation moving the peaks of the focal loss heatmaps further; with the cross derivative 0.370 and 0.310; the
      logit of the heatmap instead of its log 0.354 and 0.262, which no longer moves a peak at exactly 1 (a target
      heatmap, or a peak saturated by a reduced precision export); a parabola on the heatmap values 0.358 and 0.272.
    - **Left:** the OKS AP of a COCO run decoded both ways, the published gain of DARK being largest for small
      objects.

## H. Classification
18. **Decorrelation of the hidden features (Barlow Twins redundancy term), experiment left to run**
    (`ClassificationHead.decorrelation`, implemented): a loss term added to the cross entropy so
    that the hidden features of the classification head carry different information. Every feature is standardised
    over the batch, and the squared off-diagonal terms of the [D, D] correlation matrix of the features are added to the
    loss with a weight. This is the redundancy reduction term of Barlow Twins (arXiv 2103.03230) without its invariance
    term, which needs two augmented views of every image: with the single view of the supervised training, it is the
    DeCov penalty (Cogswell et al., ICLR 2016, arXiv 1511.06068).
    - **Literature:**
      - DeCov penalises the off-diagonal covariance of the hidden activations of supervised networks, and reports a
        smaller gap between the training and validation accuracy, often better than dropout.
      - OrthoReg (Rodríguez et al., ICLR 2017, arXiv 1611.01967) decorrelates the convolution filters instead of the
        activations.
      - VCReg (arXiv 2306.13292) adds the variance and covariance terms of VICReg (arXiv 2105.04906) to the supervised
        training, the main gain being the transfer to other tasks rather than the accuracy on the training task.
      - Barlow Twins trained together with the cross entropy from scratch lowered the accuracy of a VQA model
        (arXiv 2203.03727), the two losses having very different ranges. Training with the cross entropy first, then
        fine-tuning with both, worked.
    - **Measured redundancy:** correlation of the features of the large ImageNet run (`configuration_large_model.yaml`,
      Iteration 115000, EMA weights) on two batches of 256 test images (mean of both batches), each layer compared with
      independent Gaussian features of the same size, measured on 2026-10-18 with a temporary script:

      | Features | Units | Mean \|r\| | % pairs \|r\| > 0.3 | Largest \|r\| | Effective rank | Directions for 90 % of the variance |
      |---|---|---|---|---|---|---|
      | Backbone channels (spatial mean) | 128 | 0.098 | 2.1 | 0.53 | 43 (Gaussian 85) | 59 (Gaussian 86) |
      | Hidden layer 1, before its ReLU | 512 | 0.113 | 3.1 | 0.60 | 46 (Gaussian 171) | 61 (Gaussian 170) |
      | Hidden layer 1, after its ReLU | 512 | 0.082 | 0.5 | 0.58 | 80 (Gaussian 171) | 124 (Gaussian 170) |
      | Hidden layer 2, before its ReLU | 256 | 0.120 | 4.3 | 0.83 | 38 (Gaussian 128) | 44 (Gaussian 131) |
      | Hidden layer 2, after its ReLU | 256 | 0.100 | 1.6 | 0.66 | 51 (Gaussian 128) | 75 (Gaussian 131) |

      The independent features give a mean |r| of 0.050 (the sampling noise at N = 256) and no pair above 0.3. The
      pairwise correlations are low, but the effective rank is too: 90 % of the variance of the 256 units of the second
      hidden layer lie in 44 directions before its ReLU, against 131 for independent features. The redundancy comes from
      many small correlations, which the off-diagonal sum penalises all together. One unit of the second hidden layer
      never fires on these batches.
    - **Implemented:** `ClassificationHead.decorrelation` (`activated`, `weight`, `hidden_layer`), off in every
      configuration file. With the labels, `ImageClassifier.forward` keeps the features of the chosen hidden layer as
      its ReLU receives them (`ClassificationHead.compute_logits`) and adds `decorrelation_loss`, summed into the total
      loss, with the tracked `hidden_feature_correlation` (`compute_decorrelation_loss`). The standardisation and
      correlation of Barlow Twins moved to `compute_feature_cross_correlation`, shared by both losses: Barlow Twins
      itself was not called with the same features as both views, since its invariance term does not vanish for a
      feature varying less than its epsilon, and would be tracked unweighted inside the decorrelation loss. The setting
      gives no run name token, so the warm-up of the cross entropy is a run resumed with it turned on. The second
      hidden layer before its ReLU, where the redundancy is highest, is the first candidate.
    - **Batch size:** the correlation matrix of N images has a rank of at most N − 1, so the term cannot reach 0 with
      more features than images: at the batch size of 256, the 512 units of the first hidden layer keep a floor, while
      the 256 units of the second fit. The sampling noise alone also gives a sum of the squared off-diagonal terms of
      about D(D − 1) / (N − 1), 256 for D = N = 256, below which the term only fits noise.
    - **Experiment:** the classic model with and without the term, for the same number of iterations, compared on the
      test top-1 and on the statistics above, with a weight giving the term about a tenth of the cross entropy at the
      start, turned on after a warm-up of the cross entropy as in the VQA result.

    Cost and risk: small to medium. The large run underfits (iterations 110000 to 115000: training accuracy 0.599 with
    the augmentation and the dropout, validation accuracy 0.613), and the papers report regularisation-sized gains, so
    the term may lower the top-1 as well as raise it. Decorrelated features also stay dependent in non-linear ways.

## I. Selection of training images by embedding clustering
The items below follow the first runs of `scripts/embedding/cluster_image_embeddings.py` on 2026-10-19: the ficha
empty bins run at 512 pixels (`ficha-experiments-empty-bins`, iteration 200) embedding the 43,577 images of its
training split (11,772 `empty_bin`, 31,805 `not_empty`, 61 videos and 874 single photos), with blocks 5 and 6 on a
4 x 4 grid and the two hidden layers, clustered with HDBSCAN (`min_cluster_size` 10, `min_samples` 5, `eom`) on
normalised vectors. The clusters follow the videos (every `not_empty` cluster holds a single video), and the outputs
are in `ficha-experiments-empty-bins/embeddings/`.

19. **Judging the kept images by training the smaller model**: every clustering writes three folders of kept images,
    of the same number of images per category: the cluster representatives with the isolated images, in no cluster
    (`kept_images_by_clustering/`), images drawn at random (`kept_images_at_random/`) and images evenly spaced in every
    video (`kept_images_evenly_spaced_per_group/`).
    - **Sizes without PCA:** 3,615 images for block 5 (8.3 % of the split), 4,089 for block 6, 5,530 for the first
      hidden layer and 6,579 for the second, the isolated images being 49 % (block 5) to 78 % (second hidden layer) of
      the images kept by the clustering. The images kept by the clustering of block 5 hold 1.5 % of training images
      misclassified by the run, against 0.7 % for the images kept at random and 1.4 % for the evenly spaced ones.
    - **DBSCAN of 2026-10-20** (block 6, PCA 200, `eps` 0.16, `min_samples` 5, 5 representatives per cluster): 6,196
      of the 43,577 training images kept (2,049 `empty_bin`, 4,147 `not_empty`, 67 % of them isolated images), and
      5,831 of the 18,702 validation images, the validation images being isolated 2 to 3 times more often, `eps`
      being measured on the training images.
    - **Experiment:** the smaller model trained on the three folders of kept images of one clustering with the same
      settings and number of iterations, evaluated on the whole validation split (balanced accuracy, recall of every
      class), with a few seeds, the kept images being few.
    - **Comparing the layers:** the layers keep different numbers of images, so they are compared at the same number
      (fewer representatives per cluster, or a number of isolated images drawn), or by the accuracy against the number
      of kept images.
    - Cost and risk: one training per folder of kept images; the result may depend on the clustering settings as much
      as on the layer.
20. **The same selection on the compactor dataset**: the compactor run at 512 pixels (iteration 800,
    `ficha-experiments-compactor-to-front`), whose errors gather in a few videos, embedded and clustered the same way,
    then judged as in item 19.
21. **Representatives of the large clusters**: the largest `not_empty` cluster of block 5 is the whole video
    `video_t0180_2026-09-30_08-50-49` (2,666 frames). Its frames closest to the centre show the same full bin, while
    its farthest frame shows a bin being tipped in, which the 5 representatives closest to the centre miss.
    - **Options:** `cluster_selection_method: "leaf"` (smaller clusters, the moments of a video in clusters of their
      own), representatives spread over the ranks of a cluster (the closest one, then images evenly spaced in distance
      up to the farthest), or a number of representatives growing with the size of the cluster (e.g. its square root).
    - Cost and risk: small for the script; each option is judged by the trainings of item 19.
22. **Mosaics of the spread of a cluster**: the mosaic shows the images closest to the centre, which all look alike in
    a large cluster. A second mosaic of images taken at evenly spaced ranks, from the closest to the farthest, would
    show what a cluster gathers. Cost and risk: small.
23. **Corrupted frames**: the isolated images of a trial on 3,000 training images showed frames with decoding
    artefacts (grey images, green and cyan areas). They are isolated, so kept by the clustering with
    `isolated_image_selection = "all"`, and in the training split. Reviewing the isolated image folders lists them, to
    remove them from the `labels.json` of the dataset with a data script rather than with a filter in the framework,
    which stays independent of the datasets. Cost and risk: small, a manual review of the isolated image mosaics and
    folders.

## J. Self-supervised counted shapes: tests left to run
The synthetic counted shapes dataset (`/home/robert_kelevra/Data/self_supervised_data/synthetic_counted_shapes`,
500,000 training, 10,000 validation and 10,000 test images of 5 to 10 shapes at 512 pixels, drawn like the sequence
labelling frames) and its two configurations (`self_supervised/configuration_counted_shapes_simclr.yaml` and
`configuration_counted_shapes_barlow_twins.yaml`, the backbone and flattened feature map of
`sequence_labelling/run-9.yaml`) are ready, but no run has been trained on them yet. The goal is to see whether the
self-supervised backbone learns not only how many shapes an image holds but also which ones.

24. **The two self-supervised trainings**: SimCLR and Barlow Twins trained until the k-NN accuracy of the shape count
    (6 classes, `5_shapes` to `10_shapes`) stops rising, the input viewer checking the two views first.
    - **Watch:** `Embedding Standard Deviation` staying away from 0 (no collapse), and the positive pair accuracy of
      SimCLR not reaching 1 long before the k-NN accuracy rises, which would show a cue shared by the two views.
25. **Probe of which shapes an image holds**: the k-NN evaluation only reads the shape count, the label of every
    sample. The categories of its shapes are in `shape_category_ids` of `labels.json`, for a probe trained on the
    frozen representation: a multi-label linear classifier of the 35 shape classes, its macro F1 against the same probe
    on a backbone with random weights and on a supervised backbone.
26. **Initialisation of sequence labelling**: the self-supervised backbone given to the sequence labelling runs of the
    shape change experiment (`Backbone.pretrained`, the same architecture as `run-9.yaml`), against the same runs from
    random weights, compared on the test Macro F1 of the averaged weights at the same number of iterations.
27. **Strength of the views**: the crop keeps at least 80 % of the image area (`random_resized_crop.scale` [0.8, 1.0])
    so that both views keep most of the shapes and mostly the same count, with flips and rotations of up to 180
    degrees. A smaller crop scale (e.g. [0.5, 1.0]) makes the task harder but can change the number of shapes between
    the two views; to compare on the probes of items 25 and 26 rather than on the k-NN accuracy of the count. The
    harder views of 2026-10-20 add, among others, a random erasing of 2 to 15 % of the image (probability 0.25) and a
    plasma shadow darkening parts of it (probability 0.2), which can also hide a shape of one view and change its
    count: to compare with `random_erasing.probability` and `plasma_shadow.probability` at 0 on the same probes.

## K. README screenshots and synthetic runs: left to redo
The 20 images of `README/` were taken on 2026-10-23 (S11 of `bug_standards.md`) from runs of shortened copies of the
synthetic configurations in `experiments/readme_screenshots/` (MLflow database `mlflow.db`, console logs `<task>.log`
kept with their colours). The helpers that made them are in `experiments/readme_screenshots/helpers/` (not
committed): every image can be taken again from them without training, except where a new run is needed below.

28. **Throughput of the inference on an idle GPU** (`README/run_inference_output.png`): it was measured while three
    other trainings shared the GPU (48.72 samples per second locally, 43.14 through Triton as TorchScript and 96.59 as
    ONNX with TensorRT), so it is lower than what the model gives alone. To take again once the GPU is free: export the
    classification run into `vision_training/deployment/triton/served_models` with `helpers/export_and_measure.py
    --step=export`, start Triton from that folder (`docker compose up -d`), run the same helper with `--step=measure`,
    keep the final line of every progress bar and render them with `helpers/render_console_capture.py`.
29. **Sequence labelling that learns** (`README/sequence_labelling_sample.png`): the synthetic configuration
    (`sequence_labelling/configuration_synthetic_sequences.yaml`) predicts `no_change` for every frame after its 2000
    iterations (balanced accuracy 0.014, the share of one class out of 71, the loss stopping at about 2.85), so the
    sample shows the captions and the X of the wrong frames but its right frames are all `no_change`. The task needs a
    longer training and another learning rate before its sample is taken again.
30. **Synthetic detection runs long enough for their samples**: the detection and instance segmentation configurations
    stop at 2000 iterations, where their detections rarely reach the 0.50 score threshold of the sample images
    (mAP@0.5 of 0.31 and 0.40), the samples showing empty predictions. The README samples come from the same runs
    resumed to 5000 iterations (0.75 and 0.76). Either raise `number_iterations` of both synthetic configurations or
    say in the README that their samples show detections only after a longer training.
31. **Flatten pooling on real sequences**: flatten became the default of the sequence classification on 2026-10-23
    (`12075e7`), checked on the synthetic moving shapes only, at 128 pixels (accuracy 1.000; at 512 pixels its 65536
    features per frame stayed at chance). The example configuration for real data keeps 256 pixels, a frame vector of
    16384 features and about 12.6 million GRU weights, and was not trained: to try on a real dataset, with a smaller
    input or `average` when the class does not depend on where the objects are in the frames.
32. **The synthetic moving shapes dataset against its documentation**: the README and
    `sequence_classification_workflow.md` describe sequences of 2 to 12 frames, the length drawn by
    `create_synthetic_dataset.py`, but every sequence of `synthetic_moving_shapes` (the dataset read by
    `configuration_synthetic_sequences.yaml`) has 8 frames, so the dataset predates the variable lengths. To create it
    again and check that the flatten default still learns with padded sequences of different lengths.
33. **Example outputs recorded before the changes**: Stage 3 of `sequence_classification_workflow.md` was recorded with
    the former `average` pooling (a note says so) and could be recorded again with `flatten` at 128 pixels; the run
    name examples of the README (Run Names) have rows that no configuration gives any more (`CLS_..._HAVG.128_MC4`,
    the `T8` of M2O and CMP).
34. **A frame selection sample with a wrong choice** (`README/frame_selection_sample.png`): its description asks for
    one wrong choice if possible, but the run selects every frame right (1.000 at 2000 iterations), so the sample shows
    three right choices. The weights of an earlier iteration (`Weights/Iteration_500` of the run) or a harder sequence
    could show one.
35. **Parameter columns of the MLflow table** (`README/mlflow_runs.png`): the parameters are logged under the path of
    their setting (`ExperimentConfiguration.learning_rate`), which the narrow columns of the MLflow interface cut to
    `ExperimentCon`, so the table shows the learning rate logged with the metrics instead. Logging a few short
    parameter names (or a run description) would let the table show the settings of the runs.

## Suggested Order
1. **The detection stages scheduled in `object_detection_experiment_plan.md`, in the user's order of 2026-11-07:** the
   FPN and the path aggregation (items 1 and 37), the CIoU box loss (item 36), the detector normalization, then the
   audit fixes M3 (item 3) with L7, and L1.
2. **The FPN with a finer level (item 1), the level by box size (item 2), the shared mask head (item 7) and the
   sharper masks (item 5).** Together they close the main architectural gap for the masks, guided by the AP of the
   small objects and the proposal recall of every level, before any tuning of the anchors or of the proposal counts.
3. **The rest when the need appears:** item 12 when debugging, item 11 when deploying a class with its own threshold,
   items 8, 9 and 10 depending on the datasets, and items 38 and 39 once the detection stages are done.

## Repository Audit For Instance Segmentation

> Audit of the whole framework, carried out on 2026-10-10 at commit `84ac325`. Every line of `vision_training/` was
> read, split into six areas: data pipeline; detection and instance segmentation; backbone, classification and
> segmentation; sequence tasks; training infrastructure; configuration and deployment. Each finding was checked against
> the code. Most were reproduced with small scripts, or with short trainings on the synthetic datasets or the dummy
> task. The critical and high findings were checked a second time by hand.
>
> **Confidence:** every finding is VERIFIED (code paths read end to end, or reproduced) unless it is marked PLAUSIBLE
> (the mechanism is confirmed but the size of the effect depends on the data).
>
> **Severity:** **critical** = stalls or corrupts training, or silently mixes runs. **high** = clearly limits accuracy
> or loses training time. **medium** = limits accuracy in some settings or gives misleading results. **low** = edge
> cases and documentation.
>
> **Baseline:** fixes marked *(changes DET numbers)* change the outputs pinned by `test_detector_baseline.py`, so the
> baseline has to be regenerated. Its `.pt` reference files are only regenerated with your confirmation.

### Summary By Task

| Task | Findings |
|---|---|
| All tasks (training loop, backbone, data, configuration) | C1, C4, H5, H6, M1, M2, M6, M7, M8, M9, M10, M15, M17, M20, L13–L18 |
| DET | C3, H1, H2, H3, M3, M4, M12, M13, M14, M16, M21, L1, L2, L5, L6, L7, L8 |
| INS | Everything DET has, plus M5 (RLE masks), L19, L22 |
| CLS | C2, L3, L4, L9 |
| SEG | C2, M5, M13, L20 |
| M2O, M2M, CMP, SEL | C2, M18 (CMP), L10, L11 (M2M) |

---

### Critical

#### C1. A NaN or Inf loss is never detected, and every later weight and checkpoint becomes NaN
- **Where:**
  - `vision_training/trainers/base_trainer.py:1274-1275`, where the total loss is summed.
  - `base_trainer.py:1027-1028`, where `training_loss.backward()` and `step_optimizer` run with no check.
  - `base_trainer.py:508-510`, where the norm returned by `clip_grad_norm_` is thrown away.
  - No `isfinite` / `isnan` appears anywhere in `trainers/`.
- **Effect (reproduced):** a NaN loss at step 3 of a 10-iteration run turns every weight into NaN.
  - Checkpoints 4, 8 and 10 are NaN, and so is the registry's `last` (the one resume loads). The run ends FINISHED with
    no error line; the console only shows `nan`.
  - Gradient clipping does not help: a NaN norm scales every gradient by NaN.
  - The shipped configurations run for `number_iterations: 1e10`, so one bad batch on day 1 ruins a run left going for
    a week, and resuming reloads the NaN weights.
- **Fix:**
  - Check `torch.isfinite(total_loss)` in `run_model_iteration`. The `.item()` of the metric tracker already
    synchronises, so the check costs nothing extra.
  - On a non-finite loss, zero the gradients, skip `backward` and `step`, count the skipped step and print it.
  - Raise an error the loop does not swallow after N consecutive non-finite losses.
  - Call `clip_grad_norm_(..., error_if_nonfinite=False)`, keep the returned norm, skip the step when it is not finite
    and log the norm to TensorBoard.
  - Refuse to save or register a checkpoint whose parameters are not finite.

#### C2. Label smoothing combined with class weights trains the model to predict the rarest class
- **Where:** `cross_entropy(..., weight=class_weights, label_smoothing=...)` in:
  - `vision_training/losses/classification_losses.py:31-32`
  - `vision_training/losses/segmentation_losses.py:38-40`
  - `vision_training/losses/sequence_losses.py:29-31`
- **What is wrong:** PyTorch also applies the class weights inside the smoothing term:
  `(1-ε)·w_y·(-log p_y) + ε/K·Σ_c w_c·(-log p_c)`. With the inverse-frequency weights of
  `compute_balanced_class_weights` (`classification_losses.py:57-70`), every sample is pulled towards the heavily
  weighted rare classes.
  - For SEG the weights come from pixel counts, so ratios of 1000× are common.
  - A class with no sample gets the weight N/K because of the clamp to 1.
- **Effect (reproduced):**
  - Weights `[1, 20, 1000]` with ε = 0.1: the loss of a class-0 target is minimised by the probabilities
    `[0.027, 0.019, 0.954]`, so the argmax is the rarest class. With ε = 0 it is `[1, 0, 0]`.
  - This happens whenever `label_smoothing > 0` and `class_weighting: "balanced"` are both set, in CLS, SEG and the
    sequence tasks. The checker accepts the combination. The shipped configurations use ε = 0, so they are not
    affected yet.
- **Fix:** compute the smoothed loss without weights, then weight each sample by its target class only:
  ```python
  losses = F.cross_entropy(input=logits, target=targets, ignore_index=IGNORED_LABEL_INDEX, label_smoothing=epsilon,
                           reduction="none")
  sample_weights = class_weights[targets.clamp(min=0)] * (targets != IGNORED_LABEL_INDEX)
  loss = (losses * sample_weights).sum() / sample_weights.sum().clamp(min=1e-12)
  ```

#### C3. DET and INS count their classes as the number of label-dictionary keys, and nothing checks the dictionary
- **Where:**
  - `models/detection/two_stage_detector.py:59`: `self.number_classes = len(self.label_dictionary)`
  - `trainers/detection_trainer.py:281`: `number_classes=len(self.label_dictionary)`
  - `configuration/run_name.py:251`: `MC{len(label_dictionary) - 1}`
  - `configuration/detection_checks.py` has no label check.
  - Every other task uses `len(set(values))` (`classification_dataset.py:51`, `segmentation_dataset.py:107`,
    `sequence_dataset.py:237`, `get_class_names`). The CLS and SEG checks require class indices 0..K-1.
- **Effect (reproduced):**
  - **Merged categories** `{1:1, 2:1, 3:2}` (a documented feature: `get_class_names` joins merged names with `+`):
    - the model gets 4 outputs for 3 classes, one of which is never trained;
    - the per-class report raises `KeyError: 3` at every evaluation, so no test metrics are produced;
    - the run name says `MC3`.
  - **Gaps in the indices** `{bg:0, 1:1, 3:3, 90:90}` (original COCO ids kept): accepted by the checker, then the
    target 90 hits the 4-output cross entropy and training dies with a CUDA device-side assert.
  - **Zero-based mapping** `{1:0, 2:1, 3:2}`, as written by `write_label_mapping`:
    - no background is added (`label_mapping.py:57-59`);
    - category 1 maps to 0 and is silently dropped by `if mapped_label > 0` (`coco_dataset.py:157-159`), so its objects
      become negatives;
    - the background class is reported under that category's name.
  - `coco_dataset.py:157` defaults an unmapped category to 0, so a mapping keyed by names instead of ids silently drops
    every object.
- **Fix:**
  - Use `len(set(label_dictionary.values()))` in the three places above.
  - Add the SEG-style check to `check_detection_configuration`: the indices are exactly 0..K-1, and 0 is used only by
    `"background"`.
  - Print the manifest categories that are absent from the mapping.

#### C4. Run names collide for settings that change the weights, so a different configuration resumes another run
- **Where:** `configuration/run_name.py`
  - `build_pretrained_tokens` (99-112) only tests `experiment_folder is None`.
  - The anchor token is `A<count>.<min>-<max>` (212-216).
  - The backbone tokens (115-149) have no token for `last_max_pooling`, `input_channels` or `dtype`.
  - `base_trainer.py:604-651` only prints the changed settings, then resumes.
- **Effect (reproduced):** each of these changes keeps the run name, so the run resumes the old checkpoint:
  - **Anchor aspect ratios or middle scales changed:** the run resumes heads trained on other anchors, and old
    checkpoints are later evaluated and exported with the new anchors.
  - **New pretrained source (any `*.pretrained.experiment_folder` or `iteration`):** the run resumes the old run's
    checkpoint and prints "The Pretrained Backbone Is Not Loaded Again", so the new source is never used.
  - **`Backbone.last_max_pooling` with a `flatten` head:** `load_state_dict` fails with a size mismatch, so that
    configuration can never train in its folder.
  - **The label mapping file content with the same class count:** the run resumes a model whose classes now mean
    something else.
  - **`dtype` and `input_channels`:** same name for a different model.
  - This contradicts README "Stopping And Resuming" ("Changing any other setting gives another run name") and the
    `build_run_name` docstring.
- **Fix:**
  - Add an `LMP` token for `last_max_pooling` and an `F64` token for float64.
  - Write the anchors in full per level, or as a short hash of `Anchors.scales_and_ratios`.
  - Write `PT<6-character hash of experiment_folder and iteration>`, and the same for every prefixed `*PT` token.
  - Add a short hash of the label dictionary to the classes token.
  - Optionally make `compare_with_saved_configuration` refuse to resume when a key defining the weights changed.

---

### High

#### H1. The region proposal network never trains on plain background *(changes DET numbers)*
- **Where:**
  - `models/detection/target_assignment.py:66-80`: anchors with IoU below `background_iou_threshold.min` go to the
    ignored "grey zone".
  - Every DET and INS configuration sets `RegionProposal.background_iou_threshold.min: 0.1`.
  - `detection_checks.py:139-151` accepts it.
- **What is wrong:** on an image with objects, the only anchors that can be sampled as negatives lie in the ring with
  IoU 0.1–0.3 around an object. The objectness of true background (IoU 0) is never trained. Faster R-CNN, torchvision
  and Detectron2 label every anchor below 0.3 as negative, 0 included. The [0.1, 0.5) range is the Fast R-CNN rule
  for the detection head, not for the RPN.
- **Effect (reproduced):**
  - Example anchors, 3 objects: 772 negatives and 11,307 ignored anchors, against 11,945 negatives and 134 ignored with
    `min: 0.0`.
  - After 300 steps on synthetic squares, far-background anchors made up **39.1 %** of the top-100 proposals with
    `min: 0.1`, against **2.7 %** with `min: 0.0`.
  - `` Step 3.17 shows the same thing: "background 504, ignored 2928" out of
    3456 anchors.
- **Fix:** set `background_iou_threshold.min: 0.0` in the four DET and INS YAMLs, and make the check warn when it is
  above 0.

#### H2. Ground-truth boxes are never added to the proposals of the detection and mask heads *(changes DET numbers)*
- **Where:** `two_stage_detector.py:433-468` (`_prepare_detector_inputs`) assigns targets only to the filtered RPN
  proposals. The TODO at lines 725-727 acknowledges it.
- **What is wrong:** torchvision, Detectron2 and mmdetection append the ground-truth boxes to the proposals during
  training. Without them, the detection head and the mask head get almost no positives until the RPN is good: the
  workflow doc (Steps 3.14 and 3.21) shows 1–3 positives out of 128 sampled per image, and a validation batch with a
  regression loss of 0.
- **Effect (PLAUSIBLE):** slow and unstable training of the heads. Small objects, which rarely get a proposal at
  IoU ≥ 0.5, are under-trained the longest.
- **Fix:** in the training branch, overwrite the last (lowest-scored or padding) N slots of each image's
  `filtered_boxes` with its N ground-truth boxes, keeping the fixed `[B, P, 4]` shape. Give them the level whose anchor
  scales best match `sqrt(w·h)`, or the level of their best-IoU anchor.

#### H3. Validation crashes when `total_training_samples × foreground classes < maximum_detections`
- **Where:**
  - `two_stage_detector.py:700-707`: `postprocess_detections` runs whenever the model is in eval mode, even with ground
    truth.
  - `postprocessing.py` calls `torch.topk(k=maximum_detections)`.
  - `detection_checks.py:243-254` only checks `maximum_detections` against `RegionProposal.inference.post_nms_proposals`.
- **What is wrong:** validation runs in `eval()` with ground truth, so the detector classifies only the sampled
  proposals: min(`Detector.total_training_samples`, inference `post_nms`) per image.
- **Effect (reproduced):** binary classes, `total_training_samples: 64`, `maximum_detections: 100`.
  - The check accepts it and training works.
  - Every validation pass raises `selected index k out of range`, and the loop catches and prints it.
  - With `information_dump` a multiple of `validation_iteration_ratio` (500 and 25 in the examples), every interval
    summary is skipped, so monitoring is lost. INS goes through the same path.
- **Fix:** postprocess only when `ground_truth_bounding_boxes is None`, and skip `predict_detection_masks` in
  `InstanceSegmenter.forward` in the same case. The validation metrics of DET and INS do not use these detections.

#### H5. A failed save or evaluation (e.g. disk full) is swallowed: no checkpoint, no evaluation, skipped steps, orphan `.tmp` files
- **Where:**
  - `base_trainer.py:1002-1009`: the save and evaluation sit inside the per-iteration `try`.
  - `base_trainer.py:1072-1089`: `except Exception ... continue`.
  - `checkpointing.py:133-142` never removes the `.tmp` file of a failed save.
- **Effect (reproduced):** `ENOSPC` from iteration 4 of a 30-iteration run.
  - 13 orphan `model_*.pt.tmp` files, and only 17 optimizer steps out of 30 (each failed save skips its step).
  - No evaluation at all, yet the run ends FINISHED, because the consecutive-error counter resets after every good
    iteration.
  - Runs keep every checkpoint (3.5–6.7 GB per run already), so once the disk is full the run "trains" with no
    checkpoint. The final save at Ctrl+C fails too, losing all the progress since the disk filled.
- **Fix:**
  - Move the save and evaluation out of the swallowing `try`, so that an `OSError` there ends the run with status
    FAILED and a traceback.
  - Delete the `.tmp` file in an `except` of `save_checkpoint`.
  - Add a setting keeping only the last N intermediate checkpoints.

#### H6. An unreadable image silently drops its whole batch at every epoch, and makes every test evaluation fail
- **Where:**
  - `cv2.imread` is never checked for `None` in `coco_dataset.py:139`, `classification_dataset.py:97`,
    `segmentation_dataset.py:225` and `sequence_dataset.py:88` (`read_mask` does check).
  - `base_trainer.py:847-856` (`get_next_batch`): `except Exception: error_counts += 1; continue`, with no message
    below 10 consecutive errors.
- **Effect (reproduced):**
  - One missing file raises `AttributeError` in the worker, and the batch holding it is skipped at every epoch with no
    message. With 1 % unreadable images and batches of 64, about 47 % of the batches are thrown away.
  - A SEG mask whose size differs from its image (`segmentation_dataset.py:228`) is dropped the same way.
  - In the test split, `run_test_evaluation` raises at every checkpoint, so no metrics are produced, and the final
    evaluation (`base_trainer.py:1098`, outside the `try`) crashes the process at the end of the training.
- **Fix:**
  - Raise a `ValueError` naming the file when `imread` returns `None`, in every dataset.
  - Print every fetch error (rate-limited) with its exception, count the dropped batches in the interval summary, and
    stop above a dropped-batch ratio.
  - Optionally check at dataset construction that every listed file exists.

---

### Medium

#### M1. Squeeze-and-excitation gates the residual sum instead of the branch
- **Where:** `convolution_block.py:243-249`: the shortcut is added, then SE is applied to the sum.
- **What is wrong:** `x_{l+1} = g ⊙ (x_l + F(x_l))`, with gates g ≈ 0.5 at initialisation, so the identity path is
  halved at every layer. SE-ResNet uses `x + g ⊙ F(x)`, and the "SE after the sum" variant was the worst in the SENet
  study.
- **Effect (reproduced at initialisation, accuracy PLAUSIBLE):** 20 layers of 64 channels.
  - The product of the gates is 9.6e-7.
  - Output/input cosine: plain 0.31, residual 0.62, repository 0.43, SE on the branch 0.70.
  - The deep CLS configurations (`configuration_large_model`, `configuration_long_model`, 16–20 layers per block with
    SE and residual) lose much of the benefit of their shortcuts.
- **Fix:** on the layers closing a pair, apply SE to the layer output before adding the shortcut. The state-dict keys
  do not change.

#### M2. The shortcut is added after the ReLU, so the residual branch can only increase the features
- **Where:** `convolution_block.py:243-245` ("Add the residual shortcut after the activation"), with the layer order
  conv → norm → ReLU.
- **What is wrong:** `x + ReLU(BN(conv(...)))`. This is the "ReLU before addition" variant of He et al. 2016, which
  had 7.84 % error against 6.61 % on ResNet-110 / CIFAR-10. Activations grow with depth.
- **Fix (PLAUSIBLE gain):** add the shortcut between the normalization and the ReLU of the closing layer, or use
  pre-activation blocks. The keys do not change.

#### M3. Proposals are selected jointly over the levels, starving the upper detection levels *(changes DET numbers)*
- **Where:** `region_proposer.py:207-225` (one `topk` and one `nms` over the anchors of every level); the same in
  `inference_detector.py:131-144`.
- **Effect (PLAUSIBLE):**
  - Level 5 has 4× more anchors than each upper level, and its small boxes survive a cross-level NMS.
  - The workflow doc (Step 3.13) shows proposals per level `{5: 484, 6: 6, 7: 10}`, with 3 RoIs reaching the level-7
    head.
  - So the heads for large objects train on a handful of RoIs, and their batch-norm statistics are noisy.
- **Fix:** take the pre-NMS top-k per level, run `batched_nms` with the level as the group index, then take the global
  top `post_nms`. Change both filters identically so they keep giving the same proposals.

#### M4. The two NMS thresholds look swapped in every DET and INS configuration *(changes DET numbers)*
- **Where:** `RegionProposal.nms_iou_threshold: 0.5` and `Detector.nms_iou_threshold: 0.8` in the four YAMLs.
- **What is wrong:** the usual values are 0.7 for the RPN and 0.5 for the final per-class NMS.
  - The RPN at 0.5 suppresses the proposals of adjacent objects overlapping by 0.5–0.7.
  - The detector at 0.8 keeps duplicates overlapping by 0.5–0.8, each one a false positive at AP@0.5, in evaluation and
    deployment.
- **Fix:** RPN 0.7 and Detector 0.5, then tune.

#### M5. Masks resized with `cv2.INTER_NEAREST` are shifted half a pixel down and to the right of their images
- **Where:**
  - `segmentation_dataset.py:190` and `:237-238` (SEG masks and tiles).
  - `coco_masks.py:216-217` (RLE instance masks).
  - `augmentations.py:405`: `random_crop` resizes masks with torchvision `NEAREST`.
- **What is wrong:** legacy nearest picks `floor(dst · scale)`, which is not centre-aligned, while the images are
  resized bilinearly and the boxes by an exact affine.
- **Effect (reproduced):**
  - Mask minus image centroid is +0.40 to +0.49 px after resizing, about 0.01 px with `INTER_NEAREST_EXACT`.
  - `random_crop` adds +0.5 px, so augmented SEG samples are off by about 1 px.
  - RLE instance masks disagree with the same object given as a polygon by +0.45 px.
  - At `original_resolution: True` a 4096 → 512 shift is about 3.5 original pixels, more than the default
    `boundary_tolerance: 2`.
- **Fix:** use `cv2.INTER_NEAREST_EXACT` in the three OpenCV calls. In `random_crop`, crop the mask, then resize it
  with `InterpolationMode.NEAREST_EXACT`.

#### M6. `number_iterations` counts from the resume point instead of being the total
- **Where:** `base_trainer.py:993`: `last_iteration = self.training_iterations + self.start_iteration - 1`.
- **Effect (reproduced):** a 10-iteration run with a 10-iteration cosine decay, resumed with the same configuration,
  trains up to iteration 20, every extra iteration at the minimum learning rate. This contradicts the YAML comment
  ("Total number of training iterations"), the "Total Iterations" line printed at launch and the absolute iteration of
  the schedule.
- **Fix:** `range(self.start_iteration, self.training_iterations + 1)`, with a message when the run is already
  complete.

#### M7. `resume_training: false` reuses an existing run folder and mixes two trainings
- **Where:** `base_trainer.py:203-205` and `594-610`.
- **Effect (reproduced):** an 8-iteration run followed by a 3-iteration run with `resume_training: false`.
  - The registry ends up as `[2, 3, 4, 6, 8]` with last = 3: `Iteration_2` is overwritten, while 4, 6 and 8 are the
    old training.
  - `Results.txt` and TensorBoard hold both trainings.
  - `scripts/evaluation/evaluate_checkpoint.py` (all checkpoints) or the choice of a best checkpoint mixes the two.
- **Fix:** refuse to start when the folder already has checkpoints and `resume_training` is false, or move the old
  folder aside under a timestamped name.

#### M8. An empty split makes `get_next_batch` loop forever without a message
- **Where:** `base_trainer.py:845-846`: `except StopIteration: dataloader_iterator = iter(dataloader)`, with no limit.
- **Effect (reproduced, hangs):** `ClassificationDataset` drops the samples whose category is unmapped, so a
  validation split with no mapped category busy-loops at 100 % CPU at the first validation.
- **Fix:** check at startup that every split has at least one batch (the counts of `describe_split` are already
  there), and raise after two consecutive `StopIteration`s without a batch.

#### M9. Training-step errors are swallowed indefinitely, and the save after the error limit records a stale model
- **Where:** `base_trainer.py:1063` (the error counter resets after every good iteration), `1084-1089` (prints
  `Exception: {error}` with no traceback), `1091` (final save after the `break`).
- **Effect (reproduced):**
  - An error on every other forward pass gives a FINISHED run with half the optimizer steps and no traceback, e.g. an
    OOM on large detection batches, or a bug that only some inputs trigger.
  - After the error limit, the final save registers `Iteration_10` (4 real steps) as the resume point.
- **Fix:**
  - Print `traceback.format_exc()`.
  - Abort on an error rate over a rolling window, and report the skipped iterations in the summary.
  - Do not register the final save as the resume point after the error limit.

#### M10. Weight decay applies to every parameter, biases and normalization scales included (latent)
- **Where:** `base_trainer.py:431-440`: one `weight_decay` for every group.
- **Effect:** none today (`weight_decay: 0.0` everywhere). As soon as AdamW with 0.05 is used, the batch and layer norm
  scales are pulled towards 0 and the biases are decayed, which costs accuracy on long runs.
- **Fix:** split every group into decayed and non-decayed parameters (`ndim <= 1` get `weight_decay=0`). This changes
  the group layout, which is acceptable since no run uses weight decay yet.

#### M12. DET and INS remove crowd annotations instead of ignoring them
- **Where:** the `iscrowd` filter of `CocoDetectionDataset.__getitem__` (`coco_dataset.py`), DET leaving the crowds
  out like INS since E15 of these suggestions.
- **Effect:**
  - Crowd regions become pure background, so the anchors and proposals on them train as negatives.
  - At test time, detections on crowds count as false positives in the box AP of DET and the box and mask AP of
    INS. COCO ignores them in both cases.
- **Fix:** return the crowd boxes separately. Keep anchors and proposals covering more than half of a crowd box out of
  the negatives, and let an unmatched detection overlapping a crowd region be ignored by `evaluate_detections`.

#### M13. The decoded image size is never compared with the annotation size (EXIF orientation, resized copies) — PLAUSIBLE
- **Where:**
  - `coco_dataset.py:139` and `171` use the decoded shape and ignore the JSON width and height.
  - RLE masks are letterboxed with their own `size` (`coco_masks.py:215-217`).
  - SEG masks are read with `IMREAD_UNCHANGED`, images with `IMREAD_COLOR`.
- **Effect:**
  - OpenCV's `IMREAD_COLOR` applies the EXIF orientation (checked), `IMREAD_UNCHANGED` does not. Boxes annotated on the
    stored pixels land on a rotated image.
  - A SEG mask of a JPEG with orientation 3 passes the shape check and is upside down relative to its image.
  - Images downscaled after annotation give boxes off by the scale factor.
- **Fix:** compare `image.shape[:2]` with the JSON size (and the RLE `size`) and raise on a mismatch. Choose one EXIF
  policy and use the same flag in training and in `deployment_utilities.py:80`.

#### M14. Under mixed precision the objectness is ranked in bfloat16 — FIXED (2026-10-29)
- **Where:** `select_best_proposals` of `region_proposer.py`: `torch.sigmoid(input=proposal_scores)` inside the bf16
  autocast of `base_trainer.py:1270`. The RPN logits are bf16, and `sigmoid` is not cast to float32 by autocast.
- **Effect (reproduced):**
  - 1000 logits in [4, 9] give 6 distinct bf16 probabilities, so the pre-NMS top-k and the NMS order among confident
    anchors are close to arbitrary in training and validation.
  - The test evaluation runs in float32 and is not affected.
- **Fix:** `torch.sigmoid(input=proposal_scores.float())` in `select_best_proposals`, shared by the training filter and
  `InferenceDetector.filter_proposals`.

#### M15. A failed export check still leaves the new model in the served repository
- **Where:** `deployment/export_model.py:231-256`: `model.pt` / `model.onnx` and `config.pbtxt` are written into
  `served_models/<name>/<version>/` before `compare_outputs`.
- **Effect:** a mismatch raises, but the previous good export is already overwritten, and Triton serves the broken one
  at its next start. With several formats, one can be updated while the other stays broken.
- **Fix:** export into a temporary folder, compare, then move the folder into the repository.

#### M16. The DET and INS export comparison probably compares only padding — PLAUSIBLE
- **Where:** `export_specifications.py:278-302` and `333-355`: random-noise inputs, with the deployment
  `score_threshold` (0.5) applied.
- **Effect:** a trained detector very likely returns 0 detections on noise, so the comparison passes whatever the box
  decoding, the NMS or the mask path do.
- **Fix:** compare with `score_threshold = 0`, or compare the raw head outputs and the proposals, and assert that
  `number_detections > 0`.

#### M17. The label mapping file is not part of the run
- **Where:** `configuration_loader.py:188-198` and `label_mapping.py:49-61`. Resume, evaluation and export re-read the
  live `mapping_file_path`.
- **Effect:**
  - Editing the file with the same class count silently changes the class meaning, names, thresholds and the SEL
    `decoding_settings.json` of every old run (see also C4).
  - A different count makes the old runs fail to load, and moving the file makes `export_model` crash.
- **Fix:** copy the resolved label dictionary into the run folder (and the checkpoint metadata), and read it from there
  when resuming, evaluating and exporting.

#### M18. CMP member frames are ordered as text — PLAUSIBLE
- **Where:** `data/sequence/pair_list_manifests.py:58-59` uses `get_image_paths`, which sorts the paths as strings
  (`utilities/os_utilities.py:182`).
- **Effect (reproduced on names):**
  - `frame_1, 2, 3, 9, 10, 11, 12` come out as `[1, 10, 11, 12, 2, 3, 9]`, and the evenly spaced pick of 4 frames
    keeps `[1, 11, 2, 9]`.
  - Any dataset with unpadded frame numbers trains the order-sensitive temporal modules (the GRU of the example) on
    scrambled sequences.
- **Fix:** sort with a natural key (digit runs compared as integers), or refuse unpadded numbered names.

#### M20. The periodic checkpoint `Iteration_k` holds k−1 steps while the final one holds k
- **Where:** `base_trainer.py:1002-1009` saves before the step; `1091` saves after it.
- **Effect (reproduced: Adam steps 1, 3 and 7 in `Iteration_2`, `_4` and `_8`):**
  - The evaluation reported as iteration k is of k−1-step weights.
  - Resuming from a periodic checkpoint starts at k+1, so step k is never run.
- **Fix:** save and evaluate at the end of the iteration, after the step. The special case of the last iteration then
  goes away.

#### M21. TensorRT FP16 is never validated, and the engine cache may serve old weights — PLAUSIBLE
- **Where:**
  - `export_model.py:243` checks the export only with ONNX Runtime on the CPU in FP32.
  - `triton_configuration.py:87-94` enables `trt_engine_cache_enable` with the fixed path `/tmp/tensorrt_engine_cache`.
- **Effect:**
  - FP16 overflow or precision loss is never detected.
  - If the cache key does not cover the weights (not confirmed), re-exporting another checkpoint of the same
    architecture serves the old engine after a container restart.
- **Fix:** version the cache path per export (iteration and hash), and add a Triton-against-local output comparison
  to `run_inference.py`.

---

### Low

#### L1. Anchors are centred on the cell corners, while the backbone and RoIAlign use the cell centres *(changes DET numbers)*
- **Where:** `models/detection/anchors.py:76`: `shifts = arange(F) * stride`. RoIAlign `aligned=True` and the
  stride-1 convolutions with 2×2 pooling place cell j at `(j + 0.5) · stride`.
- **Effect:**
  - A constant half-stride offset that the regressor has to learn.
  - The anchor grid stops one stride before the right and bottom borders.
- **Fix:** `(arange(F) + 0.5) * stride`.

#### L2. Export crashes for some valid configurations
- **Too few anchors:** `inference_detector.py:151` calls `topk(k=post_nms_proposals)` over `min(pre_nms, anchors)`
  candidates. A 512 image with levels 7-9 (126 anchors) and `post_nms: 250` trains, but crashes at export.
  - **Fix:** clamp `post_nms_proposals` in `InferenceDetector`, or check it in the configuration checker.
- **float64 runs:** the export reference forward gets float32 inputs and fails.
  - **Fix:** `.float()` the inference model, or refuse float64 with a message.
- **`Backbone.input_channels` other than 3:** accepted by the checker, but the loaders always give 3-channel BGR and
  the export hard-codes 3 channels.
  - **Fix:** require 3 or remove the key.

#### L3. The threshold finder returns the lowest threshold meeting the target precision, even without any recall gain
- **Where:** `evaluation/classification/threshold_finder.py:45-53`: `last_valid_cut = int(valid_cuts[-1])`.
- **Effect (reproduced):** scores `[.9,.9,.8,.7,.7,.1]` with positives `[T,F,T,T,T,F]` and target 0.6 give the
  threshold 0.10 (precision 0.67), while 0.70 has the same recall at precision 0.80. The saved thresholds sit exactly
  on the target precision, so new data will likely fall below it.
- **Fix:** among the valid cuts, take the first one reaching the maximum number of true positives.

#### L4. `class_weighting: "balanced"` does not balance positives against negatives in multi-label mode
- **Where:** `classification_losses.py:35-37`: one weight per class column multiplies both its positive and negative
  terms.
- **Effect:** the per-class optimum is unchanged, so the recall of rare classes is not improved.
- **Fix:** use `pos_weight = (N - p_c) / p_c` in multi-label mode, or document what the option does.

#### L5. The box-loss normalisation contradicts its comment
- **Where:** `losses/detection_losses.py:65-76` and `114-118`. The losses are divided by the number of positives, with
  a comment saying this follows "standard RPN practice". torchvision divides by the number of sampled anchors or
  proposals.
- **Effect:** the box loss weighs 2× or more relative to the classification, and the factor varies with the positive
  count.
- **Fix:** divide by the sampled count, or correct the comment.

#### L6. The "COCO" mAP@[0.5:0.95] and 101-point use VOC matching
- **Where:** `evaluation/detection/mean_average_precision.py:59-90`. A detection whose best ground truth is already
  taken is a false positive; pycocotools matches the best *unmatched* ground truth above the threshold.
- **Effect (reproduced):** two overlapping objects give AP@0.5 0.50 here against 1.0 with COCO matching. Values are
  underestimated in crowded scenes and are not comparable with pycocotools.
- **Fix:** match each detection to the best unmatched ground truth, or stop calling these COCO.

#### L7. Zero-area proposals and detections are never removed
- **Where:** `region_proposer.py:217-220` (TODO), and the padding by duplication at `262-278`.
- **Effect:**
  - A proposal regressed out of the image becomes zero-area after clamping, and torchvision NMS keeps identical
    zero-area boxes.
  - In the training-model eval path (the test mAP), padding can duplicate them into false-positive detections.
- **Fix:** mask boxes narrower or shorter than 1e-3 px after clamping, in both filters and in the postprocessing.

#### L8. DET boxes are only cleaned on augmented samples
- **Where:** `clean_bounding_boxes` runs only in `augment_sample` (`augmentations.py:214-217`);
  `coco_dataset.py:160-161` neither clips nor filters.
- **Effect (reproduced):**
  - A 0.4 px wide box (`[150.0, 190.0, 150.2, 215.0]`) and a box past the image edge (`x2 = 350 > 320`) reach the
    targets of non-augmented samples and the test ground truth.
  - Thin boxes give extreme log targets through forced matches.
- **Fix:** clip to `[0, S]` and drop boxes under 1 px (with their labels and masks) in `CocoDetectionDataset`.

#### L9. Multi-label classification keeps samples whose categories are all unmapped, as all-negative targets
- **Where:** `classification_dataset.py:61`.
- **Effect:**
  - Mapping a subset of the categories turns every other image into a hard negative for every class, with no
    message, contradicting the class docstring.
  - Merged categories are counted twice in `get_class_counts`, which skews the balanced weights.
- **Fix:** drop the samples with no mapped class (or make it an option), and de-duplicate their class indices.

#### L10. Nothing checks that `frame_category_ids` has one label per frame
- **Where:** `sequence_dataset.py:282-284` (frame labels built without a length check), `346` (label tensor) and
  `128-137` (`collate_padded_sequences`).
- **Effect (reproduced on 2026-11-03 with `collate_padded_sequences`):**
  - A sequence with more or fewer labels than frames stops the loader on the first batch holding it, with a
    `RuntimeError` of `unflatten` ("Provided sizes [1, 3] don't multiply up to the size of dim 0 (5)") that names
    neither the sequence nor the labels file.
  - The SEL check of the dataset counts the labels and reports them as "frames".
- **Fix:** raise a `ValueError` listing the sequences whose label count differs from their frame count.

#### L11. M2M sequences or batches with no labelled frame give a NaN loss and inflate the exact match
- **Where:** `sequence_losses.py:29-31` (mean reduction with every target ignored), `sequence_metrics.py`
  (`compute_sequence_exact_matches`).
- **Effect (reproduced):**
  - The tracked interval loss becomes NaN. The gradients stay finite, but if NaN is the first value the "Best" marker
    stops updating.
  - Sequences without labels count as exact matches.
- **Fix:** sum the losses and divide by `clamp(number of labelled frames, 1)`. Drop these sequences from training and
  from the exact match.

#### L13. The checkpoint registry is rewritten in place without fsync
- **Where:** `checkpointing.py:106-107` (truncate, then write) and `142` (`os.replace` without fsync).
- **Effect (consequence reproduced):** an empty registry makes `get_last_checkpoint_iteration` raise `IndexError`
  while the trainer is built, so a SIGKILL, the OOM killer (memory is shared on the GB10) or a power cut during the
  write makes the run impossible to resume.
- **Fix:** write the registry to a temporary file, fsync it, then replace the original. Fall back to the newest
  complete `Iteration_*` folder when the registry cannot be read.

#### L14. `run_benchmarking_loop` crashes at its first step with every shipped configuration
- **Where:** `base_trainer.py:878` and `893` start at iteration 0; `learning_rate_schedule.py:30-31` divides by
  `warmup_iterations`, which is 0 everywhere.
- **Effect (reproduced):** `ZeroDivisionError`. With a warm-up, the first step runs at learning rate 0. The loop also
  trains the live model, which the commented line in `main.py:31` would then hand to the training.
- **Fix:** iterate from 1, guard the warm-up with `warmup_iterations > 0`, and restore the model and optimizer state
  afterwards.

#### L15. Loader workers are forked from a multi-threaded process — PLAUSIBLE deadlock
- **Where:** `base_trainer.py:984-986`.
  - The TensorBoard writer already runs a thread before `iter(train_dataloader)`.
  - The persistent test workers are forked at the first evaluation, after MLflow started its thread.
  - Python 3.13 warns about it.
- **Fix:** `multiprocessing_context="forkserver"` for the loaders, or create every loader iterator before any thread
  starts. Correct the comment.

#### L16. A host–GPU synchronisation between forward and backward at every step — PLAUSIBLE throughput loss
- **Where:** `metric_tracking.py:124-127` and `157-158`: `.item()` per metric before `backward()`.
- **Fix:** keep the interval sums as detached GPU tensors, and call `.item()` only at the summaries.

#### L17. Ragged last batch and averages over batches
- **Where:** `loaders.py:47-49` (`drop_last` False for training), `metric_tracking.py:138-142`.
- **Effect:**
  - A short last batch weighs as much as a full one in the interval means.
  - A 1-sample batch reaching 1×1 feature maps makes batch norm raise, and that step is skipped silently (M9).
- **Fix:** `drop_last=True` for the training loader only, and weight the means by the sample count.

#### L18. Extreme inputs
- **Extreme aspect ratio:** a 1×2000 image at S = 512 rounds a side to 0 and `cv2.resize` fails
  (`preprocessing.py:33`). Fix: `max(1, round(...))`.
- **16-bit images:** a 16-bit PNG holding 12-bit data becomes 0..15 under `IMREAD_COLOR`. Training and deployment
  agree, but the signal is lost silently. Fix: warn on a non-8-bit source.

#### L19. INS sends full-resolution object masks through shared memory — PLAUSIBLE
- **Where:** `coco_dataset.py:214-222`: one uint8 `[N, S, S]` array per image.
- **Effect:** at S = 1024, batch 40, 8 workers and prefetch 2, about 4.7 GiB of masks are in flight. On the unified
  memory of the GB10 this competes with the GPU, and a small `/dev/shm` kills workers.
- **Fix:** send polygons, RLE or box-cropped masks from the workers and rasterise on the device, or pack the bits.

#### L20. GPU memory peaks in the original-resolution SEG evaluation — PLAUSIBLE
- **Where:** `segmentation_trainer.py:352-362`: a float32 `[K, H, W]` probability map per image at the original
  resolution.
- **Effect:** a 6000×4000 image with 20 classes needs about 1.9 GB per image.
- **Fix:** interpolate in class chunks with a running max and argmax, or in bfloat16.

#### L22. The README recipe for INS masks squeezes masks at the image border — PLAUSIBLE
- **Where:** README Deployment / INS. `restore_boxes_to_original_image` clips the boxes to the image
  (`preprocessing.py:152`), while the M×M mask was predicted over the unclipped box.
- **Fix:** paste with the unclipped restored boxes (`paste_instance_masks` handles parts outside the image), and say so
  in the README.

---

### Statements Contradicting The Code

| Where | Statement | Reality |
|---|---|---|
| README "Stopping And Resuming", `build_run_name` docstring | "Changing any other setting gives another run name" | See C4 |
| Every YAML with `residual_nets` | "Shortcuts are added every two convolution layers" | No shortcut in the 2-layer channel-changing blocks (projection shortcuts of H4 dropped on 2026-11-05) |
| `RegionProposal.foreground_iou_threshold.max` (YAML) | "Maximum IoU limit for matching" | Never read by the model, only by the checker |
| `RegionProposal.strict_fallback_assignment` (YAML) | "Force assign best-overlapping anchors as foreground" | Removed (2026-11-05): they are always forced and regress to the ground truth they overlap the most |
| `losses/detection_losses.py:65-76` | "aligns perfectly with standard RPN normalization" | See L5 |
| `mean_average_precision.py:16` | "COCO" averages | See L6 |
| `region_proposer.py:34` | aspect ratio = width / height | `anchors.py` and the YAMLs use height / width |
| `backbone.py:42` | "Missing keys default to disabled modules" | Keys are read strictly (KeyError) |
| `augmentations.py:530-531` | brightness ±20, contrast 0.8–1.0 | The constants and YAML comments say ±40 and 0.7–1.0 |
| `preprocessing.py:124`, `deployment_utilities.py:3-4` | restore functions "used by the inference clients" | Only used in tests |
| README CLS run name, synthetic CLS YAML | `..._HAVG.128_MC4` | That configuration gives `MC9` |
| `Backbone.input_channels` comment | "3 for RGB images" | The inputs are BGR, and only 3 works (L2) |
| `sequence_classification_trainer.py:66-72` | Frame shuffle warning for every transformer | A non-causal transformer without positional embedding (the README's recommendation for frame sets) cannot see the order |
| `base_trainer.py:984-986` | MLflow opened after the loader workers are forked | See L15 |

Robustness gaps of the configuration loader: `dtype: "float16"` raises a bare `KeyError` before the checker can report
it; `int(float(...))` silently truncates non-integer counts; unknown or misspelled keys are silently ignored.

---

### Checked And Correct

- **Training loop:**
  - bfloat16 autocast without a GradScaler is correct, clipping runs on the real gradients, and `zero_grad` comes
    before the forward pass.
  - The warm-up and cosine schedule maths, applied every step before `optimizer.step`.
  - The learning-rate multipliers after resume, and disjoint parameter groups.
  - Frozen batch-norm statistics restored after every `model.train()`.
  - Validation in eval mode without gradients.
  - Atomic weight files.
  - MLflow never blocks the training thread.
  - The best-checkpoint direction.
- **Detection:**
  - Box encoding and decoding round-trip (9e-7), with deltas clamped before `exp`.
  - The anchor, objectness and offset layouts match.
  - Proposals are detached.
  - The per-class regression picked by the loss and by the postprocessing.
  - Softmax and sigmoid are consistent between training and inference.
  - RoIAlign scale and `aligned` are identical in every head and in the mask targets.
  - Empty levels, zero positives and images without objects are safe.
  - The training-model eval path equals `InferenceDetector` / `InferenceInstanceSegmenter`, padding included.
  - The AP formulas and the exclusion of classes without ground truth.
- **Data:**
  - Box and mask geometry under flips, 90° rotations, crops and small rotations (within 1 px).
  - Letterbox and restore maths.
  - BGR order everywhere, with normalisation inside the model.
  - The polygon rasteriser (0 pixels differ from a pixel-centre reference) and the RLE round trip.
  - Augmentation only on training batches.
  - Sequence padding, masks and collate.
  - The SEG lookup table and ignored pixels.
- **Backbone, CLS, SEG:**
  - U-Net skip alignment for odd sizes.
  - Deep supervision.
  - Finite losses on fully ignored masks.
  - Dice, Lovász and boundary losses.
  - Confusion matrices and IoU / Dice / boundary metrics against hand computations.
  - Sliding-window coverage and averaging.
  - Dropout and batch norm in eval for validation, test and export.
- **Sequence tasks:**
  - All 7 temporal modules are padding-invariant, and the causal ones do not leak future frames.
  - Bidirectional concatenation.
  - Fusion member order.
  - bf16 forward and backward.
  - The Hungarian and greedy SEL decoding (optimal against brute force).
  - The SEL deployed decoding matches the evaluation in 72 of 72 cases.
- **Configuration and deployment:**
  - Every shipped YAML loads, passes the checker, and has every key read.
  - Strict key access.
  - Migration of old configurations from the git history.
  - Deployment preprocessing matches the loaders.
  - Triton dims, dtypes and dynamic axes match the export.

---

### Suggested Order Of Fixes

1. **Training safety, all tasks:** C1 (NaN guard), H5 and H6 (swallowed save, evaluation and data errors), M8 (empty
   split), M9 (error handling), M20 (save after the step), M6 (total iterations), M7 (folder reuse), L13 (atomic
   registry). These protect long runs and change no model.
2. **Correctness of the targets and the run identity:** C3 (detection label dictionary), C4 (run-name collisions), M17
   (label dictionary stored in the run), C2 (smoothing with weights), M5 (exact nearest resizing).
3. **Detection accuracy** *(changes DET numbers, baseline to regenerate with your confirmation)*: H1 (RPN background),
   H2 (ground truths in the proposals), M4 (NMS thresholds), M3 (per-level proposal selection), M14 (float32
   objectness), L1 (anchor centres), H3 (no postprocessing during validation).
4. **Architecture of the shared blocks** *(changes every task)*: M1 (SE on the branch), M2 (shortcut before the
   ReLU). Compare them on the synthetic datasets before adopting them.
5. **Evaluation and deployment:** M15 and M16 (export check), L3, L6
   and the statements of the last table.
