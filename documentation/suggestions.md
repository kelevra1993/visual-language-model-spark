# Suggestions

> Proposed improvements of the object detection (DET) and instance segmentation (INS) tasks that are not implemented
> yet, kept in `documentation/` so that we can come back to them and decide what to implement. They were checked
> against the code on 2026-10-15, together with the findings of the
> [Repository Audit For Instance Segmentation](#repository-audit-for-instance-segmentation), which they do not repeat.
> The repository audit and the instructions of the
> [Foundation Models Data Downloader](#foundation-models-data-downloader) are kept at the end of this file. The items
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
> - crowd annotations left out of DET;
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
>   every task, off in every configuration file.
>
> Dropped without being implemented:
> - the masked proposal padding of the training;
> - the partial freezing of a pretrained backbone;
> - `torch.compile`, after a measurement on the GB10 showed no speed-up;
> - the predictions file of the detector;
> - the dropout and the global average pooling of the detection head;
> - the watch for the CUDA crash seen with DALI.

## Table Of Content
- [A. Detector architecture](#a-detector-architecture)
- [B. Instance segmentation](#b-instance-segmentation)
- [C. Evaluation and deployment](#c-evaluation-and-deployment)
- [D. Training data](#d-training-data)
- [E. Mask R-CNN on the open audit items](#e-mask-r-cnn-on-the-open-audit-items)
- [F. Reference: Mask R-CNN compared stage by stage](#f-reference-mask-r-cnn-compared-stage-by-stage)
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
- [Foundation Models Data Downloader](#foundation-models-data-downloader)
  - [Before Running](#before-running)
  - [Running](#running)
  - [Decisions](#decisions)
  - [Changes Forced By The Data](#changes-forced-by-the-data)
  - [Dataset Folder](#dataset-folder)
  - [Risk To Check Before Training](#risk-to-check-before-training)
  - [Dependencies](#dependencies)

## A. Detector architecture
1. **A top-down feature pyramid (FPN)**: lateral 1×1 convolutions and upsampling would give the finest detection
   level the context of the deeper ones. All levels would then share the same channel count, and could share one
   detection head.
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
4. **Box-aware mosaic / copy-paste augmentation**: combine 4 images into one, or paste objects from one image into
   another. Copy-paste is usually combined with a large scale jitter, which the zoom-out (`random_zoom_out`) provides.

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
6. **Scaled box regression targets and a weight per loss**:
   - **Mask R-CNN:** `RPN_BBOX_STD_DEV` and `BBOX_STD_DEV` = [0.1, 0.1, 0.2, 0.2] (`config.py:164-165`). The targets
     are divided by these values in training (`model.py:575, 1550`) and the predictions multiplied by them when
     decoding (`model.py:281, 709`). torchvision uses the equivalent weights (10, 10, 5, 5) on its box head.
     `LOSS_WEIGHTS` multiplies each of the five losses (`config.py:189-195`, `model.py:2172-2174`).
   - **Here:** `turn_boxes_to_transformation_targets` (`target_assignment.py`) and `apply_regression_predictions`
     (`box_operations.py`) work on raw offsets, and the five losses are summed with weight 1.
   - **Why it can help:** scaling the targets by 10 (centres) and 5 (sizes) multiplies the box regression gradient by
     those factors in the linear part of the smooth L1, and by their squares in the quadratic part. The final box
     decides where the mask is pooled and pasted, so a loose box caps the mask AP at 0.75 and above however good the
     mask head is. A weight per loss also lets the mask loss be raised when the masks lag behind the boxes.
   - **Change:** `RegionProposal.regression_weights` and `Detector.regression_weights` (multipliers of dx, dy, dw, dh;
     [1, 1, 1, 1] reproduces today), applied inside the two functions above so that training, evaluation and the
     export stay consistent (the `MAXIMUM_LOG_SCALE_OFFSET` clamp then applies after the division), and an
     `Optimization.loss_weights` dictionary keyed by the tracked loss names. The regression weights change what a
     checkpoint computes, so they need a run name token; the loss weights do not, like the learning rate.
   - **Cost and risk:** small (the two functions, the checks and tests). It changes the DET numbers and interacts
     with audit L5 (dividing by the positives or by the sampled count), so both belong in the same experiment.
   - **Measure:** box mAP@0.75 and mask mAP@0.75 / @[0.5:0.95] on the synthetic shapes, against [1, 1, 1, 1].
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
12. **Evaluation at the original resolution, with the average recall**:
    - **Mask R-CNN:** `evaluate_coco` (`coco.py:342-385`) runs `detect`, which restores the boxes and masks to the
      original image (`model.py:2417-2480`), then scores them with pycocotools, which gives the average recall at 1, 10
      and 100 detections besides the AP of every object size, with the crowds ignored.
    - **Here:** the AP of the small, medium and large objects and the proposal recall exist, but in network input
      pixels. The mask overlaps are computed at the network resolution, against ground truth masks rasterised at that
      resolution, and the objects whose mask vanishes when resized are removed from the test ground truth, so they
      never count as missed. There is no recall of the final detections, and the matching is the VOC one (audit L6).
    - **Change:** `evaluation.original_resolution` for INS, the key the SEG task already has: the boxes restored with
      `restore_boxes_to_original_image`, unclipped for the paste (audit L22), the masks pasted at the original size
      with `paste_instance_masks` in chunks as today, the ground truth masks decoded at the original size
      (`build_instance_mask` without the final resize) with every annotation kept, and the object sizes measured in
      original pixels. To bound the memory, the overlap of a pair can be computed inside the union of its two boxes
      only. Add the average recall at 100 detections, which separates the misses of the region proposal network and
      of the score threshold from bad masks.
    - **Cost and risk:** medium, and a slower evaluation on large images.
    - **Measure:** compare once with pycocotools on a few hundred COCO validation images (a test can do it when
      pycocotools is installed).
13. **Visual checks of the training targets and of the matches**:
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

## D. Training data
14. **Tiles at native resolution for large images**:
    - **Mask R-CNN:** the nucleus sample (`nucleus.py:127-133, 165, 173`) trains on random 512 × 512 crops
      (`IMAGE_RESIZE_MODE: crop`) of the image scaled up by 2 (`IMAGE_MIN_SCALE`), infers on the whole image padded to a
      multiple of 64 (`pad64`), with anchors from 8 to 128 pixels and up to 400 detections.
    - **Here:** the SEG task has the same mechanism as `Data.tiling` (`scale`, `overlap`, `sliding_window.py`). DET and
      INS always resize the whole image to the network input, so objects of a few pixels in a large image vanish, and
      are removed from the ground truth (item 12).
    - **Change:** extend `Data.tiling` to DET and INS. Training samples are tiles cut at random places, the boxes and
      masks cropped with them, an object cut by the border being kept or removed by its visible fraction like the
      fragments of the crop. Test images are processed tile by tile with `compute_tile_starts`, the detections of all
      tiles moved to image coordinates and merged with a class-wise NMS, and the masks pasted into the full image. The
      export is unchanged, one tile per request, the clients merging the tiles as the SEG clients already do.
    - **Cost and risk:** high, the objects cut by tile borders needing care in the merge. Only worth doing for
      datasets of large images with small objects.

## E. Mask R-CNN on the open audit items
What Mask R-CNN does for the items of the [repository audit](#repository-audit-for-instance-segmentation) that are still open:
- **H1 (region proposal background):** Mask R-CNN labels every anchor below 0.3 as negative, 0 included
  (`model.py:1496`). This confirms the fix (`background_iou_threshold.min: 0.0`).
- **M4 (NMS thresholds):** Mask R-CNN uses 0.7 for the region proposals and 0.3 for the detections
  (`config.py:89, 175`). This confirms the swap.
- **M10 (weight decay):** Mask R-CNN skips the BN scales and offsets but still decays the biases
  (`model.py:2179-2183`). It also divides each L2 term by the tensor size, an unusual scaling not to copy.
- **M12 (crowds):** crowd annotations are kept with a negative class id (`coco.py:253-255`) and removed from the
  targets; the anchors (`model.py:1463-1479, 1496`) and RoIs (`model.py:523-547`) whose IoU with any crowd box reaches
  0.001 are kept out of the negatives, and pycocotools ignores the detections matched to crowds. This is a simpler
  rule than M12's ("more than half of a crowd box"): any overlap is enough.
- **L1 (anchor centres):** Mask R-CNN has the same corner offset (`utils.py:603-604`), so it is no argument against
  the fix.
- **L5 (box loss normalisation):** Mask R-CNN averages the smooth L1 over the coordinates of the positives
  (`model.py:1070-1073, 1135-1139`). That is per coordinate, a quarter of a per-box sum, but on targets scaled by 5
  to 10 (item 6).
- **L6 (COCO matching):** Mask R-CNN's `compute_matches` gives each detection, in score order, the best unmatched
  ground truth above the threshold. That is the COCO rule (`utils.py:656-712`), and the COCO sample uses pycocotools
  itself.
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
instead of the argmax class only), the GPU augmentations, and the evaluation at every checkpoint with the AP by object
size and the recall of the anchors and of the proposals.

| Stage | Mask R-CNN | This repository |
|---|---|---|
| Input | Resize so the short side reaches 800 (up only) and the long side stays ≤ 1024, then pad to 1024 × 1024 centred. Modes `crop` (random crops, training) and `pad64` (native size, inference) (`utils.py:388-492`). RGB minus the mean pixel. | Fixed square input, letterboxed centred (`keep_ratio`) or stretched. BGR in [0, 255], then `input_normalization` / `per_sample_normalization`. |
| Backbone | ResNet-50/101 bottleneck stages C2–C5, pretrained on ImageNet or COCO (`model.py:171-206`). | Stack of `ConvolutionBlock`s (residual, SE, inception options). The detection levels are the outputs of the blocks numbered by the keys of `DetectionLevels`, after their pooling. Pretrained from another run of any task. |
| Feature pyramid | FPN, P6 for the region proposals only (`model.py:1902-1925`). | None (item 1). The levels keep the channels of their blocks. |
| Anchors | One scale per level (32 on P2 to 512 on P6), ratios 0.5 / 1 / 2, corner-aligned (`utils.py:583-619`). | Several scales and ratios per level (`DetectionLevels.<level>.anchors`), corner-aligned (audit L1). |
| Region proposal head | One head shared by every level: 3 × 3 conv of 512, then 1 × 1 convs for a 2-class softmax and the offsets (`model.py:830-871`). | One head per level (`DetectionLevels.<level>.region_proposal`): `ConvolutionBlock`, 3 × 3 conv, then 1 × 1 convs for a sigmoid objectness and the offsets. |
| Region proposal targets | Positive at IoU ≥ 0.7, plus the best anchor(s) of every ground truth. Negative below 0.3. Anchors touching a crowd are never negatives. 256 sampled, at most half positive (`model.py:1445-1553`). | Positive at IoU ≥ 0.5, plus the best anchor(s). Negative in [0.1, 0.3), the anchors below 0.1 ignored (audit H1). 256 sampled, at most 128 positive. |
| Proposal filter | One top-k (6000) and one NMS (0.7) over all levels, 2000 kept in training and 1000 at inference, padded with zeros (`model.py:255-329`). | One top-k (1000 / 500) and one NMS (0.5) over all levels (audit M3, M4), 500 / 250 kept, padded with copies of random survivors in training and evaluation (masked slots in deployment), at most one per anchor. |
| RoI sampling | 200 RoIs, 33 % positive (IoU ≥ 0.5). Negatives below 0.5 that touch no crowd, in proportion to the positives, so none without positives (`model.py:486-619`). The ground truth boxes are left out on purpose, a comment citing a sampling study that found no gain (`model.py:1317-1318`). | The ground truth boxes of every image are added to its proposals at the level of their best anchor (`add_ground_truth_proposals`, as in torchvision and Detectron2). 128 RoIs, up to 64 positive (IoU ≥ 0.5), the negatives filling the rest of the budget. |
| RoI pooling | `PyramidROIAlign`: level `4 + round(log2(√(wh) / 224))`, clamped to P2–P5, then `crop_and_resize`, one bilinear sample per bin (`model.py:344-447`). | `roi_align` with `aligned=True` and adaptive sampling, on the level of the anchor that produced the proposal (item 2). |
| Box head | Shared: 7 × 7 conv to 1024 + BN, 1 × 1 conv 1024 + BN, then class and per-class box layers (`model.py:900-953`). | Per level (`DetectionLevels.<level>.detector`): `ConvolutionBlock`, flattened, fully connected layers without normalization, then class and per-class box layers. |
| Box regression | Targets divided by [0.1, 0.1, 0.2, 0.2] on both stages. Smooth L1 with β = 1, averaged over the coordinates of the positives (`model.py:575, 1047-1073, 1112-1140`). | Raw offsets (item 6). Smooth L1 with β = 1/9, summed per box and divided by the positives (audit L5). |
| Mask head | Shared: 4 × (3 × 3 conv 256 + BN + ReLU), 2 × 2 transposed conv 256, 1 × 1 conv to one sigmoid per class (`model.py:956-1005`). | Per level (`DetectionLevels.<level>.mask_head`, item 7): `ConvolutionBlock` of `[layers, channels]` with module options, configurable pool size, mask resolution and upsampling, 1 × 1 conv per class. |
| Mask targets | `crop_and_resize` of the matched mask (full or mini) inside the positive RoI, rounded at 0.5 (`model.py:577-606`). | `roi_align` (aligned) of the matched full-resolution mask inside the positive proposal, thresholded at 0.5. |
| Mask loss | BCE of the ground truth class channel, mean over the positives (`model.py:1143-1179`). | Same, computed in float32. |
| Postprocessing | Class = argmax of each RoI. Score ≥ 0.7, per-class NMS 0.3, top 100. Boxes clipped to the image window, padding excluded (`model.py:684-779`). | Every foreground class of every RoI is a candidate. Score ≥ 0.5 (0.05 in evaluation), batched per-class NMS 0.8 (audit M4), fixed top 100, boxes clamped to the whole input, padding included (item 9). |
| Masks at inference | Mask head on the final boxes. Each 28 × 28 mask is resized to its integer box in the original image, then thresholded at 0.5 (`model.py:2417-2480`, `utils.py:560-576`). | Mask head on the final boxes. `paste_instance_masks` samples at pixel centres with sub-pixel boxes. The evaluation runs at the network resolution (item 12), and the clients paste at the original resolution. |
| Ground truth boxes | Always the tight box of the mask, the dataset boxes being ignored (`model.py:1269`). | The same: the tight box of the mask at the network resolution, the INS annotations needing no `bbox`. |
| Ground truth masks in memory | Optional 56 × 56 crops in the box (`config.py:103-104`, `utils.py:517-552`). | One `[N, S, S]` uint8 mask per object at the network resolution (item 8, audit L19). |
| Crowds | Kept with a negative class id, kept out of the negatives, ignored by pycocotools (section E, M12). | Removed from the annotations (audit M12). |
| Objects lost by the resizing | Skipped at original resolution when under 1 pixel (`coco.py:248-250`). Masks emptied by the resize are dropped from the training targets (`model.py:1263-1265`). The evaluation keeps them, since it runs on the original images. | Dropped from every split, the test split included, so they can never count as missed (item 12). |
| Augmentation | imgaug pipeline with a list of mask-safe augmenters (flips, crop-and-pad, Affine, PiecewiseAffine; `model.py:1232-1259`). COCO uses horizontal flips only, nucleus flips, 90° rotations, brightness and blur. | `DataAugmenter` on the GPU: crop, zoom-out, flips, ±5° and 90° rotations, nine photometric types, masks moved and boxes recomputed, fragments of objects removed. |
| Optimisation | SGD momentum 0.9, learning rate 0.001. Each gradient tensor clipped at norm 5 (Keras `clipnorm`). L2 1e-4 except the BN scales and offsets. Weight per loss (`config.py:177-211`, `model.py:2153-2199`). | Adam / AdamW / SGD Nesterov, warm-up, cosine, optional global-norm clipping. One weight decay for every parameter (audit M10). Losses summed with weight 1 (item 6). |
| Training schedule | Stages selected by layer regular expressions: `heads`, `3+`, `4+`, `5+`, `all` (`model.py:2313-2322`, `coco.py:498-521`). | One stage per run. A loaded component can be frozen whole or given a learning rate multiplier; a schedule needs successive runs. |
| Batch normalization | `TRAIN_BN: False` by default: the stored statistics are used and never updated (batches of 2). | Trained, with a configurable momentum. `layer` normalization is available for small batches. |
| Evaluation | `compute_ap` with mask IoU and COCO matching (`utils.py:656-751`). COCO sample: pycocotools `COCOeval` for boxes and masks at the original resolution (`coco.py:342-385`). | Own mAP (area, 11 and 101 points) at 0.5, 0.75 and 0.5:0.95 with VOC matching (audit L6), by object size, box and mask results per class, proposal recall, at every checkpoint, at the network resolution (item 12). |
| Visual checks | Notebooks walking through every stage, and `display_differences` (item 13). | `view_input_images`, sample grids and the task workflow notes (item 13). |
| Deployment | Keras weights only, batch size fixed at build time. | TorchScript / ONNX exports with fixed shapes, served by Triton, with a comparison check. |

Default hyperparameters of Mask R-CNN (`config.py`) against `instance_segmentation/configuration_example.yaml` (the
synthetic shapes configuration uses the same values at 512 pixels):

| Setting | Mask R-CNN | INS example | Note |
|---|---|---|---|
| Input | 800–1024, padded to 1024 | 1024 letterboxed | |
| Strides of the head levels | 4, 8, 16, 32 (region proposals also 64) | 32, 64, 64 | Item 5 |
| Anchor scales | 32, 64, 128, 256, 512 (one per level) | 32–128, 64–256, 128–512 | |
| Region proposal positive IoU | ≥ 0.7 | ≥ 0.5 | |
| Region proposal negative IoU | < 0.3 | [0.1, 0.3) | Audit H1 |
| Anchors sampled | 256, ≤ 50 % positive | 256, ≤ 128 positive | Same |
| Proposals before NMS | 6000 | 1000 / 500 | |
| Region proposal NMS | 0.7 | 0.5 | Audit M4 |
| Proposals after NMS | 2000 / 1000 | 500 / 250 | |
| RoIs sampled per image | 200, 33 % positive, without the ground truth boxes | 128, ≤ 64 positive, with the ground truth boxes | The paper: 512, 25 % |
| Box head | Shared, 2 × FC 1024 with BN | Per level, 2 conv × 64, FC 256 and 128 | |
| Mask head | Shared, 4 conv × 256 with BN | Per level, 4 conv × 64, residual, BN | Item 7 |
| Box target scaling | ÷ [0.1, 0.1, 0.2, 0.2] | None | Item 6 |
| Smooth L1 β | 1 (on scaled targets) | 1/9 (raw offsets) | Item 6 |
| Detection score threshold | 0.7 | 0.5 (0.05 in evaluation) | |
| Detection NMS | 0.3 | 0.8 | Audit M4 |
| Maximum detections | 100 | 100 | Same |
| Mask threshold | 0.5 | 0.5 | Same |
| Optimiser | SGD momentum 0.9, learning rate 0.001 | Adam, learning rate 2e-4 | The paper: 0.02. Mask R-CNN lowered it, reporting exploding weights. |
| Weight decay | 1e-4, BN parameters excluded | 0 | Audit M10 |
| Gradient clipping | Norm 5 per gradient tensor | None | Mask R-CNN always clips, and its README suspects that clipping is what keeps the higher learning rate of the paper stable. `gradient_clipping_norm: 5.0` is a cheap first INS experiment. |
| Batch | 2 images per GPU, BN frozen | 40, BN trained | |
| Loss weights | 1 each, configurable | 1 each, fixed | Item 6 |

## Suggested Order
1. **Close the open audit items that Mask R-CNN confirms** (section E): H1 (region proposal background), M4 (NMS
   thresholds), L6 (COCO matching) and M12 (crowds, with the any-overlap rule of Mask R-CNN). They change the DET and
   INS numbers, so they share one baseline regeneration.
2. **The scaled regression targets and loss weights (item 6)**, a small change, in the same experiment as audit L5.
3. **The FPN with a finer level (item 1), the level by box size (item 2), the shared mask head (item 7) and the
   sharper masks (item 5).** Together they close the main architectural gap for the masks, guided by the AP of the
   small objects and the proposal recall of every level, before any tuning of the anchors or of the proposal counts.
4. **The evaluation at the original resolution (item 12)** when the network-resolution figures need confirming
   against pycocotools.
5. **The rest when the need appears:** item 13 when debugging, item 3 if one level keeps most proposals, item 11 when
   deploying a class with its own threshold, and items 4, 8, 9, 10 and 14 depending on the datasets.

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
| All tasks (training loop, backbone, data, configuration) | C1, C4, H4, H5, H6, M1, M2, M6, M7, M8, M9, M10, M15, M17, M19, M20, L13–L18 |
| DET | C3, H1, H2, H3, M3, M4, M12, M13, M14, M16, M21, L1, L2, L5, L6, L7, L8 |
| INS | Everything DET has, plus M5 (RLE masks), L19, L22 |
| CLS | C2, L3, L4, L9 |
| SEG | C2, M5, M13, L20 |
| M2O, M2M, CMP, SEL | C2, M11 (M2M), M18 (CMP), L10, L11 (M2M), L12, L21 (M2M), L23 (SEL) |

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

#### H4. `residual_nets: True` does nothing in blocks that change the channel count, and in every 2-layer decoder level
- **Where:**
  - `models/components/convolution_block.py:125`:
    `self.residual_parity = 1 if input_channels != output_channels else 0`.
  - With parity 1 the shortcut is stored at layer 1 and added at layer 2 (lines 231 and 244). A 2-layer block has no
    layer 2, so nothing is added, and there is no 1×1 projection.
  - `segmentation_decoder.py:87-93`: a decoder level reads skip + upsampled channels, never equal to its output.
- **Effect (reproduced):**
  - A block 32→64 with 2 layers gives bitwise-identical outputs with `residual=True` and `residual=False`.
  - The whole example `SegmentationDecoder` is identical with `residual_nets` True and False.
  - Backbone blocks that change width with 2 layers (blocks 2, 4 and 6 of the SEG example and the synthetic SEG and CLS
    configurations) have no shortcut.
  - The detection, mask and region proposal heads use the same block, so they are affected too.
  - `get_active_module_names` still prints "Residual Shortcuts", and the YAML comments say "Shortcuts are added every
    two convolution layers".
- **Fix:** when the input and output channels differ, add a 1×1 convolution with normalization as the projection
  shortcut of the first pair, and set the parity to 0.
  - The new keys only appear in channel-changing residual blocks, so existing checkpoints of those blocks need a
    non-strict load or a migration.
  - Until then, report the residual module only when at least one addition happens.

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

#### M11. M2M frame metrics count the frames of overlapping windows several times
- **Where:** `trainers/sequence_labelling_trainer.py:91-106`, `227-254` and `256-269`.
- **Effect (reproduced):**
  - The accuracy, per-class P/R/F1, confusion matrix, `class_thresholds.json`, `predictions.csv` and the "Labelled
    Frames" header use window-frames. A test split of 153 frames was reported as 206 labelled frames.
  - With the example stride 8 and length 16, most frames count twice and the sequence ends count differently.
  - The thresholds saved for deployment are skewed, and a frame can get several rows with different predictions.
  - The segment metrics already merge the windows (`merge_window_predictions`).
- **Fix:** compute the frame metrics from the merged per-(sequence, frame) probabilities, and name the counts
  "windows" and "unique frames".

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

#### M14. Under mixed precision the objectness is ranked in bfloat16
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

#### M19. `evaluate_checkpoint.py` writes the evaluations of a fine-tuned run into the folder of a scratch run
- **Where:** `scripts/evaluation/evaluate_checkpoint.py:50`: `disable_pretrained_settings` runs before the trainer
  builds its run name.
- **Effect (reproduced):**
  - The output folder loses its `PT` / `HPT` / `RPT` tokens, so it is the folder of a scratch run of the same
    architecture.
  - `Results.txt` is appended to, and `Evaluation/Iteration_<n>/` and `evaluation_summary.csv` are overwritten between
    the two runs.
- **Fix:** name the output folder after the evaluated run folder (`output_folder / experiment_folder.name`), or build
  the run name before disabling the transfers.

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
- **Where:** `sequence_dataset.py:269-270`, `292` and `347`.
- **Effect (reproduced):**
  - A 3-frame sequence with 5 labels gives the padding frame a label, so it takes part in the M2M loss and can be a
    true SEL frame.
  - Fewer labels than frames silently make real frames ignored.
  - The SEL error message reports labels as "frames".
- **Fix:** raise a `ValueError` listing the sequences whose label count differs from their frame count.

#### L11. M2M windows or batches with no labelled frame give a NaN loss and inflate the exact match
- **Where:** `sequence_losses.py:29-31` (mean reduction with every target ignored), `sequence_metrics.py`
  (`compute_sequence_exact_matches`).
- **Effect (reproduced):**
  - The tracked interval loss becomes NaN. The gradients stay finite, but if NaN is the first value the "Best" marker
    stops updating.
  - Windows without labels count as exact matches.
- **Fix:** sum the losses and divide by `clamp(number of labelled frames, 1)`. Drop these windows from training and
  from the exact match.

#### L12. A `window_stride` larger than `sequence_length` leaves frames in no window
- **Where:** `sequence_dataset.py:37-61`; `sequence_checks.py:46-49` only checks that the stride is positive.
- **Effect (reproduced):** `compute_window_starts(40, 16, 20)` gives `[0, 20, 24]`, so frames 16-19 are never seen,
  and M2M segments are split at the gap. This contradicts "every frame belongs to a window".
- **Fix:** require `window_stride <= sequence_length`.

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

#### L21. Duplicate sequence names merge M2M segments
- **Where:** `sequence_labelling_trainer.py:331-333` groups windows by `sequence_name`.
- **Effect:** two sequences with the same name (the same clip under two cameras) are averaged together.
- **Fix:** group by the stored `sequence_identifier`.

#### L22. The README recipe for INS masks squeezes masks at the image border — PLAUSIBLE
- **Where:** README Deployment / INS. `restore_boxes_to_original_image` clips the boxes to the image
  (`preprocessing.py:152`), while the M×M mask was predicted over the unclipped box.
- **Fix:** paste with the unclipped restored boxes (`paste_instance_masks` handles parts outside the image), and say so
  in the README.

#### L23. The local SEL inference client cycles images
- **Where:** `deployment_utilities.py:84-91`: every request is `sequence_length` consecutive folder images, wrapping
  around.
- **Effect:** the frame selector gets duplicated frames, and the decoded indices refer to the wrapped window, so the
  selections of `run_inference` on a folder are meaningless (the throughput numbers are fine).
- **Fix:** send the folder as one request of `len(images)` frames for `frame_selection`.

---

### Statements Contradicting The Code

| Where | Statement | Reality |
|---|---|---|
| README "Stopping And Resuming", `build_run_name` docstring | "Changing any other setting gives another run name" | See C4 |
| Every YAML with `residual_nets` | "Shortcuts are added every two convolution layers" | See H4 |
| `RegionProposal.foreground_iou_threshold.max` (YAML) | "Maximum IoU limit for matching" | Never read by the model, only by the checker |
| `RegionProposal.strict_fallback_assignment` (YAML) | "Force assign best-overlapping anchors as foreground" | They are always forced; the flag only chooses which ground truth they regress to |
| `losses/detection_losses.py:65-76` | "aligns perfectly with standard RPN normalization" | See L5 |
| `mean_average_precision.py:16` | "COCO" averages | See L6 |
| `region_proposer.py:34` | aspect ratio = width / height | `anchors.py` and the YAMLs use height / width |
| `backbone.py:42` | "Missing keys default to disabled modules" | Keys are read strictly (KeyError) |
| `augmentations.py:530-531` | brightness ±20, contrast 0.8–1.0 | The constants and YAML comments say ±40 and 0.7–1.0 |
| `preprocessing.py:124`, `deployment_utilities.py:3-4` | restore functions "used by the inference clients" | Only used in tests |
| `sequence_dataset.py:41-42` | "every frame belongs to a window" | See L12 |
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
  - Window cutting, padding, masks and collate.
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
4. **Architecture of the shared blocks** *(changes every task, checkpoints of channel-changing residual blocks no
   longer load strictly)*: H4 (projection shortcuts), M1 (SE on the branch), M2 (shortcut before the ReLU). Compare
   them on the synthetic datasets before adopting them.
5. **Evaluation and deployment:** M11 (M2M unique frames), M15 and M16 (export check), M19 (evaluation folder), L3, L6
   and the statements of the last table.

## Foundation Models Data Downloader

Instructions for `vision_training/data/classification/foundation_model_image_downloader.py`, which downloads Tencent
ML-Images (ImageNet-22k + Open Images, 11,166 WordNet categories, multi-label) and builds the classification manifests.

### Before Running

1. **ImageNet-22k access**: click "Access repository" on https://huggingface.co/datasets/timm/imagenet-22k-wds with
   your Hugging Face account (approved automatically). Without it the script stops at the start with that link.
2. **ImageNet-1k test images (exclusion of their copies)**: `imagenet-1k/` does not exist yet. Either run
   `vision_training/data/classification/imagenet_download.py` first, or set `excluded_images_folder = None` in the main
   block.
3. **Disk space**: about 1.7 TB for this dataset, plus about 150 GB of ImageNet-1k images (once its parquet files are
   deleted), against 2.2 TB free. The script stops cleanly below `minimum_free_gigabytes` (100 GB) and resumes on the
   next run.

### Running

Set the paths in the main block, then run in tmux (`.venv/bin/python`, since `uv run` breaks when the output is sent to
a log file):

```
.venv/bin/python -m vision_training.data.classification.foundation_model_image_downloader
```

Every step skips the work already done, so it can be run again after any stop.

### Decisions

| Topic | Decision |
|-------|----------|
| ImageNet source | `timm/imagenet-22k-wds` (fall11), 4,608 shards of 0.29 GB, each deleted after use |
| Open Images source | Public S3 bucket through `boto3` (unsigned) |
| Tencent lists | Fetched from Google Drive with `gdown`; if Drive refuses, the error gives the link and the target path |
| Matching | Tencent URL to Open Images identifier by Flickr photo number; unmatched URLs reported |
| Images | Shorter side 512 (never enlarged), JPEG quality 85, Open Images rotation applied |
| Labels | Every label kept (all confidences, ancestors included); confidences stored in `category_confidences` |
| Duplicates | Merged into one sample with combined labels (test copy wins over training copies) |
| Exclusion | Copies of the ImageNet-1k test images removed from every split |
| Splits | Tencent validation images = test; 50,000 training images held out as validation by file name hash |
| Layout | `images/imagenet/<wnid>/`, `images/openimages/<first 2 characters>/`, `manifests/<split>/{data,labels.json}` |
| Class names | First synonym, WordNet identifier appended when two classes share a name |

### Changes Forced By The Data

- **Open Images comes from archives**: only the 1.74M boxable training images exist as single files on S3 (already in
  `open-images-v7`). The full training set is only in 16 `.tar.gz` archives (548 GB), streamed without being saved, at
  about 24 MB/s per archive. Progress is saved every 1,000 images; after an interruption the archive is downloaded again
  but stored images are skipped. The local copy is only used for the validation images.
- **About 1.07M Open Images images are lost**: 15.6% of the Tencent Open Images URLs are no longer in Open Images, so
  about 5.87M Open Images images instead of 6.9M. They are listed in `reports/`.
- **Duplicates use a perceptual hash, not MD5**: the ImageNet-22k copy was re-encoded, so byte hashes could not match
  across sources or with ImageNet-1k. A 128-bit thumbnail hash is used: identical hash for merging duplicates, at most 3
  differing bits for the ImageNet-1k exclusion. Nearly uniform images get no hash and are never merged.

### Dataset Folder

- `annotations/`: Tencent lists, category dictionary, Open Images metadata.
- `images/`: the stored images.
- `progress/`: records of finished shards, batches and archives (resumption).
- `manifests/`: `train`, `validation`, `test` splits and `label_mapping.json`.
- `reports/`: images of the lists left out, `summary.json` with the counts.

### Risk To Check Before Training

The train manifest will hold about 16M samples (about 3 GB of JSON). `ClassificationDataset` loads it into Python
objects, which may use a lot of RAM once the data loader workers start.

### Dependencies

`gdown` and `boto3` were added to `pyproject.toml` and `uv.lock`.
