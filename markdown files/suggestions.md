# Suggestions

> Proposed improvements of the object detection (DET) and instance segmentation (INS) tasks that are not implemented
> yet, kept in `markdown files/` so that we can come back to them and decide what to implement. They were checked
> against the code on 2026-10-15, together with `repository_audit.md`, whose findings they do not repeat. The items
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
What Mask R-CNN does for the items of `repository_audit.md` that are still open:
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
