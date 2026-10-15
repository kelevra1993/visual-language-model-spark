# Suggestions

> Proposed improvements of the object detector that are not implemented yet, kept in `markdown files/` so that we can
> come back to them and decide what to implement. Items 1 to 7 were checked against the code on 2026-10-09 and items
> 8 to 12 on 2026-10-15, together with `repository_audit.md`, whose findings they do not repeat.
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
> - covered objects left out of the synthetic DET images.
>
> Dropped without being implemented:
> - `torch.compile`, after a measurement on the GB10 showed no speed-up;
> - the predictions file of the detector;
> - the dropout and the global average pooling of the detection head;
> - the watch for the CUDA crash seen with DALI.

## A. Detector code
1. **Masked proposal padding in training**: the training filter and the deployed detector share the selection of
   the proposals (`select_best_proposals`), but an image left with fewer survivors than `post_nms_proposals` is
   padded with copies of random survivors in training and evaluation, which the sampling can then pick twice, and
   with masked slots in deployment. Ignoring the padding slots in the target assignment and the sampling would make
   both paddings the same.
2. **Group the per-level settings in one place** (anchors, region proposal, detector and mask head of every level):
   the `RegionProposal`, `Anchors`, `Detector` and `MaskHead` sections each list the levels, and the checker has to
   verify that they match.

## B. New ideas for the object detector
3. **A top-down feature pyramid (FPN)**: lateral 1×1 convolutions and upsampling would give the finest detection
   level the context of the deeper ones. All levels would then share the same channel count, and could share one
   detection head.
4. **Choosing the detection level by box size** (the FPN rule `k = k₀ + log₂(√(wh)/224)`), instead of the level the
   proposal came from.
5. **NMS per level in the proposal filter**: `batched_nms` with the level as the group, so that one strong level does
   not suppress the proposals of the others.
6. **Box-aware mosaic / copy-paste augmentation**: combine 4 images into one, or paste objects from one image into
   another.

## C. Instance segmentation (Mask R-CNN)
7. **Sharper masks for small objects**: the mask heads pool every box from the backbone feature map of its level
   (RoIAlign 14 × 14, upsampled to 28 × 28), never from the image, so a small object on a coarse level (blocks 5–7
   of the examples) is interpolated from a handful of feature cells and gets blurry borders. In order of cost:
   - an earlier, finer block in `Backbone.detection_convolution_block_indices` (more anchors and heads to train);
   - the FPN of B3, whose finest level gets the context of the deeper ones (the stride 4 level of Mask R-CNN);
   - a mask head reading a finer block than its detection level, e.g. the box pooled from block 4 for the masks
     while the detection head reads block 5;
   - larger masks (`MaskHead.mask_resolution` 56, a second doubling of the pooled features) or a larger pool size on
     the coarse levels (`MaskHead.head_configurations.<level>.roi_align_pool_size`), useful only once the pooled
     features themselves are fine enough;
   - a boundary refinement in the style of PointRend, predicting the uncertain border points again from finer
     features.

## D. Evaluation and deployment
8. **Average precision by object size**: the mAP and the per-class AP mix every object size, so what a finer
   detection level, the FPN of B3 or the mask options of C7 bring to the small objects cannot be measured. The COCO
   breakdown of `evaluate_detections` (small objects under 32² pixels, medium up to 96², large above, the objects and
   unmatched detections outside the range being ignored), by box area for DET and mask area for INS, would join the
   table of the methods, the TensorBoard scalars and `class_metrics.csv`. The areas would be measured in network
   input pixels, like the rest of the evaluation, which is the size the detection levels see.
9. **Recall of the anchors and of the proposals**: nothing tells whether a missed object (`Errors/MISSED`) was lost
   because no anchor fits it, because the region proposals dropped it or because the detection head rejected it.
   Two recalls would split the misses:
   - at startup, on a sample of the training images, the share of objects whose best anchor reaches
     `RegionProposal.foreground_iou_threshold.min`, by level and by object size. The others are only trained
     through the anchors overlapping them the most (the fallback assignment). This is a guide for
     `Anchors.scales_and_ratios` on a new dataset;
   - at every evaluated checkpoint, the share of test objects covered by one of the
     `RegionProposal.inference.post_nms_proposals` proposals at IoU 0.5 and 0.7, by level. This measures the
     starvation of the upper levels of the joint proposal selection (M3 of `repository_audit.md`) and how many
     proposals the second stage needs.
10. **Per-class score thresholds in the exported detector**: the evaluation finds the score threshold of every class
    reaching `evaluation.target_precision` (`class_thresholds.json`), but `InferenceDetector` applies the single
    `Detector.inference.score_threshold` to every class, whatever its precision at that score. `export_model.py`
    could read the thresholds of the exported checkpoint, as `frame_selection_deployment.py` does for SEL, and
    `postprocess_detections` could take one threshold per class, a class that never reaches the target precision
    keeping the global one. The confusion matrix, filled batch by batch before the thresholds are found, could be
    built again from the kept detections with them, so that it keeps describing the deployed detector.

## E. Training data
11. **Fragments of objects left by the crop and the rotation**: after the geometric augmentations,
    `clean_bounding_boxes` keeps every box at least 1 pixel wide and high (`MINIMUM_BOX_SIDE`). An object cut down to
    a sliver at the border of a crop (up to a quarter of each side removed) keeps its class, and the detector learns
    to fire on fragments, such as a hand labelled as a person. Keeping only the boxes with a minimum share of their
    area inside the image (of their mask pixels for INS), e.g. 25 %, would remove them. Once ignored regions exist
    (the fix of M12 of `repository_audit.md`), the removed fragments could be ignored instead of becoming background.
12. **Zoom-out augmentation (scale jitter)**: `random_crop` only zooms in (75 to 100 % of each side, objects up to
    1.33 times larger), so every object is seen at its dataset size or larger, and the finest levels only ever train
    on the objects that are already small in the dataset. A `random_zoom_out` type shrinking the image to 50–100 % and
    placing it at random on a black canvas of the input size (the padding of `preprocess_image`), its boxes and masks
    moved like those of the crop, would also show them the larger objects at a smaller size. It is simpler than the
    mosaic of B6.

**Suggested order:** the size breakdown of the mAP (D8) first: it is cheap, and it measures the small objects before
and after the FPN (B3), which also gives the masks of the small objects the finer level of C7. The recalls of D9
come before any tuning of the anchors or of the proposal counts.
