# Suggestions

> Proposed improvements of the object detector that are not implemented yet, kept in `markdown files/` so that we can
> come back to them and decide what to implement. Every item was checked against the code on 2026-10-09; the items
> implemented since the first version of this list (mAP evaluation, checkpoint evaluation script, run names, dataset
> sizes at startup, README, split of the detector and loader code, AdamW / warm-up / cosine / clipping, mixed precision,
> He initialisation (removed after stage 12 of the experiment), pre-activation residual pairs, moving average of the
> weights, its export, best value and summary columns, confusion matrix, class thresholds and error images of the
> detector, batched target assignment, sampling and proposal selection of the training, exact box overlap, crowd
> annotations left out of DET, record of the images of every split (`data.json`), configurable mask heads, covered
> objects left out of the synthetic DET images) were removed, `torch.compile` was dropped after a measurement on the
> GB10 showed no speed-up, and the predictions file of the detector, the dropout and the global average pooling of the
> detection head and the watch for the CUDA crash seen with DALI were dropped.

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

**Suggested order:** the FPN (B3) first, since it also gives the masks of the small objects the finer level of C7.
