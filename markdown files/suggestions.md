# Suggestions

> Proposed improvements of the object detector that are not implemented yet, kept in `markdown files/` so that we can
> come back to them and decide what to implement. Every item was checked against the code on 2026-10-09; the items
> implemented since the first version of this list (mAP evaluation, checkpoint evaluation script, run names, dataset
> sizes at startup, README, split of the detector and loader code, AdamW / warm-up / cosine / clipping, mixed precision,
> He initialisation (removed after stage 12 of the experiment), pre-activation residual pairs, moving average of the
> weights, its export, best value and summary columns, confusion matrix, class thresholds and error images of the
> detector) were removed, and `torch.compile` was dropped after a measurement on the GB10 showed no speed-up.

## A. Evaluation and analysis
The detector is scored with the mAP of every class at every evaluated checkpoint, with its sample images, its
confusion matrix with a background row and column, the score threshold of every class reaching a target precision and
its error images. The outputs that the classification tasks have and the detector still lacks:

1. **A predictions file in COCO results format** (`predictions.json`), so that the detections of a checkpoint can be
   analysed again without running the model.
2. **A list of the images of every split** saved in the run folder (`data.json`), recording exactly what a model was
   trained on, next to `training_configuration.yaml`.

## B. Model options
3. **Dropout before the classification layer of the detection head**: its fully connected block is built with a
   dropout rate of 0.
4. **Global average pooling in the detection head** instead of flattening the RoI Align output: cheaper, and less
   tied to `roi_align_pool_size`.

## C. Detector code
5. **One proposal filter for training and deployment**: `InferenceDetector.filter_proposals` (batched NMS) and
   `RegionProposalFilter` (a loop over the images of the batch) do the same job. Using the batched one in both would
   remove duplicated code, speed up the training and make training and deployment filter identically.
6. **Vectorise the per-image loops of the training**: the target assignment and sampling
   (`target_assignment.py`) and the proposal filter (`region_proposer.py`) loop over the images of the batch.
7. **Group the per-level settings in one place** (anchors, region proposal and detector of every level): the
    `RegionProposal`, `Anchors` and `Detector` sections each list the levels, and the checker has to verify that they
    match.
8. **Watch for the CUDA crash seen with DALI** (`cudaErrorIllegalAddress`, with `Xid 13` / `Xid 31` GPU faults in
    the kernel log): DALI has been removed, so a long training tells whether it was involved; if the crash comes back,
    run `compute-sanitizer` on a short run.

## D. New ideas for the object detector
9. **A top-down feature pyramid (FPN)**: lateral 1×1 convolutions and upsampling would give the finest detection
    level the context of the deeper ones. All levels would then share the same channel count, and could share one
    detection head.
10. **Choosing the detection level by box size** (the FPN rule `k = k₀ + log₂(√(wh)/224)`), instead of the level the
    proposal came from.
11. **NMS per level in the proposal filter**: `batched_nms` with the level as the group, so that one strong level does
    not suppress the proposals of the others.
12. **Box-aware mosaic / copy-paste augmentation**: combine 4 images into one, or paste objects from one image into
    another.

## E. Instance segmentation (Mask R-CNN)
13. **Sharper masks for small objects**: the mask heads pool every box from the backbone feature map of its level
    (RoIAlign 14 × 14, upsampled to 28 × 28), never from the image, so a small object on a coarse level (blocks 5–7
    of the examples) is interpolated from a handful of feature cells and gets blurry borders. In order of cost:
    - an earlier, finer block in `Backbone.detection_convolution_block_indices` (more anchors and heads to train);
    - the FPN of D9, whose finest level gets the context of the deeper ones (the stride 4 level of Mask R-CNN);
    - a mask head reading a finer block than its detection level, e.g. the box pooled from block 4 for the masks
      while the detection head reads block 5;
    - larger masks (`MaskHead.roi_align_pool_size` 28, masks of 56 × 56) or a second transposed convolution,
      useful only once the pooled features themselves are fine enough;
    - a boundary refinement in the style of PointRend, predicting the uncertain border points again from finer
      features.
14. **More configurable mask heads**: the first version has one `MaskHead.roi_align_pool_size` for every level and
    a single transposed convolution doubling it. The mask geometry could be opened up with a `mask_resolution`
    separate from the pool size (the number of upsamplings derived from both), a pool size per level (the masks of
    every level resized to the common resolution before the loss and the export), and a choice of upsampling
    (transposed convolution or bilinear plus convolution, like `SegmentationDecoder.upsampling`).
15. **Crowd annotations in DET**: DET trains and is evaluated on the crowd boxes (`"iscrowd": 1`), which INS leaves
    out. Leaving them out of DET too, or treating them as ignored regions in the target assignment and the evaluation
    as the COCO protocol does, would make the box results of DET and INS comparable on COCO.
16. **Epsilon of the box overlap**: `get_intersection_over_union` adds a small epsilon to the union, so an exact box
    overlap such as 3 / 4 falls just below its threshold, while the mask overlap of INS (exact pixel counts, ratio in
    float64) reaches it: on box-shaped objects the box and mask AP of INS can differ at the exact thresholds. Clamping
    the union instead of adding the epsilon removes the difference, the DET baseline numbers having to be updated.
17. **Covered objects in the synthetic DET images**: `place_shape` only limits the box IoU of the shapes to 0.3, so
    a shape can be fully covered by a later one while its box stays in the labels. The INS generator removes the
    objects whose visible part is under 64 pixels; the DET generator could remove them the same way.

**Suggested order:** C5–C6 first, for the training speed; then the FPN (D9).
