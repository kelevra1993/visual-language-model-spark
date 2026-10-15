# Suggestions

> Proposed improvements of the object detector that are not implemented yet, kept in `markdown files/` so that we can
> come back to them and decide what to implement. Items 1 to 7 were checked against the code on 2026-10-09 and items
> 8 and 9 on 2026-10-15, together with `repository_audit.md`, whose findings they do not repeat.
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
> - object fragments left by the crop and the rotation removed (25 % of their area).
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
   both paddings the same. `RegionProposalFilter` already returns the mask of the slots holding a real survivor
   (read by the proposal recall), which they could take.
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
8. **Per-class score thresholds in the exported detector**: the evaluation finds the score threshold of every class
   reaching `evaluation.target_precision` (`class_thresholds.json`), but `InferenceDetector` applies the single
   `Detector.inference.score_threshold` to every class, whatever its precision at that score. `export_model.py`
   could read the thresholds of the exported checkpoint, as `frame_selection_deployment.py` does for SEL, and
   `postprocess_detections` could take one threshold per class, a class that never reaches the target precision
   keeping the global one. The confusion matrix, filled batch by batch before the thresholds are found, could be
   built again from the kept detections with them, so that it keeps describing the deployed detector.

## E. Training data
9. **Turn on the zoom-out augmentation (scale jitter)**: `random_crop` only zooms in (75 to 100 % of each side,
   objects up to 1.33 times larger), so every object is seen at its dataset size or larger, and the finest levels
   only ever train on the objects that are already small in the dataset. `DataAugmenter.random_zoom_out` is written
   and tested: it shrinks the image to 50–100 % of its side (`MINIMUM_ZOOM_OUT_FRACTION`, drawn independently of the
   crop) at a random place of a black canvas of the input size, moves the boxes and masks with it, marks the
   uncovered border as ignored in a SEG mask and as background in the instance masks, and works for CLS too. It is
   never called yet: turning it on needs its key `Data.data_augmentation.random_zoom_out` (False by default) in the
   configuration files, the checker and the migration, and its name in `GEOMETRIC_AUGMENTATION_TYPES` right after
   `random_crop`. It is simpler than the mosaic of B6.

**Suggested order:** the zoom-out (E9) only needs its configuration key. The FPN (B3) can now be measured on the small
objects by the mAP by object size, which also tells what the finer level of C7 brings to their masks, and on the
proposals of every level by the proposal recall, before any tuning of the anchors or of the proposal counts.
