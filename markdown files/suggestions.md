# Suggestions

> Proposed improvements of the object detector that are not implemented yet, kept in `markdown files/` so that we can
> come back to them and decide what to implement. Every item was checked against the code on 2026-10-09; the items
> implemented since the first version of this list (mAP evaluation, checkpoint evaluation script, run names, dataset
> sizes at startup, README, split of the detector and loader code, AdamW / warm-up / cosine / clipping, mixed precision)
> were removed, and `torch.compile` was dropped after a measurement on the GB10 showed no speed-up.

## A. Evaluation and analysis
The detector is scored with the mAP of every class at every evaluated checkpoint, with its sample images. The outputs
that the classification tasks have and the detector lacks:

1. **A confusion matrix with a background row and column**: which classes are confused with each other, the false
   positives (background detected as an object) and the missed objects, saved as `confusion_matrix.csv` and `.png`
   like the other tasks.
2. **A per-class score threshold**, the equivalent of `find_class_thresholds`: the detection score reaching a target
   precision (e.g. 0.99) for each class, saved in `class_thresholds.json`, instead of the single
   `Detector.inference.score_threshold`. The export could then filter the detections class by class.
3. **Error images**: annotated test images saved into `Errors/FALSE-POSITIVE/<class>`, `Errors/MISSED/<class>` and
   `Errors/<TRUE>-TO-<PREDICTED>`, drawn like the sample images, at most a number per folder.
4. **A predictions file in COCO results format** (`predictions.json`), so that the detections of a checkpoint can be
   analysed again without running the model.
5. **A list of the images of every split** saved in the run folder (`data.json`), recording exactly what a model was
   trained on, next to `training_configuration.yaml`.
6. **Weight histograms in TensorBoard**, switched by a `diagnosis_modes` flag.

## B. Model options
7. **Dropout before the classification layer of the detection head**: its fully connected block is built with a
   dropout rate of 0.
8. **Global average pooling in the detection head** instead of flattening the RoI Align output: cheaper, and less
   tied to `roi_align_pool_size`.
9. **A choice of weight initialisation**, e.g. a truncated normal against the PyTorch defaults, as a configuration
   option to compare both.

## C. Detector code
10. **One proposal filter for training and deployment**: `InferenceDetector.filter_proposals` (batched NMS) and
    `RegionProposalFilter` (a loop over the images of the batch) do the same job. Using the batched one in both would
    remove duplicated code, speed up the training and make training and deployment filter identically.
11. **Vectorise the per-image loops of the training**: the target assignment and sampling
    (`target_assignment.py`) and the proposal filter (`region_proposer.py`) loop over the images of the batch.
12. **Batch the augmentations**: `DataAugmenter.augment_batch` still loops over the images of the batch.
13. **Group the per-level settings in one place** (anchors, region proposal and detector of every level): the
    `RegionProposal`, `Anchors` and `Detector` sections each list the levels, and the checker has to verify that they
    match.
14. **Watch for the CUDA crash seen with DALI** (`cudaErrorIllegalAddress`, with `Xid 13` / `Xid 31` GPU faults in
    the kernel log): DALI has been removed, so a long training tells whether it was involved; if the crash comes back,
    run `compute-sanitizer` on a short run.

## D. Training loop (all tasks)
15. **Stop at the total number of iterations**: a resumed run trains `number_iterations` more from the resume point,
    instead of stopping at `number_iterations` in total.
16. **Unbuffered console output**: the prints are buffered when the output goes to a file. The trainer could flush its
    prints, or use Python's `logging` module.
17. **Handle SIGTERM like Ctrl+C**: a background run ignores SIGINT, so it cannot be stopped with a final save. A
    SIGTERM handler would do the same final save.
18. **Refuse to train without a GPU**, instead of falling back to the CPU. The tests and the workflow examples of
    `markdown files/` run on the CPU on purpose, so it would have to be a configuration switch.

## E. New ideas for the object detector
19. **A top-down feature pyramid (FPN)**: lateral 1×1 convolutions and upsampling would give the finest detection
    level the context of the deeper ones. All levels would then share the same channel count, and could share one
    detection head.
20. **Choosing the detection level by box size** (the FPN rule `k = k₀ + log₂(√(wh)/224)`), instead of the level the
    proposal came from.
21. **NMS per level in the proposal filter**: `batched_nms` with the level as the group, so that one strong level does
    not suppress the proposals of the others.
22. **An EMA of the weights** for the evaluation and the export: a moving average of the weights, usually more stable
    than the last checkpoint.
23. **Box-aware mosaic / copy-paste augmentation**: combine 4 images into one, or paste objects from one image into
    another.
24. **Harder synthetic data**: textured or noisy backgrounds (the backgrounds are plain black or one random colour
    today) and shapes cut by the image borders, so that the synthetic task tells models apart again.

## F. Instance segmentation (Mask R-CNN)
25. **Sharper masks for small objects**: the mask heads pool every box from the backbone feature map of its level
    (RoIAlign 14 × 14, upsampled to 28 × 28), never from the image, so a small object on a coarse level (blocks 5–7
    of the examples) is interpolated from a handful of feature cells and gets blurry borders. In order of cost:
    - an earlier, finer block in `Backbone.detection_convolution_block_indices` (more anchors and heads to train);
    - the FPN of E19, whose finest level gets the context of the deeper ones (the stride 4 level of Mask R-CNN);
    - a mask head reading a finer block than its detection level, e.g. the box pooled from block 4 for the masks
      while the detection head reads block 5;
    - larger masks (`MaskHead.roi_align_pool_size` 28, masks of 56 × 56) or a second transposed convolution,
      useful only once the pooled features themselves are fine enough;
    - a boundary refinement in the style of PointRend, predicting the uncertain border points again from finer
      features.
26. **More configurable mask heads**: the first version has one `MaskHead.roi_align_pool_size` for every level and
    a single transposed convolution doubling it. The mask geometry could be opened up with a `mask_resolution`
    separate from the pool size (the number of upsamplings derived from both), a pool size per level (the masks of
    every level resized to the common resolution before the loss and the export), and a choice of upsampling
    (transposed convolution or bilinear plus convolution, like `SegmentationDecoder.upsampling`).
27. **Crowd annotations in DET**: DET trains and is evaluated on the crowd boxes (`"iscrowd": 1`), which INS leaves
    out. Leaving them out of DET too, or treating them as ignored regions in the target assignment and the evaluation
    as the COCO protocol does, would make the box results of DET and INS comparable on COCO.
28. **Epsilon of the box overlap**: `get_intersection_over_union` adds a small epsilon to the union, so an exact box
    overlap such as 3 / 4 falls just below its threshold, while the mask overlap of INS (exact pixel counts, ratio in
    float64) reaches it: on box-shaped objects the box and mask AP of INS can differ at the exact thresholds. Clamping
    the union instead of adding the epsilon removes the difference, the DET baseline numbers having to be updated.
29. **Covered objects in the synthetic DET images**: `place_shape` only limits the box IoU of the shapes to 0.3, so
    a shape can be fully covered by a later one while its box stays in the labels. The INS generator removes the
    objects whose visible part is under 64 pixels; the DET generator could remove them the same way.

**Suggested order:** the confusion matrix, thresholds and error images (A1–A3) first, since they show where the
detector fails; then C10–C12 for the training speed; then the FPN (E19).
