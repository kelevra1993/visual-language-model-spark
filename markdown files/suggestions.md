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
9. **He (Kaiming) weight initialisation, to evaluate with gradient checks before implementing it** (all tasks): every
   layer keeps the PyTorch default (uniform, std 1/√(3 × fan_in)), √6 ≈ 2.45 times smaller than the He normal
   initialisation (std √(2 / fan_in), "Delving Deep into Rectifiers", He et al., 2015), designed for ReLU networks: a
   ReLU halves the variance of its input and the factor 2 compensates for it, so that the activations keep their scale
   from the first layer to the last. The torchvision ResNets use it for every convolution.
   - **Why**: batch normalisation removes the weight scale of the convolutions from the forward pass, but the scale
     still sets the relative size of every Adam step, about 2.45 times larger with the defaults. The fully connected
     layers of the heads (linear and ReLU, no normalisation) and the Squeeze-and-Excitation layers are not normalised:
     the default divides the variance by about 6 at every ReLU layer, and the logits of the untrained large
     classification model barely differ from one image to the other.
   - **Evaluation**: a gradient check of the untrained classifier, one forward and backward pass on two batches of 32
     synthetic shapes training images (256 pixels) with two seeds, for the PyTorch default, He with fan_in, He with
     fan_out (the torchvision choice) and a truncated normal of std 0.1, on the 19-layer `[1, 2, 2, 4, 5, 5]` × 64 and
     the 61-layer `[1, 2, 2, 16, 20, 20]` layouts (SE, residual, batch normalisation, flattened head `[256, 128]`).
     Measured for every initialisation: the gradient of the first convolution of every block relative to its weights
     (ratio of block 1 and of the largest block to block 6), the spread of the logits across the images and the loss.
     References of the PyTorch default, measured on 2026-10-11 with Squeeze-and-Excitation before the shortcut:

     | Layout | Block 1 / block 6 | Largest block / block 6 | Logit spread |
     |---|---|---|---|
     | 19 layers | 1× / 1× | 2× / 2× | 0.033 / 0.037 |
     | 61 layers | 6× / 7× | 11× / 14× | 0.55 / 0.57 |

     If the check is favourable, a short training of each initialisation (e.g. 500 iterations, run-1 settings) shows
     whether it leaves the uniform prediction loss (ln 9 ≈ 2.197 for 9 classes) earlier.
   - **Implementation**: a configuration option (e.g. `"default"` or `"he"`), applied once to the built task model:
     `kaiming_normal_` with `nonlinearity="relu"` for the convolutions and the hidden linear layers, biases at 0, batch
     normalisation scales at 1 and offsets at 0, and the class logits layer, with no ReLU after it, at std
     √(1 / fan_in). It only changes the starting weights, so it needs no run name token: a resumed run and a pretrained
     component loaded by the transfer keep their weights.
   - **With zero-initialised residual branches** (the last batch normalisation scale of every pair at 0, so that every
     pair starts as a pass-through), possible since the shortcut is added before the ReLU (B10): a variant to include
     in the same gradient check.
10. **Pre-activation residual pairs, to test with gradient checks and short trainings** (all tasks): since
    2026-10-11, a pair of layers adds its shortcut before the ReLU of its second layer, as in the original ResNet
    ("Deep Residual Learning for Image Recognition", He et al., 2015): `ReLU(x + SE(BN(conv(...))))`. Before, the
    shortcut was added after that ReLU, so the branch could only add a non-negative signal and the activations grew
    from one pair to the next (the untrained 61-layer classifier with residuals and without Squeeze-and-Excitation
    started at a loss of 3.25 instead of ln 9 ≈ 2.197). The other standard placement is not available:
    - **Pre-activation ResNet** ("Identity Mappings in Deep Residual Networks", He et al., 2016):
      `x + SE(conv(ReLU(BN(conv(ReLU(BN(x)))))))`, normalisation and ReLU before every convolution and nothing after
      the addition, so that the shortcuts form a pure identity path through the whole block.
    - **Measurement before the change** (2026-10-11, untrained 61-layer classifier, head `[512, 256]`, two seeds):
      the original ResNet placement gave 13× / 11× (block 1 / block 6) and 24× / 26× (largest block / block 6),
      against 7× / 4× and 13× / 15× with the shortcut added after the ReLU. It was not better at initialisation; the
      trainings of the learning rate stage tell whether it leaves the uniform prediction loss earlier.
    - **Evaluation**: the gradient check of B9 (19 and 61 layers, two seeds, block ratios, logit spread, loss) for the
      original and pre-activation placements, with the current head `[256, 128]`, then a short training of each
      (e.g. 500 iterations, run-1 settings) to see which leaves the uniform prediction loss earliest.
    - **Implementation**: a configuration option of the convolution block, with a run name token since it changes
      what a trained checkpoint computes. The pre-activation placement normalises the input of every convolution, so
      the batch normalisation layers of the first layer of a block change size and its checkpoints cannot be loaded
      by the original placement.

## C. Detector code
11. **One proposal filter for training and deployment**: `InferenceDetector.filter_proposals` (batched NMS) and
    `RegionProposalFilter` (a loop over the images of the batch) do the same job. Using the batched one in both would
    remove duplicated code, speed up the training and make training and deployment filter identically.
12. **Vectorise the per-image loops of the training**: the target assignment and sampling
    (`target_assignment.py`) and the proposal filter (`region_proposer.py`) loop over the images of the batch.
13. **Batch the augmentations**: `DataAugmenter.augment_batch` still loops over the images of the batch.
14. **Group the per-level settings in one place** (anchors, region proposal and detector of every level): the
    `RegionProposal`, `Anchors` and `Detector` sections each list the levels, and the checker has to verify that they
    match.
15. **Watch for the CUDA crash seen with DALI** (`cudaErrorIllegalAddress`, with `Xid 13` / `Xid 31` GPU faults in
    the kernel log): DALI has been removed, so a long training tells whether it was involved; if the crash comes back,
    run `compute-sanitizer` on a short run.

## D. Training loop (all tasks)
16. **Stop at the total number of iterations**: a resumed run trains `number_iterations` more from the resume point,
    instead of stopping at `number_iterations` in total.
17. **Unbuffered console output**: the prints are buffered when the output goes to a file. The trainer could flush its
    prints, or use Python's `logging` module.
18. **Handle SIGTERM like Ctrl+C**: a background run ignores SIGINT, so it cannot be stopped with a final save. A
    SIGTERM handler would do the same final save.
19. **Refuse to train without a GPU**, instead of falling back to the CPU. The tests and the workflow examples of
    `markdown files/` run on the CPU on purpose, so it would have to be a configuration switch.

## E. New ideas for the object detector
20. **A top-down feature pyramid (FPN)**: lateral 1×1 convolutions and upsampling would give the finest detection
    level the context of the deeper ones. All levels would then share the same channel count, and could share one
    detection head.
21. **Choosing the detection level by box size** (the FPN rule `k = k₀ + log₂(√(wh)/224)`), instead of the level the
    proposal came from.
22. **NMS per level in the proposal filter**: `batched_nms` with the level as the group, so that one strong level does
    not suppress the proposals of the others.
23. **An EMA of the weights** for the evaluation and the export: a moving average of the weights, usually more stable
    than the last checkpoint.
24. **Box-aware mosaic / copy-paste augmentation**: combine 4 images into one, or paste objects from one image into
    another.
25. **Harder synthetic data**: textured or noisy backgrounds (the backgrounds are plain black or one random colour
    today) and shapes cut by the image borders, so that the synthetic task tells models apart again.

## F. Instance segmentation (Mask R-CNN)
26. **Sharper masks for small objects**: the mask heads pool every box from the backbone feature map of its level
    (RoIAlign 14 × 14, upsampled to 28 × 28), never from the image, so a small object on a coarse level (blocks 5–7
    of the examples) is interpolated from a handful of feature cells and gets blurry borders. In order of cost:
    - an earlier, finer block in `Backbone.detection_convolution_block_indices` (more anchors and heads to train);
    - the FPN of E20, whose finest level gets the context of the deeper ones (the stride 4 level of Mask R-CNN);
    - a mask head reading a finer block than its detection level, e.g. the box pooled from block 4 for the masks
      while the detection head reads block 5;
    - larger masks (`MaskHead.roi_align_pool_size` 28, masks of 56 × 56) or a second transposed convolution,
      useful only once the pooled features themselves are fine enough;
    - a boundary refinement in the style of PointRend, predicting the uncertain border points again from finer
      features.
27. **More configurable mask heads**: the first version has one `MaskHead.roi_align_pool_size` for every level and
    a single transposed convolution doubling it. The mask geometry could be opened up with a `mask_resolution`
    separate from the pool size (the number of upsamplings derived from both), a pool size per level (the masks of
    every level resized to the common resolution before the loss and the export), and a choice of upsampling
    (transposed convolution or bilinear plus convolution, like `SegmentationDecoder.upsampling`).
28. **Crowd annotations in DET**: DET trains and is evaluated on the crowd boxes (`"iscrowd": 1`), which INS leaves
    out. Leaving them out of DET too, or treating them as ignored regions in the target assignment and the evaluation
    as the COCO protocol does, would make the box results of DET and INS comparable on COCO.
29. **Epsilon of the box overlap**: `get_intersection_over_union` adds a small epsilon to the union, so an exact box
    overlap such as 3 / 4 falls just below its threshold, while the mask overlap of INS (exact pixel counts, ratio in
    float64) reaches it: on box-shaped objects the box and mask AP of INS can differ at the exact thresholds. Clamping
    the union instead of adding the epsilon removes the difference, the DET baseline numbers having to be updated.
30. **Covered objects in the synthetic DET images**: `place_shape` only limits the box IoU of the shapes to 0.3, so
    a shape can be fully covered by a later one while its box stays in the labels. The INS generator removes the
    objects whose visible part is under 64 pixels; the DET generator could remove them the same way.

**Suggested order:** the confusion matrix, thresholds and error images (A1–A3) first, since they show where the
detector fails; then C11–C13 for the training speed; then the FPN (E20).
