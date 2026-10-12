# Instance Segmentation Compared With The Mask R-CNN Repository

> Comparison of the instance segmentation task (INS) of `vision_training/` with the Mask R-CNN repository kept in
> `/home/robert_kelevra/Projects/Mask_RCNN` (the Keras / TensorFlow 1 implementation, commit `3deaec5`), written on
> 2026-10-12 at commit `39cf30b`. Below, "Mask R-CNN" means that repository, and "the paper" means He et al., 2017.
>
> **What was read:** every file of `Mask_RCNN/mrcnn/` (model, utils, config, visualize, parallel_model), the four
> samples (coco, nucleus, balloon, shapes) and the headings of the inspection notebooks. On this side, the INS path
> (instance segmenter, mask head, mask operations, mask loss, trainer, COCO masks and dataset, augmentations, checks,
> configurations) and the detector it extends (backbone, anchors, region proposal, target assignment, detection head,
> losses, postprocessing, inference detector, mean average precision).
>
> **Already listed elsewhere:** items of `suggestions.md` (E20–E22, F26–F28) and of `repository_audit.md` (H1, H2,
> M3, M4, M10, M12, L1, L5–L7, L19, L22) are not proposed again. Section 4 says what Mask R-CNN adds to each of them.
> All those audit items are still open at commit `39cf30b`.

## Table Of Content
- [1. Summary](#1-summary)
- [2. Differences Stage By Stage](#2-differences-stage-by-stage)
- [3. Default Hyperparameters](#3-default-hyperparameters)
- [4. Items Already Listed, And What Mask R-CNN Adds](#4-items-already-listed-and-what-mask-r-cnn-adds)
- [5. Proposals Not Done Yet](#5-proposals-not-done-yet)
  - [P1. Scaled box regression targets and a weight per loss](#p1-scaled-box-regression-targets-and-a-weight-per-loss)
  - [P2. One mask head shared by the levels](#p2-one-mask-head-shared-by-the-levels)
  - [P3. Evaluation at the original resolution, by object size, with recall](#p3-evaluation-at-the-original-resolution-by-object-size-with-recall)
  - [P4. Partial freezing of a pretrained backbone](#p4-partial-freezing-of-a-pretrained-backbone)
  - [P5. Visual checks of the training targets and of the matches](#p5-visual-checks-of-the-training-targets-and-of-the-matches)
  - [P6. Ground truth masks cropped to their boxes](#p6-ground-truth-masks-cropped-to-their-boxes)
  - [P7. Tiles at native resolution for large images](#p7-tiles-at-native-resolution-for-large-images)
  - [P8. Boxes always taken from the masks](#p8-boxes-always-taken-from-the-masks)
  - [P9. Detections clipped to the image window before the mask head](#p9-detections-clipped-to-the-image-window-before-the-mask-head)
  - [P10. Datasets annotating different class subsets](#p10-datasets-annotating-different-class-subsets)
- [6. Differences Not Worth Adopting](#6-differences-not-worth-adopting)
- [7. Suggested Order](#7-suggested-order)

---

## 1. Summary
- **The mask branch is the same design.** Both use RoIAlign 14 × 14, a stack of 3 × 3 convolutions, a 2 × 2 transposed
  convolution to 28 × 28 and one sigmoid mask per class. The loss is a binary cross entropy on the channel of the
  ground truth class, over the positive proposals only. At inference the masks are predicted on the final refined
  boxes. The differences lie in what feeds the mask branch and in how its outputs are scored.
- **The largest difference is the resolution the masks are pooled from.** Mask R-CNN pools from a feature pyramid at
  strides 4, 8, 16 and 32 (P2–P5), and every level gets the context of the deeper ones. The INS example configuration
  pools from the outputs of blocks 5, 6 and 7, at strides 32, 64 and 64 (measured: 32 × 32, 16 × 16 and 16 × 16
  maps at 1024 pixels), without a top-down path. A 32-pixel object, the smallest anchor, spans one feature cell of
  its level, so its 14 × 14 pooled grid is interpolated from about one cell per side.
- **Mask R-CNN shares its heads across levels.** It has one RPN head, one box head and one mask head for every level,
  and assigns each RoI to a level by its size. Here every level has its own three heads, and a RoI keeps the level of
  the anchor it came from. Each mask head therefore trains on the positives of its own level only.
- **The losses are balanced differently.** Mask R-CNN divides the box regression targets of both stages by
  [0.1, 0.1, 0.2, 0.2] and has a weight per loss. Here the raw offsets are regressed, and the five losses are summed
  with weight 1.
- **Mask R-CNN has things this repository lacks:**
  - COCO crowds treated as ignored regions;
  - evaluation at the original image resolution with pycocotools (COCO matching, AP of small, medium and large
    objects, average recall);
  - ground truth masks cropped to their boxes ("mini masks");
  - training in stages (heads, then the upper backbone stages, then everything at a tenth of the learning rate);
  - random crops at native resolution for large images (nucleus sample).

  None of these exists here yet.
- **Where this repository does more than Mask R-CNN:**
  - the export (fixed-shape TorchScript / ONNX);
  - the mask pasting (bilinear sampling at pixel centres, instead of resizing to an integer box then thresholding);
  - RoIAlign (aligned, adaptive sampling, instead of one bilinear sample per bin);
  - the detection candidates (every class of every RoI, instead of the argmax class only);
  - the GPU augmentations, and the evaluation at every checkpoint.

## 2. Differences Stage By Stage

| Stage | Mask R-CNN repository | This repository |
|---|---|---|
| Input | Resize so the short side reaches 800 (up only) and the long side stays ≤ 1024, then pad to 1024 × 1024 centred. Modes `crop` (random crops, training) and `pad64` (native size, inference) exist (`utils.py:388-492`). RGB minus the mean pixel. | Fixed square input, letterboxed centred (`keep_ratio`) or stretched (`preprocessing.py:39-80`). BGR in [0, 255], then `input_normalization` / `per_sample_normalization`. |
| Backbone | ResNet-50/101 bottleneck stages C2–C5, pretrained on ImageNet or COCO (`model.py:171-206`). | Stack of `ConvolutionBlock`s (residual, SE, inception options). The detection levels are the outputs of chosen blocks after their pooling (`backbone.py:150-158`). Pretrained from another run of any task. |
| Feature pyramid | FPN: 1 × 1 laterals to 256 channels, nearest × 2 upsampling and addition top-down, a 3 × 3 convolution on every sum, P6 by stride-2 subsampling for the RPN only (`model.py:1902-1925`). | None. The levels keep the channels of their blocks (64, 128 and 128 in the example). |
| Anchors | One scale per level (32 on P2 to 512 on P6), ratios 0.5 / 1 / 2, optional anchor stride. Corner-aligned (`utils.py:583-619`). | Several scales per level (32–128, 64–256, 128–512), ratios per level. Corner-aligned (`anchors.py:76`, audit L1). |
| RPN head | One head shared by every level: 3 × 3 conv of 512, then 1 × 1 convs for a 2-class softmax and the offsets (`model.py:830-871`). | One head per level: `ConvolutionBlock`, 3 × 3 conv, then 1 × 1 convs for a sigmoid objectness and the offsets (`region_proposer.py:19-145`). |
| RPN targets | Positive at IoU ≥ 0.7, plus the best anchor(s) of every ground truth. Negative below 0.3. Anchors touching a crowd are never negatives. 256 sampled, at most half positive (`model.py:1445-1553`). | Positive at IoU ≥ 0.5, plus the best anchor(s). Negative in [0.1, 0.3), and anchors below 0.1 are ignored (audit H1). 256 sampled, at most 128 positive (`target_assignment.py:16-132`). |
| Proposal filter | One top-k (6000) and one NMS (0.7) over all levels, 2000 kept in training and 1000 at inference, padded with zeros (`model.py:255-329`). | One top-k (1000 / 500) and one NMS (0.5) over all levels (audit M3, M4), 500 / 250 kept, padded with random duplicates in training (`region_proposer.py:172-283`). |
| RoI sampling | 200 RoIs, 33 % positive (IoU ≥ 0.5). Negatives below 0.5 that touch no crowd, in proportion to the positives, so none without positives (`model.py:486-619`). | 128 RoIs, up to 64 positive (IoU ≥ 0.5). Negatives fill the rest of the budget (`target_assignment.py:234-307`). No ground truth boxes are added in either repository (audit H2). |
| RoI pooling | `PyramidROIAlign`: level `4 + round(log2(√(wh) / 224))`, clamped to P2–P5, then `crop_and_resize`, one bilinear sample per bin (`model.py:344-447`). | `roi_align` with `aligned=True` and adaptive sampling, on the level of the anchor that produced the proposal (`detection_head.py:122-124`, suggestion E21). |
| Box head | Shared: 7 × 7 conv to 1024 + BN, 1 × 1 conv 1024 + BN, then class and per-class box layers (`model.py:900-953`). | Per level: `ConvolutionBlock`, flattened, FC [256, 128] without normalization, then class and per-class box layers (`detection_head.py:61-90`). |
| Box regression | Targets divided by [0.1, 0.1, 0.2, 0.2] on both stages. Smooth L1 with β = 1, averaged over the coordinates of the positives (`model.py:575, 1047-1073, 1112-1140`). | Raw offsets. Smooth L1 with β = 1/9, summed per box and divided by the positives (`detection_losses.py:72-76, 114-118`, audit L5). |
| Mask head | Shared: 4 × (3 × 3 conv 256 + BN + ReLU), 2 × 2 transposed conv 256, 1 × 1 conv to one sigmoid per class (`model.py:956-1005`). | Per level: `ConvolutionBlock` of `[layers, channels]` (4 × 64 in the example, with module options), 2 × 2 transposed conv, 1 × 1 conv per class (`mask_head.py:24-93`). The pool size is configurable. |
| Mask targets | `crop_and_resize` of the matched mask (full or mini) inside the positive RoI, rounded at 0.5 (`model.py:577-606`). | `roi_align` (aligned) of the matched full-resolution mask inside the positive proposal, thresholded at 0.5 (`mask_operations.py:14-46`). |
| Mask loss | BCE of the ground truth class channel, mean over the positives (`model.py:1143-1179`). | Same, computed in float32 (`instance_segmentation_losses.py:10-32`). |
| Postprocessing | Class = argmax of each RoI. Score ≥ 0.7, per-class NMS 0.3, top 100. Boxes clipped to the image window, padding excluded (`model.py:684-779`). | Every foreground class of every RoI is a candidate. Score ≥ 0.5 (0.05 in evaluation), batched per-class NMS 0.8 (audit M4), fixed top 100, boxes clamped to the whole input, padding included (`postprocessing.py:15-88`). |
| Masks at inference | Mask head on the final boxes. Each 28 × 28 mask is resized to its integer box in the original image, then thresholded at 0.5 (`model.py:2417-2480`, `utils.py:560-576`). | Mask head on the final boxes (`instance_segmenter.py:157-196`). `paste_instance_masks` samples at pixel centres with sub-pixel boxes (`mask_operations.py:77-103`). The evaluation runs at the network resolution, and clients paste at the original resolution. |
| Ground truth boxes | Always the tight box of the mask, the dataset boxes being ignored (`model.py:1269`). | The annotation box, replaced by the tight box of the mask only after a geometric augmentation (`augmentations.py:199-211`). |
| Ground truth masks in memory | Optional 56 × 56 crops in the box (`config.py:103-104`, `utils.py:517-552`). | One `[N, S, S]` uint8 mask per object at the network resolution (`coco_dataset.py:190-227`, audit L19). |
| Crowds | Kept with a negative class id. Removed from the targets, but anchors and RoIs touching them are kept out of the negatives. Ignored by pycocotools (`coco.py:253-255`, `model.py:523-547`). | Removed from the annotations (`coco_dataset.py:151-152`, audit M12). |
| Objects lost by the resizing | Skipped at original resolution when under 1 pixel (`coco.py:248-250`). Masks emptied by the resize are dropped from the training targets (`model.py:1263-1265`). The evaluation keeps them, since it runs on the original images. | Dropped from every split, the test split included (`coco_dataset.py:224-227`), so they can never count as missed. |
| Augmentation | imgaug pipeline passed to `train()`, with a list of mask-safe augmenters (flips, crop-and-pad, Affine, PiecewiseAffine; `model.py:1232-1259`). COCO uses horizontal flips only. Nucleus uses flips, 90° rotations, brightness and blur. | `DataAugmenter` on the GPU: crop (zoom in only), flips, ±5° and 90° rotations, nine photometric types, with masks moved and boxes recomputed. |
| Optimisation | SGD momentum 0.9, learning rate 0.001. Each gradient tensor clipped at norm 5 (Keras `clipnorm`). L2 1e-4 except the BN scales and offsets. Weight per loss (`config.py:177-211`, `model.py:2153-2199`). | Adam / AdamW / SGD Nesterov, warm-up, cosine, optional global-norm clipping. One weight decay for every parameter (audit M10). Losses summed with weight 1 (`base_trainer.py:1276-1278`). |
| Training schedule | Stages selected by layer regular expressions: `heads`, `3+`, `4+`, `5+`, `all`. COCO: heads, then 4+, then all at a tenth of the learning rate (`model.py:2313-2322`, `coco.py:498-521`). | One stage per run. A loaded component can be frozen whole or given a learning rate multiplier. A schedule needs successive runs. |
| Batch normalization | `TRAIN_BN: False` by default: the stored statistics are used and never updated (batches of 2). | Trained, with a configurable momentum. `layer` normalization is available for small batches. |
| Evaluation | `compute_ap` with mask IoU and COCO matching (best unmatched ground truth, `utils.py:656-751`). COCO sample: pycocotools `COCOeval` for boxes and masks at the original resolution (`coco.py:342-385`). | Own mAP (area, 11 and 101 points) at 0.5, 0.75 and 0.5:0.95, with VOC matching (audit L6). Box and mask results per class, at every checkpoint, at the network resolution. |
| Visual checks | Notebooks walking through anchors, RPN targets, proposals, sampled RoIs, refinement, NMS, mask targets against predictions, activations and weight histograms. `display_differences` draws ground truth and detections on one image with "score / IoU" captions (`visualize.py:170-205`). | `view_input_images` (inputs with boxes and masks), sample grids (ground truth next to detections), and the workflow documents. The detection debugging scripts were deleted in `6225b4a`. `visualise_anchors` remains (`visualization.py:118`). |
| Deployment | Keras weights only, batch size fixed at build time. | TorchScript / ONNX exports with fixed shapes, served by Triton, with a comparison check. |

## 3. Default Hyperparameters
Mask R-CNN defaults (`config.py`) against `configurations/instance_segmentation/configuration_example.yaml`. The
synthetic shapes configuration uses the same values, at 512 pixels.

| Setting | Mask R-CNN | INS example | Note |
|---|---|---|---|
| Input | 800–1024, padded to 1024 | 1024 letterboxed | |
| Strides of the head levels | 4, 8, 16, 32 (RPN also 64) | 32, 64, 64 | Section 1 |
| Anchor scales | 32, 64, 128, 256, 512 (one per level) | 32–128, 64–256, 128–512 | |
| RPN positive IoU | ≥ 0.7 | ≥ 0.5 | |
| RPN negative IoU | < 0.3 | [0.1, 0.3) | audit H1 |
| RPN anchors sampled | 256, ≤ 50 % positive | 256, ≤ 128 positive | Same |
| Proposals before NMS | 6000 | 1000 / 500 | |
| RPN NMS | 0.7 | 0.5 | audit M4 |
| Proposals after NMS | 2000 / 1000 | 500 / 250 | |
| RoIs sampled per image | 200, 33 % positive | 128, ≤ 64 positive | The paper: 512, 25 % |
| Box head | Shared, 2 × FC 1024 with BN | Per level, 2 conv × 64, FC 256 and 128 | |
| Mask head | Shared, 4 conv × 256 with BN | Per level, 4 conv × 64, residual, BN | |
| Box target scaling | ÷ [0.1, 0.1, 0.2, 0.2] | None | P1 |
| Smooth L1 β | 1 (on scaled targets) | 1/9 (raw offsets) | P1 |
| Detection score threshold | 0.7 | 0.5 (0.05 in evaluation) | |
| Detection NMS | 0.3 | 0.8 | audit M4 |
| Maximum detections | 100 | 100 | Same |
| Mask threshold | 0.5 | 0.5 | Same |
| Optimiser | SGD momentum 0.9, learning rate 0.001 | Adam, learning rate 2e-4 | The paper: 0.02. Mask R-CNN lowered it, reporting exploding weights. |
| Weight decay | 1e-4, BN parameters excluded | 0 | audit M10 |
| Gradient clipping | Norm 5 per gradient tensor | None | Mask R-CNN always clips, and its README suspects that clipping is what keeps the paper's higher learning rate stable. `gradient_clipping_norm: 5.0` is a cheap first INS experiment. |
| Batch | 2 images per GPU, BN frozen | 40, BN trained | |
| Loss weights | 1 each, configurable | 1 each, fixed | P1 |

## 4. Items Already Listed, And What Mask R-CNN Adds
- **E20 (feature pyramid):** Mask R-CNN gives a complete recipe (`model.py:1902-1925`):
  - 1 × 1 laterals to 256 channels on C2–C5, nearest × 2 upsampling and addition top-down, and a 3 × 3 convolution
    on every sum;
  - P6, made by subsampling P5, serves the RPN only.

  Two adaptations would be needed here:
  - the 1 × 1 laterals would also bring the levels of different channel counts (64, 128, 128) to a common count;
  - blocks 6 and 7 share the same 16 × 16 resolution, so they would be added without upsampling, or only one of them
    kept.

  What matters most for the masks is a finer level, such as block 3 or 4 (stride 8 or 16) at 1024 pixels, which gets
  the context of the deeper blocks through the top-down path.
- **E21 (level by box size):** Mask R-CNN uses `k = 4 + round(log2(√(wh) / 224))`, clamped to [2, 5], with w and h
  in pixels of the network input (`model.py:380-393`).
  - The rule chooses the level of the box head and of the mask head. During training it is applied to the sampled
    proposals; at inference, to the final refined boxes. So the mask of a detection is pooled from the level matching
    its final size, not the size of its anchor.
  - With levels at strides 32 / 64 / 64, the reference size (224 at P4, stride 16) has to be re-derived, for example
    from the middle anchor scale of every level.
  - The export already runs every level's mask head on every detection and selects per detection with `torch.where`
    (`inference_instance_segmenter.py:95-105`). Only the selector would change.
- **E22 / audit M3 (proposals per level):** Mask R-CNN does not do this either. It runs one top-k and one NMS over all
  levels (`model.py:287-328`). torchvision and Detectron2 (top-k per level, NMS grouped by level) remain the
  reference.
- **F26 (sharper masks for small objects):** Mask R-CNN gets them through P2 at stride 4 with the FPN context, which
  is E20 above.
- **F27 (configurable mask heads):** Mask R-CNN's mask head is fixed (4 × 256 and one transposed convolution, and
  `MASK_SHAPE` must match the network), so there is nothing to take there.
- **F28 / audit M12 (crowds):** Mask R-CNN's recipe:
  - crowd annotations are kept with a negative class id (`coco.py:253-255`) and removed from the targets;
  - anchors (`model.py:1463-1479, 1496`) and RoIs (`model.py:523-547`) whose IoU with any crowd box reaches 0.001 are
    kept out of the negatives;
  - pycocotools ignores the detections matched to crowds.

  This is a simpler rule than M12's ("more than half of a crowd box"): any overlap is enough.
- **Audit H1 (RPN background):** Mask R-CNN labels every anchor below 0.3 as negative, 0 included (`model.py:1496`).
  This confirms the fix (`background_iou_threshold.min: 0.0`).
- **Audit H2 (ground truths in the proposals):** Mask R-CNN leaves the ground truth boxes out on purpose. A comment
  cites Xinlei Chen's sampling study, which found no gain (`model.py:1317-1318`).
  - Instead it offers `USE_RPN_ROIS: False` with `generate_random_rois` (`model.py:1556-1627`): 90 % of the RoIs are
    drawn at random around each ground truth and 10 % anywhere, to train the heads without the RPN.
  - torchvision and Detectron2 do add the ground truth boxes. The audit saw 1–3 positives out of 128 per image early
    in training, which hurts most here, where each mask head only sees its own level.
  - Adding them stays the cheaper fix. It is worth measuring with and without, given Mask R-CNN's position.
- **Audit M4 (NMS thresholds):** Mask R-CNN uses 0.7 for the RPN and 0.3 for the detections (`config.py:89, 175`).
  This confirms the swap.
- **Audit M10 (weight decay):** Mask R-CNN skips the BN scales and offsets but still decays the biases
  (`model.py:2179-2183`). It also divides each L2 term by the tensor size, an unusual scaling not to copy.
- **Audit L1 (anchor centres):** Mask R-CNN has the same corner offset (`utils.py:603-604`), so it is no argument
  against the fix.
- **Audit L5 (box loss normalisation):** Mask R-CNN averages the smooth L1 over the coordinates of the positives
  (`model.py:1070-1073, 1135-1139`). That is per coordinate, a quarter of a per-box sum, but on targets scaled by 5
  to 10 (P1).
- **Audit L6 (COCO matching):** Mask R-CNN's `compute_matches` gives each detection, in score order, the best
  unmatched ground truth above the threshold. That is the COCO rule (`utils.py:656-712`), and the COCO sample uses
  pycocotools itself.
- **Audit L7 (zero-area boxes):** Mask R-CNN removes zero-area detections when restoring them to the image
  (`model.py:2460-2469`). It keeps small proposals on purpose, citing a loss on small objects (`model.py:313-315`).
- **Audit L19 (full-resolution masks through the loaders):** Mask R-CNN's answer is the mini masks (P6).
- **Audit L22 (masks squeezed at the border):** Mask R-CNN clips the refined boxes to the image window before the
  mask head runs on them (`model.py:711, 2043-2052`). The mask is therefore predicted on the box that is later pasted
  (P9).
- **Suggestions A3 (error images) and A4 (predictions file):** Mask R-CNN has `display_differences` (P5) and
  `build_coco_results` (`coco.py:315`), which writes boxes and RLE masks in the COCO results format. For INS, A4 can
  write the masks with the existing `encode_run_length_encoding` (`coco_masks.py:166`).

## 5. Proposals Not Done Yet
Each proposal says what Mask R-CNN does, what this repository does today, what to change and how to measure it. They
are dataset-agnostic, and each would be a configuration switch whose default reproduces today's behaviour.

### P1. Scaled box regression targets and a weight per loss
- **Mask R-CNN:**
  - `RPN_BBOX_STD_DEV` and `BBOX_STD_DEV` = [0.1, 0.1, 0.2, 0.2] (`config.py:164-165`). The targets are divided by
    these values in training (`model.py:575, 1550`) and the predictions multiplied by them when decoding
    (`model.py:281, 709`). torchvision uses the equivalent weights (10, 10, 5, 5) on its box head.
  - `LOSS_WEIGHTS` multiplies each of the five losses (`config.py:189-195`, `model.py:2172-2174`).
- **Here:** `turn_boxes_to_transformation_targets` (`target_assignment.py:185-231`) and
  `apply_regression_predictions` (`box_operations.py:118-172`) work on raw offsets, and the five losses are summed
  with weight 1.
- **Why it can help:**
  - Scaling the targets by 10 (centres) and 5 (sizes) multiplies the box regression gradient by those factors in the
    linear part of the smooth L1, and by their squares in the quadratic part.
  - The final box decides where the mask is pooled and pasted. A loose box caps the mask AP at 0.75 and above, however
    good the mask head is.
  - A weight per loss also lets the mask loss be raised when the masks lag behind the boxes.
- **Change:**
  - Add `RegionProposal.regression_weights` and `Detector.regression_weights` (multipliers of dx, dy, dw, dh;
    [1, 1, 1, 1] reproduces today).
  - Apply them inside the two functions above, so that training, evaluation and the export stay consistent: the
    inference detector calls the same functions. The `MAXIMUM_LOG_SCALE_OFFSET` clamp then applies after the division.
  - Add a loss weight per loss, for example an `Optimization.loss_weights` dictionary keyed by the tracked loss names.
  - The regression weights change what a checkpoint computes, so they need a run name token. The loss weights do not,
    like the learning rate.
- **Cost and risk:**
  - **Cost:** small. The two functions, the checks, a configuration migration and tests.
  - **Risk:** it changes the DET numbers. It interacts with audit L5 (dividing by the positives or by the sampled
    count), so both belong in the same experiment.
- **Measure:** box mAP@0.75 and mask mAP@0.75 / @[0.5:0.95] on the synthetic shapes, against [1, 1, 1, 1].

### P2. One mask head shared by the levels
- **Mask R-CNN:** `build_fpn_mask_graph` is built once (`model.py:956-1005`) and receives the RoIs of every level
  through `PyramidROIAlign`. The box head and the RPN are shared the same way (`model.py:1939-1944`).
- **Here:**
  - There is one `MaskHead` per level (`instance_segmenter.py:51-74`), and `compute_proposal_mask_loss` splits the
    positives by level (`instance_segmenter.py:139-147`). Each head trains on a fraction of the positives of the batch.
  - The export runs every level's mask head on every detection (`inference_instance_segmenter.py:95-105`), which
    triples the mask computation for three levels.
- **Change:**
  - Add `MaskHead.shared: True / False`.
  - When shared, a per-level 1 × 1 convolution with normalization brings every level to the input channel count of
    one `MaskHead`, which then sees every positive.
  - The export would run that single head once on all the detections.
  - Without the FPN (E20), the levels still differ in depth and context, so this works best together with E20 or after
    it. The detection heads could be shared the same way, as E20 already says.
- **Cost and risk:**
  - **Cost:** medium. It needs a run name token, and the model transfer must refuse to load per-level heads into a
    shared one.
  - **Risk:** a shared head may fit the coarse levels worse than dedicated heads do. That is the point of measuring it.
- **Measure:** mask mAP overall and by object size (P3), and the export throughput.

### P3. Evaluation at the original resolution, by object size, with recall
- **Mask R-CNN:**
  - `evaluate_coco` (`coco.py:342-385`) runs `detect`, which restores the boxes and masks to the original image
    (`model.py:2417-2480`), then scores them with pycocotools.
  - pycocotools gives AP at 0.5:0.95, 0.5 and 0.75, the AP of small (< 32² pixels), medium and large (> 96² pixels)
    objects, and the average recall at 1, 10 and 100 detections, with the crowds ignored.
- **Here:**
  - The mask overlaps are computed at the network resolution (`instance_segmentation_trainer.py:145-176`), against
    ground truth masks rasterised at that resolution.
  - Objects whose mask vanishes when resized are removed from the test ground truth (`coco_dataset.py:224-227`), so
    they never count as missed.
  - There is no breakdown by object size and no recall figure, and the matching is the VOC one (audit L6).
- **Change:**
  - **`evaluation.original_resolution` for INS, the key the SEG task already has** (`segmentation_trainer.py:352-362`):
    - the boxes are restored with `restore_boxes_to_original_image`, unclipped for the paste (audit L22);
    - the masks are pasted at the original size with `paste_instance_masks`, in chunks as today;
    - the ground truth masks are decoded at the original size (`build_instance_mask` without the final resize), and
      every annotation is kept.

    To bound the memory, the overlap of a pair can be computed inside the union of its two boxes only.
  - **The AP of small, medium and large objects,** from the COCO `area` of the annotations (original pixels; the
    synthetic generator writes the visible pixels). These would be rows of the methods table, columns of
    `class_metrics.csv`, and scalars.
  - **The average recall at 100 detections,** which separates misses of the RPN and the score threshold from bad
    masks.
- **Cost and risk:**
  - **Cost:** medium.
  - **Risk:** a slower evaluation on large images.
- **Measure:** compare once with pycocotools on a few hundred COCO validation images (a test can do it when
  pycocotools is installed). After that, the AP of small objects tells whether the E20 / F26 work is needed.

### P4. Partial freezing of a pretrained backbone
- **Mask R-CNN:** `train(layers=...)` takes `heads`, `3+`, `4+`, `5+` or `all` (`model.py:2313-2322`). The samples
  first train the heads alone, then from stage 4 up, then everything at a tenth of the learning rate
  (`coco.py:498-521`, `nucleus.py:283-295`).
- **Here:** `Backbone.pretrained.freeze` fixes the whole loaded backbone or nothing, and `learning_rate_multiplier`
  gives it one rate. A staged schedule is possible as successive runs that transfer each other's components.
- **Change:**
  - Add `Backbone.pretrained.frozen_blocks`, the number of first blocks kept fixed (0 for none). This is the
    equivalent of `4+`: the low-level blocks of a backbone pretrained by a large classification run stay fixed while
    the upper blocks adapt to the masks.
  - Document the heads → upper blocks → all schedule in README as three successive runs, which needs no new
    mechanism.
  - The setting needs a run name token, next to `FRZ`.
- **Cost and risk:**
  - **Cost:** small. The optimiser groups, the BN statistics of the frozen blocks, checks and tests.
  - **Risk:** none when the value is 0.
- **Measure:** INS on a small dataset from a backbone pretrained by a CLS run, with `frozen_blocks` set to 0, 3 and 5.

### P5. Visual checks of the training targets and of the matches
- **Mask R-CNN:**
  - `inspect_model.ipynb` walks through the RPN targets (positive anchors with their refinement), the proposals before
    and after NMS, the sampled RoIs with their positive ratio (`draw_rois`, `visualize.py:208`), the refinement, the
    per-class NMS, and the 28 × 28 mask targets next to the predicted masks.
  - `display_differences` (`visualize.py:170-205`) draws the ground truth (green) and the detections (red) on one
    image, each detection captioned "score / IoU".
  - `plot_overlaps` (`visualize.py:320`) draws the grid of IoUs between ground truths and detections.
- **Here:** `view_input_images` shows the augmented inputs with their boxes and masks. The sample grids show the
  ground truth next to the detections, without the IoU of each detection. The detection debugging scripts were
  removed in `6225b4a`.
- **Change:**
  - **(a)** A `diagnosis_modes.view_training_targets` flag, shown like `view_input_images` on a training batch. It
    would draw:
    - the positive anchors and a sample of the negatives;
    - the top proposals after the filter;
    - the sampled RoIs, positives with their matched ground truth;
    - for a few positive RoIs, the mask target next to the predicted mask.

    Generalise `view_input_data` and reuse `visualise_anchors` rather than writing a new viewer. This makes failures
    such as audit H1 (proposals on plain background) or M5 (shifted mask targets) visible at a glance.
  - **(b)** In `build_detection_sample_grid` (`visualization.py:555`), caption every detection with its score and its
    mask IoU to the matched ground truth.
- **Cost and risk:**
  - **Cost:** (b) is small, (a) is medium.
  - **Risk:** none, since no number changes.

### P6. Ground truth masks cropped to their boxes
- **Mask R-CNN:**
  - `USE_MINI_MASK` and `MINI_MASK_SHAPE` (56 × 56) (`config.py:103-104`): each mask is cropped to its box and
    resized (`minimize_mask`, `utils.py:517-534`).
  - The mask target of a RoI is cropped from that small mask, after expressing the RoI in the coordinates of the
    ground truth box (`model.py:584-600`).
- **Here:** one `[N, S, S]` uint8 mask per object goes through the loader workers to the GPU. Audit L19 estimates
  about 4.7 GiB in flight at 1024 pixels, batch 40, 8 workers and prefetch 2.
- **Change:**
  - An option to load the masks as `[N, m, m]` crops with their boxes.
  - `build_mask_targets` would crop with RoIAlign in the coordinates of the crop (box offset and scale).
  - The flips and the random crop act exactly on a box-relative crop. The ±5° rotation needs the crop resampled in
    the rotated box.
  - The evaluation and the sample images paste the crops back with `paste_instance_masks`.
  - `m` would be configurable (56 in Mask R-CNN, 112 for large objects).
  - The other fix that L19 suggests (sending polygons or RLE and rasterising on the device) keeps the full precision.
- **Cost and risk:**
  - **Cost:** medium to high (dataset, augmentations, targets, evaluation).
  - **Risk:** a small loss of precision for large objects.
  - It is only worth doing if memory or loader throughput becomes the limit.

### P7. Tiles at native resolution for large images
- **Mask R-CNN:** the nucleus sample (`nucleus.py:127-133, 165, 173`) handles large images with small objects:
  - training on random 512 × 512 crops (`IMAGE_RESIZE_MODE: crop`) of the image scaled up by 2
    (`IMAGE_MIN_SCALE`);
  - inference on the whole image padded to a multiple of 64 (`pad64`);
  - anchors from 8 to 128 pixels, and up to 400 detections.
- **Here:** the SEG task has the same mechanism as `Data.tiling` (`scale`, `overlap`, `sliding_window.py:14-34`).
  DET and INS always resize the whole image to the network input, so objects of a few pixels in a large image vanish,
  and are removed from the ground truth (P3).
- **Change:** extend `Data.tiling` to DET and INS.
  - **Training:** samples are tiles cut at random places. Boxes and masks are cropped with them, and an object cut by
    the border keeps its visible part or is dropped below a visible fraction.
  - **Test images:** processed tile by tile with `compute_tile_starts`. The detections of all tiles are moved to image
    coordinates and merged with a class-wise NMS, and the masks are pasted into the full image.
  - **Export:** unchanged, one tile per request. The clients merge the tiles as the SEG clients already do.
- **Cost and risk:**
  - **Cost:** high.
  - **Risk:** objects cut by tile borders need care in the merge.
  - It is only worth doing for datasets of large images with small objects.

### P8. Boxes always taken from the masks
- **Mask R-CNN:** the dataset boxes are ignored, and every box is the tight box of its mask (`utils.py:34-57`,
  `model.py:1269`). Its README reports that about 2 % of COCO boxes differ from it by 1 pixel or more.
- **Here:** the annotation box scaled to the network input is used, and replaced by the tight box of the moved mask
  only after a geometric augmentation (`augmentations.py:199-211`). The same object therefore gets two slightly
  different box targets depending on whether its image was augmented: sub-pixel rounding, or more for RLE masks
  resized with nearest (audit M5).
- **Change:**
  - In INS training, take every box from its mask with `compute_mask_bounding_boxes`, which already exists, so that
    the box and mask targets always agree.
  - Keep the annotation boxes for the box AP of the test split, as COCO does.
- **Cost and risk:**
  - **Cost:** very small.
  - **Risk:** a small effect, mostly on box AP at the high thresholds.

### P9. Detections clipped to the image window before the mask head
- **Mask R-CNN:** `refine_detections_graph` clips the refined boxes to the image window, the letterbox without its
  padding (`model.py:711`), and the mask head then runs on those boxes (`model.py:2043-2052`).
- **Here:**
  - `postprocess_detections` clamps the boxes to the whole input, padding included (`postprocessing.py:54-55`).
  - With `keep_ratio`, a box can reach into the padding. Its mask is predicted over that box, and the box is clipped
    again when restored to the original image, which squeezes the mask (audit L22).
  - In the network-resolution evaluation, such boxes also lose IoU.
- **Change:**
  - Clamp to the letterbox window (`compute_letterbox_region`, `preprocessing.py:16`) before the mask head.
  - The trainer knows the original sizes from `labels.json`.
  - The exported model would need the windows as a second input (`[B, 4]`). Otherwise the clients keep the L22 fix:
    paste with the unclipped boxes.
- **Cost and risk:**
  - **Cost:** small in the trainer. Changing the export signature is the expensive part.
  - **Risk:** a small effect, limited to non-square images.

### P10. Datasets annotating different class subsets
- **Mask R-CNN:** `active_class_ids` (`model.py:1076-1109, 1271-1276`). When the images come from several datasets
  that each annotate a subset of the classes, the classification loss ignores the predictions of classes that the
  image's source does not annotate.
- **Here:** one label mapping for every image.
- **Change:** only if INS is ever trained on merged datasets. It would need a per-image list of annotated categories
  (a COCO extension, or one per source folder), with the classification loss masked accordingly.
- **Cost and risk:**
  - **Cost:** medium.
  - **Priority:** low until such a dataset exists.

## 6. Differences Not Worth Adopting
- **Class of a RoI = its argmax only** (`model.py:700-703`). Here every foreground class of every RoI is a candidate,
  which keeps the second-best class for the AP.
- **Negatives in proportion to the positives, so none without positives** (`model.py:556-558`). Here the budget is
  filled with negatives (the torchvision rule), which keeps training the background on images with few positives.
- **Images without objects skipped in training** (`model.py:1714-1715`). Here they train the RPN and the heads on
  negatives.
- **Masks restored by resizing to an integer box, then thresholding** (`utils.py:560-576`). `paste_instance_masks`
  samples at the pixel centres with sub-pixel boxes.
- **`crop_and_resize` with one sample per bin and boxes normalised with the (size − 1) convention**
  (`model.py:421-423`, `utils.py:853-867`). `roi_align` with `aligned=True` is the corrected operation.
- **Frozen batch normalization for batches of 2** (`config.py:208`). With frozen statistics, the head BN layers become
  plain affine layers. Here the batches hold 40 images, and `layer` normalization covers small batches.
- **Variable input sizes (`pad64`, `none`).** They are incompatible with the fixed-shape export, and P7's tiles cover
  large images instead.
- **Mean pixel subtraction.** `input_normalization` and `per_sample_normalization` cover it.
- **`ParallelModel` (multi-GPU).** Training runs on a single GB10.
- **imgaug.** The `DataAugmenter` already covers flips, crops, rotations and photometric changes on the GPU. The only
  gap with Mask R-CNN's mask-safe list (Affine, PiecewiseAffine) is zooming out and translation, since the random crop
  only zooms in. Suggestion E24 (copy-paste, usually combined with large scale jitter) is the related item.
- **The 2-class softmax objectness.** It is equivalent to the sigmoid used here.

## 7. Suggested Order
1. **Close the open audit items that Mask R-CNN confirms:** H1 (RPN background), M4 (NMS thresholds), L6 (COCO
   matching) and M12 (crowds, with Mask R-CNN's any-overlap rule). They change the DET and INS numbers, so they
   share one baseline regeneration.
2. **P3 (original resolution, object sizes, recall).** This is the measurement the architecture work needs,
   especially the AP of small objects.
3. **P1 (regression weights and loss weights) and P8 (boxes from masks).** Both are small changes.
4. **E20 (FPN with a finer level), E21 (level by size) and P2 (shared mask head).** Together they close the main
   architectural gap for the masks, guided by the small-object AP from P3.
5. **The rest when the need appears.** P5 when debugging, P4 when starting from a pretrained backbone on a small
   dataset, and P6, P7, P9 and P10 depending on the datasets.
