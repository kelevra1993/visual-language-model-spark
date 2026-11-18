# Suggestions

> Proposed improvements of the object detection (DET) and instance segmentation (INS) tasks that are not implemented
> yet, kept in `documentation/` so that we can come back to them and decide what to implement. They were checked
> against the code on 2026-10-15, and again on 2026-11-13, together with the findings of the
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
> - AdamW / warm-up / cosine, mixed precision (the gradient clipping added with them was removed on 2026-11-11,
>   no run having used it);
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
> - region proposal negatives down to IoU 0 (audit H1), the DET configurations back at 0.1 since 2026-11-12, kept by
>   stage 2 of the detection benchmark, without warning;
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
> - the ImageNet-22k download, which replaced the Tencent ML-Images downloader on 2026-11-06, the only ImageNet script
>   since 2026-11-11 (`imagenet_download.py`: every shard in train, one held-out image per class as validation and
>   test; the ImageNet-1k download and the merge of both datasets removed);
> - the neck of DET and INS (`Neck.type`: `feature_pyramid`, `path_aggregation`, items 1 and 37), on 2026-11-07, the
>   feature pyramid kept by stage 3 on 2026-11-08 and set in the object detection configurations on 2026-11-12
>   (`none` in the instance segmentation ones), path aggregation tied with it by stage 4 on 2026-11-09 and kept in the
>   code for projects with very small objects;
> - a CIoU box loss for the detection heads (`Detector.box_loss: "complete_iou"`, item 36), below the smooth L1 in
>   stage 5 (0.7974 against 0.8079) and off in every configuration, `RegionProposal.box_loss` removed on 2026-11-10;
> - DIoU-NMS of the final detections (item 38), kept by stage 5b with the exponent 0.6 and set in the object
>   detection configurations on 2026-11-10, the exports running it too;
> - the region proposals selected per level (`RegionProposal.proposal_selection: "per_level"`, item 3 and audit M3),
>   the proposals clamped to an empty box left out (audit L7), kept by stage 7 (+0.0104) and set in the object
>   detection configurations on 2026-11-12, the instance segmentation ones keeping `"joint"`;
> - the audit items C3, C4, H2, H3, M1, M2, M4, M6, M8, M14, M15, M17, M20, L2, L5, L10, L13 and L18, fixed since
>   2026-10-10 (M7 gone with `resume_training`), recorded in the table of `bug_standards.md`; H6, M5, M9, M13, L7, L17,
>   L19 and L22 keep only their open part below.
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
>   shortcuts of the channel-changing residual blocks (audit H4), dropped on 2026-11-05;
> - the anchors centred on the cell centres (audit L1), tied with the cell corners in stage 8 of the detection
>   benchmark (0.7757 against 0.7765 at 8500) and removed on 2026-11-13;
> - the float32 against bfloat16 comparison of the COCO keypoint training (item 15), no longer possible since
>   `Optimization.mixed_precision` was removed, every task training under the bfloat16 autocast.
> - the fix of `run_benchmarking_loop` (audit L14), the loop being removed on 2026-11-14: only a commented line of
>   `main.py` called it, and it crashed at its first step with every shipped configuration.
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
1. **A transferable neck**: the feature pyramid and path aggregation of this item are implemented (`Neck.type`, see
   the list above), but heads trained after a neck are refused by `compare_detection_heads`, the neck not being
   transferable. A `Neck.pretrained` block would need entries in `TRANSFERABLE_COMPONENTS` (appended last, its order
   setting the optimizer groups), `PRETRAINED_COMPONENT_NAMES`, `PRETRAINED_TOKEN_PREFIXES` and `COMPONENT_NAMES`.
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
39. **Spatial pyramid pooling (SPP) after the backbone**: the last feature map concatenated with its max pools of 5, 9
    and 13, widening the receptive field of the deepest level at little cost (YOLOv4).

## B. Instance segmentation
5. **Sharper masks for small objects**: the mask heads pool every box from the backbone feature map of its level
   (RoIAlign 14 × 14, upsampled to 28 × 28), never from the image, so a small object on a coarse level (blocks 5–7
   of the examples, strides 32, 64 and 64 at 1024 pixels) is interpolated from a handful of feature cells and gets
   blurry borders: a 32-pixel object, the smallest anchor, spans one feature cell of its level. Mask R-CNN pools from
   P2–P5, at strides 4 to 32, with the context of the FPN. In order of cost:
   - an earlier, finer level in `DetectionLevels` (more anchors and heads to train);
   - the feature pyramid (`Neck.type: "feature_pyramid"`, set in the object detection configurations, `none` in the
     instance segmentation ones), whose finest level gets the context of the deeper ones (the stride 4 level of
     Mask R-CNN);
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
     single head once on all the detections. Without a neck (`Neck.type: "none"`, the instance segmentation
     configurations), the levels still differ in depth and context, so this works best with the feature pyramid. The
     detection heads could be shared the same way.
   - **Cost and risk:** medium: a run name token, and the model transfer must refuse to load per-level heads into a
     shared one. A shared head may fit the coarse levels worse than dedicated heads do, which is the point of
     measuring it.
   - **Measure:** mask mAP overall and by object size, and the export throughput.
8. **Ground truth masks cropped to their boxes ("mini masks")**:
   - **Mask R-CNN:** `USE_MINI_MASK` and `MINI_MASK_SHAPE` (56 × 56) (`config.py:103-104`): each mask is cropped to its
     box and resized (`minimize_mask`, `utils.py:517-534`), and the mask target of a RoI is cropped from that small
     mask, after expressing the RoI in the coordinates of the ground truth box (`model.py:584-600`).
   - **Here:** one `[N, S, S]` uint8 mask per object goes through the loader workers to the GPU. Audit L19 estimates
     about 4.7 GiB in flight at 1024 pixels, batch 40, 8 workers and prefetch 2. Since 2026-11-13,
     `Data.instance_index_maps` sends one `[S, S]` map per image instead when the masks never overlap (the synthetic
     datasets), expanded on the GPU; the mini masks remain the option for overlapping masks (COCO).
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
      removed in `6225b4a`, and `visualise_anchors` on 2026-11-14 (it had no caller).
    - **Change:** (a) a `diagnosis_modes.view_training_targets` flag, shown like `view_input_images` on a training
      batch, drawing the positive anchors and a sample of the negatives, the top proposals after the filter, the
      sampled RoIs (the added ground truth boxes included) with their matched ground truth, and for a few positive
      RoIs the mask target next to the predicted mask, generalising `view_input_data`.
      This makes failures such as proposals on plain background or shifted mask targets visible at a glance. (b) In
      `build_detection_sample_grid`, add to the caption of every detection (its class and score) its IoU to the
      matched ground truth.
    - **Cost and risk:** (b) is small, (a) medium. No number changes.

## E. Mask R-CNN on the open audit items
What Mask R-CNN does for the items of the [repository audit](#repository-audit-for-instance-segmentation) that are still open:
- **M10 (weight decay):** Mask R-CNN skips the BN scales and offsets but still decays the biases
  (`model.py:2179-2183`). It also divides each L2 term by the tensor size, an unusual scaling not to copy.
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
differences lie in what feeds the mask branch (resolution and context of the levels, items 2 and 5, and the neck,
`none` in the INS configurations) and in how the
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
| Feature pyramid | FPN, P6 for the region proposals only (`model.py:1902-1925`). | `Neck` (items 1 and 37): a feature pyramid in the DET configurations, none in the INS ones (the levels keeping the channels of their blocks), or path aggregation, to `output_channels`, without P6. |
| Anchors | One scale per level (32 on P2 to 512 on P6), ratios 0.5 / 1 / 2, corner-aligned (`utils.py:583-619`). | Several scales and ratios per level (`DetectionLevels.<level>.anchors`), corner-aligned, the cell centres having tied with the corners in stage 8 of the detection benchmark. |
| Region proposal head | One head shared by every level: 3 × 3 conv of 512, then 1 × 1 convs for a 2-class softmax and the offsets (`model.py:830-871`). | One head per level (`DetectionLevels.<level>.region_proposal`): `ConvolutionBlock`, 3 × 3 conv, then 1 × 1 convs for a sigmoid objectness and the offsets. |
| Region proposal targets | Positive at IoU ≥ 0.7, plus the best anchor(s) of every ground truth. Negative below 0.3. Anchors touching a crowd are never negatives. 256 sampled, at most half positive (`model.py:1445-1553`). | Positive at IoU ≥ 0.5, plus the best anchor(s). Negative below 0.3, 0 included in INS; from 0.1 in DET (stage 2), the anchors overlapping no object being ignored. 256 sampled, at most 128 positive. |
| Proposal filter | One top-k (6000) and one NMS (0.7) over all levels, 2000 kept in training and 1000 at inference, padded with zeros (`model.py:255-329`). | INS: one top-k (1000 / 500) and one NMS (0.7) over all levels (`proposal_selection: "joint"`). DET: the top-k and the NMS of every level, the proposals clamped to an empty box left out (`"per_level"`, stage 7). 500 / 250 kept, padded with copies of random survivors in training and evaluation (masked slots in deployment), at most one per anchor. |
| RoI sampling | 200 RoIs, 33 % positive (IoU ≥ 0.5). Negatives below 0.5 that touch no crowd, in proportion to the positives, so none without positives (`model.py:486-619`). The ground truth boxes are left out on purpose, a comment citing a sampling study that found no gain (`model.py:1317-1318`). | The ground truth boxes of every image are added to its proposals at the level of their best anchor (`add_ground_truth_proposals`, as in torchvision and Detectron2). 128 RoIs, up to 64 positive (IoU ≥ 0.5), the negatives filling the rest of the budget. |
| RoI pooling | `PyramidROIAlign`: level `4 + round(log2(√(wh) / 224))`, clamped to P2–P5, then `crop_and_resize`, one bilinear sample per bin (`model.py:344-447`). | `roi_align` with `aligned=True` and adaptive sampling, on the level of the anchor that produced the proposal (item 2). |
| Box head | Shared: 7 × 7 conv to 1024 + BN, 1 × 1 conv 1024 + BN, then class and per-class box layers (`model.py:900-953`). | Per level (`DetectionLevels.<level>.detector`): `ConvolutionBlock`, flattened, fully connected layers without normalization, then class and per-class box layers. |
| Box regression | Targets divided by [0.1, 0.1, 0.2, 0.2] on both stages. Smooth L1 with β = 1, averaged over the coordinates of the positives (`model.py:575, 1047-1073, 1112-1140`). | Targets left unscaled, fixed weights (1, 1, 1, 1) on both stages (`BOX_REGRESSION_WEIGHTS`), (10, 10, 5, 5) giving about 2.5 times lower mAP@0.5 on the synthetic shapes. Smooth L1 with β = 1/9, summed per box and divided by the positives, or 1 - CIoU on the detection heads (`Detector.box_loss`, item 36, off). |
| Mask head | Shared: 4 × (3 × 3 conv 256 + BN + ReLU), 2 × 2 transposed conv 256, 1 × 1 conv to one sigmoid per class (`model.py:956-1005`). | Per level (`DetectionLevels.<level>.mask_head`, item 7): `ConvolutionBlock` of `[layers, channels]` with module options, configurable pool size, mask resolution and upsampling, 1 × 1 conv per class. |
| Mask targets | `crop_and_resize` of the matched mask (full or mini) inside the positive RoI, rounded at 0.5 (`model.py:577-606`). | `roi_align` (aligned) of the matched full-resolution mask inside the positive proposal, thresholded at 0.5. |
| Mask loss | BCE of the ground truth class channel, mean over the positives (`model.py:1143-1179`). | Same, computed in float32. |
| Postprocessing | Class = argmax of each RoI. Score ≥ 0.7, per-class NMS 0.3, top 100. Boxes clipped to the image window, padding excluded (`model.py:684-779`). | Every foreground class of every RoI is a candidate. Score ≥ 0.5 (0.05 in evaluation), batched per-class NMS 0.5 (greedy in INS, DIoU-NMS with the exponent 0.6 in DET, item 38), fixed top 100, boxes clamped to the whole input, padding included (item 9). |
| Masks at inference | Mask head on the final boxes. Each 28 × 28 mask is resized to its integer box in the original image, then thresholded at 0.5 (`model.py:2417-2480`, `utils.py:560-576`). | Mask head on the final boxes. `paste_instance_masks` samples at pixel centres with sub-pixel boxes. The INS evaluation pastes them at the original resolution with `evaluation.original_resolution` (at the network resolution otherwise), and the clients paste at the original resolution. |
| Ground truth boxes | Always the tight box of the mask, the dataset boxes being ignored (`model.py:1269`). | The same: the tight box of the mask at the network resolution, the INS annotations needing no `bbox`. |
| Ground truth masks in memory | Optional 56 × 56 crops in the box (`config.py:103-104`, `utils.py:517-552`). | One `[N, S, S]` uint8 mask per object at the network resolution (item 8, audit L19), or one `[S, S]` uint8 map of object numbers per image with `Data.instance_index_maps`, expanded on the GPU. |
| Crowds | Kept with a negative class id, kept out of the negatives, ignored by pycocotools (audit M12). | Ordinary objects, learnt and scored like the others. |
| Objects lost by the resizing | Skipped at original resolution when under 1 pixel (`coco.py:248-250`). Masks emptied by the resize are dropped from the training targets (`model.py:1263-1265`). The evaluation keeps them, since it runs on the original images. | INS masks emptied by the resize are dropped from the training and validation targets, and from the test ground truth unless `evaluation.original_resolution` keeps them, so that they count as missed. |
| Augmentation | imgaug pipeline with a list of mask-safe augmenters (flips, crop-and-pad, Affine, PiecewiseAffine; `model.py:1232-1259`). COCO uses horizontal flips only, nucleus flips, 90° rotations, brightness and blur. | `DataAugmenter` on the GPU: crop, aspect crop, zoom-out, translation, flips, ±5° and 90° rotations, eighteen photometric types, masks moved and boxes recomputed, fragments of objects removed. |
| Optimisation | SGD momentum 0.9, learning rate 0.001. Each gradient tensor clipped at norm 5 (Keras `clipnorm`). L2 1e-4 except the BN scales and offsets. Weight per loss (`config.py:177-211`, `model.py:2153-2199`). | Adam / AdamW / SGD Nesterov, warm-up, cosine, no gradient clipping. One weight decay for every parameter (audit M10). One weight per loss (`Optimization.loss_weights`), 1 in every configuration file, the weights of 1 kept after a test. |
| Training schedule | Stages selected by layer regular expressions: `heads`, `3+`, `4+`, `5+`, `all` (`model.py:2313-2322`, `coco.py:498-521`). | One stage per run. A loaded component can be frozen whole or given a learning rate multiplier; a schedule needs successive runs. |
| Batch normalization | `TRAIN_BN: False` by default: the stored statistics are used and never updated (batches of 2). | Trained, with the momentum fixed at 0.1 (`BATCH_NORMALIZATION_MOMENTUM`). `layer` normalization is available for small batches. |
| Evaluation | `compute_ap` with mask IoU and COCO matching (`utils.py:656-751`). COCO sample: pycocotools `COCOeval` for boxes and masks at the original resolution (`coco.py:342-385`). | Own mAP (area, 11 and 101 points) at 0.5, 0.75 and 0.5:0.95 with VOC matching (kept, audit L6), mean average recall, both by object size, box and mask results per class, proposal recall, at every checkpoint, at the network resolution or, for INS, at the original resolution (`evaluation.original_resolution`). |
| Visual checks | Notebooks walking through every stage, and `display_differences` (item 12). | `view_input_images`, sample grids and the task workflow notes (item 12). |
| Deployment | Keras weights only, batch size fixed at build time. | TorchScript / ONNX exports with fixed shapes, served by Triton, with a comparison check. |

Default hyperparameters of Mask R-CNN (`config.py`) against `instance_segmentation/configuration_example.yaml` (the
synthetic shapes configuration uses the same values at 512 pixels, except two anchor scales per level, with a smaller
backbone and heads and a batch of 64; the object detection configurations differ since the detection benchmark: feature
pyramid, background IoU minimum 0.1, proposal selection per level, DIoU-NMS 0.6 and learning rate 1e-3):

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
| Gradient clipping | Norm 5 per gradient tensor | None | Mask R-CNN always clips, and its README suspects that clipping is what keeps the higher learning rate of the paper stable. Not tried: `Optimization.gradient_clipping_norm` was removed on 2026-11-11, no run having used it. |
| Batch | 2 images per GPU, BN frozen | 40, BN trained | |
| Loss weights | 1 each, configurable | 1 each, configurable (`Optimization.loss_weights`) | Same; tested, the weights of 1 kept |

## G. Keypoint detection
14. **Keypoint names next to the exported model**: the export of a KPT run writes no channel names. The clients
    rebuild the channel table with `build_keypoint_channel_table` from the `keypoint_categories` that
    `load_experiment_configuration` reads from the `labels.json` of the training data, so a client without the
    training data cannot name the channels. `export_model.py` could write a `keypoint_channels.json` (class, keypoint
    name and skeleton of every channel) next to the exported files, read by the clients instead of the training labels.
    Cost and risk: small.
16. **Ignored regions, experiment left to run** (`KeypointHead.ignore_unlabelled_objects`, implemented): as in
    HigherHRNet, the heatmap loss leaves out the cells at least half covered by the objects left out of the labels
    (the COCO crowds and the persons without any labelled keypoint), read from their `segmentation`. On in the COCO
    `configuration_example.yaml`, off in the synthetic one, whose shapes have no `segmentation`.
    - **Left:** the OKS AP of a COCO run with and without the setting, the crowds and the unlabelled persons being
      4652 of the 11004 persons of the validation split.
17. **Finer peaks, measure on COCO left to run** (`compute_peak_offsets`, implemented): every keypoint candidate is
    placed within its cell, along each axis, at the vertex of the parabola through the log-heatmap at its cell and at
    its two neighbours (the Taylor expansion of DARK without its cross derivative), instead of a quarter of a cell
    towards its higher neighbour.
    - **Measured** on 200 test images of the synthetic run (`keypoint_focal_synthetic`, iteration 6000, moving
      average, stride 2, 4290 matched keypoints): the mean error per axis is 0.363 pixel instead of 0.458 on the
      coordinates off the pixel centres (-21 %), and 0.293 instead of 0.127 on the pixel centres, where the quarter
      shift was exact. 59 % of the synthetic coordinates are pixel centres, while the resized COCO coordinates are
      continuous.
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
    - **Setting:** `ClassificationHead.decorrelation` (`activated`, `weight`, `hidden_layer`), off in every
      configuration file, tracked as `decorrelation_loss` and `hidden_feature_correlation`. It gives no run name
      token, so the warm-up of the cross entropy is a run resumed with it turned on. The second hidden layer before
      its ReLU, where the redundancy is highest, is the first candidate.
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
normalised vectors. The clusters follow the videos (every `not_empty` cluster holds a single video). The runs
`ficha-experiments-empty-bins` and `ficha-experiments-compactor-to-front`, with their embeddings and their folders of
kept images, have since been deleted (none on disk on 2026-11-13), so these items first need the embeddings of an
existing run, such as a 384 pixel run of `ficha-experiments-staged-schedule`; the figures below are those of the deleted
runs.

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
20. **The same selection on the compactor dataset**: the compactor dataset, whose errors gathered in a few videos with
    the deleted 512 pixel run (iteration 800, `ficha-experiments-compactor-to-front`), embedded with the 384 pixel
    compactor run of `ficha-experiments-staged-schedule` and clustered the same way, then judged as in item 19.
21. **Representatives of the large clusters**: the largest `not_empty` cluster of block 5 is the whole video
    `video_t0180_2026-09-30_08-50-49` (2,666 frames). Its frames closest to the centre show the same full bin, while
    its farthest frame shows a bin being tipped in, which the 5 representatives closest to the centre miss.
    - **Options:** `cluster_selection_method: "leaf"`, already accepted by the script (smaller clusters, the moments of
      a video in clusters of their own), representatives spread over the ranks of a cluster (the closest one, then
      images evenly spaced in distance up to the farthest), or a number of representatives growing with the size of
      the cluster (e.g. its square root).
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
`configuration_counted_shapes_barlow_twins.yaml`, the backbone and flattened feature map of the launches of
`sequence_labelling_experiment_plan.md`) are ready, but no run has been trained on them yet. The goal is to see
whether the self-supervised backbone learns not only how many shapes an image holds but also which ones.

24. **The two self-supervised trainings**: SimCLR and Barlow Twins trained until the k-NN accuracy of the shape count
    (6 classes, `5_shapes` to `10_shapes`) stops rising, the input viewer checking the two views first.
    - **Watch:** `Embedding Standard Deviation` staying away from 0 (no collapse), and the positive pair accuracy of
      SimCLR not reaching 1 long before the k-NN accuracy rises, which would show a cue shared by the two views.
25. **Probe of which shapes an image holds**: the k-NN evaluation only reads the shape count, the label of every
    sample. The categories of its shapes are in `shape_category_ids` of `labels.json`, for a probe trained on the
    frozen representation: a multi-label linear classifier of the 35 shape classes, its macro F1 against the same probe
    on a backbone with random weights and on a supervised backbone.
26. **Initialisation of sequence labelling**: the self-supervised backbone given to the sequence labelling runs of the
    shape change experiment (`Backbone.pretrained`, the same architecture as their backbone), against the same runs from
    random weights, compared on the test Macro F1 of the averaged weights at the same number of iterations.
27. **Strength of the views**: `random_crop` is off in both configurations, so that both views keep every shape and
    the same count, with flips and small rotations. Turning it on (20 to 100 % of the area for the self-supervised
    views) makes the task harder but can change the number of shapes between the two views; to compare on the probes
    of items 25 and 26 rather than on the k-NN accuracy of the count. The harder views of 2026-10-20 add, among
    others, a random erasing of 2 to 15 % of the image (probability 0.25) and a plasma shadow darkening parts of it
    (probability 0.2), which can also hide a shape of one view and change its count: to compare with
    `random_erasing` and `random_plasma_shadow` off on the same probes.

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
    sample shows the captions and the X of the wrong frames but its right frames are all `no_change`. The run has no
    class weights: the sequence labelling experiment plan saw a launch without them stay at the score of a model
    predicting only no_change, and kept balanced class weights at this run's own learning rate, 1e-3
    (`configuration_example.yaml`). The task needs balanced class weights and a longer training before its sample is
    taken again.
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
1. **The instance segmentation stages of `instance_segmentation_experiment_plan.md`:** the shared mask head (item 7),
   the level by box size (item 2) and a finer map for the masks (item 5). Together they close the main architectural
   gap for the masks, guided by the AP of the small objects and the proposal recall of every level, before any tuning
   of the anchors or of the proposal counts. The detection stages are all done.
2. **The rest when the need appears:** item 12 when debugging, item 11 when deploying a class with its own threshold,
   items 8, 9 and 10 depending on the datasets, item 39 for the detection, and item 1 when a neck has to be
   transferred.

## Repository Audit For Instance Segmentation

> Audit of the whole framework, carried out on 2026-10-10 at commit `84ac325`. Every line of `vision_training/` was
> read, split into six areas: data pipeline; detection and instance segmentation; backbone, classification and
> segmentation; sequence tasks; training infrastructure; configuration and deployment. Each finding was checked against
> the code. Most were reproduced with small scripts, or with short trainings on the synthetic datasets or the dummy
> task. The critical and high findings were checked a second time by hand.
>
> **Open items only:** the items fixed or dropped since were removed on 2026-11-13 (see the list at the top of this
> file and the table of `bug_standards.md`). The line numbers are those of `84ac325`, except in the items updated on
> 2026-11-13 (H6, M5, M9, L15, L17, L22).
>
> **Confidence:** every finding is VERIFIED (code paths read end to end, or reproduced) unless it is marked PLAUSIBLE
> (the mechanism is confirmed but the size of the effect depends on the data).
>
> **Severity:** **critical** = stalls or corrupts training, or silently mixes runs. **high** = clearly limits accuracy
> or loses training time. **medium** = limits accuracy in some settings or gives misleading results. **low** = edge
> cases and documentation.

### Summary By Task

| Task | Findings |
|---|---|
| All tasks (training loop, backbone, data, configuration) | C1, H5, H6, M9, M10, L15–L17 |
| DET | M13, M16, M21, L7, L8 |
| INS | Everything DET has, plus M5 (augmented masks), L19, L22 |
| CLS | C2, L3, L4, L9 |
| SEG | C2, M5, M13, L20 |
| M2O, M2M, CMP, SEL | C2, M18 (CMP), L11 (M2M) |

---

### Critical

#### C1. A NaN or Inf loss is never detected, and every later weight and checkpoint becomes NaN
- **Where:**
  - `vision_training/trainers/base_trainer.py:1274-1275`, where the total loss is summed.
  - `base_trainer.py:1027-1028`, where `training_loss.backward()` and `step_optimizer` run with no check.
  - No `isfinite` / `isnan` appears anywhere in `trainers/`.
- **Effect (reproduced):** a NaN loss at step 3 of a 10-iteration run turns every weight into NaN.
  - Checkpoints 4, 8 and 10 are NaN, and so is the registry's `last` (the one resume loads). The run ends FINISHED with
    no error line; the console only shows `nan`.
  - The shipped configurations run for `number_iterations: 1e10`, so one bad batch on day 1 ruins a run left going for
    a week, and resuming reloads the NaN weights.
- **Fix:**
  - Check `torch.isfinite(total_loss)` in `run_model_iteration`. The `.item()` of the metric tracker already
    synchronises, so the check costs nothing extra.
  - On a non-finite loss, zero the gradients, skip `backward` and `step`, count the skipped step and print it.
  - Raise an error the loop does not swallow after N consecutive non-finite losses.
  - Compute the norm of all the gradients together, skip the step when it is not finite and log the norm to
    TensorBoard.
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

---

### High

#### H5. A failed save or evaluation (e.g. disk full) is swallowed: no checkpoint, no evaluation
- **Where:**
  - `base_trainer.py:1002-1009`: the save and evaluation sit inside the per-iteration `try`.
  - `base_trainer.py:1072-1089`: `except Exception ... continue`.
  - Since 2026-11-14 a failed save removes its temporary weight or registry file (`checkpointing.py`).
- **Effect (reproduced):** `ENOSPC` from iteration 4 of a 30-iteration run.
  - Only 17 optimizer steps out of 30 (each failed save skipped its step, the save coming before the step until M20
    was fixed), and 13 orphan `model_*.pt.tmp` files, no longer left.
  - No evaluation at all, yet the run ends FINISHED, because the consecutive-error counter resets after every good
    iteration.
  - Runs keep every checkpoint (3.5–6.7 GB per run already), so once the disk is full the run "trains" with no
    checkpoint. The final save at Ctrl+C fails too, losing all the progress since the disk filled.
- **Fix:**
  - Move the save and evaluation out of the swallowing `try`, so that an `OSError` there ends the run with status
    FAILED and a traceback.
  - Add a setting keeping only the last N intermediate checkpoints.

#### H6. An unreadable image drops its batch without limit, and makes every test evaluation fail
- **Where:** `base_trainer.py` (`get_next_batch` and the evaluation of the weight saving iterations). Since 2026-10-21
  `read_image` raises a `ValueError` naming the file, and every fetch error is printed and its dropped batch counted.
- **Effect:**
  - Only 10 failed fetches in a row stop the training: with 1 % unreadable images and batches of 64, about 47 % of
    the batches are dropped at every epoch while the training goes on.
  - In the test split, one unreadable image makes every test evaluation fail, each counted as a failed iteration, so
    no test metrics are produced.
- **Fix:**
  - Stop above a dropped-batch ratio.
  - Optionally check at dataset construction that every listed file exists.

---

### Medium

#### M5. Masks resized with plain nearest are shifted half a pixel down and to the right of their images
- **Where:** (the RLE instance masks use `cv2.INTER_NEAREST_EXACT` since the audit)
  - `segmentation_dataset.py`: `cv2.INTER_NEAREST` for the SEG masks and tiles.
  - `augmentations.py`: `crop_and_resize` (`random_crop`, `random_aspect_crop`) and `random_zoom_out` resize the
    masks with the default nearest of torchvision (`tv_tensors.Mask`).
  - `preprocessing.py`: the class maps restored to the original image with `mode="nearest"`.
- **What is wrong:** legacy nearest picks `floor(dst · scale)`, which is not centre-aligned, while the images are
  resized bilinearly and the boxes by an exact affine.
- **Effect (reproduced):**
  - Mask minus image centroid is +0.40 to +0.49 px after resizing, about 0.01 px with `INTER_NEAREST_EXACT`.
  - `random_crop` adds +0.5 px, so augmented SEG samples are off by about 1 px.
  - At `original_resolution: True` a 4096 → 512 shift is about 3.5 original pixels, more than the default
    `boundary_tolerance: 2`.
- **Fix:** use `cv2.INTER_NEAREST_EXACT` in the SEG dataset, `InterpolationMode.NEAREST_EXACT` for the masks of the
  crops and the zoom-out, and `mode="nearest-exact"` for the restored class maps.

#### M9. Training-step errors are swallowed as long as they are not consecutive
- **Where:** `base_trainer.py`, the `except Exception` of the training loop: the counter of consecutive errors resets
  after every good iteration. The traceback of every distinct error is printed once (`report_iteration_error`), and
  the final save after the error limit names the last iteration that stepped.
- **Effect (reproduced):** an error on every other forward pass gives a FINISHED run with half the optimizer steps,
  e.g. an OOM on large detection batches, or a bug that only some inputs trigger.
- **Fix:** abort on an error rate over a rolling window, and report the skipped iterations in the summary.

#### M10. Weight decay applies to every parameter, biases and normalization scales included (latent)
- **Where:** `base_trainer.py:431-440`: one `weight_decay` for every group.
- **Effect:** none today (`weight_decay: 0.0` everywhere). As soon as AdamW with 0.05 is used, the batch and layer norm
  scales are pulled towards 0 and the biases are decayed, which costs accuracy on long runs.
- **Fix:** split every group into decayed and non-decayed parameters (`ndim <= 1` get `weight_decay=0`). This changes
  the group layout, which is acceptable since no run uses weight decay yet.

#### M13. The decoded image size is never compared with the annotation size (EXIF orientation, resized copies) — PLAUSIBLE
- **Where:**
  - `coco_dataset.py:139` and `171` use the decoded shape and ignore the JSON width and height.
  - RLE masks are letterboxed with their own `size` (`coco_masks.py:215-217`), checked against the image only at the
    original resolution and by the instance index maps.
  - SEG masks are read with `IMREAD_UNCHANGED`, images with `IMREAD_COLOR`.
- **Effect:**
  - OpenCV's `IMREAD_COLOR` applies the EXIF orientation (checked), `IMREAD_UNCHANGED` does not. Boxes annotated on the
    stored pixels land on a rotated image.
  - A SEG mask of a JPEG with orientation 3 passes the shape check and is upside down relative to its image.
  - Images downscaled after annotation give boxes off by the scale factor.
- **Fix:** compare `image.shape[:2]` with the JSON size (and the RLE `size`) and raise on a mismatch. Choose one EXIF
  policy for the images (`read_image`, shared by the training and the deployment) and the SEG masks.

#### M16. The DET and INS export comparison probably compares only padding — PLAUSIBLE
- **Where:** `export_specifications.py:278-302` and `333-355`: random-noise inputs, with the deployment
  `score_threshold` (0.5) applied.
- **Effect:** a trained detector very likely returns 0 detections on noise, so the comparison passes whatever the box
  decoding, the NMS or the mask path do.
- **Fix:** compare with `score_threshold = 0`, or compare the raw head outputs and the proposals, and assert that
  `number_detections > 0`.

#### M18. CMP member frames are ordered as text — PLAUSIBLE
- **Where:** `data/sequence/pair_list_manifests.py:58-59` uses `get_image_paths`, which sorts the paths as strings
  (`utilities/os_utilities.py:182`).
- **Effect (reproduced on names):**
  - `frame_1, 2, 3, 9, 10, 11, 12` come out as `[1, 10, 11, 12, 2, 3, 9]`, and the evenly spaced pick of 4 frames
    keeps `[1, 11, 2, 9]`.
  - Any dataset with unpadded frame numbers trains the order-sensitive temporal modules (the GRU of the example) on
    scrambled sequences.
- **Fix:** sort with a natural key (digit runs compared as integers), or refuse unpadded numbered names.

#### M21. TensorRT FP16 is never validated, and the engine cache may serve old weights — PLAUSIBLE
- **Where:**
  - `export_experiment` (`export_model.py`) checks the export only with ONNX Runtime on the CPU in FP32.
  - `triton_configuration.py:87-94` enables `trt_engine_cache_enable` with the fixed path `/tmp/tensorrt_engine_cache`.
- **Effect:**
  - FP16 overflow or precision loss is never detected.
  - If the cache key does not cover the weights (not confirmed), re-exporting another checkpoint of the same
    architecture serves the old engine after a container restart.
- **Fix:** version the cache path per export (iteration and hash), and add a Triton-against-local output comparison
  to `run_inference.py`.

---

### Low

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

#### L7. Zero-area proposals of the joint selection and zero-area detections are never removed
- **Where:** `region_proposer.py:217-220` (TODO), and the padding by duplication at `262-278`.
- **Effect:**
  - A proposal regressed out of the image becomes zero-area after clamping, and torchvision NMS keeps identical
    zero-area boxes.
  - In the training-model eval path (the test mAP), padding can duplicate them into false-positive detections.
- **Fix:** mask boxes narrower or shorter than 1e-3 px after clamping, in both filters and in the postprocessing.
- **Status (2026-11-12):** `RegionProposal.proposal_selection: "per_level"`, in the object detection configurations
  since stage 7, leaves out the proposals narrower or shorter than 1e-3 px, `"joint"` keeping them for its runs. On
  the 15,000 checkpoint of run_9 (300 test images) no kept proposal and no final detection was empty (41 empty
  candidates of 1,200,000 per level in the training counts, none kept), so the final detections get no filter.

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

#### L11. M2M sequences or batches with no labelled frame give a NaN loss and inflate the exact match
- **Where:** `sequence_losses.py:29-31` (mean reduction with every target ignored), `sequence_metrics.py`
  (`compute_sequence_exact_matches`).
- **Effect (reproduced):**
  - The tracked interval loss becomes NaN. The gradients stay finite, but if NaN is the first value the "Best" marker
    stops updating.
  - Sequences without labels count as exact matches.
- **Fix:** sum the losses and divide by `clamp(number of labelled frames, 1)`. Drop these sequences from training and
  from the exact match.

#### L15. Loader workers are forked from a multi-threaded process — PLAUSIBLE deadlock
- **Where:** `base_trainer.py:984-986`.
  - The TensorBoard writer already runs a thread before `iter(train_dataloader)`.
  - The persistent test workers are forked at the first evaluation, after MLflow started its thread (the training and
    validation workers are now forked before it).
  - Python 3.13 warns about it.
- **Fix:** `multiprocessing_context="forkserver"` for the loaders, or create every loader iterator before any thread
  starts. Correct the comment.

#### L16. A host–GPU synchronisation between forward and backward at every step — PLAUSIBLE throughput loss
- **Where:** `metric_tracking.py:124-127` and `157-158`: `.item()` per metric before `backward()`.
- **Fix:** keep the interval sums as detached GPU tensors, and call `.item()` only at the summaries.

#### L17. Ragged last batch and averages over batches
- **Where:** `loaders.py` (`drop_last` only for the self-supervised loaders and the batch-normalized classification
  heads, the other training loaders keeping their ragged last batch), `metric_tracking.py` (means over batches).
- **Effect:**
  - A short last batch weighs as much as a full one in the interval means.
  - A 1-sample batch reaching 1×1 feature maps makes batch norm raise, and that step is skipped silently (M9).
- **Fix:** `drop_last=True` for every training loader, and weight the means by the sample count.

#### L19. INS sends full-resolution object masks through shared memory — PLAUSIBLE, FIXED FOR DISJOINT MASKS (2026-11-13)
- **Where:** `coco_dataset.py:214-222`: one uint8 `[N, S, S]` array per image.
- **Effect:** at S = 1024, batch 40, 8 workers and prefetch 2, about 4.7 GiB of masks are in flight. On the unified
  memory of the GB10 this competes with the GPU, and a small `/dev/shm` kills workers.
- **Fix:** send polygons, RLE or box-cropped masks from the workers and rasterise on the device, or pack the bits.
- **Done for disjoint masks:** `Data.instance_index_maps` sends one uint8 map of object numbers per image, expanded on
  the device by `prepare_batch` (a ninth of the bytes on the instance segmentation benchmark dataset). Overlapping
  masks (COCO) still go one per object.

#### L20. GPU memory peaks in the original-resolution SEG evaluation — PLAUSIBLE
- **Where:** `segmentation_trainer.py:352-362`: a float32 `[K, H, W]` probability map per image at the original
  resolution.
- **Effect:** a 6000×4000 image with 20 classes needs about 1.9 GB per image.
- **Fix:** interpolate in class chunks with a running max and argmax, or in bfloat16.

#### L22. The INS client recipe squeezes masks at the image border — PLAUSIBLE
- **Where:** `documentation/reference/tools_reference.md` (Export, INS): the client brings the boxes back with
  `restore_boxes_to_original_image`, which clips them to the image by default, while the M×M mask was predicted over
  the unclipped box. The evaluation at the original resolution already pastes into the unclipped boxes (`clip=False`).
- **Fix:** say in the recipe to restore the boxes with `clip=False` for the pasting (`paste_instance_masks` handles
  parts outside the image), and to clip them afterwards for the reported boxes.

---

### Statements Contradicting The Code

| Where | Statement | Reality |
|---|---|---|
| README "Stopping And Resuming", `build_run_name` docstring | Any change other than the listed settings gives another run name | The IoU and NMS thresholds, the score thresholds and the sample counts give no token either |
| Every YAML with `residual_nets` | "Shortcuts are added every two convolution layers" | No shortcut in the 2-layer channel-changing blocks (projection shortcuts of H4 dropped on 2026-11-05) |
| `RegionProposal.foreground_iou_threshold.max` (YAML) | "Maximum IoU limit for matching" | Never read by the model, only by the checker |
| `mean_average_precision.py`, `average_precision_report.py` | "COCO" mAP@[0.5:0.95] and 101-point values | VOC matching, kept on purpose (L6 dropped) |
| `region_proposer.py` | aspect ratio = width / height | `anchors.py` and the YAMLs use height / width |
| `preprocessing.py`, `deployment_utilities.py` | restore functions "used by the inference clients" | No inference client of the repository calls them; the evaluations do |
| README CLS run name | `..._HAVG.128_MC4` | No shipped configuration gives it (item 33) |
| `sequence_classification_trainer.py:66-72` | Frame shuffle warning for every transformer | A non-causal transformer without positional embedding (the README's recommendation for frame sets) cannot see the order |
| `base_trainer.py` | MLflow opened after the loader workers are forked | True of the training and validation workers, not of the test workers (L15) |

Robustness gaps of the configuration loader: `int(float(...))` silently truncates non-integer counts; unknown or
misspelled keys are silently ignored, only the removed settings and the augmentation type names being refused.

---

### Checked And Correct

- **Training loop:**
  - bfloat16 autocast without a GradScaler is correct, and `zero_grad` comes before the forward pass.
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

1. **Training safety, all tasks:** C1 (NaN guard), H5 and H6 (swallowed save and evaluation, dropped batches), M9
   (error handling). These protect long runs and change no model.
2. **Correctness of the targets:** C2 (smoothing with weights), M5 (exact nearest resizing), M13 (image sizes against
   the annotations), L8 (boxes of the samples without augmentation), L7 (empty proposals of the joint selection of INS).
3. **Evaluation and deployment:** M16 (export check), M21 (TensorRT), L3, L22 and the statements of the last table.
