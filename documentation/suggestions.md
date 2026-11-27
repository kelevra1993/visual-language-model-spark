# Suggestions

> Proposed improvements that are not implemented yet, kept in `documentation/` so that we can come back to them and
> decide what to implement. They were checked against the code on 2026-10-15, 2026-11-13 and 2026-11-20; the findings of
> the repository audit of 2026-10-10 were all fixed or dropped by 2026-11-20. The items taken from Mask R-CNN come from
> a comparison with its Keras / TensorFlow 1 implementation (matterport, commit `3deaec5`) written on 2026-10-12: "Mask
> R-CNN" means that repository, and "the paper" He et al., 2017. Its file references (`model.py`, `utils.py`,
> `config.py`, `coco.py`, `nucleus.py`, `visualize.py`) are those of `mrcnn/` and of its samples. An item is removed
> from this list once it is implemented, or as soon as we decide not to do it. Since 2026-11-23 nothing is added to
> this list: it only shrinks, a new idea going to the report of the work that found it, and the file is deleted once
> empty.

## Table Of Content
- [A. Detector architecture](#a-detector-architecture)
- [B. Instance segmentation](#b-instance-segmentation)
- [C. Evaluation and deployment](#c-evaluation-and-deployment)
- [F. Reference: Mask R-CNN compared stage by stage](#f-reference-mask-r-cnn-compared-stage-by-stage)
- [G. Keypoint detection](#g-keypoint-detection)
- [H. Classification](#h-classification)
- [J. Self-supervised counted shapes: tests left to run](#j-self-supervised-counted-shapes-tests-left-to-run)
- [Suggested Order](#suggested-order)

## A. Detector architecture
1. **A transferable neck**: the feature pyramid and path aggregation of this item are implemented (`Neck.type`),
   but heads trained after a neck are refused by `compare_detection_heads`, the neck not being
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
   - **For the masks, dropped:** stage 6 of `instance_segmentation_experiment_plan.md`, which tested this rule for the
     mask heads, was dropped by the user on 2026-11-19, the image branch of design A (stage 3b) cropping every box
     from the image; the rule stays open for the detection heads.
39. **Spatial pyramid pooling (SPP) after the backbone**: the last feature map concatenated with its max pools of 5, 9
    and 13, widening the receptive field of the deepest level at little cost (YOLOv4).

## B. Instance segmentation
7. **Region proposal and detection heads shared by the levels**, as the mask heads now are:
   - **Mask R-CNN:** `build_fpn_mask_graph` is built once (`model.py:956-1005`) and receives the RoIs of every level
     through `PyramidROIAlign`. The box head and the region proposal head are shared the same way
     (`model.py:1939-1944`).
   - **Here:** `MaskHead.shared: True` gives one mask head read by every level, each box pooled at the scale of its
     level (no per-level 1×1 convolution, so it needs equal level channels). Stage 5 of
     `instance_segmentation_experiment_plan.md` tied with per-level heads on the score (+0.0050 at 10,000, inside the
     launch gap) while leading at every evaluation, gaining 0.0169 on the large objects and cutting the mask branch
     of the export by 3.3, and the user kept it on 2026-11-21. The region proposal and detection heads are still one
     per level.
   - **Change:** the same switch for `RegionProposal` and `Detector`, one head trained on the anchors or proposals of
     every level, whose export runs once instead of once per level.
   - **Measure:** box mAP overall and by object size, and the export throughput, as stage 5 measured the masks.
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
     box can reach into the padding: its mask is predicted over that box, part of it over the padding, and the clients
     restore the box unclipped (`clip=False`) so that the mask is not squeezed. In the network-resolution
     evaluation, such boxes also lose IoU.
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
40. **TensorRT FP16 checked at export**: `export_experiment` (`export_model.py`) checks every export with ONNX
    Runtime on the CPU in FP32 only, so an FP16 overflow or precision loss of the TensorRT engine is seen only by
    measuring the test metric through Triton, as `scripts/deployment/evaluate_served_formats.py` does for the models
    of the serving benchmark (it found the FP16 overflow of the NMS exported in the detectors, fixed on 2026-11-17).
    A Triton-against-local output comparison in `run_inference.py`, or a CPU check of the intermediate values of the
    graph against 65,504, the largest FP16 value, would catch it for every task. The engine cache, which could serve
    the engines of older weights, is named after the exported files since 2026-11-20
    (`build_tensorrt_engine_cache_prefix`).
41. **Keypoint and sequence models served with TensorRT**: on 2026-11-22, the tutorial models exported to ONNX and
    served by Triton 26.01 with the TensorRT `config.pbtxt` that `export_model.py` writes by default (FP32, maximum
    batch 16, dynamic batching) gave two failures, the five image models (CLS, DET, INS, SEG, SSL) serving batches of
    1 and 16 in both precisions:
    - **Keypoint detector** (`keypoint_focal_synthetic/experiments/KPT_..._MC33`): Triton crashed (signal 11 in
      `libonnxruntime.so`, inside `TRITONBACKEND_ModelInstanceExecute`) at the first batch-1 request, in FP32 and in
      FP16, about 19 seconds after the engine was built and written to the cache; the same `model.onnx` served by ONNX
      Runtime without TensorRT answered at batch 1 and 16. Until the cause is found, a keypoint export is written with
      `tensorrt_precision = None` (keypoint workflow, deployment chapter 2). To find it: serve the graph cut after the
      heatmaps, then after the candidate extraction (`TopK` and the gathers of the candidates), and try a newer image.
    - **The four sequence models** (M2O, M2M, SEL and CMP of `DEFAULT_EXPERIMENT_FOLDERS`): their batch-1 engine is
      built and answers, but the engine of the largest batch sent (6, 16, 16 and 3, the largest below 1 GB) is not:
      the request fails with "onnx runtime error 11", the log giving `IBuilder::buildSerializedNetwork: Error Code 10:
      Could not find any implementation for node {ForeignNode[node_Reshape_...node_softmax]}` with the 4 GB TensorRT
      workspace of `config.pbtxt`. In FP32 for all four, measured again alone on the idle machine; in FP16 for M2O
      only, M2M, SEL and CMP building and serving their larger batch. The serving benchmark served this M2M with
      TensorRT FP32 at batch 16 on 2026-11-17, so an export change since (the dynamic output frames of
      `restore_dynamic_output_frames`, the stripped metadata) or the larger batch request are the first suspects:
      export the run with the code of 2026-11-17, compare the graphs, then try a larger `max_workspace_size_bytes`. A
      served sequence model fails as soon as the dynamic batcher forms a batch above the one its engine was built for.
    `scripts/deployment/check_export_size_estimates.py --build_tensorrt_engines` reproduces both (the models are
    reported in red and the other engines still measured).

## F. Reference: Mask R-CNN compared stage by stage
**The mask branch is the same design.** Both use RoIAlign 14 × 14, a stack of 3 × 3 convolutions, a 2 × 2 transposed
convolution to 28 × 28 and one sigmoid mask per class. The loss is a binary cross entropy on the channel of the ground
truth class, over the positive proposals only, and at inference the masks are predicted on the final refined boxes. The
differences lie in what feeds the mask branch (resolution and context of the levels, item 2, the finer inputs of the
mask heads, off by default, and the neck, `none` in the INS configurations) and in how the
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
| Feature pyramid | FPN, P6 for the region proposals only (`model.py:1902-1925`). | `Neck` (item 1): a feature pyramid in the DET configurations, none in the INS ones (the levels keeping the channels of their blocks), or path aggregation, to `output_channels`, without P6. |
| Anchors | One scale per level (32 on P2 to 512 on P6), ratios 0.5 / 1 / 2, corner-aligned (`utils.py:583-619`). | Several scales and ratios per level (`DetectionLevels.<level>.anchors`), corner-aligned, the cell centres having tied with the corners in stage 8 of the detection benchmark. |
| Region proposal head | One head shared by every level: 3 × 3 conv of 512, then 1 × 1 convs for a 2-class softmax and the offsets (`model.py:830-871`). | One head per level (`DetectionLevels.<level>.region_proposal`): `ConvolutionBlock`, 3 × 3 conv, then 1 × 1 convs for a sigmoid objectness and the offsets. |
| Region proposal targets | Positive at IoU ≥ 0.7, plus the best anchor(s) of every ground truth. Negative below 0.3. Anchors touching a crowd are never negatives. 256 sampled, at most half positive (`model.py:1445-1553`). | Positive at IoU ≥ 0.5, plus the best anchor(s). Negative below 0.3, 0 included in INS; from 0.1 in DET (stage 2), the anchors overlapping no object being ignored. 256 sampled, at most 128 positive. |
| Proposal filter | One top-k (6000) and one NMS (0.7) over all levels, 2000 kept in training and 1000 at inference, padded with zeros (`model.py:255-329`). | INS: one top-k (1000 / 500) and one NMS (0.7) over all levels (`proposal_selection: "joint"`). DET: the top-k and the NMS of every level, the proposals clamped to an empty box left out (`"per_level"`, stage 7). 500 / 250 kept, padded with copies of random survivors in training and evaluation (masked slots in deployment), at most one per anchor. |
| RoI sampling | 200 RoIs, 33 % positive (IoU ≥ 0.5). Negatives below 0.5 that touch no crowd, in proportion to the positives, so none without positives (`model.py:486-619`). The ground truth boxes are left out on purpose, a comment citing a sampling study that found no gain (`model.py:1317-1318`). | The ground truth boxes of every image are added to its proposals at the level of their best anchor (`add_ground_truth_proposals`, as in torchvision and Detectron2). 128 RoIs, up to 64 positive (IoU ≥ 0.5), the negatives filling the rest of the budget. |
| RoI pooling | `PyramidROIAlign`: level `4 + round(log2(√(wh) / 224))`, clamped to P2–P5, then `crop_and_resize`, one bilinear sample per bin (`model.py:344-447`). | `roi_align` with `aligned=True` and adaptive sampling, on the level of the anchor that produced the proposal (item 2). |
| Box head | Shared: 7 × 7 conv to 1024 + BN, 1 × 1 conv 1024 + BN, then class and per-class box layers (`model.py:900-953`). | Per level (`DetectionLevels.<level>.detector`): `ConvolutionBlock`, flattened, fully connected layers without normalization, then class and per-class box layers. |
| Box regression | Targets divided by [0.1, 0.1, 0.2, 0.2] on both stages. Smooth L1 with β = 1, averaged over the coordinates of the positives (`model.py:575, 1047-1073, 1112-1140`). | Targets left unscaled, fixed weights (1, 1, 1, 1) on both stages (`BOX_REGRESSION_WEIGHTS`), (10, 10, 5, 5) giving about 2.5 times lower mAP@0.5 on the synthetic shapes. Smooth L1 with β = 1/9, summed per box and divided by the positives, or 1 - CIoU on the detection heads (`Detector.box_loss`, off). |
| Mask head | Shared: 4 × (3 × 3 conv 256 + BN + ReLU), 2 × 2 transposed conv 256, 1 × 1 conv to one sigmoid per class (`model.py:956-1005`). | Per level (`DetectionLevels.<level>.mask_head`), or one head read by every level with `MaskHead.shared` (item 7): `ConvolutionBlock` of `[layers, channels]` with module options, configurable pool size, mask resolution and upsampling, optional finer inputs (backbone skips, an image branch on the crop of the input image, fusion convolutions), 1 × 1 conv per class. |
| Mask targets | `crop_and_resize` of the matched mask (full or mini) inside the positive RoI, rounded at 0.5 (`model.py:577-606`). | `roi_align` (aligned) of the matched full-resolution mask inside the positive proposal, thresholded at 0.5. |
| Mask loss | BCE of the ground truth class channel, mean over the positives (`model.py:1143-1179`). | Same, computed in float32. |
| Postprocessing | Class = argmax of each RoI. Score ≥ 0.7, per-class NMS 0.3, top 100. Boxes clipped to the image window, padding excluded (`model.py:684-779`). | Every foreground class of every RoI is a candidate. Score ≥ 0.5 (0.05 in evaluation), batched per-class NMS 0.5 (greedy in INS, DIoU-NMS with the exponent 0.6 in DET), fixed top 100, boxes clamped to the whole input, padding included (item 9). |
| Masks at inference | Mask head on the final boxes. Each 28 × 28 mask is resized to its integer box in the original image, then thresholded at 0.5 (`model.py:2417-2480`, `utils.py:560-576`). | Mask head on the final boxes. `paste_instance_masks` samples at pixel centres with sub-pixel boxes. The INS evaluation pastes them at the original resolution with `evaluation.original_resolution` (at the network resolution otherwise), and the clients paste at the original resolution. |
| Ground truth boxes | Always the tight box of the mask, the dataset boxes being ignored (`model.py:1269`). | The same: the tight box of the mask at the network resolution, the INS annotations needing no `bbox`. |
| Ground truth masks in memory | Optional 56 × 56 crops in the box (`config.py:103-104`, `utils.py:517-552`). | One `[N, S, S]` uint8 mask per object at the network resolution (item 8), or one `[S, S]` uint8 map of object numbers per image with `Data.instance_index_maps`, expanded on the GPU. |
| Crowds | Kept with a negative class id, kept out of the negatives, ignored by pycocotools. | Ordinary objects, learnt and scored like the others. |
| Objects lost by the resizing | Skipped at original resolution when under 1 pixel (`coco.py:248-250`). Masks emptied by the resize are dropped from the training targets (`model.py:1263-1265`). The evaluation keeps them, since it runs on the original images. | INS masks emptied by the resize are dropped from the training and validation targets, and from the test ground truth unless `evaluation.original_resolution` keeps them, so that they count as missed. |
| Augmentation | imgaug pipeline with a list of mask-safe augmenters (flips, crop-and-pad, Affine, PiecewiseAffine; `model.py:1232-1259`). COCO uses horizontal flips only, nucleus flips, 90° rotations, brightness and blur. | `DataAugmenter` on the GPU: crop, aspect crop, zoom-out, translation, flips, ±5° and 90° rotations, eighteen photometric types, masks moved and boxes recomputed, fragments of objects removed. |
| Optimisation | SGD momentum 0.9, learning rate 0.001. Each gradient tensor clipped at norm 5 (Keras `clipnorm`). L2 1e-4 except the BN scales and offsets. Weight per loss (`config.py:177-211`, `model.py:2153-2199`). | Adam / AdamW / SGD Nesterov, warm-up, cosine, no gradient clipping. One weight decay for every parameter. One weight per loss (`Optimization.loss_weights`), 1 in every configuration file, the weights of 1 kept after a test. |
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
| Strides of the head levels | 4, 8, 16, 32 (region proposals also 64) | 32, 64, 64 | The finer inputs of the mask heads read the image and the finer blocks |
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
    - **Measured redundancy:** feature correlation of the ImageNet-1k Medium run (`configuration_medium_model.yaml`,
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
    - **Experiment:** the Small model with and without the term, for the same number of iterations, compared on the
      test top-1 and on the statistics above, with a weight giving the term about a tenth of the cross entropy at the
      start, turned on after a warm-up of the cross entropy as in the VQA result.

    Cost and risk: small to medium. The Medium run underfits (iterations 110000 to 115000: training accuracy 0.599 with
    the augmentation and the dropout, validation accuracy 0.613), and the papers report regularisation-sized gains, so
    the term may lower the top-1 as well as raise it. Decorrelated features also stay dependent in non-linear ways.

## J. Self-supervised counted shapes: tests left to run
The synthetic counted shapes dataset (`/home/robert_kelevra/Data/self_supervised_data/synthetic_counted_shapes`,
500,000 training, 10,000 validation and 10,000 test images of 5 to 10 shapes at 512 pixels, drawn like the sequence
labelling frames) and its two configurations (`self_supervised/configuration_counted_shapes_simclr.yaml` and
`configuration_counted_shapes_barlow_twins.yaml`, the backbone and flattened feature map of the launches of
`sequence_labelling_experiment_plan.md`) were written, but no run has been trained on them, and the dataset folder
was empty on 2026-11-20: the dataset is to be generated again first (`--task=self_supervised` of
`create_synthetic_dataset.py`). The goal is to see whether the self-supervised backbone learns not only how many shapes
an image holds but also which ones.

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

## Suggested Order
1. **When the need appears:** item 12 when debugging, item 11 when deploying a class with its own threshold,
   item 40 before serving a model in FP16, items 8, 9 and 10 depending on the datasets, items 2 and 39 for the
   detection, and item 1 when a neck has to be transferred.
