# Instance Segmentation Workflow

This document follows one instance segmentation run (INS) from `main()` to the evaluation of a checkpoint, to show which function runs when, what it receives, what it returns and who uses the result next. The **Call Tree Overview** gives the whole run on one page. Each stage then has one section, with one step per function: a short explanation followed by a **real** output. The steps shared by every task (configuration loading, checkpoints, training loop, MLflow tracking, standalone evaluation) are explained in detail in `classification_workflow.md`, and the steps of the two-stage detector (anchors, region proposals, target assignment, detection heads, post-processing, box mean average precision, proposal recall) in `object_detection_workflow.md`; here they keep a short explanation and the real output of the instance segmentation run. The steps specific to the instance segmentation (instance masks loaded from the COCO segmentations, mask heads, mask targets and mask loss, masks of the final detections, mask overlaps and mask mean average precision) are explained in full.

The examples of Stages 1 and 2 and of the steps of one batch come from real runs on CPU, so that the GPU stayed free for another training. The synthetic dataset of the configuration, `synthetic_shapes`, was not on the machine when they were recorded, so a small one was generated in a temporary folder with `create_synthetic_dataset(task="instance_segmentation")`: 100 training, 20 validation and 20 test images of 512 pixels, at most 6 shapes per image, 20 shape classes, every object segmented by an uncompressed run length encoding (RLE) of its visible pixels. The data folders and the mapping file of the configuration were redirected to it, read like every split by the PyTorch loader of `CocoDetectionDataset`.
- **Stages 1 and 2**: `vision_training/configurations/instance_segmentation/configuration_synthetic_shapes.yaml` as it was then (512 pixel images, batch size 40, detection and mask heads on the blocks 5, 6 and 7 of the backbone; the backbone, levels and heads of `configuration_example.yaml`, which runs at 1024 pixels), with the run folder redirected to a temporary folder.
- **Stages 3, 4 and 5, the outputs of the run** (the iteration table, curves, MLflow run, checkpoints, console summaries, evaluation reports, files and sample images, TensorBoard and MLflow scalars, standalone evaluation of the checkpoints on the first 200 test images, Step 5.4 excepted): the 5000-iteration run of [`run_configurations/instance_segmentation_workflow_run.yaml`](run_configurations/instance_segmentation_workflow_run.yaml), trained on the GPU on 2026-10-24 with the code of that day on the full synthetic dataset of that day (128,000 / 20,000 / 20,000 images, 20 classes; the dataset was generated again on 2026-11-18 with 35 classes, textured backgrounds in half the images, low contrast shapes and shapes from 16 pixels, cut by the border and occluded, which `Read This First` describes with the INS run of `experiments/readme_screenshots`). This configuration is the synthetic configuration launched by `Run It` with four changes: 5000 iterations instead of 2000, a checkpoint evaluated every 500 iterations instead of 250 (an intermediate one every 250 instead of 125), the MLflow tracking turned on, and its own experiment folder `experiments/workflow_runs`. It also keeps the augmentation probabilities the synthetic configuration had then (0.3 for most types, among them `random_haze`, the only type turned on), which are 0.25 now (0.5 for the flips). It also predates `Data.instance_index_maps`: the run loaded one mask per object, which the setting gives with `False`.
- **Stages 3 and 4, the steps of one batch** (the tensors of Steps 3.6 to 3.13, 3.15 and 3.16, and of Steps 4.1 to 4.5 inside the test batches): the configuration of Stages 1 and 2 shortened to batch size 2, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4, a validation batch every 10, 2 loading workers, with the MLflow tracking turned on, run on CPU. The model, the image size and the masks are those of Stages 1 and 2. After 12 iterations this model finds no shape, so its values only show the shapes and the order of the steps.

> The examples of the CPU runs were recorded before the following changes, which they do not show (the outputs of the 5000-iteration run, trained on 2026-10-24, show the first four; the last two came later and leave them unchanged, the neck of type `"none"` building no layer and the maps expanding into the same masks):
> - the region proposal NMS is 0.7 and the detector NMS 0.5 (they were 0.5 and 0.8), and the anchors overlapping no object (IoU 0) can be sampled as background (`RegionProposal.background_iou_threshold.min: 0.0`);
> - the evaluation shows the mask and box `Mean Average Recall` tables after the size tables, and can score the boxes and masks at the original resolution (`evaluation.original_resolution`, `False` in the synthetic configuration); its tables, scalars and `class_metrics.csv` also have a column at IoU 0.95 (`mAP@0.95`, `AP@0.95`, `average_precision_95` and `mask_average_precision_95`, `351a4d1`), so the scalar and column counts quoted below are those of that time;
> - the mask overlaps paste 16 detection masks at a time for a 1024 x 1024 image, the chunk growing as the image shrinks (`MASK_OVERLAP_REFERENCE_PIXELS`): 64 at 512 pixels, where the examples of Step 4.5 paste 16;
> - the synthetic configuration keeps a moving average of the weights (`Optimization.exponential_moving_average`), saved in the checkpoints and evaluated after the trained weights in `Evaluation/Iteration_<n>/EMA/`, `Tensorboard/Test_EMA/` and an `EMA Weights` block of `Results.txt`, and `Optimization.loss_weights` weighs the five losses (1 each);
> - the configurations have the `Neck` section of the detection (`type: "none"`, no layer, the outputs unchanged), among the model sections of Steps 1.2 and 1.3 and printed in a `Neck Architecture` panel by the summary;
> - the synthetic configuration loads one instance index map per image (`Data.instance_index_maps: True`, Step 2.9) instead of one mask per object, so its loader batches hold `instance_index_maps` [B, S, S] uint8 where the examples show lists of `instance_masks`; `prepare_batch()` expands the maps on the device into the same `instance_masks` (Step 3.7), which every later step reads; the compressed run length encodings are decoded with numpy (`decode_compressed_counts()`), same run lengths, faster.
>
> The synthetic configuration trains on the full synthetic dataset with blocks of 16 and 32 filters, two anchor scales per level and smaller heads (mask heads `[3, 32]`), batch size 64 and 2000 iterations: What A Run Writes and the Pitfalls quote the INS run of `experiments/readme_screenshots`, which adds the settings listed there, and Stages 3 to 5 show the outputs of its 5000-iteration copy. The detector sections have since gained settings that every output of this document runs with by default: `RegionProposal.proposal_selection: "joint"` (one top-k and one NMS of the proposals over all the levels), `Detector.box_loss: "smooth_l1"` and `Detector.nms_criterion: "iou"` (the greedy final NMS, `nms_distance_exponent` being read by `"distance_iou"` only). The shipped instance segmentation configurations keep these values, with no neck, a learning rate of 2e-4, `RegionProposal.background_iou_threshold.min: 0.0` and the mask heads `[3, 32]`, although the instance segmentation benchmark ran with the detection settings named next and kept the mask heads `[4, 64]` (stage 3) and the image branch of design A (stage 3b, chosen on 2026-11-19, `documentation/architecture_benchmarking/instance_segmentation_experiment_plan.md`); the object detection configurations, which ship the winners of the detection benchmark, select the proposals per level (`"per_level"`), suppress with DIoU-NMS (`"distance_iou"`, exponent 0.6) and train with the feature pyramid (`Neck.type: "feature_pyramid"`), a learning rate of 1e-3 and a background minimum of 0.1.

## Read This First
The instance segmenter learns to find every object of an image with its box, class and score, like the detector of DET, and the pixel mask of every detection. It is the two-stage detector of `object_detection_workflow.md` (backbone, anchors, region proposal and detection heads of every level of `DetectionLevels`) with one mask head per level, in the style of Mask R-CNN without feature pyramid by default (`Neck.type: "none"`; with `feature_pyramid` or `path_aggregation` every head reads the merged levels): the mask head of the level of a detection pools its refined box and predicts one 28 x 28 mask per class, of which the channel of the detected class is kept. The input is a COCO split whose objects have a `segmentation` (a run length encoding or polygons), the box of every object being taken from its mask; the output is the detections of DET with the mask probabilities of every detection in the frame of its box, pasted into the image by the evaluation and by the clients. Choose INS when the pixels of every object matter (touching or overlapping objects, their areas or outlines); DET when boxes are enough, without mask annotations; SEG (`semantic_segmentation_workflow.md`) when the class of every pixel is enough at the resolution of the image and the objects of a class need not be told apart; KPT for keypoints.

### The Task At A Glance
Shapes of the synthetic configuration (512 pixels, batch size 64, 35 classes and the background, R the positive proposals of the batch). They were checked on the CPU with the last checkpoint of a run of this configuration on the dataset of that time (20 classes); the class axis follows the 35 classes of the dataset generated on 2026-11-18.

```text
<dataset>/{train,validation,test}/data/*.png + labels.json (COCO segmentation: RLE or polygons) + label_mapping.json
   │ CocoDetectionDataset(load_instance_masks=True, load_instance_index_maps=True): one instance index map at 512
   │ per image (build_instance_index_map), box of every object from its pixels in the map
   ▼
sample  images [3, 512, 512] uint8 BGR · instance_index_maps [512, 512] uint8 · bounding_boxes [N, 4] · labels [N]
batch   images [64, 3, 512, 512] · instance_index_maps [64, 512, 512] · boxes, labels: lists of 64
   ▼ prepare_batch: maps expanded on the device into instance_masks [N, 512, 512] per image · DataAugmenter moves them
   ▼
Backbone ─► block 5 [64, 32, 16, 16] · block 6 [64, 32, 8, 8] · block 7 [64, 32, 8, 8]      the detector of DET
   ▼ RegionProposal per level: 2304 anchors, RPN losses; joint selection: 500 proposals per image (250 in evaluation)
   ▼ training: + ground truth boxes, 128 proposals per image sampled, at most 64 positive (IoU >= 0.5)
DetectionHead per level: RoIAlign 7x7 ─► logits [64, 128, 36] · offsets [64, 128, 36, 4] ─► detection losses
MaskHead of the level of every positive proposal: RoIAlign 14x14 ─► 3 conv x 32 ─► ConvTranspose2d 2x2 ─► 28x28
   ─► 1x1 conv ─► logits [R, 36, 28, 28] ─► mask_loss: BCE of the channel of the ground truth class against the
   matched ground truth mask cropped to the proposal by RoIAlign 28x28 (aligned), thresholded at 0.5
   ▼ evaluation: postprocess_detections (score >= 0.05, 0.5 deployed, greedy NMS 0.5, top 100), mask heads on the boxes
final_detection_boxes [B, 100, 4] · _scores · _labels · final_detection_masks [B, 100, 28, 28], detected class
   ▼ paste_instance_masks into 512 x 512 at the boxes, >= mask_threshold 0.5, mask IoU with the ground truth masks
evaluate_detections on the masks, then on the boxes ─► mAP@0.5 / 0.75 / 0.95 / [0.5:0.95] · by size · mAR · per class
   (AP, recall at 0.5 and 0.75 with its 95 % interval)
   headline Mask mAP@0.5/area · proposal recall · class thresholds and confusion matrix on the masks
<run>/anchor_recall.csv, Results.txt, Tensorboard/{Train, Test, Test_EMA}, Evaluation/Iteration_<n>/{class_metrics.csv
(box and mask_ columns), class_thresholds.json, confusion_matrix.csv/.png, proposal_recall.csv, Samples/, EMA/}
```

### Files To Know
| Role | File | Main functions or classes |
|---|---|---|
| Trainer | `vision_training/trainers/instance_segmentation_trainer.py` (bases `detection_trainer.py`, `coco_split_trainer.py`, `base_trainer.py`) | `InstanceSegmentationTrainer`: `prepare_batch`, `augment_batch`, `forward_pass`, `gather_image_detections`, `compute_mask_overlaps`, `compute_evaluation_results`, `get_sample_masks` |
| Dataset and masks | `vision_training/data/detection/coco_dataset.py`, `coco_masks.py` | `CocoDetectionDataset` (`load_instance_masks`, `load_instance_index_maps`, `load_original_instances`, `add_instance_masks`, `add_instance_index_map`), `build_instance_mask`, `build_instance_index_map`, `find_index_map_objects`, `expand_instance_index_maps`, `decode_run_length_encoding`, `rasterize_polygons`, `compute_mask_bounding_boxes` |
| Model | `vision_training/models/instance_segmentation/instance_segmenter.py` (base `models/detection/two_stage_detector.py`) | `InstanceSegmenter`: `forward`, `compute_proposal_mask_loss`, `predict_detection_masks`, `print_mask_summary` |
| Mask head | `vision_training/models/instance_segmentation/mask_head.py` | `MaskHead`, `count_mask_upsamplings` |
| Mask targets, pasting and IoU | `vision_training/models/instance_segmentation/mask_operations.py` | `build_mask_targets`, `paste_instance_masks`, `build_interpolation_weights`, `compute_mask_intersection_over_union`, `cut_mask_segments`, `count_segment_pixels` |
| Mask loss | `vision_training/losses/instance_segmentation_losses.py` | `compute_mask_loss` (the four detection losses come from `detection_losses.py`) |
| Detector | the files of the `Read This First` of `object_detection_workflow.md` | anchors, proposals, targets, detection heads, post-processing, proposal recall |
| Metrics | `vision_training/evaluation/detection/average_precision_report.py`, `mean_average_precision.py` | `summarise_average_precisions` (`overlap_type="masks"` then `"boxes"`), `evaluate_detections`, `gather_class_detections`, `compute_clopper_pearson_intervals` and `build_recall_interval_table` of `vision_training/evaluation/recall_intervals.py` |
| Configuration checks and run name | `vision_training/configuration_processing/instance_segmentation_checks.py`, `run_name.py` | `check_instance_segmentation_configuration`, `check_mask_head_configuration`, `build_instance_segmentation_sections` (`MRES-MBN-MUTC-M14.3x32-MR28`) |
| Export | `vision_training/models/instance_segmentation/inference_instance_segmenter.py`, `vision_training/deployment/export_specifications.py` | `InferenceInstanceSegmenter.predict_masks`, `get_export_specification` (the outputs of DET then `detection_masks`), `compare_instance_segmentations`; `build_input_contract`, `write_onnx_input_contract` and `write_preprocessing_file` of `vision_training/deployment/input_contract.py` |

### Run It
1. Generate the dataset: in the main block of `vision_training/data/synthetic/create_synthetic_dataset.py`, set `task = "instance_segmentation"` (its `output_folders`, `image_sizes`, `split_sizes` and `maximum_objects` entries give `/home/robert_kelevra/Data/instance_segmentation_data/synthetic_shapes`, 512 pixels, 128,000 / 20,000 / 20,000 images and 5 to 6 shapes, every object segmented by a compressed run length encoding of its visible pixels), then run it. Should you need the dataset of the instance segmentation benchmark instead, set `instance_segmentation_benchmark_dataset = True`, which takes the settings of `instance_segmentation_benchmark_settings`: `/home/robert_kelevra/Data/instance_segmentation_data/synthetic_hard_shapes`, 1024 pixels, 512,000 / 50,000 / 10,000 images of 5 to 12 shapes, drawn like the images of the object detection benchmark but with a test split of 10,000 images for shorter evaluations; it is on this machine (35 classes, its test split cut to its first 10,000 images, 84,922 objects). Both draw their shapes like every synthetic dataset, from 16 pixels to 80 % of the side, some in low contrast, cut by the border or on textured backgrounds (see the [tasks reference](../reference/tasks_reference.md#synthetic-datasets)), with the 35 shapes of `SHAPE_CATEGORIES` (run names ending in `MC35`); the 512 pixel dataset on this machine was generated again on 2026-11-18, and the runs shown below were trained with 20 classes (`MC20`).
```bash
uv run python -m vision_training.data.synthetic.create_synthetic_dataset
```
2. Train, after setting `experiment_folder` (and the data folders and mapping file if the dataset is elsewhere) in the configuration:
```bash
uv run python -m vision_training.main --configuration_path="vision_training/configurations/instance_segmentation/configuration_synthetic_shapes.yaml"
```
3. Evaluate a checkpoint again: set `run_name`, `experiment_folder`, `iteration` (`None` for the last checkpoint, a number, or `"all"`), `test_data_folder`, `batch_size`, `track_evaluation` and `maximum_error_images_per_class_pair` in the main block of `scripts/evaluation/evaluate_checkpoint.py` (Stage 5), and if needed `target_precisions`, `nms_criterion` and `nms_distance_exponent` (`None` keeps those of the run), then run it; it writes in `<run>/Checkpoint_Evaluation/`, or in `<run>/Checkpoint_Evaluation_nms_<criterion>[_<exponent>]/` with another NMS:
```bash
uv run python -m scripts.evaluation.evaluate_checkpoint
```

### Configuration Keys To Know First
Before changing the synthetic configuration, these are the keys that change the mask heads, the detector or the evaluation the most.

| Key | Value in the synthetic configuration | Effect |
|---|---|---|
| `MaskHead.mask_resolution` | 28 | side of the masks of every level, the `MR28` token of the run name and the `[D, 28, 28]` of the exported `detection_masks` |
| `DetectionLevels.<5, 6, 7>.mask_head` | `roi_align_pool_size: 14`, `convolutions: [3, 32]` | RoIAlign and convolution block of the mask head of every level (`M14.3x32`), the pool doubled once to 28 (`count_mask_upsamplings()`); the pool size must not exceed the mask resolution; on the benchmark dataset, the pool of 14 with the resolution of 28 beat a pool of 22 (resolution 44) and a resolution of 56 (stages 1 and 2 of `documentation/architecture_benchmarking/instance_segmentation_experiment_plan.md`), and the heads `[4, 64]` beat `[3, 32]` by 2.62 points of mask mAP@[0.5:0.95] with the averaged weights (stage 3), not written into the shipped configurations |
| `MaskHead.upsampling` | `"transposed"` | a 2 x 2 transposed convolution of stride 2 per doubling (`MUTC`), or `"bilinear"`, a bilinear doubling followed by a 3 x 3 convolution (`MUBL`) |
| `MaskHead.skip_blocks_at_pool_size`, `image_branch_convolutions`, `skip_blocks_at_mask_resolution`, `fusion_convolutions` | `[]`, `[0, 0]`, `[]`, `[0, 0]` (off; this document runs without them) | finer inputs of the mask heads (stage 3b of the instance segmentation benchmark): backbone blocks before their pooling pooled at the pool size (`MSP`), an image branch of plain 3 x 3 convolutions on the RoIAlign crop of the normalized input image (`MIB`) and blocks pooled at the mask resolution (`MSM`), both concatenated after the doublings, and plain 3 x 3 fusion convolutions before the predictor (`MF`); the benchmark kept the image branch `[5, 32]` with the fusion `[1, 64]` (design A, +7.19 points over stage 3, chosen on 2026-11-19), off in the shipped configurations |
| `MaskHead.modules`, `MaskHead.normalization` | residual shortcuts on, squeeze-and-excitation and inception off; batch normalization | the convolution blocks of the mask heads (`MRES-MBN`) |
| `MaskHead.inference.mask_threshold` | 0.5 | pixels of a pasted mask kept for the mask IoU, the confusion matrix and the sample and error images; the exported model returns the probabilities without it |
| `Optimization.loss_weights.mask_loss` | 1.0 (1.0 for the four detection losses too) | weight of the mask loss in the total loss, the losses being tracked unweighted |
| `ExperimentConfiguration.evaluation.original_resolution` | False | True scores the boxes and masks on the original test images: the boxes restored, the masks pasted at that size, the objects those of the original annotations |
| `DetectionLevels.<level>.anchors`, `.region_proposal`, `.detector`, `Neck`, `RegionProposal`, `Detector` | the keys of DET with values of their own: `Neck.type: "none"`, `RegionProposal.proposal_selection: "joint"`, `RegionProposal.background_iou_threshold.min: 0.0` and `Detector.nms_criterion: "iou"`, where DET has `"feature_pyramid"`, `"per_level"`, 0.1 and `"distance_iou"` (exponent 0.6), with a learning rate of 2e-4 against 1e-3 | the detector, whose keys the `Read This First` of `object_detection_workflow.md` describes: no feature merging between the levels, one top-k and one NMS of the proposals over all the levels, the anchors overlapping no object sampled as background, the greedy final NMS, and `Detector.box_loss: "smooth_l1"` as in DET |
| `Detector.inference.score_threshold`, `maximum_detections` | 0.5, 100 | the detections of the deployed model, each with its mask |
| `ExperimentConfiguration.evaluation.score_threshold`, `target_precision` | 0.05, 0.99 | detections kept for the mask and box mAP, the threshold refused above `Detector.inference.score_threshold`; precision of the class thresholds, found on the mask average precisions |

### What A Run Writes
Whenever a run is launched or a checkpoint evaluated, the run folder, TensorBoard and MLflow receive the following, with the results of the two runs this document quotes.

- At every launch, `anchor_recall.csv` in the run folder: the share of the training objects with a foreground anchor, per level and size, the boxes being those of the segmentations (Step 2.18).
- Before the first iteration, `Input_Samples/` with `samples.json`: the first 10 images of every loader as the network receives them, with their masks, the training ones augmented, so `Training/` and `Test/` only when `validation_iteration_ratio` is 0 (`training_reference.md`, Run Folder).
- After every evaluation, `Evaluation/evaluation_summary.csv` (one row per checkpoint, the averaged weights in `ema_` columns) and `evaluation_curves.png` next to `Results.txt`, drawn again from it.
- At every `weight_saving_iterations` checkpoint, in `Evaluation/Iteration_<n>/`: `class_metrics.csv` (37 columns: the box AP of every class at 0.5, 0.75, 0.95 and 0.5:0.95 and by size, its box recall at 0.5 and 0.75 with their 95 % confidence intervals, the same columns prefixed with `mask_`, then the threshold and recall at the target precision), `class_thresholds.json`, `confusion_matrix.csv` and `.png` (the detections matched to the objects by mask IoU 0.5), `proposal_recall.csv`, `Samples/<image>.png` (the ground truth masks and boxes next to the detections reaching 0.5 with their masks), `Errors/` with `error_dumps`, and the same files in `EMA/` for the moving average of the weights.
- `Results.txt`: one block per evaluation (two with the moving average) with eleven tables, the mask then box mAP by method and by size, the mask and box mean average recall, the proposal recall and the mask and box per-class tables, each followed by its per-class recall table, every average precision and recall written as a percentage (`63.72 %`) while the counts, score thresholds and proposals per image keep their values, the files, TensorBoard and MLflow keeping the rates between 0 and 1; `Tensorboard/Train` holds the total loss, the five losses and the learning rate, `Tensorboard/Test` and `Test_EMA` 126 evaluation scalars for 20 classes, the 8 samples and the confusion matrix.
- In the INS run of `experiments/readme_screenshots`, the run of the instance segmentation tutorials (trained from random weights on the dataset generated on 2026-11-18 with the synthetic configuration plus the detection settings of the benchmarks, feature pyramid, per-level proposal selection, DIoU-NMS 0.6, learning rate 1e-3 and background minimum 0.1, the mask heads `[4, 64]` of stage 3, and a level on block 4 (anchors of 16 and 32 pixels) with the anchor scale 384 on block 7, which no shipped configuration holds, without the image branch of stage 3b; evaluated every 1000 iterations on the first 5,000 test images, planned for 10,000 iterations and stopped after its evaluation at 6000), iteration 6000 gives `Mask mAP@0.5/area` 82.96 % (88.09 % with the averaged weights), `Mask mAP@[0.5:0.95]/area` 60.97 % (68.52 %), a box `mAP@0.5/area` of 83.54 %, a mask `mAR@0.5` of 96.81 %, and 20,452 right detections, 2,886 of the wrong class, 3,029 false positives and 4,110 missed objects at the deployment threshold, matched on the masks.
- In the run of `run_configurations/instance_segmentation_workflow_run.yaml` shown in Stages 3 to 5 (the synthetic configuration trained to 5000 iterations from scratch, evaluated every 500), iteration 5000 gives `Mask mAP@0.5/area` 63.72 % with the trained weights and 73.64 % with the averaged weights, a box `mAP@0.5/area` of 64.04 % (73.93 %), and 32,850 right detections, 8,650 of the wrong class, 3,549 false positives and 28,182 missed objects at the deployment threshold; the trained weights had their best `Mask mAP@0.5/area`, 69.49 %, at iteration 4500 (Step 4.7).

### Pitfalls
- The synthetic configuration stops at 2000 iterations, too early for its samples: in the INS run of `experiments/readme_screenshots`, same backbone and batch size, `Mask mAP@0.5/area` is 15.01 % at iteration 1000, with 147 of the 27,448 test objects found at the deployment threshold, and 40.96 % at 2000; its samples show masks at 6000 (`suggestions.md`, item 30).
- The masks are 28 x 28 in their box and pooled from feature maps of stride 32 and 64 in the synthetic configuration, 16 to 64 in the INS run of `experiments/readme_screenshots` (`suggestions.md`, item 5), so thin parts are lost at strict IoU: in that run at iteration 6000 the star has a mask AP@0.5 of 96.68 % but a mask AP@0.75 of 61.89 %, against a box AP@0.75 of 90.30 %, and the mask `mAP@0.95` is 1.69 %.
- The sizes of the mask tables are measured on the mask areas and those of the box tables on the box areas, so their rows hold different objects: 8,080 small test objects by mask area against 5,778 by box area, 9,227 large against 12,034 (`Results.txt` of the INS run of `experiments/readme_screenshots`, iteration 6000).
- The exported model returns the 28 x 28 probabilities of every detection inside its box (`detection_masks`), not image masks: the client pastes them with `paste_instance_masks()` and keeps the pixels from `MaskHead.inference.mask_threshold`, which the export does not apply. The boxes are best restored to the original image unclipped (`restore_boxes_to_original_image(..., clip=False)`, as `restore_detection_boxes()` does for the evaluation), a box clipped at the border squeezing its mask (audit L22), as the instance segmentation recipe of the Export section of `documentation/reference/tools_reference.md` also says. The client letterboxes its images as the evaluation did by following the input contract that the export writes next to the model (`preprocessing.json`) and into the metadata of `model.onnx`: BGR, `keep_ratio` to 512 pixels, bilinear resize, padding of 0 centred, NCHW, values from 0 to 255.
- Every level has its own mask head, trained on the positive proposals of its level only, and the export runs every head on all the detections (`suggestions.md`, item 7): in the INS run of `experiments/readme_screenshots` at iteration 6000, 238.4 of the 250 proposals per image come from block 4, its finest level (`proposal_recall.csv`).
- An object whose mask vanishes at the network resolution is removed with its box from the training targets and from the evaluation at that resolution (`CocoDetectionDataset.add_instance_masks()`), but kept with `evaluation.original_resolution: True`; the loaders carry one full-size uint8 mask per object (audit L19), or with `Data.instance_index_maps: True` one uint8 map per image numbering its objects, expanded into the masks on the device. The maps need run length encodings whose masks never overlap: a polygon, two masks sharing a pixel or more than 255 objects in an image stop the loading with a message naming the setting (COCO keeps it off). The test loader of `evaluation.original_resolution` keeps one mask per object.

## Table of Contents
- [Read This First](#read-this-first)
- [Call Tree Overview](#call-tree-overview)
- [Stage 1 — Entry Point and Configuration Loading](#stage-1--entry-point-and-configuration-loading)
- [Stage 2 — Trainer Initialization](#stage-2--trainer-initialization)
- [Stage 3 — Training Loop](#stage-3--training-loop)
- [Stage 4 — Test Evaluation of a Checkpoint](#stage-4--test-evaluation-of-a-checkpoint)
- [Stage 5 — Standalone Checkpoint Evaluation (Optional)](#stage-5--standalone-checkpoint-evaluation-optional)

---

## Call Tree Overview

```text
main()
│   Entry point of every task. Give an instance_segmentation configuration with --configuration_path to run this workflow.
│
├── load_experiment_configuration(configuration_path)
│   │   Splits the YAML into experiment_configuration ('ExperimentConfiguration') and model_configuration (Optimization,
│   │   Data, Backbone, DetectionLevels, Neck, RegionProposal, Detector, MaskHead), converts the types, adds tracking_root,
│   │   run_name (None), standalone_evaluation_folder (None), label_mode and label_dictionary.
│   │
│   ├── load_configuration(configuration_path)
│   ├── build_label_dictionary(label_mapping_configuration, task)
│   │       Reads the mapping file {category id: class index} and adds 'background': 0, since the detector predicts
│   │       the background as class 0.
│   └── check_configuration()
│       ├── common checks
│       └── check_instance_segmentation_configuration()
│           ├── check_detection_configuration()                  every DetectionLevels.<level> also holds a mask_head,
│           │                                                    random_perspective and random_jigsaw are off, and the
│           │                                                    loss weights include mask_loss
│           ├── check_mask_head_configuration()                  mask_resolution, upsampling, pool size <= resolution,
│           │                                                    convolutions, normalization, modules, mask_threshold,
│           │                                                    pretrained, and the finer inputs
│           │                                                    (check_mask_head_fine_inputs)
│           └── check_boolean_flags()                            Data.instance_index_maps,
│                                                                evaluation.original_resolution
│
├── get_trainer_class(task)  -> InstanceSegmentationTrainer
│
├── InstanceSegmentationTrainer(experiment_configuration, model_configuration, configuration_path)   [STAGE 2]
│   │   LOAD_INSTANCE_MASKS = True, MATCHING_OVERLAP_TYPE = "masks", reads MaskHead.inference.mask_threshold,
│   │   evaluation.original_resolution and Data.instance_index_maps.
│   │
│   └── DetectionTrainer.__init__()
│       │   Reads evaluation.score_threshold, Detector.inference.score_threshold, target_precision, error_dumps.
│       │
│       └── BaseTrainer.__init__()   (through CocoSplitTrainer)
│           ├── get_device() / get_run_name() -> build_run_name() -> build_instance_segmentation_sections()
│           ├── DataAugmenter()
│           ├── restore_run_label_dictionary() / check_resumed_data() / setup_training_paths()
│           │       the classes of a resumed run kept from its label_dictionary.json, a resume on changed labels
│           │       refused, then the run folder with the configuration and label_dictionary.json
│           │       (classification_workflow.md, Step 2.6)
│           ├── build_dataloaders()
│           │   ├── read_test_manifest()                             class_names, class_colours, test image names
│           │   └── get_detection_dataloaders(load_instance_masks=True, load_instance_index_maps=instance_index_maps,
│           │       │                         load_original_instances=original_resolution,
│           │       │                         build_validation_loader=builds_validation_loader)
│           │       └── get_detection_dataloader() for every split read (the validation one only when the run reads it,
│           │           │                          the test loader of original_resolution without maps)
│           │           └── CocoDetectionDataset(load_instance_masks=True, load_instance_index_maps=...) +
│           │               build_native_loader(collate_function)
│           ├── check_loaders_give_batches()                         refuses a split whose loader gives no batch
│           ├── save_data_record()                                   data.json
│           ├── get_test_sample_classes() + choose_sample_identifiers()
│           ├── build_model()
│           │   └── InstanceSegmenter.__init__()
│           │       ├── TwoStageDetector.__init__()                  backbone, FeatureNeck, RegionProposal and DetectionHead
│           │       │                                                per level, anchors, RegionProposalFilter
│           │       ├── MaskFineInputs()                             image branch and skips at the mask resolution shared by
│           │       │                                                the levels, none by default
│           │       └── _build_mask_heads_dictionary()
│           │           └── MaskHead() for every level -> count_mask_upsamplings()
│           │                   RoIAlign -> ConvolutionBlock -> doublings (ConvTranspose2d, or bilinear + Conv2d) ->
│           │                   1x1 mask_predictor -> bilinear resize when the doublings miss mask_resolution
│           ├── initialise_pretrained_components()                   Backbone, RegionProposal, Detector, MaskHead
│           ├── build_optimizer() / ExponentialMovingAverage() (when activated)
│           ├── get_loss_weights() / get_tracked_metrics()           5 losses + Total Loss
│           ├── print_experiment_information() / print_summary()     backbone, neck, RPN, detection, mask, receptive field,
│           │                                                        model size panels
│           ├── diagnose_data_splits()                               anchor recall of every split
│           │                                                        (only with compute_anchor_coverage)
│           ├── view_input_batches() -> show_input_samples() -> build_input_windows() -> get_input_sample_image() ->
│           │   overlay_instance_masks()                             (only with view_input_images)
│           ├── benchmark_speed()                                    (only with benchmark_speed); any of these three
│           │                                                        modes then exits, without run folder
│           ├── analyse_training_data()                              anchor recall on the boxes of the segmentations
│           │   └── gather_object_boxes() -> select_image_objects(boxes_from_segmentation=True) ->
│           │       compute_segmentation_bounding_box()              the box of every object read from its polygons
│           │                                                        or RLE, not from 'bbox'
│           └── restore_last_model()
│
└── trainer.run_training_loop()   [STAGE 3]
    └── run_training_iterations()
        ├── save_input_samples()   Input_Samples/: the first 10 images of every loader, drawn as the viewer draws them
        ├── for each training_iteration, from start_iteration to number_iterations (the total of the run):
        │   ├── get_next_batch() -> CocoDetectionDataset.__getitem__() + collate_function()
        │   │   │   read_image(); a batch failing to load is dropped and the next one taken
        │   │   ├── select_image_objects(boxes_from_segmentation=True) -> compute_segmentation_bounding_box()
        │   │   ├── add_instance_index_map()                                      with Data.instance_index_maps
        │   │   │   ├── build_instance_index_map()      RLE runs painted with their object number (one cumulative
        │   │   │   │                                   sum), overlaps refused, one resize INTER_NEAREST_EXACT
        │   │   │   └── find_index_map_objects()        vanished objects removed, the others renumbered, boxes
        │   │   └── add_instance_masks()                                          without Data.instance_index_maps
        │   │       ├── build_instance_masks() -> build_instance_mask() for every object
        │   │       │   ├── rasterize_polygons() -> fill_polygon_part()               polygons, at the network size
        │   │       │   └── decode_run_length_encoding() (-> decode_compressed_counts()) + resize INTER_NEAREST_EXACT
        │   │       ├── objects whose mask is empty once resized removed
        │   │       └── compute_mask_bounding_boxes()                                 boxes = tight boxes of the masks
        │   ├── run_model_iteration()
        │   │   ├── prepare_batch()                  instance_index_maps moved to the device and expanded there
        │   │   │                                    (expand_instance_index_maps), or instance_masks moved to it
        │   │   ├── augment_batch() -> DataAugmenter.augment_batch(instance_masks=...)
        │   │   │       masks moved with the image, boxes recomputed from the masks, fragments < 25 % removed
        │   │   ├── forward_pass() -> InstanceSegmenter.forward(..., ground_truth_masks=...)
        │   │   │   ├── forward_detection_stages()                                    the 4 detection losses
        │   │   │   └── compute_proposal_mask_loss()
        │   │   │       ├── MaskHead.forward() on the positive sampled proposals of every level
        │   │   │       ├── build_mask_targets()                                      RoIAlign crop of the true masks
        │   │   │       └── compute_mask_loss()                                       BCE on the true class channel
        │   │   └── MetricTracker.accumulate() + log_to_tensorboard()
        │   ├── training_loss.backward() / step_optimizer()
        │   ├── save_model() on the multiples of weight_saving_iterations and intermediate_weight_saving_iterations
        │   ├── validation batch (the 5 losses, without gradients), only with validation_iteration_ratio > 0
        │   ├── average_trackers() + log_interval_metrics() + print_interval_summary() + report_dropped_batches()
        │   └── run_test_evaluation() on the multiples of weight_saving_iterations, last         -> [STAGE 4]
        └── save_model() of the steps taken since the last checkpoint, if any, then shutdown_loader_workers() and
                finish_experiment_tracking()

run_test_evaluation(iteration)   [STAGE 4]
└── evaluate_test_split(), for the trained weights, then again within use_averaged_weights() for the moving average
    │   of the weights (Optimization.exponential_moving_average on: Evaluation/Iteration_<n>/EMA/, Tensorboard/Test_EMA)
    ├── create_evaluation_accumulator()
    ├── evaluation_context()                                 score threshold lowered to evaluation.score_threshold
    ├── for each test batch:
    │   ├── prepare_batch() + predict_batch() -> InstanceSegmenter.forward() without ground truth
    │   │   │                                    (test maps expanded into instance_masks, as in training)
    │   │   ├── forward_detection_stages() -> postprocess_detections()
    │   │   └── predict_detection_masks()                    mask head of the level of the proposal of every
    │   │                                                    detection, sigmoid of the detected class channel
    │   ├── accumulate_evaluation_batch()
    │   │   ├── gather_image_detections() -> compute_mask_overlaps()     chunks of 16 detections per 1024 x 1024 pixels
    │   │   │       (64 at 512): paste_instance_masks() -> >= mask_threshold -> compute_mask_intersection_over_union()
    │   │   ├── gather_image_ground_truth()                              with mask_areas
    │   │   ├── accumulate_proposal_recall()
    │   │   ├── compute_overlap_matrix(overlap_type="masks") + match_image_detections() + add_to_confusion_matrix()
    │   │   └── dump_errors() (when activated)
    │   └── save_sample_images() -> build_sample_image() -> draw_test_image() -> get_sample_masks() +
    │       build_detection_sample_grid()
    ├── compute_evaluation_results()
    │   ├── summarise_average_precisions(overlap_type="masks")   mask mAP and mAR, 3 methods, sizes by mask area
    │   ├── summarise_average_precisions(overlap_type="boxes")   box mAP and mAR, 3 methods, sizes by box area
    │   │                                                        (both with the recall of every class at 0.5 and 0.75,
    │   │                                                        its exact 95 % interval and its per-class recall table)
    │   ├── save_proposal_recall()                               proposal_recall.csv
    │   ├── write_class_metrics_file()                           box and mask columns of every class
    │   └── save_error_analysis()                                thresholds of the mask APs, confusion matrix
    └── report_evaluation() -> report_evaluation_results() + track_evaluation() + update_evaluation_curves()
            headline 'Mask mAP@0.5/area'; the row of the checkpoint added to evaluation_summary.csv,
            evaluation_curves.png drawn again

scripts/evaluation/evaluate_checkpoint.py   [STAGE 5]
└── evaluate_checkpoints() -> start_evaluation_tracking() + summarise_checkpoint_evaluations()
    -> run_test_evaluation() for every checkpoint, then evaluation_summary.csv
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below are real outputs from running the loader on `vision_training/configurations/instance_segmentation/configuration_synthetic_shapes.yaml` as it was then, its data folders and mapping file redirected to the generated dataset.

### Step 1.1 — `main()`
The entry point is the same for every task (see `classification_workflow.md`, Step 1.1).

### Step 1.2 — `load_experiment_configuration()`
The model sections are those of the object detection (see `object_detection_workflow.md`, Step 1.2) plus `MaskHead`, the settings shared by the mask heads of every level: `mask_resolution` (the side of the predicted masks, 28), `upsampling` (`transposed` or `bilinear`, how the pooled features are doubled up to the mask resolution), the four finer inputs of the heads (`skip_blocks_at_pool_size`, `image_branch_convolutions`, `skip_blocks_at_mask_resolution`, `fusion_convolutions`, off), `normalization`, `modules`, `inference.mask_threshold` (the probability from which a pixel of a predicted mask belongs to its object) and `pretrained`. Every level of `DetectionLevels` holds, next to its anchors, region proposal and detector settings, a `mask_head` with its own `roi_align_pool_size` and `convolutions` (`[number of layers, number of filters]`).

**Example: the sections and the mask settings**
```text
experiment keys: task, experiment_folder, run_name_suffix, train_data_folder, validation_data_folder, test_data_folder, batch_size,
                 information_dump, weight_saving_iterations, intermediate_weight_saving_iterations, number_iterations,
                 validation_iteration_ratio, evaluation, tracking, learning_rate, allow_data_change_on_resume,
                 diagnosis_modes, tracking_root, run_name
model keys:      Optimization, Data, Backbone, DetectionLevels, RegionProposal, Detector, MaskHead
evaluation:      {'per_class_report': True, 'number_sample_images': 8, 'score_threshold': 0.05, 'target_precision': 0.99,
                  'error_dumps': {'activated': False, 'maximum_per_class_pair': 20}}
Data keys:       image_settings, label_mapping, data_augmentation, label_mode, label_dictionary
number_iterations: 10000000000
DetectionLevels 5: {"anchors": {"scales": [32, 64, 128], "aspect_ratios": [0.5, 1.0, 2.0]}, "region_proposal": {"convolutions": [2, 64]},
                    "detector": {"roi_align_pool_size": 7, "convolutions": [2, 64], "fully_connected": [256, 128]},
                    "mask_head": {"roi_align_pool_size": 14, "convolutions": [4, 64]}}
MaskHead: {"mask_resolution": 28, "upsampling": "transposed",
           "normalization": {"feature_map_normalization": "batch"},
           "modules": {"residual_nets": true, "inception": false, "squeeze_and_excite": false},
           "inference": {"mask_threshold": 0.5},
           "pretrained": {"experiment_folder": null, "iteration": null, "freeze": false, "learning_rate_multiplier": 1.0}}
```
The current file also has a `Neck` section (`type: "none"`) before `RegionProposal`, `'original_resolution': False` in its `evaluation` section, between `target_precision` and `error_dumps` (the scoring at the original resolution of Step 4.7), `mask_loss: 1.0` in `Optimization.loss_weights`, the four finer inputs of `MaskHead` turned off (`[]`, `[0, 0]`, `[]`, `[0, 0]`) between `upsampling` and `normalization`, and `instance_index_maps: True` in its `Data` section, between `image_settings` and `label_mapping` (Step 2.9), and the current loader adds the runtime key `standalone_evaluation_folder` after `run_name` (`None` for a training, `Checkpoint_Evaluation` or `Checkpoint_Evaluation_nms_<name>` set by the evaluation script of Stage 5).

### Step 1.3 — `load_configuration()`
**Example output**
```text
top-level keys:             ['ExperimentConfiguration', 'Optimization', 'Data', 'Backbone', 'DetectionLevels', 'RegionProposal', 'Detector', 'MaskHead']
number_iterations:          '1e10'      <- still a string
learning_rate:              0.0002
```

### Step 1.4 — `build_label_dictionary()`
As for the detection, `'background': 0` is added to the mapping file.

**Example: the label dictionary of the generated dataset**
```text
label_mapping: {'mode': 'multi_class', 'mapping_file_path': '.../synthetic_shapes/label_mapping.json'}
dict: {1: 1, 2: 2, 3: 3, ..., 19: 19, 20: 20, 'background': 0}
21 entries, 21 distinct classes
```

### Step 1.5 — `check_configuration()`
The common checks run first, then `check_instance_segmentation_configuration()`: the checks of the detection (`check_detection_configuration()`, which also requires a `mask_head` in every detection level, refuses `random_perspective` and `random_jigsaw`, whose moves the boxes and the masks cannot follow, and checks the weight of the `mask_loss` with the four others), `check_mask_head_configuration()` (with `check_mask_head_fine_inputs()`: the `[layers, channels]` of the image branch and of the fusion convolutions both positive or `[0, 0]`, the skip blocks distinct numbers from 0, the normalized input image, to the last backbone block, held to the U-Net layout by `check_skip_block_strides()` (block b before its pooling being at stride 2 ** (b - 1): every skip finer than the finest detection level, block 5 (stride 16) at most with the levels 5 to 7 of this run (finest stride 32), every skip at the mask resolution finer than every skip at the pool size, and no block in both lists), and no image branch with mask heads initialised by another run, the branch not being transferred with them), and the booleans `Data.instance_index_maps` and `evaluation.original_resolution`. The RoIAlign pool size of every level must not exceed `MaskHead.mask_resolution`, since the mask head only enlarges the pooled features.

**Example: output with `MaskHead.upsampling: nearest` and `DetectionLevels.5.mask_head.roi_align_pool_size: 32` in the YAML**
```text
------------------------------------------------------
Invalid Configuration File : .../configuration_wrong.yaml
------------------------------------------------------
  - 'MaskHead.upsampling' must be one of ['transposed', 'bilinear'], got 'nearest'.
  - 'DetectionLevels.5.mask_head.roi_align_pool_size' (32) must not exceed 'MaskHead.mask_resolution' (28), the pooled features being only enlarged to the masks.
SystemExit code 1
```

### Step 1.6 — `get_trainer_class()`
**Example output**
```text
get_trainer_class(task="instance_segmentation") -> <class 'vision_training.trainers.instance_segmentation_trainer.InstanceSegmentationTrainer'>
```

---

## Stage 2 — Trainer Initialization

> The examples below are real outputs from building the trainer on the redirected `configuration_synthetic_shapes.yaml` (512 pixel images, batch size 40), with the run folder redirected to a temporary folder. The whole construction took 16.4 s on CPU.

### Step 2.1 — `InstanceSegmentationTrainer.__init__()` → `DetectionTrainer.__init__()`
The instance segmentation trainer is the detection trainer with two class attributes of its own (next to `TASK_NAME`), three settings, and a module constant:
- `LOAD_INSTANCE_MASKS = True`: the COCO loaders return the mask of every object (Step 2.9).
- `MATCHING_OVERLAP_TYPE = "masks"`: the detections are matched to the objects by the IoU of their masks, for the confusion matrix and the error images (Step 4.5).
- `self.mask_threshold`, read from `MaskHead.inference.mask_threshold`: the probability from which a pixel of a pasted mask belongs to its object, in the evaluation and the sample images.
- `self.original_resolution`, read from `evaluation.original_resolution` before the loaders are built: the test loader then also gives the objects at the original resolution, where the boxes and masks are scored (Step 4.7).
- `self.instance_index_maps`, read from `Data.instance_index_maps` (`True` in the synthetic configuration, absent from the configuration of these examples): the loaders then give one instance index map per image, expanded into the masks on the device (Step 2.9).
- `MASK_OVERLAP_CHUNK_SIZE = 16` of `instance_segmentation_trainer.py`: the detections whose masks are pasted at once for a 1024 x 1024 image when the mask overlaps are computed, scaled inversely to the pixels of the image (`MASK_OVERLAP_REFERENCE_PIXELS`), 64 at 512 pixels (Step 4.5).

The detection constructor then reads the two score thresholds, the target precision and the error images settings (see `object_detection_workflow.md`, Step 2.1).

**Example: the attributes after the construction**
```text
self.mask_threshold             = 0.5
self.evaluation_score_threshold = 0.05
self.deployment_score_threshold = 0.5
self.target_precision           = 0.99
self.error_dumps_configuration  = {'activated': False, 'maximum_per_class_pair': 20}
TASK_NAME = 'instance_segmentation'   SAMPLE_NAME = 'Images'   COUNTS_FRAMES = False
LOAD_INSTANCE_MASKS = True   MATCHING_OVERLAP_TYPE = 'masks'
```
The example predates `evaluation.original_resolution`; the current trainer also holds `self.original_resolution`, `False` in the synthetic configuration.

### Step 2.2 — `BaseTrainer.__init__()`
**Example: some attributes set by the constructor**
```text
self.device = cpu   self.per_class_report = True   self.number_sample_images = 8   self.input_image_size = 512
self.batch_size = 40   self.start_iteration = 1
```

### Step 2.3 — `get_device()`
**Example output** (forced on the CPU; a normal run on this machine prints `cuda`)
```text
For Training We Will Be Using device: cpu
```

### Step 2.4 — `get_run_name()` → `build_run_name()` → `build_instance_segmentation_sections()`
The prefix is `INS`, and the sections are those of the detection (see `object_detection_workflow.md`, Step 2.4) with a mask heads section before the classes: `MRES` (residual shortcuts; `MSE` and `MINC` for the other modules), `MBN` (batch normalization; `MLN`, `MNN`), `MUTC` (transposed convolutions; `MUBL` for bilinear), `M14.4x64` (RoIAlign of 14, 4 layers of 64 filters, written once when every level has the same mask head, otherwise per level joined with `+`) and `MR28` (mask resolution), followed, only when used, by the finer inputs: `MSP<blocks>` (skips at the pool size), `MIB<layers>x<channels>` (image branch), `MSM<blocks>` (skips at the mask resolution) and `MF<layers>x<channels>` (fusion convolutions), the blocks joined with `.`, so that the runs written before them keep their names. The mask threshold gives no token, so that it can change between two resumptions of a run.

**Example output**
```text
INS_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512.598295_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128_MRES-MBN-MUTC-M14.4x64-MR28_MC20
```

### Step 2.5 — `DataAugmenter()`
The same augmenter as the detection; with instance masks, the geometric augmentations move the masks with the image (Step 3.8). This configuration turns every type off except `random_haze`, a photometric type that never changes the masks.

### Step 2.6 — `setup_training_paths()`
As for every task (see `classification_workflow.md`, Step 2.6): a resumed run first keeps the classes of its `label_dictionary.json` (`restore_run_label_dictionary()`) and is refused when the `labels.json` of a split changed since its `data.json` (`check_resumed_data()`, unless `allow_data_change_on_resume`), then the run folder is created with the copy of the configuration and `label_dictionary.json`, the class index of every category, background included. The loaders built next are refused when one of them gives no batch (`check_loaders_give_batches()`), then `save_data_record()` writes `data.json`.

**Example: the run folder at the end of the construction**
```text
INS_IN-KAR-512_..._MRES-MBN-MUTC-M14.4x64-MR28_MC20/
├── Tensorboard/
├── Weights/
├── anchor_recall.csv              263 bytes   (Step 2.18)
├── data.json                    5,487 bytes   (save_data_record())
├── label_dictionary.json                      the class index of every category of the run
└── training_configuration.yaml 33,125 bytes
```

### Step 2.7 — `build_dataloaders()`
The hook of the detection (see `object_detection_workflow.md`, Step 2.7): `class_names`, `class_colours` and the test image names from the `labels.json` of the test split (`read_test_manifest()`), then the loaders with the instance masks, the validation one only when the run reads it (`validation_iteration_ratio > 0` or a diagnosis mode ending the program, `classification_workflow.md`, Step 2.7; none now for the configuration of the 5000-iteration run of Stage 3, without validation), the test loader also giving the objects at the original resolution when `evaluation.original_resolution` is on.

**Example output**
```text
class_names: {0: 'background', 1: 'square', 2: 'circle', 3: 'triangle', 4: 'rectangle', 5: 'pentagon', 6: 'hexagon', 7: 'plus_sign',
              8: 'star', 9: 'octagon', 10: 'rhombus', 11: 'parallelogram', 12: 'trapezoid', 13: 'kite', 14: 'heptagon', 15: 'nonagon',
              16: 'ellipse', 17: 'semicircle', 18: 'annulus', 19: 'capsule', 20: 'l_shape'}
test_image_file_names: 20 entries, [(1, 'test_000000.png'), (2, 'test_000001.png'), (3, 'test_000002.png'), ...]
```

### Step 2.8 — `get_detection_dataloaders(load_instance_masks=True, load_instance_index_maps=...)`
`load_instance_index_maps` is `Data.instance_index_maps`, read by `InstanceSegmentationTrainer.__init__()`. The test loader of `evaluation.original_resolution` is built without maps, its samples carrying one mask per object at both resolutions.
**Example: the three loaders** (recorded with 8 workers, when the number of workers was a setting; every loader now has 1, the validation loader is built only with `validation_iteration_ratio > 0`, and the batches are pinned only for a discrete GPU, `pin False` on the integrated GPU of the DGX Spark, `uses_pinned_memory()`)
```text
train_dataloader        100 images   3 batches   collate_function   num_workers 8   prefetch 2   pin True   persistent True
validation_dataloader    20 images   1 batch     collate_function   num_workers 8   prefetch 2   pin True   persistent True
test_dataloader          20 images   1 batch     collate_function   num_workers 8   prefetch 2   pin True   persistent True
```

### Step 2.9 — `CocoDetectionDataset(load_instance_masks=True)` → `select_image_objects(boxes_from_segmentation=True)`
The manifests are COCO `labels.json` files, every annotation holding a `segmentation`: a list of polygons or a run length encoding (`size` [height, width] and `counts`, the lengths of the alternating background and object runs, column by column, as a list when uncompressed or a string when compressed). With the instance masks, the box of every object is not read from its `bbox` but from its segmentation (`compute_segmentation_bounding_box()`: the extent of the polygon vertices, or of the object runs of the RLE, x2 and y2 one past the last pixel), so that the boxes and the masks agree and an annotation needs no `bbox`. Only the boxes are read here, without decoding a mask. The manifest is kept as one packed record per image (`SerializedRecords`, Step 2.9 of `object_detection_workflow.md`), so that the loader workers never copy its Python objects: parsed, the manifest of the 512,000 training images of the benchmark dataset takes about 7 GB of the main process (0.67 GB measured on its 50,000 validation images), which the training worker of the pair of 2026-11-13 copied page by page as it read the images, 9.55 GB of private memory at iteration 9000.

`__getitem__()` reads the image with `read_image()` (OpenCV, BGR, `uint8`, a `ValueError` naming a file that cannot be read), then calls `add_instance_masks()`: `build_instance_mask()` builds the mask of every object at the network resolution (polygons rasterised in the network input by `rasterize_polygons()`; an RLE decoded at the original resolution by `decode_run_length_encoding()`, its compressed string read with numpy by `decode_compressed_counts()` and its column by column pixels turned into the mask by `cv2.transpose`, then resized like the image with `cv2.INTER_NEAREST_EXACT` and padded with 0), the objects whose mask is empty once resized are removed with their box and label, and `compute_mask_bounding_boxes()` replaces every box by the tight box of its resized mask.

With `Data.instance_index_maps: True` (the synthetic configuration), `add_instance_index_map()` replaces `add_instance_masks()`: `build_instance_index_map()` paints the run length encodings of the image into one map at the original resolution, every object run adding the number of its object where it starts and removing it where it ends in a difference array, one cumulative sum giving the map, which is resized once like the masks (`cv2.INTER_NEAREST_EXACT`, padding 0). A pixel covered by two masks would hold the sum of their numbers, so the map refuses the image when its covered pixels are fewer than its object pixels; polygons and more than `MAXIMUM_INDEX_MAP_OBJECTS` (255) objects are refused too. `find_index_map_objects()` (`scipy.ndimage.find_objects`) gives the objects still covering a pixel and their tight boxes; the vanished ones are removed with their label and the others renumbered from 1 in their order. Expanded, the map gives exactly the masks of `add_instance_masks()`, with the same labels and boxes, a third of the bytes here and about a ninth on the benchmark dataset (1024 pixels, about 8 objects an image). With one worker and pinned memory, a 64-image batch of the benchmark dataset reaches the GPU in 1.20 s instead of 2.22 s, its expansion taking 23 ms, and the maps of 1,000 test images give the masks, labels and boxes of `add_instance_masks()` exactly (measured on 2026-11-13).

**Example: the manifest of the training split**
```text
images[0]:          {'id': 1, 'file_name': 'train_000000.png', 'width': 512, 'height': 512}   (100 images)
annotation keys:    ['id', 'image_id', 'category_id', 'bbox', 'segmentation']
annotations[0]:     {'id': 1, 'image_id': 1, 'category_id': 6, 'bbox': [325, 369, 133, 123],
                     'segmentation': {'size': [512, 512], 'counts': [166822, 2, 508, 6, 505, 9, 502, 13, 497, 17, ...]}}   (358 annotations)
segmentation types: 358 RLE (dict), 0 polygons (list), 0 compressed RLE
categories[0]:      {'id': 1, 'name': 'square', 'supercategory': 'shape'}   (20 categories)
```

**Example: the first sample of the training dataset** (`train_000073.png`, image id 74)
```text
annotation bbox [x, y, w, h]:   [188, 433, 34, 70]       [153, 94, 145, 147]      [408, 146, 95, 97]  ...
segmentation box [x1, y1, x2, y2]: [188, 433, 222, 503]  [153, 94, 298, 241]      [408, 146, 503, 243] ...
dataset[0]:
  images              (3, 512, 512) torch.uint8
  bounding_boxes      (6, 4) torch.float32   [[188, 433, 222, 503], [153, 94, 298, 241], [408, 146, 503, 243], [259, 335, 325, 401],
                                              [98, 277, 219, 397], [274, 144, 323, 209]]
  labels              (6,) torch.int64       [4, 2, 8, 7, 15, 10]
  sample_identifiers  74
  instance_masks      (6, 512, 512) torch.uint8, values {0, 1}
  mask pixels         [2380, 15705, 3566, 2507, 10929, 2157]
```
The images are already 512 pixels, so the masks are not resized and their boxes are the boxes of the segmentations. The sample above was recorded with one mask per object (before `Data.instance_index_maps`).

**Example: the first two training images of `synthetic_shapes` with the maps** (`train_000000.png`, `train_000001.png`, recorded on 2026-11-13)
```text
dataset[0]:
  images              (3, 512, 512) torch.uint8
  bounding_boxes      (1, 4) torch.float32
  labels              (1,) torch.int64
  instance_index_maps (512, 512) torch.uint8, values {0, 1}
collate_function:     instance_index_maps (2, 512, 512) uint8 (524,288 bytes), labels [9] and [19, 1, 14, 3, 9]
expanded masks:       (1, 512, 512) and (5, 512, 512) uint8 (1,572,864 bytes), mask pixels [18685] [2401, 16641, 4022, 866, 7001]
                      identical to the masks, boxes and labels of add_instance_masks()
```

### Step 2.10 — `build_native_loader()`
The loader of every task (see `classification_workflow.md`, Step 2.10). `collate_function()` stacks the images and the instance index maps, and keeps the boxes, labels and instance masks as lists, one tensor per image, the number of objects changing from one image to the next (Step 3.6).

### Step 2.11 — `get_test_sample_classes()` + `choose_sample_identifiers()`
The same 8 test images every evaluation, spread over the test split (see `object_detection_workflow.md`, Step 2.11).

**Example output**
```text
sample identifiers: {1, 4, 6, 9, 12, 15, 17, 20}
                    test_000000.png, test_000003.png, test_000005.png, test_000008.png, test_000011.png, test_000014.png,
                    test_000016.png, test_000019.png
```

### Step 2.12 — `build_model()` → `InstanceSegmenter.__init__()`
`InstanceSegmenter` is a `TwoStageDetector` (backbone, one region proposal head and one detection head per level, anchors, proposal filter; see `object_detection_workflow.md`, Steps 2.12 to 2.16) with one more `ModuleDict`, `mask_heads_dictionary`: one `MaskHead` per detection level, built on the feature map of that level by `_build_mask_heads_dictionary()`.

**Example: the children of the model and their parameters**
```text
backbone                      Backbone               2,245,216
region_proposer_dictionary    ModuleDict               495,239   {'5': 113,965, '6': 190,637, '7': 190,637}
detector_heads_dictionary     ModuleDict             2,869,179   {'5': 931,817, '6': 968,681, '7': 968,681}
region_proposal_filter        RegionProposalFilter           0
mask_heads_dictionary         ModuleDict               571,839   {'5': 166,037, '6': 202,901, '7': 202,901}
detection_input_information:  {'5': {'feature_map_size': 16, 'number_channels': 64}, '6': {'feature_map_size': 8, 'number_channels': 128},
                               '7': {'feature_map_size': 8, 'number_channels': 128}}
anchors: (3456, 4)
```
A model built by the current code also has a `neck` (`FeatureNeck`, no parameter with `Neck.type: "none"`) after the backbone and a `mask_fine_inputs` (`MaskFineInputs`, no parameter with the finer inputs off) before the mask heads, as the shipped synthetic configuration gives on the CPU (`neck` 0, `mask_fine_inputs` 0, `mask_heads_dictionary` 99,756 parameters).

### Step 2.13 — `MaskHead.__init__()` → `count_mask_upsamplings()`
A mask head is a small fully convolutional network applied to every box:
- `torchvision.ops.roi_align()` pools the box from the feature map of its level into a `roi_align_pool_size` square (`spatial_scale` = feature map size / image size, `sampling_ratio=0`, `aligned=True`).
- `convolutional_block`: the `convolutions` layers of 3x3 convolutions, with the normalization and modules of `MaskHead`.
- `count_mask_upsamplings()` gives the number of doublings that bring the pool size to the mask resolution without exceeding it (here 14 -> 28: 1). Each doubling is a `ConvTranspose2d` of kernel 2 and stride 2 (`transposed`) or a bilinear upsampling followed by a 3x3 convolution (`bilinear`), followed by a ReLU.
- `mask_predictor`: a 1x1 convolution giving one mask logit map per class, background included; the mask of a box is read in the channel of its class (Steps 3.13 and 4.3).
- With the finer inputs of `MaskHead` (off in this document): the backbone blocks of `skip_blocks_at_pool_size`, read before their pooling, are pooled by `roi_align()` at the same pool size with their own `spatial_scale` and concatenated to the pooled features before `convolutional_block`; the fine features of `MaskFineInputs`, built once for all the levels (the image branch, `image_branch_convolutions` plain 3x3 convolutions on the RoIAlign crop of the normalized input image at the mask resolution, then the blocks of `skip_blocks_at_mask_resolution` pooled at the mask resolution), are concatenated after the doublings; `fusion_block` (`fusion_convolutions`, plain 3x3 convolutions with normalization and ReLU) mixes them before `mask_predictor`.
- When the doublings do not reach `mask_resolution` exactly (for instance a pool of 10 doubled to 20 for a resolution of 28), the logits are resized bilinearly to it, or, with fine features, the doubled features before their concatenation.

The masks of every level have the same side, so that the masks of all the levels can be concatenated for the loss.

**Example: the mask head of block 5** (printed again on the CPU with the current code and PyTorch)
```text
MaskHead(
  (convolutional_block): ConvolutionBlock(
    (squeeze_and_excitation_layers): ModuleList()
    (block): ModuleList(
      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      ... (3) to (11): three more Conv2d(64, 64, 3x3) + BatchNorm2d + ReLU
    )
  )
  (upsampling_layers): ModuleList(
    (0): ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(2, 2))
  )
  (activation): ReLU()
  (mask_predictor): Conv2d(64, 21, kernel_size=(1, 1), stride=(1, 1))
)
mask head 5: pool 14, upsamplings 1, upsampled 28, mask 28, spatial scale 0.03125,  feature map 16
mask head 6: pool 14, upsamplings 1, upsampled 28, mask 28, spatial scale 0.015625, feature map 8
mask head 7: pool 14, upsamplings 1, upsampled 28, mask 28, spatial scale 0.015625, feature map 8
```

### Step 2.14 — `initialise_pretrained_components()` + `build_optimizer()` + `get_tracked_metrics()`
Four components can be initialised by another run: `Backbone`, `RegionProposal`, `Detector` and `MaskHead` (each section with its `pretrained` settings). The mask heads of the source run must have the same mask resolution, upsampling, normalization, modules, finer inputs and per-level `mask_head` (`compare_detection_heads()`, a run written before the finer inputs counting as without them); the image branch, built apart from the mask heads (`MaskFineInputs`), is not transferred. `get_tracked_metrics()` adds `mask_loss` to the five metrics of the detection.

**Example output**
```text
pretrained:      {'Backbone': None, 'RegionProposal': None, 'Detector': None, 'MaskHead': None}
optimizer:       Adam, 1 parameter group, learning rates [0.0002]
tracked metrics: {'total_loss': 'Total Loss', 'region_proposal_classification_loss': 'RPN Classification Loss',
                  'region_proposal_localisation_loss': 'RPN Regression Loss', 'detection_classification_loss': 'Detection Classification Loss',
                  'detection_regression_loss': 'Detection Regression Loss', 'mask_loss': 'Mask Loss'}
```

### Step 2.15 — `print_experiment_information()` (→ `describe_split()` → `count_split_images_and_annotations()`)
**Example output**
```text
--------------------------------------
=== Experiment Launch Information ===
--------------------------------------
- Run Name                : INS_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512.598295_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128_MRES-MBN-MUTC-M14.4x64-MR28_MC20
- Experiment Folder       : INS_IN-KAR-512_..._MC20
- Tensorboard Directory   : Tensorboard
- Weights Directory       : Weights
- Configuration File      : training_configuration.yaml
- Data Record             : data.json
- Train Data Folder       : train (100 images, 358 annotations)
- Validation Data Folder  : validation (20 images, 74 annotations)
- Test Data Folder        : test (20 images, 66 annotations)
- Iterations Per Epoch    : 3
- Batch Size              : 40
- Total Iterations        : 10000000000
- Save And Evaluate Every : 10000 iterations
- Save Only Every         : 2000 iterations
- Learning Rate           : 0.0002
- Optimizer               : adam (weight decay 0.0)
- Learning Rate Schedule  : constant (warm-up 0 iterations)
- Weight Moving Average   : Off
- Validation Iter. Ratio  : 25
- MLflow Tracking         : Off
```

### Step 2.16 — `print_summary()` / `print_receptive_field_summary()` / `print_model_size_summary()`
The panels of the detection (backbone, region proposal, detection, receptive field; see `object_detection_workflow.md`, Step 2.19) are followed by a **Mask Architecture** panel, one block per level, and the model size panel counts the mask heads apart. With finer inputs, the mask panel adds the `Image Branch` and `Skips At Mask Size` lines and, in every level, `Skips At Pool Size` and `Fusion Layers`, and the model size panel a `Mask Image Branch` row; this run has none. The `Modules` lines list the residual shortcuts only for a block adding one (`count_residual_additions()`): the detection heads of blocks 6 and 7 list the squeeze-and-excitation only, their 2 layers adding no shortcut once the first one has turned 128 channels into 64 (`bug_standards.md` H4, dropped), while every mask head lists `Residual Shortcuts`, the 4 layers of those of blocks 6 and 7 adding one shortcut after that first layer.

**Example output: the region proposal panel, the mask panel and the model size** (rebuilt on the CPU with the current code from the configuration of this stage, the same as printed then; the backbone, detection and receptive field panels are those of the detection)
```text
╭────────────── Region Proposal Architecture ──────────────╮
│ Convolution Block 5                                      │
│   - Feature Map Size : [16, 16]                          │
│   - Scales           : [32, 64, 128]                     │
│   - Aspect Ratios    : [0.5, 1.0, 2.0]                   │
│   - Total Anchors    : 2304                              │
│                                                          │
│ Convolution Block 6                                      │
│   - Feature Map Size : [8, 8]                            │
│   - Scales           : [64, 128, 256]                    │
│   - Aspect Ratios    : [0.5, 1.0, 2.0]                   │
│   - Total Anchors    : 576                               │
│                                                          │
│ Convolution Block 7                                      │
│   - Feature Map Size : [8, 8]                            │
│   - Scales           : [128, 256, 512]                   │
│   - Aspect Ratios    : [0.5, 1.0, 2.0]                   │
│   - Total Anchors    : 576                               │
│                                                          │
│ Total Network Anchors: 3456                              │
╰──────────────────────────────────────────────────────────╯
╭───────────────────────────── Mask Architecture ──────────────────────────────╮
│ Normalization Strategy: Batch                                                │
│ Upsampling: Transposed                                                       │
│                                                                              │
│ Mask Block 5                                                                 │
│   - Modules            : Residual Shortcuts                                  │
│   - Convolution Layers : 4                                                   │
│   - Feature Map Size   : [16, 16]                                            │
│   - RoI Align Pool     : [14, 14]                                            │
│   - Doublings          : 1, to [28, 28]                                      │
│   - Mask Size          : [28, 28]                                            │
│   - Spatial Scale      : 0.0312                                              │
│                                                                              │
│ Mask Block 6                                                                 │
│   - Modules            : Residual Shortcuts                                  │
│   - Convolution Layers : 4                                                   │
│   - Feature Map Size   : [8, 8]                                              │
│   - RoI Align Pool     : [14, 14]                                            │
│   - Doublings          : 1, to [28, 28]                                      │
│   - Mask Size          : [28, 28]                                            │
│   - Spatial Scale      : 0.0156                                              │
│                                                                              │
│ Mask Block 7                                                                 │
│   - Modules            : Residual Shortcuts                                  │
│   - Convolution Layers : 4                                                   │
│   - Feature Map Size   : [8, 8]                                              │
│   - RoI Align Pool     : [14, 14]                                            │
│   - Doublings          : 1, to [28, 28]                                      │
│   - Mask Size          : [28, 28]                                            │
│   - Spatial Scale      : 0.0156                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────── Model Size ───────────────────────╮
│ Backbone                                                 │
│   - Parameters           : 2,245,216                     │
│   - Trainable Parameters : 2,245,216                     │
│   - Estimated Size       : 9.00 MB                       │
│                                                          │
│ Region Proposal Heads                                    │
│   - Parameters           : 495,239                       │
│   - Trainable Parameters : 495,239                       │
│   - Estimated Size       : 1.98 MB                       │
│                                                          │
│ Detection Heads                                          │
│   - Parameters           : 2,869,179                     │
│   - Trainable Parameters : 2,869,179                     │
│   - Estimated Size       : 11.48 MB                      │
│                                                          │
│ Mask Heads                                               │
│   - Parameters           : 571,839                       │
│   - Trainable Parameters : 571,839                       │
│   - Estimated Size       : 2.29 MB                       │
│                                                          │
│ Total Model                                              │
│   - Parameters           : 6,181,473                     │
│   - Trainable Parameters : 6,181,473                     │
│   - Estimated Size       : 24.76 MB                      │
╰──────────────────────────────────────────────────────────╯
```

### Step 2.17 — `view_input_batches()` → `get_input_sample_image()` → `overlay_instance_masks()`
With `diagnosis_modes.view_input_images` on, the input viewer of the detection (`build_input_windows()` of `DetectionTrainer`, see `object_detection_workflow.md`, Step 2.20) draws the boxes over the masks of the objects, every mask filled with the colour of its class by `overlay_instance_masks()` (`view_input_data()` of `vision_training/utilities/visualization.py` showing the windows). The input samples written at the start of every training launch (`Input_Samples/`) are drawn the same way. Its batches come from the loaders without `prepare_batch()`: `augment_batch()` expands the maps of the training batch, and `get_input_sample_image()` the map of the validation image. The configuration turns it off.

### Step 2.18 — `analyse_training_data()` → anchor recall
The anchor recall of the detection (see `object_detection_workflow.md`, Step 2.21), on the boxes of the segmentations, the boxes the detector learns: `gather_object_boxes()` reads them with `select_image_objects(boxes_from_segmentation=True)`, since the dataset loads the instance masks. It reads the manifest only, so the 358 objects are measured without decoding any mask.

**Example output**
```text
---------------------------------------------
Anchor Recall Of The 358 Training Objects...
---------------------------------------------
Anchor Recall Of The Training Objects
 Objects               Count    Block 5   Block 6   Block 7   All Levels
─────────────────────────────────────────────────────────────────────────
 All Objects             358    88.55 %   53.07 %   44.41 %      88.55 %
 Small (< 32²)             2     0.00 %    0.00 %    0.00 %       0.00 %
 Medium (32² to 96²)     164    76.22 %   25.61 %    6.71 %      76.22 %
 Large (>= 96²)          192   100.00 %   77.08 %   77.08 %     100.00 %
Share of the objects whose best anchor overlaps them by at least 0.5 (RegionProposal.foreground_iou_threshold.min),
sizes by box area in network input pixels. An object under RegionProposal.foreground_iou_threshold.min in every level
only gets its single best anchor as foreground. — : no object of the size.
```
The table writes its recalls as percentages; it is saved in `anchor_recall.csv` (263 bytes), with the recalls between 0 and 1, and sent to MLflow as an analysis file.

### Step 2.19 — `restore_last_model()`
**Example output** (new run folder)
```text
-----------------------------------------------------------
No Initiation Model Weights Will Be Used...
We Generate A New Model That Will Be Trained From Scratch.
-----------------------------------------------------------
```

---

## Stage 3 — Training Loop

> The outputs of the run below (Steps 3.1 to 3.5, 3.14, 3.17 and 3.18) are real outputs of the 5000-iteration run of [`run_configurations/instance_segmentation_workflow_run.yaml`](run_configurations/instance_segmentation_workflow_run.yaml), trained on the GPU on 2026-10-24. The commands of `Run It` launch the synthetic configuration, which differs only by its 2000 iterations, its checkpoint evaluated every 250 iterations (an intermediate one every 125), its MLflow tracking turned off, its experiment folder, its augmentation probabilities (0.25 for `random_haze`, 0.3 in this run) and `Data.instance_index_maps` (`True`, `False` in this run, both giving the same masks). The run name is `INS_IN-KAR-512_SE-RES-BN-B1x16.2x16.2x16.3x32.3x32.5x32.5x32_L567_A6.32-256.e7b72c_RBN-R2x32_DSE-DRES-DBN-D7.2x32.64.32_MRES-MBN-MUTC-M14.3x32-MR28_MC20`, written `<run>` below; with batches of 64 images an epoch is 2000 iterations, a console summary is printed every 1000 iterations (`information_dump`) and no validation batch runs (`validation_iteration_ratio: 0`). The 5000 iterations with their 20 evaluations took 2 h 32 min, from 15:19 to 17:51, the GPU being shared with the DET run of `object_detection_workflow.md` until 16:15.
>
> The examples of one batch going through the model (Steps 3.6 to 3.13, 3.15 and 3.16) come from the earlier run on CPU of the redirected configuration of Stages 1 and 2 shortened to batch size 2, 12 iterations, a checkpoint every 6 iterations, a console summary every 4 iterations, a validation batch every 10 iterations, 2 loading workers, with the MLflow tracking on. Its model is the one of Stages 1 and 2 (64 and 128 filters, 9 anchors per position, 3456 anchors, mask heads of `[4, 64]`, run name of Step 2.4).

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
The loop of every task (see `classification_workflow.md`, Step 3.1): from `start_iteration` to `number_iterations`, the total of the run, each iteration takes its training step, then saves its checkpoint, runs its validation batch and prints its summary on the iterations set for each, and evaluates its checkpoint last.

**Example: what happens at each iteration of the run**, in the order of the rows within an iteration, after its training step (the iterations between the columns only take their training step)

| Iteration | 250 | 500 | 750 | 1000 | 1250 | 1500 | ... | 4500 | 4750 | 5000 | after the loop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Checkpoint saved | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ... | ✔ | ✔ | ✔ | nothing: iteration 5000 is saved |
| Validation batch | | | | | | | | | | | |
| Console summary | | | | ✔ | | | ... | | | ✔ | |
| Test evaluation, trained then averaged weights | | ✔ | | ✔ | | ✔ | ... | ✔ | | ✔ | nothing |

The checkpoint of iteration 500 is saved right after its training step, and evaluated at the end of iteration 500.

### Step 3.2 — `setup_tensorboard_writers()` / `initialize_training_loop_state()`
**Example: the curves after the run** (no `Validation` writer, since no validation batch runs)
```text
Tensorboard/Train/        Total Loss, RPN Classification Loss, RPN Regression Loss, Detection Classification Loss,
                          Detection Regression Loss, Mask Loss, Learning Rate                        (iterations 1 to 5000)
Tensorboard/Test/         126 scalars and 9 images per evaluation (Step 4.10)                        (iterations 500 to 5000)
Tensorboard/Test_EMA/     the same tags for the averaged weights                                     (iterations 500 to 5000)
```

### Step 3.3 — `start_experiment_tracking()`
**Example output**
```text
- MLflow Tracking         : sqlite:///.../experiments/workflow_runs/mlflow.db                <- launch information
MLflow Run Created In The Experiment 'instance_segmentation' : 4c0f509b4cfd43ccac007e094fb5769e (Browse : mlflow ui --backend-store-uri sqlite:///.../experiments/workflow_runs/mlflow.db)
```
The identifiers are also written in `<run>/mlflow_run.json` (`tracking_uri`, `experiment_id` "2", `run_id`), so that a resumed run logs to the same MLflow run, and a standalone evaluation (Stage 5) tags its own MLflow run with it (`evaluation_of`). With the current code, the `Input_Samples/` written before the first iteration are sent as soon as the run is open, next to the configuration, `anchor_recall.csv` and `data.json`.

### Step 3.4 — `save_model()` → `save_checkpoint()`
Saved right after the training step of every multiple of 250 (the 10 evaluated checkpoints every 500 iterations and the 10 intermediate ones between them), so that the checkpoint of iteration N holds N steps (see `classification_workflow.md`, Step 3.6), with the label dictionary of the run (`label_dictionary`). The checkpoint holds the source run of every transferable component of the task, here four: `pretrained_backbone`, `pretrained_region_proposer_dictionary`, `pretrained_detector_heads_dictionary` and `pretrained_mask_heads_dictionary` (None without transfer). A run keeping a moving average of the weights, as the synthetic configuration does, also saves its state (`exponential_moving_average_state`).

**Example: the checkpoints and the keys of the last one**
```text
Checkpoint Folder :
 - Iteration_5000
 - Model Saved In : model_005000.pt

Weights/Iteration_250/model_000250.pt    12,387,623 bytes      Weights/Iteration_500/model_000500.pt    12,387,623 bytes
...
Weights/Iteration_4750/model_004750.pt   12,387,623 bytes      Weights/Iteration_5000/model_005000.pt   12,387,623 bytes
Weights/full-checkpoint:  model_checkpoint_path: "Iteration_5000", all_model_checkpoint_paths: "Iteration_250" ... "Iteration_5000"

iteration 5000   format_version 2   task 'instance_segmentation'   run_name '<run>'   mlflow_run_id '4c0f509b4cfd43ccac007e094fb5769e'
pretrained_backbone None   pretrained_region_proposer_dictionary None   pretrained_detector_heads_dictionary None
pretrained_mask_heads_dictionary None
model_state: 402 tensors, 75 of them in the mask heads
  mask_heads_dictionary.5.upsampling_layers.0.weight   (32, 32, 2, 2)
  mask_heads_dictionary.5.mask_predictor.weight        (21, 32, 1, 1)
optimizer_state: {'state', 'param_groups'}
exponential_moving_average_state: {'model_state', 'number_updates'}, number_updates 5000
```
The checkpoints written by the current code also hold `pretrained_backbone_blocks` (the backbone blocks loaded from a shorter source backbone, `classification_workflow.md`, Step 2.19), which would be `None` for this run.

### Step 3.5 — `run_test_evaluation()`
At every multiple of 500, last in the iteration, after its checkpoint and its summary (see `classification_workflow.md`, Step 3.7), once for the trained weights and once for the moving average of the weights, see Stage 4.

### Step 3.6 — `get_next_batch()` → `CocoDetectionDataset.__getitem__()` + `collate_function()`
A batch whose loading fails (an image that `read_image()` cannot read, Step 2.9) is dropped and the next one taken (see `classification_workflow.md`, Step 3.8). Every sample goes through `add_instance_index_map()`, or `add_instance_masks()` without `Data.instance_index_maps` (Step 2.9). The images and the maps are stacked; the boxes, labels and instance masks stay lists, one tensor per image.

**Example: the first batch of the CPU run** (recorded with one mask per object, before `Data.instance_index_maps`; with the maps the batch holds `instance_index_maps (2, 512, 512) uint8` instead of `instance_masks`)
```text
images               (2, 3, 512, 512) torch.uint8
bounding_boxes       list of 2: (4, 4) float32, (3, 4) float32
labels               list of 2: (4,) int64, (3,) int64       [14, 16, 6, 16]   [20, 13, 17]
sample_identifiers   (2,) torch.int64
instance_masks       list of 2: (4, 512, 512) uint8, (3, 512, 512) uint8
mask pixels          [9155, 2352, 5597, 4341]   [7163, 6094, 2600]
```

### Step 3.7 — `run_model_iteration()` → `prepare_batch()`
The detection preparation (images cast to float32 and moved to the device, boxes and labels moved), plus the instance masks in uint8: the instance index maps are moved to the device and expanded there by `expand_instance_index_maps()`, one mask per object number in the order of the labels (`map == k`), or the masks of the loader are moved as they are. The prepared batch holds `instance_masks` either way, and no maps.

**Example: after the preparation**
```text
images (2, 3, 512, 512) torch.uint8 -> torch.float32
instance_masks list of 2: (4, 512, 512) uint8, (3, 512, 512) uint8      <- unchanged
```

### Step 3.8 — `augment_batch()` → `DataAugmenter.augment_batch(bounding_boxes=..., labels=..., instance_masks=...)`
The geometric augmentations move the masks with the image, as one more channel stack, with the nearest interpolation. The box of every object is then replaced by the tight box of its moved mask (`compute_mask_bounding_boxes()`): a rotated box would otherwise become the larger box enclosing its corners. The share of every object left inside the image is measured on its mask pixels instead of the box areas: its moved mask pixels over its original mask pixels times the area scale of the geometric types applied (the zoom of a crop or a zoom-out), the pixels it would have at the new scale. `clean_bounding_boxes()` then removes, with their labels and masks, the objects left narrower or shorter than 1 pixel or with less than 25 % of these pixels (`MINIMUM_VISIBLE_AREA_FRACTION`). The photometric augmentations leave the masks unchanged. This configuration turns every type off, so nothing changes.

**Example: the augmentation report of the first batch**
```text
[{'augmented': False, 'geometric_augmentations': [], 'photometric_augmentations': [], 'removed_labels': None},
 {'augmented': False, 'geometric_augmentations': [], 'photometric_augmentations': [], 'removed_labels': None}]
instance_masks: (4, 512, 512) uint8, (3, 512, 512) uint8
```

### Step 3.9 — `forward_pass()` → `InstanceSegmenter.forward(ground_truth_bounding_boxes=..., ground_truth_labels=..., ground_truth_masks=...)`
`forward()` runs `forward_detection_stages()`, the whole two-stage detector (Step 3.10), then, with the ground truth, `compute_proposal_mask_loss()` (Steps 3.11 to 3.13), and in evaluation mode `predict_detection_masks()` (Step 4.3). The ground truth boxes cannot be given without their masks: `ValueError: The instance segmenter needs the ground truth masks with the ground truth boxes.` The trainer sums every output key ending in `_loss` into the training loss, each multiplied by its weight in `Optimization.loss_weights` (`run_model_iteration()`, 1 in the shipped configurations).

**Example: the outputs of the first iteration of the CPU run**
```text
classification_scores                 (2, 128, 21) float32
box_regressions                       (2, 128, 21, 4) float32
detection_boxes                       (2, 128, 21, 4) float32
sliced_origins                        (2, 128) int64
region_proposal_classification_loss   0.6610
region_proposal_localisation_loss     0.7074
detection_classification_loss         2.9878
detection_regression_loss             0.4909
mask_loss                             0.7130
total                                 5.5602
```

### Step 3.10 — `forward_detection_stages()`
The forward pass of the detector (`two_stage_detector.py`), unchanged: backbone, region proposals of every level, aggregation with the anchors, proposal filter (one selection over all the levels, `RegionProposal.proposal_selection: "joint"`), target assignment and sampling of 128 proposals per image, detection heads, and the four detection losses (see `object_detection_workflow.md`, Steps 3.10 to 3.18). It also returns what the mask branch needs: the feature maps of the detection blocks and the sampled proposals with their labels, levels and matched ground truths.

### Step 3.11 — `compute_proposal_mask_loss()`
As in Mask R-CNN, the masks are learned on the positive sampled proposals only (label > 0), the proposals the detector learns to classify as an object. Each level runs its mask head on its own positive proposals, read with `sliced_origins` (the level of every proposal), even when it has none, so that every mask head stays in the graph. With finer inputs (`MaskHead.image_branch_convolutions` or `skip_blocks_at_mask_resolution`), `MaskFineInputs` first computes the fine features of the positive proposals of all the levels at once, so that the batch statistics of the image branch cover every box, and every level receives the rows of its own proposals. The logits of all the levels, their labels, proposals and matched ground truth indices are concatenated, `build_mask_targets()` crops the targets (Step 3.12) and `compute_mask_loss()` compares them (Step 3.13).

**Example: the first iteration of the CPU run**
```text
sliced_proposals                      (256, 5) float32     [image index, x1, y1, x2, y2] of the 2 x 128 sampled proposals
sliced_labels                         (256,) int64
sliced_origins                        (256,) int64
sliced_matched_ground_truth_indices   (256,) int64
sampled 256, positive 12
level 5: feature map (2, 64, 16, 16)   7 positive proposals -> mask logits (7, 21, 28, 28)
level 6: feature map (2, 128, 8, 8)    3 positive proposals -> mask logits (3, 21, 28, 28)
level 7: feature map (2, 128, 8, 8)    2 positive proposals -> mask logits (2, 21, 28, 28)
```

### Step 3.12 — `build_mask_targets()`
The target of a proposal is the mask of its matched ground truth object, cropped to the proposal and resized to the mask resolution by `roi_align()` on the full-size binary masks (`spatial_scale=1`, `aligned=True`), then thresholded at 0.5. The target is the part of the object inside the proposal, in the frame of the proposal, which is where the mask head predicts it.

**Example: the 12 positive proposals of the first iteration of the CPU run**
```text
proposal_boxes (12, 5) float32, 6 proposals in each image
first proposal [0, 180, 303, 293, 413] (image 0), matched ground truth 0
mask_targets (12, 28, 28) float32, values {0.0, 1.0}, foreground share 0.5178
first target: 578 foreground pixels of 784
```

**Example: a toy target** (a ground truth square filling the box [100, 200, 156, 256] of a 512 pixel image, cropped by a proposal twice its size)
```text
target (1, 28, 28): 196 foreground pixels, rows 7 to 20, columns 7 to 20      <- the 14 x 14 centre of the 28 x 28 target
```

### Step 3.13 — `compute_mask_loss()`
The binary cross-entropy with logits, averaged over the pixels of the positive proposals, between the target and the logit map of the **ground truth class** of every proposal: the other 20 channels receive no gradient, so that the classes do not compete for a pixel (the detection head decides the class). The loss is computed in float32, also under the bfloat16 autocast. Without any positive proposal it returns `sum(logits) * 0`, a zero that keeps the mask heads in the graph.

**Example: the first iteration of the CPU run**
```text
mask_logits (12, 21, 28, 28) float32   labels [14, 16, 6, 16, 20, 13, 17, 14, 13, 20, 6, 20]
class_logits (12, 28, 28)              <- the channel of the label of every proposal
mask_loss 0.7130 torch.float32         <- close to ln 2 = 0.693, the loss of a logit of 0
```

### Step 3.14 — `compute_batch_metrics()` + `MetricTracker.accumulate()` + `log_to_tensorboard()`
The detection has no batch metric besides its losses: the 6 tracked values (5 losses and the total) are accumulated and written to `Tensorboard/Train/` at every iteration.

**Example: the losses of the run** (one point per iteration, a few shown)
```text
iteration   total    RPN class.   RPN regr.   det. class.   det. regr.   mask
    1       5.6910   0.7085       0.8004      3.0862        0.3976       0.6982
    2       5.5667   0.7044       0.7212      3.0526        0.3855       0.7029
    3       5.5125   0.7030       0.6934      3.0216        0.3950       0.6995
 1000       1.0460   0.0254       0.1053      0.5723        0.1352       0.2077
 2000       0.7084   0.0114       0.0632      0.3687        0.0930       0.1720
 3000       0.6320   0.0125       0.0557      0.3137        0.0870       0.1631
 4000       0.5690   0.0097       0.0572      0.2644        0.0769       0.1609
 5000       0.4678   0.0087       0.0492      0.1912        0.0648       0.1539
```
At iteration 1 the detection classification loss is close to ln 21 = 3.0445 and the mask loss close to ln 2 = 0.693, the losses of untrained logits. In 5000 iterations the detection classification loss falls to 0.1912, still falling at the end, and the mask loss to 0.1539, most of its fall happening in the first 1000 iterations (0.2077).

### Step 3.15 — `training_loss.backward()` + `step_optimizer()`
**Example: the gradient norm of every component in the CPU run** (the 5000-iteration run does not record the gradient norm)
```text
iteration 1:  backbone 26.0453   region_proposer_dictionary 6.8338   detector_heads_dictionary 12.6682   mask_heads_dictionary 0.3247
iteration 12: backbone 15.9901   region_proposer_dictionary 4.3910   detector_heads_dictionary 8.1084    mask_heads_dictionary 0.4148
```

### Step 3.16 — validation pass
The 5000-iteration run has no validation pass (`validation_iteration_ratio: 0`). In the CPU run, at iteration 10, one validation batch in `eval()` mode, without gradients, with the ground truth boxes, labels and masks: the model computes the same five losses (the mask loss on the positive sampled proposals), and in `eval()` mode also predicts the masks of its detections, which the validation does not use.

**Example: the validation batch of iteration 10 of the CPU run**
```text
total 4.8811   RPN classification 0.6847   RPN regression 0.4954   detection classification 2.8054   detection regression 0.2007
mask 0.6948
```

### Step 3.17 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()`
**Example output: the summaries of iterations 1000 and 5000**
```text
========================================================================================================================
|   Iteration: 1000    |    Total Images: 64000    |   Interval Images: 64000   |  Throughput: 33.5 Images Per Second  |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :     1.6639
Moving Average of Train RPN Classification Loss        :     0.1255
Moving Average of Train RPN Regression Loss            :     0.1913
Moving Average of Train Detection Classification Loss  :     0.7328
Moving Average of Train Detection Regression Loss      :     0.2472
Moving Average of Train Mask Loss                      :     0.3671
------------------------------------------------------------------------------------------------------------------------
Interval Of 1000 Iterations Took 00:31:50.
========================================================================================================================
...
========================================================================================================================
|   Iteration: 5000    |    Total Images: 320000   |   Interval Images: 64000   |  Throughput: 43.7 Images Per Second  |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :     0.4768    -0.0717 Best
Moving Average of Train RPN Classification Loss        :     0.0070    -0.0017 Best
Moving Average of Train RPN Regression Loss            :     0.0450    -0.0060 Best
Moving Average of Train Detection Classification Loss  :     0.2063    -0.0482 Best
Moving Average of Train Detection Regression Loss      :     0.0677    -0.0078 Best
Moving Average of Train Mask Loss                      :     0.1509    -0.0080 Best
------------------------------------------------------------------------------------------------------------------------
Interval Of 1000 Iterations Took 00:24:25.
========================================================================================================================
```
Without validation batch, the summaries have no `Valid` rows and no `Validation Iterations In This Interval` line. The interval time is the wall-clock time since the previous summary, so it includes the evaluations run in between (trained and averaged weights) and depends on the load of the shared GPU. Every average of the summaries of iterations 2000 to 5000 is lower than all the previous ones, so all their rows are marked `Best`; the mask loss averages 0.3671 over the first 1000 iterations and 0.1509 over the last 1000.

**Example: the MLflow metrics at the end of the run**
```text
train/Mask Loss 0.1509   train/interval_seconds 1465.099   train/throughput 43.6831   learning_rate 0.0002
status FINISHED   220 parameters   263 metrics
```

### Step 3.18 — end of the loop
The checkpoint of iteration 5000 is saved right after its training step and evaluated last in its iteration, so the loop saves nothing more and evaluates nothing (see `classification_workflow.md`, Step 3.25).

**Example output: the end of iteration 5000 and of the loop**
```text
Checkpoint Folder :                                      <- right after the training step of iteration 5000
 - Iteration_5000
 - Model Saved In : model_005000.pt
...                                                      <- the summary, then the evaluations of iteration 5000 (Stage 4)
-----------------------------------------------------------
Last Checkpoint Saved At Iteration 5000. Exiting Training.
-----------------------------------------------------------
```
Then `shutdown_loader_workers()` and `finish_experiment_tracking()`.

**Example: the run folder at the end**
```text
<run>/
├── Evaluation/Iteration_500/, Iteration_1000/, ..., Iteration_5000/    class_metrics.csv, class_thresholds.json,
│                                                        confusion_matrix.csv, confusion_matrix.png, proposal_recall.csv,
│                                                        Samples/, and the same files in EMA/
├── Tensorboard/Train/, Tensorboard/Test/, Tensorboard/Test_EMA/
├── Weights/Iteration_250/, Iteration_500/, ..., Iteration_5000/, full-checkpoint
├── Results.txt                                          the reports of the 20 evaluations (2492 lines)
├── anchor_recall.csv, data.json, label_dictionary.json, mlflow_run.json, training_configuration.yaml
```

---

## Stage 4 — Test Evaluation of a Checkpoint

> The reports, files, sample images and scalars below (the progress bars of Step 4.1, Steps 4.6 to 4.10) are real outputs of the evaluations of the 5000-iteration run of Stage 3 ([`run_configurations/instance_segmentation_workflow_run.yaml`](run_configurations/instance_segmentation_workflow_run.yaml)), mostly of its last checkpoint, iteration 5000, on its whole test split (20,000 images, 69,682 objects), with `per_class_report: True`, `number_sample_images: 8`, `score_threshold: 0.05`, `target_precision: 0.99`, the error images off and the MLflow tracking on. Every checkpoint of a multiple of 500 is evaluated twice, with the trained weights and with their moving average (`EMA Weights` blocks, `EMA/` folders). The synthetic configuration of `Run It` evaluates every 250 iterations up to 2000 instead. These outputs were captured before the per-class recall intervals were added (2026-11-01): `class_metrics.csv` now holds `recall_50`, `recall_50_lower`, `recall_50_upper`, `recall_75`, `recall_75_lower` and `recall_75_upper` after the box average precisions and the same columns prefixed with `mask_` after the mask ones, and with `per_class_report` the report shows a `Mask Per Class Recall` table after `Mask Per Class Average Precision (Area)` and a `Box Per Class Recall` table after `Box Per Class Average Precision (Area)` (Steps 4.7 and 4.8).
>
> The steps inside one test batch (Steps 4.1 to 4.5 and the first example of Step 4.7) come from the checkpoint of iteration 12 of the CPU run of Stage 3, restored in a copy of its run folder (`start_iteration` 13), on its test batch of the images 2 and 7 (`test_000001.png` and `test_000006.png`, 20 test images and 66 objects in all); this model finds no shape.

### Step 4.1 — `run_test_evaluation()` → `evaluation_context()`
The evaluation loop of every task (see `classification_workflow.md`, Step 4.1) with the `evaluation_context()` of the detection, which lowers the score threshold of the model to `evaluation.score_threshold` during the evaluation and restores the deployment one afterwards.

**Example output** (the thresholds printed in the CPU run, the progress bars of iteration 5000 of the 5000-iteration run: 20,000 images, batch size 64)
```text
score threshold in the context: 0.05      after: 0.5
Test Evaluation Iteration 5000: 100%|██████████| 313/313 [02:18<00:00,  2.25it/s]
Test Evaluation Iteration 5000 (EMA Weights): 100%|██████████| 313/313 [02:15<00:00,  2.30it/s]
```

### Step 4.2 — `create_evaluation_accumulator()`
The accumulator of the detection (see `object_detection_workflow.md`, Step 4.2); the detections and ground truths of every image also keep their mask overlaps and areas (Step 4.5).

**Example output: the CPU run**
```text
{'detections': list, 'ground_truths': list, 'confusion_matrix': (21, 21), 'errors_folder': PosixPath, 'error_counts': dict,
 'proposal_recall': dict}
```

### Step 4.3 — `prepare_batch()` + `predict_batch()` → `predict_detection_masks()`
`prepare_batch()` prepares the test batch as in training (Step 3.7): with `Data.instance_index_maps`, the maps of the test loader are expanded on the device into `instance_masks`, which the evaluation reads; the test loader of `evaluation.original_resolution` has no maps and gives one mask per object at both resolutions (Step 2.8). `predict_batch()` runs the model without the ground truth, as in deployment. After the post-processing of the detections (see `object_detection_workflow.md`, Step 4.4), `predict_detection_masks()` runs the mask heads on the **final boxes**, refined by the detection head, as in Mask R-CNN where the masks follow the refined boxes: the level of every detection is the level of its proposal (`sliced_origins` read at `final_detection_proposal_indices`), every level runs its mask head on its own valid detections, and the sigmoid of the channel of the detected class is kept; with finer inputs, their fine features are computed once on the valid detections of all the levels. The padding detections get a mask of 0. The masks are 28 x 28 in the frame of their box; they are pasted into the image only for the evaluation (Step 4.5) and the sample images (Step 4.6).

**Example: the predictions of the test batch of the CPU run** (recorded with one mask per object, before `Data.instance_index_maps`; with the maps the loader batch holds `instance_index_maps (2, 512, 512) uint8`, expanded by `prepare_batch()` into the same `instance_masks`)
```text
test batch: sample identifiers [2, 7], instance_masks (2, 512, 512) and (1, 512, 512), labels [[19, 5], [16]]
classification_scores              (2, 250, 21) float32
box_regressions                    (2, 250, 21, 4) float32
detection_boxes                    (2, 250, 21, 4) float32
sliced_origins                     (2, 250) int64
sliced_valid_mask                  (2, 250) bool
final_detection_boxes              (2, 100, 4) float32
final_detection_scores             (2, 100) float32
final_detection_labels             (2, 100) int64
number_detections                  (2,) int64        [100, 100]
final_detection_proposal_indices   (2, 100) int64
proposal_boxes                     (2, 250, 4) float32
proposal_levels                    (2, 250) int64
valid_proposal_mask                (2, 250) bool
final_detection_masks              (2, 100, 28, 28) float32
image 0: mask probabilities from 0.4792 to 0.5132, mean 0.4974; levels of the detections {7: 100}
first detections: boxes [[3.7, 264.9, 60.7, 388.8], [3.7, 200.7, 60.3, 324.7], [3.7, 331.4, 61.1, 452.9]]
                  scores [0.0528, 0.0528, 0.0528]   labels [20, 20, 20] (l_shape)
```
After 12 iterations the mask probabilities of the CPU run stay around 0.5, so a pasted mask is a rough blob around the middle of its box, and every detection scores just above the evaluation threshold.

### Step 4.4 — `postprocess_detections()`
The post-processing of the detection, unchanged (see `object_detection_workflow.md`, Step 4.4), with the greedy NMS of `Detector.nms_criterion: "iou"`, which the exported `InferenceInstanceSegmenter` (an `InferenceDetector`) runs, like the proposal suppression, as one call over the batch (`suppress_stacked_groups()`: the groups moved apart in TorchScript, one `NonMaxSuppression` batch per group in ONNX), the form the exported detector keeps for any batch size: in the CPU run, 100 detections per image at the evaluation threshold 0.05, none at the deployment threshold 0.5. At iteration 5000 of the 5000-iteration run, 45,049 detections of the 20,000 test images reach 0.5 (Step 4.9).

### Step 4.5 — `accumulate_evaluation_batch()` → `gather_image_detections()` → `compute_mask_overlaps()`
For every image, `gather_image_detections()` keeps the valid detections (boxes, scores, labels) with two more entries:
- `mask_overlaps` [D, G] float64: the mask IoU of every detection with every ground truth object of the image, whatever their classes, computed by `compute_mask_overlaps()` a chunk of detections at a time: 16 masks of a 1024 x 1024 image (`MASK_OVERLAP_CHUNK_SIZE`), scaled by `MASK_OVERLAP_REFERENCE_PIXELS` over the pixels of the image, so 64 at 512 x 512 (the example below was recorded when the chunk was 16 at every size):
  - `paste_instance_masks()` pastes the 28 x 28 masks into the image at their boxes, by bilinear resampling written as two products with separable tent weights (`build_interpolation_weights()`: one [image size, 28] weight matrix for the rows and one for the columns of every box), so that the pasting runs in one batched matrix product on the device;
  - the pasted masks are thresholded at `mask_threshold` (`>= 0.5`);
  - `compute_mask_intersection_over_union()` gives the IoU of every binary detection mask with every ground truth mask: `cut_mask_segments()` cuts the flattened masks into segments of `MASK_PRODUCT_SEGMENT_PIXELS` (1024) pixels, and one batched product multiplies every segment of the chunk with the same segment of every ground truth mask, in float16 on the GPU (exact, a partial count never exceeding 1024, below the 2048 float16 holds exactly; float32 on the CPU). On 1024 x 1024 test images of the benchmark dataset with 100 detections, the mask part of the evaluation takes 16 ms per image instead of 96 with one float32 product over the whole image, which left most of the GPU idle, the overlaps being equal bit for bit. The partial counts and the mask areas (`count_segment_pixels()`) are summed and divided in float64 (in float32, 13 / 20 falls below 0.65, which would move a match across an IoU threshold).
- `mask_areas` [D] int64: the pixels of every binary detection mask, for the mAP by object size.

`gather_image_ground_truth()` adds the `mask_areas` of the ground truth objects, counted on the `instance_masks` of the prepared batch (expanded from the maps with `Data.instance_index_maps`), or on the `original_instance_masks` with `evaluation.original_resolution`. The detections reaching the deployment threshold are then matched to the objects for the confusion matrix with `compute_overlap_matrix(overlap_type="masks")`, which returns the mask overlaps instead of the box IoUs (`MATCHING_OVERLAP_TYPE`), and `match_image_detections()` at IoU 0.5 (see `object_detection_workflow.md`, Step 4.5). `accumulate_proposal_recall()` keeps measuring the proposals on the boxes.

**Example: the first image of the test batch of the CPU run** (2 objects, labels capsule (19) and pentagon (5))
```text
pasted masks of the first chunk: (16, 512, 512) float32, max 0.5121
first detection: box area 7058.3 pixels, 7552 pasted pixels > 0, 6523 pixels >= 0.5
detections:   {'boxes': (100, 4) float32, 'scores': (100,) float32, 'labels': (100,) int64, 'mask_overlaps': (100, 2) float64,
               'mask_areas': (100,) int64}
ground truth: {'boxes': (2, 4) float32, 'labels': (2,) int64, 'mask_areas': (2,) int64}
best mask IoU per object [0.2133, 0.3883]       best box IoU per object [0.3533, 0.508]
detection mask_areas [6523, 6377, 6380, 6522, 13083, 13287, 13218, 13404, ...]   38 of the 100 masks are empty at 0.5
ground truth mask_areas [3310, 5961]            box areas [5829, 8835]
after the batch: confusion matrix sum 3, all 3 objects in the background column (missed)
```
The pentagon is overlapped at 0.508 by a box but only at 0.3883 by a mask: matched on the masks, the detection does not find it at 0.5. The mask areas are smaller than the box areas, which moves objects from one size to another (Step 4.7).

**Example: a toy mask pasted at its box** (a 28 x 28 mask whose 14 x 14 centre is 1, pasted into the box [100, 200, 156, 256])
```text
pasted mask: 784 pixels, rows 214 to 241, columns 114 to 141      <- the 28 x 28 centre of the 56 x 56 box
IoU with a ground truth mask equal to that square and with an empty one: [[1.0, 0.0]]
```

### Step 4.6 — `save_sample_images()` → `draw_test_image()` → `get_sample_masks()` + `build_detection_sample_grid()`
The sample images of the detection (see `object_detection_workflow.md`, Step 4.6): two panels side by side, the ground truth and the detections reaching the deployment threshold. `get_sample_masks()` adds the masks: the ground truth masks of the image, and the masks of the kept detections pasted at their boxes and thresholded at `mask_threshold`, filled with the colour of their class under the boxes.

**Example: `Evaluation/Iteration_5000/Samples/`, and the content of `test_014285.png`**
```text
test_000000.png, test_002857.png, test_005714.png, test_008571.png, test_011428.png, test_014285.png, test_017142.png, test_019999.png
every file 1028 x 560 pixels RGB

test_014285.png:     "6 ground truth boxes | 4 detections with score >= 0.50"
                     Ground Truth (square, triangle, star, parallelogram, 2 nonagon masks and boxes) |
                     Prediction (square 0.97, triangle 0.75, star 0.99, parallelogram 0.87, with their masks)
EMA/Samples/test_014285.png: "6 ground truth boxes | 3 detections with score >= 0.50" (square 0.96, star 0.99, parallelogram 0.82)
```
Both nonagons are missed by the trained and the averaged weights, and the averaged weights also miss the triangle. The predicted masks follow the shapes, but the tips of the star come out rounded and the thin end of the triangle is left out of its box and mask: 28 x 28 masks pooled from feature maps of stride 32 and 64 lose the thin parts (`Read This First`, Pitfalls).

### Step 4.7 — `compute_evaluation_results()`
The evaluation of the detection computed twice on the same detections:
- `summarise_average_precisions(overlap_type="masks", title_prefix="Mask", name_prefix="Mask ", column_prefix="mask_")`: the mask mAP, the detections matched to the objects of their class by mask IoU, with the three methods and by object size, the sizes measured on the **mask areas** (`select_object_size_detections()` reads `mask_areas`).
- `summarise_average_precisions(overlap_type="boxes", title_prefix="Box")`: the box mAP of the detection, the sizes measured on the box areas, comparable with the object detection runs.

Then `save_proposal_recall()` (see `object_detection_workflow.md`, Step 4.15), `write_class_metrics_file()` with the box and mask columns of every class (Step 4.8), and `save_error_analysis()` with the score thresholds of the **mask** average precisions (Step 4.9). The report holds eleven tables in this order: mask mAP by method, mask mAP by size, box mAP by method, box mAP by size, mask mean average recall, box mean average recall (see `object_detection_workflow.md`, Step 4.14), proposal recall, mask per class, mask per-class recall (`Mask Per Class Recall`), box per class, box per-class recall (`Box Per Class Recall`), the recall tables giving the recall of every class at 0.5 and 0.75 with its 95 % confidence interval (see `object_detection_workflow.md`, Steps 4.11 and 4.13). With `evaluation.original_resolution`, `gather_image_detections()` restores the detection boxes to the original image and pastes the masks at that size, and `gather_image_ground_truth()` takes the boxes and mask areas of the original annotations, so that the mean average precisions and recalls, and their sizes, are those of the original images. The checkpoints are compared on `Mask mAP@0.5/area`, which the results also list in `percent_metrics` (`['Mask mAP@0.5/area']`, the report having no global metric) so that the best checkpoint line writes it as a percentage (Step 5.5). Every table lists its `percent_columns`: the report writes the average precisions, average recalls, proposal recalls, recalls at the target precision and per-class recalls with their bounds as percentages with two decimals (`63.72 %`), the target precision of the headers and legends as `99 %` and the confidence level as `95 %`, while the object and ground truth counts, the score thresholds and the proposals per image keep their values and the rows and scalars stay between 0 and 1.

**Example: the results of `gather_class_detections()` and `evaluate_detections()`** (the keys of `evaluate_detections()` printed again with the current code, which adds the IoU 0.95 and the recalls)
```text
evaluate_detections keys: ['area', '11_point', '101_point', 'class_score_thresholds']
per method:               ['mAP@0.5', 'mAP@0.75', 'mAP@0.95', 'mAP@[0.5:0.95]', 'class_AP@0.5', 'class_AP@0.75',
                           'class_AP@0.95', 'class_AP@[0.5:0.95]', 'mAR@0.5', 'mAR@0.75', 'mAR@0.95', 'mAR@[0.5:0.95]',
                           'class_AR@0.5', 'class_AR@0.75', 'class_AR@0.95', 'class_AR@[0.5:0.95]',
                           'class_ground_truths', 'sizes']
gather_class_detections(class 19, capsule) on the test batch of the CPU run, masks and boxes alike:
  {'scores': (0,), 'image_indices': (0,), 'best_intersection_over_union': (0,), 'best_ground_truth_indices': (0,), 'detection_areas': (0,),
   'best_ground_truth_areas': (0,), 'ground_truth_areas': (1,), 'number_ground_truth': 1}
```
No detection of the batch is a capsule, so the class has 1 object and no detection.

**Example: the report of iteration 5000** (the per-class tables are shortened; the same text, without the file lines, is appended to `Results.txt`)
```text
Iteration 5000 · 20000 Test Images · 2026-10-24 17:48 ──────────────────────────────────────────────────────────────────
  Confusion Matrix : score >= 0.50, IoU >= 0.5 (masks), 32850 right, 8650 wrong class, 3549 false positives, 28182
  missed
  Sample Images : 8 fixed test images

  Mask Mean Average Precision
   Method      mAP@0.5   mAP@0.75   mAP@0.95   mAP@[0.5:0.95]
  ────────────────────────────────────────────────────────────
   Area        63.72 %    54.67 %     1.57 %          46.91 %
   11 Point    61.59 %    54.13 %     2.89 %          46.50 %
   101 Point   63.53 %    54.57 %     1.70 %          46.85 %
  The mask mAP@0.5 of the area method compares the checkpoints of the run.

  Mask Mean Average Precision By Object Size (Area)
   Object Size           Objects   mAP@0.5   mAP@0.75   mAP@0.95   mAP@[0.5:0.95]
  ────────────────────────────────────────────────────────────────────────────────
   Small (< 32²)             911    5.37 %     3.79 %     0.00 %           3.24 %
   Medium (32² to 96²)     50627   63.44 %    54.02 %     1.60 %          46.16 %
   Large (>= 96²)          18144   62.29 %    56.33 %     2.06 %          47.58 %
  COCO sizes by mask area in network input pixels, the objects of the other sizes and the detections finding them being
  ignored. — : no object of the size.

  Box Mean Average Precision
   Method      mAP@0.5   mAP@0.75   mAP@0.95   mAP@[0.5:0.95]
  ────────────────────────────────────────────────────────────
   Area        64.04 %    58.18 %     0.16 %          46.08 %
   11 Point    61.69 %    57.32 %     1.18 %          45.42 %
   101 Point   63.86 %    58.04 %     0.26 %          46.00 %
  Box average precisions of the same detections, comparable with the object detection runs.

  Box Mean Average Precision By Object Size (Area)
   Object Size           Objects   mAP@0.5   mAP@0.75   mAP@0.95   mAP@[0.5:0.95]
  ────────────────────────────────────────────────────────────────────────────────
   Small (< 32²)             184    0.66 %     0.00 %     0.00 %           0.12 %
   Medium (32² to 96²)     33579   62.54 %    56.21 %     0.22 %          44.48 %
   Large (>= 96²)          35919   66.01 %    60.42 %     0.19 %          48.15 %
  COCO sizes by box area in network input pixels, the objects of the other sizes and the detections finding them being
  ignored. — : no object of the size.

  Mask Mean Average Recall
   Objects Of            Objects   mAR@0.5   mAR@0.75   mAR@0.95   mAR@[0.5:0.95]
  ────────────────────────────────────────────────────────────────────────────────
   All                     69682   96.75 %    86.61 %     7.48 %          75.70 %
   Small (< 32²)             911   20.66 %     9.93 %     0.00 %          10.29 %
   Medium (32² to 96²)     50627   96.73 %    86.20 %     7.31 %          75.12 %
   Large (>= 96²)          18144   99.17 %    92.87 %     8.56 %          78.84 %
  Objects found by any detection kept for the evaluation, averaged over the classes. COCO sizes by mask area in network
  input pixels. — : no object of the size.

  Box Mean Average Recall
   Objects Of            Objects   mAR@0.5   mAR@0.75   mAR@0.95   mAR@[0.5:0.95]
  ────────────────────────────────────────────────────────────────────────────────
   All                     69682   97.36 %    89.34 %     3.57 %          74.19 %
   Small (< 32²)             184    4.06 %     0.00 %     0.00 %           0.99 %
   Medium (32² to 96²)     33579   96.36 %    87.34 %     3.23 %          72.27 %
   Large (>= 96²)          35919   98.73 %    91.69 %     3.92 %          76.27 %
  Objects found by any detection kept for the evaluation, averaged over the classes. COCO sizes by box area in network
  input pixels. — : no object of the size.

  Proposal Recall
                                                           Proposals
   Objects               Count   Recall@0.5   Recall@0.7   Per Image
  ───────────────────────────────────────────────────────────────────
   All Objects           69682      99.63 %      97.41 %    250.0000
   Small (< 32²)           184      35.87 %       8.15 %           —
   Medium (32² to 96²)   33579      99.67 %      96.72 %           —
   Large (>= 96²)        35919      99.93 %      98.52 %           —
   Block 5 Proposals     69682      78.11 %      52.29 %    149.2502
   Block 6 Proposals     69682      80.50 %      21.68 %     50.6018
   Block 7 Proposals     69682      70.44 %      50.04 %     50.1480
  Share of the test objects overlapped by a proposal of the region proposal network by at least the IoU, sizes by box
  area in network input pixels. A block row keeps the proposals of that block only. — : no object of the size.

  Mask Per Class Average Precision (Area)
                   Ground                                                 Threshold    Recall
   Class           Truths    AP@0.5   AP@0.75   AP@0.95   AP@[0.5:0.95]     @ P99 %   @ P99 %
  ────────────────────────────────────────────────────────────────────────────────────────────
   heptagon          3466   26.85 %   26.06 %    0.15 %         22.17 %           —         —
   octagon           3454   27.64 %   26.34 %    0.49 %         22.48 %           —         —
   trapezoid         3546   29.24 %   25.14 %    0.02 %         20.50 %           —         —
   rhombus           3436   33.85 %   31.89 %    0.27 %         27.00 %      0.4817    0.06 %
   nonagon           3533   34.75 %   33.73 %    1.88 %         29.29 %           —         —
   hexagon           3471   35.08 %   33.93 %    0.56 %         28.99 %           —         —
   ...
   rectangle         3494   93.69 %   87.46 %    0.09 %         70.06 %      0.8980   74.79 %
   square            3545   96.24 %   90.49 %    0.17 %         73.71 %      0.7861   85.22 %
   plus_sign         3442   96.46 %   78.93 %    0.00 %         61.44 %      0.7324   90.96 %
   star              3549   97.29 %   30.22 %    0.00 %         48.31 %      0.8004   91.91 %
   annulus           3461   97.69 %   81.02 %    0.00 %         66.44 %      0.7104   92.40 %
  Worst AP@0.5 first, ties in class order. — : no ground truth box of the class, or the precision 99 % is never reached.

  Box Per Class Average Precision (Area)
                   Ground
   Class           Truths    AP@0.5   AP@0.75   AP@0.95   AP@[0.5:0.95]
  ──────────────────────────────────────────────────────────────────────
   heptagon          3466   26.94 %   25.57 %    0.03 %         20.17 %
   octagon           3454   27.69 %   24.50 %    0.01 %         19.11 %
   trapezoid         3546   29.35 %   24.28 %    0.04 %         19.63 %
   ...
   star              3549   98.11 %   89.34 %    0.01 %         66.11 %
   annulus           3461   98.50 %   94.35 %    0.28 %         74.10 %
  Worst AP@0.5 first, ties in class order. — : no ground truth box of the class.
  - Results : Results.txt
  - Class Metrics : class_metrics.csv
  - Proposal Recall : proposal_recall.csv
  - Class Thresholds : class_thresholds.json
  - Confusion Matrix Table : confusion_matrix.csv
  - Confusion Matrix Image : confusion_matrix.png
  - Sample Images : Samples
```
By mask area, 911 of the 69,682 test objects are small, 50,627 medium and 18,144 large; by box area, 184, 33,579 and 35,919: a shape fills only part of its box. The mask and box mAP@0.5 are close (63.72 % and 64.04 %), the same detections finding their objects by masks and by boxes alike at IoU 0.5, but the masks fall behind at 0.75 (54.67 % against 58.18 %), most for the thin shapes: the star has a mask AP@0.75 of 30.22 % against a box AP@0.75 of 89.34 %. The mask mAP@0.95 (1.57 %) is above the box one (0.16 %), carried mostly by the circle (mask AP@0.95 23.19 % against a box AP@0.95 of 1.12 %). Only the mask per-class table has the threshold columns, the class thresholds being computed on the mask average precisions. The weakest classes are the polygons with many sides (heptagon, octagon, nonagon, hexagon) and the quadrilaterals (trapezoid, rhombus, kite), each below a mask AP@0.5 of 39 %.

**Example: the methods tables of the trained weights at iteration 4500 and of the averaged weights at iteration 5000**
```text
Iteration 4500 · 20000 Test Images · 2026-10-24 17:36 ──────────────────────────────────────────────────────────────────
  Confusion Matrix : score >= 0.50, IoU >= 0.5 (masks), 32234 right, 3529 wrong class, 2956 false positives, 33919
  missed
  Sample Images : 8 fixed test images

  Mask Mean Average Precision
   Method      mAP@0.5   mAP@0.75   mAP@0.95   mAP@[0.5:0.95]
  ────────────────────────────────────────────────────────────
   Area        69.49 %    60.22 %     1.49 %          52.30 %
   11 Point    67.13 %    59.38 %     2.87 %          51.59 %
   101 Point   69.25 %    60.11 %     1.63 %          52.21 %
  ...
  Box Mean Average Precision
   Method      mAP@0.5   mAP@0.75   mAP@0.95   mAP@[0.5:0.95]
  ────────────────────────────────────────────────────────────
   Area        69.81 %    64.12 %     0.22 %          51.18 %
   11 Point    67.19 %    62.80 %     1.12 %          50.33 %
   101 Point   69.59 %    63.93 %     0.31 %          51.09 %
  ...
```
```text
Iteration 5000 · 20000 Test Images · EMA Weights · 2026-10-24 17:51 ────────────────────────────────────────────────────
  Confusion Matrix : score >= 0.50, IoU >= 0.5 (masks), 33597 right, 3131 wrong class, 2248 false positives, 32954
  missed
  Sample Images : 8 fixed test images

  Mask Mean Average Precision
   Method      mAP@0.5   mAP@0.75   mAP@0.95   mAP@[0.5:0.95]
  ────────────────────────────────────────────────────────────
   Area        73.64 %    64.81 %     4.13 %          57.43 %
   11 Point    70.68 %    63.59 %     6.10 %          56.20 %
   101 Point   73.41 %    64.62 %     4.32 %          57.30 %
  ...
  Box Mean Average Precision
   Method      mAP@0.5   mAP@0.75   mAP@0.95   mAP@[0.5:0.95]
  ────────────────────────────────────────────────────────────
   Area        73.93 %    70.08 %     2.90 %          59.31 %
   11 Point    70.74 %    68.53 %     5.17 %          57.90 %
   101 Point   73.69 %    69.86 %     3.10 %          59.17 %
  ...
```
The mask mAP@0.5 of the trained weights rises at every evaluation up to 69.49 % at iteration 4500, then falls to 63.72 % at 5000, the wrong classes jumping from 3,529 to 8,650: the polygons slipped by one side, 2,053 hexagons being found as pentagons and 1,803 heptagons as hexagons at 5000, against 450 and 402 at 4500 (`confusion_matrix.csv`), and the mask AP@0.5 of the hexagon falling from 70.08 % to 35.08 %. The averaged weights do not follow this swing: their mask mAP@0.5 rises at every evaluation, to 73.64 % at 5000, with a hexagon mask AP@0.5 of 86.38 %, and they give the best checkpoint of the run. With a constant learning rate of 0.0002, the trained weights of a single checkpoint can be caught in such a swing, which the averaged weights smooth out.

### Step 4.8 — `write_class_metrics_file()`
One row per class, the box columns of the detection followed by the same columns prefixed with `mask_`, then the threshold and recall at the target precision (of the mask average precisions): 37 columns, `average_precision_95` following `average_precision_75` and `mask_average_precision_95` following `mask_average_precision_75` (`351a4d1`), and the six recall columns `recall_50` to `recall_75_upper` (the recall at 0.5 and 0.75 with the bounds of its exact 95 % interval over the ground truths, `object_detection_workflow.md`, Step 4.11) following the box sizes, `mask_recall_50` to `mask_recall_75_upper` following the mask sizes.

**Example: `Evaluation/Iteration_5000/class_metrics.csv`** (first rows)
```text
class_index,class,ground_truths,average_precision_50,average_precision_75,average_precision_95,average_precision_50_95,
average_precision_50_small,average_precision_50_95_small,average_precision_50_medium,average_precision_50_95_medium,
average_precision_50_large,average_precision_50_95_large,mask_average_precision_50,mask_average_precision_75,
mask_average_precision_95,mask_average_precision_50_95,mask_average_precision_50_small,mask_average_precision_50_95_small,
mask_average_precision_50_medium,mask_average_precision_50_95_medium,mask_average_precision_50_large,
mask_average_precision_50_95_large,threshold_at_target_precision,recall_at_target_precision
1,square,3545,0.965184,0.882617,0.001079,0.699232,0.000000,0.000000,0.927349,0.623008,0.982403,0.731914,0.962450,0.904923,0.001740,0.737068,0.000000,0.000000,0.909091,0.657830,0.987273,0.774900,0.786123,0.852186
2,circle,3566,0.888822,0.819614,0.011202,0.645225,0.000000,0.000000,0.801425,0.612519,0.921571,0.661061,0.887047,0.865525,0.231875,0.784645,0.000000,0.000000,0.835590,0.733770,0.928919,0.825780,0.463157,0.505328
```

**Example: `Evaluation/Iteration_5000/proposal_recall.csv`**
```text
group,objects,recall_50,recall_70,proposals_per_image
all,69682,0.996341,0.974125,250.000000
small,184,0.358696,0.081522,
medium,33579,0.996665,0.967182,
large,35919,0.999304,0.985189,
block_5,69682,0.781091,0.522947,149.250250
block_6,69682,0.805043,0.216799,50.601800
block_7,69682,0.704400,0.500416,50.147950
```
As in the detection run, the proposals overlap 99.63 % of the test objects by 0.5 at iteration 5000, the small ones excepted (66 of 184): the errors come from the classification and the masks, not from the region proposals.

### Step 4.9 — `save_error_analysis()`
The class thresholds and the confusion matrix of the detection (see `object_detection_workflow.md`, Step 4.12), the thresholds being those of the mask average precisions and the confusion matrix counting the matches on the masks (Step 4.5); its image writes the shares, recalls and precisions as percentages with two decimals, in cells of 0.65 inch per class.

**Example output: iteration 5000** (`class_thresholds.json` is written indented, shortened here)
```text
class_thresholds.json:  {"target_precision": 0.99, "classes": {"square": {"threshold": 0.7861230373382568, "recall": 0.8521861777150916},
                         "circle": {"threshold": 0.4631573557853699, "recall": 0.5053280987100393}, ..., "hexagon": null, ...}}
                         15 classes with a threshold, null for the hexagon, octagon, trapezoid, heptagon and nonagon
confusion_matrix.csv:   true \ predicted,background,square,circle,triangle,rectangle,pentagon,hexagon,...
                        background,0,126,7,765,221,318,21,...       <- 3,549 false positives in the row
                        square,224,3269,1,0,51,0,0,...              <- 3,269 squares right, 224 missed
                        ...
                        hexagon,780,0,0,0,0,2053,625,...            <- 2,053 hexagons found as pentagons, 625 right
                        ...
                        heptagon,1552,0,0,0,0,77,1803,...           <- 1,803 heptagons found as hexagons
confusion_matrix.png:   2640 x 2602 pixels with the current cells, 1852 x 1815 when it was recorded
```
On the diagonal, no octagon, rhombus, trapezoid or nonagon is right at the deployment threshold, 12 kites and 23 heptagons are; the octagons go to the background column (2,834 of 3,454 missed) or are found as heptagons (507).

### Step 4.10 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
The report of Step 4.7 is printed and appended to `<run>/Results.txt`, the scalars, class scalars and images go to `Tensorboard/Test/` and to MLflow (see `object_detection_workflow.md`, Step 4.13). With the current code, `update_evaluation_curves()` then adds the row of the checkpoint to `Evaluation/evaluation_summary.csv`, draws `evaluation_curves.png` again next to `Results.txt` and sends both to the root of the MLflow run.

**Example: the test curves and the MLflow metrics after the 20 evaluations**
```text
Tensorboard/Test/: 126 scalars and 9 images per evaluation, among them
  mAP@0.5/area, mAP@0.5 By Size/medium, Mask mAP@0.5/area, Proposal Recall@0.5/block_5, Proposals Per Image/all,
  AP@0.5/<class>, Mask AP@0.5/<class>
  Mask mAP@0.5/area             500: 0.0608, 1000: 0.1673, 1500: 0.2788, 2000: 0.3977, 2500: 0.4774, 3000: 0.5732,
                                3500: 0.6099, 4000: 0.6781, 4500: 0.6949, 5000: 0.6372
Tensorboard/Test_EMA/: the same tags for the averaged weights
  Mask mAP@0.5/area             500: 0.0509, 1000: 0.1457, 1500: 0.2573, 2000: 0.3669, 2500: 0.4621, 3000: 0.5388,
                                3500: 0.6032, 4000: 0.6624, 4500: 0.7064, 5000: 0.7364
MLflow: 263 metrics: test/Mask mAP_at_0.5/area, test/best_Mask mAP_at_0.5/area (0.6949, tag best_iteration 4500),
        test_class/Mask AP_at_0.5/square, ..., the same under test_ema/ and test_ema_class/
        (test_ema/best_Mask mAP_at_0.5/area 0.7364, tag best_iteration_test_ema 5000)
```
MLflow refuses `@` in metric names, so it is written `_at_`; `best_` keeps the best value of the headline metric only (`Mask mAP@0.5/area`) over the evaluations of the run, with its iteration in the `best_iteration` tag: 4500 for the trained weights of this run, whose last evaluation is lower (Step 4.7). The `Tensorboard/Test` writer holds 126 scalars per evaluation for 20 classes: for the boxes 18 mAP (12 by method, 6 by size), 16 mAR and 20 `AP@0.5/<class>`, the same 54 for the masks under `Mask` names, and 18 proposal recall values, with the 8 sample images and the confusion matrix, and `Tensorboard/Test_EMA` the same tags for the moving average of the weights.

---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of the logic of `scripts/evaluation/evaluate_checkpoint.py` run on the 5000-iteration run of Stage 3 (its 20 checkpoints, one every 250 iterations) with `iteration = "all"`, `batch_size = 64`, `track_evaluation = False`, the evaluation written in a temporary folder and, to keep it short, a test folder of the first 200 images of the test split (its `labels.json` reduced to these images and their 709 annotations, its `data/` folder linked). It ran on the GPU, shared with other trainings, in 25 minutes. Step 5.4 comes from an earlier, tracked evaluation of the four checkpoints of the CPU run, the tracking being off here. They were captured before the per-class recall intervals were added (2026-11-01, Stage 4).

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The same script for every task (see `classification_workflow.md`, Stage 5): set `run_name` to the run folder, choose the iterations, and run it. The pretrained settings are turned off (`disable_pretrained_settings()`), since every weight comes from the evaluated checkpoint, and the default `maximum_error_images_per_class_pair = 100000` of its main block turns the error images on whatever the configuration (`None` keeps `evaluation.error_dumps`). `target_precisions` replaces the target precision of the class thresholds by a single value (a list being refused by the detection tasks), and `nms_criterion` and `nms_distance_exponent` the final NMS of the run, e.g. `"distance_iou"` with 0.6, whose evaluations then go to `<run>/Checkpoint_Evaluation_nms_distance_iou_0.6/` rather than `<run>/Checkpoint_Evaluation/` (`apply_nms_criterion()`); `None` keeps the setting of the run.

### Step 5.2 — trainer construction
The trainer is built from the `training_configuration.yaml` of the run, with the classes of the `label_dictionary.json` saved next to it, writing in the `Checkpoint_Evaluation/` folder of the run (`<run>/Checkpoint_Evaluation/`).

**Example: the lines of the launch information set by the main block**
```text
- Test Data Folder        : test_reduced_200 (200 images, 709 annotations)
- Batch Size              : 64
- MLflow Tracking         : Off
```

### Step 5.3 — `get_checkpoint_iterations()`
**Example output**
```text
evaluated_iterations = [5000, 4750, 4500, 4250, 4000, 3750, 3500, 3250, 3000, 2750, 2500, 2250, 2000, 1750, 1500, 1250, 1000, 750, 500, 250]
```

### Step 5.4 — `evaluate_checkpoints()` → `start_evaluation_tracking()`
With the tracking on, the evaluations go to an MLflow run of their own, `<run name>-evaluation`, in the backend of the evaluated run, tagged with the MLflow run of the training read from its `mlflow_run.json` (`evaluation_of`, Step 3.3), the evaluated folder, test split and iterations; the run of the training is only read, hence the new run identifier below.

**Example output** (the tracked evaluation of the CPU run)
```text
-----------------------------
Evaluating : model_000012.pt
-----------------------------
Test Evaluation Iteration 12: 100%|██████████| 5/5
MLflow Run Created In The Experiment 'instance_segmentation' : 09ee1c5e422849309be41a1c6e937c5b (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```

### Step 5.5 — `summarise_checkpoint_evaluations()`
**Example output**
```text
--------------------------------------------------------------
Evaluation Summary Of 20 Checkpoints : evaluation_summary.csv
--------------------------------------------------------------
  - Best Checkpoint : iteration 4500 (Mask mAP@0.5/area 70.29 %)
  - Best Checkpoint (EMA Weights) : iteration 5000 (Mask mAP@0.5/area 74.24 %)
```
The best value is written like in the report, as a percentage, since `percent_metrics` names the headline metric (Step 4.7); `evaluation_summary.csv` keeps every scalar between 0 and 1. With the current code, a `  - Evaluation Curves : evaluation_curves.png` line follows the title, the figure being drawn in `Checkpoint_Evaluation/`, and the summary ends with `ema_number_updates`, the number of updates of the averaged weights (254 columns for this run).

**Example: `<run>/Checkpoint_Evaluation/evaluation_summary.csv`** (253 columns: the iteration, the 126 scalars of the trained weights, that is the 86 box, mask and proposal recall scalars of Step 4.10 and the box and mask `AP@0.5` of the 20 classes, then the same 126 prefixed with `ema_`; a few columns, values rounded to 4 decimals)
```text
iteration   Mask mAP@0.5/area   mAP@0.5/area   Mask AP@0.5/hexagon   ema_Mask mAP@0.5/area   ema_Mask AP@0.5/hexagon
   5000       0.6287              0.6287         0.2877                0.7424                  0.8446
   4750       0.6958              0.6994         0.7182                0.7313                  0.8075
   4500       0.7029              0.7085         0.7340                0.7159                  0.7572
   4250       0.6835              0.6854         0.7063                0.6950                  0.6966
   ...
    500       0.0751              0.0852         0.1053                0.0628                  0.0770
    250       0.0271              0.0290         0.0321                0.0231                  0.0278
```
The checkpoints are ranked on the mask mAP. The 200 images give the same picture as the whole test split (Step 4.7): the trained weights are best at iteration 4500 (0.7029 here, 0.6949 on the whole split) and fall at 5000 (0.6287, 0.6372 on the whole split), the mask AP of hexagon falling from 0.7340 to 0.2877, with 17 of the hexagons found as pentagons and 21 of the heptagons as hexagons at the score 0.5. The averaged weights rise at every checkpoint to 0.7424 at iteration 5000 (0.7364 on the whole split), with a hexagon mask AP of 0.8446, and have the higher mask mAP@0.5 from iteration 4250 on, and at 3500 and 3750. Before iteration 3500 the trained weights lead, the average still holding the weights of the earlier iterations. To deploy the trained weights of this run, the checkpoint of iteration 4500 is the one to take; the averaged weights of iteration 5000 are better than both. The box mAP@0.5 differs from the mask mAP@0.5 by at most 0.0101 (iteration 500).

**Example: the evaluation folder**
```text
<run>/Checkpoint_Evaluation/
├── evaluation_summary.csv
├── Iteration_250/, Iteration_500/, ..., Iteration_5000/    the files of Steps 4.6 to 4.9, with their EMA/ folder
├── Tensorboard/Test/, Tensorboard/Test_EMA/
├── Results.txt                                             5000 lines, the 40 reports
└── anchor_recall.csv, data.json, label_dictionary.json, training_configuration.yaml
```

### Step 5.6 — `run_test_evaluation()` for each checkpoint
**Example output: the start of the report of the checkpoint of iteration 5000 on the 200 images**
```text
-----------------------------
Evaluating : model_005000.pt
-----------------------------

Iteration 5000 · 200 Test Images · 2026-10-25 07:23 ─────────────────────────────────────────────────────────────────
  Confusion Matrix : score >= 0.50, IoU >= 0.5 (masks), 310 right, 96 wrong class, 45 false positives, 303 missed
  Error Images : 421 in 53 folders, at most 100000 per folder
  Sample Images : 8 fixed test images

  Mask Mean Average Precision
   Method      mAP@0.5   mAP@0.75   mAP@0.95   mAP@[0.5:0.95]
  ────────────────────────────────────────────────────────────
   Area        62.87 %    52.94 %     1.29 %          45.78 %
   11 Point    61.46 %    51.89 %     1.72 %          45.04 %
   101 Point   62.71 %    52.83 %     1.35 %          45.71 %
  The mask mAP@0.5 of the area method compares the checkpoints of the run.
  ...
  Box Mean Average Precision
   Method      mAP@0.5   mAP@0.75   mAP@0.95   mAP@[0.5:0.95]
  ────────────────────────────────────────────────────────────
   Area        62.87 %    57.95 %     0.63 %          46.37 %
   11 Point    61.37 %    56.65 %     1.36 %          45.68 %
   101 Point   62.67 %    57.79 %     0.68 %          46.27 %
```
The averaged weights of the same checkpoint (the `EMA Weights` block that follows) find 321 objects right with 31 of the wrong class, against 310 and 96 for the trained weights.
