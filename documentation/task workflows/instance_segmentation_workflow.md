# Instance Segmentation Workflow

This document follows one instance segmentation run (INS) from `main()` to the evaluation of a checkpoint, one step per function, each with a **real** output. The **Call Tree Overview** gives the whole run on one page. Steps shared by every task are detailed in `classification_workflow.md`, and detector steps in `object_detection_workflow.md`; here they stay short. The mask steps are explained in full.

The outputs come from three sources, listed in the table below.

| Outputs | Source |
|---|---|
| Stages 1 and 2 | `configuration_synthetic_shapes.yaml` as it was then (512 pixels, batch size 40, heads on blocks 5, 6 and 7), run on CPU with a small generated dataset (`create_synthetic_dataset(task="instance_segmentation")`: 100 / 20 / 20 images, at most 6 shapes, 20 classes, uncompressed RLE), run folder in a temporary folder |
| Stages 3 to 5, the outputs of the run (Step 5.4 excepted) | the 5000-iteration run of [`run_configurations/instance_segmentation_workflow_run.yaml`](run_configurations/instance_segmentation_workflow_run.yaml), trained on the GPU on 2026-10-24 on the full synthetic dataset of that day (128,000 / 20,000 / 20,000 images, 20 classes) |
| Stages 3 and 4, the steps of one batch (Steps 3.6 to 3.13, 3.15, 3.16, 4.1 to 4.5) | the configuration of Stages 1 and 2 cut to batch size 2, 12 iterations, a checkpoint every 6, a summary every 4, a validation batch every 10, MLflow on, on CPU; this model finds no shape, so its values only show shapes and order |

The run configuration is the synthetic configuration with 5000 iterations instead of 2000, an evaluation every 500 instead of 250, MLflow on and the folder `experiments/workflow_runs`. It keeps the augmentation probabilities of that time (0.3, now 0.25; `random_haze` the only type on) and loaded one mask per object (`Data.instance_index_maps` did not exist). The dataset was generated again on 2026-11-18 with 35 classes, textured backgrounds, low contrast, smaller, cut and occluded shapes.

> The CPU examples predate the changes below; the 5000-iteration run shows the first four, the Neck and instance index map changes leave the outputs unchanged, and the last one changes the size tables.
> - Region proposal NMS 0.7 and detector NMS 0.5 (were 0.5 and 0.8); `RegionProposal.background_iou_threshold.min: 0.0`.
> - Mask and box `Mean Average Recall` tables, `evaluation.original_resolution`, and IoU 0.95 columns (`351a4d1`); column and scalar counts below are those of that time.
> - Mask overlaps paste 64 masks at a time at 512 pixels (`MASK_OVERLAP_REFERENCE_PIXELS`), where Step 4.5 pastes 16.
> - A moving average of the weights (`EMA/` folders, `EMA Weights` blocks) and `Optimization.loss_weights` (1 each).
> - A `Neck` section (`type: "none"`, no layer).
> - One instance index map per image (`Data.instance_index_maps: True`, Step 2.9), expanded into the same masks (Step 3.7).
> - Five object sizes relative to the image side instead of the COCO small, medium and large (`object_detection_workflow.md`, Step 4.14), and the anchor recall of the training and test splits at the first launch only, with `anchor_recall.png` (Step 2.18). The size tables and columns of Stage 4 keep the COCO sizes of that time.
>
> The synthetic configuration uses blocks of 16 and 32 filters, two anchor scales per level, mask heads `[3, 32]`, batch 64 and 2000 iterations; its detector settings are in the Configuration Keys table. The finished INS benchmark (`documentation/architecture_benchmarking/instance_segmentation_experiment_plan.md`) kept settings that are not shipped.

## Read This First
The instance segmenter finds every object with its box, class and score, like DET, plus the pixel mask of every detection. It is the two-stage detector of `object_detection_workflow.md` with one mask head per level of `DetectionLevels` (or one shared head, `MaskHead.shared`). It follows Mask R-CNN, without feature pyramid by default (`Neck.type: "none"`). Each mask head pools a refined box and predicts one 28 x 28 mask per class; the detected class channel is kept. The input is a COCO split with a `segmentation` per object (RLE or polygons). The output is the detections of DET, each with mask probabilities in its box frame.

Choose INS when the pixels of every object matter (touching or overlapping objects, areas, outlines). Choose DET when boxes are enough. Choose SEG (`semantic_segmentation_workflow.md`) when the class of every pixel is enough and objects of a class need not be told apart. Choose KPT for keypoints.

### The Task At A Glance
In order to see the whole data path at once, the diagram below gives the shapes of the synthetic configuration (512 pixels, batch 64, 35 classes and background, R the positive proposals of the batch). They were checked on CPU with the last checkpoint of a 20-class run.

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
```

### Files To Know
To find your way in the code, the table below maps each role to its file and main functions.

| Role | File | Main functions or classes |
|---|---|---|
| Trainer | `vision_training/trainers/instance_segmentation_trainer.py` (bases `detection_trainer.py`, `coco_split_trainer.py`, `base_trainer.py`) | `InstanceSegmentationTrainer`: `prepare_batch`, `augment_batch`, `forward_pass`, `gather_image_detections`, `compute_mask_overlaps`, `compute_evaluation_results`, `get_sample_masks` |
| Dataset and masks | `vision_training/data/detection/coco_dataset.py`, `coco_masks.py` | `CocoDetectionDataset` (`load_instance_masks`, `load_instance_index_maps`, `load_original_instances`, `add_instance_masks`, `add_instance_index_map`), `build_instance_mask`, `build_instance_index_map`, `find_index_map_objects`, `expand_instance_index_maps`, `decode_run_length_encoding`, `rasterize_polygons`, `compute_mask_bounding_boxes` |
| Model | `vision_training/models/instance_segmentation/instance_segmenter.py` (base `models/detection/two_stage_detector.py`) | `InstanceSegmenter`: `forward`, `compute_proposal_mask_loss`, `predict_detection_masks`, `print_mask_summary` |
| Mask head | `vision_training/models/instance_segmentation/mask_head.py` | `MaskHead`, `count_mask_upsamplings` |
| Mask targets, pasting and IoU | `vision_training/models/instance_segmentation/mask_operations.py` | `build_mask_targets`, `paste_instance_masks`, `compute_mask_intersection_over_union` |
| Mask loss | `vision_training/losses/instance_segmentation_losses.py` | `compute_mask_loss` (the four detection losses come from `detection_losses.py`) |
| Detector | see `object_detection_workflow.md` | anchors, proposals, detection heads, post-processing |
| Metrics | `vision_training/evaluation/detection/average_precision_report.py`, `mean_average_precision.py` | `summarise_average_precisions` (`overlap_type="masks"` then `"boxes"`), `evaluate_detections`, `gather_class_detections`, `compute_clopper_pearson_intervals` and `build_recall_interval_table` of `vision_training/evaluation/recall_intervals.py` |
| Configuration checks and run name | `vision_training/configuration_processing/instance_segmentation_checks.py`, `run_name.py` | `check_instance_segmentation_configuration`, `check_mask_head_configuration`, `build_instance_segmentation_sections` (`MRES-MBN-MUTC-M14.3x32-MR28`) |
| Export | `vision_training/models/instance_segmentation/inference_instance_segmenter.py`, `vision_training/deployment/export_specifications.py` | `InferenceInstanceSegmenter.predict_masks`, `get_export_specification` (the outputs of DET then `detection_masks`), `compare_instance_segmentations`; `vision_training/deployment/input_contract.py` |
| Export sizes at launch | `vision_training/deployment/export_size_estimates.py` | `estimate_export_sizes`, `probe_inference_graph`, `print_export_size_summary` |

### Run It
Suppose you want to train and evaluate an instance segmenter from nothing: these three commands do it.

The dataset script writes 128,000 / 20,000 / 20,000 images of 512 pixels with 5 to 6 shapes, each segmented by a compressed RLE (its `instance_segmentation` entries of `TASK_IMAGE_SIZES`, `TASK_SPLIT_SIZES`, `TASK_MAXIMUM_OBJECTS`). Shapes come from the 35 `SHAPE_CATEGORIES` (`MC35`; [tasks reference](../reference/tasks_reference.md#synthetic-datasets)); the runs below used 20 (`MC20`). Should you need the dataset of the finished INS benchmark, copy the `object_detection` entries of `TASK_IMAGE_SIZES` and `TASK_MAXIMUM_OBJECTS` (1024 pixels, 5 to 12 shapes) into the `instance_segmentation` ones, then add `--split_sizes train=512000 validation=50000 test=10000` (`documentation/architecture_benchmarking/instance_segmentation_experiment_plan.md`); it is on this machine (35 classes, test cut to 10,000 images, 84,922 objects).

Before training, point the three data folders, `Data.label_mapping.mapping_file_path` and `experiment_folder` of the configuration to absolute paths. To evaluate again, only `--experiment_folder` of `scripts/evaluation/evaluate_checkpoint.py` is required; `--iteration` (the last checkpoint by default, a number, or `all`) and the other arguments are in Step 5.1, those of the run kept when not given. It writes in `<run>/Checkpoint_Evaluation/`.

```bash
# Generate the synthetic instance segmentation dataset
uv run python -m vision_training.data.synthetic.create_synthetic_dataset --task=instance_segmentation --output_folder="$HOME/Data/instance_segmentation_data/synthetic_shapes"

# Train with the synthetic configuration
uv run python -m vision_training.main --configuration_path="vision_training/configurations/instance_segmentation/configuration_synthetic_shapes.yaml"

# Evaluate the last checkpoint again (run it without arguments for its help)
uv run python -m scripts.evaluation.evaluate_checkpoint --experiment_folder <run folder>
```

### Configuration Keys To Know First
Before changing the synthetic configuration, look at the keys below: they change the mask heads, the detector or the evaluation the most.

| Key | Value in the synthetic configuration | Effect |
|---|---|---|
| `MaskHead.mask_resolution` | 28 | side of every mask (`MR28`, the `[D, 28, 28]` of the exported `detection_masks`) |
| `DetectionLevels.<5, 6, 7>.mask_head` | `roi_align_pool_size: 14`, `convolutions: [3, 32]` | RoIAlign and convolutions of each level's head (`M14.3x32`), doubled once to 28 (`count_mask_upsamplings()`); the pool must not exceed the resolution. On the benchmark, pool 14 with resolution 28 beat pool 22 (resolution 44) and resolution 56; heads `[4, 64]` beat `[3, 32]` by 2.62 points of mask mAP@[0.5:0.95] (EMA), not shipped |
| `MaskHead.upsampling` | `"transposed"` | a 2 x 2 transposed convolution per doubling (`MUTC`), or bilinear then 3 x 3 convolution (`MUBL`) |
| `MaskHead.skip_blocks_at_pool_size`, `image_branch_convolutions`, `skip_blocks_at_mask_resolution`, `fusion_convolutions` | `[]`, `[0, 0]`, `[]`, `[0, 0]` (off) | finer inputs (`MSP`, `MIB`, `MSM`, `MF`, Step 2.13); the benchmark kept image branch `[5, 32]` with fusion `[1, 64]` (design A, +7.19 points), not shipped |
| `MaskHead.shared` | `False` | `True`: one head for all levels, run once in the export (`MSH`); needs the same `mask_head` and level channels (a neck) |
| `MaskHead.modules`, `normalization` | residual only; batch | the mask head blocks (`MRES-MBN`) |
| `MaskHead.inference.mask_threshold` | 0.5 | pasted mask pixels kept for mask IoU, confusion matrix and images; not applied by the export |
| `Optimization.loss_weights.mask_loss` | 1.0 (1.0 for the four others) | weight in the total loss; losses are tracked unweighted |
| `ExperimentConfiguration.evaluation.original_resolution` | False | True scores boxes and masks on the original test images |
| `Neck`, `RegionProposal`, `Detector`, `DetectionLevels.<level>.anchors` ... | `"none"`, `"joint"`, background min 0.0, `"iou"`, learning rate 2e-4 (DET: `"feature_pyramid"`, `"per_level"`, 0.1, `"distance_iou"` 0.6, 1e-3) | the detector (`object_detection_workflow.md`) |
| `Detector.inference.score_threshold`, `maximum_detections` | 0.5, 100 | the deployed detections, each with its mask |
| `ExperimentConfiguration.evaluation.score_threshold`, `target_precision` | 0.05, 0.99 | detections kept for the mAP; precision of the class thresholds, on the mask APs |

### What A Run Writes
Whenever a run is launched or a checkpoint evaluated, the table below lists what reaches the run folder, TensorBoard and MLflow.

| When | What |
|---|---|
| first launch, again when the split folders or the threshold change | `anchor_recall.csv` and `anchor_recall.png`: share of training and test objects with a foreground anchor, per level and size, boxes from the segmentations (Step 2.18) |
| before the first iteration | `Input_Samples/` with `samples.json`: first 10 images of every loader with their masks, training ones augmented (`training_reference.md`, Run Folder) |
| every evaluation | `Evaluation/evaluation_summary.csv` (one row per checkpoint, `ema_` columns) and `evaluation_curves.png` |
| every `weight_saving_iterations` checkpoint | `Evaluation/Iteration_<n>/`: `class_metrics.csv` (box and `mask_` columns, Step 4.8), `class_thresholds.json`, `confusion_matrix.csv` and `.png` (matched by mask IoU 0.5), `proposal_recall.csv`, `Samples/`, `Errors/` with `error_dumps`, the same in `EMA/` |
| `Results.txt` | one report per evaluation (two with EMA), eleven tables (Step 4.7) |
| TensorBoard | `Train`: total loss, five losses, learning rate; `Test` and `Test_EMA`: 126 evaluation scalars for 20 classes, 8 samples, confusion matrix |

The two runs this document quotes give the results below at their last checkpoint.

| Run | `Mask mAP@0.5/area` | Other results |
|---|---|---|
| INS run of `~/Data/tutorial_runs/readme_screenshots` (outside the repository), iteration 6000 | 82.96 % (88.09 % EMA) | `Mask mAP@[0.5:0.95]/area` 60.97 % (68.52 %), box `mAP@0.5/area` 83.54 %, mask `mAR@0.5` 96.81 %; 20,452 right, 2,886 wrong class, 3,029 false positives, 4,110 missed at the deployment threshold, matched on masks |
| `run_configurations/instance_segmentation_workflow_run.yaml`, iteration 5000 (Stages 3 to 5) | 63.72 % (73.64 % EMA) | best trained value 69.49 % at 4500 (Step 4.7) |

The tutorial run is the synthetic configuration plus the benchmark detection settings (feature pyramid, per-level selection, DIoU-NMS 0.6, 1e-3, background 0.1), mask heads `[4, 64]`, a level on block 4 and anchor scale 384 on block 7, without image branch. It was trained from random weights on the 2026-11-18 dataset, evaluated every 1000 iterations on 5,000 test images and stopped at 6000.

### Pitfalls
- **2000 iterations is too short.** In the tutorial run, `Mask mAP@0.5/area` is 15.01 % at 1000 (147 of 27,448 test objects found at the deployment threshold) and 40.96 % at 2000; samples show masks at 6000 (`suggestions.md`, item 30).
- **Thin parts are lost at strict IoU.** Masks are 28 x 28 in their box, pooled from strides 32 and 64 (16 to 64 in the tutorial run). There the star has mask AP@0.5 96.68 % but AP@0.75 61.89 %, against box AP@0.75 90.30 %; mask `mAP@0.95` is 1.69 %. The image branch of design A gained 7.19 points of mask mAP@[0.5:0.95] (stage 3b).
- **Mask and box size tables hold different objects.** Sizes are mask areas in mask tables and box areas in box tables: 8,080 test objects under 32 pixels by mask area against 5,778 by box area, 9,227 from 96 pixels against 12,034 (tutorial run, iteration 6000, COCO sizes).
- **The export returns box-frame probabilities.** `detection_masks` are 28 x 28 inside each box. The client pastes them with `paste_instance_masks()` and applies `MaskHead.inference.mask_threshold` itself. Restore boxes unclipped (`restore_boxes_to_original_image(..., clip=False)`), since a clipped box squeezes its mask (Export section of `documentation/reference/tools_reference.md`). Letterbox by the input contract (`preprocessing.json`): BGR, `keep_ratio` to 512, bilinear, padding 0 centred, NCHW, values 0 to 255.
- **Each level trains its own head.** A head learns only from its level's positives; the export runs every head on all detections. `MaskHead.shared: True` gives one head (same `mask_head` and level channels, `suggestions.md`, item 7). In the tutorial run, 238.4 of 250 proposals per image come from block 4 (`proposal_recall.csv`).
- **Vanished masks drop their object.** An object whose mask vanishes at the network resolution leaves the targets and the evaluation, but stays with `evaluation.original_resolution: True`.
- **Instance index maps need non-overlapping RLE.** A polygon, a shared pixel or more than 255 objects stop the loading with a message naming the setting (COCO keeps it off).

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
To see the whole run on one page, the tree below lists every function call in order, with the stage it belongs to.

```text
main()                                                     give an instance_segmentation --configuration_path
├── load_experiment_configuration()                        [STAGE 1]
│   ├── load_configuration() / build_label_dictionary()    adds 'background': 0
│   └── check_configuration() -> check_instance_segmentation_configuration()
│       ├── check_detection_configuration()                mask_head per level, no random_perspective/jigsaw
│       ├── check_mask_head_configuration()                pool <= resolution, check_mask_head_fine_inputs
│       └── check_boolean_flags()                          instance_index_maps, original_resolution
├── get_trainer_class(task)  -> InstanceSegmentationTrainer
├── InstanceSegmentationTrainer(...) -> DetectionTrainer -> BaseTrainer        [STAGE 2]
│   ├── get_run_name() -> build_instance_segmentation_sections()
│   ├── setup_training_paths() / build_dataloaders()
│   │   └── get_detection_dataloaders(load_instance_masks=True, load_instance_index_maps=...,
│   │                                 load_original_instances=...) -> CocoDetectionDataset + build_native_loader
│   ├── build_model() -> InstanceSegmenter: TwoStageDetector + MaskFineInputs + MaskHead per level
│   ├── initialise_pretrained_components() / build_optimizer() / get_tracked_metrics()   5 losses + Total Loss
│   ├── print_summary() / estimate_export_sizes() / diagnosis modes (exit)
│   ├── analyse_training_data()                            anchor recall on the boxes of the segmentations, first launch
│   └── restore_last_model()
└── run_training_loop() -> run_training_iterations()       [STAGE 3]
    ├── get_next_batch()                                   add_instance_index_map() or add_instance_masks()
    ├── run_model_iteration()
    │   ├── prepare_batch()                                expand_instance_index_maps() on the device
    │   ├── augment_batch()                                masks moved with the image, boxes recomputed
    │   └── forward_pass() -> InstanceSegmenter.forward()
    │       ├── forward_detection_stages()                 the 4 detection losses
    │       └── compute_proposal_mask_loss()               build_mask_targets() + compute_mask_loss()
    ├── backward() / step_optimizer() / save_model() / summaries
    └── run_test_evaluation()                              [STAGE 4]
        ├── predict_batch() -> postprocess_detections() + predict_detection_masks()
        ├── gather_image_detections() -> compute_mask_overlaps()     paste, threshold, mask IoU
        ├── match on masks -> confusion matrix
        ├── compute_evaluation_results()                   summarise_average_precisions("masks", then "boxes")
        └── report_evaluation() -> Results.txt, TensorBoard, MLflow, evaluation_summary.csv

scripts/evaluation/evaluate_checkpoint.py                  [STAGE 5]
└── evaluate_checkpoints() -> run_test_evaluation() for every checkpoint -> summarise_checkpoint_evaluations()
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below come from the loader run on `configuration_synthetic_shapes.yaml` as it was then, its data folders and mapping file redirected to the generated dataset.

### Step 1.1 — `main()`
The entry point is the same for every task (see `classification_workflow.md`, Step 1.1).

### Step 1.2 — `load_experiment_configuration()`
The model sections are those of DET (`object_detection_workflow.md`, Step 1.2) plus `MaskHead`, shared by the mask heads of every level. Its keys are listed in the table below.

| `MaskHead` key | Role |
|---|---|
| `mask_resolution` | side of the predicted masks (28) |
| `upsampling` | `transposed` or `bilinear`, how pooled features are doubled up to the resolution |
| `skip_blocks_at_pool_size`, `image_branch_convolutions`, `skip_blocks_at_mask_resolution`, `fusion_convolutions` | the finer inputs (off) |
| `shared` | one head per level (`False`) or one for all |
| `normalization`, `modules`, `pretrained` | as in the other heads |
| `inference.mask_threshold` | probability from which a pixel belongs to its object |

Every level of `DetectionLevels` also holds a `mask_head` with its own `roi_align_pool_size` and `convolutions` (`[number of layers, number of filters]`).

**Example: the sections and the mask settings**
```text
model keys:      Optimization, Data, Backbone, DetectionLevels, RegionProposal, Detector, MaskHead
DetectionLevels 5: {"anchors": {...}, "region_proposal": {...}, "detector": {...},
                    "mask_head": {"roi_align_pool_size": 14, "convolutions": [4, 64]}}
MaskHead: {"mask_resolution": 28, "upsampling": "transposed",
           "normalization": {"feature_map_normalization": "batch"},
           "modules": {"residual_nets": true, "inception": false, "squeeze_and_excite": false},
           "inference": {"mask_threshold": 0.5},
           "pretrained": {"experiment_folder": null, "iteration": null, "freeze": false, "learning_rate_multiplier": 1.0}}
```
The current file also has a `Neck` section (`type: "none"`), `original_resolution: False` in `evaluation` (Step 4.7), `mask_loss: 1.0` in the loss weights, the finer inputs off and `shared: False` in `MaskHead`, `class_layers: False` in the `pretrained` blocks of `Detector` and `MaskHead`, a `pretrained` block in `Neck`, and `instance_index_maps: True` in `Data` (Step 2.9). The loader adds the runtime key `standalone_evaluation_folder` (`None` in training, set by Stage 5).

### Step 1.3 — `load_configuration()`
It reads the YAML with the top-level keys of Step 1.2; values such as `number_iterations: '1e10'` are still strings here.

### Step 1.4 — `build_label_dictionary()`
As for DET, `'background': 0` is added to the mapping file.

**Example: the label dictionary of the generated dataset**
```text
dict: {1: 1, 2: 2, 3: 3, ..., 19: 19, 20: 20, 'background': 0}
21 entries, 21 distinct classes
```

### Step 1.5 — `check_configuration()`
The common checks run first, then `check_instance_segmentation_configuration()`. Its parts are listed below.
- `check_detection_configuration()`: also requires a `mask_head` per level, refuses `random_perspective` and `random_jigsaw` (the masks cannot follow them), and checks the `mask_loss` weight.
- `check_mask_head_configuration()`: the RoIAlign pool of every level must not exceed `MaskHead.mask_resolution`, since the head only enlarges.
- `check_mask_head_fine_inputs()`: image branch and fusion `[layers, channels]` both positive or `[0, 0]`; skip blocks distinct, from 0 (the normalized image) to the last block; no image branch with mask heads from another run (the branch is not transferred).
- `check_skip_block_strides()`: the U-Net layout (block b before its pooling at stride 2 ** (b - 1)). Every skip is finer than the finest level (block 5 at most here), mask resolution skips finer than pool size skips, no block in both lists.
- The booleans `Data.instance_index_maps` and `evaluation.original_resolution`.

**Example: output with `MaskHead.upsampling: nearest` and `DetectionLevels.5.mask_head.roi_align_pool_size: 32`**
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

> The examples below come from building the trainer on the redirected `configuration_synthetic_shapes.yaml` (512 pixels, batch size 40), run folder in a temporary folder. The construction took 16.4 s on CPU.

### Step 2.1 — `InstanceSegmentationTrainer.__init__()` → `DetectionTrainer.__init__()`
The trainer is the detection trainer with the additions listed in the table below.

| Attribute | Role |
|---|---|
| `LOAD_INSTANCE_MASKS = True` | the COCO loaders return the mask of every object (Step 2.9) |
| `MATCHING_OVERLAP_TYPE = "masks"` | detections matched to objects by mask IoU, for the confusion matrix and error images (Step 4.5) |
| `self.mask_threshold` | from `MaskHead.inference.mask_threshold` |
| `self.original_resolution` | from `evaluation.original_resolution`, read before the loaders: the test loader also gives the original objects (Step 4.7) |
| `self.instance_index_maps` | from `Data.instance_index_maps` (`True` in the synthetic configuration): one map per image, expanded on the device (Step 2.9) |
| `MASK_OVERLAP_CHUNK_SIZE = 16` | detections pasted at once for a 1024 x 1024 image, scaled by `MASK_OVERLAP_REFERENCE_PIXELS`: 64 at 512 (Step 4.5) |

The detection constructor then reads the thresholds and error settings (`object_detection_workflow.md`, Step 2.1).

**Example: the attributes after the construction** (before `self.original_resolution`, now `False` in the synthetic configuration)
```text
self.mask_threshold             = 0.5
self.evaluation_score_threshold = 0.05
self.deployment_score_threshold = 0.5
self.target_precision           = 0.99
TASK_NAME = 'instance_segmentation'   SAMPLE_NAME = 'Images'   COUNTS_FRAMES = False
LOAD_INSTANCE_MASKS = True   MATCHING_OVERLAP_TYPE = 'masks'
```

### Step 2.2 — `BaseTrainer.__init__()`
**Example: some attributes set by the constructor**
```text
self.device = cpu   self.number_sample_images = 8   self.input_image_size = 512   self.batch_size = 40   self.start_iteration = 1
```

### Step 2.3 — `get_device()`
It prints `For Training We Will Be Using device: cpu` here (forced on the CPU; a normal run prints `cuda`).

### Step 2.4 — `get_run_name()` → `build_run_name()` → `build_instance_segmentation_sections()`
The prefix is `INS`. The sections are those of DET (`object_detection_workflow.md`, Step 2.4) plus a mask heads section before the classes. Its tokens are listed in the table below.

| Token | Meaning |
|---|---|
| `MRES` (`MSE`, `MINC`) | modules of the mask heads |
| `MBN` (`MLN`, `MNN`) | normalization |
| `MUTC` (`MUBL`) | transposed (bilinear) upsampling |
| `M14.4x64` | RoIAlign 14, 4 layers of 64 filters; once when every level matches, otherwise per level joined with `+` |
| `MR28` | mask resolution |
| `MSP<blocks>`, `MIB<layers>x<channels>`, `MSM<blocks>`, `MF<layers>x<channels>` | finer inputs, only when used (blocks joined with `.`), so older runs keep their names |

The mask threshold gives no token, so it can change between two resumptions.

**Example output**
```text
INS_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512.598295_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128_MRES-MBN-MUTC-M14.4x64-MR28_MC20
```

### Step 2.5 — `DataAugmenter()`
The augmenter of DET; geometric augmentations move the masks with the image (Step 3.8). This configuration turns every type off except `random_haze`, which never changes the masks.

### Step 2.6 — `setup_training_paths()`
As for every task (`classification_workflow.md`, Step 2.6): resume checks, then the run folder with the configuration and `label_dictionary.json`. After the loaders, `save_data_record()` writes `data.json`.

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
The hook of DET (`object_detection_workflow.md`, Step 2.7). `read_test_manifest()` gives `class_names`, `class_colours` and the test image names, then the loaders are built with the masks. The validation loader exists only when the run reads it (`classification_workflow.md`, Step 2.7).

**Example output**
```text
class_names: {0: 'background', 1: 'square', 2: 'circle', 3: 'triangle', 4: 'rectangle', 5: 'pentagon', ..., 20: 'l_shape'}
test_image_file_names: 20 entries, [(1, 'test_000000.png'), (2, 'test_000001.png'), (3, 'test_000002.png'), ...]
```

### Step 2.8 — `get_detection_dataloaders(load_instance_masks=True, load_instance_index_maps=...)`
`load_instance_index_maps` is `Data.instance_index_maps`. The test loader of `evaluation.original_resolution` is built without maps, one mask per object at both resolutions.

**Example: the three loaders** (recorded with 8 workers; every loader now has 1, and batches are pinned only on a discrete GPU, `pin False` on the DGX Spark, `uses_pinned_memory()`)
```text
train_dataloader        100 images   3 batches   collate_function   num_workers 8   prefetch 2   pin True   persistent True
validation_dataloader    20 images   1 batch     ...
test_dataloader          20 images   1 batch     ...
```

### Step 2.9 — `CocoDetectionDataset(load_instance_masks=True)` → `select_image_objects(boxes_from_segmentation=True)`
Every annotation of the COCO `labels.json` holds a `segmentation`: polygons or an RLE (`size` and `counts`, alternating background and object run lengths, column by column; a list, or a string when compressed). The box comes from the segmentation, not `bbox` (`compute_segmentation_bounding_box()`), so boxes and masks agree. No mask is decoded here.

The manifest is kept as one packed record per image (`SerializedRecords`, `object_detection_workflow.md`, Step 2.9), so loader workers never copy its Python objects (the parsed 512,000-image benchmark manifest takes about 7 GB).

`__getitem__()` reads the image with `read_image()` (OpenCV, BGR, `uint8`, a `ValueError` naming an unreadable file). Then one of two paths builds the masks, as the table below shows.

| Setting | Path |
|---|---|
| `Data.instance_index_maps: False` | `add_instance_masks()`: `build_instance_mask()` per object at the network resolution. Polygons go through `rasterize_polygons()`; an RLE through `decode_run_length_encoding()` (`decode_compressed_counts()` for strings), then a `cv2.INTER_NEAREST_EXACT` resize and 0 padding. Empty masks are removed with their box and label; `compute_mask_bounding_boxes()` sets each box to its mask's tight box |
| `Data.instance_index_maps: True` (synthetic configuration) | `add_instance_index_map()`: `build_instance_index_map()` paints the RLE runs into one map (one cumulative sum), resized once. Overlaps, polygons and more than `MAXIMUM_INDEX_MAP_OBJECTS` (255) objects are refused. `find_index_map_objects()` gives the remaining objects and tight boxes; vanished ones are removed, the others renumbered from 1 |

Expanded, the map gives exactly the masks, labels and boxes of `add_instance_masks()`, in a third of the bytes here and about a ninth on the benchmark dataset. A 64-image benchmark batch reaches the GPU in 1.20 s instead of 2.22 s, its expansion taking 23 ms (2026-11-13).

**Example: the manifest of the training split**
```text
annotations[0]:     {'id': 1, 'image_id': 1, 'category_id': 6, 'bbox': [325, 369, 133, 123],
                     'segmentation': {'size': [512, 512], 'counts': [166822, 2, 508, 6, 505, 9, 502, 13, 497, 17, ...]}}   (358 annotations)
segmentation types: 358 RLE (dict), 0 polygons (list), 0 compressed RLE
```

**Example: the first sample of the training dataset** (`train_000073.png`, image id 74, one mask per object)
```text
dataset[0]:
  images              (3, 512, 512) torch.uint8
  bounding_boxes      (6, 4) torch.float32   [[188, 433, 222, 503], [153, 94, 298, 241], [408, 146, 503, 243], ...]
  labels              (6,) torch.int64       [4, 2, 8, 7, 15, 10]
  instance_masks      (6, 512, 512) torch.uint8, values {0, 1}
  mask pixels         [2380, 15705, 3566, 2507, 10929, 2157]
```
The images are already 512 pixels, so the masks are not resized.

**Example: the first two training images of `synthetic_shapes` with the maps** (recorded on 2026-11-13)
```text
collate_function:     instance_index_maps (2, 512, 512) uint8 (524,288 bytes), labels [9] and [19, 1, 14, 3, 9]
expanded masks:       (1, 512, 512) and (5, 512, 512) uint8 (1,572,864 bytes), mask pixels [18685] [2401, 16641, 4022, 866, 7001]
                      identical to the masks, boxes and labels of add_instance_masks()
```

### Step 2.10 — `build_native_loader()`
The loader of every task (`classification_workflow.md`, Step 2.10). `collate_function()` stacks the images and instance index maps. It keeps boxes, labels and instance masks as lists, one tensor per image (Step 3.6).

### Step 2.11 — `get_test_sample_classes()` + `choose_sample_identifiers()`
The same 8 test images every evaluation, spread over the test split (`object_detection_workflow.md`, Step 2.11).

Here the sample identifiers are `{1, 4, 6, 9, 12, 15, 17, 20}`.

### Step 2.12 — `build_model()` → `InstanceSegmenter.__init__()`
`InstanceSegmenter` is a `TwoStageDetector` (`object_detection_workflow.md`, Steps 2.12 to 2.16) with one more `ModuleDict`, `mask_heads_dictionary`: one `MaskHead` per level, built by `_build_mask_heads_dictionary()`. With `MaskHead.shared: True` it holds one head keyed `shared`. That head pools every box from its own level (`pool_box_features()`) and runs once on all boxes (`predict_from_pooled_features()`, through `predict_box_mask_logits()`).

**Example: the children of the model and their parameters**
```text
backbone                      Backbone               2,245,216
region_proposer_dictionary    ModuleDict               495,239   {'5': 113,965, '6': 190,637, '7': 190,637}
detector_heads_dictionary     ModuleDict             2,869,179   {'5': 931,817, '6': 968,681, '7': 968,681}
region_proposal_filter        RegionProposalFilter           0
mask_heads_dictionary         ModuleDict               571,839   {'5': 166,037, '6': 202,901, '7': 202,901}
anchors: (3456, 4)
```
The current code adds a `neck` (`FeatureNeck`, 0 parameters with `type: "none"`) and a `mask_fine_inputs` (`MaskFineInputs`, 0 with the finer inputs off). The shipped synthetic configuration gives `mask_heads_dictionary` 99,756 parameters.

### Step 2.13 — `MaskHead.__init__()` → `count_mask_upsamplings()`
A mask head is a small fully convolutional network run on every box, with the layers below.
- `torchvision.ops.roi_align()` pools the box into a `roi_align_pool_size` square (`spatial_scale` = feature map size / image size, `sampling_ratio=0`, `aligned=True`).
- `convolutional_block`: the `convolutions` 3x3 layers, with the normalization and modules of `MaskHead`.
- Doublings: `count_mask_upsamplings()` gives how many bring the pool to the resolution without exceeding it (14 -> 28: 1). Each is a `ConvTranspose2d` of kernel 2, stride 2 (`transposed`), or bilinear then 3x3 convolution (`bilinear`), then a ReLU.
- `mask_predictor`: a 1x1 convolution, one logit map per class with background; a box reads the channel of its class (Steps 3.13 and 4.3).
- Finer inputs (off here): `skip_blocks_at_pool_size` blocks (before their pooling) are pooled at the pool size and concatenated before `convolutional_block`. `MaskFineInputs`, shared by the levels (an image branch of 3 x 3 convolutions on the RoIAlign crop of the normalized image, then `skip_blocks_at_mask_resolution` blocks), is concatenated after the doublings. `fusion_block` mixes them before `mask_predictor`.
- When the doublings miss `mask_resolution` (a pool of 10 doubled to 20 for 28), the logits are resized bilinearly, or the doubled features before concatenation with fine features.

Every level gives the same mask side, so all levels concatenate for the loss.

**Example: the mask head of block 5** (printed again on CPU with the current code)
```text
MaskHead(
  (convolutional_block): ConvolutionBlock(... 4 x [Conv2d(64, 64, 3x3) + BatchNorm2d(64) + ReLU])
  (upsampling_layers): ModuleList((0): ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(2, 2)))
  (activation): ReLU()
  (mask_predictor): Conv2d(64, 21, kernel_size=(1, 1), stride=(1, 1))
)
mask head 5: pool 14, upsamplings 1, upsampled 28, mask 28, spatial scale 0.03125,  feature map 16
mask head 6: pool 14, upsamplings 1, upsampled 28, mask 28, spatial scale 0.015625, feature map 8
mask head 7: pool 14, upsamplings 1, upsampled 28, mask 28, spatial scale 0.015625, feature map 8
```

### Step 2.14 — `initialise_pretrained_components()` + `build_optimizer()` + `get_tracked_metrics()`
Four components can come from another run: `Backbone`, `RegionProposal`, `Detector` and `MaskHead`. Source mask heads must match in every `MaskHead` setting and per-level `mask_head` (`compare_detection_heads()`). With `MaskHead.pretrained.class_layers: False` (default, `MRCL` in the run name) the predictor keeps random weights; with `True` it takes each source class row by order or name (`compare_class_names()`, `object_detection_workflow.md`, Step 2.17). The image branch is not transferred. `get_tracked_metrics()` adds `mask_loss` to the five metrics of DET. The current code also transfers the `Neck`, so the dictionary below ends with `'Neck': None`; the heads need the same neck type and channels in both runs (`object_detection_workflow.md`, Step 2.17).

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
- Run Name                : INS_IN-KAR-512_..._MRES-MBN-MUTC-M14.4x64-MR28_MC20
- Train Data Folder       : train (100 images, 358 annotations)
- Validation Data Folder  : validation (20 images, 74 annotations)
- Test Data Folder        : test (20 images, 66 annotations)
- Iterations Per Epoch    : 3
...
- MLflow Tracking         : Off
```

### Step 2.16 — `print_summary()` / `print_receptive_field_summary()` / `print_model_size_summary()`
The panels of DET (`object_detection_workflow.md`, Step 2.19) are followed by a **Mask Architecture** panel, one block per level. The model size panel counts the mask heads apart. Finer inputs add their own lines to both panels. The Export Sizes panel follows (PyTorch, TorchScript, ONNX, TensorRT and checkpoint sizes).

**Example output: the mask panel and the model size** (rebuilt on CPU with the current code, same as then)
```text
╭───────────────────────────── Mask Architecture ──────────────────────────────╮
│ Normalization Strategy: Batch                                                │
│ Upsampling: Transposed                                                       │
│ Mask Block 5                                                                 │
│   - Modules            : Residual Shortcuts                                  │
│   - Convolution Layers : 4                                                   │
│   - RoI Align Pool     : [14, 14]                                            │
│   - Doublings          : 1, to [28, 28]                                      │
│   - Spatial Scale      : 0.0312                                              │
│ ... Mask Blocks 6 and 7: Feature Map Size [8, 8], Spatial Scale 0.0156       │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────── Model Size ───────────────────────╮
│ Backbone              2,245,216 parameters    9.00 MB    │
│ Region Proposal Heads   495,239 parameters    1.98 MB    │
│ Detection Heads       2,869,179 parameters   11.48 MB    │
│ Mask Heads              571,839 parameters    2.29 MB    │
│ Total Model           6,181,473 parameters   24.76 MB    │
╰──────────────────────────────────────────────────────────╯
```
The model size panel is condensed here: it prints parameters, trainable parameters and size on three lines per component.

### Step 2.17 — `view_input_batches()` → `get_input_sample_image()` → `overlay_instance_masks()`
In case you turn on `diagnosis_modes.view_input_images`, the input viewer of DET (`object_detection_workflow.md`, Step 2.20) draws the boxes over the masks, filled with their class colour by `overlay_instance_masks()`. `Input_Samples/` is drawn the same way. Off in the configuration.

### Step 2.18 — `analyse_training_data()` → anchor recall
The anchor recall of DET (`object_detection_workflow.md`, Step 2.21), on the boxes of the segmentations: `gather_object_boxes()` reads them with `select_image_objects(boxes_from_segmentation=True)`. It reads the manifest only, without decoding a mask.

**Example output: the first launch of the synthetic configuration on the dataset of 2026-11-27** (training table of `anchor_recall.png`)
```text
Anchor Recall Of The 703 879 Training Objects: 64.04 % In All Levels, 8.11 % For The Smallest Objects (Under 1/16)
Anchor Recall Of The 109 928 Test Objects: 64.24 % In All Levels, 8.52 % For The Smallest Objects (Under 1/16)

 Object Side         Pixels             Area    Count   Block 5   Block 6   Block 7   All Levels
 All Objects                                   703879   22.89 %   25.76 %   41.15 %      64.04 %
 Under 1/16            < 32         < 0.39 %   150120    8.11 %    0.00 %    0.00 %       8.11 %
 1/16 to 1/8       32 to 64   0.39 to 1.56 %   143890   51.12 %    9.57 %    0.00 %      51.12 %
 1/8 to 1/4       64 to 128   1.56 to 6.25 %   174260   43.28 %   50.38 %   38.84 %      82.12 %
 1/4 to 1/2      128 to 256     6.25 to 25 %   178644    0.00 %   44.64 %   94.14 %      94.14 %
 1/2 and above       >= 256          >= 25 %    56965    0.00 %    0.00 %   94.41 %      94.41 %
```
Both splits are saved in `anchor_recall.csv` (recalls between 0 and 1) and drawn in `anchor_recall.png`, then sent to MLflow. Here the boxes of the segmentations equal the annotation boxes, so the recalls are those of DET. Later launches keep both files.

### Step 2.19 — `restore_last_model()`
A new run folder prints `No Initiation Model Weights Will Be Used...` and trains from scratch.

---

## Stage 3 — Training Loop

> Steps 3.1 to 3.5, 3.14, 3.17 and 3.18 show the 5000-iteration run of [`run_configurations/instance_segmentation_workflow_run.yaml`](run_configurations/instance_segmentation_workflow_run.yaml) (2026-10-24), whose differences from `Run It` are listed at the top of this file. Its run name, written `<run>` below, is `INS_IN-KAR-512_SE-RES-BN-B1x16.2x16.2x16.3x32.3x32.5x32.5x32_L567_A6.32-256.e7b72c_RBN-R2x32_DSE-DRES-DBN-D7.2x32.64.32_MRES-MBN-MUTC-M14.3x32-MR28_MC20`. With batches of 64 an epoch is 2000 iterations; a summary prints every 1000 (`information_dump`); no validation batch runs (`validation_iteration_ratio: 0`). It took 2 h 32 min (15:19 to 17:51), the GPU shared with the DET run until 16:15.
>
> Steps 3.6 to 3.13, 3.15 and 3.16 show one batch of the CPU run (batch size 2, mask heads `[4, 64]`, 3456 anchors).

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
The loop of every task (`classification_workflow.md`, Step 3.1). Each iteration takes its training step, then saves, validates and summarises on the iterations set for each, and evaluates last. The table below shows what happens at each iteration of the run after its training step.

| Iteration | 250 | 500 | 750 | 1000 | ... | 4500 | 4750 | 5000 | after the loop |
|---|---|---|---|---|---|---|---|---|---|
| Checkpoint saved | ✔ | ✔ | ✔ | ✔ | ... | ✔ | ✔ | ✔ | nothing: iteration 5000 is saved |
| Validation batch | | | | | | | | | |
| Console summary | | | | ✔ | ... | | | ✔ | |
| Test evaluation, trained then averaged weights | | ✔ | | ✔ | ... | ✔ | | ✔ | nothing |

### Step 3.2 — `setup_tensorboard_writers()` / `initialize_training_loop_state()`
After the run, `Tensorboard/Train/` holds the six losses and the learning rate (iterations 1 to 5000), and `Test/` and `Test_EMA/` hold 126 scalars and 9 images per evaluation (500 to 5000, Step 4.10). There is no `Validation` writer.

### Step 3.3 — `start_experiment_tracking()`
**Example output**
```text
MLflow Run Created In The Experiment 'instance_segmentation' : 4c0f509b4cfd43ccac007e094fb5769e (Browse : mlflow ui --backend-store-uri sqlite:///.../experiments/workflow_runs/mlflow.db)
```
The identifiers go to `<run>/mlflow_run.json`, so a resumed run logs to the same MLflow run and a standalone evaluation tags its run with it (`evaluation_of`, Stage 5).

### Step 3.4 — `save_model()` → `save_checkpoint()`
Saved right after the training step of every multiple of 250, so checkpoint N holds N steps (`classification_workflow.md`, Step 3.6). It names the source run of the four transferable components (`pretrained_backbone`, `pretrained_region_proposer_dictionary`, `pretrained_detector_heads_dictionary`, `pretrained_mask_heads_dictionary`, None here) and saves `exponential_moving_average_state`. Current checkpoints also name the source run of the neck (`pretrained_neck`).

**Example: the checkpoints and the keys of the last one**
```text
Weights/Iteration_250/model_000250.pt ... Weights/Iteration_5000/model_005000.pt   12,387,623 bytes each

iteration 5000   format_version 2   task 'instance_segmentation'   run_name '<run>'   mlflow_run_id '4c0f509b4cfd43ccac007e094fb5769e'
model_state: 402 tensors, 75 of them in the mask heads
  mask_heads_dictionary.5.upsampling_layers.0.weight   (32, 32, 2, 2)
  mask_heads_dictionary.5.mask_predictor.weight        (21, 32, 1, 1)
exponential_moving_average_state: {'model_state', 'number_updates'}, number_updates 5000
```

### Step 3.5 — `run_test_evaluation()`
At every multiple of 500, last in the iteration (`classification_workflow.md`, Step 3.7), once for the trained weights and once for the averaged weights. See Stage 4.

### Step 3.6 — `get_next_batch()` → `CocoDetectionDataset.__getitem__()` + `collate_function()`
A batch that fails to load is dropped and the next one taken (`classification_workflow.md`, Step 3.8). Every sample goes through `add_instance_index_map()`, or `add_instance_masks()` without maps (Step 2.9). Images and maps are stacked; boxes, labels and masks stay lists.

**Example: the first batch of the CPU run** (one mask per object; with maps it holds `instance_index_maps (2, 512, 512) uint8` instead)
```text
images               (2, 3, 512, 512) torch.uint8
bounding_boxes       list of 2: (4, 4) float32, (3, 4) float32
labels               list of 2: (4,) int64, (3,) int64       [14, 16, 6, 16]   [20, 13, 17]
instance_masks       list of 2: (4, 512, 512) uint8, (3, 512, 512) uint8
mask pixels          [9155, 2352, 5597, 4341]   [7163, 6094, 2600]
```

### Step 3.7 — `run_model_iteration()` → `prepare_batch()`
The DET preparation (images to float32 on the device, boxes and labels moved), plus the uint8 masks. Maps are moved and expanded by `expand_instance_index_maps()`, one mask per object number in label order (`map == k`); loader masks are moved as they are. The batch holds `instance_masks` either way.

**Example: after the preparation**
```text
images (2, 3, 512, 512) torch.uint8 -> torch.float32
instance_masks list of 2: (4, 512, 512) uint8, (3, 512, 512) uint8      <- unchanged
```

### Step 3.8 — `augment_batch()` → `DataAugmenter.augment_batch(bounding_boxes=..., labels=..., instance_masks=...)`
Geometric augmentations move the masks with the image (`InterpolationMode.NEAREST_EXACT`). Each box becomes the tight box of its moved mask (`compute_mask_bounding_boxes()`), since a rotated box would grow. The visible share is measured on mask pixels, scaled by the zoom applied; `clean_bounding_boxes()` removes objects under 1 pixel wide or under 25 % visible (`MINIMUM_VISIBLE_AREA_FRACTION`). Photometric types leave the masks unchanged. Every type is off here.

### Step 3.9 — `forward_pass()` → `InstanceSegmenter.forward(ground_truth_bounding_boxes=..., ground_truth_labels=..., ground_truth_masks=...)`
`forward()` runs `forward_detection_stages()` (Step 3.10), then with the ground truth `compute_proposal_mask_loss()` (Steps 3.11 to 3.13), and in evaluation mode `predict_detection_masks()` (Step 4.3). Boxes without masks raise `ValueError: The instance segmenter needs the ground truth masks with the ground truth boxes.` The trainer sums every `_loss` key, each times its `Optimization.loss_weights` weight (1 in the shipped configurations).

**Example: the outputs of the first iteration of the CPU run**
```text
classification_scores                 (2, 128, 21) float32
sliced_origins                        (2, 128) int64
detection_classification_loss         2.9878      (RPN 0.6610 and 0.7074, regression 0.4909)
mask_loss                             0.7130
total                                 5.5602
```

### Step 3.10 — `forward_detection_stages()`
The unchanged detector (`two_stage_detector.py`; `object_detection_workflow.md`, Steps 3.10 to 3.18): backbone, proposals per level, joint proposal selection, target assignment, 128 sampled proposals per image, detection heads and four losses. It also returns what the mask branch needs: the feature maps and the sampled proposals with labels, levels and matched ground truths.

### Step 3.11 — `compute_proposal_mask_loss()`
As in Mask R-CNN, masks are learned on the positive sampled proposals only (label > 0). Each level runs its head on its own positives (`sliced_origins`), even when it has none, so every head stays in the graph (`predict_box_mask_logits()`). A shared head, or `MaskFineInputs`, runs once on the positives of all levels. The outputs of all levels are concatenated for `build_mask_targets()` (Step 3.12) and `compute_mask_loss()` (Step 3.13).

**Example: the first iteration of the CPU run**
```text
sliced_proposals                      (256, 5) float32     [image index, x1, y1, x2, y2] of the 2 x 128 sampled proposals
sampled 256, positive 12
level 5: feature map (2, 64, 16, 16)   7 positive proposals -> mask logits (7, 21, 28, 28)
level 6: feature map (2, 128, 8, 8)    3 positive proposals -> mask logits (3, 21, 28, 28)
level 7: feature map (2, 128, 8, 8)    2 positive proposals -> mask logits (2, 21, 28, 28)
```

### Step 3.12 — `build_mask_targets()`
The target of a proposal is its matched object's mask, cropped to the proposal and resized to the mask resolution by `roi_align()` on the full-size masks (`spatial_scale=1`, `aligned=True`), then thresholded at 0.5. It is the part of the object inside the proposal, in the proposal's frame.

**Example: the 12 positive proposals of the first iteration of the CPU run**
```text
proposal_boxes (12, 5) float32, 6 proposals in each image
mask_targets (12, 28, 28) float32, values {0.0, 1.0}, foreground share 0.5178
first target: 578 foreground pixels of 784
```

**Example: a toy target** (a square filling [100, 200, 156, 256] of a 512 pixel image, cropped by a proposal twice its size)
```text
target (1, 28, 28): 196 foreground pixels, rows 7 to 20, columns 7 to 20      <- the 14 x 14 centre of the 28 x 28 target
```

### Step 3.13 — `compute_mask_loss()`
Binary cross-entropy with logits between the target and the logit map of the **ground truth class**, in float32. The other 20 channels get no gradient, so classes do not compete for a pixel. With no positive it returns `sum(logits) * 0`, keeping the heads in the graph.

**Example: the first iteration of the CPU run**
```text
mask_logits (12, 21, 28, 28) float32   labels [14, 16, 6, 16, 20, 13, 17, 14, 13, 20, 6, 20]
class_logits (12, 28, 28)              <- the channel of the label of every proposal
mask_loss 0.7130 torch.float32         <- close to ln 2 = 0.693, the loss of a logit of 0
```

### Step 3.14 — `compute_batch_metrics()` + `MetricTracker.accumulate()` + `log_to_tensorboard()`
The 6 losses go to `Tensorboard/Train/` at every iteration; a few points of the run are shown below.

```text
iteration   total    RPN class.   RPN regr.   det. class.   det. regr.   mask
    1       5.6910   0.7085       0.8004      3.0862        0.3976       0.6982
 1000       1.0460   0.0254       0.1053      0.5723        0.1352       0.2077
 3000       0.6320   0.0125       0.0557      0.3137        0.0870       0.1631
 5000       0.4678   0.0087       0.0492      0.1912        0.0648       0.1539
```
At iteration 1 the detection classification loss is close to ln 21 = 3.0445 and the mask loss to ln 2 = 0.693, the losses of untrained logits. The detection classification loss is still falling at 5000. The mask loss falls mostly in the first 1000 iterations.

### Step 3.15 — `training_loss.backward()` + `step_optimizer()`
**Example: the gradient norm of every component in the CPU run** (the 5000-iteration run does not record it)
```text
iteration 1:  backbone 26.0453   region_proposer_dictionary 6.8338   detector_heads_dictionary 12.6682   mask_heads_dictionary 0.3247
iteration 12: backbone 15.9901   region_proposer_dictionary 4.3910   detector_heads_dictionary 8.1084    mask_heads_dictionary 0.4148
```

### Step 3.16 — validation pass
The 5000-iteration run has none. In the CPU run, at iteration 10, one batch runs in `eval()` mode without gradients, with boxes, labels and masks. It computes the same five losses and also predicts masks, which validation does not use.

**Example: the validation batch of iteration 10 of the CPU run**
```text
total 4.8811   RPN classification 0.6847   RPN regression 0.4954   detection classification 2.8054   detection regression 0.2007
mask 0.6948
```

### Step 3.17 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()`
**Example output: the summary of iteration 5000**
```text
========================================================================================================================
|   Iteration: 5 000  |   Total Images: 320 000    |  Interval Images: 64 000  |   Throughput: 43.7 Images Per Second  |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :     0.4768    -0.0717 Best
...
Moving Average of Train Mask Loss                      :     0.1509    -0.0080 Best
------------------------------------------------------------------------------------------------------------------------
Interval Of 1 000 Iterations Took 00:24:25.
========================================================================================================================
```
Iteration 1000 printed 33.5 Images Per Second and a mask loss of 0.3671. The interval time is wall-clock, so it includes the evaluations and the shared GPU load.

**Example: the MLflow metrics at the end of the run**
```text
train/Mask Loss 0.1509   train/interval_seconds 1465.099   train/throughput 43.6831   learning_rate 0.0002
status FINISHED   220 parameters   263 metrics
```

### Step 3.18 — end of the loop
Iteration 5000 is saved and evaluated in its iteration, so the loop saves and evaluates nothing more (`classification_workflow.md`, Step 3.25). Then `shutdown_loader_workers()` and `finish_experiment_tracking()` run.

**Example output: the end of the loop**
```text
Last Checkpoint Saved At Iteration 5000. Exiting Training.
```

The run folder then holds `Evaluation/Iteration_500/` to `Iteration_5000/` (with `EMA/`), the three TensorBoard writers, `Weights/`, `Results.txt` (20 reports, 2492 lines), `anchor_recall.csv`, `data.json`, `label_dictionary.json`, `mlflow_run.json` and `training_configuration.yaml`.

---

## Stage 4 — Test Evaluation of a Checkpoint

> Steps 4.6 to 4.10 and the progress bars of Step 4.1 come from the 5000-iteration run of Stage 3, mostly iteration 5000 on the whole test split (20,000 images, 69,682 objects), with `per_class_report: True`, 8 samples, `score_threshold: 0.05`, `target_precision: 0.99`, error images off, MLflow on. Every multiple of 500 is evaluated with the trained and the averaged weights. Steps 4.1 to 4.5 come from checkpoint 12 of the CPU run, on its test batch of images 2 and 7 (20 test images, 66 objects).
>
> These outputs predate the per-class recall intervals (2026-11-01): the `recall_50` to `recall_75_upper` columns (and `mask_` ones) and the `Mask Per Class Recall` and `Box Per Class Recall` tables (Steps 4.7 and 4.8).

### Step 4.1 — `run_test_evaluation()` → `evaluation_context()`
The evaluation loop of every task (`classification_workflow.md`, Step 4.1). The `evaluation_context()` of DET lowers the score threshold to `evaluation.score_threshold` and restores the deployment one afterwards.

**Example output** (thresholds from the CPU run, progress bars from iteration 5000: 20,000 images, batch 64)
```text
score threshold in the context: 0.05      after: 0.5
Test Evaluation Iteration 5000: 100%|██████████| 313/313 [02:18<00:00,  2.25it/s]
Test Evaluation Iteration 5000 (EMA Weights): 100%|██████████| 313/313 [02:15<00:00,  2.30it/s]
```

### Step 4.2 — `create_evaluation_accumulator()`
The accumulator of DET (`object_detection_workflow.md`, Step 4.2). Detections and ground truths also keep their mask overlaps and areas (Step 4.5).

**Example output: the CPU run**
```text
{'detections': list, 'ground_truths': list, 'confusion_matrix': (21, 21), ..., 'proposal_recall': dict}
```

### Step 4.3 — `prepare_batch()` + `predict_batch()` → `predict_detection_masks()`
`prepare_batch()` works as in training (Step 3.7). `predict_batch()` runs the model without ground truth. After post-processing (Step 4.4), `predict_detection_masks()` runs the mask heads on the **final, refined boxes**, as in Mask R-CNN. A detection uses the level of its proposal (`sliced_origins` at `final_detection_proposal_indices`), and the sigmoid of its class channel is kept. Padding detections get a mask of 0. Masks stay 28 x 28 in their box frame until pasted (Steps 4.5 and 4.6).

**Example: the predictions of the test batch of the CPU run**
```text
test batch: sample identifiers [2, 7], instance_masks (2, 512, 512) and (1, 512, 512), labels [[19, 5], [16]]
final_detection_boxes              (2, 100, 4) float32
final_detection_scores             (2, 100) float32
final_detection_labels             (2, 100) int64
number_detections                  (2,) int64        [100, 100]
final_detection_masks              (2, 100, 28, 28) float32
image 0: mask probabilities from 0.4792 to 0.5132, mean 0.4974; levels of the detections {7: 100}
first detections: boxes [[3.7, 264.9, 60.7, 388.8], [3.7, 200.7, 60.3, 324.7], [3.7, 331.4, 61.1, 452.9]]
                  scores [0.0528, 0.0528, 0.0528]   labels [20, 20, 20] (l_shape)
```
After 12 iterations the probabilities stay near 0.5: a pasted mask is a rough blob in the middle of its box.

### Step 4.4 — `postprocess_detections()`
The post-processing of DET, unchanged (`object_detection_workflow.md`, Step 4.4), with greedy NMS (`"iou"`). The exported `InferenceInstanceSegmenter` runs it as one call over the batch (`suppress_stacked_groups()`). The CPU run keeps 100 detections per image at 0.05, none at 0.5. At iteration 5000, 45,049 test detections reach 0.5.

### Step 4.5 — `accumulate_evaluation_batch()` → `gather_image_detections()` → `compute_mask_overlaps()`
For every image, `gather_image_detections()` keeps the valid detections with two more entries.
- `mask_overlaps` [D, G] float64: the mask IoU of every detection with every object, whatever their classes. `compute_mask_overlaps()` works by chunks (`MASK_OVERLAP_CHUNK_SIZE` 16 at 1024 pixels, 64 at 512), each in three steps.
  - `paste_instance_masks()` pastes the masks at their boxes by bilinear resampling, as batched products with separable weights (`build_interpolation_weights()`).
  - The pasted masks are thresholded at `mask_threshold` (`>= 0.5`).
  - `compute_mask_intersection_over_union()`: `cut_mask_segments()` cuts the flat masks into segments of `MASK_PRODUCT_SEGMENT_PIXELS` (1024). One batched product multiplies each segment with the same segment of every ground truth mask, in float16 on the GPU (exact, since a count never exceeds 1024; float32 on CPU): 16 ms per 1024 pixel benchmark image instead of 96. Counts and areas (`count_segment_pixels()`) are divided in float64 (in float32, 13 / 20 falls below 0.65).
- `mask_areas` [D] int64: the pixels of every binary detection mask, for the mAP by size.

`gather_image_ground_truth()` adds the ground truth `mask_areas` (of `original_instance_masks` with `evaluation.original_resolution`). Detections reaching the deployment threshold are matched on masks for the confusion matrix (`compute_overlap_matrix(overlap_type="masks")`, `match_image_detections()` at IoU 0.5). Proposal recall stays on boxes.

**Example: the first image of the test batch of the CPU run** (2 objects, capsule (19) and pentagon (5))
```text
pasted masks of the first chunk: (16, 512, 512) float32, max 0.5121
first detection: box area 7058.3 pixels, 7552 pasted pixels > 0, 6523 pixels >= 0.5
best mask IoU per object [0.2133, 0.3883]       best box IoU per object [0.3533, 0.508]
ground truth mask_areas [3310, 5961]            box areas [5829, 8835]
after the batch: confusion matrix sum 3, all 3 objects in the background column (missed)
```
A box overlaps the pentagon at 0.508 but a mask only at 0.3883, so on masks it is not found at 0.5. Mask areas are smaller than box areas, which moves objects between sizes (Step 4.7).

**Example: a toy mask pasted at its box** (a 28 x 28 mask whose 14 x 14 centre is 1, pasted into [100, 200, 156, 256])
```text
pasted mask: 784 pixels, rows 214 to 241, columns 114 to 141      <- the 28 x 28 centre of the 56 x 56 box
IoU with a ground truth mask equal to that square and with an empty one: [[1.0, 0.0]]
```

### Step 4.6 — `save_sample_images()` → `draw_test_image()` → `get_sample_masks()` + `build_detection_sample_grid()`
The samples of DET (`object_detection_workflow.md`, Step 4.6): ground truth and detections reaching the deployment threshold side by side. `get_sample_masks()` adds the ground truth masks and the detection masks pasted and thresholded at `mask_threshold`, filled with their class colour under the boxes.

**Example: `Evaluation/Iteration_5000/Samples/`, and the content of `test_014285.png`**
```text
test_000000.png, test_002857.png, test_005714.png, ..., test_019999.png      every file 1028 x 560 pixels RGB

test_014285.png:     "6 ground truth boxes | 4 detections with score >= 0.50"
                     Ground Truth (square, triangle, star, parallelogram, 2 nonagon masks and boxes) |
                     Prediction (square 0.97, triangle 0.75, star 0.99, parallelogram 0.87, with their masks)
EMA/Samples/test_014285.png: "6 ground truth boxes | 3 detections with score >= 0.50" (square 0.96, star 0.99, parallelogram 0.82)
```
Both weights miss the two nonagons. Star tips come out rounded and the thin end of the triangle is cut: 28 x 28 masks lose thin parts (Pitfalls).

### Step 4.7 — `compute_evaluation_results()`
The DET evaluation runs twice on the same detections, as the list below shows.
- `summarise_average_precisions(overlap_type="masks", title_prefix="Mask", name_prefix="Mask ", column_prefix="mask_")`: mask mAP, matched by mask IoU, three methods, sizes by **mask area** (`select_object_size_detections()` reads `mask_areas`).
- `summarise_average_precisions(overlap_type="boxes", title_prefix="Box")`: box mAP, sizes by box area, comparable with DET runs.

Then `save_proposal_recall()` (`object_detection_workflow.md`, Step 4.15), `write_class_metrics_file()` (Step 4.8) and `save_error_analysis()` with thresholds of the **mask** APs (Step 4.9). The report holds eleven tables in this order: mask mAP by method, by size, box mAP by method, by size, mask mAR, box mAR (`object_detection_workflow.md`, Step 4.14), proposal recall, mask per class, `Mask Per Class Recall`, box per class, `Box Per Class Recall` (recall at 0.5 and 0.75 with a 95 % interval, Steps 4.11 and 4.13 of that file). With `evaluation.original_resolution`, boxes are restored, masks pasted at the original size and ground truth taken from the original annotations.

Checkpoints are compared on `Mask mAP@0.5/area`, also in `percent_metrics` so the best checkpoint line shows a percentage (Step 5.5). The `percent_columns` of every table print rates as `63.72 %`; counts, thresholds and proposals per image keep their values, and files and scalars stay between 0 and 1.

**Example: the report of iteration 5000** (shortened; the same text without file lines goes to `Results.txt`)
```text
Iteration 5000 · 20000 Test Images · 2026-10-24 17:48 ──────────────────────────────────────────────────────────────────
  Confusion Matrix : score >= 0.50, IoU >= 0.5 (masks), 32850 right, 8650 wrong class, 3549 false positives, 28182
  missed

  Mask Mean Average Precision
   Method      mAP@0.5   mAP@0.75   mAP@0.95   mAP@[0.5:0.95]
  ────────────────────────────────────────────────────────────
   Area        63.72 %    54.67 %     1.57 %          46.91 %
   ...

  Mask Mean Average Precision By Object Size (Area)
   Object Size           Objects   mAP@0.5   mAP@0.75   mAP@0.95   mAP@[0.5:0.95]
  ────────────────────────────────────────────────────────────────────────────────
   Small (< 32²)             911    5.37 %     3.79 %     0.00 %           3.24 %
   Medium (32² to 96²)     50627   63.44 %    54.02 %     1.60 %          46.16 %
   Large (>= 96²)          18144   62.29 %    56.33 %     2.06 %          47.58 %

  Box Mean Average Precision
   Method      mAP@0.5   mAP@0.75   mAP@0.95   mAP@[0.5:0.95]
  ────────────────────────────────────────────────────────────
   Area        64.04 %    58.18 %     0.16 %          46.08 %
   ...
  ... box by size, mask and box mean average recall, proposal recall (Step 4.8)

  Mask Per Class Average Precision (Area)
                   Ground                                                 Threshold    Recall
   Class           Truths    AP@0.5   AP@0.75   AP@0.95   AP@[0.5:0.95]     @ P99 %   @ P99 %
  ────────────────────────────────────────────────────────────────────────────────────────────
   heptagon          3466   26.85 %   26.06 %    0.15 %         22.17 %           —         —
   rhombus           3436   33.85 %   31.89 %    0.27 %         27.00 %      0.4817    0.06 %
   hexagon           3471   35.08 %   33.93 %    0.56 %         28.99 %           —         —
   ...
   square            3545   96.24 %   90.49 %    0.17 %         73.71 %      0.7861   85.22 %
   star              3549   97.29 %   30.22 %    0.00 %         48.31 %      0.8004   91.91 %
   annulus           3461   97.69 %   81.02 %    0.00 %         66.44 %      0.7104   92.40 %

  Box Per Class Average Precision (Area)
   ...
   star              3549   98.11 %   89.34 %    0.01 %         66.11 %
  - Results : Results.txt
  ...
```
The list below gives what the report shows.
- By mask area, 911 / 50,627 / 18,144 objects are small / medium / large (COCO sizes of that time); by box area, 184 / 33,579 / 35,919. A shape fills only part of its box.
- Mask `mAR@0.5` is 96.75 % and box `mAR@0.5` 97.36 %.
- Mask and box mAP@0.5 are close (63.72 % and 64.04 %). Masks fall behind at 0.75 (54.67 % against 58.18 %), most for thin shapes: the star has mask AP@0.75 30.22 % against box 89.34 %.
- Mask mAP@0.95 (1.57 %) beats box (0.16 %), carried mostly by the circle (23.19 % against 1.12 %).
- The weakest classes are many-sided polygons (heptagon, octagon, nonagon, hexagon) and quadrilaterals (trapezoid, rhombus, kite), each under a mask AP@0.5 of 39 %.

So that the swing of the trained weights is visible, the table below compares three evaluations of the run.

| Evaluation | Mask mAP (area, 0.5 / 0.75 / 0.95 / [0.5:0.95]) | Box mAP@0.5 | Right / wrong class / false positives / missed |
|---|---|---|---|
| 4500, trained | 69.49 / 60.22 / 1.49 / 52.30 % | 69.81 % | 32,234 / 3,529 / 2,956 / 33,919 |
| 5000, trained | 63.72 / 54.67 / 1.57 / 46.91 % | 64.04 % | 32,850 / 8,650 / 3,549 / 28,182 |
| 5000, EMA | 73.64 / 64.81 / 4.13 / 57.43 % | 73.93 % | 33,597 / 3,131 / 2,248 / 32,954 |

The trained mask mAP@0.5 rises at every evaluation to 4500, then falls at 5000. Polygons slip by one side: 2,053 hexagons found as pentagons and 1,803 heptagons as hexagons at 5000, against 450 and 402 at 4500. The hexagon mask AP@0.5 falls from 70.08 % to 35.08 %. The averaged weights rise at every evaluation (hexagon 86.38 % at 5000) and give the best checkpoint. With a constant learning rate of 0.0002 a single checkpoint can be caught in such a swing; the average smooths it.

### Step 4.8 — `write_class_metrics_file()`
One row per class: the box columns of DET, the same with `mask_`, then the threshold and recall at the target precision (of the mask APs), 37 columns. `recall_50` to `recall_75_upper` (`object_detection_workflow.md`, Step 4.11) follow the box sizes; `mask_recall_50` to `mask_recall_75_upper` follow the mask sizes.

**Example: `Evaluation/Iteration_5000/class_metrics.csv`** (first rows)
```text
class_index,class,ground_truths,average_precision_50,average_precision_75,average_precision_95,average_precision_50_95,
average_precision_50_small,...,mask_average_precision_50,...,mask_average_precision_50_95_large,
threshold_at_target_precision,recall_at_target_precision
1,square,3545,0.965184,0.882617,0.001079,0.699232,...,0.786123,0.852186
2,circle,3566,0.888822,0.819614,0.011202,0.645225,...,0.463157,0.505328
```

**Example: `Evaluation/Iteration_5000/proposal_recall.csv`**
```text
group,objects,recall_50,recall_70,proposals_per_image
all,69682,0.996341,0.974125,250.000000
small,184,0.358696,0.081522,
...
block_5,69682,0.781091,0.522947,149.250250
block_6,69682,0.805043,0.216799,50.601800
block_7,69682,0.704400,0.500416,50.147950
```
The proposals overlap 99.63 % of the test objects by 0.5, small ones excepted (66 of 184). The errors come from classification and masks, not from the proposals.

### Step 4.9 — `save_error_analysis()`
The class thresholds and confusion matrix of DET (`object_detection_workflow.md`, Step 4.12), with thresholds from the mask APs and matches on masks (Step 4.5).

**Example output: iteration 5000** (`class_thresholds.json` is indented, shortened here)
```text
class_thresholds.json:  {"target_precision": 0.99, "classes": {"square": {"threshold": 0.7861230373382568, "recall": 0.8521861777150916},
                         ..., "hexagon": null, ...}}
                         15 classes with a threshold, null for the hexagon, octagon, trapezoid, heptagon and nonagon
confusion_matrix.csv:   true \ predicted,background,square,circle,triangle,rectangle,pentagon,hexagon,...
                        background,0,126,7,765,221,318,21,...       <- 3,549 false positives in the row
                        hexagon,780,0,0,0,0,2053,625,...            <- 2,053 hexagons found as pentagons, 625 right
                        heptagon,1552,0,0,0,0,77,1803,...           <- 1,803 heptagons found as hexagons
```
No octagon, rhombus, trapezoid or nonagon is right at the deployment threshold; 12 kites and 23 heptagons are. Octagons go to the background (2,834 of 3,454 missed) or to heptagons (507).

### Step 4.10 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
The report of Step 4.7 is printed and appended to `Results.txt`; scalars and images go to TensorBoard and MLflow (`object_detection_workflow.md`, Step 4.13). `update_evaluation_curves()` then adds a row to `evaluation_summary.csv` and redraws `evaluation_curves.png`.

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
MLflow writes `@` as `_at_`. `best_` keeps the best headline value, its iteration in `best_iteration`. The 126 scalars are 54 for boxes (18 mAP, 16 mAR, 20 `AP@0.5/<class>`), the same 54 for masks and 18 proposal recall values.

---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples come from the script on the 5000-iteration run (20 checkpoints) with `iteration = "all"`, `batch_size = 64`, `track_evaluation = False`, on the first 200 test images (709 annotations), in 25 minutes. Step 5.4 comes from a tracked evaluation of the CPU run. They predate the per-class recall intervals.

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The same script for every task (`classification_workflow.md`, Stage 5): give the run folder as `--experiment_folder`, choose the iterations with `--iteration` and run it. Its arguments work as listed below.
- `--maximum_error_images_per_class_pair` (`100000` by default) turns error images on; `None` keeps `evaluation.error_dumps`.
- `--target_precisions` replaces the target precision by one value (no list for detection tasks).
- `--nms_criterion` and `--nms_distance_exponent` replace the final NMS, e.g. `distance_iou` with 0.6, written to `<run>/Checkpoint_Evaluation_nms_<criterion>[_<exponent>]/` (`apply_nms_criterion()`); without them the run's is kept.

### Step 5.2 — trainer construction
The trainer is built from the run's `training_configuration.yaml`, with the classes of its `label_dictionary.json`, writing in `<run>/Checkpoint_Evaluation/`.

The launch information then shows `Test Data Folder : test_reduced_200 (200 images, 709 annotations)`.

### Step 5.3 — `get_checkpoint_iterations()`
With `--iteration all` it returns the 20 checkpoints from newest to oldest: `[5000, 4750, ..., 500, 250]`.

### Step 5.4 — `evaluate_checkpoints()` → `start_evaluation_tracking()`
With tracking on, the evaluations go to their own MLflow run, `<run name>-evaluation`, tagged with the training run (`evaluation_of`, Step 3.3), the folder, test split and iterations.

**Example output** (the tracked evaluation of the CPU run)
```text
Evaluating : model_000012.pt
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
The current code adds an `Evaluation Curves : evaluation_curves.png` line and an `ema_number_updates` column (254 columns).

**Example: `<run>/Checkpoint_Evaluation/evaluation_summary.csv`** (253 columns: the iteration, 126 trained scalars, the same 126 with `ema_`; a few columns, 4 decimals)
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
The 200 images tell the same story as the whole split (Step 4.7): the trained weights peak at 4500 and fall at 5000, the hexagon mask AP falling from 0.7340 to 0.2877. The averaged weights rise at every checkpoint. They lead from 4250 on and at 3500 and 3750; earlier the average still holds older weights. To deploy the trained weights, take iteration 4500; the averaged weights of 5000 beat both.

`<run>/Checkpoint_Evaluation/` then holds `evaluation_summary.csv`, `Iteration_250/` to `Iteration_5000/` (files of Steps 4.6 to 4.9 with `EMA/`), `Tensorboard/Test/` and `Test_EMA/`, `Results.txt` (40 reports) and the run files.

### Step 5.6 — `run_test_evaluation()` for each checkpoint
**Example output: the start of the report of iteration 5000 on the 200 images**
```text
Iteration 5000 · 200 Test Images · 2026-10-25 07:23 ─────────────────────────────────────────────────────────────────
  Confusion Matrix : score >= 0.50, IoU >= 0.5 (masks), 310 right, 96 wrong class, 45 false positives, 303 missed
  Error Images : 421 in 53 folders, at most 100000 per folder
  Sample Images : 8 fixed test images
  Mask Mean Average Precision   Area   62.87 %    52.94 %     1.29 %          45.78 %
  Box Mean Average Precision    Area   62.87 %    57.95 %     0.63 %          46.37 %
```
The averaged weights of the same checkpoint find 321 objects right with 31 of the wrong class, against 310 and 96 for the trained weights.
