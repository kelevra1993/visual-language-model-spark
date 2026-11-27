# Object Detection Workflow

This document follows one object detection run (DET) from `main()` to the evaluation of a checkpoint. The **Call Tree Overview** gives the whole run on one page. Each stage then has one step per function, with a short explanation and a **real** output. The steps shared by every task (configuration loading, checkpoints, training loop, MLflow tracking, standalone evaluation) are explained in `classification_workflow.md`. The steps specific to detection (COCO loaders, anchors, region proposals, target assignment, detection heads, the four losses, post-processing, mean average precision) are explained here in full.

The examples come from three sources, listed in this table.

| Examples | Source |
|---|---|
| Stages 1 and 2 | `vision_training/configurations/object_detection/configuration_synthetic_shapes.yaml` as it was then (1024 pixel images, batch size 40, detection on blocks 5, 6 and 7), run on CPU on a small generated dataset (`create_synthetic_dataset(task="object_detection")`: 100 / 20 / 20 images of 512 pixels, 1 to 6 shapes, 9 classes), the run folder redirected to a temporary folder |
| Stages 3 to 5, outputs of the run (checkpoints, summaries, reports, images, scalars, standalone evaluation; Step 5.4 excepted) | the 5000-iteration run of [`run_configurations/object_detection_workflow_run.yaml`](run_configurations/object_detection_workflow_run.yaml), trained on the GPU on 2026-10-24 on the dataset of that day (128,000 / 20,000 / 20,000 images, 20 classes): the synthetic configuration of that day with 5000 iterations instead of 2000, an evaluation every 500 (an intermediate checkpoint every 250), MLflow on, experiment folder `experiments/workflow_runs` |
| Stages 3 and 4, tensors of one batch (Steps 3.6 to 3.18, 3.20, 3.21, 4.2 to 4.4, 4.8, 4.9) | the CPU run: the Stage 1 configuration at 512 pixels, batch size 2, 12 iterations, a checkpoint every 6, a summary every 4, a validation batch every 10, MLflow on. It finds no shape: its values only show the shapes and order of the steps |
| Step 2.21 (anchor recall) | the first launch of the synthetic configuration of 2026-11-27 (512 pixels, blocks 5, 6 and 7, batch size 8) on the dataset of that day (128,000 / 20,000 / 20,000 images, 35 classes), the run folder redirected to a temporary folder |

The CPU runs used the PyTorch loader; the DALI loader of that time has since been removed (`19b89a0`), every split being read by `CocoDetectionDataset` on the CPU loader workers.

This table lists what changed since the examples were recorded. The 5000-iteration run shows the first four changes; its configuration keeps the old values of the DIoU-NMS, proposal selection and learning rate rows, and its reports use the COCO sizes of the last row.

| Change | What the CPU runs show |
|---|---|
| Ground truth boxes added to the detection head proposals in training and validation (`add_ground_truth_proposals()`, `a10bb7a`) | Steps 3.14 and 3.21: the region proposal positives alone |
| Region proposal NMS 0.7, detector NMS 0.5 | 0.5 and 0.8 |
| IoU 0.95 columns (`mAP@0.95`, `AP@0.95`, `average_precision_95`, `351a4d1`), `Mean Average Recall` table | not shown |
| Checkpoints record the source run of the heads; the moving average of the weights (`Optimization.exponential_moving_average`) is evaluated in `EMA/`, `Tensorboard/Test_EMA/` and an `EMA Weights` block | not shown |
| Residual shortcuts listed only for a block adding one (`count_residual_additions()`) | Step 2.19 printed again with the current code |
| `Neck` section (`check_neck_configuration()`, `Neck Architecture` panel), `Detector.box_loss` (`check_box_loss_configuration()`) | `none`, smooth L1: outputs unchanged |
| Validation loader built only when `validation_iteration_ratio` is above 0 or for a diagnosis mode ending the program (`BaseTrainer.builds_validation_loader`) | Step 2.8 builds three loaders; the synthetic and 5000-iteration configurations now build two |
| `Detector.nms_criterion: "distance_iou"` with exponent 0.6 (DIoU-NMS, https://arxiv.org/abs/1911.08287, `check_nms_criterion_configuration()`) | `"iou"` |
| `RegionProposal.proposal_selection: "per_level"` in the object detection configurations (`"joint"` in instance segmentation) | `"joint"` |
| Learning rate 1e-3, `background_iou_threshold.min: 0.1`, `Neck.type: "feature_pyramid"`, chosen by the detection benchmark (`documentation/architecture_benchmarking/object_detection_experiment_plan.md`) | 2e-4, 0.1, `"none"` (the 5000-iteration run: minimum 0.0) |
| Five object sizes relative to the image side (Step 4.14) instead of the COCO small, medium and large: `Object Side` and `Pixels` columns, `By Size/under_1_16` scalars, `average_precision_50_under_1_16` columns. Anchor recall of the training and test splits at the first launch only, in `anchor_recall.csv` and `anchor_recall.png` (Step 2.21) | `Small (< 32²)`, `Medium (32² to 96²)`, `Large (>= 96²)`, also in the 5000-iteration run (Steps 4.7, 4.11, 4.13 to 4.15) |

The current synthetic configuration trains at 512 pixels on the instance segmentation dataset (blocks of 16 and 32 filters, two anchor scales per level, batch size 64, 2000 iterations, augmentation probabilities 0.25, 0.5 for flips). `Read This First` describes it. The configuration of Stages 1 and 2 matches `vision_training/configurations/object_detection/configuration_example.yaml` (same backbone, levels and heads, at 1024 pixels).

## Read This First
Before following the stages, this summary gives the purpose of the task, its code, how to run it and what to watch.

The detector finds every object of an image: its box, its class and a score. It is a two-stage detector in the style of Faster R-CNN, with a feature pyramid. Every backbone block named in `DetectionLevels` gets its own anchors, region proposal head and detection head. The input is a COCO split (images and a `labels.json` whose objects have a `bbox`). The output is a fixed number of detections per image (`Detector.inference.maximum_detections`, padded), boxes in network input pixels.

Choose DET when boxes are enough. INS (`instance_segmentation_workflow.md`) adds a mask head per level to the same detector. KPT describes objects by keypoints (a bottom-up HigherHRNet, without boxes or anchors). SEG gives the class of every pixel without telling objects apart; CLS one label per image.

### The Task At A Glance
To find your way in the stages below, this block gives the shapes of the synthetic configuration (512 pixels, batch size 64, 35 classes and the background), checked on the CPU with the last checkpoint of a run of this configuration.

```text
<dataset>/{train,validation,test}/data/*.png + labels.json (COCO bbox [x, y, w, h], category_id) + label_mapping.json
   │ CocoDetectionDataset on the CPU workers: read_image, select_image_objects, letterbox to Data.image_settings.size
   ▼
sample  images [3, 512, 512] uint8 BGR · bounding_boxes [N, 4] x1 y1 x2 y2 · labels [N] in 1..35 (0 = background)
batch   images [64, 3, 512, 512] · boxes and labels: lists of 64 tensors · DataAugmenter moves the boxes (GPU)
   ▼
Backbone, 7 ConvolutionBlocks ─► block 5 [64, 32, 16, 16] stride 32 · block 6 [64, 32, 8, 8] · block 7 [64, 32, 8, 8]
   ▼ FeatureNeck feature_pyramid: 1x1 lateral per level, top-down sum (nearest upsampling), 3x3 conv, same shapes
   ▼ one RegionProposal per level: 6 anchors per cell (2 scales x ratios 0.5, 1, 2), 1536 + 384 + 384 = 2304 anchors
objectness [64, 2304] · offsets [64, 2304, 4] ─► RPN losses: BCE + smooth L1 (beta 1/9), 256 anchors per image sampled
   ▼ RegionProposalFilter, per level: top 1000, NMS 0.7, then the best 500 per image (evaluation: 500 per level, 250)
   ▼ training: + ground truth boxes, 128 proposals per image sampled, at most 64 positive (IoU >= 0.5)
DetectionHead of the level of every proposal: RoIAlign 7x7 ─► 2 conv x 32 ─► FC 1568 > 64 > 32 ─► class + box layers
logits [64, 128, 36] · offsets [64, 128, 36, 4] ─► detection losses: cross entropy + smooth L1 of the true class
   ▼ evaluation: postprocess_detections: softmax, score >= 0.05 (0.5 deployed), DIoU-NMS 0.5 per class, top 100
final_detection_boxes [B, 100, 4], _scores [B, 100], _labels [B, 100] · number_detections [B]
   ▼
evaluate_detections ─► mAP@0.5 / 0.75 / 0.95 / [0.5:0.95] x area, 11, 101 point · by size · mAR · AP per class
                       headline mAP@0.5/area · proposal recall · class thresholds · confusion matrix
```

### Files To Know
Should you need to change one part of the task, this table gives the file holding it and the functions to read first.

| Role | File | Main functions or classes |
|---|---|---|
| Trainer | `vision_training/trainers/detection_trainer.py` (bases `coco_split_trainer.py`, `base_trainer.py`) | `DetectionTrainer`: `build_dataloaders`, `analyse_training_data`, `forward_pass`, `predict_batch`, `accumulate_evaluation_batch`, `compute_evaluation_results` |
| Dataset and loaders | `vision_training/data/detection/coco_dataset.py`, `vision_training/data/serialized_records.py`, `vision_training/data/loaders.py` | `CocoDetectionDataset` (`get_image_record`), `SerializedRecords`, `select_image_objects`, `collate_function`, `get_detection_dataloaders` |
| Model | `vision_training/models/detection/two_stage_detector.py`, `feature_neck.py` | `TwoStageDetector`: `forward_detection_stages`, `_prepare_detector_inputs`, `get_level_anchors`, `print_summary`; `FeatureNeck` |
| Anchors and proposals | `vision_training/models/detection/anchors.py`, `region_proposer.py` | `Anchors.generate_anchors`, `RegionProposal`, `RegionProposalFilter`, `select_best_proposals` |
| Head and boxes | `vision_training/models/detection/detection_head.py`, `box_operations.py`, `postprocessing.py` | `DetectionHead`, `apply_regression_predictions`, `BOX_REGRESSION_WEIGHTS`, `postprocess_detections` |
| Targets and losses | `vision_training/models/detection/target_assignment.py`, `vision_training/losses/detection_losses.py` | `batch_assign_targets_to_anchors`, `add_ground_truth_proposals`, `batch_assign_targets_to_proposals`, `sample_positive_and_negative_training_targets`, `assign_region_proposal_targets`, `compute_region_proposal_losses`, `compute_detection_losses` |
| Metrics and evaluation | `vision_training/evaluation/detection/mean_average_precision.py`, `average_precision_report.py`, `region_recall.py`, `detection_confusion.py`, `object_sizes.py`, `vision_training/evaluation/recall_intervals.py` | `evaluate_detections`, `summarise_average_precisions`, `summarise_anchor_recall`, `summarise_proposal_recall`, `match_image_detections`, `compute_clopper_pearson_intervals` |
| Configuration checks and run name | `vision_training/configuration_processing/detection_checks.py`, `run_name.py` | `check_detection_configuration`, `build_detection_sections`, `build_anchors_hash` |
| Synthetic data | `vision_training/data/synthetic/create_synthetic_dataset.py` | `create_synthetic_dataset` (`--task=instance_segmentation` for the DET configuration) |
| Export | `vision_training/models/detection/inference_detector.py`, `vision_training/deployment/export_specifications.py` | `InferenceDetector`, `get_export_specification` (inputs `images`, outputs `detection_boxes`, `detection_scores`, `detection_labels`, `number_detections`) |
| Export sizes at launch | `vision_training/deployment/export_size_estimates.py` | `estimate_export_sizes`, `probe_inference_graph`, `print_export_size_summary` |

### Run It
In order to train on the synthetic shapes, generate the dataset, train, then evaluate checkpoints again if needed.

The DET configuration reads the instance segmentation dataset (boxes of the visible part of every shape). The `instance_segmentation` entries of `TASK_IMAGE_SIZES`, `TASK_SPLIT_SIZES` and `TASK_MAXIMUM_OBJECTS` of `create_synthetic_dataset.py` give 512 pixels, 128,000 / 20,000 / 20,000 images and 5 to 6 shapes, among the 35 shapes of `SHAPE_CATEGORIES` (`MC35`), with `--same_class_neighbour_probability` at its default 0.

```bash
# Generate the dataset (the root folder is required)
uv run python -m vision_training.data.synthetic.create_synthetic_dataset --task=instance_segmentation --output_folder="$HOME/Data/instance_segmentation_data/synthetic_shapes"

# Train (first set the three data folders, Data.label_mapping.mapping_file_path and experiment_folder as absolute paths)
uv run python -m vision_training.main --configuration_path="vision_training/configurations/object_detection/configuration_synthetic_shapes.yaml"

# Evaluate the last checkpoint again (arguments below, Stage 5; run it without any for its help)
uv run python -m scripts.evaluation.evaluate_checkpoint --experiment_folder <run folder>
```

`evaluate_checkpoint.py` requires only `--experiment_folder`. It also takes `--iteration` (the last checkpoint by default, a number, or `all`), `--test_data_folder`, `--batch_size`, `--track_evaluation`, `--maximum_error_images_per_class_pair`, if needed `--target_precisions` (a single precision), and `--nms_criterion` and `--nms_distance_exponent` to score with another final NMS. It writes in `<run>/Checkpoint_Evaluation/`, or `<run>/Checkpoint_Evaluation_nms_<criterion>[_<exponent>]/` with another NMS.

### Configuration Keys To Know First
If one wishes to change what the detector learns or how it filters its outputs, these keys of the synthetic configuration matter most.

| Key | Value in the synthetic configuration | Effect |
|---|---|---|
| `DetectionLevels.<5, 6, 7>.anchors` | scales `[32, 64]`, `[64, 128]`, `[128, 256]`, aspect ratios `[0.5, 1.0, 2.0]` | 6 anchors per cell, 2304 in all at 512 pixels (`A6.32-256.e7b72c`); fit measured by `anchor_recall.csv` |
| `DetectionLevels.<level>.region_proposal`, `.detector` | `[2, 32]`; RoIAlign 7, `[2, 32]`, fully connected `[64, 32]` | the heads of every level (`RBN-R2x32`, `DSE-DRES-DBN-D7.2x32.64.32`) |
| `Neck.type`, `output_channels`, `normalization` | `feature_pyramid`, 32, batch | merges the levels into `output_channels` channels (`L567-NFPN32BN`); `none` lets the heads read the blocks, `path_aggregation` adds a bottom-up path (`NPAN`); `pretrained` loads the neck of another run (`NPT`) |
| `RegionProposal.foreground_iou_threshold.min`, `background_iou_threshold` | 0.5; min 0.1, max 0.3 | anchors positive from 0.5, negative from 0.1 to 0.3, ignored below 0.1 and in between, plus the best anchor of every object |
| `RegionProposal.training`, `.inference`, `nms_iou_threshold` | 1000 then 500; 500 then 250; 0.7 | proposals given to the detection heads per image |
| `RegionProposal.proposal_selection` | `per_level` | top-k and NMS per level, then the best over all levels (`RPL`); `joint` ranks all anchors at once, as the run of Stages 3 to 5 (Step 3.13) |
| `RegionProposal.number_training_positives`, `total_training_samples` | 128, 256 | anchors sampled per image for the region proposal losses |
| `Detector.foreground_iou_threshold`, `number_training_positives`, `total_training_samples` | 0.5, 64, 128 | proposals sampled per image for the detection losses, ground truth boxes included |
| `Detector.inference.score_threshold`, `maximum_detections`, `Detector.nms_iou_threshold` | 0.5, 100, 0.5 | post-processing of the deployed detector, also used by the sample images, the confusion matrix and the error images |
| `Detector.nms_criterion`, `nms_distance_exponent` | `distance_iou`, 0.6 | final NMS: plain IoU (`iou`), or the IoU minus the normalized centre distance to the power of the exponent (DIoU-NMS); the run of Stages 3 to 5 ran `iou` |
| `ExperimentConfiguration.evaluation.score_threshold`, `target_precision` | 0.05, 0.99 | detections kept for the mAP (refused above `Detector.inference.score_threshold`); precision of the class thresholds of `class_thresholds.json` |
| `Optimization.loss_weights` | 1.0 for each of the four losses | weights in the total loss, the losses being tracked unweighted |
| `Data.image_settings.size` | 512 | network input; the anchor scales must fit in it |

### What A Run Writes
So that a run folder can be read without opening the code, this table lists the files a detection run writes.

| File or folder | When | Content |
|---|---|---|
| `anchor_recall.csv`, `anchor_recall.png` | first launch, again when the split folders or the threshold change | share of the training and test objects with a foreground anchor, per level and size, and its tables drawn (Step 2.21) |
| `Input_Samples/` with `samples.json` | before the first iteration | the first 10 images of every loader as the network receives them (`training_reference.md`, Run Folder) |
| `Evaluation/evaluation_summary.csv`, `evaluation_curves.png` | after every evaluation | one row per checkpoint (averaged weights in `ema_` columns), and its curves |
| `Evaluation/Iteration_<n>/` | every `weight_saving_iterations` checkpoint | `class_metrics.csv` (Step 4.11), `class_thresholds.json`, `confusion_matrix.csv` and `.png` (Step 4.12), `proposal_recall.csv` (Step 4.15), `Samples/<image>.png` (Step 4.6), `Errors/` with `error_dumps`, the same files in `EMA/` |
| `Results.txt` | every evaluation | one block per evaluation, two with the moving average (Step 4.13) |
| `Tensorboard/Train`, `Test`, `Test_EMA` | during the run | the total loss, the four losses and the learning rate; the evaluation scalars (72 for 20 classes and three levels, 90 for 35 classes and four levels), 8 samples and the confusion matrix |

This table gives the iteration 5000 results of the two runs this document quotes, at the deployment threshold for the counts.

| Run | `mAP@0.5/area` (EMA) | `mAP@[0.5:0.95]` (EMA) | Proposal recall at 0.5 | Right / wrong class / false positives / missed |
|---|---|---|---|---|
| DET run of `~/Data/tutorial_runs/tutorials` (anchors 16 to 384 pixels, first 5,000 test images, stopped at 5000 of 8000 planned iterations, `Checkpoint_Evaluation` of 2026-11-22) | 75.24 % (77.78 %) | 55.05 % (63.55 %) | 98.39 % | 17,096 / 2,347 / 1,775 / 8,005 |
| Run of Stages 3 to 5 (`object_detection_workflow_run.yaml`, mAP still rising) | 71.51 % (72.08 %) | 51.60 % (56.73 %) | 99.67 % | 34,213 / 3,228 / 3,038 / 32,241 |

### Pitfalls
Suppose a detection run behaves unexpectedly: these behaviours are the first to check.

- The synthetic configuration stops at 2000 iterations, where few detections reach 0.5. In the DET tutorials run, `mAP@0.5/area` is 12.47 % at iteration 1000 (no detection reaching 0.5) and 37.81 % at 2000 (`suggestions.md`, item 30).
- Small objects get few anchors: the finest level has a stride of 32 and anchors of 32 and 64 pixels. Block 5 gives a foreground anchor to 8.11 % of the 150,120 training objects under 1/16 of the side (under 32 pixels at 512) of the current dataset. The tutorials run adds block 4 (stride 16, anchors of 16 and 32 pixels): 43.99 %, and its 5,778 test objects under 32 pixels reach `mAP@0.5` 71.25 % at iteration 5000 (80.09 % from 32 to 96 pixels, 73.17 % above, the COCO sizes of that evaluation).
- With `"joint"` (the run of Stages 3 to 5, audit M3), 166.9 of the 250 proposals per image come from block 5 (Step 4.15). `"per_level"` still ranks all levels together in its last top-k, so it reserves no slots: in the tutorials run, 238.7 of 250 come from block 4, its finest level.
- The mAP matches detections with the PASCAL VOC rule at every IoU, so `mAP@[0.5:0.95]` and the 101-point values are not those of pycocotools (`match_detections_to_ground_truth()`, audit L6).
- A head or block whose first convolution changes the channel count adds no residual shortcut with 2 layers (`ConvolutionBlock.count_residual_additions()`): the heads of blocks 6 and 7 of Stage 1 (128 to 64 channels, no neck). The feature pyramid of `configuration_example.yaml` gives them 64 channels, so they add it.

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

To be able to follow one detection run from its launch to the evaluation of a checkpoint, this tree gives every function in the order it runs.

```text
main()                                    entry point of every task, --configuration_path
├── load_experiment_configuration()       ExperimentConfiguration + Optimization, Data, Backbone, DetectionLevels, Neck,
│   │                                     RegionProposal, Detector; adds label_mode and label_dictionary
│   ├── load_configuration() / build_label_dictionary()   {category id: class index} + 'background': 0
│   └── check_configuration() -> check_detection_configuration()
│           evaluation, anchor coverage, data (classes 0 to K-1), no sequence normalization, image-only augmentations,
│           DetectionLevels, Neck, RegionProposal, Detector, loss weights
├── get_trainer_class(task)  -> DetectionTrainer
├── DetectionTrainer(...)  [STAGE 2]      reads the evaluation and sample score thresholds
│   └── BaseTrainer.__init__()   (through CocoSplitTrainer, which reads the test manifest)
│       ├── get_device() / get_run_name() / DataAugmenter() / setup_training_paths()
│       ├── build_dataloaders() -> read_test_manifest() + get_detection_dataloaders()
│       │       CocoDetectionDataset() + build_native_loader(collate_function), decoding on the CPU workers
│       ├── check_loaders_give_batches() / save_data_record() / choose_sample_identifiers()
│       ├── build_model() -> TwoStageDetector.__init__()
│       │       build_backbone(), FeatureNeck(), RegionProposal() + Anchors per level, DetectionHead() per level,
│       │       _aggregate_all_anchors(), RegionProposalFilter()
│       ├── initialise_pretrained_components() / build_optimizer() / ExponentialMovingAverage() / MetricTracker()
│       ├── print_experiment_information() / print_summary() / export sizes
│       ├── diagnose_data_splits() / view_input_batches() / view_training_targets() / benchmark_speed()
│       │       diagnosis modes, then exit
│       ├── analyse_training_data() -> measure_anchor_recall()   anchor_recall.csv and .png, first launch
│       └── restore_last_model()
└── trainer.run_training_loop()  [STAGE 3]
    └── run_training_iterations()
        ├── save_input_samples()
        ├── for each training_iteration up to number_iterations (the total of the run):
        │   ├── get_next_batch() -> CocoDetectionDataset.__getitem__() + collate_function()
        │   ├── run_model_iteration()
        │   │   ├── prepare_batch() / augment_batch()   the geometric augmentations move the boxes
        │   │   ├── forward_pass() -> TwoStageDetector.forward() -> forward_detection_stages()
        │   │   │   ├── Backbone.forward() / FeatureNeck.forward()
        │   │   │   ├── _forward_region_proposals() -> RegionProposal.forward()
        │   │   │   ├── aggregate_proposals_and_anchors()
        │   │   │   ├── RegionProposalFilter.forward() -> select_best_proposals()
        │   │   │   ├── _prepare_detector_inputs()   add_ground_truth_proposals, assignment, sampling
        │   │   │   ├── _forward_detections() -> DetectionHead.forward()
        │   │   │   ├── aggregate_detection_outputs() + _rearrange_predictions_by_batch()
        │   │   │   ├── assign_region_proposal_targets() + compute_region_proposal_losses()
        │   │   │   └── compute_detection_losses()
        │   │   └── MetricTracker.accumulate() + log_to_tensorboard()
        │   ├── backward / step_optimizer / save_model / validation pass / interval summary
        │   └── run_test_evaluation() on the multiples of weight_saving_iterations   -> [STAGE 4]
        └── save_model() if steps remain unsaved, shutdown_loader_workers(), finish_experiment_tracking()

run_test_evaluation(iteration)  [STAGE 4]   once for the trained weights, again within use_averaged_weights() for EMA
├── for each test batch: predict_batch() -> postprocess_detections(); accumulate_evaluation_batch(); save_sample_images()
├── compute_evaluation_results() -> summarise_average_precisions() -> evaluate_detections(); save_proposal_recall(),
│       write_class_metrics_file(), save_error_analysis()
└── report_evaluation() -> report_evaluation_results() + track_evaluation() + update_evaluation_curves()

scripts/evaluation/evaluate_checkpoint.py  [STAGE 5]   evaluate_checkpoints() -> run_test_evaluation() per checkpoint
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below are real outputs of the loader on `configuration_synthetic_shapes.yaml` as it was then, redirected to the generated dataset.

### Step 1.1 — `main()`
The entry point is the same for every task (see `classification_workflow.md`, Step 1.1).

### Step 1.2 — `load_experiment_configuration()`
This table gives the model sections of the detection.

| Section | Content |
|---|---|
| `DetectionLevels` | keyed by the backbone blocks feeding the detection; each level holds its anchor scales and aspect ratios, its region proposal head and its detection head |
| `Neck` | the layers merging the levels before the heads (none with `type: "none"`) |
| `RegionProposal` | settings shared by the region proposal heads: IoU thresholds labelling the anchors, proposals kept before and after the NMS |
| `Detector` | settings shared by the detection heads and the inference post-processing |

The `evaluation` section adds `score_threshold` (Step 2.1), `target_precision` (Step 4.12) and `error_dumps` (Step 4.5).

**Example: one detection level** (shortened)
```text
DetectionLevels: {"5": {"anchors": {"scales": [32, 64, 128], "aspect_ratios": [0.5, 1.0, 2.0]}, "region_proposal": {"convolutions": [2, 64]},
                        "detector": {"roi_align_pool_size": 7, "convolutions": [2, 64], "fully_connected": [256, 128]}}, "6": ..., "7": ...}
```

### Step 1.3 — `load_configuration()`
**Example output**
```text
top-level keys:    ['ExperimentConfiguration', 'Optimization', 'Data', 'Backbone', 'DetectionLevels', 'RegionProposal', 'Detector']
number_iterations: '1e10'      <- still a string
```

### Step 1.4 — `build_label_dictionary()`
The mapping file gives the class of every category, from 1. The detector predicts the background as class 0, so `'background': 0` is added; `"binary"` mode gives `{background: 0, foreground: 1}`.

**Example: the label dictionary of the synthetic shapes**
```text
dict: {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 'background': 0}
```

### Step 1.5 — `check_configuration()`
After the common checks, `check_detection_configuration()` refuses `random_perspective` and `random_jigsaw`, since the boxes cannot follow them (`check_image_only_augmentation_types()`). It checks that classes run from 0 (background) to K-1 (`check_detection_data_configuration()`) and checks the four loss weights. Two checks keep the target assignment consistent, as this example shows.

**Example: `RegionProposal.background_iou_threshold.max: 0.6` and `Detector.total_training_samples: 600`**
```text
Invalid Configuration File : file:///.../configuration_wrong.yaml
  - 'RegionProposal.background_iou_threshold.max' (0.6) must be <= 'RegionProposal.foreground_iou_threshold.min' (0.5), otherwise anchors can be labelled both background and foreground.
  - 'Detector.total_training_samples' (600) must be <= 'RegionProposal.training.post_nms_proposals' (500), since the detector samples its targets among these proposals.
SystemExit code 1
```
Anchor scales are checked against the image size: at 256 pixels, this configuration is refused with `'DetectionLevels.7.anchors.scales' must be integers within [1, 256], got [128, 256, 512].`

### Step 1.6 — `get_trainer_class()`
**Example output**
```text
get_trainer_class(task="object_detection") -> <class 'vision_training.trainers.detection_trainer.DetectionTrainer'>
```

---

## Stage 2 — Trainer Initialization

> The examples below are real outputs of building the trainer on the redirected `configuration_synthetic_shapes.yaml` (1024 pixel images, batch size 40). The construction took 2.1 s on CPU.

### Step 2.1 — `DetectionTrainer.__init__()`
Before the shared constructor, the trainer reads `evaluation.score_threshold` (the lowest score kept for the mAP, low so the precision-recall curves keep their low-confidence end) and `Detector.inference.score_threshold` (the deployment one, for the sample images, the confusion matrix and the error images). An evaluation threshold above the deployment one is refused, since detections in between would count as missed. It also reads `target_precision` and `error_dumps`.

**Example: the attributes after the construction**
```text
self.evaluation_score_threshold = 0.05
self.deployment_score_threshold = 0.5
TASK_NAME = 'object_detection'   SAMPLE_NAME = 'Images'   COUNTS_FRAMES = False
```

### Step 2.2 — `BaseTrainer.__init__()`
As for every task (`classification_workflow.md`, Step 2.2): `input_image_size = 1024`, `batch_size = 40`.

### Step 2.3 — `get_device()`
It prints `For Training We Will Be Using device: cpu` (forced here; a normal run on this machine prints `cuda`).

### Step 2.4 — `get_run_name()` → `build_run_name()`
The input and backbone tokens are those of every image task. This table gives the tokens `build_detection_sections()` adds.

| Token | Meaning |
|---|---|
| `L567` | the detection blocks |
| `A9.32-512.598295` | 9 anchors per position, scales 32 to 512, then the first 6 characters of the SHA-256 of the scales and ratios of every level |
| `RBN-R2x64` | region proposal heads: batch normalization, 2 layers of 64 filters |
| `DSE-DRES-DBN-D7.2x64.256.128` | detection heads: squeeze-and-excitation, residual, batch normalization, RoIAlign 7, 2 layers of 64, fully connected 256 and 128 |
| `MC9` | 9 foreground classes (`BIN` in binary mode) |
| `-NFPN<channels><normalization>`, `-NPAN...` | after the levels token, with a neck (`L567-NFPN32BN` in the current synthetic configuration) |
| `RPL` | after the region proposal heads, with `proposal_selection: "per_level"` |
| `DCIOU` | after the detection heads, with `Detector.box_loss: "complete_iou"` |

A setting that differs between levels is written for every level, joined with `+`.

**Example output**
```text
DET_IN-KAR-1024_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512.598295_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128_MC9
```

### Step 2.5 — `DataAugmenter()`
The augmenter of the classification. Geometric augmentations move the boxes and drop those leaving the image (Step 3.8). This configuration turns on `random_haze` only (probability 0.4), a photometric type that never moves the boxes.

### Step 2.6 — `setup_training_paths()`
As for every task (`classification_workflow.md`, Step 2.6). `label_dictionary.json` includes the background. `data.json` holds the classes and images of every split.

**Example: the run folder after this step**
```text
DET_IN-KAR-1024_..._MC9/
├── Tensorboard/
├── Weights/
├── label_dictionary.json
└── training_configuration.yaml    5,613 bytes
```

### Step 2.7 — `build_dataloaders()`
The hook first calls `read_test_manifest()` of `CocoSplitTrainer` (shared with the keypoint detection). It reads from the test `labels.json` the `class_names`, the `class_colours` (one BGR colour per class, for the sample boxes) and the file name of every test image. Then it builds the loaders (Step 2.8); the validation one only when the run reads it (`classification_workflow.md`, Step 2.7).

**Example output**
```text
class_names = {0: 'background', 1: 'square', 2: 'circle', 3: 'triangle', 4: 'rectangle', 5: 'pentagon', 6: 'hexagon',
               7: 'plus_sign', 8: 'star', 9: 'octagon'}
test_image_file_names: 20 entries, e.g. {1: 'test_000000.png', 2: 'test_000001.png', 3: 'test_000002.png'}
```

### Step 2.8 — `get_detection_dataloaders()` → `get_detection_dataloader()`
Every split holds its images in `<folder>/data/` and its COCO manifest in `<folder>/labels.json`. A `CocoDetectionDataset` (Step 2.9) decodes and resizes on the CPU loader workers. `build_native_loader()` wraps it with `collate_function()`, which stacks the images and keeps boxes and labels as lists. `load_original_instances` and `load_instance_index_maps` are only used by instance segmentation (`instance_segmentation_workflow.md`); detection loads no mask.

**Example: size of the three splits**
```text
train       100 images -> 3 batches of 40 (one epoch = 3 iterations)
validation   20 images -> 1 batch of 40
test         20 images -> 1 batch of 40
```

### Step 2.9 — `CocoDetectionDataset.__init__()` (and `__getitem__()`)
The constructor reads the manifest (`read_manifest()`, parsed once per launch) and shuffles its images once. It packs every image with its annotations into `SerializedRecords` (a read-only NumPy byte buffer, unpacked by `get_image_record()`) and frees the parsed manifest (`release_freed_heap_memory()`), so forked workers copy no Python objects: on 50,000 images, 0.31 GB instead of 0.67 GB, and 0.05 GB copied per worker instead of 0.64 GB. Images sharing an `id` raise a `ValueError`. Categories absent from the label mapping are printed in yellow.

`__getitem__()` reads the image (`read_image()`, OpenCV, BGR, `uint8`), keeps the boxes whose category has a class (`select_image_objects()`), turns COCO `[x, y, width, height]` into `[x_min, y_min, x_max, y_max]` and resizes image and boxes together (`preprocess_image_and_boxes()`, letterbox with `keep_ratio`). Boxes are clipped and filtered only after a geometric augmentation.

**Example: the first image after shuffling, and its sample**
```text
first image:  {'id': 49, 'file_name': 'train_000048.png', 'width': 512, 'height': 512}
              bbox [[109, 169, 86, 58], [427, 235, 32, 70]]   category_id [3, 4]
dataset[0]:   images (3, 1024, 1024) torch.uint8
              bounding_boxes (2, 4) [[218.0, 338.0, 390.0, 454.0], [854.0, 470.0, 918.0, 610.0]]   <- x 2, from 512 to 1024 pixels
              labels (2,) [3, 4]
```

### Step 2.10 — `build_native_loader()`
The PyTorch `DataLoader` of the classification (`classification_workflow.md`, Step 2.10). The example was recorded with 8 workers and pinned memory; every loader now has 1 worker and pins only on a discrete GPU.

### Step 2.11 — `get_test_sample_classes()` + `choose_sample_identifiers()`
Every test image gets an empty class list, so `choose_sample_identifiers()` takes `number_sample_images` images evenly spread over the sorted COCO identifiers. They are named after their file (`get_test_sample_name()`).

**Example output** (8 images among 20)
```text
sample_image_identifiers  -> [1, 4, 6, 9, 12, 15, 17, 20]
```

### Step 2.12 — `build_model()` → `TwoStageDetector.__init__()`
The detector holds the backbone (Step 2.13), the neck (`FeatureNeck`, no layer for `none`), a region proposal head per level (Step 2.14), the proposal filter (Step 2.16) and a detection head per level (Step 2.15). It reads the inference settings: `Detector.inference.score_threshold`, `maximum_detections`, `Detector.nms_iou_threshold`, `nms_criterion` and `nms_distance_exponent`.

**Example: the components of the model**
```text
model.backbone                    Backbone                 2,245,216 parameters
model.region_proposer_dictionary  ModuleDict                 495,239 parameters      levels 5, 6, 7: 113,965 + 190,637 + 190,637
model.detector_heads_dictionary   ModuleDict               2,847,894 parameters      levels 5, 6, 7: 924,722 + 961,586 + 961,586
model.number_classes = 10                                                             <- the background included
```

### Step 2.13 — `build_backbone()` + `compute_detection_input_information()`
The shared backbone (`classification_workflow.md`, Step 2.14). The detector takes the feature maps of the blocks keyed in `DetectionLevels`. `compute_detection_input_information()` computes their size and channels once, from a zero image in evaluation mode without gradients, so batch normalization statistics stay as built.

It gives block 5 a 32 x 32 map of 64 channels, blocks 6 and 7 16 x 16 maps of 128 channels (block 7 has no max pooling, `last_max_pooling: False`).

### Step 2.14 — `RegionProposal.__init__()` → `Anchors.generate_anchors()`
Every position of a level's feature map gets one anchor per scale and aspect ratio. The ratio is height over width (0.5 wide, 2.0 tall) at the same area. Anchors are centred on the cell corner, `(column x stride, row x stride)`, as torchvision. The head is a block of 3x3 convolutions (`DetectionLevels.<level>.region_proposal.convolutions`), a 3x3 convolution and ReLU, then two 1x1 convolutions: one objectness logit and 4 box offsets per anchor. `_aggregate_all_anchors()` concatenates the anchors of every level into `model.anchors`.

**Example: the anchors of the three levels, and those of the first position of level 5** (1024 pixels)
```text
level 5: feature map 32, stride 32, anchors (9216, 4)     levels 6 and 7: feature map 16, stride 64, anchors (2304, 4)
model.anchors (13824, 4)
level 5, position 0:  scale 32: [-22.6, -11.3, 22.6, 11.3]  [-16.0, -16.0, 16.0, 16.0]  [-11.3, -22.6, 11.3, 22.6]     (ratios 0.5, 1.0, 2.0)
                      ...
```

### Step 2.15 — `DetectionHead.__init__()`
`roi_align()` cuts a 7x7 map out of the level for every proposal (`spatial_scale` = feature map size / image size, `sampling_ratio=0`, `aligned=True`). A block of 3x3 convolutions with the modules of `Detector.modules` refines it. Fully connected layers without normalization flatten it. Two linear layers give the logit of every class and 4 box offsets per class.

**Example: the head of level 5** (shortened)
```text
DetectionHead(
  (convolutional_block): ConvolutionBlock(   2 x [Conv2d(64, 64, 3x3), BatchNorm2d(64), ReLU], 2 x SqueezeAndExcitation(64 -> 32 -> 64) )
  (fully_connected_block): FullyConnectedBlock(Linear(3136, 256), ReLU, Linear(256, 128), ReLU)   <- 3136 = 7 x 7 x 64
  (classifier): Linear(in_features=128, out_features=10, bias=True)
  (bounding_box_regressor): Linear(in_features=128, out_features=40, bias=True)               <- 4 offsets for each of the 10 classes
)
```
The heads of levels 6 and 7 read 128 channels, hence 36,864 more parameters. Their first convolution changes the channel count, so they add no residual shortcut (see Pitfalls).

### Step 2.16 — `RegionProposalFilter.__init__()`
The filter keeps the `pre_nms_proposals` best scored proposals, then at most `post_nms_proposals` surviving the NMS at `nms_iou_threshold`, with different numbers in training and inference. It gets the anchor count of every level and `proposal_selection` (`"joint"` in this run).

**Example output**
```text
training 1000 -> 500   inference 500 -> 250   nms_iou_threshold 0.5
```

### Step 2.17 — `initialise_pretrained_components()` + `build_optimizer()` + `get_tracked_metrics()`
The backbone, region proposal heads and detection heads can come from other runs. This table gives the two modes of `Detector.pretrained.class_layers`.

| `class_layers` | Effect |
|---|---|
| `False` (default) | the classifier and box regressor keep their random rows (`keep_target_class_layers()`); the run name adds `DRCL` |
| `True` | every source class must be a class of this run (first classes in the same order, background first, or anywhere under the same name); `extend_class_layers()` copies the classification row and the 4 box rows of every source class, the other classes staying random |

The mask heads of an instance segmentation run do the same with `MaskHead.pretrained.class_layers: True`. With `Optimization.exponential_moving_average.activated`, an `ExponentialMovingAverage` copy is made once the transfers are loaded. The four losses are tracked with their weighted sum.

The current code also transfers the neck (`Neck.pretrained`, last in `TRANSFERABLE_COMPONENTS`), so the dictionary below ends with `'Neck': None`. Heads move only between runs of the same `Neck.type` and `output_channels` (`compare_detection_heads()`). Heads loaded behind a neck that does not come from their checkpoint print a warning (`warn_heads_behind_another_neck()`).

**Example output**
```text
pretrained_checkpoint_paths = {'Backbone': None, 'RegionProposal': None, 'Detector': None}
Adam, 1 parameter group, base_learning_rates [0.0002], weight decay 0.0
{'total_loss': 'Total Loss', 'region_proposal_classification_loss': 'RPN Classification Loss',
 'region_proposal_localisation_loss': 'RPN Regression Loss', 'detection_classification_loss': 'Detection Classification Loss',
 'detection_regression_loss': 'Detection Regression Loss'}
```

### Step 2.18 — `print_experiment_information()` (→ `describe_split()` → `count_split_images_and_annotations()`)
Every split line gives its images and annotations, such as `train (100 images, 322 annotations)`.

### Step 2.19 — `print_summary()` / `print_receptive_field_summary()` / `print_model_size_summary()`
The detector adds a neck panel, a region proposal panel (anchors per level) and a detection panel (heads per level). The receptive field panel marks the anchor scales of every detection block. The Export Sizes panel follows the model size panel.

**Example output** (key lines, rebuilt with the current code)
```text
Region Proposal Architecture:  Convolution Block 5, Feature Map Size [32, 32], Total Anchors 9216 ... Total Network Anchors 13824
Detection Architecture:        Detection Block 5, Modules Residual Shortcuts, Squeeze-and-Excitation, Spatial Scale 0.0312
                               Detection Block 6, Modules Squeeze-and-Excitation
Backbone Receptive Field:      Block 5 - Max Pooling : Output 32x32x64 | Stride 32 | Receptive Field 250, Anchor Scales [32, 64, 128]
Model Size:                    Total Model 5,588,349 parameters, 22.38 MB
```

### Step 2.20 — `view_input_batches()` / `view_training_targets()` / `restore_last_model()`
With `view_input_images`, `show_input_samples()` shows an augmented training image next to a validation image, boxes captioned with their class, with a panel giving the object count, a `Class` / `Count` table and the augmentations and removed objects (`Removed objects : 1 (dog)`). A new run starts from scratch (`start_iteration = 1`).

Should the losses need checking, `view_training_targets` shows what they train on, then exits without a run folder:
- `view_training_targets()` loads the last checkpoint of the run, if any (`load_last_checkpoint_weights()`), and prepares and augments every training batch as Steps 3.7 and 3.8 do;
- `compute_training_targets()` runs `forward_detection_stages()` in training mode with the ground truth, without gradient, and keeps its `training_targets`: the anchor labels and samples of Step 3.17, the filtered proposals of Step 3.13 and the RoIs sampled in Step 3.14;
- `build_training_target_windows()` draws one window per image with `build_training_target_grid()` (sampled positive anchors, 20 of the negatives, the best 20 proposals, the sampled RoIs with the ground truth copies dashed) and a panel counting them per object. `view_input_data()` shows the windows like the input viewer.

### Step 2.21 — `analyse_training_data()` → anchor recall
Called by `BaseTrainer.__init__()` between `view_input_batches()` and `restore_last_model()`, never in a diagnosis mode that exits. At the first launch of a run, it measures how many training and test objects the anchors can learn, through `measure_anchor_recall()` on each split:
- `gather_object_boxes()` reads the boxes of the split from the annotations (`select_image_objects()`), placed in network input pixels by `preprocess_boxes()`, without images or augmentations;
- `compute_best_anchor_overlaps()` gives the best anchor overlap of every object in every level (`TwoStageDetector.get_level_anchor_generators()`). `Anchors.compute_best_overlaps()` takes the grid anchor nearest the object centre for every anchor shape: the exact best, in float64 on the CPU;
- `summarise_anchor_recall()` counts the objects whose best anchor reaches `RegionProposal.foreground_iou_threshold.min`, per level and over all levels, for all the objects and the five sizes relative to the network input (Step 4.14).

An object under the threshold everywhere only gets its single best anchor as foreground. The console prints one line per split (`describe_anchor_recall()`): the recall over all levels, and that of the smallest size holding objects. `anchor_recall.csv` holds the rows of both splits, with their `split_folder` and `iou_threshold`, the recalls between 0 and 1. `draw_anchor_recall_figure()` draws both tables into `anchor_recall.png`, the recalls shaded from red to green.

A later launch reads the `split_folder` and `iou_threshold` columns (`read_anchor_recall_settings()`). With the same folders and threshold, it keeps both files and prints their link. Otherwise it measures both splits again. Only written files go to MLflow (Step 3.3), as `anchor_recall_resume_<n>.csv` when a resume measured them again.

With `diagnosis_modes.compute_anchor_coverage: { activated: True, iou_threshold: <threshold> }`, `diagnose_data_splits()` runs it on all three splits at `iou_threshold`, prints the three tables with their recalls coloured from red to green, and exits without a run folder.

**Example output: the first launch** (the console, then the training table of `anchor_recall.png`)
```text
Anchor Recall Of The 703 879 Training Objects: 64.04 % In All Levels, 8.11 % For The Smallest Objects (Under 1/16)
Anchor Recall Of The 109 928 Test Objects: 64.24 % In All Levels, 8.52 % For The Smallest Objects (Under 1/16)
Anchor recall of every object size: <run>/anchor_recall.png

Anchor Recall Of The Training Objects
 Object Side         Pixels             Area    Count   Block 5   Block 6   Block 7   All Levels
─────────────────────────────────────────────────────────────────────────────────────────────────
 All Objects                                   703879   22.89 %   25.76 %   41.15 %      64.04 %
 Under 1/16            < 32         < 0.39 %   150120    8.11 %    0.00 %    0.00 %       8.11 %
 1/16 to 1/8       32 to 64   0.39 to 1.56 %   143890   51.12 %    9.57 %    0.00 %      51.12 %
 1/8 to 1/4       64 to 128   1.56 to 6.25 %   174260   43.28 %   50.38 %   38.84 %      82.12 %
 1/4 to 1/2      128 to 256     6.25 to 25 %   178644    0.00 %   44.64 %   94.14 %      94.14 %
 1/2 and above       >= 256          >= 25 %    56965    0.00 %    0.00 %   94.41 %      94.41 %
```
Block 5 (anchors of 32 and 64 pixels) covers half of the objects from 1/16 to 1/8, but 8.11 % of those under 1/16. Blocks 6 and 7 cover 94 % of the objects from 1/4 up. The test split gives close shares. A second launch prints `Anchor recall measured at the first launch of the run: <run>/anchor_recall.png` instead.

---

## Stage 3 — Training Loop

> Steps 3.1 to 3.5, 3.19, 3.22 and 3.23 show the 5000-iteration run of [`run_configurations/object_detection_workflow_run.yaml`](run_configurations/object_detection_workflow_run.yaml), trained on the GPU on 2026-10-24. Its run name is `DET_IN-KAR-512_SE-RES-BN-B1x16.2x16.2x16.3x32.3x32.5x32.5x32_L567_A6.32-256.e7b72c_RBN-R2x32_DSE-DRES-DBN-D7.2x32.64.32_MC20`. An epoch is 2000 iterations, a summary is printed every 1000 (`information_dump`) and no validation batch runs. The run took 8 h 11 min on a shared GPU, so its throughput is not a speed reference. The current configuration adds `-NFPN32BN` after `L567` and `RPL` to the run name.
>
> Steps 3.6 to 3.18, 3.20 and 3.21 show the CPU run (512 pixels, batch size 2, 12 iterations). Its run name is `DET_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512.598295_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128_MC9`. Its detection maps are 16x16, 8x8 and 8x8, so it has 2304 + 576 + 576 = 3456 anchors.

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
The loop of every task (`classification_workflow.md`, Step 3.1). The table below shows what happens at each iteration of the 5000-iteration run, after its training step (the iterations between the columns only train).

| Iteration | 250 | 500 | 750 | 1000 | 1250 | 1500 | ... | 4500 | 4750 | 5000 | after the loop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Checkpoint saved | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ... | ✔ | ✔ | ✔ | nothing: iteration 5000 is saved |
| Validation batch | | | | | | | | | | | |
| Console summary | | | | ✔ | | | ... | | | ✔ | |
| Test evaluation, trained then averaged weights | | ✔ | | ✔ | | ✔ | ... | ✔ | | ✔ | nothing |

### Step 3.2 — `setup_tensorboard_writers()` / `initialize_training_loop_state()`
The run writes `Tensorboard/Train/` (the five losses and the learning rate), `Tensorboard/Test/` and `Tensorboard/Test_EMA/` (Step 4.13), and no `Validation` writer.

### Step 3.3 — `start_experiment_tracking()`
**Example output**
```text
MLflow Run Created In The Experiment 'object_detection' : 8e53387f20e24434973626e4c0394a2c (Browse : mlflow ui --backend-store-uri sqlite:///.../experiments/workflow_runs/mlflow.db)
```
A background thread creates the MLflow run, then sends the configuration, `anchor_recall.csv` (folder `analysis`), `data.json` and `Input_Samples/`. The current code also sends `anchor_recall.png`, and only when this launch wrote the anchor recall.

### Step 3.4 — `save_model()` → `save_checkpoint()`
Saved after the training step of every multiple of 250, so the checkpoint of iteration N holds N steps (`classification_workflow.md`, Step 3.6). It holds the label dictionary, the source run of every transferable component (`pretrained_backbone`, `pretrained_region_proposer_dictionary`, `pretrained_detector_heads_dictionary`, None without transfer) and, with a moving average, `exponential_moving_average_state`.

**Example: the checkpoints and the keys of the last one**
```text
Weights/Iteration_250/model_000250.pt ... Weights/Iteration_5000/model_005000.pt   10,708,317 bytes each
['iteration', 'model_state', 'optimizer_state', 'format_version', 'task', 'run_name', 'label_dictionary',
 'mlflow_run_id', 'pretrained_backbone', 'pretrained_region_proposer_dictionary', 'pretrained_detector_heads_dictionary',
 'exponential_moving_average_state']
exponential_moving_average_state: {'model_state', 'number_updates'}, number_updates 5000
```
Current checkpoints also hold `pretrained_backbone_blocks` (`classification_workflow.md`, Step 2.19) and `pretrained_neck`, both `None` here.

### Step 3.5 — `run_test_evaluation()`
At every multiple of 500, last in the iteration (`classification_workflow.md`, Step 3.7), once for the trained weights and once for their moving average. See Stage 4.

### Step 3.6 — `get_next_batch()` → `CocoDetectionDataset.__getitem__()` + `collate_function()`
A batch that fails to load is dropped and the next one taken (`classification_workflow.md`, Step 3.8). Images are stacked; boxes and labels stay lists, one tensor per image.

**Example: the first batch of the CPU run**
```text
images               [2, 3, 512, 512] torch.uint8
bounding_boxes       list of 2 tensors   [[3.0, 216.0, 164.0, 377.0], [110.0, 60.0, 249.0, 199.0]]
                                         [[242.0, 249.0, 408.0, 415.0], [172.0, 120.0, 272.0, 220.0], [15.0, 12.0, 57.0, 52.0], [399.0, 55.0, 500.0, 156.0]]
labels               list of 2 tensors   [7, 1]   [1, 1, 5, 1]
```

### Step 3.7 — `run_model_iteration()` → `prepare_batch()`
It turns the images from `uint8` to `float32` and moves boxes (`float32`) and labels (`int64`) to the device.

### Step 3.8 — `augment_batch()` → `DataAugmenter.augment_batch(bounding_boxes=..., labels=...)`
Geometric augmentations move the boxes with the image. `clean_bounding_boxes()` clips them and drops, with their labels, boxes narrower or shorter than 1 pixel or with less than 25 % of their moved area inside the image (`MINIMUM_VISIBLE_AREA_FRACTION`). This configuration turns every geometric type off; in the first batch no image was changed.

### Step 3.9 — `forward_pass()` → `TwoStageDetector.forward(ground_truth_bounding_boxes=..., ground_truth_labels=...)`
With the ground truth, the detector samples its training targets and returns its four losses (Steps 3.10 to 3.18). The total loss is their sum weighted by `Optimization.loss_weights` (`get_loss_weights()`, 1 in the shipped configurations).

**Example: the dictionary returned at iteration 1 of the CPU run**
```text
classification_scores                 [2, 128, 10] torch.float32       <- the 128 proposals sampled for every image
box_regressions                       [2, 128, 10, 4] torch.float32
region_proposal_classification_loss   0.7244          <- close to ln 2 = 0.6931
region_proposal_localisation_loss     0.8107
detection_classification_loss         2.3117          <- close to ln 10 = 2.3026
detection_regression_loss             0.6198
total_loss                            4.4665
```

> Steps 3.10 to 3.18 run the first batch of the CPU run through the model at iteration 12, in training mode.

### Step 3.10 — `Backbone.forward()`
The backbone returns its final map and the maps of the detection blocks: `{'5': (2, 64, 16, 16), '6': (2, 128, 8, 8), '7': (2, 128, 8, 8)}`.

### Step 3.11 — `_forward_region_proposals()` → `RegionProposal.forward()`
Every level gives an objectness logit and 4 offsets `[dx, dy, dw, dh]` per anchor. `apply_regression_predictions()` moves each anchor into a proposal box: the centre by `dx` times the anchor width and `dy` times its height, the size times `exp(dw)` and `exp(dh)` (clamped so the exponential stays finite). The offsets are detached for the boxes, so proposals do not back-propagate.

**Example output**
```text
level 5: classification_scores (2, 2304)   proposal_boxes (2, 2304, 4)    <- 16 x 16 x 9     levels 6 and 7: (2, 576)
```

### Step 3.12 — `aggregate_proposals_and_anchors()`
The level outputs are concatenated, with the level of every anchor (`convolution_block_origin`) and the anchors repeated per image.

**Example output**
```text
classification_scores (2, 3456)   proposal_boxes (2, 3456, 4)   convolution_block_origin (2, 3456)   anchors (2, 3456, 4)
objectness of image 0 (sigmoid): min 0.4179   max 0.5723   mean 0.4970
```

### Step 3.13 — `RegionProposalFilter.forward()`
`select_best_proposals()`, shared with the deployed `InferenceDetector`, runs on the whole batch:
1. the sigmoid of the objectness;
2. the `pre_nms_proposals` best of every image (1000 in training), boxes clamped to the image;
3. the NMS of every image at `RegionProposal.nms_iou_threshold` (0.5 in this run, 0.7 now);
4. the `post_nms_proposals` best survivors (500 in training, 250 in inference).

With `"joint"`, the top-k and NMS run over all levels together, so the many small anchors of block 5 take most slots. `"per_level"` runs them per level, leaves out boxes clamped narrower than 1e-3 pixel, then keeps the best over all levels. The NMS is the only step run image by image (`suppress_overlapping_proposals()`, one torchvision `nms` call per image). The exported detector runs one call over the batch (`suppress_stacked_groups()`), keeping the same proposals. An image with fewer survivors is padded with copies of random survivors. The filter also returns the mask of real survivors, read by the proposal recall (Step 4.15).

**Example output**
```text
filtered: boxes (2, 500, 4)   scores (2, 500)   origins (2, 500)
image 0: 500 distinct boxes, levels {5: 484, 6: 6, 7: 10}
image 1: 500 distinct boxes, levels {5: 466, 6: 22, 7: 12}
```

### Step 3.14 — `_prepare_detector_inputs()`
In training (and in the validation loss), these functions run in turn:
- `add_ground_truth_proposals()` writes the ground truth boxes into the last slots of the filtered proposals (padding copies first), each at the level of its best anchor. Every object then has a positive proposal with a zero regression target;
- `batch_assign_targets_to_proposals()` gives every proposal the class of its best ground truth box when their IoU reaches `Detector.foreground_iou_threshold` (0.5), the background (0) otherwise;
- `turn_boxes_to_transformation_targets()` computes the offsets onto the ground truth box;
- `sample_positive_and_negative_training_targets()` keeps per image at most `number_training_positives` (64) foreground proposals, filled to `total_training_samples` (128) with random background ones.

The kept proposals are written `[image index, x_min, y_min, x_max, y_max]` for `roi_align()`. In inference, every filtered proposal is kept.

`add_ground_truth_proposals()` also returns the slots it wrote, so that the training target viewer dashes the ground truth copies.

**Example output** (recorded before `add_ground_truth_proposals()`: the positives are the region proposal ones alone)
```text
proposals (256, 5)   origins (256,)   labels (256,)   regression targets (256, 4)
image 0: sampled labels {0: 125, 1: 1, 7: 2}       <- 3 foreground proposals: 1 square, 2 plus_sign
image 1: sampled labels {0: 124, 1: 3, 5: 1}
```

### Step 3.15 — `_forward_detections()` → `DetectionHead.forward()`
Every proposal goes to the head of its level: `roi_align()` cuts its 7x7 map, the head gives a logit and offsets per class, applied to give one box per class (`detection_boxes`).

In the example, levels 5, 6 and 7 receive 244, 9 and 3 of the 256 sampled proposals (`detection_boxes (244, 10, 4)` on level 5).

### Step 3.16 — `aggregate_detection_outputs()` + `_rearrange_predictions_by_batch()`
The level outputs are concatenated, then grouped by image.

**Example output**
```text
by image:   classification_scores (2, 128, 10)   box_regressions (2, 128, 10, 4)   detection_boxes (2, 128, 10, 4)
```

### Step 3.17 — `assign_region_proposal_targets()` + `compute_region_proposal_losses()`
`assign_region_proposal_targets()` gives the anchor targets, read by the losses and by the training target viewer (Step 2.20). `batch_assign_targets_to_anchors()` labels every anchor of the batch at once from its best IoU with a ground truth box (`pad_ground_truths()` pads the ground truths to the largest object count). This table gives the labels.

| IoU with the best ground truth box | Label |
|---|---|
| from `RegionProposal.foreground_iou_threshold.min` (0.5; `.max` unused) | foreground |
| between `background_iou_threshold.min` and `.max` (0.1 to 0.3 in this run) | background |
| otherwise (0.3 to 0.5, and below 0.1) | ignored |
| the best anchor of a ground truth box (IoU above 0) | foreground, whatever its IoU |

The object detection configurations keep the minimum of 0.1; the instance segmentation ones set 0.0 (audit H1). A small box sharing its best anchor with a larger one gets none. `sample_positive_and_negative_training_targets()` keeps per image at most `number_training_positives` (128) foreground anchors, filled to `total_training_samples` (256) with background ones. `compute_region_proposal_losses()` then computes both losses. The classification loss is the binary cross entropy of the sampled objectness. The regression loss is the smooth L1 (beta 1/9, `LOCALISATION_LOSS_BETA`) on the offsets of the sampled foreground anchors, summed and divided by their number (`REGION_PROPOSAL_BOX_LOSS`).

**Example: the anchors of the first batch**
```text
image 0: foreground 24   background 504   ignored 2928
image 1: foreground 28   background 696   ignored 2732
sampled: positives [24, 28]   negatives [232, 228]
foreground anchor 626 of image 0: anchor [96.0, 64.0, 224.0, 192.0]   ground truth [110.0, 60.0, 249.0, 199.0]   IoU 0.6911
                                  target offsets [0.1523, 0.0117, 0.0824, 0.0824]   predicted [0.0619, 0.0423, 0.064, 0.038]
losses: classification 0.6702   regression 0.4774
```

### Step 3.18 — `compute_detection_losses()`
The classification loss is the cross entropy of the sampled proposals, background included. The regression loss is the smooth L1 (beta 1/9) on the offsets of the true class of every foreground proposal, summed and divided by their number (0 with none). With `Detector.box_loss: "complete_iou"` it is 1 - CIoU of the decoded boxes. In the example: classification 1.0863, regression 1.4856.

### Step 3.19 — `compute_batch_metrics()` + `MetricTracker.accumulate()` + `log_to_tensorboard()`
Detection has no batch metric (`{}`): the four unweighted losses and their weighted sum are tracked.

**Example: the Train curves** (a few points)
```text
Total Loss                     1: 4.7885, 1000: 0.8791, 2000: 0.6425, 3000: 0.4555, 4000: 0.3572, 5000: 0.3252
RPN Classification Loss        1: 0.6982, 1000: 0.0368, 2000: 0.0167, 3000: 0.0108, 4000: 0.0065, 5000: 0.0056
Detection Classification Loss  1: 2.9319, 1000: 0.6279, 2000: 0.4690, 3000: 0.3291, 4000: 0.2349, 5000: 0.2072
Detection Regression Loss      1: 0.4050, 1000: 0.1307, 2000: 0.0898, 3000: 0.0727, 4000: 0.0752, 5000: 0.0717
```
At iteration 1 the losses are those of untrained logits (ln 2 = 0.6931, ln 21 = 3.0445). The region proposal loss falls below 0.04 within 1000 iterations; the detection classification loss still falls at 5000.

### Step 3.20 — `training_loss.backward()` + `step_optimizer()`
**Example output: the CPU run** (gradients applied as they are)
```text
gradient norm: iteration 1: 42.3099   iteration 2: 40.7255   iteration 3: 33.9392        learning rate 0.0002 (constant)
```

### Step 3.21 — validation pass
In the CPU run, at iteration 10, a validation batch runs in `eval()` mode with its ground truth: the filter keeps the inference numbers of proposals, the ground truth boxes are added (Step 3.14), and the detector returns its four losses.

**Example output: the CPU run**
```text
Tensorboard/Validation at iteration 10: Total Loss 3.4132, RPN Classification Loss 0.6929, RPN Regression Loss 0.5775,
                                        Detection Classification Loss 2.1428, Detection Regression Loss 0.0000
```
The regression loss was 0 because no sampled proposal reached IoU 0.5. With the ground truth proposals of the current code, it is no longer 0.

### Step 3.22 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()`
**Example output: iteration 5000**
```text
|   Iteration: 5 000  |   Total Images: 320 000    |  Interval Images: 64 000  |   Throughput: 34.5 Images Per Second  |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :     0.3332    -0.0633 Best
...
Moving Average of Train Detection Regression Loss      :     0.0691    -0.0067 Best
------------------------------------------------------------------------------------------------------------------------
Interval Of 1 000 Iterations Took 00:30:53.
```
Without validation, the summaries have no `Valid` rows. The interval time includes the evaluations in between and the load of the shared GPU (01:44:34 for the first interval).

### Step 3.23 — end of the loop
Iteration 5000 is already saved and evaluated (its EMA evaluation last), so the loop saves and evaluates nothing more (`classification_workflow.md`, Step 3.25). It prints `Last Checkpoint Saved At Iteration 5000. Exiting Training.`

---

## Stage 4 — Test Evaluation of a Checkpoint

> Steps 4.1, 4.5 to 4.7 and 4.10 to 4.15 show the evaluations of the 5000-iteration run, mostly iteration 5000, on its whole test split (20,000 images, 69,682 boxes), error images off. Every multiple of 500 is evaluated with the trained weights, then their moving average (`EMA Weights` blocks, `EMA/` folders). These outputs predate the recall columns and `Per Class Recall` table of Steps 4.11 and 4.13.
>
> Steps 4.2 to 4.4, 4.8 and 4.9 show iteration 12 of the CPU run (20 test images, 57 boxes), whose detector finds no shape (its 10 class probabilities are all close to 0.1). The error images of Step 4.5 come from the detection smoke test.

### Step 4.1 — `run_test_evaluation()` → `evaluation_context()`
The evaluation loop of every task (`classification_workflow.md`, Step 4.1). `evaluation_context()` lowers the model's score threshold from the deployment one (0.5) to `evaluation.score_threshold` (0.05) during the evaluation, and restores it afterwards, even on failure.

**Example output: the progress bars** (20,000 images, batch size 64)
```text
Test Evaluation Iteration 500: 100%|██████████| 313/313 [21:32<00:00,  4.13s/it]
Test Evaluation Iteration 5000: 100%|██████████| 313/313 [01:26<00:00,  3.62it/s]
Test Evaluation Iteration 5000 (EMA Weights): 100%|██████████| 313/313 [04:46<00:00,  1.09it/s]
```
The same batches took 21:32 and 01:26 on a GPU shared with other runs.

### Step 4.2 — `create_evaluation_accumulator()`
It holds the detections and ground truth of every test image (on the CPU, in network input pixels), the confusion matrix (background included, Step 4.5), the error image folder and counts, and the `proposal_recall` accumulator (Step 4.15, `create_proposal_recall_accumulator()`, added after the example).

**Example: the accumulator after iteration 12 of the CPU run**
```text
{'detections': 20 entries {'boxes': (100, 4), 'scores': (100,), 'labels': (100,)}, 'ground_truths': 20 entries {'boxes', 'labels'},
 'confusion_matrix': (10, 10) int64, 'errors_folder': .../Evaluation/Iteration_12/Errors, 'error_counts': {}}
first image: best scores [0.1091, 0.1091, 0.1091, 0.1091, 0.1091]   labels [9, 9, 9, 9, 9]   ground truth label [5]
```

### Step 4.3 — `prepare_batch()` + `predict_batch()`
`predict_batch()` runs the detector without ground truth, as in deployment; with it, the detector would only classify proposals sampled around the true boxes, inflating the mAP. In `eval()` mode the filter keeps 250 proposals per image. The outputs go through `postprocess_detections()` (Step 4.4) with the valid proposal mask (`sliced_valid_mask`), so padding copies never become detections, even at an NMS threshold of 1.0 or for empty boxes. The outputs also keep `proposal_boxes`, `proposal_levels` and `valid_proposal_mask` for the proposal recall (Step 4.15).

**Example: a test batch of the CPU run**
```text
classification_scores    [2, 250, 10]       detection_boxes          [2, 250, 10, 4]
final_detection_boxes    [2, 100, 4]        final_detection_scores   [2, 100]
final_detection_labels   [2, 100]           number_detections        [2] [100, 100]
```

### Step 4.4 — `postprocess_detections()`
These steps turn the head outputs into the final detections:
1. the softmax of the class logits, background left out, gives one candidate per proposal and foreground class, its box clamped to the image;
2. one `batched_nms()` call suppresses, per image and class, boxes overlapping a better one by more than `Detector.nms_iou_threshold` (0.8 in the CPU run, 0.5 now);
3. the survivors reaching the score threshold are kept, the `maximum_detections` (100) best per image returned, empty places padded with score 0, label 0 and an empty box.

This table gives the two NMS paths.

| Path | Suppression |
|---|---|
| Eager training and evaluation | only the candidates reaching the score threshold (`suppress_candidates_only=True`): the same detections, 0.02 seconds per batch of 64 images instead of 7 |
| `InferenceDetector` and exports | every candidate (`suppress_candidates_only=False`), static shapes for TorchScript and ONNX. `suppress_stacked_groups()` stacks every image and class as one row: TorchScript runs one `nms` call on rows moved apart (`suppress_groups_apart()`), ONNX one `NonMaxSuppression` without offsets (TensorRT FP16 would overflow past 65,504). Both avoid `batched_nms`, whose per-group switch the dynamo ONNX export of a dynamic batch stops on |

With `Detector.nms_criterion: "distance_iou"`, the eager path calls `batched_distance_nms()`: a box is suppressed when its IoU with a better kept box minus the normalized centre distance, to the power of `nms_distance_exponent`, exceeds the threshold (Zheng et al., AAAI 2020, Eq. 13). Two overlapping boxes with distant centres can then both be kept. The `InferenceDetector` and the exports call `suppress_every_pair_by_distance()`, the same suppression with static shapes on boxes divided by the input size, so TensorRT FP16 never overflows.

**Example: a training batch in inference mode at the deployment threshold, end of the CPU run**
```text
number_detections [0, 0]
detection 0 of image 0: box [0.0, 0.0, 0.0, 0.0]   score 0.0000   label 0        <- padding
class probabilities of the first proposal of image 0: [0.1327, 0.0894, 0.0938, 0.093, 0.1029, 0.1006, 0.0877, 0.0958, 0.0952, 0.109]
```
No class reaches 0.5, so the deployed detector returns nothing. At 0.05, every test image gets 100 `octagon` detections (Step 4.8).

### Step 4.5 — `accumulate_evaluation_batch()` → `match_image_detections()` + `add_to_confusion_matrix()` + `dump_errors()`
For every image, the valid detections and ground truth are kept on the CPU for the mAP. The detections reaching the deployment threshold are then matched to the objects for the confusion matrix:
- `compute_overlap_matrix()` gives the IoU of every detection with every object, whatever their classes;
- `match_image_detections()` goes twice through the detections by decreasing score, each taking the free object it overlaps most, at least by 0.5: first among the objects of its class, then among the remaining objects of any class (the confusions);
- `add_to_confusion_matrix()` counts each match in its cell, each unmatched detection in the background row (false positive) and each unmatched object in the background column (missed).

The mAP keeps its own matching (Step 4.9). At iteration 5000, 40,479 detections reach 0.5: 34,213 right, 3,228 of the wrong class, 3,038 false positives, and 32,241 of the 69,682 objects missed.

With `error_dumps.activated`, `dump_errors()` saves the image once per kind of error it holds (`group_image_errors()`): `Errors/FALSE-POSITIVE/<class>`, `Errors/MISSED/<class>` and `Errors/<TRUE>-TO-<PREDICTED>`. `draw_test_image()` draws it like a sample image (Step 4.6), the error boxes outlined with red dashes and described in the caption. The file is named after the image and the highest error score, at most `maximum_per_class_pair` per folder.

**Example: the error images of the detection smoke test** (`tests/trainers/test_detection_trainer.py`: 4 test images of 256 pixels, 5 iterations, deployment threshold lowered to 0.05; shortened)
```text
Evaluation/Iteration_5/Errors/
├── FALSE-POSITIVE/circle/      test_000001_0.109.png  test_000002_0.109.png
├── MISSED/hexagon/             test_000001.png  test_000002.png
├── ...
└── plus_sign-TO-triangle/      test_000000_0.109.png
caption:  "plus_sign -> triangle | 4 ground truth boxes | 100 detections with score >= 0.05"
```

### Step 4.6 — `save_sample_images()` → `build_sample_image()` → `build_detection_sample_grid()`
Two panels side by side: the ground truth boxes and the detections reaching the deployment threshold, every box captioned in its class colour. A caption bar counts both and a legend names the classes drawn.
- A ground truth box gives its class.
- A detection gives its class, score and IoU (`format_detection_caption()`, e.g. `circle 0.97 IoU 0.91`). The IoU is that with the object `match_image_detections()` gives it (Step 4.5), as `gather_detection_overlaps()` reads it.
- An unmatched detection gives the IoU of the object it overlaps most, followed by `no match` (`circle 0.41 IoU 0.32 no match`).

**Example: `Evaluation/Iteration_5000/Samples/test_014285.png`** (1028 x 560 pixels, recorded before the IoU in the captions)
```text
caption: "6 ground truth boxes | 4 detections with score >= 0.50"
panels:  Ground Truth (square, triangle, star, parallelogram, 2 nonagon boxes) | Prediction (square 0.98, triangle 0.58,
         star 0.96, nonagon 0.55)
EMA/Samples/test_014285.png: "6 ground truth boxes | 2 detections with score >= 0.50" (square 0.97, star 0.97)
```
The averaged weights return fewer detections at the deployment threshold (32,506 against 40,479 on the test split, Step 4.13), although their mAP is higher.

### Step 4.7 — `compute_evaluation_results()`
`summarise_average_precisions()` runs `evaluate_detections()` (mAP with three methods, Steps 4.8 to 4.10, and per object size, Step 4.14) and builds the class rows and tables. `save_proposal_recall()` writes the proposal recall (Step 4.15), `class_metrics.csv` gets the class rows (Step 4.11), and `save_error_analysis()` saves the class thresholds and the confusion matrix (Step 4.12). Detection has no headline metric line (`metrics` is empty); checkpoints are compared on `mAP@0.5/area`.

**Example: the results of iteration 5000**
```text
evaluated_samples = '20000 Test Images'
metrics           = {}
headline_metric   = 'mAP@0.5/area'
files             = ['Class Metrics', 'Proposal Recall', 'Class Thresholds', 'Confusion Matrix Table', 'Confusion Matrix Image', 'Sample Images']
scalars           = 52: {'mAP@0.5/area': 0.7151, 'mAP@0.75/area': 0.6451, 'mAP@0.95/area': 0.0021, 'mAP@[0.5:0.95]/area': 0.5160, ...,
                    'mAP@0.5 By Size/small': 0.0155, ..., 'mAR@0.5/all': 0.9801, ..., 'Proposal Recall@0.5/all': 0.9967, ...}
```
Current results also hold `percent_metrics = ['mAP@0.5/area']`, so the best checkpoint line writes it as a percentage (Step 5.5). The `percent_columns` of every table are written `71.51 %`; files and `scalars` stay between 0 and 1.

### Step 4.8 — `gather_class_detections()` → `find_best_ground_truth()`
For every foreground class, once for all thresholds and methods: the detections of the class sorted by decreasing score, the best IoU of each with a same-class ground truth box of its image (0 when none), and the class's ground truth count.

**Example: the class with the most detections, iteration 12 of the CPU run**
```text
detections per class {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 2000}
class 9 (octagon): 2000 detections, 7 ground truths
   largest best IoU 0.4701, detections with a best IoU >= 0.3: 7, >= 0.5: 0
```

### Step 4.9 — `match_detections_to_ground_truth()` + `compute_precision_recall()`
At one IoU threshold, detections are visited by decreasing score. A detection is a true positive when its best ground truth reaches the threshold and was not taken by a more confident one, a false positive otherwise (the PASCAL VOC rule). Every detection is one point of the precision-recall curve. In the example, octagon at 0.5 has 0 true positives of 2000 detections.

### Step 4.10 — `compute_average_precision()` + `compute_mean_average_precision()`
This table gives the three methods that summarise a class curve.

| Method | Average precision |
|---|---|
| `area` | area under the curve, every precision raised to the best precision at a higher recall (PASCAL VOC 2010 and later) |
| `11_point` | mean interpolated precision at recalls 0, 0.1, ..., 1 (PASCAL VOC 2007) |
| `101_point` | the same at 101 recall levels (COCO) |

The mAP is the mean over the classes with ground truth boxes, at 0.5, 0.75, 0.95 and averaged over 0.50, 0.55, ..., 0.95 (`mAP@[0.5:0.95]`), with the same four values per class. The matching is the PASCAL VOC rule at every threshold, unlike pycocotools (see Pitfalls).

**Example: iteration 5000**
```text
area:      mAP@0.5 71.51 %   mAP@0.75 64.51 %   mAP@0.95 0.21 %   mAP@[0.5:0.95] 51.60 %
11_point:  mAP@0.5 68.78 %   mAP@0.75 63.51 %   mAP@0.95 1.12 %   mAP@[0.5:0.95] 50.77 %
101_point: mAP@0.5 71.32 %   mAP@0.75 64.36 %   mAP@0.95 0.30 %   mAP@[0.5:0.95] 51.52 %
octagon (area): AP@0.5 51.44 %   AP@0.75 48.96 %   AP@0.95 0.41 %   AP@[0.5:0.95] 39.56 %
```
The three methods agree within 3 points. The 11-point values are lower at 0.5 and higher at 0.95, since recall 0 weighs a whole eleventh. The drop to 0.21 % at 0.95 shows boxes mostly well placed but almost never tight enough.

### Step 4.11 — `build_class_average_precision_rows()` + `write_class_metrics_file()`
Every foreground class gets its area average precisions, then the AP@0.5 and AP@[0.5:0.95] of every size (Step 4.14, empty without objects of the size), then the threshold reaching the target precision and its recall (Step 4.12, empty when never reached). Then come `recall_50` and `recall_75`: the share of the class's ground truths matched by any kept detection at IoU 0.5 and 0.75 (the per-class `mAR`), each with the bounds of its exact (Clopper-Pearson) 95 % interval (`compute_clopper_pearson_intervals()` of `vision_training/evaluation/recall_intervals.py`). The file is written whatever the per-class report.

**Example: `Evaluation/Iteration_5000/class_metrics.csv`** (the hexagon has no small test object; the rhombus never reaches precision 0.99)
```text
class_index,class,ground_truths,average_precision_50,average_precision_75,average_precision_95,average_precision_50_95,average_precision_50_small,average_precision_50_95_small,average_precision_50_medium,average_precision_50_95_medium,average_precision_50_large,average_precision_50_95_large,threshold_at_target_precision,recall_at_target_precision
1,square,3545,0.969327,0.917248,0.006597,0.732593,0.000000,0.000000,0.934682,0.659275,0.984351,0.768852,0.895632,0.884062
...
6,hexagon,3471,0.820560,0.791622,0.006872,0.645796,,,0.701038,0.510686,0.882367,0.721177,0.829466,0.009219
...
10,rhombus,3436,0.362592,0.339679,0.000007,0.250137,0.004049,0.001619,0.365671,0.252234,0.366849,0.253416,,
```

### Step 4.12 — `find_class_score_thresholds()` + `save_error_analysis()`
From the AP@0.5 matching, `find_score_threshold()` (shared with the classification) keeps the score reaching the target precision with the highest recall over every ground truth box of the class. `save_error_analysis()` writes:
- `class_thresholds.json`, in the format of the classification tasks (`null` when the precision is never reached);
- `confusion_matrix.csv`, rows the true classes, columns the predicted ones, background first;
- with the per-class report, `confusion_matrix.png`, logged to TensorBoard (shares as `94.03 %`, cells of 0.65 inch per class, the background cell blank).

The report gets a note counting the outcomes and, with error images, a note counting them. At iteration 5000, 17 classes get a threshold and 3 never reach 0.99 (rhombus, kite, capsule). The trapezoid reaches 0.99 only above 0.4119, with a recall of 0.03 %; the plus_sign keeps 91.98 % above 0.6335.

**Example: the files of iteration 5000** (shortened)
```text
confusion_matrix.csv
true \ predicted,background,square,circle,triangle,rectangle,...,annulus,capsule,l_shape
background,0,536,159,418,282,...,91,140,703
square,117,3397,0,0,31,...,0,0,0
...
octagon,3068,6,93,0,2,...
...

class_thresholds.json
{"target_precision": 0.99, "classes": {"square": {"threshold": 0.8956319689750671, "recall": 0.8840620592383639}, ...,
 "rhombus": null, ..., "l_shape": {"threshold": 0.7305112481117249, "recall": 0.036281179138321996}}}
```
The diagonal shows over 3,000 right detections for the square, circle, rectangle, plus_sign, star and annulus, none for the rhombus, trapezoid and kite. Most objects of these classes are missed rather than confused. Confusions are between similar shapes (430 heptagons found as hexagons, 400 nonagons as circles).

### Step 4.13 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
The report block lists the tables of Steps 4.10, 4.14, 4.15 and 4.11, then, with `per_class_report`, the `Per Class Recall` table (`build_recall_interval_table()`): the ground truths of every class, `Recall @0.5` and `Recall @0.75`, each with `95 % Lower` and `95 % Upper`, worst first.

**Example output: the block of iteration 5000** (shortened; the same text, without the file lines, goes to `Results.txt`)
```text
Iteration 5000 · 20000 Test Images · 2026-10-24 16:09 ──────────────────────────────────────────────────────────────────
  Confusion Matrix : score >= 0.50, IoU >= 0.5 (boxes), 34213 right, 3228 wrong class, 3038 false positives, 32241
  missed
  Sample Images : 8 fixed test images

  Mean Average Precision
   Method      mAP@0.5   mAP@0.75   mAP@0.95   mAP@[0.5:0.95]
  ────────────────────────────────────────────────────────────
   Area        71.51 %    64.51 %     0.21 %          51.60 %
   ...
  The mAP@0.5 of the area method compares the checkpoints of the run.

  Mean Average Precision By Object Size (Area)          (Step 4.14)
  Mean Average Recall                                   (Step 4.14)
  Proposal Recall                                       (Step 4.15)

  Per Class Average Precision (Area)
                   Ground                                                 Threshold    Recall
   Class           Truths    AP@0.5   AP@0.75   AP@0.95   AP@[0.5:0.95]     @ P99 %   @ P99 %
  ────────────────────────────────────────────────────────────────────────────────────────────
   trapezoid         3546   31.84 %   28.70 %    0.02 %         22.25 %      0.4119    0.03 %
   kite              3471   34.00 %   28.84 %    0.00 %         22.02 %           —         —
   ...
  Worst AP@0.5 first, ties in class order. — : no ground truth box of the class, or the precision 99 % is never reached.
  - Results : Results.txt
  - ...
```

The block of the averaged weights follows (`Iteration 5000 · 20000 Test Images · EMA Weights`). This table compares both at iteration 5000.

| Weights | mAP@0.5 | mAP@[0.5:0.95] | mAP@0.95 | Right / wrong class / false positives / missed |
|---|---|---|---|---|
| Trained | 71.51 % | 51.60 % | 0.21 % | 34,213 / 3,228 / 3,038 / 32,241 |
| Averaged (EMA) | 72.08 % | 56.73 % | 1.55 % | 29,700 / 1,401 / 1,405 / 38,581 |

The averaged weights lead more as the IoU grows, but fewer of their detections score 0.5 or more. Both curves still rise at 5000.

**Example: TensorBoard and MLflow after the 20 evaluations**
```text
Tensorboard/Test: 72 scalars (12 mAP, 6 by size, 16 mAR, 18 proposal recall, 20 AP@0.5 per class), 9 images (8 samples, confusion matrix)
    scalar mAP@0.5/area             500: 0.0496, 1000: 0.1347, 2000: 0.3241, 3000: 0.5031, 4000: 0.6440, 5000: 0.7151
MLflow: 154 metrics, among them test/best_mAP_at_0.5/area (0.7151 at step 5000); the evaluation files under evaluation/iteration_<n>/
```
`sanitise_metric_name()` writes `@` as `_at_`, drops brackets and writes `:` as `-` (`mAP@[0.5:0.95]/area` becomes `mAP_at_0.5-0.95/area`); `metric_names.json` keeps the originals. `best_` follows the headline metric only. `update_evaluation_curves()` then prints `Evaluation Curves : <run>/evaluation_curves.png` and sends `evaluation_summary.csv` and `evaluation_curves.png` to the MLflow run.

### Step 4.14 — `select_object_size_detections()` + `build_object_size_table()`
`evaluate_detections()` computes the same mAPs again for five object sizes, reusing Steps 4.8 and 4.9. The size is the side of the box (the square root of its area) as a share of the side of the network input (`OBJECT_SIZE_SIDE_SHARES` in `object_sizes.py`, `compute_side_shares()` with the `image_area` of every ground truth). The bins double: under 1/16, 1/16 to 1/8, 1/8 to 1/4, 1/4 to 1/2, 1/2 and above. So an object keeps its size whatever `Data.image_settings.size`. `select_object_size_detections()` applies the COCO rule: ground truths of other sizes are ignored, as are the detections matched to them and the unmatched detections whose own box is of another size. The table adds each size in pixels of the network input (`Pixels`: at 512, under 32, 32 to 64, 64 to 128, 128 to 256, 256 and above). A size without objects shows `—` and gives no scalar. The scalars are `mAP@0.5 By Size/<size>` and `mAP@[0.5:0.95] By Size/<size>`, the sizes being `under_1_16`, `1_16_to_1_8`, `1_8_to_1_4`, `1_4_to_1_2` and `1_2_and_above`.

`build_average_recall_table()` then gives the `Mean Average Recall` table: the share of objects found by any kept detection, at IoU 0.5, 0.75, 0.95 and over 0.5:0.95, averaged over the classes. A low mAP with a high recall means objects are found but scored below false positives; a low recall means they are not found. Its scalars are `mAR@<threshold>/<group>`, the group being `all` or a size.

**Example output: iteration 5000**, printed with the COCO sizes of that day (the generator of that run drew no shape under 64 pixels, so the 184 small objects are mostly hidden shapes). The current tables start with `Object Side` (or `Objects Of`) and `Pixels`, then one row per relative size.
```text
  Mean Average Precision By Object Size (Area)
   Object Size           Objects   mAP@0.5   mAP@0.75   mAP@0.95   mAP@[0.5:0.95]
  ────────────────────────────────────────────────────────────────────────────────
   Small (< 32²)             184    1.55 %     0.02 %     0.00 %           0.40 %
   Medium (32² to 96²)     33579   66.54 %    58.11 %     0.07 %          45.95 %
   Large (>= 96²)          35919   75.20 %    69.59 %     0.34 %          56.00 %

  Mean Average Recall
   Objects Of            Objects   mAR@0.5   mAR@0.75   mAR@0.95   mAR@[0.5:0.95]
  ────────────────────────────────────────────────────────────────────────────────
   All                     69682   98.01 %    88.93 %     3.33 %          73.81 %
   Small (< 32²)             184    8.60 %     0.53 %     0.00 %           2.32 %
   Medium (32² to 96²)     33579   97.33 %    86.13 %     1.79 %          70.73 %
   Large (>= 96²)          35919   98.97 %    91.92 %     4.34 %          76.64 %
```
Objects under 32 pixels reach 1.55 %: only 10.3 % of the 1,254 such training objects got a foreground anchor in that run. A `mAR@0.5` of 98.01 % against a `mAP@0.5` of 71.51 % shows that almost every object is found above 0.05, but weak classes rank below false positives.

### Step 4.15 — `accumulate_proposal_recall()` + `save_proposal_recall()`
The anchor recall (Step 2.21) measures what the anchors allow; the proposal recall measures what the trained region proposal network gives the heads. `accumulate_proposal_recall()` keeps, for every test object, its best IoU with the valid proposals of Step 4.3 (`valid_proposal_mask`), overall and per level. `summarise_proposal_recall()` gives the share of objects reaching 0.5 and 0.7 (`PROPOSAL_RECALL_THRESHOLDS`) for all objects, every size and every block alone, with the mean proposals per image. The rows go to `proposal_recall.csv` and the scalars (`Proposal Recall@0.5/<group>`, ...) to TensorBoard and MLflow.

**Example output: the table of iteration 5000**, with the COCO sizes of that day (the current table adds a `Pixels` column and one row per relative size)
```text
  Proposal Recall
                                                           Proposals
   Objects               Count   Recall@0.5   Recall@0.7   Per Image
  ───────────────────────────────────────────────────────────────────
   All Objects           69682      99.67 %      97.51 %    249.9534
   Small (< 32²)           184      39.67 %       8.70 %           —
   Medium (32² to 96²)   33579      99.73 %      96.80 %           —
   Large (>= 96²)        35919      99.92 %      98.64 %           —
   Block 5 Proposals     69682      80.93 %      50.95 %    166.8608
   Block 6 Proposals     69682      77.34 %      26.26 %     49.9440
   Block 7 Proposals     69682      67.36 %      42.71 %     33.1486
```
`proposal_recall.csv` holds the same rows (`group,objects,recall_50,recall_70,proposals_per_image`) with values between 0 and 1. The region proposal network is not the limit: its recall at 0.5 was already 98.80 % at iteration 500, when the `mAP@0.5` was 4.96 %. The errors of Step 4.12 come from the classification and scores of the detection heads. Objects under 32 pixels stay the exception, 73 of 184 reached at 0.5.

---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of `scripts/evaluation/evaluate_checkpoint.py` on the 20 checkpoints of the 5000-iteration run, with `iteration = "all"`, `batch_size = 64`, `track_evaluation = False` and a test folder of the first 200 test images (709 annotations). It took 59 minutes on a shared GPU. Step 5.4 comes from a tracked evaluation of the CPU run. They predate the per-class recall intervals.

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The same script for every task (`classification_workflow.md`, Stage 5), with the arguments of [Run It](#run-it). `--save_predictions` is skipped with a warning, since detection writes no predictions file. `--target_precisions` takes one precision; a list is refused. The output goes to `Checkpoint_Evaluation/` (`Checkpoint_Evaluation_split_<parent folder>_<split folder>/` for another split). To score a greedy-NMS run with DIoU-NMS, add `--nms_criterion distance_iou --nms_distance_exponent 0.6`: the output then goes to `<run>/Checkpoint_Evaluation_nms_distance_iou_0.6/` (`apply_nms_criterion()`). The default `--maximum_error_images_per_class_pair 100000` turns every error image on. The `pretrained` settings are turned off (`disable_pretrained_settings()`).

### Step 5.2 — trainer construction
The trainer is built from the run's `training_configuration.yaml`, with the classes of its `label_dictionary.json` (`classification_workflow.md`, Step 5.2).

**Example: the launch information lines set by the arguments**
```text
- Test Data Folder        : test_reduced_200 (200 images, 709 annotations)
- Batch Size              : 64
- MLflow Tracking         : Off
```

### Step 5.3 — `get_checkpoint_iterations()`
**Example output**
```text
evaluated_iterations = [5000, 4750, 4500, ..., 500, 250]       (20 checkpoints)
```

### Step 5.4 — `evaluate_checkpoints()` → `start_evaluation_tracking()`
With tracking on, the evaluations go to their own MLflow run, `<run name>-evaluation`, tagged with the training run read from `mlflow_run.json` (`evaluation_of`); the training run is only read.

**Example output** (the tracked evaluation of the CPU run)
```text
MLflow Run Created In The Experiment 'object_detection' : fcc494bfb83d4bc78638fa8ef3947ea6
```

### Step 5.5 — `summarise_checkpoint_evaluations()`
**Example output**
```text
Evaluation Summary Of 20 Checkpoints : evaluation_summary.csv
  - Best Checkpoint : iteration 5000 (mAP@0.5/area 72.10 %)
  - Best Checkpoint (EMA Weights) : iteration 5000 (mAP@0.5/area 72.87 %)
```
The best value is a percentage, as `percent_metrics` names the headline metric (Step 4.7); `evaluation_summary.csv` keeps values between 0 and 1. The current code adds an `Evaluation Curves` line and an `ema_number_updates` column (146 columns for this run).

**Example: `<run>/Checkpoint_Evaluation/evaluation_summary.csv`** (the iteration, 72 scalars, the same 72 prefixed `ema_`; a few columns)
```text
iteration   mAP@0.5/area   mAP@[0.5:0.95]/area   AP@0.5/kite   AP@0.5/plus_sign   ema_mAP@0.5/area   ema_mAP@[0.5:0.95]/area
   5000       0.7210         0.5257                0.2906        0.9992             0.7287             0.5794
   4250       0.6871         0.5000                0.3471        0.9972             0.6621             0.5184
   ...
    250       0.0242         0.0084                0.0037        0.0000             0.0192             0.0061
```
On 200 images the last checkpoint is best for both weights, close to the whole split (Step 4.13). With 23 to 45 objects per class, class columns move a lot: kite is best at 4250 (0.3471). Check a checkpoint chosen on a reduced split on the whole split before deploying it.

### Step 5.6 — `run_test_evaluation()` for each checkpoint
Each checkpoint prints `Evaluating : model_005000.pt`, then the block of Step 4.13. On the 200 images, iteration 5000 gives `335 right, 29 wrong class, 35 false positives, 345 missed`. The weakest classes are those of the whole split: kite (AP@0.5 29.06 %), trapezoid (40.58 %), rhombus (43.24 %), then capsule (44.39 %, never reaching precision 99 %). A class without ground truth box would have no average precision (`—`), be left out of the mAP and be listed last.
