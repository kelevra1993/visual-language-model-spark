# Object Detection Workflow

This document follows one object detection run (DET) from `main()` to the evaluation of a checkpoint, to show which function runs when, what it receives, what it returns and who uses the result next. The **Call Tree Overview** gives the whole run on one page. Each stage then has one section, with one step per function: a short explanation followed by a **real** output. The steps shared by every task (configuration loading, checkpoints, training loop, MLflow tracking, standalone evaluation) are explained in detail in `classification_workflow.md`; here they keep a short explanation and the real output of the detection run, and the steps specific to the detection (COCO loaders, anchors, region proposals, target assignment, detection heads, the four losses, post-processing, mean average precision) are explained in full.

The examples of Stages 1 and 2 and of the steps of one batch come from real runs on CPU, so that the GPU stayed free for another training. The synthetic dataset of the configuration, `synthetic_shapes`, was not on the machine when they were recorded, so a small one was generated in a temporary folder with `create_synthetic_dataset(task="object_detection")`: 100 training, 20 validation and 20 test images of 512 pixels, 1 to 6 shapes per image, 9 shape classes. The data folders and the mapping file of the configuration were redirected to it, and the DALI loader the configuration then used, which decoded the images on the GPU, was replaced by the PyTorch loader. The DALI loader has since been removed from the repository (`19b89a0`): every split is read by `CocoDetectionDataset` on the CPU loader workers.
- **Stages 1 and 2**: `vision_training/configurations/object_detection/configuration_synthetic_shapes.yaml` as it was then (1024 pixel images, batch size 40, detection on the blocks 5, 6 and 7 of the backbone), with the run folder redirected to a temporary folder.
- **Stages 3, 4 and 5, the outputs of the run** (the iteration table, curves, MLflow run, checkpoints, console summaries, evaluation reports, files and sample images, TensorBoard and MLflow scalars, standalone evaluation of the checkpoints on the first 200 test images, Step 5.4 excepted): the 5000-iteration run of [`run_configurations/object_detection_workflow_run.yaml`](run_configurations/object_detection_workflow_run.yaml), trained on the GPU on 2026-10-24 with the current code on the full synthetic dataset of this machine (128,000 / 20,000 / 20,000 images, 20 classes). This configuration is the synthetic configuration launched by `Run It` with four changes: 5000 iterations instead of 2000, a checkpoint evaluated every 500 iterations instead of 250 (an intermediate one every 250 instead of 125), the MLflow tracking turned on, and its own experiment folder `experiments/workflow_runs`.
- **Stages 3 and 4, the steps of one batch** (the tensors of Steps 3.6 to 3.18, 3.20 and 3.21, and of Steps 4.2 to 4.4, 4.8 and 4.9): the same configuration as Stages 1 and 2, shortened to 512 pixel images (the smallest size that keeps the anchors of the configuration, the largest being 512 pixels), batch size 2, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4, a validation batch every 10, 2 loading workers, with the MLflow tracking turned on, run on CPU. After 12 iterations this detector finds no shape, so its values only show the shapes and the order of the steps.
- **Step 2.21** (the anchor recall): a later run with the settings of that 12-iteration run, on a dataset of the same sizes made by the generator of that time, with 20 shape classes (it now draws 35).

> The examples of the CPU runs were recorded before the following changes, which they do not show (the outputs of the 5000-iteration run, recorded with the current code, show them):
> - the ground truth boxes are written into the proposals of the detection heads in training and in the validation loss (`add_ground_truth_proposals()`, `a10bb7a`): Steps 3.14 and 3.21 show the positives of the region proposal network alone;
> - the region proposal NMS is 0.7 and the detector NMS 0.5 (they were 0.5 and 0.8), and the anchors overlapping no object (IoU 0) can be sampled as background (`RegionProposal.background_iou_threshold.min: 0.0`, it was 0.1);
> - the evaluation tables, scalars and `class_metrics.csv` have a column at IoU 0.95 (`mAP@0.95`, `AP@0.95`, `average_precision_95`, `351a4d1`), and the report shows a `Mean Average Recall` table after the size tables;
> - the checkpoints also record the source run of the region proposal and detection heads (`pretrained_region_proposer_dictionary`, `pretrained_detector_heads_dictionary`), and the synthetic configuration keeps a moving average of the weights (`Optimization.exponential_moving_average`), whose checkpoint state is saved with the weights and which is evaluated after the trained weights in `Evaluation/Iteration_<n>/EMA/`, `Tensorboard/Test_EMA/` and an `EMA Weights` block of `Results.txt`;
> - the summaries list the residual shortcuts only for a block adding one (`count_residual_additions()`), the panel of Step 2.19 being printed again with the current code.
>
> The synthetic configuration now trains at 512 pixels on the instance segmentation dataset, with blocks of 16 and 32 filters, two anchor scales per level and smaller heads, batch size 64 and 2000 iterations: `Read This First` describes it with a run of `experiments/readme_screenshots`, and Stages 3 to 5 show the outputs of its 5000-iteration copy. The configuration of Stages 1 and 2 is the one of `vision_training/configurations/object_detection/configuration_example.yaml` (same backbone, levels and heads, at 1024 pixels).

## Read This First
The detector learns to find every object of an image: its box, its class and a score. It is a two-stage detector in the style of Faster R-CNN, without feature pyramid: every backbone block named in `DetectionLevels` gets its own anchors, region proposal head and detection head, and every proposal is classified by the head of the level its anchor came from. The input is a COCO split (images and a `labels.json` whose objects have a `bbox`); the output is a fixed number of detections per image (`Detector.inference.maximum_detections`, padded), boxes in network input pixels. Choose DET when boxes are enough; INS (`instance_segmentation_workflow.md`) when every object also needs its pixel mask, the same detector with one mask head per level; KPT when the objects are described by keypoints (a bottom-up HigherHRNet, without boxes or anchors); SEG when the class of every pixel is enough and the objects need not be told apart; CLS for one label per image.

### The Task At A Glance
Shapes of the synthetic configuration (512 pixels, batch size 64, 20 classes and the background), checked by running the last checkpoint of the DET run of `experiments/readme_screenshots` on the CPU.

```text
<dataset>/{train,validation,test}/data/*.png + labels.json (COCO bbox [x, y, w, h], category_id) + label_mapping.json
   │ CocoDetectionDataset on the CPU workers: read_image, select_image_objects, letterbox to Data.image_settings.size
   ▼
sample  images [3, 512, 512] uint8 BGR · bounding_boxes [N, 4] x1 y1 x2 y2 · labels [N] in 1..20 (0 = background)
batch   images [64, 3, 512, 512] · boxes and labels: lists of 64 tensors · DataAugmenter moves the boxes (GPU)
   ▼
Backbone, 7 ConvolutionBlocks ─► block 5 [64, 32, 16, 16] stride 32 · block 6 [64, 32, 8, 8] · block 7 [64, 32, 8, 8]
   ▼ one RegionProposal per level: 6 anchors per cell (2 scales x ratios 0.5, 1, 2), 1536 + 384 + 384 = 2304 anchors
objectness [64, 2304] · offsets [64, 2304, 4] ─► RPN losses: BCE + smooth L1 (beta 1/9), 256 anchors per image sampled
   ▼ RegionProposalFilter: top 1000, NMS 0.7, 500 proposals per image (evaluation: 500, NMS 0.7, 250)
   ▼ training: + ground truth boxes, 128 proposals per image sampled, at most 64 positive (IoU >= 0.5)
DetectionHead of the level of every proposal: RoIAlign 7x7 ─► 2 conv x 32 ─► FC 1568 > 64 > 32 ─► class + box layers
logits [64, 128, 21] · offsets [64, 128, 21, 4] ─► detection losses: cross entropy + smooth L1 of the true class
   ▼ evaluation: postprocess_detections: softmax, score >= 0.05 (0.5 deployed), NMS 0.5 per class, top 100
final_detection_boxes [B, 100, 4], _scores [B, 100], _labels [B, 100] · number_detections [B]
   ▼
evaluate_detections ─► mAP@0.5 / 0.75 / 0.95 / [0.5:0.95] x area, 11, 101 point · by size · mAR · AP per class
                       headline mAP@0.5/area · proposal recall · class thresholds at precision 0.99 · confusion matrix
                       · recall of every class at 0.5 and 0.75 with its 95% interval
<run>/anchor_recall.csv, Results.txt · Evaluation/Iteration_<n>/{class_metrics.csv, class_thresholds.json,
confusion_matrix.csv/.png, proposal_recall.csv, Samples/, EMA/} · Tensorboard/{Train, Test, Test_EMA}
```

### Files To Know
| Role | File | Main functions or classes |
|---|---|---|
| Trainer | `vision_training/trainers/detection_trainer.py` (bases `coco_split_trainer.py`, `base_trainer.py`) | `DetectionTrainer`: `build_dataloaders`, `analyse_training_data`, `forward_pass`, `evaluation_context`, `predict_batch`, `accumulate_evaluation_batch`, `compute_evaluation_results`, `save_error_analysis`, `draw_test_image` |
| Dataset and loaders | `vision_training/data/detection/coco_dataset.py`, `vision_training/data/loaders.py` | `CocoDetectionDataset`, `select_image_objects`, `collate_function`, `get_detection_dataloaders` |
| Model | `vision_training/models/detection/two_stage_detector.py` | `TwoStageDetector`: `forward_detection_stages`, `_prepare_detector_inputs`, `get_level_anchors`, `print_summary` |
| Anchors and proposals | `vision_training/models/detection/anchors.py`, `region_proposer.py` | `Anchors.generate_anchors`, `RegionProposal`, `RegionProposalFilter`, `select_best_proposals` |
| Head and boxes | `vision_training/models/detection/detection_head.py`, `box_operations.py`, `postprocessing.py` | `DetectionHead`, `apply_regression_predictions`, `BOX_REGRESSION_WEIGHTS`, `postprocess_detections` |
| Targets and losses | `vision_training/models/detection/target_assignment.py`, `vision_training/losses/detection_losses.py` | `batch_assign_targets_to_anchors`, `add_ground_truth_proposals`, `batch_assign_targets_to_proposals`, `sample_positive_and_negative_training_targets`, `compute_region_proposal_losses`, `compute_detection_losses` |
| Metrics and evaluation | `vision_training/evaluation/detection/mean_average_precision.py`, `average_precision_report.py`, `region_recall.py`, `detection_confusion.py`, `object_sizes.py`, `vision_training/evaluation/recall_intervals.py` | `evaluate_detections`, `summarise_average_precisions`, `summarise_anchor_recall`, `summarise_proposal_recall`, `match_image_detections`, `compute_clopper_pearson_intervals` |
| Configuration checks and run name | `vision_training/configuration_processing/detection_checks.py`, `run_name.py` | `check_detection_configuration`, `build_detection_sections`, `build_anchors_hash` |
| Synthetic data | `vision_training/data/synthetic/create_synthetic_dataset.py` | `create_synthetic_dataset` (task `"instance_segmentation"` for the DET configuration) |
| Export | `vision_training/models/detection/inference_detector.py`, `vision_training/deployment/export_specifications.py` | `InferenceDetector`, `get_export_specification` (inputs `images`, outputs `detection_boxes`, `detection_scores`, `detection_labels`, `number_detections`) |

### Run It
1. Generate the dataset: the DET configuration reads the instance segmentation dataset, whose boxes are those of the visible part of every shape. In the main block of `vision_training/data/synthetic/create_synthetic_dataset.py`, set `task = "instance_segmentation"` (its `output_folders`, `image_sizes`, `split_sizes` and `maximum_objects` entries give `/home/robert_kelevra/Data/instance_segmentation_data/synthetic_shapes`, 512 pixels, 128,000 / 20,000 / 20,000 images and up to 6 shapes), then run it. A new dataset has the 35 shapes of `SHAPE_CATEGORIES` (run names ending in `MC35`); the dataset on this machine, made earlier, has 20 (`MC20`).
```bash
uv run python -m vision_training.data.synthetic.create_synthetic_dataset
```
2. Train, after setting `experiment_folder` (and the data folders and mapping file if the dataset is elsewhere) in the configuration:
```bash
uv run python -m vision_training.main --configuration_path="vision_training/configurations/object_detection/configuration_synthetic_shapes.yaml"
```
3. Evaluate a checkpoint again: set `run_name`, `experiment_folder`, `iteration` (`None` for the last checkpoint, a number, or `"all"`), `test_data_folder`, `batch_size`, `track_evaluation` and `maximum_error_images_per_class_pair` in the main block of `scripts/evaluation/evaluate_checkpoint.py` (Stage 5), then run:
```bash
uv run python -m scripts.evaluation.evaluate_checkpoint
```

### Configuration Keys To Know First
| Key | Value in the synthetic configuration | Effect |
|---|---|---|
| `DetectionLevels.<5, 6, 7>.anchors` | scales `[32, 64]`, `[64, 128]`, `[128, 256]`, aspect ratios `[0.5, 1.0, 2.0]` | 6 anchors per cell, 2304 in all at 512 pixels; the `A6.32-256.e7b72c` token of the run name; `anchor_recall.csv` measures how they fit the training objects |
| `DetectionLevels.<level>.region_proposal`, `.detector` | `[2, 32]`; RoIAlign 7, `[2, 32]`, fully connected `[64, 32]` | the heads of every level (`RBN-R2x32`, `DSE-DRES-DBN-D7.2x32.64.32`) |
| `RegionProposal.foreground_iou_threshold.min`, `background_iou_threshold` | 0.5; min 0.0, max 0.3 | anchor labels: positive from 0.5, negative below 0.3, ignored in between, plus the best anchor of every object |
| `RegionProposal.training`, `.inference`, `nms_iou_threshold` | 1000 then 500; 500 then 250; 0.7 | proposals given to the detection heads per image |
| `RegionProposal.number_training_positives`, `total_training_samples` | 128, 256 | anchors sampled per image for the region proposal losses |
| `Detector.foreground_iou_threshold`, `number_training_positives`, `total_training_samples` | 0.5, 64, 128 | proposals sampled per image for the detection losses, the ground truth boxes included |
| `Detector.inference.score_threshold`, `maximum_detections`, `Detector.nms_iou_threshold` | 0.5, 100, 0.5 | post-processing of the deployed detector, also used by the sample images, the confusion matrix and the error images |
| `ExperimentConfiguration.evaluation.score_threshold`, `target_precision` | 0.05, 0.99 | detections kept for the mAP; precision of the class thresholds of `class_thresholds.json` |
| `Optimization.loss_weights` | 1.0 for each of the four losses | weights of the losses in the total loss, the losses being tracked unweighted |
| `Data.image_settings.size` | 512 | network input; the anchor scales must fit in it |

### What A Run Writes
- At every launch, `anchor_recall.csv` in the run folder: the share of the training objects with a foreground anchor, per level and size (Step 2.21).
- At every `weight_saving_iterations` checkpoint, in `Evaluation/Iteration_<n>/`: `class_metrics.csv` (AP of every class at 0.5, 0.75, 0.95 and 0.5:0.95, by size, recall at 0.5 and 0.75 with its 95% confidence interval, threshold and recall at the target precision), `class_thresholds.json`, `confusion_matrix.csv` and `.png` (a `background` row for the false positives and column for the missed objects), `proposal_recall.csv`, `Samples/<image>.png` (the ground truth boxes next to the detections reaching 0.5), `Errors/` with `error_dumps`, and the same files in `EMA/` for the moving average of the weights.
- `Results.txt`: one block per evaluation (two with the moving average) with the confusion matrix note and the tables of the methods, object sizes, mean average recall, proposal recall and classes, then the per-class recall table; `Tensorboard/Train` holds the total loss, the four losses and the learning rate, `Tensorboard/Test` and `Test_EMA` the evaluation scalars (72 for 20 classes), the 8 samples and the confusion matrix.
- In the DET run of `experiments/readme_screenshots` (the synthetic configuration resumed to 5000 iterations), iteration 5000 gives `mAP@0.5/area` 0.7469 (0.7429 with the averaged weights), `mAP@[0.5:0.95]` 0.5425 (0.5917), a proposal recall at 0.5 of 0.9965 and 40,068 right detections, 4,885 of the wrong class, 3,078 false positives and 24,729 missed objects at the deployment threshold.
- In the run of `run_configurations/object_detection_workflow_run.yaml` shown in Stages 3 to 5 (the synthetic configuration trained to 5000 iterations from scratch, evaluated every 500), iteration 5000 gives `mAP@0.5/area` 0.7151 (0.7208 with the averaged weights), `mAP@[0.5:0.95]` 0.5160 (0.5673), a proposal recall at 0.5 of 0.9967 and 34,213 right detections, 3,228 of the wrong class, 3,038 false positives and 32,241 missed objects at the deployment threshold; the mAP was still rising at the last evaluation.

### Pitfalls
- The synthetic configuration stops at 2000 iterations, where few detections reach the 0.5 of the sample images: in the run of `experiments/readme_screenshots`, `mAP@0.5/area` is 0.0901 at iteration 1000, with no detection reaching 0.5, and 0.3126 at 2000; the samples only show detections after the resume to 5000 (`suggestions.md`, item 30).
- Small objects are barely found: the finest level has a stride of 32 and anchors of 32 and 64 pixels, the anchors sitting on the cell corners (`suggestions.md`, items 1 and 5, audit L1). `anchor_recall.csv` of that run gives a foreground anchor to 10.3 % of the 1,254 small training objects, and at iteration 5000 the 184 small test objects have a `mAP@0.5` of 0.0147 and a proposal recall at 0.5 of 0.3315 (`Results.txt`).
- The proposals are selected over all the levels at once (`select_best_proposals()`, audit M3): at iteration 5000, 178.7 of the 250 proposals per image come from block 5 (`proposal_recall.csv`).
- The mAP matches the detections with the PASCAL VOC rule at every IoU, so `mAP@[0.5:0.95]` and the 101-point values are not those of pycocotools (`match_detections_to_ground_truth()`, audit L6).
- A head or backbone block whose first convolution changes the channel count adds no residual shortcut with 2 layers (`ConvolutionBlock.count_residual_additions()`, `bug_standards.md` H4, dropped): the detection heads of blocks 6 and 7 of `configuration_example.yaml` (128 to 64 channels) compute without shortcut, the summary listing only their squeeze-and-excitation (Step 2.19).

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
│   Entry point of every task. Give an object_detection configuration with --configuration_path to run this workflow.
│
├── load_experiment_configuration(configuration_path)
│   │   Splits the YAML into experiment_configuration ('ExperimentConfiguration') and model_configuration (Optimization,
│   │   Data, Backbone, DetectionLevels, RegionProposal, Detector), converts the types, adds tracking_root, run_name
│   │   (None), label_mode and label_dictionary.
│   │
│   ├── load_configuration(configuration_path)
│   ├── build_label_dictionary(label_mapping_configuration, task)
│   │       Reads the mapping file {category id: class index} and adds 'background': 0, since the detector predicts
│   │       the background as class 0. "binary" mode gives {background: 0, foreground: 1}.
│   └── check_configuration()
│       ├── common checks
│       └── check_detection_configuration()
│               ├── check_detection_evaluation_configuration()   evaluation.score_threshold, target_precision and
│               │                                                error_dumps
│               ├── check_detection_data_configuration()         classes 0 (background) to K-1 without gaps, the
│               │                                                categories mapped to 0 announced in yellow
│               ├── check_image_task_sequence_settings()         no sequence normalization
│               ├── check_image_only_augmentation_types()        random_perspective and random_jigsaw off, since the
│               │                                                boxes cannot follow them
│               ├── check_detection_levels()                     DetectionLevels: quoted backbone blocks in increasing
│               │                                                order, anchor scales within the image size, positive
│               │                                                ratios, region proposal and detection heads
│               ├── check_region_proposal_configuration()        IoU thresholds, proposals before and after NMS
│               ├── check_detector_configuration()               maximum detections, training samples
│               └── check_loss_weights_configuration()           Optimization.loss_weights of the four losses
│
├── get_trainer_class(task)  -> DetectionTrainer
│
├── DetectionTrainer(experiment_configuration, model_configuration, configuration_path)  [STAGE 2]
│   │   Reads the score threshold of the evaluation and the one of the sample images.
│   │
│   └── BaseTrainer.__init__()   (through CocoSplitTrainer, which adds the reading of the test manifest)
│       ├── get_device() / get_run_name() -> build_run_name() / DataAugmenter()
│       ├── restore_run_label_dictionary() / check_resumed_data() / setup_training_paths()
│       │       The classes of a resumed run kept from its label_dictionary.json, a resume on changed labels refused,
│       │       then the run folder with the configuration and label_dictionary.json (classification_workflow.md,
│       │       Step 2.6).
│       ├── build_dataloaders()
│       │   ├── read_test_manifest()     the test labels.json: class_names, class_colours, file name of every test image
│       │   └── get_detection_dataloaders() -> get_detection_dataloader() for every split
│       │       └── CocoDetectionDataset() + build_native_loader(collate_function)      decoding on the CPU workers
│       ├── check_loaders_give_batches()                                  refuses a split whose loader gives no batch
│       ├── save_data_record()                                            data.json
│       ├── get_test_sample_classes() + choose_sample_identifiers()      empty class lists: evenly spread images
│       ├── build_model()
│       │   └── TwoStageDetector.__init__()
│       │       ├── build_backbone() + compute_detection_input_information()   feature maps of the detection blocks
│       │       ├── RegionProposal() for every level -> Anchors.generate_anchors()
│       │       ├── DetectionHead() for every level
│       │       ├── _aggregate_all_anchors()                                     all the anchors in one tensor
│       │       └── RegionProposalFilter()
│       ├── initialise_pretrained_components() / build_optimizer() / ExponentialMovingAverage() (when activated)
│       ├── get_loss_weights() / get_tracked_metrics() + MetricTracker()
│       ├── print_experiment_information() -> describe_split() -> count_split_images_and_annotations()
│       ├── print_summary() / print_receptive_field_summary() / print_model_size_summary()
│       ├── diagnose_data_splits()   anchor recall of every split, then exit   (only with compute_anchor_coverage)
│       ├── view_input_batches() -> show_input_samples()   (only with view_input_images)
│       ├── analyse_training_data()   anchor recall of the training objects, anchor_recall.csv
│       │   └── measure_anchor_recall(): gather_object_boxes() + get_level_anchor_generators() +
│       │       compute_best_anchor_overlaps() + summarise_anchor_recall()
│       └── restore_last_model()
│
└── trainer.run_training_loop()  [STAGE 3]
    └── run_training_iterations()
        ├── for each training_iteration, from start_iteration to number_iterations (the total of the run):
        │   ├── get_next_batch() -> CocoDetectionDataset.__getitem__() + collate_function()
        │   │       images [B, 3, H, W] uint8, bounding_boxes and labels as lists (a different number per image); a
        │   │       batch failing to load is dropped and the next one taken
        │   ├── run_model_iteration()
        │   │   ├── prepare_batch() / augment_batch()   the geometric augmentations move the boxes
        │   │   ├── forward_pass() -> TwoStageDetector.forward(ground_truth_bounding_boxes=..., ground_truth_labels=...)
        │   │   │   │   -> forward_detection_stages(), which runs every stage below
        │   │   │   ├── Backbone.forward()                                   feature maps of the detection blocks
        │   │   │   ├── _forward_region_proposals() -> RegionProposal.forward()  objectness and boxes of every anchor
        │   │   │   ├── aggregate_proposals_and_anchors()
        │   │   │   ├── RegionProposalFilter.forward() -> select_best_proposals()   top-k, NMS, padded to a fixed number
        │   │   │   ├── _prepare_detector_inputs()
        │   │   │   │       add_ground_truth_proposals() + batch_assign_targets_to_proposals() +
        │   │   │   │       turn_boxes_to_transformation_targets() + sample_positive_and_negative_training_targets(),
        │   │   │   │       every image of the batch at once
        │   │   │   ├── _forward_detections() -> DetectionHead.forward()     RoIAlign, class logits, box regressions
        │   │   │   ├── aggregate_detection_outputs() + _rearrange_predictions_by_batch()
        │   │   │   ├── compute_region_proposal_losses()
        │   │   │   │       batch_assign_targets_to_anchors(), sampling, binary cross entropy + smooth L1
        │   │   │   └── compute_detection_losses()                           cross entropy + smooth L1
        │   │   ├── compute_batch_metrics()   none: the four losses are the tracked metrics
        │   │   └── MetricTracker.accumulate() + log_to_tensorboard()
        │   ├── training_loss.backward() / step_optimizer()
        │   ├── save_model() on the multiples of weight_saving_iterations and intermediate_weight_saving_iterations
        │   ├── validation pass
        │   ├── average_trackers() + log_interval_metrics() + print_interval_summary() + report_dropped_batches()
        │   └── run_test_evaluation() on the multiples of weight_saving_iterations, last    -> [STAGE 4]
        └── save_model() of the steps taken since the last checkpoint, if any, then shutdown_loader_workers() and
                finish_experiment_tracking()

run_test_evaluation(iteration)  [STAGE 4]
└── evaluate_test_split(), for the trained weights, then again within use_averaged_weights() for the moving average
    │   of the weights (Optimization.exponential_moving_average on: Evaluation/Iteration_<n>/EMA/, Tensorboard/Test_EMA)
    ├── evaluation_context()                the score threshold of the model lowered to evaluation.score_threshold
    ├── create_evaluation_accumulator()     the detections and ground truths of every test image, the confusion matrix
    ├── for each test batch:
    │   ├── prepare_batch() / predict_batch()   TwoStageDetector.forward() without ground truth ->
    │   │                                       postprocess_detections()
    │   ├── accumulate_evaluation_batch()       for every image:
    │   │   ├── gather_image_detections() + gather_image_ground_truth()        valid detections and objects, on the CPU
    │   │   ├── accumulate_proposal_recall()     best overlap of every object with the valid proposals
    │   │   ├── compute_overlap_matrix() + match_image_detections() + add_to_confusion_matrix()   the detections
    │   │   │                                    reaching the deployment threshold
    │   │   └── dump_errors() -> group_image_errors() + draw_test_image()      Errors/ (error_dumps on)
    │   └── save_sample_images() -> build_sample_image() -> draw_test_image() -> build_detection_sample_grid()
    ├── compute_evaluation_results()
    │   ├── summarise_average_precisions()
    │   │   ├── evaluate_detections()
    │   │   │   ├── gather_class_detections() -> find_best_ground_truth()         for every class, once
    │   │   │   ├── compute_class_true_positive_flags() -> match_detections_to_ground_truth()   for every IoU threshold,
    │   │   │   │                                                                                 once
    │   │   │   ├── for every method (area, 11 point, 101 point) and IoU threshold (0.50 to 0.95):
    │   │   │   │   compute_mean_average_precision() -> compute_precision_recall() + compute_average_precision()
    │   │   │   ├── for every object size (small, medium, large), the same with select_object_size_detections()
    │   │   │   └── find_class_score_thresholds() -> find_score_threshold()       at IoU 0.5, for every class
    │   │   ├── build_class_average_precision_rows() -> compute_clopper_pearson_intervals()   recall_50 and recall_75
    │   │   │                                    with their exact 95% interval over the ground truths of the class
    │   │   └── build_mean_average_precision_table() + build_object_size_table() + build_average_recall_table() +
    │   │       build_class_average_precision_table() + build_recall_interval_table()
    │   ├── save_proposal_recall() -> summarise_proposal_recall() + write_class_metrics_file()   proposal_recall.csv
    │   ├── write_class_metrics_file()                                                     class_metrics.csv
    │   └── save_error_analysis()      class_thresholds.json, confusion_matrix.csv and .png, the notes of the report
    └── report_evaluation() -> report_evaluation_results() + track_evaluation()

scripts/evaluation/evaluate_checkpoint.py  [STAGE 5]
└── same as every task: evaluate_checkpoints() -> start_evaluation_tracking() + summarise_checkpoint_evaluations()
    -> run_test_evaluation() for every checkpoint, then Evaluation/evaluation_summary.csv
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below are real outputs from running the loader on `vision_training/configurations/object_detection/configuration_synthetic_shapes.yaml` as it was then, its data folders and mapping file redirected to the generated dataset and the DALI loader of that time turned off.

### Step 1.1 — `main()`
The entry point is the same for every task (see `classification_workflow.md`, Step 1.1).

### Step 1.2 — `load_experiment_configuration()`
The model sections of the detection are `Backbone`, `DetectionLevels` (keyed by the backbone blocks whose feature maps feed the detection, every level holding the scales and aspect ratios of its anchors, its region proposal head and its detection head), `RegionProposal` (the settings shared by the region proposal heads: the IoU thresholds labelling the anchors, the proposals kept before and after the non-maximum suppression) and `Detector` (the settings shared by the heads classifying and refining the proposals, the post-processing of the inference). The `evaluation` section has its own `score_threshold`, the lowest score of the detections kept for the mean average precision, the `target_precision` of the score threshold of every class (Step 4.12) and the `error_dumps` of the error images (Step 4.5). `Optimization.loss_weights` gives the weight of every loss of the detector in the total loss (Step 3.9).

**Example: the sections and the detection settings**
```text
model_configuration keys: Optimization, Data, Backbone, DetectionLevels, RegionProposal, Detector
evaluation:               {'per_class_report': True, 'number_sample_images': 8, 'score_threshold': 0.05, 'target_precision': 0.99,
                           'error_dumps': {'activated': False, 'maximum_per_class_pair': 20}}
Data keys after loading:  image_settings, label_mapping, data_augmentation, label_mode, label_dictionary
DetectionLevels:          {"5": {"anchors": {"scales": [32, 64, 128], "aspect_ratios": [0.5, 1.0, 2.0]}, "region_proposal": {"convolutions": [2, 64]},
                                 "detector": {"roi_align_pool_size": 7, "convolutions": [2, 64], "fully_connected": [256, 128]}},
                           "6": {"anchors": {"scales": [64, 128, 256], ...}, ...},
                           "7": {"anchors": {"scales": [128, 256, 512], ...}, ...}}
RegionProposal:           {"normalization": {"feature_map_normalization": "batch"},
                           "training": {"pre_nms_proposals": 1000, "post_nms_proposals": 500}, "inference": {"pre_nms_proposals": 500, "post_nms_proposals": 250},
                           "foreground_iou_threshold": {"min": 0.5, "max": 1.0}, ...}
Detector:                 {"normalization": {"feature_map_normalization": "batch"}, "modules": {...}, "inference": {...}, ...}
```

### Step 1.3 — `load_configuration()`
**Example output**
```text
top-level keys:    ['ExperimentConfiguration', 'Optimization', 'Data', 'Backbone', 'DetectionLevels', 'RegionProposal', 'Detector']
number_iterations: '1e10'      <- still a string
learning_rate:     0.0002
```

### Step 1.4 — `build_label_dictionary()`
The mapping file gives the class of every category of the manifests, from 1; the detector predicts the background as class 0, so `'background': 0` is added.

**Example: the label dictionary of the synthetic shapes**
```text
label_mapping: {'mode': 'multi_class', 'mapping_file_path': '<datasets>/synthetic_shapes/label_mapping.json'}
dict: {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 'background': 0}
10 entries, 10 distinct classes
```

### Step 1.5 — `check_configuration()`
The common checks run first, then `check_detection_configuration()`, which refuses the augmentation types moving the image only, `random_perspective` and `random_jigsaw` (`check_image_only_augmentation_types()`), since the boxes cannot follow them, checks that the classes of the label mapping are indexed 0 (background) to K-1 (`check_detection_data_configuration()`) and checks the weights of the four losses (`check_loss_weights_configuration()`). Two of its checks keep the target assignment consistent: an anchor cannot be both background and foreground, and the detector samples its training targets among the proposals kept by the filter.

**Example: output with `RegionProposal.background_iou_threshold.max: 0.6` and `Detector.total_training_samples: 600` in the YAML**
```text
-----------------------------------------------------------------------------------------------
Invalid Configuration File : file:///.../configuration_wrong.yaml
-----------------------------------------------------------------------------------------------
  - 'RegionProposal.background_iou_threshold.max' (0.6) must be <= 'RegionProposal.foreground_iou_threshold.min' (0.5), otherwise anchors can be labelled both background and foreground.
  - 'Detector.total_training_samples' (600) must be <= 'RegionProposal.training.post_nms_proposals' (500), since the detector samples its targets among these proposals.
SystemExit code 1
```
The anchor scales are also checked against the image size: at 256 pixels, this configuration is refused with `'DetectionLevels.7.anchors.scales' must be integers within [1, 256], got [128, 256, 512].`

### Step 1.6 — `get_trainer_class()`
**Example output**
```text
get_trainer_class(task="object_detection") -> <class 'vision_training.trainers.detection_trainer.DetectionTrainer'>
```


---

## Stage 2 — Trainer Initialization

> The examples below are real outputs from building the trainer on the redirected `configuration_synthetic_shapes.yaml` (1024 pixel images, batch size 40), with the run folder redirected to a temporary folder. The whole construction took 2.1 s on CPU.

### Step 2.1 — `DetectionTrainer.__init__()`
Before calling the shared constructor, the trainer reads two score thresholds: `evaluation.score_threshold`, the lowest score of the detections kept for the mean average precision (low, so that the precision-recall curves keep their low-confidence end), and `Detector.inference.score_threshold`, the deployment one, used to draw the detections of the sample images and to count the detections of the confusion matrix and of the error images. It also reads the `target_precision` of the class thresholds and the `error_dumps` settings.

**Example: the attributes after the construction**
```text
self.evaluation_score_threshold = 0.05
self.deployment_score_threshold = 0.5
self.target_precision           = 0.99
self.error_dumps_configuration  = {'activated': False, 'maximum_per_class_pair': 20}
TASK_NAME = 'object_detection'   SAMPLE_NAME = 'Images'   COUNTS_FRAMES = False
```

### Step 2.2 — `BaseTrainer.__init__()`
**Example: some attributes set by the constructor**
```text
self.device = device(type='cpu')   self.per_class_report = True   self.number_sample_images = 8   self.input_image_size = 1024
self.batch_size = 40   self.start_iteration = 1
```

### Step 2.3 — `get_device()`
**Example output** (forced on the CPU; a normal run on this machine prints `cuda`)
```text
For Training We Will Be Using device: cpu
```

### Step 2.4 — `get_run_name()` → `build_run_name()`
The input and backbone tokens are those of every image task. `build_detection_sections()` adds `L567` (the detection blocks), `A9.32-512.598295` (9 anchors per position, scales from 32 to 512, then the first 6 characters of the SHA-256 of the scales and aspect ratios of every level, so that other ratios or middle scales start a new run), `RBN-R2x64` (the region proposal heads: batch normalization, 2 layers of 64 filters), `DSE-DRES-DBN-D7.2x64.256.128` (the detection heads: squeeze-and-excitation, residual, batch normalization, RoIAlign of 7, 2 layers of 64 filters, fully connected 256 and 128) and `MC9` (9 foreground classes, the categories merged into one class counting once, `BIN` in binary mode). A setting that differs between the levels is written for every level, joined with `+`.

**Example output**
```text
DET_IN-KAR-1024_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512.598295_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128_MC9
```

### Step 2.5 — `DataAugmenter()`
The same augmenter as the classification; the geometric augmentations move the boxes and drop those leaving the image (Step 3.8). This configuration turns every type off except `random_haze`, a photometric type that never moves the boxes.

**Example: the settings of the configuration**
```text
activated True   probability 0.4
1 of 28 types activated: random_haze
```

### Step 2.6 — `setup_training_paths()`
As for every task (see `classification_workflow.md`, Step 2.6): a resumed run first keeps the classes of its `label_dictionary.json` (`restore_run_label_dictionary()`) and is refused when the `labels.json` of a split changed since its `data.json` (`check_resumed_data()`, unless `allow_data_change_on_resume`), then the run folder is created with the copy of the configuration and `label_dictionary.json`, the class index of every category, background included. The loaders built next are refused when one of them gives no batch (`check_loaders_give_batches()`), then `save_data_record()` writes `data.json`, the classes and the images of every split.

**Example: the run folder after this step**
```text
DET_IN-KAR-1024_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512.598295_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128_MC9/
├── Tensorboard/
├── Weights/
├── label_dictionary.json          the class index of every category of the run
└── training_configuration.yaml    5,613 bytes
```

### Step 2.7 — `build_dataloaders()`
The hook first calls `read_test_manifest()` of `CocoSplitTrainer`, the base class shared with the keypoint detection, which reads the `labels.json` of the test split for three things: `class_names` (the categories named after the label dictionary, the background keeping its name), `class_colours` (one BGR colour per class, for the boxes of the sample images), and the file name of every test image, which names the sample images. Then it builds the loaders (Step 2.8).

**Example output**
```text
class_names = {0: 'background', 1: 'square', 2: 'circle', 3: 'triangle', 4: 'rectangle', 5: 'pentagon', 6: 'hexagon',
               7: 'plus_sign', 8: 'star', 9: 'octagon'}
test_image_file_names: 20 entries, e.g. {1: 'test_000000.png', 2: 'test_000001.png', 3: 'test_000002.png'}
```

### Step 2.8 — `get_detection_dataloaders()` → `get_detection_dataloader()`
For every split, `<folder>/data/` holds the images and `<folder>/labels.json` the COCO manifest. A `CocoDetectionDataset` (Step 2.9) decodes and resizes the images on the CPU loader workers and is wrapped in the PyTorch loader of `build_native_loader()` with `collate_function()`, which stacks the images and keeps the boxes and labels as lists, every image holding its own number of boxes. The test loader alone can also give the objects at the original resolution (`load_original_instances`), which only the instance segmentation evaluated with `evaluation.original_resolution` asks for.

**Example: size of the three splits**
```text
train       100 images -> 3 batches of 40 (one epoch = 3 iterations)
validation   20 images -> 1 batch of 40
test         20 images -> 1 batch of 40
collate      <function collate_function>
```

### Step 2.9 — `CocoDetectionDataset.__init__()` (and `__getitem__()`)
The dataset reads the manifest, shuffles its images once and groups the annotations by image. In `multi_class` mode it prints in yellow the manifest categories absent from the label mapping, `Categories Absent From The Label Mapping, Their Objects Left Out : <name> (<id>), ... (<labels.json>)`. `__getitem__()` reads the image with `read_image()` (OpenCV, BGR, `uint8`, a `ValueError` naming a file that cannot be read), keeps with `select_image_objects()` the boxes whose category has a class (an unmapped category gives the background and is dropped, the crowd annotations being kept like the other objects), turns the COCO `[x, y, width, height]` into `[x_min, y_min, x_max, y_max]`, and resizes the image and its boxes together (`preprocess_image_and_boxes()`, with a letterbox when `keep_ratio` and the image is not square). The boxes are neither clipped nor filtered here, only after a geometric augmentation (`suggestions.md`, audit L8).

**Example: a manifest, the first image after shuffling, and the sample the loader gets from it**
```text
labels.json:  images[0]      {"id": 1, "file_name": "train_000000.png", "width": 512, "height": 512}        (100 images)
              annotations[0] {"id": 1, "image_id": 1, "category_id": 1, "bbox": [301, 211, 72, 72], "area": 5184, "iscrowd": 0}   (322 annotations)
              categories[0]  {"id": 1, "name": "square", "supercategory": "shape"}                           (9 categories)
first image:  {'id': 49, 'file_name': 'train_000048.png', 'width': 512, 'height': 512}
              bbox [[109, 169, 86, 58], [427, 235, 32, 70]]   category_id [3, 4]
dataset[0]:   images (3, 1024, 1024) torch.uint8
              bounding_boxes (2, 4) torch.float32 [[218.0, 338.0, 390.0, 454.0], [854.0, 470.0, 918.0, 610.0]]   <- x 2, from 512 to 1024 pixels
              labels (2,) torch.int64 [3, 4]
              sample_identifiers 49
```

### Step 2.10 — `build_native_loader()`
The same PyTorch `DataLoader` as for the classification (see `classification_workflow.md`, Step 2.10).

**Example output** (recorded with 8 workers, when the number of workers was a setting; every loader now has 4)
```text
len(loader) = 3   batch_size = 40   num_workers = 8   prefetch_factor = 2   pin_memory = True   persistent_workers = True
```

### Step 2.11 — `get_test_sample_classes()` + `choose_sample_identifiers()`
The hook gives every test image an empty class list, so `choose_sample_identifiers()` puts them in one group and takes `number_sample_images` images evenly spread over the sorted COCO identifiers. They are named after their file (`get_test_sample_name()`).

**Example output** (8 images among 20)
```text
get_test_sample_classes() -> 20 entries, e.g. {1: [], 2: [], 3: [], 4: []}
sample_image_identifiers  -> [1, 4, 6, 9, 12, 15, 17, 20]
names                     -> test_000000, test_000003, test_000005, test_000008, test_000011, test_000014, test_000016, test_000019
```

### Step 2.12 — `build_model()` → `TwoStageDetector.__init__()`
The two-stage detector: the backbone (Step 2.13), a region proposal head on every detection level that scores the anchors and proposes boxes (Step 2.14), a filter keeping the best proposals (Step 2.16), and a detection head on every level that classifies the proposals and refines their boxes (Step 2.15). It reads the post-processing settings of the inference: `Detector.inference.score_threshold`, `maximum_detections` and `Detector.nms_iou_threshold`.

**Example: the components of the model**
```text
model.backbone                    Backbone                 2,245,216 parameters
model.region_proposer_dictionary  ModuleDict                 495,239 parameters      levels 5, 6, 7: 113,965 + 190,637 + 190,637
model.detector_heads_dictionary   ModuleDict               2,847,894 parameters      levels 5, 6, 7: 924,722 + 961,586 + 961,586
model.region_proposal_filter      RegionProposalFilter             0 parameters
model.number_classes = 10                                                             <- the background included
model.detection_score_threshold = 0.5   model.maximum_detections = 100   model.detection_nms_iou_threshold = 0.8
```

### Step 2.13 — `build_backbone()` + `compute_detection_input_information()`
The shared backbone (see `classification_workflow.md`, Step 2.14). The detector takes the feature maps of the blocks numbered by the keys of `DetectionLevels` instead of the final pooled vector; `compute_detection_input_information()` computes their size and channels once.

**Example output**
```text
detection_input_information = {'5': {'feature_map_size': 32, 'number_channels': 64}, '6': {'feature_map_size': 16, 'number_channels': 128},
                               '7': {'feature_map_size': 16, 'number_channels': 128}}
Final Backbone output shape: [1, 128, 16, 16]
```
Block 7 has no max pooling (`last_max_pooling: False`), so blocks 6 and 7 have the same size.

### Step 2.14 — `RegionProposal.__init__()` → `Anchors.generate_anchors()`
Every detection level gets its anchors: for every position of its feature map, one box per scale and aspect ratio, the ratio being the height over the width (0.5 a wide box, 2.0 a tall box) for the same area. The anchors of a position are centred on its corner in the image, `(column x stride, row x stride)`, the stride being the image size over the feature map size (`suggestions.md`, audit L1: the backbone and RoIAlign place the cells at their centres). The region proposal head is a plain block of 3x3 convolutions (`DetectionLevels.<level>.region_proposal.convolutions`, without the modules of the detection heads), a 3x3 convolution back to the channels of the level followed by a ReLU, then two 1x1 convolutions: one objectness logit per anchor, and 4 box offsets per anchor. `_aggregate_all_anchors()` then concatenates the anchors of every level into `model.anchors`.

**Example: the anchors of the three levels, and those of the first position of level 5** (1024 pixel images; the 9 anchors of a position are stored ratio by ratio, shown here scale by scale)
```text
level 5: feature map 32, stride 32, 9 anchors per position, anchors (9216, 4)
level 6: feature map 16, stride 64, 9 anchors per position, anchors (2304, 4)
level 7: feature map 16, stride 64, 9 anchors per position, anchors (2304, 4)
model.anchors (13824, 4)
level 5, position 0:  scale 32: [-22.6, -11.3, 22.6, 11.3]  [-16.0, -16.0, 16.0, 16.0]  [-11.3, -22.6, 11.3, 22.6]     (ratios 0.5, 1.0, 2.0)
                      scale 64: [-45.3, -22.6, 45.3, 22.6]  [-32.0, -32.0, 32.0, 32.0]  [-22.6, -45.3, 22.6, 45.3]
                      scale 128: [-90.5, -45.3, 90.5, 45.3]  [-64.0, -64.0, 64.0, 64.0]  [-45.3, -90.5, 45.3, 90.5]
```

### Step 2.15 — `DetectionHead.__init__()`
Every level gets a detection head: `roi_align()` cuts a 7x7 feature map out of the level for every proposal (`spatial_scale` = feature map size / image size, `sampling_ratio=0`, `aligned=True`), a block of 3x3 convolutions with the modules of `Detector.modules` refines it, fully connected layers without normalization (`build_fully_connected_block()`) flatten it, and two linear layers give the logit of every class and 4 box offsets per class.

**Example: the head of level 5** (shortened)
```text
DetectionHead(
  (convolutional_block): ConvolutionBlock(   2 x [Conv2d(64, 64, 3x3), BatchNorm2d(64), ReLU], 2 x SqueezeAndExcitation(64 -> 32 -> 64) )
  (fully_connected_block): FullyConnectedBlock(Linear(3136, 256), ReLU, Linear(256, 128), ReLU)   <- 3136 = 7 x 7 x 64
  (classifier): Linear(in_features=128, out_features=10, bias=True)
  (bounding_box_regressor): Linear(in_features=128, out_features=40, bias=True)               <- 4 offsets for each of the 10 classes
)
```
The heads of levels 6 and 7 read 128 channels instead of 64, hence their 36,864 more parameters (the first 3x3 convolution). Their first convolution changing the channel count (128 to 64), the residual pairs of their 2 layers are shifted by one layer and no shortcut is added: these two heads compute like heads without `residual_nets` (`ConvolutionBlock.count_residual_additions()` gives 0, `bug_standards.md` H4, dropped).

### Step 2.16 — `RegionProposalFilter.__init__()`
The filter keeps the proposals the detection heads classify: the `pre_nms_proposals` best scored, then those surviving the non-maximum suppression at `nms_iou_threshold`, at most `post_nms_proposals`, with different numbers in training and in inference.

**Example output**
```text
training 1000 -> 500   inference 500 -> 250   nms_iou_threshold 0.5
```

### Step 2.17 — `initialise_pretrained_components()` + `build_optimizer()` + `get_tracked_metrics()`
The backbone, the region proposal heads and the detection heads can come from other runs (their `pretrained` settings). The detection heads may have more classes than their source run: the source classes must be the first ones of this run (`class_names` of both `data.json` files), and `extend_class_layers()` copies the classification and box regression rows of the source classes into the first rows of the detection heads of this run, the new classes keeping the random rows these heads were built with. The mask heads of an instance segmentation run extend their mask prediction layer the same way. With `Optimization.exponential_moving_average.activated`, an `ExponentialMovingAverage` copy of the model is made once the transfers are loaded. `get_loss_weights()` gives `Optimization.loss_weights`, and the four losses of the detector are tracked with their weighted sum.

**Example output** (the paths recomputed with the current code from the sections of the configuration; the older run of the examples had only the `Backbone` entry)
```text
pretrained_checkpoint_paths = {'Backbone': None, 'RegionProposal': None, 'Detector': None}
Adam, 1 parameter group, base_learning_rates [0.0002], weight decay 0.0
{'total_loss': 'Total Loss', 'region_proposal_classification_loss': 'RPN Classification Loss',
 'region_proposal_localisation_loss': 'RPN Regression Loss', 'detection_classification_loss': 'Detection Classification Loss',
 'detection_regression_loss': 'Detection Regression Loss'}
```

### Step 2.18 — `print_experiment_information()` (→ `describe_split()` → `count_split_images_and_annotations()`)
**Example output**
```text
- Run Name                : DET_IN-KAR-1024_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512.598295_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128_MC9
...
- Train Data Folder       : file://<datasets>/synthetic_shapes/train (100 images, 322 annotations)
- Validation Data Folder  : file://<datasets>/synthetic_shapes/validation (20 images, 78 annotations)
- Test Data Folder        : file://<datasets>/synthetic_shapes/test (20 images, 57 annotations)
- Iterations Per Epoch    : 3
- Batch Size              : 40
...
- MLflow Tracking         : Off
```

### Step 2.19 — `print_summary()` / `print_receptive_field_summary()` / `print_model_size_summary()`
The summary of the detector adds a region proposal panel (the anchors of every level) and a detection panel (the heads of every level); the receptive field panel marks the anchor scales of every detection block, to compare them with what a position of its feature map sees. The `Modules` line of a head lists the residual shortcuts only when its block adds one (`get_active_module_names()` -> `count_residual_additions()`): the heads of blocks 6 and 7, whose first convolution turns 128 channels into 64 and whose 2 layers then add no shortcut (Step 2.15), list the squeeze-and-excitation only.

**Example output** (shortened; the panels rebuilt on the CPU with the current code from the configuration of this stage)
```text
╭────────────── Region Proposal Architecture ──────────────╮
│ Convolution Block 5                                      │
│   - Feature Map Size : [32, 32]                          │
│   - Scales           : [32, 64, 128]                     │
│   - Aspect Ratios    : [0.5, 1.0, 2.0]                   │
│   - Total Anchors    : 9216                              │
│ ...                                                      │
│ Total Network Anchors: 13824                             │
╰──────────────────────────────────────────────────────────╯
╭─────────────────────────── Detection Architecture ───────────────────────────╮
│ Normalization Strategy: Batch                                                │
│                                                                              │
│ Detection Block 5                                                            │
│   - Modules            : Residual Shortcuts, Squeeze-and-Excitation          │
│   - Convolution Layers : 2                                                   │
│   - Fully Conn. Layers : [256, 128]                                          │
│   - Feature Map Size   : [32, 32]                                            │
│   - RoI Align Pool     : [7, 7]                                              │
│   - Spatial Scale      : 0.0312                                              │
│                                                                              │
│ Detection Block 6                                                            │
│   - Modules            : Squeeze-and-Excitation                              │
│   - Convolution Layers : 2                                                   │
│   - Fully Conn. Layers : [256, 128]                                          │
│   - Feature Map Size   : [16, 16]                                            │
│   - RoI Align Pool     : [7, 7]                                              │
│   - Spatial Scale      : 0.0156                                              │
│ ...                                                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────── Backbone Receptive Field ────────────────────────────────────╮
│ ...                                                                                              │
│ Block 5 - Max Pooling    : Output 32x32x64 | Stride 32 | Receptive Field 250                     │
│   - Detection Block 5 Anchor Scales : [32, 64, 128]                                              │
│ ...                                                                                              │
│ Block 7 - Convolution 6  : Output 16x16x128 | Stride 64 | Receptive Field 1434                   │
│   - Detection Block 7 Anchor Scales : [128, 256, 512]                                            │
│ ...                                                                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────── Model Size ───────────────────────╮
│ Backbone                                                 │
│   - Parameters           : 2,245,216                     │
│ ...                                                      │
│ Region Proposal Heads                                    │
│   - Parameters           : 495,239                       │
│ ...                                                      │
│ Detection Heads                                          │
│   - Parameters           : 2,847,894                     │
│ ...                                                      │
│ Total Model                                              │
│   - Parameters           : 5,588,349                     │
│   - Trainable Parameters : 5,588,349                     │
│   - Estimated Size       : 22.38 MB                      │
╰──────────────────────────────────────────────────────────╯
```

### Step 2.20 — `view_input_batches()` / `restore_last_model()`
With `view_input_images`, `show_input_samples()` shows an augmented training image next to a validation image, every box captioned with its class, each image followed by an information panel giving its number of objects (`Objects : 3`) above a table of its classes (row `Class`) over their numbers of objects (row `Count`), the table continuing on a new pair of rows when the next class does not fit the panel (`PanelTable` of `visualization.py`), and, for the training image, the augmentations drawn with their values and the objects they removed (`Removed objects : 1 (dog)`). The classes are named after the categories of the test manifest (`self.class_names`), `square`, `circle`, ... with the synthetic shapes mapping. A new run starts from scratch.

**Example output: a new run**
```text
No Initiation Model Weights Will Be Used...
We Generate A New Model That Will Be Trained From Scratch.            -> start_iteration = 1
```

### Step 2.21 — `analyse_training_data()` → anchor recall
Called by `BaseTrainer.__init__()` between `view_input_batches()` and `restore_last_model()` (Step 2.20), at every launch, resumes included, and never in a diagnosis mode that exits (`view_input_images`, `compute_anchor_coverage`). The hook of the general trainer writes nothing; the detection one measures how many training objects the anchors can learn. `measure_anchor_recall()` does the work: `gather_object_boxes()` reads the boxes of the training split from its annotations (`select_image_objects()`, the background left out, as in `__getitem__()`), placed in network input pixels by `preprocess_boxes()` without loading any image and without the augmentations. `compute_best_anchor_overlaps()` gives the overlap of the best anchor of every object in every level from the anchor generator of the level (`TwoStageDetector.get_level_anchor_generators()`): for one anchor shape, the intersection only shrinks as the anchor centre moves away from the object centre along x or y, so `Anchors.compute_best_overlaps()` takes the grid anchor nearest the object centre for every scale and aspect ratio, the exact best of all the anchors from one overlap per anchor shape, in float64 on the CPU. `summarise_anchor_recall()` counts the objects whose best anchor reaches `RegionProposal.foreground_iou_threshold.min`, in every level alone and in all of them, since the region proposal network assigns the objects to the anchors of all the levels at once. An object under the threshold everywhere only gets its single best anchor as foreground (the fallback assignment). The table is printed and `anchor_recall.csv` is written in the run folder, then sent to the `analysis` folder of the MLflow run by `start_experiment_tracking()` (Step 3.3), as `anchor_recall_resume_<n>.csv` for a resume.

With `diagnosis_modes.compute_anchor_coverage: { activated: True, iou_threshold: <threshold> }`, `diagnose_data_splits()` runs `measure_anchor_recall()` on the training, validation and test splits at `iou_threshold`, each with a progress bar over its images (`Gathering The <Split> Objects`), prints the three tables and exits before `analyse_training_data()`: no run folder is created and no file is written, the input viewer following when `view_input_images` is also on.

**Example output: the later run** (351 training objects)
```text
Anchor Recall Of The 351 Training Objects...
---------------------------------------------
Anchor Recall Of The Training Objects
 Objects               Count   Block 5   Block 6   Block 7   All Levels
────────────────────────────────────────────────────────────────────────
 All Objects             351    0.8974    0.5442    0.4160       0.8974
 Small (< 32²)             0         —         —         —            —
 Medium (32² to 96²)     178    0.7978    0.2978    0.0449       0.7978
 Large (>= 96²)          173    1.0000    0.7977    0.7977       1.0000
Share of the objects whose best anchor overlaps them by at least 0.5 (RegionProposal.foreground_iou_threshold.min),
sizes by box area in network input pixels. An object under RegionProposal.foreground_iou_threshold.min in every level
only gets its single best anchor as foreground. — : no object of the size.
```
```text
anchor_recall.csv
objects,count,recall_block_5,recall_block_6,recall_block_7,recall_all_levels
all,351,0.897436,0.544160,0.415954,0.897436
small,0,,,,
medium,178,0.797753,0.297753,0.044944,0.797753
large,173,1.000000,0.797688,0.797688,1.000000
```
Block 5 (scales 32 to 128) gives a foreground anchor to every large object, while 36 of the 178 medium objects reach 0.5 with no anchor of any level and are only learnt through their best anchor. The blocks 6 and 7, whose smallest anchors are 64 and 128 pixels, cover few medium objects.


---

## Stage 3 — Training Loop

> The outputs of the run below (Steps 3.1 to 3.5, 3.19, 3.22 and 3.23) are real outputs of the 5000-iteration run of [`run_configurations/object_detection_workflow_run.yaml`](run_configurations/object_detection_workflow_run.yaml), trained on the GPU on 2026-10-24. The commands of `Run It` launch the synthetic configuration, which differs only by its 2000 iterations, its checkpoint evaluated every 250 iterations (an intermediate one every 125), its MLflow tracking turned off and its experiment folder. The run name is `DET_IN-KAR-512_SE-RES-BN-B1x16.2x16.2x16.3x32.3x32.5x32.5x32_L567_A6.32-256.e7b72c_RBN-R2x32_DSE-DRES-DBN-D7.2x32.64.32_MC20`; with batches of 64 images an epoch is 2000 iterations, a console summary is printed every 1000 iterations (`information_dump`) and no validation batch runs (`validation_iteration_ratio: 0`). The 5000 iterations with their 20 evaluations took 8 h 11 min, from 08:04 to 16:15, on a GPU that was not dedicated to the run (the INS run of `instance_segmentation_workflow.md` trained alongside from 15:19), so its throughput and interval times are not a speed reference.
>
> The examples of one batch going through the model (Steps 3.6 to 3.18, 3.20 and 3.21) come from the earlier run on CPU of the redirected configuration of Stages 1 and 2 shortened to 512 pixel images, batch size 2, 12 iterations, a checkpoint every 6 iterations, a console summary every 4 iterations, a validation batch every 10 iterations, with the MLflow tracking on. Its run name is `DET_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512.598295_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128_MC9` (9 classes, 64 and 128 filters, 9 anchors per position); at 512 pixels the feature maps of its detection blocks are 16x16, 8x8 and 8x8, so it has 2304 + 576 + 576 = 3456 anchors, against the 1536 + 384 + 384 = 2304 of the synthetic configuration.

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
The loop of every task (see `classification_workflow.md`, Step 3.1): from `start_iteration` to `number_iterations`, the total of the run, each iteration takes its training step, then saves its checkpoint, runs its validation batch and prints its summary on the iterations set for each, and evaluates its checkpoint last.

**Example: what happens at each iteration of the run**, in the order of the rows within an iteration, after its training step (the iterations between the columns only take their training step)

| Iteration | 250 | 500 | 750 | 1000 | 1250 | 1500 | ... | 4500 | 4750 | 5000 | after the loop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Checkpoint saved | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ... | ✔ | ✔ | ✔ | nothing: iteration 5000 is saved |
| Validation batch | | | | | | | | | | | |
| Console summary | | | | ✔ | | | ... | | | ✔ | |
| Test evaluation, trained then averaged weights | | ✔ | | ✔ | | ✔ | ... | ✔ | | ✔ | nothing |

### Step 3.2 — `setup_tensorboard_writers()` / `initialize_training_loop_state()`
**Example: the curves after the run** (no `Validation` writer, since no validation batch runs)
```text
Tensorboard/Train/        Total Loss, RPN Classification Loss, RPN Regression Loss, Detection Classification Loss,
                          Detection Regression Loss, Learning Rate                                   (iterations 1 to 5000)
Tensorboard/Test/         72 scalars and 9 images per evaluation (Step 4.13)                         (iterations 500 to 5000)
Tensorboard/Test_EMA/     the same tags for the averaged weights                                     (iterations 500 to 5000)
```

### Step 3.3 — `start_experiment_tracking()`
**Example output**
```text
- MLflow Tracking         : sqlite:///.../experiments/workflow_runs/mlflow.db                <- launch information
MLflow Run Created In The Experiment 'object_detection' : 8e53387f20e24434973626e4c0394a2c (Browse : mlflow ui --backend-store-uri sqlite:///.../experiments/workflow_runs/mlflow.db)
```
The MLflow run is created by a background thread. In this run it existed before the first checkpoint, so the 20 checkpoints hold its identifier; a checkpoint saved while the run is still being created holds `mlflow_run_id: None`. The configuration, `anchor_recall.csv` of Step 2.21 (folder `analysis`) and `data.json` are sent as soon as the run is open.

### Step 3.4 — `save_model()` → `save_checkpoint()`
Saved right after the training step of every multiple of 250 (the 10 evaluated checkpoints every 500 iterations and the 10 intermediate ones between them), so that the checkpoint of iteration N holds N steps (see `classification_workflow.md`, Step 3.6), with the label dictionary of the run and the source run of every transferable component of the task, None without transfer: `pretrained_backbone`, `pretrained_region_proposer_dictionary` and `pretrained_detector_heads_dictionary`. A run keeping a moving average of the weights, as the synthetic configuration does, also saves its state (`exponential_moving_average_state`).

**Example: the checkpoints and the keys of the last one**
```text
Checkpoint Folder :
 - Iteration_5000
 - Model Saved In : model_005000.pt

Weights/Iteration_250/model_000250.pt    10,708,317 bytes      Weights/Iteration_500/model_000500.pt    10,708,317 bytes
...
Weights/Iteration_4750/model_004750.pt   10,708,317 bytes      Weights/Iteration_5000/model_005000.pt   10,708,317 bytes
Weights/full-checkpoint:  model_checkpoint_path: "Iteration_5000", all_model_checkpoint_paths: "Iteration_250" ... "Iteration_5000"
['iteration', 'model_state', 'optimizer_state', 'format_version', 'task', 'run_name', 'label_dictionary',
 'mlflow_run_id', 'pretrained_backbone', 'pretrained_region_proposer_dictionary', 'pretrained_detector_heads_dictionary',
 'exponential_moving_average_state']
model_state: 327 tensors, the first 'backbone.convolution_blocks.0.squeeze_and_excitation_layers.0. ...'
exponential_moving_average_state: {'model_state', 'number_updates'}, number_updates 5000
mlflow_run_id: '8e53387f20e24434973626e4c0394a2c' in the 20 checkpoints
```

### Step 3.5 — `run_test_evaluation()`
At every multiple of 500, last in the iteration, after its checkpoint and its summary (see `classification_workflow.md`, Step 3.7), once for the trained weights and once for the moving average of the weights, see Stage 4.

### Step 3.6 — `get_next_batch()` → `CocoDetectionDataset.__getitem__()` + `collate_function()`
A batch whose loading fails (an image that `read_image()` cannot read, Step 2.9) is dropped and the next one taken (see `classification_workflow.md`, Step 3.8). The images are stacked; the boxes and labels stay lists, one tensor per image.

**Example: the first batch of the CPU run**
```text
images               [2, 3, 512, 512] torch.uint8
bounding_boxes       list of 2 tensors   [[3.0, 216.0, 164.0, 377.0], [110.0, 60.0, 249.0, 199.0]]
                                         [[242.0, 249.0, 408.0, 415.0], [172.0, 120.0, 272.0, 220.0], [15.0, 12.0, 57.0, 52.0], [399.0, 55.0, 500.0, 156.0]]
labels               list of 2 tensors   [7, 1]   [1, 1, 5, 1]
sample_identifiers   [2] torch.int64     [38, 52]
```

### Step 3.7 — `run_model_iteration()` → `prepare_batch()`
**Example: before and after**
```text
images [2, 3, 512, 512] torch.uint8 -> torch.float32     bounding_boxes: float32 tensors     labels: int64 tensors
```

### Step 3.8 — `augment_batch()` → `DataAugmenter.augment_batch(bounding_boxes=..., labels=...)`
The geometric augmentations move the boxes with the image, and `clean_bounding_boxes()` clips them to the image and drops with their labels the boxes left narrower or shorter than 1 pixel or with less than 25 % of their moved area inside the image (`MINIMUM_VISIBLE_AREA_FRACTION`), the fragments a crop or a rotation cuts at the border. This configuration turns every geometric type off, so the boxes never move; only `random_haze` can veil an image.

**Example output**
```text
images: changed per sample [False, False]
```

### Step 3.9 — `forward_pass()` → `TwoStageDetector.forward(ground_truth_bounding_boxes=..., ground_truth_labels=...)`
With the ground truth, the detector samples its training targets and returns its four losses (Steps 3.10 to 3.18). The total loss is their sum, each multiplied by its weight in `Optimization.loss_weights` (`get_loss_weights()`, 1 in the shipped configurations).

**Example: the dictionary returned at iteration 1 of the CPU run**
```text
classification_scores                 [2, 128, 10] torch.float32       <- the 128 proposals sampled for every image
box_regressions                       [2, 128, 10, 4] torch.float32
detection_boxes                       [2, 128, 10, 4] torch.float32
region_proposal_classification_loss   0.7244          <- close to ln 2 = 0.6931
region_proposal_localisation_loss     0.8107
detection_classification_loss         2.3117          <- close to ln 10 = 2.3026
detection_regression_loss             0.6198
total_loss                            4.4665          (TensorBoard, iteration 1)
```

> The examples of Steps 3.10 to 3.18 run the first batch of the CPU run step by step through the model at the end of that run (iteration 12), in training mode, so their values differ from those of iteration 1.

### Step 3.10 — `Backbone.forward()`
The backbone returns its final feature map and the feature maps of the detection blocks.

**Example output**
```text
final (2, 128, 8, 8)   feature maps {'5': (2, 64, 16, 16), '6': (2, 128, 8, 8), '7': (2, 128, 8, 8)}
```

### Step 3.11 — `_forward_region_proposals()` → `RegionProposal.forward()`
On every level, the region proposal head gives an objectness logit and 4 offsets `[dx, dy, dw, dh]` per anchor, flattened in the order of the anchors, and `apply_regression_predictions()` moves every anchor by its offsets into a proposal box: the centre by `dx` times the anchor width and `dy` times its height, the width and height times `exp(dw)` and `exp(dh)` (`dw` and `dh` being clamped so that the exponential stays finite). The offsets are detached for the boxes, so the proposals do not back-propagate.

**Example output**
```text
level 5: classification_scores (2, 2304)   bounding_box_regressions (2, 2304, 4)   proposal_boxes (2, 2304, 4)    <- 16 x 16 x 9
level 6: classification_scores (2, 576)    bounding_box_regressions (2, 576, 4)    proposal_boxes (2, 576, 4)     <- 8 x 8 x 9
level 7: classification_scores (2, 576)    bounding_box_regressions (2, 576, 4)    proposal_boxes (2, 576, 4)
```

### Step 3.12 — `aggregate_proposals_and_anchors()`
The outputs of the levels are concatenated, with the level of every anchor (`convolution_block_origin`) and the anchors repeated for every image.

**Example output**
```text
classification_scores (2, 3456)   bounding_box_regressions (2, 3456, 4)   proposal_boxes (2, 3456, 4)
convolution_block_origin (2, 3456)   anchors (2, 3456, 4)
objectness of image 0 (sigmoid): min 0.4179   max 0.5723   mean 0.4970
```

### Step 3.13 — `RegionProposalFilter.forward()`
For the whole batch at once, with `select_best_proposals()`, shared with the `InferenceDetector` of the deployment: the sigmoid of the objectness, the `pre_nms_proposals` best proposals of every image (1000 in training, all of them when there are fewer anchors), their boxes clamped to the image, the non-maximum suppression of every image at `RegionProposal.nms_iou_threshold` (0.5 in this run, 0.7 in the current configurations), then the `post_nms_proposals` best survivors of every image (500 in training, 250 in inference). The top-k and the suppression run over the anchors of all the levels together, so the many small anchors of block 5 can take most of the slots (`suggestions.md`, audit M3). The suppression is the only step run image by image (`suppress_overlapping_proposals()`, one torchvision `nms` call per image, its fastest form on the GPU), the exported detector running a single `batched_nms` call over the batch, which keeps the same proposals. An image with fewer survivors is padded with copies of random survivors, so that every image has the same number of proposals; the filter also returns the mask of the slots holding a real survivor, which the proposal recall of the evaluation reads (Step 4.15).

**Example output**
```text
filtered: boxes (2, 500, 4)   scores (2, 500)   origins (2, 500)
image 0: 500 distinct boxes, levels {5: 484, 6: 6, 7: 10}
image 1: 500 distinct boxes, levels {5: 466, 6: 22, 7: 12}
```

### Step 3.14 — `_prepare_detector_inputs()`
In training, `add_ground_truth_proposals()` first writes the ground truth boxes of every image into the last slots of its filtered proposals (the padding copies, then the least confident survivors), each at the level of the anchor overlapping it most, so that every object has a positive proposal with a zero regression target even when the region proposal network misses it. This happens whenever the ground truth is given, in the validation loss too; the proposals returned for the proposal recall and the test predictions, made without ground truth, are left as they are. `batch_assign_targets_to_proposals()` then gives every proposal the class of the ground truth box it overlaps most when their IoU reaches `Detector.foreground_iou_threshold` (0.5), and the background (0) otherwise; `turn_boxes_to_transformation_targets()` computes the offsets that would move the proposal onto its ground truth box; `sample_positive_and_negative_training_targets()` keeps, for every image, at most `number_training_positives` (64) foreground proposals and fills `total_training_samples` (128) with background ones, chosen at random. The kept proposals are written `[image index, x_min, y_min, x_max, y_max]` for `roi_align()`. In inference, every filtered proposal is kept.

**Example output** (recorded before `add_ground_truth_proposals()` existed, `a10bb7a`: the positives below are the proposals of the region proposal network alone; with the current code every object of the batch also gives a positive proposal)
```text
proposals (256, 5)   origins (256,)   labels (256,)   regression targets (256, 4)   batch indices (256, 1)
image 0: sampled labels {0: 125, 1: 1, 7: 2}       <- 3 foreground proposals: 1 square, 2 plus_sign
image 1: sampled labels {0: 124, 1: 3, 5: 1}
```

### Step 3.15 — `_forward_detections()` → `DetectionHead.forward()`
Every proposal goes to the detection head of the level it comes from: `roi_align()` cuts its 7x7 feature map, and the head gives the logit of every class and the offsets of every class, applied to the proposal to give one box per class (`detection_boxes`).

**Example output**
```text
level 5: classification_scores (244, 10)   box_regressions (244, 10, 4)   detection_boxes (244, 10, 4)   sliced_proposals (244, 5)
level 6: classification_scores (9, 10)     box_regressions (9, 10, 4)     detection_boxes (9, 10, 4)
level 7: classification_scores (3, 10)     box_regressions (3, 10, 4)     detection_boxes (3, 10, 4)
```

### Step 3.16 — `aggregate_detection_outputs()` + `_rearrange_predictions_by_batch()`
The outputs of the levels are concatenated, then grouped by image.

**Example output**
```text
aggregated: classification_scores (256, 10)   box_regressions (256, 10, 4)   sliced_labels (256,)   sliced_regression_targets (256, 4)
by image:   classification_scores (2, 128, 10)   box_regressions (2, 128, 10, 4)   detection_boxes (2, 128, 10, 4)
```

### Step 3.17 — `compute_region_proposal_losses()`
`batch_assign_targets_to_anchors()` labels every anchor of every image of the batch at once (`pad_ground_truths()` completes the ground truths of every image with empty boxes up to the largest number of objects of the batch) from its best IoU with a ground truth box: foreground from `RegionProposal.foreground_iou_threshold.min` (0.5, the `.max` being unused), background between `background_iou_threshold.min` and `.max` (0.1 to 0.3 in this run), ignored otherwise (between 0.3 and 0.5, and below 0.1; the current configurations set the minimum to 0.0, so that every anchor below 0.3 can be sampled as background, audit H1); the anchors overlapping a ground truth box the most are foreground whatever their IoU (above 0), each assigned to the ground truth box it overlaps most, so that almost every box has an anchor (`bug_standards.md` B40: a small box sharing its best anchor with a larger one gets none). `sample_positive_and_negative_training_targets()` keeps at most `number_training_positives` (128) foreground anchors per image, filled to `total_training_samples` (256) with background ones, drawn at random for all the images with one sort of random keys. The classification loss is the binary cross entropy of the objectness of the sampled anchors; the regression loss is the smooth L1 (beta 1/9, `LOCALISATION_LOSS_BETA`) between the predicted and target offsets of the sampled foreground anchors, summed and divided by their number.

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
The classification loss is the cross entropy of the class logits of the sampled proposals, the background ones teaching the detector to reject false positives. The regression loss is the smooth L1 (beta 1/9, `LOCALISATION_LOSS_BETA`) between the target offsets and the offsets predicted for the true class of every foreground proposal, summed and divided by their number (0 when no sampled proposal is foreground).

**Example output**
```text
losses: classification 1.0863   regression 1.4856
```

### Step 3.19 — `compute_batch_metrics()` + `MetricTracker.accumulate()` + `log_to_tensorboard()`
The detection has no batch metric (`{}`): the four losses, unweighted, and their weighted sum are the tracked metrics.

**Example: the Train curves** (one point per iteration, a few shown)
```text
Total Loss                     1: 4.7885, 2: 4.6802, 3: 4.5880, 1000: 0.8791, 2000: 0.6425, 3000: 0.4555, 4000: 0.3572, 5000: 0.3252
RPN Classification Loss        1: 0.6982, 2: 0.6976, 3: 0.6940, 1000: 0.0368, 2000: 0.0167, 3000: 0.0108, 4000: 0.0065, 5000: 0.0056
Detection Classification Loss  1: 2.9319, 2: 2.8732, 3: 2.8286, 1000: 0.6279, 2000: 0.4690, 3000: 0.3291, 4000: 0.2349, 5000: 0.2072
Detection Regression Loss      1: 0.4050, 2: 0.3901, 3: 0.3785, 1000: 0.1307, 2000: 0.0898, 3000: 0.0727, 4000: 0.0752, 5000: 0.0717
```
At iteration 1 the region proposal classification loss is close to ln 2 = 0.6931 and the detection classification loss close to ln 21 = 3.0445, the losses of untrained logits over 2 and 21 classes. The region proposal loss falls below 0.04 within 1000 iterations, while the detection classification loss is still falling at 5000, the classes being harder to tell apart than an object from the background.

### Step 3.20 — `training_loss.backward()` + `step_optimizer()`
**Example output: the CPU run** (`gradient_clipping_norm: null`, so the gradients are not clipped; the 5000-iteration run does not record the gradient norm)
```text
gradient norm: iteration 1: 42.3099   iteration 2: 40.7255   iteration 3: 33.9392        learning rate 0.0002 (constant)
```

### Step 3.21 — validation pass
The 5000-iteration run has no validation pass (`validation_iteration_ratio: 0`). In the CPU run, at iteration 10, a validation batch runs in `eval()` mode with its ground truth: the filter keeps the inference numbers of proposals (500, then 250), the ground truth boxes are written into them as in training (Step 3.14), the detector samples its targets among them and returns its four losses. In `eval()` mode the detections are also post-processed, which the validation does not use (`bug_standards.md` B42).

**Example output: the CPU run**
```text
Tensorboard/Validation at iteration 10: Total Loss 3.4132, RPN Classification Loss 0.6929, RPN Regression Loss 0.5775,
                                        Detection Classification Loss 2.1428, Detection Regression Loss 0.0000
```
The detection regression loss is 0: no proposal sampled from this batch reached an IoU of 0.5 with a ground truth box. This was recorded before the ground truth boxes were written into the proposals (`a10bb7a`); with the current code they are positive proposals of every batch holding objects, so this loss is no longer 0 (`suggestions.md`, audit H2).

### Step 3.22 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()`
**Example: the sums and the averages of the first interval** (the sums recomputed from the 1000 points of `Tensorboard/Train`, the averages and the MLflow values read from the console and the MLflow database)
```text
before: {'interval_batches': 1000, 'total_loss': 1251.5455, 'region_proposal_classification_loss': 136.1218,
         'region_proposal_localisation_loss': 183.2268, 'detection_classification_loss': 693.1488, 'detection_regression_loss': 239.0481}
after:  {'interval_batches': 1000, 'total_loss': 1.2515, 'region_proposal_classification_loss': 0.1361, 'region_proposal_localisation_loss': 0.1832,
         'detection_classification_loss': 0.6931, 'detection_regression_loss': 0.2390}
MLflow, step 1000: {'train/Total Loss': 1.2515, 'train/RPN Classification Loss': 0.1361, 'train/RPN Regression Loss': 0.1832,
                    'train/Detection Classification Loss': 0.6931, 'train/Detection Regression Loss': 0.239,
                    'train/throughput': 10.1993, 'train/interval_seconds': 6274.9516, 'learning_rate': 0.0002}
```

**Example output: iterations 1000 and 5000**
```text
========================================================================================================================
|   Iteration: 1000    |    Total Images: 64000    |   Interval Images: 64000   |  Throughput: 10.2 Images Per Second  |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   1.2515
Moving Average of Train RPN Classification Loss        :   0.1361
Moving Average of Train RPN Regression Loss            :   0.1832
Moving Average of Train Detection Classification Loss  :   0.6931
Moving Average of Train Detection Regression Loss      :   0.2390
------------------------------------------------------------------------------------------------------------------------
Interval Of 1000 Iterations Took 01:44:34.
========================================================================================================================
...
========================================================================================================================
|   Iteration: 5000    |    Total Images: 320000   |   Interval Images: 64000   |  Throughput: 34.5 Images Per Second  |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   0.3332  -0.0633 Best
Moving Average of Train RPN Classification Loss        :   0.0071  -0.0019 Best
Moving Average of Train RPN Regression Loss            :   0.0409  -0.0058 Best
Moving Average of Train Detection Classification Loss  :   0.2161  -0.0489 Best
Moving Average of Train Detection Regression Loss      :   0.0691  -0.0067 Best
------------------------------------------------------------------------------------------------------------------------
Interval Of 1000 Iterations Took 00:30:53.
========================================================================================================================
```
Without validation batch, the summaries have no `Valid` rows and no `Validation Iterations In This Interval` line. Every average of the summaries of iterations 2000 to 5000 is lower than all the previous ones, so all their rows are marked `Best`. The interval time is the wall-clock time since the previous summary, so it includes the evaluations run in between (those of iteration 500 for the first interval, the evaluation of the trained weights alone taking 21:32, Step 4.1) and depends on the load of the shared GPU: 01:44:34 for the first interval, 00:30:53 for the last one.

### Step 3.23 — end of the loop
The checkpoint of iteration 5000 is already saved and evaluated inside its iteration, so the loop saves nothing more and evaluates nothing (see `classification_workflow.md`, Step 3.25).

**Example output**
```text
Iteration 5000 · 20000 Test Images · EMA Weights · 2026-10-24 16:15 ───      <- the evaluation of iteration 5000 with the averaged
...                                                                               weights, last in the iteration
-----------------------------------------------------------
Last Checkpoint Saved At Iteration 5000. Exiting Training.
-----------------------------------------------------------
```


---

## Stage 4 — Test Evaluation of a Checkpoint

> The reports, files, sample images and scalars below (Steps 4.1, 4.5 to 4.7 and 4.10 to 4.15) are real outputs of the evaluations of the 5000-iteration run of Stage 3 ([`run_configurations/object_detection_workflow_run.yaml`](run_configurations/object_detection_workflow_run.yaml)), mostly of its last checkpoint, iteration 5000, on its whole test split (20,000 images, 69,682 boxes), with `per_class_report: True`, `number_sample_images: 8`, `score_threshold: 0.05`, `target_precision: 0.99`, the error images off and the MLflow tracking on. Every checkpoint of a multiple of 500 is evaluated twice, with the trained weights and with their moving average (`EMA Weights` blocks, `EMA/` folders). The synthetic configuration of `Run It` evaluates every 250 iterations up to 2000 instead. These outputs were captured before the per-class recall intervals were added (2026-11-01): `class_metrics.csv` now holds `recall_50`, `recall_50_lower`, `recall_50_upper`, `recall_75`, `recall_75_lower` and `recall_75_upper` after the average precisions, and with `per_class_report` the report shows a `Per Class Recall` table after `Per Class Average Precision (Area)` (Steps 4.11 and 4.13).
>
> The steps inside the test batches (the tensors of Steps 4.2 to 4.4, 4.8 and 4.9) come from the evaluation of iteration 12 of the CPU run of Stage 3 (20 test images, 57 boxes), whose detector finds no shape: its 10 class probabilities of a proposal are all close to 0.1. The error images of Step 4.5 come from the detection smoke test, the 5000-iteration run saving none.

### Step 4.1 — `run_test_evaluation()` → `evaluation_context()`
The evaluation loop of every task (see `classification_workflow.md`, Step 4.1). The `evaluation_context()` of the detection lowers the score threshold of the model from the deployment one (`Detector.inference.score_threshold`, 0.5) to `evaluation.score_threshold` (0.05) during the evaluation, so that the precision-recall curves keep their low-confidence end, and restores it afterwards, even when the evaluation fails.

**Example output: the progress bars of iterations 500 and 5000** (20,000 images, batch size 64)
```text
Test Evaluation Iteration 500: 100%|██████████| 313/313 [21:32<00:00,  4.13s/it]
...
Test Evaluation Iteration 5000: 100%|██████████| 313/313 [01:26<00:00,  3.62it/s]
Test Evaluation Iteration 5000 (EMA Weights): 100%|██████████| 313/313 [04:46<00:00,  1.09it/s]
```
The same 313 batches took 21:32 at iteration 500 and 01:26 at iteration 5000, on a GPU shared with other runs (Stage 3).

### Step 4.2 — `create_evaluation_accumulator()`
The detections and the ground truth of every test image, kept on the CPU in network input pixels, the confusion matrix of the classes (background included), filled image by image (Step 4.5), the folder and counts of the error images, and the `proposal_recall` accumulator of Step 4.15 (`create_proposal_recall_accumulator()`, added after the run of the example, whose accumulator below has no such entry).

**Example: the accumulator at the end of the loop of iteration 12 of the CPU run**
```text
{'detections': 20 entries {'boxes': (100, 4), 'scores': (100,), 'labels': (100,)}, 'ground_truths': 20 entries {'boxes', 'labels'},
 'confusion_matrix': (10, 10) int64, 'errors_folder': .../Evaluation/Iteration_12/Errors, 'error_counts': {}}
detections per image   [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
ground truths per image [1, 6, 1, 3, 6, 4, 4, 6, 1, 1, 2, 2, 1, 3, 1, 1, 3, 6, 3, 2]
first image: best scores [0.1091, 0.1091, 0.1091, 0.1091, 0.1091]   labels [9, 9, 9, 9, 9]   ground truth [[281.0, 184.0, 426.0, 336.0]] label [5]
```

### Step 4.3 — `prepare_batch()` + `predict_batch()`
`predict_batch()` runs the detector without the ground truth, as in deployment: with it, the detector would only classify the proposals sampled around the true boxes, which would inflate the mAP. In `eval()` mode, the filter keeps 250 proposals per image and the outputs go through `postprocess_detections()` (Step 4.4). The outputs also keep the filtered proposals (`proposal_boxes` [2, 250, 4], `proposal_levels` and `valid_proposal_mask` [2, 250]) for the proposal recall (Step 4.15).

**Example: the predictions of a test batch of the CPU run**
```text
classification_scores    [2, 250, 10] torch.float32
box_regressions          [2, 250, 10, 4] torch.float32
detection_boxes          [2, 250, 10, 4] torch.float32
final_detection_boxes    [2, 100, 4] torch.float32
final_detection_scores   [2, 100] torch.float32
final_detection_labels   [2, 100] torch.int64
number_detections        [2] torch.int64 [100, 100]
```

### Step 4.4 — `postprocess_detections()`
The softmax of the class logits of every proposal, the background left out, gives one candidate per proposal and foreground class, with its box clamped to the image. One `batched_nms()` call suppresses, per image and class, the boxes overlapping a better one by more than `Detector.nms_iou_threshold` (0.8 in the CPU run, 0.5 in the current configurations); the survivors reaching the score threshold are kept, the `maximum_detections` (100) best of every image returned, and the empty places padded with a score of 0, a label of 0 and an empty box. The eager training and evaluation give the suppression the candidates reaching the score threshold only (`suppress_candidates_only=True`): a box below the threshold never suppresses a better one, so the detections are the same, and the evaluation of a batch of 64 images (320000 candidates on the synthetic shapes, of which a few thousand reach 0.05) avoids the per-group loop torchvision runs above 100000 boxes on the GPU, the postprocessing of a batch taking 0.02 seconds instead of 7. The `InferenceDetector` gives it every candidate (`suppress_candidates_only=False`), without data-dependent Python control flow nor data-dependent sizes, so that the same function is exported to TorchScript and ONNX. The eager forward marks every one of the 250 proposals as valid (`torch.ones_like(...)` as `valid_proposal_mask`), the padding copies of Step 3.13 included, and relies on this NMS to remove them, while the `InferenceDetector` masks them (`bug_standards.md` B35).

**Example: the first training batch in inference mode, at the deployment threshold 0.5, at the end of the CPU run**
```text
number_detections [0, 0]
detection 0 of image 0: box [0.0, 0.0, 0.0, 0.0]   score 0.0000   label 0        <- padding
class probabilities of the first proposal of image 0: [0.1327, 0.0894, 0.0938, 0.093, 0.1029, 0.1006, 0.0877, 0.0958, 0.0952, 0.109]
```
No foreground class reaches 0.5, so the deployed detector of the CPU run returns nothing; at the evaluation threshold 0.05, every test image gets its 100 detections (Step 4.3), all of them `octagon`, the most probable foreground class (Step 4.8). At iteration 5000 of the 5000-iteration run, 40,479 detections of the 20,000 test images reach 0.5 (Step 4.12).

### Step 4.5 — `accumulate_evaluation_batch()` → `match_image_detections()` + `add_to_confusion_matrix()` + `dump_errors()`
For every image of the batch, the `number_detections` valid detections (boxes, scores, labels) and the ground truth boxes and labels are kept on the CPU for the mean average precision. The detections reaching the deployment threshold (`Detector.inference.score_threshold`), those the deployed detector returns, are then matched to the objects of the image for the confusion matrix. `compute_overlap_matrix()` gives the IoU of every detection with every object, whatever their classes, and `match_image_detections()` goes twice through the detections from the most to the least confident, each one taking the free object that overlaps it the most, at least by 0.5: first among the objects of its class, so that an object found with its class counts as found even when a detection of another class overlaps it more, then among the objects left of any class (the confusions). `add_to_confusion_matrix()` counts every matched detection in the cell of the class of its object and of its own class, every detection left without object in the background row (a false positive) and every object left without detection in the background column (missed). The mean average precision keeps its own matching, per class and down to `evaluation.score_threshold` (Step 4.9). At iteration 5000, 40,479 detections reach 0.5: 34,213 find an object of their class, 3,228 an object of another class and 3,038 no object (the background row), while 32,241 of the 69,682 test objects are found by no detection (the background column).

With `error_dumps.activated`, `dump_errors()` saves the image once in the folder of every kind of error it holds (`group_image_errors()`): `Errors/FALSE-POSITIVE/<class>` for the detections of a class finding no object, `Errors/MISSED/<class>` for the objects of a class found by no detection, and `Errors/<TRUE>-TO-<PREDICTED>` for the objects found with another class. `draw_test_image()` draws it like a sample image (Step 4.6), the boxes of these errors outlined with red dashes (an `Error` entry heading the legend) and the errors described at the start of the caption. The file is named after the image and the highest score of its detections in error (the image name alone for the missed objects), with at most `maximum_per_class_pair` images per folder, the first ones met.

**Example: the error images of the detection smoke test** (`tests/trainers/test_detection_trainer.py`: 4 test images of 256 pixels, 5 iterations, the deployment threshold lowered to 0.05 so that the detector returns detections, at most 2 images per folder; the 5000-iteration run keeps the error images off)
```text
Evaluation/Iteration_5/Errors/
├── FALSE-POSITIVE/circle/      test_000001_0.109.png  test_000002_0.109.png
├── FALSE-POSITIVE/pentagon/    test_000001_0.107.png  test_000002_0.107.png
├── FALSE-POSITIVE/rectangle/   test_000001_0.106.png  test_000002_0.106.png
├── FALSE-POSITIVE/square/      test_000001_0.106.png  test_000002_0.106.png
├── FALSE-POSITIVE/triangle/    test_000001_0.109.png  test_000002_0.109.png
├── MISSED/circle/              test_000000.png
├── MISSED/hexagon/             test_000001.png  test_000002.png
├── MISSED/octagon/             test_000001.png  test_000002.png
├── MISSED/pentagon/            test_000000.png
├── MISSED/rectangle/           test_000003.png
├── MISSED/square/              test_000002.png
└── plus_sign-TO-triangle/      test_000000_0.109.png

plus_sign-TO-triangle/test_000000_0.109.png (516 x 304 pixels = 2 x 256 + 4 by 24 + 256 + 24)
caption:  "plus_sign -> triangle | 4 ground truth boxes | 100 detections with score >= 0.05"
outlined: the plus_sign box of the Ground Truth panel and the triangle detection that found it in the Prediction panel
legend:   Error, square, circle, triangle, rectangle, pentagon
```

### Step 4.6 — `save_sample_images()` → `build_sample_image()` → `build_detection_sample_grid()`
Two titled panels side by side, the ground truth boxes and the detections reaching the deployment threshold (`Detector.inference.score_threshold`, so that the image shows what the deployed detector returns), every box captioned with its class (and score) in the colour of its class, under a caption bar counting both and above a legend bar naming the classes drawn.

**Example: `Evaluation/Iteration_5000/Samples/test_014285.png`** (1028 x 560 pixels = 2 x 512 + 4 by 24 + 512 + 24, one of the 8 images `test_000000`, `test_002857`, ..., `test_019999`)
```text
caption: "6 ground truth boxes | 4 detections with score >= 0.50"
panels:  Ground Truth (square, triangle, star, parallelogram, 2 nonagon boxes) | Prediction (square 0.98, triangle 0.58,
         star 0.96, nonagon 0.55)
legend:  square, triangle, star, parallelogram, nonagon
EMA/Samples/test_014285.png: "6 ground truth boxes | 2 detections with score >= 0.50" (square 0.97, star 0.97)
```
The trained weights find the square and the star with high scores, the triangle and the large nonagon just above 0.5, and miss the small nonagon and the thin parallelogram, which the prediction panel shows without a box. The averaged weights keep only the square and the star: at the deployment threshold they return fewer detections than the trained weights on the whole test split (32,506 against 40,479, Step 4.13), although their mAP is higher.

### Step 4.7 — `compute_evaluation_results()`
`summarise_average_precisions()` runs `evaluate_detections()`, which computes the mean average precision of the test split with the three average precision methods (Steps 4.8 to 4.10) and of every object size (Step 4.14), and builds the rows of every class and the tables. `save_proposal_recall()` writes the proposal recall (Step 4.15), the average precisions of every class go to `class_metrics.csv` (Step 4.11), and `save_error_analysis()` saves the class thresholds and the confusion matrix (Step 4.12). The detection has no headline metric line (`metrics` is empty): the report shows the table of the methods, the table of the object sizes, the `Mean Average Recall` table, the proposal recall, the per-class table and the per-class recall table. The checkpoints are compared on the `mAP@0.5/area` scalar.

**Example: the results of iteration 5000** (gathered from the report, the links and the `Tensorboard/Test` scalars of the evaluation)
```text
evaluated_samples = '20000 Test Images'
metrics           = {}
notes             = ['Confusion Matrix : score >= 0.50, IoU >= 0.5 (boxes), 34213 right, 3228 wrong class, 3038 false positives, 32241 missed',
                     'Sample Images : 8 fixed test images']
headline_metric   = 'mAP@0.5/area'
files             = ['Class Metrics', 'Proposal Recall', 'Class Thresholds', 'Confusion Matrix Table', 'Confusion Matrix Image', 'Sample Images']
images            = ['Confusion Matrix']
scalars           = 52: {'mAP@0.5/area': 0.7151, 'mAP@0.75/area': 0.6451, 'mAP@0.95/area': 0.0021, 'mAP@[0.5:0.95]/area': 0.5160, ...,
                    'mAP@0.5 By Size/small': 0.0155, ..., 'mAR@0.5/all': 0.9801, ..., 'Proposal Recall@0.5/all': 0.9967, ...}
class_scalars     = 20: AP@0.5/square, AP@0.5/circle, ..., AP@0.5/l_shape
tables            = Mean Average Precision: Method, mAP@0.5, mAP@0.75, mAP@0.95, mAP@[0.5:0.95]
                    Mean Average Precision By Object Size (Area): Object Size, Objects, mAP@0.5, mAP@0.75, mAP@0.95, mAP@[0.5:0.95]
                    Mean Average Recall: Objects Of, Objects, mAR@0.5, mAR@0.75, mAR@0.95, mAR@[0.5:0.95]
                    Proposal Recall: Objects, Count, Recall@0.5, Recall@0.7, Proposals Per Image
                    Per Class Average Precision (Area): Class, Ground Truths, AP@0.5, AP@0.75, AP@0.95, AP@[0.5:0.95],
                    Threshold @ P0.99, Recall @ P0.99
```

### Step 4.8 — `gather_class_detections()` → `find_best_ground_truth()`
For every foreground class, once for all the thresholds and methods: the detections of the class of every image, sorted by decreasing score, the best IoU of every detection with a ground truth box of the same class in the same image (0 when the image has none), and the number of ground truth boxes of the class.

**Example: the class with the most detections at iteration 12 of the CPU run**
```text
detections per class {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 2000}
class 9 (octagon): 2000 detections, 7 ground truths
   image of the first 10 detections [11, 11, 11, 11, 11, 11, 11, 11, 11, 11]
   best IoU of the first 10 detections [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
   largest best IoU 0.4701, detections with a best IoU >= 0.3: 7, >= 0.5: 0
```

### Step 4.9 — `match_detections_to_ground_truth()` + `compute_precision_recall()`
At one IoU threshold, the detections are visited from the most to the least confident: a detection is a true positive when its best ground truth reaches the threshold and was not taken by a more confident detection, a false positive otherwise (the PASCAL VOC rule). Every detection is then one point of the precision-recall curve: the precision and the recall if the score threshold were its score.

**Example: octagon at 0.5, iteration 12 of the CPU run**
```text
true positives 0 of 2000 detections
precisions of the first 5 detections [0.0, 0.0, 0.0, 0.0, 0.0]   recalls [0.0, 0.0, 0.0, 0.0, 0.0]   last point: precision 0.0, recall 0.0
```

### Step 4.10 — `compute_average_precision()` + `compute_mean_average_precision()`
The average precision of a class summarises its curve: `area`, the area under the curve once every precision is raised to the best precision at a higher recall (PASCAL VOC 2010 and later); `11_point`, the mean of the interpolated precision at the recalls 0, 0.1, ..., 1 (PASCAL VOC 2007); `101_point`, the same at 101 recall levels (COCO). The mAP of a threshold is the mean over the classes that have ground truth boxes; `evaluate_detections()` gives, for every method, the mAP at 0.5, at 0.75, at 0.95 (since `351a4d1`) and averaged over the 10 thresholds 0.50, 0.55, ..., 0.95 (`mAP@[0.5:0.95]`), with the same four values for every class. The matching is the PASCAL VOC rule of Step 4.9 at every threshold, so the `mAP@[0.5:0.95]` and the 101-point values are not those of pycocotools, which matches every detection to the best ground truth not taken yet (`suggestions.md`, audit L6, kept on purpose).

**Example: iteration 5000** (from `Results.txt` and `class_metrics.csv`)
```text
area:      mAP@0.5 0.7151   mAP@0.75 0.6451   mAP@0.95 0.0021   mAP@[0.5:0.95] 0.5160
11_point:  mAP@0.5 0.6878   mAP@0.75 0.6351   mAP@0.95 0.0112   mAP@[0.5:0.95] 0.5077
101_point: mAP@0.5 0.7132   mAP@0.75 0.6436   mAP@0.95 0.0030   mAP@[0.5:0.95] 0.5152
octagon (area): AP@0.5 0.5144   AP@0.75 0.4896   AP@0.95 0.0041   AP@[0.5:0.95] 0.3956
class_ground_truths {1: 3545, 2: 3566, 3: 3512, 4: 3494, 5: 3475, ..., 19: 3344, 20: 3528}      69,682 boxes in all
```
The three methods agree within 0.03. The 11-point values are lower at 0.5 and higher at 0.95: the interpolated precision at recall 0 counts for a whole eleventh of the mean, even for a class whose detections reach almost no recall at that IoU. The mAP drops from 0.7151 at IoU 0.5 to 0.6451 at 0.75 but to 0.0021 at 0.95: the boxes are mostly well placed, but almost never tight enough for an IoU of 0.95.

### Step 4.11 — `build_class_average_precision_rows()` + `write_class_metrics_file()`
The average precisions of every foreground class with the area method, with the score threshold of the class reaching the target precision and the recall it keeps (Step 4.12), empty when the precision is never reached, saved whatever the per-class report. Since the object sizes (Step 4.14), the AP@0.5 and AP@[0.5:0.95] of every size follow the average precisions of all the objects, empty for a size the class has no object of. Then come `recall_50` and `recall_75`, the share of the ground truths of the class matched by a detection kept for the evaluation at IoU 0.5 and 0.75, whatever its score (the per-class values of the `mAR@0.5` and `mAR@0.75` of Step 4.14), each followed by the bounds of its exact (Clopper-Pearson) 95% confidence interval over the ground truths of the class, independent objects (`compute_clopper_pearson_intervals()` of `vision_training/evaluation/recall_intervals.py`, `classification_workflow.md`, Step 4.14), empty for a class without ground truth.

**Example: `Evaluation/Iteration_5000/class_metrics.csv`** (the hexagon has no small test object; the rhombus never reaches the precision 0.99)
```text
class_index,class,ground_truths,average_precision_50,average_precision_75,average_precision_95,average_precision_50_95,average_precision_50_small,average_precision_50_95_small,average_precision_50_medium,average_precision_50_95_medium,average_precision_50_large,average_precision_50_95_large,threshold_at_target_precision,recall_at_target_precision
1,square,3545,0.969327,0.917248,0.006597,0.732593,0.000000,0.000000,0.934682,0.659275,0.984351,0.768852,0.895632,0.884062
2,circle,3566,0.912897,0.875714,0.000708,0.697367,0.000000,0.000000,0.795297,0.580532,0.952339,0.736320,0.703808,0.643298
...
6,hexagon,3471,0.820560,0.791622,0.006872,0.645796,,,0.701038,0.510686,0.882367,0.721177,0.829466,0.009219
...
10,rhombus,3436,0.362592,0.339679,0.000007,0.250137,0.004049,0.001619,0.365671,0.252234,0.366849,0.253416,,
...
20,l_shape,3528,0.732700,0.393951,0.000015,0.404265,0.076259,0.029538,0.729446,0.392496,0.739493,0.424238,0.730511,0.036281
```

### Step 4.12 — `find_class_score_thresholds()` + `save_error_analysis()`
With a target precision, `evaluate_detections()` also gives the score threshold of every class. From the matching of the AP@0.5 (Step 4.9), `find_score_threshold()`, the function of the class thresholds of the classification, keeps the lowest score whose detections of the class at or above it reach the target precision, i.e. the highest recall at this precision. The recall counts every ground truth box of the class, those found by no detection included, and only the detections down to `evaluation.score_threshold` can give a threshold. `save_error_analysis()` saves these thresholds in `class_thresholds.json`, in the format of the classification tasks (`null` for a class whose detections never reach the precision), and the confusion matrix of Step 4.5 in `confusion_matrix.csv`, rows the true classes and columns the predicted classes, the background first. With the per-class report, the confusion matrix is also drawn in `confusion_matrix.png` and logged to TensorBoard: its background cell is left blank, and its background row and column have no recall or precision. The report gets a note counting the outcomes of the detections and, with the error images on, a note counting the images and their folders.

At iteration 5000, 17 classes get a threshold and 3 never reach the precision 0.99 (`null`): the rhombus, the kite and the capsule. A threshold often keeps almost nothing: the trapezoid reaches 0.99 only above 0.4119, with a recall of 0.0003, while the plus_sign keeps a recall of 0.9198 above 0.6335.

**Example: the files of iteration 5000** (`class_thresholds.json` is written indented, shortened here)
```text
confusion_matrix.csv
true \ predicted,background,square,circle,triangle,rectangle,pentagon,hexagon,plus_sign,star,octagon,rhombus,parallelogram,trapezoid,kite,heptagon,nonagon,ellipse,semicircle,annulus,capsule,l_shape
background,0,536,159,418,282,133,28,71,271,13,0,0,0,0,0,19,167,7,91,140,703
square,117,3397,0,0,31,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
circle,515,12,3024,0,2,0,1,0,0,1,0,0,0,0,0,9,0,0,2,0,0
triangle,1075,0,0,2254,1,0,0,0,7,0,0,0,0,0,0,0,0,0,0,0,175
rectangle,222,63,0,0,3209,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
pentagon,542,0,0,0,1,2882,18,12,0,0,0,0,0,0,0,0,0,1,19,0,0
hexagon,905,1,0,0,0,207,2344,3,0,0,0,0,0,0,1,0,0,0,10,0,0
plus_sign,197,1,0,0,0,0,0,3240,4,0,0,0,0,0,0,0,0,0,0,0,0
star,192,0,0,0,0,0,0,6,3348,0,0,0,0,0,0,0,0,0,0,0,3
octagon,3068,6,93,0,2,2,22,0,0,59,0,0,0,0,135,63,0,0,4,0,0
rhombus,3341,0,0,21,1,4,0,1,1,0,0,63,0,0,0,0,0,0,0,0,4
parallelogram,3057,0,0,43,5,0,0,0,0,0,0,394,0,0,0,0,0,1,0,0,3
trapezoid,3446,0,0,39,4,12,0,12,2,0,0,10,0,0,0,0,0,1,4,0,16
kite,3402,0,0,29,0,19,0,7,1,0,0,3,0,0,0,0,0,0,3,0,7
heptagon,2199,1,2,0,0,9,430,1,0,1,0,0,0,0,815,0,0,0,8,0,0
nonagon,2349,3,400,0,1,1,11,0,0,46,0,0,0,0,13,706,0,0,3,0,0
ellipse,2207,0,0,0,17,2,2,0,0,0,0,1,0,0,1,0,713,124,2,339,0
semicircle,1944,0,0,0,5,4,0,0,0,0,0,1,0,0,0,0,3,1519,0,1,1
annulus,176,1,0,0,0,1,0,2,0,0,0,0,0,0,0,0,0,0,3281,0,0
capsule,2241,2,2,0,100,2,2,0,0,0,0,1,0,0,5,0,199,85,2,703,0
l_shape,1046,0,0,58,9,0,0,2,83,0,0,0,0,0,0,0,0,0,5,0,2325

class_thresholds.json
{"target_precision": 0.99, "classes": {"square": {"threshold": 0.8956319689750671, "recall": 0.8840620592383639},
 "circle": {"threshold": 0.7038077712059021, "recall": 0.6432978126752664}, ..., "rhombus": null, ..., "kite": null, ...,
 "capsule": null, "l_shape": {"threshold": 0.7305112481117249, "recall": 0.036281179138321996}}}

note:
  Confusion Matrix : score >= 0.50, IoU >= 0.5 (boxes), 34213 right, 3228 wrong class, 3038 false positives, 32241 missed
confusion_matrix.png (1852 x 1815 pixels)
```
The diagonal shows which shapes the detector has learnt at the deployment threshold: more than 3,000 right detections for the square, circle, rectangle, plus_sign, star and annulus, but none for the rhombus, the trapezoid and the kite, and 59 for the 3,454 octagons. Most objects of these classes are missed (3,341 of the 3,436 rhombuses, 3,446 of the 3,546 trapezoids, 3,402 of the 3,471 kites, 3,068 of the octagons) rather than confused: no detection reaching 0.5 finds them. The confusions are between similar shapes: 430 heptagons found as hexagons, 400 nonagons as circles, 339 ellipses as capsules, 199 capsules as ellipses and 63 rhombuses as parallelograms, of which only 394 of 3,503 are found. The background row counts 3,038 false positives, 703 of them `l_shape` detections.

### Step 4.13 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
With `per_class_report`, the `Per Class Recall` table (`build_recall_interval_table()`) follows the per-class table: the ground truths of every class, then `Recall @0.5` and `Recall @0.75`, each followed by `95% Lower` and `95% Upper`, sorted by the recall at 0.5, worst first.

**Example output: the block of iteration 5000** (the per-class table shortened; the same text, without the file lines, is appended to `Results.txt`)
```text
Iteration 5000 · 20000 Test Images · 2026-10-24 16:09 ──────────────────────────────────────────────────────────────────
  Confusion Matrix : score >= 0.50, IoU >= 0.5 (boxes), 34213 right, 3228 wrong class, 3038 false positives, 32241
  missed
  Sample Images : 8 fixed test images

  Mean Average Precision
   Method      mAP@0.5   mAP@0.75   mAP@0.95   mAP@[0.5:0.95]
  ────────────────────────────────────────────────────────────
   Area         0.7151     0.6451     0.0021           0.5160
   11 Point     0.6878     0.6351     0.0112           0.5077
   101 Point    0.7132     0.6436     0.0030           0.5152
  The mAP@0.5 of the area method compares the checkpoints of the run.

  Mean Average Precision By Object Size (Area)
   Object Size           Objects   mAP@0.5   mAP@0.75   mAP@0.95   mAP@[0.5:0.95]
  ────────────────────────────────────────────────────────────────────────────────
   Small (< 32²)             184    0.0155     0.0002     0.0000           0.0040
   Medium (32² to 96²)     33579    0.6654     0.5811     0.0007           0.4595
   Large (>= 96²)          35919    0.7520     0.6959     0.0034           0.5600
  COCO sizes by box area in network input pixels, the objects of the other sizes and the detections finding them being
  ignored. — : no object of the size.

  Mean Average Recall
   Objects Of            Objects   mAR@0.5   mAR@0.75   mAR@0.95   mAR@[0.5:0.95]
  ────────────────────────────────────────────────────────────────────────────────
   All                     69682    0.9801     0.8893     0.0333           0.7381
   Small (< 32²)             184    0.0860     0.0053     0.0000           0.0232
   Medium (32² to 96²)     33579    0.9733     0.8613     0.0179           0.7073
   Large (>= 96²)          35919    0.9897     0.9192     0.0434           0.7664
  Objects found by any detection kept for the evaluation, averaged over the classes. COCO sizes by box area in network
  input pixels. — : no object of the size.

  Proposal Recall
                                                           Proposals
   Objects               Count   Recall@0.5   Recall@0.7   Per Image
  ───────────────────────────────────────────────────────────────────
   All Objects           69682       0.9967       0.9751    249.9534
   Small (< 32²)           184       0.3967       0.0870           —
   Medium (32² to 96²)   33579       0.9973       0.9680           —
   Large (>= 96²)        35919       0.9992       0.9864           —
   Block 5 Proposals     69682       0.8093       0.5095    166.8608
   Block 6 Proposals     69682       0.7734       0.2626     49.9440
   Block 7 Proposals     69682       0.6736       0.4271     33.1486
  Share of the test objects overlapped by a proposal of the region proposal network by at least the IoU, sizes by box
  area in network input pixels. A block row keeps the proposals of that block only. — : no object of the size.

  Per Class Average Precision (Area)
                   Ground                                                Threshold    Recall
   Class           Truths   AP@0.5   AP@0.75   AP@0.95   AP@[0.5:0.95]     @ P0.99   @ P0.99
  ───────────────────────────────────────────────────────────────────────────────────────────
   trapezoid         3546   0.3184    0.2870    0.0002          0.2225      0.4119    0.0003
   kite              3471   0.3400    0.2884    0.0000          0.2202           —         —
   rhombus           3436   0.3626    0.3397    0.0000          0.2501           —         —
   capsule           3344   0.4964    0.4386    0.0005          0.3422           —         —
   octagon           3454   0.5144    0.4896    0.0041          0.3956      0.5593    0.0003
   ...
   plus_sign         3442   0.9719    0.9157    0.0020          0.7280      0.6335    0.9198
   star              3549   0.9749    0.8745    0.0010          0.6899      0.7606    0.8901
   annulus           3461   0.9804    0.9293    0.0039          0.7436      0.6806    0.9070
  Worst AP@0.5 first, ties in class order. — : no ground truth box of the class, or the precision 0.99 is never reached.
  - Results : Results.txt
  - Class Metrics : class_metrics.csv
  - Proposal Recall : proposal_recall.csv
  - Class Thresholds : class_thresholds.json
  - Confusion Matrix Table : confusion_matrix.csv
  - Confusion Matrix Image : confusion_matrix.png
  - Sample Images : Samples
```

**Example output: the start of the block of the averaged weights, printed next**
```text
Iteration 5000 · 20000 Test Images · EMA Weights · 2026-10-24 16:15 ────────────────────────────────────────────────────
  Confusion Matrix : score >= 0.50, IoU >= 0.5 (boxes), 29700 right, 1401 wrong class, 1405 false positives, 38581
  missed
  Sample Images : 8 fixed test images

  Mean Average Precision
   Method      mAP@0.5   mAP@0.75   mAP@0.95   mAP@[0.5:0.95]
  ────────────────────────────────────────────────────────────
   Area         0.7208     0.6812     0.0155           0.5673
   11 Point     0.6977     0.6741     0.0320           0.5610
   101 Point    0.7186     0.6793     0.0170           0.5665
  ...
```
The averaged weights reach a higher mAP than the trained ones at iteration 5000, 0.7208 against 0.7151 at IoU 0.5 and 0.5673 against 0.5160 over 0.5:0.95, the gap growing with the IoU (0.0155 against 0.0021 at 0.95). At the deployment threshold they return fewer detections (32,506 against 40,479), with fewer confusions and false positives (1,401 and 1,405 against 3,228 and 3,038) but more missed objects (38,581 against 32,241): fewer of their detections score 0.5 or more. Over the run, the averaged weights lead at iterations 2500, 4500 and 5000 and the trained weights at 3000 to 4000 (the TensorBoard values below), and both curves are still rising at 5000.

**Example: TensorBoard and MLflow after the 20 evaluations**
```text
Tensorboard/Test: 72 scalars (12 mAP, 6 by size, 16 mAR, 18 proposal recall, 20 AP@0.5 per class), 9 images (the 8 samples and
                  the confusion matrix); Tensorboard/Test_EMA: the same tags for the averaged weights
    scalar mAP@0.5/area             500: 0.0496, 1000: 0.1347, 1500: 0.2137, 2000: 0.3241, 2500: 0.3890, 3000: 0.5031,
                                    3500: 0.6003, 4000: 0.6440, 4500: 0.6460, 5000: 0.7151
    scalar mAP@0.5/area (Test_EMA)  500: 0.0427, 1000: 0.1162, 1500: 0.1911, 2000: 0.3034, 2500: 0.4108, 3000: 0.4981,
                                    3500: 0.5710, 4000: 0.6287, 4500: 0.6770, 5000: 0.7208
    scalar AP@0.5/octagon           500: 0.0923, 1000: 0.1269, ..., 4000: 0.3732, 4500: 0.3038, 5000: 0.5144
MLflow: 154 metrics: test/mAP_at_0.5/area, test/mAP_at_0.75/area, test/mAP_at_0.95/area, test/mAP_at_0.5-0.95/area, ... (3 methods),
        test/best_mAP_at_0.5/area (0.7151 at step 5000), 20 test_class/AP_at_0.5/<class>, the same under test_ema/ and
        test_ema_class/ (test_ema/best_mAP_at_0.5/area 0.7208), tags best_iteration 5000 and best_iteration_test_ema 5000
artifacts: Results.txt, configuration/training_configuration.yaml, metric_names.json, analysis/anchor_recall.csv, data/data.json,
           evaluation/iteration_<500|1000|...|5000>/confusion_matrix.png, class_metrics.csv, class_thresholds.json,
           proposal_recall.csv and the same files in ema/
```
`sanitise_metric_name()` writes `@` as `_at_`, drops the brackets and writes `:` as `-`, so `mAP@[0.5:0.95]/area` becomes `mAP_at_0.5-0.95/area` in MLflow (the original names are kept in `metric_names.json`). `best_` follows the headline metric only. The `Tensorboard/Test` writer holds 72 scalars per evaluation for 20 classes: 12 mAP (4 thresholds x 3 methods), 6 by size, 16 mAR, 18 proposal recall values and 20 `AP@0.5/<class>`, with the 8 sample images and the confusion matrix, and `Tensorboard/Test_EMA` the same tags for the moving average of the weights. The proposal recall file is sent with the other small evaluation files (`TRACKED_EVALUATION_FILES`).

### Step 4.14 — `select_object_size_detections()` + `build_object_size_table()`
`evaluate_detections()` computes the same mean average precisions again for the small, medium and large objects of COCO, with every method and threshold, reusing the gathered detections of Step 4.8 and the matching of Step 4.9. The size of an object is the area of its box in network input pixels (`OBJECT_SIZE_RANGES` in `object_sizes.py`): small below 32², medium from 32² to 96², large from 96². `select_object_size_detections()` keeps the detections of a class that count for a size with the COCO rule: the ground truths of the other sizes are ignored, and so are the detections matched to them (their best ground truth reaching the threshold) and the unmatched detections whose own box is of another size. A detection whose best ground truth of the size was already taken by a more confident detection stays a false positive, as without the sizes. A class without any object of the size is left out of the mean of that size, and a size without any object has no mAP: `—` in the table and no scalar. `build_object_size_table()` shows the mAPs of the area method of every size with its number of objects, after the table of the methods. The scalars `mAP@0.5 By Size/<size>` and `mAP@[0.5:0.95] By Size/<size>` go to TensorBoard and MLflow, and the values of every class to `class_metrics.csv` (Step 4.11). The instance segmentation has two of these tables, by mask area for the mask AP and by box area for the box AP.

`build_average_recall_table()` then gives the `Mean Average Recall` table, with the same rows (all the objects, then every size) and the objects of every row: the share of the objects found by any detection kept for the evaluation, at IoU 0.5, 0.75, 0.95 and averaged over 0.5:0.95, averaged over the classes, the same for every method. A low mAP with a high recall tells that the objects are found but scored below false positives; a low recall that they are not found at all. Its scalars are `mAR@0.5/<group>`, `mAR@0.75/<group>`, `mAR@0.95/<group>` and `mAR@[0.5:0.95]/<group>`, the group being `all` or the size, a size without object having none.

**Example output: the size and recall tables of iteration 5000** (the generator draws no shape under 64 pixels, `MINIMUM_SHAPE_SIZE`, so the 184 small test objects are shapes mostly hidden by others, their box being that of their visible part)
```text
  Mean Average Precision By Object Size (Area)
   Object Size           Objects   mAP@0.5   mAP@0.75   mAP@0.95   mAP@[0.5:0.95]
  ────────────────────────────────────────────────────────────────────────────────
   Small (< 32²)             184    0.0155     0.0002     0.0000           0.0040
   Medium (32² to 96²)     33579    0.6654     0.5811     0.0007           0.4595
   Large (>= 96²)          35919    0.7520     0.6959     0.0034           0.5600
  COCO sizes by box area in network input pixels, the objects of the other sizes and the detections finding them being
  ignored. — : no object of the size.

  Mean Average Recall
   Objects Of            Objects   mAR@0.5   mAR@0.75   mAR@0.95   mAR@[0.5:0.95]
  ────────────────────────────────────────────────────────────────────────────────
   All                     69682    0.9801     0.8893     0.0333           0.7381
   Small (< 32²)             184    0.0860     0.0053     0.0000           0.0232
   Medium (32² to 96²)     33579    0.9733     0.8613     0.0179           0.7073
   Large (>= 96²)          35919    0.9897     0.9192     0.0434           0.7664
  Objects found by any detection kept for the evaluation, averaged over the classes. COCO sizes by box area in network
  input pixels. — : no object of the size.
```
```text
Tensorboard/Test: mAP@0.5 By Size/small 0.0155, mAP@[0.5:0.95] By Size/small 0.0040, ..., mAP@[0.5:0.95] By Size/large 0.5600,
                  mAR@0.5/all 0.9801, mAR@0.75/all 0.8893, ..., mAR@[0.5:0.95]/large 0.7664              (6 + 16 scalars)
MLflow: test/mAP_at_0.5 By Size/small, test/mAP_at_0.5-0.95 By Size/small, ..., test/mAR_at_0.5/all, ..., test/mAR_at_0.5-0.95/large
```
The medium and large objects reach a `mAP@0.5` of 0.6654 and 0.7520, the small ones 0.0155: the anchors of 32 and 64 pixels on a grid of stride 32 give a foreground anchor to 10.3 % of the 1,254 small training objects (`anchor_recall.csv` of the run, Step 2.21). The `mAR@0.5` of all the objects is 0.9801 against a `mAP@0.5` of 0.7151: almost every object is found by a detection above 0.05, but for the weak classes these detections are ranked below false positives or detections of other classes, which lowers the precision. A size without objects would show `—` and give no scalar.

### Step 4.15 — `accumulate_proposal_recall()` + `save_proposal_recall()`
The anchor recall of Step 2.21 measures what the anchors allow; the proposal recall measures what the trained region proposal network gives the detection heads. `create_evaluation_accumulator()` creates its accumulator for the detection levels (`create_proposal_recall_accumulator()`), and `accumulate_evaluation_batch()` calls `accumulate_proposal_recall()` for every test image with the filtered proposals of Step 4.3: the padding copies are left out with `valid_proposal_mask`, the best IoU of every object is kept with all the proposals and with the proposals of every level alone, computed on the device of the model, and the valid proposals of every level are counted. In `compute_evaluation_results()`, `save_proposal_recall()` calls `summarise_proposal_recall()`, which gives the share of the objects whose best IoU reaches 0.5 and 0.7 (`PROPOSAL_RECALL_THRESHOLDS`) for all the objects, every size (by box area, Step 4.14) and the proposals of every block alone, with the mean number of valid proposals per image of all the blocks and of every block. A block row counts all the objects, so its recall shows how many objects that level brings to the detection heads. The table is always shown after the size table; the rows are written to `proposal_recall.csv` in the evaluation folder of the checkpoint, which is sent to MLflow, and the scalars `Proposal Recall@0.5/<group>`, `Proposal Recall@0.7/<group>` and `Proposals Per Image/<group>` go to TensorBoard and MLflow, a size without object having none. The instance segmentation shows the same table once, after its box size table.

**Example output: the table of iteration 5000**
```text
  Proposal Recall
                                                           Proposals
   Objects               Count   Recall@0.5   Recall@0.7   Per Image
  ───────────────────────────────────────────────────────────────────
   All Objects           69682       0.9967       0.9751    249.9534
   Small (< 32²)           184       0.3967       0.0870           —
   Medium (32² to 96²)   33579       0.9973       0.9680           —
   Large (>= 96²)        35919       0.9992       0.9864           —
   Block 5 Proposals     69682       0.8093       0.5095    166.8608
   Block 6 Proposals     69682       0.7734       0.2626     49.9440
   Block 7 Proposals     69682       0.6736       0.4271     33.1486
  Share of the test objects overlapped by a proposal of the region proposal network by at least the IoU, sizes by box
  area in network input pixels. A block row keeps the proposals of that block only. — : no object of the size.
```
```text
Evaluation/Iteration_5000/proposal_recall.csv
group,objects,recall_50,recall_70,proposals_per_image
all,69682,0.996714,0.975116,249.953350
small,184,0.396739,0.086957,
medium,33579,0.997350,0.967956,
large,35919,0.999193,0.986358,
block_5,69682,0.809262,0.509472,166.860800
block_6,69682,0.773413,0.262564,49.944000
block_7,69682,0.673574,0.427054,33.148550

MLflow: test/Proposal Recall_at_0.5/<group>, test/Proposal Recall_at_0.7/<group>, test/Proposals Per Image/<group>,
        evaluation/iteration_5000/proposal_recall.csv
```
At iteration 5000, the proposals overlap 99.67 % of the test objects by 0.5 and 97.51 % by 0.7, and the non-maximum suppression leaves 249.95 valid proposals of the 250 slots per image, almost no padding copies. The region proposal network is no longer the limit: at iteration 500 its recall at 0.5 was already 0.9880, when the `mAP@0.5` was 0.0496, so the errors of Step 4.12 come from the classification and the scores of the detection heads. The small objects stay the exception, 73 of the 184 reached at 0.5. Block 5 gives 166.86 of the 250 proposals per image, blocks 6 and 7 49.94 and 33.15, and the proposals of every block alone already reach 67 to 81 % of the objects at 0.5, the levels overlapping in the sizes they cover.


---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of the logic of `scripts/evaluation/evaluate_checkpoint.py` run on the 5000-iteration run of Stage 3 (its 20 checkpoints, one every 250 iterations) with `iteration = "all"`, `batch_size = 64`, `track_evaluation = False`, a temporary output folder and, to keep it short, a test folder of the first 200 images of the test split (its `labels.json` reduced to these images and their 709 annotations, its `data/` folder linked). It ran on the GPU, shared with other trainings, in 59 minutes, about 3 minutes per checkpoint for the trained and the averaged weights. Step 5.4 comes from an earlier, tracked evaluation of the four checkpoints of the CPU run, the tracking being off here. They were captured before the per-class recall intervals were added (2026-11-01), whose `Per Class Recall` table now follows the per-class table of Step 5.6.

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The same script for every task (see `classification_workflow.md`, Stage 5), configured by the variables of its main block (`run_name`, `experiment_folder`, `iteration`, `test_data_folder`, `batch_size`, `track_evaluation`, `maximum_error_images_per_class_pair`, `output_folder`). Its default `maximum_error_images_per_class_pair = 100000` turns the error images of Step 4.5 on whatever the configuration, every error image of the test split being saved, and the `pretrained` settings are turned off (`disable_pretrained_settings()`), every weight coming from the evaluated checkpoint.

### Step 5.2 — trainer construction
The trainer is built from the `training_configuration.yaml` of the run, with the classes of the `label_dictionary.json` saved next to it rather than those of the mapping file (`classification_workflow.md`, Step 5.2).

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
With the tracking on, the evaluations go to an MLflow run of their own, `<run name>-evaluation`, in the backend of the evaluated run, tagged with the MLflow run of the training read from its `mlflow_run.json` (`evaluation_of`); the run of the training is only read.

**Example output** (the tracked evaluation of the CPU run)
```text
MLflow Run Created In The Experiment 'object_detection' : fcc494bfb83d4bc78638fa8ef3947ea6 (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```

### Step 5.5 — `summarise_checkpoint_evaluations()`
**Example output**
```text
--------------------------------------------------------------
Evaluation Summary Of 20 Checkpoints : evaluation_summary.csv
--------------------------------------------------------------
  - Best Checkpoint : iteration 5000 (mAP@0.5/area 0.7210)
  - Best Checkpoint (EMA Weights) : iteration 5000 (mAP@0.5/area 0.7287)
```

**Example: `<output folder>/<run>/Evaluation/evaluation_summary.csv`** (145 columns: the iteration, the 72 scalars of the trained weights, that is the 52 of Step 4.7 and the `AP@0.5` of the 20 classes, then the same 72 prefixed with `ema_`; a few columns, values rounded to 4 decimals)
```text
iteration   mAP@0.5/area   mAP@[0.5:0.95]/area   AP@0.5/kite   AP@0.5/plus_sign   ema_mAP@0.5/area   ema_mAP@[0.5:0.95]/area
   5000       0.7210         0.5257                0.2906        0.9992             0.7287             0.5794
   4750       0.7017         0.5050                0.3043        0.9992             0.7072             0.5586
   4500       0.6595         0.4805                0.3377        0.9985             0.6843             0.5403
   4250       0.6871         0.5000                0.3471        0.9972             0.6621             0.5184
   ...
    500       0.0658         0.0321                0.0383        0.0012             0.0560             0.0275
    250       0.0242         0.0084                0.0037        0.0000             0.0192             0.0061
```
On the 200 images, the last checkpoint is the best for both kinds of weights. The `mAP@0.5` of the trained weights dips at iterations 3250 and 4500 (0.6595 after 0.6871 at 4250), that of the averaged weights rises at every checkpoint. From iteration 3000 the averaged weights have the higher `mAP@[0.5:0.95]` at every checkpoint, while the trained weights keep the higher `mAP@0.5` at most checkpoints up to 4250 and the averaged weights lead from 4500. The values at iteration 5000 are close to those of the whole test split (0.7151 and 0.7208, Step 4.13), but the 200 images hold only 23 to 45 objects per class, so the class columns move a lot from one checkpoint to the next: kite is best at iteration 4250 (0.3471) and falls to 0.2906 at 5000. A checkpoint chosen on a reduced split should be checked on the whole split before it is deployed.

### Step 5.6 — `run_test_evaluation()` for each checkpoint
**Example output: the per-class table of the checkpoint of iteration 5000 on the 200 images** (shortened)
```text
-----------------------------
Evaluating : model_005000.pt
-----------------------------

Iteration 5000 · 200 Test Images · 2026-10-25 06:18 ────────────────────────────────────────────────────────────────
  Confusion Matrix : score >= 0.50, IoU >= 0.5 (boxes), 335 right, 29 wrong class, 35 false positives, 345 missed
  ...
  Per Class Average Precision (Area)
                   Ground                                                Threshold    Recall
   Class           Truths   AP@0.5   AP@0.75   AP@0.95   AP@[0.5:0.95]     @ P0.99   @ P0.99
  ──────────────────────────────────────────────────────────────────────────────────────────
   kite                33   0.2906    0.2660    0.0000          0.1960      0.3067    0.0303
   trapezoid           44   0.4058    0.3453    0.0003          0.2836      0.3583    0.0455
   rhombus             42   0.4324    0.4063    0.0008          0.3032      0.3709    0.0238
   capsule             27   0.4439    0.3928    0.0015          0.3042           —         —
   ...
   annulus             31   0.9590    0.9213    0.0000          0.7573      0.6417    0.8710
   star                31   0.9611    0.9038    0.0000          0.6685      0.7725    0.8387
   plus_sign           36   0.9992    0.9029    0.0244          0.7671      0.7907    0.9722
  Worst AP@0.5 first, ties in class order. — : no ground truth box of the class, or the precision 0.99 is never reached.
```
The weakest classes are those of the whole test split (Step 4.13): kite, trapezoid and rhombus, then capsule, whose detections never reach the precision 0.99 on these images, hence its `—` threshold and recall. At the score 0.5 of the confusion matrix, 345 of the 709 objects are missed, about the same share as on the whole split. A class without ground truth box would have no recall, so no average precision (`—`), would be left out of the mAP and listed last.
