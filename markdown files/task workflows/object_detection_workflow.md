# Object Detection Workflow

This document follows one object detection run (DET) from `main()` to the evaluation of a checkpoint, to show which function runs when, what it receives, what it returns and who uses the result next. The **Call Tree Overview** gives the whole run on one page. Each stage then has one section, with one step per function: a short explanation followed by a **real** output. The steps shared by every task (configuration loading, checkpoints, training loop, MLflow tracking, standalone evaluation) are explained in detail in `classification_workflow.md`; here they keep a short explanation and the real output of the detection run, and the steps specific to the detection (COCO loaders, anchors, region proposals, target assignment, detection heads, the four losses, post-processing, mean average precision) are explained in full.

The examples come from real runs on CPU, so that the GPU stayed free for another training. The synthetic dataset of the configuration, `synthetic_shapes`, is not on this machine, so a small one was generated in a temporary folder with `create_synthetic_dataset(task="object_detection")`: 100 training, 20 validation and 20 test images of 512 pixels, 1 to 6 shapes per image, 9 shape classes. The data folders and the mapping file of the configuration were redirected to it, and the DALI loader, which decodes the images on the GPU, was replaced by the Native loader (`Data.DALI.activated: False`).
- **Stages 1 and 2**: `configurations/object_detection/configuration_synthetic_shapes.yaml` (1024 pixel images, batch size 40, detection on the blocks 5, 6 and 7 of the backbone), with the run folder redirected to a temporary folder.
- **Stages 3, 4 and 5**: the same configuration, shortened to 512 pixel images (the smallest size that keeps the anchors of the configuration, the largest being 512 pixels), batch size 2, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4, a validation batch every 10, 2 loading workers, with the MLflow tracking turned on. After 12 iterations the detector finds no shape, so the metrics of the examples are 0 on purpose: the point is to see the shapes, the files and the order of the steps.
- **Steps 2.21, 4.14 and 4.15** (the anchor recall, the mAP by object size and the proposal recall): a later run with the settings of Stages 3 to 5, on a dataset of the same sizes made by the current generator, with 20 shape classes; the examples of Steps 4.14 and 4.15 come from a second run on the same dataset with the same settings, trained for 400 iterations at a learning rate of 0.001 so that the region proposal network is partly trained.

## Table of Contents
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
│   Entry point of every task. Point the path to an object_detection configuration to run this workflow.
│
├── load_experiment_configuration(configuration_path)
│   │   Splits the YAML into experiment_configuration ('ExperimentConfiguration') and model_configuration (Optimization,
│   │   Data, Backbone, DetectionLevels, RegionProposal, Detector), converts the types, adds label_mode and label_dictionary.
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
│               ├── no sequence settings
│               ├── check_detection_levels()                     DetectionLevels: quoted backbone blocks in increasing
│               │                                                order, anchor scales within the image size, positive
│               │                                                ratios, region proposal and detection heads
│               ├── check_region_proposal_configuration()        IoU thresholds, proposals before and after NMS
│               └── check_detector_configuration()               maximum detections, training samples
│
├── get_trainer_class(task)  -> DetectionTrainer
│
├── DetectionTrainer(experiment_configuration, model_configuration, configuration_path)  [STAGE 2]
│   │   Reads the score threshold of the evaluation and the one of the sample images.
│   │
│   └── BaseTrainer.__init__()
│       ├── get_device() / get_run_name() -> build_run_name() / DataAugmenter() / setup_training_paths()
│       ├── build_dataloaders()
│       │   │   Reads the test labels.json for class_names, class_colours and the file name of every test image.
│       │   └── get_detection_dataloaders() -> get_detection_dataloader() for every split
│       │       ├── DALI activated: dali_pipeline() + DALIGenericIterator + DALIDataloaderWrapper (GPU decoding)
│       │       └── otherwise:      CocoDetectionDataset() + build_native_loader(collate_function)
│       ├── get_test_sample_classes() + choose_sample_identifiers()      empty class lists: evenly spread images
│       ├── build_model()
│       │   └── TwoStageDetector.__init__()
│       │       ├── build_backbone() + compute_detection_input_information()   feature maps of the detection blocks
│       │       ├── RegionProposal() for every level -> Anchors.generate_anchors()
│       │       ├── DetectionHead() for every level
│       │       ├── _aggregate_all_anchors()                                     all the anchors in one tensor
│       │       └── RegionProposalFilter()
│       ├── initialise_pretrained_components() / build_optimizer() / get_tracked_metrics() + MetricTracker()
│       ├── print_experiment_information() -> describe_split() -> count_split_images_and_annotations()
│       ├── print_summary() / print_receptive_field_summary() / print_model_size_summary()
│       ├── view_input_batches() -> show_input_samples()   (only with view_input_images)
│       ├── analyse_training_data()   anchor recall of the training objects, anchor_recall.csv
│       │   └── gather_training_object_boxes() + get_level_anchors() + compute_best_anchor_overlaps() +
│       │       summarise_anchor_recall()
│       └── restore_last_model()
│
└── trainer.run_training_loop()  [STAGE 3]
    └── run_training_iterations()
        ├── for each training_iteration:
        │   ├── save_model() + run_test_evaluation() when due                    -> [STAGE 4]
        │   ├── get_next_batch() -> CocoDetectionDataset.__getitem__() + collate_function()
        │   │       images [B, 3, H, W] uint8, bounding_boxes and labels as lists (a different number per image)
        │   ├── run_model_iteration()
        │   │   ├── prepare_batch() / augment_batch()   the geometric augmentations move the boxes
        │   │   ├── forward_pass() -> TwoStageDetector.forward(ground_truth_bounding_boxes=..., ground_truth_labels=...)
        │   │   │   ├── Backbone.forward()                                   feature maps of the detection blocks
        │   │   │   ├── _forward_region_proposals() -> RegionProposal.forward()  objectness and boxes of every anchor
        │   │   │   ├── aggregate_proposals_and_anchors()
        │   │   │   ├── RegionProposalFilter.forward() -> select_best_proposals()   top-k, NMS, padded to a fixed number
        │   │   │   ├── _prepare_detector_inputs()
        │   │   │   │       batch_assign_targets_to_proposals() + turn_boxes_to_transformation_targets() +
        │   │   │   │       sample_positive_and_negative_training_targets()        every image of the batch at once
        │   │   │   ├── _forward_detections() -> DetectionHead.forward()     RoIAlign, class logits, box regressions
        │   │   │   ├── aggregate_detection_outputs() + _rearrange_predictions_by_batch()
        │   │   │   ├── compute_region_proposal_losses()
        │   │   │   │       batch_assign_targets_to_anchors(), sampling, binary cross entropy + smooth L1
        │   │   │   └── compute_detection_losses()                           cross entropy + smooth L1
        │   │   ├── compute_batch_metrics()   none: the four losses are the tracked metrics
        │   │   └── MetricTracker.accumulate() + log_to_tensorboard()
        │   ├── training_loss.backward() / step_optimizer() / validation pass
        │   └── average_trackers() + log_interval_metrics() + print_interval_summary()
        └── final save_model() + run_test_evaluation(), finish_experiment_tracking()

run_test_evaluation(iteration)  [STAGE 4]
├── evaluation_context()                    the score threshold of the model lowered to evaluation.score_threshold
├── create_evaluation_accumulator()         the detections and ground truths of every test image, the confusion matrix
├── for each test batch:
│   ├── prepare_batch() / predict_batch()   TwoStageDetector.forward() without ground truth -> postprocess_detections()
│   ├── accumulate_evaluation_batch()       for every image, the detections reaching the deployment threshold:
│   │   ├── compute_overlap_matrix() + match_image_detections() + add_to_confusion_matrix()
│   │   ├── dump_errors() -> group_image_errors() + draw_test_image()          Errors/ (error_dumps on)
│   │   └── accumulate_proposal_recall()     best overlap of every object with the valid proposals
│   └── save_sample_images() -> build_sample_image() -> build_detection_sample_grid()
├── compute_evaluation_results()
│   ├── evaluate_detections()
│   │   ├── gather_class_detections() -> find_best_ground_truth()             for every class, once
│   │   ├── compute_class_true_positive_flags()                                for every IoU threshold, once
│   │   ├── for every method (area, 11 point, 101 point) and IoU threshold (0.50 to 0.95):
│   │   │   compute_mean_average_precision() -> match_detections_to_ground_truth() + compute_precision_recall() +
│   │   │   compute_average_precision()
│   │   ├── for every object size (small, medium, large), the same with select_object_size_detections()
│   │   └── find_class_score_thresholds() -> find_score_threshold()           at IoU 0.5, for every class
│   ├── build_class_average_precision_rows() + write_class_metrics_file()      class_metrics.csv
│   ├── build_mean_average_precision_table() + build_object_size_table() + build_class_average_precision_table()
│   ├── save_proposal_recall() -> summarise_proposal_recall() + write_class_metrics_file()   proposal_recall.csv
│   └── save_error_analysis()      class_thresholds.json, confusion_matrix.csv and .png, the notes of the report
└── report_evaluation() -> report_evaluation_results() + track_evaluation()

scripts/evaluation/evaluate_checkpoint.py  [STAGE 5]
└── same as every task
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below are real outputs from running the loader on `configurations/object_detection/configuration_synthetic_shapes.yaml`, its data folders and mapping file redirected to the generated dataset and the DALI loader turned off.

### Step 1.1 — `main()`
The entry point is the same for every task (see `classification_workflow.md`, Step 1.1).

### Step 1.2 — `load_experiment_configuration()`
The model sections of the detection are `Backbone`, `DetectionLevels` (keyed by the backbone blocks whose feature maps feed the detection, every level holding the scales and aspect ratios of its anchors, its region proposal head and its detection head), `RegionProposal` (the settings shared by the region proposal heads: the IoU thresholds labelling the anchors, the proposals kept before and after the non-maximum suppression) and `Detector` (the settings shared by the heads classifying and refining the proposals, the post-processing of the inference). The `evaluation` section has its own `score_threshold`, the lowest score of the detections kept for the mean average precision, the `target_precision` of the score threshold of every class (Step 4.12) and the `error_dumps` of the error images (Step 4.5).

**Example: the sections and the detection settings**
```text
model_configuration keys: Optimization, Data, Backbone, DetectionLevels, RegionProposal, Detector
evaluation:               {'per_class_report': True, 'number_sample_images': 8, 'score_threshold': 0.05, 'target_precision': 0.99,
                           'error_dumps': {'activated': False, 'maximum_per_class_pair': 20}}
Data keys after loading:  loading, image_settings, label_mapping, data_augmentation, label_mode, label_dictionary
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
The common checks run first, then `check_detection_configuration()`. Two of its checks keep the target assignment consistent: an anchor cannot be both background and foreground, and the detector samples its training targets among the proposals kept by the filter.

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
self.batch_size = 40   self.mixed_precision = False   self.start_iteration = 1
```

### Step 2.3 — `get_device()`
**Example output** (forced on the CPU; a normal run on this machine prints `cuda`)
```text
For Training We Will Be Using device: cpu
```

### Step 2.4 — `get_run_name()` → `build_run_name()`
The input and backbone tokens are those of every image task. `build_detection_sections()` adds `L567` (the detection blocks), `A9.32-512` (9 anchors per position, scales from 32 to 512), `RBN-R2x64` (the region proposal heads: batch normalization, 2 layers of 64 filters), `DSE-DRES-DBN-D7.2x64.256.128` (the detection heads: squeeze-and-excitation, residual, batch normalization, RoIAlign of 7, 2 layers of 64 filters, fully connected 256 and 128) and `MC9` (9 foreground classes, `BIN` in binary mode). A setting that differs between the levels is written for every level, joined with `+`.

**Example output**
```text
DET_IN-KAR-1024_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128_MC9
```

### Step 2.5 — `DataAugmenter()`
The same augmenter as the classification; the geometric augmentations move the boxes and drop those leaving the image (Step 3.8). This configuration turns every type off.

**Example: the settings of the configuration**
```text
activated True   probability 0.4
0 of 16 types activated: []
```

### Step 2.6 — `setup_training_paths()`
**Example: the run folder after this step**
```text
DET_IN-KAR-1024_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128_MC9/
├── Tensorboard/
├── Weights/
└── training_configuration.yaml    5,613 bytes
```

### Step 2.7 — `build_dataloaders()`
The hook reads the `labels.json` of the test split for three things: `class_names` (the categories named after the label dictionary, the background keeping its name), `class_colours` (one BGR colour per class, for the boxes of the sample images), and the file name of every test image, which names the sample images whatever the loader (the DALI loader has no dataset to ask). Then it builds the loaders (Step 2.8).

**Example output**
```text
class_names = {0: 'background', 1: 'square', 2: 'circle', 3: 'triangle', 4: 'rectangle', 5: 'pentagon', 6: 'hexagon',
               7: 'plus_sign', 8: 'star', 9: 'octagon'}
test_image_file_names: 20 entries, e.g. {1: 'test_000000.png', 2: 'test_000001.png', 3: 'test_000002.png'}
```

### Step 2.8 — `get_detection_dataloaders()` → `get_detection_dataloader()`
For every split, `<folder>/data/` holds the images and `<folder>/labels.json` the COCO manifest. With `Data.DALI.activated`, `dali_pipeline()` decodes and resizes the images on the GPU and `DALIDataloaderWrapper` turns its padded boxes and labels back into lists, so that its batches have the same format as the Native ones. Otherwise a `CocoDetectionDataset` (Step 2.9) is wrapped in the Native loader with `collate_function()`, which stacks the images and keeps the boxes and labels as lists, every image holding its own number of boxes.

**Example: size of the three splits** (Native loader)
```text
train       100 images -> 3 batches of 40 (one epoch = 3 iterations)
validation   20 images -> 1 batch of 40
test         20 images -> 1 batch of 40
collate      <function collate_function>
```

### Step 2.9 — `CocoDetectionDataset.__init__()` (and `__getitem__()`)
The dataset reads the manifest, shuffles its images once and groups the annotations by image. `__getitem__()` reads the image, keeps the boxes whose category has a class (an unmapped category gives the background and is dropped), turns the COCO `[x, y, width, height]` into `[x_min, y_min, x_max, y_max]`, and resizes the image and its boxes together (`preprocess_image_and_boxes()`, with a letterbox when `keep_ratio` and the image is not square).

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

**Example output**
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
Every detection level gets its anchors: for every position of its feature map, one box per scale and aspect ratio, the ratio being the height over the width (0.5 a wide box, 2.0 a tall box) for the same area. The anchors of a position are centred on its corner in the image, `(column x stride, row x stride)`, the stride being the image size over the feature map size. The region proposal head is a block of 3x3 convolutions followed by two 1x1 convolutions: one objectness logit per anchor, and 4 box offsets per anchor. `_aggregate_all_anchors()` then concatenates the anchors of every level into `model.anchors`.

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
Every level gets a detection head: `roi_align()` cuts a 7x7 feature map out of the level for every proposal (`spatial_scale` = feature map size / image size), a block of 3x3 convolutions with the modules of `Detector.modules` refines it, fully connected layers flatten it, and two linear layers give the logit of every class and 4 box offsets per class.

**Example: the head of level 5**
```text
DetectionHead(
  (convolutional_block): ConvolutionBlock(   2 x [Conv2d(64, 64, 3x3), BatchNorm2d(64), ReLU], 2 x SqueezeAndExcitation(64 -> 32 -> 64) )
  (fully_connected_block): Sequential(Linear(3136, 256), ReLU, Linear(256, 128), ReLU)        <- 3136 = 7 x 7 x 64
  (classifier): Linear(in_features=128, out_features=10, bias=True)
  (bounding_box_regressor): Linear(in_features=128, out_features=40, bias=True)               <- 4 offsets for each of the 10 classes
)
```
The heads of levels 6 and 7 read 128 channels instead of 64, hence their 36,864 more parameters (the first 3x3 convolution).

### Step 2.16 — `RegionProposalFilter.__init__()`
The filter keeps the proposals the detection heads classify: the `pre_nms_proposals` best scored, then those surviving the non-maximum suppression at `nms_iou_threshold`, at most `post_nms_proposals`, with different numbers in training and in inference.

**Example output**
```text
training 1000 -> 500   inference 500 -> 250   nms_iou_threshold 0.5
```

### Step 2.17 — `initialise_pretrained_components()` + `build_optimizer()` + `get_tracked_metrics()`
The backbone, the region proposal heads and the detection heads can come from other runs. The detection heads may have more classes than their source run: the source classes must be the first ones of this run (`class_names` of both `data.json` files), and `extend_class_layers()` copies the classification and box regression rows of the source classes into the first rows of the detection heads of this run, the new classes keeping the random rows these heads were built with. The mask heads of an instance segmentation run extend their mask prediction layer the same way. The four losses of the detector are tracked with their sum.

**Example output**
```text
pretrained_checkpoint_paths = {'Backbone': None}
Adam, 1 parameter group, base_learning_rates [0.0002], weight decay 0.0
{'total_loss': 'Total Loss', 'region_proposal_classification_loss': 'RPN Classification Loss',
 'region_proposal_localisation_loss': 'RPN Regression Loss', 'detection_classification_loss': 'Detection Classification Loss',
 'detection_regression_loss': 'Detection Regression Loss'}
```

### Step 2.18 — `print_experiment_information()` (→ `describe_split()` → `count_split_images_and_annotations()`)
**Example output**
```text
- Run Name                : DET_IN-KAR-1024_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128_MC9
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
The summary of the detector adds a region proposal panel (the anchors of every level) and a detection panel (the heads of every level); the receptive field panel marks the anchor scales of every detection block, to compare them with what a position of its feature map sees.

**Example output** (shortened)
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
│ Detection Block 5                                                            │
│   - Modules            : Residual Shortcuts, Squeeze-and-Excitation          │
│   - Convolution Layers : 2                                                   │
│   - Fully Conn. Layers : [256, 128]                                          │
│   - Feature Map Size   : [32, 32]                                            │
│   - RoI Align Pool     : [7, 7]                                              │
│   - Spatial Scale      : 0.0312                                              │
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
With `view_input_images`, `show_input_samples()` shows an augmented training image next to a validation image, every box captioned with its class name, each image followed by an information panel counting its objects per class (`Objects : 2 (cat 1, dog 1)`) and, for the training image, giving the augmentations drawn with their values and the objects they removed (`Removed objects : 1 (dog)`). A new run starts from scratch.

**Example output: a new run**
```text
No Initiation Model Weights Will Be Used...
We Generate A New Model That Will Be Trained From Scratch.            -> start_iteration = 1
```

### Step 2.21 — `analyse_training_data()` → anchor recall
Called by `BaseTrainer.__init__()` between `view_input_batches()` and `restore_last_model()` (Step 2.20), at every launch, resumes included, and never in the input viewer mode. The hook of the general trainer writes nothing; the detection one measures how many training objects the anchors can learn. `gather_training_object_boxes()` reads the boxes of the training split from its annotations (`select_image_objects()`, the crowd and the background left out, as in `__getitem__()`), placed in network input pixels by `preprocess_boxes()` without loading any image and without the augmentations. `compute_best_anchor_overlaps()` gives the overlap of the best anchor of every object in every level (`TwoStageDetector.get_level_anchors()`), 2048 objects at a time on the device of the model. `summarise_anchor_recall()` counts the objects whose best anchor reaches `RegionProposal.foreground_iou_threshold.min`, in every level alone and in all of them, since the region proposal network assigns the objects to the anchors of all the levels at once. An object under the threshold everywhere only gets its single best anchor as foreground (the fallback assignment). The table is printed and `anchor_recall.csv` is written in the run folder, then sent to the `analysis` folder of the MLflow run by `start_experiment_tracking()` (Step 3.3), as `anchor_recall_resume_<n>.csv` for a resume.

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
sizes by box area in network input pixels. The other objects only get their single best anchor as foreground. — : no
object of the size.
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

> The examples below are real outputs from a run on CPU of the redirected `configuration_synthetic_shapes.yaml` shortened to 512 pixel images, batch size 2, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4 iterations, a validation batch every 10 iterations, with the MLflow tracking on. Its run name is `DET_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128_MC9`; with batches of 2 images, an epoch is 50 iterations. At 512 pixels the feature maps of the detection blocks are 16x16, 8x8 and 8x8, so the detector has 2304 + 576 + 576 = 3456 anchors.

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
The loop of every task (see `classification_workflow.md`, Step 3.1).

**Example: what happens at each iteration of the run**

| Iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | after the loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Checkpoint saved | | | ✔ | | | ✔ | | | ✔ | | | | ✔ (iteration 12) |
| Test evaluation | | | | | | ✔ | | | | | | | ✔ (iteration 12) |
| Validation batch | | | | | | | | | | ✔ | | | |
| Console summary | | | | ✔ | | | | ✔ | | | | ✔ | |

### Step 3.2 — `setup_tensorboard_writers()` / `initialize_training_loop_state()`
**Example: the curves after the run**
```text
Tensorboard/Train/        Total Loss, RPN Classification Loss, RPN Regression Loss, Detection Classification Loss,
                          Detection Regression Loss, Learning Rate                                   (iterations 1 to 12)
Tensorboard/Validation/   Total Loss, RPN Classification Loss, RPN Regression Loss, Detection Classification Loss,
                          Detection Regression Loss                                                  (iteration 10)
```

### Step 3.3 — `start_experiment_tracking()`
**Example output**
```text
MLflow Run Created In The Experiment 'object_detection' : d34603eb46cb459982f88c208dd1a275 (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```
The MLflow run is created by a background thread. In this run it existed before the first checkpoint, so the four checkpoints hold its identifier; a checkpoint saved while the run is still being created holds `mlflow_run_id: None`. The configuration, `anchor_recall.csv` of Step 2.21 (folder `analysis`) and `data.json` are sent as soon as the run is open.

### Step 3.4 — `save_model()` → `save_checkpoint()`
The backbone is the only transferable component of the detector, so the checkpoint has only `pretrained_backbone`.

**Example: the checkpoints and the keys of the last one**
```text
Weights/Iteration_3/model_000003.pt   67,443,616 bytes      Weights/Iteration_6/model_000006.pt   67,443,616 bytes
Weights/Iteration_9/model_000009.pt   67,443,616 bytes      Weights/Iteration_12/model_000012.pt  67,443,616 bytes
['iteration', 'model_state', 'optimizer_state', 'format_version', 'task', 'run_name', 'mlflow_run_id', 'pretrained_backbone']
model_state: 363 tensors, the first 'backbone.convolution_blocks.0.squeeze_and_excitation_layers.0. ...'
mlflow_run_id: 'd34603eb46cb459982f88c208dd1a275' in the four checkpoints
```

### Step 3.5 — `run_test_evaluation()`
At iterations 6 and 12 (the final checkpoint), see Stage 4.

### Step 3.6 — `get_next_batch()` → `CocoDetectionDataset.__getitem__()` + `collate_function()`
The images are stacked; the boxes and labels stay lists, one tensor per image.

**Example: the first batch of the run**
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
The geometric augmentations move the boxes with the image, and `clean_bounding_boxes()` clips them to the image and drops with their labels the boxes left narrower or shorter than 1 pixel or with less than 25 % of their moved area inside the image (`MINIMUM_VISIBLE_AREA_FRACTION`), the fragments a crop or a rotation cuts at the border. This configuration turns every type off, so nothing changes.

**Example output**
```text
images: changed per sample [False, False]
```

### Step 3.9 — `forward_pass()` → `TwoStageDetector.forward(ground_truth_bounding_boxes=..., ground_truth_labels=...)`
With the ground truth, the detector samples its training targets and returns its four losses (Steps 3.10 to 3.18). The total loss is their sum.

**Example: the dictionary returned at iteration 1**
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

> The examples of Steps 3.10 to 3.18 run the first batch of the run step by step through the model at the end of the run (iteration 12), in training mode, so their values differ from those of iteration 1.

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
For the whole batch at once, with `select_best_proposals()`, shared with the `InferenceDetector` of the deployment: the sigmoid of the objectness, the `pre_nms_proposals` best proposals of every image (1000 in training, all of them when there are fewer anchors), their boxes clamped to the image, the non-maximum suppression at 0.5 of every image, then the `post_nms_proposals` best survivors of every image (500 in training, 250 in inference). The suppression is the only step run image by image (`suppress_overlapping_proposals()`, one torchvision `nms` call per image, its fastest form on the GPU), the exported detector running a single `batched_nms` call over the batch, which keeps the same proposals. An image with fewer survivors is padded with copies of random survivors, so that every image has the same number of proposals; the filter also returns the mask of the slots holding a real survivor, which the proposal recall of the evaluation reads (Step 4.15).

**Example output**
```text
filtered: boxes (2, 500, 4)   scores (2, 500)   origins (2, 500)
image 0: 500 distinct boxes, levels {5: 484, 6: 6, 7: 10}
image 1: 500 distinct boxes, levels {5: 466, 6: 22, 7: 12}
```

### Step 3.14 — `_prepare_detector_inputs()`
In training, `add_ground_truth_proposals()` first writes the ground truth boxes of every image into the last slots of its filtered proposals (the padding copies, then the least confident survivors), each at the level of the anchor overlapping it most, so that every object has a positive proposal with a zero regression target even when the region proposal network misses it. This happens whenever the ground truth is given, in the validation loss too; the proposals returned for the proposal recall and the test predictions, made without ground truth, are left as they are. `batch_assign_targets_to_proposals()` then gives every proposal the class of the ground truth box it overlaps most when their IoU reaches `Detector.foreground_iou_threshold` (0.5), and the background (0) otherwise; `turn_boxes_to_transformation_targets()` computes the offsets that would move the proposal onto its ground truth box; `sample_positive_and_negative_training_targets()` keeps, for every image, at most `number_training_positives` (64) foreground proposals and fills `total_training_samples` (128) with background ones, chosen at random. The kept proposals are written `[image index, x_min, y_min, x_max, y_max]` for `roi_align()`. In inference, every filtered proposal is kept.

**Example output**
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
`batch_assign_targets_to_anchors()` labels every anchor of every image of the batch at once (`pad_ground_truths()` completes the ground truths of every image with empty boxes up to the largest number of objects of the batch) from its best IoU with a ground truth box: foreground from `RegionProposal.foreground_iou_threshold.min` (0.5), background between `background_iou_threshold.min` and `.max` (0.1 to 0.3), ignored otherwise (between 0.3 and 0.5, and below 0.1); the anchors overlapping a ground truth box the most are foreground whatever their IoU (above 0), so that every box has at least one anchor. `sample_positive_and_negative_training_targets()` keeps at most `number_training_positives` (128) foreground anchors per image, filled to `total_training_samples` (256) with background ones, drawn at random for all the images with one sort of random keys. The classification loss is the binary cross entropy of the objectness of the sampled anchors; the regression loss is the smooth L1 (`localisation_loss_beta` 1/9) between the predicted and target offsets of the sampled foreground anchors, summed and divided by their number.

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
The classification loss is the cross entropy of the class logits of the sampled proposals, the background ones teaching the detector to reject false positives. The regression loss is the smooth L1 (`Detector.localisation_loss_beta` 1/9) between the target offsets and the offsets predicted for the true class of every foreground proposal, summed and divided by their number (0 when no sampled proposal is foreground).

**Example output**
```text
losses: classification 1.0863   regression 1.4856
```

### Step 3.19 — `compute_batch_metrics()` + `MetricTracker.accumulate()` + `log_to_tensorboard()`
The detection has no batch metric (`{}`): the four losses and their sum are the tracked metrics.

**Example: the Train curves**
```text
Total Loss                     1: 4.4665, 2: 4.2275, 3: 4.0651, 4: 4.2386, 5: 3.5706, 6: 3.5331, 7: 3.3495, 8: 4.8642, 9: 3.7650, 10: 3.1320, 11: 3.9600, 12: 2.9482
RPN Classification Loss        1: 0.7244, 2: 0.7241, 3: 0.7187, 4: 0.7128, 5: 0.7092, 6: 0.7015, 7: 0.6878, 8: 0.6983, 9: 0.6935, 10: 0.6827, 11: 0.6800, 12: 0.6843
Detection Classification Loss  1: 2.3117, 2: 2.1153, 3: 1.9311, 4: 1.7544, 5: 1.5924, 6: 1.5544, 7: 1.4130, 8: 1.5061, 9: 1.3203, 10: 1.2446, 11: 1.1507, 12: 1.0008
Detection Regression Loss      1: 0.6198, 2: 0.4891, 3: 0.6161, 4: 1.0374, 5: 0.6583, 6: 0.6306, 7: 0.6013, 8: 1.8935, 9: 1.0498, 10: 0.6159, 11: 1.5341, 12: 0.8023
```

### Step 3.20 — `training_loss.backward()` + `step_optimizer()`
**Example output** (`gradient_clipping_norm: null`, so the gradients are not clipped)
```text
gradient norm: iteration 1: 42.3099   iteration 2: 40.7255   iteration 3: 33.9392        learning rate 0.0002 (constant)
```

### Step 3.21 — validation pass
At iteration 10, a validation batch runs in `eval()` mode with its ground truth: the filter keeps the inference numbers of proposals (500, then 250), the detector samples its targets among them and returns its four losses.

**Example output**
```text
Tensorboard/Validation at iteration 10: Total Loss 3.4132, RPN Classification Loss 0.6929, RPN Regression Loss 0.5775,
                                        Detection Classification Loss 2.1428, Detection Regression Loss 0.0000
```
The detection regression loss is 0: no proposal sampled from this batch reached an IoU of 0.5 with a ground truth box.

### Step 3.22 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()`
**Example: the sums and the averages of the first interval**
```text
before: {'interval_batches': 4, 'total_loss': 16.9977, 'region_proposal_classification_loss': 2.88, 'region_proposal_localisation_loss': 3.2428,
         'detection_classification_loss': 8.1126, 'detection_regression_loss': 2.7623}
after:  {'interval_batches': 4, 'total_loss': 4.2494, 'region_proposal_classification_loss': 0.72, 'region_proposal_localisation_loss': 0.8107,
         'detection_classification_loss': 2.0281, 'detection_regression_loss': 0.6906}
MLflow, step 4: {'train/Total Loss': 4.2494, 'train/RPN Classification Loss': 0.72, 'train/RPN Regression Loss': 0.8107,
                 'train/Detection Classification Loss': 2.0281, 'train/Detection Regression Loss': 0.6906,
                 'train/throughput': 1.028, 'train/interval_seconds': 7.782, 'learning_rate': 0.0002}
```

**Example output: iterations 4 and 12**
```text
========================================================================================================================
|     Iteration: 4     |      Total Images: 8      |     Interval Images: 8     |  Throughput: 1.0 Images Per Second   |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   4.2494
Moving Average of Train RPN Classification Loss        :   0.7200
Moving Average of Train RPN Regression Loss            :   0.8107
Moving Average of Train Detection Classification Loss  :   2.0281
Moving Average of Train Detection Regression Loss      :   0.6906
------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:07.
Validation Iterations In This Interval: 0 (Ratio: Every 10 Training Iterations).
========================================================================================================================
...
========================================================================================================================
|    Iteration: 12     |      Total Images: 24     |     Interval Images: 8     |  Throughput: 1.2 Images Per Second   |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   3.4513  -0.3780 Best
Moving Average of Valid Total Loss                     :   3.4132
Moving Average of Train RPN Classification Loss        :   0.6851  -0.0141 Best
Moving Average of Valid RPN Classification Loss        :   0.6929
Moving Average of Train RPN Regression Loss            :   0.5866  -0.0812 Best
Moving Average of Valid RPN Regression Loss            :   0.5775
Moving Average of Train Detection Classification Loss  :   1.1791  -0.3374 Best
Moving Average of Valid Detection Classification Loss  :   2.1428
Moving Average of Train Detection Regression Loss      :   1.0005  +0.0546
Moving Average of Valid Detection Regression Loss      :   0.0000
------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:06.
Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
========================================================================================================================
```

### Step 3.23 — end of the loop
**Example output**
```text
Model successfully saved at iteration 12. Exiting Training.
Iteration 12 · 20 Test Images · 2026-10-10 07:40 ───────────────      <- the evaluation of the final checkpoint
```


---

## Stage 4 — Test Evaluation of a Checkpoint

> The examples below are real outputs of the evaluation of the checkpoints of iterations 6 and 12 of the run of Stage 3, on its whole test split (20 images, 57 boxes), with `per_class_report: True`, `number_sample_images: 8`, `score_threshold: 0.05`, `target_precision: 0.99`, the error images off and the MLflow tracking on. Both checkpoints find no shape: at iteration 12 the 10 class probabilities of a proposal are all close to 0.1.

### Step 4.1 — `run_test_evaluation()` → `evaluation_context()`
The evaluation loop of every task (see `classification_workflow.md`, Step 4.1). The `evaluation_context()` of the detection lowers the score threshold of the model from the deployment one (`Detector.inference.score_threshold`, 0.5) to `evaluation.score_threshold` (0.05) during the evaluation, so that the precision-recall curves keep their low-confidence end, and restores it afterwards, even when the evaluation fails.

**Example output: the progress bar** (20 images, batch size 2)
```text
Test Evaluation Iteration 12: 100%|██████████| 10/10 [00:03<00:00,  2.65it/s]
```

### Step 4.2 — `create_evaluation_accumulator()`
The detections and the ground truth of every test image, kept on the CPU in network input pixels, the confusion matrix of the classes (background included), filled image by image (Step 4.5), the folder and counts of the error images, and the `proposal_recall` accumulator of Step 4.15.

**Example: the accumulator at the end of the loop of iteration 12**
```text
{'detections': 20 entries {'boxes': (100, 4), 'scores': (100,), 'labels': (100,)}, 'ground_truths': 20 entries {'boxes', 'labels'},
 'confusion_matrix': (10, 10) int64, 'errors_folder': .../Evaluation/Iteration_12/Errors, 'error_counts': {}}
detections per image   [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
ground truths per image [1, 6, 1, 3, 6, 4, 4, 6, 1, 1, 2, 2, 1, 3, 1, 1, 3, 6, 3, 2]
first image: best scores [0.1091, 0.1091, 0.1091, 0.1091, 0.1091]   labels [9, 9, 9, 9, 9]   ground truth [[281.0, 184.0, 426.0, 336.0]] label [5]
```

### Step 4.3 — `prepare_batch()` + `predict_batch()`
`predict_batch()` runs the detector without the ground truth, as in deployment: with it, the detector would only classify the proposals sampled around the true boxes, which would inflate the mAP. In `eval()` mode, the filter keeps 250 proposals per image and the outputs go through `postprocess_detections()` (Step 4.4). The outputs also keep the filtered proposals (`proposal_boxes` [2, 250, 4], `proposal_levels` and `valid_proposal_mask` [2, 250]) for the proposal recall (Step 4.15).

**Example: the predictions of a test batch**
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
The softmax of the class logits of every proposal, the background left out, gives one candidate per proposal and foreground class, with its box clamped to the image. One `batched_nms()` call suppresses, per image and class, the boxes overlapping a better one by more than `Detector.nms_iou_threshold` (0.8); the survivors reaching the score threshold are kept, the `maximum_detections` (100) best of every image returned, and the empty places padded with a score of 0, a label of 0 and an empty box. The code has no data-dependent Python control flow, so that the same function is exported to TorchScript and ONNX.

**Example: the first training batch in inference mode, at the deployment threshold 0.5, at the end of the run**
```text
number_detections [0, 0]
detection 0 of image 0: box [0.0, 0.0, 0.0, 0.0]   score 0.0000   label 0        <- padding
class probabilities of the first proposal of image 0: [0.1327, 0.0894, 0.0938, 0.093, 0.1029, 0.1006, 0.0877, 0.0958, 0.0952, 0.109]
```
No foreground class reaches 0.5, so the deployed detector returns nothing; at the evaluation threshold 0.05, every test image gets its 100 detections (Step 4.3), all of them `octagon`, the most probable foreground class (Step 4.8).

### Step 4.5 — `accumulate_evaluation_batch()` → `match_image_detections()` + `add_to_confusion_matrix()` + `dump_errors()`
For every image of the batch, the `number_detections` valid detections (boxes, scores, labels) and the ground truth boxes and labels are kept on the CPU for the mean average precision. The detections reaching the deployment threshold (`Detector.inference.score_threshold`), those the deployed detector returns, are then matched to the objects of the image for the confusion matrix. `compute_overlap_matrix()` gives the IoU of every detection with every object, whatever their classes, and `match_image_detections()` goes twice through the detections from the most to the least confident, each one taking the free object that overlaps it the most, at least by 0.5: first among the objects of its class, so that an object found with its class counts as found even when a detection of another class overlaps it more, then among the objects left of any class (the confusions). `add_to_confusion_matrix()` counts every matched detection in the cell of the class of its object and of its own class, every detection left without object in the background row (a false positive) and every object left without detection in the background column (missed). The mean average precision keeps its own matching, per class and down to `evaluation.score_threshold` (Step 4.9). At iteration 12, no detection reaches 0.5, so the 57 objects of the test split go to the background column.

With `error_dumps.activated`, `dump_errors()` saves the image once in the folder of every kind of error it holds (`group_image_errors()`): `Errors/FALSE-POSITIVE/<class>` for the detections of a class finding no object, `Errors/MISSED/<class>` for the objects of a class found by no detection, and `Errors/<TRUE>-TO-<PREDICTED>` for the objects found with another class. `draw_test_image()` draws it like a sample image (Step 4.6), the boxes of these errors outlined with red dashes (an `Error` entry heading the legend) and the errors described at the start of the caption. The file is named after the image and the highest score of its detections in error (the image name alone for the missed objects), with at most `maximum_per_class_pair` images per folder, the first ones met.

**Example: the error images of the detection smoke test** (`tests/trainers/test_detection_trainer.py`: 4 test images of 256 pixels, 5 iterations, the deployment threshold lowered to 0.05 so that the detector returns detections, at most 2 images per folder; the run of Stage 3 returns no detection at 0.5)
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

**Example: `Evaluation/Iteration_12/Samples/test_000003.png`** (1028 x 560 pixels = 2 x 512 + 4 by 24 + 512 + 24)
```text
caption: "3 ground truth boxes | 0 detections with score >= 0.50"
panels:  Ground Truth (circle, octagon, triangle boxes) | Prediction (no box)
legend:  circle, triangle, octagon
build_sample_image -> [560, 1028, 3] torch.uint8
```
No detection is drawn at iterations 6 and 12, so the sample images of both evaluations are identical (same file sizes).

### Step 4.7 — `compute_evaluation_results()`
`evaluate_detections()` computes the mean average precision of the test split with the three average precision methods (Steps 4.8 to 4.10) and of every object size (Step 4.14), then the average precisions of every class go to `class_metrics.csv` (Step 4.11), `save_proposal_recall()` writes the proposal recall (Step 4.15) and `save_error_analysis()` saves the class thresholds and the confusion matrix (Step 4.12). The detection has no headline metric line (`metrics` is empty): the report shows the table of the methods, the table of the object sizes, the proposal recall and the per-class table. The checkpoints are compared on the `mAP@0.5/area` scalar. The example below comes from the first run, before the object sizes and the proposal recall: the later run adds the scalars, tables and file of Steps 4.14 and 4.15.

**Example: the results of iteration 12**
```text
evaluated_samples = '20 Test Images'
metrics           = {}
notes             = ['Confusion Matrix : score >= 0.50, IoU >= 0.5 (boxes), 0 right, 0 wrong class, 0 false positives, 57 missed',
                     'Sample Images : 8 fixed test images']
headline_metric   = 'mAP@0.5/area'
files             = ['Class Metrics', 'Class Thresholds', 'Confusion Matrix Table', 'Confusion Matrix Image', 'Sample Images']
images            = ['Confusion Matrix']
scalars           = {'mAP@0.5/area': 0.0, 'mAP@0.75/area': 0.0, 'mAP@[0.5:0.95]/area': 0.0, 'mAP@0.5/11_point': 0.0, ..., 'mAP@[0.5:0.95]/101_point': 0.0}
class_scalars     = 9: AP@0.5/square, AP@0.5/circle, ..., AP@0.5/octagon
tables            = Mean Average Precision: Method, mAP@0.5, mAP@0.75, mAP@[0.5:0.95]
                    Per Class Average Precision (Area): Class, Ground Truths, AP@0.5, AP@0.75, AP@[0.5:0.95], Threshold @ P0.99,
                    Recall @ P0.99
```

### Step 4.8 — `gather_class_detections()` → `find_best_ground_truth()`
For every foreground class, once for all the thresholds and methods: the detections of the class of every image, sorted by decreasing score, the best IoU of every detection with a ground truth box of the same class in the same image (0 when the image has none), and the number of ground truth boxes of the class.

**Example: the class with the most detections at iteration 12**
```text
detections per class {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 2000}
class 9 (octagon): 2000 detections, 7 ground truths
   image of the first 10 detections [11, 11, 11, 11, 11, 11, 11, 11, 11, 11]
   best IoU of the first 10 detections [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
   largest best IoU 0.4701, detections with a best IoU >= 0.3: 7, >= 0.5: 0
```

### Step 4.9 — `match_detections_to_ground_truth()` + `compute_precision_recall()`
At one IoU threshold, the detections are visited from the most to the least confident: a detection is a true positive when its best ground truth reaches the threshold and was not taken by a more confident detection, a false positive otherwise (the PASCAL VOC rule). Every detection is then one point of the precision-recall curve: the precision and the recall if the score threshold were its score.

**Example: octagon at 0.5, iteration 12**
```text
true positives 0 of 2000 detections
precisions of the first 5 detections [0.0, 0.0, 0.0, 0.0, 0.0]   recalls [0.0, 0.0, 0.0, 0.0, 0.0]   last point: precision 0.0, recall 0.0
```

### Step 4.10 — `compute_average_precision()` + `compute_mean_average_precision()`
The average precision of a class summarises its curve: `area`, the area under the curve once every precision is raised to the best precision at a higher recall (PASCAL VOC 2010 and later); `11_point`, the mean of the interpolated precision at the recalls 0, 0.1, ..., 1 (PASCAL VOC 2007); `101_point`, the same at 101 recall levels (COCO). The mAP of a threshold is the mean over the classes that have ground truth boxes; `evaluate_detections()` gives, for every method, the mAP at 0.5, at 0.75, and averaged over the 10 thresholds 0.50, 0.55, ..., 0.95 (`mAP@[0.5:0.95]`), with the same three values for every class.

**Example: iteration 12**
```text
octagon: AP area 0.0000   AP 11_point 0.0000   AP 101_point 0.0000
area:      mAP@0.5 0.0000   mAP@0.75 0.0000   mAP@[0.5:0.95] 0.0000
11_point:  mAP@0.5 0.0000   mAP@0.75 0.0000   mAP@[0.5:0.95] 0.0000
101_point: mAP@0.5 0.0000   mAP@0.75 0.0000   mAP@[0.5:0.95] 0.0000
class_ground_truths {1: 4, 2: 8, 3: 9, 4: 3, 5: 14, 6: 6, 7: 4, 8: 2, 9: 7}
```

### Step 4.11 — `build_class_average_precision_rows()` + `write_class_metrics_file()`
The average precisions of every foreground class with the area method, with the score threshold of the class reaching the target precision and the recall it keeps (Step 4.12), empty when the precision is never reached, saved whatever the per-class report.

**Example: `Evaluation/Iteration_12/class_metrics.csv`**
```text
class_index,class,ground_truths,average_precision_50,average_precision_75,average_precision_50_95,threshold_at_target_precision,recall_at_target_precision
1,square,4,0.000000,0.000000,0.000000,,
2,circle,8,0.000000,0.000000,0.000000,,
...
9,octagon,7,0.000000,0.000000,0.000000,,
```

Since the object sizes (Step 4.14), the AP@0.5 and AP@[0.5:0.95] of every size follow the average precisions of all the objects, empty for a size the class has no object of.

**Example: `Evaluation/Iteration_12/class_metrics.csv` of the later run** (no object under 32² pixels)
```text
class_index,class,ground_truths,average_precision_50,average_precision_75,average_precision_50_95,average_precision_50_small,average_precision_50_95_small,average_precision_50_medium,average_precision_50_95_medium,average_precision_50_large,average_precision_50_95_large,threshold_at_target_precision,recall_at_target_precision
1,square,4,0.000000,0.000000,0.000000,,,0.000000,0.000000,0.000000,0.000000,,
2,circle,5,0.000000,0.000000,0.000000,,,0.000000,0.000000,0.000000,0.000000,,
```

### Step 4.12 — `find_class_score_thresholds()` + `save_error_analysis()`
With a target precision, `evaluate_detections()` also gives the score threshold of every class. From the matching of the AP@0.5 (Step 4.9), `find_score_threshold()`, the function of the class thresholds of the classification, keeps the lowest score whose detections of the class at or above it reach the target precision, i.e. the highest recall at this precision. The recall counts every ground truth box of the class, those found by no detection included, and only the detections down to `evaluation.score_threshold` can give a threshold. `save_error_analysis()` saves these thresholds in `class_thresholds.json`, in the format of the classification tasks (`null` for a class whose detections never reach the precision), and the confusion matrix of Step 4.5 in `confusion_matrix.csv`, rows the true classes and columns the predicted classes, the background first. With the per-class report, the confusion matrix is also drawn in `confusion_matrix.png` and logged to TensorBoard: its background cell is left blank, and its background row and column have no recall or precision. The report gets a note counting the outcomes of the detections and, with the error images on, a note counting the images and their folders.

At iteration 12, the deployed detector returns nothing and no class reaches the precision 0.99: every class threshold is `null`, and the confusion matrix has the 57 objects in its background column (`0 right, 0 wrong class, 0 false positives, 57 missed`).

**Example: the files of the detection smoke test** (the run of Step 4.5, 9 objects in 4 test images, every detection reaching 0.05)
```text
confusion_matrix.csv
true \ predicted,background,square,circle,triangle,rectangle,pentagon,hexagon,plus_sign,star,octagon
background,0,168,52,59,60,60,0,0,0,0
square,1,0,0,0,0,0,0,0,0,0
circle,1,0,0,0,0,0,0,0,0,0
triangle,0,0,0,0,0,0,0,0,0,0
rectangle,1,0,0,0,0,0,0,0,0,0
pentagon,1,0,0,0,0,0,0,0,0,0
hexagon,2,0,0,0,0,0,0,0,0,0
plus_sign,0,0,0,1,0,0,0,0,0,0
star,0,0,0,0,0,0,0,0,0,0
octagon,3,0,0,0,0,0,0,0,0,0

class_thresholds.json
{"target_precision": 0.99, "classes": {"square": null, "circle": null, "triangle": null, ..., "octagon": null}}

notes:
  Confusion Matrix : score >= 0.05, IoU >= 0.5 (boxes), 0 right, 1 wrong class, 399 false positives, 9 missed
  Error Images : 19 in 12 folders, at most 2 per folder
confusion_matrix.png (1192 x 1155 pixels)
```
The plus_sign found as a triangle is the confusion of Step 4.5; the 8 other objects are missed, and the 399 other detections overlap no object enough.

### Step 4.13 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
**Example output: the block of iteration 12** (the same text is appended to `Results.txt`)
```text
Iteration 12 · 20 Test Images · 2026-10-10 07:40 ───────────────────────────────────────────────────────────────────────
  Confusion Matrix : score >= 0.50, IoU >= 0.5 (boxes), 0 right, 0 wrong class, 0 false positives, 57 missed
  Sample Images : 8 fixed test images

  Mean Average Precision
   Method      mAP@0.5   mAP@0.75   mAP@[0.5:0.95]
  ─────────────────────────────────────────────────
   Area         0.0000     0.0000           0.0000
   11 Point     0.0000     0.0000           0.0000
   101 Point    0.0000     0.0000           0.0000
  The mAP@0.5 of the area method compares the checkpoints of the run.

  Per Class Average Precision (Area)
               Ground                                      Threshold    Recall
   Class       Truths   AP@0.5   AP@0.75   AP@[0.5:0.95]     @ P0.99   @ P0.99
  ─────────────────────────────────────────────────────────────────────────────
   square           4   0.0000    0.0000          0.0000           —         —
   circle           8   0.0000    0.0000          0.0000           —         —
   ...
   octagon          7   0.0000    0.0000          0.0000           —         —
  Worst AP@0.5 first, ties in class order. — : no ground truth box of the class, or the precision 0.99 is never reached.
  - Results : file:///.../Results.txt
  - Class Metrics : file:///.../Evaluation/Iteration_12/class_metrics.csv
  - Class Thresholds : file:///.../Evaluation/Iteration_12/class_thresholds.json
  - Confusion Matrix Table : file:///.../Evaluation/Iteration_12/confusion_matrix.csv
  - Confusion Matrix Image : file:///.../Evaluation/Iteration_12/confusion_matrix.png
  - Sample Images : file:///.../Evaluation/Iteration_12/Samples
```

**Example: TensorBoard and MLflow after the two evaluations**
```text
Tensorboard/Test: 18 scalars (9 mAP + 9 AP@0.5 per class), 9 images (the samples and the confusion matrix)
    scalar mAP@0.5/area      6: 0.0000, 12: 0.0000
    scalar AP@0.5/octagon    6: 0.0000, 12: 0.0000
MLflow: test/mAP_at_0.5/area, test/mAP_at_0.75/area, test/mAP_at_0.5-0.95/area, ... (3 methods),
        test/best_mAP_at_0.5/area (steps 6, 12), 9 test_class/AP_at_0.5/<class>
artifacts: Results.txt, configuration/training_configuration.yaml, metric_names.json,
           evaluation/iteration_<6|12>/confusion_matrix.png, class_metrics.csv, class_thresholds.json
```
`sanitise_metric_name()` writes `@` as `_at_`, drops the brackets and writes `:` as `-`, so `mAP@[0.5:0.95]/area` becomes `mAP_at_0.5-0.95/area` in MLflow (the original names are kept in `metric_names.json`).

### Step 4.14 — `select_object_size_detections()` + `build_object_size_table()`
`evaluate_detections()` computes the same mean average precisions again for the small, medium and large objects of COCO, with every method and threshold, reusing the gathered detections of Step 4.8 and the matching of Step 4.9. The size of an object is the area of its box in network input pixels (`OBJECT_SIZE_RANGES` in `object_sizes.py`): small below 32², medium from 32² to 96², large from 96². `select_object_size_detections()` keeps the detections of a class that count for a size with the COCO rule: the ground truths of the other sizes are ignored, and so are the detections matched to them (their best ground truth reaching the threshold) and the unmatched detections whose own box is of another size. A detection whose best ground truth of the size was already taken by a more confident detection stays a false positive, as without the sizes. A class without any object of the size is left out of the mean of that size, and a size without any object has no mAP: `—` in the table and no scalar. `build_object_size_table()` shows the mAPs of the area method of every size with its number of objects, after the table of the methods. The scalars `mAP@0.5 By Size/<size>` and `mAP@[0.5:0.95] By Size/<size>` go to TensorBoard and MLflow, and the values of every class to `class_metrics.csv` (Step 4.11). The instance segmentation has two of these tables, by mask area for the mask AP and by box area for the box AP.

**Example output: the table of iteration 400 of the second run** (66 test objects, none under 32² pixels: the generator draws no shape under 64 pixels, `MINIMUM_SHAPE_SIZE`)
```text
  Mean Average Precision By Object Size (Area)
   Object Size           Objects   mAP@0.5   mAP@0.75   mAP@[0.5:0.95]
  ─────────────────────────────────────────────────────────────────────
   Small (< 32²)               0         —          —                —
   Medium (32² to 96²)        35    0.0000     0.0000           0.0000
   Large (>= 96²)             31    0.0000     0.0000           0.0000
  COCO sizes by box area in network input pixels, the objects of the other sizes and the detections finding them being
  ignored. — : no object of the size.
```
```text
Tensorboard/Test: 49 scalars (9 mAP + 20 AP@0.5 per class + 4 by size + 16 proposal recall of Step 4.15)
    mAP@0.5 By Size/medium, mAP@[0.5:0.95] By Size/medium, mAP@0.5 By Size/large, mAP@[0.5:0.95] By Size/large
MLflow: test/mAP_at_0.5 By Size/medium, test/mAP_at_0.5-0.95 By Size/medium, test/mAP_at_0.5 By Size/large,
        test/mAP_at_0.5-0.95 By Size/large
```
Without small objects, the small size has no scalar. After 400 iterations the detection heads still find none of the test objects, so every mAP is 0; the proposal recall of Step 4.15 shows how far the region proposal network got.

### Step 4.15 — `accumulate_proposal_recall()` + `save_proposal_recall()`
The anchor recall of Step 2.21 measures what the anchors allow; the proposal recall measures what the trained region proposal network gives the detection heads. `create_evaluation_accumulator()` creates its accumulator for the detection levels (`create_proposal_recall_accumulator()`), and `accumulate_evaluation_batch()` calls `accumulate_proposal_recall()` for every test image with the filtered proposals of Step 4.3: the padding copies are left out with `valid_proposal_mask`, the best IoU of every object is kept with all the proposals and with the proposals of every level alone, computed on the device of the model, and the valid proposals of every level are counted. In `compute_evaluation_results()`, `save_proposal_recall()` calls `summarise_proposal_recall()`, which gives the share of the objects whose best IoU reaches 0.5 and 0.7 (`PROPOSAL_RECALL_THRESHOLDS`) for all the objects, every size (by box area, Step 4.14) and the proposals of every block alone, with the mean number of valid proposals per image of all the blocks and of every block. A block row counts all the objects, so its recall shows how many objects that level brings to the detection heads. The table is always shown after the size table; the rows are written to `proposal_recall.csv` in the evaluation folder of the checkpoint, which is sent to MLflow, and the scalars `Proposal Recall@0.5/<group>`, `Proposal Recall@0.7/<group>` and `Proposals Per Image/<group>` go to TensorBoard and MLflow, a size without object having none. The instance segmentation shows the same table once, after its box size table.

**Example output: the table of iteration 400 of the second run**
```text
  Proposal Recall
                                                           Proposals
   Objects               Count   Recall@0.5   Recall@0.7   Per Image
  ───────────────────────────────────────────────────────────────────
   All Objects              66       0.4848       0.1818     72.1500
   Small (< 32²)             0            —            —           —
   Medium (32² to 96²)      35       0.1429       0.0286           —
   Large (>= 96²)           31       0.8710       0.3548           —
   Block 5 Proposals        66       0.4697       0.1818     68.4000
   Block 6 Proposals        66       0.0152       0.0000      3.0500
   Block 7 Proposals        66       0.0000       0.0000      0.7000
  Share of the test objects overlapped by a proposal of the region proposal network by at least the IoU, sizes by box
  area in network input pixels. A block row keeps the proposals of that block only. — : no object of the size.
```
```text
Evaluation/Iteration_400/proposal_recall.csv
group,objects,recall_50,recall_70,proposals_per_image
all,66,0.484848,0.181818,72.150000
small,0,,,
medium,35,0.142857,0.028571,
large,31,0.870968,0.354839,
block_5,66,0.469697,0.181818,68.400000
block_6,66,0.015152,0.000000,3.050000
block_7,66,0.000000,0.000000,0.700000

MLflow: test/Proposal Recall_at_0.5/<group>, test/Proposal Recall_at_0.7/<group>, test/Proposals Per Image/<group>,
        evaluation/iteration_400/proposal_recall.csv
```
At the end of the 12-iteration run of Step 2.21, the 250 proposals of every image all came from block 5 and reached none of the 66 objects. At iteration 400, the non-maximum suppression leaves 72.15 valid proposals per image of the 250 slots, the others being padding copies, and 68.4 of them come from block 5. The proposals reach 27 of the 31 large objects at 0.5 but only 5 of the 35 medium ones, although Step 2.21 gave a foreground anchor to 80% of the medium training objects; at 0.7 they reach 12 of the 66 objects. The proposals of blocks 6 and 7 reach almost none of them.


---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of the logic of `scripts/evaluation/evaluate_checkpoint.py` run on the run of Stage 3 (checkpoints 3, 6, 9 and 12) with `iteration = "all"`, `batch_size = 4` and, to keep it short, a test folder of the first 10 images of the test split (its `labels.json` reduced to these images and their annotations, its `data/` folder linked).

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The same script for every task (see `classification_workflow.md`, Stage 5).

### Step 5.2 — trainer construction
**Example: the split line of the launch information**
```text
- Test Data Folder        : file:///.../test_reduced (10 images, 17 annotations)
```

### Step 5.3 — `get_checkpoint_iterations()`
**Example output**
```text
evaluated_iterations = [12, 9, 6, 3]
```

### Step 5.4 — `evaluate_checkpoints()` → `start_evaluation_tracking()`
**Example output**
```text
MLflow Run Created In The Experiment 'object_detection' : fcc494bfb83d4bc78638fa8ef3947ea6 (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```

### Step 5.5 — `summarise_checkpoint_evaluations()`
**Example output** (the four checkpoints tie at 0, so the first one evaluated, 12, is kept)
```text
Evaluation Summary Of 4 Checkpoints : file:///.../runs/evaluations/DET_IN-KAR-512_..._MC9/Evaluation/evaluation_summary.csv
  - Best Checkpoint : iteration 12 (mAP@0.5/area 0.0000)

iteration,mAP@0.5/area,mAP@0.75/area,mAP@[0.5:0.95]/area,mAP@0.5/11_point,mAP@0.75/11_point,mAP@[0.5:0.95]/11_point,mAP@0.5/101_point,
mAP@0.75/101_point,mAP@[0.5:0.95]/101_point,AP@0.5/circle,AP@0.5/triangle,AP@0.5/rectangle,AP@0.5/pentagon,AP@0.5/hexagon,AP@0.5/plus_sign,
AP@0.5/star,AP@0.5/octagon
12,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
...
3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
```
The 10 images hold no square, so the summary has no `AP@0.5/square` column.

### Step 5.6 — `run_test_evaluation()` for each checkpoint
**Example output: the per-class table of the checkpoint of iteration 12 on the 10 images** (shortened)
```text
Evaluating : file:///.../runs/DET_IN-KAR-512_..._MC9/Weights/Iteration_12/model_000012.pt
Iteration 12 · 10 Test Images · 2026-10-10 07:40 ──────────────────────────────────────────────────────────────────────
  ...
  Per Class Average Precision (Area)
               Ground
   Class       Truths   AP@0.5   AP@0.75   AP@[0.5:0.95]
  ───────────────────────────────────────────────────────
   circle           3   0.0000    0.0000          0.0000
   ...
   octagon          3   0.0000    0.0000          0.0000
   square           0        —         —               —
  Worst AP@0.5 first, ties in class order. — : no ground truth box of the class.
```
A class without ground truth box has no recall, so it has no average precision (`—`), is left out of the mAP and is listed last.
