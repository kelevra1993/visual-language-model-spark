# Instance Segmentation Workflow

This document follows one instance segmentation run (INS) from `main()` to the evaluation of a checkpoint, to show which function runs when, what it receives, what it returns and who uses the result next. The **Call Tree Overview** gives the whole run on one page. Each stage then has one section, with one step per function: a short explanation followed by a **real** output. The steps shared by every task (configuration loading, checkpoints, training loop, MLflow tracking, standalone evaluation) are explained in detail in `classification_workflow.md`, and the steps of the two-stage detector (anchors, region proposals, target assignment, detection heads, post-processing, box mean average precision, proposal recall) in `object_detection_workflow.md`; here they keep a short explanation and the real output of the instance segmentation run. The steps specific to the instance segmentation (instance masks loaded from the COCO segmentations, mask heads, mask targets and mask loss, masks of the final detections, mask overlaps and mask mean average precision) are explained in full.

The examples come from real runs on CPU, so that the GPU stayed free for another training. The synthetic dataset of the configuration, `synthetic_shapes`, is not on this machine, so a small one was generated in a temporary folder with `create_synthetic_dataset(task="instance_segmentation")`: 100 training, 20 validation and 20 test images of 512 pixels, at most 6 shapes per image, 20 shape classes, every object segmented by an uncompressed run length encoding (RLE) of its visible pixels. The data folders and the mapping file of the configuration were redirected to it; the configuration already uses the Native loader.
- **Stages 1 and 2**: `configurations/instance_segmentation/configuration_synthetic_shapes.yaml` (512 pixel images, batch size 40, detection and mask heads on the blocks 5, 6 and 7 of the backbone), with the run folder redirected to a temporary folder.
- **Stages 3, 4 and 5**: the same configuration shortened to batch size 2, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4, a validation batch every 10, 2 loading workers, with the MLflow tracking turned on. The model, the image size and the masks are those of Stages 1 and 2. After 12 iterations the detector finds no shape, so the metrics of the examples are 0 on purpose: the point is to see the shapes, the files and the order of the steps.

> The examples were recorded before the following changes, which they do not show: the region proposal NMS is 0.7 and the detector NMS 0.5 (they were 0.5 and 0.8), the anchors overlapping no object (IoU 0) can be sampled as background (`RegionProposal.background_iou_threshold.min: 0.0`), and the evaluation shows the mask and box `Mean Average Recall` tables, and can score the masks at the original resolution (`evaluation.original_resolution`), after the size tables. The synthetic configurations now train at 512 pixels with smaller heads, batch size 64 and 2000 iterations, the object detection one reading the instance segmentation dataset.

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
│   Entry point of every task. Give an instance_segmentation configuration with --configuration_path to run this workflow.
│
├── load_experiment_configuration(configuration_path)
│   │   Splits the YAML into experiment_configuration ('ExperimentConfiguration') and model_configuration (Optimization,
│   │   Data, Backbone, DetectionLevels, RegionProposal, Detector, MaskHead), converts the types (localisation_loss_beta
│   │   '1/9' -> 0.111...), adds label_mode and label_dictionary.
│   │
│   ├── load_configuration(configuration_path)
│   ├── build_label_dictionary(label_mapping_configuration, task)
│   │       Reads the mapping file {category id: class index} and adds 'background': 0, since the detector predicts
│   │       the background as class 0.
│   └── check_configuration()
│       ├── common checks
│       └── check_instance_segmentation_configuration()
│           ├── check_detection_configuration()                  every DetectionLevels.<level> also holds a mask_head
│           └── check_mask_head_configuration()                  mask_resolution, upsampling, pool size <= resolution,
│                                                                convolutions, normalization, modules, mask_threshold,
│                                                                pretrained
│
├── get_trainer_class(task)  -> InstanceSegmentationTrainer
│
├── InstanceSegmentationTrainer(experiment_configuration, model_configuration, configuration_path)   [STAGE 2]
│   │   LOAD_INSTANCE_MASKS = True, MATCHING_OVERLAP_TYPE = "masks", reads MaskHead.inference.mask_threshold.
│   │
│   └── DetectionTrainer.__init__()
│       │   Reads evaluation.score_threshold, Detector.inference.score_threshold, target_precision, error_dumps.
│       │
│       └── BaseTrainer.__init__()
│           ├── get_device() / get_run_name() -> build_run_name() -> build_instance_segmentation_sections()
│           ├── DataAugmenter() / setup_training_paths()
│           ├── build_dataloaders()                                  class_names, class_colours, test image names
│           │   └── get_detection_dataloaders(load_instance_masks=True)
│           │       └── CocoDetectionDataset(load_instance_masks=True) + collate_function()
│           │           └── select_image_objects(boxes_from_segmentation=True) -> compute_segmentation_bounding_box()
│           │                   the box of every object read from its polygons or RLE, not from 'bbox'
│           ├── build_native_loader() / DataLoader
│           ├── save_data_record()                                   data.json
│           ├── get_test_sample_classes() + choose_sample_identifiers()
│           ├── build_model()
│           │   └── InstanceSegmenter.__init__()
│           │       ├── TwoStageDetector.__init__()                  backbone, RegionProposal and DetectionHead per level,
│           │       │                                                anchors, RegionProposalFilter
│           │       └── _build_mask_heads_dictionary()
│           │           └── MaskHead() for every level -> count_mask_upsamplings()
│           │                   RoIAlign -> ConvolutionBlock -> doublings (ConvTranspose2d, or bilinear + Conv2d) ->
│           │                   1x1 mask_predictor -> bilinear resize when the doublings miss mask_resolution
│           ├── initialise_pretrained_components()                   Backbone, RegionProposal, Detector, MaskHead
│           ├── build_optimizer() / get_tracked_metrics()            5 losses + Total Loss
│           ├── print_experiment_information() / print_summary()     backbone, RPN, detection, mask, receptive field,
│           │                                                        model size panels
│           ├── view_input_batches() -> get_input_sample_image() -> overlay_instance_masks()
│           ├── analyse_training_data()                              anchor recall on the boxes of the segmentations
│           └── restore_last_model()
│
└── trainer.run_training_loop()   [STAGE 3]
    └── run_training_iterations()
        ├── for each training_iteration:
        │   ├── save_model() + run_test_evaluation() when due                         -> [STAGE 4]
        │   ├── get_next_batch() -> CocoDetectionDataset.__getitem__() + collate_function()
        │   │   └── add_instance_masks()
        │   │       ├── build_instance_mask()
        │   │       │   ├── rasterize_polygons() -> fill_polygon_part()               polygons, at the network size
        │   │       │   └── decode_run_length_encoding() (-> decode_compressed_counts()) + resize INTER_NEAREST_EXACT
        │   │       ├── objects whose mask is empty once resized removed
        │   │       └── compute_mask_bounding_boxes()                                 boxes = tight boxes of the masks
        │   ├── run_model_iteration()
        │   │   ├── prepare_batch()                                                   instance_masks moved to the device
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
        │   ├── validation batch (the 5 losses, without gradients)
        │   └── average_trackers() + log_interval_metrics() + print_interval_summary()
        └── final save_model() + run_test_evaluation(), finish_experiment_tracking()

run_test_evaluation(iteration)   [STAGE 4]
├── evaluation_context()                                     score threshold lowered to evaluation.score_threshold
├── create_evaluation_accumulator()
├── for each test batch:
│   ├── prepare_batch() + predict_batch() -> InstanceSegmenter.forward() without ground truth
│   │   ├── forward_detection_stages() -> postprocess_detections()
│   │   └── predict_detection_masks()                        mask head of the level of the proposal of every
│   │                                                        detection, sigmoid of the detected class channel
│   ├── accumulate_evaluation_batch()
│   │   ├── gather_image_detections() -> compute_mask_overlaps()     chunks of 16 detections:
│   │   │       paste_instance_masks() -> >= mask_threshold -> compute_mask_intersection_over_union()
│   │   ├── gather_image_ground_truth()                              with mask_areas
│   │   ├── accumulate_proposal_recall()
│   │   ├── compute_overlap_matrix(overlap_type="masks") + match_image_detections() + add_to_confusion_matrix()
│   │   └── dump_errors() (when activated)
│   └── save_sample_images() -> draw_test_image() -> get_sample_masks() + build_detection_sample_grid()
├── compute_evaluation_results()
│   ├── summarise_average_precisions(overlap_type="masks")   mask mAP, 3 methods, sizes by mask area
│   ├── summarise_average_precisions(overlap_type="boxes")   box mAP, 3 methods, sizes by box area
│   ├── save_proposal_recall()                               proposal_recall.csv
│   ├── write_class_metrics_file()                           box and mask columns of every class
│   └── save_error_analysis()                                thresholds of the mask APs, confusion matrix
└── report_evaluation() -> report_evaluation_results() + track_evaluation()      headline 'Mask mAP@0.5/area'

scripts/evaluation/evaluate_checkpoint.py   [STAGE 5]
└── evaluate_checkpoints() -> run_test_evaluation() per checkpoint -> summarise_checkpoint_evaluations()
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below are real outputs from running the loader on `configurations/instance_segmentation/configuration_synthetic_shapes.yaml`, its data folders and mapping file redirected to the generated dataset.

### Step 1.1 — `main()`
The entry point is the same for every task (see `classification_workflow.md`, Step 1.1).

### Step 1.2 — `load_experiment_configuration()`
The model sections are those of the object detection (see `object_detection_workflow.md`, Step 1.2) plus `MaskHead`, the settings shared by the mask heads of every level: `mask_resolution` (the side of the predicted masks, 28), `upsampling` (`transposed` or `bilinear`, how the pooled features are doubled up to the mask resolution), `normalization`, `modules`, `inference.mask_threshold` (the probability from which a pixel of a predicted mask belongs to its object) and `pretrained`. Every level of `DetectionLevels` holds, next to its anchors, region proposal and detector settings, a `mask_head` with its own `roi_align_pool_size` and `convolutions` (`[number of layers, number of filters]`). The `localisation_loss_beta` fractions of `RegionProposal` and `Detector` are converted to floats.

**Example: the sections and the mask settings**
```text
experiment keys: task, experiment_folder, train_data_folder, validation_data_folder, test_data_folder, batch_size, information_dump,
                 weight_saving_iterations, intermediate_weight_saving_iterations, number_iterations, validation_iteration_ratio,
                 evaluation, tracking, learning_rate, resume_training, diagnosis_modes, tracking_root
model keys:      Optimization, Data, Backbone, DetectionLevels, RegionProposal, Detector, MaskHead
evaluation:      {'per_class_report': True, 'number_sample_images': 8, 'score_threshold': 0.05, 'target_precision': 0.99,
                  'error_dumps': {'activated': False, 'maximum_per_class_pair': 20}}
Data keys:       loading, image_settings, label_mapping, data_augmentation, label_mode, label_dictionary
number_iterations: 10000000000
DetectionLevels 5: {"anchors": {"scales": [32, 64, 128], "aspect_ratios": [0.5, 1.0, 2.0]}, "region_proposal": {"convolutions": [2, 64]},
                    "detector": {"roi_align_pool_size": 7, "convolutions": [2, 64], "fully_connected": [256, 128]},
                    "mask_head": {"roi_align_pool_size": 14, "convolutions": [4, 64]}}
MaskHead: {"mask_resolution": 28, "upsampling": "transposed",
           "normalization": {"feature_map_normalization": "batch", "batch_normalization_momentum": 0.1},
           "modules": {"residual_nets": true, "inception": false, "squeeze_and_excite": false},
           "inference": {"mask_threshold": 0.5},
           "pretrained": {"experiment_folder": null, "iteration": null, "freeze": false, "learning_rate_multiplier": 1.0}}
RPN beta: 0.1111111111111111   Detector beta: 0.1111111111111111
```

### Step 1.3 — `load_configuration()`
**Example output**
```text
top-level keys:             ['ExperimentConfiguration', 'Optimization', 'Data', 'Backbone', 'DetectionLevels', 'RegionProposal', 'Detector', 'MaskHead']
number_iterations:          '1e10'      <- still a string
learning_rate:              0.0002
localisation_loss_beta raw: '1/9'       <- still a string
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
The common checks run first, then `check_instance_segmentation_configuration()`: the checks of the detection (`check_detection_configuration()`, which also requires a `mask_head` in every detection level) and `check_mask_head_configuration()`. The RoIAlign pool size of every level must not exceed `MaskHead.mask_resolution`, since the mask head only enlarges the pooled features.

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
The instance segmentation trainer is the detection trainer with three class attributes and one setting:
- `LOAD_INSTANCE_MASKS = True`: the COCO loaders return the mask of every object (Step 2.9).
- `MATCHING_OVERLAP_TYPE = "masks"`: the detections are matched to the objects by the IoU of their masks, for the confusion matrix and the error images (Step 4.5).
- `MASK_OVERLAP_CHUNK_SIZE = 16`: the detections whose masks are pasted at once when the mask overlaps are computed (Step 4.5).
- `self.mask_threshold`, read from `MaskHead.inference.mask_threshold`: the probability from which a pixel of a pasted mask belongs to its object, in the evaluation and the sample images.

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

### Step 2.2 — `BaseTrainer.__init__()`
**Example: some attributes set by the constructor**
```text
self.device = cpu   self.per_class_report = True   self.number_sample_images = 8   self.input_image_size = 512
self.batch_size = 40   self.mixed_precision = False   self.start_iteration = 1
```

### Step 2.3 — `get_device()`
**Example output** (forced on the CPU; a normal run on this machine prints `cuda`)
```text
For Training We Will Be Using device: cpu
```

### Step 2.4 — `get_run_name()` → `build_run_name()` → `build_instance_segmentation_sections()`
The prefix is `INS`, and the sections are those of the detection (see `object_detection_workflow.md`, Step 2.4) with a mask heads section before the classes: `MRES` (residual shortcuts; `MSE` and `MINC` for the other modules), `MBN` (batch normalization; `MLN`, `MNN`), `MUTC` (transposed convolutions; `MUBL` for bilinear), `M14.4x64` (RoIAlign of 14, 4 layers of 64 filters, written once when every level has the same mask head, otherwise per level joined with `+`) and `MR28` (mask resolution). The mask threshold gives no token, so that it can change between two resumptions of a run.

**Example output**
```text
INS_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128_MRES-MBN-MUTC-M14.4x64-MR28_MC20
```

### Step 2.5 — `DataAugmenter()`
The same augmenter as the detection; with instance masks, the geometric augmentations move the masks with the image (Step 3.8). This configuration turns every type off.

### Step 2.6 — `setup_training_paths()`
**Example: the run folder at the end of the construction**
```text
INS_IN-KAR-512_..._MRES-MBN-MUTC-M14.4x64-MR28_MC20/
├── Tensorboard/
├── Weights/
├── anchor_recall.csv              263 bytes   (Step 2.21)
├── data.json                    5,487 bytes   (Step 2.11)
└── training_configuration.yaml 33,125 bytes
```

### Step 2.7 — `build_dataloaders()`
The hook of the detection (see `object_detection_workflow.md`, Step 2.7): `class_names`, `class_colours` and the test image names from the `labels.json` of the test split, then the loaders.

**Example output**
```text
class_names: {0: 'background', 1: 'square', 2: 'circle', 3: 'triangle', 4: 'rectangle', 5: 'pentagon', 6: 'hexagon', 7: 'plus_sign',
              8: 'star', 9: 'octagon', 10: 'rhombus', 11: 'parallelogram', 12: 'trapezoid', 13: 'kite', 14: 'heptagon', 15: 'nonagon',
              16: 'ellipse', 17: 'semicircle', 18: 'annulus', 19: 'capsule', 20: 'l_shape'}
test_image_file_names: 20 entries, [(1, 'test_000000.png'), (2, 'test_000001.png'), (3, 'test_000002.png'), ...]
```

### Step 2.8 — `get_detection_dataloaders(load_instance_masks=True)`
**Example: the three loaders**
```text
train_dataloader        100 images   3 batches   collate_function   num_workers 8   prefetch 2   pin True   persistent True
validation_dataloader    20 images   1 batch     collate_function   num_workers 8   prefetch 2   pin True   persistent True
test_dataloader          20 images   1 batch     collate_function   num_workers 8   prefetch 2   pin True   persistent True
```

### Step 2.9 — `CocoDetectionDataset(load_instance_masks=True)` → `select_image_objects(boxes_from_segmentation=True)`
The manifests are COCO `labels.json` files, every annotation holding a `segmentation`: a list of polygons or a run length encoding (`size` [height, width] and `counts`, the lengths of the alternating background and object runs, column by column, as a list when uncompressed or a string when compressed). With the instance masks, the box of every object is not read from its `bbox` but from its segmentation (`compute_segmentation_bounding_box()`: the extent of the polygon vertices, or of the object runs of the RLE, x2 and y2 one past the last pixel), so that the boxes and the masks agree and an annotation needs no `bbox`. Only the boxes are read here, without decoding a mask.

`__getitem__()` then calls `add_instance_masks()`: `build_instance_mask()` builds the mask of every object at the network resolution (polygons rasterised in the network input by `rasterize_polygons()`; an RLE decoded at the original resolution by `decode_run_length_encoding()`, then resized like the image with `cv2.INTER_NEAREST_EXACT` and padded with 0), the objects whose mask is empty once resized are removed with their box and label, and `compute_mask_bounding_boxes()` replaces every box by the tight box of its resized mask.

**Example: the manifest of the training split**
```text
images[0]:          {'id': 1, 'file_name': 'train_000000.png', 'width': 512, 'height': 512}   (100 images)
annotation keys:    ['id', 'image_id', 'category_id', 'bbox', 'area', 'iscrowd', 'segmentation']
annotations[0]:     {'id': 1, 'image_id': 1, 'category_id': 6, 'bbox': [325, 369, 133, 123], 'area': 11681, 'iscrowd': 0,
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
The images are already 512 pixels, so the masks are not resized and their boxes are the boxes of the segmentations.

### Step 2.10 — `build_native_loader()`
The loader of every task (see `classification_workflow.md`, Step 2.10). `collate_function()` stacks the images and keeps the boxes, labels and instance masks as lists, one tensor per image, the number of objects changing from one image to the next (Step 3.6).

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

### Step 2.13 — `MaskHead.__init__()` → `count_mask_upsamplings()`
A mask head is a small fully convolutional network applied to every box:
- `torchvision.ops.roi_align()` pools the box from the feature map of its level into a `roi_align_pool_size` square (`spatial_scale` = feature map size / image size, `sampling_ratio=0`, `aligned=True`).
- `convolutional_block`: the `convolutions` layers of 3x3 convolutions, with the normalization and modules of `MaskHead`.
- `count_mask_upsamplings()` gives the number of doublings that bring the pool size to the mask resolution without exceeding it (here 14 -> 28: 1). Each doubling is a `ConvTranspose2d` of kernel 2 and stride 2 (`transposed`) or a bilinear upsampling followed by a 3x3 convolution (`bilinear`), followed by a ReLU.
- `mask_predictor`: a 1x1 convolution giving one mask logit map per class, background included; the mask of a box is read in the channel of its class (Steps 3.13 and 4.3).
- When the doublings do not reach `mask_resolution` exactly (for instance a pool of 10 doubled to 20 for a resolution of 28), the logits are resized bilinearly to it.

The masks of every level have the same side, so that the masks of all the levels can be concatenated for the loss.

**Example: the mask head of block 5**
```text
MaskHead(
  (convolutional_block): ConvolutionBlock(
    (squeeze_and_excitation_layers): ModuleList()
    (block): ModuleList(
      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
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
Four components can be initialised by another run: `Backbone`, `RegionProposal`, `Detector` and `MaskHead` (each section with its `pretrained` settings). `get_tracked_metrics()` adds `mask_loss` to the five metrics of the detection.

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
- Run Name                : INS_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128_MRES-MBN-MUTC-M14.4x64-MR28_MC20
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
- Gradient Clipping Norm  : None
- Weight Moving Average   : Off
- Mixed Precision         : False
- Resume Training         : True
- Validation Iter. Ratio  : 25
- MLflow Tracking         : Off
```

### Step 2.16 — `print_summary()` / `print_receptive_field_summary()` / `print_model_size_summary()`
The panels of the detection (backbone, region proposal, detection, receptive field; see `object_detection_workflow.md`, Step 2.19) are followed by a **Mask Architecture** panel, one block per level, and the model size panel counts the mask heads apart.

**Example output: the region proposal panel, the mask panel and the model size** (the backbone, detection and receptive field panels are those of the detection)
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
With `diagnosis_modes.view_input_data` on, the input viewer of the detection (see `object_detection_workflow.md`, Step 2.20) draws the boxes over the masks of the objects, every mask filled with the colour of its class by `overlay_instance_masks()`. The configuration turns it off.

### Step 2.18 — `analyse_training_data()` → anchor recall
The anchor recall of the detection (see `object_detection_workflow.md`, Step 2.21), on the boxes of the segmentations (`gather_training_object_boxes(boxes_from_segmentation=True)`), the boxes the detector learns. It reads the manifest only, so the 358 objects are measured without decoding any mask.

**Example output**
```text
---------------------------------------------
Anchor Recall Of The 358 Training Objects...
---------------------------------------------
Anchor Recall Of The Training Objects
 Objects               Count   Block 5   Block 6   Block 7   All Levels
────────────────────────────────────────────────────────────────────────
 All Objects             358    0.8855    0.5307    0.4441       0.8855
 Small (< 32²)             2    0.0000    0.0000    0.0000       0.0000
 Medium (32² to 96²)     164    0.7622    0.2561    0.0671       0.7622
 Large (>= 96²)          192    1.0000    0.7708    0.7708       1.0000
Share of the objects whose best anchor overlaps them by at least 0.5 (RegionProposal.foreground_iou_threshold.min),
sizes by box area in network input pixels. The other objects only get their single best anchor as foreground. — : no
object of the size.
```
The table is saved in `anchor_recall.csv` (263 bytes) and sent to MLflow as an analysis file.

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

> The examples below are real outputs from a run on CPU of the redirected `configuration_synthetic_shapes.yaml` shortened to batch size 2, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4 iterations, a validation batch every 10 iterations, 2 loading workers, with the MLflow tracking on. Its run name is the one of Step 2.4, written `<run>` below; with batches of 2 images, an epoch is 50 iterations. The 12 iterations, with the two test evaluations, took 50.2 s.

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
The loop of every task (see `classification_workflow.md`, Step 3.1).

**Example: what happens at each iteration of the run**

| Iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | after the loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Checkpoint saved | | | ✔ | | | ✔ | | | ✔ | | | | ✔ (iteration 12) |
| Test evaluation | | | | | | ✔ | | | | | | | ✔ (iteration 12) |
| Validation batch | | | | | | | | | | ✔ | | | |
| Console summary | | | | ✔ | | | | ✔ | | | | ✔ | |

The checkpoint and the evaluation of iteration 6 run at the start of iteration 6, before its batch.

### Step 3.2 — `setup_tensorboard_writers()` / `initialize_training_loop_state()`
**Example: the curves after the run**
```text
Tensorboard/Train/        Total Loss, RPN Classification Loss, RPN Regression Loss, Detection Classification Loss,
                          Detection Regression Loss, Mask Loss, Learning Rate                        (iterations 1 to 12)
Tensorboard/Validation/   Total Loss, RPN Classification Loss, RPN Regression Loss, Detection Classification Loss,
                          Detection Regression Loss, Mask Loss                                       (iteration 10)
Tensorboard/Test/         76 scalars and 9 images per evaluation (Step 4.13)                         (iterations 6 and 12)
```

### Step 3.3 — `start_experiment_tracking()`
**Example output**
```text
- MLflow Tracking         : sqlite:///.../runs/mlflow.db                    <- launch information
MLflow Run Created In The Experiment 'instance_segmentation' : f5b453fd71c7496db3e7403d65d9e89e (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```
The identifiers are also written in `<run>/mlflow_run.json` (`tracking_uri`, `experiment_id` "1", `run_id`), so that a resumed run or a standalone evaluation (Stage 5) logs to the same MLflow run.

### Step 3.4 — `save_model()` → `save_checkpoint()`
The checkpoint holds the source run of every transferable component of the task, here four: `pretrained_backbone`, `pretrained_region_proposer_dictionary`, `pretrained_detector_heads_dictionary` and `pretrained_mask_heads_dictionary` (None without transfer).

**Example: the checkpoints and the keys of the last one**
```text
Checkpoint Folder :
 - Iteration_12
 - Model Saved In : model_000012.pt

Weights/Iteration_3/model_000003.pt   74,655,276 bytes      Weights/Iteration_6/model_000006.pt   74,655,276 bytes
Weights/Iteration_9/model_000009.pt   74,655,276 bytes      Weights/Iteration_12/model_000012.pt  74,655,276 bytes
Weights/full-checkpoint:  model_checkpoint_path: "Iteration_12", all_model_checkpoint_paths: "Iteration_3" ... "Iteration_12"

iteration 12   format_version 2   task 'instance_segmentation'   run_name '<run>'   mlflow_run_id 'f5b453fd71c7496db3e7403d65d9e89e'
pretrained_backbone None   pretrained_region_proposer_dictionary None   pretrained_detector_heads_dictionary None
pretrained_mask_heads_dictionary None
model_state: 459 tensors, 96 of them in the mask heads
  mask_heads_dictionary.5.upsampling_layers.0.weight   (64, 64, 2, 2)
  mask_heads_dictionary.5.mask_predictor.weight        (21, 64, 1, 1)
optimizer_state: {'state', 'param_groups'}
```

### Step 3.5 — `run_test_evaluation()`
At iterations 6 and 12 (the final checkpoint), see Stage 4.

### Step 3.6 — `get_next_batch()` → `CocoDetectionDataset.__getitem__()` + `collate_function()`
Every sample goes through `add_instance_masks()` (Step 2.9). The images are stacked; the boxes, labels and instance masks stay lists, one tensor per image.

**Example: the first batch of the run**
```text
images               (2, 3, 512, 512) torch.uint8
bounding_boxes       list of 2: (4, 4) float32, (3, 4) float32
labels               list of 2: (4,) int64, (3,) int64       [14, 16, 6, 16]   [20, 13, 17]
sample_identifiers   (2,) torch.int64
instance_masks       list of 2: (4, 512, 512) uint8, (3, 512, 512) uint8
mask pixels          [9155, 2352, 5597, 4341]   [7163, 6094, 2600]
```

### Step 3.7 — `run_model_iteration()` → `prepare_batch()`
The detection preparation (images cast to float32 and moved to the device, boxes and labels moved), plus the instance masks, moved to the device and kept in uint8.

**Example: after the preparation**
```text
images (2, 3, 512, 512) torch.uint8 -> torch.float32
instance_masks list of 2: (4, 512, 512) uint8, (3, 512, 512) uint8      <- unchanged
```

### Step 3.8 — `augment_batch()` → `DataAugmenter.augment_batch(bounding_boxes=..., labels=..., instance_masks=...)`
The geometric augmentations move the masks with the image, as one more channel stack, with the nearest interpolation. The box of every object is then replaced by the tight box of its moved mask (`compute_mask_bounding_boxes()`): a rotated box would otherwise become the larger box enclosing its corners. The share of every object left inside the image is measured on its mask pixels (moved mask pixels over original mask pixels) instead of the box areas, and `clean_bounding_boxes()` removes, with their labels and masks, the objects left narrower or shorter than 1 pixel or with less than 25 % of their pixels (`MINIMUM_VISIBLE_AREA_FRACTION`). The photometric augmentations leave the masks unchanged. This configuration turns every type off, so nothing changes.

**Example: the augmentation report of the first batch**
```text
[{'augmented': False, 'geometric_augmentations': [], 'photometric_augmentations': [], 'removed_labels': None},
 {'augmented': False, 'geometric_augmentations': [], 'photometric_augmentations': [], 'removed_labels': None}]
instance_masks: (4, 512, 512) uint8, (3, 512, 512) uint8
```

### Step 3.9 — `forward_pass()` → `InstanceSegmenter.forward(ground_truth_bounding_boxes=..., ground_truth_labels=..., ground_truth_masks=...)`
`forward()` runs `forward_detection_stages()`, the whole two-stage detector (Step 3.10), then, with the ground truth, `compute_proposal_mask_loss()` (Steps 3.11 to 3.13), and in evaluation mode `predict_detection_masks()` (Step 4.3). The ground truth boxes cannot be given without their masks: `ValueError: The instance segmenter needs the ground truth masks with the ground truth boxes.` The trainer sums every output key ending in `_loss` into the training loss.

**Example: the outputs of the first iteration**
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
The forward pass of the detector (`two_stage_detector.py`), unchanged: backbone, region proposals of every level, aggregation with the anchors, proposal filter, target assignment and sampling of 128 proposals per image, detection heads, and the four detection losses (see `object_detection_workflow.md`, Steps 3.10 to 3.18). It also returns what the mask branch needs: the feature maps of the detection blocks and the sampled proposals with their labels, levels and matched ground truths.

### Step 3.11 — `compute_proposal_mask_loss()`
As in Mask R-CNN, the masks are learned on the positive sampled proposals only (label > 0), the proposals the detector learns to classify as an object. Each level runs its mask head on its own positive proposals, read with `sliced_origins` (the level of every proposal), even when it has none, so that every mask head stays in the graph. The logits of all the levels, their labels, proposals and matched ground truth indices are concatenated, `build_mask_targets()` crops the targets (Step 3.12) and `compute_mask_loss()` compares them (Step 3.13).

**Example: the first iteration**
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

**Example: the 12 positive proposals of the first iteration**
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

**Example: the first iteration**
```text
mask_logits (12, 21, 28, 28) float32   labels [14, 16, 6, 16, 20, 13, 17, 14, 13, 20, 6, 20]
class_logits (12, 28, 28)              <- the channel of the label of every proposal
mask_loss 0.7130 torch.float32         <- close to ln 2 = 0.693, the loss of a logit of 0
```

### Step 3.14 — `compute_batch_metrics()` + `MetricTracker.accumulate()` + `log_to_tensorboard()`
The detection has no batch metric besides its losses: the 6 tracked values (5 losses and the total) are accumulated and written to `Tensorboard/Train/` at every iteration.

**Example: the losses of the 12 iterations**
```text
iteration   total    RPN class.   RPN regr.   det. class.   det. regr.   mask
    1       5.5602   0.6610       0.7074      2.9878        0.4909       0.7130
    2       5.5642   0.6596       0.7044      2.8753        0.6288       0.6961
    3       5.4658   0.6497       0.7850      2.7839        0.5599       0.6874
    4       5.0714   0.6428       0.5483      2.6449        0.5565       0.6788
    5       5.1298   0.6396       0.6219      2.5819        0.5759       0.7104
    6       5.2816   0.6369       0.6342      2.5206        0.7893       0.7007
    7       4.8839   0.6426       0.5293      2.3601        0.6581       0.6938
    8       4.8474   0.6237       0.5956      2.2970        0.6230       0.7080
    9       4.6502   0.6376       0.5380      2.2238        0.5453       0.7056
   10       4.3778   0.6076       0.4692      2.1503        0.4617       0.6890
   11       4.6119   0.6074       0.5187      2.0940        0.6963       0.6955
   12       4.1492   0.6243       0.5437      1.7946        0.4805       0.7062
```
In 12 iterations the detection classification loss falls from 2.99 to 1.79 while the mask loss stays around ln 2: the mask heads only see about 12 positive proposals per batch.

### Step 3.15 — `training_loss.backward()` + `step_optimizer()`
**Example: the gradient norm of every component**
```text
iteration 1:  backbone 26.0453   region_proposer_dictionary 6.8338   detector_heads_dictionary 12.6682   mask_heads_dictionary 0.3247
iteration 12: backbone 15.9901   region_proposer_dictionary 4.3910   detector_heads_dictionary 8.1084    mask_heads_dictionary 0.4148
```

### Step 3.16 — validation pass
At iteration 10, one validation batch in `eval()` mode, without gradients, with the ground truth boxes, labels and masks: the model computes the same five losses (the mask loss on the positive sampled proposals), and in `eval()` mode also predicts the masks of its detections, which the validation does not use.

**Example: the validation batch of iteration 10**
```text
total 4.8811   RPN classification 0.6847   RPN regression 0.4954   detection classification 2.8054   detection regression 0.2007
mask 0.6948
```

### Step 3.17 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()`
**Example output: the summaries of iterations 4 and 12**
```text
========================================================================================================================
|     Iteration: 4     |      Total Images: 8      |     Interval Images: 8     |  Throughput: 0.3 Images Per Second   |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   5.4154
Moving Average of Train RPN Classification Loss        :   0.6533
Moving Average of Train RPN Regression Loss            :   0.6863
Moving Average of Train Detection Classification Loss  :   2.8230
Moving Average of Train Detection Regression Loss      :   0.5590
Moving Average of Train Mask Loss                      :   0.6938
------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:25.
Validation Iterations In This Interval: 0 (Ratio: Every 10 Training Iterations).
========================================================================================================================
...
========================================================================================================================
|    Iteration: 12     |      Total Images: 24     |     Interval Images: 8     |  Throughput: 1.8 Images Per Second   |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   4.4473  -0.5884 Best
Moving Average of Valid Total Loss                     :   4.8811
Moving Average of Train RPN Classification Loss        :   0.6192  -0.0165 Best
Moving Average of Valid RPN Classification Loss        :   0.6847
Moving Average of Train RPN Regression Loss            :   0.5174  -0.0779 Best
Moving Average of Valid RPN Regression Loss            :   0.4954
Moving Average of Train Detection Classification Loss  :   2.0657  -0.3743 Best
Moving Average of Valid Detection Classification Loss  :   2.8054
Moving Average of Train Detection Regression Loss      :   0.5459  -0.1157 Best
Moving Average of Valid Detection Regression Loss      :   0.2007
Moving Average of Train Mask Loss                      :   0.6991  -0.0041
Moving Average of Valid Mask Loss                      :   0.6948
------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:04.
Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
========================================================================================================================
```
The first interval includes the start of the loader workers and the MLflow run, the second one the evaluation of iteration 6. At iteration 12 the mask loss is lower than at iteration 4 (-0.0041) but higher than its best average, at iteration 8 (0.7032 was not a best either: +0.0094), so it has no `Best` mark.

**Example: the MLflow metrics at the end of the run**
```text
train/Mask Loss 0.6991   validation/Mask Loss 0.6948   train/interval_seconds 4.5107   train/throughput 1.7735   learning_rate
status FINISHED   185 parameters   92 metrics
```

### Step 3.18 — end of the loop
**Example output**
```text
Checkpoint Folder :
 - Iteration_12
 - Model Saved In : model_000012.pt
------------------------------------------------------------
Model successfully saved at iteration 12. Exiting Training.
------------------------------------------------------------
```
Then the evaluation of iteration 12 (Stage 4) and `finish_experiment_tracking()`.

**Example: the run folder at the end**
```text
<run>/
├── Evaluation/Iteration_6/, Evaluation/Iteration_12/    class_metrics.csv, class_thresholds.json, confusion_matrix.csv,
│                                                        confusion_matrix.png, proposal_recall.csv, Samples/
├── Tensorboard/Train/, Tensorboard/Validation/, Tensorboard/Test/
├── Weights/Iteration_3/, Iteration_6/, Iteration_9/, Iteration_12/, full-checkpoint
├── Results.txt                                          the reports of both evaluations (208 lines)
├── anchor_recall.csv, data.json, mlflow_run.json, training_configuration.yaml
```

---

## Stage 4 — Test Evaluation of a Checkpoint

> The examples below are real outputs of the evaluations of the checkpoints of iterations 6 and 12 of the run of Stage 3, on its whole test split (20 images, 66 objects), with `per_class_report: True`, `number_sample_images: 8`, `score_threshold: 0.05`, `target_precision: 0.99`, the error images off and the MLflow tracking on. The steps inside one test batch come from the checkpoint of iteration 12 restored in a copy of the run folder (`start_iteration` 13), on the test batch of the images 2 and 7 (`test_000001.png` and `test_000006.png`).

### Step 4.1 — `run_test_evaluation()` → `evaluation_context()`
The evaluation loop of every task (see `classification_workflow.md`, Step 4.1) with the `evaluation_context()` of the detection, which lowers the score threshold of the model to `evaluation.score_threshold` during the evaluation and restores the deployment one afterwards.

**Example output**
```text
score threshold in the context: 0.05      after: 0.5
Test Evaluation Iteration 6: 100%|██████████| 10/10 [00:06<00:00,  1.52it/s]
```

### Step 4.2 — `create_evaluation_accumulator()`
The accumulator of the detection (see `object_detection_workflow.md`, Step 4.2); the detections and ground truths of every image also keep their mask overlaps and areas (Step 4.5).

**Example output**
```text
{'detections': list, 'ground_truths': list, 'confusion_matrix': (21, 21), 'errors_folder': PosixPath, 'error_counts': dict,
 'proposal_recall': dict}
```

### Step 4.3 — `prepare_batch()` + `predict_batch()` → `predict_detection_masks()`
`predict_batch()` runs the model without the ground truth, as in deployment. After the post-processing of the detections (see `object_detection_workflow.md`, Step 4.4), `predict_detection_masks()` runs the mask heads on the **final boxes**, refined by the detection head, as in Mask R-CNN where the masks follow the refined boxes: the level of every detection is the level of its proposal (`sliced_origins` read at `final_detection_proposal_indices`), every level runs its mask head on its own valid detections, and the sigmoid of the channel of the detected class is kept. The padding detections get a mask of 0. The masks are 28 x 28 in the frame of their box; they are pasted into the image only for the evaluation (Step 4.5) and the sample images (Step 4.6).

**Example: the predictions of the test batch**
```text
test batch: sample identifiers [2, 7], instance_masks (2, 512, 512) and (1, 512, 512), labels [[19, 5], [16]]
classification_scores              (2, 250, 21) float32
box_regressions                    (2, 250, 21, 4) float32
detection_boxes                    (2, 250, 21, 4) float32
sliced_origins                     (2, 250) int64
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
After 12 iterations the mask probabilities stay around 0.5, so a pasted mask is a rough blob around the middle of its box, and every detection scores just above the evaluation threshold.

### Step 4.4 — `postprocess_detections()`
The post-processing of the detection, unchanged (see `object_detection_workflow.md`, Step 4.4): 100 detections per image at the evaluation threshold 0.05, none at the deployment threshold 0.5.

### Step 4.5 — `accumulate_evaluation_batch()` → `gather_image_detections()` → `compute_mask_overlaps()`
For every image, `gather_image_detections()` keeps the valid detections (boxes, scores, labels) with two more entries:
- `mask_overlaps` [D, G] float64: the mask IoU of every detection with every ground truth object of the image, whatever their classes, computed by `compute_mask_overlaps()` in chunks of 16 detections (`MASK_OVERLAP_CHUNK_SIZE`), so that a 512 x 512 float mask is held for 16 detections at a time:
  - `paste_instance_masks()` pastes the 28 x 28 masks into the image at their boxes, by bilinear resampling written as two products with separable tent weights (`build_interpolation_weights()`: one [image size, 28] weight matrix for the rows and one for the columns of every box), so that the pasting runs in one batched matrix product on the device;
  - the pasted masks are thresholded at `mask_threshold` (`>= 0.5`);
  - `compute_mask_intersection_over_union()` gives the IoU of every binary detection mask with every ground truth mask, the intersections and unions summed in float32 and divided in float64 (in float32, 13 / 20 falls below 0.65, which would move a match across an IoU threshold).
- `mask_areas` [D] int64: the pixels of every binary detection mask, for the mAP by object size.

`gather_image_ground_truth()` adds the `mask_areas` of the ground truth objects. The detections reaching the deployment threshold are then matched to the objects for the confusion matrix with `compute_overlap_matrix(overlap_type="masks")`, which returns the mask overlaps instead of the box IoUs (`MATCHING_OVERLAP_TYPE`), and `match_image_detections()` at IoU 0.5 (see `object_detection_workflow.md`, Step 4.5). `accumulate_proposal_recall()` keeps measuring the proposals on the boxes.

**Example: the first image of the test batch** (2 objects, labels capsule (19) and pentagon (5))
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

**Example: `Evaluation/Iteration_12/Samples/`**
```text
test_000000.png, test_000003.png, test_000005.png, test_000008.png, test_000011.png, test_000014.png, test_000016.png, test_000019.png
every file 1028 x 560 pixels RGB
```
No detection reaches 0.5 at iterations 6 and 12, so the prediction panels are empty.

### Step 4.7 — `compute_evaluation_results()`
The evaluation of the detection computed twice on the same detections:
- `summarise_average_precisions(overlap_type="masks", title_prefix="Mask", name_prefix="Mask ", column_prefix="mask_")`: the mask mAP, the detections matched to the objects of their class by mask IoU, with the three methods and by object size, the sizes measured on the **mask areas** (`select_object_size_detections()` reads `mask_areas`).
- `summarise_average_precisions(overlap_type="boxes", title_prefix="Box")`: the box mAP of the detection, the sizes measured on the box areas, comparable with the object detection runs.

Then `save_proposal_recall()` (see `object_detection_workflow.md`, Step 4.15), `write_class_metrics_file()` with the box and mask columns of every class (Step 4.8), and `save_error_analysis()` with the score thresholds of the **mask** average precisions (Step 4.9). The report holds nine tables in this order: mask mAP by method, mask mAP by size, box mAP by method, box mAP by size, mask mean average recall, box mean average recall (see `object_detection_workflow.md`, Step 4.14), proposal recall, mask per class, box per class. With `evaluation.original_resolution`, `gather_image_detections()` restores the detection boxes to the original image and pastes the masks at that size, and `gather_image_ground_truth()` takes the boxes and mask areas of the original annotations, so that the mean average precisions and recalls, and their sizes, are those of the original images. The checkpoints are compared on `Mask mAP@0.5/area`.

**Example: the results of `gather_class_detections()` and `evaluate_detections()`**
```text
evaluate_detections keys: ['area', '11_point', '101_point', 'class_score_thresholds']
per method:               ['mAP@0.5', 'mAP@0.75', 'mAP@[0.5:0.95]', 'class_AP@0.5', 'class_AP@0.75', 'class_AP@[0.5:0.95]',
                           'class_ground_truths', 'sizes']
gather_class_detections(class 19, capsule) on the test batch, masks and boxes alike:
  {'scores': (0,), 'image_indices': (0,), 'best_intersection_over_union': (0,), 'best_ground_truth_indices': (0,), 'detection_areas': (0,),
   'best_ground_truth_areas': (0,), 'ground_truth_areas': (1,), 'number_ground_truth': 1}
```
No detection of the batch is a capsule, so the class has 1 object and no detection.

**Example: the report of iteration 6** (the box per-class table is shortened)
```text
Iteration 6 · 20 Test Images · 2026-10-16 19:33 ────────────────────────────────────────────────────────────────────────
  Confusion Matrix : score >= 0.50, IoU >= 0.5 (masks), 0 right, 0 wrong class, 0 false positives, 66 missed
  Sample Images : 8 fixed test images

  Mask Mean Average Precision
   Method      mAP@0.5   mAP@0.75   mAP@[0.5:0.95]
  ─────────────────────────────────────────────────
   Area         0.0000     0.0000           0.0000
   11 Point     0.0000     0.0000           0.0000
   101 Point    0.0000     0.0000           0.0000
  The mask mAP@0.5 of the area method compares the checkpoints of the run.

  Mask Mean Average Precision By Object Size (Area)
   Object Size           Objects   mAP@0.5   mAP@0.75   mAP@[0.5:0.95]
  ─────────────────────────────────────────────────────────────────────
   Small (< 32²)               0         —          —                —
   Medium (32² to 96²)        53    0.0000     0.0000           0.0000
   Large (>= 96²)             13    0.0000     0.0000           0.0000
  COCO sizes by mask area in network input pixels, the objects of the other sizes and the detections finding them being
  ignored. — : no object of the size.

  Box Mean Average Precision
   Method      mAP@0.5   mAP@0.75   mAP@[0.5:0.95]
  ─────────────────────────────────────────────────
   Area         0.0009     0.0000           0.0002
   11 Point     0.0009     0.0000           0.0002
   101 Point    0.0009     0.0000           0.0002
  Box average precisions of the same detections, comparable with the object detection runs.

  Box Mean Average Precision By Object Size (Area)
   Object Size           Objects   mAP@0.5   mAP@0.75   mAP@[0.5:0.95]
  ─────────────────────────────────────────────────────────────────────
   Small (< 32²)               0         —          —                —
   Medium (32² to 96²)        34    0.0333     0.0000           0.0033
   Large (>= 96²)             32    0.0007     0.0000           0.0002
  COCO sizes by box area in network input pixels, the objects of the other sizes and the detections finding them being
  ignored. — : no object of the size.

  Proposal Recall
                                                           Proposals
   Objects               Count   Recall@0.5   Recall@0.7   Per Image
  ───────────────────────────────────────────────────────────────────
   All Objects              66       0.5455       0.0909    250.0000
   Small (< 32²)             0            —            —           —
   Medium (32² to 96²)      34       0.3235       0.0000           —
   Large (>= 96²)           32       0.7812       0.1875           —
   Block 5 Proposals        66       0.1515       0.0000    205.0000
   Block 6 Proposals        66       0.0000       0.0000      0.0000
   Block 7 Proposals        66       0.3939       0.0909     45.0000
  Share of the test objects overlapped by a proposal of the region proposal network by at least the IoU, sizes by box
  area in network input pixels. A block row keeps the proposals of that block only. — : no object of the size.

  Mask Per Class Average Precision (Area)
                   Ground                                      Threshold    Recall
   Class           Truths   AP@0.5   AP@0.75   AP@[0.5:0.95]     @ P0.99   @ P0.99
  ─────────────────────────────────────────────────────────────────────────────────
   square               6   0.0000    0.0000          0.0000           —         —
   triangle             6   0.0000    0.0000          0.0000           —         —
   ...
   capsule              4   0.0000    0.0000          0.0000           —         —
   l_shape              3   0.0000    0.0000          0.0000           —         —
   circle               0        —         —               —           —         —
   rectangle            0        —         —               —           —         —
   hexagon              0        —         —               —           —         —
  Worst AP@0.5 first, ties in class order. — : no ground truth box of the class, or the precision 0.99 is never reached.

  Box Per Class Average Precision (Area)
                   Ground
   Class           Truths   AP@0.5   AP@0.75   AP@[0.5:0.95]
  ───────────────────────────────────────────────────────────
   square               6   0.0000    0.0000          0.0000
   ...
   l_shape              3   0.0000    0.0000          0.0000
   annulus              1   0.0065    0.0000          0.0013
   nonagon              4   0.0088    0.0000          0.0014
   circle               0        —         —               —
   rectangle            0        —         —               —
   hexagon              0        —         —               —
  Worst AP@0.5 first, ties in class order. — : no ground truth box of the class.
  - Results : Results.txt
  - Class Metrics : class_metrics.csv
  - Proposal Recall : proposal_recall.csv
  - Class Thresholds : class_thresholds.json
  - Confusion Matrix Table : confusion_matrix.csv
  - Confusion Matrix Image : confusion_matrix.png
  - Sample Images : Samples
```
By mask area, 53 of the 66 test objects are medium and 13 large; by box area, 34 and 32: a shape fills only part of its box. The box mAP is above 0 (a few boxes overlap an object of their class by 0.5) while the mask mAP is 0: the masks, still blobs around 0.5, overlap the objects less than their boxes (Step 4.5). Only the mask per-class table has the threshold columns, the class thresholds being computed on the mask average precisions.

**Example: the tables of iteration 12 that changed**
```text
  Box Mean Average Precision
   Area         0.0000     0.0000           0.0000
   11 Point     0.0001     0.0001           0.0000
   101 Point    0.0000     0.0000           0.0000
  Box Mean Average Precision By Object Size (Area)
   Medium (32² to 96²)        34    0.0000     0.0000           0.0000
   Large (>= 96²)             32    0.0002     0.0002           0.0001
  Proposal Recall
   All Objects              66       0.5758       0.0606    250.0000
   Medium (32² to 96²)      34       0.3235       0.0000           —
   Large (>= 96²)           32       0.8438       0.1250           —
   Block 5 Proposals        66       0.1212       0.0000    188.0000
   Block 6 Proposals        66       0.0000       0.0000      0.0000
   Block 7 Proposals        66       0.4545       0.0606     62.0000
  Box Per Class Average Precision (Area)
   l_shape              3   0.0008    0.0008          0.0005
```
The mask tables stay at 0.

### Step 4.8 — `write_class_metrics_file()`
One row per class, the box columns of the detection followed by the same columns prefixed with `mask_`, then the threshold and recall at the target precision (of the mask average precisions): 23 columns.

**Example: `Evaluation/Iteration_6/class_metrics.csv`** (first rows)
```text
class_index,class,ground_truths,average_precision_50,average_precision_75,average_precision_50_95,average_precision_50_small,
average_precision_50_95_small,average_precision_50_medium,average_precision_50_95_medium,average_precision_50_large,
average_precision_50_95_large,mask_average_precision_50,mask_average_precision_75,mask_average_precision_50_95,
mask_average_precision_50_small,mask_average_precision_50_95_small,mask_average_precision_50_medium,mask_average_precision_50_95_medium,
mask_average_precision_50_large,mask_average_precision_50_95_large,threshold_at_target_precision,recall_at_target_precision
1,square,6,0.000000,0.000000,0.000000,,,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,,,0.000000,0.000000,0.000000,0.000000,,
2,circle,0,,,,,,,,,,,,,,,,,,,,
```

**Example: `proposal_recall.csv`**
```text
group,objects,recall_50,recall_70,proposals_per_image
all,66,0.545455,0.090909,250.000000
small,0,,,
medium,34,0.323529,0.000000,
large,32,0.781250,0.187500,
block_5,66,0.151515,0.000000,205.000000
block_6,66,0.000000,0.000000,0.000000
block_7,66,0.393939,0.090909,45.000000
```

### Step 4.9 — `save_error_analysis()`
The class thresholds and the confusion matrix of the detection (see `object_detection_workflow.md`, Step 4.12), the thresholds being those of the mask average precisions and the confusion matrix counting the matches on the masks (Step 4.5).

**Example output**
```text
class_thresholds.json:  {"target_precision": 0.99, "classes": {"square": null, "circle": null, "triangle": null, ...}}   every class null
confusion_matrix.csv:   true \ predicted,background,square,circle,triangle,...
                        background,0,0,0,0,...
                        square,6,0,0,0,...          <- the 6 squares missed (background column)
                        circle,0,0,0,0,...
                        triangle,6,0,0,0,...
confusion_matrix.png:   1852 x 1815 pixels
```

### Step 4.10 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
The report of Step 4.7 is printed and appended to `<run>/Results.txt`, the scalars, class scalars and images go to `Tensorboard/Test/` and to MLflow (see `object_detection_workflow.md`, Step 4.13).

**Example: the test curves and the MLflow metrics**
```text
Tensorboard/Test/: 76 scalars and 9 images, among them
  mAP@0.5/area, mAP@0.5 By Size/medium, Mask mAP@0.5/area, Proposal Recall@0.5/block_5, Proposals Per Image/all,
  AP@0.5/<class>, Mask AP@0.5/<class>
MLflow: test/Mask mAP_at_0.5/area, test/best_Mask mAP_at_0.5/area, test_class/Mask AP_at_0.5/square, ...
```
MLflow refuses `@` in metric names, so it is written `_at_`; `best_` keeps the best value of every metric over the evaluations of the run.

---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of the logic of `scripts/evaluation/evaluate_checkpoint.py` run on the run of Stage 3 (checkpoints 3, 6, 9 and 12) with `iteration = "all"`, `batch_size = 4` and the whole test split.

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The same script for every task (see `classification_workflow.md`, Stage 5): set `run_name` to the run folder, choose the iterations, and run it. The pretrained settings are turned off (`disable_pretrained_settings()`), since every weight comes from the evaluated checkpoint.

### Step 5.2 — trainer construction
The trainer is built from the `training_configuration.yaml` of the run, its run folder being `runs/evaluations/<run>/`.

**Example: the lines of the launch information that changed**
```text
- Iterations Per Epoch    : 25
- Batch Size              : 4
- MLflow Tracking         : sqlite:///.../runs/mlflow.db                    <- the backend of the evaluated run
```

### Step 5.3 — `get_checkpoint_iterations()`
**Example output**
```text
evaluated_iterations = [12, 9, 6, 3]
```

### Step 5.4 — `evaluate_checkpoints()` → `start_evaluation_tracking()`
The evaluations are logged to the MLflow run of the training, read from `mlflow_run.json` (Step 3.3).

**Example output**
```text
-----------------------------
Evaluating : model_000012.pt
-----------------------------
Test Evaluation Iteration 12: 100%|██████████| 5/5
MLflow Run Created In The Experiment 'instance_segmentation' : 09ee1c5e422849309be41a1c6e937c5b (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```

### Step 5.5 — `summarise_checkpoint_evaluations()`
**Example output** (the four checkpoints tie at a mask mAP of 0, so the four iterations are listed in their evaluation order)
```text
-------------------------------------------------------------
Evaluation Summary Of 4 Checkpoints : evaluation_summary.csv
-------------------------------------------------------------
  - Best Checkpoint : iterations 12, 9, 6, 3 tie (Mask mAP@0.5/area 0.0000)
```

**Example: `runs/evaluations/<run>/Evaluation/evaluation_summary.csv`** (77 columns: iteration, the 13 box mAPs, the 13 mask mAPs, the 16 proposal recalls, the 17 box and 17 mask AP@0.5 of the classes with objects)
```text
iteration   mAP@0.5/area   mAP@[0.5:0.95] By Size/large   Mask mAP@0.5/area   Mask AP@0.5/l_shape
   12       0.000049       0.000143                       0.0                 0.0
    9       0.000648       0.000172                       0.0                 0.0
    6       0.000874       0.000173                       0.0                 0.0
    3       0.0            0.0                            0.0                 0.0
```
The best box mAP is at iteration 6, but the checkpoints are ranked on the mask mAP.

**Example: the evaluation folder**
```text
runs/evaluations/<run>/
├── Evaluation/evaluation_summary.csv
├── Evaluation/Iteration_3/, Iteration_6/, Iteration_9/, Iteration_12/    the files of Step 4.7
├── Tensorboard/Test/
├── Weights/
├── Results.txt
└── anchor_recall.csv, data.json, training_configuration.yaml
```

### Step 5.6 — `run_test_evaluation()` for each checkpoint
**Example output: the start of the report of the checkpoint of iteration 12** (the same values as Stage 4, the batch size changing nothing)
```text
Iteration 12 · 20 Test Images · 2026-10-16 19:36 ───────────────────────────────────────────────────────────────────────
  Confusion Matrix : score >= 0.50, IoU >= 0.5 (masks), 0 right, 0 wrong class, 0 false positives, 66 missed
  Sample Images : 8 fixed test images

  Mask Mean Average Precision
   Method      mAP@0.5   mAP@0.75   mAP@[0.5:0.95]
  ─────────────────────────────────────────────────
   Area         0.0000     0.0000           0.0000
   11 Point     0.0000     0.0000           0.0000
   101 Point    0.0000     0.0000           0.0000
  The mask mAP@0.5 of the area method compares the checkpoints of the run.
  ...
  Box Mean Average Precision
   Area         0.0000     0.0000           0.0000
   11 Point     0.0001     0.0001           0.0000
   101 Point    0.0000     0.0000           0.0000
```
