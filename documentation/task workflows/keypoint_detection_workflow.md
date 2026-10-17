# Keypoint Detection Workflow

This document follows one keypoint detection run (KPT) from `main()` to the evaluation of a checkpoint, to show which function runs when, what it receives, what it returns and who uses the result next. The **Call Tree Overview** gives the whole run on one page. Each stage then has one section, with one step per function: a short explanation followed by a **real** output. The steps shared by every task (configuration loading, checkpoints, training loop, MLflow tracking, standalone evaluation) are explained in detail in `classification_workflow.md`, and the mean average precision machinery (precision-recall curves, the three average precision methods, the object sizes, the mean average recall) in `object_detection_workflow.md`; here they keep a short explanation and the real output of the keypoint detection run. The steps specific to the keypoint detection (keypoint definitions read from the COCO categories, one crop per object, crop affines and crop augmentation, heatmap targets, the stem and the high resolution network, the heatmap decoding, the heatmap loss, the object keypoint similarity (OKS) and the keypoint accuracy, and the export of the crop model) are explained in full.

The model is a top-down keypoint detector: every object is cut out of its image by its ground truth box into a fixed size crop, the stem of the backbone and an HRNet (high resolution network) turn the crop into one heatmap per keypoint of every class, and only the heatmaps of the class of the crop are trained and decoded. The boxes and classes always come from the labels (or, in deployment, from a detector run before it); the keypoint detector never searches for objects.

The examples come from real runs on CPU, so that the GPU stayed free for two other trainings (`CUDA_VISIBLE_DEVICES=` empty for every run). The synthetic dataset of the configuration, `synthetic_shapes`, is not on this machine, so a small one was generated in a temporary folder with `create_synthetic_dataset(task="keypoint_detection")`: 300 training, 30 validation and 30 test images of 512 pixels, at most 4 objects per image, 22 shape classes, every object with the keypoints of its class (a square has 5, a circle 1, a stick figure 11). The data folders and the mapping file of the configuration were redirected to it; the configuration already uses the Native loader. The input viewer and every other window stay off.

The dataset has changed since these runs: it now holds 37 categories, the square has its 4 corners only, the plus sign its 4 tips, the capsule its 2 ends and the 4 corners of its straight sides, the hollow polygons and the hollow capsule add the `inner_<name>` corners of their hole after their outer keypoints, and `circle`, `ellipse`, `annulus` and `hollow_ellipse` are distractors without keypoints, left out of the `label_mapping.json` written by `build_keypoint_label_mapping()`, which numbers the 33 other categories from 1 (243 heatmap channels). The outputs below keep the 22 categories of the runs.
- **Stages 1 and 2**: `configurations/keypoint_detection/configuration_synthetic_shapes.yaml` (192x192 crops cut from 512 pixel images, a 3-block stem ending at 1/4 of the crop, an HRNet of base width 16 with 1, 2 and 2 modules in its stages 2, 3 and 4, a heatmap sigma of 2 cells), with the run folder redirected to a temporary folder.
- **Stages 3, 4 and 5**: the same configuration shortened to batch size 16 (it is 64), 60 iterations (20000), a checkpoint and a test evaluation every 30 iterations (2000), an intermediate checkpoint every 15 (1000), a console summary every 10 (500), a validation batch every 10 training iterations (0, no validation), a warm-up of 10 iterations and a decay of 50 (500 and 19500), 2 loading workers (8), with the MLflow tracking turned on (off). The model, the crops and the augmentations are those of Stages 1 and 2. After 60 iterations of batches of 16 crops the detector is far from trained, so the metrics of the examples are low: the point is to see the shapes, the files and the order of the steps.
- **Stage 6**: the checkpoint of iteration 60 exported to TorchScript and ONNX on CPU by a script repeating the steps of `export_experiment()`, which only runs on CUDA (see Step 6.1).

## Table of Contents
- [Call Tree Overview](#call-tree-overview)
- [Stage 1 — Entry Point and Configuration Loading](#stage-1--entry-point-and-configuration-loading)
- [Stage 2 — Trainer Initialization](#stage-2--trainer-initialization)
- [Stage 3 — Training Loop](#stage-3--training-loop)
- [Stage 4 — Test Evaluation of a Checkpoint](#stage-4--test-evaluation-of-a-checkpoint)
- [Stage 5 — Standalone Checkpoint Evaluation (Optional)](#stage-5--standalone-checkpoint-evaluation-optional)
- [Stage 6 — Export (Optional)](#stage-6--export-optional)

---

## Call Tree Overview

```text
main()
│   Entry point of every task. Give a keypoint_detection configuration with --configuration_path to run this workflow.
│
├── load_experiment_configuration(configuration_path)
│   │   Splits the YAML into experiment_configuration ('ExperimentConfiguration') and model_configuration (Optimization,
│   │   Data, Backbone, HRNet, KeypointHead), adds label_mode, label_dictionary and keypoint_categories.
│   │
│   ├── load_configuration(configuration_path)
│   ├── build_label_dictionary(label_mapping_configuration, task)       {category id: class index} + 'background': 0
│   ├── inject_keypoint_categories(experiment_configuration, model_configuration)
│   │   └── read_keypoint_categories(labels_file, label_dictionary, flip_pairs_override)
│   │           keypoint names, skeleton, rotation limit and flip pairs of every class, from the training labels.json
│   └── check_configuration()
│       ├── common checks
│       └── check_keypoint_configuration()
│           ├── check_image_task_sequence_settings() / check_keypoint_data_configuration()
│           ├── check_crop_configuration()                         crop sides multiples of the lowest branch stride
│           ├── check_crop_augmentation_configuration()            no random_rotation of the DataAugmenter
│           ├── check_keypoint_stem_configuration()                the stem must end at 1/4 of the crop
│           ├── check_high_resolution_network_configuration()      HRNet section, positive heatmap_sigma
│           └── check_keypoint_evaluation_configuration()          default_keypoint_sigma, keypoint_sigmas
│
├── get_trainer_class(task)  -> KeypointTrainer
│
├── KeypointTrainer(experiment_configuration, model_configuration, configuration_path)   [STAGE 2]
│   │   Reads the crop settings, the crop augmentation and the heatmap sigma.
│   ├── build_keypoint_tables()                                    flip_permutations, maximum_rotations, keypoint_counts
│   ├── build_keypoint_sigma_table()                               OKS sigma of every keypoint of every class
│   └── BaseTrainer.__init__()
│       ├── get_device() / get_run_name() -> build_run_name() -> build_keypoint_sections()
│       ├── DataAugmenter() / setup_training_paths()
│       ├── build_dataloaders()                                    class_names from the test labels.json
│       │   └── get_keypoint_dataloaders()
│       │       └── KeypointDataset() per split + build_native_loader()     one sample per labelled object
│       ├── save_data_record()                                     data.json
│       ├── get_test_sample_classes() + choose_sample_identifiers()       22 fixed test objects
│       ├── build_model()
│       │   └── KeypointDetector.__init__()
│       │       ├── build_keypoint_channel_table()                 107 heatmap channels, channel_table [23, 11]
│       │       ├── build_backbone(input_image_size=crop height)   the stem, 192 -> 48
│       │       ├── HighResolutionNetwork()                        4 branches, 1/4 to 1/32, back to 1/4
│       │       └── keypoint_head (1x1 convolution, 16 -> 107)
│       ├── initialise_pretrained_components()                     Backbone, HRNet, KeypointHead
│       ├── build_optimizer() / get_tracked_metrics()              Total Loss, Keypoint Heatmap Loss
│       ├── print_experiment_information() (-> describe_split()) / print_summary()
│       │                                                          backbone, HRNet, receptive field, model size panels
│       └── restore_last_model()
│
└── trainer.run_training_loop()   [STAGE 3]
    └── run_training_iterations()
        ├── for each training_iteration:
        │   ├── save_model() + run_test_evaluation() when due                         -> [STAGE 4]
        │   ├── get_next_batch() -> KeypointDataset.__getitem__()                     one object and its whole image
        │   ├── run_model_iteration()
        │   │   ├── prepare_batch()                                                   to the device, identity crops
        │   │   │   └── cut_object_crops(augmentations=create_identity_augmentations())
        │   │   │       ├── compute_crop_regions()                                    box -> square region x 1.25
        │   │   │       ├── build_crop_affines()                                      image_to_crop, crop_to_image
        │   │   │       ├── warp_crops()                                              grid_sample of the region
        │   │   │       ├── transform_points() + swap_flipped_keypoints()             keypoints in crop pixels
        │   │   │       └── build_heatmap_targets()                                   Gaussians on 48x48, weights
        │   │   ├── augment_batch()
        │   │   │   ├── draw_crop_augmentations()                                     scale, rotation, flip, shift
        │   │   │   ├── cut_object_crops(augmentations=...)                           crops cut again
        │   │   │   └── DataAugmenter.augment_batch(images=crops)                     photometric only
        │   │   ├── forward_pass() -> KeypointDetector.forward(labels, target_heatmaps, heatmap_weights)
        │   │   │   ├── backbone -> high_resolution_network -> keypoint_head          [B, 107, 48, 48]
        │   │   │   ├── decode_heatmaps()                                             argmax + quarter-cell shift
        │   │   │   ├── channel_table[labels]                                         the channels of the class
        │   │   │   └── compute_keypoint_heatmap_loss()                               weighted mean squared error
        │   │   └── MetricTracker.accumulate() + log_to_tensorboard()
        │   ├── training_loss.backward() / step_optimizer()                           + moving average of the weights
        │   ├── validation batch (identity crops, the heatmap loss without gradients)
        │   └── average_trackers() + log_interval_metrics() + print_interval_summary()
        └── final save_model() + run_test_evaluation(), finish_experiment_tracking()

run_test_evaluation(iteration)   [STAGE 4]
├── evaluate_test_split()                                     trained weights, then use_averaged_weights() for EMA
│   ├── create_evaluation_accumulator()                       lists of EVALUATED_OBJECT_KEYS
│   ├── for each test batch:
│   │   ├── prepare_batch() + predict_batch()                 keypoints of the class, back to image pixels,
│   │   │                                                     object score = mean keypoint score
│   │   ├── accumulate_evaluation_batch()
│   │   └── save_sample_images() -> build_sample_image()
│   │           measure_keypoint_accuracy() + build_keypoint_sample_grid()
│   ├── compute_evaluation_results()
│   │   ├── group_image_objects() -> compute_object_keypoint_similarity()     OKS of every object with its prediction
│   │   ├── summarise_average_precisions(overlap_type="keypoints")            OKS mAP, 3 methods, sizes, recalls
│   │   ├── summarise_keypoint_accuracy() -> measure_keypoint_accuracy()      PCK@0.05, mean OKS, mean error
│   │   └── write_class_metrics_file()                                        class_metrics.csv, keypoint_metrics.csv
│   └── report_evaluation() -> report_evaluation_results() + track_evaluation()      headline 'OKS mAP@0.5/area'

scripts/evaluation/evaluate_checkpoint.py   [STAGE 5]
└── evaluate_checkpoints() -> run_test_evaluation() per checkpoint -> summarise_checkpoint_evaluations()

vision_training/deployment/export_model.py   [STAGE 6]
└── export_experiment()
    ├── load_trained_model() -> build_inference_model() -> InferenceKeypointDetector     input: crops only
    ├── get_export_specification() -> create_random_inputs()
    ├── export_torchscript() / export_onnx()
    ├── write_triton_configuration()                          config.pbtxt
    └── compare_outputs -> compare_keypoints()                exported model against the eager model
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below are real outputs from running the loader on `configurations/keypoint_detection/configuration_synthetic_shapes.yaml`, its data folders and mapping file redirected to the generated dataset and its iterations shortened as in Stage 3.

### Step 1.1 — `main()`
The entry point is the same for every task (see `classification_workflow.md`, Step 1.1). The run of Stage 3 was started with:

```text
CUDA_VISIBLE_DEVICES= PYTHONPATH=. COLUMNS=120 .venv/bin/python vision_training/main.py --configuration_path .../configuration_workflow.yaml
```

### Step 1.2 — `load_experiment_configuration()`
The model sections are `Optimization`, `Data`, `Backbone`, `HRNet` and `KeypointHead`. Three settings of `Data` belong to this task:
- `crop_settings`: the `height` and `width` of the crops fed to the network, and `box_enlargement`, the factor applied to the box (once widened to the aspect ratio of the crop), so that the crop also shows some context around the object.
- `crop_augmentation`: the random geometry of the training crops, drawn per crop (`scale_range`, `maximum_rotation_degrees`, `rotation_probability`, `flip_probability`, `maximum_shift_fraction`), and `flip_pairs`, the left/right keypoint pairs given by hand per class (empty: deduced from the keypoint names).
- `keypoint_categories`: not in the YAML, added by `inject_keypoint_categories()` (Step 1.5).

`HRNet` holds the network after the stem (`base_width`, the channels of its highest resolution branch, doubled at every lower branch; `modules_per_stage`; `blocks_per_branch`; `modules`; `normalization`; `pretrained`), and `KeypointHead` the `heatmap_sigma` of the targets, in heatmap cells. The `evaluation` section gives the OKS sigmas: `default_keypoint_sigma`, and `keypoint_sigmas` per class and keypoint.

**Example: the sections and the keypoint settings**
```text
experiment keys: ['task', 'experiment_folder', 'train_data_folder', 'validation_data_folder', 'test_data_folder', 'batch_size',
                  'information_dump', 'weight_saving_iterations', 'intermediate_weight_saving_iterations', 'number_iterations',
                  'validation_iteration_ratio', 'evaluation', 'tracking', 'learning_rate', 'resume_training', 'diagnosis_modes',
                  'tracking_root']
model keys:      ['Optimization', 'Data', 'Backbone', 'HRNet', 'KeypointHead']
evaluation:      {'per_class_report': True, 'number_sample_images': 22, 'default_keypoint_sigma': 0.05,
                  'keypoint_sigmas': {'stick_figure': {'left_elbow': 0.072, 'right_elbow': 0.072, 'left_hand': 0.062, 'right_hand': 0.062,
                                                       'left_knee': 0.087, 'right_knee': 0.087, 'left_foot': 0.089, 'right_foot': 0.089}}}
Data keys:       ['loading', 'image_settings', 'label_mapping', 'crop_settings', 'crop_augmentation', 'data_augmentation', 'label_mode',
                  'label_dictionary', 'keypoint_categories']
crop_settings:     {'height': 192, 'width': 192, 'box_enlargement': 1.25}
crop_augmentation: {'scale_range': [0.75, 1.25], 'maximum_rotation_degrees': 180.0, 'rotation_probability': 0.6, 'flip_probability': 0.5,
                    'maximum_shift_fraction': 0.1, 'flip_pairs': {}}
HRNet:        {"base_width": 16, "modules_per_stage": [1, 2, 2], "blocks_per_branch": 2,
               "modules": {"squeeze_and_excite": true, "residual_nets": true},
               "normalization": {"feature_map_normalization": "batch", "batch_normalization_momentum": 0.1},
               "pretrained": {"experiment_folder": null, "iteration": null, "freeze": false, "learning_rate_multiplier": 1.0}}
KeypointHead: {"heatmap_sigma": 2.0, "pretrained": {"experiment_folder": null, "iteration": null, "freeze": false, "learning_rate_multiplier": 1.0}}
```

### Step 1.3 — `load_configuration()`
**Example output**
```text
top-level keys: ['ExperimentConfiguration', 'Optimization', 'Data', 'Backbone', 'HRNet', 'KeypointHead']
number_iterations raw: 60
```

### Step 1.4 — `build_label_dictionary()`
As for the detection, `'background': 0` is added to the mapping file, the classes of the objects starting at 1.

**Example: the label dictionary of the generated dataset**
```text
label_dictionary: {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, ..., 20: 20, 21: 21, 22: 22, 'background': 0}
```

### Step 1.5 — `inject_keypoint_categories()` → `read_keypoint_categories()`
The keypoints are not described in the YAML but in the `categories` of the COCO `labels.json`, the way COCO describes its person keypoints. `inject_keypoint_categories()` reads them from the training split only (the validation and test splits share them) and writes `Data.keypoint_categories`, keyed by class index. It returns without writing anything when the training folder, the label dictionary or the `crop_augmentation.flip_pairs` dictionary is missing, and the checker of Step 1.6 then reports the problem. For every class of the label dictionary (the classes mapped to 0 are skipped), `read_keypoint_categories()` gives:
- `keypoints`: the names, in the order of the COCO `keypoints` triplets.
- `skeleton`: the pairs of keypoint indices drawn as lines in the viewers and the sample images.
- `maximum_rotation_degrees`: the rotation limit of the crop augmentation for the class. A square looks the same every 90 degrees, so rotating it by more than 45 degrees would ask the network to tell `top_left` from `top_right` on an identical image.
- `flip_pairs`: the pairs whose names swap when the crop is mirrored, from `crop_augmentation.flip_pairs` when the class is listed there, otherwise deduced from the `left`/`right` words of the names.

**Example: 5 of the 22 definitions**
```text
keypoint_categories: 22 classes, keys [1, 2, 3, ..., 22]
1  {"name": "square", "keypoints": ["top_left", "top_right", "bottom_right", "bottom_left", "centre"],
    "skeleton": [[0, 1], [1, 2], [2, 3], [3, 0]], "maximum_rotation_degrees": 45.0,
    "flip_pairs": [["top_left", "top_right"], ["bottom_right", "bottom_left"]]}
3  {"name": "triangle", "keypoints": ["top", "bottom_left", "bottom_right"], "skeleton": [[0, 2], [2, 1], [1, 0]],
    "maximum_rotation_degrees": 20.0, "flip_pairs": [["bottom_left", "bottom_right"]]}
9  {"name": "octagon", "keypoints": ["top", "right_1", "right_2", "right_3", "bottom", "left_3", "left_2", "left_1"],
    "skeleton": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0]], "maximum_rotation_degrees": 22.5,
    "flip_pairs": [["right_1", "left_1"], ["right_2", "left_2"], ["right_3", "left_3"]]}
21 {"name": "stick_figure", "keypoints": ["head", "neck", "left_elbow", "right_elbow", "left_hand", "right_hand", "pelvis",
    "left_knee", "right_knee", "left_foot", "right_foot"],
    "skeleton": [[0, 1], [1, 6], [1, 2], [2, 4], [1, 3], [3, 5], [6, 7], [7, 9], [6, 8], [8, 10]],
    "maximum_rotation_degrees": 180.0,
    "flip_pairs": [["left_elbow", "right_elbow"], ["left_hand", "right_hand"], ["left_knee", "right_knee"], ["left_foot", "right_foot"]]}
22 {"name": "house", "keypoints": ["roof_apex", "roof_left", "roof_right", "base_left", "base_right"],
    "skeleton": [[0, 2], [2, 4], [4, 3], [3, 1], [1, 0]], "maximum_rotation_degrees": 180.0,
    "flip_pairs": [["roof_left", "roof_right"], ["base_left", "base_right"]]}
```
The square's `top_left` and `top_right` swap names when the crop is mirrored: after a flip, the corner on the top left of the crop is the one that was labelled `top_right`.

### Step 1.6 — `check_configuration()`
The common checks run first, then `check_keypoint_configuration()`, which calls in order `check_image_task_sequence_settings()`, `check_keypoint_data_configuration()`, `check_crop_configuration()`, `check_crop_augmentation_configuration()`, `check_keypoint_stem_configuration()`, `check_high_resolution_network_configuration()` and `check_keypoint_evaluation_configuration()`. Three rules come from the architecture:
- The crop sides must be multiples of the stride of the lowest resolution branch of the HRNet (4 × 2^(number of stages), 32 with 3 stages), so that every branch can be upsampled back exactly to the highest one.
- The stem must end at 1/4 of the crop, the resolution of the heatmaps (`HEATMAP_STRIDE = 4`).
- The `random_rotation` of the `DataAugmenter` must be off: it would rotate the crops without their keypoints, which only `crop_augmentation` moves.

**Example: output with a crop height of 200, `random_rotation` on, `Backbone.last_max_pooling: True` and `KeypointHead.heatmap_sigma: 0.0` in the YAML**
```text
------------------------------------------------------
Invalid Configuration File : configuration_wrong.yaml
------------------------------------------------------
  - 'Data.crop_settings.height' (200) must be a multiple of 32, the stride of the lowest resolution branch of the 3 stages of 'HRNet.modules_per_stage'.
  - 'Data.data_augmentation.types.random_rotation.activated' must be False for the keypoint detection: the geometry of the crops is drawn by 'Data.crop_augmentation', which moves their keypoints with them.
  - The Backbone stem of the keypoint detector must end at 1/4 of the crop, got 1/8 from its 3 blocks with 'Backbone.last_max_pooling' True: use 3 blocks without the last max pooling, or 2 blocks with it.
  - 'KeypointHead.heatmap_sigma' must be a positive number of heatmap cells, got 0.0.
SystemExit code 1
```

### Step 1.7 — `get_trainer_class()`
**Example output**
```text
get_trainer_class(task="keypoint_detection") -> <class 'vision_training.trainers.keypoint_trainer.KeypointTrainer'>
```

---

## Stage 2 — Trainer Initialization

> The examples below are real outputs from building the trainer on the redirected configuration with the iterations of Stage 3 (batch size 16), the run folder redirected to a temporary folder and the MLflow tracking off. The whole construction took 0.5 s on CPU.

### Step 2.1 — `KeypointTrainer.__init__()` → `build_keypoint_tables()` + `build_keypoint_sigma_table()`
The trainer reads the crop settings, the crop augmentation and the heatmap sigma, turns the keys of `keypoint_categories` into integers and sets `maximum_keypoints`, the keypoint count of the largest class (11, the stick figure): every per-object keypoint tensor of the batches has 11 positions, the classes with fewer keypoints being padded. It then builds the tables indexed by the labels of a batch, so that the per-class lookups run on whole batches:
- `build_keypoint_tables()` gives `flip_permutations` [23, 11] (for every class, the position each keypoint takes in a mirrored crop, identity for the padding and the classes without flip pairs), `maximum_rotations` [23] (the rotation limit of every class, `DEFAULT_MAXIMUM_ROTATION_DEGREES` = 180 for the background row) and `keypoint_counts` [23].
- `build_keypoint_sigma_table()` gives `sigma_table` [23, 11], the OKS sigma of every keypoint of every class, `default_keypoint_sigma` unless `keypoint_sigmas` names the keypoint.

The constructor of `BaseTrainer` runs last, since it builds the loaders and the model that use these tables.

**Example: the attributes after the construction**
```text
crop 192 192 1.25   sigma 2.0   maximum_keypoints 11
flip_permutations (23, 11) torch.int64
  flip_permutations[ 0 ] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]            background
  flip_permutations[ 1 ] [1, 0, 3, 2, 4, 5, 6, 7, 8, 9, 10]            square: top_left <-> top_right, bottom_right <-> bottom_left
  flip_permutations[ 3 ] [0, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10]            triangle: bottom_left <-> bottom_right
  flip_permutations[ 21 ] [0, 1, 3, 2, 5, 4, 6, 8, 7, 10, 9]           stick_figure: 4 pairs
maximum_rotations [180.0, 45.0, 180.0, 20.0, 90.0, 36.0, 30.0, 45.0, 36.0, 22.5, 90.0, 90.0, 180.0, 180.0, 25.700000762939453, 20.0,
                   90.0, 180.0, 180.0, 90.0, 180.0, 180.0, 180.0]
keypoint_counts [0, 5, 1, 3, 4, 5, 6, 5, 5, 8, 4, 4, 4, 4, 7, 9, 3, 4, 1, 3, 6, 11, 5]
sigma_table (23, 11) torch.float64
  sigma_table[1] [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
  sigma_table[21] [0.05, 0.05, 0.072, 0.072, 0.062, 0.062, 0.05, 0.087, 0.087, 0.089, 0.089]
TASK_NAME = 'keypoint_detection'   SAMPLE_NAME = 'Objects'
```

### Step 2.2 — `BaseTrainer.__init__()` → `get_device()`
**Example output** (forced on the CPU; a normal run on this machine prints `cuda`)
```text
For Training We Will Be Using device: cpu
```

### Step 2.3 — `get_run_name()` → `build_run_name()` → `build_keypoint_sections()`
The prefix is `KPT`, followed by the dataset and the backbone sections of every task, then the sections of `build_keypoint_sections()`:
- `C192x192`: the crop height and width.
- `HRSE-HRRES-HRBN-HR16.1.2.2x2`: the HRNet modules (squeeze-and-excitation, residual shortcuts), its normalization (batch), and its base width, modules per stage and blocks per branch (`HRPT` and `HRFRZ` are added for a pretrained or frozen HRNet).
- `K107`: the heatmap channels of the head (`KHPT` and `KHFRZ` for a pretrained or frozen head).
- `MC22`: the number of classes.

The crop augmentation and the heatmap sigma give no token.

**Example output**
```text
KPT_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x64_C192x192_HRSE-HRRES-HRBN-HR16.1.2.2x2_K107_MC22
```

### Step 2.4 — `DataAugmenter()` + `setup_training_paths()`
The augmenter of every task. For this task it only applies its photometric types, to the crops (Step 3.6), the geometry of the crops being drawn by the trainer. The run folder is created as for every task (see `classification_workflow.md`).

### Step 2.5 — `build_dataloaders()` → `get_keypoint_dataloaders()` → `KeypointDataset()`
`build_dataloaders()` reads `class_names` from the test `labels.json` (`get_class_names()`) and calls `get_keypoint_dataloaders()`, which builds one `KeypointDataset` per split with `build_native_loader()`. The sample of a `KeypointDataset` is **one object**, not one image: an image of 4 shapes gives 4 samples. The objects of the background class and the objects without a single labelled keypoint are left out (in every split), and the samples are identified by their annotation identifier.

**Example: the loaders**
```text
class_names {0: 'background', 1: 'square', 2: 'circle', 3: 'triangle', 4: 'rectangle', 5: 'pentagon', 6: 'hexagon', 7: 'plus_sign',
             8: 'star', 9: 'octagon', 10: 'rhombus', 11: 'parallelogram', 12: 'trapezoid', 13: 'kite', 14: 'heptagon', 15: 'nonagon',
             16: 'ellipse', 17: 'semicircle', 18: 'annulus', 19: 'capsule', 20: 'l_shape', 21: 'stick_figure', 22: 'house'}
train      KeypointDataset 893 objects 56 batches workers 2 prefetch 2 pin True persistent True drop_last False
validation KeypointDataset  86 objects  6 batches workers 2 prefetch 2 pin True persistent True drop_last False
test       KeypointDataset  82 objects  6 batches workers 2 prefetch 2 pin True persistent True drop_last False
test list_objects()[:3] [{'sample_identifier': 16, 'label': 1, 'labelled_keypoints': 5},
                         {'sample_identifier': 29, 'label': 20, 'labelled_keypoints': 6},
                         {'sample_identifier': 52, 'label': 2, 'labelled_keypoints': 1}]
```
The training split has 894 annotations but 893 objects: one object has no labelled keypoint.

### Step 2.6 — `KeypointDataset.__getitem__()`
A sample holds the **whole** resized image of the object (the crop is cut later on the device, Step 3.3), its box in `[x1, y1, x2, y2]` resized image pixels, its class, its keypoints padded to `maximum_keypoints` and their COCO visibilities (2 visible, 1 occluded, 0 unlabelled). The unlabelled keypoints and the padding positions are set to (0, 0) with visibility 0. `get_sample_name()` names the sample after its image and annotation.

**Example: the first test object, annotation 16 (a square) and its record in `labels.json`**
```text
record annotation {'id': 16, 'image_id': 6, 'category_id': 1, 'bbox': [220, 330, 146, 146], 'area': 21316, 'iscrowd': 0,
                   'keypoints': [220.0, 330.0, 2, 366.0, 330.0, 2, 366.0, 476.0, 2, 220.0, 476.0, 2, 293.0, 403.0, 2], 'num_keypoints': 5}
  images               (3, 512, 512) torch.uint8
  bounding_boxes       (4,) torch.float32 [220.0, 330.0, 366.0, 476.0]
  labels               () torch.int64 1
  keypoints            (11, 2) torch.float32 [[220.0, 330.0], [366.0, 330.0], [366.0, 476.0], [220.0, 476.0], [293.0, 403.0],
                                              [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
  visibilities         (11,) torch.int64 [2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0]
  image_identifiers    6
  sample_identifiers   16
get_sample_name -> test_000005_object_16
```

### Step 2.7 — `get_test_sample_classes()` + `choose_sample_identifiers()`
`get_test_sample_classes()` gives the class of every test object from `KeypointDataset.list_objects()`, so that the 22 fixed test objects drawn at every evaluation (`number_sample_images: 22`) are spread over the classes.

**Example output**
```text
sample_image_identifiers [1, 2, 3, 4, 5, 7, 9, 10, 11, 14, 15, 16, 18, 21, 23, 26, 31, 34, 36, 39, 41, 49]
names ['test_000000_object_1', 'test_000000_object_2', 'test_000001_object_3', 'test_000001_object_4', 'test_000001_object_5',
       'test_000002_object_7', ...]
```

### Step 2.8 — `build_model()` → `KeypointDetector.__init__()` → `build_keypoint_channel_table()`
The detector has one heatmap channel per keypoint of every class, the channels of a class being consecutive: 107 channels for the 22 classes. `build_keypoint_channel_table()` gives `channel_table` [23, 11], the channel of every keypoint position of every class (-1 for the padding and the background), registered as a buffer so that it is saved in the checkpoint, and `channel_names`. The detector is made of:
- `backbone`: `build_backbone(input_image_size=crop height)`, used as a stem ending at 1/4 of the crop; its last convolution block gives the input channels of the HRNet (64).
- `high_resolution_network`: `HighResolutionNetwork`, which keeps a 1/4 branch from start to end and adds one lower resolution branch at each of its stages, exchanging features between all branches, then returns the 1/4 branch (`base_width` = 16 channels).
- `keypoint_head`: a 1x1 `nn.Conv2d` from 16 to 107 channels, initialised with a normal distribution of standard deviation `HEAD_INITIALISATION_STANDARD_DEVIATION` (0.001) and a zero bias, so that the first heatmaps are close to 0, like most of the target cells.

**Example: the channel table**
```text
channel_table (23, 11)
  channel_table[ 0 ] [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
  channel_table[ 1 ] [0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1]
  channel_table[ 2 ] [5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
  channel_table[ 21 ] [91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101]
  channel_table[ 22 ] [102, 103, 104, 105, 106, -1, -1, -1, -1, -1, -1]
channel_names 107 ['square.top_left', 'square.top_right', 'square.bottom_right', 'square.bottom_left', 'square.centre', 'circle.centre',
                   'triangle.top'] ... ['house.roof_apex', 'house.roof_left', 'house.roof_right', 'house.base_left', 'house.base_right']
```

### Step 2.9 — `initialise_pretrained_components()` + `build_optimizer()` + `get_tracked_metrics()`
The three transferable components are the `Backbone`, the `HRNet` and the `KeypointHead`, each with its `pretrained` section (all null here). The tracked metrics are `{"total_loss": "Total Loss", "keypoint_loss": "Keypoint Heatmap Loss"}`.

### Step 2.10 — `print_experiment_information()` (→ `describe_split()` → `count_split_images_and_annotations()`)
`describe_split()` gives the number of objects of every split (the number of samples of its loader) with the images and annotations of its `labels.json`, so that the iterations per epoch are counted in objects: 893 objects / 16 = 56 iterations.

**Example output** (from the run of Stage 3, with the tracking on; the paths are shortened)
```text
--------------------------------------
=== Experiment Launch Information ===
--------------------------------------
- Run Name                : KPT_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x64_C192x192_HRSE-HRRES-HRBN-HR16.1.2.2x2_K107_MC22
- Experiment Folder       : KPT_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x64_C192x192_HRSE-HRRES-HRBN-HR16.1.2.2x2_K107_MC22
- Tensorboard Directory   : Tensorboard
- Weights Directory       : Weights
- Configuration File      : training_configuration.yaml
- Data Record             : data.json
- Train Data Folder       : train (893 objects, from 300 images and 894 annotations)
- Validation Data Folder  : validation (86 objects, from 30 images and 86 annotations)
- Test Data Folder        : test (82 objects, from 30 images and 82 annotations)
- Iterations Per Epoch    : 56
- Batch Size              : 16
- Total Iterations        : 60
- Save And Evaluate Every : 30 iterations
- Save Only Every         : 15 iterations
- Learning Rate           : 0.001
- Optimizer               : adam (weight decay 0.0)
- Learning Rate Schedule  : cosine (warm-up 10 iterations)
- Gradient Clipping Norm  : None
- Weight Moving Average   : decay 0.999, evaluated next to the trained weights
- Mixed Precision         : False
- Resume Training         : True
- Validation Iter. Ratio  : 10
- MLflow Tracking         : sqlite:///.../runs/mlflow.db
```

### Step 2.11 — `print_summary()` / `print_receptive_field_summary()` / `print_model_size_summary()`
`KeypointDetector.print_summary()` prints the backbone summary traced on a crop, then the branches of every HRNet stage, traced in evaluation mode, and the heatmap head. The stem divides the 192 pixel crop by 4: the max pooling of block 1 (1/2), the max pooling of block 2 (1/4), and no max pooling after block 3 (`last_max_pooling: False`). The HRNet panel counts the stem as its first stage, so its 3 stages of `modules_per_stage` have 2, 3 and 4 branches.

**Example output**
```text
╭─────────────────────────── Backbone Architecture ────────────────────────────╮
│ Normalization Strategy: Batch                                                │
│                                                                              │
│  - Convolution Block 1                                                       │
│    - Modules      : Squeeze-and-Excitation                                   │
│    - Input Shape  : [1, 3, 192, 192]                                         │
│    - Output Shape : [1, 16, 96, 96]                                          │
│                                                                              │
│  - Convolution Block 2                                                       │
│    - Modules      : Residual Shortcuts, Squeeze-and-Excitation               │
│    - Input Shape  : [1, 16, 96, 96]                                          │
│    - Output Shape : [1, 32, 48, 48]                                          │
│                                                                              │
│  - Convolution Block 3                                                       │
│    - Modules      : Residual Shortcuts, Squeeze-and-Excitation               │
│    - Input Shape  : [1, 32, 48, 48]                                          │
│    - Output Shape : [1, 64, 48, 48]                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
---------------------------------------------
Final Backbone output shape: [1, 64, 48, 48]
---------------------------------------------
╭─────────────────────────────── High Resolution Network ────────────────────────────────╮
│ Base Width          : 16                                                               │
│ Modules Per Stage   : [1, 2, 2]                                                        │
│ Blocks Per Branch   : 2                                                                │
│ Stem Output         : [1, 64, 48, 48]                                                  │
│                                                                                        │
│  - Stage 1 (2 branches)                                                                │
│    - Branch 1/4  : [1, 16, 48, 48]                                                     │
│    - Branch 1/8  : [1, 32, 24, 24]                                                     │
│  - Stage 2 (3 branches)                                                                │
│    - Branch 1/4  : [1, 16, 48, 48]                                                     │
│    - Branch 1/8  : [1, 32, 24, 24]                                                     │
│    - Branch 1/16 : [1, 64, 12, 12]                                                     │
│  - Stage 3 (4 branches)                                                                │
│    - Branch 1/4  : [1, 16, 48, 48]                                                     │
│    - Branch 1/8  : [1, 32, 24, 24]                                                     │
│    - Branch 1/16 : [1, 64, 12, 12]                                                     │
│    - Branch 1/32 : [1, 128, 6, 6]                                                      │
│                                                                                        │
│ Heatmap Channels    : 107 (the keypoints of 22 classes)                                │
│ Heatmap Size        : 48x48 (stride 4)                                                 │
│ Heatmap Sigma       : 2.0 cells                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────── Backbone Receptive Field ────────────────────────────────────╮
│ Input Image Size : 192x192                                                                       │
│                                                                                                  │
│ Block 1 - Convolution 1  : Output 192x192x16 | Stride 1 | Receptive Field 3                      │
│ Block 1 - Max Pooling    : Output 96x96x16 | Stride 2 | Receptive Field 4                        │
│                                                                                                  │
│ Block 2 - Convolution 1  : Output 96x96x32 | Stride 2 | Receptive Field 8                        │
│ Block 2 - Convolution 2  : Output 96x96x32 | Stride 2 | Receptive Field 12                       │
│ Block 2 - Max Pooling    : Output 48x48x32 | Stride 4 | Receptive Field 14                       │
│                                                                                                  │
│ Block 3 - Convolution 1  : Output 48x48x64 | Stride 4 | Receptive Field 22                       │
│ Block 3 - Convolution 2  : Output 48x48x64 | Stride 4 | Receptive Field 30                       │
│                                                                                                  │
│ Squeeze-and-Excitation adds global image context.                                                │
│ This is not captured by the convolutional receptive field above.                                 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────── Model Size ───────────────────────╮
│ Backbone                                                 │
│   - Parameters           : 80,672                        │
│   - Estimated Size       : 0.32 MB                       │
│ High Resolution Network                                  │
│   - Parameters           : 2,631,456                     │
│   - Estimated Size       : 10.56 MB                      │
│ Keypoint Head                                            │
│   - Parameters           : 1,819                         │
│   - Estimated Size       : 0.01 MB                       │
│ Total Model                                              │
│   - Parameters           : 2,713,947                     │
│   - Estimated Size       : 10.90 MB                      │
╰──────────────────────────────────────────────────────────╯
```
The model size panel is shortened here (it also gives the trainable parameters, equal to the parameters, and an empty line between the components). The keypoint head has 16 × 107 weights + 107 biases = 1,819 parameters; almost all of the model is in the HRNet. The receptive field of the stem (30 pixels) is small next to a 192 pixel crop: the context of the whole object comes from the lower branches of the HRNet.

### Step 2.12 — `view_input_batches()`
Off in this run (`diagnosis_modes.view_input_images: False`); the panels it would show are given in Step 3.9.

### Step 2.13 — `restore_last_model()`
**Example output** (a new run)
```text
-----------------------------------------------------------
No Initiation Model Weights Will Be Used...
We Generate A New Model That Will Be Trained From Scratch.
-----------------------------------------------------------
```

---

## Stage 3 — Training Loop

> The examples below are real outputs from a run on CPU of the redirected configuration shortened to batch size 16, 60 iterations, a checkpoint and a test evaluation every 30 iterations, an intermediate checkpoint every 15, a console summary every 10, a validation batch every 10 training iterations, 2 loading workers, with the MLflow tracking on. Its run name is the one of Step 2.3, written `<run>` below. The 60 iterations, with the two test evaluations of the trained and the averaged weights, took 3 min 8 s. The shapes inside one batch come from the trainer of Stage 2 (seed 0, the augmentations drawn with seed 3), on the first training batch.

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
The loop of every task (see `classification_workflow.md`, Step 3.1).

**Example: what happens at each iteration of the run**

| Iteration | 10 | 15 | 20 | 30 | 40 | 45 | 50 | 60 | after the loop |
|---|---|---|---|---|---|---|---|---|---|
| Checkpoint saved | | ✔ | | ✔ | | ✔ | | | ✔ (iteration 60) |
| Test evaluation (trained and EMA weights) | | | | ✔ | | | | | ✔ (iteration 60) |
| Validation batch | ✔ | | ✔ | ✔ | ✔ | | ✔ | ✔ | |
| Console summary | ✔ | | ✔ | ✔ | ✔ | | ✔ | ✔ | |

The checkpoint and the evaluation of iteration 30 run at the start of iteration 30, before its batch.

### Step 3.2 — `setup_tensorboard_writers()` / `start_experiment_tracking()` / `save_model()`
As for every task. The checkpoint holds the source run of every transferable component of the task, here three: `pretrained_backbone`, `pretrained_high_resolution_network` and `pretrained_keypoint_head` (None without transfer), and the state of the moving average of the weights.

**Example output**
```text
MLflow Run Created In The Experiment 'keypoint_detection' : 5e453d8d7cc3429eb3610310e2ca654b (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)

Checkpoint Folder :
 - Iteration_15
 - Model Saved In : model_000015.pt
```

**Example: the checkpoint of iteration 60**
```text
Weights/Iteration_15/model_000015.pt   44,608,812 bytes      Weights/Iteration_30/model_000030.pt   44,608,812 bytes
Weights/Iteration_45/model_000045.pt   44,608,812 bytes      Weights/Iteration_60/model_000060.pt   44,608,812 bytes

checkpoint keys ['exponential_moving_average_state', 'format_version', 'iteration', 'mlflow_run_id', 'model_state', 'optimizer_state',
                 'pretrained_backbone', 'pretrained_high_resolution_network', 'pretrained_keypoint_head', 'run_name', 'task']
format_version 2   iteration 60   task 'keypoint_detection'   mlflow_run_id '5e453d8d7cc3429eb3610310e2ca654b'
pretrained_backbone None   pretrained_high_resolution_network None   pretrained_keypoint_head None
model_state tensors 794
  channel_table          (23, 11)
  keypoint_head.weight   (107, 16, 1, 1)
  keypoint_head.bias     (107,)
```

### Step 3.3 — `get_next_batch()` → `run_model_iteration()` → `prepare_batch()` → `cut_object_crops()`
The loader batch holds 16 objects, each with its whole 512 pixel image. `prepare_batch()` moves the images (in the precision of the model), the boxes and keypoints (float32) and the labels and visibilities (int64) to the device, then cuts the crop of every object **without augmentation**, through `cut_object_crops(augmentations=create_identity_augmentations())`. These identity crops are the ones the validation and the test evaluation run on; `augment_batch()` cuts new ones for the training batches (Step 3.4). `cut_object_crops()` runs, on the device of the images:
- `compute_crop_regions()`: the centre of the box, and its sides widened to the aspect ratio of the crop (square here: the longest side), times `box_enlargement`. A 84x35 box gives a region of 84 × 1.25 = 105 pixels.
- `build_crop_affines()`: the 2x3 affine `image_to_crop` (`crop_affines`), which zooms the region to the crop (192 / 105 = 1.8286), and its inverse `crop_to_image` (`crop_inverse_affines`). With an augmentation, the zoom is also divided by the scale, the rotation is anticlockwise around the centre, the flip mirrors the crop and the shift moves the centre by a fraction of the region.
- `warp_crops()`: bilinear `grid_sample` of every pixel of the crop at its position in the image, zeros outside the image.
- `transform_points()`: the keypoints moved into the crop with `image_to_crop`; `swap_flipped_keypoints()` permutes them for the flipped crops (Step 3.4).
- `build_heatmap_targets()`: one Gaussian of sigma `heatmap_sigma` cells per keypoint on the 48x48 heatmap (crop / `HEATMAP_STRIDE`), centred at keypoint / 4 − 0.5 cells, and its weight, 1 for a labelled keypoint inside the crop, 0 otherwise (unlabelled, padding, or moved out of the crop by the augmentation).

**Example: the loader batch and the prepared batch**
```text
loader batch                                         prepare_batch()
  images               (16, 3, 512, 512) uint8         images                 (16, 3, 512, 512) float32
  bounding_boxes       (16, 4) float32                 bounding_boxes         (16, 4) float32
  labels               (16,) int64                     labels                 (16,) int64
  keypoints            (16, 11, 2) float32             keypoints              (16, 11, 2) float32
  visibilities         (16, 11) int64                  visibilities           (16, 11) int64
  image_identifiers    (16,) int64                     image_identifiers      (16,) int64
  sample_identifiers   (16,) int64                     sample_identifiers     (16,) int64
                                                       crops                  (16, 3, 192, 192) float32
                                                       crop_affines           (16, 2, 3) float32
                                                       crop_inverse_affines   (16, 2, 3) float32
                                                       crop_keypoints         (16, 11, 2) float32
                                                       crop_visibilities      (16, 11) int64
                                                       target_heatmaps        (16, 11, 48, 48) float32
                                                       heatmap_weights        (16, 11) float32
```

**Example: sample 0, a rectangle (4 keypoints: top_left, top_right, bottom_right, bottom_left)**
```text
box [171.0, 360.0, 255.0, 395.0]                     84x35 pixels
centre [213.0, 377.5]   size [105.0, 105.0]          the longest side 84 x 1.25, square like the crop
image_to_crop [[1.8285715579986572, 0.0, -293.4857482910156], [-0.0, 1.8285715579986572, -594.2857666015625]]
crop_to_image [[0.5468749403953552, 0.0, 160.5], [0.0, 0.5468749403953552, 325.0]]
keypoints      [[171.0, 360.0], [255.0, 360.0], [255.0, 395.0], [171.0, 395.0], [0.0, 0.0] x 7]
visibilities   [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0]
crop_keypoints [[19.2, 64.0], [172.8, 64.0], [172.8, 128.0], [19.2, 128.0], [-293.49, -594.29] x 7]
heatmap_weights [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
target maxima   [0.9584, 0.9584, 0.9584, 0.9584, 0.0, ...]
target argmax (row, column) [(15, 4), (15, 43), (31, 43), (31, 4), (0, 0), ...]
crops range 0.0 253.0
```
The box (84 pixels wide) fills 19.2 to 172.8 of the 192 pixel crop, 80 % of it, since 1 / 1.25 = 0.8. The `top_left` corner (19.2, 64) falls at (4.8 − 0.5, 16 − 0.5) = (4.3, 15.5) cells, between two cells, so the target maximum is 0.9584 and not 1. The unlabelled positions hold the image origin moved into the crop (−293.49, −594.29); their weight is 0, so they are never trained.

### Step 3.4 — `augment_batch()` → `draw_crop_augmentations()` → `cut_object_crops()` + `DataAugmenter.augment_batch()`
`draw_crop_augmentations()` draws, on the CPU, one scale in `scale_range` (above 1 the object looks smaller), one rotation in ± the limit of the class (the smaller of `maximum_rotation_degrees` and the `maximum_rotation_degrees` of the class) applied with `rotation_probability`, one flip with `flip_probability` and one shift of at most `maximum_shift_fraction` of the region. `cut_object_crops()` then cuts the crops again with them: the keypoints move with the image, `swap_flipped_keypoints()` renames them in the flipped crops with `flip_permutations`, and the targets are rebuilt. Last, the photometric augmentations of the `DataAugmenter` are applied to the crops, `DataAugmenter.augment_batch(images=crops)` returning its six values, of which the trainer keeps the crops and the `augmentation_reports`. The batch also gets the drawn `crop_augmentations`, which `describe_crop_augmentation()` turns into the lines of the input viewer.

**Example: the drawn augmentations of the batch (first 6 of 16)**
```text
augment_batch keys added ['crop_augmentations', 'augmentation_reports']
scales     (16,) [0.752, 0.803, 0.893, 0.763, 0.986, 0.78]
rotations  (16,) [-58.391, 161.706, 62.517, 134.948, 7.623, -25.672]
flips      (16,) [True, False, False, True, True, False]
shifts     (16, 2) [[-0.075, -0.083], [0.002, -0.072], [0.053, 0.045], [0.022, 0.082], [-0.045, 0.001], [0.061, 0.035]]
augmentation_reports[0] {'augmented': False, 'geometric_augmentations': [], 'photometric_augmentations': [], 'removed_labels': None}
```

**Example: sample 0 again, the rectangle, flipped (rotation limit of the class 90 degrees)**
```text
describe_crop_augmentation ['Crop :', '  scale : 0.75', '  rotation : -58.4 degrees', '  flip : yes', '  shift : -0.07 width, -0.08 height']
identity crop keypoints  [[19.2, 64.0], [172.8, 64.0], [172.8, 128.0], [19.2, 128.0], ...]
augmented crop keypoints [[50.7, -18.5], [157.7, 155.5], [85.3, 200.1], [-21.8, 26.1], ...]
augmented visibilities   [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0]
augmented weights        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```
With a scale of 0.75 the region is 0.75 times as large, so the object looks 1 / 0.75 = 1.33 times larger and, rotated by 58 degrees, three of its four corners leave the 192x192 crop: their weight drops to 0, and only the corner that stays inside (157.7, 155.5) is trained.

**Example: the rotation limits on a batch of 6 squares (limit 45 degrees) and 6 stick figures (180 degrees), seed 5**
```text
rotations [0.0, 14.3, 19.4, 0.0, 43.3, 13.5,   -159.6, 151.2, 61.1, -85.9, 0.0, 0.0]
scales    [1.17, 0.81, 1.2, 1.16, 1.21, 0.81,  0.83, 1.12, 0.77, 1.25, 1.05, 1.03]
```
The rotations of 0.0 are the crops left unrotated (`rotation_probability` 0.6).

### Step 3.5 — `forward_pass()` → `KeypointDetector.forward(input_tensor=crops, labels, target_heatmaps, heatmap_weights)`
The crops go through the stem, the HRNet and the 1x1 head, giving the 107 heatmaps of every crop, decoded by `decode_heatmaps()` into a keypoint and a score per channel. With the labels, `channel_table[labels]` picks the channels of the class of every crop (the padding positions read channel 0 and are zeroed), giving `category_keypoints` and `category_keypoint_scores` in the keypoint order of the class. With the targets, the heatmap loss is computed on these channels only (Step 3.7); the 102 other channels of a crop get no gradient from it. The scores are the raw heatmap maxima, without sigmoid, so a channel can score below 0.

**Example: the outputs on the first training batch**
```text
heatmaps                   (16, 107, 48, 48) float32
keypoints                  (16, 107, 2) float32
keypoint_scores            (16, 107) float32
category_keypoints         (16, 11, 2) float32
category_keypoint_scores   (16, 11) float32
keypoint_loss              () float32 0.005573040805757046
category_keypoints[0]       [[117.0, 190.0], [61.0, 163.0], [23.0, 93.0], [157.0, 5.0], [0.0, 0.0] x 7]
category_keypoint_scores[0] [0.03534, 0.00485, 0.02832, -0.00528, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

### Step 3.6 — `decode_heatmaps()`
The keypoint of a heatmap is its maximum cell, moved by `SUBPIXEL_SHIFT` (a quarter of a cell) toward the larger of its two neighbours, in x and in y, when they differ (no shift on the border of the heatmap), then converted to crop pixels: x = (column + shift + 0.5) × stride. The score is the value of the maximum.

**Example: a toy 5x5 heatmap with stride 4**
```text
heatmap 0: maximum 0.9 at (row 2, column 2), right neighbour 0.5 > left 0, upper neighbour 0.3 > lower 0
heatmap 1: maximum 0.8 at (row 2, column 2), left and right neighbours both 0.4
heatmap 2: maximum 0.7 at (row 0, column 4), on the border, lower neighbour 0.2
keypoints [[11.0, 9.0], [10.0, 10.0], [18.0, 2.0]]   scores [0.9, 0.8, 0.7]
```
Heatmap 0: x = (2 + 0.25 + 0.5) × 4 = 11, y = (2 − 0.25 + 0.5) × 4 = 9. Heatmap 1: equal neighbours, no shift, (10, 10). Heatmap 2: on the border, no shift, x = (4 + 0.5) × 4 = 18, y = 0.5 × 4 = 2.

### Step 3.7 — `compute_keypoint_heatmap_loss()`
The mean squared error of every heatmap over its cells, then averaged over the heatmaps weighted by `heatmap_weights` (× the valid positions of the class): Σ(error × weight) / max(Σ weight, 1), in float32. A crop whose keypoints all left it adds nothing, and a batch without any weighted keypoint gives 0 instead of a division by zero.

**Example: a toy pair of 4x4 heatmaps predicted at 0, heatmap 0 with one target cell at 1, heatmap 1 with four**
```text
weights [1.0, 1.0] loss 0.15625        (1/16 + 4/16) / 2
weights [1.0, 0.0] loss 0.0625         1/16
weights [0.0, 1.0] loss 0.25           4/16
weights [0.0, 0.0] loss 0.0
```

### Step 3.8 — `training_loss.backward()` + `step_optimizer()` + validation pass
As for every task; the moving average of the weights (decay 0.999) is updated after every optimizer step. The validation batches go through `prepare_batch()` only, so their loss is measured on the identity crops.

### Step 3.9 — `view_input_batches()` → `show_input_samples()` → `get_input_crop()` + `draw_keypoint_crop()`
Off in this run. With `view_input_images: True`, the viewer shows an augmented training crop next to a validation crop (cut alone by `get_input_crop()`), each drawn by `draw_keypoint_crop()` with its labelled keypoints and skeleton, beside a panel holding the class, the keypoints by visibility, the keypoints trained inside the crop, the colours and the augmentations. The panels below were printed by replacing the window by a print of its lines.

**Example: the panels of the first two pairs**
```text
[Training]                                                 [Validation]
Class : annulus                                            Class : nonagon
Keypoints : 1 visible, 0 occluded, 0 unlabelled, of 1      Keypoints : 6 visible, 0 occluded, 3 unlabelled, of 9
Trained : 1 labelled keypoints inside the crop             Trained : 6 labelled keypoints inside the crop
Colours : left blue, right orange, others yellow,          Colours : left blue, right orange, others yellow,
          rings occluded                                             rings occluded
Augmentations                                              Augmentations : none (validation sample)
Crop :
  scale : 0.89
  rotation : 125.0 degrees
  flip : no
  shift : +0.00 width, -0.07 height
Geometric :
  none
Photometric :
  random_blur : 9x9 kernel, sigma 2.5

[Training]                                                 [Validation]
Class : triangle                                           Class : triangle
Keypoints : 3 visible, 0 occluded, 0 unlabelled, of 3      Keypoints : 2 visible, 1 occluded, 0 unlabelled, of 3
Trained : 2 labelled keypoints inside the crop             Trained : 3 labelled keypoints inside the crop
Colours : left blue, right orange, others yellow,          Colours : left blue, right orange, others yellow,
          rings occluded                                             rings occluded
Augmentations                                              Augmentations : none (validation sample)
Crop :
  scale : 0.76
  rotation : 15.0 degrees
  flip : yes
  shift : +0.05 width, +0.04 height
Geometric :
  none
Photometric :
  brightness_contrast_saturation : brightness -32.5 levels, contrast x0.88, saturation x0.73
  random_noise : standard deviation 40 levels
  random_blur : 9x9 kernel, sigma 2.5
```
The training triangle has 3 labelled keypoints but only 2 are trained: the scale of 0.76 pushed one corner out of the crop.

### Step 3.10 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()`
The summary counts objects, not images.

**Example output: the summaries of iterations 10 and 60**
```text
|    Iteration: 10    |     Total Objects: 160    |   Interval Objects: 160    |   Throughput: 5.9 Objects Per Second  |
Moving Average of Train Total Loss                     :   0.0054
Moving Average of Valid Total Loss                     :   0.0054
Moving Average of Train Keypoint Heatmap Loss          :   0.0054
Moving Average of Valid Keypoint Heatmap Loss          :   0.0054
Interval Of 10 Iterations Took 00:00:27.
Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).

|    Iteration: 60    |     Total Objects: 960    |   Interval Objects: 160    |   Throughput: 5.7 Objects Per Second  |
Moving Average of Train Total Loss                     :   0.0050  +0.0000 Best
Moving Average of Valid Total Loss                     :   0.0049  -0.0002 Best
Moving Average of Train Keypoint Heatmap Loss          :   0.0050  +0.0000 Best
Moving Average of Valid Keypoint Heatmap Loss          :   0.0049  -0.0002 Best
Interval Of 10 Iterations Took 00:00:27.
Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
```

**Example: the training losses of the run**

| Iteration | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|
| Train Keypoint Heatmap Loss | 0.0054 | 0.0054 | 0.0052 | 0.0051 | 0.0050 | 0.0050 |
| Valid Keypoint Heatmap Loss | 0.0054 | 0.0054 | 0.0052 | 0.0051 | 0.0051 | 0.0049 |
| Throughput (Objects Per Second) | 5.9 | 6.1 | 4.4 | 5.9 | 5.9 | 5.7 |

The total loss is the heatmap loss, the only loss of the task. It is small from the start because most target cells are 0 and the head starts close to 0; it only drops once the network learns to raise the cells around the keypoints. The interval of iteration 30 took 36 s instead of 27 s, as it holds the two test evaluations of iteration 30.

### Step 3.11 — end of the loop
**Example output**
```text
Checkpoint Folder :
 - Iteration_60
 - Model Saved In : model_000060.pt
Model successfully saved at iteration 60. Exiting Training.
```

**Example: the TensorBoard curves after the run**
```text
Tensorboard/Train/        Total Loss, Keypoint Heatmap Loss, Learning Rate                          (iterations 1 to 60)
Tensorboard/Validation/   Total Loss, Keypoint Heatmap Loss                                         (iterations 10, 20, ..., 60)
Tensorboard/Test/         47 scalars and 22 images per evaluation                                   (iterations 30 and 60)
Tensorboard/Test_EMA/     47 scalars and 22 images per evaluation, the same tags                    (iterations 30 and 60)
```

---

## Stage 4 — Test Evaluation of a Checkpoint

> The examples below are real outputs of the evaluations of the checkpoints of iterations 30 and 60 of the run of Stage 3, on its whole test split (82 objects of 30 images), with `per_class_report: True` and `number_sample_images: 22`. The steps inside one test batch come from the checkpoint of iteration 60 loaded by `load_trained_model()` into the trainer of Stage 2, on the first test batch.

### Step 4.1 — `run_test_evaluation()` → `evaluate_test_split()`
The evaluation of every task (see `classification_workflow.md`, Stage 4): the trained weights are evaluated by `evaluate_test_split()`, then the averaged weights in `use_averaged_weights()`, their files written in the `EMA` subfolder of the evaluation folder. The keypoint trainer keeps the default `evaluation_context()`, which changes nothing. The model runs as in deployment: on the ground truth boxes, cut into identity crops by `prepare_batch()`.

### Step 4.2 — `create_evaluation_accumulator()`
One list per key of `EVALUATED_OBJECT_KEYS`, extended batch by batch on the CPU.

**Example: the accumulated objects of the whole test split**
```text
{'image_identifiers': (82,), 'labels': (82,), 'bounding_boxes': (82, 4), 'keypoints': (82, 11, 2), 'visibilities': (82, 11),
 'predicted_keypoints': (82, 11, 2), 'scores': (82,)}
```

### Step 4.3 — `prepare_batch()` + `predict_batch()`
`predict_batch()` runs the detector with the labels and without targets, then:
- moves the keypoints of the class from the crop back into the resized image with `transform_points(crop_inverse_affines)` (`image_keypoints`), where the ground truth is and the OKS is measured;
- scores every object with the mean of the scores of the valid keypoints of its class (`object_scores`). This score ranks the objects in the average precision, as the detection score ranks the detections.

**Example: the outputs on the first test batch**
```text
predict_batch keys ['category_keypoint_scores', 'category_keypoints', 'heatmaps', 'image_keypoints', 'keypoint_scores', 'keypoints',
                    'object_scores']
heatmaps (16, 107, 48, 48)   keypoints (16, 107, 2)   keypoint_scores (16, 107)   category_keypoints (16, 11, 2)
category_keypoint_scores (16, 11)   image_keypoints (16, 11, 2)   object_scores (16,)          all float32
```

**Example: sample 0, a kite (keypoints top, right, bottom, left)**
```text
box [224.0, 440.0, 287.0, 512.0]
ground truth image keypoints [[255.5, 440.5], [286.5, 467.5], [0.0, 0.0], [224.5, 467.5]]      bottom unlabelled
crop keypoints predicted     [[83.0, 27.0], [67.0, 157.0], [35.0, 93.0], [35.0, 93.0]]
image keypoints predicted    [[249.4, 443.7], [241.9, 504.6], [226.9, 474.6], [226.9, 474.6]]
keypoint scores              [0.0386, 0.0581, 0.0771, 0.0968]
object score 0.0677                                                                              mean of the 4 scores
object_scores[:8] [0.0677, 0.0513, 0.0655, 0.0649, 0.0421, 0.0438, 0.0407, 0.0265]
```
After 60 iterations, the `top` keypoint is 7 pixels from its ground truth, but `bottom` and `left` fall on the same cell: the network does not tell them apart yet.

### Step 4.4 — `accumulate_evaluation_batch()`
Keeps the image identifier, class, box, ground truth keypoints and visibilities, predicted image keypoints and score of every kept object (`sample_indices`), all in resized image pixels.

### Step 4.5 — `save_sample_images()` → `build_sample_image()` → `measure_keypoint_accuracy()` + `build_keypoint_sample_grid()`
For the 22 fixed test objects, `build_sample_image()` measures the accuracy of the object alone and draws, with `build_keypoint_sample_grid()`, its crop with its labelled keypoints (Ground Truth) next to the same crop with its predicted keypoints (Prediction), both with the skeleton of the class, under a caption.

**Example: `Evaluation/Iteration_60/Samples/test_000014_object_41.png`** (304x516 pixels)
```text
caption : heptagon | OKS 0.283 | 2/7 keypoints within 0.05 of the box | score 0.070
panels  : Ground Truth | Prediction        legend : Left / Right / Centre
```

### Step 4.6 — `compute_evaluation_results()` → `group_image_objects()` → `compute_object_keypoint_similarity()`
The average precision of the detection needs, for every image, the detections, the ground truth objects and an overlap matrix between them. `group_image_objects()` builds them from the objects: every object is one ground truth and its prediction is one detection with the object's box, class and score, and the overlap of a detection with a ground truth is their OKS. A prediction is only compared to its own object (the overlaps of the other objects of the image are 0), since the crop was cut on that object; the matrix is diagonal. The area of the OKS is the area of the box, computed by `compute_box_areas()`.

`compute_object_keypoint_similarity()` is the COCO formula: for every labelled keypoint (visibility > 0), exp(−d² / (2 × (2σ)² × area)), averaged over the labelled keypoints of the object, in float64 (the area clamped to `MINIMUM_OBJECT_AREA`). A keypoint at distance d from its ground truth scores less on a small object than on a large one, and less with a small sigma.

**Example: the overlaps of a test image of 3 objects, from the checkpoint of iteration 60**
```text
group_image_objects -> 30 images
image position 1  labels [20, 13, 11]  scores [0.042, 0.039, 0.075]
keypoint_overlaps
 tensor([[0.28, 0.00, 0.00],
         [0.00, 0.01, 0.00],
         [0.00, 0.00, 0.15]], dtype=torch.float64)
ground truth boxes [[37.0, 164.0, 118.0, 295.0], [361.0, 272.0, 476.0, 439.0], [299.0, 243.0, 378.0, 266.0]]
```

**Example: a toy object of area 10000 with two keypoints of sigma 0.05** (2 × (2σ)² × area = 200)

| Distance of the first keypoint | OKS, second keypoint exact | OKS, second keypoint unlabelled |
|---|---|---|
| 0 | 1.0 | 1.0 |
| 5 | 0.9412 | 0.8825 |
| 10 | 0.8033 | 0.6065 |
| 20 | 0.5677 | 0.1353 |

At 20 pixels the first keypoint scores exp(−400 / 200) = 0.1353; averaged with an exact second keypoint, the object scores 0.5677. With the threshold of OKS 0.5 of mAP@0.5, this object would be found with the second keypoint and missed without it.

### Step 4.7 — `measure_keypoint_accuracy()` + `summarise_keypoint_accuracy()`
`measure_keypoint_accuracy()` gives the distance of every keypoint to its ground truth (in resized image pixels), whether it is labelled, whether it is correct (within `CORRECT_KEYPOINT_FRACTION` = 0.05 of the longest side of the box) and the OKS of every object. `summarise_keypoint_accuracy()` averages them over all the labelled keypoints (PCK@0.05 and mean error) and all the objects (mean OKS), and per class and per keypoint of every class.

**Example: the test split at iteration 60**
```text
distances (82, 11) float64   labelled (82, 11) bool   correct (82, 11) bool   similarities (82,) float64
object 0: label 16 (ellipse)  box [228.0, 401.0, 391.0, 462.0]
  distances [1.08, 42.05, 155.54]   labelled [True, True, True]   correct [True, False, False]   similarity 0.3314
PCK 0.1164   mean OKS 0.2799   mean error 35.8753   labelled keypoints 378
```
The longest side of the ellipse box is 163 pixels, so a keypoint is correct within 8.15 pixels: only the first one (1.08 pixels) is.

### Step 4.8 — `summarise_average_precisions(overlap_type="keypoints")`
The average precision of `object_detection_workflow.md` on the OKS overlaps instead of the box IoU, with `title_prefix="OKS"`, `name_prefix="OKS "` and `column_prefix="oks_"`: the three methods (area, 11 point, 101 point) at OKS 0.5, 0.75 and 0.5 to 0.95, the COCO object sizes by box area, the mean average recall and the per-class APs. Since every object has exactly one prediction, the precision-recall curve of a class only depends on the order of the object scores: the AP is high when the objects predicted best (OKS above the threshold) also have the highest mean keypoint scores.

### Step 4.9 — `write_class_metrics_file()`
The class rows of the average precisions and of the keypoint accuracy are merged into one row per class (a `ValueError` is raised if the two do not list the same classes in the same order) and saved to `class_metrics.csv`; the keypoint rows are saved to `keypoint_metrics.csv` (`KEYPOINT_METRICS_FILE_NAME`), whatever the per-class report.

**Example: `Evaluation/Iteration_60/class_metrics.csv`** (23 lines)
```text
class_index,class,ground_truths,oks_average_precision_50,oks_average_precision_75,oks_average_precision_50_95,oks_average_precision_50_small,oks_average_precision_50_95_small,oks_average_precision_50_medium,oks_average_precision_50_95_medium,oks_average_precision_50_large,oks_average_precision_50_95_large,objects,labelled_keypoints,percentage_correct_keypoints,mean_object_keypoint_similarity,mean_error_pixels
1,square,6,0.166667,0.000000,0.041667,,,,,0.166667,0.041667,6,30,0.133333,0.409016,32.406266
2,circle,4,0.000000,0.000000,0.000000,,,0.000000,0.000000,0.000000,0.000000,4,4,0.000000,0.100797,22.342366
```
The six squares are all large, so their small and medium columns are empty.

**Example: `Evaluation/Iteration_60/keypoint_metrics.csv`** (108 lines: the 107 keypoints)
```text
class_index,class,keypoint,labelled,percentage_correct_keypoints,mean_error_pixels
1,square,top_left,6,0.000000,38.107601
1,square,top_right,6,0.333333,38.796879
```

### Step 4.10 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
The headline metric is `KEYPOINT_HEADLINE_METRIC` = `OKS mAP@0.5/area`, which compares the checkpoints. The keypoint accuracy is given first, then the tables of the methods, sizes and recalls, then, with the per-class report, the per-class APs, the accuracy per class and the accuracy per keypoint.

**Example output: the report of the checkpoint of iteration 60** (the per-class AP table and the per-keypoint table shortened)
```text
Iteration 60 · 82 Test Objects · 2026-10-17 10:02 ──────────────────────────────────────────────────────────────────────
  PCK@0.05                       0.1164    Mean OKS   0.2799
  Mean Keypoint Error (pixels)   35.8753
  Sample Images : 22 fixed test objects

  OKS Mean Average Precision
   Method      mAP@0.5   mAP@0.75   mAP@[0.5:0.95]
  ─────────────────────────────────────────────────
   Area         0.1689     0.0545           0.0646
   11 Point     0.1711     0.0579           0.0678
   101 Point    0.1693     0.0549           0.0650
  The mAP@0.5 of the area method compares the checkpoints of the run.

  OKS Mean Average Precision By Object Size (Area)
   Object Size           Objects   mAP@0.5   mAP@0.75   mAP@[0.5:0.95]
  ─────────────────────────────────────────────────────────────────────
   Small (< 32²)               0         —          —                —
   Medium (32² to 96²)        45    0.1958     0.0952           0.0815
   Large (>= 96²)             37    0.1207     0.0111           0.0434
  COCO sizes by box area in network input pixels, the objects of the other sizes and the detections finding them being
  ignored. — : no object of the size.

  OKS Mean Average Recall
   Objects Of            Objects   mAR@0.5   mAR@0.75   mAR@[0.5:0.95]
  ─────────────────────────────────────────────────────────────────────
   All                        82    0.2213     0.0636           0.0820
   Small (< 32²)               0         —          —                —
   Medium (32² to 96²)        45    0.2063     0.0952           0.0825
   Large (>= 96²)             37    0.1933     0.0222           0.0690
  Objects found by any detection kept for the evaluation, averaged over the classes. COCO sizes by box area in network
  input pixels. — : no object of the size.

  OKS Per Class Average Precision (Area)
                   Ground
   Class           Truths   AP@0.5   AP@0.75   AP@[0.5:0.95]
  ───────────────────────────────────────────────────────────
   circle               4   0.0000    0.0000          0.0000
   rectangle            2   0.0000    0.0000          0.0000
   ...                                                              (11 more classes at 0)
   ellipse              4   0.0625    0.0000          0.0063
   plus_sign            3   0.1111    0.0000          0.0222
   octagon              7   0.1429    0.0000          0.0179
   square               6   0.1667    0.0000          0.0417
   annulus              5   0.1800    0.1000          0.1320
   star                 5   0.3867    0.1000          0.1353
   hexagon              3   0.6667    0.0000          0.1667
   triangle             1   1.0000    1.0000          0.8000
   pentagon             1   1.0000    0.0000          0.1000
  Worst AP@0.5 first, ties in class order. — : no ground truth box of the class.

  Keypoint Accuracy Per Class
                              Labelled                Mean   Mean Error
   Class           Objects   Keypoints   PCK@0.05      OKS     (pixels)
  ──────────────────────────────────────────────────────────────────────
   circle                4           4     0.0000   0.1008      22.3424
   rectangle             2           8     0.0000   0.1985      28.4957
   pentagon              1           5     0.0000   0.5420       8.2864
   rhombus               2           7     0.0000   0.1246      18.3974
   parallelogram         3          10     0.0000   0.1168      15.5661
   trapezoid             3          12     0.0000   0.0320      63.4974
   semicircle            6          23     0.0000   0.0699      57.6272
   stick_figure          2          22     0.0000   0.1176      37.9541
   house                 3          13     0.0000   0.2724      32.0175
   kite                  5          19     0.0526   0.2053      54.2768
   nonagon               4          36     0.1111   0.2763      29.8331
   square                6          30     0.1333   0.4090      32.4063
   plus_sign             3          15     0.1333   0.3655      25.1290
   l_shape              10          56     0.1429   0.1816      54.4582
   octagon               7          46     0.1522   0.4042      28.2822
   annulus               5           5     0.2000   0.3554      19.4235
   capsule               2           5     0.2000   0.2951       9.7276
   star                  5          23     0.2174   0.5485      24.6696
   hexagon               3          18     0.2222   0.5590      17.7082
   ellipse               4          11     0.2727   0.2943      41.8241
   heptagon              1           7     0.2857   0.2829      39.0260
   triangle              1           3     0.6667   0.8702       2.9547
  Worst PCK@0.05 first. PCK@0.05 : labelled keypoints within 0.05 of the longest box side, errors in network input
  pixels. — : no labelled keypoint.

  Keypoint Accuracy Per Keypoint
                                                           Mean Error
   Class                  Keypoint   Labelled   PCK@0.05     (pixels)
  ────────────────────────────────────────────────────────────────────
   square                 top_left          6     0.0000      38.1076
   square             bottom_right          6     0.0000      28.8128
   square                   centre          6     0.0000      27.2982
   circle                   centre          4     0.0000      22.3424
   triangle                    top          1     0.0000       6.5048
   ...
   l_shape            outer_corner          8     0.6250      22.9946
   triangle            bottom_left          1     1.0000       1.1589
   triangle           bottom_right          1     1.0000       1.2004
   heptagon                 left_3          1     1.0000       1.4032
   heptagon                 left_2          1     1.0000       0.6737
   ellipse                left_end          3     1.0000       1.5483
   capsule                left_end          1     1.0000       1.8250
  Worst PCK@0.05 first, errors in network input pixels. — : keypoint never labelled.
  - Results : Results.txt
  - Class Metrics : class_metrics.csv
  - Keypoint Metrics : keypoint_metrics.csv
  - Sample Images : Samples
```
The pentagon has a mean OKS of 0.5420 but a PCK of 0: its 5 keypoints are all about 8 pixels off, beyond 0.05 of its box, but close enough for the OKS. The single triangle, the best predicted object, has an AP of 1 at every threshold.

**Example: the four evaluations of the run**

| Checkpoint | PCK@0.05 | Mean OKS | Mean Error (pixels) | OKS mAP@0.5 (area) | OKS mAP@0.75 | OKS mAP@[0.5:0.95] |
|---|---|---|---|---|---|---|
| Iteration 30 | 0.0503 | 0.1778 | 42.8906 | 0.0290 | 0.0000 | 0.0057 |
| Iteration 30, EMA Weights | 0.1005 | 0.2157 | 45.6810 | 0.1076 | 0.0424 | 0.0511 |
| Iteration 60 | 0.1164 | 0.2799 | 35.8753 | 0.1689 | 0.0545 | 0.0646 |
| Iteration 60, EMA Weights | 0.1217 | 0.2810 | 31.8738 | 0.0997 | 0.0121 | 0.0311 |

**Example: the run folder after the training**
```text
<run>/
├── Evaluation/Iteration_30/, Iteration_60/
│   ├── class_metrics.csv, keypoint_metrics.csv
│   ├── Samples/                       22 PNG images
│   └── EMA/                           the same files for the averaged weights
├── Tensorboard/Train/, Validation/, Test/, Test_EMA/
├── Weights/Iteration_15/ ... Iteration_60/, full-checkpoint
├── Results.txt                        808 lines
└── data.json, mlflow_run.json, training_configuration.yaml
```

---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of a copy of `scripts/evaluation/evaluate_checkpoint.py` whose `experiment_folder` points to the run of Stage 3 (checkpoints 15, 30, 45 and 60), with `iteration = "all"`, `batch_size = 32` and the whole test split. The four checkpoints, each with its averaged weights, were evaluated in 33 s.

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The script of every task (see `classification_workflow.md`, Stage 5): it loads the copied `training_configuration.yaml`, points the run folder to `<experiments>/evaluations`, sets the batch size, calls `disable_pretrained_settings()`, builds the trainer with `get_trainer_class()`, lists the checkpoints with `get_checkpoint_iterations()` (the last saved first) and calls `evaluate_checkpoints()`. With a batch of 32 objects, the test split is 3 batches and an epoch 28 iterations.

### Step 5.2 — `evaluate_checkpoints()` → `start_evaluation_tracking()` → `run_test_evaluation()` for each checkpoint
**Example output**
```text
MLflow Run Created In The Experiment 'keypoint_detection' : 1f7d036030c04a34bf671ac655522cdc (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
-----------------------------
Evaluating : model_000060.pt
-----------------------------
Test Evaluation Iteration 60: 100%|██████████| 3/3
Iteration 60 · 82 Test Objects · 2026-10-17 10:03 ──────────────────────────────────────────────────────────────────────
```
The reports are those of Stage 4 (the same PCK, mean OKS and mAP at iterations 30 and 60), the batch size changing nothing.

### Step 5.3 — `summarise_checkpoint_evaluations()`
**Example output**
```text
-------------------------------------------------------------
Evaluation Summary Of 4 Checkpoints : evaluation_summary.csv
-------------------------------------------------------------
  - Best Checkpoint : iteration 60 (OKS mAP@0.5/area 0.1689)
  - Best Checkpoint (EMA Weights) : iteration 30 (OKS mAP@0.5/area 0.1076)
```

**Example: `runs/evaluations/<run>/Evaluation/evaluation_summary.csv`** (95 columns: iteration, the 9 OKS mAPs of the methods, the 4 OKS mAPs of the medium and large objects, the 9 OKS mARs of all, medium and large objects, the 3 keypoint accuracy metrics, the 21 OKS AP@0.5 of the classes, then the same 47 columns prefixed by `ema_`)
```text
iteration   OKS mAP@0.5/area   Keypoint Accuracy/PCK@0.05   Keypoint Accuracy/Mean OKS   Keypoint Accuracy/Mean Keypoint Error (pixels)   ema_OKS mAP@0.5/area
   60       0.168930           0.116402                     0.279934                     35.875272                                        0.099686
   45       0.139448           0.132275                     0.284951                     29.006632                                        0.048295
   30       0.028968           0.050265                     0.177844                     42.890601                                        0.107576
   15       0.002273           0.031746                     0.098340                     58.518826                                        0.016364
```
The checkpoint of iteration 45 has a better PCK, mean OKS and mean error than the one of iteration 60, but the checkpoints are ranked on the OKS mAP@0.5, which also depends on how well the object scores order the objects (Step 4.8).

**Example: the evaluation folder**
```text
runs/evaluations/<run>/
├── Evaluation/evaluation_summary.csv
├── Evaluation/Iteration_15/, Iteration_30/, Iteration_45/, Iteration_60/      the files of Step 4.9, with Samples/ and EMA/
├── Tensorboard/Test/, Test_EMA/
├── Weights/                                                                    empty
├── Results.txt
└── data.json, training_configuration.yaml
```

---

## Stage 6 — Export (Optional)

> The examples below are real outputs of a script repeating the steps of `export_experiment()` on CPU, on the checkpoint of iteration 60 of the run of Stage 3 (trained weights), with a maximum batch size of 16 and the TensorRT precision FP32, the exported models being run by TorchScript and ONNX Runtime on CPU. Triton itself was not started.

### Step 6.1 — `vision_training/deployment/export_model.py` → `export_experiment()`
Run as a script after the training (the experiment folder, iteration, formats, Triton model repository, model version, maximum batch size and TensorRT precision set in its `__main__` block). `export_experiment()` turns TF32 off, loads the checkpoint on CUDA with `load_trained_model()` (the device is hard-coded, which is why this stage was run by a script repeating its steps on CPU), wraps it with `build_inference_model()`, builds the export specification, exports every format to `<repository>/<model name>_<format>/<version>/`, writes its `config.pbtxt` and compares the exported model to the eager one, raising a `RuntimeError` on a mismatch.

### Step 6.2 — `load_trained_model()` → `build_inference_model()` → `InferenceKeypointDetector`
`InferenceKeypointDetector` takes the crops only and returns, for every crop, the decoded keypoints of the 107 channels in crop pixels, their scores and the heatmaps. The class of the object is not an input: the client knows it (from the detector that gave the box) and keeps the channels of that class (Step 6.6).

**Example output**
```text
loaded KeypointDetector training mode False
inference model InferenceKeypointDetector   crop 192 192   channels 107
```

### Step 6.3 — `get_export_specification()` → `create_random_inputs()`
The keypoint branch of `get_export_specification()` gives the model name `keypoint_detector`, one input `crops` of 3x192x192, the three outputs and the comparison function `compare_keypoints()` with its tolerances. `create_random_inputs()` draws a batch of 2 crops to trace the model (`example_shape`) and a batch of 3 to check it (`check_shape`), so that a batch size frozen by the tracing would be caught.

**Example output**
```text
model_name keypoint_detector
inputs  [{'name': 'crops', 'dimensions': [3, 192, 192], 'data_type': 'TYPE_FP32', 'dynamic_frames': False, 'maximum_frames': 1,
          'example_shape': [2, 3, 192, 192], 'check_shape': [3, 3, 192, 192]}]
outputs [{'name': 'keypoints', 'dimensions': [107, 2], 'data_type': 'TYPE_FP32', 'reshape_to_scalar': False},
         {'name': 'keypoint_scores', 'dimensions': [107], 'data_type': 'TYPE_FP32', 'reshape_to_scalar': False},
         {'name': 'heatmaps', 'data_type': 'TYPE_FP32', 'reshape_to_scalar': False, 'dimensions': [107, 48, 48]}]
tolerances {'keypoint_tolerance': 0.01, 'score_tolerance': 0.0001, 'heatmap_tolerance': 0.0001, 'minimum_equal_share': 0.95}
example {'crops': (2, 3, 192, 192)}   check {'crops': (3, 3, 192, 192)}
reference outputs [((3, 107, 2), 'float32'), ((3, 107), 'float32'), ((3, 107, 48, 48), 'float32')]
```

### Step 6.4 — `export_torchscript()` / `export_onnx()` + `compare_keypoints()`
`export_torchscript()` traces the model with `torch.jit.trace`; `export_onnx()` exports it with the dynamo exporter (opset 18) and a dynamic `batch` dimension. Both are run on the check inputs and compared to the eager outputs by `compare_keypoints()`, which allows a keypoint difference of `keypoint_tolerance` pixels on at least `minimum_equal_share` of the keypoints (a near tie between two heatmap cells can move a decoded keypoint by a whole cell), and checks the scores and heatmaps within their tolerances.

**Example output**
```text
torchscript: model.pt    12,218,628 bytes   mismatches []   largest differences keypoints/scores/heatmaps [0.0, 0.0, 0.0]
onnx:        model.onnx   3,009,109 bytes   mismatches []   largest differences keypoints/scores/heatmaps [0.0, 1.64e-07, 3.28e-07]
```
PyTorch warns that `torch.jit.trace`, `torch.jit.trace_method` and `torch.jit.load` are deprecated (FutureWarning); the export works.

### Step 6.5 — `write_triton_configuration()`
**Example: the model repository**
```text
triton_repository/
├── keypoint_detector_onnx/config.pbtxt
├── keypoint_detector_onnx/1/model.onnx
├── keypoint_detector_onnx/1/model.onnx.data          the weights, written next to the graph by the ONNX exporter
├── keypoint_detector_torchscript/config.pbtxt
└── keypoint_detector_torchscript/1/model.pt
```

**Example: `keypoint_detector_onnx/config.pbtxt`**
```text
name: "keypoint_detector_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 16
input [
  { name: "crops" data_type: TYPE_FP32 dims: [ 3, 192, 192 ] }
]
output [
  { name: "keypoints" data_type: TYPE_FP32 dims: [ 107, 2 ] },
  { name: "keypoint_scores" data_type: TYPE_FP32 dims: [ 107 ] },
  { name: "heatmaps" data_type: TYPE_FP32 dims: [ 107, 48, 48 ] }
]
dynamic_batching { max_queue_delay_microseconds: 1000 }
instance_group [ { kind: KIND_GPU count: 1 } ]
version_policy: { specific: { versions: [ 1 ] } }
optimization { execution_accelerators { gpu_execution_accelerator: [ {
  name: "tensorrt"
  parameters { key: "precision_mode" value: "FP32" }
  parameters { key: "max_workspace_size_bytes" value: "4294967296" }
  parameters { key: "trt_engine_cache_enable" value: "1" }
  parameters { key: "trt_engine_cache_path" value: "/tmp/tensorrt_engine_cache" }
} ] } }
```
The TorchScript configuration is the same with `platform: "pytorch_libtorch"`, the outputs named `keypoints__0`, `keypoint_scores__1` and `heatmaps__2`, and no optimization block.

### Step 6.6 — the client: `cut_inference_crop()`, the channels of the class and the way back to the image
A client cuts the crop of every object with `cut_inference_crop()` (the same region and enlargement as `compute_crop_regions()`, from the `crop_settings` of the run), sends the crops, keeps the channels of the class of every object (its row of the channel table) and moves the keypoints back into the image with the inverse affine of the crop. `load_inference_inputs()`, used by `deployment/run_inference.py` to measure the throughput, takes the whole image as the box of one object. The example below was run on the first test object with the exported ONNX model in ONNX Runtime.

**Example: an l_shape (box [416, 195, 497, 296]) through the ONNX model**
```text
crop (1, 3, 192, 192) float32
outputs (1, 107, 2) (1, 107) (1, 107, 48, 48)
channels of the class [85, 86, 87, 88, 89, 90]  ['l_shape.arm_top_outer', 'l_shape.arm_top_inner', 'l_shape.inner_corner',
                                                 'l_shape.foot_end_top', 'l_shape.foot_end_bottom', 'l_shape.outer_corner']
crop keypoints  [[159.0, 125.0], [157.0, 125.0], [35.0, 29.0], [53.0, 21.0], [43.0, 19.0], [139.0, 167.0]]
image keypoints [[497.9, 264.6], [496.6, 264.6], [416.4, 201.4], [428.2, 196.2], [421.6, 194.9], [484.8, 292.2]]
scores          [0.0574, 0.0676, 0.0423, 0.0389, 0.052, 0.0414]
ground truth    [[416.5, 195.5], [449.5, 195.5], [449.5, 263.5], [496.5, 263.5], [496.5, 295.5], [416.5, 295.5]]
```
After 60 iterations the keypoints land near corners of the shape, but on the wrong ones: the predicted `arm_top_outer` (497.9, 264.6) is at the ground truth `foot_end_top` (496.5, 263.5).
