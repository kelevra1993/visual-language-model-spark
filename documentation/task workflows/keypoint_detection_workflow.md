# Keypoint Detection Workflow

This document follows one keypoint detection run (KPT) from `main()` to the evaluation and export of a checkpoint. It shows which function runs when, what it receives and what it returns. The **Call Tree Overview** gives the whole run on one page; each stage then has one step per function, with a **real** output. Steps shared by every task are explained in `classification_workflow.md`, the mean average precision machinery in `object_detection_workflow.md`; here they keep one line and the real output. The steps specific to keypoints are explained in full.

The model is a bottom-up keypoint detector (HigherHRNet): heatmaps give the keypoint candidates, and tags group them into objects, the keypoints of one object sharing a tag. No box is needed.

The examples come from CPU runs (`CUDA_VISIBLE_DEVICES=` empty). The dataset `synthetic_shapes` was generated in a temporary folder with `create_synthetic_dataset(task="keypoint_detection")`: 300 training, 30 validation and 30 test images of 512 pixels, 2 to 5 objects per image, 37 categories. 33 have keypoints (a square its 4 corners, a triangle 3, a stick figure 11, a hollow nonagon 18), 243 in all. `circle`, `ellipse`, `annulus` and `hollow_ellipse` are distractors without keypoints, left out of the `label_mapping.json` of `build_keypoint_label_mapping()`, which numbers the 33 others from 1. The dataset of the configuration was generated again on 2026-11-18 with textured backgrounds, low contrast, overlapping shapes from 32 pixels (`KEYPOINT_OBJECT_MINIMUM_SIZE`) cut by the border, a keypoint outside the image being unlabelled.

The table below gives the configuration behind each stage (full configuration values in brackets).

| Stages | Configuration |
|---|---|
| 1 and 2 | `vision_training/configurations/keypoint_detection/configuration_synthetic_shapes.yaml`: 512 pixel images, a 3-block stem ending at 1/4, an HRNet of base width 16 with 1, 2 and 2 modules in its stages of 2, 3 and 4 branches, hidden width 256, the higher resolution branch, heatmap sigma 2 cells, tag weights 0.1; iterations of Stage 3 |
| 3, 4 and 5 | the same with batch size 8 (16), 60 iterations (20000), a checkpoint and test evaluation every 30 (2000), an intermediate checkpoint every 15 (1000), a summary every 10 (500), a validation batch every 10 (0, none), warm-up 10 and decay 50 (500 and 19500), 2 workers (now fixed at 1), MLflow on (off) |
| 6 | the checkpoint of iteration 60 exported to TorchScript and ONNX on CPU by a script repeating `export_experiment()`, which only runs on CUDA (Step 6.1) |

The augmentation types were drawn at the probabilities of that time (0.3 for most, 0.5 for the translation); every type is now drawn at 0.25 (flips, off here, at 0.5), and `random_haze` is on. After 60 iterations the detector is far from trained, so the metrics are low on purpose. The client of Step 6.6 and the test batch steps of Stage 4 use a GPU run of the same model trained on the night of 2026-10-17 (iterations 6000 and 2000).

## Read This First

Before choosing this task, know what it gives. One whole image in; every object of every class out, with its class, the position of every keypoint in network input pixels (NaN when not found), every keypoint score and the object score. Keypoint names and skeletons come from the COCO `categories` of the training `labels.json`, so classes can have different keypoints. The network predicts one heatmap and one tag map per keypoint of every class (243 channels for the 33 synthetic classes); `group_keypoint_candidates()` groups the candidates into objects on the CPU. The dataset is the COCO layout of DET with a `keypoints` list per annotation. The stem is the shared `Backbone`, so a backbone of another task can initialise it.

| Answer wanted | Task |
|---|---|
| named points per object (corners, joints, tips) | KPT |
| a box per object | DET |
| a mask per object | INS |
| a class per pixel, without objects | SEG |

### The Task At A Glance

The diagram below follows one batch of the synthetic configuration from the files to the evaluation folder.

```text
on disk        <split>/data/*.png, <split>/labels.json (COCO: keypoints [x, y, v] per object, keypoint names and
               skeleton per category), label_mapping.json {category id: class 1..33}, 4 distractors left out
sample         CocoDetectionDataset: image [3, 512, 512] uint8, boxes [N, 4], labels [N], keypoints [N, 18, 3]
batch (16)     images [16, 3, 512, 512]; boxes, labels, keypoints: lists of 16; augment_batch moves the keypoints
stem           Backbone, 3 blocks: [16, 3, 512, 512] -> [16, 64, 128, 128] (stride 4)
HRNet          3 stages, branches [16, 16, 128, 128] [16, 32, 64, 64] [16, 64, 32, 32] [16, 128, 16, 16], exchanged
               after every module -> 1/4 branch [16, 16, 128, 128]
head           3x3 16 -> 256, 1x1 256 -> 486: heatmap logits [16, 243, 128, 128], tags [16, 243, 128, 128]
higher res.    cat [16, 502, 128, 128] -> 4x4 transposed conv -> [16, 16, 256, 256] -> 4 conv -> 3x3 -> 1x1
               -> heatmap logits [16, 243, 256, 256]
targets        Gaussians of sigma 2 cells at strides 4 and 2 (keypoint cell = 1), tag_indices [16, M, 18]
losses         focal heatmap loss (both strides summed) + 0.1 x tag pull + 0.1 x tag push = total loss
inference      heatmaps averaged at stride 2 -> 30 local maxima per channel: candidate_positions [B, 243, 30, 2],
               candidate_scores [B, 243, 30], candidate_tags [B, 243, 30] (the exported outputs)
grouping       group_keypoint_candidates (NumPy): labels [D], keypoints [D, 18, 2] (NaN not found), scores [D]
evaluation     OKS of every object with every ground truth: OKS mAP@0.5/area (headline), OKS mAR, Matched
               Objects, PCK@0.05, Mean OKS, Mean Keypoint Error, for the trained and the averaged weights; per
               class, the OKS recall at 0.5 and 0.75 with its 95 % interval
files          Evaluation/Iteration_<n>/ class_metrics.csv, keypoint_metrics.csv, Samples/, EMA/; Results.txt
```

### Files To Know

To find your way in the code, the table below lists the file and main functions of every role.

| Role | File | Main functions or classes |
|---|---|---|
| Trainer | `vision_training/trainers/keypoint_trainer.py`, `vision_training/trainers/coco_split_trainer.py` | `KeypointTrainer`: `prepare_batch`, `augment_batch`, `forward_pass`, `predict_batch`, `accumulate_evaluation_batch`, `compute_evaluation_results`, `build_sample_image`; `CocoSplitTrainer.read_test_manifest` |
| Keypoint definitions | `vision_training/data/keypoints/keypoint_categories.py`, `vision_training/configuration_processing/configuration_loader.py` | `read_keypoint_categories`, `inject_keypoint_categories` |
| Dataset and loaders | `vision_training/data/detection/coco_dataset.py`, `vision_training/data/loaders.py` | `CocoDetectionDataset.add_keypoints`, `collate_function`, `get_keypoint_dataloaders` |
| Model | `vision_training/models/keypoints/keypoint_detector.py`, `vision_training/models/keypoints/hrnet.py` | `KeypointDetector`, `build_keypoint_channel_table`, `extract_keypoint_candidates`, `compute_peak_offsets`; `HighResolutionNetwork`, `BranchTransition`, `ExchangeModule`, `HigherResolutionBranch`, `build_prediction_head` |
| Targets | `vision_training/data/keypoints/heatmap_targets.py` | `build_keypoint_targets`, `draw_gaussian_heatmaps` |
| Losses | `vision_training/losses/keypoint_losses.py` | `compute_heatmap_loss`, `compute_tag_losses` |
| Grouping | `vision_training/models/keypoints/keypoint_grouping.py` | `group_keypoint_candidates`, `match_candidates_to_objects` |
| Metrics and evaluation | `vision_training/evaluation/keypoints/object_keypoint_similarity.py`, `vision_training/evaluation/detection/average_precision_report.py` | `build_keypoint_sigma_table`, `evaluate_image_keypoints`, `compute_squared_extended_box_distances`, `match_ground_truth_objects`, `summarise_keypoint_accuracy`; `summarise_average_precisions(overlap_type="keypoints")`, with `compute_clopper_pearson_intervals` of `vision_training/evaluation/recall_intervals.py` |
| Sample images and input viewer | `vision_training/utilities/visualization.py` | `build_keypoint_sample_grid`, `draw_keypoint_objects` |
| Configuration checks | `vision_training/configuration_processing/keypoint_checks.py` | `check_keypoint_configuration` |
| Run name | `vision_training/configuration_processing/run_name.py` | `build_run_name`, `build_keypoint_sections` |
| Synthetic data generator | `vision_training/data/synthetic/create_synthetic_dataset.py` | `create_synthetic_dataset`, `build_keypoint_labels`, `build_keypoint_label_mapping`, `KEYPOINT_CATEGORIES`, `KEYPOINT_NAMES` |
| Export | `vision_training/models/keypoints/inference_keypoint_detector.py`, `vision_training/deployment/export_specifications.py`, `vision_training/deployment/input_contract.py` | `InferenceKeypointDetector` (model name `keypoint_detector`, candidates only), `get_export_specification`, `compare_keypoints`; `build_input_contract`, `write_onnx_input_contract`, `write_preprocessing_file` |
| Export sizes at launch | `vision_training/deployment/export_size_estimates.py` | `estimate_export_sizes`, `probe_inference_graph`, `print_export_size_summary` |

### Run It

If one wishes to run the whole task, generate the data, train, evaluate, then export.

```bash
# Generate 250,000 / 50,000 / 50,000 images of 512 pixels (add --split_sizes train=300 validation=30 test=30 for a first try)
uv run python -m vision_training.data.synthetic.create_synthetic_dataset --task=keypoint_detection --output_folder="$HOME/Data/keypoint_data/synthetic_shapes"

# Train, after setting experiment_folder, the three data folders and Data.label_mapping.mapping_file_path
uv run python -m vision_training.main --configuration_path="vision_training/configurations/keypoint_detection/configuration_synthetic_shapes.yaml"

# Evaluate every checkpoint into <run>/Checkpoint_Evaluation/, error images off since the task draws none (Step 5.1)
uv run python -m scripts.evaluation.evaluate_checkpoint --experiment_folder <run folder> --iteration all --maximum_error_images_per_class_pair None

# Export on CUDA only (Step 6.1); --tensorrt_precision None, since Triton crashes on the default FP32 TensorRT configuration (Step 6.5)
uv run python -m vision_training.deployment.export_model --experiment_folder <run folder> --iteration <iteration> --tensorrt_precision None
```

### Configuration Keys To Know First

Before editing a configuration, the table below gives the keys that change this task most.

| Key | Value in the synthetic configuration | Effect |
|---|---|---|
| `KeypointHead.heatmap_sigma` | `2.0` | standard deviation of the target Gaussians in cells at strides 4 and 2, truncated at ceil(3 sigma) = 6 cells |
| `KeypointHead.tag_pull_weight`, `KeypointHead.tag_push_weight` | `0.1`, `0.1` | weights of the two tag losses |
| `KeypointHead.hidden_width` | `256` | 3x3 convolution before the 1x1 predictions (`KH256`); 0 for HigherHRNet heads |
| `KeypointHead.higher_resolution` | `True` | second heatmaps at 1/2 (`HH`), averaged with the 1/4 ones at inference |
| `KeypointHead.inference` | `maximum_candidates: 30`, `keypoint_threshold: 0.1`, `tag_threshold: 1.0`, `object_score_threshold: 0.2` | candidates per channel and grouping thresholds; the evaluation keeps every object, sample images and clients those scoring at least 0.2 |
| `KeypointHead.ignore_unlabelled_objects` | `False` | leaves cells at least half covered by objects without labelled keypoint out of the heatmap loss, from their `segmentation` (True in the COCO configuration) |
| `HRNet.base_width`, `HRNet.modules_per_stage`, `HRNet.blocks_per_branch` | `16`, `[1, 2, 2]`, `2` | branches of 16, 32, 64 and 128 channels (`HR16.1.2.2x2`); the image size must be a multiple of 32 |
| `Backbone.convolutions`, `Backbone.last_max_pooling` | `[1, 16]`, `[2, 32]`, `[2, 64]`, `False` | the stem must end at 1/4: 3 blocks without the last pooling, or 2 with it |
| `ExperimentConfiguration.evaluation.default_keypoint_sigma`, `keypoint_sigmas` | `0.05`, 8 sigmas of `stick_figure` | the OKS tolerance of every keypoint (the COCO constant is 2 sigma) |
| `Data.data_augmentation.types` | flips and `random_90_rotation` off, `random_rotation` on (5 degrees) | geometric types move the keypoints without renaming them; `random_perspective` and `random_jigsaw` are refused |

### What A Run Writes

Whenever a checkpoint is evaluated, the run folder, TensorBoard and MLflow receive the following.

| Output | Content |
|---|---|
| `Results.txt` | one block per checkpoint and kind of weights: `Matched Objects`, `PCK@0.05`, `Mean OKS`, `Mean Keypoint Error (pixels)`, the OKS mAP tables; with `per_class_report` the OKS AP per class, `OKS Per Class Recall` with its 95 % interval, the keypoint accuracy per class and per keypoint (Step 4.10). Rates as percentages (`94.17 %`); files, TensorBoard and MLflow keep 0 to 1 |
| `Evaluation/Iteration_<n>/` | `class_metrics.csv` (per class: `oks_` average precisions, `oks_recall_50` to `oks_recall_75_upper`, keypoint accuracy), `keypoint_metrics.csv` (per keypoint: labelled, found, PCK, mean error), `Samples/<image>.png` (Ground Truth next to Prediction, 1028x560), `EMA/` with the same for the averaged weights. No confusion matrix, no error images |
| `Tensorboard/Test`, `Tensorboard/Test_EMA` | the OKS mAPs (overall and per size, none for a size without objects), mARs, keypoint accuracy and per-class AP@0.5 (71 scalars with the three COCO sizes of the examples, 65 without small objects), and 16 sample images |
| `Tensorboard/Train` (`Validation` only with `validation_iteration_ratio` above 0) | `Total Loss`, `Heatmap Loss`, `Tag Pull Loss`, `Tag Push Loss` (and `Learning Rate`) |
| `Weights/Iteration_<n>/model_<n>.pt` | about 51 MB, four times the 12.50 MB of weights (averaged weights and two Adam moments), with the source runs of `Backbone`, `HRNet` and `KeypointHead` |

So that you know what a trained model reaches: the tutorial run of `keypoint_focal_synthetic/experiments` (from scratch, 2026-11-19, dataset of 2026-11-18, 6000 iterations, cosine decay over 5500, first 1024 test images: 3,174 objects with keypoints, 264 small) gives at iteration 6000 `OKS mAP@0.5/area` 96.58 % (96.29 % averaged weights), `OKS mAP@0.95/area` 72.79 % (72.89 %), `Matched Objects` 97.32 %, `PCK@0.05` 93.47 %, `Mean OKS` 92.99 % and a mean keypoint error of 4.7397 pixels. Iteration 2000 gives 92.24 % and 54.63 % for the two mAPs.

### Pitfalls

In case a result surprises you, check these first.
- **A flip or a large rotation never renames a keypoint**: after a left-right flip, `left_hand` lies on the right and is still trained as `left_hand`. These types only suit names independent of side and orientation ([tasks reference, Keypoint Detection](../reference/tasks_reference.md#keypoint-detection-kpt), Augmentation).
- **`residual_nets: True` adds no shortcut in the stem**: its three blocks all change the channel count (Step 2.11). Only the HRNet branches and the higher resolution refinement have residual pairs.
- **A keypoint head transferred with its channel layers** (`KeypointHead.pretrained.class_layers: True`) is checked on the class and keypoint name of every channel, which the source `data.json` records since 2026-11-20 (`compare_keypoint_channels()` of `model_transfer.py`). An older keypoint run can only initialise a head without them. The export writes no channel names: a client rebuilds the channel table from the training `labels.json`.
- **`Matched Objects` and `Mean Keypoint Error (pixels)` need an OKS of at least 0.5** (`MINIMUM_MATCH_SIMILARITY`). Before 2026-10-25 any OKS above 0 matched, so older evaluations can give more matches and larger errors.
- **The export needs a GPU**: `export_experiment()` loads the checkpoint on CUDA (Step 6.1).
- **Triton crashes at the first request** with the default TensorRT `config.pbtxt`, while ONNX Runtime without TensorRT serves the model: export with `--tensorrt_precision None` (Step 6.5).

## Table of Contents
- [Read This First](#read-this-first)
- [Call Tree Overview](#call-tree-overview)
- [Stage 1 — Entry Point and Configuration Loading](#stage-1--entry-point-and-configuration-loading)
- [Stage 2 — Trainer Initialization](#stage-2--trainer-initialization)
- [Stage 3 — Training Loop](#stage-3--training-loop)
- [Stage 4 — Test Evaluation of a Checkpoint](#stage-4--test-evaluation-of-a-checkpoint)
- [Stage 5 — Standalone Checkpoint Evaluation (Optional)](#stage-5--standalone-checkpoint-evaluation-optional)
- [Stage 6 — Export (Optional)](#stage-6--export-optional)

---

## Call Tree Overview

So that the whole run fits on one page, the tree below lists the functions specific to this task in call order, with the stage that details them.

```text
main()  --configuration_path to a keypoint_detection configuration
├── load_experiment_configuration()  [STAGE 1]
│   ├── build_label_dictionary()                               {category id: class index} + 'background': 0
│   ├── inject_keypoint_categories() -> read_keypoint_categories()   names and skeleton of every class
│   └── check_keypoint_configuration()                         stem at 1/4, image size, head, evaluation sigmas
├── KeypointTrainer(...)  [STAGE 2]
│   ├── build_keypoint_tables()                                channel_table [34, 18], sigma_table [34, 18]
│   ├── get_run_name() -> build_keypoint_sections()
│   ├── get_keypoint_dataloaders() -> CocoDetectionDataset(load_keypoints=True)   one whole image per sample
│   ├── get_test_sample_classes() + choose_sample_identifiers()                  16 fixed test images
│   ├── KeypointDetector.__init__()   channel table, stem, HighResolutionNetwork, prediction head, higher resolution branch
│   └── initialise_pretrained_components() / print_summary() / estimate_export_sizes()
└── run_training_loop()  [STAGE 3]
    ├── get_next_batch() -> CocoDetectionDataset.__getitem__() -> add_keypoints()
    ├── augment_batch() -> DataAugmenter.augment_batch(keypoints=...)          keypoints moved, never renamed
    ├── KeypointDetector.forward()   stem -> HRNet -> heatmaps and tags at 1/4, heatmaps at 1/2
    │   ├── build_keypoint_targets() -> draw_gaussian_heatmaps()
    │   ├── compute_heatmap_loss()                             focal loss, summed over the strides
    │   └── compute_tag_losses()                               pull and push
    ├── backward / step_optimizer() + moving average of the weights
    └── run_test_evaluation()  [STAGE 4]   trained weights, then use_averaged_weights() for EMA
        ├── predict_batch() -> extract_keypoint_candidates() + group_keypoint_candidates()
        ├── save_sample_images() -> build_keypoint_sample_grid()
        ├── evaluate_image_keypoints() -> compute_squared_extended_box_distances() + match_ground_truth_objects()
        ├── summarise_average_precisions(overlap_type="keypoints") + summarise_keypoint_accuracy()
        └── write_class_metrics_file() + report_evaluation()                      headline 'OKS mAP@0.5/area'
scripts/evaluation/evaluate_checkpoint.py  [STAGE 5]   same as every task
vision_training/deployment/export_model.py  [STAGE 6]
└── export_experiment()   InferenceKeypointDetector -> TorchScript / ONNX + input contract -> config.pbtxt,
                          preprocessing.json -> compare_keypoints() -> install_staged_models()
client of the exported model  [STEP 6.6]   load_inference_inputs() -> exported model -> group_keypoint_candidates()
```

---

## Stage 1 — Entry Point and Configuration Loading

> Real outputs of the loader on `configuration_synthetic_shapes.yaml`, its data folders and mapping file redirected to the generated dataset and its iterations shortened as in Stage 3.

### Step 1.1 — `main()`
The entry point of every task (`classification_workflow.md`, Step 1.1). The run of Stage 3 was started with:
```text
CUDA_VISIBLE_DEVICES= PYTHONPATH=. COLUMNS=120 .venv/bin/python vision_training/main.py --configuration_path .../configuration_workflow.yaml
```

### Step 1.2 — `load_experiment_configuration()`
The model sections are `Optimization`, `Data`, `Backbone`, `HRNet` and `KeypointHead`. `Data.image_settings.size` is the side of the square network input. `Data.keypoint_categories` is added by `inject_keypoint_categories()` (Step 1.5). `HRNet` holds `base_width` (doubled at every lower branch), `modules_per_stage`, `blocks_per_branch`, `modules`, `normalization` (convolutions have a bias only with none) and `pretrained`. The `KeypointHead` keys are below.

| Key | Meaning |
|---|---|
| `higher_resolution` | the HigherHRNet branch, a second set of heatmaps at 1/2 (Step 2.8) |
| `hidden_width` | channels of a 3x3 convolution before the 1x1 predictions, in both heads (0 for none): a 1x1 convolution on 16 channels cannot raise each of 243 heatmaps on its own keypoints |
| `heatmap_sigma` | standard deviation of the target Gaussians, in cells, at both strides |
| `ignore_unlabelled_objects` | leaves the pixels of objects without labelled keypoint out of the heatmap loss (Steps 2.6, 3.6, 3.7), from their `segmentation`; False here (153 of 781,423 training objects), True for the COCO crowds and persons without keypoints |
| `tag_pull_weight`, `tag_push_weight` | weights of the two tag losses (Step 3.8) |
| `inference` | the grouping (Step 4.3): `maximum_candidates` per channel, `keypoint_threshold` (lowest heatmap score of a grouped candidate), `tag_threshold` (largest distance to the mean tag of the object), `object_score_threshold` (lowest object score kept by the deployment and the sample images) |

The `evaluation` section gives the OKS sigmas: `default_keypoint_sigma` (0.05), and `keypoint_sigmas` per class and keypoint.
```text
'keypoint_sigmas': {'stick_figure': {'left_elbow': 0.072, 'right_elbow': 0.072, 'left_hand': 0.062, 'right_hand': 0.062,
                                     'left_knee': 0.087, 'right_knee': 0.087, 'left_foot': 0.089, 'right_foot': 0.089}}
```

### Step 1.3 — `load_configuration()`
```text
top-level keys: ['ExperimentConfiguration', 'Optimization', 'Data', 'Backbone', 'HRNet', 'KeypointHead']
```

### Step 1.4 — `build_label_dictionary()`
As for the detection, `'background': 0` is added and the object classes start at 1. The four distractors (category identifiers 2, 16, 18 and 33) are not mapped, so their objects are not loaded: they stay in the images, where the heatmaps learn to find no keypoint.
```text
label_dictionary: {1: 1, 3: 2, 4: 3, 5: 4, 6: 5, ..., 34: 30, 35: 31, 36: 32, 37: 33, 'background': 0}
```

### Step 1.5 — `inject_keypoint_categories()` → `read_keypoint_categories()`
The keypoints are described in the `categories` of the COCO `labels.json`, as COCO describes its persons. `inject_keypoint_categories()` reads them from the training split only and writes `Data.keypoint_categories`, keyed by class index. It writes nothing when the training folder or the label dictionary is missing; Step 1.6 then reports it. For every mapped class, `read_keypoint_categories()` keeps `keypoints` (names, in the order of the COCO triplets) and `skeleton` (index pairs drawn as lines). A keypoint keeps its name whatever the augmentation, so the synthetic configuration turns the flips and the 90 degree rotation off, its rotation staying within 5 degrees (`SUPERVISED_AUGMENTATION_STRENGTHS`).
```text
keypoint_categories: 33 classes, keys [1, 2, 3, ..., 33]
1  {"name": "square", "keypoints": ["top_left", "top_right", "bottom_right", "bottom_left"], "skeleton": [[0, 1], [1, 2], [2, 3], [3, 0]]}
32 {"name": "stick_figure", "keypoints": ["head", "neck", "left_elbow", "right_elbow", "left_hand", "right_hand", "pelvis",
    "left_knee", "right_knee", "left_foot", "right_foot"],
    "skeleton": [[0, 1], [1, 6], [1, 2], [2, 4], [1, 3], [3, 5], [6, 7], [7, 9], [6, 8], [8, 10]]}
```

### Step 1.6 — `check_configuration()`
After the common checks, `check_keypoint_configuration()` calls `check_image_task_sequence_settings()`, `check_image_only_augmentation_types()` (perspective and jigsaw move the image only), `check_keypoint_data_configuration()`, `check_keypoint_stem_configuration()`, `check_high_resolution_network_configuration()`, `check_keypoint_head_configuration()` and `check_keypoint_evaluation_configuration()`. Two rules come from the architecture: the stem must end at 1/4 (`HEATMAP_STRIDE = 4`), and the image size must be a multiple of 4 × 2^(number of stages), 32 with 3 stages.

**Example: image size 500, `last_max_pooling: True`, `heatmap_sigma: 0.0`, `tag_push_weight: -1.0`, `keypoint_threshold: 1.5`**
```text
Invalid Configuration File : configuration_wrong.yaml
  - 'Data.image_settings.size' (500) must be a multiple of 8, 2 to the power of the 3 Backbone pooling layers, ...
  - The Backbone stem of the keypoint detector must end at 1/4 of the image, got 1/8 from its 3 blocks with 'Backbone.last_max_pooling' True: use 3 blocks without the last max pooling, or 2 blocks with it.
  - 'Data.image_settings.size' (500) must be a multiple of 32, the stride of the lowest resolution branch of the 3 stages of 'HRNet.modules_per_stage'.
  - 'KeypointHead.heatmap_sigma' must be a positive number of heatmap cells, got 0.0.
  - 'KeypointHead.tag_push_weight' must be a number >= 0, got -1.0.
  - 'KeypointHead.inference.keypoint_threshold' must be a number within ]0, 1[, got 1.5.
SystemExit code 1
```

### Step 1.7 — `get_trainer_class()`
```text
get_trainer_class(task="keypoint_detection") -> <class 'vision_training.trainers.keypoint_trainer.KeypointTrainer'>
```

---

## Stage 2 — Trainer Initialization

> Real outputs of building the trainer on the redirected configuration with the iterations of Stage 3 (batch size 8), MLflow off. The construction took 0.86 s on CPU.

### Step 2.1 — `KeypointTrainer.__init__()` → `build_keypoint_tables()`
`build_keypoint_tables()` turns the keys of `keypoint_categories` into integers and sets `maximum_keypoints`, the keypoint count of the largest class (18, the hollow nonagon): every per-object keypoint tensor has 18 positions, smaller classes being padded. It builds two tables indexed by class:

| Table | Content |
|---|---|
| `channel_table` [34, 18] (NumPy) | the heatmap channel of every keypoint of every class (Step 2.8), read by `group_keypoint_candidates()` |
| `sigma_table` [34, 18] | the OKS sigma of every keypoint (`build_keypoint_sigma_table()`): `default_keypoint_sigma` unless `keypoint_sigmas` names it |

It also keeps the three thresholds of `KeypointHead.inference`. The `BaseTrainer` constructor runs last. For a resumed run whose mapping file now gives other classes, `restore_run_label_dictionary()` runs `build_keypoint_tables()` again on the restored classes.
```text
maximum_keypoints 18   channel_table (34, 18) int64   thresholds keypoint 0.1 tag 1.0 object score 0.2
sigma_table[32] [0.05, 0.05, 0.072, 0.072, 0.062, 0.062, 0.05, 0.087, 0.087, 0.089, 0.089, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
```

### Step 2.2 — `BaseTrainer.__init__()` → `get_device()`
Forced on the CPU here; a normal run on this machine prints `cuda`.
```text
For Training We Will Be Using device: cpu
```

### Step 2.3 — `get_run_name()` → `build_run_name()` → `build_keypoint_sections()`
After `KPT`, the input section and the backbone section, `build_keypoint_sections()` adds the tokens below. Augmentation, sigma, tag weights and grouping settings give no token.

| Token | Meaning |
|---|---|
| `HRSE-HRRES-HRBN-HR16.1.2.2x2` | HRNet modules, normalization, base width, modules per stage, blocks per branch (`HRPT`, `HRFRZ` for pretrained or frozen) |
| `K243-KH256-HH` | heatmap channels, hidden width, higher resolution branch (`KHPT` pretrained, `KHRCL` loaded without its keypoint channel layers, `KHFRZ` frozen) |
| `MC33` | number of classes |

```text
KPT_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x64_HRSE-HRRES-HRBN-HR16.1.2.2x2_K243-KH256-HH_MC33
```

### Step 2.4 — `DataAugmenter()` + `setup_training_paths()`
The augmenter of every task; its geometric types move the keypoints with the image (Step 3.4). The run folder is created as for every task (`classification_workflow.md`, Step 2.6). A resumed run first keeps the classes of its `label_dictionary.json`, their keypoint definitions being read again and the tables of Step 2.1 rebuilt. `check_loaders_give_batches()` refuses a loader without batch.

### Step 2.5 — `build_dataloaders()` → `get_keypoint_dataloaders()` → `CocoDetectionDataset(load_keypoints=True)`
`read_test_manifest()` reads `class_names`, colours and file names from the test `labels.json`. `get_keypoint_dataloaders()` then builds one `CocoDetectionDataset` per split (the dataset of the detection, `object_detection_workflow.md`, Step 2.9; validation only when the run reads it), with `load_keypoints=True` and `maximum_keypoints=18`. Training and validation add `load_ignored_regions=True` when `ignore_unlabelled_objects` is on; the test split always adds `load_ignored_objects=True` for Stage 4. A sample is **one whole image** with all its objects.

**Example: the loaders** (recorded with 2 workers and pinned memory; every loader now has 1 worker and pins only on a discrete GPU, `uses_pinned_memory()`)
```text
class_names {0: 'background', 1: 'square', 2: 'triangle', 3: 'rectangle', ..., 32: 'stick_figure', 33: 'house'}
train      CocoDetectionDataset 300 images 38 batches workers 2 prefetch 2 pin True persistent True drop_last False
test       CocoDetectionDataset  30 images  4 batches workers 2 prefetch 2 pin True persistent True drop_last False
keypoint_counts {1: 4, 2: 0, 3: 3, 4: 4, 5: 5, ..., 16: 0, 17: 4, 18: 0, ..., 33: 0, 34: 4, 35: 12, 36: 11, 37: 5}
```
`keypoint_counts` is keyed by COCO category: the four distractors (2, 16, 18, 33) have none.

### Step 2.6 — `CocoDetectionDataset.__getitem__()` → `add_keypoints()`
The sample holds the resized image, the boxes (`[x1, y1, x2, y2]`), the classes and `keypoints` [N, 18, 3]: (x, y, visibility) moved to the resized image like the boxes (the centre of the first pixel at 0.5). COCO visibilities are 2 visible, 1 occluded, 0 unlabelled. `add_keypoints()` raises a `ValueError` when an annotation has no `keypoints` or the wrong number. It sets unlabelled and padding positions to (0, 0, 0) and removes objects without labelled keypoint.

| Option | What the removed objects become |
|---|---|
| `load_ignored_regions` | their masks, from `segmentation`, united into `ignored_regions` [S, S] uint8 (a missing `segmentation` raises a `ValueError`); collated to [B, S, S] |
| `load_ignored_objects` (test) | `ignored_bounding_boxes` [U, 4], `ignored_labels` [U], `ignored_crowds` [U] (`iscrowd`, 0 when missing), ignored by the evaluation (Step 4.8) |

**Example: test image 25, `test_000024.png`** (shortened)
```text
annotation {'id': 82, 'image_id': 25, 'category_id': 11, 'bbox': [44, 274, 79, 29],
            'keypoints': [44.5, 274.5, 2, 107.5, 274.5, 2, 122.5, 302.5, 2, 59.5, 302.5, 2], 'num_keypoints': 4}
  images               (3, 512, 512) torch.uint8
  labels               (4,) torch.int64 [10, 24, 25, 15]
  keypoints            (4, 18, 3) torch.float32
     object 0, class 10 (trapezoid)  [[44.5, 274.5, 2.0], [107.5, 274.5, 2.0], [122.5, 302.5, 2.0], [59.5, 302.5, 2.0], [0.0, 0.0, 0.0] x 14]
```
The image is already 512 pixels, so the keypoints keep the annotation values.

### Step 2.7 — `get_test_sample_classes()` + `choose_sample_identifiers()`
The test images are listed without classes, so the 16 fixed sample images (`number_sample_images: 16`) are spread evenly over the test split.
```text
sample_image_identifiers {1, 3, 5, 7, 9, 11, 13, 15, 16, 18, 20, 22, 24, 26, 28, 30}
```

### Step 2.8 — `build_model()` → `KeypointDetector.__init__()` → `build_keypoint_channel_table()`
The detector has one heatmap and one tag channel per keypoint of every class, a class's channels being consecutive: 243 channels. `build_keypoint_channel_table()` gives `channel_table` [34, 18] (-1 for padding and background), registered as a buffer so the checkpoint holds it, and `channel_names`. The parts of the detector:

| Part | What it is |
|---|---|
| `backbone` | `build_backbone(input_image_size=512)`, a stem ending at 1/4 with 64 channels |
| `high_resolution_network` | `HighResolutionNetwork`: keeps a 1/4 branch throughout, adds a lower branch at each stage, exchanges features between all branches, returns the 1/4 branch (16 channels) |
| `keypoint_head["prediction_convolution"]` | `build_prediction_head()`: 3x3 from 16 to 256 channels with batch normalization and ReLU, then 1x1 to 486 (243 heatmap logits, then 243 tags) at 1/4 |
| `keypoint_head["higher_resolution_branch"]` | `HigherResolutionBranch`: concatenates the 16 features with the 486 predictions, 4x4 transposed convolution of stride 2 (502 to 16 channels), 2 × `blocks_per_branch` refinement convolutions, a head 16 → 256 → 243 heatmap logits at 1/2; transferred and frozen with the head |

The last 1x1 convolution of both heads starts with a normal of standard deviation 0.001 (`HEAD_INITIALISATION_STANDARD_DEVIATION`), the heatmap bias at logit(0.01) = −4.595 (`HEATMAP_PRIOR_PROBABILITY`) and the tag bias at 0. Every cell thus starts at probability 0.01, so the focal loss of the many empty cells is small from the start.
```text
channel_table[ 1 ] [0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
channel_table[ 33 ] [238, 239, 240, 241, 242, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
prediction_convolution    Conv2d(16, 256, 3x3) + BatchNorm2d(256) + ReLU -> Conv2d(256, 486, 1x1)
higher_resolution_branch  ConvTranspose2d(502, 16, 4x4, stride 2) ... heatmap_head -> Conv2d(256, 243, 1x1)
```

### Step 2.9 — `initialise_pretrained_components()` + `build_optimizer()` + `get_tracked_metrics()`
The transferable components are `Backbone`, `HRNet` and `KeypointHead` (with its higher resolution branch), each with a `pretrained` section (all null here). With `KeypointHead.pretrained.class_layers: False` (the default), the head is loaded without its last heatmap and tag convolutions, and the transposed convolution takes the source rows of the features only: a run of any keypoints gives the hidden layers. With `True`, the source must predict the same `class.keypoint` channels in the same order, read from its `data.json`; a source started before 2026-11-20 is refused. The tracked metrics are `Total Loss`, `Heatmap Loss`, `Tag Pull Loss` and `Tag Push Loss`.

### Step 2.10 — `print_experiment_information()` (→ `describe_split()` → `count_split_images_and_annotations()`)
Every split line gives its images and annotations, distractors included; an epoch is 300 / 8 = 38 iterations.
```text
- Train Data Folder       : train (300 images, 1084 annotations)
- Test Data Folder        : test (30 images, 102 annotations)
- Iterations Per Epoch    : 38
- Learning Rate Schedule  : cosine (warm-up 10 iterations)
- Weight Moving Average   : decay 0.999, evaluated next to the trained weights
```

### Step 2.11 — `print_summary()` / `print_receptive_field_summary()` / `print_model_size_summary()`
`KeypointDetector.print_summary()` prints the backbone summary, the branches of every HRNet stage and the heads. The stem divides the image by 4 (max pooling after blocks 1 and 2, none after block 3). Every stem block changes the channel count (3 → 16 → 32 → 64), so none adds a residual shortcut, despite `residual_nets: True` and the `RES` token (`ConvolutionBlock.count_residual_additions()`). The HRNet branches and the refinement keep their width, so their 4 layers add 2 shortcuts each. The HRNet stages are numbered from 1; they are stages 2 to 4 of the HRNet paper.

**Example output** (the model rebuilt on CPU; shortened)
```text
Final Backbone output shape: [1, 64, 128, 128]
HRNet Stage 3 (4 branches): 1/4 [1, 16, 128, 128], 1/8 [1, 32, 64, 64], 1/16 [1, 64, 32, 32], 1/32 [1, 128, 16, 16]
Model Size: Backbone 80,672 | High Resolution Network 2,631,456 | Keypoint Head 162,278 | Higher Resolution Branch 238,739
            Total Model 3,113,145 parameters (12.50 MB)
```
The keypoint head is 16 × 9 × 256 + 2 × 256 + 256 × 486 + 486 = 162,278 parameters; the branch is mostly its transposed convolution (128,512). The stem sees only 30 pixels; context comes from the lower HRNet branches.

### Step 2.12 — `view_input_batches()`
Off in this run (`diagnosis_modes.view_input_images: False`); Step 3.9 shows its panels.

### Step 2.13 — `restore_last_model()`
```text
No Initiation Model Weights Will Be Used...
We Generate A New Model That Will Be Trained From Scratch.
```

---

## Stage 3 — Training Loop

> Real outputs of a CPU run of the shortened configuration, MLflow on: 60 iterations and two test evaluations in 16 min 13 s. The shapes inside one batch come from the Stage 2 trainer on a batch drawn with seed 0, augmented with seed 3.

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
The loop of every task (`classification_workflow.md`, Step 3.1). The table shows what each iteration does after its training step.

| Iteration | 10 | 15 | 20 | 30 | 40 | 45 | 50 | 60 | after the loop |
|---|---|---|---|---|---|---|---|---|---|
| Checkpoint saved | | ✔ | | ✔ | | ✔ | | ✔ | nothing: iteration 60 is saved |
| Validation batch | ✔ | | ✔ | ✔ | ✔ | | ✔ | ✔ | |
| Console summary | ✔ | | ✔ | ✔ | ✔ | | ✔ | ✔ | |
| Test evaluation (trained and EMA weights) | | | | ✔ | | | | ✔ | nothing |

### Step 3.2 — `setup_tensorboard_writers()` / `start_experiment_tracking()` / `save_model()`
Checkpoint N is saved right after its training step (`classification_workflow.md`, Step 3.6). It holds the source run of the three transferable components (`pretrained_backbone`, `pretrained_high_resolution_network`, `pretrained_keypoint_head`, None here) and the moving average of the weights.
```text
Weights/Iteration_60/model_000060.pt   51,083,151 bytes   (the same for 15, 30 and 45)
checkpoint keys ['exponential_moving_average_state', 'format_version', 'iteration', 'label_dictionary', 'mlflow_run_id', 'model_state',
                 'optimizer_state', 'pretrained_backbone', 'pretrained_high_resolution_network', 'pretrained_keypoint_head', 'run_name', 'task']
model_state tensors 846
  channel_table                                                          (34, 18)
  keypoint_head.prediction_convolution.1.weight                          (486, 256, 1, 1)
  keypoint_head.higher_resolution_branch.transposed_convolution.weight   (502, 16, 4, 4)
```
Current checkpoints also hold `pretrained_backbone_blocks` (`None` here). A checkpoint is about four times the 12.50 MB of weights: it adds the averaged weights and the two Adam moments.

### Step 3.3 — `get_next_batch()` → `run_model_iteration()` → `prepare_batch()`
The batch holds 8 whole images with their own numbers of objects, so boxes, labels and keypoints are lists of one tensor per image (`collate_function()`). `prepare_batch()` moves them to the device unchanged (with the ignored regions when present); the model builds the targets later from the augmented keypoints (Step 3.6).
```text
images               (8, 3, 512, 512) uint8 -> float32, 0 to 254
keypoints            list of 8, (1, 18, 3) (1, 18, 3) (3, 18, 3) (2, 18, 3) ... float32
objects per image    1, 1, 3, 2, 4, 5, 4, 3   (23 objects)
classes per image    [house] [hollow_parallelogram] [hollow_square, l_shape, hollow_rhombus] ...
```
A house uses the first 5 of its 18 positions and leaves the 13 others at visibility 0.

### Step 3.4 — `augment_batch()` → `DataAugmenter.augment_batch(keypoints=...)`
Geometric types move boxes and keypoints with the image, **without renaming them**; a keypoint leaving the image becomes unlabelled. An object is removed only when less than `MINIMUM_VISIBLE_AREA_FRACTION` (0.25) of its box and none of its labelled keypoints stay inside. The trainer keeps the images, boxes, labels, keypoints and `augmentation_reports` (for the viewer, Step 3.9). Ignored regions are passed as the augmenter `masks` and move with the image; uncovered borders are filled with 0 and so trained.
```text
image 0: objects 1 -> 1  labelled keypoints  5 ->  5   brightness -38.9 levels, contrast x0.95, saturation x0.78; noise of 20 levels
image 1: objects 1 -> 1  labelled keypoints  8 ->  8   Not augmented (global probability 0.4 not drawn)
...
image 7: objects 3 -> 3  labelled keypoints 20 -> 14   random_crop : 80 % of each side, at top 86 and left 55 pixels; blur 9x9, sigma 2.5
```
The crop of image 7 pushes 6 of its 20 labelled keypoints outside, but its 3 objects stay. Photometric types leave the keypoints in place.

### Step 3.5 — `forward_pass()` → `KeypointDetector.forward(input_tensor, ground_truth_keypoints, ground_truth_labels)`
The images go through the stem (512 → 128), the HRNet (back to 1/4) and the prediction convolution, split into 243 heatmap logits and 243 tags. The higher resolution branch predicts 243 more heatmap logits at 1/2. Heatmaps are returned as sigmoids, tags as they are. With the ground truth, the model builds the targets (Step 3.6) and returns the focal heatmap loss summed over both strides (Step 3.7) and the two weighted tag losses (Step 3.8). Without it, it extracts the candidates (Step 3.10).

**Example: the untrained model on the training batch** (7.0 s without gradients)
```text
heatmaps (8, 243, 128, 128)   tags (8, 243, 128, 128)   higher_resolution_heatmaps (8, 243, 256, 256)
heatmap_loss 10.188147   tag_pull_loss 6.3e-05   tag_push_loss 0.0
```
The push loss is 0 because no image holds two objects of the same class (Step 3.8).

### Step 3.6 — `build_keypoint_targets()` → `draw_gaussian_heatmaps()`
Targets are built on the device for every batch, from the augmented keypoints. Objects are padded to the most objects of an image (5 here) with class 0. `channel_table[labels]` gives the channel of every keypoint. A keypoint is trained when labelled, defined for its class (channel ≥ 0) and inside the image.

| Target | How it is built |
|---|---|
| heatmaps, at strides 4 and 2 | `draw_gaussian_heatmaps()`: a Gaussian of sigma 2 cells centred at keypoint / stride − 0.5, truncated at 6 cells; the keypoint cell set to exactly 1; overlaps keep the maximum |
| `tag_indices` | the keypoint cell in the 1/4 tag map: (channel × 128 + floor(y / 4)) × 128 + floor(x / 4) |
| `ignored_cells` (with ignored regions) | [B, 1, h, w] per stride: cells at least half (`IGNORED_CELL_FRACTION`, 0.5) inside the regions; None here |

```text
heatmaps (8, 243, 128, 128) max 1.0 nonzero cells 23101      heatmaps (8, 243, 256, 256) max 1.0 nonzero cells 23270
tag_indices (8, 5, 18) int64   tag_valid (8, 5, 18) bool   trained keypoints 138
roof_apex of the house of image 0 at (289.5, 385.5): stride 4 peak cell (column 72, row 96), stride 2 (column 144, row 192)
                                                     tag index 3911752 -> channel 238, row 96, column 72
row 96 of channel 238, columns 68 to 76: [0.153, 0.355, 0.643, 0.907, 1.0, 0.852, 0.568, 0.294, 0.119]
```
The 138 trained keypoints are the 144 labelled ones minus the 6 the crop removed. The roof apex lies at 71.875 cells, left of the centre of column 72, so its left neighbour (0.907) is higher than its right one (0.852): the decoding reads this (Step 3.10). About 23,000 of 31.8 million cells are not 0.

### Step 3.7 — `compute_heatmap_loss()`
The penalty-reduced focal loss of CornerNet and CenterNet, in float32 from the logits, summed over the strides. With p the predicted probability and y the target, a cell at 1 gives −(1 − p)² log(p), any other −(1 − y)⁴ p² log(1 − p): cells near a keypoint and cells near 0 cost little. The sum is divided by the number of cells at 1 (at least 1). The heatmap bias starts at −4.595 (probability 0.01) so the empty cells do not swamp the first iterations.

**Example: a toy pair of 8x8 heatmaps, channel 0 with one keypoint cell at 1 and its right neighbour at 0.5**
```text
probability 0.01 everywhere              4.513654   = 0.99² x −log(0.01) + (126 + 0.5⁴) x 0.01² x −log(0.99)
keypoint cell at 0.99                    0.000128   = (1 + 126 + 0.5⁴) x 0.01² x −log(0.99)
plus a false peak of 0.99 in channel 1   4.513654   = 0.99² x −log(0.01) + (1 + 125 + 0.5⁴) x 0.01² x −log(0.99)
```
A missed keypoint and a false peak cost the same, so the detector learns to stay silent where there is no object. The untrained 10.188 is about 4.51 + 0.23 at stride 4 plus 4.51 + 0.93 at stride 2 (four times as many empty cells).

With `ignore_unlabelled_objects`, the ignored cells leave the sum in every channel, except keypoint cells: a crowd teaches neither keypoints nor their absence, and a labelled keypoint inside a crowd stays trained. In the toy pair, a false peak inside 3x3 ignored cells drops the loss to 0.000110, while a missed keypoint inside them keeps its 4.51 (4.513638).

### Step 3.8 — `compute_tag_losses()`
The associative embedding losses, in float32. The tag of every trained keypoint is read at its `tag_indices`; the reference tag of an object is the mean of its keypoint tags.

| Loss | Per image |
|---|---|
| pull | mean over objects of the mean squared distance of their tags to their reference |
| push | mean of exp(−(h_m − h_n)²) over ordered pairs of distinct objects of the **same class** (the only ones sharing channels) |

Both are averaged over images (0 without object or pair), then weighted by 0.1.

**Example: object 0 of class 1 with tags 0.9 and 1.1, object 1 of class 1 with 3.0 and 3.0, object 2 of class 2 with 1.0**
```text
                              pull       push
as given                      0.003333   0.018316   exp(−4) for the pair of class 1
all three of class 1          0.003333   0.345544   objects 0 and 2 share the tag 1
object 1 tags at 1.2          0.003333   0.960789   two objects of class 1 almost on the same tag
```

### Step 3.9 — `view_input_batches()` → `show_input_samples()` → `draw_keypoint_objects()`
Off in this run. With `view_input_images: True`, an augmented training image is shown next to a validation image, each with its labelled keypoints and class skeleton, boxes captioned with class names, beside a panel of objects, visibilities, colours and augmentations.
```text
[Training]                                                 [Validation]
Objects : 1 (house 1)                                      Objects : 1 (hollow_capsule 1)
Keypoints : 5 visible, 0 occluded, 0 unlabelled            Keypoints : 12 visible, 0 occluded, 0 unlabelled
Colours : left blue, right orange, others yellow, rings occluded, skeleton in the class colour
```

### Step 3.10 — `extract_keypoint_candidates()` and `group_keypoint_candidates()` (evaluation and deployment only)
Used by the test evaluation (Step 4.3) and the client (Step 6.6). `extract_keypoint_candidates()` runs in the model:
1. With the branch, the 1/4 heatmaps and tags are upsampled by 2 (bilinear) and the heatmaps averaged with the 1/2 ones.
2. Candidates are the 3x3 local maxima of each channel; the 30 best are kept, padded with zero-score cells.
3. Each moves within its cell to the vertex of the parabola through the log-heatmap and its two neighbours (`compute_peak_offsets()`, DARK): with a and b the drops to the next and previous neighbours, offset (b − a) / (2 (a + b)), within half a cell, none on the border.
4. Back to pixels: x = (column + offset + 0.5) × stride; the tag is read at the cell.

**Example: a toy 8x8 heatmap (stride 4), 0.9 at (row 2, column 3) with neighbours 0.5 right, 0.2 left, 0.6 down, 0.3 up, and 0.7 at (0, 7) on the border**
```text
without the branch                     positions [[14.8761, 10.9217], [30.0, 2.0], [22.0, 22.0]]   scores [0.9, 0.7, 0.0]       tags [1.9, 0.7, 4.5]
with a 16x16 branch, 0.8 at (5, 7)     positions [[14.8489, 10.8932], [31.0, 1.0], [3.0, 1.0]]     scores [0.7563, 0.35, 0.0]   tags [2.125, 0.7, 0.025]
```
In x, a = ln(0.9 / 0.5) = 0.588 and b = ln(0.9 / 0.2) = 1.504, so x = (3 + 0.219 + 0.5) × 4 = 14.876. The border peak stays at (30, 2); the third is a zero-score filler.

`group_keypoint_candidates()` runs on the CPU (NumPy and SciPy), per image and class:
1. The candidates above `keypoint_threshold` are taken in keypoint order; those of the first keypoint start one object each.
2. The next ones are matched to the objects by the Hungarian method (`match_candidates_to_objects()`), cost round(|tag − mean tag of the object|) × 100 − score; a match needs a tag distance below `tag_threshold`, else the candidate starts a new object.
3. The object score is the sum of its keypoint scores over the keypoints of its class (missing ones NaN, score 0); objects below `object_score_threshold` are dropped (0 in the evaluation, to keep the low-confidence end of the curves).

**Example: a toy class of 3 keypoints, keypoint threshold 0.1, tag threshold 1.0**
```text
candidates  channel 0  (10, 10) score 0.9 tag 0.1    (50, 50) score 0.8 tag 2.0    (90, 90) score 0.05 tag 5.0
            channel 1  (20, 10) score 0.7 tag 0.2    (60, 50) score 0.6 tag 1.9    (5, 5)   score 0.3  tag 4.0
            channel 2  (15, 20) score 0.5 tag 2.1    (55, 60) score 0.4 tag 0.15   (0, 0)   score 0.0  tag 0.0
object_score_threshold 0.0   keypoints [[[10, 10], [20, 10], [55, 60]], [[50, 50], [60, 50], [15, 20]], [[nan, nan], [5, 5], [nan, nan]]]
                             scores [0.6667, 0.6333, 0.1]
object_score_threshold 0.2   scores [0.6667, 0.6333]
```
The tags group the keypoints, not their positions: the first object takes (55, 60), whose tag 0.15 is closest to its mean tag, far as it is. The candidate (5, 5) starts a third object of score 0.1.

On the untrained model every cell sits near the 0.01 prior (scores 0.01 to 0.0119), below the threshold, so image 0 gives no object; a threshold of 0 would give 990.

### Step 3.11 — `training_loss.backward()` + `step_optimizer()` + validation pass
As for every task: the total loss is the heatmap loss plus the two weighted tag losses, and the moving average of the weights (decay 0.999) is updated after every optimizer step. Validation batches go through `prepare_batch()` and `forward_pass()` only, without augmentation or gradients.

### Step 3.12 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()`
The summary counts images (iteration 60: Train Total Loss 7.9859, 0.5 Images Per Second). The table below gives the losses of the run per interval.

| Iteration | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|
| Train Heatmap Loss | 9.8744 | 9.1222 | 8.6250 | 8.2795 | 7.9930 | 7.9735 |
| Valid Heatmap Loss | 10.0152 | 10.2595 | 10.8387 | 9.6482 | 8.0675 | 7.9893 |
| Train Tag Pull Loss | 0.0021 | 0.0071 | 0.0059 | 0.0042 | 0.0037 | 0.0026 |
| Train Tag Push Loss | 0.0162 | 0.0124 | 0.0123 | 0.0160 | 0.0148 | 0.0098 |
| Throughput (Images Per Second) | 0.6 | 0.6 | 0.4 | 0.6 | 0.5 | 0.5 |

The heatmap loss makes almost all of the total. It drops first as the empty cells go below 0.01; the keypoint cells rise later. The tag losses stay small because all tags start near 0 (a same-class pair on one tag would cost 0.1 × exp(0) = 0.1). The valid losses come from one batch of 8 per interval, so they jump. When this run was captured, the test evaluations ran at the start of their iteration, so the interval of iteration 30 took 3 min 5 s; they now fall in the interval of iteration 40.

### Step 3.13 — end of the loop
Iteration 60 is already saved and evaluated, so the loop saves and evaluates nothing more (`classification_workflow.md`, Step 3.25).
```text
Last Checkpoint Saved At Iteration 60. Exiting Training.
Tensorboard/Test/         65 scalars and 16 sample images per evaluation                             (iterations 30 and 60)
Tensorboard/Test_EMA/     64 scalars and 16 sample images per evaluation                             (iterations 30 and 60)
```
Test_EMA lacks `Keypoint Accuracy/Mean Keypoint Error (pixels)`: the averaged weights matched no object, so they have no error to log.

---

## Stage 4 — Test Evaluation of a Checkpoint

> The reports are real outputs of the evaluations of iterations 30 and 60 of the Stage 3 run on its test split (30 images, 102 objects: 90 with keypoints, 12 distractors). They predate the per-class recall intervals (2026-11-01): `class_metrics.csv` now holds `oks_recall_50` to `oks_recall_75_upper` and the report an `OKS Per Class Recall` table. Iteration 30 was captured with 29 steps. After 60 CPU iterations the detector finds almost nothing, so the steps inside a test batch also show iteration 2000 of the GPU run of 2026-10-17 (`configuration_synthetic_shapes.yaml`, 6000 iterations, cosine decay over 5500, 250000 training images), loaded into the Stage 2 trainer and run on the same 30 test images. The loaders shuffle, so the first test batch differs between passes; the evaluation does not depend on order.

### Step 4.1 — `run_test_evaluation()` → `evaluate_test_split()`
The evaluation of every task (`classification_workflow.md`, Stage 4): trained weights first, then the averaged weights in `use_averaged_weights()`, written to `EMA/`. The model runs as in deployment, on the whole image without ground truth. One evaluation of the 30 test images takes about 20 seconds on CPU.

### Step 4.2 — `create_evaluation_accumulator()`
An empty `images` list, filled image by image on the CPU, since objects can only be compared within one image.

### Step 4.3 — `prepare_batch()` + `predict_batch()`
`predict_batch()` runs the detector without ground truth (candidates of Step 3.10, no loss), then groups every image with `group_keypoint_candidates()` and the `KeypointHead.inference` settings, except `object_score_threshold`, set to 0 to keep the low-confidence end of the curves. Objects come by decreasing score, keypoints in network input pixels, NaN when not found.
```text
candidate_positions (8, 243, 30, 2)   candidate_scores (8, 243, 30)   candidate_tags (8, 243, 30)
grouped_objects   list of 8, every image: labels (D,)  keypoints (D, 18, 2)  keypoint_scores (D, 18)  scores (D,)
```

The table below counts the objects found in the first test batch by both checkpoints.

| Checkpoint | Candidates above 0.1 per image | Ground truth objects | Objects found | Objects found with score >= 0.2 |
|---|---|---|---|---|
| Iteration 60 (Stage 3) | 0, 0, 0, 0, 0, 0, 0, 0 | 3, 5, 2, 3, 2, 4, 5, 2 | 0 in every image | 0 |
| Iteration 2000 (GPU run) | 13, 50, 39, 30, 77, 29, 64, 36 | 2, 3, 4, 2, 5, 2, 3, 3 | 6, 23, 15, 10, 33, 13, 27, 9 | 2, 3, 4, 2, 6, 2, 3, 3 |

**Example: image `test_000002` with iteration 2000**
```text
ground truth semicircle  [[108.5, 98.0], [108.5, 177.0], [30.0, 177.0], [187.0, 177.0]]
ground truth rhombus     [[460.5, 394.5], [501.5, 415.5], [460.5, 436.5], [419.5, 415.5]]
object 0 rhombus            score 0.8514  keypoints [[460.5, 394.5], [501.5, 414.5], [460.5, 436.5], [419.5, 415.5]]
object 1 semicircle         score 0.7696  keypoints [[106.5, 98.5], [108.5, 177.5], [30.5, 177.5], [187.5, 177.5]]
object 2 hollow_semicircle  score 0.0854  keypoints [[110.5, 98.5], [nan, nan], [nan, nan], [186.5, 177.5]]
objects 3 to 5: hexagon 0.0256, house 0.0208, hollow_rhombus 0.0139, one keypoint each
```
The two real objects are found within 2 pixels. The others are weak answers of other classes on the same corners (hexagon: 0.153 / 6 = 0.0256); samples and deployment drop them at 0.2.

### Step 4.4 — `accumulate_evaluation_batch()`
For every kept image (`sample_indices`), it keeps the ground truth objects, the objects without labelled keypoint (`ignored_bounding_boxes`, `ignored_labels`, `ignored_crowds`, Step 2.6) and the objects found, in network input pixels. The example predates the ignored objects.
```text
labels (2,) int64   bounding_boxes (2, 4) float32   keypoints (2, 18, 2) float32   visibilities (2, 18) int64
predicted_labels (6,) int64   predicted_keypoints (6, 18, 2) float32   predicted_scores (6,) float32
```
On the whole split, iteration 2000 finds 422 objects for 90 ground truth objects, 91 of them scoring at least 0.2.

### Step 4.5 — `save_sample_images()` → `build_sample_image()` → `build_keypoint_sample_grid()`
For the 16 fixed images, `build_keypoint_sample_grid()` draws the labelled keypoints (Ground Truth) next to the objects found scoring at least `object_score_threshold` (Prediction), each with its class skeleton and caption. The legend names the keypoint sides, then the classes.
```text
Evaluation/Iteration_60/Samples/test_000002.png   1028x560 pixels: two 512x512 panels, a caption bar, a legend bar
caption : 2 ground truth objects | 0 predicted objects with score >= 0.20
```
With iteration 2000 the caption would read `2 predicted objects`.

### Step 4.6 — `compute_evaluation_results()` → `evaluate_image_keypoints()`
For every image, `evaluate_image_keypoints()` builds what the detection average precision needs, with the OKS in place of the box IoU.

| Part | What it gives |
|---|---|
| `compute_squared_distances()` | squared distance of every found keypoint to the same keypoint of every ground truth object (NaN infinitely far) |
| `compute_object_keypoint_similarity()` | the [D, G] OKS, with the ground truth box area (`compute_box_areas()`) and the class sigmas; 0 across classes |
| detections | class, score and the box of the found keypoints (`compute_keypoint_boxes()`), which sizes unmatched objects |
| `ignored_overlaps` [D, U] | the pycocotools rule: `compute_squared_extended_box_distances()` measures every found keypoint to the box of an object without labelled keypoint, extended by its width left and right and its height above and below (0 inside), turned into an OKS averaged over the found keypoints; these objects take no part in the keypoint accuracy |
| `match_ground_truth_objects()` | one-to-one matching (`linear_sum_assignment()`) of the largest total OKS over pairs reaching 0.5 (`MINIMUM_MATCH_SIMILARITY`); only for the keypoint accuracy of Step 4.7 |

The OKS is the COCO formula: for every labelled keypoint, exp(−d² / (2 × (2σ)² × area)), averaged over the labelled keypoints, in float64 (area clamped to `MINIMUM_OBJECT_AREA`).

**Example: image `test_000002` with iteration 2000**
```text
keypoint_overlaps          semicircle   rhombus
  rhombus                      0.0000    0.9965
  semicircle                   0.9945    0.0000
  hollow_semicircle, hexagon, house, hollow_rhombus   0   0
match_ground_truth_objects [1, 0]   matched [True, True]
```
The hollow semicircle lies on the semicircle but, of another class, has an OKS of 0: it can only be a false positive.

The table below gives the OKS of a toy object of area 10000 with two keypoints of sigma 0.05 (2 × (2σ)² × area = 200).

| Distance of the first keypoint | OKS, second keypoint exact | OKS, second keypoint unlabelled | OKS, second keypoint not found |
|---|---|---|---|
| 0 | 1.0 | 1.0 | 0.5 |
| 5 | 0.9412 | 0.8825 | 0.4412 |
| 10 | 0.8033 | 0.6065 | 0.3033 |
| 20 | 0.5677 | 0.1353 | 0.0677 |

An unlabelled keypoint leaves the mean, but a labelled keypoint not found counts 0: an object missing half its keypoints reaches 0.5 only when every other keypoint is exact.

### Step 4.7 — `measure_keypoint_accuracy()` + `summarise_keypoint_accuracy()`
`measure_keypoint_accuracy()` compares every ground truth object with its match of Step 4.6: the distance of every keypoint (infinite when not found), whether it is labelled, found and correct (within `CORRECT_KEYPOINT_FRACTION` = 0.05 of the longest box side), and the object OKS. `summarise_keypoint_accuracy()` gives four metrics, globally, per class and per keypoint.

| Metric | Meaning |
|---|---|
| `Matched Objects` | fraction of ground truth objects matched with an OKS of at least 0.5 |
| `PCK@0.05` | fraction of labelled keypoints that are correct, missed ones wrong |
| `Mean OKS` | mean OKS of the objects with labelled keypoints, missed ones 0 |
| `Mean Keypoint Error (pixels)` | mean distance of the found labelled keypoints only |

```text
iteration 2000, test_000002: semicircle  longest side 157.0, correct within 7.85   distances [2.06, 0.5, 0.71, 0.71]   similarity 0.9945
iteration 60, whole split:   Matched Objects 0.0111   PCK@0.05 0.0017   Mean OKS 0.0014   Mean Keypoint Error (pixels) 2.2361
```
At iteration 60 only a hollow trapezoid was matched, with 1 of its 8 keypoints found (√5 = 2.24 pixels away) and an OKS of 0.1223. These outputs predate 2026-10-25, when any OKS above 0 matched; with the minimum of 0.5 that object is missed and the three rates are 0. Iteration 2000 gives Matched Objects 100.00 %, PCK@0.05 94.57 %, Mean OKS 93.37 % and a mean error of 6.1648 pixels.

### Step 4.8 — `summarise_average_precisions(overlap_type="keypoints")`
The average precision of `object_detection_workflow.md` on the OKS instead of the box IoU (`title_prefix="OKS"`, `name_prefix="OKS "`, `column_prefix="oks_"`): three methods at OKS 0.5, 0.75, 0.95 and 0.5 to 0.95, the five object sizes relative to the network input (`object_detection_workflow.md`, Step 4.14), the mean average recall, per-class APs, and the OKS recall of every class at 0.5 and 0.75 with its exact (Clopper-Pearson) 95 % interval (`object_detection_workflow.md`, Step 4.11). Objects are taken by decreasing score; each is a true positive when its OKS with its best ground truth reaches the threshold and no more confident object took it. As in pycocotools, a non-true-positive is ignored when an object without labelled keypoint reaches the threshold with it (`ignored_overlaps`) and is a crowd or still free (`find_free_ignored_object()`): a crowd takes any number, another object only one.
```text
iteration 2000:  OKS mAP@0.5/area 0.9417   OKS mAP@0.75/area 0.9099   OKS mAP@0.95/area 0.7966   OKS mAP@[0.5:0.95]/area 0.8960
```

### Step 4.9 — `write_class_metrics_file()`
The class rows of the average precisions and of the keypoint accuracy are merged into one row per class (a `ValueError` if they list different classes) in `class_metrics.csv`; keypoint rows go to `keypoint_metrics.csv` (`KEYPOINT_METRICS_FILE_NAME`), whatever the per-class report.
```text
class_metrics.csv (34 lines)
class_index,class,ground_truths,oks_average_precision_50,...,oks_average_precision_50_95_large,objects,matched_objects,labelled_keypoints,percentage_correct_keypoints,mean_object_keypoint_similarity,mean_error_pixels
26,hollow_trapezoid,3,0.000000,0.000000,0.000000,0.000000,,,0.000000,0.000000,0.000000,0.000000,3,1,24,0.041667,0.040779,2.236068
keypoint_metrics.csv (244 lines)
class_index,class,keypoint,labelled,found,percentage_correct_keypoints,mean_error_pixels
26,hollow_trapezoid,bottom_left,3,1,0.333333,2.236068
```
No test object is under 32 pixels, so the `_small` columns of that time are empty. The current columns are `_under_1_16` to `_1_2_and_above`.

### Step 4.10 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
The headline metric is `KEYPOINT_HEADLINE_METRIC` = `OKS mAP@0.5/area`. The report gives the keypoint accuracy, then the tables of methods, sizes and recalls, then with the per-class report the per-class APs, `OKS Per Class Recall`, and the accuracy per class and per keypoint. Rates are written as percentages (`report_formatting.py`); counts and the mean error (4 decimals) keep their values.

**Example: the report of iteration 60** (shortened, with the COCO sizes of that time; the current size table starts with `Object Side` and `Pixels`, then one row per relative size)
```text
Iteration 60 · 30 Test Images · 2026-10-18 00:24 ───────────────────────────────────────────────────────────────────────
  Matched Objects   1.11 %   PCK@0.05                       0.17 %
  Mean OKS          0.14 %   Mean Keypoint Error (pixels)   2.2361

  OKS Mean Average Precision By Object Size (Area)
   Object Size           Objects   mAP@0.5   mAP@0.75   mAP@0.95   mAP@[0.5:0.95]
   Small (< 32²)               0         —          —          —                —
   Medium (32² to 96²)        56    0.00 %     0.00 %     0.00 %           0.00 %
  ...
```

The table below compares the four evaluations of the run with iteration 2000 of the GPU run on the same images.

| Checkpoint | Matched Objects | PCK@0.05 | Mean OKS | Mean Error (pixels) | OKS mAP@0.5 (area) | OKS mAP@0.75 | OKS mAP@0.95 | OKS mAP@[0.5:0.95] |
|---|---|---|---|---|---|---|---|---|
| Iteration 30 | 0.00 % | 0.00 % | 0.00 % | — | 0.00 % | 0.00 % | 0.00 % | 0.00 % |
| Iteration 30, EMA Weights | 0.00 % | 0.00 % | 0.00 % | — | 0.00 % | 0.00 % | 0.00 % | 0.00 % |
| Iteration 60 | 1.11 % | 0.17 % | 0.14 % | 2.2361 | 0.00 % | 0.00 % | 0.00 % | 0.00 % |
| Iteration 60, EMA Weights | 0.00 % | 0.00 % | 0.00 % | — | 0.00 % | 0.00 % | 0.00 % | 0.00 % |
| Iteration 2000 of the GPU run | 100.00 % | 94.57 % | 93.37 % | 6.1648 | 94.17 % | 90.99 % | 79.66 % | 89.60 % |

The averaged weights lag a few iterations behind (decay min(0.999, (1 + n) / (10 + n)), 0.87 at the 60th update). With the minimum OKS of a match (2026-10-25), the iteration 60 row is now 0.00 % and —; the OKS mAPs are unchanged.
```text
<run>/Evaluation/Iteration_30/, Iteration_60/   class_metrics.csv, keypoint_metrics.csv, Samples/ (16 PNG), EMA/
<run>/Results.txt                               1444 lines
```

---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> Real outputs of `scripts/evaluation/evaluate_checkpoint.py` pointed at the Stage 3 run (checkpoints 15, 30, 45 and 60, the first three captured before their training step), with `iteration = "all"`, `batch_size = 16`, the whole test split, on CPU with MLflow on. The four checkpoints with their averaged weights took 2 min 27 s, about 17 s per evaluation. Captured before the per-class recall intervals.

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The script of every task (`classification_workflow.md`, Stage 5). The keypoint configurations have no `evaluation.error_dumps`, so `--maximum_error_images_per_class_pair` is skipped with a yellow warning unless it is `None`. `--save_predictions`, `--target_precisions`, `--nms_criterion` and `--nms_distance_exponent`, not given by default, would be skipped the same way when given: no predictions file, no class thresholds, no `Detector` section.

### Step 5.2 — trainer construction
Built from the `training_configuration.yaml` of the run, with the classes of its `label_dictionary.json` (`classification_workflow.md`, Step 5.2). The panels of Steps 2.10 and 2.11 are printed again; with batches of 16, an epoch is 19 iterations.

### Step 5.3 — `get_checkpoint_iterations()`
```text
evaluated_iterations = [60, 45, 30, 15]
```

### Step 5.4 — `evaluate_checkpoints()` → `start_evaluation_tracking()`
The evaluation has its own MLflow run, `<run name>-evaluation`, tagged with the training run (`evaluation_of`).
```text
MLflow Run Created In The Experiment 'keypoint_detection' : 8974fb1c8f1541aaa07bb503baacf79a
```

### Step 5.5 — `summarise_checkpoint_evaluations()`
Checkpoints are ranked on the OKS mAP@0.5 (area), for the trained and the averaged weights. Here the four tie at 0.
```text
Evaluation Summary Of 4 Checkpoints : evaluation_summary.csv
  - Best Checkpoint : iterations 60, 45, 30, 15 tie (OKS mAP@0.5/area 0.00 %)
  - Best Checkpoint (EMA Weights) : iterations 60, 45, 30, 15 tie (OKS mAP@0.5/area 0.00 %)

iteration   OKS mAP@0.5/area   Keypoint Accuracy/Matched Objects   ...   Keypoint Accuracy/Mean Keypoint Error (pixels)   ema_OKS mAP@0.5/area
   60       0.0                0.011111                            ...   2.236068                                         0.0
   45       0.0                0.000000                            ...   NaN                                              0.0
```
`evaluation_summary.csv` has 130 columns: the iteration, 64 scalars of the trained weights, the same 64 prefixed `ema_`. The mean error is NaN when no keypoint was found. It is written in `experiments/<run>/Checkpoint_Evaluation/` with the files of Step 4.9 per iteration and a `Results.txt` of 2888 lines.

### Step 5.6 — `run_test_evaluation()` for each checkpoint
Every checkpoint is loaded with its moving average and evaluated as in Stage 4, trained weights first. The batch size changes nothing, the metrics covering the whole split.
```text
Evaluating : model_000060.pt
Test Evaluation Iteration 60: 100%|██████████| 2/2 [00:17<00:00,  8.61s/it]
Iteration 60 · 30 Test Images · 2026-10-18 00:25 ──────────────────────────────────────────────
Test Evaluation Iteration 60 (EMA Weights): 100%|██████████| 2/2 [00:17<00:00,  8.60s/it]
```

---

## Stage 6 — Export (Optional)

> Steps 6.2 to 6.4 are real outputs of a script repeating `export_experiment()` on CPU on iteration 60 (trained weights), maximum batch size 16, TensorRT precision FP32, the models run by TorchScript and ONNX Runtime on CPU; Triton was not started. The input contract and the model repository of Step 6.5 come from the same steps run on 2026-11-12 on iteration 6000 of the GPU run. The client of Step 6.6 uses the ONNX export of that iteration 6000 (its sample `test_000000.png` shows the same three objects, scored 0.90, 0.87 and 0.80).

### Step 6.1 — `vision_training/deployment/export_model.py` → `export_experiment()`
Run as a script after training, its settings given as arguments (`parse_export_arguments()`). `export_experiment()` turns TF32 off, loads the checkpoint on CUDA with `load_trained_model()` (hard-coded, hence the CPU script here) and wraps it with `build_inference_model()`. It builds the export specification and the input contract (Step 6.5), exports every format into a staging folder, writes `config.pbtxt` and `preprocessing.json`, and compares the export to the eager model, raising a `RuntimeError` on a mismatch. With TensorRT, the engine is checked too (Step 6.5). `install_staged_models()` then moves the checked models to `<repository>/<model name>_<format>/<version>/`.

### Step 6.2 — `load_trained_model()` → `build_inference_model()` → `InferenceKeypointDetector`
`InferenceKeypointDetector` takes preprocessed images and returns the candidates of every channel (Step 3.10): positions in network input pixels, scores and tags. The grouping is left to the client (Step 6.6). Candidates per channel are `maximum_candidates`, capped by the number of heatmap cells.
```text
inference model InferenceKeypointDetector input_image_size 512 number_channels 243 number_candidates 30 maximum_candidates 30 higher_resolution True
```

### Step 6.3 — `get_export_specification()` → `create_random_inputs()`
The keypoint branch gives the model name `keypoint_detector`, one input `images` of 3x512x512, three candidate outputs and `compare_keypoints()` with its tolerances. `create_random_inputs()` traces with a batch of 2 and checks with a batch of 3, so a batch size frozen by the tracing would be caught.
```text
outputs [{'name': 'candidate_positions', 'dimensions': [243, 30, 2], ...}, {'name': 'candidate_scores', 'dimensions': [243, 30], ...},
         {'name': 'candidate_tags', 'dimensions': [243, 30], ...}]
compare_outputs compare_keypoints tolerances {'position_tolerance': 0.01, 'score_tolerance': 0.0001, 'tag_tolerance': 0.001, 'minimum_equal_share': 0.95}
```

### Step 6.4 — `export_torchscript()` / `export_onnx()` + `compare_keypoints()`
`export_torchscript()` traces with `torch.jit.trace`. `export_onnx()` uses the dynamo exporter (opset 18) with a dynamic `batch` dimension and strips the exporter debugging metadata (`strip_exporter_metadata()`). `compare_keypoints()` pairs candidates by position, not by rank: every positive-score candidate is paired with the nearest one of the same image and channel within `position_tolerance`. Paired scores must agree within `score_tolerance`, and at least `minimum_equal_share` of the candidates must have a pair with the same tag. A cell tied with a neighbour up to rounding can become a maximum in one runtime only and shift the ranks. Zero-score fillers are never paired.
```text
torchscript: model.pt    13,902,249 bytes   mismatches []   scored candidates 21870 + 21870   without pair at the same position 0 + 0
onnx:        model.onnx   3,168,184 bytes   mismatches []   scored candidates 21870 + 21870   without pair at the same position 7 + 7
```
In ONNX Runtime 54 candidates sit at another position than the eager candidate of the same rank; a rank-by-rank comparison would report up to 462 pixels between two correct candidates. `hide_known_export_warnings()` hides the `torch.jit` deprecation notices unless `show_warnings` is True.

### Step 6.5 — `write_triton_configuration()` + `write_onnx_input_contract()` / `write_preprocessing_file()`
So that a server builds the network input from a photo as the loaders did, every export writes the input contract (`input_contract.py`). `build_input_contract()` reads it from `Data.image_settings`; `write_onnx_input_contract()` puts it in the `metadata_props` of `model.onnx`; `write_preprocessing_file()` writes `preprocessing.json` next to both formats. It holds no channel names (Step 6.6).

**Example: the model repository** (2026-11-12, iteration 6000; `model.onnx` predates the metadata stripping, now about 269,845 bytes)
```text
triton_repository/
├── keypoint_detector_onnx/config.pbtxt
├── keypoint_detector_onnx/1/model.onnx               3,430,029 bytes, with the input contract in its metadata
├── keypoint_detector_onnx/1/model.onnx.data         12,416,548 bytes
├── keypoint_detector_onnx/1/preprocessing.json             346 bytes
├── keypoint_detector_torchscript/config.pbtxt
├── keypoint_detector_torchscript/1/model.pt         13,917,098 bytes
└── keypoint_detector_torchscript/1/preprocessing.json      346 bytes
```

**Example: `preprocessing.json`**, also the custom metadata of `model.onnx` (2026-11-12; version 3 adds `"input_image_windows": "none"`)
```text
{"input_contract_version": "2", "input_color_order": "BGR", "input_resize": "keep_ratio", "input_size": "512",
 "input_interpolation": "linear", "input_padding_value": "0", "input_padding_placement": "centred",
 "input_tile_scale": "none", "input_tile_overlap": "none", "input_layout": "NCHW", "input_value_range": "0-255"}
```
The photo is decoded in BGR, resized with its aspect ratio kept (longest side to 512, bilinear), centred on black padding, and given as float32 from 0 to 255 in NCHW; the normalization is in the graph. The tile keys only apply to segmenters trained on tiles.

**Example: `keypoint_detector_onnx/config.pbtxt`** (shortened)
```text
name: "keypoint_detector_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 16
input [ { name: "images" data_type: TYPE_FP32 dims: [ 3, 512, 512 ] } ]
output [ { name: "candidate_positions" ... dims: [ 243, 30, 2 ] }, { name: "candidate_scores" ... }, { name: "candidate_tags" ... } ]
dynamic_batching { max_queue_delay_microseconds: 1000 }
optimization { execution_accelerators { gpu_execution_accelerator: [ {
  name: "tensorrt"
  parameters { key: "precision_mode" value: "FP32" }
  parameters { key: "trt_engine_cache_prefix" value: "keypoint_detector_onnx_1_fp32_<hash>" }
  ...
} ] } }
```
The TorchScript configuration uses `platform: "pytorch_libtorch"`, outputs suffixed `__0`, `__1`, `__2`, and no optimization block. `<hash>` comes from the SHA-256 of `model.onnx` and `model.onnx.data`, so other weights never load old engines.

Do not serve this configuration yet. On 2026-11-22, the tutorial `model.onnx` served by Triton 26.01 on this machine with the TensorRT block above, in FP32 or FP16, crashed the server (signal 11 in `libonnxruntime`) at its first request, about 19 seconds after building its engine. The same model served by ONNX Runtime without TensorRT answered at batch 1 and 16. Until this is fixed, export with `--tensorrt_precision None`, which writes no `optimization` block. Since 2026-11-27 the [TensorRT check](../reference/tools_reference.md#tensorrt-check) of the export refuses the TensorRT configurations of iteration 6000. In FP32 the engine check fails with the crash (`Signal (11) received`). In FP16 the overflow scan fails first: `node_Cast_1446` (Cast) reaches 65,535, above the largest FP16 value.

### Step 6.6 — the client: `load_inference_inputs()` → the exported model → `group_keypoint_candidates()`
A client preprocesses every image as the input contract says (`load_inference_inputs()`, `preprocess_image()`), sends it, and groups the candidates with `group_keypoint_candidates()` (Step 3.10), the channel table of the run (`build_keypoint_channel_table()`, Step 1.5) and the `KeypointHead.inference` thresholds. Positions are in network input pixels; for another image size, undo the resize and padding, for example with `restore_boxes_to_original_image()` on every point as the box [x, y, x, y]. `vision_training/deployment/run_inference.py` only measures throughput, without grouping.

**Example: `test_000000.png` (512x512) through the ONNX model of iteration 6000, then the grouping**
```text
outputs (1, 243, 30, 2) (1, 243, 30) (1, 243, 30)
candidates above the keypoint threshold 28 of 7290
grouped {'labels': (3,), 'keypoints': (3, 18, 2), 'keypoint_scores': (3, 18), 'scores': (3,)}
object 0  hollow_capsule    score 0.9009   channels [215, ..., 226]
object 1  hollow_capsule    score 0.8694   channels [215, ..., 226]
object 2  trapezoid         score 0.7993   channels [47, 48, 49, 50]
          keypoints     [[245.94, 369.32], [329.25, 369.52], [345.99, 433.42], [198.34, 433.83]]
          ground truth  [[245.5, 369.5], [329.5, 369.5], [346.5, 433.5], [198.5, 433.5]]
```
The trapezoid is found within 0.51 pixel of its corners. The two hollow capsules lie side by side: all 24 keypoints are found within 2.4 pixels, but the tags exchange three of them (`left_end`, `inner_top_left`, `inner_right_end`), so each object holds 9 keypoints of one capsule and 3 of the other. The keypoint arrays have 18 rows, those past the keypoints of the class being NaN.
