# Keypoint Detection Workflow

This document follows one keypoint detection run (KPT) from `main()` to the evaluation of a checkpoint, to show which function runs when, what it receives, what it returns and who uses the result next. The **Call Tree Overview** gives the whole run on one page. Each stage then has one section, with one step per function: a short explanation followed by a **real** output. The steps shared by every task (configuration loading, checkpoints, training loop, MLflow tracking, standalone evaluation) are explained in detail in `classification_workflow.md`, and the mean average precision machinery (precision-recall curves, the three average precision methods, the object sizes, the mean average recall) in `object_detection_workflow.md`; here they keep a short explanation and the real output of the keypoint detection run. The steps specific to the keypoint detection (keypoint definitions read from the COCO categories, keypoints moved by the augmentations, heatmap and tag targets, the stem, the high resolution network and the higher resolution branch, the focal heatmap loss and the tag losses, the keypoint candidates and their grouping into objects, the object keypoint similarity (OKS) of the objects found and the keypoint accuracy, and the export of the candidates) are explained in full.

The model is a bottom-up keypoint detector (HigherHRNet): the whole image goes through the stem of the backbone and an HRNet (high resolution network), and its head predicts, at 1/4 of the image, one heatmap and one tag map per keypoint of every class. A higher resolution branch predicts a second set of heatmaps at 1/2 of the image. The heatmaps give the keypoint candidates of every channel (its local maxima), and the tags group the candidates of the keypoints of a class into objects: the keypoints of one object share the same tag, two objects of the same class get distant tags. The detector never needs a box: it finds the objects of every class, and none when an image has none.

The examples come from real runs on CPU, so that the GPU stayed free for another training (`CUDA_VISIBLE_DEVICES=` empty for every run). The synthetic dataset of the configuration, `synthetic_shapes`, was generated in a temporary folder with `create_synthetic_dataset(task="keypoint_detection")` at the size of the examples: 300 training, 30 validation and 30 test images of 512 pixels, 2 to 5 objects per image, 37 categories. 33 of them have keypoints (a square has its 4 corners, a triangle 3, a stick figure 11, a hollow nonagon 18), 243 in all; `circle`, `ellipse`, `annulus` and `hollow_ellipse` are distractors without keypoints, left out of the `label_mapping.json` written by `build_keypoint_label_mapping()`, which numbers the 33 others from 1. The data folders and the mapping file of the configuration were redirected to it. The input viewer and every other window stay off.
- **Stages 1 and 2**: `vision_training/configurations/keypoint_detection/configuration_synthetic_shapes.yaml` (512 pixel images, a 3-block stem ending at 1/4 of the image, an HRNet of base width 16 with 1, 2 and 2 modules in its three stages of 2, 3 and 4 branches, a hidden width of 256, the higher resolution branch, a heatmap sigma of 2 cells, tag weights of 0.1), with the run folder redirected to a temporary folder and the iterations of Stage 3.
- **Stages 3, 4 and 5**: the same configuration shortened to batch size 8 (it is 16), 60 iterations (20000), a checkpoint and a test evaluation every 30 iterations (2000), an intermediate checkpoint every 15 (1000), a console summary every 10 (500), a validation batch every 10 training iterations (0, no validation), a warm-up of 10 iterations and a decay of 50 (500 and 19500), 2 loading workers (a setting then, now fixed at 1), with the MLflow tracking turned on (off). The model and the augmentation types are those of Stages 1 and 2, drawn with the probabilities of that time (0.3 for most types, 0.5 for the translation); every type is now drawn at 0.25 (both flips, off here, at 0.5), and `random_haze`, added since, is on. After 60 iterations of batches of 8 images the detector is far from trained, so the metrics of the examples are low: the point is to see the shapes, the files and the order of the steps.
- **Stage 6**: the checkpoint of iteration 60 exported to TorchScript and ONNX on CPU by a script repeating the steps of `export_experiment()`, which only runs on CUDA (see Step 6.1). The client example of Step 6.6 uses the trained weights of iteration 6000 of a GPU run of the same model, whose iteration 2000 also serves the steps inside a test batch of Stage 4.

## Read This First

The keypoint detection task (`task: "keypoint_detection"`, run name code `KPT`) takes one whole image and gives every object of every class it finds: its class, the position of every keypoint of its class in network input pixels (NaN when not found), the score of every keypoint and the score of the object. The keypoint names and skeleton of every class are read from the COCO `categories` of the training `labels.json`, so classes can have different keypoints (4 corners for a square, 11 joints for the stick figure). It is bottom-up (HigherHRNet): the network predicts one heatmap and one tag map per keypoint of every class, 243 channels for the 33 synthetic classes, and `group_keypoint_candidates()` groups the candidates into objects on the CPU, so no box is needed and an image without object gives none. Choose it when every object is described by named points (corners, joints, tips); choose DET for boxes, INS for the mask of every object, SEG for a class per pixel without objects. The dataset is the COCO layout of DET with a `keypoints` list per annotation, and the stem is the shared `Backbone`, so a backbone trained by another task can initialise it.

### The Task At A Glance

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

| Role | File | Main functions or classes |
|---|---|---|
| Trainer | `vision_training/trainers/keypoint_trainer.py`, `vision_training/trainers/coco_split_trainer.py` | `KeypointTrainer`: `prepare_batch`, `augment_batch`, `forward_pass`, `predict_batch`, `accumulate_evaluation_batch`, `compute_evaluation_results`, `build_sample_image`; `CocoSplitTrainer.read_test_manifest` |
| Keypoint definitions | `vision_training/data/keypoints/keypoint_categories.py`, `vision_training/configuration_processing/configuration_loader.py` | `read_keypoint_categories`, `inject_keypoint_categories` |
| Dataset and loaders | `vision_training/data/detection/coco_dataset.py`, `vision_training/data/loaders.py` | `CocoDetectionDataset.add_keypoints`, `collate_function`, `get_keypoint_dataloaders` |
| Model | `vision_training/models/keypoints/keypoint_detector.py`, `vision_training/models/keypoints/hrnet.py` | `KeypointDetector`, `build_keypoint_channel_table`, `extract_keypoint_candidates`, `compute_peak_offsets`; `HighResolutionNetwork`, `BranchTransition`, `ExchangeModule`, `HigherResolutionBranch`, `build_prediction_head` |
| Targets | `vision_training/data/keypoints/heatmap_targets.py` | `build_keypoint_targets`, `draw_gaussian_heatmaps` |
| Losses | `vision_training/losses/keypoint_losses.py` | `compute_heatmap_loss`, `compute_tag_losses` |
| Grouping | `vision_training/models/keypoints/keypoint_grouping.py` | `group_keypoint_candidates`, `match_candidates_to_objects` |
| Metrics and evaluation | `vision_training/evaluation/keypoints/object_keypoint_similarity.py`, `vision_training/evaluation/detection/average_precision_report.py` | `build_keypoint_sigma_table`, `evaluate_image_keypoints`, `match_ground_truth_objects`, `summarise_keypoint_accuracy`; `summarise_average_precisions(overlap_type="keypoints")`, with `compute_clopper_pearson_intervals` of `vision_training/evaluation/recall_intervals.py` |
| Sample images and input viewer | `vision_training/utilities/visualization.py` | `build_keypoint_sample_grid`, `draw_keypoint_objects` |
| Configuration checks | `vision_training/configuration_processing/keypoint_checks.py` | `check_keypoint_configuration` |
| Run name | `vision_training/configuration_processing/run_name.py` | `build_run_name`, `build_keypoint_sections` |
| Synthetic data generator | `vision_training/data/synthetic/create_synthetic_dataset.py` | `create_synthetic_dataset`, `build_keypoint_labels`, `build_keypoint_label_mapping`, `KEYPOINT_CATEGORIES`, `KEYPOINT_NAMES` |
| Export | `vision_training/models/keypoints/inference_keypoint_detector.py`, `vision_training/deployment/export_specifications.py`, `vision_training/deployment/input_contract.py` | `InferenceKeypointDetector` (model name `keypoint_detector`, candidates only), `get_export_specification`, `compare_keypoints`; `build_input_contract`, `write_onnx_input_contract`, `write_preprocessing_file` |

### Run It

```bash
# 1. Synthetic dataset: in the main block of vision_training/data/synthetic/create_synthetic_dataset.py, set
#    task = "keypoint_detection" and output_folders["keypoint_detection"] (250,000 / 50,000 / 50,000 images of 512
#    pixels with 2 to 5 objects; lower split_sizes["keypoint_detection"] for a first try, 300 / 30 / 30 here)
uv run python -m vision_training.data.synthetic.create_synthetic_dataset

# 2. Training: set experiment_folder, the three data folders and Data.label_mapping.mapping_file_path first
uv run python -m vision_training.main --configuration_path="vision_training/configurations/keypoint_detection/configuration_synthetic_shapes.yaml"

# 3. Evaluation of checkpoints: set run_name, experiment_folder, iteration (None, an integer or "all"),
#    test_data_folder, batch_size, track_evaluation and maximum_error_images_per_class_pair = None (Step 5.1) in the
#    main block of scripts/evaluation/evaluate_checkpoint.py, which writes in <run>/Checkpoint_Evaluation/
uv run python -m scripts.evaluation.evaluate_checkpoint

# 4. Export (CUDA only, Step 6.1): set experiment_folder, iteration and formats in the main block of
#    vision_training/deployment/export_model.py
uv run python -m vision_training.deployment.export_model
```

### Configuration Keys To Know First

| Key | Value in the synthetic configuration | Effect |
|---|---|---|
| `KeypointHead.heatmap_sigma` | `2.0` | standard deviation of the target Gaussians in heatmap cells at strides 4 and 2, truncated at ceil(3 sigma) = 6 cells |
| `KeypointHead.tag_pull_weight`, `KeypointHead.tag_push_weight` | `0.1`, `0.1` | weights of the two associative embedding losses in the total loss |
| `KeypointHead.hidden_width` | `256` | 3x3 convolution of 256 channels before the 1x1 predictions of both heads (`KH256`): the 243 heatmaps outnumber the 16 channels of the 1/4 branch; 0 for the 1x1 heads of HigherHRNet |
| `KeypointHead.higher_resolution` | `True` | second set of heatmaps at 1/2 of the image (`HH`), averaged with the upsampled 1/4 heatmaps at inference |
| `KeypointHead.inference` | `maximum_candidates: 30`, `keypoint_threshold: 0.1`, `tag_threshold: 1.0`, `object_score_threshold: 0.2` | candidates per channel and grouping thresholds; the evaluation keeps every object, the sample images and the clients those scoring at least 0.2 |
| `KeypointHead.ignore_unlabelled_objects` | `False` | leaves the cells at least half covered by the objects without labelled keypoint out of the heatmap loss, from their `segmentation` (True in the COCO configuration) |
| `HRNet.base_width`, `HRNet.modules_per_stage`, `HRNet.blocks_per_branch` | `16`, `[1, 2, 2]`, `2` | branches of 16, 32, 64 and 128 channels, exchange modules per stage, residual pairs per branch (`HR16.1.2.2x2`); the image size must be a multiple of 4 x 2^3 = 32 |
| `Backbone.convolutions`, `Backbone.last_max_pooling` | `[1, 16]`, `[2, 32]`, `[2, 64]`, `False` | the stem, which must end at 1/4 of the image: 3 blocks without the last pooling, or 2 with it |
| `ExperimentConfiguration.evaluation.default_keypoint_sigma`, `keypoint_sigmas` | `0.05`, 8 sigmas of `stick_figure` | the OKS tolerance of every keypoint, the COCO constant being 2 sigma |
| `Data.data_augmentation.types` | both flips and `random_90_rotation` off, `random_rotation` on (5 degrees) | the geometric types move the keypoints without renaming them; `random_perspective` and `random_jigsaw` are refused |

### What A Run Writes

- `Results.txt`: one block per evaluated checkpoint and kind of weights, with `Matched Objects`, `PCK@0.05`, `Mean OKS` and `Mean Keypoint Error (pixels)`, then the OKS mAP tables of the methods, object sizes and recalls, and with `per_class_report` the OKS AP per class, the OKS recall per class with its 95 % confidence interval (`OKS Per Class Recall`), the keypoint accuracy per class and per keypoint (Step 4.10). Every rate of the block is written as a percentage with two decimals (`94.17 %`), the counts and the pixel errors keeping their values; the files, TensorBoard and MLflow keep every rate between 0 and 1.
- `Evaluation/Iteration_<n>/class_metrics.csv` (one row per class: the `oks_` average precisions, the OKS recall at 0.5 and 0.75 with the bounds of its 95 % confidence interval, `oks_recall_50` to `oks_recall_75_upper`, and the keypoint accuracy), `keypoint_metrics.csv` (one row per keypoint of every class: labelled, found, PCK, mean error), `Samples/<image>.png` (Ground Truth next to Prediction, 1028x560 for 512 pixel images, the objects of score >= `object_score_threshold`), and `EMA/` with the same files for the moving average of the weights. No confusion matrix and no error images: the task has no `evaluation.error_dumps`.
- `Tensorboard/Test` and `Tensorboard/Test_EMA`: 65 scalars on the synthetic dataset, which has no small object (the 12 `OKS mAP@<threshold>/<method>`, `OKS mAP@0.5 By Size/<size>` and `OKS mAP@[0.5:0.95] By Size/<size>` of the medium and large objects, the 12 `OKS mAR@<threshold>/<all, medium or large>`, the 4 `Keypoint Accuracy/<metric>` and the 33 `OKS AP@0.5/<class>`), and the 16 sample images; `Tensorboard/Train`, and `Tensorboard/Validation` only when `validation_iteration_ratio` is above 0 (0 in this configuration): `Total Loss`, `Heatmap Loss`, `Tag Pull Loss`, `Tag Push Loss` (and `Learning Rate` in Train).
- `Weights/Iteration_<n>/model_<n>.pt`: about 51 MB, four times the 12.50 MB of the weights (the moving average and the two Adam moments), with the source runs of `Backbone`, `HRNet` and `KeypointHead`.

### Pitfalls

- A flip or a large rotation never renames a keypoint: after a left-right flip `left_hand` lies on the right and is still trained in the `left_hand` channel, so these types only suit names that depend neither on the side nor on the orientation (`DataAugmenter.augment_batch()` of `augmentations.py`; [tasks reference, Keypoint Detection](../reference/tasks_reference.md#keypoint-detection-kpt), Augmentation). The comment of `create_synthetic_dataset.py` saying that a flip swaps them is wrong (`bug_standards.md` B28).
- `residual_nets: True` adds no shortcut in the stem: its three blocks all change the channel count (`ConvolutionBlock.count_residual_additions()`; `bug_standards.md` H4, dropped, Step 2.11); only the HRNet branches and the refinement of the higher resolution branch have residual pairs.
- A keypoint head transferred from another run is checked on its class names and tensor shapes only, so keypoints in another order land on the wrong channels (`model_transfer.py`; `bug_standards.md` B10). The export writes no channel names either: a client rebuilds the channel table from the training `labels.json` (`suggestions.md` G14).
- `Matched Objects` and `Mean Keypoint Error (pixels)` match a ground truth object to an object found with an OKS of at least 0.5, the overlap of the OKS mAP@0.5 (`MINIMUM_MATCH_SIMILARITY` of `object_keypoint_similarity.py`). Before 2026-10-25 any OKS above 0 matched, however far the object (`bug_standards.md` B51), so older evaluations can give more matched objects and larger errors.
- `export_experiment()` loads the checkpoint on CUDA, hard-coded, so the export needs a GPU (`export_model.py`, Step 6.1).

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
│   │   └── read_keypoint_categories(labels_file, label_dictionary)
│   │           keypoint names and skeleton of every class, from the training labels.json
│   └── check_configuration()
│       ├── common checks
│       └── check_keypoint_configuration()
│           ├── check_image_task_sequence_settings()               no sequence normalization
│           ├── check_image_only_augmentation_types()              random_perspective and random_jigsaw off
│           ├── check_keypoint_data_configuration()
│           ├── check_keypoint_stem_configuration()                the stem must end at 1/4 of the image
│           ├── check_high_resolution_network_configuration()      image size a multiple of the lowest branch stride
│           ├── check_keypoint_head_configuration()                hidden width, sigma, tag weights, grouping settings
│           └── check_keypoint_evaluation_configuration()          default_keypoint_sigma, keypoint_sigmas
│
├── get_trainer_class(task)  -> KeypointTrainer
│
├── KeypointTrainer(experiment_configuration, model_configuration, configuration_path)   [STAGE 2]
│   │   Reads the keypoint categories and the grouping settings of 'KeypointHead.inference'.
│   ├── build_keypoint_channel_table()                             channel_table [34, 18] for the grouping
│   ├── build_keypoint_sigma_table()                               OKS sigma of every keypoint of every class
│   └── BaseTrainer.__init__()
│       ├── get_device() / get_run_name() -> build_run_name() -> build_keypoint_sections()
│       ├── DataAugmenter()
│       ├── restore_run_label_dictionary() / setup_training_paths()
│       │       the classes of a resumed run kept from its label_dictionary.json (its keypoint definitions read
│       │       again), then the run folder with the configuration and label_dictionary.json
│       ├── build_dataloaders()                                    class_names from the test labels.json
│       │   └── get_keypoint_dataloaders()
│       │       └── CocoDetectionDataset(load_keypoints=True) per split    one whole image per sample
│       ├── check_loaders_give_batches()                           refuses a split whose loader gives no batch
│       ├── save_data_record()                                     data.json
│       ├── get_test_sample_classes() + choose_sample_identifiers()       16 fixed test images
│       ├── build_model()
│       │   └── KeypointDetector.__init__()
│       │       ├── build_keypoint_channel_table()                 243 heatmap channels, registered as a buffer
│       │       ├── build_backbone(input_image_size=512)           the stem, 512 -> 128
│       │       ├── HighResolutionNetwork()                        4 branches, 1/4 to 1/32, back to 1/4
│       │       └── keypoint_head
│       │           ├── prediction_convolution                     3x3 (16 -> 256) + 1x1 (256 -> 243 heatmaps + 243 tags)
│       │           └── higher_resolution_branch                   transposed convolution to 1/2, 243 heatmaps
│       ├── initialise_pretrained_components()                     Backbone, HRNet, KeypointHead
│       ├── build_optimizer() / get_tracked_metrics()              Total, Heatmap, Tag Pull and Tag Push Loss
│       ├── print_experiment_information() (-> describe_split()) / print_summary()
│       │                                                          backbone, HRNet, receptive field, model size panels
│       └── restore_last_model()
│
└── trainer.run_training_loop()   [STAGE 3]
    └── run_training_iterations()
        ├── for each training_iteration, from start_iteration to number_iterations (the total of the run):
        │   ├── get_next_batch() -> CocoDetectionDataset.__getitem__() -> add_keypoints()
        │   │       read_image(); a batch failing to load is dropped and the next one taken
        │   ├── run_training_step()  set_training_mode() to step_optimizer() below, the step also timed by
        │   │       the speed benchmark (diagnosis_modes.benchmark_speed)
        │   ├── run_model_iteration()
        │   │   ├── prepare_batch()                                                   to the device
        │   │   ├── augment_batch() -> DataAugmenter.augment_batch(keypoints=...)     keypoints moved, never renamed
        │   │   ├── forward_pass() -> KeypointDetector.forward(ground_truth_keypoints, ground_truth_labels)
        │   │   │   ├── backbone -> high_resolution_network -> prediction_convolution  heatmaps and tags at 1/4
        │   │   │   ├── higher_resolution_branch                                      heatmaps at 1/2
        │   │   │   ├── build_keypoint_targets() -> draw_gaussian_heatmaps()          Gaussians at strides 4 and 2,
        │   │   │   │                                                                 tag cells of the keypoints
        │   │   │   ├── compute_heatmap_loss()                                        focal loss, summed over the strides
        │   │   │   └── compute_tag_losses()                                          pull and push
        │   │   └── MetricTracker.accumulate() + log_to_tensorboard()
        │   ├── training_loss.backward() / step_optimizer()                           + moving average of the weights
        │   ├── save_model() on the multiples of weight_saving_iterations and intermediate_weight_saving_iterations
        │   ├── validation batch (no augmentation, the losses without gradients), when validation_iteration_ratio > 0
        │   ├── average_trackers() + log_interval_metrics() + print_interval_summary() + report_dropped_batches()
        │   └── run_test_evaluation() on the multiples of weight_saving_iterations, last         -> [STAGE 4]
        └── save_model() of the steps taken since the last checkpoint, if any, then shutdown_loader_workers() and
                finish_experiment_tracking()

run_test_evaluation(iteration)   [STAGE 4]
├── evaluate_test_split()                                     trained weights, then use_averaged_weights() for EMA
│   ├── create_evaluation_accumulator()                       the list of the evaluated images
│   ├── for each test batch:
│   │   ├── prepare_batch() + predict_batch()
│   │   │   ├── KeypointDetector.forward() -> extract_keypoint_candidates()   30 candidates per channel
│   │   │   └── group_keypoint_candidates()                   objects of every class, by their tags
│   │   ├── accumulate_evaluation_batch()
│   │   └── save_sample_images() -> build_sample_image() -> build_keypoint_sample_grid()
│   ├── compute_evaluation_results()
│   │   ├── evaluate_image_keypoints()                        OKS of every object found with every ground truth,
│   │   │   └── match_ground_truth_objects()                  one object found per ground truth object
│   │   ├── summarise_average_precisions(overlap_type="keypoints")            OKS mAP, 3 methods, sizes, recalls,
│   │   │                                                                     recall of every class with its interval
│   │   ├── summarise_keypoint_accuracy() -> measure_keypoint_accuracy()      matched objects, PCK@0.05, mean OKS,
│   │   │                                                                     mean error
│   │   └── write_class_metrics_file()                                        class_metrics.csv, keypoint_metrics.csv
│   └── report_evaluation() -> report_evaluation_results() + track_evaluation()      headline 'OKS mAP@0.5/area'

scripts/evaluation/evaluate_checkpoint.py   [STAGE 5]
└── evaluate_checkpoints() -> start_evaluation_tracking() + summarise_checkpoint_evaluations() -> run_test_evaluation() per checkpoint

vision_training/deployment/export_model.py   [STAGE 6]
└── export_experiment()
    ├── load_trained_model() -> build_inference_model() -> InferenceKeypointDetector     input: whole images
    ├── get_export_specification() -> build_input_contract() -> create_random_inputs()
    ├── export_torchscript() / export_onnx() + write_onnx_input_contract()      the input contract in the ONNX metadata
    ├── write_triton_configuration() + write_preprocessing_file()               config.pbtxt, preprocessing.json
    ├── compare_outputs -> compare_keypoints()                exported candidates against the eager candidates, paired by position
    └── install_staged_models()                               checked models moved from the staging folder into the repository

client of the exported model   [STEP 6.6]
└── load_inference_inputs() -> exported model -> group_keypoint_candidates()     objects of every class, outside the network
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below are real outputs from running the loader on `vision_training/configurations/keypoint_detection/configuration_synthetic_shapes.yaml`, its data folders and mapping file redirected to the generated dataset and its iterations shortened as in Stage 3.

### Step 1.1 — `main()`
The entry point is the same for every task (see `classification_workflow.md`, Step 1.1). The run of Stage 3 was started with:
```text
CUDA_VISIBLE_DEVICES= PYTHONPATH=. COLUMNS=120 .venv/bin/python vision_training/main.py --configuration_path .../configuration_workflow.yaml
```

### Step 1.2 — `load_experiment_configuration()`
The model sections are `Optimization`, `Data`, `Backbone`, `HRNet` and `KeypointHead`. `Data.image_settings.size` is the side of the square network input; the whole image is resized to it. `Data.keypoint_categories` is not in the YAML, it is added by `inject_keypoint_categories()` (Step 1.5).

`HRNet` holds the network after the stem (`base_width`, the channels of its highest resolution branch, doubled at every lower branch; `modules_per_stage`; `blocks_per_branch`; `modules`; `normalization`; `pretrained`). `KeypointHead` holds:
- `higher_resolution`: the higher resolution branch of HigherHRNet, which predicts a second set of heatmaps at 1/2 of the image (Step 2.8).
- `hidden_width`: the channels of a 3x3 convolution before the 1x1 convolution of the predictions, in both heads (0 for none). The 243 heatmaps outnumber the 16 channels of the highest resolution branch by far, and a 1x1 convolution on 16 channels cannot raise every heatmap on its own keypoints only.
- `heatmap_sigma`: the standard deviation of the target Gaussians, in heatmap cells, at both strides.
- `ignore_unlabelled_objects`: leaves the pixels of the objects without labelled keypoint out of the heatmap loss (Steps 2.6, 3.6 and 3.7), which needs their `segmentation`. False here, the synthetic shapes having none; True for the COCO crowds and persons without keypoints.
- `tag_pull_weight` and `tag_push_weight`: the weights of the two tag losses (Step 3.8).
- `inference`: the grouping of the candidates into objects (Step 4.3): `maximum_candidates` per channel, `keypoint_threshold` (the lowest heatmap score of a grouped candidate), `tag_threshold` (the largest distance between the tag of a keypoint and the mean tag of its object) and `object_score_threshold` (the lowest score of an object kept by the deployment and the sample images, the evaluation keeping them all).

The `evaluation` section gives the OKS sigmas: `default_keypoint_sigma`, and `keypoint_sigmas` per class and keypoint.

**Example: the sections and the keypoint settings**
```text
experiment keys: ['task', 'experiment_folder', 'run_name_suffix', 'train_data_folder', 'validation_data_folder', 'test_data_folder',
                  'batch_size', 'information_dump', 'weight_saving_iterations', 'intermediate_weight_saving_iterations',
                  'number_iterations', 'validation_iteration_ratio', 'evaluation', 'tracking', 'learning_rate',
                  'allow_data_change_on_resume', 'diagnosis_modes', 'tracking_root', 'run_name',
                  'standalone_evaluation_folder']
model keys:      ['Optimization', 'Data', 'Backbone', 'HRNet', 'KeypointHead']
evaluation:      {'per_class_report': True, 'number_sample_images': 16, 'default_keypoint_sigma': 0.05,
                  'keypoint_sigmas': {'stick_figure': {'left_elbow': 0.072, 'right_elbow': 0.072, 'left_hand': 0.062, 'right_hand': 0.062,
                                                       'left_knee': 0.087, 'right_knee': 0.087, 'left_foot': 0.089, 'right_foot': 0.089}}}
Data keys:       ['image_settings', 'label_mapping', 'data_augmentation', 'label_mode', 'label_dictionary', 'keypoint_categories']
image_settings:  {'size': 512, 'keep_ratio': True}
HRNet:        {"base_width": 16, "modules_per_stage": [1, 2, 2], "blocks_per_branch": 2,
               "modules": {"squeeze_and_excite": true, "residual_nets": true},
               "normalization": {"feature_map_normalization": "batch"},
               "pretrained": {"experiment_folder": null, "iteration": null, "freeze": false, "learning_rate_multiplier": 1.0}}
KeypointHead: {"higher_resolution": true, "hidden_width": 256, "heatmap_sigma": 2.0, "ignore_unlabelled_objects": false,
               "tag_pull_weight": 0.1, "tag_push_weight": 0.1,
               "inference": {"maximum_candidates": 30, "keypoint_threshold": 0.1, "tag_threshold": 1.0, "object_score_threshold": 0.2},
               "pretrained": {"experiment_folder": null, "iteration": null, "freeze": false, "learning_rate_multiplier": 1.0}}
```

### Step 1.3 — `load_configuration()`
**Example output**
```text
top-level keys: ['ExperimentConfiguration', 'Optimization', 'Data', 'Backbone', 'HRNet', 'KeypointHead']
number_iterations raw: 60
```

### Step 1.4 — `build_label_dictionary()`
As for the detection, `'background': 0` is added to the mapping file, the classes of the objects starting at 1. The four distractors (category identifiers 2, 16, 18 and 33) are not in the mapping file, so their objects are not loaded: they stay in the images, where the heatmaps learn to find no keypoint on them.

**Example: the label dictionary of the generated dataset**
```text
label_dictionary: {1: 1, 3: 2, 4: 3, 5: 4, 6: 5, ..., 34: 30, 35: 31, 36: 32, 37: 33, 'background': 0}
```

### Step 1.5 — `inject_keypoint_categories()` → `read_keypoint_categories()`
The keypoints are not described in the YAML but in the `categories` of the COCO `labels.json`, the way COCO describes its person keypoints. `inject_keypoint_categories()` reads them from the training split only (the validation and test splits share them) and writes `Data.keypoint_categories`, keyed by class index. It returns without writing anything when the training folder or the label dictionary is missing, and the checker of Step 1.6 then reports the problem. For every class of the label dictionary (the classes mapped to 0 are skipped), `read_keypoint_categories()` gives:
- `keypoints`: the names, in the order of the COCO `keypoints` triplets.
- `skeleton`: the pairs of keypoint indices drawn as lines in the viewers and the sample images.

Any other field of a category is ignored. A keypoint keeps its name whatever the augmentation of its image: after a flip, the keypoint labelled `top_left` lies on the top right of the object and is still trained in the `top_left` channel. A flip or a large rotation therefore only suits keypoints whose names do not depend on their side or their orientation; the synthetic configuration turns both flips and the 90 degree rotation off, its rotation staying within the 5 degrees of the supervised tasks (`SUPERVISED_AUGMENTATION_STRENGTHS`).

**Example: 5 of the 33 definitions**
```text
keypoint_categories: 33 classes, keys [1, 2, 3, ..., 33]
1  {"name": "square", "keypoints": ["top_left", "top_right", "bottom_right", "bottom_left"], "skeleton": [[0, 1], [1, 2], [2, 3], [3, 0]]}
2  {"name": "triangle", "keypoints": ["top", "bottom_left", "bottom_right"], "skeleton": [[0, 2], [2, 1], [1, 0]]}
24 {"name": "hollow_rhombus", "keypoints": ["top", "right", "bottom", "left", "inner_top", "inner_right", "inner_bottom", "inner_left"],
    "skeleton": [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4]]}
32 {"name": "stick_figure", "keypoints": ["head", "neck", "left_elbow", "right_elbow", "left_hand", "right_hand", "pelvis",
    "left_knee", "right_knee", "left_foot", "right_foot"],
    "skeleton": [[0, 1], [1, 6], [1, 2], [2, 4], [1, 3], [3, 5], [6, 7], [7, 9], [6, 8], [8, 10]]}
33 {"name": "house", "keypoints": ["roof_apex", "roof_left", "roof_right", "base_left", "base_right"],
    "skeleton": [[0, 2], [2, 4], [4, 3], [3, 1], [1, 0]]}
```

### Step 1.6 — `check_configuration()`
The common checks run first, then `check_keypoint_configuration()`, which calls in order `check_image_task_sequence_settings()`, `check_image_only_augmentation_types()` (the perspective and the jigsaw move the image only, the keypoints cannot follow them), `check_keypoint_data_configuration()`, `check_keypoint_stem_configuration()`, `check_high_resolution_network_configuration()`, `check_keypoint_head_configuration()` and `check_keypoint_evaluation_configuration()`. Two rules come from the architecture:
- The stem must end at 1/4 of the image, the resolution of the heatmaps and tags (`HEATMAP_STRIDE = 4`).
- The image size must be a multiple of the stride of the lowest resolution branch of the HRNet (4 × 2^(number of stages), 32 with 3 stages), so that every branch can be upsampled back exactly to the highest one.

**Example: output with an image size of 500, `Backbone.last_max_pooling: True`, `KeypointHead.heatmap_sigma: 0.0`, `KeypointHead.tag_push_weight: -1.0` and `KeypointHead.inference.keypoint_threshold: 1.5` in the YAML**
```text
------------------------------------------------------
Invalid Configuration File : configuration_wrong.yaml
------------------------------------------------------
  - 'Data.image_settings.size' (500) must be a multiple of 8, 2 to the power of the 3 Backbone pooling layers, so that every feature map entering a pooling is even, e.g. 496 or 504.
  - The Backbone stem of the keypoint detector must end at 1/4 of the image, got 1/8 from its 3 blocks with 'Backbone.last_max_pooling' True: use 3 blocks without the last max pooling, or 2 blocks with it.
  - 'Data.image_settings.size' (500) must be a multiple of 32, the stride of the lowest resolution branch of the 3 stages of 'HRNet.modules_per_stage'.
  - 'KeypointHead.heatmap_sigma' must be a positive number of heatmap cells, got 0.0.
  - 'KeypointHead.tag_push_weight' must be a number >= 0, got -1.0.
  - 'KeypointHead.inference.keypoint_threshold' must be a number within ]0, 1[, got 1.5.
SystemExit code 1
```
The first error comes from the common checks of the image settings.

### Step 1.7 — `get_trainer_class()`
**Example output**
```text
get_trainer_class(task="keypoint_detection") -> <class 'vision_training.trainers.keypoint_trainer.KeypointTrainer'>
```

---

## Stage 2 — Trainer Initialization

> The examples below are real outputs from building the trainer on the redirected configuration with the iterations of Stage 3 (batch size 8), the run folder redirected to a temporary folder and the MLflow tracking off. The whole construction took 0.86 s on CPU.

### Step 2.1 — `KeypointTrainer.__init__()` → `build_keypoint_channel_table()` + `build_keypoint_sigma_table()`
The trainer turns the keys of `keypoint_categories` into integers and sets `maximum_keypoints`, the keypoint count of the largest class (18, the hollow nonagon): every per-object keypoint tensor of the batches has 18 positions, the classes with fewer keypoints being padded. It then builds the tables indexed by class:
- `channel_table` [34, 18], from `build_keypoint_channel_table()` (Step 2.8), as a NumPy array: the heatmap channel of every keypoint of every class, which `group_keypoint_candidates()` reads the candidates of a class with.
- `sigma_table` [34, 18], from `build_keypoint_sigma_table()`: the OKS sigma of every keypoint of every class, `default_keypoint_sigma` unless `keypoint_sigmas` names the keypoint.

It also keeps the three thresholds of `KeypointHead.inference` for the grouping. The constructor of `BaseTrainer` runs last, since it builds the loaders and the model.

**Example: the attributes after the construction**
```text
maximum_keypoints 18   channel_table (34, 18) int64
thresholds keypoint 0.1 tag 1.0 object score 0.2
sigma_table (34, 18) torch.float64
  sigma_table[1]  [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
  sigma_table[32] [0.05, 0.05, 0.072, 0.072, 0.062, 0.062, 0.05, 0.087, 0.087, 0.089, 0.089, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
TASK_NAME = 'keypoint_detection'   SAMPLE_NAME = 'Images'
```

### Step 2.2 — `BaseTrainer.__init__()` → `get_device()`
**Example output** (forced on the CPU; a normal run on this machine prints `cuda`)
```text
For Training We Will Be Using device: cpu
```

### Step 2.3 — `get_run_name()` → `build_run_name()` → `build_keypoint_sections()`
The prefix is `KPT`, followed by the input section (`IN-KAR-512`: input normalization, kept aspect ratio, image size) and the backbone section of every image task, then the sections of `build_keypoint_sections()`:
- `HRSE-HRRES-HRBN-HR16.1.2.2x2`: the HRNet modules (squeeze-and-excitation, residual shortcuts), its normalization (batch), and its base width, modules per stage and blocks per branch (`HRPT` and `HRFRZ` are added for a pretrained or frozen HRNet).
- `K243-KH256-HH`: the heatmap channels of the head, its hidden width and the higher resolution branch (`KHPT` and `KHFRZ` for a pretrained or frozen head).
- `MC33`: the number of classes.

The augmentation, the heatmap sigma, the tag loss weights and the grouping settings give no token. An optimizer section (`ADAMW`, `SGD`, `WD<weight decay>`) and the `run_name_suffix` would come last; plain Adam without weight decay and a null suffix give none.

**Example output**
```text
KPT_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x64_HRSE-HRRES-HRBN-HR16.1.2.2x2_K243-KH256-HH_MC33
```

### Step 2.4 — `DataAugmenter()` + `setup_training_paths()`
The augmenter of every task. Its geometric types move the keypoints of the objects with the image (Step 3.4). The run folder is created as for every task (see `classification_workflow.md`, Step 2.6), with the copy of the configuration and `label_dictionary.json`, the class index of every category of the run: before it, a resumed run keeps the classes of its `label_dictionary.json` when the mapping file now gives others, its keypoint definitions, indexed by class, being read again for them (`restore_run_label_dictionary()`). The loaders built next are refused when one of them gives no batch (`check_loaders_give_batches()`).

### Step 2.5 — `build_dataloaders()` → `get_keypoint_dataloaders()` → `CocoDetectionDataset(load_keypoints=True)`
`build_dataloaders()` calls `read_test_manifest()` of `CocoSplitTrainer`, which reads `class_names` from the test `labels.json` (`get_class_names()`), gives every class a colour for the sample images and keeps the file name of every test image, then calls `get_keypoint_dataloaders()`. It builds one `CocoDetectionDataset` per split (the validation one only when the run reads it, `classification_workflow.md`, Step 2.7: built here, the shortened configuration reading a validation batch every 10 iterations, none with the `validation_iteration_ratio` of 0 of the synthetic configuration), the dataset of the object detection, which packs the image and annotation records of its manifest into one read-only byte buffer (`SerializedRecords`, read by `get_image_record()`, `object_detection_workflow.md`, Step 2.9), with `load_keypoints=True` and `maximum_keypoints=18`, and with `load_ignored_regions=True` for the training and validation splits when `KeypointHead.ignore_unlabelled_objects` is on (off here; never for the test split, which computes no loss). The sample is **one whole image** with all its objects, as for the detection. The dataset reads the number of keypoints of every category from the `categories` of the split (0 for the distractors).

**Example: the loaders** (recorded with 2 workers, when the number of workers was a setting, and pinned memory; every loader now has 1 worker and pins its batches only on a discrete GPU, `uses_pinned_memory()` of `vision_training/data/loaders.py`, so neither on CPU nor on the integrated GPU of this machine)
```text
class_names {0: 'background', 1: 'square', 2: 'triangle', 3: 'rectangle', 4: 'pentagon', 5: 'hexagon', 6: 'plus_sign', 7: 'star',
             8: 'octagon', 9: 'rhombus', 10: 'parallelogram', 11: 'trapezoid', 12: 'kite', 13: 'heptagon', 14: 'nonagon',
             15: 'semicircle', ..., 32: 'stick_figure', 33: 'house'}
train      CocoDetectionDataset 300 images 38 batches workers 2 prefetch 2 pin True persistent True drop_last False
validation CocoDetectionDataset  30 images  4 batches workers 2 prefetch 2 pin True persistent True drop_last False
test       CocoDetectionDataset  30 images  4 batches workers 2 prefetch 2 pin True persistent True drop_last False
keypoint_counts {1: 4, 2: 0, 3: 3, 4: 4, 5: 5, 6: 6, 7: 4, 8: 5, 9: 8, ..., 16: 0, 17: 4, 18: 0, ..., 33: 0, 34: 4, 35: 12, 36: 11, 37: 5}
```
`keypoint_counts` is keyed by COCO category identifier: the circle (2), the ellipse (16), the annulus (18) and the hollow ellipse (33) have none.

### Step 2.6 — `CocoDetectionDataset.__getitem__()` → `add_keypoints()`
The sample holds the resized image, the boxes in `[x1, y1, x2, y2]` resized image pixels, the classes, and `keypoints` [N, 18, 3]: the (x, y, visibility) of every keypoint of every object, moved to the resized image like the boxes (continuous pixel coordinates, the centre of the first pixel at 0.5), the COCO visibilities being 2 visible, 1 occluded and 0 unlabelled. `add_keypoints()` raises a `ValueError` when an annotation has no `keypoints` or not the number of its category, sets the unlabelled keypoints and the padding positions to (0, 0, 0), and removes the objects without any labelled keypoint with their box and label. With `load_ignored_regions`, the masks of the removed objects whose category has keypoints, built from their `segmentation` like the instance masks of the instance segmentation, are united into the [S, S] uint8 `ignored_regions` of the sample (1 on their pixels), a removed object without `segmentation` raising a `ValueError`; the collate stacks them into [B, S, S].

**Example: the test image 25, `test_000024.png`, and its records in `labels.json`** (the keypoint lists shortened)
```text
image record {'id': 25, 'file_name': 'test_000024.png', 'width': 512, 'height': 512}
annotation {'id': 82, 'image_id': 25, 'category_id': 11, 'bbox': [44, 274, 79, 29], 'area': 2291, 'iscrowd': 0,
            'keypoints': [44.5, 274.5, 2, 107.5, 274.5, 2, 122.5, 302.5, 2, 59.5, 302.5, 2], 'num_keypoints': 4}
annotation {'id': 83, 'image_id': 25, 'category_id': 27, 'bbox': [68, 191, 67, 37], 'area': 2479, 'iscrowd': 0,
            'keypoints': [101.5, 191.5, 2, 134.5, 209.5, 2, 101.5, 227.5, 2, 68.5, 209.5, 2, 101.5, 201.77, 2, ...], ...}
annotation {'id': 84, 'image_id': 25, 'category_id': 28, 'bbox': [242, 247, 67, 21], ...}
annotation {'id': 85, 'image_id': 25, 'category_id': 17, 'bbox': [176, 31, 121, 61], 'area': 7381, 'iscrowd': 0,
            'keypoints': [236.5, 31.0, 2, 236.5, 92.0, 2, 176.0, 92.0, 2, 297.0, 92.0, 2], 'num_keypoints': 4}
  images               (3, 512, 512) torch.uint8
  bounding_boxes       (4, 4) torch.float32 [[44.0, 274.0, 123.0, 303.0], [68.0, 191.0, 135.0, 228.0], [242.0, 247.0, 309.0, 268.0],
                                             [176.0, 31.0, 297.0, 92.0]]
  labels               (4,) torch.int64 [10, 24, 25, 15]
  keypoints            (4, 18, 3) torch.float32
     object 0, class 10 (trapezoid)  [[44.5, 274.5, 2.0], [107.5, 274.5, 2.0], [122.5, 302.5, 2.0], [59.5, 302.5, 2.0], [0.0, 0.0, 0.0] x 14]
     object 1, class 24 (hollow_rhombus) [[101.5, 191.5, 2.0], [134.5, 209.5, 2.0], [101.5, 227.5, 2.0], [68.5, 209.5, 2.0],
                                          [101.5, 201.77, 2.0], [115.67, 209.5, 2.0], [101.5, 217.23, 2.0], [87.33, 209.5, 2.0],
                                          [0.0, 0.0, 0.0] x 10]
  sample_identifiers   25
get_test_sample_name -> test_000024
```
The image is already 512 pixels, so the keypoints keep the values of the annotations. The hollow rhombus has its 4 outer corners, then the 4 corners of its hole (`inner_top` at (101.5, 201.77)).

### Step 2.7 — `get_test_sample_classes()` + `choose_sample_identifiers()`
`get_test_sample_classes()` lists the test images without their classes, so that the 16 fixed test images drawn at every evaluation (`number_sample_images: 16`) are spread evenly over the test split.

**Example output**
```text
sample_image_identifiers {1, 3, 5, 7, 9, 11, 13, 15, 16, 18, 20, 22, 24, 26, 28, 30}
names ['test_000000', 'test_000002', 'test_000004', 'test_000006', 'test_000008', 'test_000010', 'test_000012', 'test_000014',
       'test_000015', 'test_000017', 'test_000019', 'test_000021', 'test_000023', 'test_000025', 'test_000027', 'test_000029']
```

### Step 2.8 — `build_model()` → `KeypointDetector.__init__()` → `build_keypoint_channel_table()`
The detector has one heatmap channel and one tag channel per keypoint of every class, the channels of a class being consecutive: 243 channels for the 33 classes. `build_keypoint_channel_table()` gives `channel_table` [34, 18], the channel of every keypoint position of every class (-1 for the padding and the background), registered as a buffer so that it is saved in the checkpoint, and `channel_names`. The detector is made of:
- `backbone`: `build_backbone(input_image_size=512)`, used as a stem ending at 1/4 of the image; its last convolution block gives the input channels of the HRNet (64).
- `high_resolution_network`: `HighResolutionNetwork`, which keeps a 1/4 branch from start to end and adds one lower resolution branch at each of its stages, exchanging features between all branches, then returns the 1/4 branch (`base_width` = 16 channels).
- `keypoint_head["prediction_convolution"]`: `build_prediction_head()`, a 3x3 convolution from 16 to `hidden_width` (256) channels with its batch normalization and a ReLU, then a 1x1 convolution to 486 channels: the 243 heatmap logits, then the 243 tags, at 1/4 of the image.
- `keypoint_head["higher_resolution_branch"]`: `HigherResolutionBranch`, which concatenates the 16 features with the 486 predictions, upsamples them to 1/2 of the image with a 4x4 transposed convolution of stride 2 (502 to 16 channels, batch normalization and ReLU), refines them with 2 × `blocks_per_branch` convolutions (with the squeeze-and-excitation and residual modules of the HRNet), and predicts 243 heatmap logits with a head like the one above (16 → 256 → 243). It belongs to the `KeypointHead` section, so it is transferred and frozen with the head.

The last 1x1 convolution of both heads is initialised with a normal distribution of standard deviation `HEAD_INITIALISATION_STANDARD_DEVIATION` (0.001), the bias of the heatmaps at logit(0.01) = −4.595 (`HEATMAP_PRIOR_PROBABILITY`) and the bias of the tags at 0: every heatmap cell starts at a probability of 0.01, so that the focal loss of the many empty cells is small from the first iteration.

**Example: the channel table and the heads**
```text
channel_table (34, 18) torch.int64 cpu
  channel_table[ 0 ] [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
  channel_table[ 1 ] [0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
  channel_table[ 2 ] [4, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
  channel_table[ 32 ] [227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, -1, -1, -1, -1, -1, -1, -1]
  channel_table[ 33 ] [238, 239, 240, 241, 242, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
channel_names 243 ['square.top_left', 'square.top_right', 'square.bottom_right', 'square.bottom_left', 'triangle.top',
                   'triangle.bottom_left', 'triangle.bottom_right'] ... ['house.roof_apex', 'house.roof_left', 'house.roof_right',
                   'house.base_left', 'house.base_right']
prediction_convolution  Conv2d(16, 256, 3x3, no bias) + BatchNorm2d(256) + ReLU -> Conv2d(256, 486, 1x1)
  last layer (486, 256, 1, 1) std 0.0010   bias abs max 4.595
higher_resolution_branch
  transposed_convolution ConvTranspose2d(502, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
  heatmap_head           Conv2d(16, 256, 3x3, no bias) + BatchNorm2d(256) + ReLU -> Conv2d(256, 243, 1x1)
  last layer (243, 256, 1, 1) std 0.0010   bias abs max 4.595
```

### Step 2.9 — `initialise_pretrained_components()` + `build_optimizer()` + `get_tracked_metrics()`
The three transferable components are the `Backbone`, the `HRNet` and the `KeypointHead` (with its higher resolution branch), each with its `pretrained` section (all null here). The tracked metrics are `{"total_loss": "Total Loss", "heatmap_loss": "Heatmap Loss", "tag_pull_loss": "Tag Pull Loss", "tag_push_loss": "Tag Push Loss"}`.

### Step 2.10 — `print_experiment_information()` (→ `describe_split()` → `count_split_images_and_annotations()`)
`describe_split()` gives the number of images and annotations of every split from its `labels.json`, the distractors included; the iterations per epoch are counted in images: 300 / 8 = 38 iterations.

**Example output** (from the run of Stage 3, with the tracking on; the paths are shortened)
```text
--------------------------------------
=== Experiment Launch Information ===
--------------------------------------
- Run Name                : KPT_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x64_HRSE-HRRES-HRBN-HR16.1.2.2x2_K243-KH256-HH_MC33
- Experiment Folder       : KPT_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x64_HRSE-HRRES-HRBN-HR16.1.2.2x2_K243-KH256-HH_MC33
- Tensorboard Directory   : Tensorboard
- Weights Directory       : Weights
- Configuration File      : training_configuration.yaml
- Data Record             : data.json
- Train Data Folder       : train (300 images, 1084 annotations)
- Validation Data Folder  : validation (30 images, 102 annotations)
- Test Data Folder        : test (30 images, 102 annotations)
- Iterations Per Epoch    : 38
- Batch Size              : 8
- Total Iterations        : 60
- Save And Evaluate Every : 30 iterations
- Save Only Every         : 15 iterations
- Learning Rate           : 0.001
- Optimizer               : adam (weight decay 0.0)
- Learning Rate Schedule  : cosine (warm-up 10 iterations)
- Weight Moving Average   : decay 0.999, evaluated next to the trained weights
- Validation Iter. Ratio  : 10
- MLflow Tracking         : sqlite:///.../experiments/mlflow.db
```

### Step 2.11 — `print_summary()` / `print_receptive_field_summary()` / `print_model_size_summary()`
`KeypointDetector.print_summary()` prints the backbone summary traced on an image, then the branches of every HRNet stage, traced in evaluation mode, and the heads. The stem divides the 512 pixel image by 4: the max pooling of block 1 (1/2), the max pooling of block 2 (1/4), and no max pooling after block 3 (`last_max_pooling: False`). The backbone panel lists the residual shortcuts of a block only when it adds one (`ConvolutionBlock.count_residual_additions()`): every block of the stem changes the channel count (3 → 16, 16 → 32, 32 → 64), which shifts its residual pairs by one layer, so with 1 or 2 layers none adds a shortcut, despite `residual_nets: True` and the `RES` token of the run name (`documentation/bug_standards.md`, H4, dropped). The branches of the HRNet and the refinement of the higher resolution branch keep their width, so their 4 layers add 2 shortcuts each. The HRNet panel numbers the 3 stages of `modules_per_stage` from 1, with 2, 3 and 4 branches: the stages 2 to 4 of the HRNet paper, the stem standing for its stem and its single branch stage 1.

**Example output** (the model of the configuration rebuilt on CPU with the current code)
```text
------------------------------
Backbone Architecture Summary
------------------------------
------------------------------------
Input image shape: [1, 3, 512, 512]
------------------------------------
╭─────────────────────────── Backbone Architecture ────────────────────────────╮
│ Normalization Strategy: Batch                                                │
│                                                                              │
│  - Convolution Block 1                                                       │
│    - Modules      : Squeeze-and-Excitation                                   │
│    - Input Shape  : [1, 3, 512, 512]                                         │
│    - Output Shape : [1, 16, 256, 256]                                        │
│                                                                              │
│  - Convolution Block 2                                                       │
│    - Modules      : Squeeze-and-Excitation                                   │
│    - Input Shape  : [1, 16, 256, 256]                                        │
│    - Output Shape : [1, 32, 128, 128]                                        │
│                                                                              │
│  - Convolution Block 3                                                       │
│    - Modules      : Squeeze-and-Excitation                                   │
│    - Input Shape  : [1, 32, 128, 128]                                        │
│    - Output Shape : [1, 64, 128, 128]                                        │
╰──────────────────────────────────────────────────────────────────────────────╯
-----------------------------------------------
Final Backbone output shape: [1, 64, 128, 128]
-----------------------------------------------
╭─────────────────────────────── High Resolution Network ────────────────────────────────╮
│ Base Width          : 16                                                               │
│ Modules Per Stage   : [1, 2, 2]                                                        │
│ Blocks Per Branch   : 2                                                                │
│ Stem Output         : [1, 64, 128, 128]                                                │
│                                                                                        │
│  - Stage 1 (2 branches)                                                                │
│    - Branch 1/4  : [1, 16, 128, 128]                                                   │
│    - Branch 1/8  : [1, 32, 64, 64]                                                     │
│  - Stage 2 (3 branches)                                                                │
│    - Branch 1/4  : [1, 16, 128, 128]                                                   │
│    - Branch 1/8  : [1, 32, 64, 64]                                                     │
│    - Branch 1/16 : [1, 64, 32, 32]                                                     │
│  - Stage 3 (4 branches)                                                                │
│    - Branch 1/4  : [1, 16, 128, 128]                                                   │
│    - Branch 1/8  : [1, 32, 64, 64]                                                     │
│    - Branch 1/16 : [1, 64, 32, 32]                                                     │
│    - Branch 1/32 : [1, 128, 16, 16]                                                    │
│                                                                                        │
│ Heatmap Channels    : 243 (the keypoints of 33 classes)                                │
│ Tag Channels        : 243                                                              │
│ Hidden Width        : 256 (3x3 convolution before the predictions)                     │
│ Heatmap Size        : 128x128 (stride 4)                                               │
│ Higher Resolution   : 256x256 (stride 2)                                               │
│ Heatmap Sigma       : 2.0 cells                                                        │
│ Tag Weights         : pull 0.1, push 0.1                                               │
╰────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────── Backbone Receptive Field ────────────────────────────────────╮
│ Input Image Size : 512x512                                                                       │
│                                                                                                  │
│ Block 1 - Convolution 1  : Output 512x512x16 | Stride 1 | Receptive Field 3                      │
│ Block 1 - Max Pooling    : Output 256x256x16 | Stride 2 | Receptive Field 4                      │
│                                                                                                  │
│ Block 2 - Convolution 1  : Output 256x256x32 | Stride 2 | Receptive Field 8                      │
│ Block 2 - Convolution 2  : Output 256x256x32 | Stride 2 | Receptive Field 12                     │
│ Block 2 - Max Pooling    : Output 128x128x32 | Stride 4 | Receptive Field 14                     │
│                                                                                                  │
│ Block 3 - Convolution 1  : Output 128x128x64 | Stride 4 | Receptive Field 22                     │
│ Block 3 - Convolution 2  : Output 128x128x64 | Stride 4 | Receptive Field 30                     │
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
│   - Parameters           : 162,278                       │
│   - Estimated Size       : 0.65 MB                       │
│ Higher Resolution Branch                                 │
│   - Parameters           : 238,739                       │
│   - Estimated Size       : 0.96 MB                       │
│ Total Model                                              │
│   - Parameters           : 3,113,145                     │
│   - Estimated Size       : 12.50 MB                      │
╰──────────────────────────────────────────────────────────╯
```
The model size panel is shortened here (it also gives the trainable parameters, equal to the parameters, and an empty line between the components). The keypoint head has 16 × 9 × 256 weights for its 3x3 convolution, 2 × 256 for its batch normalization and 256 × 486 + 486 for its 1x1 convolution: 162,278 parameters. The higher resolution branch is mostly its transposed convolution (502 × 16 × 16 = 128,512) and its heatmap head. The receptive field of the stem (30 pixels) is small next to a 512 pixel image: the context of the objects comes from the lower branches of the HRNet.

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

> The examples below are real outputs from a run on CPU of the shortened configuration of the introduction, with the MLflow tracking on. Its run name is the one of Step 2.3. The 60 iterations, with the two test evaluations of the trained and the averaged weights, took 16 min 13 s, about 2 min 30 s per interval of 10 iterations at 0.5 to 0.6 Images Per Second. The shapes inside one batch come from the trainer of Stage 2 on a training batch drawn with seed 0, augmented with seed 3.

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
The loop of every task (see `classification_workflow.md`, Step 3.1): from `start_iteration` to `number_iterations`, the total of the run, each iteration takes its training step, then saves its checkpoint, runs its validation batch and prints its summary on the iterations set for each, and evaluates its checkpoint last.

**Example: what happens at each iteration of the run**, in the order of the rows within an iteration, after its training step

| Iteration | 10 | 15 | 20 | 30 | 40 | 45 | 50 | 60 | after the loop |
|---|---|---|---|---|---|---|---|---|---|
| Checkpoint saved | | ✔ | | ✔ | | ✔ | | ✔ | nothing: iteration 60 is saved |
| Validation batch | ✔ | | ✔ | ✔ | ✔ | | ✔ | ✔ | |
| Console summary | ✔ | | ✔ | ✔ | ✔ | | ✔ | ✔ | |
| Test evaluation (trained and EMA weights) | | | | ✔ | | | | ✔ | nothing |

The checkpoint of iteration 30 is saved right after its training step, and evaluated last in iteration 30, after its validation batch and its summary.

### Step 3.2 — `setup_tensorboard_writers()` / `start_experiment_tracking()` / `save_model()`
As for every task: the checkpoints are saved right after the training step of iterations 15, 30, 45 and 60, so that the checkpoint of iteration N holds N steps (see `classification_workflow.md`, Step 3.6), with the label dictionary of the run. The checkpoint holds the source run of every transferable component of the task, here three: `pretrained_backbone`, `pretrained_high_resolution_network` and `pretrained_keypoint_head` (None without transfer), and the state of the moving average of the weights. The higher resolution branch is part of the keypoint head, so it is transferred with it.

**Example output**
```text
MLflow Run Created In The Experiment 'keypoint_detection' : 9176e12f26c04c99a115a8f34342d970 (Browse : mlflow ui --backend-store-uri sqlite:///.../experiments/mlflow.db)

Checkpoint Folder :
 - Iteration_15
 - Model Saved In : model_000015.pt
```

**Example: the checkpoint of iteration 60**
```text
Weights/Iteration_15/model_000015.pt   51,083,151 bytes      Weights/Iteration_30/model_000030.pt   51,083,151 bytes
Weights/Iteration_45/model_000045.pt   51,083,151 bytes      Weights/Iteration_60/model_000060.pt   51,083,151 bytes

checkpoint keys ['exponential_moving_average_state', 'format_version', 'iteration', 'label_dictionary', 'mlflow_run_id', 'model_state',
                 'optimizer_state', 'pretrained_backbone', 'pretrained_high_resolution_network', 'pretrained_keypoint_head', 'run_name',
                 'task']
format_version 2   iteration 60   task 'keypoint_detection'   mlflow_run_id '9176e12f26c04c99a115a8f34342d970'
pretrained_backbone None   pretrained_high_resolution_network None   pretrained_keypoint_head None
exponential_moving_average_state keys ['model_state', 'number_updates']
model_state tensors 846
  channel_table                                                          (34, 18)
  keypoint_head.prediction_convolution.0.block.0.weight                  (256, 16, 3, 3)
  keypoint_head.prediction_convolution.1.weight                          (486, 256, 1, 1)
  keypoint_head.higher_resolution_branch.transposed_convolution.weight   (502, 16, 4, 4)
  keypoint_head.higher_resolution_branch.heatmap_head.0.block.0.weight   (256, 16, 3, 3)
  keypoint_head.higher_resolution_branch.heatmap_head.1.weight           (243, 256, 1, 1)
```
The checkpoints written since a deeper backbone can start from a shorter one also hold `pretrained_backbone_blocks`, the number of backbone blocks loaded from a shorter source backbone, which would be `None` for this run (`save_model()` of `base_trainer.py`). A checkpoint (51 MB) is about four times the 12.50 MB of the weights: it also holds the moving average of the weights and the two Adam moments of every weight. The 1x1 prediction convolution gives 486 channels, the 243 heatmaps then the 243 tags. The transposed convolution of the branch reads the 16 features of the highest resolution branch with the 486 predictions (16 + 486 = 502 input channels).

### Step 3.3 — `get_next_batch()` → `run_model_iteration()` → `prepare_batch()`
The loader batch holds 8 whole images, each with its own number of objects, so the boxes, labels and keypoints are lists with one tensor per image (`collate_function()`). `prepare_batch()` moves the images (in the precision of the model) and these lists to the device, with the uint8 ignored regions when the loaders give them, without any other change: the targets are built later by the model, from the keypoints of the batch after the augmentations (Step 3.6). The validation batches and the test evaluation run on these prepared batches.

**Example: the loader batch and the prepared batch**
```text
loader batch                                                       prepare_batch()
  images               (8, 3, 512, 512) uint8                        images               (8, 3, 512, 512) float32, 0 to 254
  bounding_boxes       list of 8, (1, 4) (1, 4) (3, 4) (2, 4) ...    bounding_boxes       list of 8, float32
  labels               list of 8, (1,) (1,) (3,) (2,) ... int64      labels               list of 8, int64
  keypoints            list of 8, (1, 18, 3) (1, 18, 3) (3, 18, 3)   keypoints            list of 8, float32
                       (2, 18, 3) ... float32
  sample_identifiers   (8,) int64                                    sample_identifiers   (8,) int64

objects per image  1, 1, 3, 2, 4, 5, 4, 3   (23 objects)
classes per image  [house] [hollow_parallelogram] [hollow_square, l_shape, hollow_rhombus] [hollow_pentagon, hollow_parallelogram]
                   [stick_figure, square, semicircle, hollow_pentagon] [hollow_trapezoid, l_shape, house, capsule, square]
                   [triangle, rhombus, hollow_rhombus, capsule] [octagon, hollow_trapezoid, star]
```
Every object has 18 keypoint positions (x, y, visibility), the most keypoints of a class; a house uses the first 5 and leaves the 13 others at visibility 0.

### Step 3.4 — `augment_batch()` → `DataAugmenter.augment_batch(keypoints=...)`
The augmentations of every task, applied to the whole images. Every geometric type moves the boxes and the keypoints of the objects with the image, **without renaming them**: after a flip the left corner of a square stays its `top_left` keypoint, wherever it lands. A keypoint leaving the image becomes unlabelled (visibility 0), and an object is removed with its box, label and keypoints only when less than `MINIMUM_VISIBLE_AREA_FRACTION` (0.25) of its box and none of its labelled keypoints stay inside. `DataAugmenter.augment_batch()` returns seven values; the trainer keeps the images, the boxes, the labels, the keypoints and the `augmentation_reports`, which the input viewer turns into text (Step 3.9). When the batch has ignored regions, the trainer gives them as the binary `masks` of the augmenter and keeps them too: they move with the image, the border uncovered by a zoom-out, a translation or a rotation being filled with 0 and so trained.

**Example: the augmentations of the batch**
```text
keys ['images', 'bounding_boxes', 'labels', 'keypoints', 'sample_identifiers', 'augmentation_reports']
image 0: objects 1 -> 1  labelled keypoints  5 ->  5   brightness -38.9 levels, contrast x0.95, saturation x0.78; noise of 20 levels
image 1: objects 1 -> 1  labelled keypoints  8 ->  8   Not augmented (global probability 0.4 not drawn)
image 2: objects 3 -> 3  labelled keypoints 22 -> 22   Not augmented (global probability 0.4 not drawn)
image 3: objects 2 -> 2  labelled keypoints 17 -> 17   noise of 20 levels
image 4: objects 4 -> 4  labelled keypoints 28 -> 28   augmented, but no augmentation type drawn
image 5: objects 5 -> 5  labelled keypoints 24 -> 24   Not augmented (global probability 0.4 not drawn)
image 6: objects 4 -> 4  labelled keypoints 20 -> 20   Not augmented (global probability 0.4 not drawn)
image 7: objects 3 -> 3  labelled keypoints 20 -> 14   random_crop : 80 % of each side, at top 86 and left 55 pixels; blur 9x9, sigma 2.5

image 0 object 0 class house
  before [[289.5, 385.5, 2.0], [263.5, 409.5, 2.0], [315.5, 409.5, 2.0], [263.5, 461.5, 2.0], [315.5, 461.5, 2.0]]
  after  [[289.5, 385.5, 2.0], [263.5, 409.5, 2.0], [315.5, 409.5, 2.0], [263.5, 461.5, 2.0], [315.5, 461.5, 2.0]]
```
The crop of image 7 keeps 80 % of each side and is resized back to 512 pixels: 6 of its 20 labelled keypoints fall outside and become unlabelled, but each of its 3 objects keeps enough of itself to stay. The photometric augmentations of image 0 leave its keypoints in place.

### Step 3.5 — `forward_pass()` → `KeypointDetector.forward(input_tensor, ground_truth_keypoints, ground_truth_labels)`
The images go through the stem (512 → 128), the HRNet (its four branches merged back to 1/4 of the image) and the prediction convolution, whose 486 channels are split into 243 heatmap logits and 243 tags. The higher resolution branch upsamples the features and these predictions to 1/2 of the image and predicts 243 more heatmap logits. The heatmaps returned are the sigmoids of the logits; the tags are returned as they are. With the ground truth, the model builds the targets (Step 3.6) and returns three losses: the focal heatmap loss of the logits at both strides, summed (Step 3.7), and the pull and push tag losses, each multiplied by its weight (0.1, Step 3.8). Without the ground truth, it extracts the keypoint candidates instead (Step 3.10).

**Example: the outputs on the training batch (untrained model, 7.0 s without gradients)**
```text
heatmaps                     (8, 243, 128, 128) float32
tags                         (8, 243, 128, 128) float32
higher_resolution_heatmaps   (8, 243, 256, 256) float32
heatmap_loss                 () float32 10.188147
tag_pull_loss                () float32 6.3e-05
tag_push_loss                () float32 0.0
```
The tag push loss is 0 because no image of this batch holds two objects of the same class: the push only compares objects that share their channels (Step 3.8).

### Step 3.6 — `build_keypoint_targets()` → `draw_gaussian_heatmaps()`
The targets are built on the device, for every batch, from the keypoints after the augmentations. The objects of every image are padded to the most objects of an image (5 here), the padding objects having class 0 and no channel. `channel_table[labels]` gives the heatmap channel of every keypoint, and a keypoint is trained when it is labelled (visibility above 0), its class has this keypoint (channel ≥ 0) and it lies inside the image. For every heatmap stride (4, and 2 with the branch), `draw_gaussian_heatmaps()` draws a Gaussian of sigma `heatmap_sigma` cells (2) around every trained keypoint, centred at keypoint / stride − 0.5 cells and truncated at ceil(3 × sigma) = 6 cells; the cell containing the keypoint is set to exactly 1, the positive cell of the focal loss, and overlapping Gaussians of the same channel keep their maximum. The tag of a keypoint is read at the cell of its channel containing it in the 1/4 tag map: `tag_indices` = (channel × 128 + floor(y / 4)) × 128 + floor(x / 4). With ignored regions, `build_keypoint_targets()` also returns the `ignored_cells` of every stride, [B, 1, h, w] booleans shared by the channels: the cells whose pixels lie at least half (`IGNORED_CELL_FRACTION`, 0.5, the rule of HigherHRNet) in the regions, from an average pooling of the regions over the cells of the stride. They are None here.

**Example: the targets of the training batch**
```text
heatmaps (8, 243, 128, 128) float32 max 1.0 nonzero cells 23101
heatmaps (8, 243, 256, 256) float32 max 1.0 nonzero cells 23270
tag_indices (8, 5, 18) int64   tag_valid (8, 5, 18) bool   trained keypoints 138
object_labels [[33, 0, 0, 0, 0], [25, 0, 0, 0, 0], [18, 17, 24, 0, 0], [21, 25, 0, 0, 0], [32, 1, 15, 21, 0], [26, 17, 33, 16, 1],
               [2, 9, 24, 16, 0], [8, 26, 7, 0, 0]]
```

**Example: the first keypoints of the house of image 0 (class 33, channels 238 to 242, all 5 trained)**
```text
roof_apex  at (289.5, 385.5)   stride 4: peak cell (column 72, row 96) = 1, cell centre (290.0, 386.0)
                               stride 2: peak cell (column 144, row 192) = 1, cell centre (289.0, 385.0)
                               tag index 3911752 -> channel 238, row 96, column 72
roof_left  at (263.5, 409.5)   stride 4: peak cell (column 65, row 102) = 1      stride 2: (column 131, row 204) = 1
                               tag index 3928897 -> channel 239, row 102, column 65
roof_right at (315.5, 409.5)   stride 4: peak cell (column 78, row 102) = 1      stride 2: (column 157, row 204) = 1
                               tag index 3945294 -> channel 240, row 102, column 78
row 96 of channel 238 around the roof apex, columns 68 to 76, at stride 4:
  [0.153, 0.355, 0.643, 0.907, 1.0, 0.852, 0.568, 0.294, 0.119]
```
The 138 trained keypoints are all the labelled keypoints of the batch after the augmentations: 144 before, minus the 6 that the crop of image 7 made unlabelled. The roof apex lies at 289.5 / 4 − 0.5 = 71.875 cells, just left of the centre of column 72: its left neighbour (0.907) is higher than its right one (0.852), which tells the decoding on which side of the cell the keypoint lies (Step 3.10). About 23,000 of the 31.8 million cells of the batch at stride 4 are not 0, so most cells of every heatmap are empty.

### Step 3.7 — `compute_heatmap_loss()`
The penalty-reduced focal loss of CornerNet and CenterNet, computed in float32 from the logits at every stride, then summed over the strides. With p the predicted probability of a cell and y its target, a cell at 1 gives −(1 − p)² log(p), and every other cell −(1 − y)⁴ p² log(1 − p): the cells near a keypoint are penalised less, and the cells already close to 0 hardly at all. The sum over all the cells and channels of the batch is divided by the number of cells at 1 (at least 1), so that the empty cells and the 243 channels do not dilute the keypoints. The bias of the heatmap channels starts at log(0.01 / 0.99) = −4.595, a probability of 0.01 at every cell, so that the millions of empty cells do not swamp the first iterations.

**Example: a toy pair of 8x8 heatmaps (128 cells), channel 0 holding one keypoint cell (row 3, column 3) at 1 and its right neighbour at 0.5**
```text
probability 0.01 everywhere              4.513654   = 0.99² x −log(0.01) + (126 + 0.5⁴) x 0.01² x −log(0.99)
keypoint cell at 0.99                    0.000128   = (1 + 126 + 0.5⁴) x 0.01² x −log(0.99)
plus a false peak of 0.99 in channel 1   4.513654   = 0.99² x −log(0.01) + (1 + 125 + 0.5⁴) x 0.01² x −log(0.99)
```
A missed keypoint and a false peak cost the same: the empty cells of the channels without keypoint are trained towards 0 in every image, so the detector learns to stay silent where there is no object. The heatmap loss of the untrained model, 10.188, is the same sum on the real batch: about 4.51 + 0.23 at stride 4 (the keypoint cells at 0.01, and 31.8 million empty cells at about 10⁻⁶ each over 138 keypoints) plus 4.51 + 0.93 at stride 2 (four times as many empty cells).

With `ignore_unlabelled_objects`, the terms of the ignored cells are left out in every channel, but those of the keypoint cells: a crowd or a person without keypoints teaches the heatmaps neither keypoints nor their absence, and the keypoint of a labelled person standing in a crowd stays trained. The division by the number of keypoint cells is unchanged.

**Example: the toy pair with 3x3 blocks of ignored cells in both channels**
```text
keypoint cell at 0.99, false peak of 0.99 at (row 6, column 6) of channel 1,      0.000110   = (1 + 108 + 0.5⁴) x 0.01² x −log(0.99)
  rows and columns 5 to 7 ignored
probability 0.01 everywhere, rows and columns 2 to 4 ignored around the keypoint  4.513638   = 0.99² x −log(0.01) + 110 x 0.01² x −log(0.99)
```
The false peak, which costs 4.51 outside the ignored cells, costs nothing inside them. The keypoint cell keeps its 4.51 inside them, while its 8 neighbours (the cell at 0.5 among them) and the 9 cells of channel 1 leave the sum.

### Step 3.8 — `compute_tag_losses()`
The associative embedding losses, in float32. The tag of every trained keypoint is read at its `tag_indices`, and the reference tag of an object is the mean tag of its trained keypoints. The pull loss of an image is the mean over its objects of the mean squared distance of their keypoint tags to their reference tag; the push loss of an image is the mean of exp(−(h_m − h_n)²) over the ordered pairs of distinct objects of the **same class**, the only ones that share channels. Both are averaged over the images, an image without object or without pair giving 0, then multiplied by `tag_pull_weight` and `tag_push_weight` (0.1) in the model.

**Example: toy tags, object 0 of class 1 with tags 0.9 and 1.1, object 1 of class 1 with tags 3.0 and 3.0, object 2 of class 2 with tag 1.0**
```text
                              pull       push
as given                      0.003333   0.018316   exp(−(1 − 3)²) = exp(−4) for the pair of class 1
all three of class 1          0.003333   0.345544   (2 exp(−4) + 2 exp(−4) + 2 exp(0)) / 6, objects 0 and 2 share the tag 1
object 1 tags at 1.2          0.003333   0.960789   exp(−(1 − 1.2)²) = exp(−0.04), two objects of class 1 almost on the same tag
```
The pull loss is (0.01 + 0 + 0) / 3: object 0 is 0.1 away from its mean tag on each keypoint, the others sit on it. Object 2 is far from nothing in the first case, being alone in its class.

### Step 3.9 — `view_input_batches()` → `show_input_samples()` → `draw_keypoint_objects()`
Off in this run. With `view_input_images: True`, the viewer shows an augmented training image next to a validation image, each with the labelled keypoints of its objects and the skeleton of their class (`draw_keypoint_objects()`), under their boxes captioned with their class names (`self.class_names`), beside a panel holding the objects per class, the keypoints by visibility, the colours and the augmentations. The panels below were printed by replacing the window by a print of its lines.

**Example: the panels of the first pair**
```text
[Training]                                                 [Validation]
captions ['house']                                         captions ['hollow_capsule']
Objects : 1 (house 1)                                      Objects : 1 (hollow_capsule 1)
Keypoints : 5 visible, 0 occluded, 0 unlabelled            Keypoints : 12 visible, 0 occluded, 0 unlabelled
Colours : left blue, right orange, others yellow,          Colours : left blue, right orange, others yellow,
          rings occluded, skeleton in the class colour               rings occluded, skeleton in the class colour
Augmentations                                              Augmentations : none (validation sample)
Geometric :
  none
Photometric :
  brightness_contrast_saturation : brightness -38.9 levels, contrast x0.95, saturation x0.78
  random_noise : standard deviation 20 levels
Removed objects : none
```

### Step 3.10 — `extract_keypoint_candidates()` and `group_keypoint_candidates()` (evaluation and deployment only)
Not called by the training iterations, but by the test evaluation (Step 4.3) and after the exported model (Step 6.6); they are shown here, next to the targets they decode. `extract_keypoint_candidates()` runs inside the model: with the branch, the 1/4 heatmaps and tags are upsampled by 2 (bilinear) and the heatmaps averaged with the 1/2 ones. The candidates of a channel are its local maxima over a 3x3 window, the `maximum_candidates` (30) highest kept by decreasing score, padded with zero-score cells when there are fewer. Each is placed within its cell, in x and in y, at the vertex of the parabola through the log-heatmap at its cell and at its two neighbours (`compute_peak_offsets()`, the Taylor expansion of DARK along each axis): with a and b the drops of the log-heatmap from the cell to its next and previous neighbours, the offset is (b - a) / (2 (a + b)) cells, towards the higher neighbour and within half a cell (no move on the border), then the candidate is brought back to image pixels: x = (column + offset + 0.5) × stride; its tag is read at its cell.

**Example: a toy 8x8 heatmap (stride 4) with 0.9 at (row 2, column 3), right neighbour 0.5 > left 0.2, lower 0.6 > upper 0.3, and 0.7 at (row 0, column 7) on the border; tags = cell index / 10; 3 candidates**
```text
without the branch                     positions [[14.8761, 10.9217], [30.0, 2.0], [22.0, 22.0]]   scores [0.9, 0.7, 0.0]       tags [1.9, 0.7, 4.5]
with a 16x16 branch, 0.8 at (5, 7)     positions [[14.8489, 10.8932], [31.0, 1.0], [3.0, 1.0]]     scores [0.7563, 0.35, 0.0]   tags [2.125, 0.7, 0.025]
```
Without the branch: in x, a = ln(0.9 / 0.5) = 0.588 and b = ln(0.9 / 0.2) = 1.504, an offset of (1.504 - 0.588) / (2 × 2.092) = 0.219 and x = (3 + 0.219 + 0.5) × 4 = 14.876; in y, a = ln(0.9 / 0.6) = 0.405 and b = ln(0.9 / 0.3) = 1.099, an offset of 0.230 and y = 10.922; the border candidate is not moved, (7.5 × 4, 0.5 × 4) = (30, 2); the third is a zero-score filler, whose neighbours are all 0. With the branch the stride is 2: the upsampled 0.9 peak gives 0.7125 at the cell (row 5, column 7), averaged with the 0.8 of the branch, (0.7125 + 0.8) / 2 = 0.7563; its left and upper neighbours are 0.3281 and its right and lower ones 0.2438 and 0.2688, which moves it 0.075 cell left and 0.053 cell up, to ((7 - 0.075 + 0.5) × 2, (5 - 0.053 + 0.5) × 2) = (14.849, 10.893); the border peak at 0.7 is halved, the branch holding 0 there; the tag 2.125 is the bilinear tag of the cell.

`group_keypoint_candidates()` runs on the CPU (NumPy and SciPy), for every image. For every class with keypoints, the candidates of each keypoint above `keypoint_threshold` are taken in the keypoint order of the class: those of the first keypoint start one object each, and those of the next keypoints are matched to the objects formed so far by the Hungarian method (`match_candidates_to_objects()`), with the cost round(|tag − mean tag of the object|) × 100 − score. A match is kept when the tag distance is below `tag_threshold`; the other candidates start new objects. The score of an object is the sum of its keypoint scores over the number of keypoints of its class, its missing keypoints being NaN with a score of 0, and the objects below `object_score_threshold` are dropped (0 in the evaluation, so that the precision-recall curves keep their low-confidence end).

**Example: a toy class of 3 keypoints (channels 0, 1, 2), keypoint threshold 0.1, tag threshold 1.0**
```text
candidates  channel 0  (10, 10) score 0.9 tag 0.1    (50, 50) score 0.8 tag 2.0    (90, 90) score 0.05 tag 5.0
            channel 1  (20, 10) score 0.7 tag 0.2    (60, 50) score 0.6 tag 1.9    (5, 5)   score 0.3  tag 4.0
            channel 2  (15, 20) score 0.5 tag 2.1    (55, 60) score 0.4 tag 0.15   (0, 0)   score 0.0  tag 0.0

object_score_threshold 0.0   labels [1, 1, 1]
   keypoints [[[10, 10], [20, 10], [55, 60]], [[50, 50], [60, 50], [15, 20]], [[nan, nan], [5, 5], [nan, nan]]]
   keypoint_scores [[0.9, 0.7, 0.4], [0.8, 0.6, 0.5], [0.0, 0.3, 0.0]]   scores [0.6667, 0.6333, 0.1]
object_score_threshold 0.2   labels [1, 1]   scores [0.6667, 0.6333]
```
The candidate (90, 90) is below the keypoint threshold. The tags group the keypoints, not their positions: the first object gets (55, 60) as its third keypoint, the candidate whose tag 0.15 is closest to its mean tag 0.15, far as it is from its two other keypoints. The candidate (5, 5), with a tag 2 away from both objects, starts a third object of score 0.3 / 3 = 0.1, dropped at a threshold of 0.2.

**Example: the untrained model on the training batch (6.6 s)**
```text
heatmaps                     (8, 243, 128, 128)   0.0077 to 0.0127
tags                         (8, 243, 128, 128)   −0.2698 to 0.2883
higher_resolution_heatmaps   (8, 243, 256, 256)   0.0084 to 0.0118
candidate_positions          (8, 243, 30, 2)      1.0 to 511.0
candidate_scores             (8, 243, 30)         0.01 to 0.0119
candidate_tags               (8, 243, 30)         −0.2358 to 0.2277
image 0: candidates above the keypoint threshold 0.1: 0, objects 0
         with a keypoint threshold of 0: 990 objects of classes 1, 2, 3, ..., best scores 0.0106, 0.0106, 0.0105
```
All the cells of the untrained model are close to the prior of 0.01, below the keypoint threshold of 0.1: it finds no object, where a threshold of 0 would turn every candidate into objects.

### Step 3.11 — `training_loss.backward()` + `step_optimizer()` + validation pass
As for every task: the total loss is the sum of the heatmap loss and of the two weighted tag losses, and the moving average of the weights (decay 0.999) is updated after every optimizer step. The validation batches go through `prepare_batch()` and `forward_pass()` only, without augmentation and without gradients.

### Step 3.12 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()`
The summary counts images.

**Example output: the summaries of iterations 10 and 60**
```text
========================================================================================================================
|    Iteration: 10     |      Total Images: 80     |    Interval Images: 80     |  Throughput: 0.6 Images Per Second   |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :     9.8927
Moving Average of Valid Total Loss                     :    10.0279
Moving Average of Train Heatmap Loss                   :     9.8744
Moving Average of Valid Heatmap Loss                   :    10.0152
Moving Average of Train Tag Pull Loss                  :     0.0021
Moving Average of Valid Tag Pull Loss                  :     0.0003
Moving Average of Train Tag Push Loss                  :     0.0162
Moving Average of Valid Tag Push Loss                  :     0.0125
------------------------------------------------------------------------------------------------------------------------
Interval Of 10 Iterations Took 00:02:24.
Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
========================================================================================================================
...
========================================================================================================================
|    Iteration: 60     |     Total Images: 480     |    Interval Images: 80     |  Throughput: 0.5 Images Per Second   |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :     7.9859    -0.0256 Best
Moving Average of Valid Total Loss                     :     8.0162    -0.0787 Best
Moving Average of Train Heatmap Loss                   :     7.9735    -0.0195 Best
Moving Average of Valid Heatmap Loss                   :     7.9893    -0.0782 Best
Moving Average of Train Tag Pull Loss                  :     0.0026    -0.0011
Moving Average of Valid Tag Pull Loss                  :     0.0019    -0.0005
Moving Average of Train Tag Push Loss                  :     0.0098    -0.0050 Best
Moving Average of Valid Tag Push Loss                  :     0.0250    +0.0000
------------------------------------------------------------------------------------------------------------------------
Interval Of 10 Iterations Took 00:02:34.
Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
========================================================================================================================
```

**Example: the training losses of the run**

| Iteration | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|
| Train Heatmap Loss | 9.8744 | 9.1222 | 8.6250 | 8.2795 | 7.9930 | 7.9735 |
| Valid Heatmap Loss | 10.0152 | 10.2595 | 10.8387 | 9.6482 | 8.0675 | 7.9893 |
| Train Tag Pull Loss | 0.0021 | 0.0071 | 0.0059 | 0.0042 | 0.0037 | 0.0026 |
| Train Tag Push Loss | 0.0162 | 0.0124 | 0.0123 | 0.0160 | 0.0148 | 0.0098 |
| Throughput (Images Per Second) | 0.6 | 0.6 | 0.4 | 0.6 | 0.5 | 0.5 |

The heatmap loss makes almost all of the total loss. It starts near the 10.19 of the untrained model (Step 3.7) and drops as the empty cells go below 0.01; the keypoint cells only rise later. The tag losses stay small at first because the tags all start near 0, which costs nothing to the pull loss; the push loss of a pair of same-class objects on the same tag would be 0.1 × exp(0) = 0.1. The valid losses come from a single batch of 8 images per interval, so they jump from one interval to the next. The interval of iteration 30 took 3 min 5 s instead of about 2 min 30 s, as it held the two test evaluations of iteration 30 when this run was captured, the evaluations then running at the start of their iteration; they now run after the summary of iteration 30, in the interval of iteration 40.

### Step 3.13 — end of the loop
The checkpoint of iteration 60 is saved right after its training step and evaluated last in its iteration, so the loop saves nothing more and evaluates nothing (see `classification_workflow.md`, Step 3.25).

**Example output: the end of iteration 60 and of the loop**
```text
Checkpoint Folder :                                   <- right after the training step of iteration 60
 - Iteration_60
 - Model Saved In : model_000060.pt
...                                                   <- the summary, then the two evaluations of iteration 60 (Stage 4)
Last Checkpoint Saved At Iteration 60. Exiting Training.
```

**Example: the TensorBoard curves after the run**
```text
Tensorboard/Train/        Total Loss, Heatmap Loss, Tag Pull Loss, Tag Push Loss, Learning Rate      (iterations 1 to 60)
Tensorboard/Validation/   Total Loss, Heatmap Loss, Tag Pull Loss, Tag Push Loss                     (iterations 10, 20, ..., 60)
Tensorboard/Test/         65 scalars and 16 sample images per evaluation                             (iterations 30 and 60)
Tensorboard/Test_EMA/     64 scalars and 16 sample images per evaluation                             (iterations 30 and 60)
```
Test_EMA has one scalar less, `Keypoint Accuracy/Mean Keypoint Error (pixels)`: the averaged weights matched no object at either evaluation, so they have no keypoint error to log (the trained weights have one at iteration 60 only, Step 4.7).

---


## Stage 4 — Test Evaluation of a Checkpoint

> The reports below are real outputs of the evaluations of the checkpoints of iterations 30 and 60 of the run of Stage 3, on its whole test split (30 images of 102 objects, of which 90 have keypoints, the 12 others being distractors), with `per_class_report: True` and `number_sample_images: 16`. They were captured before the per-class recall intervals were added (2026-11-01): `class_metrics.csv` now holds `oks_recall_50`, `oks_recall_50_lower`, `oks_recall_50_upper`, `oks_recall_75`, `oks_recall_75_lower` and `oks_recall_75_upper` after the OKS average precisions, and with `per_class_report` the report shows an `OKS Per Class Recall` table after `OKS Per Class Average Precision (Area)` (Steps 4.8 and 4.10). The outputs of iteration 30 were captured when its checkpoint was still saved before its training step, after 29 steps; it now holds 30 steps. After 60 iterations on CPU the detector finds almost nothing, so the steps inside one test batch also show the trained weights of iteration 2000 of a GPU run of the same model: `configuration_synthetic_shapes.yaml` with 6000 iterations (cosine decay over 5500) on the 250000 training images, loaded by `load_state_dict()` into the trainer of Stage 2 and run on the first test batch of the same 30 test images. The loaders shuffle every split, so the first test batch differs between two passes; the evaluation does not depend on the order.

### Step 4.1 — `run_test_evaluation()` → `evaluate_test_split()`
The evaluation of every task (see `classification_workflow.md`, Stage 4): the trained weights are evaluated by `evaluate_test_split()`, then the averaged weights in `use_averaged_weights()`, their files written in the `EMA` subfolder of the evaluation folder. The keypoint trainer keeps the default `evaluation_context()`, which changes nothing. The model runs as in deployment: on the whole resized image, without ground truth, every object being found by the grouping of the keypoint candidates. On the CPU of Stage 3, one evaluation of the 30 test images (4 batches of 8) takes about 20 seconds.

### Step 4.2 — `create_evaluation_accumulator()`
An empty `images` list, extended image by image on the CPU: the ground truth objects of an image cannot be compared with the objects found in another one, so they are kept image by image instead of in one flat list of objects.

### Step 4.3 — `prepare_batch()` + `predict_batch()`
`prepare_batch()` is the one of Step 3.3, without augmentation. `predict_batch()` runs the detector without ground truth, so that it returns the candidates of Step 3.10 and no loss, then groups the candidates of every image into objects on the CPU with `group_keypoint_candidates()` and the settings of `KeypointHead.inference` (`keypoint_threshold` 0.1, `tag_threshold` 1.0), except `object_score_threshold`, set to 0: every object is kept whatever its score, so that the precision-recall curves keep their low-confidence end. The objects of every image come by decreasing score, their keypoints in network input pixels, NaN for the keypoints not found.

**Example: the outputs on the first test batch**
```text
heatmaps (8, 243, 128, 128)   tags (8, 243, 128, 128)   higher_resolution_heatmaps (8, 243, 256, 256)        all float32
candidate_positions (8, 243, 30, 2)   candidate_scores (8, 243, 30)   candidate_tags (8, 243, 30)
grouped_objects   list of 8, every image: labels (D,)  keypoints (D, 18, 2)  keypoint_scores (D, 18)  scores (D,)
```

**Example: the objects found in the first test batch**

| Checkpoint | Candidates above 0.1 per image | Ground truth objects | Objects found | Objects found with score >= 0.2 |
|---|---|---|---|---|
| Iteration 60 (Stage 3) | 0, 0, 0, 0, 0, 0, 0, 0 | 3, 5, 2, 3, 2, 4, 5, 2 | 0 in every image | 0 |
| Iteration 2000 (GPU run) | 13, 50, 39, 30, 77, 29, 64, 36 | 2, 3, 4, 2, 5, 2, 3, 3 | 6, 23, 15, 10, 33, 13, 27, 9 | 2, 3, 4, 2, 6, 2, 3, 3 |

After 60 iterations no candidate reaches the keypoint threshold of 0.1, so the grouping starts no object. After 2000 iterations, the objects above 0.2 are, except one, as many as the ground truth objects.

**Example: image `test_000002` with the checkpoint of iteration 2000** (semicircle keypoints arc_top, base_centre, base_left, base_right; rhombus keypoints top, right, bottom, left)
```text
ground truth semicircle  [[108.5, 98.0], [108.5, 177.0], [30.0, 177.0], [187.0, 177.0]]
ground truth rhombus     [[460.5, 394.5], [501.5, 415.5], [460.5, 436.5], [419.5, 415.5]]

object 0 rhombus            score 0.8514  keypoints [[460.5, 394.5], [501.5, 414.5], [460.5, 436.5], [419.5, 415.5]]
                                          keypoint scores [0.885, 0.795, 0.927, 0.798]
object 1 semicircle         score 0.7696  keypoints [[106.5, 98.5], [108.5, 177.5], [30.5, 177.5], [187.5, 177.5]]
                                          keypoint scores [0.752, 0.742, 0.733, 0.852]
object 2 hollow_semicircle  score 0.0854  keypoints [[110.5, 98.5], [nan, nan], [nan, nan], [186.5, 177.5]]
                                          keypoint scores [0.129, 0.0, 0.0, 0.213]
object 3 hexagon            score 0.0256  one keypoint (bottom, 0.153) at [420.5, 414.5]
object 4 house              score 0.0208  one keypoint (base_left, 0.104) at [30.5, 177.5]
object 5 hollow_rhombus     score 0.0139  one keypoint (right, 0.111) at [501.5, 414.5]
```
The two real objects are found within 2 pixels, with scores of 0.85 and 0.77. The four others are weak answers of the channels of other classes on the corners of the two shapes: the hollow semicircle shares the outline of the semicircle, and a corner of the rhombus looks like a corner of a hexagon. An object score is the sum of its keypoint scores divided by the keypoints of its class (Step 3.10), so the hexagon found on one corner scores 0.153 / 6 = 0.0256. These objects stay in the evaluation at the end of the precision-recall curves; the sample images and deployment drop them with the threshold of 0.2.

### Step 4.4 — `accumulate_evaluation_batch()`
Keeps, for every kept image of the batch (`sample_indices`), its ground truth objects and the objects found in it, all in network input pixels.

**Example: the accumulated entry of image `test_000002` with the checkpoint of iteration 2000**
```text
labels (2,) int64   bounding_boxes (2, 4) float32   keypoints (2, 18, 2) float32   visibilities (2, 18) int64
predicted_labels (6,) int64   predicted_keypoints (6, 18, 2) float32   predicted_scores (6,) float32
```
On the whole test split, the checkpoint of iteration 2000 finds 422 objects for the 90 ground truth objects, 91 of them with a score of at least 0.2.

### Step 4.5 — `save_sample_images()` → `build_sample_image()` → `build_keypoint_sample_grid()`
For the 16 fixed test images, `build_sample_image()` draws, with `build_keypoint_sample_grid()`, the image with its labelled ground truth keypoints (Ground Truth) next to the same image with the objects found whose score reaches `object_score_threshold` (Prediction), every object with the skeleton of its class and captioned with its class (and its score in the Prediction panel). The legend bar names the keypoint sides (Left, Right, Centre), then the classes drawn.

**Example: `Evaluation/Iteration_60/Samples/test_000002.png`** (1028x560 pixels: two panels of 512x512, a caption bar and a legend bar)
```text
caption : 2 ground truth objects | 0 predicted objects with score >= 0.20
panels  : Ground Truth | Prediction        legend : Left / Right / Centre, rhombus, semicircle
```
The Prediction panel shows the two shapes without any keypoint. With the checkpoint of iteration 2000, the caption of the same image would be `2 ground truth objects | 2 predicted objects with score >= 0.20`.

### Step 4.6 — `compute_evaluation_results()` → `evaluate_image_keypoints()`
For every image, `evaluate_image_keypoints()` builds what the average precision of the detection needs, with the OKS in place of the box IoU:
- `compute_squared_distances()` gives the squared distance of every keypoint of every object found to the same keypoint of every ground truth object, a keypoint not found (NaN) being infinitely far;
- `compute_object_keypoint_similarity()` turns them into the [D, G] OKS of every object found with every ground truth object, measured with the area of the ground truth box (`compute_box_areas()`) and the sigmas of the class (`sigma_table`), then set to 0 across classes, since the keypoints of two classes do not correspond;
- the detections are the objects found, with their class, score and the box of their keypoints found (`compute_keypoint_boxes()`), which sizes the objects that match no ground truth in the table of the object sizes;
- `match_ground_truth_objects()` matches every ground truth object to at most one object found, whatever the scores: the one-to-one matching (`linear_sum_assignment()`) of the largest total OKS over the pairs reaching an OKS of 0.5 (`MINIMUM_MATCH_SIMILARITY`, the overlap of the OKS mAP@0.5), a pair below it (another class, or keypoints too far) never matching. It only serves the keypoint accuracy of Step 4.7; the average precision does its own matching in score order.

`compute_object_keypoint_similarity()` is the COCO formula: for every labelled keypoint (visibility > 0), exp(−d² / (2 × (2σ)² × area)), averaged over the labelled keypoints of the object, in float64 (the area clamped to `MINIMUM_OBJECT_AREA`). A keypoint at distance d from its ground truth scores less on a small object than on a large one, and less with a small sigma.

**Example: image `test_000002` with the checkpoint of iteration 2000**
```text
detections   rhombus, semicircle, hollow_semicircle, hexagon, house, hollow_rhombus
scores       [0.8514, 0.7696, 0.0854, 0.0256, 0.0208, 0.0139]
boxes        [[419.5, 394.5, 501.5, 436.5], [30.5, 98.5, 187.5, 177.5], [110.5, 98.5, 186.5, 177.5],
              [420.5, 414.5, 420.5, 414.5], [30.5, 177.5, 30.5, 177.5], [501.5, 414.5, 501.5, 414.5]]
keypoint_overlaps          semicircle   rhombus
  rhombus                      0.0000    0.9965
  semicircle                   0.9945    0.0000
  hollow_semicircle            0.0000    0.0000
  hexagon, house, hollow_rhombus   0          0
ground truth boxes  [[30.0, 98.0, 187.0, 177.0], [419.0, 394.0, 502.0, 437.0]]
match_ground_truth_objects [1, 0]   matched [True, True]
```
The hollow semicircle lies on the semicircle but has an OKS of 0 with it, being of another class: it can only be a false positive. The objects found on a single keypoint have a box of zero area.

**Example: a toy object of area 10000 with two keypoints of sigma 0.05** (2 × (2σ)² × area = 200)

| Distance of the first keypoint | OKS, second keypoint exact | OKS, second keypoint unlabelled | OKS, second keypoint not found |
|---|---|---|---|
| 0 | 1.0 | 1.0 | 0.5 |
| 5 | 0.9412 | 0.8825 | 0.4412 |
| 10 | 0.8033 | 0.6065 | 0.3033 |
| 20 | 0.5677 | 0.1353 | 0.0677 |

At 20 pixels the first keypoint scores exp(−400 / 200) = 0.1353; averaged with an exact second keypoint, the object scores 0.5677. An unlabelled keypoint is left out of the mean, but a labelled keypoint that the grouping did not find counts as 0: an object with half of its keypoints missing has an OKS of 0.5 at most, reached only when every other keypoint is exact, so it is all but never found at the threshold of OKS 0.5 of mAP@0.5.

### Step 4.7 — `measure_keypoint_accuracy()` + `summarise_keypoint_accuracy()`
`measure_keypoint_accuracy()` compares every ground truth object with the object matched to it in Step 4.6 (NaN keypoints for a missed object): the distance of every keypoint (in network input pixels, infinite when not found), whether it is labelled, found and correct (within `CORRECT_KEYPOINT_FRACTION` = 0.05 of the longest side of the box) and the OKS of every object. `summarise_keypoint_accuracy()` gives four global metrics, then the same values per class and per keypoint of every class:
- `Matched Objects`: the fraction of the ground truth objects matched to an object found with an OKS of at least 0.5;
- `PCK@0.05`: the fraction of the labelled keypoints that are correct, a missed object or keypoint counting as wrong;
- `Mean OKS`: the mean OKS of the objects with labelled keypoints, a missed object counting as 0;
- `Mean Keypoint Error (pixels)`: the mean distance of the labelled keypoints found only.

**Example: image `test_000002` with the checkpoint of iteration 2000**
```text
distances (2, 18) float64   labelled (2, 18) bool   found (2, 18) bool   correct (2, 18) bool   similarities (2,) float64
object 0 semicircle  longest side 157.0, correct within 7.85   distances [2.06, 0.5, 0.71, 0.71]   correct [True, True, True, True]
                     similarity 0.9945
object 1 rhombus     longest side 83.0, correct within 4.15    distances [0.0, 1.0, 0.0, 0.0]      correct [True, True, True, True]
                     similarity 0.9965
```
With the checkpoint of iteration 60, the three objects of the image `test_000022` (a trapezoid, a plus sign and a capsule) are missed: every distance is infinite, no keypoint is correct and every similarity is 0.

**Example: the whole test split at iteration 60** (the `metrics` returned, between 0 and 1 like the scalars and `class_metrics.csv`; the report writes the first three as percentages, Step 4.10)
```text
Matched Objects 0.0111   PCK@0.05 0.0017   Mean OKS 0.0014   Mean Keypoint Error (pixels) 2.2361   labelled keypoints 589
```
Only 1 of the 90 objects is matched, a hollow trapezoid, with 1 of its 8 keypoints found (bottom_left, √5 = 2.24 pixels from its ground truth) and an OKS of 0.1223. The mean error only counts the keypoints found, so its 2.24 pixels describe this single keypoint; `Matched Objects` and `PCK@0.05` show that nearly everything is missed. The same split with the checkpoint of iteration 2000 gives Matched Objects 100.00 %, PCK@0.05 94.57 %, Mean OKS 93.37 % and a mean error of 6.1648 pixels.

> These outputs were recorded before 2026-10-25, when a match still needed only an OKS above 0. Since then a match needs an OKS of at least 0.5 (`MINIMUM_MATCH_SIMILARITY`, `bug_standards.md` B51), so the hollow trapezoid of OKS 0.1223 is missed: at iteration 60 `Matched Objects`, `PCK@0.05` and `Mean OKS` are 0 and the mean keypoint error has no value, like the other CPU checkpoints.

### Step 4.8 — `summarise_average_precisions(overlap_type="keypoints")`
The average precision of `object_detection_workflow.md` on the OKS overlaps instead of the box IoU, with `title_prefix="OKS"`, `name_prefix="OKS "` and `column_prefix="oks_"`: the three methods (area, 11 point, 101 point) at OKS 0.5, 0.75, 0.95 and 0.5 to 0.95, the COCO object sizes by box area, the mean average recall, the per-class APs, and the OKS recall of every class at 0.5 and 0.75 with the bounds of its exact (Clopper-Pearson) 95 % confidence interval over the ground truth objects (`oks_recall_50` to `oks_recall_75_upper`, the `OKS Per Class Recall` table, `object_detection_workflow.md`, Step 4.11). As for the boxes, the objects found are taken by decreasing score, and each one is a true positive when its OKS with its best ground truth object reaches the threshold and no more confident object took that ground truth (the PASCAL VOC rule): the AP is high when the objects found are placed well and the false positives have low scores, like the objects 2 to 5 of Step 4.3.

**Example: the scalars of the 30 test images with the checkpoint of iteration 2000** (between 0 and 1, the report writing them as percentages)
```text
OKS mAP@0.5/area 0.9417   OKS mAP@0.75/area 0.9099   OKS mAP@0.95/area 0.7966   OKS mAP@[0.5:0.95]/area 0.8960
```

### Step 4.9 — `write_class_metrics_file()`
The class rows of the average precisions and of the keypoint accuracy are merged into one row per class (a `ValueError` is raised if the two do not list the same classes in the same order) and saved to `class_metrics.csv`; the keypoint rows are saved to `keypoint_metrics.csv` (`KEYPOINT_METRICS_FILE_NAME`), whatever the per-class report.

**Example: `Evaluation/Iteration_60/class_metrics.csv`** (34 lines: the header and the 33 classes)
```text
class_index,class,ground_truths,oks_average_precision_50,oks_average_precision_75,oks_average_precision_95,oks_average_precision_50_95,oks_average_precision_50_small,oks_average_precision_50_95_small,oks_average_precision_50_medium,oks_average_precision_50_95_medium,oks_average_precision_50_large,oks_average_precision_50_95_large,objects,matched_objects,labelled_keypoints,percentage_correct_keypoints,mean_object_keypoint_similarity,mean_error_pixels
1,square,2,0.000000,0.000000,0.000000,0.000000,,,0.000000,0.000000,0.000000,0.000000,2,0,8,0.000000,0.000000,
26,hollow_trapezoid,3,0.000000,0.000000,0.000000,0.000000,,,0.000000,0.000000,0.000000,0.000000,3,1,24,0.041667,0.040779,2.236068
```
No test object is small, so the small columns are empty; the square has no keypoint found, so its mean error is empty.

**Example: `Evaluation/Iteration_60/keypoint_metrics.csv`** (244 lines: the header and the 243 keypoints)
```text
class_index,class,keypoint,labelled,found,percentage_correct_keypoints,mean_error_pixels
1,square,top_left,2,0,0.000000,
26,hollow_trapezoid,bottom_left,3,1,0.333333,2.236068
```

### Step 4.10 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
The headline metric is `KEYPOINT_HEADLINE_METRIC` = `OKS mAP@0.5/area`, which compares the checkpoints. The keypoint accuracy is given first, then the tables of the methods, sizes and recalls, then, with the per-class report, the per-class APs, the per-class OKS recalls with their intervals (`OKS Per Class Recall`), the accuracy per class and the accuracy per keypoint. Like the classification report, it writes its rates as percentages with two decimals ("94.03 %", `report_formatting.py`): the `percent_metrics` of the results name `Matched Objects`, `PCK@0.05` and `Mean OKS` of the grid and the headline `OKS mAP@0.5/area`, written so in the best checkpoint line of Stage 5, and the `percent_columns` of the tables hold the mAPs, mARs, APs, OKS recalls with their `95 %` bounds, PCK and mean OKS. The counts of objects and keypoints and the mean keypoint error (4 decimals, in pixels) keep their values, and `class_metrics.csv`, `keypoint_metrics.csv`, `evaluation_summary.csv`, TensorBoard and MLflow keep every rate between 0 and 1.

**Example output: the report of the checkpoint of iteration 60** (the per-class and per-keypoint tables shortened)
```text
Iteration 60 · 30 Test Images · 2026-10-18 00:24 ───────────────────────────────────────────────────────────────────────
  Matched Objects   1.11 %   PCK@0.05                       0.17 %
  Mean OKS          0.14 %   Mean Keypoint Error (pixels)   2.2361
  Sample Images : 16 fixed test images

  OKS Mean Average Precision
   Method      mAP@0.5   mAP@0.75   mAP@0.95   mAP@[0.5:0.95]
  ────────────────────────────────────────────────────────────
   Area         0.00 %     0.00 %     0.00 %           0.00 %
   11 Point     0.00 %     0.00 %     0.00 %           0.00 %
   101 Point    0.00 %     0.00 %     0.00 %           0.00 %
  The mAP@0.5 of the area method compares the checkpoints of the run.

  OKS Mean Average Precision By Object Size (Area)
   Object Size           Objects   mAP@0.5   mAP@0.75   mAP@0.95   mAP@[0.5:0.95]
  ────────────────────────────────────────────────────────────────────────────────
   Small (< 32²)               0         —          —          —                —
   Medium (32² to 96²)        56    0.00 %     0.00 %     0.00 %           0.00 %
   Large (>= 96²)             34    0.00 %     0.00 %     0.00 %           0.00 %
  COCO sizes by box area in network input pixels, the objects of the other sizes and the detections finding them being
  ignored. — : no object of the size.

  OKS Mean Average Recall
   Objects Of            Objects   mAR@0.5   mAR@0.75   mAR@0.95   mAR@[0.5:0.95]
  ────────────────────────────────────────────────────────────────────────────────
   All                        90    0.00 %     0.00 %     0.00 %           0.00 %
   Small (< 32²)               0         —          —          —                —
   Medium (32² to 96²)        56    0.00 %     0.00 %     0.00 %           0.00 %
   Large (>= 96²)             34    0.00 %     0.00 %     0.00 %           0.00 %
  Objects found by any detection kept for the evaluation, averaged over the classes. COCO sizes by box area in network
  input pixels. — : no object of the size.

  OKS Per Class Average Precision (Area)
                          Ground
   Class                  Truths   AP@0.5   AP@0.75   AP@0.95   AP@[0.5:0.95]
  ────────────────────────────────────────────────────────────────────────────
   square                      2   0.00 %    0.00 %    0.00 %          0.00 %
   triangle                    2   0.00 %    0.00 %    0.00 %          0.00 %
   ...                                                                          (the 31 other classes at 0)
  Worst AP@0.5 first, ties in class order. — : no ground truth box of the class.

  Keypoint Accuracy Per Class
                                    Matched    Labelled                Mean   Mean Error
   Class                  Objects   Objects   Keypoints   PCK@0.05      OKS     (pixels)
  ───────────────────────────────────────────────────────────────────────────────────────
   square                       2         0           8     0.00 %   0.00 %            —
   triangle                     2         0           4     0.00 %   0.00 %            —
   ...                                                                                      (29 more classes missed)
   house                        4         0          19     0.00 %   0.00 %            —
   hollow_trapezoid             3         1          24     4.17 %   4.08 %       2.2361
  Worst PCK@0.05 first. PCK@0.05 : labelled keypoints within 0.05 of the longest box side, a missed object or keypoint
  counting as wrong; errors of the keypoints found, in network input pixels. — : no labelled keypoint.

  Keypoint Accuracy Per Keypoint
                                                                             Mean Error
   Class                            Keypoint   Labelled   Found   PCK@0.05     (pixels)
  ──────────────────────────────────────────────────────────────────────────────────────
   square                           top_left          2       0     0.00 %            —
   ...                                                                                      (240 more keypoints)
   hollow_trapezoid              bottom_left          3       1    33.33 %       2.2361
   hollow_semicircle              base_right          0       0          —            —
  Worst PCK@0.05 first, a missed keypoint counting as wrong; errors of the keypoints found, in network input pixels. — :
  keypoint never labelled or never found.
  - Results : Results.txt
  - Class Metrics : class_metrics.csv
  - Keypoint Metrics : keypoint_metrics.csv
  - Sample Images : Samples
```
The base_right keypoint of the hollow semicircle comes last with no value: none of the test hollow semicircles has it labelled.

**Example: the four evaluations of the run, and the checkpoint of iteration 2000 of the GPU run on the same test images**

| Checkpoint | Matched Objects | PCK@0.05 | Mean OKS | Mean Error (pixels) | OKS mAP@0.5 (area) | OKS mAP@0.75 | OKS mAP@0.95 | OKS mAP@[0.5:0.95] |
|---|---|---|---|---|---|---|---|---|
| Iteration 30 | 0.00 % | 0.00 % | 0.00 % | — | 0.00 % | 0.00 % | 0.00 % | 0.00 % |
| Iteration 30, EMA Weights | 0.00 % | 0.00 % | 0.00 % | — | 0.00 % | 0.00 % | 0.00 % | 0.00 % |
| Iteration 60 | 1.11 % | 0.17 % | 0.14 % | 2.2361 | 0.00 % | 0.00 % | 0.00 % | 0.00 % |
| Iteration 60, EMA Weights | 0.00 % | 0.00 % | 0.00 % | — | 0.00 % | 0.00 % | 0.00 % | 0.00 % |
| Iteration 2000 of the GPU run | 100.00 % | 94.57 % | 93.37 % | 6.1648 | 94.17 % | 90.99 % | 79.66 % | 89.60 % |

The averaged weights follow the trained weights a few iterations behind (the n-th update uses the decay min(0.999, (1 + n) / (10 + n)), 0.87 at the 60th update), so at iteration 60 they match no object yet.

> Recorded before the minimum OKS of a match (2026-10-25, B51): the row of iteration 60 is now 0.00 %, 0.00 %, 0.00 % and —, and the row of the GPU run can count fewer matched objects if some object was matched below an OKS of 0.5; the OKS mAPs are unchanged.

**Example: the run folder after the training**
```text
<run>/
├── Evaluation/Iteration_30/, Iteration_60/
│   ├── class_metrics.csv, keypoint_metrics.csv
│   ├── Samples/                       16 PNG images
│   └── EMA/                           the same files for the averaged weights
├── Tensorboard/Train/, Validation/, Test/, Test_EMA/
├── Weights/Iteration_15/ ... Iteration_60/, full-checkpoint
├── Results.txt                        1444 lines
└── data.json, label_dictionary.json, mlflow_run.json, training_configuration.yaml
```

---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of a copy of `scripts/evaluation/evaluate_checkpoint.py` whose `run_name` and `experiment_folder` point to the run of Stage 3 (checkpoints 15, 30, 45 and 60, captured when those of iterations 15, 30 and 45 were still saved before their training step), with `iteration = "all"`, `batch_size = 16` and the whole test split, on CPU with the MLflow tracking on. The four checkpoints, each with its averaged weights, were evaluated in 2 min 27 s, about 17 s per evaluation. They were captured before the per-class recall intervals were added (2026-11-01, Stage 4).

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The same script for every task (see `classification_workflow.md`, Stage 5). The keypoint configurations have no `evaluation.error_dumps` section, so `maximum_error_images_per_class_pair` is skipped with a yellow warning: the keypoint evaluation saves no error image. `save_predictions`, `target_precisions`, `nms_criterion` and `nms_distance_exponent`, None in the main block, would be skipped the same way when set, the task having no predictions file (`evaluation.save_predictions`), no class thresholds (`evaluation.target_precision`) and no `Detector` section.

### Step 5.2 — trainer construction
The trainer is built from the `training_configuration.yaml` of the run, with the classes of the `label_dictionary.json` saved next to it rather than those of the mapping file (`classification_workflow.md`, Step 5.2).

Building the trainer prints the panels of Steps 2.10 and 2.11 again. With a batch of 16 images, the test split is 2 batches and an epoch 19 iterations.

**Example: the lines of the launch information that differ from Step 2.10**
```text
- Iterations Per Epoch    : 19
- Batch Size              : 16
```

### Step 5.3 — `get_checkpoint_iterations()`
**Example output**
```text
evaluated_iterations = [60, 45, 30, 15]
```

### Step 5.4 — `evaluate_checkpoints()` → `start_evaluation_tracking()`
The evaluation has an MLflow run of its own, named `<run name>-evaluation` and tagged with the training run (`evaluation_of`), in the backend of the evaluated experiment.

**Example output**
```text
MLflow Run Created In The Experiment 'keypoint_detection' : 8974fb1c8f1541aaa07bb503baacf79a (Browse : mlflow ui --backend-store-uri sqlite:///.../experiments/mlflow.db)
```

### Step 5.5 — `summarise_checkpoint_evaluations()`
The checkpoints are ranked on the headline metric of the task, the OKS mAP@0.5 (area method), for the trained and the averaged weights.

**Example output** (the four checkpoints tie at 0, so the four iterations are listed in their evaluation order; the file name is a terminal link to `experiments/<run>/Checkpoint_Evaluation/evaluation_summary.csv`)
```text
-------------------------------------------------------------
Evaluation Summary Of 4 Checkpoints : evaluation_summary.csv
-------------------------------------------------------------
  - Best Checkpoint : iterations 60, 45, 30, 15 tie (OKS mAP@0.5/area 0.00 %)
  - Best Checkpoint (EMA Weights) : iterations 60, 45, 30, 15 tie (OKS mAP@0.5/area 0.00 %)
```

**Example: `evaluation_summary.csv`** (130 columns: iteration, the 12 OKS mAPs of the 3 methods at 0.5, 0.75, 0.95 and [0.5:0.95], the 4 OKS mAPs at 0.5 and [0.5:0.95] of the medium and large objects, the 12 OKS mARs of all, medium and large objects, the 4 keypoint accuracy metrics, the 33 OKS AP@0.5 of the classes, then the same 64 columns prefixed by `ema_`; the test split has no small object, so no small column)
```text
iteration   OKS mAP@0.5/area   Keypoint Accuracy/Matched Objects   Keypoint Accuracy/PCK@0.05   Keypoint Accuracy/Mean OKS   Keypoint Accuracy/Mean Keypoint Error (pixels)   ema_OKS mAP@0.5/area
   60       0.0                0.011111                            0.001698                     0.001359                     2.236068                                         0.0
   45       0.0                0.000000                            0.000000                     0.000000                     NaN                                              0.0
   30       0.0                0.000000                            0.000000                     0.000000                     NaN                                              0.0
   15       0.0                0.000000                            0.000000                     0.000000                     NaN                                              0.0
```
The checkpoint of iteration 60 matched one object when this file was written, but its OKS of 0.12 is below the 0.5 of the mAP@0.5, so the four checkpoints tie; since the minimum OKS of a match (2026-10-25, B51) it matches none, and its row is 0.0, 0, 0, 0 and NaN like the others. The mean keypoint error is NaN when no labelled keypoint was found.

**Example: the evaluation folder**
```text
experiments/<run>/Checkpoint_Evaluation/
├── evaluation_summary.csv
├── Iteration_15/, Iteration_30/, Iteration_45/, Iteration_60/      the files of Step 4.9, with Samples/ (16 PNG images) and EMA/
├── Tensorboard/Test/, Test_EMA/
├── Results.txt                                                     2888 lines
└── data.json, label_dictionary.json, training_configuration.yaml
```

### Step 5.6 — `run_test_evaluation()` for each checkpoint
Every checkpoint is loaded with its moving average and evaluated as in Stage 4, the trained weights first, then the averaged weights.

**Example output**
```text
-----------------------------
Evaluating : model_000060.pt
-----------------------------
Test Evaluation Iteration 60: 100%|██████████| 2/2 [00:17<00:00,  8.61s/it]
Iteration 60 · 30 Test Images · 2026-10-18 00:25 ───────────────────────────────────────────────────────────────────────
  ...
  - Results : Results.txt
  - Class Metrics : class_metrics.csv
  - Keypoint Metrics : keypoint_metrics.csv
  - Sample Images : Samples
Test Evaluation Iteration 60 (EMA Weights): 100%|██████████| 2/2 [00:17<00:00,  8.60s/it]
Iteration 60 · 30 Test Images · EMA Weights · 2026-10-18 00:25 ─────────────────────────────────────────────────────────
```
The reports are those of Step 4.10: at iteration 60 the trained weights match one hollow trapezoid out of the 90 objects with keypoints, and the seven other evaluations match none. The batch size changes nothing, the metrics being computed on the whole split. On CPU no batch is pinned (`uses_pinned_memory()` of `vision_training/data/loaders.py`), so PyTorch no longer warns that `pin_memory` is set without an accelerator.

---

## Stage 6 — Export (Optional)

> The examples of Steps 6.2 to 6.5 are real outputs of a script repeating the steps of `export_experiment()` on CPU, on the checkpoint of iteration 60 of the run of Stage 3 (trained weights), with a maximum batch size of 16 and the TensorRT precision FP32, the exported models being run by TorchScript and ONNX Runtime on CPU. Triton itself was not started. They were captured before every export wrote the input contract of its run; the input contract and the model repository of Step 6.5 come from the same steps run again on 2026-11-12 with the current code, on the iteration 6000 checkpoint of the GPU run of Stage 4. After 60 iterations the client of Step 6.6 groups no object, so its example uses the ONNX export of the trained weights of iteration 6000 of the GPU run of Stage 4 (its sample image `Evaluation/Iteration_6000/Samples/test_000000.png` shows the same three objects, scored 0.90, 0.87 and 0.80).

### Step 6.1 — `vision_training/deployment/export_model.py` → `export_experiment()`
Run as a script after the training (the experiment folder, iteration, trained or averaged weights, formats, Triton model repository, model version, maximum batch size and TensorRT precision set in its `__main__` block). `export_experiment()` turns TF32 off, loads the checkpoint on CUDA with `load_trained_model()` (the device is hard-coded, which is why this stage was run by a script repeating its steps on CPU), wraps it with `build_inference_model()`, builds the export specification and the input contract of the run (`build_input_contract()`, Step 6.5), exports every format into a staging folder next to the repository, writes its `config.pbtxt` and its `preprocessing.json` and compares the exported model to the eager one, raising a `RuntimeError` on a mismatch that leaves the repository unchanged, then moves the checked models to `<repository>/<model name>_<format>/<version>/` (`install_staged_models()`).

### Step 6.2 — `load_trained_model()` → `build_inference_model()` → `InferenceKeypointDetector`
`InferenceKeypointDetector` takes the preprocessed images and returns the candidates of every heatmap channel extracted inside the network (Step 3.10): their positions in network input pixels, their scores and their tags. The grouping of the candidates into objects is left to the client (Step 6.6), a NumPy and SciPy function run outside the exported network. The number of candidates per channel is `maximum_candidates`, capped by the number of heatmap cells.

**Example output**
```text
checkpoint model_000060.pt model KeypointDetector training mode False
inference model InferenceKeypointDetector input_image_size 512 number_channels 243 number_candidates 30 maximum_candidates 30 higher_resolution True
```

### Step 6.3 — `get_export_specification()` → `create_random_inputs()`
The keypoint branch of `get_export_specification()` gives the model name `keypoint_detector`, one input `images` of 3x512x512, the three candidate outputs and the comparison function `compare_keypoints()` with its tolerances. `create_random_inputs()` draws a batch of 2 images to trace the model (`example_shape`) and a batch of 3 to check it (`check_shape`), so that a batch size frozen by the tracing would be caught.

**Example output**
```text
model_name keypoint_detector
inputs  [{'name': 'images', 'dimensions': [3, 512, 512], 'data_type': 'TYPE_FP32', 'dynamic_frames': False, 'maximum_frames': 1,
          'example_shape': [2, 3, 512, 512], 'check_shape': [3, 3, 512, 512]}]
outputs [{'name': 'candidate_positions', 'dimensions': [243, 30, 2], 'data_type': 'TYPE_FP32', 'reshape_to_scalar': False},
         {'name': 'candidate_scores', 'dimensions': [243, 30], 'data_type': 'TYPE_FP32', 'reshape_to_scalar': False},
         {'name': 'candidate_tags', 'dimensions': [243, 30], 'data_type': 'TYPE_FP32', 'reshape_to_scalar': False}]
compare_outputs compare_keypoints tolerances {'position_tolerance': 0.01, 'score_tolerance': 0.0001, 'tag_tolerance': 0.001, 'minimum_equal_share': 0.95}
example {'images': (2, 3, 512, 512)} check {'images': (3, 3, 512, 512)}
reference outputs [((3, 243, 30, 2), 'float32'), ((3, 243, 30), 'float32'), ((3, 243, 30), 'float32')]
```

### Step 6.4 — `export_torchscript()` / `export_onnx()` + `compare_keypoints()`
`export_torchscript()` traces the model with `torch.jit.trace`; `export_onnx()` exports it with the dynamo exporter (opset 18) and a dynamic `batch` dimension. Both are run on the check inputs and compared to the eager outputs by `compare_keypoints()`. The candidates are paired by position, not by rank: every candidate of positive score is paired with the nearest candidate of positive score of the same image and channel in the other output, within `position_tolerance` pixels. The scores of the pairs must agree within `score_tolerance`, and at least `minimum_equal_share` of the candidates of both outputs must have a pair with the same tag. A cell tied with a neighbour up to the rounding of the export can become a local maximum in one runtime only: it inserts a candidate and shifts the ranks of the following ones. The zero-score fillers are never paired.

**Example output**
```text
torchscript: model.pt    13,902,249 bytes   mismatches []   scored candidates 21870 + 21870   without pair at the same position 0 + 0
             largest paired score difference 0         largest paired tag difference 0          equal share 1.0000
onnx:        model.onnx   3,168,184 bytes   mismatches []   scored candidates 21870 + 21870   without pair at the same position 7 + 7
             largest paired score difference 8.01e-08  largest paired tag difference 5.96e-07   equal share 0.9997
```
The 3 check images give 3 × 243 × 30 = 21870 candidates, all of positive score: the random images give many local maxima to the scarcely trained model. In ONNX Runtime 54 candidates are at another position than the eager candidate of the same rank, and 7 have no eager candidate at their position (and 7 eager candidates no exported one); a comparison rank by rank would report a position difference of up to 462 pixels between two candidates that are both correct. The deprecation notices of `torch.jit.trace` and `torch.jit.load` are hidden by `hide_known_export_warnings()` unless `show_warnings` is True.

### Step 6.5 — `write_triton_configuration()` + `write_onnx_input_contract()` / `write_preprocessing_file()`
So that a server builds the network input from a photo as the data loaders did instead of guessing it, every export writes the input contract of the run (`input_contract.py`): `build_input_contract()` reads it from `Data.image_settings` and the preprocessing code before any file is written, `write_onnx_input_contract()` puts it in the `metadata_props` of `model.onnx`, and `write_preprocessing_file()` writes it as `preprocessing.json` next to the model of both formats, the TorchScript file carrying no metadata. It holds no channel names (Step 6.6).

**Example: the model repository** (the steps run again on 2026-11-12 on the iteration 6000 checkpoint of the GPU run; the graph files of the current code and libraries are larger than those of Step 6.4)
```text
triton_repository/
├── keypoint_detector_onnx/config.pbtxt
├── keypoint_detector_onnx/1/model.onnx               3,430,029 bytes, with the input contract in its metadata
├── keypoint_detector_onnx/1/model.onnx.data         12,416,548 bytes, the weights written next to the graph by the ONNX exporter
├── keypoint_detector_onnx/1/preprocessing.json             346 bytes
├── keypoint_detector_torchscript/config.pbtxt
├── keypoint_detector_torchscript/1/model.pt         13,917,098 bytes
└── keypoint_detector_torchscript/1/preprocessing.json      346 bytes
```

**Example: the keys and values of `keypoint_detector_onnx/1/preprocessing.json`** (written one per line), also given by ONNX Runtime as the custom metadata of `model.onnx`
```text
{"input_contract_version": "2", "input_color_order": "BGR", "input_resize": "keep_ratio", "input_size": "512",
 "input_interpolation": "linear", "input_padding_value": "0", "input_padding_placement": "centred",
 "input_tile_scale": "none", "input_tile_overlap": "none", "input_layout": "NCHW", "input_value_range": "0-255"}
```
The photo is decoded in BGR, resized with its aspect ratio kept (the longest side to 512 pixels, bilinear), centred on a square of black padding, and given as float32 values from 0 to 255 in the batch, channels, height, width layout, the input normalization being in the graph. The tile keys only apply to the segmenters trained on tiles.

**Example: `keypoint_detector_onnx/config.pbtxt`**
```text
name: "keypoint_detector_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 16
input [
  { name: "images" data_type: TYPE_FP32 dims: [ 3, 512, 512 ] }
]
output [
  { name: "candidate_positions" data_type: TYPE_FP32 dims: [ 243, 30, 2 ] },
  { name: "candidate_scores" data_type: TYPE_FP32 dims: [ 243, 30 ] },
  { name: "candidate_tags" data_type: TYPE_FP32 dims: [ 243, 30 ] }
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
The TorchScript configuration is the same with `platform: "pytorch_libtorch"`, the outputs named `candidate_positions__0`, `candidate_scores__1` and `candidate_tags__2`, and no optimization block.

### Step 6.6 — the client: `load_inference_inputs()` → the exported model → `group_keypoint_candidates()`
A client preprocesses every image like the data loaders, as the input contract of Step 6.5 describes (`load_inference_inputs()`, the resize and centred padding of `preprocess_image()`), sends it, and groups the returned candidates into objects with `group_keypoint_candidates()` (Step 3.10), using the channel table of the run (`build_keypoint_channel_table()` on the `Data.keypoint_categories` that `load_experiment_configuration()` reads from the dataset, Step 1.5) and the `KeypointHead.inference` thresholds. The positions are in network input pixels; for an image of another size the client undoes the resize and padding, for example with `restore_boxes_to_original_image()` on every point given as the box [x, y, x, y]. `vision_training/deployment/run_inference.py` only measures the throughput of the exported models on these requests, without the grouping. The example below was run on the first test image (512x512, so the network input pixels are the image pixels) with the ONNX export of the trained weights of the iteration 6000 checkpoint (the settings of `export_model.py`) in ONNX Runtime.

**Example: test_000000.png through the ONNX model, then the grouping**
```text
requests 30 {'images': ((1, 3, 512, 512), 'float32')}
outputs (1, 243, 30, 2) (1, 243, 30) (1, 243, 30)
candidates above the keypoint threshold 28 of 7290
inference settings {'maximum_candidates': 30, 'keypoint_threshold': 0.1, 'tag_threshold': 1.0, 'object_score_threshold': 0.2}
grouped {'labels': (3,), 'keypoints': (3, 18, 2), 'keypoint_scores': (3, 18), 'scores': (3,)}

object 0  hollow_capsule    score 0.9009   channels [215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226]
          keypoints     [[339.39, 185.12], [323.79, 227.58], [380.38, 227.54], [400.29, 246.07], [380.95, 263.92], [323.64, 263.4], [327.82, 244.76], [367.73, 174.5], [368.05, 236.54], [410.26, 184.35], [368.01, 254.19], [336.16, 254.7]]
          scores        [0.878, 0.886, 0.879, 0.888, 0.91, 0.924, 0.925, 0.91, 0.89, 0.911, 0.882, 0.928]
          ground truth  [[304.44, 245.5], [322.29, 227.64], [382.71, 227.64], [400.56, 245.5], [382.71, 263.36], [322.29, 263.36], [327.58, 245.5], [336.84, 236.23], [368.19, 236.23], [377.46, 245.5], [368.19, 254.77], [336.84, 254.77]]
object 1  hollow_capsule    score 0.8694   channels [215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226]
          keypoints     [[304.1, 246.21], [355.62, 167.36], [413.23, 167.28], [429.76, 184.18], [414.1, 201.62], [356.12, 201.61], [358.0, 184.27], [336.8, 236.06], [401.64, 174.24], [377.26, 245.94], [400.8, 194.15], [368.19, 194.82]]
          scores        [0.834, 0.829, 0.872, 0.868, 0.852, 0.861, 0.883, 0.881, 0.887, 0.899, 0.852, 0.915]
          ground truth  [[339.44, 184.5], [356.75, 167.19], [412.25, 167.19], [429.56, 184.5], [412.25, 201.81], [356.75, 201.81], [358.08, 184.5], [368.23, 174.35], [400.76, 174.35], [410.91, 184.5], [400.76, 194.65], [368.23, 194.65]]
object 2  trapezoid         score 0.7993   channels [47, 48, 49, 50]
          keypoints     [[245.94, 369.32], [329.25, 369.52], [345.99, 433.42], [198.34, 433.83]]
          scores        [0.819, 0.842, 0.7, 0.835]
          ground truth  [[245.5, 369.5], [329.5, 369.5], [346.5, 433.5], [198.5, 433.5]]
```
The trapezoid is found within 0.51 pixel of its labelled corners, at positions within the cells given by the log-heatmap (Step 3.10). The two hollow capsules lie side by side: every one of their 24 keypoints is found within 2.4 pixels of a labelled keypoint, but the tags exchange three keypoints between them (`left_end`, `inner_top_left` and `inner_right_end`), so that each object found holds 9 keypoints of one capsule, whose ground truth is printed, and 3 of the other. The keypoints arrays have 18 rows, the largest number of keypoints of a class (the hollow nonagon), the rows past the keypoints of the class being NaN.
